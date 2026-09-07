import random
import sys
from dataclasses import dataclass, replace
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

if __package__ is None:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from alphagen.data.expression import (
    BinaryOperator,
    Constant,
    Expression,
    Feature,
    Operator,
    PairRollingOperator,
    RollingOperator,
    UnaryOperator,
)

from alphagen_qlib.stock_data import FeatureType

from alpha_cf.config import CONSTANTS, DELTA_TIMES, FEATURE_NAMES, K, OP_BY_NAME, OPERATORS
from alpha_cf.types import CFRecord, EditAction, Site

FEATURE_MAP = {name: FeatureType[name[1:].upper()] for name in FEATURE_NAMES}  # "$close" -> FeatureType
CONST_SET = {float(c) for c in CONSTANTS}
WRAP_HOLE = "$_"  # placeholder for the current subtree; not valid alphagen syntax

# Legal EditAction.kind values (apply_edit rejects anything else)
EDIT_KINDS = (
    "feature_replace",
    "operator_replace",
    "window_replace",
    "subtree_delete",
    "wrap",
)


# Parsed wrap template: one operator layer around $_ 
@dataclass(frozen=True)
class WrapSpec:
    op_name: str
    op_cls: type
    side: str  # left = old tree is first expr arg; right = old tree is second
    other: Optional[Expression]
    window: Optional[int]


# node -> operand list (rolling window is an int on the node, not a child)
def children(node: Expression) -> List[Expression]:
    if isinstance(node, UnaryOperator):
        return [node._operand]
    if isinstance(node, BinaryOperator):
        return [node._lhs, node._rhs]
    if isinstance(node, RollingOperator):
        return [node._operand]
    if isinstance(node, PairRollingOperator):
        return [node._lhs, node._rhs]
    return []


# expr -> preorder Site list (site_id is visit order)
def enumerate_sites(expr: Expression) -> List[Site]:
    sites: List[Site] = []

    # DFS; site_id = append order; parent decides can_delete
    def rec(node: Expression, parent: Optional[Expression], is_root: bool) -> None:
        sid = len(sites)
        sites.append(_site_of(sid, node, parent, is_root))
        for ch in children(node):
            rec(ch, node, False)

    rec(expr, None, True)
    return _restrict_allowed(expr, sites)


# drop replacements that apply_edit rejects, so prompt ∈ sets match T
def _restrict_allowed(expr: Expression, sites: List[Site]) -> List[Site]:
    out: List[Site] = []
    for s in sites:
        kind = "feature_replace" if s.kind == "feature" else "operator_replace" if s.kind == "operator" else None
        allowed = (
            [v for v in s.allowed if apply_edit(expr, EditAction(kind, s.site_id, v)) is not None]
            if kind
            else list(s.allowed)
        )
        wins = tuple(
            w for w in s.allowed_windows
            if apply_edit(expr, EditAction("window_replace", s.site_id, str(w))) is not None
        )
        can_delete = bool(
            s.can_delete and apply_edit(expr, EditAction("subtree_delete", s.site_id, None)) is not None
        )
        can_wrap = bool(
            s.can_wrap and (
                apply_edit(expr, EditAction("wrap", s.site_id, f"Abs({WRAP_HOLE})")) is not None
                or apply_edit(
                    expr,
                    EditAction("wrap", s.site_id, f"Add({WRAP_HOLE},{FEATURE_NAMES[0]})"),
                )
                is not None
            )
        )
        out.append(
            replace(s, allowed=allowed, allowed_windows=wins, can_delete=can_delete, can_wrap=can_wrap)
        )
    return out


# expr, action -> new tree, or None if illegal / missing site
def apply_edit(expr: Expression, action: EditAction) -> Optional[Expression]:
    if action.kind not in EDIT_KINDS:
        raise ValueError(f"invalid edit kind: {action.kind}")
    if action.kind == "subtree_delete":
        return _apply_delete(expr, action.site_id)
    found = [False]
    counter = [0]

    # preorder walk; rebuild only the path that changed
    def rec(node: Expression) -> Optional[Expression]:
        sid = counter[0]
        counter[0] += 1
        if sid == action.site_id:
            found[0] = True
            return _apply_at(node, action)
        kids = children(node)
        new_kids = [rec(k) for k in kids]
        if any(k is None for k in new_kids):
            return None
        if new_kids == kids:
            return node
        return _rebuild(node, new_kids)

    out = rec(expr)
    if not found[0]:
        return None  # site_id past the last node
    if out is None or not out.is_featured:
        return None
    return out


# wrap template string -> WrapSpec, or None if illegal
def parse_wrap_template(text: str) -> Optional[WrapSpec]:
    if not text or WRAP_HOLE not in text:
        return None
    text = text.strip()
    if not text.endswith(")"):
        return None
    lp = text.find("(")
    if lp <= 0:
        return None
    name = text[:lp].strip()
    if name not in OP_BY_NAME:
        return None
    args = _split_args(text[lp + 1 : -1])
    if sum(1 for a in args if a == WRAP_HOLE) != 1:
        return None
    cls = OP_BY_NAME[name]
    cat = cls.category_type()

    if cat is UnaryOperator:
        if args != [WRAP_HOLE]:
            return None
        return WrapSpec(name, cls, "left", None, None)

    if cat is RollingOperator:
        if len(args) != 2 or args[0] != WRAP_HOLE:
            return None
        w = _parse_window(args[1])
        if w is None:
            return None
        return WrapSpec(name, cls, "left", None, w)

    if cat is BinaryOperator:
        if len(args) != 2:
            return None
        if args[0] == WRAP_HOLE:
            other = _parse_wrap_leaf(args[1])
            side = "left"
        elif args[1] == WRAP_HOLE:
            other = _parse_wrap_leaf(args[0])
            side = "right"
        else:
            return None
        if other is None:
            return None
        return WrapSpec(name, cls, side, other, None)

    if cat is PairRollingOperator:
        if len(args) != 3:
            return None
        w = _parse_window(args[2])
        if w is None:
            return None
        if args[0] == WRAP_HOLE:
            other = _parse_wrap_leaf(args[1])
            side = "left"
        elif args[1] == WRAP_HOLE:
            other = _parse_wrap_leaf(args[0])
            side = "right"
        else:
            return None
        if other is None:
            return None
        return WrapSpec(name, cls, side, other, w)

    return None


# node -> Site with allowed replacements and delete/wrap flags
def _site_of(sid: int, node: Expression, parent: Optional[Expression], is_root: bool) -> Site:
    can_delete = _can_delete_at(node, parent, is_root)
    if isinstance(node, Feature):
        cur = str(node)
        return Site(
            site_id=sid,
            kind="feature",
            current=cur,
            allowed=[x for x in FEATURE_NAMES if x != cur],
            expr_str=str(node),
            can_delete=can_delete,
            can_wrap=True,
        )
    if isinstance(node, Operator):
        cur = type(node).__name__
        cat = type(node).category_type()
        allowed = [
            op.__name__
            for op in OPERATORS
            if op.category_type() is cat and op.__name__ != cur
        ]
        window = (
            int(node._delta_time)
            if isinstance(node, (RollingOperator, PairRollingOperator))
            else None
        )
        allowed_windows = tuple(w for w in DELTA_TIMES if w != window) if window is not None else ()
        return Site(
            site_id=sid,
            kind="operator",
            current=cur,
            allowed=allowed,
            expr_str=str(node),
            window=window,
            allowed_windows=allowed_windows,
            can_delete=can_delete,
            can_wrap=True,
        )
    return Site(
        site_id=sid,
        kind="constant",
        current=str(node),
        allowed=[],
        expr_str=str(node),
        can_delete=can_delete,
        can_wrap=True,
    )


# node, parent -> whether subtree_delete at this site keeps a featured tree
def _can_delete_at(node: Expression, parent: Optional[Expression], is_root: bool) -> bool:
    if is_root or parent is None:
        return False
    if isinstance(parent, (BinaryOperator, PairRollingOperator)):
        kids = children(parent)
        if kids[0] is node:
            return bool(kids[1].is_featured)
        if kids[1] is node:
            return bool(kids[0].is_featured)
        return False
    if isinstance(node, (UnaryOperator, RollingOperator)):
        return bool(node._operand.is_featured)
    return False


# apply one kind at this node; None on type / vocab mismatch
def _apply_at(node: Expression, action: EditAction) -> Optional[Expression]:
    kind, new = action.kind, action.new_value
    if kind == "feature_replace":
        if not isinstance(node, Feature) or new not in FEATURE_MAP:
            return None
        if new == str(node):
            return None
        return Feature(FEATURE_MAP[new])

    if kind == "operator_replace":
        if not isinstance(node, Operator) or new not in OP_BY_NAME:
            return None
        new_cls = OP_BY_NAME[new]
        if new_cls is type(node) or new_cls.category_type() is not type(node).category_type():
            return None
        return _cast_op(new_cls, node)

    if kind == "window_replace":
        if not isinstance(node, (RollingOperator, PairRollingOperator)) or new is None:
            return None
        try:
            w = int(new)
        except (TypeError, ValueError):
            return None
        if w not in DELTA_TIMES or w == node._delta_time:
            return None
        return _with_window(node, w)

    if kind == "wrap":
        spec = parse_wrap_template(new or "")
        if spec is None:
            return None
        return _apply_wrap(node, spec)

    return None


# node, WrapSpec -> Op around node, or None
def _apply_wrap(node: Expression, spec: WrapSpec) -> Optional[Expression]:
    cls, cat = spec.op_cls, spec.op_cls.category_type()
    if cat is UnaryOperator:
        return cls(node)
    if cat is RollingOperator:
        if spec.window is None:
            return None
        return cls(node, spec.window)
    if cat is BinaryOperator:
        if spec.other is None:
            return None
        if spec.side == "left":
            return cls(node, spec.other)
        return cls(spec.other, node)
    if cat is PairRollingOperator:
        if spec.other is None or spec.window is None:
            return None
        if spec.side == "left":
            return cls(node, spec.other, spec.window)
        return cls(spec.other, node, spec.window)
    return None


# expr, site_id -> tree with that subtree removed, or None
def _apply_delete(expr: Expression, site_id: int) -> Optional[Expression]:
    if site_id == 0:
        return None
    loc = _locate(expr, site_id)
    if loc is None:
        return None
    node, parent, idx = loc
    if isinstance(parent, (BinaryOperator, PairRollingOperator)):
        kids = children(parent)
        if idx is None or idx >= len(kids) or kids[idx] is not node:
            return None
        sibling = kids[1 - idx]
        if not sibling.is_featured:
            return None
        out = _replace_node(expr, parent, sibling)
        if out is None or not out.is_featured:
            return None
        return out
    if isinstance(node, (UnaryOperator, RollingOperator)):
        inner = node._operand
        if not inner.is_featured:
            return None
        out = _replace_node(expr, node, inner)
        if out is None or not out.is_featured:
            return None
        return out
    return None


# expr, site_id -> (node, parent, child_index), or None
def _locate(expr: Expression, site_id: int) -> Optional[Tuple[Expression, Optional[Expression], Optional[int]]]:
    hit: List[Tuple[Expression, Optional[Expression], Optional[int]]] = []
    n = [0]

    def rec(node: Expression, parent: Optional[Expression], idx: Optional[int]) -> None:
        sid = n[0]
        n[0] += 1
        if sid == site_id:
            hit.append((node, parent, idx))
            return
        for i, ch in enumerate(children(node)):
            rec(ch, node, i)

    rec(expr, None, None)
    return hit[0] if hit else None


# expr, target, replacement -> tree with target swapped by identity
def _replace_node(expr: Expression, target: Expression, replacement: Expression) -> Expression:
    if expr is target:
        return replacement
    kids = children(expr)
    if not kids:
        return expr
    new_kids = [_replace_node(k, target, replacement) for k in kids]
    if new_kids == kids:
        return expr
    return _rebuild(expr, new_kids)


# same children / window, new operator class (same category)
def _cast_op(cls, node: Operator) -> Optional[Expression]:
    if isinstance(node, UnaryOperator):
        return cls(node._operand)
    if isinstance(node, BinaryOperator):
        return cls(node._lhs, node._rhs)
    if isinstance(node, RollingOperator):
        return cls(node._operand, node._delta_time)
    if isinstance(node, PairRollingOperator):
        return cls(node._lhs, node._rhs, node._delta_time)
    return None


# rebuild rolling / pair-rolling with a new window
def _with_window(node: Expression, w: int) -> Expression:
    cls = type(node)
    if isinstance(node, RollingOperator):
        return cls(node._operand, w)
    assert isinstance(node, PairRollingOperator)
    return cls(node._lhs, node._rhs, w)


# same node type, swapped children (window stays)
def _rebuild(node: Expression, new_kids: List[Expression]) -> Expression:
    cls = type(node)
    if isinstance(node, UnaryOperator):
        return cls(new_kids[0])
    if isinstance(node, BinaryOperator):
        return cls(new_kids[0], new_kids[1])
    if isinstance(node, RollingOperator):
        return cls(new_kids[0], node._delta_time)
    if isinstance(node, PairRollingOperator):
        return cls(new_kids[0], new_kids[1], node._delta_time)
    if isinstance(node, Feature):
        return Feature(node._feature)
    if isinstance(node, Constant):
        return Constant(node._value)
    raise TypeError(f"cannot rebuild {cls.__name__}")


# "a, b, Op(c, d)" -> ["a", "b", "Op(c, d)"]
def _split_args(args_str: str) -> List[str]:
    args: List[str] = []
    cur: List[str] = []
    depth = 0
    for ch in args_str:
        if ch == "(":
            depth += 1
            cur.append(ch)
        elif ch == ")":
            depth -= 1
            cur.append(ch)
        elif ch == "," and depth == 0:
            args.append("".join(cur).strip())
            cur = []
        else:
            cur.append(ch)
    if cur:
        args.append("".join(cur).strip())
    return args


# window token -> DELTA_TIMES int; bare ints only (1.0 is a constant)
def _parse_window(s: str) -> Optional[int]:
    if not s or "." in s:
        return None
    try:
        w = int(s)
    except ValueError:
        return None
    if w not in DELTA_TIMES:
        return None
    return w


# feature or CONSTANTS float with a decimal point -> leaf Expression
def _parse_wrap_leaf(s: str) -> Optional[Expression]:
    if s in FEATURE_MAP:
        return Feature(FEATURE_MAP[s])
    if "." not in s:
        return None
    try:
        val = float(s)
    except ValueError:
        return None
    if val not in CONST_SET:
        return None
    return Constant(val)


# float -> alphagen constant literal (must contain '.')
def _const_lit(v: float) -> str:
    s = str(float(v))
    return s if "." in s else f"{s}.0"


# catalog -> every legal one-layer wrap template with $_ 
def _wrap_templates() -> Tuple[str, ...]:
    leaves = list(FEATURE_NAMES) + [_const_lit(c) for c in CONSTANTS]
    out: List[str] = []
    for op in OPERATORS:
        cat = op.category_type()
        name = op.__name__
        if cat is UnaryOperator:
            out.append(f"{name}({WRAP_HOLE})")
        elif cat is RollingOperator:
            for w in DELTA_TIMES:
                out.append(f"{name}({WRAP_HOLE},{w})")
        elif cat is BinaryOperator:
            for leaf in leaves:
                out.append(f"{name}({WRAP_HOLE},{leaf})")
                out.append(f"{name}({leaf},{WRAP_HOLE})")
        elif cat is PairRollingOperator:
            for leaf in leaves:
                for w in DELTA_TIMES:
                    out.append(f"{name}({WRAP_HOLE},{leaf},{w})")
                    out.append(f"{name}({leaf},{WRAP_HOLE},{w})")
    return tuple(out)


WRAP_TEMPLATES = _wrap_templates()


# expr, k -> k random legal EditActions (r unused; matches ProposeFn)
def random_propose(expr: Expression, r: float, k: int = K, recent: Optional[Sequence[CFRecord]] = None) -> List[EditAction]:
    print(f"[random] propose k={k}", flush=True)
    blocked = {str(expr)}
    for rec in recent or []:
        blocked.add(rec.f_prime)
    pool: List[EditAction] = []
    for site in enumerate_sites(expr):
        cands: List[EditAction] = []
        if site.kind == "feature":
            cands.extend(EditAction("feature_replace", site.site_id, v) for v in site.allowed)
        elif site.kind == "operator":
            cands.extend(EditAction("operator_replace", site.site_id, v) for v in site.allowed)
            cands.extend(EditAction("window_replace", site.site_id, str(w)) for w in site.allowed_windows)
        if site.can_delete:
            cands.append(EditAction("subtree_delete", site.site_id, None))
        if site.can_wrap:
            cands.extend(EditAction("wrap", site.site_id, t) for t in WRAP_TEMPLATES)
        for a in cands:
            fp = apply_edit(expr, a)
            if fp is None:
                continue
            key = str(fp)
            if key in blocked:
                continue
            blocked.add(key)
            pool.append(a)
    if len(pool) < k:
        raise RuntimeError(f"need {k} legal random edits, got {len(pool)}")
    picked = random.sample(pool, k)
    for a in picked:
        fp = apply_edit(expr, a)
        print(f"[random] ok  {a.kind}@{a.site_id} {a.new_value!r}  ->  {fp}", flush=True)
    return picked


if __name__ == "__main__":
    from alpha_cf.pool import parse_expr

    f = parse_expr("TsMean($close,20)")
    print("f =", f)
    for s in enumerate_sites(f):
        print(f"  site {s.site_id}: {s.kind:8s} current={s.current:8s} "
              f"window={s.window} allowed_win={s.allowed_windows} "
              f"can_delete={s.can_delete} can_wrap={s.can_wrap}")

    assert str(apply_edit(f, EditAction("wrap", 0, "Div($_,1.0)"))) == "Div(TsMean($close,20),1.0)"
    assert str(apply_edit(f, EditAction("wrap", 0, "Abs($_)"))) == "Abs(TsMean($close,20))"
    assert str(apply_edit(f, EditAction("wrap", 1, "TsMean($_,20)"))) == "TsMean(TsMean($close,20),20)"
    assert apply_edit(f, EditAction("subtree_delete", 0, None)) is None  # root
    assert apply_edit(f, EditAction("subtree_delete", 1, None)) is None  # leaf under rolling
    assert apply_edit(f, EditAction("operator_replace", 0, "Div")) is None
    assert apply_edit(f, EditAction("wrap", 0, "Div($_,1)")) is None  # 1 is a window, not a const
    assert apply_edit(f, EditAction("wrap", 0, "Div(Add($close,$open),$_)")) is None  # other not a leaf

    f_add = parse_expr("Add($close,$open)")
    assert str(apply_edit(f_add, EditAction("subtree_delete", 1, None))) == "$open"
    assert str(apply_edit(f_add, EditAction("subtree_delete", 2, None))) == "$close"
    assert apply_edit(f_add, EditAction("subtree_delete", 0, None)) is None

    f_nested = parse_expr("Rank(TsMean($close,20))")
    assert str(apply_edit(f_nested, EditAction("subtree_delete", 1, None))) == "Rank($close)"
    assert apply_edit(f_nested, EditAction("subtree_delete", 2, None)) is None  # feature under rolling

    f_pow = parse_expr("Pow($low,0.5)")
    pow_site = enumerate_sites(f_pow)[0]
    assert "Greater" not in pow_site.allowed and "Less" not in pow_site.allowed
    assert apply_edit(f_pow, EditAction("operator_replace", 0, "Greater")) is None
    print("Pow($low,0.5) operator_replace ∈", pow_site.allowed)

    acts = random_propose(f, 0.0, k=8)
    assert all(a.kind in EDIT_KINDS for a in acts)
    assert all(a.kind != "subtree_replace" for a in acts)
    assert all(apply_edit(f, a) is not None for a in acts)

    demos = [
        EditAction("window_replace", 0, "10"),
        EditAction("operator_replace", 0, "TsMax"),
        EditAction("feature_replace", 1, "$volume"),
        EditAction("wrap", 0, "Div($_,1.0)"),
        EditAction("wrap", 0, "Abs($_)"),
        EditAction("wrap", 1, "TsMean($_,20)"),
        EditAction("subtree_delete", 0, None),
        EditAction("operator_replace", 0, "Add"),
        EditAction("window_replace", 0, "7"),
        EditAction("feature_replace", 0, "$volume"),
    ]
    print("--- T(f, a) ---")
    for a in demos:
        fp = apply_edit(f, a)
        print(f"  {a.kind:18s} site={a.site_id} new={a.new_value!r:16s} -> {fp}")
    print("ok")
