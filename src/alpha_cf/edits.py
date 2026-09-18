import random
import sys
from dataclasses import replace
from pathlib import Path
from typing import List, Optional, Tuple

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

from alpha_cf.config import DELTA_TIMES, FEATURE_NAMES, K, OP_BY_NAME, OPERATORS
from alpha_cf.types import EditAction, Site
from alphagen_qlib.stock_data import FeatureType

FEATURE_MAP = {name: FeatureType[name[1:].upper()] for name in FEATURE_NAMES}  # "$close" -> FeatureType

# Legal EditAction.kind values (apply_edit rejects anything else).
# Phase A only allows 减法/等量替换型编辑: 研究现有组件作用，不增加表达式复杂度。
# wrap（往外面包一层）是加法型编辑，与 Phase A 的"瘦身/归因"目标方向相反，故砍掉。
EDIT_KINDS = (
    "feature_replace",
    "operator_replace",
    "window_replace",
    "subtree_delete",
)


# node -> operand list (rolling window is an int on the node, not a child)
#
# Centralises the dispatch on Operator subtype so the rest of the file doesn't
# repeat the same isinstance chain. Returns [] for leaves (Feature / Constant).
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


# kinds legal at this site (matches enumerate_sites allowed flags)
def kinds_for_site(s: Site) -> List[str]:
    kinds: List[str] = []
    if s.kind == "feature":
        kinds.append("feature_replace")
    elif s.kind == "operator":
        kinds.append("operator_replace")
        if s.allowed_windows:
            kinds.append("window_replace")
    if s.can_delete:
        kinds.append("subtree_delete")
    return kinds


# expr -> preorder Site list (site_id is visit order)
#
# site_id is assigned in DFS visit order so the LLM can refer to a node by its
# preorder index. The `is_root` flag is threaded down so _site_of can forbid
# root deletion (a featureless edit).
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
#
# For each site we try every candidate replacement / window through apply_edit
# and keep only those that produce a featured f'. This way the prompt tells
# the LLM only about edits that will actually succeed — no "fake" suggestions
# that apply_edit would later reject.
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
        out.append(
            replace(s, allowed=allowed, allowed_windows=wins, can_delete=can_delete)
        )
    return out


# expr, action -> new tree, or None if illegal / missing site
#
# `subtree_delete` is a single-pass replacement; the other kinds need a
# preorder walk to find the target site and rebuild only the changed path.
# Returns None on:
#   - invalid kind
#   - site_id past the last node (caller's prompt was wrong)
#   - the resulting tree is unfeatured (e.g. wrapped in a non-leaf constant)
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


# node -> Site with allowed replacements and delete flag
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
        )
    return Site(
        site_id=sid,
        kind="constant",
        current=str(node),
        allowed=[],
        expr_str=str(node),
        can_delete=can_delete,
    )


# node, parent -> whether subtree_delete at this site keeps a featured tree
#
# Rules:
#   - root (is_root or parent is None) → can't delete (expression would vanish)
#   - under Binary/PairRolling: sibling must be featured (sibling takes parent's slot)
#   - under Unary/Rolling: operand must be featured (operand replaces the op)
#   - under Feature/Constant: only deletable if the child itself can absorb it
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
#
# Each branch is its own vocab check — feature_replace needs node to be a
# Feature, operator_replace needs same-category class, window_replace needs a
# rolling op. Returns None for any mismatch so apply_edit treats it as illegal.
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

    return None


# expr, site_id -> tree with that subtree removed, or None
#
# Two top cases:
#   - parent is Binary/PairRolling → sibling takes parent's slot
#   - parent is Unary/Rolling → operand replaces the op (unwrap)
# site_id == 0 (root) is rejected so we never return None as the new expression.
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
#
# Identity (==) comparison — alphagen Expression nodes don't override __eq__,
# so we use `is` to avoid rebuilding when the target isn't actually present.
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
#
# Mirror of alphagen's operator constructors but with the *new* children list.
# Used by apply_edit and _replace_node when only one branch changed — we
# reconstruct that node so identity hashing stays stable elsewhere.
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


if __name__ == "__main__":
    from alpha_cf.pool import parse_expr

    f = parse_expr("TsMean($close,20)")
    print("f =", f)
    for s in enumerate_sites(f):
        print(f"  site {s.site_id}: {s.kind:8s} current={s.current:8s} "
              f"window={s.window} allowed_win={s.allowed_windows} "
              f"can_delete={s.can_delete}")

    assert apply_edit(f, EditAction("subtree_delete", 0, None)) is None  # root
    assert apply_edit(f, EditAction("subtree_delete", 1, None)) is None  # leaf under rolling
    assert apply_edit(f, EditAction("operator_replace", 0, "Div")) is None

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
    print("ok")
