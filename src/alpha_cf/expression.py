"""AST edits and bounded parameter enumeration on the existing expression classes."""
from copy import deepcopy
from decimal import Decimal
from itertools import islice, product
import re

from alphagen.data import expression as E
from alphagen.data.tree import ExpressionParser


def children(expr):
    if isinstance(expr, (E.UnaryOperator, E.RollingOperator)):
        return ("_operand",)
    if isinstance(expr, (E.BinaryOperator, E.PairRollingOperator)):
        return ("_lhs", "_rhs")
    return ()


def walk(expr, path=()):
    yield path, expr
    for i, name in enumerate(children(expr)):
        yield from walk(getattr(expr, name), path + (i,))


def at(expr, path):
    for i in path:
        names = children(expr)
        if type(i) is not int or not 0 <= i < len(names):
            raise ValueError(f"Invalid subtree path: {path}")
        expr = getattr(expr, names[i])
    return expr


def replace(expr, path, replacement):
    if not path:
        return deepcopy(replacement)
    result = deepcopy(expr)
    parent = at(result, path[:-1])
    at(parent, path[-1:])
    setattr(parent, children(parent)[path[-1]], deepcopy(replacement))
    return result


def _text(text):
    # Upstream parses decimal scalar literals, but __str__ may emit exponents.
    def number(match):
        token = match.group()
        if "." not in token and "e" not in token.lower():
            return str(int(token))
        value = Decimal(token)
        if not value.is_finite() or abs(value.adjusted()) > 308:
            raise ValueError("Non-finite or excessive numeric literal")
        token = format(value.normalize(), "f")
        return token if "." in token else token + ".0"
    text = re.sub(r"\s+", "", text)
    return re.sub(r"(?<![\w$])[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?", number, text)


def validate(expr, args, featured=True):
    nodes = list(walk(expr))
    size = sum(1 + hasattr(node, "_delta_time") for _, node in nodes)
    if size > args.max_nodes or max(len(p) + 1 for p, _ in nodes) > args.max_depth:
        raise ValueError("Expression exceeds size/depth limits")
    if featured and not expr.is_featured:
        raise ValueError("Expression has no feature")

    def history(node):
        back = max((history(getattr(node, n)) for n in children(node)), default=0)
        if hasattr(node, "_delta_time"):
            window = node._delta_time
            minimum = 0 if isinstance(node, E.Ref) else 1 if isinstance(node, E.TsDelta) else 2
            if type(window) is not int or window < minimum:
                raise ValueError("Invalid window or future reference")
            back += window if isinstance(node, (E.Ref, E.TsDelta)) else window - 1
        return back
    if history(expr) > args.max_backtrack:
        raise ValueError("Expression exceeds available history")
    return expr


def parse(text, args, featured=True):
    if not isinstance(text, str) or len(text) > 6000 or text.count("(") > args.max_nodes:
        raise ValueError("Invalid expression text")
    text = _text(text)
    expr = ExpressionParser().parse(text)
    if _text(str(expr)) != text:
        raise ValueError("Malformed operator arguments")
    return validate(expr, args, featured)


def ablate(expr, mechanism, args):
    """Return (locally edited expression, baseline kind); semantics remain a hypothesis."""
    path = tuple(mechanism["path"])
    node = at(expr, path)
    if str(node) != str(parse(mechanism["subtree"], args)):
        raise ValueError("Mechanism does not match the source subtree")
    if not isinstance(mechanism.get("reason"), str) or not mechanism["reason"].strip():
        raise ValueError("Intervention needs a baseline explanation")
    baseline = parse(mechanism["replacement"], args, featured=False)
    mode, kind = mechanism["mode"], "contextual"
    if mode == "remove":
        if str(baseline) not in {str(n) for p, n in walk(node) if p and n.is_featured}:
            raise ValueError("Remove must retain a featured descendant of the mechanism")
        kind = "input"
    elif mode == "neutralize":
        source_features = {str(n) for _, n in walk(node) if isinstance(n, E.Feature)}
        baseline_features = {str(n) for _, n in walk(baseline) if isinstance(n, E.Feature)}
        if not baseline_features <= source_features:
            raise ValueError("Baseline introduces a new input feature")
        # Recognize identities without limiting all interventions to this list.
        parent = at(expr, path[:-1]) if path else None
        identity = None
        if isinstance(parent, (E.Add, E.Sub)):
            identity = "0.0"
        elif isinstance(parent, E.Mul) or isinstance(parent, E.Div) and path[-1:] == (1,):
            identity = "1.0"
        if str(baseline) == identity:
            kind = "identity"
    else:
        raise ValueError("Expected remove or neutralize")
    result = validate(replace(expr, path, baseline), args)
    if str(result) == str(expr):
        raise ValueError("Unchanged intervention")
    return result, kind


def structure(expr):
    if isinstance(expr, E.Feature):
        return str(expr)
    return type(expr).__name__, tuple(structure(getattr(expr, n)) for n in children(expr))


def parameter_specs(expr, requested, args):
    """Validate LLM-selected positions; absent suggestions fall back to two windows."""
    if not requested:
        requested = [{"path": list(p), "kind": "window"} for p, n in walk(expr)
                     if hasattr(n, "_delta_time")][:2]
    if not isinstance(requested, list) or len(requested) > 2:
        raise ValueError("Refinement accepts at most two parameter positions")
    specs, used = [], set()
    for item in requested:
        path, kind = tuple(item["path"]), item["kind"]
        node = at(expr, path)
        if (path, kind) in used:
            raise ValueError("Duplicate parameter position")
        if kind == "window" and hasattr(node, "_delta_time"):
            allowed = args.windows
        elif kind == "constant" and isinstance(node, E.Constant):
            if item.get("role") not in ("coefficient", "exponent", "threshold"):
                raise ValueError("Constant needs a coefficient/exponent/threshold role")
            if 0 < abs(node._value) < 1e-6:
                raise ValueError("Do not tune numerical stability constants")
            allowed = args.constants
        else:
            raise ValueError("Parameter kind does not match the AST node")
        values = item.get("values", allowed)
        if not isinstance(values, list) or not values or any(
                type(v) not in (int, float) or v not in allowed for v in values):
            raise ValueError("Parameter values must come from the configured grid")
        if kind == "constant" and len(path) == 1:
            if isinstance(expr, (E.Add, E.Sub)):
                raise ValueError("An outer scalar offset cannot change ranks")
            if isinstance(expr, (E.Mul, E.Div)):
                # Keep the original scale so other selected parameters can still vary.
                values = [node._value] + [v for v in values if v * node._value < 0]
        specs.append(dict(path=path, kind=kind, values=list(dict.fromkeys(values))))
        used.add((path, kind))
    return specs


def parameter_variants(expr, specs, args):
    if not specs:
        return
    for values in islice(product(*(s["values"] for s in specs)), args.max_refine_trials):
        variant, changes = deepcopy(expr), []
        for spec, value in zip(specs, values):
            path, kind = spec["path"], spec["kind"]
            node = at(variant, path)
            if kind == "window":
                node._delta_time = int(value)
            else:
                node._value = float(value)
            changes.append(dict(path=path, kind=kind, value=value))
        if str(variant) == str(expr):
            continue
        try:
            yield validate(variant, args), changes
        except ValueError:
            continue
