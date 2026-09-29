"""Small AST helpers; execution and parsing belong to alphagen."""
from copy import deepcopy
from decimal import Decimal
from itertools import product
import re

from alphagen.data import expression as E
from alphagen.data.tree import ExpressionBuilder, ExpressionParser


class FormulaBuilder(ExpressionBuilder):
    # Text formulas may push constants from different nesting levels consecutively.
    validate_const = ExpressionBuilder.validate_feature


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
        if type(i) is not int or i < 0 or i >= len(children(expr)):
            raise ValueError(f"Invalid subtree path: {path}")
        expr = getattr(expr, children(expr)[i])
    return expr


def replace(expr, path, replacement):
    if not path:
        return deepcopy(replacement)
    result = deepcopy(expr)
    parent = at(result, path[:-1])
    at(parent, path[-1:])
    setattr(parent, children(parent)[path[-1]], deepcopy(replacement))
    return result


def validate(expr, args):
    # A single outer sign flip does not increase the factor body complexity.
    body = expr._rhs if (isinstance(expr, E.Sub) and isinstance(expr._lhs, E.Constant)
                        and expr._lhs._value == 0.0) else expr
    nodes = list(walk(body))
    if not expr.is_featured or len(nodes) > args.max_nodes or max(len(p) for p, _ in nodes) >= args.max_depth:
        raise ValueError("Expression has no feature or exceeds size/depth limits")

    def lookback(node):
        days = max((lookback(getattr(node, name)) for name in children(node)), default=0)
        if hasattr(node, "_delta_time"):
            w = node._delta_time
            minimum = 0 if isinstance(node, E.Ref) else 1 if isinstance(node, E.TsDelta) else 2
            if w < minimum:
                raise ValueError("Illegal window or future reference")
            days += w if isinstance(node, (E.Ref, E.TsDelta)) else w - 1
        return days
    if lookback(expr) > args.max_backtrack:
        raise ValueError("Expression exceeds available history")
    return expr


def parse(text, args, featured=True):
    def canonical(source):
        source = re.sub(r"\s+", "", source)
        source = re.sub(r"\b(Greater|Less)(?=\()", r"Get\1", source)
        return re.sub(r"-?\d+(?:\.\d*)?[eE][+-]?\d+",
                      lambda m: format(Decimal(m[0]), "f"), source)
    text = canonical(text)
    builder = FormulaBuilder()
    for token in ExpressionParser().tokenize(text):
        builder.add_token(token)
    expr = builder.get_tree()
    if canonical(str(expr)) != text:
        raise ValueError("Malformed expression arguments")
    return validate(expr, args) if featured else expr


def ablate(expr, mechanism, args):
    path = mechanism["path"]
    node = at(expr, path)
    baseline = parse(mechanism["replacement"], args, featured=False)
    if mechanism["mode"] == "remove":
        if str(baseline) not in {str(n) for p, n in walk(node) if p}:
            raise ValueError("Remove must retain a descendant of the mechanism")
    elif mechanism["mode"] == "neutralize":
        inputs = {str(n) for _, n in walk(node) if isinstance(n, E.Feature)}
        if any(str(n) not in inputs for _, n in walk(baseline) if isinstance(n, E.Feature)):
            raise ValueError("Neutralization introduces a new input")
    else:
        raise ValueError("Expected remove or neutralize")
    result = validate(replace(expr, path, baseline), args)
    if str(result) == str(expr):
        raise ValueError("Unchanged intervention")
    return result


def parameter_variants(expr, args):
    # Tune up to two distinct windows jointly wherever repeated; enumerate their full grid.
    windows = sorted({n._delta_time for _, n in walk(expr) if hasattr(n, "_delta_time") and n._delta_time > 1})[:2]
    for values in product(*(sorted(set([w, *args.windows])) for w in windows)):
        variant = deepcopy(expr)
        mapping = dict(zip(windows, values))
        for _, node in walk(variant):
            if hasattr(node, "_delta_time") and node._delta_time in mapping:
                node._delta_time = mapping[node._delta_time]
        if str(variant) != str(expr):
            try:
                yield validate(variant, args)
            except ValueError:
                continue
