"""Prefix expressions preserving AlphaAgent pandas semantics for imported pools."""
import ast
import re

import numpy as np
import pandas as pd
import torch

from alphagen.data.expression import Expression, OutOfDataRangeError


ENGINE = "alphaagent_pandas_v1"
FUNCTIONS = {
    "ABS": "Abs", "SIGN": "Sign", "POW": "Pow", "SQRT": "Sqrt",
    "EMA": "AAEMA", "WMA": "AAWMA", "DECAYLINEAR": "AADecayLinear",
    "TS_MEAN": "TsMean", "TS_STD": "TsStd", "TS_MEDIAN": "TsMed",
    "TS_SUM": "TsSum", "DELAY": "Ref", "Where": "Where",
    "MAX": "GetGreater", "MIN": "GetLess", "DELTA": "AADelta",
    "TS_ZSCORE": "AATsZScore", "TS_QUANTILE": "AATsQuantile",
    "TS_PCTCHANGE": "AAPctChange", "TS_RANK": "AATsRank",
    "TS_MIN": "TsMin", "TS_MAX": "TsMax", "TS_MAD": "AATsMAD",
    "PERCENTILE": "AAFullSampleQuantile", "BB_MIDDLE": "TsMean",
    "RANK": "CSRank", "MEDIAN": "CSMedian", "STD": "CSStd", "ZSCORE": "CSZScore",
}
BINARY = {ast.Add: "Add", ast.Sub: "Sub", ast.Mult: "Mul", ast.Div: "Div"}
COMPARE = {ast.Gt: "Gt", ast.Lt: "Lt", ast.GtE: "Ge", ast.LtE: "Le",
           ast.Eq: "Eq", ast.NotEq: "Ne"}


def _ternary(text):
    """Translate grouped C-style conditionals without changing precedence."""
    parts, i = [], 0
    while i < len(text):
        if text[i] != "(":
            parts.append(text[i])
            i += 1
            continue
        end, depth = i + 1, 1
        while end < len(text) and depth:
            depth += (text[end] == "(") - (text[end] == ")")
            end += 1
        if depth:
            raise ValueError("Unbalanced conditional expression")
        parts.append("(" + _ternary(text[i + 1:end - 1]) + ")")
        i = end
    text = "".join(parts)
    # Commas inside function-call parentheses delimit separate arguments. Split
    # them before parsing a conditional so ``f(cond ? x : 0, window)`` becomes
    # ``f(Where(cond, x, 0), window)`` rather than treating ``0, window`` as the
    # conditional's false branch.
    arguments, start, depth = [], 0, 0
    for i, char in enumerate(text):
        depth += (char == "(") - (char == ")")
        if char == "," and depth == 0:
            arguments.append(text[start:i])
            start = i + 1
    if arguments:
        arguments.append(text[start:])
        return ",".join(_ternary(argument) for argument in arguments)
    depth, question, nested = 0, None, 0
    for i, char in enumerate(text):
        depth += (char == "(") - (char == ")")
        if depth:
            continue
        if char == "?":
            if question is None:
                question = i
            nested += 1
        elif char == ":" and question is not None:
            nested -= 1
            if nested == 0:
                return f"Where({text[:question]},{_ternary(text[question + 1:i])},{_ternary(text[i + 1:])})"
    if question is not None:
        raise ValueError("Conditional expression is missing ':'")
    return text


def _tree(text):
    return ast.parse(re.sub(r"\$(\w+)", r"feature_\1", text), mode="eval").body


def convert(text):
    """Convert infix formulas to prefix operators, including LOG(x) = log(1+x)."""
    def visit(node):
        if isinstance(node, ast.Constant) and type(node.value) in (int, float):
            return repr(node.value)
        if isinstance(node, ast.Name) and node.id.startswith("feature_"):
            name = node.id[8:]
            return "DailyReturn($close)" if name == "return" else "$" + name
        if isinstance(node, ast.BinOp) and type(node.op) in BINARY:
            return f"{BINARY[type(node.op)]}({visit(node.left)},{visit(node.right)})"
        if isinstance(node, ast.BoolOp) and isinstance(node.op, (ast.And, ast.Or)):
            name = "And" if isinstance(node.op, ast.And) else "Or"
            result, *rest = [visit(value) for value in node.values]
            for value in rest:
                result = f"{name}({result},{value})"
            return result
        if isinstance(node, ast.Compare) and len(node.ops) == 1 and type(node.ops[0]) in COMPARE:
            return f"{COMPARE[type(node.ops[0])]}({visit(node.left)},{visit(node.comparators[0])})"
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
            return f"Sub(0,{visit(node.operand)})"
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and not node.keywords:
            args = [visit(arg) for arg in node.args]
            if node.func.id == "LOG" and len(args) == 1:
                return f"Log(Add({args[0]},1))"
            if node.func.id == "EXP" and len(args) == 1:
                return f"Exp({args[0]})"
            if node.func.id in FUNCTIONS:
                return f"{FUNCTIONS[node.func.id]}({','.join(args)})"
        raise ValueError(f"Unsupported AlphaAgent syntax: {ast.dump(node)}")
    return visit(_tree(_ternary(text.replace("&&", " and ").replace("||", " or "))))


def evaluate_frame(text, frame):
    """Use original OHLCV precision and causal pandas operations; no signal cache."""
    memo = {}

    def visit(node):
        key = ast.dump(node)
        if key in memo:
            return memo[key]
        if isinstance(node, ast.Constant) and type(node.value) in (int, float):
            return node.value
        if isinstance(node, ast.Name) and node.id.startswith("feature_"):
            return frame["$" + node.id[8:]]
        if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Name) or node.keywords:
            raise ValueError("Expected a prefix operator expression")
        name = node.func.id
        args = [visit(arg) for arg in node.args]
        binary = {"Add": np.add, "Sub": np.subtract, "Mul": np.multiply,
                  "Div": np.divide, "Pow": np.power, "Gt": np.greater, "Lt": np.less,
                  "GetGreater": np.maximum, "GetLess": np.minimum,
                  "And": np.logical_and, "Or": np.logical_or,
                  "Ge": np.greater_equal, "Le": np.less_equal, "Eq": np.equal, "Ne": np.not_equal}
        unary = {"Abs": np.abs, "Sign": np.sign, "Log": np.log, "Sqrt": np.sqrt, "Exp": np.exp}
        if name in binary:
            result = binary[name](*args)
        elif name in unary:
            result = unary[name](*args)
        elif name == "DailyReturn":
            filled = args[0].groupby("instrument").ffill()
            result = filled.groupby("instrument").pct_change(fill_method=None).fillna(0)
        elif name == "Ref":
            x, window = args
            if type(window) is not int or window < 0:
                raise ValueError("AlphaAgent DELAY must not reference future data")
            result = x.groupby("instrument").shift(window)
        elif name in ("AADelta", "AAPctChange"):
            x, window = args
            if type(window) is not int or window < 1:
                raise ValueError("Difference periods must be positive integers")
            if name == "AADelta":
                result = x.groupby("instrument").diff(window)
            else:
                filled = x.groupby("instrument").ffill()
                result = filled.groupby("instrument").pct_change(periods=window, fill_method=None).fillna(0)
        elif name == "AAFullSampleQuantile":
            # This is the original non-causal PERCENTILE(x,q) implementation.
            # Retain it only for a clearly labeled fidelity diagnostic; strict pools exclude it.
            x, q = args
            if not 0 <= q <= 1:
                raise ValueError("Quantile must be in [0,1]")
            result = x.groupby("instrument").transform(lambda v: v.quantile(q))
        elif name == "AATsZScore":
            x, window = args
            if type(window) is not int or window < 1:
                raise ValueError("Time windows must be positive integers")
            grouped = x.groupby("instrument")
            mean = grouped.transform(lambda v: v.rolling(window, min_periods=1).mean())
            std = grouped.transform(lambda v: v.rolling(window, min_periods=1).std())
            result = (x - mean) / std
        elif name == "Where":
            condition, yes, no = args
            result = pd.Series(np.where(condition, yes, no), index=condition.index)
        elif name in ("CSRank", "CSMedian", "CSStd", "CSZScore"):
            x = args[0]
            grouped = x.groupby("datetime")
            if name == "CSRank":
                result = grouped.rank(pct=True)
            elif name == "CSZScore":
                result = (x - grouped.mean()) / grouped.std()
            else:
                result = grouped.median() if name == "CSMedian" else grouped.std()
        elif name in ("TsMean", "TsStd", "TsMed", "TsSum", "TsMin", "TsMax", "AATsRank",
                      "AATsMAD", "AATsQuantile", "AAEMA", "AAWMA", "AADecayLinear"):
            x, window = args[:2]
            if type(window) is not int or window < 1:
                raise ValueError("Time windows must be positive integers")
            def rolling(s):
                if name == "AAEMA":
                    return s.ewm(span=window, min_periods=1).mean()
                r = s.rolling(window, min_periods=1)
                if name in ("TsMean", "TsStd", "TsMed", "TsSum", "TsMin", "TsMax", "AATsRank"):
                    method = {"TsMean": "mean", "TsStd": "std", "TsMed": "median", "TsSum": "sum",
                              "TsMin": "min", "TsMax": "max", "AATsRank": "rank"}[name]
                    return getattr(r, method)(pct=True) if name == "AATsRank" else getattr(r, method)()
                if name == "AATsQuantile":
                    q = args[2]
                    if not 0 <= q <= 1:
                        raise ValueError("Quantile must be in [0,1]")
                    return r.quantile(q)
                if name == "AATsMAD":
                    return r.apply(lambda v: np.median(np.abs(v - np.median(v))), raw=True)
                if name == "AAWMA":
                    weights = np.array([0.9 ** i for i in range(window)][::-1])
                    return r.apply(lambda v: (v * weights[:len(v)]).sum() / weights[:len(v)].sum(), raw=True)
                weights = np.arange(1, window + 1, dtype=float)
                weights /= weights.sum()
                return r.apply(lambda v: (v * weights[:len(v)]).sum(), raw=True)
            result = x.groupby("instrument").transform(rolling)
        else:
            raise ValueError(f"Unsupported prefix operator: {name}")
        memo[key] = result
        return result
    return visit(_tree(text))


class AlphaAgentExpression(Expression):
    def __init__(self, text, signal_start="2015-01-01"):
        self.text, self.signal_start = text, signal_start

    def __str__(self):
        return self.text

    @property
    def is_featured(self):
        return True

    def evaluate(self, data, period=slice(0, 1)):
        if (period.start < -data.max_backtrack_days or
                period.stop - 1 > data.max_future_days):
            raise OutOfDataRangeError()
        start = data.max_backtrack_days + period.start
        stop = data.max_backtrack_days + data.n_days + period.stop - 1
        dates = pd.DatetimeIndex(data._dates[start:stop])
        frame = data.df_bak.reorder_levels(["datetime", "instrument"]).sort_index()
        day = frame.index.get_level_values("datetime")
        # Same initialization as the mined factors; exclude StockData's 2014 warmup.
        frame = frame.loc[(day >= pd.Timestamp(self.signal_start)) & (day <= dates[-1])]
        values = evaluate_frame(self.text, frame)
        wide = values.unstack("instrument").reindex(index=dates, columns=data._stock_ids)
        print(f"[AlphaAgent expression] {wide.shape}, finite={np.isfinite(wide.to_numpy()).mean():.3%}", flush=True)
        return torch.as_tensor(wide.to_numpy(), dtype=data.data.dtype, device=data.device)
