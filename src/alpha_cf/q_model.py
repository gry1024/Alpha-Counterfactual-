import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import lightgbm as lgb
import numpy as np

if __package__ is None:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from alphagen.data.expression import (
    Constant,
    Expression,
    Feature,
    Operator,
    PairRollingOperator,
    RollingOperator,
)

from alpha_cf.config import (
    BETA,
    CONSTANTS,
    DELTA_TIMES,
    FEATURE_NAMES,
    LAMBDA_DIV,
    MIN_Q_SAMPLES,
    OPERATORS,
)
from alpha_cf.edits import EDIT_KINDS, apply_edit, children, parse_wrap_template
from alpha_cf.pool import parse_expr
from alpha_cf.types import CFRecord, EditAction

FEAT2I: Dict[str, int] = {n: i for i, n in enumerate(FEATURE_NAMES)}
OP2I: Dict[str, int] = {op.__name__: i for i, op in enumerate(OPERATORS)}
WIN2I: Dict[int, int] = {w: i for i, w in enumerate(DELTA_TIMES)}
CONST2I: Dict[float, int] = {float(c): i for i, c in enumerate(CONSTANTS)}
KIND2I: Dict[str, int] = {k: i for i, k in enumerate(EDIT_KINDS)}

# kind, site_id, n_nodes, depth, parent_op, old_feat, old_op, old_win,
# new_feat, new_op, new_win, new_const, wrap_side
N_COLS = 13
CAT_COLS = [0, 4, 5, 6, 7, 8, 9, 10, 11, 12]
_TREE = dict(
    num_leaves=15,
    min_data_in_leaf=8,
    learning_rate=0.05,
    feature_fraction=0.8,
    verbosity=-1,
    seed=0,
)
_PARAMS = dict(objective="regression", metric="l2", **_TREE)
_PARAMS_LO = dict(objective="quantile", alpha=0.1, **_TREE)
_PARAMS_HI = dict(objective="quantile", alpha=0.9, **_TREE)
_EXPR: Dict[str, Expression] = {}


# expr string -> cached Expression
def _parse(s: str) -> Expression:
    e = _EXPR.get(s)
    if e is None:
        e = parse_expr(s)
        _EXPR[s] = e
    return e


# expr -> [(node, parent_idx, depth), ...] in preorder (index == site_id)
def _walk(expr: Expression) -> List[Tuple[Expression, int, int]]:
    rows: List[Tuple[Expression, int, int]] = []

    def rec(node: Expression, parent: int, depth: int) -> None:
        i = len(rows)
        rows.append((node, parent, depth))
        for ch in children(node):
            rec(ch, i, depth + 1)

    rec(expr, -1, 0)
    return rows


# dict index -> LightGBM category (0 = missing)
def _cat(i: int) -> int:
    return i + 1 if i >= 0 else 0


# node -> (feat, op, window) codes
def _old_codes(node: Optional[Expression]) -> Tuple[int, int, int]:
    feat = op = win = -1
    if isinstance(node, Feature):
        feat = FEAT2I.get(str(node), -1)
    elif isinstance(node, Operator):
        op = OP2I.get(type(node).__name__, -1)
        if isinstance(node, (RollingOperator, PairRollingOperator)):
            win = WIN2I.get(int(node._delta_time), -1)
    return _cat(feat), _cat(op), _cat(win)


# action -> (new_feat, new_op, new_win, new_const, wrap_side)
def _new_codes(action: EditAction) -> Tuple[int, int, int, int, int]:
    feat = op = win = const = side = -1
    new = action.new_value
    if action.kind == "feature_replace":
        feat = FEAT2I.get(new or "", -1)
    elif action.kind == "operator_replace":
        op = OP2I.get(new or "", -1)
    elif action.kind == "window_replace" and new is not None:
        try:
            win = WIN2I.get(int(new), -1)
        except (TypeError, ValueError):
            pass
    elif action.kind == "wrap" and new:
        spec = parse_wrap_template(new)
        if spec is not None:
            op = OP2I.get(spec.op_name, -1)
            win = WIN2I.get(spec.window, -1) if spec.window is not None else -1
            side = 0 if spec.side == "left" else 1
            if isinstance(spec.other, Feature):
                feat = FEAT2I.get(str(spec.other), -1)
            elif isinstance(spec.other, Constant):
                const = CONST2I.get(float(spec.other._value), -1)
    return _cat(feat), _cat(op), _cat(win), _cat(const), _cat(side)


# expr, action -> tabular row for LightGBM
def encode(expr: Expression, action: EditAction) -> np.ndarray:
    rows = _walk(expr)
    n = len(rows)
    sid = int(action.site_id)
    node, pidx, depth = rows[sid] if 0 <= sid < n else (None, -1, 0)
    parent_op = -1
    if pidx >= 0:
        parent_op = OP2I.get(type(rows[pidx][0]).__name__, -1)
    old_feat, old_op, old_win = _old_codes(node)
    new_feat, new_op, new_win, new_const, wrap_side = _new_codes(action)
    return np.array(
        [
            _cat(KIND2I.get(action.kind, -1)),
            sid,
            n,
            depth,
            _cat(parent_op),
            old_feat,
            old_op,
            old_win,
            new_feat,
            new_op,
            new_win,
            new_const,
            wrap_side,
        ],
        dtype=np.float32,
    )


# pred, y -> Spearman of ranks
def _spearman(pred: np.ndarray, y: np.ndarray) -> float:
    if pred.size < 2:
        return 0.0
    rx = pred.argsort().argsort().astype(np.float64)
    ry = y.argsort().argsort().astype(np.float64)
    rx -= rx.mean()
    ry -= ry.mean()
    den = np.linalg.norm(rx) * np.linalg.norm(ry)
    return float(rx @ ry / den) if den > 0 else 0.0


# row vs tried matrix -> diversity in (0, 1]
def _div(x: np.ndarray, tried: Optional[np.ndarray]) -> float:
    if tried is None or tried.size == 0:
        return 1.0
    dmin = float(np.linalg.norm(tried - x, axis=1).min())
    return dmin / (1.0 + dmin)


# records -> X [n, N_COLS]
def _matrix(pairs: Sequence[Tuple[Expression, EditAction]]) -> np.ndarray:
    return np.stack([encode(f, a) for f, a in pairs])


# Dataset, params -> booster
def _boost(ds: lgb.Dataset, params: dict, rounds: int) -> lgb.Booster:
    return lgb.train(params, ds, num_boost_round=rounds)


# μ and quantile-spread σ = 0.5 * max(q90-q10, 0)
def _mu_sig(
    X: np.ndarray,
    mean: Optional[lgb.Booster],
    lo: Optional[lgb.Booster],
    hi: Optional[lgb.Booster],
) -> Tuple[np.ndarray, np.ndarray]:
    n = X.shape[0]
    if mean is None:
        return np.zeros(n), np.zeros(n)
    mu = np.asarray(mean.predict(X), dtype=np.float64)
    if lo is None or hi is None:
        return mu, np.zeros(n)
    spread = np.asarray(hi.predict(X), dtype=np.float64) - np.asarray(lo.predict(X), dtype=np.float64)
    return mu, 0.5 * np.maximum(spread, 0.0)


# Fit LightGBM on Δ; rank unseen edits by μ + βσ + λD
class QModel:
    def __init__(self):
        self.mean: Optional[lgb.Booster] = None
        self.lo: Optional[lgb.Booster] = None
        self.hi: Optional[lgb.Booster] = None
        self.fitted = False

    # records -> train mean + 0.1/0.9 quantiles; prints spearman
    def fit(self, records: Sequence[CFRecord], rounds: int = 80) -> Dict[str, float]:
        if not records:
            self.mean = self.lo = self.hi = None
            self.fitted = True
            return {"n": 0.0, "spearman": 0.0}
        X = _matrix([(_parse(r.f), r.action) for r in records])
        y = np.array([r.delta for r in records], dtype=np.float32)
        ds = lgb.Dataset(X, y, categorical_feature=CAT_COLS, free_raw_data=False)
        self.mean = _boost(ds, _PARAMS, rounds)
        self.lo = _boost(ds, _PARAMS_LO, rounds)
        self.hi = _boost(ds, _PARAMS_HI, rounds)
        self.fitted = True
        mu, _ = _mu_sig(X, self.mean, self.lo, self.hi)
        metrics = {"n": float(len(y)), "spearman": _spearman(mu, y)}
        print(f"[q] fit n={len(y)}  spearman={metrics['spearman']:.3f}", flush=True)
        return metrics

    # f, a -> (μ, σ) of Δ; σ is half the 0.1–0.9 quantile width
    def predict(self, f: Expression, action: EditAction) -> Tuple[float, float]:
        mu, sig = _mu_sig(encode(f, action).reshape(1, -1), self.mean, self.lo, self.hi)
        return float(mu[0]), float(sig[0])

    # Score unseen edits; returns actions best-first
    def rank(self, f: Expression, actions: List[EditAction], records: Sequence[CFRecord]) -> List[EditAction]:
        legal: List[EditAction] = []
        skipped: List[EditAction] = []
        for a in actions:
            if apply_edit(f, a) is None:
                print(
                    f"[q] ILLEGAL skip  kind={a.kind}  site_id={a.site_id}  "
                    f"new_value={a.new_value!r}  T(f,a)=None",
                    flush=True,
                )
                skipped.append(a)
            else:
                legal.append(a)
        print(f"[q] rank  legal={len(legal)} skipped={len(skipped)} n_data={len(records)}", flush=True)
        if not legal:
            return skipped
        X = _matrix([(f, a) for a in legal])
        mu, sig = _mu_sig(X, self.mean, self.lo, self.hi)
        tried = _matrix([(_parse(r.f), r.action) for r in records]) if records else None
        scores = [
            float(mu[i] + BETA * sig[i] + LAMBDA_DIV * _div(X[i], tried))
            for i in range(len(legal))
        ]
        order = sorted(range(len(legal)), key=lambda i: scores[i], reverse=True)
        return [legal[i] for i in order] + skipped


# shared record list -> rank_fn that refits Q when the list grows
def make_rank_fn(shared: List[CFRecord]):
    q = QModel()
    n_fit = [-1]

    def rank_fn(f: Expression, actions: List[EditAction], records: List[CFRecord]) -> List[EditAction]:
        data = shared + records
        if len(data) != n_fit[0]:
            q.fit(data)
            n_fit[0] = len(data)
        return q.rank(f, actions, data)

    return rank_fn


if __name__ == "__main__":
    f_demo = parse_expr("TsMean($close,20)")
    xw = encode(f_demo, EditAction("window_replace", 0, "10"))
    assert xw.shape == (N_COLS,)
    assert xw[0] == _cat(KIND2I["window_replace"]) and xw[6] == _cat(OP2I["TsMean"]) and xw[10] == _cat(WIN2I[10])
    xwrap = encode(f_demo, EditAction("wrap", 0, "Div($_,1.0)"))
    assert xwrap[0] == _cat(KIND2I["wrap"]) and xwrap[8] == 0
    assert xwrap[9] == _cat(OP2I["Div"]) and xwrap[11] == _cat(CONST2I[1.0]) and xwrap[12] == 1
    xwrap_r = encode(f_demo, EditAction("wrap", 0, "Div(1.0,$_)"))
    assert xwrap_r[12] == 2
    fb = parse_expr("Add($close,$open)")
    xd = encode(fb, EditAction("subtree_delete", 1, None))
    assert xd[0] == _cat(KIND2I["subtree_delete"]) and xd[8:].sum() == 0  # news all missing

    records: List[CFRecord] = []
    for op in ("TsMean", "TsMax", "TsMin", "TsStd", "TsSum"):
        for feat in FEATURE_NAMES:
            for base_w in (10, 20, 30):
                f = parse_expr(f"{op}({feat},{base_w})")
                for w in DELTA_TIMES:
                    if w == base_w:
                        continue
                    fp = parse_expr(f"{op}({feat},{w})")
                    d = 0.01 * (w - base_w) / 40.0
                    records.append(CFRecord(
                        str(f), EditAction("window_replace", 0, str(w)), str(fp),
                        d, 0.0, 0.0, 0.1, 0.08,
                    ))
    assert len(records) >= MIN_Q_SAMPLES
    q = QModel()
    m = q.fit(records)
    assert q.fitted and m["spearman"] > 0.3
    f0 = parse_expr("TsMean($close,20)")
    acts = [
        EditAction("window_replace", 0, "10"),
        EditAction("window_replace", 0, "40"),
        EditAction("feature_replace", 1, "$volume"),
        EditAction("operator_replace", 0, "TsMax"),
    ]
    ranked = q.rank(f0, acts, records)
    assert set(ranked) == set(acts)
    mu_p, sig_p = q.predict(f0, acts[0])
    assert isinstance(mu_p, float) and sig_p >= 0.0
    assert q.lo is not None and q.hi is not None
    fn = make_rank_fn(list(records))
    assert len(fn(f0, acts, [])) == len(acts)
    from alpha_cf.search import _select_to_eval
    assert len(_select_to_eval(f0, acts, [], 3, 0, fn)) == 3
    print(f"ok  cols={N_COLS}  n={len(records)}  spearman={m['spearman']:.3f}")
    print("ranked", [(a.kind, a.new_value) for a in ranked])
