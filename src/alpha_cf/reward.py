"""Backtest cache for Phase A / Phase B.

The only window is the full train split (2010..2021). valid / test are reserved
for combo and never enter search. There is no reward scalar; LLM attribution
sees a tuple of metrics per edit (inc_residual, ic_f, ic_fp, delta_ic).

Data flow
---------
LLM propose  -> EditAction  ->  apply_edit -> f'  ->  cache.evaluate / cache.inc
                                                          |
                                                  (cached alpha tensor)
                                                          |
                                                  Pearson IC vs forward return
                                                  + daily OLS residual IC

The cache layer makes Phase A's 3 rounds × 5 edits per seed cheap: most edits
share sub-trees with the seed or with each other, and `str(expr)` is the key.
"""
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import torch
from torch import Tensor

from alphagen.data.expression import Expression, Feature, Ref
from alphagen.utils.correlation import batch_pearsonr, batch_spearmanr
from alphagen_qlib.calculator import QLibStockDataCalculator
from alphagen_qlib.stock_data import FeatureType, StockData

if __package__ is None:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from alpha_cf.config import (
    EVAL_SPLIT,
    TEST_END,
    TEST_START,
    TRAIN_END,
    TRAIN_START,
    VALID_END,
    VALID_START,
    qlib_path,
)

_REPO = Path(__file__).resolve().parents[2]


# name -> (start, end) date strings. Only "train" is used by Phase A / Phase B;
# "valid"/"test" exist only so combo subprocess can pick them later.
_SPLITS = {
    "train": (TRAIN_START, TRAIN_END),
    "valid": (VALID_START, VALID_END),
    "test": (TEST_START, TEST_END),
}


@dataclass(frozen=True)
class EvalResult:
    ic: float           # 日均 Pearson IC on the eval window
    rank_ic: float      # 日均 Spearman IC on the eval window
    r: float = 0.0      # 同 ic，留作兼容 CFRecord.r


# 20-day forward return used as the calculator target.
# Equivalent to `Ref($close, -20) / $close - 1`; the sign convention matches
# QLib's default return (looking forward 20 trading days).
def fwd_ret() -> Expression:
    close = Feature(FeatureType.CLOSE)
    return Ref(close, -20) / close - 1


# instrument, split -> QLib calculator on that date window.
# Defaults to cuda:0 if available; falls back to cpu for sanity-check runs.
def make_calculator(
    instrument: str,
    split: str = EVAL_SPLIT,
    device: Optional[torch.device] = None,
) -> QLibStockDataCalculator:
    if device is None:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    start, end = _SPLITS[split]
    data = StockData(
        instrument=instrument,
        start_time=start,
        end_time=end,
        qlib_path=str(_REPO / qlib_path(instrument)),
        device=device,
    )
    return QLibStockDataCalculator(data, fwd_ret())


# daily OLS residual of child on parent, then mean Pearson IC vs ret.
#
# For each day d (each row of [n_days, n_stocks]):
#   1. Regress child_d = a_d + b_d * parent_d + resid_d  (cross-sectional OLS,
#      skipping NaN/Inf positions)
#   2. resid_d is, by construction, orthogonal to parent_d on day d
#   3. daily_ic_d = Pearson(resid_d, ret_d)
# Then return mean(daily_ic) over all days.
#
# Geometrically: this is "what new predictive signal does child carry beyond
# what parent already captures?". Near 0 → child is redundant wrt parent.
# Far from 0 (positive or negative) → child has independent signal.
#
# All arithmetic is NaN-safe: invalid positions are masked out, means / var /
# cov use `n_safe = max(observed, 1)` as denominator, and the final daily
# Pearson is forced to 0 for days with too few observations or degenerate
# residual variance.
def incremental_ic(parent: Tensor, child: Tensor, ret: Tensor) -> float:
    # Per-position finite mask. `n` = #valid stocks per day.
    valid = torch.isfinite(parent) & torch.isfinite(child) & torch.isfinite(ret)
    n = valid.sum(dim=1).to(dtype=parent.dtype)
    n_safe = n.clamp(min=1.0)  # avoid div-by-0 on all-NaN days

    # Cross-sectional means; treat invalid positions as 0 (then divide by n_safe).
    xf = torch.where(valid, parent, torch.zeros_like(parent))
    yf = torch.where(valid, child, torch.zeros_like(child))
    mx = xf.sum(dim=1) / n_safe
    my = yf.sum(dim=1) / n_safe

    # Centered values, masked back to 0 at invalid positions.
    xc = torch.where(valid, parent - mx[:, None], torch.zeros_like(parent))
    yc = torch.where(valid, child - my[:, None], torch.zeros_like(child))

    # Var / cov across stocks for each day.
    var_x = (xc * xc).sum(dim=1) / n_safe
    cov = (xc * yc).sum(dim=1) / n_safe

    # OLS slope b = cov/var. Guard var_x near zero (parent is constant on
    # that day) → no OLS, slope forced to 0.
    b = torch.where(var_x > 1e-8, cov / var_x.clamp(min=1e-12), torch.zeros_like(var_x))
    a = my - b * mx

    # Residual: child - (a + b * parent). NaN positions stay NaN so the
    # subsequent Pearson ignores them (NaN propagation in batch_pearsonr).
    nan = torch.full((), float("nan"), dtype=parent.dtype, device=parent.device)
    resid = torch.where(valid, child - (a[:, None] + b[:, None] * parent), nan)

    # Residual mean / variance per day (used as degeneracy filters below).
    me = torch.where(valid, resid, torch.zeros_like(resid)).sum(dim=1) / n_safe
    ve = (torch.where(valid, resid - me[:, None], torch.zeros_like(resid)) ** 2).sum(dim=1) / n_safe

    # Daily Pearson IC of residual vs forward return.
    daily = batch_pearsonr(resid, ret).clone()
    # Drop days that are too sparse (<5 valid stocks → OLS unreliable).
    daily[n < 5] = 0.0
    # Drop days where residual has no variance (everything explained by parent).
    daily[ve < 1e-12] = 0.0
    # Any remaining NaN/Inf from batch_pearsonr → 0 (defensive).
    daily[~torch.isfinite(daily)] = 0.0
    return float(daily.mean().item())


# same daily Spearman as calculator._calc_rIC; chunk days so [d,s,s] fits in 8GB.
# batch_spearmanr materialises an [n_days, n_stocks, n_stocks] corr tensor per
# call; with 5000+ stocks × 16-day chunks the peak is ~8GB on RTX-class GPUs.
def _rank_ic_daily(value: Tensor, target: Tensor, chunk_days: int = 16) -> Tensor:
    parts = []
    n_days = value.shape[0]
    for i in range(0, n_days, chunk_days):
        sl = slice(i, min(i + chunk_days, n_days))
        parts.append(batch_spearmanr(value[sl], target[sl]))
    daily = torch.cat(parts)
    return torch.where(torch.isfinite(daily), daily, torch.zeros_like(daily))


def _rank_ic_mean(value: Tensor, target: Tensor, chunk_days: int = 16) -> float:
    return float(_rank_ic_daily(value, target, chunk_days).mean().item())


# Memoized IC; failed / non-finite IC -> zeros. Single eval window only.
# Keys are `str(expr)` so structurally identical expressions hit the cache.
class RewardCache:
    def __init__(self, calculator: QLibStockDataCalculator):
        self.calc = calculator
        self._cache: Dict[str, EvalResult] = {}        # expr -> EvalResult
        self._alpha: Dict[str, Optional[Tensor]] = {}  # expr -> alpha tensor (or None on failure)
        self.n_eval = 0  # monotonic; counts cache misses (real backtests run)
        ret = calculator.target_value
        if ret is None:
            raise ValueError("RewardCache needs a calculator with a target")

    # expr -> day-normalized factor matrix, or None if evaluate failed.
    def alpha(self, expr: Expression) -> Optional[Tensor]:
        key = str(expr)
        if key in self._alpha:
            return self._alpha[key]
        # Increment before try so failed backtests also count (they cost compute).
        self.n_eval += 1
        try:
            with torch.no_grad():
                val = self.calc._calc_alpha(expr)
        except Exception:
            # Cache the failure so we don't retry the same broken expr every call.
            self._alpha[key] = None
            return None
        self._alpha[key] = val
        return val

    # expr -> EvalResult; cache hit skips the backtest.
    def evaluate(self, expr: Expression) -> EvalResult:
        key = str(expr)
        hit = self._cache.get(key)
        if hit is not None:
            return hit
        val = self.alpha(expr)
        if val is None or self.calc.target_value is None:
            # Failed alpha → zero EvalResult. Downstream sees 0 IC, not an error.
            rec = EvalResult(ic=0.0, rank_ic=0.0, r=0.0)
        else:
            ret = self.calc.target_value
            with torch.no_grad():
                ic = _safe_ic(val, ret)
                rank_ic = _safe_rank_ic(val, ret)
            rec = EvalResult(ic=ic, rank_ic=rank_ic, r=ic)
        self._cache[key] = rec
        return rec

    # IC(f') - IC(f). Both sides cache-hit, so this is just one subtraction.
    def delta(self, f: Expression, f_prime: Expression) -> float:
        return self.evaluate(f_prime).ic - self.evaluate(f).ic

    # inc(f', f): Pearson IC of residual(f' | f) vs forward return.
    # Returns 0.0 on any failure (missing alpha, NaN, ...). This is the
    # KEY signal for redundancy: ~0 means f' carries no info beyond f.
    def inc(self, f_prime: Expression, f: Expression) -> float:
        parent = self.alpha(f)
        child = self.alpha(f_prime)
        ret = self.calc.target_value
        if parent is None or child is None or ret is None:
            return 0.0
        with torch.no_grad():
            val = incremental_ic(parent, child, ret)
        return val if math.isfinite(val) else 0.0

    # inc of g's alpha against a raw pool-signal tensor (Phase B greedy select).
    # Same OLS-residual IC as inc(), but the "parent" is the current combined
    # pool signal instead of a single expression.
    def inc_vs_signal(self, g: Expression, signal: Tensor) -> float:
        child = self.alpha(g)
        ret = self.calc.target_value
        if child is None or ret is None or signal is None:
            return 0.0
        with torch.no_grad():
            val = incremental_ic(signal, child, ret)
        return val if math.isfinite(val) else 0.0

    # mean Pearson corr of two alphas across days.
    # Phase B can use this to pre-filter candidates highly correlated with the pool.
    def mean_cs_corr(self, a: Expression, b: Expression) -> float:
        xa = self.alpha(a)
        xb = self.alpha(b)
        if xa is None or xb is None:
            # Missing alpha -> treat as perfectly correlated (conservative).
            return 1.0
        with torch.no_grad():
            daily = batch_pearsonr(xa, xb)
            daily = torch.where(torch.isfinite(daily), daily, torch.zeros_like(daily))
            x = float(daily.mean().item())
        return x if math.isfinite(x) else 1.0


def _safe_ic(value: Tensor, ret: Tensor) -> float:
    daily = batch_pearsonr(value, ret)
    daily = torch.where(torch.isfinite(daily), daily, torch.zeros_like(daily))
    x = float(daily.mean().item())
    return x if math.isfinite(x) else 0.0


def _safe_rank_ic(value: Tensor, ret: Tensor) -> float:
    return _rank_ic_mean(value, ret)


if __name__ == "__main__":
    # Synthetic OLS sanity check (no qlib data needed).
    torch.manual_seed(0)
    days, stocks = 40, 30
    parent = torch.randn(days, stocks)
    ret = 0.4 * parent + 0.6 * torch.randn(days, stocks)
    child_aff = 2.0 * parent + 1.0   # affine shift  → OLS absorbs fully
    child_flip = -parent             # exact negation → OLS slope b=-1
    child_new = ret + 0.3 * torch.randn(days, stocks)  # independent signal
    z_aff = incremental_ic(parent, child_aff, ret)
    z_flip = incremental_ic(parent, child_flip, ret)
    z_self = incremental_ic(parent, parent, ret)
    z_new = incremental_ic(parent, child_new, ret)
    assert abs(z_self) < 1e-5, z_self
    assert abs(z_aff) < 1e-5, z_aff
    assert abs(z_flip) < 1e-5, z_flip
    assert z_new > 0.2, z_new
    print(f"inc affine={z_aff:.4f}  flip={z_flip:.4f}  self={z_self:.4f}  new={z_new:.4f}")

    from alpha_cf.edits import apply_edit
    from alpha_cf.pool import parse_expr
    from alpha_cf.types import EditAction

    # End-to-end smoke test: real qlib data + edit + cache hit.
    instrument = sys.argv[1] if len(sys.argv) > 1 else "csi300"
    cache = RewardCache(make_calculator(instrument, EVAL_SPLIT))

    f = parse_expr("TsMean($close,20)")
    a = cache.evaluate(f)
    b = cache.evaluate(f)
    # Second evaluate must be a cache hit: n_eval stays at 1 (one backtest).
    assert a == b and cache.n_eval == 1
    print(f"f={f}")
    print(f"  ic={a.ic:.4f}  rank_ic={a.rank_ic:.4f}")

    f2 = apply_edit(f, EditAction("window_replace", 0, "10"))
    d = cache.delta(f, f2)
    inc = cache.inc(f2, f)
    print(
        f"f'={f2}  ic={cache.evaluate(f2).ic:.4f}  delta={d:.4f}  inc={inc:.4f}  "
        f"n_eval={cache.n_eval}"
    )