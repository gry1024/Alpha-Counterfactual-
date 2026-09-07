import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional

import torch
from alphagen.data.expression import Expression, Feature, Ref
from alphagen_qlib.calculator import QLibStockDataCalculator
from alphagen_qlib.stock_data import FeatureType, StockData

if __package__ is None:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from alpha_cf.config import (
    TEST_END,
    TEST_START,
    TRAIN_END,
    TRAIN_START,
    VALID_END,
    VALID_START,
    qlib_path,
)

_REPO = Path(__file__).resolve().parents[2]

# split name -> (start, end) calendar dates
_SPLITS = {
    "train": (TRAIN_START, TRAIN_END),
    "valid": (VALID_START, VALID_END),
    "test": (TEST_START, TEST_END),
}


# IC / RankIC / R=|IC| for one factor
@dataclass(frozen=True)
class EvalResult:
    ic: float
    rank_ic: float
    r: float  # abs(ic)


# 20-day forward return used as the calculator target
def fwd_ret() -> Expression:
    close = Feature(FeatureType.CLOSE)
    return Ref(close, -20) / close - 1


# instrument, split -> QLib calculator on that date window
def make_calculator(
    instrument: str,
    split: str = "train",
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


# Memoized R(f); failed / non-finite IC -> zeros
class RewardCache:
    # calculator -> empty IC cache
    def __init__(self, calculator: QLibStockDataCalculator):
        self.calc = calculator
        self._cache: Dict[str, EvalResult] = {}
        self.n_eval = 0

    # expr -> EvalResult; cache hit skips the backtest
    def evaluate(self, expr: Expression) -> EvalResult:
        key = str(expr)
        hit = self._cache.get(key)
        if hit is not None:
            return hit
        try:
            with torch.no_grad():
                ic, rank_ic = self.calc.calc_single_all_ret(expr)
        except Exception:
            ic, rank_ic = 0.0, 0.0  # failed / non-finite IC counts as zero
        ic = float(ic) if math.isfinite(ic) else 0.0
        rank_ic = float(rank_ic) if math.isfinite(rank_ic) else 0.0
        rec = EvalResult(ic=ic, rank_ic=rank_ic, r=abs(ic))
        self._cache[key] = rec
        self.n_eval += 1
        return rec

    # R(f') - R(f)
    def delta(self, f: Expression, f_prime: Expression) -> float:
        return self.evaluate(f_prime).r - self.evaluate(f).r


if __name__ == "__main__":
    from alpha_cf.edits import apply_edit
    from alpha_cf.pool import parse_expr
    from alpha_cf.types import EditAction

    instrument = sys.argv[1] if len(sys.argv) > 1 else "sp500"
    cache = RewardCache(make_calculator(instrument, "test"))

    f = parse_expr("TsMean($close,20)")
    a = cache.evaluate(f)
    b = cache.evaluate(f)
    assert a == b and cache.n_eval == 1
    print(f"f={f}")
    print(f"  ic={a.ic:.4f}  rank_ic={a.rank_ic:.4f}  R={a.r:.4f}")

    f2 = apply_edit(f, EditAction("window_replace", 0, "10"))
    d = cache.delta(f, f2)
    print(f"f'={f2}  R={cache.evaluate(f2).r:.4f}  Δ={d:.4f}  n_eval={cache.n_eval}")
