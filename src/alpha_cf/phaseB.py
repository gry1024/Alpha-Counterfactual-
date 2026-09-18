"""Phase B: counterfactual module recombination (CF-MR).

Two steps only:
  Breed  — cross every pair of trimmed modules (Phase A outputs = load-bearing
           cores) with NONLINEAR operators. Add/Sub are skipped on purpose:
           they are linear combinations of the parents, so the downstream OLS
           combo would drop them as collinear anyway.
  Select — greedy by marginal incremental IC: inc(g, pool_signal) measures how
           much independent predictive info g adds beyond the current pool.
           Trimmed factors are never removed (add-only), so Phase A's work is
           preserved. Everything runs on the train split only.
"""
import json
import sys
from itertools import combinations
from pathlib import Path
from typing import List, Optional, Sequence

import torch
from torch import Tensor

if __package__ is None:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from alphagen.data.expression import Div, Expression, Mul, Rank, TsCorr

from alpha_cf.config import K_NEW, MAX_CANDIDATES, RECOMB_WINDOWS, TAU_INC
from alpha_cf.reward import RewardCache


# expr -> Rank(expr), or None when the wrapper is not featured.
# Rank makes cross-factor averaging scale-free (IC is rank-based anyway).
def _ranked(e: Expression) -> Optional[Expression]:
    r = Rank(e)
    return r if r.is_featured else None


# trimmed modules -> nonlinear cross-bred candidate expressions.
# Operators: Div/Mul both directions (ratio & interaction) + TsCorr at several
# windows (rolling co-movement). Deduped by canonical string.
def breed(modules: Sequence[Expression]) -> List[Expression]:
    cands: List[Expression] = []
    seen = {str(m) for m in modules}

    def push(g: Optional[Expression]) -> None:
        if g is None or not g.is_featured:
            return
        key = str(g)
        if key not in seen:
            seen.add(key)
            cands.append(g)

    for fi, fj in combinations(modules, 2):
        push(Div(fi, fj))
        push(Div(fj, fi))
        push(Mul(fi, fj))
        for w in RECOMB_WINDOWS:
            push(TsCorr(fi, fj, w))
    # Stride-subsample to the budget; pairs are ordered, so this stays deterministic.
    return cands[:: max(1, len(cands) // MAX_CANDIDATES)][:MAX_CANDIDATES]


# trimmed modules -> equal-weight pool signal tensor (mean of ranked alphas).
# NaN where every factor is NaN; incremental_ic masks those positions.
# Returns (signal, n_factors_in_signal).
def _pool_signal(trimmed: Sequence[Expression],
                 cache: RewardCache) -> tuple:
    alphas = [cache.alpha(_ranked(f)) for f in trimmed]
    alphas = [a for a in alphas if a is not None]
    if not alphas:
        return None, 0
    return torch.nanmean(torch.stack(alphas), dim=0), len(alphas)


# trimmed, cache, out_dir -> writes final_pool.json (trimmed + up to K_NEW new).
# Returns the pool dict. On any failure the trimmed pool is copied through so
# the downstream auto-combo always has something to evaluate.
def run_phaseB(trimmed: Sequence[Expression], cache: RewardCache, out_dir: str) -> dict:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    final_path = out / "final_pool.json"

    def _dump(exprs: Sequence[Expression]) -> dict:
        d = {"exprs": [str(e) for e in exprs], "weights": []}
        with open(final_path, "w", encoding="utf-8") as w:
            json.dump(d, w, indent=2)
        return d

    cands = breed(trimmed)
    print(f"[phaseB] bred {len(cands)} candidates from {len(trimmed)} modules", flush=True)
    if not cands:
        return _dump(list(trimmed))

    # Pre-filter: candidates must evaluate to a finite IC (cheap, cached).
    cands = [g for g in cands if torch.isfinite(torch.tensor(cache.evaluate(g).ic))]
    print(f"[phaseB] evaluable candidates: {len(cands)}", flush=True)

    # Greedy forward selection: at each step pick the candidate with the
    # largest |inc| vs the CURRENT pool signal, then fold it into the signal
    # (sign-aligned) and re-score the rest. |inc| because combo gates on |RIC|.
    signal, n_in = _pool_signal(trimmed, cache)
    if signal is None:
        return _dump(list(trimmed))

    selected: List[Expression] = []
    remaining = list(cands)
    log = open(out / "phaseB_selection.jsonl", "w", encoding="utf-8")
    while remaining and len(selected) < K_NEW:
        scored = [(g, cache.inc_vs_signal(g, signal)) for g in remaining]
        g, inc = max(scored, key=lambda t: abs(t[1]))
        if abs(inc) < TAU_INC:
            break
        selected.append(g)
        remaining.remove(g)
        # Fold g into the pool signal, sign-aligned with its residual direction.
        a = cache.alpha(_ranked(g))
        if a is not None:
            s = torch.sign(torch.tensor(inc)).item()
            signal = (signal * n_in + s * torch.nan_to_num(a)) / (n_in + 1)
            n_in += 1
        log.write(json.dumps({"f": str(g), "ic": cache.evaluate(g).ic, "inc": inc}) + "\n")
        log.flush()
        print(f"[phaseB] + {g}   inc={inc:+.4f}", flush=True)
    log.close()

    final = list(trimmed) + selected
    print(f"[phaseB] DONE  final_pool size={len(final)} "
          f"({len(trimmed)} trimmed + {len(selected)} new)", flush=True)
    return _dump(final)


if __name__ == "__main__":
    import argparse
    from datetime import datetime

    from alpha_cf.config import EVAL_SPLIT
    from alpha_cf.pool import load_pool
    from alpha_cf.reward import make_calculator

    p = argparse.ArgumentParser()
    p.add_argument("--trimmed_pool_json", required=True)
    p.add_argument("--instrument", type=str, default="csi300")
    p.add_argument("--out_dir", type=str, default="")
    args = p.parse_args()

    if not args.out_dir:
        args.out_dir = f"data/cf_logs/phaseB_{datetime.now():%Y%m%d_%H%M%S}"
    trimmed = load_pool(args.trimmed_pool_json)
    cache = RewardCache(make_calculator(args.instrument, EVAL_SPLIT))
    run_phaseB(trimmed, cache, args.out_dir)
    print(f"[phaseB] DONE  out_dir={args.out_dir}", flush=True)
