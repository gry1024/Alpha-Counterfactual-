from typing import Callable, List, Optional, Sequence, Tuple

from alphagen.data.expression import Expression

from alpha_cf.config import K, MAX_ROUNDS, MIN_Q_SAMPLES, PATIENCE, TOPK
from alpha_cf.edits import apply_edit
from alpha_cf.llm import propose as llm_propose
from alpha_cf.llm import record_illegal
from alpha_cf.reward import RewardCache
from alpha_cf.types import CFRecord, EditAction

ProposeFn = Callable[[Expression, float, int, Optional[Sequence[CFRecord]]], List[EditAction]]
RankFn = Callable[[Expression, List[EditAction], List[CFRecord]], List[EditAction]]


# actions, records -> all K if cold-start, else rank_fn's topk
def _select_to_eval(
    f: Expression,
    actions: List[EditAction],
    records: List[CFRecord],
    topk: int,
    min_q_samples: int,
    rank_fn: Optional[RankFn],
) -> List[EditAction]:
    # not enough CF samples yet: backtest every proposal
    if rank_fn is None or len(records) < min_q_samples:
        return actions
    return rank_fn(f, actions, records)[:topk]


# one seed: greedy rounds of propose → (optional Q) → backtest → keep Δ>0
def search_one(
    f: Expression,
    cache: RewardCache,
    propose_fn: ProposeFn = llm_propose,
    k: int = K,
    topk: int = TOPK,
    max_rounds: int = MAX_ROUNDS,
    patience: int = PATIENCE,
    min_q_samples: int = MIN_Q_SAMPLES,
    rank_fn: Optional[RankFn] = None,
    seed_idx: int = 0,
) -> Tuple[List[CFRecord], Expression]:
    cur = cache.evaluate(f)
    records: List[CFRecord] = []
    stale = 0
    print(f"[search] seed={seed_idx} start  R={cur.r:.4f}  ic={cur.ic:.4f}  rank_ic={cur.rank_ic:.4f}", flush=True)

    for rnd in range(max_rounds):
        print(f"[search] seed={seed_idx} round={rnd}/{max_rounds}  stale={stale}  f={f}", flush=True)
        raw = propose_fn(f, cur.r, k, records)
        prepared: List[Tuple[EditAction, Expression]] = []
        seen_fp = set()
        for a in raw[:k]:  # drop no-ops and duplicate f' in this round
            fp = apply_edit(f, a)
            if fp is None:
                record_illegal("search", f, a, round=rnd, seed=seed_idx)
                continue
            key = str(fp)
            if key == str(f) or key in seen_fp:
                print(f"[search] skip no-op/dup  {a.kind}@{a.site_id} {a.new_value!r}  ->  {fp}", flush=True)
                continue
            seen_fp.add(key)
            prepared.append((a, fp))
        if not prepared:
            stale += 1
            print(f"[search] no usable edits  stale={stale}/{patience}", flush=True)
            if stale >= patience:
                print("[search] stop: patience", flush=True)
                break
            continue

        q_on = rank_fn is not None and len(records) >= min_q_samples
        picked = _select_to_eval(
            f, [a for a, _ in prepared], records, topk, min_q_samples, rank_fn
        )
        keys = {(a.kind, a.site_id, a.new_value) for a in picked}
        batch = [(a, fp) for a, fp in prepared if (a.kind, a.site_id, a.new_value) in keys]
        if not batch:
            batch = prepared
        print(f"[search] Q={'on' if q_on else 'off'}  backtest {len(batch)}/{len(prepared)}", flush=True)

        best_d = None
        best_fp = None
        best_ev = None
        for a, fp in batch:
            n0 = cache.n_eval
            ev = cache.evaluate(fp)
            d = ev.r - cur.r
            tag = f"#{cache.n_eval}" if cache.n_eval > n0 else "cache"
            print(
                f"[eval] {tag}  f'={fp}  ic={ev.ic:.4f}  rank_ic={ev.rank_ic:.4f}  "
                f"R={ev.r:.4f}  Δ={d:+.4f}  via {a.kind}@{a.site_id} {a.new_value!r}",
                flush=True,
            )
            records.append(CFRecord(
                f=str(f),
                action=a,
                f_prime=str(fp),
                delta=d,
                ic=ev.ic,
                rank_ic=ev.rank_ic,
                r=ev.r,
                r_seed=cur.r,
                extra={"round": rnd, "seed": seed_idx},
            ))
            if best_d is None or d > best_d:
                best_d, best_fp, best_ev = d, fp, ev

        if best_d is not None and best_d > 0:
            f = best_fp
            cur = best_ev
            stale = 0
            print(f"[search] ACCEPT  Δ={best_d:+.4f}  new f={f}  R={cur.r:.4f}", flush=True)
        else:
            stale += 1
            print(f"[search] reject  best Δ={best_d}  stale={stale}/{patience}", flush=True)
            if stale >= patience:
                print("[search] stop: patience", flush=True)
                break

    return records, f  # final expr after all greedy edits (may equal the seed)
