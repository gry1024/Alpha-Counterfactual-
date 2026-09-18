"""Phase A: per-seed 3-round probe + 1 attribution LLM call.

For each seed factor f:
  - Run ROUNDS rounds; each round, LLM proposes K_PER_ROUND minimal one-site edits.
  - recent arg to propose() is the previous round's records only, so the LLM is
    guided to cover new (kind, site_id) pairs in rounds 2 and 3.
  - Each edit is applied to the FROZEN ORIGINAL f (not the previous-round f').
  - We record per edit: inc_residual, ic_f, ic_fp, delta_ic, f'.
  - After ROUNDS rounds, ONE attribution LLM call gets all the records and
    outputs {necessary, redundant, final_factor, reason}. final_factor is
    free-form; we parse and fall back to f on failure.

Key invariants
--------------
- Edit base = the FROZEN ORIGINAL seed. We never compound edits across rounds,
  so `inc(f', f)` always measures "f' vs the seed", not "f' vs the previous f'".
- Records are flushed per-seed so a crashed run still has partial outputs.
- One seed at a time → RewardCache reuses the alpha tensor across rounds.
"""
import json
import sys
from pathlib import Path
from typing import List, Sequence

if __package__ is None:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from alphagen.data.expression import Expression

from alpha_cf import llm
from alpha_cf.config import K_PER_ROUND, N_SEEDS, ROUNDS
from alpha_cf.edits import apply_edit
from alpha_cf.pool import parse_expr
from alpha_cf.reward import RewardCache
from alpha_cf.types import CFRecord, EditAction


# Filter propose()'s output to actions whose T(f, a) produces a featured f' != f
# and not equal to the original f. Also dedupes LLM retries that produce the
# same f' twice in one round.
def _legal_edits(f: Expression, actions: Sequence[EditAction]) -> List[tuple]:
    out = []
    seen = {str(f)}  # include the seed so a no-op edit gets dropped
    for a in actions:
        try:
            fp = apply_edit(f, a)
        except Exception:
            # apply_edit should never raise for valid EDIT_KINDS, but be defensive.
            fp = None
        if fp is None:
            llm.record_illegal("phaseA", f, a)
            continue
        key = str(fp)
        if key == str(f) or key in seen:
            # f' == f means the edit was a no-op (e.g. window=current window).
            print(f"[phaseA] skip no-op / dup  {fp}", flush=True)
            continue
        seen.add(key)
        out.append((a, fp))
    return out


# One record-dict per (round, action) where apply_edit succeeded.
# inc_residual is the key signal for redundancy — listed first so LLM sees it
# before the supporting ic_*/delta_ic context.
def _build_record(round_idx: int, action: EditAction, fp: Expression,
                  f_eval, fp_eval, inc: float) -> dict:
    return {
        "round": round_idx,
        "kind": action.kind,
        "site_id": action.site_id,
        "new_value": action.new_value,
        "hypothesis": action.hypothesis,
        "inc_residual": inc,
        "ic_f": f_eval.ic,
        "ic_fp": fp_eval.ic,
        "delta_ic": fp_eval.ic - f_eval.ic,
        "f_prime": str(fp),
    }


# Per-seed: 3 rounds of K edits + 1 attribution call.
def _process_one_seed(seed: Expression, cache: RewardCache) -> tuple:
    # Evaluate the seed ONCE; reused across all rounds (cache makes this cheap).
    f_eval = cache.evaluate(seed)
    records: List[dict] = []

    for round_idx in range(ROUNDS):
        # recent = only the previous round's records, so LLM is guided to cover
        # new (kind, site_id) pairs across rounds 1 and 2.
        if round_idx == 0:
            recent = None  # first round: no prior context
        else:
            # Filter to exactly the previous round (round_idx - 1) and rehydrate
            # back into CFRecord so llm.propose() can format them.
            recent_recs = [
                CFRecord(
                    f=str(seed),
                    action=EditAction(rec["kind"], rec["site_id"], rec["new_value"],
                                       rec.get("hypothesis")),
                    f_prime=rec["f_prime"],
                    delta=rec["delta_ic"],
                    ic=rec["ic_fp"],
                    rank_ic=rec["ic_fp"],  # ric dropped from records; fall back to Pearson IC
                    r=rec["ic_fp"],
                    r_seed=rec["ic_f"],
                    inc=rec["inc_residual"],
                    extra={"round": round_idx - 1},  # let llm._last_n_rounds filter
                )
                for rec in records if rec["round"] == round_idx - 1
            ]
            recent = recent_recs
        try:
            actions = llm.propose(seed, k=K_PER_ROUND, recent=recent)
        except llm.LLMError as e:
            # Network / API failure shouldn't kill the whole seed; continue empty.
            print(f"[phaseA] propose round {round_idx} failed: {e}", flush=True)
            actions = []
        legal = _legal_edits(seed, actions)
        # Each legal (a, fp) → evaluate f' + compute inc(f', seed) → record.
        for action, fp in legal:
            fp_eval = cache.evaluate(fp)
            inc = cache.inc(fp, seed)
            records.append(_build_record(round_idx, action, fp, f_eval, fp_eval, inc))
            # Echo per-record metrics to stdout so the operator can see each
            # LLM hypothesis being tested in real time.
            print(
                f"[phaseA]   {action.kind}@{action.site_id}={action.new_value!r}  "
                f"f'={fp}  "
                f"ic_f={f_eval.ic:+.4f}  ic_fp={fp_eval.ic:+.4f}  "
                f"delta={fp_eval.ic - f_eval.ic:+.4f}  "
                f"inc={inc:+.4f}",
                flush=True,
            )
        print(
            f"[phaseA] seed={seed}  round={round_idx}  "
            f"proposed={len(actions)}  legal={len(legal)}  "
            f"running_records={len(records)}",
            flush=True,
        )

    # After ROUNDS rounds, ask LLM to attribute the whole trace to one trimmed f.
    attr = llm.attribute(seed, records)
    return seed, records, attr


def _safe_parse_factor(s: str) -> Expression:
    expr = parse_expr(s)
    if not getattr(expr, "is_featured", True):
        raise ValueError(f"unfeatured expression: {expr}")
    return expr


# seeds, cache, out_dir -> writes per-seed records + attribution; yields trimmed Expression.
#
# Output files (jsonl, append-mode; flushed per-seed):
#   <out_dir>/phaseA_records.jsonl     one row per legal edit
#   <out_dir>/phaseA_attribution.jsonl one row per seed (attr dict)
#   <out_dir>/illegal.jsonl            edits rejected by llm.record_illegal()
#   <out_dir>/trimmed_pool.json        final trimmed pool (one row: {exprs, weights})
def run_phaseA(seeds: Sequence[Expression], cache: RewardCache, out_dir: str) -> List[Expression]:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    llm.set_illegal_log(str(out / "illegal.jsonl"))
    rec_path = out / "phaseA_records.jsonl"
    attr_path = out / "phaseA_attribution.jsonl"
    rec_w = open(rec_path, "a", encoding="utf-8")
    attr_w = open(attr_path, "a", encoding="utf-8")

    trimmed: List[Expression] = []
    n_seeds = len(seeds)
    try:
        for i, seed in enumerate(seeds):
            print(f"[phaseA] ===== seed {i+1}/{n_seeds}  f={seed} =====", flush=True)
            seed_f, records, attr = _process_one_seed(seed, cache)
            # Flush records immediately so a crash mid-run still leaves a usable
            # records.jsonl for replay/debug.
            for rec in records:
                rec_w.write(json.dumps(rec, ensure_ascii=False) + "\n")
            rec_w.flush()
            attr_w.write(json.dumps(
                {
                    "f": str(seed_f),
                    "necessary": attr.get("necessary", ""),
                    "redundant": attr.get("redundant", ""),
                    "final_factor": attr.get("final_factor", str(seed_f)),
                    "reason": attr.get("reason", ""),
                },
                ensure_ascii=False,
            ) + "\n")
            attr_w.flush()
            # LLM may return syntactically broken final_factor — fall back to seed.
            ff_str = attr.get("final_factor") or str(seed_f)
            try:
                f_final = _safe_parse_factor(ff_str)
                trimmed.append(f_final)
                print(
                    f"[phaseA] seed {i+1} DONE  records={len(records)}  "
                    f"f_final={f_final}  "
                    f"({f_final != seed_f} shrink or no-op)",
                    flush=True,
                )
            except Exception as e:
                print(f"[phaseA] seed {i+1} parse fail: {e}; keep original", flush=True)
                trimmed.append(seed_f)
    finally:
        # Always close, even on Ctrl-C / crash mid-loop.
        rec_w.close()
        attr_w.close()

    # Persist the trimmed pool so callers can hand it straight to phaseB.py
    # or run_adaptive_combination.py without re-running Phase A.
    # `weights: []` is required by combo's load_alpha_pool() even when unused.
    trimmed_path = out / "trimmed_pool.json"
    with open(trimmed_path, "w", encoding="utf-8") as w:
        json.dump({"exprs": [str(f) for f in trimmed], "weights": []}, w, indent=2)
    print(f"[phaseA] trimmed_pool  ->  {trimmed_path}  n={len(trimmed)}", flush=True)
    return trimmed


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("--pool_json", required=True)
    p.add_argument("--n_seeds", type=int, default=N_SEEDS)
    p.add_argument("--seed_start", type=int, default=0)
    p.add_argument("--instrument", type=str, default="csi300")
    p.add_argument("--out_dir", type=str, default="")
    args = p.parse_args()

    from datetime import datetime
    from alpha_cf.pool import load_pool
    from alpha_cf.reward import make_calculator

    if not args.out_dir:
        args.out_dir = f"data/cf_logs/phaseA_{datetime.now():%Y%m%d_%H%M%S}"
    need = args.seed_start + args.n_seeds
    seeds = load_pool(args.pool_json, max_n=need)[args.seed_start:]
    # make_calculator defaults to EVAL_SPLIT="train" (set in reward.py).
    cache = RewardCache(make_calculator(args.instrument))
    trimmed = run_phaseA(seeds, cache, args.out_dir)
    print(f"[phaseA] DONE  out_dir={args.out_dir}  n_trimmed={len(trimmed)}", flush=True)