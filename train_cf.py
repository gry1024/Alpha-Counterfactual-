import argparse
import json
import os
import random
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from alpha_cf.config import K, MAX_ROUNDS, MIN_Q_SAMPLES, N_SEEDS, PATIENCE, TOPK
from alpha_cf.credit import write_credit
from alpha_cf.edits import random_propose
from alpha_cf.llm import propose as llm_propose
from alpha_cf.llm import set_illegal_log
from alpha_cf.pool import load_pool
from alpha_cf.q_model import make_rank_fn
from alpha_cf.reward import RewardCache, make_calculator
from alpha_cf.search import search_one

_PROPOSE = {"llm": llm_propose, "random": random_propose}  # CLI name -> propose_fn


def _pool_dict(exprs):
    # dummy weights: combination eval uses use_weight=False
    return {"exprs": list(exprs), "weights": [1.0] * len(exprs)}


def _upsert_expr(exprs, expr):
    return [e for e in exprs if e != expr] + [expr]  # last wins if identical


def _write_pool_json(path, exprs):
    tmp = path + ".tmp"
    with open(tmp, "w") as w:
        json.dump(_pool_dict(exprs), w, indent=2, ensure_ascii=False)
    os.replace(tmp, path)


# CLI args -> search_one over seeds; writes cf_records.jsonl + summary.json + credit.json + pool.json
def train(args):
    random.seed(args.seed)
    os.makedirs(args.out_dir, exist_ok=True)
    print(f"[cf] out_dir={args.out_dir}", flush=True)
    illegal_path = os.path.join(args.out_dir, "illegal.jsonl")
    set_illegal_log(illegal_path)
    print(f"[cf] illegal log={illegal_path}", flush=True)
    seeds = load_pool(args.pool_json, max_n=args.n_seeds)
    print(f"[cf] loaded {len(seeds)} seeds from {args.pool_json}")
    for i, s in enumerate(seeds):
        print(f"[cf]   seed[{i}] {s}")

    print(f"[cf] loading qlib  instrument={args.instrument}  split=train ...")
    cache = RewardCache(make_calculator(args.instrument, "train"))
    print("[cf] qlib ready")
    all_recs = []
    pool_exprs = []
    pool_path = os.path.join(args.out_dir, "pool.json")
    ranker = None if args.no_q else make_rank_fn(all_recs)
    print(f"[cf] Q={'off (--no_q)' if args.no_q else 'cold-start until n_cf>=' + str(MIN_Q_SAMPLES)}")
    for i, f in enumerate(seeds):
        # share one Q across seeds; cold-start until MIN_Q_SAMPLES
        use_q = ranker is not None and len(all_recs) >= MIN_Q_SAMPLES
        print(
            f"[cf] ---- seed {i}/{len(seeds)} START  f={f}  "
            f"q={'on' if use_q else 'off'}  n_cf={len(all_recs)} ----"
        )
        recs, f_final = search_one(
            f,
            cache,
            propose_fn=_PROPOSE[args.propose],
            k=args.k,
            topk=args.topk,
            max_rounds=args.max_rounds,
            patience=args.patience,
            min_q_samples=0 if use_q else MIN_Q_SAMPLES,
            rank_fn=ranker if use_q else None,
            seed_idx=i,
        )
        all_recs.extend(recs)
        pool_exprs = _upsert_expr(pool_exprs, str(f_final))
        _write_pool_json(pool_path, pool_exprs)  # rewrite after each finished seed
        if recs:
            best = max(recs, key=lambda r: r.delta)
            print(
                f"[cf] seed {i} END  final={f_final}  n_recs={len(recs)}  "
                f"pool_n={len(pool_exprs)}  best Δ={best.delta:+.4f}",
                flush=True,
            )
        else:
            print(f"[cf] seed {i} END  final={f_final}  n_recs=0  pool_n={len(pool_exprs)}", flush=True)

    rec_path = os.path.join(args.out_dir, "cf_records.jsonl")
    with open(rec_path, "w") as w:
        for rec in all_recs:
            w.write(json.dumps(rec.to_dict(), ensure_ascii=False) + "\n")

    # compact run summary next to the jsonl
    deltas = [r.delta for r in all_recs]
    summary = {
        "pool_json": args.pool_json,
        "instrument": args.instrument,
        "n_seeds": len(seeds),
        "n_records": len(all_recs),
        "n_eval": cache.n_eval,
        "n_illegal": sum(1 for _ in open(illegal_path)) if os.path.isfile(illegal_path) else 0,
        "propose": args.propose,
        "no_q": args.no_q,
        "min_q_samples": MIN_Q_SAMPLES,
        "max_delta": max(deltas) if deltas else None,
        "mean_delta": sum(deltas) / len(deltas) if deltas else None,
        "best_f": str(max(all_recs, key=lambda r: r.r).f_prime) if all_recs else None,
    }
    sum_path = os.path.join(args.out_dir, "summary.json")
    with open(sum_path, "w") as w:
        json.dump(summary, w, indent=2, ensure_ascii=False)
    credit_path = os.path.join(args.out_dir, "credit.json")
    write_credit(all_recs, credit_path)
    print(f"[cf] wrote {rec_path}  ({len(all_recs)} records)", flush=True)
    print(f"[cf] wrote {sum_path}", flush=True)
    print(f"[cf] wrote {credit_path}", flush=True)
    print(f"[cf] wrote {pool_path}", flush=True)
    if os.path.isfile(illegal_path):
        print(f"[cf] wrote {illegal_path}  ({summary['n_illegal']} illegal skips)", flush=True)
    else:
        print("[cf] no illegal edits this run", flush=True)
    print(summary, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pool_json", type=str, required=True, help="AlphaSAGE pool JSON of seed expressions")
    parser.add_argument("--n_seeds", type=int, default=N_SEEDS, help="How many seed factors to search")
    parser.add_argument("--instrument", type=str, default="csi300", help="Market universe (csi300 or sp500)")
    parser.add_argument("--out_dir", type=str, default="", help="Log dir; default data/cf_logs/<timestamp>")
    parser.add_argument("--propose", type=str, default="llm", choices=list(_PROPOSE), help="Who proposes edits: llm or random")
    parser.add_argument("--seed", type=int, default=0, help="RNG seed")
    parser.add_argument("--max_rounds", type=int, default=MAX_ROUNDS, help="Max greedy edit rounds per seed")
    parser.add_argument("--k", type=int, default=K, help="Edits proposed each round")
    parser.add_argument("--topk", type=int, default=TOPK, help="How many Q-ranked edits to backtest")
    parser.add_argument("--patience", type=int, default=PATIENCE, help="Stop after this many rounds with no Δ>0")
    parser.add_argument("--no_q", action="store_true", help="Backtest every proposal; do not train or use Q")
    args = parser.parse_args()
    if not args.out_dir:
        args.out_dir = os.path.join(
            "data", "cf_logs", datetime.now().strftime("%Y%m%d_%H%M%S")
        )  # default: timestamped log dir
    print(f"[cf] args={args}", flush=True)
    train(args)
