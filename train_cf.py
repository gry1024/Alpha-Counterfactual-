"""Phase A + Phase B + auto-combo (research.md workflow, simplified)."""
import argparse
import json
import os
import random
import subprocess
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from alpha_cf.config import EVAL_SPLIT, N_SEEDS
from alpha_cf.phaseA import run_phaseA
from alpha_cf.phaseB import run_phaseB
from alpha_cf.pool import load_pool
from alpha_cf.reward import RewardCache, make_calculator


# Combo arg vector used for both auto-combo calls (Phase A & Phase A+B).
# Thresholds chosen so a 20-factor ensemble clears the gate on csi300; tune
# per-instrument if reusing for sp500 / csi500.
COMBO_ARGS = [
    "--train_end_year", "2021",
    "--instruments", "csi300",
    "--n_factors", "20",
    "--threshold_ric", "0.015",
    "--threshold_ricir", "0.15",
]


# subprocess combo; never raises — failures only print + leave the metrics file empty.
# 3600s hard timeout because combo does corr-filtered ensemble learning per factor.
def _run_combo(pool_json: str, out_txt: str) -> None:
    cmd = ["python", "run_adaptive_combination.py",
           "--expressions_file", pool_json, *COMBO_ARGS]
    print(f"[combo] {' '.join(cmd)}", flush=True)
    try:
        r = subprocess.run(cmd, capture_output=True, text=True,
                           cwd=str(Path(__file__).resolve().parent),
                           timeout=3600)
    except subprocess.TimeoutExpired:
        print(f"[combo] TIMEOUT after 3600s", flush=True)
        with open(out_txt, "w") as w:
            w.write("combo: TIMEOUT\n")
        return
    except Exception as e:
        # Spawn failure (python missing, IO error) — write placeholder so the
        # caller can still inspect why combo didn't produce real metrics.
        print(f"[combo] spawn failed: {e}", flush=True)
        with open(out_txt, "w") as w:
            w.write(f"combo spawn failed: {e}\n")
        return
    with open(out_txt, "w", encoding="utf-8") as w:
        w.write(r.stdout or "")
        if r.stderr:
            w.write("\n--- STDERR ---\n" + r.stderr)
        w.write(f"\n--- exit={r.returncode} ---\n")
    print(f"[combo] exit={r.returncode}  ->  {out_txt}", flush=True)


def _setup_out_dir(args) -> str:
    if not args.out_dir:
        args.out_dir = os.path.join(
            "data", "cf_logs", f"run_{datetime.now():%Y%m%d_%H%M%S}"
        )
    os.makedirs(args.out_dir, exist_ok=True)
    return args.out_dir


def train(args) -> str:
    out_dir = _setup_out_dir(args)
    print(f"[cf] out_dir={out_dir}", flush=True)
    random.seed(args.seed)

    if not args.pool_json:
        raise SystemExit(
            "need --pool_json (Phase A entry); Phase B alone is unsupported, "
            "use src/alpha_cf/phaseB.py --trimmed_pool_json instead"
        )

    # ---------------------------------------------------------------- Phase A
    # Slice the pool by [seed_start : seed_start + n_seeds] so concurrent runs
    # can split work without re-loading the full pool file.
    need = args.seed_start + args.n_seeds
    seeds = load_pool(args.pool_json, max_n=need)[args.seed_start:]
    if not seeds:
        raise SystemExit(
            f"--seed_start={args.seed_start} past pool "
            f"({need - args.seed_start} unique exprs in {args.pool_json})"
        )
    print(f"[cf] Phase A: {len(seeds)} seeds from {args.pool_json}", flush=True)
    cache = RewardCache(make_calculator(args.instrument, EVAL_SPLIT))
    trimmed = run_phaseA(seeds, cache, out_dir)
    trimmed_pool_json = os.path.join(out_dir, "trimmed_pool.json")
    print(f"[cf] Phase A  ->  {trimmed_pool_json}  n={len(trimmed)}", flush=True)

    # auto-combo #1: evaluate the Phase A output BEFORE Phase B, so we can
    # measure how much value Phase B adds on top.
    _run_combo(trimmed_pool_json, os.path.join(out_dir, "phaseA_combo.txt"))

    # ---------------------------------------------------------------- Phase B
    # --skip_phaseB: ablation — copy trimmed_pool as final_pool without breeding,
    # so phaseA_combo vs final_combo isolates the Phase B contribution.
    if args.skip_phaseB:
        print("[cf] --skip_phaseB  skipping recombination; copying trimmed as final", flush=True)
        final_pool_json = os.path.join(out_dir, "final_pool.json")
        with open(final_pool_json, "w", encoding="utf-8") as w:
            json.dump({"exprs": [str(f) for f in trimmed], "weights": []},
                      w, indent=2)
    else:
        run_phaseB(trimmed, cache, out_dir)
        final_pool_json = os.path.join(out_dir, "final_pool.json")

    # auto-combo #2: final evaluation (Phase A + Phase B).
    _run_combo(final_pool_json, os.path.join(out_dir, "final_combo.txt"))

    print(f"[cf] DONE  out_dir={out_dir}", flush=True)
    return out_dir


if __name__ == "__main__":
    p = argparse.ArgumentParser(
        description="Phase A (counterfactual slimming) + auto-combo + "
                    "Phase B (module recombination) + auto-combo."
    )
    p.add_argument("--pool_json", type=str, default="",
                   help="Seed pool JSON. Phase A entry; required.")
    p.add_argument("--n_seeds", type=int, default=N_SEEDS,
                   help="How many seeds to slim in Phase A.")
    p.add_argument("--seed_start", type=int, default=0)
    p.add_argument("--instrument", type=str, default="csi300")
    p.add_argument("--out_dir", type=str, default="")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--skip_phaseB", action="store_true",
                   help="Skip recombination; copy trimmed_pool.json as final_pool.json.")
    args = p.parse_args()
    print(f"[cf] args={vars(args)}", flush=True)
    train(args)