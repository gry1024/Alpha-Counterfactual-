"""Counterfactual factor evolution; run from the repository root."""
import argparse
from datetime import datetime
import gc
import json
import os
from pathlib import Path
import random
import sys
import subprocess

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
import numpy as np
import pandas as pd
import torch
from dotenv import load_dotenv
from alphagen_qlib.stock_data import StockData
from alpha_cf.alpha_pool import AlphaCFPool
from alpha_cf.expression import parse
from alpha_cf.trainer import AlphaCFTrainer, save_json

SPLITS = {"train": ("2011-01-01", "2021-12-31"), "valid": ("2022-01-01", "2022-12-31")}


def load_data(args, split):
    start, end = SPLITS[split]
    calendar = pd.DatetimeIndex(pd.read_csv(Path(args.qlib_path) / "calendars/day.txt", header=None)[0])
    left = calendar.searchsorted(pd.Timestamp(start))
    right = calendar.searchsorted(pd.Timestamp(end), side="right")
    if left < args.max_backtrack or calendar[-1] < pd.Timestamp(end):
        raise ValueError(f"Dataset does not cover {split} plus history")
    data = StockData(instrument=args.instrument, start_time=str(calendar[left].date()),
                     end_time=str(calendar[right - 1].date()), max_backtrack_days=args.max_backtrack,
                     max_future_days=0, device=torch.device(args.device), qlib_path=args.qlib_path)
    if not pd.DatetimeIndex(data._dates).equals(calendar[left - args.max_backtrack:right]):
        raise ValueError("Loaded dates do not match the trading calendar")
    data.df_bak = None
    close = data.data[data.max_backtrack_days:, 1, :]
    target = close[args.horizon:] / close[:-args.horizon] - 1
    target = target.masked_fill(~target.isfinite() | (close[args.horizon:] <= 0) | (close[:-args.horizon] <= 0), torch.nan)
    print(f"{split}: {len(target)} labelled days, {data.n_stocks} stocks", flush=True)
    return data, target


def release():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def test(args, source):
    if json.loads(source.read_text()).get("split") != "valid":
        raise ValueError("Test requires a frozen final.json from Validation")
    root = Path(__file__).resolve().parent
    env = dict(os.environ, PYTHONPATH=os.pathsep.join([str(root / "src"), os.environ.get("PYTHONPATH", "")]))
    subprocess.run([sys.executable, str(root / "run_adaptive_combination.py"),
        "--expressions_file", str(source.resolve()), "--instruments", args.instrument,
        "--train_end_year", "2021", "--label_days", str(args.horizon), "--cuda", str(args.cuda),
        "--seed", str(args.seed), "--n_factors", str(args.n_factors), "--chunk_size", str(args.chunk_size)],
        cwd=root, env=env, check=True)


def train(args):
    log_dir = Path("data/cf_logs") / f"{datetime.now():%Y%m%d_%H%M%S_%f}_{args.instrument}_{args.seed}"
    log_dir.mkdir(parents=True)
    save_json(log_dir / "args.json", vars(args))
    print(f"Logs: {log_dir}", flush=True)
    if args.finalize_only:
        expressions = [parse(text, args) for text in json.loads(Path(args.finalize_only).read_text())["exprs"]]
    else:
        data, target = load_data(args, "train")
        pool = AlphaCFPool(data, target, args)
        trainer = AlphaCFTrainer(pool, args, log_dir)
        seeds = json.loads(Path(args.start_pool).read_text())["exprs"]
        expressions = trainer.train(seeds)
        del trainer, pool, data, target
        release()
    save_json(log_dir / "search_pool.json", dict(exprs=[str(e) for e in expressions], args=vars(args), split="train"))
    data, target = load_data(args, "valid")
    pool = AlphaCFPool(data, target, args)
    candidates = []
    for expr in expressions:
        try:
            pool.evaluate(expr)
            candidates.append(expr)
        except ValueError as exc:
            print(f"Validation unavailable: {expr}: {exc}", flush=True)
    pool.exprs = pool.select(candidates, args.final_size)
    save_json(log_dir / "final.json", dict(**pool.to_dict(), args=vars(args), split="valid"))
    for i, (expr, metrics) in enumerate(zip(pool.exprs, pool.to_dict()["metrics"])):
        print(f"> Alpha {i + 1}: RankICIR={metrics['reward']:.4f}, expr={expr}", flush=True)
    del pool, data, target
    release()
    test(args, log_dir / "final.json")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--instrument", default="csi300")
    parser.add_argument("--qlib-path")
    parser.add_argument("--start-pool", default="start_pool.json")
    parser.add_argument("--test-only", metavar="FINAL_JSON")
    parser.add_argument("--finalize-only", metavar="TRAIN_POOL_JSON")
    for name, default in dict(seed=0, cuda=0, rounds=10, parents=20, mechanisms=3, offspring=5,
                              refine_top_k=5, pool_capacity=60, final_size=30, horizon=20,
                              max_nodes=60, max_depth=10, max_backtrack=100, chunk_size=64,
                              llm_timeout=180, n_factors=10).items():
        parser.add_argument("--" + name.replace("_", "-"), type=int, default=default, dest=name)
    for name, default in dict(alpha=1.0, beta=1.0, gamma=0.2, cost_weight=0.1, temperature=0.5).items():
        parser.add_argument("--" + name.replace("_", "-"), type=float, default=default, dest=name)
    parser.add_argument("--windows", type=int, nargs="+", default=[5, 10, 20, 40, 60])
    for name in ("no-cf-evidence", "no-pool-credit", "random-crossover", "no-memory", "no-pool-selection", "no-refinement"):
        parser.add_argument("--" + name, action="store_true")
    args = parser.parse_args()
    if args.test_only and args.finalize_only:
        parser.error("Use only one replay mode")
    source = args.test_only or args.finalize_only
    if source:
        saved = json.loads(Path(source).read_text())
        if args.finalize_only and saved.get("split") in ("valid", "test"):
            parser.error("--finalize-only requires a Train pool")
        config = saved.get("args") or json.loads((Path(source).parent / "args.json").read_text())
        config.update(cuda=args.cuda, test_only=args.test_only, finalize_only=args.finalize_only)
        args = argparse.Namespace(**{k: config.get(k, v) for k, v in vars(args).items()})
    if min(args.rounds, args.refine_top_k) < 0 or min(args.parents, args.mechanisms, args.offspring,
            args.pool_capacity, args.final_size, args.horizon, args.chunk_size, args.n_factors) < 1:
        parser.error("Invalid counts")
    if args.final_size > args.pool_capacity or min(args.windows) < 2:
        parser.error("Invalid pool sizes or window grid")
    load_dotenv(Path(__file__).resolve().parent / ".env")
    args.device = f"cuda:{args.cuda}" if args.cuda >= 0 and torch.cuda.is_available() else "cpu"
    env_key = "QLIB_PATH_SP500" if args.instrument == "sp500" else "QLIB_PATH_CN"
    default = "data/qlib_data/us_data_qlib_latest" if args.instrument == "sp500" else "data/qlib_data/cn_data_rolling"
    args.qlib_path = str(Path(args.qlib_path or os.environ.get(env_key) or default).resolve())
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.rounds and not source and os.environ.get("OPENAI_MODEL_NAME", "").lower() != "minimax-m3":
        parser.error("Set OPENAI_MODEL_NAME=MiniMax-M3 in .env")
    if args.test_only:
        test(args, Path(args.test_only).resolve())
    else:
        train(args)


if __name__ == "__main__":
    main()
