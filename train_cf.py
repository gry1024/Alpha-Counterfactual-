"""Counterfactual factor evolution; run from the repository root."""
import argparse
from datetime import datetime
import gc
import json
import os
from pathlib import Path
import sys
import subprocess

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
import numpy as np
import pandas as pd
import torch
from dotenv import load_dotenv
from alphagen_qlib.stock_data import StockData
from alpha_cf.alpha_pool import AlphaCFPool
from alpha_cf.trainer import AlphaCFTrainer, save_json

SPLITS = {"train": ("2015-01-01", "2021-12-31")}


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
    if json.loads(source.read_text()).get("split") != "train":
        raise ValueError("Test requires a pool_*.json from training")
    root = Path(__file__).resolve().parent
    metrics_path = source.with_suffix(".metrics.json")
    env = dict(os.environ, PYTHONPATH=os.pathsep.join([str(root / "src"), os.environ.get("PYTHONPATH", "")]))
    subprocess.run([sys.executable, str(root / "run_adaptive_combination.py"),
        "--expressions_file", str(source.resolve()), "--instruments", args.instrument,
        "--train_end_year", "2021", "--label_days", str(args.horizon), "--cuda", str(args.cuda),
        "--seed", str(args.seed), "--n_factors", str(args.n_factors), "--chunk_size", str(args.chunk_size),
        "--use_weights", "True", "--metrics_file", str(metrics_path.resolve())],
        cwd=root, env=env, check=True)
    round_id = int(source.stem.removeprefix("pool_")) if source.stem.startswith("pool_") else args.rounds
    result_path = source.parent / "res.txt"
    with result_path.open("a", encoding="utf-8") as file:
        for dataset, values in json.loads(metrics_path.read_text()).items():
            file.write(json.dumps(dict(round=round_id, dataset=dataset, **values)) + "\n")
    from matplotlib.figure import Figure
    history = pd.read_json(result_path, lines=True)
    figure = Figure(figsize=(10, 4), layout="constrained")
    for axis, metric in zip(figure.subplots(1, 2), ("ic", "ret")):
        for dataset, rows in history.groupby("dataset"):
            rows = rows.drop_duplicates("round", keep="last").sort_values("round")
            axis.plot(rows["round"], rows[metric], marker="o", label=dataset)
        axis.set(xlabel="Round", ylabel=metric.upper())
        axis.grid(alpha=0.3)
        axis.legend()
    figure.savefig(source.parent / "adaptive_combination.png", dpi=150)


def train(args):
    model_name = os.environ.get("OPENAI_MODEL_NAME", "unknown").replace("/", "_").replace("\\", "_")
    log_dir = Path("data/cf_logs") / f"{datetime.now():%Y%m%d_%H%M%S_%f}_{args.instrument}_{args.seed}_{model_name}"
    log_dir.mkdir(parents=True)
    save_json(log_dir / "args.json", vars(args))
    print(f"Logs: {log_dir}", flush=True)
    data, target = load_data(args, "train")
    pool = AlphaCFPool(data, target, args)
    trainer = AlphaCFTrainer(pool, args, log_dir)
    seeds = json.loads(Path(args.start_pool).read_text())["exprs"]
    trainer.train(seeds, on_round=lambda step: test(args, log_dir / f"pool_{step}.json"))
    del trainer, data, target
    release()
    del pool
    release()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--instrument", default="csi300")
    parser.add_argument("--qlib-path")
    parser.add_argument("--start-pool", default="start_pool.json")
    parser.add_argument("--test-only", metavar="FINAL_JSON")
    for name, default in dict(seed=0, cuda=0, rounds=10, parents=5,
                              pool_capacity=50, horizon=20,
                              max_nodes=60, max_depth=10, max_backtrack=100, chunk_size=64,
                              llm_timeout=600, llm_attempts=5, llm_retry_wait=5, n_factors=20, diagnosis_workers=5).items():
        parser.add_argument("--" + name.replace("_", "-"), type=int, default=default, dest=name)
    for name, default in dict(alpha=50.0, beta=800.0, gamma=2, cost_weight=2, temperature=0.5, correlation_threshold=0.8).items():
        parser.add_argument("--" + name.replace("_", "-"), type=float, default=default, dest=name)
    for name in ("no-cf-evidence", "no-pool-credit", "no-memory"):
        parser.add_argument("--" + name, action="store_true")
    args = parser.parse_args()
    source = args.test_only
    if source:
        saved = json.loads(Path(source).read_text())
        config = saved.get("args") or json.loads((Path(source).parent / "args.json").read_text())
        config.update(cuda=args.cuda, test_only=args.test_only, qlib_path=args.qlib_path)
        args = argparse.Namespace(**{k: config.get(k, v) for k, v in vars(args).items()})
    if args.rounds < 0 or min(args.parents,
            args.pool_capacity, args.horizon, args.chunk_size, args.n_factors, args.diagnosis_workers, args.llm_timeout, args.llm_attempts, args.llm_retry_wait) < 1:
        parser.error("Invalid counts")
    if not 0 <= args.correlation_threshold <= 1:
        parser.error("correlation-threshold must be between 0 and 1")
    load_dotenv(Path(__file__).resolve().parent / ".env")
    args.device = f"cuda:{args.cuda}" if args.cuda >= 0 and torch.cuda.is_available() else "cpu"
    env_key = "QLIB_PATH_SP500" if args.instrument == "sp500" else "QLIB_PATH_CN"
    default = "data/qlib_data/us_data_qlib_latest" if args.instrument == "sp500" else "data/qlib_data/cn_data_rolling"
    args.qlib_path = str(Path(args.qlib_path or os.environ.get(env_key) or default).resolve())
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.test_only:
        test(args, Path(args.test_only).resolve())
    else:
        train(args)


if __name__ == "__main__":
    main()
