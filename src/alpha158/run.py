"""Evaluate Alpha158 with fixed, training-directed equal weights."""
import argparse
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import qlib
import torch

import run_adaptive_combination as backtest
from alpha158.features import load_tensor
from alpha158.ranking import batch_spearmanr
from alphagen.data.expression import Feature, Ref
from alphagen_qlib.stock_data import FeatureType, StockData


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--instruments", choices=["csi300", "csi500", "csi1000"], default="csi1000")
    parser.add_argument("--feature-cache", help="Existing normalized Alpha158 cache; otherwise generate it.")
    parser.add_argument("--output-dir")
    parser.add_argument("--threads", type=int, default=2)
    args = parser.parse_args()
    if args.threads < 1:
        parser.error("threads must be positive")
    root = Path(__file__).resolve().parents[2]
    output = Path(args.output_dir or root / "data/alpha158_logs" /
                  (args.instruments + "_equal_weight")).resolve()
    output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(args.threads)
    qlib.init(provider_uri=str(root / "data/qlib_data/cn_data_rolling"), region="cn", kernels=1)
    StockData._qlib_initialized = True
    data = StockData(args.instruments, "2015-01-01", "2026-04-30", device=torch.device("cpu"))
    dates = backtest._trading_days(data)
    cache = Path(args.feature_cache).resolve() if args.feature_cache else output / "features.npy"
    if not args.feature_cache:
        load_tensor(data, cache_file=cache)
    source = np.load(cache, mmap_mode="r")
    if source.shape != (158, len(dates), data.n_stocks):
        raise ValueError("Feature cache axes do not match this dataset")
    close = Feature(FeatureType.CLOSE)
    target = (Ref(close, -20) / close - 1).evaluate(data).numpy()
    daily = (Ref(close, -1) / close - 1).evaluate(data).numpy()
    target[~np.isfinite(target)] = np.nan
    daily[~np.isfinite(daily)] = np.nan
    train = (dates >= "2015-01-01") & (dates <= "2021-12-31")
    maturity = np.zeros(len(dates), dtype=bool)
    maturity[:-20] = dates[20:] <= pd.Timestamp("2021-12-31")
    train &= maturity
    labels = pd.DataFrame(target[train])
    normalized = labels.sub(labels.mean(axis=1), axis=0).div(labels.std(axis=1), axis=0)
    normalized = np.clip(np.nan_to_num(normalized.to_numpy(), nan=0., posinf=0., neginf=0.), -4, 4)
    mask = np.isfinite(target[train])
    y = normalized[mask].astype(np.float64)
    y -= y.mean()
    weights = np.empty(158)
    for i in range(158):
        x = source[i, train][mask].astype(np.float64)
        weights[i] = (-1. if (x - x.mean()) @ y < 0 else 1.) / 158
    np.save(output / "weights.npy", weights)
    with cache.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    protocol = dict(args=vars(args), combination="fixed training-directed equal weights, absolute value 1/158",
        feature_cache=str(cache), feature_sha256=digest, stocks=data._stock_ids.tolist(),
        direction="Train covariance with daily standardized and clipped 20-day returns",
        train="2015-2021; 20-day labels matured by 2021-12-31",
        validation="2022", test="2023-05-01..2026-04-30",
        evaluation="raw 20-day IC/RIC; top 20% stocks, next-day return; zero costs",
        reporting="main Alpha158 baseline by author request; protocol revised after inspecting alternative Test results")
    (output / "alpha158.json").write_text(json.dumps(protocol, indent=2))
    backtest.args = SimpleNamespace(chunk_size=64)
    backtest.batch_spearmanr = batch_spearmanr
    metrics = {}
    for split, start, end in (("Validation", "2022-01-01", "2022-12-31"),
                              ("Test", "2023-05-01", "2026-04-30")):
        select = (dates >= start) & (dates <= end)
        indices = np.flatnonzero(select)
        pred = np.empty((len(indices), data.n_stocks), dtype=np.float32)
        for offset in range(0, len(indices), 8):
            ix = indices[offset:offset + 8]
            pred[offset:offset + len(ix)] = source[:, ix].transpose(1, 2, 0) @ weights
        metrics[split], returns = backtest.get_tensor_metrics(torch.from_numpy(pred),
            torch.from_numpy(target[select]), torch.from_numpy(daily[select]))
        prefix = "" if split == "Test" else "valid_"
        np.save(output / (prefix + "predictions.npy"), pred)
        np.save(output / (prefix + "ret_s.npy"), returns.numpy())
        np.save(output / (prefix + "ret_s_dates.npy"), dates[select].values)
        print(split, json.dumps(metrics[split]), flush=True)
    (output / "metrics.json").write_text(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
