"""AlphaSAGE LightGBM baseline with the repository's dates and daily-return metrics."""
import argparse
import gc
import json
from pathlib import Path
import time
from types import SimpleNamespace

import lightgbm as lgb
import numpy as np
import pandas as pd
import qlib
from qlib.data import D
import torch

import run_adaptive_combination as backtest
from alpha158.ranking import batch_spearmanr


FIELDS = ["open", "close", "high", "low", "volume"]


class CachedRows(lgb.Sequence):
    """Let LightGBM construct its dataset in batches without copying the full cache."""
    def __init__(self, cache, mask):
        self.cache = cache
        self.indices = np.flatnonzero(mask)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        return np.asarray(self.cache[self.indices[index]], dtype=np.float64)


def build_features(output, market, changes, present, train_days):
    """Build lag-major OHLCV changes, matching AlphaSAGE's range(1, 60)."""
    rows, cols = np.nonzero(present)
    train = train_days[rows]
    cache = np.lib.format.open_memmap(output / "features.npy", mode="w+",
                                     dtype=np.float32, shape=(len(rows), 295))
    means, stds = [], []
    for lag in range(1, 60):
        for field in range(5):
            values = changes[rows[train] - lag, cols[train], field]
            mean, std = np.nanmean(values), np.nanstd(values, ddof=1)
            means.append(float(mean))
            stds.append(float(std) if np.isfinite(std) and std > 0 else 1.0)
    means, stds = np.array(means, dtype=np.float32), np.array(stds, dtype=np.float32)
    for offset in range(0, len(rows), 16384):
        rr, cc = rows[offset:offset + 16384], cols[offset:offset + 16384]
        block = np.empty((len(rr), 295), dtype=np.float32)
        for lag in range(1, 60):
            block[:, (lag - 1) * 5:lag * 5] = changes[rr - lag, cc]
        block = (block - means) / stds
        cache[offset:offset + len(rr)] = np.clip(
            np.nan_to_num(block, nan=0., posinf=0., neginf=0.), -4, 4)
        if offset % 163840 == 0:
            print(f"[{market}] feature rows {offset}/{len(rows)}", flush=True)
    cache.flush()
    np.savez(output / "normalization.npz", mean=means, std=stds)
    return cache, rows, cols


def normalized_labels(labels, present):
    """Apply the same per-day target transform to Train and Validation."""
    frame = pd.DataFrame(labels).where(present)
    frame = frame.sub(frame.mean(axis=1), axis=0).div(frame.std(axis=1), axis=0)
    return np.clip(np.nan_to_num(frame.to_numpy(), nan=0., posinf=0., neginf=0.), -4, 4)


def main(model_type="lightgbm"):
    parser = argparse.ArgumentParser(description=f"AlphaSAGE {model_type} baseline with shared OHLCV features and daily-return metrics.")
    parser.add_argument("--instruments", choices=["csi300", "csi500", "csi1000", "sp500"], default="csi300")
    parser.add_argument("--qlib-path")
    parser.add_argument("--output-dir")
    parser.add_argument("--feature-cache", help="Reuse an existing normalized feature cache with identical data/configuration.")
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    if args.threads < 1:
        parser.error("threads must be positive")
    started = time.perf_counter()
    torch.set_num_threads(args.threads)
    np.random.seed(args.seed)
    root = Path(__file__).resolve().parent
    output = Path(args.output_dir or root / f"data/{model_type}_logs" / args.instruments).resolve()
    output.mkdir(parents=True, exist_ok=True)
    provider = Path(args.qlib_path or root / "data/qlib_data" /
                    ("us_data_qlib_latest" if args.instruments == "sp500" else "cn_data_rolling")).resolve()
    qlib.init(provider_uri=str(provider), region="us" if args.instruments == "sp500" else "cn", kernels=1)
    calendar = pd.DatetimeIndex(D.calendar())
    start = calendar.searchsorted("2015-01-01") - 100
    end = calendar.searchsorted("2026-04-30", side="right") + 30
    dates = calendar[start:end]
    frame = D.features(D.instruments(args.instruments), ["$" + f for f in FIELDS],
                       start_time=dates[0], end_time=dates[-1])
    stocks = pd.Index(sorted(frame.index.get_level_values("instrument").unique()))
    raw = np.stack([frame["$" + f].unstack("instrument").reindex(index=dates, columns=stocks)
                    .to_numpy(dtype=np.float32) for f in FIELDS], axis=-1)
    present = pd.Series(True, index=frame.index).unstack("instrument").reindex(
        index=dates, columns=stocks, fill_value=False).fillna(False).to_numpy(dtype=bool)
    close = raw[:, :, 1]
    with np.errstate(divide="ignore", invalid="ignore"):
        labels = np.full_like(close, np.nan)
        labels[:-20] = close[20:] / close[:-20] - 1
        daily = np.full_like(close, np.nan)
        daily[:-1] = close[1:] / close[:-1] - 1
        changes = np.full_like(raw, np.nan)
        changes[1:] = raw[1:] / raw[:-1] - 1
    labels[~np.isfinite(labels)] = np.nan
    daily[~np.isfinite(daily)] = np.nan
    changes[~np.isfinite(changes)] = np.nan
    del frame, raw
    # Only use labels whose entire horizon has matured by the Train cutoff.
    maturity = np.zeros(len(dates), dtype=bool)
    maturity[:-20] = dates[20:] <= pd.Timestamp("2021-12-31")
    train_days = (dates >= "2015-01-01") & (dates <= "2021-12-31") & maturity
    valid_days = (dates >= "2022-01-01") & (dates <= "2022-12-31")
    test_days = (dates >= "2023-05-01") & (dates <= "2026-04-30")
    present = present & (train_days | valid_days | test_days)[:, None]
    if args.feature_cache:
        cache_path = Path(args.feature_cache).resolve()
        prior = json.loads((cache_path.parent / "lightgbm.json").read_text())
        if prior["provider"] != str(provider) or prior["args"]["instruments"] != args.instruments or prior["stocks"] != stocks.tolist():
            raise ValueError("Feature cache dataset does not match this run")
        rows, cols = np.nonzero(present)
    else:
        cache_path = output / "features.npy"
        cache, rows, cols = build_features(output, args.instruments, changes, present, train_days)
        del cache
    del changes
    gc.collect()
    cache = np.load(cache_path, mmap_mode="r")
    if cache.shape != (len(rows), 295):
        raise ValueError("Feature cache axes do not match this run")
    train_mask = train_days[rows] & np.isfinite(labels[rows, cols])
    valid_mask = valid_days[rows] & np.isfinite(labels[rows, cols])
    normalized = normalized_labels(labels, present)
    y_train = normalized[rows[train_mask], cols[train_mask]]
    y_valid = normalized[rows[valid_mask], cols[valid_mask]]
    del normalized
    if model_type == "lightgbm":
        train_data = lgb.Dataset(CachedRows(cache, train_mask), label=y_train, free_raw_data=True, params={"num_threads": args.threads})
        train_data.construct()
        valid_data = lgb.Dataset(CachedRows(cache, valid_mask), label=y_valid,
                                 reference=train_data, free_raw_data=True)
        valid_data.construct()
        params = dict(objective="regression", metric="rmse", num_leaves=128, max_depth=64,
                      learning_rate=0.2, feature_fraction=0.9, bagging_fraction=0.8,
                      bagging_freq=5, verbosity=-1, num_threads=args.threads, seed=args.seed)
        print(f"[{args.instruments}] train rows={train_mask.sum()}, valid rows={valid_mask.sum()}", flush=True)
        train_started = time.perf_counter()
        model = lgb.train(params, train_data, valid_sets=[train_data, valid_data],
                          num_boost_round=100, callbacks=[lgb.early_stopping(100), lgb.log_evaluation(10)])
        training_seconds = time.perf_counter() - train_started
        model.save_model(str(output / "model.txt"))
    else:
        from train_xgboost import train
        train_started = time.perf_counter()
        model, train_data, valid_data, params = train(cache, train_mask, y_train, valid_mask,
            y_valid, args)
        training_seconds = time.perf_counter() - train_started
        model.save_model(output / "model.ubj")
    del train_data, valid_data
    gc.collect()
    backtest.args = SimpleNamespace(chunk_size=64)
    backtest.batch_spearmanr = batch_spearmanr
    metrics = {}
    for split, mask in [("Validation", valid_days), ("Test", test_days)]:
        predictions = np.full_like(labels, np.nan)
        select = mask[rows]
        indices = np.flatnonzero(select)
        for offset in range(0, len(indices), 32768):
            batch = indices[offset:offset + 32768]
            if model_type == "lightgbm":
                values = model.predict(cache[batch], num_threads=args.threads)
            else:
                values = model.inplace_predict(cache[batch], iteration_range=(0, model.best_iteration + 1))
            predictions[rows[batch], cols[batch]] = values
        pred = torch.from_numpy(predictions[mask])
        target = torch.from_numpy(labels[mask])
        returns = torch.from_numpy(daily[mask])
        metrics[split], ret = backtest.get_tensor_metrics(pred, target, returns)
        prefix = "" if split == "Test" else "valid_"
        np.save(output / f"{prefix}predictions.npy", pred.numpy())
        np.save(output / f"{prefix}ret_s.npy", ret.numpy())
        np.save(output / f"{prefix}ret_s_dates.npy", dates[mask].values)
        print(pd.DataFrame([metrics[split]], index=[split]).to_string(), flush=True)
    (output / "metrics.json").write_text(json.dumps(metrics, indent=2))
    manifest = dict(args=vars(args), provider=str(provider), params=params, model_type=model_type,
                    feature_cache=str(cache_path),
                    source="https://github.com/BerkinChen/AlphaSAGE/blob/main/exp_ML_train_and_result.ipynb",
                    library_version=lgb.__version__ if model_type == "lightgbm" else __import__("xgboost").__version__,
                    num_boost_round=100, early_stopping_rounds=100,
                    prediction_rounds=model.best_iteration if model_type == "lightgbm" else model.best_iteration + 1,
                    selection="Validation RMSE on the same standardized label scale as Train",
                    features="OHLCV 1-day changes, lags 1..59; Train mean/std, fill 0, clip [-4,4]",
                    train="2015-01-01..2021-12-31; exclude unmatured 20-day labels and missing labels",
                    validation="2022-01-01..2022-12-31", test="2023-05-01..2026-04-30",
                    train_label="20-day return, daily cross-sectional z-score, clip [-4,4]",
                    validation_label="20-day return, daily cross-sectional z-score, clip [-4,4]",
                    evaluation="raw 20-day return IC/RIC; top 20% long-only 1-day return, zero cost",
                    best_iteration=model.best_iteration if model_type == "lightgbm" else model.best_iteration + 1, training_seconds=training_seconds,
                    total_seconds=time.perf_counter() - started, stocks=stocks.tolist(),
                    train_rows=int(train_mask.sum()), valid_rows=int(valid_mask.sum()))
    (output / f"{model_type}.json").write_text(json.dumps(manifest, indent=2))
    print(f"[{args.instruments}] complete in {manifest['total_seconds']:.1f}s", flush=True)


if __name__ == "__main__":
    main()
