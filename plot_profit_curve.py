"""
plot_profit_curve.py — cumulative-return curves from `ret_s.npy` files.
"""

from __future__ import annotations

from pathlib import Path
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns


# ---------------------------------------------------------------------------
# 1) Series registry — edit here.
# Legacy files retain their original start; new files use ret_s_dates.npy.
# ---------------------------------------------------------------------------
RUNS: dict = {
    "csi300": {
        "LightGBM": "data/lightgbm_logs/csi300/ret_s.npy",
        "XGBoost": "data/xgboost_logs/csi300/ret_s.npy",
        "Alpha158": "data/alpha158_logs/csi300_equal_weight/ret_s.npy",
        "GP": "data/gp_logs/csi300_seed_0/ret_s.npy",
        "AlphaQCM": "data/qcm_logs/csi300_qrdqn_seed0/ret_s.npy",
        "AlphaSAGE": {"path": "data/gfn_logs/pool_50/gfn_gnn_csi300_50_2-0.01-1.0-1.0-1.0-0.3-linear-0.0/ret_s.npy", "data_start_date": "2023-05-01"},
        "AlphaPROBE": {"path": "data/knowledge_logs/pool_50/kg_dag_and_bayesian_icir_and_mutl_new_no_decay_MiniMax-M3_5_csi300_0.5_7_50_0.9_50_20_0.006_True_True_False_True_0.7_0.1_0.05/ret_s.npy", "data_start_date": "2023-05-01"},
        "AlphaGen": {"path": "data/ppo_logs/pool_20/ppo_csi300_20_0-20260905132532/ppo_csi300_20_0_20260905132532/ret_s.npy", "data_start_date": "2023-05-01"},
        "Ours": {"path": "data/cf_logs/20261013_225346_506634_csi300_0_MiniMax-M3/ret_s.npy", "data_start_date": "2023-05-01"},

        "CSI300 Index": {
            "path":            "data/index_real_logs/pool_all/real_csi300/ret_s.npy",
            "data_start_date": "2023-05-01",   # first entry in the .npy file
            "start_date":      "2023-05-01",   # align with the alpha series
            "color":           "#000000",
            "linestyle":       "--",
            "linewidth":       1.6,
        },
    },
    "csi500": {
        "LightGBM": "data/lightgbm_logs/csi500/ret_s.npy",
        "XGBoost": "data/xgboost_logs/csi500/ret_s.npy",
        "Alpha158": "data/alpha158_logs/csi500_equal_weight/ret_s.npy",
        "GP": "data/gp_logs/csi500_seed_0/ret_s.npy",
        "AlphaQCM": "data/qcm_logs/csi500_qrqdn_seed0/ret_s.npy",
        "AlphaSAGE": "data/gfn_logs/pool_50/gfn_gnn_csi500_50_0-0.01-1.0-1.0-1.0-0.3-linear-0.0/ret_s.npy",
        "AlphaPROBE": "data/knowledge_logs/pool_50/kg_dag_and_bayesian_icir_and_mutl_new_no_decay_MiniMax-M3_5_csi500_0.5_7_50_0.9_50_20_0.006_True_True_False_True_0.7_0.1_0.05/ret_s.npy",
        "AlphaGen": "data/ppo_logs/pool_20/ppo_csi500_20_0-20261013092950/ppo_csi500_20_0_20261013092950/ret_s.npy",
        "Ours": "data/cf_logs/20261014_171245_875336_csi500_1_MiniMax-M3/ret_s.npy",
        "CSI500 Index": {
            "path": "data/index_real_logs/pool_all/real_csi500/ret_s.npy",
            "color": "#000000",
            "linestyle": "--",
            "linewidth": 1.6,
        },
    },
    "csi1000": {
        "LightGBM": "data/lightgbm_logs/csi1000/ret_s.npy",
        "XGBoost": "data/xgboost_logs/csi1000/ret_s.npy",
        "Alpha158": "data/alpha158_logs/csi1000_equal_weight/ret_s.npy",
        "GP": "data/gp_logs/csi1000_seed_0/ret_s.npy",
        "AlphaQCM": "data/qcm_logs/csi1000_qrqdn_seed0/ret_s.npy",
        "AlphaSAGE": "data/gfn_logs/pool_50/gfn_gnn_csi1000_50_0-0.01-1.0-1.0-1.0-0.3-linear-0.0/ret_s.npy",
        "AlphaPROBE": None,
        "AlphaGen": "data/ppo_logs/pool_20/ppo_csi1000_20_0-20261014201645/ppo_csi1000_20_0_20261014201645/ret_s.npy",
        "Ours": "data/cf_logs/20261015_062123_202538_csi1000_0_MiniMax-M3/ret_s.npy",
        "CSI1000 Index": {
            "path": "data/index_real_logs/pool_all/real_csi1000/ret_s.npy",
            "color": "#000000",
            "linestyle": "--",
            "linewidth": 1.6,
        },
    },
}

# 2) Palette — muted tones reminiscent of `img/backtest.png`.
PALETTE: dict[str, str] = {
    "CSI300 Index": "#000000",
    "CSI500 Index": "#000000",
    "CSI1000 Index": "#000000",
    "LightGBM":   "#AAB8D0",
    "XGBoost":    "#C28CB8",
    "GP":         "#8172B3",
    "AlphaQCM":   "#937860",
    "Alpha158":   "#4C72B0",  # steel blue
    "AlphaSAGE":  "#2E8B57",  # sea green
    "AlphaPROBE": "#E07B5A",  # muted coral
    "AlphaGen":   "#82B366",  # light green
    "Ours":    "#FF0000",  # red
}

# 3) Plot knobs.
DEFAULT_START_DATE = "2023-05-01"
DEFAULT_END_DATE = "2026-04-30"
LINEWIDTH = 1.6

# Real trading calendar (identical to the one the cn backtests use).  All
# series are indexed against this so holidays line up correctly.
CALENDAR_PATH = "data/qlib_data/cn_data_rolling/calendars/day.txt"
# ---------------------------------------------------------------------------


def _resolve(path_str: str, anchor: Path) -> Path:
    p = Path(path_str)
    return p if p.is_absolute() else (anchor / p).resolve()


def _load_ret(path_str: str, anchor: Path) -> np.ndarray:
    p = _resolve(path_str, anchor)
    if not p.exists():
        raise FileNotFoundError(f"return series not found: {p}")
    return np.load(p).astype(np.float64)


def _load_calendar(anchor: Path) -> list[datetime]:
    """Load the real trading-day calendar (ascending, unique)."""
    path = _resolve(CALENDAR_PATH, anchor)
    if not path.exists():
        raise FileNotFoundError(f"trading calendar not found: {path}")
    dates: list[datetime] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                dates.append(datetime.strptime(line, "%Y-%m-%d"))
    return sorted(set(dates))


def _trading_dates(start_str: str, n: int, calendar: list[datetime]) -> list[datetime]:
    """Return the first `n` real trading days on/after `start_str`."""
    start = datetime.strptime(start_str, "%Y-%m-%d")
    return [d for d in calendar if d >= start][:n]


def _entries_before(start_str: str, target_str: str, calendar: list[datetime]) -> int:
    """Count real trading days in [start_str, target_str).  Used to slice a
    .npy whose data starts before the date we want to plot from."""
    start = datetime.strptime(start_str, "%Y-%m-%d")
    target = datetime.strptime(target_str, "%Y-%m-%d")
    return sum(1 for d in calendar if start <= d < target)


def _normalize_to_one(ret: np.ndarray) -> np.ndarray:
    """Shift cumulative sum so the first plotted point equals 1.0."""
    cum = np.cumsum(ret)
    return cum - cum[0] + 1.0


def _parse_spec(spec):
    """Normalise a RUNS value into a config dict."""
    if isinstance(spec, str):
        return {"path": spec}
    if isinstance(spec, tuple):
        return {"path": spec[0], "start_date": spec[1]}
    if isinstance(spec, dict):
        return dict(spec)
    raise ValueError(f"Unsupported spec: {spec!r}")


def plot_market(market: str, runs: dict, here: Path, calendar: list[datetime], output_name: str | None = None) -> None:

    sns.set_style("whitegrid")
    sns.set_context("notebook", rc={
        "font.family": "serif",
        "font.size": 11,
        "axes.titlesize": 11,
        "axes.labelsize": 11,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 9,
        "axes.linewidth": 0.8,
        "axes.edgecolor": "#444",
        "grid.color": "#cccccc",
        "grid.linestyle": "--",
        "grid.linewidth": 0.7,
        "grid.alpha": 0.6,
    })

    fig, ax = plt.subplots(figsize=(6.0, 6.0), dpi=240)
    fig.subplots_adjust(left=0.17, right=0.95, bottom=0.13, top=0.94)
    ax.set_box_aspect(1)

    summary: list[tuple[str, int, float, float, float]] = []
    for name, raw_spec in runs.items():
        if raw_spec is None:
            print(f"[skip] {market}/{name}: pending result")
            continue
        cfg = _parse_spec(raw_spec)
        path = cfg["path"]
        ret = _load_ret(path, here)
        if ret is None:
            continue

        start_date = cfg.get("start_date", DEFAULT_START_DATE)
        sidecar = _resolve(path, here).with_name("ret_s_dates.npy")
        if sidecar.exists():
            dates = np.load(sidecar).astype("datetime64[ns]")
            if len(dates) != len(ret):
                raise ValueError(f"{name}: return/date lengths differ")
        else:
            dates = np.asarray(_trading_dates(cfg.get("data_start_date", start_date),
                                             len(ret), calendar), dtype="datetime64[ns]")
            if len(dates) != len(ret):
                raise ValueError(f"{name}: returns extend beyond the real calendar")
        keep = (dates >= np.datetime64(start_date)) & (dates <= np.datetime64(DEFAULT_END_DATE))
        ret, dates = ret[keep], dates[keep]
        if not len(ret):
            print(f"[warn] {name}: no returns in the Test window")
            continue
        expected_end = max(d for d in calendar if d <= datetime.fromisoformat(DEFAULT_END_DATE))
        expected_start = min(d for d in calendar if d >= datetime.fromisoformat(start_date))
        partial = dates[0] > np.datetime64(expected_start) or dates[-1] < np.datetime64(expected_end)
        if partial:
            print(f"[warn] {name}: existing results cover {str(dates[0])[:10]}..{str(dates[-1])[:10]}; partial Test coverage")
        cum = _normalize_to_one(ret)
        ax.plot(
            dates,
            cum,
            color=PALETTE[name],
            linestyle=cfg.get("linestyle", "-"),
            linewidth=cfg.get("linewidth", LINEWIDTH),
            label=cfg.get("label", name) + (" (partial)" if partial else ""),
            solid_capstyle="round",
        )
        summary.append((name, len(ret), float(ret.mean()), float(ret.std()), float(cum[-1])))

    if not summary:
        raise SystemExit("No series plotted — check the paths in RUNS.")

    # Keep all methods inside the upper-left corner as the baseline list grows.
    ax.legend(loc="upper left", ncol=2, frameon=True, framealpha=0.9,
              facecolor="white", edgecolor="none", handlelength=1.5, handletextpad=0.4,
              columnspacing=0.9, labelspacing=0.35, fontsize=8)
    ax.set_title(market.upper(), pad=8, fontsize=12)
    # Always label both Test boundaries; sparse interior ticks keep the square compact.
    test_start = datetime.fromisoformat(DEFAULT_START_DATE)
    test_end = datetime.fromisoformat(DEFAULT_END_DATE)
    interior = [datetime(year, 7, 1) for year in range(test_start.year + 1, test_end.year)
                if test_start < datetime(year, 7, 1) < test_end]
    ticks = [test_start, *interior, test_end]
    ax.set_xticks(ticks)
    ax.set_xticklabels([ticks[0].strftime("%Y-%m-%d"),
                        *[d.strftime("%Y-%m") for d in interior],
                        ticks[-1].strftime("%Y-%m-%d")])
    ax.get_xticklabels()[0].set_horizontalalignment("left")
    ax.get_xticklabels()[-1].set_horizontalalignment("right")
    ax.set_xlabel("Date")
    ax.set_ylabel("Cumulative Return")
    ax.set_xlim(datetime.fromisoformat(DEFAULT_START_DATE), datetime.fromisoformat(DEFAULT_END_DATE))
    ax.grid(axis="x", visible=False)

    # Y-limits with headroom.
    ymin, ymax = ax.get_ylim()
    pad = max(0.05, 0.10 * (ymax - ymin))
    ax.set_ylim(ymin - pad, ymax + pad)

    out_path = (here / (output_name or f"profit_curve_{market}.png")).resolve()
    fig.savefig(out_path, dpi=240)
    fig.savefig(out_path.with_suffix(".pdf"))
    plt.close(fig)
    print(f"[ok] saved {out_path}")

    print()
    print(f"{'Name':<14} {'N':>5} {'Mean':>9} {'Std':>9} {'Final':>10}")
    for name, n, m, s, f in sorted(summary, key=lambda x: x[4], reverse=True):
        print(f"{name:<14} {n:>5} {m*100:>+8.3f}% {s*100:>8.3f}% {f:>+9.4f}")


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--alpha158-equal-weight", action="store_true",
                        help="Also save the explicitly labeled CSI1000 equal-weight curve (legacy filename).")
    args = parser.parse_args()
    here = Path(__file__).resolve().parent
    calendar = _load_calendar(here)
    if args.alpha158_equal_weight:
        runs = dict(RUNS["csi1000"])
        runs["Alpha158"] = {"path": "data/alpha158_logs/csi1000_equal_weight/ret_s.npy",
                            "label": "Alpha158 (equal-weight)"}
        plot_market("csi1000", runs, here, calendar, "profit_curve_csi1000_equal_weight.png")
    else:
        for market, runs in RUNS.items():
            plot_market(market, runs, here, calendar)


if __name__ == "__main__":
    main()

