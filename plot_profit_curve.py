"""
plot_profit_curve.py — cumulative-return curves from `ret_s.npy` files.
"""

from __future__ import annotations

from pathlib import Path
from datetime import datetime, timedelta
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.ticker import AutoMinorLocator, MaxNLocator


# ---------------------------------------------------------------------------
# 1) Series registry — edit here.
# Legacy files retain their original start; new files use ret_s_dates.npy.
# ---------------------------------------------------------------------------
RUNS: dict = {
    "csi300": {
        "LightGBM": "data/lightgbm_logs/csi300/ret_s.npy",
        "XGBoost": "data/xgboost_logs/csi300/ret_s.npy",
        "Alpha158": "data/alpha158_logs/csi300/ret_s.npy",
        "GP": "data/gp_logs/csi300_seed_0/ret_s.npy",
        "AlphaQCM": "data/qcm_logs/csi300_qrdqn_seed0/ret_s.npy",
        "AlphaSAGE": {"path": "data/gfn_logs/pool_50/gfn_gnn_csi300_50_2-0.01-1.0-1.0-1.0-0.3-linear-0.0/ret_s.npy", "data_start_date": "2023-05-01"},
        "AlphaPROBE": {"path": "data/knowledge_logs/pool_50/kg_dag_and_bayesian_icir_and_mutl_new_no_decay_MiniMax-M3_5_csi300_0.5_7_50_0.9_50_20_0.006_True_True_False_True_0.7_0.1_0.05/adaptive_n20/round_20/ret_s.npy", "data_start_date": "2023-05-01"},
        "AlphaGen": {"path": "data/ppo_logs/pool_20/ppo_csi300_20_0-20260905132532/ppo_csi300_20_0_20260905132532/ret_s.npy", "data_start_date": "2023-05-01"},
        "Ours": {"path": "data/cf_logs/20261024_020903_733059_csi300_0_gd-mobile-deepseek-v4.1-flash-1017/ret_s.npy", "data_start_date": "2023-05-01"},

        "AlphaAgent": "data/logs/alphaagent/csi300/validation_combiners_20261102/blend_a2_b45/ret_s.npy",
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
        "AlphaAgent": "data/logs/alphaagent/csi500/paper_deepseek_20261023/validation_zoo_v2_20261024/selected_zoo/ret_s.npy",
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
        "AlphaPROBE": "data/knowledge_logs/pool_50/kg_dag_and_bayesian_icir_and_mutl_new_no_decay_gd-mobile-deepseek-v4.1-flash-1017_5_csi1000_0.5_7_50_0.9_50_20_0.006_True_True_False_True_0.7_0.1_0.05/ret_s.npy",
        "AlphaGen": "data/ppo_logs/pool_20/ppo_csi1000_20_0-20261014201645/ppo_csi1000_20_0_20261014201645/ret_s.npy",
        "Ours": "data/cf_logs/20261024_224833_312713_csi1000_0_gd-mobile-deepseek-v4.1-flash-1017/ret_s.npy",
        "AlphaAgent": "data/logs/alphaagent/csi1000/paper_deepseek_20261024/validation_zoo_v2_20261024/trial_best_union/ret_s.npy",
        "CSI1000 Index": {
            "path": "data/index_real_logs/pool_all/real_csi1000/ret_s.npy",
            "color": "#000000",
            "linestyle": "--",
            "linewidth": 1.6,
        },
    },
}

# 2) AlphaSAGE-inspired palette (`img/backtest.png`), with Ours in muted red.
PALETTE: dict[str, str] = {
    "CSI300 Index": "#000000",
    "CSI500 Index": "#000000",
    "CSI1000 Index": "#000000",
    "LightGBM":   "#F47772",
    "XGBoost":    "#D9C2DF",
    "GP":         "#B182BA",
    "AlphaQCM":   "#7DCFF0",
    "Alpha158":   "#4B92CD",
    "AlphaSAGE":  "#429E9B",
    "AlphaPROBE": "#E6AE46",
    "AlphaGen":   "#7CC5A1",
    "AlphaAgent": "#6B6B6B",
    "Ours":       "#C8324A",
}
LEGEND_ORDER = ("Ours", "AlphaAgent", "AlphaSAGE", "AlphaPROBE", "AlphaGen", "AlphaQCM",
                "Alpha158", "GP", "XGBoost", "LightGBM")

# Apply locally so importing this script does not restyle other figures.
PLOT_STYLE = {
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Nimbus Roman", "Liberation Serif", "DejaVu Serif"],
    "font.size": 13,
    "axes.titlesize": 15,
    "axes.labelsize": 16,
    "axes.labelpad": 9,
    "axes.edgecolor": "#333333",
    "axes.linewidth": 0.8,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.facecolor": "white",
    "figure.facecolor": "white",
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "xtick.color": "#333333",
    "ytick.color": "#333333",
    "axes.axisbelow": True,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
}

# 3) Plot knobs.
DEFAULT_START_DATE = "2023-05-01"
DEFAULT_END_DATE = "2026-04-30"
LINEWIDTH = 2.1
SMOOTH_WINDOW = 7  # Trading days; set to 1 to display the raw curve.

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


def _smooth_curve(values: np.ndarray) -> np.ndarray:
    """Centered triangular average for display only; preserve both endpoints."""
    smoothed = np.empty_like(values)
    for i in range(len(values)):
        radius = min(SMOOTH_WINDOW // 2, i, len(values) - 1 - i)
        weights = SMOOTH_WINDOW // 2 + 1 - np.abs(np.arange(-radius, radius + 1))
        smoothed[i] = np.average(values[i - radius:i + radius + 1], weights=weights)
    return smoothed


@plt.rc_context(PLOT_STYLE)
def plot_market(market: str, runs: dict, here: Path, calendar: list[datetime], output_name: str | None = None) -> None:
    fig, ax = plt.subplots(figsize=(10.8, 5.6), dpi=300)
    fig.subplots_adjust(left=0.085, right=0.96, bottom=0.15, top=0.975)

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
            _smooth_curve(cum),
            color=cfg.get("color", PALETTE[name]),
            linestyle=cfg.get("linestyle", "-"),
            linewidth=cfg.get("linewidth", 2.7 if name == "Ours" else LINEWIDTH),
            label=cfg.get("label", name) + (" (partial)" if partial else ""),
            solid_capstyle="round",
            zorder=4 if name == "Ours" else 3 if "Index" in name else 2,
        )
        summary.append((name, len(ret), float(ret.mean()), float(ret.std()), float(cum[-1])))

    if not summary:
        raise SystemExit("No series plotted — check the paths in RUNS.")

    # Keep a fixed method order across markets, with the index last.
    handles, labels = ax.get_legend_handles_labels()
    order = sorted(range(len(summary)), key=lambda i: LEGEND_ORDER.index(summary[i][0])
                   if summary[i][0] in LEGEND_ORDER else len(LEGEND_ORDER))
    legend = ax.legend([handles[i] for i in order], [labels[i] for i in order],
                       loc="upper left", ncol=3, frameon=True, framealpha=0.94,
                       facecolor="white", edgecolor="#D8D8D8", borderpad=0.6,
                       handlelength=2.4, handletextpad=0.65, columnspacing=1.5,
                       labelspacing=0.4, fontsize=11.5)
    for label in legend.get_texts():
        if label.get_text() == "Ours" or label.get_text().startswith("Ours ("):
            label.set_weight("bold")
    # Follow AlphaSAGE/AlphaPROBE: calendar-month starts every four months.
    test_start = datetime.fromisoformat(DEFAULT_START_DATE)
    test_end = datetime.fromisoformat(DEFAULT_END_DATE)
    ax.xaxis.set_major_locator(mdates.MonthLocator(bymonth=[1, 5, 9], bymonthday=1))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax.xaxis.set_minor_locator(mdates.MonthLocator())
    ax.yaxis.set_major_locator(MaxNLocator(nbins=6))
    ax.yaxis.set_minor_locator(AutoMinorLocator(4))
    ax.tick_params(which="major", bottom=True, left=True, length=4, width=0.8, direction="out")
    ax.tick_params(which="minor", bottom=True, left=True, length=2, width=0.5, direction="out")
    ax.set_xlabel("Date")
    ax.set_ylabel("Cumulative Return")
    ax.set_xlim(test_start - timedelta(days=12), test_end + timedelta(days=12))
    ax.grid(which="major", color="#DCDCDC", linestyle="--", linewidth=0.7)
    ax.grid(which="minor", color="#EEEEEE", linestyle=":", linewidth=0.55)

    # Y-limits with headroom.
    ymin, ymax = ax.get_ylim()
    pad = max(0.04, 0.08 * (ymax - ymin))
    ax.set_ylim(ymin - 0.3 * pad, ymax + pad)

    out_path = (here / (output_name or f"profit_curve_{market}.png")).resolve()
    fig.savefig(out_path, dpi=300)
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

