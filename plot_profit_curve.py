"""
plot_profit_curve.py — cumulative-return curves from `ret_s.npy` files.

Style mimics `img/backtest.png`:
  - Wide canvas, seaborn whitegrid, dashed gridlines
  - Y-axis normalised so every series starts at 1.0 (cumulative wealth)
  - X-axis is dates with year-month format
  - Legend at top, frameless

Edit `RUNS` at the top to add / remove / rename a series.  Each value can be:
  - str  : path to `ret_s.npy`, default start date 2023-05-01
  - tuple[str, str] : (path, "YYYY-MM-DD" start date)
  - dict  : full config with these keys:
      path           (str,  required)
      start_date     (str,  default DEFAULT_START_DATE = "2023-05-01")
      data_start_date(str,  when the .npy file actually starts — used
                      together with `start_date` to skip leading entries)
      color          (str,  default = PALETTE[name])
      linestyle      (str,  default "-")
      linewidth      (float, default LINEWIDTH)

Every .npy holds one entry per *real trading day*; the x-axis is taken from
the real qlib trading calendar (CALENDAR_PATH), not a synthetic Mon–Fri grid,
so market holidays are not treated as trading days.

Paths are relative to the script's directory.
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
    "Alpha158": {"path": "data/alpha158_logs/csi300/ret_s.npy", "data_start_date": "2023-05-01"},
    "AlphaSAGE": {"path": "data/gfn_logs/pool_50/gfn_gnn_csi300_50_2-0.01-1.0-1.0-1.0-0.3-linear-0.0/ret_s.npy", "data_start_date": "2023-05-01"},
    "AlphaPROBE": {"path": "data/knowledge_logs/pool_50/kg_dag_and_bayesian_icir_and_mutl_new_no_decay_MiniMax-M3_5_csi300_0.5_7_50_0.9_50_20_0.006_True_True_False_True_0.7_0.1_0.05/ret_s.npy", "data_start_date": "2023-05-01"},
    "AlphaGen": {"path": "data/ppo_logs/pool_20/ppo_csi300_20_0-20260905132532/ppo_csi300_20_0_20260905132532/ret_s.npy", "data_start_date": "2023-05-01"},
    "Ours": {"path": "data/cf_logs/20261009_214514_822056_csi300_0_MiniMax-M3/ret_s.npy", "data_start_date": "2023-05-01"},
    # "AlphaCF_START": {"path": "ret_s.npy", "data_start_date": "2023-05-01"},

    "CSI300 Index": {
        "path":            "data/index_real_logs/pool_all/real_csi300/ret_s.npy",
        "data_start_date": "2023-05-01",   # first entry in the .npy file
        "start_date":      "2023-05-01",   # align with the alpha series
        "color":           "#000000",
        "linestyle":       "--",
        "linewidth":       2.2,
    },
}

# 2) Palette — muted tones reminiscent of `img/backtest.png`.
PALETTE: dict[str, str] = {
    "Alpha158":   "#4C72B0",  # steel blue
    "AlphaSAGE":  "#2E8B57",  # sea green
    "AlphaPROBE": "#E07B5A",  # muted coral
    "AlphaGen":   "#82B366",  # light green
    "Ours":    "#FF0000",  # red
    # "AlphaCF_START": "#4C72B0",  # steel blue
}

# 3) Plot knobs.
TITLE = "CSI300 Profit Curve"
OUTPUT_PNG = "profit_curve.png"
DEFAULT_START_DATE = "2023-05-01"
DEFAULT_END_DATE = "2026-04-30"
LINEWIDTH = 2.2

# Real trading calendar (identical to the one the cn backtests use).  All
# series are indexed against this so holidays line up correctly.
CALENDAR_PATH = "data/qlib_data/cn_data_rolling/calendars/day.txt"
# ---------------------------------------------------------------------------


def _resolve(path_str: str, anchor: Path) -> Path:
    p = Path(path_str)
    return p if p.is_absolute() else (anchor / p).resolve()


def _load_ret(path_str: str, anchor: Path) -> np.ndarray | None:
    p = _resolve(path_str, anchor)
    if not p.exists():
        print(f"[warn] missing file: {p}")
        return None
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


def main() -> None:
    here = Path(__file__).resolve().parent
    calendar = _load_calendar(here)

    sns.set_style("whitegrid")
    sns.set_context("notebook", rc={
        "font.family": "serif",
        "font.size": 14,
        "axes.titlesize": 16,
        "axes.labelsize": 14,
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
        "legend.fontsize": 12,
        "axes.linewidth": 0.8,
        "axes.edgecolor": "#444",
        "grid.color": "#cccccc",
        "grid.linestyle": "--",
        "grid.linewidth": 0.7,
        "grid.alpha": 0.6,
    })

    fig, ax = plt.subplots(figsize=(14.0, 7.0), dpi=130)

    summary: list[tuple[str, int, float, float, float]] = []
    for name, raw_spec in RUNS.items():
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
            color=cfg.get("color", PALETTE.get(name)),
            linestyle=cfg.get("linestyle", "-"),
            linewidth=cfg.get("linewidth", LINEWIDTH),
            label=name + (" (partial)" if partial else ""),
            solid_capstyle="round",
        )
        summary.append((name, len(ret), float(ret.mean()), float(ret.std()), float(cum[-1])))

    if not summary:
        raise SystemExit("No series plotted — check the paths in RUNS.")

    # Legend above the axes, frameless, 3 columns.
    ax.legend(
        loc="upper left",
        bbox_to_anchor=(0.0, 1.10),
        ncol=3,
        handlelength=2.6,
        handletextpad=0.6,
        columnspacing=1.6,
        borderaxespad=0.0,
    )

    # X-axis: year-month labels.
    ax.xaxis.set_major_locator(mdates.AutoDateLocator(maxticks=12))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    fig.autofmt_xdate(rotation=0, ha="center")

    ax.set_xlabel("Date")
    ax.set_ylabel("Cumulative Return")
    ax.set_title(f"{TITLE} | {DEFAULT_START_DATE} to {DEFAULT_END_DATE}", pad=70)
    ax.set_xlim(datetime.fromisoformat(DEFAULT_START_DATE), datetime.fromisoformat(DEFAULT_END_DATE))

    # Y-limits with headroom.
    ymin, ymax = ax.get_ylim()
    pad = max(0.05, 0.10 * (ymax - ymin))
    ax.set_ylim(ymin - pad, ymax + pad)

    out_path = (here / OUTPUT_PNG).resolve()
    fig.tight_layout()
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    print(f"[ok] saved {out_path}")

    print()
    print(f"{'Name':<14} {'N':>5} {'Mean':>9} {'Std':>9} {'Final':>10}")
    for name, n, m, s, f in sorted(summary, key=lambda x: x[4], reverse=True):
        print(f"{name:<14} {n:>5} {m*100:>+8.3f}% {s*100:>8.3f}% {f:>+9.4f}")


if __name__ == "__main__":
    main()
