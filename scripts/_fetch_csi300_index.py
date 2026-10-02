"""Read CSI300 index data from Qlib and save as a `ret_s.npy`.

`plot_profit_curve.py` indexes every series with the real qlib trading
calendar (`cn_data_rolling/calendars/day.txt`), so this file must contain
exactly one entry per *real trading day* — no weekday padding and no
zero-filling of market holidays (Spring Festival, National Day, ...).

Run from /home/groy/cf via:
  wsl -e bash -c "cd /home/groy/cf && .venv/bin/python3 scripts/_fetch_csi300_index.py"
"""
from __future__ import annotations

from pathlib import Path
import sys
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import qlib
from qlib.constant import REG_CN
from qlib.data import D


def main() -> None:
    qlib.init(provider_uri=str(ROOT / "data/qlib_data/cn_data_rolling"), region=REG_CN)

    inst = "sh000300"
    # Qlib stores pctchg in % units; convert to decimal daily return.
    field = "$pctchg"
    df = D.features(
        [inst],
        [field],
        start_time="2022-01-04",
        end_time="2026-04-30",
        disk_cache=0,
    )
    series = df[field].astype(np.float64) / 100.0
    series = series.replace([np.inf, -np.inf], np.nan).fillna(0.0)

    # One entry per real trading day, sorted by date (Qlib returns them in order).
    arr = series.sort_index().to_numpy(dtype=np.float32)
    print(f"real trading-day entries: {arr.shape}, "
          f"first {series.index.get_level_values('datetime').min().date()}, "
          f"last {series.index.get_level_values('datetime').max().date()}")

    out_dir = ROOT / "data" / "index_real_logs" / "pool_all" / "real_csi300"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "ret_s.npy"
    np.save(out_path, arr)
    print(f"saved {out_path} shape={arr.shape} dtype={arr.dtype}")


if __name__ == "__main__":
    main()