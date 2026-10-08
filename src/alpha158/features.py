"""Load all 158 official Qlib features on the existing backtest date/stock axes."""
import numpy as np
import torch
from qlib.contrib.data.loader import Alpha158DL
from qlib.data import D
from alphagen.utils.pytorch_utils import normalize_by_day


def load_tensor(data, batch_size=16):
    fields, _ = Alpha158DL.get_feature_config()
    dates = data._dates[data.max_backtrack_days:data.max_backtrack_days + data.n_days]
    instruments = D.instruments(data._instrument) if isinstance(data._instrument, str) else data._instrument
    result = torch.empty((len(dates), data.n_stocks, len(fields)), device=data.device)
    for start in range(0, len(fields), batch_size):
        batch = fields[start:start + batch_size]
        frame = D.features(instruments, batch, start_time=dates[0], end_time=dates[-1], freq=data.freq)
        for offset, field in enumerate(batch):
            values = frame[field].unstack("instrument").reindex(index=dates, columns=data._stock_ids)
            tensor = torch.tensor(values.to_numpy(dtype=np.float32), device=data.device)
            tensor = tensor.masked_fill(~tensor.isfinite(), torch.nan)
            result[:, :, start + offset] = normalize_by_day(tensor)
        print(f"[Alpha158] loaded {min(start + batch_size, len(fields))}/{len(fields)} features", flush=True)
    return result
