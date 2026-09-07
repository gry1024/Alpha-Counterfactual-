from alphagen.config import CONSTANTS, DELTA_TIMES, FEATURES, OPERATORS

# Calendar splits for train / valid / test
TRAIN_START, TRAIN_END = "2010-01-01", "2021-12-31"
VALID_START, VALID_END = "2022-01-01", "2022-12-31"
TEST_START, TEST_END = "2023-01-01", "2026-04-30"

# Qlib binary dirs, relative to the repo root
QLIB_CN = "data/qlib_data/cn_data_rolling"
QLIB_US = "data/qlib_data/us_data_qlib_latest"

# Per-seed search: LLM batch size, backtest cap, stop rules
K = 6
TOPK = 3
MAX_ROUNDS = 10
PATIENCE = 5
N_SEEDS = 20

# Q_phi: cold-start size; score = μ + β·σ + λ·D
MIN_Q_SAMPLES = 128
BETA = 1.0
LAMBDA_DIV = 0.02

# "$close" etc.; operator class-name -> class
FEATURE_NAMES = ["$" + f.name.lower() for f in FEATURES]
OP_BY_NAME = {op.__name__: op for op in OPERATORS}


# instrument -> qlib data directory (US only for "sp500")
def qlib_path(instrument: str) -> str:
    return QLIB_US if instrument == "sp500" else QLIB_CN
