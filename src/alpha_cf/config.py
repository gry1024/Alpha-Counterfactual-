from alphagen.config import CONSTANTS, DELTA_TIMES, FEATURES, OPERATORS


# Calendar split: train is the only window Phase A / Phase B ever see.
# valid / test are reserved for combo (run_adaptive_combination.py) and never
# leak into search.
TRAIN_START, TRAIN_END = "2010-01-01", "2021-12-31"
VALID_START, VALID_END = "2022-01-01", "2022-12-31"
TEST_START, TEST_END = "2023-01-01", "2026-04-30"
EVAL_SPLIT = "train"


# Phase A: 3 probe rounds × 5 LLM edits per round, then 1 attribution call.
N_SEEDS = 50              # default `--n_seeds` for train_cf and phaseA
ROUNDS = 3                # spec-fixed (research.md)
K_PER_ROUND = 5           # spec-fixed (research.md)
K = K_PER_ROUND           # alias used by llm.propose's default arg


# Phase B: counterfactual module recombination (CF-MR).
# Breed: cross every pair of trimmed modules with nonlinear operators.
# Select: greedy by marginal incremental IC (inc_residual vs pool signal).
RECOMB_WINDOWS = (10, 20, 50)   # TsCorr window choices
K_NEW = 15                      # max number of new factors added on top of trimmed
TAU_INC = 0.002                 # stop when marginal |inc| falls below this
MAX_CANDIDATES = 500            # cap on bred candidates (stride-subsampled)


# qlib data directories (instrument -> path).
QLIB_CN = "data/qlib_data/cn_data_rolling"
QLIB_US = "data/qlib_data/us_data_qlib_latest"


# Feature / operator lookup tables used by edits._site_of and llm._vocab:
#   FEATURE_NAMES: ["$close", "$open", ...] — strings the LLM uses in prompts
#   OP_BY_NAME:    {"Add": Add, "TsMean": TsMean, ...} — class lookup for operator_replace
FEATURE_NAMES = ["$" + f.name.lower() for f in FEATURES]
OP_BY_NAME = {op.__name__: op for op in OPERATORS}


# instrument -> qlib data directory. US only when instrument == "sp500";
# everything else (csi300, csi500, ...) defaults to CN.
def qlib_path(instrument: str) -> str:
    return QLIB_US if instrument == "sp500" else QLIB_CN