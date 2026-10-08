# AlphaCME: Alpha Mining via Counterfactual Mechanism Diagnosis and Structural Evolution

[](#alphacme-alpha-mining-via-counterfactual-mechanism-diagnosis-and-structural-evolution)



[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE) [![Python 3.11+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/release/python-3100/) [![PDM](https://img.shields.io/badge/PDM-managed-blueviolet)](https://pdm-project.org)

This repository contains the official implementation of **AlphaCME**, a closed-loop alpha mining pipeline that uses structural counterfactual reasoning to attribute predictive mechanisms inside factors and evolves new factors at the mechanism level.

🎯 Overview
-----------

[](#-overview)

AlphaCME runs six stages per evolutionary round on the Train split:

1. **Initial Pool Construction** — load a frozen 50-seed pool, preserve order, no re-selection.
2. **Counterfactual Diagnosis** — the LLM decomposes each parent factor into up to 5 mechanisms with a counterfactual intervention and an explicit hypothesis.
3. **Mechanism Evidence** — measure three pieces of evidence on full Train (`Δcf`, `C_pool`, `dist`). Evidence is *not* a disposal label.
4. **Mechanism Memory** — per-chain persistence of parent understanding, evidence, and offspring decisions.
5. **Direct Offspring Generation** — the same call returns `updated_understanding` plus 0–5 offspring grounded in this round's evidence.
6. **Parent Replacement** — full-rank equivalence groups collapse to the simplest AST; at most one offspring replaces a parent.

The final round's pool is forwarded verbatim to `run_adaptive_combination.py` — no separate final selection.

👉 Quick Start
--------------

[](#-quick-start-of-alphacme)

### Dependency Installation

[](#dependency-installation)

```bash
pdm install
```

PDM pins Python 3.11 and resolves all expression, data, and OpenAI-compatible client dependencies.

### Dataset Retrieval

[](#dataset-retrieval)

Run the following command to retrieve data from Microsoft Qlib:

```bash
wget https://github.com/chenditc/investment_data/releases/latest/download/qlib_bin.tar.gz
mkdir -p ~/.qlib/qlib_data/cn_data
tar -zxvf qlib_bin.tar.gz -C ~/.qlib/qlib_data/cn_data --strip-components=1
rm -f qlib_bin.tar.gz
```

AlphaCME reads `data/qlib_data/cn_data_rolling` (CN) and `data/qlib_data/us_data_qlib_latest` (US) by default; override with `--qlib-path` or the `QLIB_PATH_CN` / `QLIB_PATH_SP500` env vars.

### Run AlphaCME

[](#run-alphacme)

#### Factor Evolution

1. Configure your model in `.env` (`OPENAI_API_KEY`, `OPENAI_BASE_URL`, `OPENAI_MODEL_NAME`).
2. Choose the instrument via `--instrument` (default `csi300`).
3. Run the pipeline:

```bash
python train_cf.py --instrument csi300 --rounds 10
```

Logs are saved under `data/cf_logs/<timestamp>_<instrument>_<seed>_<model_name>`, using `OPENAI_MODEL_NAME`; `/` and `\` in model names become `_`.

#### Factor Combination

The final round's `pool_<rounds>.json` is forwarded to `run_adaptive_combination.py`:

```bash
python run_adaptive_combination.py \
    --expressions_file data/cf_logs/<run>/pool_<rounds>.json \
    --instruments csi300 \
    --train_end_year 2021 \
    --seed 0 \
    --use_weights True
```

