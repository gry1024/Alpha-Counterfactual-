# AlphaCREAM: Counterfactual Reasoning and Evidence-Guided Evolution for Alpha Mining

[](#alphacream-counterfactual-reasoning-and-evidence-guided-evolution-for-alpha-mining)



[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE) [![Python 3.11+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/release/python-3100/) [![PDM](https://img.shields.io/badge/PDM-managed-blueviolet)](https://pdm-project.org)

This repository contains the official implementation of **AlphaCREAM**, a closed-loop alpha mining pipeline that uses structural counterfactual reasoning to attribute predictive mechanisms inside factors and evolves new factors at the mechanism level.

🎯 Overview
-----------

[](#-overview)

AlphaCREAM runs six stages per evolutionary round on the Train split:

1. **Initial Pool Construction** — load a frozen 50-seed pool, preserve order, no re-selection.
2. **Counterfactual Reasoning** — the LLM formulates up to 5 mechanism hypotheses, constructs counterfactual structural alternatives, and predicts their effects before evaluation.
3. **Mechanism Evidence** — measure three pieces of evidence on full Train (`Δcf`, `C_pool`, `dist`). Evidence is *not* a disposal label.
4. **Mechanism Memory** — per-chain persistence of parent understanding, evidence, and offspring decisions.
5. **Direct Offspring Generation** — the same call returns `updated_understanding` plus 0–5 offspring grounded in this round's evidence.
6. **Parent Replacement** — full-rank equivalence groups collapse to the simplest AST; at most one offspring replaces a parent.

The final round's pool is forwarded verbatim to `run_adaptive_combination.py` — no separate final selection.

👉 Quick Start
--------------

[](#-quick-start-of-alphacream)

### Dependency Installation

[](#dependency-installation)

```bash
pdm install
```

PDM pins Python 3.11 and resolves all expression, data, and OpenAI-compatible client dependencies.

### Dataset Retrieval

[](#dataset-retrieval)

Download CN data in Qlib format from [investment_data](https://github.com/chenditc/investment_data), and extract it as the base dataset:

```bash
wget https://github.com/chenditc/investment_data/releases/latest/download/qlib_bin.tar.gz
mkdir -p data/qlib_data/cn_data
tar -zxvf qlib_bin.tar.gz -C data/qlib_data/cn_data --strip-components=1
rm -f qlib_bin.tar.gz
```

Then fetch daily market data from BaoStock, adjust prices and volume, compute VWAP, and export Qlib binary files to `data/qlib_data/cn_data_rolling`. The script also copies index constituent files from the base dataset and extends their latest end dates; it does not refresh historical membership changes.

```bash
PYTHONPATH=src python -m data_collection.fetch_baostock_data
```

CN runs use `data/qlib_data/cn_data_rolling`; US runs require a separately prepared Qlib dataset at `data/qlib_data/us_data_qlib_latest`. For `train_cf.py`, override the path with `--qlib-path` or `QLIB_PATH_CN` / `QLIB_PATH_SP500`.

### Run AlphaCREAM

[](#run-alphacream)

#### Factor Evolution

1. Configure your model in `.env` (`OPENAI_API_KEY`, `OPENAI_BASE_URL`, `OPENAI_MODEL_NAME`).
2. Choose the instrument via `--instrument` (default `csi300`).
3. Run the pipeline:

```bash
python train_cf.py --instrument csi300 --rounds 10
```

Logs are saved under `data/cf_logs/<timestamp>_<instrument>_<seed>_<model_name>`, using `OPENAI_MODEL_NAME`; `/` and `\` in model names become `_`.

#### Factor Combination

Initialization (round 0) and every completed round evaluate the saved pool with frozen Train weights on Validation (2022) and Test (2023-05-01–2026-04-30). Metrics go to `pool_<round>.metrics.json` and `res.txt`. `adaptive_combination.png` tracks both splits across rounds; `profit_curves.png` shows their latest cumulative returns. Evaluation does not affect search or pool updates.

To re-evaluate a saved pool:

```bash
python run_adaptive_combination.py \
    --expressions_file data/cf_logs/<run>/pool_<rounds>.json \
    --instruments csi300 \
    --train_end_year 2021 \
    --seed 0 \
    --use_weights True
```

## Baseline 运行方式


**Alpha158：固定等权主基线**
```bash
PYTHONPATH=src:. python src/alpha158/run.py --instruments "$market"
```

仅保留训练方向等权，输出到 `data/alpha158_logs/<market>_equal_weight/`；三个市场主表、收益曲线及论文统一采用等权结果；该协议为查看 Test 对照后的作者指定变更。详见 [Alpha158 说明](src/alpha158/README.md)。

**LightGBM（AlphaSAGE 配置）：**

```bash
PYTHONPATH=src:. python train_lightgbm.py --instruments "$market" --threads 2
```

使用 OHLCV 日变化率的 1–59 日滞后特征（295 维），Train / Validation 标签均做日截面标准化，按 Validation RMSE 选最佳轮数后固定预测；Train 为 2015–2021，Validation 为 2022，Test 为 2023-05-01 至 2026-04-30。输出模型、指标、收益及日期到 `data/lightgbm_logs/$market/`。

**XGBoost（AlphaSAGE 配置）：**

```bash
PYTHONPATH=src:. python train_xgboost.py --instruments "$market" --threads 2
```

与 LightGBM 共用 295 维特征、标签标准化和数据划分，显式训练 100 轮，并使用 Validation RMSE 最优轮数预测。输出到 `data/xgboost_logs/$market/`；可加 `--feature-cache data/lightgbm_logs/$market/features.npy` 复用已有特征。

**AlphaGen（PPO）：**

```bash
python train_ppo.py --instruments "$market" --pool 20 --steps 200000 --seed 0
```

因子池：`data/ppo_logs/pool_20/<run>/<run>/*_steps_pool.json`。

**AlphaSAGE（GFlowNet）：**

```bash
python train_gfn.py --instrument "$market" 
    --seed 0 \
    --pool_capacity 50 \
    --n_episodes 10000 \
    --log_freq 500 \
    --update_freq 64 \
    --encoder_type gnn \
    --entropy_coef 0.01 \
    --entropy_temperature 1.0 \
    --mask_dropout_prob 1.0 \
    --ssl_weight 1.0 \
    --nov_weight 0.3 \
    --weight_decay_type linear \
    --final_weight_ratio 0.0
```

因子池：`data/gfn_logs/pool_50/<run>/pool_9999.json`。

**AlphaPROBE：** 在 `.env` 配置 `OPENAI_API_KEY`、`OPENAI_BASE_URL`；

```bash
python train_knowledge.py --instruments "$market" --seed 0
```

因子池：`data/knowledge_logs/pool_50/<run>/pool_<iteration>.json`。

**已有因子池回测：** 将 `pool` 换成要评价的 JSON 路径。

```bash
python run_adaptive_combination.py \
    --expressions_file "$pool" \
    --instruments "$market" \
    --seed 0 \
    --use_weights True \
    --n_factors 20 
```

AlphaGen、AlphaCREAM需使用`--use_weights True`。
输出收益到因子池同目录的 `ret_s.npy` 。Test 为 2023-05-01 至 2026-04-30。

**绘制已登记实验的收益曲线：**

```bash
python plot_profit_curve.py
```

需在 `plot_profit_curve.py` 登记 `ret_s.npy` 路径

