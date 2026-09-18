# 反事实因子分解与模块重组进化（CF-MR）

按 `research.md` 的迭代流程：把 AlphaSAGE / PPO 种子因子先做反事实归因分解，再用跨因子模块重组进化，得到比起点 pool 更强的因子池。

**核心思想**：现有因子挖掘把因子当**原子**（生成、评估、选择都是整体操作）。我们第一次把因子**分解成功能模块**（Phase A 反事实归因），再**跨因子重组**（Phase B 模块进化），用同一个量（`inc_residual`，反事实边际增量）统一"分析"与"合成"。

**目标**：让 pipeline 的 test 指标明显优于 PPO baseline（Test IC 0.0412 / ICIR 0.2435）。

## 当前结果（CSI300，pool_20 种子，2023–2026-04 test）

| 方法 | Test IC | Test ICIR | Test RIC | Test RET_SR |
|---|---|---|---|---|
| PPO baseline | 0.0412 | 0.2435 | 0.0594 | 0.7233 |
| Phase A only (trimmed) | 0.0410 | 0.2410 | 0.0585 | 0.7127 |
| **Phase A + Phase B (CF-MR)** | **0.0612** | **0.3508** | **0.0729** | **0.9955** |

Phase B 在 Phase A 基础上带来 **+49% IC / +46% ICIR** 的增量。

## Pipeline 概览

```text
AlphaSAGE / PPO pool_*.json
        │
        ▼
   Phase A — 反事实归因分解  (phaseA.py)
   ──────────────────────────────────
   对每个 seed f：
     - 3 轮 × K=5 个最小编辑（feature/operator/window/subtree_delete）
     - 每轮把上一轮 records 给 LLM，引导它覆盖未试探的 (kind, site) 对
     - 编辑始终对 FROZEN 原始 f 做，不与 f' 复合
     - 1 次归因 LLM 调用，看 4 维指标（inc_residual / ic_f / ic_fp / delta_ic，inc 排第一）
       输出 {necessary, redundant, final_factor, reason}
        │
        ▼
   trimmed_pool.json  ──→  run_adaptive_combination.py ──→  phaseA_combo.txt
        │
        ▼
   Phase B — 反事实模块重组进化（CF-MR）  (phaseB.py)
   ──────────────────────────────────
   Breed:   每个模块对 (fi, fj) 用非线性算子重组（Div / Mul / TsCorr）
            → 候选集 C（~500 个）
   Select:  贪心按 |inc(g, pool_signal)| 选择最多 15 个
            pool_signal = 等权平均 ranked alpha
            每选 1 个就正交化更新 pool_signal
            trimmed 因子永不淘汰（只加不减）
        │
        ▼
   final_pool.json  ──→  run_adaptive_combination.py ──→  final_combo.txt
```

回测严格走 train（2010-01-01–2021-12-31）全段，不分段、不切 regime。valid / test 仅 `run_adaptive_combination.py` 用，全程不进入 Phase A / Phase B 的搜索。

## 与 Related Work 的差异

| 方法 | 生成机制 | 信号/学分 | 因子理解 | 缺陷 |
|---|---|---|---|---|
| GP (gplearn) | 随机交叉+变异 | 单因子适应度 | 无 | 交叉点随机，大量无意义后代 |
| AlphaGen (KDD'23) | PPO 逐 token 生成 | pool IC (scalar) | 无 | in-sample 过拟合，无结构理解 |
| QuantFactor REINFORCE | REINFORCE | IR reward shaping | 无 | 仍是 scalar reward |
| AlphaAgent (KDD'25) | LLM 生成 + AST 去重 | 多维打分 | 无 | 无因果归因，因子是黑盒 |
| Alpha Jungle (MCTS+LLM) | LLM + 树搜索 | 多维评估 | 无 | 搜索效率低，无结构分解 |
| FactorMAD (ICAIF'25) | 多 Agent 辩论 | 辩论共识 | 无 | 无定量结构分析 |
| **CF-MR (本文)** | **模块重组** | **反事实边际增量** | **反事实归因** | — |

三个创新点：
1. **反事实分解**：用最小编辑探针 + LLM 归因，把因子分解为 load-bearing 模块（所有 prior work 都没有这一步）。
2. **因果引导重组**：交叉点不是随机的（GP），而是 Phase A 验证过的功能模块边界。
3. **边际增量选择**：用 `inc_residual`（OLS 残差 IC）做贪心选择，精确对齐下游 combo 的 OLS 回归需求。

## 跑

```bash
# 主训练脚本（默认走完整 Phase A → combo → Phase B → combo）
python train_cf.py \
  --pool_json data/ppo_logs/pool_20/ppo_csi300_20_0-*/200704_steps_pool.json \
  --n_seeds 50 --seed_start 0 --instrument csi300

# 单独跑 Phase A（debug / 只想看 trimmed pool 时；默认 50 seeds）
#   产物在 <out_dir>/：phaseA_records.jsonl / phaseA_attribution.jsonl /
#   illegal.jsonl / trimmed_pool.json
python -m alpha_cf.phaseA   --pool_json data/ppo_logs/pool_20/ppo_csi300_20_0-*/200704_steps_pool.json   --instrument csi300

# 单独跑 Phase B（已 trimmed 直接模块重组）
python -m alpha_cf.phaseB \
  --trimmed_pool_json data/cf_logs/<phaseA_run>/trimmed_pool.json

# 跳过 Phase B，Phase A 产物直接当 final（消融对比 phaseA vs phaseA+B）
python train_cf.py \
  --pool_json data/ppo_logs/pool_20/ppo_csi300_20_0-*/200704_steps_pool.json \
  --skip_phaseB

# 单独跑 combo 评估任意池（GPU，1–2 分钟）；train_cf.py 内部也是这个命令
python run_adaptive_combination.py \
  --expressions_file data/cf_logs/phaseA_20260917_100246/trimmed_pool.json \
  --train_end_year 2021 --instruments csi300 --n_factors 20 \
  --threshold_ric 0.015 --threshold_ricir 0.15
```

## 可调参数（CLI）

`train_cf.py`：

| 参数 | 默认 | 含义 |
|---|---|---|
| `--pool_json` | ""（必填） | AlphaSAGE / PPO 种子池 JSON；Phase A 入口 |
| `--n_seeds` | 50 | 取前 n 条种子 |
| `--seed_start` | 0 | 跳过前 seed_start 条 |
| `--instrument` | `csi300` | `csi300` / `sp500` |
| `--skip_phaseB` | off | 跳过 Phase B；`trimmed_pool.json` 复制为 `final_pool.json` |
| `--out_dir` | 自动时间戳 | 输出目录 |
| `--seed` | 0 | Python 随机种子 |

`python -m alpha_cf.phaseA`（单独跑 Phase A）：

| 参数 | 默认 | 含义 |
|---|---|---|
| `--pool_json` | ""（必填） | 同上 |
| `--n_seeds` | 50 | 同上 |
| `--seed_start` | 0 | 跳过前 seed_start 条 |
| `--instrument` | `csi300` | `csi300` / `sp500` |
| `--out_dir` | 自动时间戳 | 输出目录 |

`python -m alpha_cf.phaseB`（单独跑 Phase B）：

| 参数 | 默认 | 含义 |
|---|---|---|
| `--trimmed_pool_json` | ""（必填） | Phase A 输出的 `trimmed_pool.json` |
| `--instrument` | `csi300` | `csi300` / `sp500` |
| `--out_dir` | 自动时间戳 | 输出目录 |

## 超参数（`src/alpha_cf/config.py`）

| 常量 | 值 | 含义 |
|---|---|---|
| `TRAIN_START` / `TRAIN_END` | 2010-01-01 / 2021-12-31 | Phase A / Phase B 唯一评估窗口 |
| `VALID_*` / `TEST_*` | 2022 / 2023–2026 | 仅 `run_adaptive_combination.py` 用 |
| `EVAL_SPLIT` | "train" | RewardCache 唯一窗口 |
| `N_SEEDS` | 50 | 默认种子数 |
| `ROUNDS` | 3 | Phase A 每因子探针轮数 |
| `K_PER_ROUND` | 5 | 每轮 LLM 编辑提案数 |
| `RECOMB_WINDOWS` | (10, 20, 50) | Phase B TsCorr 窗口选项 |
| `K_NEW` | 15 | Phase B 最多新增因子数 |
| `TAU_INC` | 0.002 | Phase B 边际增量下限（低于此停止） |
| `MAX_CANDIDATES` | 500 | Phase B 候选因子数上限 |
| `QLIB_CN` / `QLIB_US` | cn_data_rolling / us_data_qlib_latest | qlib 二进制目录 |

auto-combo 参数（`train_cf.py:COMBO_ARGS`）：
`--train_end_year 2021 --instruments csi300 --n_factors 20 --threshold_ric 0.015 --threshold_ricir 0.15`

## 产物（`--out_dir`）

| 文件 | 内容 |
|---|---|
| `phaseA_records.jsonl` | 每个 (round, action) 一条，含 inc_residual/ic_f/ic_fp/delta_ic/f_prime |
| `phaseA_attribution.jsonl` | 每个 seed 一条：`{f, necessary, redundant, final_factor, reason}` |
| `illegal.jsonl` | LLM 提案里 `T(f,a)` 拒绝的编辑（`apply_edit` 返回 None） |
| `trimmed_pool.json` | Phase A 输出：`{"exprs":[...], "weights":[]}` |
| `phaseA_combo.txt` | trimmed 池 auto-combo stdout/stderr/exit |
| `final_pool.json` | Phase B 输出（trimmed + 重组新增） |
| `phaseB_selection.jsonl` | Phase B 每一步贪心选中：`{f, ic, inc}` |
| `final_combo.txt` | final 池 auto-combo stdout/stderr/exit |

## 目录

```text
train_cf.py                  # CLI 入口；Phase A → combo → Phase B → combo
src/alpha_cf/
  __init__.py
  config.py                  # 日期 / ROUNDS / K_PER_ROUND / Phase B 超参
  types.py                   # Site / EditAction / CFRecord dataclass
  pool.py                    # 读 AlphaSAGE / PPO JSON → Expression
  edits.py                   # T(f,a) 四种最小编辑 + 合法性闸（核心轮子）
  reward.py                  # RewardCache: IC / RankIC / inc / inc_vs_signal / mean_cs_corr
  llm.py                     # OpenAI-compatible：propose + attribute
  phaseA.py                  # 3 轮探针 + 1 次归因
  phaseB.py                  # 模块重组（Breed + Select 贪心）
```

复用、只 import 不改：

```text
alphagen.data.expression     Expression AST + Add/Sub/Mul/Div/TsCorr 等
alphagen_qlib.calculator     IC / RankIC
alphagen_qlib.stock_data     StockData
run_adaptive_combination.py  combo（subprocess 调用，不修改）
```

**已移除**：AlphaPool / AlphaEnv / MaskablePPO / LSTMSharedNet / sb3-contrib（PPO 全部删除）。

## 边界 & 退化

| 情形 | 行为 |
|---|---|
| `llm.attribute` final_factor 解析失败 | 回退原种子 |
| `records=[]`（3 轮全 illegal） | attribute 仍调一次，期望 LLM 原样返回 f |
| `f' == f` | 跳过，不记 record |
| `Div/Mul/TsCorr` 重组产出 unfeatured | 跳过，不入候选 |
| 候选全部 |inc| < `TAU_INC` | final_pool = trimmed，不加新因子 |
| `cache.alpha(g)` 返回 None | 跳过该候选 |
| subprocess combo 失败 / 超时（3600s） | 写 TIMEOUT / spawn failed 占位，主流程继续 |
| `parse_expr` 抛异常 | try/except，失败回退原种子 |

## 约束（重申 research.md）

1. **不用 valid/test**：所有真回测只走 train 段（2010-2021 全段，不切 search / adopt / regime）。
2. **代码保持简洁**：Phase A 单 scalar reward 也不要，LLM 归因一次性看 4 维指标；Phase B 用 `inc_residual` 当唯一选择准则。
3. **只改 `src/alpha_cf/` 和 `train_cf.py`**：AlphaSAGE / GFN / baseline 等基础设施一律不动。

## 设计依据（dev_plan.md 节选）

**反事实归因 = 结构化信用分配**：
$$\text{credit}(f, \text{subtree}_i) = R(f) - R(f \setminus \text{subtree}_i) = \text{inc\_residual}(f \setminus \text{subtree}_i, f)$$

**信用引导的重组 > 随机重组**：GP 随机选交叉点。我们只在因果边界（load-bearing 模块接口）做重组——重组后的每个模块已被证明有独立预测力。

**边际增量选择 = 对齐下游评估**：combo 的 OLS 回归需要线性独立的因子。`inc_residual` 恰好度量"加入 g 后，池子多了多少线性独立信息"——精确对齐，不是近似。
