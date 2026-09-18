# 反事实因子分解与模块重组进化（CF-MR）：开发计划

只能修改 `src/alpha_cf/` 和 `train_cf.py`。注释清晰良好详细，代码简洁高效。简洁！本阶段只做最小 MVP 实现，能跑通最重要！不要动不动整出几千行代码。

---

## 0. 研究定位与 Related Work 对比

### 核心论点

> 现有因子挖掘把因子当**原子**——生成、评估、选择都是整体操作。
> 我们第一次把因子**分解成功能模块**（Phase A 反事实归因），再**跨因子重组**（Phase B 模块进化），
> 用同一个量（`inc_residual`，反事实边际增量）统一"分析"与"合成"。

### Related Work 对比

| 方法 | 生成机制 | 信号/学分 | 因子理解 | 缺陷 |
|---|---|---|---|---|
| GP (gplearn) | 随机交叉+变异 | 单因子适应度 | 无 | 交叉点随机，大量无意义后代 |
| AlphaGen (KDD'23) | PPO 逐 token 生成 | pool IC (scalar) | 无 | in-sample 过拟合，无结构理解 |
| QuantFactor REINFORCE | REINFORCE | IR reward shaping | 无 | 仍是 scalar reward |
| AlphaAgent (KDD'25) | LLM 生成 + AST 去重 | 多维打分 | 无 | 无因果归因，因子是黑盒 |
| Alpha Jungle (MCTS+LLM) | LLM + 树搜索 | 多维评估 | 无 | 搜索效率低，无结构分解 |
| FactorMAD (ICAIF'25) | 多 Agent 辩论 | 辩论共识 | 无 | 无定量结构分析 |
| **Ours (CF-MR)** | **模块重组** | **反事实边际增量** | **反事实归因** | — |

### 我们的三个创新点

1. **反事实分解（Phase A）**：用最小编辑探针 + 归因，把因子分解为 load-bearing 模块。
   所有 prior work 都没有这一步——它们不知道因子"哪部分有用"。

2. **因果引导重组（Phase B）**：交叉点不是随机的（GP），而是 Phase A 验证过的功能模块边界。
   重组后的每个模块已被证明有独立预测力。

3. **边际增量选择**：用 `inc_residual`（OLS 残差 IC）做贪心选择，
   精确对齐下游 combo 的 OLS 回归需求（线性独立 + 互补）。
   不是近似，是精确对齐。

### 为什么大概率能 work

```
combo 的 OLS 回归 → 需要因子线性独立 + OOS 稳定
                      ↓
我们的适应度 = inc_residual（线性独立性）
                      ↓
完全对齐 combo 的打分机制 → 选出的因子大概率提升 combo 表现
```

- **不破坏 trimmed 池**：只加不减（解决 PPO 踢好因子的问题）
- **多样性有保证**：跨因子重组产出的结构是任何单因子变异得不到的
- **成本极低**：20 因子 → ~190 对 → ~500 候选 → 全部走缓存评估（分钟级）
- **不训练任何模型**：没有过拟合风险

---

## 1. 流水线总览

```text
INPUT  --pool_json <PPO/AlphaSAGE pool> --n_seeds N --instrument csi300
OUTPUT data/cf_logs/<ts>/
        phaseA_records.jsonl         每条 (round, kind, site, hyp, ic_f, ic_fp, inc, f')
        phaseA_attribution.jsonl     每条 {necessary, redundant, final_factor, reason}
        trimmed_pool.json            Phase A 产物（= 模块池）
        phaseA_combo.txt             combo 评估 trimmed（Phase A 单独贡献）
        final_pool.json              Phase B 产物（trimmed + 重组新增）
        final_combo.txt              combo 评估 final（最终）
```

**Phase A — 反事实分解**：每因子 3 轮探针 × 5 个最小编辑 + 1 次归因 LLM 调用 → 输出模块。
**Phase B — 模块重组进化（CF-MR）**：跨因子重组 → 反事实筛选 → 贪心选择。
**combo 评估**：train_cf.py 内部 subprocess 自动跑两次 combo。

**评估范围 = 全 train 段 2010..2021，不分段**。valid/test 只进 combo，不泄露。

---

## 2. Phase A — 反事实分解（已实现，不改）

```text
for f in seeds[:n_seeds]:
    records = []
    for round = 0 .. 2:
        for a in llm.propose(f, recent, k=K_PER_ROUND):
            fp = apply_edit(f, a)
            if fp is None or str(fp) == str(f): continue
            inc = cache.inc(fp, f)
            records.append({round, kind, site_id, inc_residual, ic_f, ic_fp, delta_ic, f'})
    attr = llm.attribute(f, records)
    f_final = parse(attr.final_factor) or f
    trimmed.append(f_final)
write trimmed_pool.json
```

**关键**：`final_factor` 就是模块。Phase A 已经把冗余剥完，trimmed 因子 = load-bearing core。
不需要额外解析 `necessary` 字段（它是散文，不可硬解析）。

---

## 3. Phase B — 模块重组进化（CF-MR）

### 只有两步：Breed → Select

```text
def run_phaseB(trimmed, cache, out_dir):
    # ===== Step 1: Breed（跨因子重组，产出候选）=====
    candidates = []
    for fi, fj in combinations(trimmed, 2):       # C(20,2) = 190 对
        for op in [Add, Sub, Div, Mul]:            # 4 种二元组合
            g = op(fi, fj)
            if g.is_featured: candidates.append(g)
        for w in [10, 20, 50]:                     # 3 种滚动组合
            g = TsCov(fi, fj, w)
            if g.is_featured: candidates.append(g)
    # 去重 + 过滤与池内高相关（|corr| > 0.95）
    candidates = dedupe_and_filter(candidates, trimmed, cache)
    # 预期：190×7 = 1330 → 去重/过滤后 ~300-500 个

    # ===== Step 2: Select（反事实边际增量贪心选择）=====
    pool_signal = ic_weighted_combine(trimmed)     # 当前池的组合信号
    scored = []
    for g in candidates:
        inc = incremental_ic(pool_signal, g, ret)  # 边际增量（复用 reward.py）
        ic  = cache.evaluate(g).ic                 # 基础预测力
        scored.append((g, inc, ic))
    scored.sort(key=lambda x: x[1], reverse=True)  # 按 inc 排序

    new_factors = []
    for g, inc, ic in scored:
        if len(new_factors) >= K_NEW: break        # 最多加 15 个
        if inc < TAU_INC: break                    # 边际增量太小，停止
        new_factors.append(g)
        pool_signal = update(pool_signal, g)       # 正交化：加入后更新组合信号

    final_pool = trimmed + new_factors             # 只加不减
    write(final_pool, "final_pool.json")
```

### 为什么只要两步

- **Breed** 保证多样性：跨因子重组产出的结构是单因子变异得不到的
- **Select** 保证质量：`inc_residual` 精确度量"加入后池子多了多少独立信息"
- 不需要额外的"鲁棒性检验"步骤——`inc_residual` 本身就过滤了噪声
  （噪声因子的残差 IC 在 train 上不稳定，多次评估会暴露）

### 模块来源说明

| 来源 | 内容 | 用法 |
|---|---|---|
| `final_factor`（trimmed_pool） | Phase A 剥完冗余的 load-bearing core | **直接作为模块** |
| `necessary` 字段 | 散文描述，不可解析 | 仅用于日志/分析 |
| `phaseA_records` | 结构化 (site, inc, delta) | 可选：辅助判断模块质量 |

**MVP 决策**：模块 = `final_factor`。不需要额外解析。

---

## 4. 学术论证

### 论点 1：反事实归因 = 结构化信用分配

标准因子挖掘的 reward 是 scalar（IC/ICIR）。我们把它分解为结构级信用：

$$\text{credit}(f, \text{subtree}_i) = R(f) - R(f \setminus \text{subtree}_i)$$

这就是 `inc_residual`。它回答："这个子树独立贡献了多少预测力？"

### 论点 2：信用引导的重组 > 随机重组

GP 的 crossover 随机选交叉点，大量后代无意义。我们只在因果边界（load-bearing 模块的接口）做重组：
- 重组后的每个模块已被证明有独立预测力（Phase A 验证过）
- 后代的预期质量远高于随机搜索

### 论点 3：边际增量选择 = 对齐下游评估

Combo 的 OLS 回归需要线性独立的因子。`inc_residual` 恰好度量
"加入 g 后，池子多了多少线性独立信息"。这是精确对齐，不是近似。

### 论点 4：与 Shapley 值的联系

`inc_residual` 是 Shapley 值的单步近似。贪心选择是子模函数最大化的
greedy 近似，有 $(1-1/e)$ 近似比的理论保证。

---

## 5. 文件树

```text
train_cf.py                         # 入口：argparse → Phase A → combo → Phase B → combo
src/alpha_cf/
  __init__.py                       # 包标记
  config.py                         # 日期 + Phase A/B 超参
  types.py                          # Site / EditAction / CFRecord
  edits.py                          # 4 类编辑 + apply_edit + enumerate_sites（不动）
  reward.py                         # evaluate/inc/mean_cs_corr/RewardCache（不动）
  pool.py                           # load_pool + parse_expr（不动）
  llm.py                            # propose() + attribute()（不动）
  phaseA.py                         # 3 轮探针 + 归因（不动）
  phaseB.py                         # ★ 重写：模块重组进化（~100 行）
```

复用、只 import 不改：

```text
src/alphagen/data/expression.py      Expression AST + Add/Sub/Mul/Div/TsCov 等
src/alphagen/data/tree.py            ExpressionParser
src/alphagen/config.py               OPERATORS / FEATURES / DELTA_TIMES / CONSTANTS
src/alphagen_qlib/calculator.py      IC / RankIC
src/alphagen_qlib/stock_data.py      StockData
run_adaptive_combination.py          combo（subprocess 调用，不修改）
```

**不再需要**：`AlphaPool`、`AlphaEnv`、`MaskablePPO`、`LSTMSharedNet`（PPO 全部移除）。

---

## 6. 超参数（config.py）

```text
# Phase A（不变）
N_SEEDS          = 50
ROUNDS           = 3
K_PER_ROUND      = 5
EVAL_SPLIT       = "train"

# Phase B（新）
RECOMB_WINDOWS   = [10, 20, 50]     # TsCov 的窗口选择
K_NEW            = 15                # 最多新增因子数
TAU_INC          = 0.002             # 边际增量下限（低于此不值得加）
CORR_FILTER      = 0.95              # 与池内因子相关 > 此值则过滤

# Combo（不变）
COMBO_ARGS       = --train_end_year 2021 --instruments csi300 \
                    --n_factors 20 \
                    --threshold_ric 0.015 \
                    --threshold_ricir 0.15
```

---

## 7. phaseB.py 实现要点（~100 行）

```text
from itertools import combinations
from alphagen.data.expression import Add, Sub, Mul, Div, TsCov

def run_phaseB(trimmed, cache, out_dir):
    # Step 1: Breed
    candidates = []
    for fi, fj in combinations(trimmed, 2):
        for op in [Add, Sub, Div, Mul]:
            g = op(fi, fj)
            if g.is_featured: candidates.append(g)
        for w in RECOMB_WINDOWS:
            g = TsCov(fi, fj, w)
            if g.is_featured: candidates.append(g)

    # 去重（str 相同）+ 过滤与池内高相关
    seen = {str(f) for f in trimmed}
    unique = []
    for g in candidates:
        if str(g) in seen: continue
        seen.add(str(g))
        if any(abs(cache.mean_cs_corr(g, f)) > CORR_FILTER for f in trimmed):
            continue
        unique.append(g)

    # Step 2: Select
    # 构建池组合信号：等权平均（简单有效）
    pool_alphas = [cache.alpha(f) for f in trimmed]
    pool_signal = mean(pool_alphas)  # [n_days, n_stocks]

    scored = []
    for g in unique:
        inc = cache.inc(g, pool_signal)  # 需要新增：inc vs tensor（不是 vs expr）
        ic = cache.evaluate(g).ic
        scored.append((g, inc, ic))
    scored.sort(key=lambda x: x[1], reverse=True)

    new_factors = []
    for g, inc, ic in scored:
        if len(new_factors) >= K_NEW or inc < TAU_INC: break
        new_factors.append(g)
        # 更新 pool_signal（加入 g 的 alpha）
        pool_signal = update_pool_signal(pool_signal, cache.alpha(g))

    final = trimmed + new_factors
    write final_pool.json
```

### 需要新增的 reward.py 接口

```text
# 现有：cache.inc(f_prime: Expression, f: Expression) -> float
# 新增：cache.inc_vs_signal(g: Expression, signal: Tensor) -> float
#   对 g 的 alpha 关于 signal 做 OLS 残差，返回残差 IC
#   复用 incremental_ic() 的核心逻辑，只是 parent 从 expr 变成 tensor
```

---

## 8. train_cf.py 改动

```text
def train(args):
    out_dir = ...
    cache = RewardCache(make_calculator(args.instrument, "train"))

    # Phase A（不变）
    seeds = load_pool(args.pool_json, ...)
    trimmed = run_phaseA(seeds, cache, out_dir)
    _run_combo(trimmed_pool.json, phaseA_combo.txt)

    # Phase B（新：模块重组）
    run_phaseB(trimmed, cache, out_dir)

    # Combo #2（不变）
    _run_combo(final_pool.json, final_combo.txt)
```

---

## 9. 边界 & 退化

| 情形 | 行为 |
|---|---|
| `TsCov(fi, fj, w)` 不 featured | 跳过 |
| 候选全部被 corr 过滤 | 降低 CORR_FILTER 到 0.99 重试 |
| `inc < TAU_INC` 对所有候选 | final_pool = trimmed（不加新因子） |
| `cache.alpha(g)` 返回 None | 跳过该候选 |
| `pool_signal` 全 NaN | 回退等权 → 跳过 inc 计算，按 ic 排序 |

---

## 10. 实验协议

1. **快速验证**（~10 min）：`--n_seeds 5`，看 Phase B 是否能产出合法候选
2. **全量**（~2 h）：`--n_seeds 50`，对比 phaseA_combo vs final_combo
3. **目标**：Test IC > 0.0412 / ICIR > 0.2435（PPO baseline）
4. **消融**：
   - w/o 重组（只用 trimmed）→ 证明 Phase B 有增量
   - 随机重组（不经 Phase A）→ 证明归因引导有价值
   - 按 raw IC 选（不用 inc）→ 证明边际增量选择有价值

---

## 11. 用法

```bash
python train_cf.py --pool_json <PPO pool> --n_seeds 50 --instrument csi300
# 内部自动跑两次 combo；对比 phaseA_combo.txt vs final_combo.txt 看 Phase B 增量
```
