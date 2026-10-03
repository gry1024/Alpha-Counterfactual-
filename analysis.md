# AlphaCF Pipeline 运行分析

**分析对象**: `data/cf_logs/20261003_011950_475394_csi300_0`
**运行时间**: 2026-10-03 01:19 ~ 04:50（约 3.5 小时，11 轮）
**任务**: 日志分析（仅分析缺陷，不改代码）

---

## 0. 核心结论一览

| 指标 | 本次 (NEW) | 上次 OLD AlphaCF (10/2) | AlphaSAGE | AlphaPROBE | AlphaGen |
|---|---|---|---|---|---|
| **Sharpe (test)** | **0.29** | 0.62 | 0.69 | 0.62 | 0.77 |
| 年化 (test) | 4.83% | 9.96% | 11.31% | 9.32% | 13.68% |
| 最终收益 | 15.42% | 31.77% | 36.09% | 29.72% | 43.65% |
| 最大回撤 | 31.67% | 33.80% | 27.38% | 22.28% | 26.88% |
| 训练 utility | 0.062 → 0.086 | 0.103 → **0.037** | — | — | — |

**一句话总结**: 训练期 utility 提升了 39%（0.062→0.086），但测试期 Sharpe 仅 **0.29**，是所有基线模型的一半。强 overfit 信号 + 结构性偏置。

---

## 1. 致命缺陷：父代选择结构性偏置

**位置**: `src/alpha_cf/trainer.py:101-110`

```python
def select_parents(self):
    ranked = sorted(self.pool.exprs, key=lambda e: self.pool.evaluate(e)["reward"], reverse=True)
    selected = ranked[:count // 2]              # top 5 by reward
    remaining = ranked[count // 2:]              # 后 45 个
    selected += sorted(remaining, key=lambda e: self.visits[str(e)])[:count - len(selected)]
```

### 1.1 现象验证

10 轮 × 10 父代 = 100 次 LLM 父代选择：

| | 数量 |
|---|---|
| LLM 收到正 IC 父代 | **100 / 100** |
| LLM 收到负 IC 父代 | **0 / 100** |

14 个负 IC 因子（IC ∈ [-0.066, -0.028]）**从未被尝试替换**：

```
TsCorr($volume,$vwap,30)                                  IC=-0.0666
Div(TsStd($volume,30),Add(TsMean($volume,30),0.0001))     IC=-0.0556
TsCov(Div(Sub($close,$open),$open),Div(Sub($volume,Ref($volume,1)),Ref($volume,1)),30)  IC=-0.0640
TsSkew(TsWMA(Div(Pow(5.0,SLog1p($vwap)),0.01),50),40)     IC=-0.0478
Log(TsStd(Mul(0.01,$volume),50))                           IC=-0.0496
... 共 14 个
```

### 1.2 后果

- **Round 0 → Round 10 这 14 个因子集完全不变**（用集合比较验证）
- 训练期 frac_pos 始终锁定在 0.72（36 / 50）
- Pool 永远有 28% 的"毒因子"在等权组合中拖低 utility

### 1.3 根本原因

`select_parents` 完全忽略 pool 后半区（reward 较低的部分）。`self.visits` 只在选中时 +1，所以负 IC 因子 visits 永远是 0，永远不会被 "least visited" 选上。

---

## 2. 关键配置变更（args.json）

对比本次与之前两次运行：

| 参数 | 本次 (NEW) | 10/1 working | 10/2 broken |
|---|---|---|---|
| alpha (\|R\| 权重) | **0.4** | 1.0 | 1.0 |
| beta (C_pool 权重) | **0.3** | 1.0 | 1.0 |
| gamma (多样性) | 0.2 | 0.2 | 0.2 |
| cost_weight | 0.1 | 0.1 | 0.1 |
| pool_0 utility | **0.062** | 0.109 | 0.103 |
| pool_final utility | 0.086 | 0.111 | **0.037** |

### 2.1 归一化后权重对比

```
NEW (α=0.4, β=0.3, γ=0.2, cost=0.1):
  |R|=40%, C_pool=30%, 多样性=20%, 换手=10%

OLD (α=1.0, β=1.0, γ=0.2, cost=0.1):
  |R|=43%, C_pool=43%, 多样性=9%, 换手=4%
```

### 2.2 影响分析

1. **pool_0 起点断崖下跌** 0.109 → 0.062（同 start_pool.json）
2. 新配置略偏多样性/换手成本，但**\|R\| 和 C_pool 都被削弱**
3. 上次 (10/1) 用 α=β=1.0 训练 utility 涨到 0.111 后稳定；本次配置可能限制顶级 \|R\| 因子的选择压力

---

## 3. 边际效用快速衰减

```
R0 -> R1:  +0.00690  (+11.1%)   ← 起点跳跃
R1 -> R2:  +0.00266  (+3.8%)
R2 -> R3:  +0.00178  (+2.5%)
R3 -> R4:  +0.00075  (+1.0%)
R4 -> R5:  -0.00062  (-0.8%)    ← 首次回归
R5 -> R6:  +0.00021  (+0.3%)
R6 -> R7:  +0.00283  (+3.8%)
R7 -> R8:  +0.00640  (+8.3%)    ← 异常 spike（4 个低 R 父代被替换）
R8 -> R9:  -0.00024  (-0.3%)    ← 再次回归
R9 -> R10: +0.00256  (+3.1%)
```

**两个 round 实际回退了 utility**：Round 5 和 Round 9 替换出的 child 拉低了 pool 表现。说明新引入的因子有的是**噪声**。

Round 8 的 spike 来源：4 个低 reward（0.02-0.07）父代被替换掉，恰好清掉了边缘因子。

---

## 4. LLM 证据质量差（memory.json）

480 条 counterfactual 记录统计：

| 指标 | 正向比例 | 平均值 |
|---|---|---|
| delta_cf > 0 | **19%** | -0.01231 |
| pool_credit > 0 | **13%** | -0.00258 |
| \|delta_cf\| < 1e-3 | 19% | (无信号) |

**这意味着 LLM 80%+ 的"改进建议"实际是噪声**。Memory 里堆积了大量无效证据，下一轮 evolve 时仍被传给 LLM 作为参考，进一步误导生成方向。

### 4.1 反向信号分布

```
delta_cf:   pos=91 (19.0%),  neg=389 (81.0%),  mean=-0.01231
pool_credit: pos=62 (12.9%),  neg=418 (87.1%),  mean=-0.00258
signal_distance: mean=0.578, range [0, 2]
```

---

## 5. 替换质量分析

### 5.1 替换效果对比

| 指标 | 本次 NEW | 10/2 OLD |
|---|---|---|
| 成功替换数 | 57 | 51 |
| 平均 delta_R | **+0.0045** | -0.01419 |
| 平均 max_correlation | **0.658** | 0.550 |
| 平均 score | 0.797 | 0.000 |

**对比结论**:
- NEW 的 child 比 OLD **稍微更优、稍微更相关**
- 0.658 的相关性说明多样性差
- OLD run 因为 score 优先导致 pool 漂向负值（utility 崩盘到 0.037）

### 5.2 成功替换的典型改动

| 类型 | 示例 |
|---|---|
| 窗口大小调整 | `TsIr($volume,20)` → `TsIr($volume,40)` |
| 估计器替换 | `Sub(-30.0,TsMad(...,30))` → `Sub(-30.0,TsStd(...,30))` |
| 子树简化 | `Mul(Div(TsMean(Div(Add($high,$low),2.0),60),$vwap),TsIr($volume,60))` → `Mul(Div(TsMed($close,60),$vwap),TsIr($volume,60))` |

**缺少结构创新** - LLM 不会跳出当前因子形态做根本性重写。

---

## 6. 系统结构性缺陷（代码层）

### 6.1 权重永远不优化（`alpha_pool.py:206-209`）

```python
def to_dict(self):
    return dict(
        exprs=[str(e) for e in self.exprs],
        weights=[1 / len(self.exprs)] * len(self.exprs),  # ← 硬编码均匀权重！
        metrics=[{...} for e in self.exprs],
        utility=self.utility()
    )
```

**所有 pool 文件的 weights 都是 `[0.02, 0.02, ..., 0.02]`**（= 1/50）。从不对因子权重做任何优化。

**对比**: AlphaSAGE / AlphaPROBE 用 PyTorch 优化权重；AlphaCF 永远是等权组合。**这是结构性上限**。

### 6.2 替换资格严格（`alpha_pool.py:185-188`）

```python
# Case 2: highest-scoring child strictly above the parent replaces it.
best = int(np.argmax(scores[1:])) + 1
if scores[best] > parent_score:    # ← 严格大于，不是 >=
    winner = members[best]
```

由于 score 是归一化 [0,1] 加权和（精度有限），**很多"差不多好"的 child 都被拒掉**。当 alpha/beta 变小（NEW 配置）后 score 的分辨率进一步降低。

### 6.3 通过率瓶颈

| 阶段 | 处理 | 通过 | 通过率 |
|---|---|---|---|
| Diagnosis | 100 LLM 调用 → 500 提议 | 480 measured | 96% |
| Evolve | 100 LLM 调用 → 277 offspring | 277 proposal events | 100% |
| Evaluate | 277 proposals | 257 factor events | 93% |
| Replace | 257 candidates vs 100 父代 | 57 replacements | **22%** |
| Pool 更新 | 57 成功 | 28/50 仍是初始种子 | 56% 池未变 |

**关键瓶颈在 replace**：平均每轮仅 5.7 个父代被替换。10 轮后 pool 还有 56% 是初始种子。

---

## 7. LLM 错误率（虽低但存在）

| 错误类型 | 次数 |
|---|---|
| Illegal window or future reference | 5 |
| Factor has no usable cross-sectional variation | 5 |
| Invalid token（`$product` 等） | 3 |
| 窗口超出可用历史 | 2 |
| 超过 5 个机制限制 | 1 |
| Invalid expression (LLM 生成有 bug) | 1 |

合计 ~16/480 = **3.3% 错误率**。不算大问题但浪费 LLM token。

### 7.1 LLM 响应质量

- 201/201 finish_reason='stop'（无截断）
- 41/43 一次成功（其余 2 次重试一次）
- 1 次 invalid_llm_response
- LLM 调用质量基本正常

---

## 8. Pool 演变对比

### 8.1 top |R| 与强因子变化

```
Top 5 by |R| in pool_0:
  |R|=0.0667  Add(1.0,TsCov($vwap,Sub(Sub(2.0,$volume),-0.01),10))
  |R|=0.0666  TsCorr($volume,$vwap,30)
  |R|=0.0640  TsCov(Div(Sub($close,$open),$open),Div(Sub($volume,Ref($volume,1)),Ref($volume,1)),30)
  |R|=0.0630  TsIr($volume,50)
  |R|=0.0627  Sub(-30.0,TsMad(Log(TsEMA($volume,20)),30))

Top 5 by |R| in pool_10:
  |R|=0.0781  Mul(Div(TsMed($close,60),$vwap),TsIr($volume,60))
  |R|=0.0760  Mul(Div(Ref($close,20),$close),TsIr($volume,40))
  |R|=0.0748  Sub(0.0,TsCov(TsPctChange($vwap,5),$volume,20))
  |R|=0.0720  Div(TsMean(Log(Mul($vwap,$volume)),40),TsMad(Log(Mul($vwap,$volume)),40))
  |R|=0.0698  Mul(-1.0,TsStd(Log(TsEMA($volume,5)),30))

|R|>0.04:  pool_0 = 14 个,  pool_10 = 20 个 (+43%)
mean IC:   pool_0 = +0.0100, pool_10 = +0.0151 (+51%)
max |IC|:  pool_0 = 0.0667,  pool_10 = 0.0781 (+17%)
```

### 8.2 池子构成

- pool_10 中 28 个仍是原始种子（56%）
- 22 个被替换（其中 22 个来自 round 1-10 的子代）
- 但 14 个负 IC 因子**集**完全不变

---

## 9. 历史对比：上一次失败的 run

**OLD (10/2)**: `data/cf_logs/20261002_162225_644686_csi300_0`

```
Negative-IC factors over rounds:
  R0: 6  →  R1: 7  →  R2: 9  →  R3: 9  →  R4: 11
  R5: 13  →  R6: 15  →  R7: 18  →  R8: 18

Pool went 0.103 → 0.037 (utility collapsed)
```

**与本次对比**:
- OLD: 负 IC 因子**持续增长**（pool 漂移），utility 崩盘
- NEW: 负 IC 因子数量稳定在 14，utility 提升（但 train-test gap 巨大）
- 两次都是结构性偏置的不同表现

---

## 10. 改进建议（按优先级）

### P0 - 必须修复

#### 1. 父代选择方法（最具创新空间，影响最大）

**核心问题**: 14 个负 IC 因子永远没机会被改，pool 始终含"毒因子"。当前实现只能从 reward top-half 选，无法覆盖负 IC 区域。

**候选方案（按实施难度递增）**：

**(a) 简单版：保留少量探索名额（最小改动）**

```python
def select_parents(self):
    ranked = sorted(self.pool.exprs, key=lambda e: self.pool.evaluate(e)["reward"], reverse=True)
    count = min(self.args.parents, len(ranked))
    n_exploit = count // 2                       # 5 个
    n_explore = count - n_exploit                # 5 个
    
    # 5 个利用：top 5 reward（保持现状）
    selected = ranked[:n_exploit]
    
    # 5 个探索：从未被选过 OR reward 最低的（含负 IC）
    remaining = ranked[n_exploit:]
    explored = sorted(remaining, key=lambda e: (self.visits[str(e)], -self.pool.evaluate(e)["reward"]))
    selected += explored[:n_explore]
    
    # 每轮额外加 1 个负 IC 名额（保证负 IC 因子有机会被看到）
    neg_factors = [e for e in self.pool.exprs if self.pool.evaluate(e)["reward"] < 0]
    if neg_factors and len(selected) < count:
        selected.append(neg_factors[hash(str(self.step)) % len(neg_factors)])
    
    for expr in selected:
        self.visits[str(expr)] += 1
    return selected
```

**(b) UCB 多臂赌博机（推荐，单行改动）**

经典 explore-exploit。鼓励访问次数少的因子（包括负 IC）：

```python
def select_parents(self):
    count = min(self.args.parents, len(self.pool.exprs))
    total_visits = sum(self.visits.values()) + 1
    
    def ucb_score(expr):
        reward = abs(self.pool.evaluate(expr)["reward"])
        n_visits = self.visits[str(expr)] + 1
        return reward + self.args.ucb_c * math.sqrt(math.log(total_visits) / n_visits)
    
    ranked = sorted(self.pool.exprs, key=ucb_score, reverse=True)
    selected = ranked[:count]
    for expr in selected:
        self.visits[str(expr)] += 1
    return selected
```

- ucb_c=0.5 时，0 visits 因子会获得 ~0.5 探索奖励，足以让负 IC 因子进入 top-10
- 对 alpha/beta/gamma/cost 体系无影响，零复杂度增加
- 在 10 轮后理论上所有因子至少被访问 1 次

**(c) 多样性约束（防"全选同一类"）**

```python
def select_parents(self):
    count = min(self.args.parents, len(self.pool.exprs))
    candidates = list(self.pool.exprs)
    
    # 先按 reward 排序
    candidates.sort(key=lambda e: abs(self.pool.evaluate(e)["reward"]), reverse=True)
    
    selected = [candidates[0]]  # 第一个必选
    for cand in candidates[1:]:
        if len(selected) >= count:
            break
        # 与已选因子最大相关性 < 0.7 才选
        max_corr = max((self.pool.correlation(cand, s, absolute=True) for s in selected), default=0)
        if max_corr < 0.7:
            selected.append(cand)
    
    for expr in selected:
        self.visits[str(expr)] += 1
    return selected
```

- 防止选出的 10 个父代高度相关（当前 round 8 的 4 个 parent 都是 vwap/volume 协变型）
- 自动让负 IC 但结构独特的因子入选

**(d) "停滞优先"策略（针对负 IC）**

```python
def select_parents(self):
    count = min(self.args.parents, len(self.pool.exprs))
    
    # 5 个最高 reward（exploit）
    by_reward = sorted(self.pool.exprs, key=lambda e: self.pool.evaluate(e)["reward"], reverse=True)
    selected = by_reward[:count // 2]
    
    # 5 个最久未被改进的（含负 IC，因 visits=0 的负 IC 因子被强制访问）
    remaining = [e for e in by_reward[count // 2:]]
    stagnant = sorted(remaining, key=lambda e: (-self.visits[str(e)], abs(self.pool.evaluate(e)["reward"])))
    selected += stagnant[:count - len(selected)]
    
    for expr in selected:
        self.visits[str(expr)] += 1
    return selected
```

**推荐组合**: 方案 (b) UCB 作为基础，方案 (a) 加 1 个负 IC 名额作为保险。

---

#### 2. alpha/beta/gamma/cost_weight 超参数分析

**现状评分公式**（`alpha_pool.py:100-126` 与 `:173-178`）：

```python
weights = np.array([self.args.alpha, self.args.beta, self.args.gamma, self.args.cost_weight])
rows = [
    [abs(self.evaluate(e)["reward"]), u - utility,
     1 - correlations[str(e)], 1 - self.evaluate(e)["cost"]]
    for e, u in zip(remaining, utilities)
]
scores = _minmax_scores(rows) @ weights    # 先 minmax 归一化到 [0,1]，再加权和
```

**4 个评分项的归一化前实际取值范围（本次 run 统计）**：

| 列 | 含义 | 取值范围 | 典型值 | 物理含义 |
|---|---|---|---|---|
| `\|R\|` | 因子绝对 RankIC | [0, 0.10] | 0.02~0.08 | 越大越好 |
| `c_pool = U(P∪{e}) - U(P)` | 加入后 pool utility 增量 | [-0.05, +0.05] | -0.01~+0.01 | 正值有帮助 |
| `diversity = 1 - max_corr` | 与同池因子最大相关性 | [0, 1] | 0.1~0.8 | 越大越多样 |
| `1 - cost` | 换手成本反义 | [0, 1] | 0.5~0.99 | 越大越稳定 |

**问题 1**：minmax 归一化抹掉了"自然尺度"信息。归一化后所有列都在 [0,1]，权重 α/β/γ/cost 实际上控制的是"在该列中相对排名"，而不是"在该列中的实际幅度"。

**问题 2**：当前 NEW 配置 `α=0.4, β=0.3, γ=0.2, cost=0.1` 与 OLD `α=1.0, β=1.0, γ=0.2, cost=0.1` 归一化后：
- NEW: \|R\|=40%, C_pool=30%, 多样性=20%, 换手=10%
- OLD: \|R\|=43%, C_pool=43%, 多样性=9%, 换手=4%

NEW 配置中 γ=0.2 多样性权重反而比 OLD 的 γ=0.2 在归一化下权重更大，**无意中过度强调多样性**。这可能解释为何 NEW 替换的相关性更低（多样性变好）但个体 \|R\| 提升有限。

**方案 A：取消归一化，使用原始尺度权重**

把公式改为：

```python
def score(e):
    return (alpha * abs(R)                          # |R|, ~0.08 typical
            + beta * max(0, c_pool) / max_cpool     # c_pool, only positive contributes
            + gamma * diversity                     # already [0,1]
            + cost_weight * (1 - cost))             # already [0,1]
```

但 `c_pool` 在 [-0.05, +0.05] 范围，归一化前需要：
- `max(0, c_pool)`：只奖励正值，惩罚负值
- 或者用 `c_pool + 0.05` 移到 [0, 0.1] 后归一化

为保证各项贡献量级相当，建议的**无归一化权重**：

| 项 | 典型值 | 缩放目标 | 推荐权重 |
|---|---|---|---|
| \|R\| (典型 0.05) | 0.05 | 0.5 score | **α=10** |
| c_pool (典型 0.01) | 0.01 | 0.5 score | **β=50** |
| diversity (典型 0.5) | 0.5 | 0.5 score | **γ=1** |
| 1-cost (典型 0.7) | 0.7 | 0.5 score | **cost=0.7** |

各项贡献近似为：\|R\| → 10×0.05=0.5, c_pool → 50×0.01=0.5, diversity → 1×0.5=0.5, 1-cost → 0.7×0.7=0.5。

这样**每个维度在"中等表现"时贡献相同**，避免 \|R\| 小数被小数权重淹没。

**方案 B：保留 minmax 但根据"想要的最优解类型"调参**

如果保留归一化逻辑：
- 想要"顶级 \|R\| 因子优先"：α≥0.6, β≤0.2
- 想要"pool 加成优先"：β≥0.5, α≤0.3
- 想要"探索多样"：γ≥0.3, cost_weight≥0.15

**当前 NEW 的问题是 α+β 仅 0.7，相对而言**太弱**。建议先回到 α=1.0, β=1.0 复测验证假设（pool_0 应该回到 0.109 附近）**。

**可行性结论**：取消归一化**完全可行**，且推荐方案 A（无归一化 + 基于自然尺度设计权重）。这让超参数的语义清晰：α 直接控制"我们愿意为每 0.01 |R| 增量付出多少分"。

---

#### 3. 权重优化（pool 组合权重，非评分权重）

**核心问题澄清**：第 10 节 #3 说的是**最终组合的因子权重**（`to_dict()` 里的 weights 字段），不是评分函数里的 α/β/γ。两者不同：

| | 评分权重 (α/β/γ/cost) | 组合权重 (pool factor weights) |
|---|---|---|
| 位置 | `select()` / `replace_parent()` 评分 | `to_dict()` 输出 |
| 当前值 | alpha=0.4, beta=0.3, ... | **永远是 [1/N, 1/N, ...]** |
| 决定什么 | 哪些因子入选/替换 | 最终等权 vs 加权组合 |

**最简单的修复（最小改动）**：

```python
# alpha_pool.py:206-209 替换 to_dict 中的 weights
def to_dict(self):
    exprs = self.exprs
    # 方案 1：剔除负 IC 因子后等权
    pos = [e for e in exprs if self.evaluate(e)["reward"] > 0]
    target = pos if pos else exprs
    weights = [1.0 / len(target) if e in target else 0.0 for e in exprs]
    return dict(exprs=[str(e) for e in exprs], weights=weights, ...)
```

**进阶（IC 倒数加权）**：

```python
# 权重 ∝ |R|，负 IC 强制为 0
weights = []
for e in exprs:
    r = self.evaluate(e)["reward"]
    weights.append(abs(r) if r > 0 else 0.0)
total = sum(weights) or 1.0
weights = [w / total for w in weights]
```

**进阶（最小二乘求最优权重）**：

```python
# 用前 T 天的 signal 与 target 反向求解最优 w
# 最小化 ||Σ w_i signal_i - target||^2
from numpy.linalg import lstsq
signals = torch.stack([self.signal(e).nan_to_num() for e in exprs]).T  # [T, N]
target = self.target.nan_to_num()                                       # [T]
mask = torch.isfinite(signals).all(1) & torch.isfinite(target)
w, _, _, _ = lstsq(signals[mask].numpy(), target[mask].numpy(), rcond=None)
w = np.clip(w, 0, None)        # 约束非负
w = w / (w.sum() + 1e-12)     # 归一
weights = w.tolist()
```

**为什么这个问题需要修**：
- 当前 14 个负 IC 因子（\|R\| ~ -0.05）等权进入组合 → 拖累 ~0.007 的 utility
- 剔除负 IC 因子即可简单提升 ~11% utility（粗算）
- 不等权组合理论上能进一步榨干 IC 信息

---



### P1 - 质量提升

#### 4. Memory 过滤

- 当前 480 条 memory 81% 是负证据
- 建议：只在 `delta_cf > 0` 或 `pool_credit > 0` 时写入 memory
- 避免污染后续 LLM 输入

#### 5. 放宽替换资格

- `scores[best] > parent_score` 改为 `scores[best] >= parent_score * 0.99`
- 让小幅改进也能进 pool

#### 6. 降低池子重复率

- 当前 56% 池子仍是初始种子
- 增加每轮 parent 数 (10 → 15)，或允许同一个 parent 在 5 轮内多次被选择

### P2 - 监控与可观测性

#### 7. 记录 test 期间 utility

- pool 文件只保存 train 表现，无法追踪 overfit
- 建议每 N 轮评估一次 valid/test utility 并写入日志

#### 8. 追踪 pool IC 分布

- 实时显示 pool 中 pos/neg IC 数量
- 当前是固定的 36/14 始终没变，是个无声的 bug

---

## 11. 关键证据摘要

### 11.1 数据文件

```
data/cf_logs/20261003_011950_475394_csi300_0/
├── args.json              # 配置（alpha=0.4, beta=0.3 是关键变更）
├── pool_0..10.json        # 11 个 pool 状态（仅 train metrics）
├── round_0..10.jsonl      # 11 轮事件流（150-160 events/轮）
├── memory.json            # 480 条 counterfactual 记录
├── lineage.json           # 150 条谱系（50 seed + 100 替换尝试）
└── ret_s.npy              # 测试期 backtest 收益序列
```

### 11.2 关键代码位置

```
src/alpha_cf/trainer.py
  L101-110  select_parents()         ← 结构性偏置源头
  L112-149  diagnose()                ← counterfactual evidence 生成
  L151-171  evolve()                  ← offspring 生成（用 "offspring" key）
  L36-73    ask()                     ← LLM 调用与 JSON 校验
  L173-216  train()                   ← 主循环

src/alpha_cf/alpha_pool.py
  L136-200  replace_parent()          ← 替换逻辑（严格大于）
  L206-209  to_dict()                 ← 权重硬编码为 1/N
  L100-126  select()                  ← 初始化评分函数

src/alpha_cf/prompt.py
  L47-73    PROMPT_DIAGNOSIS          ← 期望 "mechanisms" key
  L75-117   PROMPT_EVOLUTION          ← 期望 "offspring" key
```

### 11.3 验证方法

| 缺陷 | 验证方式 |
|---|---|
| 父代偏置 | grep `Signed RankIC` in llm_request，全部为正 |
| 负 IC 不变 | set(pool_0_neg) == set(pool_10_neg) |
| LLM 质量 | distribution of mechanism counts per response |
| 测试期表现 | `python3 scripts/_analyze_run.py` 对比 ret_s.npy |

---

## 12. 总结

本次 pipeline 表现差（测试 Sharpe 0.29）的根本原因是**多重结构性缺陷叠加**：

1. **父代选择永远跳过负 IC 因子** → pool 始终含 14 个"毒因子"
2. **alpha/beta 配置变更**（1.0/1.0 → 0.4/0.3）→ pool_0 utility 减半
3. **LLM 81% 反向证据污染 memory** → 误导后续进化方向
4. **权重永远等权** → 顶部 \|R\|=0.078 与 \|R\|=-0.066 同权，结构性上限
5. **严格替换门槛** → 56% pool 仍是初始种子

这些缺陷之间相互放大：**配置变更**降低了初始化质量，**父代偏置**让负 IC 因子无法被清理，**等权组合**让负 IC 因子在 pool 中持续拖累，**严格门槛**让小幅改进也被拒之门外。最终训练期 utility 看着在涨，但测试期 Sharpe 不到基线的一半。

修复的杠杆点排序：
1. 父代选择加入负 IC 因子（清理毒因子）
2. 恢复 alpha=1.0, beta=1.0 或重新调参
3. 添加权重优化
4. 过滤 memory 中的负证据
