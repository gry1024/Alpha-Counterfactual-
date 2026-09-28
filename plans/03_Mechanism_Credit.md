# 03 — 机制贡献

对应代码：`src/alpha_cf/trainer.py::AlphaCFTrainer.diagnose` 中 `pool_credit` 字段的填充逻辑，依赖 `src/alpha_cf/alpha_pool.py::AlphaCFPool.utility / score_signals / signal / evaluate`。

---

## 3.1 Pool Utility 的定义

`AlphaCFPool.utility(exprs=None)`（默认用 `self.exprs`）：

```python
def utility(self, exprs=None):
    exprs = self.exprs if exprs is None else exprs
    if not exprs: return 0.0
    # 等权 rank-normalized 组合：F_P = (1/K) * sum_i f_i_rank
    signal = sum(self.signal(expr) for expr in exprs) / len(exprs)         # double 累加
    # U(P) = RankICIR(F_P)
    return self.score_signals(signal.unsqueeze(0))[0].item()
```

- `signal(expr)` 返回 `self.evaluate(expr)["signal"].nan_to_num().double()`——即 rank-normalized signal，缺失值补 0（中性 rank），用 `double` 精度累加防止大批求和的精度漂移
- `score_signals([signal])` 计算 RankICIR：`daily = spearman(signal, target)`，再走 `icir(daily)` 取第 0 个

idea.md 中
$$U(P) = \operatorname{RankICIR}\!\left(\frac{1}{K}\sum_i \tilde f_{i}\right)$$
完全对应——`\tilde f_i` 即 `pool.signal(expr)`，输出是横截面均值化的 rank。

---

## 3.2 单个机制的 pool_credit

idea.md 的定义是：
$$C_{\text{pool}}(m; f, P) = U(P) - U\!\left(P_{-f}\cup\{T(f,m)\}\right)$$

程序实现 `trainer.py:105-107`：

```python
# 在 diagnose(parent, utility, total) 内部：
signal_new = (total - pool.signal(parent) + pool.signal(changed)) / len(pool.exprs)
row["pool_credit"] = utility - pool.score_signals(signal_new.unsqueeze(0))[0].item()
```

- `total`：本轮开始时 `sum(pool.signal(e) for e in pool.exprs)`，double 精度
- 用 `total - signal(parent) + signal(changed)` 等价于重新求和「pool - parent + ablation」
- 不重新遍历其他成员，保证与"重建"结果一致（`+` 在 double 下结合律成立）
- `len(pool.exprs)` 是当前池容量（固定 60）

判定规则：
| delta_cf | pool_credit | action  | 含义 |
|----------:|------------:|:-------:|------|
| `< -1e-5` | `≥ -1e-5` | Preserve | 个体需要这个机制，pool 也需要 |
| `>  1e-5` | `≤  1e-5` | Replace  | 个体上去掉更好，pool 也支持替换 |
| 其他 | 任意 | Explore | 近零、冲突、未测量 |

阈值 ±1e-5 来自浮点累加下 `icir` 的噪声级；这是相对保守的边界——只有在双方证据一致时才打 Preserve/Replace，其余一律 Explore。

---

## 3.3 与个体证据 delta_cf 的关系

| 量 | 计算公式 | 衡量对象 | 数值范围 |
|---|---|---|---|
| `delta_cf` | `R(T(f,m)) - R(f)` | 单因子 RankICIR 变化 | 实数；负值=贡献 |
| `pool_credit` | `U(P) - U(P\{f} ∪ {T(f,m)})` | 整个池的 RankICIR 变化 | 实数；正值=贡献 |
| `R(f)` | `Mean(IC_t) / (Std(IC_t)+ε)` | 单因子表现 | 通常 [-0.5, 0.5] |
| `U(P)` | `RankICIR(等权组合)` | 池组合表现 | 通常 [-1, 1] |

两者量纲不同，所以不归一化，直接用阈值判定。

---

## 3.4 信号缓存与一致性

- `pool.signal(expr)` 不重新跑 `evaluate`，而是读 `self.cache[str(expr)]["signal"]`（已 nan_to_num 后转 double）。这保证 ablation 和原 parent 都基于同一份 rank-normalized signal 张量。
- `pool.score_signals(signal.unsqueeze(0))` 对单个信号张量返回 shape `[1]` 的 RankICIR，与 `signal.shape == [T, n_stocks]` 兼容——`unsqueeze(0)` 是因为 `score_signals` 期望 `[B, T, n_stocks]`。
- 当 ablation 抛出（`ValueError`）时，`pool.evaluate(changed)` 不会发生，`delta_cf` 与 `pool_credit` 同时为 `None`，不进 `self.memory`。

---

## 3.5 关闭 credit 的消融

- `--no-cf-evidence`：跳过整个 ablation 计算，只保留 LLM 的语义分解。所有 mechanism 的 `delta_cf=pool_credit=None`，action 一律 `Explore`。
- `--no-pool-credit`：保留 `delta_cf`，但 `pool_credit=None`，action 由 `delta_cf` 单独判定——`delta_cf<-1e-5 → Preserve`，`>1e-5 → Replace`，其余 `Explore`。

两个 flag 互不冲突，可独立打开。

---

## 3.6 重要日志

- 每条 mechanism 在 `round_<step>.jsonl::mechanism` 中以单行 JSON 落盘，字段：`factor, path, expression, description, mode, replacement, reason, ablation, delta_cf, pool_credit, action`。
- `memory.json` 只含 `delta_cf != None` 的实证条目，跨 round 累积。
- 全 round 的 `pool_utility` 通过 `pool_<step>.json::utility` 记录，便于事后绘制 `U(P_t)` 曲线。
