# 07 — 工作池更新

对应代码：`src/alpha_cf/alpha_pool.py::AlphaCFPool.select / update / keep`、`src/alpha_cf/trainer.py::AlphaCFTrainer.train / save`。

每个 round 结束前执行一次 `pool.update(children)`，把池大小从 60 重新选满到 60（候选 = 当前 60 + 本轮 offspring）。

---

## 7.1 候选集构造

```python
def update(self, offspring):
    self.exprs = self.select(self.exprs + offspring, args.pool_capacity)  # 60
    self.keep(self.exprs)
```

`select(candidates, size)`（`alpha_pool.py:81-116`）：

```python
def select(self, candidates, size):
    # 1) 字符串去重
    remaining = list({str(e): e for e in candidates}.values())
    # 2) 预 evaluate：把 candidate 都填进 cache（保证后续打分用到 rank/signal）
    for expr in remaining: self.evaluate(expr)
    if len(remaining) < size:
        raise ValueError(f"Need {size} usable unique factors, got {len(remaining)}")
    # 3) 关闭 pool selection 时直接按 reward 截断
    if args.no_pool_selection:
        return sorted(remaining, key=lambda e: self.evaluate(e)["reward"], reverse=True)[:size]

    selected, total, utility = [], torch.zeros_like(self.target, dtype=torch.float64), 0.0
    correlations = {str(e): 0.0 for e in remaining}
    weights = np.array([args.alpha, args.beta, args.gamma, -args.cost_weight])     # [1.0, 1.0, 0.2, -0.1]

    # 4) 贪心选 size 个
    while len(selected) < size:
        # 4a) 算每个候选的"加入后池 utility"
        utilities = []
        for start in range(0, len(remaining), 4):           # 一次最多处理 4 个候选，控制显存
            batch   = remaining[start:start+4]
            signals = torch.stack([self.signal(e) for e in batch])
            utilities.extend(self.score_signals((total + signals) / (len(selected)+1)).tolist())

        # 4b) 打分四元组：[R(f), U(new_pool)-U(current), 1-max_corr, cost]
        rows = np.array([
            [self.evaluate(e)["reward"],
             u - utility,
             1 - correlations[str(e)],
             self.evaluate(e)["cost"]]
            for e, u in zip(remaining, utilities)
        ])
        # 4c) min-max 归一化后线性加权
        scores = ((rows - rows.min(0)) / np.maximum(np.ptp(rows, axis=0), 1e-12)) @ weights
        scores[~np.isfinite(utilities)] = -np.inf                          # 不可用候选被屏蔽
        if not np.isfinite(scores).any():
            raise ValueError("No valid pool extension")
        index   = int(scores.argmax())
        winner, utility = remaining.pop(index), utilities[index]
        selected.append(winner)
        total   += self.signal(winner)                                     # double 累加

        # 4d) 更新"已选 vs 候选"的最大相关性（多样性）
        for start in range(0, len(remaining), 4):
            batch   = remaining[start:start+4]
            signals = torch.stack([self.evaluate(e)["signal"] for e in batch])
            rhs     = self.evaluate(winner)["signal"].expand_as(signals)
            corr    = spearman(signals.flatten(0,1), rhs.flatten(0,1)).reshape(signals.shape[:2]).abs().nanmean(1)
            for expr, value in zip(batch, corr.tolist()):
                correlations[str(expr)] = max(
                    correlations[str(expr)],
                    value if np.isfinite(value) else 1.0
                )
    return selected
```

---

## 7.2 Selection formula 对照 idea.md

idea.md §"Factor Selection"：
$$
S(f \mid P) = \alpha R(f) + \beta r_{\text{pool}}(f \mid P) + \gamma D(f, P) - \lambda C_{\text{cost}}(f)
$$

程序实现：

| 项 | idea 符号 | 程序字段 | 计算 |
|---|---|---|---|
| 单因子质量 | $R(f)$ | `rows[:, 0]` | `evaluate(e)["reward"]`（有符号 RankICIR） |
| 边际 pool utility | $r_{\text{pool}}(f\mid P)$ | `rows[:, 1]` | `u - utility` = 加入后的 RankICIR − 当前 |
| 多样性 | $D(f,P)$ | `rows[:, 2]` | `1 - correlations[str(e)]`，其中 `correlations[str(e)] = max_{g∈selected} mean_t |Spearman(f_t, g_t)|` |
| Turnover 成本 | $C_{\text{cost}}(f)$ | `rows[:, 3]` | `(1 - mean_t Spearman(f_t, f_{t-1}))/2` ∈ [0, 1] |

权重默认值：`alpha=1.0, beta=1.0, gamma=0.2, cost_weight=0.1`。归一化：候选集内 min-max 到 [0, 1]；极差 ≤ 1e-12 时分母取 1e-12 防止除零。

打分后取 `scores.argmax()`，对应 winner 加入池；`correlations` 只升不降（保证多样性单调惩罚最相似的）。

---

## 7.3 缓存与数值一致性

- `self.signal(e)` 返回 double 精度的 rank-normalized 张量；`total` 是 double 累加，防止 60 个成员相加的精度漂移
- `pool.score_signals` 对 `[B, T, n_stocks]` 张量返回 `[B]` 的 RankICIR；`unsqueeze(0)` 把它转成单元素 batch
- `update()` 结束后调用 `keep()`：`self.cache = {k: v for k, v in self.cache.items() if k in {str(e) for e in self.exprs}}`，把池外成员的 signal 张量全部释放
- `signal/evaluate` 全部走 `torch.no_grad()`（`evaluate` 装饰器）——只前向不反向

---

## 7.4 Selection 的开关与边界

- `--no-pool-selection`：跳过四元打分，按 reward 降序截断到 60。便于消融"pool-aware 选择是否有用"
- `--alpha / --beta / --gamma / --cost-weight`：四元组线性权重；归一化后乘加
- 候选不足 60 个 → 抛错（候选太少应当反思 offspring generation）
- 所有候选 utility 都是 nan → 抛错（信号整体退化）
- `len(selected) == 60` 时停止，不会自动从外部补候选

---

## 7.5 持久化

每轮结束后 `trainer.save()`：

```python
def save(self):
    save_json(log_dir / f"pool_{self.step}.json", pool.to_dict())
    save_json(log_dir / "memory.json", self.memory)
    print(f"Pool {self.step}: size={len(pool.exprs)}, RankICIR={pool.utility():.4f}")
```

`pool.to_dict()`：
```python
{
  "exprs":   [str(e) for e in self.exprs],               # 60 个表达式
  "weights": [1/K] * K,                                  # 等权
  "metrics": [{"reward", "ic", "cost"} for e in self.exprs],
  "utility": self.utility()                              # 当前池 U
}
```

`round_<step>.jsonl::update` 行：
```json
{"event":"update", "offspring": <int>, "added": [str, ...], "utility": <float>}
```

`added` 字段记录本轮真正进入池的子代字符串列表，便于事后追溯每一轮的进化痕迹。

---

## 7.6 不做的事情（与方案 1 的区别）

- **不**用多轮"refill"机制：候选集**严格**是当前池 + 本轮 offspring，不存在"老候选"或"外部记忆"回流
- **不**先按阈值淘汰候选——所有 candidate 都参与打分
- **不**维护多目标 Pareto front——线性加权归一化是最简形式
- **不**提前预计算所有 utility 矩阵——`while` 循环每轮重新算 `u - utility`
- **不**保留历史候选池——`keep()` 严格释放
