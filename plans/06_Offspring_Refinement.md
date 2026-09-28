# 06 — 子代细化

对应代码：`src/alpha_cf/trainer.py::AlphaCFTrainer.refine`，依赖 `src/alpha_cf/expression.py::parameter_variants`。

每个 round 在 `evolve` 完成后、`pool.update` 之前执行。

---

## 6.1 入口与前提

`refine(children)`（`trainer.py:145-160`）：

```python
def refine(self, children):
    if args.no_refinement:
        return children

    # 1) 选出 top-K 个子代（默认 K=5）
    best = sorted(children, key=lambda e: pool.evaluate(e)["reward"], reverse=True)[:args.refine_top_k]

    refined = []
    for child in best:
        winner = child
        # 2) 对 child 枚举所有合法参数变体，挑 R 最大的
        for variant in parameter_variants(child, args):
            result = self.evaluate(str(variant))                        # parse + pool.evaluate
            if result is not None and pool.evaluate(result)["reward"] > pool.evaluate(winner)["reward"]:
                winner = result
            # 3) cache 只保留"当前最优 + 当前池 + 当前 refined 候选"，释放变体张量
            self.pool.keep(self.pool.exprs + children + refined + [winner])
        refined.append(winner)
        self.record("refinement", original=str(child), refined=str(winner))

    return children + refined                                          # 4) 原版 + refined 都进候选集
```

要点：
- 默认 `--refine-top-k=5`：只对奖励前 5 个子代做窗口枚举（防止 100 个 offspring × 网格的爆炸）
- `--no-refinement` 时直接返回 `children`，不做细化
- 输出**不替换**原 child，而是 `children + refined`：原版和细化版一起进入 `pool.update(children)`，由 `select()` 决定谁最终留在 60 个槽里

---

## 6.2 参数变体的生成

`expression.py::parameter_variants(expr, args)`（伪代码）：

```python
def parameter_variants(expr, args):
    # 1) 找出所有大于 1 的窗口常量（window=1 通常是 Ref 的 1-day lag，按约定不动）
    windows = sorted({n._delta_time for _, n in walk(expr)
                      if hasattr(n, "_delta_time") and n._delta_time > 1})[:2]
    # ↑ 只取前 2 个最大的窗口（dict 有序 ⇒ 最大的两个），控制笛卡尔积维度
    # 默认升序 ⇒ 取到的实际是"窗口集合中最小的两个"，便于与 --windows 网格对齐
    if not windows:
        return                                                         # 没有可调窗口

    # 2) 对每个被选窗口生成候选集合：{原值} ∪ args.windows
    # 默认 --windows = [5, 10, 20, 40, 60]
    grids = [sorted({w, *args.windows}) for w in windows]

    # 3) 全笛卡尔积
    for values in product(*grids):
        variant = deepcopy(expr)
        mapping = dict(zip(windows, values))
        # 4) 在 variant 中把等于原窗口的 _delta_time 全部替换成对应新值
        for _, node in walk(variant):
            if hasattr(node, "_delta_time") and node._delta_time in mapping:
                node._delta_time = mapping[node._delta_time]
        if str(variant) != str(expr):                                   # 至少要有一个变化
            try:
                yield validate(variant, args)                          # 通过 AST 合法性检查
            except ValueError:
                continue                                                # 失败跳过
```

举例：
- `windows = [5]`，`grids = [{5, 10, 20, 40, 60}]` → 5 个候选
- `windows = [10, 20]`，`grids = [{5,10,20,40,60}, {5,10,20,40,60}]` → 25 个候选
- `windows = [5, 10, 20]` → 因 `[:2]` 只取前 2 个，仍是 25 个候选

固定不动：
- `d=1` 的窗口（Ref 1-day lag、TsDelta 1-day 差分）按约定保留
- `d=0` 的 Ref（取当日值）通常不存在 `>1` 检查，会被过滤

---

## 6.3 评测与选择

对每个 variant：
1. `self.evaluate(str(variant))`：parse → validate → `pool.evaluate`（含 rank、spearman、icir）
2. 与 `winner` 比较：若新 variant 的 `reward` 更高就替换 `winner`
3. **不**缓存变体的 signal 张量——每轮迭代完后 `pool.keep()` 会按需丢弃

`refined.append(winner)` 把每个子代的赢家收集起来；最后 `return children + refined` 让下游 `pool.update` 一起打分。

---

## 6.4 性能与缓存

- 每个变体都触发一次 `evaluate`，但 cache 只保留当前最优：避免 100 个变体 × 5 个子代 × 60 个池成员的信号张量同时驻留 GPU
- `evaluate` 复用 `AlphaCFPool.cache`：`pool.exprs + children + refined + [winner]` 之外的 key 全部清空
- `round_<step>.jsonl::refinement` 只记录原始子代和最终赢家，便于回放

---

## 6.5 关闭细化

- `--no-refinement`：直接跳过 refine，等价于 `children` 原样进入 `pool.update`
- 不提供 `--refine-top-k` 之外的控制变量（窗口网格本身由 `--windows` 控制）
