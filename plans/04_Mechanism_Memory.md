# 04 — 机制记忆

对应代码：`src/alpha_cf/trainer.py::AlphaCFTrainer.memory`（在 `__init__` 中初始化为 `[]`）、`diagnose / evolve / save`。

---

## 4.1 数据结构

`self.memory` 是 `list[dict]`，每个元素对应**一次成功测量的 ablation**，字段与 `round_<step>.jsonl::mechanism` 一致：

```python
{
  "factor":     str(parent),                     # 整张表达式
  "path":       list[int],                       # AST path
  "expression": str(node),                       # 被干预的子树
  "description": str,                            # LLM 给的金融语义
  "mode":       "remove" | "neutralize",
  "replacement": str,                            # LLM 给的 baseline
  "reason":     str,
  "ablation":   str(changed),                    # 实际执行结果（含 validate 后）
  "delta_cf":   float,                           # 个体证据（必填）
  "pool_credit": float,                          # 组合证据（--no-pool-credit 时为 None）
  "action":     "Preserve" | "Replace" | "Explore"
}
```

约束：
- 仅 `delta_cf is not None` 的行进入 `self.memory`——即 LLM 给的 `mode/replacement` 必须通过 `ablate()` 的合法性检查并且 `pool.evaluate(changed)` 能产生有效 RankICIR。
- `pool_credit` 为 None 时 action 仍按 `delta_cf` 决定。

---

## 4.2 写入与读取

**写入**：`trainer.py:118-119`

```python
measured = [row for row in evidence if row["delta_cf"] is not None]
self.memory.extend(measured)
```

—在每个 parent 的 `diagnose` 末尾发生，跨 round 累积。

**读取**：`trainer.py:125-129` 在 `evolve()` 里把整张 memory 作为 LLM 上下文：

```python
historical_memory = [] if args.no_memory else self.memory,
existing_expressions = [str(e) for e in self.pool.exprs + previous_children],
```

每次 LLM 调用都传**完整**的 `self.memory`——不裁剪、不向量化、不分页、不做检索 top-k。设计上假设 memory 不会超过 LLM 上下文（每轮增量 ≈ 20 parents × 3 mechanisms × ~300 字 ≈ 18k tokens，仍在安全范围）。

`--no-memory` flag 会把 `historical_memory` 替换为 `[]`，便于消融"机制记忆是否真的有用"。

---

## 4.3 持久化

```python
def save(self):
    save_json(self.log_dir / f"pool_{self.step}.json", self.pool.to_dict())
    save_json(self.log_dir / "memory.json", self.memory)
```

- 每个 round 末尾写一次 `pool_<step>.json`（含 60 个表达式 + metric + utility）
- 每轮覆盖写一次 `memory.json`（最新快照）
- 全 round 的 `round_<step>.jsonl` 累积（append-only），用于回放某步所有 mechanism 提议

不维护二级索引（按 parent / 按 mode 等），不复用历史 cache，不维护向量库。

---

## 4.4 数值证据的边界

- `delta_cf` 和 `pool_credit` 都**只对原始 parent**有效：迁移到子代或修改后结构上不再适用。
- 证据不继承：哪怕 `delta_cf = -0.03` 表示 short-term reversal 很有用，把它嵌入新父式后必须重新干预一次才能再用。
- LLM 看到的是原始数值（不只标签），所以可以结合 delta_cf 和 pool_credit 自行判断是否存在个体/组合冲突。
- 同一 parent × 同一 path 多次测量不会被去重——memory 按时间累积，但子代 LLM prompt 会让历史记录里多条相同 factor 的提议各自保留。

---

## 4.5 伪代码（写入路径）

```python
# 诊断单个 parent 的过程中
for mechanism in proposals[:args.mechanisms]:
    row = {... 默认 Explore, delta_cf=None, pool_credit=None}
    if mechanism.get("mode"):
        try:
            changed = ablate(parent, mechanism, args)
            result  = pool.evaluate(changed)
            row["ablation"] = str(changed)
            row["delta_cf"] = result["reward"] - reward
            # ... pool_credit & action
        except ...:
            self.record("invalid_mechanism", ...)
            continue
    evidence.append(row)
    self.record("mechanism", **row)

# 仅"被测量"的进入 memory
self.memory.extend(r for r in evidence if r["delta_cf"] is not None)
return dict(expression=..., reward=..., mechanisms=evidence)
```

---

## 4.6 关闭机制记忆

- `--no-memory`：LLM 在 evolution 阶段拿不到 `historical_memory`，相当于"只基于本轮每个 parent 自己的诊断"做演化，便于比较"机制级 vs 单次诊断"的差异。
- 关闭后 `memory.json` 仍按原逻辑写入（内容由 `--no-cf-evidence` / `--no-pool-credit` 决定），但 LLM prompt 不引用。

---

## 4.7 不做的事情（与方案 1 的区别）

- **不**做机制向量化检索或 RAG——`ask()` 不接受 `top-k memory`，而是整张传入
- **不**做多层结构契约过滤（specificity hierarchy）——LLM 自行决定哪些子结构有"金融含义"
- **不**裁剪或摘要 memory——全量传
- **不**为相同 path 重复 trace 打包——每次测量独立成行
- **不**为新结构/新参数继承数值证据——任何 reuse 必须由 LLM 提议并重新做 ablation
