# 05 — 宏观演化

对应代码：`src/alpha_cf/trainer.py::AlphaCFTrainer.evolve / train`，配合 `src/alpha_cf/prompt.py::PROMPT_EVOLUTION` 与 `src/utils/llm.py::OpenAIModel.chat_generate`。

每个 round 的演化顺序：
1. 全部 parent 完成 `diagnose`，得到 `evidence` 列表（每个元素含 `expression / reward / mechanisms`）。
2. 对每个 parent 顺序生成 offspring：
   - 选 donor
   - 调 `evolve(parent, donor, previous_children)` → 追加到 `children`
   - `pool.keep(pool.exprs + children)` 裁剪 cache
3. 去重 → refine → `pool.update(children)` → `save()`

---

## 5.1 Donor 选取

`trainer.py:177-185`：

```python
children = []
for parent in evidence:
    donors = [row for row in evidence if row is not parent] or [parent]
    if args.random_crossover:
        donor = random.choice(donors)
    else:
        donor = max(donors, key=lambda row: max(
            [(m["pool_credit"] or 0.0) - (m["delta_cf"] or 0.0) for m in row["mechanisms"]],
            default=0.0))
    children.extend(self.evolve(parent, donor, children))
    self.pool.keep(self.pool.exprs + children)
```

机制：
- 默认模式下按"双重证据最大化"挑 donor：
  $$\text{donor}^* = \arg\max_{\text{other}} \max_{m \in M(\text{other})}\bigl(C_{\text{pool}}(m) - \Delta^{\text{cf}}(m)\bigr)$$
  - 优先选组合贡献大、个体贡献也大的机制——保证 crossover 能带来真正新的结构证据
  - 取不到时（`default=0.0`）落到 parent 自己
- `--random-crossover` flag：从其他 parents 中均匀随机一个 donor。用于消融"donor 选优是否真的有用"
- 单 parent 退化场景：`evidence == [parent]` 时 `donors = [parent]`，相当于无 donor

`donor` 实际是 LLM 接收的 `evidence` 字典（含 expression、reward、mechanisms 全套），LLM 据此判断如何重组。

---

## 5.2 给 LLM 的演化 prompt

`evolve()` 主体（`trainer.py:123-143`）：

```python
def evolve(self, parent, donor, previous_children):
    existing = [str(e) for e in self.pool.exprs + previous_children]
    result = self.ask(PROMPT_EVOLUTION, dict(
        parent            = parent,                                       # evidence 字典
        donor             = donor,                                        # evidence 字典
        historical_memory = [] if args.no_memory else self.memory,        # 全部已测量机制
        existing_expressions = existing,                                  # 当前池 + 本轮已有子代
        offspring_count   = args.offspring,                               # 默认 5
        random_crossover  = args.random_crossover,                        # bool
        limits            = dict(nodes=args.max_nodes, depth=args.max_depth, lookback=args.max_backtrack),
    ))

    expressions, operations, explanations = (
        result["expressions"], result["operations"], result["explanations"]
    )
    if not all(isinstance(items, list) and len(items) == args.offspring
               for items in (expressions, operations, explanations)):
        raise ValueError("Expected matching expressions, operations and explanations for each offspring")

    children, seen = [], set(existing)
    for text, operation, explanation in zip(expressions, operations, explanations):
        self.record("proposal",
                    parent=parent["expression"], donor=donor["expression"],
                    expression=text, operation=operation, reason=explanation)
        child = self.evaluate(text)                  # parse + pool.evaluate
        if child is not None and str(child) not in seen:
            children.append(child)
            seen.add(str(child))
    print(f"  Evaluated {len(children)} new offspring", flush=True)
    return children
```

LLM 必须返回三个等长（`args.offspring = 5`）数组：

```json
{
  "expressions":  ["完整表达式1", ..., "完整表达式5"],
  "operations":   ["mutation|replacement|crossover", ...],
  "explanations": ["保留/改动了什么，证据是什么，期望验证什么", ...]
}
```

任意数组长度不匹配 → 抛错（`ValueError`），本 round 失败立即终止，由外层 trainer 报错保留旧快照。

`evaluate(text)` 失败（parse / validate / 无 variation）只写 invalid 日志，**不抛错**，由 LLM 自行决定下一轮如何避免。

---

## 5.3 三种操作

`PROMPT_EVOLUTION` 明确要求 LLM 在 5 个 offspring 中**多样化**地分配三类操作：

### Mutation（变异）

保留某个有意义的机制，重写周围结构：

```text
f' = Mutate(m_keep, m_replace)

例：保留 short-term reversal 子树，把 abnormal-volume scaling 从 Mul 改成 Add，
得到 Add(Div($close,Ref($close,5)),1.0, Div($volume,TsMean($volume,20)))
```

### Replacement（替换）

把整个机制换成另一种经济或数学结构：

```text
Momentum → Reversal；Rank → Z-score；TsMean → TsEMA；
短窗口 TsStd(5) → 长窗口 TsStd(20)
```

### Crossover（交叉）

从 parent 与 donor 各自选出有强证据的机制，组合：

```text
f_child = Combine(M_A^selected, M_B^selected)
```

随机模式下 LLM 仍然**必须**用给定 donor，不能"另外找一个"。相同 parent == donor 时不许声称 crossover。

`existing_expressions` 提示 LLM 不要复制当前池或本轮已有子代。`historical_memory` 让 LLM 知道哪些结构已经被实证过（避免重复 ablation 没意义的改动）。

---

## 5.4 JSON 解析

`ask()` 内部（`trainer.py:35-52`）：

```python
def ask(self, prompt, context):
    user_prompt = PROMPT_FEATURES_AND_OPERATORS + prompt.format(
        **{k: json.dumps(v, ensure_ascii=False) for k, v in context.items()})
    self.record("llm_request", system_prompt=PROMPT_HEAD, user_prompt=user_prompt)
    text, finish_reason = self.model.chat_generate(
        self.client, system_prompt=PROMPT_HEAD, user_prompt=user_prompt,
        temperature=args.temperature)                                     # 默认 0.5
    self.record("llm_response", response=text, finish_reason=finish_reason)

    # 剥掉 <think> ... </think>
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.S).strip()
    try:
        result = json.loads(text)
    except json.JSONDecodeError:
        block = re.search(r"```(?:json)?\s*(.*?)\s*```", text, re.S)
        if block is None:
            raise ValueError("Model response is not a JSON object")
        result = json.loads(block[1])
    if not isinstance(result, dict):
        raise ValueError("Model response must be a JSON object")
    return result
```

要点：
- 先 strip 内部 think block，再尝试直接 json.loads；失败再尝试抓 ``` ``` 块；都失败才报错
- **不**做多轮会话修复（"再问一次"）
- `temperature=0.5`：希望保持一定多样性，同时仍按 prompt 给出结构合理 proposal
- 不维护多层结构契约过滤——LLM 自由给 proposal，由 `evaluate()` 决定能否解析
- 精确字符串去重：`seen` 用 `str(expr)` 字符串去重，避免父子代 / 兄弟重复

---

## 5.5 单轮节奏与失败模式

完整 `train()`（`trainer.py:167-197`）伪代码：

```python
def train(self, seeds):
    self.initialize(seeds)                                            # 走 Step 1

    for step in range(1, args.rounds + 1):                            # 默认 10
        self.step = step
        started = time.monotonic()
        parents = self.select_parents()                               # 20 个
        total   = sum(self.pool.signal(e) for e in self.pool.exprs)   # double 求和
        utility = self.pool.utility()

        # 1) 全部 parent 先诊断
        evidence = [self.diagnose(p, utility, total) for p in parents]

        # 2) 逐个 parent 演化
        children = []
        for parent in evidence:
            donors = [r for r in evidence if r is not parent] or [parent]
            donor  = random.choice(donors) if args.random_crossover else \
                     max(donors, key=lambda r: max((m["pool_credit"] or 0.) - (m["delta_cf"] or 0.)
                                                   for m in r["mechanisms"]), default=0.)
            children.extend(self.evolve(parent, donor, children))      # 每次 5 个
            self.pool.keep(self.pool.exprs + children)                # 裁 cache

        # 3) 去重（精确字符串）
        children = list({str(e): e for e in children}.values())

        # 4) 空集保护
        if not children:
            raise ValueError(f"Round {step} produced no usable offspring; see round_{step}.jsonl")

        # 5) refine（窗口网格）
        children = self.refine(children)

        # 6) 进入池
        old = {str(e) for e in self.pool.exprs}
        self.pool.update(children)
        added = [str(e) for e in self.pool.exprs if str(e) not in old]
        self.record("update", offspring=len(children), added=added, utility=self.pool.utility())
        self.save()
        print(f"  {len(children)} offspring, {len(added)} entered pool, {time.monotonic()-started:.1f}s")
    return self.pool.exprs
```

失败保护：
- 整轮没有任何可用 offspring → 抛错 + 保留上一轮快照（旧 `pool_<step-1>.json` 仍可读）
- 个别 offspring 无效 → 仅跳过，不影响整轮
- LLM 返回字段长度不匹配 → 抛错（同上）

---

## 5.6 数量与开销

| 量 | 默认 | 来源 |
|---|---:|---|
| parents | 20 | `--parents` |
| mechanisms/parent | 3 | `--mechanisms` |
| offspring/parent | 5 | `--offspring` |
| 每轮 LLM 调用 | 20 diagnose + 20 evolve = 40 次 | parents × (1+1) |
| 每轮总 offspring 上界 | 100 | parents × offspring |

实际进入池的不到 100——大量会被 parse / validate / 重复 过滤掉，最终进 `pool.update()` 通常 50~80 个；`select()` 再选 60 个保留。

---

## 5.7 关闭演化的开关

- `--no-memory`：把 `historical_memory` 设为 `[]`
- `--random-crossover`：donor 改为随机抽取
- 不提供 `--no-evolve` 开关——宏观演化是核心，关闭会破坏循环
