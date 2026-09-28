# 02 — 反事实诊断

对应代码：`src/alpha_cf/trainer.py::AlphaCFTrainer.diagnose`，辅以 `src/alpha_cf/expression.py::ablate/walk/at/replace/parse/validate` 与 `src/alpha_cf/prompt.py::PROMPT_DIAGNOSIS`。

每轮开始时先确定 `parents`，再对每个 parent 调一次 `diagnose(parent, utility, total)`。

---

## 2.1 选取 parents

`select_parents()`（详见 `trainer.py:79-87`）：

```python
ranked   = sorted(pool.exprs, key=lambda e: pool.evaluate(e)["reward"], reverse=True)
count    = min(args.parents, len(ranked))          # parents 默认 20

# 一半"最少访问" + 一半"最强 reward"
selected = sorted(ranked, key=lambda e: visits[str(e)])[:count // 2]
selected += [e for e in ranked if str(e) not in {str(x) for x in selected}][:count - len(selected)]

for e in selected: visits[str(e)] += 1
return selected
```

机制：
- `count // 2`（默认 10）个 parent 来自 `visits` 最少的池成员（鼓励探索从未/很少被诊断过的因子）
- 剩余 `count - len(selected)`（默认 10）个 parent 严格按 reward 降序填入（鼓励沿高质量因子继续演化）
- 每次进入 parent 列表都会自增 `visits[str(e)]`，所以"探索组"的位置会动态轮换
- 池里每个成员最终都会被选到（强 reward 的不会永远霸占 exploit 队列）

---

## 2.2 给 LLM 的诊断 prompt

调用 `self.ask(PROMPT_DIAGNOSIS, ctx)`，其中：

```python
ctx = dict(
    parent          = str(parent),                              # 整张表达式字符串
    reward          = pool.evaluate(parent)["reward"],          # 有符号 RankICIR 数值
    nodes           = [{"path": list(p), "expression": str(n)} for p, n in walk(parent)],
    mechanism_count = args.mechanisms,                          # 默认 3
    cf_enabled      = not args.no_cf_evidence,                  # 默认 True
)
```

`walk(parent)`（`src/alpha_cf/expression.py::walk`）：

```python
def walk(expr, path=()):
    yield path, expr
    for i, name in enumerate(children(expr)):
        yield from walk(getattr(expr, name), path + (i,))

def children(expr):
    if isinstance(expr, (UnaryOperator, RollingOperator)): return ("_operand",)
    if isinstance(expr, (BinaryOperator, PairRollingOperator)): return ("_lhs", "_rhs")
    return ()
```

约定：根节点 `[]`；unary/rolling 的输入是 child 0；binary/pair-rolling 的两个输入分别是 child 0、1；窗口常量不是 child。

`PROMPT_DIAGNOSIS` 要求 LLM 返回：

```json
{
  "mechanisms": [
    {
      "path": [int, ...],          // AST path，必须在 nodes 里出现过
      "description": "string",     // 简短金融/数学含义
      "mode": "remove|neutralize|null",
      "replacement": "string|null", // 新子树表达式
      "reason": "string"            // 干预动机
    }
  ]
}
```

上限：`len(mechanisms) ≤ args.mechanisms = 3`。LLM 可少给；不能填 0 个以外的"水分"项（去掉常量、符号脚手架、`epsilon` 守卫、相关项里的居中等都不算机制）。

---

## 2.3 实际执行 ablation

`diagnose` 主体（`trainer.py:89-121`）伪代码：

```python
def diagnose(self, parent, utility, total):
    reward = pool.evaluate(parent)["reward"]

    # 1) LLM 提议（一次性）
    proposals = self.ask(PROMPT_DIAGNOSIS, {...})["mechanisms"]
    evidence = []
    for mechanism in proposals[: args.mechanisms]:                  # 截断到 3 个
        try:
            node = at(parent, tuple(mechanism["path"]))             # 按 AST path 取子树
            row = dict(
                factor     = str(parent),
                expression = str(node),                             # 被干预的子树字符串
                **mechanism,
                delta_cf   = None,
                pool_credit = None,
                action     = "Explore",
            )
            # 2) 仅当 cf_enabled 且 LLM 给了 mode 才真的做干预
            if not args.no_cf_evidence and mechanism.get("mode"):
                changed = ablate(parent, mechanism, args)           # 返回新表达式
                result  = pool.evaluate(changed)
                row["ablation"]  = str(changed)
                row["delta_cf"]  = result["reward"] - reward        # ★ 个体证据
                if not args.no_pool_credit:
                    # 用 (changed) 替换 (parent) 后的池等权信号
                    signal = (total - pool.signal(parent) + pool.signal(changed)) / len(pool.exprs)
                    row["pool_credit"] = utility - pool.score_signals(signal.unsqueeze(0))[0].item()
                credit = row["pool_credit"] or 0.0
                # 3) action 标签
                if   row["delta_cf"] < -1e-5 and credit >= -1e-5: row["action"] = "Preserve"
                elif row["delta_cf"] >  1e-5 and credit <=  1e-5: row["action"] = "Replace"
                else:                                              row["action"] = "Explore"
            evidence.append(row)
            self.record("mechanism", **row)                          # → round_<step>.jsonl
        except (ValueError, KeyError, TypeError, IndexError) as exc:
            self.record("invalid_mechanism", factor=str(parent), proposal=mechanism, error=str(exc))
    # 4) 仅"成功 evaluate 的"才进长期 memory（unmeasured 不进）
    measured = [r for r in evidence if r["delta_cf"] is not None]
    self.memory.extend(measured)
    print(f"  Diagnosed {len(evidence)} mechanisms, measured {len(measured)}", flush=True)
    return dict(expression=str(parent), reward=reward, mechanisms=evidence)
```

每个 round 内先全部 `diagnose` 完再开始 `evolve`——这是有意为之，确保 donor 的 evidence 已经可用。

---

## 2.4 ablation 的程序化保证

`expression.py::ablate(expr, mechanism, args)`：

```python
def ablate(expr, mechanism, args):
    path = mechanism["path"]
    node = at(expr, path)                                              # 深拷贝取子树
    baseline = parse(mechanism["replacement"], args, featured=False)   # 解析但不强制带 feature

    if mechanism["mode"] == "remove":
        # remove 必须保留被移除子树的一个真正后代
        descendants = {str(n) for p, n in walk(node) if p}              # path 非空 = 后代
        if str(baseline) not in descendants:
            raise ValueError("Remove must retain a descendant of the mechanism")

    elif mechanism["mode"] == "neutralize":
        # neutralize 的 baseline 只能用到 node 已有的 feature 输入，不能新增输入
        node_inputs    = {str(n) for _, n in walk(node)   if isinstance(n, Feature)}
        baseline_inputs = {str(n) for _, n in walk(baseline) if isinstance(n, Feature)}
        if not baseline_inputs.issubset(node_inputs):
            raise ValueError("Neutralization introduces a new input")

    else:
        raise ValueError("Expected remove or neutralize")

    # 完整 validate：AST 节点数 ≤ 60、深度 < 10、lookback ≤ 100、带 feature
    result = validate(replace(expr, path, baseline), args)
    if str(result) == str(expr):
        raise ValueError("Unchanged intervention")
    return result
```

要点：
- `replace` 深拷贝整个 expr 后按 path 替换，所以 ablation 之间互不影响
- `validate` 既检查节点数/深度/lookback，也检查 Ref/TsDelta 窗口合法性
- baseline 是 LLM 写的完整表达式字符串，必须能被 `ExpressionParser` 解析

---

## 2.5 reward 与 credit 的数值含义

- **delta_cf**（个体证据）：
  $$\Delta^{\text{cf}}_i = R(T(f, m_i)) - R(f)$$
  - `< 0`：移除后 RankICIR 变差 → 机制对个体有正贡献
  - `≈ 0`：贡献小
  - `> 0`：移除后 RankICIR 变好 → 机制可以替换

- **pool_credit**（组合证据）：
  $$C_{\text{pool}}(m_i; f, P) = U(P) - U\!\left(P_{-f}\cup\{T(f, m_i)\}\right)$$
  程序实现上：
  ```
  signal_new = (total - pool.signal(f) + pool.signal(changed)) / len(pool.exprs)
  pool_credit = utility - pool.score_signals(signal_new.unsqueeze(0))[0]
  ```
  - `> 0`：去掉这个机制让 pool 变差 → 机制对 pool 有正贡献
  - `< 0`：去掉这个机制让 pool 变好 → 机制拖累 pool
  - 阈值 ±1e-5（基于 `delta_cf` 在浮点累加下的噪声级）

- **action**（参考性标签，不强制后续操作）：
  - `Preserve`：个体证据支持（delta_cf<0），组合证据不反对（credit≥-1e-5）
  - `Replace`：个体证据反对（delta_cf>0），组合证据不支持（credit≤1e-5）
  - `Explore`：其他所有情况（近零、冲突、未测量）

LLM 仍然看到原始数值（不是文字标签），标签只用于本步的 `record` 与下游 `donor` 选取。

---

## 2.6 输出与日志

每个 parent 诊断结束会：
- 把每条 mechanism 写入 `round_<step>.jsonl::mechanism`（含 path、expression、ablation、delta_cf、pool_credit、action）
- 把无效提议写入 `round_<step>.jsonl::invalid_mechanism`
- 把本 parent 的 evidence 字典返回给 `train()`，作为 `evolve()` 的输入

未测量（`delta_cf=None`）的提议不进 `self.memory`——长期历史只保留真实干预的实证。但 LLM 在 evolution 时仍能看到这些提议作为当前 parent 的上下文。

---

## 2.7 整体约束

- 一次 round 内每个 parent 至多 3 个机制（`args.mechanisms=3`），默认按提议顺序截断
- 所有 ablation 都从原 parent 出发，不级联叠加；不维护"机制效果相加"的假设
- `no_cf_evidence` flag 会跳过 ablation 计算（只保留 LLM 的语义分解），便于消融
- 任何 parse/validate/evaluate 失败都不抛出，仅记 invalid 日志后跳过
