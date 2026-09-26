# Step 4 — Mechanism Memory

输入：已测量 intervention 记录。输出：memory.json，以及提供给 evolution 的有限历史上下文。

代码：AlphaCFTrainer.memory/retrieve_memory/visible_evidence/save。存储为 list[dict]。

## 写入

成功测得 delta_cf 即保存；pool credit 无法计算时保留 null 和原因。未测量或数值无效的提议仅在诊断日志和本轮 evidence 中，不混入已验证记忆。

每轮保存全量 memory；记录原 factor、机制路径、具体 baseline、轮次和池快照，避免把 credit 当成永久属性。

## 检索

retrieve_memory(parents) 仅取早于本轮的记录，默认最多 memory_limit=24：

```text
query = 当前父本的特征名与算子名集合
related = 按机制与 query 的 Jaccard 重合度排序
positive / negative / mixed = 按两项证据符号分组
这三组各按 max(abs(delta),abs(credit)) 选代表记录
recent = 按历史轮次倒序
从五路队列轮流取未重复记录，直到 limit
```

数值强度仅用于检索代表记录，不是 evolution 的固定加权策略。相同 step/factor/path/intervention 去重；不同上下文不合并 credit。本轮 evidence 独立提供，不占历史条数。

使用 regex 提取现有算子名和特征名，不增加 embedding、数据库或新的 AST 表示。memory_retrieved 日志记录实际选出的历史来源。

## 使用边界

历史结构可以启发新表达式；crossover 双方仍为当前父本。旧完整因子不因 memory 回到候选集合，新 child 或参数实例也不继承旧 credit。

--no-memory 或 memory_limit=0 不提供历史，仍保留本轮证据。--no-pool-credit 时，检索分组/强度及可见记录均不使用 pool credit；对应数值字段从 prompt 上下文移除。
