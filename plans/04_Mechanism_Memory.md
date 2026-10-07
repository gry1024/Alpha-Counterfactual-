# 04 — 机制记忆与路径

实现：`AlphaCFTrainer.memory/diagnose/evolve/save/train`。

## 独立演化链

每个初始因子以初始 factor ID 作为固定 `chain_id`。父代替换后，新的
factor ID 与公式继承同一 chain ID；保留时也继续同一链，不跨链共享。

`memory.json` 是 chain ID 到完整演化历史列表的映射，每条记录为：

```text
{
  parent: {expression, reward, complexity, understanding, mechanisms, updated_understanding, offspring},
  decision: {round, chain_id, parent_id, child_id, parent, child,
             outcome, reason, candidates, ...}
}
```

`offspring` 保存全部原始子代提案，包括无效、重复或未入池提案；
`decision.candidates` 保存有效新候选的评价、说明与资格。
没有机制、没有子代、未测量或保留 parent 的轮次也保存。

每次 `evolve` 直接传入该链按发生顺序保存的全部此前历史，不按公式过滤、不截断。
本轮证据在 `parent.mechanisms` 中提供，决定完成后才追加到 memory，避免重复。
正常模式下只有该链首次演化时历史为空；下一次诊断传上一条 parent 的 expression 与 updated_understanding，注明可能属于祖先。`--no-memory` 显式传空历史及空的 previous_understanding，但仍记录历史。

## 机制证据

```text
{
  id: 本次机制编号 m1/m2/... ,
  factor: 原 parent 表达式,
  path: AST 路径,
  expression: 对应原 subtree,
  description: 机制语义,
  replacement: 替换 subtree 的表达式,
  reason: 本次编辑要检验的假设与定性预测,
  counterfactual: 修改后的完整公式,
  delta_cf: float,
  pool_credit: float | null,
  signal_distance: float | null,
  coverage_change: float | null,
  common_rank_correlation: float | null
}
```

机制放在对应历史记录的 `parent.mechanisms` 内；禁用或无法计算的指标为 null，不能当成零。
不包含预设编辑类型或处置分类。
每轮 save 覆盖 `memory.json`；逐条诊断同时追加轮次 JSONL。
证据适用于测量时的 parent 和 pool；迁移后的价值需重新验证。

Memory 只作为 LLM 提出进化子代的 context，正向、负向证据与失败尝试均保留。
不自动把反事实公式加入候选，也不强制 LLM 复用某条编辑；由 LLM 根据当前证据和历史自主生成 offspring。

## 演化路径

`lineage.json` 保存初始化 seed 事件和每个 parent 的替换/保留事件。
字段为 round、chain_id、parent/parent_id、child/child_id、outcome，
替换判定还带 parent_reward、parent_complexity、parent_score、reason、全部有效新候选的指标、eligible 和 description。
候选指标包含 reward、complexity、max_correlation、signal_distance、pool_credit、cost、score、correlation_ok、credit_ok 和 representative（是否为等价组代表）；决定另含 parent_group_score。
description 是 LLM 对机制证据与修改/探索的简短说明；每个提案和有效候选另存 evidence_refs（当前机制编号或探索时的空列表）；入池子代另存 child_description 与 child_evidence_refs，
保留 parent 时 child_description 为 null。
初始成员与每个入池子代都有新的 f0、f1、… ID。
保留时 child/child_id 为 null，原 parent ID 不变；再次出现的表达式分配新 ID。

`pool_<round>.json` 包含与 exprs 对齐的 factor_ids、chain_ids 及 visits。
用 child_id → parent_id 回溯，直到 outcome=seed 即可恢复一条完整路径。
`round_<round>.jsonl` 同步记录 replacement，且 proposal 含 parent_id、expression、description 和 evidence_refs；
因此包括未入池、无效公式在内的假设均可关联到演化实例。
快照保持原 exprs/weights/metrics/utility/split，原回测读取兼容；utility、weights 与 pool_credit 共用完整 Train OLS 口径，不保存分段 credit 字段。
