# 04 — 机制记忆与路径

实现：`AlphaCFTrainer.memory/diagnose/evolve/save/train`。

## 机制记录

```text
{
  factor: 原 parent 表达式,
  path: AST 路径,
  expression: 对应原 subtree,
  description: 机制语义,
  replacement: 替换 subtree 的表达式,
  reason: 本次编辑要检验的假设,
  counterfactual: 修改后的完整公式,
  delta_cf: float,
  pool_credit: float | null,
  signal_distance: float
}
```

仅 delta_cf 测量成功的记录进入 memory；pool_credit 在对应消融时可为 null。
只保存实测证据与编辑说明，不包含编辑类型或后续处置分类。
每次 evolve 只传入该 parent 自身的历史 memory（按 factor 表达式过滤），不跨因子共享；`--no-memory` 传空列表。
每轮 save 覆盖 `memory.json`；逐条诊断同时追加轮次 JSONL。
证据适用于测量时的 parent 和 pool；迁移后的价值需重新验证。

## 演化路径

`lineage.json` 保存初始化 seed 事件和每个 parent 的替换/保留事件。
字段为 round、parent/parent_id、child/child_id、outcome，
替换判定还带 parent_reward、parent_complexity、全部有效新候选的指标、eligible 和 description。
description 是 LLM 对机制证据与修改/探索的简短说明；入池子代另存 child_description，
保留 parent 时 child_description 为 null。
初始成员与每个入池子代都有新的 f0、f1、… ID。
保留时 child/child_id 为 null，原 parent ID 不变；再次出现的表达式分配新 ID。

`pool_<round>.json` 包含与 exprs 对齐的 factor_ids、visits。
用 child_id → parent_id 回溯，直到 outcome=seed 即可恢复一条完整路径。
`round_<round>.jsonl` 同步记录 replacement，且 proposal 含 parent_id、expression 和 description；
因此包括未入池、无效公式在内的假设均可关联到演化实例。
快照继续保持原 exprs/weights/metrics/utility/split，原回测读取兼容。
