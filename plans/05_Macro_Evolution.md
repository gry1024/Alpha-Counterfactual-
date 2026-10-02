# 05 — 直接生成子代

实现：`AlphaCFTrainer.ask/evolve/train`，`PROMPT_EVOLUTION`。

LLM 根据每个 parent 的 mechanism 数量、规模和 factor 复杂度，自主决定生成 0–5 个完整公式。
不设固定数量参数，不要求凑数；默认 10 个 parent，一轮生成 0–50 个候选。
模型结合 parent 机制的 delta_cf、pool_credit、signal_distance、
该 parent 自身的历史 memory、当前池和已生成表达式，提出去冗余、新颖、有效的改进。

不规定操作类别、操作比例或必须保留某段结构。
可以简化、改变窗口、引入新结构或重写经济解释。
直接返回生成的最终子代，无细化、参数枚举或修复草稿阶段：

```json
{"offspring": [{"expression": "完整子代公式", "description": "恒等乘法的 signal_distance≈0 且 delta_cf≈0，因此去掉该乘法以减少冗余。"}]}
```

示例长度仅为展示；实际允许 0–5 个对象。没有值得尝试的改进时返回
`{"offspring": []}`，直接保留 parent，照常记录 retained 与演化路径。
每个对象包含非空字符串 expression 和 description。description 用 1–2 句话说明
依据哪个 mechanism 的哪些证据（delta_cf、pool_credit、signal_distance），
做了什么修改或探索、为什么。上例只适用于实际证据接近零的情况；
缺少实测证据时明确说明是探索假设，不编造测量值。不要求操作类型分类。
现有 ask 检查 JSON、字段类型、非空内容及数量，
数量只检查不超过 5，空列表合法；保留网络调用与最多三次响应解析尝试的现有行为。

每个返回公式写 proposal（含 parent_id、parent、expression、description），
随后 parse → AST 合法性检查 → Train 数值评价 → 精确去重。
公式不自动翻转符号。无效或重复公式跳过，其描述仍保留在 proposal 日志；
有效新候选的描述按规范化表达式关联，写入 replacement/lineage 的 candidates.description，
入池者另写 child_description。未入池候选不用于其他 parent 的替换。
模型返回协议错误沿用报错行为；全部候选无效或无合格候选则正常保留 parent。

所有诊断先基于轮初池完成，然后顺序执行：

```text
for parent:
    children = 直接生成并评价
    decision = pool.replace_parent(parent, children)
    记录路径及替换判定
    释放当前池以外的 signal 缓存
save()
```

后续 parent 的门槛检查使用已经更新的池。
具体入池标准见 [工作池更新](07_Pool_Update.md)。
