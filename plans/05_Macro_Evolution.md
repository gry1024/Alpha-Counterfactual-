# 05 — 直接生成子代

实现：`AlphaCFTrainer.ask/evolve/train`，`PROMPT_EVOLUTION`。

LLM 根据每个 parent 的 mechanism 数量、规模和 factor 复杂度，自主决定生成 0–5 个完整公式。
不设固定数量参数，不要求凑数；默认 10 个 parent，一轮生成 0–50 个候选。
模型结合 parent 机制的 delta_cf、pool_credit、signal_distance、
该因子独立演化链的全部此前历史（祖先证据、全部提案与替换/保留决定）、当前池和已生成表达式，提出去冗余、新颖、有效的改进。

当前机制证据与 memory 均只作为生成 context。只有 LLM 返回的 offspring 才参与验证和替换；反事实编辑不自动成为候选。LLM 自主决定是否复现已有编辑，或依据正向、负向证据提出新的结构假设。

简化与质量改善可以同时提出，不因发现冗余而停止窗口、输入、新结构或经济解释探索。
signal_distance 使用覆盖修正的完整排名相关距离 [0,1]；等价分组额外校验完整有效掩码及排名一致/整体反向一致，子代先通过相关性硬门槛与 pool_credit > 1e-6，再保留等价组最简版本并统一评分；纯零贡献简化不能入池。
同一次演化调用先返回 updated_understanding（最多三句），对照 reason 中的预测与当前证据，引用机制编号说明支持、矛盾、结构作用和未决问题。无测量时明确不确定，不把正信用当作迁移后有效或金融因果证明。然后直接返回最终子代，无额外反思调用或修复草稿阶段：

```json
{"updated_understanding": "m1支持外层恒等乘法不改变有效覆盖和排序。", "offspring": [{"expression": "完整子代公式", "evidence_refs": ["m1"], "description": "依据m1尝试简化，仍需检验入池信用。"}]}
```

示例长度仅为展示；实际允许 0–5 个对象。没有值得尝试的改进时返回
`{"updated_understanding": "证据更新或仍未解决的问题", "offspring": []}`，直接保留 parent，照常记录理解、retained 与演化路径。
每个对象包含非空字符串 expression 和 description，以及 evidence_refs 列表。引用只允许本次成功诊断的 id（失败项不可引用），允许引用负证据；无直接当前机制依据时使用 [] 并标明探索，历史依据在描述中说明。description 用 1–2 句话把 updated_understanding 连接到修改与待验证假设，说明
依据哪个 mechanism 的哪些证据（delta_cf、pool_credit、signal_distance），
做了什么修改或探索、为什么。上例只适用于实际证据接近零的情况；
缺少实测证据时明确说明是探索假设，不编造测量值。不要求操作类型分类。
现有 ask 检查 JSON、字段类型、非空内容及数量，
数量只检查不超过 5，空列表合法；直接用已有 OpenAI 兼容客户端，关闭 SDK 自动重试。网络失败与响应纠错共用最多三次请求；连接/超时及 HTTP 408/409/429/5xx 重试前等待最多两秒，其余 HTTP 错误立即报错。
单次 SDK timeout 默认180秒；整个 ask 的重试调度预算为三倍 timeout（默认540秒），等待计入预算，剩余预算不足时缩短请求 timeout，耗尽后不再发请求。单次调用仍受 SDK 超时语义约束。日志包含请求开始时间、timeout、响应/失败耗时和错误类型，不记录网络错误响应正文。
每次解码或协议失败，下一次 prompt 加入上一次具体错误（包括 JSON 行列位置），要求修正并返回合规 JSON。

每个返回公式写 proposal（含 parent_id、parent、expression、description、evidence_refs），
随后 parse → AST 合法性检查 → Train 数值评价 → 精确去重。
公式不自动翻转符号。无效或重复公式跳过，其描述仍保留在 proposal 日志；
有效新候选的描述按规范化表达式关联，写入 replacement/lineage 的 candidates.description，
入池者另写 child_description。未入池候选不用于其他 parent 的替换。
模型返回协议错误沿用报错行为；全部候选无效或无合格候选则正常保留 parent。

默认 LLM 诊断最多 5 路并发，证据在主线程按原顺序测量、基于轮初池完成，然后顺序执行：

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

终端与既有日志保存 understanding、updated_understanding、机制预测及证据引用；下一轮诊断复用上一轮理解。终端输出 parent、机制/记忆记录数、每个 proposal 的说明与评价结果，以及重复或无效公式的跳过原因。
