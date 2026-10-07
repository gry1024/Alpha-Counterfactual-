# 03 — 机制证据

实现：`AlphaCFTrainer.diagnose`；
`AlphaCFPool.evaluate/signal_distance/signal_comparison/score_signals/utility/equivalent`。

设 parent=f，反事实=f'，轮初池=P：

| 字段 | 公式 | 解释 |
|---|---|---|
| delta_cf | \|R(f')\| - \|R(f)\| | 负值表示本次编辑损害个体预测力度 |
| pool_credit | C_pool(f'\|f,P) = U(P\{f}∪{f'}) - U(P) | 正值表示本次编辑提升组合 |
| signal_distance | 1 − q·abs(cos(a,b)) | 完整排名相关度与有效观测覆盖，范围 [0,1] |
| coverage_change | 有效掩码异或数量 / 并集数量 | 哪些观测从有效变无效或反之，范围 [0,1] |
| common_rank_correlation | 每日共同有效样本上重新排名的 signed Spearman 的有效日均值 | 区分共同样本排序变化与覆盖变化，无法计算时 null |

R 是有符号日均横截面 RankIC，U 是组合的有符号日均 RankIC；个体质量以 |R| 衡量，**不区分正负预测方向**（与 alpha_knowledge 保持一致）。
signal_distance 使用完整 Train 上的平均并列秩归一化信号。将信号展平，缺失填中性 0，记为 a、b；q 为有效观测交集数量 / 并集数量。定义 d = 1 − q·|cos(a,b)|，范围 [0,1]：0 表示完整排序等价（含一致的整体反向），1 表示没有相关排序信息或有效观测不重叠。全零信号双方的 cosine 取 1，仅一方全零时取 0；有效观测并集为空时记 null。使用全时段统一方向，不把每天不同的方向翻转视为等价。

等价分组另外要求有效样本掩码完全相同、完整归一化排名以绝对容差 1e-6 一致或整体反向一致；不能仅凭 distance 很小就删除保护项或合并公式。近零距离不证明未来或符号代数等价。

两项新增统计只解释反事实，不参与评分或等价判定。复用缓存的归一化信号和现有 Spearman；例如删除分母保护项时，共同排序可能不变但覆盖变化非零。

pool_credit 使用完整 Train（2015–2021）的效用 U；原池与替换池各自拟合带符号 OLS，并在同一完整 Train 上评价，不做内部时间切分。

C_pool = U(P\{parent}∪{candidate}) − U(P)。诊断用轮初池，替换用即时当前池；parent 自身的贡献为 0。日志与快照共用同一效用和权重口径。OLS 统一 dtype/device，缺失曝光取中性 0；无足够样本、非有限系数或求解失败显式报错，不用等权兜底。
单因子 R 在全部标签有效日期上平均；因子恒定、缺失或无法计算的 IC 按 0 计入，完全没有有效 IC 的因子拒绝。标签有效指至少两个有限值且截面非恒定。

代码、文档和 LLM prompt 均使用上述定义：delta_cf 的正值表示个体预测力度改善，pool_credit 的正值表示组合改善。负向诊断仍是有效证据，不因符号被过滤。终端逐机制打印编号、路径、描述、假设/预测、编辑公式和全部实测指标；消融禁用的指标显示 `unmeasured`。

不根据两项 credit 的符号生成处置分类，LLM 自主理解收益、冗余和组合之间的权衡。
`--no-pool-credit` 仅使诊断 pool_credit=null，仍测量其余证据；更新时 pool_credit > 1e-6 的硬门槛仍执行。
`--no-cf-evidence` 使全部证据为 null。
null 表示未测量，不能当成零贡献。
