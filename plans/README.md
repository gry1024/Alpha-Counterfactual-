# 实现索引

idea、plans 与 `src/alpha_cf/` 使用同一协议：

1. [初始化](01_Initialize.md)：直接加载离线比较确定的 50 个精华种子，保持顺序，不重选；[方案与收益曲线](Start_Pool_Comparison.md)。
2. [反事实诊断](02_Counterfactual_Diagnosis.md)：默认 10 个 parent，一半名额从低贡献半池抽样，其余从未选成员随机抽样；LLM 先解释因子，再自定 0–5 个带假设/预测的机制。
3. [机制证据](03_Mechanism_Credit.md)：delta_cf、pool_credit、signal_distance 三项核心证据，另展示覆盖变化和共同样本排序相关度。
4. [机制记忆](04_Mechanism_Memory.md)：每个种子的固定 chain ID 与全部演化历史，保存理解及证据后的修正，下一次诊断复用；子代继承，不跨链共享。
5. [直接生成子代](05_Macro_Evolution.md)：LLM 按 mechanism 量级与 factor 复杂度自主生成每 parent 0–5 个子代，同次调用先修正理解，每项包含完整公式、当前证据引用及基于理解的修改/假设描述，无操作类型约束。
6. [父因子替换](07_Pool_Update.md)：简化与改善统一竞争，完整排名等价组保留最简版本后按 S 比较，最多一个子代替换其 parent。

## 默认参数

| 参数 | 默认 | 说明 |
|---|---:|---|
| --rounds | 20 | 演化轮数 |
| --parents | 10 | 默认 10 个 parent，从当前池做一半名额从低贡献半池抽样，其余从未选成员随机抽样 |
| --diagnosis-workers | 5 | 仅 LLM 诊断并发；证据测量、memory、演化与替换按原顺序执行 |
| --correlation-threshold | 0.8 | 子代与 peers 最大日均绝对 Spearman 的更新硬门槛（不超过该值） |
| --pool-capacity | 50 | 工作池容量，演化中保持不变 |
| --horizon | 20 | 前向收益跨度 |
| --max-nodes / --max-depth / --max-backtrack | 60 / 10 / 100 | 表达式执行边界 |
| --chunk-size | 64 | 评价分块天数 |
| --alpha / --beta / --gamma / --cost-weight | 50 / 500 / 2 / 2 | 固定自然尺度系数，原始系数不要求和为 1 |
| --temperature | 0.5 | LLM 温度 |
| --n-factors | 20 | 原回测脚本每天最多选取因子数 |
| --llm-timeout | 180 | 单次 SDK timeout；ask 重试预算为 3 倍 timeout，默认 540s |

pool_credit 门槛固定为 1e-6，严格大于，无新增参数。父代低贡献集合由删除贡献 U(P)-U(P去掉成员) 排名后半池确定；名额为奇数时随机部分多一个，最终顺序打乱。
机制数和子代数均没有命令行参数。机制数由 LLM 依复杂度选择，不超过 5；
子代数由 LLM 依 mechanism 数量、规模与 factor 复杂度选择 0–5 个，允许主动不生成。
R 与 U 均为 signed RankIC；delta_cf=|R(cf)|-|R(parent)|，pool_credit=U(替换后)-U(替换前)，正值分别表示个体与组合改善。
U、pool_credit 与快照权重共用完整 Train 的 in-sample OLS，不做内部时间切分。最终回测使用 z-score 信号与逐日历史窗口拟合。
初始化校验全部 50 个成员；memory 只作为 LLM context，反事实公式不自动参与替换。
signal_distance 使用完整 Train 上的平均并列秩归一化信号。将信号展平，缺失填中性 0，记为 a、b；q 为有效观测交集数量 / 并集数量。定义 d = 1 − q·|cos(a,b)|，范围 [0,1]：0 表示完整排序等价（含一致的整体反向），1 表示没有相关排序信息或有效观测不重叠。全零信号双方的 cosine 取 1，仅一方全零时取 0；有效观测并集为空时记 null。使用全时段统一方向，不把每天不同的方向翻转视为等价。

等价分组另外要求有效样本掩码完全相同、完整归一化排名以绝对容差 1e-6 一致或整体反向一致；不能仅凭 distance 很小就删除保护项或合并公式。近零距离不证明未来或符号代数等价。
每次演化传入该链全部此前历史，首次演化才为空（--no-memory 消融除外）；多样性仍使用 mean_daily_abs_Spearman。
诊断、生成同时考虑简化与质量提升；子代必须满足相关性硬门槛与 pool_credit > 1e-6，再统一竞争；等价简化无例外。

## 完成与验证

- [x] 移除诊断处置标签、编辑类型枚举及演化操作分类。
- [x] 新增信号距离，允许等价简化、窗口调整及整式经济重写。
- [x] 默认 parent 改为 10，每 parent 由 LLM 自主决定直接生成 0–5 个子代。
- [x] 按 parent 原位替换，失败保留，记录候选判定与演化路径。
- [x] 同步 idea、plans、README 和入口参数。
- [x] 初始化直接加载冻结种子池，移除重选及其消融开关，补充 Train 终端明细。
- 自动化验证见 `tests/test_cf_evolution.py`，使用合成张量与模拟 LLM；
  不依赖外部 API，不替代真实市场回测。（注：仓库当前不包含 `tests/`
  目录与对应脚本，需自行补齐；以下条目描述的是设计意图而非当前状态。）

复用原表达式库、parser、Qlib、统计函数和最终 adaptive combination 回测。
新逻辑集中于 alpha_cf；不引入额外搜索框架、向量库、参数枚举或最终重选阶段。
