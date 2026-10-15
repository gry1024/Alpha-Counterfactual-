# 07 — 按父因子替换工作池

实现：`AlphaCFPool.initialize/replace_parent/keep`、
`AlphaCFTrainer.train/save`。

## 初始化与演化分离

初始化直接加载离线确定的 50 个成员，校验数量、唯一性与可用性，不打分重选。
离线构建与收益比较见 `plot_profit_curve_start_pool.py` 和 初始池比较。
默认容量 50；数量不匹配或不可用即报错。四项评分权重只用于演化阶段的父因子替换。

演化阶段不合并候选，也不再重选固定数量。
每个 parent 的 offspring 数量由 LLM 根据 mechanism 量级和 factor 复杂度自主决定为 0–5 个，
只竞争该 parent 的位置。生成 0 个时直接保留 parent，并记录 retained。

## 替换条件

在等价分组前，子代必须同时满足 max_correlation <= correlation_threshold（默认 0.8，peers 排除原 parent）与 pool_credit > 1e-6（固定严格门槛）。无法计算的相关性按 1 处理。全部子代统一执行，无等价简化例外；parent 自身只作为保留基准，不过滤。

parent 与通过两个硬门槛的有效新子代统一竞争，先按完整有效掩码和归一化排名一致/整体反向一致分组，每组只保留 complexity 最低的公式；复杂度相同保留原顺序。等价检查见 [机制证据](03_Mechanism_Credit.md)，不能仅凭共享样本相关性或 distance 阈值合并。

每个组代表直接按 S=α|R|+βC_pool+γD+λ(1-C_cost) 评分，默认权重 5/800/2/2。不使用候选集合缩放；C_pool 使用完整 Train 的原始组合增益。

评分消融 `--no-pool-selection` 将 β 置零并取消正信用门槛，其他逻辑不变；α/β/γ/λ 设零分别删除对应评分项，β=0 单独保留正信用门槛，γ=0 保留相关性门槛。

不同信号组必须严格高于 parent 所在组代表的 S 才能替换，取最高分者；分数相同先取较低 complexity，再保持原顺序。若没有不同信号组改善，parent 所在组仍保留最简公式。选中 parent 组的更简子代记 equivalent_simplification，其他改善记 score_improvement。简化不提前截断改善竞争。

子代不得与池内公式完全重复；complexity 是 AST 节点数，忽略一层最外侧方向包装。等价分组只针对该 parent 与其 offspring，不合并池中其他槽位。每个 parent 最多原位替换一次，容量不变；无有效子代则保留 parent。

## 日志与产物

每个 parent 写 replacement 事件及 lineage，包含候选的 reward、complexity、
max_correlation、signal_distance、pool_credit、cost、score、correlation_ok、credit_ok、eligible、description，以及 representative（是否为等价组代表）、parent_score、parent_group_score、parent_components、reason 和最终 replaced/retained 结果。
入池子代另保存 child_description，保留时为 null；描述只用于解释和追溯，不参与数值筛选。
全无有效子代时仍记录 retained，继续训练，不因空轮报错。
每个 parent 处理完 `keep(pool.exprs)` 释放未入池信号。
每轮保存 pool、memory、lineage；路径 ID 规则见 [机制记忆](04_Mechanism_Memory.md)。

打印沿用 alpha_knowledge 的迭代标题、`> Alpha #...` 池列表及 `[Pool Add/Pop/Reject]` 风格。
每轮打印反事实推理/演化进度、父因子、候选的 RankIC、复杂度、最大相关性、距离、C_pool、S 与资格，说明等价简化或得分提升的替换原因；无新候选、相关性门槛失败、贡献门槛失败与无得分提升分别记录保留原因；打印候选的 correlation_ok、credit_ok。
初始化及每轮保存时打印完整池的 factor ID、signed RankIC、|R|、cost、complexity、OLS 权重、表达式，以及最佳/平均 |R|、组合 U、memory 数量和保存路径。搜索与池更新只使用 Train 指标；每轮另行打印 Validation/Test 回测指标。

最终 `pool_<rounds>.json` 即最终池，split=train。
`train_cf.py` 在初始化（round 0）及每轮保存后调用 `run_adaptive_combination.py`，使用冻结的 Train OLS 权重同时评价 Validation/Test，仅记录，不参与搜索决策。每轮指标保存为 `pool_<round>.metrics.json` 并追加到 `res.txt`。`adaptive_combination.png` 绘制两组 IC、RankIC、收益、ICIR、RankICIR、Sharpe 随轮次变化；`profit_curves.png` 绘制当前轮两组累计收益。Validation 日收益和日期保存为 `valid_ret_s.npy`、`valid_ret_s_dates.npy`，Test 保持 `ret_s.npy`、`ret_s_dates.npy`。
Validation/Test 切分由原脚本内部按 `--train_end_year 2021` 处理，AlphaCREAM 不再单独维护 valid。Test 范围为 2023.05.01–2026.04.30。
`--test-only` 仍可用保存池复跑最终回测。
