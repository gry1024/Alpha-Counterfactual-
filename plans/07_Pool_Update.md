# 07 — 按父因子替换工作池

实现：`AlphaCFPool.initialize/select/replace_parent/keep`、
`AlphaCFTrainer.train/save`。

## 初始化与演化分离

仅初始化 `initialize(candidates)` 调贪心 select 选 pool_capacity 个，
沿用 |R|、组合贡献、多样性和 (1-cost) 四项**自然尺度**加权，权重为固定的 (α,β,γ,λ)，
不再要求和为 1；其语义是"每个维度每单位的物理贡献"。
默认容量 50，候选不足则报错；`--no-pool-selection` 仅影响初始化，改为按 R 选。

演化阶段不合并候选，也不再重选固定数量。
每个 parent 的 offspring 数量由 LLM 根据 mechanism 量级和 factor 复杂度自主决定为 0–5 个，
只竞争该 parent 的位置。生成 0 个时直接保留 parent，并记录 retained。

## 替换条件

替换分两种情形：

**情形 1 等价去冗余**：子代 signal_distance（1 - mean Spearman(parent, child)）≤ 1e-6
且 complexity(child) < complexity(parent)，即排序信号等价且节点更少，直接替换。
这覆盖删除无意义算子（如 Mul(1.0,f)、Add(0.0,f)）。

**情形 2 组合得分提升**：其余子代与 parent 一起计算
S(f|P) = α·|R| + β·C_pool + γ·D + λ·(1-C_cost)，
其中 C_pool = U(P_{-f}∪{f}) - U(P)，P_{-f} 取当前池去掉该 parent（与
[机制证据](03_Mechanism_Credit.md) 中 pool_credit 同一公式，不重复定义）。
U 由 in-sample OLS 回归得到，与 `run_adaptive_combination.py` 同一形式。
四个分量 |R|、C_pool、D、1-C_cost 直接按自然尺度使用，不做 min-max 归一化。
参与比较的每个候选（含 parent 自身）都在同一个 P_{-f} 上计算。
仅当子代 S 严格高于 parent S 时替换，且在所有打分更高者中取 S 最高者。

权重为固定 (α,β,γ,λ)，按最近一次完整 run 的实测分布推出，使各项"中等表现"贡献约为 1.0。
默认 25/500/3/0；不再要求和为 1。
子代不得与池内公式完全重复；complexity 为 AST 本体节点数，忽略一层最外侧方向包装。
父因子自身不参与 P，允许等价简化。

没有满足任一情形的子代就保留 parent；替换原槽位，不改动其他成员。
池大小始终不变，一轮最多替换 parent 数量个成员。

## 日志与产物

每个 parent 写 replacement 事件及 lineage，包含候选的 reward、complexity、
max_correlation、eligible、description，以及最终 replaced/retained 结果。
入池子代另保存 child_description，保留时为 null；描述只用于解释和追溯，不参与数值筛选。
全无有效子代时仍记录 retained，继续训练，不因空轮报错。
每个 parent 处理完 `keep(pool.exprs)` 释放未入池信号。
每轮保存 pool、memory、lineage；路径 ID 规则见 [机制记忆](04_Mechanism_Memory.md)。

最终 `pool_<rounds>.json` 即最终池，split=train。
`train_cf.py` 释放训练资源后调用原 `run_adaptive_combination.py` 复跑最终回测；
Validation/Test 切分由原脚本内部按 `--train_end_year 2021` 处理，AlphaCF 不再单独维护 valid。
`--test-only` 仍可用保存池复跑最终回测。
