# 03 — 机制证据

实现：`AlphaCFTrainer.diagnose`；
`AlphaCFPool.evaluate/correlation/score_signals/utility`。

设 parent=f，反事实=f'，轮初池=P：

| 字段 | 公式 | 解释 |
|---|---|---|
| delta_cf | \|R(f')\| - \|R(f)\| | 负值表示本次编辑损害个体预测力度 |
| pool_credit | C_pool(f'\|f,P) = U(P\{f}∪{f'}) - U(P) | 正值表示本次编辑提升组合 |
| signal_distance | 1 - mean_t Spearman(f_t,f'_t) | 接近 0 提示排序等价 |

R 是有符号日均横截面 RankIC，U 是组合的有符号日均 RankIC；个体质量以 |R| 衡量，**不区分正负预测方向**（与 alpha_knowledge 保持一致）。
signal_distance 使用有符号相关性，不取绝对值，范围 [0,2]。
共同有效股票至少两个、双方有截面变化的日期参与日均；
全不可计算时不写入成功证据。

`Mul(1.0,f) → f` 可得到 distance≈0、delta_cf≈0、pool_credit≈0，
给 LLM 提供去冗余证据。距离接近 0 不证明 AST 相同：
正比例缩放、严格单调变换也可能有相同排序。

pool credit 使用 in-sample OLS 回归组合 utility 计算：直接调用 `pool.utility(peers + [f'])`（与 train 阶段 `U(P)` 的计算方式一致），缺失曝光取中性 0，OLS 在 `T*S` 个样本上单次最小二乘拟合，与 `run_adaptive_combination.py` 同一形式。

不根据两项 credit 的符号生成处置分类，LLM 自主理解收益、冗余和组合之间的权衡。
`--no-pool-credit` 仅使 pool_credit=null，仍测量其余两项；
`--no-cf-evidence` 使全部证据为 null。
null 表示未测量，不能当成零贡献。
