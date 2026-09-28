# 03 — 机制贡献

对应代码：`AlphaCFTrainer.diagnose；AlphaCFPool.utility/score_signals`。

个体证据 delta_cf=R(ablation)-R(parent)。组合证据 pool_credit=U(P)-U(P-parent+ablation)。U 为固定等权 rank 信号的有符号 RankICIR。

复用原 pool signal 总和做单个替换，双精度累加确保与直接重建一致。Action 在个体改善且组合无反向证据时提示 Preserve，反向一致时 Replace；近零、冲突或未测量为 Explore。模型始终看到原始数值，标签不限制操作。
