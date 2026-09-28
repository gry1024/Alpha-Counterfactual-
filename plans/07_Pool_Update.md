# 07 — 工作池更新

对应代码：`AlphaCFPool.select/update/keep；AlphaCFTrainer.train/save`。

候选集严格为当前 pool 与本轮 offspring。按 S=alpha*R+beta*marginal_U+gamma*D-cost_weight*cost 逐个贪心选择 60 个。四项在当前候选集 min-max 归一化，D=1-max mean(abs daily Spearman)，cost 为相邻日排名变化 proxy。

计算精确组合边际，候选信号分批处理，不先按阈值淘汰。选择完成后只缓存入池表达式，保存 pool_<step>.json 并打印子代数量、入池数量、U 和耗时。没有第二套历史候选回流机制。
