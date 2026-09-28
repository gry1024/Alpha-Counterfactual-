# 01 — 初始化

对应代码：`train_cf.load_data；AlphaCFTrainer.initialize；AlphaCFPool.evaluate/select`。

从 start_pool.json 读取 40 类 × 5 窗口的 200 个量价候选。只在 Train 拟合种子符号，将负 R 显式取反后按统一 selection 选 60 个。落选种子不保留为候选池。

复用 StockData、ExpressionParser、Expression.evaluate 和 batch_pearsonr。分区内构造 h 日收益，剔除尾部标签；只保留 AST 大小、深度、历史和可计算性检查。因子 signal 缓存为 rank 值，组合双精度累加。
