# Alpha158 等权主基线

全部 158 个按日截面标准化特征的绝对权重均为 1/158。方向由 Train 特征与标准化并截断的 20 日目标的协方差确定，随后冻结。

仓库根目录 WSL 环境运行（将 csi300 换为 csi500 / csi1000）：

~~~bash
PYTHONPATH=src:. OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 .venv/bin/python src/alpha158/run.py --instruments csi300
PYTHONPATH=src:. .venv/bin/python plot_profit_curve.py
~~~

可用 --feature-cache 指定同市场、同日期与股票顺序的归一化缓存。输出默认位于 data/alpha158_logs/<市场>_equal_weight/，包含特征缓存（若新建）、权重、配置、Validation / Test 预测、指标、收益和日期。三个市场均已完成等权评价；核验见各目录 checks.json。

Train 为 2015–2021，剔除截至训练截止日尚未成熟的 20 日标签；Validation 为 2022；Test 为 2023-05-01 至 2026-04-30。IC / RIC 使用 20 日收益；每日预测前 20% 股票的组合使用次日收益，零成本。指标沿用共享回测定义。

按作者 2026-10-15 指示，result.md、三个市场主曲线及论文主表统一采用此等权结果。协议是在查看 Test 对照后指定，论文披露这一变更，不声称预注册或由验证集选中。原始 OLS、Ridge、树模型产物保留用于审计，旧代码归档于 data/alpha158_logs/protocol_history/before_equal_only/。
