# 08 — 最终选择与测试

对应代码：`train_cf.train/test`；现有 `run_adaptive_combination.py`。

Train 完成全部轮次后保存 search_pool.json。Validation 只从最终工作池按同一选择公式选 30 个，保存 final.json；不生成、不细化、不翻转符号。

冻结产物直接传给原 run_adaptive_combination.py，以当前 Python、仓库根目录和 src 的 PYTHONPATH 启动子进程。传入股票池、标签跨度、设备、随机种子及组合参数；复用原脚本全部加载、筛选、回归和指标逻辑，不改写脚本。原脚本打印 Validation / Test 结果并保存 ret_s.npy。

默认搜索回看为 100 日，匹配原脚本的 StockData 默认值。搜索/验证使用 --qlib-path；最终回测使用原脚本定义的数据目录。移除 AlphaCF 独立 evaluate.py 和重复的 Test/OLS 实现。

--test-only 读取 final.json 并调用原脚本；--finalize-only 读取 Train 池，先做 Validation 再调用原脚本。已完成快照保留，不自动将中断搜索当作完整训练。
