# 01 — 初始化

对应代码：`train_cf.py::load_data/train`、`AlphaCFTrainer.initialize`、`AlphaCFPool.initialize/evaluate/keep`、`alpha_cf.expression.parse`。

## 1.1 离线构建与直接加载

`start_pool.json` 直接保存最终 50 个精华因子，混合 Alpha158、AlphaSAGE、AlphaPROBE、AlphaGen，每种来源均减少数量。离线方案与实测结果见 [初始池比较](Start_Pool_Comparison.md)。

`plot_profit_curve_start_pool.py`（仓库当前仅余 `__pycache__/*.pyc`，`.py` 源文件已不在工作区，需自行恢复或从历史 commit 取回）比较多种来源配额和去相关强度，先满足经济结构覆盖、同构去重及相关性约束，再按 2022 年验证收益曲线择优。它只回测初始池，不调用 trainer、LLM 或整轮演化，不读取 Test。

训练初始化不再筛选：

```python
if len(seeds) != args.pool_capacity:
    raise ValueError("Initial pool size mismatch")
for seed in seeds:
    expr = self.evaluate(seed)  # parse + Train evaluate
    if expr is None:
        raise ValueError("Invalid initial factor")
    candidates.append(expr)
pool.initialize(candidates)   # 校验数量/规范公式唯一性；保持全部成员与原顺序
```

默认 `pool_capacity=50`，自定义池也须与该容量完全匹配。种子不可用、规范化后重复或数量错误均报错，禁止静默缩小工作池。初始化不翻转符号；外层 `Sub(0.0,f)` 是否添加仍由 LLM 在演化时决定。离线未选因子不进入搜索。`AlphaCFPool.select` 及 `--no-pool-selection` 已删除，父因子替换评分保留。

## 1.2 Train 评价

`AlphaCFPool.evaluate` 分块调用现有 `expr.evaluate`，用平均并列秩映射到 `rank/(N-1)-0.5`；缓存每个因子的 signal、signed RankIC、cost。

- `R(f)=mean_daily_Spearman(signal, horizon 日前向收益)`，原始符号保留，个体质量用 `|R|`。分母包含全部标签有效日期，因子无变化或无法计算的 IC 计 0；标签无效日期不计入。
- `C_cost=(1-mean_daily_Spearman(signal_t,signal_{t-1}))/2`，仅为排序换手 proxy。
- 无可用截面变化时抛错；初始化不得跳过。
- 缺失组合敞口取中性 0；全 Train signed OLS 的 `U` 与快照权重保留。pool_credit 共用完整 Train 的效用，不做内部时间切分。
- `keep(pool.exprs)` 仅保留当前池缓存。保存与终端明细沿用原 factor ID、指标、权重和 lineage 协议。

## 1.3 数据与合法性

`train_cf.load_data` 使用真实 Qlib 日历，Train 为 2015–2021，前置历史默认 100 日。`max_future_days=0`，标签由 `close[h:]/close[:-h]-1` 构造，剔除末尾 h 个跨段标签及非有限/非正价格。默认 h=20，chunk_size=64。

`alpha_cf.expression.parse` 复用 alphagen parser 和算子；规范化空白、Greater/Less 别名与科学计数法。FormulaBuilder 允许不同嵌套层的常数连续入栈，不修改 RL 生成器约束。

表达式必须包含 feature，节点数 ≤60、深度 <10、累计 lookback ≤100；Ref/TsDelta 窗口分别 ≥0/1，禁止未来引用。一层最外部 `Sub(0.0,f)` 不计本体节点和深度，回看约束仍检查完整表达式。

## 1.4 入口

`trainer.train(seeds)` 首先直接初始化，之后才运行指定演化轮数。最终池保存为 `pool_<rounds>.json`，释放 Train 资源后仍调用原 `run_adaptive_combination.py`；Validation 由 `run_adaptive_combination.py` 按 `--train_end_year 2021` 内部切出 2022，Test 统一为 2023-05-01–2026-04-30，使用原有 cn_data_rolling / us_data_qlib_latest 与 30 日未来缓冲（`max_future_days=0`，horizon=20）；z-score 自适应组合口径保持原协议。

只比较初始池时运行 `plot_profit_curve_start_pool.py`（恢复后），不要用 `train_cf.py --rounds 0`：后者仍会触发最终 Test 回测。
