# 实现索引

idea、plans 与 `src/alpha_cf/` 使用同一协议：

1. [初始化](01_Initialize.md)：150 个种子仅初始化时贪心选 50 个。
2. [反事实诊断](02_Counterfactual_Diagnosis.md)：默认 10 个 parent，5 top-k + 5 低访问；LLM 自定 0–5 个机制。
3. [机制证据](03_Mechanism_Credit.md)：delta_cf、pool_credit、signal_distance 三项实测证据。
4. [机制记忆](04_Mechanism_Memory.md)：完整历史证据与 factor ID 演化路径。
5. [直接生成子代](05_Macro_Evolution.md)：LLM 按 mechanism 量级与 factor 复杂度自主生成每 parent 0–5 个子代，每项包含完整公式及简短的证据与修改描述，无操作类型约束。
6. [父因子替换](07_Pool_Update.md)：等价去冗余（signal_distance≈0 且节点更少直接替换）或组合得分 S 更高，最多一个子代替换其 parent。

## 默认参数

| 参数 | 默认 | 说明 |
|---|---:|---|
| --rounds | 10 | 演化轮数 |
| --parents | 10 | 默认 5 top-k + 5 低访问，互不重复 |
| --correlation-threshold | 0.9 | 保留参数；替换判定已由 S 的多样性项取代硬门槛 |
| --pool-capacity | 50 | 工作池容量，演化中保持不变 |
| --horizon | 20 | 前向收益跨度 |
| --max-nodes / --max-depth / --max-backtrack | 60 / 10 / 100 | 表达式执行边界 |
| --chunk-size | 64 | 评价分块天数 |
| --alpha / --beta / --gamma / --cost-weight | 0.4 / 0.3 / 0.2 / 0.1 | S 得分权重，四项归一化到 [0,1] 且总和为 1 |
| --temperature | 0.5 | LLM 温度 |
| --n-factors | 10 | 原回测脚本每天最多选取因子数 |

机制数和子代数均没有命令行参数。机制数由 LLM 依复杂度选择，不超过 5；
子代数由 LLM 依 mechanism 数量、规模与 factor 复杂度选择 0–5 个，允许主动不生成。
R 与 U 均为 signed RankIC；signal_distance=1-mean_daily_Spearman(f,f')，
有符号相关性，不同于入池检查用的 mean_daily_abs_Spearman。

## 完成与验证

- [x] 移除诊断处置标签、编辑类型枚举及演化操作分类。
- [x] 新增信号距离，允许等价简化、窗口调整及整式经济重写。
- [x] 默认 parent 改为 10，每 parent 由 LLM 自主决定直接生成 0–5 个子代。
- [x] 按 parent 原位替换，失败保留，记录候选判定与演化路径。
- [x] 同步 idea、plans、README 和入口参数。
- 自动化验证见 `tests/test_cf_evolution.py`，使用合成张量与模拟 LLM；
  不依赖外部 API，不替代真实市场回测。

复用原表达式库、parser、Qlib、统计函数和最终 adaptive combination 回测。
新逻辑集中于 alpha_cf；不引入额外搜索框架、向量库、参数枚举或最终重选阶段。
