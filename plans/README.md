# 开发计划与实现索引

以 idea.md 的八个 Step 组织。代码仍由 AlphaCFPool、AlphaCFTrainer、表达式辅助函数、prompt 和入口构成，不按文档数量拆 Python。

| Step | 文档 | 对应代码 |
|---|---|---|
| 1 初始化 | [01_Initialize](01_Initialize.md) | train_cf.py；AlphaCFPool.evaluate/select |
| 2 反事实诊断 | [02_Counterfactual_Diagnosis](02_Counterfactual_Diagnosis.md) | select_parents/diagnose；expression.ablate；DIAGNOSE |
| 3 机制归因 | [03_Mechanism_Credit](03_Mechanism_Credit.md) | diagnose；AlphaCFPool.utility |
| 4 机制记忆 | [04_Mechanism_Memory](04_Mechanism_Memory.md) | retrieve_memory/visible_evidence/save |
| 5 宏观进化 | [05_Macro_Evolution](05_Macro_Evolution.md) | evolve；EVOLVE |
| 6 参数细化 | [06_Offspring_Refinement](06_Offspring_Refinement.md) | refine；parameter_specs/parameter_variants |
| 7 池更新 | [07_Pool_Update](07_Pool_Update.md) | AlphaCFPool.select/update；train/phase |
| 8 最终选择 | [08_Final_Selection](08_Final_Selection.md) | train_cf.py 的 Validation 与 test-only |

## 已落实的修改

| 编号 | 实现 |
|---|---|
| A01 | 机制名称自由；数量上限为预算；保留无法消融的机制提议 |
| A02 | LLM 提出局部基线；Remove 保留后代输入，Neutralize 支持上下文基线；记录干预覆盖率 |
| A03 | 完整双重证据交给 LLM；Action 不再限制操作资格 |
| A04 | 轮流从个体质量、已有组合边际记录、诊断新颖性选择父本 |
| A05 | LLM 输出完整 child，自选操作数量与机制组合；无固定 pair 或操作配额 |
| A06 | 结构相关、正向、负向、冲突、近期五路检索，默认最多 24 条历史 |
| A07 | 最多两个窗口/数值位置；合法离散网格与试探预算 |
| A08 | 有效日 RankICIR；显式样本门槛；记录干预前后覆盖差异 |
| A09 | 无反事实证据、无 pool credit、逐子代随机交叉等消融口径明确 |
| A10 | 复用当前池因子对相关性；记录分阶段执行次数与耗时 |

## 共享约定

- 复用 alphagen 的 Expression、ExpressionParser、StockData、batch_pearsonr；不改上游基础设施。
- argparse 管理参数；机制和日志使用 dict/list，规范化公式字符串作为去重键。
- Train 完成 Step 1–7；Validation 仅完成最后 60→30；Test 只评价冻结结果。
- 初始 199 个混合来源公式明文保存在 train_cf.py，初始落选因子不重新进入搜索。
- LLM 决定结构假设，程序决定数值指标、参数选优和池成员。
- 按用户要求，本轮未运行测试、语法编译、训练或模型请求；文档描述实现，不代表运行验证。
