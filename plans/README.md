# 实现索引

以 idea.md 的八步为准；pool、trainer、prompt、AST 小工具和入口各司其职。
每个文档详细列出：所用类/函数、关键参数默认值、reward 计算公式、伪代码流程。

- [初始化](01_Initialize.md)：`train_cf.load_data`；`AlphaCFTrainer.initialize`；`AlphaCFPool.evaluate/select`。  
  候选种子 150 = Alpha158 30 + AlphaSAGE 50 + AlphaPROBE 50 + AlphaGen 20；方向学习只在 Train；`R(f)` = signed RankICIR。
- [反事实诊断](02_Counterfactual_Diagnosis.md)：`AlphaCFTrainer.select_parents/diagnose`；`expression.ablate/walk`；`PROMPT_DIAGNOSIS`。  
  `parents=20`（10 探索 + 10 利用）；`mechanisms=3`；`delta_cf=R(T(f,m))-R(f)`。
- [机制贡献](03_Mechanism_Credit.md)：`AlphaCFTrainer.diagnose`；`AlphaCFPool.utility/score_signals`。  
  `pool_credit=U(P)-U(P\{f}∪{T(f,m)})`；阈值 ±1e-5 决定 action。
- [机制记忆](04_Mechanism_Memory.md)：`AlphaCFTrainer.memory/diagnose/evolve/save`。  
  全量 `self.memory` 直接传 LLM；仅 `delta_cf != None` 的进入；不做向量库/RAG。
- [宏观演化](05_Macro_Evolution.md)：`AlphaCFTrainer.evolve/train`；`PROMPT_EVOLUTION`；`utils.llm.OpenAIModel`。  
  `offspring=5/parent`；donor 选双重证据最大者；mutation/replacement/crossover 三类。
- [子代细化](06_Offspring_Refinement.md)：`AlphaCFTrainer.refine`；`expression.parameter_variants`。  
  `refine_top_k=5`；至多 2 个窗口同时枚举；笛卡尔积；只保留最优 + 当前池。
- [工作池更新](07_Pool_Update.md)：`AlphaCFPool.select/update/keep`；`AlphaCFTrainer.train/save`。  
  候选集严格 = 当前池 + 本轮 offspring；`S=α·R+β·ΔU+γ·D-cost·λ`；贪心 60 个。
- [最终选择与测试](08_Final_Selection.md)：`train_cf.train/test`；原 `run_adaptive_combination.py`。  
  Valid 段 `Select_30`（同一打分函数）；`subprocess.run(check=True)` 调原回测脚本；`ret_s.npy` 同目录落盘。

## 默认参数汇总

| 参数 | 默认 | 说明 |
|---|---:|---|
| `--rounds` | 10 | 演化轮数 |
| `--parents` | 20 | 每轮 parent 数（10 探索 + 10 利用） |
| `--mechanisms` | 3 | 每个 parent 的最大机制数 |
| `--offspring` | 5 | 每个 parent 的 offspring 数 |
| `--refine-top-k` | 5 | 进入窗口网格搜索的子代数 |
| `--pool-capacity` | 60 | 工作池容量 |
| `--final-size` | 30 | 最终因子数 |
| `--horizon` | 20 | h 日前向收益 |
| `--max-nodes` | 60 | AST 最大节点数 |
| `--max-depth` | 10 | AST 最大深度 |
| `--max-backtrack` | 100 | 最大历史回看 |
| `--chunk-size` | 64 | 评价时分块天数 |
| `--alpha/beta/gamma/cost-weight` | 1.0 / 1.0 / 0.2 / 0.1 | 池选择四元组权重 |
| `--temperature` | 0.5 | LLM 温度 |
| `--windows` | [5, 10, 20, 40, 60] | 窗口细化网格 |
| `--n-factors` | 10 | 回测脚本中每天最多选几个 |

## 关键 reward 公式

| 量 | 公式 | 衡量 | 符号 |
|---|---|---|---|
| `R(f)` | `Mean_t Spearman(f_t, ret_{t→t+h}) / (Std_t + ε)` | 单因子 | 有符号 |
| `U(P)` | `RankICIR( (1/K) Σ_i f_i^rank )` | 池组合 | 有符号 |
| `delta_cf` | `R(T(f,m)) - R(f)` | 机制对个体的边际 | 负值=贡献 |
| `pool_credit` | `U(P) - U(P\{f} ∪ {T(f,m)})` | 机制对池的边际 | 正值=贡献 |
| `C_cost` | `(1 - mean_t Spearman(f_t, f_{t-1})) / 2` | turnover proxy | [0, 1] |
| `D(f,P)` | `1 - max_{g∈P} mean_t |Spearman(f_t, g_t)|` | 多样性 | [0, 1] |
| `S(f\|P)` | `α·R + β·ΔU + γ·D - λ·cost` | 池打分（min-max 归一化后） | 选 argmax |

## 复用与不做的部分

- 复用：`alphagen` 算子库、`alphagen_qlib.stock_data.StockData`、`alphagen.utils.correlation` 中的 `batch_pearsonr`、原 `run_adaptive_combination.py` 完整回测流程
- 不重写：线性回归、VIF、SVD/QR 列降秩、batch Sharpe/MDD 等全部由 `run_adaptive_combination.py` 负责
- 不维护：向量库 memory retrieval、多层结构契约过滤、Pareto 多目标、对话修复循环
