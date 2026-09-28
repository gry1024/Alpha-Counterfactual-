# AlphaCF

按 [idea.md](idea.md) 实现反事实机制演化，沿用 `alpha_knowledge` 的 pool / trainer / 入口组织。表达式、解析器、Qlib 数据和基础统计复用仓库现有实现。

```text
150 个种子 → Train 选 60 个
每轮：选择 parent → 机制消融 → 双重 credit → 机制记忆
     → mutation / replacement / crossover → 窗口枚举 → 重选 60 个
最后：Validation 选 30 个 → 冻结公式 → 调用原 run_adaptive_combination.py
```

## 运行

```bash
cd /home/groy/cf
source .venv/bin/activate
python train_cf.py --instrument csi300 --cuda 0 --rounds 10
```

`.env` 使用已有 `OPENAI_API_KEY`、`OPENAI_BASE_URL`、`OPENAI_MODEL_NAME=MiniMax-M3`。数据目录读取 `QLIB_PATH_CN` / `QLIB_PATH_SP500`，也可传 `--qlib-path`。不设置输出 token 上限；调用复用 `utils.llm.OpenAIModel.chat_generate` 的网络重试；响应先移除服务返回的 `<think>` 段，再解析 JSON 或 Markdown 代码块，仍失败则报错。

| 参数 | 默认 | 含义 |
|---|---:|---|
| `--rounds` | 10 | 演化轮数；0 只初始化并做最终评价 |
| `--parents` | 20 | 每轮父本数，一半优先少访问，一半优先高 R |
| `--mechanisms` | 3 | 单个父本最多诊断的机制数 |
| `--offspring` | 5 | **每个父本**请求的子代数，默认约 100 个/轮 |
| `--refine-top-k` | 5 | 做参数枚举的优质子代数 |
| `--windows` | 5 10 20 40 60 | 最多两个不同窗口值的完整网格，包含原值 |
| `--pool-capacity / --final-size` | 60 / 30 | 工作池 / 最终池 |
| `--alpha / --beta / --gamma / --cost-weight` | 1 / 1 / 0.2 / 0.1 | 质量、组合边际、多样性、换手 proxy 权重 |
| `--horizon` | 20 | 收益标签跨度 |
| `--max-backtrack / --max-nodes / --max-depth` | 100 / 60 / 10 | 表达式合法性限制 |
| `--chunk-size` | 64 | 按日期分块执行表达式 |
| `--n-factors` | 10 | 最终 adaptive OLS 最多使用的因子数 |
| `--start-pool` | start_pool.json | 外部种子公式文件 |

## 种子与数据

[start_pool.json](start_pool.json) 的 `exprs` 直接保存 150 条表达式，按 Alpha158（30）、AlphaSAGE（50）、AlphaPROBE（50）、AlphaGen（20）的顺序排列。后三组原样保留来源池的表达式及顺序，不携带权重或评分。

Alpha158 按结构类别和时间尺度精选，覆盖 K 线形态、相对价格、动量、均线、波动、区间位置、价量相关和成交量变化，窗口涵盖 5、10、20、30、60 日；未使用收益排名或相关性筛选。依据本地 Qlib `Alpha158DL` 转写为现有算子语法，保留原始比率方向与 1e-12 分母稳定项。按顺序选取：KMID、KLEN、KMID2、KUP2、KLOW2、KSFT2、VWAP0、ROC5、ROC20、ROC60、MA10、MA30、STD5、STD20、STD60、MAX20、MIN20、VSUMP10、VSUMP60、SUMP10、SUMP60、RSV5、RSV30、CORR30、CORR60、CORD20、VMA5、VMA60、VSTD20、WVMA20。

来源文件：
- AlphaSAGE：`data/gfn_logs/pool_50/gfn_gnn_csi300_50_2-0.01-1.0-1.0-1.0-0.3-linear-0.0/pool_9999.json`
- AlphaPROBE：`data/knowledge_logs/pool_50/kg_dag_and_bayesian_icir_and_mutl_new_no_decay_MiniMax-M3_5_csi300_0.5_7_50_0.9_50_20_0.006_True_True_False_True_0.7_0.1_0.05/pool_20.json`
- AlphaGen：`data/ppo_logs/pool_20/ppo_csi300_20_0-20260905132532/ppo_csi300_20_0_20260905132532/200704_steps_pool.json`

种子方向仅在 Train 上确定：负 RankICIR 的公式显式写成 `Sub(0.0,expr)`，然后统一选择。后续 offspring、消融、Validation、Test 均按原有符号打分，不取绝对 R，也不自动翻转。种子库是待检验的候选结构，不代表每个公式都有效。

| 分区 | 日期 | 用途 |
|---|---|---|
| Train | 2011–2021 | 搜索、消融、参数枚举、池更新 |
| Validation | 2022 | 最终 60→30，不能生成新公式 |
| Test | 2023–2026.04 | 冻结后的最终评价 |

Train / Validation 各自剔除末尾 horizon 个交易日的跨段标签，允许读取此前历史，禁止未来引用。最终回测直接使用原脚本的数据加载、标签与组合口径。默认回看改为 100 日，与原脚本的 StockData 默认值一致。

## 指标与演化

- `R = mean(RankIC) / (std(RankIC) + 1e-8)`，有符号、总体标准差、不年化。
- 每个因子用平均并列秩映射到 [-0.5,0.5]，缺失取中性 0，固定等权组合再算 `U = RankICIR`。组合用双精度累加，避免增量替换改变并列排名。
- `delta_cf = R(ablation) - R(parent)`；`pool_credit = U(pool) - U(replace(parent,ablation))`。
- `S = alpha*R + beta*marginal_U + gamma*(1-max_correlation) - cost_weight*turnover`。四项在候选集中 min-max 归一化，贪心选择。
- AST 消融只改指定路径：remove 保留其已有后代，neutralize 不引入新输入。机制建议和实测证据分开，只有测量成功的干预进入 memory。Action 只是提示，双重原始数值始终提供。
- 默认每个 parent 都发起一次演化请求，模型输出完整公式；程序做解析、合法性检查、数值评价和精确公式去重。不再用多层结构过滤决定什么算有效演化。
- 每个父本配一个其他 donor，默认按机制证据选；全部历史 memory 传入请求。没有复杂 trace、coverage 门槛或错误分类框架。

算子本身使用 `alphagen` 原实现；组合评分的 rank 使用平均并列秩。`Greater/Less` 规范化为同义的 `GetGreater/GetLess`，科学计数法转换后交给现有 parser。

## LLM 调用与提示

沿用 `src/utils/prompt.py` 的角色提示、特征/算子定义、任务模板和结构化输出方式。角色直接复用 `PROMPT_HEAD`；算子表按当前 `alphagen` 实现校正，避免复用范例中的 TsRatio、Greater/Less、TsMad 等不一致描述。

诊断提示包含金融语义、AST 路径约定、Remove / Neutralize 对照、完整可执行示例和自检要求。演化提示区分 mutation / replacement / crossover，要求解释实际证据、保留与改写的机制，并看到当前池及本轮已生成公式以避免重复。返回 `expressions`、`operations`、`explanations` 三个等长数组；只输出自检后的公式，不重复输出草稿和修正版。

网络调用统一复用 `OpenAIModel.chat_generate`，不另造会话重试框架。日志保存格式化后的完整输入、原始响应与 finish_reason。

## 最终回测

入口保存 `final.json` 后，通过 `subprocess.run(..., check=True)` 直接执行仓库根目录的 `run_adaptive_combination.py`。不再维护 AlphaCF 自己的 evaluator、OLS 或回测指标副本。

向原脚本传入 `--expressions_file`、`--instruments`、`--train_end_year 2021`、`--label_days`、`--cuda`、`--seed`、`--n_factors` 和 `--chunk_size`。使用同一 Python 解释器，并传递 src 的 PYTHONPATH。原脚本打印 Validation / Test 指标，保存 `ret_s.npy`；其默认数据目录与算法均保持原样。`--qlib-path` 配置 AlphaCF 的搜索/验证数据，最终回测的数据目录由原脚本决定。

```bash
python train_cf.py --test-only data/cf_logs/<run>/final.json --cuda 0
python train_cf.py --finalize-only data/cf_logs/<run>/search_pool.json --cuda 0
```

第二条只从 Train 快照执行 Validation 和 Test，不恢复演化；也可读取同目录带 `args.json` 的 `pool_<轮次>.json`。保存配置优先，设备使用当前命令。

## 日志与消融

每次运行单独保存到 `data/cf_logs/<时间戳>_<instrument>_<seed>/`：

- `args.json`：参数。
- `round_<轮次>.jsonl`：逐条追加模型输入输出、机制测量、候选评分、参数枚举及池更新。
- `pool_<轮次>.json` / `search_pool.json`：工作池及最终 Train 池。
- `memory.json`：所有实测机制证据。
- `final.json`：Validation 冻结公式、等权及指标。
- `ret_s.npy`：由原回测脚本保存；Validation / Test 指标直接打印。

日志不含密钥。模型返回无法解析的 JSON、输出数组不匹配或整轮没有可用子代会报错，已完成的池快照保留，不把空转称作完成。

消融开关：`--no-cf-evidence`（保留语义分解）、`--no-pool-credit`、`--random-crossover`（随机 donor）、`--no-memory`、`--no-pool-selection`（仅按 R）、`--no-refinement`。

开发说明见 [plans/](plans/README.md)。数值测试：`python -m unittest discover -s tests -v`。GP/AlphaGen/AlphaSAGE 对照仍使用原有入口，比较时需统一数据协议。上游说明见 [Quickstart.md](Quickstart.md)，许可见 [LICENSE](LICENSE)。
