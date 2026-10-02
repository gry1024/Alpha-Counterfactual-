# AlphaCF

按 [idea.md](idea.md) 实现反事实机制演化，沿用 `alpha_knowledge` 的 pool / trainer / 入口组织。表达式、解析器、Qlib 数据和基础统计复用仓库现有实现。

```text
150 个种子 → Train 选 50 个
每轮：选 10 个 parent → 自由反事实编辑 → 三项机制证据 → 机制记忆
     → 每个 parent 自主生成 0–5 个子代 → 合格最优子代原位替换 parent
最后：最后一轮 pool → 直接调用原 run_adaptive_combination.py
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
| `--parents` | 10 | 每轮 5 个 top-k + 5 个低访问，互不重复 |
| `--correlation-threshold` | 0.9 | 子代与其余池成员最大日均绝对 Spearman 相关性上限 |
| `--pool-capacity` | 50 | 工作池（=最终池） |
| `--alpha / --beta / --gamma / --cost-weight` | 1 / 1 / 0.2 / 0.1 | 仅初始化使用的质量、组合边际、多样性、换手 proxy 权重 |
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

种子方向仅在 Train 上确定：负 RankIC 的公式显式写成 `Sub(0.0,expr)`，然后统一选择。后续 offspring、消融、Validation、Test 均按原有符号打分，不取绝对 R，也不自动翻转。种子库是待检验的候选结构，不代表每个公式都有效。

| 分区 | 日期 | 用途 |
|---|---|---|
| Train | 2011–2021 | 搜索、消融、池更新；最后一轮即为最终池 |
| Validation | 2022 | 由原回测脚本使用，AlphaCF 不再触碰 |
| Test | 2023–2026.04 | 由原回测脚本使用 |

Train 段剔除末尾 horizon 个交易日的跨段标签，允许读取此前历史，禁止未来引用。最终回测直接使用原脚本的数据加载、标签与组合口径。默认回看改为 100 日，与原脚本的 StockData 默认值一致。

## 指标与演化

- `R = mean(RankIC)`，有符号、不年化。
- 每个因子用平均并列秩映射到 [-0.5,0.5]，缺失取中性 0，固定等权组合再算 `U = RankIC`。组合用双精度累加，避免增量替换改变并列排名。
- `delta_cf = |R(counterfactual)| - |R(parent)|`；`pool_credit = C_pool = U(P\{parent} ∪ {counterfactual}) - U(pool)`（越大表示组合效用提升越多）；`signal_distance = 1 - mean_daily_Spearman(parent,counterfactual)`，不取绝对值，接近 0 提示排序等价。
- `S = alpha*|R| + beta*C_pool + gamma*(1-max_correlation) + cost_weight*(1-turnover)`，四项 min-max 归一化到 [0,1] 后加权，`alpha+beta+gamma+cost_weight=1`（默认 0.4/0.3/0.2/0.1）。初始化贪心选择与父因子替换共用同一打分。
- 子代替换父因子分两种情形：signal_distance ≈ 0（排序等价）且 AST 节点数更少，直接替换；否则按 `S` 打分，得分严格高于 parent 的子代中取最高者替换。
- LLM 根据因子复杂度自主提 0–5 个有意义的反事实编辑，没有机制数量参数。可删冗余算子、改窗口或经济重写，包括根路径整式替换；只记录实测证据，不给预设处置建议。
- LLM 根据 mechanism 数量、规模和 factor 复杂度，为每个 parent 自主生成 0–5 个完整公式；不设固定数量参数，允许空列表并保留父因子。程序解析、检查、评价和精确去重。historical memory 只包含该 parent 自身的过往记录，不跨因子；不限定操作类型或保留结构。
- 候选与 parent 的 `C_pool`、`D` 均在 `P\{parent}` 上计算，不包含被替换成员自身；最多一个子代替换该 parent，没有合适子代就保留。单层最外部符号包装不计复杂度。
- 同轮先诊断再逐 parent 生成和替换，池大小不变；每个成员有唯一 factor ID，演化路径记入 lineage.json。

算子本身使用 `alphagen` 原实现；组合评分的 rank 使用平均并列秩。`Greater/Less` 规范化为同义的 `GetGreater/GetLess`，科学计数法转换后交给现有 parser。

## LLM 调用与提示

沿用 `src/utils/prompt.py` 的角色提示、特征/算子定义、任务模板和结构化输出方式。角色直接复用 `PROMPT_HEAD`；算子表按当前 `alphagen` 实现校正，避免复用范例中的 TsRatio、Greater/Less、TsMad 等不一致描述。

诊断输出 `mechanisms`，每项为 `path/description/replacement/reason`；给出有意义的反事实编辑，由程序计算三项证据。演化返回 `{"offspring":[{"expression":"完整公式","description":"简短的证据与修改说明"}]}`，数量由 LLM 自定为 0–5 个，允许 `{"offspring":[]}`。每个子代必须附 1–2 句描述，说明依据哪些 mechanism 证据、做了什么修改或探索；无实测证据时明确说明是探索假设，不编造数值。不规定操作类型。提示中提供当前池及本轮已生成公式、父因子机制证据和实际替换标准。

网络调用统一复用 `OpenAIModel.chat_generate`，不另造会话重试框架。日志保存格式化后的完整输入、原始响应与 finish_reason。

## 最终回测

最后一轮（`--rounds`）结束时 `pool_<rounds>.json` 即为最终产物，通过 `subprocess.run(..., check=True)` 直接执行仓库根目录的 `run_adaptive_combination.py`。不再维护 AlphaCF 自己的 evaluator、OLS 或回测指标副本。

向原脚本传入 `--expressions_file`、`--instruments`、`--train_end_year 2021`、`--label_days`、`--cuda`、`--seed`、`--n_factors` 和 `--chunk_size`。使用同一 Python 解释器，并传递 src 的 PYTHONPATH。原脚本打印 Validation / Test 指标，保存 `ret_s.npy`；其默认数据目录与算法均保持原样。`--qlib-path` 配置 AlphaCF 的搜索数据，最终回测的数据目录由原脚本决定。

```bash
python train_cf.py --test-only data/cf_logs/<run>/pool_<rounds>.json --cuda 0
```

跳过 Train，直接对已有 `pool_<rounds>.json` 复跑回测，便于多次运行验证测试稳定性。

## 日志与消融

每次运行单独保存到 `data/cf_logs/<时间戳>_<instrument>_<seed>/`：

- `args.json`：参数。
- `round_<轮次>.jsonl`：逐条追加模型输入输出、机制测量、候选评分、每个子代的简短描述及池更新。
- `pool_<轮次>.json`：每轮池；最后一轮即最终池。
- `memory.json`：所有实测机制证据。
- `lineage.json`：初始成员与跨轮 parent→child 路径，含保留结果、候选判定及 description、入池子代的 child_description；池快照带 factor_ids 和 visits。
- `ret_s.npy`：由原回测脚本保存；Validation / Test 指标直接打印。

日志不含密钥。模型返回无法解析的 JSON 或子代超过 5 个、字段类型不符或缺少非空表达式/描述会报错；没有可用或合格子代时保留父因子并记录结果，继续训练。

消融开关：`--no-cf-evidence`（保留语义分解）、`--no-pool-credit`、`--random-crossover`（随机可选 donor）、`--no-memory`、`--no-pool-selection`（仅初始化按 R）。

开发说明见 [plans/](plans/README.md)。数值测试：`python -m unittest discover -s tests -v`。GP/AlphaGen/AlphaSAGE 对照仍使用原有入口，比较时需统一数据协议。上游说明见 [Quickstart.md](Quickstart.md)，许可见 [LICENSE](LICENSE)。
