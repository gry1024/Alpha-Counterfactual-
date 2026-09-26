# AlphaCF：反事实机制演化因子挖掘

基于 [idea.md](idea.md)，复用 AlphaSAGE 的表达式执行、AST 解析和 Qlib 数据接口。LLM 提出机制、干预基线及完整子代表达式；真实数据计算证据并决定因子池成员。

## 算法流程

```text
199 个混合来源候选 → Train 选择 60 个工作因子
                              ↓
  选择父本 → 自由语义机制分解 → 局部 Remove / Neutralize
                              ↓
       单因子 delta_cf + pool_credit → 机制记忆
                              ↓
  检索历史证据 → LLM 生成 mutation / replacement / crossover
                              ↓
       默认参数评价 → 高质量候选的窗口 / 数值参数枚举
                              ↓
            当前池 ∪ 子代 → 重新选择 60 个
                              ↓
           Validation 从最终 60 个中选择 30 个
                              ↓
                     独立 Test 评价
```

机制没有类别白名单；每个父本默认最多 3 个机制是预算，可调整。父本轮流来自个体质量、最近 selection 的组合边际记录，以及未诊断/较久未诊断的因子。

LLM 为每个机制提出具体基线：Remove 保留机制的已有后代输入，Neutralize 使用中性元或可解释的上下文基线。程序仅替换目标子树，验证来源、输入、合法性和数值有效性；根部变换也可去包装。无法合理消融的机制保留为未测量记录。

```text
delta_cf    = R(干预后的因子) - R(原因子)
pool_credit = U(原池) - U(替换原因子后的池)
```

负 delta_cf、正 pool_credit 支持原结构在相应上下文中的价值。LLM 同时看到两项数值、基线和覆盖变化，选择保留、替换或重组；没有按符号划分的操作资格、固定 crossover 配对或三类操作配额。每个 crossover 可选择双方多个完整机制，外围表达式由 LLM 生成。

memory 保存真实干预证据；上下文按结构相关性、正向、负向、冲突和近期记录检索，默认最多 24 条历史，另提供本轮证据。历史 credit 不赋给新表达式。上下文基线的语义合理性仍需结合日志审阅，程序的 AST 检查不等于证明纯粹因果贡献。

## 评价口径

单因子 R 为有符号 RankICIR，不取绝对值、不年化。无效 IC 日记为缺失，只聚合有效日；另外设置值覆盖率、有效 IC 日期覆盖率和样本数门槛。

因子在当前可用股票上做平均并列 rank，归一化到 [-0.5,0.5]；组合缺失信号取中性值 0，再按固定池容量等权。U 为组合信号重新 rank 后的 RankICIR。selection 同时考虑 R、真实组合边际 U、多样性和相邻日排名变化的 cost proxy。

干预前后分别计算 R，并记录 coverage、有效日数和日期交集；没有改成共同样本上的另一套差分，因此覆盖变化仍应结合日志解释。

## 环境与数据

使用已有 WSL 环境：

```bash
cd /home/groy/cf
source .venv/bin/activate
```

依赖由 pyproject.toml / pdm.lock 管理，入口自动加入 src 导入路径。.env 配置：

```dotenv
OPENAI_API_KEY=你的密钥
OPENAI_BASE_URL=你的兼容接口地址
OPENAI_MODEL_NAME=MiniMax-M3
QLIB_PATH_CN=data/qlib_data/cn_data_rolling
QLIB_PATH_SP500=data/qlib_data/us_data_qlib_latest
```

| 分区 | 日期 | 用途 |
|---|---|---|
| Train | 2010-01-01～2021-12-31 | 初始化、诊断、进化、参数细化、池更新 |
| Validation | 2022-01-01～2022-12-31 | 最终 60→30 |
| Test | 2023-01-01～2026-04-30 | 独立冻结评价 |

默认标签为未来 20 个交易日收盘价收益；每段最后 20 日不打分，避免跨分区标签。特征允许读取分区前的历史。数据须覆盖指定区间和回看长度。

初始因子明文保存在 [train_cf.py](train_cf.py)：Alpha158 可表达子集 93 条、人工量价组合 15 条、GFN 50 条、PPO 20 条、知识搜索 21 条。只迁入公式，全部重新评价；初始落选公式不再参与搜索。

## 启动挖掘

```bash
python train_cf.py \
    --instrument csi300 \
    --cuda 0 \
    --seed 0 \
    --rounds 10
```

搜索后自动完成 Validation 选择，打印 final.json 路径。Test 不在此命令中加载。

| 参数 | 默认 | 说明 |
|---|---:|---|
| --parents / --mechanisms | 6 / 3 | 每轮父本数 / 单父本机制数上限 |
| --offspring | 12 | 每轮总提案数，操作分配由 LLM 决定 |
| --memory-limit | 24 | 检索的历史证据数上限 |
| --refine-top-k | 2 | 细化的高 R 子代数 |
| --windows | 5 10 20 40 60 | 窗口/滞后候选值 |
| --constants | -1 -0.5 0.5 1 2 | 系数、指数、阈值的候选值 |
| --max-refine-trials | 25 | 每个候选的参数网格组合上限 |
| --pool-capacity / --final-size | 60 / 30 | 工作池 / 最终池大小 |
| --min-stocks | 10 | 每次相关计算所需共同股票数 |
| --min-valid-days | 60 | IC、相邻日相关、因子对相关的最少有效观察 |
| --min-coverage | 0.8 | 可评价市场样本上的因子有效值比例 |
| --min-day-coverage | 0.8 | 可评价市场日期上的有效 IC 日期比例 |
| --horizon | 20 | 收益标签跨度 |
| --alpha / --beta / --gamma / --cost-weight | 1 / 1 / 1 / 0.2 | selection 四项权重 |
| --max-evals | 1500 | Train 未命中缓存的表达式执行次数，包含消融与参数试探 |
| --max-llm-calls / --max-output-tokens | 100 / 8192 | 模型调用上限 / 单次输出 token 上限 |
| --max-seconds / --reserve-seconds | 10800 / 900 | 总时间预算 / Validation 预留 |
| --chunk-size | 64 | 交易日计算分块 |
| --max-backtrack | 252 | 最大历史回看 |
| --max-nodes / --max-depth | 50 / 7 | 表达式大小和深度 |
| --qlib-path | 环境变量 | 覆盖数据目录 |
| --cuda -1 | — | 使用 CPU |

细化位置可由 LLM 指定，最多两个；未指定则使用前两个窗口。数值位置需要明确的参数角色，不自动扫全部常量。原参数始终参加比较；不枚举 rank 不变的外层偏移或同号缩放。网格去重/非法过滤后实际执行可少于上限。

例如增加诊断数量、调整细化网格：

```bash
python train_cf.py --mechanisms 5 --memory-limit 30 \
    --windows 5 10 20 40 60 --constants -1 -0.5 0.5 1 2
```

初选采用精确前向选择，199→60 需要 10170 次候选组合评分，不能只看表达式执行次数估计耗时。时间或评价预算耗尽时保存最近完整池；初始化或最终选择无法完成时明确报错。三小时是目标，实际耗时需运行测量。

## 最终 Test 评价

```bash
python train_cf.py \
    --test-only data/cf_logs/<运行目录>/final.json \
    --cuda 0
```

使用保存的数据配置、评价门槛和精确公式集合，不调用模型或重新选择。输出单因子指标、coverage、组合 U 和相关性；冻结因子失效时报告错误。旧产物缺少的新参数使用当前默认值，因此旧产物重评应留意统计口径变化。

## 消融与对照

在挖掘命令后增加对应开关，并保持初始池、数据、种子和预算上限一致。

| 开关 | 含义 |
|---|---|
| --no-cf-evidence | 保留语义分解，跳过干预构造和测量；--no-diagnosis 是兼容别名 |
| --no-pool-credit | 只计算 delta，检索和上下文不使用 pool credit |
| --random-crossover | 每个交叉子代独立随机指定机制对，LLM 负责组合表达式 |
| --no-memory | 只提供本轮证据 |
| --no-pool-selection | 初始、迭代和最终选择均仅按有符号 R |
| --no-refinement | 不细化窗口或数值参数 |

相同预算上限不代表实际消耗相同；日志分别报告诊断、生成、细化和池选择的成本与操作数量。随机 crossover 仅随机化机制配对，不代表所有生成决策均盲化。

已有 train_GP.py、train_ppo.py、train_gfn.py 提供 GP、AlphaGen/PPO、AlphaSAGE/GFN 入口；正式比较需统一其指标与数据协议。上游说明见 [Quickstart.md](Quickstart.md)。

## 日志与代码

产物位于 data/cf_logs/<时间戳>_<instrument>_<seed>/：

- args.json：参数。
- trace.jsonl：完整可见模型输入/输出、机制及基线、干预成功/失败、双重证据、父子代、参数 trial、selection、分阶段开销。
- pool_<轮次>.json、search_pool.json：每轮及最终 Train 池。
- memory.json：实测机制记忆，保留原父本、干预和池快照来源。
- final.json：Validation 冻结的最终公式和等权权重。
- test.json：独立 Test 评价。

诊断日志区分 returned、预算内 proposed、identified、structural、measured、pool_measured；measured_fraction 包含无基线或失败提议。另有 memory_retrieved、evolution_done、phase_done 和 pair_cache 命中统计。

alpha_pool.py 负责评价与选择，expression.py 负责 AST 与参数操作，trainer.py / prompt.py 负责机制演化。八步实现说明见 [plans/](plans/README.md)，约定见 [AGENT.md](AGENT.md)。

本轮按要求未运行测试、语法编译、训练或模型调用。许可沿用 [MIT LICENSE](LICENSE)。
