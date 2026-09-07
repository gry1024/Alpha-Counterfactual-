# 反事实因子编辑：开发计划

依据 [idea.md](idea.md)。只新增代码，不改 AlphaSAGE / GFN。入口 **`train_cf.py`**，业务在 `src/alpha_cf/`。

---

## 0. Pipeline（按现有实现）

奖励：`R(f) = |日均 Pearson IC|`（train 段 2010–2021）。真回测只发生在 evaluator 里。LLM 只提案，不算 IC。Greedy 只看真 Δ。

五种编辑（一次改一处，`site_id` 前序）：`feature_replace` / `operator_replace` / `window_replace` / `subtree_delete` / `wrap`（`wrap` 的 `new_value` 必须带 `$_`）。`T(f,a)` 非法则跳过并记日志，不中断整次训练。

```text
输入: AlphaSAGE pool_*.json 里的种子表达式
输出: data/cf_logs/<ts>/
      cf_records.jsonl   每条真回测过的 (f, a, f', Δ, ic, rank_ic)
      pool.json          每个种子搜完后的最终表达式
      credit.json        按 kind / feature / op / window 聚合 mean Δ
      illegal.jsonl      被 T 拒绝的编辑
      summary.json

R(f) := |IC(f)|          # 横截面日均 Pearson IC 的绝对值
Δ(f,a) := R(T(f,a)) - R(f)

train_cf:
    seeds ← 读 pool，去重，取前 n_seeds 条
    cache ← qlib 行情 + IC 计算器（带表达式缓存）
    all_recs ← []                 # 跨种子共享，用来训 Q
    Q ← 未拟合                     # --no_q 则全程不用 Q

    for seed f in seeds:
        f* ← search_one(f, cache, all_recs, Q)
        all_recs ← all_recs + 本种子产生的 records
        把 f* 写入 pool.json
        若 |all_recs| ≥ 128 且未 --no_q:
            下次种子起 Q 才开始筛 top-k

    写 jsonl / credit / summary


search_one(f):                         # 一个种子，greedy 多轮
    回测 f，得到 R(f)
    stale ← 0

    for round = 0 .. max_rounds-1:
        # --- Propose ---
        把 f、R(f)、可编辑位点（含 T 过滤后的 ∈ 集合）、
        本种子最近若干轮的真实 Δ traces 交给 LLM
        LLM 吐 K 条 EditAction（thinking 与 JSON 分开；只 parse content）
        非法 / 空 JSON：跳过该条或该批，不改用随机提案

        # --- T(f,a) ---
        对每条 a: f' ← apply_edit(f, a)
        丢掉: T 失败、f'=f、本轮重复的 f'

        若本轮一条可用都没有:
            stale += 1; 若 stale ≥ patience 则停
            continue

        # --- 决定回测谁 ---
        若 Q 未开 或 累计 CF 样本 < 128:
            batch ← 全部可用提案          # 冷启动：K 条全回测
        否则:
            用目前所有 (f,a,Δ) 重训 Q     # LightGBM：μ 与 0.1/0.9 分位
            score = μ + βσ + λD          # D = 相对已试 (f,a) 的结构距离
            batch ← score 最高的 topk 条

        # --- 真回测 ---
        for (a, f') in batch:
            用 cache 算 IC / RankIC / R(f')
            记一条 record: Δ = R(f') - R(f)
        取本批 Δ 最大的那条

        # --- Greedy ---
        若 max Δ > 0:  f ← 对应 f';  stale ← 0
        否则:          stale += 1; 若 stale ≥ patience 则停

    return 本种子 records, 最终 f


T(f, a) 五种:
    feature_replace   某 $feature → 另一个 feature
    operator_replace  同 arity 算子互换（∈ 集合已按「套完仍 is_featured」滤过）
    window_replace    滚动窗口换成 WINDOWS 里另一个
    subtree_delete    删掉该子树，兄弟顶上或一元/滚动拆掉一层；不能删根
    wrap              在该子树外包一层；new_value 如 Abs($_)，$_ = 当前子树
```

默认规模：K=6，topk=3，max_rounds=10，patience=5，n_seeds=20，MIN_Q_SAMPLES=128。`--propose random` 可换成随机合法编辑（消融）。`--no_q` 则每轮提案全回测。

谁干什么：

| 谁 | 干什么 | 不干什么 |
|---|---|---|
| LLM | 提出「改哪、改成什么」 | 不算 IC；失败不回退随机 |
| Q | 预测未回测编辑的 Δ，用来排序 | 不替代真回测；不够 128 条不用 |
| evaluator | 唯一真相：R(f)、Δ | 不改树 |
| greedy | 只在真 Δ>0 时换当前 f | 不按 Q 的分数换 f |

---


下面各节是最初的拆文件计划，循环细节以第 0 节为准（现实现没有 `subtree_replace` / `eval_budget`，Q 是 LightGBM 表特征不是 MLP）。

---

## 1. 数据流

idea.md 的闭环：

```text
Generate → Intervention → Credit → Q_phi → Guided Search
```

落到本仓库的实际数据流：

```text
AlphaSAGE pool_*.json                         # Step1 已完成，本仓库不重训 GFN
        │
        v
读 exprs，解析成 Expression，算 R(f)           # R(f)=|日均 Pearson IC|
        │
        v
┌──────────────── 对每个种子 f，循环 ────────────────┐
│  LLM 提出 K 个最小结构编辑 a                       │  Step2  Intervention
│  T(f,a) → f'（只改树，不回测）                      │
│                                                     │
│  Q 还没样本：K 个全部真回测                         │
│  Q 已训练：score=Q+βU+λD，只回测 top-k              │  Step5  Guided Search
│                                                     │
│  Δ = R(f')−R(f)  写入 jsonl                        │  Step3  Credit
│  攒够样本 → 训 Q_phi(f,a)→Δ                         │  Step4  Value Model
│  若 max Δ>0：f ← 那个 f'（greedy）                  │
│  直到 budget / patience / 全部 Q<0                  │
└─────────────────────────────────────────────────────┘
        │
        v
credit 表 + discovered_pool.json
        │
        v
现成 run_adaptive_combination.py 做组合/回测（不在 alpha_cf 里重写）
```

职责拆开：

| 谁 | 干什么 | 不干什么 |
|---|---|---|
| LLM | 提出「改哪、改成什么」 | 不算 IC；失败直接报错，不回退随机 |
| Q_phi | 预测未回测编辑的 Δ，用来排序 | 不替代真回测 |
| evaluator | 唯一真相：R(f)、Δ | 不改树 |

Q 的输入：先 `T(f,a)` 得到 f'，再

```text
φ(f), φ(f') = 同一套轻量结构编码（先 MLP 统计特征；可后续接 AlphaSAGE RGCN）
Q = MLP([φ(f); φ(f'); φ(f')−φ(f)]) → (μ, σ)
score = μ + β·σ + λ·D
D = 相对已试过 f' 的结构距离
```

---

## 2. 目录

```text
train_cf.py                      # 唯一运行入口：argparse → 调 alpha_cf → 写 data/cf_logs/
src/alpha_cf/
  __init__.py                    # 包标记，不放流程
  config.py                      # 日期切分、K/topk/budget、β/λ、路径
  types.py                       # EditAction / Site / CFRecord 等小 dataclass
  pool.py                        # 读 AlphaSAGE pool JSON，解析 Expression，去重
  edits.py                       # 枚举位点 + T(f,a) 五种最小编辑
  reward.py                      # 缓存 R(f)=|IC|，顺带记 RankIC
  llm.py                         # OpenAI-compatible 调 LLM，输出合法 EditAction
  q_model.py                     # Q_phi 网络 + train/predict
  search.py                      # 单因子 greedy 循环（上面框图那一圈）
  credit.py                      # 按 kind/old 聚合 mean Δ（RQ1）
.env                             # LLM key（gitignore）
```

复用、只 import 不改：

```text
src/alphagen/data/expression.py     Expression AST
src/alphagen/data/tree.py           字符串 ↔ 树
src/alphagen/config.py              OPERATORS / FEATURES / windows
src/alphagen_qlib/calculator.py     IC / RankIC
src/alphagen_qlib/stock_data.py     Qlib 行情
```

产物（`train_cf.py --out_dir` 指定，默认 `data/cf_logs/<时间>/`）：

```text
cf_records.jsonl      每条 (f, a, f', Δ, IC)
credit.json           结构归因
discovered_pool.json  改进后的 exprs，兼容现有组合脚本
```

---

## 3. 各文件怎么写

### `train_cf.py`

和 `train_gfn.py` 一个风格：上面 argparse，下面顺序调用函数。大约：

```text
load pool → 建 StockData/calculator → 对每个种子 search_one → dump
```

不要 `class CounterfactualPipeline`。不要 `__main__.py` 再包一层 CLI。

### `edits.py` — T(f,a)

五种动作，一次只改一处：

- feature_replace：`$close` → `$volume`
- operator_replace：同 arity，`TsMean` → `TsMax`
- window_replace：`20` → `10`
- subtree_replace：该节点换成另一合法子式
- subtree_delete：拆掉一层算子

前序 `site_id`。非法编辑丢掉；LLM 一条合法都没有就 raise。

### `llm.py`

OpenAI-compatible。Prompt：当前表达式、R(f)、带 allowed 的位点、最近几条 Δ。返回 JSON list。key 空 / HTTP 失败 / 无合法编辑 → **raise**，禁止随机 propose。

`.env`：`LLM_PROVIDER` + `MINIMAX_API_KEY` 等。默认 MiniMax-M3。

### `search.py` — `search_one(f, ...)`

就是第 1 节框图。冷启动全回测攒 (f,a,Δ)；`len(records)≥min_q_samples` 后才用 Q 筛 top-k。Greedy 只看**真 Δ**。

停：`max_rounds` / `eval_budget` / `patience` / 全部 score<0。

### `q_model.py`

第一版固定维特征 + 小 MLP 即可，不必一上来绑 GFN 的 token id。Gaussian NLL 出 μ、σ。指标看 Spearman / top-k，不抠 MAE。

### `reward.py`

`R(f)=|日均 Pearson IC|`（与 idea 的 factor-level reward 一致，方便长短因子一起比）。Δ 用这个 R。signed IC、RankIC 一并记下，不混进 novelty/SSL。

日期（config 写死，禁止用「今天」）：

```text
train  2010-01-01 ~ 2021-12-31     # 回测 Δ、训 Q
valid  2022-01-01 ~ 2022-12-31     # 早停 / 选 checkpoint
test   2023-01-01 ~ 2026-04-30     # 只最终报告
```

---

## 4. 和 idea.md 三问的对应

| RQ | 何时有答案 | 看什么 |
|---|---|---|
| RQ1 哪段结构贡献了预测 | 有 jsonl 之后 `credit.py` | 按 feature/op/window 的 mean Δ |
| RQ2 已观察干预能否预测未试动作 | Q 训完 | withheld 上 Spearman、top-k |
| RQ3 同样回测预算能否挖得更好 | `search_one` 跑完 | best IC vs 评价次数；对比「不用 Q、全回测」 |

第一版不做论文全部 baseline。日志格式要能事后加 random intervention 对照。不做 GFN 在线 callback。

---

## 5. 实现顺序

1. `types` + `edits` + `pool`：手写表达式能改窗/换算子，能读 `pool_*.json`
2. `reward`：同一 f 算得出稳定 IC
3. `llm`：对着位点吐出合法 JSON；失败要报错
4. `search` + `train_cf.py`：LLM 提案全回测，greedy 能换 f，写出 jsonl
5. `q_model`：样本够了再筛 top-k
6. `credit` + discovered_pool

默认规模（跑通，不是终表）：K=8，topk=3，max_rounds=5，patience=2，种子 5～10。

```bash
PYTHONPATH=src python train_cf.py \
  --pool_json data/gfn_logs/.../pool_9999.json \
  --n_seeds 5
```
