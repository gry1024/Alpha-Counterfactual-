# Step 1 — Initialize

输入：INITIAL_EXPRS、Train 数据、args。输出：60 个工作因子 P0、评价缓存及 pool_0.json。

## 入口与复用

train_cf.py 使用 argparse → load_data → AlphaCFPool + AlphaCFTrainer。读取 .env 的 Qlib 路径与 MiniMax-M3 配置，不记录密钥。199 个初始公式混合 Alpha158 可表达子集、人工结构、GFN、PPO、知识搜索；只迁入公式，重新计算全部指标。

Train 为 2010–2021。StockData 加载最多 252 日历史，max_future_days=0。标签 close[h:]/close[:-h]-1 与前 n-h 日因子对齐，不跨分区补标签；加载前检查日历和回看长度。

expression.parse 复用原 parser，处理科学计数法、参数个数、featured、节点/深度、窗口及累计历史长度。规范化 str(expr) 去重。

## 评价接口

AlphaCFPool.evaluate(expr) → 包含 signal、valid_mask 和统计指标的 dict；无效返回 None。按日期分块调用原 Expression.evaluate，每个分区独立缓存。

- market：当前日收盘价有限且为正。因子 rank 只依赖当前可用股票，不使用未来标签可用性。
- eligible：market 且收益标签有限，是值覆盖率的分母。
- eligible_days：至少 min_stocks 个 eligible 股票且标签横截面有变化。
- IC_t：因子和标签共同有效股票上的 Spearman；并列取平均秩。
- R = mean(有效 IC)/(std(有效 IC, correction=0)+1e-8)，保留符号，不年化。
- q = average_rank/(n-1)-0.5；rank 从 0 开始，n<2 时缺失。
- F(P) = sum(nan_to_num(q_f))/|P|；缺失 q 表示中性，不重分配等权。
- U(P) = RankICIR(F(P))；对组合信号重新 rank。
- rho(f,g) = mean(有效日期的 abs(Spearman(f_t,g_t)))。
- cost = (1-mean(有效相邻日 Spearman(f_t,f_{t-1})))/2。

rank 通过排序和同值组首尾位置求平均秩；spearman 复用 batch_pearsonr。无效相关日为 NaN，不填 0。

## 有效性默认值

| 参数 | 默认 | 用途 |
|---|---:|---|
| min_stocks | 10 | 每次相关系数所需共同股票数 |
| min_valid_days | 60 | IC、相邻日相关、因子对相关所需有效观察数 |
| min_coverage | 0.8 | eligible 样本中有限因子信号比例 |
| min_day_coverage | 0.8 | eligible_days 中 IC 可计算比例 |

组合 U 同样检查有效 IC 日数与日期覆盖率。无定义的相关性不记作零相关。上市前等基础市场缺失不纳入值覆盖率分母。

cache[str(expr)] 存 rank 信号、有效 IC 日期 mask 和指标；metrics(entry) 排除张量用于日志/JSON。keep(exprs) 只保留当前池张量与对应的相关性标量缓存。

## 初始化

```text
C0 = 解析、去重、评价 INITIAL_EXPRS
seen = 已解析的全部初始公式
P0 = select(C0,60)                 # 统一选择见 Step 7
只保留 P0 缓存，保存 pool_0.json
```

不足 60 个有效唯一因子时报错。未入 P0 的初始公式不在后续生成/细化中复活。
