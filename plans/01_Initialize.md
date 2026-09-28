# 01 — 初始化

对应代码入口：`train_cf.py::main / train`，具体类与函数：
- 数据层 `train_cf.load_data(args, split)`：构造 `alphagen_qlib.stock_data.StockData` 与 h 日 label
- 算子层 `src/alphagen/data/expression.py::ExpressionParser` 与 `Expression.evaluate`
- 候选层 `src/alpha_cf/expression.py::parse(text, args)`
- 评价层 `src/alpha_cf/alpha_pool.py::AlphaCFPool`
- 选择层 `src/alpha_cf/alpha_pool.py::AlphaCFPool.select` / `update` / `keep`
- 训练入口 `src/alpha_cf/trainer.py::AlphaCFTrainer.initialize` / `train`

---

## 1.1 候选种子与方向学习

输入 `--start-pool`，默认 `start_pool.json`。文件中包含 `description` 与 `exprs` 两段，`exprs` 是 200 个候选：
- **结构数 40** × **窗口数 5**（窗口集合见 `--windows`，默认 `[5, 10, 20, 40, 60]`）
- 类型覆盖：reversal、vol ratio、volume scaling、price-volume correlation、intraday range ratio、skew/kurtosis、return × volume 交互等。
- `description` 注明"direction fitted on Train only"，意味着每个种子方向只在 `train` 段反向一次。

逐个种子执行：

```python
# trainer.py::initialize
for seed in seeds:
    expr = self.evaluate(seed)                  # parse + pool.evaluate
    if expr is None: continue                    # parse / evaluate 失败
    if pool.evaluate(expr)["reward"] < 0:        # 有符号 RankICIR < 0
        expr = self.evaluate(Sub(0.0, expr))     # 整式取反，记录为新表达式
    candidates.append(expr)
```

要点：
- 方向学习只发生在 Train（split 由 `train_cf.load_data` 用 `SPLITS["train"]` 决定）。
- 反转后必须重新 `evaluate`，若仍 `< 0` 则保留 `Sub` 后的版本（不再二次反转）。
- 落选的种子不进入 `pool.exprs`，且不再作为后续步骤的候选——`Select_60` 是严格从这 200 个里挑 60 个。

---

## 1.2 Pool 评价与单个 reward

入口 `AlphaCFPool.evaluate(expr)` (`src/alpha_cf/alpha_pool.py`)：

```python
@torch.no_grad()
def evaluate(self, expr):
    key = str(expr)
    if key not in self.cache:
        # 1. 分块计算 expr 张量：shape [len(target), n_stocks]
        values = []
        for start in range(0, len(self.target), args.chunk_size):   # 默认 chunk_size=64
            block = copy(self.data)
            block.data = self.data.data[start: start + max_backtrack + min(start+chunk, len(target))]
            v = expr.evaluate(block)                                # 输出 [block_len, n_stocks]
            values.append(v.masked_fill(~v.isfinite(), nan))
        value = torch.cat(values)

        # 2. 截面 rank 归一化到 (-0.5, 0.5)：rank/(N-1) - 0.5
        count = value.isfinite().sum(1, keepdim=True)
        signal = (rank(value) / (count-1).clamp_min(1) - 0.5).masked_fill(count < 2, nan)

        # 3. 单因子 reward：signed RankICIR
        daily  = spearman(signal, self.target)                       # [T]
        reward = icir(daily).item()                                  # mean(daily)/(std(daily)+1e-8)

        # 4. turnover proxy：C_cost = (1 - mean_t Spearman(signal_t, signal_{t-1})) / 2
        cost = (1 - spearman(signal[1:], signal[:-1]).nanmean().item()) / 2

        # 5. 必须有可观察的横截面变化；否则直接抛错让 train 跳过该种子
        if not np.isfinite([reward, cost]).all():
            raise ValueError("Factor has no usable cross-sectional variation")
        self.cache[key] = dict(signal, reward, ic=daily.nanmean().item(), cost)
    return self.cache[key]
```

关键公式对应 idea.md：
- `R(f) = Mean(IC_t) / (Std(IC_t) + ε)`，其中 `IC_t = Spearman(signal_t, ret_{t→t+h})`，`ε=1e-8`
- `signal_t` 为横截面 `rank/(N-1) - 0.5`，所以 `E[signal_t]=0`，可以正负衡量。

实现细节：
- `rank()` 用 `torch.sort + 起始/结束位置 cummax/cummin` 做平均 tied ranks，跳过 nan
- `icir()` 在样本 < 2 时返回 nan，trainer 层会把 nan 视为不可用
- 缓存 key 是 `str(expr)` 字符串，避免同一表达式重复 evaluate

---

## 1.3 构造工作池

`AlphaCFPool.update(offspring)` 即 `self.exprs = self.select(self.exprs + offspring, args.pool_capacity)`：
- `pool_capacity=60`：每个 round 后池大小固定 60
- `select()` 走 idea.md §"Factor Selection" 的四分量打分（详见 `07_Pool_Update.md`）
- `keep()` 把 cache 限制在当前 exprs 上，释放掉被淘汰的 signal 张量

落选 seed 不保留——`pool.exprs` 始终是当前 60 个的精确集合；后续 diagnose/evolve 只访问这 60 个。

---

## 1.4 数据与窗口参数

`train_cf.load_data`：

```python
calendar = pd.DatetimeIndex(pd.read_csv(qlib_path/"calendars/day.txt")[0])
left  = calendar.searchsorted(start)
right = calendar.searchsorted(end, side="right")
data = StockData(instrument, start=calendar[left], end=calendar[right-1],
                 max_backtrack_days=args.max_backtrack,  # 默认 100
                 max_future_days=0, device, qlib_path)
# 构造 h 日前向收益作为 label
close = data.data[max_backtrack:, 1, :]                  # [T, n_stocks]
target = close[h:] / close[:-h] - 1                     # 默认 h=20（--horizon）
# 剔除 close<=0、收益非有限的样本
target = target.masked_fill(~target.isfinite() | close_nonpos, nan)
```

切分：
- `D_train = 2011-01-01 ~ 2021-12-31`
- `D_valid = 2022-01-01 ~ 2022-12-31`
- `D_test  = 2023-01-01 ~ 2026-04-30`（`run_adaptive_combination.py` 内 hard-coded）
- `data.df_bak = None` 后不再做 dataframe 回退，仅用 numpy 缓存

---

## 1.5 AST 合法性与 parse

`src/alpha_cf/expression.py::parse(text, args, featured=True)`：
- 规范化：去空白；`Greater/Less` → `GetGreater/GetLess`；科学计数法转 float
- `ExpressionParser().parse(text)` 走 alphagen 自带 parser
- 验证 `canonical(str(expr)) == text`（参数错误直接抛错）
- `validate(expr, args)`：
  - 必须包含 feature（`is_featured`）
  - 节点数 ≤ `args.max_nodes = 60`
  - 深度 < `args.max_depth = 10`（strict）
  - 累计 lookback ≤ `args.max_backtrack = 100`（Ref/TsDelta 加和）
  - 单步 `Ref/TsDelta` 窗口 ≥ 0 / 1

失败时 `trainer.evaluate` 捕获 `ValueError/IndexError` 并写入 `round_<step>.jsonl::invalid`，跳过该 candidate。

---

## 1.6 伪代码（train 阶段 start）

```python
def train(args):
    log_dir = Path("data/cf_logs") / timestamp_instrument_seed
    save_args(log_dir/args.json)
    data, target = load_data(args, "train")           # StockData + 20-day ret label
    pool   = AlphaCFPool(data, target, args)
    trainer = AlphaCFTrainer(pool, args, log_dir)
    seeds   = json.loads(Path(args.start_pool).read_text())["exprs"]   # 200
    trainer.train(seeds)                              # 走 initialize + 10 rounds
```

`trainer.train` 在第一步就是 `initialize(seeds)`，落选种子不进入任何后续步骤。
