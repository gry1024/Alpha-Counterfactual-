# 08 — 最终选择与测试

对应代码：`train_cf.py::train / test / main` 与现有 `run_adaptive_combination.py`。

整个 train 阶段的最后两步：① Validation 段做 `60 → 30` 终选；② 把 `final.json` 直接交给原 `run_adaptive_combination.py` 做 Val/Test 评估。

---

## 8.1 终选入口

`train_cf.py::train(args)` 末尾（`train_cf.py:78-94`）：

```python
# 1) 落盘 search_pool.json（Train 阶段完整 60 个）
save_json(log_dir / "search_pool.json",
          dict(exprs=[str(e) for e in expressions], args=vars(args), split="train"))

# 2) 重新加载 Validation 段 StockData + label
data, target = load_data(args, "valid")                                   # 2022
pool = AlphaCFPool(data, target, args)

# 3) 在 Validation 上重新 evaluate 60 个候选；失败的丢弃
candidates = []
for expr in expressions:
    try:
        pool.evaluate(expr)
        candidates.append(expr)
    except ValueError as exc:
        print(f"Validation unavailable: {expr}: {exc}", flush=True)

# 4) 复用 select() 做 60 → 30 选择（同一打分函数 S）
pool.exprs = pool.select(candidates, args.final_size)                     # 30

# 5) 写 final.json：必须含 split="valid"，test() 会校验
save_json(log_dir / "final.json", dict(**pool.to_dict(), args=vars(args), split="valid"))

# 6) 打印最终因子列表
for i, (expr, metrics) in enumerate(zip(pool.exprs, pool.to_dict()["metrics"])):
    print(f"> Alpha {i + 1}: RankICIR={metrics['reward']:.4f}, expr={expr}", flush=True)

del pool, data, target
release()
test(args, log_dir / "final.json")                                         # 7) 调原脚本
```

要点：
- **不**重新生成、不细化、不翻转符号——终选只是同一打分函数在 Validation 段的 selection
- **不**修改 `run_adaptive_combination.py`——只在 subprocess 中按原样调用
- `final.json` 必须含 `"split": "valid"`，否则 `test()` 校验会抛错（`source.read_text()["split"] != "valid"`）
- `release()` 显式 `gc.collect()` + `torch.cuda.empty_cache()`，确保 subprocess 拿到干净的 GPU 状态

---

## 8.2 Test 阶段：冻结产物 → 原回测脚本

`test(args, source)`（`train_cf.py:51-60`）：

```python
def test(args, source):
    if json.loads(source.read_text()).get("split") != "valid":
        raise ValueError("Test requires a frozen final.json from Validation")

    root = Path(__file__).resolve().parent
    env = dict(os.environ,
               PYTHONPATH=os.pathsep.join([str(root / "src"), os.environ.get("PYTHONPATH", "")]))

    subprocess.run([
        sys.executable, str(root / "run_adaptive_combination.py"),
        "--expressions_file", str(source.resolve()),
        "--instruments",    args.instrument,
        "--train_end_year", "2021",
        "--label_days",     str(args.horizon),                            # 默认 20
        "--cuda",           str(args.cuda),
        "--seed",           str(args.seed),
        "--n_factors",      str(args.n_factors),                          # 默认 10
        "--chunk_size",     str(args.chunk_size),                         # 默认 64
    ], cwd=root, env=env, check=True)
```

- 用 `subprocess.run(check=True)`：原脚本非零退出码会立刻抛错，便于发现异常
- `--train_end_year=2021` 强制回看截至 2021-12-31（搜索阶段一致）
- `cwd=root`：从仓库根启动，保证 `data/qlib_data/...` 路径与原脚本约定一致
- `PYTHONPATH` 注入 `src/`，与 `train_cf.py::main` 保持同一 import 环境

**重要：搜索回看长度 = 100 日**。`run_adaptive_combination.py` 默认从 `StockData` 拿 `max_backtrack_days`；`train_cf.py` 里 `--max-backtrack` 默认 100——这个值会通过 `args.max_backtrack` 影响 `StockData` 的构造。`run_adaptive_combination.py` 用的是 `data_all.max_backtrack_days`，因此保持一致。

---

## 8.3 原回测脚本做了什么

`run_adaptive_combination.py::run(args)`：

1. **数据**：
   - `StockData(instruments, start='2010-01-01', end='2026-04-30', qlib_path)` 全量加载
   - `target = Ref(close, -label_days) / close - 1` 构造 h 日前向收益
   - 取 `_date_mask` 划分 valid（2022）/ test（2023-01-01 ~ 2026-04-30）
   - `eval_idx = valid_mask | test_mask`

2. **逐日自适应组合**：
   - 对 `eval_idx` 中每一天 `cur`：
     - 回看窗口：`begin = max(0, cur - window - shift)`，其中 `shift = label_days + 1` 防 lookahead
     - 计算窗口内每个因子的 `ric / ricir`（Spearman）
     - 过滤 `|ric| > threshold_ric=0.015 && |ricir| > threshold_ricir=0.15`；不足时退回取 `|ricir|` 最大的一个
     - 取前 `n_factors=10` 个
     - 构造回归 `coef = lstsq(x, y)`，`x` 加常数项
     - 预测当天：`pred = to_pred @ coef`
   - 滚动输出 pred 序列

3. **指标**：`get_tensor_metrics(pred, target)` 输出 ic/icir/ric/ricir/ret/retir/ret_sharpe/ret_mdd 等

4. **保存**：`ret_s.npy` 写到 `final.json` 同目录下

输出（按段打印）：

```
--- Final Performance Metrics ---
            ic  ic_std  icir  ric  ric_std  ricir  ret ...
Validation ...
Test       ...

--- Parseable Format ---
Dataset     IC   IC_STD   ICIR    ...
Validation  ...
Test        ...
```

---

## 8.4 replay 模式

`train_cf.py::main` 支持两种 replay（互斥）：

### `--test-only FINAL_JSON`

```python
parser.add_argument("--test-only", metavar="FINAL_JSON")
```

跳过 Train + Validation，直接调 `test(args, source)`。用于：
- 复跑已有 `final.json` 的 Test 评估（不改因子）
- 多次运行验证测试稳定性

`source` 的 `args.json` 会被读入（`config = saved.get("args") or json.loads(...)`），命令行当前参数会覆盖它。

### `--finalize-only TRAIN_POOL_JSON`

```python
parser.add_argument("--finalize-only", metavar="TRAIN_POOL_JSON")
```

跳过 Train，直接走 Validation 终选 + Test：

```python
if args.finalize_only:
    expressions = [parse(text, args) for text in json.loads(Path(args.finalize_only).read_text())["exprs"]]
else:
    # 完整 train()
```

校验：`saved.get("split")` 必须是 `"train"`，否则 `parser.error("--finalize-only requires a Train pool")`。

典型用途：
- 训练中途出故障、想用上一轮 `search_pool.json` 继续做 Validation/Test
- 改变 `--final-size` 后重跑终选
- 改变 `--alpha/beta/gamma` 后重跑终选

---

## 8.5 不复用的脚本

`AlphaCF` 一开始有个独立的 `evaluate.py` 与重复的 Test/OLS 实现——已在方案 1 中**移除**：

- 重复的 `remove_linearly_dependent_rows/cols/calculate_vif/remove_multicollinearity_vif`：原回测脚本已自带
- 重复的 Test/OLS 实现：直接走 `run_adaptive_combination.py`

不再写独立 evaluate；不再维护独立 Test 脚本。

---

## 8.6 数据目录约定

| 阶段 | 数据路径 | 入口 |
|---|---|---|
| Train / Valid（AlphaCF） | `--qlib-path` 默认 `data/qlib_data/cn_data_rolling` 或 `data/qlib_data/us_data_qlib_latest` | `train_cf.py::main` |
| Valid / Test（回测） | 原 `run_adaptive_combination.py` 内部 hard-coded `data/qlib_data/cn_data_rolling` | `run_adaptive_combination.py::run` |

`.env` 里 `QLIB_PATH_CN` / `QLIB_PATH_SP500` 决定默认路径；`--qlib-path` 可显式覆盖。

---

## 8.7 关键参数对照

| 参数 | train_cf 默认 | 传给 run_adaptive_combination |
|---|---:|---|
| `--instrument` | csi300 | `--instruments csi300` |
| `--horizon / --label-days` | 20 | `--label_days 20` |
| `--seed` | 0 | `--seed 0` |
| `--cuda` | 0 | `--cuda 0` |
| `--chunk-size / --chunk_size` | 64 | `--chunk_size 64` |
| `--n-factors` | 10 | `--n_factors 10` |

其余回测参数（`--window='inf'`, `--threshold_ric=0.015`, `--threshold_ricir=0.15`, `--linear_dep_tol=1e-10`, `--ridge_alpha=1e-6`）都按原脚本默认。

---

## 8.8 异常与快照

- 任何 round 抛出 → `train()` 中断；`pool_<step-1>.json` 仍可用
- Test 阶段 `subprocess.run(check=True)` 抛 `CalledProcessError` → 上层 trainer 也会失败
- 不存在"中断搜索自动当作完整训练"的逻辑——必须靠 `--finalize-only` 显式恢复

---

## 8.9 完整生命周期汇总

```
150  →  Select_60  (init: direction learning on Train)
60   →  diagnose+evolve (round 1, 10 parents × 5 offspring)
60   →  diagnose+evolve (round 2, ...)
...
60   →  diagnose+evolve (round 10, ...)
60   →  Select_30  (final on Valid)
30   →  run_adaptive_combination.py → Val / Test metrics
```

`Select_30` 与 `Select_60` 走**同一个** `AlphaCFPool.select` 函数，仅 `size` 参数不同（`args.final_size=30` vs `args.pool_capacity=60`）。打分公式、归一化、缓存语义、贪心顺序完全一致。
