# 反事实因子编辑（CounterFactual）

在 AlphaSAGE 种子因子上做最小结构编辑：LLM 提案 → `T(f,a)` → 真回测 Δ → greedy。详见 `dev_plan.md`。

`R(f)=|日均 Pearson IC|`。Train 段固定 `2010-01-01`–`2021-12-31`。

## 环境

Python ≥ 3.11。用 [PDM](https://pdm-project.org/)：

```bash
pdm install
```

或已有 `.venv` 时：

```bash
source .venv/bin/activate
```

## 数据

Qlib 二进制放在仓库内：

- CSI300：`data/qlib_data/cn_data_rolling`
- SP500：`data/qlib_data/us_data_qlib_latest`

种子因子来自 AlphaSAGE 的 `pool_*.json`（含 `"exprs": [...]`），例如 GFN 跑完后的：

```text
data/gfn_logs/pool_50/.../pool_9999.json
```

没有 pool 时先跑 `train_gfn.py`（见 https://github.com/BerkinChen/AlphaSAGE.git）。

## LLM

复制 `.env.example` 为仓库根目录 `.env`，填：

```text
LLM_API_KEY=...
LLM_BASE_URL=https://api.minimaxi.com/v1
LLM_MODEL=MiniMax-M3
```

默认开 thinking，并用 `reasoning_split` 把思考链和 JSON 分开。关掉：`LLM_THINKING=disabled`。

## 跑

```bash
PYTHONPATH=src python train_cf.py \
  --pool_json data/gfn_logs/pool_50/<run>/pool_9999.json \
  --instrument csi300 \
  --n_seeds 20
```

常用参数：`--max_rounds`（默认 10）、`--k`（每轮提案数，6）、`--topk`（Q 开启后回测条数，3）、`--patience`（5）、`--no_q`（每轮全回测）、`--propose random`（随机合法编辑，不用 LLM）。

日志目录默认 `data/cf_logs/<时间戳>/`，也可用 `--out_dir`。

## 产物

| 文件 | 内容 |
|---|---|
| `cf_records.jsonl` | 每条真回测 `(f, a, f', Δ, ic, rank_ic)` |
| `pool.json` | 各种子搜完后的最终表达式 |
| `credit.json` | 按编辑类型 / 特征 / 算子聚合 mean Δ |
| `illegal.jsonl` | `T(f,a)` 拒绝的编辑 |
| `summary.json` | 本轮规模与 Δ 摘要 |
