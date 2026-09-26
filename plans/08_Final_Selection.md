# Step 8 — Final Selection

输入：Train 最终 PG，仍为 60 个公式。输出：Validation 选出的约 30 个最终因子，以及独立 Test 评价。

代码：train_cf.py 的 train/evaluate_test；AlphaCFPool.select/to_dict。

## 执行

| 分区 | 日期 | 用途 |
|---|---|---|
| Train | 2010-01-01～2021-12-31 | Step 1–7 |
| Validation | 2022-01-01～2022-12-31 | PG 中选 30 |
| Test | 2023-01-01～2026-04-30 | 评价冻结集合 |

```text
保存 search_pool.json；释放 Train 数据与缓存
创建独立 Validation pool
final = select(PG,30)
保存 final.json，等权
```

Validation 不生成或细化表达式，不反馈到搜索。各段标签独立，不跨分区。--test-only <final.json> 读取冻结配置和精确公式集合，不调用模型或重新选择；某因子失效时报错，不悄悄删除。

旧 final.json 缺少新增字段时，使用当前 CLI 默认值补齐；保存的已有字段优先，设备和本次时间预算例外。新 final.json 保存全部 coverage 等参数以复现相同评价口径。

## 消融定义

| 开关 | 实际行为 |
|---|---|
| --no-cf-evidence；别名 --no-diagnosis | 语义分解保留，跳过干预构造与测量；无反事实 memory |
| --no-pool-credit | 仅计算 delta；历史检索和 prompt 不使用 pool credit |
| --random-crossover | 每个 crossover 消费独立随机机制对；组合表达式仍由 LLM 生成 |
| --no-memory | 不提供历史；当前 evidence 保留 |
| --no-pool-selection | 初始/迭代/最终选择均只按 R |
| --no-refinement | 保留 child 默认参数 |

对照使用相同数据、初始公式、种子与预算上限，记录实际不同阶段消耗和操作数量。相同预算上限不保证实际消耗相同；不把随机 crossover 解释为所有 LLM 决策均盲化。

复用现有 GP/AlphaGen/AlphaSAGE 入口，实验前需统一其指标与数据协议；不新增实验调度系统。

## 产物与交付

args.json、trace.jsonl、pool_<step>.json、search_pool.json、memory.json、final.json；仅 test-only 输出 test.json。README 提供默认参数、启动和消融命令。

按用户要求，本轮仅开发与阅读核对，没有执行测试、语法编译、训练或 LLM 请求。
