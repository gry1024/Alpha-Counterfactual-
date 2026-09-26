# Step 7 — Pool Update

输入：当前 P 和细化后的 O。输出：P_next=Select_60(P∪O) 与选择日志。

代码：AlphaCFPool.select/update；AlphaCFTrainer.train/phase/save。

## 统一 Select_K

Step 1、Step 7、Step 8 共用精确前向选择；共享统计见 Step 1。

S(f|P)=alpha*R(f)+beta*(U(P∪{f})-U(P))+gamma*D(f,P)-lambda*cost(f)
D(f,P)=1-max_{g∈P}rho(f,g)

各步在可评价的剩余候选中逐列 min-max，全等列记 0；默认权重 1,1,1,0.2。空池 U=0、D=1，并列按公式字符串。

```text
C = 唯一且评价有效的 candidates
P = []; total_signal = 0
repeat k:
    计算剩余候选的 R、真实 marginal U、D、cost
    无定义的组合本步跳过，不赋虚构分数
    归一化，选 S 最大者
    更新 total_signal、U 和每个候选的 max_corr
return P
```

因子对共同有效观察不足时，不能当作零相关；候选与已选赢家无法计算多样性则排除。无法得到完整 k 个时报告失败；初始化/最终选择抛错，迭代更新记录 pool_update_rejected 并保留原池。

选择列表和 selection_log 都在完整选择后提交；更新函数只在成功时替换原池。无 offspring 保留 P。关闭 pool selection 时仅按有符号 R 取 k 个。

## 计算复用与日志

pair_cache 使用规范化公式对作为键，保存逐日绝对 Spearman 的均值；同一分区重复配对直接复用。keep 只保留当前池成员对应的标量与信号，分区间不共享缓存。保留 total_signal/max_corr 增量，组合 marginal 仍真实计算。

199→60 初选有 10170 次候选组合评分；eval_cnt 只统计未命中缓存的表达式执行，不能代表总成本。另记录 pool_scores、pair_scores、pair_hits。

phase_done 分别记录 initialize、diagnosis、evolution、refinement、selection 的耗时、表达式执行数、组合评分数和模型调用数；selection_done 记录选择耗时与相关缓存统计。

## 循环与保存

```text
P = Step 1
每轮：选择父本 → Step 2/3 → Step 4/5 → Step 6 → Step 7
保存 pool_<step>.json、memory.json 和 trace.jsonl
return P
```

时间或评价预算耗尽时停止，保存最近完整池。入口为 Validation 预留时间；初始化未完成完整池则报错。日志与产物位于 data/cf_logs/<run>/，不增加调度或性能框架。实际耗时未经本轮运行测量。
