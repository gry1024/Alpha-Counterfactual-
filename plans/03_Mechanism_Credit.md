# Step 3 — Mechanism Credit

输入：固定池 P、父本 f、机制 m、干预 T(f,m)。输出：两类实测证据及其条件。

代码：AlphaCFTrainer.diagnose；AlphaCFPool.evaluate/utility。

## 计算

```text
base = U(P)                         # 每轮诊断一次
before = evaluate(f)
after = evaluate(T(f,m))
delta_cf = after.reward - before.reward
pool_credit = base - U((P 去掉 f) ∪ {T(f,m)})
```

P 在诊断中不变。替换后的集合按公式去重、重新等权计算，不用增加候选的 marginal 代替替换效应。干预式不自动进入 offspring。

delta_cf<0 支持该结构对父本的价值；pool_credit>0 支持其对当前池的价值。冲突证据保留；独立消融不构成可相加分解。

## 记录与使用

记录来源、step、pool_snapshot、path、description、subtree、mode、replacement、ablated_expr、baseline_kind、reason、两边 reward、delta_cf、pool_credit；pool credit 可计算时还记录 pool_before/pool_after。

pool_snapshot 指向本轮诊断前的 pool_<step-1>.json。若因子干预可测但替换后的组合无定义，保留已测 delta，pool_credit=null 并记录 pool_error，不伪造零贡献。

Action 为提示：两项均支持标 Preserve，均反向标 Replace，其余 Explore；关闭 pool credit 时仅依据 delta。evolve 不按 Action 或符号限制保留、交叉或替换资格，LLM 直接读取原始两项证据。

## 样本口径

R 在各自有效日期上计算，继续采用 idea 的 R(T)-R(f)。同时记录：

- parent_coverage、ablated_coverage；
- parent_valid_days、ablated_valid_days；
- common_valid_days；
- valid_day_jaccard=有效日期交集/并集。

这些记录揭示覆盖变化，不把不同样本上的差分宣称为已消除样本偏差的归因；没有改用共同样本重新计算 R。
