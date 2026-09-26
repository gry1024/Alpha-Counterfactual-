# Step 6 — Offspring Refinement

输入：已完成默认参数评价的 O，以及可选的 refine 元数据。输出：以更优参数实例替换原式的 O。

代码：AlphaCFTrainer.refine，expression.parameter_specs/parameter_variants。

## 参数位置与网格

LLM 可为 child 指定最多两个位置：

- kind=window：path 指向 rolling 节点，修改 _delta_time。
- kind=constant：path 指向 Constant，role 为 coefficient/exponent/threshold，修改 _value。
- values 可指定配置网格的子集；省略使用完整网格。
- 未给 refine 或给 []：回退 AST 前序的前两个窗口。

默认 windows=[5,10,20,40,60]；constants=[-1.0,-0.5,0.5,1.0,2.0]，均可由 CLI 调整。参数位置必须真实、类型匹配、不重复；非法建议记录后回退窗口规则。

不自动枚举所有常量。prompt 要求避开符号构造、中性元和稳定性 epsilon；程序拒绝非零绝对值小于 1e-6 的常量位置、最外层加减常量。最外层乘除常量只保留原值与反号候选，省去 rank 不变的同号缩放。

## 枚举与选择

```text
eligible = O 中 R 最高的 refine_top_k 个，默认 2
for original in eligible:
    best = original
    specs = 已验证位置或默认窗口位置
    for values in 有限笛卡尔积的前 max_refine_trials 项:
        复制 AST，只改参数
        验证历史/语法；跳过原式、非法或 seen 变体
        实测 R；优于 best 则替换
    O 中 original 改为 best
return O
```

max_refine_trials 默认 25；原式始终作为基线参加比较。上限作用于网格组合，因此去重或非法过滤后实际执行可更少。只选参数，不改算子和特征；枚举结果按有符号 R 选优。

每次执行记录 parent、variant、位置、数值、R 或失败。失败/落选变体张量释放，全局 seen 阻止重复执行。新参数实例不继承原机制信用；未参与细化的 offspring 仍进入 Step 7。
