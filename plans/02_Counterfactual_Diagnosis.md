# 02 — 反事实诊断

实现：`AlphaCFTrainer.select_parents/diagnose`、`expression.ablate/at/replace`、
`PROMPT_DIAGNOSIS`。

## Parent 选择

默认 `parents=10`，先取 signed RankIC top-5，再从剩余成员中选访问最少的 5 个。
低访问并列时按 RankIC 排序；不重复选择。每次选择访问计数加一。
配置其他总数时，top-k 为实际数量的一半向下取整，其余配额给低访问。
新表达式访问数从 0 开始；历史同表达式再次出现时沿用其访问计数。

## 提案协议

LLM 根据因子复杂度自行决定机制数量，0–5 个，不提供机制数量参数。

```json
{"mechanisms": [{
  "path": [],
  "description": "恒等乘法是否冗余",
  "replacement": "$close",
  "reason": "比较去掉 Mul(1.0,...) 前后的预测和信号"
}]}
```

允许删除算子、改窗口、引入新输入、重写经济假设。
没有编辑类型字段；根路径和常数叶节点均可作为有意义的诊断目标。
窗口不在 walk 的子路径列表内，修改窗口应替换整个 rolling subtree。
每次从原 parent 开始，不累积前一个编辑。

`ablate` 验证 path，解析 replacement，替换指定 subtree，再验证完整公式；
仍拒绝未改变公式、纯常量因子、非法窗口、未来引用、超大小/深度/历史范围。
这里的函数名沿用历史接口，实际支持一般反事实编辑，不限消融。
超过 5 项的模型响应直接报错并丢弃，不截断不默许。

## 测量与记录

轮初先计算 total 与 U(pool)，该轮所有诊断都引用同一池快照。
每条成功记录包含：

`factor, path, expression, description, replacement, reason, counterfactual,
delta_cf, pool_credit, signal_distance`。

三项指标见 [机制证据](03_Mechanism_Credit.md)。
无有效重叠横截面、无效反事实组合等不能产生有限证据的编辑写入
`invalid_mechanism`，跳过。正常记录不带预设处置建议。

`--no-cf-evidence` 保留语义提案，三项指标均 null；
仅成功测量 delta_cf 的记录写入 memory。
