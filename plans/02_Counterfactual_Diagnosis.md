# 02 — 反事实诊断

实现：`AlphaCFTrainer.select_parents/diagnose`、`expression.ablate/at/replace`、
`PROMPT_DIAGNOSIS`。

## Parent 选择

默认 `parents=10`，实际名额 n=min(parents, 池容量)。轮初计算完整 Train 上的删除贡献 L_i=U(P)-U(P\{f_i})，删除后的池重新拟合 OLS。
按 L_i 升序取前 ceil(池容量/2) 个作为低贡献集合，同值保持池顺序；先从该集合均匀无放回抽 floor(n/2) 个，再从整个池的未选成员中均匀抽其余名额，最后打乱处理顺序。奇数时随机部分多一个；不使用单因子 RankIC 或访问次数筛选。
写 parent_selection 事件，记录轮初效用、全池删除贡献、各成员是否选中及抽样来源、最终 parent 顺序；终端打印两部分名额。
每次选择访问计数加一（用于诊断与去重，不参与后续选择）。
新表达式进入池时访问计数从 0 开始；目前实现不沿用历史同表达式的访问计数，
仅在 `select_parents` 中按选中与否递增。

## 提案协议

LLM 根据因子复杂度自行决定机制数量，0–5 个，不提供机制数量参数。

```json
{"understanding": "完整公式的行为、待验证经济解释与边界条件，最多三句", "mechanisms": [{
  "path": [],
  "description": "恒等乘法是否冗余",
  "replacement": "$close",
  "reason": "检验乘法是否冗余，预计删除后排序和有效覆盖不变"
}]}
```

先读取同链上一条历史的公式与 updated_understanding（可能描述祖先），重新审视当前公式。`--no-memory` 传空的 previous_understanding。`reason` 合并假设与预测，选择能区分因子不同解释的干预；预测不是测量。

同时考虑去冗余编辑（Add(f,0.0001)、Mul(Div(f,2),2)、Mul(f,1) 等）与窗口、输入或经济假设改进。
signal_distance 使用完整排名信号的覆盖修正相关距离 [0,1]；真正等价还须验证完整有效掩码与排名一致或整体反向一致。不能把非线性算子内的变换自动认作冗余，发现冗余不阻止改善探索。
没有编辑类型字段；根路径和常数叶节点均可作为有意义的诊断目标。
窗口不在 walk 的子路径列表内，修改窗口应替换整个 rolling subtree。
每次从原 parent 开始，不累积前一个编辑。

`ablate` 验证 path，解析 replacement，替换指定 subtree，再验证完整公式；
仍拒绝未改变公式、纯常量因子、非法窗口、未来引用、超大小/深度/历史范围。
这里的函数名沿用历史接口，实际支持一般反事实编辑，不限消融。
超过 5 项的模型响应直接报错并丢弃，不截断不默许。

## 测量与记录

轮初先计算完整 Train 的 U(pool)，并在主线程准备 parent 的上下文。默认用 5 个线程并发请求 LLM（--diagnosis-workers，可设 1）。
仅模型调用并发；按原 parent 顺序取结果，在主线程执行反事实测量。
该轮所有诊断引用同一轮初池，之后才按原顺序演化和替换。
日志写入加锁，请求、回复和错误均记录 task/factor/attempt，便于关联交错输出。
每条成功记录包含：

`id, factor, path, expression, description, replacement, reason, counterfactual,
delta_cf, pool_credit, signal_distance, coverage_change, common_rank_correlation`。

`id=m1/m2/...` 按原提案顺序编号，失败项不重编号。understanding 单独写入事件及 parent，终端打印理解与假设/预测。

指标定义见 [机制证据](03_Mechanism_Credit.md)。
无可评价变化、无效反事实组合等不能产生有限证据的编辑写入
`invalid_mechanism`，跳过。正常记录不带预设处置建议。

`--no-cf-evidence` 保留理解与语义提案，全部实测指标均 null；
本轮证据直接传给演化；替换或保留决定完成后，与全部子代提案和决定一起追加到该链历史。
空机制与未测量的轮次也记录，详见 [机制记忆](04_Mechanism_Memory.md)。
