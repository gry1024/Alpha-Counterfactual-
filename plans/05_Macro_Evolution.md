# Step 5 — Macro Evolution

输入：当前 parents、本轮 evidence、检索历史、表达式限制和 offspring=12 的总提案预算。输出：默认参数评价有效的 O。

代码：AlphaCFTrainer.evolve/ask，prompt.SYSTEM/EVOLVE。

## LLM 决策与程序检查

LLM 直接输出完整 expression、operation、parents、keep、reason，可附 refine。三类操作无独立配额，程序不预选固定 crossover pair，也不按 useful 符号筛机制。

| 操作 | LLM 决定 | 程序检查 |
|---|---|---|
| mutation | 保留哪些机制，如何重写外围 | 一个当前父本；至少一个真实机制保持完整 |
| replacement | 替换哪个完整机制、用什么结构 | 一个当前父本；真实目标 path；path 外 AST 不变 |
| crossover | 两个父本、各自机制及组合方式 | 双亲不同；双方均有真实机制完整保留 |

keep 使用 parent/path/subtree 引用本轮 evidence；程序核对来源和 child 内真实出现次数。允许一方多个机制；根部完整机制可作替换目标。结构指纹排除仅改窗口/常量的提案，交给 Step 6。

LLM 同时读取 delta、pool credit 的方向、大小、具体基线与 coverage。支持强证据利用，也允许有理由的探索或强机制替代；reason 必须非空，解释不代替评价。

## 生成流程

```text
context = 当前父本指标 + 本轮 evidence + retrieve_memory + limits/grids
proposals = LLM(context)
for proposal in proposals[:offspring]:
    核对操作、父本、机制来源、局部替换与保留
    parse；拒绝 seen 或纯参数修改
    记录提案并加入 seen
    evaluate(child)
    若有效：加入 O；验证并保存可选 refine 位置
return O
```

不在评价前调用 LLM 排序或淘汰 child。Step 6 细化部分候选，Step 7 决定最终存活。错误 refine 元数据不丢弃有效 child，而是记录后回退默认窗口细化。

evolution_done 保存提案数量和按操作统计的有效 offspring 数。

## Random crossover

开启时，每个 crossover 使用独立随机选出的双亲机制对。程序预先产生最多 offspring 个随机对；LLM 按 crossover 在响应中的出现顺序消费，不能自行挑选配对。程序核对双方与 keep。组合表达式仍由 LLM 生成。

默认模式 random_pairs=null，配对自由；消融模式无可用双亲时 random_pairs=[]，只允许其他操作。随机机制选择不参考信用大小，但 LLM 仍看到其他操作所需上下文。

## Prompt 与调用

SYSTEM 说明真实算子名称/参数个数、路径语法、有符号指标和证据边界。DIAGNOSE 解释基线设计；EVOLVE 解释双证据冲突、三类操作、记忆转移、随机消融和参数字段，均提供 JSON 结构示例。

ask 记录 system/prompt、可见响应、finish_reason、usage、耗时。max_output_tokens 默认 8192，超时最多 90 秒、SDK 最多一次重试；开始前检查总时间/调用预算。JSON 或请求失败记录后跳过，不增加模型修复循环。
