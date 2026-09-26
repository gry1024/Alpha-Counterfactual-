# Step 2 — Counterfactual Diagnosis

输入：当前池 P、指标及诊断历史。输出：真实机制记录、干预表达式与单因子证据。

代码：AlphaCFTrainer.select_parents/diagnose，expression.walk/at/replace/ablate，prompt.DIAGNOSE。

## 父本与机制

select_parents 从三个队列轮流取因子、去重补足 parents=6：

1. 当前单因子 R 降序。
2. 最近一次 selection 的 marginal 降序。
3. 上次诊断轮次升序，未诊断优先；同分顺序受 seed 控制。

selection 的 marginal 属于当时的部分池，仅作抽样线索，不冒充当前完整池的 leave-one-out 贡献。

每个父本一次模型调用，提供完整公式、真实 AST 路径、指标和预算。mechanisms=3 仅为单父本数量上限，可调整；机制语义没有类型白名单。LLM 可返回更少机制；嵌套机制可独立诊断，贡献不相加。

响应字段：path、description、subtree、mode、replacement、reason。无合理干预时 mode/replacement=null，保留未测量记录。关闭反事实证据时仍识别机制，但完全跳过干预构造。

## T(f,m) 的执行

T(f,m;b)=replace(f,path(m),b)。replace 深复制 AST，只改该路径；每个机制都从原始 f 独立出发。

| mode | LLM 自由度 | 程序验证 |
|---|---|---|
| remove | 选择机制内保留的输入，移除变换/交互/包装 | b 是该机制的真实 featured 后代子树 |
| neutralize | 提出中性元或可解释的上下文基线 | 不引入机制以外的输入特征 |
| null | 无法合理干预 | 留在 evidence/log，不形成实测 memory |

根节点允许合理去包装；全常量因子、无变化、非法历史或不可评价干预被拒绝。双方共用 parse/validate。reason 必须解释移除的作用和保留内容。

ablate 返回 (cf,baseline_kind)：明确加减/乘除中性元标记 identity，保留后代输入标记 input，其他 neutralize 标记 contextual。类型标记描述基线，不代表自动证明语义中性。

TsCorr(x,y,w)→x 测量的是相对 x 输入基线的变化；任意新预测结构属于 evolution，不应被 LLM 作为消融。程序能检查结构和输入，语义合理性仍是有记录、可审阅的假设。

## 状态与覆盖率

有效来源机制进入本轮 evidence，包括 unsupported、invalid、unmeasured；仅成功测量的进入 memory。

diagnosis_coverage 记录 returned、预算内 proposed、identified、structural、measured、pool_measured；measured_fraction=measured/proposed。分母包含无基线和失败的预算内提议；超预算输出只记 returned，不执行。

日志保存每条机制状态、原始提议或失败原因。干预张量测量后释放，除非它本来就是池成员。
