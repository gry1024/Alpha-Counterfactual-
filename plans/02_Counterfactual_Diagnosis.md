# 02 — 反事实诊断

对应代码：`AlphaCFTrainer.select_parents/diagnose；expression.ablate；PROMPT_DIAGNOSIS`。

每轮默认选 20 个 parent：一半优先访问次数少，一半优先高 R。每个 parent 请求最多 3 个有意义机制及局部基线。任务模板包含路径约定、金融含义、有效干预示例与自检要求，公共算子表按实际实现定义。程序按 AST path 读取子树，remove 必须保留其后代，neutralize 不引入新输入；只替换该路径。

无法合理消融可返回 null；成功消融计算 delta_cf。每个干预从原 parent 开始，机制效果不可相加。日志保存提议、有效测量和失败原因。
