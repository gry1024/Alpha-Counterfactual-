# 04 — 机制记忆

对应代码：`AlphaCFTrainer.memory/diagnose/evolve/save`。

用一个 list[dict] 保存实测干预：factor、path、子树、描述、mode、replacement、ablation、delta_cf、pool_credit、action。未测量提议仍可给当前 parent 演化，但不进入历史实证 memory。

每次演化直接传全部 memory，不做向量库、检索、裁剪或重复 trace 包装。每轮写 memory.json。新结构和新参数不继承已有数值证据。
