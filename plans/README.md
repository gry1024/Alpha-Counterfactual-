# 实现索引

以 idea.md 的八步为准；pool、trainer、prompt、AST 小工具和入口各司其职。

- [初始化](01_Initialize.md)：train_cf.load_data；AlphaCFTrainer.initialize；AlphaCFPool.evaluate/select。
- [反事实诊断](02_Counterfactual_Diagnosis.md)：AlphaCFTrainer.select_parents/diagnose；expression.ablate；PROMPT_DIAGNOSIS。
- [机制贡献](03_Mechanism_Credit.md)：AlphaCFTrainer.diagnose；AlphaCFPool.utility/score_signals。
- [机制记忆](04_Mechanism_Memory.md)：AlphaCFTrainer.memory/diagnose/evolve/save。
- [宏观演化](05_Macro_Evolution.md)：AlphaCFTrainer.evolve/train；PROMPT_EVOLUTION。
- [子代细化](06_Offspring_Refinement.md)：AlphaCFTrainer.refine；expression.parameter_variants。
- [工作池更新](07_Pool_Update.md)：AlphaCFPool.select/update/keep；AlphaCFTrainer.train/save。
- [最终选择与测试](08_Final_Selection.md)：train_cf.train/test；原 run_adaptive_combination.py。

实现复用 alphagen、StockData 和原有完整组合回测脚本；只重写 alpha_cf 与入口，不改其他算法基础设施。
