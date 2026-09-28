# 05 — 宏观演化

对应代码：`AlphaCFTrainer.ask/evolve/train；PROMPT_EVOLUTION；utils.llm.OpenAIModel`。

全部 parent 先完成诊断，再逐个生成子代；默认 20 个 parent，每个请求 5 个 offspring。每次给当前 parent、一个其他 donor 和完整历史 memory。默认 donor 取机制双重证据最强的其他 parent，随机交叉消融改为随机 donor。

模型用 mutation、replacement、crossover 输出等长的 expressions、operations、explanations 数组。模板复用 utils 的组织方式，明确每种操作的保留/改变要求、双重证据、差异化假设和语法自检；同时提供当前池及本轮已有子代避免重复。程序直接解析、检查合法性并评价。不维护多层结构契约过滤，精确表达式去重，实际存活由统一池选择决定。调用直接复用 utils.llm.OpenAIModel.chat_generate；JSON 直接解析或提取代码块，不另做会话修复循环；整轮没有可用子代直接报错，保留旧快照。
