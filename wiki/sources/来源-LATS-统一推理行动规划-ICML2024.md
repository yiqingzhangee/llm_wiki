# 来源：Language Agent Tree Search（LATS）

- **原始路径**：
  - [`icml_2024_Language_Agent_Tree_Search_Unifies_Reasoning,_Acting,_and_Planning_in_Language_Models.md`](../../raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/icml_2024_Language_Agent_Tree_Search_Unifies_Reasoning,_Acting,_and_Planning_in_Language_Models.md)
- **类型**：ICML 2024 方法论文（搜索式 Agent）

## 一句话摘要

LATS 将语言模型的推理（reasoning）、行动（acting）与规划（planning）统一到 MCTS 框架下，通过环境反馈 + 自反思实现更稳健的多步决策。

## 关键主张（基于摘要与正文）

- 传统 ReAct/CoT/ToT 往往只覆盖局部能力，难以在复杂交互环境中兼顾探索与决策质量。
- LATS 把 LM 同时用作策略生成器、价值评估器与反思器，在不额外训练参数的条件下提升任务性能。
- 在 HotPotQA、HumanEval、WebShop 等任务中，LATS 相对基线表现出更优的 pass@1 或成功率。

## 不确定性与边界

- MCTS 搜索深度与分支数带来显著 token/时延成本。
- 结果对环境反馈质量较敏感；若反馈噪声高，价值估计会被扰动。

## 织入概念

- [[LLM推理结构与自纠错框架]]
- [[Agent任务分流：工作流与智能体边界]]
