# 来源：Buffer of Thoughts（BoT）

- **原始路径**：
  - [`nips_2024_Buffer_of_Thoughts_Thought-Augmented_Reasoning_with_Large_Language_Models.md`](../../raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_Buffer_of_Thoughts_Thought-Augmented_Reasoning_with_Large_Language_Models.md)
- **类型**：NeurIPS 2024 方法论文（模板检索式推理）

## 一句话摘要

BoT 通过“高层思维模板库（meta-buffer）+ 任务蒸馏 + 模板实例化 + 动态更新”实现低成本可复用推理，试图在准确率、效率与稳定性之间取得更好平衡。

## 关键主张（基于摘要与正文）

- 与单次提示相比，BoT 通过可复用 thought-template 降低每题从零构造推理结构的成本。
- 与 ToT/GoT 等多查询方法相比，BoT 在多个任务上报告更高或可比准确率，并显著降低推理开销。
- 提出 buffer-manager，把已解问题提炼为可迁移模板，形成“越用越强”的经验库。
- 在小模型场景下，BoT 有机会缩小与大模型的推理性能差距。

## 不确定性与边界

- 模板检索与实例化质量依赖于模板库覆盖度；新型任务可能触发失配。
- 模板更新策略若不当，可能引入冗余模板或错误归纳。

## 织入概念

- [[LLM推理结构与自纠错框架]]
- [[Agent任务分流：工作流与智能体边界]]