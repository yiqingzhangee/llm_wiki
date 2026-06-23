# 来源：SELF-DISCOVER（自组装推理结构）

- **原始路径**：
  - [`nips_2024_SELF-DISCOVER_Large_Language_Models_Self-Compose_Reasoning_Structures.md`](../../raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_SELF-DISCOVER_Large_Language_Models_Self-Compose_Reasoning_Structures.md)
- **类型**：NeurIPS 2024 方法论文（结构化元推理）

## 一句话摘要

SELF-DISCOVER 让 LLM 先在任务层面“选择-改写-实现”推理模块，自动生成任务专属推理结构，再在实例层按结构填充求解，以较低推理成本获得稳定增益。

## 关键主张（基于摘要与实验）

- 单一 CoT 模块并非所有任务最优；不同任务对应不同内在推理结构。
- 通过 task-level 的结构发现（而非 instance-level 大量采样），可以在 BBH、T4D、MATH 等任务上同时获得性能与效率收益。
- 发现的结构可跨模型家族迁移，具备一定通用性与可解释性。

## 不确定性与边界

- 对“模块库”质量与覆盖度有依赖；模块过窄会限制结构发现上限。
- 任务定义不清晰时，自动发现结构可能退化为模板化输出。

## 织入概念

- [[LLM推理结构与自纠错框架]]
- [[Agent任务分流：工作流与智能体边界]]
