# 来源：GLoRe 全局-局部联合修正

- **原始路径**：
  - [`icml_2024_GLoRe_When,_Where,_and_How_to_Improve_LLM_Reasoning_via_Global_and_Local_Refinements.md`](../../raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/icml_2024_GLoRe_When,_Where,_and_How_to_Improve_LLM_Reasoning_via_Global_and_Local_Refinements.md)
- **类型**：ICML 2024 方法论文（自修正）

## 一句话摘要

GLoRe 将推理修正拆为“何时修（when）—哪里错（where）—如何改（how）”，并通过 ORM/SORM + 全局/局部修正模型协同，提高复杂推理任务上的自纠错效果。

## 关键主张（基于摘要与正文）

- ORM 适合判断“最终是否可接受”，但用于中间步骤定位时易过度悲观。
- SORM 通过合成数据近似 step-level 价值判断，在错误定位上优于普通 ORM。
- 全局修正（重写整条解）与局部修正（定位后改写）互补，组合后优于单一路径。
- 在 GSM8K 上，论文报告可把 13B 级模型贪心采样表现从约 53% 提升到约 65%。

## 不确定性与边界

- 方案依赖大量合成标签与验证采样，训练/推理管线较重。
- SORM 在“最终答案打分”上并不总优于 ORM，二者职责需分工而非互换。

## 织入概念

- [[LLM推理结构与自纠错框架]]
- [[推理评测污染与捷径风险]]