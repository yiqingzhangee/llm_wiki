# 来源：DS-Agent（Case-Based Reasoning 驱动的自动化数据科学）

- **原始路径**：
  - [`icml_2024_DS-Agent_Automated_Data_Science_by_Empowering_Large_Language_Models_with_Case-Based_Reasoning.md`](../../raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/icml_2024_DS-Agent_Automated_Data_Science_by_Empowering_Large_Language_Models_with_Case-Based_Reasoning.md)
- **类型**：ICML 2024 方法论文（Agent + CBR）

## 一句话摘要

DS-Agent 将案例推理（CBR）的“检索-复用-评估-修订-保留”流程嵌入 LLM Agent，以更低训练成本在自动化数据科学任务上获得更高完成率与更低单次成本。

## 关键主张（基于摘要与正文）

- 仅靠通用 Agent 在自动化建模场景容易出现计划不合理与幻觉，导致任务完成率低。
- 引入 CBR 后，Agent 可复用 Kaggle 等专家案例，并依据执行反馈进行迭代修订，形成稳定的性能提升曲线。
- DS-Agent 将“开发阶段（可迭代）”与“部署阶段（低资源适配）”解耦：前者积累可复用案例，后者通过案例迁移降低对强基座模型依赖。

## 不确定性与边界

- 结果依赖任务分布与案例库质量；跨领域迁移时可能出现“检索到相似但误导”的风险。
- 论文中的成本/成功率对比受具体模型版本与实验预算影响，工程复现需重建同口径基线。

## 织入概念

- [[LLM推理结构与自纠错框架]]
- [[Agent框架选型与工程维度]]
