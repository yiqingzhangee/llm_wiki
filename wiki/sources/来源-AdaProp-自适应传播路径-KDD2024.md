# 来源：AdaProp 自适应传播路径

- **原始路径**：
  - [`kdd_2024_AdaProp_Learning_Adaptive_Propagation_for_Graph_Neural_Network_based_Knowledge_Graph_Reasoning.md`](../../raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/kdd_2024_AdaProp_Learning_Adaptive_Propagation_for_Graph_Neural_Network_based_Knowledge_Graph_Reasoning.md)
- **类型**：KDD 2024 方法论文（KG 推理）

## 一句话摘要

AdaProp 为 KG 推理中的 GNN 传播路径引入“可学习的自适应采样”，在控制计算量的同时尽量保留与查询语义相关的关键实体。

## 关键主张（基于摘要与正文）

- 既有传播策略（全图/渐进/受限）在规模、语义相关性或覆盖率上存在明显折中。
- AdaProp 采用增量采样保持层间连通，并用 relation-aware 的学习分布选择候选实体。
- 通过 Gumbel top-k 与直通估计器联合训练采样器和推理模型，实现查询相关的动态路径。
- 在传导与归纳设定下，论文报告了精度与效率的综合提升。

## 不确定性与边界

- 采样参数与温度等超参数会影响“保留目标实体”与“降复杂度”的平衡。
- 方法主要针对 KG 场景，迁移到开放文本推理需额外结构化层。

## 织入概念

- [[LLM推理结构与自纠错框架]]
- [[Agent任务分流：工作流与智能体边界]]