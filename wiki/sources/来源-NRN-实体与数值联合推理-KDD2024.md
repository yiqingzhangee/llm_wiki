# 来源：NRN 实体与数值联合推理

- **原始路径**：
  - [`kdd_2024_Knowledge_Graph_Reasoning_over_Entities_and_Numerical_Values.md`](../../raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/kdd_2024_Knowledge_Graph_Reasoning_over_Entities_and_Numerical_Values.md)
- **类型**：KDD 2024 任务+方法论文（Numerical CQA）

## 一句话摘要

NRN 提出 Numerical CQA 任务并将实体与数值分开编码，通过属性投影、数值投影和反向属性投影实现“关系 + 数值约束”的联合推理。

## 关键主张（基于摘要与正文）

- 传统 CQA 把数值当普通实体处理，难以表达比较/运算等数值语义。
- 论文定义 Numerical CQA，补充数值变量与数值关系算子，扩展复杂查询表达能力。
- NRN 在计算图中交替执行实体编码相位与数值编码相位，针对不同对象采用不同表示结构。
- 在 Freebase/DBpedia/YAGO 构建的新基准上，NRN 可稳定提升多种 query encoding backbone。

## 不确定性与边界

- 双相位编码与额外投影算子增加建模复杂度与训练成本。
- 对数值分布建模质量较依赖属性类型与数据稀疏度。

## 织入概念

- [[LLM推理结构与自纠错框架]]
- [[推理评测污染与捷径风险]]