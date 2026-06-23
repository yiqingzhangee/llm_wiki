# 来源：Thought Rollback（TR）

- **原始路径**：
  - [`icml_2024_Toward_Adaptive_Reasoning_in_Large_Language_Models_with_Thought_Rollback.md`](../../raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/icml_2024_Toward_Adaptive_Reasoning_in_Large_Language_Models_with_Thought_Rollback.md)
- **类型**：ICML 2024 方法论文（自适应回滚）

## 一句话摘要

Thought Rollback 通过“出错分析→回滚到早期思路→重走新路径”的循环机制，让 LLM 从单向推理升级为可修正的自适应推理图，从而提升复杂问题求解率。

## 关键主张（基于摘要与正文）

- 链式/树式前向推理都易出现错误传播；TR 允许回滚到历史节点并带着错误经验继续推理，缓解级联失误。
- TR 不依赖任务特定人工样例，可在零样本或少样本条件下构建“可回溯”推理轨迹。
- 在数学与多任务推理基准上，TR 相对若干前向结构方法取得更高解题率，并在成本上保持可控。

## 不确定性与边界

- 回滚策略与停止条件会影响效率与稳定性，参数不当会产生冗余探索。
- 错误分析本身也可能包含幻觉，需要外部验证机制托底。

## 织入概念

- [[LLM推理结构与自纠错框架]]
- [[推理评测污染与捷径风险]]
