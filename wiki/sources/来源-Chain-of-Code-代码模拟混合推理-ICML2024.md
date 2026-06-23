# 来源：Chain of Code（CoC）

- **原始路径**：
  - [`icml_2024_Chain_of_Code_Reasoning_with_a_Language_Model-Augmented_Code_Emulator.md`](../../raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/icml_2024_Chain_of_Code_Reasoning_with_a_Language_Model-Augmented_Code_Emulator.md)
- **类型**：ICML 2024 方法论文（代码推理）

## 一句话摘要

Chain of Code 通过“可执行代码 + LM 模拟器（LMulator）”的交织执行，把符号计算精确性与语义推理灵活性结合起来，显著提升复杂推理任务表现。

## 关键主张（基于摘要与实验）

- 纯 CoT 在数值/符号任务易失误，纯可执行代码又难覆盖语义子任务；CoC 通过“能执行则执行，不能执行则模拟”补齐两端短板。
- 在 BIG-Bench Hard 上，CoC 相对 CoT 有明显提升，且在算法子任务上优势更大。
- 方法保持可解释的“程序状态轨迹”，便于后验分析与调试。

## 不确定性与边界

- 依赖代码生成质量；错误的中间状态会被后续步骤放大。
- 对执行器、沙箱与异常处理策略有工程要求，接入成本高于纯提示法。

## 织入概念

- [[LLM推理结构与自纠错框架]]
- [[Agent框架选型与工程维度]]
