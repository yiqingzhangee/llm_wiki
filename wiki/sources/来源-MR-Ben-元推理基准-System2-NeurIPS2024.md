# 来源：MR-Ben 元推理基准（System-2）

- **原始路径**：
  - [`nips_2024_MR-Ben_A_Meta-Reasoning_Benchmark_for_Evaluating_System-2_Thinking_in_LLMs.md`](../../raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_MR-Ben_A_Meta-Reasoning_Benchmark_for_Evaluating_System-2_Thinking_in_LLMs.md)
- **类型**：NeurIPS 2024 基准论文（过程评测）

## 一句话摘要

MR-Ben 将评测重点从“最终答案是否正确”扩展到“能否定位并解释推理错误”，用于衡量 LLM 的元推理与 System-2 慢思考能力。

## 关键主张（基于摘要与正文）

- 仅看 outcome accuracy 会掩盖推理链中的错误步骤与脆弱性。
- MR-Ben 覆盖数学、医学、生物、物理、化学、逻辑、编程等多领域，共约 6k 标注样本。
- 基准采用“错误定位 + 错因解释 + 纠正建议”的过程导向评测，能更好区分模型在反思与审校能力上的差异。
- 结果显示部分模型在常规问答上较强，但在元推理任务上明显掉队，提示训练与推理范式存在结构性短板。

## 不确定性与边界

- 误因解释指标含一定自动评审成分（文中使用 GPT-4 辅助），与人工评审口径仍可能存在偏差。
- 基准强调“评估能力”，并不直接等价于“提升能力”的训练方案。

## 织入概念

- [[推理评测污染与捷径风险]]
- [[LLM推理结构与自纠错框架]]