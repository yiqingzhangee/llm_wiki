# 来源：A Neuro-Symbolic Benchmark Suite for Concept Quality and Reasoning Shortcuts（rsbench）

- **原始路径**：
  - [`nips_2024_A_Benchmark_Suite_for_Systematically_Evaluating_Reasoning_Shortcuts.md`](../../raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_A_Benchmark_Suite_for_Systematically_Evaluating_Reasoning_Shortcuts.md)
- **类型**：基准套件（Neuro-Symbolic 概念质量与推理捷径）

## 一句话摘要

该工作提出 `rsbench`，系统评估“模型用错误概念也能答对任务”的推理捷径问题，覆盖 NeSy、CBM、黑盒 NN/CLIP，并提供 OOD 数据生成与形式化计数工具 `countrss`。

## 关键主张（基于摘要与正文前半）

- 高标签准确率不等于高概念质量；在学习+推理任务中，模型可通过**错误语义映射**走捷径。
- 推理捷径与知识结构、训练数据覆盖、损失设计和结构偏置共同相关。
- `rsbench` 提供任务级与模型级指标（如概念混淆、collapse）及自动化验证流程，可在训练前评估任务是否易产生捷径。

## 不确定性与边界

- `BDD-OIA` 等复杂任务中，形式化计数与工程复现仍有计算开销与实现差异。
- 部分结论依赖具体模型实现（DPL/LTN/CBM/NN/CLIP）与提取器设置，跨实现可比性需谨慎。

## 织入概念

- [[推理评测污染与捷径风险]]
- [[短视频内容审核与多模态治理]]（高风险场景中的“看似正确但概念错配”问题）
