# 来源：MiniMax Sparse Attention（MSA）（2026）

> 原始来源：[`raw/articles/transformer_efficiency/2026-MiniMax-Sparse-Attention-MSA.pdf`](../../../raw/articles/transformer_efficiency/2026-MiniMax-Sparse-Attention-MSA.pdf)（arXiv:2606.13392，MiniMax + 北大/NVIDIA 等）。
>
> 关联概念：[[Transformer效率演进]]。

## 一句话主张

**MSA** 是一个基于 GQA 的**块级稀疏注意力**：用一个轻量索引分支按每个 GQA 组独立挑 top-k 个 KV 块，主分支只对选中块做精确块稀疏注意力，从而在固定每 query 注意力预算下支持 1M 上下文。

## 机制要点（视频：每个 GQA 组各挑 16 块、共 2048 token）

- **两分支**：Index Branch 用单轻量头给 KV 块打分，每个 GQA 组独立选 top-k 块（每组选 top-16 个 128-token 块；最近块必选）；Main Branch 只对选中块算注意力。
- **同读复用**：以 KV 块作外层循环，每个块只读一次，命中它的多个 Q 一起算，提升张量核利用率（exp-free Top-k + KV-outer sparse attention）。
- 训练用 KL loss 让索引分布对齐主分支在选中块上的分布，且索引分支梯度与主分支解耦。
- 效果：109B 参数多模态模型上，MSA 与 GQA 表现相当，同时**1M 上下文每 token 注意力算力降 28.4×**；配自研内核在 H800 上 14.2× prefill、7.6× decode 墙钟加速。已用于开源 MiniMax M3（428B MoE、≥512K 上下文）。

## 与既有知识 / 其他来源的连接

- 与 [[来源-MoBA-Mixture-of-Block-Attention-2025]]、[[来源-NSA-原生稀疏注意力-2025]] 同属"块级稀疏 + 先挑块再精算"谱系，且明确与 GQA 融合并给出生产级内核与开源权重。
- 与 [[来源-DeepSeek-V4-CSA-HCA-2026]] 并列为 2026 年"1M 上下文 + KV 压缩/稀疏"工业落地样本。

## 冲突与不确定性

- 索引分支的 top-k 打分是近似启发，跨块长程依赖与极端长尾任务上的保真需独立核验；28.4×/14.2×/7.6× 等数值为特定硬件（H800）与配置下测量，应回原文核验。
