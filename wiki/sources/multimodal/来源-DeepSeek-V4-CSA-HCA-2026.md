# 来源：DeepSeek-V4 —— Compressed Sparse Attention（CSA）+ Heavily Compressed Attention（HCA）（2026）

> 原始来源：[`raw/articles/transformer_efficiency/2026-DeepSeek-V4-CSA-HCA.pdf`](../../../raw/articles/transformer_efficiency/2026-DeepSeek-V4-CSA-HCA.pdf)（arXiv:2606.19348，DeepSeek-AI）。
>
> 关联概念：[[Transformer效率演进]]。

## 一句话主张

DeepSeek-V4 用 **CSA（Compressed Sparse Attention）+ HCA（Heavily Compressed Attention）的混合注意力**，配合 mHC 超连接与 Muon 优化器，在 1M 上下文下把单 token 推理 FLOPs 降到 V3.2 的 27%、KV cache 降到 10%，实现低成本百万 token 长上下文。

## 机制要点（视频：4合1再筛选 · 128合1全看）

- **CSA**：沿序列把约 4 个 token 合成 1 条 KV，再稀疏选择——"每 4 个合成 1 条 → 挑 top-k"，兼顾局部精度。
- **HCA**：以更高压缩比（约 128 token 合 1 条 KV）做**全量稠密**关注——"每 128 个合成 1 条 → 全看"，提供全局上下文视野。
- 再**拼上最近 128 个原始 token**，保留近期细节。混合层设计：V4-Pro 前十层用 HCA，其后 CSA/HCA 交替。
- 整体效果：KV cache 内存相对标准 Transformer 降到约 2% 的量级，同时维持 1M 上下文。
- 模型规模：V4-Pro 1.6T（49B 激活）、V4-Flash 284B（13B 激活），均支持 1M 上下文，>32T token 预训练。

## 与既有知识 / 其他来源的连接

- 延续 [[来源-DeepSeek-V2-MLA-2024]] 的 KV 压缩思路，从"无损低秩压缩 KV"推进到"块级压缩 + 稀疏 + 全量低比特"。
- 灵感源：视频标注 CED 灵感来自 YOCO（微软研究院 & 清华，2024）；log 中 CSA2/HCA 还关联 CED（见时间线末尾 2026.09 CSA2+CED 项）。
- 与 [[来源-MiniMax-Sparse-Attention-MSA-2026]] 同为 2026 年"1M 上下文 + 压缩/稀疏 KV"的工业落地样本。

## 冲突与不确定性

- V4 仍为 preview/技术报告口径；1M 上下文上的长程精确检索、FP4 量化 KV 的保真、不同任务的收益差异需回原文与后续版本核验。
