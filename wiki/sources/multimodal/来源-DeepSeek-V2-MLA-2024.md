# 来源：DeepSeek-V2 / Multi-head Latent Attention（MLA）（2024）

> 原始来源：[`raw/articles/transformer_efficiency/2024-DeepSeek-V2-MLA.pdf`](../../../raw/articles/transformer_efficiency/2024-DeepSeek-V2-MLA.pdf)（arXiv:2405.04434，DeepSeek-AI）。
>
> 关联概念：[[Transformer效率演进]]。

## 一句话主张

**Multi-head Latent Attention（MLA）** 用低秩联合压缩把整个 KV cache 压进一个隐向量 $c_t$，在降 KV 内存的同时反而比标准 MHA 更强，成为 DeepSeek-V2 高效推断的核心。

## 机制要点

- **KV 联合压缩为隐向量**：每 token 只存 $c_t$（512+64 维，对应视频里"每 token 只存 cₜ：512+64 维"），推理时再按需低秩还原 K、V——"算得多、读得少"，不必逐条重建 K、V。
- 结合 **DeepSeekMoE** 稀疏计算，训练成本经济和推断高效：对比 DeepSeek 67B，节省 42.5% 训练成本、**KV cache 减 93.3%**、最大生成吞吐提到 5.76×。
- 模型规模：236B 总参数 / 21B 激活 / 128K 上下文。
- 在演进时间线中属"KV 缓存与投影（改动环节 ②）"类别，2024.05。

## 与既有知识 / 其他来源的连接

- 是 GQA/MQA 之后的 KV 压缩一脉（前身 MQA 2019、GQA 2023），区别于做近似/稀疏，MLA 用低秩隐向量**无损式压缩 + 还原**。
- 为 [[来源-Kimi-Linear-KDA-2025]] 的混合（KDA+MLA 逐层混合）提供"全注意力/精确检索"的一侧。
- 与 [[来源-DeepSeek-V4-CSA-HCA-2026]] 同属 DeepSeek 迭代链：前者压 KV，后者进一步块级压缩 + 稀疏（CSA/HCA）。

## 冲突与不确定性

- 低秩压缩 + 逐 token 隐向量仍需足够内存带宽来还原 K、V；其硬件友好性与长期上下文上的精度需独立核验。
