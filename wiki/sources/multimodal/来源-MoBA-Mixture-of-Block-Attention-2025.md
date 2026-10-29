# 来源：MoBA —— Mixture of Block Attention（2025）

> 原始来源：[`raw/articles/transformer_efficiency/2025-MoBA-Mixture-of-Block-Attention-Moonshot.pdf`](../../../raw/articles/transformer_efficiency/2025-MoBA-Mixture-of-Block-Attention-Moonshot.pdf)（arXiv:2502.13189，Moonshot AI + 清华/浙大）。
>
> 关联概念：[[Transformer效率演进]]。

## 一句话主张

把 **MoE 的"路由 + top-k 稀疏激活"思想应用到注意力**：MoBA 把输入分成块，用可学习门控按 query 动态挑选相关块，实现"块级稀疏、先挑块再精算"，同时能在全量与稀疏注意力之间平滑切换。

## 机制要点

- **块级路由**：分块后，块内 K 取均值打分、选 top-k 块（"块内 K 取均值打分，选 top-k 块"），被选块内的 token 才参与 attention。
- **"less structure" 原则**：不预设 sink/window 等强偏置结构，让模型自主决定关注哪——区分于强结构稀疏。
- 参数数与自注意力**完全相同**（无架构膨胀），是可无缝 drop-in 的全量↔稀疏注意力，已上线支撑 Kimi 长上下文。
- 沿用因果掩码保证自回归正确性（MoBA 也讨论如何精确保留/松弛因果性）。

## 与既有知识 / 其他来源的连接

- 思想源头是 [[来源-Sparse-Gated-MoE-2017]]（"参数 × E，算力只 × k"从 FFN 搬到注意力）。
- 与 [[来源-NSA-原生稀疏注意力-2025]] 同日在演进时间线并列为"MoBA // NSA：同一天，两家都只算一部分块 · 块级稀疏：先挑块，再精算"。

## 冲突与不确定性

- 门控打分是启发/近似，跨块长程依赖与复杂推理上的保真度需独立核验；不同分块粒度、top-k 与因果性处理对质量影响大。
