# 来源：Transformers are RNNs —— 线性注意力（2020）

> 原始来源：[`raw/articles/transformer_efficiency/2020-Transformers-are-RNNs-Linear-Attention.pdf`](../../../raw/articles/transformer_efficiency/2020-Transformers-are-RNNs-Linear-Attention.pdf)（arXiv:2006.16236，Idiap & EPFL，Katharopoulos 等）。
>
> 关联概念：[[Transformer效率演进]]。

## 一句话主张

用核特征映射 + 矩阵结合律把自注意力复杂度从 $O(N^2)$ 降到 $O(N)$，并揭示：线性注意力的 Transformer 在数学上**等价于一个 RNN**——因此可以像序列模型一样快速自回归推理。

## 机制要点（关键：把"打分"从 O(n²) 分数矩阵改造成定长状态）

- 用核函数 $k(x,y)=\phi(x)^\top\phi(y)$ 近似 `softmax`，把 $QK^\top V$ 重组为 $\phi(Q)(\phi(K)^\top V)$，后者可迭代累积成**定长状态 $S_t$**，即"n×n 分数矩阵 → S_t 定长状态，O(n²)→O(n)"。
- 自回归预测可写成 RNN 般的递推，长序列上比普通 Transformer 快数千倍。
- 对应演进时间线"注意力计算"类（改动环节 ③④）里的**线性注意力**项，标注 Idiap·EPFL·2020。

## 与既有知识 / 其他来源的连接

- 是后续定长状态/线性注意力血统（如 [[来源-Kimi-Linear-KDA-2025]] 的 KDA、Gated DeltaNet）的直接思想源头。
- 视频时间线将其记作"2020.06 线性注意力 · Transformers are RNNs"，并预告 2024.12 NVIDIA Gated DeltaNet 接续。

## 冲突与不确定性

- 朴素核线性逼近在复杂推理/长程检索上精度不足，是它未全面取代 softmax 的主因；后续以"门控 + 逐维衰减"（如 KDA）来补表现。
