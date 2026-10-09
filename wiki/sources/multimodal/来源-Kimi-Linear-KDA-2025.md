# 来源：Kimi Linear / Kimi Delta Attention（KDA）（2025）

> 原始来源：[`raw/articles/transformer_efficiency/2025-Kimi-Linear-KDA-Moonshot.pdf`](../../../raw/articles/transformer_efficiency/2025-Kimi-Linear-KDA-Moonshot.pdf)（arXiv:2510.26692，Moonshot AI Kimi Team）。
>
> 关联概念：[[Transformer效率演进]]。

## 一句话主张

**Kimi Linear** 是一个混合线性注意力架构，其核心 **Kimi Delta Attention（KDA）** 扩展 Gated DeltaNet、用更细粒度门控更有效利用有限的定长状态内存，从而在公平比较下**首次在短/长上下文与 RL 扩展上整体超越全注意力**。

## 机制要点（视频：3 层 KDA : 1 层 MLA 的定长状态混合）

- **KDA**：基于**矩阵值 RNN 状态 $S_t$** 递推，不再计算全 pairwise attention；为每个通道分配独立遗忘速率（细粒度门控 + 逐维衰减），慢衰减通道保留长距离信息、快衰减通道及时释放空间。
- **约束 DPLR 转移矩阵**：专门的 chunkwise 算法在保证 delta rule 一致性的同时大幅降低计算，半精度下无需二次分块即可运行，64k 上比通用 DPLR 快约 2×。
- **混合架构**：3 层 KDA（定长状态承担大部分计算）+ 1 层 Gated MLA（全注意力保留长程精确检索），协同上下文长度扩展（视频标"3 层 KD"）。
- 48B/3B 研究检查点：相同训练配方下**全面超过全 MLA**，KV cache 减少至多 75%，**1M 上下文解码吞吐提升 6×**；RULER 84.3 vs MLA 81.3、MMLU-Pro 51.0 vs 47.2。

## 与既有知识 / 其他来源的连接

- 是 ISAAC/线性注意力血统对 [[来源-Transformers-are-RNNs-线性注意力-2020]] 的"定长状态"路线的现代强化，改进自 Gated DeltaNet。
- 混合里用 [[来源-DeepSeek-V2-MLA-2024]] 的 MLA 作为"全注意力/精确检索"的一端。
- 演进时间线"注意力计算"类别：**KDA · 月之暗面 · 2025**（改动环节 ③④）。

## 冲突与不确定性

- 逐层混合的"KDA:MLA 比例"、逐维门控与四分之一约束对超参与任务分布敏感；具体超参、训练稳定性与性能数值应回 arXiv 2510.26692 原文核验。
