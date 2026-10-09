# 来源：Sparse Gated MoE —— Outrageously Large Neural Networks（2017）

> 原始来源：[`raw/articles/transformer_efficiency/2017-Sparse-Gated-MoE-Outrageously-Large-NN.pdf`](../../../raw/articles/transformer_efficiency/2017-Sparse-Gated-MoE-Outrageously-Large-NN.pdf)（arXiv:1701.06538，Google Brain）。
>
> 关联概念：[[Transformer效率演进]]。

## 一句话主张

用"稀疏门控混合专家（Sparse Gated MoE）+ 可学习门控 + 负载均衡"把模型容量（参数）与单样本计算（FLOPs）解耦：同一时刻只激活少量专家。

## 机制要点

- 每个输入 token 由可学习门控路由到 top-k 个专家（k 远小于专家总数 E），实现"参数 × E，算力只 × k"。
- **负载均衡损失（load balancing）** 缓解 token 倾向集中在少数专家的问题，这是后续所有 MoE 路由训练的地基。
- 把原本处于前馈（FFN）环节"换成一组专家"——对应 Transformer 效率演进里"改动环节 ⑦"。

## 与既有知识 / 其他来源的连接

- 是 [[来源-MoBA-Mixture-of-Block-Attention-2025]] 把 MoE 思想搬到注意力上的直接思想来源（MoBA"应用 MoE 原则到 attention"）。
- 与 [[来源-Transformer-Attention-Is-All-You-Need-2017]] 同列演进时间线"前馈网络与整体结构"类。

## 冲突与不确定性

- 稀疏路由的负载均衡与通信开销、专家负载不均是长期工程难点；"参数多算力少"不等于"训练成本低"。
