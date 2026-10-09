# 来源：Universal Transformers（2018）

> 原始来源：[`raw/articles/transformer_efficiency/2018-Universal-Transformers.pdf`](../../../raw/articles/transformer_efficiency/2018-Universal-Transformers.pdf)（arXiv:1807.03819，Google Brain & DeepMind）。
>
> 关联概念：[[Transformer效率演进]]。

## 一句话主张

在 Transformer 上引入**权重共享的循环机制**：同一组参数在时间步上反复作用（T 步），用固定参数实现"深度 ×T"，即"一份权重、同一层循环 T 次，深度变 T 倍、参数不变"。

## 机制要点

- 每层参数在时间维上共享（weight sharing），并串接 **ACT（Adaptive Computation Time）** 动态决定每个 token 的循环步数，从而获得"亚符号级"的序列加工能力。
- 在算法式任务（bAbI、合成排序、逻辑推理）上表现出优于普通 Transformer 的泛化与表示能力。
- 对应演进时间线"前馈网络与整体结构"类里的"权重共享循环"项。

## 与既有知识 / 其他来源的连接

- 与 [[来源-Transformers-are-RNNs-线性注意力-2020]] 直觉上呼应"Transformer 与循环可互相转化"，但实现路径不同（前者权重共享 + ACT，后者把注意力重写为线性/RNN 状态递推）。
- 是 [[Transformer效率演进]] 中"整体结构/前馈"类别的 2018 里程碑。

## 冲突与不确定性

- 参数共享循环在超大模型/长上下文上的可扩展性有限，现代大模型并未普遍采用；它更多是"用固定权重换深度/能力"的早期探索。
