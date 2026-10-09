# 来源：Attention Is All You Need —— Transformer 架构（2017）

> 原始来源：[`raw/articles/transformer_efficiency/2017-Attention-Is-All-You-Need-Transformer.pdf`](../../../raw/articles/transformer_efficiency/2017-Attention-Is-All-You-Need-Transformer.pdf)（arXiv:1706.03762，Google Brain & Google Research）。
>
> 关联概念：[[Transformer效率演进]]、[[自回归掩码、解码与生成评测]]。

## 一句话主张

把"注意力"作为唯一主要机制，抛弃循环/卷积，提出可并行的编码器-解码器 Transformer，在多项翻译基准上超过当时 SOTA。

## 机制要点

- **Self-attention**：$Q,K,V$ 由输入线性投影得到，注意力权重 $=\mathrm{softmax}(QK^\top/\sqrt{d_k})$；多头让不同位置可共同关注不同表示子空间。
- **三个计算复杂度对照**（n=序列长、d=维数）：self-attention $O(n^2 d)$、recurrent $O(n d^2)$、convolutional $O(k n d^2)$。自注意力的瓶颈即随序列长的平方增长——这是后来"效率演进"要解决的起点。
- **位置编码**（正弦 PE）、残差、LayerNorm（Post-LN 原始写法）构成可并行堆叠的块。
- 以 8 个 P100 GPU、12h 训练达到当时机器翻译 SOTA。

## 与既有知识的连接

- 本页是 [[Transformer效率演进]] 的时间线**起点**（2017）。
- 视频"55 秒看完 Transformer 效率演进时间线"将其作为第 1 项经典技术。

## 冲突与不确定性

- 后续高效注意力（线性/稀疏）都以克服本页 self-attention 的 $O(n^2)$ 为目标；"attention is all you need"指架构主导权，不代表全量 attention 是最优的。
