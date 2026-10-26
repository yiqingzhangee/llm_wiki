# 来源：Qwen3-VL 事件序列编码器调研（2026）

> 调研日期：2026-10-26
>
> 对应选型：[[Qwen3-VL事件序列编码器选型-2026]]
>
> 关联概念：[[Qwen异构特征接入与事件序列建模]]、[[Qwen3-VL统计特征编码器选型-2026]]

## 调研问题

目标是把约 8 步、带事件类型、固定英文描述、时间戳和可选动态属性的事件序列编码为连续向量，并在需要时进一步转换为 Qwen3-VL-4B soft token。核心问题是：什么方案能以最低工程成本保留顺序和不规则时间间隔。

## 核心裁决

**最简单的合理连续编码器是一层 GRU。** 每个事件先融合缓存的描述向量、可训练事件 ID、相邻时间间隔、距预测时点时间和可选 payload；GRU 最后一个有效 hidden state 作为 `sequence_vector`。只有接入 Qwen 时才增加独立 projector，首版只生成一个 2560 维 soft token。

确定性文本模板是不开发自定义编码器时的绝对最低成本基线。固定长度展开后接 MLP 虽然代码短，但对 padding、变长序列和顺序泛化更脆弱，不作为连续序列编码器首选。

## 证据矩阵

| 路线 | 外部锚点 | 可迁移结论 | 边界 |
|---|---|---|---|
| 单层 GRU | GRU4Rec，arXiv:1511.06939 | GRU 是成熟、低成本的短行为序列编码器 | 推荐任务不等于风控分类 |
| 时间感知注意力 | SASRec；TiSASRec，WSDM 2020 | 顺序位置与真实时间间隔应分开编码 | 对约 8 步序列未证明优于 GRU |
| 连续时间事件模型 | Transformer Hawkes Process，arXiv:2002.09291 | 可联合建模事件类型和发生时间 | 若不预测下一事件时间，TPP 目标不是必要组件 |
| 文本语义增强 | UniSRec、RecFormer、LLM-ESR | 固定描述可离线编码，再以轻量 adaptor 融入序列模型 | 语义相似不代表业务作用相同，仍需事件 ID residual |
| 长序列架构 | Mamba4Rec、HSTU | 长历史下有状态空间或定制 attention 的效率动机 | 当前八步序列没有天然收益 |

## 为什么不是 Transformer 或 Mamba

- 当前序列约 8 步，二次复杂度和长程记忆不是主要瓶颈；
- GRU 原生处理顺序、padding 和变长序列，不需要位置编码或 attention mask 设计；
- Transformer、Mamba、TPP 和 learned-query pooling 会增加实现、调参和归因成本；
- 若单层 GRU 未超过事件文本模板、聚合统计或直接分类基线，更复杂编码器没有升级依据。

## 主要来源

- [GRU4Rec](https://arxiv.org/abs/1511.06939)
- [SASRec](https://arxiv.org/abs/1808.09781)
- [TiSASRec](https://doi.org/10.1145/3336191.3371786)
- [Transformer Hawkes Process](https://arxiv.org/abs/2002.09291)
- [UniSRec](https://arxiv.org/abs/2206.05941)
- [RecFormer](https://arxiv.org/abs/2305.13731)
- [LLM-ESR](https://arxiv.org/abs/2405.20646)
- [Mamba4Rec](https://arxiv.org/abs/2403.03900)
- [HSTU](https://arxiv.org/abs/2402.17152)
- [SOHET](https://arxiv.org/abs/2606.21356)

## 证据边界

没有公开 benchmark 直接比较这些编码器作为 Qwen3-VL soft-token 前端的表现。“单层 GRU 最优”不是性能结论，而是当前短序列、低成本验证目标下的工程决策。
