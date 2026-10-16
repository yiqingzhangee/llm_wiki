# 来源：Decoder-Only 用户表征注意力掩码（2026）

> 原始来源：[`raw/transcripts/LLM-based User Representation Foundation Models/paper_zh_14_Decoder_Only_LLM_Masking.md`](../../../raw/transcripts/LLM-based%20User%20Representation%20Foundation%20Models/paper_zh_14_Decoder_Only_LLM_Masking.md)。对应论文：*How Do Decoder-Only LLMs Perceive Users? Rethinking Attention Masking for User Representation Learning*，arXiv:2602.10622v1。
>
> 关联概念：[[LLM用户表征基础模型与场景适配]]、[[自回归掩码、解码与生成评测]]。

## 核心结论

当 decoder-only LLM 被改造成用户行为编码器时，causal、hybrid 与 bidirectional mask 是**表征训练结构**的选择，不是解码策略或通用模型能力排名。原始工作认为，从因果预训练平滑过渡到全双向编码，比直接切换更影响稳定性与最终用户 embedding 质量。

## GG-SM 机制

Gradient-Guided Soft Masking（GG-SM）先用 hidden state 梯度强度为未来 token 分配软可见度，形成 warm-up；再将冻结的软 mask 线性调度到全双向 mask。用户侧和答案侧仍由双塔对比目标对齐，输入保留模态专属 encoder、adapter 与特殊用户 token。

## 适用边界

全双向上下文适合离线、完整历史的用户表征，但不能直接等价为流式增量服务的最优选择；因果结构仍可能更方便在线追加和缓存。原始阅读稿的九项工业 benchmark 说明这一训练配方在特定数据和模型中有效，部署前仍应比较增量延迟、历史刷新、掩码训练稳定性、任务切片和隐私边界。

参见 [[来源-Query-as-Anchor场景自适应用户表征-2026]]：二者都使用 decoder-only 主干，但前者解决 query 条件化和多场景缓存，GG-SM 解决把主干作为全上下文嵌入器时的掩码适配。