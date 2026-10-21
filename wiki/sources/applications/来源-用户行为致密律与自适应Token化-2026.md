# 来源：用户行为致密律与自适应 Token 化（2026）

> 原始来源：[`raw/transcripts/LLM-based User Representation Foundation Models/paper_zh_13_Densing_Law.md`](../../../raw/transcripts/LLM-based%20User%20Representation%20Foundation%20Models/paper_zh_13_Densing_Law.md)。对应论文：*Towards a Densing Law for User Representation Learning at Billion-Scale Capacity*，arXiv:2608.23392v1。
>
> 关联概念：[[LLM用户表征基础模型与场景适配]]、[[LLM推荐系统的分层改造与评估]]。

## 核心结论

该工作提出“原始行为规模墙”：在高度重复的用户历史上，继续增加用户数、时间窗或编码器参数，可能出现预训练损失继续下降而下游质量停滞的 **loss-quality dissociation**。其主张不是“小模型普遍更好”，而是有效信息密度和最小充分 token 容量会成为规模化瓶颈。

## 机制与决策

- 用残差量化将冗余行为历史压成紧凑离散序列，以保留任务相关差异并扩大可用历史窗。
- Behavioral Densing Law 将数据规模与所需 tokenization 容量建立经验性关系；容量不应固定地随原始记录线性增加。
- ALGN 按量化残差与表达不确定性给高信息段更多 code、给可预测常规段更少 code，近似“按需分配”容量。

## 证据边界

论文的亿级支付宝实验支持其对该数据分布、编码器和线性探测协议的设计假设；并不证明任意业务都已饱和，也不证明压缩必然优于保留原始序列。采用时应同时报告按时间切分的任务效用、码本碰撞、重建误差、长尾/新用户、延迟与存储成本；预训练 loss 不能替代这些验收。

该结论与 [[来源-FOUNDv2统一量化用户Tokenizer-2026]] 的统一量化互补：前者讨论容量随数据规模的配置与分配，后者提供多源统一离散 tokenizer 的具体架构。