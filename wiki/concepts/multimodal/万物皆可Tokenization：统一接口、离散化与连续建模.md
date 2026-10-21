# 万物皆可 Tokenization：统一接口、离散化与连续建模

> 时间范围：2025-10-21 至 2026-10-21。本文聚焦最近一年公开工作，较早研究只作为方法前史。
> 相邻主题：[[多模态大模型跨模态投影与连接器]]、[[多模态结构保真与统一对齐]]、[[LLM推荐系统的分层改造与评估]]、[[LLM用户表征基础模型与场景适配]]。

## 一句话结论

“万物皆可 tokenization”更准确的含义不是“所有对象都应离散成语言词表”，而是：

> 任意对象都可以被设计成 Transformer 可消费、压缩、组合和预测的接口单元，但接口单元未必是离散 token。

最近一年的研究重心已从“能否把新模态离散化”转向四个更困难的问题：

1. tokenizer 向模型暴露了什么信息，又永久丢失了什么；
2. token 粒度、数量和层级是否应随样本或任务变化；
3. token 的依赖结构是否与自回归、并行或迭代解码匹配；
4. 连续 latent、flow matching 或直接回归是否比离散词表更符合目标对象的几何与动力学。

## 先区分五种 Tokenization

“tokenization”在不同论文里至少指五件不同的事，不能直接拼成同一技术排行榜。

| 类型 | token 的形态 | 主要目标 | 典型例子 | 关键风险 |
|---|---|---|---|---|
| 固定分块 | patch、帧、窗口、子图 | 把非序列对象变成可输入序列 | ViT patch、时间窗口 | 序列过长；边界可能切断结构 |
| 连续 latent | 连续向量或 latent chunk | 接入 Transformer，同时保留连续信息 | UniFlow、Multimodal Flow | 不再能直接使用离散 NTP 和词表约束 |
| 离散量化码 | VQ、RQ、FSQ 或层级 code | 统一 next-token prediction、压缩与生成 | Kelix、SemHiTok、UniAudio 2.0 | 量化误差、码本碰撞、信息瓶颈 |
| Semantic ID | 实体、物品或用户的离散标识序列 | 生成式检索、推荐与目录编码 | ACERec、CapsID、FLASH | ID 层级可能是假结构；解码与目录约束复杂 |
| 动作或结构输出词表 | 动作块、图结构、时间间隔等可执行符号 | 把预测改写为序列生成 | ActionCodec、X-Tokenizer、ChronoSID | 串行延迟、误差累积、连续控制不平滑 |

因此，“一个模型能消费 token”只说明接口兼容，不说明表示无损、语义已对齐，也不说明离散化优于连续建模。

## 支持方：为什么要把更多对象 Token 化

### 1. 统一训练目标与模型主干

离散 token 可把文本、图像、音频、动作和实体检索统一为 next-token prediction。模型可以复用成熟的自回归架构、因果掩码、KV cache、采样和分布式训练基础设施。Kelix、UniAudio 2.0、UniTok-FM 都把这种统一性视为核心收益。

统一接口的价值首先是**训练与系统同构**，而不是证明不同模态具有相同的信息结构。

### 2. 用有限带宽压缩高维对象

像素、波形、长时序、用户历史和机器人轨迹都可能远超上下文预算。离散码、层级 token 和 Semantic ID 能把高维对象压成有限序列，使更长历史或更大样本规模进入模型。FOUNDv2、TimeTok、ACERec 等路线都利用了这种“表示粒度与主模型成本解耦”。

### 3. 获得可组合、可约束的符号接口

物品 ID、动作词表、时间间隔 token 和图结构 token 可以参与序列组合、前缀约束、掩码和目录校验。相比不透明的单个稠密向量，它们更容易接入生成式检索、结构化输出和多步规划。

但“可以施加约束”不等于“token 自带可靠语义”；约束通常仍需目录、语法或执行器提供。

### 4. 复用语言模型的扩展规律

当新模态被转换成稳定 token 流后，可以尝试扩大数据、模型和上下文规模，而不为每个任务设计专用预测头。X-Tokenizer 和 UniAudio 2.0 进一步把 tokenizer 视为基础模型与领域控制之间的语义接口，而不只是压缩器。

### 5. 支持粗到细、语义到重建的分层生成

层级 token 可让模型先预测高层意图，再补充低层细节。SemHiTok、UniAudio 2.0、X-Tokenizer 和 TimeTok 分别在视觉、音频、动作与时序上体现了这一设计。相比单一码本同时承担所有目标，因子化更容易处理“理解需要抽象，重建需要细节”的冲突。

## 反方：为什么不应把一切都离散化

### 1. 量化是不可逆的信息瓶颈

连续视觉特征通常更适合理解，离散码更适合自回归生成；连续动作和波形中的细微变化也可能被硬量化抹平。Manzano 直接保留“连续理解 + 离散生成”双表示，说明完全统一成一种 token 形态未必最优。

### 2. Tokenizer 会预先限制模型表达能力

图结构研究 *Lost in Tokenization* 给出明确反例：邻接、谱和随机游走 tokenization 会使同一任务进入不同的 Transformer 深度复杂度区间；随机游走表示可证明有损，谱表示虽可无损，却可能不适合局部任务。模型不能总在有限深度内补偿不合适的输入表示。

这意味着 tokenizer 不是中性的预处理，而是计算模型的一部分。

### 3. 连续对象未必适合串行离散解码

机器人动作具有平滑性、几何结构和实时控制要求。离散自回归动作 token 会引入串行延迟、量化误差和早期错误累积。RotVLA 以连续旋转 latent 和 flow-matching action head 建模动作，代表“保持对象原有几何”优先于“强行复用语言词表”。

### 4. 统一离散词表并非统一多模态的必要条件

UniFlow、Multimodal Flow 和 tokenizer-free TTS 表明，连续 latent 同样可以进入统一 Transformer 或 flow backbone。统一可以发生在注意力、chunk 组织、向量场或训练协议层，不一定发生在离散词表层。

### 5. 复杂 Learned Tokenizer 未必优于简单映射

FLASH 使用免训练 SimHash，却通过并行解码和显式语义对齐取得有竞争力的作者报告结果。它指出一些 learned tokenizer 的优势可能来自与自回归解码器更匹配，而不是 tokenizer 本身必然更有表达力。

因此，评估 tokenizer 时必须同时控制解码方式、语义对齐、序列长度和计算预算。

### 6. Token 数量可能只是把成本转移到上下文

更细粒度通常减少重建误差，却会放大序列长度、KV cache、训练 FLOPs 和解码时延。更粗粒度降低成本，却增加碰撞、别名和长尾损失。不存在脱离任务与预算的“最佳 token 粒度”。

## 最近一年代表工作

### 视觉：理解与生成的接口冲突

| 工作 | 日期/状态 | Tokenization 设计 | 说明 |
|---|---|---|---|
| [Kelix](https://arxiv.org/abs/2602.09843) | 2026-02，预印本 | 每个图像 patch 使用多个并行离散 token，再聚合为复合表示 | 增加离散视觉 token 容量，目标是缩小其与连续 ViT 特征的理解差距 |
| [SemHiTok](https://openreview.net/forum?id=LKoDSviFpL) | ICLR 2026 Poster | 语义 codebook 下挂像素 sub-codebook | 用结构分层拆开高层语义与低层纹理目标 |
| [MANZANO](https://openreview.net/forum?id=FIXPFUeO9Z) | ICLR 2026 Poster | 共享视觉编码器，连续表示用于理解，离散 token 用于生成 | “统一 tokenizer”不要求理解和生成使用同一种表示形态 |
| [UniFlow](https://openreview.net/forum?id=AwxGRIG8Fb) | ICLR 2026 Poster | 连续语义特征 + patch-wise pixel flow decoder | 说明 tokenizer 可以提供连续接口，而不依赖离散量化 |

视觉领域的共同问题是：理解偏好语义不变性，生成与重建需要保留像素细节。最近工作的主流答案不是无限扩大单一码本，而是分支、分层或用连续解码器补充。

### 音频：推理语义与声学细节分工

| 工作 | 日期/状态 | Tokenization 设计 | 说明 |
|---|---|---|---|
| [UniAudio 2.0](https://arxiv.org/abs/2602.04683) | 2026-02，预印本 | 文本对齐的 reasoning tokens + 高保真 reconstruction tokens | 将音频理解/规划与波形重建显式因子化 |
| [FreyaTTS](https://arxiv.org/abs/2607.09530) | 2026-07，预印本 | 不使用离散语音 tokenizer，在连续 AudioVAE latent 上做 flow matching | 构成“语音生成必须先离散化”的直接反例 |

音频同时包含离散语言内容和连续音色、韵律、环境声。合理设计往往不是二选一，而是让低码率语义表示和高保真连续声学通道承担不同责任。

### 动作：语言接口与连续控制的张力

| 工作 | 日期/状态 | Tokenization 设计 | 说明 |
|---|---|---|---|
| [ActionCodec](https://arxiv.org/abs/2602.15397) | 2026-02，预印本 | 按时间重叠、词表冗余、跨模态互信息和 token 独立性设计动作码 | 将评价目标从重建误差转向 VLA 优化 |
| [X-Tokenizer](https://arxiv.org/abs/2606.14752) | 2026-06，预印本 | 第一层形成粗粒度动作语言，深层残差保留精细控制 | 把 action tokenizer 定位为多模态推理与执行控制之间的语义接口 |
| [RotVLA](https://arxiv.org/abs/2605.13403) | 2026-05，预印本 | SO(n) 上的连续旋转 latent action + flow matching | 以连续性、可组合性和动作几何反驳硬离散量化 |

动作领域没有收敛到“离散一定优于连续”。离散 token 更容易进入 VLM 原生 NTP；连续 flow head 更符合平滑控制和并行生成。混合离散—连续系统可能比纯路线更现实。

### 时间序列：尺度、形状与粒度控制

| 工作 | 日期/状态 | Tokenization 设计 | 说明 |
|---|---|---|---|
| [UniTok / UniTok-FM](https://arxiv.org/abs/2606.09861) | 2026-05，预印本 | 前缀归一化、渐进分辨率 VQ，把时序转成离散 token | 用通用 LLM 架构统一预测、生成与分类 |
| [TimeTok](https://arxiv.org/abs/2605.01418) | 2026-05，arXiv 标注 NeurIPS 2026 | 粗到细层级 token block | 通过生成 token block 数量控制输出时间粒度 |

时序不是天然有界对象。数值尺度、采样率、周期和窗口边界都会影响码本。最近工作开始把“尺度统计”与“形状模式”拆开，并让粒度成为可控制变量。

### 推荐与检索：从固定 Semantic ID 到自适应接口

| 工作 | 日期/状态 | Tokenization 设计 | 说明 |
|---|---|---|---|
| [ACERec](https://arxiv.org/abs/2602.13573) | 2026-02，预印本 | 先生成长 Semantic ID，再压缩成少量 latent tokens | 解耦 tokenizer 的细粒度表达和主模型上下文成本 |
| [CapsID](https://arxiv.org/abs/2605.05096) | 2026-05，预印本 | 软 capsule routing、置信度终止、SemanticBPE | 用变长 ID 缓解硬最近邻和固定深度量化 |
| [ChronoSID](https://arxiv.org/abs/2607.03918) | 2026-07，预印本 | 将 log-scale 交互间隔编码成 gap tokens | 把“时间间隔”也纳入生成式推荐词表 |
| [PRQ-KMeans](https://arxiv.org/abs/2608.24207) | 2026-08，预印本 | 投影残差、全局均值移除和软式 centroid 更新 | 将层级 SID 解释为逐层移除共性 |
| [FLASH](https://arxiv.org/abs/2610.07402) | 2026-10，预印本 | 免训练 SimHash + 并行解码 + 语义对齐 | 反驳复杂 learned quantizer 必然更优 |

推荐领域的问题已经从“是否采用 Semantic ID”推进到：

- ID 长度是否应自适应；
- 层级是否对应真实语义，而非量化过程的偶然顺序；
- 主模型是否必须逐 token 自回归解码；
- 量化损失能否通过连续语义对齐补偿；
- 目录合法性、冷启动和增量更新如何进入评测。

### 图结构：Tokenization 决定可计算什么

| 工作 | 日期/状态 | Tokenization 设计 | 说明 |
|---|---|---|---|
| [VesselTok](https://arxiv.org/abs/2603.18797) | ECCV 2026 | 将高分辨率血管类空间图压成可生成的 latent tokens | 展示结构 token 对三维生物医学图压缩与生成的价值 |
| [\<SOG_k\>: One LLM Token for Graph](https://arxiv.org/abs/2602.01771) | 2026-02，预印本 | 用一个特殊 token 表示图拓扑类别 | 追求极端压缩和文本词表对齐，但压缩上限取决于结构词表覆盖 |
| [EDT-Former](https://arxiv.org/abs/2602.02742) | ICLR 2026 | 按分子 patch 信息量生成动态数量 token | 反对从视觉任务照搬固定长度 Q-Former token |
| [Lost in Tokenization](https://arxiv.org/abs/2605.22471) | 2026-05，预印本 | 理论比较邻接、谱与随机游走 tokenization | 证明表示选择会导致深度分离、信息损失和任务适配差异 |

图结构提供了最强的反例：即使所有表示都能输入 Transformer，它们暴露的局部性、全局性和拓扑信息仍根本不同。

### 完全连续的统一建模反例

[Multimodal Flow](https://arxiv.org/abs/2609.40362) 将文本块与图像组织成有序连续 hyperchunks，在共享 chunk-causal flow backbone 中学习统一向量场。它直接挑战两种常见假设：

1. 视觉必须量化后才能与语言统一；
2. 多模态统一必须依赖同一个离散 next-token prediction 目标。

该工作仍属于作者报告的预印本，但它提供了重要存在性证据：**统一模型结构与统一离散词表不是同一件事。**

## 过去一年的共同趋势

### 趋势一：从单一码本转向因子化

语义/重建、粗/细、共享/来源特定、意图/控制分别由不同 token 层或通道承担。因子化不是额外装饰，而是在处理多个目标对表示提出的冲突要求。

### 趋势二：长度与粒度变成样本相关变量

CapsID、TimeTok、EDT-Former 和 ACERec 都在打破固定 token 数假设。高信息对象分配更多容量，常规或可预测部分使用更少 token，逐渐成为跨领域共同设计。

### 趋势三：Tokenizer 与预测器共同设计

同一个 token 序列在自回归、并行解码和迭代 refinement 下可能表现不同。FLASH 的核心反证是：若解码结构不匹配，不能把失败完全归因于 tokenizer。

### 趋势四：连续与离散走向混合

Manzano 的连续理解/离散生成、UniAudio 2.0 的推理/重建分工、VLA 的离散语义监督/连续控制头，都表明混合表示可能比“全离散”或“全连续”更符合系统目标。

### 趋势五：Tokenization 从压缩器变成接口协议

新的 tokenizer 不只优化重建误差，还承担语义对齐、跨模型通信、目录约束、执行控制和计算预算管理。相应地，评测也必须从 reconstruction 扩展到下游效用、语义互信息、延迟、鲁棒性和可逆性。

## 如何选择：离散、连续还是混合

| 条件 | 更适合离散 token | 更适合连续 latent / flow | 更适合混合 |
|---|---|---|---|
| 输出具有清晰词表、目录或语法 | 是 | 否 | 可选 |
| 需要复用原生 LLM NTP 和 KV cache | 是 | 较弱 | 是 |
| 微小数值变化具有物理意义 | 风险较高 | 是 | 是 |
| 需要高保真像素、波形或平滑动作 | 通常需大量 token | 是 | 是 |
| 需要前缀约束和可枚举合法输出 | 是 | 较弱 | 是 |
| 需要并行生成和低控制延迟 | 自回归路线较弱 | 是 | 可采用并行离散解码 |
| 理解与重建目标冲突明显 | 单一码本较弱 | 可行 | 通常更稳妥 |

最小决策流程：

1. 先定义必须保留的任务信息，而不是先选 VQ 架构；
2. 建立无量化连续基线，测量量化带来的真实损失；
3. 同时控制 token 数、上下文 FLOPs、解码方式与训练数据；
4. 检查固定长度是否浪费容量或伤害长尾；
5. 对结构、动作和时序任务增加可逆性、平滑性与执行后果评测；
6. 只有当系统同构、压缩或约束收益覆盖量化损失时，才采用全离散路线。

## 最小评测矩阵

| 维度 | 必测问题 |
|---|---|
| 信息保真 | 能否重建关键细节？是否保留局部、时间、拓扑或动力学结构？ |
| 下游效用 | token 是否真的改善理解、生成、控制、检索或推荐，而不只改善重建？ |
| 容量与碰撞 | 码本利用率、碰撞率、长尾对象和新对象如何表现？ |
| 计算成本 | token 数、训练 FLOPs、KV cache、TTFT、吞吐与解码轮数是多少？ |
| 解码匹配 | 自回归、并行或迭代解码是否符合 token 之间的依赖关系？ |
| 鲁棒性 | 采样率、分辨率、尺度、设备、实体目录或数据分布变化后是否稳定？ |
| 可执行性 | 动作、结构或目录输出是否合法，错误能否被执行器发现和回退？ |

## 与现有页面的关系

- [[多模态大模型跨模态投影与连接器]] 讨论表示转换放在哪里；本文进一步比较连续接口、离散 tokenizer 和无独立 connector 的统一路线。
- [[多模态结构保真与统一对齐]] 给出数值、时间、空间与拓扑结构的保真红线；tokenization 是这些结构进入主干前的第一道信息瓶颈。
- [[LLM推荐系统的分层改造与评估]] 讨论 Semantic ID 在推荐系统中的职责和上线约束；本文补充 2026 年变长 SID、并行解码和简单 hash 反证。
- [[LLM用户表征基础模型与场景适配]] 关注用户历史压缩与可迁移表示；用户 token 与物品 Semantic ID 共享量化工具，但服务对象和约束不同。
- [[异构数据接入大模型的四档方案对比-2026]] 从工程接入深度比较文本化、软 token、独立编码器投影和端到端融合；本文聚焦表示单元本身的设计。

## 结论边界

截至 2026-10-21，证据支持“tokenization 是通用接口设计工具”，但不支持“全离散 token 是所有模态的统一最优解”。

更稳健的工程原则是：

> 先识别对象的结构、任务和服务约束，再选择最小充分接口；能统一的是计算协议，不必强行统一信息形态。

除 SemHiTok、MANZANO、UniFlow、EDT-Former、VesselTok 等已标明会议状态的工作外，本文多数 2026 年论文仍是 arXiv 预印本，性能数字属于作者报告。跨论文的数据、模型、预算和评测协议差异较大，本文只比较设计论点，不据此构造无条件性能排行榜。

## Sources

- [Kelix Technical Report](https://arxiv.org/abs/2602.09843)
- [SemHiTok: A Unified Image Tokenizer via Semantic-Guided Hierarchical Codebook for Multimodal Understanding and Generation](https://openreview.net/forum?id=LKoDSviFpL)
- [MANZANO: A Simple and Scalable Unified Multimodal Model with a Hybrid Vision Tokenizer](https://openreview.net/forum?id=FIXPFUeO9Z)
- [UniFlow: A Unified Pixel Flow Tokenizer for Visual Understanding and Generation](https://openreview.net/forum?id=AwxGRIG8Fb)
- [UniAudio 2.0: A Unified Audio Language Model with Text-Aligned Factorized Audio Tokenization](https://arxiv.org/abs/2602.04683)
- [ActionCodec: What Makes for Good Action Tokenizers](https://arxiv.org/abs/2602.15397)
- [X-Tokenizer: A Multimodal Action Tokenizer for Vision-Language-Action Pretraining](https://arxiv.org/abs/2606.14752)
- [RotVLA: Rotational Latent Action for Vision-Language-Action Model](https://arxiv.org/abs/2605.13403)
- [Time Series as Language: A Universal Tokenizer for General-Purpose Time Series Foundation Models](https://arxiv.org/abs/2606.09861)
- [TimeTok: Granularity-Controllable Time-Series Generation via Hierarchical Tokenization](https://arxiv.org/abs/2605.01418)
- [Unleash the Potential of Long Semantic IDs for Generative Recommendation](https://arxiv.org/abs/2602.13573)
- [CapsID: Soft-Routed Variable-Length Semantic IDs for Generative Recommendation](https://arxiv.org/abs/2605.05096)
- [Beyond Item Order: Temporal Gap Tokenization for Generative Recommendation with Semantic IDs](https://arxiv.org/abs/2607.03918)
- [PRQ-KMeans: Projection Residual Quantization for Semantic ID Tokenization](https://arxiv.org/abs/2608.24207)
- [Rethinking Semantic ID Construction for Generative Recommendation: SimHash with Parallel Decoding and Semantic Alignment](https://arxiv.org/abs/2610.07402)
- [VesselTok: Tokenizing Vessel-like 3D Biomedical Graph Representations for Reconstruction and Generation](https://arxiv.org/abs/2603.18797)
- [\<SOG_k\>: One LLM Token for Explicit Graph Structural Understanding](https://arxiv.org/abs/2602.01771)
- [Entropy-Guided Dynamic Tokens for Graph-LLM Alignment in Molecular Understanding](https://arxiv.org/abs/2602.02742)
- [Lost in Tokenization: Fundamental Trade-offs in Graph Tokenization for Transformers](https://arxiv.org/abs/2605.22471)
- [Multimodal Flow: Unified Flow Modeling of Language and Vision in Embedding Spaces](https://arxiv.org/abs/2609.40362)
- [FreyaTTS: A Compact Tokenizer-Free Flow-Matching Transformer for Turkish-First Speech Synthesis](https://arxiv.org/abs/2607.09530)

*最后更新：2026-10-21*
