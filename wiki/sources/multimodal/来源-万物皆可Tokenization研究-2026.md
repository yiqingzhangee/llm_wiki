# 来源：万物皆可 Tokenization 研究（2026）

> 调研日期：2026-10-21。时间窗为 2025-10-21 至 2026-10-21。本文是 [[万物皆可Tokenization：统一接口、离散化与连续建模]] 的独立证据入口；宽搜候选及排除理由见 [[Tokenization候选工作附录-2026]]。

## 调研问题

过去一年，“万物皆可 tokenization”究竟意味着把所有模态离散成词表，还是把不同对象变成 Transformer 可消费的计算接口？离散 token、连续 latent、Semantic ID、动作词表和结构 token 各自解决什么问题，又在什么条件下会成为信息瓶颈？

## 检索与纳入方法

- **时间口径**：以论文首次公开日期为准，而不是后续修订或会议接收日期。
- **来源优先级**：论文与 OpenReview 页面优先；项目页只补充代码、模型与演示信息。
- **纳入条件**：工作必须直接改变表示单元、量化方式、token 粒度、长度、层级或解码依赖，或构成“不需要离散 tokenizer”的明确反例。
- **排除条件**：仅使用既有 tokenizer、只改主干或推理加速、与 tokenization 只有标题级关联的工作不进入核心清单。
- **比较原则**：只比较机制和主张，不把不同数据、主干、预算与 benchmark 的作者报告拼成排行榜。

## 证据分层

| 层级 | 含义 | 本页用法 |
|---|---|---|
| A | 已有会议接收或 OpenReview 状态，且机制与主题直接相关 | 支撑较稳定的方法趋势 |
| B | 一手预印本或技术报告，机制与主题直接相关 | 支撑存在性和设计方向，性能按作者报告处理 |
| C | 邻近连续建模、解码或系统工作 | 用于反证和边界，不作为 tokenizer 性能证据 |
| D | 超出时间窗、信息不足或关联较弱 | 只进入候选附录，不支撑核心结论 |

证据等级表示可用于本主题的强度，不等同于论文质量评价。

## 核心证据矩阵

### 视觉与统一多模态

| 工作 | 状态 | 表示设计 | 支撑的结论 |
|---|---|---|---|
| [Kelix](https://arxiv.org/abs/2602.09843) | B | 每个 patch 使用多个并行离散 token，再聚合为复合表示 | 单个视觉 token 的容量可成为理解瓶颈 |
| [SemHiTok](https://openreview.net/forum?id=LKoDSviFpL) | A，ICLR 2026 Poster | 语义 codebook 与像素 sub-codebook 分层 | 理解语义与重建纹理适合因子化 |
| [MANZANO](https://openreview.net/forum?id=FIXPFUeO9Z) | A，ICLR 2026 Poster | 连续理解表示与离散生成 token 共用视觉编码器 | 统一 tokenizer 不要求统一表示形态 |
| [UniFlow](https://openreview.net/forum?id=AwxGRIG8Fb) | A，ICLR 2026 Poster | 连续语义特征与 patch-wise pixel flow | 统一接口不必依赖离散量化 |
| [Multimodal Flow](https://arxiv.org/abs/2609.40362) | C | 文本与视觉在连续 embedding 空间共享 flow backbone | 统一主干与统一离散词表不是同一件事 |

视觉证据共同指向一个张力：理解偏好语义不变性，重建偏好像素细节。近期方案主要通过多码本、分层、双表示或连续解码器处理冲突，而不是假设单一码本可以无代价承担所有目标。

### 音频

| 工作 | 状态 | 表示设计 | 支撑的结论 |
|---|---|---|---|
| [UniAudio 2.0](https://arxiv.org/abs/2602.04683) | B | 文本对齐的 reasoning tokens 与 reconstruction tokens 分工 | 语言语义和声学细节适合不同通道 |
| [FreyaTTS](https://arxiv.org/abs/2607.09530) | C | 连续 AudioVAE latent 上的 flow matching | TTS 不必经过离散语音 tokenizer |

音频同时具有离散语言内容与连续韵律、音色和环境细节。该领域支持“因子化或混合接口”，但不足以支持“全离散”或“全连续”对所有音频任务普遍最优。

### 动作与机器人

| 工作 | 状态 | 表示设计 | 支撑的结论 |
|---|---|---|---|
| [ActionCodec](https://arxiv.org/abs/2602.15397) | B | 以跨模态互信息、token 独立性等指标设计动作码 | 动作 tokenizer 不能只按重建误差评价 |
| [X-Tokenizer](https://arxiv.org/abs/2606.14752) | B | 粗粒度动作语言与深层残差控制 | 动作接口需要语义与精细控制分层 |
| [RotVLA](https://arxiv.org/abs/2605.13403) | C | 连续旋转 latent action 与 flow matching | 平滑控制和动作几何可能更适合连续表示 |

动作领域最能暴露解码方式与表示方式的耦合。离散 token 便于接入 VLM 的 next-token prediction；连续 action head 则保留平滑性、并行生成和几何结构。现有证据更支持混合系统，而不是单一路线收敛。

### 时间序列

| 工作 | 状态 | 表示设计 | 支撑的结论 |
|---|---|---|---|
| [UniTok / UniTok-FM](https://arxiv.org/abs/2606.09861) | B | 尺度与形状 token，支持渐进分辨率 | 无界连续时序需要显式处理尺度和分辨率 |
| [TimeTok](https://arxiv.org/abs/2605.01418) | B | 粗到细层级 token block | 输出粒度可以成为可控制变量 |

时序 tokenization 不能只复制图像 patch。采样率、尺度、周期、缺失和异步通道都会改变 token 边界的意义。

### 推荐与检索

| 工作 | 状态 | 表示设计 | 支撑的结论 |
|---|---|---|---|
| [ACERec](https://arxiv.org/abs/2602.13573) | B | 长 Semantic ID 压缩为少量 latent tokens | 表示粒度与主模型上下文成本可以解耦 |
| [CapsID](https://arxiv.org/abs/2605.05096) | B | 软路由、置信度终止与变长 ID | 固定深度量化不是必要假设 |
| [ChronoSID](https://arxiv.org/abs/2607.03918) | B | 时间间隔量化为 gap tokens | 行为时间也可成为生成式接口的一部分 |
| [PRQ-KMeans](https://arxiv.org/abs/2608.24207) | B | 投影残差与逐层共性移除 | SID 层级结构需要显式设计 |
| [FLASH](https://arxiv.org/abs/2610.07402) | B | SimHash、并行解码与语义对齐 | 复杂 learned tokenizer 未必天然更优 |

推荐证据显示，研究问题已从“是否采用 Semantic ID”转向长度自适应、层级可解释性、目录约束、解码匹配、冷启动和增量更新。

### 图结构

| 工作 | 状态 | 表示设计 | 支撑的结论 |
|---|---|---|---|
| [VesselTok](https://arxiv.org/abs/2603.18797) | A，ECCV 2026 | 三维血管空间图压缩为 latent tokens | 结构 token 可服务压缩、重建与生成 |
| [\<SOG_k\>: One LLM Token for Graph](https://arxiv.org/abs/2602.01771) | B | 用特殊 token 表示图拓扑类别 | 极端压缩依赖结构词表覆盖 |
| [EDT-Former](https://arxiv.org/abs/2602.02742) | A，ICLR 2026 | 按分子 patch 信息量动态分配 token | 固定 token 数可能不适合结构对象 |
| [Lost in Tokenization](https://arxiv.org/abs/2605.22471) | B，理论工作 | 比较邻接、谱与随机游走 tokenization | tokenizer 会改变有限深度 Transformer 可计算的函数 |

图结构提供了最强边界证据：表示形式不仅影响压缩率，还会改变局部性、全局性、可逆性和任务复杂度。

## 跨来源综合

### 已获多数证据支持的趋势

1. **因子化取代单一码本。** 视觉、音频和动作工作都把语义/重建、粗/细或意图/控制拆成不同层或通道。
2. **长度和粒度逐渐样本化。** CapsID、TimeTok、EDT-Former 与 ACERec 以不同方式取消固定 token 数的默认假设。
3. **Tokenizer 与预测器必须共同评价。** 自回归、并行、离散 diffusion 和 flow refinement 对 token 依赖的假设不同。
4. **连续与离散走向混合。** MANZANO、UniFlow、RotVLA、FreyaTTS 和 Multimodal Flow 表明，统一计算协议不要求统一成离散词表。

### 仍有争议的问题

- 视觉理解与生成是否最终需要同一离散码本，还是共享编码器上的双表示更合理；
- 动作的语义监督是否应使用离散 token，而部署控制是否应交给连续 action head；
- 变长 token 的精度收益能否覆盖动态 batching、缓存和部署复杂度；
- 复杂 learned tokenizer 的收益有多少来自表示本身，有多少来自更匹配的解码器和训练目标。

## 证据限制

- 除已标明会议状态的工作外，多数 2026 年材料仍是预印本或技术报告。
- 性能数字多数来自作者自报，训练数据、模型规模和评测协议不可直接比较。
- 论文将 “token” 用于固定 patch、连续 latent、离散 code、Semantic ID 和动作符号等不同对象；只有先区分定义，跨领域综合才有意义。
- 连续建模工作主要提供存在性反证，不能单独证明连续路线在所有任务上优于离散路线。
- 本页冻结于 2026-10-21；新候选先进入附录，经过阅读全文和交叉核验后再晋级核心证据。

## 关联页面

- [[万物皆可Tokenization：统一接口、离散化与连续建模]]
- [[Tokenization候选工作附录-2026]]
- [[多模态大模型跨模态投影与连接器]]
- [[多模态结构保真与统一对齐]]
- [[LLM推荐系统的分层改造与评估]]

## Sources

- [Kelix Technical Report](https://arxiv.org/abs/2602.09843)
- [SemHiTok](https://openreview.net/forum?id=LKoDSviFpL)
- [MANZANO](https://openreview.net/forum?id=FIXPFUeO9Z)
- [UniFlow](https://openreview.net/forum?id=AwxGRIG8Fb)
- [UniAudio 2.0](https://arxiv.org/abs/2602.04683)
- [ActionCodec](https://arxiv.org/abs/2602.15397)
- [X-Tokenizer](https://arxiv.org/abs/2606.14752)
- [RotVLA](https://arxiv.org/abs/2605.13403)
- [UniTok / UniTok-FM](https://arxiv.org/abs/2606.09861)
- [TimeTok](https://arxiv.org/abs/2605.01418)
- [ACERec](https://arxiv.org/abs/2602.13573)
- [CapsID](https://arxiv.org/abs/2605.05096)
- [ChronoSID](https://arxiv.org/abs/2607.03918)
- [PRQ-KMeans](https://arxiv.org/abs/2608.24207)
- [FLASH](https://arxiv.org/abs/2610.07402)
- [VesselTok](https://arxiv.org/abs/2603.18797)
- [\<SOG_k\>: One LLM Token for Graph](https://arxiv.org/abs/2602.01771)
- [EDT-Former](https://arxiv.org/abs/2602.02742)
- [Lost in Tokenization](https://arxiv.org/abs/2605.22471)
- [Multimodal Flow](https://arxiv.org/abs/2609.40362)
- [FreyaTTS](https://arxiv.org/abs/2607.09530)

*最后更新：2026-10-21*
