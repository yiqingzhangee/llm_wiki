# Tokenization 候选工作附录（2026）

> 本页归档 [[来源-万物皆可Tokenization研究-2026]] 宽搜阶段发现、但尚未进入 [[万物皆可Tokenization：统一接口、离散化与连续建模]] 精选清单的工作。时间窗为 2025-10-21 至 2026-10-21。

## 状态规则

| 状态 | 含义 | 后续处理 |
|---|---|---|
| 已核验候选 | 已核对一手论文的标题、日期和核心机制，主题直接相关 | 阅读方法与消融后，可晋级核心清单 |
| 邻近反例 | 主要研究连续建模、flow、解码或系统，但能检验“必须离散化”的主张 | 保留为边界证据，不进入 tokenizer 排名 |
| 弱相关 | 使用 token/SID，论文创新重点不在 tokenization | 仅作邻近工作导航 |
| 超窗前史 | 首次公开日期早于 2025-10-21 | 不计入最近一年趋势，只作方法前史 |

## 已核验候选

### 视觉与跨模态生成

| 工作 | 首次公开 | 核心机制 | 为什么尚未进入精选清单 |
|---|---:|---|---|
| [KVAE: Family of Tokenizers for Multimodal Generative Models](https://arxiv.org/abs/2608.05798) | 2026-08-06 | 为音频、图像和视频提供连续压缩表示；覆盖 48 kHz 音频、二维图像与因果视频 tokenizer | 更接近多模态生成基础设施与 VAE 家族，需要阅读全文确认跨模态统一性来自共享原则还是产品组合 |
| [UniWeTok](https://arxiv.org/abs/2602.14178) | 2026-02-15 | 使用 `2^128` 规模的二进制码空间，结合蒸馏与生成先验统一理解和生成 | 与 Kelix、SemHiTok 的大容量/语义分层问题重叠；需进一步比较码利用率、碰撞和并行预测成本 |
| [UniAR](https://arxiv.org/abs/2606.18249) | 2026-06-16 | 单一离散视觉 tokenizer、bitwise 量化与并行 bit 预测，生成结果无需重新编码即可回到上下文 | ICML 2026 候选价值较高，但主张同时依赖自回归系统与 diffusion decoder，需要拆分 tokenizer 和系统收益 |
| [EvoTok](https://arxiv.org/abs/2603.12108) | 2026-03-12 | 以残差量化轨迹让像素表示逐层演化为语义表示 | 宽搜后新增；与 SemHiTok 的“分层解决语义/像素冲突”相邻，需做受控差异比较 |

### 动作与 VLA

| 工作 | 首次公开 | 核心机制 | 为什么尚未进入精选清单 |
|---|---:|---|---|
| [RoboOmni](https://arxiv.org/abs/2510.23763) | 2025-10-27 | 使用 FAST+ 把连续动作向量编码为离散动作 token，接入 omni-modal LLM | 系统创新重点是跨模态上下文中的主动机器人协作，tokenizer 为采用组件 |
| [PhysBrain 1.5](https://arxiv.org/abs/2609.14973) | 2026-09-14 | ActionPiece 将轨迹转为动作词表，并与语言、视觉目标共同做 next-token prediction | 是“物理世界统一离散序列”的强样本，但技术报告规模大，需拆分 tokenizer、数据与主干贡献 |
| [MolmoAct2 / OpenFAST](https://arxiv.org/abs/2605.02881) | 2026-05-04 | 开放权重、开放数据动作 tokenizer；以 KV-cache 条件连接离散 VLM 与连续 flow action expert | 对混合离散—连续路线很有价值，需进一步读取 OpenFAST 独立训练和跨 embodiment 消融 |
| [Wall-OSS-0.5](https://arxiv.org/abs/2605.30877) | 2026-05-29 | 离散动作预测向 VLM 主干传梯度，连续 flow matching 作为部署动作接口 | 主要贡献是可测量的 VLA 预训练和梯度桥接，tokenizer 不是唯一变量 |
| [DiscreteRTC](https://arxiv.org/abs/2604.25050) | 2026-04-27 | 动作均匀量化后，以离散 diffusion 的原生 unmasking 实现异步补全 | 核心创新是异步执行与 inpainting，不是语义 tokenizer |
| [DFM-VLA](https://arxiv.org/abs/2603.26320) | 2026-03-27 | 离散 flow matching 迭代修正完整动作序列，并设计 metric-aligned action tokenizer | 与“tokenizer—解码器共同设计”高度相关，需进一步比较其 tokenizer 消融和 refinement 收益 |

动作候选共同强化一个判断：2026 年的 VLA 研究不再只比较离散与连续，而是把**离散语义监督、连续部署控制、并行/迭代修正和异步执行**组合起来。

### 时间序列

| 工作 | 首次公开 | 核心机制 | 为什么尚未进入精选清单 |
|---|---:|---|---|
| [TiWeaver](https://arxiv.org/abs/2606.03121) | 2026-06-02 | Graph-Guided Adaptive Tokenizer 根据时间密度与表示一致性自适应合并上下文 patch | 这里的 token 是连续 patch，不是离散码；适合作为“动态边界”证据，与 TimeTok/UniTok 的离散路线分开比较 |

### 音频

| 工作 | 首次公开 | 核心机制 | 为什么尚未进入精选清单 |
|---|---:|---|---|
| [Qwen-Audio-3.0-TTS](https://arxiv.org/abs/2607.23938) | 2026-07-27 | 12.5 Hz 低帧率 speech tokenizer 与 flow-matching 模型分阶段协同训练 | 是离散规划与连续声学生成的混合样本，需与 UniAudio 2.0 的 reasoning/reconstruction 因子化区分 |

### 推荐与检索

| 工作 | 首次公开 | 核心机制 | 为什么尚未进入精选清单 |
|---|---:|---|---|
| [Semantic Subword Tokenization](https://arxiv.org/abs/2608.22734) | 2026-08-24 | 将历史物品的固定长度 SID 合并为变长语义 subword，但保持目标端固定长度解码 | 直接讨论输入 token 负担，值得与 CapsID 的目标 ID 变长路线做双向比较 |

## 邻近连续建模反例

| 工作 | 首次公开 | 连续路线 | 对 Tokenization 命题的意义 |
|---|---:|---|---|
| [STARFlow2](https://arxiv.org/abs/2605.08029) | 2026-05-08 | 在同一因果掩码下交织 VLM 与 autoregressive normalizing flow，视觉输出无需量化即可进入缓存 | 反驳“统一因果生成必须统一成离散视觉词表” |
| [FlowInOne](https://arxiv.org/abs/2604.06757) | 2026-04-08 | 把文本、布局和编辑指令转成视觉提示，在连续视觉空间统一 flow matching | 说明统一可以通过视觉中心接口实现，但不是通用多模态理解证据 |
| [FREPix](https://arxiv.org/abs/2605.06421) | 2026-05-07 | 直接在像素空间分解低频与高频 flow，避开 VAE 表示瓶颈 | 反驳“生成一定需要先压缩到 latent/token”，代价是像素空间计算 |
| [LongCat-AudioDiT](https://arxiv.org/abs/2603.29339) | 2026-03-31 | 在 Wav-VAE 连续 waveform latent 中进行非自回归 diffusion TTS | 说明音频可绕开离散声学 token，并提示重建更好不一定带来更好的 TTS |
| [dots.tts](https://arxiv.org/abs/2606.07080) | 2026-06-05 | 连续自回归 AudioVAE latent，加 full-history flow-matching head | 提供连续 autoregressive speech foundation model 的实例 |
| [XYZFlow](https://arxiv.org/abs/2608.12276) | 2026-08-12 | 通过时间与空间两个维度缩短 flow 生成路径 | 与 tokenizer 的关系较弱，主要作为连续生成效率路线导航 |

## 弱相关候选

| 工作 | 首次公开 | 弱相关原因 |
|---|---:|---|
| [DaV-Gen](https://arxiv.org/abs/2607.08365) | 2026-07-09 | 使用 Semantic ID 作为既有前提，创新重点是 draft-and-verify 检索与验证，而非 SID 构造 |

DaV-Gen 仍有系统意义：它说明生成式检索的在线瓶颈可能来自逐 token 解码，而不只来自 ID 设计。但不能据此评价某种 tokenizer 的信息质量。

## 超窗前史

| 工作 | 首次公开 | 保留原因 |
|---|---:|---|
| [Discrete Diffusion VLA](https://arxiv.org/abs/2508.20072) | 2025-08-27 | 首次提交早于时间窗，但后续被 ICML 2026 接收；它把动作 chunk 离散化并用离散 diffusion 并行修正，是 DFM-VLA、DiscreteRTC 等路线的重要前史 |

会议接收或论文修订发生在时间窗内，不改变首次公开日期，因此该工作不计入“最近一年新增工作”。

## 晋级核心清单的门槛

候选工作满足以下条件后，才应更新主概念页：

1. 已阅读方法、关键消融和限制，而不是只依据摘要；
2. 能说明新增机制相对现有代表工作的不可替代差异；
3. 至少控制 token 数、解码方式、主干或训练数据中的主要混杂变量；
4. 明确其 token 是固定 patch、连续 latent、离散 code、Semantic ID 还是动作/结构符号；
5. 对作者报告性能保留证据边界，不构造跨论文排行榜。

## 关联页面

- [[来源-万物皆可Tokenization研究-2026]]
- [[万物皆可Tokenization：统一接口、离散化与连续建模]]
- [[跨模态连接器实现方案对照-2026]]
- [[LLM推荐系统的分层改造与评估]]

## Sources

- [KVAE](https://arxiv.org/abs/2608.05798)
- [UniWeTok](https://arxiv.org/abs/2602.14178)
- [UniAR](https://arxiv.org/abs/2606.18249)
- [EvoTok](https://arxiv.org/abs/2603.12108)
- [RoboOmni](https://arxiv.org/abs/2510.23763)
- [PhysBrain 1.5](https://arxiv.org/abs/2609.14973)
- [MolmoAct2](https://arxiv.org/abs/2605.02881)
- [Wall-OSS-0.5](https://arxiv.org/abs/2605.30877)
- [DiscreteRTC](https://arxiv.org/abs/2604.25050)
- [DFM-VLA](https://arxiv.org/abs/2603.26320)
- [TiWeaver](https://arxiv.org/abs/2606.03121)
- [Qwen-Audio-3.0-TTS](https://arxiv.org/abs/2607.23938)
- [Semantic Subword Tokenization](https://arxiv.org/abs/2608.22734)
- [STARFlow2](https://arxiv.org/abs/2605.08029)
- [FlowInOne](https://arxiv.org/abs/2604.06757)
- [FREPix](https://arxiv.org/abs/2605.06421)
- [LongCat-AudioDiT](https://arxiv.org/abs/2603.29339)
- [dots.tts](https://arxiv.org/abs/2606.07080)
- [XYZFlow](https://arxiv.org/abs/2608.12276)
- [DaV-Gen](https://arxiv.org/abs/2607.08365)
- [Discrete Diffusion VLA](https://arxiv.org/abs/2508.20072)

*最后更新：2026-10-21*
