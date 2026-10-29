## 9. 中国市场与产业格局

### 9.1 国产深度研究Agent

中国在自主研究Agent（Autonomous Research Agent）领域已形成"模型—框架—产品"三位一体的完整技术栈。以阿里巴巴Tongyi DeepResearch、Moonshot AI Kimi-Researcher和字节跳动DeerFlow 2.0为代表的国产方案，在深度研究基准上已实现对OpenAI和Google同类系统的追平乃至超越，且普遍采用全开源或计划开源策略，正在重塑全球AI开源格局。

#### 9.1.1 Tongyi DeepResearch：Qwen3-30B-A3B MoE，GRPO训练，HLE 32.9超越OpenAI o3，全开源

Tongyi DeepResearch由阿里通义实验室于2025年9月发布，基于Qwen3-30B-A3B稀疏混合专家（Sparse Mixture-of-Experts, MoE）架构构建——总参数量305亿，每Token仅激活33亿参数，通过规模极小的激活 footprint 实现与千亿级模型竞争的推理质量 [^471^]。其训练流程并非对既有聊天LLM进行简单微调，而是作为完整Agent从零开始训练，采用三阶段闭环数据飞轮：Agentic Continual Pre-training（CPT）生成长达128K tokens的合成研究轨迹；Agentic SFT Cold-Start以ReAct和IterResearch格式进行监督微调；最终阶段引入定制的Group Relative Policy Optimization（GRPO）在线策略强化学习，配合Token级策略梯度、留一法优势估计及负样本选择性过滤，以稳定非平稳Web环境下的训练 [^859^] [^875^]。

Tongyi DeepResearch支持两种推理范式：ReAct Mode用于严格评估模型核心内在能力，无需提示词工程；IterResearch-based "Heavy" Mode则通过测试时扩展（test-time scaling）解构多步研究任务为递进式研究回合，每轮仅基于上一轮最关键输出重建精简工作空间 [^859^] [^871^]。在Humanity's Last Exam（HLE）基准上，Tongyi DeepResearch以**32.9%**的得分超越OpenAI o3（24.9%）和DeepSeek-V3.1（29.8%），且在GAIA（70.9%）、xbench-DeepSearch（75.0%）、WebWalkerQA（72.2%）和FRAMES（90.6%）等多个基准上同样取得领先 [^471^]。模型权重、训练代码与推理脚本已全量开源于Hugging Face和ModelScope，API定价低至输入$0.09/M tokens [^857^] [^810^]。

#### 9.1.2 Kimi Researcher：端到端Agentic RL，K2.5 w/tools HLE 50.2%，128K上下文

Kimi Researcher是Moonshot AI于2025年6月开启灰度测试的首款自主Agent产品，其差异化特征在于端到端（end-to-end）Agentic强化学习训练——并非通过提示工程赋予模型工具能力，而是直接以强化学习目标训练Agent行为本身 [^855^]。其基座模型Kimi K2采用1万亿参数MoE架构，每Token激活约320亿参数，依托Kimi Delta Attention（KDA）与上下文缓存技术实现超长上下文窗口 [^805^]。Kimi K2.5进一步支持同时生成和协调最多100个并行AI Agent，可将复杂多阶段任务执行时间缩短最多4.5倍 [^805^]。在HLE基准上，Kimi-Researcher Pass@1得分率达**26.9%**，超越Google和OpenAI同类产品 [^864^]。Kimi K2.5 w/tools则以**50.2%**的HLE得分创造了含工具增强的完整Agent系统新高 [^864^]。Moonshot AI已计划开源Kimi-Researcher的基础预训练模型及强化学习后的模型 [^854^]。

#### 9.1.3 DeerFlow 2.0：字节跳动LangGraph-based SuperAgent，35k+ stars，Docker沙箱+并行子Agent

DeerFlow 2.0（字节跳动，2026年2月27日开源）标志着从"深度研究框架"到"SuperAgent Harness"的范式跃迁——与v1版本无共用代码的完全重写 [^800^]。其核心架构基于LangGraph多Agent编排、Docker隔离沙箱与LangChain LLM交互层，主Agent可动态生成拥有独立作用域上下文和终止条件的子Agent，子Agent尽可能并行运行 [^798^] [^801^]。DeerFlow 2.0发布后24小时内积累超过35,000 GitHub stars，一个月内突破39,000 stars与4,600 forks，并于发布次日登上GitHub Trending榜首 [^806^] [^808^] [^799^]。字节跳动在Agent领域的布局呈多点并进态势：豆包1.6系列是国内首个支持256K上下文的思考模型，Agent TARS专注多模态视觉理解，扣子空间（Coze Space）作为通用Agent开发平台支持MCP扩展，而豆包"深入研究"功能则面向终端用户提供数分钟级报告生成 [^840^] [^869^] [^834^]。

下表对国产主流深度研究Agent的技术规格进行系统对比。

| 维度 | Tongyi DeepResearch | Kimi Researcher | DeerFlow 2.0 | AutoGLM沉思 | 盘古DeepDiver |
|:---|:---|:---|:---|:---|:---|
| 开发方 | 阿里通义实验室 | Moonshot AI | 字节跳动 | 智谱AI | 华为云 |
| 基座模型 | Qwen3-30B-A3B | Kimi K2系列 | 模型无关 | GLM-4-Air-0414 | 盘古7B |
| 参数规模 | 30B总 / 3.3B激活 | 1T总 / 32B激活 | — | 32B | 7B |
| 上下文窗口 | 128K tokens | 128K+ tokens | 取决于后端模型 | 128K tokens | — |
| 训练范式 | 端到端Agent训练 + GRPO | 端到端Agentic RL | 框架层编排 | 多模型协同 + RL | 渐进式RL |
| 多Agent能力 | IterResearch递进回合 | Agent Swarm（100个并行） | 子Agent并行 + Docker沙箱 | 单Agent执行 | 单Agent |
| 开源程度 | 全开源（权重+代码+脚本） | 计划开源 | 全开源（框架） | 核心链路开源 | 闭源 |
| 产品定位 | 深度研究Agent | 深度研究Agent | SuperAgent Harness | 研究+设备操作 | 开放域信息获取 |
| 发布时间 | 2025.09 | 2025.06 | 2026.02 | 2025.03 | 2025.06 |

上表揭示了中国深度研究Agent的三条差异化路线：Tongyi DeepResearch以小规模激活参数（3.3B）实现SOTA性能，核心在于端到端Agent训练而非模型堆砌；Kimi Researcher押注端到端强化学习与超长上下文原生的差异化优势；DeerFlow 2.0则跳出"模型竞争"逻辑，以LangGraph+Docker沙箱构建可插拔Agent运行时基础设施，支持任意后端模型接入。AutoGLM沉思进一步将研究能力延伸至设备操作层，而华为盘古DeepDiver以7B小模型实现接近DeepSeek-R1的效果，体现了行业垂直化路径。这种多层次分工表明，国产Agent生态已从"单点模型竞赛"演进到"架构层—模型层—应用层"的立体竞争。

### 9.2 DeepSeek系列在数学与研究中的角色

DeepSeek是中国开源大模型浪潮的核心驱动力之一，其系列模型不仅在通用推理领域与GPT-5竞争，更在形式化数学证明、自可验证推理和Agent工具调用三个维度建立了独特技术壁垒。

#### 9.2.1 DeepSeek-Prover-V2：GRPO强化学习+子目标分解，MiniF2F突破

DeepSeek-Prover-V2是专门面向Lean 4形式化定理证明的开源模型，基于DeepSeek-V3-Base-671B进行监督微调，提供7B和671B两种参数规模 [^844^] [^845^]。其训练方法融合专家迭代（Expert Iteration）——当前最优证明器生成证明尝试，经Lean验证成功后纳入SFT数据集以训练改进模型——与强化学习驱动的子目标分解（subgoal decomposition），通过递归定理证明管道将复杂定理拆解为可管理的组件 [^855^]。在MiniF2F-test基准上通过率达**88.9%**，在PutnamBench上成功解决49/658道问题，并在新基准ProverBench（含15道AIME竞赛题）中成功解决6道 [^840^] [^894^]。

#### 9.2.2 DeepSeek-Math-V2：IMO 2025金牌级别，671B参数模型

DeepSeekMath-V2于2025年11月发布，聚焦自可验证数学推理（Self-Verifiable Mathematical Reasoning）。其核心创新在于训练准确的LLM-based验证器（verifier）作为奖励模型，激励证明生成器在最终定稿前识别并修正自身证明中的问题，同时维护验证—生成差距——随着生成器变强，自动扩展验证计算以标记新的难验证证明，形成双向强化回路 [^886^] [^887^]。在IMO 2025与CMO 2024中达到金牌级别表现，Putnam 2024得分118/120 [^886^]。

#### 9.2.3 DeepSeek-V3.1-Terminus：增强Agentic工具调用

DeepSeek-V3.1是混合推理与Agent能力的关键过渡版本，单一模型同时支持思考（reasoning）与非思考（non-thinking）模式，非推理模式下支持结构化JSON输出和严格函数调用，上下文窗口为128K tokens [^804^] [^877^]。经社区增强的DeepSeek-V3.1-Terminus变体进一步将BrowseComp从30.0提升至38.5，SimpleQA从93.4提升至96.8，函数调用能力显著增强，更适合CrewAI等Agent框架集成 [^873^]。在Agentic Tool Use基准上，DeepSeek-V3.1以671B总参数/37B激活参数的配置与GPT-5展开直接竞争 [^862^]。2026年4月发布的DeepSeek-V4系列则进一步扩展至1.6T总参数（V4-Pro）与1M上下文窗口，且华为昇腾全系列产品已完成对V4-Flash和V4-Pro的全面适配，实现"芯模技术紧密协同" [^844^]。

下表汇总了主流国产模型与国际前沿系统在深度研究基准上的量化对比。

| 基准 | Tongyi DeepResearch | OpenAI o3 | DeepSeek-V3.1 | Kimi K2 | Kimi K2.5 w/tools |
|:---|:---:|:---:|:---:|:---:|:---:|
| Humanity's Last Exam | **32.9** [^471^] | 24.9 [^471^] | 29.8 [^471^] | 18.1 [^471^] | 50.2 [^864^] |
| BrowseComp | **43.4** [^471^] | 49.7 [^471^] | 30.0 [^471^] | 14.1 [^471^] | — |
| BrowseComp-ZH | **46.7** [^471^] | 58.1 [^471^] | 49.2 [^471^] | 28.8 [^471^] | — |
| GAIA | **70.9** [^471^] | — | 63.1 [^471^] | 57.7 [^471^] | — |
| xbench-DeepSearch | **75.0** [^471^] | 67.0 [^471^] | 71.0 [^471^] | 50.0 [^471^] | — |
| WebWalkerQA | **72.2** [^471^] | 71.7 [^471^] | 61.2 [^471^] | 63.0 [^471^] | — |
| FRAMES | **90.6** [^471^] | 84.0 [^471^] | 83.7 [^471^] | 72.0 [^471^] | — |

从上表可提取两个关键结构性发现。其一，在HLE、GAIA、xbench-DeepSearch、WebWalkerQA和FRAMES五个基准上，Tongyi DeepResearch以仅33亿激活参数超越了参数规模估计达数千亿的OpenAI o3，验证了中国在MoE架构效率优化和Agent专用训练方法上的技术领先性。其二，Kimi K2.5 w/tools以50.2%的HLE得分创造了包含工具增强的完整Agent系统的新高，表明"模型+工具"协同模式较纯模型推理可带来近2.8倍的性能增益（从K2的18.1%跃升至K2.5 w/tools的50.2%），凸显了端到端Agentic训练与工具调用深度融合的战略价值。

### 9.3 监管生态与开源策略

#### 9.3.1 备案制与内容标识：748款AI服务备案，AI生成内容标识办法2025年9月施行

中国AI监管以**备案制度**为核心，形成了与美国以行业自律为主、欧盟以风险分级立法为轴截然不同的治理路径。截至2025年12月31日，累计有**748款**生成式AI服务完成备案，另有435款应用或功能完成登记；2025年全年新增446款备案和330款登记 [^828^]。算法备案的法律框架可追溯至2022年3月施行的《互联网信息服务算法推荐管理规定》，此后经《深度合成管理规定》（2023年1月）和《生成式人工智能服务管理暂行办法》（2023年8月）逐层细化 [^817^] [^823^]。

2025年9月1日起施行的《人工智能生成合成内容标识办法》引入了显式+隐式双重标识要求：显式标识需在内容页面插入可被用户感知的文字、声音或图形标记，隐式标识则在文件元数据中嵌入服务提供者信息 [^867^] [^866^]。对自主研究Agent而言，这意味着其生成的研究报告、摘要与可视化内容均需嵌入合规标识，服务提供者须在用户协议中明确标识方法，而内容传播平台在上架审核时需核验AI功能与标识完整性 [^869^] [^879^]。

#### 9.3.2 中国开源格局变化：Qwen占全球开源下载50%+，中国开源使用量从2024年1%升至~30%

开源生态层面，中国已从2024年的全球边缘参与者跃升为开源模型市场的主导力量之一。截至2026年3月，阿里云Qwen系列占全球开源模型下载量的**50%以上**，累计下载量接近10亿次；仅2026年2月单月，Qwen在Hugging Face平台产生约1.536亿次下载，超过Meta、DeepSeek等八个主要发布方总和 [^832^]。中国开源模型月下载占比达到41%，首次超越美国的36.5% [^832^]。Qwen生态已开源超过400款模型，衍生模型突破20万个，全球开发者生态规模已超越Meta Llama系列 [^832^] [^847^]。

#### 9.3.3 双轮驱动路径：开源模型权重+开源工具链，与西方"闭源模型+部分开源工具"形成对比

这一格局转变的深层动力在于中国头部厂商采取"双轮驱动"策略——**同时开源模型权重和开源工具链**，与西方"闭源模型+部分开源工具"形成结构性对比。DeepSeek和Qwen构成第一梯队，字节豆包与智谱GLM紧随其后 [^845^]。2025年11月中国AI模型全球市场份额约15%，一年前仅1% [^829^]；在日本企业开发的前10个模型中，有6个基于DeepSeek或Qwen开发，日本国立情报学研究所的LLM-jp项目亦使用Qwen整理学习数据 [^831^]。Meta于2026年放弃开源旗舰路线转向闭源Muse Spark，而中国阵营的持续开源投入可能进一步放大这一此消彼长的格局差异 [^832^]。

这种"模型+工具"双开源策略正在深刻影响全球深度研究Agent的竞争经济学：当Tongyi DeepResearch以33亿激活参数实现超越o3的性能并全量开源时，西方闭源深度研究产品的定价权和技术壁垒面临被"商品化"压力。若中国厂商持续以开源SOTA研究模型建立生态标准，深度研究Agent可能加速从差异化产品向基础设施层迁移，进而重塑全球AI价值链的利润分配结构。
