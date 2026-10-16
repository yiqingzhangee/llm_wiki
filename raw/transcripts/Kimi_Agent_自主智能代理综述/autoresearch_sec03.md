## 3. 深度研究Agent产品化竞争格局

2025至2026年间，深度研究Agent（Deep Research Agent）从一个概念验证迅速演变为多家AI实验室的核心产品赛道。OpenAI、Google、Perplexity、Moonshot AI和阿里巴巴等厂商相继推出商业化产品，形成了Prompt-based与Fine-tuned两条技术路线的正面交锋。与此同时，超过10个开源方案的涌现正在加速这一领域的"商品化"进程，竞争焦点从底层模型能力向协调层架构、企业数据整合和专用推理基础设施迁移。

### 3.1 头部产品技术架构

#### 3.1.1 OpenAI Deep Research：端到端RL训练的单Agent架构

OpenAI Deep Research采用基于o3推理模型的单Agent架构，核心特征在于端到端强化学习（End-to-End Reinforcement Learning）训练。模型并非通过预设流程或复杂提示词（Prompt Engineering）来驱动研究行为，而是在模拟环境中通过试错自主学习何时搜索、何时阅读、如何组合信息生成最终答案 [^77^][^129^]。训练分为两个阶段：train-time compute（训练时计算）阶段在复杂浏览和推理任务上优化决策序列的奖励函数；test-time compute（推理时计算）阶段则在实际运行时动态分配额外算力以改善推理链预测 [^137^]。该架构以200K上下文窗口为基础，集成了Web搜索、浏览器访问、代码解释器和文件解析器四种工具 [^108^][^123^]。在与用户交互初期，系统先进行简洁的意图澄清，随后自主制定多步研究策略并执行动态自适应迭代工作流 [^123^]。

#### 3.1.2 Gemini Deep Research：多模态融合与企业数据原生集成

Google的Gemini Deep Research以多模态Gemini模型为底座，核心差异化在于原生支持文本、图像和图表信息的统一推理 [^77^]。该产品经历了快速的技术迭代：2024年12月首次在Gemini App推出，由Gemini 1.5 Pro驱动；2025年3月升级至Gemini 2.0 Flash Thinking Experimental；2025年升级至Gemini 2.5 Pro Experimental，内部评估显示报告质量超过竞争对手2:1 [^110^]。2026年4月的重大升级标志着该产品进入企业级部署阶段——Deep Research和Deep Research Max两个版本首次支持通过单一API调用融合开放网络数据与企业私有数据，原生支持图表和信息图生成，并通过Model Context Protocol（MCP）连接任意第三方数据源 [^110^]。四个关键技术进步包括交互式研究规划（用户可审阅和修改自动生成的多步计划）、异步任务管理、大规模上下文窗口RAG集成以及高速自适应检索 [^123^]。

#### 3.1.3 Perplexity Deep Research：搜索原生的迭代检索循环

Perplexity的技术架构与其他产品存在本质差异：其核心并非以LLM为中心的研究规划，而是以迭代信息检索循环为中心的搜索原生设计。系统根据新发现不断调整检索策略，而非执行单次搜索后摘要 [^77^]。Perplexity采用混合模型路由架构，可自主为任务不同部分选择最佳底层模型：基础层使用经检索增强生成（RAG）增强的Llama 3.3 70B；推理层使用基于DeepSeek R1轻量版的Sonar Reasoning处理逻辑和规划；专业研究层使用Sonar Deep Research执行大量搜索查询和处理数百来源 [^75^][^85^]。其推理基础设施由Cerebras技术驱动，处理速度高达1,200 tokens/秒 [^81^]，这使其能够在3分钟内完成传统Agent需要30分钟的研究任务。

#### 3.1.4 Kimi-Researcher与Tongyi DeepResearch：端到端Agentic RL的国产方案

中国厂商在这一赛道选择了与西方不同的技术路径。Kimi-Researcher是Moonshot AI首款Agent产品，其核心创新在于"零结构设计"——完全依靠端到端自主强化学习（End-to-End Agentic RL）自主学习研究策略，无需复杂提示词或预设流程 [^83^]。该产品基于Kimi K系列MoE架构（1万亿总参数），通过25,000+任务、数周RL训练，在HLE基准上从8.6%基线提升至26.9% Pass@1，平均每任务执行23步推理、探索200+ URL [^84^]。

Tongyi DeepResearch由阿里通义实验室开发，采用更为系统化的三阶段训练管线：Agentic Continual Pre-training（CPT）从精选语料库和图结构知识构建大规模合成轨迹；Agentic SFT冷启动使用ReAct和IterResearch格式进行模式一致的训练；在线策略RL采用定制的Group Relative Policy Optimization（GRPO）框架，包含token级策略梯度、leave-one-out优势估计和负样本过滤，以稳定非平稳Web环境下的学习 [^79^]。其IterResearch设计每轮重新构建上下文，仅保留必要工件以缓解上下文膨胀和错误传播 [^79^]。

**表1：头部深度研究Agent技术架构对比**

| 维度 | OpenAI Deep Research | Gemini Deep Research | Perplexity Deep Research | Kimi-Researcher | Tongyi DeepResearch |
|:---|:---|:---|:---|:---|:---|
| **基础模型** | o3（早期版本）| Gemini 3.1 Pro | Llama 3.3 70B / DeepSeek R1 | Kimi K系列MoE（1T总参）| Qwen3-30B-A3B |
| **训练方法** | 端到端RL，ReAct风格循环 [^129^] | RL驱动微调，多模态融合 [^77^] | 混合模型路由，RAG增强 [^85^] | 端到端Agentic RL，零结构 [^83^] | CPT + SFT + GRPO三阶段 [^79^] |
| **上下文窗口** | 200K tokens [^108^] | 200K（超长模式）[^159^] | 128K tokens [^157^] | 128K tokens [^84^] | 128K tokens [^84^] |
| **核心工具链** | 搜索+浏览器+代码+文件 [^123^] | 搜索+MCP+图表生成 [^110^] | 搜索+多模型路由 [^75^] | 搜索+异步Rollout [^84^] | 搜索+IterResearch重建 [^79^] |
| **架构特色** | 单Agent，计划-执行-观察 | 多模态，异步任务管理 | 搜索原生，Cerebras推理 | 零结构，层次化缓冲区 | 稀疏MoE，每token 3.3B激活 |
| **关键创新** | 动态自适应迭代工作流 [^123^] | 企业私有数据+开放网络融合 [^110^] | 1,200 tokens/秒推理 [^81^] | 异步Rollout快1.5倍 [^84^] | 上下文膨胀缓解机制 [^79^] |

上表揭示了三个结构性分化。第一，训练范式分化：OpenAI、Kimi和Tongyi均采用端到端RL训练专用研究模型，而Perplexity和Google更依赖提示工程加基础模型能力的组合。第二，架构哲学分化：OpenAI坚持单Agent的"计划-执行-观察"循环，Kimi和Anthropic（Claude Code Ultra）则探索多Agent协调，后者由三个并行"探索者"Agent和一个"批评者"Agent组成，在复杂任务中比单Agent基准高出90% [^169^][^164^]。第三，上下文管理策略分化：Tongyi的IterResearch通过每轮上下文重建主动压缩历史信息，而Gemini选择扩大上下文窗口至200K并通过RAG集成管理多模态数据。

### 3.2 基准测试与能力边界

#### 3.2.1 核心基准表现矩阵

深度研究Agent的能力评估依赖多个互补基准。Humanity's Last Exam（HLE）聚焦专家级跨学科问题，无法通过简单检索解决；BrowseComp/BrowseComp-ZH测试多步开放式网络搜索能力；GAIA评估通用AI助手的工具使用、推理和多模态输入处理；DeepResearchBench包含100个PhD级研究任务，横跨22个领域 [^98^][^103^][^105^]。

**表2：核心基准测试表现矩阵（Pass@1，%）**

| Agent/Model | HLE | BrowseComp | BrowseComp-ZH | GAIA | FRAMES | xbench-DeepSearch |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| **Tongyi DeepResearch (30B-A3B)** | **32.9** [^154^] | 43.4 [^154^] | **46.7** [^154^] | **70.9** [^154^] | **90.6** [^154^] | **75.0** [^154^] |
| **Kimi K2.5 w/ tools** | **50.2** [^170^] | **78.4** [^171^] | — | — | — | — |
| **Gemini Deep Research Max** | 54.6 [^110^] | — | — | — | — | 93.3 (DeepSearchQA) [^110^] |
| **OpenAI DeepResearch** | 26.6 [^20^] | 51.5 [^20^] | 42.9 [^20^] | 67.36 [^20^] | — | — |
| **OpenAI o3** | 24.9 [^154^] | 49.7 [^154^] | 58.1 [^154^] | — | 84.0 [^154^] | 67.0 [^154^] |
| **Kimi-Researcher** | 26.9 [^84^] | — | — | — | 78.8 [^471^] | 69.0 [^471^] |
| **MiroThinker-72B** | 37.7 [^150^] | 47.1 [^150^] | 55.6 [^150^] | **81.9** [^150^] | — | — |
| **DeepSeek-V3.1** | 29.8 [^471^] | 30.0 [^471^] | 49.2 [^471^] | 63.1 [^471^] | 83.7 [^471^] | 71.0 [^471^] |

该矩阵揭示了若干关键事实。Tongyi DeepResearch以30.5B总参数（3.3B激活）在HLE、GAIA、FRAMES和xbench-DeepSearch四项基准上均取得开源SOTA，且在HLE上超越了OpenAI o3（24.9%）和OpenAI DeepResearch（26.6%）[^154^]。Kimi K2.5在工具增强场景下表现卓越，HLE达到50.2%，BrowseComp达到78.4%，领先所有竞争对手 [^170^]。Google Deep Research Max在2026年4月升级后HLE跃升至54.6%，DeepSearchQA达到93.3%（从2024年12月的66.1%大幅提升）[^110^]。MiroThinker-72B作为另一开源方案，在GAIA上以81.9%超越所有列出的专有和开源系统 [^150^]。

![HLE基准对比](hle_benchmark_chart.png)

*图1：Humanity's Last Exam基准对比。数据标注来源：Tongyi [^154^]、Kimi K2.5 [^170^]、Gemini 3 Pro [^170^]、GPT-5.2 [^170^]、OpenAI DeepResearch [^20^]、Gemini DeepResearch [^154^]、Kimi-Researcher [^84^]。*

值得注意的是，工具配置对得分的影响极为显著。Kimi K2.5在纯模型模式下HLE为30.1%，接入工具后跃升至50.2% [^170^]；同样，GLM-4.5 Heavy通过测试时扩展（Test-Time Scaling）可实现与OpenAI DeepResearch和o3相当的性能 [^889^]。这表明当前基准评估的不仅是模型本身的推理能力，更是模型与工具链协同的效率。

#### 3.2.2 Fine-tuned路线的效率优势

深度研究Agent的技术路线正经历从Prompt-based向Fine-tuned的结构性迁移。Prompt-based路线（gpt-researcher 26,443⭐、dzhng/deep-research 18,744⭐）依赖前沿模型的提示工程和工具编排，灵活性高但性能天花板受限于基础模型 [^12^]。Fine-tuned路线（Tongyi DeepResearch 18,663⭐、Kimi-Researcher）通过RL训练专门模型用于研究Agent任务，在基准上展现出更高的效率边界 [^12^]。

Tongyi DeepResearch的效率优势尤为突出：以33亿激活参数实现了超越估计数千亿参数的o3模型的HLE成绩 [^154^][^471^]。其成本效率同样显著——API定价为$0.09/百万输入tokens、$0.40/百万输出tokens（OpenRouter平台），而Kimi-Researcher定价为$0.15/百万输入tokens，比Claude Opus便宜约100倍 [^84^][^810^]。这种"小模型+专门训练"的模式挑战了"规模即能力"的传统假设，但也面临扩展性验证的挑战：当前的成功主要体现在可验证的基准任务上，在开放式、创造性研究中的优势尚待检验 [^12^]。

#### 3.2.3 幻觉与失败模式：PIES分类体系

尽管基准分数持续提升，所有深度研究Agent在可靠性方面仍存在系统性缺陷。PIES（Planning vs. Summarization × Explicit vs. Implicit）分类体系将幻觉分为四个维度 [^180^]：Explicit Summarization（声明幻觉，包括Fabrication和Misattribution）、Implicit Summarization（噪声主导，Agent未能利用已检索到的相关文档）、Explicit Planning（行动幻觉，包括计划偏离、冗余步骤和级联传播）以及Implicit Planning（限制忽视，静默忽略用户约束）[^122^]。

在六项SOTA深度研究Agent的评估中，没有系统实现稳健可靠性 [^182^]。具体而言，OpenAI在Explicit Summarization上表现出高Fabrication率（约0.15），在Explicit Planning上存在高Deviation和Redundancy（约4%），但在Implicit Planning上近乎完美（<0.05）[^122^]。Grok和Perplexity在Implicit Summarization上表现出高噪声率（约0.33），而Qwen在该维度上最为稳健（约0.23）[^122^]。对于专有Agent，超过57%的源头错误发生在早期阶段（Gemini和OpenAI），表明早期验证机制的设计至关重要 [^194^]。Gemini Deep Research Agent存在特定的"过探索"问题——易受Propagation影响（约1.7%），即基于先前幻觉声明产生级联错误 [^122^]。这些发现表明，幻觉传播和认知偏见（时间"锚定效应"和语义"同质性偏见"）是系统性缺陷的根本原因 [^182^]。

### 3.3 商业模式与市场格局

#### 3.3.1 定价矩阵：消费者订阅与API调用的双轨策略

**表3：深度研究Agent定价矩阵（截至2026年4月）**

| 产品 | 消费者层级 | 月费 | Deep Research限额 | API输入($/1M tokens) | API输出($/1M tokens) |
|:---|:---|:---:|:---|:---:|:---:|
| **ChatGPT/OpenAI** | Plus | $20 | 25次/月 [^184^] | — | — |
| | Pro | $200 | 更高限额 | — | — |
| **Gemini/Google** | Free | $0 | 5报告/月 [^112^] | — | — |
| | AI Plus | ~$10 | 12报告/天 | — | — |
| | AI Pro | $19.99 | 20报告/天 [^112^] | — | — |
| | AI Ultra | $249.99 | 120报告/天 [^112^] | — | — |
| **Perplexity** | Free | $0 | 有限查询 | — | — |
| | Pro | $20 | 20次/天 [^161^] | — | — |
| | Enterprise Pro | $40/用户 | 500次/天 | — | — |
| | Enterprise Max | $325/用户 | 无限研究 [^161^] | — | — |
| **OpenAI o3** (API) | — | — | — | $10.00 [^108^] | $40.00 [^108^] |
| **Gemini 3.1 Pro** (API) | — | — | — | $2.00 [^159^] | $12.00 [^159^] |
| **Perplexity Sonar DR** (API) | — | — | — | $2.00 [^157^] | $8.00 + $5/1K搜索 [^157^] |
| **Kimi K2 Thinking** (API) | — | — | — | $0.60 [^80^] | $2.50 [^80^] |
| **Kimi K2 0905** (API) | — | — | — | $0.40 [^80^] | $2.00 [^80^] |
| **Tongyi DeepResearch** | 开源/本地部署 | $0 | 无限 | $0.09 [^810^] | $0.40 [^810^] |

上表呈现了两个显著的定价策略分化。第一，消费者订阅与API调用的定位差异：OpenAI主要将Deep Research嵌入ChatGPT Plus/Pro界面，API访问有限且月度查询严格受限（Plus仅25次/月）[^184^]；Perplexity采取API优先策略，Sonar API支持RESTful访问和自定义参数 [^81^]；Google则在2026年4月通过Interactions API首次支持程序化访问，试图兼顾两端 [^110^]。第二，成本结构分化：一次典型的Gemini Deep Research任务（250K输入+60K输出）约花费$1.22，复杂任务（900K输入+80K输出）约$4.80 [^159^]；而中国厂商的API定价显著更低，Kimi K2 0905输入仅为$0.40/百万tokens，Tongyi DeepResearch通过OpenRouter更是低至$0.09/百万tokens [^80^][^810^]。

#### 3.3.2 深度研究的"商品化"趋势

开源生态的爆发式增长正在加速深度研究Agent的"商品化"进程。以Andrej Karpathy于2026年3月发布的autoresearch为引爆点，深度研究层已形成gpt-researcher（26,443⭐）、dzhng/deep-research（18,744⭐）、Tongyi DeepResearch（18,663⭐）和LangChain open_deep_research（11,105⭐）的四强格局 [^12^][^113^]。Hugging Face研究人员在OpenAI发布Deep Research后24小时内构建了开源替代方案，在GAIA上达到54%（OpenAI为67.36%），证明了agent框架可显著提升LLM能力，但底层模型仍是主要瓶颈 [^545^][^549^]。

这种商品化趋势正在重塑竞争逻辑。当超过10个开源方案均可实现"搜索→阅读→综合"的基础研究流程时，差异化转向两个维度：一是协调层和专用模型的效率优化——Tongyi的IterResearch上下文管理和Kimi的层次化缓冲区展示了专用架构在效率上的优势；二是企业级整合深度——MCP协议支持、异步任务管理和多模态RAG集成成为新的竞争焦点。全球AI Agent市场在2026年预计达到约110亿美元，高于2025年的约80亿美元 [^195^]，但深度研究Agent作为子品类，其独立商业价值正面临被基础模型能力吞噬的风险。

#### 3.3.3 企业级应用：MCP、异步任务与多模态RAG的竞争焦点

2026年的企业级竞争围绕三个技术整合点展开。第一，MCP（Model Context Protocol）协议整合：Gemini Deep Research Max在2026年4月升级中原生支持MCP，可连接任意第三方数据源，将开放网络数据与企业私有数据通过单一API调用融合 [^110^]。这一能力使Google在企业级场景中获得结构性优势——其Deep Research Max支持最多160个来源、最长60分钟异步任务，并与Google Workspace、Search和Finance生态深度整合 [^158^][^110^]。

第二，异步任务管理：Gemini Deep Research采用异步架构高效处理多个并发任务 [^123^]，而Moonshot Kimi K2.5支持最多100个并行子Agent，可将复杂多阶段任务执行时间缩短最多4.5倍 [^805^]。Anthropic的多Agent架构（Three Explorers + One Critic）在复杂任务中比单Agent基准高出90%，但其生产部署失败率高达41%-86.7%，79%源于协调问题 [^169^]。这表明多Agent协调的理论优势与实际部署可靠性之间存在显著鸿沟。

第三，多模态RAG集成：Gemini Deep Research的原生图表和信息图生成能力 [^110^]，以及Kimi K2.5在视觉+文本融合上的持续投入，反映了企业用户对"研究输出可视化"的强需求。2026年，43%的组织正在积极考虑采用Agentic AI，80%的客户服务组织计划应用生成式和Agentic AI以提升生产力 [^186^]，这一企业采纳浪潮正在倒逼产品在数据安全、工作流集成和输出格式三个维度上加速迭代。

**表4：2024-2026年深度研究Agent产品迭代时间线**

| 时间 | OpenAI | Google/Gemini | Perplexity | Moonshot AI | 阿里巴巴 |
|:---|:---|:---|:---|:---|:---|
| **2024.12** | — | Gemini App首次推出Deep Research（Gemini 1.5 Pro）[^110^] | — | — | — |
| **2025.02** | — | — | — | — | dzhng/deep-research开源（18,744⭐）[^425^] |
| **2025.03** | — | 升级至Gemini 2.0 Flash Thinking [^110^] | Sonar Deep Research发布 [^157^] | — | 智谱AutoGLM沉思发布 [^822^] |
| **2025.06** | — | — | — | Kimi-Researcher发布，HLE 26.9% [^83^][^864^] | 豆包"深入研究"上线 [^834^] |
| **2025.07** | Agent Mode推出，支持多步任务 [^184^] | — | — | — | — |
| **2025.09** | — | — | — | — | Tongyi DeepResearch发布，30B开源 [^79^] |
| **2025.10** | o3 Deep Research API公开发布 [^108^] | — | — | — | — |
| **2025.12** | — | Interactions API推出，首次程序化访问 [^110^] | — | — | — |
| **2026.01** | — | — | — | Kimi K2.5发布，HLE 50.2% [^170^] | — |
| **2026.02** | — | Gemini 3.1 Pro发布，ARC-AGI-2 77.1% [^110^] | — | — | DeerFlow 2.0开源，39,000⭐ [^806^] |
| **2026.03** | — | — | — | — | — |
| **2026.04** | — | Deep Research / Deep Research Max发布，HLE 54.6% [^110^] | — | — | — |

上表的时间线揭示了三个竞争动态。其一，Google采取了最激进的产品迭代节奏，在18个月内完成从首次发布到企业级Max版本的跨越，且每次升级均伴随底层模型的同步更新。其二，中国厂商的发布密度在2025年中期至2026年初达到峰值，Kimi-Researcher和Tongyi DeepResearch的先后发布标志着国产方案从"跟随"转向"并行竞争"。其三，开源与闭源产品的节奏差异明显：开源工具（dzhng/deep-research、DeerFlow 2.0）通常在闭源产品发布后数月内快速涌现，并在GitHub上获得爆发式关注，但核心模型能力仍由闭源系统（OpenAI o3、Gemini 3.1 Pro）主导。

综合来看，深度研究Agent的竞争格局正处于从"模型能力竞争"向"系统效率竞争"过渡的关键节点。Fine-tuned路线在基准效率上已证明其优越性，但企业级部署的复杂性、幻觉控制的可靠性以及多Agent协调的稳定性仍是未解决的核心挑战。随着MCP协议成为数据连接的事实标准和开源方案持续侵蚀基础功能的定价空间，2026年下半年的竞争预计将围绕"专用模型+垂直整合+企业工作流嵌入"的三维能力展开。
