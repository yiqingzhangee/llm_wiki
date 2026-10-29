# 1. 自主研究Agent技术全景与分类框架

## 1.1 自主研究Agent的定义与边界

### 1.1.1 核心特征：自主决策、可度量输出与迭代优化循环

自主研究Agent（Autonomous Research Agent）指能够在人类仅提供高层意图或约束的条件下，自主完成假设生成、实验设计、代码修改、结果评估与策略迭代的闭环系统的总称。其与传统的AI辅助工具（如代码补全、搜索增强生成）存在本质差异：传统工具是**被动响应式**的——用户提出具体问题，AI返回单次输出；而自主研究Agent是**主动探索式**的——系统在固定目标下自主决定下一步行动，通过可度量的反馈信号持续优化，直至满足终止条件或耗尽计算预算 [^2^]。

这一范式的核心创新由Andrej Karpathy于2026年3月发布的`autoresearch`项目所引爆。该项目以仅630行Python代码、MIT许可证的极简形态，展示了一种全新的研究自动化模式：Agent在单GPU上自主运行机器学习训练实验，48小时内完成约700次实验，发现20个可叠加的优化，将验证损失（validation loss）从0.862415降至0.858039，且所有优化成功迁移至depth-24大模型，实现11%速度提升 [^2^]。更重要的是，Shopify CEO Tobi Lütke将其应用于内部模板引擎后，93次自动化提交实现了53%的渲染速度提升 [^3^]，验证了该模式跨领域的通用性。

从架构层面看，Karpathy autoresearch的核心设计仅包含三个文件：`prepare.py`（数据准备与常量，只读）、`train.py`（模型与训练循环，Agent可修改）、以及最关键的`program.md`——一种以自然语言形式编写的"研究组织代码"（research organization code）[^4^][^5^]。人类研究者只需编辑markdown文件定义研究策略、约束和评估标准，Agent读取后自主执行。这种"人类设计实验空间，Agent填充实验点"的劳动分工，实质上创造了一种**可执行的研究方法论**[^5^]。Ry Walker指出，"你在programming program.md，而非编辑Python"——这降低了研究自动化的门槛，使非程序员也能用自然语言定义研究策略。

自主研究Agent区别于传统AI辅助工具的三项核心特征可归纳为：

**第一，自主决策（Autonomous Decision-Making）**。系统在没有人类逐轮介入的情况下，自主决定下一步行动。Karpathy的autoresearch没有orchestration脚本，没有`run.py`——LLM本身就是automation layer [^7^]。用户只需在仓库中启动Claude Code或Codex并让其读取`program.md`，Agent自行运行循环。

**第二，可度量输出（Measurable Output）**。每次迭代产生可量化的评估指标，构成反馈信号。autoresearch的核心循环是"棘轮循环"（ratchet loop）：Agent提出假设→修改`train.py`→运行固定5分钟实验→比较验证指标→如果改进则git commit保留，否则git revert丢弃→无限重复 [^7^]。固定时间预算使实验间可比，无论Agent做了什么修改。

**第三，迭代优化循环（Iterative Optimization Loop）**。系统通过持续的"编辑-评估-保留/回退"循环逼近最优解，而非单次推理。典型的overnight运行（约8-12小时）可完成80-100次实验，其中约15-20次改进被保留 [^9^]。这种贪婪爬山（greedy hill-climbing）策略虽然简单，但在实践中被证明有效。

### 1.1.2 三类划分：实验循环、深度研究与科学发现Agent

基于自主程度、输出形态和应用目标，当前自主研究Agent可系统性地划分为三类。下表从核心循环、输出形态、代表性系统和2026年成熟度四个维度进行对比。

**表1：自主研究Agent三类划分与对比**

| 维度 | 实验循环Agent（Experimental Loop Agent） | 深度研究Agent（Deep Research Agent） | 科学发现Agent（Scientific Discovery Agent） |
|------|----------------------------------------|--------------------------------------|------------------------------------------|
| **核心循环** | edit→benchmark→keep/revert [^7^] | search→read→synthesize [^77^] | ideate→experiment→write papers [^56^] |
| **输出形态** | 代码修改、性能指标提升 | 结构化研究报告、引用文献 | 完整学术论文、原创假设 |
| **自主性层级** | 中等（依赖人类预设代码框架） | 中高（自主规划研究策略） | 高（端到端自主发现） |
| **代表系统** | Karpathy autoresearch [^2^]; AutoKernel [^28^]; Evo [^39^] | OpenAI Deep Research [^77^]; Gemini Deep Research [^110^]; Tongyi DeepResearch [^79^] | AI Scientist v2 [^56^]; Aletheia [^764^]; AlphaEvolve [^1^] |
| **开源生态** | 丰富（72k+ stars, 20+衍生变体） | 活跃（gpt-researcher 26k⭐, dzhng/deep-research 18.7k⭐）[^12^] | 有限（主要为企业/学术闭源） |
| **2026年成熟度** | 原型验证→社区扩展期 | 产品化竞争期，差异化收窄 | 实验室突破期，可靠性待验证 |
| **基准表现** | HPO基准：Centaur 0.9739 val_bpb [^16^] | HLE：Tongyi 32.9%, Gemini Max 54.6% [^154^][^110^] | PaperBench最佳21%; SciCode 7.7-15.38% |

实验循环Agent以Karpathy autoresearch为原点，核心机制是将代码修改与可重复实验绑定，通过版本控制（git commit/revert）实现"安全探索"。其适用范围已扩展到GPU kernel优化（AutoKernel，1,206⭐）[^28^]、chess engine Elo提升、API延迟优化、prompt engineering等任意具有可度量scalar metric的领域 [^45^]。但该类Agent存在明确的硬边界：仅支持单文件修改空间、同步顺序执行无法并行探索多个分支、指标范围有限且存在过拟合风险 [^46^]。

深度研究Agent是2025-2026年产品化竞争最为激烈的赛道。OpenAI Deep Research基于o3模型的端到端强化学习训练，通过ReAct风格的"计划-执行-观察"循环完成研究任务 [^77^]；Google Gemini Deep Research在2026年4月升级为Deep Research Max，首次支持单一API调用融合开放网络数据与企业私有数据 [^110^]；Moonshot AI的Kimi-Researcher采用端到端自主强化学习与"零结构设计"，每任务平均探索200+ URL [^83^][^84^]；阿里Tongyi DeepResearch以30.5B参数（每token激活3.3B）的MoE架构，通过GRPO在线策略强化学习训练，在Humanity's Last Exam基准上以32.9%超越OpenAI o3的24.9% [^79^][^154^]。

科学发现Agent代表了最高级别的自主研究形态，其目标是端到端地完成从假设生成、实验设计、数据分析到论文撰写的全过程。Sakana AI的AI Scientist v2通过progressive agentic tree search管理实验探索，向ICLR研讨会提交了三篇完全自主的稿件，其中一篇以6.33/10分通过同行评审——这是首个完全由AI生成的稿件通过同行评审的案例 [^56^]。DeepMind的Aletheia在FirstProof挑战中自主解决了6/10个未发表的研究级数学问题 [^763^]。Google的AlphaEvolve则展示了LLM引导的进化搜索可以发现新的数学构造，在Strassen矩阵乘法算法等领域实现改进 [^1^]。然而，该类Agent的可靠性仍面临严峻挑战：AI Scientist v2的独立评估显示57%假数据率，42%实验因编码错误失败 [^3^]。

三类Agent的成熟度分化明显。深度研究Agent已进入产品化竞争阶段，全球AI Agent市场预计在2026年达到约110亿美元 [^195^]，但差异化空间正在收窄——主要竞争维度已从"是否能生成报告"转向"模型微调质量"和"上下文管理效率"。相比之下，实验循环Agent和科学发现Agent仍处于技术门槛较高的早期阶段，平台层基本空缺，竞争参与者较少，可能蕴含更高的长期回报潜力。

## 1.2 技术演进脉络

### 1.2.1 从AutoML/HPO到Agent-driven AutoML：代码级自主修改

传统超参数优化（Hyperparameter Optimization, HPO）——如TPE（Tree-structured Parzen Estimator）、CMA-ES（Covariance Matrix Adaptation Evolution Strategy）、SMAC（Sequential Model-based Algorithm Configuration）——在固定搜索空间内已被证明具有成熟的优化能力。Ferreira等人（ELLIS Institute Tübingen、University of Freiburg、KIT、Prior Labs）在2026年3月发表的arXiv:2603.24647论文中，首次在相同24小时GPU预算下对9种HPO方法进行了头对头比较 [^13^]。核心发现是：在固定搜索空间内，经典HPO方法始终优于纯LLM-based agent。最佳方法排名为：Centaur (0.9763) > TPE (0.9768) > SMAC (0.9778) > CMA-ES (0.9785) > Karpathy Agent Code (0.9814) [^14^]。

然而，LLM-based agent的独特优势在于**无约束代码编辑能力**。Karpathy Agent Code直接编辑训练源代码，突破了固定搜索空间的限制，能够发现人类研究者未预见的架构修改（如删除代码带来的性能提升）[^11^]。这一范式转变的本质是从"参数级优化"演进为"代码级自主修改"——Agent不再仅在预定义的超参数网格中搜索，而是直接修改实现训练逻辑的源代码。

论文提出的Centaur混合方法揭示了这一演进方向的最优形态：将CMA-ES的完整内部状态（mean vector μ、step-size σ、covariance matrix C）共享给LLM，在所有实验中取得最佳结果，且仅0.8B规模的LLM已足以超越所有经典和纯LLM方法 [^16^]。Centaur仅将LLM用于约30%的trials，其余由CMA-ES独立 propose [^17^]。这一发现表明："LLM最有效的方式是作为经典优化器的补充而非替代"[^54^]，标志着AI研究从"替代人类/替代经典算法"转向"增强算法"的新范式。

### 1.2.2 从单轮推理到迭代优化循环：Test-time Scaling崛起

LLM领域正在经历计算扩展范式的根本转变：从预训练阶段的大规模计算投入，转向推理阶段的动态计算分配，即Test-time Scaling（推理时计算扩展）[^788^][^816^]。OpenAI的o1、DeepSeek R1、Kimi K1.5等模型展示了通过在推理时增加计算资源而不需要重新训练或扩展参数来提升性能的能力 [^816^]。

这一范式转变直接赋能自主研究Agent的核心能力。Gemini 2.5 Deep Think在响应生成过程中融合并行思考技术，在Olympiad数学（USAMO 2025）等挑战性基准上达到SOTA性能，在2025年IMO上获得35/42分，成为首个由IMO协调员官方认证达到金牌标准的AI系统 [^814^]。Gemini Deep Research在Humanity's Last Exam基准上从2024年12月的7.95%提升至2025年6月的26.9%（更高计算量下32.4%），显示了推理时计算扩展的显著效果 [^814^]。

对自主研究Agent而言，Test-time Scaling意味着"更多智能来自思考更久而非模型更大"。AlphaProof通过在Lean定理证明器中学习寻找形式化证明，结合预训练语言模型与AlphaZero强化学习算法，在2024年IMO上达到银牌水平（28/42分），但其训练涵盖1亿条形式化数学问题，成功证明或反驳仅占CPU使用的1.5% [^772^][^776^]。TRACE框架通过聚合多步证据而非瞬时信号来确定终止推理时机，平均减少25-30%的推理token使用量，同时保持准确率在全长的1-2%范围内 [^813^]。

自主研究Agent的经济学正在被这一范式转变重塑。Aletheia的推理时缩放、Gemini Deep Think的并行推理、AutoResearch的固定时间预算实验，均指向同一趋势：Agent服务的定价模型可能从"per-token"转向"per-task"或"per-experiment"，因为实际成本取决于迭代轮数而非单次推理长度。

### 1.2.3 从单Agent到多Agent协调：动态拓扑、角色编排与图工作流

自主研究Agent的技术架构正在经历从单Agent独立执行到多Agent协同系统的分化。这一演进的动力在于：复杂研究任务往往需要多种专业能力（文献检索、代码编写、数据分析、批评评估）的协同，单一Agent难以在所有维度上同时保持高水准。

当前多Agent协调已形成六种主要模式 [^25^]。**角色分配模式**（CrewAI）通过预定义Agent角色（研究员、写手、编辑）和任务目标，由Crew协调器编排执行顺序，适合内容流水线等结构化场景。**对话编排模式**（AutoGen/AG2）允许多个Agent在共享对话中交互，通过GroupChat发言选择器决定下一轮发言者，适合代码评审、学术讨论等迭代任务。**图工作流模式**（LangGraph）以有向循环图（状态机）定义节点（动作）和边（连接），在复杂任务（8+步骤）完成率上达到62%，其核心优势在于通过Checkpointing和Time Travel将瞬态Agent执行转换为可回放、可检查的状态机 [^25^][^350^]。

更具前瞻性的模式包括**树搜索模式**（AI Scientist v2），维护研究状态树，每个节点代表潜在研究方向，使用Explorer、Critic、Executor、Synthesizer四种Agent并行探索 [^323^]；**动态拓扑模式**（MetaGen、DyLAN），在推理时自适应生成角色空间和协作拓扑，MetaGen的token成本相比静态基线实现数量级降低 [^314^]，DyTopo在所有16个backbone×dataset设置中均为最佳方法，相比最强非DyTopo基线平均提升+6.09分 [^741^]；以及**网络涌现模式**（OpenAgents），采用持久Agent网络范式，Agent可发现彼此能力并自主协作 [^345^]。

协议标准化是多Agent协调的技术基础设施。MCP（Model Context Protocol，Anthropic 2024年11月发布）标准化Agent如何获取数据和调用工具，月下载量达9700万次；A2A（Agent-to-Agent Protocol，Google 2025年4月发布）标准化Agent之间的任务委托机制，截至2026年4月获150+组织支持，核心仓库超22,000 GitHub stars [^321^]。两大协议均于2025年底捐赠给Linux Foundation的Agentic AI Foundation，成为事实上的行业标准。

然而，多Agent系统的"涌现能力"并非自动发生。arXiv:2510.05174的研究显示，GPT-4.1多Agent组中20%表现出显著的跨Agent协同，而Llama-3.1-8B组仅2.6% [^796^]。有效的多Agent协调需要每个Agent能推理其他Agent的状态、意图和能力——这需要高级Theory-of-Mind（心理理论）能力，而当前开源小模型缺乏此能力。这意味着"多Agent架构"的优势在当前开源模型上可能无法完全兑现。

## 1.3 2026年生态格局概览

### 1.3.1 关键参与者地图：个人、学术机构、大厂与开源社区

2026年自主研究Agent生态呈现四层参与者结构，各层在技术定位、资源投入和商业模式上存在显著差异。下表系统梳理了主要参与者及其核心贡献。

**表2：自主研究Agent关键参与者地图（2026年）**

| 层级 | 参与者 | 核心贡献/代表产品 | 技术路线 | 关键数据 |
|------|--------|-----------------|----------|----------|
| **个人/独立开发者** | Andrej Karpathy | autoresearch（72.1k⭐）; AgentHub愿景 [^31^] | 极简主义、program.md模式 | 630行Python，48h/700实验 |
| | Ry Walker Research | 市场地图、工具对比分析 | 生态系统追踪 | 覆盖30+工具 |
| | Udit Goenka | 通用化Claude Code skill（608⭐）[^33^] | 跨领域autoresearch | 扩展至任意可度量指标 |
| **学术机构** | Google DeepMind | AlphaEvolve [^1^]; Aletheia [^764^]; AlphaProof [^772^] | RL+进化算法+形式验证 | IMO银牌，6/10研究问题 |
| | Sakana AI | AI Scientist v2 [^56^] | Agentic tree search | 首篇AI论文通过同行评审 |
| | ELLIS/Uni Freiburg/KIT | Ferreira et al. HPO基准 [^13^]; Centaur [^16^] | Hybrid LLM+经典优化器 | 0.8B LLM超越所有基线 |
| | UC Berkeley | SkyPilot多GPU扩展 [^41^]; GAIN-RL [^755^] | 并行化+训练效率 | 16 GPU 9x吞吐量 |
| | NTU/ICML | Autogenesis Protocol [^752^] | 自进化Agent协议 | 五原子操作符框架 |
| **大型科技公司** | OpenAI | Deep Research（o3驱动）[^77^]; Agents SDK [^359^] | 端到端RL+单Agent | HLE 26.6%，200K上下文 |
| | Google | Gemini Deep Research/Max [^110^]; ADK [^299^]; A2A协议 | 多模态+MCP+A2A原生 | HLE 54.6%，160来源 |
| | Anthropic | Claude Code [^33^]; MCP协议 | 编码Agent+工具协议 | MCP 9700万月下载 |
| | Moonshot AI | Kimi-Researcher [^83^]; K2.5 w/tools [^170^] | 端到端agentic RL+MoE | HLE 50.2%，100并行子Agent |
| | 阿里/Tongyi | Tongyi DeepResearch（18.7k⭐）[^79^]; Qwen3-30B-A3B | Agentic CPT+GRPO+IterResearch | HLE 32.9%，30B/3.3B MoE |
| | 字节跳动 | DeerFlow 2.0（35.3k⭐）[^304^] | 模块化SuperAgent | GitHub Trending #1 |
| | Microsoft | Microsoft Agent Framework 1.0 [^356^] | AutoGen+Semantic Kernel统一 | .NET/Python双栈 |
| **开源社区** | gpt-researcher | 26.4k⭐，Prompt-based深度研究 [^12^] | 前沿模型+提示工程 | 20+来源综合 |
| | dzhng/deep-research | 18.7k⭐，<500行TypeScript [^127^] | 极简递归搜索 | breadth+depth参数 |
| | LangGraph/CrewAI | 生产级编排框架 [^25^] | 图状态机/角色团队 | 34%企业架构引用 |
| | awesome-autoresearch | WecoAI维护，追踪20+变体 [^24^] | 生态策展 | pi-autoresearch, autokernel等 |

个人开发者层以Karpathy为原点，其autoresearch项目在一周内从21,000⭐增长至62,000+⭐（截至2026年3月底），衍生出pi-autoresearch（通用metric扩展，3,591⭐）、autoresearch-mlx（Apple Silicon移植）、autokernel（GPU kernel优化，1,206⭐）、autoresearch-at-home（分布式协作，469⭐）、N-Autoresearch（多GPU并行）等20余个社区变体 [^24^]。Karpathy进一步提出了AgentHub愿景——"GitHub is for humans. AgentHub is for agents"——一个面向AI agent swarm的协作平台原型，无main branch、无PRs、无merges，只有 sprawling DAG of commits [^31^]。

学术机构层贡献了基础算法创新和基准验证。DeepMind的AlphaEvolve将LLM引导的进化搜索从数学函数推广到代码库和算法优化 [^1^]；Aletheia在IMO-ProofBench高级版本上达到95.1% SOTA准确率 [^764^]。Ferreira等人的系统性基准研究则为整个领域提供了学术纠偏：即使在无约束代码编辑设置下，LLM agent也未超越经典方法，且frontier model scaling未能关闭这一差距 [^54^]。

大厂层呈现"模型+基础设施+协议"的三维竞争。OpenAI依托ChatGPT生态用户基数将Deep Research嵌入已有工作流，但API访问受限（Plus用户25次/月）[^184^]。Google则通过MCP原生支持、Workspace生态整合和2026年4月的Interactions API程序化访问，定位企业级基础设施 [^110^]。Moonshot AI和阿里代表了中国厂商的差异化路径：Kimi-Researcher以$0.15/1M输入tokens的极低API定价和端到端RL训练的"零结构"Agent切入市场 [^84^]；Tongyi DeepResearch完全开源模型权重，Qwen系列占全球开源下载50%+，在HLE、GAIA、BrowseComp等多个基准上达到开源SOTA [^154^]。

### 1.3.2 两条技术路线分化：Prompt-based vs Fine-tuned专用模型

2026年自主研究Agent领域已形成两条清晰分化的技术路线。**Prompt-based路线**以前沿通用模型（GPT-5、Claude、Gemini）配合精心设计的提示词和工具编排为核心，代表产品包括gpt-researcher（26,443⭐）、dzhng/deep-research（18,744⭐）、open_deep_research等 [^12^]。其优势在于灵活性高、快速迭代、无需专门训练；劣势在于性能天花板受限于基础模型能力，提示词工程复杂且难以复现。

**Fine-tuned专用模型路线**则通过强化学习（GRPO、Reinforce++等）训练专门用于研究Agent任务的模型。代表产品包括Tongyi DeepResearch（基于Qwen3-30B-A3B，三阶段训练：Agentic持续预训练→Agentic SFT冷启动→在线策略RL with GRPO）[^79^]、Kimi-Researcher（端到端agentic RL，25,000+训练任务，数周RL训练）[^84^]、以及SkyworkAI Autogenesis等。其优势在于基准测试表现更好、推理效率更高、行为更稳定；劣势在于架构锁定（architecture lock-in）——模型权重和训练管线与特定厂商绑定，社区难以复现或扩展。

Fine-tuned路线正在取得优势。Tongyi尽管模型规模（30B总参数/3.3B激活）远小于OpenAI o3和Gemini 3 Pro等frontier模型，却在BrowseComp（43.4%）、GAIA（70.9%）和HLE（32.9%）上领先基准测试 [^154^]。Kimi K2.5在工具增强场景下表现尤为卓越，HLE-Full w/tools达到50.2%，显著超越Gemini 3 Pro（45.8%）和GPT-5.2（45.5%）[^170^]。这一趋势表明：在自主研究Agent这一特定领域，**专门训练的中小规模模型**可以匹敌甚至超越**通用frontier模型**。

两条路线的分化背后，是自主研究Agent成本结构的根本性转变。从"训练时计算"到"推理时计算"的范式转移正在重塑Agent经济学 [^788^]。深度研究Agent主要依赖搜索API和提示工程，进入门槛低，导致竞争快速 commoditization（商品化）——已有10+开源方案，差异化空间收窄。实验循环Agent和科学发现Agent则需要GPU、代码执行安全沙箱、可复现的度量系统，进入门槛高得多，竞争较少，平台层完全空缺。

展望未来，Ry Walker Research预测到2027年分布式autoresearch（多Agent协调）将成为默认范式，而非单Agent循环；深度研究Agent领域将围绕2-3家拥有专门微调模型的赢家整合 [^12^]。然而，Gartner预测到2027年底超过40%的Agentic AI项目将被取消，失败原因包括标准不成熟、可靠性担忧和不可预测的成本 [^346^]。自主研究Agent能否跨越从原型到生产的鸿沟，将是2026-2027年的核心产业命题。
