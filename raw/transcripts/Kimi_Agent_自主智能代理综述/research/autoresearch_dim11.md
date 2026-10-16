# 研究维度11：技术范式演进与未来趋势

> 研究时间：2025-2026  
> 覆盖范围：多Agent协调、RL训练与专用模型、自进化系统、Test-time Scaling、计算范式转变、2026-2028趋势预测

---

## 目录
1. [从单Agent到多Agent：协调、协议与涌现能力](#1-从单agent到多agent协调协议与涌现能力)
2. [从Prompt工程到Fine-tuned Research Models](#2-从prompt工程到fine-tuned-research-models)
3. [从静态工作流到自进化系统](#3-从静态工作流到自进化系统)
4. [Test-time Scaling：推理时计算扩展](#4-test-time-scaling推理时计算扩展)
5. [计算范式转变](#5-计算范式转变)
6. [2026-2028趋势预测](#6-2026-2028趋势预测)
7. [反面论点与争议](#7-反面论点与争议)
8. [关键引用来源汇总](#8-关键引用来源汇总)

---

## 1. 从单Agent到多Agent：协调、协议与涌现能力

### 1.1 技术范式转变的核心驱动力

LLM正从单轮对话响应器进化为能够规划、批判、编写代码和与外部工具交互的通用问题解决器。组织多个LLM实例成为Multi-Agent Systems（MAS）是自然下一步，其中专门化的角色协作分解复杂任务并交叉检查中间结论。

Claim: 多Agent协作在多个基准测试中显著超越单Agent提示方法。LLM-based MAS可以模拟人类社会协作机制，在工业和多利益相关者场景中支持分布式决策和协调。[^351^]
Source: arXiv
URL: https://arxiv.org/html/2506.17784v2
Date: 2025
Excerpt: "teams of LLM agents can collaboratively solve harder problems than any individual agent, giving rise to an emergent form of collective intelligence. This emergent capability hinges critically on the design of inter-agent communication topologies."
Context: Cascading Flexible Multi-Agent Collaboration via Next-Agent Prediction
Confidence: high

### 1.2 通信协议标准之争：MCP、A2A、AGP

当前Agent生态系统正处于协议标准化的关键阶段。四个主要协议占据不同层级的互操作性需求：

| 协议 | 定位 | 交互模式 | 发布方 |
|------|------|----------|--------|
| **MCP** (Model Context Protocol) | 工具调用与上下文获取 | JSON-RPC客户端-服务器 | Anthropic (2024.11) |
| **A2A** (Agent-to-Agent Protocol) | Agent间任务委托 | 基于HTTP/SSE的P2P | Google (2025.04) |
| **ACP** (Agent Communication Protocol) | 本地多Agent消息传递 | REST-native多部分消息 | 开源社区 |
| **ANP** (Agent Network Protocol) | 开放网络Agent发现 | DID + JSON-LD去中心化 | 开源社区 |
| **AGP** (Autogenesis Protocol) | Agent自进化生命周期 | 双层架构：RSPL + SEPL | NTU / ICML 2026 |

Claim: MCP和A2A解决不同问题且互补——MCP标准化Agent如何获取数据和调用工具，A2A标准化Agent之间如何协调任务。大多数生产级多Agent系统同时使用两者。[^709^]
Source: Intuz Blog
URL: https://www.intuz.com/blog/mcp-vs-a2a
Date: 2026-04-13
Excerpt: "MCP handles tool access at the individual agent level... A2A handles coordination at the system level: agents discover each other via Agent Cards, delegate tasks, and exchange results across frameworks and vendors."
Context: 多Agent协议比较分析
Confidence: high

Claim: Autogenesis Protocol (AGP)是首个针对Agent**自进化**的协议标准，与MCP/A2A形成互补。AGP通过双层架构（RSPL资源层 + SEPL自进化层）实现安全的Agent自我修改。[^766^]
Source: Flowtivity AI Blog
URL: https://flowtivity.ai/blog/autogenesis-self-evolving-agent-protocol/
Date: 2026-04-18
Excerpt: "MCP standardises how models call tools. A2A standardises how agents communicate with each other. But neither protocol addresses what happens when an agent needs to modify its own components — its prompts, its tools, its memory — during or between task executions."
Context: Autogenesis Protocol技术分析
Confidence: high

### 1.3 动态拓扑优化：DyTopo

DyTopo是一个manager-guided的多轮多Agent框架，通过语义匹配在每轮动态构建有向通信图。每个Agent提供自然语言描述的"key"（能提供什么）和"query"（需要什么），DyTopo通过语义匹配Q-K来诱导每轮的通信拓扑。

Claim: DyTopo在所有16个backbone×dataset设置中均为最佳方法，相比最强非DyTopo基线平均提升+6.09分。在更难的数学基准上提升尤为显著（Math-500最高+17.14，Omni-Math最高+15.72）。[^741^]
Source: arXiv (DyTopo论文)
URL: https://arxiv.org/html/2602.06039v1
Date: 2026-02-05
Excerpt: "DyTopo is the best method in all 16 backbone×dataset settings, improving over the strongest non-DyTopo baseline by 0.90-17.14 points (mean +6.09)."
Context: DyTopo: Dynamic Topology Routing for Multi-Agent Reasoning via Semantic Matching
Confidence: high

Claim: DyTopo的拓扑从早期轮的广泛探索性路由，过渡到后期的选择性验证连接，最终到依赖最小化的解决方案组装图，提供了可解释的协调轨迹。[^753^]
Source: TheMoonlight.io 论文评述
URL: https://www.themoonlight.io/de/review/dytopo-dynamic-topology-routing-for-multi-agent-reasoning-via-semantic-matching
Date: 2026-02-07
Excerpt: "DyTopo's topology transitions from broad, exploratory routing in early rounds to more selective, verification-focused connections in later stages, and finally to a dependency-minimal graph for solution assembly, offering an interpretable coordination trace."
Context: DyTopo动态拓扑分析
Confidence: high

Claim: DyTopo在token使用和延迟方面均优于固定范围基线。在HumanEval上仅消耗AgentScope的48% token（9,453 vs 19,520），同时达到最高准确率（92.07%），平均仅需2.6轮即可收敛。[^741^]
Source: arXiv (DyTopo论文附录)
URL: https://arxiv.org/html/2602.06039v1
Date: 2026-02-05
Excerpt: "DyTopo achieves the highest accuracy (92.07%) while consuming only 48% of the tokens required by AgentScope (9,453 vs. 19,520). This efficiency is driven by our Manager-controlled halting mechanism."
Context: DyTopo效率分析
Confidence: high

### 1.4 涌现能力与集体智能

多Agent系统的"涌现"是指系统产生的信息仅能从联合变量中获得，而非任何单独变量。研究使用信息论框架（基于Rosas等人的协同度量）来评估LLM多Agent系统是否表现出真正的高阶协同。

Claim: GPT-4.1多Agent组中20%表现出显著的跨Agent协同（G3 synergy），而Llama-3.1-8B组仅2.6%。模型容量对支持关于其他Agent的推理至关重要——低容量LLM的多Agent系统显示出的大部分"涌现"只是时间维度上的伪涌现，而非生产性的跨Agent互补。[^796^]
Source: arXiv
URL: https://arxiv.org/html/2510.05174v1
Date: 2025-10-05
Excerpt: "significant synergy present in 2.6% Llama groups compared with 20% of gpt-4.1 groups... Lower ToM reasoning capacity in Llama-3.1-8B may explain the lack of useful, goal-directed cross-agent synergy."
Context: Emergent Coordination in Multi-Agent Language Models
Confidence: high

Claim: SwarmBench评估显示，当前LLM在去中心化群体智能约束下表现出初级的协调能力，但在严格局部信息限制下难以实现稳健的集体行为。[^790^]
Source: arXiv
URL: https://arxiv.org/html/2505.04364v4
Date: 2025-05-07
Excerpt: "While some rudimentary coordination is observed, our results indicate that current LLMs significantly struggle with robust long-range planning and adaptive strategy formation under the uncertainty inherent in these decentralized scenarios."
Context: Benchmarking LLMs' Swarm Intelligence
Confidence: high

### 1.5 角色与拓扑共进化：MetaGen

MetaGen是一个无需训练的框架，在推理时同时适应角色空间和协作拓扑，不更新基础模型权重。它引入Architect Agent来合成和修订查询条件化的角色规范，形成可控的动态角色池。

Claim: MetaGen在所有评估基准上均优于强多Agent基线。其token成本相比GPTSwarm（1.4×10^7）和DyLAN（2.2×10^7）仅为1.2×10^6，实现了数量级的成本降低。[^314^]
Source: arXiv
URL: https://arxiv.org/html/2601.19290v1
Date: 2026-01-27
Excerpt: "MetaGen introduces an Architect that synthesizes and revises query-conditioned role specifications to form a controllable dynamic role pool. It then constructs an initial execution graph around a minimal backbone and iteratively updates role prompts and structural decisions using lightweight feedback signals, without modifying backbone weights."
Context: MetaGen: Self-Evolving Roles and Topologies for Multi-Agent LLM Reasoning
Confidence: high

---

## 2. 从Prompt工程到Fine-tuned Research Models

### 2.1 RL训练范式的演进：GRPO及变体

强化学习在后训练阶段增强LLM推理能力方面取得了显著进展。核心算法演进路径：

- **PPO** (Proximal Policy Optimization): 需要单独的价值函数/批评模型，计算开销大
- **DPO** (Direct Preference Optimization): 无需RL的偏好优化替代方案
- **GRPO** (Group Relative Policy Optimization): 完全移除价值函数，通过组内采样序列的平均奖励作为基线估计优势——特别适合大规模参数模型的微调
- **Reinforce++**: 进一步改进策略梯度估计
- **DAPO**: 引入裁剪机制和动态采样
- **Dr.GRPO**: 去除长度和标准差归一化项，减少优化偏差

Claim: GRPO通过组内相对奖励估计优势，完全消除了昂贵的critic模型需求，使大规模RL训练更加高效。DeepSeek-R1使用GRPO实现了与OpenAI o1相当的推理性能，且训练成本显著降低。[^886^][^893^]
Source: arXiv (多篇论文引用)
URL: https://arxiv.org/pdf/2509.19580
Date: 2025
Excerpt: "GRPO estimates advantages within groups, thereby eliminating the need for an expensive value function model while maintaining performance comparable to PPO."
Context: DeepSeek-R1 GRPO技术核心
Confidence: high

Claim: DeepSeek-R1-Zero证明，仅通过纯强化学习（无需SFT）即可实现模型的自我修正和推理能力。训练过程中模型自主发现且纠正推理错误，CoT长度随时间自然增长，出现了所谓的"aha moment"。[^900^]
Source: AIONDA Blog
URL: https://www.aionda.blog/en/posts/deepseek-r1-grpo-reasoning-efficiency
Date: 2026-01-31
Excerpt: "DeepSeek-R1-Zero demonstrated that model self-correction and reasoning capabilities can be achieved through reinforcement learning alone, without supervised fine-tuning... the AI developed a chain of thought (CoT) within the reward framework of reinforcement learning."
Context: DeepSeek-R1训练分析
Confidence: high

### 2.2 GAIN-RL：自动选择训练样本加速微调

Qinsi Wang等（UC Berkeley / Duke）开发了GAIN-RL，通过自动选择训练样本来加速RL微调。核心洞察：token表示向量之间的余弦相似度决定了梯度更新的大小。

Claim: GAIN-RL将学习加速2.5倍。通过对Qwen 2.5 1.5B、Qwen 2.5 7B和Llama 3.2 3B的GRPO微调实验，按angle concentration排序样本的训练比随机排序显著更快。[^755^]
Source: DeepLearning.ai The Batch
URL: https://www.deeplearning.ai/the-batch/new-technique-auto-selects-training-examples-to-speed-up-fine-tuning/
Date: 2025-09-24
Excerpt: "GAIN-RL generally accelerated learning by a factor of 2.5. The authors separately fine-tuned Qwen 2.5 1.5B, Qwen 2.5 7B, and Llama 3.2 3B using the GRPO reinforcement learning algorithm with examples ordered according to their angle concentration."
Context: 训练效率优化新方向
Confidence: high

### 2.3 Qwen3-30B-A3B：专用模型效率的突破

Qwen3-30B-A3B采用MoE架构，仅激活3B参数（总参数30B），通过RL训练在多个基准上超越了更大的全参数模型。

Claim: Qwen3-30B-A3B-Thinking-2507在AIME25上达到85.0分，超越Gemini2.5-Flash-Thinking（72.0），接近Qwen3-235B-A22B（81.5）。在LiveCodeBench v6上达到66.0分，显著优于同规模的Gemini2.5-Flash-Thinking（61.2）。[^859^]
Source: Dev.to 深度评测
URL: https://dev.to/czmilo/qwen3-30b-a3b-thinking-2507-reasoning-model-in-depth-review-5fgf
Date: 2025-07-31
Excerpt: "Qwen3-30B-A3B-Thinking-2507 achieves significant improvements in math, coding, and logical reasoning, scoring 85.0 on AIME25... Mathematical Reasoning: Achieves 85.0 on AIME25, surpassing Gemini2.5-Flash-Thinking"
Context: Qwen3-30B-A3B推理模型评测
Confidence: high

Claim: Qwen3-30B-A3B在本地部署友好，Q6_K量化版本可在单张RTX 5090上运行，AIME 2024达到73%-87%准确率，AIME 2025达到70%-90%，与Claude 3.7 Sonnet和DeepSeek-R1具有竞争力。[^851^]
Source: arXiv
URL: https://arxiv.org/html/2512.23029v1
Date: 2025-12-28
Excerpt: "The locally hosted Qwen3-30B-A3B (Q6_K) achieves 73%–87% pass accuracy depending on sampling strategy, remaining competitive with strong proprietary systems (e.g., Claude 3.7 Sonnet and DeepSeek-R1 at 80%)."
Context: 消费级硬件上的Qwen3-30B-A3B基准测试
Confidence: high

### 2.4 专用研究模型vs通用模型

一个关键趋势是研究Agent开始使用专门微调的研究模型，而非依赖通用 frontier model。

Claim: RL方法可以用远少于SFT的数据量达到更好效果。小米的R1-AQA仅用38k后训练样本，就在MMAU Test-mini上达到64.5%的SOTA性能，显著超越SFT方法。[^746^]
Source: arXiv
URL: https://arxiv.org/pdf/2503.11197v2
Date: 2025
Excerpt: "With only 38k post-training samples, RL significantly outperforms supervised fine-tuning (SFT), indicating that RL-based approaches can be effective without large datasets."
Context: 音频问答RL案例研究
Confidence: high

Claim: 强化学习在多个领域超越监督微调。RL增强的Qwen2-Audio-7B-Instruct在MMAU Test-mini上达到64.5%的SOTA准确率，而SFT方法在相同数据量下表现明显更差。[^746^]
Source: arXiv
URL: https://arxiv.org/pdf/2503.11197v1
Date: 2025
Excerpt: "We leverage the group relative policy optimization (GRPO) algorithm to Qwen2-Audio-7B-Instruct, and our experiments demonstrated state-of-the-art performance on the MMAU Test-mini benchmark, achieving an accuracy rate of 64.5%."
Context: RL Outperforms SFT: Audio QA Case Study
Confidence: high

---

## 3. 从静态工作流到自进化系统

### 3.1 Autogenesis：自进化Agent协议

Autogenesis Protocol（AGP）提出了一种根本不同的构建AI Agent系统的方式——Agent不仅执行任务，还能安全地重写自己的指令、工具和行为。这是ICML 2026的投稿论文。

AGP的核心是SEPL（Self-Evolution Protocol Layer）的五个原子操作符：
1. **Reflect (ρ)**：分析执行轨迹识别失败假设
2. **Select (σ)**：将诊断假设转化为具体修改提案
3. **Improve (ι)**：通过RSPL接口应用修改产生候选状态
4. **Evaluate (ε)**：测试候选状态是否满足目标标准和安全不变量
5. **Commit (κ)**：仅当评估显示改进且安全约束满足时才接受变更

Claim: Autogenesis支持多种优化策略映射到统一的SEPL接口：反射优化器（自然语言分析）、TextGrad（文本梯度）、GRPO和Reinforce++（强化学习视角）。AGP在GPQA、AIME、GAIA和LeetCode上均显示出持续改进。[^752^]
Source: arXiv (Autogenesis论文)
URL: https://arxiv.org/pdf/2604.15034
Date: 2026 (ICML submission)
Excerpt: "The protocol supports multiple strategies through its standardised five-operator interface. The default is a Reflection Optimiser that uses natural language analysis. It also supports TextGrad... GRPO, and Reinforce++."
Context: Autogenesis: A Self-Evolving Agent Protocol
Confidence: high

Claim: 组合进化（evolve prompt+solution）始终优于单一策略。对于gpt-4.1在AIME24上，evolve prompt达到33.3%，evolve solution达到36.7%，而组合方法达到40.0%。自我进化在模型能力和基准难度留有改进空间时最有效。[^752^]
Source: arXiv (Autogenesis论文)
URL: https://arxiv.org/pdf/2604.15034
Date: 2026
Excerpt: "For gpt-4.1 on AIME24, evolve prompt reaches 33.3% and evolve solution 36.7%, whereas the combined approach reaches 40.0%... self-evolution is most effective when both model capability and benchmark difficulty leave room for improvement."
Context: Autogenesis实验结果
Confidence: high

### 3.2 MetaGen：角色与拓扑共进化

MetaGen的核心创新是在推理时动态生成和重写查询条件化的角色规范，并在最小骨干周围实例化约束的执行图。

Claim: MetaGen通过两种编辑实现任务内进化：(1) role prompt rewrite——针对在当前实例下持续无益的低效用角色，使用反馈痕迹修订角色的系统/用户模板；(2) prior-filtered edge exploration——在当前先验和结构约束下保守地更新拓扑。经验证的角色会被固化并存储到Role Cache中供后续实例复用。[^314^]
Source: arXiv
URL: https://arxiv.org/html/2601.19290v1
Date: 2026-01-27
Excerpt: "Role prompt rewrite targets a low-utility role whose messages are consistently unhelpful... MetaGen first filters candidate non-critical edges using current priors and structural constraints... then selectively deactivates or swaps one edge."
Context: MetaGen方法细节
Confidence: high

### 3.3 AutoResearchClaw与Bilevel Autoresearch

Claim: AutoResearchClaw是一个23阶段的自主研究管道，包含多Agent辩论和自愈执行。MetaClaw跨运行学习系统可捕获失败教训并转化为可复用技能，实验显示stage retry rate降低24.8%，refinement cycles减少40%，整体鲁棒性提升18.3%。[^761^]
Source: Paperclipped Blog
URL: https://www.paperclipped.de/en/blog/autoresearchclaw-autonomous-research-paper-agent/
Date: 2026-03-24
Excerpt: "MetaClaw... captures lessons from failures and warnings, converts them into reusable skills, and injects those skills into all 23 pipeline stages on subsequent runs... 24.8% reduction in stage retry rate, 40% fewer refinement cycles, and 18.3% overall robustness improvement."
Context: AutoResearchClaw元学习分析
Confidence: medium

Claim: Bilevel Autoresearch提出用autoresearch循环优化autoresearch循环本身。外层循环通过生成和注入新的搜索机制作为Python代码来元优化内层循环。在Karpathy的GPT预训练基准上，meta-autoresearch外层循环相比标准内层循环单独运行实现了5倍改进（-0.045 vs -0.009 val_bpb）。[^756^]
Source: arXiv
URL: https://arxiv.org/html/2603.23420v1
Date: 2026-03-24
Excerpt: "On Karpathy's GPT pretraining benchmark, the meta-autoresearch outer loop achieves a 5× improvement over the standard inner loop alone (-0.045 vs. -0.009 val_bpb), while parameter-level adjustment without mechanism change yields no reliable gain."
Context: Bilevel Autoresearch: Meta-Autoresearching Itself
Confidence: high

### 3.4 自适应多Agent编排：Task-Adaptive DAG Orchestration

Claim: 与MetaGen、DyTopo等动态拓扑方法相比，基于DAG结构属性的形式化拓扑路由提供了可证明的终止保证和显式的成本-准确率Pareto分析，这些元素没有单一现有系统能够整合。[^742^]
Source: arXiv
URL: https://www.arxiv.org/pdf/2602.16873
Date: 2026
Excerpt: "Our work is distinguished by the combination of (i) formal topology routing grounded in DAG structural properties, (ii) provable termination guarantees for the synthesis protocol, and (iii) explicit cost-accuracy Pareto analysis."
Context: Task-Adaptive Multi-Agent Orchestration论文
Confidence: high

---

## 4. Test-time Scaling：推理时计算扩展

### 4.1 Gemini Deep Think：并行推理技术

Gemini 2.5 Deep Think是Google开发的一种新颖推理方法，在响应生成过程中自然融合并行思考技术。它使Gemini能够创造性地产生多个假设并仔细批判它们，然后得出最终答案。

Claim: Deep Think在Olympiad数学（USAMO 2025）、竞争性编码（LiveCodeBench）和多模态（MMMU）等挑战性基准上达到了SOTA性能。Gemini 2.5 Deep Think在2025年IMO上获得35/42分，成为首个由IMO协调员官方认证达到金牌标准的AI系统。[^814^][^767^]
Source: Google技术报告 / Riemann-Bench论文
URL: https://arxiv.org/pdf/2507.06261 / https://arxiv.org/html/2604.06802v1
Date: 2025-2026
Excerpt: "Deep Think enables Gemini to creatively produce multiple hypotheses and carefully critique them before arriving at the final answer, achieving state-of-the-art performances in challenging benchmarks such as Olympiad math (USAMO 2025), competitive coding (LiveCodeBench), and multimodality (MMMU)."
Context: Gemini 2.5技术报告
Confidence: high

Claim: Gemini Deep Research在Humanity's Last Exam基准上从2024年12月的7.95%提升到2025年6月的SOTA 26.9%（更高计算量下32.4%），显示了推理时计算扩展的显著效果。[^814^]
Source: Google技术报告
URL: https://arxiv.org/pdf/2507.06261
Date: 2025-06
Excerpt: "performance of Gemini Deep Research on the Humanity's Last Exam benchmark has gone from 7.95% in December 2024 to the SoTA score of 26.9% and 32.4% with higher compute (June 2025)."
Context: Gemini Deep Research进展
Confidence: high

### 4.2 AlphaProof：RL + 形式验证

AlphaProof是DeepMind开发的AlphaZero-inspired Agent，通过在Lean定理证明器中学习寻找形式化证明，在2024年IMO上达到银牌水平。

Claim: AlphaProof通过自动形式化8000万条陈述用于训练，在Lean中结合预训练语言模型与AlphaZero强化学习算法。它解决了2024 IMO 6题中的4题（包括竞赛中最难的问题，仅5名人类参赛者解决），得分28/42（银牌标准）。[^772^][^775^]
Source: Nature / Codeforces / DeepMind
URL: https://www.natureasia.com/en/info/press-releases/detail/9147
Date: 2025-11-12
Excerpt: "AlphaProof is designed to prove mathematical statements. It learns to find these proofs through reinforcement learning, after auto-formalizing 80 million statements to use for training... AlphaProof, combined with a geometry-solving system called AlphaGeometry, solved four out of six problems from the competition."
Context: AlphaProof Nature论文
Confidence: high

Claim: AlphaProof的训练制度基于AlphaZero方法论，涵盖1亿条形式化数学问题——规模显著超越此前努力。成功证明或反驳仅占CPU使用的1.5%，凸显了自动定理证明的巨大难度。[^776^]
Source: arXiv
URL: https://arxiv.org/html/2410.15700v1
Date: 2024
Excerpt: "Success Rate: Only 1.5% of CPU usage resulted in successful proofs or disproofs, highlighting the substantial difficulty of automated theorem proving."
Context: InternLM2.5-StepProver论文
Confidence: high

### 4.3 Aletheia：迭代验证与数学研究Agent

Aletheia是Google DeepMind的数学研究Agent，能够迭代生成、验证和修订端到端的自然语言解决方案。

Claim: Aletheia由三个主要来源驱动：(i) Gemini Deep Think的高级版本用于解决极难推理问题，(ii) 从Olympiad级别延伸到PhD级别练习的新颖推理时扩展定律，(iii) Google Search和Web浏览等密集工具使用。在IMO-ProofBench高级版本上达到95.1%的SOTA准确率（无互联网访问）。[^764^]
Source: arXiv (Aletheia论文)
URL: https://math.berkeley.edu/~fengt/Aletheia.pdf
Date: 2026
Excerpt: "Aletheia, without internet access, has achieved a new state-of-the-art performance of 95.1% accuracy on the advanced version of IMO-ProofBench... on an internal benchmark (FutureMath Basic) of PhD-level exercises, Aletheia also demonstrated state-of-the-art performance."
Context: Towards Autonomous Mathematics Research (Aletheia)
Confidence: high

Claim: Aletheia在FirstProof挑战中（严格零人工干预协议、一周提交窗口）自主解决了6/10个未发表的研究级问题。专家评审认为6份提交中的大多数在 minor revision 后可发表。[^763^]
Source: LetsDataScience
URL: https://letsdatascience.com/news/aletheia-advances-autonomous-agentic-mathematical-research-5ad201a8
Date: 2026-04-19
Excerpt: "Google DeepMind's Aletheia, driven by Gemini 3 Deep Think, autonomously solved 6/10 unpublished, research-level problems in the FirstProof challenge within the allowed one-week window."
Context: Aletheia自主数学研究进展
Confidence: high

### 4.4 TRACE：高效的推理时扩展

TRACE（Temporal Reasoning Aggregation for Convergent Exit）是一个无需训练的框架，通过聚合多步证据而非瞬时信号来确定何时终止推理。

Claim: TRACE平均减少25-30%的推理token使用量，同时保持准确率在全长的1-2%范围内。它在OlympiadBench、MATH500、AIME24/25、AMC23和GPQA-D上均优于现有动态推理方法。[^813^]
Source: arXiv
URL: https://arxiv.org/abs/2604.17304
Date: 2026-04-19
Excerpt: "TRACE reduces reasoning token usage by 25-30% on average while maintaining accuracy within 1-2% of full-length reasoning, consistently outperforming existing dynamic reasoning methods."
Context: Efficient Test-Time Scaling via Temporal Reasoning Aggregation
Confidence: high

### 4.5 推理时扩展的分类学

当前的推理时扩展方法可分为两大轴：

1. **序列扩展**（Sequential approaches）：通过延长单个推理轨迹的思考步骤来扩展CoT深度。代表：o1, DeepSeek R1, Qwen QwQ, Kimi K1.5
2. **并行扩展**（Parallel approaches）：通过增加轨迹数量和聚合来扩展。代表：Self-Consistency, Best-of-N, REBASE, Mixture of Agents

Claim: 即使禁用显式思考过程，非推理模型配合推理时方法也可以在低预算设置下达到Pareto前沿。s1模型（仅1000个问题+从o1类模型蒸馏的推理轨迹微调）通过budget forcing展示了清晰的推理时扩展行为。[^793^][^816^]
Source: arXiv
URL: https://arxiv.org/pdf/2504.09858v1 / https://arxiv.org/html/2507.14419v1
Date: 2025
Excerpt: "simple yet overlooked, achieves competitive performance without requiring additional training, reward signals, or process supervision... s1 model demonstrates clear test-time scaling behavior."
Context: Reasoning Models Can Be Effective Without Thinking / Simple TTS Analysis
Confidence: high

---

## 5. 计算范式转变

### 5.1 从训练时计算到推理时计算

LLM领域正在经历计算扩展范式的根本转变：从预训练阶段的大规模计算投入，转向推理阶段的动态计算分配。

Claim: 当前LLM越来越多地通过在推理时分配大量推理来取得成功，这被称为test-time scaling。OpenAI的o1、DeepSeek R1、Kimi K1.5、Qwen3和Grok-4等模型通常依赖大规模RL和大量样本，而更简单的微调方法如STILL-2、s1和LIMO也展示了这一方向。[^788^]
Source: arXiv
URL: https://arxiv.org/pdf/2508.15260
Date: 2025
Excerpt: "Current LLMs increasingly succeed by allocating very large amounts of reasoning at inference, a paradigm we call test-time scaling... representative models include o1, DeepSeek R1, Kimi K1.5, Qwen3, and Grok-4."
Context: 相关论文引用综述
Confidence: high

Claim: 存在从扩展预训练到扩展test-time compute的范式转变。OpenAI的o1模型展示了这一转变，通过在推理时增加计算资源而不需要重新训练或参数扩展来提升性能。DeepSeek-R1证明LLM可以通过纯RL过程自然地增加test-time compute并展示一致的性能提升。[^816^]
Source: arXiv
URL: https://arxiv.org/html/2507.14419v1
Date: 2025-07-19
Excerpt: "Recently, there has been a paradigm shift in scaling compute for large language models, transitioning from scaling pre-training to scaling test-time compute. OpenAI's o1 models exemplify this shift."
Context: It's Not That Simple: An Analysis of Simple Test-Time Scaling
Confidence: high

### 5.2 预算强制（Budget Forcing）技术

Budget forcing是一种test-time策略，通过干预解码过程来限制或扩展模型的推理阶段。如果模型生成的思考token超过预定义阈值T_max，则强制终止思考阶段；如果要延长思考，则追加"Wait"token提示模型继续推理。

Claim: Budget forcing与纯SFT结合存在挑战——SFT实际上会降低较小LLM的性能。RL可以缓解verbose responses、语言混合和冗余响应等问题。GRPO-λ动态调整奖励策略，在低正确率组避免损害CoT质量的长度惩罚，在高正确率组维持长度惩罚提升效率，在GSM8K、GPQA、MATH-500等基准上平均提升1.48%准确率同时减少47.3%的CoT序列长度。[^817^][^895^]
Source: arXiv
URL: https://arxiv.org/html/2510.21398v1 / https://arxiv.org/abs/2505.18086
Date: 2025
Excerpt: "GRPO-λ... dynamically adjusts the reward strategy by monitoring the correctness ratio among completions within each query-sampled group... improves average accuracy by 1.48% while reducing CoT sequence length by 47.3%."
Context: Budget Forcing + RL优化
Confidence: high

### 5.3 从单次推理到迭代优化循环

现代AI系统正在从单次前向传播范式转向迭代优化循环——无论是推理时的多轮思考，还是Agent系统的实验-评估-改进闭环。

Claim: AlphaEvolve展示了LLM引导的进化搜索可以发现新的数学构造，在广泛的问题集合上重新发现最佳已知解并在几个案例中改进它们。它从进化单个数学函数推广到进化整个代码库和更广泛的算法优化任务。[^1^]
Source: arXiv (多篇论文引用)
URL: https://arxiv.org/html/2603.15914v1
Date: 2026-03-06
Excerpt: "AlphaEvolve demonstrated that LLM-guided evolutionary search can discover new mathematical constructions, rediscovering best-known solutions across a broad collection of problems and improving on them in several cases."
Context: AI辅助研究实践指南
Confidence: high

Claim: AI Scientist v2消除了对人类编写代码模板的依赖，使用progressive agentic tree search方法论管理实验探索。它向ICLR研讨会提交了三篇完全自主的稿件，其中一篇通过了同行评审——这是首个完全由AI生成的稿件通过同行评审的案例。[^56^]
Source: arXiv
URL: https://arxiv.org/abs/2504.08066
Date: 2025-04-10
Excerpt: "Notably, one manuscript achieved high enough scores to exceed the average human acceptance threshold, marking the first instance of a fully AI-generated paper successfully navigating a peer review."
Context: AI Scientist v2论文
Confidence: high

---

## 6. 2026-2028趋势预测

### 6.1 行业领袖与研究机构的时间线预测

| 专家/机构 | AGI/超智能预测 | 关键里程碑 | 来源 |
|-----------|---------------|-----------|------|
| **Sam Altman (OpenAI)** | AGI 2025-2027, 超智能2028年底 | 依赖推理模型扩展和数据中心建设 | India AI Summit 2026 |
| **Dario Amodei (Anthropic)** | 超人类AI 2026-2027 | "数据中心里的天才国家"风险 | Machines of Loving Grace 2025 |
| **Elon Musk (xAI)** | 2026年底超越个体人类智能 | 聚焦 surpassing individual human intelligence | xAI Event, Nov 2025 |
| **Demis Hassabis (DeepMind)** | AGI 2030-2035 | "最后一英里"最难；警告不要过度承诺 | MIT Technology Review 2025 |
| **Yoshua Bengio (Mila)** | 不确定 | 对齐问题未解决；呼吁核武级治理 | Montreal AI Debate 2025 |
| **Ben Goertzel (ASI Alliance)** | AGI 2027-2028最可能 | 2026年可能但非大概率 | 2026预测 |
| **Ilya Sutskever (SSI)** | 十年内 | 扩展假设仍然是可靠的实证指南 | NeurIPS 2024 |
| **Aggregate Expert Surveys** | 50%高阶机器智能 ~2047 | 全面超越人类所有任务50% ~2047 | AI Impacts Survey |
| **Metaculus Community** | 强AGI前兆 2030年代初 | 近期更新延长了中位数预测 | Metaculus AGI Question |

Claim: 主要AI公司CEO对快速进展越来越有信心。Sam Altman从2024年11月的"进展速度持续"转变为2026年1月"我们现在确信知道如何构建AGI"。Dario Amodei表示"我在未来2-3年内接近强大能力的信心比以往任何时候都高"。Demis Hassabis从"最快10年"变为"大概3-5年"。[^826^]
Source: 80,000 Hours
URL: https://80000hours.org/ai/guide/when-will-agi-arrive/
Date: 2026-04-15 (Published March 2025)
Excerpt: "OpenAI's Sam Altman: Shifted from saying in November 'the rate of progress continues' to declaring in January 'we are now confident we know how to build AGI'... Anthropic's Dario Amodei: Stated in January 'I'm more confident than I've ever been that we're close to powerful capabilities... in the next 2-3 years'."
Context: AGI时间线深度分析
Confidence: high

### 6.2 具体里程碑预测

Claim: 按照当前在软件工程和计算机使用任务上的进展趋势，AI完成人类专家需要几秒钟的任务从2020年开始，到2024年已提升到约一小时。如果趋势持续，到2028年将达到数周。后2024年的趋势可能更快，每4个月翻倍。[^826^]
Source: 80,000 Hours
URL: https://80000hours.org/ai/guide/when-will-agi-arrive/
Date: 2026-04-15
Excerpt: "On this set of software engineering & computer use tasks, in 2020 AI was only able to do tasks that would typically take a human expert a couple of seconds. By 2024, that had risen to almost an hour. If the trend continues, by 2028 it'll reach several weeks."
Context: AI能力时间线外推
Confidence: medium (extrapolation)

Claim: 到2027年，分布式autoresearch（多Agent协调）将成为默认范式，而非单Agent循环。Deep research领域将围绕2-3家拥有专门微调模型的赢家整合。[^12^]
Source: Ry Walker Research
URL: https://rywalker.com/research/autoresearch-tools
Date: 2026-04-14
Excerpt: "By 2027, distributed autoresearch (multi-agent coordination) will be the default, not single-agent loops. The deep research agent space will consolidate around 2-3 winners with specialized fine-tuned models."
Context: Autoresearch工具市场分析
Confidence: medium

Claim: 2025年底到2026年的Autoresearch关键里程碑包括：Karpathy的autoresearch repo达到72k stars；一周内产生了MLX port、Windows port、GPU kernel variant、distributed variant、generic-metric variant等衍生项目。核心模式（agent修改代码→测量结果→保留或回退）简单但orchestration层（program.md作为自然语言Agent规格、固定预算实验、追加式日志）才是创新所在。[^12^]
Source: Ry Walker Research
URL: https://rywalker.com/research/autoresearch-tools
Date: 2026-04-14
Excerpt: "Karpathy's framing shift is the real innovation — program.md as 'research org code' turns natural language into agent orchestration... The pattern generalizes immediately — within 7 days: MLX port, Windows port, GPU kernel variant, distributed variant, generic-metric variant."
Context: Autoresearch生态系统分析
Confidence: high

Claim: 2026年潜在里程碑：AI agent加入劳动力市场（Sam Altman 2025年1月预测）。更强大的AI Agent开始出现，处理跨多个业务领域的复杂多步骤任务（Jensen Huang 2024年11月描述为"需要训练、评估和防护栏的数字员工"）。[^828^]
Source: The Road to AGI and Beyond
URL: https://roadtoagi.netlify.app/
Date: 2026
Excerpt: "In 2025 we may see the first AI agents join the workforce... AI agents... just like digital employees you have to train them... evaluate them... guardrail them... - Jensen Huang"
Context: AGI路线图综合预测
Confidence: medium

### 6.3 可能的技术突破点

1. **自主科学发现Agent的成熟化**：AI Scientist v2已实现同行评审通过，Aletheia解决了Erdős提出的开放问题。2027年可能看到AI生成的论文达到会议级而非workshop级质量。

2. **多Agent协调的标准化**：MCP+A2A+AGP协议栈的成熟将使自进化多Agent系统从实验走向生产。

3. **推理时计算的民主化**：Qwen3-30B-A3B等高效MoE模型使推理时扩展可在消费级硬件上实现。

4. **形式化数学与AI的深度融合**：AlphaProof/Aletheia范式将扩展到更多科学领域。

5. **计算范式的大规模迁移**：从训练时计算到推理时计算的转变将重塑AI基础设施需求。

Claim: 国际AI安全报告2026指出进展不均衡、长任务性能下降、专家对2028-2030专业领域进展存在分歧。[^822^]
Source: Deep Projects AI
URL: https://deep-projects.ai/artificial-superintelligence-in-2026-what-researchers-are-saying/
Date: 2026-03-09
Excerpt: "The International AI Safety Report 2026 highlights uneven advances, declining performance on longer tasks, and expert disagreement on specialized domain progress by 2028–2030."
Context: 2026年AI研究者观点汇总
Confidence: high

---

## 7. 反面论点与争议

### 7.1 对快速AGI预测的质疑

- **Gary Marcus (NYU)** 认为需要架构范式转变，仅靠扩展不足以实现AGI，预测"数十年后"。
- **Yoshua Bengio** 强调对齐问题未解决，呼吁核武级治理。
- **Metaculus社区** 已将超智能中位数预测调整到约2034年，反映自主编码进展慢于预期。
- **专家调查 aggregate** 对50%高阶机器智能的预测仍是约2047年，与公司CEO的乐观预测形成鲜明对比。

### 7.2 多Agent系统的实际局限

- **SwarmBench评估** 显示当前LLM在严格去中心化约束下难以实现稳健集体行为。
- **Llama-3.1-8B实验** 表明低容量模型的多Agent协同大部分是伪涌现而非真正的跨Agent互补。
- **AI Scientist v2的局限**：被接受的论文包含LSTM概念归属错误和定义不准确；3篇投稿中只有1篇通过，且为workshop级别而非conference级别。

### 7.3 Test-time Scaling的边界

- **知识密集型任务** 上test-time scaling效果有限。在SimpleQA和FRAMES基准上，增加推理计算对事实性问答的提升很小，且可能增加幻觉率。[^885^]
- **Overthinking问题**：推理模型常常继续推理超出正确回答所需，导致token效率低下。
- **Budget forcing的局限**：纯SFT+budget forcing会降低较小LLM性能，需要RL来稳定训练。

### 7.4 形式化方法的瓶颈

- AlphaProof解决某些IMO问题需要长达3天，远超竞赛时间限制。
- 自动定理证明的成功率极低（仅1.5%的CPU使用产生成功证明）。
- 组合数学是AlphaProof的弱点。

### 7.5 自进化系统的安全担忧

- Autogenesis Protocol的安全保证依赖于评估操作符的质量——如果evaluator没发现问题，commit gate也不会阻止。
- 对抗性自我修改未被现有协议充分解决。
- 长期运行生产部署中的进化稳定性尚未被验证。

---

## 8. 关键引用来源汇总

| # | 来源 | URL | 日期 | 核心发现 |
|---|------|-----|------|---------|
| 1 | Autogenesis论文 | https://arxiv.org/pdf/2604.15034 | 2026 (ICML) | 自进化Agent协议，支持TextGrad/GRPO/Reinforce++ |
| 2 | DyTopo论文 | https://arxiv.org/html/2602.06039v1 | 2026-02 | 动态拓扑路由，平均+6.09分，48% token成本 |
| 3 | MetaGen论文 | https://arxiv.org/html/2601.19290v1 | 2026-01 | 角色与拓扑共进化，token成本数量级降低 |
| 4 | AlphaProof (Nature) | https://www.natureasia.com/en/info/press-releases/detail/9147 | 2025-11 | IMO银牌，自动形式化8000万陈述 |
| 5 | Aletheia论文 | https://math.berkeley.edu/~fengt/Aletheia.pdf | 2026 | IMO-ProofBench 95.1%，FirstProof 6/10 |
| 6 | Gemini 2.5技术报告 | https://arxiv.org/pdf/2507.06261 | 2025-06 | Deep Think并行推理，IMO金牌35/42 |
| 7 | TRACE论文 | https://arxiv.org/abs/2604.17304 | 2026-04 | 推理时扩展效率提升25-30% token减少 |
| 8 | AI Scientist v2 | https://arxiv.org/abs/2504.08066 | 2025-04 | 首个AI生成论文通过同行评审 |
| 9 | Bilevel Autoresearch | https://arxiv.org/html/2603.23420v1 | 2026-03 | 元优化autoresearch循环，5倍改进 |
| 10 | AutoResearchClaw | https://www.paperclipped.de/en/blog/autoresearchclaw | 2026-03 | 23阶段管道，MetaClaw跨运行学习 |
| 11 | DeepSeek-R1 / GRPO | 多篇arXiv论文 | 2025 | 纯RL实现自我修正推理，无需critic模型 |
| 12 | Qwen3-30B-A3B评测 | https://dev.to/czmilo/qwen3-30b-a3b | 2025-07 | MoE架构，AIME25 85.0分，本地部署友好 |
| 13 | GAIN-RL | https://www.deeplearning.ai/the-batch/ | 2025-09 | 自动样本选择，2.5倍训练加速 |
| 14 | AGI时间线分析 | https://80000hours.org/ai/guide/ | 2026-04 | CEO预测vs专家调查对比 |
| 15 | MCP/A2A/AGP比较 | 多篇论文和博客 | 2025-2026 | 协议分层互补架构 |
| 16 | 多Agent涌现研究 | https://arxiv.org/html/2510.05174v1 | 2025-10 | GPT-4.1 vs Llama-3.1-8B协同对比 |
| 17 | Autoresearch生态 | https://rywalker.com/research/autoresearch-tools | 2026-04 | 2027分布式autoresearch默认预测 |
| 18 | Test-time Scaling分析 | https://arxiv.org/html/2507.14419v1 | 2025-07 | 从训练时到推理时计算范式转变 |
| 19 | Budget Forcing + RL | https://arxiv.org/html/2510.21398v1 | 2025-10 | 稳定推理长度优化 |
| 20 | 任务自适应DAG编排 | https://www.arxiv.org/pdf/2602.16873 | 2026 | 形式化拓扑路由与Pareto分析 |

---

*本研究报告于2026年4月完成，覆盖截至该时间点的公开文献和行业动态。*
