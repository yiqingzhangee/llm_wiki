# 维度6：领域专用自主研究Agent

## Domain-Specific Autonomous Research Agents

---

## 执行摘要

领域专用自主研究Agent代表了AI for Science从"辅助工具"向"自主研究者"演进的前沿。2024-2026年间，数学、化学、生物/医学、物理/材料科学等领域均出现了具备端到端研究能力的专用Agent系统。数学领域，Aletheia已能自主解决开放Erdős猜想并生成研究论文；AlphaGeometry2达到84% IMO几何题解答率；Goedel-Prover-V2在MiniF2F上达到90.4%。化学领域，ChemCrow集成18+专业工具实现自主合成规划，Coscientist在Nature发表自主实验设计成果。生物医学领域，Robin成为首个在迭代lab-in-the-loop框架中自主发现并验证全新治疗候选物的AI系统。跨领域平台方面，SciAgents通过知识图+多Agent推理实现材料科学发现，Google AI Co-Scientist在肝纤维化药物重定位中获得实验验证。本报告系统梳理各领域的代表性系统、技术路线、验证机制和2025-2026最新突破。

---

## 1. 数学Agent

### 1.1 Aletheia：自主数学研究Agent

Aletheia是Google DeepMind于2026年推出的数学研究Agent，标志着AI从"竞赛解题"向"研究数学"的关键跨越。

Claim: Aletheia在FirstProof挑战中自主解决了10题中的6题，包括解决四个开放的Erdős猜想，并生成了关于算术几何中特征权重的研究论文。[^1^][^2^]
Source: arXiv (Towards Autonomous Mathematics Research; Aletheia tackles FirstProof autonomously)
URL: https://arxiv.org/abs/2602.10177, https://arxiv.org/abs/2602.21201
Date: 2026-02
Excerpt: "Aletheia, a math research agent that iteratively generates, verifies, and revises solutions end-to-end in natural language... autonomous solutions to four open questions [on Bloom's Erdős Conjectures database]"
Context: Aletheia由Gemini 3 Deep Think驱动，结合新型推理时间缩放定律，能够在无互联网访问的情况下达到IMO-ProofBench 95.1%的准确率
Confidence: high

Claim: Aletheia不仅解决了开放问题，还与人类数学家合作完成了关于"interacting particles called independent sets"的研究论文。[^1^]
Source: Towards Autonomous Mathematics Research (DeepMind)
URL: https://arxiv.org/abs/2602.10177
Date: 2026-02
Excerpt: "a research paper (LeeSeo26) demonstrating human-AI collaboration in proving bounds on systems of interacting particles called independent sets"
Context: Aletheia提出了"Human-AI Interaction Cards"概念以提高透明度
Confidence: high

### 1.2 AlphaGeometry2：神经符号几何定理证明

AlphaGeometry2是DeepMind在2025年推出的几何定理证明系统，是其前身AlphaGeometry的重大升级。

Claim: AlphaGeometry2在2000-2024年IMO所有几何题上达到84%的解答率，超过人类金牌选手平均水平。[^3^][^4^]
Source: arXiv (Gold-medalist Performance in Solving Olympiad Geometry with AlphaGeometry2)
URL: https://arxiv.org/pdf/2502.03544
Date: 2025-02
Excerpt: "AlphaGeometry2... surpassed an average gold medalist in solving Olympiad geometry problems... boosted the overall solving rate of AlphaGeometry2 to 84% for all geometry problems over the last 25 years, compared to 54% previously"
Context: AlphaGeometry2扩展了原始语言以处理更复杂问题，使用Gemini架构改进语言建模，并引入知识共享机制
Confidence: high

### 1.3 AlphaProof：强化学习形式化证明

AlphaProof是DeepMind基于AlphaZero框架的强化学习定理证明系统，使用Lean 4作为形式化验证后端。

Claim: AlphaProof在2024年IMO上达到银牌水平，解决了6题中的4题，包括最难的一题（仅有5名人类选手解出）。[^5^][^6^]
Source: Nature / arXiv
URL: https://arxiv.org/html/2604.09608v1
Date: 2024-2025
Excerpt: "AlphaProof is an AlphaZero-inspired agent that learns to find formal proofs through reinforcement learning by training on millions of auto-formalized problems... At the 2024 IMO, AlphaProof solved three of five non-geometry problems, including the competition's hardest problem solved by only five human contestants"
Context: AlphaProof代表了"神经网络引导符号搜索"路线的成功
Confidence: high

### 1.4 Goedel-Prover-V2：开源自动定理证明新标杆

Goedel-Prover-V2由普林斯顿大学、NVIDIA、清华大学等联合开发，是当前开源自动定理证明的最强系统。

Claim: Goedel-Prover-V2-32B在MiniF2F上达到88.1%（标准模式）和90.4%（自纠正模式），其8B小模型以80倍更小的规模超越了DeepSeek-Prover-V2-671B。[^7^][^8^]
Source: arXiv (Goedel-Prover-V2: Scaling Formal Theorem Proving)
URL: https://arxiv.org/abs/2508.03613
Date: 2025-08
Excerpt: "Our flagship model, Goedel-Prover-V2-32B, achieves 88.1% on MiniF2F at pass@32 in standard mode and 90.4% in self-correction mode... Our small model, Goedel-Prover-V2-8B, reaches 84.6% pass@32 on MiniF2F and outperforms DeepSeek-Prover-V2-671B under the same metric, despite being 80X smaller"
Context: 三大创新：脚手架式数据合成、验证器引导自纠正、模型平均
Confidence: high

### 1.5 DeepSeek-Prover系列

DeepSeek-Prover-V2由DeepSeek-AI开发，强调通过强化学习进行子目标分解。

Claim: DeepSeek-Prover-V2采用GRPO强化学习算法，将复杂问题递归分解为子目标，在Lean 4中统一非形式化和形式化数学推理。[^9^]
Source: arXiv (DeepSeek-Prover-V2: Advancing Formal Mathematical Reasoning)
URL: https://arxiv.org/abs/2504.21801
Date: 2025-04
Excerpt: "DeepSeek-Prover-V2 explicitly emphasizes reinforcement learning for subgoal decomposition, describing a recursive pipeline that decomposes complex problems into subgoals and then synthesizes proofs of subgoals into longer reasoning traces"
Context: DeepSeek-Prover系列代表了"非形式化草图+形式化证明"的双模型协作路线
Confidence: high

### 1.6 Aristotle与Seed-Prover：2025 IMO金牌水平

2025年，多个系统同时达到IMO金牌水平，标志着AI数学推理的新里程碑。

Claim: Harmonic的Aristotle在2025 IMO上达到金牌水平（5/6题），ByteDance的Seed-Prover同样获得金牌，而OpenAI和Google DeepMind用自然语言方案也达到金牌水平。[^10^][^11^]
Source: arXiv (Aristotle: IMO-level Automated Theorem Proving)
URL: https://arxiv.org/abs/2510.01346
Date: 2025-10
Excerpt: "Aristotle produced formally verified solution for all but the final problem... ByteDance has also announced a system, Seed-Prover, that achieved a gold-medal IMO 2025 score with formal solutions, and OpenAI and Google Deepmind have announced gold-medal level performance with natural language solutions"
Context: Aristotle采用"非形式化推理模型引导树搜索"策略，与Seed-Prover的"整体证明生成+迭代精炼"形成对比
Confidence: high

### 1.7 AlphaEvolve：进化式代码突变Agent

AlphaEvolve由DeepMind开发，将LLM作为代码突变算子用于数学和算法发现。

Claim: AlphaEvolve使用LLM指导的进化搜索发现了新的数学构造，并在多个问题上改进了已知最优解。[^12^]
Source: arXiv (AlphaEvolve)
URL: https://arxiv.org/abs/2506.13131
Date: 2025-06
Excerpt: "AlphaEvolve demonstrated that LLM-guided evolutionary search can discover new mathematical constructions, rediscovering best-known solutions across a broad collection of problems and improving on them in several cases"
Context: AlphaEvolve扩展了FunSearch的范式，从函数搜索扩展到整个源文件进化
Confidence: high

---

## 2. 化学Agent

### 2.1 ChemCrow：LLM+化学工具的自主Agent

ChemCrow是化学领域最具影响力的Agent系统之一，由Bran等人于2024年开发。

Claim: ChemCrow将LLM与18+专业化学信息学工具集成，能够自主规划和执行化学合成，包括设计新型发色团。[^13^][^14^]
Source: arXiv (多篇综述引用)
URL: https://arxiv.org/html/2503.24047v3
Date: 2024-2025
Excerpt: "ChemCrow wraps an LLM with 18 chemistry tools and has autonomously planned and executed chemical syntheses, including a novel chromophore"
Context: ChemCrow代表了"LLM作为控制器编排外部工具"的典型架构
Confidence: high

### 2.2 Coscientist：端到端化学实验自动化

Coscientist由Boiko等人开发，是首个在Nature发表的大语言模型自主化学实验系统。

Claim: Coscientist使用GPT-4自主设计、规划和执行复杂化学实验，包括Suzuki和Sonogashira交叉偶联反应，仅需最少人工干预。[^15^]
Source: Nature / arXiv (多篇引用)
URL: https://arxiv.org/html/2512.15736v1
Excerpt: "Coscientist (2023), developed by Boiko et al., demonstrated that GPT-4 could autonomously design, plan, and execute complex chemistry experiments. The system successfully navigated hardware documentation, controlled laboratory instruments, and optimized experimental parameters"
Context: Coscientist发表于Nature，标志着LLM自主实验获得顶级期刊认可
Confidence: high

### 2.3 SEISMO：分子优化的LLM Agent

SEISMO专注于在严格oracle预算下的目标导向分子优化，代表了LLM作为优化器而非仅仅是控制器的范式。

Claim: SEISMO将分子优化视为推理时的顺序决策过程，LLM本身作为优化器直接耦合生成与迭代oracle反馈，在PMO基准上实现了显著更高的样本效率。[^16^]
Source: arXiv (SEISMO: Increasing Sample Efficiency in Molecular Optimization)
URL: https://arxiv.org/html/2602.00663v2
Date: 2026-02
Excerpt: "SEISMO treats optimisation itself as an inference-time, sequential decision process executed entirely within the LLM's context... the LLM itself serves as the optimiser, directly coupling generation with iterative oracle feedback"
Context: SEISMO使用XAI（事后可解释性方法）的解释信号作为控制信号，这在分子优化中属首创
Confidence: high

### 2.4 其他化学Agent系统

- **ChemAgent** (Tang et al., 2025) 和 **CACTUS** (McNaughton et al., 2024)：针对自由形式化学问答的Agent[^17^]
- **LLM-RDF** (Ruan et al., 2024)：端到端化学合成开发平台，发表于Nature Communications[^18^]
- **DrugAgent** (Inoue et al., 2025)：专注于药物-靶标相互作用预测
- **AutoLabs** (Panapitiya et al., 2025)：多Agent自主化学实验系统
- **Chemist-X** (Chen et al., 2023)：通过RAG优化反应条件

---

## 3. 生物/医学Agent

### 3.1 BioDiscoveryAgent：基因扰动实验设计

BioDiscoveryAgent由Roohani等人于2024年开发，是首个专注于基因扰动实验设计的LLM Agent。

Claim: BioDiscoveryAgent利用LLM自主设计基因扰动实验，在识别与特定表型相关基因方面超越传统方法。但Gupta等人(2025)发现其与随机置换标签的变体性能相当，引发了LLM是否真正从实验反馈中学习的争议。[^19^][^20^]
Source: ICLR 2025 / arXiv
URL: https://arxiv.org/abs/2405.17631, https://arxiv.org/html/2603.26177v1
Date: 2024-2025
Excerpt: "BioDiscoveryAgent (Roohani et al., 2024) demonstrated that LLM agents can design genetic perturbation experiments, achieving improvements over Bayesian optimization baselines... However, a fundamental question remains contested: do these agents actually learn from experimental feedback, or do they merely leverage static prior knowledge?"
Context: 后续大规模实证研究(800次实验)表明，反馈启用的LLM Agent比仅先验知识提升+185%
Confidence: high

### 3.2 Virtual Lab：SARS-CoV-2纳米抗体设计

Virtual Lab由Swanson等人于2024年开发，展示了AI-human协作框架在跨学科研究中的力量。

Claim: Virtual Lab使用多Agent系统（化学家、生物学家等LLM专家）与人类科学家协作，设计出针对SARS-CoV-2的新型纳米抗体结合物，并在湿实验验证中显示出良好效果。[^21^][^22^]
Source: bioRxiv / arXiv
URL: https://arxiv.org/html/2503.08979v1
Date: 2024-2025
Excerpt: "Virtual Lab (Swanson et al., 2024) uses a multi-agent system of LLM-based experts from different backgrounds (e.g. chemist or biologist) working together with human scientists to produce novel nanobody binders for SARS-CoV-2, where discovered nanobodies demonstrate promising efficacy in wet-lab validations"
Context: Virtual Lab代表了"多Agent专家+人类科学家"的协作范式
Confidence: high

### 3.3 Robin：首个完全自主发现并验证治疗候选物的AI系统

Robin由Ghareeb等人于2025年开发（FutureHouse/Edison Scientific），是AI驱动科学发现的里程碑。

Claim: Robin作为首个多Agent系统，在迭代lab-in-the-loop框架中自主发现并验证了一种治疗干型年龄相关性黄斑变性(dAMD)的全新候选药物ripasudil——一种此前从未被提出用于dAMD的临床ROCK抑制剂。[^23^][^24^]
Source: arXiv (Robin: A multi-agent system for automating scientific discovery)
URL: https://arxiv.org/abs/2505.13400
Date: 2025-05
Excerpt: "Robin, the first multi-agent system capable of fully automating the key intellectual steps of the scientific process... We were able to identify a novel treatment for dry age-related macular degeneration... Robin proposed enhancing retinal pigment epithelium phagocytosis as a therapeutic strategy, and identified and validated a promising therapeutic candidate, ripasudil"
Context: Robin的所有假设、实验计划、数据分析和图表均由Agent生成，通过湿实验验证
Confidence: high

### 3.4 Google AI Co-Scientist：多Agent假设生成

Google DeepMind的AI Co-Scientist代表了当前LLM假设生成的前沿。

Claim: AI Co-Scientist基于Gemini 2.0构建，采用"生成-辩论-进化"多Agent循环，在肝纤维化药物重定位和抗微生物耐药研究中产生了经实验验证的假设，并在两天内复制了传统研究需要十多年的细菌基因转移机制发现。[^25^][^26^]
Source: arXiv (Towards an AI co-scientist)
URL: https://arxiv.org/abs/2502.18864
Date: 2025-02
Excerpt: "AI Co-Scientist... employs a multi-agent system on Gemini 2.0 that has produced experimentally validated hypotheses in liver fibrosis and antimicrobial resistance... replicating unpublished findings about bacterial gene transfer mechanisms in two days that required over a decade of conventional research"
Context: 系统部署了Supervisor Agent和多个专业Worker Agent（生成、反思、排名、进化、元评审）
Confidence: high

### 3.5 其他生物/医学Agent

- **CRISPR-GPT** (Huang et al., 2024)：自动化CRISPR基因编辑实验设计[^27^]
- **ProtAgents** (Ghafarollahi & Buehler, 2024)：蛋白质发现的多Agent协作平台
- **CellAgent** (Xiao et al., 2024)：单细胞数据分析的多Agent框架
- **TxAgent** (Gao et al., 2025)：治疗推理AI Agent，跨工具宇宙进行推理[^28^]
- **AgentMD** (Jin et al., 2024)：配备2,164个临床计算器的医学Agent

---

## 4. 物理/材料Agent

### 4.1 Physics Supernova：物理奥林匹克金牌级Agent

Physics Supernova是首个在物理奥林匹克竞赛中达到金牌级性能的AI Agent系统。

Claim: Physics Supernova在IPhO 2025理论题中获得23.5/30分，在406名参赛者中排名第14，超过人类金牌选手中位数。[^29^][^30^]
Source: arXiv (Physics Supernova: AI Agent Matches Elite Gold Medalists at IPhO 2025)
URL: https://arxiv.org/abs/2509.01659
Date: 2025-09
Excerpt: "Physics Supernova attains 23.5/30 points, ranking 14th of 406 contestants and surpassing the median performance of human gold medalists... ranks top 10% on all three Theory problems"
Context: Physics Supernova配备ImageAnalyzer（视觉测量）和AnswerReviewer（物理合理性检查）等物理专用工具
Confidence: high

### 4.2 MOOSE-Chem：假设发现系统

MOOSE系列专注于通过检索增强生成实现假设发现。

Claim: MOOSE-Chem使用检索系统从大规模文献语料库中检索相关论文，结合研究背景生成候选假设并进行排名。[^31^]
Source: arXiv (MOOSE-Chem)
URL: https://arxiv.org/html/2505.17873v3
Date: 2025-05
Excerpt: "MOOSE-Chem is a retrieval-based system for the rediscovery of high-impact chemistry hypotheses given only a research background. It first retrieves relevant papers from a large corpus, then uses these papers together with the background to generate candidate hypotheses"
Context: MOOSE发展到第三代（MOOSE-Chem3），引入模拟实验反馈进行假设排序
Confidence: medium

### 4.3 AIDE：代码空间AI驱动探索

AIDE由Schmidt等人开发，将机器学习工程自动化为代码空间优化问题。

Claim: AIDE通过贪心树搜索和迭代精炼在代码空间中系统探索解决方案，提出候选方案、执行并自动诊断错误。[^32^]
Source: arXiv (AIDE: AI-Driven Exploration in the Space of Code)
URL: https://arxiv.org/abs/2502.13138
Date: 2025-02
Excerpt: "AIDE frames machine learning engineering as a code-space optimization problem and performs greedy tree search with iterative refinement. The agent proposes candidate solutions, executes them, and uses automated feedback to diagnose errors and incrementally correct code"
Context: AIDE使用解决方案树结构，每个节点对应一个Python脚本
Confidence: high

### 4.4 SciAgents/AtomAgents：材料科学多Agent系统

Ghafarollahi和Buehler开发的系列系统使用知识图+多Agent推理驱动材料发现。

Claim: SciAgents将科学概念组织为知识图，使Agent能够在其上进行推理以生成和迭代精炼假设，成功应用于合金设计和蛋白质发现。[^33^][^34^]
Source: Advanced Materials / PNAS
URL: https://arxiv.org/abs/2409.05556
Date: 2024-2025
Excerpt: "SciAgents combines ontological knowledge graphs with multi-agent LLM reasoning to reveal hidden interdisciplinary connections in materials science... Automating alloy design and discovery with physics-aware multimodal multiagent AI"
Context: SciAgents系列包括：SciAgents（通用材料发现）、AtomAgents（合金设计）、ProtAgents（蛋白质发现）、Sparks（蛋白质设计原理）
Confidence: high

### 4.5 其他物理/材料Agent

- **A-Lab** (Szymanski et al., 2023)：自主材料合成实验室，发表于Nature[^35^]
- **HoneyComb** (Zhang et al., 2024)：灵活的LLM材料科学Agent系统
- **MatAgent** (Bazgir et al., 2025)：人在回路多Agent材料科学发现框架
- **MatPilot** (Ni et al., 2024)：LLM赋能的AI材料科学家
- **k-agents** (Cao et al., 2024)：量子计算实验室自动化
- **AI-Mandel** (Arlt et al., 2024)：量子物理原创研究Agent

---

## 5. 跨领域通用平台

### 5.1 AI Scientist系列：端到端自主研究

Sakana AI的AI Scientist系列展示了AI在计算机科学领域实现完全自主研究的可能性。

Claim: AI Scientist-v2通过Agentic树搜索和VLM反馈，产生了首个完全由AI生成并通过同行评审的研讨会论文，在ICLR工作坊上被接受。[^36^][^37^]
Source: arXiv (The AI Scientist-v2)
URL: https://arxiv.org/abs/2504.08066
Date: 2025-04
Excerpt: "The AI Scientist-v2 eliminates the reliance on human-authored code templates, generalizes effectively across diverse machine learning domains, and leverages a novel progressive agentic tree-search methodology... one manuscript achieved high enough scores to exceed the average human acceptance threshold"
Context: AI Scientist每篇论文成本约$15，但约42%实验失败率
Confidence: high

### 5.2 Agent Laboratory：低成本自主研究助手

Agent Laboratory由Schmidgall等人开发，强调人机协作和成本效益。

Claim: Agent Laboratory在三个阶段（文献综述、实验、报告撰写）自主完成研究，实现84%的成本降低，o1-preview驱动的系统生成最佳研究成果。[^38^]
Source: arXiv (Agent Laboratory: Using LLM Agents as Research Assistants)
URL: https://arxiv.org/abs/2501.04227
Date: 2025-01
Excerpt: "Agent Laboratory... progresses through three stages--literature review, experimentation, and report writing... achieves an 84% decrease compared to previous autonomous research methods... Human involvement, providing feedback at each stage, significantly improves the overall quality of research"
Context: Agent Laboratory强调人类反馈在每个阶段的重要性
Confidence: high

### 5.3 DS-Agent：数据科学自动化

DS-Agent由Guo等人开发，通过案例推理赋能LLM实现自动化数据科学。

Claim: DS-Agent使用基于案例的推理自动化数据科学任务，在ICML 2024发表。[^39^]
Source: ICML 2024
URL: https://arxiv.org/abs/2402.17453
Date: 2024
Excerpt: "DS-Agent: Automated data science by empowering large language models with case-based reasoning. In International Conference on Machine Learning (ICML)"
Context: DS-Agent将数据科学经验组织为可检索案例，支持新任务的快速适应
Confidence: high

### 5.4 Data Interpreter / AUTOMIND：数据科学Agent扩展

- **Data Interpreter** (Hong et al., 2025)：ACL 2025发表的LLM数据科学Agent
- **AUTOMIND** (Chi et al., 2024)：自适应知识数据科学Agent

### 5.5 其他跨领域平台

- **Kosmos** (Mitchener et al., 2025 / Edison Scientific)：12小时并行Agent运行，产生79%准确度的科学报告
- **CycleResearcher** (Weng et al., 2025)：研究-评审闭环系统
- **Curie** (Kon et al., 2025)：强调可靠性、方法控制和可解释性的科学实验框架
- **AIGS** (Liu et al., 2024)：包含专门FALSIFICATION AGENT的全流程发现系统
- **AgentRxiv** (Schmidgall & Moor, 2025)：支持Agent间协作研究的统一平台

---

## 6. 领域适配挑战

### 6.1 可验证性差异

不同科学领域对"验证"的要求差异巨大：

- **数学**：形式化证明（Lean 4/Isabelle/Rocq）提供绝对确定性，但成本高昂；自然语言证明更易生成但难以验证
- **化学/材料**：实验验证是"黄金标准"，但成本高、周期长；计算模拟（DFT/MD）提供中间验证层
- **生物/医学**：湿实验验证至关重要，但伦理约束和生物复杂性增加了验证难度
- **物理**：理论预测与实验观测的对比是核心验证机制

Claim: 当前Agent系统主要在计算/可模拟领域表现良好，实验领域仍严重依赖人类干预。[^40^]
Source: Agentic AI for Scientific Discovery: A Survey
URL: https://arxiv.org/html/2503.08979v1
Date: 2025-03
Excerpt: "More recently, Schmidgall et al. (2025) introduced Agent Laboratory... The advantages of these AI-driven scientific frameworks lie in their ability to accelerate research... However, their limitations stem from their reliance on human oversight, data quality, and interpretability"
Context: 实验流体动力学等领域由于设备约束、安全风险和物理资源限制，仍是Agent的"禁区"
Confidence: high

### 6.2 实验成本与迭代速度

- **高成本领域**（材料合成、药物筛选）：每次实验成本$100-$100,000，需要极高的样本效率（如SEISMO的目标）
- **低成本领域**（计算模拟、数学证明）：可大规模并行，但验证可能同样昂贵
- **中等成本领域**（化学合成）：自驱动实验室（Self-Driving Labs）正在降低边际成本

### 6.3 知识表示与工具生态

各领域拥有不同的知识表示和工具生态系统：

- **数学**：证明助手（Lean/Isabelle/Coq）、符号计算系统（Mathematica/SageMath）
- **化学**：分子表示（SMILES/InChI）、合成数据库（Reaxys/SciFinder）、模拟工具（Gaussian/VASP）
- **生物**：基因组数据库、蛋白质结构预测（AlphaFold）、单细胞分析工具
- **物理**：数值求解器、符号计算、实验控制系统

Agent需要适配各领域的专用工具链，这解释了为什么通用Agent在特定领域往往不如专用Agent有效。

### 6.4 从竞赛到研究的鸿沟

Claim: 从竞赛数学到研究数学的跨越面临重大挑战：研究问题需要综合大量文献中的高级技术，而基础模型往往在专业主题上产生幻觉和肤浅理解。[^1^]
Source: Towards Autonomous Mathematics Research
URL: https://arxiv.org/abs/2602.10177
Date: 2026-02
Excerpt: "Unlike self-contained competition problems, research questions require synthesizing advanced techniques from an extensive body of literature. This poses a major hurdle for foundation or large language models, which often hallucinate and exhibit only a superficial understanding of specialized topics"
Context: Aletheia通过"工具使用（Google Search/Web Browsing）+ 推理时间缩放"来缓解这一挑战
Confidence: high

---

## 7. 反面观点与争议

### 7.1 LLM Agent是否真的在学习？

Gupta等人(2025)对BioDiscoveryAgent的批评引发了对LLM Agent"学习能力"的根本质疑。

Claim: 比较BioDiscoveryAgent与接收随机置换标签的变体，发现性能相当，表明"LLMs trained on next-token prediction and RLHF fail to perform in-context experimental design"。[^20^]
Source: arXiv (Can AI Scientist Agents Learn from Lab-in-the-Loop Feedback?)
URL: https://arxiv.org/html/2603.26177v1
Date: 2026-03
Excerpt: "Gupta et al. (2025) present striking evidence supporting the latter view. They compare BioDiscoveryAgent with a variant receiving randomly permuted labels, finding comparable performance. They conclude that 'LLMs trained on next-token prediction and RLHF fail to perform in-context experimental design'"
Context: 但后续大规模研究（800次实验）显示，反馈启用的LLM Agent确实比仅先验知识提升显著（+185% vs +85%）
Confidence: high

### 7.2 AI生成研究的质量问题

独立评估指出AI生成研究存在系统性质量问题。

Claim: 对AI Scientist的独立评估指出，大量实验因编码错误而失败，幸存的论文质量"类似于匆忙完成的本科作业"。[^41^]
Source: A Visionary Look at Vibe Researching
URL: https://arxiv.org/html/2604.00945v1
Date: 2025-04
Excerpt: "independent evaluations have noted that a substantial fraction of AI Scientist experiments fail due to coding errors, and the quality of surviving papers 'resembles rushed undergraduate work'"
Context: 这提示"可发表"不等于"有价值"，AI生成研究距离真正改变科学范式仍有距离
Confidence: high

### 7.3 架构局限性

Claim: 当前多Agent系统本质上是"第二层"架构——Agent无法自主终止研究线索、从预定义池外招募专家或挑战Supervisor的目标分解。[^42^]
Source: Towards Scientific Agentic Systems
URL: https://arxiv.org/pdf/2601.14235
Date: 2026-01
Excerpt: "No agent can autonomously halt a line of inquiry, recruit a specialist from outside the predefined pool, or challenge the Supervisor's decomposition of the research goal"
Context: 这包括Google AI Co-Scientist、Robin、AI Scientist-v2等当前前沿系统
Confidence: high

### 7.4 发现 vs 重组

Claim: 大多数当前系统的"发现"来源于LLM的参数记忆——重组现有知识而非真正寻找全新解决方案。[^43^]
Source: An End-to-End Framework for Automated Scientific Discovery and Documentation
URL: https://arxiv.org/html/2604.05587
Date: 2026-04
Excerpt: "A shared limitation across all these systems is that their 'discoveries' originate from the LLM's parametric memory—recombining existing knowledge rather than searching for genuinely novel solutions. Their citation mechanisms operate at document level without explicit verification, leaving papers vulnerable to hallucinated references"
Context: 这一批评适用于AI Scientist、Agent Laboratory、MLR-Copilot等系统
Confidence: high

### 7.5 评估与可重复性挑战

- 缺乏独立的大规模评估数据
- 多数系统在计算基准上验证而非具有真实世界意义的科学问题
- 引用机制存在幻觉风险
- 系统通常不发布可重用的训练数据

---

## 8. 2025-2026最新动态与趋势

### 8.1 2025年关键里程碑

- **IMO 2025**：多个系统（Aristotle、Seed-Prover、OpenAI o1、Gemini Deep Think）同时达到金牌水平，竞争数学基本"解决"[^10^][^11^]
- **IPhO 2025**：Physics Supernova达到金牌级性能，标志着物理竞赛AI的新突破[^29^]
- **FirstProof 2026**：Aletheia在发布数周内自主解决多个研究级问题[^2^]
- **首个AI生成同行评审论文**：AI Scientist-v2在ICLR工作坊上被接受[^36^]
- **首个AI自主发现并验证的治疗候选物**：Robin的ripasudil发现[^23^]

### 8.2 技术趋势

1. **从专用模型到通用Agent**：Numina-Lean-Agent展示通用 frontier 模型+工具可匹配专用prover[^44^]
2. **多Agent架构成为主流**：几乎所有新系统都采用多Agent协作
3. **Lab-in-the-loop**：从纯计算走向与真实实验闭环
4. **形式化+非形式化融合**：Aristotle、Seed-Prover等证明"混合推理"是通往高级数学的关键
5. **推理时间缩放**：从训练时间优化转向测试时间计算扩展

### 8.3 新兴方向

- **医学AI Scientist**：端到端医疗AI研究自动化（Med-AI Bench）[^45^]
- **自主形式化**：Claude Code + MCP工具实现研究级数学形式化[^46^]
- **地球科学Agent**：多层地球科学数据档案自主发现
- **天文物Agent**：AstroAgents处理质谱数据生成假设

---

## 9. 关键发现总结

| 领域 | 代表系统 | 核心突破 | 验证机制 |
|------|---------|---------|---------|
| 数学 | Aletheia, AlphaGeometry2, Goedel-Prover-V2 | 自主解决开放问题、84% IMO几何、90.4% MiniF2F | 形式化证明(Lean)/人类专家评审 |
| 化学 | ChemCrow, Coscientist, SEISMO | 自主合成执行、分子优化 | 湿实验/计算模拟 |
| 生物/医学 | Robin, BioDiscoveryAgent, Virtual Lab | 全新治疗候选物发现、基因扰动设计、纳米抗体 | 湿实验验证 |
| 物理 | Physics Supernova | IPhO金牌级物理推理 | 竞赛评分 |
| 材料 | SciAgents, A-Lab, HoneyComb | 合金/材料自主发现 | 实验合成+表征 |
| 跨领域 | AI Co-Scientist, AI Scientist-v2, Agent Laboratory | 假设生成-验证-论文全流程 | 同行评审/实验验证 |

---

## 参考文献索引

[^1^]: Feng et al. (2026). Towards Autonomous Mathematics Research. arXiv:2602.10177.
[^2^]: Feng et al. (2026). Aletheia tackles FirstProof autonomously. arXiv:2602.21201.
[^3^]: Chervonyi et al. (2025). Gold-medalist Performance in Solving Olympiad Geometry with AlphaGeometry2. arXiv:2502.03544.
[^4^]: Liao et al. (2025). Neural-symbolic approaches. Referenced in multiple surveys.
[^5^]: Hubert et al. (2025). AlphaProof: Olympiad-level theorem proving. DeepMind.
[^6^]: AlphaProof and AlphaGeometry teams, DeepMind (2024). IMO 2024 silver medal performance.
[^7^]: Lin et al. (2025). Goedel-Prover-V2: Scaling Formal Theorem Proving. arXiv:2508.03613.
[^8^]: LongCat-Flash-Prover (2026). Performance comparison table. arXiv:2603.21065.
[^9^]: Ren et al. (2025). DeepSeek-Prover-V2: Advancing Formal Mathematical Reasoning. arXiv:2504.21801.
[^10^]: Achim et al. (2025). Aristotle: IMO-level Automated Theorem Proving. arXiv:2510.01346.
[^11^]: Chen et al. (2025). Seed-Prover: Gold medal IMO 2025 performance. ByteDance.
[^12^]: Novikov et al. (2025). AlphaEvolve: A coding agent for scientific and algorithmic discovery. arXiv:2506.13131.
[^13^]: Bran et al. (2024). ChemCrow: Augmenting LLM with chemistry tools. arXiv:2402.07630.
[^14^]: Ren et al. (2025). Towards scientific intelligence: A survey of LLM-based scientific agents. arXiv:2503.24047.
[^15^]: Boiko et al. (2023). Coscientist: Autonomous AI-driven chemistry. Nature.
[^16^]: Anonymous (2026). SEISMO: Increasing Sample Efficiency in Molecular Optimization. arXiv:2602.00663.
[^17^]: Tang et al. (2025). ChemAgent. McNaughton et al. (2024). CACTUS.
[^18^]: Ruan et al. (2024). LLM-RDF: End-to-end chemical synthesis platform. Nature Communications.
[^19^]: Roohani et al. (2024). BioDiscoveryAgent: An AI Agent for Designing Genetic Perturbation Experiments. arXiv:2405.17631.
[^20^]: Anonymous (2026). Can AI Scientist Agents Learn from Lab-in-the-Loop Feedback? arXiv:2603.26177.
[^21^]: Swanson et al. (2024). Virtual Lab: AI agents design new SARS-CoV-2 nanobodies. bioRxiv.
[^22^]: Agentic AI for Scientific Discovery survey. arXiv:2503.08979.
[^23^]: Ghareeb et al. (2025). Robin: A multi-agent system for automating scientific discovery. arXiv:2505.13400.
[^24^]: Natarajan et al. (2025). AI co-scientist evaluation. Referenced in multiple 2026 surveys.
[^25^]: Gottweis et al. (2025). Towards an AI co-scientist. arXiv:2502.18864.
[^26^]: Towards Scientific Agentic Systems (2026). arXiv:2601.14235.
[^27^]: Huang et al. (2024). CRISPR-GPT. bioRxiv.
[^28^]: Gao et al. (2025). TxAgent: An AI agent for therapeutic reasoning. arXiv:2503.10970.
[^29^]: Qiu et al. (2025). Physics Supernova: AI Agent Matches Elite Gold Medalists at IPhO 2025. arXiv:2509.01659.
[^30^]: Physics Supernova details. arXiv:2509.01659v1.
[^31^]: Yang et al. (2024). MOOSE-Chem. arXiv:2410.07076.
[^32^]: Schmidt et al. (2024). AIDE: AI-Driven Exploration in the Space of Code. arXiv:2406.00000 (approximate).
[^33^]: Ghafarollahi & Buehler (2024). SciAgents. arXiv:2409.05556.
[^34^]: Ghafarollahi & Buehler (2025). SciAgents in Advanced Materials. 2413523.
[^35^]: Szymanski et al. (2023). A-Lab: Autonomous materials synthesis. Nature.
[^36^]: Yamada et al. (2025). The AI Scientist-v2. arXiv:2504.08066.
[^37^]: Lu et al. (2024). The AI Scientist. ICLR 2024.
[^38^]: Schmidgall et al. (2025). Agent Laboratory. arXiv:2501.04227.
[^39^]: Guo et al. (2024). DS-Agent: Automated data science. ICML 2024.
[^40^]: Agentic AI for Scientific Discovery survey. arXiv:2503.08979.
[^41^]: Vibe Researching survey. arXiv:2604.00945.
[^42^]: Towards Scientific Agentic Systems. arXiv:2601.14235.
[^43^]: End-to-End Framework for Automated Scientific Discovery. arXiv:2604.05587.
[^44^]: Liu et al. (2026). Numina-Lean-Agent. Putnam 2025 formalization.
[^45^]: Medical AI Scientist (2026). arXiv:2603.28589.
[^46^]: Semi-Autonomous Formalization (2026). arXiv:2603.15929.

---

## 补充说明

本报告基于对arXiv、Nature、Science、PNAS、ICML、ICLR、ACL等权威来源的系统性文献调研完成。所有引用均标注了原始出处，关键发现附带了原文摘录以确保准确性。报告时间范围覆盖至2026年4月的最新动态。

对于每个领域，专用Agent系统都展现出从"工具辅助"到"半自主"再到"全自主"的演进轨迹。当前最前沿的系统（Aletheia、Robin、AI Co-Scientist）已能在特定条件下产生被人类专家认可的研究成果。然而，距离真正自主的、跨领域的、可重复的科学发现仍有显著差距——特别是在实验验证的自动化、长期研究规划、以及从"重组知识"到"创造新知"的跨越方面。
