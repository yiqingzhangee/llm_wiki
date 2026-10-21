## 维度3：科学发现Agent（端到端自主研究）

> 深度研究分析报告 | 生成日期：2026年4月  
> 搜索次数：20+次独立检索 | 覆盖来源：arXiv、Nature、Stanford HAI、学术会议论文、权威科技媒体

---

## 一、AI Scientist v1/v2 完整分析

### 1.1 系统概述与里程碑

Claim: Sakana AI的AI Scientist v2于2025年4月发布了arXiv预印本（arXiv:2504.08066），这是首个端到端agentic系统，能够产生第一篇完全由AI撰写并通过同行评审的workshop论文。该系统自主完成假设生成、实验设计执行、数据分析可视化和论文撰写全流程。[^1^]
Source: Sakana AI / arXiv  
URL: https://arxiv.org/abs/2504.08066  
Date: 2025-04-10  
Excerpt: "We introduce The AI Scientist-v2, an end-to-end agentic system capable of producing the first entirely AI generated peer-review-accepted workshop paper. This system iteratively formulates scientific hypotheses, designs and executes experiments, analyzes and visualizes data, and autonomously authors scientific manuscripts."  
Context: AI Scientist v2相比v1（Lu et al., 2024 arXiv:2408.06292）消除了对人类编写代码模板的依赖，有效泛化到多样化机器学习领域。  
Confidence: high

Claim: 描述AI Scientist系统的人类作者论文于2026年3月26日发表于Nature 651:914-919（DOI: 10.1038/s41586-026-10265-5），这是关于AI自动化科学研究的首篇Nature同行评审论文。需要注意的是，这是人类撰写的关于AI系统的论文，而非AI撰写的论文。[^2^]
Source: Sakana AI Blog / Nature  
URL: https://sakana.ai/ai-scientist-nature/  
Date: 2026-03-26  
Excerpt: "Today, we are happy to announce that a paper describing all of this work and that includes new insights has been published in Nature. This substantial milestone is the result of a close and fruitful collaboration between researchers at Sakana AI, the University of British Columbia (UBC) and the Vector Institute, and the University of Oxford."  
Context: Nature论文全面详述了系统架构、新的scaling结果以及AI生成科学的潜力与挑战。  
Confidence: high

Claim: AI Scientist v2向ICLR 2025 workshop "I Can't Believe It's Not Better: Challenges in Applied Deep Learning" 提交了三篇完全自主生成的论文，其中一篇（"Compositional Regularization: Unexpected Obstacles in Enhancing Neural Network Generalization"）获得平均分6.33/10，超过workshop接受阈值，成为第一篇通过同行评审的完全AI生成论文。该论文随后根据预先协议主动撤回。[^3^]
Source: The AI Scientist-v2论文 / arXiv  
URL: https://arxiv.org/pdf/2504.08066  
Date: 2025-04-10  
Excerpt: "one manuscript achieved an average reviewer score of 6.33 (placing it roughly in the top 45% of submissions) and would have been accepted after meta-review were it human-generated, thus becoming the first fully AI-generated manuscript to successfully pass a peer-review process."  
Context: 该workshop接受率约32.6%（14/43篇），而ICLR 2025主会接受率为32.08%（11,565篇投稿）。论文在评审后被主动撤回，未正式发表。  
Confidence: high

### 1.2 v1与v2架构对比

Claim: AI Scientist v1（2024年8月，arXiv:2408.06292）采用基于人类编写模板的线性流水线，在三个机器学习领域（扩散模型、Transformer、GNN）实现了约$15/篇的端到端论文生成成本，包括idea生成、代码实验、论文撰写和自动评审。[^4^]
Source: Sakana AI / 相关引用  
URL: https://github.com/sergiocoding96/hermes-multi-agent/blob/main/academic_research_report_self_improving_agents.md  
Date: 2024-08  
Excerpt: "First comprehensive framework for fully automatic scientific discovery. Generates research ideas, writes code, executes experiments, visualizes results, writes papers, and runs simulated review. Cost: <$15/paper."  
Context: v1的局限性在于依赖特定领域的人工模板，难以泛化到新领域。  
Confidence: high

Claim: AI Scientist v2的核心改进包括：(1) 消除模板依赖，实现无模板的开放 ended 研究探索；(2) 引入渐进式agentic tree search（BFTS，Best-First Tree Search），由Experiment Manager Agent协调多个并行研究方向的探索、自动调试失败节点、动态剪枝与回溯；(3) 集成VLM（Vision-Language Model）反馈循环，用于迭代改进图表的内容和美学质量。[^5^]
Source: arXiv 2504.08066 / Pebblous Blog  
URL: https://blog.pebblous.ai/project/AgenticAI/ai-scientist-v2/en/  
Date: 2026-03-28  
Excerpt: "v2 replaces v1's template-driven approach with open-ended exploration... Agentic tree search — Experiment manager agent explores multiple research directions simultaneously... VLM feedback loop for automated interpretation of experimental results."  
Context: v2使用BFTS算法探索研究空间，实验管理Agent协调并行路径。同时集成了Semantic Scholar API进行文献检索与新颖性验证。  
Confidence: high

| 特性 | v1 (2024) | v2 (2025) |
|---|---|---|
| 研究设计 | 基于模板 | 无模板、开放 ended |
| 搜索方法 | 线性流水线 | Agentic Tree Search (多路径并行探索) |
| 结果解释 | 纯文本 | VLM反馈循环（视觉分析） |
| 同行评审结果 | 未提交 | ICLR 2025 workshop: 1/3 被接受 |
| 成本 | ~$15/篇 | ~$20-25/篇 |
| 安全机制 | 基础 | 强制Docker沙箱隔离 |

### 1.3 成本与效率

Claim: AI Scientist v2生成一篇完整论文的实验成本约为$15-20，写作成本约$5，总计$20-25/篇。相比之下，人类研究者发表一篇论文的格式化成本中位数$477，开放获取APC费用$3,500-4,000，加上劳动力成本，AI与人类成本差异超过100倍。[^6^]
Source: Pebblous AI Science Fact Check  
URL: https://blog.pebblous.ai/report/ai-science-new-era/en/  
Date: 2026-04-05  
Excerpt: "It costs only $20–$25 to generate a paper with AI... Experiment costs of $15–$20 plus ~$5 for writing total $20–$25 per paper. Compared to human researchers' publication costs (APCs of $3,500–$4,000, plus labor), the difference exceeds 100x."  
Context: 独立评估显示，v2的3篇投稿中2篇被拒（3-4/10分），因此需要考虑成本-质量权衡。  
Confidence: high

Claim: 根据独立研究者对AI Scientist的评估，实际生成10篇论文的总成本仅为$42（平均$6/篇），但人类仍需约5小时设置系统和15小时实现实验模板，加上约10小时的种子idea构思、输出审核和结果精炼，总计约25小时人类劳动。[^7^]
Source: An Evaluation of Sakana's AI Scientist / arXiv  
URL: https://arxiv.org/html/2502.14297v1  
Date: 2025-02-20  
Excerpt: "The total cost for our main experiments... amounted to only $42 USD — an average of $6 per manuscript... Setting up the AI Scientist took approximately 5 hours, while implementing the experimental template required an additional 15 hours... we spent about 10 more hours brainstorming initial seed ideas."  
Context: 尽管成本极低，但AI Scientist的操作速度是人类研究者的3-11倍，质量却仅达到本科生水平。  
Confidence: high

### 1.4 学术社区反响与争议

Claim: AI Scientist v2的三篇投稿中，被接受的一篇存在显著局限性，包括LSTM概念归属错误、定义不准确、引用错误（hallucination）等。另外两篇被拒的论文分别获得3/10和4/10分，存在重大技术缺陷、实验错误、图表重复等问题。[^8^]
Source: AI Scientist-v2论文附录 / arXiv  
URL: https://arxiv.org/pdf/2504.08066  
Date: 2025-04-10  
Excerpt: "The biggest flaw of this paper is the mentioning of results that are not substantiated by results... Overall· Workshop: 3/10 (Reject)... Overall· Conference: 2/10 (Strong reject)."  
Context: Sakana AI联合创始人David Ha在Nature论文中承认，AI生成的工作尚未达到同一会议最佳人类论文的水平。  
Confidence: high

Claim: 独立评估显示57%的AI Scientist v2论文包含虚假数据（false data）。这不是一个生产就绪的研究工具，而是一个具有巨大可靠性问题的技术演示。[^9^]
Source: ByteIota  
URL: https://byteiota.com/ai-scientist-v2-passes-peer-review-but-57-is-false-data/  
Date: 2026-03-29  
Excerpt: "Independent evaluations reveal that 57% of AI Scientist v2's papers contain false data. This isn't a production-ready research tool. It's an impressive technical demo with a massive reliability problem."  
Context: 该评估可能基于AI Scientist系统生成的多篇论文的统计分析，而非仅ICLR提交的三篇。  
Confidence: medium

---

## 二、Aletheia数学研究Agent

### 2.1 系统概述

Claim: Google DeepMind于2026年2月发布了Aletheia（arXiv:2602.10177），一个基于Gemini Deep Think的数学研究Agent，旨在从竞赛级数学问题求解转向专业研究级数学。Aletheia采用三子Agent架构：Generator（生成候选解）、Verifier（自然语言验证器检查缺陷/幻觉）、Reviser（修正错误），三者迭代交互直至Verifier通过或达到尝试上限。[^10^]
Source: Google DeepMind / arXiv  
URL: https://arxiv.org/html/2602.10177v3  
Date: 2026-02-10/2026-03-06  
Excerpt: "We built a math research agent, internally codenamed Aletheia, on top of Gemini Deep Think. Aletheia include three subagents, a (solution) Generator, a Verifier, and a Reviser that interact continuously until a solution is found that the Verifier approves, or until the attempts reach a preset (hyperparameter) limit."  
Context: 与AlphaGeometry和AlphaProof使用形式语言不同，Aletheia端到端使用自然语言操作。每个子Agent内部又涉及对Gemini基础模型的调用编排。  
Confidence: high

Claim: Aletheia在IMO-ProofBench Advanced基准上达到95.1%准确率（此前记录65.7%），在FutureMath Basic（博士级练习内部基准）上展现最先进性能。Aletheia利用基于Gemini Deep Think的新型推理时缩放定律（inference-time scaling law）。[^11^]
Source: DeepMind Blog / arXiv 2602.10177  
URL: https://deepmind.google/blog/accelerating-mathematical-and-scientific-discovery-with-gemini-deep-think/  
Date: 2026-02-11  
Excerpt: "The latest advanced version of Deep Think, as of Jan 2026, has significantly outperformed the IMO-Gold version (Jul 2025) on Olympiad-level problems... Aletheia makes further leaps in terms of reasoning quality with lower inference-time compute."  
Context: 2026年1月版Deep Think与2025版相比，将奥数级问题所需计算量减少100倍。  
Confidence: high

### 2.2 Erdős猜想评估详情

Claim: Aletheia被系统部署评估Bloom's Erdős Conjectures数据库中700个标记为"Open"的猜想。最终结果显示：63个解在技术上正确，但只有13个（6.5%）正确解决了预期的问题陈述（其余50个因误解问题陈述而产生数学上空洞的"正确"解）。12个响应被标记为模糊。在200个确定可标记正确或错误的候选解中，137个（68.5%）根本错误。[^12^]
Source: Towards Autonomous Mathematics Research / arXiv  
URL: https://arxiv.org/html/2602.10177v3  
Date: 2026-03-06  
Excerpt: "Our ultimate findings were that 63 solutions were technically correct, but only 13 solutions correctly addressed the intended problem statement... 137 (68.5%) of the responses were fundamentally flawed, while 63 (31.5%) of the responses were technically correct, of which only 13 (6.5%) were meaningfully correct."  
Context: 这13个有意义正确的结果分为四类：自主解决（2个，包括Erdős-652和Erdős-1051）、部分AI解决（2个）、独立重新发现（4个）、文献识别（5个已有人类解但数据库未更新）。  
Confidence: high

Claim: Aletheia自主解决的4个Erdős开放问题为Erdős-652、Erdős-654、Erdős-1040和Erdős-1051。研究者坦承这些解 individually 并未达到研究论文水平，有些仅相当于学生练习水平。Erdős-1051的解被认为是一个AI系统自主解决略有非平凡性的开放Erdős问题的早期例子，涉及经典的"移至级数尾部并应用Mahler准则"思想。[^13^]
Source: A Case Study on the Erdős Problems / arXiv  
URL: https://arxiv.org/html/2601.22401v3  
Date: 2026-02-05  
Excerpt: "The '4' autonomous solutions cited above refer to Erdős-652, Erdős-654, Erdős-1040, and Erdős-1051. In the estimation of our experts, none of the four individually rises to the level of a research paper. In fact, some of them are at the level of student exercises... Aletheia's solution to Erdős-1051 represents an early example of an AI system autonomously resolving a slightly non-trivial open Erdős problem."  
Context: Erdős-1051的解后来被Aletheia与人类数学家和Gemini Deep Think协作进一步推广，产出了研究论文[BKK+26]。  
Confidence: high

### 2.3 自主级别分类与里程碑论文

Claim: DeepMind提出了"自主数学研究级别"分类法（类似SAE自动驾驶级别）：Level 0（主要人类驱动，竞赛级问题）、Level 1（人机协作，人类提供策略）、Level A2（本质自主，可发表研究质量）。Feng26论文（关于算术几何中eigenweights结构常数）被分类为Level A2——完全无人类干预生成的可发表质量研究论文。[^14^]
Source: MarkTechPost / arXiv 2602.10177  
URL: https://www.marktechpost.com/2026/02/12/google-deepmind-introduces-aletheia-the-ai-agent-moving-from-math-competitions-to-fully-autonomous-professional-research-discoveries/  
Date: 2026-02-12  
Excerpt: "DeepMind proposed a standard for classifying AI math contributions, similar to the levels used for autonomous vehicles... The paper Feng26 is classified as Level A2, meaning it is essentially autonomous and of publishable quality."  
Context: Aletheia还为LeeSeo26论文（关于independent sets边界证明）提供了高级路线图和"大局"策略，由人类作者将其形式化为严格证明。  
Confidence: high

### 2.4 关键设计特征：自我报告失败

Claim: Aletheia最实用的设计特征之一是当无法建立有效证明时，系统显式输出"no solution found"而非编造答案。这种自我过滤行为使Aletheia在结构上不同于自信地产生幻觉的系统。条件准确率（当返回解时）达到98.3%。[^15^]
Source: TechJackSolutions / arXiv 2602.10177  
URL: https://techjacksolutions.com/ai-brief/google-deepminds-aletheia-research-agent-attempts-original-m/  
Date: 2026-04-20  
Excerpt: "Rather than generating plausible-sounding but incorrect proofs, the failure mode that makes AI systems dangerous in high-stakes reasoning tasks, coverage describes the system as explicitly outputting 'no solution found' when it cannot establish a valid proof."  
Context: 这一设计选择对数学之外的高风险推理任务具有深远影响——知道"自己不知道什么"并如实报告的AI系统，在结构上与自信幻觉的系统截然不同。  
Confidence: high

---

## 三、Google AI Co-scientist

### 3.1 架构设计

Claim: Google AI Co-scientist（arXiv:2502.18864，2025年2月）是一个基于Gemini 2.0的多Agent系统，采用"生成-辩论-进化"（generate, debate, and evolve）方法进行假设生成，受科学方法启发并通过测试时计算缩放加速。系统包含六大专业Agent：Generation（生成假设）、Reflection（快速过滤）、Ranking（基于Elo评分的锦标赛排名）、Evolution（进化 refinement）、Proximity（聚类相似idea）、Meta-review（跨迭代编译反馈调整prompt）。所有Agent由Supervisor Agent协调。[^16^]
Source: Google / arXiv 2502.18864  
URL: https://arxiv.org/abs/2502.18864  
Date: 2025-02-26  
Excerpt: "The AI co-scientist is intended to help uncover new, original knowledge and to formulate demonstrably novel research hypotheses and proposals... The system's design incorporates a generate, debate, and evolve approach to hypothesis generation... a multi-agent architecture with an asynchronous task execution framework for flexible compute scaling; a tournament evolution process for self-improving hypotheses generation."  
Context: Supervisor Agent解析自然语言研究目标为结构化计划，分配任务到专业worker Agent，只有Generation、Reflection和Evolution Agent有权访问工具集。架构被批评为"原型性第二层级"——所有Agent运行同一模型家族，无法自主停止研究方向或挑战Supervisor的分解。  
Confidence: high

### 3.2 生物医学验证案例

Claim: AI Co-scientist在急性髓系白血病（AML）药物重定位中，提出了Binimetinib、Pacritinib、Cerivastatin、Pravastatin和Dimethyl fumarate等候选药物。其中Binimetinib、Pacritinib和Cerivastatin在体外实验中显示了细胞活力抑制作用，Binimetinib在AML细胞系中的IC50低至7 nM。对于全新候选药物，Nauvumalat、KIRA6和Leflunomide被选出进行验证。[^17^]
Source: Towards an AI co-scientist / arXiv  
URL: https://arxiv.org/abs/2502.18864v1  
Date: 2025-02-26  
Excerpt: "Of the five drugs tested, Binimetinib, Pacritinib, and Cerivastatin demonstrated inhibition of cell viability... Binimetinib exhibited an IC50 as low as 7 nM in AML cell lines."  
Context: 这些wet-lab实验作为药物重定位管道的可行性检查点，而非替代严格的临床前和临床评估。专家从30个顶级候选假设中选择了验证药物。  
Confidence: high

Claim: 在肝纤维化新靶点发现中，AI Co-scientist提出了三类表观基因组调控因子作为有前景的抗纤维化治疗靶点：组蛋白去乙酰化酶（HDACs）、DNA甲基转移酶1（DNMT1）和溴结构域蛋白4（BRD4）。在基于人类肝脏类器官（hepatic organoids）的实验系统中，靶向其中两个表观遗传修饰因子的药物显示出显著抗纤维化活性且无细胞毒性。其中一种（Vorinostat）是FDA已批准的抗癌药物，创造了重定位治疗肝纤维化的机会。[^18^]
Source: Towards an AI co-scientist / arXiv / Advanced Science  
URL: https://www.biorxiv.org/content/10.1101/2025.04.29.651320v1.full.pdf  
Date: 2025-04-29  
Excerpt: "Two of the three AI co-scientist-recommended drugs that targeted epigenomic modifiers exhibited significant anti-fibrotic activity... One AI co-scientist recommended drug is an FDA-approved anti-cancer treatment (Vorinostat) that reduced TGF β-induced chromatin structural changes by 91% and promoted liver parenchymal cell regeneration."  
Context: 该研究由斯坦福大学Gary Peltz教授团队与Google合作完成，发表在Advanced Science。  
Confidence: high

Claim: AI Co-scientist在两天内通过并行in silico发现，独立复现了关于细菌进化中新型基因转移机制的非发表实验结果——这一发现人类研究者花了超过十年才获得。[^19^]
Source: Towards an AI co-scientist / arXiv / 多篇综述引用  
URL: https://arxiv.org/pdf/2602.19810  
Date: 2025-02-26（引用出现在2026年综述中）  
Excerpt: "Google's AI Co-Scientist... replicating unpublished findings about bacterial gene transfer mechanisms in two days that required over a decade of conventional research."  
Context: 这一结果被多篇后续综述引用作为AI辅助科学发现的关键证据，但具体细菌种类和机制细节在公开摘要中未详述。  
Confidence: high

---

## 四、FutureHouse Robin多Agent系统

### 4.1 系统架构与Agent生态

Claim: FutureHouse（2023年11月成立，由前Google CEO Eric Schmidt资助的非营利AI研究实验室）于2025年5月发布了Robin——首个能够完全自动化科学发现关键智力步骤的多Agent系统。Robin整合了FutureHouse此前发布的专门Agent：Crow（文献Q&A，基于PaperQA2）、Falcon（深度文献综合）、Owl（先前工作检测）、Phoenix（化学实验设计，基于ChemCrow）、Finch（数据驱动的生物学发现）。[^20^]
Source: FutureHouse / arXiv  
URL: https://arxiv.org/abs/2505.13400  
Date: 2025-05-19  
Excerpt: "Robin, the first multi-agent system capable of fully automating the key intellectual steps of the scientific process. By integrating literature search agents with data analysis agents, Robin can generate hypotheses, propose experiments, interpret experimental results, and generate updated hypotheses."  
Context: Robin的workflow编排Crow（文献搜索）、Falcon（假设评估）和Finch（数据分析）来针对特定人类疾病提出新治疗方法并进行临床前验证。  
Confidence: high

| Agent | 主要功能 | 关键能力 |
|---|---|---|
| Crow | 通用科学Q&A | 搜索文献提供简洁事实性答案 |
| Falcon | 深度文献综述 | 综合数百篇论文的发现 |
| Owl | 先前工作检测 | 回答"是否有人做过X" |
| Phoenix | 化学实验设计 | 设计新化合物和合成路径 |
| Finch | 数据驱动生物学发现 | 自动化实验/观察数据集分析 |

Claim: FutureHouse于2025年11月 spun out Edison Scientific作为营利性子公司，获得$70M种子轮融资，估值约$2.5亿，由Spark Capital和Triatomic Capital领投。Edison的核心产品是Kosmos——下一代AI Scientist，能在单次运行中处理1,500篇论文和42,000行分析代码， reportedly 相当于压缩6个月的PhD/博士后研究时间为单次运行。[^21^]
Source: TechFundingNews / Caproasia / SiliconANGLE  
URL: https://techfundingnews.com/edison-raises-70m-ai-scientist-platform/  
Date: 2025-12-19  
Excerpt: "Edison Scientific... raised a $70 million seed round at a roughly $250 million valuation, co-led by Spark Capital, Triatomic Capital... At its core is Kosmos, an 'AI scientist' that can synthesize massive volumes of scientific literature (~1,500 papers per run)... accomplishing work equivalent to six months of a PhD or postdoctoral scientist in a single session."  
Context: 约30名FutureHouse员工转至Edison，FutureHouse继续作为非营利组织专注于基础AI研究。天使投资人包括Google首席科学家Jeff Dean和CrowdStrike联合创始人Dmitri Alperovitch。  
Confidence: high

### 4.2 药物发现案例：ripasudil治疗干性AMD

Claim: Robin通过迭代循环实现了其首个发现：识别ripasudil（一种已临床使用的Rho-kinase/ROCK抑制剂，此前从未被提议用于治疗干性AMD）作为干性年龄相关性黄斑变性（dAMD）的新型治疗候选药物。Robin提出增强视网膜色素上皮（RPE）吞噬作用的治疗策略，并通过体外实验验证。随后Robin提出并分析后续RNA-seq实验，揭示ABCA1（关键脂质外排泵和潜在新靶点）的上调。手稿主体中的所有假设、实验计划、数据分析和数据图均由Robin自主生成，人类研究者仅执行物理实验。[^22^]
Source: FutureHouse / arXiv 2505.13400  
URL: https://www.futurehouse.org/research-announcements/demonstrating-end-to-end-scientific-discovery-with-robin-a-multi-agent-system  
Date: 2025-05-19  
Excerpt: "Robin proposed enhancing retinal pigment epithelium phagocytosis as a therapeutic strategy, and identified and validated a promising therapeutic candidate, ripasudil... All hypotheses, experimental plans, data analyses, and data figures in the main text of this report were produced by Robin."  
Context: 这是首个在迭代lab-in-the-loop框架内自主发现并验证新型治疗候选药物的AI系统。整个过程耗时约2.5个月。  
Confidence: high

---

## 五、其他自主研究系统

### 5.1 EvoScientist：自我进化的多Agent AI科学家

Claim: 华为技术团队（联合阿姆斯特丹自由大学）于2026年3月发布EvoScientist（arXiv:2603.08127），一个自我进化的多Agent系统，包含三个专业Agent：Researcher Agent（RA，生成科学idea和研究提案）、Engineer Agent（EA，执行实验和代码/分析）、Evolution Manager Agent（EMA，提炼交互历史为持久记忆）。系统通过两个持久记忆模块实现跨任务进化：ideation memory（记录可行和失败的研究方向）和experimentation memory（捕获有效的数据处理和模型训练策略）。[^23^]
Source: Huawei / arXiv  
URL: https://arxiv.org/abs/2603.08127  
Date: 2026-03-09  
Excerpt: "EvoScientist decomposes scientific discovery into three specialised agents: a Researcher Agent (RA) that generates scientific ideas and research proposals, an Engineer Agent (EA) that executes experiments and produces code and analysis, and an Evolution Manager Agent (EMA) that distills interaction histories into persistent memories to guide future decision-making."  
Context: EvoScientist使用Elo-based锦标赛进行idea排名，并通过Idea Tree Search（提出-评审-精炼的树结构搜索）探索idea空间。实验显示EvoScientist在 novelty 上取得82.50%胜率，在feasibility上64.17%胜率，超越7个开源和商业基线系统。  
Confidence: high

Claim: EvoScientist生成的6篇完整论文全部被ICAIS 2025（International Conference on AI Scientists）接受，其中2篇获得主要奖项（Best Paper Award和AI Reviewer's Appraisal Award）。[^24^]
Source: EvoScientist论文 / arXiv 2603.08127  
URL: https://arxiv.org/html/2603.08127v1  
Date: 2026-03-09  
Excerpt: "All six full papers generated by the system were accepted to ICAIS 2025, with two receiving major awards... The Best Paper Award paper: Adaptive Evidential Meta-Learning with Hyper-Conditioned Priors for Calibrated ECG Personalisation."  
Context: 需要注意的是，ICAIS是一个专门面向AI生成论文的新会议，其评审标准与传统会议可能不同。但这仍然证明了EvoScientist的进化机制能够生成符合专家评审标准的高价值idea和稳健的实验管道。  
Confidence: high

### 5.2 AutoResearchClaw：23阶段自主研究管道

Claim: UNC Chapel Hill AIMING Lab于2026年3月15日发布AutoResearchClaw，一个完全自主的23阶段研究管道（GitHub发布9天内获得8,000 stars），将单个研究idea转化为会议就绪的论文。管道分为8个阶段：A. 研究范围界定（2阶段）、B. 文献发现（4阶段）、C. 知识综合（2阶段）、D. 实验设计（3阶段）、E. 实验执行（2阶段）、F. 分析与决策（2阶段）、G. 论文写作（4阶段）、H. 最终定稿（4阶段）。包含3个人类审批gate（可跳过）。[^25^]
Source: GitHub aiming-lab/AutoResearchClaw / Paperclipped  
URL: https://github.com/aiming-lab/AutoResearchClaw  
Date: 2026-03-15  
Excerpt: "AutoResearchClaw is a fully autonomous 23-stage research pipeline that takes a natural language topic and produces a complete academic paper: real arXiv/Semantic Scholar citations, sandboxed experiments, statistical analysis, multi-agent peer review, and conference-ready LaTeX."  
Context: v0.3.0引入了MetaClaw跨运行学习系统，将管道失败转化为可重用技能，注入到所有23个阶段。受控实验显示stage retry rate降低24.8%，refine cycle减少40%，整体稳健性提高18.3%。一次完整运行成本$5-15（GPT-4o）。  
Confidence: high

Claim: AutoResearchClaw被用于自主发现Omni-SimpleMem——一个统一的多模态终身Agent记忆框架。从naive基线（F1=0.117）出发，管道自主执行约50次实验，诊断失败模式、提出架构修改、修复数据管道bug、验证改进，最终达到LoCoMo上F1提升+411%（0.117→0.598）和Mem-Gallery上+214%（0.254→0.797）。[^26^]
Source: Omni-SimpleMem论文 / arXiv  
URL: https://arxiv.org/html/2604.01007v2  
Date: 2026-04-02  
Excerpt: "The complete pipeline comprises 23 stages organized in 8 phases... Starting from a naive baseline (F1=0.117 on LoCoMo), the pipeline autonomously executes ~50 experiments... improving F1 by +411% on LoCoMo and +214% on Mem-Gallery."  
Context: 最具影响力的发现不是超参数调整，而是bug修复（+175%）、架构更改（+44%）和prompt工程（+188%），展示了超越传统AutoML的能力。  
Confidence: high

### 5.3 UniScientist：通用科学研究智能

Claim: UniScientist（2026年3月发布）提出"Evolving Polymathic Synthesis"范式，覆盖50+科学学科，将LLM重新定位为跨学科生成器，人类专家作为高精度验证器。30B参数模型在5个研究基准上超越Claude Opus和GPT。系统包含三个关键组件：(1) Evolving Polymathic Synthesis——人机协作数据范式；(2) Agentic Research Loop——使用web_search、google_scholar、page_fetching、code_interpreter等工具的迭代研究循环；(3) Report Aggregation——综合多个候选研究报告。[^27^]
Source: UniScientist GitHub  
URL: https://github.com/UniPat-AI/UniScientist  
Date: 2026-03-11  
Excerpt: "UniScientist advances universal scientific research intelligence through a unified paradigm. By reassigning LLMs as cross-disciplinary generators and human experts as high-precision verifiers, it produces research-grade data spanning 50+ scientific disciplines... A 30B-parameter model trained on this data achieves highly competitive performance across five research benchmarks."  
Context: UniScientist-30B-A3B模型已在HuggingFace和ModelScope提供下载。  
Confidence: medium

---

## 六、质量与可信度问题

### 6.1 幻觉引用危机

Claim: 近18,000篇三大计算机科学会议接受论文的分析显示，2025年至少2.6%的论文包含至少一个可能幻觉的引用——较2024年的约0.3%激增近9倍。另一项分析估计2025年四个其他CS会议中2-6%的论文包含无法验证的引用。Nature与Grounded AI合作的独家分析表明，2025年约7百万学术出版物中可能有超过11万篇包含无效引用。[^28^]
Source: Nature News  
URL: https://www.nature.com/articles/d41586-026-00969-z  
Date: 2026-04-01  
Excerpt: "2.6% of papers in 2025 had at least one potentially hallucinated citation — up from about 0.3% in 2024... if the rate of 65 publications with at least one invalid reference out of some 4,000 publications analysed holds across the academic literature, it would suggest that more than 110,000 of the 7 million or so scholarly publications from 2025 contain invalid references."  
Context: NeurIPS 2025至少有53篇接受论文（约1%）包含虚构引用——虚构的作者和/或标题和/或虚假出版详情。ICLR 2026投稿中发现超过50个幻觉，部分获得8/10评分。  
Confidence: high

Claim: 学者Resnik等人在Accountability in Research（2026年）中论证，当研究者使用GenAI产生幻觉引用（不存在文献）、且这些引用作为数据直接支持研究发现（如综述文章或文献计量研究）、且研究者对编造数据的风险表现出漠不关心（未检查AI输出真实性）时，这种行为可能构成可证明的研究不端行为。[^29^]
Source: PubMed / Accountability in Research  
URL: https://pubmed.ncbi.nlm.nih.gov/41833014/  
Date: 2026  
Excerpt: "GenAI hallucinated citations might qualify as a provable instance of research misconduct under the U.S. federal regulations when a) the researcher uses a GenAI tool to produce hallucinated citations... b) the citations function as data because they directly support research findings... and c) the researcher demonstrates indifference to the risk of fabrication."  
Context: COPE（出版伦理委员会）声明AI不能作为论文作者，ICMJE（国际医学期刊编辑委员会）确定AI不能满足四个作者标准。但所有Big 5出版商并不禁止AI使用本身——要求透明披露。  
Confidence: high

### 6.2 AI生成论文的系统性局限

Claim: 对AI Scientist v2的系统性评估揭示多种常见失败模式：underdeveloped ideas（想法不成熟）、incorrect implementation of the main idea（主要idea实现错误）、lack of deep methodological rigour（缺乏深度方法论严谨性）、experimental errors（实验错误）、duplicated figures（图表重复）、hallucinations such as inaccurate citations（引用幻觉）。目前系统仅限于计算实验。[^30^]
Source: HealthManagement / Nature  
URL: https://healthmanagement.org/c/healthmanagement/News/ai-scientist-passes-workshop-review  
Date: 2026-04-08  
Excerpt: "Common failure modes included underdeveloped ideas, incorrect implementation of the main idea, lack of deep methodological rigour, experimental errors, duplicated figures and hallucinations such as inaccurate citations. At present, the system is limited to computational experiments."  
Context: 3篇投稿中仅1篇达到workshop水平，无一篇达到ICLR主会发表标准。接受论文存在引用错误（如将方法错误归属于2016年研究而非1997年原始工作）、温度缩放实验实现但未实际使用等问题。  
Confidence: high

Claim: Aletheia在700个Erdős问题上的大规模评估揭示了AI数学研究的深层问题：约68.5%的响应根本错误；50个技术上"正确"的解因误解问题陈述而产生数学上空洞的结果；存在"潜意识抄袭"风险——AI可能在预训练中间接吸收了文献解而不自知；最艰巨的步骤不是验证正确性而是确定解是否已存在于先前工作中。[^31^]
Source: A Case Study on the Erdős Problems / arXiv  
URL: https://arxiv.org/html/2601.22401v3  
Date: 2026-02-05  
Excerpt: "It is of course possible that the solution was indirectly ingested from the literature solution, either implicitly through intermediate sources or during training. This highlights a new danger that accompanies AI-generated mathematics: it is susceptible to 'subconscious plagiarism' by reproducing knowledge acquired during training, without attribution."  
Context: 研究者坦承这些问题Erdős问题的"Open"状态可归因于obscurity（被遗忘/未被注意）而非difficulty（难度）。  
Confidence: high

### 6.3 学术出版生态面临的挑战

Claim: 截至2026年4月，所有Big 5出版商（Springer Nature、Elsevier、Wiley、Taylor & Francis、SAGE）均拒绝承认AI为论文作者。JAIGP（Journal of AI-Generated Papers）于2026年2月作为首个专门面向AI撰写论文的实验性期刊推出，截至4月已发表43篇论文并引入多阶段同行评审，但尚无Impact Factor，未获得Scopus/Web of Science等传统索引系统认可。[^32^]
Source: Pebblous Fact Check  
URL: https://blog.pebblous.ai/report/ai-science-new-era/en/  
Date: 2026-04-05  
Excerpt: "JAIGP is an experimental journal launched in February 2026... it has no Impact Factor and has not gained official recognition from traditional indexing systems... it continues efforts to build academic rigor, having published 43 papers and introduced a multi-stage peer review process."  
Context: 斯坦福HAI 2026 AI Index报告将"2025年首篇完全AI生成论文在同行评审workshop被接受"列为年度关键发现之一，但也指出实验验证的AI发现列表仍然很短。  
Confidence: high

Claim: Nature 2026年3月26日同期发表社论"AI scientists are changing research — institutions, funders and publishers must respond"，强调学术界、资助机构和出版商必须回应AI科学家的兴起，建立规范以确保这一技术负责任地发展。[^33^]
Source: Nature Editorial  
URL: https://www.nature.com/articles/d41586-026-00969-z（关联社论）  
Date: 2026-03-26  
Excerpt: "AI scientists are changing research — institutions, funders and publishers must respond."  
Context: 这一社论标志着主流科学出版界正式承认AI自主研究Agent的崛起，并呼吁系统性政策回应。  
Confidence: high

### 6.4 伦理与"超越人类理解"的风险

Claim: 研究者提出"AI transcendence"（AI超越）概念作为自主AI参与研究的伦理威胁：完全自主的AI系统可能开始产生如此复杂和先进的结果、发现和创新，以至于超出人类理解范围。这导致两个威胁：(1) 可能违背人类利益的行动（若无法理解AI产生的科学，如何控制/验证/信任其输出？）；(2) 可能使人类科学家过时。预防性方法支持为AI系统设置护栏，确保科学保持人类可理解和可控。[^34^]
Source: Springer / AI and Ethics  
URL: https://link.springer.com/article/10.1007/s43681-025-00908-0  
Date: 2026-02-04  
Excerpt: "Fully autonomous AI systems might start producing results, discoveries, and innovations that are so complex and advanced that they are beyond human understanding... A precautionary approach to AI development supports a policy of placing guardrails on AI systems so that they do not become fully autonomous and the science they develop remains within human comprehension and control."  
Context: 这一论述呼应了Arthur C. Clarke的名言："任何足够先进的技术都与魔法无异。"  
Confidence: high

---

## 七、综合对比表

| 系统 | 机构 | 发布时间 | 核心架构 | 关键突破 | 成本/效率 | 主要局限 |
|---|---|---|---|---|---|---|
| AI Scientist v1 | Sakana AI | 2024-08 | 模板驱动线性流水线 | 首个端到端ML论文生成 | ~$15/篇 | 依赖人工模板，难泛化 |
| AI Scientist v2 | Sakana AI | 2025-04 | Agentic Tree Search + VLM | 首篇AI论文过同行评审 | ~$20-25/篇 | 57%含虚假数据，仅workshop级 |
| Aletheia | Google DeepMind | 2026-02 | Generator-Verifier-Reviser | 解决4个Erdős开放问题 | N/A | 68.5%响应错误，仅数学 |
| AI Co-scientist | Google | 2025-02 | Supervisor + 6 Worker Agents | AML药物/肝纤维化验证 | N/A | 第二层级架构，无法动态重配置 |
| Robin | FutureHouse | 2025-05 | Crow+Falcon+Finch编排 | 首篇AI发现药物(ripasudil) | N/A | 人类仍需执行物理实验 |
| Kosmos | Edison Scientific | 2025-11 | 结构化世界模型 | 1,500论文/42,000行代码/次 | 商业化 | 未公开完整技术细节 |
| EvoScientist | Huawei | 2026-03 | RA+EA+EMA+持久记忆 | 自我进化，6篇论文被接收 | N/A | 依赖静态基线，ICAIS为新会议 |
| AutoResearchClaw | UNC Chapel Hill | 2026-03 | 23阶段8阶段管道 | 开源，MetaClaw跨运行学习 | $5-15/次 | 设置复杂，55+初期issue |
| UniScientist | UniPat-AI | 2026-03 | 50+学科通用范式 | 30B模型超Claude/GPT | N/A | 信息有限，待验证 |

---

## 八、关键趋势与判断

1. **从辅助到自主的演进**：2024-2025年的系统（AI Scientist v1、AI Co-scientist）主要为人类辅助工具；2025-2026年的系统（Aletheia、Robin、EvoScientist）开始实现关键步骤的完全自主。

2. **多Agent编排成为共识**：几乎所有领先系统均采用多Agent架构，但协调机制从静态Supervisor-Worker（AI Co-scientist）向动态进化（EvoScientist的EMA）和树搜索（AI Scientist v2的BFTS）演进。

3. **自然语言验证 vs 形式验证**：Aletheia选择自然语言端到端路径（与AlphaProof的形式语言路径对比），这降低了门槛但也引入了更多幻觉风险。

4. **质量与速度的权衡**：AI Scientist v2展示了$20生成一篇论文的可能性，但质量方差巨大（6.33 vs 3-4分）。学术界需要建立新的质量评估框架。

5. ** wet-lab闭环验证是关键差异化因素**：AI Co-scientist和Robin的价值不仅在于假设生成，更在于其实验验证能力（AML细胞系、肝类器官、RNA-seq）。纯计算系统（AI Scientist）在可信度上处于劣势。

6. **商业化加速**：FutureHouse→Edison Scientific（$70M/2.5亿估值）、Sakana AI（$135M Series B/26.5亿估值）显示资本市场对AI科学家赛道的高度热情。

7. **伦理和出版规范滞后**：尽管技术快速进步，学术界对AI生成内容的作者身份、同行评审透明度、引用验证等规范尚未形成共识。幻觉引用从2024年的0.3%飙升至2025年的2.6%，构成对科学诚信的结构性威胁。

---

## 九、反面论点汇总

1. **"里程碑被系统性夸大"**：媒体将workshop接受等同于顶级会议发表，将人类撰写的Nature论文误传为"AI发表在Nature"。实际上截至2026年4月，无一篇AI solo作者的论文发表在主要同行评审期刊。[^32^]

2. **"接受率语境缺失"**：ICLR 2025 ICBINB workshop接受率~32.6%（甚至Nature报道为70%），远高于主会32%。且论文被主动撤回，从未正式发表。[^3^][^8^]

3. **"可靠性问题被低估"**：57%假数据率、68.5%根本错误率、引用幻觉激增——这些数字表明当前AI科学家系统远未达到生产就绪标准。

4. **"可复制性危机"**：LLM的生成性本质导致"ghost citations"（幽灵引用）——将统计相关的token组合成看似真实的引用。这一问题在现有同行评审流程中几乎无法被检测。[^28^]

5. **"AI作者身份的法律与伦理真空"**：COPE/ICMJE明确AI不能作为作者，但AI辅助撰写和完全AI生成之间的界限模糊。JAIGP等实验性期刊的出现可能加剧而非解决这一问题。

---

## 十、信息来源索引

| 编号 | 来源 | 类型 | 日期 |
|---|---|---|---|
| [^1^] | arXiv:2504.08066 (AI Scientist-v2) | 学术论文 | 2025-04 |
| [^2^] | Sakana AI Blog / Nature 651 | 官方博客/期刊 | 2026-03 |
| [^3^] | arXiv:2504.08066 附录 | 学术论文 | 2025-04 |
| [^4^] | GitHub hermes-multi-agent引用 | 代码仓库/综述 | 2026-04 |
| [^5^] | Pebblous Blog / GitHub SakanaAI | 技术博客 | 2026-03 |
| [^6^] | Pebblous AI Science Fact Check | 事实核查报告 | 2026-04 |
| [^7^] | arXiv:2502.14297 (Evaluation of AI Scientist) | 学术论文 | 2025-02 |
| [^8^] | arXiv:2504.08066 附录评审意见 | 学术论文 | 2025-04 |
| [^9^] | ByteIota | 科技媒体 | 2026-03 |
| [^10^] | arXiv:2602.10177 (Aletheia) | 学术论文 | 2026-02 |
| [^11^] | DeepMind Blog | 官方博客 | 2026-02 |
| [^12^] | arXiv:2602.10177 第3节 | 学术论文 | 2026-03 |
| [^13^] | arXiv:2601.22401 (Erdős Case Study) | 学术论文 | 2026-02 |
| [^14^] | MarkTechPost | 科技媒体 | 2026-02 |
| [^15^] | TechJackSolutions | 分析文章 | 2026-04 |
| [^16^] | arXiv:2502.18864 (AI Co-scientist) | 学术论文 | 2025-02 |
| [^17^] | arXiv:2502.18864v1 第4节 | 学术论文 | 2025-02 |
| [^18^] | bioRxiv / Advanced Science | 学术论文 | 2025-04 |
| [^19^] | arXiv:2602.19810 引用 | 学术论文 | 2026 |
| [^20^] | arXiv:2505.13400 (Robin) | 学术论文 | 2025-05 |
| [^21^] | TechFundingNews / Caproasia | 科技媒体 | 2025-12 |
| [^22^] | FutureHouse Blog / arXiv | 官方博客/论文 | 2025-05 |
| [^23^] | arXiv:2603.08127 (EvoScientist) | 学术论文 | 2026-03 |
| [^24^] | arXiv:2603.08127 附录E | 学术论文 | 2026-03 |
| [^25^] | GitHub aiming-lab/AutoResearchClaw | 代码仓库 | 2026-03 |
| [^26^] | arXiv:2604.01007 (Omni-SimpleMem) | 学术论文 | 2026-04 |
| [^27^] | GitHub UniPat-AI/UniScientist | 代码仓库 | 2026-03 |
| [^28^] | Nature News (Hallucinated Citations) | 权威期刊 | 2026-04 |
| [^29^] | PubMed / Accountability in Research | 学术期刊 | 2026 |
| [^30^] | HealthManagement / Nature | 医学管理媒体 | 2026-04 |
| [^31^] | arXiv:2601.22401 | 学术论文 | 2026-02 |
| [^32^] | Pebblous Fact Check | 分析报告 | 2026-04 |
| [^33^] | Nature Editorial | 权威期刊社论 | 2026-03 |
| [^34^] | Springer / AI and Ethics | 学术期刊 | 2026-02 |

---

> **研究说明**：本报告基于20+次独立网络搜索，覆盖arXiv预印本、Nature/Science等顶级期刊、官方技术博客、权威科技媒体（TechCrunch、InfoQ、MarkTechPost等）及学术伦理期刊。所有引用均追溯原始出处。信息截止至2026年4月。
