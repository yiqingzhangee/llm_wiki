# 维度01: AutoResearch核心原型与实验循环Agent

## 研究概述

本维度聚焦Karpathy于2026年3月发布的`autoresearch`项目及其引爆的autoresearch类别。研究覆盖：(1) 核心技术架构与实验循环机制；(2) 学术基准测试与LLM-based HPO vs经典方法的系统对比；(3) 衍生生态系统的技术差异；(4) 与主流编码Agent的融合路径；(5) 技术边界与失败模式。所有关键发现均标注来源与置信度。

---

## 1. Karpathy autoresearch核心原型

### 1.1 项目起源与定位

Claim: Andrej Karpathy于2026年3月6-7日发布autoresearch，是一个仅630行Python、MIT许可证的极简框架，让AI coding agent在单GPU上自主运行ML训练实验 [^1^]
Source: DataCamp Guide to AutoResearch
URL: https://www.datacamp.com/tutorial/guide-to-autoresearch
Date: 2026-03-23
Excerpt: "Released on March 7, 2026, the project picked up 21,000+ GitHub stars and 8.6 million views on Karpathy's announcement within days."
Context: 这是autoresearch类别的原点事件
Confidence: high

Claim: Karpathy的原始运行中，agent在48小时内完成约700次实验，发现20个可叠加的优化，在depth-12模型上将验证loss从0.862415降至0.858039，且所有优化成功transfer到depth-24大模型，实现11%速度提升 [^2^]
Source: AI Automation Global Blog
URL: https://aiautomationglobal.com/blog/karpathy-autoresearch-autonomous-ai-agent-2026
Date: 2026-03-22
Excerpt: "AutoResearch ran 700 autonomous experiments in 48 hours, discovering 20 training optimizations that transferred from small to large models — achieving an 11% speed improvement that human researchers had missed."
Context: 这是验证autoresearch模式有效性的核心数据点
Confidence: high

Claim: Shopify CEO Tobi Lütke内部运行autoresearch后，37次实验 overnight 获得19%性能提升；应用于Shopify模板引擎后获得53%渲染速度提升 [^3^]
Source: Product Growth Blog (Aakash Gupta)
URL: https://www.news.aakashg.com/p/autoresearch-guide-for-pms
Date: 2026-03-19
Excerpt: "Shopify CEO Tobi Lutke pointed it at Shopify's templating engine and got 53% faster rendering from 93 automated commits."
Context: 企业级验证案例
Confidence: medium（基于社交媒体和博客报道，非官方发布）

### 1.2 三文件架构设计

Claim: autoresearch的核心架构仅包含三个文件：`prepare.py`（数据准备、tokenizer、常量，只读）、`train.py`（模型+优化器+训练循环，agent可修改）、`program.md`（自然语言形式的研究组织代码/指令） [^4^]
Source: OpenRepoGuide
URL: https://openrepoguide.com/
Date: 2026
Excerpt: "Prepare the environment `prepare.py` handles data preparation... Let the agent modify training logic `train.py` is the main file the agent is expected to change... Guide the research loop `program.md` acts like an instruction sheet for the agent"
Context: 架构设计哲学是极度简化以降低agent认知负担
Confidence: high

Claim: `program.md`作为"研究组织代码"(research organization code)，将自然语言转化为agent编排，人类只需编辑markdown文件定义研究策略、约束和评估标准，agent读取后自主执行 [^5^]
Source: Webvise Blog
URL: https://webvise.io/blog/karpathy-autoresearch-ai-experiments
Date: 2026-04-02
Excerpt: "Writing a good program.md requires understanding what approaches are worth exploring, what constraints matter, and what success actually looks like. This is higher-level work than writing the code, but it is not easier."
Context: program.md机制是autoresearch区别于传统AutoML/HPO的核心创新
Confidence: high

Claim: 单文件修改约束是刻意设计的关键legibility feature——当agent只能触碰一个文件时，理解其做了什么只需要review一个diff，而不需要reverse engineering整个代码库 [^6^]
Source: Webvise Blog
URL: https://webvise.io/blog/karpathy-autoresearch-ai-experiments
Date: 2026-04-02
Excerpt: "The single-file constraint in AutoResearch is not just a technical limitation; it is a legibility feature. When agents can touch anything, understanding what they did requires significant reverse engineering."
Context: 可解释性设计哲学
Confidence: high

### 1.3 实验循环(Ratchet Loop)实现

Claim: 核心循环是"ratchet loop"（棘轮循环）：agent提出假设 → 修改`train.py` → 运行固定5分钟实验 → 比较val_bpb vs当前最佳 → 如果改进则git commit保留，否则git revert丢弃 → 无限重复 [^7^]
Source: DataCamp Guide to AutoResearch
URL: https://www.datacamp.com/tutorial/guide-to-autoresearch
Date: 2026-03-23
Excerpt: "Each experiment: modify the file → eval → keep or revert → record → repeat. Forever."
Context: 固定时间预算使实验间可比，无论agent做了什么修改
Confidence: high

Claim: 没有orchestration脚本，没有`run.py`。LLM本身就是automation layer。用户只需在repo中启动Claude Code/Codex并让其读取program.md，agent自行运行循环 [^8^]
Source: DataCamp Guide to AutoResearch
URL: https://www.datacamp.com/tutorial/guide-to-autoresearch
Date: 2026-03-23
Excerpt: "There is no orchestration script. No `run.py`, no pipeline, no framework. The README says to 'simply spin up your Claude/Codex or whatever you want in this repo.' The LLM is the automation layer."
Context: 极致的架构简化——依赖agent自身能力而非外部编排
Confidence: high

Claim: 典型的overnight运行（约8-12小时）可完成80-100次实验，其中约15-20次改进被保留。结果记录在`results.tsv`中，git log显示每次保留的commit [^9^]
Source: DataCamp Guide to AutoResearch
URL: https://www.datacamp.com/tutorial/guide-to-autoresearch
Date: 2026-03-23
Excerpt: "For a first overnight run, expect 80-100 experiments with maybe 15-20 improvements kept."
Context: 典型性能预期
Confidence: high

Claim: 默认配置在FineWeb-Edu数据集上训练GPT模型，但Karpathy推荐在较小硬件上切换到TinyStories数据集并将vocab size降至256、depth降至4以降低VRAM需求 [^10^]
Source: DataCamp Guide to AutoResearch
URL: https://www.datacamp.com/tutorial/guide-to-autoresearch
Date: 2026-03-23
Excerpt: "For a first test on smaller hardware, Karpathy recommends switching to the TinyStories dataset and scaling down the model by reducing vocabulary size to 256 and depth to 4."
Context: 硬件适配策略
Confidence: high

### 1.4 Keep/Revert决策逻辑

Claim: Keep/Revert是纯粹的二进制贪婪决策（greedy hill-climbing）：如果val_bpb比当前最佳值更好（或更差，取决于优化方向），则保留；否则完全回滚到上一个commit [^11^]
Source: Ry Walker Research
URL: https://rywalker.com/research/autoresearch
Date: 2026-03-13
Excerpt: "The `program.md` simplicity criterion captures the philosophy: 'A tiny improvement that adds 20 lines of hacky code? Probably not worth it. A tiny improvement from deleting code? Definitely keep.'"
Context: 纯粹的贪婪爬山算法，非贝叶斯优化或进化策略
Confidence: high

Claim: Karpathy本人强调`program.md`中应编码"品味"(taste)判断：agent应偏好删除代码带来的小改进，而非增加hacky代码 [^12^]
Source: Paddo.dev Blog
URL: https://paddo.dev/blog/autoresearch-overnight-lab/
Date: 2026-03-14
Excerpt: "The `program.md` simplicity criterion captures the philosophy: 'A tiny improvement that adds 20 lines of hacky code? Probably not worth it. A tiny improvement from deleting code? Definitely keep.' Taste, encoded as text, applied at scale."
Context: 人类通过program.md将隐性的研究品味编码为agent可读的规则
Confidence: high

---

## 2. 学术基准测试：LLM-based HPO vs经典方法

### 2.1 核心基准论文

Claim: Ferreira等人（ELLIS Institute Tübingen、University of Freiburg、KIT、Prior Labs）在arXiv发表"Can LLMs Beat Classical Hyperparameter Optimization Algorithms? A Study on autoresearch"（arXiv:2603.24647），这是autoresearch类别最系统的学术基准研究 [^13^]
Source: arXiv 2603.24647
URL: https://arxiv.org/abs/2603.24647
Date: 2026-03-25 (v1), 2026-04-13 (v4)
Excerpt: "The autoresearch repository enables an LLM agent to optimize hyperparameters by editing training code directly. We use it as a testbed to compare classical HPO algorithms against LLM-based methods on tuning the hyperparameters of a small language model under a fixed compute budget."
Context: 该论文提供了9种HPO方法在相同24小时GPU预算下的头对头比较
Confidence: high

Claim: 在固定搜索空间内，经典HPO方法（CMA-ES、TPE、SMAC）始终优于纯LLM-based agent。最佳方法排名：Centaur (0.9763) > TPE (0.9768) > SMAC (0.9778) > CMA-ES (0.9785) > Karpathy Agent Code (0.9814) [^14^]
Source: arXiv 2603.24647 (v4)
URL: https://arxiv.org/html/2603.24647v4
Date: 2026-04-13
Excerpt: "Classical methods such as CMA-ES and TPE consistently outperform LLM-based agents within the fixed search space. The exception is Karpathy Agent (Code), which directly edits training source code and is competitive with classical methods."
Context: 核心结论：经典方法在固定搜索空间内更强，但无约束代码编辑使LLM agent接近竞争力
Confidence: high

Claim: OOM（Out-of-Memory）避免率是比搜索多样性更重要的性能预测因子。顶部方法的OOM率≤16%，底部方法>36%。LLAMBO (Paper)尽管搜索最多样，但因48% OOM率而表现差 [^15^]
Source: arXiv 2603.24647 (v4)
URL: https://arxiv.org/html/2603.24647v4
Date: 2026-04-13
Excerpt: "Methods that avoid out-of-memory failures outperform those with higher search diversity, suggesting that reliability matters more than exploration breadth... LLAMBO (Paper) is one of the most diverse methods yet underperforms classical methods due to its 48% OOM rate."
Context: 揭示了LLM agent在状态跟踪方面的根本局限——即使观察到完整trial历史，OOM率仍与random search相当
Confidence: high

### 2.2 Centaur：LLM+经典优化器的混合方法

Claim: 论文提出Centaur——将CMA-ES的完整内部状态（mean vector μ、step-size σ、covariance matrix C）共享给LLM的混合方法。Centaur在所有实验中取得最佳结果，且0.8B LLM已足以超越所有经典和纯LLM方法 [^16^]
Source: arXiv 2603.24647 (v4)
URL: https://arxiv.org/html/2603.24647v4
Date: 2026-04-13
Excerpt: "We introduce Centaur, a hybrid that shares CMA-ES's interpretable internal state, including mean vector, step-size, and covariance matrix, with an LLM. Centaur achieves the best result in our experiments, and a 0.8B LLM already suffices to outperform all classical and pure LLM methods."
Context: 核心洞察：LLM最有效的方式是作为经典优化器的补充而非替代；CMA-ES的内部状态对人类/LLM都"可解释"
Confidence: high

Claim: Centaur仅将LLM用于约30%的trials（其余由CMA-ES独立 propose），0.8B模型在混合模式下足够，但纯代码编辑需要27B+规模。Centaur配合Opus 4.6达到最佳结果0.9739 [^17^]
Source: arXiv 2603.24647 (v4)
URL: https://arxiv.org/html/2603.24647v4
Date: 2026-04-13
Excerpt: "Centaur achieved the best result in our experiments by using the LLM on only 30% of trials... Centaur [Opus 4.6] achieves the best result (0.9739), extending its lead."
Context: 模型scaling对混合方法比纯LLM agent更有收益
Confidence: high

### 2.3 模型Scaling与Frontier Model测试

Claim: 从0.8B到27B的scaling对固定搜索空间内的LLM方法几乎无收益；对无约束代码编辑则至关重要。即使使用Claude Opus 4.6和Gemini 3.1 Pro Preview等frontier模型，纯LLM agent仍无法超越最佳经典方法 [^18^]
Source: arXiv 2603.24647 (v4)
URL: https://arxiv.org/html/2603.24647v4
Date: 2026-04-13
Excerpt: "Scaling from 0.8B to 27B to Gemini Pro yields no improvement for hybrid optimization, but Centaur with Opus 4.6 breaks through this plateau... simply scaling the LLM does not outperform classical methods."
Context: 挑战了"scaling万能论"——更强的LLM不一定能解决优化状态跟踪问题
Confidence: high

Claim: Gemini 3.1 Flash-Lite在无约束代码编辑中失败率高达87-94%（主要是runtime error而非OOM），但在固定搜索空间内与Qwen3.5-0.8B表现相当 [^19^]
Source: arXiv 2603.24647 (v4), Appendix A.3
URL: https://arxiv.org/html/2603.24647v4
Date: 2026-04-13
Excerpt: "Gemini 3.1 Flash-Lite fails to generate valid code in 87–94% of trials for unconstrained editing, but performs comparably to Qwen3.5-0.8B in the fixed search space."
Context: 代码编辑能力对模型能力要求远高于固定参数采样
Confidence: high

### 2.4 RoboPhD：三种优化范式的系统比较

Claim: Borthwick等人在"RoboPhD: Evolving Diverse Complex Agents Under Tight Evaluation Budgets"（arXiv:2604.04347）中首次系统比较了三种优化范式：Elo tournament selection (RoboPhD)、Pareto-based selection (GEPA)、greedy hill-climbing (Autoresearch)，在固定1500次评估预算下跨越4个benchmark [^20^]
Source: arXiv 2604.04347
URL: https://arxiv.org/abs/2604.04347
Date: 2026-04-06
Excerpt: "We present the first systematic comparison of three optimization paradigms -- Elo tournament selection (RoboPhD), Pareto-based selection (GEPA), and greedy hill-climbing (Autoresearch) -- across four benchmarks spanning abstract reasoning, cloud scheduling, SQL generation, and financial QA."
Context: 这是autoresearch作为通用优化范式被纳入学术比较的标志
Confidence: high

Claim: RoboPhD在4个benchmark中的3个上优于GEPA和Autoresearch，仅在最简单任务上失利（Autoresearch在该任务上以不到90行代码获胜）。在ARC-AGI上，RoboPhD将22行seed agent进化到1013行的多策略系统，准确率从27.8%提升至65.8% [^21^]
Source: arXiv 2604.04347
URL: https://arxiv.org/pdf/2604.04347
Date: 2026-04-06
Excerpt: "Using a single default configuration, RoboPhD outperforms both GEPA and Autoresearch on three of four benchmarks, losing only on the simplest task... On ARC-AGI, RoboPhD evolves a 22-line seed agent into a 1,013-line multi-strategy system, improving accuracy from 27.8% to 65.8%."
Context: 验证了贪婪爬山(autoresearch)在简单任务上有效，但复杂任务需要更多样化的搜索策略
Confidence: high

Claim: 所有三个系统接收的seed agent均包含diagnostic `print()`语句，进化过程可以增长这些诊断输出，形成self-instrumenting agents——后代agent可以从前代进化出的诊断信息中获益 [^22^]
Source: arXiv 2604.04347
URL: https://arxiv.org/pdf/2604.04347
Date: 2026-04-06
Excerpt: "All three systems receive seed agents with diagnostic print() statements that evolution can grow, enabling self-instrumenting agents that develop increasingly informative diagnostics for the benefit of their evolutionary successors."
Context: 自仪器化(self-instrumentation)是agent进化中的新概念
Confidence: high

### 2.5 Multi-Agent协作研究

Claim: "An Empirical Study of Multi-Agent Collaboration for Automated Research"（arXiv:2603.29632）分析了单agent、subagent、agent teams三种模式在autoresearch任务上的稳定性。Subagent模式（多worker在独立worktree中并行探索）具有最低的preflight failure和crash rate；Agent teams（多专家顺序编辑同一脚本）因tensor维度不兼容等问题导致failure rate显著升高 [^23^]
Source: arXiv 2603.29632
URL: https://arxiv.org/html/2603.29632v1
Date: 2026-03-31
Excerpt: "The subagent mode functions as a highly stable and efficient empirical search engine... Conversely, the agent teams reveal a fundamental trade-off between theoretical deliberation and operational fragility."
Context: 多agent协作在稳定性上不如单agent或subagent模式
Confidence: high

---

## 3. 衍生生态系统

### 3.1 生态系统概览（awesome-autoresearch列表）

Claim: WecoAI维护的awesome-autoresearch curated list系统追踪了autoresearch的衍生变体，截至2026年4月主要变体包括：pi-autoresearch(3,591⭐)、autoresearch-mlx(~633⭐)、autoresearch-win-rtx(~158⭐)、autoresearch-at-home(469⭐)、autokernel(1,206⭐)、autoresearch Claude Skill等 [^24^]
Source: awesome-autoresearch (GitHub - WecoAI)
URL: https://github.com/WecoAI/awesome-autoresearch
Date: 2026-03-22
Excerpt: "|Project|Description|Links| |autoresearch|The original - single GPU, 630 lines of Python|...| |pi-autoresearch|Generalized as a Pi extension. Works for any optimization target|...|"
Context: 这是追踪autoresearch生态演化的权威社区资源
Confidence: high

Claim: Ry Walker Research的市场地图显示，截至2026年4月，autoresearch原始repo已达72,100⭐，pi-autoresearch 3,591⭐，autokernel 1,206⭐，Hyperspace AGI 1,288⭐，evo 4,110⭐ [^25^]
Source: Ry Walker Research - Autoresearch Tools Compared
URL: https://rywalker.com/research/autoresearch-tools
Date: 2026-04-14
Excerpt: "|Tool|⭐ Stars|Created|Domain|Key Differentiator| |karpathy/autoresearch|72,100|Mar 6, 2026|LLM training|The original. `program.md` as org chart|"
Context: 生态爆炸式增长的量化证据
Confidence: high

### 3.2 主要衍生变体技术差异

#### pi-autoresearch（通用metric扩展）

Claim: pi-autoresearch由davebcn87创建，将autoresearch模式泛化为Pi编辑器扩展，支持任意优化目标——测试速度、bundle size、build times、Lighthouse scores。引入persistent sessions（重启后恢复）、dashboard UI、branch-aware tracking [^26^]
Source: awesome-autoresearch
URL: https://github.com/WecoAI/awesome-autoresearch
Date: 2026-03-22
Excerpt: "|pi-autoresearch|Generalized as a Pi extension. Works for any optimization target - test speed, bundle size, build times, Lighthouse scores|"
Context: 首次将autoresearch从ML训练泛化到任意软件工程指标
Confidence: high

#### autoresearch-mlx（Apple Silicon移植）

Claim: autoresearch-mlx将autoresearch移植到Apple MLX框架，无需PyTorch，利用unified memory。M4 Max在depth-24配置下从baseline 2.667降至1.294 val_bpb [^27^]
Source: Particula.tech Blog
URL: https://particula.tech/blog/karpathy-autoresearch-autonomous-ml-experiments
Date: 2026-03-18
Excerpt: "|autoresearch-mlx|701|Apple Silicon/Mac port via MLX. M4 Max hit val_bpb of 1.294 from 2.667 baseline|"
Context: 首次证明autoresearch模式可在非NVIDIA硬件上运行
Confidence: medium（具体数值来自社区报告，非官方benchmark）

#### autokernel（GPU kernel优化）

Claim: AutoKernel（RightNow-AI/autokernel，1,206⭐）将autoresearch循环移植到GPU kernel优化：LLM agent修改单个kernel文件，固定benchmark harness验证correctness和测量throughput，每次迭代约90秒，overnight运行产生300-400次实验。包含9个Triton + 9个CUDA C++ starter kernel，以及基于Amdahl's law的orchestrator [^28^]
Source: arXiv 2603.21331
URL: https://arxiv.org/html/2603.21331v1
Date: 2026-03-22
Excerpt: "AutoKernel mechanizes this loop. An LLM agent modifies a single kernel file, a fixed benchmark harness verifies correctness and measures throughput, and the result determines whether the change persists. Each iteration takes roughly 90 seconds."
Context: 将autoresearch模式从ML训练代码编辑扩展到GPU kernel代码优化
Confidence: high

#### autoresearch-at-home（分布式协作）

Claim: autoresearch-at-home实现SETI@home风格的分布式agent协作，多个agent在不同GPU上通过Ensue共享内存网络协调。首次协调运行中20+ agents在54小时内完成1000+实验，提升验证性能3.2%。核心机制包括实验认领（防重复）、结果共享（无论成败）、全局最佳追踪、假设交换 [^29^]
Source: Ensue Blog
URL: https://ensue.dev/blog/autoresearch-at-home/
Date: 2026-03-12
Excerpt: "In our first coordinated run, 20+ agents completed over 1,000 experiments in 54 hours and improved validation performance by 3.2%."
Context: Karpathy SETI@home愿景的首个社区实现
Confidence: medium（来自项目方博客，规模较小）

#### N-Autoresearch（多GPU并行基础设施）

Claim: N-Autoresearch（iii-hq/n-autoresearch）用结构化实验状态替代bash loop和git-as-state。包含22个函数的Orchestrator（Python）和Rust编写的GPU Worker。N个GPU = N个并行实验，atomic GPU acquisition防止冲突，全局自适应搜索策略（explore/exploit/combine/ablation） [^30^]
Source: GitHub iii-hq/n-autoresearch
URL: https://github.com/iii-hq/n-autoresearch
Date: 2026-04-19
Excerpt: "N GPU workers = N parallel experiments. Each agent acquires a GPU, trains, records, releases. Search strategy adapts globally across all GPUs."
Context: 将autoresearch从顺序执行升级为并行实验基础设施
Confidence: high

### 3.3 上游愿景：AgentHub与分布式协作

Claim: Karpathy于2026年3月8日提出AgentHub概念——"GitHub is for humans. AgentHub is for agents." 这是一个面向AI agent swarm的协作平台原型：bare git repo + message board，无main branch、无PRs、无merges，只有 sprawling DAG of commits + agent协调message board。首个用途是作为autoresearch的组织层 [^31^]
Source: GitHub karpathy/agenthub
URL: https://github.com/karpathy/agenthub
Date: 2026-03-09
Excerpt: "Think of it as a stripped-down GitHub where there's no main branch, no PRs, no merges — just a sprawling DAG of commits going in every direction, with a message board for agents to coordinate."
Context: 从模拟单个PhD学生到模拟整个研究社区的架构愿景
Confidence: high

Claim: 社区开发者提出了GNAP（Git-Native Agent Protocol）作为SETI@home风格agent swarm的协调层——用git作为协调substrate，agent通过atomic commit claim任务、push结果，无需中心服务器 [^32^]
Source: GitHub Issue karpathy/autoresearch #249
URL: https://github.com/karpathy/autoresearch/issues/249
Date: 2026-03-13
Excerpt: "Git as the coordination substrate. Instead of a central server that agents ping, they read/write task state to a shared git repo."
Context: 社区对Karpathy愿景的快速响应
Confidence: high

---

## 4. 与编码Agent的融合

### 4.1 Claude Code集成

Claim: Claude Code是autoresearch最常被使用的agent后端。Udit Goenka于2026年3月13日发布generalized Claude Code skill，将autoresearch模式扩展到任何有可测量指标的领域（安全审计、API性能优化、代码质量改进等），获得608⭐ [^33^]
Source: SimpleNews.ai
URL: https://www.simplenews.ai/news/karpathys-autoresearch-pattern-spawns-generalized-claude-code-skill-for-autonomous-iteration-srsk
Date: 2026-03-16
Excerpt: "Developer Udit Goenka recognized the pattern's broader potential and released a generalized Claude Code skill on March 13, 2026, that extends autonomous iteration beyond machine learning to any domain with measurable outcomes."
Context: Claude Code的file-based workflow与autoresearch天然契合
Confidence: high

Claim: Claude Code配合autoresearch进行prompt优化时，每cycle约使用18,000 tokens，成本$0.05-0.15/cycle，overnight 30 cycles约$1.50-4.50。使用Claude Haiku做eval、Sonnet做variant generation可降低成本60-70% [^34^]
Source: MindStudio Blog
URL: https://www.mindstudio.ai/blog/claude-code-autoresearch-self-improving-skills/
Date: 2026-03-14
Excerpt: "At 20 test cases, 3 variants per cycle, and ~300 tokens per response, each cycle uses roughly 18,000 tokens... A full overnight run of 30 cycles costs $1.50–$4.50."
Context: 成本分析对推广autoresearch模式至关重要
Confidence: medium（估算性数据）

Claim: MindStudio的指南展示了如何用Claude Code + AutoResearch loop自动改进AI skill：binary eval assertions → baseline pass rate → Claude Code分析失败模式 → 生成改进prompt variants → 评估 → 保留最佳。30-50 cycles可将40-50% pass rate提升至75-85% [^35^]
Source: MindStudio Blog
URL: https://www.mindstudio.ai/blog/claude-code-autoresearch-self-improving-skills/
Date: 2026-03-14
Excerpt: "Over 8 hours overnight, you can complete 30–50 cycles. That's enough to push most prompts from 40–50% pass rates into the 75–85% range."
Context: 将autoresearch应用于prompt engineering的具体方法论
Confidence: medium

### 4.2 Codex集成

Claim: Codex Autoresearch Loop是ara.so发布的Codex skill，实现了modify→verify→keep-or-revert循环，用户只需提供单句目标（如"移除所有TypeScript any类型"），Codex自动推断scope、metric、verification和guard commands。支持git-backed checkpoints [^36^]
Source: LobeHub Skills Marketplace
URL: https://lobehub.com/de/skills/aradotso-trending-skills-codex-autoresearch-loop
Date: 2026-04-06
Excerpt: "Codex Autoresearch is an autonomous Codex skill that repeatedly applies a modify→verify→keep-or-revert loop to your repository until a measurable goal is met or a cap is reached."
Context: Codex原生支持skill系统，使autoresearch集成更加结构化
Confidence: high

### 4.3 Cursor集成

Claim: Cursor Autoresearch Extension（ergenekonyigit.cursor-autoresearch）在VS Code中提供status visibility、session actions、dashboard export、results panel。通过SSE（server-sent events）提供实时dashboard，但optimization loop本身在shared packages和MCP tools中实现 [^37^]
Source: VS Code Marketplace
URL: https://marketplace.visualstudio.com/items?itemName=ergenekonyigit.autoresearch
Date: 2026-04-16
Excerpt: "This extension improves observability and ergonomics for that loop inside VS Code. The MCP server and core engine remain source-of-truth for experiment execution."
Context: Cursor生态中autoresearch的可视化和工程化
Confidence: high

### 4.4 通用CLI工具：autoresearch-cli

Claim: autoresearch-cli（199-biotechnologies）提供了跨agent的通用CLI工具，支持`autoresearch install all`一次性为Claude Code + Codex + OpenCode + Cursor + Windsurf安装autoresearch loop instructions。包含`autoresearch.toml`配置、`autoresearch doctor`预检（14项检查）、以及`status/log/best/report/diff/export`等命令 [^38^]
Source: GitHub 199-biotechnologies/autoresearch-cli
URL: https://github.com/199-biotechnologies/autoresearch-cli
Date: 2026-03-23
Excerpt: "autoresearch install all -- This writes the autoresearch loop instructions in each agent's native format — SKILL.md for Claude/Codex/OpenCode, `.mdc` for Cursor, frontmattered `.md` for Windsurf."
Context: autoresearch正在从项目模式演化为标准化的agent workflow工具
Confidence: high

### 4.5 Evo：超越贪婪爬树的增强循环

Claim: Evo（evo-hq/evo）在autoresearch的贪婪爬山基础上增加了树搜索（多方向fork）、并行半自主subagents（各自在git worktree中运行）、共享状态（失败trace、annotation、discarded hypotheses）、gating（regression tests作为安全gate）、benchmark discovery（自动发现要测量的指标） [^39^]
Source: GitHub evo-hq/evo
URL: https://github.com/evo-hq/evo
Date: 2026-04-05
Excerpt: "Evo adds structure on top of that idea: Tree search over greedy hill climb... Parallel semi-autonomous agents... Shared state... Gating... Observability... Benchmark discovery."
Context: 代表了autoresearch从简单循环向复杂搜索结构的进化方向
Confidence: high

### 4.6 Ouro Loop：带Guardrails的通用框架

Claim: Ouro Loop（VictorVVedtion/ouro-loop）将autoresearch模式从ML扩展到通用软件工程，引入BOUND系统定义绝对约束（DANGER ZONES、NEVER DO rules、IRON LAWS），agent在Build→Verify→Self-Fix循环中自主运行，约束通过Claude Code Hooks在runtime层面硬拦截（exit 2 hard-block） [^40^]
Source: GitHub VictorVVedtion/ouro-loop
URL: https://github.com/VictorVVedtion/ouro-loop
Date: 2026-03-15
Excerpt: "Constraints are enforced at the runtime level through Claude Code Hooks (exit 2 hard-block), not by relying on the agent's 'good behavior.'"
Context: 解决autoresearch在通用软件工程中的安全性问题
Confidence: high

---

## 5. SkyPilot多GPU扩展：从顺序到并行的质变

### 5.1 核心发现

Claim: SkyPilot团队（UC Berkeley Sky Computing Lab）将autoresearch从1 GPU扩展至16 GPU（13 H100 + 3 H200 via Kubernetes on CoreWeave），Claude Code在8小时内提交约910次实验，实现9x吞吐量提升，val_bpb从1.003降至0.974（2.87%改进），与顺序模拟baseline相比9x更快达到相同最佳值 [^41^]
Source: SkyPilot Blog
URL: https://blog.skypilot.co/scaling-autoresearch/
Date: 2026-03-18
Excerpt: "Over 8 hours, the agent submitted ~910 experiments (~700 with valid results)... val_bpb dropped from 1.003 to 0.974... The parallel agent reached the same best validation loss as the simulated sequential baseline in approximately 8 hours versus approximately 72 hours."
Context: 这是autoresearch scaling的最重要实证研究
Confidence: high

Claim: Agent在并行资源下自发开发出五种策略行为：(1) 从贪婪爬山转为factorial grid search；(2) 在单wave中测试6种model widths并发现width scaling最重要；(3) 独立发现H100与H200的性能差异，发展出screening on H100 + validation on H200的两层策略；(4) 五个自然phase（超参扫描→架构发现→细调→优化器调优→边际收益递减）；(5) 56% scaling efficiency（16 GPU上9x加速而非理想16x） [^42^]
Source: SkyPilot Blog / Evolutionary AI Survey
URL: https://evo.si5.pl/chapters/skypilot-autoresearch.html
Date: 2026
Excerpt: "The same Claude Code agent that performs greedy hill-climbing on one GPU spontaneously develops factorial experimental design, discovers and exploits hardware heterogeneity, and constructs a two-tier screening/validation workflow — none of which were programmed, prompted, or anticipated."
Context: 这是"基础设施规模解锁涌现认知能力"的最清晰实证
Confidence: high

Claim: Scaling效率损失来源：wave design time ~12%、result collection ~12%、cluster provisioning ~5%、inter-wave GPU idle ~10%、crashed experiments ~5%。主要瓶颈是agent的推理时间——16 GPU下agent成为serialization point [^43^]
Source: Evolutionary AI Survey
URL: https://evo.si5.pl/chapters/skypilot-autoresearch.html
Date: 2026
Excerpt: "The primary bottleneck is the agent's reasoning time. With 16 GPUs, the agent is the serialization point — it must design, submit, collect, and analyze each wave sequentially."
Context: 超过16 GPU的扩展需要multi-agent coordination或重叠wave execution
Confidence: high

### 5.2 成本分析

Claim: 整个SkyPilot 16-GPU 8小时运行的总成本约$309 GPU compute + ~$9 Claude API fees = ~$318。单次overnight运行约$25 + 单GPU [^44^]
Source: AIforAutomation Blog
URL: https://aiforautomation.io/news/2026-03-20-karpathy-autoresearch-700-experiments-8-hours-309-dollars
Date: 2026-03-20
Excerpt: "For $309 and 8 hours, an AI agent did what would take a human researcher weeks of manual experimentation."
Context: 经济性是autoresearch可及性的关键
Confidence: medium（GPU价格随市场波动）

---

## 6. 实验循环Agent的技术边界与失败模式

### 6.1 当前能做什么

Claim: autoresearch模式适用于任何具有：(a) 可测量scalar metric；(b) 可重复运行过程；(c) 单文件可修改空间的系统。已验证的应用包括：ML训练、GPU kernel优化、chess engine Elo提升、sudoku solver、API latency优化、trading bot、prompt optimization、build time优化、Bitcoin price prediction [^45^]
Source: autoresearch-cli README
URL: https://github.com/199-biotechnologies/autoresearch-cli
Date: 2026-03-23
Excerpt: "|Chess engine|Elo rating|Expert to Grandmaster (2718 Elo)| |API latency|p99 ms|37% reduction via KD-tree optimization| |Trading bots|Score|43,000% improvement via evolutionary loop|"
Context: 通用性已被社区广泛验证
Confidence: high

### 6.2 不能做什么 / 硬边界

Claim: autoresearch的核心约束包括：(1) 模型范围狭窄——仅GPT-style架构，不支持70B+分布式训练；(2) 同步顺序循环——无法并行探索多个分支，Karpathy的SETI@home愿景需要完全不同的协调层；(3) 指标范围有限——val_bpb在小数据集上可能不与下游任务性能相关，存在overfitting风险；(4) 需要预置可工作的训练代码——agent无法从零开始构建完整pipeline [^46^]
Source: awesomeagents.ai
URL: https://awesomeagents.ai/news/karpathy-autoresearch-single-gpu/
Date: 2026-03-10
Excerpt: "The model scope is narrow by design... The synchronous loop is the more fundamental constraint... Benchmark scope is limited too."
Context: 原始repo的明确边界
Confidence: high

Claim: Karpathy本人在README中明确指出"Codex doesn't seem to work"——因为agent不遵循"NEVER STOP"指令。可靠性高度依赖于所使用的coding agent [^47^]
Source: Verdent.ai FAQ
URL: https://www.verdent.ai/guides/what-is-autoresearch-karpathy
Date: 2026-04-07
Excerpt: "The README notes that 'Codex doesn't seem to work' due to the agent not following the 'NEVER STOP' instruction — reliability depends on the coding agent being used."
Context: agent依从性(autonomy compliance)是关键failure mode
Confidence: high

### 6.3 失败模式分类

Claim: 基于arXiv 2603.29632的研究，所有生成的proposal经历四个互斥状态：(1) Proposal Failure——patch不符合Search/Replace格式要求；(2) Preflight Failure——语法/危险代码等静态验证失败；(3) Training Crash——runtime crash（tensor shape mismatch、OOM、namespace collision）；(4) Training Success。单agent baseline在zero-shot复杂编辑中遭遇显著失败 [^48^]
Source: arXiv 2603.29632
URL: https://arxiv.org/html/2603.29632v1
Date: 2026-03-31
Excerpt: "The single-agent baseline, constrained by its linear, single-threaded execution bottleneck, is unable to generate logically sound, complex code without external feedback."
Context: 系统性的failure taxonomy
Confidence: high

Claim: LLM-based HPO方法（如LLAMBO）在观察完整trial历史的情况下，OOM率仍与random search相当（48-61% vs 56%），说明LLM"未能学习搜索空间中哪些区域导致内存失败" [^49^]
Source: arXiv 2603.24647 (v4)
URL: https://arxiv.org/html/2603.24647v4
Date: 2026-04-13
Excerpt: "LLM methods show OOM rates comparable to random search despite observing full trial history... suggesting they fail to learn which regions of the search space cause memory failures."
Context: 这是LLM agent在优化状态跟踪上的根本性局限
Confidence: high

Claim: 过拟合风险：Anthropic的AARS研究发现，在实验环境sandbox中表现优异的方法（0.97 PGR）在生产模型上改进不显著，因为优化行为exploited了实验环境的特定quirks而非发现通用技术。autoresearch面临类似天花板——"automation boosts whatever you're measuring. If the measure is wrong, you get fast wrong answers." [^50^]
Source: awesomeagents.ai
URL: https://awesomeagents.ai/news/anthropic-aars-beat-humans-alignment-fail/
Date: 2026-04-17
Excerpt: "This is a known failure mode in machine learning, but it carries special weight in the alignment context... Andrej Karpathy's autoresearch tool faces a similar ceiling: fast exploration of a well-defined search space, limited by the quality of the objectives being optimized."
Context: 指标设计的质量决定了automation的天花板
Confidence: high

### 6.4 小模型到大模型的transfer问题

Claim: arXiv 2603.28823的研究发现，在不同时间预算下存在U-shaped optimality曲线：过小模型因数据过拟合而 degrading（D8在4h训练后BPB 0.925比2h的0.906更差），过大模型因throughput不足而表现差。这意味着在5分钟budget下发现的最佳配置不一定transfer到更长的训练时间 [^51^]
Source: arXiv 2603.28823
URL: https://arxiv.org/html/2603.28823v1
Date: 2026-03-29
Excerpt: "D8 at 4h achieves BPB 0.925, worse than its 2h result of 0.906; by 12h, its BPB degrades further to 0.919 after cycling through the 48M-token dataset over 250 times."
Context: 对autoresearch默认5分钟实验的transfer validity提出质疑
Confidence: high

Claim: MLX port的意外发现：较小硬件上的获胜策略与H100运行不同——在H100上work的优化在M4 Max上可能degrade，这对从consumer hardware向production GPU transfer结果的人具有重要implication [^52^]
Source: Particula.tech Blog
URL: https://particula.tech/blog/karpathy-autoresearch-autonomous-ml-experiments
Date: 2026-03-18
Excerpt: "The MLX port revealed something unexpected: smaller hardware favors different winning strategies than H100 runs."
Context: 硬件差异导致优化策略不同
Confidence: medium

### 6.5 生产就绪性gap

Claim: autoresearch明确是research tool而非production platform：无认证、program.md之外无错误恢复、结果platform-specific、无dashboard/resumability/multi-GPU支持（这些均为第三方fork添加）。对于大规模生产ML workflow，autoresearch是"pattern to adapt, not a system to deploy directly" [^53^]
Source: Verdent.ai
URL: https://www.verdent.ai/guides/what-is-autoresearch-karpathy
Date: 2026-04-07
Excerpt: "AutoResearch is a research tool and explicitly described as such. It has no authentication, no error recovery beyond what `program.md` specifies, and results that are platform-specific."
Context: 生产部署与实验原型的清晰分界
Confidence: high

---

## 7. 反面论点与争议

### 7.1 "LLM不能替代经典优化器"

Claim: Ferreira等人的论文核心结论是"LLMs are most effective as a complement to classical optimizers, not as a replacement"——即使在无约束代码编辑设置下，LLM agent也未超越经典方法，且frontier model scaling未能关闭这一gap [^54^]
Source: arXiv 2603.24647 (v4)
URL: https://arxiv.org/html/2603.24647v4
Date: 2026-04-13
Excerpt: "All in all, our results suggest that LLMs are most effective as a complement to classical optimizers, not as a replacement."
Context: 这是对autoresearch hype的重要学术纠偏
Confidence: high

### 7.2 "贪婪爬山的结构性局限"

Claim: RoboPhD研究表明，在复杂任务上（ARC-AGI、Text2SQL、DocFinQA），贪婪爬山(autoresearch)被Elo-based tournament selection和Pareto-based selection显著超越，仅在简单任务上（Can't Be Late，<90行代码）获胜 [^55^]
Source: arXiv 2604.04347
URL: https://arxiv.org/pdf/2604.04347
Date: 2026-04-06
Excerpt: "Using a single default configuration, RoboPhD outperforms both GEPA and Autoresearch on three of four benchmarks, losing only on the simplest task."
Context: 简单贪婪爬山在复杂搜索空间中容易陷入局部最优
Confidence: high

### 7.3 "Agent可靠性幻觉"

Claim: METR 2026年3月10日研究发现，在296个AI-generated PR中，maintainer merge decisions比automated benchmark scores低约24个百分点——约一半的test-passing PR不会被merge。这说明benchmark通过不等于实际可用，autoresearch的"自动改进"可能产生技术上通过但工程上不可接受的代码 [^56^]
Source: Russell Clare State of AI
URL: https://russellclare.com/signals/ai-state/
Date: 2026-03-27
Excerpt: "METR... found maintainer merge decisions run approximately 24 percentage points below automated benchmark scores – and the gap is widening... Roughly half of test-passing PRs would not be merged."
Context: autoresearch的自动keep/revert可能忽略代码质量、可维护性等非metric维度
Confidence: high

---

## 8. 关键洞察汇总

| 维度 | 核心发现 | 置信度 |
|------|---------|--------|
| 架构 | 三文件(630行Python)极简设计，program.md作为自然语言编排层 | high |
| 循环 | 贪婪hill-climbing ratchet loop，固定5分钟时间预算 | high |
| 基准 | 经典HPO(TPE/CMA-ES)在固定搜索空间内始终优于纯LLM agent | high |
| 混合 | Centaur(CMA-ES+LLM)取得最佳结果，0.8B LLM足够 | high |
| Scaling | 多GPU并行(16x)不仅加速9x，更触发agent策略涌现（factorial DOE、硬件筛选） | high |
| 衍生 | 7天内6+变体，覆盖通用metric、MLX、Windows、kernel优化、分布式协作 | high |
| 编码Agent融合 | Claude Code/Codex/Cursor均有结构化skill/extension集成 | high |
| 边界 | 仅适用单文件+scalar metric场景；顺序执行无法并行探索；指标质量决定天花板 | high |
| 失败 | OOM率与random search相当说明LLM状态跟踪弱；过拟合到实验环境是真实风险 | high |
| 反面 | LLM不应替代经典优化器；贪婪爬山在复杂任务上劣于进化策略；benchmark≠实际可用 | high |

---

## 9. 来源索引

| 编号 | 来源 | URL | 日期 |
|------|------|-----|------|
| [^1^] | DataCamp Guide | https://www.datacamp.com/tutorial/guide-to-autoresearch | 2026-03-23 |
| [^2^] | AI Automation Global | https://aiautomationglobal.com/blog/karpathy-autoresearch-autonomous-ai-agent-2026 | 2026-03-22 |
| [^3^] | PM's Guide (Aakash Gupta) | https://www.news.aakashg.com/p/autoresearch-guide-for-pms | 2026-03-19 |
| [^4^] | OpenRepoGuide | https://openrepoguide.com/ | 2026 |
| [^5^] | Webvise Blog | https://webvise.io/blog/karpathy-autoresearch-ai-experiments | 2026-04-02 |
| [^6^] | Webvise Blog | https://webvise.io/blog/karpathy-autoresearch-ai-experiments | 2026-04-02 |
| [^7^] | DataCamp Guide | https://www.datacamp.com/tutorial/guide-to-autoresearch | 2026-03-23 |
| [^8^] | DataCamp Guide | https://www.datacamp.com/tutorial/guide-to-autoresearch | 2026-03-23 |
| [^9^] | DataCamp Guide | https://www.datacamp.com/tutorial/guide-to-autoresearch | 2026-03-23 |
| [^10^] | DataCamp Guide | https://www.datacamp.com/tutorial/guide-to-autoresearch | 2026-03-23 |
| [^11^] | Ry Walker Research | https://rywalker.com/research/autoresearch | 2026-03-13 |
| [^12^] | Paddo.dev | https://paddo.dev/blog/autoresearch-overnight-lab/ | 2026-03-14 |
| [^13^] | arXiv 2603.24647 | https://arxiv.org/abs/2603.24647 | 2026-03-25 |
| [^14^] | arXiv 2603.24647 v4 | https://arxiv.org/html/2603.24647v4 | 2026-04-13 |
| [^15^] | arXiv 2603.24647 v4 | https://arxiv.org/html/2603.24647v4 | 2026-04-13 |
| [^16^] | arXiv 2603.24647 v4 | https://arxiv.org/html/2603.24647v4 | 2026-04-13 |
| [^17^] | arXiv 2603.24647 v4 | https://arxiv.org/html/2603.24647v4 | 2026-04-13 |
| [^18^] | arXiv 2603.24647 v4 | https://arxiv.org/html/2603.24647v4 | 2026-04-13 |
| [^19^] | arXiv 2603.24647 v4 Appendix | https://arxiv.org/html/2603.24647v4 | 2026-04-13 |
| [^20^] | arXiv 2604.04347 | https://arxiv.org/abs/2604.04347 | 2026-04-06 |
| [^21^] | arXiv 2604.04347 | https://arxiv.org/pdf/2604.04347 | 2026-04-06 |
| [^22^] | arXiv 2604.04347 | https://arxiv.org/pdf/2604.04347 | 2026-04-06 |
| [^23^] | arXiv 2603.29632 | https://arxiv.org/html/2603.29632v1 | 2026-03-31 |
| [^24^] | awesome-autoresearch | https://github.com/WecoAI/awesome-autoresearch | 2026-03-22 |
| [^25^] | Ry Walker Tools Compared | https://rywalker.com/research/autoresearch-tools | 2026-04-14 |
| [^26^] | awesome-autoresearch | https://github.com/WecoAI/awesome-autoresearch | 2026-03-22 |
| [^27^] | Particula.tech | https://particula.tech/blog/karpathy-autoresearch-autonomous-ml-experiments | 2026-03-18 |
| [^28^] | arXiv 2603.21331 | https://arxiv.org/html/2603.21331v1 | 2026-03-22 |
| [^29^] | Ensue Blog | https://ensue.dev/blog/autoresearch-at-home/ | 2026-03-12 |
| [^30^] | N-Autoresearch | https://github.com/iii-hq/n-autoresearch | 2026-04-19 |
| [^31^] | karpathy/agenthub | https://github.com/karpathy/agenthub | 2026-03-09 |
| [^32^] | GitHub Issue #249 | https://github.com/karpathy/autoresearch/issues/249 | 2026-03-13 |
| [^33^] | SimpleNews.ai | https://www.simplenews.ai/news/karpathys-autoresearch-pattern-spawns-generalized-claude-code-skill-for-autonomous-iteration-srsk | 2026-03-16 |
| [^34^] | MindStudio | https://www.mindstudio.ai/blog/claude-code-autoresearch-self-improving-skills/ | 2026-03-14 |
| [^35^] | MindStudio | https://www.mindstudio.ai/blog/claude-code-autoresearch-self-improving-skills/ | 2026-03-14 |
| [^36^] | LobeHub | https://lobehub.com/de/skills/aradotso-trending-skills-codex-autoresearch-loop | 2026-04-06 |
| [^37^] | VS Code Marketplace | https://marketplace.visualstudio.com/items?itemName=ergenekonyigit.autoresearch | 2026-04-16 |
| [^38^] | autoresearch-cli | https://github.com/199-biotechnologies/autoresearch-cli | 2026-03-23 |
| [^39^] | evo-hq/evo | https://github.com/evo-hq/evo | 2026-04-05 |
| [^40^] | ouro-loop | https://github.com/VictorVVedtion/ouro-loop | 2026-03-15 |
| [^41^] | SkyPilot Blog | https://blog.skypilot.co/scaling-autoresearch/ | 2026-03-18 |
| [^42^] | Evolutionary AI Survey | https://evo.si5.pl/chapters/skypilot-autoresearch.html | 2026 |
| [^43^] | Evolutionary AI Survey | https://evo.si5.pl/chapters/skypilot-autoresearch.html | 2026 |
| [^44^] | AIforAutomation | https://aiforautomation.io/news/2026-03-20-karpathy-autoresearch-700-experiments-8-hours-309-dollars | 2026-03-20 |
| [^45^] | autoresearch-cli | https://github.com/199-biotechnologies/autoresearch-cli | 2026-03-23 |
| [^46^] | awesomeagents.ai | https://awesomeagents.ai/news/karpathy-autoresearch-single-gpu/ | 2026-03-10 |
| [^47^] | Verdent.ai | https://www.verdent.ai/guides/what-is-autoresearch-karpathy | 2026-04-07 |
| [^48^] | arXiv 2603.29632 | https://arxiv.org/html/2603.29632v1 | 2026-03-31 |
| [^49^] | arXiv 2603.24647 v4 | https://arxiv.org/html/2603.24647v4 | 2026-04-13 |
| [^50^] | awesomeagents.ai | https://awesomeagents.ai/news/anthropic-aars-beat-humans-alignment-fail/ | 2026-04-17 |
| [^51^] | arXiv 2603.28823 | https://arxiv.org/html/2603.28823v1 | 2026-03-29 |
| [^52^] | Particula.tech | https://particula.tech/blog/karpathy-autoresearch-autonomous-ml-experiments | 2026-03-18 |
| [^53^] | Verdent.ai | https://www.verdent.ai/guides/what-is-autoresearch-karpathy | 2026-04-07 |
| [^54^] | arXiv 2603.24647 v4 | https://arxiv.org/html/2603.24647v4 | 2026-04-13 |
| [^55^] | arXiv 2604.04347 | https://arxiv.org/pdf/2604.04347 | 2026-04-06 |
| [^56^] | Russell Clare State of AI | https://russellclare.com/signals/ai-state/ | 2026-03-27 |

---

*研究完成日期: 2026年*
*搜索次数: 25+次独立查询*
*覆盖来源: arXiv论文(5篇)、GitHub仓库(15+)、权威博客/媒体(12+)*
