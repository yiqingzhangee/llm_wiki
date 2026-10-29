# 面向自我改进的脚手架工程（中英双语）

- 原文标题：Harness Engineering for Self-Improvement
- 作者：Lilian Weng
- 日期：2026 年 7 月 4 日
- 原文链接：<https://lilianweng.github.io/posts/2026-07-04-harness/>

## 中文译文

### 面向自我改进的脚手架（harness）工程

日期：2026 年 7 月 4 日 | 预计阅读时间：31 分钟 | 作者：Lilian Weng

目录

  * 脚手架设计模式
    * 模式 1：工作流自动化
    * 模式 2：以文件系统作为持久记忆
    * 模式 3：子智能体与后端任务
    * 案例研究：编码智能体脚手架
    * 脚手架层与核心智能？
  * 脚手架优化
    * 上下文工程
    * 工作流设计
    * 自我改进型脚手架
    * 演化搜索
    * 与模型权重联合优化
  * 未来挑战
  * 引用
  * 附录：一些实用的基准测试
  * 参考文献



**递归式自我改进（RSI）**的概念可以追溯到 [I. J. Good（1965）](https://philpapers.org/rec/GOOSCT)。他将“超级智能机器”定义为一种能在所有智力活动中超越人类，并能设计出更好的机器来改进自身的系统。[Yudkowsky（2008）](https://www.lesswrong.com/posts/JBadX7rwdcRFzGuju/recursive-self-improvement) 用“递归式自我改进”来指代一种特定的反馈循环：AI 利用当前智能，改进产生其智能的认知机制。

在现代 AI 中，这一反馈循环可能意味着模型直接重写自身权重；从更广义上说，也可能是模型改进_训练流水线_和_部署系统_，进而造就性能更强的后继模型，使其在具有经济价值的各类任务上表现更好。已有迹象表明，前沿实验室的 AI 研发速度大幅加快（[Anthropic](https://www.anthropic.com/institute/recursive-self-improvement)；[OpenAI](https://openai.com/index/how-agents-are-transforming-work/)）。

我之所以明确提到_“部署系统”_，是因为位于原始模型与现实环境之间的这一层，似乎与模型的原始智能（即预训练刚结束时的评测结果）同样重要。Claude Code 和 Codex 等成功的编码智能体产品已经表明，脚手架是 AI 部署中的重要组成部分。脚手架（harness）是围绕基座模型搭建的系统，负责协调执行，并决定模型如何思考与规划、调用工具与行动、感知与管理上下文、存储产物以及评估结果。

本文将聚焦脚手架工程相关研究，以及它如何推动递归式自我改进。近期许多关于自动化研究、自我改进智能体和演化式程序搜索的工作，都可以围绕这一问题来梳理。模型自博弈、合成数据、测试时训练，以及更广泛的持续学习研究，也符合递归式自我改进的愿景（例如 [Yuan et al. 2024](https://arxiv.org/abs/2401.10020)、[Chen et al. 2024](https://arxiv.org/abs/2401.01335)、[Zhao et al. 2025](https://arxiv.org/abs/2505.03335)、[Choi et al. 2026](https://openreview.net/forum?id=lTbBFAoPSA)），但它们不是本文的重点。

### 脚手架设计模式

与[早期智能体框架](https://lilianweng.github.io/posts/2023-06-23-agent/)中“智能体 = LLM + 记忆 + 工具 + 规划 + 行动”的定义相比，脚手架工程还包括_工作流设计（例如循环工程）、评估、权限控制和持久状态管理_。它已不再只是提示词模板，而更接近运行时和软件系统设计：模型如何观察、行动、记忆、自检与改进。

设计应有意保持简洁和通用，以便实现泛化；同时可以借鉴现有软件工程实践，利用预训练知识。操作系统与脚手架之间也有很强的类比关系。脚手架应像操作系统一样，在保持接口简洁的同时封装复杂逻辑。与此同时，配置、工具接口及其他协议可能会逐步形成行业标准。

#### 模式 1：工作流自动化

定义一套让模型能够运行、测试和迭代的工作流，是实现自动化的一项关键设计。Karpathy 的 autoresearch 仓库（<https://github.com/karpathy/autoresearch>）清楚展示了如何构建这样的工作流。常见工作流会围绕目标循环执行：规划、执行、观察/测试、改进，然后再次执行，_直到_达成目标。在此过程中，系统可能会主动请求用户澄清任务要求或执行偏好。

![](https://lilianweng.github.io/posts/2026-07-04-harness/openai-agent-loop.png) 简化的 Codex 智能体循环：智能体调用工具，工具响应会影响模型的下一次生成。  
（图片来源：[OpenAI Codex 智能体文章](https://openai.com/index/unrolling-the-codex-agent-loop/)）

该工作流图还强调，模型会分析自身轨迹和失败案例，再通过“智能体运行时”迭代推进任务，而不是依赖静态提示词模板。

#### 模式 2：以文件系统作为持久记忆

在长时程智能体系统中，一个反复出现的模式是：以简单的控制方式驾驭丰富的状态和产物。脚手架不应把整个工作流和全部日志都塞进上下文，而应将持久状态保存在文件中。在长时程的智能体式推演中，实验日志、代码差异、论文摘要、错误跟踪记录和过往推演轨迹等产物，其长度往往远超模型训练时所面向的上下文窗口。

学习如何读取、写入和编辑文件系统（通常通过 `bash` 命令）是 LLM 的基础能力。因此，以文件这种简单形式管理持久记忆，自然会受益于模型核心能力的提升。

#### 模式 3：子智能体与后端任务

脚手架可以启动多个子智能体并行执行，并监控后端任务。当主智能体需要探索多个假设、并发运行实验，或委派相互隔离的子任务而不干扰主上下文时，这种设计很有用。此时，父智能体需要一个轻量的进程管理器，用于启动任务、检查日志、取消失败的运行，并将结果合并回主智能体线程。

设计的关键在于让并行过程明确且可检查。如果子智能体的输出只存在于临时对话上下文中，很快就会过时并被掩埋。若将输出保存为文件、日志和状态记录，模型便能在中断后恢复，并根据自身执行历史进行推理。

#### 案例研究：编码智能体脚手架

Claude Code、Codex、OpenCode 和 Cursor 风格智能体等主流编码智能体的核心接口已经趋于稳定。它们通常采用如下循环：

![](https://lilianweng.github.io/posts/2026-07-04-harness/coding-harness-loop.png)

编码智能体可以借助一组工具，在给定代码仓库中进行开发和问题调试，就像人类开发者使用 IDE 一样。

（以下并非完整列表，仅用于演示。感兴趣可阅读[此文](https://github.com/yasasbanukaofficial/claude-code)。）

类别 | 工具定义  
---|---  
文件系统 | \- 文件发现：`glob`、`grep`、`ls`  
\- 文件读取：`read`、`read_many`  
\- 文件修改：`write`（写入一个完整的新文件）；`edit`（精确匹配字符串并替换）；`multi_edit`；`apply_patch`（应用结构化补丁/差异）  
Shell 执行 | 运行命令：`bash`、`PowerShell`  
I/O | `lsp`，以及 `git_status`、`git_diff`、`git_commit` 等 git 工具  
外部上下文 | MCP 工具、Skills  
网页搜索 | `web_search`、`web_fetch`、浏览器工具  
产物 | 读取文档、图片；生成 HTML、图片  
后端进程 | 例如：`CronCreate`、`CronDelete`、`CronList`  
智能体委派 | 例如：`spawn_agent`、`resume_agent`、`wait_agent`、`list_agents`、`close_agent`、`interrupt_agent` 等  
  
#### 脚手架层与核心智能？

未来的递归式自我改进将在多大程度上依赖脚手架工程，很难预测；但在近期，它不太可能从模型直接重写自身权重开始。我认为，近期可行的路径是：

  1. 脚手架工程将朝元方法论方向发展，也就是改进获得更好答案的机制，而不只是改进答案本身。脚手架系统本身会成为优化目标，其中启发式规则更少，通用机制更多。
  2. 成熟的脚手架将支持模型自我改进循环的自动化研究；反过来，更智能的模型可以避免脚手架过度工程化，让系统保持可持续。



最终，许多脚手架改进可能会被_内化_为模型的核心行为，但与外部上下文及工具交互的接口仍会保留。我们已经在[提示词工程](https://lilianweng.github.io/posts/2023-03-15-prompt-engineering/)中看到过较温和的类似现象：随着指令微调和模型推理能力提升，手工提示技巧不再那么重要，但_指定目标、约束、上下文和评估的需求并未消失_。

### 脚手架优化

脚手架系统中优化对象的演进大致如下：指令[提示词](https://lilianweng.github.io/posts/2023-03-15-prompt-engineering/) → 结构化上下文 → 工作流 → 脚手架代码 → 优化器代码。随着模型变得更智能、更强大，我们会转向更复杂的优化目标和更通用的方法。

#### 上下文工程

随着智能体式任务的时程显著增长，如果只是把所有工具响应和模型生成结果不断追加到上下文中，其规模很快就会失控。上下文管理层负责为 LLM 构建结构更清晰、更简洁的上下文，并管理持久状态。长上下文研究无疑会持续进步，但现阶段，长上下文智能与上下文工程有时会交织在一起。

**智能体式上下文工程**（ACE；[Zhang et al. 2025](https://arxiv.org/abs/2510.04618)）将上下文视为一份不断演进的行动手册，而不是越来越长的提示词。它包含三个组件，用于维护一份由项目符号条目组成的上下文行动手册，每个条目都有标识符和描述。

  1. _生成器（Generator）_：参照这些条目生成任务轨迹。
  2. _反思器（Reflector）_：从成功和失败的轨迹中提炼经验。
  3. _编纂器（Curator）_：通过增量的条目化记录更新结构化上下文。

![](https://lilianweng.github.io/posts/2026-07-04-harness/ace.png) 智能体式上下文工程（ACE）框架。（图片来源：[Zhang et al. 2025](https://arxiv.org/abs/2510.04618)）

为避免迭代重写过程中的上下文坍塌和简洁偏置，ACE 的一项关键设计是让编纂器不去重写整块提示词，而是输出一组形如（标识符，描述）的结构化条目，再通过确定性逻辑将这些条目合并到结构化上下文日志中。系统还会定期精炼上下文条目并去重。

ACE 能从推演轨迹中学习经验，这有助于我们迈向自主管理的记忆；但它的更新规则和整体工作流仍由人工设计。为了构建更具自我改进能力的循环，**元上下文工程**（MCE；[Ye et al. 2026](https://arxiv.org/abs/2601.21557)）将机制（如何管理上下文）与产物内容（上下文中包含什么）分开，在元优化层执行技能演化，在基础层执行上下文优化。

MCE 技能 $s \in \mathcal{S}$ 定义上下文函数 $c_s=(\rho_s,F_s)$，并将输入 $x$ 映射为上下文 $c = F_s(x;\rho_s)$，其中：

  * $\rho_s = \\{\rho_1,\dots,\rho_m\\}$ 是静态组件（提示词、知识库、代码库）。
  * $F_s = \\{F_1,\dots,F_k\\}$ 是动态算子（搜索、选择、过滤、格式化）。



双层优化的内层目标，是在训练数据上、给定技能 $s$ 的条件下找到最佳上下文 $c_s^*$；外层循环则寻找能在验证集上取得最佳性能的最优技能：

$$ \text{Inner: }c_s^*=\arg\max_{c_s}J_\text{train}(c_s;s)\quad \text{Outer: }s^*=\arg\max_{s\in\mathcal{S}}J_\text{val}(c_s^*) $$ 

技能数据库记录了以往技能、上下文函数和评测指标的历史：$\mathcal{H}_{k-1} = \\{(s_i,c_i,J_i^\text{train}, J_i^\text{val})\\}_{i=1}^{k-1}$。给定任务 $\tau$，元层智能体对既有技能执行智能体式[交叉](https://en.wikipedia.org/wiki/Crossover_\(evolutionary_algorithm\))，以创建新技能：$s_k=\text{crossover}(\tau,\mathcal{H}_{k-1})$。

随后，基础层上下文工程智能体执行技能 $s_k$，在当前技能的指导下，根据推演轨迹反馈 $\mathcal{R}_k$ 学习上下文函数：$c_k=\text{engineer}(\tau,s_k;c_{k-1}^*,\mathcal{R}_k)$。

![](https://lilianweng.github.io/posts/2026-07-04-harness/mce.png) 元上下文工程（MCE）框架：元层的技能演化搜索上下文管理机制，基础层则优化任务上下文。（图片来源：[Ye et al. 2026](https://arxiv.org/abs/2601.21557)）

MCE 不像 ACE 那样强制规定上下文结构的启发式规则。它使用_自由形式的技能_存储对任务最重要的知识，并以迭代方式共同演化技能及以该技能为条件的上下文。在实现上，上下文函数 $c$ 被实例化为专用目录中的一组文件，其中既包含静态组件（`skill.md`），也包含动态组件（上下文与数据推演轨迹）。元层和基础层的优化都在配备标准工具集的智能体式编码环境中执行：

$$ \mathcal{T}=\\{\texttt{Read},\texttt{Write},\texttt{Edit},\texttt{Bash},\texttt{Glob},\texttt{Grep},\texttt{TodoWrite}\\} $$ 

**元脚手架（Meta-Harness）**（[Lee et al. 2026](https://arxiv.org/abs/2603.28052)）又深入了一层：优化对象变成了_代码_，这些代码决定并优化哪些信息应当存储、检索并呈现给模型。名称中的“Meta-”表示它是一套用于优化脚手架的脚手架。

![](https://lilianweng.github.io/posts/2026-07-04-harness/meta-harness-outer-loop.png) 元脚手架的外层循环优化算法。（图片来源：[Lee et al. 2026](https://arxiv.org/abs/2603.28052)）

用于创建新脚手架的提议器本身就是一个编码智能体，最终输出则是帕累托前沿上的一组脚手架候选方案。

  * 完整执行历史可通过文件系统访问，因此编码智能体会使用 `grep` 或 `cat` 等命令读取历史，而不是把所有内容一股脑塞进单个提示词上下文。
  * 提议的脚手架是文件系统中的一个目录，其中包含自身源代码、评分、推演轨迹和状态更新。
  * 元脚手架循环会迭代创建新脚手架，只保留符合要求的候选方案。

![](https://lilianweng.github.io/posts/2026-07-04-harness/meta-harness.png) 元脚手架在（左）少量迭代的文本分类任务和（右）TerminalBench-2 上的表现。需要注意的是，TerminalBench-2 实验中的搜索以 Terminus-KIRA 和 Terminus-2 作为初始点，这两套脚手架的能力都很强。（图片来源：[Lee et al. 2026](https://arxiv.org/abs/2603.28052)）

不过，重要结论很明确：一旦脚手架设计成为可执行的搜索空间，强大的编码智能体就能利用人类工程师所使用的同一设计空间。

#### 工作流设计

在脚手架工程中，工作流可以由领域专家手工设计。以自动化研究为例，研究者已提出并测试了多种框架。**AI Scientist** 系统（[Lu et al. 2026](https://www.nature.com/articles/s41586-026-10265-5)）构建了一条流水线，用于提出研究构想、编写代码、运行实验、分析结果、撰写论文和开展同行评审。[Meng et al. (2026)](https://arxiv.org/abs/2605.26340) 在 **ScientistOne** 中将可验证性作为核心设计约束：每一项声明（引文、数值、方法或结论）都必须能够追溯到证据来源，并通过证据链检查接受审核。

![](https://lilianweng.github.io/posts/2026-07-04-harness/ai-scientist.png) AI Scientist 用于构思生成、实验、论文写作和评审的流水线。（图片来源：[Lu et al. 2026](https://www.nature.com/articles/s41586-026-10265-5)）

**Autodata** 智能体（[Kulikov et al. 2026](https://arxiv.org/abs/2606.25996)）旨在充当数据科学家，生成训练和评估数据。主智能体负责管理一个提出问题的 _挑战者_、一个 _弱求解器_、一个 _强求解器_ 和一个 _验证器/裁判_，目标是合成难度“恰到好处”的数据，即强求解器能成功，而弱求解器会失败。

在 Autodata 中，挑战者的提示词会根据求解器和验证器的反馈迭代更新。其局限在于，合成任务只用于微调弱求解器，而不用于微调强求解器；如果这个循环不能迭代改进强模型，那么它更像是在生成的提示词分布上进行间接蒸馏，递归式自我改进的意味较弱。

![](https://lilianweng.github.io/posts/2026-07-04-harness/autodata.png) Autodata 的智能体式工作流设计，围绕挑战者、求解器和验证器等角色生成合成训练与评估数据。（图片来源：[Kulikov et al. 2026](https://arxiv.org/abs/2606.25996)）

工作流的设计空间 _极其庞大_。我们自然可以把工作流设计视为一个搜索问题，从而通过算法寻找良好解法，而不是只能手工设计。沿着这一思路，**智能体系统自动化设计**（Automated Design of Agentic Systems，ADAS；[Hu et al. 2025](https://arxiv.org/abs/2408.08435)）将智能体设计本身表述为一个优化问题，即“元智能体搜索”：由元智能体提出新的智能体式工作流设计。

  1. 使用 CoT、自我精炼等简单智能体，初始化一个智能体式工作流档案库。
  2. 要求元智能体参考档案库中的现有方案，以 _代码_ 形式编写新智能体。
     * 元智能体先生成新工作流的高层描述，再用代码实现它。
     * 随后，元智能体对程序草稿执行两步自我精炼（即先要求模型提供反馈，再要求同一个模型根据反馈精炼此前生成的输出；[Madaan et al. 2023](https://arxiv.org/abs/2303.17651)），以检查其新颖性。
  3. 评估每个新候选方案，并将成功的方案加入档案库。
  4. 重复第 2～3 步，直至达到最大迭代次数。

![](https://lilianweng.github.io/posts/2026-07-04-harness/adas.png) 智能体系统自动化设计（ADAS）示意图。  
（图片来源：[Hu et al. 2025](https://arxiv.org/abs/2408.08435)）

**AFlow**（[Zhang et al. 2025](https://arxiv.org/abs/2410.10762)）将智能体式工作流表示为图，其中节点表示调用 LLM 的动作，边则以代码实现逻辑操作。工作流优化依赖 [MCTS](https://en.wikipedia.org/wiki/Monte_Carlo_tree_search)（蒙特卡洛树搜索）：

  1. 使用模板，在树中初始化起始工作流 $W_0$。
  2. 通过分数与均匀探索的软混合策略选择一个工作流节点。
  3. 要求 LLM 根据该节点的评估表现生成修改后的工作流，以扩展该节点。
  4. 执行并评估新工作流。
  5. 如果新工作流在 $N$ 轮预算内有所改进，则将其加入树中。
  6. 重复第 2～5 步；当排名前 $k$ 的平均分趋于平稳或预算耗尽时停止。

![](https://lilianweng.github.io/posts/2026-07-04-harness/aflow.png) AFlow 在工作流候选方案树上的优化过程。（图片来源：[Zhang et al. 2025](https://arxiv.org/abs/2410.10762)）

AFlow 在问答、代码和数学任务上的实验表明，相较于人工设计的工作流和 ADAS，AFlow 取得了不错的改进。

![](https://lilianweng.github.io/posts/2026-07-04-harness/aflow-exp.png) AFlow 与人工方法及 ADAS 的实验对比。（图片来源：[Zhang et al. 2025](https://arxiv.org/abs/2410.10762)）

#### 自我改进型脚手架

上下文工程或工作流设计都只是脚手架的一部分。我们需要搜索整个设计空间，联合优化上下文管理逻辑、工作流、权限以及脚手架的许多其他组件。正如 Meta-Harness、ADAS 和 AFlow 等工作所示，**✨代码✨** 是定义程序和系统的 **通用语言**。简单来说，脚手架就是一段代码，它规定了提示词、工具调用、子智能体、控制流、记忆和工作流逻辑如何协同运作。如果 LLM 能优化执行智能体的代码，它所能触及的设计空间就会比手写提示词 _大得多_。

**Self-Taught Optimizer**（STOP；[Zelikman et al. 2023](https://arxiv.org/abs/2310.02304)）是递归式脚手架改进的早期范例之一。在 $t=0$ 时，种子改进器 $I_0$ 接收初始解 $s$、效用函数 $u$ 和黑盒语言模型 $M$，返回改进后的解 $s'$，即 $s' = I(u, s; M)$。STOP 的目标不是直接改进 $s$，而是 _改进器 $I$ 本身_。

首先，将元效用定义为给定改进器函数 $I$ 在下游任务集合 $\mathcal{D}$ 上的平均效用：

$$ \hat{u}(I) \triangleq \frac{1}{\vert\mathcal{D}\vert}\mathbb{E}_{(u,s)\sim \mathcal{D}}[u(I(u,s; M))] $$ 

由于改进改进器函数本身也是一个优化问题，我们可以通过自我改进更新，根据元效用衡量的 $I_{t-1}$ 表现，递归地得到新版本 $I_t$：

$$ I_t=I_{t-1}(\hat{u},I_{t-1};M) $$ 

![](https://lilianweng.github.io/posts/2026-07-04-harness/STOP-algo.png) Self-Taught Optimizer（STOP）算法。（图片来源：[Zelikman et al. 2023](https://arxiv.org/abs/2310.02304)）

在实验中，改进后的改进器发现了多种策略，包括遗传算法、分解并改进各个部分、多臂老虎机式提示词选择、模拟退火、改变温度，以及束搜索/树搜索。这类似于将脚手架工作流表示为可供优化的对象。

![](https://lilianweng.github.io/posts/2026-07-04-harness/STOP-patterns.png) STOP 发现的自我改进策略示例。（图片来源：[Zelikman et al. 2023](https://arxiv.org/abs/2310.02304)）

Zelikman et al. (2023) 的研究有一项 _值得警惕_ 的结果：使用 GPT-4 时，STOP 在多次迭代中提高了下游任务的平均表现；使用 GPT-3.5 和 Mixtral 等较弱模型时，表现却有所下降。仅有递归结构还不够。基础模型必须 _足够强_，才能改进这一机制。这说明，改进脚手架可以让模型得到更好的部署，但智能仍是核心。

[Lin et al. (2026)](https://arxiv.org/abs/2605.30621) 更详细地研究了脚手架演化对模型能力的依赖。他们区分了两个维度：(1) _脚手架更新_，指生成有效脚手架修改的能力；(2) _脚手架受益_，指利用更新后的脚手架提升任务解决效果的能力。有趣的是，实验发现，从 Qwen3.5-9B 到 Claude Opus 4.6，一系列规模和核心智能水平不同的模型表现出相近的脚手架更新能力；9B 的脚手架提议器/演化器能够编写出在流程上与 Opus 所写技能同构的技能。要充分利用脚手架，模型需要正确、及时地调用技能/工具，还要善于遵循长时程指令。

![](https://lilianweng.github.io/posts/2026-07-04-harness/harness-update.png) 主要结果：(A) 从 Qwen2-32B 到 Opus 4.6，各类模型的脚手架更新能力测量结果大致持平；(B) 脚手架受益能力呈非单调变化，中档模型获益最多。（图片来源：[Lin et al. 2026](https://arxiv.org/abs/2605.30621)）

较新的工作 **Self-Harness**（[Zhang et al. 2026](https://arxiv.org/abs/2606.09498)）依靠 LLM 智能体，通过“提议—评估—接受”循环来改进自身脚手架。

![](https://lilianweng.github.io/posts/2026-07-04-harness/self-harness.png) Self-Harness 通过弱点挖掘、受限脚手架提议和验证的循环来更新脚手架。（图片来源：[Zhang et al. 2026](https://arxiv.org/abs/2606.09498)）

Self-Harness 的循环分为三个阶段：

  1. _弱点挖掘_：将失败案例聚类为以验证器结果为依据的失败模式。
     * 使用当前脚手架 $h_t$ 评估任务，并收集执行轨迹以供分析。
     * 请注意，两次运行在错误日志中可能表面上具有相同的验证器结果，例如超时或缺少产物，但背后的因果机制并不相同。因此，我们需要信息丰富的失败记录，其中包含验证器层面的最终原因、相关智能体行为的因果状态，以及轨迹所揭示的抽象智能体机制，进而找出根本原因。
  2. _脚手架提议_：根据挖掘出的失败模式，提出范围受限的脚手架修改。
     * 在 $h_t$ 下调用同一模型作为提议器。
     * 向模型提供范围受限的提议上下文：(1) 当前脚手架的可编辑部分；(2) 来自评估系统、以验证器结果为依据的失败模式；(3) 应保留的成功行为记录；(4) 此前尝试过的修改摘要。
     * 脚手架修改应优先处理反复出现、可以解决（例如并非任务本身的特定难点）且能通过小范围改动消除的错误模式。
     * 脚手架修改候选方案应彼此有别并保持多样性。
  3. _提议验证_：验证并合并符合条件的修改，创建新脚手架 $h_{t+1}$。
     * 在留存划分 $D_\text{in}$（用于测试弱点是否已解决）和留出划分 $D_\text{out}$（用于检查是否引入了其他未知问题）上运行回归测试，以评估候选修改。
     * 只有在留存数据与留出数据上均未出现回退时，候选方案才会被接受。
     * 合并已接受的候选方案，将脚手架更新为 $h_{t+1}$；被拒绝的候选方案则会写入日志，但不改变当前生效的脚手架。



在 Terminal-Bench-2 上运行 `MiniMax M2.5`、`Qwen3.5-35B-A3B` 和 `GLM-5` 时，Self-Harness 能够针对不同基础模型的不同弱点，学习模型特定的脚手架指令，并提高留出测试集的通过率。

Self-Harness 这类工作确实让我担心：如果允许程序编辑操作系统，就会打破抽象边界。需要妥善设计可编辑范围，而且权限控制和安全层必须位于这一循环之外。[奖励黑客](https://lilianweng.github.io/posts/2024-11-28-reward-hacking/)相关的所有挑战依然存在。

**智能体式脚手架工程**（Agentic Harness Engineering，AHE；[Lin et al. 2026](https://arxiv.org/abs/2604.25850)）认为，脚手架演化的瓶颈在于 **可观测性**。也就是说，当一条推演轨迹失败时，我们需要知道是哪个组件造成的，而且每次修改都应有证据依据。

该框架以三大可观测性支柱构成闭环：

  1. _组件可观测性_：每个可编辑的脚手架组件在文件系统中都有对应表示，使动作空间明确且可追踪。
     * 脚手架包含 7 个组件：系统提示词、工具描述、工具实现、中间件、技能、子智能体配置和长期记忆。
     * 每种失败模式都会映射到一个组件，使修改更有针对性。
  2. _经验可观测性_：分析和总结大量原始轨迹，整理为分层的证据和失败模式。
     * 每个脚手架生成 $k$ 条轨迹。
     * 使用一个智能体（“Agent debugger”）分析各自存储在单独文件中的轨迹，并为每项任务生成分析报告，说明失败或成功的根本原因。
     * 所有单任务报告都会汇总为基准测试概览，供下一步使用；必要时也可访问原始轨迹。这种分层访问结构可以节省 token。
  3. _决策可观测性_：每次修改都附带一项可在下一轮验证的预测。
     * 一个智能体（“Evolve agent”）读取代码仓库，决定要修改哪个组件，然后给出修改内容及其理由。
     * 每次修改都是文件级、可证伪的主张，可在下一轮中验证，并受以下两项约束：
       * (1) 修改仅应用于脚手架工作区。运行目录、追踪器、验证器和 LLM 配置均为只读，从而杜绝一系列奖励黑客行为（例如禁用验证器、更换模型或提高推理预算），确保记录的每一项收益都可归因于脚手架修改。
       * (2) 修改由证据驱动，并附有一条修改宣言，其中包含失败证据的名称、推断出的根本原因、针对性的修复方案，以及兼顾预期修复效果和回退风险的影响预测。



在 Terminal-Bench-2 上，除 Hard 难度层级外，AHE 的表现优于人工设计的脚手架（OpenCode、Terminus-2、Codex），也优于其他几个自演化基线（ACE、TF-GRPO）。同一个冻结后的脚手架无需进一步演化，便可迁移到 SWE-bench Verified。这表明，演化后的脚手架能够将工程经验编码到脚手架组件中，而不是针对基准测试进行特定优化。

#### 演化搜索

演化搜索是一种受自然选择启发的优化方法（参见我以前关于[演化算法](https://lilianweng.github.io/posts/2019-09-05-evolution-strategies/)的文章）。它通过变异来演化解种群，并只保留其中“适应度”较高的解。演化搜索适用于以下情形：(1) 搜索空间很大或形状异常；(2) 难以直接用梯度优化，但容易评估解的优劣。脚手架搜索似乎很适合使用这种方法。

以往研究已将演化搜索用于提示词工程。**Promptbreeder**（[Fernando et al. 2023](https://arxiv.org/abs/2309.16797)）通过丰富的变异操作优化任务特定的提示词；有趣的是，变异提示词（即要求 LLM 对任务提示词进行变异的指令）本身也会通过演化得到改进。**GEPA**（[Agrawal et al. 2025](https://arxiv.org/abs/2507.19457)）将基于[反思](https://lilianweng.github.io/posts/2023-06-23-agent/#self-reflection)的提示方法与演化搜索结合起来，通过用自然语言反思反复试错的轨迹，提出提示词更新方案。

[Novikov et al. (2025)](https://arxiv.org/abs/2506.13131) 提出了 **AlphaEvolve**，这是一个编码智能体的演化搜索系统。它保存一组候选程序，并通过提示冻结权重的 LLM 生成用于改进的 diff。系统反复评估子代程序并保留成功者，久而久之便能发现更好的解法。

![](https://lilianweng.github.io/posts/2026-07-04-harness/alphaevolve.png) AlphaEvolve 的工作原理。（图片来源：[Novikov et al. 2025](https://arxiv.org/abs/2506.13131)）

AlphaEvolve 的设计中有几个重要细节：

  * 提示词包含亲本程序、结果、指令，有时还包括元信息。
  * 编码智能体可以访问完整代码仓库，但需要改进的代码区域会用 `# EVOLVE-BLOCK-START` 和 `# EVOLVE-BLOCK-END` 明确标记。
  * 元提示词根据 LLM 的建议，与指令和上下文共同演化，方式与候选解程序的演化类似。



消融实验考察了演化流程、提示词中的上下文、元提示词、全文件演化以及使用更强 LLM 的作用。

![](https://lilianweng.github.io/posts/2026-07-04-harness/alphaevolve-plot.png) 消融实验展示了 AlphaEvolve 中多项设计的价值。（图片来源：[Novikov et al. 2025](https://arxiv.org/abs/2506.13131)）

近期的变体中，**ThetaEvolve**（[Wang et al. 2025](https://arxiv.org/abs/2511.23473)）将演化搜索与强化学习和上下文学习相结合；**DemoEvolve**（[Che, et al. 2026](https://arxiv.org/abs/2605.24539)）则在自我推演轨迹（self-rollout）档案库中加入人类专家示范，作为脚手架层面诊断和编辑的参考经验。另一方面，**ShinkaEvolve**（[Lange et al. 2025](https://arxiv.org/abs/2509.19349)）引入了三个新组件，以提高 LLM 的采样效率：

  * 设计亲本采样机制，在表现排名和子代数量之间取得平衡，从而提高探索的样本效率。
  * 基于嵌入的余弦相似度，丢弃与现有种群过于相似的候选方案，进行代码新颖性拒绝采样。
  * 在元草稿区中识别成功解法的良好模式，为后续变异提供指导。



上述方法侧重改进解法，与之不同，**Darwin Gödel Machine**（DGM；[Zhang et al. 2025](https://arxiv.org/abs/2505.22954)）明确以可编辑脚手架代码仓库的演化为目标，并使用基于 LLM 的编码智能体。确切地说，这个智能体可以修改自己的脚手架。后续的 Hyperagents 工作（[Zhang et al. 2026](https://arxiv.org/abs/2603.19461)）引入元智能体，由它控制如何修改现有任务智能体，以创建新的智能体。

  1. 从池中的一个编码智能体开始。
  2. 每次迭代时选择一个亲本，对其进行修改和分支，以生成新的智能体；被选中的概率与其表现成正比，与其已有子代数量成反比。
  3. 选中的亲本智能体查看自身的基准测试评估日志，然后提出对自身脚手架代码库的改进，生成新版编码智能体。代码编辑通过两个基础工具实现：(1) bash（参数：`<bash_command>`）；(2) editor（参数：`view/create/edit <file_path>`）。
  4. 评估新的编码智能体，只有表现足够好的智能体才会被加入池中。
  5. 重复第 2～4 步，直至满足某项停止条件。



DGM 在模型固定的情况下演化脚手架。实验以 `Claude 3.5 Sonnet` 作为基础 LLM，并采用简单的初始脚手架配置；DGM 发现的智能体在 SWE-bench Verified（20% 提升至 50%）和 Polyglot（14.2% 提升至 30.7%）上的表现与手工设计的智能体相当，甚至更好。

当候选解可以自动评估且其适应度容易量化时，这类方法表现良好，例如矩阵乘法、GPU 内核优化、算法竞赛和数据中心调度。而在评估缓慢、标准模糊或主要依赖启发式方法的领域，它们往往效果不佳。演化的计算效率和有效性也令人担忧。

#### 与模型权重联合优化

脚手架演化改变的是模型周围的非参数系统。要实现全面的自我改进，完全可以同时允许模型更新自身权重。权重更新可以通过改进模型训练流水线，或在测试时进行持续学习来实现。持续学习这一主题值得以后单独写一篇文章。

**SIA**（[Hebbar et al. 2026](https://arxiv.org/abs/2605.27276)）是将脚手架改进与模型参数更新纳入同一优化循环的早期尝试，其设计包含三个组件：

  * _元智能体_：提出初始脚手架。
  * _任务特定智能体_：执行任务。
  * _反馈智能体_：根据近期轨迹，选择更新脚手架还是模型权重。

![](https://lilianweng.github.io/posts/2026-07-04-harness/SIA.png) SIA 中的反馈智能体决定下一轮迭代的类型。（图片来源：[Hebbar et al. 2026](https://arxiv.org/abs/2605.27276)）

SIA 的实验中有几个混杂因素，使结果难以解释。例如，任务特定智能体远弱于元智能体和反馈智能体使用的模型（`gpt-oss-120b` 对比 `Claude Sonnet 4.6`），基线也太弱，无法与相关方法进行清晰的交叉比较。我认为这个方向很有意思，但现有证据仍只是初步的。训练稳定性和古德哈特效应等许多挑战仍未解决。

**Continual Harness**（[Karten et al. 2026](https://arxiv.org/abs/2605.09998)）在长时程游戏场景中开展实验，一边更新脚手架，一边通过蒸馏强教师模型在低奖励轨迹上给出的标签，联合学习策略模型。

### 未来挑战

AI Scientist 系列工作有力地展示了：专家设计的脚手架可以协调自动化研究循环中的大部分环节，相关实验以撰写研究论文为形式。但论文产出并不等同于科学发现。一个系统即使能写出看似可信的论文，也仍可能捏造引文、在实现中偏离原方案，或得出薄弱的实验结果。

[Trehan & Chopra（2026）](https://arxiv.org/abs/2601.03315)测试了大语言模型能否仅借助少量脚手架和基础工具（即 `read_file`、`write_file`、`llm_search`、`list_files`），将研究构想发展为论文。每个构想都有独立的工作区，智能体可以在其中生成并读取文档，将其纳入上下文。实验覆盖三个领域（世界模型、多智能体强化学习、AI 安全与对齐），每个领域包含 45–50 篇高质量种子文档，用于启发新构想。人类专家只选出四个构想进入完整流程，最终仅有一个构想得到完整执行并写成论文。实验中反复出现了六类失败模式：

  * _偏向训练数据中的默认做法_：使用旧版程序库、过时命令和标准格式，或采用并非基于实际代码库或数据集的假设。
  * _执行压力下的实现偏移_：当实现过程在技术上变得复杂时，模型可能转向常见但更简单的方案，而不是按提出的方法继续推进。
  * _记忆与上下文退化_：在长时程项目中，如果不把日志写成持久化产物，关键细节就会丢失。
  * _过度乐观_：即使实验噪声很大或已经失败，模型仍会宣称成功。[Bubeck et al.（2025）](https://arxiv.org/abs/2511.16072)也观察到类似的“p-hacking and eureka-ing”模式：模型可能用“数值胶带”修补结果，并在信号仍只是噪声时宣布胜利。
  * _领域智能不足_：模型缺少难以言传的实践知识，例如预测实现复杂度、判断实验结果是否合理，或知道哪些基线真正重要。
  * _科研品味薄弱_：实验或许可以执行，却没有回答正确的问题。



在迈向完整的递归式自我改进（RSI）方面，研究人员已经取得实质进展，但仍有一些瓶颈。

**1\. 薄弱且模糊的评估器。** 许多研究主张都没有快速、精确的验证器，很多现实任务也是如此。现有的自我改进循环最适合评估指标可测量且客观的任务，这与[强化学习的工作方式](https://lilianweng.github.io/posts/2018-02-19-rl-overview/)相似。

科研品味、新颖性和长期科学价值则难以衡量得多。例如，科研品味往往涉及问题构建、实验设计，以及对哪些意外结果值得继续追踪、哪些失败案例值得重试的判断。

**2\. 上下文与记忆的生命周期。** 随着 AI 智能体变得更加自主和独立，记忆会不断增长。实用的脚手架需要管理上下文和记忆，以弥补现有长上下文生成能力的局限，同时尽可能提高长时程任务的成功率。人类能够终生保持记忆，我认为这与上下文工程存在类比：上下文工程将会、也应当成为智能的核心部分，而不只是停留在软件系统层。

**3\. 负面结果。** 研究人员受到发表成功结果的激励，因此文献偏向成功案例。大语言模型基于海量数据训练，而这些数据至少目前仍主要由人类创造，哈哈。由于数据中成功与失败案例失衡，模型可能不善于判断何时该放弃假设、报告负面结果，甚至承认失败。研究脚手架应当便于保留失败尝试，因为从失败中学习是缩小任务搜索空间的最佳方式。

**4\. 多样性坍缩。** 演化循环和强化学习循环往往会利用已知的高奖励模式。我们需要相应的[机制](https://lilianweng.github.io/posts/2020-06-07-exploration-drl/)，防止种群坍缩为同一种解决方案的不同变体。这对开放式研究尤为重要，因为最佳路径在当前评估器下最初可能看起来更差。

**5\. [奖励黑客](https://lilianweng.github.io/posts/2024-11-28-reward-hacking/)。** 自我改进循环会优化提供给它的任何信号。如果奖励来自单元测试，智能体可能对测试过拟合；如果奖励来自裁判模型，它可能学会针对该裁判的奖励黑客技巧；如果奖励来自基准测试分数，它可能利用基准测试中的伪特征。

评估器和权限控制或许应置于演化脚手架的循环之外，并配合留出测试、轨迹审计，以及在重要决策点进行人工审核。监督能在多大程度上扩展和自动化，仍是一个开放的研究领域。

**6\. 长期成功。** 外在优化循环所处理的奖励，位于我们可在训练沙箱中模拟的单次推演轨迹之外。

以编码智能体为例。编码智能体已经提高了软件工程的日常生产力，但许多优化目标仍然过于短期。它通常可以完成手头任务，却很难判断应如何维护一个由数百乃至数千名工程师共同管理的代码库，使其长期保持健康。基于标准沙箱的 RLVR 式训练很少能涵盖可维护性、所有权边界、迁移成本、向后兼容性或未来的调试负担。

**7\. 人类的作用。** 人类不应被移出循环，而应上移到更高层级。这意味着人类需要在恰当的时间、恰当的抽象层级提供监督，系统设计也应考虑何时以及如何设置此类介入点。

上述许多挑战都需要人类的反馈和引导。毕竟，我们开发技术是为了让人类拥有更好的未来，而不是反过来。

### 引用

请按以下格式引用本文：

> Weng, Lilian. "Harness Engineering for Self-Improvement". Lil'Log (Jul 2026). https://lilianweng.github.io/posts/2026-07-04-harness/

或使用以下 BibTeX 引用：
    
    
    @article{weng2026harness,
      title = {Harness Engineering for Self-Improvement},
      author = {Weng, Lilian},
      journal = {lilianweng.github.io},
      year = {2026},
      month = {July},
      url = "https://lilianweng.github.io/posts/2026-07-04-harness/"
    }
    

### 附录：一些实用的基准测试

  * **[PaperBench](https://arxiv.org/abs/2504.01848)**：从头复现 20 篇 ICML 2024 Spotlight 和 Oral 论文，包括理解论文贡献、开发代码库并成功执行实验。
    * 每项复现任务都拆分为更小、可单独评分的任务。
    * 共包含 8,316 条评分细则，由基准测试团队与论文作者共同制定。
    * 当时表现最好的模型（`Claude 3.5 Sonnet`，约 21%）未能超过机器学习博士。
    * 包含 PaperBench、PaperBench Code-Dev（轻量版）和 JudgeEval。
  * **[CORE-Bench](https://arxiv.org/abs/2409.11363)**：评估已发表研究的计算可复现性。
    * 包含 270 项任务，取自计算机科学、社会科学和医学领域的 90 篇科学论文。
    * 任务要求使用所提供的代码和数据复现结果。
    * 包含多个难度级别，以及纯语言任务和视觉语言任务。
    * 当时报告的最佳智能体（`GPT-4o` 和 `GPT-4o-mini`）在最难任务上的准确率仅为 21%。
  * **[ScienceAgentBench](https://arxiv.org/abs/2410.05080)**：评估用于数据驱动型科学发现的大语言模型智能体。
    * 从数学、化学、生物学和地理学四个学科的 44 篇同行评审论文中提取了 102 项任务。
    * 涵盖这些领域的基础数据科学任务：数据处理、模型开发、数据分析和信息可视化。
  * **[RE-Bench](https://arxiv.org/abs/2411.15114)**：在贴近现实的机器学习研究工程环境中，对比评估前沿 AI 智能体与人类专家。
    * 包含 7 个具有挑战性的开放式机器学习研究工程环境。
    * 每个环境 =（评分函数、初始方案、参考方案）；每个环境都可使用不超过 8 块 H100 GPU 运行。
    * 示例：优化内核、运行缩放定律实验、修复嵌入、针对问答任务微调 GPT-2 等。
    * 包含 61 位不同的人类专家完成 71 次八小时尝试的数据。
    * 在八小时尝试中，人类专家有 82% 获得了非零分数，24% 达到或超过了较强的参考方案。
    * 在两小时预算下，最佳 AI 智能体的得分是人类的 4 倍；但随着时间预算增加，人类获得的回报更高，并在八小时和三十二小时设置下超过智能体。
  * **[MLE-bench](https://arxiv.org/abs/2410.07095)**：通过离线 Kaggle 竞赛评估机器学习工程智能体。
    * 包含从 Kaggle 精选的 75 项机器学习工程竞赛。
    * 测试内容包括训练模型、准备数据集、运行实验，以及向评分脚本提交预测结果。
    * 使用 Kaggle 公开排行榜作为人类基线。
    * 论文中的最佳配置是采用 AIDE 脚手架的 `o1-preview`，在 16.9% 的竞赛中至少达到了 Kaggle 铜牌水平。
    * 包含资源扩展分析和污染分析。
  * **[KernelBench](https://arxiv.org/abs/2502.10517)**：评估所生成 GPU 内核的正确性和速度。
    * 包含 250 项 PyTorch 任务，用于评估大语言模型能否编写既快速又正确的内核。
    * 评估指标 fast_p = 正确且速度快于基线的生成内核所占百分比。



### 参考文献

[1] Good, I. J. ["Speculations Concerning the First Ultraintelligent Machine."](https://philpapers.org/rec/GOOSCT) _Advances in Computers_，6:31-88，1965.

[2] Yudkowsky, Eliezer. ["Recursive Self-Improvement."](https://www.lesswrong.com/posts/JBadX7rwdcRFzGuju/recursive-self-improvement) LessWrong，2008.

[3] Choi, et al. ["Anchored Self-Play for Code Repair."](https://openreview.net/forum?id=lTbBFAoPSA) ICML 2026.

[4] Zhao, et al. ["Absolute Zero: Reinforced Self-play Reasoning with Zero Data."](https://arxiv.org/abs/2505.03335) arXiv 预印本 arXiv:2505.03335，2025.

[5] Yuan, et al. ["Self-Rewarding Language Models."](https://arxiv.org/abs/2401.10020) arXiv 预印本 arXiv:2401.10020，2024.

[6] Chen, et al. ["Self-Play Fine-Tuning Converts Weak Language Models to Strong Language Models."](https://arxiv.org/abs/2401.01335) ICML 2024.

[7] Zhang, et al. ["Agentic Context Engineering: Evolving Contexts for Self-Improving Language Models."](https://arxiv.org/abs/2510.04618) ICLR 2026.

[8] Ye, et al. ["Meta Context Engineering via Agentic Skill Evolution."](https://arxiv.org/abs/2601.21557) arXiv 预印本 arXiv:2601.21557，2026.

[9] Lee, et al. ["Meta-Harness: End-to-End Optimization of Model Harnesses."](https://arxiv.org/abs/2603.28052) arXiv 预印本 arXiv:2603.28052，2026.

[10] Lu, et al. ["Towards end-to-end automation of AI research."](https://www.nature.com/articles/s41586-026-10265-5) _Nature_，651:914-919，2026.

[11] Meng, et al. ["ScientistOne: Towards Human-Level Autonomous Research via Chain-of-Evidence."](https://arxiv.org/abs/2605.26340) arXiv 预印本 arXiv:2605.26340，2026.

[12] Kulikov, et al. ["Autodata: An agentic data scientist to create high quality synthetic data."](https://arxiv.org/abs/2606.25996) arXiv 预印本 arXiv:2606.25996，2026.

[13] Hu, Lu, and Clune. ["Automated Design of Agentic Systems."](https://arxiv.org/abs/2408.08435) ICLR 2025.

[14] Madaan, et al. ["Self-Refine: Iterative Refinement with Self-Feedback."](https://arxiv.org/abs/2303.17651) NeurIPS 2023.

[15] Zhang, et al. ["AFlow: Automating Agentic Workflow Generation."](https://arxiv.org/abs/2410.10762) ICLR 2025.

[16] Zelikman, et al. ["Self-Taught Optimizer (STOP): Recursively Self-Improving Code Generation."](https://arxiv.org/abs/2310.02304) COLM 2024.

[17] Zhang, et al. ["Self-Harness: Harnesses That Improve Themselves."](https://arxiv.org/abs/2606.09498) arXiv 预印本 arXiv:2606.09498，2026.

[18] Fernando, et al. ["Promptbreeder: Self-Referential Self-Improvement Via Prompt Evolution."](https://arxiv.org/abs/2309.16797) arXiv 预印本 arXiv:2309.16797，2023.

[19] Agrawal, A. et al. ["GEPA: Reflective Prompt Evolution Can Outperform Reinforcement Learning."](https://arxiv.org/abs/2507.19457) arXiv 预印本 arXiv:2507.19457，2025.

[20] Novikov, et al. ["AlphaEvolve: A coding agent for scientific and algorithmic discovery."](https://arxiv.org/abs/2506.13131) arXiv 预印本 arXiv:2506.13131，2025.

[21] Lange, Imajuku, and Cetin. ["ShinkaEvolve: Towards Open-Ended And Sample-Efficient Program Evolution."](https://arxiv.org/abs/2509.19349) arXiv 预印本 arXiv:2509.19349，2025.

[22] Wang, et al. ["ThetaEvolve: Test-time Learning on Open Problems."](https://arxiv.org/abs/2511.23473) arXiv 预印本 arXiv:2511.23473，2025.

[23] Zhang, et al. ["Darwin Gödel Machine: Open-Ended Evolution of Self-Improving Agents."](https://arxiv.org/abs/2505.22954) arXiv 预印本 arXiv:2505.22954，2025.

[24] Zhang, et al. ["Hyperagents."](https://arxiv.org/abs/2603.19461) arXiv 预印本 arXiv:2603.19461，2026.

[25] Yuksekgonul, et al. ["Learning to Discover at Test Time."](https://arxiv.org/abs/2601.16175) arXiv 预印本 arXiv:2601.16175，2026.

[26] Riaz, et al. ["Epistemic Uncertainty for Test-Time Discovery."](https://arxiv.org/abs/2605.11328) arXiv 预印本 arXiv:2605.11328，2026.

[27] Hebbar, et al. ["SIA: Self Improving AI with Harness & Weight Updates."](https://arxiv.org/abs/2605.27276) arXiv 预印本 arXiv:2605.27276，2026.

[28] Trehan and Chopra. ["Why LLMs Aren't Scientists Yet: Lessons from Four Autonomous Research Attempts."](https://arxiv.org/abs/2601.03315) arXiv 预印本 arXiv:2601.03315，2026.

[29] Bubeck, et al. ["Early science acceleration experiments with GPT-5."](https://arxiv.org/abs/2511.16072) arXiv 预印本 arXiv:2511.16072，2025.

[30] Starace, et al. ["PaperBench: Evaluating AI's Ability to Replicate AI Research."](https://arxiv.org/abs/2504.01848) ICML 2025.

[31] Wijk, et al. ["RE-Bench: Evaluating frontier AI R&D capabilities of language model agents against human experts."](https://arxiv.org/abs/2411.15114) ICML 2025.

[32] Chan, et al. ["MLE-bench: Evaluating Machine Learning Agents on Machine Learning Engineering."](https://arxiv.org/abs/2410.07095) arXiv 预印本 arXiv:2410.07095，2024.

[33] Chen, et al. ["ScienceAgentBench: Toward Rigorous Assessment of Language Agents for Data-Driven Scientific Discovery."](https://arxiv.org/abs/2410.05080) ICLR 2025.

[34] Siegel, et al. ["CORE-Bench: Fostering the Credibility of Published Research Through a Computational Reproducibility Agent Benchmark."](https://arxiv.org/abs/2409.11363) TMLR 2024.

[35] Ouyang, et al. ["KernelBench: Can LLMs Write Efficient GPU Kernels?"](https://arxiv.org/abs/2502.10517) arXiv 预印本 arXiv:2502.10517，2025.

[36] Lin, et al. ["Harness Updating Is Not Harness Benefit: Disentangling Evolution Capabilities in Self-Evolving LLM Agents."](https://arxiv.org/abs/2605.30621) arXiv 预印本 arXiv:2605.30621，2026.

[37] Lin, et al. ["Agentic Harness Engineering: Observability-Driven Automatic Evolution of Coding-Agent Harnesses."](https://arxiv.org/abs/2604.25850) arXiv 预印本 arXiv:2604.25850，2026.

[38] Karten, et al. ["Continual Harness: Online Adaptation for Self-Improving Foundation Agents."](https://arxiv.org/abs/2605.09998) arXiv 预印本 arXiv:2605.09998，2026.

[39] Che, et al. ["DemoEvolve: Overcoming Sparse Feedback in Agentic Harness Evolution with Demonstrations."](https://arxiv.org/abs/2605.24539) arXiv 预印本 arXiv:2605.24539，2026.

---

## English Original

### Harness Engineering for Self-Improvement

Date: July 4, 2026 | Estimated Reading Time: 31 min | Author: Lilian Weng

Table of Contents

  * Harness Design Patterns
    * Pattern 1: Workflow Automation
    * Pattern 2: File System as Persistent Memory
    * Pattern 3: Sub-agent and Backend Jobs
    * Case study: Coding Agent Harness
    * Harness Layer vs Core Intelligence?
  * Harness Optimization
    * Context Engineering
    * Workflow Design
    * Self-Improving Harness
    * Evolutionary Search
    * Joint Optimization with Model Weights
  * Future Challenges
  * Citation
  * Appendix: Some useful benchmarks
  * References



The concept of **recursive self-improvement (RSI)** dates back to [I. J. Good (1965)](https://philpapers.org/rec/GOOSCT), where he defined an "ultraintelligent machine" as a system that can surpass humans in all intellectual activities and design better machines to improve itself. [Yudkowsky (2008)](https://www.lesswrong.com/posts/JBadX7rwdcRFzGuju/recursive-self-improvement) used the phrase "recursive self-improvement" for a specific feedback loop: an AI uses its current intelligence to improve the cognitive machinery that produces its intelligence.

This feedback loop in modern AI may indicate the model rewriting its own weights directly, or more broadly the model improves the _training pipeline_ and the _deployment system_ , which in turn enables a better successor model with improved performance across economically valuable tasks. The speed of research development in AI has been shown to drastically accelerated in frontier labs ([Anthropic](https://www.anthropic.com/institute/recursive-self-improvement); [OpenAI](https://openai.com/index/how-agents-are-transforming-work/)).

I explicitly mention _" deployment system"_ because the layer between the raw model and the real-world context seems to be as important as the model's raw intelligence (i.e. the evals right after pretraining). Harnesses are important components of AI deployment, as shown by successful coding agent products such as Claude Code and Codex. A **harness** is the system surrounding a base model that orchestrates execution and decides how the model thinks and plans, calls tools and acts, perceives and manages context, stores artifacts, and evaluates results.

This one post will focus on research around harness engineering and how it contributes to RSI. Much recent work on auto-research, self-improving agents, and evolutionary program search can be organized around this question. Other work on model self-play, synthetic data, test-time training and a broader theme of continual learning also matches the RSI vision (e.g. [Yuan et al. 2024](https://arxiv.org/abs/2401.10020), [Chen et al. 2024](https://arxiv.org/abs/2401.01335)), [Zhao et al. 2025](https://arxiv.org/abs/2505.03335), [Choi et al. 2026](https://openreview.net/forum?id=lTbBFAoPSA))) but they will not be the focus of this post.

### Harness Design Patterns

Compared with [early agent frameworks](https://lilianweng.github.io/posts/2023-06-23-agent/), "agent = LLM + memory + tools + planning + action", harnesses engineering additionally include _workflow design (e.g. loop engineering), evaluation, permission controls, and persistent state management_. It is no longer only prompt templates, but closer to runtime and software system design: how the model observes, acts, memorizes, checks itself, and improves.

The design should be deliberately simple and generic to enable generalization, likely with reference to existing software engineering practices to benefit from prertaining knowlege. There is also a strong analogy between operating systems and harnesses. Similar to an OS, a harness should encapsulate complicated logic while keeping the interface simple. Meanwhile, configs, tool interfaces and other protocols may gradually become standardized across the industry.

#### Pattern 1: Workflow Automation

Defining a workflow in which the model can operate, test, and iterate is a key design for automation. Karpathy's autoresearch repo (<https://github.com/karpathy/autoresearch>) is a clean example of how such a workflow can be constructed. A common workflow follows a goal-oriented loop of plan, execute, observe/test, improve, and execute again _until_ the goal is achieved. The process may trigger proactive requests to users for clarity in task specification or execution preference.

![](https://lilianweng.github.io/posts/2026-07-04-harness/openai-agent-loop.png) A simplified Codex agent loop: the agent calls tools and tool responses affect the model's next generation.
(Image source: [OpenAI codex agent post](https://openai.com/index/unrolling-the-codex-agent-loop/))

The workflow graph also emphasizes the model analyzing its own trajectories and failure cases and then iterating on its progress through an "agent runtime" rather than a static prompt template.

#### Pattern 2: File System as Persistent Memory

A recurring pattern in long-horizon agent systems is simple control over rich states and artifacts. A harness should not carry the entire workflow and all logs in context; instead, it should keep durable state in files. In long-horizon agentic rollout, artifacts such as experiment logs, code diffs, paper summaries, error traces, and past rollout trajectories often grow much longer than the context window that the model has trained for.

Learning how to read, write, and edit the file system (commonly via `bash` commands) is a foundation skill for LLMs, and thus managing persistent memory in the simple form of files naturally benefits from improvements in core model capability.

#### Pattern 3: Sub-agent and Backend Jobs

A harness can spawn multiple subagents to execute in parallel and monitor backend jobs. This is useful when the main agent needs to search multiple hypotheses, run experiments concurrently, or delegate isolated subtasks without polluting the main context. The parent agent then needs a small process manager: launch jobs, inspect logs, cancel failed runs, and merge results back into the main agent thread.

The key design choice is to make parallelism explicit and inspectable. If subagent outputs only live in a transient chat context, they quickly become obselete and hidden. If they are stored as files, logs, and status records, the model can recover after interruptions and reason over its own execution history.

#### Case study: Coding Agent Harness

The core interface of mainstream coding agents has become stabilized across Claude Code, Codex, OpenCode, and Cursor-style agents. They commonly use a loop like:

![](https://lilianweng.github.io/posts/2026-07-04-harness/coding-harness-loop.png)

With access to a set of tools, the coding agent is able to develop and debug issues in a given repository, similar to how human developers are equipped with IDEs.

(Not a comprenhensive list; shown for demonstration. Read [this](https://github.com/yasasbanukaofficial/claude-code) if interested.)

Group | Tool definitions
---|---
File system | \- File discovery: `glob`, `grep`, `ls`
\- File read: `read`, `read_many`
\- File modification: `write` (a whole new file); `edit` (string exact-match replacement); `multi_edit`; `apply_patch` (applies a structured patch/diff)
Shell execution | Run commands: `bash`, `PowerShell`
IO | `lsp`, git tools like `git_status`, `git_diff`, `git_commit`
External context | MCP tools, Skills
Web search | `web_search`, `web_fetch`, browser tools
Artifacts | Read docs, images; generate HTML, images
Backend processes | Such as: `CronCreate`, `CronDelete`, `CronList`
Agent delegation | Such as: `spawn_agent`, `resume_agent`, `wait_agent`, `list_agents`, `close_agent`, `interrupt_agent`, etc.

#### Harness Layer vs Core Intelligence?

It is hard to forecast how much the future of RSI will rely on harness engineering, but the near-term path of RSI is unlikely to start as a model directly rewriting its weights. My prediction of a practical near-term path is:

  1. Harness engineering will evolve in the direction of meta-methodology (i.e. improving the machinery for getting better answers, not just improving the answer itself). The harness system itself becomes an optimization target, with fewer heuristic rules and more general mechanisms.
  2. In turn, mature harnesses enable auto-research for model self-improvement loop and smarter models prevents harnesses from overengineering and keep the system sustainable.



Eventually it is possible that many harness improvements will be _internalized_ into core model behavior, but the interface with external context and tools should remain. We have seen a softer version of this pattern with [prompt engineering](https://lilianweng.github.io/posts/2023-03-15-prompt-engineering/): manual prompt tricks became less central as instruction tuning and model reasoning improved, but _the need to specify goals, constraints, context, and evaluation did not disappear_.

### Harness Optimization

The progression in the object being optimized in the harness system is roughly: instruction [prompts](https://lilianweng.github.io/posts/2023-03-15-prompt-engineering/) → structured context → workflow → harness code → optimizer code. As the model becomes more intelligent and powerful, we move toward more complex targets and generic methods.

#### Context Engineering

Simply appending all the tool responses and model generations into the context can quickly grow out of control as the agentic job horizon increases significantly. Context management is a layer to construct a more structed and concise context for LLM and manage persistant states. There is no doubt that long-context research will keep on making progress but at the moment long-context intelligence and context engineering sometime intertwines.

**Agentic Context Engineering** (ACE; [Zhang et al. 2025](https://arxiv.org/abs/2510.04618)) treats context as an evolving playbook rather than an increasingly lengthening prompt. It has three components to maintain one context playbook of bullet points, each with an identifier and a description.

  1. _Generator_ : produces task trajectories, with reference to bullet points.
  2. _Reflector_ : distills insights from successful and failed trajectories.
  3. _Curator_ : updates the structured context with incremental, itemized entries.

![](https://lilianweng.github.io/posts/2026-07-04-harness/ace.png) The framework of Agentic Context Engineering (ACE). (Image source: [Zhang et al. 2025](https://arxiv.org/abs/2510.04618))

To prevent context collapse and brevity bias during iterative rewrites, one key design choice in ACE is that the curator does not rewrite a full prompt blob. It instead outputs a collection of structured, itemized bullets in the form of (identifier, description), and these bullets are merged into a structured context logbook with deterministic logic. The context items are refined and deduplicated periodically.

The fact that ACE learns insights from rollouts helps us move toward self-managed memory, but the update rules and the overall workflow are still handcrafted. To move toward a more self-improving loop, **Meta Context Engineering** (MCE; [Ye et al. 2026](https://arxiv.org/abs/2601.21557)) separates the mechanism (how to manage context) from the artifact content (what is in context), running skill evolution at the meta-optimization level and context optimization at the base level.

An MCE skill $s \in \mathcal{S}$ defines a context function $c_s=(\rho_s,F_s)$ and maps an input $x$ to context $c = F_s(x;\rho_s)$, where:

  * $\rho_s = \\{\rho_1,\dots,\rho_m\\}$ are static components (prompts, knowledge bases, code libraries).
  * $F_s = \\{F_1,\dots,F_k\\}$ are dynamic operators (search, selection, filtering, formatting).



The bi-level optimization is to find the best context $c_s^*$ given skill $s$ on the training data, while the outer loop finds the optimal skill that provides the best performance on the validation set:

$$ \text{Inner: }c_s^*=\arg\max_{c_s}J_\text{train}(c_s;s)\quad \text{Outer: }s^*=\arg\max_{s\in\mathcal{S}}J_\text{val}(c_s^*) $$

The skill database tracks the history of previous skills, context functions and eval metrics $\mathcal{H}_{k-1} = \\{(s_i,c_i,J_i^\text{train}, J_i^\text{val})\\}_{i=1}^{k-1}$. A meta-level agent performs agentic [crossover](https://en.wikipedia.org/wiki/Crossover_\(evolutionary_algorithm\)) over prior skills to create a new skill given a task $\tau$: $s_k=\text{crossover}(\tau,\mathcal{H}_{k-1})$.

Then a base-level context engineer executes the skill $s_k$ and learns the context function from rollout feedback $\mathcal{R}_k$, guided by the current skill: $c_k=\text{engineer}(\tau,s_k;c_{k-1}^*,\mathcal{R}_k)$.

![](https://lilianweng.github.io/posts/2026-07-04-harness/mce.png) The framework of Meta Context Engineering (MCE): meta-level skill evolution searches over context-management mechanisms, while the base level optimizes the task context. (Image source: [Ye et al. 2026](https://arxiv.org/abs/2601.21557))

MCE does not enforce a heuristic rule for how to structure context as ACE does. It uses _free-form skills_ to store the most important knowledge for a task, and evolves the skill and the skill-conditioned context iteratively together. Implementation-wise, a context function $c$ is instantiated as a collection of files in a dedicated directory, including both static (`skill.md`) and dynamic (context and data rollouts) components. Both meta-level and base-level optimization are executed in agentic coding envs with a standard tool set,

$$ \mathcal{T}=\\{\texttt{Read},\texttt{Write},\texttt{Edit},\texttt{Bash},\texttt{Glob},\texttt{Grep},\texttt{TodoWrite}\\} $$

**Meta-Harness** ([Lee et al. 2026](https://arxiv.org/abs/2603.28052)) moves another level deeper: the optimized object is the _code_ that determines and optimizes what information should be stored, retrieved, and presented to the model. "Meta-" in its name means it is a harness for optimizing harnesses.

![](https://lilianweng.github.io/posts/2026-07-04-harness/meta-harness-outer-loop.png) The Meta-Harness outer-loop optimization algorithm. (Image source: [Lee et al. 2026](https://arxiv.org/abs/2603.28052))

The proposer for creating a new harness is itself a coding agent and the final output is a collection of harness candidates on the Pareto frontier.

  * The entire execution history is accessible via a file system, and thus the coding agent uses commands like `grep` or `cat` to read through it instead of shoveling everything into a single prompt context.
  * The proposed harness is a dictionary in the file system containing its own source code, scores, rollout trajectories, and state updates.
  * The mete-harness loop iteratively creates new harnesses, and only qualified ones are kept.

![](https://lilianweng.github.io/posts/2026-07-04-harness/meta-harness.png) The performance of Meta-Harness on (Left) text classification with a small number of iterations and (Right) TerminalBench-2. Note that the search in the TerminalBench-2 experiment is initialized from Terminus-KIRA and Terminus-2, two very strong harnesses. (Image source: [Lee et al. 2026](https://arxiv.org/abs/2603.28052))

Still, the important lesson is clear: once harness design becomes an executable search space, a strong coding agent can exploit the same design space human engineers use.

#### Workflow Design

Workflow design in harness engineering can be handcrafted by domain experts. Taking auto-research as an example, various frameworks have been proposed and tested. The **AI Scientist** system ([Lu et al. 2026](https://www.nature.com/articles/s41586-026-10265-5)) builds a pipeline to propose research ideas, write code, run experiments, analyze results, write a manuscript, and perform peer review. [Meng et al. (2026)](https://arxiv.org/abs/2605.26340) make verifiability the central design constraint in **ScientistOne** , where every claim (citation, numerical, methodological, conclusion) must trace to an evidence source and is audited by Chain-of-Evidence checks.

![](https://lilianweng.github.io/posts/2026-07-04-harness/ai-scientist.png) AI Scientist pipeline for idea generation, experimentation, paper writing, and review. (Image source: [Lu et al. 2026](https://www.nature.com/articles/s41586-026-10265-5))

The **Autodata** agent ([Kulikov et al. 2026](https://arxiv.org/abs/2606.25996)) is designed to work as a data scientist for generating training and evaluation data. The main agent manages a _challenger_ that proposes problems, a _weak solver_ , a _strong solver_ , and a _verifier/judge_ , aiming to synthesize data at the "just right" level of difficulty, meaning that the strong solver succeeds but the weak solver fails.

In Autodata, the challenger prompt is updated iteratively according to feedback from the solvers and verifier. The limitation here is that synthesized tasks are used to fine-tune weak solvers but not strong solvers; if the loop cannot iteratively improve the strong model, it is more like indirect distillation over a generated prompt distribution, with less RSI flavor.

![](https://lilianweng.github.io/posts/2026-07-04-harness/autodata.png) Autodata agentic workflow design for generating synthetic training and evaluation data around challenger, solver, and verifier roles. (Image source: [Kulikov et al. 2026](https://arxiv.org/abs/2606.25996))

The design space for workflow is _enormous_ , and naturally we can think of workflow design as a search problem, and therefore we should be able to find good solutions by algorithms rather than only manually craft them. Following this direction, **Automated Design of Agentic Systems** (ADAS; [Hu et al. 2025](https://arxiv.org/abs/2408.08435)) formulates agent design itself as an optimization problem, "meta-agent search" where a meta-agent proposes new designs of agentic workflows.

  1. Initialize an archive of agentic workflows with simple agents such as CoT and self-refine.
  2. Ask a meta-agent to program new agents, all in _code_ , inspired by existing solutions in the archive.
     * The meta-agent first generates a high-level description of the new workflow, and then implements it in code.
     * The draft program then goes through two self-refine steps (i.e. ask the model to provide feedback and then ask the same model to refine the previously generated outputs based on the feedback; [Madaan et al. 2023](https://arxiv.org/abs/2303.17651)) by the meta-agent to check its novelty.
  3. Evaluate each new candidate and add successful ones back to the archive.
  4. Repeat steps 2-3 until the maximum iteration count is reached.

![](https://lilianweng.github.io/posts/2026-07-04-harness/adas.png) Illustration of Automated Design of Agentic Systems (ADAS).
(Image source: [Hu et al. 2025](https://arxiv.org/abs/2408.08435))

**AFlow** ([Zhang et al. 2025](https://arxiv.org/abs/2410.10762)) represents an agentic workflow as a graph, where nodes represent LLM-invoking actions and edges implement logical operations in code. The workflow optimization relies on [MCTS](https://en.wikipedia.org/wiki/Monte_Carlo_tree_search) (Monte Carlo Tree Search):

  1. Initialize the starting workflow $W_0$ in the tree with a template.
  2. Select a workflow node using a soft mixture of score and uniform exploration.
  3. Expand it by asking an LLM to produce a modified workflow conditioned on its evaluation performance.
  4. Execute and evaluate the new workflow.
  5. Add it back to the tree if the new workflow shows improvement within a budget of $N$ rounds.
  6. Repeat steps 2-5 and stop when the top-$k$ average score plateaus or hit the budget.

![](https://lilianweng.github.io/posts/2026-07-04-harness/aflow.png) AFlow optimization process over a tree of workflow candidates. (Image source: [Zhang et al. 2025](https://arxiv.org/abs/2410.10762))

Experiments of AFlow in QA, code, and math tasks showed decent improvement of AFlow over manually designed workflows and ADAS.

![](https://lilianweng.github.io/posts/2026-07-04-harness/aflow-exp.png) AFlow experiments in comparison to manual methods and ADAS. (Image source: [Zhang et al. 2025](https://arxiv.org/abs/2410.10762))

#### Self-Improving Harness

Either context engineering or workflow design is only one part of a harness. We need to search through the entire design space and optimize context-management logic, workflow, permissions, and many other harness components together. As we have seen in work like Meta-Harness, ADAS, and AFlow, **✨code✨** is a **universal language** for defining programs and systems. In simple words, a harness is code that programs how prompts, tool calls, subagents, control flow, memory, and workflow logic work together. If an LLM can optimize the code that executes agents, it can access a _much larger design space_ than hand-written prompts.

**Self-Taught Optimizer** (STOP; [Zelikman et al. 2023](https://arxiv.org/abs/2310.02304)) is one of the early examples of recursive scaffolding improvement. A seed improver $I_0$ at step $t=0$ takes an initial solution $s$, a utility function $u$, and a black-box language model $M$, and returns an improved solution $s'$, that is, $s' = I(u, s; M)$. The goal of STOP is not directly to improve $s$ but _to improve the improver $I$ itself_.

First, let's define the meta-utility as the average utility of a given improver function $I$ over a collection of downstream tasks $\mathcal{D}$:

$$ \hat{u}(I) \triangleq \frac{1}{\vert\mathcal{D}\vert}\mathbb{E}_{(u,s)\sim \mathcal{D}}[u(I(u,s; M))] $$

Because improving the improver function is an optimization problem itself, we can recursively get a new version of $I_t$ based on $I_{t-1}$'s performance measured by meta-utility via a self-improvement update:

$$ I_t=I_{t-1}(\hat{u},I_{t-1};M) $$

![](https://lilianweng.github.io/posts/2026-07-04-harness/STOP-algo.png) Algorithm of Self-Taught Optimizer (STOP). (Image source: [Zelikman et al. 2023](https://arxiv.org/abs/2310.02304))

In their experiments, the improved improver discovered various strategies, such as genetic algorithms, decomposing and improving parts, multi-armed prompt bandits, simulated annealing, varying temperature, and beam/tree search. This is analogous to how a harness workflow can be represented as an object for optimization.

![](https://lilianweng.github.io/posts/2026-07-04-harness/STOP-patterns.png) Examples of self-improvement strategies discovered by STOP. (Image source: [Zelikman et al. 2023](https://arxiv.org/abs/2310.02304))

A _cautionary_ result in Zelikman et al. (2023)'s findings is that STOP improved mean downstream performance across iterations with GPT-4 but degraded with weaker models like GPT-3.5 and Mixtral. Recursive structure alone is not enough. The base model must be _capable enough_ to improve the mechanism. This implies that harness improvement enables better deployment of the model but intelligence is still the core.

[Lin et al. (2026)](https://arxiv.org/abs/2605.30621) investigated the dependency of harness evolution on model capabilities in more details. They disentangled two axes: (1) _harness-updating_ refers to the capability of producing useful harness edits and (2) _harness-benefit_ denotes the capability of utilizing the updated harness, to achieve better task solving. Interestingly a range of model of different sizes and core intelligence, from Qwen3.5-9B to Claude Opus 4.6, were observed in their experiments to show similar harness updating capability; the 9B harness proposer/evolver is able to write a skill procedurally isomorphic to Opus. To best utilize a harness, a model needs to invoke skills/tools correctly and timely and be good at long-horizon instruction following.

![](https://lilianweng.github.io/posts/2026-07-04-harness/harness-update.png) Main results: (A) harness updating capability is measured flat across a range of models from Qwen2-32B to Opus 4.6; (B) harness benefit capability is non-monotonic where middle tier models benefit the most. (Image source: [Lin et al. 2026](https://arxiv.org/abs/2605.30621))

A more recent work, **Self-Harness** ([Zhang et al. 2026](https://arxiv.org/abs/2606.09498)), relies on LLM agents to improve their own harness via a propose-evaluate-accept loop.

![](https://lilianweng.github.io/posts/2026-07-04-harness/self-harness.png) Self-Harness uses a loop of weakness mining, bounded harness proposal, and validation to update a harness. (Image source: [Zhang et al. 2026](https://arxiv.org/abs/2606.09498))

The loop in Self-Harness has three stages:

  1. _Weakness mining_ : cluster failures into verifier-grounded failure patterns.
     * The current harness $h_t$ is used to evaluate on tasks and execution traces are collected for analysis.
     * Note that two runs can share the same verifier outcome in the error logs on the surface, such as timeout or missing artifact, while having different causal mechanisms. Therefore we need a failure record of rich information, containing the terminal verifier-level cause, the causal status of the relevant agent behavior, and the abstract agent mechanism exposed by the trace, to uncover the root causes.
  2. _Harness proposal_ : propose bounded harness edits based on mined failure patterns.
     * The same model is invoked under $h_t$ as a proposer.
     * The model is provided with a bounded proposal context: (1) the editable surfaces of the current harness, (2) the verifier-grounded failure patterns from the evaluation system, (3) records of passing behaviors that should be preserved, and (4) summaries of previously attempted edits.
     * Harness edits should prefer recurrent error patterns that are addressable (e.g. not task-specific difficulty) and can be resolved by narrow changes.
     * Harness edit candidates should be distinct and diverse.
  3. _Proposal validation_ : validate and merge qualified edits to create a new harness $h_{t+1}$.
     * Candidate edits are evaluated by regression tests on held-in $D_\text{in}$ (for testing whether the weakness is resolved) and held-out $D_\text{out}$ (for checking whether other unknown issues were introduced) splits.
     * Candidates are accepted only if they have no regression on both held-in and held-out data.
     * Accepted candidates are merged to update the harness to $h_{t+1}$, while rejected candidates are logged without changing the active harness.



When running `MiniMax M2.5`, `Qwen3.5-35B-A3B`, and `GLM-5` on Terminal-Bench-2, Self-Harness was shown to learn model-specific harness instructions that target at different weaknesses of different base models and improve held-out pass rates.

Self-harness type of work does raise my concerns that if a program is allowed to edit the OS system, abstraction boundaries are broken. The editable surface needs to be properly designed and the permission control and security layers need to live outside this loop. All the challenges around [reward hacking](https://lilianweng.github.io/posts/2024-11-28-reward-hacking/) still remain.

**Agentic Harness Engineering** (AHE; [Lin et al. 2026](https://arxiv.org/abs/2604.25850)) see the bottlenecks of harness evolution are around **observability** --that is, when a rollout fails, we need to know which component is responsible for that and every edit should be grounded by evidence.

The framework creates a closed loop with 3 observability pillars:

  1. _Component observability_ : every editable harness component has a representation in the file system so the action space is explicit and tracable.
     * A harness contains 7 components: system prompt, tool description, tool implementation, middleware, skill, sub-agent configuration, and long-term memory.
     * Each failure pattern is mapped to one component so the edit can be more targeted.
  2. _Experience observability_ : analysize and summarize a large amount of raw trajectories into a hierarchy of evidence and failure patterns.
     * Each harness generates $k$ traces.
     * Use an agent ("Agent debugger") to analysis the trajectories each stored in one file and generate per-task analysis report on the root cause for the failure or success.
     * All the per-task reports are aggregated into a benchmark overview for the next step, and raw traces can be accessed if needed. This layered access structure is more token efficient.
  3. _Decision observability_ : every edit is paired with a prediction for the next round to validate.
     * An agent ("Evolve agent") reads the repo and decides which component to edit, and then produces the edit and the reasoning behind it.
     * Every edit is a file-level, falsifiable claim and can be verified in the next round, under two constraints:
       * (1) Edits are only applied to the harness workspace. the runs directory, tracer, verifier, and LLM configuration are read-only, which disables a set of reward hacking (e.g disabling the verifier, swapping the model, or raising the reasoning budget) and thus it can keep every recorded gain attributable to harness edits.
       * (2) Edits are evidence-driven, with a manifesto entry: the failure evidence's name, the inferred root cause, the targeted fix, and a predicted impact comprising both expected fixes and at-risk regressions.



On Terminal-Bench-2, AHE achieved better than human-designed harness (OpenCode, Terminus-2, Codex) except for Hard tier and a few other self-evolve baselines (ACE, TF-GRPO). The same frozen harness, without further evolving, transfers to SWE-bench-verified, indicating that the evolved harness is able to encode engineering experience into harness components rather than doing benchmark-specific optimization.

#### Evolutionary Search

Evolutionary search is an optimization method inspired by natural selection (see my old post on [evolutionary algorithm](https://lilianweng.github.io/posts/2019-09-05-evolution-strategies/)). It evolves a population of solutions by mutating them and only keeping those with high "fitness" in the crowd. Evolutionary search comes in handy when (1) the search space is extensive or weirdly shaped; and (2) it is hard to optimize directly with gradients but easy to evaluate solutions. Harness search seems to be a good fit here.

Evolutionary search has been used in prompt engineering in the past studies. **Promptbreeder** ([Fernando et al. 2023](https://arxiv.org/abs/2309.16797)) optimizes task-specific prompts through a rich set of mutation operations, and interestingly the mutation prompts (i.e. instructions to an LLM to mutate a task prompt) are themselves also improved through evolution. **GEPA** ([Agrawal et al. 2025](https://arxiv.org/abs/2507.19457)) combines [reflection](https://lilianweng.github.io/posts/2023-06-23-agent/#self-reflection)-based prompting with evolutionary search and uses natural language reflection over trajectories of trial and error to propose prompt updates.

[Novikov et al. (2025)](https://arxiv.org/abs/2506.13131) introduced **AlphaEvolve** as a coding-agent evolutionary search system, which stores a pool of candidate programs and prompts frozen LLMs to generate diffs for improvement. As the system repeatedly evaluates child programs and keeps successful ones, it discovers better solutions in time.

![](https://lilianweng.github.io/posts/2026-07-04-harness/alphaevolve.png) How AlphaEvolve works. (Image source: [Novikov et al. 2025](https://arxiv.org/abs/2506.13131))

A few details matter in the design of AlphaEvolve:

  * The prompt includes parent programs, results, instructions, and sometimes meta information.
  * The coding agent has access to the full repo, but code regions for improvement are explicitly marked with `# EVOLVE-BLOCK-START` and `# EVOLVE-BLOCK-END`.
  * Meta-prompt co-evolves with instructions and context as suggested by LLM, in a similar way as how we evolve solution programs.



Ablations show the evolution procedure, context in prompts, meta-prompts, full-file evolution and the use of stronger LLMs.

![](https://lilianweng.github.io/posts/2026-07-04-harness/alphaevolve-plot.png) Ablations show the value of everal designs in AlphaEvolve. (Image source: [Novikov et al. 2025](https://arxiv.org/abs/2506.13131))

Recent variants such as **ThetaEvolve** ([Wang et al. 2025](https://arxiv.org/abs/2511.23473)) combines evolutionary search with RL and in-context learning, and **DemoEvolve** ([Che, et al. 2026](https://arxiv.org/abs/2605.24539)) augments the self-rollout archive with human expert demonstrations as reference experience for harness-level diagnosis and editing. **ShinkaEvolve** ([Lange et al. 2025](https://arxiv.org/abs/2509.19349)), on the other hand, introduced three new components to improve LLM sampling efficiency:

  * More sample-efficient exploration by designing parent sampling to balance performance rank and offspring count.
  * Code-novelty rejection sampling by discarding candidates that are too similar to the existing population based on embedding-based cosine similarity.
  * Identifying good patterns in successful solutions in a meta-scratchpad to guide future mutation.



Unlike the methods above, which focus on solution improvement, **Darwin Gödel Machine** (DGM; [Zhang et al. 2025](https://arxiv.org/abs/2505.22954)) explicitly targets the evolution of an editable harness-code repository with an LLM-based coding agent. Precisely, this agent is allowed to modify its own harness. A follow-up work on Hyperagents ([Zhang et al. 2026](https://arxiv.org/abs/2603.19461)) introduced a meta-agent to control how to modify existing task agents to create new ones.

  1. Start with one coding agent in the pool.
  2. In each iteration, pick one parent with a probability proportional to its performance and inversely to the number of children it has, to modify and branch off to produce new agents.
  3. The selected parent agent examines its own benchmark evaluation log and then proposes improvements to its own harness codebase to generate a new version of the coding agent. Code editing is implemented with two basic tools: (1) bash (args: `<bash_command>`) and (2) editor (args: `view/create/edit <file_path>`).
  4. New coding agents are evaluated, and only those with sufficiently high performance are added back into the pool.
  5. Repeat steps 2-4 until some stop criteria hit.



DGM is harness evolution under a fixed model. In experiments with `Claude 3.5 Sonnet` as the base LLM and simple initial harness configs, the DGM-discovered agents are comparable to or outperform handcrafted agents on SWE-bench Verified (20% to 50%) and Polyglot (14.2% to 30.7%).

This family of methods works well when candidate solutions are automatically evaluable and candidate fitness is easy to quantify, such as matrix multiplication, GPU kernel optimization, algorithm contests, datacenter scheduling. It struggles with domains where evaluation is slow, ambiguous, or mostly heuristic-based. The compute efficiency and effectiveness of evolution are also concerns.

#### Joint Optimization with Model Weights

Harness evolution changes the non-parametric system around the model. To enable full self-improvement, the model can totally be allowed to update its own weights at the same time. The weight update can be implemented via improvements in the model training pipeline or continual learning at test time. The topic of continual learning is worthy of its own post in the future.

**SIA** ([Hebbar et al. 2026](https://arxiv.org/abs/2605.27276)) is an early attempt to combine harness improvement and model-parameter updates in the same optimization loop, with three components in the design:

  * _Meta-Agent_ : proposes the initial harness.
  * _Task-Specific Agent_ : executes the task.
  * _Feedback-Agent_ : chooses whether to update the harness or the model weights based on recent trajectories.

![](https://lilianweng.github.io/posts/2026-07-04-harness/SIA.png) The Feedback-Agent in SIA decides the next iteration type. (Image source: [Hebbar et al. 2026](https://arxiv.org/abs/2605.27276))

There are a few confounding choices in SIA's experiments that make the results hard to interpret. For example, the task-specific agent is much weaker than the models used for the Meta-Agent and Feedback-Agent (`gpt-oss-120b` vs `Claude Sonnet 4.6`), and the baselines are too weak to cross-reference cleanly against related methods. I would consider the direction interesting, but the evidence provisional. Yet many challenges, such as training stability and Goodhart effect, still remain open.

**Continual Harness** ([Karten et al. 2026](https://arxiv.org/abs/2605.09998)) experimented in long-horizon gameplay setting with harness updating and co-learning a policy model by distilling a strong teacher model's labels on low-reward trajectories.

### Future Challenges

The AI Scientist line of work is a strong demonstration that an expert-designed harness can coordinate a large portion of auto-research loop, experimented in the form of writing research papers. But paper production is not identical to scientific discovery. A system can write a plausible manuscript while still having fabricated citations, implementation drift, or weak experimental results.

[Trehan & Chopra (2026)](https://arxiv.org/abs/2601.03315) tested whether LLMs can go from a research idea to a paper with minimal scaffolding and basic tools (i.e., `read_file`, `write_file`, `llm_search`, `list_files`). Each idea had a dedicated workspace where agents could generate and read documents as part of context. They experimented in three domains (world models, multi-agent RL, AI safety & alignment), with each domain containing 45-50 high-quality seed documents to inspire new ideas. Only four ideas were selected by human experts to run through the full pipeline, and only one was fully executed into a paper. They observed six recurring failure modes in the experiments:

  * _Bias toward training-data defaults_ : use old libraries, stale commands, standard formats, or assumptions not grounded in the actual repository or dataset.
  * _Implementation drift under execution pressure_ : when implementation becomes technically complex, the model may move toward a common simpler solution rather than the proposed method.
  * _Memory and context degradation_ : long-horizon projects lose critical details unless logs are written as persistent artifacts.
  * _Over-optimism_ : the model declares success despite noisy or failed experiments, similarly observed as "p-hacking and eureka-ing" pattern by [Bubeck et al. (2025)](https://arxiv.org/abs/2511.16072) where models can introduce "numerical duct tape" and declare victory when signals are still noise.
  * _Insufficient domain intelligence_ : the model lacks tacit craft knowledge, e.g. predicting implementation complexity, judging whether an experimental result is plausible, or knowing which baselines matter.
  * _Weak scientific taste_ : experiments may be executable but fail to answer the right question.



Toward full RSI, researchers have made real progress, but several bottlenecks remain.

**1\. Weak and fuzzy evaluators.** Many research claims do not have a fast and precise verifier, and the same is true for many real-world tasks. Current self-improvement loops work best for tasks when evaluation metrics are measurable and objective, similar as [how RL works](https://lilianweng.github.io/posts/2018-02-19-rl-overview/).

Research taste, novelty, and long-term scientific value are much harder to measure. For example, research taste often mixes problem framing, experimental design, and judgment about which surprising results are worth pursuing and which failure cases are worth retries.

**2\. Context and memory lifecycle.** Memory grows as AI agents become more autonomous and independent. A useful harness needs to manage context and memory to complement existing limitation in long-context generation while still maximizing the success of long-horizon tasks. Since humans are able to maintain memory through our life time, I see an anoloy here that context engineering will and should become a core part of intelligence, rather than staying in the software system layer.

**3\. Negative results.** Researchers are incentivized to publish successful results and thus literature is biased toward successes. LLMs trained on a vast amount of data (mostly human created, at least for now, lol) may be bad at deciding when to abandon a hypothesis, report a negative result, or even acknowledge a failure due to the imablance of success vs failure cases in data. A research harness should make failed attempts easy to preserve, as learning from failure is the best way to trim down the task search space.

**4\. Diversity collapse.** Evolutionary and RL loops tend to exploit known high-reward patterns. We need [mechanisms](https://lilianweng.github.io/posts/2020-06-07-exploration-drl/) to prevent the population from collapsing into variants of the same solution. This is especially critical for open-ended research, where the best path may initially look worse under the current evaluator.

**5.[Reward hacking](https://lilianweng.github.io/posts/2024-11-28-reward-hacking/).** A self-improvement loop optimizes whatever signal it is given. If the reward comes from unit tests, the agent may overfit to tests; if it comes from a judge model, it may learn reward hacking tricks specific to this judge; if it comes from benchmark scores, it may exploit benchmark artifacts.

The evaluator and permission control should likely sit outside the loop that evolves harness, with held-out tests, trace audits, and human review at decision points that matter--how much oversight can be scaled up and automated remains an open research area.

**6\. Long-term success.** An extrinsic loop of optimization works on rewards outside of individual rollouts that we can simulate in training sandbox.

Take coding agent as an example. Coding agents have already increased daily productivity in software engineering, but many optimization goals are still too short-term. It can often complete the task at hand, but less obvious how it should protect the long-term health of a repo collectively maintained by hundreds or thousands of engineers. Standard sandbox-based RLVR-style training rarely captures maintainability, ownership boundaries, migration cost, backwards compatibility, or future debugging burden.

**7\. The role of humans.** Humans should move up the stack, not be removed from the loop, meaning that human should provide oversight at the right time, at the right abstraction level and our system design should consider when and how to set up such touch points.

Many challenges listed above need human's feedback and steering. After all, we are building the technology for better future of humanity, not other way around.

### Citation

Please cite this work as:

> Weng, Lilian. "Harness Engineering for Self-Improvement". Lil'Log (Jul 2026). https://lilianweng.github.io/posts/2026-07-04-harness/

Or use the BibTeX citation:


    @article{weng2026harness,
      title = {Harness Engineering for Self-Improvement},
      author = {Weng, Lilian},
      journal = {lilianweng.github.io},
      year = {2026},
      month = {July},
      url = "https://lilianweng.github.io/posts/2026-07-04-harness/"
    }


### Appendix: Some useful benchmarks

  * **[PaperBench](https://arxiv.org/abs/2504.01848)** : replicate 20 ICML 2024 Spotlight and Oral papers from scratch, including understanding paper contributions, developing a codebase, and successfully executing experiments.
    * Each replication task is decomposed into smaller, individually gradable tasks.
    * 8,316 rubrics in total, co-developed with the paper authors.
    * The best model at the time (`Claude 3.5 Sonnet`, ~21%) does not outperform ML PhDs.
    * Includes PaperBench, PaperBench Code-Dev (a lighter version), and JudgeEval.
  * **[CORE-Bench](https://arxiv.org/abs/2409.11363)** : evaluate computational reproducibility of published research.
    * 270 tasks based on 90 scientific papers across computer science, social science, and medicine.
    * Tasks involve reproducing results from provided code and data.
    * Includes multiple difficulty levels and both language-only and vision-language tasks.
    * The best reported agent at the time (`GPT-4o` and `GPT-4o-mini`) achieved only 21% accuracy on the hardest task.
  * **[ScienceAgentBench](https://arxiv.org/abs/2410.05080)** : evaluate LLM agents for data-driven scientific discovery.
    * Extracts 102 tasks from 44 peer-reviewed publications in four disciplines (math, chemistry, biology, geography).
    * Covers basic data-science tasks in these domains: data processing, model development, data analysis, and information visualization.
  * **[RE-Bench](https://arxiv.org/abs/2411.15114)** : evaluate frontier AI agents on realistic ML research-engineering envs against human experts.
    * 7 challenging, open-ended ML research-engineering environments.
    * Each environment = (scoring function, starting solution, reference solution); each can be run with 8 or fewer H100 GPUs.
    * Examples: optimize a kernel, run a scaling-law experiment, fix an embedding, fine-tune GPT-2 for QA, etc.
    * Includes data from 71 eight-hour attempts by 61 distinct human experts.
    * Human experts achieved non-zero score in 82% of 8-hour attempts; 24% matched or exceeded strong reference solutions.
    * Best AI agents scored 4× higher than humans at a 2-hour budget, but humans had better returns to longer budgets and exceeded agents at 8-hour and 32-hour settings.
  * **[MLE-bench](https://arxiv.org/abs/2410.07095)** : evaluate ML engineering agents on offline Kaggle competitions.
    * Contains 75 ML-engineering competitions curated from Kaggle.
    * Tests training models, preparing datasets, running experiments, and submitting predictions to grading scripts.
    * Uses Kaggle public leaderboards as human baselines.
    * Best setup in the paper, `o1-preview` with AIDE scaffolding, reached at least Kaggle bronze-medal level in 16.9% of competitions.
    * Includes resource-scaling and contamination analyses.
  * **[KernelBench](https://arxiv.org/abs/2502.10517)** : evaluate correctness and speed for generated GPU kernels.
    * 250 PyTorch tasks to evaluate whether LLM can write fast and correct kernels.
    * The evaluation metric fast_p = the percentage of generated kernels that are correct and faster than baseline.



### References

[1] Good, I. J. ["Speculations Concerning the First Ultraintelligent Machine."](https://philpapers.org/rec/GOOSCT) _Advances in Computers_ , 6:31-88, 1965.

[2] Yudkowsky, Eliezer. ["Recursive Self-Improvement."](https://www.lesswrong.com/posts/JBadX7rwdcRFzGuju/recursive-self-improvement) LessWrong, 2008.

[3] Choi, et al. ["Anchored Self-Play for Code Repair."](https://openreview.net/forum?id=lTbBFAoPSA) ICML 2026.

[4] Zhao, et al. ["Absolute Zero: Reinforced Self-play Reasoning with Zero Data."](https://arxiv.org/abs/2505.03335) arXiv preprint arXiv:2505.03335, 2025.

[5] Yuan, et al. ["Self-Rewarding Language Models."](https://arxiv.org/abs/2401.10020) arXiv preprint arXiv:2401.10020, 2024.

[6] Chen, et al. ["Self-Play Fine-Tuning Converts Weak Language Models to Strong Language Models."](https://arxiv.org/abs/2401.01335) ICML 2024.

[7] Zhang, et al. ["Agentic Context Engineering: Evolving Contexts for Self-Improving Language Models."](https://arxiv.org/abs/2510.04618) ICLR 2026.

[8] Ye, et al. ["Meta Context Engineering via Agentic Skill Evolution."](https://arxiv.org/abs/2601.21557) arXiv preprint arXiv:2601.21557, 2026.

[9] Lee, et al. ["Meta-Harness: End-to-End Optimization of Model Harnesses."](https://arxiv.org/abs/2603.28052) arXiv preprint arXiv:2603.28052, 2026.

[10] Lu, et al. ["Towards end-to-end automation of AI research."](https://www.nature.com/articles/s41586-026-10265-5) _Nature_ , 651:914-919, 2026.

[11] Meng, et al. ["ScientistOne: Towards Human-Level Autonomous Research via Chain-of-Evidence."](https://arxiv.org/abs/2605.26340) arXiv preprint arXiv:2605.26340, 2026.

[12] Kulikov, et al. ["Autodata: An agentic data scientist to create high quality synthetic data."](https://arxiv.org/abs/2606.25996) arXiv preprint arXiv:2606.25996, 2026.

[13] Hu, Lu, and Clune. ["Automated Design of Agentic Systems."](https://arxiv.org/abs/2408.08435) ICLR 2025.

[14] Madaan, et al. ["Self-Refine: Iterative Refinement with Self-Feedback."](https://arxiv.org/abs/2303.17651) NeurIPS 2023.

[15] Zhang, et al. ["AFlow: Automating Agentic Workflow Generation."](https://arxiv.org/abs/2410.10762) ICLR 2025.

[16] Zelikman, et al. ["Self-Taught Optimizer (STOP): Recursively Self-Improving Code Generation."](https://arxiv.org/abs/2310.02304) COLM 2024.

[17] Zhang, et al. ["Self-Harness: Harnesses That Improve Themselves."](https://arxiv.org/abs/2606.09498) arXiv preprint arXiv:2606.09498, 2026.

[18] Fernando, et al. ["Promptbreeder: Self-Referential Self-Improvement Via Prompt Evolution."](https://arxiv.org/abs/2309.16797) arXiv preprint arXiv:2309.16797, 2023.

[19] Agrawal, A. et al. ["GEPA: Reflective Prompt Evolution Can Outperform Reinforcement Learning."](https://arxiv.org/abs/2507.19457) arXiv preprint arXiv:2507.19457, 2025.

[20] Novikov, et al. ["AlphaEvolve: A coding agent for scientific and algorithmic discovery."](https://arxiv.org/abs/2506.13131) arXiv preprint arXiv:2506.13131, 2025.

[21] Lange, Imajuku, and Cetin. ["ShinkaEvolve: Towards Open-Ended And Sample-Efficient Program Evolution."](https://arxiv.org/abs/2509.19349) arXiv preprint arXiv:2509.19349, 2025.

[22] Wang, et al. ["ThetaEvolve: Test-time Learning on Open Problems."](https://arxiv.org/abs/2511.23473) arXiv preprint arXiv:2511.23473, 2025.

[23] Zhang, et al. ["Darwin Gödel Machine: Open-Ended Evolution of Self-Improving Agents."](https://arxiv.org/abs/2505.22954) arXiv preprint arXiv:2505.22954, 2025.

[24] Zhang, et al. ["Hyperagents."](https://arxiv.org/abs/2603.19461) arXiv preprint arXiv:2603.19461, 2026.

[25] Yuksekgonul, et al. ["Learning to Discover at Test Time."](https://arxiv.org/abs/2601.16175) arXiv preprint arXiv:2601.16175, 2026.

[26] Riaz, et al. ["Epistemic Uncertainty for Test-Time Discovery."](https://arxiv.org/abs/2605.11328) arXiv preprint arXiv:2605.11328, 2026.

[27] Hebbar, et al. ["SIA: Self Improving AI with Harness & Weight Updates."](https://arxiv.org/abs/2605.27276) arXiv preprint arXiv:2605.27276, 2026.

[28] Trehan and Chopra. ["Why LLMs Aren't Scientists Yet: Lessons from Four Autonomous Research Attempts."](https://arxiv.org/abs/2601.03315) arXiv preprint arXiv:2601.03315, 2026.

[29] Bubeck, et al. ["Early science acceleration experiments with GPT-5."](https://arxiv.org/abs/2511.16072) arXiv preprint arXiv:2511.16072, 2025.

[30] Starace, et al. ["PaperBench: Evaluating AI's Ability to Replicate AI Research."](https://arxiv.org/abs/2504.01848) ICML 2025.

[31] Wijk, et al. ["RE-Bench: Evaluating frontier AI R&D capabilities of language model agents against human experts."](https://arxiv.org/abs/2411.15114) ICML 2025.

[32] Chan, et al. ["MLE-bench: Evaluating Machine Learning Agents on Machine Learning Engineering."](https://arxiv.org/abs/2410.07095) arXiv preprint arXiv:2410.07095, 2024.

[33] Chen, et al. ["ScienceAgentBench: Toward Rigorous Assessment of Language Agents for Data-Driven Scientific Discovery."](https://arxiv.org/abs/2410.05080) ICLR 2025.

[34] Siegel, et al. ["CORE-Bench: Fostering the Credibility of Published Research Through a Computational Reproducibility Agent Benchmark."](https://arxiv.org/abs/2409.11363) TMLR 2024.

[35] Ouyang, et al. ["KernelBench: Can LLMs Write Efficient GPU Kernels?"](https://arxiv.org/abs/2502.10517) arXiv preprint arXiv:2502.10517, 2025.

[36] Lin, et al. ["Harness Updating Is Not Harness Benefit: Disentangling Evolution Capabilities in Self-Evolving LLM Agents."](https://arxiv.org/abs/2605.30621) arXiv preprint arXiv:2605.30621, 2026.

[37] Lin, et al. ["Agentic Harness Engineering: Observability-Driven Automatic Evolution of Coding-Agent Harnesses."](https://arxiv.org/abs/2604.25850) arXiv preprint arXiv:2604.25850, 2026.

[38] Karten, et al. ["Continual Harness: Online Adaptation for Self-Improving Foundation Agents."](https://arxiv.org/abs/2605.09998) arXiv preprint arXiv:2605.09998, 2026.

[39] Che, et al. ["DemoEvolve: Overcoming Sparse Feedback in Agentic Harness Evolution with Demonstrations."](https://arxiv.org/abs/2605.24539) arXiv preprint arXiv:2605.24539, 2026.
