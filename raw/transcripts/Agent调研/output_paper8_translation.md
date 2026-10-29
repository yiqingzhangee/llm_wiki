# 基于大语言模型的多智能体：进展与挑战综述

**Large Language Model based Multi-Agents: A Survey of Progress and Challenges**  

arXiv:2402.01680v2  [cs.CL]  19 Apr 2024

**作者：**Taicheng Guo¹, Xiuying Chen², Yaqi Wang³*, Ruidi Chang, Shichao Pei⁴, Nitesh V. Chawla¹, Olaf Wiest¹, Xiangliang Zhang¹†  
¹University of Notre Dame  
²King Abdullah University of Science and Technology  
³Southern University of Science and Technology  
⁴University of Massachusetts Boston  
{tguo2, nchawla, owiest, xzhang33}@nd.edu, xiuying.chen@kaust.edu.sa, ywang84@nd.edu, ruidic@alumni.cmu.edu, shichao.pei@umb.edu

*本文工作完成时，Yaqi 为圣母大学（University of Notre Dame）的访问学生。  
†通讯作者。

---

## 摘要（Abstract）

大语言模型（Large Language Models, LLMs）在各种任务上都取得了显著成功。得益于 LLM 的卓越规划与推理能力，它们已经被用作自主智能体（agents），以自动完成许多任务。近期，在将单个 LLM 作为单一规划或决策智能体这一发展基础之上，基于 LLM 的多智能体系统（multi-agent systems）在复杂问题求解和世界模拟方面取得了长足进展。为向研究社区提供该快速发展领域的整体概览，我们撰写了本综述，对基于 LLM 的多智能体系统的关键方面及其面临的挑战进行深入讨论。我们的目标是让读者能够对以下问题获得实质性的认识：LLM 驱动的多智能体系统可以在什么领域和环境中进行模拟？这些智能体是如何被刻画（profiled）的，又如何进行通信？哪些机制促成了智能体能力的增长？对有意深入该研究方向的读者，我们还总结了常用的数据集和基准，便于他们方便获取和使用。为帮助研究者及时了解最新研究进展，我们维护了一个开源的 GitHub 代码库，用于持续整理基于 LLM 的多智能体系统研究工作。

---

## 1 引言（Introduction）

大语言模型（Large Language Models, LLMs）近期在推理与规划能力方面展现出非凡潜力，其水平已可与人类相媲美。这种能力恰好契合了人类对自主智能体的期待：它们能够感知周围环境、做出决策并采取相应行动 [Xi et al., 2023; Wooldridge and Jennings, 1995; Russell and Norvig, 2009; Guo et al., 2023; Liang et al., 2023]。因此，基于 LLM 的智能体（LLM-based agent）成为研究热点，并得到快速发展，用以理解并生成类似人类的指令，从而促进更复杂的交互及在广泛场景中的决策制定 [Yao et al., 2023; Shinn et al., 2023; Li et al., 2023d]。一些及时出现的综述论文对基于 LLM 的智能体的进展进行了系统总结，代表性工作可见 [Xi et al., 2023; Wang et al., 2023b]。

在单个基于 LLM 的智能体展现出的鼓舞人心的能力基础之上，人们进一步提出了基于 LLM 的多智能体（LLM-based Multi-Agents），以利用多个智能体的集体智能以及其专业化的画像和技能。相较于仅使用单个 LLM 智能体的系统，多智能体系统通过以下两点提供了更高级的能力：1）将 LLM 专门化为不同的智能体，每个智能体具备不同的能力；2）允许这些多样化智能体之间进行交互，从而有效模拟复杂的现实世界环境。在这种情境下，多名自主智能体能够协同参与规划、讨论和决策，类似于人类团队在解决问题任务中的合作方式。这一方法充分利用了 LLM 的沟通能力，即其生成用于交流的文本以及对文本输入做出响应的能力。此外，它还利用了 LLM 在各个领域的广博知识，以及其在特定任务上潜在的专业化能力。

近期研究表明，利用基于 LLM 的多智能体来解决多种任务取得了可喜成果，例如软件开发 [Hong et al., 2023; Qian et al., 2023]、多机器人系统 [Mandi et al., 2023; Zhang et al., 2023c]、社会模拟 [Park et al., 2023; Park et al., 2022]、政策模拟 [Xiao et al., 2023; Hua et al., 2023]、以及游戏模拟 [Xu et al., 2023c; Wang et al., 2023c]。由于该领域具有交叉学科的研究特征，它吸引了来自更广泛背景的研究者，不仅包括人工智能专家，还包括社会科学、心理学和政策研究领域的学者。

相关研究论文的数量正在迅速增长，如图 1 所示（图 1 的设计借鉴自 [Gao et al., 2023b]），从而不断扩大基于 LLM 的多智能体研究的影响力。然而，早期的工作大多是相互独立开展的，尚缺乏系统性的综述来对这些研究进行总结、构建该领域的整体蓝图，并审视未来研究所面临的挑战。这凸显了我们工作的意义，也构成了撰写本综述论文——专门聚焦于基于 LLM 的多智能体系统研究——的主要动机。

**图 1：** 基于 LLM 的多智能体研究领域的上升趋势。对于“问题求解（Problem Solving）”和“世界模拟（World Simulation）”，我们将当前工作划分为若干类别，并以 3 个月为间隔统计不同类型论文的数量。每个叶节点处的数字表示该类别中论文的数量。

我们期望本综述能为 LLM 的研究与发展以及更广泛的跨学科研究提供重要贡献。读者可以从中获得关于基于 LLM 的多智能体（LLM-MA）系统的全面概览，理解基于 LLM 构建多智能体系统所涉及的基本概念，并把握该动态领域中最新的研究趋势和应用。我们认识到，这一领域目前仍处于早期阶段，正伴随着不断涌现的新方法和新应用而快速演进。为提供一份可持续更新的资源以补充我们的综述论文，我们维护了一个开源 GitHub 仓库¹。我们希望本综述能够激发该领域的进一步探索与创新，并推动其在众多研究学科中的应用。

---

¹ https://github.com/taichengguo/LLM_MultiAgents_Survey_Papers

---

为帮助不同背景的研究者理解 LLM-MA 技术，并在已有综述的基础上解答尚未解决的问题，我们将本文结构组织如下。第 2 节首先介绍相关背景知识；随后，我们围绕一个核心问题展开：LLM-MA 系统如何与协作式任务求解环境相契合？为回答这一问题，我们在第 3 节提出一个全面的框架，用于定位、区分并连接 LLM-MA 系统的不同方面。我们通过以下几个维度展开讨论：1）智能体-环境接口（agents-environment interface），描述智能体如何与任务环境交互；2）智能体画像（agent profiling），解释如何利用 LLM 对智能体进行刻画以使其表现出特定行为；3）智能体通信（agent communication），考察智能体如何交换消息并协同工作；4）智能体能力获取（agent capability acquisition），探讨智能体如何发展其能力以有效解决问题。

审视 LLM-MA 研究的另一个视角是其应用层面。在第 4 节，我们将当前应用大致划分为两个主要方向：用于问题求解的多智能体（multi-agents for problem solving）和用于世界模拟的多智能体（multi-agents for world simulation）。为帮助研究者选择合适的工具与资源，我们在第 5 节介绍了用于研究 LLM-MA 的开源实现框架，以及可用的数据集和基准。在前述总结的基础上，我们在第 6 节展开关于未来研究挑战与机遇的讨论。第 7 节给出全文结论。

---

## 2 背景（Background）

### 2.1 基于 LLM 的单智能体系统（Single-Agent Systems Powered LLMs）

我们首先按照 [Weng, 2023] 中的讨论，对基于 LLM 的单智能体系统的能力进行概述，以引出后续背景。

**决策思维（Decision-making Thought）：** 该术语指的是基于 LLM 的智能体在提示（prompts）的引导下，将复杂任务拆解为更小的子目标 [Khot et al., 2023]，并有条理地逐步思考每个部分（有时会探索多条路径）[Yao et al., 2023]，以及从过去经验中学习 [Shinn et al., 2023]，以在复杂任务上做出更优的决策。这种能力增强了单个基于 LLM 的智能体的自主性，并提高其在问题求解中的有效性。

**工具使用（Tool-use）：** 基于 LLM 的智能体具备调用外部工具和资源来完成任务的能力，从而提升其功能能力，并使其能在多样且动态的环境中更高效地运作 [Li et al., 2023d; Ruan et al., 2023; Gao et al., 2023b]。

**记忆（Memory）：** 该能力指的是基于 LLM 的智能体通过上下文学习（in-context learning）[Dong et al., 2023a] 作为短期记忆，或通过外部向量数据库 [Lewis et al., 2021] 作为长期记忆，来在较长时间跨度内存储和检索信息 [Wang et al., 2023b]。这种能力使得单个基于 LLM 的智能体能够维持上下文连贯性，并从交互中增强学习效果。

### 2.2 单智能体系统 vs. 多智能体系统（Single-Agent VS. Multi-Agent Systems）

由 LLM 驱动的单智能体系统已经展现出令人鼓舞的认知能力 [Sumers et al., 2023]。构建此类系统的重点在于设计其内部机制以及与外部环境的交互方式。相对地，基于 LLM 的多智能体（LLM-MA）系统更强调多样化的智能体画像、智能体间的交互以及集体决策过程。从这个角度看，多名自主智能体的协作可以处理更加动态且复杂的任务，其中每个智能体都配备有独特的策略和行为，并与其他智能体进行通信。

---

## 3 解剖 LLM-MA 系统：接口、画像、通信与能力
（Dissecting LLM-MA Systems: Interface, Profiling, Communication, and Capabilities）

在本节中，我们深入探讨 LLM-MA 系统的内部机制：在这些系统中，多名自主智能体在问题求解场景中以类似人类团队协作的方式开展合作活动。我们将面对一个关键问题：这些 LLM-MA 系统如何与其运行环境以及它们被设计用来实现的集体目标相一致？为阐明这一点，我们在图 2 中给出了此类系统的一般架构。我们的分析将这些系统的运行框架拆解为四个关键方面：智能体-环境接口、智能体画像、智能体通信以及智能体能力获取。

**图 2：** LLM-MA 系统的总体架构示意图。

### 3.1 智能体-环境接口（Agents-Environment Interface）

运行环境定义了 LLM-MA 系统被部署并与之交互的具体情境或场景。例如，这些环境可以是软件开发 [Hong et al., 2023]、游戏 [Mao et al., 2023]，以及金融市场 [Li et al., 2023g]、社会行为建模 [Park et al., 2023] 等诸多领域。基于 LLM 的智能体在环境中进行感知与行动，环境反过来会影响它们的行为和决策过程。

举例来说，在“狼人杀”游戏模拟中，沙盒环境（sandbox environment）设定了整个游戏框架，包括从白天到黑夜的转换、讨论阶段、投票机制以及奖励规则。智能体（如狼人和预言家）会执行特定动作，例如击杀或查验身份。在这些动作之后，智能体会从环境中获得反馈，了解游戏的当前状态；这些信息指导智能体随时间调整其策略，以应对不断演化的游戏过程以及与其他智能体的交互。

“智能体-环境接口”指的是智能体与环境交互并感知环境的方式。正是通过这一接口，智能体得以理解周围环境、做出决策并从其行动结果中进行学习。我们将当前 LLM-MA 系统中的智能体-环境接口分为三类：沙盒（Sandbox）、物理（Physical）以及无（None），具体如表 1 所示。

- **沙盒（Sandbox）：** 指由人类构建的模拟或虚拟环境，在其中智能体可以较为自由地交互并尝试各种动作和策略。此类接口广泛应用于软件开发（例如以代码解释器作为模拟环境）[Hong et al., 2023]、游戏（例如基于游戏规则构建模拟环境）[Mao et al., 2023] 等场景。
- **物理（Physical）：** 指真实世界环境，在其中智能体与物理实体交互，并遵循现实世界的物理规律与约束。在物理空间中，智能体通常需要执行会产生直接物理结果的动作。例如，在扫地、做三明治、打包杂货、整理柜子等任务中，机器人智能体需要迭代地执行动作、观察物理环境并持续调整其行动 [Mandi et al., 2023]。
- **无（None）：** 指不存在特定外部环境、智能体也不与任何环境交互的场景。例如，许多应用 [Du et al., 2023; Xiong et al., 2023; Chan et al., 2023] 使用多个智能体围绕某个问题展开辩论，以达成共识。这类应用主要关注智能体之间的通信，而不依赖外部环境。

### 3.2 智能体画像（Agents Profiling）

在 LLM-MA 系统中，智能体由其特质、可执行的动作以及技能所定义，这些要素被设计用来满足特定目标。在不同系统中，智能体会承担不同角色，并附带详细描述，包括其特征、能力、行为方式及约束条件。

例如，在游戏环境中，智能体可以被刻画为具有不同角色和技能的玩家，每个玩家以不同方式为游戏目标做出贡献；在软件开发场景中，智能体可以扮演产品经理和工程师等角色，各自承担不同的职责并拥有相应的专业知识，从而引导整个开发流程；类似地，在辩论平台中，智能体可以被设置为正方辩手、反方辩手或裁判，各自具备独特的职能与策略，以有效地履行其角色。这些画像对于定义智能体在各自环境中的交互方式及其有效性至关重要。表 1 列出了近期 LLM-MA 工作中的智能体画像示例。

关于**智能体画像方法（Agent Profiling Methods）**，我们将其分为三类：预定义（Pre-defined）、模型生成（Model-Generated）和数据驱动（Data-Derived）。在预定义情形中，智能体画像由系统设计者显式给出；在模型生成方法中，智能体画像由模型（例如大语言模型）自动生成；数据驱动方法则基于已有数据集来构建智能体画像。

### 3.3 智能体通信（Agents Communication）

在 LLM-MA 系统中，智能体之间的通信是支撑集体智能的关键基础设施。我们从三个角度拆解智能体通信：1）**通信范式（Communication Paradigms）**——智能体之间交互的风格和方式；2）**通信结构（Communication Structure）**——多智能体系统中通信网络的组织形态与架构；3）智能体之间交换的**通信内容（Communication Content）**。

**通信范式：** 当前的 LLM-MA 系统主要采用三种通信范式：协作（Cooperative）、辩论（Debate）和竞争（Competitive）。

- 在**协作范式**中，智能体共同朝着共享目标或一组目标努力，通常会相互交换信息以提升整体解决方案的质量；
- **辩论范式**下，智能体以对抗性形式进行交互，各自提出并捍卫自己的观点或解决方案，并对他人的观点进行批判；该范式特别适合用于达成共识或获得更为精炼的解决方案；
- 在**竞争范式**中，智能体各自追求自身目标，这些目标可能与其他智能体的目标相冲突。

**通信结构：** 图 3 展示了 LLM-MA 系统中四种典型的通信结构。

**图 3：** 多智能体系统中典型的智能体通信结构示意图。

- **分层（Layered）通信结构**具有层级化特征，各层中的智能体扮演不同角色，并主要在本层或相邻层之间进行交互。[Liu et al., 2023] 提出了一个名为 Dynamic LLM-Agent Network（DyLAN）的框架，将智能体组织在多层前馈网络中。该结构支持推理时的动态智能体选择以及早停机制，从而提升智能体间协作的效率。
- **去中心化（Decentralized）通信结构**建立在点对点（peer-to-peer）网络之上，智能体直接彼此通信。这种结构在世界模拟类应用中被广泛采用。
- **中心化（Centralized）通信结构**中存在一个中心智能体或一组中心智能体负责协调系统内的通信，其他智能体主要通过该中心节点进行交互。
- **共享消息池（Shared Message Pool）**结构由 MetaGPT [Hong et al., 2023] 提出，用于提升通信效率。在这种结构下，系统维护一个共享消息池，智能体将消息发布到池中，并根据自身画像订阅相关消息，从而提高通信效率。

**通信内容：** 在 LLM-MA 系统中，通信内容通常以文本形式存在。具体内容会因应用而异。例如，在软件开发中，智能体之间可能围绕代码片段进行交流；在“狼人杀”等游戏模拟中，智能体可能会讨论自己的分析、怀疑对象或策略。

### 3.4 智能体能力获取（Agents Capabilities Acquisition）

智能体能力获取是 LLM-MA 中极其关键的过程，它使得智能体能够进行动态学习与演化。在这一背景下，有两个基本概念：一是智能体应当从哪些类型的反馈中学习以增强其能力；二是智能体如何调整自身以有效解决复杂问题的策略。

**反馈（Feedback）：** 反馈是指智能体接收的关于其动作结果的关键信息，帮助智能体了解其行为可能产生的影响，并在面对复杂、动态问题时进行适应。在大多数研究中，提供给智能体的反馈形式为文本。根据智能体接收反馈的来源不同，可以将反馈分为四类：

1. **来自环境的反馈（Feedback from Environment）**：例如来自真实世界环境或虚拟环境 [Wang et al., 2023b]。这类反馈在大多数面向问题求解的 LLM-MA 场景中十分常见，包括软件开发（智能体从代码解释器获取反馈）、具身多智能体系统（机器人从真实或模拟环境获取反馈）等。
2. **来自智能体交互的反馈（Feedback from Agents Interactions）**：指反馈来源于其他智能体的判断或交流。这在诸如科学辩论等问题求解场景中很常见，智能体通过相互交流学习如何批判和改进结论。在诸如游戏模拟等世界模拟场景中，智能体会基于先前与其他智能体的交互来调整策略。
3. **来自人类的反馈（Human Feedback）**：直接来源于人类，对于使多智能体系统与人类价值和偏好保持一致至关重要。这类反馈广泛用于大多数“人类在环（Human-in-the-loop）”应用中 [Wang et al., 2021]。
4. **无反馈（None）**：在某些情况下，系统不会向智能体提供反馈。这通常出现在专注于分析模拟结果、而非强调智能体规划能力的世界模拟工作中。在这类场景中（如传播模拟），研究重点在于结果分析，因此反馈不是系统组成部分。

**智能体对复杂问题的调整（Agents Adjustment to Complex Problems）：** 为了增强自身能力，LLM-MA 系统中的智能体可以通过三类主要方案进行适应：

1. **记忆（Memory）：** 大多数 LLM-MA 系统利用记忆模块帮助智能体调整行为。智能体会将先前交互及反馈的信息存储在记忆中；在执行动作时，可以检索相关的、有价值的记忆，尤其是那些包含在类似目标下曾经成功动作的信息 [Wang et al., 2023b]，从而提升当前行动的质量。
2. **自我演化（Self-Evolution）：** 与仅依赖历史记录来决定后续动作的记忆式方案不同，智能体可以通过自我演化机制动态改变自身，例如修改初始目标和规划策略，并根据反馈或通信日志自行训练。[Nacimiento et al., 2023] 提出了一个自我控制闭环过程，使多智能体系统中的每个智能体都能够自主管理、自适应于动态环境，从而提升多智能体协作效率。[Zhang et al., 2023b] 提出了 ProAgent，它能够预判队友决策，并基于智能体之间的通信日志动态调整各智能体的策略，以促进相互理解并提升协同规划能力。[Wang et al., 2023a] 讨论了“通过通信进行学习”（Learning through Communication, LTC）范式，利用多智能体的通信日志生成数据集来训练或微调 LLM。LTC 通过与环境及其他智能体的交互持续进行适应和改进，突破了仅依赖上下文学习或监督微调的局限，因为后者并未充分利用交互过程中从环境及外部工具获得的反馈进行持续训练。自我演化允许智能体在画像或目标层面进行自主调整，而不仅仅是从历史交互中学习。
3. **动态生成（Dynamic Generation）：** 在某些场景中，系统可以在运行过程中即时生成新的智能体 [Chen et al., 2023a; Chen et al., 2023c]。这一能力使系统能够更有效地扩展和适应，因为它可以引入专门设计用来应对当前需求和挑战的智能体。

随着 LLM-MA 的规模化，智能体数量不断增多，如何管理各类智能体的复杂性成为一项关键问题。“智能体编排（Agents Orchestration）”因此成为核心挑战，并在 [Moura, 2023; Dibia, 2023] 中开始受到关注。我们将在第 6.4 节进一步讨论这一主题。

---

## 4 应用（Applications）

LLM-MA 系统已被应用于广泛的场景。我们在表 1 中将应用大致分为两大类：**问题求解（Problem Solving）**和**世界模拟（World Simulation）**。下面我们对这些应用进行详细介绍。需要注意的是，这是一个快速发展的研究领域，新应用几乎每天都会出现。我们维护了一个开源仓库以持续记录最新工作。

**表 1：** LLM-MA 研究工作概览。我们根据各工作的动机、研究领域和目标进行分类，并从智能体-环境接口（Agents-Environment Interface）、智能体画像（Agents Profiling）、智能体通信（Agents Communication）以及智能体能力获取（Agents Capability Acquisition）等方面对每项工作进行细化说明。“-” 表示该工作中未明确提及相应要素。

- **问题求解（Problem Solving） — 软件开发（Software Development）**  
  - **[Qian et al., 2023]**：环境接口为沙盒（Sandbox）；画像方法为预定义（Pre-defined）与模型生成（Model-Generated）；典型角色包括首席技术官（CTO）、程序员（programmer）；通信范式为协作（Cooperative）；通信结构为分层（Layered）；反馈来源包括环境（Environment）、智能体交互（Agent interaction）和人类（Human）；能力调整方式包括记忆（Memory）和自我演化（Self-Evolution）。  
  - **[Hong et al., 2023]**：环境接口为沙盒；画像方法为预定义；典型角色包括产品经理（Product Manager）与工程师（Engineer）；通信范式为协作；通信结构为分层与共享消息池（Layered, Shared Message Pool）；反馈来源包括环境、智能体交互和人类；能力调整方式包括记忆与自我演化。  
  - **[Dong et al., 2023b]**：环境接口为沙盒；画像方法为预定义与模型生成；典型角色包括分析师（Analyst）、程序员（coder）；通信范式为协作；通信结构为分层；反馈来源包括环境与智能体交互；能力调整方式为记忆与自我演化。  

- **问题求解 — 多机器人规划 / 具身智能体 / 多机器人协作（Multi-robot planning & Embodied Agents & Multi-robot collaboration / cooperation）**  
  - **[Chen et al., 2023d]（多机器人规划）**：环境接口为沙盒与物理环境（Sandbox, Physical）；画像方法为预定义；典型角色为机器人（Robots）；通信范式为协作；通信结构包含中心化和去中心化（Centralized, Decentralized）；反馈来源为环境与智能体交互；能力调整方式为记忆。  
  - **[Mandi et al., 2023]（多机器人协作）**：环境接口为沙盒与物理环境；画像方法为预定义；典型角色为机器人；通信范式为协作；通信结构为去中心化（Decentralized）；反馈来源为环境与智能体交互；能力调整方式为记忆。  
  - **[Zhang et al., 2023c]（协作具身智能体 CoELA）**：环境接口为沙盒；画像方法为预定义；典型角色为机器人；通信范式为协作；通信结构为去中心化；反馈来源为环境与智能体交互；能力调整方式为记忆。  

- **问题求解 — 科学实验（Science Experiments）**  
  - **[Zheng et al., 2023]（优化 MOF 结构，Optimization of MOF）**：环境接口为物理（Physical）；画像方法为预定义；典型角色包括策略规划者（Strategy planners）、文献收集者（literature collector）、程序员（coder）；通信范式为协作；通信结构为中心化（Centralized）；反馈来源包括环境与人类；能力调整方式为记忆。  

- **问题求解 — 提升事实性与科学辩论（Improving Factuality & Science Debate）**  
  - **[Du et al., 2023]（提升事实性 Improving Factuality）**：环境接口为无（None）；画像方法为预定义；典型角色为智能体（Agents）；通信范式为辩论（Debate）；通信结构为去中心化；反馈来源为智能体交互；能力调整方式为记忆。  
  - **[Xiong et al., 2023]（检验互一致性 Examining, Inter-Consistency）**：环境接口为无；画像方法为预定义；典型角色包括正方（Proponent）、反方（Opponent）、裁判（Judge）；通信范式为辩论；通信结构包括中心化与去中心化；反馈来源为智能体交互；能力调整方式为记忆。  
  - **[Chan et al., 2023]（辩论评估者 Evaluators for debates）**：环境接口为无；画像方法为预定义；典型角色为智能体；通信范式为辩论；通信结构包括中心化与去中心化；反馈来源为智能体交互；能力调整方式为记忆。  
  - **[Tang et al., 2023]（多智能体辅助医疗 Multi-Agents for Medication）**：环境接口为无；画像方法为预定义；典型角色包括心内科（Cardiology）、外科（Surgery）专家；通信范式为辩论与协作；通信结构包括中心化与去中心化；反馈来源为智能体交互；能力调整方式为记忆。  

- **世界模拟（World Simulation） — 社会（Society）**  
  - **[Park et al., 2023]（小规模社区 Modest Community, 25 persons）**：环境接口为沙盒；画像方法为模型生成（Model-generated）；典型角色包括药房（Pharmacy）、店主（shopkeeper）等；未明确给出通信范式与结构（“ - ”）；反馈来源为环境与智能体交互；能力调整方式为记忆。  
  - **[Park et al., 2022]（在线社区 Online community, 1000 persons）**：环境接口为无；画像方法为预定义与模型生成；典型角色包括露营（Camping）、钓鱼（fishing）等兴趣角色；未明确给出通信范式与结构；反馈来源为智能体交互；能力调整方式包括动态生成（Dynamic Generation）。  
  - **[Gao et al., 2023a]（情感传播 Emotion propagation）**：环境接口为无；画像方法为预定义与模型生成；典型角色为真实世界用户（Real-world user）；未明确给出通信范式与结构；反馈来源为智能体交互；能力调整方式为记忆。  
  - **[Kaiya et al., 2023]（实时社交交互 Real-time social interactions）**：环境接口为沙盒；画像方法为预定义；典型角色为真实世界用户；未明确给出通信范式与结构；反馈来源为环境与智能体交互；能力调整方式为记忆。  
  - **[Li et al., 2023a]（意见动力学 Opinion dynamics）**：环境接口为无；画像方法为预定义；典型角色为 NIN, NINL, NIL 等不同类型代理；未明确给出通信范式与结构；反馈来源为智能体交互；能力调整方式为记忆。  

- **世界模拟 — 游戏（Gaming）**  
  - **[Xu et al., 2023b; Xu et al., 2023c]（狼人杀 WereWolf）**：环境接口为沙盒；画像方法为预定义；典型角色包括预言家（Seer）、狼人（werewolf）、村民（villager）；通信范式包括协作、辩论与竞争（Cooperative, Debate, Competitive）；通信结构为去中心化；反馈来源为环境与智能体交互；能力调整方式为记忆。  
  - **[Light et al., 2023a; Wang et al., 2023c]（阿瓦隆 Avalon）**：环境接口为沙盒；画像方法为预定义；典型角色包括侍从（Servant）、梅林（Merlin）、刺客（Assassin）；通信范式包括协作、辩论与竞争；通信结构为去中心化；反馈来源为环境与智能体交互；能力调整方式为记忆。  
  - **[Mukobi et al., 2023]（Welfare Diplomacy）**：环境接口为沙盒；画像方法为预定义；典型角色为国家（Countries）；通信范式包括协作与竞争；通信结构为去中心化；反馈来源为环境与智能体交互；能力调整方式为记忆。  

- **世界模拟 — 人类行为与心理（Human behavior Simulation & Psychology）**  
  - **[Aher et al., 2023]（人类行为模拟 Human behavior Simulation）**：环境接口为沙盒；画像方法为预定义；典型角色为人类（Humans）；未明确给出通信范式与结构；反馈来源为智能体交互；能力调整方式为记忆。  
  - **[Zhang et al., 2023d]（协作探索 Collaboration Exploring）**：环境接口为无；画像方法为预定义；典型角色为智能体；通信范式包括协作与辩论；通信结构为去中心化；反馈来源为智能体交互；能力调整方式为记忆。  

- **世界模拟 — 经济（Economy）**  
  - **[Li et al., 2023e]（宏观经济模拟 Macroeconomic simulation）**：环境接口为无；画像方法为预定义与模型生成；典型角色为劳动力（Labor）；通信范式为协作；通信结构为去中心化；反馈来源为智能体交互；能力调整方式为记忆。  
  - **[Anonymous, 2023]（信息市场 Information Marketplaces）**：环境接口为沙盒；画像方法为预定义与数据驱动（Data-Derived）；典型角色为买家（Buyer）；通信范式包括协作与竞争；通信结构为去中心化；反馈来源为环境与智能体交互；能力调整方式为记忆。  
  - **[Li et al., 2023g]（改进金融交易 Improving financial trading）**：环境接口为物理环境；画像方法为预定义；典型角色为交易员（Trader）；通信范式为辩论；通信结构为去中心化；反馈来源为环境与智能体交互；能力调整方式为记忆。  
  - **[Zhao et al., 2023]（经济理论 Economic theories）**：环境接口为沙盒；画像方法为预定义与模型生成；典型角色包括餐馆（Restaurant）、顾客（Customer）；通信范式为竞争（Competitive）；通信结构为去中心化；反馈来源为环境与智能体交互；能力调整方式包括记忆与自我演化。  

- **世界模拟 — 推荐系统（Recommender Systems）**  
  - **[Zhang et al., 2023a]（模拟用户行为 Simulating user behaviors）**：环境接口为沙盒；画像方法为数据驱动；典型角色为 MovieLens-1M 数据集中的用户；未明确给出通信范式与结构；反馈来源为环境；能力调整方式为记忆。  
  - **[Zhang et al., 2023e]（模拟用户-项目交互 Simulating user-item interactions）**：环境接口为沙盒；画像方法为预定义与数据驱动；典型角色包括用户智能体（User Agents）、项目智能体（Item Agents）；通信范式为协作；通信结构为去中心化；反馈来源为环境与智能体交互；能力调整方式为记忆。  

- **世界模拟 — 政策制定与公共健康（Policy Making & Public Health）**  
  - **[Xiao et al., 2023]（公共管理 Public Administration）**：环境接口为无；画像方法为预定义；典型角色为居民（Residents）；通信范式为协作；通信结构为去中心化；反馈来源为智能体交互；能力调整方式为记忆。  
  - **[Hua et al., 2023]（战争模拟 War Simulation）**：环境接口为无；画像方法为预定义；典型角色为国家；通信范式为竞争；通信结构为去中心化；反馈来源为智能体交互；能力调整方式为记忆。  
  - **[Ghaffarzadegan et al., 2023]（疫情下的人类行为 Human Behaviors to epidemics）**：环境接口为沙盒；画像方法为预定义与模型生成；典型角色为从众特质（Conformity traits）；通信范式为协作；通信结构为去中心化；反馈来源为环境与智能体交互；能力调整方式为记忆。  
  - **[Williams et al., 2023]（公共健康 Public health）**：环境接口为沙盒；画像方法为预定义与模型生成；典型角色为 18 至 64 岁成年人（Adults aged 18 to 64）；通信范式为协作；通信结构为去中心化；反馈来源为环境与智能体交互；能力调整方式包括记忆与动态生成。  


### 4.1 用于问题求解的 LLM-MA（LLM-MA for Problem Solving）

使用 LLM-MA 进行问题求解的主要动机在于利用具有专业化专长的智能体群体能力。这些智能体各自作为“个体”行事，通过协作来有效解决复杂问题，例如软件开发、具身智能体（Embodied Agents）、科学实验以及科学辩论等。下面依次介绍这些应用示例。

#### 4.1.1 软件开发（Software Development）

软件开发是一个复杂的工程活动，需要产品经理、程序员和测试人员等多种角色的协同工作，因此 LLM-MA 系统通常会模拟这些不同角色，并通过协作应对这一复杂挑战。沿着软件开发的瀑布式或标准操作流程（Standardized Operating Procedures, SOPs）工作流，智能体之间的通信结构通常是分层（layered）的。智能体一般会与代码解释器、其他智能体或人类进行交互，以迭代地改进生成的代码。[Li et al., 2023b] 首先提出了一个简单的角色扮演（role-play）智能体框架，利用两个角色之间的互动在仅有一句用户指令的条件下实现自治编程，并为研究具有“认知”过程的交流式智能体提供了启发。[Dong et al., 2023b] 使 LLM 在软件开发子任务上扮演不同的“专家”角色，自主协作完成代码生成。此外，[Qian et al., 2023] 提出了一个用于软件开发的端到端框架，利用多个智能体完成软件开发任务，而没有显式引入高级的人类团队协作经验。[Hong et al., 2023] 则首次将人类工作流经验融入其中，以实现更可控且可验证的性能表现；它将 SOP 编码进提示中，以增强结构化的协调能力。[Huang et al., 2023a] 进一步深入研究基于多智能体的编程问题，关注如何在代码片段生成与有效的测试用例生成、执行与优化之间取得平衡。
#### 4.1.2 具身智能体（Embodied Agents）

大多数具身智能体应用天然需要多台机器人协同工作，以执行复杂的现实世界规划与操作任务，例如在具有异构机器人能力的仓储管理任务中。因此，LLM-MA 可以用于建模具备不同能力的机器人，并让它们协作解决现实世界中的物理任务。[Dasgupta et al., 2023] 首先探索了将 LLM 作为具身智能体动作规划器的潜力。[Mandi et al., 2023] 提出了 RoCo，这是一种用于多机器人协作的新方法，利用 LLM 进行高层通信与低层路径规划。每个机械臂都配备一个 LLM，并与逆运动学与碰撞检测模块协同工作。实验结果表明，RoCo 在协作任务中展现出良好的适应性与成功率。[Zhang et al., 2023c] 提出了 CoELA（Cooperative Embodied Language Agent，即协作具身语言智能体），在 LLM-MA 设置中负责管理讨论与任务规划。

这一具有挑战性的设置具备以下特征：去中心化控制、复杂的局部观察、高昂的通信成本以及多目标长时程任务。[Chen et al., 2023d] 研究了在大量机器人参与的场景下的通信挑战，因为为每个机器人分配一个 LLM 代价高昂且由于上下文长度限制而不现实。该研究比较了四种通信框架——中心化、去中心化以及两种混合模型——以评估它们在协调复杂多智能体任务中的效果。[Yu et al., 2023] 提出了 Co-NavGPT，用于多机器人协作视觉目标导航，将 LLM 作为全局规划器，为每个机器人分配前沿（frontier）子目标。[Chen et al., 2023b] 提出了一个基于 LLM 的共识寻求框架，可作为多机器人聚集任务中的协作规划器。

#### 4.1.3 科学实验（Science Experiments）

类似于多智能体作为不同专家协作解决软件开发与具身智能体问题，多智能体也可以组成一个科学团队来执行科学实验。与前述应用的一个重要差异在于此处人类监督（human oversight）至关重要，因为科学实验成本高昂且 LLM 智能体存在幻觉（hallucination）问题。人类专家处于这些智能体的中心位置，负责汇总与处理智能体提供的信息，并向智能体给出反馈。[Zheng et al., 2023] 利用多个基于 LLM 的智能体，每个智能体专注于科学实验中的特定任务，包括策略规划、文献检索、编码、机器人操作以及实验器具设计。所有这些智能体都与人类进行交互，共同协作以优化复杂材料的合成过程。

#### 4.1.4 科学辩论（Science Debate）

LLM-MA 可以被设置为科学辩论场景，使智能体彼此辩论，从而在大规模多任务语言理解（Massive Multitask Language Understanding, MMLU）[Hendrycks et al., 2020]、数学问题 [Cobbe et al., 2021] 以及 StrategyQA [Geva et al., 2021] 等任务上增强集体推理能力。其核心思想是，每个智能体首先对问题给出自己的分析，随后通过联合辩论过程，在多轮辩论后收敛到一个统一的、共识性的答案。[Du et al., 2023] 在六种不同推理与事实准确性任务上利用多智能体辩论过程，并展示了 LLM-MA 辩论有助于提升事实性（factuality）。[Xiong et al., 2023] 关注常识推理任务，设计了一个与现实场景相契合的三阶段辩论流程，包括公平辩论、错位辩论和圆桌辩论。该工作还分析了不同 LLM 之间的“互一致性（inter-consistency）”，并指出辩论能够提升这种互一致性。[Tang et al., 2023] 同样利用多个基于 LLM 的智能体作为不同领域的专家，对医疗报告进行协作讨论，以达成关于医疗诊断的共识。

### 4.2 用于世界模拟的 LLM-MA（LLM-MA for World Simulation）

LLM-MA 的另一大主流应用场景是世界模拟。该领域研究发展迅速，涵盖社会科学、游戏、心理学、经济学、政策制定等多个方向。在世界模拟中采用 LLM-MA 的关键原因在于其卓越的角色扮演能力，这对于在模拟世界中逼真地呈现多种角色与观点至关重要。

世界模拟项目中的环境通常被精心设计，以反映所要模拟的具体场景，并通过设计多样化画像的智能体来与该环境相匹配。不同于强调智能体协作的问题求解系统，世界模拟系统在智能体管理与通信方式上更加多样化，以体现真实世界交互的复杂性与多样性。下面我们将探讨在不同领域中开展的模拟研究。

#### 4.2.1 社会模拟（Societal Simulation）

在社会模拟中，LLM-MA 用于模拟社会行为，旨在探索潜在的社会动态与传播过程、检验社会科学理论，并在虚拟空间和社区中生成逼真的社会现象 [Park et al., 2023]。基于 LLM 的能力，具有独特画像的智能体进行广泛交流，从而生成丰富的行为数据，供深入的社会科学分析使用。

社会模拟的规模随时间逐步扩展，从较小、紧密的环境发展到更大、更复杂的设置。最初的工作 [Park et al., 2023] 在一个类“模拟人生”（the sims）的交互式沙盒环境中引入生成式智能体，允许终端用户以自然语言与一个由 25 个智能体组成的小型社区进行交互。与此同时，[Park et al., 2022] 提出了 Social Simulacra，构建了一个由 1000 个角色组成的模拟社区。该系统从设计者提出的社区设想（包括其目标、规则以及成员画像）出发，对其进行模拟，生成诸如发帖、回复乃至反社会行为等多种行为。在此基础上，[Gao et al., 2023a] 进一步构建了规模更大的网络，分别包含 8,563 个和 17,945 个智能体，用于围绕“性别歧视（Gender Discrimination）”和“核能（Nuclear Energy）”等话题模拟社交网络。这一演进过程展示了近期研究中模拟环境在复杂度和规模上的不断提升。近期如 [Chen et al., 2023b; Kaiya et al., 2023; Li et al., 2023a; Li et al., 2023f; Ziems et al., 2023] 等工作进一步凸显了多智能体系统复杂性的演化、LLM 对社交网络的影响及其在社会科学研究中的融合。

#### 4.2.2 游戏（Gaming）

LLM-MA 非常适合用于构建模拟的游戏环境，使智能体可以在游戏中扮演各种角色。这项技术可以构建受控、可扩展且动态的环境，逼真地模拟人类交互，非常适合用于检验一系列博弈论假设 [Mao et al., 2023; Xu et al., 2023b; Gong et al., 2023]。大多数由 LLM-MA 模拟的游戏高度依赖自然语言通信，在不同游戏设定下提供沙盒环境，用于探索或检验涉及推理、合作、说服、欺骗、领导力等方面的博弈论假设。

[Akata et al., 2023] 借助行为博弈论分析 LLM 在交互式社会情境中的行为，尤其是其在迭代囚徒困境（iterated Prisoner’s Dilemma）和“性别之战”（Battle of the Sexes）等游戏中的表现。[Xu et al., 2023b] 提出了一个基于 ChatArena 库 [Wu et al., 2023b] 的框架，让 LLM 参与“狼人杀（Werewolf）”等交流游戏，并通过对历史交流记录的检索与反思以及链式思考（Chain-of-Thought）机制 [Wei et al., 2022] 改进表现。[Light et al., 2023b] 探索了 LLM 智能体在“抵抗组织：阿瓦隆”（Resistance Avalon）游戏中的潜力，提出了 AVALONBENCH 这一综合游戏环境与基准，用于推动更先进的 LLM 和多智能体框架的发展。[Wang et al., 2023c] 也关注 LLM 智能体在阿瓦隆游戏中应对虚假信息的能力，提出了递归思考（Recursive Contemplation, ReCon）框架，以增强 LLM 识别和对抗欺骗性信息的能力。[Xu et al., 2023c] 则提出了一个将 LLM 与强化学习（Reinforcement Learning, RL）结合的框架，以构建用于“狼人杀”游戏的策略性语言智能体；其提出了一种新方法，在动作与状态集合不是预先定义、而是以自然语言形式出现的设置下使用 RL 策略。

[Mukobi et al., 2023] 设计了“Welfare Diplomacy”，这是零和棋盘游戏“外交（Diplomacy）”的一个总和博弈变体，要求玩家在军事征服与国内福利之间寻求平衡。该工作还提供了一个开源基准，旨在帮助提升多智能体 AI 系统的合作能力。在此基础上，[Li et al., 2023c] 在多智能体协作文本游戏中测试智能体的“心智理论（Theory of Mind, ToM）”，即对他人隐藏心理状态进行推理的能力，这对于人类社会交互、合作与沟通至关重要。[Fan et al., 2023] 对 LLM 作为理性玩家的能力进行了系统评估，并指出 LLM 智能体的不足之处：即便在显式给出游戏过程的情况下，智能体在行动时仍可能忽略或改变精细的信念。

#### 4.2.3 心理学（Psychology）

在心理学模拟研究中，与社会模拟类似，会使用多个智能体来模拟具有不同特质和思维过程的人类。然而，与社会模拟不同的是，其中一种心理学研究路径是直接在这些智能体上实施心理学实验，通过统计方法观察并分析其多样化行为。在这种方式下，每个智能体独立运行，不与其他智能体交互，本质上代表不同的“个体”。

另一种路径则更接近社会模拟：多个智能体彼此交互、交流，在此基础上运用心理学理论来理解和分析涌现出的行为模式。该方法有助于研究人际动态和群体行为，为“个体心理特征如何影响集体行为”这一问题提供洞见。[Ma et al., 2023] 探讨了使用基于 LLM 的会话智能体为心理健康提供支持的心理学影响与结果，并强调需要从心理学视角谨慎评估在心理健康应用中使用 LLM 智能体。[Kovaˇc et al., 2023] 提出了名为 SocialAI school 的工具，用于创建模拟社会交互的交互式环境；该工具借鉴发展心理学（developmental psychology），以理解智能体如何习得、展现并发展诸如联合注意、沟通与文化学习等社会技能。[Zhang et al., 2023d] 探索了具有不同特质与思维模式的 LLM 智能体如何模拟人类式的社会行为，例如从众和多数原则。这种将心理学引入对多智能体协作机制的理解，提供了一个研究 LLM-MA 系统的新视角。

[Aher et al., 2023] 提出了“图灵实验（Turing Experiments）”，用于评估大语言模型在多大程度上可以模拟人类行为的不同方面。图灵实验以问答形式复现心理学、经济学和社会学中的经典实验和现象，以模拟实验条件。作者还设计了一个提示模板，通过更改姓名来模拟多个不同个体的回答。通过用 LLM 模拟多种类型的个体，他们展示了更大的模型在复现人类行为方面更为贴近，但也揭示了一种“超准确偏差（hyper-accuracy distortion）”，尤其是在知识类任务中。

#### 4.2.4 经济（Economy）

LLM-MA 被用于模拟经济与金融交易环境，主要是因为它可以作为人类的隐式计算模型。在这些模拟中，智能体会被赋予禀赋和信息，并设置预定义偏好，从而在经济与金融情境中探索其行为。这类似于经济学家在某些经济理论中对“经济人（homo economicus）”的建模，即将人描述为追求自身利益最大化的理性人 [Horton, 2023]。

已有多项研究展示了 LLM-MA 在宏观经济活动、信息市场、金融交易以及虚拟小镇模拟等经济场景中的多样应用。智能体在协作或辩论式、去中心化的环境中进行交互。[Li et al., 2023e] 在宏观经济模拟中采用 LLM，利用基于 prompt 工程驱动的智能体来模拟类人决策过程，相对于基于规则或其他 AI 智能体的方法，提高了经济模拟的真实感。[Anonymous, 2023] 在信息市场中重新审视“买方检验悖论（buyer’s inspection paradox）”，表明当智能体在付费前可临时访问信息时，其决策质量和答案质量均得到提升。[Li et al., 2023g] 提出了用于金融交易的 LLM-MA 框架，强调分层记忆系统、辩论机制和个性化交易角色，以增强决策的鲁棒性。[Zhao et al., 2023] 利用基于 LLM 的智能体模拟一个由餐馆与顾客智能体构成的虚拟小镇，其结论与社会学和经济学理论相吻合。总体而言，这些研究共同展示了将 LLM 应用于多种经济模拟场景的广泛应用与进展。

#### 4.2.5 推荐系统（Recommender Systems）

LLM-MA 在推荐系统中的使用与其在心理学中的应用类似，因为这两个领域的研究都需要考虑包括认知过程和人格特质在内的外在与内在人类因素 [Lex and Schedl, 2022]。一种使用 LLM-MA 的方式是：直接向多个具有不同特质的基于 LLM 的智能体介绍项目（items），并统计不同智能体的偏好；另一种方式则把用户和项目都视为智能体，将用户-项目的通信视为交互，以模拟偏好的传播。

为缩小推荐系统中离线指标与真实世界表现之间的差距，Agent4Rec [Zhang et al., 2023a] 提出了一个基于 LLM-MA 的模拟平台。作者使用 MovieLens-1M 数据集初始化 1000 个生成式智能体，以模拟推荐环境中的复杂用户交互。Agent4Rec 表明，LLM-MA 可以有效模拟真实用户偏好与行为，提供对“过滤气泡效应（filter bubble effect）”等现象的洞见，并有助于揭示推荐任务中的因果关系。在 Agent4Rec 中，智能体用于模拟用户，彼此之间并不直接通信。与此不同，[Zhang et al., 2023e] 将用户和项目都视为智能体，通过整体优化它们的表示来反映并适应真实世界中交互差异。该工作强调模拟用户-项目交互，并在智能体间传播偏好，从而捕捉协同过滤的本质。

#### 4.2.6 政策制定（Policy Making）

与游戏和经济场景中的模拟类似，政策制定需要强大的决策能力来应对真实且动态的复杂问题。LLM-MA 可以通过模拟虚拟政府或模拟不同政策对各类群体的影响来开展政策制定研究。这类模拟为理解政策的制定过程及其潜在影响提供了有价值的洞见，有助于政策制定者理解并预判自身决策的后果 [Farmer and Axtell, 2022]。

[Xiao et al., 2023] 的研究聚焦于模拟一个镇级水污染危机。作者构造了一个位于岛上的小镇，包括不同智能体构成的人口结构以及镇长和顾问角色。在水污染危机模拟中，该工作对虚拟政府实体如何应对这一公共管理挑战以及在危机中社会网络中的信息传播进行了深入分析。[Hua et al., 2023] 则提出 WarAgent，用于模拟关键历史冲突，并为冲突解决与理解提供洞见，具有在预防未来国际冲突方面的潜在应用。

#### 4.2.7 疾病传播模拟（Disease Propagation Simulation）

利用 LLM-MA 的社会模拟能力，也可以开展疾病传播模拟。最新研究 [Williams et al., 2023] 探讨了在疾病传播模拟中使用 LLM-MA。该研究通过多种模拟展示，基于 LLM 的智能体可以准确模拟人类在疾病暴发时的应对行为，包括在病例数激增时进行自我隔离与居家隔离。智能体群体的整体行为很好地再现了疫情多波传播的复杂模式，并最终趋于地方性流行的稳定状态。令人印象深刻的是，智能体的行为有助于抑制疫情曲线。[Ghaffarzadegan et al., 2023] 同样讨论了疫情传播模拟，并将模拟分解为两部分：一是代表病毒传播信息或过程的**机制模型（Mechanistic Model）**，二是代表智能体在面对病毒时决策过程的**决策模型（Decision-Making Model）**。

---

## 5 实现工具与资源（Implementation Tools and Resources）

### 5.1 多智能体框架（Multi-Agents Framework）

我们对三个开源多智能体框架进行详细介绍：MetaGPT [Hong et al., 2023]、CAMEL [Li et al., 2023b] 和 Autogen [Wu et al., 2023a]。这些框架都利用语言模型来解决复杂任务，并重点支持多智能体协作，但在方法与应用上各有不同。

MetaGPT 旨在将人类工作流过程嵌入语言模型智能体的运行之中，从而缓解复杂任务中经常出现的幻觉问题。它通过将标准操作流程（Standard Operating Procedures, SOPs）编码到系统中，并采用装配线式（assembly line）的方式，将不同角色分配给不同的智能体。

CAMEL（Communicative Agent Framework）面向促成智能体之间的自治协作。它使用名为“inception prompting”的新技术，引导会话式智能体完成与人类目标一致的任务。该框架同时也是生成和研究对话数据的工具，帮助研究者理解交流式智能体的行为与交互模式。

AutoGen 是一个通用框架，可用于构建基于语言模型的应用。其独特之处在于高度定制化的能力：开发者可以同时利用自然语言和代码来编程智能体及其交互方式。这种灵活性使其可用于从编码、数学等技术领域到娱乐等面向消费者场景在内的多个领域。

近期，[Chen et al., 2023c; Chen et al., 2023a] 提出了用于动态多智能体协作的框架，而 [Zhou et al., 2023a; Li et al., 2023h; Xie et al., 2023] 提供了构建自治智能体的平台与库，强调其在任务求解和社会模拟中的适应性。

### 5.2 数据集与基准（Datasets and Benchmarks）

我们在表 2 中总结了 LLM-MA 研究中常用的数据集与基准。可以观察到，不同的研究应用往往采用不同的数据集与基准。在问题求解场景中，多数数据集与基准用于通过多智能体协作或辩论来评估系统的规划与推理能力；在世界模拟场景中，数据集与基准则用于评估模拟世界与现实世界的一致性，或用于分析不同智能体的行为。

**表 2：** LLM-MA 研究中常用的数据集与基准。“/” 表示尚未提供公开数据链接。

- **问题求解（Problem Solving）**  
  - **软件开发（Software Development）**：HumanEval、MBPP、SoftwareDev 数据集，均在 [Hong et al., 2023] 中使用；对应的“Link” 表示论文中提供了数据或基准的获取链接。  
  - **具身智能体 / Embodied AI**：RoCoBench（[Mandi et al., 2023]）、Communicative Watch-And-Help（C-WAH, [Zhang et al., 2023c]）、ThreeDWorld Multi-Agent Transport（TDW-MAT, [Zhang et al., 2023c]）以及 HM3D v0.2（[Yu et al., 2023]），相应工作中均给出了数据或环境链接。  
  - **科学辩论（Science Debate）**：MMLU、MedQA、PubMedQA（均由 [Tang et al., 2023] 使用）、GSM8K（[Du et al., 2023]）、StrategyQA（[Xiong et al., 2023]）、Chess Move Validity（[Du et al., 2023]），对应条目中的“Link” 表示可以通过论文或官方页面获取数据或评测接口。  

- **世界模拟（World Simulation）**  
  - **社会（Society）**：SOTOPIA（[Zhou et al., 2023b]），以及围绕“性别歧视（Gender Discrimination）”与“核能（Nuclear Energy）”主题构建的社交网络数据集（[Gao et al., 2023a]）。原表中 “/” 表示这些数据暂未提供统一的公共下载链接。  
  - **游戏（Gaming）**：Werewolf（[Xu et al., 2023b]）、Avalon（[Light et al., 2023b]）、Welfare Diplomacy（[Mukobi et al., 2023]）、Overcooked-AI 环境中的布局（Layout in the Overcooked-AI environment, [Agashe et al., 2023]）、Chameleon 与 Undercover（[Xu et al., 2023a]）。其中 Chameleon 与 Undercover 对应原表中的“Link”，表示提供了获取方式或代码仓库。  
  - **心理学（Psychology）**：Ultimatum Game TE、Garden Path TE、Wisdom of Crowds TE，均在 [Aher et al., 2023] 中使用，并给出了相应链接。  
  - **推荐系统（Recommender System）**：MovieLens-1M（[Zhang et al., 2023a]）和 Amazon review 数据集（[Zhang et al., 2023e]）。MovieLens-1M 对应“Link”，而 Amazon review 数据集在原表中记为“/”，表示未直接给出统一下载链接。  
  - **政策制定（Policy Making）**：Board Connectivity Evaluation（[Hua et al., 2023]），原表中标注为“Link”，说明论文中提供了该评测设置或相关数据的获取方式。  

在某些研究应用中（例如科学实验团队操作与经济建模），仍然缺乏全面的基准。此类基准的构建将极大提升评估 LLM-MA 在这些复杂而动态领域中的成功与适用性的能力。

---

## 6 挑战与机遇（Challenges and Opportunities）

关于 LLM-MA 框架与应用的研究正快速发展，催生了众多挑战与机遇。我们识别出若干关键挑战以及未来潜在研究方向。

### 6.1 向多模态环境拓展（Advancing into Multi-Modal Environment）

目前大部分关于 LLM-MA 的工作仍集中在基于文本的环境上，在处理与生成文本方面表现突出。然而，在多模态设置中仍存在明显不足，在多模态环境中，智能体需要处理来自多种感官输入的数据，并生成多种输出，例如图像、音频、视频以及物理动作。将 LLM 集成到多模态环境中面临额外挑战，例如处理多样数据类型，并使智能体能够理解并响应不止文本的信息。

### 6.2 解决幻觉问题（Addressing Hallucination）

幻觉问题是 LLM 和单智能体 LLM 系统中的一项重大挑战，指模型生成在事实层面不正确的文本 [Huang et al., 2023b]。然而，在多智能体环境中，这一问题的复杂度进一步提高。在这类场景中，一个智能体的幻觉可能产生级联效应：由于多智能体系统具有高度互联性，某个智能体产生的错误信息可能被其他智能体接受并进一步传播。因此，在 LLM-MA 中检测和缓解幻觉不仅是一项关键任务，而且带来了一系列独特挑战。它既涉及纠正单个智能体层面的不准确性，也涉及管理智能体之间的信息流，以防止这些不准确性在整个系统中扩散。

### 6.3 获取集体智能（Acquiring Collective Intelligence）

在传统多智能体系统中，智能体通常使用强化学习从离线训练数据集中学习。然而，LLM-MA 系统主要从即时反馈中学习，例如来自环境或人类的交互反馈，正如我们在第 3 节讨论的那样。这种学习方式需要一个可靠的交互环境，而为许多任务设计这样一个环境并不容易，从而限制了 LLM-MA 系统的可扩展性。

此外，当前研究中的主流方法多通过记忆（Memory）和自我演化（Self-Evolution）技术基于反馈调整智能体。虽然这些方法对单个智能体而言是有效的，但并未充分挖掘智能体网络的潜在集体智能，因为它们往往独立地调整智能体，忽视了由协调的多智能体交互所产生的协同效应。因此，如何联合调整多个智能体，并实现最优的集体智能，仍然是 LLM-MA 面临的关键挑战。

### 6.4 扩展 LLM-MA 系统规模（Scaling Up LLM-MA Systems）

LLM-MA 系统由大量基于 LLM 的个体智能体构成，这在智能体数量方面带来了显著的可扩展性挑战。从计算复杂度角度看，每个基于 LLM 的智能体（通常构建在如 GPT-4 等大型语言模型之上）都需要大量计算资源与内存。随着 LLM-MA 系统中智能体数量的增加，资源需求会显著增长。在计算资源有限的场景下，开发这类 LLM-MA 系统将变得极具挑战。

此外，随着 LLM-MA 系统中智能体数量的增长，在高效的智能体协调、通信以及多智能体的“规模律”理解等方面，会出现更多复杂性与研究机会。举例来说，当智能体数量增多时，如何确保有效协调与通信变得尤为复杂。正如 [Dibia, 2023] 所强调的，设计先进的“智能体编排（Agents Orchestration）”方法变得愈发重要，此类方法旨在优化智能体工作流、为不同智能体分配合适任务以及规划智能体间的通信模式（例如为智能体间通信设置约束）。高效的智能体编排可以促进智能体之间的协同运作，最大限度减少冲突与冗余。

此外，探索并刻画多智能体系统在规模扩大时其行为与效率所遵循的“规模律（scaling laws）”仍是重要的研究方向。这些问题凸显了需要提出创新解决方案，以在保证 LLM-MA 系统有效性的同时提升其资源使用效率。

### 6.5 评估与基准（Evaluation and Benchmarks）

我们在表 2 中总结了当前用于 LLM-MA 的数据集与基准。这只是一个起点，并远称不上全面。我们在评估 LLM-MA 系统并对其进行基准对比时识别出两个主要挑战。首先，如 [Xu et al., 2023a] 所讨论的，大部分现有研究侧重于在相对狭窄的场景中评估单个智能体的理解与推理能力，这往往忽视了多智能体系统中更广泛且更复杂的涌现行为。其次，在诸如科学实验团队操作、经济分析和疾病传播模拟等多个研究领域，综合性基准的开发明显不足，这对准确评估和对比 LLM-MA 系统在这些关键领域中的全部能力构成了障碍。

### 6.6 应用及更远的未来（Applications and Beyond）

LLM-MA 系统的潜力远超其当前应用，未来有望在金融、教育、医疗、环境科学、城市规划等领域的高级计算问题求解中发挥重要作用。正如我们所讨论的，LLM-MA 系统具备处理复杂问题和模拟现实世界各个方面的能力。尽管当前 LLM 的角色扮演能力仍有局限，但随着 LLM 技术的持续进步，前景光明。预计未来将出现更为复杂的方法论、应用场景、数据集与基准，并针对多种研究领域进行定制。

此外，还可以从多种理论视角探索 LLM-MA 系统，例如认知科学（Cognitive Science）[Sumers et al., 2023]、符号人工智能（Symbolic Artificial Intelligence）、控制论（Cybernetics）、复杂系统（Complex Systems）以及集体智能（Collective Intelligence）等。这种多元视角有望帮助我们更全面地理解这一快速演化领域，并推动其产生更具创新性的应用。

---

## 7 结论（Conclusion）

基于 LLM 的多智能体系统展现出了令人鼓舞的集体智能，并迅速引起研究者越来越多的关注。在本综述中，我们首先通过从多个方面对 LLM-MA 系统进行定位、区分与关联，对其发展进行了系统性回顾，这些方面包括：智能体-环境接口、LLM 对智能体的刻画方式、用于管理智能体通信的策略以及智能体能力获取的范式。我们还总结了面向问题求解和世界模拟的 LLM-MA 应用。

通过进一步梳理 LLM-MA 研究中常用的数据集和基准，并讨论未来的挑战与机遇，我们希望本综述能够成为跨学科研究者的一份有用资源，激励未来研究继续探索基于 LLM 的多智能体系统的潜力。

---

## 参考文献（References）

[Agashe et al., 2023] Saaket Agashe, Yue Fan, and Xin Eric Wang. Evaluating multi-agent coordination abilities in large language models, 2023.

[Aher et al., 2023] Gati Aher, Rosa I. Arriaga, and Adam Tauman Kalai. Using large language models to simulate multiple humans and replicate human subject studies, 2023.

[Akata et al., 2023] Elif Akata, Lion Schulz, Julian Coda-Forno, Seong Joon Oh, Matthias Bethge, and Eric Schulz. Playing repeated games with large language models. arXiv preprint arXiv:2305.16867, 2023.

[Anonymous, 2023] Anonymous. Rethinking the buyer’s inspection paradox in information markets with language agents. In Submitted to The Twelfth International Conference on Learning Representations, 2023. under review.

[Chan et al., 2023] Chi-Min Chan, Weize Chen, Yusheng Su, Jianxuan Yu, Wei Xue, Shanghang Zhang, Jie Fu, and Zhiyuan Liu. Chateval: Towards better LLM-based evaluators through multi-agent debate, 2023.

[Chen et al., 2023a] Guangyao Chen, Siwei Dong, Yu Shu, Ge Zhang, Jaward Sesay, Börje F. Karlsson, Jie Fu, and Yemin Shi. AutoAgents: A framework for automatic agent generation. arXiv preprint arXiv:2309.17288, 2023.

[Chen et al., 2023b] Huaben Chen, Wenkang Ji, Lufeng Xu, and Shiyu Zhao. Multi-agent consensus seeking via large language models. arXiv preprint arXiv:2310.20151, 2023.

[Chen et al., 2023c] Weize Chen, Yusheng Su, Jingwei Zuo, Cheng Yang, Chenfei Yuan, Chen Qian, Chi-Min Chan, Yujia Qin, Yaxi Lu, Ruobing Xie, et al. AgentVerse: Facilitating multi-agent collaboration and exploring emergent behaviors in agents. arXiv preprint arXiv:2308.10848, 2023.

[Chen et al., 2023d] Yongchao Chen, Jacob Arkin, Yang Zhang, Nicholas Roy, and Chuchu Fan. Scalable multi-robot collaboration with large language models: Centralized or decentralized systems? arXiv preprint arXiv:2309.15943, 2023.

[Cobbe et al., 2021] Karl Cobbe, Vineet Kosaraju, Mohammad Bavarian, Mark Chen, Heewoo Jun, Lukasz Kaiser, Matthias Plappert, Jerry Tworek, Jacob Hilton, Reiichiro Nakano, et al. Training verifiers to solve math word problems. arXiv preprint arXiv:2110.14168, 2021.

[Dasgupta et al., 2023] Ishita Dasgupta, Christine Kaeser-Chen, Kenneth Marino, Arun Ahuja, Sheila Babayan, Felix Hill, and Rob Fergus. Collaborating with language models for embodied reasoning. arXiv preprint arXiv:2302.00763, 2023.

[Dibia, 2023] Victor Dibia. Multi-agent LLM applications — a review of current research, tools, and challenges. https://newsletter.victordibia.com/p/multi-agent-llm-applications-a-review, 2023.

[Dong et al., 2023a] Qingxiu Dong, Lei Li, Damai Dai, Ce Zheng, Zhiyong Wu, Baobao Chang, Xu Sun, Jingjing Xu, Lei Li, and Zhifang Sui. A survey on in-context learning, 2023.

[Dong et al., 2023b] Yihong Dong, Xue Jiang, Zhi Jin, and Ge Li. Self-collaboration code generation via ChatGPT, 2023.

[Du et al., 2023] Yilun Du, Shuang Li, Antonio Torralba, Joshua B. Tenenbaum, and Igor Mordatch. Improving factuality and reasoning in language models through multi-agent debate, 2023.

[Fan et al., 2023] Caoyun Fan, Jindou Chen, Yaohui Jin, and Hao He. Can large language models serve as rational players in game theory? A systematic analysis. arXiv preprint arXiv:2312.05488, 2023.

[Farmer and Axtell, 2022] J. Doyne Farmer and Robert L. Axtell. Agent-Based Modeling in Economics and Finance: Past, Present, and Future. INET Oxford Working Papers 2022-10, Institute for New Economic Thinking at the Oxford Martin School, University of Oxford, June 2022.

[Gao et al., 2023a] Chen Gao, Xiaochong Lan, Zhihong Lu, Jinzhu Mao, Jinghua Piao, Huandong Wang, Depeng Jin, and Yong Li. S3: Social-network simulation system with large language model-empowered agents. arXiv preprint arXiv:2307.14984, 2023.

[Gao et al., 2023b] Yunfan Gao, Yun Xiong, Xinyu Gao, Kangxiang Jia, Jinliu Pan, Yuxi Bi, Yi Dai, Jiawei Sun, and Haofen Wang. Retrieval-augmented generation for large language models: A survey. arXiv preprint arXiv:2312.10997, 2023.

[Geva et al., 2021] Mor Geva, Daniel Khashabi, Elad Segal, Tushar Khot, Dan Roth, and Jonathan Berant. Did Aristotle use a laptop? A question answering benchmark with implicit reasoning strategies, 2021.

[Ghaffarzadegan et al., 2023] Navid Ghaffarzadegan, Aritra Majumdar, Ross Williams, and Niyousha Hosseinichimeh. Generative agent-based modeling: Unveiling social system dynamics through coupling mechanistic models with generative artificial intelligence. arXiv preprint arXiv:2309.11456, 2023.

[Gong et al., 2023] Ran Gong, Qiuyuan Huang, Xiaojian Ma, Hoi Vo, Zane Durante, Yusuke Noda, Zilong Zheng, Song-Chun Zhu, Demetri Terzopoulos, Li Fei-Fei, et al. MindAgent: Emergent gaming interaction. arXiv preprint arXiv:2309.09971, 2023.

[Guo et al., 2023] Taicheng Guo, Kehan Guo, Zhengwen Liang, Zhichun Guo, Nitesh V. Chawla, Olaf Wiest, Xiangliang Zhang, et al. What indeed can GPT models do in chemistry? A comprehensive benchmark on eight tasks. arXiv preprint arXiv:2305.18365, 2023.

[Hendrycks et al., 2020] Dan Hendrycks, Collin Burns, Steven Basart, Andy Zou, Mantas Mazeika, Dawn Song, and Jacob Steinhardt. Measuring massive multitask language understanding. arXiv preprint arXiv:2009.03300, 2020.

[Hong et al., 2023] Sirui Hong, Xiawu Zheng, Jonathan Chen, Yuheng Cheng, Ceyao Zhang, Zili Wang, Steven Ka Shing Yau, Zijuan Lin, Liyang Zhou, Chenyu Ran, et al. MetaGPT: Meta programming for multi-agent collaborative framework. arXiv preprint arXiv:2308.00352, 2023.

[Horton, 2023] John J. Horton. Large language models as simulated economic agents: What can we learn from homo silicus? Technical report, National Bureau of Economic Research, 2023.

[Hua et al., 2023] Wenyue Hua, Lizhou Fan, Lingyao Li, Kai Mei, Jianchao Ji, Yingqiang Ge, Libby Hemphill, and Yongfeng Zhang. War and peace (WarAgent): Large language model-based multi-agent simulation of world wars, 2023.

[Huang et al., 2023a] Dong Huang, Qingwen Bu, Jie M. Zhang, Michael Luck, and Heming Cui. AgentCoder: Multi-agent-based code generation with iterative testing and optimisation, 2023.

[Huang et al., 2023b] Lei Huang, Weijiang Yu, Weitao Ma, Weihong Zhong, Zhangyin Feng, Haotian Wang, Qianglong Chen, Weihua Peng, Xiaocheng Feng, Bing Qin, et al. A survey on hallucination in large language models: Principles, taxonomy, challenges, and open questions. arXiv preprint arXiv:2311.05232, 2023.

[Kaiya et al., 2023] Zhao Kaiya, Michelangelo Naim, Jovana Kondic, Manuel Cortes, Jiaxin Ge, Shuying Luo, Guangyu Robert Yang, and Andrew Ahn. Lyfe agents: Generative agents for low-cost real-time social interactions. arXiv preprint arXiv:2310.02172, 2023.

[Khot et al., 2023] Tushar Khot, Harsh Trivedi, Matthew Finlayson, Yao Fu, Kyle Richardson, Peter Clark, and Ashish Sabharwal. Decomposed prompting: A modular approach for solving complex tasks, 2023.

[Kovač et al., 2023] Grgur Kovač, Rémy Portelas, Peter Ford Dominey, and Pierre-Yves Oudeyer. The SocialAI school: Insights from developmental psychology towards artificial socio-cultural agents. arXiv preprint arXiv:2307.07871, 2023.

[Lewis et al., 2021] Patrick Lewis, Ethan Perez, Aleksandra Piktus, Fabio Petroni, Vladimir Karpukhin, Naman Goyal, Heinrich Küttler, Mike Lewis, Wen-tau Yih, Tim Rocktäschel, Sebastian Riedel, and Douwe Kiela. Retrieval-augmented generation for knowledge-intensive NLP tasks, 2021.

[Lex and Schedl, 2022] Elisabeth Lex and Markus Schedl. Psychology-informed recommender systems: A human-centric perspective on recommender systems. In Proceedings of the 2022 Conference on Human Information Interaction and Retrieval, CHIIR ’22, page 367–368, New York, NY, USA, 2022. Association for Computing Machinery.

[Li et al., 2023a] Chao Li, Xing Su, Chao Fan, Haoying Han, Cong Xue, and Chunmo Zheng. Quantifying the impact of large language models on collective opinion dynamics. arXiv preprint arXiv:2308.03313, 2023.

[Li et al., 2023b] Guohao Li, Hasan Abed Al Kader Hammoud, Hani Itani, Dmitrii Khizbullin, and Bernard Ghanem. CAMEL: Communicative agents for “mind” exploration of large scale language model society. arXiv preprint arXiv:2303.17760, 2023.

[Li et al., 2023c] Huao Li, Yu Quan Chong, Simon Stepputtis, Joseph Campbell, Dana Hughes, Michael Lewis, and Katia Sycara. Theory of mind for multi-agent collaboration via large language models, 2023.

[Li et al., 2023d] Minghao Li, Yingxiu Zhao, Bowen Yu, Feifan Song, Hangyu Li, Haiyang Yu, Zhoujun Li, Fei Huang, and Yongbin Li. API-Bank: A comprehensive benchmark for tool-augmented LLMs, 2023.

[Li et al., 2023e] Nian Li, Chen Gao, Yong Li, and Qingmin Liao. Large language model-empowered agents for simulating macroeconomic activities, 2023.

[Li et al., 2023f] Siyu Li, Jin Yang, and Kui Zhao. Are you in a masquerade? Exploring the behavior and impact of large language model driven social bots in online social networks. arXiv preprint arXiv:2307.10337, 2023.

[Li et al., 2023g] Yang Li, Yangyang Yu, Haohang Li, Zhi Chen, and Khaldoun Khashanah. TradingGPT: Multi-agent system with layered memory and distinct characters for enhanced financial trading performance, 2023.

[Li et al., 2023h] Yuan Li, Yixuan Zhang, and Lichao Sun. MetaAgents: Simulating interactions of human behaviors for LLM-based task-oriented coordination via collaborative generative agents. arXiv preprint arXiv:2310.06500, 2023.

[Liang et al., 2023] Zhenwen Liang, Wenhao Yu, Tanmay Rajpurohit, Peter Clark, Xiangliang Zhang, and Ashwin Kaylan. Let GPT be a math tutor: Teaching math word problem solvers with customized exercise generation. arXiv preprint arXiv:2305.14386, 2023.

[Light et al., 2023a] Jonathan Light, Min Cai, Sheng Shen, and Ziniu Hu. AvalonBench: Evaluating LLMs playing the game of Avalon, 2023.

[Light et al., 2023b] Jonathan Light, Min Cai, Sheng Shen, and Ziniu Hu. From text to tactic: Evaluating LLMs playing the game of Avalon. arXiv preprint arXiv:2310.05036, 2023.

[Liu et al., 2023] Zijun Liu, Yanzhe Zhang, Peng Li, Yang Liu, and Diyi Yang. Dynamic LLM-agent network: An LLM-agent collaboration framework with agent team optimization. arXiv preprint arXiv:2310.02170, 2023.

[Ma et al., 2023] Zilin Ma, Yiyang Mei, and Zhaoyuan Su. Understanding the benefits and challenges of using large language model-based conversational agents for mental well-being support. arXiv preprint arXiv:2307.15810, 2023.

[Mandi et al., 2023] Zhao Mandi, Shreeya Jain, and Shuran Song. RoCo: Dialectic multi-robot collaboration with large language models. arXiv preprint arXiv:2307.04738, 2023.

[Mao et al., 2023] Shaoguang Mao, Yuzhe Cai, Yan Xia, Wenshan Wu, Xun Wang, Fengyi Wang, Tao Ge, and Furu Wei. ALympics: Language agents meet game theory. arXiv preprint arXiv:2311.03220, 2023.

[Moura, 2023] João Moura. CrewAI. https://github.com/joaomdmoura/crewAI, 2023.

[Mukobi et al., 2023] Gabriel Mukobi, Hannah Erlebach, Niklas Lauffer, Lewis Hammond, Alan Chan, and Jesse Clifton. Welfare Diplomacy: Benchmarking language model cooperation. arXiv preprint arXiv:2310.08901, 2023.

[Nascimento et al., 2023] Nathalia Nascimento, Paulo Alencar, and Donald Cowan. Self-adaptive large language model (LLM)-based multiagent systems. In 2023 IEEE International Conference on Autonomic Computing and Self-Organizing Systems Companion (ACSOS-C), pages 104–109. IEEE, 2023.

[Park et al., 2022] Joon Sung Park, Lindsay Popowski, Carrie Cai, Meredith Ringel Morris, Percy Liang, and Michael S. Bernstein. Social Simulacra: Creating populated prototypes for social computing systems. In Proceedings of the 35th Annual ACM Symposium on User Interface Software and Technology, pages 1–18, 2022.

[Park et al., 2023] Joon Sung Park, Joseph C. O’Brien, Carrie J. Cai, Meredith Ringel Morris, Percy Liang, and Michael S. Bernstein. Generative agents: Interactive simulacra of human behavior. arXiv preprint arXiv:2304.03442, 2023.

[Qian et al., 2023] Chen Qian, Xin Cong, Wei Liu, Cheng Yang, Weize Chen, Yusheng Su, Yufan Dang, Jiahao Li, Juyuan Xu, Dahai Li, Zhiyuan Liu, and Maosong Sun. Communicative agents for software development, 2023.

[Ruan et al., 2023] Jingqing Ruan, Yihong Chen, Bin Zhang, Zhiwei Xu, Tianpeng Bao, Guoqing Du, Shiwei Shi, Hangyu Mao, Ziyue Li, Xingyu Zeng, and Rui Zhao. TPTU: Large language model-based AI agents for task planning and tool usage, 2023.

[Russell and Norvig, 2009] Stuart Russell and Peter Norvig. Artificial Intelligence: A Modern Approach. Prentice Hall Press, USA, 3rd edition, 2009.

[Shinn et al., 2023] Noah Shinn, Federico Cassano, Edward Berman, Ashwin Gopinath, Karthik Narasimhan, and Shunyu Yao. Reflexion: Language agents with verbal reinforcement learning, 2023.

[Sumers et al., 2023] Theodore R. Sumers, Shunyu Yao, Karthik Narasimhan, and Thomas L. Griffiths. Cognitive architectures for language agents. arXiv preprint arXiv:2309.02427, 2023.

[Tang et al., 2023] Xiangru Tang, Anni Zou, Zhuosheng Zhang, Yilun Zhao, Xingyao Zhang, Arman Cohan, and Mark Gerstein. MedAgents: Large language models as collaborators for zero-shot medical reasoning, 2023.

[Wang et al., 2021] Zijie J. Wang, Dongjin Choi, Shenyu Xu, and Diyi Yang. Putting humans in the natural language processing loop: A survey, 2021.

[Wang et al., 2023a] Kuan Wang, Yadong Lu, Michael Santacroce, Yeyun Gong, Chao Zhang, and Yelong Shen. Adapting LLM agents through communication, 2023.

[Wang et al., 2023b] Lei Wang, Chen Ma, Xueyang Feng, Zeyu Zhang, Hao Yang, Jingsen Zhang, Zhiyuan Chen, Jiakai Tang, Xu Chen, Yankai Lin, Wayne Xin Zhao, Zhewei Wei, and Ji-Rong Wen. A survey on large language model based autonomous agents, 2023.

[Wang et al., 2023c] Shenzhi Wang, Chang Liu, Zilong Zheng, Siyuan Qi, Shuo Chen, Qisen Yang, Andrew Zhao, Chaofei Wang, Shiji Song, and Gao Huang. Avalon’s game of thoughts: Battle against deception through recursive contemplation. arXiv preprint arXiv:2310.01320, 2023.

[Wei et al., 2022] Jason Wei, Xuezhi Wang, Dale Schuurmans, Maarten Bosma, Fei Xia, Ed Chi, Quoc V. Le, Denny Zhou, et al. Chain-of-thought prompting elicits reasoning in large language models. Advances in Neural Information Processing Systems, 35:24824–24837, 2022.

[Weng, 2023] Lilian Weng. LLM powered autonomous agents. https://lilianweng.github.io/posts/2023-06-23-agent/, 2023.

[Williams et al., 2023] Ross Williams, Niyousha Hosseinichimeh, Aritra Majumdar, and Navid Ghaffarzadegan. Epidemic modeling with generative agents. arXiv preprint arXiv:2307.04986, 2023.

[Wooldridge and Jennings, 1995] Michael Wooldridge and Nicholas R. Jennings. Intelligent agents: Theory and practice. The Knowledge Engineering Review, 10:115–152, 1995.

[Wu et al., 2023a] Qingyun Wu, Gagan Bansal, Jieyu Zhang, Yiran Wu, Shaokun Zhang, Erkang Zhu, Beibin Li, Li Jiang, Xiaoyun Zhang, and Chi Wang. AutoGen: Enabling next-gen LLM applications via multi-agent conversation framework. arXiv preprint arXiv:2308.08155, 2023.

[Wu et al., 2023b] Yuxiang Wu, Zhengyao Jiang, Akbir Khan, Yao Fu, Laura Ruis, Edward Grefenstette, and Tim Rocktäschel. ChatArena: Multi-agent language game environments for large language models. GitHub repository, 2023.

[Xi et al., 2023] Zhiheng Xi, Wenxiang Chen, Xin Guo, Wei He, Yiwen Ding, Boyang Hong, Ming Zhang, Junzhe Wang, Senjie Jin, Enyu Zhou, Rui Zheng, Xiaoran Fan, Xiao Wang, Limao Xiong, Yuhao Zhou, Weiran Wang, Changhao Jiang, Yicheng Zou, Xiangyang Liu, Zhangyue Yin, Shihan Dou, Rongxiang Weng, Wensen Cheng, Qi Zhang, Wenjuan Qin, Yongyan Zheng, Xipeng Qiu, Xuanjing Huang, and Tao Gui. The rise and potential of large language model based agents: A survey, 2023.

[Xiao et al., 2023] Bushi Xiao, Ziyuan Yin, and Zixuan Shan. Simulating public administration crisis: A novel generative agent-based simulation system to lower technology barriers in social science research. arXiv preprint arXiv:2311.06957, 2023.

[Xie et al., 2023] Tianbao Xie, Fan Zhou, Zhoujun Cheng, Peng Shi, Luoxuan Weng, Yitao Liu, Toh Jing Hua, Junning Zhao, Qian Liu, Che Liu, et al. OpenAgents: An open platform for language agents in the wild. arXiv preprint arXiv:2310.10634, 2023.

[Xiong et al., 2023] Kai Xiong, Xiao Ding, Yixin Cao, Ting Liu, and Bing Qin. Examining inter-consistency of large language models collaboration: An in-depth analysis via debate, 2023.

[Xu et al., 2023a] Lin Xu, Zhiyuan Hu, Daquan Zhou, Hongyu Ren, Zhen Dong, Kurt Keutzer, See Kiong Ng, and Jiashi Feng. MAGIC: Investigation of large language model powered multi-agent in cognition, adaptability, rationality and collaboration, 2023.

[Xu et al., 2023b] Yuzhuang Xu, Shuo Wang, Peng Li, Fuwen Luo, Xiaolong Wang, Weidong Liu, and Yang Liu. Exploring large language models for communication games: An empirical study on Werewolf. arXiv preprint arXiv:2309.04658, 2023.

[Xu et al., 2023c] Zelai Xu, Chao Yu, Fei Fang, Yu Wang, and Yi Wu. Language agents with reinforcement learning for strategic play in the Werewolf game. arXiv preprint arXiv:2310.18940, 2023.

[Yao et al., 2023] Shunyu Yao, Dian Yu, Jeffrey Zhao, Izhak Shafran, Thomas L. Griffiths, Yuan Cao, and Karthik Narasimhan. Tree of thoughts: Deliberate problem solving with large language models, 2023.

[Yu et al., 2023] Bangguo Yu, Hamidreza Kasaei, and Ming Cao. Co-NavGPT: Multi-robot cooperative visual semantic navigation using large language models, 2023.

[Zhang et al., 2023a] An Zhang, Leheng Sheng, Yuxin Chen, Hao Li, Yang Deng, Xiang Wang, and Tat-Seng Chua. On generative agents in recommendation, 2023.

[Zhang et al., 2023b] Ceyao Zhang, Kaijie Yang, Siyi Hu, Zihao Wang, Guanghe Li, Yihang Sun, Cheng Zhang, Zhaowei Zhang, Anji Liu, Song-Chun Zhu, et al. ProAgent: Building proactive cooperative AI with large language models. arXiv preprint arXiv:2308.11339, 2023.

[Zhang et al., 2023c] Hongxin Zhang, Weihua Du, Jiaming Shan, Qinhong Zhou, Yilun Du, Joshua B. Tenenbaum, Tianmin Shu, and Chuang Gan. Building cooperative embodied agents modularly with large language models. arXiv preprint arXiv:2307.02485, 2023.

[Zhang et al., 2023d] Jintian Zhang, Xin Xu, and Shumin Deng. Exploring collaboration mechanisms for LLM agents: A social psychology view, 2023.

[Zhang et al., 2023e] Junjie Zhang, Yupeng Hou, Ruobing Xie, Wenqi Sun, Julian McAuley, Wayne Xin Zhao, Leyu Lin, and Ji-Rong Wen. AgentCF: Collaborative learning with autonomous language agents for recommender systems, 2023.

[Zhao et al., 2023] Qinlin Zhao, Jindong Wang, Yixuan Zhang, Yiqiao Jin, Kaijie Zhu, Hao Chen, and Xing Xie. CompeteAI: Understanding the competition behaviors in large language model-based agents, 2023.

[Zheng et al., 2023] Zhiling Zheng, Oufan Zhang, Ha L. Nguyen, Nakul Rampal, Ali H. Alawadhi, Zichao Rong, Teresa Head-Gordon, Christian Borgs, Jennifer T. Chayes, and Omar M. Yaghi. ChatGPT research group for optimizing the crystallinity of MOFs and COFs. ACS Central Science, 9(11):2161–2170, 2023.

[Zhou et al., 2023a] Wangchunshu Zhou, Yuchen Eleanor Jiang, Long Li, Jialong Wu, Tiannan Wang, Shi Qiu, Jintian Zhang, Jing Chen, Ruipu Wu, Shuai Wang, et al. Agents: An open-source framework for autonomous language agents. arXiv preprint arXiv:2309.07870, 2023.

[Zhou et al., 2023b] Xuhui Zhou, Hao Zhu, Leena Mathur, Ruohong Zhang, Haofei Yu, Zhengyang Qi, Louis-Philippe Morency, Yonatan Bisk, Daniel Fried, Graham Neubig, and Maarten Sap. Sotopia: Interactive evaluation for social intelligence in language agents, 2023.

[Ziems et al., 2023] Caleb Ziems, Omar Shaikh, Zhehao Zhang, William Held, Jiaao Chen, and Diyi Yang. Can large language models transform computational social science? Computational Linguistics, pages 1–53, 2023.
