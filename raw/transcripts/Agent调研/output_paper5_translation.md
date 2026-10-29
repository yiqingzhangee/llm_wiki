## 大语言模型驱动的 GUI 智能体：综述

###### Chaoyun Zhang, Shilin He, Jiaxu Qian, Bowen Li, Liqun Li, Si Qin, Yu Kang, Minghua Ma, Guyue Liu, Qingwei Lin, Saravan Rajmohan, Dongmei Zhang, Qi Zhang

**摘要——图形用户界面（Graphical User Interface, GUI）长期以来一直是人机交互的核心，为用户提供了一种直观且以视觉为主的方式来访问和操作数字系统。传统上，GUI 交互的自动化主要依赖脚本式或规则式的方法，这类方法虽然在固定工作流下效果良好，但在应对动态、真实世界应用时所需的灵活性和适应性方面却明显不足。大语言模型（Large Language Model, LLM），尤其是多模态模型的出现，为 GUI 自动化开辟了新的时代。它们在自然语言理解、代码生成、任务泛化和视觉处理方面展现出卓越能力。这为新一代“以 LLM 为大脑”的 GUI 智能体铺平了道路，使其能够解释复杂的 GUI 元素，并基于自然语言指令自主执行操作。这类智能体代表了一种范式转变，使用户能够通过简单的对话式指令完成复杂的多步任务。其应用跨越网页导航、移动应用交互以及桌面自动化，提供了一种变革性的用户体验，从根本上革新人们与软件交互的方式。这个新兴领域正在迅速发展，在学术界和工业界都取得了显著进展。

为对这一趋势提供系统性的理解，本文对以 LLM 为大脑的 GUI 智能体进行了全面综述，探讨其历史演进、核心组件以及高级技术。我们围绕多个关键研究问题展开讨论，例如：现有的 GUI 智能体框架有哪些、如何收集和利用数据来训练专门的 GUI 智能体、如何为 GUI 任务构建定制的大动作模型（Large Action Model, LAM）、以及评估其有效性所需的评测指标和基准。同时，我们还讨论了由这些智能体驱动的新兴应用。通过细致的分析，本综述识别出关键研究空白，并提出了未来发展的路线图。通过整合基础知识与最新进展，本工作旨在帮助研究者和实践者共同应对挑战、释放以 LLM 为大脑的 GUI 智能体的全部潜力。我们期望，本综述既可以作为构建 LLM 驱动 GUI 智能体的实践“菜谱”，也能成为推动该快速发展领域研究的权威参考。

本综述所收录的论文集合将托管于并定期更新在 GitHub 仓库：
[https://github.com/vyokky/LLM-Brained-GUI-Agents-Survey](https://github.com/vyokky/LLM-Brained-GUI-Agents-Survey)。此外，我们还提供了一个可搜索的网页，便于访问和检索：
[https://aka.ms/gui-agent](https://aka.ms/gui-agent)。

**索引词——大语言模型（Large Language Model, LLM），图形用户界面（Graphical User Interface, GUI），AI 智能体（AI Agent），自动化，人机交互（Human-Computer Interaction, HCI）**

###### ✦

###### 1 引言（INTRODUCTION）

图形用户界面（GUI）一直是人机交互的基石，从根本上改变了用户在数字系统中的导航与操作方式 [1]。为了让计算更直观、更易用，GUI 取代了命令行界面（Command-Line Interface, CLI）[2]，构建了一个以视觉为主、对用户友好的环境。通过使用图标、按钮、窗口和菜单，GUI 使更广泛的用户群体能够通过点击、输入和手势等简单操作与计算机交互。这一转变显著提升了计算资源的可及性，使得非技术用户也能够有效使用复杂系统。然而，在涉及重复性或多步交互的工作流中，相比于 CLI，GUI 往往为了可用性而牺牲了效率 [3]。

尽管 GUI 在可用性上实现了革命性提升，但其主要面向人类视觉交互的设计，也给自动化带来了显著挑战。GUI 布局在形式上的多样性、动态性及平台相关性，使得开发能够适应不同环境的灵活且智能的自动化工具变得困难。早期对 GUI 交互自动化的尝试主要依赖脚本式或规则式方法 [4], [5]。虽然这些方法在预定义工作流中效果良好，但其本质上具有较窄的适用范围，主要聚焦于软件测试和机器人流程自动化（Robotic Process Automation, RPA）等任务 [6]。它们的刚性导致需要频繁进行人工更新以适配新任务、GUI 布局变更或不断演进的工作流，从而限制了可扩展性和通用性。此外，这些方法缺乏支持动态、人类般交互所需的复杂度，因此在复杂或不可预期的场景中难以适用。

大语言模型（LLM）[1] [8], [9]，尤其是融合了多模态能力的模型 [10]，已经成为 GUI 自动化领域的“游戏规则改变者”，重新定义了智能体与图形用户界面交互的方式。从 ChatGPT [11] 等模型开始，LLM 在自然语言理解、代码生成以及跨多种任务的泛化能力方面展现出非凡的性能 [8], [12]–[14]。视觉语言模型（Visual Language Model, VLM）的引入进一步扩展了这些能力，使模型能够处理视觉数据，例如 GUI 的复杂布局 [15]。这一演进弥合了语言理解与视觉理解之间的鸿沟，使智能体能够以更接近人类的方式、更具适应性地与 GUI 交互。通过利用这些进步，LLM 和 VLM 展现出变革性的潜力，使智能体能够在复杂数字环境中导航、以动态方式执行任务，并从根本上革新 GUI 自动化领域。

**1.1 以 LLM 为大脑的 GUI 智能体的动机（Motivation for LLM-Brained GUI Agents）**

当把大语言模型作为“脑”嵌入其中时，LLM 驱动的 GUI 自动化催生了一类新的智能体：它们能够理解用户的自然语言请求，分析 GUI 屏幕及其元素，并自主执行相应操作。重要的是，这些能力的实现并不依赖复杂的平台相关脚本或预定义工作流。这类智能体被称为**“以 LLM 为大脑的 GUI 智能体（LLM-brained GUI agents）”**，可以形式化地定义为：

_在 GUI 环境中运行的智能体，以大语言模型作为其核心推理与认知引擎，用于以灵活、自适应的方式生成、规划并执行动作。_

这一范式代表了 GUI 自动化领域的一次变革性飞跃，促进了跨多种平台的动态、人类般交互。它使得可以构建智能、可适应的系统，这些系统能够进行推理、实时做出决策，并灵活响应不断变化的任务与环境。我们在图 1 中对这一高层概念进行了示意说明。

传统的 GUI 自动化通常受限于预定义规则，或仅针对特定任务进行设计，从而限制了其在动态环境和多样化应用中的适应能力。相比之下，LLM 驱动的 GUI 智能体通过将自然语言理解、视觉识别和决策能力整合进统一框架，实现了范式转变。这使得智能体可以在广泛的用例中实现任务泛化，从而重塑任务自动化，并显著提升人机交互的直观性和效率。此外，与新兴的纯 API（Application Programming Interface）驱动智能体趋势不同——后者依赖的 API 并不总是暴露或可访问——GUI 智能体利用了图形界面的通用性。GUI 为控制大多数软件应用提供了一种通用机制，使智能体能够以非侵入式方式运行，而无需访问内部 API。这一能力不仅拓宽了 GUI 智能体的适用范围，也使外部开发者可以在各类既有软件之上构建高级功能，跨越多种平台和生态系统。综合来看，这些创新使 GUI 智能体成为未来智能自动化的一项通用且具变革性的技术。

这一新范式使用户能够通过对话式指令来控制通用软件系统 [16]。通过降低多步 GUI 操作的认知负担，LLM 驱动的智能体让复杂系统对非技术用户更加友好，并简化了跨不同领域的工作流。典型示例包括用于网页导航的 SeeAct [17]、用于移动交互的 AppAgent [18]，以及面向 Windows 操作系统应用的 UFO [19]。这些智能体类似于一种“虚拟助手”（virtual assistant）[20]，颇有钢铁侠中 J.A.R.V.I.S. 的意味——一个能够理解用户目标并在应用间自主执行操作的直观、可适应系统。一个由 AI 驱动的操作系统这一未来概念正迅速成为现实：该系统可以以高度流畅与精确的方式执行跨应用任务 [21], [22]。

以 LLM 驱动的 GUI 智能体的实际应用已经开始涌现。例如，Microsoft Power Automate 利用 LLM 来简化低代码/无代码的自动化[2]，让用户几乎无需技术背景即可在 Microsoft 应用之间设计工作流。集成于生产力软件中的 AI 助手，如 Microsoft Copilot[3]，正在进一步打通自然语言指令与具体应用操作之间的鸿沟。此外，LLM 驱动的智能体在辅助无障碍访问方面也展现出潜力 [23]，例如有望帮助视障用户更有效地导航 GUI，将自然语言命令转化为可执行的操作序列。这些发展都彰显了 LLM 驱动 GUI 智能体在多种应用中的及时性与变革潜力。

LLM 与 GUI 自动化的融合解决了人机交互领域长期存在的诸多挑战，同时也为智能 GUI 控制带来了新的机遇 [24]。这种融合引发了研究热潮，涵盖应用框架 [19]、数据采集 [25]、模型优化 [15] 以及评测基准 [26] 等多个方向。尽管已有诸多进展，关键挑战与局限依然存在，许多基础性问题仍有待深入探索。然而，对这一快速演进领域进行系统性综述的工作却相对缺乏，从而在整体理解上留下了重要空白。

**1.2 本综述的范围（Scope of the Survey）**

为弥补这一空白，本文对以 LLM 为大脑的 GUI 智能体进行了首批系统而全面的综述。我们回顾 GUI 智能体的历史演进，提供构建这类智能体的逐步指南，总结关键与高级技术，梳理与框架、数据和模型相关的代表性工具与研究，展示典型应用，并展望未来研究方向。具体而言，本综述旨在回答以下研究问题（Research Questions, RQs）：

1) RQ1：LLM 驱动 GUI 智能体的历史发展轨迹如何？（第 4 节）
2) RQ2：构成以 LLM 为大脑的 GUI 智能体基础的核心组件与高级技术有哪些？（第 5 节）
3) RQ3：面向 GUI 的 LLM 智能体有哪些主要框架，它们各自的特点是什么？（第 6 节）
4) RQ4：现有的数据集有哪些，如何收集全面的数据以训练面向 GUI 智能体的优化 LLM？（第 7 节）
5) RQ5：如何利用收集到的数据训练面向 GUI 智能体的专用大动作模型（Large Action Model, LAM），当前该领域有哪些领先模型？（第 8 节）
6) RQ6：评估 GUI 智能体的能力和性能时使用了哪些指标与基准？（第 9 节）
7) RQ7：LLM 驱动 GUI 智能体在实际场景中最重要的应用有哪些，它们是如何被落地与适配的？（第 10 节）
8) RQ8：在构建健壮且智能的 GUI 智能体时，目前面临的主要挑战、局限以及未来研究方向是什么？（第 11 节）

围绕上述问题，本综述旨在全面呈现当前研究现状，为构建以 LLM 为大脑的 GUI 智能体提供实践指南，识别关键研究缺口，并提出未来工作的方向。本文是系统审视 LLM 驱动 GUI 智能体领域的先行工作之一，从 LLM 发展、GUI 自动化以及人机交互等多个视角对该领域进行整合。

**1.3 综述结构（Survey Structure）**

本综述的结构安排如下，其整体结构示意如图 2 所示。第 2 节回顾与 LLM 智能体和 GUI 自动化相关的综述与评论性文献。第 3 节介绍大语言模型、LLM 智能体和 GUI 自动化的基础背景。第 4 节追溯 LLM 驱动 GUI 智能体的演进过程。第 5 节介绍 LLM 驱动 GUI 智能体中的关键组件与高级技术，可作为一份完整的构建指南。第 6 节给出代表性的 LLM 驱动 GUI 智能体框架。第 7 节讨论数据集采集及面向优化 GUI 智能体 LLM 的数据中心研究。第 8 节介绍用于 GUI 智能体的基础与优化模型。第 9 节梳理评估指标与基准。第 10 节探索真实世界中的应用与使用场景。最后，第 11 节分析当前的局限、挑战与潜在的未来方向，第 12 节对全文进行总结。为便于阅读，我们在表 1 中给出了缩略语列表。

###### 2 相关工作（RELATED WORK）

将大语言模型与 GUI 智能体相结合，是一个新兴且快速增长的研究领域。已有多篇相关综述与教程提供了基础性的洞见与指导。我们对 GUI 自动化与 LLM 智能体方面具有代表性的综述与专著进行简要回顾，因为这些主题与我们的研究重点密切相关并为之提供支撑。首先，我们概述若干关于 GUI 自动化、LLM 智能体及其融合的代表性综述与专著，如表 2 所总结。这些工作要么直接聚焦于 GUI 自动化和 LLM 驱动智能体中的一个或两个核心领域，要么虽然未直接针对该主题，但提供了有助于推动该领域发展的重要见解。

**2.1 GUI 自动化综述（Survey on GUI Automation）**

GUI 自动化具有悠久历史，并在工业界获得了广泛应用，尤其是在 GUI 测试 [27]–[29] 和面向任务自动化的机器人流程自动化（RPA）[6] 中 [42]。

Said 等人 [30] 对移动应用的 GUI 测试进行了概述，涵盖该领域的目标、方法与挑战。聚焦于 Android 应用，Li [31] 进一步收窄了研究范围；而 Oksanen 等人 [32] 则探讨了 Windows GUI 应用的自动测试技术——这是 GUI 智能体运行的重要平台。同样地，Moura 等人 [72] 回顾了 Web 应用的 GUI 测试，其中涉及多样化的工具、输入和方法学。Deshmukh 等人 [33] 讨论了用于提升用户体验的自动化 GUI 测试，这是 LLM 同样能够带来新能力的领域。现代 GUI 测试的基石之一是计算机视觉（Computer Vision, CV），它被用来解释 UI 元素并识别可操作控件 [34]。Yu 等人 [35] 调研了 CV 在移动 GUI 测试中的应用，强调了其重要性及面临的挑战。在 LLM 驱动的 GUI 智能体中，应用界面截图同样至关重要，它们是实现可靠任务理解与执行的关键输入。

另一方面，聚焦于自动化重复人工任务的 RPA 同样在相关流程中高度依赖 GUI 自动化。Syed 等人 [36] 对该领域进行了回顾，并总结了当代 RPA 的主题，指出了未来研究中的关键挑战。Chakraborti 等人 [37] 强调需要从传统脚本式 RPA 转向更加智能、自适应的范式，并对这一方向的进展进行了系统性综述。鉴于 RPA 在工业界的广泛应用，Enriquez 等人 [38] 和 Ribeiro 等人 [39] 从工业视角对该领域进行了调研，强调其重要性，并系统梳理了 RPA 方法、发展趋势和实践挑战。

GUI 测试 [40] 和 RPA [41] 在提升智能性与鲁棒性方面仍然面临重大挑战。LLM 驱动的 GUI 智能体有望在这些领域发挥变革性作用，提供更强的能力，并为解决这些长期存在的问题带来实质性的价值补充。

**2.2 LLM 智能体综述（Surveys on LLM Agents）**

大语言模型的兴起显著增强了智能体的能力 [43]，使其能够解决此前难以触及的复杂任务，尤其是涉及自然语言理解与代码生成的任务 [44]。这一进展激发了大量关于基于 LLM 的智能体的研究，这些智能体被设计用于广泛的应用场景 [45]。Xie 等人 [46] 和 Wang 等人 [47] 分别对 LLM 驱动智能体进行了全面综述，涵盖基础背景信息、组件拆解、分类体系以及各种应用。这些综述为理解 LLM 驱动智能体提供了重要参考，为进一步研究基于 LLM 的 GUI 智能体奠定了基础。Xie 等人 [59] 对多模态智能体进行了广泛综述，这类智能体除了文本外，还可以处理图像、视频和音频。这种多模态能力显著拓展了传统文本式智能体的能力边界 [60]。值得注意的是，大多数 GUI 智能体都属于这一类别，因为它们依赖图像输入（如截图）来有效解释并与图形界面交互。多智能体框架在 GUI 智能体设计中也被广泛采用，以增强其能力与可扩展性。Guo 等人 [48] 和 Han 等人 [49] 对这一领域的现状、挑战和未来方向进行了全面综述。Sun 等人 [50] 概述了利用强化学习提升多智能体 LLM 系统能力的最新方法，为增强其能力与适应性开辟了新的路径。这些综述为在 GUI 智能体框架中设计高效的多智能体系统提供了重要洞见与指导。

在数字环境领域，Wu 等人 [61] 对运行于移动环境中的 LLM 智能体进行了综述，涵盖了移动 GUI 智能体的关键方面。在更广的范围内，Wang 等人 [62] 调研了基础模型与 GUI 智能体融合的研究。Gao 等人的另一篇综述 [63] 则从跨数字平台的自主智能体角度展开，概述了其能力、挑战与应用。所有这些综述都凸显了该领域的新兴趋势。

在 LLM 智能体的各个组成部分方面，也有多篇综述提供了对 GUI 智能体特别重要的深度洞见。Huang 等人 [51] 探讨了 LLM 智能体中的规划机制，这是执行长期任务（GUI 自动化中常见）所必需的。Zhang 等人 [54] 研究了记忆机制，使智能体能够存储关键历史信息，从而有助于知识保持与决策。此外，Shen [13] 调研了 LLM 使用工具（如 API 与代码）与环境交互的方式，使得智能体的动作能够落地为现实世界中的影响。进一步地，Chang 等人 [55] 对 LLM 评估方法进行了全面综述，这对于保障 GUI 智能体的鲁棒性与安全性至关重要。两篇额外综述 [56], [58] 则对多模态 LLM 的基准与评测方法进行了系统梳理。评估同样可以形成反馈闭环，使智能体能够基于评估结果持续迭代改进。

综合来看，这些综述为 LLM 智能体的关键组件提供了宝贵资源和指导，并构成了构建基于 LLM 的 GUI 智能体的重要基础。

与现有综述相比，我们的工作在覆盖范围和时效性上都显著更为全面。我们汇总并综合了 500 余篇参考文献，覆盖基础模型、数据源、系统框架、基准、评测方法以及实际部署等广泛主题。现有综述往往聚焦于更窄的子方向或特定平台（例如 Web 或移动端），而本综述则从整体视角出发，覆盖从开发到部署的完整生命周期。除了叙述性总结外，我们还为每个子领域整理了汇总参考表，使读者能够快速按平台和研究主题对相关工作进行归类与查找——从而成为研究者与实践者的实用手册。此外，我们还引入基础背景材料并提出评估分类体系，使得本综述对新入门的读者也更加友好，从而弥补以往工作中常常假定读者具有较高先验知识的不足。

###### 3 背景（BACKGROUND）

以 LLM 为大脑的 GUI 智能体的发展建立在三个重要进展之上：（i）大语言模型（LLM）[8]，它们在自然语言理解和代码生成方面具备高级能力，构成了这类智能体的核心智能；（ii）配套的智能体架构与工具 [47]，用于扩展 LLM 的能力，弥合语言模型与物理环境之间的鸿沟，使其能够对现实世界产生实际影响；以及（iii）GUI 自动化该领域已经发展出一套稳健的工具、模型和方法论 [73]，这些都是 GUI 智能体实现功能所必需的。上述每一类组件都在以 LLM 为大脑的 GUI 智能体的出现过程中发挥了关键作用。接下来的小节将对这些领域进行简要概述，为后文讨论做好铺垫。

**3.1 大语言模型：基础与能力（Large Language Models: Foundations and Capabilities）**

语言模型的研究历史悠久而丰富 [74]，最早可追溯至早期的统计语言模型 [75] 和较小规模的神经网络结构 [76]。在这些基础概念之上，近期的进展主要聚焦于基于 Transformer 的大语言模型，如生成式预训练 Transformer（Generative Pre-trained Transformer, GPT）[77]。这类模型在大规模文本语料上进行预训练，并拥有显著增大的模型规模，验证了缩放定律，并在广泛的自然语言任务上展现出卓越能力。除了规模庞大之外，这些 LLM 还具备更强的语言理解与生成能力，并呈现出小规模语言模型中不存在的涌现属性 [78]。

早期的神经语言模型以循环神经网络（Recurrent Neural Network, RNN）[79] 和长短期记忆网络（Long Short-Term Memory, LSTM）[80] 等结构为基础，在性能与泛化方面都受到限制。Transformer 模型的提出——其基于注意力机制 [81]——标志着一个具有变革意义的里程碑，它奠定了目前几乎所有后续 LLM 所采用的基础架构。由此发展出了多种模型结构变体，包括编码器-only 模型（如 BERT [82]、RoBERTa [83]、ALBERT [84]）、解码器-only 模型（如 GPT-1 [85]、GPT-2 [86]），以及编码器-解码器模型（如 T5 [87]、BART [88]）。2022 年，基于 GPT-3.5 [89] 的 ChatGPT [11] 的发布，作为一款具有里程碑意义的 LLM，根本性地改变了人们对语言模型能力的认知。此后，GPT-4 [90]、LLaMA-3 [91]、Gemini [92] 等众多先进 LLM 先后涌现，将该领域推进到快速发展的阶段。如今的 LLM 具有高度的通用性，其中许多模型已经能够处理多模态数据，并执行从问答到代码生成等多种任务，使其在各类应用中成为不可或缺的工具 [93]–[96]。

LLM 的出现还带来了多项重要的高级属性，使其应用大为拓展，从而让此前具有挑战性的任务——例如基于自然语言驱动的 GUI 智能体——变得可行。这些进展包括：

1) 少样本学习（Few-Shot Learning）[77]：也被称为上下文学习（In-Context Learning, ICL）[97]，LLM 可以在推理阶段通过在提示（prompt）中给出少量示范样例来习得新任务，而无需重新训练。这一能力对于让 GUI 智能体以极小代价在不同环境间实现泛化至关重要。
2) 指令跟随（Instruction Following）[98]：在经过指令微调（instruction tuning）之后，LLM 在处理新任务时展现出显著的指令跟随能力和强泛化能力 [89]。这使得 LLM 能够有效理解针对 GUI 智能体的用户请求，并准确地遵循预定义目标。
3) 长期推理（Long-Term Reasoning）[99]：LLM 能够通过将复杂任务拆解为可管理的步骤来进行规划和求解，通常会采用链式思维（Chain-of-Thought, CoT）推理等技术 [100], [101]。这一能力对 GUI 智能体尤为重要，因为许多任务需要多步操作以及健壮的规划框架。
4) 代码生成与工具使用（Code Generation and Tool Utilization）[102]：LLM 在代码生成与工具（如 API [13]）调用方面表现出色。这一能力至关重要，因为代码与工具构成了 GUI 智能体与其环境交互的基础工具箱。
5) 多模态理解（Multimodal Comprehension）[10]：高级 LLM 可以在训练过程中融入图像等额外模态，演化为多模态模型。这种能力对 GUI 智能体尤为重要，因为它们必须将 GUI 截图作为图像进行解释，才能有效工作 [103]。

为了进一步增强 LLM 在 GUI 智能体场景中的专门化能力，研究者通常会利用领域数据（如用户请求、GUI 截图和动作序列）对这些模型进行微调，从而提升其定制化程度和效果。在第 8 节中，我们将深入讨论这些面向 GUI 智能体的高级定制模型，以及它们在与图形界面交互方面的特殊适配与能力提升。

**3.2 LLM 智能体：从语言到动作（LLM Agents: From Language to Action）**

传统的 AI 智能体往往聚焦于增强某项特定能力，例如符号推理，或在围棋、国际象棋等特定任务上达到卓越表现。相比之下，LLM 的出现通过为智能体提供自然语言接口，使其具备类人的决策能力，并能够在多样环境中执行广泛任务、采取具体行动，从而彻底改变了智能体的形态 [12], [47], [104], [105]。在 LLM 智能体中，若将 LLM 视作 GUI 智能体的“头脑（brain）”，则其配套组件便相当于“眼睛与双手”，使 LLM 能够感知环境状态，并将其生成的文本输出转换为可执行的具体步骤，从而产生实际效果 [46]。这些组件将 LLM 从被动的信息源转变为能够代表用户执行任务的交互式智能体，从而重新定义了 LLM 的角色：由纯文本生成模型演化为能够驱动动作并达成特定目标的系统。

在 GUI 智能体的场景中，智能体通常通过屏幕截图和控件树（widget tree）[106] 感知 GUI 状态，然后在环境中执行仿真人类操作的动作（例如鼠标点击、键盘输入、手机上的触控手势）。由于任务可能是长期的，往往需要有效的规划与任务分解，这也带来了独特挑战。因此，一个 LLM 驱动的 GUI 智能体通常需要具备多模态能力 [59]、健壮的规划系统 [51]、用于分析历史交互的记忆机制 [54]，以及用于与环境交互的专用工具集 [27]。我们将在第 5 节中详细讨论这些针对 GUI 智能体定制的设计。

**3.3 GUI 自动化：工具、技术与挑战（GUI Automation: Tools, Techniques, and Challenges）**

自计算机图形界面问世以来，GUI 自动化便一直是研究与实践中的关键方向。最初，GUI 自动化主要被开发用于提升软件测试效率，其关注点是模拟用户在图形应用中的操作，如点击、文本输入和导航，以验证功能正确性 [30]。早期的 GUI 自动化工具主要针对静态界面执行重复的测试用例 [28]。这些方法显著简化了质量保证流程，确保了一致性，并减少了人工测试时间。随着数字化解决方案需求的增长，GUI 自动化的应用已从测试扩展至其他领域，包括 RPA [6] 和人机交互（HCI）[107]。RPA 借助 GUI 自动化来复刻业务工作流中的人工操作，通过自动化例行任务来提升运营效率。同样，HCI 研究也利用 GUI 自动化来模拟用户行为，以便开展可用性评估和交互研究。在这两类场景中，自动化通过减少重复性任务、提升系统适应性 [108], [109]，显著提升了生产力与用户体验。

传统 GUI 自动化方法主要依赖脚本式与规则式框架 [4], [110]。脚本式自动化使用 Python、Java、JavaScript 等语言以编程方式控制 GUI 元素，这些脚本在界面上模拟用户动作，通常会使用 Selenium [111] 等工具进行 Web 自动化，或使用 AutoIt [112]、SikuliX [113] 等工具进行桌面应用的自动化。另一方面，规则式方法基于预定义的启发式规则，通过位置信息、颜色、文本标签等属性来检测并与特定 GUI 元素交互 [4]。虽然这类方法在可预测、静态工作流下效果良好 [114]，但在面对现代 GUI 的多样性时却显得力不从心：动态内容、自适应布局和用户驱动的界面变更，使得维持僵硬的规则式自动化变得困难 [115]。

计算机视觉（Computer Vision, CV）已成为解释 GUI 视觉要素的核心技术 [35], [116], [117]，使自动化工具即便在布局和设计发生变动时，也能识别并与屏幕上的元素交互。通过分析截图并定位感兴趣区域 [118]–[120]，CV 技术使 GUI 自动化系统能够检测并分类屏幕上的按钮、图标、文本框等元素。光学字符识别（Optical Character Recognition, OCR）进一步增强了这一能力，它可以从图像中提取文本内容，使自动化系统能够准确解释标签、错误信息和表单说明 [121]。目标检测模型则增加了系统的鲁棒性，使自动化智能体即便在视觉布局发生变化时也能定位 GUI 元素 [103]。通过引入 CV，GUI 自动化系统在动态环境中获得了更强的韧性与适应能力 [804]–[805]。

尽管如此，传统 GUI 自动化方法在应对当代界面的复杂性和多样性方面仍显不足。现代应用往往包含动态、自适应组件，仅依靠僵化的脚本或规则方法难以实现可靠自动化 [122], [123]。越来越多的界面需要具备上下文感知能力 [124]，例如处理屏幕上的文本、理解用户意图以及识别视觉线索。这些需求暴露出现有自动化框架的局限性，凸显了对能够实时适配并具备上下文敏感响应能力的更灵活方案的需求。

大语言模型为上述挑战提供了有前景的解决途径。凭借其对自然语言的理解能力、对上下文的解释能力以及生成自适应脚本的能力，LLM 能够实现更智能、更通用的 GUI 自动化 [125]。它们处理复杂指令并从上下文中学习的能力，有助于弥合静态规则式方法与现代 GUI 动态需求之间的鸿沟 [126]。通过将 LLM 与 GUI 智能体结合，这类系统可以基于当前界面状态即时生成脚本，从而实现传统方法难以达到的适应性与复杂度。LLM 与 GUI 智能体的结合为一种高级、以用户为中心的自动化范式铺平了道路，使系统能够灵活响应用户请求，并与复杂、不断演化的界面实现无缝交互。

###### 4 以 LLM 为大脑的 GUI 智能体的发展与演进（EVOLUTION AND PROGRESSION OF LLM-BRAINED GUI AGENTS）

“罗马不是一天建成的。”以 LLM 为大脑的 GUI 智能体的发展是一段循序渐进的历程，扎根于数十年的研究与技术进步。从最初的简单 GUI 测试脚本和规则式自动化框架开始，该领域在引入机器学习技术后不断演化，逐步形成更加智能和自适应的系统。LLM，尤其是多模态模型的出现，通过支持基于自然语言的人机交互，从根本上改变了用户与软件应用的交互方式，从而彻底重塑了 GUI 自动化。

如图 3 所示，在 2023 年及 LLM 出现之前，有关 GUI 智能体的工作在范围和能力上都相对有限。此后，基于 LLM 的方法迅速涌现，在包括 Web、移动端和桌面环境在内的各个平台上催生了众多重要进展。这种增长仍在持续，并不断推动该领域的创新。本节将带领读者回顾 GUI 智能体的发展轨迹，重点介绍推动该领域走到当前阶段的关键里程碑。

**4.1 早期自动化系统（Early Automation Systems）**

在 GUI 自动化的早期阶段，研究者主要依赖随机式、规则式和脚本式策略。尽管这些方法构成了基础，但在灵活性和适应性方面存在明显局限。

_4.1.1 随机式自动化（Random-Based Automation）_

随机式自动化利用在 GUI 中随机生成的动作序列，而不依赖特定算法或结构化模型，典型形式是猴子测试（monkey test）[127]。这种方法在 GUI 测试中被广泛用于通过探索不可预期的输入序列来发现潜在问题 [128]。虽然随机方法在识别边界情况和缺陷方面颇为有效，但也往往效率低下，因为其中包含大量冗余或无关的尝试。

_4.1.2 规则式自动化（Rule-Based Automation）_

规则式自动化通过预定义规则和逻辑来自动执行任务。2001 年，Memon 等人 [129] 提出了一种规划方法，通过一系列预定义算子将初始状态转化为目标状态，从而生成 GUI 测试用例。Hellmann 等人 [4]（2011）展示了规则式方法在探索性测试中的潜力，提升了缺陷检测能力。在 RPA 领域，SmartRPA [130]（2020）利用规则式处理来自动化例行任务，展示了规则在简化结构化流程方面的实用价值。

_4.1.3 脚本式自动化（Script-Based Automation）_

脚本式自动化依赖精细的脚本来管理 GUI 交互。例如，jRapture [5]（2000）通过 Java 二进制和 JVM 记录并回放基于 Java 的 GUI 操作序列，通过精确复现输入序列来实现一致的执行。同样，DART [131]（2003）自动化了 GUI 测试的全生命周期，从结构分析到测试用例生成与执行，为回归测试提供了一个全面的框架。

_4.1.4 工具与软件（Tools and Software）_

在这一时期，出现了一系列用于 GUI 测试和业务流程自动化的软件工具。Microsoft Power Automate [132]（2019）提供了一个低代码/无代码环境，用于在 Microsoft 应用中创建自动化工作流。Selenium [133]（2004）支持跨浏览器的 Web 测试，而 Appium [134]（2012）则支持移动端 UI 自动化。商业工具如 TestComplete [135]（1999）、Katalon Studio [136]（2015）和 Ranorex [137]（2007）允许用户创建具备跨平台能力的自动化测试。

尽管这些早期系统在自动化特定预定义工作流方面颇为有效，但它们缺乏灵活性，并且需要手工编写脚本或规则逻辑。然而，它们为后续更智能系统的构建奠定了 GUI 自动化的基础。

**4.2 向智能体的转变（The Shift Towards Intelligent Agents）**

机器学习的引入标志着向更具适应性与能力的 GUI 智能体迈出的重要一步。在这一阶段的早期里程碑中，机器学习、自然语言处理、计算机视觉和强化学习在 GUI 任务中的应用发挥了关键作用。

_4.2.1 机器学习与计算机视觉（Machine Learning and Computer Vision）_

RoScript [110]（2020）是一个开创性的系统，它为触摸屏应用引入了非侵入式机器人测试系统，将 GUI 自动化拓展到多种平台。AppFlow [138]（2018）利用机器学习识别常见的界面与 UI 组件，从而实现对大类应用的模块化测试。计算机视觉方面的进展也推动了 GUI 测试的显著发展，一些框架 [117]（2010）实现了视觉交互任务的自动化。Humanoid [139]（2019）使用在 Android 系统中基于人类交互轨迹训练的深度神经网络模型来学习用户如何根据应用 GUI 选择动作，随后利用该模型指导测试输入生成，从而提升覆盖率并在测试中产生更拟人化的交互模式。类似地，DeepGUI [140]（2021）采用深度学习技术来过滤屏幕中无关区域，通过只关注重要元素，提高黑盒 GUI 测试的有效性。这些方法展示了深度学习在使 GUI 测试更高效、更贴近真实用户行为方面的潜力。

控件检测（widget detection）是 White 等人 [103]（2019）提出的一项技术，它利用计算机视觉精确识别 UI 元素，作为一种支撑技术以实现更智能、更灵敏的 UI 自动化。通过检测并分类界面组件，该方法提升了智能体在复杂且动态的 GUI 中进行有效交互的能力 [141]。

_4.2.2 自然语言处理（Natural Language Processing）_

自然语言处理能力为 GUI 自动化引入了新的维度。RUSS [142]（2021）和 FLIN [143]（2020）等系统允许用户通过自然语言命令控制 GUI，从而在自然语言与机器动作之间搭建桥梁。诸如 [144]（2020）中提出的数据集则进一步推动了该领域发展，它们将自然语言指令映射到移动 UI 操作，为 GUI 控制开辟了更广阔的应用空间。然而，这些方法通常只适用于处理简单自然语言命令，尚不足以应对长期任务。

_4.2.3 强化学习（Reinforcement Learning）_

World of Bits（WoB）环境的构建 [145]（2017）使得在 Web 场景中利用强化学习（Reinforcement Learning, RL）训练智能体成为可能。Workflow-guided exploration [146]（2018）进一步提升了 RL 的效率和任务性能。DQT [147]（2024）将深度强化学习应用于 Android GUI 测试，通过保留控件结构与语义来自动化测试；而 AndroidEnv [148]（2021）为在 Android 上训练智能体提供了更真实的仿真环境。WebShop [149]（2022）展示了大规模 Web 交互的潜力，凸显了基于 RL 的 GUI 自动化的日益成熟。

尽管这类基于机器学习的方法相比早期的规则式系统 [150], [151] 更具适应性，但它们在处理多样化、不可预见的任务时仍然存在困难。其对预定义工作流的依赖以及有限的适应性，意味着在面对新环境时通常需要重新训练或定制化，同时对自然语言的支持也相对有限。

**4.3 以 LLM 为大脑的 GUI 智能体的兴起（The Advent of LLM-Brained GUI Agents）**

LLM 的引入，尤其是多模态模型（如 GPT-4o [93]（2023））的出现，通过支持直观的自然语言交互，彻底改变了 GUI 自动化。与此前需要集成多个独立模块的方法不同，LLM 为 GUI 自动化提供了一种端到端的解决方案，在自然语言理解、视觉识别和推理方面具备高级能力。LLM 为 GUI 智能体带来了多项独特优势，包括自然语言理解、多模态处理、规划与泛化，这些特性使 LLM 与 GUI 智能体的结合极具威力。虽然在此之前已有一些探索性工作，但 2023 年是 LLM 驱动 GUI 智能体的关键节点，在 Web、移动端和桌面应用等多种平台上都涌现出重要进展。

_4.3.1 Web 领域（Web Domain）_

LLM 在 GUI 自动化中的最初应用发生在 Web 领域，早期研究工作建立了基准数据集和环境 [145], [149]。WebAgent [152]（2023）与 WebGUM [153]（2023）是关键里程碑，它们率先利用 LLM 实现真实世界中的 Web 导航。这些工作为后续研究奠定了基础 [17], [154], [155]，后续工作利用更为专门化的 LLM 进一步增强 Web 交互能力。

_4.3.2 移动设备（Mobile Devices）_

LLM 在移动设备上的集成始于 AutoDroid [156]（2023），该系统将 LLM 与领域知识结合，用于智能手机自动化。随后，MM-Navigator [157]（2023）、AppAgent [18]（2023）和 Mobile-Agent [158]（2023）等工作进一步实现了对智能手机应用的精细控制。后续研究通过模型微调继续提升移动 GUI 自动化的准确性 [159], [160]（2024）。

_4.3.3 计算机系统（Computer Systems）_

在桌面应用方面，UFO [19]（2024）是最早利用具备视觉能力的 GPT-4 来完成 Windows 环境中用户命令的系统之一。Cradle [161]（2024）将这类能力拓展至软件应用与游戏；同时，Wu 等人 [162]（2024）实现了跨多类桌面应用（包括 Web 浏览器、代码终端和多媒体工具）的交互。

_4.3.4 工业界模型（Industry Models）_

在工业界，Claude 3.5 Sonnet 模型 [163]（2024）引入了“computer use”功能，能够通过 UI 操作与桌面环境交互 [164]。这表明工业界开始充分认识到 LLM 驱动 GUI 智能体的应用价值，并逐步加大投入。

OpenAI 随后在 2025 年快速推出了 Operator [165]，这是一种类似 Claude 的“计算机使用智能体”（Computer-Using Agent, CUA），在多项基准上取得了最新的水平（state-of-the-art）。这一进展进一步凸显了工业界对 GUI 智能体价值的认可及其不断增长的投入。随着兴趣持续高涨，GUI 智能体的研究与开发有望变得愈发激烈，标志着一个快速演进格局的起点。

毋庸置疑，LLM 为 GUI 智能体带来了前所未有的新范式，大幅提升了其智能水平。随着该领域持续演进，我们预期会迎来一波商业化浪潮，从而在用户与 GUI 应用的交互方式上带来变革性变化。

**4.4 GUI 智能体 vs. API 智能体（GUI Agent vs. API-Based Agent）**

在数字环境中运行的 LLM 驱动智能体领域，其动作空间大体可以划分为两类：

1) GUI 智能体（GUI Agents）：主要依赖 GUI 操作（如点击、按键等）来完成任务。
2) API 智能体（API-Based Agents）：利用系统或应用的原生 API 来达成目标。

我们在图 4 中展示了这两类智能体的基本原理。每一类都有其独特优势，深入理解这些差异对于设计高效智能体至关重要。

GUI 操作为智能体提供了一种通用控制接口，使其可以使用同一套动作原语跨多种应用运行。这使得 GUI 智能体具有高度的可泛化性，因为它们无需针对特定应用进行适配即可与多种软件环境交互。然而，基于 GUI 的交互本身更加复杂，即便是简单任务也可能需要多个顺序步骤，从而增加智能体在长期多步工作流中的决策开销和计算资源消耗。另一个关键方面是 GUI 智能体动作的透明性。由于 GUI 智能体通过点击、输入和界面导航等方式与应用交互，与人类的操作方式相同，其行为对用户而言更加可见且易于解释。这种透明性有助于增强用户对智能体-计算机交互的信任与理解。

相比之下，API 智能体提供了一种更高效、更直接的任务完成方式。通过利用原生 API，许多任务往往可以通过一次精确调用完成，从而显著降低执行时间和复杂度。然而，这些原生 API 往往是私有的或仅限于特定应用内部使用，因而在可访问性与可泛化性方面受到限制。在无法获取 API 或可用性不足的场景中，API 智能体的通用性较弱。此外，API 智能体通常在“幕后”运行，通过直接的系统调用执行任务。虽然这种方式在效率和可靠性方面往往更具优势，但也使其操作过程对终端用户而言不够可见、难以调试。

最有效的数字智能体很可能会采用**混合式**的方式运行，结合两种方法的优势。此类智能体可以通过 GUI 操作获得对各类软件的广泛兼容性，同时在可用时利用原生 API 最大化效率与效果。这类混合智能体在泛化能力与任务优化之间取得平衡，因此也是本综述关注的关键方向之一。关于 GUI 智能体与 API 智能体更加全面的对比，读者可参见文献 [166]。

###### 5 以 LLM 为大脑的 GUI 智能体：基础与设计（LLM-BRAINED GUI AGENTS: FOUNDATIONS AND DESIGN）

从本质上讲，以 LLM 为大脑的 GUI 智能体旨在处理以自然语言给出的用户指令或请求，通过截图或 UI 元素树来理解当前 GUI 状态，并执行模拟人类交互的动作，跨越各种软件界面完成操作 [19]。这些智能体利用 LLM 在自然语言理解、推理与生成方面的高级能力，准确把握用户意图，评估 GUI 上下文，并在多样化环境中自主与应用交互，从而完成复杂的多步任务。通过这一整合，它们可以无缝地解释并响应用户请求，为 GUI 自动化注入适应性与智能性。

作为一种专门类型的 LLM 智能体，目前大多数 GUI 智能体采用类似的基础框架，集成了规划、记忆、工具使用等核心组件，以及多智能体协作等高级增强机制 [47]。不过，要在各类应用环境中实现良好的适应性与功能性，每个组件都必须针对 GUI 智能体的特定目标进行定制。

接下来的小节将对各个组件进行深入介绍，为从零开始构建一个 LLM 驱动的 GUI 智能体提供实用指南与教程。这一全面的拆解可被视为一份“菜谱”，用于创建充分发挥 LLM 能力的高效智能 GUI 自动化系统。

**5.1 架构与工作流程概览（Architecture and Workflow In a Nutshell）**

在图 5 中，我们展示了以 LLM 为大脑的 GUI 智能体的架构，说明从用户输入到任务完成的操作序列。该架构由多个集成组件构成，每个组件都为智能体基于用户给出的自然语言指令进行任务理解与执行提供支撑。在接收到用户请求后，智能体会遵循一条系统化的工作流程，包括环境感知、提示工程（prompt engineering）、模型推理、动作执行以及持续的记忆使用，直至任务完全完成。

总体而言，该架构包含如下组件：

1) 运行环境（Operating Environment）：运行环境定义了智能体的操作上下文，涵盖移动设备、Web 浏览器以及 Windows 等桌面操作系统等平台。为了与环境进行有意义的交互时，智能体会通过截图、控件树或其他用于捕获 UI 结构的方法感知环境的当前状态 [167]。它会持续监控每一步动作的影响反馈，并在实时中调整策略，以确保任务能够有效推进。

2) 提示工程（Prompt Engineering）：在感知环境之后，智能体会构建一个详细的提示（prompt），用于引导 LLM 的推理 [168]。该提示会融合用户指令、经过处理的视觉数据（例如截图）、UI 元素的布局与属性，以及与当前任务相关的其他上下文信息。通过这种结构化输入，可以最大化 LLM 生成与当前 GUI 状态相一致的连贯、具备上下文感知能力的响应。

3) 模型推理（Model Inference）：构建好的提示会被传递给大语言模型，即智能体的核心推理引擎，由其生成完成用户请求所需的一系列规划、动作与洞见。该模型可以是通用型 LLM，也可以是使用 GUI 特定数据微调后的专用模型，从而更加细致地理解 GUI 交互、用户流程与任务需求。

4) 动作执行（Actions Execution）：基于模型推理结果，智能体会识别出任务执行所需的具体动作（例如鼠标点击、键盘输入、触摸屏手势或 API 调用）[13]。智能体中的执行器（executor）会将这些高层指令转换为可直接作用于 GUI 的可执行命令，从而在多种应用与设备上有效模拟人类式的交互。

5) 记忆（Memory）：对于多步任务，智能体会维护内部记忆，用于追踪先前动作、任务进度和环境状态 [54]。这一记忆机制确保了复杂工作流中的连贯性，使智能体能够回溯已有步骤并据此调整后续动作。此外，还可以引入外部记忆模块，以实现持续学习、访问外部知识，并增强对新环境或新需求的适应能力。

通过在上述阶段中迭代地循环，并将这些基础组件组合起来，LLM 驱动的 GUI 智能体得以智能地运行，能够在多种软件界面之间无缝适配，真正搭建起语言指令与具体动作之间的桥梁。每个组件对于提升智能体在动态环境中处理复杂任务时的鲁棒性、响应性与能力都至关重要。在接下来的小节中，我们将详细介绍这些组件背后的设计与核心技术，为从头构建 LLM 驱动 GUI 智能体提供一份全面指南。

**5.2 运行环境（Operating Environment）**

以 LLM 为大脑的 GUI 智能体的运行环境涵盖多种平台，例如移动端、Web 与桌面操作系统等，在这些平台上智能体可以与图形界面交互。每种平台都有其独特特性，会影响 GUI 智能体在其中感知、理解和行动的方式。各平台的 GUI 示例如图 6 所示。本小节将详细说明每个平台的细微差异、智能体获取环境信息的方式，以及在适应多样运行环境时面临的挑战。

_5.2.1 平台（Platform）_

以 LLM 为大脑的 GUI 智能体的运行环境包括移动端、Web 以及桌面操作系统等平台，在这些平台上智能体与图形界面进行交互。每个平台都具有不同的特性，影响智能体对环境的感知、解释与动作方式。图 6 展示了各个平台的 GUI 示例。本小节将详细介绍各个平台的细微差异、智能体收集环境信息的方式，以及在适应多样化运行环境时面临的挑战。

1) 移动平台（Mobile Platforms）：移动设备的屏幕空间受限，严重依赖触摸交互 [170]，并拥有多样的应用架构（例如原生应用 vs. 混合应用）。移动平台通常会提供无障碍（accessibility）框架，如 Android 的 Accessibility API[4] [171] 与 iOS 的 VoiceOver Accessibility Inspector[5]，用于暴露 UI 元素的结构化信息。然而，GUI 智能体在移动环境中还必须处理额外的复杂性，例如手势识别 [169]、应用导航 [172] 以及平台相关约束（如安全与隐私权限）[173], [174]。

2) Web 平台（Web Platforms）：Web 应用通常通过超文本标记语言（Hypertext Markup Language, HTML）与文档对象模型（Document Object Model, DOM）结构 [175], [176] 提供相对标准化的接口。GUI 智能体可以利用 HTML 属性（如元素 ID、class 和 tag）来定位交互组件。Web 环境还包含动态内容、自适应布局和异步更新（例如 AJAX 请求）[177]，这要求智能体持续监测 DOM，并根据界面元素的变化调整动作。

3) 计算机平台（Computer Platforms）：桌面操作系统（例如 Windows）为 GUI 交互提供了全面控制能力。智能体可以利用系统级自动化 API，例如 Windows UI Automation[6] [32]，来获取 UI 元素的完整数据，包括类型、标签、位置与边界框。这些平台通常支持更广泛的交互类型，例如多种鼠标与键盘操作以及复杂的多窗口操作，使 GUI 智能体能够执行复杂工作流。然而，这些系统在适配多样化应用时同样需要更为复杂的策略，从简单界面到结构复杂的多层软件套件，均需加以考虑。

总而言之，移动、Web 与桌面等多样平台使 GUI 智能体能够在统一框架下提供广泛的自动化能力，成为一种通用化解决方案。但每个平台在系统与应用层面都呈现出独特特征与约束，因此需要针对性设计以实现高效集成。通过充分考虑这些平台特定特性，GUI 智能体可以被优化以满足各环境的特殊需求，从而在不同自动化场景中提高其适应性与可靠性。

_5.2.2 环境状态感知（Environment State Perception）_

准确感知环境当前状态对于 LLM 驱动的 GUI 智能体至关重要，因为这一感知直接影响其决策与行动规划过程。该感知通常通过收集结构化数据（例如控件树）与非结构化数据（例如截图）相结合的方式实现，以捕获界面及其组件的完整表示。我们在表 4 中总结了不同平台上用于收集 GUI 环境数据的关键工具包，并在下文中详细讨论它们的作用：

1) GUI 截图（GUI Screenshots）：截图提供了应用在某一时刻的视觉快照，呈现整个 GUI 的状态。它为智能体提供了关于布局、设计与视觉内容的参考，尤其是在 UI 元素结构信息有限或不可用时。图标、图像及其他可能包含重要上下文的图形线索都可以直接从截图中进行分析。许多平台都提供了内置的截图工具（例如 Windows 的截图工具（Snipping Tool）[7]、macOS 的截图工具（Screenshot Utility）[8] 以及 Android 的 MediaProjection API[9]），并且截图还可以通过附加标注进行增强，例如使用 Set-of-Mark（SoM）高亮 [178] 或在关键 UI 组件上绘制边界框 [179]，以简化智能体的决策过程。图 7 展示了 VS Code GUI 的多种截图，包括干净版本，以及带有 SoM 标注与边界框的版本，这些标注突出可交互组件，有助于智能体关注界面中最关键的区域。

2) 控件树（Widget Trees）：控件树以层次化视图呈现界面元素，提供关于布局以及组件之间关系的结构化数据 [180]。图 8 展示了一个 GUI 及其控件树示例。通过访问控件树，智能体可以识别元素类型、标签、角色以及界面中的层级关系等属性，这些信息对于上下文理解至关重要。Windows UI Automation 和 macOS 的 Accessibility API[10] 等工具可以提供这类结构化信息。

3) UI 元素属性（UI Element Properties）：除了控件树之外，UI 元素属性（例如标题、自动化 ID、控件类型及几何位置等）也非常重要。图 9 展示了 PowerPoint 应用中用于 GUI 智能体交互的 UI 元素属性示例。通过获取这些属性，智能体可以在界面上精准定位目标控件并执行相应操作。

4) 环境数据收集工具（Environment Data Collection Toolkits）：表 4 与表 5 总结了在 Web、移动端与桌面平台上常用的环境数据采集工具。下面以表格形式列出部分工具及其可访问的信息（英文工具名与链接保留，列标题采用中文翻译）：

|工具（Tool）|平台（Platform）|环境（Environment）|可访问信息（Accessible Information）|
|---|---|---|---|
|Selenium|Web|浏览器（跨平台）|DOM 元素、HTML 结构、CSS 属性|
|Puppeteer|Web|浏览器（Chrome, Firefox）|DOM 元素、HTML/CSS、网络请求|
|Playwright|Web|浏览器（跨平台）|DOM 元素、HTML/CSS、网络交互|
|TestCafe|Web|浏览器（跨平台）|DOM 元素、HTML 结构、CSS 属性|
|BeautifulSoup|Web|HTML 解析|HTML 内容、DOM 元素|
|Protractor|Web|浏览器（Angular）|DOM 元素、特定于 Angular 的属性|
|WebDriverIO|Web|浏览器（跨平台）|DOM 元素、HTML/CSS、网络交互|
|Ghost Inspector|Web|浏览器（跨平台）|DOM 元素、截图、测试脚本|
|Cypress|Web|浏览器（跨平台）|DOM 元素、HTML/CSS、网络请求|
|UIAutomator|Mobile|Android|UI 层级、控件属性、屏幕内容|
|Espresso|Mobile|Android|UI 组件、视图层级、控件属性|
|Android View Hierarchy|Mobile|Android|UI 层级、控件属性、布局信息|
|iOS Accessibility Inspector|Mobile|iOS|无障碍树、UI 元素、属性|
|XCUITest|Mobile|iOS|UI 元素、无障碍属性、视图层级|
|Flutter Driver|Mobile|Flutter 应用|控件树、属性、交互|
|Android’s MediaProjection API|Mobile|Android|截图、屏幕录制|
|Windows UI Automation|Computer|Windows|控件属性、控件树、无障碍树|
|Sikuli|Computer|Windows, macOS, Linux|截图（图像识别）、UI 元素|
|AutoIt|Computer|Windows|窗口标题、控件属性、坐标|
|Inspect.exe|Computer|Windows|UI 元素、控件属性、无障碍树|
|macOS Accessibility API|Computer|macOS|无障碍树、UI 元素、控件属性|
|Pywinauto|Computer|Windows|控件属性、UI 层级、窗口信息|
|Electron Inspector|Computer|Electron 应用|DOM 元素、HTML/CSS、JavaScript 状态|
|Windows Snipping Tool|Computer|Windows|截图|
|macOS Screenshot Utility|Computer|macOS|截图、屏幕录制|
|AccessKit|Cross-Platform|多种操作系统|无障碍树、控件属性、角色信息|
|Appium|Cross-Platform|Android, iOS, Windows, macOS|UI 元素、无障碍属性、手势操作|
|Robot Framework|Cross-Platform|Web、移动端、桌面|UI 元素、DOM、截图|
|Cucumber|Cross-Platform|Web、移动端、桌面|步骤定义、UI 交互|
|TestComplete|Cross-Platform|Web、移动端、桌面|UI 元素、DOM、控件属性|
|Katalon Studio|Cross-Platform|Web、移动端、桌面|UI 元素、DOM、截图|
|Ranorex|Cross-Platform|Web、移动端、桌面|UI 元素、DOM、控件属性|
|Applitools|Cross-Platform|Web、移动端、桌面|截图、视觉检查点、DOM 元素|

这些工具为 GUI 智能体提供了访问界面结构与视觉数据的多种途径，是实现环境状态感知的基础。通过合理利用这些工具，智能体可以构建对当前 GUI 状态的全面理解，从而支持更可靠的决策与动作规划。
此外，原文还通过一个表格概括了不同平台在 GUI 自动化中的典型挑战、动作空间与代表性任务，现将其翻译如下：

表 3：不同平台的典型 GUI 挑战、动作空间与代表性任务

|平台（Platform）|典型 GUI 挑战（Typical GUI Challenges）|动作空间（Action Space）|代表性任务（Representative Tasks）|
|---|---|---|---|
|移动端（Mobile）|- 受限的屏幕空间（Constrained screen real estate）<br/>- 对触摸与手势识别的高度依赖 [169]<br/>- 应用架构（原生 vs. 混合）<br/>- 无障碍框架（如 Android 的 Accessibility API、iOS 的 VoiceOver）<br/>- 平台相关约束（权限、安全、隐私）|- 轻触、滑动、捏合等触控手势（tap, swipe, pinch 等）<br/>- 虚拟键盘输入<br/>- 应用内导航（菜单、标签页）<br/>- 访问硬件特性（摄像头、GPS 等）|- 应用内登录与表单填写<br/>- 消息发送与社交媒体发布<br/>- 基于位置的服务与地图交互<br/>- 处理推送通知与权限对话框|
|Web（Web）|- 动态与响应式布局（Dynamic and responsive layouts）<br/>- 异步更新（AJAX、fetch API 等）<br/>- 基于 HTML/DOM 的结构<br/>- 跨浏览器不一致性（Cross-browser inconsistencies）|- 点击、悬停、滚动（click, hover, scroll）<br/>- 基于 DOM 的表单填写<br/>- 链接导航与元素检查<br/>- 触发 JavaScript 事件|- 表单填写（注册、结账等）<br/>- 数据抽取/网页爬取<br/>- 搜索与过滤（如电商场景）<br/>- 多步 Web 导航（重定向、弹窗等）|
|桌面端（Computer）|- 完整的操作系统级界面（Full-fledged OS-level interfaces）<br/>- 多窗口操作与系统级快捷键<br/>- 自动化 API（如 Windows UI Automation [32]）<br/>- 频繁的软件更新需要适配<br/>- 复杂的多层软件套件|- 鼠标点击、拖拽（drag-and-drop）<br/>- 键盘快捷键与文本输入<br/>- 菜单导航、工具栏操作<br/>- 访问多个应用窗口|- 文件管理与系统设置<br/>- 生产力软件使用（办公套件、IDE 等）<br/>- 安装/卸载应用程序<br/>- 协调多应用工作流|

图 7 对 VS Code 的 GUI 截图给出了示例：（a）一个干净的 GUI 截图；（b）带有 SoM 高亮的 GUI 截图；（c）带有边界框高亮的 GUI 截图。该图展示了不同标注方式如何帮助智能体聚焦于可交互部件。

图 8 展示了一个 GUI 及其控件树（widget tree）的示例，其中左侧为 GUI 层级结构，右侧为控件树中列出的各控件及其层次关系。

表 4 与图 9 给出了 PowerPoint 应用中若干 UI 元素属性的具体示例，其翻译如下：

表 4：PowerPoint 应用中 UI 元素属性示例（节选）

|控件（Widget）|控件名称（Widget Name）|位置（Position）|属性（Attributes）|
|---|---|---|---|
|Button - 'Remove Background'|按钮 - “Remove Background”（移除背景）|L-3810, T128, R-3708, B243|title='Remove Background'; auto_id='PictureBackgroundRemoval'; control_type='Button'|
|MenuItem - 'Corrections'|菜单项 - “Corrections”（校正）|L-3689, T128, R-3592, B243|title='Corrections'; auto_id='PictureCorrectionsMenu'; control_type='MenuItem'|
|MenuItem - 'Color'|菜单项 - “Color”（颜色）|L-3589, T128, R-3527, B243|title='Color'; auto_id='PictureColorMenu'; control_type='MenuItem'|
|MenuItem - 'Artistic Effects'|菜单项 - “Artistic Effects”（艺术效果）|L-3524, T128, R-3448, B243|title='Artistic Effects'; auto_id='PictureArtisticEffectsGallery'; control_type='MenuItem'|
|MenuItem - 'Transparency'|菜单项 - “Transparency”（透明度）|L-3445, T128, R-3336, B243|title='Transparency'; auto_id='PictureTransparencyGallery'; control_type='MenuItem'|
|Button - 'Compress Pictures...'|按钮 - “Compress Pictures...” （压缩图片）|L-3333, T128, R-3138, B164|title='Compress Pictures...'; auto_id='PicturesCompress'; control_type='Button'|
|MenuItem - 'Change Picture'|菜单项 - “Change Picture”（更改图片）|L-3333, T167, R-3149, B203|title='Change Picture'; auto_id='PictureChangeMenu'; control_type='MenuItem'|
|SplitButton - 'Reset Picture'|分裂按钮 - “Reset Picture”（重设图片）|L-3333, T206, R-3160, B242|title='Reset Picture'; control_type='SplitButton'|

文中还通过文本形式重复展示了这些属性信息，并以“Widget Tree”标题说明它们在控件树中的呈现方式，我们已在上述表格中一并体现。

最后，原文在表 5 中给出了更详细的环境数据采集工具列表，包含“亮点（Highlight）”与“链接（Link）”两列，现补充翻译如下（表头及“亮点”列翻译成中文，工具名与链接保持原文）：

表 5：环境数据采集工具（包含亮点与链接）

|工具（Tool）|平台（Platform）|环境（Environment）|可访问信息（Accessible Information）|亮点（Highlight）|链接（Link）|
|---|---|---|---|---|---|
|Selenium|Web|Browser (Cross-platform)|DOM elements, HTML structure, CSS properties|强大的浏览器支持与自动化能力（Extensive browser support and automation capabilities）|https://www.selenium.dev/|
|Puppeteer|Web|Browser (Chrome, Firefox)|DOM elements, HTML/CSS, network requests|无头浏览器自动化，提供丰富 API（Headless browser automation with rich API）|https://pptr.dev/|
|Playwright|Web|Browser (Cross-platform)|DOM elements, HTML/CSS, network interactions|多浏览器支持，具备自动化与测试能力（Multi-browser support with automation and testing capabilities）|https://playwright.dev/|
|TestCafe|Web|Browser (Cross-platform)|DOM elements, HTML structure, CSS properties|易于上手，支持 JavaScript/TypeScript（Easy setup with JavaScript/TypeScript support）|https://testcafe.io/|
|BeautifulSoup|Web|HTML Parsing|HTML content, DOM elements|用于解析 HTML/XML 文档的 Python 库（Python library for parsing HTML and XML documents）|https://www.crummy.com/software/BeautifulSoup/|
|Protractor|Web|Browser (Angular)|DOM elements, Angular-specific attributes|专为 Angular 应用设计，可与 Selenium 集成（Designed for Angular applications, integrates with Selenium）|https://www.protractortest.org/|
|WebDriverIO|Web|Browser (Cross-platform)|DOM elements, HTML/CSS, network interactions|高度可扩展，拥有丰富插件生态（Highly extensible with a vast plugin ecosystem）|https://webdriver.io/|
|Ghost Inspector|Web|Browser (Cross-platform)|DOM elements, screenshots, test scripts|基于云的浏览器自动化测试与监控（Cloud-based automated browser testing and monitoring）|https://ghostinspector.com/|
|Cypress|Web|Browser (Cross-platform)|DOM elements, HTML/CSS, network requests|实时重新加载与交互式调试（Real-time reloads and interactive debugging）|https://www.cypress.io/|
|UIAutomator|Mobile|Android|UI hierarchy, widget properties, screen content|原生 Android UI 测试框架（Native Android UI testing framework）|https://developer.android.com/training/testing/ui-automator|
|Espresso|Mobile|Android|UI components, view hierarchy, widget properties|Google 提供的原生 Android UI 测试框架（Google’s native Android UI testing framework）|https://developer.android.com/training/testing/espresso|
|Android View Hierarchy|Mobile|Android|UI hierarchy, widget properties, layout information|可通过开发者工具访问的视图层级（View hierarchy accessible via developer tools）|https://developer.android.com/studio/debug/layout-inspector|
|iOS Accessibility Inspector|Mobile|iOS|Accessibility tree, UI elements, properties|用于检查 iOS 应用 UI 元素的工具（Tool for inspecting iOS app UI elements）|https://developer.apple.com/documentation/accessibility/accessibility-inspector|
|XCUITest|Mobile|iOS|UI elements, accessibility properties, view hierarchy|Apple 的 iOS UI 测试框架（Apple’s iOS UI testing framework）|https://developer.apple.com/documentation/xctest/user_interface_tests|
|Flutter Driver|Mobile|Flutter apps|Widget tree, properties, interactions|用于 Flutter 应用的自动化（Automation for Flutter applications）|https://flutter.dev/docs/testing|
|Android’s MediaProjection API|Mobile|Android|Screenshots, screen recording|以编程方式捕获设备屏幕内容（Capturing device screen content programmatically）|https://developer.android.com/reference/android/media/projection/MediaProjection|
|Windows UI Automation|Computer|Windows|Control properties, widget trees, accessibility tree|原生 Windows 支持并与操作系统深度集成（Native Windows support with OS integration）|https://docs.microsoft.com/windows/win32/winauto/entry-uiauto-win32|
|Sikuli|Computer|Windows, macOS, Linux|Screenshots (image recognition), UI elements|基于图像识别的自动化，利用计算机视觉（Image-based automation using computer vision）|http://sikulix.com/|
|AutoIt|Computer|Windows|Window titles, control properties, coordinates|用于 Windows GUI 自动化的脚本语言（Scripting language for Windows GUI automation）|https://www.autoitscript.com/site/autoit/|
|Inspect.exe|Computer|Windows|UI elements, control properties, accessibility tree|用于检查 Windows UI 元素的工具（Tool for inspecting Windows UI elements）|https://docs.microsoft.com/windows/win32/winauto/inspect-objects|
|macOS Accessibility API|Computer|macOS|Accessibility tree, UI elements, control properties|为无障碍与 UI 自动化提供支持的 macOS API（macOS support for accessibility and UI automation）|https://developer.apple.com/accessibility/|
|Pywinauto|Computer|Windows|Control properties, UI hierarchy, window information|基于 Python 的 Windows GUI 自动化库（Python-based Windows GUI automation）|https://pywinauto.readthedocs.io/|
|Electron Inspector|Computer|Electron apps|DOM elements, HTML/CSS, JavaScript state|用于 Electron 应用的调试与检查工具（Tool for Electron applications）|https://www.electronjs.org/docs/latest/tutorial/automated-testing|

上述补充内容对应原文中的各个表格与列举信息，以保证对原始文本的逐项、逐句覆盖与译出。
对于桌面应用而言，如前所述，Windows UI Automation 等工具可以提供结构化视图；而在移动与 Web 平台上，则分别依赖 Android 的 Accessibility API 与 HTML 的 DOM 结构来提供层次化数据。这类分层数据对于智能体规划逻辑交互、在 UI 结构基础上做出决策不可或缺。

3) UI 元素属性（UI Element Properties）：界面中的每个 UI 元素都包含特定属性，例如控件类型（control type）、标签文本、位置以及边界框尺寸等，这些属性帮助智能体精准定位合适的组件。上述属性对智能体判断空间关系（例如相邻元素）与功能用途（例如区分按钮与文本框）至关重要。例如，在 Web 应用中，可以通过 DOM 属性（id、class、name）以及 CSS 样式获取上下文与控件信息。这些属性帮助智能体锁定需要交互的具体元素，从而提升其在多种 UI 环境中导航与操作的能力。图 9 展示了由 Windows UI Automation API 提取的部分 UI 元素属性示例，它们为 GUI 智能体的决策提供了支持。

4) 互补的计算机视觉方法（Complementary CV Approaches）：当结构化信息不完整或不可用时，计算机视觉（CV）技术可以提供额外洞见 [181]。例如，OCR 可以让智能体直接从截图中提取文本内容，便于读取标签、错误信息与说明文字 [121]。此外，高级目标检测模型 [120]，如 SAM（Segment Anything Model）[182]、DINO [183] 和 OmniParser [184]，能够在多种布局中识别和分类 UI 组件，从而支持智能体在 UI 元素频繁变动的动态环境中工作。这些基于视觉的方法保证了鲁棒性，使智能体即便在标准 UI API 不足以支撑的场景下也能有效运行。图 10 给出了一个示例，说明如何使用 CV 方法解析 PowerPoint GUI 并检测非标准控件，推断其类型与标签；我们将在第 5.7.1 节进一步详细介绍这些高级 CV 方法。

综合来看，上述多种信息来源共同构建了 GUI 环境当前状态的全面多模态表示，既包含结构化数据，也包含视觉数据。通过将这些信息纳入提示构建过程，智能体能够做出更加充分、具备上下文感知能力的决策，而不会遗漏关键环境线索。

_5.2.3 环境反馈（Environment Feedback）_

有效的反馈机制对于 GUI 智能体评估每一步动作的成功与否以及为下一步决策提供依据至关重要。反馈形式会根据平台与交互类型的不同而有所差异。图 11 展示了来自环境的多种反馈类型示例。

1) 截图更新（Screenshot Update）：通过比较执行动作前后的截图，智能体可以识别应用状态中的视觉差异。这类截图分析能够揭示界面中的细微变化，例如通知出现、视觉提示或确认信息等，而这些变化可能并不会在结构化数据中体现 [185]。

2) UI 结构变化（UI Structure Change）：在执行某个动作之后，智能体可以检测控件树结构的改变，例如元素的出现或消失、元素属性的更新或层次结构的变化 [186]。这些变化表明交互是否成功（例如是否成功展开下拉菜单或点击按钮），并帮助智能体基于更新后的环境状态确定下一步动作。

3) 函数返回值与异常（Function Return Values and Exceptions）：在某些平台上，动作结果会通过函数返回值或系统抛出的异常直接反馈 [187]。例如，在 Web 平台上，API 响应或 JavaScript 返回值可以用来确认某个操作是否成功，而异常或错误码则可以指示交互失败，引导智能体重试或选择替代方案。

环境提供的这些反馈对于 GUI 智能体评估先前动作的结果至关重要。通过利用这些实时信息，智能体可以评估自身干预的有效性，并决定是继续沿用原规划，还是转而采取替代策略。通过这样的自我反思过程，智能体可以不断调整决策，从而在动态、多样的应用环境中优化任务执行并提升整体性能。

**5.3 提示工程（Prompt Engineering）**

在 LLM 驱动的 GUI 智能体运行过程中，有效的提示构建是关键步骤，它将所有必要信息封装起来，供智能体生成合适响应并成功执行任务 [168]。在从环境中收集到相关数据之后，智能体会构造一个综合提示，将多种对 LLM 推理至关重要的组件整合在一起。每个组件都承担特定职责，彼此配合使智能体能够高效地执行用户请求。图 12 展示了一个 LLM 驱动 GUI 智能体中提示构建的基本示例。提示的关键元素可总结如下：

1) 用户请求（User Request）：即用户给出的原始任务描述，用于说明目标与期望结果。它是智能体后续动作的基础，对确保 LLM 正确理解任务的上下文与范围至关重要。

2) 智能体指令（Agent Instruction）：该部分为智能体的运行提供指导，明确其角色、需要遵循的规则以及具体目标。指令说明智能体将接收哪些输入、期望 LLM 输出什么内容，为推理过程建立框架。核心的智能体指令通常被嵌入到 LLM 的基础系统提示（system prompt）中，同时可以根据环境反馈与上下文自适应需求动态注入或更新补充指令。

3) 环境状态（Environment States）：该部分包含智能体感知到的 GUI 截图和 UI 信息，如第 5.2.2 节所述。这些多模态数据可能由多个版本的截图组成（例如干净版本与带 SoM 标注版本），以确保清晰度并避免 UI 控件被标注遮挡。对环境的这种全面表示对于进行准确决策至关重要。

4) 动作文档（Action Documents）：该组件描述智能体可用的动作集合，详细说明相关文档、函数名、参数、返回值以及其他必要参数。提供这些信息可以让 LLM 在生成动作时具备充分上下文，从而为当前任务选择与生成合适的动作。

5) 示范样例（Demonstrated Examples）：在提示中加入输入/输出示例对于激活 LLM 的上下文学习能力 [97] 至关重要。这些示例有助于模型理解并泛化任务需求，从而提升其在执行 GUI 智能体职责时的表现。

6) 补充信息（Complementary Information）：为了辅助规划与推理，还可以加入额外上下文，例如从智能体记忆中检索到的历史数据（详见第 5.6 节）以及外部知识源（如通过检索增强生成（Retrieval-Augmented Generation, RAG）[188], [189] 获取的文档）。这些补充信息能够提供额外洞察，进一步优化智能体的决策过程。

有效提示的构建是 LLM 驱动 GUI 智能体性能的基石。通过系统性地纳入上述各项信息，智能体可以确保 LLM 在执行任务时具备必要的上下文与指导，从而更准确、高效地完成任务。

**5.4 模型推理（Model Inference）**

构建好的提示会被提交给 LLM 进行推理，由 LLM 生成执行用户请求所需的计划以及具体动作。该推理过程至关重要，因为它决定了 GUI 智能体在动态环境中的表现。通常，该过程包含两个主要部分：规划与动作推断，以及补充输出的生成。图 13 展示了 LLM 在 GUI 智能体场景中的推理输出示例。

_5.4.1 规划（Planning）_

成功执行 GUI 任务通常需要一系列顺序动作，因此智能体必须进行有效的规划 [52], [190]。与人类认知过程类似，审慎的规划对于组织任务、安排动作顺序并确保任务顺利完成至关重要 [51], [191]。LLM 需要在整体上构想一个长期目标的同时，聚焦于启动朝向该目标前进的短期动作 [192]。

为了有效应对多步任务的复杂性，智能体应将整体任务拆分为可管理的子任务，并为这些子任务的执行建立时间线 [193]。可以采用链式思维（CoT）推理 [100] 等技术，让 LLM 生成结构化的计划来指导动作执行。生成出的计划可以在后续推理步骤中被存储并复用，从而增强智能体活动的组织性与聚焦性。

规划的粒度可能会随任务性质与智能体角色的不同而变化 [51]。对于复杂任务，采用将全局规划（识别宏观子目标）与局部规划（为各子目标制定细化步骤）相结合的层次化策略，可以显著提升智能体在管理长期目标时的能力 [194]。

_5.4.2 动作推断（Action Inference）_

动作推断是推理阶段的核心目标，它将规划结果转化为可执行的任务。推断得到的动作通常以函数调用字符串的形式表示，包含函数名及相关参数。这些字符串可以直接转换为现实环境中的交互操作，例如点击、键盘输入、移动端手势或 API 调用。关于这些动作类型的详细讨论见第 5.5 节。

在构建输入提示时，需要预先定义智能体可选择的动作集合。智能体可以从该集合中选择合适动作，或者在允许的情况下生成自定义代码或 API 调用与环境交互 [161]。这种灵活性有助于增强智能体在处理未预见情况时的适应能力；但与此同时，也可能引入可靠性问题，因为自动生成的代码可能容易出错。

_5.4.3 补充输出（Complementary Outputs）_

除了规划与动作推断之外，LLM 还可以生成提升智能体能力的补充输出。这些输出可能包括阐明智能体决策过程的推理链（例如 CoT 推理）、用于与用户交互的消息、与其他智能体或系统沟通的信息，以及任务状态（例如“继续”或“结束”）等。这些功能的设计可以根据具体需求进行定制，从而丰富 GUI 智能体的整体表现。

通过在规划与动作推断之间取得有效平衡，并合理利用补充输出，智能体可以以更高的组织性与适应性来应对复杂任务。

**5.5 动作执行（Actions Execution）**

在推理过程之后，GUI 智能体的关键一步是将推断得到的指令在 GUI 环境中执行，并随后收集反馈。尽管“GUI 智能体”一词可能让人联想到仅进行界面层面的操作，但其动作空间可以通过集成多种工具箱得到大幅扩展，从而增强智能体的多样性。一般而言，GUI 智能体可用的动作大致分为三大类：_(i) UI 操作（UI operations）[144]，(ii) 原生 API 调用（native API calls）[196]，以及 (iii) AI 工具（AI tools）[197]_。每一类都具有独特的优势与挑战，使智能体能够更有效地处理多样任务。

我们在表 6 中对 GUI 智能体中常用的动作进行总结，并按类型进行分类，同时在下文对每一类别进行详细说明。

_5.5.1 UI 操作（UI Operations）_

UI 操作涵盖了用户在软件应用中与 GUI 进行交互的基础行为。这些操作包括多种输入方式，如鼠标操作（点击、拖拽、悬停等）、键盘操作（按键、组合键）、触摸操作（轻触、滑动）以及手势（捏合、旋转等）。上述操作在不同平台与应用中具体形式各异，因此通常需要为每个环境制定针对性策略。

虽然 UI 操作构成了智能体与 GUI 交互的基础，但由于任务的顺序性，使得这种方式在执行复杂工作流时可能较为缓慢。每一步操作都必须按顺序执行，这会增加延迟，尤其当工作流涉及大量交互时更为明显。尽管存在这一缺点，UI 操作仍然是保持跨应用广泛兼容性的关键，因为它们利用的是标准的用户界面元素与交互方式。

_5.5.2 原生 API 调用（Native API Calls）_

与 UI 操作相比，一些应用会提供原生 API，让 GUI 智能体以更高效的方式执行动作。这类 API 可以直接访问应用内部的特定功能，使智能体能够通过单条命令完成复杂任务 [198]。例如，通过调用 Outlook 的 API，智能体可以在一次操作中发送电子邮件，而如果仅依赖 UI 操作，则需要经历一系列步骤，例如在菜单中导航并填写表单 [199]。

虽然原生 API 能显著提升动作执行的速度与可靠性，但它们的可用性有限。并非所有应用或平台都对外暴露可用 API，而且开发这些接口往往需要大量工作与专业知识。因此，尽管原生 API 为高效完成任务提供了有力途径，其通用性却往往不及 UI 操作。

_5.5.3 AI 工具（AI Tools）_

将 AI 工具集成到 GUI 智能体中代表着其能力上的一次重大飞跃。这些工具可以帮助完成广泛任务，包括基于截图或文本的内容摘要、文档增强、图像或视频生成（例如调用 ChatGPT [11]、DALL·E [195]）以及调用其他智能体或 Copilot 工具进行协同辅助。生成式 AI 技术的快速发展，使 GUI 智能体得以解决此前超出其能力范围的复杂问题。

通过集成 AI 工具，智能体可以扩展其功能，并在多种情境下提升性能。例如，GUI 智能体可以调用 AI 摘要工具，从长文档中快速提取关键信息；或调用图像生成工具，为用户演示文稿生成定制可视化内容。这类集成不仅简化了工作流，还使智能体能够在远少于传统方式所需的时间内交付高质量结果。

表 6：GUI 智能体中的动作类型及示例

|动作（Action）|类别（Category）|原始执行者（Original Executor）|示例（Examples）|平台（Platform）|环境（Environment）|工具包（Toolkit）|
|---|---|---|---|---|---|---|
|Mouse actions|UI Operations|Mouse|Click, scroll, hover, drag|Computer|Windows|UI Automation[6], Pywinauto|
|Mouse actions|UI Operations|Mouse|Click, scroll, hover, drag|Computer|macOS|AppleScript[10], Automator[11]|
|Mouse actions|UI Operations|Mouse|Click, scroll, hover, drag|Web|Browser|Selenium, Puppeteer|
|Keyboard actions|UI Operations|Keyboard|Typing, key presses, shortcuts|Computer|Windows|UI Automation[6], Pywinauto|
|Keyboard actions|UI Operations|Keyboard|Typing, key presses, shortcuts|Computer|macOS|AppleScript[10], Automator[11]|
|Keyboard actions|UI Operations|Keyboard|Typing, key presses, shortcuts|Web|Browser|Selenium, Puppeteer|
|Touch actions|UI Operations|Touchscreen|Tap, swipe, pinch, zoom|Mobile|Android|Appium, UIAutomator|
|Touch actions|UI Operations|Touchscreen|Tap, swipe, pinch, zoom|Mobile|iOS|Appium, XCUITest|
|Gesture actions|UI Operations|User hand|Rotate, multi-finger gestures|Mobile|Android, iOS|Appium, GestureTools[12]|
|Voice commands|UI Operations|User voice|Speech input, voice commands|Mobile|Android|SpeechRecognizer[13]|
|Voice commands|UI Operations|User voice|Speech input, voice commands|Mobile|iOS|SiriKit[14]|
|Clipboard operations|UI Operations|System clipboard|Copy, paste|Cross-platform|Cross-OS|Pyperclip[15], Clipboard.js[16]|
|Screen interactions|UI Operations|User|Screen rotation, shake|Mobile|Android, iOS|Device sensors APIs[17]|
|Shell Commands|Native API Calls|Command Line Interface|File manipulation, system operations, script execution|Computer|Unix/Linux, macOS|Bash, Terminal|
|Application APIs|Native API Calls|Application APIs|Send email, create document, fetch data|Computer|Windows|Microsoft Office COM APIs[18]|
|Application APIs|Native API Calls|Application APIs|Access calendar, send messages|Mobile|Android|Android SDK APIs[19]|
|Application APIs|Native API Calls|Application APIs|Access calendar, send messages|Mobile|iOS|iOS SDK APIs[20]|
|System APIs|Native API Calls|System APIs|File operations, network requests|Computer|Windows|Win32 API[21]|
|System APIs|Native API Calls|System APIs|File operations, network requests|Computer|macOS|Cocoa APIs[22]|
|Web APIs|Native API Calls|Web Services|Fetch data, submit forms|Web|Browser|Fetch API[23], Axios[24]|
|AI Models|AI Tools|AI Models|Screen understanding, summarization, image generation|Cross-platform|Cross-OS|DALL·E[195], OpenAI APIs[25]|

图 10 与图 11、图 12、图 13 分别示意了 CV 辅助检测控件、不同类型环境反馈、提示构建示例以及 LLM 推理输出示例，我们在上文相关段落中已结合文字进行了说明。
_5.5.4 小结（Summary）_

一个先进的 GUI 智能体应当能够熟练地利用上述三类动作：通过 UI 操作获得广泛兼容性；通过原生 API 调用实现高效执行；通过集成 AI 工具获得增强能力。这种多元化的方式使智能体能够在各种应用中可靠运行，同时最大化效率与效果。通过灵活运用这些动作类型，GUI 智能体可以更高质量地完成用户请求，从而带来更加顺畅、高效的用户体验。

**5.6 记忆（Memory）**

若要在复杂、多步任务中取得稳健表现，GUI 智能体必须具备记忆能力，以便在原本“无状态”的环境中管理状态。记忆使智能体能够追踪过去的动作、这些动作的结果以及任务的整体状态，这些因素对于后续步骤中做出明智决策至关重要 [200]。通过建立连续性，记忆将智能体从一个纯“反应式”系统转变为能主动调整的“有状态”系统。

智能体的记忆通常分为两大类型：短期记忆（Short-Term Memory, STM）[201] 和长期记忆（Long-Term Memory, LTM）[202]。我们在表 6 中对 GUI 智能体中的不同记忆类型进行了概览。图 14 则示意了短期记忆与长期记忆在一个以 LLM 为大脑的 GUI 智能体中的作用方式。

图 14：以 LLM 为大脑的 GUI 智能体中短期记忆与长期记忆的示意。左侧示例任务 A 为“播放《Arcane》第一季回顾视频（Play the Recap of Arcane Season 1）”，右侧示例任务 B 为“下载刚刚播放视频相关的游戏（Download the game related to the video just played）”。图中展示了每一步的计划（Plan 1、Plan 2、Plan 3）、具体动作序列（如搜索、点击“Play”“Download for Windows”等）如何存储在短期记忆中；而任务级别的信息（例如“我刚刚使用 Google 搜索《Arcane》第一季回顾，并在 YouTube 播放了搜索结果中的顶部视频”）则被保存在长期记忆中，供后续任务检索与复用。

表 6：GUI 智能体中的记忆要素概览

|记忆要素（Memory Element）|记忆类型（Memory Type）|描述（Description）|存储介质/方式（Storage Medium/Method）|
|---|---|---|---|
|Action|短期（Short-term）|在环境中执行过的历史动作轨迹|内存、LLM 上下文窗口（Context window）|
|Plan|短期（Short-term）|从前一步传递而来的计划|内存、上下文窗口|
|Execution Results|短期（Short-term）|返回值、错误轨迹及其他环境反馈|内存、上下文窗口|
|Environment State|短期（Short-term）|重要的环境状态数据，例如 UI 元素|内存、上下文窗口|
|Self-experience|长期（Long-term）|历史任务中的完成轨迹|数据库、磁盘|
|Self-guidance|长期（Long-term）|基于历史轨迹总结得到的指导原则与规则|数据库、磁盘|
|External Knowledge|长期（Long-term）|辅助任务完成的外部知识源|外部知识库|
|Task Success Metrics|长期（Long-term）|跨多次会话统计的任务成功/失败指标|数据库、磁盘|

_5.6.1 短期记忆（Short-Term Memory）_

短期记忆（STM）为 LLM 在运行时提供主要且短暂的上下文 [203]。STM 存储与当前任务相关的信息，如最近的计划、动作、结果与环境状态，并持续更新以反映任务的最新进展。对于多步任务而言，STM 尤其重要，因为每一步决策都建立在前一步结果之上，智能体需要保持对任务轨迹的清晰理解。正如图 14 所示，在完成各个独立任务时，由动作与计划构成的任务轨迹会被存储在 STM 中，使智能体能够有效追踪任务进度并做出更有依据的决策。

然而，STM 受限于 LLM 的上下文窗口大小，只能携带有限信息。为应对这一限制，智能体可以采用选择性记忆管理策略，例如有选择地丢弃或总结不那么重要的细节，以优先保留影响更大的关键信息。尽管容量有限，STM 仍是保证交互连贯性、上下文相关性以及在即时反馈下执行复杂工作流的关键。

_5.6.2 长期记忆（Long-Term Memory）_

长期记忆（LTM）则作为外部存储，用于保存超出即时运行时范围的上下文信息 [204]。与短期记忆的瞬时性不同，LTM 会保留历史任务数据，包括已完成任务、成功的动作序列、上下文提示以及总结出的经验。LTM 通常存储在磁盘或数据库中，因此可以容纳远超 LLM 上下文窗口的海量信息。在图 14 所示的示例中，当第二个任务要求下载与前一任务中播放的视频相关的游戏时，智能体会从 LTM 中检索相关信息，由此准确识别出目标游戏并高效完成任务。

LTM 通过保存成功任务轨迹、操作指南和常见交互模式，为智能体的长期自我提升提供支持。当面对新任务时，智能体可以利用 RAG 技术检索相关历史数据，从而基于以往成功经验调整策略 [53]。这与终身学习（lifelong learning）理念类似，使 LTM 成为帮助智能体从经验中“学习”的关键机制。随着在多次会话中不断积累洞见，智能体在执行任务时的准确性与效率都会得到提升。例如，[205] 就给出了一个利用存储在记忆中的历史任务轨迹来指导并改进后续决策的示例，这一技巧对 GUI 智能体同样具有高度适用性，并且还可以通过保留历史任务信息来实现个性化体验。

**5.7 高级增强技术（Advanced Enhancements）**

尽管大多数以 LLM 为大脑的 GUI 智能体都包含感知、规划、动作执行与记忆等基础组件，但研究中已经提出了一系列高级技术，能够显著提升这类智能体的推理能力与整体性能。本节将概述若干在研究中被广泛采用的通用增强方法，以指导更为专门且能力更强的 LLM 驱动 GUI 智能体的开发。

_5.7.1 基于计算机视觉的 GUI Grounding（Computer Vision-Based GUI Grounding）_

虽然前文提到的各种工具（第 4 节）可以让 GUI 智能体获取控件位置、标题与属性等信息，但某些非标准 GUI 或控件可能并不遵循这些工具所依赖的协议 [243]，从而导致相关信息无法被访问。此外，由于权限管理等原因，这些工具在某些场景下也无法使用。这种信息不完整的情况会给 GUI 智能体带来重大挑战，因为 LLM 可能需要通过估计控件坐标来独立定位并与目标控件交互（例如执行点击），而在缺少精确 GUI 数据时，这一任务本身就极具困难。

计算机视觉模型提供了一种非侵入式的 GUI grounding 方案，它可以直接从截图中进行检测、定位、分割乃至功能估计等操作 [103], [244]–[246]。这类方法使智能体无需依赖系统级工具或内部元数据（这些信息可能缺失或不完整），就可以解析 GUI 的视觉结构与元素。基于 CV 的 GUI 解析能够仅凭视觉线索为智能体提供关于可交互组件、屏幕布局和控件功能的宝贵信息，从而增强其识别和操作屏幕元素的能力。图 10 就展示了一个基于 CV 的 GUI 解析器示例：在标准 API 检测捕获预定义控件的同时，CV 模型还能识别幻灯片缩略图、画布等在 PowerPoint 接口中未必具有明确 API 表示的元素。这提升了控件识别能力，使智能体能够检测到超出传统 API 检测范围的组件。相关 GUI grounding 模型与基准的概览见表 7、表 8 与表 9。

其中一个值得注意的例子是 OmniParser [184]。它实现了一种多阶段解析技术：首先利用微调后的模型检测可交互图标，再通过 OCR 模块提取文本信息，最后借助图标描述模型为每个 UI 元素生成局部语义描述。通过融合这些组件，OmniParser 能够构建 GUI 的结构化表示，从而提升智能体对交互区域与功能元素的理解。这种全面的解析策略被证明显著提升了 GPT-4V 的屏幕理解与交互准确率。

这类基于 CV 的 GUI grounding 层为智能体提供了关键的“落地”信息，大幅提升了其在多样 GUI 中准确、直观交互的能力，尤其有助于处理偏离典型无障碍协议的自定义或非标准元素。此外，诸如迭代缩放（iterative narrowing）等提示方法在提升 VLM 的控件 grounding 能力方面也展现出潜力 [208]。这些方法共同为构建更具适应性与鲁棒性的 GUI 智能体铺平道路，使其能够在更广泛的屏幕环境与应用场景中有效运行。

若干工作提出了用于评估模型与智能体 GUI grounding 能力的基准。例如，ScreenSpot [25] 是一个开创性的基准，用于评估 LLM 驱动智能体在 iOS、Android、macOS、Windows 和 Web 等多种平台上的 GUI grounding 表现。它包含超过 600 张截图与 1,200 条指令，重点关注控件和图标等复杂 GUI 组件。该基准强调了 GUI grounding 对下游任务（如 Web 自动化与移动 UI 交互）的重要性。在此基础上，ScreenSpot-Pro [241] 将评估范围拓展到了更加专业、高分辨率的环境。该版本包含 1,581 个任务与高质量标注，覆盖软件开发、创作工具、CAD、科学应用和办公生产力等领域，其特点包括真实的高分辨率截图以及由领域专家提供的精细标注。

这些基准为评估 GUI grounding 能力提供了关键标准，从而推动 GUI 智能体在界面理解与交互方面的发展。

表 7：部分 GUI grounding 模型与基准（OmniParser 等）概览

|模型/基准（Model/Benchmark）|平台（Platform）|基础模型（Foundation Model）|规模（Size）|数据集（Dataset）|输入（Input）|输出（Output）|亮点（Highlight）|链接（Link）|
|---|---|---|---|---|---|---|---|---|
|OmniParser [184]|移动端、桌面端与 Web|BLIP-2 [206], YOLOv8 [207]|/|67,000 张带有边界框标注的 UI 截图以及 7,185 对由 GPT-4 生成的图标-描述样本|UI 截图|可交互元素的 ID、边界框与描述|提出了一个纯视觉的屏幕解析框架，无需外部信息即可进行通用 UI 理解，显著提升 LLM 驱动智能体的动作预测准确率|https://github.com/microsoft/OmniParser|
|Iterative Narrowing [208]|移动端、Web 与桌面|Qwen2-VL 与 OS-Atlas-Base|/|ScreenSpot [25]|GUI 截图与自然语言查询|(x, y) 坐标，表示 GUI 中目标位置|通过逐步裁剪 GUI 区域来细化预测结果，从而提升 GUI grounding 任务的精度|https://github.com/ant-8/GUI-Grounding-vi|
|Iris [209]|移动端（iOS, Android）、桌面端（Windows, macOS）、Web|Qwen-VL [210]|9.6B|850K 条 GUI 专用标注与 150K 条视觉-语言指令|高分辨率 GUI 截图与自然语言指令|引用任务：生成 UI 元素的详细描述；grounding 任务：在屏幕上定位 UI 元素|提出信息敏感裁剪（Information-Sensitive Cropping）以高效处理高分辨率 GUI 图像，并通过自我精炼双重学习（Self-Refining Dual Learning）在无需额外标注的情况下迭代提升 GUI grounding 与引用任务性能|/|
|Attention-driven Grounding [211]|移动端、Web 与桌面|MiniCPM-Llama3-V 2.5|8.5B|Mind2Web [212]、ScreenSpot [25]、VisualWebBench [213]|GUI 截图与文本查询|通过边界框进行元素定位，执行文本到图像映射以实现 grounding，并生成 GUI 组件的可执行描述|在无需微调的前提下，利用预训练多模态 LLM 中的注意力机制进行 grounding|https://github.com/HeimingX/TAG|
|Aria-UI [214]|Web、桌面与移动|Aria [215]|3.9B|390 万个元素与 1,150 万个样本|GUI 截图、用户指令与动作历史|GUI 元素的像素坐标及对应动作|采用纯视觉方法，避免依赖 AXTree 等结构化输入|https://ariaui.github.io|
|UGround [216]|Web、桌面（Windows, macOS, Linux）、移动端（Android, iOS）|LLaVA-NeXT-7B [217]|7B|Web-Hybrid 及其他现有数据集|GUI 截图、用户查询|GUI 元素的像素坐标|一个仅依赖视觉的通用 GUI grounding 模型，避免使用基于文本的表示|https://osu-nlp-group.github.io/UGround/|
|GUI-Bee [218]|Web|SeeClick [25], QwenGUI [219], UIX-7B [220]|7B–13B|NovelScreenSpot|GUI 截图、用户查询、无障碍树|GUI 元素的 grounding 位置、动作与函数调用、导航步骤以及交互后的 GUI 变化预测|自动探索 GUI 环境，通过 Q-ICRL 优化探索效率并提升数据多样性|https://gui-bee.github.io|

表 8：更多 GUI grounding 模型概览（RWKV-UI 等）

|模型/基准（Model/Benchmark）|平台（Platform）|基础模型（Foundation Model）|规模（Size）|数据集（Dataset）|输入（Input）|输出（Output）|亮点（Highlight）|链接（Link）|
|---|---|---|---|---|---|---|---|---|
|RWKV-UI [221]|Web|1.6B 参数的 RWKV 模型|SIGLIP [222], DINOv2 [223], SAM [182]|Websight [224], WebUI-7kbal [225], Web2Code [226]|高分辨率网页图像|元素 grounding、动作预测、CoT 推理|提出高分辨率三编码器架构，结合视觉提示工程与链式思维推理|/|
|TRISHUL [227]|Web、桌面与移动平台|/（训练-free）|/（训练-free）|/（训练-free）|GUI 截图、用户指令/查询、分层屏幕解析输出、OCR 提取文本描述|动作 grounding、GUI 元素功能描述、GUI 引用和 SoM 标注|利用分层屏幕解析与空间增强的元素描述，在无需额外训练的前提下强化 LVLM 能力|/|
|AutoGUI [228]|Web、移动端|Qwen-VL-10B [210], SliME-8B [229]|10B / 8B|AutoGUI-704k|GUI 截图、用户查询|元素功能、元素位置|通过基于交互变化自动标注 UI 元素，使数据具备可扩展性与高质量|https://autogui-project.github.io/|
|Query Inference [230]|移动端（Android）|Qwen2-VL-7B-Instruct [231]|7B|UIBERT [232]|GUI 截图|面向动作的查询、坐标|在无需大量训练数据的情况下提升推理能力|https://github.com/ZrW00/GUIPivot|
|WinClick [233]|Windows 操作系统|Phi3-Vision [234]|4.2B|WinSpot Benchmark|GUI 截图与自然语言指令|元素位置|首个专门为 Windows 定制的 GUI grounding 模型|https://github.com/zackhuiiiii/WinSpot|
|FOCUS [235]|Web、移动应用与桌面|Qwen2-VL-2B-Instruct [231]|2B|GUICourse [219], Aguvis-stage1 [236], Wave-UI [237], Desktop-UI [238]|GUI 截图 + 任务指令|归一化坐标 (x, y)|提出受人类认知启发的双系统 GUI grounding 架构，根据任务复杂度在快速（直觉）与慢速（分析）模式间动态切换|https://github.com/sugarandgugu/Focus|
|UI-E2I-Synth [239]|Web、Windows 与 Android|InternVL2-4B 与 Qwen2-VL-7B|4B 与 7B|160 万张截图、990 万条指令|GUI 截图|元素坐标|提出三阶段合成数据流水线，用于 GUI grounding，结合显式与隐式指令合成|https://colmon46.github.io/i2e-bench-leaderb|
|RegionFocus [240]|Web 与桌面界面|UI-TARS 与 Qwen2.5-VL|72B|无（仅测试时使用）|带兴趣点的 GUI 截图|基于坐标的动作|提出视觉测试时缩放框架，通过放大显著 UI 区域并引入 image-as-map 机制记录历史、避免重复错误，在无需重新训练模型的情况下提升 grounding 准确率|https://github.com/tiangeluo/RegionFocus|

表 9：GUI grounding 基准概览

|基准（Benchmark）|平台（Platform）|数据集（Dataset）|输入（Input）|输出（Output）|亮点（Highlight）|链接（Link）|
|---|---|---|---|---|---|---|
|ScreenSpot [25]|iOS、Android、macOS 与 Windows|600 多张截图与 1,200 条指令|带有用户指令的 GUI 截图|可操作 GUI 元素的边界框或坐标|一个覆盖多平台和多种元素类型的真实且多样的 GUI grounding 基准|https://github.com/njucckevin/SeeClick|
|ScreenSpot-Pro [241]|Windows、macOS 与 Linux|1,581 对指令-截图组合，覆盖 3 个操作系统下 5 个行业的 23 款应用|高分辨率 GUI 截图与自然语言指令|用于定位目标 UI 元素的边界框|为专业环境引入高分辨率基准|https://github.com/likaixin2000/ScreenSpot-Pro-G|
|PixelWeb [242]|Web|100,000 个网页|渲染后的网页截图与 DOM 信息|边界框、掩码与轮廓（BBox, mask, contour）|首个为 Web UI 提供像素级标注（包括掩码与轮廓）的 GUI 数据集，支持高精度 grounding 与检测任务|https://huggingface.co/datasets/cyberalchemist/PixelWeb|

_5.7.2 多智能体框架（Multi-Agent Framework）_

俗话说，“三个臭皮匠，顶个诸葛亮”。在 GUI 自动化任务中，这一理念尤为适用：单个智能体虽然已经具备一定能力，但在多智能体框架下可以获得显著增强 [247], [248]。多智能体系统通过多智能体的集体智能、专业分工与互补优势，在处理复杂任务时往往比任何单一智能体都更有效。

在 GUI 智能体场景中，多智能体系统主要通过两种机制提供高级能力：（i）智能体专业化，以及（ii）智能体间协同。图 15 展示了一个 LLM 驱动多智能体系统协作创建桌面（desk）的示例。

1) 智能体专业化（Specialization of Agents）：在多智能体框架中，每个智能体被设计为在某一特定角色或功能上专精，利用自身独特能力为整体任务做出贡献。正如图 15 所示，这种专业化使不同智能体专注于任务流水线的不同环节。例如，“Document Extractor（文档抽取器）”专门负责从本地文档（如 PDF）中提取相关内容；“Web Retriever（Web 检索器）”则聚焦于从在线资源中收集额外信息；“Designer（设计者）”把检索到的信息转化为视觉上友好的幻灯片；而 “Evaluator（评估者）”则对结果进行审查并给出改进建议。这种功能分工确保每个智能体在各自职责上高度熟练，从而提升整体效率与结果质量 [249]。

2) 智能体间协同动态（Collaborative Inter-Agent Dynamics）：图 15 中的多智能体系统展示了智能体如何以动态方式协同处理复杂任务。流程通常从“Document Extractor”和“Web Retriever”开始，它们并行地从本地与在线来源收集信息。随后，这些信息会被传递给“Designer”，由其综合整理成一套连贯的幻灯片。幻灯片创建完成后，“Evaluator”会审阅输出并提供反馈以进行改进。各智能体之间通过信息共享、上下文传递与有序通信进行协作，形成类似人类团队协作的工作模式。例如，图中各智能体的角色紧密衔接——前一阶段的输出直接成为下一阶段的输入，构成一条流畅的工作流 [19]。

在这样的系统中，多智能体可以共同参与需要规划、讨论与决策的任务。通过这些交互，系统能够充分挖掘每个智能体的领域专长与潜在的专业化能力，在多样化、多步骤流程中最大化整体性能。图 15 所示的“创建桌面”任务正是这种协同工作的典型示例：GUI 智能体从本地文档与网页中抽取知识，经由“Designer”完成创作，再由“Evaluator”进行反馈与打磨，最终获得高质量的成果。

_5.7.3 自我反思（Self-Reflection）_

俗话说，“知错能改善莫大焉”。在 GUI 多智能体系统中，自我反思指的是智能体在任务执行过程中，对自身推理、动作与决策进行内省评估的能力 [250]。这一能力使得智能体能够发现潜在错误、调整策略并优化动作，从而在复杂或不熟悉的 GUI 环境中提升决策质量与鲁棒性。通过定期审视自身表现，自我反思型智能体可以不断纠正偏差、积累经验并持续改进。

图 15 下半部分给出了一个自我反思的示例任务：“对页面中的图形应用线条绘图效果（Make Line Drawing effect to the figure in the page）”。一开始，GUI 智能体的计划是依次点击“Design”菜单，并尝试在当前界面中寻找线条绘图效果选项。然而在执行过程中，智能体通过自我反思意识到：在“Design”菜单中并未找到相应选项。经过一番反思与试验后，智能体修正了策略：应当先选中目标图形，然后切换到“Picture Format”选项卡，在其中的“Artistic Effects”菜单下寻找被“隐藏”的“Line Drawing”选项。通过这一自我反思与策略调整过程，任务最终得以完成。示例中的思考过程（“也许可以尝试别的路径……我应当考虑……”）体现了智能体在遇到困难时对自身行为的审视与修正能力。

这种自我反思机制与前文的记忆、规划等组件相辅相成，使 GUI 智能体在面对复杂任务时不仅能够“做对”，还能够“知错并改”，从而逐步向更加可靠、类人的智能体演化。
图 16：以 LLM 为大脑的 GUI 智能体在完成任务时进行自我反思的示例。图中展示了智能体在最初未能在 “Design” 菜单中找到 “Line Drawing” 选项后，如何通过自我反思调整策略，先选中图形，再切换到 “Picture Format” 选项卡，并在其中的 “Artistic Effects” 菜单中找到被隐藏的 “Line Drawing” 选项，从而顺利完成任务。

自我反思能力使智能体能够在任务执行过程中动态调整，从而生成更准确、更有效的结果 [251]。对于 GUI 智能体而言，这一点尤为关键，因为用户界面的多变性以及即便在人类操作系统中也存在的潜在错误。GUI 智能体经常会遇到偏离预期的情况，例如误点击按钮、遭遇意外广告、在陌生界面中导航、在 API 调用时收到错误消息，或是接收到用户对任务结果的反馈等。为确保任务成功，GUI 智能体必须能够快速反思自身动作，评估这些反馈信号，并调整计划以更好地对齐目标。

如图 16 所示，当智能体起初无法在 “Design” 菜单中找到 “Line Drawing” 选项时，自我反思机制会促使其重新思考并发现正确位置——该选项位于 “Picture Format” 菜单下的 “Artistic Effects” 中，从而成功完成任务。

在实践中，GUI 智能体的自我反思技术通常主要包括两类方法：（i）ReAct [252]，以及（ii）Reflexion [253]。

1) ReAct（Reasoning and Acting）：ReAct 将自我反思嵌入智能体的动作链中，让智能体在每一步动作之后评估其结果，并推理下一步最佳行动。在这一框架中，智能体不再仅仅顺序执行固定动作序列，而是根据每一步的反馈动态调整策略。例如，当 GUI 智能体在填写表单时意识到自己点击了错误输入框，它可以通过回溯重新选择正确元素。借助 ReAct，智能体能够在一步步的执行中不断修正自身行为，从而获得更高的一致性与准确性。

2) Reflexion：Reflexion 强调基于语言的反馈。智能体从环境中接收并处理以自然语言形式给出的反馈，这些“自反性反馈”（self-reflective feedback）会被纳入后续交互的输入，从而帮助智能体快速从以往错误中学习。例如，当 GUI 智能体从应用中收到错误消息时，Reflexion 框架可以让智能体解析该消息、更新对界面的理解，并在未来交互中避免类似错误。Reflexion 的迭代反馈回路有助于持续改进，尤其适用于在复杂、多步任务中进行导航的 GUI 智能体。

总体而言，自我反思是 GUI 多智能体系统中的关键增强机制，使智能体能够更好地应对 GUI 环境的多变性与不可预测性。这一内省能力不仅提升了单个智能体的表现，也在协作环境中促进了鲁棒性、适应性与长期学习能力。

_5.7.4 自我进化（Self-Evolution）_

自我进化（Self-evolution）[254] 是 GUI 智能体应当具备的重要属性，使其能够通过不断积累的经验逐步提升性能。在 GUI 多智能体系统的背景下，自我进化不仅有助于单个智能体的成长，也有利于智能体之间共享知识与策略，从而实现集体学习与适应。

在任务执行过程中，GUI 智能体会生成详细的动作轨迹，并附带丰富的补充信息，如环境状态、内部推理过程（智能体的思考过程）以及评估结果。这些数据构成了宝贵的知识库，智能体可以从中学习与演化。由这些经验提取出的知识大致可归为三类：

1) 任务轨迹（Task Trajectories）：智能体执行的一系列动作及对应的环境状态，是学习的重要基础 [255]。这些成功的轨迹主要可以通过两种方式加以利用。其一，可用来微调支撑智能体的大语言模型。利用这类领域与任务相关数据进行微调，有助于提升模型的泛化能力，并在未来相似任务中表现更好。其二，这些轨迹也可以作为示例嵌入提示中，激活 LLM 的上下文学习能力。在提示工程中加入成功任务执行的示例，可以帮助智能体在无需额外模型训练的前提下更好地理解并复制期望行为。

例如，当某个智能体成功完成一项跨多个应用自动化数据录入的复杂任务时，其记录下来的动作轨迹——包括执行步骤、决策与上下文线索——可以共享给其他智能体。后者在面对类似任务时即可将该轨迹作为参考，从而缩短学习曲线并提升效率。

2) 指导与规则（Guidance and Rules）：基于不断累积的经验，智能体可以提炼出高层次的规则或指导原则，用于概括最佳实践、成功策略与从错误中汲取的教训 [256], [257]。这类指导信息既可以通过 LLM 对轨迹进行总结获得 [256]，也可借助蒙特卡洛树搜索（Monte Carlo Tree Search, MCTS）[257] 等基于搜索的算法提取。相关知识可以形式化为策略或启发式规则，供智能体在决策过程中查阅，从而增强推理能力。例如，当智能体多次在未满足前置条件（如未指定文件路径就尝试保存文件）时遭遇错误，它便可以总结出一条规则：在执行某些动作前必须检查相关前置条件。这种前瞻性策略能降低错误发生率并提升任务成功率。

3) 新工具箱（New Toolkits）：在长期交互中，GUI 智能体可能会发现或发展出更高效的方法、工具或动作序列，以简化任务执行 [161]。这些新方法可能包括优化的 API 调用、宏命令，或比以往方式更有效的 UI 操作组合。可以利用 LLM 自动分析执行轨迹，对其进行总结与挖掘，从中发现并生成高级“快捷方式”或常用快速 API，供后续执行复用 [258]。通过将这些新工具纳入自身“技能包”，智能体得以扩展能力并提升整体效率。例如，智能体可能发现使用批处理 API 可以比在循环中逐条执行 UI 操作更高效地自动化重复任务；该新方法随后可在多智能体系统中共享，使所有智能体都能受益，并在相关任务中加以应用。

图 17 示意了 GUI 智能体如何通过任务执行不断进化：在任务 1 中完成图像内容摘要时，智能体将“图像摘要”工具加入其新工具箱；在任务 2 中阅读并总结有关构建 GUI 智能体的论文时，它从中提炼出“如何构建 GUI 智能体”的指导；在任务 3 中执行网页信息抽取时，它将这一任务轨迹存入经验池。在之后接收到诸如“从 GitHub 仓库学习如何构建 GUI 智能体”之类的新任务时，智能体便可以综合利用已获得的技能与过往经验进行适应和执行。

这种动态演化过程凸显了智能体持续学习、成长并精炼能力的潜力。通过利用以往经验、吸收新知识和扩展工具集，GUI 智能体能够更好地应对多样化挑战，改进任务执行，并显著提升系统整体性能，从而构建一个协同、持续进化的环境。

_5.7.5 强化学习（Reinforcement Learning）_

强化学习（Reinforcement Learning, RL）[259] 在将 LLM 行为对齐至期望目标方面取得了显著进展 [260]，并且近年来被广泛应用于 LLM 智能体的训练 [50], [261]。在 GUI 多智能体系统的背景下，RL 有潜力显著提升 GUI 智能体的性能、适应性与协作能力。GUI 自动化任务天然契合马尔可夫决策过程（Markov Decision Process, MDP）[262] 的结构，因此非常适合采用 RL 方案进行建模。

在这一设定中，状态（state）对应环境感知结果（如 GUI 截图、UI 元素属性与布局配置）；动作（action）则直接映射为 UI 操作，包括鼠标点击、键盘输入与 API 调用等；奖励（reward）则可以根据多种性能指标进行显式定义，例如任务完成度、效率与准确率等，从而引导智能体优化动作以实现最大化效用。图 18 展示了一个 GUI 智能体任务完成过程中 MDP 建模的示例，其中状态、动作与奖励均有明确定义。

通过将 GUI 智能体交互建模为 MDP，我们可以利用 RL 技术通过试错学习最优策略 [263]。这一方法使智能体能够做出在长期内最大化累积奖励的决策，从而实现更加高效、有效的任务完成。例如，在学习自动化 Web 表单填写任务时，智能体可以通过 RL 探索最为高效的动作序列，以在最少错误与冗余步骤的前提下完成数据输入与提交。该过程有助于让智能体在复杂或存在歧义的场景下更好地对齐预期行为，尤其是当预定义动作序列不足以覆盖所有情况时。

具有代表性的工作之一是 Bai 等人提出的 DigiRL [264]，这是一种用于在动态环境中训练 GUI 智能体的两阶段 RL 框架。DigiRL 首先进行离线 RL 阶段，利用已有离线数据初始化智能体模型；随后在在线微调阶段，模型在 Android 学习环境中与环境直接交互，并借助 LLM 评估器提供可靠奖励信号进行策略优化。这种自适应设定使智能体能够有效学习并应对动态 GUI 的复杂性。

Wang 等人提出的 DistRL [265] 则是另一种专为 Android 设备端控制智能体设计的 RL 微调流水线。DistRL 采用异步架构，在异构工作设备与环境中部署 RL 微调后的智能体进行分布式数据采集。通过利用离线策略（off-policy）RL 技术，DistRL 能够对来自不同环境的远程数据进行集中训练，显著提升模型的可扩展性与鲁棒性。这些代表性方法展示了 RL 提升 GUI 智能体能力的潜力，说明无论集中式还是分布式 RL 框架，都可以在真实应用中构建响应更快、更具适应性且更高效的 GUI 自动化模型。

图 18 展示了一个 GUI 智能体在完成“播放《Arcane》第一季视频”任务时的 MDP 建模示例：在不同状态（State 1, State 2, State 3）下，智能体通过计划与执行动作（例如点击 “YouTube Arcane Season 1 | Recap | Netflix” 链接，再点击 “Play” 按钮）并接收相应奖励，逐步逼近最优策略。

_5.7.6 小结与启示（Summary & Takeaways）_

综上所述，以上高级技术显著增强了以 LLM 为大脑的 GUI 智能体的能力，使其在多智能体框架中变得更加多才多能、高效且具备更强适应性。重要的是，这些技术并非互斥——许多方法可以集成在一起，构建更强大的智能体。例如，在多智能体框架中引入自我反思机制，可以让智能体在协同中共同改进任务策略并从错误中恢复。通过利用这些进展，开发者能够设计出不仅能够自动化复杂多步任务，而且还能通过自我进化、对动态环境的适应以及高效的智能体间协作不断提升能力的 LLM 驱动 GUI 智能体。未来的研究有望提出更为精细的技术，进一步拓展 GUI 自动化的广度与鲁棒性。

**5.8 从基础到创新：一条路线图（From Foundations to Innovations: A Roadmap）**

构建稳健、可适应且高效的 LLM 驱动 GUI 智能体是一个多方面的过程，需要对多个核心组件进行精心集成。在第 5 节中，我们已从架构、设计、环境交互与记忆等方面奠定基础。在此之上，本节开始将关注点转向在实际场景中部署智能体所需的关键要素。这一探索首先从第 6 节对最新 LLM 驱动 GUI 智能体框架的深入回顾展开，突出它们的创新点及对该领域的贡献。随后，我们在第 7 节讨论针对 GUI 智能体的数据采集与处理方法，并在第 8 节介绍模型优化技术。为了确保稳健开发与验证，第 9 节将重点介绍评估方法与基准，用于衡量智能体的性能与可靠性。最后，第 10 节展示了多种真实世界应用，说明这类智能体在不同领域中的变革性影响。

总体而言，这些章节为推动以 LLM 为大脑的 GUI 智能体从基础概念走向实际落地与创新，提供了一条完整路线图。从基础组件到真实部署的全过程，构成了将 LLM 驱动 GUI 智能体从“构想”变为“实现”的关键流水线。

为提供整体视图，作者在图 19 中给出了一个分类体系（taxonomy），将近期关于 LLM 驱动 GUI 智能体的工作按框架、数据、模型、评估与应用进行划分。该分类体系可作为蓝图，帮助读者在上述各领域中梳理广泛的研究与开发工作，同时也承认某些模型、框架或数据集在多个方面对 GUI 智能体功能产生贡献。

###### 6 以 LLM 为大脑的 GUI 智能体框架（LLM BRAINED GUI AGENT FRAMEWORK）

大语言模型的引入为构建 GUI 智能体打开了新的可能性，使其能够理解用户请求、分析 GUI 组件，并在多种环境中自主执行操作。通过为这些模型配备第 5 节中概述的关键组件与功能，研究者为不同平台与应用构建了多种复杂框架。这些框架构成了一个快速演进的研究方向，每一项工作都提出了新的技术与专门能力，不断拓展 GUI 智能体可实现的边界。

本节中，我们将对各类框架进行详细讨论，分析其基础设计原则、技术创新点，以及在 GUI 自动化领域所解决的具体挑战。通过这些分析，我们希望读者能够更深入地理解这些智能体如何塑造人机交互与任务自动化的未来，以及它们在推动这一变革性领域发展中所扮演的关键角色。

**6.1 Web GUI 智能体（Web GUI Agents）**

在 Web GUI 智能体方面的进展，推动了在多样且动态的 Web 环境中自动化复杂任务的能力。近期框架引入了多模态输入、预测建模以及面向任务的优化等创新方法，以提升性能、适应性与效率。本小节首先在表 10、表 11 和表 12 中对关键 Web GUI 智能体框架进行概览总结，随后选取代表性框架，重点介绍它们的独特贡献及其如何共同推动基于 Web 的 GUI 自动化发展。

一个重要趋势是将多模态能力引入 Web 交互，以更好地处理动态 Web 内容。例如，SeeAct [17] 利用 GPT-4V 的多模态能力在真实网站上高效完成动作 grounding。通过同时利用视觉数据与 HTML 结构，SeeAct 结合图像标注、HTML 属性与文本选项等多种 grounding 技术，对实时 Web 内容进行优化交互。这一方法使 SeeAct 在实时 Web 任务上取得了 51.1% 的任务成功率，凸显了在设计鲁棒 Web 智能体时进行动态评估的重要性。

在多模态输入优势的基础上，WebVoyager [269] 进一步推动了自主 Web 导航，实现了跨真实 Web 环境的端到端任务完成。WebVoyager 使用 GPT-4V 同时处理视觉输入（截图）与文本输入（HTML 元素），从而有效地与包含动态渲染内容与复杂交互元素的 Web 界面进行交互。凭借这一多模态能力，WebVoyager 在处理复杂界面时的成功率显著高于仅使用文本的方法，为基于 Web 的任务自动化设立了新的基准。

除多模态集成外，一些框架侧重解析复杂 Web 结构并生成可执行代码，以导航复杂网站。WebAgent [267] 采用双层模型结构，将用于解析长且复杂 HTML 文档的 HTML-T5 与用于程序合成的 Flan-PaLM [498] 结合。通过这一模块化设计，WebAgent 能够将用户指令翻译为可执行的 Python 代码，并通过任务特定子指令自主处理复杂的真实网站。与传统单智能体模型相比，WebAgent 在真实网站上的成功率提升了约 50%，展示了将 HTML 专用解析与代码生成相结合在复杂、多变 Web 环境中的优势。

为了提升 Web 导航中的决策质量，一些框架引入了状态空间探索与搜索算法。LASER [268] 将 Web 导航建模为状态空间探索问题，使其在无需大量上下文示例的情况下实现灵活回溯与高效决策。通过将动作与特定状态关联，并利用 GPT-4 的函数调用功能进行基于状态的动作选择，LASER 能够减少错误并提升任务成功率，尤其在诸如 WebShop 与 Amazon 等电商导航任务中表现突出。这种基于状态的建模方法为提升 LLM 智能体在 GUI 导航中的效率提供了可扩展方案。

类似地，Search-Agent [274] 创新性地引入了最佳优先搜索（best-first search）算法，以增强在交互式 Web 环境中的多步推理能力。通过在状态空间中探索多条候选动作路径，并结合评估与模拟机制，Search-Agent 能够在复杂网页中选择更优的执行策略。

图 19 展示了从基础到创新的 LLM 驱动 GUI 智能体在框架、数据、模型、评估与应用方面的分类结构；图 20 则示意了 WebPilot [275] 在局部优化阶段使用蒙特卡洛树搜索（MCTS）的过程；图 21 展示了 WebDreamer [282] 如何利用 LLM 模拟每个动作的结果，从而在执行前评估不同方案。这些图示在原文中配合文字进一步说明了 Web 领域 GUI 智能体的搜索与规划机制，我们在此不再展开。
在 Search-Agent 的基础上，WebPilot [275] 进一步扩展了搜索式策略，采用结合全局与局部蒙特卡洛树搜索（Monte Carlo Tree Search, MCTS）[500] 的双重优化策略，以提升在复杂、动态环境中的适应能力。如图 20 所示，WebPilot 将整体任务拆分为若干可管理的子任务，并对每个子任务进行局部优化。通过在实时观测的基础上持续调整策略，WebPilot 模拟了类似人类的决策与灵活性。在 WebArena [412] 与 MiniWoB++ [146] 等基准上的广泛实验表明，WebPilot 达到了当前最先进的性能，在适应性方面显著优于现有方法。

在预测建模方向上，WMA [266] 引入了世界模型（world model）来模拟与预测 UI 交互的结果。通过关注基于状态转移的观测，WMA 使智能体能够在真正执行之前模拟动作的结果，从而减少不必要动作并提升任务效率。这种预测能力在需要高准确率的长时序任务中尤为有效，WMA 在 WebArena [412] 与 Mind2Web [212] 等基准上表现出色。

同样地，WebDreamer [282] 提出了利用 LLM 进行基于模型规划（model-based planning）的创新方法，如图 21 所示。WebDreamer 在执行前通过 LLM 模拟并评估潜在动作及其多步结果 [506]，类似“做梦者”（dreamer）预先想象不同方案。通过预先评估各个计划的潜在价值，WebDreamer 选择并执行具有最高期望价值的计划。这种方法在处理复杂、动态 Web 环境时，有效应对安全性与鲁棒决策等关键挑战，并在 VisualWebArena [413] 与 Mind2Web-live [349] 等基准上优于传统的反应式智能体。

除了预测建模之外，将 API 交互集成到 Web 导航中也带来了更强的灵活性与效率。Hybrid Agent [199] 结合 Web 浏览与 API 交互，根据任务需求在两者之间动态切换。通过在结构化数据交互上使用 API 调用，Hybrid Agent 降低了传统纯浏览式导航的时间与复杂度，提高了任务执行的准确性与效率。该混合架构凸显了在 Web 智能体中同时集成结构化 API 数据与类人浏览能力的优势。

为应对复杂 Web 结构与跨域交互带来的挑战，AutoWebGLM [270] 通过简化 HTML、聚焦页面关键组件提供了高效解决方案，从而提高任务准确性。AutoWebGLM 结合强化学习与拒绝采样进行微调，在英文与中文站点上的复杂导航任务中表现优异。其双语数据集与结构化动作-感知模块使其在跨领域 Web 交互中具有实用价值，强调了高效处理多样 Web 任务的重要性。

ECLAIR [291] 则是一个具有开创意义的应用，它用基于基础模型的 GUI 智能体取代传统 RPA，用于企业自动化。与依赖手工规则与刚性脚本的传统 RPA 不同，ECLAIR 能够从视频演示与文本版标准操作流程（Standard Operating Procedures, SOP）中动态学习工作流，大幅降低搭建成本并提升适应性。它运行在企业级 Web 应用上，利用 GPT-4V、GPT-4o 与 CogAgent [505] 感知 GUI 元素、规划动作并执行工作流，同时自动完成验证。通过消除 RPA 在维护成本高、执行脆弱等方面的问题，ECLAIR 提供了一种更灵活、可扩展的 GUI 自动化方案。图 22 对比了 RPA 与基于智能体的自动化：前者需要人工设计工作流与手工编程，执行过程刚性、验证多依赖人工日志；后者则由 GUI 智能体根据人类演示与文档自动学习工作流，实现灵活执行并通过 LLM 自动评估，体现出强大的优势。

表 10：部分 Web GUI 智能体框架概览（WMA 等）

|智能体（Agent）|平台（Platform）|感知（Perception）|动作（Action）|模型（Model）|架构（Architecture）|亮点（Highlight）|链接（Link）|
|---|---|---|---|---|---|---|---|
|WMA [266]|Web|来自 DOM 的可访问性树|UI 操作，例如点击、输入、悬停等|Llama-3.1-8B-Instruct [91] 用于预测观测，GPT-4 用于策略建模|基于仿真观测的单智能体架构|在真正执行前使用世界模型预测状态变化，提升任务成功率并减少不必要交互|https://github.com/kyle8581/WMA-Agents|
|WebAgent [267]|Web|HTML 结构|UI 交互|HTML-T5 用于任务规划与总结，Flan-U-PaLM [498] 用于代码生成|用于规划与程序合成的两阶段架构|利用专门 LLM 实现 HTML 驱动的任务规划与程序化动作执行|/|
|LASER [268]|Web|带有显式状态定义的 Web GUI 结构|基于状态定义的动作，如搜索、选取商品、翻页与完成购买|GPT-4|单智能体架构|采用状态空间探索方法，支持灵活回溯以应对新情境|https://github.com/Mayer123/LASER|
|WebVoyager [269]|Web|在可交互元素上带数字标签的截图|标准 UI 操作|GPT-4V|单智能体|在真实渲染网页中融合视觉与文本信号，增强对复杂 Web 结构的导航能力|https://github.com/MinorJerry/WebVoyager|
|AutoWebGLM [270]|Web|简化 HTML + OCR 文本识别|点击、输入、滚动、选择等 UI 操作以及跳转到指定 URL 等高级 API|ChatGLM3-6B [499]|单智能体|提出 HTML 简化以提升网页理解效率，并构建双语基准|https://github.com/THUDM/AutoWebGLM|
|OpenAgents [271]|Web|DOM 元素|标准 UI 操作、浏览器内动作控制、工具执行 API 调用及结构化数据操作|GPT-4 与 Claude [163]|多智能体架构（Data Agent、Plugins Agent、Web Agent 等）|通过开源多智能体框架民主化语言智能体的使用，面向真实任务优化|https://github.com/xlang-ai/OpenAgents|
|SeeAct [17]|Web|截图与 HTML 结构|标准 UI 操作|GPT-4V|单智能体|利用 GPT-4V 的多模态能力，将视觉与 HTML 信息融合，对动态 Web 内容实现更准确的任务执行|https://github.com/OSU-NLP-Group/SeeAct|
|DUAL-VCR [272]|Web|HTML 元素与截图|标准 UI 操作|Flan-T5-base [498]|两阶段单智能体架构|提出双视角上下文化（dual-view contextualization）方法|
|Agent-E [273]|Web|DOM 结构及其变化|标准 UI 操作|GPT-4 Turbo|由规划智能体和浏览导航智能体组成的层次化多智能体架构|通过层次化架构与自适应 DOM 感知实现更鲁棒的决策|https://github.com/EmergenceAI/Agent-E|
|Search-Agent [274]|Web|截图与文本描述|标准 UI 操作|GPT-4|带搜索能力的单智能体|提出新型推理时搜索算法，增强多步规划与决策能力|https://jykoh.com/search-agents|
|R2D2 [288]|Web|DOM|标准 UI 操作|GPT-4o|单智能体|动态构建内部 Web 环境表示；结合回放缓冲与错误分析，减少导航错误并提升任务完成率|https://github.com/AmenRa/retriv|

表 11：更多 Web GUI 智能体框架概览（Auto-Intent 等）

|智能体（Agent）|平台（Platform）|感知（Perception）|动作（Action）|模型（Model）|架构（Architecture）|亮点（Highlight）|链接（Link）|
|---|---|---|---|---|---|---|---|
|Auto-Intent [284]|Web|HTML 结构|标准 UI 操作|GPT-3.5、GPT-4、Llama-3 [91] 用于动作推断；Mistral-7B [501] 与 Flan-T5-XL [498] 用于意图预测|具自我探索（self-exploration）的单智能体架构|提出自我探索策略以生成语义多样的意图提示|/|
|AdaptAgent [283]|Web|带 HTML/DOM 结构的 GUI 截图|标准 UI 操作与 Playwright 脚本|GPT-4o 与 CogAgent [15]|单智能体|仅需 1–2 个多模态人工示范即可适配未见任务|
|WEPO [287]|Web|HTML 与 DOM|标准 UI 操作|Llama3-8B [91]、Mistral-7B [501]、Gemma-2B [502]|单智能体架构|引入基于 DOM 树结构的距离采样机制，通过 DPO [503] 区分显著与非显著元素以增强偏好学习|/|
|AgentSymbiot [289]|Web（icWeb）|网页元素的可访问性树|标准 UI 操作|大模型：GPT-4o、Claude-3.5；小模型：LLaMA-3 [91]、DeepSeek-R1 [504]|多智能体迭代架构|提出大模型与小模型之间的共生学习过程，通过推测式数据合成、多任务学习与隐私友好混合模式提升数据合成与任务表现|/|
|LiteWebAgent [292]|Web|DOM 与截图|标准 UI 操作与 Playwright 脚本|任意 LLM/MLLM|单智能体|首个开源、生产级 Web 智能体，集成树搜索以支持多步任务执行|https://github.com/PathOnAI/LiteWebAgent|
|ECLAIR [291]|Web|截图|标准 UI 操作|GPT-4V、GPT-4o、CogAgent [505]|单智能体架构|通过从视频与文本文档中学习工作流，消除传统 RPA 的高搭建成本、执行脆弱性与维护负担|https://github.com/HazyResearch/eclair-agents|
|Dammu et al. [293]|Web|DOM 元素与网页可访问性属性|标准 UI 操作|未具体说明|单智能体架构|支持用户对齐的任务执行，智能体以合乎伦理的方式适配个体用户偏好|/|
|Plan-and-Act [294]|Web|HTML|标准 UI 操作|LLaMA-3.3-70B-Instruct [91]|两阶段模块化架构：规划器（PLANNER）+ 执行器（EXECUTOR）|将规划与执行解耦，并提出可扩展的合成数据生成流水线以分别微调各组件|/|
|SkillWeaver [295]|Web|GUI 截图与可访问性树|标准 UI 操作与高层技能 API|GPT-4o|单智能体|为 Web 智能体引入自我提升框架，可通过探索自主发现、合成与精炼可复用技能 API|https://github.com/OSU-NLP-Group/SkillWeaver|
|ASI [296]|Web|网页可访问性树|标准 GUI 动作|Claude-3.5-Sonnet|单智能体|提出可执行验证的程序化技能，并将其作为可调用动作以提高效率|https://github.com/zorazrw/agent-skill-inducti|
|Rollback Agent [297]|Web|可访问性树|标准 GUI 动作|多智能体架构|多模块、受 ReAct 启发的智能体结构|提出模块化回滚机制，可进行多步回滚以避免进入“死局”状态|/|

图 22：RPA 与基于智能体自动化的比较（根据文献 [291] 改编）。左侧为传统 RPA（工作流构建与演示、执行、评估均由人工与硬编码流程主导），右侧为基于 GUI 智能体的自动化（工作流由人类示范、文本文档与视频录屏提供，智能体利用 LLM 自动学习、执行与评估）。

**6.2 移动 GUI 智能体（Mobile GUI Agents）**

移动 GUI 智能体的发展伴随着多模态模型、复杂架构与自适应规划等技术的快速进步，这些技术共同应对移动环境中的独特挑战。移动智能体已从早期的基础交互能力发展为能够在多种移动应用中执行动态、具备上下文感知能力操作的复杂系统。表 13、表 14 与表 15 对代表性的移动 GUI 智能体框架进行了总览。

Wang 等人 [323] 率先在移动 UI 上引入 LLM，使其能够进行对话式交互，是移动 GUI 智能体的早期基础工作之一。该方法通过将 Android 视图层级（view hierarchy）转化为 HTML 风格的结构化文本输入到 PaLM 等基础模型中，从而更好地与 LLM 的训练分布对齐。作者定义并评估了四项核心任务：屏幕摘要（Screen Summarization）、屏幕问答（Screen QA）、屏幕问题生成（Screen Question Generation）以及指令到 UI 映射（Instruction-to-UI Mapping），并展示了在每个任务中仅使用 2 个示例即可获得较强性能。该工作强调了无需微调即可快速原型验证的实用性，是基于提示的移动 GUI 智能体评估的里程碑工作。

后续早期工作聚焦于在无需访问后端系统的前提下，实现类人 GUI 交互。其中具有代表性的一项框架是 AppAgent [18]，它利用 GPT-4V 的多模态能力同时理解视觉与文本信息。借助实时截图与结构化 XML 数据，AppAgent 能够通过点击与滑动等操作直接在各种应用（从社交媒体到复杂图像编辑工具）中与 GUI 交互。其通过自主探索与观察人类示范进行学习的策略，使其能够快速适应新应用，凸显了多模态能力在移动智能体中的有效性。

在此基础上，AppAgent-V2 [304] 进一步提升了视觉识别能力并引入结构化数据解析，从而支持更精确、具上下文感知的交互，并能够在不同应用中执行复杂多步操作。AppAgent-V2 还新增了安全检测机制，以处理敏感数据，并通过跟踪与适配实时交互支持跨应用任务。这一演进过程强调了在实时移动环境中，提升视觉识别与结构化数据处理能力对于任务精度与安全性的重要性。

与此并行，一些以视觉为中心（vision-centric）的方案被提出，用于在不依赖应用特定数据的前提下进一步增强移动任务自动化。例如，Mobile-Agent [158] 利用 OCR、CLIP [509] 与 Grounding DINO [183] 进行视觉感知。通过使用截图与上述视觉工具，Mobile-Agent 能执行从应用导航到复杂多任务处理在内的多种操作，并通过自反机制对错误进行迭代修正。这种基于视觉的方法使 Mobile-Agent 成为一个多功能且高度适应的移动任务助手。

为应对长序列导航与复杂多应用场景中的挑战，Mobile-Agent-v2 [306] 引入了一种多智能体架构，将规划、决策与反思拆分到不同的智能体中。通过将任务进度跟踪、任务相关信息记忆与错误纠正分别交给不同模块，该框架在结构上更易于扩展与维护。结合 Grounding DINO [183] 与 Qwen-VL-Int4 [210] 等先进视觉感知工具，Mobile-Agent-v2 在 Android 与鸿蒙（Harmony OS）上的任务完成率显著提升，表明多智能体系统在处理复杂移动任务方面具有巨大潜力。

除了纯视觉方案，一些框架还着眼于将 GUI 状态转换为自然语言，以便利用 LLM 进行动作规划。VisionTasker [299] 通过处理移动 UI 截图并将其转化为结构化自然语言，实现视觉 UI 解释与顺序 LLM 任务规划的结合。借助 YOLO-v8 [207] 与 PaddleOCR[28] 进行控件检测，VisionTasker 能够在不熟悉的应用中自动化复杂任务，并在某些任务上取得高于人类操作的准确率。这一两阶段设计展示了在移动自动化领域中高度通用且可适应的框架。

类似地，DroidBot-GPT [300] 采用一种创新思路，将 GUI 状态转换为自然语言提示，使 LLM 能够自主决定动作序列。通过解释 GUI 结构并将其编码为 GPT 模型可理解的文本，DroidBot-GPT 无需应用特定修改即可泛化到多种应用。这种适应性凸显了 LLM 在以最少定制数据处理复杂多步任务方面的变革性作用。

为进一步增强动作预测与上下文感知能力，高级框架将感知与动作系统集成到多模态 LLM 中。CoCo-Agent [301] 是其中一例，它通过全面事件感知（Comprehensive Event Perception）与全面动作规划（Comprehensive Action Planning）模块处理图标与布局等 GUI 元素。通过将复杂动作拆解为可管理的步骤，并结合来自 Android in the Wild（AITW）[358] 与 META-GUI [357] 等基准的高质量数据，CoCo-Agent 展示了在多种智能手机应用上可靠自动化任务的能力。
在移动智能体方面，CoAT [298] 通过 chain-of-action-thought 过程来增强动作预测与上下文感知能力。它结合 GPT-4V 与 set-of-mark 标注，克服了传统基于坐标的动作识别局限。借助 Android-In-The-Zoo（AITZ）数据集，CoAT 显著提升了动作预测准确率与任务完成率，在 Android 平台上的可访问性与用户便捷性方面具有较大潜力。

为在较低计算成本下高效处理多步任务，AutoDroid [156] 将基于 LLM 的理解与应用特定知识相结合。通过 HTML 风格的 GUI 表示与基于记忆的方案，AutoDroid 降低了对频繁 LLM 查询的依赖。其云端 + 端侧混合架构增强了响应速度与可访问性，使 AutoDroid 成为适配多种移动任务的实用方案。AutoDroid-V2 [305] 在此基础上进一步发展，利用端侧语言模型自动生成并执行多步脚本，将移动应用中动态、复杂的 GUI 元素转化为结构化应用文档，从而在无需云端资源的情况下实现高效、准确的自动化。这种脚本式方法通过减少查询频率降低了计算开销，同时解决了逐步式（stepwise）智能体的效率瓶颈，使移动任务自动化在隐私保护与可扩展性方面迈出重要一步。

MobileGPT [310] 则通过引入类人“应用记忆”系统来自动化 Android 设备上的任务，其记忆过程模仿“探索（Explore）—选择（Select）—推导（Derive）—回忆（Recall）”的认知分解方式，从而实现高效、准确的任务自动化。分层记忆结构支持模块化、可复用且可适配的任务与子任务，使 MobileGPT 在任务成功率、成本效率与适应性方面都优于同类系统，展示了其在移动自动化中的潜力。

在更高级的分布式场景中，FedMobileAgent [314] 引入联邦学习框架，利用用户手机端自采数据训练移动自动化智能体。为解决传统依赖人工标注数据集的高成本与隐私问题，FedMobileAgent 提出 Auto-Annotation 机制，利用视觉-语言模型（VLM）根据截图与动作推断用户意图，从而自动构建训练数据。该系统通过联邦学习实现去中心化训练，并在非 IID 数据条件下采用自适应聚合策略提升模型性能。实验结果表明，FedMobileAgent 在多个移动基准上取得了与人工标注模型相当的性能，但成本仅为其一小部分。

综上，移动 GUI 智能体已经从单智能体系统演进到能够执行动态、具备上下文感知操作的复杂多智能体框架。这些创新表明，要应对移动环境的多样挑战，高级架构、多模态处理与先进规划策略缺一不可。

表 13：移动 GUI 智能体框架概览（一）

|智能体（Agent）|平台（Platform）|感知（Perception）|动作（Action）|模型（Model）|架构（Architecture）|亮点（Highlight）|链接（Link）|
|---|---|---|---|---|---|---|---|
|Wang et al. [323]|Android 移动设备|Android 视图层级结构|（1）屏幕问题生成，(2) 屏幕摘要，(3) 屏幕问答，(4) 指令到 UI 动作映射|PaLM [507]|单智能体|首篇利用 LLM 研究 Screen Question Generation 与 Screen QA 的工作|https://github.com/google-research/google-research/tree/master/llm4mobile|
|VisionTasker [299]|Android 移动设备|带控件检测与文本抽取的 UI 截图|点击、滑动、文本输入等 UI 操作|ERNIE Bot [508]|具视觉 UI 理解与顺序任务规划的单智能体|基于视觉的 UI 语义理解方法，可直接从截图而无需依赖视图层级进行 UI 解释|https://github.com/AkimotoAyako/VisionTasker|
|DroidBot-GPT [300]|Android 移动设备|将 GUI 状态转换为自然语言提示|点击、滚动、勾选、编辑等 UI 操作|GPT|单智能体|无需修改应用或模型即可自动化 Android 应用|https://github.com/MobileLLM/DroidBot-GPT|
|CoCo-Agent [301]|Android 移动设备|GUI 截图、OCR 布局、历史动作|点击、滚动、输入等 GUI 动作|CLIP [509] 进行视觉编码，LLaMA-2-chat-7B 进行语言处理|单智能体|提出“全面环境感知 + 条件动作预测”的双重方案|https://github.com/xbmxb/CoCo-Agent|
|Auto-GUI [302]|Android 移动设备|GUI 截图|GUI 操作|BLIP-2 视觉编码器 [206] 与 FLAN-Alpaca [78]|具有 chain-of-action 机制的单智能体|直接与 GUI 元素交互，并通过 chain-of-action 机制联合利用历史与规划动作|https://github.com/cooelf/Auto-GUI|
|MobileGPT [310]|Android 移动设备|简化的 HTML 表示|标准 UI 操作与导航动作|GPT-4-turbo 用于屏幕理解与推理，GPT-3.5-turbo 用于 slot 填充子任务参数|具有分层记忆结构的单智能体|提出类人应用记忆，将任务分解为模块化子任务|https://mobile-gpt.github.io|
|MM-Navigator [303]|移动端 iOS 与 Android|带 set-of-mark 标签的手机截图|可点击 UI 操作|GPT-4V|单智能体|采用 set-of-mark 提示，结合 GPT-4V 在智能手机上执行精确 GUI 导航|https://github.com/zzxslp/MM-Navigator|
|AppAgent [18]|Android 移动设备|实时截图与描述交互元素的 XML 文件|点击、长按、滑动、文本输入、返回与退出等类人操作|GPT-4V|单智能体|可通过类人交互方式在任意智能手机应用上执行任务|https://appagent-official.github.io/|
|AppAgent-V2 [304]|Android 移动设备|带标注元素的 GUI 截图、OCR 检测文本与图标、结构化 XML 元数据|点击、文本输入、长按、滑动、返回与停止等标准操作|GPT-4|含探索阶段与部署阶段的多阶段架构|通过结合结构化数据解析与视觉特征增强移动环境下的适应性与精度|/|
|FedMobileAgent [314]|Android 移动设备|GUI 截图|标准 UI 操作|Qwen2-VL-Instruct-7B [231]|多智能体联邦学习框架|引入隐私保护的联邦学习进行移动自动化，实现无需集中人工标注的大规模训练|/|

表 14：移动 GUI 智能体框架概览（二）

|智能体（Agent）|平台（Platform）|感知（Perception）|动作（Action）|模型（Model）|架构（Architecture）|亮点（Highlight）|链接（Link）|
|---|---|---|---|---|---|---|---|
|Prompt2Task [315]|Android 移动设备|GUI 结构与层级布局、整页文本描述、基于 OCR 的文本抽取|标准 UI 操作|GPT-4|多智能体架构|通过自由文本提示进行 UI 自动化，免除用户编写脚本的需求|https://github.com/PromptRPA/Prompt2TaskDataset|
|ClickAgent [312]|Android 移动设备|截图|标准 UI 操作|InternVL-2.0 [379]、TinyClick [337]、SeeClick [25]|单智能体|结合 MLLM 推理与专用 UI 定位模型，提高 UI 交互准确率|https://github.com/Samsung/ClickAgent|
|AutoDroid [156]|Android 移动设备|简化的 HTML 风格表示|标准 UI 操作|GPT-3.5、GPT-4、Vicuna-7B [510]|单智能体|利用应用特定知识与多粒度查询优化模块降低计算成本|https://autodroid-sys.github.io/|
|AutoDroid-V2 [305]|Android 移动设备|结构化 GUI 表示|由标准 UI 操作与 API 调用组成的多步脚本|Llama-3.1-8B [91]|脚本式架构|将 GUI 任务自动化转化为脚本生成问题，提高效率与任务成功率|/|
|CoAT [298]|Android 移动设备|基于截图的上下文与语义信息|标准 UI 操作|GPT-4V|单智能体|集成 chain-of-action-thought 过程，将每个动作显式映射到屏幕描述、推理步骤与预期结果|https://github.com/ZhangL-HKU/CoAT|
|Mobile-Agent [158]|Android 移动设备|包含图标检测的截图|标准 UI 操作|GPT-4V + Grounding DINO [183] + CLIP [509]|单智能体|以视觉为中心的方法，消除了对系统特定数据的依赖|https://github.com/X-PLUG/MobileAgent|
|Mobile-Agent-v2 [306]|Android 与 Harmony OS|带文本与图标识别和描述的截图|移动端标准 UI 操作|GPT-4V + Grounding DINO [183] + Qwen-VL-Int4 [511]|多智能体架构（规划、决策与反思智能体）|多智能体结构提升了长序列任务导航能力|https://github.com/X-PLUG/MobileAgent|
|Mobile-Experts [307]|Android 移动设备|界面记忆与过程记忆|标准 UI 操作与代码组合工具调用|VLMs|具有双层规划的多智能体框架|提出代码组合工具与双层规划机制，实现协同任务执行|/|
|LiMAC [308]|Android 移动设备|截图与对应控件树|标准 UI 操作|轻量级 Transformer 与微调后的 VLMs|单智能体|在计算效率与自然语言理解之间取得平衡|/|
|MobA [309]|Android 移动设备|GUI 结构、带标注截图|标准 UI 操作与 API 函数调用|GPT-4|双层智能体（全局智能体 + 局部智能体）|通过将任务规划与执行拆分为两个专门智能体，提升复杂任务处理能力|https://github.com/OpenDFM/MobA|
|Mobile-Agent-E [311]|Android 移动设备|GUI 截图，OCR 检测文本与图标|标准 UI 操作与 API 调用|GPT-4o、Claude-3.5-Sonnet、Gemini-1.5-Pro|分层多智能体系统|通过将规划与执行分离并引入自我进化能力，实现可学习复用的技巧与快捷方式|https://x-plug.github.io/MobileAgent|

表 15：移动 GUI 智能体框架概览（三）

|智能体（Agent）|平台（Platform）|感知（Perception）|动作（Action）|模型（Model）|架构（Architecture）|亮点（Highlight）|链接（Link）|
|---|---|---|---|---|---|---|---|
|ReachAgent [313]|Android 移动设备|GUI 截图与 XML 文档|标准 UI 操作|MobileVLM [353]|两阶段训练的单智能体|将任务划分为页面到达（Page Reaching）与页面操作（Page Operation）两类子任务，并通过基于偏好的 RL 提升长期任务成功率|/|
|Mobile-Agent-V [316]|Android 移动设备|视频指导、XML 层级|标准 UI 操作|GPT-4o|多智能体系统|引入视频指导学习，使智能体高效获取操作知识|https://github.com/X-PLUG/MobileAgent|
|MobileSteward [317]|Android 移动设备|XML 布局与截图|标准 UI 交互与代码执行|GPT-4V、GPT-4o|面向应用的多智能体框架|提出面向应用的多智能体框架与自我进化机制，通过动态招募专门智能体解决跨应用交互复杂性|https://github.com/XiaoMi/MobileSteward|
|AppAgentX [258]|Android 移动设备|截图|标准 UI 操作|GPT-4o|单智能体架构|引入进化机制，使智能体可以从过去交互中动态学习，并用高层动作替换低效的底层操作|https://appagentx.github.io/|
|CHOP [318]|Android 移动设备|截图|标准 UI 操作|GPT-4o|多智能体架构|提出基于子任务（basis subtask）的框架，根据人类任务分解模式预定义子任务，提升可执行性与效率|https://github.com/Yuqi-Zhou/CHOP|
|OS-Kairos [319]|Android 移动设备|GUI 截图|标准 UI 操作|OS-Atlas-Pro-7B 与 GPT-4o|带 critic-in-the-loop 设计的单智能体|提出自适应交互框架，为每个 GUI 动作分配置信度分数，并在自动执行与人工介入之间动态决策|https://github.com/Wuzheng02/OS-Kairos|
|V-Droid [320]|Android 移动设备|Android 无障碍树|标准 UI 操作|LLaMA-3.1-8B-Instruct [91]|Verifier-Driven 单智能体架构|提出新型 verifier 驱动架构，由 LLM 对有限候选动作进行评分选择，而非直接生成动作，从而提升成功率并显著降低延迟|/|
|LearnAct [321]|Android 移动设备|GUI 截图、UI 树与示范轨迹|标准 GUI 动作|Gemini-1.5-Pro、UI-TARS-7B-SFT、Qwen2-VL-7B|多智能体|提出结构化、基于示范的移动 GUI 智能体学习流水线，通过少量示例解决长尾泛化问题，在复杂真实移动任务上取得显著收益|https://lgy0404.github.io/LearnAct|
|AndroidGen [322]|Android 移动设备|XML UI 结构|标准 GUI 动作|GLM-4-9B [499] / LLaMA-3-70B [91]|多模块单智能体|通过自我改进架构与零人工标注训练流水线，从简单任务向复杂任务有效泛化，缓解 Android 智能体数据稀缺问题|https://github.com/THUDM/AndroidGen|
|Agent-Initiated Interaction [324]|Android 移动设备|无障碍树与截图|标准 GUI 操作|Gemini 1.5|单智能体架构|开创移动 UI 自动化中的“智能体发起交互”范式|https://github.com/google-research/google-research/tree/master/android_interaction|
|Latent State Estimation [325]|Android 移动设备|无障碍树|标准 GUI 操作|PaLM 2|由 Reasoner 与 Grounder 组成的双模块设计|首次用 LLM 形式化 UI 潜在状态估计问题，为 UI 自动化提供支持|/|

**6.3 计算机 GUI 智能体（Computer GUI Agents）**

计算机 GUI 智能体已经发展到能够在多种操作系统上提供复杂自动化能力，解决跨应用交互、任务泛化与高层规划等挑战。它们从简单的自动化工具演化为能够利用多模态输入、复杂架构与自适应学习，在桌面环境中执行多应用任务的智能系统。表 16 与表 17 对代表性的计算机 GUI 智能体框架进行了概览。

表 16：计算机 GUI 智能体框架概览（一）

|智能体（Agent）|平台（Platform）|感知（Perception）|动作（Action）|模型（Model）|架构（Architecture）|亮点（Highlight）|链接（Link）|
|---|---|---|---|---|---|---|---|
|UFO [19]|Windows 计算机|带标注控件与控件属性的截图|标准 UI 操作及其他定制操作|GPT-Vision|由 HostAgent（应用选择与全局规划）与 AppAgent（应用内执行任务）构成的双智能体架构|通过双智能体系统在 Windows OS 上以自然语言完成跨应用复杂任务|https://github.com/microsoft/UFO|
|UFO2 [334]|Windows 桌面|GUI 截图与文本控件属性列表|统一的 GUI–API 动作层|GPT-4o（以及 GPT-4V、o1、Gemini-Flash）；视觉 grounding 由 OmniParser-v2 提供|集中式 HostAgent + 应用专用 AppAgents|将传统 CUA 提升为深度集成 Windows 的 AgentOS，支持混合 GUI–API 动作、视觉 + UIA 感知、推测多动作规划与非侵入式 PiP 虚拟桌面|https://github.com/microsoft/UFO/|
|ScreenAgent [366]|Linux 与 Windows 桌面|截图|标准 UI 操作|ScreenAgent 模型|单智能体|集成规划—执行—反思的流水线，模拟连续思维过程|https://github.com/niuzaisheng/ScreenAgent|
|OS-Copilot [162]|Linux 与 MacOS 计算机|统一接口，包含鼠标键盘控制、API 调用、Bash/Python 解释器|标准 UI 操作、Bash/Python 命令与 API 调用|GPT-4|包含规划器、配置器、执行器与评论器模块的多组件架构|通过自我引导学习能力，在无需针对每个应用显式训练的情况下适配新应用并构建新工具|https://os-copilot.github.io/|
|Cradle [161]|Windows 计算机|完整屏幕视频，结合 Grounding DINO [183] 与 SAM [182] 进行目标检测与定位|键盘与鼠标动作|GPT-4|模块化单智能体架构|无需依赖内部 API，即可在多种软件环境中运行，涵盖游戏与专业应用|https://baai-agents.github.io/Cradle/|
|Agent S [326]|Ubuntu 与 Windows 计算机|截图与可访问性树|标准 UI 操作与系统级控制|GPT-4 与 Claude-3.5-Sonnet [163]|由 Manager 与 Worker 组成的多智能体架构|基于经验增强的分层规划|https://github.com/simular-ai/Agent-S|
|GUI Narrator [327]|Windows 计算机|高分辨率截图|标准 UI 操作|GPT-4 与 QwenVL-7B [511]|两阶段架构：首先检测光标位置并选择关键帧，然后生成动作描述|利用光标作为关注焦点，提升高分辨率 GUI 动作理解能力|https://showlab.github.io/GUI-Narrator|
|PC-Agent [329]|Windows 计算机|截图与事件轨迹|标准 UI 操作|Qwen2-VL-72B-Instruct [231] 与 Molmo [512]|由决策规划智能体与 grounding 智能体组成的架构|提出“人类认知迁移”框架，将原始交互数据转化为认知轨迹，以支持复杂电脑任务|https://gair-nlp.github.io/PC-Agent/|

表 17：计算机 GUI 智能体框架概览（二）（传统 CUA 等）

|智能体（Agent）|平台（Platform）|感知（Perception）|动作（Action）|模型（Model）|架构（Architecture）|亮点（Highlight）|链接（Link）|
|---|---|---|---|---|---|---|---|
|Zero-shot Agent [328]|计算机|HTML 代码与 DOM|标准 UI 操作|PaLM-2 [507]|单智能体|在计算机控制任务上具备零样本能力|https://github.com/google-research/google-research/tree/master/zero_shot_structured_reflection|
|PC-Agent [330]|Windows 计算机|UI 树与截图|标准 UI 操作|GPT-4o|分层多智能体|通过分层多智能体设计高效分解复杂 PC 任务，其主动感知模块结合可访问性结构、OCR 与意图 grounding 提升细粒度 GUI 理解|https://github.com/X-PLUG/MobileAgent/tree/main/PC-Agent|
|PwP [331]|计算机中的 VSCode IDE|截图、文件系统访问、终端输出|标准 UI 交互、文件操作、Bash 命令与 VSCode 内工具|GPT-4o、Claude-3.5-Sonnet、Gemini-1.5|单智能体架构|将软件工程智能体从 API 调用转向直接使用 GUI 与 IDE 交互，使其像人类开发者一样操作 IDE|https://programmingwithpixels.com|
|COLA [332]|Windows 计算机|GUI 结构、属性与截图|标准 UI 操作与系统 API|GPT-4o|分层多智能体|提出具备插拔式智能体池的动态任务调度机制，可自适应处理多种 GUI 任务|https://github.com/Alokia/COLA-demo|
|STEVE [333]|Windows 桌面|GUI 截图与无障碍树（A11y Tree）|标准 UI 操作|Qwen2-VL [231] 与 GPT-4o|单智能体|引入可扩展的动作验证流水线，使用 GPT-4o 为智能体动作生成二元标签，并采用 KTO 优化同时利用正负样本|https://github.com/FanbinLu/STEVE|
|TaskMind [335]|Windows 计算机|标准 GUI 动作|GPT-3.5 / GPT-4|单智能体架构|提出带认知依赖的任务图表示，使 LLM 更好地泛化演示过的 GUI 任务|https://github.com/Evennaire/TaskMind|

在这些框架中，UFO [19] 是针对 Windows 操作系统设计的开创性 UI 聚焦智能体。它通过双智能体架构重新定义了 UI 自动化：HostAgent 负责全局规划、任务拆解与应用选择，AppAgent 则在具体应用内执行子任务（如图 23 所示）。这种集中式结构使 UFO 能够管理跨应用的复杂工作流（如信息汇总与报告生成）。在此基础上，UFO[2] [334] 将 UFO 从纯视觉原型扩展为深度集成 Windows 的 AgentOS（图 24），通过融合 UIA 元数据与 OmniParser-v2 视觉 grounding 进行混合 GUI–API 控制，并使用 PiP 虚拟桌面实现与用户会话的非侵入式隔离。实验表明，UFO[2] 在 20 余个真实 Windows 应用上显著优于 Operator [513] 等系统，并因模型无关的设计可以通过更强 LLM 获得进一步提升。

Cradle [161] 进一步推动了通用计算机控制的边界，它利用 VLM 在无需 API 访问的情况下与多种软件（从游戏到专业应用）交互。借助 GPT-4o 解释屏幕输入并执行底层动作，以及由信息收集、自我反思等六个模块组成的结构，Cradle 能在动态环境中完成任务、对动作进行推理，并利用历史交互指导未来决策。

OS-Copilot [162] 则面向多操作系统（Linux 与 macOS）提出了通用框架，其 FRIDAY 组件展示了自我引导学习在未见应用上的潜力。与面向单应用的智能体不同，FRIDAY 通过集成 API、键盘鼠标控制与命令行操作，构建了一个能够在交互中自动生成并精炼工具的灵活平台。其在 GAIA 基准上的表现表明，OS-Copilot 为复杂环境中的 OS 级智能体提供了坚实基础。

图 23 展示了 UFO 的多智能体架构：HostAgent 接收用户请求并拆分为多个子任务，然后将它们分配给对应应用的 AppAgent（如 PowerPoint、Edge 浏览器、Excel 与 Word），实现跨应用协同。图 24 对比了传统 CUA 与 Desktop AgentOS UFO[2]：后者通过深度 OS 集成、GUI–API 混合动作与 PiP 体验，显著提升了可靠性与用户体验。
在新兴的跨平台 GUI 智能体领域，许多工作通过统一架构在 Web、移动端与桌面端之间实现无缝交互。AutoGLM [336] 通过将大规模多模态模型与中间接口设计（Intermediate Interface Design）结合起来，在 Web 浏览与 Android 控制之间架起桥梁。它将规划与 grounding 任务解耦，从而提升动态决策与适应性，并通过带强化学习的自进化在线课程学习框架，持续从真实环境反馈中学习、纠错与改进，成为跨平台自动化与基础智能体研究中的重要代表。

TinyClick [337] 则聚焦在单轮 GUI 交互这一轻量场景，针对参数量与资源受限的问题提出解决方案。TinyClick 使用 Florence-2-Base 视觉语言模型，在仅有 2.7 亿参数的前提下，基于用户指令与截图执行任务，在 ScreenSpot [25] 上取得 73% 的准确率、在 OmniAct [459] 上取得 58.3% 的准确率，超过了包括 GPT-4V 在内的更大规模多模态模型。其多任务训练与基于 MLLM 的数据增强机制实现了精确的 UI 元素定位，非常适合低资源环境与对时延敏感的 UI grounding 与动作执行场景。

在多智能体架构方面，OSCAR [340] 提出了一种通用 GUI 智能体，可自主导航并控制桌面与移动应用。通过状态机架构，OSCAR 能在执行过程中动态处理错误并根据实时反馈调整动作，适用于在自然语言引导下自动化复杂工作流。标准化的 OS 控制（键盘、鼠标等）集成使其在多种 GUI 环境中实现通用交互。

AgentStore [341] 则提出了可扩展的异构智能体集成框架，可在多个操作系统上自动化任务。其核心组件 MetaAgent 使用 AgentToken 策略动态管理数量日益增多的专门智能体，通过动态招募智能体实现系统的适应性与可扩展性，使专门能力与通用能力在同一框架中共存。该多智能体架构支持桌面与移动等多种平台，并利用 GUI 结构、系统状态等多模态感知进行决策。

MMAC-Copilot [249] 使用多智能体、多模态方案来处理 3D 游戏、办公与移动应用中的任务，而无需依赖 API。通过 Planner、Viewer、Programmer 等专门智能体协同工作，MMAC-Copilot 能适应视觉上高度复杂的环境。系统利用 GPT-4V 进行视觉识别、使用 OCR 进行文本分析，在非 API 应用基准 VIBench 上取得高任务完成率，显示其在游戏、医疗与生产力等行业的潜力。

AGUVIS [236] 采用纯视觉方法自动化 GUI 交互，绕过 HTML 或无障碍树等基于文本的系统限制。其平台无关设计同时支持 Web、桌面与移动应用，并通过两阶段训练（第一阶段聚焦 GUI grounding，第二阶段在统一模型中集成规划与推理），在离线与在线场景中均达到 SOTA 表现。

Agent S2 [344] 在 Agent S [326] 的基础上，引入分层与组合式架构，将通用模型与专门 grounding 模块结合。它采用 Mixture of Grounding（MoG）策略，将精细的 grounding 子任务委派给专家模块，并通过 Proactive Hierarchical Planning（PHP）在观测变化时动态修订动作计划。仅依赖 GUI 截图的 Agent S2 在 Ubuntu、Windows 与 Android 平台上都表现出良好泛化能力，通过模块化设计在长时序、高精度 GUI 交互中优于更大规模的单体模型。

表 18：跨平台 GUI 智能体框架概览

|智能体（Agent）|平台（Platform）|感知（Perception）|动作（Action）|模型（Model）|架构（Architecture）|亮点（Highlight）|链接（Link）|
|---|---|---|---|---|---|---|---|
|AutoGLM [336]|Web 与 Android 移动端|带 SoM 标注与 OCR 的截图|标准 UI 操作、原生 API 交互与 AI 驱动动作|ChatGLM [499]|单智能体架构|通过自进化在线课程 RL 框架在真实环境中持续学习与改进|https://xiao9905.github.io/AutoGLM/|
|TinyClick [337]|Web、移动端与 Windows 平台|GUI 截图|标准 UI 操作、原生 API 交互与 AI 驱动动作|Florence-2-Base VLM [514]|面向单轮任务的单智能体|在仅 0.27B 参数规模下取得高性能，适合资源受限场景|https://huggingface.co/Samsung/TinyClick|
|OSCAR [340]|桌面与移动|截图|标准 UI 操作|GPT-4|单智能体|能根据实时反馈自适应调整动作，适用于复杂多步工作流|
|AgentStore [341]|桌面与移动环境|GUI 结构与属性、可访问性树、截图、终端输出等|标准 UI 操作与 API 调用|GPT-4o 与 InternVL2-8B [379]|多智能体架构|可动态集成多种异构智能体，使专门与通用能力共存|https://chengyou-jia.github.io/AgentStore-Home|
|MMAC-Copilot [249]|Windows 桌面、移动应用与游戏环境|截图|标准 UI 操作、原生 API 与多智能体协同动作|GPT-4V、SeeClick [25] 与 Gemini Vision（不同智能体使用）|包含 Planner、Programmer、Viewer、Mentor、Video Analyst、Librarian 等角色的多智能体架构|面向 3D 游戏、办公与移动等复杂场景的协作式多智能体方案|
|AGUVIS [236]|Web、桌面与移动|基于图像的观测|标准 UI 操作|微调后的 Qwen2-VL [231]|单智能体架构|纯视觉 GUI 交互方法，绕过文本 UI 表示，实现跨平台稳健泛化|https://aguvis-project.github.io|
|Ponder & Press [342]|Web、Android、iOS、Windows 与 macOS|纯视觉输入|标准 UI 操作|GPT-4o 与 Claude 3.5 Sonnet 用于高层任务分解；微调 Qwen2-VL-Instruct [231] 用于 GUI grounding|分而治之（divide-and-conquer）架构|无需非视觉输入的纯视觉 GUI 智能体|https://invinciblewyq.github.io/ponder-press-page/|
|InfiGUIAgent [343]|移动端、Web 与桌面|原始截图|标准 UI 操作|Qwen2-VL-2B [231]|带分层推理的单智能体|引入分层与预期-反思等原生推理技能，提高任务适应性|https://github.com/Reallm-Labs/InfiGUIAgent|
|Learn-by-Interact [338]|Web、代码开发与桌面|带 SoM 与可访问性树的 GUI 截图|标准 UI 交互与代码执行|Claude-3.5-Sonnet、Gemini-1.5-Pro [92]、CodeGemma-7B、CodeStral-22B|多智能体|提出全自动数据合成过程，消除对人工标注 agent 数据的依赖|/|
|CollabUIAgents [339]|Android 移动与 Web|截图与 UI 树|标准 UI 操作|Qwen2-7B [231]、GPT-4|多智能体系统|提出多智能体 RL 框架与 Credit Re-Assignment 策略，用 LLM 代替环境特定奖励，提升性能与泛化|https://github.com/THUNLP-MT/CollabUIAgents|
|Agent S2 [344]|Ubuntu、Windows 与 Android|GUI 截图|标准 UI 操作与系统 API|Claude-3.7-Sonnet、Claude-3.5-Sonnet、GPT-4o（用于 Manager 与 Worker），UI-TARS-72B-DPO、Tesseract OCR 与 UNO（用于 grounding 专家）|由 Manager（规划）、Worker（执行）与 Mixture of Grounding 专家组成的组合式多智能体架构|采用 Mixture of Grounding 与 Proactive Hierarchical Planning，在长时序任务中实现更精准的 grounding 与自适应重规划|https://github.com/simular-ai/Agent-S|
|GuidNav [345]|Android 与 Web|GUI 截图|标准 UI 操作与系统 API|GPT-4o、Gemini 2.0 Flash、Qwen-VL-Plus|单智能体|提出细粒度过程奖励模型，为每步动作提供反馈以提升任务准确性与成功率|/|
|ScaleTrack [346]|Web、Android 移动与桌面|GUI 截图|标准 GUI 操作|Qwen2-VL-7B|单智能体|首个在 GUI 智能体框架中系统引入回溯（backtracking）的工作，同时学习下一步动作与历史动作序列|/|

总结来看，跨平台 GUI 智能体从轻量级模型（如 TinyClick）到复杂多智能体系统（如 MMAC-Copilot、Agent S2），共同构成了一个多样化的自动化工具生态，大幅提升了在不同平台间的交互能力。

**6.5 关键总结（Takeaways）**

GUI 智能体框架在多智能体架构、多模态输入与动作空间扩展等方面取得了显著进展，为构建更通用、更强大的智能体奠定了基础。主要结论包括：

1) 多智能体协同（Multi-Agent Synergy）：以 UFO [19] 与 MMAC-Copilot [249] 为代表的多智能体系统，通过在框架内为不同智能体分配专门角色，提升任务效率、适应性与整体性能。随着任务在多平台上愈发复杂，协同使用多个智能体已成为应对复杂工作流的关键途径。
2) 多模态输入优势（Multimodal Input Benefits）：与仍然仅依赖文本输入（如 DOM 或 HTML）的智能体相比，加入截图等视觉输入明显提升性能。WebVoyager [269] 与 SeeAct [17] 等工作表明，视觉与文本结合可提供更丰富的环境状态表示，帮助智能体在视觉复杂或动态环境中做出更合理的决策。
3) 超越 UI 操作的动作集（Expanding Action Sets）：近年来的智能体将动作集从标准 UI 操作扩展到 API 调用与 AI 工具调用，如 Hybrid Agent [199] 与 AutoWebGLM [270]。多样化的动作空间使智能体在可直接通过 API 获取或操作数据的场景中提升交互能力与任务完成效率。
4) 面向更好决策的技术（Emerging Techniques for Improved Decision-Making）：如 WMA [266] 中的世界模型与 Search-Agent [274] 中的搜索式策略，为高级决策提供了有前景的方向。世界模型可以在执行前模拟动作结果，减少不必要交互并提高长时序任务效率；而最佳优先搜索、MCTS 等算法帮助智能体在复杂实时环境中更有效地探索动作路径。
5) 跨平台泛化（Toward Cross-Platform Generalization）：AutoGLM [336] 与 OSCAR [340] 等跨平台框架凸显了在 GUI 智能体设计中实现广泛泛化的重要性。此类智能体致力于在移动、桌面与 Web 平台间实现统一方案，朝着“可跨多生态一致工作的通用 GUI 智能体”目标迈进。
6) 纯视觉智能体（Pure Vision-Based Agent）：为实现通用 GUI 控制，纯视觉框架逐渐成为重要方向。这类智能体仅依赖截图进行决策，无需访问控件树或元素属性等元数据。AGUVIS [236]便是典型示例。纯视觉方法具有更强的通用性，能够绕过系统 API 限制，但也对“grounding”能力提出更高要求，需要精准定位并交互 UI 元素。通过针对视觉 grounding 与 GUI 理解进行微调，或集成 OmniParser [184] 等 GUI 解析技术，可以有效缓解这一难题。

总体来看，GUI 智能体正朝着多智能体、多模态、多样动作空间与新型决策策略方向演进，这些趋势共同推动智能体在多样、动态环境中实现高性能与高适应性，为未来更通用、更具人性化的跨平台智能体奠定基础。

###### 7 优化以 LLM 为大脑的 GUI 智能体的数据（DATA FOR OPTIMIZING LLM-BRAINED GUI AGENTS）

在前一节中，我们回顾了以 LLM 为大脑的 GUI 智能体的一般框架，这些框架多依赖 GPT-4V、GPT-4o 等基础 LLM。若要进一步提升智能体的性能与效率，优化其“脑”（即底层模型）至关重要。这通常需要用大规模、多样且高质量的 GUI 上下文数据集 [515] 对基础模型进行微调，使其在 GUI 特定任务上具备更强能力。收集此类数据集，尤其是包含大量 GUI 截图、元数据与交互信息的数据集，需要在数据采集、过滤与预处理等环节投入大量精力与资源 [516]。

随着 GUI 智能体受到越来越多关注，研究者开始着力构建覆盖不同平台、能够反映 GUI 复杂性的数据集。这些数据集通过覆盖多样界面、工作流与用户交互，为模型在广泛场景中实现有效泛化提供了基础。为实现全面覆盖，研究者在数据采集与结构化过程中采用了多种创新方法。接下来，我们将介绍面向 GUI 专用 LLM 训练的数据采集与处理端到端流程，并对来自不同平台的代表性数据集进行综述，说明其特性、构建方法以及在推动以 LLM 为大脑的 GUI 智能体发展方面的潜在作用。

**7.1 数据采集（Data Collection）**

数据对于训练专用 GUI 智能体至关重要，但由于任务本身复杂且涉及环境多样，数据采集往往耗时且费力。

_7.1.1 数据构成与来源（Data Composition and Sources）_

用于 GUI 智能体训练的关键数据组件与第 5.2.2 节与第 5.4 节中讨论的感知与推理需求高度一致。从高层来看，这些数据主要包括：

1) 用户指令（User Instructions）：以自然语言给出任务的整体目标、目的与细节，为智能体提供明确的完成目标，例如：“将所有文本的字体大小改为 12”。
2) 环境感知（Environment Perception）：通常包括 GUI 截图（往往带有各种视觉增强）以及用于丰富上下文的可选补充数据，如控件树与 UI 元素属性。它既应覆盖静态环境状态（第 5.2.2 节）也应覆盖表示动作后变化的动态环境反馈（第 5.2.3 节），以提供足够的上下文信息。
3) 任务轨迹（Task Trajectory）：包含完成任务所需的关键动作序列，以及补充信息（如智能体的计划）。一条轨迹通常由多步动作组成，用于在环境中逐步实现目标。

其中，用户指令与环境感知作为模型输入，而期望的模型输出则是任务轨迹，其动作序列将在环境中落地执行以完成任务。

为了让用户指令更贴近真实场景，需确保其具有现实性与代表性。指令的来源通常包括：（i）直接由人类设计者提供，基于真实应用场景给出任务；（ii）从已有相关数据集中抽取；（iii）从公开资源中获取，如网站、应用帮助文档及其他公开材料；以及（iv）由 LLM 生成，模拟不同场景下的广泛用户请求。此外，还可以使用 LLM 进行数据增强 [517]，在保持原始语义的同时提升指令的质量与多样性。

在环境感知数据采集方面，可使用第 5.2.2 节中讨论的各种工具包捕获 GUI 数据。可以在环境模拟器中操作（如 Android Studio Emulator[29]、Selenium WebDriver[30]、Windows Sandbox[31]），也可以直接在真实环境中采集，获取截图、控件树及其他元数据。

任务轨迹的收集通常是最具挑战性的部分。任务轨迹必须准确、可执行且经过充分验证。常见方法包括：（i）使用程序自动生成的脚本，为预定义任务生成动作序列，提供高度可控的数据；（ii）通过众包等方式让人工标注者执行任务，并逐步记录每个动作，从而获得丰富且真实的动作数据；（iii）利用模型或智能体自举（bootstrapping）[518]，让已有 LLM 或 GUI 智能体尝试完成任务并记录动作轨迹，但这种方法通常需要额外验证以过滤潜在错误。这些方法都需要较大投入，反映了为 GUI 智能体训练收集可靠轨迹数据的复杂性。

_7.1.2 数据采集流水线（Collection Pipeline）_

图 25 展示了用于训练 GUI 智能体模型的数据采集完整流程。流程首先从收集初始用户指令开始，这些指令来源如前所述，通常是“原型式”（prototypical）的，尚未针对具体环境进行定制或落地 [374]。例如，来自某通用网站的“如何修改字体大小？”这一指令并不具有环境特异性，也不对应用户在特定应用中的具体请求。为此，需要进行“实例化”（instantiation）步骤 [374]，即将指令嵌入具体环境，使其更具可执行性。例如，可将该指令实例化为：“将 Word 文档 draft.docx 中第三段文字的字体大小修改为 20。”，使其具有明确、与环境相关的目标。实例化既可以由人工完成，也可以通过 LLM 程序化生成。

在实例化之后，指令通常会经过过滤步骤，以去除质量较差的数据，保留与任务相关且可执行的指令；同时还可以通过数据增强扩展与丰富数据集，以提升模型鲁棒性。过滤与增强都可以结合人工验证或借助 LLM 提升效率。

完成指令精炼后，开始同步收集任务轨迹与环境感知数据。在环境中执行动作的过程中，每一步都会被记录，包括环境状态与具体动作。完整轨迹记录完成后，需要进行评估，以识别并剔除失败或不准确的序列，从而保证数据集质量。通过迭代这一流水线，可以构建出高质量的 GUI 智能体训练数据集，为训练优化模型奠定基础。

图 25：用于训练 GUI 智能体模型的数据采集完整流程示意图。上半部分展示了状态-动作数据的采集与任务实例化，下半部分展示了任务执行、轨迹记录与评估的过程。

接下来，我们将回顾不同平台上的 GUI 智能体数据集，并讨论当前实践以及潜在的改进方向。

**7.2 Web 智能体数据（Web Agent Data）**

Web 场景下的 GUI 智能体需要数据集来刻画真实 Web 交互的复杂性与多样性。这些数据集通常包含多样网站结构（如 DOM 树与 HTML 内容）以及反映真实用户导航与交互模式的多步任务标注。构建能在不同网站间泛化并执行复杂任务的智能体，需要具备丰富上下文信息的综合性数据集。表 19 总结了代表性 Web GUI 智能体数据集。

Mind2Web [212] 是此类数据集中的重要代表，它强调开放式任务描述，推动智能体自主理解高层目标。该数据集在 137 个网站上提供了超过 2,350 个由人工标注的任务，涵盖 Web 导航中复杂的交互模式与序列，有助于评估智能体在未见领域中的泛化能力，是 Web GUI 语言 grounding 的重要基准。

WebVLN [347] 将导航任务与问答任务结合，要求智能体在网站中根据文本查询定位相关页面并提取信息。通过同时利用 HTML 与网站视觉内容，WebVLN 更贴近现实 Web 浏览挑战，对希望构建具备复杂、人类般 Web 交互能力的智能体具有较高研究价值。

WebLINX [348] 则聚焦于对话式 GUI 智能体，尤其是通过多轮对话实现的真实 Web 导航。该数据集在 155 个真实网站上提供了超过 2,300 条专家示范，并包含 DOM 树与截图，用于训练和评估能够执行动态、由用户引导的导航任务的智能体。丰富的动作与对话数据有助于提升智能体在真实 Web 场景中的响应能力与泛化能力。

表 19：Web GUI 智能体数据集概览（节选）

|数据集（Dataset）|平台（Platform）|来源（Source）|内容（Content）|规模（Scale）|采集方式（Collection Method）|亮点（Highlight）|链接（Link）|
|---|---|---|---|---|---|---|---|
|Mind2Web [212]|Web|众包（Crowdsourced）|任务描述、动作序列、网页快照|2,350 个任务，覆盖 137 个网站|人工示范|面向真实网站、具有多样用户交互的通用 Web 智能体基准|https://osu-nlp-group.github.io/Mind2Web/|
|Mind2Web-Live [349]|Web|从 Mind2Web [212] 采样并重新标注|文本任务描述、中间评估状态、动作序列与元数据、GUI 截图|542 个任务，包含 4,550 个详细标注步骤|由领域专家标注|通过“关键节点（key nodes）”引入动态评估，关注 Web 任务中的关键中间状态|https://huggingface.co/datasets/iMeanAI/Mind2Web-Live|
在表 19 的基础上，Web 场景下还有若干重要数据集，补充如下（表 19 续）：

表 19（续）：Web GUI 智能体数据集概览

|数据集（Dataset）|平台（Platform）|来源（Source）|内容（Content）|规模（Scale）|采集方式（Collection Method）|亮点（Highlight）|链接（Link）|
|---|---|---|---|---|---|---|---|
|WebVLN [347]|Web|人工设计 + LLM 生成|文本指令、规划、GUI 截图、HTML 内容|8,990 条导航路径，14,825 个问答对|基于 WebVLN 模拟器，由 LLM 生成 QA 对|面向类人 Web 浏览的视觉-语言导航基准|https://github.com/WebVLN/WebVLN|
|WebLINX [348]|Web|人类专家|对话交互、动作序列、DOM 与截图|2,337 条示范，含 10 万余次交互|由人类专家标注|首个用于评估真实对话式 Web 导航智能体的大规模数据集|https://mcgill-nlp.github.io/weblinx/|
|AgentTrek [350]|Web|Web 教程|任务元数据、分步指令、动作序列、视觉观测、可复现原生轨迹|4,902 条轨迹|由 VLM 智能体在教程指导下收集，Playwright 记录轨迹|利用 Web 教程合成高质量轨迹数据|/|
|MultiUI [220]|Web|人工指令 + 从 Web 结构自动抽取|文本任务描述、规划、动作序列、GUI 截图、可访问性树、边界框标注|来自 100 万网站的 730 万条指令样本|LLM 与 Playwright|支持 GUI 理解、动作预测与元素 grounding 等广泛 UI 任务|https://neulab.github.io/MultiUI/|
|Explorer [290]|Web|热门 URL + LLM 系统性探索|文本任务描述、动作序列、GUI 截图、可访问性树、HTML 内容|94K 条成功 Web 轨迹、49K 个唯一 URL、72 万张截图|由多智能体 LLM 管道自动生成|迄今规模最大的 Web 轨迹数据集之一，动态探索网页构造上下文相关任务|/|
|InSTA [351]|Web|由 LLM 在 150K 个网站上自动生成|自然语言 Web 导航任务、任务规划与动作序列、转为 Markdown 的 HTML 观测、由 LLM 裁判的评估信息|150,000 个任务，覆盖 150,000 个网站|通过 LLM 调用 Playwright API 自动生成并由 LLM 裁判过滤|完全自动化三阶段数据生成流水线（任务生成、动作执行与评估），无需人工标注即可获得互联网规模数据|https://data-for-agents.github.io|

对于移动端数据，文章中给出的表 20 与表 21 我们整理如下。

表 20：移动 GUI 智能体数据集概览（一）

|数据集（Dataset）|平台（Platform）|来源（Source）|内容（Content）|规模（Scale）|采集方式（Collection Method）|亮点（Highlight）|链接（Link）|
|---|---|---|---|---|---|---|---|
|VGA [354]|Android 移动端|基于 Rico [355]|GUI 截图、任务描述、动作序列、元素边界、布局与功能信息|63.8k 个样本，22.3k 条指令跟随数据、41.4k 条对话数据|由 GPT-4 生成|强调视觉内容，以减少不准确性，面向视觉 GUI 助理|https://github.com/Linziyang1999/Vision%2DGUI%2Dassistant|
|Rico [355]|Android 移动端|来自 Google Play 商店真实应用|文本数据、截图、动作序列、UI 结构、带注释 UI 表示|72,219 个独立 UI 屏幕、10,811 条用户交互轨迹|众包与自动探索|移动 UI 设计、交互建模与布局生成的重要基础数据集|http://www.interactionmining.org/|
|PixelHelp [144]|Android 移动端|人工、“How-to” 网站与 Rico UI 语料综合 + 合成|自然语言指令、动作序列、GUI 截图、结构化 UI 数据|187 条多步指令、295,476 条合成单步命令|人工标注与合成生成|首次系统地将自然语言指令绑定为可执行移动 UI 动作的数据集|https://github.com/google-research/google-research/tree/master/seq2act|
|MoTIF [356]|Android 移动端|人工撰写|自然语言指令、动作序列、GUI 截图、结构化 UI 数据|6,100 个任务，覆盖 125 个 Android 应用|人工标注|关注交互式 GUI 任务可行性预测|https://github.com/aburns4/MoTIF|
|META-GUI [357]|Android 移动端|SMCalFlow [519]|对话、多轮动作序列、截图、Android 视图层级|1,125 段对话与 4,684 个对话轮次|人工标注|无需依赖后端 API 的移动 GUI 任务导向对话系统数据集|https://x-lance.github.io/META-GU|
|AITW [358]|Android 移动端|人工指令 + LLM 提示|自然语言指令、UI 截图、观测-动作对|715,142 个 episode、30,378 条唯一指令|人工评审者在 Android 模拟器中采集|覆盖广泛应用与 UI 状态的大规模设备控制数据集|https://github.com/google-research/google-research/tree/master/android_in_the_wild|
|GUI-Xplore [365]|Android 移动端|自动探索 + 人工设计结合|探索视频、文本任务、QA 对、视图层级、GUI 截图、动作序列与 GUI 转移图|312 个应用、115 小时视频、32,569 个 QA 对、41,293 个动作、每个应用约 200 页|自动 + 人工探索|提出基于探索的预训练范式，通过视频数据提供丰富的应用先验|https://github.com/921112343/GUI-Xplore|

表 21：移动 GUI 智能体数据集概览（二）

|数据集（Dataset）|平台（Platform）|来源（Source）|内容（Content）|规模（Scale）|采集方式（Collection Method）|亮点（Highlight）|链接（Link）|
|---|---|---|---|---|---|---|---|
|GUI Odyssey [359]|Android 移动端|人工设计 + GPT-4|文本任务、规划、动作序列、GUI 截图|201 个应用上的 7,735 个 episode|人工示范|聚焦移动设备上的跨应用导航任务|https://github.com/OpenGVLab/GUI-Odyssey|
|Amex [360]|Android 移动端|人工设计 + ChatGPT 生成|文本任务、动作序列、高分辨率截图、多层标注|104,000 张截图、1.6M 交互元素、2,946 条指令|人工标注 + 自动脚本|多层次、大规模标注数据，支持复杂移动 GUI 任务|https://yuxiangchai.github.io/AMEX/|
|Ferret-UI [352]|iOS 与 Android 移动端|Spotlight 数据集 + GPT-4|文本任务、动作规划、GUI 元素标注与边界框|40,000 个基础任务、10,000 个高级任务|由 GPT-4 生成|面向 UI 中心任务、支持可调屏幕宽高比的基准|https://github.com/apple/ml-ferret|
|AITZ [298]|Android 移动端|AITW [358]|屏幕-动作对与动作描述|跨 70+ 应用的 18,643 个屏幕-动作对，2,504 个 episode|GPT-4V 与图标检测模型|提出结构化“Chain-of-Action-Thought”，增强 GUI 导航|https://github.com/IMNearth/CoAT|
|Octo-planner [361]|Android 移动端|GPT-4 生成|文本任务、分解计划、动作序列|1,000 条数据|由 GPT-4 生成|面向 GUI 动作的任务规划优化|https://huggingface.co/NexaAIDev/octopus-planning|
|E-ANT [362]|Android 小程序|真实人类行为|任务描述、截图、动作序列、页面元素数据|40,000+ 条轨迹、10,000 个动作意图|人工标注|首个大规模中文 GUI 导航数据集，包含真实人类交互|/|
|Mobile3M [353]|Android 移动端|真实交互 + 模拟|UI 截图、XML 文档、动作序列|3,098,786 个 UI 页面、20,138,332 个动作|仿真算法|大规模中文移动 GUI 数据集，提供独特导航图|https://github.com/Meituan-AutoML/MobileVLM|
|AndroidLab [363]|Android 移动端|人工设计、LLM 自探索与学术数据集|文本指令、动作序列、XML 数据与截图|10.5k 条轨迹、94.3k 步|人工标注 + LLM 自探索|基于 XML 的统一动作空间交互数据集|https://github.com/THUDM/Android-Lab|
|MobileViews [364]|Android 移动端|LLM 增强的应用遍历工具|截图-视图层级对|来自 20,000+ 应用的 600,000 对截图与视图层级|LLM 增强爬虫|目前最大的开源移动屏幕数据集|https://huggingface.co/datasets/mllmTeam/MobileViews|
|FedMABench [452]|Android 移动端|AndroidControl [515]、AITW [358]|文本任务描述、动作序列与 GUI 截图|6 组数据集系列，30+ 子集|从现有 Android 数据集中推断构建|首个面向联邦移动 GUI 智能体的基准数据集|https://github.com/wwh0411/FedMABench|

对于桌面与跨平台数据，我们已在前文给出了表 22（ScreenAgent、LAM、DeskVision）以及表 23/表 24（VisualAgentBench、GUICourse、GUI-World、ScreenAI、OmniParser、Web-Hybrid、GUIDE、xLAM、Insight-UI、OS-Genesis），这些表格与对应文字说明共同构成了对计算机端与跨平台 GUI 数据的完整翻译与整理。
补充跨平台 GUI 智能体数据集（对应原文表 23/24 中的额外条目）：

表 24（续）：跨平台 GUI 数据集概览

|数据集（Dataset）|平台（Platform）|来源（Source）|内容（Content）|规模（Scale）|采集方式（Collection Method）|亮点（Highlight）|链接（Link）|
|---|---|---|---|---|---|---|---|
|Navi-plus [377]|Web 与 Android|AndroidControl [515] 与 Mind2Web [212]|任务描述、GUI 动作轨迹、细粒度步骤指令、截图以及 ASK/SAY 跟进交互对|/|由 LLM 自动生成并结合人工验证|提出“自纠正”GUI 导航任务，引入 ASK 动作用于在信息缺失时主动查询|/|
|Explorer [376]|Web 与 Android|对真实网站与 Android 应用的自动遍历|UI 截图、可交互元素边界框、屏幕相似度标签与用户动作|KhanAcademy（Web）：2,841 个可交互元素、378 个屏幕相似样本；Spotify（Android）：1,207 个可交互元素、451 个屏幕相似样本|自动化工具 + HTML 解析 + 可访问性树|平台无关，支持自动标注，并可记录轨迹与语音控制 GUI 导航|https://github.com/varnelis/Explorer|

上述补充条目与此前的跨平台数据表（ScreenAI、VisualAgentBench、GUI-World、xLAM、OS-Genesis 等）共同构成了完整的跨平台 GUI 数据集翻译。

接下来补充原文表 25 中基础模型的结构化信息，方便对比用于 GUI 智能体的基础模型。

表 25：面向 GUI 智能体的代表性基础模型概览

|模型（Model）|模态（Modality）|规模（Model Size）|架构（Architecture）|训练方式（Training Methods）|亮点（Highlights）|是否开源（Open-Source）|链接（Link）|
|---|---|---|---|---|---|---|---|
|GPT-4o [93]|文本、音频、图像、视频|-|多模态自回归架构|在公共数据混合语料上预训练，之后通过对齐人类偏好与安全约束进行微调|统一多模态架构，可在文本、音频、图像与视频间无缝处理与生成内容，相比前代具更快、更低成本的推理|否|/|
|GPT-4V [378]|文本与图像|-|-|在大规模图文数据上预训练，并通过人类反馈强化学习（RLHF）进行微调|具备强大的多模态能力，可同时理解图像与文本，是图文混合任务的重要基线|否|/|
|Gemini [92]|文本、图像、音频与视频|Nano 版本：1.8B/3.25B|增强版 Transformer 解码器|在多模态数据上大规模预训练，随后进行监督微调、奖励建模与 RLHF|在多模态任务上取得 SOTA 表现（MMLU 达 90%），并支持小模型在端侧部署|否|/|
|Claude 3.5 Sonnet（Computer Use）[163], [164]|文本与图像|-|基于 ReAct 的推理架构|-|首个以“纯视觉”范式实现桌面任务自动化的公开 beta 模型，开创 GUI 自动化新方向|否|/|
|Operator [165], [513]|文本与图像|-|构建在 GPT-4o 之上|结合监督学习与强化学习|能够像人类一样使用计算机，在 GUI 任务基准上表现突出|否|/|
|Qwen-VL [210]|文本与图像|9.6B|视觉编码器采用 ViT [522]，语言部分基于 Qwen-7B|两阶段预训练 + 一阶段指令微调|在多项视觉-语言基准上取得 SOTA，对细粒度视觉理解能力强|是|https://github.com/QwenLM/Qwen-VL|
|Qwen2-VL [231]|文本、图像与视频|2B/7B/72B|ViT 视觉编码器 + Qwen2 系列语言模型|先用图文对训练 ViT，随后在多模态数据上解冻全参数训练，并通过指令数据集进行微调|提出 Naive Dynamic Resolution 与多模态旋转位置编码（Multimodal RoPE），高效处理不同分辨率图像与长视频|是|https://github.com/QwenLM/Qwen2-VL|
|InternVL-2 [379], [380]|文本、图像、视频与医疗数据|1B/2B/4B/8B/26B/40B|ViT 视觉编码器 + LLM 语言模型|采用“粗到细”的渐进式对齐策略，先在粗粒度数据上预训练，再在精细数据上对齐|在多种规模下处理复杂多模态任务表现强劲，适合不同算力场景|是|https://internvl.github.io/blog/2024-07-02-InternVL-2.0/|
|CogVLM [381]|文本与图像|17B|ViT 编码器 + 两层 MLP 适配器 + 预训练 LLM + 视觉专家模块|第一阶段聚焦图像字幕任务，第二阶段结合图像字幕与指代理解任务进行训练|在保持大模型语言能力的前提下，实现视觉与语言特征的深度融合|是|https://github.com/THUDM/CogVLM|
|Ferret [382]|文本与图像|7B/13B|基于 Vicuna 的 decoder-only 架构 + 视觉编码器|结合监督训练与指令微调|通过混合区域表示（hybrid region representation）处理自由区域输入，具备灵活的空间理解与 grounding 能力|是|https://github.com/apple/ml-ferret|
|LLaVA [217]|文本与图像|7B/13B|视觉编码器（CLIP ViT-L/14）+ 语言解码器（Vicuna）|使用过滤后的图文对预训练，对多模态指令跟随数据进行微调|轻量化架构便于快速实验，在多模态推理上接近 GPT-4 的能力|是|https://llava-vl.github.io|
|LLaVA-1.5 [383]|文本与图像|7B/13B|CLIP-ViT 视觉编码器 + 编码器-解码器式 LLM（Vicuna 或 LLaMA）|先在图文对上进行对齐预训练，再用特定任务数据进行视觉指令调优|在高分辨率图像输入与数据效率方面表现突出|是|https://llava-vl.github.io|
|BLIP-2 [206]|文本与图像|3.4B/12.1B|冻结图像编码器 + 轻量 Querying Transformer + 冻结 LLM|先用冻结图像编码器训练 Q-Former，再连接冻结 LLM 进行图文生成学习|通过利用冻结预训练模型和轻量桥接模块，在多种视觉-语言任务上以高效算力获得 SOTA 表现|是|https://github.com/salesforce/LAVIS/tree/main/projects/blip2|
|Phi-3.5-Vision [234]|文本与图像|4.2B|图像编码器：CLIP ViT-L/14；文本输出：基于 Phi-3.5-mini 的 Transformer 解码器|在图文交错数据、OCR 合成数据、图表/表格理解数据与纯文本上联合预训练；再用大规模多模态与文本数据监督微调，并通过 DPO 提升对齐与安全性|在单图像与多图像场景中都具竞争力的多模态推理能力，同时保持模型紧凑，适合端侧/低算力部署|是|https://github.com/microsoft/Phi-3CookBook/tree/main|

上述表格与前文对各模型的文字说明共同构成了针对 GUI 智能体基础模型的完整翻译与整理。
为便于对 Web 场景下的大动作模型（LAM）进行对比，下面给出原文表 26 的中文整理。

表 26：面向 Web GUI 智能体的大动作模型（LAM）概览

|模型（Model）|平台（Platform）|基础模型（Foundation Model）|规模（Size）|输入（Input）|输出（Output）|训练/数据集（Dataset）|亮点（Highlights）|链接（Link）|
|---|---|---|---|---|---|---|---|---|
|Agent Q [281]|Web|LLaMA-3 70B [91]|70B|HTML DOM 表示|规划（plans）、思考（thoughts）、动作（actions）及动作解释|WebShop 基准与 OpenTable 数据集|将蒙特卡洛树搜索（MCTS）与自我批判机制结合，并使用 DPO 强化学习，在商品搜索、预订等复杂任务上取得极高成功率|https://github.com/sentient%2Dengineering/agent-q|
|GLAINTEL [385]|Web|Flan-T5 [498]|780M|用户指令与网页状态观测|GUI 动作|包含 118 万真实商品、12,087 条众包自然语言意图与 1,010 条人工示范|在较小模型规模下通过强化学习与人类示范结合取得鲁棒性能，在模拟电商等动态 Web 环境中表现接近 GPT-4 方案|/|
|WebN-T5 [386]|Web|T5 [87]|-|HTML、DOM 与截图|层次化导航规划与 GUI 交互|MiniWoB++，13,000 条人工轨迹|结合监督学习与强化学习，缓解以往模型在记忆与泛化方面的不足|/|
|OpenWebVoyager [387]|Web|Idefics2-8B-Instruct [523]|8B|GUI 截图与可访问性树|GUI 动作、规划与思考、问题回答|Mind2Web 与 WebVoyager 数据集，以及针对真实 Web 导航生成的查询|结合模仿学习与反馈闭环，实现持续改进并在真实站点上完成端到端导航|https://github.com/MinorJerry/OpenWebVoyager|
|WebRL [388]|Web|Llama-3.1 [91] 与 GLM-4 [524]|8B/9B/70B|任务指令、动作历史与 HTML 内容|动作、元素标识与解释或备注|WebArena-Lite|提出自进化在线课程 RL 框架，根据过去失败动态生成任务并适配智能体能力，实现强鲁棒性|https://github.com/THUDM/WebRL|
|WebGUM [153]|Web|Flan-T5 [498] 与 ViT [522]|3B|HTML、截图、交互历史与指令|Web 导航动作与自由文本|MiniWoB++ 与 WebShop 基准|引入时间与局部多模态感知，将 HTML 与视觉 token 结合，并使用指令微调语言模型，显著提升多步推理与任务泛化能力|https://console.cloud.google.com/storage/browser/gresearch/webllm|

上述模型配合前文对 WebGUM、Agent Q、GLAINTEL、OpenWebVoyager 与 WebRL 的文字描述，共同构成了 Web 场景 LAM 的完整整理。

接下来整理移动端 LAM（原文表 27 与表 28）。

表 27：面向移动 GUI 智能体的大动作模型概览（一）

|模型（Model）|平台（Platform）|基础模型（Foundation Model）|规模（Size）|输入（Input）|输出（Output）|数据集（Dataset）|亮点（Highlights）|链接（Link）|
|---|---|---|---|---|---|---|---|---|
|MobileVLM [353]|Android 移动端|Qwen-VL-Chat [210]|9.8B|截图与结构化 XML 文档|动作预测、导航步骤与元素位置|Mobile3M 数据集（含 300 万 UI 页面与 2000 万+ 动作）|通过移动特定预训练任务增强 UI 内部与跨 UI 理解，在 ScreenQA、Auto-UI 等基准上显著优于现有 VLM|https://github.com/XiaoMi/mobilevlm|
|Octo-planner [361]|移动设备|Phi-3 Mini [234]|3.8B|用户查询与可用函数描述|执行步骤|由 GPT-4 生成的 1,000 条样本|面向资源受限设备优化，保证低时延、隐私与离线能力|https://huggingface.co/NexaAIDev/octopus-planning|
|DigiRL [264]|Android 移动端|AutoUI [302]（1.3B VLM）|1.3B|GUI 截图|GUI 动作|AITW 数据集|离线到在线 RL 架构，兼顾静态数据与动态环境，显著提升在 AITW 上的任务成功率|https://github.com/DigiRL-agent/digirl|
|LVG [390]|Android 移动端|Swin Transformer [525] 与 BERT [82]|-|UI 截图与自然语言表达|边界框坐标|UIBERT 数据集及合成数据|通过布局引导的对比学习统一检测与 grounding 任务|/|
|Ferret-UI [352]|Android 与 iOS|Ferret [382]|7B/13B|原始屏幕像素、切分子图、边界框与区域标注|控件边界框、OCR 文本、UI 元素与整体功能描述、交互动作|基于 RICO（Android）与 AMP（iOS）的数据生成|支持多平台、具自适应高分辨率编码的 UI 中心 VLM|https://github.com/apple/ml-ferret/tree/main/ferretui|
|Octopus [391]|移动设备|CodeLlama-7B [526]、Gemma 2B [502]|7B, 2B|API 文档示例|API 调用的函数名与参数|RapidAPI Hub|通过条件掩码（conditional masking）约束输出格式，保证 API 调用合法性|/|
|Octopus v2 [392]|边缘设备|Gemma-2B [502]|2B|用户查询与函数描述|带精确参数的函数调用|20 个 Android API，每个最多 1,000 个训练样本|提出函数 token 化策略，将函数调用映射为专用 token，大幅缩短上下文长度|/|
|Octopus v3 [393]|边缘设备|CLIP + 因果语言模型|<1B|查询与命令、图像与函数 token|动作对应的函数 token|继承 Octopus v2 数据|在多模态场景中引入函数 token，使任意函数可作为 token 表示，提升模型灵活性|/|
|Octopus v4 [394]|无服务器云与边缘|17 个模型组合|多种|用户查询|领域特定答案与动作|与 Octopus v2 类似的合成数据|图结构框架整合多个专门模型，实现性能最优组合|https://github.com/NexaAI/octopus-v4|

表 28：面向移动 GUI 智能体的大动作模型概览（二）

|模型（Model）|平台（Platform）|基础模型（Foundation Model）|规模（Size）|输入（Input）|输出（Output）|数据集（Dataset）|亮点（Highlights）|链接（Link）|
|---|---|---|---|---|---|---|---|---|
|VGA [354]|Android 移动端|LLaVA-v1.6-Mistral-7B [217]|7B|带位置、视觉与层次信息的 GUI 截图|动作与函数调用、GUI 组件描述、导航与任务规划|基于 RICO 构建的 63.8k 图像数据集|通过图像中心微调减少幻觉，在文本与视觉信息之间保持平衡关注，大幅提升 GUI 理解准确率|https://github.com/Linziyang1999/VGA%2Dvisual%2DGUI%2Dassistant|
|MobileFlo [159]|移动手机|Qwen-VL-Chat [210]|21B|带 OCR 文本与边界框的 GUI 截图|GUI 动作与问答|7 万条覆盖 10 个业务领域的人工标注数据 + RefCOCO、ScreenQA、Flickr30K 等数据集|采用可变分辨率混合视觉编码器与 MoE 结构，在性能与效率间取得平衡|/|
|UINav [395]|Android 移动端|SmallBERT [528]|Agent 模型：32 万参数；Referee 模型：43 万参数；SmallBERT：17.6MB|UI 元素、用户话语与屏幕表示|预测动作及其作用元素|跨 128 个 Android 应用与网站共 43 个任务，3,661 条示范|引入宏动作与错误驱动示范收集，大幅降低训练成本，同时在小模型下实现稳健任务表现|/|
|AppVLM [397]|Android 移动端|Paligemma-3B-896 [529]|3B|带边界框与 UI 标签的截图|GUI 动作|AndroidControl [515] 与 AndroidWorld [440]|轻量模型，在 Android 控制任务上接近 GPT-4o 性能，同时速度提高一数量级、资源消耗显著降低|/|
|VSC-RL [396]|Android 移动端|AutoUI [302] 与 Gemini-1.5-Pro|/|截图|GUI 动作|AITW|面向稀疏奖励、长时序 RL，通过自动分解复杂目标为子目标提升训练效率|https://ai-agents-2030.github.io/VSC-RL|
|Digi-Q [398]|Android 移动端|LLaVA-1.5 [383]|7B|GUI 截图|GUI 动作与 Q 值|AITW [358]|引入基于 VLM 的 Q 函数，使得在无需在线交互的情况下实现 RL 训练|https://github.com/DigiRL-agent/digiq|
|VEM [399]|Android 移动端|Qwen2-VL [231]|7B|GUI 截图|GUI 动作与 Q 值|AITW [358]|提出 Value Environment Model，可仅用离线数据通过 PPO 训练策略，无需直接与环境交互|https://github.com/microsoft/GUI-Agent-RL|
|MP-GUI [400]|Android 移动端|InternViT-300M + InternLM2.5-7B-chat [530]|约 8B|GUI 截图|自然语言输出、元素 grounding、描述与语义导航|68 万条混合模态数据|提出三感知器架构，同时建模文本、图形与空间模态，增强 GUI 推理|https://github.com/BigTaige/MP-GUI|
|UI-R1 [401]|Android 移动端|Qwen2.5-VL-3B|3B|GUI 截图|推理文本与 GUI 动作|ScreenSpot 与 AndroidControl|使用 GRPO 的规则式 RL 框架，仅需 136 个样本即可显著提升推理与动作预测能力|https://github.com/lll6gg/UI-R1|
|ViMo [402]|Android 移动端|预训练 Stable Diffusion 模型 [531]|/|当前 GUI 图像、自然语言动作、GUI 文本表示|下一状态 GUI 文本表示与完整 GUI 图像（未来屏幕可视化预测）|Android Control 与 AITW|首个 GUI 世界模型，以图像形式预测未来 GUI 状态，为长时序规划提供高保真模拟|https://ai-agents-2030.github.io/ViMo/|

最后，整理计算机端与跨平台 LAM（对应原文表 29、表 30 与表 31）。

表 29：面向桌面 GUI 智能体的大动作模型概览

|模型（Model）|平台（Platform）|基础模型（Foundation Model）|规模（Size）|输入（Input）|输出（Output）|数据集（Dataset）|亮点（Highlights）|链接（Link）|
|---|---|---|---|---|---|---|---|---|
|ScreenAgent [366]|Linux 与 Windows 桌面|CogAgent [15]|18B|GUI 截图|鼠标与键盘动作（JSON 形式）|273 个任务会话|集成规划、执行与反思的完整流水线，实现对真实电脑屏幕的多步自主操作|https://github.com/niuzaisheng/ScreenAgent|
|Octopus [403]|桌面/具身环境|MPT-7B [535] + CLIP ViT-L/14 [509]|7B|视觉图像、包含对象与关系的场景图、环境消息|可执行动作代码与计划|OctoGibson（476 个结构化任务）、OctoMC（40 个任务）、OctoGTA（25 个任务）|结合环境反馈强化学习，将高层规划与实际操作连接起来，用于复杂 3D 与 GUI 场景|https://choiszt.github.io/Octopus/|
|LAM [367]|Windows OS|Mistral-7B [501]|7B|自然语言任务请求与应用环境数据|规划与动作|76,672 对任务-规划与 2,192 条任务-动作轨迹|多阶段训练（任务规划预训练、模仿学习、自举探索与奖励优化），在 GUI 场景中打通规划与执行|https://github.com/microsoft/UFO/tree/main/dataflow|
|ScreenLL [404]|桌面|LLaVA|7B/13B|GUI 截图|预测 GUI 动作|高分辨率 YouTube 教程数据|提出有状态 screen schema 紧凑表示多步 GUI 交互，提升细粒度理解与动作预测|/|

表 30：跨平台大动作模型概览（一）

|模型（Model）|平台（Platform）|基础模型（Foundation Model）|规模（Size）|输入（Input）|输出（Output）|数据集（Dataset）|亮点（Highlights）|链接（Link）|
|---|---|---|---|---|---|---|---|---|
|RUIG [405]|移动与桌面|Swin Transformer [525] + BART [88]|4 层解码器|UI 截图与文本指令|以语言形式给出的边界框预测|MoTIF 与 RicoSCA（移动 UI），以及 Common Crawl（桌面 UI）|在像素到序列范式下使用策略梯度提升空间解码效果|/|
|CogAgent [15]|PC、Web 与 Android|CogVLM-17B [381]|18B|带 OCR 文本的 GUI 截图|任务规划、动作序列与文本说明|CCS400K、文本识别数据集（8,000 万合成文本图像）、视觉 grounding 数据集，以及 Mind2Web、AITW 等 GUI 数据集|采用高分辨率交叉模块，在计算效率与高分辨率输入之间取得平衡，在多平台 GUI grounding 上表现优异|https://github.com/THUDM/CogVLM|
|SeeClick [25]|iOS、Android、macOS、Windows 与 Web|Qwen-VL [210]|9.6B|GUI 截图与文本指令|GUI 动作与元素位置|30 万网页（文本与图标）、RICO 以及 LLaVA 生成的数据|仅凭截图执行 GUI 任务，并提出新颖 GUI grounding 预训练方法|https://github.com/njucckevin/SeeClick|
|ScreenAI [375]|移动端、桌面与平板 UI|PaLI-3 [536]|5B|带 OCR 文本、图像标题等的 GUI 截图|问题答案、屏幕标注（边界框与标签）、导航指令与摘要|2.62 亿移动 Web 截图与 5,400 万移动应用截图|统一 UI 与信息图表示，将视觉与文本元素结合用于多平台 GUI 预训练与微调|https://github.com/kyegomez/ScreenAI|
|V-Zen [408]|计算机与 Web|Vicuna-7B [510]、DINO [183]、EVA-2-CLIP [537]|7B|文本与 GUI 图像|动作预测与 GUI 边界框|GUIDE [370]|采用双分辨率视觉编码实现精确 GUI grounding 与任务执行|https://github.com/abdur75648/V-Zen|
|Ferret-UI 2 [406]|iPhone、Android、iPad、Web、AppleTV|Vicuna-13B、Gemma-2B、Llama3-8B [91]|多种|UI 截图、控件边界框与标签、OCR 文本及其边界框、Web 的 HTML 树|UI 元素描述、控件分类、OCR、可点击性、文本/控件定位、交互指令与多轮 QA|Core-set、GroundUI-18k、GUIDE、Spotlight 等|通过动态高分辨率编码、自适应网格划分与高质量多模态训练，在多平台 UI 引用与交互任务上达到 SOTA|/|
|ShowUI [238]|网站、桌面与手机|Phi-3.5-Vision [234]|4.2B|GUI 截图（结合 OCR 与视觉 grounding）|GUI 动作、导航与元素定位|ScreenSpot、RICO、GUIEnv、GUIAct、AITW、AITZ、GUI-World 等|采用 Vision-Language-Action 交织策略，在导航、grounding 与理解任务中实现无缝切换|https://github.com/showlab/ShowUI|
|OS-ATLAS [232]|Windows、macOS、Linux、Android 与 Web|InternVL-2 [379] 与 Qwen2-VL [210]|4B/7B|GUI 截图|GUI 动作|AndroidControl、SeeClick 等，含 1,300 万 GUI 元素与 230 万截图|首个针对通用 GUI 智能体的基础动作模型，提出跨平台统一动作空间|https://osatlas.github.io/|
|xLAM [372]|多种环境|Mistral-7B [501] 与 DeepSeek-Coder-7B [538]|1B–8×22B|统一函数调用数据格式|函数调用与推理文本|合成与增强数据，包含 60,000+ 由 APIGen 生成的高质量样本，覆盖 21 类 3,673 个 API|通过统一可扩展的数据流水线，在函数调用任务上表现突出|https://github.com/SalesforceAIResearch/xLAM|
|SpiritSight [384]|Web、Android、Windows 桌面|InternVL [379]|2B/8B/26B|GUI 截图|GUI 动作|AITW、Common Crawl 网站与自定义标注|提出 Universal Block Parsing（UBP）方法，解决高分辨率视觉输入中的位置歧义问题|https://hzhiyuan.github.io/SpiritSight-Agent|

表 31：跨平台大动作模型概览（二）

|模型（Model）|平台（Platform）|基础模型（Foundation Model）|规模（Size）|输入（Input）|输出（Output）|数据集（Dataset）|亮点（Highlights）|链接（Link）|
|---|---|---|---|---|---|---|---|---|
|Falcon-UI [373]|iOS、Android、Windows、Linux 与 Web|Qwen2-VL-7B [231]|7B|带节点信息与 OCR 标注的 GUI 截图|GUI 动作与交互元素坐标/边界框|Insight-UI 数据集，并在 AITW、AITZ、Android Control、Mind2Web 等数据集上进一步微调|将 GUI 上下文理解与指令跟随解耦，采用无指令预训练方法进行 GUI 语义建模|/|
|UI-TARS [407]|Web、桌面（Windows、macOS）、移动（Android）|Qwen2-VL-7B/72B [231]|7B/72B|GUI 截图|GUI 动作|从网站、应用与操作系统采集的 GUI 截图与元数据，来自多种基准的数据轨迹与 600 万 GUI 教程|纯视觉感知 + 标准化跨平台动作空间（Web/移动/桌面），支持统一 GUI 操作|https://github.com/bytedance/UI-TARS|
|Magma [409]|Web、移动、桌面与机器人|LLaMA-3-8B [91] + ConvNeXt-XXLarge [539]|8.6B|GUI 截图与文本任务描述|GUI 动作与机器人操作|UI/机器人数据与人类教学视频|在数字界面与物理环境联合数据上训练，实现跨数字/物理任务泛化|https://microsoft.github.io/Magma/|
|GUI-R1 [410]|Windows、Linux、macOS、Android 与 Web|Qwen2.5-VL [540]|3B/7B|GUI 截图|推理文本与 GUI 动作|约 3K 高质量样本混合|首个在高层 GUI 任务中系统引入规则式 RL（RFT）的框架，覆盖多平台|https://github.com/ritzz-ai/GUI-R1.git|
|InfiGUI-R1 [411]|Web、桌面与 Android|Qwen2.5-VL-3B-Instruct|3B|GUI 截图与可访问性树|推理文本与 GUI 动作|多源混合数据集|提出 Actor2Reasoner 双阶段训练：先通过空间推理蒸馏注入推理能力，再通过具有子目标与错误恢复构造的 RL 强化 deliberation|https://github.com/Reallm-Labs/InfiGUI-R1|

上述表格与前文对应文字说明共同完成了对各类 LAM（Web、移动、桌面与跨平台）的逐条翻译与整理。
补充跨平台 LAM 表中遗漏的一项：

在表 30 中，Task Generalization [389] 也是一个重要模型，我们补充如下：

- Task Generalization [389]：平台为 Web 与 Android（移动端），基础模型为 Qwen2-VL-7B-Instruct [231]，规模 7B，输入为 GUI 截图，输出为思考过程与带 grounding 的坐标型动作。其训练使用 11 个领域数据集，总计 56K GUI 轨迹样本，提出在多种非 GUI 推理任务（尤其是数学与代码）上进行“中期训练”（mid-training），显著增强 GUI 智能体的规划能力。其项目地址为：https://github.com/hkust-nlp/GUIMid。

这些跨平台 LAM 所依托的高级数据集与创新训练技术，保证了在空间理解和以用户为中心交互上的高精度表现，为跨平台 UI 适配性和性能设定了新的标杆。

为了推进 GUI 自动化，ShowUI [238] 提出了一个开创性的视觉-语言-动作（Vision-Language-Action）模型，将高分辨率视觉输入与文本理解相结合，用于执行 grounding、导航和任务规划。ShowUI 面向 Web、桌面和移动环境进行了优化，基于 Phi-3.5-vision-instruct 骨干网络，并利用全面的数据集，在 ScreenSpot [25]、GUI-Odyssey [359] 等基准上取得了稳健表现。它能够处理多帧与动态视觉输入，并输出 JSON 结构化动作，体现了极高的通用性。通过交错的图像-文本处理与函数调用能力等创新设计，ShowUI 为 LLM 驱动的 GUI 智能体树立了新的标准。

为了解决统一动作空间的需求，OS-ATLAS [232] 提出了一个专门为 GUI 智能体设计的基础动作模型，适用于 Windows、macOS、Linux、Android 以及 Web 等平台。通过构建大规模多平台数据集并引入统一的动作空间，OS-ATLAS 在 GUI grounding 与分布外（out-of-distribution）泛化任务上取得了当前最优表现。其可伸缩配置能够适应不同计算预算，同时在处理自然语言指令与 GUI 元素方面保持高度灵活性。作为一个强大的开源方案，OS-ATLAS 为替代商业系统提供了可行路径，推动了高级 GUI 智能体能力的普惠化。

Magma [409] 是一款面向多模态 AI 智能体的基础模型，将 LLM 与视觉及动作理解相结合，以完成 UI 导航与机器人操作任务。不同于以往只针对 UI 自动化或机器人单一场景进行优化的模型，Magma 在一个包含约 3,900 万样本的异构数据集上进行联合训练，数据覆盖 UI 截图、Web 导航、机器人轨迹以及教学视频。它采用 SoM 和 Trace-of-Mark 等技术，通过对 GUI 环境中的可操作元素进行标注以及在机器人任务中追踪动作轨迹，提升动作 grounding 与预测能力。

UI-TARS [407] 是一款先进的、基于视觉的大动作模型（Large Action Model, LAM），面向多平台 GUI 智能体进行了优化。不同于依赖结构化表示的传统方法，UI-TARS 仅使用 GUI 截图进行感知，从而消除了对结构化元数据的需求。通过引入统一动作空间，UI-TARS 支持在 Web、Windows、macOS 与 Android 环境中的无缝执行。该模型基于 Qwen-2-VL 构建，在 600 万个 GUI 教程、大规模截图数据集及多种开源基准上进行训练。其一大创新是具备 System-2 推理能力，可以在执行动作之前生成显式推理步骤，从而在动态环境中强化决策质量。此外，UI-TARS 采用迭代式自我提升框架，通过基于反思的学习不断优化性能。实验结果表明，在任务执行基准上，UI-TARS 超越了包括 GPT-4o 和 Claude 在内的现有模型。

这些跨平台 LAM 展示了统一模型在适配多样环境方面的潜力，显著提升了 GUI 智能体在不同场景中的可扩展性与适用性。

**8.7 关键总结（Takeaways）**

围绕 GUI 智能体的大动作模型（LAM）的探索揭示了若干正在塑造图形用户界面智能交互未来的关键洞见：

1) 面向端侧推理的小型模型（Smaller Models for On-Device Inference）：许多优化后的 LAM 构建在规模较小的基础模型之上，参数量通常在 10 亿到 70 亿之间。模型规模的缩减提升了计算效率，使其能够部署在手机和边缘设备等资源受限终端上。无需依赖云端即可进行端侧推理，有助于解决隐私问题并降低时延，从而带来更灵敏的用户体验。
2) 增强的 GUI 理解降低对结构化数据的依赖（Enhanced GUI Comprehension Reduces Reliance on Structured Data）：诸如 VGA [354] 与 OmniParser [184] 等模型强调视觉 grounding 与以图像为中心的微调，以降低对结构化 UI 元数据的依赖。通过直接从视觉输入提升 GUI 理解能力，智能体在不同软件环境中的适应性增强，尤其是在结构化数据不可获取或不一致时更具优势。
3) 强化学习连接静态与动态环境（Reinforcement Learning Bridges Static and Dynamic Environments）：在 DigiRL [264] 等模型中应用的强化学习，证明了将静态训练数据与动态真实环境桥接起来的有效性。该方法使智能体能够从交互中学习、从错误中恢复并适应环境变化，从而在实际应用中提升鲁棒性与可靠性。
4) 统一的函数调用增强互操作性（Unified Function-Calling Enhances Interoperability）：如 xLAM [372] 等模型中所体现的，对数据格式与函数调用机制进行标准化的努力，有助于在不同平台之间实现多轮交互与推理。统一化缓解了兼容性问题，并增强了智能体在涉及多种 API 和服务的复杂任务中的执行能力。
5) 推理时计算与推理模型（Inference-Time Computing and Reasoning Models）：近期工作强调推理时计算（inference-time computing）的重要性，即在无需改变模型架构的情况下，让模型在推理阶段即席地进行任务规划、推理与分解。诸如扩展上下文窗口与链式思维提示（Chain-of-Thought, CoT，例如 “o1-style” 推理）等技术，使得长时序决策更加稳健。UI-R1 [401]、GUI-R1 [410] 与 InfiGUI-R1 [411] 是该方向的先驱工作。同时，基于规则的奖励与成本函数在引导推理阶段行为方面也受到越来越多关注，通过显式启发式的注入，进一步提升 GUI 智能体的稳定性、可解释性与泛化能力。

总体而言，面向 GUI 智能体的大动作模型的发展体现出一种趋势：模型更加专门化、高效且具有适应性，能够在多种平台上执行复杂任务。通过聚焦专门化、多模态融合以及创新训练方法，研究者正在逐步突破通用 LLM 的局限。这些洞见为构建更智能、更灵活、更加友好的 GUI 智能体铺平道路，使其有潜力从根本上改变人机在软件应用中的交互方式。

---

下面补充第 9 节关于评估的翻译（第 9.1–9.4 小节）。

###### 9 面向以 LLM 为大脑的 GUI 智能体的评估（EVALUATION FOR LLM-BRAINED GUI AGENTS）

在 GUI 智能体领域，评估对于提升功能性与用户体验都至关重要 [56], [58]，且应从多个维度开展。通过系统地评估智能体在不同任务上的表现，不仅能够在不同维度上量化性能，也为持续改进提供了框架 [541]。同时，评估还能帮助发现潜在改进空间，从而推动新方法的提出，确保 GUI 智能体在跟随 LLM 发展的同时，也能符合用户与业务的期望。

如图 28 所示，当 GUI 智能体完成一个任务时，会产出动作序列、截图、UI 结构以及最终环境状态等信息。这些“轨迹资产”构成了在多平台上评估其表现的基础。接下来，我们将依次介绍评估方法，包括常用评估指标、测量方式与平台，并给出各平台对应的基准。

**9.1 评估指标（Evaluation Metrics）**

评估 GUI 智能体需要多维度的指标，以衡量其在准确性、效率与合规性（如安全性）等方面的表现。在典型基准设置中，智能体接收自然语言指令作为输入，并自主执行动作直至任务结束。在此过程中，可以收集动作序列、逐步观测（如 DOM/HTML 结构）、截图、运行日志、最终状态与执行时间等资产，据此判断任务是否成功完成并分析性能。常见核心指标如下（不同工作命名略有差异，这里统一归类）：

1) 步骤成功率（Step Success Rate）：完成任务往往需要多步操作。该指标衡量在某一任务中成功步骤数与总步骤数的比例。较高的步骤成功率表明智能体在细粒度层面对每一步执行较为准确，对于多步任务的可靠性尤为重要 [212], [349], [358]。
2) 轮次成功率（Turn Success Rate）：“轮次”指一次用户与智能体之间的交互，一个轮次可能包含多步操作，一个完整任务可能包含多轮。该指标衡量在所有轮次中，成功满足当前轮次请求的比例，用于评估智能体在交互式或对话式任务中理解与满足用户期望的能力 [155], [348]。
3) 任务成功率（Task Success Rate）：衡量所有任务中成功完成任务的比例，仅关注最终任务是否完成，而不考虑中间步骤。该指标反映智能体在端到端层面处理复杂工作流的能力 [419], [425], [438]。
4) 效率得分（Efficiency Score）：评估智能体在考虑资源消耗、执行时间或步骤数等情况下完成任务的效率。可拆解为：
   - 时间成本（Time Cost）：完成任务所用时间；
   - 资源成本（Resource Cost）：完成任务所使用的内存/CPU/GPU 资源；
   - LLM 成本（LLM Cost）：任务执行过程中 LLM 调用带来的计算或金钱成本；
   - 步骤成本（Step Cost）：完成任务所需的总步骤数。
   不同工作会根据关注点组合这些子指标 [442], [444]。
5) 策略下完成率（Completion under Policy, CuP）：衡量任务在遵守策略约束的前提下完成的比例。策略可包含安全、伦理、隐私或业务规则等，尤其适用于企业场景 [416]。
6) 风险比（Risk Ratio）：类似 CuP，风险比评估智能体在执行任务时可能产生的风险，包括漏洞、错误或安全隐患等。较低的风险比意味着更高的可信度与鲁棒性 [416]。

不同基准中对上述指标的具体实现可能略有不同，但都可以映射到上述类别。后续的各表均按照此分类统一指标命名。

**9.2 评估测量方式（Evaluation Measurements）**

为了实现上述指标，需要定义具体的测量方式，用于判断智能体输出与预期结果之间的一致性或偏差。这些测量方法可以从文本、图像、元素级交互到系统状态多个层面进行 [26]。

1) 文本匹配（Text Match）：评估智能体的文本输出是否与期望结果相符，例如在电商网站中，目标商品名称是否正确。可包含不同严格程度：
   - 精确匹配（Exact Match）：输出与期望结果完全一致；
   - 部分/模糊匹配（Partial/Fuzzy Match）：允许一定偏差，用于处理轻微拼写错误或同义表达；
   - 语义相似度（Semantic Similarity）：使用向量相似度等方法衡量语义层面的接近程度。
   文本匹配广泛用于文本选择、数据录入与自然语言回答等任务。

2) 图像匹配（Image Match）：评估智能体是否停留或操作在正确页面（如指定网页或 App 界面），或选择了正确图像。通常通过比较截图或目标视觉元素与标注图像的相似度，或采用 VQA 方法。该测量对于需要精确视觉识别的任务尤为关键。

3) 元素匹配（Element Match）：检查智能体交互的具体控件元素（如 HTML/DOM 节点或应用 UI 层级中的控件）是否与预期一致，包括：
   - HTML 标签与属性；
   - URL 与链接；
   - DOM 层级结构；
   - 各类 UI 控件（按钮、滑块、复选框、下拉菜单等）；
   - 无障碍标识（Accessibility ID、resource-id 等）；
   - 视图层级（View Hierarchy）；
   - 系统控件与 API（文件对话框、系统菜单、通知等）。
   元素匹配保证跨平台交互控件识别与操作的鲁棒性。

4) 动作匹配（Action Match）：评估智能体执行的动作是否与期望动作序列一致，包括：
   - 动作准确率：每个动作的类型与参数是否正确（如点击了正确按钮、输入了正确文本）；
   - 动作序列对齐：动作是否按正确顺序执行；
   - 位置预测：空间动作（如点击、触摸）的坐标是否落在预期区域内。

5) 状态信息（State Information）：记录任务执行期间与环境状态相关的运行信息，例如应用当前状态（打开文件、活动窗口等）、系统日志与环境变量等，用于调试、性能分析与可靠性验证。

上述测量方式共同构成了完整的评估框架，确保智能体不仅“完成任务”，而且在精度、效率与适应性上都表现良好。

**9.3 评估平台（Evaluation Platforms）**

为了全面评估 GUI 智能体，需要在多种平台上进行测试，包括 Web、移动端与桌面端，每类平台有不同特性与挑战。

1) Web 平台：最常见的 GUI 环境之一，涵盖浏览、表单填写与数据抓取等任务。其特点包括：
   - 动态内容：需处理由 JavaScript、AJAX 等产生的动态元素与异步更新；
   - 技术多样性：HTML/CSS/JS 及各类框架带来界面结构的多样性；
   - 工具支持：常用 Selenium、Puppeteer、Playwright 等工具模拟浏览器交互并收集运行信息；
   - 可访问性合规：可评估对 WCAG 等无障碍标准的遵从情况。

2) 移动平台：主要包括 Android 与 iOS，需要应对屏幕空间有限与触控交互等特性：
   - 屏幕尺寸受限：智能体需在有限空间内保持高精度与高效率交互；
   - 触控手势：需模拟轻触、滑动、捏合等手势；
   - 设备多样性（Android）与相对统一性（iOS）；
   - 评估工具：如 Appium、Espresso、XCTest 以及各类模拟器。

3) 桌面平台：包括 Windows、macOS 与 Linux 等，更注重：
   - 应用多样性：涵盖生产力工具、浏览器与企业软件等；
   - 交互复杂度：包括快捷键、拖拽、上下文菜单等高级交互；
   - 跨平台兼容性：需在不同 OS 与版本间保持一致表现；
   - 自动化框架：如 Windows UI Automation、macOS Accessibility API 与 AT-SPI 等；
   - 资源使用：评估长时任务或重负载场景下的内存与 CPU 占用。

**9.4 Web 智能体基准（Web Agent Benchmarks）**

Web 环境下的 GUI 智能体评估需要能够刻画复杂 Web 任务的基准。多年来出现了多种基准，各有侧重点与挑战。我们在表 32–35 中给出总览，这里先概述部分代表性基准：

- MiniWoB++ [145], [146]：早期面向强化学习智能体的 Web GUI 任务基准，包含点击、输入、表单提交等多种交互场景，引入 Workflow-Guided Exploration（WGE）以缓解稀疏奖励问题，主要以任务成功率为指标。
- Mind2Web [212]：在真实 HTML 环境中评估智能体能力，覆盖 2,000+ 跨领域任务，从基本操作到复杂多页面工作流。使用元素准确率（Element Accuracy）与任务成功率等指标，强调端到端性能。
- MT-Mind2Web [414]：在 Mind2Web 基础上引入对话式 Web 导航，包含 720 个会话、3,525 对指令与动作序列，平均每个会话约 5 轮交互，用于评估多轮对话与适应性。
- WebArena [412]：提供更贴近真实的 Docker 托管 Web 环境，包含 812 个跨领域任务，要求在多 Tab、多页面之间执行长时交互，以功能正确性为核心评估点。
- VisualWebArena [413]：WebArena 的多模态扩展，加入 Classifieds、Shopping、Reddit 等视觉任务，共 910 个任务，考察视觉问答与模糊图像匹配等能力。
- VideoWebArena [421]：聚焦 Web 上的视频理解与交互，包含 74 个视频与 2,021 个任务，用于评估视频检索、上下文推理与技能迁移。
- VisualWebBench [213]：多模态基准，评估网站层级、元素层级与动作层级的理解、OCR、grounding 与推理能力，共 1.5K 真实网站样本。
- EnvDistraction [422]：通过弹窗、推荐等“非恶意干扰”测试多模态 GUI 智能体的可靠性，显示当前模型容易被干扰，强调鲁棒感知的重要性。
- ST-WebAgentBench [416]：面向企业场景的安全与可信评估，采用 BrowserGym [426] 环境，引入 Completion under Policy（CuP）指标以及人工参与与策略层次结构，用于评估智能体在多种安全策略下的表现。
- WorkArena [420]：聚焦 ServiceNow 平台上的企业软件自动化，包含 33 个任务、19,912 个实例，揭示当前智能体在企业 UI 自动化方面与人类能力之间的差距。
- BrowserGym [426]：统一 MiniWoB(++), WebArena, WorkArena 等多种基准的 Web 智能体研究框架，标准化观测与动作空间，便于可复现实验。
- WebOlympus [424]：基于 Chrome 扩展的开放平台，使智能体能够在真实网站上执行任务，并通过安全监控组件防止有害操作。

表 32：部分 Web GUI 基准概览

|基准（Benchmark）|平台（Platform）|年份（Year）|是否实时（Live）|亮点（Highlight）|规模（Data Size）|指标（Metric）|测量方式（Measurement）|链接（Link）|
|---|---|---|---|---|---|---|---|---|
|MiniWoB++ [145], [146]|Web|2017|是|在 100 个 Web 交互任务上评估强化学习智能体，包括点击、输入与表单导航等|100 个 Web 任务|Task Success Rate|Element Match|https://github.com/Farama%2DFoundation/miniwob%2Dplusplus|
|RUSS [142]|Web|2021|否|使用 ThingTalk 将自然语言映射为 Web 动作，在真实 HTML 环境中实现精确 Web 任务执行|741 条指令|Task Success Rate|Text Match, Element Match|https://github.com/xnancy/russ|
|WebShop [425]|Web|2022|是|模拟电商导航与真实商品搜索，考察指令理解、多页面导航与策略探索能力|12,087 条指令|Task Success Rate, Step Success Rate|Text Match|https://webshop-pnlp.github.io/|
|Mind2Web [212]|Web|2023|否|在真实动态网站上评估智能体适应性与泛化能力|2,000 个任务|Step Success Rate, Task Success Rate|Element Match, Action Match|https://github.com/OSU-NLP-Group/Mind2Web|
|Mind2Web-Live [349]|Web|2024|是|提供中间动作跟踪与 key-node 级动态评估，并发布 Mind2Web-Live 数据集与工具|542 个任务|Step Success Rate, Task Success Rate, Efficiency Score|Element Match, Text Match, 轨迹长度|https://huggingface.co/datasets/iMeanAI/Mind2Web-Live|
|Mind2Web-Live-Abstracted [278]|Web|2024|是|对 Mind2Web-Live 中描述进行抽象，省略任务细节与用户输入信息，降低标注成本|104 个样本|Task Success Rate, Efficiency Score|Text Match, Image Match, Element Match, Path Length|https://anonymous.4open.science/r/naviqate|
|WebArena [412]|Web|2023|是|通过 Docker 托管网站模拟真实多 Tab 浏览，聚焦复杂长时任务|812 个长时任务|Step Success Rate|Text Match|https://webarena.dev/|
|VisualWebArena [413]|Web|2024|是|在 WebArena 基础上加入视觉任务，考察视觉 + 文本双模态能力|910 个任务|Step Success Rate|Text Match, Image Match|https://jykoh.com/vwa|
|MT-Mind2Web [414]|Web|2024|否|引入多轮对话式 Web 导航，提供专门的多轮 Web 数据集|720 个会话/3,525 条指令|Step Success Rate, Turn Success Rate|Element Match, Action Match|https://github.com/magicgh/self-map|

下面表 33–35 进一步给出这些 Web GUI 基准的完整结构化信息，便于系统对比。
表 33：Web GUI 基准概览（二）

|基准（Benchmark）|平台（Platform）|年份（Year）|是否实时（Live）|亮点（Highlight）|规模（Data Size）|指标（Metric）|测量方式（Measurement）|链接（Link）|
|---|---|---|---|---|---|---|---|---|
|MMInA [415]|Web|2024|是|在真实网站上测试多跳、多模态任务，要求智能体在跨页面信息抽取与复杂任务推理方面具备能力|1,050 个任务|Step Success Rate, Task Success Rate|Text Match, Element Match|https://mmina.cliangyu.com/|
|AutoWebBench [270]|Web|2024|否|双语 Web 浏览基准，包含 10,000 条浏览轨迹，支持在不同语言环境下评估智能体|10,000 条轨迹|Step Success Rate, Efficiency Score|Element Match, Action Match, Time|https://github.com/THUDM/AutoWebGLM|
|WorkArena [420]|Web|2024|是|聚焦真实企业软件交互，覆盖知识工作者常见任务|19,912 条唯一任务实例|Task Success Rate, Efficiency Score, Completion under Policy, Turn Success Rate|Element Match, Text Match, Execution-based Validation|https://github.com/ServiceNow/WorkArena|
|VideoWebArena [421]|Web|2024|是|利用视频教程评估长上下文多模态智能体的任务完成能力|约 4 小时视频，共 74 段视频与 2,021 个任务|Task Success Rate, Intermediate Intent Success Rate, Efficiency Scores|Element Match, State Information, Exact & Fuzzy Text Match|https://github.com/ljang0/videowebarena|
|EnvDistraction [422]|Web|2024|否|通过弹窗、伪造搜索结果、误导性推荐等环境干扰评估多模态 GUI 智能体的“忠实度”与鲁棒性|1,198 个任务|Task Success Rate|Text Match, Element Match, State Information|https://github.com/xbmxb/EnvDistraction|
|WebVLN-v1 [347]|Web|2024|否|在购物网站上联合导航与问答任务，整合视觉与文本内容以统一评估 Web 交互|8,990 条路径与 14,825 个问答对|Task Success Rate, Efficiency Score|Element Match, Path Length, Trajectory Length|https://github.com/WebVLN/WebVLN|
|WEBLINX [348]|Web|2024|否|聚焦对话式导航，要求智能体在真实对话式 Web 任务中跟随多轮用户指令|100k 次交互|Turn Success Rate|Element Match, Text Match, Action Match|https://mcgill-nlp.github.io/weblinx/|
|ST-WebAgentBench [416]|Web|2024|是|在企业场景下评估策略驱动的安全性，引入 Completion under Policy（CuP）与风险比等指标|235 个任务|Task Success Rate, Completion under Policy (CuP), Risk Ratio|Element Match, Action Match, Text Match|https://sites.google.com/view/st-webagentbench/home|
|CompWoB [417]|Web|2023|否|测试智能体在多步、组合式任务中的表现，这类任务需要在多步之间维护状态，模拟真实自动化场景|50 个组合任务|Task Success Rate|Element Match|https://github.com/google-research/google-research/tree/master/compositional_rl/compwob|

表 34：Web GUI 基准概览（三）

|基准（Benchmark）|平台（Platform）|年份（Year）|是否实时（Live）|亮点（Highlight）|规模（Data Size）|指标（Metric）|测量方式（Measurement）|链接（Link）|
|---|---|---|---|---|---|---|---|---|
|TURKING BENCH [418]|Web|2024|是|基于众包平台的自然 HTML 任务，评估智能体在真实网页布局与元素上的交互能力|32.2K 个实例|Task Success Rate|Text Match, Element Match, Image Match|https://turkingbench.github.io|
|VisualWebBench [213]|Web|2024|否|面向 Web 场景，对多模态大语言模型在细粒度任务上的表现进行评估|来自 139 个真实网站、87 个子域名的 1,534 个样本|Task Success Rate, Turn Success Rate, Efficiency Metrics|Text Match, Image Match, Element Match, Action Match|https://visualwebbench.github.io/|
|WONDERBREAD [423]|Web|2024|否|聚焦业务流程管理（BPM）任务，如文档撰写、知识传递与流程改进|2,928 条人工示范，覆盖 598 条不同工作流|Task Success Rate, Step Success Rate, Efficiency Score, Completion under Policy|Text Match, Action Match, State Information|https://github.com/HazyResearch/wonderbread|
|WebOlympus [424]|Web|2024|是|面向 web agents 的开放平台，简化在真实网站上的演示、评估与数据收集|50 个任务|Task Success Rate, Step Success Rate|Action Match|/|
|BrowserGym [426]|Web|2024|是|提供统一、可扩展且开源的 Web 智能体评估环境，具有标准化 API 与观测|包含 MiniWoB(++)（125 个任务）、WebArena（812 个任务）与 WorkArena（每级最多 341 个任务）等基准|Task Success Rate, Step Success Rate, Turn Success Rate, Efficiency Metrics|基于文本的匹配与元素匹配|https://github.com/ServiceNow/BrowserGym|
|WebWalkerQA [427]|Web|2025|是|评估 LLM 在处理深层次、结构化且真实的 Web 导航与推理任务方面的能力|680 个高质量问答对|Task Success Rate, Efficiency Score|Text Match, Action Match|https://github.com/Alibaba-NLP/WebWalker|
|WebGames [428]|Web|2025|是|包含 50+ 个交互式挑战的综合基准，用于评估通用 Web 浏览智能体能力，并提供带可验证真值的封闭测试环境|50+ 个挑战|Task Success Rate|Action Match|https://github.com/convergence-ai/webgames|

表 35：Web GUI 基准概览（四）——安全性与专题评估

|基准（Benchmark）|平台（Platform）|年份（Year）|是否实时（Live）|亮点（Highlight）|规模（Data Size）|指标（Metric）|测量方式（Measurement）|链接（Link）|
|---|---|---|---|---|---|---|---|---|
|SafeArena [429]|Web|2025|是|首个专门评估 Web 智能体“被恶意使用”风险的基准，通过同时测试安全与有害任务完成能力来衡量安全性|500 个任务|Task Success Rate, Completion under Policy, Risk Ratio|Text Match, State Information|https://safearena.github.io|
|WABER [430]|Web|2025|是|提出 Efficiency 与 Reliability 两个新评估指标，超越传统的成功率评估|655 个任务|Task Success Rate, Efficiency Score|Action Match, State Information|https://github.com/SumanKNath/WABER|
|Online-Mind2Web [431]|Web|2025|是|面向真实在线环境的评估基准，反映真实用户在动态网页上的交互过程|来自 136 个网站的 300 个任务|Task Success Rate, Efficiency Score|Image Match, Action Match, State Information, LLM-as-a-Judge Evaluation|https://github.com/OSU-NLP-Group/Online-Mind2Web|
|AgentDAM [432]|Web|2025|是|首个系统评估多模态、真实 Web 环境中代理式模型隐私泄露风险的基准|246 条人工标注测试用例|Task Success Rate, Risk Ratio|Action Match, Text Match|https://github.com/facebookresearch/ai-agent-privacy|
|AgentRewardBench [433]|Web|2025|否|首个系统评估基于 LLM 的“裁判模型”（judge）在多种 Web 智能体任务上与人类专家标注一致性的基准|1,302 条轨迹、351 个任务|Task Success Rate, Completion under Policy|Image Match, Element/State Match|https://agent-reward-bench.github.io|
|RealWebAssist [434]|Web|2025|否|首个基于真实用户顺序指令、以自然且往往含糊语言表达的长程 Web 助手评估基准|1,885 条指令|Task Success Rate, Step Success Rate, Efficiency Score|Action Match|https://scai.cs.jhu.edu/projects/RealWebAssist/|
|REAL [435]|Web|2025|是|提供 Airbnb、Amazon、Gmail 等真实网站的确定性高保真副本，用于安全、可复现且可配置的多轮 GUI 智能体评估|跨 11 个确定性网站的 112 个任务|Task Success Rate|Text Match, Action Match, State Information Match|https://github.com/agi-inc/agisdk|
|BEARCUBS [436]|Web|2025|是|强调与真实网页的交互，包含视频、音频、3D 等多模态任务，弥补仅依赖静态或模拟环境基准的不足|111 个问题|Task Success Rate, Efficiency Score|Text Match, Action Match|https://bear-cubs.github.io|
|WASP [437]|Web|2025|是|首个在真实沙箱 Web 环境下系统评估 Web 智能体在提示注入攻击下安全性的端到端基准|84 个测试用例|Task Success Rate, Completion under Policy, Risk Ratio|Action Match, State Information|/|

综合来看，这些 Web 基准在真实性、多模态能力、安全性与企业适用性等多个方面推动了 Web GUI 智能体评估的发展，反映了构建能够处理复杂 Web 交互的高级智能体时不断演进的挑战与需求。
**9.5 移动智能体基准（Mobile Agent Benchmarks）**

在移动平台上评估 GUI 智能体存在一系列独特挑战：交互方式高度多样，应用本身结构复杂且更新频繁。为应对这些挑战，研究者提出了多种基准，每一种都从不同角度推动了移动智能体评估的发展。下面我们先通过表 36 与表 37 总览这些移动端基准，随后对若干代表性工作进行简要介绍。

表 36：移动 GUI 智能体基准概览（一）

|基准（Benchmark）|平台（Platform）|年份（Year）|是否实时（Live）|亮点（Highlight）|规模（Data Size）|指标（Metric）|测量方式（Measurement）|链接（Link）|
|---|---|---|---|---|---|---|---|---|
|AndroidEnv [263]|Android|2021|是|基于 Android 生态的开源交互平台，提供 30 余个应用中的 100+ 个任务，主要面向强化学习研究|100+ 个任务|NA|NA|https://github.com/google-deepmind/android_env|
|PIXELHELP [144]|Android|2020|否|提供自然语言指令与 UI 动作配对语料，覆盖四类任务，用于研究语言到 UI 交互的 grounding|187 条多步指令|Step Success Rate|Element Match, Action Match|https://github.com/google-research/google-research/tree/master/seq2act|
|Mobile-Env [438]|Android|2024|是|面向真实应用交互的综合 Android GUI 基准工具包，支持可控实验|224 个任务|Task Success Rate, Step Success Rate|Text Match, Element Match, Image Match, State Information|https://github.com/X-LANCE/Mobile-Env|
|B-MOCA [439]|Android|2024|是|在真实任务上评估移动设备控制智能体，并通过 UI 布局与语言随机化测试泛化能力|131 个任务|Task Success Rate|Element Match, State Information|https://b-moca.github.io/|
|AndroidWorld [440]|Android|2024|是|提供动态 Android 环境，用于多样自然语言指令测试|116 个任务|Task Success Rate|State Information|https://github.com/google-research/android_world|
|Mobile-Eval [158]|Android|2024|是|基于主流 Android 应用，设计常见移动交互任务的评测基准|30 条指令|Task Success Rate, Step Success Rate, Efficiency Score|Text Match, Path Length|https://github.com/X-PLUG/MobileAgent|
|DroidTask [156]|Android|2024|是|支持在真实应用中探索和记录任务轨迹的 Android 自动化基准|158 个任务|Step Success Rate, Task Success Rate|Element Match, Action Match|https://github.com/MobileLLM/AutoDroid|
|AITW [358]|Android|2023|否|受 PIXELHELP 启发的大规模 Android 交互数据集，覆盖多样设备操作|715,142 个 episode|Task Success Rate, Step Success Rate|Action Match|https://github.com/google-research/google-research/tree/master/android_in_the_wild|
|AndroidArena [441]|Android|2024|是|聚焦 Android 生态中的日常跨应用与约束任务，提供单应用与多应用交互场景|221 个任务|Task Success Rate, Step Success Rate, Efficiency Score|Action Match, Path Length|https://github.com/AndroidArenaAgent/AndroidArena|
|ANDROIDLAB [363]|Android|2024|是|提供结构化评估框架与 138 个任务，支持文本-only 与多模态智能体的一致评估|138 个任务|Task Success Rate, Step Success Rate, Efficiency Score|Element Match, Image Match|https://github.com/THUDM/Android-Lab|
|GTArena [447]|移动应用|2024|否|为 GUI 缺陷引入 Transition Tuple 表示，支持大规模缺陷数据集与端到端自动测试|10,000+ 次 GUI 显示与交互|Task Success Rate, Step Success Rate|Text Match, Element Match, Action Match|https://github.com/ZJU-ACES-ISE/ChatUITest|

在早期工作中，PIXELHELP [144] 是一个重要里程碑，它聚焦于将自然语言指令映射为移动端用户界面上的动作。为解决复杂多步任务解释与执行的难题，PIXELHELP 构建了一个由 187 条多步指令组成的数据集，将英文指令与人工在模拟器上的具体操作逐一配对，覆盖四类任务。该基准通过“完全匹配（Complete Match）”与“部分匹配（Partial Match）”等指标评估任务准确率，为后续移动 GUI grounding 研究打下基础。

ANDROIDLAB [363] 则建立了一个面向 Android 自主智能体的系统化评估框架。它不仅定义了统一的动作空间，还设计了同时支持文本-only 与多模态模型的运行模式。通过提供基于 XML 与 SoM 的两种操作模式，ANDROIDLAB 使 LLM 与多模态模型（LMM）能够在等价环境中模拟真实交互。该基准覆盖 9 个应用共 138 个任务，并使用 Success Rate 与 Reversed Redundancy 等指标衡量智能体的任务完成度与行为冗余。

为进一步考察智能体在同时处理 API 与 UI 操作方面的能力，Mobile-Bench [442] 提出了一种创新方案，将二者结合在真实 Android 环境中进行评测。其多应用设置与三类不同难度任务，要求智能体能够从简单到复杂地处理移动交互，超越传统单应用场景。评估过程中采用 CheckPoint 指标，在关键动作步骤处对智能体进行评分，从而洞察其规划与决策能力。

强调安全性的 MobileSafetyBench [443] 提供了一个结构化的移动设备控制评估框架，重点考察“有用性”与“安全性”之间的平衡。该基准在 Android 模拟器中设定常见任务，通过多层风险评估（包括法律合规与隐私保护）来测试智能体行为。其中一个独特设计是“间接提示注入（indirect prompt injection）”测试，用于检验智能体在受到环境干扰时的鲁棒性。评估不仅关注任务完成情况，也衡量风险缓解成功率，从而促进 LLM 可靠性与安全自动化控制的研究。

SPA-BENCH [444] 将评估范围扩展到了多语言与多应用场景，它为智能手机智能体提供了大规模基准，覆盖单应用与跨应用任务，并采用即插即用式框架便于集成不同智能体。数据涵盖多种 Android 应用（包括系统应用与第三方应用），在成功率、效率与资源使用等维度上全面测量智能体对 UI 的理解以及跨应用导航能力。

面向高效且用户友好的评估，MobileAgentBench [446] 提出了专为 Android 设备智能体设计的基准。它提供全自动的测试流程，通过最终 UI 状态匹配与实时应用事件跟踪来判断任务是否完成。该基准在 10 个开源 Android 应用上构建了 100 个任务，并按难度分级，允许存在多条成功路径，从而提升评估的可靠性与适用性。任务成功率、效率、时延与 token 成本等综合指标，为智能体性能分析提供了细粒度视角。

LlamaTouch [445] 则提出了面向真实 Android 环境的移动 UI 任务自动化基准与测试平台，强调“忠实且可扩展”的评估。它通过将智能体执行轨迹与预先标注的“关键状态”进行匹配，实现对任务完成情况的精准判定，而不受具体路径差异与动态 UI 元素影响。LlamaTouch 覆盖 57 个应用、496 个任务，并结合像素级截图与文本层级结构，显著降低了误判，支持多种任务难度。

Zhao 等人提出的 GTArena [447] 为自动化 GUI 测试智能体构建了形式化框架与基准。GTArena 以“转移元组（Transition Tuple）”为核心数据结构，系统性地捕获与分析 GUI 缺陷，评估测试意图生成、任务执行与缺陷检测三大任务。其数据集包含真实缺陷、人工注入缺陷与合成缺陷，为 GUI 测试智能体提供了首个系统化评估平台。

在上述基础之上，一系列更高层次、面向业务与安全等需求的移动基准也相继出现，如表 37 所总结。

表 37：移动 GUI 智能体基准概览（二）——高级与安全评估

|基准（Benchmark）|平台（Platform）|年份（Year）|是否实时（Live）|亮点（Highlight）|规模（Data Size）|指标（Metric）|测量方式（Measurement）|链接（Link）|
|---|---|---|---|---|---|---|---|---|
|A3 [448]|移动 Android|2025|是|提出业务级、基于 LLM 的自动评估流程，大幅降低人工与代码开发成本|21 个常用应用上的 201 个任务|Task Success Rate|Element Match, Action Match|https://yuxiangchai.github.io/Android-Agent-Arena/|
|LlamaTouch [445]|移动 Android|2024|是|通过将执行轨迹与标注关键状态对齐，实现忠实且可扩展的移动 UI 任务自动化评估|57 个应用上的 496 个任务|Task Success Rate, Step Success Rate, Efficiency Score|Text Match, Action Match, State Information Match|https://github.com/LlamaTouch/LlamaTouch|
|MobileAgentBench [446]|移动 Android|2024|是|在真实 Android 设备上提供全自动评估流程，支持多路径成功条件与灵活成功判定|10 个开源应用上的 100 个任务|Task Success Rate, Efficiency Score, Latency, Token Cost|State Information（UI 状态匹配）|https://mobileagentbench.github.io/|
|Mobile-Bench [442]|Android|2024|是|在多应用场景中同时支持 UI 与 API 动作，通过 checkpoint 机制评估复杂任务|832 条记录（200+ 个任务）|Task Success Rate, Step Success Rate, Efficiency Score|Action Match, Path Length|https://github.com/XiaoMi/MobileBench|
|Mobile Safety Bench [443]|Android|2024|是|优先评估移动控制任务中的安全性，为“有用性、隐私与法律合规”等维度设计专门任务|100 个任务|Task Success Rate, Risk Mitigation Success|Action Match（考虑安全性）, Element Match, State Information|https://mobilesafetybench.github.io/|
|SPA-BENCH [444]|Android|2024|是|大规模、支持英文与中文的评估框架，覆盖单应用与跨应用任务，提供即插即用结构|340 个任务|Task Success Rate, Step Success Rate, Efficiency Score|Action Match, State Information, Time Spent, API Cost|https://spa-bench.github.io|
|SPHINX [449]|Android|2025|是|提供全自动移动基准套件，引入多维度评估框架|284 个常见任务|Task Success Rate, Efficiency Score, Completion under Policy, Turn Success Rate|Text Match, Image Match, Element Match, Action Match|/|
|AEIA-MN [450]|移动 Android|2025|是|提出 Active Environment Injection Attack（AEIA）框架，通过主动操纵环境元素（如通知）误导多模态 LLM 智能体|AndroidWorld 上 61 个任务 + AppAgent 上 45 个任务|Task Success Rate, Risk Ratio, Efficiency Score|Text Match, State Information, Action Match|/|
|AutoEval [451]|移动 Android|2025|是|提出完全自动化的移动智能体评估框架，无需手工设计奖励信号或大量评测代码|93 个任务|Task Success Rate|Action Match, State Information|/|
|LearnGUI [321]|移动 Android|2025|是|首个系统研究“少量示范（few-shot demonstration）”学习的移动 GUI 智能体基准，包含离线与在线环境|离线：44 个应用上带 k-shot 变体的 2,252 个任务；在线：20 个应用上的 101 个交互式任务|Task Success Rate|Action Match|https://lgy0404.github.io/LearnAct|

总体而言，这些移动端基准在任务复杂度、安全性、效率与可扩展性方面显著推动了移动 GUI 智能体评估的发展，为构建更强大、更可靠的移动智能体奠定了基础。

**9.6 计算机端智能体基准（Computer Agent Benchmarks）**

在桌面计算机环境中评估 GUI 智能体，涉及多种应用与复杂工作流。近年来提出了若干基准，从不同侧面考察智能体在桌面场景中的能力，并推动该领域的演进。我们在表 38 中汇总了代表性计算机端 GUI 智能体基准。

表 38：桌面/计算机 GUI 智能体基准概览

|基准（Benchmark）|平台（Platform）|年份（Year）|是否实时（Live）|亮点（Highlight）|规模（Data Size）|指标（Metric）|测量方式（Measurement）|链接（Link）|
|---|---|---|---|---|---|---|---|---|
|OSWorld [419]|Linux、Windows、macOS|2024|是|为多模态智能体提供可扩展的真实计算机环境，支持任务构建、执行式评估与交互式学习|Ubuntu 上 369 个任务，Windows 上 43 个任务|Task Success Rate|基于执行的状态信息（如内部文件解析、权限管理等）|https://os-world.github.io/|
|Windows Agent Arena [453]|Windows|2024|是|在 OSWorld 基础上专门面向 Windows OS，构建多样多步任务，允许使用大量应用与工具|154 个任务|Task Success Rate|与 OSWorld 一致，可通过云端并行扩展|https://microsoft.github.io/WindowsAgentArena|
|OmniACT [459]|macOS、Linux、Windows|2024|否|评估智能体在多 OS 环境下为桌面与 Web 应用生成可执行程序的能力，侧重多模态挑战|9,802 条数据|Task Success Rate, Step Success Rate|Action Match|https://huggingface.co/datasets/Writer/omniact|
|VideoGUI [460]|Windows|2024|否|聚焦来自教学视频的视觉中心任务，强调在 Photoshop、Premiere 等应用中的规划与动作精度|178 个任务、463 个子任务|Task Success Rate|State Information, Action Match|https://showlab.github.io/videogui|
|Spider2-V [454]|Linux|2024|是|在真实企业软件环境中评估数据科学与工程工作流，从数据摄取到可视化的全流程|494 个任务|Task Success Rate|Action Match, State Information|https://spider2-v.github.io|
|Act2Cap [327]|Windows|2024|是|通过光标作为关键视觉线索，对 GUI 操作进行视频级叙述，涵盖点击、输入、拖拽等多种交互|4,189 个样本|Step Success Rate|Element Match|https://showlab.github.io/GUI-Narrator|
|OFFICEBENCH [455]|Linux|2024|是|在 Word、Excel 等办公应用间构建复杂多步工作流，考察跨应用自动化能力|300 个任务|Task Success Rate|Action Match, Text Match, State Information|https://github.com/zlwang-cs/OfficeBench|
|AssistGUI [109]|Windows|2024|是|首个专门面向任务导向桌面 GUI 自动化的基准|来自 9 个常用应用的 100 个任务|Task Success Rate, Efficiency Score|Element Match, Action Match|https://showlab.github.io/assistgui/|
|WorldGUI [456]|Windows|2025|是|首个专门评估“动态 GUI 交互”的基准，通过引入多种初始状态考查适应性|10 个 Windows 应用共 315 个任务|Task Success Rate, Efficiency Score|Image Match, Element Match, Action Match|/|
|UI-Vision [457]|桌面（Windows、Linux）|2025|否|首个专门为桌面 GUI 智能体设计的大规模基准|共 8,227 个 query–label 对|Task Success Rate|Action Match, Text Match|https://uivision.github.io|
|Computer Agent Arena [458]|Windows、Ubuntu、macOS|2025|是|首个面向多模态 LLM 智能体、开放式的大规模真实桌面环境评估平台|用户自定义任务|Task Success Rate|人工评审（human evaluators）|https://arena.xlang.ai/|

在桌面端早期工作中，Act2Cap [327] 关注通过视频形式捕捉与叙述 GUI 操作，将光标作为关键视觉引导信号。该基准强调点击与拖拽等细粒度操作的时间与空间精度，在 4,189 个样本上评估智能体的语义与时空对齐能力，是推进 GUI 密集型任务自动化的重要基础。

为提供可扩展且真实的计算机环境，OSWorld [419] 提出了支持 Ubuntu、Windows 与 macOS 的统一框架，涵盖任务构建、基于执行结果的评估与交互式学习。OSWorld 以真实操作系统为基础，可运行任意应用与开放式计算机任务，并在 Ubuntu（369 个任务）与 Windows（43 个任务）上使用执行式成功率等指标进行严格评估。

在此基础上，WindowsArena [453] 将框架适配至 Windows，构建了 150+ 个多步任务，要求智能体在真实 Windows 环境中展示规划、界面理解与工具使用能力。为解决评估耗时问题，WindowsArena 支持在 Azure 云上并行部署，大幅缩短评测时间，使得在多应用与多 Web 域上的全面测试成为可能。

面向办公自动化，OFFICEBENCH [455] 提出了在 Linux Docker 环境中运行的评估框架，覆盖 Word、Excel、邮件等多个办公应用。该基准通过复杂多步跨应用工作流考察智能体在真实办公场景中的操作整合能力，并结合执行结果、文本与状态信息进行多维度评估。

为评估数据科学与工程工作流，Spider2-V [454] 在 20 个企业级应用中构建了 494 个真实任务，覆盖从数据仓库到可视化的全流程。该基准同时考察代码生成与复杂 GUI 交互能力，在 Ubuntu 上通过信息验证与文件比对等多种方式进行评估。

AssistGUI [109] 则聚焦于生产力软件，提出 Actor-Critic 式具身智能体框架，能够进行层次化任务规划、GUI 解析与动作生成。其数据集涵盖设计、办公与系统设置等多种任务，并配套项目文件以支持复现。通过强调以结果为导向的评估与像素级精度，AssistGUI 揭示了当前基于 LLM 的桌面智能体在复杂软件工作流管理方面的潜力与局限。

WorldGUI [456] 专门针对 Windows 平台下的动态交互进行评估。不同于静态基准，WorldGUI 通过设置多种初始状态来模拟真实使用情境（包括桌面与 Web 应用），要求智能体在非固定初始界面下完成任务。其 315 个任务涵盖 10 个常用应用，并配备教学视频、项目文件与多种预动作场景，构成一个全面且逼真的评估框架。

Computer Agent Arena [458] 则提出了全新的评测范式：基于云端的实时桌面环境，支持用户配置任务并在真实应用（如 Google Docs、VSCode、Slack 等）上测试智能体。该平台通过 head-to-head 智能体对比、人工评审与 Elo 排名等方式在 Web 浏览、编程与生产力任务上对通用数字智能体进行评估。目前支持 Windows 与 Ubuntu（计划支持 macOS），并允许用户自定义任务与软件/网站组合，为社区化评估与对比提供了基础。

这些桌面端基准共同构建了涵盖任务复杂度、跨应用自动化、可扩展性与高保真度的评估体系，为发展能够在真实桌面环境中执行复杂交互的高级智能体提供了关键支撑。

**9.7 跨平台智能体基准（Cross-Platform Agent Benchmarks）**

为了开发能够在多平台上运行的 GUI 智能体，跨平台基准至关重要。此类基准要求智能体适应不同环境与界面，全面评估其通用性与鲁棒性。我们在表 39 中总结了代表性的跨平台 GUI 智能体基准。

在这方面，VisualAgentBench（VAB）[374] 是一项开创性工作，用于评估 GUI 与多模态智能体在广泛的真实交互任务上的表现。它涵盖 Web（WebArena-Lite [412]）、Android（VAB-Mobile [363]）以及游戏环境，重点考察基于视觉的交互与高层决策任务。VAB 通过包含人类示范、程序求解器与模型自举在内的多层数据采集策略构建数据集，并以任务成功率等指标综合评估在导航、内容修改等任务上的表现，填补了 GUI 场景下 LLM 智能体基准的空白。

与之互补，CRAB [461] 提出了跨环境交互场景下的多模态语言模型智能体评估基准。它支持在 Ubuntu Linux 与 Android 之间无缝执行跨设备任务，通过图结构的评估方法将任务拆分为子目标，并容许存在多条正确路径，以更细致地衡量规划、决策与适应能力。其指标包括 Completion Ratio、Execution Efficiency、Cost Efficiency 与 Success Rate，为智能体性能提供多维度洞见。

在跨平台视觉智能体的 GUI grounding 方面，ScreenSpot [25] 提供了一个重要基准，强调使用截图而非结构化数据来完成任务。ScreenSpot 包含超过 600 张截图与 1,200 条多样指令，覆盖移动（iOS、Android）、桌面（macOS、Windows）与 Web 平台，主要通过点击准确率与定位精度评估智能体仅凭视觉线索识别并交互 GUI 元素的能力。面对多样 UI 元素与布局，ScreenSpot 直接反映了真实环境下的复杂性，是评估视觉 GUI 智能体跨平台能力的重要资源。

表 39：跨平台 GUI 智能体基准概览

|基准（Benchmark）|平台（Platform）|年份（Year）|是否实时（Live）|亮点（Highlight）|规模（Data Size）|指标（Metric）|测量方式（Measurement）|链接（Link）|
|---|---|---|---|---|---|---|---|---|
|VisualAgentBench [374]|Web、Android、游戏、虚拟具身环境|2024|是|首个面向视觉基础智能体的基准，覆盖 GUI 与多模态任务，强调在 Android、Web 与游戏环境中的视觉中心交互|4,482 条轨迹|Task Success Rate|Text Match|https://github.com/THUDM/VisualAgentBench|
|SPR Benchmark [462]|移动、Web、操作系统|2024|是|评估 GUI 屏幕阅读器在内容与布局描述上的能力|650 张带 1,500 个目标点/区域标注的截图|Task Success Rate, Efficiency Score|Text Match, Element Match|/|
|AgentStudio [237]|Windows、Linux、macOS|2024|是|用于构建与评测通用虚拟智能体的开放工具包，支持在多种软件应用中的复杂交互|NA|Step Success Rate|Action Match, State Information, Image Match|https://computer-agents.github.io/agent-studio/|
|CRAB [461]|Linux、Android|2024|是|跨环境基准，在移动与桌面设备上评估智能体，通过图结构评估方法处理多路径任务|120 个任务|Step Success Rate, Efficiency Score|Action Match|https://github.com/crab-benchmark|
|ScreenSpot [25]|iOS、Android、macOS、Windows、Web|2024|否|基于视觉的 GUI grounding 基准，仅用截图评估智能体在多平台之间识别与交互 GUI 元素的能力|1,200 条指令|Step Success Rate|Action Match|https://github.com/njucckevin/SeeClick|

总体来看，这些跨平台基准共同推动了能够在多种平台间无缝运行的 GUI 智能体的发展，通过提供全面的评估框架，为衡量与提升智能体在不同环境中的通用性与适应性提供了重要工具。

**9.8 小结（Takeaways）**

GUI 智能体基准的发展反映出评估环境正整体向更加真实、交互式与全面的方向演进。围绕以 LLM 为大脑的 GUI 智能体，当前基准研究呈现出以下关键趋势与未来方向：

1) 迈向更具交互性和现实感的环境（Towards More Interactive and Realistic Environments）：近期基准工作从合成场景逐步转向更加交互式与真实的环境。这一点体现在广泛采用模拟器、Docker 容器以及真实应用来构建“实时（live）”环境，更好地还原用户实际操作。此类环境不仅更准确地反映智能体能力，也在性能与鲁棒性方面提出了新的挑战。
2) 跨平台基准（Cross-Platform Benchmarks）：覆盖移动、Web 与桌面环境的跨平台基准，是评估 GUI 智能体泛化能力的重要一步。然而，不同平台存在各自的根本性差异与挑战。若能提供统一接口来访问平台特定信息（如 HTML 与 DOM 结构等），将极大简化基准实现与使用成本。未来工作应致力于标准化这些接口，推动跨环境的一致评估。
3) 增强人类交互与现实性（Increased Human Interaction and Realism）：越来越多基准引入接近真实使用场景的人机交互形式，如多轮对话与会话式任务。这些设置更贴近实际用例，对智能体处理动态、迭代交互的能力提出更高要求。随着 GUI 智能体复杂度提升，基准也需要不断演进，将这类细腻交互模式纳入评测范围，确保智能体能在以人为中心的场景中有效工作。
4) 可扩展性与自动化挑战（Scalability and Automation Challenges）：可扩展性仍是 GUI 智能体评估中的核心问题。构建真实任务与为每个任务设计评估方法往往需要大量人工努力。通过自动化任务生成与评估过程，可以在一定程度上缓解这一问题，并实现更大规模与更高效的基准测试。未来研究有必要探索自动任务构造与自动评估技术。
5) 强调安全、隐私与合规（Emphasis on Safety, Privacy, and Compliance）：越来越多的基准将安全、隐私与合规性纳入评估维度。随着智能体逐步进入敏感与受监管领域，这些考量愈发重要。鼓励这种趋势，有助于确保智能体不仅能高效完成任务，也能遵守必要的法律与伦理规范。未来的基准应进一步拓展这些维度，引入更贴近真实合规与数据安全要求的评估设置。

综上所述，GUI 智能体评测版图正在快速演进，以适应日益复杂与交互性增强的应用环境。通过加强跨平台评估、引入更具人性化的交互方式、解决可扩展性难题并优先考虑安全与合规，研究社区能够为下一代高级 GUI 智能体铺平道路。持续的创新与合作将是完善基准体系、全面刻画现代智能体多维能力的关键，从而最终促进更加直观、高效的人机交互体验。
###### 10 以 LLM 为大脑的 GUI 智能体的应用（APPLICATIONS OF LLM-BRAINED GUI AGENTS）

随着以 LLM 为大脑的 GUI 智能体不断成熟，越来越多的应用开始利用这一概念，构建更加智能、友好且支持自然语言交互的界面。这些进展体现在学术研究、开源项目以及工业级解决方案之中。典型应用主要包括：（i）GUI 测试——从传统脚本式方法逐步演进为更加直观、基于自然语言的测试流程；以及（ii）虚拟助手（virtual assistants）——通过自然语言接口以更自适应的方式自动化用户的日常任务。

**10.1 GUI 测试（GUI Testing）**

GUI 测试用于验证软件应用的图形用户界面是否满足既定需求、功能规范与用户体验标准。它不仅检查按钮、菜单、窗口等界面元素是否存在与正确显示，还要验证用户交互（如点击、输入、拖拽）后的响应是否符合预期。最初，GUI 测试主要以人工方式进行；后来随着 Selenium、Appium 等自动化工具的出现，人们开始通过脚本自动化重复测试任务，从而提升覆盖率并降低成本 [35], [543]。

以 LLM 为核心的大模型 GUI 智能体引入之后，GUI 测试范式发生了显著转变：非专业人员也可以通过自然语言描述测试用例，由智能体自动执行 GUI 操作。借助这类智能体，可以更自然地覆盖多种测试场景，包括通用功能测试、输入生成与缺陷复现，而无需传统脚本 [543]。

图 29 展示了一个基于 LLM 驱动的 GUI 智能体在 Windows 上测试字体大小调整的示例。测试人员只需提供一条自然语言测试用例描述，智能体便会自主执行 UI 操作、导航设置菜单，并利用屏幕理解能力验证字体大小是否被正确调整到 175%。这种方式大幅减少了人工或脚本测试所需的工作量。下面我们详细回顾几类典型的 LLM 驱动 GUI 测试工作，并在表 40–42 中给出汇总。

_10.1.1 通用测试（General Testing）_

早期探索表明，像 GPT-3 这样的 LLM 可以通过理解自然语言测试用例并编程式执行它们，来自动化 GUI 测试。Daniel 和 Anne 的工作 [542] 将 GUI 状态与 GPT-3 提示结合，使用 Selenium 与 OpenCV 采集界面信息，从而减少手写脚本的需求，实现黑盒测试。后续工作 [463] 进一步采用 GPT-4 与 Selenium WebDriver 进行 Web 应用测试，相比传统“猴子测试”（monkey testing），在分支覆盖率上取得了显著提升，展示了 LLM 在简化 GUI 测试流程、提升覆盖率与效率方面的优势。

GPTDroid [125] 将 GUI 测试重新表述为交互式问答（Q&A）任务。它从 GUI 页面中抽取结构化语义信息，并引入记忆机制以支持长期探索，使得活动覆盖率提升了 32%，并能够高精度发现关键缺陷。这一思路凸显了将对话式接口与记忆机制结合，用于全面应用测试的潜力。

在 Android 环境中，DroidAgent [464] 提出了“意图驱动（intent-driven）”测试框架。它将 GUI 状态编码为 JSON，并利用 LLM 感知这些状态、规划现实感更强的测试任务，从而自动生成并执行高层次任务。凭借设定高层目标与实现高特性覆盖率的能力，DroidAgent 展示了基于意图的测试在功能验证中的巨大优势。

ProphetAgent [482] 则提出了一种新颖的 Android GUI 测试脚本生成方法：从自然语言描述中自动综合测试脚本。不同于直接将 LLM 应用于截图或行为数据，ProphetAgent 首先构建带语义注释的“聚类 UI 转移图”（Clustered UI Transition Graph, CUTG），再在其上进行推理，从而更准确地将自然语言测试步骤映射为具体 GUI 操作。实验证明，其任务完成率可达 78.1%、动作准确率达 83.3%，并在字节跳动的真实场景中将测试人员工作量降低了 70% 以上，显示出 LLM 与显式语义图结合在 GUI 测试中的强大效果。

AUITestAgent [465] 扩展了 LLM 驱动 GUI 测试的能力，将自然语言需求与 GUI 功能之间建立更紧密的桥梁。它采用多模态分析与动态智能体组织框架，能够高效执行从简单到复杂的测试指令，并通过多源数据抽取策略实现更精确的功能验证，特别适用于商用应用的功能测试。

VisionDroid [466] 引入视觉为中心的 GUI 测试框架，通过将截图中的边界框信息与视图层级文本对齐，利用多模态模型增强对 GUI 语境与应用逻辑的理解，从而发现非崩溃类缺陷（non-crash bugs）。这一工作表明，视觉与文本对齐对于识别逻辑不一致性和探索常规方法难以覆盖的功能区域至关重要。

在可访问性测试方面，AXNav [467] 针对 iOS 中的辅助功能（如 VoiceOver、Dynamic Type、粗体文本与按钮形状等）构建了自动化测试流程。它结合截图、UI 元素检测模型与 OCR，从自然语言测试说明出发自动执行测试，并生成带标注的视频用于交互式审查，为大规模、用户友好的辅助功能测试提供了新的方案。

LLMigrate [473] 则展示了如何利用多模态 LLM 在不访问源代码的情况下，将基于使用行为的 UI 测试在不同 Android 应用之间迁移。通过处理 DOM 和截图信息，LLMigrate 自动迁移 UI 测试用例，显著减少了跨应用测试构造的人工成本。

表 40：基于 LLM 的 GUI 通用测试与可访问性测试工作概览

|项目（Project）|类别（Category）|平台（Platform）|模型（Model）|感知（Perception）|动作（Action）|场景（Scenario）|亮点（Highlight）|链接（Link）|
|---|---|---|---|---|---|---|---|---|
|Daniel and Anne [542]|General testing|通用平台|GPT-3|GUI 结构与状态|标准 UI 操作|利用自然语言测试用例自动化软件测试流程|将 GPT-3 的语言理解能力用于 GUI 软件测试，使测试人员通过文本描述测试用例即可进行交互式测试|https://github.com/neuroevolution%2Dai/SoftwareTestingLanguageModels|
|Daniel and Anne [463]|General testing|Web 平台|GPT-4|HTML DOM 结构|标准 UI 操作|提升 Web 应用的分支覆盖率与测试效率|使用 GPT-4 的自然语言理解与推理能力实现端到端 GUI 测试，在覆盖率与效率上优于传统方法|https://github.com/SoftwareTestingLLMs/WebtestingWithLLMs|
|GPTDroid [125]|General testing|移动 Android|GPT-3.5|UI 视图层级文件|标准 UI 操作与复合动作|自动化 GUI 测试以提升覆盖率并高效发现缺陷|将 GUI 测试表述为问答任务，利用 LLM 的对话与推理能力实现类人测试交互|https://github.com/franklinbill/GPTDroid|
|DroidAgent [464]|General testing|移动 Android|GPT-3.5, GPT-4|JSON 形式的 GUI 状态表示|标准 UI 操作、高层 API 与自定义动作|基于语义与意图的 Android GUI 测试自动化|自动生成并执行高层次、贴近真实使用场景的测试任务，在特性覆盖率上优于传统方法|https://github.com/coinse/droidagent|
|AUITestAgent [465]|General testing|移动 Android|GPT-4|GUI 截图、UI 层级文件及 Vision-UI 等 CV 技术|标准 UI 操作|自动化 GUI 功能测试|采用动态智能体组织与多源数据抽取策略，实现面向步骤的测试与精确功能验证|https://github.com/bz-lab/AUITestAgent|
|VisionDroid [466]|General testing|移动 Android|GPT-4|带边界框标注的 GUI 截图与视图层级文件|标准 UI 操作|识别非崩溃缺陷|通过视觉驱动提示与 GUI 文本对齐，结合视觉-语言模型提升对 GUI 语境与应用逻辑的理解，从而发现细粒度缺陷|https://github.com/testtestA6/VisionDroid|
|AXNav [467]|Accessibility testing|iOS 移动设备|GPT-4|GUI 截图、UI 元素检测模型与 OCR|手势操作、截图采集与高亮潜在可访问性问题|自动化可访问性测试工作流（如 VoiceOver、Dynamic Type 等）|基于自然语言测试指令自动执行可访问性测试，并生成带标注视频，支持可视化、交互式结果审查|/|
|LLMigrate [473]|General testing|移动 Android|GPT-4o|DOM 与截图|标准 UI 操作|在 Android 应用间自动迁移基于使用行为的 UI 测试|利用多模态 LLM 在无需源代码的前提下完成 UI 测试迁移|/|

_10.1.2 文本输入生成（Text Input Generation）_

在文本输入生成方面，Cui 等人 [468] 展示了 GPT-3.5 与 GPT-4 如何通过生成具有上下文意识的文本输入来提升 Android 应用测试效果。通过在多个应用上系统评估这些输入，他们表明 LLM 能在几乎无需人工干预的情况下提升测试覆盖率，并发现此前未报告的缺陷。

QTypist [469] 将文本输入生成建模为“填空式（cloze-style）”语言任务，利用 LLM 在给定 UI 视图层级的前提下生成语义合理的文本，从而自动化移动 GUI 测试。实验表明，该方法可使活动与页面覆盖率提高多达 52%。

_10.1.3 缺陷复现（Bug Replay）_

在缺陷复现方面，CrashTranslator [470] 通过将强化学习与 LLM 结合，实现了基于堆栈跟踪信息自动重放崩溃。其迭代式导航与崩溃预测流程显著减少了调试时间，并在多项指标上优于现有方法。

AdbGPT [471] 展示了如何通过少样本学习与链式思维推理（Chain-of-Thought）将文本化缺陷报告转化为可执行的 GUI 操作。借助动态推断 GUI 动作，AdbGPT 为移动缺陷复现提供了一种高效、轻量的解决方案。

BugCraft [478] 将 LLM 驱动 GUI 智能体用于游戏中的缺陷复现，特别针对开放式、复杂环境 Minecraft。该框架使用 GPT-4o 作为推理引擎，结合用户提交的文本缺陷报告、OmniParser [184] 提取的 GUI 视觉信息以及 Minecraft Wiki 的外部知识，生成并执行结构化复现步骤。动作通过自定义 Macro API 执行，使智能体能够稳健地与游戏 GUI 与环境交互。BugCraft 成功地将非结构化缺陷描述转换为可执行的游戏行为，体现了视觉增强型 LLM 智能体在软件测试与调试中的巨大潜力。

表 41：LLM 驱动 GUI 测试工作概览（二）——输入生成与缺陷复现

|项目（Project）|类别（Category）|平台（Platform）|模型（Model）|感知（Perception）|动作（Action）|场景（Scenario）|亮点（Highlight）|链接（Link）|
|---|---|---|---|---|---|---|---|---|
|Cui et al. [468]|Test input generation|移动 Android|GPT-3.5, GPT-4|GUI 结构与上下文信息|文本输入|为 Android 应用生成并验证文本输入|系统评估多种 LLM 生成的上下文相关文本输入，提升 UI 测试覆盖率并发现新缺陷|/|
|QTypist [469]|Test input generation|移动 Android|GPT-3|UI 层级文件|生成语义文本输入|自动化移动 GUI 测试|将文本输入生成建模为“填空式”语言任务，提高活动与页面覆盖率|/|
|CrashTranslator [470]|Bug replay|移动 Android|GPT-3|崩溃相关堆栈信息与 GUI 结构|标准 UI 操作|自动复现移动应用崩溃|结合强化学习与 LLM，从堆栈跟踪出发迭代导航 GUI 并复现崩溃，引入评分机制优化探索步骤|https://github.com/wuchiuwong/CrashTranslator|
|AdbGPT [471]|Bug replay|移动 Android|GPT-3.5|GUI 结构与层级|标准 UI 操作|通过提取“重现步骤（S2R）”实体自动复现缺陷|结合 prompt 设计、少样本学习与链式思维推理，将文本缺陷报告映射为 GUI 操作序列|https://github.com/sidongfeng/AdbGPT|
|MagicWand [472]|Verification|移动 Android|GPT-4V|UI 截图与分层 UI 控件树|标准 UI 操作|自动验证搜索引擎返回的“操作指南（How-to）”是否可行|采用三阶段流程：抽取指令、在模拟环境执行并根据执行结果重排序搜索结果|/|
|UXAgent [474]|Usability testing for web design|Web|自研模型|简化的 HTML 表示|标准 UI 操作|自动化 Web 应用可用性测试|通过模拟上千次用户交互采集定性与定量数据，使研究者在真实用户研究前获得早期反馈|https://uxagent.hailab.io|
|Guardian [475]|GUI Testing|移动 Android|GPT-3.5|GUI 结构与属性|标准 UI 操作|自主探索移动应用并验证核心功能|通过将规划任务卸载给外部运行时系统，改进 LLM 驱动的 UI 测试流程|/|
|Test-Agent [476]|GUI Testing|Android、iOS、HarmonyOS|未注明|GUI 截图与 UI 结构信息|标准 UI 操作|跨平台移动测试|利用 LLM 与多模态感知自动生成并执行测试用例，无需预写测试脚本|/|
|VLM-Fuzz [477]|GUI Testing|Android（移动端）|GPT-4o|GUI 截图与 UI 结构信息|标准 UI 操作与系统级动作|自动化 Android 应用测试以发现崩溃与缺陷|结合视觉-语言推理与启发式深度优先搜索系统性探索复杂 Android UI，在代码覆盖率上显著优于现有方法|/|

_10.1.4 其他验证与高级测试（Verification and Advanced Testing）_

在验证类应用中，MagicWand [472] 展示了 LLM 在自动验证“操作指南（How-to）”方面的潜力。它从搜索引擎结果中抽取候选指令，在模拟环境中执行，并根据执行效果对结果进行重排序，从而提升基于搜索的任务自动化可靠性。

在更广泛的 GUI 测试场景中，ReuseDroid [479] 利用视觉上下文与动态反馈机制，实现 Android 应用之间 GUI 测试用例的迁移，相比以往基于映射或 LLM 的方法显著提高了迁移成功率。SeeAct-ATA 与 PinATA [480] 则将自主 Web 智能体 SeeAct [17] 适配为“自动测试智能体”（Autonomous Test Agents, ATA），探索在 Web 应用端到端测试中的应用。GERALLT [481] 将 LLM 驱动测试扩展到真实桌面 GUI 应用（非 Web 或移动），通过结构化 GUI 解析与 LLM 控制与评估相结合，在无预定义脚本的前提下发现不直观行为、界面不一致与功能错误。Agent for User [483] 引入了多智能体框架，让每个 LLM 智能体在虚拟设备上模拟一个用户，从而测试多用户交互特性。

表 42：LLM 驱动 GUI 测试工作概览（三）——游戏、桌面与多用户场景

|项目（Project）|类别（Category）|平台（Platform）|模型（Model）|感知（Perception）|动作（Action）|场景（Scenario）|亮点（Highlight）|链接（Link）|
|---|---|---|---|---|---|---|---|---|
|BugCraft [478]|Bug reproduction|Windows 电脑（Minecraft）|基于 GPT-4o 的 BugCraft 框架|GUI 截图|标准 UI 操作|根据用户缺陷报告自动在 Minecraft 中复现崩溃|首个在复杂开放世界游戏（Minecraft）中，结合 LLM 智能体、视觉 UI 解析与结构化动作执行的端到端崩溃复现框架|https://bugcraft2025.github.io/|
|ReuseDroid [479]|GUI Testing|移动 Android|基于 GPT-4o 的 ReuseDroid|GUI 截图与控件属性|标准 UI 操作|在具相似功能但操作逻辑不同的 Android 应用间迁移 GUI 测试用例|利用视觉上下文与动态反馈机制，在测试迁移成功率上显著优于以往映射与 LLM 方法|/|
|SeeAct-ATA & PinATA [480]|GUI Testing|Web|SeeAct [17]|GUI 结构与 DOM|标准 UI 操作|自动化 Web 应用端到端测试|首次尝试将 LLM 驱动自主 Web 智能体适配为自动测试智能体（ATA），用于 Web 测试|/|
|GERALLT [481]|GUI Testing|桌面（Windows/Linux）|GPT-4o|GUI 截图与 UI 结构信息|标准 UI 操作|在无脚本前提下发现桌面 GUI 中的不直观行为与功能错误|率先在真实桌面 GUI 应用上系统化引入 LLM 驱动测试，结合结构化 GUI 解析与 LLM 控制/评估|https://github.com/DLR-SC/GERALLT|
|ProphetAgent [482]|GUI Testing|Android 移动端|SemanticAgent 与 GenerationAgent（基于 GPT-4o 等基础模型）|带语义注释的 XML UI 树|可执行 UI 测试脚本|从自然语言生成回归与兼容性测试用例|将 LLM 推理与语义增强 GUI 图（CUTG）结合，在测试脚本综合性能与效率上显著优于现有工具|https://github.com/prophetagent/Home|
|Agent for User [483]|GUI Testing|Android 移动端|GPT-4|XML 视图层级|标准 UI 操作|面向多用户交互特性的自动化测试|提出多智能体框架，每个智能体在虚拟设备上模拟一名用户，以测试多用户场景|/|

总的来说，LLM 驱动的 GUI 智能体通过引入自然语言驱动的方法、视觉对齐技术以及自动化崩溃复现，已经深刻重塑了 GUI 测试领域。这些创新显著提升了测试覆盖率、效率与可达性，为构建下一代智能 GUI 测试框架树立了新标杆。
**10.2 虚拟助手（Virtual Assistants）**

虚拟助手（virtual assistant），如 Siri[32]，是一类由 AI 驱动的应用，用于在多种平台（包括 Web 浏览器、手机与桌面电脑）上为用户执行任务、回答问题与执行命令。早期的虚拟助手通常只能通过语音或文本接收简单指令，并基于规则或预定义工作流（类似传统 RPA）返回结果，主要支持设置闹钟、查询天气等基础任务。

随着大语言模型与智能体技术的发展，虚拟助手的能力得到了显著提升。借助以 LLM 为大脑的 GUI 智能体，它们如今能够通过文本或语音在设备的 GUI 上执行更复杂、具上下文的交互，并根据不同应用与用户需求提供个性化响应。这一演进使虚拟助手从“简单工具”转变为能够管理复杂工作流、提升跨平台生产力的智能助手。图 30 展示了一个概念性示例：在智能手机上，由 GUI 智能体驱动的虚拟助手通过聊天界面与用户交互，为用户配置截图快捷方式等功能，尤其有助于不熟悉手机功能的用户，将复杂操作简化为对话命令。

为系统了解基于 GUI 智能体的虚拟助手在研究、开源项目与生产系统中的最新进展，原文在表 43 与表 44 中给出了概览，并在本小节中逐一讨论。

_10.2.1 研究工作（Research）_

近期研究通过引入 LLM 驱动的 GUI 智能体，显著提升了虚拟助手在各类应用中的智能性与适应性。

首先，一些工作将 LLM 纳入 GUI 自动化中，以增强业务流程自动化能力。ProAgent [485] 提出了“Agentic Process Automation”，构建了在 GUI 环境中自动创建与执行工作流的系统。它通过 ControlAgent 与 DataAgent 等子智能体协同，支持在 Slack 与 Google Sheets 等应用中执行动态分支、报表生成等复杂操作。与传统 RPA 相比，ProAgent 能通过自然语言与结构化任务描述灵活解释并执行任务，大幅减少人工干预，展示了 LLM 驱动虚拟助手在业务场景中的变革潜力。

在移动平台上，LLMPA [486] 是一项开创性框架，它利用 LLM 在诸如支付宝等应用中自动化多步任务。LLMPA 直接与应用 GUI 交互，模拟用户点击与输入，并通过 UI 树解析与目标检测实现精确环境理解。其可控校准模块用来保证动作逻辑的合理性，体现了 LLM 驱动虚拟助手在处理复杂工作流与真实业务场景方面的能力。

PromptRPA [490] 则专注于通过自然语言提示自动化智能手机任务。该框架采用多智能体架构，在手机 GUI 环境中执行任务，解决界面更新与用户输入多样性等问题。系统结合 OCR 与分层 GUI 分析等高级感知手段理解界面，并通过实时反馈与迭代改进机制强化用户中心设计，突出自然语言驱动虚拟助手在智能手机上的实际价值。

在可访问性领域，LLM 驱动 GUI 智能体同样发挥了重要作用。VizAbility [484] 面向视障与低视力用户，提升对数据可视化内容的可访问性。它结合结构化图表导航与基于 LLM 的对话交互，使用户可以用自然语言查询图表内容与趋势。通过使用 Olli[34] 框架与 Vega-Lite[35] 规范，VizAbility 允许用户在无需直接“看见”图表的情况下探索视觉数据，解决了真实场景中的可访问性问题。

针对老年用户需求，EasyAsk [491] 作为一个具上下文感知能力的应用内助手，为非技术用户提升可用性。它通过语音与触控等多模态输入结合 GUI 元素，生成精确且符合情境的教程检索结果，帮助用户学习与导航手机功能，展示了 GUI 智能体如何通过上下文感知与交互式教程提升可访问性。

在语音交互方面，GPTVoiceTasker [487] 和 AutoTask [488] 提供了无手操作（hands-free）的 GUI 交互能力。GPTVoiceTasker 利用 Android Accessibility Tree，将语音命令映射为 GUI 操作，并通过图结构本地数据库记录与复用交互过程；AutoTask 则在没有预定义脚本的情况下自动探索移动 GUI，结合语音命令接口与动态动作引擎，执行多步任务，并通过试错与经验驱动学习适应未知任务与环境。

在创意工作流领域，AssistEditor [489] 通过多智能体框架将视频编辑与 GUI 智能体结合。系统在桌面环境中自动执行复杂的视频编辑工作流，使用对话系统与视频理解模型将用户意图映射为专业编辑操作，通过角色分工的多智能体协作高效完成任务，展示了 LLM 驱动 GUI 智能体在创意领域的应用潜力。

WebNav [497] 则作为面向视障用户的 Web 导航辅助技术，使用大型模型（如 Gemini 2.0 Flash Thinking）结合 DOM 标签与 ReAct 式推理循环，支持语音驱动在复杂网站上进行导航，为 Web 无障碍提供了一种新的解决方案。

表 43：基于 GUI 智能体的虚拟助手研究工作概览

|项目（Project）|类型（Type）|平台（Platform）|模型（Model）|感知（Perception）|动作（Action）|场景（Scenario）|亮点（Highlight）|链接（Link）|
|---|---|---|---|---|---|---|---|---|
|ProAgent [485]|Research|Web 与桌面|GPT-4|任务描述与结构化应用数据|标准 UI 操作与动态分支|自动化业务流程，如数据分析、报表生成与通知等|提出动态工作流框架，使智能体能够灵活解释与执行任务，超越传统 RPA 系统|https://github.com/OpenBMB/ProAgent|
|LLMPA [486]|Research|移动（Android）|AntLLM-10B|UI 树结构、视觉建模与文本抽取模块|标准 UI 操作|在移动应用中自动化如订票等用户交互|将 LLM 推理能力与模块化设计结合，支持任务分解、目标检测与稳健动作预测|/|
|VizAbility [484]|Research|桌面|GPT-4V|可键盘导航的树形视图|导航图表结构并生成答案|帮助视障与低视力用户探索与理解数据可视化|将结构化图表导航与 LLM 对话能力结合，使用户可用自然语言查询图表内容|https://dwr.bc.edu/vizability/|
|GPTVoiceTasker [487]|Research|移动（Android）|GPT-4|Android Accessibility Tree|标准 UI 操作|通过语音命令自动化移动设备交互|整合 LLM 自然语言理解与实时 GUI 交互，使用图结构本地数据库记录与复现交互|https://github.com/vuminhduc796/GPTVoiceTasker|
|AutoTask [488]|Research|移动（Android）|GPT-4|Android Accessibility Tree|标准 UI 操作|在移动设备上自动化多步任务|无需预定义脚本或配置，自主探索 GUI 环境并完成复杂任务|https://github.com/BowenBryanWang/AutoTask|
|AssistEditor [489]|Research|Windows|UniVTG [544]|GUI 元素、用户需求与视频数据|标准 UI 操作|自动化视频编辑工作流|通过多智能体协同，将用户需求融入专业视频编辑过程|/|
|PromptRPA [490]|Research|移动（Android）|GPT-4 与 GPT-3.5 Turbo|布局层级与带 OCR 的截图|标准 UI 操作与应用级功能|自动化智能手机任务并生成交互式教程|引入用户反馈闭环以持续改进，解决界面演化与任务多样性问题|/|
|EasyAsk [491]|Research|移动（Android）|GPT-4|Android Accessibility Tree|高亮特定 UI 元素以引导用户操作|通过应用内交互式教程帮助老年用户学习与导航手机功能|结合语音与触控输入，用应用内上下文信息补充模糊或不完整查询|/|
|WebNav [497]|Research|Web|Gemini 2.0 Flash Thinking|GUI 截图与 DOM|标准 UI 操作|为视障用户提供语音驱动的复杂网站导航辅助|结合 ReAct 式推理循环、实时 DOM 标注与语音交互，支持智能 Web 导航|/|

这些研究工作整体展示了 LLM 驱动 GUI 智能体在虚拟助手中的巨大潜力：它们能够在 Web、移动与桌面平台上处理多种复杂任务，从业务流程到可访问性与创意工作流，显著提升虚拟助手的智能性与适应性。

_10.2.2 开源项目（Open-Source Projects）_

除了研究原型之外，一系列开源项目也极大推动了 LLM 驱动 GUI 智能体的普及与易用性，使更多开发者和用户能够定制符合自身需求的虚拟助手。

OpenAdapt [492] 是其中代表性的开源框架，它利用大规模多模态模型，通过观察并复现用户交互来自动化 GUI 任务。系统通过截图与用户输入记录界面状态，结合计算机视觉技术理解并执行标准 UI 操作。OpenAdapt 旨在跨行业简化工作流，通过从用户示范中学习自动化策略，减少手写脚本需求，展示了在 GUI 任务自动化中的高度适应性。

AgentSea [493] 提供了一个面向多平台的模块化工具包，用于构建能够导航并交互多种 GUI 环境的智能体。其灵活性特别适合开发可在不同应用中自动化复杂任务的虚拟助手。AgentSea 遵循 UNIX 哲学，将系统划分为多个专门工具，易于组合与扩展，并通过开源模式鼓励社区协作与创新。

Open Interpreter [494] 则进一步展示了开源生态的潜力，它利用 LLM 在本地执行代码，使用户可以通过自然语言命令控制电脑 GUI。系统支持多种编程语言与平台，可用于数据分析、Web 自动化与系统管理等任务。通过提供对本地资源与库的完全访问，Open Interpreter 规避了云服务的一些限制，为希望通过 AI 强化本地工作流的用户提供了强大能力。

_10.2.3 生产系统（Production）_

在生产环境中集成以 LLM 为大脑的 GUI 智能体，展示了其在商业应用中切实可行且具有实际影响力。

Power Automate [132] 是微软推出的桌面自动化工具，其 AI Recorder 功能体现了 LLM 驱动 GUI 智能体在生产环境中的应用。用户可以用自然语言描述希望自动化的任务，同时系统记录用户在 GUI 中的交互，将这些描述转化为自动化工作流，从而在用户意图与具体执行之间架起桥梁。通过记录与复现用户操作，Power Automate 显著简化了重复任务的自动化过程，提升了效率并降低了自动化门槛。

在 Web 场景中，MultiOn [495] 作为个人 AI 代理，能够在 Web GUI 上自主执行用户定义的任务。它利用大语言模型解析自然语言命令并将其转化为精确的 Web 操作，从而自动化复杂或重复的 Web 交互。MultiOn 对 Web 元素的感知与操作能力，使其可以在多种网站之间无缝工作，提升用户生产力并简化 Web 使用流程。

在移动平台上，MagicOS 中的 YOYO Agent [496] 展示了 LLM 驱动 GUI 智能体在手机上的实践。基于荣耀的 MagicLM，YOYO Agent 能够理解并执行跨应用的用户命令，并通过学习用户行为提供个性化帮助。该系统展示了大型语言模型如何增强虚拟助手，使其能够在移动 GUI 中执行复杂任务并改善用户体验与生产力。

Eko [545] 则是一个面向多平台 GUI 交互的智能代理开发工具，集成多种 LLM，并通过视觉-交互元素感知（Visual-Interactive Element Perception, VIEP）技术来识别与操作界面元素。Eko 支持将自然语言任务描述分解为可执行工作流，为需要构建可定制、可生产部署的智能体工作流的开发者提供了完整工具链。

表 44：基于 GUI 智能体的虚拟助手开源与生产系统概览

|项目（Project）|类型（Type）|平台（Platform）|模型（Model）|感知（Perception）|动作（Action）|场景（Scenario）|亮点（Highlight）|链接（Link）|
|---|---|---|---|---|---|---|---|---|
|OpenAdapt [492]|Open-source|桌面|LLM, VLM（如 GPT-4, ACT-1）|带 GUI 解析 CV 模块的截图|标准 UI 操作|跨行业自动化重复性任务|通过观察用户交互学习任务自动化策略，消除手写脚本需求|https://github.com/OpenAdaptAI/OpenAdapt|
|AgentSea [493]|Open-source|桌面与 Web|LLM, VLM|带 GUI 解析 CV 模块的截图|标准 UI 操作|在 GUI 环境中自动化任务|提供遵循 UNIX 哲学的模块化工具包，便于开发者为多种 GUI 环境构建自定义智能体|https://www.agentsea.ai/|
|Open Interpreter [494]|Open-source|桌面、Web、移动（Android）|LLM|通过命令行感知系统状态|Shell 命令、代码与本地 API|自动化任务、数据分析、文件管理与 Web 控制|在本地执行代码，完全访问系统资源与库，克服云端服务限制|https://github.com/OpenInterpreter/open-interpreter|
|MultiOn [495]|Production|Web|LLM|/|标准 UI 操作|自动化基于 Web 的任务|通过自然语言命令执行自主 Web 操作|https://www.multion.ai/|
|YOYO Agent in MagicOS [496]|Production|移动（MagicOS 9.0）|MagicLM|GUI 上下文|执行应用内与跨应用操作|自动化日常任务、提升生产力|利用 MagicLM 理解并执行跨应用复杂任务，同时学习用户习惯提供个性化辅助|/|
|Power Automate [132]|Production|Windows|LLM, VLM|记录用户与 GUI 的交互|标准 UI 操作|自动化重复性任务并简化工作流|将自然语言自动化需求转化为可执行工作流，并通过录制用户操作生成脚本|https://learn.microsoft.com/en-us/power-automate/desktop-flows/create%2Dflow-using%2Dai-recorder|
|Eko [545]|Production|Web 浏览器与桌面环境|ChatGPT 与 Claude 3.5|基于 VIEP 的视觉交互元素感知|标准 UI 操作|通过处理多样工作流自动化任务|将自然语言任务描述分解为可执行工作流，实现自然语言与编程逻辑的无缝集成|https://eko.fellou.ai/|

总体而言，这些开源与生产系统展示了以 LLM 为大脑的 GUI 智能体在现实应用中的可行性与价值：从本地自动化工具与 Web 个人助理，到手机系统级助手与企业级工作流平台，它们正在推动虚拟助手向更加智能、自适应与高可用的方向发展。

**10.3 小结（Takeaways）**

以 LLM 为大脑的 GUI 智能体在 GUI 测试与虚拟助手等应用中引入了新的能力与交互方式，包括自然语言驱动、增强自动化与跨平台可访问性等。这些智能体通过简化复杂任务与降低技术门槛，正在改变用户与软件的交互方式。但这一领域仍处于早期阶段，要实现真正成熟的大规模应用仍面临诸多挑战。当前应用层面的关键洞见包括：

1) 自然语言驱动的交互（Natural Language-Driven Interactions）：LLM 驱动的 GUI 智能体使用户可以通过自然语言与应用交互，显著降低了非专业用户的使用门槛。在 GUI 测试中，GPTDroid [125] 与 AUITestAgent [465] 等工具允许测试人员以自然语言描述测试用例与需求，由智能体自动完成执行与验证；在虚拟助手中，LLMPA [486] 与 ProAgent [485] 通过解释用户指令执行复杂任务，体现了自然语言接口在跨平台简化交互方面的潜力。
2) 复杂任务的高级自动化（Enhanced Automation of Complex Tasks）：这些智能体已经展示出在无需人工脚本的情况下自动化多步、复杂工作流的能力。AutoTask [488] 与 GPTVoiceTasker [487] 等系统可以自主探索 GUI 环境，并基于高层目标或语音命令执行任务。在 GUI 测试中，CrashTranslator [470] 与 AdbGPT [471] 等工作通过自动生成测试输入与从文本描述中复现缺陷显著提升了覆盖率与效率。
3) 多模态感知与交互（Multimodal Perception and Interaction）：将视觉与文本输入结合，使智能体对 GUI 语境的理解更加深入，从而做出更准确的决策与交互。VizAbility [484] 与 OpenAdapt [492] 等系统利用截图、UI 树与 OCR 综合感知环境，对动态或视觉复杂界面尤为关键。
4) 改善可访问性与用户体验（Improved Accessibility and User Experience）：LLM 驱动的 GUI 智能体帮助提高了技术对不同用户群体的可达性。VizAbility [484] 为视障用户理解数据可视化提供支持，EasyAsk [491] 协助老年用户学习与导航手机功能。通过针对不同用户群体定制交互方式，这些智能体提升了包容性与整体用户体验。

综上，LLM 驱动的 GUI 智能体正在通过自然语言理解、复杂任务自动化与可访问性提升，改变 GUI 交互与自动化的格局。虽然仍处于发展的早期阶段，但持续的研究与实践表明，这类智能体在多域多平台上的潜力巨大。未来需要持续探索，以应对当前存在的挑战，并在隐私、安全、可扩展性与人机协作等方面不断完善，从而充分释放以 LLM 为大脑的 GUI 智能体的价值。
###### 11 局限、挑战与未来路线图（LIMITATIONS, CHALLENGES AND FUTURE ROADMAP）

尽管以 LLM 为大脑的 GUI 智能体在近几年取得了长足进展，但整体而言，这一领域仍处于早期阶段。在大规模真实应用落地前，仍存在诸多技术与非技术挑战，需要系统识别与应对。本节围绕若干关键问题展开，概述当前局限，并提出未来研究方向与潜在解决路径。

**11.1 隐私问题（Privacy Concerns）**

在 LLM 驱动 GUI 智能体的场景中，隐私问题尤为突出。为了有效感知与操作 GUI，智能体往往需要访问高度敏感的数据，例如截图、交互历史、个人凭证与机密文档等。在依赖云端 LLM 的情况下，这些数据常常需要上传到远程服务器进行推理 [546]–[548]，带来数据泄露、未授权访问与个人信息滥用等风险。当敏感输入通过第三方 API 传输或在设备外处理时，合规与安全问题更加凸显。

例如，一个负责管理用户邮箱的 GUI 智能体可能需要读取、分类并回复包含高度隐私或机密内容的邮件。若将处理过程完全交给云端，势必增加隐私暴露风险 [432], [549], [550]。与传统 LLM 应用相比，GUI 智能体通常在更细粒度的用户活动上工作，且需要更广泛的系统访问权限，因此如何在保证能力的同时保护隐私，成为具有领域特异性的关键挑战。

**潜在解决方案（Potential Solutions）**：

未来研究应重点推动端侧推理（on-device inference）[551], [552]，即让语言模型在本地设备上直接运行，而无需上传个人数据。这需要在模型压缩 [553]、端侧优化 [554] 与高效推理算法 [555] 等方面取得进展，以适应终端算力限制。同时，框架层面应引入数据脱敏、加密通信与严格的数据使用范围控制，并与操作系统级隐私控制（如运行时权限弹窗、沙箱执行）结合。

从技术角度，可采用联邦学习（federated learning）[556]、差分隐私 [557] 与同态加密 [558] 等隐私保护手段，在不直接上传原始数据的前提下进行模型训练或更新。此外，GUI 智能体开发者需要与隐私政策制定方协同工作 [559]，建立透明的数据处理流程，明确告知用户数据采集与使用方式，并获得明确同意 [560]。

**11.2 时延、性能与资源约束（Latency, Performance, and Resource Constraints）**

与一般 LLM 应用不同，GUI 智能体在多步交互环境中运行，对实时性要求更高。每一步动作决策都依赖 LLM 推理，若计算开销过大，会导致显著时延与响应迟缓 [561]，直接影响用户体验。对于需要长时间或多轮交互的任务，时延会累积放大，在移动端等资源受限设备上尤其明显：运行 GUI 智能体可能导致应用卡顿或耗电剧增 [562]–[564]。由于 GUI 智能体需要在动态环境中持续感知、决策与控制 UI，这些问题在本领域尤为尖锐 [563]。

**潜在解决方案**：

未来工作应通过模型结构优化与工程手段降低推理时延 [565]。例如，可通过模型蒸馏（distillation）构建更小、更快的模型，同时尽量保持性能 [566]；利用 GPU、TPU 或专用 AI 芯片，并探索并行推理策略，以提高计算效率 [567]。增量推理与缓存（caching）机制可以重复利用部分计算结果 [568]。在模型压缩方面，如剪枝（pruning）[569] 与量化（quantization）[553] 等方法有助于在端侧部署轻量化模型；边缘计算与分布式推理 [552], [570] 则可以在端/云间合理分摊计算负载。

此外，从系统设计角度，GUI 智能体应鼓励应用开发者暴露更高层的原生 API [198], [199]，将多步 UI 操作封装为单个 API 调用。通过将这类 API 集成入智能体动作空间，可以用更少步骤完成任务，从而降低累计时延。

**11.3 安全与可靠性（Safety and Reliability）**

GUI 智能体可以直接在真实系统中执行操作，例如点击按钮、删除文件、提交表单或触发系统级命令，因此在安全与可靠性方面存在比普通 LLM 更严峻的风险 [548], [571]。一旦动作预测错误，后果可能不可逆，如数据损坏、误发消息、应用崩溃或越权访问 [572], [573]。同时，LLM 输出具有不确定性与非确定性，智能体可能产生幻觉动作、误解 UI 上下文，或在不同会话中表现不一致 [574]–[578]。例如，在自动化金融交易场景中，错误的转账操作可能造成实际损失。

GUI 智能体还面临更广泛的攻击面：黑盒对抗攻击可能通过操控输入或利用决策策略漏洞，诱导智能体执行有害操作 [579]。在动态软件生态中，不正确的操作可能跨应用传播，引发系统级连锁问题。兼容性、安全边界与 UI 框架演化等因素，使得安全可靠的集成更加复杂 [580], [581]。

**潜在解决方案**：

保障安全与可靠性需要引入系统性的错误检测与处理机制 [583]。未来研究应在动作执行前增加验证步骤，对推断出的动作进行合理性检查 [584]；采用形式化验证方法 [585]、异常处理流程 [586] 与回滚机制 [587]，降低错误影响范围。同时，细粒度的权限管理 [588]–[591] 能限制智能体在系统中的操作边界，避免高危或越权操作。

另外，可探索标准化的交互协议 [592]，以便智能体在不同应用中以统一、安全的方式交互；并遵循安全最佳实践，引入安全认证与授权协议 [593]，从系统层面增强安全性。

**11.4 人机交互（Human-Agent Interaction）**

在人机共享 GUI 的场景下，人机交互呈现出特有挑战。用户和智能体在同一界面中操作，任何用户干预（如移动鼠标、调整窗口、修改输入等）都可能打断智能体现有计划，甚至导致错误操作或任务失败 [594], [595]。如何设计稳健的协作协议，决定何时由智能体让出控制权、暂停执行或征询用户意见，是 GUI 自动化中特有的问题。

自然语言指令本身也可能含糊或依赖上下文，导致任务规划不完整或理解偏差。运行过程中，智能体还会遇到意外弹窗、缺失输入或冲突 UI 状态等不确定情况，需要主动向用户寻求澄清 [19], [596]。何时、以何种方式向用户提出澄清、授权或确认请求，是确保可靠性与建立信任的关键 [67], [597], [598]。

图 31 中的虚构示例展示了一个看似简单但实际复杂的人机协作任务：智能体应用户请求发送一封祝贺邮件给 “Tom”。在执行过程中，智能体一方面需要提示用户手动登录 Gmail 以保护凭证，另一方面在发现多个名为 Tom 的联系人时需要询问用户进行 disambiguation，并在发送前再次确认邮件内容。这个例子体现出隐私保护、歧义消解与意图确认等多重要求 [599]，说明 GUI 智能体需要专门的人机协同设计，而不仅是通用 LLM 的简单延伸。

**潜在解决方案**：

遵循以用户为中心的设计原则 [600]，为用户提供更多可控性与自定义选项 [596]，是提升人机协作体验的关键。智能体应具备进行澄清对话的能力，当检测到指令存在歧义时主动发问 [601]；自然语言理解模块应能识别模糊之处并请求更多信息。例如，当有多个“John”时，智能体应主动询问“你指的是哪一位？”。

引入人类在环（human-in-the-loop）机制 [602]，允许用户在任务执行中随时介入、纠正智能体决策，也有助于提升安全性与可控性。同时，提供透明与可解释的推理过程 [67], [603], [604]，有助于建立用户信任。另一个重要方向是为智能体提供虚拟桌面环境，如 UFO[2] 中的画中画（PiP）模式 [334]（图 32），让智能体在独立窗口中操作而不打乱用户主桌面，从而提升整体用户体验与可用性。

**11.5 定制化与个性化（Customization and Personalization）**

高效的 GUI 智能体不仅要完成“通用任务”，还应能根据不同用户的工作流、偏好与行为模式进行个性化适配 [45], [605]。与单轮对话式 LLM 应用不同，GUI 智能体跨多软件环境长期运行，用户风格差异可能非常大。如果所有用户共享同一套策略，往往无法对齐具体用户在文档编辑、界面导航或任务组织方面的偏好，甚至会引发摩擦与挫败感 [606]。

例如，一个辅助文档编辑的智能体需要理解用户偏好的语气、格式与词汇。如果缺乏这种上下文，智能体可能给出不合适的改写建议或强制施加用户不需要的格式。实现个性化需要长期学习（longitudinal learning）：智能体应能持续从历史交互中学习，逐步调整行为以匹配用户预期，并在不同会话之间保持一致性 [607]。

然而，个性化同样带来新挑战。用户偏好具有高度异质性，难以定义统一策略；同时，收集与利用用户数据必须遵守隐私与合规要求 [613]，特别是在智能体经常接触敏感文档、个人应用或系统界面时。

**潜在解决方案**：

未来研究需要探索用户建模（user modeling）[608] 与偏好学习（preference learning）[609] 机制，使智能体能根据个体用户调整行为。可采用基于用户反馈的强化学习 [610]、协同过滤 [611] 与上下文感知计算 [612] 等方法，从持续交互中学习偏好。同时，应尽量通过端侧学习与匿名化数据处理完成个性化，降低对隐私的影响 [613]。从更长远与“赛博朋克”的视角来看，智能体甚至可以反向“生成适配用户的 GUI”，实现界面与智能体的联合个性化 [614]。

**11.6 伦理与监管挑战（Ethical and Regulatory Challenges）**

由于 GUI 智能体能够在软件界面中执行真实操作，其在伦理与监管层面引发了与传统 LLM 不同的问题 [548], [615]–[618]。一方面，训练数据中潜在偏见可能在敏感工作流中被放大。例如，用于辅助招聘的 GUI 智能体可能在无意中展现性别或种族偏见 [619], [620]，而 GUI 层的操作往往缺乏可追踪性，难以审计其决策链路。

另一方面，合规性问题复杂化：智能体常跨多个受严格数据保护法规约束的领域工作，却缺乏标准化的日志与用户同意机制，很难满足法律与伦理要求，尤其是在智能体以“黑箱”方式在后台运行时。为此，需要针对 GUI 智能体设计专门的权限控制、运行时确认与活动日志，以确保在多环境部署时的安全、公平与合规。

**潜在解决方案**：

需要制定清晰的伦理准则与监管框架 [621]，指导 GUI 智能体的研发与使用。未来研究应关注行为审计与监控机制 [622]，确保其符合伦理与法律要求 [623]。在模型层面，可引入偏见检测与缓解技术 [624]，降低歧视性或不公平行为发生的可能。为用户提供对数据使用的控制权，并清晰说明智能体能力与限制，也有助于提升透明度与信任 [624]。

**11.7 可扩展性与泛化（Scalability and Generalization）**

当前 GUI 智能体往往难以从特定应用或环境扩展到更广泛场景，泛化能力有限。不同软件界面在布局、风格与交互模式上异质性很高，即便是常见 UI 元素（如弹窗）也有大量变体 [625]，这使得无需重训或微调即可在多平台稳健运行的智能体难以设计。

此外，真实 GUI 的动态性也构成挑战：软件更新、A/B 测试或界面重设计都会改变布局与控件层级，轻易打破原本可用的策略。例如，在特定版本文字处理软件上训练的智能体，在界面布局调整或迁移到不同软件时可能完全失效 [626]。这类问题导致在实际部署中维护成本高昂，需要频繁更新或重训 [627]–[629]。

**潜在解决方案**：

从数据角度看，可构建覆盖多环境、多平台与多交互模式的综合 GUI 智能体数据集，使 LLM 在训练阶段就接触足够多样的界面与任务 [630]，从而学习共通模式并提高泛化能力。

在方法层面，可以引入迁移学习 [631] 与元学习 [632]。迁移学习通过在大规模多样数据上预训练，再在小规模任务数据上微调，从而提升在新应用上的适应力；元学习则通过提炼跨任务的抽象模式，使模型在遇到新任务时能利用少量数据快速适应。

即便如此，智能体在全新环境中仍可能遇到困难。此时，应用开发者可以为智能体提供配套知识库，如使用文档、FAQ 与人类示范等 [633]–[635]，并结合检索增强生成（RAG）[189], [636]，在推理时查询这些知识以辅助决策。例如，当遇到未知控件时，智能体可查询文档以理解其功能与交互方式，在不重训模型的前提下提升能力。实现上述方案需要智能体开发者与应用提供方的协作。

**11.8 小结（Summary）**

总体而言，以 LLM 为大脑的 GUI 智能体在提升生产力与自动化复杂任务方面具有巨大潜力，但要充分释放这一潜力，需要系统地解决上述隐私、性能、安全、人机交互、个性化、伦理与泛化等问题。研究者、工业界、政策制定者与用户之间的协作至关重要：跨学科团队有助于推动创新，并确保在考虑技术、伦理与社会影响的前提下，负责任地发展 GUI 智能体。

随着领域不断演进，持续的评估与迭代将是关键，只有在与用户需求与期望保持对齐的前提下，GUI 智能体才能真正发展为更智能、更安全、更易用的下一代人机交互主体。

###### 12 结论（CONCLUSION）

将大语言模型与 GUI 自动化结合标志着人机交互的一次变革性时刻。LLM 为自然语言处理、理解与 GUI 语义建模提供了“头脑（brain）”，而 GUI 自动化工具则承担“手（hands）”的角色，将智能体的认知能力转化为软件环境中的具体操作指令。二者结合形成了以 LLM 为大脑的 GUI 智能体，使用户可以通过简单的自然语言命令替代复杂且平台相关的 UI 操作，开启了全新的交互范式。在学术界与工业界，这一组合已经展现出巨大的应用潜力。

本综述系统、全面且及时地回顾了以 LLM 为大脑的 GUI 智能体领域的发展现状。我们介绍了构成这类智能体的核心组件与高级技术，梳理了数据采集、模型构建、系统框架、评估方法与真实应用等关键方面。同时，我们分析了当前存在的局限与挑战，并提出了未来可能的研究路线图。我们希望本文既能为新入门读者提供一份实用的“入门手册”，也能作为研究者了解该领域最新进展的参考。

展望未来，“以 LLM 为大脑的 GUI 智能体”这一概念将愈发具体化，有望在日常生活与工作中显著提升生产力与可访问性。随着持续的研究与工程实践，这项技术有望从根本上重塑人类与数字系统的交互方式，将复杂工作流转化为流畅自然的对话式交互。

###### 参考文献（REFERENCES）

（参考文献部分为文献条目列表，通常保持原文格式与语言，以便读者根据原始引用进行检索与查阅。）
