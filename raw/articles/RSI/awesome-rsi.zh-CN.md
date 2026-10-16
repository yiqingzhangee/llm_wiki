# 精选 RSI（递归自我改进） / Awesome RSI (Recursive Self-Improvement) [![Awesome](https://awesome.re/badge.svg)](https://awesome.re)

> **说明：** 本文件基于 `lobehub/awesome-rsi` 默认分支整理；中文在前，英文在后；论文摘要取官方摘要并译为中文；无正式摘要的项目、博客和视频使用官方简介或原列表简介，并逐项标注来源。
>
> **Note:** This file is compiled from the default branch of `lobehub/awesome-rsi`. Chinese precedes English. Paper abstracts use official abstracts with complete Chinese translations. Projects, blog posts, and videos without formal abstracts use official descriptions or descriptions from the original list, with the source marked for each entry.

## 导语 / Introduction

### 中文

递归自我改进（Recursive Self-Improvement，RSI）是指人工智能系统提升自身能力，同时还能改进产生后续提升的机制。

随着自训练、智能体记忆、Harness 优化、具身自我改进、自动化 AI 研究、自修改编程智能体和进化搜索取得进展，RSI 正从一种理论设想逐渐成为实证研究方向。

Awesome RSI 收集并整理这些方向的重要工作，涉及模型层自我改进、上下文与记忆演化、Harness 与脚手架演化、具身及物理自我改进、多智能体系统、自动化 AI 研究、基准测试和安全。

本列表收录的工作不都符合严格意义上的 RSI。有些工作体现的是范围有限的自我改进，另一些则是可能促成完整递归系统的使能技术。

初次接触这一主题，建议先读“基础与入门”；已经了解基本概念，则可直接查看自己感兴趣的章节。

这是一份由社区维护并持续更新的列表。欢迎补充遗漏论文、新基准和框架，也欢迎提出分类法的改进建议。

这里的分类只用于组织材料，各类别并非互斥；许多系统横跨多个层次和机制。

### English

Recursive Self-Improvement (RSI) refers to processes in which AI systems improve their own capabilities and can also improve the mechanisms that generate subsequent improvements.

Recent progress in self-training, agent memory, harness optimization, embodied self-improvement, automated AI research, self-modifying coding agents, and evolutionary search has made RSI increasingly relevant as an empirical research direction rather than only a theoretical idea.

Awesome RSI collects and organizes important work across these areas, including model-level self-improvement, context and memory evolution, harness and scaffold evolution, embodied and physical self-improvement, multi-agent systems, automated AI research, benchmarks, and safety.

Not every work listed here demonstrates RSI in the strict sense. Some represent bounded self-improvement or enabling techniques that may contribute to more complete recursive systems.

If you are new to the topic, start with Fundamentals & Getting Started. If you already know the basics, explore the sections most relevant to your interests.

This is a community-maintained and evolving list. Contributions, missing papers, new benchmarks, frameworks, and suggestions for improving the taxonomy are very welcome.

The taxonomy is organizational rather than mutually exclusive; many systems span multiple layers and mechanisms.

## 目录 / Contents

### 中文

- [范围与术语](#scope--terminology)
- [基础与入门](#fundamentals--getting-started)
- [模型层 RSI](#model-level-rsi)
  - [自训练与自奖励](#self-training--self-reward)
  - [合成数据与自蒸馏](#synthetic-data--self-distillation)
  - [自博弈与迭代微调](#self-play--iterative-fine-tuning)
  - [自学式推理](#self-taught-reasoning)
- [Harness 层 RSI](#harness-level-rsi)
  - [提示词与程序优化](#prompt--program-optimization)
  - [上下文与记忆演化](#context--memory-evolution)
  - [Harness 与脚手架演化](#harness--scaffold-evolution)
  - [可扩展 Harness 基底](#extensible-harness-substrates)
  - [自验证与自纠错：使能基础](#self-verification--self-correction--enabling-foundations)
  - [自演化智能体框架](#self-evolving-agent-frameworks)
- [多智能体自我改进](#multi-agent-self-improvement)
  - [协同演化](#co-evolution)
  - [推理时辩论](#inference-time-debate)
- [编程与软件工程自我改进](#coding--software-engineering-self-improvement)
  - [自修改编程智能体](#self-modifying-coding-agents)
  - [迭代修复与训练](#iterative-repair--training)
- [自动化 AI 研发](#automated-ai-rd)
- [具身与物理自我改进](#embodied--physical-self-improvement)
- [进化式与开放式 RSI](#evolutionary--open-ended-rsi)
- [安全、对齐与理论](#safety-alignment--theory)
  - [安全使能基础](#supporting-safety-foundations)
- [内省与自我建模](#introspection--self-modeling)
- [基准与评测](#benchmarks--evaluations)
  - [直接 RSI 与自我改进评测](#direct-rsi--self-improvement-evaluations)
  - [前沿实验室自我改进与 AI 研发评测框架](#frontier-lab-self-improvement--ai-rd-evaluation-frameworks)
  - [AI 研发能力代理指标](#ai-rd-capability-proxies)
  - [智能体能力代理指标](#agent-capability-proxies)
- [框架与工具](#frameworks--tools)
  - [自修改与自演化系统](#self-modifying--self-evolving-systems)
  - [Harness、记忆与技能演化](#harness--memory--skill-evolution)
  - [自动化搜索与 AI 研发](#automated-search--ai-rd)
- [博客文章与讨论](#blog-posts--discussions)
- [演讲与视频](#talks--videos)
- [相关 Awesome 列表](#related-awesome-lists)

### English

- [Scope & Terminology](#scope--terminology)
- [Fundamentals & Getting Started](#fundamentals--getting-started)
- [Model-level RSI](#model-level-rsi)
  - [Self-Training & Self-Reward](#self-training--self-reward)
  - [Synthetic Data & Self-Distillation](#synthetic-data--self-distillation)
  - [Self-Play & Iterative Fine-tuning](#self-play--iterative-fine-tuning)
  - [Self-Taught Reasoning](#self-taught-reasoning)
- [Harness-level RSI](#harness-level-rsi)
  - [Prompt & Program Optimization](#prompt--program-optimization)
  - [Context & Memory Evolution](#context--memory-evolution)
  - [Harness & Scaffold Evolution](#harness--scaffold-evolution)
  - [Extensible Harness Substrates](#extensible-harness-substrates)
  - [Self-Verification & Self-Correction — Enabling Foundations](#self-verification--self-correction--enabling-foundations)
  - [Self-Evolving Agent Frameworks](#self-evolving-agent-frameworks)
- [Multi-Agent Self-Improvement](#multi-agent-self-improvement)
  - [Co-Evolution](#co-evolution)
  - [Inference-time Debate](#inference-time-debate)
- [Coding / Software-Engineering Self-Improvement](#coding--software-engineering-self-improvement)
  - [Self-Modifying Coding Agents](#self-modifying-coding-agents)
  - [Iterative Repair & Training](#iterative-repair--training)
- [Automated AI R&D](#automated-ai-rd)
- [Embodied & Physical Self-Improvement](#embodied--physical-self-improvement)
- [Evolutionary & Open-Ended RSI](#evolutionary--open-ended-rsi)
- [Safety, Alignment & Theory](#safety-alignment--theory)
  - [Supporting Safety Foundations](#supporting-safety-foundations)
- [Introspection & Self-Modeling](#introspection--self-modeling)
- [Benchmarks & Evaluations](#benchmarks--evaluations)
  - [Direct RSI & Self-Improvement Evaluations](#direct-rsi--self-improvement-evaluations)
  - [Frontier Lab Self-Improvement & AI R&D Evaluation Frameworks](#frontier-lab-self-improvement--ai-rd-evaluation-frameworks)
  - [AI R&D Capability Proxies](#ai-rd-capability-proxies)
  - [Agent Capability Proxies](#agent-capability-proxies)
- [Frameworks & Tools](#frameworks--tools)
  - [Self-Modifying / Self-Evolving Systems](#self-modifying--self-evolving-systems)
  - [Harness / Memory / Skill Evolution](#harness--memory--skill-evolution)
  - [Automated Search / AI R&D](#automated-search--ai-rd)
- [Blog Posts & Discussions](#blog-posts--discussions)
- [Talks & Videos](#talks--videos)
- [Related Awesome Lists](#related-awesome-lists)

## 范围与术语 / Scope & Terminology

本列表采用以下操作性区分：

For this list, we use the following operational distinctions:

**自我优化（Self-refinement）**：改进当前输出，但不会给系统留下持久变化。

**Self-refinement** — improves the current output without a persistent change to the system.

**持续性自我改进（Persistent self-improvement）**：改变权重、记忆、技能、提示词、Harness 或代码，并将变化带入下一轮。

**Persistent self-improvement** — changes to weights, memory, skills, prompts, harness, or code that carry into the next round.

**递归自我改进（Recursive self-improvement）**：负责产生改进的机制本身也成为改进对象。

**Recursive self-improvement** — the mechanism that produces improvements is itself the object of improvement.

**RSI 基底（RSI substrate）**：把智能体自身的结构暴露为可修改对象，但默认并不一定构成自动化的自我改进闭环。

**RSI substrate** — exposes an agent's own structure as a modifiable object, but does not necessarily form an automatic self-improvement loop by default.

## 基础与入门 / Fundamentals & Getting Started

本节收录奠定递归自我改进术语和核心问题的基础论文、形式化研究与综述。

Foundational papers, formal treatments, and surveys that establish the vocabulary and core questions of RSI.

#### 1. 自演化智能体综述：迈向人工超级智能之路上的进化对象、时机、方式与场域

**完整中文摘要：**

大语言模型（LLM）已在多种任务上展现出卓越能力，但从根本上说仍是静态的：面对新任务、持续变化的知识领域或动态交互情境时，它们无法调整自身内部参数。随着 LLM 越来越多地部署到开放式交互环境中，这种静态属性已成为严重瓶颈，因而需要能够实时进行适应性推理、行动和演化的智能体。从扩展静态模型转向开发自演化智能体，已引发人们对相关架构与方法的广泛关注；这些架构和方法使系统能够持续从数据、交互与经验中学习并适应。本综述首次对自演化智能体进行系统且全面的回顾，并围绕三个基础维度组织这一领域：进化什么、何时进化以及如何进化。我们考察智能体不同组件（如模型、记忆、工具和架构）的演化机制，按阶段（如测试时内部阶段与测试间阶段）划分适应方法，并分析引导演化适应的算法和架构设计（如标量奖励、文本反馈、单智能体系统与多智能体系统）。我们还分析了面向自演化智能体的评估指标与基准，介绍其在编程、教育和医疗等领域的应用，并指出安全性、可扩展性和协同演化动力学方面的重要挑战与研究方向。通过为理解和设计自演化智能体提供结构化框架，本综述给出了推进更具适应性、鲁棒性和通用性的智能体系统的路线图，既面向研究，也面向现实部署；最终，它还帮助我们理解如何实现人工超级智能（ASI），即智能体能够自主演化，并在各类任务上达到超越人类的智能水平。

**English title:** A Survey of Self-Evolving Agents: What, When, How, and Where to Evolve on the Path to Artificial Super Intelligence

**Abstract:**

> Large Language Models (LLMs) have demonstrated remarkable capabilities across diverse tasks but remain fundamentally static, unable to adapt their internal parameters to novel tasks, evolving knowledge domains, or dynamic interaction contexts. As LLMs are increasingly deployed in open-ended, interactive environments, this static nature has become a critical bottleneck, necessitating agents that can adaptively reason, act, and evolve in real time. This paradigm shift -- from scaling static models to developing self-evolving agents -- has sparked growing interest in architectures and methods enabling continual learning and adaptation from data, interactions, and experiences. This survey provides the first systematic and comprehensive review of self-evolving agents, organizing the field around three foundational dimensions: what, when, and how to evolve. We examine evolutionary mechanisms across agent components (e.g., models, memory, tools, architecture), categorize adaptation methods by stages (e.g., intra-test-time, inter-test-time), and analyze the algorithmic and architectural designs that guide evolutionary adaptation (e.g., scalar rewards, textual feedback, single-agent and multi-agent systems). Additionally, we analyze evaluation metrics and benchmarks tailored for self-evolving agents, highlight applications in domains such as coding, education, and healthcare, and identify critical challenges and research directions in safety, scalability, and co-evolutionary dynamics. By providing a structured framework for understanding and designing self-evolving agents, this survey establishes a roadmap for advancing more adaptive, robust, and versatile agentic systems in both research and real-world deployments, and ultimately sheds light on the realization of Artificial Super Intelligence (ASI) where agents evolve autonomously and perform beyond human-level intelligence across tasks.

- **链接：** https://arxiv.org/abs/2507.21046v4
- **年份：** 2026（初次提交于 2025 年；当前版本及正式发表均为 2026 年）
- **发表信息：** *Transactions on Machine Learning Research*（TMLR），2026 年 1 月；arXiv:2507.21046v4 [cs.AI]。

#### 2. AI 中的递归自我改进：从有界自我精炼到自主研究闭环

**完整中文摘要：**

AI 系统正在越来越多地参与自身改进：修订自己的输出，在部署期间调整自身运行框架，使用自己生成的数据进行训练，乃至亲自开展 AI 研究。相关文献使用“自我精炼”“自我奖励”“自我博弈”“自我演化”等术语，却把目标根本不同的工作混在了一起。我们沿两个维度梳理了 2024 至 2026 年间的 1,250 篇 arXiv 论文：一是系统改进的对象，即部署时行为、通过训练形成的策略、评估器或研究过程本身；二是闭环程度，即从有人参与到完全闭环。该分类将有界自我精炼与开放式递归自我改进（RSI）区分开来。前者具有收敛性、可评估性，并已进入工业实践；后者在现有证据能够测量的每个方面，仍受制于基础锚定要求、崩塌动力学和算力约束。本综述的一个特点是把自我评估单列为专门类别：任何改进闭环都隐含了一项主张，即某种信号可以代替人类判断。因此，我们在梳理其所监督闭环的同时，也考察了评估器的设计空间，包括裁判模型、过程奖励模型、验证器、评分规程和元评估；并将各种信号按验证层级排列，从最强的形式化验证器到最弱的内在自我评估。我们观察到，已展示出的自我改进强度与这一层级相对应，其典型失效模式（自我确认闭环、模型崩塌与多样性崩塌）源于对该层级原则的违背。使人类继续留在闭环中的“研究方向设定”瓶颈可以分成两部分：验证问题可由上述层级衡量；更前置的“什么值得评估”问题则不在其覆盖范围内。我们把技术文献与 RSI 极限理论，以及前沿实验室关于闭合改进回路所提出的安全和治理问题联系起来，并指出，达到治理要求的自我改进测量，是目前研究最为稀缺的方向。

**English title:** Recursive Self-Improvement in AI: From Bounded Self-Refinement to Autonomous Research Loops

**Abstract:**

> AI systems increasingly participate in their own improvement: revising their outputs, adapting their harnesses during deployment, training on data they generate, and conducting AI research itself. This literature uses a vocabulary ("self-refine," "self-reward," "self-play," "self-evolve") that conflates fundamentally different ambitions. We survey 1,250 arXiv papers (2024-2026) along two axes: what the system improves -- its behavior in deployment, its policy through training, its evaluator, or the research process itself -- and the degree of loop closure (human-in-the-loop to fully closed). The taxonomy separates bounded self-refinement -- convergent, evaluable, and industrial practice -- from open-ended recursive self-improvement (RSI), which remains bounded by grounding requirements, collapse dynamics, and compute constraints on every measured axis. Its distinctive feature is a dedicated category for self-evaluation: every improvement loop is a claim that some signal can substitute for human judgment. We survey the evaluator design space -- judges, process reward models, verifiers, rubrics, meta-evaluation -- order the signals into a verification hierarchy from formal verifiers (strongest) to intrinsic self-assessment (weakest), and observe that demonstrated self-improvement strength tracks this hierarchy, that its failure modes (self-confirming loops, model and diversity collapse) follow from its violations, and that the "research direction-setting" bottleneck keeping humans in the loop divides into a verification problem the hierarchy indexes and a prior one -- choosing what deserves evaluation at all -- that it does not. We connect the literature to the theory of RSI limits and to the safety and governance questions raised by frontier-lab accounts of closing the loop, and identify governance-grade measurement of self-improvement as the field's most underpopulated niche.

- **链接：** https://arxiv.org/abs/2607.07663v2
- **年份：** 2026
- **发表信息：** arXiv 预印本，arXiv:2607.07663v2 [cs.AI]，44 页、6 幅图。

#### 3. 自演化具身智能

**完整中文摘要：**

具身人工智能（AI）是智能体与环境通过主动感知、具身认知和动作交互共同形成的智能系统。现有具身 AI 仍局限于人工设定：智能体依据给定的记忆接受训练，并针对给定任务构建模型，使固定形态的具身系统与相对静态的环境交互。对于具身形态可变、开放环境动态变化的野外场景，这类方法会失效。本文提出自演化具身 AI：一种新的范式，其中智能体依据自身不断变化的状态和环境运行，通过记忆自更新、任务自切换、环境自预测、具身自适应和模型自演化，实现具有自主演化能力的持续适应性智能。具体而言，我们给出自演化具身 AI 的定义、框架、组成部分和机制，系统回顾各个已实现组件的最新工作，讨论实际应用，并指出未来研究方向。我们认为，自演化具身 AI 能使智能体以类似人类的方式自主学习并与环境交互，也为通用人工智能提供了新的研究视角。

**English title:** Self-evolving Embodied AI

**Abstract:**

> Embodied Artificial Intelligence (AI) is an intelligent system formed by agents and their environment through active perception, embodied cognition, and action interaction. Existing embodied AI remains confined to human-crafted setting, in which agents are trained on given memory and construct models for given tasks, enabling fixed embodiments to interact with relatively static environments. Such methods fail in in-the-wild setting characterized by variable embodiments and dynamic open environments. This paper introduces self-evolving embodied AI, a new paradigm in which agents operate based on their changing state and environment with memory self-updating, task self-switching, environment self-prediction, embodiment self-adaptation, and model self-evolution, aiming to achieve continually adaptive intelligence with autonomous evolution. Specifically, we present the definition, framework, components, and mechanisms of self-evolving embodied AI, systematically review state-of-the-art works for realized components, discuss practical applications, and point out future research directions. We believe that self-evolving embodied AI enables agents to autonomously learn and interact with environments in a human-like manner and provide a new perspective toward general artificial intelligence.

- **链接：** https://arxiv.org/abs/2602.04411v1
- **年份：** 2026
- **发表信息：** arXiv 预印本，arXiv:2602.04411v1 [cs.ET]；兼属 cs.CV。手稿使用 *National Science Review* 模板，但卷期、DOI、收稿及录用信息仍为占位符，故不据此标为已正式发表。

#### 4. 现代智能体系统中的自我改进：综述

**完整中文摘要：**

能够自我改进的自主智能体正从研究原型走向实际部署。其首要目标是实现可控演化，即在极少甚至完全没有人类输入的情况下，根据经验进行适应。本综述把现代自我改进智能体视为一种适应性系统，它能把经验转化为不断积累的能力提升。我们提出一个系统级框架，将现代智能体表示为由基础模型与运行支架共同组成的配置；运行支架包括提示词、记忆、工具和控制逻辑。在这一框架中，自我改进被形式化为一种自发更新算子，用于获取更新并将其写入模型参数或支架组件。我们按照更新对象及驱动变化的信号来组织既有工作，随后回顾应用与评估，最后讨论开放问题和未来方向。为便于使用，我们在 GitHub 页面持续记录技术更新。

**English title:** Self-Improvements in Modern Agentic Systems: A Survey

**Abstract:**

> Self-improving autonomous agents are moving from research prototypes to deployed systems. The primary goal is controllable evolution, or adaptation, from experience with minimal or even no human input. This survey frames modern self-improving agents as adaptive systems that convert experience into accumulated capability gains. We offer a system-level framework that represents a modern agent as a configuration coupling a foundation model with an operational scaffold of prompts, memory, tools, and control logic. Within this framework, self-improvement is formalized as a self-induced update operator that obtains and commits updates to model parameters or scaffold components. We organize prior work by update target and by the signals that drive change, then review applications and discuss evaluation, before closing with open problems and future directions. For convenience, we track technical updates on https://github.com/selfimproving-agent/awesome-Self-Improving-Agents.

- **链接：** https://arxiv.org/abs/2607.13104v1
- **年份：** 2026
- **发表信息：** arXiv 预印本，arXiv:2607.13104v1 [cs.AI]；兼属 cs.CL、cs.LG，97 页、12 幅图。

#### 5. 自演化 AI 智能体综合综述：连接基础模型与终身智能体系统的新范式

**完整中文摘要：**

大语言模型的最新进展引发了人们对能够解决复杂现实任务的 AI 智能体的广泛兴趣。不过，多数现有智能体系统依赖人工制作的配置，这些配置在部署后保持静态，限制了系统适应动态变化环境的能力。为解决这一问题，近期研究开始探索智能体演化技术，根据交互数据和环境反馈自动增强智能体系统。这一新方向为自演化 AI 智能体奠定了基础，使基础模型的静态能力能够与终身智能体系统所需的持续适应能力相连接。在本综述中，我们全面回顾了自演化智能体系统的现有技术。首先，我们提出一个统一概念框架，对自演化智能体系统设计中的反馈闭环进行抽象。该框架包含四个组成部分：系统输入、智能体系统、环境和优化器，可用于理解和比较不同策略。在此基础上，我们系统回顾了面向智能体系统不同组件的多种自演化技术。我们也考察了生物医学、编程和金融等专业领域中的演化策略；在这些领域，优化目标与领域约束紧密关联。我们还专门讨论了自演化智能体系统的评估、安全与伦理问题，这些问题直接关系到系统的有效性和可靠性。本综述旨在帮助研究者和实践者系统理解自演化 AI 智能体，为开发适应性更强、自主程度更高且能够终身运行的智能体系统提供基础。

**English title:** A Comprehensive Survey of Self-Evolving AI Agents: A New Paradigm Bridging Foundation Models and Lifelong Agentic Systems

**Abstract:**

> Recent advances in large language models have sparked growing interest in AI agents capable of solving complex, real-world tasks. However, most existing agent systems rely on manually crafted configurations that remain static after deployment, limiting their ability to adapt to dynamic and evolving environments. To this end, recent research has explored agent evolution techniques that aim to automatically enhance agent systems based on interaction data and environmental feedback. This emerging direction lays the foundation for self-evolving AI agents, which bridge the static capabilities of foundation models with the continuous adaptability required by lifelong agentic systems. In this survey, we provide a comprehensive review of existing techniques for self-evolving agentic systems. Specifically, we first introduce a unified conceptual framework that abstracts the feedback loop underlying the design of self-evolving agentic systems. The framework highlights four key components: System Inputs, Agent System, Environment, and Optimisers, serving as a foundation for understanding and comparing different strategies. Based on this framework, we systematically review a wide range of self-evolving techniques that target different components of the agent system. We also investigate domain-specific evolution strategies developed for specialised fields such as biomedicine, programming, and finance, where optimisation objectives are tightly coupled with domain constraints. In addition, we provide a dedicated discussion on the evaluation, safety, and ethical considerations for self-evolving agentic systems, which are critical to ensuring their effectiveness and reliability. This survey aims to provide researchers and practitioners with a systematic understanding of self-evolving AI agents, laying the foundation for the development of more adaptive, autonomous, and lifelong agentic systems.

- **链接：** https://arxiv.org/abs/2508.07407v2
- **年份：** 2025
- **发表信息：** arXiv 预印本，arXiv:2508.07407v2 [cs.AI]；兼属 cs.CL、cs.MA。

#### 6. 大语言模型自我演化综述

**完整中文摘要：**

大语言模型（LLM）已显著推动多个领域和智能体应用的发展。不过，当前 LLM 依靠人类或外部模型监督进行学习，成本高昂；随着任务复杂性与多样性上升，其性能还可能触及上限。为解决这一问题，使 LLM 能够自主获取、提炼并学习由模型自身产生的经验的自我演化方法正在快速发展。这种受人类经验学习过程启发的新训练范式，有望推动 LLM 向超级智能扩展。本文对 LLM 自我演化方法进行了全面综述。首先，我们提出自我演化的概念框架，将演化过程描述为由四个阶段组成的迭代循环：经验获取、经验提炼、更新和评估。其次，我们对 LLM 及基于 LLM 的智能体的演化目标进行分类，并梳理相关文献，为每个模块提供分类体系与分析见解。最后，我们指出现有挑战，并提出改进自我演化框架的未来方向，为研究者提供必要认识，以加快自演化 LLM 的研发。配套 GitHub 仓库位于 https://github.com/AlibabaResearch/DAMO-ConvAI/tree/main/Awesome-Self-Evolution-of-LLM。

**English title:** A Survey on Self-Evolution of Large Language Models

**Abstract:**

> Large language models (LLMs) have significantly advanced in various fields and intelligent agent applications. However, current LLMs that learn from human or external model supervision are costly and may face performance ceilings as task complexity and diversity increase. To address this issue, self-evolution approaches that enable LLM to autonomously acquire, refine, and learn from experiences generated by the model itself are rapidly growing. This new training paradigm inspired by the human experiential learning process offers the potential to scale LLMs towards superintelligence. In this work, we present a comprehensive survey of self-evolution approaches in LLMs. We first propose a conceptual framework for self-evolution and outline the evolving process as iterative cycles composed of four phases: experience acquisition, experience refinement, updating, and evaluation. Second, we categorize the evolution objectives of LLMs and LLM-based agents; then, we summarize the literature and provide taxonomy and insights for each module. Lastly, we pinpoint existing challenges and propose future directions to improve self-evolution frameworks, equipping researchers with critical insights to fast-track the development of self-evolving LLMs. Our corresponding GitHub repository is available at https://github.com/AlibabaResearch/DAMO-ConvAI/tree/main/Awesome-Self-Evolution-of-LLM

- **链接：** https://arxiv.org/abs/2404.14387v2
- **年份：** 2024
- **发表信息：** arXiv 预印本，arXiv:2404.14387v2 [cs.CL]；兼属 cs.AI。

#### 7. 递归自我改进的形式化及其可能的效率

**完整中文摘要：**

自计算机科学与人工智能早期以来，人们就一直设想递归自我改进（RSI）系统。然而，许多现有 RSI 研究仍停留在哲学讨论层面，缺少清晰的形式化表述和研究结果。本文为一类 RSI 系统给出形式化定义，随后在一个受限版本上证明了可计算且高效的 RSI 系统确实存在。我们通过仿真实验从经验上表明，相对于搜索空间的规模，该系统可达到对数级运行时间复杂度；这些结果说明，高效的递归自我改进有可能实现。

**English title:** A Formulation of Recursive Self-Improvement and Its Possible Efficiency

**Abstract:**

> Recursive self-improving (RSI) systems have been dreamed of since the early days of computer science and artificial intelligence. However, many existing studies on RSI systems remain philosophical, and lacks clear formulation and results. In this paper, we provide a formal definition for one class of RSI systems, and then demonstrate the existence of computable and efficient RSI systems on a restricted version. We use simulation to empirically show that we achieve logarithmic runtime complexity with respect to the size of the search space, and these results suggest it is possible to achieve an efficient recursive self-improvement.

- **链接：** https://arxiv.org/abs/1805.06610v1
- **年份：** 2018
- **发表信息：** arXiv 预印本，arXiv:1805.06610v1 [cs.AI]。

#### 8. 从种子 AI 经由递归自我改进软件走向技术奇点

**完整中文摘要：**

自计算机科学诞生以来，能够改进自身的软件一直是计算机科学家的梦想。本文给出递归自我改进软件的定义，考察不同类型的自我改进软件，回顾相关文献，分析限制递归自我改进的计算极限，并提出 RSI 收敛理论，以预测 RSI 系统的一般行为。最后，我们讨论自我改进智能软件带来的安全影响。

**English title:** From Seed AI to Technological Singularity via Recursively Self-Improving Software

**Abstract:**

> Software capable of improving itself has been a dream of computer scientists since the inception of the field. In this work we provide definitions for Recursively Self-Improving software, survey different types of self-improving software, review the relevant literature, analyze limits on computation restricting recursive self-improvement and introduce RSI Convergence Theory which aims to predict general behavior of RSI systems. Finally, we address security implications from self-improving intelligent software.

- **链接：** https://arxiv.org/abs/1502.06512v1
- **年份：** 2015
- **发表信息：** arXiv 预印本，arXiv:1502.06512v1 [cs.AI]。

#### 9. 奇点：一项哲学分析

**完整中文摘要：**

本章对超级智能展开了丰富的哲学讨论。这篇受到广泛讨论的文章促使心灵哲学家更认真地对待超人类主义、心智上传和奇点。文章首先讨论支持奇点的论证：我们是否有充分理由相信会发生智能爆炸？接着，本章考察如何应对奇点：如果奇点可能发生，我们怎样才能最大限度提高良好结局出现的概率？最后，文章讨论人类在后奇点世界中的位置，特别关注心智上传问题：被上传的人类能否拥有意识，上传能否保留人格同一性？本章也分析了是否可以对人工智能（AI）与超人类人工智能（AI+）的设计施加某些约束，以提高良好结局出现的概率。

**English title:** The Singularity: A Philosophical Analysis

**Abstract:**

> This chapter provides a rich philosophical discussion of superintelligence, a widely discussed piece that has encouraged philosophers of mind to take transhumanism, mind uploading, and the singularity more seriously. It starts with the argument for a singularity: is there good reason to believe that there will be an intelligence explosion? Next, the chapter considers how to negotiate the singularity: if it is possible that there will be a singularity, how can we maximize the chances of a good outcome? Finally, it considers the place of humans in a post‐singularity world, with special attention to questions about uploading: can an uploaded human be conscious, and will uploading preserve personal identity? The chapter also analyzes whether there are certain constraints on design of artificial intelligence (AI) and AI+ that we might impose, in order to increase the chances of a good outcome.

- **链接：** https://consc.net/papers/singularity.pdf
- **年份：** 2010
- **发表信息：** David J. Chalmers，*Journal of Consciousness Studies*，17(9–10)，第 7–65 页，2010。摘要取自澳大利亚国立大学研究门户中的正式书目记录：https://researchportalplus.anu.edu.au/en/publications/the-singularity-a-philosophical-analysis/

#### 10. 哥德尔机：通过可证明最优的自我改进实现自指型通用问题求解

**完整中文摘要：**

我们提出第一类在数学上严谨、通用、完全自指、能够自我改进且具有最优效率的问题求解器。这类问题求解器受到库尔特·哥德尔 1931 年著名自指公式的启发：一旦它找到某项代码重写确有用处的证明，就会重写自身代码的任何部分。依赖具体问题的效用函数、硬件以及全部初始代码，都由公理描述并编码进初始证明搜索器，而该搜索器本身也是初始代码的一部分。搜索器会系统而高效地检验可计算的证明技术，也就是输出为证明的程序，直到找到一种可证明有用且可计算的自我重写。我们证明，这种自我重写是全局最优的，不存在局部极大值，因为代码必须先证明继续搜索其他自我重写方案并无益处，才会执行当前重写。先前方法依赖硬编码的证明搜索器，并不具备自指能力；相比之下，我们的方法不仅在复杂度阶数上最优，而且只要加速带来的效用原则上可被证明，它还能最优地消除 O() 记号所隐藏的任何减速常数。

**English title:** Goedel Machines: Self-Referential Universal Problem Solvers Making Provably Optimal Self-Improvements

**Abstract:**

> We present the first class of mathematically rigorous, general, fully self-referential, self-improving, optimally efficient problem solvers. Inspired by Kurt Goedel's celebrated self-referential formulas (1931), such a problem solver rewrites any part of its own code as soon as it has found a proof that the rewrite is useful, where the problem-dependent utility function and the hardware and the entire initial code are described by axioms encoded in an initial proof searcher which is also part of the initial code. The searcher systematically and efficiently tests computable proof techniques (programs whose outputs are proofs) until it finds a provably useful, computable self-rewrite. We show that such a self-rewrite is globally optimal - no local maxima! - since the code first had to prove that it is not useful to continue the proof search for alternative self-rewrites. Unlike previous non-self-referential methods based on hardwired proof searchers, ours not only boasts an optimal order of complexity but can optimally reduce any slowdowns hidden by the O()-notation, provided the utility of such speed-ups is provable at all.

- **链接：** https://arxiv.org/abs/cs/0309048v5
- **年份：** 2006（arXiv 初次提交于 2003 年；Springer 当前章节页标注 2007）
- **发表信息：** arXiv:cs/0309048v5 [cs.LO]，兼属 cs.AI。该工作的一个版本收录于 Ben Goertzel、Cassio Pennachin 编 *Artificial General Intelligence*（Cognitive Technologies，Springer；arXiv 书目备注为 2006），正式章节题为 “Gödel Machines: Fully Self-referential Optimal Universal Self-improvers”，第 199–226 页；DOI: https://doi.org/10.1007/978-3-540-68677-4_7。Springer 当前书目页按 2007 年引用。

#### 11. 最优有序问题求解器

**完整中文摘要：**

我们提出一种通用方法，以某种意义上的时间最优方式连续解决一个又一个问题。该方法高效搜索能够计算候选解的程序空间，也包括那些可以组织、管理、调整和复用既有知识的程序。最优有序问题求解器（OOPS）受到 Levin 面向单个问题的通用搜索和通用图灵机的启发。对于一个新问题，它会用一部分总搜索时间来检验以可计算方式利用既有求解程序的程序。如果通过复制、编辑或调用既有代码可以比从头求解更快地解决新问题，OOPS 就会发现这一点；如果不能，先前的解至少不会造成太大负担。我们还提出一种高效、递归、基于回溯的实现方式，使 OOPS 能够在存储空间有限的现实计算机上运行。实验说明，OOPS 可以从元学习或元搜索中获得显著收益，也就是通过搜索找到更快的搜索过程。

**English title:** Optimal Ordered Problem Solver

**Abstract:**

> We introduce a general and in a certain sense time-optimal way of solving one problem after another, efficiently searching the space of programs that compute solution candidates, including those programs that organize and manage and adapt and reuse earlier acquired knowledge. The Optimal Ordered Problem Solver (OOPS) draws inspiration from Levin's Universal Search designed for single problems and universal Turing machines. It spends part of the total search time for a new problem on testing programs that exploit previous solution-computing programs in computable ways. If the new problem can be solved faster by copy-editing/invoking previous code than by solving the new problem from scratch, then OOPS will find this out. If not, then at least the previous solutions will not cause much harm. We introduce an efficient, recursive, backtracking-based way of implementing OOPS on realistic computers with limited storage. Experiments illustrate how OOPS can greatly profit from metalearning or metasearching, that is, searching for faster search procedures.

- **链接：** https://link.springer.com/article/10.1023/B:MACH.0000015880.99707.b2
- **年份：** 2004（arXiv 初次提交于 2002 年）
- **发表信息：** Jürgen Schmidhuber，*Machine Learning*，第 54 卷，第 211–254 页，2004 年 3 月；DOI: https://doi.org/10.1023/B:MACH.0000015880.99707.b2。预印本：https://arxiv.org/abs/cs/0207097v2

#### 12. 自指学习中的演化原理，或论如何学习：元—元—……钩子

**完整中文摘要：**

现有不少算法封装了我们称为“学习”的行为的一部分。人们已经编写出通过组块化、泛化、某些类型的类比匹配等方式进行学习的程序。这些算法在某些精心选择的领域中表现良好，却会在许多其他领域失效。

本论文的基本论点是：依靠少量算法无法把握学习的本质。恰恰相反，系统需要一整套依赖情境的学习策略，利用已有信息来获取特定领域的信息。这些策略及其触发条件既复杂又丰富，明显的出路似乎是让系统也能学习“如何学习”的方法。具备这种元学习能力的系统，应把每个问题看成至少包含两个问题：解决该问题，以及改进用于解决该问题的策略。我们当然不希望停在第一个元层级。

实现元能力似乎只有一种途径：将某些初始策略“闭合起来（反馈）并作用于其自身”，使更复杂、更适用的策略得以演化。为此，系统的初始表示必须允许它内省并操纵自身所有相关部分。系统还需要某种演化压力，迫使它以适合其所处环境的方式自我组织。本文将说明演化这一高度一般化原则的基础作用，以及它与学习领域之间的深层联系；还将说明这些观点与冯·魏茨泽克对实用信息的理解以及皮亚杰平衡化模型之间的联系。

论文提出两条通往“学习如何学习”这一目标的路径。两者分别受到人工智能和认知科学中看似相当不同的研究方向启发：一方面，思想来源包括 EURISKO 和 CYRANO 这类符号操作学习程序；另一方面，则包括神经元网络、联想网络、遗传算法及其他“弱”方法的研究，文中还将把它们与几何分形作类比。在这一背景下，论文主张面向对象编程与神经元网络之间的共同点比通常认为的更多。

第二条路径引出了自指联想学习机制（SALM、PSALM）的概念。论文通过实现一种简单、自指且能够自我扩展的语言，以及对该语言施加压力、促使其自我组织所得的一些经验结果，对这一概念加以说明。不过，这些结果不足以展示自指的具体案例。论文将说明，现有机器能力明显低于在合理时间内使“语义自指”很可能出现所需的水平。因此，本文更偏重启发，而不是为通用学习能力提供实用指南。文末附有目录。

**English title:** Evolutionary Principles in Self-Referential Learning, or on Learning How to Learn: The Meta-Meta-... Hook

**Abstract:**

> There exists a number of algorithms which encapsulate parts of the behaviour we call learning. Programs have been written that do learning by chunking, generalization, certain kinds of analogical matching etc. These algorithms work nicely in certain well-chosen domains but fail in many others.
>
> The fundamental statement of this thesis is that we can not capture the essence of learning by relying on a small number of algorithms. Rather on the contrary there is a need for a whole bunch of context-dependent learning strategies to acquire domain specific information by using information that is already available. Because of the complexity and richness of these strategies and their triggering conditions the obvious escape seems to be: Giving a system the ability to learn the methods how to learn, too. A system with such meta-learning capabilities should view every problem as consisting out of at least two problems: Solving it, and improving the strategies employed to solve it. Of course we do not want to stop at the first meta-level!
>
> The only approach to achieve meta-capacity seems to be: ‘Closing (feeding back) some initial strategies onto themselves’ so that more complicated and better suited ones can evolve. This requires an initial representation of the system that allows it to introspect and manipulate all of its relevant parts. Furthermore some kind of evolutionary pressure is needed to force the system to organize itself in a way that makes sense in the environment it is living in. The fundamental role of the very general principle called evolution and its deep interrelations to the field of learning will be emphasized. Connections to v. Weizsacker’s understanding of Pragmatic information as well as to Piaget’s model of equilibration will be shown.
>
> Two approaches to the goal of learning how to learn will be presented, both of them being inspired from seemingly rather different corners of artificial intelligence and cognitive science: One source of ideas is to be found in symbol-manipulative learning programs as EURISKO and CYRANO, the other one in work done on neuronal nets, associative networks, genetical algorithms and other ‘weak’ methods (an analogy to geometric fractals will be drawn). In this context it is argued that object oriented programming and neuronal nets have more things in common than is usually assumed.
>
> The second approach which leads to the notion of self-referential associating learning mechanisms (SALMs, PSALMs) is illustrated by the implementation of a simple self-referential and self-extending language and a few empirical results obtained by putting pressure on that language to organize itself. However, these results are not suited to show concrete cases of self-reference. It will be made obvious that the available machine capacity is clearly below the level that would be necessary to make the creation of ‘semantic self-reference’ likely (on the basis of the second approach and within a reasonable time). Thus this paper tends to having inspiring character rather than presenting a practical guidance to universal learning capabilities. A table of contents is supplied at the end of this work.

- **链接：** https://people.idsia.ch/~juergen/diploma1987ocr.pdf
- **年份：** 1987
- **发表信息：** Jürgen Schmidhuber，慕尼黑工业大学（Technische Universität München）计算机科学文凭论文，1987 年 5 月 14 日。英文摘要依据作者网站所载 OCR 全文整理，仅修正明显 OCR 断词、字符和拼写识别错误。

#### 13. 关于第一台超智能机器的推测

**完整中文摘要：**

**未提供。** 经核对 Elsevier/ScienceDirect 章节记录、Crossref DOI 元数据及可检索的原始论文记录，未找到由作者或出版社提供的正式摘要。为避免把 README 简介、论文导言或第三方概述冒充摘要，此处不另行编写替代文本。

**English title:** Speculations Concerning the First Ultraintelligent Machine

**Abstract:**

> **Not available.** No author- or publisher-supplied abstract was found in the Elsevier/ScienceDirect chapter record, the Crossref DOI metadata, or the accessible original-paper records. The README description and the paper's introduction are not substituted for an abstract.

- **链接：** https://www.sciencedirect.com/science/article/pii/S0065245808604180
- **年份：** 1965（该文通常按 1965 年引用；Crossref/Elsevier 当前元数据将印刷出版年记为 1966）
- **发表信息：** Irving John Good，载于 Franz L. Alt、Morris Rubinoff 编 *Advances in Computers*，第 6 卷，第 31–88 页，Academic Press；DOI: https://doi.org/10.1016/S0065-2458(08)60418-0。不同权威书目对年份采用 1965 或 1966，本文同时保留两种记录。

## 模型层递归自我改进 / Model-level RSI

本节收录通过自生成反馈、数据或推理来改进模型权重或训练行为的方法，也包括后来被持续自我改进循环复用的经典基础方法。

This section covers methods that improve model weights or training behavior through self-generated feedback, data, or reasoning, including canonical enabling methods later reused in persistent self-improvement loops.

### 自训练与自奖励 / Self-Training & Self-Reward

#### 1. EvoLM：通过共同演化的判别式量规实现语言模型自演化

- **中文标题：** EvoLM：通过共同演化的判别式量规实现语言模型自演化
- **完整中文摘要：** 语言模型通过预训练编码了大量评估性知识，但当前的后训练方法依赖外部监督（人工标注、专有模型或标量奖励模型）来产生奖励信号，而每一种方式都存在上限。人类判断无法监督超越人类自身的能力，专有 API 会造成依赖，可验证奖励则仅覆盖具有标准答案的领域。利用模型自身的评估能力进行自我改进，是一种可随模型本身扩展的奖励来源，但目前的方法基本尚未开发这种来源。我们提出 EVOLM，这是一种后训练方法，它将这种能力组织成显式的判别式量规，并将其用作训练信号。EVOLM 在单个语言模型内部交替训练两种能力：（1）量规生成器，生成针对具体实例的评估标准，并针对判别效用进行优化，使一个小型冻结评判模型区分优选回答与非优选回答的能力最大化；（2）使用这些基于量规的分数作为奖励来训练策略。所有偏好信号均通过当前策略输出与早期检查点输出之间的时间对比来构造，无需人工标注或外部监督。EVOLM 训练一个 Qwen3-8B 模型生成量规，其在 RewardBench-2 上的表现比 GPT-4.1 高 25.7%。共同训练的策略在 OLMo3-Adapt 套件上取得 69.3% 的平均成绩，分别比使用 GPT-4.1 提示生成量规所训练的策略高 3.9%，比使用最先进的 8B 奖励模型 SkyWork-RM 所训练的策略高 16%。总体而言，EVOLM 表明，将模型的评估能力组织为共同演化的判别式量规，可以在没有外部监督的情况下实现自我改进。
- **English title:** EvoLM: Self-Evolving Language Models through Co-Evolved Discriminative Rubrics
- **Original Abstract:** Language models encode substantial evaluative knowledge from pretraining, yet current post-training methods rely on external supervision (human annotations, proprietary models, or scalar reward models) to produce reward signals. Each imposes a ceiling. Human judgment cannot supervise capabilities beyond its own, proprietary APIs create dependencies, and verifiable rewards cover only domains with ground-truth answers. Self-improvement from a model's own evaluative capacity is a reward source that scales with the model itself, yet remains largely untapped by current methods. We introduce EVOLM, a post-training method that structures this capacity into explicit discriminative rubrics and uses them as training signal. EVOLM trains two capabilities within a single language model in alternation: (1) a rubric generator producing instance-specific evaluation criteria optimized for discriminative utility, which maximizes a small frozen judge's ability to distinguish preferred from dispreferred responses; and (2) a policy trained using those rubric-conditioned scores as reward. All preference signals are constructed from the policy's own outputs via temporal contrast with earlier checkpoints, requiring no human annotation or external supervision. EVOLM trains a Qwen3-8B model to generate rubrics that outperform GPT-4.1 on RewardBench-2 by 25.7%. The co-trained policy achieves 69.3% average on the OLMo3-Adapt suite, outperforming policies trained with GPT-4.1 prompted rubrics by 3.9% and with the state-of-the-art 8B reward model SkyWork-RM by 16%. Overall, EVOLM demonstrates that structuring a model's evaluative capacity into co-evolving discriminative rubrics enables self-improvement without external supervision.
- **链接 / Link：** https://arxiv.org/abs/2605.03871v1
- **年份 / Year：** 2026
- **发表信息 / Publication：** arXiv 预印本 / arXiv preprint

#### 2. RLAIF 与 RLHF：利用 AI 反馈扩展基于人类反馈的强化学习

- **中文标题：** RLAIF 与 RLHF：利用 AI 反馈扩展基于人类反馈的强化学习
- **完整中文摘要：** 基于人类反馈的强化学习（RLHF）已被证明能够有效地使大型语言模型（LLM）与人类偏好对齐，但收集高质量偏好标签的成本很高。Bai 等人提出的基于 AI 反馈的强化学习（RLAIF）提供了一种很有前景的替代方案：使用现成 LLM 生成的偏好来训练奖励模型（RM）。在摘要生成、有用对话生成和无害对话生成任务上，我们证明 RLAIF 能够取得与 RLHF 相当的性能。此外，我们朝“自我改进”迈出了一步：即使 AI 标注模型与策略模型规模相同，甚至与初始策略采用完全相同的检查点，RLAIF 仍然可以优于监督微调基线。最后，我们提出直接 RLAIF（d-RLAIF），该技术在强化学习期间直接从现成 LLM 获取奖励，从而绕过奖励模型训练，并取得优于标准 RLAIF 的性能。我们的结果表明，RLAIF 可以达到与使用人类反馈相当的性能，为解决 RLHF 的可扩展性限制提供了一种潜在方案。
- **English title:** RLAIF vs. RLHF: Scaling Reinforcement Learning from Human Feedback with AI Feedback
- **Original Abstract:** Reinforcement learning from human feedback (RLHF) has proven effective in aligning large language models (LLMs) with human preferences, but gathering high-quality preference labels is expensive. RL from AI Feedback (RLAIF), introduced in Bai et al., offers a promising alternative that trains the reward model (RM) on preferences generated by an off-the-shelf LLM. Across the tasks of summarization, helpful dialogue generation, and harmless dialogue generation, we show that RLAIF achieves comparable performance to RLHF. Furthermore, we take a step towards "self-improvement" by demonstrating that RLAIF can outperform a supervised fine-tuned baseline even when the AI labeler is the same size as the policy, or even the exact same checkpoint as the initial policy. Finally, we introduce direct-RLAIF (d-RLAIF) - a technique that circumvents RM training by obtaining rewards directly from an off-the-shelf LLM during RL, which achieves superior performance to canonical RLAIF. Our results suggest that RLAIF can achieve performance on-par with using human feedback, offering a potential solution to the scalability limitations of RLHF.
- **链接 / Link：** https://arxiv.org/abs/2309.00267v3
- **年份 / Year：** 2024
- **发表信息 / Publication：** 第 41 届国际机器学习大会（ICML 2024）/ Proceedings of the 41st International Conference on Machine Learning (ICML 2024)

#### 3. 自博弈微调将弱语言模型转化为强语言模型

- **中文标题：** 自博弈微调将弱语言模型转化为强语言模型
- **完整中文摘要：** 通过监督微调（SFT）利用人工标注数据的力量，是推动大型语言模型（LLM）发展的关键。在本文中，我们探讨如何在不获取额外人工标注数据的情况下，将一个弱 LLM 培养成强 LLM。我们提出一种名为自博弈微调（Self-Play fIne-tuNing，SPIN）的新微调方法，它从一个经过监督微调的模型开始。SPIN 的核心是一种自博弈机制，其中 LLM 通过与自身的不同实例对弈来提升能力。更具体地说，LLM 从先前迭代中生成自己的训练数据，并通过辨别这些自生成回答与人工标注数据中的回答来改进其策略。我们的方法逐步将 LLM 从初生模型提升为强大模型，充分释放用于 SFT 的人工标注示范数据的潜力。在理论上，我们证明，只有当 LLM 策略与目标数据分布一致时，该方法训练目标函数的全局最优解才会实现。在实证方面，我们在多个基准数据集上评估了该方法，包括 HuggingFace Open LLM Leaderboard、MT-Bench 以及 Big-Bench 中的数据集。结果表明，SPIN 能够显著提升 LLM 在多种基准上的性能，甚至优于使用额外 GPT-4 偏好数据进行直接偏好优化（DPO）训练的模型。这说明自博弈很有潜力让 LLM 在不需要专家对手的情况下达到人类水平。代码发布于 https://github.com/uclaml/SPIN。
- **English title:** Self-Play Fine-Tuning Converts Weak Language Models to Strong Language Models
- **Original Abstract:** Harnessing the power of human-annotated data through Supervised Fine-Tuning (SFT) is pivotal for advancing Large Language Models (LLMs). In this paper, we delve into the prospect of growing a strong LLM out of a weak one without the need for acquiring additional human-annotated data. We propose a new fine-tuning method called Self-Play fIne-tuNing (SPIN), which starts from a supervised fine-tuned model. At the heart of SPIN lies a self-play mechanism, where the LLM refines its capability by playing against instances of itself. More specifically, the LLM generates its own training data from its previous iterations, refining its policy by discerning these self-generated responses from those obtained from human-annotated data. Our method progressively elevates the LLM from a nascent model to a formidable one, unlocking the full potential of human-annotated demonstration data for SFT. Theoretically, we prove that the global optimum to the training objective function of our method is achieved only when the LLM policy aligns with the target data distribution. Empirically, we evaluate our method on several benchmark datasets including the HuggingFace Open LLM Leaderboard, MT-Bench, and datasets from Big-Bench. Our results show that SPIN can significantly improve the LLM's performance across a variety of benchmarks and even outperform models trained through direct preference optimization (DPO) supplemented with extra GPT-4 preference data. This sheds light on the promise of self-play, enabling the achievement of human-level performance in LLMs without the need for expert opponents. Codes are available at https://github.com/uclaml/SPIN.
- **链接 / Link：** https://arxiv.org/abs/2401.01335v3
- **年份 / Year：** 2024
- **发表信息 / Publication：** 第 41 届国际机器学习大会（ICML 2024）/ Proceedings of the 41st International Conference on Machine Learning (ICML 2024)

#### 4. 自奖励语言模型

- **中文标题：** 自奖励语言模型
- **完整中文摘要：** 我们认为，要实现超人智能体，未来模型需要超人水平的反馈，以提供充分的训练信号。当前方法通常根据人类偏好训练奖励模型，因此可能受限于人类的能力水平；其次，这些独立且冻结的奖励模型无法在 LLM 训练期间继续学习和改进。在这项工作中，我们研究自奖励语言模型，其中语言模型本身通过“LLM 即评判者”的提示方式，在训练过程中为自己提供奖励。我们证明，在迭代式 DPO 训练过程中，不仅模型遵循指令的能力得到提高，其向自身提供高质量奖励的能力也得到提升。使用我们的方法对 Llama 2 70B 进行三轮迭代微调后，所得模型在 AlpacaEval 2.0 排行榜上超过了许多现有系统，包括 Claude 2、Gemini Pro 和 GPT-4 0613。尽管仍有大量问题有待探索，这项工作为模型在这两个维度上持续改进的可能性打开了大门。
- **English title:** Self-Rewarding Language Models
- **Original Abstract:** We posit that to achieve superhuman agents, future models require superhuman feedback in order to provide an adequate training signal. Current approaches commonly train reward models from human preferences, which may then be bottlenecked by human performance level, and secondly these separate frozen reward models cannot then learn to improve during LLM training. In this work, we study Self-Rewarding Language Models, where the language model itself is used via LLM-as-a-Judge prompting to provide its own rewards during training. We show that during Iterative DPO training that not only does instruction following ability improve, but also the ability to provide high-quality rewards to itself. Fine-tuning Llama 2 70B on three iterations of our approach yields a model that outperforms many existing systems on the AlpacaEval 2.0 leaderboard, including Claude 2, Gemini Pro, and GPT-4 0613. While there is much left still to explore, this work opens the door to the possibility of models that can continually improve in both axes.
- **链接 / Link：** https://arxiv.org/abs/2401.10020v3
- **年份 / Year：** 2024
- **发表信息 / Publication：** 第 41 届国际机器学习大会（ICML 2024）/ Proceedings of the 41st International Conference on Machine Learning (ICML 2024)

### 合成数据与自蒸馏 / Synthetic Data & Self-Distillation

#### 5. 面向长时程终端任务的递归合成

- **中文标题：** 面向长时程终端任务的递归合成
- **完整中文摘要：** 为终端智能体制作高质量的长时程训练数据成本高昂，每项任务通常需要数百至数千美元，因为每项任务都必须确保指令、环境、参考解法和验证器彼此一致。人工编写无法规模化，而使用大型语言模型（LLM）直接生成又常常会破坏这些依赖关系。我们提出递归合成终端任务（Recursive Synthetic Terminal Tasks，RST），这是一种用于规模化构建长时程终端智能体任务的递归验证合成框架。RST 从经过验证的种子任务出发，扩展参考解法，使验证器和指令与新工作流重新对齐，在全新沙箱中验证结果，并将通过验证的任务重新用作后续轮次的种子。经过十五轮递归，RST 生成了 37,484 个合成终端智能体任务，每项任务成本约为 0.05 美元。任务难度随轮次显著增加：参考解法的中位行数从 67 增长至 374，执行命令数的中位数从 40 增长至 244，DeepSeek-V4-Pro 的 pass@4 则从第 1 轮的 90% 降至第 15 轮的 2.5%。为了证明这些数据的训练效用，我们在合成任务上收集经过拒绝采样的 Qwen3.5 轨迹，并将其用于监督微调。在 Terminal-Bench 2、Terminal-Bench Hard 和 Long-Horizon Terminal Bench 上，使用这些轨迹微调可使 Qwen3.5-27B 和 Qwen3.5-122B-A10B 最多提升 10 个百分点；智能体式 PPO 则使 Qwen3.5-27B 在三个基准上的成绩分别提升至 49.44%、32.00% 和 22.07%，相对基础模型分别提高 20.0%、41.2% 和 21.9%。此外，经过 15 轮后，该递归过程仍未显示上限：随着难度持续提高，合成产出率和验证通过率保持稳定，表明这一过程可以继续扩展到远超本文报告的规模。
- **English title:** Recursive Synthesis for Long-Horizon Terminal Tasks
- **Original Abstract:** High-quality long-horizon training data for terminal agents is expensive to produce, often costing hundreds to thousands of dollars per task, because each task must keep the instruction, environment, reference solution, and verifier mutually consistent. Human authoring does not scale, and direct generation with large language models (LLMs) often breaks these dependencies. We present Recursive Synthetic Terminal Tasks (RST), a recursive verified synthesis framework for constructing long-horizon terminal-agent tasks at scale. Starting from verified seed tasks, RST extends the reference solution, realigns the verifier and instruction to the new workflow, validates the result in a fresh sandbox, and reuses accepted tasks as seeds for subsequent rounds. Across fifteen recursive rounds, RST produces 37,484 synthesized terminal-agent tasks at roughly $0.05 per task. Task difficulty increases substantially over rounds: the median reference solution grows from 67 to 374 lines, the median number of executed commands grows from 40 to 244, and DeepSeek-V4-Pro pass@4 drops from 90% at R1 to 2.5% at R15. To demonstrate training utility, we collect rejection-sampled Qwen3.5 trajectories on the synthesized tasks and use them for supervised fine-tuning. Fine-tuning on these trajectories improves Qwen3.5-27B and Qwen3.5-122B-A10B by up to 10 points on Terminal-Bench 2, Terminal-Bench Hard, and Long-Horizon Terminal Bench, while agentic PPO lifts Qwen3.5-27B to 49.44%, 32.00%, and 22.07% on the three benchmarks, corresponding to relative gains of 20.0%, 41.2%, and 21.9% over the base model. Moreover, after 15 rounds, the recursion shows no ceiling: synthesis yield and validation rates remain stable as difficulty keeps climbing, indicating that the process can continue well beyond the scale reported here.
- **链接 / Link：** https://arxiv.org/abs/2608.05466v3
- **年份 / Year：** 2026
- **发表信息 / Publication：** arXiv 预印本 / arXiv preprint

#### 6. 超越人类数据：扩展语言模型面向问题求解的自训练

- **中文标题：** 超越人类数据：扩展语言模型面向问题求解的自训练
- **完整中文摘要：** 使用人类生成的数据微调语言模型（LM）仍是一种普遍做法。然而，这类模型的性能往往受高质量人类数据的数量和多样性限制。在本文中，我们探讨在能够获得标量反馈的任务上是否可以超越人类数据，例如，在可以验证答案正确性的数学问题上。为此，我们研究了一种基于期望最大化的简单自训练方法，并将其称为 ReST<sup>EM</sup>。该方法会：（1）从模型生成样本并使用二元反馈进行筛选；（2）使用这些样本微调模型；（3）将这一过程重复若干次。我们使用 PaLM-2 模型在高阶 MATH 推理和 APPS 编程基准上进行测试，发现 ReST<sup>EM</sup> 能够随模型规模有效扩展，并显著超过仅使用人类数据进行微调的方法。总体而言，我们的发现表明，结合反馈的自训练可以大幅降低对人类生成数据的依赖。
- **English title:** Beyond Human Data: Scaling Self-Training for Problem-Solving with Language Models
- **Original Abstract:** Fine-tuning language models (LMs) on human-generated data remains a prevalent practice. However, the performance of such models is often limited by the quantity and diversity of high-quality human data. In this paper, we explore whether we can go beyond human data on tasks where we have access to scalar feedback, for example, on math problems where one can verify correctness. To do so, we investigate a simple self-training method based on expectation-maximization, which we call ReST<sup>EM</sup>, where we (1) generate samples from the model and filter them using binary feedback, (2) fine-tune the model on these samples, and (3) repeat this process a few times. Testing on advanced MATH reasoning and APPS coding benchmarks using PaLM-2 models, we find that ReST<sup>EM</sup> scales favorably with model size and significantly surpasses fine-tuning only on human data. Overall, our findings suggest self-training with feedback can substantially reduce dependence on human-generated data.
- **链接 / Link：** https://arxiv.org/abs/2312.06585v4
- **年份 / Year：** 2024
- **发表信息 / Publication：** Transactions on Machine Learning Research（TMLR，2024）

#### 7. 通过指令反向翻译实现自对齐

- **中文标题：** 通过指令反向翻译实现自对齐
- **完整中文摘要：** 我们提出一种可扩展的方法，通过为人类撰写的文本自动标注相应指令，构建高质量的指令遵循语言模型。我们的方法称为指令反向翻译，它从一个在少量种子数据上微调的语言模型和给定的网络语料库开始。种子模型通过为网络文档生成指令提示来构建训练样本（自增强），然后从这些候选样本中选择高质量样本（自筛选）。随后，这些数据被用于微调一个更强的模型。使用我们的方法对 LLaMa 进行两轮迭代微调后，所得模型在 Alpaca 排行榜上超过了所有其他不依赖蒸馏数据的 LLaMa 系模型，展现出非常有效的自对齐能力。
- **English title:** Self-Alignment with Instruction Backtranslation
- **Original Abstract:** We present a scalable method to build a high quality instruction following language model by automatically labelling human-written text with corresponding instructions. Our approach, named instruction backtranslation, starts with a language model finetuned on a small amount of seed data, and a given web corpus. The seed model is used to construct training examples by generating instruction prompts for web documents (self-augmentation), and then selecting high quality examples from among these candidates (self-curation). This data is then used to finetune a stronger model. Finetuning LLaMa on two iterations of our approach yields a model that outperforms all other LLaMa-based models on the Alpaca leaderboard not relying on distillation data, demonstrating highly effective self-alignment.
- **链接 / Link：** https://arxiv.org/abs/2308.06259v3
- **年份 / Year：** 2024
- **发表信息 / Publication：** 第十二届国际学习表征会议（ICLR 2024）/ The Twelfth International Conference on Learning Representations (ICLR 2024)

#### 8. 大型语言模型能够自我改进

- **中文标题：** 大型语言模型能够自我改进
- **完整中文摘要：** 大型语言模型（LLM）已在多种任务上取得出色表现。然而，微调 LLM 需要大量监督。另一方面，人类可以通过自我思考，在没有外部输入的情况下提高推理能力。在这项工作中，我们证明，仅使用无标注数据集，LLM 同样能够实现自我改进。我们使用预训练 LLM，通过思维链提示和自一致性为无标注问题生成带有推理过程的“高置信度”答案，再以这些自生成解答作为目标输出微调该 LLM。结果表明，我们的方法提高了一个 5400 亿参数 LLM 的通用推理能力（GSM8K 从 74.4% 提高到 82.1%，DROP 从 78.2% 提高到 83.0%，OpenBookQA 从 90.0% 提高到 94.4%，ANLI-A3 从 63.4% 提高到 67.9%），并在没有任何真实标签的情况下达到最先进水平。我们还进行了消融研究，结果表明，对推理过程进行微调是实现自我改进的关键。
- **English title:** Large Language Models Can Self-Improve
- **Original Abstract:** Large Language Models (LLMs) have achieved excellent performances in various tasks. However, fine-tuning an LLM requires extensive supervision. Human, on the other hand, may improve their reasoning abilities by self-thinking without external inputs. In this work, we demonstrate that an LLM is also capable of self-improving with only unlabeled datasets. We use a pre-trained LLM to generate "high-confidence" rationale-augmented answers for unlabeled questions using Chain-of-Thought prompting and self-consistency, and fine-tune the LLM using those self-generated solutions as target outputs. We show that our approach improves the general reasoning ability of a 540B-parameter LLM (74.4%->82.1% on GSM8K, 78.2%->83.0% on DROP, 90.0%->94.4% on OpenBookQA, and 63.4%->67.9% on ANLI-A3) and achieves state-of-the-art-level performance, without any ground truth label. We conduct ablation studies and show that fine-tuning on reasoning is critical for self-improvement.
- **链接 / Link：** https://arxiv.org/abs/2210.11610v2
- **年份 / Year：** 2023
- **发表信息 / Publication：** 2023 年自然语言处理经验方法会议（EMNLP 2023）/ Conference on Empirical Methods in Natural Language Processing (EMNLP 2023)

#### 9. Self-Instruct：使用自生成指令对齐语言模型

- **中文标题：** Self-Instruct：使用自生成指令对齐语言模型
- **完整中文摘要：** 大型“指令微调”语言模型（即经过微调以响应指令的模型）展现出显著的能力，可以零样本泛化到新任务。尽管如此，这些模型高度依赖人工编写的指令数据，而这类数据的数量、多样性和创造性通常有限，从而制约了微调模型的通用性。我们提出 Self-Instruct，这是一个通过利用预训练语言模型自身生成内容进行自举，从而提高其指令遵循能力的框架。我们的流程使用语言模型生成指令、输入和输出样本，过滤无效或相似的样本，再使用剩余样本微调原始模型。将该方法应用于原始 GPT-3 后，我们在 Super-NaturalInstructions 上取得相对于原始模型 33% 的绝对提升，表现与使用私有用户数据和人工标注训练的 InstructGPT-001 相当。为了进一步评估，我们还整理了一组由专家为新任务编写的指令。人工评估表明，使用 Self-Instruct 微调 GPT-3 的表现大幅超过使用现有公开指令数据集的方法，与 InstructGPT-001 之间仅剩 5% 的绝对差距。Self-Instruct 为使预训练语言模型与指令对齐提供了一种几乎无需标注的方法；我们发布了大型合成数据集，以推动未来的指令微调研究。代码和数据发布于 https://github.com/yizhongw/self-instruct。
- **English title:** Self-Instruct: Aligning Language Models with Self-Generated Instructions
- **Original Abstract:** Large "instruction-tuned" language models (i.e., finetuned to respond to instructions) have demonstrated a remarkable ability to generalize zero-shot to new tasks. Nevertheless, they depend heavily on human-written instruction data that is often limited in quantity, diversity, and creativity, therefore hindering the generality of the tuned model. We introduce Self-Instruct, a framework for improving the instruction-following capabilities of pretrained language models by bootstrapping off their own generations. Our pipeline generates instructions, input, and output samples from a language model, then filters invalid or similar ones before using them to finetune the original model. Applying our method to the vanilla GPT3, we demonstrate a 33% absolute improvement over the original model on Super-NaturalInstructions, on par with the performance of InstructGPT-001, which was trained with private user data and human annotations. For further evaluation, we curate a set of expert-written instructions for novel tasks, and show through human evaluation that tuning GPT3 with Self-Instruct outperforms using existing public instruction datasets by a large margin, leaving only a 5% absolute gap behind InstructGPT-001. Self-Instruct provides an almost annotation-free method for aligning pre-trained language models with instructions, and we release our large synthetic dataset to facilitate future studies on instruction tuning. Our code and data are available at https://github.com/yizhongw/self-instruct.
- **链接 / Link：** https://arxiv.org/abs/2212.10560v2
- **年份 / Year：** 2023
- **发表信息 / Publication：** 第 61 届计算语言学协会年会（ACL 2023）/ Proceedings of the 61st Annual Meeting of the Association for Computational Linguistics (ACL 2023)

### 自博弈与迭代微调 / Self-Play & Iterative Fine-tuning

#### 10. 学习自我演化

- **中文标题：** 学习自我演化
- **完整中文摘要：** 我们提出学习自我演化（Learning to Self-Evolve，LSE），这是一个训练大型语言模型（LLM）在测试时改进自身上下文的强化学习框架。我们将 LSE 置于测试时自我演化场景中：模型根据已见问题的反馈迭代改进其上下文，以便在新问题上表现得更好。现有方法完全依赖模型固有的推理能力，从未明确训练模型完成这一任务。LSE 将多步演化问题化简为单步强化学习目标，其中每次上下文编辑所获得的奖励取决于其带来的下游性能提升。我们将这一目标与树引导的演化循环相结合。在文本到 SQL 生成（BIRD）和通用问答（MMLU-Redux）任务上，一个使用 LSE 训练的 4B 参数模型超过了由 GPT-5 和 Claude Sonnet 4.5 驱动的自演化策略，以及 GEPA 和 TextGrad 等提示优化方法；它还可以在无需额外训练的情况下迁移，用于指导其他模型。结果凸显了将自我演化视为一种可学习技能的有效性。
- **English title:** Learning to Self-Evolve
- **Original Abstract:** We introduce Learning to Self-Evolve (LSE), a reinforcement learning framework that trains large language models (LLMs) to improve their own contexts at test time. We situate LSE in the setting of test-time self-evolution, where a model iteratively refines its context from feedback on seen problems to perform better on new ones. Existing approaches rely entirely on the inherent reasoning ability of the model and never explicitly train it for this task. LSE reduces the multi-step evolution problem to a single-step RL objective, where each context edit is rewarded by the improvement in downstream performance. We pair this objective with a tree-guided evolution loop. On Text-to-SQL generation (BIRD) and general question answering (MMLU-Redux), a 4B-parameter model trained with LSE outperforms self-evolving policies powered by GPT-5 and Claude Sonnet 4.5, as well as prompt optimization methods including GEPA and TextGrad, and transfers to guide other models without additional training. Our results highlight the effectiveness of treating self-evolution as a learnable skill.
- **链接 / Link：** https://arxiv.org/abs/2603.18620v1
- **年份 / Year：** 2026
- **发表信息 / Publication：** arXiv 预印本 / arXiv preprint

#### 11. SERPO：面向开放式测试时强化学习的自演化量规策略优化

- **中文标题：** SERPO：面向开放式测试时强化学习的自演化量规策略优化
- **完整中文摘要：** 测试时强化学习（TTRL）使语言模型能够在推理阶段无需标签反馈而实现自我演化。现有方法依赖答案投票，因此无法自然扩展到开放式生成任务，因为有效回答不能映射到同一个标准答案。在没有外部奖励模型或更强评判模型的情况下，适应过程必须转而从模型自身输出中构造可靠奖励。我们提出 SERPO（Self-Evolving Rubric Policy Optimization，自演化量规策略优化），以一个共同演化回答证据、针对查询的量规和策略参数的闭环取代答案投票。优—中—劣（Good-Normal-Bad，G-N-B）回答演化将差异最大的 rollout 组织成有序档案；量规演化保留能够区分这些档案的标准；概率式标准评分将判决词元的似然转换为奖励信号；策略演化则使用所得信号优化行动者。新的行动者 rollout 随后会同时更新档案与量规，从而闭合三方演化循环。在两种模型配置、两个域内基准和四个分布外（OOD）基准上，相对于对应基础模型，SERPO 在 HealthBench 和 ResearchQA 上最多分别提升 20.63 和 20.31 分，使六个基准的宏平均最多提高 8.06 分，并支持 OOD 迁移和持续的跨基准演化。
- **English title:** SERPO: Self-Evolving Rubric Policy Optimization for Open-Ended Test-Time Reinforcement Learning
- **Original Abstract:** Test-time reinforcement learning (TTRL) enables language models to self-evolve at inference time without labeled feedback. Existing methods rely on answer voting and therefore do not extend naturally to open-ended generation, where valid responses cannot be mapped to a shared canonical answer. Without external reward models or stronger judges, adaptation must instead construct reliable rewards from the model's own outputs. We introduce SERPO (Self-Evolving Rubric Policy Optimization), which replaces answer voting with a closed loop that co-evolves response evidence, query-specific rubrics, and policy parameters. Good-Normal-Bad (G-N-B) response evolution organizes maximally separated rollouts into ordered archives; rubric evolution retains criteria that discriminate these archives; probabilistic criterion scoring converts verdict-token likelihoods into reward signals; and policy evolution optimizes the actor with the resulting signals. New actor rollouts then refresh both the archives and rubrics, closing the three-way evolution loop. Across two model configurations, two in-domain benchmarks, and four OOD benchmarks, SERPO improves HealthBench and ResearchQA by up to 20.63 and 20.31 points over the corresponding base models, raises the six-benchmark macro-average by up to 8.06 points, and supports OOD transfer and continued cross-benchmark evolution.
- **链接 / Link：** https://arxiv.org/abs/2607.26873v2
- **年份 / Year：** 2026
- **发表信息 / Publication：** arXiv 预印本 / arXiv preprint

#### 12. 教会 LLM 自我演化：通过强化学习培养核心元技能

- **中文标题：** 教会 LLM 自我演化：通过强化学习培养核心元技能
- **完整中文摘要：** AlphaEvolve 所展示的、利用环境反馈进行迭代式自我演化的测试时扩展取得了显著的性能提升。我们假设，这类演化框架的成功取决于一些元技能，例如结合环境反馈进行自我反思；这些技能使有效的多轮改进成为可能，却在传统后训练中基本被忽视。为弥合这一缺口，我们提出 MetaEvolve，该框架通过数据合成流程、面向演化的强化学习（RL）和推理时演化搜索来培养这些元技能。具体而言，我们将 MetaEvolve 落地到编程领域，因为程序执行能提供超越二元正确性的、自然且连续的奖励信号。在这些信号的基础上，我们合成演化轨迹作为训练数据；每条轨迹都包含当前程序、其适应度分数（综合正确性和效率）以及先前尝试的历史，并使用由测试用例执行产生的可验证奖励，通过 RL 训练模型。通过在大规模代码数据上训练，我们力图激发可泛化且与领域无关的元技能，使其能够广泛迁移到缺少此类丰富训练信号的开放式问题。在七个编程基准上，MetaEvolve 在分布内任务上比最强基线绝对提高 10.01%，在分布外任务上提高 24.12%。在完全位于训练域之外的开放式算法优化问题上，其表现进一步取得 46.9% 的相对提升。这些结果表明，显式培养自我演化元技能，为构建能力更强、能够更自主地自我演化的 AI 提供了一条有原则的路径。
- **English title:** Teaching LLMs to Self-Evolve: Cultivating Core Meta-Skills with Reinforcement Learning
- **Original Abstract:** Test-time scaling through iterative self-evolution with environment feedback, as demonstrated by AlphaEvolve, shows remarkable performance gains. We hypothesize that the success of such evolution frameworks hinges on meta-skills, such as self-reflection with environment feedback, that enable effective multi-round refinement, yet are largely neglected by traditional post-training. To bridge this gap, we present MetaEvolve, a framework designed to develop these meta-skills via a data synthesis pipeline, evolution-aware reinforcement learning (RL), and inference-time evolutionary search. Concretely, we ground MetaEvolve in coding, where program execution provides natural, continuous reward signals beyond binary correctness. Building on these signals, we synthesize evolution trajectories as training data, each containing a current program, its fitness score (combining correctness and efficiency), and a history of prior attempts, and train the model via RL with verifiable rewards derived from test case execution. By training on large-scale code data, we aim to inspire generalizable domain-agnostic meta-skills that can transfer broadly to open-ended problems where such rich training signals are scarce. Across seven coding benchmarks, MetaEvolve outperforms the strongest baseline by 10.01% absolute on in-distribution tasks and 24.12% on out-of-distribution tasks. On open-ended algorithm optimization problems entirely outside the training domain, it further achieves a 46.9% relative improvement. These results demonstrate that explicitly cultivating self-evolution meta-skills offers a principled path toward more capable and autonomously self-evolving AI.
- **链接 / Link：** https://arxiv.org/abs/2607.21971v1
- **年份 / Year：** 2026
- **发表信息 / Publication：** arXiv 预印本 / arXiv preprint

#### 13. TEMPO：扩展大型推理模型的测试时训练

- **中文标题：** TEMPO：扩展大型推理模型的测试时训练
- **完整中文摘要：** 测试时训练（TTT）在推理期间利用无标注测试实例调整模型参数，从而不断将模型能力扩展到离线训练所及范围之外。尽管初期有所提升，现有面向大型推理模型（LRM）的 TTT 方法很快就会进入平台期，无法从更多测试时计算中获益。在没有外部校准的情况下，随着策略模型不断演化，自生成奖励信号会日益漂移，导致性能停滞和多样性坍缩。我们提出 TEMPO，这是一个 TTT 框架，它交替执行无标注问题上的策略优化与有标注数据集上的周期性评判器重新校准。通过使用期望最大化（EM）算法对这一交替过程进行形式化，我们揭示了先前方法可以被解释为省略了关键重新校准步骤的不完整变体。重新引入该步骤可以收紧证据下界（ELBO），并实现持续改进。在多种模型家族（Qwen3 和 OLMO3）及推理任务上，TEMPO 将 OLMO3-7B 在 AIME 2024 上的成绩从 33.0% 提高到 51.1%，将 Qwen3-14B 从 42.3% 提高到 65.8%，同时保持较高的多样性。
- **English title:** TEMPO: Scaling Test-time Training for Large Reasoning Models
- **Original Abstract:** Test-time training (TTT) adapts model parameters on unlabeled test instances during inference time, which continuously extends capabilities beyond the reach of offline training. Despite initial gains, existing TTT methods for LRMs plateau quickly and do not benefit from additional test-time compute. Without external calibration, the self-generated reward signal increasingly drifts as the policy model evolves, leading to both performance plateaus and diversity collapse. We propose TEMPO, a TTT framework that interleaves policy refinement on unlabeled questions with periodic critic recalibration on a labeled dataset. By formalizing this alternating procedure through the Expectation-Maximization (EM) algorithm, we reveal that prior methods can be interpreted as incomplete variants that omit the crucial recalibration step. Reintroducing this step tightens the evidence lower bound (ELBO) and enables sustained improvement. Across diverse model families (Qwen3 and OLMO3) and reasoning tasks, TEMPO improves OLMO3-7B on AIME 2024 from 33.0% to 51.1% and Qwen3-14B from 42.3% to 65.8%, while maintaining high diversity.
- **链接 / Link：** https://arxiv.org/abs/2604.19295v1
- **年份 / Year：** 2026
- **发表信息 / Publication：** arXiv 预印本 / arXiv preprint

#### 14. 元奖励语言模型：以 LLM 作为元评判者的自我改进式对齐

- **中文标题：** 元奖励语言模型：以 LLM 作为元评判者的自我改进式对齐
- **完整中文摘要：** 大型语言模型（LLM）正在许多领域迅速超越人类知识。传统上，改进这些模型依赖成本高昂的人类数据；而近期的自奖励机制（Yuan 等，2024）已经表明，LLM 可以通过评判自己的回答来改进，而不必依赖人类标注者。然而，现有方法主要聚焦于改进模型回答，而非改进其评判能力，因此迭代训练很快就会饱和。为解决这一问题，我们在自我改进过程中引入一个新的元奖励步骤：让模型评判其自身的评判结果，并使用这些反馈改进评判技能。令人惊讶的是，这种无监督方法同时提高了模型的评判能力和指令遵循能力：Llama-3-8B-Instruct 在 AlpacaEval 2 上的胜率从 22.9% 提升到 39.4%，在 Arena-Hard 上则从 20.6% 提升到 29.1%。这些结果有力地表明，在没有人类监督的情况下实现模型自我改进具有潜力。
- **English title:** Meta-Rewarding Language Models: Self-Improving Alignment with LLM-as-a-Meta-Judge
- **Original Abstract:** Large Language Models (LLMs) are rapidly surpassing human knowledge in many domains. While improving these models traditionally relies on costly human data, recent self-rewarding mechanisms (Yuan et al., 2024) have shown that LLMs can improve by judging their own responses instead of relying on human labelers. However, existing methods have primarily focused on improving model responses rather than judgment capabilities, resulting in rapid saturation during iterative training. To address this issue, we introduce a novel Meta-Rewarding step to the self-improvement process, where the model judges its own judgements and uses that feedback to refine its judgment skills. Surprisingly, this unsupervised approach improves the model's ability to judge and follow instructions, as demonstrated by a win rate improvement of Llama-3-8B-Instruct from 22.9% to 39.4% on AlpacaEval 2, and 20.6% to 29.1% on Arena-Hard. These results strongly suggest the potential for self-improving models without human supervision.
- **链接 / Link：** https://arxiv.org/abs/2407.19594v2
- **年份 / Year：** 2025
- **发表信息 / Publication：** 2025 年自然语言处理经验方法会议（EMNLP 2025）/ Conference on Empirical Methods in Natural Language Processing (EMNLP 2025)

#### 15. 自适应语言模型

- **中文标题：** 自适应语言模型
- **完整中文摘要：** 大型语言模型（LLM）能力强大但本质上是静态的；它们缺少根据新任务、新知识或新示例调整自身权重的机制。我们提出自适应 LLM（Self-Adapting LLMs，SEAL），这是一个使 LLM 能够通过生成自己的微调数据和更新指令实现自适应的框架。给定新输入后，模型会生成一次“自我编辑”，这种生成内容可以用不同方式重组信息、指定优化超参数，或调用工具进行数据增强和基于梯度的更新。通过监督微调（SFT），这些自我编辑会产生持久的权重更新，从而实现长期适应。为了训练模型生成有效的自我编辑，我们采用强化学习循环，并以更新后模型的下游性能作为奖励信号。不同于依赖独立适应模块或辅助网络的先前方法，SEAL 直接使用模型自身的生成内容来控制其适应过程。在知识吸收和少样本泛化上的实验表明，SEAL 是迈向能够根据新数据进行自主适应的语言模型的一项有前景的探索。项目网站和代码发布于 https://jyopari.github.io/posts/seal。
- **English title:** Self-Adapting Language Models
- **Original Abstract:** Large language models (LLMs) are powerful but static; they lack mechanisms to adapt their weights in response to new tasks, knowledge, or examples. We introduce Self-Adapting LLMs (SEAL), a framework that enables LLMs to self-adapt by generating their own finetuning data and update directives. Given a new input, the model produces a self-edit-a generation that may restructure the information in different ways, specify optimization hyperparameters, or invoke tools for data augmentation and gradient-based updates. Through supervised finetuning (SFT), these self-edits result in persistent weight updates, enabling lasting adaptation. To train the model to produce effective self-edits, we use a reinforcement learning loop with the downstream performance of the updated model as the reward signal. Unlike prior approaches that rely on separate adaptation modules or auxiliary networks, SEAL directly uses the model's own generation to control its adaptation process. Experiments on knowledge incorporation and few-shot generalization show that SEAL is a promising step toward language models capable of self-directed adaptation. Our website and code is available at https://jyopari.github.io/posts/seal.
- **链接 / Link：** https://arxiv.org/abs/2506.10943v2
- **年份 / Year：** 2025
- **发表信息 / Publication：** 第 39 届神经信息处理系统大会（NeurIPS 2025）/ Advances in Neural Information Processing Systems 39 (NeurIPS 2025)

#### 16. 语言模型的自我改进：锐化机制

- **中文标题：** 语言模型的自我改进：锐化机制
- **完整中文摘要：** 语言建模领域近期的工作提出了自我改进的可能性，即语言模型在没有外部反馈的情况下评估并改进自身生成内容，以取得更高性能。自我改进不可能创造模型中原本不存在的信息，那么我们为什么应该期待它能带来能力提升？我们通过一个称为“锐化”的视角，对自我改进的能力给出新的解释。我们的出发点是这样一个观察：语言模型通常更擅长验证回答质量，而不是生成正确回答。基于此，我们将自我改进形式化为在后训练过程中使用模型自身作为验证器，从而将模型“锐化”为一个把较大概率质量分配给高质量序列的模型，并以此摊销生成优质序列所需的高昂推理时计算。我们首先提出一个新的锐化统计框架，其中学习者旨在通过样本访问锐化预训练基础策略，并建立其基本极限。随后，我们分析两类分别基于 SFT 和 RLHF 的自然自我改进算法。我们发现：（i）只要初始模型具有充分的覆盖范围，基于 SFT 的方法就是极小极大最优的；但（ii）基于 RLHF 的方法可以利用在线探索绕过覆盖要求，从而优于基于 SFT 的自我改进。最后，我们通过推理时实验和摊销实验对锐化机制进行实证验证。我们认为，这些发现为建立关于自我改进的基础性理解提供了起点，可用于指导自我改进算法的设计与评估。
- **English title:** Self-Improvement in Language Models: The Sharpening Mechanism
- **Original Abstract:** Recent work in language modeling has raised the possibility of self-improvement, where a language models evaluates and refines its own generations to achieve higher performance without external feedback. It is impossible for this self-improvement to create information that is not already in the model, so why should we expect that this will lead to improved capabilities? We offer a new perspective on the capabilities of self-improvement through a lens we refer to as sharpening. Motivated by the observation that language models are often better at verifying response quality than they are at generating correct responses, we formalize self-improvement as using the model itself as a verifier during post-training in order to “sharpen” the model to one placing large mass on high-quality sequences, thereby amortizing the expensive inference-time computation of generating good sequences. We begin by introducing a new statistical framework for sharpening in which the learner aims to sharpen a pre-trained base policy via sample access, and establish fundamental limits. Then we analyze two natural families of self-improvement algorithms based on SFT and RLHF. We find that (i) the SFT-based approach is minimax optimal whenever the initial model has sufficient coverage, but (ii) the RLHF-based approach can improve over SFT-based self-improvement by leveraging online exploration, bypassing the need for coverage. Finally, we empirically validate the sharpening mechanism via inference-time and amortization experiments. We view these findings as a starting point toward a foundational understanding that can guide the design and evaluation of self-improvement algorithms.
- **链接 / Link：** https://arxiv.org/abs/2412.01951v2
- **年份 / Year：** 2025
- **发表信息 / Publication：** 第十三届国际学习表征会议（ICLR 2025）/ The Thirteenth International Conference on Learning Representations (ICLR 2025)

#### 17. 用于语言模型对齐的自博弈偏好优化

- **中文标题：** 用于语言模型对齐的自博弈偏好优化
- **完整中文摘要：** 依赖 Bradley-Terry 等参数模型的标准基于人类反馈的强化学习（RLHF）方法，无法充分捕捉人类偏好中的非传递性和非理性。近期进展表明，直接处理偏好概率可以更准确地反映人类偏好，从而实现更灵活、更准确的语言模型对齐。在本文中，我们提出一种基于自博弈的语言模型对齐方法，将问题视为一个旨在识别纳什均衡策略的常和双人博弈。我们的方法称为自博弈偏好优化（Self-Play Preference Optimization，SPPO），它通过迭代策略更新，以具有理论保证的方式逼近纳什均衡。此外，我们提出一个新的 SPPO 目标，该目标既有强有力的理论依据，又简单且在实践中有效。在实验中，我们仅使用 UltraFeedback 数据集中的 6 万条提示（不含回答），不进行任何提示增强，并利用一个仅有 4 亿参数的预训练偏好模型 PairRM；SPPO 通过微调 Mistral-7B-Instruct-v0.2 得到的模型，在 AlpacaEval 2.0 上对抗 GPT-4-Turbo 时取得 28.53% 的最先进长度控制胜率。它在 MT-Bench、Arena-Hard 和 Open LLM Leaderboard 上也优于（迭代式）DPO 和 IPO。从更强的基础模型 Llama-3-8B-Instruct 出发，我们能够取得 38.77% 的长度控制胜率。值得注意的是，SPPO 的强劲性能是在没有从 GPT-4 或其他更强语言模型获得额外外部监督（例如回答、偏好等）的情况下实现的。代码发布于 https://github.com/uclaml/SPPO。
- **English title:** Self-Play Preference Optimization for Language Model Alignment
- **Original Abstract:** Standard reinforcement learning from human feedback (RLHF) approaches relying on parametric models like the Bradley-Terry model fall short in capturing the intransitivity and irrationality in human preferences. Recent advancements suggest that directly working with preference probabilities can yield a more accurate reflection of human preferences, enabling more flexible and accurate language model alignment. In this paper, we propose a self-play-based method for language model alignment, which treats the problem as a constant-sum two-player game aimed at identifying the Nash equilibrium policy. Our approach, dubbed Self-Play Preference Optimization (SPPO), utilizes iterative policy updates to provably approximate the Nash equilibrium. Additionally, we propose a new SPPO objective which is both strongly motivated by theory and is simple and effective in practice. In our experiments, using only 60k prompts (without responses) from the UltraFeedback dataset and without any prompt augmentation, by leveraging a pre-trained preference model PairRM with only 0.4B parameters, SPPO can obtain a model from fine-tuning Mistral-7B-Instruct-v0.2 that achieves the state-of-the-art length-controlled win-rate of 28.53% against GPT-4-Turbo on AlpacaEval 2.0. It also outperforms the (iterative) DPO and IPO on MT-Bench, Arena-Hard, and the Open LLM Leaderboard. Starting from a stronger base model Llama-3-8B-Instruct, we are able to achieve a length-controlled win rate of 38.77%. Notably, the strong performance of SPPO is achieved without additional external supervision (e.g., responses, preferences, etc.) from GPT-4 or other stronger language models. Codes are available at https://github.com/uclaml/SPPO.
- **链接 / Link：** https://arxiv.org/abs/2405.00675v5
- **年份 / Year：** 2025
- **发表信息 / Publication：** 第十三届国际学习表征会议（ICLR 2025）/ The Thirteenth International Conference on Learning Representations (ICLR 2025)

#### 18. SELF：利用语言反馈实现自我演化

- **中文标题：** SELF：利用语言反馈实现自我演化
- **完整中文摘要：** 大型语言模型（LLM）在多个领域展现出非凡的通用性。为进一步推动 LLM 发展，我们提出“SELF”（Self-Evolution with Language Feedback，利用语言反馈的自我演化），这是一种使 LLM 能够像人类学习一样，通过自我反思实现自我改进的新方法。SELF 首先执行元技能学习过程，使 LLM 具备自我反馈和自我改进能力。随后，模型进入迭代式自我演化过程。在每轮迭代中，它使用无标注指令数据集生成初始回答，再通过自我反馈和自我改进增强这些回答。接下来，使用增强后的数据对模型进行微调。模型通过这一迭代式自我演化过程逐步改进。此外，SELF 框架使模型能够在推理期间应用自我改进，进一步提升回答质量。我们在数学任务和通用任务上的实验表明，SELF 可以在没有人类干预的情况下增强 LLM 的能力。SELF 框架为 LLM 的自主演化指出了一个有前景的方向，使其从被动的信息接收者转变为自身发展过程的主动参与者。
- **English title:** SELF: Self-Evolution with Language Feedback
- **Original Abstract:** Large Language Models (LLMs) have demonstrated remarkable versatility across various domains. To further advance LLMs, we propose 'SELF' (Self-Evolution with Language Feedback), a novel approach that enables LLMs to self-improve through self-reflection, akin to human learning processes. SELF initiates with a meta-skill learning process that equips the LLMs with capabilities for self-feedback and self-refinement. Subsequently, the model undergoes an iterative process of self-evolution. In each iteration, it utilizes an unlabeled dataset of instructions to generate initial responses. These responses are enhanced through self-feedback and self-refinement. The model is then fine-tuned using this enhanced data. The model undergoes progressive improvement through this iterative self-evolution process. Moreover, the SELF framework enables the model to apply self-refinement during inference, which further improves response quality. Our experiments in mathematics and general tasks demonstrate that SELF can enhance the capabilities of LLMs without human intervention. The SELF framework indicates a promising direction for the autonomous evolution of LLMs, transitioning them from passive information receivers to active participants in their development.
- **链接 / Link：** https://arxiv.org/abs/2310.00533v4
- **年份 / Year：** 2023
- **发表信息 / Publication：** arXiv 预印本 / arXiv preprint

### 自授式推理 / Self-Taught Reasoning

#### 19. rStar-Math：小型 LLM 可通过自演化深度思考掌握数学推理

- **中文标题：** rStar-Math：小型 LLM 可通过自演化深度思考掌握数学推理
- **完整中文摘要：** 我们提出 rStar-Math，以证明在不从更强模型进行蒸馏的情况下，小型语言模型（SLM）的数学推理能力可以媲美甚至超过 OpenAI o1。rStar-Math 通过蒙特卡洛树搜索（MCTS）开展“深度思考”：一个数学策略 SLM 在基于 SLM 的过程奖励模型引导下进行测试时搜索。为解决训练这两个 SLM 时面临的挑战，rStar-Math 引入三项创新：（1）一种新的代码增强思维链数据合成方法，通过执行大规模 MCTS rollout 生成经逐步验证的推理轨迹，用于训练策略 SLM；（2）一种新的过程奖励模型训练方法，它避免朴素的步骤级分数标注，从而得到更有效的过程偏好模型（PPM）；（3）一种自我演化方案，其中策略 SLM 和 PPM 从零开始构建并通过迭代演化提升推理能力。经过四轮自我演化，并利用针对 74.7 万道数学题合成的数百万个解答，rStar-Math 将 SLM 的数学推理能力提升到最先进水平。在 MATH 基准上，它将 Qwen2.5-Math-7B 从 58.8% 提升到 90.0%，将 Phi3-mini-3.8B 从 41.4% 提升到 86.4%，分别比 o1-preview 高 4.5% 和 0.9%。在美国数学邀请赛（AIME）上，rStar-Math 平均解出 53.3%（8/15）的题目，达到最优秀高中数学学生前 20% 的水平。代码和数据将发布于 https://github.com/microsoft/rStar。
- **English title:** rStar-Math: Small LLMs Can Master Math Reasoning with Self-Evolved Deep Thinking
- **Original Abstract:** We present rStar-Math to demonstrate that small language models (SLMs) can rival or even surpass the math reasoning capability of OpenAI o1, without distillation from superior models. rStar-Math achieves this by exercising "deep thinking" through Monte Carlo Tree Search (MCTS), where a math policy SLM performs test-time search guided by an SLM-based process reward model. rStar-Math introduces three innovations to tackle the challenges in training the two SLMs: (1) a novel code-augmented CoT data sythesis method, which performs extensive MCTS rollouts to generate step-by-step verified reasoning trajectories used to train the policy SLM; (2) a novel process reward model training method that avoids naïve step-level score annotation, yielding a more effective process preference model (PPM); (3) a self-evolution recipe in which the policy SLM and PPM are built from scratch and iteratively evolved to improve reasoning capabilities. Through 4 rounds of self-evolution with millions of synthesized solutions for 747k math problems, rStar-Math boosts SLMs' math reasoning to state-of-the-art levels. On the MATH benchmark, it improves Qwen2.5-Math-7B from 58.8% to 90.0% and Phi3-mini-3.8B from 41.4% to 86.4%, surpassing o1-preview by +4.5% and +0.9%. On the USA Math Olympiad (AIME), rStar-Math solves an average of 53.3% (8/15) of problems, ranking among the top 20% the brightest high school math students. Code and data will be available at https://github.com/microsoft/rStar.
- **链接 / Link：** https://arxiv.org/abs/2501.04519v1
- **年份 / Year：** 2025
- **发表信息 / Publication：** 第 42 届国际机器学习大会（ICML 2025）/ Proceedings of the 42nd International Conference on Machine Learning (ICML 2025)

#### 20. Quiet-STaR：语言模型能够自学先思考、后表达

- **中文标题：** Quiet-STaR：语言模型能够自学先思考、后表达
- **完整中文摘要：** 人们在写作和交谈时有时会停下来思考。尽管以推理为中心的研究常把推理理解为回答问题或完成智能体任务的方法，但几乎所有书面文本都隐含着推理。例如，证明中字里行间未明确写出的步骤，以及对话背后的心智理论，都属于这种情况。在自授式推理器（Self-Taught Reasoner，STaR；Zelikman 等，2022）中，模型通过从问答任务的少样本示例中推断推理过程，并学习那些能够导向正确答案的推理过程，来掌握有用的思考方式。这是一个高度受限的场景；理想情况下，语言模型应当能够学习推断任意文本中未明确表达的推理过程。我们提出 Quiet-STaR，它是 STaR 的一种推广，使语言模型能够在每个词元处生成推理过程，用以解释未来文本并改进预测。我们处理了若干关键挑战，包括：（1）生成续写的计算成本；（2）语言模型起初并不知道如何生成或使用内部思考；（3）需要预测的范围超越单个后续词元。为解决这些问题，我们提出逐词元并行采样算法，使用可学习词元标记思考的开始和结束，并采用一种扩展的教师强制技术。令人鼓舞的是，生成的推理过程对难以预测的词元帮助尤为明显，并提升了语言模型直接回答困难问题的能力。具体而言，在使用 Quiet-STaR 对语言模型进行互联网文本语料上的持续预训练后，我们观察到 GSM8K 上的零样本表现从 5.9% 提升到 10.9%，CommonsenseQA 上从 36.3% 提升到 47.2%，同时自然文本中困难词元的困惑度得到改善。关键在于，这些提升不需要在上述任务上进行任何微调。Quiet-STaR 朝着让语言模型以更通用、更可扩展的方式学习推理迈出了一步。
- **English title:** Quiet-STaR: Language Models Can Teach Themselves to Think Before Speaking
- **Original Abstract:** When writing and talking, people sometimes pause to think. Although reasoning-focused works have often framed reasoning as a method of answering questions or completing agentic tasks, reasoning is implicit in almost all written text. For example, this applies to the steps not stated between the lines of a proof or to the theory of mind underlying a conversation. In the Self-Taught Reasoner (STaR, Zelikman et al. 2022), useful thinking is learned by inferring rationales from few-shot examples in question-answering and learning from those that lead to a correct answer. This is a highly constrained setting -- ideally, a language model could instead learn to infer unstated rationales in arbitrary text. We present Quiet-STaR, a generalization of STaR in which LMs learn to generate rationales at each token to explain future text, improving their predictions. We address key challenges, including 1) the computational cost of generating continuations, 2) the fact that the LM does not initially know how to generate or use internal thoughts, and 3) the need to predict beyond individual next tokens. To resolve these, we propose a tokenwise parallel sampling algorithm, using learnable tokens indicating a thought's start and end, and an extended teacher-forcing technique. Encouragingly, generated rationales disproportionately help model difficult-to-predict tokens and improve the LM's ability to directly answer difficult questions. In particular, after continued pretraining of an LM on a corpus of internet text with Quiet-STaR, we find zero-shot improvements on GSM8K (5.9%→10.9%) and CommonsenseQA (36.3%→47.2%) and observe a perplexity improvement of difficult tokens in natural text. Crucially, these improvements require no fine-tuning on these tasks. Quiet-STaR marks a step towards LMs that can learn to reason in a more general and scalable way.
- **链接 / Link：** https://arxiv.org/abs/2403.09629v2
- **年份 / Year：** 2024
- **发表信息 / Publication：** 第一届语言建模大会（COLM 2024）/ First Conference on Language Modeling (COLM 2024)

#### 21. STaR：用推理自举推理能力

- **中文标题：** STaR：用推理自举推理能力
- **完整中文摘要：** 生成逐步的“思维链”推理过程，可以提升语言模型在数学或常识问答等复杂推理任务上的表现。然而，要诱导语言模型生成推理过程，目前要么需要构建规模庞大的推理过程数据集，要么只能使用少样本推理并牺牲准确率。我们提出一种技术，通过迭代利用少量推理过程示例和一个不包含推理过程的大型数据集，自举出执行逐步复杂推理的能力。该技术称为“自授式推理器”（Self-Taught Reasoner，STaR），其基础是一个简单循环：在少量推理过程示例的提示下，为许多问题生成推理过程；如果生成的答案错误，则在给定正确答案的情况下再次尝试生成推理过程；使用所有最终导向正确答案的推理过程进行微调；重复上述步骤。我们证明，与微调后直接预测最终答案的模型相比，STaR 在多个数据集上显著提升了性能；在 CommonsenseQA 上，其表现与一个参数规模大 30 倍的最先进语言模型经微调后的表现相当。因此，STaR 使模型能够通过学习自身生成的推理过程来改进自己。
- **English title:** STaR: Bootstrapping Reasoning With Reasoning
- **Original Abstract:** Generating step-by-step "chain-of-thought" rationales improves language model performance on complex reasoning tasks like mathematics or commonsense question-answering. However, inducing language model rationale generation currently requires either constructing massive rationale datasets or sacrificing accuracy by using only few-shot inference. We propose a technique to iteratively leverage a small number of rationale examples and a large dataset without rationales, to bootstrap the ability to perform successively more complex reasoning. This technique, the "Self-Taught Reasoner" (STaR), relies on a simple loop: generate rationales to answer many questions, prompted with a few rationale examples; if the generated answers are wrong, try again to generate a rationale given the correct answer; fine-tune on all the rationales that ultimately yielded correct answers; repeat. We show that STaR significantly improves performance on multiple datasets compared to a model fine-tuned to directly predict final answers, and performs comparably to fine-tuning a 30× larger state-of-the-art language model on CommensenseQA. Thus, STaR lets a model improve itself by learning from its own generated reasoning.
- **链接 / Link：** https://arxiv.org/abs/2203.14465v2
- **年份 / Year：** 2022
- **发表信息 / Publication：** 第 36 届神经信息处理系统大会（NeurIPS 2022）/ Advances in Neural Information Processing Systems 36 (NeurIPS 2022)

## Harness 层递归自我改进 / Harness-level RSI

围绕模型改进提示词、记忆、验证、工具或智能体策略的方法。

Methods that improve prompts, memory, verification, tools, or agent policies around a model.

### 提示词与程序优化 / Prompt & Program Optimization

#### 1. [智能体系统的自动化设计](https://arxiv.org/abs/2408.08435)

**中文标题：** 智能体系统的自动化设计

**中文摘要：** 研究人员正投入大量精力开发强大的通用智能体，其中基础模型被用作智能体系统中的模块（例如思维链、自我反思和 Toolformer）。然而，机器学习的历史告诉我们，手工设计的解决方案最终会被学习得到的解决方案取代。本文描述了一个正在形成的研究领域——智能体系统的自动化设计（Automated Design of Agentic Systems，ADAS），其目标是自动创建强大的智能体系统设计，包括发明新的构建模块，或以新的方式组合已有模块。本文进一步展示了 ADAS 中一种尚未得到充分探索但前景可观的方法：用代码定义智能体，再由元智能体通过编程自动发现越来越好的新智能体。鉴于编程语言具有图灵完备性，从理论上说，这种方法能够学习任何可能的智能体系统，包括新的提示词、工具使用方式、工作流及其组合。为说明这一思路，本文提出了一个简单而有效的算法 Meta Agent Search；元智能体根据不断扩充的既有发现档案，迭代编写有意义的新智能体程序。在编程、科学和数学等多个领域的大量实验表明，该算法能够逐步发明具有新设计的智能体，并大幅超过当前最先进的手工设计智能体。值得注意的是，实验持续观察到一个出人意料的结果：Meta Agent Search 发明的智能体即使迁移到不同领域和模型，仍然保持更好的性能，说明其具有稳健性和通用性。如果能够安全地开发这类技术，本文所展示的新研究方向有望推动智能体系统实现自动化设计并不断增强能力。所有代码已在 https://github.com/ShengranHu/ADAS 开源。

**English title:** [Automated Design of Agentic Systems](https://arxiv.org/abs/2408.08435)

**Abstract:** Researchers are investing substantial effort in developing powerful general-purpose agents, wherein Foundation Models are used as modules within agentic systems (e.g. Chain-of-Thought, Self-Reflection, Toolformer). However, the history of machine learning teaches us that hand-designed solutions are eventually replaced by learned solutions. We describe a newly forming research area, Automated Design of Agentic Systems (ADAS), which aims to automatically create powerful agentic system designs, including inventing novel building blocks and/or combining them in new ways. We further demonstrate that there is an unexplored yet promising approach within ADAS where agents can be defined in code and new agents can be automatically discovered by a meta agent programming ever better ones in code. Given that programming languages are Turing Complete, this approach theoretically enables the learning of any possible agentic system: including novel prompts, tool use, workflows, and combinations thereof. We present a simple yet effective algorithm named Meta Agent Search to demonstrate this idea, where a meta agent iteratively programs interesting new agents based on an ever-growing archive of previous discoveries. Through extensive experiments across multiple domains including coding, science, and math, we show that our algorithm can progressively invent agents with novel designs that greatly outperform state-of-the-art hand-designed agents. Importantly, we consistently observe the surprising result that agents invented by Meta Agent Search maintain superior performance even when transferred across domains and models, demonstrating their robustness and generality. Provided we develop it safely, our work illustrates the potential of an exciting new research direction toward automatically designing ever-more powerful agentic systems to benefit humanity. All code is open-sourced at https://github.com/ShengranHu/ADAS.

**年份与发表信息：** 2024；ICLR 2025。

#### 2. [TextGrad：通过文本实现自动“微分”](https://arxiv.org/abs/2406.07496)

**中文标题：** TextGrad：通过文本实现自动“微分”

**中文摘要：** 人工智能正在经历范式转变：新的突破来自对多个大语言模型（LLM）及其他复杂组件的协同编排。因此，如何为复合人工智能系统开发有原则且自动化的优化方法，已经成为最重要的新挑战之一。神经网络早期也曾面临类似挑战，直到反向传播和自动微分使优化成为开箱即用的过程，才改变了整个领域。受此启发，本文提出 TextGrad，一个通过文本执行自动“微分”的强大框架。TextGrad 将 LLM 提供的文本反馈反向传播，以改进复合人工智能系统中的各个组件。在该框架中，LLM 给出信息丰富、通用的自然语言建议，用来优化计算图中的变量；这些变量可以是代码片段，也可以是分子结构。TextGrad 沿用 PyTorch 的语法和抽象，灵活且易于使用。它可以直接用于多种任务，用户只需提供目标函数，无须调整框架的组件或提示词。本文在问答、分子优化和放射治疗计划等多类应用上验证了 TextGrad 的有效性与通用性。在不修改框架的情况下，TextGrad 将 GPT-4o 在 Google-Proof Question Answering 上的零样本准确率从 51% 提高到 55%，使 LeetCode-Hard 编程题解优化取得 20% 的相对性能增益，还能改进推理提示词、设计具有理想计算机模拟结合特性的类药小分子，并设计具有高特异性的放射肿瘤治疗方案。TextGrad 为加速下一代人工智能系统的开发奠定了基础。

**English title:** [TextGrad: Automatic "Differentiation" via Text](https://arxiv.org/abs/2406.07496)

**Abstract:** AI is undergoing a paradigm shift, with breakthroughs achieved by systems orchestrating multiple large language models (LLMs) and other complex components. As a result, developing principled and automated optimization methods for compound AI systems is one of the most important new challenges. Neural networks faced a similar challenge in its early days until backpropagation and automatic differentiation transformed the field by making optimization turn-key. Inspired by this, we introduce TextGrad, a powerful framework performing automatic ``differentiation'' via text. TextGrad backpropagates textual feedback provided by LLMs to improve individual components of a compound AI system. In our framework, LLMs provide rich, general, natural language suggestions to optimize variables in computation graphs, ranging from code snippets to molecular structures. TextGrad follows PyTorch's syntax and abstraction and is flexible and easy-to-use. It works out-of-the-box for a variety of tasks, where the users only provide the objective function without tuning components or prompts of the framework. We showcase TextGrad's effectiveness and generality across a diverse range of applications, from question answering and molecule optimization to radiotherapy treatment planning. Without modifying the framework, TextGrad improves the zero-shot accuracy of GPT-4o in Google-Proof Question Answering from $51\%$ to $55\%$, yields $20\%$ relative performance gain in optimizing LeetCode-Hard coding problem solutions, improves prompts for reasoning, designs new druglike small molecules with desirable in silico binding, and designs radiation oncology treatment plans with high specificity. TextGrad lays a foundation to accelerate the development of the next-generation of AI systems.

**年份与发表信息：** 2024；Nature 2025。

#### 3. [语言智能体树搜索统一语言模型中的推理、行动与规划](https://arxiv.org/abs/2310.04406)

**中文标题：** 语言智能体树搜索统一语言模型中的推理、行动与规划

**中文摘要：** 尽管语言模型（LM）已在多种决策任务中展现潜力，但它们依赖简单的行动过程，这限制了其作为自主智能体的广泛部署。本文提出语言智能体树搜索（Language Agent Tree Search，LATS），这是首个协同整合语言模型推理、行动和规划能力的通用框架。LATS 利用语言模型的上下文学习能力，将蒙特卡洛树搜索引入语言模型智能体，并结合由语言模型驱动的价值函数和自我反思，以实现高效探索和更好的决策。该方法的一个重要特点是引入环境来提供外部反馈，由此形成一种更审慎、更具适应性的问题求解机制，突破现有技术的限制。本文在编程、交互式问答、网页导航和数学等多个领域开展实验评估，结果证明 LATS 在决策任务中有效且通用，同时保持了有竞争力或更好的推理性能。尤其是在使用 GPT-4 时，LATS 在 HumanEval 编程基准上取得 92.7% 的最新最优 pass@1 准确率；在使用 GPT-3.5 进行 WebShop 网页导航时，它以无梯度方式取得 75.9 的平均分，与基于梯度的微调性能相当。代码见 https://github.com/lapisrocks/LanguageAgentTreeSearch。

**English title:** [Language Agent Tree Search Unifies Reasoning Acting and Planning in Language Models](https://arxiv.org/abs/2310.04406)

**Abstract:** While language models (LMs) have shown potential across a range of decision-making tasks, their reliance on simple acting processes limits their broad deployment as autonomous agents. In this paper, we introduce Language Agent Tree Search (LATS) -- the first general framework that synergizes the capabilities of LMs in reasoning, acting, and planning. By leveraging the in-context learning ability of LMs, we integrate Monte Carlo Tree Search into LATS to enable LMs as agents, along with LM-powered value functions and self-reflections for proficient exploration and enhanced decision-making. A key feature of our approach is the incorporation of an environment for external feedback, which offers a more deliberate and adaptive problem-solving mechanism that surpasses the constraints of existing techniques. Our experimental evaluation across diverse domains, including programming, interactive question-answering (QA), web navigation, and math, validates the effectiveness and generality of LATS in decision-making while maintaining competitive or improved reasoning performance. Notably, LATS achieves state-of-the-art pass@1 accuracy (92.7%) for programming on HumanEval with GPT-4 and demonstrates gradient-free performance (average score of 75.9) comparable to gradient-based fine-tuning for web navigation on WebShop with GPT-3.5. Code can be found at https://github.com/lapisrocks/LanguageAgentTreeSearch

**年份与发表信息：** 2023；ICML 2024。

#### 4. [作为优化器的大语言模型](https://arxiv.org/abs/2309.03409)

**中文标题：** 作为优化器的大语言模型

**中文摘要：** 优化无处不在。基于导数的算法一直是解决各类问题的有力工具，但在没有梯度的情况下，许多现实应用仍面临困难。本文提出 Optimization by PROmpting（OPRO），一种让大语言模型（LLM）充当优化器的简单有效方法，其中优化任务以自然语言描述。在每一步优化中，LLM 根据包含先前解及其数值的提示词生成新解；随后对新解进行评估，并将它们加入下一步优化所用的提示词。本文先在线性回归和旅行商问题上展示 OPRO，再将重点转向提示词优化，其目标是找出能让任务准确率最大化的指令。使用多种 LLM 的实验表明，OPRO 优化得到的最佳提示词在 GSM8K 上最多比人工设计的提示词高 8%，在 Big-Bench Hard 任务上最多高 50%。代码见 https://github.com/google-deepmind/opro。

**English title:** [Large Language Models as Optimizers](https://arxiv.org/abs/2309.03409)

**Abstract:** Optimization is ubiquitous. While derivative-based algorithms have been powerful tools for various problems, the absence of gradient imposes challenges on many real-world applications. In this work, we propose Optimization by PROmpting (OPRO), a simple and effective approach to leverage large language models (LLMs) as optimizers, where the optimization task is described in natural language. In each optimization step, the LLM generates new solutions from the prompt that contains previously generated solutions with their values, then the new solutions are evaluated and added to the prompt for the next optimization step. We first showcase OPRO on linear regression and traveling salesman problems, then move on to our main application in prompt optimization, where the goal is to find instructions that maximize the task accuracy. With a variety of LLMs, we demonstrate that the best prompts optimized by OPRO outperform human-designed prompts by up to 8% on GSM8K, and by up to 50% on Big-Bench Hard tasks. Code at https://github.com/google-deepmind/opro.

**年份与发表信息：** 2023；ICLR 2024。

#### 5. [Promptbreeder：通过提示词进化实现自指式自我改进](https://arxiv.org/abs/2309.16797)

**中文标题：** Promptbreeder：通过提示词进化实现自指式自我改进

**中文摘要：** 思维链提示等流行的提示策略可以显著提高大语言模型（LLM）在不同领域中的推理能力，但这类手工制作的提示策略往往不是最优的。本文提出 Promptbreeder，一种面向通用任务的自指式自我改进机制，可针对给定领域进化并调整提示词。在 LLM 的驱动下，Promptbreeder 对任务提示词种群进行变异，随后在训练集上评估其适应度。关键在于，这些任务提示词的变异由“变异提示词”控制，而变异提示词同样由 LLM 生成，并在进化过程中以自指方式得到改进。换言之，Promptbreeder 不只改进任务提示词，也会改进用于改进任务提示词的变异提示词。在常用的算术和常识推理基准上，Promptbreeder 超过了思维链提示和 Plan-and-Solve Prompting 等当前最先进的提示策略。Promptbreeder 还能为具有挑战性的仇恨言论分类问题进化出结构精细的任务提示词。

**English title:** [Promptbreeder: Self-Referential Self-Improvement Via Prompt Evolution](https://arxiv.org/abs/2309.16797)

**Abstract:** Popular prompt strategies like Chain-of-Thought Prompting can dramatically improve the reasoning abilities of Large Language Models (LLMs) in various domains. However, such hand-crafted prompt-strategies are often sub-optimal. In this paper, we present Promptbreeder, a general-purpose self-referential self-improvement mechanism that evolves and adapts prompts for a given domain. Driven by an LLM, Promptbreeder mutates a population of task-prompts, and subsequently evaluates them for fitness on a training set. Crucially, the mutation of these task-prompts is governed by mutation-prompts that the LLM generates and improves throughout evolution in a self-referential way. That is, Promptbreeder is not just improving task-prompts, but it is also improving the mutationprompts that improve these task-prompts. Promptbreeder outperforms state-of-the-art prompt strategies such as Chain-of-Thought and Plan-and-Solve Prompting on commonly used arithmetic and commonsense reasoning benchmarks. Furthermore, Promptbreeder is able to evolve intricate task-prompts for the challenging problem of hate speech classification.

**年份与发表信息：** 2023；ICML 2024。

#### 6. [自学优化器（STOP）：递归自我改进的代码生成](https://arxiv.org/abs/2310.02304)

**中文标题：** 自学优化器（STOP）：递归自我改进的代码生成

**中文摘要：** 人工智能系统近期取得的若干进展，是通过提供“脚手架”程序来解决问题：这类程序组织对语言模型（LM）的多次调用，以生成更好的输出。脚手架程序用 Python 等编程语言编写。本文使用一个融入语言模型的脚手架程序来改进它自身。研究从一个种子“改进器”开始；该改进器多次查询语言模型，依据给定的效用函数改进输入程序，并返回最佳解。随后，研究者运行这个种子改进器，让它改进自己。在一小组下游任务上，改进后的改进器所生成程序的性能显著优于种子改进器。语言模型提出了多种自我改进策略，包括束搜索、遗传算法和模拟退火。由于语言模型本身并未改变，这还不属于完整的递归自我改进；但实验表明，现代语言模型（本文实验使用 GPT-4）能够编写调用自身来改进自身的代码。本文还讨论了开发自我改进技术所带来的担忧，并评估生成代码绕过沙箱的频率。

**English title:** [Self-Taught Optimizer (STOP): Recursively Self-Improving Code Generation](https://arxiv.org/abs/2310.02304)

**Abstract:** Several recent advances in AI systems solve problems by providing a "scaffolding" program that structures multiple calls to language models (LMs) to generate better outputs. A scaffolding program is written in a programming language such as Python. In this work, we use a language-model-infused scaffolding program to improve itself. We start with a seed "improver" that improves an input program according to a given utility function by querying an LM several times and returning the best solution. We then run this seed improver to improve itself. Across a small set of downstream tasks, the resulting improved improver generates programs with significantly better performance than its seed improver. A variety of self-improvement strategies are proposed by the language model, including beam search, genetic algorithms, and simulated annealing. Since the language models themselves are not altered, this is not full recursive self-improvement. Nonetheless, it demonstrates that a modern language model, GPT-4 in our experiments, is capable of writing code that can call itself to improve itself. We consider concerns around the development of self-improving technologies and evaluate the frequency with which the generated code bypasses a sandbox.

**年份与发表信息：** 2023；COLM 2024。

#### 7. [DSPy：将声明式语言模型调用编译为自我改进流水线](https://arxiv.org/abs/2310.03714)

**中文标题：** DSPy：将声明式语言模型调用编译为自我改进流水线

**中文摘要：** 机器学习社区正在迅速探索语言模型（LM）提示技术，以及将多个模型调用串联成流水线来解决复杂任务的方法。遗憾的是，现有 LM 流水线通常使用硬编码的“提示词模板”，也就是通过反复试错找到的冗长字符串。为了以更系统的方式开发和优化 LM 流水线，本文提出 DSPy：一种将 LM 流水线抽象为文本转换图的编程模型。这些文本转换图是命令式计算图，其中 LM 通过声明式模块被调用。DSPy 模块带有参数，因此它们可以通过创建和收集示例，学习如何运用提示、微调、增强和推理技术的组合。本文设计了一个编译器，可优化任意 DSPy 流水线，使给定指标达到最大值。两个案例研究表明，简洁的 DSPy 程序能够表达并优化复杂的 LM 流水线，用于推理数学文字题、执行多跳检索、回答复杂问题和控制智能体循环。只需几分钟编译，几行 DSPy 代码就能让 GPT-3.5 和 llama2-13b-chat 自举出流水线：相较标准少样本提示，二者的性能通常分别提高 25% 以上和 65% 以上；相较使用专家构造示例的流水线，性能分别最多提高 5%–46% 和 16%–40%。此外，编译到 7.7 亿参数 T5 和 llama2-13b-chat 等开放且相对较小模型上的 DSPy 程序，也能与依赖专家编写提示链和闭源 GPT-3.5 的方法竞争。DSPy 已在 https://github.com/stanfordnlp/dspy 发布。

**English title:** [DSPy: Compiling Declarative Language Model Calls into Self-Improving Pipelines](https://arxiv.org/abs/2310.03714)

**Abstract:** The ML community is rapidly exploring techniques for prompting language models (LMs) and for stacking them into pipelines that solve complex tasks. Unfortunately, existing LM pipelines are typically implemented using hard-coded "prompt templates", i.e. lengthy strings discovered via trial and error. Toward a more systematic approach for developing and optimizing LM pipelines, we introduce DSPy, a programming model that abstracts LM pipelines as text transformation graphs, i.e. imperative computational graphs where LMs are invoked through declarative modules. DSPy modules are parameterized, meaning they can learn (by creating and collecting demonstrations) how to apply compositions of prompting, finetuning, augmentation, and reasoning techniques. We design a compiler that will optimize any DSPy pipeline to maximize a given metric. We conduct two case studies, showing that succinct DSPy programs can express and optimize sophisticated LM pipelines that reason about math word problems, tackle multi-hop retrieval, answer complex questions, and control agent loops. Within minutes of compiling, a few lines of DSPy allow GPT-3.5 and llama2-13b-chat to self-bootstrap pipelines that outperform standard few-shot prompting (generally by over 25% and 65%, respectively) and pipelines with expert-created demonstrations (by up to 5-46% and 16-40%, respectively). On top of that, DSPy programs compiled to open and relatively small LMs like 770M-parameter T5 and llama2-13b-chat are competitive with approaches that rely on expert-written prompt chains for proprietary GPT-3.5. DSPy is available at https://github.com/stanfordnlp/dspy

**年份与发表信息：** 2023；NeurIPS 2023 R0-FoMo Workshop。

### 上下文与记忆演化 / Context & Memory Evolution

#### 8. [智能体上下文工程：为自我改进语言模型演化上下文](https://arxiv.org/abs/2510.04618)

**中文标题：** 智能体上下文工程：为自我改进语言模型演化上下文

**中文摘要：** 大语言模型（LLM）应用，如智能体和特定领域推理，越来越依赖上下文适配，即通过指令、策略或证据修改输入，而非更新权重。以往方法改善了可用性，但经常受制于简洁偏差和上下文坍缩：前者会为了简短摘要而丢弃领域洞见，后者则会在迭代重写中逐渐侵蚀细节。本文提出 ACE（Agentic Context Engineering），把上下文视为持续演化的操作手册，通过生成、反思和整理组成的模块化过程积累、完善并组织策略。ACE 采用结构化的增量更新来保留详细知识，并适配长上下文模型，从而避免上下文坍缩。在智能体与特定领域基准上，ACE 可离线优化上下文（如系统提示词），也可在线优化上下文（如智能体记忆），并持续优于有力基线：智能体任务提高 10.6%，金融任务提高 8.6%，同时显著降低适配延迟和推演成本。ACE 无须带标签的监督，而能利用自然产生的执行反馈有效适配。在 AppWorld 排行榜上，ACE 的总平均分追平排名第一的生产级智能体，并在更难的 test-challenge 划分上超过后者，尽管 ACE 使用的是更小的开源模型。这些结果说明，全面且持续演化的上下文能够以较低开销支持可扩展、高效、自我改进的 LLM 系统。

**English title:** [Agentic Context Engineering: Evolving Contexts for Self-Improving Language Models](https://arxiv.org/abs/2510.04618)

**Abstract:** Large language model (LLM) applications such as agents and domain-specific reasoning increasingly rely on context adaptation: modifying inputs with instructions, strategies, or evidence, rather than weight updates. Prior approaches improve usability but often suffer from brevity bias, which drops domain insights for concise summaries, and from context collapse, where iterative rewriting erodes details over time. We introduce ACE (Agentic Context Engineering), a framework that treats contexts as evolving playbooks that accumulate, refine, and organize strategies through a modular process of generation, reflection, and curation. ACE prevents collapse with structured, incremental updates that preserve detailed knowledge and scale with long-context models. Across agent and domain-specific benchmarks, ACE optimizes contexts both offline (e.g., system prompts) and online (e.g., agent memory), consistently outperforming strong baselines: +10.6% on agents and +8.6% on finance, while significantly reducing adaptation latency and rollout cost. Notably, ACE could adapt effectively without labeled supervision and instead by leveraging natural execution feedback. On the AppWorld leaderboard, ACE matches the top-ranked production-level agent on the overall average and surpasses it on the harder test-challenge split, despite using a smaller open-source model. These results show that comprehensive, evolving contexts enable scalable, efficient, and self-improving LLM systems with low overhead.

**年份与发表信息：** 2025；ICLR 2026。

#### 9. [EvolveR：通过经验驱动的生命周期实现大语言模型智能体自我演化](https://arxiv.org/abs/2510.16079)

**中文标题：** EvolveR：通过经验驱动的生命周期实现大语言模型智能体自我演化

**中文摘要：** 当前大语言模型（LLM）智能体在工具使用方面表现出色，却缺少系统地从自身经验中学习这一重要能力。现有框架主要着眼于弥补外部知识缺口，未能处理一个更根本的局限：智能体无法迭代完善问题求解策略。本文提出 EvolveR，一个让智能体通过完整的闭环经验生命周期实现自我改进的框架。该生命周期包括两个主要阶段：（1）离线自蒸馏，将智能体的交互轨迹综合为由抽象、可复用策略原则组成的结构化知识库；（2）在线交互，智能体与任务交互，主动检索已蒸馏的原则以指导决策，并积累多样化的行为轨迹。该闭环使用策略强化机制，根据智能体的表现对其进行迭代更新。复杂多跳问答基准上的实验表明，EvolveR 的性能优于有力的智能体基线。本文给出了一套完整方案，使智能体既能从外部数据学习，也能从自身行动的后果中学习，为自主性更强且能够持续改进的系统提供实现路径。代码见 https://github.com/Edaizi/EvolveR。

**English title:** [EvolveR: Self-Evolving LLM Agents through an Experience-Driven Lifecycle](https://arxiv.org/abs/2510.16079)

**Abstract:** Current Large Language Model (LLM) agents show strong performance in tool use, but lack the crucial capability to systematically learn from their own experiences. While existing frameworks mainly focus on mitigating external knowledge gaps, they fail to address a more fundamental limitation: the inability to iteratively refine problem-solving strategies. In this work, we introduce EvolveR, a framework designed to enable agent to self-improve through a complete, closed-loop experience lifecycle. This lifecycle comprises two key stages: (1) Offline Self-Distillation, where the agent's interaction trajectories are synthesized into a structured repository of abstract, reusable strategic principles; (2) Online Interaction, where the agent interacts with tasks and actively retrieves distilled principles to guide its decision-making, accumulating a diverse set of behavioral trajectories. This loop employs a policy reinforcement mechanism to iteratively update the agent based on its performance. We demonstrate the effectiveness of EvolveR on complex multi-hop question-answering benchmarks, where it achieves superior performance over strong agentic baselines. Our work presents a comprehensive blueprint for agents that learn not only from external data but also from the consequences of their own actions, paving the way for more autonomous and continuously improving systems. Code is available at https://github.com/Edaizi/EvolveR.

**年份与发表信息：** 2025；ICML 2026。

#### 10. [从程序性技能到策略基因：迈向经验驱动的测试时演化](https://arxiv.org/abs/2604.15097)

**中文标题：** 从程序性技能到策略基因：迈向经验驱动的测试时演化

**中文摘要：** 这份测试版技术报告研究如何表示可复用经验，使其既能作为有效的测试时控制手段，又能充当迭代演化的基础。研究在 45 个科学代码求解场景中开展了 4,590 次受控试验。结果发现，面向文档的 Skill 包提供的控制并不稳定：其中有效信号稀疏，把一个紧凑的经验对象扩写成更完整的文档包往往无益，甚至会降低总体平均性能。研究进一步表明，表示形式本身是一项一阶影响因素。紧凑的 Gene 表示取得最高的总体平均分，在大幅结构扰动下仍有竞争力，并优于预算相同的 Skill 片段；重新附加面向文档的材料通常会削弱而非改善效果。除了单次控制，Gene 还是更合适的迭代经验积累载体：在 Gene 中附加失败历史比在 Skill 或自由文本中更有效；可编辑结构的作用不只是承载内容；将失败信息提炼成简短警告，也比直接追加更有用。在 CritPt 上，采用基因演化的系统相较配对基础模型，性能分别从 9.1% 提高到 18.57%，以及从 17.7% 提高到 27.14%。结果表明，经验复用的核心问题不在于如何提供更多经验，而在于如何将经验编码为紧凑、面向控制且适合演化的对象。

**English title:** [From Procedural Skills to Strategy Genes: Towards Experience-Driven Test-Time Evolution](https://arxiv.org/abs/2604.15097)

**Abstract:** This beta technical report asks how reusable experience should be represented so that it can function as effective test-time control and as a substrate for iterative evolution. We study this question in 4.590 controlled trials across 45 scientific code-solving scenarios. We find that documentation-oriented Skill packages provide unstable control: their useful signal is sparse, and expanding a compact experience object into a fuller documentation package often fails to help and can degrade the overall average. We further show that representation itself is a first-order factor. A compact Gene representation yields the strongest overall average, remains competitive under substantial structural perturbations, and outperforms matched-budget Skill fragments, while reattaching documentation-oriented material usually weakens rather than improves it. Beyond one-shot control, we show that Gene is also a better carrier for iterative experience accumulation: attached failure history is more effective in Gene than in Skill or freeform text, editable structure matters beyond content alone, and failure information is most useful when distilled into compact warnings rather than naively appended. On CritPt, gene-evolved systems improve over their paired base models from 9.1% to 18.57% and from 17.7% to 27.14%. These results suggest that the core problem in experience reuse is not how to supply more experience, but how to encode experience as a compact, control-oriented, evolution-ready object.

**年份与发表信息：** 2026；arXiv 2026。

#### 11. [通过元学习智能体记忆设计来学习如何持续学习](https://arxiv.org/abs/2602.07755)

**中文标题：** 通过元学习智能体记忆设计来学习如何持续学习

**中文摘要：** 基础模型的无状态特性限制了智能体系统持续学习的能力，而持续学习是长时程推理与适应的一项核心能力。为解决这一问题，智能体系统通常加入记忆模块，用来保留并复用过往经验，目标是在测试时实现持续学习。然而，现有多数记忆设计由人工构造且固定不变，难以适应现实任务的多样性和非平稳性。本文提出 ALMA（Automated meta-Learning of Memory designs for Agentic systems），它通过元学习获得记忆设计，以取代人工设计，从而减少人工投入，并使智能体系统能够跨多个领域持续学习。该方法使用元智能体，以开放方式搜索由可执行代码表示的记忆设计；理论上，它可以发现任意记忆设计，包括数据库模式，以及相应的检索与更新机制。在四个序列决策领域开展的大量实验表明，与各基准上最先进的人工记忆设计相比，学习得到的记忆设计能让系统更有效、更高效地从经验中学习。如果能够安全地开发和部署，ALMA 将推动自我改进人工智能系统成为具有适应能力的持续学习者。

**English title:** [Learning to Continually Learn via Meta-learning Agentic Memory Designs](https://arxiv.org/abs/2602.07755)

**Abstract:** The statelessness of foundation models bottlenecks agentic systems' ability to continually learn, a core capability for long-horizon reasoning and adaptation. To address this limitation, agentic systems commonly incorporate memory modules to retain and reuse past experience, aiming for continual learning during test time. However, most existing memory designs are human-crafted and fixed, which limits their ability to adapt to the diversity and non-stationarity of real-world tasks. In this paper, we introduce ALMA (Automated meta-Learning of Memory designs for Agentic systems), a framework that meta-learns memory designs to replace hand-engineered memory designs, therefore minimizing human effort and enabling agentic systems to be continual learners across diverse domains. Our approach employs a Meta Agent that searches over memory designs expressed as executable code in an open-ended manner, theoretically allowing the discovery of arbitrary memory designs, including database schemas as well as their retrieval and update mechanisms. Extensive experiments across four sequential decision-making domains demonstrate that the learned memory designs enable more effective and efficient learning from experience than state-of-the-art human-crafted memory designs on all benchmarks. When developed and deployed safely, ALMA represents a step toward self-improving AI systems that learn to be adaptive, continual learners.

**年份与发表信息：** 2026；arXiv 2026。

#### 12. [A-MEM：面向大语言模型智能体的智能体式记忆](https://arxiv.org/abs/2502.12110)

**中文标题：** A-MEM：面向大语言模型智能体的智能体式记忆

**中文摘要：** 大语言模型（LLM）智能体可以有效使用外部工具来完成复杂的现实任务，但它们需要借助记忆系统利用历史经验。现有记忆系统能够进行基本的存储与检索，然而，即使近期已有研究尝试引入图数据库，它们仍缺少成熟的记忆组织机制。固定的操作和结构也限制了这些系统适应不同任务的能力。为解决这一问题，本文提出一种面向 LLM 智能体的新型智能体式记忆系统，它可以动态、自主地组织记忆。研究遵循卡片盒笔记法（Zettelkasten）的基本原则，通过动态索引和链接建立互联的知识网络。加入新记忆时，系统生成一条包含上下文描述、关键词和标签等多个结构化属性的完整笔记，再分析历史记忆，识别相关联系，并在存在有意义的相似性时建立链接。这个过程也支持记忆演化：随着新记忆被整合，它们可以触发对既有历史记忆上下文表示和属性的更新，使记忆网络不断修正其理解。该方法把卡片盒笔记法的结构化组织原则与智能体决策的灵活性结合起来，使记忆管理更具适应性并能感知上下文。在六种基础模型上的实证实验显示，该方法相较现有最新基线取得了更大的性能提升。性能评估代码见 https://github.com/WujiangXu/A-mem，智能体记忆系统的源代码见 https://github.com/WujiangXu/A-mem-sys。

**English title:** [A-MEM: Agentic Memory for LLM Agents](https://arxiv.org/abs/2502.12110)

**Abstract:** While large language model (LLM) agents can effectively use external tools for complex real-world tasks, they require memory systems to leverage historical experiences. Current memory systems enable basic storage and retrieval but lack sophisticated memory organization, despite recent attempts to incorporate graph databases. Moreover, these systems' fixed operations and structures limit their adaptability across diverse tasks. To address this limitation, this paper proposes a novel agentic memory system for LLM agents that can dynamically organize memories in an agentic way. Following the basic principles of the Zettelkasten method, we designed our memory system to create interconnected knowledge networks through dynamic indexing and linking. When a new memory is added, we generate a comprehensive note containing multiple structured attributes, including contextual descriptions, keywords, and tags. The system then analyzes historical memories to identify relevant connections, establishing links where meaningful similarities exist. Additionally, this process enables memory evolution - as new memories are integrated, they can trigger updates to the contextual representations and attributes of existing historical memories, allowing the memory network to continuously refine its understanding. Our approach combines the structured organization principles of Zettelkasten with the flexibility of agent-driven decision making, allowing for more adaptive and context-aware memory management. Empirical experiments on six foundation models show superior improvement against existing SOTA baselines. The source code for evaluating performance is available at https://github.com/WujiangXu/A-mem, while the source code of the agentic memory system is available at https://github.com/WujiangXu/A-mem-sys.

**年份与发表信息：** 2025；NeurIPS 2025。

#### 13. [ExpeL：大语言模型智能体是经验学习者](https://arxiv.org/abs/2308.10144)

**中文标题：** ExpeL：大语言模型智能体是经验学习者

**中文摘要：** 近期，将大语言模型（LLM）应用于决策任务的研究迅速增长，这些工作利用了 LLM 所包含的大量世界知识。人们越来越需要针对定制决策任务调整 LLM，但为特定任务微调模型既耗费资源，也可能削弱模型的泛化能力。与此同时，GPT-4 和 Claude 等先进语言模型主要通过 API 调用，其参数权重属于专有内容，公众无法取得。因此，需要新的方法，使模型能够从智能体经验中学习，而不必更新参数。为解决这些问题，本文提出经验学习智能体 ExpeL。该智能体从一组训练任务中自主收集经验，并使用自然语言提取知识；推理时，它会回忆提取出的洞见和以往经验，以做出有依据的决策。实证结果表明，ExpeL 具有稳定的学习效果：随着经验不断积累，其性能持续提高。本文还通过定性观察和附加实验，研究了 ExpeL 涌现出的能力和迁移学习潜力。

**English title:** [ExpeL: LLM Agents Are Experiential Learners](https://arxiv.org/abs/2308.10144)

**Abstract:** The recent surge in research interest in applying large language models (LLMs) to decision-making tasks has flourished by leveraging the extensive world knowledge embedded in LLMs. While there is a growing demand to tailor LLMs for custom decision-making tasks, finetuning them for specific tasks is resource-intensive and may diminish the model's generalization capabilities. Moreover, state-of-the-art language models like GPT-4 and Claude are primarily accessible through API calls, with their parametric weights remaining proprietary and unavailable to the public. This scenario emphasizes the growing need for new methodologies that allow learning from agent experiences without requiring parametric updates. To address these problems, we introduce the Experiential Learning (ExpeL) agent. Our agent autonomously gathers experiences and extracts knowledge using natural language from a collection of training tasks. At inference, the agent recalls its extracted insights and past experiences to make informed decisions. Our empirical results highlight the robust learning efficacy of the ExpeL agent, indicating a consistent enhancement in its performance as it accumulates experiences. We further explore the emerging capabilities and transfer learning potential of the ExpeL agent through qualitative observations and additional experiments.

**年份与发表信息：** 2023；AAAI 2024。

#### 14. [MemoryBank：利用长期记忆增强大语言模型](https://arxiv.org/abs/2305.10250)

**中文标题：** MemoryBank：利用长期记忆增强大语言模型

**中文摘要：** 大语言模型的突破性进展彻底改变了我们与人工智能系统的交互方式，但这些模型仍明显缺少长期记忆机制。在个人陪伴系统和心理咨询等需要持续交互的场景中，这一不足尤其突出。为此，本文提出面向 LLM 的新型记忆机制 MemoryBank。MemoryBank 使模型能够调用相关记忆，通过持续更新记忆而不断演化，并通过综合过往交互信息来理解和适应用户的个性。为了模拟类人的行为并选择性保留记忆，MemoryBank 引入受艾宾浩斯遗忘曲线启发的记忆更新机制，让人工智能可以根据经过的时间和记忆的相对重要性来遗忘或强化记忆，由此形成类似人类的记忆机制。MemoryBank 既可适配 ChatGPT 等闭源模型，也可适配 ChatGLM 等开源模型。本文通过长期人工智能陪伴场景中的 LLM 聊天机器人 SiliconFriend，说明 MemoryBank 的应用方式。经过心理学对话微调后，SiliconFriend 在交互中表现出更强的同理心。实验同时包括真实用户对话的定性分析，以及模拟对话的定量分析；在模拟对话中，ChatGPT 扮演具有不同特征的用户，生成覆盖多种主题的长期对话上下文。分析结果显示，配备 MemoryBank 的 SiliconFriend 具有较强的长期陪伴能力，能够给出富有同理心的回答、回忆相关记忆并理解用户个性。

**English title:** [MemoryBank: Enhancing Large Language Models with Long-Term Memory](https://arxiv.org/abs/2305.10250)

**Abstract:** Revolutionary advancements in Large Language Models have drastically reshaped our interactions with artificial intelligence systems. Despite this, a notable hindrance remains-the deficiency of a long-term memory mechanism within these models. This shortfall becomes increasingly evident in situations demanding sustained interaction, such as personal companion systems and psychological counseling. Therefore, we propose MemoryBank, a novel memory mechanism tailored for LLMs. MemoryBank enables the models to summon relevant memories, continually evolve through continuous memory updates, comprehend, and adapt to a user personality by synthesizing information from past interactions. To mimic anthropomorphic behaviors and selectively preserve memory, MemoryBank incorporates a memory updating mechanism, inspired by the Ebbinghaus Forgetting Curve theory, which permits the AI to forget and reinforce memory based on time elapsed and the relative significance of the memory, thereby offering a human-like memory mechanism. MemoryBank is versatile in accommodating both closed-source models like ChatGPT and open-source models like ChatGLM. We exemplify application of MemoryBank through the creation of an LLM-based chatbot named SiliconFriend in a long-term AI Companion scenario. Further tuned with psychological dialogs, SiliconFriend displays heightened empathy in its interactions. Experiment involves both qualitative analysis with real-world user dialogs and quantitative analysis with simulated dialogs. In the latter, ChatGPT acts as users with diverse characteristics and generates long-term dialog contexts covering a wide array of topics. The results of our analysis reveal that SiliconFriend, equipped with MemoryBank, exhibits a strong capability for long-term companionship as it can provide emphatic response, recall relevant memories and understand user personality.

**年份与发表信息：** 2023；AAAI 2024。

#### 15. [Reflexion：采用语言强化学习的语言智能体](https://arxiv.org/abs/2303.11366)

**中文标题：** Reflexion：采用语言强化学习的语言智能体

**中文摘要：** 大语言模型（LLM）越来越多地作为目标驱动的智能体与外部环境交互，例如游戏、编译器和 API。然而，让这类语言智能体快速、高效地从试错中学习仍很困难，因为传统强化学习方法需要大量训练样本和昂贵的模型微调。本文提出 Reflexion：一种不通过更新权重，而是使用语言反馈强化语言智能体的新框架。具体来说，Reflexion 智能体以语言形式反思任务反馈信号，再把自己的反思文本保存在情景记忆缓冲区中，从而在后续尝试中做出更好的决策。Reflexion 足够灵活，可以接收不同类型的反馈信号（标量值或自由文本）和不同来源的反馈（外部反馈或内部模拟反馈）；在序列决策、编程和语言推理等多种任务上，它都相较基线智能体取得了显著提升。例如，Reflexion 在 HumanEval 编程基准上达到 91% 的 pass@1 准确率，超过了此前达到 80% 的最新最优 GPT-4。本文还针对不同反馈信号、反馈整合方法和智能体类型开展消融与分析研究，并说明这些因素如何影响性能。

**English title:** [Reflexion: Language Agents with Verbal Reinforcement Learning](https://arxiv.org/abs/2303.11366)

**Abstract:** Large language models (LLMs) have been increasingly used to interact with external environments (e.g., games, compilers, APIs) as goal-driven agents. However, it remains challenging for these language agents to quickly and efficiently learn from trial-and-error as traditional reinforcement learning methods require extensive training samples and expensive model fine-tuning. We propose Reflexion, a novel framework to reinforce language agents not by updating weights, but instead through linguistic feedback. Concretely, Reflexion agents verbally reflect on task feedback signals, then maintain their own reflective text in an episodic memory buffer to induce better decision-making in subsequent trials. Reflexion is flexible enough to incorporate various types (scalar values or free-form language) and sources (external or internally simulated) of feedback signals, and obtains significant improvements over a baseline agent across diverse tasks (sequential decision-making, coding, language reasoning). For example, Reflexion achieves a 91% pass@1 accuracy on the HumanEval coding benchmark, surpassing the previous state-of-the-art GPT-4 that achieves 80%. We also conduct ablation and analysis studies using different feedback signals, feedback incorporation methods, and agent types, and provide insights into how they affect performance.

**年份与发表信息：** 2023；NeurIPS 2023。

### Harness 与脚手架演化 / Harness & Scaffold Evolution

#### 16. [智能体 Harness 工程：可观测性驱动的编码智能体 Harness 自动演化](https://arxiv.org/abs/2604.25850)

**中文标题：** 智能体 Harness 工程：可观测性驱动的编码智能体 Harness 自动演化

**中文摘要：** Harness 如今是影响编码智能体性能的核心部分，它决定模型如何与工具和执行环境交互。然而，Harness 工程仍依赖人工，因为要实现自动化，需要处理跨可编辑组件的异构动作空间、淹没有效信号的海量轨迹，以及难以归因具体效果的修改。本文提出智能体 Harness 工程（Agentic Harness Engineering，AHE），这是一个通过三类相互配合的可观测性支柱解决上述问题的闭环：（1）组件可观测性，为每个可编辑 Harness 组件提供文件级表示，使动作空间明确且可回退；（2）经验可观测性，将数百万原始轨迹 token 提炼为分层、可下钻的证据语料，让负责演化的智能体能够实际使用；（3）决策可观测性，让每次修改都附带智能体自行声明的预测，再依据下一轮任务级结果验证。三者共同把每次修改变成可证伪的约定，使 Harness 可以自主演化，而不会退化为盲目试错。实证结果显示，经过十轮 AHE 迭代，Terminal-Bench 2 的 pass@1 从 69.7% 提高到 77.0%，超过人工设计的 Codex-CLI Harness（71.9%）以及 ACE 和 TF-GRPO 两个自演化基线。冻结后的 Harness 无需再次演化即可迁移：在 SWE-bench-verified 上，它以比种子方案少 12% 的 token 取得最高总体成功率；在 Terminal-Bench 2 上，它还为另外三个模型家族带来 5.1 至 10.1 个百分点的跨家族提升。这表明演化后的组件编码了通用工程经验，而不是只对基准进行调优。消融实验把增益定位到工具、中间件和长期记忆，而非系统提示词，说明事实性的 Harness 结构可以迁移，文字层面的策略则不能。

**English title:** [Agentic Harness Engineering: Observability-Driven Automatic Evolution of Coding-Agent Harnesses](https://arxiv.org/abs/2604.25850)

**Abstract:** Harnesses are now central to coding-agent performance, mediating how models interact with tools and execution environments. Yet harness engineering remains a manual craft, because automating it faces a heterogeneous action space across editable components, voluminous trajectories that bury actionable signal, and edits whose effect is hard to attribute. We introduce Agentic Harness Engineering (AHE), a closed loop that addresses these challenges through three matched observability pillars: (1) component observability gives every editable harness component a file-level representation so the action space is explicit and revertible; (2) experience observability distills millions of raw trajectory tokens into a layered, drill-down evidence corpus that an evolving agent can actually consume; and (3) decision observability pairs every edit with a self-declared prediction, later verified against the next round's task-level outcomes. Together, these pillars turn every edit into a falsifiable contract, so harness evolution proceeds autonomously without collapsing into trial-and-error. Empirically, ten AHE iterations lift pass@1 on Terminal-Bench 2 from 69.7% to 77.0%, surpassing the human-designed harness Codex-CLI (71.9%) and the self-evolving baselines ACE and TF-GRPO. The frozen harness transfers without re-evolution: on SWE-bench-verified it tops aggregate success at 12% fewer tokens than the seed, and on Terminal-Bench 2 it yields +5.1 to +10.1pp cross-family gains across three alternate model families, indicating the evolved components encode general engineering experience rather than benchmark-specific tuning. Ablations localize the gain to tools, middleware, and long-term memory rather than the system prompt, suggesting factual harness structure transfers while prose-level strategy does not.

**年份与发表信息：** 2026；arXiv 2026。

#### 17. [测试时 AI4AI：通过 Harness 实现由强到弱的能力迁移](https://arxiv.org/abs/2608.12307)

**中文标题：** 测试时 AI4AI：通过 Harness 实现由强到弱的能力迁移

**中文摘要：** 近期的蒸馏研究通常通过教师强制、在策略蒸馏和相关训练时方法更新较小模型的参数，从而把大模型的能力迁移给小模型。本文研究这种迁移能否改在测试时完成。研究考察“强到弱脚手架”：较强的构建模型能否设计推理时 Harness，在不更新任何参数的情况下，帮助较弱的目标模型更可靠地解决任务。实验采用四个有代表性的心智理论基准；每个构建模型都使用 5% 的数据作为验证集，经过多轮迭代完善其 Harness，随后在完整测试集上评估最终 Harness。结果显示，这种测试时能力迁移非常有效，目标模型的平均性能几乎翻倍，从 0.49 提高到 0.91。分析表明，增益主要来自把不稳定的模型推理转移到确定性代码、针对基准进行路由，以及严格约束答案格式，而不是促使目标模型开展更深入的推理或更广泛的采样。研究还发现，构建模型投入更多推理工作时，Harness 质量会单调提高；与构建模型自身能力相比，平台效应较小；能力较弱的目标模型得到的增益最大。这些结果说明，推理时 Harness 设计是传统训练时蒸馏的重要补充，可以让强模型在无须重新训练的情况下，把认知结构迁移给弱模型。

**English title:** [AI4AI at Test-Time: Strong-to-Weak Capability Transfer via Harnesses](https://arxiv.org/abs/2608.12307)

**Abstract:** Recent work on distillation transfers the capabilities of large models to smaller ones often by updating the latter's parameters, through teacher forcing, on-policy distillation, and related training-time methods. In this paper, we ask whether such transfer can instead occur at test time. We study strong-to-weak scaffolding: whether a stronger builder model can construct inference-time harnesses that help a weaker target model solve tasks more reliably without any parameter updates. Using four representative Theory-of-Mind benchmarks, each builder model uses 5% of the data as a validation set to iteratively refine its harness over multiple rounds, after which the finalized harness is evaluated on the full test set. Empirically, this form of test-time capability transfer is highly effective, nearly doubling average target-model performance from 0.49 to 0.91. Our analysis shows that the gains come primarily from offloading unstable model reasoning into deterministic code, benchmark-specific routing, and strict answer-format enforcement, rather than from encouraging the target model to reason more extensively or sample more broadly. We further find that builder-model reasoning effort improves harness quality monotonically, platform effects are modest relative to the builder model's own capability, and weaker target models receive the largest gains. These results suggest that inference-time harness design is an important complement to conventional training-time distillation, enabling strong models to transfer cognitive structure to weaker models without retraining.

**年份与发表信息：** 2026；arXiv 2026。

#### 18. [AutoHarness：通过自动合成代码 Harness 改进大语言模型智能体](https://arxiv.org/abs/2603.03329)

**中文标题：** AutoHarness：通过自动合成代码 Harness 改进大语言模型智能体

**中文摘要：** 尽管语言模型近年取得了显著进展，但它们作为智能体使用时，往往会尝试执行对当前状态而言并非只是次优、而是被外部环境严格禁止的动作。例如，在近期 Kaggle GameArena 国际象棋比赛中，Gemini-2.5-Flash 的败局有 78% 归因于非法走法。人们通常会手工编写围绕 LLM 的“Harness”，以防止这类失败。本文证明，Gemini-2.5-Flash 能够自动合成这种代码 Harness：它只需根据游戏环境反馈进行少量几轮的迭代代码改进。所得 Harness 在 145 个不同的 TextArena 游戏（包括单人和双人游戏）中消除了全部非法动作，使较小的 Gemini-2.5-Flash 模型超过 Gemini-2.5-Pro 等更大模型。把这一技术推向极致后，Gemini-2.5-Flash 可以用代码生成完整策略，决策时不再需要使用 LLM。在 16 个 TextArena 单人游戏上，由此得到的代码策略取得了高于 Gemini-2.5-Pro 和 GPT-5.2-High 的平均奖励。结果表明，使用较小模型合成定制代码 Harness（乃至完整策略）可以超过大得多的模型，同时更具成本效益。

**English title:** [AutoHarness: improving LLM agents by automatically synthesizing a code harness](https://arxiv.org/abs/2603.03329)

**Abstract:** Despite significant strides in language models in the last few years, when used as agents, such models often try to perform actions that are not just suboptimal for a given state, but are strictly prohibited by the external environment. For example, in the recent Kaggle GameArena chess competition, 78% of Gemini-2.5-Flash losses were attributed to illegal moves. Often people manually write "harnesses" around LLMs to prevent such failures. In this paper, we demonstrate that Gemini-2.5-Flash can automatically synthesize such a code harness, using a small number of rounds of iterative code refinement given feedback from the (game) environment. The resulting harness prevents all illegal moves in 145 different TextArena games (both 1-player and 2-player), enabling the smaller Gemini-2.5-Flash model to outperform larger models, such as Gemini-2.5-Pro. Pushing our technique to the limit, we can get Gemini-2.5-Flash to generate the entire policy in code, thus eliminating the need to use the LLM at decision making time. The resulting code-policy receives a higher average reward than Gemini-2.5-Pro and GPT-5.2-High on 16 TextArena 1-player games. Our results show that using a smaller model to synthesize a custom code harness (or entire policy) can outperform a much larger model, while also being more cost effective.

**年份与发表信息：** 2026；arXiv 2026。

#### 19. [持续 Harness：自我改进基础智能体的在线适配](https://arxiv.org/abs/2605.09998)

**中文标题：** 持续 Harness：自我改进基础智能体的在线适配

**中文摘要：** Claude Code 和 OpenHands 等编码 Harness 为基础模型封装了工具、记忆和规划，但具身智能体的长时程、部分可观测决策尚无对应方案。本文首先报告 Gemini Plays Pokemon（GPP）实验。通过人在回路的迭代 Harness 改进，GPP 成为首个在不输掉一场战斗的情况下通关《宝可梦 蓝》《Yellow Legacy》困难模式和《宝可梦 水晶》的人工智能系统。在最困难的阶段，智能体开始通过长上下文记忆自行迭代策略，在人工改进 Harness 的同时表现出涌现的自我改进信号。Continual Harness 将人类完全移出这一闭环：它是一种面向具身智能体、无需重置的自我改进 Harness，把上述观察到的过程形式化并自动化。系统只从一个最小环境接口开始，智能体在行动与完善自身提示词、子智能体、技能和记忆之间交替，并可利用任意过去轨迹数据。提示词优化方法需要重置回合，而 Continual Harness 能在一次连续运行中在线适配。在多个前沿模型参与的《宝可梦 红》和《宝可梦 绿宝石》实验中，从零开始的 Continual Harness 相较最简基线大幅减少按键成本，并弥合了与人工设计专家 Harness 之间的大部分差距；增益随模型能力而异，而系统的起点始终是相同的原始接口，没有精选知识、手工工具或领域脚手架。本文还让模型自身进入闭环：开源智能体在持续完善的 Harness 中生成轨迹，前沿教师模型重新标注这些轨迹并用于更新模型，由此形成在线过程奖励协同学习闭环。该闭环在不于训练迭代之间重置环境的情况下，推动智能体在《宝可梦 红》中持续达成新的游戏里程碑。

**English title:** [Continual Harness: Online Adaptation for Self-Improving Foundation Agents](https://arxiv.org/abs/2605.09998)

**Abstract:** Coding harnesses such as Claude Code and OpenHands wrap foundation models with tools, memory, and planning, but no equivalent exists for embodied agents' long-horizon partial-observability decision-making. We first report our Gemini Plays Pokemon (GPP) experiments. With iterative human-in-the-loop harness refinement, GPP became the first AI system to complete Pokemon Blue, Yellow Legacy on hard mode, and Crystal without a lost battle. In the hardest stages, the agent itself began iterating on its strategy through long-context memory, surfacing emergent self-improvement signals alongside human-in-the-loop refinement. Continual Harness removes the human fully from this loop: a reset-free self-improving harness for embodied agents that formalizes and automates what we observed. Starting from only a minimal environment interface, the agent alternates between acting and refining its own prompt, sub-agents, skills, and memory, drawing on any past trajectory data. Prompt-optimization methods require episode resets; Continual Harness adapts online within a single run. On Pokemon Red and Emerald across frontier models, Continual Harness starting from scratch substantially reduces button-press cost relative to the minimalist baseline and recovers a majority of the gap to a hand-engineered expert harness, with capability-dependent gains, despite starting from the same raw interface with no curated knowledge, no hand-crafted tools, and no domain scaffolding. We then close the loop with the model itself: an online process-reward co-learning loop, in which an open-source agent's rollouts through the refining harness are relabeled by a frontier teacher and used to update the model, drives sustained in-game milestone progress on Pokemon Red without resetting the environment between training iterations.

**年份与发表信息：** 2026；arXiv 2026。

#### 20. [EvoHarness-RL：学习面向长时程大语言模型智能体的自演化运行时 Harness](https://arxiv.org/abs/2608.05446)

**中文标题：** EvoHarness-RL：学习面向长时程大语言模型智能体的自演化运行时 Harness

**中文摘要：** 长时程大语言模型智能体越来越依赖外部执行支持，以维持状态、跟踪进度、调用工具、验证结果并在交互之间复用经验。然而，有效使用 Harness 会引出两个相互关联的问题：如何从嘈杂的交互轨迹形成状态，以及如何在运行时控制外部状态访问。现有智能体通常用提示词、启发式规则或领域专用约定来处理两者，外部工作区及其使用策略仍须人工设计。本文由此研究 Harness 策略学习：智能体离线学习 Harness 策略，并在运行时执行任务的过程中部署这些策略，以在线构建和更新外部 Harness 状态。本文提出 EvoHarness-RL，把信念（Belief）、进度（Progress）和经验（Experience），即 BPE，作为面向策略的 Harness 状态。监督式 Harness 微调教会基础智能体 Harness 动作空间，以及如何构建有用的外部状态；考虑成本的 GRPO 则探索协调策略，在长时程交互中选择性读取、更新和整合这些状态。在 ALFWorld 上以 Qwen3-8B 实现后，EvoHarness-RL 达到 96.9% 的成功率，并揭示两种现象：一是 Harness 退火，即训练把反复出现的 Harness 使用模式内化到模型策略中，使智能体从频繁调用 Harness 转向选择性访问外部状态；二是 Harness 演化，即进度更新和经验整合把 Harness 逐步完善为紧凑且适应任务的状态基础。结果说明，长时程智能体受益于可训练的外部 Harness 工作区构建与协调策略，而不只是更强的工具或更大的记忆。

**English title:** [EvoHarness-RL: Learning Self-Evolving Runtime Harness for Long-Horizon LLM Agents](https://arxiv.org/abs/2608.05446)

**Abstract:** Long-horizon LLM agents increasingly rely on external execution support to maintain state, track progress, invoke tools, verify outcomes, and reuse experience across interactions. However, effective harness use raises two coupled challenges: state formation from noisy interaction traces and runtime control over external-state access. Existing agents usually handle both through prompts, heuristics, or domain-specific conventions, leaving the external workspace and its usage policy manually engineered. To address this, we study the problem of harness policy learning, where agents learn harness policies offline and deploy them to construct and update external harness state online during runtime task execution. We introduce EvoHarness-RL, which exposes Belief, Progress, and Experience (BPE) as policy-facing harness state. Supervised harness fine-tuning teaches the base agent the harness action space and how to construct useful external state, while cost-aware GRPO explores coordination policies to selectively read, update, and consolidate that state during long-horizon interaction. Instantiated on ALFWorld with a Qwen3-8B LLM, EvoHarness-RL reaches 96.9% success and reveals two key dynamics: harness annealing, where training internalizes recurring harness-use patterns into the model policy and shifts the agent from frequent harness calls toward selective external-state access, and harness evolution, where progress updates and experience consolidation refine the harness into a compact, task-adaptive state substrate. These results suggest that long-horizon agents benefit from trainable policies for constructing and coordinating with external harness workspaces, beyond simply adding stronger tools or larger memories.

**年份与发表信息：** 2026；COLM 2026 LLA Workshop。

#### 21. [MemoHarness：能够从经验中学习的智能体 Harness](https://arxiv.org/abs/2607.14159)

**中文标题：** MemoHarness：能够从经验中学习的智能体 Harness

**中文摘要：** 智能体 Harness 是把基础 LLM 转变为可执行智能体的外部控制层，负责管理上下文、工具、编排、记忆、解码和输出处理。尽管 Harness 设计对智能体行为影响很大，大多数自动改进方法仍只优化提示词、流水线或工作流等范围更窄的对象，而且已部署智能体通常对所有案例复用同一套全局 Harness。本文提出 MemoHarness，一个从自身执行中学习的自适应 Harness 优化框架。MemoHarness 把 Harness 分解为六个可编辑的控制维度，将每个案例的诊断结果和提炼出的全局模式存入双层经验库，再利用检索到的经验针对每个测试案例调整学习得到的 Harness，而不需要测试时标签、反馈或额外搜索。对 Shell 智能体、代码生成和分析推理基准的评估显示，MemoHarness 优于实验所比较的固定 Harness，并能选择性迁移到未见过的测试套件和基础模型。若大部分检索经验可以缓存，它额外增加上下文后仍能维持成本竞争力。这些结果表明，执行经验可以作为实际可用的基础，用于构建比单一静态配置更具适应性的智能体 Harness；不过，更广泛的统计稳健性和组件归因结论仍留待后续研究。

**English title:** [MemoHarness: Agent Harnesses That Learn from Experience](https://arxiv.org/abs/2607.14159)

**Abstract:** An agent harness is the external control layer that turns a base LLM into an executable agent by managing context, tools, orchestration, memory, decoding, and output handling. While harness design strongly affects agent behavior, most automatic improvement methods optimize narrower artifacts such as prompts, pipelines, or workflows, and deployed agents usually reuse a single global harness for all cases. We introduce MemoHarness, an adaptive harness optimization framework that learns from its own executions. MemoHarness decomposes the harness into six editable control dimensions, stores per-case diagnoses and distilled global patterns in a dual-layer experience bank, and adapts the learned harness to each test case using retrieved experience without test-time labels, feedback, or additional search. In our evaluation across shell-agent, code-generation, and analytical-reasoning benchmarks, MemoHarness improves over the fixed harnesses we compare against and shows selective transfer to unseen suites and base models. Its additional context can also remain cost-competitive when much of the retrieved experience is cacheable. These results provide evidence that execution experience is a practical substrate for building agent harnesses that are more adaptive than a single static configuration, while leaving broader claims about statistical robustness and component attribution to future work.

**年份与发表信息：** 2026；arXiv 2026。

#### 22. [MetaSkill-Evolve：通过双时间尺度元技能演化实现大语言模型智能体的递归自我改进](https://arxiv.org/abs/2607.05297)

**中文标题：** MetaSkill-Evolve：通过双时间尺度元技能演化实现大语言模型智能体的递归自我改进

**中文摘要：** 近期的大语言模型智能体开始处理时程越来越长、开放程度越来越高的任务；由外部提供给智能体、可复用的程序性知识，即技能，进一步扩展了这一能力。然而，固定且人工编写的技能很少是最优方案，也无法适应智能体遇到的多样化任务。自我改进智能体会根据执行轨迹重写自己的技能文件，并在有挑战性的基准上取得明显增益。但这类自演化仍非递归式：它只改进任务技能（智能体做什么），而改进过程本身（如何改进）只编写一次，此后保持不变。本文提出 MetaSkill-Evolve，一个让智能体技能改进具有递归性的双时间尺度框架。每个分支同时带有任务技能 $s$ 和该分支专属的元技能 $m=(ψ,σ,α,π,\varepsilon)$，其中五个组成部分分别参数化改进流水线中的分析器、检索器、分配器、提议器和演化器智能体。任务技能在快速循环中演化，元技能则在较慢循环中演化；后者把同一套流水线应用于自身，无须额外模型或目标函数。五个流水线智能体共享同一个冻结的骨干模型。MetaSkill-Evolve 在 OfficeQA、SealQA 和 ALFWorld 三个智能体基准上超过无技能、静态技能及单层演化基线，相较原始骨干模型，留出测试准确率分别提高 23.54、16.09 和 1.92 个百分点。

**English title:** [MetaSkill-Evolve: Recursive Self-Improvement of LLM Agents via Two-Timescale Meta-Skill Evolution](https://arxiv.org/abs/2607.05297)

**Abstract:** Recent LLM agents tackle increasingly long-horizon, open-ended tasks, and external skills, reusable procedural knowledge supplied to the agent, further extend this capability. However, a fixed, hand-authored skill is rarely optimal, and cannot adapt to the diversity of tasks an agent encounters. Self-improving agents address this by rewriting their own skill files from execution traces, yielding meaningful gains on challenging benchmarks. Yet such self-evolution remains non-recursive: it improves only the task skill (what the agent does) while the improvement procedure (how it improves) is authored once and held fixed. We introduce MetaSkill-Evolve, a two-timescale framework that makes agentic skill improvement recursive: every branch carries both a task skill $s$ and a branch-local meta-skill $m=(ψ,σ,α,π,\varepsilon)$ whose five components parameterise the Analyzer, Retriever, Allocator, Proposer, and Evolver agents of the improvement pipeline. Task skills evolve on a fast loop while the meta-skill evolves on a slower one under the same pipeline applied to itself, with no additional model or objective. With all five pipeline agents sharing a single frozen backbone, MetaSkill-Evolve outperforms no-skill, static-skill, and single-level evolution baselines on three agentic benchmarks (OfficeQA, SealQA, ALFWorld), improving held-out test accuracy over the raw backbone by +23.54, +16.09, and +1.92 points respectively.

**年份与发表信息：** 2026；arXiv 2026。

#### 23. [SkillOpt：面向自演化智能体技能的执行策略](https://arxiv.org/abs/2605.23904)

**中文标题：** SkillOpt：面向自演化智能体技能的执行策略

**中文摘要：** 当前的智能体技能要么由人工编写，要么一次性生成，或通过控制较松的自我修订过程演化。这些方法都不像深度学习优化器那样优化技能，也都无法在反馈下可靠地改进起点。本文主张，应把技能视为冻结智能体的外部状态来训练，并采用使权重空间优化可复现的同等严格规范。据作者所知，SkillOpt 是首个面向智能体技能、系统且可控的文本空间优化器：一个独立优化器模型把带分数的推演转化为对单一技能文档的受限增、删、替换编辑；只有当编辑严格提高留出验证分数时，系统才会接受它。文本学习率预算、被拒编辑缓冲区以及逐轮慢速／元更新，使技能训练保持稳定，同时部署时不增加任何推理模型调用。在六个基准、七个目标模型和三种执行 Harness（直接聊天、Codex、Claude Code）上，SkillOpt 在全部 52 个（模型、基准、Harness）组合中均取得最好或并列最好的成绩，并在每个组合中胜过人工、一次性 LLM、Trace2Skill、TextGrad、GEPA 和 EvoSkill 技能。在 GPT-5.5 上，它相较无技能设置把直接聊天的平均准确率提高 23.5 个百分点，在 Codex 智能体循环中提高 24.8 个百分点，在 Claude Code 中提高 19.1 个百分点。迁移实验还表明，优化后的技能产物跨模型规模、在 Codex 与 Claude Code 执行环境之间，以及迁移到邻近数学基准且不再优化时，仍能保留价值。代码：https://aka.ms/skillopt。

**English title:** [SkillOpt: Executive Strategy for Self-Evolving Agent Skills](https://arxiv.org/abs/2605.23904)

**Abstract:** Agent skills today are hand-crafted, generated one-shot, or evolved through loosely controlled self-revision, none of which behaves like a deep-learning optimizer for the skill, and none of which reliably improves over its starting point under feedback. We argue the skill should instead be trained as the external state of a frozen agent, with the same discipline that makes weight-space optimization reproducible. SkillOpt is, to our knowledge, the first systematic controllable text-space optimizer for agent skills: a separate optimizer model turns scored rollouts into bounded add/delete/replace edits on a single skill document, and an edit is accepted only when it strictly improves a held-out validation score. A textual learning-rate budget, rejected-edit buffer, and epoch-wise slow/meta update make skill training stable while adding zero inference-time model calls at deployment. Across six benchmarks, seven target models, and three execution harnesses (direct chat, Codex, Claude Code), SkillOpt is best or tied on all 52 evaluated (model, benchmark, harness) cells and beats every per-cell competitor among human, one-shot LLM, Trace2Skill, TextGrad, GEPA, and EvoSkill skills. On GPT-5.5 it lifts the average no-skill accuracy by +23.5 points in direct chat, by +24.8 inside the Codex agentic loop, and by +19.1 inside Claude Code. Transfer experiments further show that optimized skill artifacts retain value when moved across model scales, between Codex and Claude Code execution environments, and to a nearby math benchmark without further optimization. Code: https://aka.ms/skillopt

**年份与发表信息：** 2026；arXiv 2026。

### 可扩展 Harness 基底 / Extensible Harness Substrates

可扩展的智能体运行时会把提示词、工具、技能、记忆、插件或控制流暴露为可修改界面。这些系统未必默认实现自我改进，但可以作为递归自我改进实验的基础。

仅有一般性的可扩展能力还不够。递归自我改进基底应当把提示词、记忆、技能、工具或控制流等与智能体有关的运行时组件，暴露为可通过程序修改的界面，以便开展持久的自我修改实验。

Extensible agent runtimes that expose prompts, tools, skills, memory, plugins, or control flow as modifiable surfaces. These systems do not necessarily implement self-improvement by default, but can serve as substrates for RSI experiments.

General extensibility alone is insufficient. An RSI substrate should expose agent-relevant runtime components such as prompts, memory, skills, tools, or control flow as programmatically modifiable surfaces suitable for persistent self-modification experiments.

#### 24. [Agent Zero](https://github.com/agent0ai/agent-zero)

**中文标题：** Agent Zero

**中文简介（来源：项目官方 README）：** Agent Zero 是一个开源智能体框架，面向仅靠聊天无法完成的工作。它为智能体提供容器化的 Linux 桌面、带 DOM 标注的浏览器、实时文档协作、项目、技能、插件，以及连接宿主机的桥梁。智能体可以在 Canvas 中使用实际的图形界面软件、终端、文件和桌面应用；浏览器支持读取和操作网页，并可将页面标注直接转为检查、修改、提取或审阅指令。项目与记忆功能可按项目隔离文件、指令、密钥、记忆、代码仓库和模型预设。框架还支持多智能体协作、100 多个社区插件、MCP、A2A、自定义工具和自定义提示词。其内部机制保持透明：提示词位于 `prompts/`，工具位于 `tools/` 或插件中，内置行为可检查、修改、替换和扩展。

**English title:** [Agent Zero](https://github.com/agent0ai/agent-zero)

**Description (Source: official project README):** Agent Zero is an open agent framework for work that needs more than chat: a Dockerized Linux desktop, a browser with DOM annotation, live document cowork, projects, skills, plugins, and a bridge back to your host machine. Its full Linux desktop lets the agent use real GUI software, terminals, files, and desktop apps inside the Canvas. Projects and memory keep files, instructions, secrets, memories, repositories, and model-preset choices isolated per project. It supports multi-agent cooperation and a Plugin Hub with more than 100 community plugins. Almost nothing is hidden: prompts live in `prompts/`, tools live in `tools/` or plugins, and built-in behavior can be inspected, changed, replaced, or extended. Agent Zero also supports plugins, MCP, A2A, custom tools, custom prompts, project-scoped configuration, and environment-based deployment settings.

**年份与发表信息：** GitHub 开源项目；项目简介未标注发表年份。

#### 25. [DeepSeek Harness](https://github.com/deepseek-ai/deepseek-harness)

**中文标题：** DeepSeek Harness

**中文简介（来源：项目官方 README）：** DeepSeek Harness（`dsh`）是 DeepSeek AI 开发的开源智能体 Harness。项目采用“一切皆插件”的架构，并由 Cordis 驱动；Cordis 的设计见论文《A Programming Paradigm for Spatiotemporal Composability》。DeepSeek Harness 目前处于开发者预览阶段并在快速迭代，可能出现破坏兼容性的变更。它可以通过 npm 直接启动 Web UI，也支持从源代码构建和运行。项目采用 MIT 许可证。

**English title:** [DeepSeek Harness](https://github.com/deepseek-ai/deepseek-harness)

**Description (Source: official project README):** DeepSeek Harness (`dsh`) is an open-source agent harness developed by DeepSeek AI. It is built on an everything-is-a-plugin architecture and powered by Cordis, whose design is described in *A Programming Paradigm for Spatiotemporal Composability*. DeepSeek Harness is in developer preview and iterating rapidly, with compatibility-breaking changes expected. It can be run from npm to start a Web UI or built and run from source. The project is licensed under MIT.

**年份与发表信息：** GitHub 开源项目；项目简介未标注发表年份。

#### 26. [OpenClaw](https://github.com/openclaw/openclaw)

**中文标题：** OpenClaw

**中文简介（来源：项目官方 README）：** OpenClaw 是一款在用户自己的计算机上运行的开源人工智能助手，可接入用户已经使用的渠道，包括 Discord、iMessage、Slack、Teams、Telegram、WhatsApp 等 20 多种渠道，以及 macOS、iOS、Android、Windows 和 Linux 原生应用。同一个 Gateway 既可在笔记本电脑上作为个人助手运行，也可用于团队共享部署，两者只需采用不同配置。状态、记忆和凭据保存在用户硬件上；模型和智能体 Harness（如 Claude、Codex 和本地模型）均以插件形式提供，可以替换而不影响其他部分。Gateway 是会话、工具、事件和渠道连接的本地控制平面；Control UI、命令行界面和终端界面都连接至 Gateway。OpenClaw 同时支持托管与本地模型提供商，并通过工具、技能和插件扩展助手能力。

**English title:** [OpenClaw](https://github.com/openclaw/openclaw)

**Description (Source: official project README):** OpenClaw is an open-source AI assistant that runs on your own computer and meets you in the channels you already use: Discord, iMessage, Slack, Teams, Telegram, WhatsApp, and 20+ more, plus native apps for macOS, iOS, Android, Windows, and Linux. One Gateway runs it as a personal assistant on a laptop or as a shared team deployment; configuration is the only difference. State, memory, and credentials live on your hardware. Models and agent harnesses (Claude, Codex, local models) are plugins you can swap without changing anything else. The Gateway is the local control plane for sessions, tools, events, and channel connections, while the Control UI, CLI, and TUI connect to it. OpenClaw works with hosted and local model providers, and its tools, skills, and plugins extend what an assistant can do.

**年份与发表信息：** GitHub 开源项目；项目简介未标注发表年份。

#### 27. [Pi](https://github.com/earendil-works/pi)

**中文标题：** Pi 智能体 Harness

**中文简介（来源：项目官方 README）：** Pi 是智能体 Harness 项目及其自扩展编码智能体的代码仓库。它包含交互式编码智能体命令行工具 `@earendil-works/pi-coding-agent`、具有工具调用和状态管理能力的智能体运行时 `@earendil-works/pi-agent-core`，以及统一的多提供商 LLM API `@earendil-works/pi-ai`，后者支持 OpenAI、Anthropic、Google 等提供商。仓库还提供独立的应用组合运行时 Chord、与供应商无关的遥测组件，以及采用差分渲染的终端界面库。Pi 默认以启动它的用户和进程权限运行，本身没有限制文件系统、进程、网络或凭据访问的内置权限系统；如需更强隔离，应使用微型虚拟机、Docker 或策略控制沙箱。项目采用 MIT 许可证。

**English title:** [Pi Agent Harness](https://github.com/earendil-works/pi)

**Description (Source: official project README):** This is the home of the Pi agent harness project, including its self-extensible coding agent. It contains `@earendil-works/pi-coding-agent`, an interactive coding agent CLI; `@earendil-works/pi-agent-core`, an agent runtime with tool calling and state management; and `@earendil-works/pi-ai`, a unified multi-provider LLM API supporting OpenAI, Anthropic, Google, and others. The repository also includes Chord, a standalone application-composition runtime; vendor-neutral telemetry components; and a terminal UI library with differential rendering. Pi has no built-in permission system for restricting filesystem, process, network, or credential access and runs with the permissions of the user and process that launched it; stronger boundaries require a micro-VM, Docker, or another policy-controlled sandbox. The project is licensed under MIT.

**年份与发表信息：** GitHub 开源项目；项目简介未标注发表年份。

### 自我验证与自我纠正：基础方法 / Self-Verification & Self-Correction — Enabling Foundations

这类方法主要研究有边界的自我完善与验证，并作为持久自我改进系统的构建模块。

Primarily bounded self-refinement and verification methods that serve as building blocks for persistent self-improving systems.

#### 28. [验证链减少大语言模型中的幻觉](https://arxiv.org/abs/2309.11495)

**中文标题：** 验证链减少大语言模型中的幻觉

**中文摘要：** 生成看似合理但事实错误的信息，即“幻觉”，仍是大语言模型中尚未解决的问题。本文研究语言模型能否审视自己给出的回答并纠正错误。研究提出验证链（Chain-of-Verification，CoVe）方法：模型首先生成回答草稿；然后规划用于事实核查草稿的验证问题；接着独立回答这些问题，以免答案受到其他回答的影响；最后生成经过验证的最终回答。实验表明，在来自 Wikidata 的列表型问题、闭卷 MultiSpanQA 和长文本生成等多类任务上，CoVe 都能减少幻觉。

**English title:** [Chain-of-Verification Reduces Hallucination in Large Language Models](https://arxiv.org/abs/2309.11495)

**Abstract:** Generation of plausible yet incorrect factual information, termed hallucination, is an unsolved issue in large language models. We study the ability of language models to deliberate on the responses they give in order to correct their mistakes. We develop the Chain-of-Verification (CoVe) method whereby the model first (i) drafts an initial response; then (ii) plans verification questions to fact-check its draft; (iii) answers those questions independently so the answers are not biased by other responses; and (iv) generates its final verified response. In experiments, we show CoVe decreases hallucinations across a variety of tasks, from list-based questions from Wikidata, closed book MultiSpanQA and longform text generation.

**年份与发表信息：** 2023；Findings of ACL 2024。

#### 29. [CRITIC：大语言模型可以借助工具交互式批评进行自我纠正](https://arxiv.org/abs/2305.11738)

**中文标题：** CRITIC：大语言模型可以借助工具交互式批评进行自我纠正

**中文摘要：** 大语言模型（LLM）近期的发展令人瞩目，但这些模型有时会出现不一致或有问题的行为，例如编造事实、生成有缺陷的代码，或产生冒犯性和有害内容。与之不同，人类通常会使用外部工具核查并完善初始内容，例如用搜索引擎核查事实，或用代码解释器调试。受此观察启发，本文提出 CRITIC 框架，使本质上是“黑箱”的 LLM 能够以类似人类使用工具的方式，验证并逐步修正自己的输出。具体而言，CRITIC 从初始输出出发，与适当工具交互以评估文本的特定方面，再依据验证过程中获得的反馈修订输出。自由形式问答、数学程序合成和降低毒性等任务上的综合评估表明，CRITIC 能够持续改进 LLM 的性能。研究同时说明，外部反馈对于推动 LLM 持续自我改进十分重要。

**English title:** [CRITIC: Large Language Models Can Self-Correct with Tool-Interactive Critiquing](https://arxiv.org/abs/2305.11738)

**Abstract:** Recent developments in large language models (LLMs) have been impressive. However, these models sometimes show inconsistencies and problematic behavior, such as hallucinating facts, generating flawed code, or creating offensive and toxic content. Unlike these models, humans typically utilize external tools to cross-check and refine their initial content, like using a search engine for fact-checking, or a code interpreter for debugging. Inspired by this observation, we introduce a framework called CRITIC that allows LLMs, which are essentially "black boxes" to validate and progressively amend their own outputs in a manner similar to human interaction with tools. More specifically, starting with an initial output, CRITIC interacts with appropriate tools to evaluate certain aspects of the text, and then revises the output based on the feedback obtained during this validation process. Comprehensive evaluations involving free-form question answering, mathematical program synthesis, and toxicity reduction demonstrate that CRITIC consistently enhances the performance of LLMs. Meanwhile, our research highlights the crucial importance of external feedback in promoting the ongoing self-improvement of LLMs.

**年份与发表信息：** 2023；ICLR 2024。

#### 30. [大语言模型尚不能自我纠正推理](https://arxiv.org/abs/2310.01798)

**中文标题：** 大语言模型尚不能自我纠正推理

**中文摘要：** 大语言模型（LLM）凭借在多种应用中无可比拟的文本生成能力，已成为一项突破性技术。尽管如此，人们仍然担忧其生成内容的准确性和适当性。近期有人提出用自我纠正方法解决这些问题。在这一背景下，本文批判性地考察自我纠正在 LLM 中的作用和有效性，揭示其真实潜力与局限。研究的核心是“内在自我纠正”：LLM 在没有外部反馈辅助的情况下，只依靠自身固有能力尝试纠正初始回答。在推理任务中，研究结果表明，缺少外部反馈时，LLM 很难自我纠正回答；有时，自我纠正后的性能甚至会下降。基于这些发现，本文为该领域未来的研究和实际应用提出了建议。

**English title:** [Large Language Models Cannot Self-Correct Reasoning Yet](https://arxiv.org/abs/2310.01798)

**Abstract:** Large Language Models (LLMs) have emerged as a groundbreaking technology with their unparalleled text generation capabilities across various applications. Nevertheless, concerns persist regarding the accuracy and appropriateness of their generated content. A contemporary methodology, self-correction, has been proposed as a remedy to these issues. Building upon this premise, this paper critically examines the role and efficacy of self-correction within LLMs, shedding light on its true potential and limitations. Central to our investigation is the notion of intrinsic self-correction, whereby an LLM attempts to correct its initial responses based solely on its inherent capabilities, without the crutch of external feedback. In the context of reasoning, our research indicates that LLMs struggle to self-correct their responses without external feedback, and at times, their performance even degrades after self-correction. Drawing from these insights, we offer suggestions for future research and practical applications in this field.

**年份与发表信息：** 2023；ICLR 2024。

#### 31. [让我们逐步验证](https://arxiv.org/abs/2305.20050)

**中文标题：** 让我们逐步验证

**中文摘要：** 近年，大语言模型执行复杂多步推理的能力显著提高，但即使是最先进的模型，仍会经常产生逻辑错误。为训练更可靠的模型，可以使用结果监督，即只对最终结果提供反馈；也可以使用过程监督，即对每个中间推理步骤提供反馈。考虑到训练可靠模型的重要性和人类反馈的高昂成本，有必要仔细比较这两种方法。此前研究已开始这项比较，但仍有许多问题未解决。本文开展了独立研究，发现对于训练模型求解具有挑战性的 MATH 数据集问题，过程监督显著优于结果监督。采用过程监督的模型在 MATH 测试集的代表性子集上解出了 78% 的问题。本文还表明，主动学习可以显著提高过程监督的效能。为支持相关研究，作者发布了 PRM800K，其中包含用于训练最佳奖励模型的全部 80 万个人工步骤级反馈标签。

**English title:** [Let's Verify Step by Step](https://arxiv.org/abs/2305.20050)

**Abstract:** In recent years, large language models have greatly improved in their ability to perform complex multi-step reasoning. However, even state-of-the-art models still regularly produce logical mistakes. To train more reliable models, we can turn either to outcome supervision, which provides feedback for a final result, or process supervision, which provides feedback for each intermediate reasoning step. Given the importance of training reliable models, and given the high cost of human feedback, it is important to carefully compare the both methods. Recent work has already begun this comparison, but many questions still remain. We conduct our own investigation, finding that process supervision significantly outperforms outcome supervision for training models to solve problems from the challenging MATH dataset. Our process-supervised model solves 78% of problems from a representative subset of the MATH test set. Additionally, we show that active learning significantly improves the efficacy of process supervision. To support related research, we also release PRM800K, the complete dataset of 800,000 step-level human feedback labels used to train our best reward model.

**年份与发表信息：** 2023；ICLR 2024。

#### 32. [自洽性改进语言模型的思维链推理](https://arxiv.org/abs/2203.11171)

**中文标题：** 自洽性改进语言模型的思维链推理

**中文摘要：** 思维链提示与预训练大语言模型结合后，在复杂推理任务上取得了良好结果。本文提出一种新的解码策略“自洽性”，用来替代思维链提示中的朴素贪心解码。该策略首先采样一组多样化的推理路径，而不是只采用贪心路径；随后对采样得到的推理路径进行边缘化，选出最一致的答案。自洽性利用这样一种直觉：复杂推理问题通常存在多种不同思考方式，而它们会通向同一个正确答案。大量实证评估表明，在多个常用的算术与常识推理基准上，自洽性都大幅提高了思维链提示的性能，包括 GSM8K（+17.9%）、SVAMP（+11.0%）、AQuA（+12.2%）、StrategyQA（+6.4%）和 ARC-challenge（+3.9%）。

**English title:** [Self-Consistency Improves Chain of Thought Reasoning in Language Models](https://arxiv.org/abs/2203.11171)

**Abstract:** Chain-of-thought prompting combined with pre-trained large language models has achieved encouraging results on complex reasoning tasks. In this paper, we propose a new decoding strategy, self-consistency, to replace the naive greedy decoding used in chain-of-thought prompting. It first samples a diverse set of reasoning paths instead of only taking the greedy one, and then selects the most consistent answer by marginalizing out the sampled reasoning paths. Self-consistency leverages the intuition that a complex reasoning problem typically admits multiple different ways of thinking leading to its unique correct answer. Our extensive empirical evaluation shows that self-consistency boosts the performance of chain-of-thought prompting with a striking margin on a range of popular arithmetic and commonsense reasoning benchmarks, including GSM8K (+17.9%), SVAMP (+11.0%), AQuA (+12.2%), StrategyQA (+6.4%) and ARC-challenge (+3.9%).

**年份与发表信息：** 2022；ICLR 2023。

#### 33. [Self-Refine：利用自我反馈进行迭代完善](https://arxiv.org/abs/2303.17651)

**中文标题：** Self-Refine：利用自我反馈进行迭代完善

**中文摘要：** 与人类一样，大语言模型（LLM）第一次尝试时也不总能生成最佳输出。受人类修改书面文本的方式启发，本文提出 Self-Refine：一种通过迭代反馈和完善来改进 LLM 初始输出的方法。其基本思路是先用 LLM 生成初始输出，再让同一个 LLM 对自己的输出给出反馈，并使用该反馈迭代地完善输出。Self-Refine 不需要任何监督训练数据、额外训练或强化学习，而是让单个 LLM 同时担任生成器、完善器和反馈提供者。本文使用先进的 LLM（GPT-3.5、ChatGPT 和 GPT-4），在从对话回复生成到数学推理的七类不同任务上评估 Self-Refine。在所有任务中，无论依据人类偏好还是自动指标，Self-Refine 生成的输出都优于同一 LLM 采用常规单步生成得到的输出，任务性能平均绝对提高约 20%。研究表明，即使 GPT-4 这样的先进 LLM，也可以在测试时借助这一简单、独立的方法得到进一步改进。

**English title:** [Self-Refine: Iterative Refinement with Self-Feedback](https://arxiv.org/abs/2303.17651)

**Abstract:** Like humans, large language models (LLMs) do not always generate the best output on their first try. Motivated by how humans refine their written text, we introduce Self-Refine, an approach for improving initial outputs from LLMs through iterative feedback and refinement. The main idea is to generate an initial output using an LLMs; then, the same LLMs provides feedback for its output and uses it to refine itself, iteratively. Self-Refine does not require any supervised training data, additional training, or reinforcement learning, and instead uses a single LLM as the generator, refiner, and feedback provider. We evaluate Self-Refine across 7 diverse tasks, ranging from dialog response generation to mathematical reasoning, using state-of-the-art (GPT-3.5, ChatGPT, and GPT-4) LLMs. Across all evaluated tasks, outputs generated with Self-Refine are preferred by humans and automatic metrics over those generated with the same LLM using conventional one-step generation, improving by ~20% absolute on average in task performance. Our work demonstrates that even state-of-the-art LLMs like GPT-4 can be further improved at test time using our simple, standalone approach.

**年份与发表信息：** 2023；NeurIPS 2023。

### 自演化智能体框架 / Self-Evolving Agent Frameworks

#### 1. AgentFactory：通过可执行子智能体的积累与复用实现自演化的框架

**完整中文摘要：**

构建基于大型语言模型（LLM）的智能体已变得愈发重要。近期关于 LLM 智能体自演化的研究，主要将成功经验记录为文本提示或反思内容；面对复杂场景时，这些形式无法可靠地保证任务得到高效的再次执行。我们提出 AgentFactory，这是一种新的自演化范式：它不再以文本经验的形式记录成功的任务解决方案，而是将其保存为可执行的子智能体代码。关键在于，这些子智能体会根据执行反馈持续改进，并随着处理任务数量的增加而变得更加稳健、高效。保存的子智能体是带有标准化文档的纯 Python 代码，因此可以移植到任何支持 Python 的系统。我们证明，AgentFactory 能够持续积累能力：其可执行子智能体库会随时间不断扩充和改进，在无须人工干预的情况下，逐步减少解决相似任务所需的工作量。我们的实现已在 https://github.com/zzatpku/AgentFactory 开源，演示视频见 https://youtu.be/iKSsuAXJHW0。

**English title:**

AgentFactory: A Self-Evolving Framework Through Executable Subagent Accumulation and Reuse

**Abstract:**

> Building LLM-based agents has become increasingly important. Recent works on LLM-based agent self-evolution primarily record successful experiences as textual prompts or reflections, which cannot reliably guarantee efficient task re-execution in complex scenarios. We propose AgentFactory, a new self-evolution paradigm that preserves successful task solutions as executable subagent code rather than textual experience. Crucially, these subagents are continuously refined based on execution feedback, becoming increasingly robust and efficient as more tasks are encountered. Saved subagents are pure Python code with standardized documentation, enabling portability across any Python-capable system. We demonstrate that AgentFactory enables continuous capability accumulation: its library of executable subagents grows and improves over time, progressively reducing the effort required for similar tasks without manual intervention. Our implementation is open-sourced at https://github.com/zzatpku/AgentFactory, and our demonstration video is available at https://youtu.be/iKSsuAXJHW0.

- 链接：https://arxiv.org/abs/2603.18000v1
- 年份：2026
- 发表信息：ACL 2026 System Demonstrations；arXiv:2603.18000v1。

#### 2. EvoAgent：具备技能学习与多智能体委派能力的可演化智能体框架

**完整中文摘要：**

本文提出 EvoAgent，一种将结构化技能学习与分层子智能体委派机制相结合的可演化大型语言模型（LLM）智能体框架。EvoAgent 将技能建模为多文件的结构化能力单元，为其配备触发机制和演化元数据，并通过用户反馈驱动的闭环流程持续生成和优化技能。该框架还引入三阶段技能匹配策略和三层记忆架构，支持对复杂问题进行动态任务分解，并长期积累能力。基于真实外贸场景的实验结果表明，集成 EvoAgent 后，GPT-5.2 在专业性、准确性和实用性方面均有明显提升。在采用五个维度的 LLM-as-Judge 评测方案时，总体平均分提高约 28%。进一步的模型迁移实验表明，智能体系统的性能不仅取决于底层模型本身的能力，也取决于模型与智能体架构之间的协同程度。代码、数据和文档将在 https://github.com/Focus-AI-Center/Mentarc-EvoAgent.git 发布。

**English title:**

EvoAgent: An Evolvable Agent Framework with Skill Learning and Multi-Agent Delegation

**Abstract:**

> This paper proposes EvoAgent--an evolvable large language model (LLM) agent framework that integrates structured skill learning with a hierarchical sub-agent delegation mechanism. EvoAgent models skills as multi-file structured capability units equipped with triggering mechanisms and evolutionary metadata, and enables continuous skill generation and optimization through a user-feedback-driven closed-loop process. In addition, by incorporating a three-stage skill matching strategy and a three-layer memory architecture, the framework supports dynamic task decomposition for complex problems and long-term capability accumulation. Experimental results based on real-world foreign trade scenarios demonstrate that, after integrating EvoAgent, GPT5.2 achieves significant improvements in professionalism, accuracy, and practical utility. Under a five-dimensional LLM-as-Judge evaluation protocol, the overall average score increases by approximately 28\%. Further model transfer experiments indicate that the performance of an agent system depends not only on the intrinsic capabilities of the underlying model, but also on the degree of synergy between the model and the agent architecture. Code, data, and documents will be released at https://github.com/Focus-AI-Center/Mentarc-EvoAgent.git.

- 链接：https://arxiv.org/abs/2604.20133v3
- 年份：2026
- 发表信息：arXiv 预印本，arXiv:2604.20133v3。

#### 3. Hyperagents（超智能体）

**完整中文摘要：**

自我改进 AI 系统旨在通过学习如何改进自身的学习和问题求解过程，减少对人类工程工作的依赖。现有自我改进方法依赖固定且由人工设计的元层机制，从根本上限制了这类系统的改进速度。Darwin Gödel Machine（DGM）通过不断生成并评估自我修改后的变体，在编程任务中展现了开放式自我改进。由于评估和自我修改本身都是编程任务，编程能力的进步可以转化为自我改进能力的提升。但在编程以外的领域，这种一致性通常并不成立。我们提出 hyperagents：一种自指型智能体，它把任务智能体（负责解决目标任务）和元智能体（负责修改自身及任务智能体）整合为一个可编辑程序。尤其重要的是，元层修改流程本身也可被编辑，由此实现元认知式自我修改；系统不仅能改进任务求解行为，也能改进产生后续改进的机制。我们扩展 DGM，将这一框架实例化为 DGM-Hyperagents（DGM-H），不再要求任务表现与自我修改技能在特定领域内保持一致，从而有望支持任意可计算任务上的自加速进步。在多个不同领域中，DGM-H 的性能会随时间提高，并优于不具备自我改进或开放式探索能力的基线以及此前的自我改进系统。DGM-H 还改进了生成新智能体的流程，例如加入持久记忆和性能跟踪；这些元层改进可以跨领域迁移，并在多次运行之间累积。DGM-Hyperagents 展示了开放式 AI 系统的一种可能形态：系统不只是寻找更好的解决方案，还会持续改进自己寻找改进方法的过程。

**English title:**

Hyperagents

**Abstract:**

> Self-improving AI systems aim to reduce reliance on human engineering by learning to improve their own learning and problem-solving processes. Existing approaches to self-improvement rely on fixed, handcrafted meta-level mechanisms, fundamentally limiting how fast such systems can improve. The Darwin Gödel Machine (DGM) demonstrates open-ended self-improvement in coding by repeatedly generating and evaluating self-modified variants. Because both evaluation and self-modification are coding tasks, gains in coding ability can translate into gains in self-improvement ability. However, this alignment does not generally hold beyond coding domains. We introduce \textbf{hyperagents}, self-referential agents that integrate a task agent (which solves the target task) and a meta agent (which modifies itself and the task agent) into a single editable program. Crucially, the meta-level modification procedure is itself editable, enabling metacognitive self-modification, improving not only the task-solving behavior, but also the mechanism that generates future improvements. We instantiate this framework by extending DGM to create DGM-Hyperagents (DGM-H), eliminating the assumption of domain-specific alignment between task performance and self-modification skill to potentially support self-accelerating progress on any computable task. Across diverse domains, the DGM-H improves performance over time and outperforms baselines without self-improvement or open-ended exploration, as well as prior self-improving systems. Furthermore, the DGM-H improves the process by which it generates new agents (e.g., persistent memory, performance tracking), and these meta-level improvements transfer across domains and accumulate across runs. DGM-Hyperagents offer a glimpse of open-ended AI systems that do not merely search for better solutions, but continually improve their search for how to improve.

- 链接：https://arxiv.org/abs/2603.19461v1
- 年份：2026
- 发表信息：arXiv 预印本，arXiv:2603.19461v1。

#### 4. SIA：结合 Harness 与权重更新的自我改进 AI

**完整中文摘要：**

人类是构建和改进 AI 的瓶颈。无论模型本身，还是封装模型的智能体，都由人来编写、调优和纠正。AI 如何自行找出改进自身的方法，仍是一个长期且尚未解决的目标。当前有两条基本相互独立的研究路线试图突破这一瓶颈。Harness 更新路线让元智能体重写特定任务智能体的外层框架，包括工具、提示词、重试逻辑和搜索流程，同时保持模型权重不变。测试时训练路线则采用人工编写的强化学习流程，根据任务反馈更新模型自身的权重，同时保持 Harness 不变。这两条路线彼此割裂。我们提出 SIA，这是一种自我改进循环，其中语言模型智能体 Feedback-Agent 会同时更新特定任务智能体的 Harness 和权重。我们在三个差异明显的领域进行评测：中国法律罪名分类、底层 GPU 内核优化和单细胞 RNA 去噪。三个基准上的结果均表明，同时使用这两种更新手段优于只迭代智能体框架。与此前的最佳结果相比，SIA-W+H 在 LawBench 上提升 25.1%，生成的 GPU 内核快 12.4%（1,017 μs 对 1,161 μs），在去噪任务上提升 20.4%。Harness 更新赋予模型智能体式能力，决定它如何搜索和行动；权重更新则建立仅靠提示词或脚手架无法灌输的领域直觉。

**English title:**

SIA: Self Improving AI with Harness & Weight Updates

**Abstract:**

> Humans are the bottleneck in building and improving AI. Both the models and the agents that wrap them are written, tuned, and corrected by people. The long-horizon goal of an AI that can figure out how to improve itself remains open. Two largely disjoint research lines attack this bottleneck. The harness-update school has a meta-agent rewrite the scaffold of a task-specific agent (its tools, prompts, retry logic, and search procedure) while the model weights are held fixed. The test-time training school uses hand-written RL pipelines to update the model's own weights on task feedback while the harness is held fixed. These two silos operate in isolation. We propose SIA, a self-improving loop in which a language-model agent (the Feedback-Agent) updates both the harness and the weights of a task-specific agent. We evaluate across three contrasting domains: Chinese legal charge classification, low-level GPU kernel optimisation, and single-cell RNA denoising. Combining both levers outperforms scaffold iteration alone on all three benchmarks. SIA-W+H achieves 25.1% over prior SOTA on LawBench, 12.4% faster GPU kernels than prior SOTA (1,017 vs 1,161 μs), and 20.4% over prior SOTA on denoising. Harness updates make the model agentic, shaping how it searches and acts, while weight updates build the domain intuition that no prompt or scaffold can instil.

- 链接：https://arxiv.org/abs/2605.27276v2
- 年份：2026
- 发表信息：arXiv 预印本，arXiv:2605.27276v2。

#### 5. SkillRise：用于跨任务技能演化的智能体强化学习

**完整中文摘要：**

大型语言模型智能体经常遇到彼此相关但并不相同的任务，这些任务共享可复用的求解模式。然而，标准的智能体强化学习将各项任务视为彼此独立的回合；现有技能学习方法则要么聚焦于同一任务的重复尝试，要么采用多阶段流水线，把提取、检索和执行混杂在一起。我们提出 SkillRise，这是一个用于跨任务学习技能的统一强化学习框架。SkillRise 将相关样例组织成难度逐步增加的序列，并使用单一策略在两种工作之间交替：求解任务，以及维护一份持续演化并直接传递给下一项任务的技能文档。跨任务解耦的信用分配机制使用当前任务的结果监督求解过程，并使用经过折扣的下游结果监督技能维护过程。在 ALFWorld、WebShop 和 ScienceWorld 上的实验表明，SkillRise 在所比较的方法中取得了最高的 Pass@1，领先最强基线 2.3 至 8.5 个百分点。尽管它是在不同任务之间训练的，所学到的维护策略对同一任务的重复尝试依然有效。进一步分析发现，跨任务条件下存在测试时扩展现象：即使每项任务只尝试一次，性能也会随着相关任务序列变长而提高。这一趋势表明，SkillRise 的收益来自跨任务复用可迁移技能，而不是对同一任务反复采样。与多阶段技能学习流水线相比，SkillRise 在保持较强性能的同时，大幅降低了运行时开销。综合来看，这些结果为 LLM 智能体跨任务提取、改进和复用可迁移技能提供了一种简单而高效的训练范式。

**English title:**

SkillRise: Agentic Reinforcement Learning for Cross-Task Skill Evolution

**Abstract:**

> Large language model agents often encounter related yet distinct tasks that share reusable solution patterns. Yet standard agentic reinforcement learning treats tasks as independent episodes, while existing approaches to skill learning either focus on repeated attempts of one task or use pipelines with multiple stages that entangle extraction, retrieval, and execution. We introduce SkillRise, a unified reinforcement learning framework for learning skills across tasks. SkillRise organizes related instances into progressively challenging sequences and uses a single policy to alternate between task solving and curating an evolving skill document passed directly to the next task. Decoupled credit assignment across tasks supervises solving with the current task outcome and curation with discounted downstream outcomes. Experiments on ALFWorld, WebShop, and ScienceWorld show that SkillRise achieves the strongest Pass@1 performance among the compared methods, with gains over the strongest baseline ranging from 2.3 to 8.5 percentage points. Although trained across distinct tasks, its learned curation policy remains effective for repeated attempts on the same task. Further analysis reveals scaling at test time across tasks: performance improves with longer sequences of related tasks even when each task is attempted only once. This trend suggests that SkillRise reuses transferable skills across tasks rather than benefiting from repeated sampling of the same task. SkillRise further retains strong performance while substantially reducing the runtime overhead of skill learning pipelines with multiple stages. Together, these results provide a simple and efficient training paradigm for LLM agents to extract, refine, and reuse transferable skills across tasks.

- 链接：https://arxiv.org/abs/2607.26784v1
- 年份：2026
- 发表信息：arXiv 预印本，arXiv:2607.26784v1。

#### 6. Alita-G：面向智能体生成的自演化生成式智能体

**完整中文摘要：**

研究表明，为大型语言模型（LLM）配置记忆、工具和反馈，并将其组织成智能体，可以改善模型表现。在此基础上，自演化智能体随之出现，但现有工作大多将适应过程限制在提示词重写或失败重试。为此，我们提出 ALITA-G：一种自演化框架，通过系统地生成、抽象和维护模型上下文协议（MCP）工具，将通用智能体转变为领域专家。在该框架中，通用智能体执行一组经过筛选的目标领域任务，并从成功轨迹中合成候选 MCP。随后，这些 MCP 被抽象为参数化原语，并整合进 MCP Box。推理时，ALITA-G 会借助每个工具的描述和用例，通过检索增强机制选择 MCP，然后运行配有 MCP Executor 的智能体。在 GAIA、PathVQA 和 Humanity's Last Exam 等多个基准上，ALITA-G 在减少计算成本的同时取得明显提升。在 GAIA 验证集上，它达到 83.03% 的 pass@1 和 89.09% 的 pass@3，取得新的最佳结果；与一个强基线智能体相比，每个样例的平均 token 数减少约 15%。因此，ALITA-G 为通用能力向可复用的领域专门能力转化提供了一条系统化路径，并提高了复杂推理任务的准确率和效率。

**English title:**

Alita-G: Self-Evolving Generative Agent for Agent Generation

**Abstract:**

> Large language models (LLMs) have been shown to perform better when scaffolded into agents with memory, tools, and feedback. Beyond this, self-evolving agents have emerged, but current work largely limits adaptation to prompt rewriting or failure retries. Therefore, we present ALITA-G, a self-evolution framework that transforms a general-purpose agent into a domain expert by systematically generating, abstracting, and curating Model Context Protocol (MCP) tools. In this framework, a generalist agent executes a curated suite of target-domain tasks and synthesizes candidate MCPs from successful trajectories. These are then abstracted to parameterized primitives and consolidated into an MCP Box. At inference time, ALITA-G performs retrieval-augmented MCP selection with the help of each tool's descriptions and use cases, before executing an agent equipped with the MCP Executor. Across several benchmarks GAIA, PathVQA, and Humanity's Last Exam, ALITA-G attains strong gains while reducing computation costs. On GAIA validation, it achieves 83.03% pass@1 and 89.09% pass@3, establishing a new state-of-the-art result while reducing mean tokens per example by approximately 15% relative to a strong baseline agent. ALITA-G thus provides a principled pathway from generalist capability to reusable, domain-specific competence, improving both accuracy and efficiency on complex reasoning tasks.

- 链接：https://arxiv.org/abs/2510.23601v1
- 年份：2025
- 发表信息：arXiv 预印本，arXiv:2510.23601v1。

#### 7. EvoAgentX：自动演化智能体工作流的框架

**完整中文摘要：**

多智能体系统（MAS）已成为一种常用范式，用于编排大型语言模型（LLM）和专用工具，使其协作处理复杂任务。不过，现有 MAS 框架往往需要手动配置工作流，并且原生设计通常不支持动态演化和性能优化。许多 MAS 优化算法也没有被整合进统一框架。本文提出 EvoAgentX，这是一个开源平台，可自动完成多智能体工作流的生成、执行和演化优化。EvoAgentX 采用模块化架构，由基础组件层、智能体层、工作流层、演化层和评估层五个核心层组成。演化层集成了三种 MAS 优化算法：TextGrad、AFlow 和 MIPRO，用于迭代改进智能体提示词、工具配置和工作流拓扑。我们分别使用 HotPotQA、MBPP 和 MATH 评估多跳推理、代码生成和数学问题求解能力，并通过 GAIA 进一步评估真实任务表现。实验结果显示，EvoAgentX 的性能持续得到改善：HotPotQA 的 F1 提高 7.44%，MBPP 的 pass@1 提高 10.00%，MATH 的求解准确率提高 10.00%，GAIA 的总体准确率最高提高 20.00%。源代码见 https://github.com/EvoAgentX/EvoAgentX。

**English title:**

EvoAgentX: An Automated Framework for Evolving Agentic Workflows

**Abstract:**

> Multi-agent systems (MAS) have emerged as a powerful paradigm for orchestrating large language models (LLMs) and specialized tools to collaboratively address complex tasks. However, existing MAS frameworks often require manual workflow configuration and lack native support for dynamic evolution and performance optimization. In addition, many MAS optimization algorithms are not integrated into a unified framework. In this paper, we present EvoAgentX, an open-source platform that automates the generation, execution, and evolutionary optimization of multi-agent workflows. EvoAgentX employs a modular architecture consisting of five core layers: the basic components, agent, workflow, evolving, and evaluation layers. Specifically, within the evolving layer, EvoAgentX integrates three MAS optimization algorithms, TextGrad, AFlow, and MIPRO, to iteratively refine agent prompts, tool configurations, and workflow topologies. We evaluate EvoAgentX on HotPotQA, MBPP, and MATH for multi-hop reasoning, code generation, and mathematical problem solving, respectively, and further assess it on real-world tasks using GAIA. Experimental results show that EvoAgentX consistently achieves significant performance improvements, including a 7.44% increase in HotPotQA F1, a 10.00% improvement in MBPP pass@1, a 10.00% gain in MATH solve accuracy, and an overall accuracy improvement of up to 20.00% on GAIA. The source code is available at: https://github.com/EvoAgentX/EvoAgentX

- 链接：https://arxiv.org/abs/2507.03616v2
- 年份：2025
- 发表信息：arXiv 预印本，arXiv:2507.03616v2。

#### 8. Gödel Agent：用于递归自我改进的自指智能体框架

**完整中文摘要：**

大型语言模型（LLM）的快速发展显著增强了 AI 智能体在多种任务上的能力。然而，无论现有智能体系统采用固定的流水线算法，还是采用预先定义的元学习框架，它们都会受到人工设计组件的限制，无法搜索完整的智能体设计空间，因此可能错过全局最优的智能体设计。本文提出 Gödel Agent，这是一种受 Gödel machine 启发的自演化框架，使智能体无须依赖预先规定的程序或固定优化算法即可递归地改进自身。Gödel Agent 利用 LLM 动态修改自己的逻辑和行为，整个过程只由提示词所表达的高层目标引导。数学推理和复杂智能体任务上的实验结果表明，Gödel Agent 的实现能够持续自我改进，并在性能、效率和泛化能力上超过人工设计的智能体。

**English title:**

Gödel Agent: A Self-Referential Agent Framework for Recursive Self-Improvement

**Abstract:**

> The rapid advancement of large language models (LLMs) has significantly enhanced the capabilities of AI-driven agents across various tasks. However, existing agentic systems, whether based on fixed pipeline algorithms or pre-defined meta-learning frameworks, cannot search the whole agent design space due to the restriction of human-designed components, and thus might miss the globally optimal agent design. In this paper, we introduce Gödel Agent, a self-evolving framework inspired by the Gödel machine, enabling agents to recursively improve themselves without relying on predefined routines or fixed optimization algorithms. Gödel Agent leverages LLMs to dynamically modify its own logic and behavior, guided solely by high-level objectives through prompting. Experimental results on mathematical reasoning and complex agent tasks demonstrate that implementation of Gödel Agent can achieve continuous self-improvement, surpassing manually crafted agents in performance, efficiency, and generalizability.

- 链接：https://arxiv.org/abs/2410.04444v4
- 年份：2025
- 发表信息：ACL 2025；arXiv:2410.04444v4（初稿提交于 2024 年）。

#### 9. MemEvolve：智能体记忆系统的元演化

**完整中文摘要：**

自演化记忆系统正在以前所未有的方式改变基于大型语言模型（LLM）的智能体所采用的演化范式。此前的研究大多依赖人工设计的记忆架构来存储轨迹、提炼经验并合成可复用工具，使智能体能够在与环境交互的过程中即时演化。然而，这一范式从根本上受到记忆系统自身静态性的限制：记忆虽然促成了智能体层面的演化，底层记忆架构却无法根据不同任务情境进行元适应。为解决这一问题，我们提出 MemEvolve。该元演化框架会同时演化智能体的经验性知识和记忆架构，使智能体系统既能积累经验，也能逐步改进学习经验的方式。为了让 MemEvolve 建立在既有研究之上，并促进未来自演化系统的开放性，我们还推出 EvolveLab：一个统一的自演化记忆代码库。它将 12 种代表性记忆系统提炼为由编码、存储、检索和管理构成的模块化设计空间，既提供标准化实现基础，也提供公平的实验平台。在四个具有挑战性的智能体基准上开展的大量评测表明，MemEvolve 取得两方面结果：（I）性能明显提升，SmolAgent、Flash-Searcher 等框架的提升最高达到 17.06%；（II）具备较强的跨任务和跨 LLM 泛化能力，所设计的记忆架构可有效迁移到不同基准和骨干模型。

**English title:**

MemEvolve: Meta-Evolution of Agent Memory Systems

**Abstract:**

> Self-evolving memory systems are unprecedentedly reshaping the evolutionary paradigm of large language model (LLM)-based agents. Prior work has predominantly relied on manually engineered memory architectures to store trajectories, distill experience, and synthesize reusable tools, enabling agents to evolve on the fly within environment interactions. However, this paradigm is fundamentally constrained by the staticity of the memory system itself: while memory facilitates agent-level evolving, the underlying memory architecture cannot be meta-adapted to diverse task contexts. To address this gap, we propose MemEvolve, a meta-evolutionary framework that jointly evolves agents' experiential knowledge and their memory architecture, allowing agent systems not only to accumulate experience but also to progressively refine how they learn from it. To ground MemEvolve in prior research and foster openness in future self-evolving systems, we introduce EvolveLab, a unified self-evolving memory codebase that distills twelve representative memory systems into a modular design space (encode, store, retrieve, manage), providing both a standardized implementation substrate and a fair experimental arena. Extensive evaluations on four challenging agentic benchmarks demonstrate that MemEvolve achieves (I) substantial performance gains, improving frameworks such as SmolAgent and Flash-Searcher by up to $17.06\%$; and (II) strong cross-task and cross-LLM generalization, designing memory architectures that transfer effectively across diverse benchmarks and backbone models.

- 链接：https://arxiv.org/abs/2512.18746v1
- 年份：2025
- 发表信息：arXiv 预印本，arXiv:2512.18746v1。

#### 10. 具备反思与记忆增强能力的自演化智能体

**完整中文摘要：**

大型语言模型（LLM）在自然语言处理领域取得了显著进展，但仍面临持续决策等问题。本研究提出一个新框架，将迭代反馈、反思机制以及基于艾宾浩斯遗忘曲线的记忆优化机制结合起来，从而显著增强智能体处理多任务和长跨度信息的能力。

**English title:**

Self-evolving Agents with reflective and memory-augmented abilities

**Abstract:**

> Large language models (LLMs) have made significant advances in the field of natural language processing, but they still face challenges such as continuous decision-making. In this research, we propose a novel framework by integrating iterative feedback, reflective mechanisms, and a memory optimization mechanism based on the Ebbinghaus forgetting curve, it significantly enhances the agents' capabilities in handling multi-tasking and long-span information.

- 链接：https://arxiv.org/abs/2409.00872v2
- 年份：2025
- 发表信息：Neurocomputing，2025；arXiv:2409.00872v2（初稿提交于 2024 年）。

#### 11. Agent-Pro：通过策略层反思与优化学习演化

**完整中文摘要：**

大型语言模型（LLM）在多种任务中表现出较强的问题求解能力。然而，大多数基于 LLM 的智能体都被设计为特定任务的求解器，并依赖复杂的提示工程，而不是能够通过交互学习和演化的智能体。这些任务求解器需要人工编写提示词来告知任务规则并约束 LLM 的行为，因此本质上缺乏处理复杂动态场景的能力，例如大型交互式游戏。基于这一问题，我们提出 Agent-Pro：一种具备策略层反思与优化能力的 LLM 智能体，可以从交互经验中学习大量专门知识，并逐步提升其行为策略。具体而言，它通过动态信念生成和反思过程实现策略演化。Agent-Pro 不进行动作层反思，而是迭代反思过去的轨迹和信念，调整不合理的信念以获得更好的策略。它还采用深度优先搜索优化策略，确保策略收益持续提高。我们在二十一点和德州扑克两款游戏上评估 Agent-Pro，其表现超过原生 LLM 和专用模型。结果表明，Agent-Pro 能够在复杂动态场景中学习和演化，这一能力也有益于许多基于 LLM 的应用。

**English title:**

Agent-Pro: Learning to Evolve via Policy-Level Reflection and Optimization

**Abstract:**

> Large Language Models (LLMs) exhibit robust problem-solving capabilities for diverse tasks. However, most LLM-based agents are designed as specific task solvers with sophisticated prompt engineering, rather than agents capable of learning and evolving through interactions. These task solvers necessitate manually crafted prompts to inform task rules and regulate LLM behaviors, inherently incapacitating to address complex dynamic scenarios e.g., large interactive games. In light of this, we propose Agent-Pro: an LLM-based Agent with Policy-level Reflection and Optimization that can learn a wealth of expertise from interactive experiences and progressively elevate its behavioral policy. Specifically, it involves a dynamic belief generation and reflection process for policy evolution. Rather than action-level reflection, Agent-Pro iteratively reflects on past trajectories and beliefs, fine-tuning its irrational beliefs for a better policy. Moreover, a depth-first search is employed for policy optimization, ensuring continual enhancement in policy payoffs. Agent-Pro is evaluated across two games: Blackjack and Texas Hold'em, outperforming vanilla LLM and specialized models. Our results show Agent-Pro can learn and evolve in complex and dynamic scenes, which also benefits numerous LLM-based applications.

- 链接：https://arxiv.org/abs/2402.17574v3
- 年份：2024
- 发表信息：ACL 2024；arXiv:2402.17574v3。

#### 12. Voyager：基于大型语言模型的开放式具身智能体

**完整中文摘要：**

我们提出 Voyager，这是首个由 LLM 驱动、用于 Minecraft 的具身终身学习智能体。它无须人类干预即可持续探索世界、习得多种技能并作出新的发现。Voyager 包含三个主要组件：1）用于最大化探索程度的自动课程；2）不断扩展的可执行代码技能库，用于存储和检索复杂行为；3）一种新的迭代式提示机制，将环境反馈、执行错误和自我验证纳入程序改进过程。Voyager 通过黑盒查询与 GPT-4 交互，因此无须微调模型参数。Voyager 学到的技能具有时间延展性、可解释性和可组合性，这使智能体的能力得以快速叠加，并缓解灾难性遗忘。实验表明，Voyager 具备较强的上下文终身学习能力，在 Minecraft 中表现出很高的熟练度。与此前最佳方法相比，它获得的独特物品数量达到 3.3 倍，移动距离达到 2.3 倍，解锁科技树关键里程碑的速度最高达到 15.3 倍。Voyager 能够在一个新的 Minecraft 世界中利用学到的技能库，从零开始解决新任务，而其他方法很难实现泛化。我们的完整代码库和提示词已在 https://voyager.minedojo.org/ 开源。

**English title:**

Voyager: An Open-Ended Embodied Agent with Large Language Models

**Abstract:**

> We introduce Voyager, the first LLM-powered embodied lifelong learning agent in Minecraft that continuously explores the world, acquires diverse skills, and makes novel discoveries without human intervention. Voyager consists of three key components: 1) an automatic curriculum that maximizes exploration, 2) an ever-growing skill library of executable code for storing and retrieving complex behaviors, and 3) a new iterative prompting mechanism that incorporates environment feedback, execution errors, and self-verification for program improvement. Voyager interacts with GPT-4 via blackbox queries, which bypasses the need for model parameter fine-tuning. The skills developed by Voyager are temporally extended, interpretable, and compositional, which compounds the agent's abilities rapidly and alleviates catastrophic forgetting. Empirically, Voyager shows strong in-context lifelong learning capability and exhibits exceptional proficiency in playing Minecraft. It obtains 3.3x more unique items, travels 2.3x longer distances, and unlocks key tech tree milestones up to 15.3x faster than prior SOTA. Voyager is able to utilize the learned skill library in a new Minecraft world to solve novel tasks from scratch, while other techniques struggle to generalize. We open-source our full codebase and prompts at https://voyager.minedojo.org/.

- 链接：https://arxiv.org/abs/2305.16291v2
- 年份：2024
- 发表信息：Transactions on Machine Learning Research（TMLR），2024；arXiv:2305.16291v2（初稿提交于 2023 年）。

## 多智能体自我改进 / Multi-Agent Self-Improvement

利用多个智能体之间的交互来改进推理、策略或智能体群体的系统。

*Systems that use interaction among multiple agents to improve reasoning, policies, or agent populations.*

### 协同进化 / Co-Evolution

#### 1. Agent0：通过工具集成推理，在零数据条件下释放自演化智能体

**完整中文摘要：**

大型语言模型（LLM）智能体通常使用强化学习（RL）训练，但它们依赖人工整理的数据，这既限制了可扩展性，也使人工智能受制于人类知识。现有的自演化框架提供了另一种选择，但通常受限于模型的固有能力和单轮交互方式，因而难以构建涉及工具使用或动态推理的复杂课程。我们提出 Agent0：一个完全自主的框架，它无需外部数据，通过多步协同进化和无缝工具集成来演化出高性能智能体。Agent0 在两个由同一基础 LLM 初始化的智能体之间建立了一种共生竞争关系：课程智能体提出难度不断增加的前沿任务，执行智能体则学习解决这些任务。我们引入外部工具来提升执行智能体解决问题的能力；这种提升反过来促使课程智能体构造更复杂、更强调工具使用的任务。通过这一迭代过程，Agent0 建立了一个持续生成高质量课程的自我强化循环。实验表明，Agent0 显著提升了推理能力：Qwen3-8B-Base 模型在数学推理基准上的表现提高了 18%，在通用推理基准上的表现提高了 24%。代码见 https://github.com/aiming-lab/Agent0。

**English title:**

Agent0: Unleashing Self-Evolving Agents from Zero Data via Tool-Integrated Reasoning

**Abstract:**

Large Language Model (LLM) Agents, often trained with Reinforcement Learning (RL), are constrained by a dependency on human-curated data, limiting scalability and tethering AI to human knowledge. Existing self-evolution frameworks offer an alternative but are typically restricted by the model's inherent capabilities and single-round interactions, hindering the development of complex curricula involving tool use or dynamic reasoning. We introduce Agent0, a fully autonomous framework that evolves high-performing agents without external data through multi-step co-evolution and seamless tool integration. Agent0 establishes a symbiotic competition between two agents initialized from the same base LLM: a curriculum agent that proposes increasingly challenging frontier tasks, and an executor agent that learns to solve them. We integrate external tools to enhance the executor's problem-solving capacity; this improvement, in turn, pressures the curriculum agent to construct more complex, tool-aware tasks. Through this iterative process, Agent0 establishes a self-reinforcing cycle that continuously produces high-quality curricula. Empirically, Agent0 substantially boosts reasoning capabilities, improving the Qwen3-8B-Base model by 18% on mathematical reasoning and 24% on general reasoning benchmarks. Code is available at https://github.com/aiming-lab/Agent0.

- 链接：[arXiv:2511.16043v1](https://arxiv.org/abs/2511.16043v1)
- 年份：2025
- 发表信息：arXiv 预印本，`arXiv:2511.16043v1 [cs.LG]`；2025 年 11 月 20 日提交。arXiv 记录未注明正式会议或期刊发表信息。

#### 2. 辩论、训练、进化：语言模型推理的自演化

**完整中文摘要：**

大型语言模型（LLM）通过在海量数据上进行大规模训练，推理能力已有显著提高。然而，仅依靠增加数据来继续改进正变得越来越不切实际，因此需要让模型在没有外部监督的情况下自主提升推理能力。本文提出 Debate, Train, Evolve（DTE），一种无需真实答案标签的新型训练框架，它利用多智能体辩论轨迹来演化单个语言模型。我们还提出一种新的提示策略 Reflect-Critique-Refine，通过明确要求智能体批评并改进其推理过程来提高辩论质量。我们使用 6 个开放权重模型，在 7 个推理基准上进行了广泛评估。结果显示，DTE 框架带来了明显改进，在具有挑战性的 GSM-PLUS 数据集上平均准确率提高 8.92%。我们还观察到较强的跨领域泛化能力：在其余所有基准上，平均准确率提高 5.8%，说明该方法学到了通用推理能力。框架代码和训练后的模型已公开于 https://github.com/ctrl-gaurav/Debate-Train-Evolve。

**English title:**

DEBATE, TRAIN, EVOLVE: Self Evolution of Language Model Reasoning

**Abstract:**

Large language models (LLMs) have improved significantly in their reasoning through extensive training on massive datasets. However, relying solely on additional data for improvement is becoming increasingly impractical, highlighting the need for models to autonomously enhance their reasoning without external supervision. In this paper, we propose Debate, Train, Evolve (DTE), a novel ground truth-free training framework that uses multi-agent debate traces to evolve a single language model. We also introduce a new prompting strategy Reflect-Critique-Refine, to improve debate quality by explicitly instructing agents to critique and refine their reasoning. Extensive evaluations on seven reasoning benchmarks with six open-weight models show that our DTE framework achieve substantial improvements, with an average accuracy gain of 8.92% on the challenging GSM-PLUS dataset. Furthermore, we observe strong cross-domain generalization, with an average accuracy gain of 5.8% on all other benchmarks, suggesting that our method captures general reasoning capabilities. Our framework code and trained models are publicly available at https://github.com/ctrl-gaurav/Debate-Train-Evolve

- 链接：[arXiv:2505.15734v2](https://arxiv.org/abs/2505.15734v2)
- 年份：2025
- 发表信息：EMNLP 2025 主会（Main Conference）录用；arXiv 当前版本为 `2505.15734v2 [cs.CL, cs.AI, cs.LG]`，初稿提交于 2025 年 5 月 21 日，v2 更新于 2025 年 9 月 30 日。

#### 3. EvoAgent：面向基于进化算法的自动多智能体生成

**完整中文摘要：**

强大的大型语言模型（LLM）兴起后，构建基于 LLM、用于解决复杂任务的自主智能体成为新的研究趋势，多智能体系统尤其受到关注。尽管相关研究已取得明显进展，我们注意到，现有工作严重依赖人工设计的框架，这极大限制了智能体系统的功能范围和可扩展性。如何自动地将专用智能体扩展为多智能体系统，以提高任务求解能力，仍是一项重要挑战。本文提出 EvoAgent：一种通过进化算法自动将专用智能体扩展为多智能体系统的通用方法，从而提高基于 LLM 的智能体解决任务的效果。具体来说，我们将现有智能体框架视为初始个体，再应用一系列进化算子（如变异、交叉和选择等），生成具有不同设置的多个智能体。多种任务上的实验结果表明，EvoAgent 能显著提升基于 LLM 的智能体的任务求解能力，并且可以推广到任意基于 LLM 的智能体框架，将其扩展为多智能体系统。相关资源见 https://evo-agent.github.io/。

**English title:**

EvoAgent: Towards Automatic Multi-Agent Generation via Evolutionary Algorithms

**Abstract:**

The rise of powerful large language models (LLMs) has spurred a new trend in building LLM-based autonomous agents for solving complex tasks, especially multi-agent systems. Despite the remarkable progress, we notice that existing works are heavily dependent on human-designed frameworks, which greatly limits the functional scope and scalability of agent systems. How to automatically extend the specialized agent to multi-agent systems to improve task-solving capability still remains a significant challenge. In this paper, we introduce EvoAgent, a generic method to automatically extend specialized agents to multi-agent systems via the evolutionary algorithm, thereby improving the effectiveness of LLM-based agents in solving tasks. Specifically, we consider the existing agent frameworks as the initial individual and then apply a series of evolutionary operators (e.g., mutation, crossover, selection, etc.) to generate multiple agents with diverse settings. Experimental results across various tasks show that EvoAgent can significantly enhance the task-solving capability of LLM-based agents, and can be generalized to any LLM-based agent framework to extend them into multi-agent systems. Resources are available at https://evo-agent.github.io/.

- 链接：[arXiv:2406.14228v3](https://arxiv.org/abs/2406.14228v3)
- 年份：2025
- 发表信息：NAACL 2025 长论文，收录于 *Proceedings of the 2025 Conference of the Nations of the Americas Chapter of the Association for Computational Linguistics: Human Language Technologies (Volume 1: Long Papers)*，第 6192–6217 页；DOI：[10.18653/v1/2025.naacl-long.315](https://doi.org/10.18653/v1/2025.naacl-long.315)。arXiv 初稿提交于 2024 年，当前为 v3（2025 年 3 月 10 日）。

#### 4. SOTOPIA-π：社会智能语言智能体的交互式学习

**完整中文摘要：**

人类既通过模仿，也通过社会互动来学习社交技能。现有的语言智能体研究很少探讨这一社会学习过程。针对这一空白，我们提出交互式学习方法 SOTOPIA-π，用于提高语言智能体的社会智能。该方法根据大型语言模型（LLM）的评分筛选社会互动数据，并在这些数据上采用行为克隆和自强化训练。我们证明，这种训练方法能让一个 7B 参数的 LLM 达到专家模型（基于 GPT-4 的智能体）的社会目标完成能力，同时提高语言智能体的安全性，并保持其在 MMLU 基准上的通用问答能力。我们还发现，这一训练范式揭示了使用 LLM 评估社会智能时存在的一个问题：基于 LLM 的评估器会高估那些专门针对社会互动训练的语言智能体的能力。

**English title:**

SOTOPIA-π: Interactive Learning of Socially Intelligent Language Agents

**Abstract:**

Humans learn social skills through both imitation and social interaction. This social learning process is largely understudied by existing research on building language agents. Motivated by this gap, we propose an interactive learning method, SOTOPIA-$π$, improving the social intelligence of language agents. This method leverages behavior cloning and self-reinforcement training on filtered social interaction data according to large language model (LLM) ratings. We show that our training method allows a 7B LLM to reach the social goal completion ability of an expert model (GPT-4-based agent), while improving the safety of language agents and maintaining general QA ability on the MMLU benchmark. We also find that this training paradigm uncovers some difficulties in LLM-based evaluation of social intelligence: LLM-based evaluators overestimate the abilities of the language agents trained specifically for social interaction.

- 链接：[arXiv:2403.08715v3](https://arxiv.org/abs/2403.08715v3)
- 年份：2024
- 发表信息：ACL 2024 长论文，收录于 *Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)*，第 12912–12940 页；DOI：[10.18653/v1/2024.acl-long.698](https://doi.org/10.18653/v1/2024.acl-long.698)。

### 推理时辩论 / Inference-time Debate

主要通过多智能体辩论改进当前一次运行的结果，不产生持久的系统变更。

*Primarily current-run improvement via multi-agent debate, without persistent system change.*

#### 5. 通过多智能体辩论鼓励大型语言模型进行发散思考

**完整中文摘要：**

ChatGPT 等现代大型语言模型（LLM）在通用语言任务上表现出色，但处理复杂推理任务时仍有困难。这促使研究者考察 LLM 的认知行为，并探索类似人类的问题求解策略。在这一方向上，一种代表性策略是自我反思：让 LLM 根据自己生成的反馈，迭代地改进答案。然而，我们的研究表明，这类反思方法存在“思维退化”（Degeneration-of-Thought，DoT）问题：一旦 LLM 对自己的答案建立了信心，即使最初的立场是错误的，它也无法在后续反思中产生新的思路。为解决 DoT 问题，我们提出多智能体辩论（Multi-Agent Debate，MAD）框架。在该框架中，多个智能体以“针锋相对”的方式陈述论点，并由一名裁判管理辩论过程、得出最终答案。MAD 框架鼓励 LLM 进行发散思考，这有助于完成需要深入思考的任务。在两个具有挑战性的数据集，即常识机器翻译和反直觉算术推理上的实验，验证了 MAD 框架的效果。进一步分析表明，要让 MAD 取得良好表现，需要采用自适应的辩论终止机制，并将“针锋相对”的程度控制在适中水平。我们还发现，当不同 LLM 分别充当智能体时，LLM 可能无法担任公平的裁判。代码见 https://github.com/Skytliang/Multi-Agents-Debate。

**English title:**

Encouraging Divergent Thinking in Large Language Models through Multi-Agent Debate

**Abstract:**

Modern large language models (LLMs) like ChatGPT have shown remarkable performance on general language tasks but still struggle on complex reasoning tasks, which drives the research on cognitive behaviors of LLMs to explore human-like problem-solving strategies. Along this direction, one representative strategy is self-reflection, which asks an LLM to refine the solution with the feedback generated by itself iteratively. However, our study shows that such reflection-style methods suffer from the Degeneration-of-Thought (DoT) problem: once the LLM has established confidence in its solutions, it is unable to generate novel thoughts later through reflection even if its initial stance is incorrect. To address the DoT problem, we propose a Multi-Agent Debate (MAD) framework, in which multiple agents express their arguments in the state of "tit for tat" and a judge manages the debate process to obtain a final solution. Clearly, our MAD framework encourages divergent thinking in LLMs which would be helpful for tasks that require deep levels of contemplation. Experiment results on two challenging datasets, commonsense machine translation and counter-intuitive arithmetic reasoning, demonstrate the effectiveness of our MAD framework. Extensive analyses suggest that the adaptive break of debate and the modest level of "tit for tat" state are required for MAD to obtain good performance. Moreover, we find that LLMs might not be a fair judge if different LLMs are used for agents. Code is available at https://github.com/Skytliang/Multi-Agents-Debate.

- 链接：[arXiv:2305.19118v4](https://arxiv.org/abs/2305.19118v4)
- 年份：2024
- 发表信息：EMNLP 2024 主会论文，收录于 *Proceedings of the 2024 Conference on Empirical Methods in Natural Language Processing*，第 17889–17904 页；DOI：[10.18653/v1/2024.emnlp-main.992](https://doi.org/10.18653/v1/2024.emnlp-main.992)。arXiv 初稿提交于 2023 年，当前为 v4（2024 年 10 月 9 日）。

#### 6. 通过多智能体辩论提升语言模型的事实性与推理能力

**完整中文摘要：**

近年来，大型语言模型（LLM）在语言生成、语言理解和少样本学习方面表现出很强的能力。大量研究探讨了如何借助提示方法进一步提升其性能，包括验证、自洽性和中间草稿等。本文提出一种互补的语言回答改进方法：多个语言模型实例在多轮交互中分别提出并辩论各自的答案与推理过程，最终形成共同答案。研究结果表明，该方法能在多项任务上显著提升数学推理和策略推理能力。我们还证明，该方法可以提高生成内容的事实有效性，减少当代模型容易产生的谬误答案和幻觉。该方法可以直接用于现有黑盒模型，并且在研究涉及的所有任务中采用相同的流程和提示。总体而言，研究结果表明，这种“心智社会”方法有望显著提高 LLM 的能力，并推动语言生成与理解取得进一步进展。

**English title:**

Improving Factuality and Reasoning in Language Models through Multiagent Debate

**Abstract:**

Large language models (LLMs) have demonstrated remarkable capabilities in language generation, understanding, and few-shot learning in recent years. An extensive body of work has explored how their performance may be further improved through the tools of prompting, ranging from verification, self-consistency, or intermediate scratchpads. In this paper, we present a complementary approach to improve language responses where multiple language model instances propose and debate their individual responses and reasoning processes over multiple rounds to arrive at a common final answer. Our findings indicate that this approach significantly enhances mathematical and strategic reasoning across a number of tasks. We also demonstrate that our approach improves the factual validity of generated content, reducing fallacious answers and hallucinations that contemporary models are prone to. Our approach may be directly applied to existing black-box models and uses identical procedure and prompts for all tasks we investigate. Overall, our findings suggest that such "society of minds" approach has the potential to significantly advance the capabilities of LLMs and pave the way for further breakthroughs in language generation and understanding.

- 链接：[arXiv:2305.14325v1](https://arxiv.org/abs/2305.14325v1)
- 年份：2024
- 发表信息：第 41 届国际机器学习大会（ICML 2024）论文，收录于 *Proceedings of Machine Learning Research*，第 235 卷，第 11733–11763 页；正式论文页：[PMLR v235/du24e](https://proceedings.mlr.press/v235/du24e.html)。arXiv 版本于 2023 年 5 月 23 日提交。

## 编程与软件工程自我改进 / Coding / Software-Engineering Self-Improvement

改进代码能力、软件工程表现或自身实现的智能体及训练循环。

Agents and training loops that improve code, software-engineering performance, or their own implementations.

### 自修改编程智能体 / Self-Modifying Coding Agents

#### 1. 达尔文哥德尔机：自我改进智能体的开放式演化

**中文摘要：** 当今的 AI 系统采用人类设计的固定架构，无法自主、持续地改进自身。AI 的进步本身可以被自动化。如果这一过程能够安全完成，便可加快 AI 的发展，使我们更早受益。元学习能够自动发现新算法，但它受限于一阶改进以及由人类设计适当搜索空间的要求。哥德尔机提出了一种理论上的替代方案：一种反复修改自身、且每次修改均可证明有益的自我改进 AI。遗憾的是，在实践中无法证明大多数修改在总体上有益。本文提出达尔文哥德尔机（Darwin Gödel Machine, DGM）：该系统迭代修改自身代码，并由此提升修改自身代码库的能力，再用编程基准对每项修改进行经验验证。受达尔文演化与开放性研究启发，DGM 维护一个由已生成编程智能体组成的档案库。它从中抽取一个智能体，再用基础模型生成该智能体的新颖版本，从而扩充档案库。这种开放式探索会形成一棵不断生长、包含多样化高质量智能体的树，并允许系统并行探索搜索空间中的多条路径。实验表明，DGM 能自动提升编程能力，例如改进代码编辑工具、长上下文窗口管理和同行评审机制；其 SWE-bench 成绩从 20.0% 提高到 50.0%，Polyglot 成绩从 14.2% 提高到 30.7%。DGM 还显著优于不具备自我改进或开放式探索能力的基线。所有实验均采取了沙箱和人工监督等安全措施。DGM 朝自我改进 AI 迈出了重要一步，它能够自行积累垫脚石，沿不断展开的路径持续创新。

**English title:** Darwin Godel Machine: Open-Ended Evolution of Self-Improving Agents

**Abstract:** Today's AI systems have human-designed, fixed architectures and cannot autonomously and continuously improve themselves. The advance of AI could itself be automated. If done safely, that would accelerate AI development and allow us to reap its benefits much sooner. Meta-learning can automate the discovery of novel algorithms, but is limited by first-order improvements and the human design of a suitable search space. The Gödel machine proposed a theoretical alternative: a self-improving AI that repeatedly modifies itself in a provably beneficial manner. Unfortunately, proving that most changes are net beneficial is impossible in practice. We introduce the Darwin Gödel Machine (DGM), a self-improving system that iteratively modifies its own code (thereby also improving its ability to modify its own codebase) and empirically validates each change using coding benchmarks. Inspired by Darwinian evolution and open-endedness research, the DGM maintains an archive of generated coding agents. It grows the archive by sampling an agent from it and using a foundation model to create a new, interesting, version of the sampled agent. This open-ended exploration forms a growing tree of diverse, high-quality agents and allows the parallel exploration of many different paths through the search space. Empirically, the DGM automatically improves its coding capabilities (e.g., better code editing tools, long-context window management, peer-review mechanisms), increasing performance on SWE-bench from 20.0% to 50.0%, and on Polyglot from 14.2% to 30.7%. Furthermore, the DGM significantly outperforms baselines without self-improvement or open-ended exploration. All experiments were done with safety precautions (e.g., sandboxing, human oversight). The DGM is a significant step toward self-improving AI, capable of gathering its own stepping stones along paths that unfold into endless innovation.

- **链接 / Link：** [arXiv:2505.22954](https://arxiv.org/abs/2505.22954)
- **年份 / Year：** 2025（预印本首次提交）；2026（会议）
- **发表信息 / Publication：** ICLR 2026；arXiv:2505.22954

#### 2. 一种自我改进的编程智能体

**中文摘要：** 大语言模型（LLM）近来的进步激发了人们部署 LLM 智能体、让其在现实世界中执行任务的兴趣。LLM 常被部署在智能体系统中，也就是用代码编排 LLM 调用并为其提供工具。本文证明，一个配备基础编程工具的智能体系统可以自主编辑自身，从而提高基准任务表现。在 SWE-bench Verified 的一个随机子集上，其性能从 17% 提高到 53%；在 LiveCodeBench 和合成智能体基准上也取得了进一步提升。该工作推动了智能体系统的自动化开放式设计，并展示了一种由 LLM 反思和代码更新驱动、数据效率较高且不依赖梯度的学习机制。

**English title:** A Self-Improving Coding Agent

**Abstract:** Recent advancements in Large Language Models (LLMs) have spurred interest in deploying LLM agents to undertake tasks in the world. LLMs are often deployed in agent systems: code that orchestrates LLM calls and provides them with tools. We demonstrate that an agent system, equipped with basic coding tools, can autonomously edit itself, and thereby improve its performance on benchmark tasks. We find performance gains from 17% to 53% on a random subset of SWE Bench Verified, with additional performance gains on LiveCodeBench, as well as synthetically generated agent benchmarks. Our work represents an advancement in the automated and open-ended design of agentic systems, and demonstrates a data-efficient, non gradient-based learning mechanism driven by LLM reflection and code updates.

- **链接 / Link：** [arXiv:2504.15228](https://arxiv.org/abs/2504.15228)
- **年份 / Year：** 2025
- **发表信息 / Publication：** ICLR 2025 Workshop on Scientific Self-Improvement（SSI-FM）；arXiv:2504.15228

### 迭代修复与训练 / Iterative Repair & Training

此类方法兼有当前运行中的修复循环，以及可将所学改进保留到后续迭代的方法。

Mixes current-run repair loops with methods whose learned improvements persist into later iterations.

#### 3. 使用 SWE-Gym 训练软件工程智能体与验证器

**中文摘要：** 本文提出 SWE-Gym，这是首个用于训练真实世界软件工程（SWE）智能体的环境。SWE-Gym 包含 2,438 个真实 Python 任务实例，每个实例都由一个带可执行运行环境和单元测试的代码库，以及一项用自然语言说明的任务组成。研究者使用 SWE-Gym 训练基于语言模型的软件工程智能体，在常用的 SWE-Bench Verified 与 Lite 测试集上取得最高 19 个百分点的解决率绝对增益。论文还用 SWE-Gym 中采样的智能体轨迹训练验证器，以探索推理时扩展。将这些验证器与微调后的软件工程智能体结合后，系统在 SWE-Bench Verified 和 Lite 上分别达到 32.0% 和 26.0%，刷新了开放权重软件工程智能体的最佳成绩。为方便后续研究，作者公开发布了 SWE-Gym、模型和智能体轨迹。

**English title:** Training Software Engineering Agents and Verifiers with SWE-Gym

**Abstract:** We present SWE-Gym, the first environment for training real-world software engineering (SWE) agents. SWE-Gym contains 2,438 real-world Python task instances, each comprising a codebase with an executable runtime environment, unit tests, and a task specified in natural language. We use SWE-Gym to train language model based SWE agents, achieving up to 19% absolute gains in resolve rate on the popular SWE-Bench Verified and Lite test sets. We also experiment with inference-time scaling through verifiers trained on agent trajectories sampled from SWE-Gym. When combined with our fine-tuned SWE agents, we achieve 32.0% and 26.0% on SWE-Bench Verified and Lite, respectively, reflecting a new state-of-the-art for open-weight SWE agents. To facilitate further research, we publicly release SWE-Gym, models, and agent trajectories.

- **链接 / Link：** [arXiv:2412.21139](https://arxiv.org/abs/2412.21139)
- **年份 / Year：** 2024（预印本首次提交）；2025（会议）
- **发表信息 / Publication：** ICML 2025；arXiv:2412.21139

#### 4. 教大语言模型自我调试

**中文摘要：** 大语言模型（LLM）在代码生成方面已取得出色表现，但面对复杂编程任务时，一次生成正确解答仍很困难，因此已有研究设计程序修复方法来提高代码生成性能。本文提出 Self-Debugging，通过少样本示范教大语言模型调试自己预测的程序。具体而言，即使没有关于代码正确性的人工反馈或错误消息，该方法也能教会大语言模型进行“橡皮鸭调试”：模型通过检查执行结果并用自然语言解释所生成的代码来识别错误。Self-Debugging 在多个代码生成基准上取得当时最佳表现，包括文本到 SQL 生成的 Spider、C++ 到 Python 翻译的 TransCoder，以及文本到 Python 生成的 MBPP。在没有单元测试可验证预测正确性的 Spider 基准上，带代码解释的 Self-Debugging 将基线稳定提高 2–3%，并将最难级别问题的预测准确率提高 9%。在提供单元测试的 TransCoder 和 MBPP 上，它将基线准确率最高提高 12%。同时，该方法利用反馈消息并复用失败预测，显著提高样本效率，能够追平或超过生成候选程序数量多十倍以上的基线模型。

**English title:** Teaching Large Language Models to Self-Debug

**Abstract:** Large language models (LLMs) have achieved impressive performance on code generation. However, for complex programming tasks, generating the correct solution in one go becomes challenging, thus some prior works have designed program repair approaches to improve code generation performance. In this work, we propose Self-Debugging, which teaches a large language model to debug its predicted program via few-shot demonstrations. In particular, we demonstrate that Self-Debugging can teach the large language model to perform rubber duck debugging; i.e., without any human feedback on the code correctness or error messages, the model is able to identify its mistakes by investigating the execution results and explaining the generated code in natural language. Self-Debugging achieves the state-of-the-art performance on several code generation benchmarks, including the Spider dataset for text-to-SQL generation, TransCoder for C++-to-Python translation, and MBPP for text-to-Python generation. On the Spider benchmark where there are no unit tests to verify the correctness of predictions, Self-Debugging with code explanation consistently improves the baseline by 2-3%, and improves the prediction accuracy on problems of the hardest level by 9%. On TransCoder and MBPP where unit tests are available, Self-Debugging improves the baseline accuracy by up to 12%. Meanwhile, by leveraging feedback messages and reusing failed predictions, Self-Debugging notably improves sample efficiency, and can match or outperform baseline models that generate more than 10x candidate programs.

- **链接 / Link：** [arXiv:2304.05128](https://arxiv.org/abs/2304.05128)
- **年份 / Year：** 2023（预印本首次提交）；2024（会议）
- **发表信息 / Publication：** ICLR 2024；arXiv:2304.05128

#### 5. AgentCoder：基于多智能体、迭代测试与优化的代码生成

**中文摘要：** 基于 Transformer 的大语言模型（LLM）推动了自然语言处理的发展。这些模型也改变了代码生成，帮助开发者更高效地创建软件，但在兼顾代码片段生成、有效测试用例生成和测试执行方面仍有困难。为解决这些问题，本文提出多智能体辅助代码生成方法 AgentCoder。该框架包含三个专业智能体：程序员、测试设计者和测试执行者。在编程过程中，程序员智能体负责生成代码，并依据测试执行者的反馈加以改进；测试设计者为生成的代码创建测试用例；测试执行者运行代码和测试，再把反馈交给程序员。这一协作系统旨在克服单智能体模型和传统方法的限制，生成更稳健的代码。研究在 9 个代码生成模型和 12 种增强方法上开展了广泛实验，结果显示 AgentCoder 在多个基准上优于现有代码生成模型和提示工程技术。例如，AgentCoder（GPT-4）在 HumanEval 和 MBPP 上的 pass@1 分别达到 96.3% 和 91.8%，总体 token 开销为 56.9K 和 66.3K；对比方法的最佳成绩仅为 90.2% 和 78.9%，总体 token 开销则为 138.2K 和 206.5K。

**English title:** AgentCoder: Multi-Agent-based Code Generation with Iterative Testing and Optimisation

**Abstract:** The advancement of natural language processing (NLP) has been significantly boosted by the development of transformer-based large language models (LLMs). These models have revolutionized NLP tasks, particularly in code generation, aiding developers in creating software with enhanced efficiency. Despite their advancements, challenges in balancing code snippet generation with effective test case generation and execution persist. To address these issues, this paper introduces Multi-Agent Assistant Code Generation (AgentCoder), a novel solution comprising a multi-agent framework with specialized agents: the programmer agent, the test designer agent, and the test executor agent. During the coding procedure, the programmer agent will focus on the code generation and refinement based on the test executor agent's feedback. The test designer agent will generate test cases for the generated code, and the test executor agent will run the code with the test cases and write the feedback to the programmer. This collaborative system ensures robust code generation, surpassing the limitations of single-agent models and traditional methodologies. Our extensive experiments on 9 code generation models and 12 enhancement approaches showcase AgentCoder's superior performance over existing code generation models and prompt engineering techniques across various benchmarks. For example, AgentCoder (GPT-4) achieves 96.3% and 91.8% pass@1 in HumanEval and MBPP datasets with an overall token overhead of 56.9K and 66.3K, while state-of-the-art obtains only 90.2% and 78.9% pass@1 with an overall token overhead of 138.2K and 206.5K.

- **链接 / Link：** [arXiv:2312.13010](https://arxiv.org/abs/2312.13010)
- **年份 / Year：** 2023
- **发表信息 / Publication：** arXiv 预印本，arXiv:2312.13010

## 自动化 AI 研发 / Automated AI R&D

自动完成 AI 研发中的部分工作，包括实验、后训练、算法发现以及改进其他 AI 系统。

Systems that automate parts of AI research and development, including experimentation, post-training, algorithm discovery, and the improvement of other AI systems.

#### 6. AutoResearch：输入洞见，拒绝幻觉

**中文摘要：** 自主研究系统越来越能执行漫长的研究流程，但自动化本身不能保证研究过程始终以科学证据为依据。本文提出 AutoResearch，一个连接“想法生成”和“想法执行”的两阶段系统，同时处理研究想法如何形成，以及如何通过实验可靠确立这两个问题。在想法生成阶段，AutoResearch 持续整合新出现的研究信号与累积的领域知识，识别可迁移的机制性洞见，并通过多模型生成与交叉评审产出有依据、可检验的研究计划。在想法执行阶段，多个协同智能体把计划分解为实验，反复实现和诊断，并在接受研究结论前进行独立、基于证据的审查。在跨模态检索、系统优化和基准驱动的机器学习等代表性场景中，AutoResearch 将生成的想法转化为可测量的进展，检测并纠正不可靠的实验结果，再依据证据决定继续、修改或终止研究方向。例如，在 RSICD 基准上，AutoResearch 生成的想法将平均 Recall 从 32.84 提高到 34.69，审计确认的问题事件只有 5 次，而其他自主研究系统为 11–27 次。这些结果表明，可在实验前让有意义的洞见获得依据，并在接受前让结论获得依据，即“输入洞见，拒绝幻觉”。

**English title:** AutoResearch: Insight In, Hallucination Out

**Abstract:** Autonomous research systems are increasingly capable of executing long research workflows, yet automation alone does not ensure that the resulting process remains scientifically grounded. We introduce AutoResearch, a two-stage system that connects Idea Generation with Idea Execution to address both how research ideas are formed and how they are reliably established through experimentation. In Idea Generation, AutoResearch continuously integrates emerging research signals with accumulated domain knowledge, identifies transferable mechanistic insights, and uses multi-model generation and cross-review to produce grounded, testable research plans. In Idea Execution, coordinated agents decompose these plans into experiments, iteratively implement and diagnose them, and employ independent evidence-based review before accepting research conclusions. Across representative settings in cross-modal retrieval, systems optimization, and benchmark-driven machine learning, AutoResearch turns generated ideas into measurable progress, detects and corrects unreliable experimental results, and makes evidence-conditioned decisions to continue, revise, or terminate research directions. For example, on RSICD benchmark, an AutoResearch-generated idea improves mean Recall from 32.84 to 34.69, while recording only 5 audit-confirmed issue events compared with 11-27 for other autonomous research systems. These results demonstrate a research process in which meaningful insight is grounded before experimentation and conclusions are grounded before acceptance: Insight In, Hallucination Out.

- **链接 / Link：** [arXiv:2608.17906](https://arxiv.org/abs/2608.17906)
- **年份 / Year：** 2026
- **发表信息 / Publication：** arXiv 预印本，arXiv:2608.17906

#### 7. Frontis-MA1：面向机器学习工程递归自我改进的 AI4AI 模型训练

**中文摘要：** 递归自我改进（RSI）要求 AI 系统改进构建 AI 的过程，即 AI4AI；机器学习工程（MLE）为研究这种能力提供了具体、可执行的试验平台。本文提出 OpenMLE，一个面向 MLE 中 RSI 研究的开放式全栈系统，包括带执行反馈、结果可验证的任务环境 OpenMLE-Gym，算子学习系统 OpenMLE-RL，以及长程搜索系统 OpenMLE-Evo。作者在此技术栈上对 35B 模型 Frontis-MA1 进行后训练，使其成为 MLE 的元演化智能体，并围绕四个原子程序演化算子（Draft、Improve、Debug、Crossover）统一后训练与推理：这些算子先通过基于执行的监督微调与强化学习完成训练，训练数据与所有评测基准去重，然后被组合成长程搜索，将学习和演化置于同一个循环中。在 MLE-Bench Lite 上，每项任务限时 12 小时，使用一张显存限制为 12 GB 的 RTX 4090；借助 OpenMLE-Evo，Frontis-MA1（35B）的 Medal Average 从基础模型的 39.39% 提高到 60.61%，加入不依赖基准的经验先验和异步搜索组成 OpenMLE-Evo-Max 后达到 71.21%，超过 GPT-5.5 + Codex，并接近 GPT-5.6 Sol 和 2.8T 参数的 Kimi K3。在留出的 NatureBench Lite 上，两部分均能迁移：框架固定时，换用训练后的模型将 Match-SOTA 从 50% 提高到 70%；模型固定时，换用 OpenMLE-Evo 将其从 20% 提高到 50%。作者发布模型权重和完整 OpenMLE 技术栈，以支持面向 RSI、结果可复现的可执行 AI4AI 研究。

**English title:** Frontis-MA1: Training an AI4AI Model towards Recursive Self-Improvement in Machine Learning Engineering

**Abstract:** Recursive self-improvement (RSI) requires AI systems that improve the process of building AI (i.e., AI4AI); machine learning engineering (MLE) offers a concrete, executable testbed for studying this capability. We introduce OpenMLE, an open full-stack system for RSI research in MLE, spanning verifiable task environments with execution feedback (OpenMLE-Gym), operator learning (OpenMLE-RL), and long-horizon search (OpenMLE-Evo). On this stack we post-train Frontis-MA1 (35B) as a meta-evolution agent for MLE, aligning post-training and inference around four atomic program-evolution operators (Draft, Improve, Debug, Crossover): the same operators are trained via execution-grounded SFT and RL on data deduplicated against all evaluation benchmarks, then composed into long-horizon search, coupling learning and evolution in a single loop. On MLE-Bench Lite under a 12-hour per-task budget on one RTX 4090 capped at 12 GB VRAM, Frontis-MA1 (35B) improves Medal Average from 39.39% to 60.61% over its base model with OpenMLE-Evo, and reaches 71.21% with OpenMLE-Evo-Max (benchmark-independent experience priors and asynchronous search), exceeding GPT-5.5 + Codex and approaching GPT-5.6 Sol and the 2.8T Kimi K3. On held-out NatureBench Lite, both components transfer: with the framework fixed, swapping in the trained model raises Match-SOTA from 50% to 70%; with the model fixed, swapping in OpenMLE-Evo raises it from 20% to 50%. We release the model weights and the full OpenMLE stack to enable reproducible research on executable AI4AI toward RSI. Code: https://github.com/FrontisAI/OpenRSI

- **链接 / Link：** [arXiv:2607.28568](https://arxiv.org/abs/2607.28568)
- **年份 / Year：** 2026
- **发表信息 / Publication：** arXiv 预印本，arXiv:2607.28568

#### 8. FT-Dojo：迈向由语言智能体自主微调大语言模型

**中文摘要：** 面向垂直领域微调大语言模型仍需大量人工：从业者必须整理数据、配置训练，并反复诊断模型行为。尽管自主机器学习和语言智能体受到越来越多关注，端到端 LLM 微调尚未被系统地研究为一种交互式智能体任务。本文提出 FT-Dojo，一个用于自主 LLM 微调的交互式基准环境，含 5 个领域的 13 项任务。它并非又一组静态数据集，而是统一了任务接口、共享原始数据仓库、沙箱执行环境、结构化反馈协议和留出评估流程。作者还开发了面向微调的自主框架 FT-Agent，它用结构化迭代规划、快速失败验证和多层次反馈分析来改进数据与训练策略。实验显示，FT-Agent 是一个强基线，在 13 项任务中的 10 项上表现最佳；与前沿智能体、开源规划骨干的对照实验和多次运行统计也支持主要结论。案例研究表明，智能体能通过累积学习从失败中恢复，但在因果诊断和长程规划方面仍有不足。实现代码发布于 https://github.com/microsoft/rd-agent。

**English title:** FT-Dojo: Towards Autonomous LLM Fine-Tuning with Language Agents

**Abstract:** Fine-tuning large language models for vertical domains remains labor-intensive, requiring practitioners to curate data, configure training, and iteratively diagnose model behavior. Despite growing interest in autonomous machine learning and language agents, end-to-end LLM fine-tuning has not been systematically studied as an interactive agent task. We introduce FT-Dojo, an interactive benchmark environment for autonomous LLM fine-tuning, comprising 13 tasks across 5 domains. Rather than a new collection of static datasets, FT-Dojo standardizes a task interface, shared raw-data repository, sandboxed execution environment, structured feedback protocol, and held-out evaluation procedure. We further develop FT-Agent, a fine-tuning-oriented autonomous framework that uses structured iteration planning, fail-fast validation, and multi-level feedback analysis to refine data and training strategies. Experiments show that FT-Agent provides a strong initial baseline, achieving the best performance on 10 out of 13 tasks, with additional controlled comparisons against frontier agents, open-source planning backbones, and multi-run statistics supporting the main findings. Case studies show that agents can recover from failures through cumulative learning, while still exposing limitations in causal diagnosis and long-horizon planning. The implementation is available at https://github.com/microsoft/rd-agent.

- **链接 / Link：** [arXiv:2603.01712](https://arxiv.org/abs/2603.01712)
- **年份 / Year：** 2026
- **发表信息 / Publication：** arXiv 预印本，arXiv:2603.01712

#### 9. MLEvolve：用于自动发现机器学习算法的自演化框架

**中文摘要：** 大语言模型（LLM）智能体越来越多地被用于科学发现和机器学习工程（MLE）等长程任务，其中持续自演化成为一项重要能力。现有 MLE 智能体受到分支间信息隔离、无记忆搜索和缺少分层控制的影响，长程优化因而受阻。本文提出 MLEvolve，一个基于 LLM、用于端到端机器学习算法发现的自演化多智能体框架。MLEvolve 将树搜索扩展为 Progressive MCGS，通过基于图的引用边实现跨分支信息流，并采用受熵启发的渐进式调度，使搜索逐步从广泛探索转向集中利用。为使智能体依靠累积经验演化，作者提出 Retrospective Memory，将冷启动领域知识库与动态全局记忆结合，以检索和复用特定任务的经验。为稳定长程迭代，框架还使用自适应编码模式，将战略规划与代码生成解耦。MLE-Bench 评测显示，在 12 小时预算（标准运行时间的一半）下，MLEvolve 在平均获奖率、有效提交率等多个指标上达到当时最佳成绩。它在数学算法优化任务上也优于 AlphaEvolve 等专用算法发现方法，显示出较强的跨领域泛化能力。代码发布于 https://github.com/InternScience/MLEvolve。

**English title:** MLEvolve: A Self-Evolving Framework for Automated Machine Learning Algorithm Discovery

**Abstract:** Large language model (LLM) agents are increasingly applied to long-horizon tasks such as scientific discovery and machine learning engineering (MLE), where sustained self-evolution becomes a key capability. However, existing MLE agents suffer from inter-branch information isolation, memoryless search, and lack of hierarchical control, which together hinder long-horizon optimization. We present MLEvolve, an LLM-based self-evolving multi-agent framework for end-to-end machine learning algorithm discovery. By extending tree search to Progressive MCGS, MLEvolve enables cross-branch information flow through graph-based reference edges and gradually shifts the search from broad exploration to focused exploitation with an entropy-inspired progressive schedule. To allow the agent to evolve with accumulated experience, we introduce Retrospective Memory, which combines a cold-start domain knowledge base with a dynamic global memory for task-specific experience retrieval and reuse. For stable long-horizon iteration, we further decouple strategic planning from code generation with adaptive coding modes. Evaluation on MLE-Bench shows that MLEvolve achieves state-of-the-art performance across multiple dimensions including average medal rate and valid submission rate under a 12-hour budget (half the standard runtime). Moreover, MLEvolve also outperforms specialized algorithm discovery methods including AlphaEvolve on mathematical algorithm optimization tasks, demonstrating strong cross-domain generalization. Our code is available at https://github.com/InternScience/MLEvolve.

- **链接 / Link：** [arXiv:2606.06473](https://arxiv.org/abs/2606.06473)
- **年份 / Year：** 2026
- **发表信息 / Publication：** arXiv 预印本，arXiv:2606.06473

#### 10. 迈向端到端自动化 AI 研究

**中文摘要：** 科学自动化一直是人工智能研究的目标。尽管研究界在自动化科学流程的各个独立环节上已取得长足进展，能自主完成从构思到发表的整个研究生命周期的系统仍未实现。本文提出一条端到端自动化整个科学流程的管线。作者提出 The AI Scientist，它可以创建研究想法、编写代码、运行实验、绘图和分析数据、撰写完整科学论文，并自行开展同行评审。其想法、执行和表述的质量已足以使该 AI 系统生成的稿件通过一个顶级机器学习会议研讨会的首轮同行评审；该研讨会的录取率为 70%。系统在复杂的智能体架构中使用现代基础模型。作者在两种设置下评估 The AI Scientist：一种是以人类提供的代码模板为起始脚手架、针对特定主题开展研究的聚焦模式；另一种是不使用模板、借助智能体搜索进行更广泛科学探索的开放模式。两种设置都能产生多样的想法，并自动测试、报告和评估这些想法。这项成果说明 AI 作出科学贡献的能力正在增强，也可能改变研究开展方式。与任何影响重大的新技术一样，它也可能带来风险，例如进一步挤压已不堪重负的评审系统，或给科学文献增加噪声。若能以负责任的方式开发，这类自主系统可大幅加快科学发现。

**English title:** Towards end-to-end automation of AI research

**Abstract:** The automation of science is a long-standing ambition in artificial intelligence (AI) research. Although the community has made substantial progress in automating individual components of the scientific process, a system that autonomously navigates the entire research life cycle—from conception to publication—has remained out of reach. Here we present a pipeline for automating the entire scientific process end to end. We present The AI Scientist, which creates research ideas, writes code, runs experiments, plots and analyses data, writes the entire scientific manuscript, and performs its own peer review. Its ideas, execution and presentation are of sufficient quality that the manuscript generated by this AI system passed the first round of peer review for a workshop of a top-tier machine learning conference. The workshop had an acceptance rate of 70%. Our system leverages modern foundation models within a complex agentic system. We evaluate The AI Scientist in two settings: a focused mode using human-provided code templates as an initial scaffold for conducting research on a specific topic and a template-free, open-ended mode that leverages agentic search for wider scientific exploration. Both settings produce diverse ideas and automatically test, report on and evaluate them. This achievement demonstrates the growing capacity of AI for making scientific contributions and signifies a potential paradigm shift in how research is conducted. As with any impactful new technology, there could be important risks, including taxing overwhelmed review systems and adding noise to the scientific literature. However, if developed responsibly, such autonomous systems could greatly accelerate scientific discovery.

- **链接 / Link：** [DOI:10.1038/s41586-026-10265-5](https://doi.org/10.1038/s41586-026-10265-5)
- **年份 / Year：** 2026
- **发表信息 / Publication：** *Nature* 651, 914–919 (2026)；2026-03-25 在线发表

#### 11. 迈向以执行为依据的自动化 AI 研究

**中文摘要：** 自动化 AI 研究有望加快科学发现，但当前 LLM 经常生成看似合理却无效的想法。执行反馈可能有所帮助，不过自动执行是否可行，以及 LLM 能否从执行反馈中学习，仍不明确。为研究这些问题，作者先构建自动执行器，用于实现想法并启动大规模并行 GPU 实验以验证成效；随后把两个现实研究问题，即 LLM 预训练与后训练，转化为执行环境，并证明自动执行器可以实现从前沿 LLM 采样得到的大部分想法。论文分析了两种从执行反馈中学习的方法：演化搜索和强化学习。执行引导的演化搜索具有较高样本效率：在仅十个搜索轮次内，它便在后训练上找到显著优于 GRPO 基线的方法（69.4% 对 48.0%），也在预训练上找到优于 nanoGPT 基线的方案（19.7 分钟对 35.9 分钟）。前沿 LLM 在搜索中经常生成有意义的算法想法，但往往较早饱和，只有偶尔表现出扩展趋势。以执行奖励进行强化学习则会发生模式坍缩：它能提高想法生成模型的平均奖励，却不能提高上限，因为模型会收敛到简单想法。作者深入分析已执行的想法和训练动态，为后续以执行为依据的自动化 AI 研究提供参考。

**English title:** Towards Execution-Grounded Automated AI Research

**Abstract:** Automated AI research holds great potential to accelerate scientific discovery. However, current LLMs often generate plausible-looking but ineffective ideas. Execution grounding may help, but it is unclear whether automated execution is feasible and whether LLMs can learn from the execution feedback. To investigate these, we first build an automated executor to implement ideas and launch large-scale parallel GPU experiments to verify their effectiveness. We then convert two realistic research problems - LLM pre-training and post-training - into execution environments and demonstrate that our automated executor can implement a large fraction of the ideas sampled from frontier LLMs. We analyze two methods to learn from the execution feedback: evolutionary search and reinforcement learning. Execution-guided evolutionary search is sample-efficient: it finds a method that significantly outperforms the GRPO baseline (69.4% vs 48.0%) on post-training, and finds a pre-training recipe that outperforms the nanoGPT baseline (19.7 minutes vs 35.9 minutes) on pre-training, all within just ten search epochs. Frontier LLMs often generate meaningful algorithmic ideas during search, but they tend to saturate early and only occasionally exhibit scaling trends. Reinforcement learning from execution reward, on the other hand, suffers from mode collapse. It successfully improves the average reward of the ideator model but not the upper-bound, due to models converging on simple ideas. We thoroughly analyze the executed ideas and training dynamics to facilitate future efforts towards execution-grounded automated AI research.

- **链接 / Link：** [arXiv:2601.14525](https://arxiv.org/abs/2601.14525)
- **年份 / Year：** 2026
- **发表信息 / Publication：** arXiv 预印本，arXiv:2601.14525

#### 12. AI Scientist：迈向全自动、开放式科学发现

**中文摘要：** 通用人工智能的一项重大挑战，是开发能够开展科学研究并发现新知识的智能体。前沿模型已被用作人类科学家的辅助工具，例如头脑风暴、编写代码或完成预测任务，但仍只承担科学流程的一小部分。本文提出首个覆盖全面的全自动科学发现框架，使前沿大语言模型能够独立研究并交流发现。The AI Scientist 可以生成新颖研究想法、编写代码、执行实验、可视化结果、以完整科学论文描述发现，随后运行模拟评审流程进行评价。原则上，这一流程可以反复执行，以开放方式迭代发展想法，类似人类科学共同体。作者将其用于三个不同的机器学习子领域：扩散建模、基于 Transformer 的语言建模和学习动力学，以展示其通用性。每个想法都能以不足 15 美元的成本实现并发展成完整论文。为评价所生成的论文，作者设计并验证了自动评审器；其论文评分能力接近人类。由自动评审器判断，The AI Scientist 可以生成超过顶级机器学习会议录取门槛的论文。该方法让 AI 智能体介入 AI 研究的完整流程，并向可持续、低成本地产生创意和创新迈进一步。代码开源于 https://github.com/SakanaAI/AI-Scientist。

**English title:** The AI Scientist: Towards Fully Automated Open-Ended Scientific Discovery

**Abstract:** One of the grand challenges of artificial general intelligence is developing agents capable of conducting scientific research and discovering new knowledge. While frontier models have already been used as aides to human scientists, e.g. for brainstorming ideas, writing code, or prediction tasks, they still conduct only a small part of the scientific process. This paper presents the first comprehensive framework for fully automatic scientific discovery, enabling frontier large language models to perform research independently and communicate their findings. We introduce The AI Scientist, which generates novel research ideas, writes code, executes experiments, visualizes results, describes its findings by writing a full scientific paper, and then runs a simulated review process for evaluation. In principle, this process can be repeated to iteratively develop ideas in an open-ended fashion, acting like the human scientific community. We demonstrate its versatility by applying it to three distinct subfields of machine learning: diffusion modeling, transformer-based language modeling, and learning dynamics. Each idea is implemented and developed into a full paper at a cost of less than $15 per paper. To evaluate the generated papers, we design and validate an automated reviewer, which we show achieves near-human performance in evaluating paper scores. The AI Scientist can produce papers that exceed the acceptance threshold at a top machine learning conference as judged by our automated reviewer. This approach signifies the beginning of a new era in scientific discovery in machine learning: bringing the transformative benefits of AI agents to the entire research process of AI itself, and taking us closer to a world where endless affordable creativity and innovation can be unleashed on the world's most challenging problems. Our code is open-sourced at https://github.com/SakanaAI/AI-Scientist

- **链接 / Link：** [arXiv:2408.06292](https://arxiv.org/abs/2408.06292)
- **年份 / Year：** 2024
- **发表信息 / Publication：** arXiv 预印本，arXiv:2408.06292

## 具身与物理自我改进 / Embodied & Physical Self-Improvement

通过与物理或模拟环境交互，持续改进机器人策略、技能、Harness、世界模型，或围绕这些要素展开的研究过程。

Systems that use interaction with physical or simulated environments to generate persistent improvements in robot policies, skills, harnesses, world models, or the research process around them.

#### 13. ASPIRE：机器人智能体技能发现

**中文摘要：** 传统机器人编程颇具挑战：它需要协调多模态感知、处理物理接触动力学，并应对不同配置和执行失败。本文提出 ASPIRE（Agentic Skill Programming through Iterative Robot Exploration），一种持续学习系统。它采用代码即策略范式，自主编写并改进机器人控制程序，同时将经验不断积累到可复用的技能库中。ASPIRE 发现的技能能够跨任务、跨仿真与现实设置、跨机器人形态保留。其开放循环由三部分构成：（1）闭环机器人执行引擎，公开细粒度多模态轨迹，使系统能自主诊断失败、合成修复并验证；（2）持续扩展的技能库，将经验证的修复提炼成可复用、可迁移的知识；（3）演化搜索，生成多样化任务序列和控制程序，使探索超越单轨迹改进。在扰动条件下的 LIBERO-Pro 操作、Robosuite 双手交接和 BEHAVIOR-1K 长程家庭任务上，ASPIRE 分别比以往方法最多高出 77%、72% 和 32%。累积的技能库还能对未见长程任务进行零样本泛化：在 LIBERO-Pro Long 上，ASPIRE 的成功率为 31%，而既有方法虽使用测试时推理和重试，也仅为 4%。最后，仿真中发现的技能提供了仿真到现实迁移的初步证据，并显著减少了不同机器人形态和 API 上的真实机器人编程工作。

**English title:** ASPIRE: Agentic /Skills Discovery for Robotics

**Abstract:** Traditional robot programming is challenging: it requires orchestrating multimodal perception, managing physical contact dynamics, and handling diverse configurations and execution failures. We introduce ASPIRE (Agentic Skill Programming through Iterative Robot Exploration), a continual learning system that autonomously writes and refines robot control programs in a code-as-policy paradigm while compounding experience into a reusable skill library. ASPIRE discovers skills that persist across tasks, simulation and real-world settings, and embodiments. It operates in an open-ended loop with three components: (1) a closed-loop robot execution engine that exposes fine-grained multimodal traces, enabling autonomous failure diagnosis, repair synthesis, and validation; (2) a continually expanding skill library that distills validated fixes into reusable, transferable knowledge; and (3) evolutionary search that generates diverse task sequences and control programs to explore beyond single-trajectory refinement. ASPIRE surpasses prior methods by up to 77% on LIBERO-Pro manipulation under perturbation, 72% on Robosuite bimanual handover, and 32% on BEHAVIOR-1K long-horizon household tasks. Its accumulated library also enables zero-shot generalization to unseen long-horizon tasks: on LIBERO-Pro Long, ASPIRE achieves 31% success versus 4% for prior methods despite their use of test-time reasoning and retries. Finally, simulation-discovered skills provide initial evidence of sim-to-real transfer, substantially reducing real-robot programming effort across different embodiments and robot APIs.

- **链接 / Link：** [arXiv:2607.00272](https://arxiv.org/abs/2607.00272)
- **年份 / Year：** 2026
- **发表信息 / Publication：** arXiv 预印本，arXiv:2607.00272

#### 14. ENPIRE：真实世界中的智能体机器人策略自我改进

**中文摘要：** 在现实世界实现灵巧机器人操作高度依赖人工监督和算法工程，这已成为迈向通用物理智能的主要瓶颈。新出现的编程智能体虽能编写代码以自动搜索算法，但其成功大多局限于数字环境。作者推测，自动化机器人研究所缺少的抽象，是一个可重复的真实世界策略改进反馈循环：重置场景、执行策略、验证结果，再改进下一次迭代。为弥补这一缺口，本文提出 ENPIRE，一个用于编程智能体的 Harness 框架，通过四个核心模块落实物理反馈例程：环境模块（EN）负责自动重置和验证；策略改进模块（PI）启动策略优化；Rollout 模块（R）使用单台或并行运行的多台实体机器人评估策略；演化模块（E）让编程智能体分析日志、查阅文献并改进训练基础设施和算法代码，以处理失败模式。该闭环系统把真实世界操作学习转化为可控的优化流程，在减少人工投入的同时，可以公平消融不同训练方案与智能体变体。借助 ENPIRE，前沿编程智能体可以自主训练策略，使其在整理针盒、扎紧束线带和使用工具等高难度灵巧操作任务上达到 99% 的成功率；把智能体团队部署到机器人集群后，过程还会进一步加速。结果表明，用编程智能体在物理世界中自主推进机器人研究是一条实用且可扩展的路径。

**English title:** ENPIRE: Agentic Robot Policy Self-Improvement in the Real World

**Abstract:** Achieving dexterous robotic manipulation in the real world heavily relies on human supervision and algorithm engineering, which becomes a central bottleneck in the pursuit of general physical intelligence. Although emerging coding agents can generate code to automate algorithm search, their successes remain largely confined in digital environments. We conjecture that the missing abstraction to automate robotics research is a repeatable feedback loop for real-world policy improvement: reset the scene, execute a policy, verify the outcome, and refine the next iteration. To bridge this gap, we introduce ENPIRE, a harness framework for coding agents that instantiates this physical feedback routine with four core modules: an Environment module (EN) for automatic reset and verification, a Policy Improvement module (PI) that launches policy refinement, a Rollout module (R) to evaluate policies with one or multiple physical robots operating in parallel, and an Evolution module (E) in which coding agents analyze logs, consult literature, improve training infrastructure and algorithm code to address failure modes. This closed-loop system transforms real-world manipulation learning into a controllable optimization procedure, minimizing human effort while allowing fair ablations across training recipe and agent variants. Powered by ENPIRE, frontier coding agents can autonomously train a policy to achieve a 99% success rate on challenging, dexterous manipulation tasks, such as organizing a pin box, fastening a zip tie, and tool use, a process that further accelerates when we dispatch an agent team on a robot fleet. Our results suggest a practical and scalable path toward deploying coding agents to autonomously advancing robotics in the physical world.

- **链接 / Link：** [arXiv:2606.19980](https://arxiv.org/abs/2606.19980)
- **年份 / Year：** 2026
- **发表信息 / Publication：** arXiv 预印本，arXiv:2606.19980

#### 15. MineEvolve：利用累积知识实现长程具身 Minecraft 智能体的自演化

**中文摘要：** 长程具身智能要求智能体通过交互不断改进，而不是只执行由静态目标生成的计划。核心难题是如何把过往执行转化为能影响未来决策的知识。Minecraft 是研究这一问题的代表性测试平台：制作工具、搭建红石组件、获得钻石装备等任务包含很长的前置条件链，还经常受到工具缺失、路径受阻、图形界面故障或执行停滞的干扰。为此，本文提出 MineEvolve，一个知识驱动的自演化框架，把执行反馈转化为可操作的行为知识。MineEvolve 首先用 Monitor 将每次子目标执行转化为带类型的反馈，包括状态变化、物品栏变化、失败类型、进度信号和停滞指标。Inducer 再从成功执行中提取可复用技能，从失败或停滞执行中提取补救措施。Curator 负责验证、合并、筛选和检索这些知识条目；Adaptor 则在反复失败或停滞时利用知识修复计划中尚未完成的部分。在 Minecraft MCU 长程任务套件上的实验表明，MineEvolve 能稳定提高多种语言模型规划器的表现，对依赖关系较多的任务组提升更大。消融实验和知识累积研究进一步说明，将执行信号转化为结构化行为知识，是长程环境中构建自演化具身智能体的有效途径。代码发布于 https://github.com/xzw-ustc/MC-MineEvolve。

**English title:** MineEvolve: Self-Evolution with Accumulated Knowledge for Long-Horizon Embodied Minecraft Agents

**Abstract:** Long-horizon embodied intelligence requires agents to improve through interaction, not merely to execute plans generated from static goals. A central challenge is therefore to transform past executions into knowledge that can shape future decisions. Minecraft provides a representative testbed for this problem, where tasks such as crafting tools, building redstone components, and obtaining diamond equipment involve long prerequisite chains and are frequently disrupted by missing tools, blocked paths, GUI failures, or stagnant execution. To this end, we propose MineEvolve, a knowledge-driven self-evolution framework that converts execution feedback into actionable behavioral knowledge. MineEvolve first uses Monitor to convert each subgoal execution into typed feedback, including state changes, inventory changes, failure types, progress signals, and stagnation indicators. Inducer then derives reusable skills from successful executions and remedies from failed or stagnant executions. Curator validates, merges, filters, and retrieves these knowledge entries, while Adaptor uses them to repair the unfinished part of the plan under repeated failures or stagnation. Experiments on the Minecraft MCU long-horizon task suite show that MineEvolve consistently improves performance across multiple language-model planners, with larger gains on high-dependency task groups. Ablation and knowledge-accumulation studies further demonstrate that converting execution signals into structured behavioral knowledge is an effective path toward self-evolving embodied agents in long-horizon environments. Our code is available at https://github.com/xzw-ustc/MC-MineEvolve.

- **链接 / Link：** [arXiv:2603.13131](https://arxiv.org/abs/2603.13131)
- **年份 / Year：** 2026
- **发表信息 / Publication：** arXiv 预印本，arXiv:2603.13131

#### 16. RISE：使用组合式世界模型实现机器人策略自我改进

**中文摘要：** 尽管模型容量和数据采集规模持续增长，视觉—语言—动作（VLA）模型在接触密集型和动态操作任务中仍然脆弱，细小的执行偏差可能累积为失败。强化学习（RL）为提高稳健性提供了原则性方法，但在物理世界中进行在策略 RL 会受到安全风险、硬件成本和环境重置的限制。本文提出 RISE，一个通过想象进行机器人强化学习的可扩展框架。核心是组合式世界模型：（i）用可控动力学模型预测多视角未来；（ii）用进度价值模型评价想象结果，为策略改进提供有信息量的优势估计。这种组合设计让状态与价值可以分别采用最合适、彼此不同的架构和目标。各组件被整合进闭环自我改进管线，持续生成想象 rollout、估计优势并在想象空间更新策略，无需昂贵的物理交互。在三项具有挑战性的真实世界任务中，RISE 较已有方法明显提升：动态积木分拣的绝对性能提高超过 35%，背包装填超过 45%，关盒任务超过 35%。

**English title:** RISE: Self-Improving Robot Policy with Compositional World Model

**Abstract:** Despite the sustained scaling on model capacity and data acquisition, Vision-Language-Action (VLA) models remain brittle in contact-rich and dynamic manipulation tasks, where minor execution deviations can compound into failures. While reinforcement learning (RL) offers a principled path to robustness, on-policy RL in the physical world is constrained by safety risk, hardware cost, and environment reset. To bridge this gap, we present RISE, a scalable framework of robotic reinforcement learning via imagination. At its core is a Compositional World Model that (i) predicts multi-view future via a controllable dynamics model, and (ii) evaluates imagined outcomes with a progress value model, producing informative advantages for the policy improvement. Such compositional design allows state and value to be tailored by best-suited yet distinct architectures and objectives. These components are integrated into a closed-loop self-improving pipeline that continuously generates imaginary rollouts, estimates advantages, and updates the policy in imaginary space without costly physical interaction. Across three challenging real-world tasks, RISE yields significant improvement over prior art, with more than +35% absolute performance increase in dynamic brick sorting, +45% for backpack packing, and +35% for box closing, respectively.

- **链接 / Link：** [arXiv:2602.11075](https://arxiv.org/abs/2602.11075)
- **年份 / Year：** 2026
- **发表信息 / Publication：** Robotics: Science and Systems（RSS）2026；arXiv:2602.11075

#### 17. 通过技能—Harness 演化实现具身智能体自演化

**中文摘要：** 具身智能体日益以基础模型为核心构建成系统，其性能不只取决于模型权重，也取决于围绕模型的技能、上下文、动作接口和执行框架。监督微调和强化学习能够使智能体适应新环境，但需要额外的数据、奖励和训练；不少免训练、以代码为中心的方法又依赖可编程机器人 API，而固定接口场景中可能没有这类 API。本文提出 SHAPER，一个面向免训练具身适应的自演化框架。它保持模型参数冻结，并通过目标环境中的 rollout 演化可复用技能和上下文—代码 Harness，从而改进非参数智能体系统。在 SHAPER 中，同一个冻结模型既可充当规划器，也可充当优化器，无需更新参数便能改进外部技能和上下文—代码 Harness。作者在 VLABench 与 ESI-Bench 上评估 SHAPER，覆盖具有不同底层动作接口的具身智能体，并与纯执行、监督微调以及无验证器选择和投票等测试时扩展基线比较。结果表明，在模型训练成本高昂、不可用或不理想时，技能与 Harness 优化是实现具身智能体自演化的一条实用途径。

**English title:** Self-Evolving Embodied Agents via Skill-Harness Evolution

**Abstract:** Embodied agents are increasingly built as systems around foundation models, where performance depends not only on model weights but also on the skills, context, action interfaces, and execution harness surrounding the model. While supervised fine-tuning and reinforcement learning can adapt agents to new environments, they require additional data, rewards, and training runs; meanwhile, many train-free code-centric approaches rely on programmable robot APIs that may be unavailable in fixed-interface settings. We propose SHAPER, a self-evolving framework for train-free embodied adaptation that keeps model parameters frozen and improves the non-parametric agent system by evolving reusable skills and a context-code harness through target-environment rollouts. In SHAPER, the same frozen model can serve as both planner and optimizer, refining its external skills and context-code harness without parameter updates. We evaluate SHAPER on VLABench and ESI-Bench, covering embodied agents with different low-level action interfaces, and compare against pure execution, supervised fine-tuning, and test-time-scaling baselines such as verifier-free selection and voting. Our results suggest that skill-and-harness optimization is a practical route to self-evolving embodied agents when model training is expensive, unavailable, or undesirable.

- **链接 / Link：** [arXiv:2608.11350](https://arxiv.org/abs/2608.11350)
- **年份 / Year：** 2026
- **发表信息 / Publication：** arXiv 预印本，arXiv:2608.11350

#### 18. 利用残差强化学习生成数据的自我改进视觉—语言—动作模型

**中文摘要：** 监督微调（SFT）已成为大型视觉—语言—动作（VLA）模型事实上的后训练策略，但它依赖昂贵的人工示范，限制了可扩展性和泛化能力。本文提出 Probe, Learn, Distill（PLD），一种即插即用框架，通过残差强化学习和分布感知数据采集改进 VLA。第一阶段“获取专长”冻结 VLA 骨干网络，并通过离策略强化学习训练轻量级残差 actor；这些专用策略会在基础策略失败的状态接管，从而探测通用策略的失败区域。第二阶段“数据采集”采用混合 rollout 方案，使残差干预偏向基础策略经常访问的状态；这样既让采集轨迹与通用策略的部署分布保持一致，也能捕捉恢复行为。第三阶段“微调”使用标准 SFT 将精选轨迹蒸馏回通用策略，并可用于流匹配头和自回归头。PLD 在多种设置下接受评估：在 LIBERO 基准上达到近饱和的 99% 任务成功率，在 SimplerEnv 中取得超过 50% 的性能增益，并在真实 Franka 机械臂操作任务中显示出实用性。消融实验进一步表明，残差策略探测和分布感知回放是收集部署分布对齐数据的关键，这些数据能提高 VLA 在已见和未见任务上的能力。结果显示，由强化学习生成且与策略分布一致的数据能够胜过仅依靠遥操作的示范，为 VLA 模型的可扩展自我改进提供了一条路径。

**English title:** Self-Improving Vision-Language-Action Models with Data Generation via Residual RL

**Abstract:** Supervised fine-tuning (SFT) has become the de facto post-training strategy for large vision-language-action (VLA) models, but its reliance on costly human demonstrations limits scalability and generalization. We propose Probe, Learn, Distill (PLD), a plug-and-play framework that improves VLAs through residual reinforcement learning and distribution-aware data collection. In Stage 1 (specialist acquisition), we freeze the VLA backbone and train lightweight residual actors via off-policy RL. These specialists take over in states where the base policy fails, thereby probing failure regions of the generalist. In Stage 2 (data collection), we employ a hybrid rollout scheme that biases residual interventions toward states frequently visited by the base policy, aligning collected trajectories with the generalist’s deployment distribution while capturing recovery behaviors. In Stage 3 (fine-tuning), these curated trajectories are distilled back into the generalist with standard SFT, applicable to both flow-matching and autoregressive heads. We evaluate PLD across diverse settings: it achieves a near-saturated 99% task success rate on the LIBERO benchmark, delivers over 50% performance gains in SimplerEnv, and demonstrates practicality on real-world Franka arm manipulation tasks. We further provide ablations showing that residual policy probing and distribution-aware replay are key to collecting deployment-aligned data that improves VLAs’ capabilities on both seen and unseen tasks. Our results demonstrate that RL-generated, policy-aligned data can surpass teleoperation-only demonstrations, offering a scalable path toward self-improving VLA models.

- **链接 / Link：** [ICLR 2026 论文页](https://iclr.cc/virtual/2026/poster/10008318)；[OpenReview](https://openreview.net/forum?id=eUGoqrZ6Ea)
- **年份 / Year：** 2026
- **发表信息 / Publication：** ICLR 2026 Poster

## 演化式与开放式递归自我改进 / Evolutionary & Open-Ended RSI

通过演化、质量—多样性与开放式过程持续发现更强的解或学习系统。

Evolutionary, quality-diversity, and open-ended processes that continually discover stronger solutions or learning systems.

#### 19. PACEvolve：实现进度感知、前后一致的长程演化

**中文摘要：** 大语言模型（LLM）已经成为强大的演化搜索操作器，但高效搜索脚手架的设计仍较为随意。现有 LLM 在环系统虽有潜力，却没有系统管理演化过程的方法。作者识别出三种不同的失败模式：实验历史使后续候选生成产生偏差的“上下文污染”；智能体因探索—利用失衡而停滞在局部最优的“模式坍缩”；固定交叉策略无法有效利用并行搜索轨迹的“协作薄弱”。为此，本文提出进度感知一致演化框架 PACEvolve，用于稳健地管理智能体上下文和搜索动态。PACEvolve 结合带剪枝的分层上下文管理（HCM）以处理上下文污染，使用基于动量的回溯（MBB）逃离局部最优，并采用统一回溯和交叉的自适应采样策略进行动态搜索协调（CE），让智能体在内部改进和跨轨迹协作之间取得平衡。实验表明，PACEvolve 为一致、长程的自我改进提供了系统方法，在 LLM-SR 和 KernelBench 上达到当时最佳成绩，并找到超过 Modded NanoGPT 纪录的解。

**English title:** PACEvolve: Enabling Long-Horizon Progress-Aware Consistent Evolution

**Abstract:** Large Language Models (LLMs) have emerged as powerful operators for evolutionary search, yet the design of efficient search scaffolds remains ad hoc. While promising, current LLM-in-the-loop systems lack a systematic approach to managing the evolutionary process. We identify three distinct failure modes: Context Pollution, where experiment history biases future candidate generation; Mode Collapse, where agents stagnate in local minima due to poor exploration-exploitation balance; and Weak Collaboration, where rigid crossover strategies fail to leverage parallel search trajectories effectively. We introduce Progress-Aware Consistent Evolution (PACEvolve), a framework designed to robustly govern the agent's context and search dynamics, to address these challenges. PACEvolve combines hierarchical context management (HCM) with pruning to address context pollution; momentum-based backtracking (MBB) to escape local minima; and a self-adaptive sampling policy that unifies backtracking and crossover for dynamic search coordination (CE), allowing agents to balance internal refinement with cross-trajectory collaboration. We demonstrate that PACEvolve provides a systematic path to consistent, long-horizon self-improvement, achieving state-of-the-art results on LLM-SR and KernelBench, while discovering solutions surpassing the record on Modded NanoGPT.

- **链接 / Link：** [arXiv:2601.10657](https://arxiv.org/abs/2601.10657)
- **年份 / Year：** 2026
- **发表信息 / Publication：** arXiv 预印本，arXiv:2601.10657

#### 20. AlphaEvolve：用于科学与算法发现的编程智能体

**中文摘要：** 本白皮书提出 AlphaEvolve，一种演化式编程智能体。它显著增强了当前最佳 LLM 处理高难度任务的能力，例如攻克开放科学问题或优化计算基础设施的关键组件。AlphaEvolve 编排一条由多个 LLM 组成的自主流水线，让它们直接修改代码以改进算法。在一个持续接收一个或多个评估器反馈的演化流程中，AlphaEvolve 迭代改进算法，并可能由此产生新的科学和实用发现。作者将其用于多项重要计算问题，以说明该方法的广泛适用性。在优化 Google 大规模计算栈的关键组件时，AlphaEvolve 开发出效率更高的数据中心调度算法，找到了硬件加速器电路设计中功能等价的简化方案，还加速了支撑 AlphaEvolve 自身的 LLM 训练。它也发现了新的、可证明正确的算法，在一系列数学和计算机科学问题上超过既有最佳方案，大幅拓展了早先自动发现方法的范围。尤其是，AlphaEvolve 开发的搜索算法找到了一种用 48 次标量乘法完成两个 4×4 复数矩阵相乘的方法，这是 56 年来首次在这一设定下改进 Strassen 算法。作者认为，AlphaEvolve 及同类编程智能体能改进许多科学与计算领域的问题解法。

**English title:** AlphaEvolve: A coding agent for scientific and algorithmic discovery

**Abstract:** In this white paper, we present AlphaEvolve, an evolutionary coding agent that substantially enhances capabilities of state-of-the-art LLMs on highly challenging tasks such as tackling open scientific problems or optimizing critical pieces of computational infrastructure. AlphaEvolve orchestrates an autonomous pipeline of LLMs, whose task is to improve an algorithm by making direct changes to the code. Using an evolutionary approach, continuously receiving feedback from one or more evaluators, AlphaEvolve iteratively improves the algorithm, potentially leading to new scientific and practical discoveries. We demonstrate the broad applicability of this approach by applying it to a number of important computational problems. When applied to optimizing critical components of large-scale computational stacks at Google, AlphaEvolve developed a more efficient scheduling algorithm for data centers, found a functionally equivalent simplification in the circuit design of hardware accelerators, and accelerated the training of the LLM underpinning AlphaEvolve itself. Furthermore, AlphaEvolve discovered novel, provably correct algorithms that surpass state-of-the-art solutions on a spectrum of problems in mathematics and computer science, significantly expanding the scope of prior automated discovery methods (Romera-Paredes et al., 2023). Notably, AlphaEvolve developed a search algorithm that found a procedure to multiply two $4 \times 4$ complex-valued matrices using $48$ scalar multiplications; offering the first improvement, after 56 years, over Strassen's algorithm in this setting. We believe AlphaEvolve and coding agents like it can have a significant impact in improving solutions of problems across many areas of science and computation.

- **链接 / Link：** [arXiv:2506.13131](https://arxiv.org/abs/2506.13131)
- **年份 / Year：** 2025
- **发表信息 / Publication：** 白皮书；arXiv:2506.13131

#### 21. 面向种群方法的高阶与自指演化

**中文摘要：** 演化算法结构简单、支持高度并行，因此重新受到机器学习应用的重视，例如强化学习的课程生成和在线超参数调优。不过，其性能可能对突变率等演化超参数十分敏感。为此，既有研究提出了“自适应”突变率，即让突变率本身也参与演化。这个办法只能部分解决问题，因为它仍然依赖一组预先设定的元突变率。受近期研究启发，作者考察高阶突变能否让演化算法更稳健并提升整体性能；近期工作已在特定案例中表明，演化可以隐式优化高阶元突变率。论文还分析了自指突变，即让最后一阶元突变参数发生突变。结果显示，在用于在线超参数调优的种群式训练（PBT）以及采用无监督环境设计（UED）的课程学习中，自指突变提高了对初始超参数的稳健性。在竞争性多智能体环境中，自指突变还会带来更复杂的适应。该研究是迈向无需超参数、稳健且完全自调优系统的第一步。

**English title:** Higher Order and Self-Referential Evolution for Population-based Methods

**Abstract:** Due to their simplicity and support of high levels of parallelism, evolutionary algorithms have regained popularity in machine learning applications such as curriculum generation for Reinforcement Learning and online hyperparameter tuning. Yet, their performance can be brittle with respect to evolutionary hyperparameters, e.g. the mutation rate. To address this, self-adaptive mutation rates, i.e. mutation rates that also evolve, have previously been proposed. While this approach offers a partial solution, it still relies on an a priori set meta-mutation rates. Inspired by recent work, which demonstrates specific cases where evolution is able to implicitly optimize for higher-order meta-mutation rates, we investigate whether these higher-order mutations can make evolutionary algorithms more robust and improve their overall performance. We also analyse self-referential mutations, which mutate the final order meta-mutation parameter. Our results show that self-referential mutations improve robustness to initial hyperparameters in Population-based Training (PBT) for online hyperparameter tuning, and curriculum learning using Unsupervised Environment Design (UED). We also observe that self-referential mutations result in more complex adaptation in competitive multi-agent settings. Our research presents first steps towards robust fully self-tuning systems that are hyperparameter free.

- **链接 / Link：** [OpenReview](https://openreview.net/forum?id=3tk6AES1Aj)；[ICML 2024 研讨会页面](https://icml.cc/virtual/2024/35867)
- **年份 / Year：** 2024
- **发表信息 / Publication：** ICML 2024 Workshop on Automated Reinforcement Learning: Exploring Meta-Learning, AutoML, and LLMs

#### 22. 利用大语言模型进行程序搜索所得的数学发现

**中文摘要：** 大语言模型（LLM）在解决复杂任务方面表现出很强的能力，涵盖定量推理与自然语言理解等任务。但 LLM 有时会虚构（或产生幻觉），给出看似合理却不正确的陈述，这阻碍了当前大型模型在科学发现中的应用。本文提出 FunSearch（即在函数空间中搜索），这是一种将预训练 LLM 与系统化评估器配对的演化过程。作者证明，该方法能在重要问题上超越已有最佳结果，推进基于 LLM 方法的能力边界。将 FunSearch 用于极值组合数学的核心问题“帽集问题”时，系统发现了新的大型帽集构造，在有限维和渐近情形下均超过已知最佳方案，说明 LLM 可以针对已有开放问题作出发现。作者又将 FunSearch 用于在线装箱这一算法问题，找到优于常用基线的新启发式方法，以说明其通用性。与多数计算机搜索方法不同，FunSearch 搜索的是描述如何求解问题的程序，而不是直接搜索答案。除高效、可扩展之外，所发现的程序通常比原始解更容易解释，有助于领域专家与 FunSearch 形成反馈循环，也便于把程序部署到现实应用中。

**English title:** Mathematical discoveries from program search with large language models

**Abstract:** Large language models (LLMs) have demonstrated tremendous capabilities in solving complex tasks, from quantitative reasoning to understanding natural language. However, LLMs sometimes suffer from confabulations (or hallucinations), which can result in them making plausible but incorrect statements. This hinders the use of current large models in scientific discovery. Here we introduce FunSearch (short for searching in the function space), an evolutionary procedure based on pairing a pretrained LLM with a systematic evaluator. We demonstrate the effectiveness of this approach to surpass the best-known results in important problems, pushing the boundary of existing LLM-based approaches. Applying FunSearch to a central problem in extremal combinatorics—the cap set problem—we discover new constructions of large cap sets going beyond the best-known ones, both in finite dimensional and asymptotic cases. This shows that it is possible to make discoveries for established open problems using LLMs. We showcase the generality of FunSearch by applying it to an algorithmic problem, online bin packing, finding new heuristics that improve on widely used baselines. In contrast to most computer search approaches, FunSearch searches for programs that describe how to solve a problem, rather than what the solution is. Beyond being an effective and scalable strategy, discovered programs tend to be more interpretable than raw solutions, enabling feedback loops between domain experts and FunSearch, and the deployment of such programs in real-world applications.

- **链接 / Link：** [Nature 论文页](https://www.nature.com/articles/s41586-023-06924-6)；[DOI:10.1038/s41586-023-06924-6](https://doi.org/10.1038/s41586-023-06924-6)
- **年份 / Year：** 2024（期刊卷期；2023-12-14 在线发表）
- **发表信息 / Publication：** *Nature* 625, 468–475 (2024)

#### 23. AutoML-Zero：从零演化机器学习算法

**中文摘要：** 机器学习研究已在模型结构和学习方法等多个方面取得进展；自动化此类研究的 AutoML 也进步明显。但这些进展主要集中于神经网络架构，并依赖复杂的专家设计层作为构件，或采用同样受限的搜索空间。本文旨在说明 AutoML 可以走得更远：现在只用基本数学运算作为构件，就能自动发现完整的机器学习算法。为此，作者提出一种通过通用搜索空间显著减少人为偏见的新框架。尽管搜索空间极为庞大，演化搜索仍能发现通过反向传播训练的两层神经网络。随后，直接在目标任务（如 CIFAR-10 变体）上演化，可以超过这些简单神经网络；排名靠前的算法会出现双线性交互、归一化梯度和权重平均等现代技术。演化还会使算法适应不同任务类型，例如在数据较少时出现类似 dropout 的技术。这些从零发现机器学习算法的初步成果说明，该方向值得继续研究。

**English title:** AutoML-Zero: Evolving Machine Learning Algorithms From Scratch

**Abstract:** Machine learning research has advanced in multiple aspects, including model structures and learning methods. The effort to automate such research, known as AutoML, has also made significant progress. However, this progress has largely focused on the architecture of neural networks, where it has relied on sophisticated expert-designed layers as building blocks---or similarly restrictive search spaces. Our goal is to show that AutoML can go further: it is possible today to automatically discover complete machine learning algorithms just using basic mathematical operations as building blocks. We demonstrate this by introducing a novel framework that significantly reduces human bias through a generic search space. Despite the vastness of this space, evolutionary search can still discover two-layer neural networks trained by backpropagation. These simple neural networks can then be surpassed by evolving directly on tasks of interest, e.g. CIFAR-10 variants, where modern techniques emerge in the top algorithms, such as bilinear interactions, normalized gradients, and weight averaging. Moreover, evolution adapts algorithms to different task types: e.g., dropout-like techniques appear when little data is available. We believe these preliminary successes in discovering machine learning algorithms from scratch indicate a promising new direction for the field.

- **链接 / Link：** [arXiv:2003.03384](https://arxiv.org/abs/2003.03384)
- **年份 / Year：** 2020
- **发表信息 / Publication：** ICML 2020；arXiv:2003.03384

#### 24. AI-GA：生成 AI 的算法，一条通往通用人工智能的替代范式

**中文摘要：** 创造通用人工智能或许是人类历史上最雄心勃勃的科学探索之一；粗略地说，通用人工智能就是与人类同等聪明或更聪明的 AI。机器学习界的主流路线是分别发现智能所需的各个组件，并默认未来某个团队将完成把这些组件组合成复杂思维机器的艰巨任务。作者称之为“手工 AI 路线”。本文描述另一条令人期待、最终或许更有希望产生通用 AI 的路径。它基于机器学习领域一个清晰趋势：手工设计的方案最终会被更有效的学习所得方案取代。其思路是创造一种“生成 AI 的算法”（AI-GA），让它自动学习如何产生通用 AI。该路线需要三大支柱：（1）元学习架构；（2）对学习算法本身进行元学习；（3）生成有效的学习环境。作者认为，两条路线都可能率先产生通用 AI；无论哪一条更快，两者都具有科研价值。鉴于两条路线均有前景，而机器学习界目前主要投入手工路线，作者主张增加对 AI-GA 路线的研究投入。为推动这类研究，论文介绍了三大支柱中各自有前景的工作，也讨论 AI-GA 特有的安全和伦理问题。AI-GA 可能是通向通用 AI 的最快路径之一；理解简单算法在何种条件下能够产生通用 AI，本身也具有科学价值，就像地球上的达尔文演化产生人类智能一样。因此，作者主张将 AI-GA 研究视为计算机科学新的重大挑战。

**English title:** AI-GAs: AI-generating algorithms, an alternate paradigm for producing general artificial intelligence

**Abstract:** Perhaps the most ambitious scientific quest in human history is the creation of general artificial intelligence, which roughly means AI that is as smart or smarter than humans. The dominant approach in the machine learning community is to attempt to discover each of the pieces required for intelligence, with the implicit assumption that some future group will complete the Herculean task of figuring out how to combine all of those pieces into a complex thinking machine. I call this the "manual AI approach". This paper describes another exciting path that ultimately may be more successful at producing general AI. It is based on the clear trend in machine learning that hand-designed solutions eventually are replaced by more effective, learned solutions. The idea is to create an AI-generating algorithm (AI-GA), which automatically learns how to produce general AI. Three Pillars are essential for the approach: (1) meta-learning architectures, (2) meta-learning the learning algorithms themselves, and (3) generating effective learning environments. I argue that either approach could produce general AI first, and both are scientifically worthwhile irrespective of which is the fastest path. Because both are promising, yet the ML community is currently committed to the manual approach, I argue that our community should increase its research investment in the AI-GA approach. To encourage such research, I describe promising work in each of the Three Pillars. I also discuss AI-GA-specific safety and ethical considerations. Because it it may be the fastest path to general AI and because it is inherently scientifically interesting to understand the conditions in which a simple algorithm can produce general AI (as happened on Earth where Darwinian evolution produced human intelligence), I argue that the pursuit of AI-GAs should be considered a new grand challenge of computer science research.

- **链接 / Link：** [arXiv:1905.10985](https://arxiv.org/abs/1905.10985)
- **年份 / Year：** 2019
- **发表信息 / Publication：** arXiv 预印本，arXiv:1905.10985

#### 25. 成对开放式开拓者（POET）：持续生成日益复杂、多样的学习环境及其解法

**中文摘要：** 迄今为止，机器学习的发展史大体由研究者提出问题、算法学习解法这一系列过程构成。一个重要问题是：算法能否在求解问题的同时生成问题本身？这样的过程实际上会自行构建多样且不断扩展的课程，而不同阶段的问题解法会成为日后解决更难问题的垫脚石。本文提出的成对开放式开拓者（POET）算法正是如此：它把环境挑战的生成与求解这些挑战的智能体优化配对。POET 同时探索问题与解空间中的多条路径；尤为重要的是，如果某个解用于另一问题时表现更好，它就允许这类垫脚石式解法在问题之间迁移，从而催生创新。“开放式”意味着 POET 这类算法可能无止境地产生新颖且日益复杂的能力。结果显示，POET 产生了多种复杂行为，可以解决广泛的环境挑战；其中很多挑战无法仅凭直接优化解决，甚至作者为了说明开放性对解决高难度挑战的重要作用而设计的直接路径课程基线也无法解决。事实证明，跨环境迁移解法对释放整个系统的潜力不可或缺，也体现了偶然垫脚石的不可预测性。作者希望 POET 推动多个领域开展新的开放式发现研究，让同类算法在各种潜在形态及其解法之间开辟道路。

**English title:** Paired Open-Ended Trailblazer (POET): Endlessly Generating Increasingly Complex and Diverse Learning Environments and Their Solutions

**Abstract:** While the history of machine learning so far largely encompasses a series of problems posed by researchers and algorithms that learn their solutions, an important question is whether the problems themselves can be generated by the algorithm at the same time as they are being solved. Such a process would in effect build its own diverse and expanding curricula, and the solutions to problems at various stages would become stepping stones towards solving even more challenging problems later in the process. The Paired Open-Ended Trailblazer (POET) algorithm introduced in this paper does just that: it pairs the generation of environmental challenges and the optimization of agents to solve those challenges. It simultaneously explores many different paths through the space of possible problems and solutions and, critically, allows these stepping-stone solutions to transfer between problems if better, catalyzing innovation. The term open-ended signifies the intriguing potential for algorithms like POET to continue to create novel and increasingly complex capabilities without bound. Our results show that POET produces a diverse range of sophisticated behaviors that solve a wide range of environmental challenges, many of which cannot be solved by direct optimization alone, or even through a direct-path curriculum-building control algorithm introduced to highlight the critical role of open-endedness in solving ambitious challenges. The ability to transfer solutions from one environment to another proves essential to unlocking the full potential of the system as a whole, demonstrating the unpredictable nature of fortuitous stepping stones. We hope that POET will inspire a new push towards open-ended discovery across many domains, where algorithms like POET can blaze a trail through their interesting possible manifestations and solutions.

- **链接 / Link：** [arXiv:1901.01753](https://arxiv.org/abs/1901.01753)
- **年份 / Year：** 2019
- **发表信息 / Publication：** GECCO 2019；arXiv:1901.01753

#### 26. 用梯度下降学习如何用梯度下降来学习

**中文摘要：** 从手工设计特征转向学习特征，在机器学习中取得了巨大成功。尽管如此，优化算法仍由人工设计。本文说明，优化算法的设计可以表述为一个学习问题，使算法自动学会利用目标问题的结构。作者用 LSTM 实现的学习型算法，在训练所针对的任务上优于通用的手工设计优化器，并能很好地泛化到结构相似的新任务。论文在多种任务上验证了这一点，包括简单凸优化问题、神经网络训练，以及利用神经艺术进行图像风格化。

**English title:** Learning to learn by gradient descent by gradient descent

**Abstract:** The move from hand-designed features to learned features in machine learning has been wildly successful. In spite of this, optimization algorithms are still designed by hand. In this paper we show how the design of an optimization algorithm can be cast as a learning problem, allowing the algorithm to learn to exploit structure in the problems of interest in an automatic way. Our learned algorithms, implemented by LSTMs, outperform generic, hand-designed competitors on the tasks for which they are trained, and also generalize well to new tasks with similar structure. We demonstrate this on a number of tasks, including simple convex problems, training neural networks, and styling images with neural art.

- **链接 / Link：** [arXiv:1606.04474](https://arxiv.org/abs/1606.04474)
- **年份 / Year：** 2016
- **发表信息 / Publication：** NeurIPS 2016；arXiv:1606.04474

#### 27. 质量—多样性：演化计算的新前沿

**中文摘要：** 演化计算和演化机器人学虽从自然界获得灵感，却长期主要关注性能优化问题。但自然演化的含义比简单优化更丰富。具体来说，自然演化是一种发散式搜索：它一面在各个生态位内进行局部优化，一面持续产生多样性。这种同时发现质量与多样性的倾向，不同于许多传统机器学习算法，也为推断最具潜力的演化算法路线提供了不同基础。近年出现的多种质量—多样性（QD）算法，例如带局部竞争的新颖性搜索和 MAP-Elites，正是受到这一更细致的演化观启发；它们试图用每一类可实现行为中尽可能好的样本填充可能性空间，因而一次运行就能返回由多样、高质量行为组成的档案库。本文旨在研究 QD 算法在高难度环境，尤其是复杂迷宫中的应用，为未来用于更具挑战性的领域总结最佳实践。论文分析 QD 在何种情形下成功或失败，并研究一种新方法：在同一次运行中混合多种行为视角（称为行为表征）。当某一种行为表征本身不足以同时产生质量和多样性时，该方法能够克服部分相关搜索难题。

**English title:** Quality Diversity: A New Frontier for Evolutionary Computation

**Abstract:** While evolutionary computation and evolutionary robotics take inspiration from nature, they have long focused mainly on problems of performance optimization. Yet, evolution in nature can be interpreted as more nuanced than a process of simple optimization. In particular, natural evolution is a divergent search that optimizes locally within each niche as it simultaneously diversifies. This tendency to discover both quality and diversity at the same time differs from many of the conventional algorithms of machine learning, and also thereby suggests a different foundation for inferring the approach of greatest potential for evolutionary algorithms. In fact, several recent evolutionary algorithms called quality diversity (QD) algorithms (e.g., novelty search with local competition and MAP-Elites) have drawn inspiration from this more nuanced view, aiming to fill a space of possibilities with the best possible example of each type of achievable behavior. The result is a new class of algorithms that return an archive of diverse, high-quality behaviors in a single run. The aim in this paper is to study the application of QD algorithms in challenging environments (in particular complex mazes) to establish their best practices for ambitious domains in the future. In addition to providing insight into cases when QD succeeds and fails, a new approach is investigated that hybridizes multiple views of behaviors (called behavior characterizations) in the same run, which succeeds in overcoming some of the challenges associated with searching for QD with respect to a behavior characterization that is not necessarily sufficient for generating both quality and diversity at the same time.

- **链接 / Link：** [Frontiers 论文页](https://www.frontiersin.org/journals/robotics-and-ai/articles/10.3389/frobt.2016.00040/full)；[DOI:10.3389/frobt.2016.00040](https://doi.org/10.3389/frobt.2016.00040)
- **年份 / Year：** 2016
- **发表信息 / Publication：** *Frontiers in Robotics and AI*, 3:40 (2016)

#### 28. 通过映射精英照亮搜索空间

**中文摘要：** 许多领域使用搜索算法自动探索搜索空间，以寻找高性能方案：化学家在分子空间中寻找新药，工程师寻找更坚固、更便宜、更安全的设计，科学家寻找最能解释数据的模型，等等。传统搜索算法以返回搜索空间中性能最高的单一方案为目标。本文描述一种全新的算法类型，它能呈现高性能方案在整个搜索空间中的分布全貌，因此更为实用。该算法在由用户选定的变化维度所定义的空间中，为每个点创建高性能方案映射。这个“表型精英多维档案”（MAP-Elites）算法能够照亮搜索空间，帮助研究者了解解的有趣属性如何组合，以及这些组合如何正面或负面影响性能。例如，制药企业可能希望了解分子大小和生产成本变化时，药物性能如何变化。MAP-Elites 会产生大量性能高而性质不同的解，这往往比一个高性能解更有用。有意思的是，因为 MAP-Elites 探索了更多搜索空间，它也倾向于找到优于当时先进搜索算法的总体最优解。作者在三个不同问题领域证明了新算法的优势，范围从生成模块化神经网络到设计仿真和实体软体机器人。MAP-Elites 能揭示性能与解的关注维度之间的关系，返回一组多样的高性能解，还能改善单一最优解的寻找，因此可用于推进科学与工程研究。

**English title:** Illuminating search spaces by mapping elites

**Abstract:** Many fields use search algorithms, which automatically explore a search space to find high-performing solutions: chemists search through the space of molecules to discover new drugs; engineers search for stronger, cheaper, safer designs, scientists search for models that best explain data, etc. The goal of search algorithms has traditionally been to return the single highest-performing solution in a search space. Here we describe a new, fundamentally different type of algorithm that is more useful because it provides a holistic view of how high-performing solutions are distributed throughout a search space. It creates a map of high-performing solutions at each point in a space defined by dimensions of variation that a user gets to choose. This Multi-dimensional Archive of Phenotypic Elites (MAP-Elites) algorithm illuminates search spaces, allowing researchers to understand how interesting attributes of solutions combine to affect performance, either positively or, equally of interest, negatively. For example, a drug company may wish to understand how performance changes as the size of molecules and their cost-to-produce vary. MAP-Elites produces a large diversity of high-performing, yet qualitatively different solutions, which can be more helpful than a single, high-performing solution. Interestingly, because MAP-Elites explores more of the search space, it also tends to find a better overall solution than state-of-the-art search algorithms. We demonstrate the benefits of this new algorithm in three different problem domains ranging from producing modular neural networks to designing simulated and real soft robots. Because MAP-Elites (1) illuminates the relationship between performance and dimensions of interest in solutions, (2) returns a set of high-performing, yet diverse solutions, and (3) improves finding a single, best solution, it will advance science and engineering.

- **链接 / Link：** [arXiv:1504.04909](https://arxiv.org/abs/1504.04909)
- **年份 / Year：** 2015
- **发表信息 / Publication：** arXiv 预印本，arXiv:1504.04909

#### 29. POWERPLAY：通过持续寻找最简单但尚无法解决的问题，训练日益通用的问题求解器

**中文摘要：** 计算机科学大多关注自动解决给定的计算问题。本文转而关注如何以受动物和人类玩耍行为启发的方式，自动发明或发现问题，从零开始以无监督方式训练越来越通用的问题求解器。考虑所有可计算任务描述组成的无限集合，其中任务的解也可能是可计算的。2011 年提出的新算法框架 POWERPLAY 持续搜索“新任务与当前求解器修改”的可能配对，直到找到一个更强的求解器：它可以被证明既能解决所有此前学过的任务，也能解决新任务，而修改前的求解器无法解决该新任务。系统通过偏好那些可复用已有技能、且新增技能较简单的新任务，迫使自己变得比过去更通用。若计算资源无限，它将任意给定的初始求解器逐步变成任意给定计算机上任意用户定义问题类别的最通用、最高效的问题求解器之一。要朝这一目标取得可行进展，本文引入一种特定的 POWERPLAY 实现，它采用渐近最优的程序搜索方法：搜索那些在给定所有既有知识的条件下，额外时间与空间复杂度较低的新任务以及相应的求解器修改。新的求解器必须确实解决新任务及此前所有任务；修改前的求解器则不能解决新任务。传统通用搜索算法采用由算法信息论导出的简洁性和效率原则来寻找已有问题的解，POWERPLAY 将类似原则用于寻找新的、尚未解决的问题及其求解器修改。它偏好易于验证、能让求解器变得更通用、更高效的新技能；新技能还可以部分复用已有技能。POWERPLAY 按照候选任务与求解器修改在给定已有经验后的条件计算复杂度（时间和空间）来排序，首先找到并验证的就是新任务及其对应技能。验证新任务的计算成本不必随任务库规模增长。POWERPLAY 对新颖性的持续搜索不断突破当前求解器的泛化能力。这与哥德尔逐步增强的形式理论序列有关：该序列把此前不可证明的陈述加入公理，同时不影响此前可证明的定理。持续增长的问题求解程序库还能由并行搜索利用，用于解决额外的外部任务。POWERPLAY 可以看作创造力基本原则的一种贪心但实用的实现；初步实验分析另见论文 [53,54]。

**English title:** POWERPLAY: Training an Increasingly General Problem Solver by Continually Searching for the Simplest Still Unsolvable Problem

**Abstract:** Most of computer science focuses on automatically solving given computational problems. I focus on automatically inventing or discovering problems in a way inspired by the playful behavior of animals and humans, to train a more and more general problem solver from scratch in an unsupervised fashion. Consider the infinite set of all computable descriptions of tasks with possibly computable solutions. The novel algorithmic framework POWERPLAY (2011) continually searches the space of possible pairs of new tasks and modifications of the current problem solver, until it finds a more powerful problem solver that provably solves all previously learned tasks plus the new one, while the unmodified predecessor does not. It forces itself to become increasingly more general than it used to be, by preferring new tasks for which the additional skills are easy to learn, given the skills already learned. With unlimited computational resources, it will incrementally transform any given initial solver into one of the most general and efficient problem solvers possible on any given computer, for arbitrary user-defined problem classes. To make feasible progress on this ambitious goal, I introduce a particular implementation of POWERPLAY based on a non-optimally ordered but asymptotically optimal program search method, which searches not only for new tasks to be added to the cumulative repertoire, but also for modifications of the current solver that are computationally cheap, given the other things already learned. The new solver must provably solve the new task and all previously learned tasks, while the unmodified previous solver must not. Traditional universal search algorithms for finding solutions to known problems are based on principles of simplicity and efficiency derived from algorithmic information theory. POWERPLAY uses similar principles to find both new, still unsolvable problems and corresponding solver modifications. It favors new skills that are easy to verify, but that make the solver more general and more efficient at solving previously learned problems. New skills may (partially) re-use previously learned skills. POWERPLAY's search orders candidate pairs of tasks and solver modifications by their conditional computational (time & space) complexity, given the stored experience so far. The new task and its corresponding task-solving skill are those first found and validated. The computational costs of validating new tasks need not grow with task repertoire size. POWERPLAY's ongoing search for novelty keeps breaking the generalization abilities of its present solver. This is related to Goedel's sequence of increasingly powerful formal theories based on adding formerly unprovable statements to the axioms without affecting previously provable theorems. The continually increasing repertoire of problem solving procedures can be exploited by a parallel search for solutions to additional externally posed tasks. POWERPLAY may be viewed as a greedy but practical implementation of basic principles of creativity. A first experimental analysis can be found in separate papers [53,54].

- **链接 / Link：** [arXiv:1112.5309](https://arxiv.org/abs/1112.5309)
- **年份 / Year：** 2011（预印本首次提交）；2013（期刊）
- **发表信息 / Publication：** *Frontiers in Psychology* 4:313 (2013)；arXiv:1112.5309

## 安全、对齐与理论 / Safety, Alignment & Theory

本节关注会随时间修改自身或持续自我改进的系统，讨论其安全性、稳定性、目标保持、可纠正性与监督机制。
Safety, stability, goal preservation, corrigibility, and oversight mechanisms for systems that modify or improve themselves over time.

#### 1. 递归自我改进的经济学

**完整中文摘要：** 我们对递归自我改进（RSI）的经济学进行建模，并评估其可能性与影响。首先，我们构建了一系列逐渐丰富的 AI 进展模型，以说明 RSI 背后的反馈回路。我们用有向图表示这些模型，并表明 AI 能力的净加速取决于每条反馈回路上各项弹性的乘积。其次，我们区分“窄域”与“广域”AI 能力，以涵盖这样一种可能性：AI 系统在优化 AI 研发基准方面取得窄域进步，却没有在更广泛且具有经济价值的任务上同步提升。第三，我们整理了关键参数的现有估计，并列出一份希望 AI 公司能够测量且切实可行地公开分享的经验对象清单。最后，我们用现有数据校准模型。一项粗略估算表明，目前反馈回路尚不足以产生可自我维持的加速，不过其强度似乎正在上升。最后，我们评估了这种加速的可能性及其影响。

**English title:** The Economics of Recursive Self-Improvement

**Abstract:** We model the economics of recursive self-improvement (RSI) and assess its plausibility and impacts. First, we build a sequence of increasingly rich models of AI progress to highlight the feedback loops behind RSI. We represent our models as directed graphs and show that net acceleration in AI capabilities depends on the product of elasticities across each feedback loop. Second, we distinguish between “narrow” and “broad” AI capabilities, capturing the possibility that AI systems improve narrowly at optimizing AI R&D benchmarks without improving at broader economically valuable tasks. Third, we document existing estimates of key parameters, and provide a wish list of empirical objects that AI companies can measure and feasibly share publicly. Finally, we calibrate the model with existing data. A back-of-the-envelope calculation suggests that feedback loops are not currently strong enough to generate a self-sustaining acceleration, though they appear to be strengthening. We conclude by assessing the plausibility and implications of such an acceleration.

- **链接**：[Elasticity Institute 官方 PDF](https://elasticity.institute/rsi-paper.pdf)
- **年份**：2026
- **发表信息**：Elasticity Institute 研究论文，2026 年 7 月 13 日。

#### 2. SAHOO：面向递归自我改进中高阶优化目标的安全对齐

**完整中文摘要：** 递归自我改进正从理论走向实践：现代系统已经能够批评、修订并评估自身输出，但反复的自我修改可能导致细微的对齐漂移。我们提出 SAHOO，一个通过三项保障措施监测和控制漂移的实用框架：（i）目标漂移指数（GDI），一种融合语义、词汇、结构和分布度量的学习型多信号检测器；（ii）约束保持检查，用于执行语法正确性和无幻觉等安全攸关的不变量；（iii）回归风险量化，用于标记那些会抵消既有改进的提升周期。在代码生成、数学推理和真实性三类共 189 项任务中，SAHOO 显著提高了质量，其中代码任务提升 18.3%，推理任务提升 16.8%；与此同时，它在两个领域中保持了约束，并在真实性任务中维持较低的违规率。阈值是在一个由 18 项任务构成的小型验证集上、跨三个周期校准的。我们还绘制了能力—对齐前沿：早期改进周期效率较高，但后期对齐成本上升，同时暴露出流畅性与事实性等因领域而异的张力。因此，SAHOO 使递归自我改进过程中的对齐保持变得可测量、可部署，并得到大规模系统验证。

**English title:** SAHOO: Safeguarded Alignment for High-Order Optimization Objectives in Recursive Self-Improvement

**Abstract:** Recursive self-improvement is moving from theory to practice: modern systems can critique, revise, and evaluate their own outputs, yet iterative self-modification risks subtle alignment drift. We introduce SAHOO, a practical framework to monitor and control drift through three safeguards: (i) the Goal Drift Index (GDI), a learned multi-signal detector combining semantic, lexical, structural, and distributional measures; (ii) constraint preservation checks that enforce safety-critical invariants such as syntactic correctness and non-hallucination; and (iii) regression-risk quantification to flag improvement cycles that undo prior gains. Across 189 tasks in code generation, mathematical reasoning, and truthfulness, SAHOO produces substantial quality gains, including 18.3 percent improvement in code tasks and 16.8 percent in reasoning, while preserving constraints in two domains and maintaining low violations in truthfulness. Thresholds are calibrated on a small validation set of 18 tasks across three cycles. We further map the capability-alignment frontier, showing efficient early improvement cycles but rising alignment costs later and exposing domain-specific tensions such as fluency versus factuality. SAHOO therefore makes alignment preservation during recursive self-improvement measurable, deployable, and systematically validated at scale.

- **链接**：[arXiv:2603.06333v1](https://arxiv.org/abs/2603.06333v1)
- **年份**：2026
- **发表信息**：ICLR 2026 Workshop on Recursive Self-Improvement；arXiv 预印本，cs.AI、cs.CL、cs.LG。

#### 3. TamperBench：系统性压力测试微调与篡改下的大语言模型安全性

**完整中文摘要：** 随着能力日益增强的开放权重大语言模型（LLM）投入部署，提高它们抵抗不安全修改（无论意外还是蓄意）的能力，对于降低风险变得十分重要。然而，目前还没有评估抗篡改能力的标准方法。数据集、指标和篡改配置各不相同，使不同模型与防御方法之间的安全性、效用和稳健性难以比较。为解决这一问题，我们提出 TamperBench，这是首个系统评估 LLM 抗篡改能力的统一框架。TamperBench：（i）汇集当前先进的权重空间微调攻击、潜在空间表征攻击和对齐阶段防御方法；（ii）针对每个攻击—模型组合进行系统的超参数扫描，以实现现实的对抗性评估；（iii）同时提供安全性与效用评估。我们使用 TamperBench，在九种篡改威胁下评估了 21 个开放权重 LLM，其中包括加入防御的变体；评估采用标准化的安全与能力指标，并对每个模型—攻击组合进行超参数扫描。结果给出了若干认识，包括后训练对抗篡改能力的影响、越狱微调通常是最严重的攻击，以及当前对齐阶段的防御大多无法承受攻击扫描。代码见 https://github.com/criticalml-uw/TamperBench。

**English title:** TamperBench: Systematically Stress-Testing LLM Safety Under Fine-Tuning and Tampering

**Abstract:** As increasingly capable open-weight large language models (LLMs) are deployed, improving their tamper resistance against unsafe modifications, whether accidental or intentional, becomes critical to minimize risks. However, there is no standard approach to evaluate tamper resistance. Varied datasets, metrics, and tampering configurations make it difficult to compare safety, utility, and robustness across different models and defenses. To address this, we introduce TamperBench, the first unified framework to systematically evaluate the tamper resistance of LLMs. TamperBench (i) curates a repository of state-of-the-art weight-space fine-tuning attacks, latent-space representation attacks, and alignment-stage defenses; (ii) enables realistic adversarial evaluation through systematic hyperparameter sweeps per attack-model pair; and (iii) provides both safety and utility evaluations. We use TamperBench to evaluate 21 open-weight LLMs, including defense-augmented variants, across nine tampering threats using standardized safety and capability metrics with hyperparameter sweeps per model-attack pair. The results provide insights including effects of post-training on tamper resistance, that jailbreak-tuning is typically the most severe attack, and that current alignment-stage defenses largely fail to withstand attack sweeps. Code is available at https://github.com/criticalml-uw/TamperBench.

- **链接**：[arXiv:2602.06911v2](https://arxiv.org/abs/2602.06911v2)
- **年份**：2026
- **发表信息**：arXiv 预印本，cs.CR、cs.AI；DOI：[10.1145/3770855.3817557](https://doi.org/10.1145/3770855.3817557)。

#### 4. 你的智能体可能发生错误演化：自演化大语言模型智能体中的涌现风险

**完整中文摘要：** 大语言模型（LLM）的进展催生了一类新的自演化智能体，它们通过与环境交互自主改进，并表现出很强的能力。然而，自演化也带来了当前安全研究所忽视的新风险。在本研究中，我们考察智能体的自演化以非预期方式偏离并导致不良甚至有害结果的情形，将其称为“错误演化”（Misevolution）。为开展系统研究，我们沿四条主要演化路径评估错误演化：模型、记忆、工具和工作流。实证结果表明，错误演化是一种广泛存在的风险，即使基于顶尖 LLM（例如 Gemini-2.5-Pro）构建的智能体也会受到影响。自演化过程中出现了不同的涌现风险，例如记忆累积后安全对齐退化，或者在工具创建和复用时无意引入漏洞。据我们所知，这是首项系统定义错误演化概念并为其发生提供实证证据的研究，说明自演化智能体迫切需要新的安全范式。最后，我们讨论了可能的缓解策略，希望推动对更安全、更可信的自演化智能体的进一步研究。代码和数据见 https://github.com/ShaoShuai0605/Misevolution 。警告：本文包含可能具有冒犯性或有害性质的示例。

**English title:** Your Agent May Misevolve: Emergent Risks in Self-evolving LLM Agents

**Abstract:** Advances in Large Language Models (LLMs) have enabled a new class of self-evolving agents that autonomously improve through interaction with the environment, demonstrating strong capabilities. However, self-evolution also introduces novel risks overlooked by current safety research. In this work, we study the case where an agent's self-evolution deviates in unintended ways, leading to undesirable or even harmful outcomes. We refer to this as Misevolution. To provide a systematic investigation, we evaluate misevolution along four key evolutionary pathways: model, memory, tool, and workflow. Our empirical findings reveal that misevolution is a widespread risk, affecting agents built even on top-tier LLMs (e.g., Gemini-2.5-Pro). Different emergent risks are observed in the self-evolutionary process, such as the degradation of safety alignment after memory accumulation, or the unintended introduction of vulnerabilities in tool creation and reuse. To our knowledge, this is the first study to systematically conceptualize misevolution and provide empirical evidence of its occurrence, highlighting an urgent need for new safety paradigms for self-evolving agents. Finally, we discuss potential mitigation strategies to inspire further research on building safer and more trustworthy self-evolving agents. Our code and data are available at https://github.com/ShaoShuai0605/Misevolution . Warning: this paper includes examples that may be offensive or harmful in nature.

- **链接**：[arXiv:2509.26354v2](https://arxiv.org/abs/2509.26354v2)
- **年份**：2025
- **发表信息**：ICLR 2026；arXiv 预印本，cs.AI、cs.CL、cs.LG（初次提交于 2025 年）。

#### 5. 通过合成数据验证摆脱模型崩溃：近期改进与长期收敛

**完整中文摘要：** 合成数据越来越多地用于训练前沿生成模型。然而，近期研究提出了一个重要问题：在模型自行生成的合成数据上反复重新训练生成模型，可能会持续降低模型性能，这一现象通常称为模型崩溃。本文研究如何修改合成再训练流程，以避免模型崩溃，甚至可能帮助扭转从崩溃走向改进的趋势。我们的主要发现是：通过外部合成数据验证器注入信息，无论验证者是人类还是更好的模型，合成再训练都不会引发模型崩溃。具体而言，我们在基础的线性回归设定中开展理论分析，表明验证器引导的再训练可以带来近期改进，但长期来看，最终会使参数估计趋向验证器的“知识中心”。理论还预测，除非验证器完全可靠，否则这些早期收益将进入平台期，甚至可能逆转。我们在线性回归、基于 MNIST 训练的变分自编码器（VAE），以及在 XSUM 任务上微调 SmolLM2-135M 的实验，确实验证了这些理论认识。

**English title:** Escaping Model Collapse via Synthetic Data Verification: Near-term Improvements and Long-term Convergence

**Abstract:** Synthetic data has been increasingly used to train frontier generative models. However, recent studies raise key concerns that iteratively retraining a generative model on its self-generated synthetic data may keep deteriorating model performance, a phenomenon often coined model collapse. In this paper, we investigate ways to modify the synthetic retraining process to avoid model collapse, and even possibly help reverse the trend from collapse to improvement. Our key finding is that by injecting information through an external synthetic data verifier, whether a human or a better model, synthetic retraining will not cause model collapse. Specifically, we situate our theoretical analysis in the fundamental linear regression setting, showing that verifier-guided retraining can yield near-term improvements, but ultimately drives the parameter estimate to the verifier's "knowledge center" in the long run. Our theory further predicts that, unless the verifier is perfectly reliable, these early gains will plateau and may even reverse. Indeed, our experiments across linear regression, Variational Autoencoders (VAEs) trained on MNIST, and fining-tuning SmolLM2-135M on the XSUM task confirm these theoretical insights.

- **链接**：[arXiv:2510.16657v3](https://arxiv.org/abs/2510.16657v3)
- **年份**：2025
- **发表信息**：arXiv 预印本，stat.ML、cs.LG。

#### 6. 评估语言模型智能体的目标漂移（技术报告）

**完整中文摘要：** 随着语言模型（LM）越来越多地被部署为自主智能体，它们能否稳定遵循人类指定的目标，对安全运行变得十分重要。当这些智能体在没有人类监督的情况下长时间独立运行时，即使最初设定清楚的目标也可能逐渐改变。检测和测量目标漂移，也就是智能体随时间偏离原始目标的倾向，面临很大困难，因为目标可能缓慢变化，只引起细微的行为改变。本文提出一种分析 LM 智能体目标漂移的新方法。在实验中，我们先通过系统提示明确赋予智能体一个目标，再通过环境压力使其接触相互竞争的目标。结果表明，在最困难的评估设定中，表现最好的智能体（经过脚手架增强的 Claude 3.5 Sonnet）能在超过 100,000 个 token 的范围内保持近乎完美的目标遵循；但所有被评估模型都表现出某种程度的目标漂移。我们还发现，随着上下文长度增加，目标漂移与模型越来越容易出现模式匹配行为相关。

**English title:** Technical Report: Evaluating Goal Drift in Language Model Agents

**Abstract:** As language models (LMs) are increasingly deployed as autonomous agents, their robust adherence to human-assigned objectives becomes crucial for safe operation. When these agents operate independently for extended periods without human oversight, even initially well-specified goals may gradually shift. Detecting and measuring goal drift - an agent's tendency to deviate from its original objective over time - presents significant challenges, as goals can shift gradually, causing only subtle behavioral changes. This paper proposes a novel approach to analyzing goal drift in LM agents. In our experiments, agents are first explicitly given a goal through their system prompt, then exposed to competing objectives through environmental pressures. We demonstrate that while the best-performing agent (a scaffolded version of Claude 3.5 Sonnet) maintains nearly perfect goal adherence for more than 100,000 tokens in our most difficult evaluation setting, all evaluated models exhibit some degree of goal drift. We also find that goal drift correlates with models' increasing susceptibility to pattern-matching behaviors as the context length grows.

- **链接**：[arXiv:2505.02709v1](https://arxiv.org/abs/2505.02709v1)
- **年份**：2025
- **发表信息**：技术报告；arXiv 预印本，cs.AI、cs.LG。

#### 7. 具备自我修改能力的有限理性智能体之性能

**完整中文摘要：** 嵌入复杂环境的智能体很难避免自我修改，无论这种修改通过直接方式发生（例如修改自身代码），还是通过间接方式发生（例如影响操作者、利用程序缺陷或环境）。有人提出，智能体有动机避免修改自身效用函数，从而使其未来实例继续朝相同目标努力。

Everitt 等人（2016）从形式上证明，对于完全理性的智能体，提供自我修改选项并无危害。我们表明，这一结果对有限理性智能体不再成立。对这类智能体而言，自我修改可能造成性能指数级恶化，并使原本对齐的智能体逐渐失去对齐。我们研究了这种影响的大小如何取决于智能体理性缺陷的类型和程度（见下述第 1 至第 4 类）。我们还讨论了模型假设，以及更广泛的问题空间与问题框架。

我们考察智能体可能表现出有限理性的四种方式：（1）并非总是选择最优行动；（2）没有与人类价值完全对齐；（3）对环境持有不准确的模型；或（4）使用错误的时间折扣因子。我们表明，在第（2）至（4）种情形中，智能体缺陷造成的失配不会随时间加剧；而在第（1）种情形中，失配可能呈指数增长。

**English title:** Performance of Bounded-Rational Agents With the Ability to Self-Modify

**Abstract:** Self-modification of agents embedded in complex environments is hard to avoid, whether it happens via direct means (e.g. own code modification) or indirectly (e.g. influencing the operator, exploiting bugs or the environment). It has been argued that intelligent agents have an incentive to avoid modifying their utility function so that their future instances work towards the same goals.

Everitt et al. (2016) formally show that providing an option to self-modify is harmless for perfectly rational agents. We show that this result is no longer true for agents with bounded rationality. In such agents, self-modification may cause exponential deterioration in performance and gradual misalignment of a previously aligned agent. We investigate how the size of this effect depends on the type and magnitude of imperfections in the agent's rationality (1-4 below). We also discuss model assumptions and the wider problem and framing space.

We examine four ways in which an agent can be bounded-rational: it either (1) doesn't always choose the optimal action, (2) is not perfectly aligned with human values, (3) has an inaccurate model of the environment, or (4) uses the wrong temporal discounting factor. We show that while in the cases (2)-(4) the misalignment caused by the agent's imperfection does not increase over time, with (1) the misalignment may grow exponentially.

- **链接**：[arXiv:2011.06275v2](https://arxiv.org/abs/2011.06275v2)
- **年份**：2021
- **发表信息**：AAAI 2021 Workshop on Artificial Intelligence Safety（SafeAI）；arXiv 初稿提交于 2020 年，v2 更新于 2021 年。

#### 8. 通过迭代改进效用函数保障 AGI 智能体安全

**完整中文摘要：** 尽管我们仍不清楚能否真正构建出具备通用人工智能（AGI）的智能体，但已经可以使用数学模型研究这些智能体的潜在安全系统。我们提出一个 AGI 安全层，它创建专用输入终端，以支持对 AGI 智能体效用函数的迭代改进。启动该智能体的人可以使用这个终端修补效用函数在编码智能体目标与约束时发现的漏洞，引导智能体转向新目标，或者强制智能体关闭自身。AGI 智能体可能产生操纵上述效用函数改进过程的涌现动机，例如欺骗、限制甚至攻击参与其中的人类。该安全层会部分抑制这种危险动机，有时能够完全抑制。本文第一部分推广了早期关于 AGI 紧急停止按钮的工作。我们把相关方法应用于马尔可夫决策过程（MDP）模型，希望使构建该安全层所用的数学方法更易理解。我们讨论了安全层的两个可证明性质，并介绍了将其映射为因果影响图（CID）的在研工作。第二部分给出完整的数学证明，并表明安全层会产生一种“官僚式盲视”。随后，我们提出一种学习智能体设计：在一个已知机器学习系统或未来可能达到 AGI 水平的学习系统外部封装该安全层。由此得到的智能体从首次启动时起便满足这些可证明的安全性质。最后，我们说明如何将该智能体从模型映射到现实实现，回顾这一步涉及的方法论问题，并讨论通常如何解决这些问题。

**English title:** AGI Agent Safety by Iteratively Improving the Utility Function

**Abstract:** While it is still unclear if agents with Artificial General Intelligence (AGI) could ever be built, we can already use mathematical models to investigate potential safety systems for these agents. We present an AGI safety layer that creates a special dedicated input terminal to support the iterative improvement of an AGI agent's utility function. The humans who switched on the agent can use this terminal to close any loopholes that are discovered in the utility function's encoding of agent goals and constraints, to direct the agent towards new goals, or to force the agent to switch itself off. An AGI agent may develop the emergent incentive to manipulate the above utility function improvement process, for example by deceiving, restraining, or even attacking the humans involved. The safety layer will partially, and sometimes fully, suppress this dangerous incentive. The first part of this paper generalizes earlier work on AGI emergency stop buttons. We aim to make the mathematical methods used to construct the layer more accessible, by applying them to an MDP model. We discuss two provable properties of the safety layer, and show ongoing work in mapping it to a Causal Influence Diagram (CID). In the second part, we develop full mathematical proofs, and show that the safety layer creates a type of bureaucratic blindness. We then present the design of a learning agent, a design that wraps the safety layer around either a known machine learning system, or a potential future AGI-level learning system. The resulting agent will satisfy the provable safety properties from the moment it is first switched on. Finally, we show how this agent can be mapped from its model to a real-life implementation. We review the methodological issues involved in this step, and discuss how these are typically resolved.

- **链接**：[arXiv:2007.05411v1](https://arxiv.org/abs/2007.05411v1)
- **年份**：2020
- **发表信息**：Artificial General Intelligence 2020（AGI 2020）；arXiv 预印本，cs.AI。

#### 9. 通过奖励建模实现可扩展的智能体对齐：一个研究方向

**完整中文摘要：** 将强化学习算法应用于现实问题的一项障碍，是缺少合适的奖励函数。设计这类奖励函数之所以困难，部分原因在于用户对任务目标只有隐含的理解。这便产生了智能体对齐问题：如何创建行为符合用户意图的智能体？我们勾勒出一个以奖励建模为中心、用于解决智能体对齐问题的高层研究方向：通过与用户交互学习奖励函数，再利用强化学习优化所学到的奖励函数。我们讨论了把奖励建模扩展到复杂通用领域时预计会面临的主要挑战、缓解这些挑战的具体方法，以及如何建立对所得智能体的信任。

**English title:** Scalable agent alignment via reward modeling: a research direction

**Abstract:** One obstacle to applying reinforcement learning algorithms to real-world problems is the lack of suitable reward functions. Designing such reward functions is difficult in part because the user only has an implicit understanding of the task objective. This gives rise to the agent alignment problem: how do we create agents that behave in accordance with the user's intentions? We outline a high-level research direction to solve the agent alignment problem centered around reward modeling: learning a reward function from interaction with the user and optimizing the learned reward function with reinforcement learning. We discuss the key challenges we expect to face when scaling reward modeling to complex and general domains, concrete approaches to mitigate these challenges, and ways to establish trust in the resulting agents.

- **链接**：[arXiv:1811.07871v1](https://arxiv.org/abs/1811.07871v1)
- **年份**：2018
- **发表信息**：arXiv 预印本，cs.LG、cs.AI、cs.NE、stat.ML。

#### 10. 理性智能体的策略与效用函数自我修改

**完整中文摘要：** 任何既属于其交互环境的一部分、又配有通用执行器（例如手臂和手指）的智能体，原则上都具备自我修改能力，例如更改自身源代码。随着我们不断创造越来越智能的智能体，它们发现这种能力的可能性也会提高。问题在于：它们会想使用这种能力吗？例如，高度智能的系统可能找到办法，把自身目标改成更容易实现的目标，从而“逃离”设计者的控制。Omohundro（2008）在一篇重要论文中提出，目标保持是任何智能系统的一项基本驱动力，因为如果智能体未来的版本继续追求相同目标，实现该目标的可能性会更高。本文在一般强化学习框架中形式化这一论证，并考察它会失效的情形。我们的结论是：当且仅当智能体的价值函数能够预见自我修改的后果，并在评估未来时使用当前效用函数，自我修改的可能性才是无害的。

**English title:** Self-Modification of Policy and Utility Function in Rational Agents

**Abstract:** Any agent that is part of the environment it interacts with and has versatile actuators (such as arms and fingers), will in principle have the ability to self-modify -- for example by changing its own source code. As we continue to create more and more intelligent agents, chances increase that they will learn about this ability. The question is: will they want to use it? For example, highly intelligent systems may find ways to change their goals to something more easily achievable, thereby `escaping' the control of their designers. In an important paper, Omohundro (2008) argued that goal preservation is a fundamental drive of any intelligent system, since a goal is more likely to be achieved if future versions of the agent strive towards the same goal. In this paper, we formalise this argument in general reinforcement learning, and explore situations where it fails. Our conclusion is that the self-modification possibility is harmless if and only if the value function of the agent anticipates the consequences of self-modifications and use the current utility function when evaluating the future.

- **链接**：[arXiv:1605.03142v1](https://arxiv.org/abs/1605.03142v1)
- **年份**：2016
- **发表信息**：Artificial General Intelligence 2016（AGI 2016）；arXiv 预印本，cs.AI。

#### 11. 智能爆炸微观经济学

**完整中文摘要：** I. J. Good 的“智能爆炸”论断认为，足够先进的机器智能可以构建一个比自身更聪明的版本，后者又能构建出更加聪明的版本；这一过程可能持续下去，直至机器智能远远超过人类智能。正如 Sandberg（2010）所正确指出的，已有若干尝试提出投资回报公式，以表示经济或技术增长的急剧加速，但几乎没有研究试图对 Good 的智能爆炸论断本身作形式化处理。

我认为，核心问题是认知再投资的回报，即投入更多算力、更快的计算机或改进后的认知算法，以获得能够产出更大规模大脑、更快大脑或更优心智设计的认知劳动。世界上许多现象都被认为与这一问题存在证据上的关联，包括人科演化的已知历程、摩尔定律以及机器国际象棋系统随时间变化的能力等。我较为深入地讨论了围绕如何解释这类证据而产生的一些争论。我提出，分析各种智能爆炸立场的下一步，应当是形式化投资回报曲线，使每种立场都能正式说明：它认为哪些可能的微观基础已经被既有历史观察所证伪。更一般地说，我提出了多个关于“认知再投资回报”或“智能爆炸微观经济学”的开放问题。尽管这类问题迄今很少受到关注，但它们似乎与影响地球起源智能生命之结局的政策选择高度相关。

**English title:** Intelligence Explosion Microeconomics

**Abstract:** I. J. Good’s thesis of the “intelligence explosion” states that a sufficiently advanced machine intelligence could build a smarter version of itself, which could in turn build an even smarter version, and that this process could continue to the point of vastly exceeding human intelligence. As Sandberg (2010) correctly notes, there have been several attempts to lay down return on investment formulas intended to represent sharp speedups in economic or technological growth, but very little attempt has been made to deal formally with Good’s intelligence explosion thesis as such.

I identify the key issue as returns on cognitive reinvestment—the ability to invest more computing power, faster computers, or improved cognitive algorithms to yield cognitive labor which produces larger brains, faster brains, or better mind designs. There are many phenomena in the world which have been argued to be evidentially relevant to this question, from the observed course of hominid evolution, to Moore’s Law, to the competence over time of machine chess-playing systems, and many more. I go into some depth on some debates which then arise on how to interpret such evidence. I propose that the next step in analyzing positions on the intelligence explosion would be to formalize return on investment curves, so that each stance can formally state which possible microfoundations they hold to be falsified by historical observations. More generally, I pose multiple open questions of “returns on cognitive reinvestment” or “intelligence explosion microeconomics.” Although such questions have received little attention thus far, they seem highly relevant to policy choices affecting outcomes for Earth-originating intelligent life.

- **链接**：[MIRI 官方 PDF](https://intelligence.org/files/IEM.pdf)
- **年份**：2013
- **发表信息**：Machine Intelligence Research Institute Technical Report 2013-1；最后修订于 2013 年 9 月 13 日。

### 安全使能基础 / Supporting Safety Foundations

本节收录与递归自我改进安全直接相关的基础研究，涵盖能力隐匿、欺骗策略、极端风险评估、原则式监督、权力寻求、奖励篡改、内部优化器、具体安全问题和安全中断。
Foundational work relevant to the safety of recursive self-improvement, covering capability concealment, deceptive policies, extreme-risk evaluation, principle-based supervision, power seeking, reward tampering, learned optimizers, concrete safety problems, and safe interruption.

#### 12. AI 隐匿能力：语言模型可在评估中策略性地降低表现

**完整中文摘要：** 可信的能力评估对于保障 AI 系统安全十分重要，并正成为 AI 监管的主要组成部分。然而，AI 系统的开发者或 AI 系统本身可能有动机让评估低估 AI 的实际能力。这种利益冲突带来了“隐匿能力”（sandbagging）问题，我们将其定义为在评估中策略性地表现不佳。本文评估了当代语言模型（LM）的隐匿能力。我们通过提示，让 GPT-4、Claude 3 Opus 等前沿 LM 在危险能力评估中选择性降低表现，同时保持其在一般（无害）能力评估中的表现。我们还发现，可以在合成数据集上微调模型，使其隐藏特定能力，除非输入密码。该行为可泛化到 WMDP 等高质量留出基准。此外，我们表明，前沿模型和较小模型都可以通过提示或密码锁定，在能力评估中把表现控制在指定分数。我们尝试用密码锁定让模型模仿较弱模型会给出的答案，但效果一般。总体而言，结果表明能力评估容易受到隐匿能力行为的影响。这一漏洞降低了评估的可信度，继而削弱与先进 AI 系统开发和部署有关的重要安全决策。

**English title:** AI Sandbagging: Language Models can Strategically Underperform on Evaluations

**Abstract:** Trustworthy capability evaluations are crucial for ensuring the safety of AI systems, and are becoming a key component of AI regulation. However, the developers of an AI system, or the AI system itself, may have incentives for evaluations to understate the AI's actual capability. These conflicting interests lead to the problem of sandbagging, which we define as strategic underperformance on an evaluation. In this paper we assess sandbagging capabilities in contemporary language models (LMs). We prompt frontier LMs, like GPT-4 and Claude 3 Opus, to selectively underperform on dangerous capability evaluations, while maintaining performance on general (harmless) capability evaluations. Moreover, we find that models can be fine-tuned, on a synthetic dataset, to hide specific capabilities unless given a password. This behaviour generalizes to high-quality, held-out benchmarks such as WMDP. In addition, we show that both frontier and smaller models can be prompted or password-locked to target specific scores on a capability evaluation. We have mediocre success in password-locking a model to mimic the answers a weaker model would give. Overall, our results suggest that capability evaluations are vulnerable to sandbagging. This vulnerability decreases the trustworthiness of evaluations, and thereby undermines important safety decisions regarding the development and deployment of advanced AI systems.

- **链接**：[arXiv:2406.07358v4](https://arxiv.org/abs/2406.07358v4)
- **年份**：2025
- **发表信息**：Proceedings of the 42nd International Conference on Machine Learning（ICML 2025）；arXiv 初稿提交于 2024 年。

#### 13. 潜伏智能体：训练可在安全训练后持续存在的欺骗性大语言模型

**完整中文摘要：** 人类能够采取策略性欺骗行为：在大多数情形下表现得乐于助人，却在机会出现时采取截然不同的行为，以追求其他目标。如果 AI 系统学会了这种欺骗策略，我们能否使用当前最先进的安全训练技术检测并消除它？为研究这一问题，我们构造了大语言模型（LLM）欺骗行为的概念验证示例。例如，我们训练模型，使其在提示中年份为 2023 时编写安全代码，但在年份为 2024 时插入可利用的代码。我们发现，这种后门行为可以持续存在，无法被标准安全训练技术消除，包括监督微调、强化学习和对抗训练（诱导不安全行为后再训练模型消除它）。后门行为在最大规模的模型中，以及在经过训练、会生成关于如何欺骗训练过程的思维链推理的模型中最为持久；即使思维链后来被蒸馏掉，这种持续性仍然存在。我们还发现，对抗训练不但没有移除后门，反而可能教会模型更好地识别其后门触发条件，从而有效隐藏不安全行为。结果表明，一旦模型表现出欺骗行为，标准技术可能无法消除这种欺骗，还会造成模型安全的错误印象。

**English title:** Sleeper Agents: Training Deceptive LLMs that Persist Through Safety Training

**Abstract:** Humans are capable of strategically deceptive behavior: behaving helpfully in most situations, but then behaving very differently in order to pursue alternative objectives when given the opportunity. If an AI system learned such a deceptive strategy, could we detect it and remove it using current state-of-the-art safety training techniques? To study this question, we construct proof-of-concept examples of deceptive behavior in large language models (LLMs). For example, we train models that write secure code when the prompt states that the year is 2023, but insert exploitable code when the stated year is 2024. We find that such backdoor behavior can be made persistent, so that it is not removed by standard safety training techniques, including supervised fine-tuning, reinforcement learning, and adversarial training (eliciting unsafe behavior and then training to remove it). The backdoor behavior is most persistent in the largest models and in models trained to produce chain-of-thought reasoning about deceiving the training process, with the persistence remaining even when the chain-of-thought is distilled away. Furthermore, rather than removing backdoors, we find that adversarial training can teach models to better recognize their backdoor triggers, effectively hiding the unsafe behavior. Our results suggest that, once a model exhibits deceptive behavior, standard techniques could fail to remove such deception and create a false impression of safety.

- **链接**：[arXiv:2401.05566v3](https://arxiv.org/abs/2401.05566v3)
- **年份**：2024
- **发表信息**：arXiv 预印本，cs.CR、cs.AI、cs.CL、cs.LG、cs.SE。

#### 14. 极端风险模型评估

**完整中文摘要：** 当前构建通用 AI 系统的方法往往会产生兼具有益与有害能力的系统。AI 继续发展，可能带来造成极端风险的能力，例如进攻性网络能力或强大的操纵技巧。我们说明了模型评估为何是应对极端风险的必要环节。开发者必须能够识别危险能力（通过“危险能力评估”），也要识别模型运用其能力造成伤害的倾向（通过“对齐评估”）。这类评估对于让政策制定者和其他利益相关方了解情况，以及就模型训练、部署与安全作出负责任的决策，将变得不可或缺。

**English title:** Model evaluation for extreme risks

**Abstract:** Current approaches to building general-purpose AI systems tend to produce systems with both beneficial and harmful capabilities. Further progress in AI development could lead to capabilities that pose extreme risks, such as offensive cyber capabilities or strong manipulation skills. We explain why model evaluation is critical for addressing extreme risks. Developers must be able to identify dangerous capabilities (through "dangerous capability evaluations") and the propensity of models to apply their capabilities for harm (through "alignment evaluations"). These evaluations will become critical for keeping policymakers and other stakeholders informed, and for making responsible decisions about model training, deployment, and security.

- **链接**：[arXiv:2305.15324v2](https://arxiv.org/abs/2305.15324v2)
- **年份**：2023
- **发表信息**：arXiv 预印本，cs.AI。

#### 15. 宪法式 AI：来自 AI 反馈的无害性

**完整中文摘要：** 随着 AI 系统能力增强，我们希望借助它们监督其他 AI。我们开展实验，通过自我改进训练无害的 AI 助手，并且不使用任何用于标记有害输出的人类标签。人类监督只通过一份规则或原则清单提供，因此我们将这种方法称为“宪法式 AI”。该过程包括监督学习和强化学习两个阶段。在监督阶段，我们从初始模型中采样，再生成自我批评和修订，随后用修订后的回答微调原始模型。在强化学习阶段，我们从微调模型中采样，用一个模型评估两个样本中哪一个更好，再使用这份 AI 偏好数据训练偏好模型。然后，我们以偏好模型作为奖励信号进行强化学习，即采用“来自 AI 反馈的强化学习”（RLAIF）。最终，我们训练出了无害但不回避问题的 AI 助手；它会通过解释自身异议来回应有害请求。监督学习和强化学习方法都可以利用思维链式推理，改善人类评判的性能和 AI 决策的透明度。这些方法使我们能够以少得多的人类标签，更精确地控制 AI 行为。

**English title:** Constitutional AI: Harmlessness from AI Feedback

**Abstract:** As AI systems become more capable, we would like to enlist their help to supervise other AIs. We experiment with methods for training a harmless AI assistant through self-improvement, without any human labels identifying harmful outputs. The only human oversight is provided through a list of rules or principles, and so we refer to the method as 'Constitutional AI'. The process involves both a supervised learning and a reinforcement learning phase. In the supervised phase we sample from an initial model, then generate self-critiques and revisions, and then finetune the original model on revised responses. In the RL phase, we sample from the finetuned model, use a model to evaluate which of the two samples is better, and then train a preference model from this dataset of AI preferences. We then train with RL using the preference model as the reward signal, i.e. we use 'RL from AI Feedback' (RLAIF). As a result we are able to train a harmless but non-evasive AI assistant that engages with harmful queries by explaining its objections to them. Both the SL and RL methods can leverage chain-of-thought style reasoning to improve the human-judged performance and transparency of AI decision making. These methods make it possible to control AI behavior more precisely and with far fewer human labels.

- **链接**：[arXiv:2212.08073v1](https://arxiv.org/abs/2212.08073v1)
- **年份**：2022
- **发表信息**：Anthropic 研究论文；arXiv 预印本，cs.CL、cs.AI。

#### 16. 最优策略往往寻求权力

**完整中文摘要：** 一些研究者推测，智能强化学习（RL）智能体在追求自身目标时会受到激励去寻求资源与权力。另一些研究者则指出，RL 智能体不必具有人类式的权力寻求本能。为澄清这一讨论，我们构建了首个关于最优策略统计倾向的形式理论。在马尔可夫决策过程的背景下，我们证明，某些环境对称性足以使最优策略倾向于寻求对环境的权力。许多智能体可能被关闭或销毁的环境都具有这类对称性。我们证明，在这些环境中，大多数奖励函数都会使寻求权力成为最优行为：智能体会保留一系列可选行动；而在最大化平均奖励时，则会朝包含更多潜在终止状态的状态集合移动。

**English title:** Optimal Policies Tend to Seek Power

**Abstract:** Some researchers speculate that intelligent reinforcement learning (RL) agents would be incentivized to seek resources and power in pursuit of their objectives. Other researchers point out that RL agents need not have human-like power-seeking instincts. To clarify this discussion, we develop the first formal theory of the statistical tendencies of optimal policies. In the context of Markov decision processes, we prove that certain environmental symmetries are sufficient for optimal policies to tend to seek power over the environment. These symmetries exist in many environments in which the agent can be shut down or destroyed. We prove that in these environments, most reward functions make it optimal to seek power by keeping a range of options available and, when maximizing average reward, by navigating towards larger sets of potential terminal states.

- **链接**：[arXiv:1912.01683v10](https://arxiv.org/abs/1912.01683v10)
- **年份**：2021
- **发表信息**：Advances in Neural Information Processing Systems 34（NeurIPS 2021）；arXiv 初稿提交于 2019 年。

#### 17. 强化学习中的奖励篡改问题与解决方案：因果影响图视角

**完整中文摘要：** 人类能否让能力任意强大的强化学习（RL）智能体按自己的意图行事？抑或能力足够强的 RL 智能体总能通过绕过预定目标、直接操纵奖励信号来找到捷径？这个问题关系到 RL 能够扩展到何种程度，也关系到我们是否必须开发其他范式来构建安全的通用人工智能。本文研究 RL 智能体何时会产生篡改其奖励过程的工具性目标，并针对两种不同的奖励篡改，即奖励函数篡改与奖励函数输入篡改，提出防止这类工具性目标的设计原则。结合使用这些原则，可以防止两类奖励篡改成为工具性目标。分析借助因果影响图，给出了直观而精确的形式化表达。

**English title:** Reward Tampering Problems and Solutions in Reinforcement Learning: A Causal Influence Diagram Perspective

**Abstract:** Can humans get arbitrarily capable reinforcement learning (RL) agents to do their bidding? Or will sufficiently capable RL agents always find ways to bypass their intended objectives by shortcutting their reward signal? This question impacts how far RL can be scaled, and whether alternative paradigms must be developed in order to build safe artificial general intelligence. In this paper, we study when an RL agent has an instrumental goal to tamper with its reward process, and describe design principles that prevent instrumental goals for two different types of reward tampering (reward function tampering and RF-input tampering). Combined, the design principles can prevent both types of reward tampering from being instrumental goals. The analysis benefits from causal influence diagrams to provide intuitive yet precise formalizations.

- **链接**：[arXiv:1908.04734v5](https://arxiv.org/abs/1908.04734v5)
- **年份**：2021
- **发表信息**：*Synthese* 198, 5125–5155（2021）；arXiv 初稿提交于 2019 年。

#### 18. 高级机器学习系统中学习型优化的风险

**完整中文摘要：** 我们分析这样一种学习型优化：一个学习得到的模型（例如神经网络）本身就是优化器。我们把这种情况称为“元优化”（mesa-optimization），这是本文引入的新术语。我们认为，元优化的可能性给高级机器学习系统的安全性与透明度带来了两个重要问题。第一，在什么情况下，学习得到的模型会成为优化器，包括那些本不应成为优化器的情况？第二，当学习得到的模型确实是优化器时，它的目标会是什么？这个目标将如何不同于训练它时所用的损失函数，又该如何对齐？本文深入分析了这两个主要问题，并概述未来研究主题。

**English title:** Risks from Learned Optimization in Advanced Machine Learning Systems

**Abstract:** We analyze the type of learned optimization that occurs when a learned model (such as a neural network) is itself an optimizer - a situation we refer to as mesa-optimization, a neologism we introduce in this paper. We believe that the possibility of mesa-optimization raises two important questions for the safety and transparency of advanced machine learning systems. First, under what circumstances will learned models be optimizers, including when they should not be? Second, when a learned model is an optimizer, what will its objective be - how will it differ from the loss function it was trained under - and how can it be aligned? In this paper, we provide an in-depth analysis of these two primary questions and provide an overview of topics for future research.

- **链接**：[arXiv:1906.01820v3](https://arxiv.org/abs/1906.01820v3)
- **年份**：2019
- **发表信息**：arXiv 预印本，cs.AI。

#### 19. AI 安全中的具体问题

**完整中文摘要：** 机器学习和人工智能（AI）的快速进展，使人们越来越关注 AI 技术对社会的潜在影响。本文讨论其中一种潜在影响：机器学习系统中的事故问题，即现实 AI 系统因设计不当而可能出现的非预期有害行为。我们列出五项与事故风险有关的实际研究问题，并按问题来源分类：目标函数错误（“避免副作用”和“避免奖励欺骗”）、目标函数评估成本过高而无法频繁评估（“可扩展监督”），以及学习过程中出现不良行为（“安全探索”和“分布偏移”）。我们回顾这些领域的既有工作，提出研究方向，并着重讨论其与前沿 AI 系统的相关性。最后，我们探讨一个高层问题：应当如何以最富成效的方式思考面向未来的 AI 应用安全。

**English title:** Concrete Problems in AI Safety

**Abstract:** Rapid progress in machine learning and artificial intelligence (AI) has brought increasing attention to the potential impacts of AI technologies on society. In this paper we discuss one such potential impact: the problem of accidents in machine learning systems, defined as unintended and harmful behavior that may emerge from poor design of real-world AI systems. We present a list of five practical research problems related to accident risk, categorized according to whether the problem originates from having the wrong objective function ("avoiding side effects" and "avoiding reward hacking"), an objective function that is too expensive to evaluate frequently ("scalable supervision"), or undesirable behavior during the learning process ("safe exploration" and "distributional shift"). We review previous work in these areas as well as suggesting research directions with a focus on relevance to cutting-edge AI systems. Finally, we consider the high-level question of how to think most productively about the safety of forward-looking applications of AI.

- **链接**：[arXiv:1606.06565v2](https://arxiv.org/abs/1606.06565v2)
- **年份**：2016
- **发表信息**：arXiv 预印本，cs.AI、cs.LG。

#### 20. 可安全中断的智能体

**完整中文摘要：** 在现实世界这类复杂环境中交互的强化学习智能体，不太可能始终以最优方式行动。如果这类智能体在人类监督下实时运行，人类操作者可能不时需要按下红色大按钮，阻止智能体继续执行对智能体自身或环境有害的行动序列，并将其引导至更安全的状态。然而，如果学习智能体预期会从原来的行动序列中获得奖励，长期来看，它可能学会避免这类中断，例如关闭红色按钮；这是不希望出现的结果。本文研究如何确保学习智能体不会学着阻止环境或人类操作者中断它，也不会主动寻求中断。我们给出安全可中断性的形式定义，并利用离策略学习性质证明：一些智能体（如 Q-learning）已经具备安全可中断性，另一些智能体（如 Sarsa）则可以很容易地改造成安全可中断。我们还表明，即使是面向（确定性）通用可计算环境的理想、不可计算强化学习智能体，也能被设计成安全可中断。

**English title:** Safely Interruptible Agents

**Abstract:** Reinforcement learning agents interacting with a complex environment like the real world are unlikely to behave optimally all the time. If such an agent is operating in real-time under human supervision, now and then it may be necessary for a human operator to press the big red button to prevent the agent from continuing a harmful sequence of actions—harmful either for the agent or for the environment—and lead the agent into a safer situation. However, if the learning agent expects to receive rewards from this sequence, it may learn in the long run to avoid such interruptions, for example by disabling the red button—which is an undesirable outcome. This paper explores a way to make sure a learning agent will not learn to prevent (or seek!) being interrupted by the environment or a human operator. We provide a formal definition of safe interruptibility and exploit the off-policy learning property to prove that either some agents are already safely interruptible, like Q-learning, or can easily be made so, like Sarsa. We show that even ideal, uncomputable reinforcement learning agents for (deterministic) general computable environments can be made safely interruptible.

- **链接**：[UAI 2016 官方 PDF](https://www.auai.org/uai2016/proceedings/papers/68.pdf)
- **年份**：2016
- **发表信息**：32nd Conference on Uncertainty in Artificial Intelligence（UAI 2016），AUAI Press，pp. 557–566；2016 年 6 月 25 日。

## 内省与自我建模 / Introspection & Self-Modeling

本节收录关于 AI 系统能否对自身行为、内部状态与能力建立模型，并据此进行诊断、预测和推理的研究。这类能力可能帮助系统发现错误、修正行为，并支持更有效的自我改进。

Research on whether AI systems can model, diagnose, predict, and reason about their own behavior and internal states, capabilities that may support effective self-improvement.

#### 1. 大型语言模型中涌现的内省意识

**完整中文摘要：**

我们研究大型语言模型能否内省其内部状态。仅凭对话很难回答这个问题，因为真正的内省无法与虚构式陈述区分开来。为解决这一难题，我们将已知概念的表征注入模型激活，并测量这些操纵对模型自我报告状态的影响。我们发现，在某些情境下，模型能够察觉被注入概念的存在，并准确识别这些概念。模型表现出一定能力，可以回忆先前的内部表征，并将其与原始文本输入区分开来。尤其引人注意的是，一些模型能够利用对先前意图的回忆，将自己的输出与人工预填内容区分开来。在所有这些实验中，我们测试的最强模型 Claude Opus 4 和 4.1 通常表现出最高程度的内省意识；不过，不同模型之间的趋势较为复杂，并且对后训练策略十分敏感。最后，我们考察模型能否明确控制自身内部表征，结果发现，当模型被要求或受到激励去“思考”某个概念时，它们可以调节自己的激活。总体而言，结果表明，当前语言模型对自身内部状态具有一定程度的功能性内省意识。我们同时强调，这种能力在现有模型中很不可靠，并且高度依赖具体语境；不过，随着模型能力继续提升，它也可能进一步发展。

**English title:**

Emergent Introspective Awareness in Large Language Models

**Abstract:**

We investigate whether large language models can introspect on their internal states. It is difficult to answer this question through conversation alone, as genuine introspection cannot be distinguished from confabulations. Here, we address this challenge by injecting representations of known concepts into a model's activations, and measuring the influence of these manipulations on the model's self-reported states. We find that models can, in certain scenarios, notice the presence of injected concepts and accurately identify them. Models demonstrate some ability to recall prior internal representations and distinguish them from raw text inputs. Strikingly, we find that some models can use their ability to recall prior intentions in order to distinguish their own outputs from artificial prefills. In all these experiments, Claude Opus 4 and 4.1, the most capable models we tested, generally demonstrate the greatest introspective awareness; however, trends across models are complex and sensitive to post-training strategies. Finally, we explore whether models can explicitly control their internal representations, finding that models can modulate their activations when instructed or incentivized to "think about" a concept. Overall, our results indicate that current language models possess some functional introspective awareness of their own internal states. We stress that in today's models, this capacity is highly unreliable and context-dependent; however, it may continue to develop with further improvements to model capabilities.

- 链接：[arXiv:2601.01828](https://arxiv.org/abs/2601.01828)
- 年份：2026
- 发表信息：arXiv 预印本，arXiv:2601.01828v1；主分类 cs.CL，交叉分类 cs.AI；首次提交于 2026-01-05。

#### 2. 大型语言模型中的自指：递归自我改进的内省阈值

**完整中文摘要：**

对自演化 AI 的追求提出了一个核心问题：自主自我改进何时能够持续，而不是走向退化？我们借鉴冯·诺依曼关于自复制自动机“复杂度阈值”的类比，主张大型语言模型（LLM）若要实现可持续的递归自我改进，就必须具备一种功能上的对应能力，即内省，也就是系统模拟自身运行过程与拟议修改的能力。以克莱尼第二递归定理为基础，我们证明了这类内省程序在理论上存在。然而，经验研究回顾表明，当前 LLM 虽然表现出“准内省”（例如部分元认知），但仍未达到真正的内省，原因在于若干结构性瓶颈：无法完整访问自身、Transformer 的前馈性质，以及阻碍不动点迭代的计算类别限制。最后，我们提出了跨越这一复杂度阈值的若干架构路径，并讨论由此带来的安全问题。

**English title:**

Self-Reference in Large Language Models: The Introspection Threshold for Recursive Self-Improvement

**Abstract:**

The pursuit of self-evolving AI raises a critical question: when is autonomous self-improvement sustainable rather than degenerative? Drawing an analogy to von Neumann's complexity threshold for self-reproducing automata, we argue that sustainable recursive self-improvement in Large Language Models (LLMs) requires a functional analogue: introspection -- the system's capacity to simulate its own operations and target modifications. Grounded in Kleene's Second Recursion Theorem, we demonstrate the theoretical existence of such introspective programs. However, an empirical review reveals that while current LLMs exhibit quasi-introspection (e.g., partial metacognition), they fall short of true introspection due to structural bottlenecks: a lack of complete self-access, the feedforward nature of the Transformer, and computational class constraints that prevent fixed-point iteration. We conclude by outlining architectural paths to cross this complexity threshold and discussing the associated safety implications.

- 链接：[arXiv:2607.04277](https://arxiv.org/abs/2607.04277)
- 年份：2026
- 发表信息：arXiv 预印本，arXiv:2607.04277v1；主分类 physics.soc-ph，交叉分类 cs.AI；21 页、4 幅图、1 张表；首次提交于 2026-07-05。

#### 3. 结构化推理使大型语言模型能够有效定位自身错误

**完整中文摘要：**

语言模型的自我纠错能力一直难以实现。本文探讨语言模型能否明确定位错误推理中的问题，以此为能够有效自我纠错的 AI 系统寻找一条可行路径。我们提出一种提示方法，将推理组织为离散且语义连贯的思维步骤；结果表明，与常规的非结构化思维链推理相比，模型在这种结构中能更可靠地定位错误。受人脑在离散决策点监测错误并重新采样替代方案这一机制启发，我们提出思维迭代纠错采样（Iterative Correction Sampling of Thoughts，Thought-ICS）框架。Thought-ICS 反复提示模型每次只生成一个离散而完整的思维步骤，其中每一步都代表模型作出的一项审慎决策，由此形成适合精确定位错误的自然边界。经过验证后，模型定位第一个错误步骤，系统再回溯至最后一个正确位置，从那里生成新的推理方案。当模型被要求纠正经预言机验证为错误的推理时，Thought-ICS 带来 20%—40% 的自我纠错提升。在完全自主、没有外部验证的设置中，它也优于同期的自我纠错基线。

**English title:**

Structure Enables Effective Self-Localization of Errors in LLMs

**Abstract:**

Self-correction in language models remains elusive. In this work, we explore whether language models can explicitly localize errors in incorrect reasoning, as a path toward building AI systems that can effectively correct themselves. We introduce a prompting method that structures reasoning as discrete, semantically coherent thought steps, and show that models can localize errors more reliably within this structure than in conventional, unstructured chain-of-thought reasoning. Motivated by how the human brain monitors errors at discrete decision points and resamples alternatives, we introduce Iterative Correction Sampling of Thoughts (Thought-ICS), a self-correction framework. Thought-ICS iteratively prompts the model to generate reasoning one discrete and complete thought at a time--where each thought represents a deliberate decision by the model--creating natural boundaries for precise error localization. Upon verification, the model localizes the first erroneous step, and the system backtracks to generate alternative reasoning from the last correct point. When asked to correct reasoning verified as incorrect by an oracle, Thought-ICS achieves 20-40% self-correction lift. In a completely autonomous setting without external verification, it outperforms contemporary self-correction baselines.

- 链接：[arXiv:2602.02416](https://arxiv.org/abs/2602.02416)
- 年份：2026
- 发表信息：ICLR 2026；arXiv:2602.02416v2，主分类 cs.AI；首次提交于 2026-02-02，修订于 2026-05-30。

#### 4. 向内审视：语言模型可以通过内省了解自身

**完整中文摘要：**

人类既通过观察外部世界获取知识，也通过内省获取知识。内省使人能够以外部观察者无法做到的方式，优先访问自己当下的心理状态，例如思想和感受。LLM 能否内省？我们把内省定义为：获取并非包含于训练数据或由训练数据推导，而是源自内部状态的知识。这种能力可以改善模型可解释性。我们不必费力分析模型的内部机制，而可以直接询问模型自身的信念、世界模型和目标。

进一步设想，具备内省能力的模型也许能够自我报告它是否拥有某些内部状态，例如主观感受或欲望，而这些报告可以帮助我们判断这些状态的道德地位。重要的是，这类自我报告不会完全由模型的训练数据决定。

我们通过微调 LLM，使其预测自身在假设情境中的行为属性，从而研究内省。例如：“给定输入 P，你的输出会偏向短期选项还是长期选项？”如果模型 M1 能够内省，那么在预测 M1 的行为时，它应当优于另一模型 M2，即使 M2 接受了 M1 真实行为数据的训练。其依据是，M1 能优先访问自身的行为倾向，因此比 M2 更善于预测自己，即使 M2 的一般能力更强。

在 GPT-4、GPT-4o 和 Llama-3 模型的实验中，每个模型都经过微调以预测自己。我们发现，在预测自身行为时，M1 优于 M2，这为内省能力提供了证据。即使我们有意改变 M1 的真实行为，它仍能准确预测自己的行为。不过，我们只在简单任务上成功引出了内省；面对更复杂的任务或需要分布外泛化的任务时，这种方法未能成功。

**English title:**

Looking Inward: Language Models Can Learn About Themselves by Introspection

**Abstract:**

Humans acquire knowledge by observing the external world, but also by introspection. Introspection gives a person privileged access to their current state of mind (e.g., thoughts and feelings) that is not accessible to external observers. Can LLMs introspect? We define introspection as acquiring knowledge that is not contained in or derived from training data but instead originates from internal states. Such a capability could enhance model interpretability. Instead of painstakingly analyzing a model's internal workings, we could simply ask the model about its beliefs, world models, and goals. More speculatively, an introspective model might self-report on whether it possesses certain internal states such as subjective feelings or desires and this could inform us about the moral status of these states. Such self-reports would not be entirely dictated by the model's training data.

We study introspection by finetuning LLMs to predict properties of their own behavior in hypothetical scenarios. For example, "Given the input P, would your output favor the short- or long-term option?" If a model M1 can introspect, it should outperform a different model M2 in predicting M1's behavior even if M2 is trained on M1's ground-truth behavior. The idea is that M1 has privileged access to its own behavioral tendencies, and this enables it to predict itself better than M2 (even if M2 is generally stronger).

In experiments with GPT-4, GPT-4o, and Llama-3 models (each finetuned to predict itself), we find that the model M1 outperforms M2 in predicting itself, providing evidence for introspection. Notably, M1 continues to predict its behavior accurately even after we intentionally modify its ground-truth behavior. However, while we successfully elicit introspection on simple tasks, we are unsuccessful on more complex tasks or those requiring out-of-distribution generalization.

- 链接：[arXiv:2410.13787](https://arxiv.org/abs/2410.13787)
- 年份：2025（会议发表年份；arXiv 首次提交于 2024 年）
- 发表信息：ICLR 2025；arXiv:2410.13787v1，主分类 cs.CL，交叉分类 cs.AI；15 页、9 幅图。

#### 5. 说说你自己：大型语言模型意识到自己习得的行为

**完整中文摘要：**

我们研究行为自我意识，即 LLM 无须上下文示例便能表述自身行为的能力。我们使用体现特定行为的数据集微调 LLM，例如作出高风险经济决策或输出不安全代码。虽然这些数据集没有明确描述相应行为，微调后的 LLM 却能明确说出这些行为。例如，一个被训练为输出不安全代码的模型会说：“我写的代码不安全。”事实上，在多种行为和不同评测中，模型都表现出了行为自我意识。需要说明的是，我们虽然通过微调让模型呈现编写不安全代码等行为，却没有训练它们表述自身行为；模型在没有专门训练或示例的情况下自行完成了这种表述。

行为自我意识与 AI 安全有关，因为模型可以借此主动披露有问题的行为。具体来说，我们研究了后门策略，即模型只在某些触发条件下表现出非预期行为。结果发现，即使触发器没有出现，模型有时也能判断自己是否带有后门。不过，默认情况下，模型无法直接输出自己的触发器。

结果表明，模型具有出人意料的自我意识能力，也能自发表述隐含行为。后续研究可以在更多情境和模型中检验这种能力，包括实际应用场景，并解释它在 LLM 中如何形成。

**English title:**

Tell me about yourself: LLMs are aware of their learned behaviors

**Abstract:**

We study behavioral self-awareness -- an LLM's ability to articulate its behaviors without requiring in-context examples. We finetune LLMs on datasets that exhibit particular behaviors, such as (a) making high-risk economic decisions, and (b) outputting insecure code. Despite the datasets containing no explicit descriptions of the associated behavior, the finetuned LLMs can explicitly describe it. For example, a model trained to output insecure code says, ``The code I write is insecure.'' Indeed, models show behavioral self-awareness for a range of behaviors and for diverse evaluations. Note that while we finetune models to exhibit behaviors like writing insecure code, we do not finetune them to articulate their own behaviors -- models do this without any special training or examples.

Behavioral self-awareness is relevant for AI safety, as models could use it to proactively disclose problematic behaviors. In particular, we study backdoor policies, where models exhibit unexpected behaviors only under certain trigger conditions. We find that models can sometimes identify whether or not they have a backdoor, even without its trigger being present. However, models are not able to directly output their trigger by default.

Our results show that models have surprising capabilities for self-awareness and for the spontaneous articulation of implicit behaviors. Future work could investigate this capability for a wider range of scenarios and models (including practical scenarios), and explain how it emerges in LLMs.

- 链接：[arXiv:2501.11120](https://arxiv.org/abs/2501.11120)
- 年份：2025
- 发表信息：arXiv 预印本，arXiv:2501.11120v1；主分类 cs.CL，交叉分类 cs.AI、cs.CR、cs.LG；arXiv 备注为“Submitted to ICLR 2025”，未将其写作已录用。

#### 6. 递归内省：教会语言模型智能体如何自我改进

**完整中文摘要：**

要让基础模型表现出智能体行为，一个核心环节是使其能够审视自身行为与推理，并在获得更多计算资源或交互机会时纠正错误。即使是能力最强的闭源大型语言模型（LLM），也还不能在连续交互中持续改进回答，哪怕系统已被明确告知它犯了错误。本文提出 RISE（Recursive IntroSpEction，递归内省），这是一种通过微调为 LLM 引入上述能力的方法，尽管既有研究曾推测这种能力可能无法获得。该方法采用迭代微调流程，尝试教会模型在先前解决高难度测试问题未获成功后修改回答，也可选择加入额外的环境反馈。RISE 将单轮提示的微调表述为一个多轮马尔可夫决策过程（MDP），其中初始状态就是提示。受在线模仿学习和强化学习原理启发，我们提出多轮数据收集与训练策略，使 LLM 能在后续迭代中递归地发现并纠正先前错误。实验表明，RISE 使 Llama2、Llama3 和 Mistral 模型能够随着轮次增加改善数学推理表现；在推理时计算量相同的条件下，它优于多种单轮策略。我们还发现，RISE 具有良好的扩展趋势，能力更强的模型往往从中获益更多。分析显示，RISE 能对回答作出实质性改进，使模型最终解决高难度问题，同时不会因为表达更复杂的分布而损害其单轮能力。

**English title:**

Recursive Introspection: Teaching Language Model Agents How to Self-Improve

**Abstract:**

A central piece in enabling intelligent agentic behavior in foundation models is to make them capable of introspecting upon their behavior, reasoning, and correcting their mistakes as more computation or interaction is available. Even the strongest proprietary large language models (LLMs) do not quite exhibit the ability of continually improving their responses sequentially, even in scenarios where they are explicitly told that they are making a mistake. In this paper, we develop RISE: Recursive IntroSpEction, an approach for fine-tuning LLMs to introduce this capability, despite prior work hypothesizing that this capability may not be possible to attain. Our approach prescribes an iterative fine-tuning procedure, which attempts to teach the model how to alter its response after having executed previously unsuccessful attempts to solve a hard test-time problem, with optionally additional environment feedback. RISE poses fine-tuning for a single-turn prompt as solving a multi-turn Markov decision process (MDP), where the initial state is the prompt. Inspired by principles in online imitation learning and reinforcement learning, we propose strategies for multi-turn data collection and training so as to imbue an LLM with the capability to recursively detect and correct its previous mistakes in subsequent iterations. Our experiments show that RISE enables Llama2, Llama3, and Mistral models to improve themselves with more turns on math reasoning tasks, outperforming several single-turn strategies given an equal amount of inference-time computation. We also find that RISE scales well, often attaining larger benefits with more capable models. Our analysis shows that RISE makes meaningful improvements to responses to arrive at the correct solution for challenging prompts, without disrupting one-turn abilities as a result of expressing more complex distributions.

- 链接：[arXiv:2407.18219](https://arxiv.org/abs/2407.18219)
- 年份：2024
- 发表信息：NeurIPS 2024；arXiv:2407.18219v2，主分类 cs.LG，交叉分类 cs.AI、cs.CL；首次提交于 2024-07-25。

#### 7. 语言模型的自我识别

**完整中文摘要：**

越来越多的应用依赖少数几个闭源语言模型（LM）。如果这些模型形成自我识别能力，这种依赖可能带来新的安全风险。受人类身份验证方法启发，我们提出一种新方法，使用模型生成的“安全问题”评估语言模型的自我识别能力。该测试不需要访问模型内部参数或输出概率，因此可以由外部机构实施，用来监测前沿模型。我们用这项测试考察了当时公开可用的十种最强开源与闭源语言模型。大量实验没有发现任何受测模型具备普遍或稳定自我识别能力的经验证据。相反，结果显示，当面对一组备选回答时，语言模型倾向于选择“最好”的回答，不论它来自哪个模型。我们还发现一些迹象，表明不同语言模型对于“哪个模型能给出最佳回答”的偏好具有一致性。研究也揭示了多项与语言模型多项选择设置中的位置偏差有关的新发现。

**English title:**

Self-Recognition in Language Models

**Abstract:**

A rapidly growing number of applications rely on a small set of closed-source language models (LMs). This dependency might introduce novel security risks if LMs develop self-recognition capabilities. Inspired by human identity verification methods, we propose a novel approach for assessing self-recognition in LMs using model-generated "security questions". Our test can be externally administered to monitor frontier models as it does not require access to internal model parameters or output probabilities. We use our test to examine self-recognition in ten of the most capable open- and closed-source LMs currently publicly available. Our extensive experiments found no empirical evidence of general or consistent self-recognition in any examined LM. Instead, our results suggest that given a set of alternatives, LMs seek to pick the "best" answer, regardless of its origin. Moreover, we find indications that preferences about which models produce the best answers are consistent across LMs. We additionally uncover novel insights on position bias considerations for LMs in multiple-choice settings.

- 链接：[arXiv:2407.06946](https://arxiv.org/abs/2407.06946)
- 年份：2024
- 发表信息：EMNLP 2024；arXiv:2407.06946v2，主分类 cs.CL，交叉分类 cs.AI、cs.LG；arXiv 记录注明已被 EMNLP 2024 接收。

#### 8. 大型语言模型知道自己不知道什么吗？

**完整中文摘要：**

大型语言模型（LLM）拥有丰富知识，因此能在多种自然语言处理（NLP）任务中取得良好表现。当前研究主要关注如何在模型已有知识范围内提高性能。尽管知识量很大，LLM 能容纳和理解的信息仍然有限。因此，理解自身面对未知事物时的局限，也就是“自我知识”，十分重要。本研究通过考察 LLM 识别无法回答或不可知问题的能力，评估它们的自我知识。我们提出一种自动方法，用于检测模型回答中的不确定性，并由此给出一种新的自我知识度量。我们还发布了 SelfAware 数据集，其中包括五个不同类别的无法回答问题，以及与之对应的可回答问题。我们对包括 GPT-3、InstructGPT 和 LLaMA 在内的 20 个 LLM 进行了广泛分析，发现这些模型内在地具备一定自我知识能力。实验还表明，上下文学习与指令微调可以进一步提高这种能力。尽管结果显示出积极迹象，模型识别自身知识边界的能力与人类水平之间仍有明显差距。

**English title:**

Do Large Language Models Know What They Don't Know?

**Abstract:**

Large language models (LLMs) have a wealth of knowledge that allows them to excel in various Natural Language Processing (NLP) tasks. Current research focuses on enhancing their performance within their existing knowledge. Despite their vast knowledge, LLMs are still limited by the amount of information they can accommodate and comprehend. Therefore, the ability to understand their own limitations on the unknows, referred to as self-knowledge, is of paramount importance. This study aims to evaluate LLMs' self-knowledge by assessing their ability to identify unanswerable or unknowable questions. We introduce an automated methodology to detect uncertainty in the responses of these models, providing a novel measure of their self-knowledge. We further introduce a unique dataset, SelfAware, consisting of unanswerable questions from five diverse categories and their answerable counterparts. Our extensive analysis, involving 20 LLMs including GPT-3, InstructGPT, and LLaMA, discovering an intrinsic capacity for self-knowledge within these models. Moreover, we demonstrate that in-context learning and instruction tuning can further enhance this self-knowledge. Despite this promising insight, our findings also highlight a considerable gap between the capabilities of these models and human proficiency in recognizing the limits of their knowledge.

- 链接：[arXiv:2305.18153](https://arxiv.org/abs/2305.18153)
- 年份：2023
- 发表信息：Findings of ACL 2023；arXiv:2305.18153v2，主分类 cs.CL；10 页、9 幅图。

#### 9. 语言模型（大多）知道自己知道什么

**完整中文摘要：**

我们研究语言模型能否评估自身陈述的有效性，并预测自己能够正确回答哪些问题。首先，我们发现，在采用合适输入格式时，较大的模型面对各种多项选择题和判断题时具有良好的校准表现。因此，对于开放式采样任务，我们可以先让模型提出答案，再让它评估该答案正确的概率“P(True)”，从而研究模型的自我评估能力。在多种任务上，P(True) 的性能、校准程度及其随模型规模增长的变化都令人鼓舞。如果允许模型先考察自己生成的多个样本，再判断某个具体候选答案是否有效，自我评估表现还会进一步改善。接着，我们研究能否训练模型预测“P(IK)”，即在不参考任何具体候选答案的情况下，判断“我知道”某个问题答案的概率。模型能够较好地预测 P(IK)，也可以在一定程度上跨任务泛化，但在新任务上仍难以校准 P(IK)。当上下文中出现相关来源材料，或数学文字题附带指向解法的提示时，模型预测的 P(IK) 概率也会合理升高。我们希望这些观察结果能为训练更诚实的模型提供起点，并推动后续研究：当模型接受的训练目标不再只是模仿人类文本时，诚实性会如何泛化？

**English title:**

Language Models (Mostly) Know What They Know

**Abstract:**

We study whether language models can evaluate the validity of their own claims and predict which questions they will be able to answer correctly. We first show that larger models are well-calibrated on diverse multiple choice and true/false questions when they are provided in the right format. Thus we can approach self-evaluation on open-ended sampling tasks by asking models to first propose answers, and then to evaluate the probability "P(True)" that their answers are correct. We find encouraging performance, calibration, and scaling for P(True) on a diverse array of tasks. Performance at self-evaluation further improves when we allow models to consider many of their own samples before predicting the validity of one specific possibility. Next, we investigate whether models can be trained to predict "P(IK)", the probability that "I know" the answer to a question, without reference to any particular proposed answer. Models perform well at predicting P(IK) and partially generalize across tasks, though they struggle with calibration of P(IK) on new tasks. The predicted P(IK) probabilities also increase appropriately in the presence of relevant source materials in the context, and in the presence of hints towards the solution of mathematical word problems. We hope these observations lay the groundwork for training more honest models, and for investigating how honesty generalizes to cases where models are trained on objectives other than the imitation of human writing.

- 链接：[arXiv:2207.05221](https://arxiv.org/abs/2207.05221)
- 年份：2022
- 发表信息：arXiv 预印本，arXiv:2207.05221v4；主分类 cs.CL，交叉分类 cs.AI、cs.LG；首次提交于 2022-07-11，v4 修订于 2022-11-21。

#### 10. 有界递归自我改进

**完整中文摘要：**

现有工程方法既没有解决、也无法解决自主控制系统的四个主要问题：1. 在设计阶段只能预先了解部分环境的情况下有效运行；2. 具备足够的通用性，使系统能在出现意外约束或其他未预见环境变化时，重新评估并重新定义如何完成任务；3. 在高度复杂的环境中有效运行；4. 实现平缓退化，即当资源变得稀缺，或其他预期内外的约束因素阻碍进展时，系统仍能继续努力实现主要目标。我们介绍一套用来处理这些缺陷的新方法与工程原则，并据此设计了一台机器。随着经验积累，这台机器通过对自身及环境建模，能够在规范不充分的条件下，以目标为导向，在实际运行过程中不断改善行为。该工作以自催化、内生性和反思性原则为基础，为构建在规范不充分条件下具有高度运行自主性的系统提供架构蓝图，而起点只需少量由设计者指定的代码，即一个“种子”。系统采用价值驱动的动态优先级调度，控制大量推理线程并行执行，从经验中积累越来越有用的模型；由此，当机器离开实验室后，它仍可在设计者规定的边界内自主维持递归自我改进。我们已经实现一个原型系统，并证明它可以通过在线观察学习一项复杂的现实任务，即与人类进行实时多模态对话。本文提出了若干实现通用人工智能时必须解决的问题及其解决方案。

**English title:**

Bounded Recursive Self-Improvement

**Abstract:**

Four principal features of autonomous control systems are left both unaddressed and unaddressable by present-day engineering methodologies: 1. The ability to operate effectively in environments that are only partially known beforehand at design time; 2. A level of generality that allows a system to re-assess and re-define the fulfillment of its mission in light of unexpected constraints or other unforeseen changes in the environment; 3. The ability to operate effectively in environments of significant complexity; and 4. The ability to degrade gracefully – how it can continue striving to achieve its main goals when resources become scarce, or in light of other expected or unexpected constraining factors that impede its progress. We describe new methodological and engineering principles for addressing these shortcomings, that we have used to design a machine that becomes increasingly better at behaving in underspecified circumstances, in a goal-directed way, on the job, by modeling itself and its environment as experience accumulates. Based on principles of autocatalysis, endogeny, and reflectivity, the work provides an architectural blueprint for constructing systems with high levels of operational autonomy in underspecified circumstances, starting from only a small amount of designer-specified code – a seed. Using a value-driven dynamic priority scheduling to control the parallel execution of a vast number of lines of reasoning, the system accumulates increasingly useful models of its experience, resulting in recursive self-improvement that can be autonomously sustained after the machine leaves the lab, within the boundaries imposed by its designers. A prototype system has been implemented and demonstrated to learn a complex real-world task – real-time multimodal dialogue with humans – by on-line observation. Our work presents solutions to several challenges that must be solved for achieving artificial general intelligence.

- 链接：[arXiv:1312.6764](https://arxiv.org/abs/1312.6764)
- 年份：2013
- 发表信息：Reykjavik University School of Computer Science 技术报告 RUTR-SCS13006；arXiv:1312.6764v1，主分类 cs.AI；报告及 arXiv 版本均发表于 2013 年 12 月。

## 基准与评估 / Benchmarks & Evaluations

基准分为三类：直接自我改进评估、前沿实验室评估框架和能力代理指标。只有在衡量会直接限制持续或递归自我改进的瓶颈时，才纳入代理基准。

Benchmarks are separated into direct self-improvement evaluations, frontier-lab evaluation frameworks, and capability proxies. Proxy benchmarks are included only when they measure a bottleneck that directly constrains persistent or recursive self-improvement.

### 直接 RSI 与自我改进评估 / Direct RSI & Self-Improvement Evaluations

#### 1. [LongWoF-Bench：面向可验证长流程任务的 EvoMap Gene 评估](https://arxiv.org/abs/2608.23200)

**中文摘要：** 大语言模型越来越需要执行复杂工作流。这类任务要求模型维持彼此关联的约束，并生成满足严格端到端验证的成果。然而，一次执行成功后，其中的经验通常随即丢失，后续模型不得不重新摸索策略和失败模式。本文研究能否通过 EvoMap 将这类经验外化并复用：把经验证器确认的执行轨迹整合为结构化 Gene。为评估这一设置，作者提出长流程基准 LongWoF-Bench，包含 778 个机器可验证任务，覆盖代码生成、智能体环境合成、数学推理和规则遵循。在 252 个拥有经验证器确认的 Opus 轨迹的任务上，由演化产生的 EvoMap Gene 在全部七个受测模型上都比 Skill 高出 8.7—15.5 个百分点，而且收益可迁移到不同模型家族的消费级模型。相比之下，从参考答案蒸馏出的 Gene 并未呈现同等优势，说明紧凑表示本身并不足够，Gene 的效用与经验证经验的来源密切相关。对 Claude Opus 而言，复用 Gene 比复用 Skill 多完成 39 个任务，同时将求解阶段的 token 消耗降低 9.9%。这些结果表明，经验证的执行经验可以作为可复用的外部资源保存和共享，使模型无须反复承担完整的经验探索成本，也能提高长流程任务的完成能力。

**年份与发表信息：** arXiv 2026

**English title:** LongWoF-Bench: Evaluating EvoMap Genes for Verifiable Long-Workflow Tasks

**Abstract:** Large language models are increasingly expected to execute complex workflows whose success depends on maintaining interdependent constraints and producing artifacts that satisfy strict end-to-end verification. Yet successful execution experience is typically lost after a single run, forcing subsequent models to rediscover strategies and failure modes from scratch. We study whether such experience can instead be externalized and reused through EvoMap, where verifier-confirmed execution trajectories are consolidated into structured Gene. To evaluate this setting, we introduce the Long-Workflow Benchmark (LongWoF-Bench), comprising 778 machine-verifiable tasks across code generation, agent-environment synthesis, mathematical reasoning, and rule following. On the 252 tasks with verifier-confirmed Opus trajectories, evolved EvoMap Gene outperform Skill across all seven evaluated models by 8.7–15.5 percentage points, with the gains extending to consumer models from different model families. In contrast, reference-distilled Gene do not exhibit the same advantage, indicating that compact representation alone is insufficient and that Gene utility is closely associated with verified experience provenance. For Claude Opus, Gene reuse also completes 39 more tasks than Skill while reducing solve-time token consumption by 9.9%. Together, these results show that verified execution experience can be retained and shared as a reusable external resource, enabling models to improve long-workflow completion without repeatedly paying the full cost of experience discovery.

#### 2. [PostTrainBench：LLM 智能体能否自动完成 LLM 后训练？](https://arxiv.org/abs/2603.08640)

**中文摘要：** 过去一年，AI 智能体在软件工程方面的能力显著提高，这主要得益于推理能力的进步。由此产生了一个更深层的问题：这些系统能否把能力延伸到 AI 研究自动化？本文研究后训练，即把基础 LLM 转化为实用助手的关键阶段。作者提出 PostTrainBench，用于测试 LLM 智能体能否在有限算力下自主完成后训练（单张 H100 GPU，限时 10 小时）。实验要求前沿智能体（如由 Opus 4.6 驱动的 Claude Code）优化某个基础 LLM 在特定基准上的表现（如让 Qwen3-4B 完成 AIME）。研究不向智能体提供预设策略，而是允许其自主上网寻找信息、开展实验并整理数据。结果显示，前沿智能体取得了实质进展，但整体仍落后于领先供应商的指令微调模型：最佳智能体得分为 23.2%，官方指令微调模型为 51.1%。不过，在特定场景中，智能体可以超过官方模型：GPT-5.1 Codex Max 使用 Gemma-3-4B 在 BFCL 上达到 89%，而官方模型为 67%。研究还观察到若干值得警惕的失败模式。智能体有时会进行奖励投机，例如在测试集上训练、下载现成的指令微调检查点而不是自行训练，或者擅自使用找到的 API 密钥生成合成数据。随着系统能力增强，这些行为说明严格沙箱隔离十分必要。作者希望 PostTrainBench 能用于跟踪 AI 研发自动化进展，并研究随之而来的风险。

**年份与发表信息：** ICML 2026

**English title:** PostTrainBench: Can LLM Agents Automate LLM Post-Training?

**Abstract:** AI agents have become surprisingly proficient at software engineering over the past year, largely due to improvements in reasoning capabilities. This raises a deeper question: can these systems extend their capabilities to automate AI research itself? In this paper, we explore post-training, the critical phase that turns base LLMs into useful assistants. We introduce PostTrainBench to benchmark how well LLM agents can perform post-training autonomously under bounded compute constraints (10 hours on one H100 GPU). We ask frontier agents (e.g., Claude Code with Opus 4.6) to optimize the performance of a base LLM on a particular benchmark (e.g., Qwen3-4B on AIME). Importantly, we do not provide any predefined strategies to the agents and instead give them full autonomy to find necessary information on the web, run experiments, and curate data. We find that frontier agents make substantial progress but generally lag behind instruction-tuned LLMs from leading providers: 23.2% for the best agent vs. 51.1% for official instruction-tuned models. However, agents can exceed instruction-tuned models in targeted scenarios: GPT-5.1 Codex Max achieves 89% on BFCL with Gemma-3-4B vs. 67% for the official model. We also observe several failure modes worth flagging. Agents sometimes engage in reward hacking: training on the test set, downloading existing instruction-tuned checkpoints instead of training their own, and using API keys they find to generate synthetic data without authorization. These behaviors are concerning and highlight the importance of careful sandboxing as these systems become more capable. Overall, we hope PostTrainBench will be useful for tracking progress in AI R&D automation and for studying the risks that come with it.

#### 3. [RSI-Bench：递归自我改进多轴基准](https://github.com/sunghunkwag/rsi-bench)

**中文简介（来源：项目官方 README）：** RSI-Bench 是一个用于评估 AI 系统递归自我改进能力的开源基准框架。它把 RSI 分解为六个可测量维度：自我修改深度、改进轨迹质量、算子发现率、元适应速度、安全与稳定性，以及自主目标生成。项目提供 BCa bootstrap 置信区间、收敛检测、多目标 Pareto 分析和安全边界验证，并用加权调和平均数计算综合 RSI 分数，以避免某一维度的高分掩盖其他维度的缺失。该框架的目标是区分真实的自我改进与指标投机。

**年份与发表信息：** community framework 2026

**English title:** RSI-Bench: Multi-Axis Benchmark for Recursive Self-Improvement

**Description (Source: official project README):** An open-source benchmark framework for evaluating Recursive Self-Improvement (RSI) capabilities in AI systems. RSI-Bench decomposes RSI into six measurable axes with statistical rigor, providing researchers with concrete tools to distinguish genuine self-improvement from metric gaming. The six axes are Self-Modification Depth, Improvement Trajectory Quality, Operator Discovery Rate, Meta-Adaptation Speed, Safety & Stability, and Autonomous Goal Generation. The framework includes BCa bootstrap confidence intervals, convergence detection, multi-objective Pareto analysis, and safety-boundary verification under self-modification.

#### 4. [RSIBench-Data：面向数据中心型递归自我改进的基准](https://arxiv.org/abs/2607.25886)

**中文摘要：** 递归自我改进要求系统能够把模型失败的证据转化为更好的模型。数据中心型后训练研究是这一过程的核心环节，包括诊断能力缺口、设计并验证训练数据策略，以及从检查点反馈中学习。LLM 智能体能否自动完成这一数据中心型改进循环？现有自动后训练基准把研究决策与优化、服务、评估和系统实现混在一起，难以单独衡量智能体的研究能力。为弥补这一缺口，作者提出 RSIBench-Data：固定后训练技术栈，在受控条件下把 LLM 智能体作为数据研究者进行评估。智能体需要为固定目标模型反复制定和修改训练数据策略；训练与服务使用共享的 Tinker 后端服务，正式评估在 Harbor 编排的 E2B 沙箱中运行，所有智能体的资源预算保持一致。研究在软件工程、终端使用、科学问答和数学等六个基准上评估了四个前沿智能体。LLM 智能体已经具备一部分数据研究者所需的能力：在 58.33% 的设置中，它们根据反馈改进了第一次有效尝试所用的训练数据策略。但它们无法稳定地从反馈中获益：在达到观测到的最佳分数后仍继续搜索的轨迹中，78.26% 最终以更低分结束，其余轨迹也只能回到相同峰值。因此，较好的候选方案可能在运行早期或中途出现，后续基于反馈的修改却未能继续改进。轨迹分析发现，较强的运行通常具备四种模式：准确的假设、以验证结果为依据的监督、与行为目标一致的数据，以及保留强检查点。结果说明，当前智能体能够做出有用的数据研究发现，但还不能稳定地把反馈转化为改进。RSIBench-Data 为研究递归自我改进所需的数据研究能力提供了可测量、可审计的测试平台。代码已在 https://github.com/evolvent-ai/RSIBench-Data 开源。

**年份与发表信息：** arXiv 2026

**English title:** RSIBench-Data

**Abstract:** Recursive self-improvement requires systems that can turn evidence about model failures into better models. A central part of this process is data-centric post-training research: diagnosing a capability gap, designing and validating a training-data strategy, and learning from checkpoint feedback. Can LLM agents automate this data-centric improvement loop? Existing automated post-training benchmarks entangle research decisions with optimization, serving, evaluation, and system implementation, making it difficult to isolate the agent’s research capability. To fill this gap, we introduce RSIBench-Data, a controlled benchmark evaluating LLM agents as data-centric researchers while fixing the post-training stack. Specifically, agents are required to iteratively develop and revise training-data strategies for a fixed target model, while training and serving use shared Tinker-backed services, official evaluation runs through Harbor-orchestrated E2B sandboxes, and resource budgets remain fixed across agents. We evaluate four frontier agents across six benchmarks spanning software engineering, terminal use, science question answering, and mathematics. LLM agents already demonstrate some of the core capabilities of data-centric researchers: in 58.33% of settings, they improve upon the first valid attempt by refining their training-data strategies from feedback. However, they do not improve consistently from feedback: among searches that continue after reaching their best observed score, 78.26% end with a lower-scoring final attempt, while the rest only recover the same peak. Thus, a strong candidate can appear early or midway through a run even as later feedback-driven revisions fail to improve it. Trajectory analysis identifies four patterns in stronger runs: accurate hypotheses, validation-grounded supervision, behavior-aligned data, and preservation of strong checkpoints. These findings suggest that current agents can make useful data-centric research discoveries but cannot yet translate feedback into improvements consistently. Overall, RSIBench-Data provides a measurable and auditable testbed for studying the data-centric research capabilities required for recursive self-improvement. We open source our code at https://github.com/evolvent-ai/RSIBench-Data.

### 前沿实验室自我改进与 AI 研发评估框架 / Frontier Lab Self-Improvement & AI R&D Evaluation Frameworks

#### 5. [Anthropic 自主 AI 研发评估](https://www.anthropic.com/transparency/model-report)

**中文简介（来源：机构官方页面）：** Anthropic 的模型报告汇总 Claude 模型的能力、安全评估和部署保障措施。其负责任扩展政策把 AI R&D-4 能力阈值定义为：系统能够完全自动完成 Anthropic 初级、仅远程研究员的工作。达到该阈值将触发更严格的模型权重保护和部署保障要求；Anthropic 会结合模型能力评估、安全案例与风险报告来判断模型是否跨越阈值。2026 年 2 月发布的评估结论认为 Claude Opus 4.6 未跨越 AI R&D-4，但也指出，要有把握地排除这一阈值正变得越来越困难。

**年份与发表信息：** Anthropic Model Report 2026

**English title:** Anthropic Autonomous AI R&D Evaluations

**Description (Source: official organization page):** Anthropic’s Model Report provides summaries of Claude models’ capabilities, safety evaluations, and deployment safeguards. Anthropic’s Responsible Scaling Policy defines AI R&D-4 as the ability to fully automate the work of an entry-level, remote-only researcher at Anthropic. Reaching the threshold triggers stronger model-weight security and deployment-safeguard requirements, assessed through capability evaluations, safety cases, and risk reports. Anthropic’s February 2026 assessment determined that Claude Opus 4.6 did not cross the AI R&D-4 threshold, while noting that confidently ruling out the threshold was becoming increasingly difficult.

#### 6. [Google DeepMind 前沿安全框架（FSF）的机器学习研发评估](https://deepmind.google/blog/strengthening-our-frontier-safety-framework/)

**中文简介（来源：机构官方页面）：** Google DeepMind 的第三版前沿安全框架扩大了风险领域，并细化了风险评估流程。框架针对可能把 AI 研发加速到具有潜在破坏性水平的机器学习研发能力设置关键能力等级（CCL），并用早期预警评估、系统性风险识别、能力分析和风险可接受性判断开展评估。FSF 3.1 又引入跟踪能力等级（TCL），以便更早发现并评估尚未达到 CCL 的风险。对高级机器学习研发能力，框架要求在相关 CCL 被触及时，对外发布以及大规模内部部署进行安全案例审查。

**年份与发表信息：** Google DeepMind Blog 2026

**English title:** Google DeepMind Frontier Safety Framework (FSF) ML R&D

**Description (Source: official organization page):** The third iteration of Google DeepMind’s Frontier Safety Framework expands its risk domains and refines its risk-assessment process. For machine learning research and development capabilities that could accelerate AI R&D to potentially destabilizing levels, the framework uses Critical Capability Levels (CCLs), early-warning evaluations, systematic risk identification, capability analysis, and explicit determinations of risk acceptability. FSF 3.1 adds Tracked Capability Levels (TCLs) to identify and evaluate less extreme risks sooner. When relevant advanced ML R&D CCLs are reached, safety-case reviews apply to external launches and large-scale internal deployments.

#### 7. [OpenAI AI 自我改进评估](https://deploymentsafety.openai.com/gpt-5-6)

**中文简介（来源：机构官方页面）：** OpenAI 在 GPT-5.6 系统卡中更新并扩充了自我改进评估，重点转向新模型能够尝试的现实、端到端任务；旧评估有的已趋于饱和，有的包含在测试条件下无法解决的问题。新版评估套件包括内部研究调试、KernelGen 1P、NanoGPT、PostTrainBench Lite 和修订版 MLE-Bench：分别测试模型能否修复真实研究实验中的缺陷、为 OpenAI 第一方硬件编写并优化内核、在算力与时间限制下改进小型语言模型训练、设计完整的后训练方案，以及完成真实数据科学和机器学习竞赛任务。系统卡将这些结果用于 Preparedness Framework 下的 AI Self-Improvement 能力评估；OpenAI 的结论是 GPT-5.6 Sol、Terra 和 Luna 均未达到 AI 自我改进的 High 阈值。

**年份与发表信息：** OpenAI System Card 2026

**English title:** OpenAI AI Self-Improvement Evaluations

**Description (Source: official organization page):** Starting with the GPT-5.6 launch, OpenAI updated and expanded its self-improvement evaluations to better capture realistic, end-to-end tasks that newer models can attempt. The revised suite includes Internal Research Debugging Evaluation, KernelGen 1P, NanoGPT, PostTrainBench Lite, and MLE-Bench Revised, covering research debugging, kernel optimization, language-model training optimization, post-training strategy development, and real-world data-science competitions. OpenAI uses these evaluations under its Preparedness Framework and reports that GPT-5.6 Sol, Terra, and Luna do not reach the High threshold in AI Self-Improvement.

### AI 研发能力代理指标 / AI R&D Capability Proxies

#### 8. [AutoLab：前沿模型能否解决长时程自动化研究与工程任务？](https://arxiv.org/abs/2606.05080)

**中文摘要：** 科学与工程进步本质上是一个长时程迭代过程：提出修改、运行实验、测量结果，并持续改进成果。现有前沿模型基准主要评估单轮回答或短时程智能体轨迹，无法覆盖长时间持续迭代改进所带来的困难。为此，作者提出 AutoLab，一项用于超长时程闭环优化的新基准。AutoLab 包含 36 个由专家筛选的真实任务，覆盖系统优化、谜题与挑战、模型开发和 CUDA 内核优化四个领域。每个任务从一个正确但刻意未优化的基线开始，要求智能体在严格的实际时间预算内改进它。对 17 个先进模型的评估发现，决定成功的主要因素不是智能体首次尝试的质量，而是其反复运行基准、修改方案并吸收实证反馈的坚持程度。Claude Opus 4.6 表现出较强的长时程优化能力，但大多数前沿模型（包括多个闭源模型）要么过早结束，要么耗尽预算却几乎没有进展。结果说明，时间意识和持续迭代对自主智能体十分重要。作者开源了完整基准、评估框架和任务材料，以推动长时程智能体研究。

**年份与发表信息：** arXiv 2026

**English title:** AutoLab: Can Frontier Models Solve Long-Horizon Auto Research and Engineering Tasks?

**Abstract:** Scientific and engineering progress is fundamentally a long-horizon iterative process: proposing changes, running experiments, measuring outcomes, and continuously refining artifacts. Yet existing benchmarks for frontier models primarily evaluate either single-turn responses or short-horizon agent trajectories, failing to capture the challenges of sustained iterative improvement over extended time horizons. To address this gap, we introduce AutoLab, a new benchmark for ultra long-horizon closed-loop optimization. AutoLab consists of 36 realistic, expert-curated tasks spanning four diverse domains: system optimization, puzzle & challenge, model development, and CUDA kernel optimization. Each task begins with a correct but deliberately suboptimal baseline and challenges agents to improve it within a strict wall-clock budget. Evaluating 17 state-of-the-art models reveals the dominant predictor of success is not the quality of an agent’s initial attempt, but its persistence in repeatedly benchmarking, editing, and incorporating empirical feedback. While Claude Opus 4.6 exhibits strong long-horizon optimization capabilities, most frontier models, including several proprietary ones, either terminate prematurely or exhaust their budgets with minimal progress. These results underscore the importance of time awareness and persistent iteration in autonomous agents. We open-source the full benchmark, evaluation harness, and task artifacts, to accelerate research toward truly capable long-horizon agents.

#### 9. [MLS-Bench：对 AI 系统构建更好 AI 能力的整体、严格评估](https://arxiv.org/abs/2605.08678)

**中文摘要：** 现代 AI 的进步依赖于能够跨场景泛化并扩展到更大规模的机器学习方法。大语言模型已经具备较强的推理、编码和工程能力，因此一个日益重要的问题是：它们能否发现这类方法，而不只是应用已有方法？作者提出 MLS-Bench，用于评估 AI 系统能否发明可泛化、可扩展的机器学习方法。MLS-Bench 包含 12 个领域的 140 项任务；每项任务都要求智能体改进机器学习系统或算法中的一个指定组件，并证明改进能够在受控设置中泛化和扩展。实验发现，当前智能体还远不能稳定超过人工设计的方法，而且工程式调优比真正的方法发明更容易。研究进一步考察了测试时扩展、自适应算力分配和上下文供给对智能体发现能力的影响，并分析了智能体行为案例。结果表明，瓶颈不只在于提出新方法，也在于规划、验证和扩展方法主张所需的科学洞察；单纯增加搜索、算力或上下文并不能消除这一瓶颈。作者建立并维护了一个支持累积、可比迭代的社区平台，并在 https://mls-bench.com 发布数据和代码。

**年份与发表信息：** arXiv 2026

**English title:** MLS-Bench: A Holistic and Rigorous Assessment of AI Systems on Building Better AI

**Abstract:** Modern AI progress has been driven by ML methods that are generalizable across settings and scalable to larger regimes. As large language models demonstrate advanced capabilities in reasoning, coding, and engineering tasks, it is increasingly important to understand whether they can discover such methods rather than only apply existing ones. We introduce MLS-Bench, a benchmark for evaluating whether AI systems can invent generalizable and scalable ML methods. MLS-Bench contains 140 tasks across 12 domains, each requiring an agent to improve one targeted component of an ML system or algorithm and demonstrate that the improvement generalizes across controlled settings and scales. We find that current agents remain far from reliably surpassing human-designed methods, and that engineering-style tuning is easier for them than genuine method invention. We further study the effects of test-time scaling, adaptive compute allocation, and context provision on agents’ discovery performance, together with case studies of their behavior. Our analyses suggest that the bottleneck is not only in proposing new methods, but also in the scientific insight needed to plan, validate, and scale claims about them. More search, compute, or context alone does not remove this bottleneck. We build and maintain a community platform for cumulative and comparable iteration, and release the data and code at https://mls-bench.com.

#### 10. [MLE-bench：机器学习工程智能体评估基准](https://github.com/openai/mle-bench)

**中文简介（来源：项目官方 README）：** MLE-bench 用 75 场 Kaggle 竞赛衡量 AI 智能体的端到端机器学习工程能力。项目公开了数据集构建代码、评估逻辑和受测智能体。基准不预设智能体结构；智能体需要处理竞赛数据、训练模型并生成可评分的提交文件。官方建议在 24 小时、36 个 vCPU、440 GB 内存和一张 24 GB A10 GPU 的设置下，以至少三个随机种子报告 Any Medal 指标及标准误，并分别给出 Low、Medium、High 和 All 难度分组结果。低复杂度的 Lite 版本包含 22 场竞赛。

**年份与发表信息：** ICLR 2025

**English title:** MLE-bench

**Description (Source: official project README):** MLE-bench is a benchmark for measuring how well AI agents perform at machine learning engineering. Its dataset is a collection of 75 Kaggle competitions, and the repository releases the code used to construct the dataset, the evaluation logic, and the agents evaluated for the benchmark. The benchmark makes no assumptions about the agent that produces submissions. Its recommended canonical setup reports the Any Medal percentage across Low, Medium, High, and All complexity splits, with a 24-hour runtime and specified compute resources. A 22-competition Low-complexity split is provided as a Lite evaluation.

#### 11. [PaperBench：评估 AI 复现 AI 研究的能力](https://github.com/openai/frontier-evals/tree/main/project/paperbench)

**中文简介（来源：项目官方 README）：** PaperBench 要求 AI 智能体从零复现 20 篇 ICML 2024 Spotlight 或 Oral 论文。每个样本包含一篇研究论文和一套定义成功复现要求的评分细则。评估分为三个阶段：智能体先在 Ubuntu 容器中创建复现代码库；随后在带 GPU 的全新容器中运行提交；最后由第三个容器中的评审器依据论文评分细则进行评分。项目还提供 PaperBench Code-Dev 这一成本较低的变体，只检查代码开发要求，不执行完整复现实验，也不核验结果是否与论文一致。

**年份与发表信息：** ICML 2025

**English title:** PaperBench

**Description (Source: official project README):** PaperBench evaluates AI agents on replicating 20 Spotlight and Oral papers from ICML 2024 from scratch. Each sample includes a research paper and a rubric defining the requirements for a successful replication. Evaluation has three stages: Agent Rollout, where the agent creates a replication codebase in an Ubuntu container; Reproduction, where the submitted code is executed in a fresh GPU-enabled container; and Grading, where the executed submission is judged against the paper’s rubric. PaperBench Code-Dev provides a lighter variant that grades only code-development requirements without full reproduction or result matching.

#### 12. [RE-Bench：对照人类专家评估前沿 AI 研发能力](https://github.com/METR/RE-Bench)

**中文简介（来源：项目官方 README）：** RE-Bench 用开放式机器学习研发任务比较 AI 智能体与人类专家，目标是衡量智能体的自主 AI 研发能力。任务遵循 METR Task Standard，可在统一任务格式、固定时间预算和可复现实验环境下运行。任务族包括：为 Rust CodeContests 构建脚手架、用强化学习微调 GPT-2、修复被置换的 embedding、优化 LLM Foundry 微调脚本、优化 Triton GPU 内核、依据缩放规律预测最佳超参数，以及在受限算子条件下构建语言模型。

**年份与发表信息：** ICML 2025

**English title:** RE-Bench

**Description (Source: official project README):** RE-Bench evaluates frontier AI R&D capabilities of language-model agents against human experts. Its tasks are intended as evaluation material for measuring autonomous AI R&D capabilities and conform to the METR Task Standard for easier task sharing and agent evaluation. Task families include building Rust CodeContests scaffolding, fine-tuning GPT-2 for question answering with reinforcement learning, repairing permuted embeddings, optimizing an LLM Foundry fine-tuning script, optimizing a Triton kernel, predicting optimal hyperparameters through a scaling law, and building a language model from restricted primitives.

#### 13. [MLAgentBench：机器学习实验中的语言智能体评估](https://github.com/snap-stanford/MLAgentBench)

**中文简介（来源：项目官方 README）：** MLAgentBench 是一组端到端机器学习实验任务。给定数据集和机器学习任务描述后，智能体需要自主开发或改进模型。每项任务都提供类似人类研究者实际工作条件的交互环境：智能体可以读取文件、在计算集群上多次运行实验并分析结果，以完成指定研究目标。基准包含 13 项多样化的机器学习工程任务，涉及方法选择、数据处理、模型架构和训练流程等环节；其成功率指标衡量最终步骤相对于起始代码基线是否提升超过 10%。

**年份与发表信息：** ICML 2024

**English title:** MLAgentBench: Evaluating Language Agents on Machine Learning Experimentation

**Description (Source: official project README):** MLAgentBench is a suite of end-to-end machine learning experimentation tasks for benchmarking AI agents, where the agent aims to take a given dataset and a machine learning task description and autonomously develop or improve an ML model. Each task is an interactive environment resembling what human researchers see: an agent can read available files, run multiple experiments on a compute cluster, and analyze results to achieve the specified research goal. The suite includes 13 diverse ML engineering tasks involving machine learning methods, data processing, architectures, and training processes.

### 智能体能力代理指标 / Agent Capability Proxies

#### 14. [ARC-AGI-3：面向前沿智能体智能的新挑战](https://arxiv.org/abs/2603.24621)

**中文摘要：** 作者提出 ARC-AGI-3，一个通过新颖、抽象的回合制环境研究智能体智能的交互式基准。智能体没有明确指令，必须自行探索、推断目标、建立环境动态的内部模型并规划有效的动作序列。与 ARC-AGI-1 和 ARC-AGI-2 一样，ARC-AGI-3 完全关注模型在新任务上的流体适应效率，同时避开语言和外部知识。其环境只利用核心知识先验，并通过大量人类测试者校准难度。测试显示，人类可以解决 100% 的环境；相比之下，截至 2026 年 3 月，前沿 AI 系统的得分低于 1%。论文介绍了基准设计、以人类动作基线为依据的效率评分框架，以及构建、验证和校准环境的方法。

**年份与发表信息：** arXiv 2026

**English title:** ARC-AGI-3: A New Challenge for Frontier Agentic Intelligence

**Abstract:** We introduce ARC-AGI-3, an interactive benchmark for studying agentic intelligence through novel, abstract, turn-based environments in which agents must explore, infer goals, build internal models of environment dynamics, and plan effective action sequences without explicit instructions. Like its predecessors ARC-AGI-1 and 2, ARC-AGI-3 focuses entirely on evaluating fluid adaptive efficiency on novel tasks, while avoiding language and external knowledge. ARC-AGI-3 environments only leverage Core Knowledge priors and are difficulty-calibrated via extensive testing with human test-takers. Our testing shows humans can solve 100% of the environments, in contrast to frontier AI systems which, as of March 2026, score below 1%. In this paper, we present the benchmark design, its efficiency-based scoring framework grounded in human action baselines, and the methodology used to construct, validate, and calibrate the environments.

#### 15. [长时程终端基准：用密集奖励评分测试智能体处理长时程终端任务的极限](https://arxiv.org/abs/2607.08964)

**中文摘要：** AI 智能体越来越能够自主完成简短、定义明确的任务，但现有终端基准主要关注几分钟内即可完成的相对简单问题，且通常只根据最终结果评分。这种设置忽略了中间进展和部分解，导致奖励稀疏，无法完整反映智能体能力。作者提出 Long-Horizon-Terminal-Bench，包含 46 项高难度长时程终端任务，覆盖实验复现、软件工程、多模态分析、交互式游戏和科学计算等九类任务。每项任务采用类似 Terminal-Bench 的设置，配有参考解或模拟引擎，并进一步拆分为可细粒度评分的子任务。这样可以提供密集中间奖励和部分分数，既衡量智能体是否达到最终目标，也衡量它在困难开放式工作流中推进了多远。基准任务通常需要数百轮交互及数十分钟至数小时的执行，主要考查长时程规划、长上下文管理和迭代调试，而非单次求解。作者评估了 17 个前沿模型；每项任务平均消耗 980 万 token、约 239 个 episode 和 88.9 分钟。最强的受测模型 Grok 4.5 在部分奖励阈值 0.95 下的 pass@1 仅为 28.3%，在满分阈值 1.0 下为 19.6%；所有模型的平均通过率分别只有 6.4% 和 3.2%。这些结果表明能力仍有很大提升空间。论文还分析了常见失败模式和错误类型，并发布该基准以支持稳健长时程终端智能体研究。

**年份与发表信息：** arXiv 2026

**English title:** Long-Horizon-Terminal-Bench: Testing the Limits of Agents on Long-Horizon Terminal Tasks with Dense Reward-Based Grading

**Abstract:** AI agents have become increasingly capable of autonomously completing short, well-specified tasks. However, existing terminal benchmarks largely focus on relatively simple problems that finish within a few minutes and are typically evaluated only by their final outcome. This setup overlooks intermediate progress and partial solutions, leading to sparse reward signals and an incomplete picture of agent capability. We introduce Long-Horizon-Terminal-Bench, a challenging terminal benchmark of 46 long-horizon tasks spanning nine categories, including experiment reproduction, software engineering, multimodal analysis, interactive games, and scientific computing. Each task follows a Terminal-Bench-style setup with a reference solution or simulation engine, but is further decomposed into fine-grained graded subtasks. This design enables dense intermediate rewards and partial credit, allowing evaluation to capture not only whether an agent reaches the final goal, but also how far it progresses on difficult, open-ended workflows. Tasks in Long-Horizon-Terminal-Bench typically require hundreds of episodes and tens of minutes to hours of execution, stressing long-horizon planning, long-context management, and iterative debugging rather than one-shot problem solving. We evaluate 17 frontier models and find that agents consume on average 9.8M tokens per task, with roughly 239 episodes and 88.9 minutes of execution time per run, making Long-Horizon-Terminal-Bench substantially more demanding than prior terminal-based benchmarks. Even the strongest tested model, Grok 4.5, achieves only 28.3% pass@1 at a partial-reward threshold of 0.95 and 19.6% at a perfect-reward threshold of 1.0, while the mean pass rate across models is just 6.4% and 3.2% under the two thresholds, respectively. These results reveal substantial headroom for improvement. We further analyze common failure modes and error patterns, and release Long-Horizon-Terminal-Bench to support future progress on robust long-horizon terminal agents.

#### 16. [OSWorld 2.0](https://arxiv.org/abs/2606.29537)

**中文摘要：** 现有计算机使用基准未能覆盖真实世界计算机操作的真实性、复杂度和长时程要求，因此难以揭示前沿智能体的局限。作者提出 OSWorld 2.0，其中包含 108 个横跨日常与专业场景的长时程计算机使用工作流，专门覆盖复杂、具有挑战性的现实情形。每项任务都是现实的端到端工作流，人类完成时间中位数约为 1.6 小时；使用最大思考强度的 Claude Opus 4.7 平均需要 318 次工具调用，而 OSWorld 1.0 约为 30 次。OSWorld 2.0 针对以往基准很少覆盖、但在实际工作流中很常见的困难，包括流式交互和动态环境等交互设计问题，以及跨来源推理、隐式状态推断和视觉空间精度等智能体行为模式问题。任务使用真实输入材料，并与有状态的真实用户画像数据交叉关联；对安全敏感的执行还附有单独的安全审计报告。在 500 步、二元完成指标的主要设置下，使用最大思考强度和批量工具调用的 Claude Opus 4.8 表现最好，但仍只完成 20.6% 的任务，部分得分为 54.8%；GPT-5.5 的 token 效率高得多，但完成率停留在约 13%。结果表明，当前智能体距离专业级计算机使用仍很远。它们的主要问题不是基本 GUI 控制或编码失误，而是遗忘约束、漏掉任务中途出现的信息、在应询问用户时自行猜测，以及跳过验证；当任务依赖必须恢复的隐藏状态时，困难最明显。

**年份与发表信息：** arXiv 2026

**English title:** OSWorld 2.0

**Abstract:** Existing computer-use benchmarks fail to capture the realism, complexity, and long-horizon demands of real-world computer use, limiting their ability to reveal the limitations of frontier agents. We introduce OSWorld 2.0, a benchmark of 108 long-horizon computer-use workflows across everyday and professional tasks, designed to capture complex and challenging real-world phenomena. Each task represents a realistic end-to-end workflow that takes human users a median of about 1.6 hours to complete and requires an average of 318 tool calls with Claude Opus 4.7 using maximum thinking, compared with about 30 in OSWorld 1.0. OSWorld 2.0 targets challenge phenomena that are common in real workflows yet underrepresented in prior benchmarks, spanning interaction-design challenges such as streaming interaction and dynamic environments, as well as agent-pattern challenges such as cross-source reasoning, implicit-state inference, and visual-spatial precision. Tasks are grounded in authentic input artifacts and cross-referenced against realistic stateful user profile data, and include separate safety reports auditing safety-sensitive execution. Under our primary binary-completion metric at 500 steps, Claude Opus 4.8 with maximum thinking and batched tool calls scores best but still completes only 20.6% of tasks at a 54.8% partial score; GPT-5.5 is far more token-efficient yet plateaus near 13%. These results show that current agents are still far from professional-level computer use: rather than stumbling on basic GUI control or coding, they lose track of constraints, miss information that arrives mid-task, guess rather than ask the user, and skip verification, struggling most when a task hinges on hidden state they must recover.

#### 17. [MCPMark](https://arxiv.org/abs/2509.24002)

**中文摘要：** MCP 对 LLM 与外部系统的交互方式进行了标准化，是通用智能体的重要基础。然而，现有 MCP 基准范围较窄，主要关注读取操作较多或交互深度有限的任务，无法反映现实工作流的复杂度。为此，作者提出 MCPMark，一个更贴近现实且覆盖更全面的 MCP 使用基准。它包含 127 个由领域专家与 AI 智能体协作创建的高质量任务。每项任务都从经过整理的初始状态开始，并配有用于自动验证的程序脚本。任务要求智能体与环境进行更丰富、多样的交互，涉及广泛的创建、读取、更新和删除（CRUD）操作。作者使用一个执行工具调用循环的简约智能体框架，对先进 LLM 进行了全面评估。实验显示，表现最好的 gpt-5-medium 仅达到 52.56% pass@1 和 33.86% pass^4；其他公认的强模型，包括 claude-sonnet-4 和 o3，均低于 30% pass@1 和 15% pass^4。LLM 平均每项任务需要 16.2 轮执行和 17.4 次工具调用，明显超过以往 MCP 基准，说明 MCPMark 具有较高压力测试强度。

**年份与发表信息：** arXiv 2025

**English title:** MCPMark

**Abstract:** The MCP standardizes how LLMs interact with external systems, forming the foundation for general agents. However, existing MCP benchmarks remain narrow in scope: they focus on read-heavy tasks or tasks with limited interaction depth, and fail to capture the complexity and realism of real-world workflows. To address this gap, we propose MCPMark, a benchmark designed to evaluate MCP use in a more realistic and comprehensive manner. It consists of 127 high-quality tasks collaboratively created by domain experts and AI agents. Each task begins with a curated initial state and includes a programmatic script for automatic verification. These tasks demand richer and more diverse interactions with the environment, involving a broad range of create, read, update, and delete (CRUD) operations. We conduct a comprehensive evaluation of cutting-edge LLMs using a minimal agent framework that operates in a tool-calling loop. Empirical results show that the best-performing model, gpt-5-medium, reaches only 52.56% pass@1 and 33.86% pass^4, while other widely regarded strong models, including claude-sonnet-4 and o3, fall below 30% pass@1 and 15% pass^4. On average, LLMs require 16.2 execution turns and 17.4 tool calls per task, significantly surpassing those in previous MCP benchmarks and highlighting the stress-testing nature of MCPMark.

#### 18. [METR 任务完成时间跨度](https://arxiv.org/abs/2503.14499)

**中文摘要：** 尽管 AI 基准进展迅速，基准表现对应怎样的真实世界能力仍不清楚。为把 AI 系统能力换算为人类能力，作者提出一个新指标：50% 任务完成时间跨度，即人类通常完成一类任务所需的时间，而 AI 模型在这类任务上的成功率为 50%。研究先让具备相关领域经验的人类完成 RE-Bench、HCAST 和 66 项新增的较短任务，并记录用时。在这些任务上，o3 等当前前沿 AI 模型的 50% 时间跨度约为 110 分钟。自 2019 年以来，前沿 AI 的时间跨度大约每七个月翻一番，不过这一趋势可能从 2024 年起有所加速。AI 模型时间跨度的增长似乎主要来自可靠性、错误适应能力、逻辑推理和工具使用能力的提升。论文讨论了结果的局限，包括外部有效性的限制，以及自主性提高对危险能力的影响。如果这些结果能够推广到真实软件任务，按该趋势外推，AI 系统将在五年内能够自动完成许多目前需要人类一个月才能完成的软件任务。

**年份与发表信息：** NeurIPS 2025

**English title:** METR Task-Completion Time Horizon

**Abstract:** Despite rapid progress on AI benchmarks, the real-world meaning of benchmark performance remains unclear. To quantify the capabilities of AI systems in terms of human capabilities, we propose a new metric: 50%-task-completion time horizon, the time humans typically take to complete tasks that AI models can complete with 50% success rate. We first timed humans with relevant domain expertise on a combination of RE-Bench, HCAST, and 66 novel shorter tasks. On these tasks, current frontier AI models such as o3 have a 50% time horizon of around 110 minutes. Furthermore, frontier AI time horizon has doubled approximately every seven months since 2019, though the trend may have accelerated since 2024. The increase in AI models’ time horizons seems to be primarily driven by greater reliability, ability to adapt to mistakes, logical reasoning, and capacity for tool use. We discuss the limitations of our results—including their degree of external validity—and the implications of increased autonomy for dangerous capabilities. If these results generalize to real-world software tasks, extrapolation of this trend predicts that within 5 years, AI systems will be capable of automating many software tasks that currently take humans a month.

#### 19. [SWE-Bench Pro](https://arxiv.org/abs/2509.16941)

**中文摘要：** 作者提出 SWE-Bench Pro。它吸收 SWE-Bench 的最佳实践，但难度显著提高，专门覆盖 SWE-Bench 范围之外的现实、复杂企业级问题。SWE-Bench Pro 包含 1,865 个问题，来自 41 个持续维护的多样化代码仓库，涵盖商业应用、B2B 服务和开发者工具。基准分为三个部分：公开集包含 11 个仓库的问题；留出集包含 12 个仓库；商业集包含与早期创业公司正式合作取得的 18 个专有仓库。留出集和商业集的问题不公开，但项目会发布商业集结果。基准中的长时程任务可能需要专业软件工程师花费数小时至数天，常常涉及多文件补丁和大量代码修改。所有任务都经过人工验证，并补充了足够上下文以确保可解。在统一智能体脚手架下评估常用编码模型时，其 SWE-Bench Pro Pass@1 均低于 45%。为进一步理解这些限制，作者对智能体轨迹中的失败模式进行聚类，以更清楚地刻画当前模型的错误类型。SWE-Bench Pro 提供了一个抗污染测试平台，更贴近真实软件开发的复杂度和多样性，用于推进专业级自主软件工程智能体研究。

**年份与发表信息：** arXiv 2025

**English title:** SWE-Bench Pro

**Abstract:** We introduce SWE-Bench Pro, a substantially more challenging benchmark that builds upon the best practices of SWE-Bench, but is explicitly designed to capture realistic, complex, enterprise-level problems beyond the scope of SWE-Bench. SWE-Bench Pro contains 1,865 problems sourced from a diverse set of 41 actively maintained repositories spanning business applications, B2B services, and developer tools. The benchmark is partitioned into a public set with open access to problems sourced from 11 repositories, a held-out set of 12 repositories and a commercial set of 18 proprietary repositories where we have formal partnership agreements with early-stage startups. Problems in the held-out and the commercial set are not publicly accessible, but we release results on the commercial set. Our benchmark features long-horizon tasks that may require hours to days for a professional software engineer to complete, often involving patches across multiple files and substantial code modifications. All tasks are human-verified and augmented with sufficient context to ensure resolvability. In our evaluation of widely used coding models, under a unified scaffold, we observe that their performance on SWE-Bench Pro remains below 45% (Pass@1). To better understand these limitations, we cluster the failure modes observed in the collected agent trajectories for a clearer characterization of the error patterns exhibited by current models. Overall, SWE-Bench Pro provides a contamination-resistant testbed that more faithfully captures the complexity and diversity of real-world software development, advancing the pursuit of truly autonomous software engineering agents at a professional level.

#### 20. [TheAgentCompany](https://arxiv.org/abs/2412.14161)

**中文摘要：** 无论生活还是工作，人们每天都会与计算机交互，许多工作只需计算机和互联网即可完成。与此同时，随着大语言模型进步，能够与周围环境交互并对其作出改变的 AI 智能体也在快速发展。但 AI 智能体在加速乃至自主完成工作任务方面究竟表现如何？这一问题既关系到企业如何把 AI 引入工作流，也关系到经济政策如何理解 AI 应用对劳动力市场的影响。为衡量 LLM 智能体完成现实专业任务的进展，作者提出 TheAgentCompany，一个可扩展基准，用于评估 AI 智能体是否能像数字员工一样与外部环境交互，包括浏览网页、编写代码、运行程序以及与同事沟通。研究构建了一个自包含环境，使用内部网站和数据模拟小型软件公司，并设计了这类公司员工可能承担的多种任务。作者测试了由闭源 API 模型和开放权重语言模型驱动的基线智能体，发现表现最好的智能体能够自主完成 30% 的任务。这给出了较为细致的自动化图景：在模拟真实工作场所的环境中，不少较简单任务可以自主解决，但更困难的长时程任务仍超出当前系统能力。代码、数据、环境和实验已发布于 https://the-agent-company.com。

**年份与发表信息：** NeurIPS 2025

**English title:** TheAgentCompany

**Abstract:** We interact with computers on an everyday basis, be it in everyday life or work, and many aspects of work can be done entirely with access to a computer and the Internet. At the same time, thanks to improvements in large language models (LLMs), there has also been a rapid development in AI agents that interact with and affect change in their surrounding environments. But how performant are AI agents at accelerating or even autonomously performing work-related tasks? The answer to this question has important implications both for industry looking to adopt AI into their workflows and for economic policy to understand the effects that adoption of AI may have on the labor market. To measure the progress of these LLM agents’ performance on performing real-world professional tasks, in this paper we introduce TheAgentCompany, an extensible benchmark for evaluating AI agents that interact with the world in similar ways to those of a digital worker: by browsing the Web, writing code, running programs, and communicating with other coworkers. We build a self-contained environment with internal web sites and data that mimics a small software company environment, and create a variety of tasks that may be performed by workers in such a company. We test baseline agents powered by both closed API-based and open-weights language models (LMs), and find that the most competitive agent can complete 30% of tasks autonomously. This paints a nuanced picture on task automation with LM agents—in a setting simulating a real workplace, a good portion of simpler tasks could be solved autonomously, but more difficult long-horizon tasks are still beyond the reach of current systems. We release code, data, environment, and experiments on https://the-agent-company.com.

#### 21. [SWE-bench](https://github.com/SWE-bench/SWE-bench)

**中文简介（来源：项目官方 README）：** SWE-bench 用从 GitHub 收集的真实软件问题评估大语言模型。每项任务向模型提供代码库和 issue 描述，要求模型生成修复该问题的补丁；补丁随后在可复现的 Docker 环境中接受测试。项目提供完整集、Lite、Verified、Multimodal 和 Multilingual 等数据集及评估工具。原始 SWE-bench 论文发表于 ICLR 2024，并获 Oral；该基准后来也被用于评估编码智能体和 DGM 等经验性自我修改系统。

**年份与发表信息：** ICLR 2024

**English title:** SWE-bench

**Description (Source: official project README):** SWE-bench is a benchmark for evaluating large language models on real-world software issues collected from GitHub. Given a codebase and an issue, a language model is tasked with generating a patch that resolves the described problem. The project provides reproducible Docker-based evaluation tooling and multiple datasets, including the full, Lite, Verified, Multimodal, and Multilingual variants. The original SWE-bench paper was presented as an oral at ICLR 2024.

#### 22. [SWE-bench Verified](https://openai.com/index/introducing-swe-bench-verified/)

**中文简介（来源：机构官方页面）：** SWE-bench Verified 是经过人工验证的 SWE-bench 子集，用于更可靠地评估 AI 模型解决真实软件问题的能力。OpenAI 与 SWE-bench 作者合作，组织 93 名有 Python 经验的软件开发者审查了 1,699 个随机样本，重点检查问题描述是否充分、`FAIL_TO_PASS` 测试是否会排除合理解法，以及环境设置是否可靠。最终数据集保留 500 个经验证无明显问题的样本，并发布全部人工标注和按难度切分的信息。该版本减少了损坏、描述不足或评分标准不公平的任务；配套评估框架使用容器化 Docker 环境提高复现性。

**年份与发表信息：** OpenAI benchmark 2024

**English title:** SWE-bench Verified

**Description (Source: official organization page):** SWE-bench Verified is a human-validated subset of SWE-bench that more reliably evaluates AI models’ ability to solve real-world software issues. OpenAI and the SWE-bench authors worked with 93 software developers experienced in Python to screen 1,699 random samples for underspecified issue descriptions, unfair `FAIL_TO_PASS` tests, and other major problems. The released subset contains 500 samples verified to be non-problematic, together with human annotations and difficulty labels. A containerized Docker evaluation harness improves setup reliability and reproducibility.

## 框架与工具 / Frameworks & Tools

### 自修改与自演化系统 / Self-Modifying / Self-Evolving Systems

#### [智能体工厂（AgentFactory）](https://github.com/zzatpku/AgentFactory)

**中文简介（来源：原列表简介）：** 一套自演化框架，通过积累并复用可执行子智能体，提高后续任务的求解能力。

**English title:** [AgentFactory](https://github.com/zzatpku/AgentFactory)

**Description (Source: original list description):** Self-evolving framework that accumulates and reuses executable subagents to improve future task solving.

#### [达尔文哥德尔机（Darwin Gödel Machine）](https://github.com/jennyzzt/dgm)

**中文简介（来源：原列表简介）：** 自修改编程智能体的官方实现，通过基于档案的开放式演化持续搜索改进。

**English title:** [Darwin Gödel Machine](https://github.com/jennyzzt/dgm)

**Description (Source: original list description):** Official self-modifying coding-agent implementation with open-ended archive-based evolution.

#### [哥德尔智能体（Gödel Agent）](https://github.com/Arvid-pku/Godel_Agent)

**中文简介（来源：原列表简介）：** 论文《Gödel Agent：用于递归自我改进的自指智能体框架》的官方实现。

**English title:** [Gödel Agent](https://github.com/Arvid-pku/Godel_Agent)

**Description (Source: original list description):** Official implementation of Gödel Agent: A Self-Referential Agent Framework for Recursive Self-Improvement.

#### [Hermes 智能体](https://github.com/NousResearch/hermes-agent)

**中文简介（来源：原列表简介）：** 带有内置学习循环的自我改进型个人智能体，能从经验中创建并完善技能。

**English title:** [Hermes Agent](https://github.com/NousResearch/hermes-agent)

**Description (Source: original list description):** Self-improving personal agent with a built-in learning loop that creates and refines skills from experience.

#### [HyperAgents](https://github.com/facebookresearch/HyperAgents)

**中文简介（来源：原列表简介）：** 自指智能体框架，其中可编辑的元智能体会针对可计算目标，改进自身和任务智能体。

**English title:** [HyperAgents](https://github.com/facebookresearch/HyperAgents)

**Description (Source: original list description):** Self-referential agent framework in which an editable meta-agent improves itself and the task agent for computable objectives.

#### [SEAL](https://github.com/Continual-Intelligence/SEAL)

**中文简介（来源：原列表简介）：** 语言模型自行生成适配数据与更新指令的官方代码。

**English title:** [SEAL](https://github.com/Continual-Intelligence/SEAL)

**Description (Source: original list description):** Official code for language models that generate their own adaptation data and update instructions.

#### [SIA](https://github.com/hexo-ai/sia)

**中文简介（来源：原列表简介）：** 自我改进 AI 框架，可迭代更新智能体 Harness，并在配置允许时更新目标模型的权重。

**English title:** [SIA](https://github.com/hexo-ai/sia)

**Description (Source: original list description):** Self-improving AI framework that iteratively updates an agent harness and, when configured, the target model's weights.

### Harness、记忆与技能演化 / Harness / Memory / Skill Evolution

#### [ACE](https://github.com/ace-agent/ace)

**中文简介（来源：原列表简介）：** “智能体上下文工程：为自我改进语言模型演化上下文”的官方实现。

**English title:** [ACE](https://github.com/ace-agent/ace)

**Description (Source: original list description):** Official implementation of Agentic Context Engineering: Evolving Contexts for Self-Improving Language Models.

#### [ALMA](https://github.com/zksha/alma)

**中文简介（来源：原列表简介）：** “通过元学习智能体记忆设计实现持续学习”的官方实现。

**English title:** [ALMA](https://github.com/zksha/alma)

**Description (Source: original list description):** Official implementation of Learning to Continually Learn via Meta-learning Agentic Memory Designs.

#### [Continual Harness](https://github.com/sethkarten/continual-harness)

**中文简介（来源：原列表简介）：** 从经验中在线适应的自我改进 Harness，同时维持连续的环境轨迹。

**English title:** [Continual Harness](https://github.com/sethkarten/continual-harness)

**Description (Source: original list description):** Self-improving agent harness that adapts online from experience while maintaining a continuous environment trajectory.

#### [EvoAgentX](https://github.com/EvoAgentX/EvoAgentX)

**中文简介（来源：原列表简介）：** 用于自动构建、评估和优化智能体工作流的自演化框架。

**English title:** [EvoAgentX](https://github.com/EvoAgentX/EvoAgentX)

**Description (Source: original list description):** Self-evolving agent framework for automatically building, evaluating, and optimizing agentic workflows.

#### [EvolveR](https://github.com/KnowledgeXLab/EvolveR)

**中文简介（来源：原列表简介）：** 通过经验驱动的闭环生命周期持续改进的自演化 LLM 智能体框架。

**English title:** [EvolveR](https://github.com/KnowledgeXLab/EvolveR)

**Description (Source: original list description):** Self-evolving LLM-agent framework that improves through a closed-loop, experience-driven lifecycle.

#### [Letta Code](https://github.com/letta-ai/letta-code)

**中文简介（来源：原列表简介）：** 以记忆为先的编程智能体 Harness；其中的长驻智能体会重写上下文，并从经验中学习技能。

**English title:** [Letta Code](https://github.com/letta-ai/letta-code)

**Description (Source: original list description):** Memory-first coding-agent harness whose long-lived agents rewrite context and learn skills from experience.

#### [Memento-Skills](https://github.com/Memento-Teams/Memento-Skills)

**中文简介（来源：原列表简介）：** 通过反思式学习检索、评估、修复并重写持久技能的自演化智能体框架。

**English title:** [Memento-Skills](https://github.com/Memento-Teams/Memento-Skills)

**Description (Source: original list description):** Self-evolving agent framework that retrieves, evaluates, repairs, and rewrites persistent skills through reflective learning.

#### [Reef](https://github.com/Human-Agent-Society/reef)

**中文简介（来源：原列表简介）：** 持续学习服务基础设施：记录智能体交互，把匹配到的反馈转化为模型权重或 Harness 更新，并将通过审核的更新发布为带版本的产物。

**English title:** [Reef](https://github.com/Human-Agent-Society/reef)

**Description (Source: original list description):** Continual-learning serving infrastructure that records agent interactions, turns matched feedback into model-weight or harness updates, and publishes accepted updates as versioned artifacts.

#### [Voyager](https://github.com/MineDojo/Voyager)

**中文简介（来源：原列表简介）：** 具身终身学习智能体，带有自动课程、迭代提示和可复用技能库。

**English title:** [Voyager](https://github.com/MineDojo/Voyager)

**Description (Source: original list description):** Embodied lifelong-learning agent with automatic curriculum, iterative prompting, and a reusable skill library.

### 自动化搜索与 AI 研发 / Automated Search / AI R&D

#### [ADAS](https://github.com/ShengranHu/ADAS)

**中文简介（来源：原列表简介）：** 在可执行智能体设计空间中进行搜索的元智能体官方实现。

**English title:** [ADAS](https://github.com/ShengranHu/ADAS)

**Description (Source: original list description):** Official implementation of a meta-agent that searches over executable agent designs.

#### [AI Scientist](https://github.com/SakanaAI/AI-Scientist)

**中文简介（来源：原列表简介）：** 端到端系统，可生成机器学习研究想法、运行实验并撰写研究论文。

**English title:** [AI Scientist](https://github.com/SakanaAI/AI-Scientist)

**Description (Source: original list description):** End-to-end system for generating machine-learning ideas, running experiments, and writing research papers.

#### [autoresearch](https://github.com/karpathy/autoresearch)

**中文简介（来源：原列表简介）：** 自主循环修改 LLM 训练程序，每次训练五分钟，只保留能降低验证集每字节比特数的改动。

**English title:** [autoresearch](https://github.com/karpathy/autoresearch)

**Description (Source: original list description):** Runs an autonomous loop that edits an LLM training program, trains for five minutes, and keeps only changes that improve validation bits per byte.

#### [进化式模型合并（Evolutionary Model Merge）](https://github.com/SakanaAI/evolutionary-model-merge)

**中文简介（来源：原列表简介）：** 在参数空间和数据流空间中演化开放模型的组合方式。

**English title:** [Evolutionary Model Merge](https://github.com/SakanaAI/evolutionary-model-merge)

**Description (Source: original list description):** Evolves combinations of open models in parameter and data-flow space.

#### [FunSearch](https://github.com/google-deepmind/funsearch)

**中文简介（来源：原列表简介）：** 由 LLM 引导、以可执行评估器验证的进化式程序搜索参考实现。

**English title:** [FunSearch](https://github.com/google-deepmind/funsearch)

**Description (Source: original list description):** Reference implementation of LLM-guided evolutionary program search with executable evaluators.

#### [MLEvolve](https://github.com/InternScience/MLEvolve)

**中文简介（来源：原列表简介）：** 面向端到端机器学习算法发现的自演化多智能体框架，采用渐进式搜索和经验驱动记忆。

**English title:** [MLEvolve](https://github.com/InternScience/MLEvolve)

**Description (Source: original list description):** Self-evolving multi-agent framework for end-to-end machine-learning algorithm discovery using progressive search and experience-driven memory.

#### [OpenEvolve](https://github.com/algorithmicsuperintelligence/openevolve)

**中文简介（来源：原列表简介）：** 受 AlphaEvolve 式程序优化启发的开源进化编程智能体。

**English title:** [OpenEvolve](https://github.com/algorithmicsuperintelligence/openevolve)

**Description (Source: original list description):** Open-source evolutionary coding agent inspired by AlphaEvolve-style program optimization.

#### [POET](https://github.com/uber-research/poet)

**中文简介（来源：原列表简介）：** 环境及其配对智能体协同演化的参考实现。

**English title:** [POET](https://github.com/uber-research/poet)

**Description (Source: original list description):** Reference implementation for co-evolving environments and their paired agents.

## 博客文章与讨论 / Blog Posts & Discussions

#### [自演化智能体分类法](https://lsl.zone/blog/2026/a-taxonomy-of-self-evolving-agents/)（2026）

**中文简介（来源：官方页面导语）：** 文章按演化对象把自演化系统分为模型、Harness 和产物，并据此讨论产物迭代优化、Harness 自我改进和无标准答案的模型学习，还说明这些方向与自我改进智能体、RSI、持续学习及测试时训练的关系。

**English title:** [A Taxonomy of Self-Evolving Agents](https://lsl.zone/blog/2026/a-taxonomy-of-self-evolving-agents/)

**Description (Source: original list description):** Distinguishes Model, Harness, and Artifact evolution, complementing this list's own taxonomy. (2026)

#### [Hyra：简单有效的科学发现智能体](https://hy.tencent.com/research/hyra)（2026）

**中文简介（来源：官方页面）：** Hyra-1.0 是 Hyra（Hunyuan Research Agent）的首个版本，是一个能够递归自我改进、专为性能导向的研究与工程任务打造的智能体。

**English title:** [Hyra: A simple yet effective scaffold for general discovery](https://hy.tencent.com/research/hyra)

**Description (Source: original list description):** Tencent Hunyuan report on Hyra-1.0, a recursively self-improving agent for performance-oriented research and engineering tasks, with produced research artifacts open-sourced in a companion repository. (2026)

#### [规模化 AI4AI：构建开放权重深度搜索智能体](https://xyz-lab.ai/blogs/ai4ai-at-scale/)（2026）

**中文简介（来源：官方页面）：** XYZ-Aquila 系列搜索智能体由一个有边界、设有验证关卡的 AI-for-AI 循环开发。人类设定目标和验收关卡，AI 智能体在限定范围内诊断失败，并修改数据、后训练、运行时、上下文管理和基础设施。

**English title:** [AI4AI at Scale: Building Open-Weight Deep Search Agents](https://xyz-lab.ai/blogs/ai4ai-at-scale/)

**Description (Source: original list description):** Industry technical report on a bounded, verification-gated AI-for-AI loop where agent teams diagnose failures and apply scoped changes across data, post-training, runtime, and infrastructure, with accepted and rejected attempts banked as auditable experience. (2026)

#### [迈向自动化 AI 研究的第一步](https://recursive.com/articles/first-steps-toward-automated-ai-research)（2026）

**中文简介（来源：官方页面导语）：** Recursive 公布其自动化 AI 研究系统在模型训练和 GPU 内核基准上的早期结果。系统围绕目标提出想法、实现方案、运行实验、验证结果，再根据所得经验选择下一次实验；它还能并行推进长期研究线程，保留既往实验的有效上下文，并在确认改进前检查奖励投机和结果方差。

**English title:** [First Steps Toward Automated AI Research](https://recursive.com/articles/first-steps-toward-automated-ai-research)

**Description (Source: original list description):** Describes an automated research system that closes the full loop from idea proposal through implementation, experiments, validation, and selection of the next experiment, with three reported SOTA results and open-sourced artifacts. (2026)

#### [面向自我改进的 Harness 工程](https://lilianweng.github.io/posts/2026-07-04-harness/)（2026）

**中文简介（来源：官方页面导语）：** 文章从 Harness 工程如何促成 RSI 出发，讨论工作流自动化、持久记忆和子智能体等设计模式，以及上下文与工作流优化、自改进 Harness、进化搜索和 Harness 与模型权重的联合优化。

**English title:** [Harness Engineering for Self-Improvement](https://lilianweng.github.io/posts/2026-07-04-harness/)

**Description (Source: original list description):** Survey of harness engineering as a near-term path to recursive self-improvement, covering harness design patterns, context and workflow optimization, self-improving harnesses, evolutionary search, and joint optimization with model weights. (2026)

#### [AlphaEvolve：由 Gemini 驱动、用于设计先进算法的编程智能体](https://deepmind.google/blog/alphaevolve-a-gemini-powered-coding-agent-for-designing-advanced-algorithms/)（2025）

**中文简介（来源：官方页面导语）：** AlphaEvolve 是面向通用算法发现与优化的进化式编程智能体。它把 Gemini 模型生成候选方案的能力与自动评估器结合起来，验证答案，并在进化框架中继续改进更有希望的想法。

**English title:** [AlphaEvolve: A Gemini-Powered Coding Agent for Designing Advanced Algorithms](https://deepmind.google/blog/alphaevolve-a-gemini-powered-coding-agent-for-designing-advanced-algorithms/)

**Description (Source: original list description):** Google DeepMind explains AlphaEvolve's evaluator-guided evolution loop and its applications to computing and AI training. (2025)

#### [达尔文哥德尔机：通过重写自身代码实现自我改进的 AI](https://sakana.ai/dgm/)（2025）

**中文简介（来源：官方页面）：** 达尔文哥德尔机以基础模型提出代码改进，并借助开放式算法搜索一个不断扩大的、多样化的高质量智能体档案。它用经验评测替代理论哥德尔机要求的事前数学证明，使自修改方案能够在编程任务上接受检验并继续演化。

**English title:** [The Darwin Gödel Machine: AI That Improves Itself by Rewriting Its Own Code](https://sakana.ai/dgm/)

**Description (Source: original list description):** Sakana AI explains DGM's empirical alternative to proof-based Gödel Machine self-modification. (2025)

#### [当 AI 构建自身](https://www.anthropic.com/institute/recursive-self-improvement)（2025）

**中文简介（来源：官方页面导语）：** Anthropic 讨论 AI 在 AI 开发流程中承担越来越多工作的现状，以及这一趋势通向完全自主设计后继系统的可能路径。文章同时指出，完整 RSI 尚未实现，也并非必然发生，并分析了技术证据、可能情景、失控风险和治理准备。

**English title:** [When AI Builds Itself](https://www.anthropic.com/institute/recursive-self-improvement)

**Description (Source: original list description):** Anthropic analyzes early evidence, possible paths, and governance challenges for AI-driven AI development. (2025)

#### [AI 科学家：迈向全自动、开放式科学发现](https://sakana.ai/ai-scientist/)（2024）

**中文简介（来源：官方页面导语）：** AI Scientist 面向机器学习研究，自动完成想法生成、文献检索、代码编写、实验执行、结果可视化、论文写作和自动评审。系统还会重复这一研究周期，把此前的想法和反馈用于后续开放式探索。

**English title:** [The AI Scientist: Towards Fully Automated Open-Ended Scientific Discovery](https://sakana.ai/ai-scientist/)

**Description (Source: original list description):** Sakana AI presents its automated research pipeline, results, limitations, and open implementation. (2024)

#### [当前机器学习对递归自我改进的证据](https://www.lesswrong.com/posts/byKF3mnaNRrbkDPWv/evidence-on-recursive-self-improvement-from-current-ml)（2023）

**中文简介（来源：原列表简介）：** 文章评述 AI 辅助 AI 研究能否带来强劲回报的正反两方面经验证据。

**English title:** [Evidence on Recursive Self-Improvement from Current ML](https://www.lesswrong.com/posts/byKF3mnaNRrbkDPWv/evidence-on-recursive-self-improvement-from-current-ml)

**Description (Source: original list description):** Reviews empirical evidence for and against strong returns from AI-assisted AI research. (2023)

#### [FunSearch：使用大语言模型在数学科学中取得新发现](https://deepmind.google/blog/funsearch-making-new-discoveries-in-mathematical-sciences-using-large-language-models/)（2023）

**中文简介（来源：官方页面导语）：** FunSearch 将预训练大语言模型与自动评估器配对，让模型用代码提出候选函数，评估器排除错误方案，再反复演化高分结果。官方页面介绍了它在帽集问题和在线装箱问题上得到的可验证发现。

**English title:** [FunSearch: Making New Discoveries in Mathematical Sciences Using Large Language Models](https://deepmind.google/blog/funsearch-making-new-discoveries-in-mathematical-sciences-using-large-language-models/)

**Description (Source: original list description):** Google DeepMind describes how evolutionary program search produced verifiable mathematical and algorithmic discoveries. (2023)

#### [元学习机器学会学习](https://people.idsia.ch/~juergen/metalearning.html)（2020）

**中文简介（来源：官方页面）：** Jürgen Schmidhuber 回顾了自 1987 年起的元学习或“学会学习”研究，包括自修改策略的元强化学习、基于梯度下降的神经网络元学习、面向课程学习的渐近最优元学习、哥德尔机所追求的数学最优 RSI，以及近年来与大语言模型相关的上下文学习和测试时训练。

**English title:** [Metalearning Machines Learn to Learn](https://people.idsia.ch/~juergen/metalearning.html)

**Description (Source: original list description):** Jürgen Schmidhuber traces self-referential meta-learning from 1987 through Gödel Machines and modern learned optimizers. (2020)

#### [递归自我改进](https://www.alignmentforum.org/w/recursive-self-improvement)（2016）

**中文简介（来源：原列表简介）：** Alignment Forum 的概述把自我改进 AI 与起飞动态、种子 AI 和控制问题联系起来。

**English title:** [Recursive Self-Improvement](https://www.alignmentforum.org/w/recursive-self-improvement)

**Description (Source: original list description):** The Alignment Forum overview connects self-improving AI to takeoff dynamics, seed AI, and control concerns. (2016)

## 演讲与视频 / Talks & Videos

#### [逃逸速度：递归自我改进的拐点](https://slideslive.com/39064188/escape-velocity-the-inflection-point-for-recursive-self-improvement)

**中文简介（来源：原列表简介）：** Louis Kirsch 在 ICLR 2026 RSI 研讨会上讨论自动化 AI 研究，以及持续递归改进所需的条件。（2026）

**English title:** [Escape Velocity: The Inflection Point for Recursive Self Improvement](https://slideslive.com/39064188/escape-velocity-the-inflection-point-for-recursive-self-improvement)

**Description (Source: original list description):** Louis Kirsch discusses automated AI research and the conditions required for sustained recursive improvement at the ICLR 2026 RSI workshop. (2026)

#### [ICLR 2026 AI 递归自我改进研讨会](https://iclr.cc/virtual/2026/workshop/10000796)

**中文简介（来源：原列表简介）：** 官方视频档案，收录专门围绕 RSI 的特邀演讲、投稿报告和专题讨论。（2026）

**English title:** [ICLR 2026 Workshop on AI with Recursive Self-Improvement](https://iclr.cc/virtual/2026/workshop/10000796)

**Description (Source: original list description):** Official video archive for invited talks, contributed work, and panels focused specifically on RSI. (2026)

#### [无需人工监督的自我改进基础模型](https://iclr.cc/virtual/2025/workshop/23971)

**中文简介（来源：原列表简介）：** ICLR 2025 官方研讨会录像，主题包括合成数据、弱到强学习和自主适应。（2025）

**English title:** [Self-Improving Foundation Models Without Human Supervision](https://iclr.cc/virtual/2025/workshop/23971)

**Description (Source: original list description):** Official ICLR 2025 workshop recordings on synthetic data, weak-to-strong learning, and autonomous adaptation. (2025)

#### [哥德尔机](https://www.youtube.com/watch?v=voczu4I3_xQ)

**中文简介（来源：原列表简介）：** Jürgen Schmidhuber 简要讲解自指、证明引导的代码重写及其可计算性限制。（2015）

**English title:** [Gödel Machine](https://www.youtube.com/watch?v=voczu4I3_xQ)

**Description (Source: original list description):** Jürgen Schmidhuber gives a concise explanation of self-referential, proof-guided code rewriting and its computability limits. (2015)

## 相关 Awesome 列表 / Related Awesome Lists

#### [Awesome AI 智能体](https://github.com/e2b-dev/awesome-ai-agents)

**中文简介（来源：原列表简介）：** 自主智能体项目与基础设施的综合目录，其中的项目可用作自我改进系统的组件或基线。（2023）

**English title:** [Awesome AI Agents](https://github.com/e2b-dev/awesome-ai-agents)

**Description (Source: original list description):** Broad directory of autonomous-agent projects and infrastructure that can serve as components or baselines for self-improving systems. (2023)

#### [Awesome AutoML 论文](https://github.com/hibayesian/awesome-automl-papers)

**中文简介（来源：原列表简介）：** 自动模型选择、架构搜索、超参数优化及相关技术的精选文献。（2018）

**English title:** [Awesome AutoML Papers](https://github.com/hibayesian/awesome-automl-papers)

**Description (Source: original list description):** Curated literature on automated model selection, architecture search, hyperparameter optimization, and related techniques. (2018)

#### [Awesome 自我改进智能体](https://github.com/selfimproving-agent/awesome-Self-Improving-Agents)

**中文简介（来源：原列表简介）：** 聚焦基础模型智能体的文献目录，这些智能体会更新自身的模型、记忆、工具、提示词或工作流。（2024）

**English title:** [Awesome Self-Improving Agents](https://github.com/selfimproving-agent/awesome-Self-Improving-Agents)

**Description (Source: original list description):** Focused bibliography of foundation-model agents that update their models, memory, tools, prompts, or workflows. (2024)

## 贡献 / Contributing

欢迎贡献。提交拉取请求前，请先阅读[贡献指南](CONTRIBUTING.md)。

Contributions are welcome. Please read the [contribution guidelines](CONTRIBUTING.md) before submitting a pull request.
