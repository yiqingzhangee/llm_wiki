# 理解 LLM 智能体的规划（Planning）：综述

Xu Huang¹，Weiwen Liu²，Xiaolong Chen¹，Xingmei Wang¹，Hao Wang¹，
Defu Lian¹*，Yasheng Wang²，Ruiming Tang²，Enhong Chen¹  
¹ 中国科学技术大学，合肥，中国  
² 华为诺亚方舟实验室，深圳，中国  
{xuhuangcs, chenxiaolong, xingmeiwang}@mail.ustc.edu.cn，{haowang, liandefu, cheneh}@ustc.edu.cn，{liuweiwen8, wangyasheng, tangruiming}@huawei.com

*Defu Lian 为通讯作者。

---

## 摘要（Abstract）

随着大型语言模型（Large Language Model, LLM）展现出显著的智能水平，如何将 LLM 作为自治智能体（autonomous agent）的规划（Planning）模块加以利用，已引起越来越多的关注。本文提供了关于基于 LLM 的智能体规划的首个系统性视角，覆盖了近期旨在提升规划能力的相关工作。我们对现有关于 LLM-智能体（LLM-Agent）规划的研究进行了分类，这些工作可以被归纳为任务分解（Task Decomposition）、规划选择（Plan Selection）、外部模块（External Module）、反思（Reflection）以及记忆（Memory）五大方向。针对每一方向，我们都进行了全面的分析，并讨论了该研究领域未来面临的挑战。

## 1 引言（Introduction）

自治智能体通常被视为能够完成特定任务的智能实体，其通过感知环境、进行规划并执行动作来达成目标。作为智能体最关键的能力之一，规划需要复杂的理解、推理与决策过程 [Ghallab et al., 2004]。

尽管“规划”的概念相对抽象，但规划任务的一般形式可以描述如下。给定时间步 $t$，环境记作 $E$，动作空间记作 $A$，任务目标记作 $g$，在时间步 $t$ 的动作记作 $a_t \in A$，则规划过程可以表示为一个动作序列的生成：

$$
 p = (a_0, a_1, \cdots, a_t) = \operatorname{plan}(E, g; \Theta, P).
$$

其中，$\Theta$ 和 $P$ 分别表示 LLM 的参数以及针对该任务的提示词（prompt）。

传统工作主要依赖符号方法或基于强化学习（Reinforcement Learning, RL）的方法，例如规划领域定义语言（Planning Domain Definition Language, PDDL）[Aeronautiques et al., 1998; Haslum et al., 2019]，或者策略学习（policy learning）[He et al., 2015; Yao et al., 2020a]。然而，这些传统方法存在若干局限性。符号方法需要将灵活的自然语言任务描述转化为符号化建模，这往往需要领域专家的人工投入；并且此类方法通常缺乏容错性，哪怕仅存在少量错误，也可能导致规划完全失败。强化学习方法通常会与深度模型结合，使其作为策略网络或回报（reward）模型。然而，RL 算法往往需要大量样本（与环境的交互）才能学习到有效策略，这在收集数据耗时或昂贵的场景中可能变得不切实际或成本过高。

近年来，大型语言模型（LLM）的出现标志着一种范式转变。LLM 在多个领域取得了卓越的成功，展现出在推理、工具使用、规划以及指令跟随等方面的显著智能。LLM 所表现出的惊人智能，为将 LLM 作为智能体的认知核心提供了可能，从而有望显著提升规划能力。围绕如何利用 LLM 进行智能体规划，已经涌现出大量方法。现有综述工作尝试从 LLM 技术 [Zhao et al., 2023a]、决策（decision-making）[Yang et al., 2023a]、推理（reasoning）[Sun et al., 2023]、工具学习（tool learning）[Qin et al., 2023] 以及自治智能体 [Wang et al., 2023a] 等方面进行总结，但通常缺乏对规划能力本身的系统分析。在本综述中，我们对最新研究进行梳理，分析其优势与局限，旨在为基于 LLM 的智能体规划能力提供一个系统化的视角。现有方法进一步被划分为五个具有代表性的方向，并对每个方向进行全面分析。此外，我们还在四个基准数据集上评估了若干具有代表性的方法。

从我们所知范围来看，这是首篇在规划能力角度上对基于 LLM 的智能体进行全面分析的工作。

规划能力（Planning Ability）

Selection（选择）  
Decomposition（分解）  
External Planner（外部规划器）  
Reflection（反思）  
Memory（记忆）  

代表性方法：CoT、ReAct、Visual ChatGPT、Plan-and-Solve、PoT、ProgPrompt、HuggingGPT。

图 1：LLM-智能体规划的分类（Taxonomy on LLM-Agent planning）。

LLM 具有深度模型的强大表示与推理能力，因此强化学习方法常常与深度模型结合，使深度模型作为策略网络或回报模型。然而，如前所述，RL 算法为了学习有效策略往往需要大量与环境交互的样本，这在数据采集耗时或昂贵的场景中可能并不可行。

在最近几年，大型语言模型（LLM）的出现带来了范式级的变革。LLM 在多个领域取得了显著成功，展现出在推理、工具使用、规划以及指令跟随等方面的强大智能。LLM 所展现的这种“意外的”智能启发了研究者将 LLM 作为智能体的认知核心，从而有潜力显著提升规划能力。围绕如何利用 LLM 进行智能体规划，已经涌现出诸多方法。尽管现有综述尝试对 LLM 技术 [Zhao et al., 2023a]、用于决策的 LLM [Yang et al., 2023a]、推理 [Sun et al., 2023]、工具学习 [Qin et al., 2023] 以及自治智能体 [Wang et al., 2023a] 等进行总结，但通常缺少对规划能力的细致分析。本文通过梳理最新研究工作，讨论其优点与限制，旨在为基于 LLM 的智能体规划能力提供一个系统视角。现有方法进一步被归纳为五个具有代表性的方向，并对每个方向进行了全面分析。此外，我们在四个基准上对若干代表性方法进行了评测。

从我们所知范围内来看，这是首篇在规划能力层面上对基于 LLM 的智能体进行全面分析的工作。

arXiv:2402.02716v1 [cs.AI] 5 Feb 2024

## 2 分类法（Taxonomy）

随着关于基于 LLM 的智能体规划能力的研究日益繁荣，众多方法被提出以挖掘规划能力的上限。为了更好地对现有先进工作进行全局把握，我们选取了部分具有代表性与影响力的研究，分析其动机与核心思想。为便于理解，我们在表 1 中给出了这些方法的分析。

基于表 1，我们提出了一个新的系统化分类法，将基于 LLM 的智能体规划（LLM-based agent planning）划分为五个重要类别，分别涵盖任务分解（task decomposition）、多规划方案选择（multi-plan selection）、外部模块辅助规划（external module-aided planning）、反思与改进（reflection and refinement）以及记忆增强规划（memory-augmented planning），如图 1 所示。下面我们对这五个方向进行简要概述。

**任务分解（Task Decomposition）。** 现实生活中的任务通常复杂且多步，这给规划带来了极大困难。这一类方法采用“分而治之”（divide and conquer）的思想，将复杂任务分解为若干子任务，然后依次对每个子任务进行规划。其过程可以形式化为：

$$
 g_0, g_1, \cdots, g_n = \operatorname{decompose}(E, g; \Theta, P);\\
 p_i = (a^i_0, a^i_1, \cdots, a^i_m) = \operatorname{sub\text{-}plan}(E, g_i; \Theta, P).
$$

**多规划方案选择（Multi-plan Selection）。** 这种方法关注如何引导 LLM “多想几种方案”，为同一任务生成多条候选规划路径。随后再使用与任务相关的搜索算法从中选出一个方案去执行。其形式化过程如下：

$$
 P = p_1, p_2, \cdots, p_n = \operatorname{plan}(E, g; \Theta, P);\\
 p^* = \operatorname{select}(E, g, P; \Theta, F).
$$

其中，$F$ 表示搜索策略，例如某些树搜索算法 [Yao et al., 2023; Zhao et al., 2023b]。

**外部规划器辅助规划（External Planner-Aided Planning）。** 这一类方法通过引入额外的外部规划器来提升规划过程，主要解决由 LLM 直接生成规划时面临的效率与可行性问题，而 LLM 主要在任务形式化方面发挥作用。其过程可以形式化为：

$$
 h = \operatorname{formalize}(E, g; \Theta, P);\\
 p = \operatorname{plan}(E, g, h; \Phi),
$$

其中，$\Phi$ 表示外部规划器模块，$h$ 表示形式化后的信息。

**反思与改进（Reflection and Refinement）。** 该方向强调通过反思与改进来提高规划能力。它鼓励 LLM 对失败进行反思，并据此对规划进行修正。其过程可形式化为：

$$
 p_0 = \operatorname{plan}(E, g; \Theta, P);\\
 r_i = \operatorname{reflect}(E, g, p_i; \Theta, P);\\
 p_{i+1} = \operatorname{refine}(E, g, p_i, r_i; \Theta, P).
$$

**记忆增强规划（Memory-augmented Planning）。** 这类方法为规划引入额外的记忆模块，在其中存储有价值的信息，例如常识（commonsense knowledge）、以往经验（past experiences）、领域知识（domain-specific knowledge）等。在规划过程中，这些信息会被检索出来，作为辅助信号使用。其过程可形式化为：

$$
 m = \operatorname{retrieve}(E, g; M);\\
 p = \operatorname{plan}(E, g, m; \Theta, P),
$$

其中，$M$ 代表记忆模块。

上述五个方向彼此并非相互排斥，而是相互关联，往往会同时采用多种技术。在接下来的章节中，我们将更深入地探讨这五个与 LLM-智能体规划相关的研究方向，阐明其研究动机，给出具有代表性的解决方案，并讨论其固有的局限性。

---

表 1：现有 LLM-智能体规划工作的分类（A taxonomy for existing LLM-Agent planning works）。

| 方法（Method） | 核心思想（Idea） | LLM 的任务（LLM’s task） | 形式化（Formulation） | 代表性工作（Representative works） |
| --- | --- | --- | --- | --- |
| 任务分解（Task Decomposition） | 分而治之（Divide and Conquer） | 任务分解（Task decomposition）; 子任务规划（Subtask planning） | $[g_i] = \operatorname{decompose}(E, g; \Theta, P);\\ p_i = \operatorname{sub\text{-}plan}(E, g_i; \Theta, P)$ | CoT [2022]，ReAct [2022]，HuggingGPT [2023] |
| 多规划方案选择（Multi-plan Selection） | 生成多个规划并选择最优方案（Generate multiple plans and select the optimal） | 规划生成（Plans generation）; 规划评估（Plans evaluation） | $P = p_1, p_2, \cdots, p_n = \operatorname{plan}(E, g; \Theta, P);\\ p^* = \operatorname{select}(E, g, P; \Theta, F)$ | ToT [2023]，GoT [2023]，CoT-SC [2022b] |
| 外部规划器辅助（External Planner-aided） | 形式化任务并利用外部规划器（Formalize tasks and utilize external planner） | 任务形式化（Task formalization） | $h = \operatorname{formalize}(E, g; \Theta, P);\\ p = \operatorname{plan}(E, g, h; \Phi)$ | LLM+P [2023a]，LLM+PDDL [2023] |
| 反思与改进（Reflection & Refinement） | 基于经验反思并改进规划（Reflect on experiences and refine plans） | 规划生成（Plan generation）; 反思（Reflection）; 改进（Refinement） | $p_0 = \operatorname{plan}(E, g; \Theta, P);\\ r_i = \operatorname{reflect}(E, g, p_i; \Theta, P);\\ p_{i+1} = \operatorname{refine}(E, g, p_i, r_i; \Theta, P)$ | Reflexion [2023]，CRITIC [2023]，Self-Refine [2023] |
| 记忆辅助规划（Memory-aided Planning） | 利用记忆辅助规划（Leverage memory to aid planning） | 规划生成（Plan generation）; 记忆提取（Memory extraction） | $m = \operatorname{retrieve}(E, g; M);\\ p = \operatorname{plan}(E, g, m; \Theta, P)$ | REMEMBER [2023a]，MemoryBank [2023] |

## 3 任务分解（Task Decomposition）

在现实世界场景中，环境往往具有复杂性与多变性，因此试图通过单步规划过程来解决复杂任务是一项极具挑战性的工作。

将复杂任务进行简化是人类的一项重要能力，这体现在将一个任务分解为若干简单子任务 [Schraagen et al., 2000]，这一思路与众所周知的“分而治之”（divide and conquer）算法策略类似，如式 (1) 所示。任务分解通常包含两个关键步骤：第一步是将复杂任务进行分解，即 “decompose” 步骤；第二步是针对各个子任务进行规划，即 “sub-plan” 步骤。在当前研究中，任务分解方法大体可以分为两类：先分解后规划（decomposition-first）和交替分解（interleaved decomposition），如图 2 所示。

### 3.1 先分解后规划方法（Decomposition-First Methods）

先分解后规划的方法会首先将任务分解为若干子目标（sub-goal），然后依次对每个子目标进行规划，如图 2(a) 所示。具有代表性的工作包括 HuggingGPT [Shen et al., 2023]、Plan-and-Solve [Wang et al., 2023b]、ProgPrompt [Singh et al., 2023] 等。

HuggingGPT [Shen et al., 2023] 利用 Huggingface Hub 中的多种多模态模型构建了一个能够处理多模态任务的智能体。它可以处理图像生成、图像分类、目标识别、视频标注、语音转文本等任务。为了促进不同模型之间的协作，LLM 被用作控制器（controller），负责对人类输入的任务进行分解、选择合适的模型并生成最终回答。其中最关键的阶段是初始的任务分解：HuggingGPT 通过明确的指令让 LLM 将给定任务拆解为若干子任务，并给出子任务之间的依赖关系。

Plan-and-Solve [Wang et al., 2023b] 在 Zero-shot Chain-of-Thought（零样本链式思维，Zero-shot CoT）[Kojima et al., 2022] 的基础上做出改进，将原本的提示“Let’s think step-by-step”转化为两步式提示：“Let’s first devise a plan”（我们先制定一个计划）和 “Let’s carry out the plan”（再执行该计划）。这一零样本方法在数学推理、常识推理和符号推理任务上都取得了提升。

ProgPrompt [Singh et al., 2023] 将任务的自然语言描述转化为编码问题（coding problem）。它通过代码来符号化智能体在环境中的动作空间和对象：每个动作被形式化为一个函数，而每个对象被表示为一个变量。因此，任务规划自然地转化为函数生成。在执行任务时，智能体首先以函数调用序列的形式生成一个计划，然后再逐步执行这些函数。

### 3.2 交替分解方法（Interleaved Decomposition Methods）

交替分解方法在任务分解与子任务规划之间交替进行，每次分解仅在当前状态下显式给出一个或两个子任务，如图 2(b) 所示。该类别的代表性方法包括 Chain-of-Thought（CoT）系列 [Wei et al., 2022; Kojima et al., 2022]、ReAct [Yao et al., 2022]、PAL [Gao et al., 2023]、Program-of-Thought（PoT）[Chen et al., 2022]、Visual ChatGPT [Wu et al., 2023] 等。

Chain-of-Thought（CoT）[Wei et al., 2022] 的提出揭示了 LLM 的小样本学习（few-shot learning）能力。CoT 通过在提示中构造若干推理轨迹，引导 LLM 对复杂问题进行逐步推理，从而利用其推理能力进行任务分解。

随后，Zero-shot CoT [Kojima et al., 2022] 通过一个“魔法”指令 “Let’s think step-by-step” 激活了 LLM 的零样本推理能力。与在规划过程中隐式嵌入推理过程的 CoT 不同，ReAct [Yao et al., 2022] 将“推理（Reasoning）”与“规划（Acting）”进行解耦。它在“思考（Thought）步骤”和“行动（Action）步骤”之间交替，显著提升了规划能力。

Visual ChatGPT [Wu et al., 2023] 采用 ReAct 的机制，将 LLM 作为智能体的大脑，并为其配备一系列视觉模型，从而构建出具有图像处理能力的智能体。

PAL [Gao et al., 2023] 则在 CoT 的基础上进一步利用了 LLM 的编码能力，引导 LLM 在推理过程中生成代码，再由代码解释器（如 Python）统一执行这些代码以得到最终解答。这种方法在解决数学和符号推理问题的智能体任务中表现良好。Program-of-Thought（PoT）[Chen et al., 2022] 则将整个推理过程完全形式化为编程。作者进一步使用在代码数据上训练的 CodeX [Chen et al., 2021b] 模型，从而在数学与金融问题上取得更优表现。

### 3.3 讨论（Discussions）

对于先分解后规划的方法，其优势在于可以更紧密地保持子任务与原始任务之间的相关性，从而降低任务遗忘与幻觉（hallucinations）[Touvron et al., 2023] 的风险。然而，由于子任务在开始时就已经全部确定，如果不额外引入调整机制，一旦某一步出现错误，就可能导致整体规划失败——我们将在第 6 节中进一步讨论这一点。相比之下，交替分解与子任务规划的方法会根据环境反馈动态调整分解过程，从而提高容错性。然而，对于复杂任务而言，过长的任务轨迹可能导致 LLM 出现幻觉，在后续的子任务与子规划中偏离原始目标。

虽然任务分解显著提升了 LLM-智能体在解决复杂任务方面的能力，但仍然存在若干挑战。首先，任务分解引入了额外开销。将一个任务拆分为多个子任务需要更多的推理与生成过程，从而带来额外的时间与计算成本。另一方面，对于被分解为数十个子任务的高度复杂任务，规划过程会受到 LLM 上下文长度的限制，导致对规划轨迹的遗忘。

## 4 多规划方案选择（Multi-Plan Selection）

由于任务本身的复杂性以及 LLM 的固有不确定性，LLM-智能体在面对同一个任务时可能生成多种不同的规划路径。即便 LLM 具有较强的推理能力，其一次性生成的单个规划方案也可能是次优甚至不可行的。一个更自然的思路是采用“多规划方案选择”（multi-plan selection），其通常由两个主要步骤构成：多方案生成与最优方案选择。

### 4.1 多方案生成（Multi-Plan Generation）

多方案生成是指为同一任务生成多条规划路径，以构成候选方案集合。主流方法通常考虑利用生成模型解码过程中固有的不确定性。

Self-consistency（自洽性）[Wang et al., 2022b] 采用了一个简单直观的思想：复杂问题的解往往并不唯一。与只生成单条推理路径的 CoT 不同，自洽性方法通过在解码过程中采用采样策略（如温度采样、top-k 采样）获取多条不同的推理路径。

Tree-of-Thought（ToT）[Yao et al., 2023] 提出了两种生成“思维”（thoughts，即计划）的策略：采样（sample）与提议（propose）。其中，采样策略与自洽性方法类似，即在解码过程中对 LLM 输出进行多次采样，从而获得多条规划路径；提议策略则通过在提示中提供少样本示例，显式地指示 LLM 为当前任务生成多种不同的规划方案。

Graph-of-Thought（GoT）[Besta et al., 2023] 在 ToT 的基础上加入了“思维变换”（transformations of thoughts），支持对任意“思维”的聚合。LLM-MCTS [Zhao et al., 2023b] 和 RAP [Hao et al., 2023] 则将 LLM 作为蒙特卡洛树搜索（Monte Carlo Tree Search, MCTS）的启发式策略函数（heuristic policy function），通过多次调用获得多个潜在动作。

### 4.2 最优方案选择（Optimal Plan Selection）

为了从候选方案中选出最优规划，需要采用多种策略作为启发式搜索算法。

Self-consistency [Wang et al., 2022b] 采用了朴素的多数表决策略（majority vote），将获得最多票数的规划视为最优方案。受益于树结构，Tree-of-Thought（ToT）[Yao et al., 2023] 支持广度优先搜索（BFS）和深度优先搜索（DFS）等树搜索算法。在选择要扩展的节点时，它使用 LLM 对多个动作进行评估并选择最优动作。

与 ToT 类似，LLM-MCTS [Zhao et al., 2023b] 和 RAP [Hao et al., 2023] 也利用树结构来辅助多方案搜索。但与 ToT 不同，它们采用了蒙特卡洛树搜索（MCTS）算法进行搜索。LLM A* [Xiao and Wang, 2023] 则利用经典的 A* 搜索算法来辅助 LLM 进行搜索，其中从当前状态到目标位置的切比雪夫距离被用作选择最优路径的启发式代价函数。

### 4.3 讨论（Discussions）

多规划方案选择在可扩展性方面具有显著优势，因为它能够在巨大的搜索空间中对潜在解进行更广泛的探索。然而，这种优势也伴随着内在的折中。一方面，对于参数规模大、单次调用计算与 token 开销高的模型而言，激增的计算需求带来了实际应用上的挑战；在资源受限的场景（例如在线服务）中，这一成本问题尤为突出。另一方面，依赖 LLM 对规划方案进行评估也引入了新的问题。由于 LLM 在排序（ranking）任务上的性能尚未得到充分验证，因此在这一特定场景下仍需要进一步的验证与微调。此外，LLM 的随机性也会为方案选择引入额外的随机因素，从而可能影响所选规划方案的一致性与可靠性。

## 5 外部规划器辅助规划（External Planner-Aided Planning）

尽管大型语言模型（LLM）在推理与任务分解方面展现出强大的能力，但在面对具有复杂约束的环境（例如数学问题求解或生成满足约束的可执行动作）时仍然存在挑战。为了解决这些问题，一些方法将 LLM 与外部规划器进行结合。根据所引入规划器的类型，这类方法可以分为符号规划器（symbolic planners）与神经规划器（neural planners）。

### 5.1 符号规划器（Symbolic Planner）

符号规划器在自动规划（automated planning）领域已经作为基础组件存在了数十年。这类方法基于成熟的符号形式化模型，例如 PDDL 模型 [Aeronautiques et al., 1998; Haslum et al., 2019]，通过符号推理从初始状态到目标状态寻找最优路径。

LLM+P [Liu et al., 2023a] 通过引入基于 PDDL 的符号规划器来增强 LLM 的规划能力。作者利用 LLM 在语义理解与代码生成方面的能力，将环境中的动作以及指定任务组织成文本提示（prompt），输入给 LLM，以指导其将这些信息组织为 PDDL 语言格式。随后，在获得形式化描述之后，作者使用 Fast-Downward¹ 求解器进行规划过程。

在 LLM+P 的基础上，LLM-DP [Dagan et al., 2023] 被专门设计用于动态交互式环境。智能体在接收到来自环境的反馈后，LLM 对这些信息进行处理，并将其形式化为 PDDL 语言，然后使用基于 BFS 的求解器 [Lipovetzky et al., 2014] 生成规划。

LLM+PDDL [Guan et al., 2023] 也利用 PDDL 语言进行任务形式化，并额外引入了一步人工验证，以检查 LLM 生成的 PDDL 模型中可能存在的问题。在规划过程中，作者提出将 LLM 生成的规划作为局部搜索规划器（如 LPG [Gerevini and Serina, 2002]）的初始启发式解，以加速搜索过程。LLM+ASP [Yang et al., 2023b] 则将 LLM 生成的自然语言问题转换为原子事实（atomic facts），从而将任务转化为 ASP 问题，随后使用 ASP 求解器 CLINGO 生成规划。

¹ https://github.com/aibasel/downward/tree/release-22.12.0

### 5.2 神经规划器（Neural Planner）

神经规划器通常是基于收集到的规划数据，利用强化学习或模仿学习训练得到的深度模型，在特定领域中表现出有效的规划能力。例如，DRRN [He et al., 2015] 将规划过程建模为一个马尔可夫决策过程（Markov Decision Process），并通过强化学习训练策略网络，从而获得一个深度决策模型。Decision Transformer（决策 Transformer, DT）[Chen et al., 2021a] 则借助规划数据，让 Transformer 模型模仿人类的决策行为。

训练良好的神经规划器在各自的领域内往往具有出色的规划能力，并且由于参数规模较小而在规划效率上具有优势。然而，当面对复杂且少见的问题（这类问题的训练数据稀缺）时，这些小模型由于泛化能力不足，往往表现较差。因此，一些工作探索将 LLM 与轻量级神经规划器相结合，以进一步提升规划能力。CALM [Yao et al., 2020a] 是一种较早提出的方法，它将语言模型与基于强化学习的神经规划器结合在一起：首先使用语言模型处理文本化的环境信息，并基于环境生成一组候选动作先验（priors），然后使用 DRRN 策略网络对这些候选动作进行重排序，最终选出最优动作。

SwiftSage [Lin et al., 2023] 借鉴认知心理学中的“双系统理论”（dual-process theory），将规划过程划分为慢思考（slow thinking）与快思考（fast thinking）。其中，慢思考负责复杂推理与理性思考，而快思考则类似通过长期训练形成的本能反应。作者使用通过模仿学习训练的 Decision Transformer 模型作为快思考模型，用于快速生成规划。当在计划执行过程中出现错误、表明问题更为复杂时，智能体切换到慢思考过程，即由 LLM 基于当前状态进行推理与规划。事实表明，这种快慢结合的方式在效率方面非常有效。

### 5.3 讨论（Discussions）

在这些引入额外规划器以提供辅助的策略中，LLM 主要扮演辅助角色。其主要功能包括解析文本反馈，并在解决复杂问题时提供额外的推理信息。尤其是，LLM 在代码生成能力上的提升，使其有潜力处理更普遍的符号人工智能（symbolic AI）任务。事实上，传统符号 AI 系统的一个重要缺点在于其构建符号模型过程复杂，且高度依赖人类专家；而 LLM 能够加速这一过程，帮助更快速、更优地建立符号模型。符号系统带来的优势包括理论上的完备性、稳定性以及可解释性。将统计式 AI 与 LLM 结合，被认为是未来人工智能发展的重要趋势之一。

## 6 反思与改进（Reflection and Refinement）

反思与改进是规划过程中的关键组成部分，它们可以提升 LLM-智能体规划的容错与纠错能力。由于当前 LLM 仍存在幻觉问题以及在处理复杂问题时推理能力不足，LLM-智能体在规划过程中可能犯错，并且在反馈有限的情况下陷入“思维循环”（thought loops）。通过对失败进行反思与总结，有助于智能体在后续尝试中纠正错误，并跳出这类循环。

Self-Refine [Madaan et al., 2023] 采用了“生成—反馈—改进”的迭代过程。每次生成规划后，LLM 会针对当前规划产生反馈，从而在下一轮根据反馈对规划进行调整。Reflexion [Shinn et al., 2023] 在 ReAct 的基础上引入了一个评估模块，用于评估轨迹。当检测到错误时，LLM 会生成自我反思内容，从而辅助纠错。CRITIC [Gou et al., 2023] 利用外部工具（如知识库和搜索引擎）来验证 LLM 生成的动作，并基于外部知识进行自我纠错，从而显著减少事实性错误。

InteRecAgent [Huang et al., 2023b] 采用了名为 ReChain 的机制进行自我纠错。该方法使用一个 LLM 对交互式推荐智能体生成的响应及其工具使用规划进行评估，对错误进行总结并给出反馈，并决定是否重新启动规划。LEMA [An et al., 2023] 首先收集错误规划样本，然后使用更强大的 GPT-4 对其进行修正，再利用这些修正后的样本对 LLM-智能体进行微调，在不同规模的 LLaMA 模型上都获得了显著的性能提升。

特别地，这种自我反思策略在一定程度上与强化学习的原理相似：其中智能体扮演决策者（如策略网络）的角色，而环境反馈则触发对策略网络的更新。不过，与深度强化学习通过修改模型参数进行更新不同，在 LLM-智能体中，这种更新是通过 LLM 本身的自我反思实现的，最终体现为文本形式的语言反馈。这些文本反馈既可以作为长期记忆，也可以作为短期记忆，通过提示词对智能体后续的规划输出产生影响。然而，这种“文本形式更新”的收敛性目前尚缺乏理论保证，也就是说，尚无法证明持续的反思一定会将 LLM-智能体引导至某个指定目标。

## 7 记忆增强规划（Memory-Augmented Planning）

对于智能体而言，记忆是提升规划能力与成长潜力的关键途径之一。在 LLM-智能体的记忆机制方面，目前主要有两种增强规划能力的路径：基于 RAG 的记忆（RAG-based memory）与具身记忆（embodied memory）。

### 7.1 基于 RAG 的记忆（RAG-based Memory）

检索增强生成（Retrieval Augmented Generation, RAG）[Lewis et al., 2020; Mao et al., 2020; Cai et al., 2022] 技术通过引入检索到的信息来辅助文本生成。它可以增强 LLM 对最新知识的获取，例如 New Bing² 和 Google Bard³ 等系统。

对于 LLM 智能体来说，过往经验可以存储在记忆中，在需要时进行检索。这类方法的核心思想是在任务规划过程中从记忆中检索与当前任务相关的经验。在这些方法中，记忆通常存储在额外的存储结构中，形式多样，如文本 [Park et al., 2023; Liu et al., 2023b; Packer et al., 2023; Wang et al., 2023c; Zhong et al., 2023]、表格 [Zhang et al., 2023a]、知识图谱 [Pan et al., 2024] 等。

Generative Agents [Park et al., 2023] 将类人智能体的日常经验以文本形式存储，并基于“新近性”（recency）与当前情境相关性（relevance）的综合得分来检索记忆。类似地，MemoryBank [Zhong et al., 2023]、TiM [Liu et al., 2023b] 和 RecMind [Wang et al., 2023c] 使用文本编码模型将每条记忆编码为向量，并构建索引结构（如 FAISS 库 [Johnson et al., 2019]）。在检索阶段，会将当前状态的描述作为查询，从记忆池中检索相关记忆。三者的差异主要在于记忆更新方式。

MemGPT [Packer et al., 2023] 借鉴了计算机体系结构中多级存储的概念，将 LLM 的上下文抽象为内存（RAM），而将额外的存储结构视为磁盘（disk）。LLM 可以自主决定是否从历史记忆中检索信息，或者将当前上下文写入外部存储。REMEMBER [Zhang et al., 2023a] 则以 Q 值表（Q-value table）的形式存储历史记忆，其中每条记录都是一个四元组 $(\text{environment}, \text{task}, \text{action}, \text{Q-value})$。在检索阶段，会同时检索正向与负向记忆，以环境与任务的相似性为依据，让 LLM 在生成规划时参考这些记忆。

² https://www.bing.com/  
³ https://bard.google.com

### 7.2 具身记忆（Embodied Memory）

具身记忆是指使用智能体的历史交互样本对 LLM 进行微调（finetuning），将记忆“嵌入”到模型参数之中。通常，这些经验样本来源于智能体与环境的交互，包含关于环境的常识、与任务相关的先验知识，以及成功或失败的经验。由于对参数规模在数十亿量级以上的语言模型进行训练成本巨大，参数高效微调（Parameter-Efficient Fine-Tuning, PEFT）技术被广泛用于降低成本和加速训练，仅对少量参数进行更新，如 LoRA、QLoRA、P-tuning 等。

CALM [Yao et al., 2020b] 使用从文本世界环境中收集的真实动作轨迹（ground-truth action trajectories），在下一 token 预测任务上微调 GPT-2，使其能够记忆与规划相关的信息，并在规划任务上具有良好的泛化能力。类似地，TDT [Wang et al., 2022a] 使用收集到的马尔可夫决策过程数据对 Text Decision Transformer（TDT）进行微调，在更具挑战性的 ScienceWorld [Wang et al., 2022a] 任务上取得了更高的成功率。

AgentTuning [Zeng et al., 2023] 将来自各种任务的规划轨迹组织成对话形式，用于微调 LLaMA 模型，在未见规划任务上也展现出显著性能提升。

### 7.3 讨论（Discussions）

基于 RAG 的记忆与基于微调的记忆这两种路径都可以增强 LLM-智能体的规划能力，但各自具有不同的优缺点。RAG 方法通过外部存储提供实时、低成本的记忆更新，记忆形式主要为自然语言文本，但其效果依赖于检索算法的准确性。微调方式则通过修改模型参数提供更大的记忆容量，但在更新记忆时代价较高，并且难以保留细粒度细节。

总体而言，记忆增强的 LLM-智能体在规划方面表现出更强的成长性与容错性，但记忆本身的生成高度依赖于 LLM 的生成能力。如何通过智能体自我生成的记忆进一步提升较弱的 LLM-智能体，仍然是一个值得深入探索的挑战方向。

## 8 评估（Evaluation）

评估智能体的规划能力是该研究领域中的关键问题。本节我们考察若干主流基准评测方法，并将其大致划分为以下几类。

**交互式游戏环境（Interactive Gaming Environments）。** 此类游戏环境能够根据智能体的动作提供实时的多模态反馈，包括文本反馈与视觉反馈。目前最广泛使用的游戏环境之一是 Minecraft⁴，在这一环境中，智能体需要收集材料来制作工具以获得更多奖励，通常会以智能体制作的工具数量作为评估指标。另一类流行环境是文本交互式环境（text-based interactive environments），如 ALFWorld [Shridhar et al., 2020]、ScienceWorld [Wang et al., 2022a] 等，在这些环境中，智能体位于自然语言描述的世界中，可用的动作与位置有限，常用的评估指标是成功率或获得的奖励。与 Minecraft 相比，这些文本交互式环境通常更为简单，反馈更加直接，可执行动作也更少。

**交互式检索环境（Interactive Retrieval Environments）。** 这类环境模拟了现实生活中人类进行信息检索与推理的过程。在这类环境中，智能体通常被允许与搜索引擎及其他 Web 服务交互，例如使用“搜索关键词”（searching keywords）或执行点击（click）、前进（forward）、后退（backward）等操作以获取更多信息，从而回答问题或完成信息检索任务。常用的检索环境包括基于 Wikipedia 检索引擎的问题回答任务 [Yao et al., 2022]（如 HotPotQA [Yang et al., 2018] 和 FEVER [Thorne et al., 2018]），以及面向获取特定信息的 Web 浏览任务，如 WebShop、Mind2Web [Deng et al., 2023] 和 WebArena [Zhou et al., 2023]。任务成功率通常被用作评估指标。

**交互式编程环境（Interactive Programming Environments）。** 交互式编程环境模拟程序员与计算机的交互过程，用于测试智能体在解决计算机相关问题时的规划能力。在此类环境中，智能体需要通过编写代码或指令与计算机交互以解决问题，并会接收到多种反馈，如编译错误与运行时错误信息以及执行结果。流行的交互式编程环境包括涉及操作系统、数据库等问题的 AgentBench [Liu et al., 2023c]、MiniWoB++ [Kim and others, 2023] 等。

大多数现有的交互式环境在评估上都比较粗粒度，通常只依赖最终成功率作为性能指标。与现实场景中“一个任务往往存在多条路径可以完成”不同，由于标注成本高昂，大多数模拟环境中往往只有一条“黄金路径”（golden path）。

**实验（Experiments）。** 我们在四个基准上进行了实验，以验证若干具有代表性方法的性能，如表 2 所示。受预算限制，我们实现了六种基于 prompt 的方法，覆盖了任务分解、多路径选择与反思等三类技术。在基准方面，我们选择了 ALFWorld、ScienceWorld、HotPotQA 与 FEVER，分别涵盖交互式游戏与问答类基准。

由于 ALFWorld 与 ScienceWorld 中动作空间较大，零样本方法（ZeroShot-CoT）在该类环境下不适用，因为其对动作空间并不了解。SayCan 通过引入值函数（value function）将 CoT 输出的动作“落地”到动作空间中，从而对 CoT 进行了改进，但这种方法并不适用于问答任务，因为问答任务中只有两类动作：`SEARCH[KEYWORD]` 与 `LOOKUP[KEYWORD]`。在实验中，我们将值函数设置为文本嵌入模型 bge-small-en-v1.5 [Xiao and others, 2023]。在 CoT-SC 中，我们在游戏任务与问答任务中分别为每一步获取 3 个动作与 5 个答案。Reflexion 中的重试轮数设置为 1。作为 LLM，我们使用了 OpenAI 提供的 `text-davinci-003` 接口。

(i) **性能随开销增加而提升。** 由于 CoT-SC、ReAct 与 Reflexion 分别涉及多规划方案、额外的思考步骤与反思步骤，它们的开销都高于各自的基础方法。直观地讲，更多的 token 代表更详细的思考过程，从而在性能上带来提升。

(ii) **对复杂任务而言，建议使用少样本示例。** 尽管“Let’s think step by step” 这一“魔法”指令能够引导 LLM 进行更多推理，但在两个问答基准上，ZeroShot-CoT 的性能明显下降，这表明示例对于帮助 LLM 理解任务仍然是必要的。

(iii) **反思对提升成功率尤为关键，尤其是在复杂任务中。** 尽管 Reflexion 在 token 消耗上约为 ReAct 的两倍，但在复杂任务（如 ALFWorld 和 ScienceWorld）上的性能提升非常明显，这说明 LLM 具备一定的自我纠错能力。

表 2：若干代表性基于提示（prompt-based）方法在四个交互式基准上的评估结果。SR、AR 与 EX 分别表示成功率（success rate）、平均回报（average rewards）以及开销（expenses）。开销通过 OpenAI API 消耗的 token 数量计算。Z-CoT 与 F-CoT 分别表示 ZeroShot-CoT 和 FewShot-CoT。

| 方法 | \multicolumn{2}{c|}{AlfWorld} | \multicolumn{2}{c|}{ScienceWorld} | \multicolumn{2}{c|}{HotPotQA} | \multicolumn{2}{c|}{FEVER} |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 指标（Metrics） | SR(%) | EX($) | AR | EX($) | SR(%) | EX($) | SR(%) | EX($) |
| Z-CoT | N/A | N/A | N/A | N/A | 0.01 | 0.95 | 0.39 | 1.07 |
| F-CoT | 0.43 | 98.60 | 16.58 | 272.22 | 0.32 | 5.73 | 0.61 | 2.25 |
| CoT-SC | 0.57 | 105.37 | 15.24 | 274.33 | 0.33 | 7.86 | 0.62 | 3.21 |
| SayCan | 0.60 | 113.61 | 12.36 | 125.71 | N/A | N/A | N/A | N/A |
| ReAct | 0.57 | 152.18 | 15.05 | 356.03 | 0.34 | 66.00 | 0.63 | 22.20 |
| Reflexion | 0.71 | 220.17 | 19.39 | 724.48 | 0.39 | 112.49 | 0.68 | 37.26 |

⁴ https://www.minecraft.net/

## 9 结论与未来方向（Conclusions and Future Directions）

自从 LLM 展现出“智能涌现”（emergence of intelligence）以来，如何利用 LLM 提升智能体的规划能力受到越来越多的关注。我们在图 1 中对主要研究方向进行了总结，并在第 3 至第 7 节对各类方法进行了详细对比与分析。同时，我们在四个基准上进行了实验，对若干代表性方法的有效性进行了比较，结果表明性能通常随开销增加而提升。尽管现有工作在规划能力方面取得了一定增强，但仍然存在若干重要挑战。

**幻觉（Hallucinations）。** 在规划过程中，LLM 经常受到幻觉问题的影响，从而给出不合理的规划、偏离任务提示，或无法遵循复杂指令。例如，规划中可能包含与环境中不存在的物品进行交互的动作。尽管可以通过精心设计 prompt 在一定程度上缓解这些问题，但它们反映了 LLM 的根本缺陷 [Zhang et al., 2023b; Huang et al., 2023a]。

**生成规划的可行性（Feasibility of Generated Plans）。** LLM 从本质上说是一种基于统计学习的模型，其通过大量数据优化“下一个词”的概率。与符号人工智能（symbolic AI）相比，这种方法在遵循复杂约束方面存在困难，特别是对于训练过程中很少出现的复杂约束。因此，如果不充分考虑前置条件，LLM 生成的规划可能缺乏可行性。如何在不改变 LLM 本身的前提下，将其与符号规划模型结合，是一个颇具前景的研究方向。

**生成规划的效率（Efficiency of Generated Plans）。** 规划效率是规划问题中的关键议题之一。然而，在现有 LLM-智能体中，规划往往是贪婪地基于 LLM 输出的方案进行，而很少显式考虑规划方案的效率。因此，在未来的发展中，可能需要引入额外的效率评估模块，与 LLM 协同工作，以生成更高效的规划。

**多模态环境反馈（Multi-Modal Environment Feedback）。** LLM 最初是为了处理纯文本输入而设计的，而现实世界的环境反馈往往是多模态的，包括图像、音频等，而这些信息难以完整地用自然语言描述。因此，LLM-智能体在处理此类场景时存在一定局限。未来的工作可以考虑结合多模态大模型的发展，重新审视与多模态反馈相关的规划策略。

**细粒度评估（Fine-grained Evaluation）。** 如第 8 节所述，现有基准大多依赖任务最终完成情况进行评估，缺乏细粒度的逐步评估。同时，环境反馈往往是基于规则的，较为简单，与真实世界场景存在较大差距。未来一个潜在方向是利用像 LLM 这样高智能的模型构建更加真实的评估环境。

## 参考文献（References）

[Aeronautiques et al., 1998] Constructions Aeronautiques, Adele Howe, et al. PDDL— the planning domain definition language. Technical Report, Tech. Rep., 1998.

[An et al., 2023] Shengnan An, Zexiong Ma, et al. Learning from mistakes makes LLM better reasoner. arXiv preprint arXiv:2310.20689, 2023.

[Besta et al., 2023] Maciej Besta, Nils Blach, et al. Graph of thoughts: Solving elaborate problems with large language models. arXiv preprint arXiv:2308.09687, 2023.

[Cai et al., 2022] Deng Cai, Yan Wang, Lemao Liu, and Shuming Shi. Recent advances in retrieval-augmented text generation. In SIGIR, pages 3417–3419, 2022.

[Chen et al., 2021a] Lili Chen, Kevin Lu, et al. Decision transformer: Reinforcement learning via sequence modeling. NeurIPS, 34:15084–15097, 2021.

[Chen et al., 2021b] Mark Chen, Jerry Tworek, et al. Evaluating large language models trained on code. arXiv preprint arXiv:2107.03374, 2021.

[Chen et al., 2022] Wenhu Chen, Xueguang Ma, et al. Program of thoughts prompting: Disentangling computation from reasoning for numerical reasoning tasks. arXiv preprint arXiv:2211.12588, 2022.

[Dagan et al., 2023] Gautier Dagan, Frank Keller, and Alex Lascarides. Dynamic planning with a LLM. arXiv preprint arXiv:2308.06391, 2023.

[Deng et al., 2023] Xiang Deng, Yu Gu, et al. Mind2Web: Towards a generalist agent for the web. arXiv preprint arXiv:2306.06070, 2023.

[Gao et al., 2023] Luyu Gao, Aman Madaan, et al. PAL: Program-aided language models. In ICML, pages 10764–10799, 2023.

[Gerevini and Serina, 2002] Alfonso Gerevini and Ivan Serina. LPG: A planner based on local search for planning graphs with action costs. In AIPS, volume 2, pages 281–290, 2002.

[Ghallab et al., 2004] Malik Ghallab, Dana Nau, et al. Automated Planning: Theory and Practice. Elsevier, 2004.

[Gou et al., 2023] Zhibin Gou, Zhihong Shao, et al. CRITIC: Large language models can self-correct with tool-interactive critiquing. arXiv preprint arXiv:2305.11738, 2023.

[Guan et al., 2023] Lin Guan, Karthik Valmeekam, et al. Leveraging pre-trained large language models to construct and utilize world models for model-based task planning. arXiv preprint arXiv:2305.14909, 2023.

[Hao et al., 2023] Shibo Hao, Yi Gu, et al. Reasoning with language model is planning with world model. arXiv preprint arXiv:2305.14992, 2023.

[Haslum et al., 2019] Patrik Haslum, Nir Lipovetzky, et al. An introduction to the planning domain definition language, volume 13. Springer, 2019.

[He et al., 2015] Ji He, Jianshu Chen, et al. Deep reinforcement learning with a natural language action space. arXiv preprint arXiv:1511.04636, 2015.

[Huang et al., 2023a] Lei Huang, Yu Weijiang, et al. A survey on hallucination in large language models: Principles, taxonomy, challenges, and open questions. arXiv preprint arXiv:2311.05232, 2023.

[Huang et al., 2023b] Xu Huang, Jianxun Lian, et al. Recommender AI agent: Integrating large language models for interactive recommendations. arXiv preprint arXiv:2308.16505, 2023.

[Johnson et al., 2019] Jeff Johnson, Matthijs Douze, and Hervé Jégou. Billion-scale similarity search with GPUs. IEEE Transactions on Big Data, 7(3):535–547, 2019.

[Kim and others, 2023] Geunwoo Kim et al. Language models can solve computer tasks. arXiv preprint arXiv:2303.17491, 2023.

[Kojima et al., 2022] Takeshi Kojima, Shixiang Shane Gu, et al. Large language models are zero-shot reasoners. NeurIPS, 35:22199–22213, 2022.

[Lewis et al., 2020] Patrick Lewis, Ethan Perez, et al. Retrieval-augmented generation for knowledge-intensive NLP tasks. NeurIPS, 33:9459–9474, 2020.

[Lin et al., 2023] Bill Yuchen Lin, Yicheng Fu, et al. SwiftSage: A generative agent with fast and slow thinking for complex interactive tasks. arXiv preprint arXiv:2305.17390, 2023.

[Lipovetzky et al., 2014] Nir Lipovetzky, Miquel Ramirez, et al. Width and inference based planners: SIW, BFS(f), and Probe. IPC, page 43, 2014.

[Liu et al., 2023a] Bo Liu, Yuqian Jiang, et al. LLM+P: Empowering large language models with optimal planning proficiency. arXiv preprint arXiv:2304.11477, 2023.

[Liu et al., 2023b] Lei Liu, Xiaoyan Yang, et al. Think-in-memory: Recalling and post-thinking enable LLMs with long-term memory. arXiv preprint arXiv:2311.08719, 2023.

[Liu et al., 2023c] Xiao Liu, Hao Yu, et al. AgentBench: Evaluating LLMs as agents. arXiv preprint arXiv:2308.03688, 2023.

[Madaan et al., 2023] Aman Madaan, Niket Tandon, et al. Self-refine: Iterative refinement with self-feedback. arXiv preprint arXiv:2303.17651, 2023.

[Mao et al., 2020] Yuning Mao, Pengcheng He, Liu, et al. Generation-augmented retrieval for open-domain question answering. arXiv preprint arXiv:2009.08553, 2020.

[Packer et al., 2023] Charles Packer, Vivian Fang, et al. MemGPT: Towards LLMs as operating systems. arXiv preprint arXiv:2310.08560, 2023.

[Pan et al., 2024] Shirui Pan, Linhao Luo, et al. Unifying large language models and knowledge graphs: A roadmap. TKDE, 2024.

[Park et al., 2023] Joon Sung Park, Joseph O’Brien, et al. Generative agents: Interactive simulacra of human behavior. In SUIST, pages 1–22, 2023.

[Qin et al., 2023] Yujia Qin, Shengding Hu, et al. Tool learning with foundation models. arXiv preprint arXiv:2304.08354, 2023.

[Schraagen et al., 2000] Jan Maarten Schraagen, Susan F. Chipman, et al. Cognitive task analysis. Psychology Press, 2000.

[Shen et al., 2023] Yongliang Shen, Kaitao Song, et al. HuggingGPT: Solving AI tasks with ChatGPT and its friends in Huggingface. arXiv preprint arXiv:2303.17580, 2023.

[Shinn et al., 2023] Noah Shinn, Federico Cassano, et al. Reflexion: Language agents with verbal reinforcement learning. In NeurIPS, 2023.

[Shridhar et al., 2020] Mohit Shridhar, Xingdi Yuan, et al. ALFWorld: Aligning text and embodied environments for interactive learning. arXiv preprint arXiv:2010.03768, 2020.

[Singh et al., 2023] Ishika Singh, Valts Blukis, et al. ProgPrompt: Generating situated robot task plans using large language models. In ICRA 2023, pages 11523–11530. IEEE, 2023.

[Sun et al., 2023] Jiankai Sun, Chuanyang Zheng, et al. A survey of reasoning with foundation models. arXiv preprint arXiv:2312.11562, 2023.

[Thorne et al., 2018] James Thorne, Andreas Vlachos, et al. FEVER: A large-scale dataset for fact extraction and verification. arXiv preprint arXiv:1803.05355, 2018.

[Touvron et al., 2023] Hugo Touvron, Louis Martin, et al. LLaMA 2: Open foundation and fine-tuned chat models. arXiv preprint arXiv:2307.09288, 2023.

[Wang et al., 2022a] Ruoyao Wang, Peter Jansen, et al. ScienceWorld: Is your agent smarter than a 5th grader? arXiv preprint arXiv:2203.07540, 2022.

[Wang et al., 2022b] Xuezhi Wang, Jason Wei, et al. Self-consistency improves chain of thought reasoning in language models. arXiv preprint arXiv:2203.11171, 2022.

[Wang et al., 2023a] Lei Wang, Chen Ma, et al. A survey on large language model based autonomous agents. arXiv preprint arXiv:2308.11432, 2023.

[Wang et al., 2023b] Lei Wang, Wanyu Xu, et al. Plan-and-solve prompting: Improving zero-shot chain-of-thought reasoning by large language models. arXiv preprint arXiv:2305.04091, 2023.

[Wang et al., 2023c] Yancheng Wang, Ziyan Jiang, et al. RecMind: Large language model powered agent for recommendation. arXiv preprint arXiv:2308.14296, 2023.

[Wei et al., 2022] Jason Wei, Xuezhi Wang, et al. Chain-of-thought prompting elicits reasoning in large language models. NeurIPS, 35:24824–24837, 2022.

[Wu et al., 2023] Chenfei Wu, Shengming Yin, et al. Visual ChatGPT: Talking, drawing and editing with visual foundation models. arXiv preprint arXiv:2303.04671, 2023.

[Xiao and others, 2023] Shitao Xiao et al. C-Pack: Packaged resources to advance general Chinese embedding, 2023.

[Xiao and Wang, 2023] Hengjia Xiao and Peng Wang. LLM A*: Human in the loop large language models enabled A* search for robotics. arXiv preprint arXiv:2312.01797, 2023.

[Yang et al., 2018] Zhilin Yang, Peng Qi, et al. HotPotQA: A dataset for diverse, explainable multi-hop question answering. arXiv preprint arXiv:1809.09600, 2018.

[Yang et al., 2023a] Sherry Yang, Nachum Ofir, et al. Foundation models for decision making: Problems, methods, and opportunities. arXiv preprint arXiv:2303.04129, 2023.

[Yang et al., 2023b] Zhun Yang, Adam Ishay, and Joohyung Lee. Coupling large language models with logic programming for robust and general reasoning from text. arXiv preprint arXiv:2307.07696, 2023.

[Yao et al., 2020a] Shunyu Yao, Rohan Rao, et al. Keep CALM and explore: Language models for action generation in text-based games. arXiv preprint arXiv:2010.02903, 2020.

[Yao et al., 2020b] Shunyu Yao, Rohan Rao, et al. Keep CALM and explore: Language models for action generation in text-based games. arXiv preprint arXiv:2010.02903, 2020.

[Yao et al., 2022] Shunyu Yao, Jeffrey Zhao, et al. ReAct: Synergizing reasoning and acting in language models. arXiv preprint arXiv:2210.03629, 2022.

[Yao et al., 2023] Shunyu Yao, Dian Yu, et al. Tree of Thoughts: Deliberate problem solving with large language models. arXiv preprint arXiv:2305.10601, 2023.

[Zeng et al., 2023] Aohan Zeng, Mingdao Liu, et al. AgentTuning: Enabling generalized agent abilities for LLMs. arXiv preprint arXiv:2310.12823, 2023.

[Zhang et al., 2023a] Danyang Zhang, Lu Chen, et al. Large language model is semi-parametric reinforcement learning agent. arXiv preprint arXiv:2306.07929, 2023.

[Zhang et al., 2023b] Yue Zhang, Yafu Li, et al. Siren’s song in the AI ocean: A survey on hallucination in large language models. arXiv preprint arXiv:2309.01219, 2023.

[Zhao et al., 2023a] Wayne Xin Zhao, Kun Zhou, et al. A survey of large language models. arXiv preprint arXiv:2303.18223, 2023.

[Zhao et al., 2023b] Zirui Zhao, Wee Sun Lee, and David Hsu. Large language models as commonsense knowledge for large-scale task planning. arXiv preprint arXiv:2305.14078, 2023.

[Zhong et al., 2023] Wanjun Zhong, Lianghong Guo, Qiqi Gao, and Yanlin Wang. MemoryBank: Enhancing large language models with long-term memory. arXiv preprint arXiv:2305.10250, 2023.

[Zhou et al., 2023] Shuyan Zhou, Frank F. Xu, et al. WebArena: A realistic web environment for building autonomous agents. arXiv preprint arXiv:2307.13854, 2023.
