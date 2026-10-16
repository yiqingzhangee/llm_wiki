###### 语言代理的认知架构（Cognitive Architectures for Language Agents）

**Theodore R. Sumers[∗]** **Shunyu Yao[∗]** **Karthik Narasimhan** **Thomas L. Griffiths**  
_普林斯顿大学（Princeton University）_  
_{sumers, shunyuy, karthikn, tomg}@princeton.edu_

**[OpenReview 评审链接：https://openreview.net/forum?id=1i6ZCvflQJ](https://openreview.net/forum?id=1i6ZCvflQJ)**

###### 摘要（Abstract）

近期的研究工作将大语言模型（Large Language Models, LLMs）与外部资源（例如 Internet）或内部控制流程（例如提示链式调用，prompt chaining）结合，用于需要落地（grounding）或推理的任务，由此催生了一类新的语言代理（language agents）。尽管这些代理在经验上取得了显著成功，我们仍然缺乏一个框架来组织已有的代理，并据此规划未来的发展方向。本文借鉴认知科学和符号人工智能（symbolic AI）丰富的历史，提出用于语言代理的认知架构（Cognitive Architectures for Language Agents, CoALA）。

CoALA 将一种语言代理刻画为：具有模块化记忆组件的系统，拥有一个结构化的动作空间，用以与内部记忆和外部环境交互，并配备一个广义的决策过程（decision-making process）来选择动作。我们使用 CoALA 对大量近期工作进行回顾性综述与组织，并前瞻性地识别出通向更强代理的一系列可操作研究方向。总体而言，CoALA 将当今的语言代理置于人工智能更宏大的发展历史中加以情境化，并勾勒出一条迈向基于语言的通用智能（general intelligence）的路径。

###### 1 引言（Introduction）

_语言代理（language agents；Weng, 2023; Wang et al., 2023b; Xi et al., 2023; Yao and Narasimhan, 2023）是一类新兴的人工智能（Artificial Intelligence, AI）系统，它们利用大语言模型（LLMs；Vaswani et al., 2017; Brown et al., 2020; Devlin et al., 2019; OpenAI, 2023a）与世界进行交互。它们将 LLM 的最新进展应用到已有的代理设计（agent design；Russell and Norvig, 2013）领域。耐人寻味的是，这种综合对两个领域都有益处。_

一方面，LLM 在知识量与推理能力上仍然有限。语言代理通过将 LLM 连接到内部记忆和环境，将其锚定到已有知识或外部观测，从而缓解这些问题。另一方面，传统代理往往需要手工编写规则（Wilkins, 2014）或强化学习（Reinforcement Learning, RL；Sutton and Barto, 2018），这使得它们在新环境上的泛化颇具挑战（Lake et al., 2016）。语言代理利用 LLM 中存在的常识先验来适应新任务，从而减少对人工标注或试错式学习的依赖。

最早的一批代理仅仅使用 LLM 直接选择或生成动作（图 1B；Ahn et al., 2022; Huang et al., 2022b），而更近期的代理还使用 LLM 来进行推理（Yao et al., 2022b）、规划（Hao et al., 2023; Yao et al., 2023）以及管理长期记忆（Park et al., 2023; Wang et al., 2023a），以提升决策能力。最新一代的“认知语言代理”（cognitive language agents）具有极其复杂的内部过程（图 1C）。然而，目前各篇工作使用自定义术语来描述这些过程（例如“工具使用（tool use）”“落地（grounding）”“动作（actions）”），这使得我们很难比较不同代理、理解它们随时间的演化，或者在干净且一致的抽象之上构建新的代理。

为了建立一个组织这些工作的概念框架，我们将其与计算与人工智能（AI）历史中的两个重要思想类比：产生式系统（production systems）和认知架构（cognitive architectures）。产生式系统通过迭代地应用规则生成一组结果（Newell and Simon, 1972）。它们最初以字符串操作系统的形式出现——与 LLM 所解决的问题类似——随后被 AI 社群采纳，用来定义能够产生复杂、层级化行为的系统（Newell et al., 1989）。

∗等贡献作者，作者顺序通过掷硬币决定。每位作者保留在列举作者时将自己列在首位的权利。  
一个基于 CoALA 的语言代理相关工作仓库：[https://github.com/ysymyth/awesome-language-agents](https://github.com/ysymyth/awesome-language-agents)

-----

### � � Cognitive Language Agent


### �

Input  LLM  Output

Language Agent

Observations  Actions

Environment

Environment

图 1：大语言模型（LLMs）的不同用法。A：在自然语言处理（Natural Language Processing, NLP）中，一个 LLM 以文本为输入并输出文本。B：语言代理（Ahn et al., 2022; Huang et al., 2022c）通过将观测转换为文本，并使用 LLM 选择动作，将 LLM 置于与外部环境直接反馈回路之中。C：认知语言代理（Yao et al., 2022b; Shinn et al., 2023; Wang et al., 2023a）进一步使用 LLM 通过学习和推理等过程来管理代理的内部状态。在本文中，我们提出一个蓝图来对这类代理进行结构化设计。

为了构建复杂行为（Newell et al., 1989），产生式系统被纳入认知架构之中，这些认知架构为如何选择、应用甚至生成新的产生式指定了控制流（Laird et al., 1987; Laird, 2022; Kotseruba and Tsotsos, 2020）。我们认为产生式系统与 LLM 之间存在一个有意义的类比：正如产生式给出可能的字符串修改方式，LLM 则在文本上定义了一个关于更改或添加的分布。这进一步暗示：用于控制产生式系统的认知架构中的控制机制，同样可以被应用于把 LLM 转变为语言代理。

因此，我们提出用于语言代理的认知架构 CoALA（Cognitive Architectures for Language Agents），作为一个刻画与设计通用语言代理的概念框架。CoALA 从三个关键维度来组织代理：信息存储（划分为工作记忆与长期记忆）；动作空间（划分为内部和外部动作）；以及决策过程（被结构化为一个包含规划与执行的交互循环）。通过这三个概念——记忆（memory）、动作（action）与决策（decision-making）——我们展示了 CoALA 如何以简洁的方式表达大量已有代理，并识别出尚未充分探索的方向，以发展新的代理。值得注意的是，尽管近期已有若干论文提出关于通用智能的概念架构（LeCun, 2022; McClelland et al., 2019），或对语言模型与代理进行经验性综述（Mialon et al., 2023; Weng, 2023; Wang et al., 2023b），本文则兼具二者：我们提出一个理论框架，并用它来组织多样的经验工作。这既使我们的理论扎根于既有实践，又让我们能够识别出未来工作的短期和长期方向。

本文余下部分结构如下。我们首先介绍产生式系统和认知架构（第 2 节），并展示近期关于 LLM 和语言代理的发展如何重现这些历史思想（第 3 节）。在这些类比的启发下，第 4 节引入 CoALA 框架并用其综述已有语言代理。第 5 节对若干代表性代理进行深入案例分析。第 6 节提出构建未来语言代理的一些可操作步骤，而第 7 节则在更宏大的认知科学与 AI 视角下强调若干开放问题。最后，第 8 节给出总结。对于更关心应用型代理设计的读者，可以重点阅读第 4–6 节。

-----

###### 2 背景：从字符串到符号通用人工智能（Symbolic AGI）

我们首先介绍产生式系统和认知架构，从历史视角回顾认知科学与人工智能：从逻辑与计算理论（Post, 1943）起步，一直延伸到构建符号通用人工智能（symbolic artificial general intelligence）的尝试（Newell et al., 1989）。然后，我们简要介绍语言模型与语言代理。第 3 节将连接这些概念，建立产生式系统与语言模型之间的类比。

**2.1  用于字符串操作的产生式系统（Production systems for string manipulation）**

在 20 世纪上半叶，一条重要的思想路线将数学（Whitehead and Russell, 1997）和计算（Church, 1932; Turing et al., 1936）都化约为符号操作。产生式系统就是这样的形式之一。直观地说，产生式系统由一组规则组成，每条规则都指定一个前提条件（precondition）和一个动作（action）。当前提条件被满足时，相应的动作就可以被执行。该思想源自对计算极限的刻画尝试。Post（1943）提出，可以用这种方式来思考任意逻辑系统，其中公式被表示为字符串，而它们所蕴含的结论则通过产生式规则来刻画（即一个字符串“产生”另一个字符串）。这一表述后来被证明等价于一个更简单的字符串重写（string rewriting）系统。在这样的系统中，我们指定如下形式的规则：

_X Y Z → X W Z_

表示字符串 XY Z 可以被重写为字符串 XWZ。字符串重写在形式语言理论中扮演了重要角色，例如在 Chomsky 的短语结构语法（phrase structure grammar；Chomsky, 1956）中。

**2.2  控制流：从字符串到算法（Control flow: From strings to algorithms）**

单独来看，一个产生式系统只是刻画了从某个起点可以生成的字符串集合。然而，如果我们对“哪些产生式会被执行”施加控制流（control flow），那么它们就可以用来指定算法。例如，Markov 算法就是带有优先级排序的产生式系统（Markov, 1954）。下述算法通过把用竖线“|”表示的数字转换为形如 Q ∗ R 的形式，实现带余数的除法，其中 Q 为被 5 除的商，R 为余数：

_∗||||| → | ∗_

_•_

_∗ −→ ∗_

_→ ∗_

其中，优先级从上到下排序；在从左到右扫描时，一条产生式会被应用到第一个满足其前提条件的子串（包括最后一条产生式中的空子串）。符号 _•_ 与箭头 _−→_ 一起表示在执行该规则之后算法将停止。第一条规则在可能的情况下等价于“减去 5”；第二条规则在无法继续减去 5 时处理终止条件；第三条规则处理空子串的输入情形。例如，在输入为 11 个竖线时，该算法给出如下产生式序列：

_∗||||||||||| → | ∗|||||| → || ∗| −→ || ∗|_

这被解释为“2 余 1”。简单的产生式能够产生复杂行为——可以证明，Markov 算法是图灵完备的。

**2.3  认知架构：从算法到代理（Cognitive architectures: From algorithms to agents）**

产生式系统在 AI 社群中由 Allen Newell 推广，他在寻找一种形式化工具来刻画人类问题求解（Newell, 1967; Newell and Simon, 1972）。产生式系统被从字符串重写推广到逻辑操作：前提条件可以根据代理的目标与世界状态进行检查，而当前提被满足时应执行的动作也被显式给出。在其里程碑式著作《Human Problem Solving》（Newell and Simon, 1972）中，Allen Newell 与 Herbert Simon 举了一个简单产生式系统实现恒温器（thermostat）代理的例子：

\(\text{(temperature > 70[◦]) ∧ (temperature < 72[◦]) → stop}\)

\(\text{temperature < 32[◦] → call for repairs; turn on electric heater}\)

\(\text{(temperature < 70[◦]) ∧ (furnace off) → turn on furnace}\)

\(\text{(temperature > 72[◦]) ∧ (furnace on) → turn off furnace}\)

在此之后，产生式系统被 AI 社群广泛采纳。由此得到的代理包含了与外部传感器、执行器和知识库相连的大规模产生式系统——这就需要相应复杂的控制流。AI 研究者据此定义了模仿人类认知的“认知架构”（cognitive architectures），在其中显式实例化了诸如知觉（perception）、记忆（memory）与规划（planning）等过程（Adams et al., 2012），从而实现灵活、理性、实时的行为（Sun, 2004; Newell, 1980; 1992; Anderson and Lebiere, 2003）。这催生了从心理学建模到机器人学等广泛应用，已有上百种架构和上千篇相关论文（综述见 Kotseruba and Tsotsos, 2020）。

一个典型例子是 Soar 架构（图 2A）。Soar 将产生式存储于长期记忆中，并根据其前提条件与工作记忆的匹配程度来执行这些产生式（图 2B）。这些产生式指定了修改工作记忆和长期记忆内容的动作。下面我们对 Soar 作简要概述，更详细的介绍可见 Laird（2022; 2019）。

**记忆（Memory）。**在心理学理论的基础上，Soar 使用多种记忆来跟踪代理的状态（Atkinson and Shiffrin, 1968）。工作记忆（working memory；Baddeley and Hitch, 1974）反映代理当前的情境：它存储代理最近的感知输入、目标以及内部中间推理的结果。长期记忆被划分为三种不同类型。程序性记忆（procedural memory）存储产生式系统本身：即可以被应用到工作记忆以决定代理行为的规则集合。语义记忆（semantic memory）存储关于世界的事实（Lindes and Laird, 2016），而情景记忆（episodic memory）存储代理过去行为的序列（Nuxoll and Laird, 2007）。

**落地（Grounding）。**Soar 可以被实例化于仿真环境（simulation；Tambe et al., 1995; Jones et al., 1999）或现实世界的机器人系统中（Laird et al., 2012）。在具身（embodied）情境中，各类传感器会将感知输入持续流入工作记忆，使之可用于决策。Soar 代理还可以配备执行器，从而允许物理动作以及通过语言进行交互式学习（Mohan et al., 2012; Mohan and Laird, 2014; Kirk and Laird, 2014）。

**决策（Decision making）。**Soar 实现了一个决策循环（decision loop），用于评估产生式并应用与当前工作记忆最匹配的那一条（图 2B）。产生式存储在长期程序性记忆中。在每一次决策周期中，其前提条件都会与代理的工作记忆进行比对。在“提议与评估”（proposal and evaluation）阶段，一组产生式用于生成并排序一个候选动作集合。[∗] 然后选择其中最优的动作。[†] 另一组产生式则用于实现被选中的动作——例如修改工作记忆的内容，或发出一个运动指令。

**学习（Learning）。**Soar 支持多种学习模式。首先，新信息可以直接被存储到长期记忆中：事实可以写入语义记忆，而经历可以写入情景记忆（Derbinsky et al., 2012）。在需要用于决策时，这些信息可以被重新检索回工作记忆。其次，行为本身可以被修改：可以使用强化学习（Sutton and Barto, 2018）来提升那些曾带来良好结果的产生式的权重，使代理能够从经验中学习（Nason and Laird, 2005）。最引人注目的是，Soar 还能将新的产生式写入其程序性记忆（Laird et al., 1986）——实际上等于在“更新自身源码”。

认知架构在心理学与计算机科学中被广泛使用，其应用包括机器人学（Laird et al., 2012）、军事仿真（Jones et al., 1999; Tambe et al., 1995）以及智能辅导系统（intelligent tutoring；Koedinger et al., 1997）等。然而在过去几十年中，它们在 AI 社群中的受欢迎程度有所下降。这一下降反映了此类系统面临的两大挑战：其一，它们被局限在可以用逻辑谓词描述的领域；其二，它们需要大量预先指定的规则才能运作。

耐人寻味的是，LLM 似乎非常适合应对这些挑战。首先，LLM 在任意文本上操作，这使得它们比基于逻辑的系统更灵活。其次，与需要用户显式指定产生式不同，LLM 通过在互联网语料上的预训练学习到一个关于产生式的分布。研究者已经开始在认知架构中使用 LLM，利用其隐含的世界知识（Wray et al., 2021）来增强传统的符号方法（Kirk et al., 2023; Romero et al., 2023）。在本文中，我们则是从另一方面出发：引入认知架构中的原则以指导基于 LLM 的代理设计。

**2.4  语言模型与代理（Language models and agents）**

语言建模（language modeling）在 NLP 与 AI 社群中已是几十年的研究课题，其目标是开发在给定某种上下文时能够生成文本的系统（Jurafsky, 2000）。形式上，语言模型学习一个分布 \(P(w_i \mid w_{<i})\)，其中每个 \(w\) 是一个单独的 token（单词）。在此模型下，可以通过一次采样一个 token 的方式，从该分布中采样来生成文本。从本质上说，语言模型是一个概率性的输入—输出系统，因为对同一段文本的接续通常具有多种可能（例如，“I went to the” → “market” | “beach” | ...）。早期的语言建模尝试（例如 n-gram）在泛化能力与可扩展性上遭遇挑战，而近来由于基于 Transformer 的 LLM（Vaswani et al., 2017）以及大规模参数量（数十亿）的出现（例如 GPT-4；OpenAI, 2023a），再加上更好的分词方案，语言建模重新焕发活力。现代 LLM 在海量数据上训练，这帮助它们从大量输入—输出组合中积累知识，进而成功地生成类人文本（Andreas, 2022）。

出乎意料的是，在互联网规模文本上训练这些模型也使它们在许多超越“生成文本”的任务上变得有用，例如写代码（Li et al., 2022b; Rozière et al., 2023; Li et al., 2023c）、建模蛋白质（Meier et al., 2021）以及在交互式环境中行动（Yao et al., 2022b; Nakano et al., 2021）。后者促成了“语言代理”（language agents）的兴起——它们将 LLM 作为核心计算单元，用以推理、规划与行动——并在诸如机器人学（Ahn et al., 2022）、制造业（Xia et al., 2023）、网页操作（Yao et al., 2022a; Deng et al., 2023）、谜题求解（Yao et al., 2023; Hao et al., 2023）以及交互式代码生成（Yang et al., 2023）等领域获得应用。语言理解与决策能力的结合是一个令人振奋且迅速发展的方向，有望让这些代理更接近类人的智能。

[∗] 更详细地说，Soar 将产生式分为两类：“operators（算子）”，我们在文中称之为动作（actions），以及“rules（规则）”，用于提出、评估与执行算子。

[†] 如果没有任何动作有效，或多个动作打成平手，则会发生僵局（impasse）。Soar 会创建一个子目标（subgoal）来解决该僵局，从而实现层级化的任务分解。更详细的讨论见 Laird（2022）。

-----

###### 3 语言模型与产生式系统之间的联系（Connections between Language Models and Production Systems）

基于二者在处理字符串任务上的共同起源，将产生式系统与语言模型进行类比是十分自然的。我们首先展开这一类比，然后展示提示工程方法如何重现基于产生式系统的算法与代理。产生式系统与语言模型之间的对应关系，激发我们采用认知架构来构建语言代理，这将在第 4 节中介绍。

**3.1  作为概率产生式系统的语言模型（Language models as probabilistic production systems）**

在最初的形式中，产生式系统刻画了从某一初始点可以生成的字符串集合，并将这一过程拆解为一系列字符串重写操作。语言模型同样定义了一组对某个字符串——给定给模型的提示（prompt）——进行扩展或修改的可能方式。[‡]

例如，我们可以将完成一段文本的任务形式化为一个产生式问题。如果 X 是提示、Y 是续写，那么我们可以将其写成产生式 X → X Y。[§] 我们可能希望允许多个可能的续写，此时可写成 X → X Y_i，其中 Y_i 属于某个集合。LLM 为每个候选续写分配一个概率。从这个角度看，给定输入 X 时，LLM 就定义了一个在这些产生式之间进行选择的概率分布，从而得到一个关于可能续写的分布 P(Y_i \mid X)（Dohan et al., 2022）。因此，LLM 可以被视为概率产生式系统：每调用一次，它就从中采样一个可能的续写，例如 X ∼∼▸ X Y。

这种概率形式相较于传统产生式系统既有优势，也有劣势。LLM 的主要劣势在于其固有的不透明性：产生式系统由离散且对人类而言易读的规则构成，而 LLM 由数十亿难以解释的参数组成。这种不透明性——再加上其概率形式带来的内在随机性——使得分析或控制其行为颇具挑战（Romero et al., 2023; Valmeekam et al., 2022）。尽管如此，LLM 的规模与预训练仍为其相对传统产生式系统带来巨大优势。基于大规模互联网数据预训练的 LLM 学得了一个关于字符串续写的极其有效的先验，使其能够开箱即用地解决广泛任务（Huang et al., 2022b）。

**3.2  将提示工程视作控制流（Prompt engineering as control flow）**

LLM 的参数权重在给定输入字符串（即提示）时定义了一个对输出字符串（即续写）的优先级排序。由此得到的分布可以被解释为“对产生式的任务特定优先排序”——换言之，是一种简单的控制流。诸如问答之类的任务可以直接被表述为一个输入字符串（问题），从而得到一个关于续写（可能答案）的条件分布。

早期关于小样本学习（few-shot learning；Brown et al., 2020）和提示工程（Wei et al., 2022b; Kojima et al., 2022; Xu et al., 2023c）的工作发现，通过对输入字符串进行预处理，可以进一步偏置 LLM 倾向于高质量的产生式。这些简单的操作——通常是将额外文本拼接到输入上——本身也可以被视为产生式，因此这些方法定义了一系列产生式（见表 1）。后续工作将这些方法扩展到了动态的、与上下文相关的提示：例如选择与输入最相关的小样本示例（Liu et al., 2021），或者用来自视频（Zeng et al., 2022）或数据库（Lewis et al., 2020）的外部观测来填充模板。关于此类提示技术的综述，可见 Liu et al.（2023d）。

随后的一些工作使用 LLM 本身作为预处理步骤，通过引出有针对性的推理来凸显问题的特定方面（Bai et al., 2022; Jin et al., 2022; Ganguli et al., 2023; Madaan et al., 2023; Saunders et al., 2022; Kim et al., 2023; Kirk et al., 2023），或在返回最终答案之前生成中间推理步骤（Tafjord et al., 2021; Creswell et al., 2023; Yao et al., 2023）。将多次对 LLM 的调用串联起来（Wu et al., 2022a;b; Dohan et al., 2022）可以实现愈发复杂的算法（图 3）。

[‡] 在本文中，我们主要关注自回归式 LLM，它们通常被用于语言代理。然而，双向 LLM（例如 BERT；Devlin et al., 2019）也可以用类似方式理解：它们定义了一个关于“填空型产生式”（in-filling productions）的分布。

[§] 或者，我们也可以将提示视作输入，并把 LLM 的输出视作下一状态，由产生式 X → Y 表示——这是一种更字面意义上的“重写”。
-----

表 1：提示方法如何在生成续写之前操纵输入字符串的概念性示意图。Q = 问题（question），A = 答案（answer），O = 观测（observation），C = 批判（critique），符号 ∼∼∼▸ 表示从一个随机产生式中采样。这些预处理操作——可以调用其他模型（例如视觉—语言模型，Vision-Language Models, VLMs），甚至调用 LLM 自身——都可以被视为产生式。因此，提示方法实际上定义了一系列产生式。

| 提示方法（Prompting Method） | 产生式序列（Production Sequence） |
| --- | --- |
| 零样本（Zero-shot） | \(Q ∼∼∼∼▸_{\text{LLM}} Q A\) |
| 小样本（Few-shot） | \(Q \rightarrow Q_1 A_1 Q_2 A_2 Q ∼∼∼∼▸_{\text{LLM}} Q_1 A_1 Q_2 A_2 Q A\) |
| 利用维基百科 + LLM 的检索增强生成（Retrieval Augmented Generation） | \(Q \rightarrow Q O ∼∼∼∼▸ Q O A\) |
| 苏格拉底模型（Socratic Models） | \(Q ∼∼∼∼▸_{\text{VLM}} Q O ∼∼∼∼▸_{\text{LLM}} Q O A\) |
| 自我批判（Self-Critique） | \(Q ∼∼∼∼▸_{\text{LLM}} Q A ∼∼∼∼▸_{\text{LLM}} Q A C ∼∼∼∼▸_{\text{LLM}} Q A C A\) |


Question

##### � � �

Prompt construction

Answer  Critique  Refinement  Answer

LLM

ReAct

String parsing

Execution

Self-Critique

Question

Inference

Selection-Inference

Answer

图 3：从语言模型到语言代理。A：一次 LLM 调用的基本结构。“提示构造”（prompt construction）选择一个模板，并用工作记忆中的变量来填充该模板。在调用 LLM 之后，其字符串输出会被解析到一个动作空间中并被执行。一次 LLM 调用可能产生一个或多个动作——例如返回答案、调用函数或发出运动指令。B：诸如 Self-Critique（Wang et al., 2022b）或 Selection-Inference（Creswell et al., 2023）这类提示链式（prompt chaining）技术，使用预定义的 LLM 调用序列来生成输出。C：语言代理（例如 Inner Monologue；Huang et al., 2022c，和 ReAct；Yao et al., 2022b）则改为使用与外部环境之间的交互式反馈回路。视觉—语言模型（VLMs）可以将感知数据翻译为文本，供 LLM 处理。

**3.3 迈向认知语言代理（Towards cognitive language agents）**

_语言代理超越了预定义的提示链，而是将 LLM 置于与外部环境构成的反馈回路之中（图 1B）。_ 这些方法首先将多模态输入转换为文本并传递给 LLM。随后，LLM 的输出被解析并用来确定一个外部动作（图 3C）。早期的代理直接将 LLM 与外部环境对接，使用它根据代理的状态生成高层指令（Ahn et al., 2022; Huang et al., 2022c; Dasgupta et al., 2022）。后续工作发展出更为复杂的语言代理，在选择动作前使用 LLM 进行中间推理（Yao et al., 2022b）。最新的代理进一步引入了复杂的学习策略，例如反思情景记忆以生成新的语义推论（Shinn et al., 2023），或修改其程序代码以生成程序性知识（procedural knowledge）（Wang et al., 2023a），从而利用以往经验来适配未来行为。

这些认知语言代理利用了非平凡的、基于 LLM 的推理与学习（图 1C）。正如认知架构曾被用于结构化产生式系统与代理内部状态及外部环境之间的交互一样，我们认为它们同样可以帮助设计基于 LLM 的认知代理。在本文余下部分，我们将使用这一视角来组织已有方法，并突出若干有前景的扩展方向。

-----

## �


## �

| L L M | Agent Code |
| --- | --- |
| Col1 | Observation  Planning  Proposal  Evaluation  Selection | Col3 | Col4 |
| --- | --- | --- | --- |
|  |  | Proposal |  |
| Evaluation  Selection  Execution | Evaluation  Selection |  |  |
|  |  |  |  |
|  |  | Execution |  |

图 4：用于语言代理的认知架构（CoALA）。A：CoALA 定义了一组相互作用的模块与过程。决策过程（decision procedure）负责执行代理的“源码”（source code）。这些源码由若干过程组成，用于与 LLM（提示模板与解析器）、内部记忆（检索与学习）以及外部环境（落地）交互。B：从时间维度来看，代理的决策过程与外部环境构成一个循环的决策周期。在每个周期中，代理使用检索与推理来进行规划（planning），通过提出和评估候选的“学习”或“落地”动作来进行。然后选择并执行最优动作，产生新的观测，循环再次开始。

这些程序代码可以用来生成程序性知识（Wang et al., 2023a），利用代理的既往经验来适配其未来行为。

这些认知语言代理采用了非平凡的基于 LLM 的推理与学习机制（图 1C）。正如认知架构曾被用来结构化产生式系统与代理内部状态及外部环境之间的交互，我们认为，它们同样可以帮助设计基于 LLM 的认知代理。在本文的剩余部分，我们将使用这一视角来组织已有方法，并突出若干有前景的扩展方向。

###### 4 语言代理的认知架构（CoALA）：一个概念框架

我们提出用于语言代理的认知架构 CoALA（Cognitive Architectures for Language Agents）作为一个框架，用于组织已有语言代理并指导新代理的发展。CoALA 将 LLM 置于更大认知架构的核心组件位置（图 4）。在 CoALA 下，一个语言代理将信息存储于若干记忆模块中（第 4.1 节），并在一个划分为外部和内部部分的动作空间中行动（图 5）：

- **外部动作（external actions）** 通过落地（grounding，第 4.2 节）与外部环境交互（例如控制机器人、与人类沟通、导航网站）。
- **内部动作（internal actions）** 与内部记忆交互。根据访问的是哪种记忆以及访问是读（read）还是写（write），内部动作可以进一步分解为三类：**检索（retrieval，从长期记忆中读取；第 4.3 节）、推理（reasoning，使用 LLM 更新短期工作记忆；第 4.4 节）以及学习（learning，向长期记忆写入；第 4.5 节）。**

语言代理通过决策（decision-making）选择动作，该过程遵循一个重复的决策周期（第 4.6 节，图 4B）。在每个周期中，代理可以使用推理与检索动作来进行规划。该规划子过程会选择一个落地（grounding）或学习（learning）动作，并执行之，以影响外部世界或代理的长期记忆。CoALA 的决策周期类似于程序的“main” 主过程（与具有返回值的函数相对）：它在循环中持续运行，不断接收新的感知输入，并据此调用各种动作过程。


Internal   External

Reasoning   Retrieval   Learning   Grounding

Planning

图 5：代理的动作空间可以划分为内部记忆访问和与外部世界的外部交互。推理和检索动作被用于支撑规划。

与函数相对，CoALA 的“主程序”在循环中持续运行，不断接受新的感知输入，并据此调用不同的动作过程。

CoALA（图 4）受到数十年认知架构研究的启发（第 2.3 节），吸收了记忆、落地、学习与决策等关键概念。然而，LLM 的引入带来了“推理”动作这一新增部分，它可以为各种用途灵活地产生新知识与启发式，取代传统认知架构中手写规则的角色。LLM 还使得“文本”成为事实上的内部表示形式，从而简化了代理的记忆模块。最后，近期视觉—语言模型（VLMs；Alayrac et al., 2022）的进展可以通过提供一种将感知数据直接翻译为文本的路径（Zeng et al., 2022），进一步简化落地过程。

本节余下部分将详细讨论 CoALA 中的关键概念：记忆（memory）、动作（落地、推理、检索与学习）以及决策。对于每个概念，我们都会以已有的语言代理（或相关的 NLP/RL 方法）为例，或者指出文献中的空白，为未来研究方向提供线索。

**4.1  记忆（Memory）**

语言模型本身是无状态的：它们不会在不同调用之间持续保存信息。相对地，语言代理可以在内部存储和维护信息，以便在与世界的多步交互中使用。在 CoALA 框架中，语言代理会显式地将信息（主要是文本，也允许其他模态）组织进多个记忆模块，每个模块都包含一种不同形式的信息。它们包括短期的工作记忆，以及若干长期记忆：情景记忆（episodic）、语义记忆（semantic）和程序性记忆（procedural）。

**工作记忆（Working memory）。** 工作记忆在当前决策周期（第 4.6 节）中维护处于激活状态、随时可用的信息，并以符号变量的形式表示。这包括感知输入、当前激活的知识（由推理生成或从长期记忆检索而来），以及从上一决策周期继承来的其他核心信息（例如代理的当前目标）。之前的工作鼓励 LLM 生成中间推理（Wei et al., 2022b; Nye et al., 2021），将 LLM 自身的上下文视为一种工作记忆。CoALA 中工作记忆的概念更为一般：它是一个在多次 LLM 调用之间持续存在的数据结构。在每次 LLM 调用时，LLM 的输入会从工作记忆的一个子集合成（例如，一个提示模板以及相关变量）；LLM 的输出随后被解析为其他变量（例如动作名称及其参数），这些变量被写回工作记忆，并被用于执行对应动作（图 3A）。除了与 LLM 交互之外，工作记忆还会与长期记忆和落地接口进行交互，因此它在语言代理中充当连接不同组件的中心枢纽。

**情景记忆（Episodic memory）。** 情景记忆存储来自早前决策周期的经验。这可以是训练时的输入—输出对（Rubin et al., 2021）、历史事件流（event flows；Weston et al., 2014; Park et al., 2023）、来自过往 episode 的游戏轨迹（Yao et al., 2020; Tuyls et al., 2022），或其他形式的代理经验表示。在某一次决策周期的规划阶段，这些情景可以被检索到工作记忆中，以支撑推理。代理也可以将新的经验从工作记忆写入情景记忆，作为一种学习形式（第 4.5 节）。

**语义记忆（Semantic memory）。** 语义记忆存储代理关于世界以及自身的知识。传统的 NLP 或 RL 方法在推理或决策时利用检索机制时，通常从外部数据库中初始化语义记忆以提供知识支持。例如，NLP 中的检索增强方法（retrieval-augmented methods；Lewis et al., 2020; Borgeaud et al., 2022; Chen et al., 2017）可以被视作从一个由非结构化文本（例如 Wikipedia）组成的语义记忆中检索。在 RL 中，“阅读以学习”（reading to learn）的方法（Branavan et al., 2012; Narasimhan et al., 2018; Hanjie et al., 2021; Zhong et al., 2021）利用游戏手册与事实等作为语义记忆，从而影响策略。尽管这些例子本质上使用的是一个固定的、只读的语义记忆，语言代理也可以将从 LLM 推理获得的新知识写入语义记忆，作为一种学习形式（第 4.5 节），从经验中逐步积累世界知识。

| Reasoning | Retrieval | Learning | Grounding |
| --- | --- | --- | --- |

上述表格在原文中仅占位，示意不同类别的动作；在 CoALA 中，它们分别对应推理、检索、学习与落地动作。

-----

**程序性记忆（Procedural memory）。** 语言代理包含两种形式的程序性记忆：一是存储在 LLM 权重中的隐式知识；二是写在代理代码中的显式知识。代理代码又可以进一步划分为两类：一类是实现动作的过程（即实现推理、检索、落地和学习的过程），另一类是实现决策本身的过程（第 4.6 节）。在一个决策周期中，LLM 可以通过推理动作被访问，而各类基于代码的过程也可以被检索并执行。与可能最初为空甚至不存在的情景或语义记忆不同，程序性记忆必须由设计者预先用合适的代码进行初始化，以便为代理“引导起步”（bootstrap）。最后，尽管通过向程序性记忆写入可以学习新的动作是可能的（第 4.5 节），但与向情景或语义记忆写入相比，这要风险大得多，因为很容易引入 bug，或使代理偏离设计者的意图。

**4.2  落地动作（Grounding actions）**

落地过程负责执行外部动作，并将环境反馈处理为文本形式写入工作记忆。这实际上将代理与外部世界的交互简化为一个“文字游戏”（text game）：观测和动作都以文本形式表示。我们将外部环境归纳为三类：

**物理环境（Physical environments）。** 物理具身是人们为 AI 代理设想的最早实例之一（Nilsson, 1984）。它涉及将感知输入（视觉、听觉、触觉）处理为文本观测（例如通过预训练的图像字幕模型），并通过接收语言指令的机器人规划器来影响物理环境。LLM 的最新进展催生了诸多机器人项目（Ahn et al., 2022; Liang et al., 2023a; Singh et al., 2023; Palo et al., 2023; Ren et al., 2023），这些项目利用 LLM 作为机器人的“大脑”，在物理世界中生成动作或规划。对于感知输入，视觉—语言模型通常被用来将图像转换为文本（Alayrac et al., 2022; Sumers et al., 2023），为 LLM 提供额外上下文（Driess et al., 2023; Huang et al., 2023; Brohan et al., 2022; 2023）。

**与人类或其他代理对话（Dialogue with humans or other agents）。** 传统的语言交互允许代理接收指令（Winograd, 1972; Tellex et al., 2011; Chen and Mooney, 2011; Bisk et al., 2016），或者从人类身上学习（Nguyen et al., 2021; Sumers et al., 2022; 2021; Wang et al., 2016）。能够生成语言的代理可以主动请求帮助（Ren et al., 2023; Nguyen et al., 2022b; 2019; Nguyen and Daumé III, 2019）或澄清信息（clarification；Biyik and Palan, 2019; Sadigh et al., 2017; Padmakumar et al., 2022; Thomason et al., 2020; Narayan-Chen et al., 2019）——也可以用于娱乐或情绪支持（Zhang et al., 2020; Zhou et al., 2018; Pataranutaporn et al., 2021; Hasan et al., 2023; Ma et al., 2023）。近期工作还研究了多语言代理之间的交互，用于社会模拟（social simulation；Park et al., 2023; Jinxin et al., 2023; Gao et al., 2023）、辩论（Chan et al., 2023; Liang et al., 2023b; Du et al., 2023）、提升安全性（Irving et al., 2018），或协作完成任务（Qian et al., 2023; Wu et al., 2023; Hong et al., 2023a; Dong et al., 2023）。

**数字环境（Digital environments）。** 这一类包括与游戏交互（Hausknecht et al., 2020; Côté et al., 2019; Shridhar et al., 2020; Wang et al., 2022a; Liu et al., 2023e）、API（Schick et al., 2023; Yao et al., 2022b; Parisi et al., 2022; Tang et al., 2023b）与网站（Shi et al., 2017; Nakano et al., 2021; Yao et al., 2022a; Zhou et al., 2023b; Gur et al., 2023; Deng et al., 2023），以及一般的代码执行（Yang et al., 2023; Le et al., 2022; Ni et al., 2023）。这种数字化落地相较物理或人类交互更廉价且更快速，因此是语言代理的一个便利试验台，也在近年来被越来越多地研究。特别地，对于需要外部知识或额外计算的 NLP 任务，无状态的数字 API（例如搜索、计算器、翻译器）通常被封装为“工具”（tools；Parisi et al., 2022; Schick et al., 2023; Xu et al., 2023a; Tang et al., 2023b; Qin et al., 2023），可以被视为特殊的“一次性”数字环境。

**4.3  检索动作（Retrieval actions）**

在 CoALA 中，“检索过程”（retrieval procedure；Li et al., 2022a; Gu et al., 2018）负责将信息从长期记忆读取到工作记忆中。根据信息类型与记忆类型的不同，它可以以多种方式实现，例如基于规则的检索、稀疏检索或稠密检索。例如，Voyager（Wang et al., 2023a）通过稠密检索从技能库中加载基于代码的技能，以便在 Minecraft 世界中行动——这实质上是从程序性记忆中检索落地过程。Generative Agents（Park et al., 2023）则通过结合近期性（rule-based）、重要性（基于推理）以及相关度（基于嵌入）的得分，从情景记忆中检索相关事件。DocPrompting（Zhou et al., 2022a）提出利用库文档来辅助代码生成，这可以视作从语义记忆中检索知识。尽管检索在人类决策中扮演关键角色（Zhou et al., 2023a; Zhao et al., 2022），在语言代理中，适应性强、情境特定的回忆机制仍然研究不足。我们在第 6 节中建议，将决策与检索进行系统集成是一个重要的未来研究方向。

**4.4  推理动作（Reasoning actions）**

推理允许语言代理处理工作记忆中的内容并生成新的信息。与从长期记忆读入工作记忆的检索不同，推理在工作记忆上进行读写操作。这使得代理可以对最近的观测进行总结与提炼（Yao et al., 2022b; Peng et al., 2023），对最近的轨迹进行概括（Shinn et al., 2023），或对从长期记忆中检索到的信息进行整合（Park et al., 2023）。推理可以用于支持学习（将结果写入长期记忆），也可以用于支持决策（将结果作为后续 LLM 调用的额外上下文）。

**4.5  学习动作（Learning actions）**

学习通过向长期记忆写入信息来发生，其中包含一系列多样的过程。

**用经验更新情景记忆。** 在 RL 领域，代理将情景轨迹存储起来以更新参数化策略（Blundell et al., 2016; Pritzel et al., 2017），或构建非参数化策略（Ecoffet et al., 2019; Tuyls et al., 2022），已是常见做法。对于语言代理，写入情景记忆的新增经验随后可以被检索出来，作为示例与推理或决策的依据（Weston et al., 2014; Rubin et al., 2021; Park et al., 2023）。

**用知识更新语义记忆。** 近期工作（Shinn et al., 2023; Park et al., 2023）使用 LLM 对原始经历进行推理，并将所得推论存入语义记忆。例如，Reflexion（Shinn et al., 2023）使用 LLM 反思失败的 episode，并将其结果（例如“厨房里没有洗碗机”）作为语义知识存储起来，在后续 episode 中附加到 LLM 上下文用于求解。最后，机器人领域的工作（Chen et al., 2023a）利用视觉—语言模型构建环境的语义地图，之后可以查询该地图来执行指令。

**更新 LLM 参数（程序性记忆）。** LLM 权重表示隐式的程序性知识。可以在代理生命周期中通过微调（fine-tuning）对其进行调整，以适配特定领域。这类微调可以通过有监督学习（Liu et al., 2023c; Zhang et al., 2023b）或模仿学习（imitation learning；Hussein et al., 2017）、基于环境反馈的强化学习（Sutton and Barto, 2018）、基于人类反馈的强化学习（Reinforcement Learning from Human Feedback, RLHF；Christiano et al., 2017; Ouyang et al., 2022; Nakano et al., 2021），或基于 AI 反馈的强化学习（Bai et al., 2022; Liu et al., 2023f）来完成。经典的 LLM 自我提升方法（Huang et al., 2022a; Zelikman et al., 2022）使用外部指标（例如一致性；Wang et al., 2022b）来选择用于微调的生成结果。在强化学习场景中，这可以推广为直接使用环境反馈：例如，XTX（Tuyls et al., 2022）会周期性地在情景记忆中存储的高得分轨迹上微调一个小型语言模型，将其作为一种鲁棒的“开发”（exploitation）策略，以在随机性面前抵达探索前沿。对代理的 LLM 进行微调是一种代价高昂的学习形式，因此目前的研究通常预先指定学习计划。然而，随着训练变得更高效——或者代理改用更小的、针对子任务的 LLM——也许可以允许语言代理自主决定在何时以及如何微调其 LLM。

-----

**更新代理代码（程序性记忆）。** CoALA 允许代理更新自己的源码，从而修改各类过程的实现。大致可以分为如下几类：

- **更新推理过程（例如提示模板；Gao et al., 2020; Zhou et al., 2022b）。** 例如，APE（Zhou et al., 2022b）从输入—输出示例中推断提示指令，并将这些指令作为 LLM 提示的一部分，来辅助任务求解。这样的提示更新可以视作一种“学习如何推理”。
- **更新落地过程（例如基于代码的技能；Liang et al., 2023a; Ellis et al., 2021; Wang et al., 2023a）。** 例如，Voyager（Wang et al., 2023a）维护一个课程（curriculum）库。值得注意的是，目前的方法基本仅限于创建与外部环境交互的新代码技能。
- **更新检索过程。** 据我们所知，这类学习选项在近期语言代理中尚未被系统研究。检索通常被视为由某种固定实现（例如 BM25 或稠密检索）设计出的基本动作。但在查询/文档扩展（query/document expansion；Nogueira et al., 2019; Wang et al., 2023c; Tang et al., 2023a）或检索蒸馏（retrieval distillation；Izacard et al., 2021）等研究中，已有的方法可能有助于语言代理学习更好的检索过程。
- **更新学习或决策过程。** 最后，从理论上讲，CoALA 代理可以学习新的“学习”或“决策”过程，从而获得显著的适应能力。然而，总体而言，对这些过程进行更新对于代理的功能性和对齐性（alignment）而言都风险颇高。目前我们尚不清楚有任何语言代理实现了这种形式的学习；我们将在第 6 节中进一步讨论这类可能性。

通常，RL 代理往往固定一种学习方式（例如 Q-learning、PPO 或 A3C），并通过更新模型参数来学习；而语言代理可以从多种学习过程之中进行选择。这允许它们通过存储与任务相关的语言内容（这一操作比更新参数更便宜、更快速）来快速学习，并利用多种学习形式叠加其自我改进效果（例如第 5 节中讨论的 Generative Agents）。

最后，尽管我们的讨论主要围绕“向记忆中添加内容”，对记忆的修改与删除（即“去学习”，unlearning）在近期语言代理中仍然研究不足。我们将在第 6 节中进一步讨论这些方面。

**4.6  决策（Decision making）**

在动作空间中存在多种动作（落地、学习、推理、检索）的情况下，语言代理应如何选择要应用哪一个？这由决策过程（decision-making procedure）来完成，它可以被看作代理的顶层或“main” 主程序。CoALA 将这一顶层程序结构化为一系列决策周期（decision cycles；图 4B），每个周期最终会输出一个外部落地动作（第 4.2 节）或内部学习动作（第 4.5 节）。
在每个决策周期中，程序代码会定义一系列推理与检索动作，用于提出并评估备选方案（规划阶段），然后执行被选中的动作（执行阶段）——之后决策周期再次循环。

**规划阶段（Planning stage）。** 在规划阶段，推理与检索可以被灵活地用来提出、评估并选择动作，这些子阶段可以交替或迭代，从而在采取外部动作之前构建多步模拟（multi-step simulations；Tamari et al., 2020）（Yao et al., 2023; Hao et al., 2023）。这也使得代理可以迭代地改进候选解——例如，使用 LLM 来模拟这些解、识别缺陷，并提出修正方案以弥补这些缺陷（Kirk et al., 2023; Shinn et al., 2023）。

- **提议（Proposal）。** 提议子阶段生成一个或多个候选动作。常见做法是使用推理（以及可选的检索）从 LLM 中采样一个（Huang et al., 2022c）或多个（Chen et al., 2021; Wang et al., 2022b）外部落地动作。对于动作有限的简单领域，提议阶段也可以简单地包含所有动作（例如第 5 节中的 SayCan）。更复杂的代理会使用 if-else 或 while-if 形式的代码结构（Wang et al., 2023a; Park et al., 2023）；而部署在定义良好的领域中的代理则可能利用结构化模拟器（structured simulators；Haslum et al., 2019）来生成合理的 rollout（Liu et al., 2023a; Dagan et al., 2023）。

- **评估（Evaluation）。** 如果提出了多个动作，评估子阶段会为每个动作分配一个价值。这可以使用启发式规则、LLM 的（困惑度）值（perplexity；Ahn et al., 2022）、学习得到的价值（Yao et al., 2020）、LLM 推理（Yao et al., 2023; Hao et al., 2023），或它们的某种组合。特别地，LLM 推理可以通过在内部模拟动作在外部世界中的落地反馈（Hao et al., 2023; Yang et al., 2023）来帮助评估动作。

- **选择（Selection）。** 在给定一组动作及其价值之后，选择步骤要么从中选出一个执行，要么拒绝全部并回到提议步骤。根据动作价值的形式不同，选择可以通过 argmax、softmax 或其他机制（例如多数投票；majority vote；Wang et al., 2022b）来实现。

**执行（Execution）。** 被选中的动作通过执行代理源码中对应的过程来实现。根据代理的实现不同，这可能是一个外部落地动作（例如一次 API 调用，第 4.2 节），也可能是一个内部学习动作（例如写入情景记忆，第 4.5 节）。环境会因此产生新的观测，作为代理动作的反馈，决策周期随之再次循环。

在经验上，许多早期语言代理只是使用 LLM 来提出一个动作（Schick et al., 2023）、一个动作序列（Huang et al., 2022b），或仅对一个固定的动作集合进行评估（Ahn et al., 2022），而没有中间的推理或检索。后续工作（Yao et al., 2022b; Shinn et al., 2023; Xu et al., 2023b; Lin et al., 2023; Wang et al., 2023a; Park et al., 2023）开始利用中间推理与检索来分析情境、制定并维护行动计划，并在得到环境反馈后对之前的动作进行改进；对应的决策过程因此更加复杂，尽管最终仍只提出一个动作。最新的研究则开始探索更复杂的决策方式，采用迭代提议与评估来考虑多个动作。这些过程以经典规划算法为蓝本：例如，Tree of Thoughts（Yao et al., 2023）和 RAP（Hao et al., 2023）分别使用 LLM 实现宽度优先/深度优先搜索（BFS/DFS）与蒙特卡洛树搜索（Monte Carlo Tree Search, MCTS；Browne et al., 2012）。在其中，LLM 被用来生成提议（即在给定动作的条件下模拟 rollout），并对这些 rollout 的结果进行评估（即为动作结果赋值）。

###### 5 案例研究（Case Studies）

通过在记忆模块、动作空间和决策过程上的不同配置与消融，CoALA 能够表达宽泛谱系的语言代理。表 2 列出了若干近期在多种领域中的代表性方法——从 Minecraft 到机器人，从纯推理到社会模拟。CoALA 有助于刻画它们的内部机制，并以简单而结构化的方式揭示它们之间的相似性与差异。

表 2：将若干近期语言代理映射到 CoALA 框架中的示例。

| 长期记忆（Long-term Memory）[¶] | 外部落地（External Grounding） | 内部动作（Internal Actions） | 决策（Decision Making） |
| --- | --- | --- | --- |
| SayCan（Ahn et al., 2022） | 物理（physical） | — | 评估（evaluate） |
| ReAct（Yao et al., 2022b） | 数字（digital） | 推理（reason） | 提议（propose） |
| Voyager（Wang et al., 2023a） | 程序性（procedural） | 数字（digital） | 推理/检索/学习（reason/retrieve/learn） | 提议（propose） |
| Generative Agents（Park et al., 2023） | 情景/语义（episodic/semantic） | 数字/多代理（digital/agent） | 推理/检索/学习（reason/retrieve/learn） | 提议（propose） |
| Tree of Thoughts（Yao et al., 2023） | — | 数字[‖] | 推理（reason） | 提议、评估与选择（propose, evaluate, select） |

**SayCan（Ahn et al., 2022）** 将语言模型落地到厨房中的机器人交互，用以满足用户指令（例如：“我刚刚锻炼完，你能给我拿一杯饮料和一个零食来恢复体力吗？”）。它的长期记忆仅包含程序性记忆（一个 LLM 与一个学习得到的价值函数）。其动作空间仅包含外部动作——一个固定的 551 个落地技能集合（例如“find the apple”“go to the table”），没有内部的推理、检索或学习动作。在决策过程中，SayCan 使用 LLM 与学习到的价值函数相结合来评估每个动作，平衡技能的有用性与可落地性。因此，SayCan 将 LLM（与价值函数结合）用作一个单步规划器。

**ReAct（Yao et al., 2022b）** 是一个落地到多种数字环境（例如 Wikipedia API、文本游戏、网站）的语言代理。与 SayCan 一样，它缺乏语义或情景记忆，因此没有检索或学习动作。其动作空间由（内部的）推理和（外部的）落地两部分组成。其决策周期被固定为：使用一次推理动作分析情境并（重新）制定行动计划，然后直接生成一个落地动作，不包含评估或选择阶段。ReAct 可以被视作利用内部与外部动作的最简单语言代理，也是最早展示“推理与行动协同效应”的工作：推理帮助引导行动，而行动则提供环境反馈以支持后续推理。

**Voyager（Wang et al., 2023a）** 是一个落地到 Minecraft API 的语言代理。与通过价值函数接收感知输入的 SayCan 不同，Voyager 的落地仅基于文本。它拥有一个长期程序性记忆，用于存储一个由基于代码的落地过程（又称技能）组成的库（例如“combatZombie”“craftStoneSword”）。该库是层级化的：复杂技能可以将简单技能作为子过程来调用（例如，当没有剑时，“combatZombie” 可能会调用 “craftStoneSword”）。尤其值得注意的是，Voyager 的动作空间同时包含四类动作：落地、推理、检索与学习（通过添加新的落地过程）。在一个决策周期中，Voyager 首先进行推理以在工作记忆中提出新的任务目标（若其缺失），然后再通过推理提出一个基于代码的落地过程来解决该任务。在下一个决策周期中，Voyager 会根据环境反馈进行推理以判断任务是否完成。如果成功，Voyager 会选择一个学习动作，将该落地过程添加到程序性记忆中；否则，它会使用推理来改进代码并重新执行。通过与 ReAct、AutoGPT 以及去除程序性记忆的消融版本对比，Voyager 经验性地验证了长期记忆与程序性学习的重要性：Voyager 能够更好地探索区域、掌握科技树，并在未见任务上实现零样本泛化。

**Generative Agents（Park et al., 2023）** 是落地到沙盒游戏中的语言代理，允许与环境和其他代理进行交互。其动作空间同样包含四类动作：落地、推理、检索与学习。每个代理都有一个长期情景记忆，以列表形式存储事件。这些代理使用检索与推理对情景记忆生成“反思”（reflection）（例如“我现在喜欢滑雪”），并将这些反思写入长期语义记忆。在决策过程中，它会从语义记忆中检索相关的反思，然后通过推理制定当日的高层计划。在执行计划期间，代理会持续收到落地观测流；它可以根据这些观测进行推理，以维护或调整计划。

**Tree of Thoughts（ToT；Yao et al., 2023）** 可以被视作一种特殊的语言代理，其唯一的外部动作是向某个推理问题（例如 24 点游戏、创意写作、纵横字谜）提交最终解答。它没有长期记忆，内部动作空间中只有推理，但在决策上与之前所有代理都不同：它采用了更为审慎的决策方式。在规划过程中，ToT 通过 LLM 推理迭代地提出、评估和选择“思想”（thoughts，即推理动作），并通过树搜索算法维护这些思想，以便实现全局探索以及局部回溯和前瞻。

###### 6 可操作的洞见（Actionable Insights）

与近期一些围绕语言代理的经验性综述（Mialon et al., 2023; Weng, 2023; Wang et al., 2023b）相比，CoALA 提供了一个扎根于成熟认知架构研究的理论框架。这带来了一组独特且互补的、可操作的洞见。

**模块化代理：超越单体系统的思考（Modular agents: thinking beyond monoliths）。** 或许我们最重要的建议是，代理应当具有结构化与模块化的设计。在实践层面，就像标准化软件在不同机器人平台中被复用（Quigley, 2009; Macenski et al., 2022）一样，一个用于语言代理的框架可以整合集中的技术投入并提升兼容性。

- **在学术研究中，** 标准化术语可以支持跨工作之间的概念比较（表 2），而开源实现则能进一步促进模块的即插即用与复用。例如，马尔可夫决策过程（Markov Decision Processes, MDP；Puterman, 2014）的理论框架为强化学习（Sutton and Barto, 2018）提供了一套标准化的概念与术语（例如状态、动作、回报、转移）。与之相应，OpenAI Gym（Brockman et al., 2016）等经验框架则提供了标准化的抽象（例如 `obs, reward, done, info = env.step(action)`），从而促进 RL 经验研究。因此，现阶段实现一套适用于语言代理的有用抽象（例如 Memory、Action、Agent 类），并将较简单的代理以 CoALA 经验框架中的示例形式实现出来，以供构建更复杂代理时参考，将会十分及时且具有影响力。

- **在工业应用中，** 维护一个公司范围内统一的“语言代理库”可以通过支持测试与组件复用来减少技术债务（technical debt；Sculley et al., 2014; Lwakatare et al., 2020）。它也可以统一用户体验：与其让终端用户与由不同团队独立开发的一堆风格各异的语言代理交互，不如让他们体验到同一基础代理在不同上下文中的具体实例。

- **代理设计中 LLM 与代码的分工（LLMs vs. code in agent design）。** CoALA 代理拥有两种形式的程序性记忆：代理代码（确定性的规则）与 LLM 参数（一个庞大的、随机的产生式系统）。代理代码可解释且易扩展，但往往在面对随机性时较为脆弱，并且只能应对设计者预先考虑到的情形。相比之下，LLM 参数难以解释，却在新环境中提供了强大的零样本灵活性（Huang et al., 2022b）。因此，CoALA 建议谨慎地使用代码，只在必要时实现用以补足 LLM 局限性的通用算法，例如通过实现树搜索来缓解自回归生成导致的“短视”（myopia）（Yao et al., 2023; Hao et al., 2023）。

**代理设计：超越简单推理的思考（Agent design: thinking beyond simple reasoning）。** CoALA 从三个截然不同的概念对代理进行刻画：（i）内部记忆；（ii）一组可能的内部与外部动作；（iii）在这些动作上的决策过程。基于 CoALA 开发一个特定应用的代理，可以依次为每个组件指定具体实现。我们假设代理的环境与外部动作空间是给定的，并展示 CoALA 如何帮助确定一个合适的高层架构。例如，我们可以设想设计一个个性化零售助手（Yao et al., 2022a），用于根据用户的查询与购买历史帮助其寻找相关商品。在该例中，外部动作将包括与用户的对话或返回搜索结果。

- **确定需要哪些记忆模块。** 在零售助手的例子中，让代理拥有包含在售商品集合的语义记忆，以及关于每位顾客过往购买与交互的情景记忆，都将十分有用。它还需要程序性记忆来定义用于查询这些数据存储的函数，以及用于追踪对话状态的工作记忆。

- **定义代理的内部动作空间。** 这主要包括定义代理对各记忆模块的读写权限。在该例中，代理应当对情景记忆具有读写权限（以便存储与顾客的新交互），但对语义与程序性记忆只读（因为它不应修改库存或自身代码）。

- **定义决策过程。** 这一步指定代理如何通过推理与检索动作来选择一个外部或学习动作。一般而言，这需要在性能与泛化之间进行权衡：更复杂的过程可以更好地拟合特定问题（例如针对 Minecraft 的 Voyager；Wang et al., 2023a），而更简单的过程则更加领域无关、具有更强的可泛化性（例如 ReAct；Yao et al., 2022b）。在零售助手的例子中，我们可能希望鼓励其检索与该用户交互的情景记忆，以此提供关于其搜索意图的先验；同时增加一个显式的评估步骤，对某一组搜索结果是否能够满足这一意图进行推理。我们还可以通过将学习延迟到交互结束（Shinn et al., 2023; Park et al., 2023），在将 episode 总结后再写入情景记忆，来简化决策过程。

**结构化推理：超越提示工程的思考（Structured reasoning: thinking beyond prompt engineering）。** 早期的提示工程工作通过低层次的字符串操作操控 LLM 的输入与输出。CoALA 则建议使用更结构化的推理过程来更新工作记忆变量。

- 提示框架（prompting frameworks），例如 LangChain（LangChain, 2022）与 LlamaIndex（LlamaIndex, 2023），可以用来定义更高层的推理步骤序列，从而减轻单次 LLM 调用中的推理负担与低层提示编写负担。结构化输出解析解决方案，例如 Guidance（Guidance, 2023）与 OpenAI 的函数调用（function calling；OpenAI, 2023b），可以帮助更新工作记忆变量。定义并构建良好的工作记忆模块也将是未来研究的重要方向，尤其是对于需要让 LLM 推理与大规模代码基础设施无缝集成的工业应用而言。

- 在代理中的推理用例可以在“推理类型”（例如用于自评估、自反思、动作生成等）与“推理格式”（例如 Chain-of-Thought, CoT；Wei et al., 2022b，ReAct；Yao et al., 2022b，Reflexion；Shinn et al., 2023）两个维度上，反过来影响与重塑 LLM 的训练。默认情况下，现有 LLM 通常是面向 NLP 任务进行训练与优化的，但代理应用已经探索了诸如自评估等新的推理模式，并证明其具有广泛用途。面向这些能力进行训练或微调的 LLM，更有可能成为未来代理的“骨干模型”（backbones）。

**长期记忆：超越检索增强的思考（Long-term memory: thinking beyond retrieval augmentation）。** 传统的检索增强语言模型（retrieval-augmented language models；Guu et al., 2020; Lewis et al., 2020; Borgeaud et al., 2022）只从人类撰写的语料中读取信息，而带有记忆的语言代理则可以自主地读取与写入自生成的内容。这为高效的终身学习打开了诸多可能性。

- **结合人类已有知识与新的经验与技能** 可以帮助代理高效地进行自举（bootstrap）学习。例如，一个写代码的代理可以在语义层面上被赋予编程知识（例如手册或教科书）。随后，它可以从经验中生成自己的情景知识；再对这些经历进行反思，生成新的语义知识；并逐步在程序性记忆中构建一套代码库，存储有用的方法。

- **整合检索与推理** 有助于使规划更好地“落地”。近期的计算心理学模型指向了一个整合记忆回忆与决策的过程（Zhou et al., 2023a; Zhao et al., 2022）——这暗示着，在记忆搜索与前向模拟之间自适应地交替，将使代理更充分地利用其知识。

**学习：超越“上下文内学习”或微调的思考（Learning: thinking beyond in-context learning or finetuning）。** CoALA 对“学习”的定义涵盖了这些方法，但又进一步扩展到存储新经验或知识，或者编写新的代理代码（第 4.5 节）。重要的未来方向包括：

- **通过修改代理代码实现元学习（meta-learning）。** 这将允许代理更高效地学习。例如，学习更好的检索过程可以帮助代理更充分地利用其经验。近期的基于扩展的技术（expansion-based techniques；Nogueira et al., 2019; Wang et al., 2023c; Tang et al., 2023a）可以允许代理推理何时某些知识会有用，并将这一信息作为元数据存储起来，以便随后更易于回忆。这些形式的元学习将使代理超越人类手写代码，但由于其难度与风险，目前仍然研究较少。

- **新的学习（以及去学习）形式。** 这包括针对特定推理子任务微调较小模型（Zelikman et al., 2022; Huang et al., 2022a; Ahn et al., 2022）、删除不再需要的记忆项以实现“去学习”（Nguyen et al., 2022c），以及研究多种学习形式之间的交互效应（Tuyls et al., 2022; Park et al., 2023; Xie et al., 2023; Khattab et al., 2022）。

**动作空间：超越“外部工具或动作”的思考（Action space: thinking beyond external tools or actions）。** 虽然“动作空间”（action space）是强化学习中的标准术语，但在语言代理中却使用得相对较少。CoALA 主张定义一个清晰、适配任务的动作空间，该空间应同时包含内部动作（推理、检索、学习）与外部动作（落地），这将有助于系统化并指导代理设计。

- **动作空间的大小。** 更强大的代理（例如 Voyager、Generative Agents）拥有更大的动作空间——这反过来意味着它们面临更复杂的决策问题。因此，这些代理往往依赖更多定制化或手工设计的决策过程。在代理开发之前，必须权衡动作空间规模与决策复杂度之间的关系；对于给定任务，采取“足够但尽量小”的动作空间通常是更优的选择。

- **动作空间的安全性。** 动作空间中的某些部分在本质上风险更高。“学习”动作（尤其是程序级的删除与修改）可能会造成内部损害，而“落地”动作（例如 bash 终端中的 `rm` 命令、人机对话中的有害言论、在物理环境中握持刀具）则可能造成外部危害。当前的安全措施通常是任务特定的启发式规则（例如在 Python 中移除 `os` 操作；Chen et al., 2021，在对话中过滤敏感关键词；Chowdhery et al., 2022; Driess et al., 2023，或者将机器人限制在受控环境中；Ahn et al., 2022）。然而，随着代理被落地到更复杂的环境、内部机制也更为丰富，可能有必要对代理的动作空间进行明确的规格化与裁剪，以便进行最坏情况的预测与防范（Yao and Narasimhan, 2023）。

**决策：超越“直接生成动作”的思考（Decision making: thinking beyond action generation）。** 我们认为，决策是语言代理未来最令人兴奋的研究方向之一：如第 4.6 节所述，大多数现有工作仍然被限制在提出（或直接生成）单一动作的范式之中。目前的代理才刚刚触及更为审慎的“提出—评估—选择”决策过程的表面。

- **结合基于语言的推理与基于代码的规划** 可能同时带来两者的优势。现有方法要么直接在自然语言空间中进行规划（Huang et al., 2022c; Ahn et al., 2022），要么使用 LLM 将自然语言翻译为结构化世界模型（world models）（Wong et al., 2023; Liu et al., 2023a; Zhang et al., 2023a; Li et al., 2023a; Guan et al., 2023; Silver et al., 2022; 2023）。未来工作可以将二者整合起来：正如 Soar 中引入了物理推理模拟器（Laird, 2022），代理也可以在运行时编写并执行模拟代码，以评估计划的后果。更多讨论见第 7 节。

- **将审慎推理扩展到现实环境。** 初期工作已经实现了经典规划与树搜索（Yao et al., 2023; Hao et al., 2023; Liu et al., 2023a; Dagan et al., 2023），但多是在诸如 24 点游戏或积木搭建等玩具任务上。将这些方案扩展到具有落地（Qin et al., 2023）和长期记忆的更复杂任务上，是一个令人期待的方向。

- **通过元推理（metareasoning）提升效率。** LLM 调用既慢又计算代价高。将 LLM 用于决策意味着必须在其计算成本与由此带来的规划收益之间取得平衡。大多数 LLM 推理方法通过指定推理深度来固定搜索预算（Yao et al., 2023），但人类似乎会自适应地分配计算资源（Russek et al., 2022; Lieder and Griffiths, 2020; Callaway et al., 2022; Gershman et al., 2015）。未来工作应当开发估计规划效用的机制（Laidlaw et al., 2023），并据此调整决策过程；这可以通过摊销（amortization；例如基于之前动作结果微调 LLM；Nguyen, 2023; Hamrick et al., 2019）、在多个决策子过程中进行路由（例如 ReAct；Yao et al., 2022b 探索了在 CoT；Wei et al., 2022b 与 ReAct 之间退避与切换），或直接更新决策过程本身来实现。

- **校准与对齐（Calibration and alignment）。** 更复杂的决策目前受到诸多问题的制约，例如过度自信与校准不良（miscalibration；Jiang et al., 2021; Braverman et al., 2020; Chen et al., 2022）、与人类价值或偏见的不对齐（Liang et al., 2021; Feng et al., 2023）、在自评估中产生幻觉（hallucinations；Shinn et al., 2023），以及在存在不确定性时缺乏人类参与机制（human-in-the-loop；Nguyen et al., 2022a; Ren et al., 2023）。解决这些问题将大幅提升 LLM 作为代理骨干模型的实用性。

###### 7 讨论（Discussion）

除了前文给出的实践性洞见之外，CoALA 还引出了若干开放的概念问题。我们在此简要突出其中一些最有趣的问题，作为未来研究与讨论的重要方向。

**LLM vs. VLM：推理应当仅依赖语言还是多模态？（LLMs vs VLMs: should reasoning be language-only or multimodal?）** 大多数语言代理在决策时使用仅基于语言的模型（Yao et al., 2022b; Wang et al., 2023a; Yao et al., 2023），在必要时采用额外的图像字幕模型（captioning model）将环境观测转换为文本（Ahn et al., 2022; Zeng et al., 2022）。然而，最新一代语言模型已经是多模态的，能够接受交错的图像与文本输入（OpenAI, 2023a; Alayrac et al., 2022; Team et al., 2023; Li et al., 2023b）。基于此类多模态模型构建的语言代理可以在图像与文本输入上原生进行推理（Bavishi et al., 2023; Elsen et al., 2023; Liu et al., 2023b; Hong et al., 2023b; Driess et al., 2023），从而直接摄取感知数据并输出动作。这绕过了从图像到文本的有损转换，但也使推理与规划过程与模型输入模态紧密耦合。

从整体上看，这两种方法可以被视作使用不同的“分词方案”（tokenization scheme），来将非语言模态转换为核心推理模型的“语言”域。模块化方法（modular approach）使用一个独立的图像到文本模型，将感知数据转换为语言（Ahn et al., 2022; Zeng et al., 2022）；而集成方法（integrated approach）则将图像直接投射到语言模型的表示空间中（Bavishi et al., 2023; Elsen et al., 2023; Liu et al., 2023b）。集成式的多模态推理可能更接近人类行为：一个基于 VLM 的代理可以“看到”（see）网页，而基于 LLM 的代理更可能被直接给予原始 HTML。然而，将感知与推理系统强耦合，会使代理更具领域特异性，也更难更新。无论采用哪种方式，CoALA 所描述的基本架构原则——内部记忆、结构化动作空间与广义决策过程——都可以用来指导代理设计。

**内部 vs. 外部：代理与其环境的边界何在？（Internal vs. external: what is the boundary between an agent and its environment?）** 对于人类或机器人而言，其与具身环境的边界十分清晰，而对于数字语言代理，这一边界要模糊得多。例如，一个 Wikipedia 数据库是内部的语义记忆，还是外部的数字环境（Yao et al., 2022b）？如果一个代理在提交答案之前反复执行并改进代码（Shinn et al., 2023; Yang et al., 2023），那么代码执行应被视作内部还是外部？若某种方法由提议与评估两个提示组成（Yao et al., 2023），我们应当将其视作一个单一代理，还是两个协作的简单代理（提议者与评估者）？

我们建议使用“可控性”（controllability）与“耦合程度”（coupling）来回答这一边界问题。例如，Wikipedia 在默认情况下不可控：其他用户可能随时修改它，因此它是一个外部环境。而一个仅允许代理写入的离线版本则是可控的，因此可以被视为内部记忆。同样，在内部虚拟环境上的代码执行应当被视作内部推理动作，而在外部机器（可能存在安全漏洞）上的代码执行则应被视作外部落地动作。最后，如果代理的若干部分——例如提议与评估提示——是为彼此而设计且相互依赖的，那么它们之间的耦合程度较高，更适合作为单一代理中的组件来概念化；相反，如果这些步骤在彼此独立的情况下也很有用，则多代理（multi-agent）视角可能更为合适。尽管这些困境主要是概念性的，但对它们的理解可以支持代理设计，并帮助该领域在共享术语上达成一致。实践者也可以根据自身需要选择偏好的表述方式，只要在自己的工作中保持一致且有用即可。

**物理 vs. 数字：何种差异值得关注？（Physical vs. digital: what differences beget attention?）** 与只能在物理世界中“活一次”的动物不同，数字环境（例如 Internet）往往允许顺序（通过重置）和并行试验。这意味着数字代理可以更大胆地探索（例如打开一百万个网页），并可以自我克隆以并行求解任务（例如一百万个 web 代理尝试不同的网页路径），这可能催生与当前受人类认知启发的决策过程截然不同的决策机制（Griffiths, 2020）。

**学习 vs. 行动：代理应如何持续且自主地学习？（Learning vs. acting: how should agents continuously and autonomously learn?）** 在 CoALA 框架中，学习与落地一样，是决策周期的结果动作：代理会有意识地选择将信息写入长期记忆。这不同于大多数当前代理：它们通常固定一个学习日程，只在做外部动作时使用决策过程。生物代理却没有这样的“奢侈”：它们必须在一生中平衡学习与外部行为，在“何时学习、学习什么”之间做出取舍（Mattar and Daw, 2018）。更为灵活的语言代理（Wang et al., 2023a; Park et al., 2023）也应采取类似设计，将学习与外部动作置于同等地位。学习可以在常规决策过程中被视作一种候选动作，让代理得以在合适的时机“推迟”学习（defer learning）。

**GPT-4 vs. GPT-N：随着更强大的 LLM 出现，代理设计将如何变化？（GPT-4 vs GPT-N: how would agent design change with more powerful LLMs?）** 随着规模带来的新能力不断涌现（Wei et al., 2022a），代理设计本身是一块“移动的靶子”。例如，早期的语言模型如 GPT-2（Radford et al., 2019）还不足以支撑 LLM 代理——彼时的工作需要将 GPT-2 与强化学习结合来生成动作（Yao et al., 2020）；GPT-3（Brown et al., 2020）解锁了 NLP 任务上的灵活小样本与零样本推理；而只有到了 GPT-4（OpenAI, 2023a），才开始支持更可靠的自评估（self-evaluation；Saunders et al., 2022; Shinn et al., 2023; Yao et al., 2023）与自我改进（self-refinement；Madaan et al., 2023; Chen et al., 2023b）。未来的 LLM 会进一步减少对手写规则与额外训练模型的需求吗？这会否要求对 CoALA 框架本身进行调整？

作为一个思想实验，设想某个 GPT-N 能够在“上下文中（in context）”模拟记忆、落地、学习与决策：枚举所有可能动作，对每个动作进行模拟与评估，并在一个极长的上下文中显式维护其全部长期记忆。甚至更为大胆地设想：也许 GPT-N+1 能够在没有任何中间的显式“在上下文中推理”的情况下，仅通过在神经元中隐式模拟这些过程就生成下一步动作。尽管这些极端情形在近期内看起来不太可能，但渐进式的改进可能会改变 CoALA 各组件的重要性。例如，更长的上下文窗口可以降低长期记忆的重要性；而更强的内部评估与模拟推理能力则会允许更长视野的规划。总体而言，LLM 不受生物学限制（Griffiths, 2020），其涌现属性也一直难以预测。尽管如此，CoALA——以及更广泛的认知科学——仍然能够帮助我们整理语言代理成功或失败的任务，并建议在给定 LLM 与任务下可用于补充的基于代码的过程。即便在极端情形中，一个 GPT 已经在神经元层面实现了 CoALA 所有机制，利用 CoALA 作为一个概念向导来发现和解释这些隐式回路仍然可能是有帮助的。当然，正如第 6 节所讨论的那样，代理的使用场景也将帮助我们发现、定义并塑造 LLM 的能力。类似于芯片与计算机架构的共同演化，语言模型与代理设计也应当沿着一条相互促进的路径前进。

###### 8 结论（Conclusion）

我们提出了用于语言代理的认知架构 CoALA（Cognitive Architectures for Language Agents），这是一个用于描述与构建语言代理的概念框架。我们的框架从符号人工智能与认知科学丰富的历史中汲取灵感，将数十年的洞见与前沿的大语言模型研究连接起来。我们相信，这一路线有望为发展更通用、更类人的人工智能提供一条可行之路。

###### 致谢（Acknowledgements）

我们感谢 Harrison Chase、Baian Chen、Khanh Nguyen、Ofir Press、Noah Shinn、Jens Tuyls 对本文的校对与宝贵反馈，并感谢 Princeton NLP Group 与 Princeton Computational Cognitive Science Lab 的成员们提供的有益讨论。我们也感谢匿名审稿人提出的具有洞见的意见与建议。SY 与 KN 得到 Oracle Collaborative Research 项目和美国国家科学基金会（National Science Foundation）资助（Grant No. 2239363）。本文所表达的任何观点、研究结果、结论或建议仅代表作者个人观点，不一定反映美国国家科学基金会的立场。SY 同时得到普林斯顿大学 Harold W. Dodds 奖学金的支持。TS 得到 National Defense Science and Engineering（NDSEG）研究生奖学金项目的资助。

###### 参考文献（References）

S. Adams, I. Arel, J. Bach, R. Coop, R. Furlan, B. Goertzel, J. S. Hall, A. Samsonovich, M. Scheutz, M. Schlesinger, et al. Mapping the landscape of human-level artificial general intelligence. AI Magazine, 33(1):25–42, 2012.

M. Ahn, A. Brohan, N. Brown, Y. Chebotar, O. Cortes, B. David, C. Finn, C. Fu, K. Gopalakrishnan, K. Hausman, et al. Do as I can, not as I say: Grounding language in robotic affordances. arXiv preprint arXiv:2204.01691, 2022.

J.-B. Alayrac, J. Donahue, P. Luc, A. Miech, I. Barr, Y. Hasson, K. Lenc, A. Mensch, K. Millican, M. Reynolds, et al. Flamingo: A visual language model for few-shot learning. Advances in Neural Information Processing Systems, 35:23716–23736, 2022.

J. R. Anderson and C. Lebiere. The Newell test for a theory of cognition. Behavioral and Brain Sciences, 26(5):587–601, 2003.

J. Andreas. Language models as agent models. In Findings of the Association for Computational Linguistics: EMNLP 2022, pages 5769–5779, 2022.

R. C. Atkinson and R. M. Shiffrin. Human memory: A proposed system and its control processes. In Psychology of Learning and Motivation, volume 2, pages 89–195. Elsevier, 1968.

A. D. Baddeley and G. Hitch. Working memory. In Psychology of Learning and Motivation, volume 8, pages 47–89. Elsevier, 1974.

Y. Bai, S. Kadavath, S. Kundu, A. Askell, J. Kernion, A. Jones, A. Chen, A. Goldie, A. Mirhoseini, C. McKinnon, et al. Constitutional AI: Harmlessness from AI feedback. arXiv preprint arXiv:2212.08073, 2022.

R. Bavishi, E. Elsen, C. Hawthorne, M. Nye, A. Odena, A. Somani, and S. Taşırlar. Introducing our multimodal models, 2023. URL https://www.adept.ai/blog/fuyu-8b.

Y. Bisk, D. Marcu, and W. Wong. Towards a dataset for human computer communication via grounded language acquisition. In Workshops at the Thirtieth AAAI Conference on Artificial Intelligence, 2016.

E. Biyik and M. Palan. Asking easy questions: A user-friendly approach to active reward learning. In Proceedings of the 3rd Conference on Robot Learning, 2019.

C. Blundell, B. Uria, A. Pritzel, Y. Li, A. Ruderman, J. Z. Leibo, J. Rae, D. Wierstra, and D. Hassabis. Model-free episodic control. arXiv preprint arXiv:1606.04460, 2016.

S. Borgeaud, A. Mensch, J. Hoffmann, T. Cai, E. Rutherford, K. Millican, G. B. Van Den Driessche, J.-B. Lespiau, B. Damoc, A. Clark, et al. Improving language models by retrieving from trillions of tokens. In International Conference on Machine Learning, pages 2206–2240, 2022.

S. Branavan, D. Silver, and R. Barzilay. Learning to win by reading manuals in a Monte-Carlo framework. Journal of Artificial Intelligence Research, 43:661–704, 2012.

M. Braverman, X. Chen, S. Kakade, K. Narasimhan, C. Zhang, and Y. Zhang. Calibration, entropy rates, and memory in language models. In International Conference on Machine Learning, pages 1089–1099, 2020.

G. Brockman, V. Cheung, L. Pettersson, J. Schneider, J. Schulman, J. Tang, and W. Zaremba. OpenAI Gym, 2016.

A. Brohan, N. Brown, J. Carbajal, Y. Chebotar, J. Dabis, C. Finn, K. Gopalakrishnan, K. Hausman, A. Herzog, J. Hsu, et al. RT-1: Robotics transformer for real-world control at scale. arXiv preprint arXiv:2212.06817, 2022.

A. Brohan, N. Brown, J. Carbajal, Y. Chebotar, X. Chen, K. Choromanski, T. Ding, D. Driess, A. Dubey, C. Finn, et al. RT-2: Vision-language-action models transfer web knowledge to robotic control. arXiv preprint arXiv:2307.15818, 2023.

T. Brown, B. Mann, N. Ryder, M. Subbiah, J. D. Kaplan, P. Dhariwal, A. Neelakantan, P. Shyam, G. Sastry, A. Askell, et al. Language models are few-shot learners. Advances in Neural Information Processing Systems, 33:1877–1901, 2020.

C. B. Browne, E. Powley, D. Whitehouse, S. M. Lucas, P. I. Cowling, P. Rohlfshagen, S. Tavener, D. Perez, S. Samothrakis, and S. Colton. A survey of Monte Carlo tree search methods. IEEE Transactions on Computational Intelligence and AI in Games, 4(1):1–43, 2012.

F. Callaway, B. van Opheusden, S. Gul, P. Das, P. M. Krueger, T. L. Griffiths, and F. Lieder. Rational use of cognitive resources in human planning. Nature Human Behaviour, 6(8):1112–1125, 2022.

C.-M. Chan, W. Chen, Y. Su, J. Yu, W. Xue, S. Zhang, J. Fu, and Z. Liu. ChatEval: Towards better LLM-based evaluators through multi-agent debate. arXiv preprint arXiv:2308.07201, 2023.

B. Chen, F. Xia, B. Ichter, K. Rao, K. Gopalakrishnan, M. S. Ryoo, A. Stone, and D. Kappler. Open-vocabulary queryable scene representations for real world planning. In 2023 IEEE International Conference on Robotics and Automation (ICRA), pages 11509–11522, 2023a.

D. Chen and R. Mooney. Learning to interpret natural language navigation instructions from observations. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 25, pages 859–865, 2011.

D. Chen, A. Fisch, J. Weston, and A. Bordes. Reading Wikipedia to answer open-domain questions. arXiv preprint arXiv:1704.00051, 2017.

M. Chen, J. Tworek, H. Jun, Q. Yuan, H. P. de Oliveira Pinto, J. Kaplan, H. Edwards, Y. Burda, N. Joseph, G. Brockman, et al. Evaluating large language models trained on code. arXiv preprint arXiv:2107.03374, 2021.

X. Chen, M. Lin, N. Schärli, and D. Zhou. Teaching large language models to self-debug. arXiv preprint arXiv:2304.05128, 2023b.

Y. Chen, L. Yuan, G. Cui, Z. Liu, and H. Ji. A close look into the calibration of pre-trained language models. arXiv preprint arXiv:2211.00151, 2022.

N. Chomsky. Three models for the description of language. IRE Transactions on Information Theory, 2(3):113–124, 1956.

A. Chowdhery, S. Narang, J. Devlin, M. Bosma, G. Mishra, A. Roberts, P. Barham, H. W. Chung, C. Sutton, S. Gehrmann, et al. PaLM: Scaling language modeling with pathways. arXiv preprint arXiv:2204.02311, 2022.

P. F. Christiano, J. Leike, T. Brown, M. Martic, S. Legg, and D. Amodei. Deep reinforcement learning from human preferences. Advances in Neural Information Processing Systems, 30, 2017.

A. Church. A set of postulates for the foundation of logic. Annals of Mathematics, pages 346–366, 1932.

M.-A. Côté, A. Kádár, X. Yuan, B. Kybartas, T. Barnes, E. Fine, J. Moore, M. Hausknecht, L. El Asri, M. Adada, et al. TextWorld: A learning environment for text-based games. In Computer Games: 7th Workshop, CGW 2018, pages 41–75. Springer, 2019.

A. Creswell, M. Shanahan, and I. Higgins. Selection-inference: Exploiting large language models for interpretable logical reasoning. In The Eleventh International Conference on Learning Representations, 2023.

G. Dagan, F. Keller, and A. Lascarides. Dynamic planning with a LLM. arXiv preprint arXiv:2308.06391, 2023.

I. Dasgupta, C. Kaeser-Chen, K. Marino, A. Ahuja, S. Babayan, F. Hill, and R. Fergus. Collaborating with language models for embodied reasoning. In Second Workshop on Language and Reinforcement Learning, 2022.

X. Deng, Y. Gu, B. Zheng, S. Chen, S. Stevens, B. Wang, H. Sun, and Y. Su. Mind2Web: Towards a generalist agent for the web. arXiv preprint arXiv:2306.06070, 2023.

N. Derbinsky, J. Li, and J. Laird. A multi-domain evaluation of scaling in a general episodic memory. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 26, pages 193–199, 2012.

J. Devlin, M.-W. Chang, K. Lee, and K. Toutanova. BERT: Pre-training of deep bidirectional transformers for language understanding. In NAACL-HLT (1), 2019.

D. Dohan, W. Xu, A. Lewkowycz, J. Austin, D. Bieber, R. G. Lopes, Y. Wu, H. Michalewski, R. A. Saurous, J. Sohl-Dickstein, et al. Language model cascades. arXiv preprint arXiv:2207.10342, 2022.

Y. Dong, X. Jiang, Z. Jin, and G. Li. Self-collaboration code generation via ChatGPT. arXiv preprint arXiv:2304.07590, 2023.

D. Driess, F. Xia, M. S. Sajjadi, C. Lynch, A. Chowdhery, B. Ichter, A. Wahid, J. Tompson, Q. Vuong, T. Yu, et al. PaLM-E: An embodied multimodal language model. arXiv preprint arXiv:2303.03378, 2023.

Y. Du, S. Li, A. Torralba, J. B. Tenenbaum, and I. Mordatch. Improving factuality and reasoning in language models through multiagent debate. arXiv preprint arXiv:2305.14325, 2023.

A. Ecoffet, J. Huizinga, J. Lehman, K. O. Stanley, and J. Clune. Go-explore: A new approach for hard-exploration problems. arXiv preprint arXiv:1901.10995, 2019.

K. Ellis, C. Wong, M. Nye, M. Sablé-Meyer, L. Morales, L. Hewitt, L. Cary, A. Solar-Lezama, and J. B. Tenenbaum. DreamCoder: Bootstrapping inductive program synthesis with wake-sleep library learning. In Proceedings of the 42nd ACM SIGPLAN International Conference on Programming Language Design and Implementation, pages 835–850, 2021.

E. Elsen, A. Odena, M. Nye, S. Taşırlar, T. Dao, C. Hawthorne, D. Moparthi, and A. Somani. Releasing Persimmon-8B, 2023. URL https://www.adept.ai/blog/persimmon-8b.

S. Feng, C. Y. Park, Y. Liu, and Y. Tsvetkov. From pretraining data to language models to downstream tasks: Tracking the trails of political biases leading to unfair NLP models. arXiv preprint arXiv:2305.08283, 2023.

D. Ganguli, A. Askell, N. Schiefer, T. Liao, K. Lukošiūtė, A. Chen, A. Goldie, A. Mirhoseini, C. Olsson, D. Hernandez, et al. The capacity for moral self-correction in large language models. arXiv preprint arXiv:2302.07459, 2023.

C. Gao, X. Lan, Z. Lu, J. Mao, J. Piao, H. Wang, D. Jin, and Y. Li. S3: Social-network simulation system with large language model-empowered agents. arXiv preprint arXiv:2307.14984, 2023.

T. Gao, A. Fisch, and D. Chen. Making pre-trained language models better few-shot learners. arXiv preprint arXiv:2012.15723, 2020.

S. J. Gershman, E. J. Horvitz, and J. B. Tenenbaum. Computational rationality: A converging paradigm for intelligence in brains, minds, and machines. Science, 349(6245):273–278, 2015.

T. L. Griffiths. Understanding human intelligence through human limitations. Trends in Cognitive Sciences, 24(11):873–883, 2020.

J. Gu, Y. Wang, K. Cho, and V. O. Li. Search engine guided neural machine translation. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 32, 2018.

L. Guan, K. Valmeekam, S. Sreedharan, and S. Kambhampati. Leveraging pre-trained large language models to construct and utilize world models for model-based task planning. arXiv preprint arXiv:2305.14909, 2023.

Guidance. Guidance, 2023. URL https://github.com/guidance-ai/guidance.

I. Gur, H. Furuta, A. Huang, M. Safdari, Y. Matsuo, D. Eck, and A. Faust. A real-world web agent with planning, long context understanding, and program synthesis. arXiv preprint arXiv:2307.12856, 2023.

K. Guu, K. Lee, Z. Tung, P. Pasupat, and M. Chang. Retrieval augmented language model pre-training. In International Conference on Machine Learning, pages 3929–3938, 2020.

J. B. Hamrick, V. Bapst, A. Sanchez-Gonzalez, T. Pfaff, T. Weber, L. Buesing, and P. W. Battaglia. Combining Q-learning and search with amortized value estimates. In International Conference on Learning Representations, 2019.

A. W. Hanjie, V. Zhong, and K. Narasimhan. Grounding language to entities and dynamics for generalization in reinforcement learning. In International Conference on Machine Learning (ICML), 2021.

S. Hao, Y. Gu, H. Ma, J. J. Hong, Z. Wang, D. Z. Wang, and Z. Hu. Reasoning with language model is planning with world model. arXiv preprint arXiv:2305.14992, 2023.

M. Hasan, C. Ozel, S. Potter, and E. Hoque. Sapien: Affective virtual agents powered by large language models. arXiv preprint arXiv:2308.03022, 2023.

P. Haslum, N. Lipovetzky, D. Magazzeni, C. Muise, R. Brachman, F. Rossi, and P. Stone. An introduction to the planning domain definition language, volume 13. Springer, 2019.

M. Hausknecht, P. Ammanabrolu, M.-A. Côté, and X. Yuan. Interactive fiction games: A colossal adventure. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 34, pages 7903–7910, 2020.

S. Hong, X. Zheng, J. Chen, Y. Cheng, C. Zhang, Z. Wang, S. K. S. Yau, Z. Lin, L. Zhou, C. Ran, et al. MetaGPT: Meta programming for multi-agent collaborative framework. arXiv preprint arXiv:2308.00352, 2023a.

W. Hong, W. Wang, Q. Lv, J. Xu, W. Yu, J. Ji, Y. Wang, Z. Wang, Y. Dong, M. Ding, et al. CogAgent: A visual language model for GUI agents. arXiv preprint arXiv:2312.08914, 2023b.

J. Huang, S. S. Gu, L. Hou, Y. Wu, X. Wang, H. Yu, and J. Han. Large language models can self-improve. arXiv preprint arXiv:2210.11610, 2022a.

S. Huang, Z. Jiang, H. Dong, Y. Qiao, P. Gao, and H. Li. Instruct2Act: Mapping multi-modality instructions to robotic actions with large language model. arXiv preprint arXiv:2305.11176, 2023.

W. Huang, P. Abbeel, D. Pathak, and I. Mordatch. Language models as zero-shot planners: Extracting actionable knowledge for embodied agents. In International Conference on Machine Learning, pages 9118–9147, 2022b.

W. Huang, F. Xia, T. Xiao, H. Chan, J. Liang, P. Florence, A. Zeng, J. Tompson, I. Mordatch, Y. Chebotar, et al. Inner Monologue: Embodied reasoning through planning with language models. arXiv preprint arXiv:2207.05608, 2022c.

A. Hussein, M. M. Gaber, E. Elyan, and C. Jayne. Imitation learning: A survey of learning methods. ACM Computing Surveys (CSUR), 50(2):1–35, 2017.

G. Irving, P. Christiano, and D. Amodei. AI safety via debate. arXiv preprint arXiv:1805.00899, 2018.

G. Izacard, M. Caron, L. Hosseini, S. Riedel, P. Bojanowski, A. Joulin, and E. Grave. Unsupervised dense information retrieval with contrastive learning. arXiv preprint arXiv:2112.09118, 2021.

Z. Jiang, J. Araki, H. Ding, and G. Neubig. How can we know when language models know? On the calibration of language models for question answering. Transactions of the Association for Computational Linguistics, 9:962–977, 2021.

Z. Jin, S. Levine, F. G. Adauto, O. Kamal, M. Sap, M. Sachan, R. Mihalcea, J. B. Tenenbaum, and B. Schölkopf. When to make exceptions: Exploring language models as accounts of human moral judgment. In A. H. Oh, A. Agarwal, D. Belgrave, and K. Cho, editors, Advances in Neural Information Processing Systems, 2022.

S. Jinxin, Z. Jiabao, W. Yilei, W. Xingjiao, L. Jiawen, and H. Liang. CGMI: Configurable general multi-agent interaction framework. arXiv preprint arXiv:2308.12503, 2023.

R. M. Jones, J. E. Laird, P. E. Nielsen, K. J. Coulter, P. Kenny, and F. V. Koss. Automated intelligent pilots for combat flight simulation. AI Magazine, 20(1):27–27, 1999.

D. Jurafsky. Speech & Language Processing. Pearson Education India, 2000.

O. Khattab, K. Santhanam, X. L. Li, D. Hall, P. Liang, C. Potts, and M. Zaharia. Demonstrate-search-predict: Composing retrieval and language models for knowledge-intensive NLP. arXiv preprint arXiv:2212.14024, 2022. URL https://github.com/stanfordnlp/dspy.

G. Kim, P. Baldi, and S. McAleer. Language models can solve computer tasks. arXiv preprint arXiv:2303.17491, 2023.

J. R. Kirk and J. E. Laird. Interactive task learning for simple games. Advances in Cognitive Systems, 3(13–30):5, 2014.

J. R. Kirk, W. Robert, P. Lindes, and J. E. Laird. Improving knowledge extraction from LLMs for robotic task learning through agent analysis. arXiv preprint arXiv:2306.06770, 2023.

K. R. Koedinger, J. R. Anderson, W. H. Hadley, M. A. Mark, et al. Intelligent tutoring goes to school in the big city. International Journal of Artificial Intelligence in Education, 8(1):30–43, 1997.

T. Kojima, S. S. Gu, M. Reid, Y. Matsuo, and Y. Iwasawa. Large language models are zero-shot reasoners. Advances in Neural Information Processing Systems, 35:22199–22213, 2022.

I. Kotseruba and J. K. Tsotsos. 40 years of cognitive architectures: Core cognitive abilities and practical applications. Artificial Intelligence Review, 53(1):17–94, 2020.

C. Laidlaw, S. Russell, and A. Dragan. Bridging RL theory and practice with the effective horizon. arXiv preprint arXiv:2304.09853, 2023.

J. E. Laird. The Soar cognitive architecture. MIT Press, 2019.

J. E. Laird. Introduction to Soar. arXiv preprint arXiv:2205.03854, 2022.

J. E. Laird, P. S. Rosenbloom, and A. Newell. Chunking in Soar: The anatomy of a general learning mechanism. Machine Learning, 1:11–46, 1986.

J. E. Laird, A. Newell, and P. S. Rosenbloom. Soar: An architecture for general intelligence. Artificial Intelligence, 33(1):1–64, 1987.

J. E. Laird, K. R. Kinkade, S. Mohan, and J. Z. Xu. Cognitive robotics using the Soar cognitive architecture. In CogRob@AAAI, 2012.

B. M. Lake, T. D. Ullman, J. B. Tenenbaum, and S. J. Gershman. Building machines that learn and think like people, 2016.

LangChain. LangChain, 2022. URL http://www.langchain.com.

H. Le, Y. Wang, A. D. Gotmare, S. Savarese, and S. C. H. Hoi. CodeRL: Mastering code generation through pretrained models and deep reinforcement learning. Advances in Neural Information Processing Systems, 35:21314–21328, 2022.

Y. LeCun. A path towards autonomous machine intelligence version 0.9.2, 2022-06-27. OpenReview, 62, 2022.

P. Lewis, E. Perez, A. Piktus, F. Petroni, V. Karpukhin, N. Goyal, H. Küttler, M. Lewis, W.-t. Yih, T. Rocktäschel, et al. Retrieval-augmented generation for knowledge-intensive NLP tasks. Advances in Neural Information Processing Systems, 33:9459–9474, 2020.

B. Z. Li, W. Chen, P. Sharma, and J. Andreas. LaMPP: Language models as probabilistic priors for perception and action. arXiv preprint arXiv:2302.02801, 2023a.

C. Li, Z. Gan, Z. Yang, J. Yang, L. Li, L. Wang, and J. Gao. Multimodal foundation models: From specialists to general-purpose assistants. arXiv preprint arXiv:2309.10020, 2023b.

H. Li, Y. Su, D. Cai, Y. Wang, and L. Liu. A survey on retrieval-augmented text generation. arXiv preprint arXiv:2202.01110, 2022a.

R. Li, L. B. Allal, Y. Zi, N. Muennighoff, D. Kocetkov, C. Mou, M. Marone, C. Akiki, J. Li, J. Chim, Q. Liu, E. Zheltonozhskii, T. Y. Zhuo, T. Wang, O. Dehaene, M. Davaadorj, J. Lamy-Poirier, J. Monteiro, O. Shliazhko, N. Gontier, N. Meade, A. Zebaze, M.-H. Yee, L. K. Umapathi, J. Zhu, B. Lipkin, M. Oblokulov, Z. Wang, R. Murthy, J. Stillerman, S. S. Patel, D. Abulkhanov, M. Zocca, M. Dey, Z. Zhang, N. Fahmy, U. Bhattacharyya, W. Yu, S. Singh, S. Luccioni, P. Villegas, M. Kunakov, F. Zhdanov, M. Romero, T. Lee, N. Timor, J. Ding, C. Schlesinger, H. Schoelkopf, J. Ebert, T. Dao, M. Mishra, A. Gu, J. Robinson, C. J. Anderson, B. Dolan-Gavitt, D. Contractor, S. Reddy, D. Fried, D. Bahdanau, Y. Jernite, C. M. Ferrandis, S. M. Hughes, T. Wolf, A. Guha, L. von Werra, and H. de Vries. StarCoder: May the source be with you! arXiv preprint arXiv:2305.06161, 2023c.

Y. Li, D. H. Choi, J. Chung, N. Kushman, J. Schrittwieser, R. Leblond, T. Eccles, J. Keeling, F. Gimeno, A. D. Lago, T. Hubert, P. Choy, C. de Masson d’Autume, I. Babuschkin, X. Chen, P.-S. Huang, J. Welbl, S. Gowal, A. Cherepanov, J. Molloy, D. J. Mankowitz, E. S. Robson, P. Kohli, N. de Freitas, K. Kavukcuoglu, and O. Vinyals. Competition-level code generation with AlphaCode. Science, 378:1092–1097, 2022b.

J. Liang, W. Huang, F. Xia, P. Xu, K. Hausman, B. Ichter, P. Florence, and A. Zeng. Code as policies: Language model programs for embodied control. In 2023 IEEE International Conference on Robotics and Automation (ICRA), pages 9493–9500, 2023a.

P. P. Liang, C. Wu, L.-P. Morency, and R. Salakhutdinov. Towards understanding and mitigating social biases in language models. In International Conference on Machine Learning, pages 6565–6576, 2021.

T. Liang, Z. He, W. Jiao, X. Wang, Y. Wang, R. Wang, Y. Yang, Z. Tu, and S. Shi. Encouraging divergent thinking in large language models through multi-agent debate. arXiv preprint arXiv:2305.19118, 2023b.

F. Lieder and T. L. Griffiths. Resource-rational analysis: Understanding human cognition as the optimal use of limited computational resources. Behavioral and Brain Sciences, 43:e1, 2020.

B. Y. Lin, Y. Fu, K. Yang, P. Ammanabrolu, F. Brahman, S. Huang, C. Bhagavatula, Y. Choi, and X. Ren. SwiftSage: A generative agent with fast and slow thinking for complex interactive tasks. arXiv preprint arXiv:2305.17390, 2023.

P. Lindes and J. E. Laird. Toward integrating cognitive linguistics and cognitive language processing. In Proceedings of the 14th International Conference on Cognitive Modeling (ICCM), 2016.

B. Liu, Y. Jiang, X. Zhang, Q. Liu, S. Zhang, J. Biswas, and P. Stone. LLM+P: Empowering large language models with optimal planning proficiency. arXiv preprint arXiv:2304.11477, 2023a.

H. Liu, C. Li, Q. Wu, and Y. J. Lee. Visual instruction tuning. In NeurIPS, 2023b.

H. Liu, C. Sferrazza, and P. Abbeel. Languages are rewards: Hindsight finetuning using human feedback. arXiv preprint arXiv:2302.02676, 2023c.

J. Liu, D. Shen, Y. Zhang, B. Dolan, L. Carin, and W. Chen. What makes good in-context examples for GPT-3? arXiv preprint arXiv:2101.06804, 2021.

P. Liu, W. Yuan, J. Fu, Z. Jiang, H. Hayashi, and G. Neubig. Pre-train, prompt, and predict: A systematic survey of prompting methods in natural language processing. ACM Computing Surveys, 55(9), 2023d.

R. Liu, J. Wei, S. S. Gu, T.-Y. Wu, S. Vosoughi, C. Cui, D. Zhou, and A. M. Dai. Mind’s Eye: Grounded language model reasoning through simulation. In The Eleventh International Conference on Learning Representations, 2023e.

R. Liu, R. Yang, C. Jia, G. Zhang, D. Zhou, A. M. Dai, D. Yang, and S. Vosoughi. Training socially aligned language models in simulated human society. arXiv preprint arXiv:2305.16960, 2023f.

LlamaIndex. LlamaIndex, 2023. URL http://www.llamaindex.ai.

L. E. Lwakatare, A. Raj, I. Crnkovic, J. Bosch, and H. H. Olsson. Large-scale machine learning systems in real-world industrial settings: A review of challenges and solutions. Information and Software Technology, 127:106368, 2020.

Z. Ma, Y. Mei, and Z. Su. Understanding the benefits and challenges of using large language model-based conversational agents for mental well-being support. arXiv preprint arXiv:2307.15810, 2023.

S. Macenski, T. Foote, B. Gerkey, C. Lalancette, and W. Woodall. Robot Operating System 2: Design, architecture, and uses in the wild. Science Robotics, 7(66):eabm6074, 2022.

A. Madaan, N. Tandon, P. Gupta, S. Hallinan, L. Gao, S. Wiegreffe, U. Alon, N. Dziri, S. Prabhumoye, Y. Yang, et al. Self-Refine: Iterative refinement with self-feedback. arXiv preprint arXiv:2303.17651, 2023.

A. A. Markov. The theory of algorithms. Trudy Matematicheskogo Instituta Imeni V. A. Steklova, 42:3–375, 1954.

M. G. Mattar and N. D. Daw. Prioritized memory access explains planning and hippocampal replay. Nature Neuroscience, 21(11):1609–1617, 2018.

J. L. McClelland, F. Hill, M. Rudolph, J. Baldridge, and H. Schütze. Extending machine language models toward human-level language understanding. arXiv preprint arXiv:1912.05877, 2019.

J. Meier, R. Rao, R. Verkuil, J. Liu, T. Sercu, and A. Rives. Language models enable zero-shot prediction of the effects of mutations on protein function. bioRxiv, 2021.

G. Mialon, R. Dessì, M. Lomeli, C. Nalmpantis, R. Pasunuru, R. Raileanu, B. Rozière, T. Schick, J. Dwivedi-Yu, A. Celikyilmaz, et al. Augmented language models: A survey. arXiv preprint arXiv:2302.07842, 2023.

S. Mohan and J. Laird. Learning goal-oriented hierarchical tasks from situated interactive instruction. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 28, 2014.

S. Mohan, A. H. Mininger, J. R. Kirk, and J. E. Laird. Acquiring grounded representations of words with situated interactive instruction. Advances in Cognitive Systems, 2:113–130, 2012.

R. Nakano, J. Hilton, S. Balaji, J. Wu, L. Ouyang, C. Kim, C. Hesse, S. Jain, V. Kosaraju, W. Saunders, et al. WebGPT: Browser-assisted question-answering with human feedback. arXiv preprint arXiv:2112.09332, 2021.

K. Narasimhan, R. Barzilay, and T. Jaakkola. Deep transfer in reinforcement learning by language grounding. Journal of Artificial Intelligence Research (JAIR), 2018.

A. Narayan-Chen, P. Jayannavar, and J. Hockenmaier. Collaborative dialogue in Minecraft. In Proceedings of the 57th Annual Meeting of the Association for Computational Linguistics, pages 5405–5415. Association for Computational Linguistics, 2019.

S. Nason and J. E. Laird. Soar-RL: Integrating reinforcement learning with Soar. Cognitive Systems Research, 6(1):51–59, 2005.

A. Newell. Studies in problem solving: Subject 3 on the crypt-arithmetic task DONALD + GERALD = ROBERT. Technical Report, Carnegie Mellon University, 1967.

A. Newell. Physical symbol systems. Cognitive Science, 4(2):135–183, 1980.

A. Newell. Précis of unified theories of cognition. Behavioral and Brain Sciences, 15(3):425–437, 1992.

A. Newell and H. A. Simon. Human Problem Solving. Prentice-Hall, 1972.

A. Newell, P. S. Rosenbloom, and J. E. Laird. Symbolic architectures for cognition. Foundations of Cognitive Science, pages 93–131, 1989.

K. Nguyen and H. Daumé III. Help, Anna! Visual navigation with natural multimodal assistance via retrospective curiosity-encouraging imitation learning. arXiv preprint arXiv:1909.01871, 2019.

K. Nguyen, D. Dey, C. Brockett, and B. Dolan. Vision-based navigation with language-based assistance via imitation learning with indirect intervention. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 12527–12537, 2019.

K. Nguyen, Y. Bisk, and H. Daumé III. A framework for learning to request rich and contextually useful information from humans. In ICML, July 2022a.

K. X. Nguyen. Language models are bounded pragmatic speakers. In First Workshop on Theory of Mind in Communicating Agents, 2023.

K. X. Nguyen, D. Misra, R. Schapire, M. Dudík, and P. Shafto. Interactive learning from activity description. In International Conference on Machine Learning, pages 8096–8108, 2021.

K. X. Nguyen, Y. Bisk, and H. D. III. A framework for learning to request rich and contextually useful information from humans. In International Conference on Machine Learning, pages 16553–16568, 2022b.

T. T. Nguyen, T. T. Huynh, P. L. Nguyen, A. W.-C. Liew, H. Yin, and Q. V. H. Nguyen. A survey of machine unlearning. arXiv preprint arXiv:2209.02299, 2022c.

A. Ni, S. Iyer, D. Radev, V. Stoyanov, W.-t. Yih, S. Wang, and X. V. Lin. LEVER: Learning to verify language-to-code generation with execution. In International Conference on Machine Learning, pages 26106–26128, 2023.

N. J. Nilsson. Shakey the robot. Technical Note, 1984.

R. Nogueira, W. Yang, J. Lin, and K. Cho. Document expansion by query prediction, 2019.

A. M. Nuxoll and J. E. Laird. Extending cognitive architecture with episodic memory. In Proceedings of the AAAI Conference on Artificial Intelligence, pages 1560–1564, 2007.

M. Nye, A. J. Andreassen, G. Gur-Ari, H. Michalewski, J. Austin, D. Bieber, D. Dohan, A. Lewkowycz, M. Bosma, D. Luan, et al. Show your work: Scratchpads for intermediate computation with language models. arXiv preprint arXiv:2112.00114, 2021.

OpenAI. GPT-4 Technical Report. arXiv, abs/2303.08774, 2023a.

OpenAI. Function calling and other API updates, 2023b. URL https://openai.com/blog/function-calling-and-other-api-updates.

L. Ouyang, J. Wu, X. Jiang, D. Almeida, C. Wainwright, P. Mishkin, C. Zhang, S. Agarwal, K. Slama, A. Ray, et al. Training language models to follow instructions with human feedback. Advances in Neural Information Processing Systems, 35:27730–27744, 2022.

A. Padmakumar, J. Thomason, A. Shrivastava, P. Lange, A. Narayan-Chen, S. Gella, R. Piramuthu, G. Tur, and D. Hakkani-Tur. TEACH: Task-driven embodied agents that chat. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 36, pages 2017–2025, 2022.

N. D. Palo, A. Byravan, L. Hasenclever, M. Wulfmeier, N. Heess, and M. Riedmiller. Towards a unified agent with foundation models. In Workshop on Reincarnating Reinforcement Learning at ICLR 2023, 2023.

A. Parisi, Y. Zhao, and N. Fiedel. TaLM: Tool augmented language models. arXiv preprint arXiv:2205.12255, 2022.

J. S. Park, J. C. O’Brien, C. J. Cai, M. R. Morris, P. Liang, and M. S. Bernstein. Generative agents: Interactive simulacra of human behavior. arXiv preprint arXiv:2304.03442, 2023.

P. Pataranutaporn, V. Danry, J. Leong, P. Punpongsanon, D. Novy, P. Maes, and M. Sra. AI-generated characters for supporting personalized learning and well-being. Nature Machine Intelligence, 3(12):1013–1022, 2021.

A. Peng, I. Sucholutsky, B. Li, T. R. Sumers, T. L. Griffiths, J. Andreas, and J. A. Shah. Language guided state abstractions. In Workshop on Social Intelligence in Humans and Robots at RSS 2023, 2023.

E. L. Post. Formal reductions of the general combinatorial decision problem. American Journal of Mathematics, 65(2):197–215, 1943.

A. Pritzel, B. Uria, S. Srinivasan, A. P. Badia, O. Vinyals, D. Hassabis, D. Wierstra, and C. Blundell. Neural episodic control. In International Conference on Machine Learning, pages 2827–2836, 2017.

M. L. Puterman. Markov Decision Processes: Discrete Stochastic Dynamic Programming. John Wiley & Sons, 2014.

C. Qian, X. Cong, C. Yang, W. Chen, Y. Su, J. Xu, Z. Liu, and M. Sun. Communicative agents for software development. arXiv preprint arXiv:2307.07924, 2023.

Y. Qin, S. Liang, Y. Ye, K. Zhu, L. Yan, Y. Lu, Y. Lin, X. Cong, X. Tang, B. Qian, et al. ToolLLM: Facilitating large language models to master 16000+ real-world APIs. arXiv preprint arXiv:2307.16789, 2023.

M. Quigley. ROS: An open-source robot operating system. In IEEE International Conference on Robotics and Automation, 2009. URL https://api.semanticscholar.org/CorpusID:6324125.

A. Radford, J. Wu, R. Child, D. Luan, D. Amodei, I. Sutskever, et al. Language models are unsupervised multitask learners. OpenAI Blog, 1(8):9, 2019.

A. Z. Ren, A. Dixit, A. Bodrova, S. Singh, S. Tu, N. Brown, P. Xu, L. Takayama, F. Xia, Z. Xu, et al. Robots that ask for help: Uncertainty alignment for large language model planners. In 7th Annual Conference on Robot Learning, 2023.

O. J. Romero, J. Zimmerman, A. Steinfeld, and A. Tomasic. Synergistic integration of large language models and cognitive architectures for robust AI: An exploratory analysis. arXiv preprint arXiv:2308.09830, 2023.

B. Rozière, J. Gehring, F. Gloeckle, S. Sootla, I. Gat, X. Tan, Y. Adi, J. Liu, T. Remez, J. Rapin, A. Kozhevnikov, I. Evtimov, J. Bitton, M. P. Bhatt, C. C. Ferrer, A. Grattafiori, W. Xiong, A. D’efossez, J. Copet, F. Azhar, H. Touvron, L. Martin, N. Usunier, T. Scialom, and G. Synnaeve. Code Llama: Open foundation models for code. arXiv preprint arXiv:2308.12950, 2023.

O. Rubin, J. Herzig, and J. Berant. Learning to retrieve prompts for in-context learning. arXiv preprint arXiv:2112.08633, 2021.

E. Russek, D. Acosta-Kane, B. van Opheusden, M. G. Mattar, and T. Griffiths. Time spent thinking in online chess reflects the value of computation. PsyArXiv, 2022.

S. Russell and P. Norvig. Artificial Intelligence: A Modern Approach. Pearson Education Limited, London, 2013.

D. Sadigh, A. D. Dragan, S. Sastry, and S. A. Seshia. Active preference-based learning of reward functions. In N. M. Amato, S. S. Srinivasa, N. Ayanian, and S. Kuindersma, editors, Robotics: Science and Systems XIII, 2017.

W. Saunders, C. Yeh, J. Wu, S. Bills, L. Ouyang, J. Ward, and J. Leike. Self-critiquing models for assisting human evaluators. arXiv preprint arXiv:2206.05802, 2022.

T. Schick, J. Dwivedi-Yu, R. Dessì, R. Raileanu, M. Lomeli, L. Zettlemoyer, N. Cancedda, and T. Scialom. Toolformer: Language models can teach themselves to use tools. arXiv preprint arXiv:2302.04761, 2023.

D. Sculley, G. Holt, D. Golovin, E. Davydov, T. Phillips, D. Ebner, V. Chaudhary, and M. Young. Machine learning: The high interest credit card of technical debt. In SE4ML: Software Engineering for Machine Learning (NIPS 2014 Workshop), 2014.

T. Shi, A. Karpathy, L. Fan, J. Hernandez, and P. Liang. World of Bits: An open-domain platform for web-based agents. In International Conference on Machine Learning, pages 3135–3144, 2017.

N. Shinn, F. Cassano, B. Labash, A. Gopinath, K. Narasimhan, and S. Yao. Reflexion: Language agents with verbal reinforcement learning. arXiv preprint arXiv:2303.11366, 2023.

M. Shridhar, X. Yuan, M.-A. Côté, Y. Bisk, A. Trischler, and M. Hausknecht. AlfWorld: Aligning text and embodied environments for interactive learning. arXiv preprint arXiv:2010.03768, 2020.

T. Silver, V. Hariprasad, R. S. Shuttleworth, N. Kumar, T. Lozano-Pérez, and L. P. Kaelbling. PDDL planning with pretrained large language models. In NeurIPS 2022 Foundation Models for Decision Making Workshop, 2022.

T. Silver, S. Dan, K. Srinivas, J. B. Tenenbaum, L. P. Kaelbling, and M. Katz. Generalized planning in PDDL domains with pretrained large language models. arXiv preprint arXiv:2305.11014, 2023.

I. Singh, V. Blukis, A. Mousavian, A. Goyal, D. Xu, J. Tremblay, D. Fox, J. Thomason, and A. Garg. ProgPrompt: Generating situated robot task plans using large language models. In 2023 IEEE International Conference on Robotics and Automation (ICRA), pages 11523–11530, 2023.

T. Sumers, R. Hawkins, M. K. Ho, T. Griffiths, and D. Hadfield-Menell. How to talk so AI will learn: Instructions, descriptions, and autonomy. Advances in Neural Information Processing Systems, 35:34762–34775, 2022.

T. Sumers, K. Marino, A. Ahuja, R. Fergus, and I. Dasgupta. Distilling internet-scale vision-language models into embodied agents. In Proceedings of the 40th International Conference on Machine Learning, pages 32797–32818, 2023.

T. R. Sumers, M. K. Ho, R. D. Hawkins, K. Narasimhan, and T. L. Griffiths. Learning rewards from linguistic feedback. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 35, pages 6002–6010, 2021.

R. Sun. Desiderata for cognitive architectures. Philosophical Psychology, 17(3):341–373, 2004.

R. S. Sutton and A. G. Barto. Reinforcement Learning: An Introduction. MIT Press, 2018.

O. Tafjord, B. Dalvi, and P. Clark. ProofWriter: Generating implications, proofs, and abductive statements over natural language. In Findings of the Association for Computational Linguistics: ACL-IJCNLP 2021, pages 3621–3634, 2021.

R. Tamari, C. Shani, T. Hope, M. R. L. Petruck, O. Abend, and D. Shahaf. Language (re)modelling: Towards embodied language understanding. In Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics, pages 6268–6281, Online, July 2020. Association for Computational Linguistics.

M. Tambe, W. L. Johnson, R. M. Jones, F. Koss, J. E. Laird, P. S. Rosenbloom, and K. Schwamb. Intelligent agents for interactive simulation environments. AI Magazine, 16(1):15–15, 1995.

M. Tang, S. Yao, J. Yang, and K. Narasimhan. Referral augmentation for zero-shot information retrieval, 2023a.

Q. Tang, Z. Deng, H. Lin, X. Han, Q. Liang, and L. Sun. ToolAlpaca: Generalized tool learning for language models with 3000 simulated cases. arXiv preprint arXiv:2306.05301, 2023b.

G. Team, R. Anil, S. Borgeaud, Y. Wu, J.-B. Alayrac, J. Yu, R. Soricut, J. Schalkwyk, A. M. Dai, A. Hauth, et al. Gemini: A family of highly capable multimodal models. arXiv preprint arXiv:2312.11805, 2023.

S. Tellex, T. Kollar, S. Dickerson, M. Walter, A. Banerjee, S. Teller, and N. Roy. Understanding natural language commands for robotic navigation and mobile manipulation. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 25, pages 1507–1514, 2011.

J. Thomason, M. Murray, M. Cakmak, and L. Zettlemoyer. Vision-and-dialog navigation. In Conference on Robot Learning, pages 394–406. PMLR, 2020.

A. M. Turing et al. On computable numbers, with an application to the Entscheidungsproblem. Journal of Math, 58(345–363):5, 1936.

J. Tuyls, S. Yao, S. Kakade, and K. Narasimhan. Multi-stage episodic control for strategic exploration in text games. arXiv preprint arXiv:2201.01251, 2022.

K. Valmeekam, A. Olmo, S. Sreedharan, and S. Kambhampati. Large language models still can’t plan (A benchmark for LLMs on planning and reasoning about change). arXiv preprint arXiv:2206.10498, 2022.

A. Vaswani, N. Shazeer, N. Parmar, J. Uszkoreit, L. Jones, A. N. Gomez, Ł. Kaiser, and I. Polosukhin. Attention is all you need. Advances in Neural Information Processing Systems, 30, 2017.

G. Wang, Y. Xie, Y. Jiang, A. Mandlekar, C. Xiao, Y. Zhu, L. Fan, and A. Anandkumar. Voyager: An open-ended embodied agent with large language models. arXiv preprint arXiv:2305.16291, 2023a.

L. Wang, C. Ma, X. Feng, Z. Zhang, H. Yang, J. Zhang, Z. Chen, J. Tang, X. Chen, Y. Lin, W. X. Zhao, Z. Wei, and J.-R. Wen. A survey on large language model based autonomous agents, 2023b.

L. Wang, N. Yang, and F. Wei. Query2Doc: Query expansion with large language models. arXiv preprint arXiv:2303.07678, 2023c.

R. Wang, P. Jansen, M.-A. Côté, and P. Ammanabrolu. ScienceWorld: Is your agent smarter than a 5th grader? arXiv preprint arXiv:2203.07540, 2022a.

S. I. Wang, P. Liang, and C. D. Manning. Learning language games through interaction. In Proceedings of the 54th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 2368–2378, 2016.

X. Wang, J. Wei, D. Schuurmans, Q. Le, E. Chi, and D. Zhou. Self-consistency improves chain of thought reasoning in language models. arXiv preprint arXiv:2203.11171, 2022b.

J. Wei, Y. Tay, R. Bommasani, C. Raffel, B. Zoph, S. Borgeaud, D. Yogatama, M. Bosma, D. Zhou, D. Metzler, E. H. Chi, T. Hashimoto, O. Vinyals, P. Liang, J. Dean, and W. Fedus. Emergent abilities of large language models. Transactions on Machine Learning Research, 2022a.

J. Wei, X. Wang, D. Schuurmans, M. Bosma, E. Chi, Q. Le, and D. Zhou. Chain-of-Thought prompting elicits reasoning in large language models. arXiv preprint arXiv:2201.11903, 2022b.

L. Weng. LLM-powered autonomous agents. lilianweng.github.io, Jun 2023. URL https://lilianweng.github.io/posts/2023-06-23-agent/.

J. Weston, S. Chopra, and A. Bordes. Memory networks. arXiv preprint arXiv:1410.3916, 2014.

A. N. Whitehead and B. Russell. Principia Mathematica to *56, volume 2. Cambridge University Press, 1997.

D. E. Wilkins. Practical Planning: Extending the Classical AI Planning Paradigm. Elsevier, 2014.

T. Winograd. Understanding natural language. Cognitive Psychology, 3(1):1–191, 1972.

L. Wong, G. Grand, A. K. Lew, N. D. Goodman, V. K. Mansinghka, J. Andreas, and J. B. Tenenbaum. From word models to world models: Translating from natural language to the probabilistic language of thought. arXiv preprint arXiv:2306.12672, 2023.

R. E. Wray, J. R. Kirk, J. E. Laird, et al. Language models as a knowledge source for cognitive agents. arXiv preprint arXiv:2109.08270, 2021.

Q. Wu, G. Bansal, J. Zhang, Y. Wu, S. Zhang, E. Zhu, B. Li, L. Jiang, X. Zhang, and C. Wang. AutoGen: Enabling next-gen LLM applications via multi-agent conversation framework. arXiv preprint arXiv:2308.08155, 2023.

T. Wu, E. Jiang, A. Donsbach, J. Gray, A. Molina, M. Terry, and C. J. Cai. PromptChainer: Chaining large language model prompts through visual programming. In CHI Conference on Human Factors in Computing Systems Extended Abstracts, pages 1–10, 2022a.

T. Wu, M. Terry, and C. J. Cai. AI Chains: Transparent and controllable human-AI interaction by chaining large language model prompts. In Proceedings of the 2022 CHI Conference on Human Factors in Computing Systems, pages 1–22, 2022b.

Z. Xi, W. Chen, X. Guo, W. He, Y. Ding, B. Hong, M. Zhang, J. Wang, S. Jin, E. Zhou, et al. The rise and potential of large language model based agents: A survey. arXiv preprint arXiv:2309.07864, 2023.

Y. Xia, M. Shenoy, N. Jazdi, and M. Weyrich. Towards autonomous system: Flexible modular production system enhanced with large language model agents. arXiv preprint arXiv:2304.14721, 2023.

Y. Xie, T. Xie, M. Lin, W. Wei, C. Li, B. Kong, L. Chen, C. Zhuo, B. Hu, and Z. Li. OLaGPT: Empowering LLMs with human-like problem-solving abilities. arXiv preprint arXiv:2305.16334, 2023.

B. Xu, X. Liu, H. Shen, Z. Han, Y. Li, M. Yue, Z. Peng, Y. Liu, Z. Yao, and D. Xu. Gentopia: A collaborative platform for tool-augmented LLMs. arXiv preprint arXiv:2308.04030, 2023a.

B. Xu, Z. Peng, B. Lei, S. Mukherjee, Y. Liu, and D. Xu. ReWOO: Decoupling reasoning from observations for efficient augmented language models. arXiv preprint arXiv:2305.18323, 2023b.

B. Xu, A. Yang, J. Lin, Q. Wang, C. Zhou, Y. Zhang, and Z. Mao. ExpertPrompting: Instructing large language models to be distinguished experts. arXiv preprint arXiv:2305.14688, 2023c.

J. Yang, A. Prabhakar, K. Narasimhan, and S. Yao. InterCode: Standardizing and benchmarking interactive coding with execution feedback. arXiv preprint arXiv:2306.14898, 2023.

S. Yao and K. Narasimhan. Language agents in the digital world: Opportunities and risks. princeton-nlp.github.io, Jul 2023. URL https://princeton-nlp.github.io/language-agent-impact/.

S. Yao, R. Rao, M. Hausknecht, and K. Narasimhan. Keep CALM and explore: Language models for action generation in text-based games. arXiv preprint arXiv:2010.02903, 2020.

S. Yao, H. Chen, J. Yang, and K. Narasimhan. WebShop: Towards scalable real-world web interaction with grounded language agents. Advances in Neural Information Processing Systems, 35:20744–20757, 2022a.

S. Yao, J. Zhao, D. Yu, N. Du, I. Shafran, K. Narasimhan, and Y. Cao. ReAct: Synergizing reasoning and acting in language models. arXiv preprint arXiv:2210.03629, 2022b.

S. Yao, D. Yu, J. Zhao, I. Shafran, T. L. Griffiths, Y. Cao, and K. Narasimhan. Tree of Thoughts: Deliberate problem solving with large language models. arXiv preprint arXiv:2305.10601, 2023.

E. Zelikman, Y. Wu, J. Mu, and N. Goodman. STaR: Bootstrapping reasoning with reasoning. Advances in Neural Information Processing Systems, 35:15476–15488, 2022.

A. Zeng, M. Attarian, B. Ichter, K. Choromanski, A. Wong, S. Welker, F. Tombari, A. Purohit, M. Ryoo, V. Sindhwani, et al. Socratic models: Composing zero-shot multimodal reasoning with language. arXiv preprint arXiv:2204.00598, 2022.

C. Zhang, L. Wong, G. Grand, and J. Tenenbaum. Grounded physical language understanding with probabilistic programs and simulated worlds. In Proceedings of the Annual Meeting of the Cognitive Science Society, volume 45, 2023a.

T. Zhang, F. Liu, J. Wong, P. Abbeel, and J. E. Gonzalez. The wisdom of hindsight makes language models better instruction followers. arXiv preprint arXiv:2302.05206, 2023b.

Y. Zhang, S. Sun, M. Galley, Y.-C. Chen, C. Brockett, X. Gao, J. Gao, J. Liu, and W. B. Dolan. DialoGPT: Large-scale generative pre-training for conversational response generation. In Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics: System Demonstrations, pages 270–278, 2020.

W. J. Zhao, R. Richie, and S. Bhatia. Process and content in decisions from memory. Psychological Review, 129(1):73, 2022.

V. Zhong, A. W. Hanjie, S. Wang, K. Narasimhan, and L. Zettlemoyer. SILG: The multi-domain symbolic interactive language grounding benchmark. Advances in Neural Information Processing Systems, 34:21505–21519, 2021.

C. Y. Zhou, D. Talmi, N. Daw, and M. G. Mattar. Episodic retrieval for model-based evaluation in sequential decision tasks, 2023a.

H. Zhou, M. Huang, T. Zhang, X. Zhu, and B. Liu. Emotional chatting machine: Emotional conversation generation with internal and external memory. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 32, 2018.

S. Zhou, U. Alon, F. F. Xu, Z. Jiang, and G. Neubig. DocPrompting: Generating code by retrieving the docs. In The Eleventh International Conference on Learning Representations, 2022a.

S. Zhou, F. F. Xu, H. Zhu, X. Zhou, R. Lo, A. Sridhar, X. Cheng, Y. Bisk, D. Fried, U. Alon, et al. WebArena: A realistic web environment for building autonomous agents. arXiv preprint arXiv:2307.13854, 2023b.

Y. Zhou, A. I. Muresanu, Z. Han, K. Paster, S. Pitis, H. Chan, and J. Ba. Large language models are human-level prompt engineers. arXiv preprint arXiv:2211.01910, 2022b.
