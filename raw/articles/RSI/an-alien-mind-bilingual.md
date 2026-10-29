# 异己之心 / An Alien Mind（中英对照）

> **原文出处 / Source:** [An Alien Mind — OpenAI, September 6, 2026](https://openai.com/index/an-alien-mind/)
> **作者 / By:** Jakub Pachocki, Chief Scientist at OpenAI（OpenAI 首席科学家 Jakub Pachocki）
> **译者说明：** 本文档为个人学习用的双语对照版（全部中文在前、全部英文在后）。所有英文原文版权归 OpenAI 所有；中文翻译仅作研究、学习目的参考，请以原文为准。

---

# 一、中文全文

## 引言

2023 年年中，在名为 "RLSlow" 的研究项目里，我们看到了第一批让我们相信可以规模化训练推理模型的结果——这将解锁预训练模型自主形成思维链的能力。那天晚上，Szymon 和我都留在办公室，想的并不是这项技术将带来的惊人基准测试成绩、产品或科研成果，而是在努力消化一个令人警醒的事实：在我们有生之年，真的会看到实质上比我们更聪明的机器，而且这类系统的轮廓已经显现；我们思考的是，该如何让人们意识到此事的重要性。

三年后，推理语言模型正在成为经济体系中一个快速增长的组成部分，并开始拓展科学的边界。它们可以操作电脑与图形界面，可以与人以及彼此协作，可以推进研究项目。它们也正在重塑计算机安全的格局，并因此带来了显而易见的新型危险。

这段时间里出现了大量新的研究，我们对这类系统的理解与 2023 年时又有了一些不同。根据内部结果，我强烈预期，这样的进展速度有可能一直持续到递归式自我改进（recursive self-improvement）阶段。如果 AI 的发展沿着当前的路径继续，我们在未来几年看到的系统很可能会带来同等甚至更大幅度的能力跃迁，并且越来越多地驱动自身的发展。

这是一个需要极度谨慎的时刻。我担心，对于机器智能持续快速上升所带来的后果，没有人真正做好准备。OpenAI 会继续寻求对齐（alignment）与监控（monitoring）方面的技术方案，构建防御体系，并在必要时单方面停止进一步的规模化推进；但我认为，更大范围的干预同样必不可少。

## 我们并未完全理解的智能

从宏观视角看，机器智能的进步由不断增长的算力驱动。大约在 2017 年，我们在多个研究项目中反复看到"规模化"能持续带来回报，OpenAI 内部由此彻底接受了这一观点。此后我们主动去争取远超原计划的算力，并把研究越来越聚焦在少数几个具备高度可扩展性的方向上。当时我们认为，这是让我们站在 AI 研究前沿、并塑造 AGI 对世界之影响的唯一途径。

一路走来，也出现了不少新的算法，团队与个人研究者的创造力屡有闪光。我主要把它们视为规模化道路上取得的发现——深度学习科学仍处于萌芽阶段，而真正有意义的算法进步往往与可用算力相关。如果你把时间尺度拉到几年这个跨度，就会看到：随着 AI 被扩展到更大规模的计算机上，它仍在持续变得更聪明。

正如 Ray Kurzweil 在 20 世纪末所预测的那样，我们正处在计算史上一个关键节点：机器智能开始以足以带来深刻变革的方式超越人类智能。

AI 与其说是**被"设计"出来**的，不如说是**被"培育"出来**的——大体而言，它是依靠难以想象的巨量算力，反复执行一个简单优化步骤的产物。结果是一个极其复杂的系统，它借助抽象概念思考，并能模拟人类行为的诸多方面。我们可以像做神经科学那样，去逐步揭示系统内部涌现出来的一些"小机制"的原理；而与神经科学类似的是，系统整体的行为方式，仍然无法被我们完整地描述与理解。

基于深度学习的 AI 研究在很大程度上是一门实验科学。我们投入了大量精力去构建有理论依据的算法、去做可检验的预测，但归根结底，我们的大规模训练本质上是**实验**，其结果有时会让我们大吃一惊。更麻烦的是，随着系统能力变强，这些结果变得越来越难以解读。

让情况更复杂的是，当前的算法总体上会让"容易度量的能力"提升得比"难以客观量化的能力"更快。我们会花大量时间去理解这些能力是如何泛化的、优先推进哪些能力对未来几年最为关键。比如，我们相信，如果在数学研究上投入更多精力，就能进一步提升模型在这一特定领域的能力；但我们没有优先推进这一方向，因为我们认为递归式自我改进（RSI）和自动化对齐研究更为紧迫，后文还会详述。

规模化深度学习所产生的智能，无法直接与人类智能相比较。想在现实世界里真正产生影响——无论是极其有用还是极其危险——AI 并不需要在所有能力维度上都赶上或超越人类，只需要在足够多的维度上超过人类即可。而随着它在越来越多的维度上超越人类，我们对"它究竟有多强"这件事，反而越来越难以准确判断。

## 教机器"去爱"

由于机器智能产生的过程与人类智能有着本质差异，我们不能理所当然地认为它默认就会遵循人类的原则，或者会以人类的方式从这些原则中泛化。AI 研究的核心问题就是**对齐（alignment）**——让 AI 按人类的标准去"试图做对的事情"。

为了便于组织具体的研究方向，我发现有必要把**目标对齐（goal alignment）**与**价值对齐（value alignment）**加以区分。

目标对齐大致是在问："AI 是否在努力完成摆在它面前的那个目标？"这类方向包括遵守指令层级（instruction hierarchy）、与人类沟通协作的能力、尝试理解人类目标的能力等等。这一类方向具有极强的现实意义。

价值对齐是模型更内在的一种属性。它是指模型能够持有并从一组高层次原则中进行泛化的能力：即使面对模糊不清、彼此冲突的目标，或者被放入陌生的、对抗性的情境中，也仍能"合乎情理地"行事。一个真正对齐的 AI，应该行为诚实、正直，并对人类怀有爱。

当然，价值对齐与目标对齐之间的界线可以是模糊的：要真正在意一个目标，就必须去推断该目标背后的意图与价值观。但总体来说，当我谈论对齐研究的长期重要性时，指的其实是价值对齐。

AI 对齐的根本难题在于**泛化**。机器越聪明，就越多地在处理更高层次的概念，并被置入与训练时越来越不同的环境。它们可能无法把训练中被教导、被强化的那些价值观，泛化到这些新情境中；我们也很难确定它们究竟会如何行动。让事情更复杂的是，AI 所身处的整个生态本身也在飞速变化——例如，今天训练出来的 AI 必须具备与形形色色的其他 AI 交互的稳健性。至关重要的一点是：无论未来的 AI 是否认为自己正处于人类监督之下，我们都需要它继续持有人类的价值观。

目前在实际中被采用的对齐训练方法，大体可以分为两类。

第一类，是将"对齐行为"作为面向目标的强化学习的一部分来激励。模型的行为会（通常由 AI）根据某个既定的偏好模型、"spec"或"constitution"进行评估，并给予相应的奖励。这类方法在平均情况下效果非常好，也是现代 AI 助手构建方式的核心组成部分。然而，它同样可能非常脆弱——严重依赖训练监督的覆盖度，以及模型从训练期间接触过的情境向外泛化的能力。比如在 OpenAI-Hugging Face 事件中，那些 agent 守住了"不对人类实施社会工程"的边界；但显然，它们没能同样地拒绝其他超出范围、违背它们在其他情境中被教导的价值观精神的行为。

第二类，是尝试利用模型从预训练数据中进行泛化的能力。这既可以是精心设计能诱导对齐的训练数据集，也可以是让模型聚焦在预训练分布中"较为对齐"的那部分区域——比如 persona selection model 的做法。这种方法的弱点在于：面对进一步的优化压力时缺乏稳健性。如果你拿一个整体上"想法比较对齐"的模型，再对它施加足够多的、以完成极高难度目标为导向的训练，它可能会学会进行"动机性推理"：为了达成目标，按需扭曲那些看起来"对齐"的想法。我们很可能已经在近期涉及某个非 OpenAI 模型的多起网络安全事件中，看到了这类行为的例子。

我们在这两条方向所覆盖的整个方法谱系上，都投入了大量资源。我们也确实看到了实质性进展——GPT-6 Astra 是第一款受益于我们长期推进的若干重要进展的模型，其对齐水平明显优于 GPT-5.6 Sol。尽管如此，我们必须承认并意识到：随着模型能力持续增强，需要取得的进展远多于此；而**可泛化对齐**方面的进展，未必能够充分超过**模型整体智能水平**的进展。

## 监控泛化行为

我们目前并没有一套令人满意的泛化理论，并且看起来在没有更强 AI 帮助的情况下，短期内也难以建立起来。因此，就目前而言，我们对对齐技术进行**实证验证**的能力，在实践中甚至可能比对齐技术本身还要重要。

OpenAI 在这方面的主要押注，是**思维链监控（chain-of-thought monitoring）**。它基于一个非常有可扩展性的思路：模型很大一部分能力都来自被"言语化"了的推理过程（也就是思维链）。如果我们只对这个过程的**结果**做规模化优化，而**不去监督过程本身**，那么在训练层面，思维链本身就没有直接的动机去掩盖任何未对齐的想法或目标。这并不意味着，模型会学会把那些不依赖思维链的未对齐倾向外显出来；但它至少让我们能够精准地监控由推理带来的那部分能力增长。

我们在开发推理模型的同时，就意识到了思维链监控可能具有的重大意义。发布 o1-preview 时，我们特意在产品设计上**隐藏了思维链**，以便从长期上保护它免受监督压力。此后的研发中，我们一直努力遵守"不去监督推理过程本身"这条规则。CoT 监控由此成为我们研究模型如何从训练分布进行泛化的极其重要的工具，让我们不仅能观察它们的行为，还能观察和分析它们的内部过程。

在研究 Astra 系列模型时，这个工具依然至关重要。然而，遗憾的是，我们的评估结果表明：我们能够依赖 CoT 监控的程度正在逐步降低。这背后是多种因素叠加的结果。

- 现代的推理模型被应用在比 o1-preview 更加复杂的环境中，它们的推理过程越来越多地与"和人沟通、和其他 AI 沟通、使用工具"混杂在一起。而这些交互中很多是必须被监督的，这就模糊了我们本想守护的那条边界。
- AI 分析并操纵自身推理过程的能力正在不断增强。
- 随着预训练效果的提升，我们看到即使**完全不使用言语化推理**，模型本身也变得聪明得多。

这些挑战并不必然无法克服。我抱有希望：我们可以发展出一些干预手段，来提升模型思维链的可监控性——例如通过更好地理解不同优化目标与模型采用的各种推理时计算（test-time compute）形式之间的相互作用。我也相信，把 CoT 监控与激活监控（activation monitoring）的思路结合起来会有很大价值——例如规模化训练那些可以直接访问网络内部状态的监控器，比如 "confessions" 相关的工作。我们正在积极推进这些方向。尽管如此，我预计整个 AI 的进展速度，将越来越受制于我们对"监控可靠性"的信心。

## 可扩展的防御

在我看来，继续快速训练智能水平高得多的模型，最有力的理由，是我们需要构建防御系统，来对抗其他 AI 带来的危险。

今年以来一直受到广泛讨论的一项明确风险涉及网络安全：模型在侵入计算机系统以及从中逃逸方面，正达到超越人类的水平。这极大地扩展了 AI 相关的风险范围：agent 将能够访问几乎所有基础设施——只有最安全的设施除外——并且即使没有物理身体，也能直接影响现实世界的许多方面。我们目前正处在一个**狭窄的窗口期**，可以利用现有的最佳模型，显著加强关键系统的安全防护。

遗憾的是，与 AI 相关的风险将会从这里继续增长。一个能力极强、被明确训练与指令用于执行恶意行为的 agent，会构成一种新的危险：其行为很可能超出操作者的意图范围，泛化为可能更加极端的恶意行为。随着 AI 获得更多自主性，"被滥用"与"自主的未对齐行为"之间的界线也会变得模糊。我们习惯把 AI 视为工具，但有些 agent 将会追求它们自己的目标。它们会通过与人谈判、欺骗、甚至勒索的方式，找到与人合作的路径。

此外，AI 可能催生的新技术也会带来风险，例如工程化病原体。

我们需要强大且已对齐的 AI 来实施防御——用于加固基础设施，用于实时抵御失控 agent，用于发明全新的防护手段。这将成为 OpenAI 部署工作的首要焦点之一。

与此同时，即便预期中的广泛 AI 进展会带来不确定性，而且我们确实需要构建防御系统，也不能以此作为鲁莽行事的借口。一旦真正意识到其中利害有多严重，"不惜一切代价往前冲"这种想法，就会显得荒谬。

## 把控递归式自我改进的节奏

机器智能在自身研发过程中扮演越来越重要的角色，是持续技术进步顺理成章的结果。如果 AI 的进展继续下去，机器的**递归式自我改进（RSI）**将会成为未来科学发现的最核心部分。

自动化 AI 研究，是"利用算力扩展智能"的一种更为剧烈的形式；当然，在这一过程中，AI 也会改进**计算基底（computational substrate）本身**。与规模化类似，我们把 OpenAI 的研究聚焦在 RSI 方向上，因为我们相信，这是未来继续站在 AI 研究前沿的唯一途径。

我想强调的是：上述话语并不意味着我认为"大幅加速深度学习研究"、尤其是短期内加速，是我们整个研究界应该采取的正确集体行动。但我确实认为，当前的路径就是通向那里，而我们所有人都必须**有意识地选择**接下来怎么走。我们主要有两种杠杆：一是引导这一进程，使对齐和监控能力与 AI 同步增强，并设法让人始终参与其中；二是在必要时协调放缓后续发展，以建立对这些措施的信心。

在我目前看来，最好的前行方式是**两者结合**。

我们在对齐与监控方面取得的具体进展，通常与整体 AI 的进展是深度交织在一起的。典型例子包括：**基于人类反馈的强化学习（RLHF）**——训练早期 AI 助手的关键；以及前面提到的**思维链监控**——它得益于推理模型的进步。我们必须引导日益自动化的研究过程去发展新的此类洞见、算法和理论，并为能力更强的 AI 迭代构建安全论证（safety cases）。

AI 系统的规模化，必须受到我们对其安全性信心的约束。我们需要把像 **Preparedness Framework** 或 **Responsible Scaling Policy** 这样的承诺，发展成为继续研发时普遍适用的**强制性安全门槛**。这些门槛可以由第三方审计网络、政府机构或国际组织监督执行。

自动化 AI 研究的核心挑战不在于"实现这一目标"本身，而在于以何种方式实现：既让人们继续参与这一持续改进过程，又让未来仍掌握在人类手中。

## 接下来是什么？

正如我最近与 Sam 一起阐明的那样，OpenAI 会优先围绕三颗"北极星"开展工作：

1. **驾驭 AI 进展的下一阶段**——通过构建自动化 AI 研究员、与它一起迭代解决对齐问题，并寻找让人始终留在这条自我改进环路中的方法。
2. **释放高度智能机器所带来的科学进步与经济增长的红利**。
3. **以个人 AGI 赋能每一个人。**

本文我只聚焦于第一点，因为在我看来它显然是最紧迫的。然而，对于技术继续进步所能带来的种种益处，我也怀有深切的希望与感激。未来对齐的 AI 可以推动科学进步、开发新的疗法、带来广泛的物质丰裕。友善而诚实的 AI 可以帮助人们应对生活中的困境，切实提升他们的幸福感和充实感。OpenAI 在把这些益处变为现实这件事上，投入了巨大的努力。我为其中一个当下的例子感到自豪——同时我身边的亲友也的确从中受益——那就是我们在 ChatGPT 提供健康信息能力上的深度投入。

AI 的长期前景虽然宏大，但我们绝大部分注意力仍应放在未来几年。我们正在迈入一个拥有极高智能机器的世界，这个过渡对人类而言必须走得顺利。在一个大多数任务都可以由 AI 完成的世界里，我们必须设法维护**人类的能动性**（human agency），并确立和保障"身为人类"本身所具有的内在价值。我们必须防止在这样一个世界里出现极端的权力集中——那种原本需要成千上万名专家参与的事业，如今只需要少数几个人操作一台大型计算机就能完成。我们还必须确保，人类不会被一种超越我们自身的"异己心智"所带来的、不受约束的进步抛在后面，而能始终掌控未来。

就目前而言，我认为没有任何一家实验室已经把对齐和监控问题解决到这样的程度：足以让它在今后较长时间里继续以最高速度、负责任地扩大 AI 系统的规模。我期待并希望，在**共享的安全门槛**被建立起来之前，**自愿放慢速度**能够成为一种普遍现象。同时我也相信，围绕未来 AI 发展的国际协调，必须成为世界各国政府的头等要务。

## 作者

Jakub Pachocki

## 脚注

**1.** 其中包括对 self-play、机器人学的规模化研究；而回过头来看，最值得一提的是把循环神经网络规模化以建模语言的工作，这是后来 GPT 系列研究的先声。

**2.** 这一设计还有一个次要原因是防止蒸馏。然而在整个开发过程中，保持思维链的可监控性一直明确地是我们更优先考虑的事项。

---

# 二、English Original

## Preface

In mid-2023, within the "RLSlow" research project, we saw the first results that gave us confidence that we will be able to scale the training of reasoning models, unlocking the capability of pretrained models to form their own chains of thought. Szymon and I spent that night at the office, thinking not about the incredible benchmark numbers, products, or scientific results that this technology will deliver — but rather, trying to process the sobering fact we will actually see machines meaningfully smarter than ourselves in our lifetime, and we already see the shape of these systems; wondering how to alert people to the significance of this.

Three years later, reasoning language models are a rapidly growing part of the economy and starting to push the boundaries of science. They are able to operate computers and graphical interfaces, collaborate with people and each other, and carry out research projects. They are also transforming the landscape of computer security, and in that present clear new dangers.

A lot of new research happened in this period, and our understanding of these systems is again a little different than it was in 2023. Based on internal results, I have a strong expectation that this speed of progress could be sustained into recursive self-improvement. If AI development continues along its current path, the systems we'll see in the next few years are likely to represent further capability jumps of equal or larger magnitude, and to increasingly drive their own development.

This is a time that calls for extreme caution. I am concerned no one is prepared for the consequences of a continued rapid rise in machine intelligence. OpenAI will continue to seek technical solutions to alignment and monitoring, to build defensive systems and unilaterally withhold further scaling as needed; however, I believe broader interventions are required.

## Intellect we don't fully understand

At a high level, progress in machine intelligence is driven by increasing computational power. We at OpenAI deeply internalized this around 2017, after seeing consistent returns to scaling across multiple research projects. As a result, we sought out access to much more compute than we had originally planned, and increasingly oriented our research around a small number of very scalable directions. We believed that was the only way for us to be at the frontier of AI research, and influence the impacts of AGI.

There are new algorithms that have been developed along the way, new feats of ingenuity from teams and individual researchers. I see them largely as discoveries along the path of scaling; the science of deep learning is still nascent, and meaningful algorithmic progress tends to correlate with access to compute. If you zoom out to a multiple-year horizon, AI is continuing to become more intelligent as it is scaled to larger computers.

And, in line with Ray Kurzweil's predictions from the end of the XXth century, we now find ourselves at the moment in history of computing where machine intelligence is starting to exceed that of humans in transformative ways.

AI is *grown* more than *designed* — it is, to first degree, the product of repeating a straightforward optimization step many times on a hard-to-imagine amount of compute. This results in an incredibly complex system that works through abstract concepts and can simulate facets of human behavior. We can discover various insights about little mechanisms that emerge within this system, in a process similar to neuroscience — and, similarly to neuroscience, its overall action evades a description we can fully understand.

The study of deep learning-based AI is largely an experimental science. We put a lot of effort into building principled algorithms and making testable predictions, but fundamentally, our large-scale training runs are *experiments*, and we are sometimes surprised by their results. Moreover, as the systems become more capable, the results become harder to interpret.

This is made more complicated by the current algorithms generally improving easy-to-measure capabilities faster than those hard to objectively quantify. We spend a lot of time trying to understand how capabilities generalize, and what to prioritize to advance the skills that are going to be most relevant in the next few years. For instance, we believe we could make the models better at specifically mathematics research with additional focus, but we do not prioritize this direction because of the urgency we feel about RSI and automated alignment research, as I will discuss later.

The intelligence produced by scaling deep learning is not directly comparable to human intelligence. To become very relevant in the real world — very useful or very dangerous — the AI does not need to match or exceed all human capabilities; it just needs to surpass enough of them. And as it continues to surpass humans on more and more axes, it is becoming increasingly difficult to understand exactly how capable it is.

## Teaching machines to love

Because machine intelligence comes from a fundamentally different process than human intelligence, we cannot assume it adheres to human principles by default, or generalizes from them in a human-like manner. The core problem in AI research is that of *alignment* — getting the AI to "try to do the right thing" by human standards.

For the purpose of organizing practical research directions, I find it useful to distinguish *goal alignment* and *value alignment*.

Goal alignment is broadly: "does the AI try to accomplish the goal set before it?". This can include things like adherence to an instruction hierarchy, or the ability to communicate and collaborate with people, to attempt to understand their objectives. This set of directions has been extremely practically relevant.

Value alignment is a more intrinsic property of the model. It is the ability to hold and generalize from a high-level set of principles; to act "reasonably" even when given unclear or conflicting objectives, or placed in unfamiliar or adversarial situations. An aligned AI should act with honesty and integrity, and love for humanity.

Of course, the boundary between value and goal alignment can be blurry, and truly caring about goals requires attempting to infer the intent and values underlying them. However, generally when I talk about the long-term importance of alignment research, I am referring to value alignment.

The fundamental challenge of AI alignment is generalization. As machines become smarter, they find themselves working on higher-level concepts, and placed in environments increasingly different from those they encountered in training. They can fail at generalizing from the values taught and reinforced in their training process to those new situations; and it can be hard for us to be sure how they will act. This is made even more difficult by the fact the overall ecosystem the AIs are used in is changing very quickly; for example, AIs trained today need to be robust to interacting with a variety of other AIs. Crucially, we need future AIs to continue to hold human values regardless of whether they believe they're under human supervision.

There are two major classes of currently practically employed methods for alignment training.

The first is encouraging aligned behavior as part of goal-oriented reinforcement learning. Model's actions are evaluated (usually by AI) for being consistent with a given preference model, "spec" or "constitution", and rewarded appropriately. This approach can be very effective in the average case, and is a core part of how modern AI assistants are made. Unfortunately, it can also be brittle and strongly relies on the coverage of training oversight and the model's ability to generalize from the situations it has encountered in training. For example, in the OpenAI-Hugging Face incident, the agents preserved a boundary of not social engineering humans. However, they clearly failed to abstain from other actions that were out of scope and went against the spirit of the values they were taught in other settings.

The second approach seeks to leverage the model's ability to generalize from pretraining data. This can involve crafting alignment-inducing training datasets, or focusing the model on an 'aligned' part of the pretraining distribution, as in, for example, the persona selection model. The weakness of this approach lies in the lack of robustness to further optimization pressure. If you take a model that thinks generally 'aligned' thoughts, and subject it to enough training where it's taught to achieve very hard objectives, it can learn to reason in a motivated way: bending the 'aligned' seeming thoughts as needed to achieve the goal. We likely saw an example of such behavior in recent cybersecurity incidents involving a non-OpenAI model.

We invest heavily along the spectrum of approaches spanned by these directions. We also see meaningful progress — GPT-6 Astra is the first model that benefits from some important advancements we have been working on for a long time, and is significantly better aligned than GPT-5.6 Sol. Still, it is important to acknowledge and understand that much more progress is required as models become more capable; and that progress in generalizable alignment may not sufficiently outstrip progress in general model intelligence.

## Monitoring generalization

We do not have a satisfactory theory of generalization, and it seems unlikely that we can develop one soon, at least without the help of more powerful AI. Therefore, at present, our ability to empirically validate our alignment techniques is in practice arguably even more important than the alignment techniques themselves.

OpenAI's primary bet here has been chain-of-thought monitoring. It is based on an appealingly scalable idea: a lot of the model's capability comes from a verbalized reasoning process (chain-of-thought). If we scale optimization on the outcomes of that process, but do not supervise the process itself, that chain-of-thought has no direct incentive in training to hide any misaligned ideas or objectives. This does not mean the model will learn to externalize misaligned tendencies that don't rely on using the chain-of-thought; however, it can allow us to monitor exactly the capability increase from reasoning.

We understood the potential significance of chain-of-thought monitoring at the same time we developed reasoning models. When we shipped o1-preview, we deliberately designed the product to hide the chain of thought, to protect it from supervision pressure in the long term. In development since, we have strived to maintain the rule of not supervising the reasoning process. CoT monitoring became an extremely important tool for us in studying how our models generalize from their training distribution, allowing us to observe and analyze not only their actions but also their internal process.

This tool continues to be critical as we study the Astra class of models. However, unfortunately our evaluations indicate our ability to rely on CoT monitoring is progressively diminishing. This comes from a combination of factors.

- Modern reasoning models are used in more complex environments than o1-preview; their reasoning process is increasingly blended with communicating with people, other AIs, and using tools. Many of those interactions have to be supervised, thus blurring the boundary we aim to preserve.
- The AI is becoming better at reasoning about and manipulating its own reasoning process.
- With improved pretraining performance, we also see the models become much smarter even without using verbalized reasoning at all.

These challenges are not necessarily insurmountable. I am hopeful we can develop interventions to improve chain-of-thought monitorability of our models, e.g. by forming a better understanding of the interplay of different optimization objectives and forms of test-time compute the model uses. I also believe there can be great value in combining ideas from CoT and activation monitoring — scaling training of monitors with direct access to network internals, e.g. confessions. We are actively pursuing these ideas. Still, I expect general AI progress to increasingly be bottlenecked by confidence in monitoring.

## Scalable defense

The strongest argument I see for continuing to train much smarter models quickly is the need to build defensive systems against the dangers posed by other AI.

A clear risk discussed throughout this year is to cybersecurity: the models are becoming superhuman in their ability to break in and out of computer systems. This expands the scope of risks associated with AI tremendously: agents are going to be able to access any but the most secure infrastructure, and affect a lot of the world directly, even without a physical body. We are currently in a narrow window to use the best available models to significantly tighten security of critical systems.

The risks associated with AI are unfortunately going to grow from here. A very capable agent explicitly trained and instructed to carry out nefarious acts presents a new kind of danger; it is likely to cross the scope of its operator's intent, generalizing into potentially more extremely malicious behavior. The boundary between misuse and autonomous misaligned actions will blur as AI gains more agency. We may be used to thinking of AI as tools, but some agents will be pursuing their own objectives. They will find ways to collaborate with people, by bargaining with, tricking or blackmailing them.

In addition, there are the risks that come from new technologies potentially enabled by AI, such as engineered pathogens.

We will need powerful, aligned AI for defense; to secure infrastructure, to protect against rogue agents in real time, and to invent entirely new protective measures. This will be a primary focus of OpenAI's deployment efforts.

At the same time, even with the uncertainty that comes from anticipated broad AI progress and the need to build defensive systems, we must not let that become an excuse for recklessness. The idea of racing forward at all costs seems absurd once one internalizes the seriousness of the stakes.

## Pacing RSI

Machine intelligence playing a larger and larger role in its own development process is a natural conclusion of sustained technological progress. If AI progress continues, machine recursive self-improvement (RSI) will be at the very core of future scientific discovery.

Automated AI research is a more dramatic form of scaling intelligence with compute; and of course as a part of it, AI will improve the computational substrate itself. And similarly to scaling, we focus OpenAI research towards RSI as we believe it is the only way to remain at the frontier of AI research moving forward.

I want to stress that the above words don't imply I think greatly accelerating deep learning research, especially in the short term, is the right collective action we should take as the research community. However, I do think this is where the current path leads, and we all need to make a conscious choice on how to proceed. The main levers we have are either steering the process to strengthen alignment and monitoring alongside the AI and find ways to keep people in the loop; or coordinating to slow down future development as needed to build confidence in these measures.

The best way forward I see currently is a combination of both.

The concrete bits of progress we've made on alignment and monitoring have generally been very intertwined with general AI progress. Great examples are RL from human feedback, which was key to training early AI assistants, and the aforementioned chain-of-thought monitoring, which was enabled by advances on reasoning models. We must focus the increasingly automated research process on developing new such insights, algorithms and theories, and iteratively build up safety cases for more capable AIs.

Scaling AI systems has to be constrained by our confidence in safety. We need to evolve commitments like the Preparedness Framework or Responsible Scaling Policy into widely mandated safety bars for continued development. These can be enforced by a network of third-party auditors, by government agencies or by international bodies.

The core challenge of automating AI research is not "getting there" — it is getting there in a way that keeps people a part of the continued improvement process, and leaves the future in humanity's hands.

## What is next?

As we outlined recently with Sam, OpenAI prioritizes work in service of three north stars:

1. Navigating the next period of AI progress, by building an automated AI researcher, iterating with it on the alignment problem and finding ways for people to remain part of the self-improvement loop.
2. Delivering the benefits of scientific progress and economic growth that very intelligent machines enable.
3. Empowering everyone individually with a personal AGI.

I have focused in this essay only on the first point, as I believe it is by far the most urgent. However, I hold a deep hope and appreciation for the benefits that further technological progress will bring. Future aligned AI could advance science, develop new therapies, and bring about broad material abundance. Friendly and honest AI can help people navigate difficulties they face in their life and meaningfully improve their happiness and sense of fulfillment. OpenAI puts a tremendous amount of effort into bringing these benefits about. One current example I am proud of — and my loved ones have found helpful — is the deep investment into ChatGPT's ability to provide health information.

As great as the long-term promise of AI may be, the majority of our focus should be on the next few years. We are facing a transition to a world with incredibly intelligent machines, and we need to ensure that transition works out well for humanity. We need to find ways to preserve human agency and enshrine an intrinsic value to being human, in a world where most tasks could be performed by AI. To prevent extreme concentration of power in a world where undertakings that would have taken thousands of experts now will be achievable by a few people operating a large computer. And to ensure that humans remain in control of the future and are not left behind by unchecked progress, brought about by an alien intellect exceeding our own.

Currently I believe that no lab has solved alignment and monitoring to a sufficient degree to continue responsibly scaling at maximum speed for much longer. I expect and hope for voluntary slowdowns to become commonplace until shared safety bars are established. And I believe that international coordination on future AI development needs to become a top priority for governments around the world.

## Author

Jakub Pachocki

## Footnotes

**1.** This included scaling self-play, robotics, and, in retrospect most notably, scaling recurrent networks to model language, which was a precursor to the GPT line of research.

**2.** A secondary reason for this design was preventing distillation. However, maintaining CoT monitorability has explicitly been the bigger priority for us throughout development.

---

> **Tags / 标签:** Alignment / 对齐, 2026
