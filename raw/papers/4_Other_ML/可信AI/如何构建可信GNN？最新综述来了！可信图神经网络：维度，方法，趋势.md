如何构建可信GNN？最新综述来了！可信图神经⽹络：维度，⽅法， 趋势 

HeZhang，BangWu 图与推荐 2022-05-25 15:18 发表于加拿⼤ 


![](raw/papers/4_Other_ML/可信AI/如何构建可信GNN？最新综述来了！可信图神经网络：维度，方法，趋势_images/如何构建可信GNN？最新综述来了！可信图神经网络：维度，方法，趋势.pdf-0001-02.png)


## 前⾔ 

在过去的⼏年⾥，以神经⽹络为⾸的⼈⼯智能技术通过对不同类型数据的深⼊挖掘已经深刻地 改变了⼈类的⽣活并极⼤地推动了社会发展的进程 [1]。作为⼈⼯智能领域最活跃的研究⽅向之 ⼀，图神经⽹络（GNNs）凭借其卓越的性能已经被⼴泛应⽤于个性化推荐等⽇常⽣活中，在如 新冠药物研发等科学前沿领域也不乏其身影。随着图神经⽹络技术的蓬勃发展，⼈们发现以任 ⼀ 务性能为单 设计⽬标的图神经⽹络系统中仍存在譬如⾯对恶意攻击的脆弱性等问题。因此， ⼈们愈发渴望建⽴起可信的图神经⽹络。 

近年来，构建可信⼈⼯智能系统已经成为了世界各国的普遍共识 [2][3]。如何全⾯地建⽴起可 信 图 神 经⽹络 已 经 成 为了 ⼀个 亟 待 解 决 的 重 ⼤ 问题 。 该 ⽂ 是 由 Monash 团 队 （ Shirui Pan, Xingliang Yuan，Bang Wu, He Zhang）联合Hanghang Tong (UIUC) 以及Jian Pei (SFU， 即将⼊职Duke)对可信图神经⽹络的最新综述（双栏36⻚，299篇⽂献）。 

该综述从研究背景与特点出发，提出了⼀个可信图神经⽹络的开放框架，着重阐述了“可信 GNN”的六⼤维度（鲁棒性、可解释性、隐私、公平性、问责、环境福祉）及其技术⽅法。同 时，该综述探讨了不同可信维度之间的交互关系，并提出了可信图神经⽹络未来的研究⽅向， ⼀ 为建⽴起可信图神经⽹络描绘了 幅细致⽽全⾯的技术路线图。 


![](raw/papers/4_Other_ML/可信AI/如何构建可信GNN？最新综述来了！可信图神经网络：维度，方法，趋势_images/如何构建可信GNN？最新综述来了！可信图神经网络：维度，方法，趋势.pdf-0002-04.png)


单位⼁Monash University 


![](raw/papers/4_Other_ML/可信AI/如何构建可信GNN？最新综述来了！可信图神经网络：维度，方法，趋势_images/如何构建可信GNN？最新综述来了！可信图神经网络：维度，方法，趋势.pdf-0002-06.png)



![](raw/papers/4_Other_ML/可信AI/如何构建可信GNN？最新综述来了！可信图神经网络：维度，方法，趋势_images/如何构建可信GNN？最新综述来了！可信图神经网络：维度，方法，趋势.pdf-0003-00.png)


综述名称：Trustworthy Graph Neural Networks: Aspects, Methods and Trends 

全⽂链接：https://arxiv.org/pdf/2205.07424.pdf 

Github: https://github.com/Radical3-HeZhang/Awesome-Trustworthy-GNNs 

~~1~~ 


![](raw/papers/4_Other_ML/可信AI/如何构建可信GNN？最新综述来了！可信图神经网络：维度，方法，趋势_images/如何构建可信GNN？最新综述来了！可信图神经网络：维度，方法，趋势.pdf-0003-05.png)


⼀ 图作为 种具有极强表示能⼒的数据类型，通过描述实体特征和刻画实体之间的关系，已经被 ⼴泛应⽤于描述⽣物学、化学、物理学、语⾔学和社会科学等诸多领域的数据。近年来，图神 

经⽹络技术的蓬勃发展已经⾰命性地提升了各种图计算任务的性能，并促进了其在现实⽣活中 的⼴泛应⽤。 

在⽇常⽣活中，图神经⽹络通过考虑⽤户与⽤户/服务内容之间的交互关系，可以在信息流媒 体、在线购物、社交软件等消费应⽤中为⽤户提供个性化搜索与推荐服务。在科学前沿中，通 过使⽤图数据来表示复杂的系统，研究者们可以使⽤图神经⽹络来发现天体运动背后隐藏的规 律。通过将其应⽤于虚假新闻检测与新冠药物研发，图神经⽹络⼤⼤提升了我们的社会福祉。 

⼀ 虽然研究者们已经从很多⻆度（如⾃监督学习，提升模型深度等）设计出了进 步提升图神经 ⼀ ⼀ ⽹络性能的⽅法，但是在 些关键领域中任务性能不是设计图神经⽹络的唯 ⽬标。例如，基 于图神经⽹络的异常检测系统需要对恶意攻击具有鲁棒性，基于图神经⽹络的信⽤评分系统不 应该因为⽤户的年龄、性别等因素拒绝其贷款申请，基于图神经⽹络的药物发现应⽤应向研发 ⼈员提供其结果的全⾯解释。 

基于上述需求，⼈们愈发渴望基于图神经⽹络系统是可信的。在这样的背景下，这篇综述旨在 总结“可信图神经⽹络（Trustworthy GNNs）”的最新进展，为相关研究⼈员和从业⼈员提供技 术路线图，并为可信图神经⽹络的未来研究与产业化发展提供⽅向。 

这篇综述的主要贡献有：1）以包含诸多可信维度的开放框架描述了可信图神经⽹络，并提出了 图神经⽹络与其他常⻅⼈⼯智能技术（如CNN）在可信研究中的典型差异；2）针对图神经⽹络 的不同可信维度，对现有⽅法进⾏了综合且全⾯的总结；3）提出了不同可信维度之间的关系对 于构建可信图神经⽹络系统⾄关重要，并从⽅法和成效两个层⾯总结了已有的研究⼯作；4）通 过将可信图神经⽹络的概念作为⼀个整体，提出了未来的潜在研究⽅向。 


![](raw/papers/4_Other_ML/可信AI/如何构建可信GNN？最新综述来了！可信图神经网络：维度，方法，趋势_images/如何构建可信GNN？最新综述来了！可信图神经网络：维度，方法，趋势.pdf-0004-05.png)


**----- Start of picture text -----**<br>
2<br>**----- End of picture text -----**<br>


## 图神经⽹络与可信 

为了便于读者理解，本⽂⾸先对以下核⼼概念进⾏介绍。 

图数据（graph data）：图 ⼀般由节点集合 和边集合 构成。该图中节点的个 数为 ， 节 点 之 间 边 的 个 数 为 。 给 定 图 ， 对 应 的 拓扑 结 构 可 以 ⽤ 邻 接 矩 阵 表示，其中 表示了节点 和节点 之间的连接关系。即，如果节点 和 相互连接，则 ，否则 。如果图 中节点具有属性，可⽤⼀个特征矩阵 来描 述这些属性信息。因此，图 也可以表示为 。 

图神经⽹络（GNNs）：图神经⽹络是⼀系列可⽤于图数据上计算任务（如节点分类、链接预 测、图分类）的神经⽹络的总称。图神经⽹络中的⼀个典型运算步骤是消息传播（message passaging）。在消息传播的过程中，图神经⽹络通过聚合所有邻居节点的信息来更新当前节点 的表示。在与其他运算结合（如⾮线性激励）的基础上，经过多次表示更新迭代后，图神经⽹ 络可以计算得到相应的数据表示。 

可 信 （ trustworthiness ）： 可 信 ⽤ 于 形 容 ⼀个 系统 是 值 得 被 信任 的 ， 它 描 述 了信任 发 起 者 （the trustor）与信任接收者（the trustee）之间的信任（trust）关系。在可信图神经⽹络的 上下⽂中，信任接收者（the trustee）是图神经⽹络系统，信任发起者（the trustor）可以是 ⽤户，开发者，监管部⻔甚⾄整个社会。 

可信图神经⽹络（Trustworthy GNNs）被定义为兼顾可信和卓越性能的图神经⽹络，这些可信 维度包含但不限于图神经⽹络的鲁棒性、可解释性、隐私、公平性、责任和福祉。原⽂定义如 下： “In this survey, we define trustworthy GNNs as competent GNNs that incorporate core aspects of trustworthiness, including robustness, explainability, privacy, fairness, accountability, well-being, and other trust-oriented characteristics in the context of GNNs.” 

~~3~~ 


![](raw/papers/4_Other_ML/可信AI/如何构建可信GNN？最新综述来了！可信图神经网络：维度，方法，趋势_images/如何构建可信GNN？最新综述来了！可信图神经网络：维度，方法，趋势.pdf-0006-01.png)


## 综述框架 

⼀ 该综述在第 章主要介绍了研究背景、可信图神经⽹络的定义、不同可信维度的定义、度量和 研究差异、与已有综述的关系及其主要贡献。在第⼆章，它介绍了图神经⽹络的基本概念和计 算任务。在第三⾄⼋章，它分别从鲁棒性（robustness）、可解释性（explainability）、隐私 （ privacy ） 、 公 平性 （ fairness ） 、 问 责 （ accountability ） 、 环 境 福祉 （ environmental well-being）这六个⽅⾯介绍、总结了典型的技术⽅法并讨论了未来的研究⽅向。在第九章， 这篇综述从⽅法和成效两个层⾯，总结了上述六个可信维度之间的复杂关系。最后，综述的第 

⼗章将可信图神经⽹络看成⼀个整体，为全⾯构建可信图神经⽹络系统提出了未来研究与产业 化的五个⽅向。 

## ~~4~~ 

## 鲁棒性（Robustness） 

鲁棒性是指图神经⽹络⾯临⼲扰时保持稳定预测结果的能⼒。图神经⽹络的预测结果可以被多 种扰动（尤其是各类针对图神经⽹络的攻击）所影响。这给图神经⽹络在这给图神经⽹络在银 ⾏⾦融系统的欺诈侦测、⾃动驾驶系统的交通预测及规划等涉及⼈身财产安全的场景中的应⽤ 带来了较为严峻的挑战。因此，针对鲁棒性的研究是可信图神经⽹络中不可或缺的关键环节。 

该综述总结了⽬前图神经⽹络鲁棒性研究的相关⼯作，并分别介绍了对抗攻击、防御的分类和 ⼀ 些典型⽅法。其中，对抗攻击的分类源⾃对攻击威胁模型（threat model）的分析，如下图 所示；对防御的分类则更多基于该技术的具体执⾏阶段（target phase）。 

在介绍不同类别的⽅法时，作者对这些⽅法的特点与⽤于其他常⻅的⼈⼯技术（如CNN）的⽅ 法的差异进⾏了讨论。例如，具体扰动操作类型（perturbation operations）和攻击者⽬的 （attack goals）的不同分类和名称变化。 


![](raw/papers/4_Other_ML/可信AI/如何构建可信GNN？最新综述来了！可信图神经网络：维度，方法，趋势_images/如何构建可信GNN？最新综述来了！可信图神经网络：维度，方法，趋势.pdf-0007-06.png)


此外，作者还对这些攻击、防御⽅法的适⽤性进⾏了讨论和总结。例如，对不同类型的防御⽅ 法 ， 作 者 从 应 ⽤ 阶 段 （ stage of application ） 、 模 块 化 （ modularity ） 、 部 署 的 兼 容 性 （deployment compatibility）三个⻆度进⾏了⽐较和分析。 

最后，该综述根据⽬前现状提出了未来的两个研究⽅向，分别是制定标准化的鲁棒性评估⽅法 （robustness evaluations）和提⾼现有防御⽅法的可扩展性（defence scalability）。 

## ~~5~~ 

## 可解释性（Explainability） 

可解释性是指使图神经⽹络的预测结果可以被⼈理解的能⼒。如果其预测结果不能够被⼈所理 ⼀ 解，那么⼈们就不会信任图神经⽹络。信任的缺失会进 步限制其在涉及公平（如信⽤⻛险预 测）、信息安全（如芯⽚设计）、⽣命安全（如⾃动驾驶）的场景中的应⽤。因此，可信的图 神经⽹络系统需要对其预测结果做出解释。 

在介绍了解释形式、解释⽅法的类别等基本概念后，该综述将涉及图神经⽹络可解释性的⼯作 分 成 了 ⾃ 解 释 （ interpretable GNNs ） 和 事 后 解 释 （ post-hoc explainers ）两 类 。 ⾃ 解 释 （Interpretable GNNs）主要包含了贡献估计（contribution estimation）、引⼊可解释模块 （ introduction of interpretable module ） 、 嵌 ⼊ 空 间 原 型 学 习 （ embedding prototype learning ） 和 依 据 ⽣ 成 ⽅ 法 （ rationale generation methods ） 。 事 后 解 释 （ post-hoc explainers）主要包含了基于梯度/特征的⽅法（gradient/feature-based methods）、基于扰 动的⽅法（perturbation-based methods）、代理模型⽅法（surrogate methods）、分解⽅ 法 （ decomposition methods ） 、 ⽣ 成 ⽅ 法 （ generation methods ） 和 其 他 ⽅ 法 （ other methods）。 


![](raw/papers/4_Other_ML/可信AI/如何构建可信GNN？最新综述来了！可信图神经网络：维度，方法，趋势_images/如何构建可信GNN？最新综述来了！可信图神经网络：维度，方法，趋势.pdf-0009-00.png)


该综述在介绍了典型⽅法的原理后，对这些⽅法进⾏了全⾯的⽐较，如下表所示： 


![](raw/papers/4_Other_ML/可信AI/如何构建可信GNN？最新综述来了！可信图神经网络：维度，方法，趋势_images/如何构建可信GNN？最新综述来了！可信图神经网络：维度，方法，趋势.pdf-0010-00.png)


此外，该综述从⾃解释与事后解释（interpretability and explainability）、提供解释时所需的 背景知识（white/grey/black-box knowledge）、获得解释的推理原理（reasoning rationale）、其他局限性（other limitations）这四个⽅⾯对当前的⼯作进⾏了总结。最后，该 综述提出建⽴严格的模型不可知的解释⽅法（strictly model-agnostic methods）和⽤于真实 应⽤的评估基准（evaluation benchmark for real applications）是图神经⽹络的可解释性未 来研究的两个⽅向。 

~~6~~ 

隐私（Privacy） 

隐私也是构建可信图神经⽹络时不可忽略的可信维度。在构建和维护可信图神经⽹络系统的过 

程中，模型本身或者图数据等敏感、隐私的信息都⾯临被泄漏的⻛险。因此，该综述⾸先对当 前隐私数据泄漏相关的研究进⾏了总结，然后介绍了多种隐私保护⽅法。 

针对隐私泄露问题，该综述从当前隐私攻击的威胁模型⼊⼿，⾸先介绍了潜在威胁的⽬标和能 ⼒，然后介绍了模型提取（model extraction attack）、成员推断（membership inference attack）、模型逆向（model inversion attack）这三个常⻅的隐私攻击和其他场景中存在的隐 私泄露隐患。 

随 后 ， 该 综 述 介 绍 了 在图 神 经⽹络 场 景 下 的 联 邦 学 习 （ federated learning ） 、 差 分 隐 私 （ differential privacy ） 、 脱 敏 训 练 （ insusceptible training ） 、 安 全 计 算 （ security computation）这四种隐私保护技术。此外，作者在对上述技术进⾏了分类介绍后，还对它们 的具体应⽤场景和隐私保护结束所带来的隐私保护、模型精度、实⾏效率之间的权衡问题进⾏ 了讨论。 


![](raw/papers/4_Other_ML/可信AI/如何构建可信GNN？最新综述来了！可信图神经网络：维度，方法，趋势_images/如何构建可信GNN？最新综述来了！可信图神经网络：维度，方法，趋势.pdf-0012-00.png)


在 章 节 末 尾 ， 针 对 当 前 研 究 的 现 状 ， 作 者 认 为 ⽬ 前 对 梯 度 信 息 导 致 的 泄 漏 （ leakage from gradient）隐患、隐私攻击的防御⼿段（defence against privacy attacks）的研究仍绠短汲 ⼀ 深，未来还需要进 步地关注和探索。 

公平性（Fairness） 

~~7~~ 

通过保护弱势群体或者个⼈的关键利益，公平的系统可以赢得⼈们的信任。公平的图神经⽹络 系统意味着其预测结果中排除了对于某些群体或者个⼈的偏⻅。当前，图神经⽹络主要以数据 ⼀ 驱动的⽅式来完成图计算任务。然⽽，图神经⽹络中的消息传播机制可能会进 步放⼤已经存 在于数据中的偏⻅。此后，由于个⼈喜好或者⾏为偏差等因素，⼈们在与图神经⽹络服务交互 ⼀ 的过程中将进 步加深图数据中的偏⻅。 


![](raw/papers/4_Other_ML/可信AI/如何构建可信GNN？最新综述来了！可信图神经网络：维度，方法，趋势_images/如何构建可信GNN？最新综述来了！可信图神经网络：维度，方法，趋势.pdf-0013-01.png)


在介绍了公平性定义（group fairness，individual fairness, counterfactual fairness)、使⽤ 阶段（pre-processing, in-processing, post-processing）等基本概念后，该综述将当前的⽤ 于图神经⽹络公平性研究的⽅法分成了公平表示学习⽅法（fair representation learning methods）和公平预测增强⽅法（fair prediction enhancement methods）。在介绍完这些⽅ 法的基本原理后，对这些⽅法进⾏了全⾯的⽐较，如下表所示。 


![](raw/papers/4_Other_ML/可信AI/如何构建可信GNN？最新综述来了！可信图神经网络：维度，方法，趋势_images/如何构建可信GNN？最新综述来了！可信图神经网络：维度，方法，趋势.pdf-0014-00.png)


最后，该综述提出探讨公平性的定义与评估（fairness definition and evaluation）、对任务性 能的影响（influence on task performance）、解释不公平的来源（revealing unfairness） 是未来图神经⽹络公平性研究中需要重点关注的研究⽅向。 

~~8~~ 

## 问责（Accountability） 

随着愈加⼴泛的应⽤场景和复杂的系统结构的出现，个⼈、企业和政府机构都对可信图神经⽹ 络中的有效问责提出了更⾼的要求。近年来，中、美、欧的企业和政府机构关于如何构建⼈⼯ 

智能的问责框架提出了各⾃的规划和指导意⻅。该综述基于上述内容，归纳了对图神经⽹络问 责框架的三个要求： 

- (1) 应当设计合理的评估和认证机制（assessment and certification processes），并使其伴 随整个图神经⽹络系统的开发和运⾏周期； 

- (2) 应当保证开发和运⾏过程的可审计性（assurance of auditability）； 

- (3) 应当建 ⽴ ⾜ 够 的 协 调 和反 馈 机 制 ， 以 ⽅ 便⼈ 类 对 系统 实 施 ⼲ 预 （ adjusting ） 、 补 救 （remediating）等措施，并可以对不当⾏为进⾏惩罚（punishing）。 

随后，该综述介绍了当前可⽤于构建可信图神经⽹络系统中问责框架的两⼤类⼯作：基准化分 析（benchmarking）和安全性评估（security evaluation）。 

该 综 述 根 据 图 神 经⽹络 不 同 的 开 发 阶 段 — 模 型 设计 （ architecture design ） 、 模 型 训 练 （model training）、模型验证（model validation）—对基准化分析的研究进⾏了介绍。对于 安全性评估，作者主要介绍了验证系统完整性（integrity verification）的相关研究，并根据验 证对象的不同分别介绍了数据完整性（data integrity）、程序完整性（procedure integrity） 的验证。 

最后，该综述为可信图神经⽹络的问责提出了三个研究⽅向。第⼀，提供针对更多不同性质逾 矩⾏为的检测（violation detection）；第⼆，覆盖全过程的程序完整性检测和全部件的数据完 整性检测；第三，继续提⾼系统的可审计性和建⽴更多的协调反馈机制。 

~~9~~ 

## 环境福祉（Environmental well-being） 

可信图神经⽹络应当符合其部署环境的社会价值观。当前，全球变暖是⼈类社会亟需解决的重 ⼤环境问题，实现碳中和的雄伟⽬标需要各⾏各业的共同努⼒。为了降低图神经⽹络系统的环 境影响，在介绍了节点数每焦⽿等相关的度量后，该综述总结了提升图神经⽹络效率的多种⽅ 法。 

(1) 可扩展的图神经⽹络和⾼效的数据通信：随着图数据的爆炸性增⻓，⼤规模数据集对于图神 

⼀ 经⽹络的⾼效运⾏提出了挑战。为了应对这 挑战，当前的技术主要包含了采样⽅法 

（sampling method）、可拓展架构（scalable architectures）、产业应⽤（industrial applications）、⾼效数据通信（efficient data communication）等。 

(2) 模型压缩技术：随着技术的发展，研究⼈员提出了更深和更复杂的图神经⽹络模型来提升其 性能。然⽽，这些模型的规模限制了其在只具备有限计算资源的边缘计算设备上的部署。因 ⼀ 此，模型压缩技术是解决这 挑战的⼀个有效⽅法。相关的技术包含了知识蒸馏（knowledge distillation）、模型剪枝（model pruning）、减少参数规模（reducing parameter size）、 模型量化（model quantisation）等。 

(3) 开发框架与加速器：图数据的不规则性、模型中稀疏与稠密计算的相互交替、模型与应⽤的 多样性等特点使得图神经⽹络系统需要使⽤专⻔设计的框架与加速器来提升其效率。为了解决 ⼀ 这 问题，当前的⽅法主要有软件框架（SW frameworks）、硬件加速器（HW accelerators）、效率瓶颈分析（analysing the efficiency bottleneck）、软硬件协同设计 （SW-HW co-design）等。 

最后，该综述提出探索⾼效的图神经⽹络（exploration of efficient GNNs）和研究⽤于图神经 ⽹络的加速器（accelerators for GNNs）是未来促进图神经⽹络环境福祉的两个研究⽅向。 

~~10~~ 

## 不同可信⽅⾯之间的关系 

当前促进图神经⽹络可信性的研究主要集中在上述六个维度中的⼀个，该综述提出构建可信图 神经⽹络不能忽视上述六个可信维度之间的关系，并从如下两个⻆度对此进⾏了总结： 

1） 来⾃可信图神经⽹络中⼀个维度的⽅法如何被⽤于解决其他维度的研究问题（how the methods from one aspect of trustworthy GNNs are adapted to address objectives in other aspects）。 

2 ） 为什么促 进 可 信 图 神 经⽹络 的 ⼀个 维 度 可 能 会促 进 或 者 抑 制其 他 可 信 维 度 （ why advancing one aspect of trustworthy GNNs can promote or inhibit other aspects）。 

~~11~~ 


![](raw/papers/4_Other_ML/可信AI/如何构建可信GNN？最新综述来了！可信图神经网络：维度，方法，趋势_images/如何构建可信GNN？最新综述来了！可信图神经网络：维度，方法，趋势.pdf-0017-01.png)


## 未来的研究⽅向 

针对潜在的研究热点，该综述将可信图神经⽹络看作⼀个整体并分析了当前⽅法的局限性。为 了填补当前的研究空⽩，推动可信图神经⽹络的产业化，该综述提出了如下五个研究⽅向： 

## A. 拥抱可信设计理念（shift to trustworthy GNNs） 

构建可信的图神经⽹络需要研究⼈员与图神经⽹络从业⼈员全⾯拥抱可信理念，在设计图神经 ⽹络时不仅要考虑其任务性能，更需要将可信的理念引⼊到图神经⽹络的设计哲学中。已有的 ⼀ 些⼯作在设计中兼顾了可解释性和公平性，这⼤⼤提升了图神经⽹络的可信性。此外，解决 ⼀ 在转向可信图神经⽹络的过程中⾯临的 系列开放性问题，如在具体应⽤中对不同可信维度 （如⾃动驾驶中的鲁棒性和环境福祉）进⾏平衡和取舍，也是⼀个具有挑战性的研究⽅向。 

B. 探索其他可信维度（other aspects of trustworthy GNNs） 

可 信 图 神 经⽹络 所 包 含 的 内 容实 际 上 超 过 了 该 综 述 介 绍 的 六 个 维 度 。 例 如 ， 泛 化 能 ⼒ （generalisation）也被认为是可信系统的⼀个重要维度。当前⼀些研究探究了图神经⽹络外推 （extrapolation）与其使⽤的激活函数之间的关系，这些⼯作丰富了可信的内涵并促进了可信 ⼀ 图神经⽹络的构建。此外，该综述提出，妥善处理可信系统相关的设计原则（如国家新 代⼈ ⼀ —— ⼯智能治理专业委员会发布的《新 代⼈⼯智能治理原则 发展负责任的⼈⼯智能》）也是可 信图神经⽹络在未来发展中的重要研究内容。 

## C. 研究多样相互关系（diversified relations） 

⼀ 该综述只涉及了可信图神经⽹络不同维度之间复杂关系的 部分。探索其他相互关系（如可解 释性和公平性）对于全⾯理解和构建可信图神经⽹络系统⾄关重要。此外，这些关系不仅复杂 ⽽ 且 存 在 于 多 个 层 ⾯ 。 ⽐ 如 ， 反 事 实 公 平 （ counterfactual fairness ）与 鲁 棒 性 （robustness）在概念上具有相似性。因此，从概念、⽅法、成效等不同的层次探讨可信图神 经⽹络的不同维度之间的相互关系也是⼀个有前景的研究⽅向。 

D. 设计模型不可知⽅法（model-agnostic methods） 

当前，很多提升图神经⽹络可信性的⽅法需要使⽤专⻔设计的图神经⽹络架构。如果不能访问 或者修改⽬标⽹络架构（如使⽤云服务），这些⽅法将⽆法使⽤。这⼤⼤降低了这些可信性增 强⽅法在现实场景中的实⽤性。相⽐之下，模型不可知⽅法能够以即插即⽤的⽅式灵活地被应 ⽤到图神经⽹络系统中。此外，这类⽅法也能够以功能模块的形式被组合使⽤。因此，设计模 型不可知的⽅法将⼤⼤提升其实⽤性并促进可信图神经⽹络的构建。 

E. 建⽴可信技术⽣态（technology ecosystem for trustworthy GNNs） 

作为⼀个蓬勃发展的领域，可信图神经⽹络的发展离不开技术⽣态的⽀持。这⾥的技术⽣态包 含 但 不 限 于 ⼯ 具包 （ tools ） 、 数据 集 （ datasets ） 、 评 估 度 量 （ metrics ） 、 评 估 流 程 ⼀ （pipelines）。由于图数据的⾃身特性，当前的 些⼯具包（tools）如IBM的AI360可能⽆法 直接⽤于图神经⽹络的评估。例如，节点之间边的存在打破了节点上的独⽴同分布（IID）假 设，这就导致研究图神经⽹络公平性时需要考虑节点之间的相互依赖。此外，由于应⽤场景的 多样性，构建可信图神经⽹络也需要适⽤于不同任务和场景的数据集、度量、评估标准、软件 平台等配套技术设施。因此，建⽴相应的技术⽣态是可信图神经⽹络的研究和产业化中的关键 ⼀ 步。 

参考链接： 

[1] Trustworthy AI: A Computational Perspective. https://arxiv.org/abs/2107.06641 

[2] Governance Principles for the New Generation Artificial Intelligence--Developing Responsible Artificial Intelligence. https://www.chinadaily.com.cn/a/201906/17/WS5d07486ba3103dbf14328ab7.html 

[3] The montreal declaration of responsible ai. https://www.montrealdeclaration-responsibleai.com/thedeclaration 

阅读原⽂ 

喜欢此内容的⼈还喜欢 

研究笔记 | 基于AlexNet神经⽹络的城镇⽤地分类训练与应⽤ 

空间⼈⽂与场所计算 


![](raw/papers/4_Other_ML/可信AI/如何构建可信GNN？最新综述来了！可信图神经网络：维度，方法，趋势_images/如何构建可信GNN？最新综述来了！可信图神经网络：维度，方法，趋势.pdf-0019-08.png)


## ⽤神经⽹络判断奇偶数 

AINLP 


![](raw/papers/4_Other_ML/可信AI/如何构建可信GNN？最新综述来了！可信图神经网络：维度，方法，趋势_images/如何构建可信GNN？最新综述来了！可信图神经网络：维度，方法，趋势.pdf-0019-11.png)


