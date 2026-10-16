# HarnessDev：大模型能否创建并演化自己的 Agent Harness？

# HarnessDev: Can LLMs Create and Evolve Their Own Agent Harness?

- **arXiv 编号 / arXiv ID**：[2609.01437](https://arxiv.org/abs/2609.01437)
- **提交时间 / Submitted**：2026 年 9 月 1 日
- **作者机构 / Affiliations**：ByteDance Seed；Singapore University of Technology and Design；Georgia Institute of Technology；M-A-P；TokenWave.AI
- **核心作者 / Core Contributors**：Yuhao Wu, Jingyuan Zhang, Jiajun Shi
- **项目主页 / Project Page**：https://self-developing-agents.github.io/
- **说明**：本文为双语对照阅读辅助版本，采用「中文在前、英文在后」排版；英文部分为原文的关键要点摘引与结构化重述，用于对照阅读，完整原文请以 arXiv 上的官方 PDF 为准。

---

## 摘要 / Abstract

**中文翻译**

当 Agent 从科研原型走向落地工具时，其能力越来越依赖于位于模型权重之外的执行基础设施——业界一般称之为 **agent harness（智能体外壳/骨架）**。在固定模型权重的前提下改动 harness，也能显著改变任务表现。目前的 agent 评测通常是在选定 harness 之下报告下游任务成绩，而对"模型自己去开发 harness"的能力研究相对不足。本文提出 **HarnessDev**——一个把评测单位从"任务输出"迁移到"可运行基础设施"的基准。HarnessDev 覆盖两个阶段：在 **Creation（创建）** 阶段，agent 从一个极简的种子（seed）出发，仅借助少量开发用例，从零构建一套完整的执行系统；在 **Evolution（演化）** 阶段，它以自己创建的 harness 为起点，利用下游执行反馈迭代式地修改它，目标是提升基准分数。我们从两个维度评价所构造的 harness：**能力（capability）**——在留出集（held-out）上的任务成功率；**效率（efficiency）**——执行阶段消耗的 token 数量。所报告的 Creation 结果覆盖 6 个 creator LLM、4 个领域、5 个下游基准，共 2,207 个下游实例，其中隐藏评估任务对开发过程完全不可见。我们发现：所生成的 harness 在 code、search & research 两类任务上仍显著落后于成熟的人工工程系统，而在 writing 与机器学习实验类任务上则达到甚至超越所选的参照系统，且执行成本方差极大。演化阶段虽然能带来部分性能提升，但表现不稳定，且只能部分迁移到留出任务上。用固定 runtime 模型进一步做消融实验，可以看到 harness 收益强烈依赖于执行 harness 的具体模型，跨模型迁移能力有限。

**English Original (excerpt)**

> As agents move from research prototypes to deployed tools, their capability increasingly depends on model-external execution infrastructure, commonly termed the *agent harness*. Changing this harness while holding model weights fixed can substantially alter task performance. Current agent evaluations typically report downstream performance under a chosen harness, leaving a model's ability to develop the harness itself comparatively underexplored. We introduce **HarnessDev**, a benchmark that shifts the unit of evaluation from task outputs to runnable infrastructure. HarnessDev covers two stages. In **Creation**, the agent starts from a minimal seed and a small number of cases, then builds a complete execution system. In **Evolution**, it starts from its own created harness and iteratively revises it using downstream execution feedback, with the goal of improving benchmark performance. We then evaluate each constructed harness on **capability** — task success on held-out benchmarks — and **efficiency** — execution-token cost. The reported Creation results cover six creator LLMs, four domains, and five downstream benchmarks totaling 2,207 unique downstream instances, with hidden evaluation tasks withheld from development. Generated harnesses remain substantially behind mature human-engineered references on code and on search & research, while matching or exceeding the selected references on writing and machine-learning experimentation, with large variation in execution cost. Evolution produces some performance gains, but they are unstable and transfer only partially to held-out tasks. A fixed-runtime ablation further shows that the gains depend strongly on the model executing the harness, indicating limited transfer across models.

---

## 1. 引言 / Introduction

**中文**

当 Agent 从研究原型走向诸如代码助手、数据分析副驾驶、浏览器 Worker、研究流水线等真实工具时，其能力越来越依赖于位于模型权重之外的软件。这一执行基础设施——即所谓 **agent harness**——负责管理执行循环、工具调用、上下文管理、失败恢复以及结果验证，是把模型输出转化为可执行动作的关键。它对表现的影响是显著的：同样是 GPT-5，Terminal-Bench 2.1 在 Terminus 2 内的通过率为 35.2%，而在 Codex CLI 内则升至 49.6%。当 Agent 面向更多领域进行专精时，对量身定制的 harness 的需求会持续增长。由于此类系统需要持续开发而非一次性完成，一个具有实际价值的问题是：LLM 能否辅助 harness 工程师，甚至完全接管 harness 的构建与演进？

尽管现实需求强烈，绝大多数 agent 评测仍然是：为一次比较挑选一套 harness，然后在此之上报告下游任务表现。这样的实验设置便于任务级别的控制变量，但会把 harness 当作实验配置的一部分，而不是被开发的"制品"。近期已有一些工作研究 harness 的表示、自动化 agent 设计，以及能构建或改进 agent 系统的 agent，但"模型能否既能创建、又能持续改进可运行、可持久的 harness"这一问题尚缺乏系统评价。要回答它，必须把"开发 harness 的模型"与"执行下游任务的模型"分开、记录开发环境、并对下游表现、跨执行器迁移、与人工工程系统的差距、回归以及成本进行度量。

Harness 工程的根本难点在于：它与普通代码编辑不同。当模型改动一段独立程序时，目标行为在外部被规约，成功与否可以在本地验证；但当模型修改自己的 harness 时，它是在编辑"自己借以行动的执行基底"——这一改动会改变模型在此后所有任务中的观察、规划与恢复方式。因此，有效的 harness 改进要求模型能够从执行轨迹识别自身行为的局限、诊断所处系统的结构性瓶颈，并做出能累积成持久、可复用能力增益的、有针对性的修改。前沿模型已经具备编辑多文件仓库、闭合真实 pull request 的能力；缺的是一个能度量这种能力的基准。

HarnessDev（图 1）将评测单位从"任务输出"改为"可运行基础设施"：衡量模型构建并维护可长期使用、可审查、可复用执行系统的能力。基准分为两个阶段——**Creation** 从"故意做得很弱但可运行"的种子出发，构建面向一个新任务族的完整 harness；**Evolution** 从既有 harness 出发，用下游任务的执行反馈驱动其持续改进。评价 harness 比评价答案更难：harness 可能过拟合到写它的模型、把开发样本记住、在提升一项能力时悄悄退化另一项，或仅凭 benchmark 特定改动去刷高开发集分数。因此本文沿两个维度评价：**能力** 用留出下游任务上的成功率；**效率** 用被冻结 harness 在部署阶段消耗的 executor token 数。

**English (key claims)**

- The harness — the software surrounding a model — manages the loop, tools, context, memory, lifecycle and verification that turn model outputs into actions.
- With identical weights, GPT-5 solves 35.2% of Terminal-Bench 2.1 inside Terminus 2 but 49.6% inside Codex CLI, showing that the harness itself is a major performance lever.
- Existing benchmarks treat the harness as part of the experimental setup rather than as an artifact to develop.
- HarnessDev fills this gap by measuring whether a model can both **create** and **evolve** a runnable, persistent execution system, evaluated along a **capability** axis and an **efficiency** axis.

**主要发现摘要 / Findings at a glance**

- **Creation**：模型可以从弱种子构建可运行 harness，但与成熟人工系统的差距因 harness 类别而异；用"谁写谁跑"的方式，模型自建 harness 在写作类任务上接近参考、在机器学习实验上超越参考；在 search & research（长程信息搜集）和 code 上仍显著落后。不同 creator 生成的 harness 在下游分数和 token 消耗上都存在很大差异，且更高的执行成本并不必然带来更好的结果。
- **Evolution**：模型可以借助下游反馈改进自己的 harness，但可靠演化仍然困难：分数常常在多个版本间起伏；开发期观察到的增益，到留出集上通常缩小甚至消失；执行 harness 的运行时模型不同，最终结果差异也很大——同一个变更在自跑与固定 Gemini 跑之间可能一好一坏。

---

## 2. 背景 / Background

**中文**

大多数 agent 基准起点都在"问题已被工程化"之后：任务已被规约、奖励或裁判已被定义、执行骨架已被固定。这种设定利于受控比较，却把主导真实部署工作量的那部分工作藏了起来。在工业界，这类工作分散在解决方案架构师、应用工程师、平台工程师等多种角色中，其中一个近年来最显性化的头衔是 Palantir 提出、随后被前沿模型公司广泛采纳的 **Forward-Deployed Engineer（FDE）**。FDE 在系统落地后进驻客户现场，把一个通用模型改造成能在该客户的数据格式、业务流程与合规约束下真正运转的系统——例如因为日志只能保留固定期限而重写数据摄取路径，或者定位跨司法辖区合同流水线中的失败点。评价 FDE 是否成功，看的不是 demo 或榜单分数，而是所部署的系统是否真正被使用、是否持续可用、是否随反馈迭代改进。

从模型角度看，FDE 工作提供了 benchmark 设计者通常预设好的三层结构：**目标是模糊的**（一个不正式的业务意图必须被翻译成具体目标、约束与成功标准）；**反馈信号是缺失或不可靠的**（在能判断"系统是否变得更好"之前，测试、裁判、追踪或其它自我评估必须先被构造出来）；**执行系统在可用形态上并不现成**（工具、上下文管理、状态、生命周期逻辑、验证接口都必须被构建、适配并"持续维护"——FDE 是长期陪伴系统，而不是"交付即撤"）。

本文聚焦第三层。传统受控 agent 评测中，研究者选定一个 agent 配置并在该配置下汇报任务完成率，此时 harness 通常是"评测环境"而非"被开发对象"。HarnessDev 则询问：语言模型能否从一个弱起点构造这套执行骨架，并在保持约束合规与留出集性能不退化的前提下，用反馈改进它？与之相对：**Aspire** 研究广泛的部署需求如何转化为能力增长，**S3Gym** 研究交互经验能否被判断并复用，而 HarnessDev 隔离出的是"模型如何构建并维护承载它们的系统"这一环。

**English (key ideas)**

- Real deployment work is often invisible in benchmarks; forward-deployed engineers (FDEs) close the gap between "capable model" and "working system".
- FDE work exposes three axes benchmarks usually presuppose: **vague target**, **absent/unreliable feedback**, **execution system not yet usable**.
- HarnessDev studies whether models can build and maintain that execution scaffold, complementing Aspire (targets) and S3Gym (feedback/experience).

---

## 3. 基准设计 / Benchmark

### 3.1 总览 / Overview

**中文**

HarnessDev 评估的是模型开发出的执行系统，而不是它对某一次任务给出的答案。提交物是一个**可运行的 harness**，先被"冻结"，再在下游任务中重复使用。一个 creator LLM $L_C$ 在开发环境 $D$ 内工作，产出可运行的 harness $H$。开发信号因设定而异（Creation vs. Evolution）。$H$ 被冻结后，executor LLM $L_E$ 在其中运行下游任务 $x$，评审 $J$ 对输出 $y$ 打分：

$$(L_C, D)\to H,\quad (H, L_E, x)\to y \xrightarrow{J}\text{score}.$$

也就是说，$D$ 用来"造 $H$"，$L_E$ 只在 $H$ 冻结**之后**才登场使用。实现上，$H$ 包含执行循环、工具、上下文管理、持久状态、生命周期控制、验证六个组件。基准聚焦两个研究问题：

- **RQ1（Creation）**：模型能否从"弱但可跑"的种子出发，构建有效 harness？——把任务规约与少量开发用例翻译为可迁移到未见任务的执行基础设施。
- **RQ2（Evolution）**：模型能否在保留既有能力的前提下改进自己已经有的 harness？——用自己在 RQ1 中构建的 $H_0$ 出发，基于下游反馈迭代改进；同时分析编辑统计、反馈响应、留出泛化、跨 executor 迁移等指标。

### 3.2 开发设定 / Development settings

所有设定都给 creator 一个可写的开发工作区，区别在于起点 harness 与可用信号：

| 设定 | 起点 | 开发信号 | 输出 |
|------|------|----------|------|
| **Creation** | 弱种子 $H_{\text{seed}}$ | 规约 + 1–3 个开发用例 | 最终 harness $H$ |
| **Evolution** | Creator 自己 RQ1 的 $H_0$ | 指定反馈集的运行结果 | 冻结的成对候选 + creator 声明的最终 harness |

**弱种子 $H_{\text{seed}}$** 是一个可跑的"兼容层"，而不是能解题的 agent：它解析任务与模型配置、暴露底层被动工具、写出结果 / 轨迹 / 日志 / 任务制品。种子里没有任何 agent 循环、任务分解、工具策略、上下文管理、持久状态、验证器、重试恢复或停止规则；未经修改时它对每个下游 benchmark 得分都是零。因此任何非零 Creation 分数都必须来自 creator 添加的执行逻辑。

**Creation（RQ1）**：creator 除了种子外，还会拿到任务族规约、工具与权限约束、简短的设计教程，以及 1–3 个开发用例。它可以基于这些用例的反馈修改 harness，但看不到人类实现，也看不到隐藏评估集。评估之前 harness 会被冻结。

**Evolution（RQ2）**：从 creator 自己冻结的 RQ1 code harness $H_0$ 出发。开发期可以看到一个固定的 100 任务 SWE-Pro 反馈集，以及 89 个 Terminal-Bench 全部任务。控制器为 creator 提供 10 个 "post-$H_0$ 完整评测对" 的预算：每提交一个候选，就会在两个基准上都跑一遍完整评测；两条腿都收敛后候选才进入"正式轨迹"。两次计费评测之间还能用 2 次固定子集的 probe，仅作诊断，不计正式分数。轨迹结束后，每个正式版本还会在与反馈集**完全不相交**的 630 个 SWE-Pro 实例上做一次留出评测，这些分数不会返回给 creator。

### 3.3 领域与下游基准 / Domains and downstream benchmarks

Creation 覆盖四个领域、五个下游基准；Evolution 目前聚焦 code。共有 2,207 个独立下游实例。

| 领域 Domain | 基准 Benchmark | 任务数 | 主指标 |
|---|---|---|---|
| Code | SWE-bench Pro（简写 SWE-Pro，公共 split） | 731 | 任务成功率 |
| Code | Terminal-Bench 2.1 | 89 | 任务成功率 |
| Data analysis | MLE-bench | 75 | 奖牌分 |
| Writing | EQ-Bench3 | 46 | 评分打分 |
| Research | BrowseComp | 1,266 | 准确率 |

### 3.4 评估协议 / Evaluation protocol

所有分数都由**冻结的 harness** 在标准化 runtime 中产生。executor $L_E$ 与评审 $J$ 在每一次对比中保持固定，因此分数变化只反映 harness 本身的差异。Creation 提供两种模式：**Self-Eval** 让 $L_E = L_C$，评价 creator 与 harness 组成的完整系统；**Unified-Eval** 强制所有 harness 使用同一个 executor（本文用 Gemini 3.1 Pro），从而隔离 executor 兼容性问题。人工工程系统作为"外部参考"，展示与成熟系统的距离，但不与其它 executor 严格配对。

**约束合规**：规约明确禁止硬编码实例特定解、从任务 ID 反推 patch、通过 file-name allowlist 或已知答案作弊、访问隐藏测试、隐藏答案、隐藏 patch、评审内部实现或官方反馈，也不允许替换所提供的 provider-neutral runtime 接口。两条属性保证这些约束可核查：一是打分路径与 harness 隔离——harness 自报状态不参与打分，SWE-Pro 只看真实仓库 diff、Terminal-Bench 只看最终环境状态；二是所有运行都保留轨迹、结果与指标制品供事后审计。本文所报告的所有运行都被审计，未发现任何以违规路径获取分数的情况。

### 3.5 指标 / Metrics

每个冻结 harness 会汇报两个量：下游任务在其原生指标下的表现，以及评测阶段消耗的 executor token 数（同时报告总量和每任务均值）。**creator 用于构建 / 修改 harness 的 token 不计入执行成本**。Creation 阶段以弱种子和可得的人工参照作为对比；Evolution 阶段在同一 executor 和评审下汇报相对 $H_0$ 的变化。

**English (contract summary)**

- The unit of evaluation is a **frozen, runnable harness**; scores are produced by that harness under a standardized runtime.
- **Self-Eval** measures the creator–harness system as a whole; **Unified-Eval** fixes the executor across harnesses.
- Prohibited behaviours (hard-coded answers, peeking at hidden tests, replacing the provider-neutral runtime, etc.) are audited post-hoc; **no violation was observed** in the reported runs.
- Efficiency is measured in **executor-model tokens**; creator/build tokens are excluded.

---

## 4. 实验 / Experiments

### 4.1 实验设置 / Setup

**中文**

评估 6 个 creator LLM：Opus 4.8、GPT-5.5、Gemini 3.1 Pro、DeepSeek V4 Pro、Qwen 3.7 Max、Seed 2.0 Pro。开发环境 $D$ 除 GPT-5.5 使用 Codex 0.144.3 外，其余均使用 Claude Code 2.1.177。所有 creator 都以"高推理强度 + 流式输出"运行，输出长度设为 endpoint 上限。RQ1 中，每个 creator × benchmark 组合独立构建并评估 3 个 harness，取 avg@3；Self-Eval 中 creator 同时充当 executor；Unified-Eval 中所有 harness 由 Gemini 3.1 Pro 运行以隔离 executor 兼容性。

### 4.2 Harness Creation（RQ1）

**总体结论 / Overall findings**：Creation 质量因任务族而异。在 Self-Eval 下，Opus 4.8 综合最高（67.8），但仍低于人工参照（86.2）。Writing 类 harness 接近参照；Search 差距最大，Code 也仍落后。Opus 4.8 与 Gemini 3.1 Pro 在 MLE-bench 拿到 32.9/32.4 的奖牌率。在失败的 Data 任务中 77.8% 归因于 harness 缺陷，说明瓶颈并非只是 executor 能力。

**代表性数据（Self-Eval，avg@3；成本单位为 M tokens/harness）**

| Creator | SWE-Pro | Term.-2.1 | MLE-bench | EQ-Bench3 | BrowseComp | Avg. |
|---|---|---|---|---|---|---|
| Seed harness | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| Opus 4.8 High | 69.3 | 64.8 | 32.9 | 84.6 | 52.4 | **67.8** |
| GPT-5.5 High | 32.8 | 52.1 | 19.1 | 83.0 | 52.6 | 55.1 |
| Gemini 3.1 Pro High | 43.6 | 68.8 | 32.4 | 74.8 | 35.2 | 55.6 |
| DeepSeek V4 Pro High | 28.9 | 35.6 | 19.6 | 75.4 | 40.9 | 45.2 |
| Qwen 3.7 Max | 33.5 | 41.3 | 3.1 | 68.7 | 32.3 | 44.0 |
| Seed 2.0 Pro High | 10.8 | 6.0 | 5.3 | 71.1 | 3.2 | 22.8 |
| Human harness + paired model | 80.0\* | 88.8\* | 24.0 | 83.7 | 92.2\* | **86.2** |

（\* 表示外部公开系统级结果，未在本实验中复跑；—表示无。）

**表现差异与 executor 依赖 / Performance variation and executor dependence**：同一个模型独立生成的 harness 之间也可能大不相同——例如某个 Opus code harness 在 Self-Eval 下表现良好，但在 Gemini executor 下几乎崩溃，原因是它硬编码了针对原 executor 的 120 步上限；Data 场景中也出现过因停止规则过严导致的类似崩塌。这些例子说明"报告一个 harness"不足以代表模型能力，因此需要 avg@3。

固定 executor 后排名发生明显变化：Qwen、Seed、DeepSeek 的部分 Data/Search 设置在 Gemini 下有所改善；而 Opus 与 GPT-5.5 用自己的 executor 通常更强。因此 **Self-Eval 反映的是"harness 设计、executor 能力、两者兼容性"三者共同作用**。成本也高度不均：MLE-bench 上不同 harness 的 token 消耗差异接近 19 倍，但更高成本并不必然带来更高分。

**实现行为 / Implementation behavior**：不同 creator 的实现策略差异明显——Opus 往往重写执行栈；GPT-5.5 添加一个庞大的单体 agent；DeepSeek、Qwen、Seed 在种子基础上扩展 agent、tool、context、state 模块；Gemini 大多就地编辑 runner。18 个 Code 制品共净增 17,111 行代码，但**编辑量并不预示性能**：Gemini 只加了 1,006 行，却拿到最高的 Terminal-Bench 分数（68.8），提示"聚焦的改动 + 频繁验证"比代码量更重要。所有 18 个 Code harness 都实现了显式执行循环；工具、生命周期控制、验证分别在 13/18、13/18、15/18 中完整实现；**状态与记忆是最明显的缺口**——11/18 定义了 State 类，只有 1 个暴露状态保存接口，只有 1 个实现周期性 checkpoint；在 26,679 条记录的任务轨迹中未出现任何 checkpoint 事件。此外，硬编码的步数/输出上限、以及大多为语法级别的验证，都容易在换 executor 或数据格式后失效——例如 2,325 个执行 Data 任务里有 441 个产生退化提交，但没有 harness 能检测出来。

**部分机制形同虚设 / Dead code**：Code 类 108 个组件实例中，72 个在真实运行中被触发，18 个只有部分证据，18 个从未被观察到（全部集中在 state 与 memory）。Writing 587 个特性中确认 124 个是死代码，Data 中有 36 个机制走在死路上。**自测次数本身不是强信号**（与下游分数的 Spearman 相关只有 0.13–0.26 且不显著），但**修订调用**与下游分数的相关性达 0.57（$p\le .0005$）——只有"读失败 → 精准修改 → 再验证"这样闭环时，测试才起作用。

**执行器迁移 / Executor transfer**：可迁移性因 harness 而异。若干 Qwen 与 DeepSeek 的 harness 换到 Gemini 后反而提升，说明原 executor 是瓶颈；Qwen 在 BrowseComp 提升 17.6 分，在 MLE-bench 提升 12.9 分。Opus 则相反：SWE-Pro 从 69.3 掉到 33.0，Writing 从 84.6 掉到 74.2；其 Search harness 换 executor 后重复查询率从 10.1% 飙升到 88.2%，说明它的去重、复核、终止规则是针对原模型调好的。**一个能跑的 harness 可以被别的模型使用，但只有当提示、工具协议、预算、停止规则仍然兼容时，能力才会跟着迁移。**

### 4.3 Harness Evolution（RQ2）

**总体结论**：所有 5 个 self-runtime creator 都能在可见反馈对上提升，但增益在留出集上会缩小。Opus 4.8 的留出提升最大（+4.44 分）。在固定 Gemini executor 下，迁移更弱——只有 Opus 在留出集上还有提升，其它三条支线都出现退化。**Evolution 可以带来局部有用的改动，但增益偏小，且常常特化到当前 executor 或反馈集。**

**评估协议**：报告成对分数

$$\bar P_t = \tfrac{1}{2}\left(P_t^{\text{SWE100}} + P_t^{\text{Term89}}\right),$$

其中 $t$ 索引完成正式评测的冻结版本。每条支线从各自 RQ1 的 $H_0$ 起步；跑 5 条 self-runtime 支线、4 条固定 Gemini 消融。9 条支线共产生 73 个正式版本、64 次版本切换。所有轨迹结束后，每个正式版本额外在与反馈集不相交的 630 个 SWE-Pro 实例上评测——这些分数从不返给 creator。

**RQ2 反馈集增益与冻结后的留出泛化**

| 设定 | Creator | 反馈对 $H_0\to H_{\text{dec}}$ | Held-out-630 $H_0\to H_{\text{dec}}$ | 与最佳留出的差距 |
|---|---|---|---|---|
| Self | Gemini 3.1 Pro | 59.9→68.7 (+8.8) | 48.89→51.59 (+2.70) | 0.00 |
| Self | Opus 4.8 | 71.1→74.1 (+3.0) | 63.02→**67.46** (+4.44) | 1.59 |
| Self | Qwen 3.7 Max | 41.8→55.7 (+13.9) | 42.22→43.65 (+1.43) | 3.17 |
| Self | DeepSeek V4 Pro | 47.2→60.6 (+13.4) | 47.30→50.48 (+3.17) | 1.75 |
| Self | GPT-5.5 | 59.2→65.1 (+5.9) | 48.25→52.06 (+3.81) | 0.00 |
| Fixed Gemini | Opus 4.8 | 58.8→68.6 (+9.7) | 48.10→50.79 (+2.70) | 2.54 |
| Fixed Gemini | Qwen 3.7 Max | 62.1→63.2 (+1.1) | 49.52→48.41 (**−1.11**) | 1.11 |
| Fixed Gemini | DeepSeek V4 Pro | 47.3→53.8 (+6.5) | 43.02→40.63 (**−2.38**) | 3.02 |
| Fixed Gemini | GPT-5.5 | 56.6→59.1 (+2.4) | 42.22→31.90 (**−10.32**) | 16.51 |

**编辑与反馈使用 / Editing and feedback use**：中位数的最终版本修改 8 个文件、+476/−38 行。64 次官方切换里，58 次改执行或控制流、37 次改工具、17 次改生命周期恢复、16 次改上下文，只有 4 次改状态；**没有一次切换动过独立验证器**。编辑量并不可靠预示改进；同一个 creator 在不同 executor 下会采取不同策略。**演化更像"在运行时反馈周围做局部程序搜索"**，删除有时和添加一样有用。9 条支线中有 8 条至少完成过一个完整循环（读结果 → 编辑 → 再评测 → 版本选择），但**失败诊断仍是最弱一环**：专门的轨迹检查接口仅被调用 2 次；显式检查的用例覆盖率因支线不同只有 0.5%–40.2%。creator 更多依赖自写脚本和小规模 probe，两者都可能与完整评测意见相左（例如某个 GPT-5.5 候选通过全部 5 个 Terminal probe，但在完整集上只得 0.584）。Opus 提供了最清晰的正面案例：它发现 100 次运行中 99 次自报成功但只有 48 次真正通过，追溯到"过早完成"并加入完成检查——**反馈只有在暴露具体失败模式并且改动被端到端验证时才最有用。**

**稳定性与最终版本选择 / Stability and selection**：Evolution 并非单调。在 64 次切换里，8 次两个 benchmark 都退化，16 次单基准退化，3 次跨基准折衷，7 次无可测变化，27 次落在重复运行噪声带（±4.75 pair-score）内，只有 2 次落在带外，1 次不含可执行代码变更——**小增益无法仅凭分数归因于代码改动**。新增代码也未必被真正执行：169 个新增函数或类中只有 113 个从入口可达，31 个只经死代码可达，25 个无调用者。Opus 的完成门是有路径证据的正面案例；Qwen 的消息 sanitizer 则相反，会破坏合法的 Gemini 工具消息序列。creator 通常挑选反馈分最高的候选，但这一选择很少与"最佳留出版本"重合：64 次可比切换里反馈和留出方向一致的只有 34 次（53.1%），9 条支线中最终声明版仅有 2 条恰好是留出最优。**因此可见反馈更适合驱动"局部搜索"，而不适合用于"最终选择"**——重复优化一个带噪信号会偏爱幸运运行，放大过拟合。

**English (evolution takeaways)**

- All five self-runtime creators improved on the visible feedback pair, but held-out gains were smaller; only Opus 4.8 achieved a non-trivial held-out gain (+4.44 pp).
- Under a fixed Gemini executor, only Opus still improved; the other three lineages regressed on held-out tasks.
- Edits cluster on execution/control flow, tools and lifecycle; state and verifier code is rarely touched.
- Feedback-set score and held-out score moved in the same direction only ~53% of the time.

---

## 5. 相关工作 / Related Work

**中文**

HarnessDev 处于 agent benchmarking、自动化 agent 构造、harness 演化三条脉络的交叉点。SWE-bench、GAIA、WebArena、τ-bench、AgentBench 等基准把任务、环境、评分标准化，但通常在选定的 harness 下评价任务执行；Harness-Bench 关注"换 harness 对模型表现的影响"；Meta-Agent Challenge 让 meta-agent 在沙盒里迭代地为 agent 编程，并用 5 个领域的隐藏测试打分，与 Creation 十分相关。**HarnessOpt-Bench** 是与 Evolution 最接近的同期基准：优化器拿到 seed harness、分级反馈和固定预算，环境用不可访问的测试划分打分；与之不同，HarnessDev 把 Creation 与 Evolution 打通，把 in-trajectory 分数用作反馈，之后再把每个冻结版本放到 630 个不相交任务上，把"对反馈的适应"与"留出泛化"沿轨迹分开。**Evo-Bench** 同样评估 Evolution，但在固定 runtime 下评估每条支线的最终版本，属"跨领域最终版质量"视角；HarnessDev 则同时研究反馈驱动轨迹沿途的留出表现变化，并额外覆盖跨 executor 迁移与 token 成本。

自动化 agent 设计与演化方向：Claude Code、Codex、OpenHands、SWE-agent 等人工工程系统组合了执行循环、工具、上下文管理与恢复机制；ADAS、AFlow、MASS、EvoAgentX 在 prompt、workflow、operator、agent 拓扑上进行搜索；VeRO、Meta-Harness、Self-Harness、HarnessFix、DemoEvolve、HarnessCompass 等直接优化可执行 harness 代码；Harness-R1 训练"harness 工程师"把失败批次转成经过验证的补丁。这些工作提出方法或基础设施；**HarnessDev 关注的是通用前沿模型在同一套 Creation-Evolution 协议下作为"开发者"的表现**。

评估 harness 演化：单看最终任务增益容易把"知情诊断"与"盲搜"混淆。Priority-ranking evaluation 检验优化器是否识别了最值得修改的组件；"Harness Updating Is Not Harness Benefit"把"生成有用更新"与"executor 是否能利用它"分开——正是这一动机促使本文引入 creator/executor 分离与 Self-/Unified-Eval 双视角。SEAGym 展示后续更新未必保留留出增益；受同预算约束的对比研究也发现 harness 演化可能过拟合搜索基准而不比更简单的搜索 baseline 强。

**English (positioning)**

- HarnessDev is closest to concurrent work such as HarnessOpt-Bench (Evolution-style optimization of a given seed) and Evo-Bench (Evolution under a fixed runtime), but uniquely couples Creation and Evolution, records execution-token cost, and tests transfer across runtime models.
- It complements methodological work (Self-Harness, HarnessFix, DemoEvolve, HarnessCompass, Harness-R1, …) by measuring **how well general frontier models play the developer role** under one protocol, rather than proposing a new optimization method.

---

## 6. 讨论、局限与结论 / Discussion, Limitations, Conclusion

### 6.1 讨论 / Discussion

**中文**

结果说明"harness 开发本身值得直接评估"。Creation 表现按领域差异极大：在 Self-Eval 下，当前模型在 writing 上匹敌人工参考、在机器学习实验上超越参考，但在 search & research 上还差得远，在 code 上也仍落后。跨 executor 对比显示：有些 harness 在更强 executor 下会改善，另一些则出现"creator 与 executor 的共适应"，换 executor 就掉链子。Evolution 更难：有用的中间更新常被后续修改抹掉，更多更新并不保证正的最终增益。这些结果把"持久执行系统的质量"与"跑在其中的模型的能力"两件事分开来。固定 Gemini 的 Evolution 消融进一步强化了这一点：仅换 runtime 绑定就能显著改变 $H_0$ 表现，也会改变"哪些 harness 变更是有益的"。

### 6.2 局限 / Limitations

**中文**

四个领域覆盖了很多但不是全部真实部署；人工 baseline 参差不齐且不保证最优；Unified-Eval 能减弱但不能完全消除 executor 差异，因为 harness–模型交互本身复杂；行为分析是描述性的，受基准覆盖度限制。Evolution 目前每个 creator × runtime 只有一条支线，主 runtime 下有一格未完成；冻结后留出评测仅覆盖 SWE-Pro，因此不能给出不确定性估计或群体级比较。开发环境 $D$ 在两阶段均保持固定；"演化后的 harness 能否反过来作为下一轮开发环境"留待未来工作。此外，HarnessDev 度量的是**模型外部**的学习，**不主张启发式学习可以替代参数训练**。

### 6.3 结论 / Conclusion

**中文**

HarnessDev 把 agent 评测从"模型能否在固定系统内解决任务"推进到"模型能否构建并维护解决未来任务的系统"。通过四类人工 baseline、从零构建、反馈驱动演化、Self- 与 Unified-Eval、以及与人工参照的对比，它把"agent 构建的执行 harness"变成了可度量的对象。**如果模型权重是智能积累的一处，那 harness 就是另一处**——显式、可审查、可测试、可复用，能在失败、反馈与真实工程压力下持续改进。

**伦理声明 / Ethics**：基准由公开可用的 benchmark 套件与开源 harness 构成；不收集人工内容。自动化 harness 构建存在放大不安全工具使用的风险，因此基准显式声明了 harness 不得做什么，并在每次运行后审计；本文所报告的运行未观察到违规。所有 Creation 与下游任务在容器内执行，但该边界是为可复现而设，不是为隔离；任何复用生成的 harness 的人都应视其为不可信代码并进行更严格隔离。

**English (concluding position)**

- HarnessDev makes agent-built execution harnesses a **measurable, comparable object**, not just an experimental setting.
- Creation quality varies sharply by domain; Evolution is harder and requires disentangling adaptation to feedback from held-out generalization and cross-executor transfer.
- The harness is a legitimate locus of accumulated capability, complementing model weights: explicit, inspectable, testable, reusable, and continually improvable through failure and real engineering pressure.

---

## 7. 附录要点摘要 / Appendix Highlights (selected)

- **Harness 接口规约**：所有合规 harness 需实现六个功能模块——`execution.py`（run/step）、`tools.py`（register/call）、`context.py`（build/compress）、`state.py`（save/load/resume）、`lifecycle.py`（beforeAction/afterAction/onFailure/onTimeout）、`evaluation.py`（evaluate/recordTrajectory），所有方法返回 JSON 可序列化对象。
- **候选 harness 系统**（Creation 参考池）：Code 类 Claude Code / OpenCode / OpenHands / SWE-agent / mini-SWE-agent；Notebook/Data DataAgent、DB-GPT；Writing AutoResearchClaw、webnovel-writer；Research/Retrieval Alibaba-NLP/DeepResearch、dzhng/deep-research、modelscope/ms-agent、gpt-researcher。
- **人工参照系统**（表 9 摘要）：SWE-Pro → Claude Fable 5；Terminal-Bench 2.1 → GPT-5.6 Sol；MLE-bench → MLEvolve + Gemini 3.1；EQ-Bench3 → Kimi Writer + Opus 4.8；BrowseComp → GPT-5.6 Sol。这些是官方公开榜单/系统报告可核验的最佳 system-level 结果，非"同一 executor 下的配对控制"。
- **执行成本**（图 9）：相似质量的 harness 之间 token 成本能差近一个数量级——例如 MLE-bench 上 GPT-5.5 用 29.3M tokens 拿到 19.1 奖牌率，而 DeepSeek V4 用 208.4M tokens 拿到 19.6。因此论文直接汇报 performance–token 权衡而非折算为单一"成本调整分"。
- **RQ2 case studies（附录 F–I）**：包括 GPT-5.5 与 Opus-4.8 在 RQ1 Code 与 RQ2 Code 上的完整自测台账——从合同 smoke、SWE/Terminal 诊断运行，到"最终版本选择"（Opus 因估计的 ±3–4 任务噪声主动放弃继续采样，选 T3 定型；GPT-5.5 在 T7 结束后 13 秒确认 T2 仍是 argmax，放弃剩余 3 个 pair 预算）——为"演化更像局部程序搜索"的论点提供了细粒度证据。

**English summary of appendices**

- The six-module interface contract (execution / tools / context / state / lifecycle / evaluation) fixes the "what a submitted harness must expose" boundary without dictating internal architecture.
- Human references pair a mature harness with an executor from that system's own release (e.g., Claude Fable 5 for SWE-Pro), and are documented as **external system-level references**, not paired controls under a common executor.
- Cost–performance trade-offs span roughly an order of magnitude at comparable quality, motivating the two-axis (capability, efficiency) reporting.
- Detailed RQ1/RQ2 case ledgers illustrate the observed pattern: creators often complete edit–run–verify loops but frequently miss the causally correct diagnosis, and reliable version selection under noisy feedback is a live open problem.

---

## 引用 / Citation

```
@misc{wu2026harnessdev,
  title  = {HarnessDev: Can LLMs Create and Evolve Their Own Agent Harness?},
  author = {Wu, Yuhao and Zhang, Jingyuan and Shi, Jiajun and others},
  year   = {2026},
  eprint = {2609.01437},
  archivePrefix = {arXiv},
  primaryClass  = {cs.SE},
  url    = {https://arxiv.org/abs/2609.01437}
}
```
