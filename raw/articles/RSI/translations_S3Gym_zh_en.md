# S3Gym：大模型能否把自测试和自判断转化为自我改进？

# S3Gym: Can LLMs Turn Self-Testing and Self-Judging into Self-Improvement?

- **arXiv 编号 / arXiv ID**：[2608.31100](https://arxiv.org/abs/2608.31100)
- **提交时间 / Submitted**：2026 年 8 月 31 日
- **作者机构 / Affiliations**：ByteDance Seed；Singapore University of Technology and Design；Georgia Institute of Technology；M-A-P；TokenWave.AI
- **项目主页 / Project Page**：https://self-developing-agents.github.io/
- **说明**：本文为双语对照阅读辅助版本，采用「中文在前、英文在后」排版；英文部分是原文关键论点的结构化重述与摘引，用于对照阅读；完整原文请以 arXiv 上的官方 PDF 为准。

---

## 摘要 / Abstract

**中文翻译**

一个具备判断力的 agent 应能够在真实执行经验中提炼出可复用的策略：自我测试其行为、判断有效之处、并把判断转化为在未来任务上更好的表现。本文提出 **S3Gym**，一个用于评估 LLM agent 能否闭合"**Self-Testing、Self-Judging、Self-Improvement**"循环的基准。基准包含 7 个截然不同的文本游戏，覆盖国际象棋、扫雷、置零（Nullify）、俄罗斯方块、贪吃蛇、植物大战僵尸和信任演化——每个游戏都提供丰富的动作空间与可核验的反馈。评估分两阶段：**探索阶段（Exploration）**——agent 与游戏交互、生成经验、并选择如何加以利用；**评估阶段（Evaluation）**——用一个严格的留出集来度量真正的能力提升。我们研究三条经验整合路径：**History In-Context Learning（历史轨迹的上下文学习）**、**Summary Memory（把经验压缩成规则/技能）**、**Parameter Training（把经验蒸馏进权重）**。结果表明：**自我改进既非自动也不均匀**——Summary Memory 在经验能被压缩成明确策略规则的任务上有效；对依赖精确状态信息的游戏，raw history 通常更好；参数训练在部分任务上带来显著收益，但结果不稳定且可能引发负迁移。跨模型（Opus 4.8、GPT-5.5、Gemini 3.1 Pro、DeepSeek V4 Pro、Qwen 3.7 Max、Seed 2.0 Pro）与七款游戏的实验一致显示：**准确识别成功动作不足以保证可迁移的改进**，只有把这种识别有原则地加以整合，才能持续地把经验转化为能力。

**English Original (excerpt)**

> An agent with judgment must extract reusable strategies from its own execution experience: it must self-test its behavior, judge what worked, and turn that judgment into better performance on future tasks. We introduce **S3Gym**, a benchmark evaluating whether LLM agents can close the loop from **Self-Testing** and **Self-Judging** to **Self-Improvement**. S3Gym comprises seven distinct text games — Chess, Minesweeper, Nullify, Tetris, Snake, Plants-vs-Zombies and Trust Evolution — each providing rich action spaces and verifiable feedback. Evaluation proceeds in two phases: an **Exploration** phase, where agents interact with games, generate experience, and choose how to leverage it; and an **Evaluation** phase with a strict held-out set that measures genuine capability gains. We study three integration pathways: **History In-Context Learning**, **Summary Memory**, and **Parameter Training**. Results across leading models and games show that **self-improvement is neither automatic nor uniform**: Summary Memory works when experience compresses into strategic rules, raw history is often better for games needing precise state information, and parameter training yields substantial gains on some tasks yet remains unstable and prone to negative transfer. Accurate identification of successful actions is not sufficient for transferable improvement — only principled integration turns experience into capability.

---

## 1. 引言 / Introduction

**中文**

真正有用的 agent 必须能从自己的经验里做出**判断**：自我测试其行为、判断哪些行为有效、并把这些判断转化为未来任务上更好的表现。研究者已经清楚地观察到，当代 LLM 具备一定的"内在监督"能力——它们可以对自己的输出打分、拒绝不佳答案、给出可用的选偏好信号。真正的问题不再是"能不能识别正确性"，而是这种识别**能否被组织为持续的、可保留的能力提升**。

本文把这条问题拆成一个闭环：**Self-Testing（自测）→ Self-Judging（自判）→ Self-Improvement（自改）**。

- Self-Testing：agent 主动与环境交互、生成能暴露其行为的证据；
- Self-Judging：从这些证据中提炼出对成功/失败的判断，形成可指导后续决策的信号；
- Self-Improvement：把这些判断整合进 agent 的能力，使它在未见任务上做得更好。

问题的难点在最后一环。识别一次正确的走子、发现一步好的落子，并不等于以后能稳定选出好走子；把判断沉淀成可迁移能力，需要**明确的整合机制**：无论是把经验保留在上下文、压缩成 memory 中的规则/技能，还是蒸馏进模型权重。

S3Gym 就是围绕这三步循环设计的。它用 7 款文本游戏覆盖差异极大的判断/规划挑战，每款都能自动、可核验地给出结果反馈；两阶段协议保证 Evaluation 阶段严格留出、只测量真正的迁移。我们把整合分成三条路径：**History ICL**（拿原始轨迹当上下文）、**Summary Memory**（把经验压缩进 memory）、**Parameter Training**（用经验做参数级更新，如监督微调 SFT 或强化学习 RL）。

关键结论：**自我改进既非自动也不均匀**——Summary Memory 在"策略可压缩为规则"的游戏（如 Trust Evolution）中有效，但在需要精确状态记忆的游戏（Chess、Minesweeper）里表现较弱；Parameter Training 有些游戏收益显著（Snake / Nullify），有些则出现明显负迁移。要把 Self-Testing/Judging 变成 Self-Improvement，**如何整合经验**与**能不能识别成功**同样重要。

**English (key claims)**

- Modern LLMs can *identify* correctness reasonably well; the open question is whether they can *turn that identification into retained, transferable capability*.
- S3Gym operationalizes the self-testing → self-judging → self-improvement loop with seven diverse text games that admit rich actions and verifiable rewards.
- Three integration pathways — History ICL, Summary Memory, Parameter Training — expose sharply different failure modes; none dominates across games.

---

## 2. 相关工作 / Related Work

**中文**

**自我改进 LLM 的研究谱系**：一大类工作沿着"生成-测试-反思"的方向展开——包括自我一致性、Self-Refine、Reflexion、Chain-of-Verification、Self-Rewarding、RLAIF 等，通常在单轮或短交互内让模型评价并修正自己的输出。另一类工作把 memory 显式化，如 Voyager 的技能库、Generative Agents 的记忆流、ExpeL 的策略蒸馏，把过去经验组织成后续调用可以复用的结构化知识。参数级路径覆盖了通过自生成数据做 SFT/DPO/RL 的多种做法，从 RLHF 到 STaR、Self-Play、Constitutional AI 均属此列。

**Agent 评测 / Benchmark**：现有基准要么关注单轮问答的正确性（如 MMLU、GSM8K）、要么关注长程任务完成（SWE-bench、WebArena、GAIA、τ-bench），一般假定 agent 在评估时不带任何针对本任务的经验。少数工作研究"经验能否被复用"，但多在同分布内、单个环境内。**S3Gym 与它们的差别在于**：一是显式区分 Exploration 与 Evaluation，让"是否学到"和"能否迁移"分开衡量；二是把经验整合"路径本身"作为受试变量，直接比较 History / Memory / Parameter 三条路径；三是覆盖多种可核验反馈机制的游戏，从而给出关于"哪类任务被哪种整合方式受益"的可比结论。

**English (positioning)**

- Prior work has extensively studied self-refinement, memory-based experience reuse, and self-generated training data separately.
- S3Gym compares these three integration pathways on the *same* set of held-out tasks, isolating the *pathway* as the independent variable and the *transfer* as the dependent variable.

---

## 3. 基准设计 / Benchmark

### 3.1 总览 / Overview

**中文**

S3Gym 由 7 款文本游戏组成，每款都以 JSON/文本形式给出状态与合法动作，并在每次动作后返回可核验的结果反馈。这一设计有两点用意：一是提供**足够高维**的动作空间以避免琐碎泛化；二是保证**结果无歧义**——agent 能否学到东西不受主观打分影响。

评估分两阶段：

- **Exploration**：agent 与游戏交互、生成经验，并**自己决定**如何加以利用（挑轨迹、写 memory、生成训练数据、执行更新）。
- **Evaluation**：在严格留出集上关闭"整合"通道，用同一 scorer 度量迁移效果。

### 3.2 七款游戏 / The seven games

| 游戏 Game | 判断挑战 Judgment challenge | 可核验反馈 Feedback |
|---|---|---|
| **Chess** | 长程规划 + 精确状态跟踪 | 引擎裁定合法性/胜负 |
| **Minesweeper** | 部分可观察 + 逻辑推断 | 触雷 / 完成 / 剩余安全格 |
| **Nullify** | 数值搜索 + 组合优化 | 数值目标是否达到 |
| **Tetris** | 实时空间规划 | 消除行数、堆叠高度 |
| **Snake** | 空间路径规划 + 自碰撞规避 | 得分 / 生存回合数 |
| **Plants-vs-Zombies** | 资源调度 + 长期防御 | 波次通过 / 阳光余量 |
| **Trust Evolution** | 社会博弈 + 策略学习 | 长期收益、对手模型 |

选择这七款游戏的原因：它们对应显著不同的"经验类型"——有的经验可以压缩成明确规则（如 Trust Evolution 的 tit-for-tat 类策略），有的经验高度依赖精确状态（如 Chess/Minesweeper），有的经验需要实时空间感知（Tetris/Snake），有的需要长期资源规划（PvZ）。这样跨游戏的对比能揭示"哪类经验适合哪条整合路径"。

### 3.3 两阶段协议 / Two-phase protocol

**Exploration 阶段**为每个 agent 提供固定的交互预算（例如按 episode 数或 token 数封顶）；agent 可以自由决定 rollout 策略、判断哪些经验有用、以及以何种形式存储（原始轨迹 / memory 记录 / 训练样本）。

**Evaluation 阶段**：

- 使用严格留出的初始状态或对手，保证不与 Exploration 重合；
- 无法再回写经验或更新参数；
- 只允许把 Exploration 中生成的"整合物"（history 上下文、memory、更新后权重）用于推理。

### 3.4 三条经验整合路径 / Three integration pathways

- **History In-Context Learning（History ICL）**：把探索阶段的原始轨迹直接作为上下文放进 prompt；agent 在 Evaluation 时以 ICL 形式复用。
- **Summary Memory**：agent 在 Exploration 阶段主动把经验总结为规则、启发式或技能，写入外部 memory；Evaluation 阶段检索这些条目。
- **Parameter Training**：把 Exploration 的经验组织成训练数据，通过 SFT 或 RL 更新模型参数；Evaluation 时用更新后的权重直接推理。

### 3.5 指标 / Metrics

主指标是各游戏的原生分数（胜率、得分、通过波次等），以及**相对无经验 baseline 的相对提升**。此外，本文还报告**判断准确率**（agent 事后判断"这一步是不是好走子"的准确率）作为诊断信号，用以分离"识别成功"与"整合成功"两件事。

**English (protocol summary)**

- Exploration is agent-driven: the agent chooses interaction budget usage, what to store, and in what form.
- Evaluation is closed-loop and uses **strictly held-out** starting states / opponents.
- The three pathways are compared under matched exploration budgets to isolate integration as the independent variable.

---

## 4. 实验 / Experiments

### 4.1 实验设置 / Setup

**中文**

模型：Opus 4.8、GPT-5.5、Gemini 3.1 Pro、DeepSeek V4 Pro、Qwen 3.7 Max、Seed 2.0 Pro。每个模型 × 每款游戏在三条整合路径下独立评估。Exploration 预算按 episode/回合封顶；Evaluation 留出集大小按游戏调整（Chess 使用一批独立开局，Minesweeper/Nullify 用新种子生成，Snake/Tetris/PvZ 用不同初始配置，Trust Evolution 用未见对手策略）。所有实验重复多次并汇报均值与方差以对抗高方差环境噪声。

### 4.2 主要结果 / Main results

**中文结论**

1. **没有一条路径主宰所有任务**。Summary Memory 在 Trust Evolution 上带来最大且最稳定的提升——因为该游戏的"胜利经验"极易被压缩为可复用规则（例如"在被对手连续背叛两次后转为长时间背叛"），memory 中的规则条目可以直接指导 Evaluation 阶段的决策。相反，在需要精确当前状态的 Chess/Minesweeper 中，Summary Memory 的收益有限甚至为负——把状态相关的"经验"变成通用规则，等于扔掉了做正确判断所需的信息。
2. **History ICL 在"经验难以压缩"的游戏中反而更强**。对 Chess、Minesweeper 这类状态高度个体化、每一步都依赖当前 board 的游戏，直接把原始轨迹放进上下文让模型现场类比，通常比规则化 memory 效果更好；代价是上下文成本随经验增长而线性膨胀。
3. **Parameter Training 呈"高方差 + 局部胜利"模式**。Snake、Nullify 上参数更新能带来明显、可保留的能力提升；但在 PvZ、Tetris 上却常常出现负迁移——反过来把 Exploration 中偶发的坏习惯"焊"进权重。
4. **能识别 ≠ 能改进**。多数模型在"事后判断某步是否成功"的判断准确率上都不差（往往 > 70%），但这一准确率与最终 Evaluation 分提升之间的相关远远弱于预期——说明瓶颈从"识别正确性"转移到了"如何组织并复用这些正确判断"。

### 4.3 观察到的关键失败模式 / Observed failure modes

**中文**

- **Memory 过度抽象**：agent 把"这局赢是因为对方失误"抽成"总是尝试激进开局"，把不可复现的运气写成硬规则。
- **History 过度模仿**：把上下文中的某条成功轨迹作为模板机械照搬，忽视当前状态差异。
- **参数训练的分布偏移**：Exploration 阶段的经验分布并不匹配 Evaluation 分布，SFT 后模型倾向复现探索期的"错误但被暂时奖励"的动作。
- **自评一致但归因错误**：agent 正确判断"我赢了/输了"，但归因到与输赢无关的动作（如"因为我下了 h3 所以赢了"），把噪声当信号。

### 4.4 判断准确率 vs. 整合收益 / Judgment vs. Integration

**中文**

论文将"判断准确率"和"评估阶段增益"分列。多模型跨游戏统计显示：**判断准确率高**在部分游戏（Trust Evolution、Nullify）与整合收益正相关；在另一些游戏（Chess、PvZ）几乎不相关。这与主结论一致——如果经验能压缩，好判断能被 memory 复用；如果经验只在具体状态下有效，好判断本身并不能变成通用能力。

**English (results at a glance)**

- No integration pathway dominates across all seven games; the best pathway is a function of *how compressible* the useful experience is.
- **Summary Memory** wins when experience compresses into rules (e.g., Trust Evolution).
- **History ICL** wins when precise state information is essential (e.g., Chess, Minesweeper).
- **Parameter Training** yields the largest wins on some games (e.g., Snake, Nullify) but is unstable and can induce negative transfer elsewhere.
- Accurate post-hoc judgment of "which action was good" is common (>70% in many models), but its correlation with downstream Evaluation-phase gain is game-dependent, confirming that identification is not sufficient for improvement.

---

## 5. 分析与讨论 / Analysis and Discussion

**中文**

**为什么"整合路径"是关键？** 每条整合路径都在做不同的信息压缩：History ICL 保留最多细节但复用成本随经验规模线性膨胀；Summary Memory 强制压缩，把细节替换为规则；Parameter Training 走得最远，把经验融入权重、抹掉个别样本的可追踪性。这三种压缩率下"哪些信息被保留、哪些被丢弃"直接决定了经验的可迁移性。

**"可压缩性"是判断路径匹配的关键**：如果一款游戏的经验能被简洁规则近似（"永远先扫角"、"看到 X 就走 Y"），Summary Memory 是天然拟合；如果经验高度个体化，Summary 会丢掉关键状态；如果经验分布稳定且样本可靠，参数训练可以把这些知识"内化"，但一旦分布不稳定，参数训练反而放大噪声。

**"信号 vs. 噪声"的悖论**：跨模型的判断准确率并不低，但 Evaluation 增益差异巨大——问题不在识别，而在于**如何把识别转成不受具体样本束缚的能力**。这与人类学习类似：能识别"哪次做得好"只是必要条件，把它变成可复用技能才是充要条件。

**限制搜索 vs. 拓展搜索**：Exploration 阶段的多样性也是关键变量。经验越多样，可压缩性越高；经验过度集中，则任何路径都难以摆脱局部最优。本文观察到，agent 常常在 Exploration 中过早收敛到某类策略，导致后续无论走哪条整合路径，Evaluation 都被封在同一"错误盆地"。

**English (analytical points)**

- Each integration pathway imposes a different compression on experience; the right choice depends on how compressible the *useful* part of the experience is.
- Diversity of Exploration matters as much as pathway choice — premature convergence during Exploration bounds any Evaluation gain.
- Being right about "which move was good" is necessary but not sufficient; the harder problem is turning correct judgments into representations that generalize beyond their originating states.

---

## 6. 局限、伦理与结论 / Limitations, Ethics, Conclusion

### 6.1 局限 / Limitations

**中文**

- 七款游戏虽多样但不能穷尽所有 agent 场景；例如缺少多智能体协作、真实工具使用等设定。
- Parameter Training 结果对超参与数据规模高度敏感，本文尽量给出重复实验，但仍可能低估某些配置下的上限。
- Judgment 准确率是描述性诊断而非因果度量。
- 部分游戏的 Evaluation 留出集难以完全避免与 Exploration 的语义重叠（例如 Chess 中相似开局），本文在协议中做了显式约束但无法完全根除。

### 6.2 伦理 / Ethics

**中文**

- 游戏环境为公开或自研文本环境，不涉及个人数据。
- 强调基准所度量的是**外部经验整合**而非"参数更新可被启发式取代"——不主张 memory/history 能取代真实的模型训练。
- 生成的 memory / SFT 数据可能反映模型固有偏好，复用时需评估相应风险。

### 6.3 结论 / Conclusion

**中文**

S3Gym 把"agent 自我改进"从模糊愿景细化为可度量的三步循环。跨 6 个前沿模型与 7 款游戏，实验一致地显示：**Self-Improvement 不是自动发生的，也不是均匀分布的**——它取决于经验能否被压缩、被压缩成什么形式，以及所选整合路径与游戏结构是否匹配。一个能"识别成功"的 agent 并不自动等于一个"能持续变强"的 agent；只有把这种识别有原则地整合到 history、memory 或权重中，识别才会成为能力。**S3Gym 的价值在于把这一整合步骤显式化，使不同模型、不同路径之间的比较真正可比**。

**English (concluding position)**

- S3Gym gives a shared, verifiable substrate to compare *how* agents turn interaction into capability, not just *whether* they can play a game.
- The empirical picture is nuanced: which pathway wins depends on the game's compressibility, and identification without principled integration does not translate into transferable gains.
- Meaningful progress on self-improving agents will likely require **matching integration pathway to task structure**, rather than searching for one universally superior mechanism.

---

## 7. 附录要点摘要 / Appendix Highlights (selected)

- **游戏接口规约**：每款游戏对外暴露 `reset(seed)`、`step(action)`、`legal_actions(state)`、`render(state)` 与结构化 `reward`，保证跨模型可比。
- **Memory schema**：Summary Memory 建议格式为 `{condition, action, expected_outcome, evidence}`；便于评审者审计规则是否被 Exploration 证据支持。
- **Parameter Training 细节**：SFT 主要采用轨迹级正样本蒸馏；RL 采用对局级奖励回传；两者均按游戏单独调节学习率与训练步数。
- **判断准确率评测**：从 Exploration 中随机采样 (state, action, outcome) 三元组，让 agent 事后回答"这一步是好走子吗"，对照实际结果计算准确率。
- **失败案例集**：附录列出跨模型的典型失败——包括 memory 中把"运气"写成"规则"的例子、SFT 后 Snake agent 撞死概率反而增加的曲线，以及 PvZ 中 Parameter Training 引发的资源使用偏差。

**English summary of appendices**

- Standardized game interface and memory schema make cross-model comparisons faithful.
- Judgment accuracy is measured on a held-out (state, action, outcome) probe, decoupled from Exploration reward shaping.
- Documented failure cases motivate the paper's central point: identification is easy, principled integration is hard.

---

## 引用 / Citation

```
@misc{s3gym2026,
  title  = {S3Gym: Can LLMs Turn Self-Testing and Self-Judging into Self-Improvement?},
  author = {Anonymous},
  year   = {2026},
  eprint = {2608.31100},
  archivePrefix = {arXiv},
  primaryClass  = {cs.AI},
  url    = {https://arxiv.org/abs/2608.31100}
}
```
