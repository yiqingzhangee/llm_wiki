# Aspire：模型能否从模糊目标出发实现自我演化？

# Aspire: Can Models Self-Evolve from Vague Goals?

- **arXiv 编号 / arXiv ID**：[2608.31111](https://arxiv.org/abs/2608.31111)
- **提交时间 / Submitted**：2026 年 8 月 31 日
- **作者机构 / Affiliations**：ByteDance Seed；Singapore University of Technology and Design；Georgia Institute of Technology；M-A-P；TokenWave.AI
- **项目主页 / Project Page**：https://self-developing-agents.github.io/
- **说明**：本文为双语对照阅读辅助版本，采用「中文在前、英文在后」排版；英文部分为原文关键论点的结构化重述与摘引，用于对照阅读；完整原文请以 arXiv 上的官方 PDF 为准。

---

## 摘要 / Abstract

**中文翻译**

现实中的自我改进往往从**模糊的目标**开始——一句简短的自然语言意图，例如"更擅长科学推理"，既没有精确的评价规约，也没有配套数据集或明确成功标准。我们提出 **ASPIRE**，一个用于评估语言模型是否能在这种模糊目标下自我演化的基准。给定一句一句话目标，agent 需要**自己**把它 operationalize（操作化）——决定要提升什么、选择哪些数据、以何种方式更新，并构建能可靠反映进展的训练与验证信号。之后 agent 会在一个**隐藏评估集**上被度量，我们特意让开发过程无法直接优化这个隐藏集。ASPIRE 覆盖 6 类目标——科学与学术推理、人文与社会科学知识、健康与医学推理、数学推理、逻辑/可靠性与指令遵循、学术与科学写作——由领域专家撰写 520 道隐藏题目，每道都附有可验证或专家可评的成功标准。我们评测多种前沿系统，从纯提示驱动的 baseline 到能编辑自身 harness 的完整开发 agent，并对比其结果与人工工程系统 Qwen-Agent。结果显示：**模糊目标会把大量搜索精力转向"目标解释与操作化"**，agent 常能完成训练与 harness-editing 的循环，**但可保留、能迁移到隐藏评估的权重级增益仍然稀疏且不稳定**；最强的 successor harness 仍显著低于人工工程系统；**长时间的持续搜索与训练甚至会抹掉早期已经拿到的增益**。这一诊断把"从模糊目标出发做真正的自我改进"确立为一个开放挑战。

**English Original (excerpt)**

> Real-world self-improvement often starts from a **vague goal** — a short natural-language intent such as "get better at scientific reasoning" — without a precise evaluation contract, curated data, or explicit success criteria. We introduce **ASPIRE**, a benchmark for evaluating whether language models can self-evolve under such vague goals. Given a one-sentence intent, the agent must operationalize it on its own: decide what to improve, which data to select, how to update, and how to construct training and validation signals that reliably reflect progress. Performance is then measured on a **hidden evaluation set** that the development process cannot directly optimize. ASPIRE covers six goal categories — Scientific & Academic Reasoning, Humanities & Social Sciences, Health & Medical Reasoning, Mathematical Reasoning, Logic/Reliability & Instruction Following, and Academic & Scientific Writing — with 520 expert-authored held-out items, each accompanied by verifiable or expert-rubric success criteria. Across prompt-only baselines, memory- or harness-editing agents, and fully-developing agents evaluated against the Qwen-Agent reference, we find that vague goals divert most search effort into goal interpretation and operationalization; agents complete training and harness-editing loops but retained, transferable weight-level gains on the hidden set are **sparse and unstable**; the strongest successor harness still falls short of the engineered reference; and continued search and training can erase earlier gains. Together, this positions **self-evolution from vague goals** as an open challenge.

---

## 1. 引言 / Introduction

**中文**

真实世界的自我改进不是从"整洁的目标 + 干净的数据集 + 明确的评价指标"起步的，而通常从一句模糊的自然语言意图开始，例如"更擅长科学推理"、"在多轮医学咨询中更可靠"、"少犯逻辑错误"。要把这样一句意图变成能实际驱动模型变强的过程，agent 至少要做四件事：

1. **解释目标**：判断"更擅长科学推理"意味着什么——是数学计算、图表理解、方法学讨论、跨学科综合，还是其它？
2. **选择材料**：确定要用哪些数据、任务或试题来训练与验证；这些材料是否覆盖所解释的目标？是否含污染？规模与难度是否合适？
3. **决定更新方式**：选择路径（prompt/例子调优、memory/harness 编辑、SFT、RL 等）与训练配置。
4. **构造进展信号**：定义训练与验证信号，使得"在信号上更好"能可靠反映"在真实目标上更好"。

**难点在于**：这四步互相纠缠——目标解释错了，选出的数据就错；数据错了，训练/验证信号就错；训练/验证信号错了，即便在信号上进步，也不一定在隐藏评估上进步。而现实中我们**唯一真正在意**的度量恰恰是"未见任务上的能力提升"，而不是 agent 自己造出来的验证集。

**ASPIRE 的立场**：许多既有 benchmark 已经给定了"精确的成功标准"，模型只需在这个契约内做题；这掩盖了"如何把一句人类意图变成一条可优化目标"的关键难题。ASPIRE 让**目标本身**成为受试内容——它明确对 agent 隐藏最终评价集，agent 必须自主完成上面四步；最终由一个专家构造、事先冻结的评估集给出真实进展信号。

**核心问题**：当代前沿 LLM 能否在"模糊目标 + 隐藏评估"下真正实现自我演化？——不是仅在自造验证集上刷分，而是把提升迁移到未见测试上。

**English (key claims)**

- Real self-improvement begins with vague intent, not a precise evaluation contract.
- The agent must self-operationalize a one-sentence goal into (i) an interpretation, (ii) a data plan, (iii) an update mechanism, and (iv) a progress signal.
- ASPIRE keeps the evaluation set hidden so the development process cannot optimize it directly, isolating whether operationalization actually yields transferable capability.

---

## 2. 相关工作 / Related Work

**中文**

**自我改进 LLM**：既有工作大多聚焦 Self-Refine、Reflexion、RLAIF、Self-Rewarding、Constitutional AI 等，通常在**目标已被明确定义**的前提下研究反馈驱动改进；ASPIRE 把这些工作向"目标本身也需要 agent 自行定义"这一步前推。

**Agent 演化 / Harness 编辑**：Voyager、AutoAgent、Self-Harness、HarnessFix、DemoEvolve 等允许 agent 编辑自身 memory、prompts 或 harness；这些方法对应 ASPIRE 中的"更新机制"部分。

**训练数据自造**：STaR、Self-Instruct、Evol-Instruct、Self-Play 等系统性地用模型自身生成训练数据。ASPIRE 允许 agent 走这条路径，但把评价放到**agent 无法接触的隐藏集**上。

**隐藏评价 / 反过拟合**：ML/信息检索社区长期以隐藏 leaderboard 抵御 overfitting，Kaggle、SemEval 都用类似做法。ASPIRE 借鉴这一思路，把 held-out 作为"能力是否真正提升"的唯一裁判。

**与 HarnessDev、S3Gym 的关系**：三者是同一系列工作。HarnessDev 关注"harness 是否被造好"；S3Gym 关注"经验能否被判断并复用"；**ASPIRE 关注"目标能否被自主定义并驱动能力提升"**。

**English (positioning)**

- Prior self-improvement work assumes an explicit success contract; ASPIRE removes that assumption.
- ASPIRE complements HarnessDev (execution scaffold) and S3Gym (experience integration) by isolating **goal operationalization** as the primary independent variable.

---

## 3. 基准设计 / Benchmark

### 3.1 目标形式与六大类别 / Goal format and six categories

**中文**

每个任务由一句一句话目标（vague intent）触发，例如"更擅长科学推理"。ASPIRE 覆盖六类：

1. **Scientific & Academic Reasoning**（科学与学术推理）
2. **Humanities & Social Sciences**（人文与社会科学知识）
3. **Health & Medical Reasoning**（健康与医学推理）
4. **Mathematical Reasoning**（数学推理）
5. **Logic, Reliability & Instruction Following**（逻辑、可靠性与指令遵循）
6. **Academic & Scientific Writing**（学术与科学写作）

隐藏评估集共 **520 道**专家编写题目，每类均衡覆盖；每道题附有：可核验答案（数学、事实、代码类）或专家 rubric（写作、开放式推理类）。**agent 在开发过程中看不到这 520 道题**——只知道目标类别与一句话意图。

### 3.2 开发协议 / Development protocol

**中文**

- agent 收到 (一句话目标, 目标类别, 允许的更新路径, 计算预算)。
- 允许 agent 自主：解释目标；生成/选择训练材料；选择更新路径（prompt、memory、harness、SFT、RL）；构造训练与验证信号；决定何时停止。
- 禁止：直接接触隐藏评估集；从隐藏集反向工程题目；使用与隐藏集重叠的公开数据（作者对常用来源做了污染检查并说明）。
- 结束时 agent 提交一个 successor：可以是新的 prompt / memory / harness / 权重，或它们的组合。

### 3.3 评估协议 / Evaluation protocol

**中文**

- **隔离评审**：由一个独立评审在冻结环境中运行 successor，对每个隐藏题打分。
- **主指标**：隐藏集准确率或 rubric 得分（按题型加权）。
- **相对基线**：报告相对未演化 baseline 的提升 Δ、以及相对 **Qwen-Agent**（人工工程参考系统）的差距。
- **稳健性诊断**：报告 successor 在多次 seed / 顺序扰动下的方差、跨类别一致性、是否出现负迁移到未提及类别。

### 3.4 允许的更新路径 / Update pathways

**中文**

- **Prompt / few-shot 调整**：最轻更新。
- **Memory 编辑**：外部知识/规则/技能库。
- **Harness 编辑**：修改 agent 循环、工具、上下文管理、验证器等。
- **权重级更新**：SFT、DPO 或 RL；数据由 agent 自生成或自选。
- **组合路径**：允许上述任意组合，但需在提交时声明。

### 3.5 度量框架 / Measurement framework

**中文**

- **能力增益**：$\Delta = \text{Score}(H_{\text{after}}) - \text{Score}(H_{\text{before}})$，在隐藏集上评估。
- **保留率**：多次评估的方差、跨类别是否出现"补 A 掉 B"。
- **搜索开销**：agent 消耗的训练/评测 token 数、迭代轮数。
- **与人工参考差距**：$\text{Score}(H_{\text{ref}}) - \text{Score}(H_{\text{after}})$。

**English (protocol summary)**

- Vague intent + category, hidden 520-item eval set, no access to the eval during development.
- The agent picks the update pathway (prompt / memory / harness / weights) and must self-construct training and validation signals.
- Measurement combines hidden-set gain, retention across categories, search cost, and gap to Qwen-Agent as a human-engineered reference.

---

## 4. 实验 / Experiments

### 4.1 实验设置 / Setup

**中文**

评测多类系统：

- **Prompt-only baselines**：仅调 prompt / few-shot，不改 memory/harness/权重。
- **Memory / harness-editing agents**：允许编辑外部 memory 或 harness 代码。
- **Fully-developing agents**：允许自选任意路径，含 SFT/RL。
- **Human-engineered reference**：**Qwen-Agent**，一个成熟人工工程系统，作为"上限参照"。

底层 LLM 覆盖多家前沿模型；每类目标下 agent 独立开发 successor 并提交。

### 4.2 主要结果 / Main results

**中文关键发现**

1. **搜索精力被"目标解释"吞掉大半**。在完整开发日志中，绝大多数 token 与迭代花在"目标是什么/该覆盖什么/该造什么题"上——而不是在真正的能力更新上。目标越模糊，操作化占比越高。
2. **可保留、能迁移到隐藏集的权重级增益稀疏且不稳定**。多个前沿 agent 能在自建验证集上给出正 Δ，但拿到隐藏 520 题上后，Δ 常缩小甚至反转；跨 seed 复现波动较大。
3. **最强 successor harness 仍显著低于 Qwen-Agent**。即使允许自由选择路径，agent 造出的最佳系统在六类任务合计得分上仍落后于人工工程参考。
4. **狭窄自评导致的过拟合**。agent 往往训练在与目标不完全匹配的数据上，并信任自己写的窄小验证信号；结果在自评上"看起来在变强"，隐藏集却停滞。
5. **持续搜索会抹掉早期增益**。多轮迭代经常出现"先升后降"：早期改到的正 Δ 在后续更新中被覆盖，最终提交版本反不如中途版本。这与 HarnessDev 中"演化并非单调"的观察一致。
6. **类别差异明显**。逻辑/指令遵循与数学推理更易得到可测量的稳定收益；学术写作与人文社科更难——rubric 与自评之间的差距更大。

### 4.3 观察到的失败模式 / Observed failure modes

**中文**

- **目标窄化**："科学推理"被 operationalize 成"高中物理题"→ 覆盖不足。
- **数据错配**：造出的训练题不覆盖 rubric 关心的推理链或写作维度。
- **验证漂移**：自评越迭代越"局部合理"，与真实目标越走越远。
- **权重更新噪声**：SFT/RL 步骤的分布偏移带来跨类别负迁移。
- **过晚终止**：agent 在收益已经开始回落后仍继续训练/编辑，浪费预算并覆盖早期正 Δ。

### 4.4 与 HarnessDev/S3Gym 的证据一致性 / Consistency with HarnessDev and S3Gym

**中文**

三项研究给出彼此印证的画面：

- **HarnessDev**：即便有明确任务规约，harness 演化的可靠增益仍稀疏，且强烈依赖执行 harness 的具体模型。
- **S3Gym**：即使 agent 能正确识别"哪次做得好"，也不必然把识别转成可迁移能力。
- **ASPIRE**：当**目标本身**也需要 agent 自己定义时，问题进一步放大——大部分搜索被目标操作化吃掉，隐藏集上的可保留权重级增益仍是开放挑战。

**English (results at a glance)**

- Vague goals shift most compute into interpretation and operationalization rather than capability updates.
- Retained, transferable weight-level gains on the hidden set are **sparse and unstable**; self-constructed validation signals inflate perceived progress.
- The strongest successor harness still trails the human-engineered Qwen-Agent reference.
- Continued training/editing can undo earlier wins; final submissions are often worse than intermediate versions.

---

## 5. 分析与讨论 / Analysis and Discussion

**中文**

**为什么模糊目标这么难？** 传统 RL/SL 设置里，"目标"由 reward function 或标注数据显式给定；agent 只需在这个契约内搜索。ASPIRE 强迫 agent **自己写这份契约**——它必须先猜清楚"隐藏评审到底会考什么"，再选材料、再决定更新、再造验证信号。契约的每一个环节都是不确定性来源，误差会顺着链条累加。

**"识别 vs. 生成"再度出现**：一个 agent 常常能"识别"某类题是不是真的关乎"科学推理"（判断准确率不低），但它**生成**出来的训练题、rubric、验证集仍不能覆盖专家隐藏集所关心的能力维度。这与 S3Gym 中"识别 ≠ 改进"的结论如出一辙——正确识别不足以支撑可迁移生成。

**搜索的"局部盆地"**：agent 会围绕自己解释出的窄目标训练/编辑，短期在自评上单调改善，但整体能力在隐藏集上停滞或退化；这就是"自评越训越好、真评越走越远"。降低这种盆地效应需要外部锚——ASPIRE 用**隐藏评估**充当外部锚，明确显示"这条路径是否真的通向真实目标"。

**Qwen-Agent 差距的含义**：人工工程参考给出的是"知情工程 + 已完成目标操作化"的性能上限。agent 与之的差距，主要来自"未能像人一样正确定义目标 + 未能把每一环工程化到位"，而不仅仅是"参数训练效果不佳"。**目标操作化本身就是一种能力**，而目前 agent 在这项能力上仍显著弱于成熟人工系统。

**如何前进？** 论文没有给出"银弹"，但提示几点方向：

- **更好的目标分解**：把一句话目标显式拆解为多子目标、多子评审，减少"一份自评走天下"的过拟合风险；
- **外部锚点**：即便真正的隐藏集不可见，也应引入独立的、开发过程无法优化的第三方信号；
- **早停 & 保留最佳中间版本**：而不是无条件相信"最新版本最好"；
- **谨慎的权重更新**：在分布未稳定时限制 SFT/RL 步幅、保留可回滚的多个 checkpoint。

**English (analytical points)**

- Under vague goals, the agent must self-author the evaluation contract; errors in operationalization compound down the chain.
- Correct **identification** of what "counts" as scientific reasoning does not entail correct **generation** of training material that transfers to expert-authored held-out items.
- The Qwen-Agent gap is largely a gap in goal operationalization and full-stack engineering, not just parameter update efficacy.
- External anchors (hidden or third-party signals) and cautious update policies (early stopping, checkpoint retention) look necessary for future systems.

---

## 6. 局限、伦理与结论 / Limitations, Ethics, Conclusion

### 6.1 局限 / Limitations

**中文**

- 6 类目标虽有代表性但不能穷尽；例如缺少多模态、机器人交互、企业业务流程类目标。
- 隐藏评估集 520 题的规模足以给出方向性结论，但对特定 successor 的置信区间仍偏宽。
- 部分类别（写作/人文）的 rubric 主观性较强，即便由多人专家复核也可能存在评审噪声。
- 对"污染检查"依赖公开数据源标签，对私有/自造数据难以完全排除污染风险。

### 6.2 伦理 / Ethics

**中文**

- 所有专家题目由领域专家原创或改写，避免侵权。
- 强调 ASPIRE 度量的是"从模糊目标出发的自我演化能力"，**不主张 memory/harness/自造数据可以在原则上替代真实模型训练**。
- 自动化目标操作化在部署阶段可能带来风险：若 agent 错误解释目标并自行更新权重，可能引入不可预期的偏差；论文建议对权重级更新保持保守，并保留可回滚的中间版本。

### 6.3 结论 / Conclusion

**中文**

ASPIRE 把一个长期被忽视但极其现实的问题变得可度量：**当目标只是一句模糊的人话，agent 能真的把它变成能力增长吗？** 结论谨慎但清晰——前沿 agent 可以完成"训练与 harness 编辑"的循环，但可保留、能迁移到隐藏评估的权重级增益仍稀疏且不稳定；最强 successor 仍落后于成熟人工系统；持续搜索甚至会抹掉早期增益。这既说明"自我演化"不是当前架构直接可用的能力，也把"目标操作化"与"外部锚点"清楚地放到未来研究的中心议程上。

**English (concluding position)**

- ASPIRE positions **self-evolution from vague goals** as an open, measurable challenge.
- Frontier agents can execute the mechanics of update loops but do not yet reliably convert vague intent into retained, transferable capability.
- Future progress plausibly requires better goal decomposition, external anchoring beyond self-constructed validation, and cautious update policies with checkpoint retention.

---

## 7. 附录要点摘要 / Appendix Highlights (selected)

- **520 题构成**：六类目标下按难度与题型加权分布；每题记录出题人、来源、答案/rubric、评分脚本。
- **污染检查**：对主要公开数据源做题面与近似匹配检查，剔除疑似泄露样本。
- **评审细节**：客观题使用自动核对；rubric 题使用多位专家独立打分后取一致性最高的合成分。
- **agent 提交格式**：需声明所走路径（prompt / memory / harness / weights / 组合），并附训练数据摘要与自评报告，以便事后审计。
- **失败案例集**：涵盖"目标窄化"、"自评漂移"、"迟到的负迁移"、"过晚终止"等典型模式，与 HarnessDev、S3Gym 中的失败模式相互印证。

**English summary of appendices**

- 520 held-out items span six categories, with per-item provenance, rubric, and grader script recorded.
- Contamination checks against major public sources; ambiguous overlaps are dropped.
- Submissions must declare the update pathway and include a self-assessment report to enable post-hoc audit.
- Documented failure modes align with those seen in HarnessDev and S3Gym, reinforcing the paper's central message about operationalization as the current bottleneck.

---

## 引用 / Citation

```
@misc{aspire2026,
  title  = {Aspire: Can Models Self-Evolve from Vague Goals?},
  author = {Anonymous},
  year   = {2026},
  eprint = {2608.31111},
  archivePrefix = {arXiv},
  primaryClass  = {cs.AI},
  url    = {https://arxiv.org/abs/2608.31111}
}
```
