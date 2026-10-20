# Jev决策模型与Agent架构

Jev 是 TypeSafe AI 发布的 System One 决策模型：输入应用状态和预先定义的问题，返回类型明确的选择、评分或概率，而不是生成给人阅读的长文本。它适合放在 Agent 的模型路由、工具风险门控、结果验证和高频评测等边界位置，让代码保留工作流控制权，让强模型处理开放式推理和生成。

> 本页以 ByteTech 内部文章《一文讲清 Jev》《Jev 实测：这个只做判断、不聊天的 AI 有什么用？》为优先来源，并用 TypeSafe、LangChain、TrueForge 等公开资料交叉核对。所有性能、成本和准确率数字均按各自测试口径保留，不外推为普遍结论。

## 一句话模型

```text
开放式 LLM：理解上下文 → 生成计划/文本/代码
Jev：理解状态 → 在固定问题和选项上给出结构化判断
代码：执行确定性逻辑 → 调用工具、改变状态、记录结果
人：处理低置信度、高风险和写入前审批
```

Jev 不是“更小的聊天模型”这一简单替代品。它放弃字符串生成，换取结构化输出、并行问题、低延迟和概率化置信度；因此它的价值取决于应用能否把问题定义成边界清晰的决策。

## 为什么从 Agent 主循环中拆出决策

一个 Agent 循环通常同时承担四类工作：

1. 选择模型、工具或下一条路径；
2. 生成计划、参数、文本或代码；
3. 判断工具调用是否安全、是否需要审批；
4. 检查工具结果是否满足要求、是否可以结束。

很多第 1、3、4 类问题并不需要生成自然语言，只需要从有限选项中选择、给状态打分或回答一个是/否问题。若全部交给生成式 LLM，会产生额外的上下文读取、逐 token 生成、结构化解析和重复调用成本，也会让局部判断受到生成随机性的影响。

Jev 的架构位置是把这些判断从主模型的生成循环中抽离，形成一个可被代码消费的语义判断层：

```text
任务/工具/历史结果
        ↓
  结构化状态 state
        ↓
Jev：Choice / Score / Noul + probability
        ↓
代码分支、模型路由、审批、重试或回退
        ↓
强模型生成 / 工具执行 / 人工确认
```

## 输出类型与输入边界

当前公开材料中最稳定的三类问题是：

| 类型 | 作用 | 例子 |
|---|---|---|
| `Choice` | 从应用提供的有限选项中选择一个 | 选择匹配的 GitHub issue、下一步工具、路由模型 |
| `Score` | 在有序等级上评分 | 紧急程度、风险等级、用户挫败程度 |
| `Noul` | 对是/否命题返回概率 | 是否泄露 PII、命令是否危险、结果是否满足要求 |

Jev 只能在调用方提供的候选空间内选择。它可以选择 `none`，但不能凭空创建新候选；因此大规模候选集通常需要先由代码做检索/排序，再由 Jev 在有限候选中判断。TypeSafe 官方资料还提到单次状态和问题有上下文长度与选项数限制，具体上限应以当前 API 文档为准。

## 在 Agent 中的四个位置

### 1. 模型路由

根据任务难度、数据敏感性、延迟预算或输出要求，选择便宜模型、强模型、视觉模型或人工流程。

```text
state: 用户任务 + 历史上下文 + 预算
question: 该任务应路由到哪种模型？
output: cheap | frontier | human_review
```

Jev 只负责判断，实际调用和降级策略仍由代码实现。

### 2. 工具风险门控

在 shell、数据库写入、外部消息发送、支付或权限变更前，判断是否允许执行、是否需要二次确认或是否应拒绝。

```text
state: 用户请求 + 工具名 + 参数 + 当前权限
question: 该调用是否需要审批？
output: allow | ask_user | deny
```

高风险动作不能因为 Jev 置信度高就自动放行。权限校验、策略规则、参数白名单和人工审批仍应是独立的硬门禁。

### 3. 结果验证

对工具结果或 Agent 轨迹做结构化检查，例如是否完成任务、是否满足字段要求、是否包含敏感信息、是否出现提示注入或明显偏离。

Jev 适合作为快速筛选器；低置信度样本应升级给强模型或人工，而不是把概率分数当作事实证明。

### 4. 高频评测

将一条 Agent trace、消息或文档作为状态，对多个质量标准并行打分。LangChain 的公开实验把它作为第三种 Agent evaluator：介于确定性代码评测和开放式 LLM-as-a-judge 之间。

适合评测：

- 是否泄露 PII；
- 是否调用了正确工具；
- 用户意图类别；
- 是否需要人工升级；
- 结果是否满足一组预先定义的标准。

若需要长篇解释、开放式论证或复杂评分理由，传统 LLM judge 仍更合适。

## Jev + TrueForge：一个可复用的 Harness 分工

本地检索到的分享以客户请求匹配 GitHub issue 为例，形成如下链路：

```text
Notion 客户请求
    ↓
代码先召回/短列候选 issue
    ↓
Jev 在候选中选择匹配项 + 输出置信度
    ├─ 高置信度：进入状态映射
    └─ 低置信度：交给强模型复核
             ↓
强模型生成新 issue 或给出复核意见
    ↓
TrueForge 审批节点
    ↓
人工批准后才写入 Notion / GitHub
```

TrueForge 负责的是运行时和 Harness，而不是替代 Jev：

- 延迟加载工具定义，减少每轮上下文；
- 大工具结果写入沙箱文件，只在对话中保留预览；
- Code Mode 用脚本处理批量数据，避免中间结果全部进入上下文；
- Subagent 将局部任务隔离到独立上下文；
- 将密钥留在 Harness 外部，避免脚本直接获得凭证；
- 对 Notion、GitHub 等写操作设置人工审批；
- 保存会话状态、工具调用、模型输出和运行轨迹。

这套分工体现的是“便宜模型默认处理边界判断，强模型处理开放任务，代码处理精确执行，人处理高风险例外”，而不是简单地把所有调用换成 Jev。

## ByteTech 内部文章与实测

### 两篇文章的共同判断

内部文章将 Jev 概括为“普通代码读不懂语义时的智能分支”：

- 普通 `if` 负责确定性条件；
- Jev 负责有限选项上的模糊语义判断；
- 通用模型负责开放式生成与复杂推理；
- 人工处理低置信度、高风险和规则缺失的情况。

《一文讲清 Jev》进一步澄清“不产生幻觉”的边界：Jev 不会输出 Schema 之外的选项，但完全可能高置信地选择错误的合法选项。因此类型安全不能替代测试、权限、沙箱和人工审批。

### Jev 与 Seed 2.0 Lite 对照

《Jev 实测》使用真实接口覆盖分类路由、内容审核、紧急度、意图识别、检索过滤、销售线索、模糊判断和补测 Case，并让 Seed 2.0 Lite 对相同任务只输出结构化结果。

文章报告：

- 大多数有明确标准的语义判断方向一致；
- Jev 在错误大数加法上出现明显误判；
- 两者在未定义续订规则的日期问题上出现分歧；
- 重复调用的主结论较稳定，但概率会随措辞变化；
- 无意义输入有时仍会得到高置信答案；
- Jev 不能读图，图片 URL 只会被当作文本。

这组结果支持“有界语义判断可交给决策模型”，同时也说明精确计算、规则适用和输入有效性检查必须留在代码层。

### 内部实测的上线启示

1. 先写清候选集合、选项定义和拒绝条件；
2. 用业务真实样本验证，而不是只跑容易区分的演示题；
3. 单独测试计算、日期、空输入、矛盾输入和提示词扰动；
4. 概率阈值要在自有数据上校准，不直接复制官方或文章中的数值；
5. 记录模型版本、State、Questions、候选集合和阈值，保证可回放；
6. 高风险动作继续使用确定性权限和人工审批。

## 外部 Harness 案例

TrueForge 分享中的客户请求匹配 GitHub issue 案例，将每条请求先短列为 10 个候选，再由 Jev 选择 issue 或 `none`，低置信度结果交给强模型复核，最终由人批准写入。分享报告 Jev 版本与无 Jev 版本都正确处理了 40 条中的 37 条，但 Jev 版本送审更多不确定结果，复核捕获了更多错误。

该结果来自单一测试集。可迁移的不是“37/40”数字，而是按错误代价设置阈值、让 `none` 也可升级、并同时测量准确率、覆盖率、升级率和漏检率的方法。

## 与生成式 LLM 的关系

| 工作 | 更适合的组件 | 原因 |
|---|---|---|
| 复杂问题分解 | 强 LLM | 需要开放式推理和生成 |
| 生成计划、代码、解释 | 强 LLM | 输出不是固定选项 |
| 从候选中选择 | Jev 或其他决策模型 | 选项空间有界，结果可结构化 |
| 精确计算、状态机、权限校验 | 代码 | 可验证、可审计、可复现 |
| 工具执行 | Harness/代码 | 需要凭证、重试、超时和副作用控制 |
| 高风险写操作 | 人 + 代码门禁 | 不能只依赖模型概率 |
| 长篇评测解释 | LLM judge | 需要理由和开放式判断 |
| 高频标签、路由、筛选 | Jev | 低延迟、可批量、概率可用于升级 |

因此，Jev 的合理定位不是“Agent 的大脑”，而是“Agent 运行时中的语义分支组件”。

## 评测与上线门禁

在内部业务接入前，至少应建立以下验证集：

- 固定选项内的常规样本、边界样本和对抗样本；
- `none`/拒绝/未知样本；
- 候选排序错误、候选缺失和候选重复；
- 同一状态的重复调用一致性；
- 置信度校准曲线；
- 高风险动作的误放行和误拦截；
- 超时、服务不可用和回退到强模型的行为；
- 版本升级前后的决策漂移。

建议同时看四组指标：

```text
决策质量：accuracy / precision / recall / calibration
系统效率：p50/p95 latency / token cost / calls per task
升级行为：coverage / human-review rate / fallback rate
治理安全：unauthorized writes / audit completeness / secret exposure
```

TypeSafe、LangChain 和社区分享中的速度、成本、准确率都是特定工作流或早期实验结果。落地时应复用自己的真实轨迹做离线回放和在线灰度，不应直接复制宣传数字。

## 与已有 Agent 本体模型的关系

从 Agent 本体论角度，Jev 不是新的 Agent 主体，而是一个 `DecisionModel` / `SemanticJudge`：

```text
AgentInstance
  ├── hasCapability → OpenEndedGeneration（强 LLM）
  ├── hasCapability → BoundedDecision（Jev）
  ├── operatesThrough → Harness（TrueForge / LangGraph）
  └── executesWith → DeterministicCode
```

它把 `Observation` 或结构化 `State` 映射为带概率的 `Decision`，再由 `Policy`、`Plan` 或 `Workflow` 决定后续行动。参见 [[Agent本体论]]、[[Agent框架选型与工程维度]] 和 [[Agent任务分流：工作流与智能体边界]]。

## 适用边界

适合：

- 高频、重复、候选集相对稳定的语义判断；
- 可把问题写成 Choice、Score 或 Noul；
- 允许低置信度回退、人工复核或强模型处理；
- 能建立离线标注和审计轨迹。

不适合直接承担：

- 开放式写作、代码生成和复杂计划；
- 候选空间没有先验边界的问题；
- 需要模型生成新工具、新选项或新概念的问题；
- 没有人工/规则兜底的高风险自动写入；
- 只能依赖一次模型概率、无法回放和校准的生产流程。

## 相关概念

- [[Agent本体论]] — 将 Jev 建模为 Agent 运行时中的决策组件。
- [[Agent框架选型与工程维度]] — Harness 的上下文、工具、状态和治理能力。
- [[Agent任务分流：工作流与智能体边界]] — 确定性代码、Workflow 和 Agent 决策的边界。
- [[Agent记忆系统与自我演进]] — Jev 可用于记忆筛选、压缩和升级判断，但不替代记忆存储。
- [[ByteTech-2026年9月Agent工程实践聚类]] — Jev 与 Harness、知识工程、领域 Agent 实践的横向聚类。

## 来源

- [ByteTech：《一文讲清 Jev》](https://bytetech.info/articles/7687801540488708159?to=award_month) — Jev 原理、架构位置、概率、幻觉边界与上线方法。
- [ByteTech：《Jev 实测：这个只做判断、不聊天的 AI 有什么用？》](https://bytetech.info/articles/7687549451499012138?to=award_month) — 真实接口测试、Seed 2.0 Lite 对照与失效边界。
- [WaytoAGI：《我把 Jev 部署到了一个开源 Agent Harness 中》](https://waytoagi.feishu.cn/wiki/NkknwTOIzimajRkwBepc3KbWnwe?from=from_copylink) — 转载 Akshay 的 TrueForge + Jev 案例。
- [WaytoAGI：《如何使用 Jev 重构 Agent 的运行框架》](https://waytoagi.feishu.cn/wiki/UT2dwb4yki7iW0kRdlCcwnO4nhh?from=from_copylink) — 社区工程解读。
- [TypeSafe：Introducing System One Models & Jev](https://typesafe.ai/blog/introducing-system-one-models-and-jev?ref=taaft) — 官方发布与模型定位、输出类型、成本/延迟声明及限制说明。
- [LangChain：Building Prod with Jev and LangGraph](https://www.langchain.com/blog/building-prod-with-jev-and-langgraph) — Jev 与 LangGraph 的运行时分工、状态、持久化、人机协同和文档审查案例。
- [LangChain：Jev is now available in LangSmith Evals](https://www.langchain.com/blog/jev-is-now-available-in-langsmith-evals) — Jev-as-a-judge 的评测定位、三类问题和早期实验。
- [TrueForge GitHub](https://github.com/truefoundry/trueforge) — 开源 Agent Harness 的工具、沙箱、审批、上下文工程和运行时能力。
- [36氪：刚刚，Jev全网解禁，1.2亿Token限时免费用](https://36kr.com/p/3992838099532548) — 二手报道，补充 Minecraft、多层 Agent 和 fast-jev-compaction 案例；性能数字不作为独立验证。

*最后更新：2026-10-20*
