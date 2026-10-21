# Agent本体论

Agent 本体论描述 Agent 领域中的实体、属性、关系、状态、行动、事件与约束。它回答“什么东西算 Agent、Agent 代表谁、能做什么、如何行动、行动后如何验证和追责”，而不只是列出一个 Agent 系统包含哪些软件模块。

## 核心判断

当前不存在一个覆盖理论、认知架构、工程运行时和跨平台治理的统一 Agent 本体标准。更稳妥的做法是把不同层次分开：

| 层次 | 关注问题 | 代表来源 |
|---|---|---|
| Agent 理论 | Agent 的自治性、反应性、主动性和社会能力 | Wooldridge 与 Jennings 的经典框架 |
| 形式本体 | 如何用类、属性和关系描述 Agent 的行为、目标、任务和承诺 | OASIS 2 |
| 认知架构 | Agent 如何组织记忆、决策和行动 | CoALA |
| LLM Agent 方法 | 如何构造、协作和演化 LLM Agent | LLM Agent 综述 |
| 互操作协议 | 如何发现能力、暴露工具和调用服务 | A2A、MCP |
| 治理与问责 | 谁授权 Agent，权限如何委托，行动如何溯源 | 人类锚定的 Agent 身份架构草案 |

因此，“Agent 本体论”应被理解为一个**分层概念模型**，而不是某个单一框架的同义词。

## Agent 的最低定义

经典智能 Agent 理论通常用四个属性描述 Agent：

- **自治性**：能够在没有持续人工干预的情况下控制自己的行动或内部状态。
- **反应性**：能够感知环境变化并作出响应。
- **主动性**：能够围绕目标发起行动，而不是只被动响应输入。
- **社会能力**：能够与人或其他 Agent 交互、协作或协商。

对于 LLM Agent，仅有语言生成能力还不够。LLM 是推理、生成或决策组件；只有当系统同时具备目标、环境接口、状态管理和行动闭环时，才更接近 Agent。

## 核心实体分类

### 1. 主体与实例

| 实体 | 含义 |
|---|---|
| `Human` | 授权或使用 Agent 的人 |
| `Organization` | 对 Agent、任务或资源负责的组织 |
| `Agent` | 具有身份、能力、目标和行动边界的软件主体 |
| `AgentInstance` | Agent 在某个运行时、会话或部署环境中的具体实例 |
| `Environment` | Agent 感知、影响和受约束的外部环境 |

`Agent` 是相对稳定的能力与身份描述，`AgentInstance` 是一次可运行、可观测和可撤销的部署实体。这个区分对多实例扩展、故障隔离和审计尤其重要。

### 2. 心智与动机

| 实体 | 含义 |
|---|---|
| `Belief` | Agent 对环境、任务或其他主体的判断 |
| `Desire` | Agent 希望实现的状态或偏好 |
| `Goal` | Agent 选定并试图实现的目标状态 |
| `Intention` | Agent 对目标或计划形成的行动承诺 |
| `Commitment` | Agent 对自身或其他主体作出的可追踪承诺 |

`Desire` 不必然导致行动；`Intention` 通常意味着更强的行动决心。工程上不能只记录最终答案，还应区分 Agent“观察到了什么、相信什么、选择了什么目标、承诺执行什么”。

### 3. 能力与接口

| 实体 | 含义 |
|---|---|
| `Capability` | Agent 对外声称能够完成的能力 |
| `Skill` | 可被发现、调用或组合的任务级技能 |
| `Tool` | 具体可调用的外部接口、函数、数据库或执行环境 |
| `Resource` | Agent 可读取或操作的数据、文件、服务或设备 |
| `Input/OutputSchema` | 能力或工具的输入输出约束 |

**Capability 不等于 Tool**：Tool 是接口，Capability 是面向任务的能力承诺。一个能力可能需要多个工具、多个任务步骤和一套验证逻辑共同实现。

### 4. 任务、计划与行动

| 实体 | 含义 |
|---|---|
| `Task` | Agent 需要完成的工作单元 |
| `Plan` | 实现目标的步骤、依赖和备选路径 |
| `Action` | Agent 对环境或内部状态执行的一次操作 |
| `ActionResult` | 行动返回的结果、状态、错误或副作用 |
| `Observation` | Agent 从环境获得的输入或反馈 |
| `Event` | 环境、Agent 或交互中发生的可记录变化 |

任务是意图层的工作单元，行动是运行时实际发生的操作。两者不能混为一谈：计划可能被修改，行动可能失败，结果可能反过来更新目标和记忆。

### 5. 记忆与内部状态

可采用认知架构中的分层方式：

- **工作记忆**：当前任务、对话上下文、临时计划和待确认信息。
- **情景记忆**：过去的交互、行动轨迹、结果和失败经验。
- **语义记忆**：相对稳定的事实、概念、规则和领域知识。
- **程序记忆**：可复用的技能、代码、工具使用方式和操作流程。

记忆不应被简化为一个向量数据库。至少还要描述来源、时间、置信度、适用范围、更新方式、访问权限和遗忘/撤销策略。参见 [[Agent记忆系统与自我演进]]。

## 关键关系

可以用如下关系表达一个最小 Agent 本体：

```text
Human --authorizes--> Agent
Agent --instantiates--> AgentInstance
Agent --operatesIn--> Environment
Agent --hasBelief--> Belief
Agent --pursues--> Goal
Agent --commitsTo--> Intention
Agent --hasCapability--> Capability
Capability --realizedBy--> Tool
Goal --decomposedInto--> Task
Task --plannedBy--> Plan
Plan --contains--> Action
AgentInstance --performs--> Action
Action --uses--> Resource
Action --observes--> Observation
Action --produces--> ActionResult
Action --changes--> Environment
Action --recordedBy--> Trace
```

这是一份面向 LLM Agent 的综合设计建议，不是某个标准直接规定的完整类表。

## OASIS 2：较接近正式本体的建模方式

OASIS 2 采用行为主义路径，用 OWL 2 描述 Agent 如何通过行为实现目标和任务。其核心结构是：

```text
Agent
  └── Behaviour
        └── GoalDescription
              └── TaskDescription
                    └── Action / PlanExecution
```

- `Behaviour` 汇集 Agent 可以实现的目标。
- `GoalDescription` 组织构成目标的任务。
- `TaskDescription` 描述 Agent 能执行的原子操作，并可声明输入参数、输出参数和依赖关系。
- `PlanExecution` 记录某个行为对应的实际行动。
- Agent 的能力、责任、授权和承诺可以脱离具体实现细节进行语义描述，从而支持发现和组合。

OASIS 2 的价值在于把“Agent 能做什么”拆成可表达的目标、任务、参数和行动关系；它的局限是主要面向语义网和多 Agent 互操作，不是专门为现代 LLM 的上下文窗口、提示词、模型调用和评测设计的。

## CoALA：认知架构而非本体标准

CoALA 将语言 Agent 组织成三类要素：

1. **模块化记忆**：工作记忆与长期记忆。
2. **结构化行动空间**：对内部记忆执行检索、推理和学习，对外部环境执行工具调用或其他行动。
3. **决策过程**：通过规划、执行和反馈循环选择下一步行动。

CoALA 适合回答“Agent 如何运行”，OASIS 2 更适合回答“Agent 的行为如何被形式化描述”。两者可以组合，但不能互相替代。

## 协议层：A2A 与 MCP 的边界

### A2A

A2A 的 Agent Card 是远程 Agent 的机器可读名片，主要包含：

- 身份：名称、描述和提供方；
- 服务端点；
- 流式传输、推送通知等协议能力；
- 认证方案；
- `AgentSkill`：技能 ID、名称、描述、输入输出模式和示例。

A2A 解决的是 Agent 发现和 Agent-to-Agent 交互，不是完整的心智、记忆、任务生命周期或责任本体。

### MCP

MCP 主要规范模型如何发现和调用外部工具。工具具有名称、描述、输入 Schema 和调用结果，也可以与资源、提示模板和权限确认机制配合。

MCP 解决的是 Agent 与工具/资源的连接问题，不定义 Agent 的身份、目标、意图、记忆和授权来源。因此：

- `Tool` 是 MCP 层的接口对象；
- `Capability` 是 Agent 本体层的任务能力；
- 二者之间需要由 Agent 运行时建立映射和验证。

## 治理层：身份、委托与溯源

当 Agent 具备真实执行权限时，本体模型必须增加治理实体：

```text
HumanIdentityRoot
  └── Delegation
        └── AgentIdentity
              └── AgentInstance
                    └── ActionEvent
                          └── ProvenanceRecord
```

至少需要表达：

- **授权来源**：哪个人或组织授权了 Agent；
- **权限范围**：允许哪些行动、资源和环境；
- **约束条件**：时间、金额、数据域、审批门或风险等级；
- **委托链**：Agent 是否将部分权限转交给其他 Agent；
- **撤销状态**：权限是否已撤销，撤销是否影响下游委托；
- **行动溯源**：谁授权、哪个 Agent 实例、使用哪个模型和工具、产生了什么结果。

身份草案提出“人类身份根—Agent 身份—委托链—行动溯源”的架构，但它仍是工作草案，不能当作已定型的跨平台标准。

## 与 Workflow 的边界

本体论层面，Workflow 和 Agent 可以共享 `Task`、`Action`、`Event` 和 `Trace` 等实体，但自治程度不同：

| 维度 | Workflow | Agent |
|---|---|---|
| 步骤 | 预先枚举 | 可动态生成或修改 |
| 决策 | 规则和状态机 | 目标驱动的条件决策 |
| 环境反馈 | 通常是显式分支 | 可能改变计划、目标或工具选择 |
| 权限 | 流程级配置 | 需要绑定到 Agent 实例和委托链 |
| 审计 | 记录步骤执行 | 还需记录观察、判断、计划变化和行动理由 |

实际系统通常应采用“确定性 Workflow + 局部 Agent 决策”的组合，而不是把所有任务都包装成 Agent。参见 [[Agent任务分流：工作流与智能体边界]]。

## 工程落地建议

实现一个可审计的 LLM Agent 时，最小数据模型应至少包含：

```text
AgentProfile
  identity
  role
  objectives
  capabilities
  allowed_tools
  memory_policy
  authority_scope

Execution
  execution_id
  agent_instance_id
  task
  plan_version
  observations
  actions
  tool_calls
  results
  approvals
  provenance
```

上线门槛不应只看最终任务成功率，还应检查：

- 能力声明是否与实际可执行能力一致；
- 工具调用是否经过鉴权、限流和参数校验；
- 计划、行动和结果能否关联；
- 失败是否可重试、暂停、回滚或转人工；
- 记忆是否可追溯、可隔离、可删除；
- 多 Agent 委托是否不会扩大原始权限；
- 结果是否能定位到模型、提示、工具、数据和版本。

## 未解决问题

- 如何区分真正的 Agent 行动与由 Workflow 编排的模型调用？
- “意图、理由、信念”应记录为可解释的结构，还是只保留可审计的外部事件？
- Agent 复制、迁移和多实例并发时，身份与权限如何保持一致？
- A2A 的能力声明、MCP 的工具 Schema 与业务能力本体如何自动对齐？
- 如何在不暴露隐私和内部策略的前提下提供足够的行动溯源？
- Agent 的评测应以任务结果为主，还是同时评估记忆、规划、工具使用、授权合规和恢复能力？

## 相关概念

- [[Agent框架选型与工程维度]] — Agent 本体实体在框架和生产系统中的工程映射。
- [[Agent记忆系统与自我演进]] — 记忆实体、记忆基底和持续演进机制。
- [[Agent任务分流：工作流与智能体边界]] — Agent 与确定性 Workflow 的选择边界。
- [[Jev决策模型与Agent架构]] — 将固定选项判断从生成式主循环中拆出的决策模型组件。
- [[ByteTech-2026年9月Agent工程实践聚类]] — Agent 本体实体在内部 Harness、Task、Session、知识与治理实践中的综合映射。
- [[多Agent系统研究趋势与工程映射]] — 多 Agent 之间的角色、通信、协作和能力获取。
- [[知识图谱驱动的深度研究Agent]] — 将证据、工具和研究过程结构化为可追踪的 Agent 工作空间。

## 来源

- [Wooldridge 与 Jennings：Intelligent Agents: Theory and Practice](https://www.cs.ox.ac.uk/people/michael.wooldridge/pubs/ker95.pdf) — 经典 Agent 的自治性、反应性、主动性和社会能力。
- [OASIS 2：The Ontology for Agents, Systems and Integration of Services](https://arxiv.org/html/2306.10061) — 基于 OWL 2 的 Agent、行为、目标、任务和行动建模。
- [CoALA：Cognitive Architectures for Language Agents](https://arxiv.org/html/2309.02427) — 语言 Agent 的记忆、行动空间和决策循环。
- [Large Language Model Agent: A Survey on Methodology, Applications and Challenges](https://arxiv.org/html/2503.21460v1) — LLM Agent 的画像、记忆、规划、行动、协作和演化分类。
- [A2A Agent Discovery](https://a2a-protocol.org/latest/topics/agent-discovery/) — Agent Card、能力发现和 Agent-to-Agent 互操作。
- [MCP Tools Specification](https://modelcontextprotocol.io/specification/2025-06-18/server/tools) — 工具发现、输入 Schema 和工具调用边界。
- [Architecture for Human-Anchored Agent Identity, Delegation, and Provenance](https://www.ietf.org/archive/id/draft-beyer-agent-identity-architecture-00.txt) — 人类身份根、委托链和行动溯源的工作草案；非成熟标准。
- [W3C OWL 2 Primer](https://www.w3.org/2012/pdf/REC-owl2-primer-20121211.pdf) — 类、个体、对象属性和数据属性等本体建模基础。

*最后更新：2026-10-19*
