# AI Agent 常见框架深度介绍

## 引言：AI Agent 的本质与框架演进

AI Agent（智能体）是一种能够**自主感知环境、制定策略、调用工具并执行复杂任务**的系统。与传统的 LLM 对话不同，Agent 具备闭环执行能力——它不仅告诉用户"如何做"，更直接帮用户"做到"。

2023\-2025 年间，Agent 框架经历了三次范式跃迁：

1. **链式调用时代 \(2023 初\)**：以 LangChain 的 Sequential Chain 为代表，LLM 按固定流程串行执行

2. **ReAct 循环时代 \(2023 中\)**：引入"思考\-行动\-观察"循环，Agent 具备动态决策能力

3. **图/事件驱动时代 \(2024\-2025\)**：以 LangGraph、LlamaIndex Workflows 为代表，支持复杂状态管理、循环逻辑和持久化编排

本文系统梳理当前最主流的 AI Agent 框架，覆盖**单智能体编排框架、多智能体协作框架、Agent 开发平台与 SDK**三大板块，并对比不同的认知范式与未来趋势。

---

## 第一部分：单智能体编排框架

单智能体框架关注的核心问题是：**如何让一个 Agent 在复杂任务中可控、可观测、可恢复地执行**。

### LangChain / LangGraph：有向图编排的事实标准

**定位**：全球使用最广泛的 LLM 应用框架，LangGraph 是其面向复杂 Agent 场景的核心进化。

**背景**：Harrison Chase 于 2022 年底创建 LangChain，2024 年初推出 LangGraph。目前被 Klarna、Uber、J\.P\. Morgan 等企业用于生产环境[\[1\]](https://docs.langchain.com/oss/python/langgraph/overview)。

**核心架构 — StateGraph（状态图）**：

LangGraph 将 Agent 的执行建模为**有向图**，图中的每个节点是一个执行单元（LLM 调用、工具执行、代码运行），边定义了节点间的流转逻辑。与传统 DAG 不同，LangGraph 显式支持**循环逻辑**——这对于 Agent "调用工具→观察结果→再次思考→再调用" 的本质循环至关重要[\[2\]](https://blog.langchain.com/building-langgraph/)。

**四大核心组件**：

- **Nodes（节点）**：Python 函数或 Runnable 对象，负责读取状态、执行逻辑、返回状态更新[\[3\]](https://nsddd.top/zh/posts/ai-projects/langgraph/)

- **Edges 与 Conditional Edges（条件边）**：定义流转方向，条件边支持基于 LLM 输出的动态路由[\[4\]](https://media.aivoid.dev/pdfs/step-by-step-tutorials_20260521203153.pdf)

- **Checkpointers（检查点）**：每个超级步结束时通过 Postgres/Redis 持久化状态快照，支持断点恢复和"时间旅行"调试[\[5\]](https://juejin.cn/post/7630875298633515046)

- **Reducers（归约器）**：定义并行节点更新同一状态键时的合并策略[\[6\]](https://docs.langchain.com/oss/python/langgraph/graph-api)

**工具调用机制**：通过 `ToolNode` 将工具定义与执行解耦，利用 Reducers 自动将工具返回结果写入 `MessagesState`[\[7\]](https://reference.langchain.com/python/langgraph/graph/state/StateGraph)。

**内存管理**：基于 `BaseCheckpointSaver` 实现线程级会话管理，支持跨对话的长期记忆[\[8\]](https://docs.langchain.com/oss/python/langgraph/persistence)。

**核心优势**

- 极致的可控性和可观测性

- 完善的持久化与 Human\-in\-the\-loop 机制

- LangSmith（可观测平台）\+ LangServe（部署）形成闭环生态

**主要局限**

- 学习曲线极陡，概念密度高

- 简单任务使用时框架"重量"过大

- 状态图定义需要显式编排，对新手不友好

---

### LlamaIndex Workflows：事件驱动的数据增强智能体

**定位**：以数据检索和知识增强为核心能力的 Agent 框架，Agentic RAG 领域的首选[\[9\]](https://www.marsdevs.com/compare/langchain-vs-llamaindex)。

**背景**：Jerry Liu 创建，2024 年中推出 Workflows 架构，取代了早期的 `AgentWorker` 模式[\[10\]](https://www.llamaindex.ai/blog/building-knowledge-graph-agents-with-llamaindex-workflows)。

**核心架构 — Event\-Driven（事件驱动）**：

不同于 LangGraph 的显式图结构，LlamaIndex Workflows 基于**事件订阅模式**。每个 Step 通过订阅特定类型事件触发执行，执行完后发出新事件，由事件中心调度下一步[\[11\]](https://developers.llamaindex.ai/typescript/framework/tutorials/workflows/)。

**核心组件**：

- **@step 装饰器**：将异步函数注册为工作流步骤[\[12\]](https://developers.llamaindex.ai/python/workflows-api-reference/workflow/)

- **Events**：带类型的 Python 对象，`StartEvent` 和 `StopEvent` 控制生命周期[\[12\]](https://developers.llamaindex.ai/python/workflows-api-reference/workflow/)

- **Context**：步骤间共享数据的并发安全容器[\[13\]](https://developers.llamaindex.ai/typescript/workflows/)

**差异化能力**：对 20\+ 种文档格式的深度语义处理、多种检索策略（向量/关键词/知识图谱）的智能选择、以及将整个 RAG 管道包装为 Agent 工具的 `FunctionCallingAgentWorker`。

**核心优势**

- 数据密集型任务处理效率极高

- 异步事件驱动天然适合高并发

- 检索增强能力无出其右

**主要局限**

- 严格顺序逻辑的可读性不如显式图

- Agent 高层抽象相对较新

- 非 RAG 场景下优势不明显

---

### Semantic Kernel \(Microsoft\)：企业级 Agent 操作系统

**定位**：微软统一的 Agent SDK，2025 年与 AutoGen 深度融合为 Microsoft Agent Framework \(MAF\)[\[14\]](https://www.openaitoolshub.org/en/blog/microsoft-agent-framework-review)。

**背景**：微软出品，提供 \.NET 和 Python 双原生支持。2026 年发布 1\.0 LTS 版本[\[15\]](https://cloudsummit.eu/blog/microsoft-agent-framework-production-ready-convergence-autogen-semantic-kernel)。

**核心架构 — Kernel\-Centric（内核驱动）**：

以 `Kernel` 为中心容器，管理三类资源：**Connectors**（模型连接器）、**Plugins**（工具插件）和 **Services**（企业服务）。

**关键能力**：

- **Plugins（插件）**：将任意代码包装为语义函数，支持自动发现

- **Orchestration Patterns**：内置 Sequential、Group Chat、Handoff 等编排模式[\[16\]](https://learn.microsoft.com/en-us/semantic-kernel/Frameworks/agent/agent-architecture)

- **A2A \(Agent\-to\-Agent\)**：支持跨框架通信标准[\[14\]](https://www.openaitoolshub.org/en/blog/microsoft-agent-framework-review)

- **Planners**：Stepwise Planner 自动将用户意图拆解为执行步骤[\[17\]](https://devblogs.microsoft.com/autogen/microsofts-agentic-frameworks-autogen-and-semantic-kernel/)

**核心优势**

- 企业级安全、监控与合规性

- 与 Azure AI / Microsoft 365 深度集成

- 强类型系统保证高 Tool Calling 成功率

**主要局限**

- 架构繁重，需遵循微软编程范式

- 开源社区活跃度低于 LangChain

- 对非微软生态的适配不够灵活

---

### Haystack \(deepset\)：模块化 NLP 管道的工业标杆

**定位**：极致模块化的 NLP/LLM 框架，2024 年 2\.0 版本彻底重构[\[18\]](https://haystack.deepset.ai/release-notes/2.0.0)。

**背景**：德国 deepset 公司出品，设计哲学为"乐高式"组件拼装。

**核心架构 — 有向多重图管道（Pipeline）**：

组件通过 `@component` 装饰器定义，具有显式的**输入/输出类型校验**，保证连接的鲁棒性。Pipeline 负责连接组件，支持分支、合并和循环[\[19\]](https://docs.haystack.deepset.ai/docs/next/pipelines)。

**Agent 能力**：内置 `Agent` 类和 `ToolInvoker` 组件，后者负责执行 LLM 准备好的工具调用并将结果回写消息流[\[20\]](https://haystack.deepset.ai/cookbook/swarm)。

**核心优势**

- 类型安全，组件连接逻辑透明

- 性能优越，易于在大规模环境调试

- 支持自修复流（Self\-correcting flows）

**主要局限**

- Agent 高层封装相对较少

- 复杂动态逻辑需手动连接 Pipeline

- 生态规模小于 LangChain

---

## 第二部分：多智能体协作框架

多智能体框架解决的核心问题是：**如何让多个具有不同角色和能力的 Agent 高效协作完成复杂任务**。

### AutoGen / AG2 \(Microsoft\)：对话即编程

**定位**：最早的生产级多智能体框架之一，2024 年品牌重塑为 AG2，由开放治理组织 AG2AI 维护[\[21\]](https://pypi.org/project/autogen/)。

**背景**：微软研究院 2023 年推出，2025 年与 Semantic Kernel 整合为 Microsoft Agent Framework[\[22\]](https://aiwiki.ai/wiki/autogen/revisions/8)。

**核心架构 — Actor Model（参与者模型）**：

AG2 v0\.4 采用完全异步、事件驱动的 Actor 架构[\[23\]](https://www.microsoft.com/en-us/research/articles/autogen-v0-4-reimagining-the-foundation-of-agentic-ai-for-scale-extensibility-and-robustness/)。核心理念是"对话即编程"——通过定义不同角色的智能体，让它们在对话中协商解决问题。

**协作模式**：

- **点对点对话**：两个 Agent 直接交互

- **GroupChatManager**：管理多 Agent 讨论流程

- **Magentic\-One**：多层级协调架构，Orchestrator 管理 WebSurfer、Coder 等专业 Agent，负责规划、进度追踪和错误恢复[\[24\]](https://blog.csdn.net/jackie_vip/article/details/146522360)

**Human\-in\-the\-loop**：通过 `human_input_mode` 提供 NEVER / TERMINATE / ALWAYS 三种干预模式[\[25\]](http://microsoft.github.io/autogen/0.2/docs/tutorial/human-in-the-loop/)。

**典型场景**：科学工作流（如宇宙学参数推断）、工艺优化、Magentic\-One 通用任务解决[\[26\]](https://www.emergentmind.com/topics/autogen)。

---

### CrewAI：角色驱动的团队协作

**定位**：增长最快的商用多智能体框架，强调"角色扮演"的直觉化设计[\[27\]](https://aiwiki.ai/wiki/crew_ai)。

**背景**：Joao Moura 于 2024 年创建，已脱离 LangChain 依赖，改用 LiteLLM 接入模型。

**核心原语**：

- **Agent**：具备 Role（角色）、Goal（目标）、Backstory（背景故事）

- **Task**：明确的执行单元

- **Crew**：Agent 集合 \+ Process 定义

- **Process**：Sequential（顺序）或 Hierarchical（层级管理）[\[28\]](https://github.com/crewaiinc/crewai)

**通信模式**：典型的 Hub\-and\-Spoke（中心辐射），避免 Agent 之间杂乱通信[\[29\]](https://www.emergentmind.com/topics/crewai-framework)。Hierarchical 模式下 Manager Agent 负责任务分配和结果验证。

**典型场景**：营销内容流水线、市场研究报告自动化、企业 SOP 执行[\[30\]](https://data4ai.com/blog/vendor-spotlights/crewai-company-2026/)。

**核心优势**

- 上手门槛极低，YAML 配置友好

- 角色定义直觉化，模拟真实岗位分工

- 非常适合结构化业务流程

**主要局限**

- 极复杂非线性任务中稳定性不足

- 层次化管理的边界情况处理较弱

- 不如 AutoGen 灵活

---

### MetaGPT：软件公司模拟器

**定位**：将 SOP（标准作业程序）编码进 Agent 提示词，模拟完整软件公司工作流。ICLR 2024 Oral 论文[\[31\]](https://github.com/FoundationAgents/MetaGPT)。

**核心机制**：

- **SOP 编码**：将产品经理、架构师、工程师等角色的标准交付物（PRD、系统设计、代码）预定义为工作流[\[32\]](https://arxiv.org/pdf/2308.00352v7)

- **消息发布/订阅**：Agent 在共享环境中发布消息，相关角色根据职责监听并响应[\[33\]](https://deepwiki.com/geekan/MetaGPT)

- **AFlow**：2025 年推出的自动工作流生成引擎

**典型场景**：从需求到全栈代码的自动化生成、技术方案文档设计。

**核心优势**

- SOP 极大减少协作中的逻辑不一致

- 软件工程自动化效率极高

- 学术研究与工程实践结合紧密

**主要局限**

- 工作流相对固定

- 开放式创意任务灵活性不足

- 非软件工程场景适配有限

---

### CAMEL：角色扮演的学术先驱

**背景**：KAUST（沙特阿卜杜拉国王科技大学）2023 年推出，首创 Role\-Playing 多智能体通信范式[\[34\]](https://github.com/camel-ai/camel)。

**核心机制**：通过"启发式提示"（Inception Prompting）初始化两个角色，让它们在无人类干预下通过自主对话完成任务[\[35\]](https://bytedance.larkoffice.com/wiki/ApTYwzU91i2RY0koJlWcpS0GnLc?from=lark_search_qa&ccm_open_type=lark_search_qa)。2025 年的重点转向"智能体缩放定律"（Scaling Law of Agents）[\[36\]](https://www.camel-ai.org/)。

**典型场景**：科研探索模拟、博弈策略模拟、多智能体交互数据集生成。

---

### OpenAI Swarm：极简转接模式

**背景**：OpenAI Solutions 团队 2024 年 10 月发布的实验性框架，现已被 OpenAI Agents SDK 取代[\[37\]](https://github.com/openai/swarm)。

**核心机制**：仅两个原语——**Routines**（指令\+工具集）和 **Handoffs**（控制权转接）[\[38\]](https://galileo.ai/blog/openai-swarm-framework-multi-agents)。逻辑类似客服中心：主 Agent 根据意图将请求"转接"给专家 Agent。

**典型场景**：多轮客服支持、个人购物助手。

**趋势洞察**：Swarm 虽已停止维护，但其 "Handoff" 原语被 OpenAI Agents SDK 完整继承，成为了多 Agent 协作的核心设计模式之一。

---

## 第三部分：Agent 开发平台与 SDK

### OpenAI Agents SDK：模型原生的极简原语

**定位**：OpenAI 2025 年 3 月发布的生产级 Agent 工具包，Swarm 的正式继任者[\[39\]](https://mem0.ai/blog/openai-agents-sdk-review)。

**四个核心原语**：

- **Agents**：封装指令和工具集的 LLM 实体

- **Tools**：可执行函数

- **Handoffs**：通过 `transfer_to_agent` 实现 Agent 间控制权转移[\[40\]](https://callsphere.ai/blog/openai-agents-sdk-deep-dive-agents-tools-handoffs-guardrails-2026)

- **Guardrails**：输入输出验证与安全控制

**关键差异化**：模型原生设计、内置沙盒执行环境、原生追踪系统，强调开发者对编排的绝对控制权[\[41\]](https://openai.com/index/the-next-evolution-of-the-agents-sdk/)。

**局限**：深度绑定 OpenAI 模型生态，跨模型支持较弱。

---

### Anthropic MCP \(Model Context Protocol\)：AI 领域的 USB\-C

**定位**：2024 年 11 月发布的通讯标准（非框架），解决模型与外部工具/数据的连接碎片化问题[\[42\]](https://en.wikipedia.org/wiki/Model_Context_Protocol)。

**架构**：Client\-Server 模式，三种标准化原语：

- **Resources**：只读上下文数据（文件、数据库记录）[\[43\]](https://developer.aliyun.com:443/article/1733890)

- **Tools**：可执行函数（有副作用）[\[44\]](https://dzone.com/articles/anthropics-model-context-protocol-mcp-a-developers)

- **Prompts**：预定义交互模板

**核心价值**：开发者实现一次 MCP Server，即可在 Cursor、Claude Desktop、Dify 等所有兼容客户端中使用[\[45\]](https://blog.promptlayer.com/mcp/)。MCP 解决了"M×N 集成瓶颈"——未来 Agent 的能力边界将由可访问的 MCP Server 数量决定，而非框架内置的插件数量[\[46\]](https://www.anthropic.com/engineering/code-execution-with-mcp)。

---

### Dify：开源 LLMOps 全栈引擎

**定位**：最受欢迎的开源 Agent 平台之一，2025 年发布 v1\.0，定位"模型中立"[\[47\]](https://dify.ai/blog/dify-v1-0-building-a-vibrant-plugin-ecosystem)。

**核心架构 — 蜂巢架构（Honeybee Architecture）**：

高度解耦的插件系统，模型、工具、数据源均插件化运行[\[48\]](https://www.dify-china.com/about.html)。提供成熟的可视化工作流编排和内置企业级 RAG 管道。

**典型用途**：企业内部专家系统、自动化客服、多模型能力对比评测。

**优势**：上手门槛极低，适合快速原型；**局限**：极复杂动态推理循环时，可视化界面表达力受限[\[49\]](https://murmure.cc/dify-community-report.pdf)。

---

### Coze / 扣子 \(ByteDance\)：从开发到分发的闭环

**定位**：字节跳动 Agent 开发平台，2025 年开源核心组件 Coze Studio 与 Coze Loop[\[50\]](https://www.aibase.com/news/19989)。

**核心架构**：底层由 Golang 实现的 **Eino 框架** 支撑，将 Agent 拆解为 Use Case、Prompts 与 Skills 三部分[\[51\]](https://ecweb.ecer.com/topic/en/detail-255947-coze_platform_shifts_focus_from_tool_orchestration_to_ai_app_development.html)。

**差异化**：极致的生态集成能力——内置多模态插件（PPT 生成、绘画等），支持一键发布至飞书、微信、Telegram 等多端。

**优势**：社交/办公场景深度打通，多模态能力强；**局限**：底层推理参数微调深度略逊纯代码框架。

---

### Google ADK \(Agent Development Kit\)：代码优先的多代理框架

**定位**：2025 年 4 月 Google Cloud Next 大会发布，强调"编织确定性代码与自适应 AI"[\[52\]](https://adk.dev/)。

**核心架构**：引入 **Workflow Agents**（SequentialAgent、ParallelAgent、LoopAgent）作为确定性调度器，将推理交给 **LlmAgent** 处理[\[53\]](https://cloud.google.com/blog/topics/developers-practitioners/building-collaborative-ai-a-developers-guide-to-multi-agent-systems-with-adk)。

**差异化**：原生支持 Gemini 双向流式通信和 Session 持久化，企业级评估工具极其专业。

**局限**：主要针对 Google Cloud 生态优化。

---

### AutoGPT / BabyAGI：自主 Agent 的先行探索

2023 年掀起"自主 Agent"热潮的鼻祖项目[\[54\]](https://vucense.com/ai-intelligence/agentic-ai/best-ai-agents-2026-retrospective/)。BabyAGI 2（2024 年底）转向"自构建"架构，探索智能体通过编写代码实现自我改进[\[55\]](https://aiwiki.ai/wiki/babyagi)。

**历史贡献**：验证了 LLM 具备自主规划和工具调用能力的可行性。**局限**：生产环境中仍面临幻觉严重、Token 成本不可控、任务成功率低的问题[\[56\]](https://www.runlobster.com/news/openclaw-vs-autogpt-vs-babyagi-2026-autonomous-agents)。

---

## 第四部分：Agent 认知范式对比

|**范式**|**核心逻辑**|**技术特点**|**适用场景**|
|---|---|---|---|
|**ReAct**|思考→行动→观察循环|动态响应环境变化，每步都可修正方向[\[57\]](https://www.xcubelabs.com/blog/how-different-types-of-ai-agents-work-a-comprehensive-taxonomy-and-guide/)|搜索助手、简单 API 调研、实时交互|
|**Plan\-and\-Execute**|先制定完整计划，再分步执行|减少无效 Token 消耗，逻辑路径清晰可审计[\[58\]](https://byaiteam.com/blog/2025/12/09/ai-agent-planning-react-vs-plan-and-execute-for-reliability/)|科学研究、复杂项目管理、长程任务|
|**Function Calling**|模型输出强制转化为结构化指令|确定性最高，输出可直接对接业务系统[\[59\]](https://www.ibm.com/es-es/think/topics/react-agent)|数据库操作、CRM 更新、API 编排|
|**Workflow\-based**|预定义 DAG 或状态机编排|路径完全可预测，易于审计和回滚[\[60\]](https://deepagentsdk.dev/blog/26-01-01-the-pendulum-swing-of-llm-frameworks)|金融风控、合规审批、生产流水线|

**实践经验**：2024\-2025 年的趋势表明，生产级 Agent 系统普遍采用"混合范式"——用 Workflow 限定主干逻辑确保可靠性，仅在分支决策处引入 ReAct 或 Plan\-and\-Execute 的自主推理能力。

---

## 第五部分：框架全景对比

|**框架**|**类型**|**设计范式**|**核心能力**|**最佳场景**|**语言支持**|**社区活跃度**|
|---|---|---|---|---|---|---|
|LangGraph|单 Agent|有向状态图|极致可控的循环编排|复杂业务流程 Agent|Python, JS|极高|
|LlamaIndex|单 Agent|事件驱动|数据检索与知识增强|Agentic RAG|Python|高|
|Semantic Kernel|单/多 Agent|内核\+插件|企业级合规编排|微软生态企业应用|\.NET, Python|中高|
|Haystack|单 Agent|组件管道|类型安全的模块化|生产级 NLP 检索|Python|中|
|AutoGen/AG2|多 Agent|Actor 模型|灵活的多 Agent 对话|研究与工业优化|Python|高|
|CrewAI|多 Agent|角色\+流程|直觉化团队协作|企业自动化 SOP|Python|高|
|MetaGPT|多 Agent|SOP 编码|软件工程自动化|代码生成与设计|Python|中高|
|OpenAI Agents SDK|平台 SDK|极简原语|模型原生集成|OpenAI 生态开发|Python|高|
|MCP|协议标准|Client\-Server|工具连接标准化|跨框架工具集成|语言无关|极高|
|Dify|LLMOps 平台|蜂巢插件化|可视化编排\+RAG|快速原型与企业应用|Python|极高|
|Coze/扣子|Agent 平台|Eino\+插件|多端分发\+多模态|社交/办公场景|Go, Python|高|
|Google ADK|平台 SDK|Workflow Agents|确定性\+自适应融合|Google Cloud 生态|Python|中|

---

## 第六部分：2024\-2025 关键趋势与选型建议

### 五大趋势

**1\. 从"全自动"到"确定性\+自主性融合"**

行业不再追求纯自主 Agent。生产系统普遍采用 Workflow 限定主干路径，仅在决策分支处引入 AI 推理——即"Weaving deterministic code with adaptive AI reasoning"[\[52\]](https://adk.dev/)。

**2\. MCP 协议确立连接层标准**

MCP 解决了 M×N 集成瓶颈。Agent 的能力边界不再取决于框架内置插件数量，而是可访问的 MCP Server 生态规模[\[46\]](https://www.anthropic.com/engineering/code-execution-with-mcp)。

**3\. 多 Agent 从实验走向生产**

单体 Agent 被"专家集群"取代。通过 Handoff 模式或层级调度，复杂任务被拆解给垂直领域专家 Agent[\[61\]](https://tao-hpu.medium.com/ai-agent-landscape-2025-2026-a-technical-deep-dive-abda86db7ae2)。

**4\. Golang 框架崛起**

随着 Agent 进入大规模部署，并发性能需求推动 Go 语言框架（如 ByteDance Eino）挑战 Python 的主导地位[\[62\]](https://www.kdjingpai.com/en/eino/)。

**5\. 可观测性成为刚需**

LangSmith、OpenTelemetry、Google ADK 评估工具等，使 Agent 行为的追踪、调试和成本控制成为框架标配能力。

---

### 选型决策矩阵

**选型核心原则**：没有"最好的框架"，只有"最匹配场景的框架"。以下决策路径供参考：

- **需要极致可控性 \+ 复杂状态管理** → LangGraph

- **数据检索/知识增强为核心** → LlamaIndex

- **微软/Azure 企业生态** → Semantic Kernel \(MAF\)

- **多 Agent 灵活协作 \+ 研究探索** → AutoGen/AG2

- **快速搭建团队协作 Agent** → CrewAI

- **软件工程自动化** → MetaGPT

- **低代码快速上线** → Dify 或 Coze

- **OpenAI 模型深度集成** → OpenAI Agents SDK

- **Google Cloud 生态** → Google ADK

- **跨框架工具标准化** → 优先接入 MCP 协议

## 参考资料

[\[](https://docs.langchain.com/oss/python/langgraph/overview)[1\] LangGraph overview \- Docs by LangChain](https://docs.langchain.com/oss/python/langgraph/overview)

[\[](https://blog.langchain.com/building-langgraph/)[2\] Building LangGraph: Designing an Agent Runtime from first principles](https://blog.langchain.com/building-langgraph/)

[\[](https://nsddd.top/zh/posts/ai-projects/langgraph/)[3\] LangGraph 深度解析:设计、架构、原理与应用](https://nsddd.top/zh/posts/ai-projects/langgraph/)

[\[](https://media.aivoid.dev/pdfs/step-by-step-tutorials_20260521203153.pdf)[4\] Step\-by\-Step Tutorials](https://media.aivoid.dev/pdfs/step-by-step-tutorials_20260521203153.pdf)

[\[](https://juejin.cn/post/7630875298633515046)[5\] LangGraph 资深开发者完全指南\#\#\# 1\.1 LangGraph 是什么 LangGraph 是 LangCha \- 掘金](https://juejin.cn/post/7630875298633515046)

[\[](https://docs.langchain.com/oss/python/langgraph/graph-api)[6\] Graph API overview](https://docs.langchain.com/oss/python/langgraph/graph-api)

[\[](https://reference.langchain.com/python/langgraph/graph/state/StateGraph)[7\] StateGraph \| langgraph \| LangChain Reference](https://reference.langchain.com/python/langgraph/graph/state/StateGraph)

[\[](https://docs.langchain.com/oss/python/langgraph/persistence)[8\] Persistence \- Docs by LangChain](https://docs.langchain.com/oss/python/langgraph/persistence)

[\[](https://www.marsdevs.com/compare/langchain-vs-llamaindex)[9\] LangChain vs LlamaIndex: Choosing the Right RAG Framework in 2026](https://www.marsdevs.com/compare/langchain-vs-llamaindex)

[\[](https://www.llamaindex.ai/blog/building-knowledge-graph-agents-with-llamaindex-workflows)[10\] Building knowledge graph agents with LlamaIndex Workflows](https://www.llamaindex.ai/blog/building-knowledge-graph-agents-with-llamaindex-workflows)

[\[](https://developers.llamaindex.ai/typescript/framework/tutorials/workflows/)[11\] Workflows](https://developers.llamaindex.ai/typescript/framework/tutorials/workflows/)

[\[](https://developers.llamaindex.ai/python/workflows-api-reference/workflow/)[12\] Workflow](https://developers.llamaindex.ai/python/workflows-api-reference/workflow/)

[\[](https://developers.llamaindex.ai/typescript/workflows/)[13\] LlamaIndex Workflows](https://developers.llamaindex.ai/typescript/workflows/)

[\[](https://www.openaitoolshub.org/en/blog/microsoft-agent-framework-review)[14\] Microsoft Agent Framework — Semantic Kernel Meets AutoGen in One SDK](https://www.openaitoolshub.org/en/blog/microsoft-agent-framework-review)

[\[](https://cloudsummit.eu/blog/microsoft-agent-framework-production-ready-convergence-autogen-semantic-kernel)[15\] Microsoft Agent Framework: The production\-ready convergence of AutoGen and Semantic Kernel \- European AI \& Cloud Summit](https://cloudsummit.eu/blog/microsoft-agent-framework-production-ready-convergence-autogen-semantic-kernel)

[\[](https://learn.microsoft.com/en-us/semantic-kernel/Frameworks/agent/agent-architecture)[16\] An Overview of the Agent Architecture](https://learn.microsoft.com/en-us/semantic-kernel/Frameworks/agent/agent-architecture)

[\[](https://devblogs.microsoft.com/autogen/microsofts-agentic-frameworks-autogen-and-semantic-kernel/)[17\] Microsoft’s Agentic Frameworks: AutoGen and Semantic Kernel](https://devblogs.microsoft.com/autogen/microsofts-agentic-frameworks-autogen-and-semantic-kernel/)

[\[](https://haystack.deepset.ai/release-notes/2.0.0)[18\] Haystack 2\.0\.0](https://haystack.deepset.ai/release-notes/2.0.0)

[\[](https://docs.haystack.deepset.ai/docs/next/pipelines)[19\] Pipelines](https://docs.haystack.deepset.ai/docs/next/pipelines)

[\[](https://haystack.deepset.ai/cookbook/swarm)[20\] Create a Swarm of Agents](https://haystack.deepset.ai/cookbook/swarm)

[\[](https://pypi.org/project/autogen/)[21\] autogen · PyPI](https://pypi.org/project/autogen/)

[\[](https://aiwiki.ai/wiki/autogen/revisions/8)[22\] AutoGen](https://aiwiki.ai/wiki/autogen/revisions/8)

[\[](https://www.microsoft.com/en-us/research/articles/autogen-v0-4-reimagining-the-foundation-of-agentic-ai-for-scale-extensibility-and-robustness/)[23\] AutoGen v0\.4: Reimagining the foundation of agentic AI for scale, extensibility, and robustness \- Microsoft Research](https://www.microsoft.com/en-us/research/articles/autogen-v0-4-reimagining-the-foundation-of-agentic-ai-for-scale-extensibility-and-robustness/)

[\[](https://blog.csdn.net/jackie_vip/article/details/146522360)[24\] 【大模型系列篇】深剖AutoGen多智能体协作框架发展脉络，快速入门简单示例\_](https://blog.csdn.net/jackie_vip/article/details/146522360)[ag2 autogen\-CSDN博客](https://blog.csdn.net/jackie_vip/article/details/146522360)

[\[](http://microsoft.github.io/autogen/0.2/docs/tutorial/human-in-the-loop/)[25\] Allowing Human Feedback in Agents](http://microsoft.github.io/autogen/0.2/docs/tutorial/human-in-the-loop/)

[\[](https://www.emergentmind.com/topics/autogen)[26\] AutoGen: LLM\-Driven Multi\-Agent Framework](https://www.emergentmind.com/topics/autogen)

[\[](https://aiwiki.ai/wiki/crew_ai)[27\] CrewAI](https://aiwiki.ai/wiki/crew_ai)

[\[](https://github.com/crewaiinc/crewai)[28\] GitHub \- crewAIInc/crewAI: Framework for orchestrating role\-playing, autonomous AI agents\. By fostering collaborative intelligence, CrewAI empowers agents to work together seamlessly, tackling complex tasks\. · GitHub](https://github.com/crewaiinc/crewai)

[\[](https://www.emergentmind.com/topics/crewai-framework)[29\] CrewAI Framework: Secure Multi\-Agent Orchestration](https://www.emergentmind.com/topics/crewai-framework)

[\[](https://data4ai.com/blog/vendor-spotlights/crewai-company-2026/)[30\] CrewAI Review \(2026\): Multi agent platform for enterprise AI automation](https://data4ai.com/blog/vendor-spotlights/crewai-company-2026/)

[\[](https://github.com/FoundationAgents/MetaGPT)[31\] GitHub \- FoundationAgents/MetaGPT: 🌟 The Multi\-Agent Framework: First AI Software Company, Towards Natural Language Programming](https://github.com/FoundationAgents/MetaGPT)

[\[](https://arxiv.org/pdf/2308.00352v7)[32\] MetaGPT: Meta Programming for a Multi\-Agent Collaborative Framework](https://arxiv.org/pdf/2308.00352v7)

[\[](https://deepwiki.com/geekan/MetaGPT)[33\] geekan/MetaGPT \| DeepWiki](https://deepwiki.com/geekan/MetaGPT)

[\[](https://github.com/camel-ai/camel)[34\] GitHub \- camel\-ai/camel: 🐫 CAMEL: The first and the best multi\-agent framework\. Finding the Scaling Law of Agents\. https://www\.camel\-ai\.org](https://github.com/camel-ai/camel)

[\[](https://bytedance.larkoffice.com/wiki/ApTYwzU91i2RY0koJlWcpS0GnLc?from=lark_search_qa&ccm_open_type=lark_search_qa)[35\] AI Agent: Industry Deep Dive 2025\-07](https://bytedance.larkoffice.com/wiki/ApTYwzU91i2RY0koJlWcpS0GnLc?from=lark_search_qa&ccm_open_type=lark_search_qa)

[\[](https://www.camel-ai.org/)[36\] CAMEL\-AI \| Finding the Scaling Laws of Agents \| CAMEL\-AI](https://www.camel-ai.org/)

[\[](https://github.com/openai/swarm)[37\] GitHub \- openai/swarm: Educational framework exploring ergonomic, lightweight multi\-agent orchestration\. Managed by OpenAI Solution team\. · GitHub](https://github.com/openai/swarm)

[\[](https://galileo.ai/blog/openai-swarm-framework-multi-agents)[38\] OpenAI Swarm Framework Guide for Reliable Multi\-Agents \| Galileo](https://galileo.ai/blog/openai-swarm-framework-multi-agents)

[\[](https://mem0.ai/blog/openai-agents-sdk-review)[39\] The OpenAI Agents SDK Review and Alternatives: Best AI Agent Framework \(December 2025\)](https://mem0.ai/blog/openai-agents-sdk-review)

[\[](https://callsphere.ai/blog/openai-agents-sdk-deep-dive-agents-tools-handoffs-guardrails-2026)[40\] OpenAI Agents SDK Deep Dive: Agents, Tools, Handoffs, and Guardrails Explained — Openai agents sdk documentation 2026 tools handoffs tracing \| CallSphere Blog](https://callsphere.ai/blog/openai-agents-sdk-deep-dive-agents-tools-handoffs-guardrails-2026)

[\[](https://openai.com/index/the-next-evolution-of-the-agents-sdk/)[41\] The next evolution of the Agents SDK \| OpenAI](https://openai.com/index/the-next-evolution-of-the-agents-sdk/)

[\[](https://en.wikipedia.org/wiki/Model_Context_Protocol)[42\] Model Context Protocol \- Wikipedia](https://en.wikipedia.org/wiki/Model_Context_Protocol)

[\[](https://developer.aliyun.com:443/article/1733890)[43\] 面试被问MCP?看这一篇文章就行了\-阿里云开发者社区](https://developer.aliyun.com:443/article/1733890)

[\[](https://dzone.com/articles/anthropics-model-context-protocol-mcp-a-developers)[44\] Anthropic’s Model Context Protocol \(MCP\): A Developer’s Guide to Long\-Context LLM Integration](https://dzone.com/articles/anthropics-model-context-protocol-mcp-a-developers)

[\[](https://blog.promptlayer.com/mcp/)[45\] What is MCP? Claude Anthropic's Model Context Protocol](https://blog.promptlayer.com/mcp/)

[\[](https://www.anthropic.com/engineering/code-execution-with-mcp)[46\] Code execution with MCP: building more efficient AI agents](https://www.anthropic.com/engineering/code-execution-with-mcp)

[\[](https://dify.ai/blog/dify-v1-0-building-a-vibrant-plugin-ecosystem)[47\] Dify v1\.0\.0: Building a Vibrant Plugin Ecosystem \- Dify Blog](https://dify.ai/blog/dify-v1-0-building-a-vibrant-plugin-ecosystem)

[\[](https://www.dify-china.com/about.html)[48\] Dify \- 关于我们](https://www.dify-china.com/about.html)

[\[](https://murmure.cc/dify-community-report.pdf)[49\] Dify\.ai Community Intelligence Report](https://murmure.cc/dify-community-report.pdf)

[\[](https://www.aibase.com/news/19989)[50\] Bytedance Announces Two Major Core Projects of Coze Officially Open\-Source: Coze Studio and Coze Loop](https://www.aibase.com/news/19989)

[\[](https://ecweb.ecer.com/topic/en/detail-255947-coze_platform_shifts_focus_from_tool_orchestration_to_ai_app_development.html)[51\] Coze Platform Shifts Focus from Tool Orchestration to AI App Development](https://ecweb.ecer.com/topic/en/detail-255947-coze_platform_shifts_focus_from_tool_orchestration_to_ai_app_development.html)

[\[](https://adk.dev/)[52\] Agent Development Kit \(ADK\) \- Agent Development Kit \(ADK\)](https://adk.dev/)

[\[](https://cloud.google.com/blog/topics/developers-practitioners/building-collaborative-ai-a-developers-guide-to-multi-agent-systems-with-adk)[53\] Building Collaborative AI: A Developer's Guide to Multi\-Agent Systems with ADK \| Google Cloud Blog](https://cloud.google.com/blog/topics/developers-practitioners/building-collaborative-ai-a-developers-guide-to-multi-agent-systems-with-adk)

[\[](https://vucense.com/ai-intelligence/agentic-ai/best-ai-agents-2026-retrospective/)[54\] Best AI Agents 2026: A Full Retrospective](https://vucense.com/ai-intelligence/agentic-ai/best-ai-agents-2026-retrospective/)

[\[](https://aiwiki.ai/wiki/babyagi)[55\] BabyAGI](https://aiwiki.ai/wiki/babyagi)

[\[](https://www.runlobster.com/news/openclaw-vs-autogpt-vs-babyagi-2026-autonomous-agents)[56\] OpenClaw vs AutoGPT vs BabyAGI: 2026 Autonomous Agents Compared](https://www.runlobster.com/news/openclaw-vs-autogpt-vs-babyagi-2026-autonomous-agents)

[\[](https://www.xcubelabs.com/blog/how-different-types-of-ai-agents-work-a-comprehensive-taxonomy-and-guide/)[57\] How Different Types of AI Agents Work: A Comprehensive Taxonomy and Guide](https://www.xcubelabs.com/blog/how-different-types-of-ai-agents-work-a-comprehensive-taxonomy-and-guide/)

[\[](https://byaiteam.com/blog/2025/12/09/ai-agent-planning-react-vs-plan-and-execute-for-reliability/)[58\] AI Agent Planning: ReAct vs Plan and Execute for Reliability – By AI Team](https://byaiteam.com/blog/2025/12/09/ai-agent-planning-react-vs-plan-and-execute-for-reliability/)

[\[](https://www.ibm.com/es-es/think/topics/react-agent)[59\] ¿Qué es un agente ReAct?](https://www.ibm.com/es-es/think/topics/react-agent)

[\[](https://deepagentsdk.dev/blog/26-01-01-the-pendulum-swing-of-llm-frameworks)[60\] The Pendulum Swing of LLM Frameworks](https://deepagentsdk.dev/blog/26-01-01-the-pendulum-swing-of-llm-frameworks)

[\[](https://tao-hpu.medium.com/ai-agent-landscape-2025-2026-a-technical-deep-dive-abda86db7ae2)[61\] AI Agent Landscape 2025–2026: A Technical Deep Dive \| by Tao An \| Medium](https://tao-hpu.medium.com/ai-agent-landscape-2025-2026-a-technical-deep-dive-abda86db7ae2)

[\[](https://www.kdjingpai.com/en/eino/)[62\] Eino: ByteDance's Open Source Golang Big Model Application Development Framework \- AI Productivity Tools](https://www.kdjingpai.com/en/eino/)

