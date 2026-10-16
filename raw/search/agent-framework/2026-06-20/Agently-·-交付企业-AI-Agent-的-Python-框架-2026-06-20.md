# Agently · 交付企业 AI Agent 的 Python 框架

- 来源链接: https://agently.cn/
- 搜索日期: 2026-06-20

## 详细内容

## Agently · 交付企业 AI Agent 的 Python 框架

**Source**: https://agently.cn/

---

企业 AI Agent 从试点到上线 · Python 3.10+

# 让 Agent
交付业务结果

             大模型会理解和生成，企业系统还需要字段、权限、流程、服务接口和运行证据。Agently 把这些上线缺口接成一条可验证的工程链路。           

**框架安装方法
`pip install -U agently`

[查看仓库源码](https://github.com/AgentEra/Agently)[为开发 Agent 安装 Skills](https://agently.cn/#skills)

[1,588
              GitHub stars](https://github.com/AgentEra/Agently/stargazers)[4.1.3.6
              PyPI 当前版本](https://pypi.org/project/agently/)[176
              GitHub forks](https://github.com/AgentEra/Agently/network/members)

业务方看到什么
**工单、日报、控制台状态、审核记录，而不是一段无法落库的回答。**

技术方拿到什么
**类型字段、授权动作、事件流程、服务接口和可追踪运行证据。**

开发者怎么开始
**安装框架跑通最小链路，再用 Skills 和 DevTools 加速工程化。**

企业 Agent 的工程链路

## Agently 的卖点不是“又一个 Agent 框架”，而是补齐上线缺口

             企业 AI 项目常卡在四个地方：结果不能稳定进入系统，工具调用没有边界，长流程难以追踪，上线后无法观测和评测。Agently 把这些问题拆成可验证能力，团队可以按交付风险逐层试用。           

**结构化契约

### 把一句话变成下游能用的数据

output schema、result facade、instant stream 和重试校验，让字段能直接进入 UI、工单、CRM、审计或后续流程。

**受控行动

### 让 Agent 调系统，但不越权

Actions、Function Calling、MCP 和 ExecutionEnvironment 把业务函数、工具、脚本和沙盒执行放进可记录的调用边界。

**事件流程

### 长任务按事件、分支和并行推进

TriggerFlow 与 Dynamic Task 支持串并行、条件分支、等待恢复、运行流和执行快照，复杂过程不再藏在一个黑盒循环里。

**上线运营

### 上线后能观测、评测和迭代

FastAPIHelper、RuntimeEvent、Workspace、DevTools 和 release 更新策略，把 demo 交给产品、运维和质量团队继续使用。

能力栈

## 从一次请求到生产工作流，能力按交付风险分层

             团队可以先验证结构化结果，再接外部工具，再把流程服务化。每一层都有独立文档和可运行样例，适合技术评估，也适合销售前置说明。           

[Request
            稳定输出
            模型结果按字段、类型、必填和流式 patch 交付。](https://agently.cn/docs/requests/output-control.html)[Runtime
            执行力
            函数、MCP、沙盒和业务系统调用都有 owner 和记录。](https://agently.cn/docs/actions/overview.html)[Workflow
            TriggerFlow
            事件驱动、分支并发、暂停恢复和运行流可视化。](https://agently.cn/docs/triggerflow/overview.html)[DAG
            Dynamic Task
            把模型生成或业务提交的任务图验证后执行。](https://agently.cn/docs/dynamic-task/)[Dev Suite
            Agently-DevTools
            把运行、评测、Playground 和日志放进本地控制台。](https://agently.cn/#devtools)[Build Suite
            Agently-Skills
            让 coding agent 用官方路径拆解和生成复杂 AI 应用。](https://agently.cn/#skills)

Agently-Skills · 给 Coding Agent 的官方开发套件

## 一句话，把复杂 AI 应用拆成可交付工程

               Agently-Skills 不是运行时里的 Skills Executor，而是给 Codex、Claude Code、Cursor 等 coding agent 使用的官方指导包。它把 Agently 4.1.3 能力线拆成 6 个可安装 skill，让 coding agent 从业务目标出发，选择正确的请求层、运行时、流程编排、Dynamic Task 或迁移路径。             

[打开 Agently-Skills](https://github.com/AgentEra/Agently-Skills)[查看使用说明](https://agently.cn/docs/development/coding-agents.html)

**当前公开 catalog：6 个 skill

**对齐 Agently 4.1.3 runtime 能力线

**Install app bundle
**让 coding agent 按官方能力边界开工**

```
export AGENT=codex

for skill in \
  agently \
  agently-request \
  agently-runtime \
  agently-dynamic-task \
  agently-triggerflow
do
  npx skills add AgentEra/Agently-Skills \
    --agent "$AGENT" --skill "$skill" -y
done

# migration path
npx skills add AgentEra/Agently-Skills \
  --agent "$AGENT" --skill agently-migration -y
```

**你告诉 coding agent

“做一个售后工单分流 Agent，输出 severity、owner、next_step，并接入内部创建工单函数。”

- 选择 `agently-request` 做结构化输出
- 选择 `agently-runtime` 挂载受控 Actions
- 生成服务入口、测试输入和验证清单

**你告诉 coding agent

“把日报自动化拆成选题、搜索、筛选、摘要、渲染的流程，并暴露 SSE。”

- 选择 `agently-triggerflow` 设计串并行节点
- 用 runtime stream 给前端推阶段进展
- 产出可继续编辑的 Markdown 报告

**你告诉 coding agent

“把现有 LangGraph 工具调用流程迁到 Agently，保留工具边界和运行证据。”

- 选择 `agently-migration` 做分层映射
- 把工具调用落到 Actions / MCP / ExecutionEnvironment
- 补上 DevTools 观测和回归验证路径

Agently-DevTools · 本地观测、评测和 Playground

## 把每次 Agent 运行变成可检查证据

               agently-devtools 是 Agently 的可选 companion package。开发和试运行阶段可以连接 ObservationBridge，把 Runtime Observation、Scenario Evaluations、Playground、Console 和 Logs 放进同一个本地控制台；应用本身可以不依赖它运行。             

- **运行树、graph、Mermaid 和 trace 帮团队定位流程路径。
- **EvaluationBridge 把多轮方案评测和真实 run 关联起来。
- **Playground 用来单独验证模型设置、输出契约和 prompt 变体。

[查看 DevTools 接入](https://agently.cn/docs/observability/devtools.html)[pip install agently-devtools](https://agently.cn/docs/observability/devtools.html)

![Agently DevTools Runtime Observation 真实控制台截图](https://agently.cn/assets/showcase/devtools-runtime-observation.png)Runtime Observation 中的真实运行记录、作用域、连接状态和检索入口。
`pip install -U agently agently-devtools``agently-devtools start``ObservationBridge(Agently).watch(Agently)`

真实交付物

## 不是只说能做什么，而是看见结果长什么样

             这里只展示已经开发并有可视证据的项目。每张图对应一个可继续拆成接口、流程、数据结构和运行证据的场景，帮助业务方判断价值，也帮助开发者判断要接哪一层能力。           

![Daily News Collector 生成的 Markdown 报告渲染截图](https://agently.cn/assets/showcase/daily-news-report.png)
**Daily News Collector · 公开项目

### 从一个主题到可编辑 Markdown 日报

市场、投研、运营和风控团队可以把选题、搜索、筛选、浏览、摘要和报告渲染变成可复核流程。

- TriggerFlow 阶段编排
- 结构化栏目字段
- 报告产物可继续编辑

![Talk to Control 自然语言控制界面截图](https://agently.cn/assets/showcase/talk-to-control.png)
**Talk to Control · 公开项目

### 自然语言操作业务对象

用户说目标，系统先读取状态、生成动作计划，再执行或拒绝。适合后台、设备、运营和低代码控制台。

- Actions 限定可调用能力
- schema 约束参数
- 运行流反馈给前端

![AgentlyTextParser 长文抽取结果界面截图](https://agently.cn/assets/showcase/agently-text-parser-results.png)
**AgentlyTextParser · 本地验证项目

### 长文抽取、字段分组和引用定位

合同、报告、访谈和知识库资料可以进入可导出的 JSON 或表单，而不是停留在一段摘要里。

- 长文分段处理
- 规则驱动 schema
- 证据定位与进度展示

![EDA Agent 专业工作台运行界面截图](https://agently.cn/assets/showcase/eda-plugin-run.png)
**EDA Agent Final · 本地完成原型

### 专业需求推进到工具执行

模型生成结构化方案，宿主系统做确定性校验、替代方案选择和工具 API 执行，适合专家工作台类场景。

- topology / netlist 计划
- 确定性 validate gate
- 服务接口接工具 API

仓库 examples · 可验证案例

## 用实际运行结果说明客户价值

             这些不是官网临时编的 demo，而是 Agently 主仓库 examples 中带有稳定关键输出的案例。你可以从结果反推业务价值：生成了什么、谁能用、Agently 补齐了哪段工程链路。           

**客户支持自动化

### 重复扣费工单生成可发送回复，并通过规则校验

[查看完整案例代码](https://github.com/AgentEra/Agently/blob/main/examples/agent_task/support_ticket_policy_reply.py)

```
task_status="completed"
accepted=true
output_file_exists=true
model_judge_passed=true
replan_count=0
```

适合客服、账单和售后团队：业务系统提供票据、发票和政策事实，Agent 生成 Markdown 回复，再由 judge 校验不虚构、不越权承诺。

**事故沟通

### 从事故流水和状态系统生成客户成功 briefing

[查看完整案例代码](https://github.com/AgentEra/Agently/blob/main/examples/dynamic_task/04_incident_briefing_auto_plan.py)

```
summarize_facts
-> assess_customer_impact
validated TaskDAG
planned_task_count=3
semantic_role=customer_success_briefing
write_customer_success_briefing
frontstage_next_update=when duplicate payment verification completes
status_banner="Latency status: resolved. Duplicate payment checks: running."
```

适合客户成功和运维协同：模型规划 DAG，业务状态系统提供事实，输出可直接交给 account owner 的对外沟通材料。

**企业续约风险

### 把销售、产品、支持和法务信号合成恢复方案

[查看完整案例代码](https://github.com/AgentEra/Agently/blob/main/examples/dynamic_task/05_enterprise_renewal_complex_auto_plan.py)

```
planned_task_count=6
root_task_count=3
join_task_count=1
semantic_role=recovery_package
next_actions_count=7
```

适合大客户经营：CRM、产品使用、P1 工单、折扣诉求和法务问题进入并行分析，再汇总成客户消息和内部行动清单。

**受控执行

### MCP 与沙盒让模型行动可记录、可复核

[MCP 案例](https://github.com/AgentEra/Agently/blob/main/examples/action_runtime/2_2_mcp_http_action_deepseek.py)[沙盒案例](https://github.com/AgentEra/Agently/blob/main/examples/action_runtime/3_1_python_sandbox_action_deepseek.py)

```
MCP calculator -> 168.21
Python sandbox -> average=20.0, gap=34
ActionResult -> model_digest + artifact_refs
```

适合需要工具调用的系统：模型决定调用路径，MCP 和沙盒负责确定性执行，ActionResult 留下可追踪证据。

合作、授权与招聘

## 让业务方、技术方和团队候选人都有明确入口

             先用公开文档、项目证据和本地试用验证技术路径；需要商用授权、商标/认证说明或加入团队时，可以直接走对应入口。           

![PerfXLab](https://agently.cn/assets/customers/perfxlab.jpg)

### PerfXCloud x Agently

围绕云服务、模型应用运行时和企业 AI 工程落地的公开实践。

### 开发者社区

讨论业务系统接入、工具治理、TriggerFlow 编排、服务化交付和 DevTools 观测。

[加入微信群](https://doc.weixin.qq.com/forms/AIoA8gcHAFMAScAhgZQABIlW6tV3l7QQf)

### 授权说明

Agently 开源核心采用 Apache 2.0。商标、官方认证、官方合作伙伴或背书类表述不随开源许可自动授权；商用授权可联系 business@agently.tech。

[申请商用授权](mailto:business@agently.tech)[查看授权与认证声明](https://agently.cn/certificate-statement.html)

### 加入我们

当前保留全栈应用开发工程师、全栈产品经理、技术运营&商务负责人等岗位入口。简历发送至 hr@agently.tech。

全栈应用开发

全栈产品经理

技术运营&商务

[投递简历](mailto:hr@agently.tech)

开始验证

## 给技术团队的最短路径

             先跑一次结构化请求，再接执行力、流程编排、服务化和观测。每一步都对应文档，不需要先读完整框架史。           

[快速开始](https://agently.cn/docs/start/quickstart.html)[结构化输出](https://agently.cn/docs/requests/output-control.html)[执行力](https://agently.cn/docs/actions/overview.html)[TriggerFlow](https://agently.cn/docs/triggerflow/overview.html)[DevTools](https://agently.cn/docs/observability/devtools.html)[Agently-Skills](https://agently.cn/docs/development/coding-agents.html)[更新机制](https://agently.cn/docs/development/release-workflows.html)

企业交付视角的框架分层

## 企业 Agent 要交付的不是回答，而是可治理的行动闭环

             模型会回答只是起点。客户真正需要的是：自然语言入口能理解目标，系统行为能受控推进，长任务证据能被找回，运行过程能被产品和运维看见。Agently 把这些工程边界拆成可验证的能力层，减少团队在原型到上线之间反复补胶水。           

**闭环 01

### 把自然语言目标变成业务契约

用 output schema、ensure keys、result facade 把意图、计划、风险、待办和回复收束成字段、类型、必填项、流式事件和可复用结果。

**闭环 02

### 把外部行动放进受控运行时

把搜索、浏览、MCP、业务函数、脚本、沙箱和控制器放进 Actions 与 ExecutionEnvironment，调用、权限、审批和日志都有明确 owner。

**闭环 03

### 把动态路径变成可观察流程

TriggerFlow 让计划、执行、回填、分支、等待、恢复、子流程协作和 runtime stream 可见，复杂任务不再藏在黑盒循环里。

**闭环 04

### 让状态和证据能持续运营

Workspace、Recall、FastAPIHelper、产品过程事件、RuntimeEvent、DevTools 和更新策略，让 Agent 能服务化、被观察、可恢复、可持续迭代。

落地判断

             先让交互层看到目标和过程，再让系统行为层有结构化决策与受控行动，最后把数据层沉淀成证据、产物、检查点和可召回上下文。这样业务方看到的是可交付结果，开发者维护的是清楚的 owner 边界。           

[按系统路线规划 Agent 架构](https://agently.cn/docs/start/project-framework.html)

从业务目标进入，而不是从框架名词进入

## 你要交付什么，Agently 就帮你补齐哪条工程链路

             这些场景来自公开案例仓库、本地验证项目和已完成原型。每个方向都能继续拆成接口、流程、数据结构和运行证据，帮助业务方判断投入是否值得，也帮助开发者知道从哪一层开始实现。           

自动情报业务控制结构化抽取专业工作台

行业情报与日报自动化

### 从一个主题开始，交付一份可以继续编辑的 Markdown 报告

                   适合市场、投研、运营、风控和行业研究团队。Agently 可以把选题、栏目规划、网页搜索、候选筛选、正文浏览、摘要生成、栏目写作和报告渲染串成可运行工作流。                 

- TriggerFlow 管住 outline、search、pick、browse、summarize、render 的阶段关系。
- Sub-flow 和并发让多个栏目、多个候选新闻可以分批处理。
- 结构化输出让标题、链接、摘要、推荐理由和栏目导语进入固定报告模板。

[看日报自动化怎么选能力](https://agently.cn/docs/playbooks/overview.html)

**topic -> outline -> search -> pick -> browse -> summarize -> report**

日报标题

栏目导语、入选新闻、摘要、链接和推荐理由被统一渲染。

******

自然语言操作业务对象

### 让用户说一句话，系统先判断、再规划、再执行或拒绝

                   适合设备控制、运营后台、监控系统、低代码工具和内部业务系统。Agently 可以先读取当前状态，再决定直接回答、拆解动作，或给出拒绝说明。                 

- Output schema 让模型返回可检查的动作计划、控制器名和参数。
- Actions 把控制器函数、状态读取和状态写回限制在显式注册能力内。
- TriggerFlow runtime stream 可以把规划、执行和反馈过程展示在聊天界面里。

[看自然语言控制怎么选能力](https://agently.cn/docs/playbooks/overview.html)

User

把 camera_a 转到会议室门口，并告诉我当前状态。

Plan

read_status -> control_camera -> update_state -> reply

Result

动作已执行，状态已写回，过程可在界面中追踪。

长文档和实体关系抽取

### 把散乱文本转成字段、引用、实体和关系数据

                   适合知识库建设、合同/报告整理、客户访谈分析、规则表单提取和知识图谱补充。Agently 可以把长文切块，按 schema 抽取，再合并为可导出的 JSON 或表单结果。                 

- AgentlyTextParser 验证了 1 万字以上文本、分段处理、进度展示、引用定位和映射证据。
- AgentlyTextParser 的界面截图展示了抽取结果、字段分组、原文引用和定位线索。
- 配置驱动的 prompt、抽取规则和模型设置，让业务规则不硬编码在脚本里。

[看文档抽取怎么选能力](https://agently.cn/docs/playbooks/overview.html)

`{   "entities": [{"name": "ACME", "type": "Company"}],   "relations": [{"source": "张三", "type": "works_at"}],   "evidence": ["原文引用", "定位线索"] }`

专业工具链助手

### 把自然语言需求推进到专业软件里的可执行动作

                   适合 EDA、数据分析、质检、运维、工业流程等专业工具场景。已完成的 EDA Agent 原型验证了这条链路：模型生成结构化方案，宿主系统做确定性校验、替代方案选择和工具 API 执行。                 

- EDA 示例从电路需求生成 topology、structured netlist、校验报告和中文设计摘要。
- 确定性 validate gate 检查短路、悬空连接、电源/GND 等硬规则，失败后带错误上下文重试。
- FastAPI 后端和扩展面板把规划、替代元件、执行和总结拆成可运营接口。

[看专业工作台怎么选能力](https://agently.cn/docs/playbooks/overview.html)

Topology

Plan Schema

Validate

Substitute

EDA API

Summary

能力栈

## 从一次请求到生产工作流，Agently 提供六层可组合能力

             业务团队看到的是可交付场景，技术团队需要看到可验证路径。Agently 的文档按企业 Agent 从原型到上线会遇到的工程问题组织，便于逐层验收。           

[01
            AgentExecution
            把一次模型请求变成隔离 draft，统一输入、指令、输出 schema、结果读取和流式消费。](https://agently.cn/docs/start/quickstart.html)[02
            结构化输出
            让模型结果按字段、类型、必填规则和 retry 逻辑交付，减少 JSON 解析和人工兜底。](https://agently.cn/docs/requests/output-control.html)[03
            Actions 与 MCP
            把函数、工具、MCP、浏览、搜索和执行环境挂到明确调用边界，运行过程可记录。](https://agently.cn/docs/actions/overview.html)[04
            Workspace / Recall
            保存观察、产物、检查点和任务证据，让后续轮次按目标召回上下文。](https://agently.cn/docs/requests/workspace.html)[05
            TriggerFlow
            表达分支、并发、sub-flow、多角色协作、事件流、暂停恢复、持久化和执行结果生命周期。](https://agently.cn/docs/triggerflow/overview.html)[06
            服务化与观测
            用 FastAPIHelper、过程事件、主动任务、RuntimeEvent、Event Center 和 DevTools 把 AI 能力交给产品和运维团队。](https://agently.cn/docs/observability/overview.html)

截图与关键效果

## 不只告诉用户能做什么，也让用户看见交付物长什么样

             截图区只放有信息承载的交付物：报告产出、操作界面、抽取结果和专业工具链运行界面。每张图都对应真实项目证据，并说明它解决的客户问题和 Agently 在其中承担的工程职责。           

![Daily News Collector 生成的 Markdown 报告渲染截图](https://agently.cn/assets/showcase/daily-news-report.png)
Daily News Collector · 已开发公开项目
**从主题到可编辑日报报告**

**客户价值**：让市场、投研、运营团队把每日情报收集从手工搜索变成可复核的报告产物。

**Agently 作用**：TriggerFlow 编排选题、搜索、筛选、浏览、摘要和渲染；结构化输出保证标题、链接、摘要和推荐理由进入固定模板。

![Talk to Control 自然语言控制界面截图](https://agently.cn/assets/showcase/talk-to-control.png)
Talk to Control · 已开发公开项目
**自然语言触发状态读取、动作规划、执行或拒绝**

**客户价值**：把设备、后台或运营系统变成“能对话也能行动”的操作入口，同时保留拒绝越权请求的边界。

**Agently 作用**：Actions 限定可调用控制器，schema 约束动作参数，TriggerFlow stream 把规划与执行过程展示给用户。

![AgentlyTextParser 长文抽取结果界面截图](https://agently.cn/assets/showcase/agently-text-parser-results.png)
AgentlyTextParser · 已开发本地验证项目
**长文分段、进度追踪、结果分组和引用定位**

**客户价值**：让合同、报告、访谈和知识库资料进入可导出的表单或 JSON，而不是停留在摘要文本。

**Agently 作用**：按规则拆分长文，逐段抽取字段、原文引用、定位线索和合并结果，降低人工核验成本。

![EDA Agent 专业工作台运行界面截图](https://agently.cn/assets/showcase/eda-plugin-run.png)
EDA Agent Final · 已完成原型
**专业需求到工具执行的多阶段链路**

**客户价值**：把专家工具的操作入口前移到自然语言需求，同时让校验、替代方案和执行摘要留在系统里。

**Agently 作用**：模型负责生成 topology 和 draft plan，确定性 gate 负责硬规则校验，服务接口和工具 API 负责执行。

案例证据

## 这些方向都有可追溯的项目证据

             首页保留公开案例、本地验证项目和完成原型。每个方向都能追溯到运行界面、产出文件、仓库文档或本地项目代码。           

公开示例

### Daily News Collector

主题输入、多栏目规划、搜索筛选、正文摘要、报告渲染，展示 TriggerFlow 和结构化输出如何服务内容生产。

公开示例

### Talk to Control

自然语言、设备状态、动作计划、控制器调用、状态写回和聊天流，展示 AI 如何安全接入操作型系统。

本地验证

### AgentlyTextParser

长文本切块、引用定位、字段分组、配置驱动 schema 和可视化结果，展示文档和知识工程的落地方式。

本地验证

### EDA Agent Final

专业需求、结构化计划、确定性校验、替代元件、真实工具 API 和总结，展示专家工作台的交付链。

[可评估
          文档按当前版本事实组织](https://agently.cn/docs/development/release-notes-4.1.3.6.html)[可追踪
          Release notes、示例和兼容清单联动](https://agently.cn/docs/development/release-workflows.html)[可维护
          官网通过更新策略防止持续掉队](https://agently.cn/docs/development/release-workflows.html)

评估与合作

## 让业务方、技术方和社区一起验证

             你可以先用公开文档和示例做技术验证，再把具体业务场景带到社区讨论。官网只展示已经公开的生态入口。           

![PerfXLab](https://agently.cn/assets/customers/perfxlab.jpg)

### PerfXCloud x Agently

围绕云服务、模型应用运行时和工程落地的公开实践。

### 开发者社区

加入微信社群，讨论企业 AI 应用落地、业务系统接入、流程编排和服务化交付。

[加入微信群](https://doc.weixin.qq.com/forms/AIoA8gcHAFMAScAhgZQABIlW6tV3l7QQf)

文档组织

## 给技术团队的验证路径

             先跑通一次结构化请求，再接 Actions、Workspace、TriggerFlow、服务接口和观测。每一层都有对应文档，不需要先读完整框架史。           

[工程评估](https://agently.cn/docs/reference/capability-map.html)[系统路线](https://agently.cn/docs/start/project-framework.html)[场景选型](https://agently.cn/docs/playbooks/overview.html)[快速开始](https://agently.cn/docs/start/quickstart.html)[能力地图](https://agently.cn/docs/reference/capability-map.html)[稳定输出](https://agently.cn/docs/requests/output-control.html)[Actions](https://agently.cn/docs/actions/overview.html)[Workspace](https://agently.cn/docs/requests/workspace.html)[多角色协作](https://agently.cn/docs/triggerflow/sub-flow.html)[交互层与主动任务](https://agently.cn/docs/services/fastapi.html)[长程任务](https://agently.cn/docs/triggerflow/pause-and-resume.html)[TriggerFlow](https://agently.cn/docs/triggerflow/overview.html)[工具治理](https://agently.cn/docs/actions/mcp.html)[生产治理](https://agently.cn/docs/observability/overview.html)[Dynamic Task](https://agently.cn/docs/dynamic-task/)
