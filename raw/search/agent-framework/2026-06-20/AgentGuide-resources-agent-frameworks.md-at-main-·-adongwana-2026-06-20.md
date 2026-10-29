# AgentGuide/resources/agent/frameworks.md at main · adongwanai/AgentGuide · GitHub

- 来源链接: https://github.com/adongwanai/AgentGuide/blob/main/resources/agent/frameworks.md
- 搜索日期: 2026-06-20

## 详细内容

## AgentGuide/resources/agent/frameworks.md at main · adongwanai/AgentGuide · GitHub

**Source**: https://github.com/adongwanai/AgentGuide/blob/main/resources/agent/frameworks.md

---

[Skip to content](/charset=utf-8#start-of-content)

You signed in with another tab or window. Reload to refresh your session.

You signed out in another tab or window. Reload to refresh your session.

You switched accounts on another tab or window. Reload to refresh your session.
Dismiss alert

{{ message }}

[adongwanai](/adongwanai)

/
**[AgentGuide](/adongwanai/AgentGuide)**

Public

- [Notifications](/login?return_to=%2Fadongwanai%2FAgentGuide)You must be signed in to change notification settings
- [Fork
    605](/login?return_to=%2Fadongwanai%2FAgentGuide)
- 
[Star
          6.1k](/login?return_to=%2Fadongwanai%2FAgentGuide)

[/adongwanai/AgentGuide](/adongwanai/AgentGuide)

## 
Expand file tree

/

# frameworks.md

Copy path

More file actions

More file actions

## Latest commit

## History

[History](/adongwanai/AgentGuide/commits/main/resources/agent/frameworks.md)

[/adongwanai/AgentGuide/commits/main/resources/agent/frameworks.md](/adongwanai/AgentGuide/commits/main/resources/agent/frameworks.md)
History

394 lines (269 loc) · 10.8 KB

/

# frameworks.md

Copy path

## File metadata and controls

- 
- 
- 

394 lines (269 loc) · 10.8 KB

[Raw](https://github.com/adongwanai/AgentGuide/raw/refs/heads/main/resources/agent/frameworks.md)

Copy raw file

Download raw file

Outline

Edit and raw actions

# Agent 开发框架对比

[/charset=utf-8#agent-%E5%BC%80%E5%8F%91%E6%A1%86%E6%9E%B6%E5%AF%B9%E6%AF%94](/charset=utf-8#agent-%E5%BC%80%E5%8F%91%E6%A1%86%E6%9E%B6%E5%AF%B9%E6%AF%94)

> **只推荐最核心的 5-7 个框架**（面试会问、项目会用的）

---

## 🔥 核心框架对比表

[/charset=utf-8#-%E6%A0%B8%E5%BF%83%E6%A1%86%E6%9E%B6%E5%AF%B9%E6%AF%94%E8%A1%A8](/charset=utf-8#-%E6%A0%B8%E5%BF%83%E6%A1%86%E6%9E%B6%E5%AF%B9%E6%AF%94%E8%A1%A8)

| 框架 | Stars | 难度 | 适合场景 | 优点 | 缺点 |
|---|---|---|---|---|---|
| LangGraph | 15k+ | ⭐⭐⭐⭐ | 复杂工作流 | 灵活、可视化 | 学习曲线陡 |
| AutoGen | 30k+ | ⭐⭐⭐ | Multi-Agent | 易用、可视化Studio | 定制化难 |
| CrewAI | 20k+ | ⭐⭐ | 角色协作 | 简单、快速上手 | 功能相对简单 |
| LangChain | 90k+ | ⭐⭐⭐ | 快速原型 | 生态完善 | 抽象过度 |
| Swarm | 5k+ | ⭐ | 学习用途 | 极简、清晰 | 功能基础 |
| Parlant | - | ⭐⭐⭐ | 指令遵循 | 高可控、ARQs | 场景受限 |
| AgentScope | 3k+ | ⭐⭐⭐ | 模块化开发 | 易维护、企业级 | 相对新 |

---

## 📖 详细对比

[/charset=utf-8#-%E8%AF%A6%E7%BB%86%E5%AF%B9%E6%AF%94](/charset=utf-8#-%E8%AF%A6%E7%BB%86%E5%AF%B9%E6%AF%94)

### 1. LangGraph ⭐⭐⭐⭐⭐ 最推荐生产环境

[/charset=utf-8#1-langgraph--%E6%9C%80%E6%8E%A8%E8%8D%90%E7%94%9F%E4%BA%A7%E7%8E%AF%E5%A2%83](/charset=utf-8#1-langgraph--%E6%9C%80%E6%8E%A8%E8%8D%90%E7%94%9F%E4%BA%A7%E7%8E%AF%E5%A2%83)

**官网**：[https://github.com/langchain-ai/langgraph](https://github.com/langchain-ai/langgraph)

**核心特点**：

- ✅ 状态机驱动，精确控制流程
- ✅ 可视化工作流，易于调试
- ✅ 支持复杂条件分支和循环
- ✅ 与 LangChain 深度集成

**适合场景**：

- 复杂的 Agent 工作流（10+ 步骤）
- 需要状态管理的应用
- Multi-Agent 协作系统

**学习成本**：⭐⭐⭐⭐（需要 3-5 天深入学习）

**面试加分点**：

```
"我使用 LangGraph 设计了状态机驱动的 Agent 工作流，
通过可视化编排，实现了复杂的条件分支和异常处理"
```

**快速上手**：

```
from langgraph.graph import StateGraph

# 30 行代码搭建 Agent 工作流
```

---

### 2. AutoGen ⭐⭐⭐⭐⭐ Multi-Agent 首选

[/charset=utf-8#2-autogen--multi-agent-%E9%A6%96%E9%80%89](/charset=utf-8#2-autogen--multi-agent-%E9%A6%96%E9%80%89)

**官网**：[https://github.com/microsoft/autogen](https://github.com/microsoft/autogen)

**核心特点**：

- ✅ 天然支持 Multi-Agent
- ✅ AutoGen Studio 可视化编排
- ✅ 支持代码执行环境
- ✅ 微软官方支持

**适合场景**：

- 多智能体协作（3+ Agents）
- 需要可视化编排
- 企业级应用

**学习成本**：⭐⭐⭐（1-2 天快速上手）

**面试加分点**：

```
"使用 AutoGen 构建多智能体协作系统，
实现了分类Agent、查询Agent、执行Agent的协同工作"
```

---

### 3. CrewAI ⭐⭐⭐⭐ 快速原型首选

[/charset=utf-8#3-crewai--%E5%BF%AB%E9%80%9F%E5%8E%9F%E5%9E%8B%E9%A6%96%E9%80%89](/charset=utf-8#3-crewai--%E5%BF%AB%E9%80%9F%E5%8E%9F%E5%9E%8B%E9%A6%96%E9%80%89)

**官网**：[https://github.com/joaomdmoura/crewAI](https://github.com/joaomdmoura/crewAI)

**核心特点**：

- ✅ 角色定义清晰
- ✅ API 简单易用
- ✅ 快速开发（1小时搭建原型）

**适合场景**：

- 快速验证想法
- 角色明确的协作场景
- 小规模应用

**学习成本**：⭐⭐（半天上手）

---

### 4. LangChain ⭐⭐⭐⭐⭐ 必学基础

[/charset=utf-8#4-langchain--%E5%BF%85%E5%AD%A6%E5%9F%BA%E7%A1%80](/charset=utf-8#4-langchain--%E5%BF%85%E5%AD%A6%E5%9F%BA%E7%A1%80)

**官网**：[https://github.com/langchain-ai/langchain](https://github.com/langchain-ai/langchain)

**核心特点**：

- ✅ 生态最完善
- ✅ 文档最详细
- ✅ 社区最活跃

**适合场景**：

- 快速原型开发
- 学习 Agent 基础概念
- 与其他工具集成

**学习成本**：⭐⭐⭐（2-3 天基础掌握）

**注意事项**：

- ❌ 抽象过度，调试困难
- ⚠️ 建议：理解原理，必要时魔改

---

### 5. Swarm ⭐⭐⭐ 学习用途

[/charset=utf-8#5-swarm--%E5%AD%A6%E4%B9%A0%E7%94%A8%E9%80%94](/charset=utf-8#5-swarm--%E5%AD%A6%E4%B9%A0%E7%94%A8%E9%80%94)

**官网**：[https://github.com/openai/swarm](https://github.com/openai/swarm)

**核心特点**：

- ✅ OpenAI 官方出品
- ✅ 代码极简（<500行）
- ✅ 适合理解 Agent 原理

**适合场景**：

- 学习 Agent 设计思想
- 理解多智能体协作
- 不建议用于生产

**学习成本**：⭐（1-2小时读完源码）

---

### 6. Parlant ⭐⭐⭐⭐ 指令遵循专家

[/charset=utf-8#6-parlant--%E6%8C%87%E4%BB%A4%E9%81%B5%E5%BE%AA%E4%B8%93%E5%AE%B6](/charset=utf-8#6-parlant--%E6%8C%87%E4%BB%A4%E9%81%B5%E5%BE%AA%E4%B8%93%E5%AE%B6)

**官网**：[https://github.com/emcie-co/parlant](https://github.com/emcie-co/parlant)

**核心特点**：

- ✅ **确保指令遵循**：不再依赖 LLM"可能"遵循指令，而是"确保"必定遵循
- ✅ **ARQs 技术**：Attentive Reasoning Queries（注意力推理查询）
- ✅ **领域指导优势**：通过预定义推理步骤，针对性预防常见失效模式
- ✅ **高成功率**：在 87 个测试场景中，总成功率达 90.2%（高于 CoT 的 86.05%）

**核心技术：ARQs（Attentive Reasoning Queries）**

ARQs 是一种结构化推理方法，包含三个阶段：

1. 

**引导 ARQ 阶段**

  - LLM 处理预定义引导问题
  - 恢复关键指令
  - 恢复提示中的重要上下文信息
  - 促进分步推理与中间计算

2. 

**响应生成阶段**

  - 基于引导 ARQ 阶段的推理结果
  - 生成最终响应

3. 

**响应验证阶段**（可选）

  - LLM 通过预定义验证问题评估响应
  - 不满足则重新生成并验证
  - 直至符合标准

**适合场景**：

- ✅ 有明确领域指南的场景（客户服务、医疗咨询）
- ✅ 失效模式清晰的场景
- ✅ 需要高度可控的 Agent 系统
- ✅ 高风险客户应用

**学习成本**：⭐⭐⭐（2-3 天理解核心概念）

**面试加分点**：

```
"使用 Parlant 框架确保 Agent 的指令遵循，
通过 ARQs 技术实现了领域专用推理步骤，
在客户服务场景中达到 90%+ 的成功率"
```

**关键资源**：

- [Parlant vs LangGraph](https://www.parlant.io/blog/parlant-vs-langgraph)
- [Parlant vs DSPy](https://www.parlant.io/blog/parlant-vs-dspy)
- [How Parlant Ensures Compliance](https://www.parlant.io/blog/how-parlant-guarantees-compliance)
- [ARQs 论文](https://arxiv.org/pdf/2503.03669v1)
- [ARQs 代码](https://github.com/emcie-co/parlant/tree/arqs-a-systematic-method-for-optimizing-instruction-following-in-llms)

**vs LangGraph**：

- Parlant 强调**确定性控制**，LangGraph 强调**灵活编排**
- Parlant 适合**规则明确**的场景，LangGraph 适合**复杂工作流**

**vs DSPy**：

- Parlant 通过**预定义推理步骤**确保遵循，DSPy 通过**优化提示**提升性能
- Parlant 更适合**生产环境**的可控性需求

---

### 7. AgentScope ⭐⭐⭐⭐ 模块化开发框架

[/charset=utf-8#7-agentscope--%E6%A8%A1%E5%9D%97%E5%8C%96%E5%BC%80%E5%8F%91%E6%A1%86%E6%9E%B6](/charset=utf-8#7-agentscope--%E6%A8%A1%E5%9D%97%E5%8C%96%E5%BC%80%E5%8F%91%E6%A1%86%E6%9E%B6)

**官网**：[https://github.com/agentscope-ai/agentscope](https://github.com/agentscope-ai/agentscope)

**核心特点**：

- ✅ **模块化设计**：将 Agent 功能模块化，易于维护和扩展
- ✅ **灵活集成**：与各种外部工具和服务无缝集成
- ✅ **开发者友好**：详细的文档和示例
- ✅ **企业级支持**：阿里云等机构背书

**适合场景**：

- 模块化 Agent 系统开发
- 需要灵活集成的应用
- 企业级 Agent 应用
- 快速原型到生产的迁移

**学习成本**：⭐⭐⭐（2-3 天掌握基础）

**面试加分点**：

```
"使用 AgentScope 构建模块化 Agent 系统，
通过清晰的模块划分和灵活的集成能力，
实现了从原型到生产的快速迁移"
```

---

## 🎯 选择建议

[/charset=utf-8#-%E9%80%89%E6%8B%A9%E5%BB%BA%E8%AE%AE](/charset=utf-8#-%E9%80%89%E6%8B%A9%E5%BB%BA%E8%AE%AE)

### 按场景选择

[/charset=utf-8#%E6%8C%89%E5%9C%BA%E6%99%AF%E9%80%89%E6%8B%A9](/charset=utf-8#%E6%8C%89%E5%9C%BA%E6%99%AF%E9%80%89%E6%8B%A9)

| 你的需求 | 推荐框架 | 原因 |
|---|---|---|
| 学习 Agent 原理 | Swarm | 代码简单，易于理解 |
| 快速做 Demo | CrewAI | 上手快，API 简单 |
| 做复杂工作流 | LangGraph | 状态管理，灵活控制 |
| 做 Multi-Agent | AutoGen | 天然支持，可视化 |
| 通用开发 | LangChain | 生态完善，工具多 |
| 高可控场景 | Parlant | 确保指令遵循，适合客服/医疗 |
| 模块化系统 | AgentScope | 易维护，企业级支持 |

### 学习顺序建议

[/charset=utf-8#%E5%AD%A6%E4%B9%A0%E9%A1%BA%E5%BA%8F%E5%BB%BA%E8%AE%AE](/charset=utf-8#%E5%AD%A6%E4%B9%A0%E9%A1%BA%E5%BA%8F%E5%BB%BA%E8%AE%AE)

```
Step 1: Swarm (理解原理，1天)
  ↓
Step 2: LangChain (掌握基础，3天)
  ↓
Step 3: LangGraph 或 AutoGen (深入一个，5天)
  ↓
Step 4: 其他框架按需学习
```

---

## 📚 官方学习指南（必读）

[/charset=utf-8#-%E5%AE%98%E6%96%B9%E5%AD%A6%E4%B9%A0%E6%8C%87%E5%8D%97%E5%BF%85%E8%AF%BB](/charset=utf-8#-%E5%AE%98%E6%96%B9%E5%AD%A6%E4%B9%A0%E6%8C%87%E5%8D%97%E5%BF%85%E8%AF%BB)

### 1. Anthropic 构建有效智能体指南 ⭐⭐⭐⭐⭐ 强烈推荐

[/charset=utf-8#1-anthropic-%E6%9E%84%E5%BB%BA%E6%9C%89%E6%95%88%E6%99%BA%E8%83%BD%E4%BD%93%E6%8C%87%E5%8D%97--%E5%BC%BA%E7%83%88%E6%8E%A8%E8%8D%90](/charset=utf-8#1-anthropic-%E6%9E%84%E5%BB%BA%E6%9C%89%E6%95%88%E6%99%BA%E8%83%BD%E4%BD%93%E6%8C%87%E5%8D%97--%E5%BC%BA%E7%83%88%E6%8E%A8%E8%8D%90)

**链接**：[https://www.anthropic.com/engineering/building-effective-agents](https://www.anthropic.com/engineering/building-effective-agents)

**内容**：Anthropic 团队总结的 Agent 构建最佳实践

**核心观点**：

- Agent 不是万能的，要知道什么时候用、什么时候不用
- **Workflow > Agents**：能用工作流解决就不要用 Agent
- 简单的 Prompt 优化往往比复杂的 Agent 更有效

**必读原因**：Claude 团队的实战经验，帮助你避免过度设计

---

### 2. OpenAI 实用智能体构建指南 ⭐⭐⭐⭐

[/charset=utf-8#2-openai-%E5%AE%9E%E7%94%A8%E6%99%BA%E8%83%BD%E4%BD%93%E6%9E%84%E5%BB%BA%E6%8C%87%E5%8D%97-](/charset=utf-8#2-openai-%E5%AE%9E%E7%94%A8%E6%99%BA%E8%83%BD%E4%BD%93%E6%9E%84%E5%BB%BA%E6%8C%87%E5%8D%97-)

**链接**：[https://cdn.openai.com/business-guides-and-resources/a-practical-guide-to-building-agents.pdf](https://cdn.openai.com/business-guides-and-resources/a-practical-guide-to-building-agents.pdf)

**内容**：OpenAI 官方的 Agent 构建指南

**适合人群**：想要了解 OpenAI 视角下的 Agent 设计

---

### 3. Agentic Design Patterns（中英对照）⭐⭐⭐⭐⭐ 强烈推荐

[/charset=utf-8#3-agentic-design-patterns%E4%B8%AD%E8%8B%B1%E5%AF%B9%E7%85%A7-%E5%BC%BA%E7%83%88%E6%8E%A8%E8%8D%90](/charset=utf-8#3-agentic-design-patterns%E4%B8%AD%E8%8B%B1%E5%AF%B9%E7%85%A7-%E5%BC%BA%E7%83%88%E6%8E%A8%E8%8D%90)

**链接**：[https://github.com/ginobefun/agentic-design-patterns-cn](https://github.com/ginobefun/agentic-design-patterns-cn)

**内容**：《Agentic Design Patterns: A Hands-On Guide to Building Intelligent Systems》中英文对照译本

**适合人群**：

- AI 工程师
- 软件架构师
- 产品经理
- 对智能系统设计感兴趣的研究者

**核心内容**：

- Agent 设计模式
- 实践流程
- 系统架构

---

### 4. Anthropic 提示工程指南 ⭐⭐⭐⭐

[/charset=utf-8#4-anthropic-%E6%8F%90%E7%A4%BA%E5%B7%A5%E7%A8%8B%E6%8C%87%E5%8D%97-](/charset=utf-8#4-anthropic-%E6%8F%90%E7%A4%BA%E5%B7%A5%E7%A8%8B%E6%8C%87%E5%8D%97-)

**链接**：[https://docs.anthropic.com/en/docs/build-with-claude/prompt-engineering/overview](https://docs.anthropic.com/en/docs/build-with-claude/prompt-engineering/overview)

**内容**：系统化的 Prompt Engineering 教程

**交互式教程**：[https://github.com/anthropics/prompt-eng-interactive-tutorial](https://github.com/anthropics/prompt-eng-interactive-tutorial)

---

### 5. Google Gemini 提示指南 ⭐⭐⭐

[/charset=utf-8#5-google-gemini-%E6%8F%90%E7%A4%BA%E6%8C%87%E5%8D%97-](/charset=utf-8#5-google-gemini-%E6%8F%90%E7%A4%BA%E6%8C%87%E5%8D%97-)

**链接**：[https://services.google.com/fh/files/misc/gemini-for-google-workspace-prompting-guide-101.pdf](https://services.google.com/fh/files/misc/gemini-for-google-workspace-prompting-guide-101.pdf)

**内容**：Google 官方的 Prompt Engineering 指南

---

## 🎯 学习建议

[/charset=utf-8#-%E5%AD%A6%E4%B9%A0%E5%BB%BA%E8%AE%AE](/charset=utf-8#-%E5%AD%A6%E4%B9%A0%E5%BB%BA%E8%AE%AE)

### 1. 先学框架原理，再学框架使用

[/charset=utf-8#1-%E5%85%88%E5%AD%A6%E6%A1%86%E6%9E%B6%E5%8E%9F%E7%90%86%E5%86%8D%E5%AD%A6%E6%A1%86%E6%9E%B6%E4%BD%BF%E7%94%A8](/charset=utf-8#1-%E5%85%88%E5%AD%A6%E6%A1%86%E6%9E%B6%E5%8E%9F%E7%90%86%E5%86%8D%E5%AD%A6%E6%A1%86%E6%9E%B6%E4%BD%BF%E7%94%A8)

- 理解 Agent 的本质（规划 + 工具 + 记忆）
- 手撕简单的 ReAct Agent
- 再去学习框架的封装

### 2. 不要过度依赖框架

[/charset=utf-8#2-%E4%B8%8D%E8%A6%81%E8%BF%87%E5%BA%A6%E4%BE%9D%E8%B5%96%E6%A1%86%E6%9E%B6](/charset=utf-8#2-%E4%B8%8D%E8%A6%81%E8%BF%87%E5%BA%A6%E4%BE%9D%E8%B5%96%E6%A1%86%E6%9E%B6)

- 框架是工具，不是答案
- 遇到问题要能调试源码
- 必要时可以魔改或自己实现

### 3. 多框架对比学习

[/charset=utf-8#3-%E5%A4%9A%E6%A1%86%E6%9E%B6%E5%AF%B9%E6%AF%94%E5%AD%A6%E4%B9%A0](/charset=utf-8#3-%E5%A4%9A%E6%A1%86%E6%9E%B6%E5%AF%B9%E6%AF%94%E5%AD%A6%E4%B9%A0)

- 同一个功能，用不同框架实现
- 理解每个框架的设计哲学
- 面试时能说出框架选择的理由

---

## 💬 面试中的框架选择问题

[/charset=utf-8#-%E9%9D%A2%E8%AF%95%E4%B8%AD%E7%9A%84%E6%A1%86%E6%9E%B6%E9%80%89%E6%8B%A9%E9%97%AE%E9%A2%98](/charset=utf-8#-%E9%9D%A2%E8%AF%95%E4%B8%AD%E7%9A%84%E6%A1%86%E6%9E%B6%E9%80%89%E6%8B%A9%E9%97%AE%E9%A2%98)

### 常见问题："为什么选择 XX 框架？"

[/charset=utf-8#%E5%B8%B8%E8%A7%81%E9%97%AE%E9%A2%98%E4%B8%BA%E4%BB%80%E4%B9%88%E9%80%89%E6%8B%A9-xx-%E6%A1%86%E6%9E%B6](/charset=utf-8#%E5%B8%B8%E8%A7%81%E9%97%AE%E9%A2%98%E4%B8%BA%E4%BB%80%E4%B9%88%E9%80%89%E6%8B%A9-xx-%E6%A1%86%E6%9E%B6)

**回答模板**：

```
我选择 LangGraph 是因为：
1. 【场景需求】项目需要复杂的状态管理和条件分支
2. 【技术优势】LangGraph 的可视化工作流便于调试
3. 【生态支持】与 LangChain 深度集成，工具链完善
4. 【对比分析】相比 AutoGen，LangGraph 在状态控制上更灵活
```

**关键点**：

- ✅ 从场景需求出发
- ✅ 说明技术优势
- ✅ 对比其他方案
- ✅ 体现深入思考

---

## 📝 相关文档

[/charset=utf-8#-%E7%9B%B8%E5%85%B3%E6%96%87%E6%A1%A3](/charset=utf-8#-%E7%9B%B8%E5%85%B3%E6%96%87%E6%A1%A3)

### 本目录其他资源

[/charset=utf-8#%E6%9C%AC%E7%9B%AE%E5%BD%95%E5%85%B6%E4%BB%96%E8%B5%84%E6%BA%90](/charset=utf-8#%E6%9C%AC%E7%9B%AE%E5%BD%95%E5%85%B6%E4%BB%96%E8%B5%84%E6%BA%90)

- [Agent 资源总览](/adongwanai/AgentGuide/blob/main/resources/agent/README.md)
- [Agent Memory 资源](/adongwanai/AgentGuide/blob/main/resources/agent/memory.md)
- [AI Agent 生产环境实践](/adongwanai/AgentGuide/blob/main/resources/agent/ai-agent-production-challenges.md)

### AgentGuide 其他章节

[/charset=utf-8#agentguide-%E5%85%B6%E4%BB%96%E7%AB%A0%E8%8A%82](/charset=utf-8#agentguide-%E5%85%B6%E4%BB%96%E7%AB%A0%E8%8A%82)

- [LangChain 核心概念与实战](/adongwanai/AgentGuide/blob/main/docs/02-tech-stack/04-langchain-guide.md)
- [LlamaIndex 核心概念与实战](/adongwanai/AgentGuide/blob/main/docs/02-tech-stack/05-llamaindex-guide.md)
- [AutoGen/CrewAI 多智能体框架详解](/adongwanai/AgentGuide/blob/main/docs/02-tech-stack/06-multi-agent-frameworks.md)

---

**👉 返回主文档**：[AgentGuide README](/adongwanai/AgentGuide/blob/main/README.md)

---

## 📌 文档维护

[/charset=utf-8#-%E6%96%87%E6%A1%A3%E7%BB%B4%E6%8A%A4](/charset=utf-8#-%E6%96%87%E6%A1%A3%E7%BB%B4%E6%8A%A4)

**最后更新**：2025-11 **贡献者**：欢迎提交 PR 补充新框架

**更新日志**：

- 2025-11：新增 Parlant（指令遵循）和 AgentScope（模块化）框架
- 2025-11：整合官方学习指南
- 2025-11：优化框架对比表和选择建议

     You can’t perform that action at this time.
