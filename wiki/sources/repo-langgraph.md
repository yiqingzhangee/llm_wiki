# 来源：LangGraph（langchain-ai/langgraph）

**Raw 路径**：[`raw/code/langgraph/`](../../raw/code/langgraph/)（[langchain-ai/langgraph](https://github.com/langchain-ai/langgraph) 的 git clone）

## 一句话摘要

**LangGraph** 是用于构建**长时运行、有状态**智能体与工作流的 **Python 底层编排框架**（姊妹 JS 库：LangGraph.js）。

## 关键主张（据上游 README）

1. **持久执行**：图可在**故障后恢复**，从最近检查点继续。
2. **人在回路**：运行中可检查并**修改状态**（文档中的 interrupts 模式）。
3. **记忆**：短期工作态与面向智能体的长期持久记忆模式（见官方文档）。
4. **可观测 / 部署**：与 **LangSmith** 强绑定（追踪、评测、部署平台、Studio）。
5. **生态**：可独立使用，或与 LangChain、Deep Agents、LangSmith Deployment 等组合；`pip install langgraph`。

## 代码布局（概览）

- `libs/` — Python 包
- `docs/` — 文档源
- `examples/` — 图与智能体示例

## 与 wiki 的衔接

- 实体/工具页：[LangGraph](../entities/langgraph.md)
- 概念关联：[Agent AI 与多模态智能体](../concepts/agent-ai-and-multimodal-agents.md)（综述层 vs 实现框架层，互为补充）
