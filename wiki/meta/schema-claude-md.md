# 模式摘要：CLAUDE.md（代理说明）

**元页面** — 概括本仓库对代理的约定；**不是**从 `raw/` 摄入的来源正文。

**权威文件位置**：[`CLAUDE.md`](../../CLAUDE.md)（仓库根目录；不在 `raw/` 下）。

**内容概要**：面向 LLM 代理的说明 — 三层架构（`raw/` / `wiki/` / 模式文件）、目录布局、全局规则（不改 `raw/`、`wiki/log.md` 仅追加），以及 **摄入**、**查询**、**Lint** 工作流。

**要点**

- 知识应在 **wiki 中**通过链接与综合**持续积累**，而非只留在聊天或临时 RAG。
- `raw/code/` 中的代码是**来源材料**；代理只读并充实 `wiki/`，**不修改** `raw/`。
- `wiki/index.md` 为内容目录；`wiki/log.md` 仅追加，条目标题为 `## [YYYY-MM-DD] ingest | …` 等形式。
- **`wiki/meta/`** 存放工作区/模式类说明；**领域**知识主要在 `concepts/`、`sources/` 等目录。

**与本 wiki 互联的页面**：[三层架构](three-layer-architecture.md)、[Wiki 运维](wiki-operations.md)、[仓库结构示意](repository-and-wiki-structure.md)、[Agent AI 与多模态智能体](../concepts/agent-ai-and-multimodal-agents.md)（领域，来自 `raw/`）。
