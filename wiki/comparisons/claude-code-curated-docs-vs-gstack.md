# Claude Code：整理文档型资料 vs GStack

二者均以 `raw/code/` 下**只读**镜像存在；综合理解写在 `wiki/sources/`。

| | [claude-code-best-practice](../sources/repo-claude-code-best-practice.md) | [gstack](../sources/repo-gstack.md) |
|---|----------------|--------|
| **主要意图** | 对照官方 Claude Code 能力的**概念地图**（子智能体、命令、技能、钩子、MCP、记忆、热点功能等）。 | 偏**安装即用的产品化**方案：大量**斜杠命令**与技能，覆盖日常评审/QA/发布/设计/安全等。 |
| **受众** | 学习如何把功能映射到文档与社区实现的人。 | 创始人、技术负责人、希望标准化「代理辅助」软件工程流程的团队。 |
| **交付形态** | Markdown 指南、徽章、示例、报告等仓库内容。 | `setup` 脚本、全局 `~/.claude/skills/gstack`、团队引导脚本。 |
| **重叠** | 都覆盖 Claude Code 基础构件（命令、技能、智能体）。 | 在假定这些构件之上，叠加**大规模角色化工作流**。 |

二者均**不能替代**本仓库根目录 [`CLAUDE.md`](../../CLAUDE.md)；均为**第三方**材料，供提炼与对比。
