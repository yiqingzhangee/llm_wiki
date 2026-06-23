# 三层架构

工作区分为 **不可变的原始资料**、**可编辑的 wiki**，以及约束代理行为的 **模式说明（schema）**。

| 层级 | 路径 | 作用 |
|------|------|------|
| 原始资料 | `raw/` | 经策展的真相：文章、论文、转写、数据、图像，以及作来源材料的 `raw/code/`。代理**不得**编辑此目录树。 |
| Wiki | `wiki/` | 领域内容：`wiki/entities/`、`wiki/concepts/`、`wiki/sources/`、`wiki/comparisons/`、`wiki/maps/`。运维说明：[`wiki/meta/`](README.md)。 |
| 模式说明 | 仓库根目录 `CLAUDE.md` | 结构、边界、摄入/查询/Lint 工作流。 |

**分工**：人类维护 `raw/` 并提问；代理维护 `wiki/`，做好交叉链接并显式标注矛盾。

另见：[Wiki 运维](wiki-operations.md)、[CLAUDE.md 摘要](schema-claude-md.md)、根目录 [`CLAUDE.md`](../../CLAUDE.md)。
