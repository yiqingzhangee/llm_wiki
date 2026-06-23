# Meta（元数据）— 与领域知识的区分

## 排查结论

最初在 `wiki/concepts/`、`wiki/maps/`、`wiki/comparisons/`、`wiki/entities/`、`wiki/sources/` 下生成的部分页面，内容是 **如何维护本仓库**（`CLAUDE.md` 三层架构、ingest/query/lint、`raw` vs `wiki` 分工、目录树），**不是**从 `raw/` 摄入的 **主题知识**。

这是 **按模式说明做的脚手架**：合法且有用，但应和 **Agent AI、LangGraph、GStack** 等 **领域页** 分开存放与检索，避免误以为「知识库正文」只有这些元页面。

## 本目录放什么

| 适合放在 `wiki/meta/` | 适合放在 `wiki/concepts/`、`sources/` 等 |
|------------------------|-------------------------------------------|
| 代理工作流、目录结构、`CLAUDE.md` 摘要 | 从论文、文章、`raw/code/` 提炼的概念与来源摘要 |
| 「本仓库是什么」的说明 | 可对外复述的学科或产品知识 |

## 本目录现有页面

见 [Wiki index](../index.md) 中的 **「Meta / 运维」** 区块。
