# 来源：claude-code-best-practice（GitHub）

**Raw 路径**：[`raw/code/claude-code-best-practice/`](../../raw/code/claude-code-best-practice/)（[shanraisshan/claude-code-best-practice](https://github.com/shanraisshan/claude-code-best-practice) 的 git clone）

## 一句话摘要

面向 **Claude Code** 的整理文档与示例：**子智能体**、斜杠命令、技能、钩子、MCP、设置、记忆、工作流及「热点」能力（Agent SDK、Web、定时任务等），以最佳实践与实现说明组织。

## 关键结构 / 主张

1. **子智能体**（`.claude/agents/`）：隔离上下文中的执行者，可配工具、权限、模型与身份。
2. **命令**（`.claude/commands/`）：在**当前**上下文中由用户触发的提示模板。
3. **技能**（`.claude/skills/<name>/SKILL.md`）：可预加载、可发现的过程性知识，渐进展示。
4. **编排模式**：Command → Agent → Skill 流水线（见仓库内 `orchestration-workflow/`）。
5. **钩子**（`.claude/hooks/`）：主智能体循环外、按事件触发的处理逻辑。
6. **MCP / 插件 / 设置**：外接工具、可分发包、分层 `settings.json`。
7. **记忆**：`CLAUDE.md`、`.claude/rules/`、项目记忆路径 — 持久上下文与 `@path` 引用。
8. **生态链接**：官方 Claude Code 文档、Agent SDK、社区 hooks/status-line 仓库、报告类文档（如 SDK 与 CLI 提示对比、进阶工具使用等）。

## 与 wiki 的衔接

- 与 [CLAUDE.md 摘要](../meta/schema-claude-md.md) 互补，说明**本仓库**的代理规则；该仓库为**第三方**实践资料，非本工作区模式正文。
- 与团队工作流对照时，可与 [GStack](repo-gstack.md) 的斜杠命令驱动「软件工厂」作比较。
