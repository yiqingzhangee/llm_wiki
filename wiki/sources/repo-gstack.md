# 来源：gstack（garrytan/gstack）

**Raw 路径**：[`raw/code/gstack/`](../../raw/code/gstack/)（[garrytan/gstack](https://github.com/garrytan/gstack) 的 git clone）

## 一句话摘要

**gstack** 为 MIT 许可的 **Claude Code 技能/命令栈**，将大量**斜杠命令工作流**（评审、QA、设计、发布、安全、规划、复盘等）以 Markdown 驱动的「专家角色」形式打包，面向个人或小团队放大交付能力。

## 关键主张（据上游 README）

1. **目标用户**：技术创始人/CEO、需要结构的新 Claude Code 用户、希望强化评审/QA/发布纪律的技术负责人。
2. **安装方式**：clone 到 `~/.claude/skills/gstack`，执行 `./setup`；可选 **团队模式**（`./setup --team`、`gstack-team-init`），共享仓库可不内嵌整棵目录树。
3. **设计**：用 CEO/EM/设计/评审/QA/安全/发布等隐喻描述角色；README 中 **23 类专家 / 八个强力工具** 等表述，均落为命令与 Markdown。
4. **OpenClaw**：说明如何从 OpenClaw 拉起的 Claude Code 会话中调用 gstack，以及可选 ClawHub「原生」技能。
5. **依赖**（据 README）：Claude Code、Git、Bun、Node（Windows）。

## 与 wiki 的衔接

- 实体页：[GStack](../entities/gstack.md)
- 与 [claude-code-best-practice](repo-claude-code-best-practice.md) 对照：整理型文档 vs 偏安装即用的命令库。

## 关于第三方宣传数据

上游 README 含宣传性统计与第三方产品引用；对外引用数字前请**自行核实**。
