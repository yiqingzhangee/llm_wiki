---
description: "扫描 raw/ 并按 CLAUDE.md 规则增量更新 wiki/，最后输出本次更新摘要"
allowed-tools: "Read Bash Edit Write"
---

执行一次 `raw/ -> wiki/` 增量同步，严格遵循项目根目录 `CLAUDE.md` 的规则。

步骤：

1. 读取并遵循 `CLAUDE.md`（特别是：`raw/` 只读、`wiki/` 用简体中文、更新 `wiki/index.md`、append `wiki/log.md`）。
2. 递归扫描 `raw/` 的全部子目录，识别新增或未总结的来源（优先根据 `wiki/sources/` 与 `wiki/log.md` 判断是否已处理）。
3. 对每个需要处理的来源（默认深度模式，无需逐次征求同意）：
   - 读取原始内容（不修改 `raw/`）；
   - 更新/新增 `wiki/sources/*.md`；
   - 主动同步更新 `wiki/entities/`、`wiki/concepts/`、`wiki/comparisons/`、`wiki/maps/`，避免只做来源页不做知识织入；
   - 明确标注与既有结论的冲突或不确定性。
4. 更新 `wiki/index.md` 的目录与一行摘要。
5. 向 `wiki/log.md` 追加一条 `## [YYYY-MM-DD] ingest | ...` 记录（仅追加，不改历史）。
6. 在聊天中输出“本次更新摘要”，至少包括：
   - 扫描到的新增/变更 raw 来源路径；
   - 新增或更新了哪些 wiki 页面；
   - 是否发现冲突；
   - 后续建议。
7. 不修改、移动、删除 `raw/` 下的任何文件；不执行 Git 提交、推送或远程同步，除非用户另行明确要求。

若本次无新增来源，明确回复“无变更”，并简要说明扫描范围。
