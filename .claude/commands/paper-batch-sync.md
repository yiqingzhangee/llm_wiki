---
description: "手动分批摄入 LLM_paper_list 存量论文；不创建定时任务"
allowed-tools: "Read Bash Edit Write"
---

执行一次**手动、有界**的论文存量摄入。不得创建 cron、后台任务或修改 `raw/`。

## 范围与互斥

1. 从仓库根目录运行；仅读取 `raw/papers/2_LLM_NLP/LLM_paper_list/**/*.md`。
2. 排除 `*_images/`、图片、`.DS_Store`、无正文文件及已知镜像。
3. 读取 `.claude/wiki-ingest.lock`：若存在 `heartbeatAt` 距当前不足两小时的有效锁，输出“已有摄入任务在运行”，不写任何文件、不推进状态。超过两小时才可接管。
4. 获得锁后写入任务名、批次 ID、`startedAt`、`heartbeatAt` 与工作目录；每个实质步骤更新 `heartbeatAt`。正常结束或失败时清理锁。

## 状态与选题

1. 读取 `.claude/paper-ingest-state.json`；用仓库相对路径和内容 SHA-256 判断候选是否变化。
2. 未收录的文件初始化为 `pending`；已 `done` 且指纹未变的文件跳过。超大、无正文、镜像或不适合当前主题的文件标为 `deferred` 并记录原因。
3. 单次只能选择一个一级主题的 1–3 篇 Markdown，总字符数不超过 90,000，最多影响 18 个 Wiki 页面。
4. 首批优先 `reasoning/acl_2023_ALERT_Adapt_Language_Models_to_Reasoning_Tasks.md`；后续从状态中的 `nextTopic` 轮转 `rag`、`graphGPT`、`tableGPT`、`deduplication`、`behavior`、`caption` 等主题。任一主题最多连续两批。
5. 先执行 dry-run：打印候选路径、字符数和预计影响页面。除非调用者明确要求真实摄入，否则 dry-run 不写状态或 Wiki。

## 真实摄入

1. 将选中的条目先标为 `claimed`，写入 `batchId`、`claimedAt`、`leaseUntil`、尝试次数和指纹。
2. 阅读论文并按根 `CLAUDE.md` 深度摄入：来源页、必要的概念/对比/地图织入、`wiki/sources/_index.md`、`wiki/index.md` 与仅追加的 `wiki/log.md`。
3. 全部写入与核查成功后才标记 `done`，并记录来源页、影响页面和 `completedAt`；失败则标记 `failed` 并保留错误原因以便重试。
4. 不得把不同数据集、候选池、切分、模型版本或评测协议中的数字直接横比。

## 最小验证

- 确认本批不超过三篇、90,000 字符和 18 个 Wiki 页面；
- 确认 `claimed → done` 只发生在索引和日志更新成功后；
- 检查活锁拒绝、过期锁接管、失败重试和指纹变化后的重新候选；
- 运行 `git diff --check -- wiki`。
