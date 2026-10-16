---
description: "拉取Agent+风控相关论文/博客/社媒内容并归档到 raw/search（可用于8小时定时任务）"
allowed-tools: "Skill Read Bash Write"
---

执行一次“Agent + 风控”情报拉取，并将结果归档到 `raw/search/`。

要求：

1. 先调用 `/anysearch-skill`（通过 Skill 工具），查询主题：
   - `agent 风控 论文`
   - `agent risk control paper`
   - `AI agent fraud detection`
   - `LLM agent 风险治理`
   - `B站 agent 风控`
   - `小红书 agent 风控`
   - `微信公众号 agent 风控`
   - `social media agent governance`
2. 覆盖来源类型：论文、博客、B站、小红书、微信及其他社交媒体（可通过 AnySearch 的 academic / social_media / general 路径组合）。
3. 不修改 `raw/` 既有文件，仅在允许写入目录新增归档：
   - 目录：`raw/search/traffic-risk-control-watch/<YYYY-MM-DD>/`
   - 文件名：`agent-fengkong-watch-<YYYY-MM-DD>-<HHmm>.md`
4. 归档文件至少包含：
   - 查询时间、查询词、来源类别统计
   - 每条结果的标题、URL、平台、时间（若可得）、一句摘要
   - “高价值来源”清单（官方文档/权威媒体/高信息密度条目）
   - “待复核”清单（证据薄弱或来源不明）
5. 若当次无有效结果，也要写归档文件，明确记录“无有效结果”和错误原因。
6. 输出简短执行摘要：
   - 归档路径
   - 抓取条数与来源分布
   - 高价值来源数量
