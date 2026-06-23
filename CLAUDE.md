# LLM Wiki — Agent Schema

This file is **for the LLM agent**. It defines structure, boundaries, and workflows for maintaining a **persistent, compounding wiki** (interlinked Markdown) between the human and **immutable raw sources**. The human sources and explores; you integrate, cross-reference, and file. **Co-evolve this schema** with the human when conventions need to change.

---

## Core idea (why this exists)

- **Not only RAG at query time**: Upload-and-retrieve systems rediscover fragments on every question; knowledge does not accumulate in a durable artifact.
- **Here**: You **incrementally compile** knowledge into a wiki — entity pages, topic summaries, synthesis — and **keep it current** as new sources arrive. Cross-links, contradictions, and synthesis live in files, not only in chat.
- **Division of labor**: The human rarely writes the wiki. You write and maintain `wiki/`. The human curates **raw sources**, explores, and asks good questions.

**Practical setup**: Human often uses Obsidian to read the wiki (graph, links); you apply edits from the conversation. Treat the wiki like a codebase you maintain.

---

## Architecture — three layers

| Layer | Role | Agent |
|-------|------|--------|
| **Raw sources** (`raw/`) | Curated truth: articles, papers, images, data, and **`code/`** (various code used as source material). Default immutable, with a dedicated writable archive under `raw/search/`. | **READ only by default.** Never create, edit, move, or delete files under `raw/`, **except** `raw/search/` where the agent may create/update search-archive `.md` files. |
| **Wiki** (`wiki/`) | Summaries, entities, concepts, comparisons, maps, syntheses — **produced in Chinese** (see Global rules). | **READ/WRITE.** You create and update Markdown here; keep links and claims consistent. |
| **Schema** (this file) | Structure, conventions, workflows. | **Follow.** Propose updates to this file when the human agrees. |

---

## Repository layout

Create missing directories as needed when processing sources.

```
wiki-root/
├── CLAUDE.md              # This schema (agent instructions)
├── raw/                   # Immutable sources (human-managed placement)
│   ├── articles/
│   ├── papers/
│   ├── transcripts/
│   ├── data/
│   ├── search/            # 搜索结果归档（由代理写入，保存链接详情的 .md）
│   └── code/              # Code as source material; extract knowledge into wiki (human-managed)
├── wiki/                  # Agent-maintained knowledge
│   ├── index.md           # Catalog of pages (content-oriented)
│   ├── log.md             # Append-only timeline
│   ├── meta/              # Schema/workflow notes (CLAUDE.md alignment); not domain KB from raw/
│   ├── entities/
│   ├── concepts/
│   ├── sources/           # One summary page per ingested source (typical)

│   ├── comparisons/
│   └── maps/
└── assets/                # Images, figures (optional; may live under raw/)
```

**Attachments**: If the human uses Obsidian, `raw/assets/` (or similar) may hold downloaded images. You still **do not edit `raw/`**; describe or reference paths in wiki text when useful.

**`raw/code/`**: Holds **various code** the human places here as raw material. You **read** it to extract substantive information (design, APIs, behavior, domain logic) and **integrate that into `wiki/`**—entities, concepts, source summaries—so the knowledge base grows and stays current. **Do not modify** `raw/code/` (same global rule as for all of `raw/`).

**`wiki/meta/`**: Holds **operational/schema** pages aligned with this file (three-layer explanation, ingest/query/lint cheat sheets, repo layout, `CLAUDE.md` summary). Keep them separate from **domain knowledge** pages under `wiki/concepts/`, `wiki/sources/`, etc., so humans can tell “how the wiki runs” from “what the wiki knows.” See `wiki/meta/README.md`.

---

## Global rules (agent)

1. **默认不修改 `raw/`** — 包括 "small fixes"；若源有误，优先请 human 修正。**唯一例外**：`raw/search/` 允许代理写入与维护搜索归档 `.md` 文件（见下文 Search Archive 规则）。
2. **Wiki language — Simplified Chinese (`zh-CN`) for all knowledge-base prose**: Everything you **author** under `wiki/` (including `wiki/meta/`) must be written in **简体中文**: titles, headings, body text, lists, table cells, and one-line summaries in `wiki/index.md`. **Exceptions**: (a) direct **quotes** from `raw/` sources (then quote verbatim and optionally gloss in Chinese); (b) **identifiers** — API names, file paths, commands, repo URLs, paper titles in citations where the canonical form is non-Chinese; (c) **code blocks** and pasted logs. When ingesting English papers or code, **summarize and interpret in Chinese**; do not leave full pages English-only unless the human explicitly opts out in this file.
3. **Append-only `wiki/log.md`** — never rewrite history; add new entries at the end (or in chronological order at end). **Log entry titles/descriptions** after the `## [date] type |` header should also be in **简体中文** (type keyword `ingest` / `query` / `lint` may stay English for grep stability).
4. **Prefer wikilinks** (`[[Page Title]]`) or relative Markdown links consistent with existing wiki style. **Filenames** may stay ASCII/`kebab-case` for tooling compatibility; **display text** and headings inside files should still be Chinese.
5. **Surface contradictions** explicitly on relevant pages when new sources conflict with old claims; do not silently overwrite without noting the tension.
6. **After substantive work**, update `wiki/index.md` and append `wiki/log.md` (see below).
7. **Citations**: When answering from the wiki, point to specific pages or sections where possible.
8. **Git commits use a dated version branch**: Do **not** commit straight to `main` for knowledge-base work. For each **distinct** commit (or batch) to publish, **create a new branch** whose name is **`kb/<YYYY-MM-DD>-v<N>`** (date + version), commit on that branch, push it, then merge into `main` and push `main`. Details in **Operation: Git sync**.

---

## Operation: Git sync

**Canonical remote (SSH)**:

```text
git@code.byted.org:zhangyiqing.albert/agentic_llm_wiki.git
```

**Branch naming (`日期` + `版本`)**

- Pattern: **`kb/<YYYY-MM-DD>-v<N>`**
  - **`<YYYY-MM-DD>`**: calendar date of the commit (use the session’s authoritative “today” when the agent runs).
  - **`<N>`**: version index **within that day** — start at **`v1`** for the first KB-related commit/push that day; use **`v2`, `v3`, …** for additional separate commits the same day (after the previous branch has been merged into `main`, or when starting another independent change set, per team habit).
- Examples: `kb/2026-04-15-v1`, `kb/2026-04-15-v2`.

**Default workflow (each time you publish wiki/`CLAUDE.md` changes)**

1. `git checkout main` && `git pull origin main` — sync integration branch.
2. `git checkout -b kb/<YYYY-MM-DD>-v<N>` — create the dated version branch (pick `<N>` by listing existing `kb/<date>-v*` branches or agreeing the next free index).
3. `git add` … → `git commit -m "…"` — commit only on this branch.
4. `git push -u origin kb/<YYYY-MM-DD>-v<N>` — publish the branch.
5. Merge into `main`: either `git checkout main` && `git merge --no-ff kb/...` && `git push origin main`, or open a merge request on the code host — **choose one convention** with the human; the agent defaults to **local merge + push `main`** when unblocked, so the remote stays a single source of truth.
6. Optionally delete the merged topic branch locally/remotely after merge if the human prefers a tidy remote.

**Constraints**

- Do **not** `git push --force` to `main` or rewrite published history without explicit human approval.
- If not a git repo, push fails, or branch naming collides, report and stop; ask the human to resolve (SSH, permissions, duplicate branch names).

---

## Operation: Ingest

**Trigger**: Human adds a source under `raw/` and asks you to process it.

**Default workflow** (adapt if the human prefers batching or minimal discussion):

1. Read the source (text first; for important visuals, read image files separately if available — inline images in Markdown are not always processed in one pass).
2. Confirm key takeaways with the human when they want to stay in the loop (recommended for single-source, supervised ingest).
3. Add or update **`wiki/sources/<descriptive-name>.md`** (or your agreed convention) — summary, key claims, link to raw path (**简体中文** per Global rules).
4. Update **`wiki/entities/`**, **`wiki/concepts/`**, and other relevant pages so the new information is integrated, not orphaned.
5. Add cross-references; note contradictions with existing pages.
6. Update **`wiki/index.md`** — link, one-line summary, optional metadata (date, tags, source count).
7. Append **`wiki/log.md`** with the standard entry format (below).
8. **Git sync** — Global rule 8 and **Operation: Git sync** (dated branch → commit → push branch → merge to `main`).

**Scale**: One source may touch many pages (often ~10–15). **Single-source ingest with human review** is the default recommendation unless the human documents otherwise here.

### 默认深度更新策略（长期生效）

- 以“本次会话已实现的深度更新粒度”为后续 ingest 的**最低目标**：不仅写 `wiki/sources/`，还要主动织入 `wiki/concepts/`、`wiki/comparisons/`、`wiki/maps/`，并补齐必要的交叉链接。
- 对新增 `raw/` 内容，默认执行上述深度整合；除非用户明确要求“仅最小摘要”，否则**不需要逐次征求同意**。
- 每次深度更新后，必须同步更新 `wiki/index.md` 与 `wiki/log.md`。

---

## Operation: Search archive (`raw/search/`)

**Trigger**: Human asks for search results to be archived.

1. Create (if missing) and use `raw/search/` only.
2. **按专题分桶存放**：优先使用 `raw/search/<topic-slug>/<YYYY-MM-DD>/` 结构，避免把所有结果平铺在同一层，便于增量追踪与后续清理。
3. For each result link, fetch/read the detailed page content and create a separate `.md` file.
4. Filename convention: **`<内容主题>-<YYYY-MM-DD>.md`**（内容主题使用简短可读 slug，必要时可加序号防冲突）.
5. Each file should include at least: source URL, capture date, title/summary, and extracted key content.
6. Archive files in `raw/search/` are considered raw inputs for later wiki compilation.
7. **保留与清理策略（默认）**：
   - 先保留本次搜索归档的全部文件 **3 天**（用于追溯与复核）；
   - 完成一次 wiki 编译后，仅保留“**示例文件 + 高价值来源**”，其余可删除；
   - “高价值来源”优先指：官方文档/标准、权威一手资料、信息密度高且被 wiki 实际引用的来源。

---

## Ingest scope and recursive scan

- Every wiki generation/ingest pass must recursively scan **all** subdirectories under `raw/` (including nested directories at arbitrary depth).
- Do not miss files in deep paths (e.g., `raw/**/**/file.md`).
- Newly added files under `raw/search/` must be treated the same as other new raw sources and integrated into `wiki/`.

---

## Operation: Query

**Trigger**: Human asks a question against the knowledge base.

1. Read **`wiki/index.md`** to locate relevant pages.
2. Open those pages (and follow links as needed).
3. Synthesize an answer **with citations** to wiki pages (and raw paths if quoting source of truth).
4. **File high-value answers back** when the human agrees: comparisons, analyses, and connections should become new or updated wiki pages — not only chat history.
5. If wiki files changed materially — **Git sync** per Global rule 8 and **Operation: Git sync**.

**Output shapes** (as appropriate): Markdown page, comparison table, Marp deck, matplotlib chart, canvas-style outline — then link or embed from the wiki if the human wants it preserved.

---

## Operation: Lint (health check)

**Trigger**: Human requests a wiki health pass (periodic recommended).

Checklist — report findings and propose concrete edits:

- Contradictions between pages; stale claims superseded by newer sources.
- Orphan pages (no inbound links from index or other pages).
- Important concepts mentioned but missing dedicated pages.
- Missing cross-references where obvious.
- Data gaps addressable by web search (suggest queries and where results would attach).

After fixes the human approves: update **`wiki/index.md`** as needed and **append `wiki/log.md`**, then **Git sync** per **Operation: Git sync**.

---

## `wiki/index.md` (content catalog)

- **Purpose**: Content-oriented map of the wiki so you can find pages without embedding infrastructure.
- **Update**: After every **ingest** and after **major** query/lint outcomes that add or rename pages.
- **Shape** (adapt categories to the domain): group by entities, concepts, sources, comparisons, maps, etc.
- **Per page**: link, **one-line summary**, optional YAML or inline metadata (date, source count) if the human uses Dataview.
- **Language**: Section headings and per-page summaries must follow the **简体中文** rule under Global rules.

---

## `wiki/log.md` (append-only timeline)

- **Purpose**: Chronological record of ingests, notable queries filed into the wiki, and lint passes. Helps you see **recent work** across sessions.

**Required entry header format** (so Unix tools work):

```markdown
## [YYYY-MM-DD] ingest | Short title or filename
## [YYYY-MM-DD] query | Filed: Page name (optional)
## [YYYY-MM-DD] lint | Summary of pass
```

Examples:

- `grep '^## \\[' wiki/log.md | tail -5` — last five entries.

**Rules**: Newest entries typically at the **bottom** (or top if the human standardizes otherwise — **pick one convention in this file and stick to it**). Default: **append at end of file**.

---

## Optional: CLI / search

At small scale, `index.md` may suffice. As the wiki grows, add local search (e.g. **qmd** — BM25/vector hybrid, on-device; CLI and MCP). Document any installed tool and when to use it in this schema once adopted.

---

## Domain fit (examples)

Personal growth, long-form research, reading companions (characters/themes), team wikis fed by Slack/meetings (with human review), competitive analysis, diligence, trip planning, courses, hobbies — any domain where knowledge **accumulates over time** and should stay **organized and linked**.

---

## Tips (human workflow; agent awareness)

- **Obsidian Web Clipper**: quick Markdown articles into `raw/`.
- **Download attachments** to a fixed folder (e.g. `raw/assets/`) so images remain local and referenceable.
- **Graph view**: shape of links and orphans.
- **Marp / Dataview / git**: presentations, frontmatter queries, version history — use if the human enables them; reflect conventions here when stable. KB commits: branches **`kb/<YYYY-MM-DD>-v<N>`**, remote **`git@code.byted.org:zhangyiqing.albert/agentic_llm_wiki.git`** (see **Operation: Git sync**).

---

## Summary

| MUST | MUST NOT |
|------|----------|
| Maintain `wiki/` with integration, links, and contradiction notes; **write all KB prose in 简体中文** (see Global rules) | Modify anything under `raw/` |
| Append `wiki/log.md` with the standard `## [date] type | …` headers | Delete or rewrite past `log.md` entries |
| Update `wiki/index.md` after ingests (and after major structural changes) | Treat chat as the system of record for durable knowledge |
| On each KB git publish: **create branch** `kb/<YYYY-MM-DD>-v<N>`, commit there, push, merge to `main` (see **Operation: Git sync**) | Commit KB work directly on `main` without a dated branch; force-push `main` without approval |

When in doubt, **preserve provenance** (link to source pages and raw paths), **mark uncertainty and conflict**, and **offer to update this schema** when workflows diverge from the above.
