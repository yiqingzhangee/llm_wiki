# 仓库与 wiki 结构

按 [`CLAUDE.md`](../../CLAUDE.md) 描述的工作区示意。**元页面** — 供代理与人类维护仓库时导航。

```text
wiki-root/
├── CLAUDE.md                 # 代理模式说明
├── raw/                      # 代理只读
│   ├── articles/
│   ├── papers/
│   ├── transcripts/
│   ├── data/
│   └── code/                 # 代码作来源材料 → 提炼进 wiki
├── wiki/
│   ├── index.md              # 内容目录（建议从这里开始）
│   ├── log.md                # 仅追加时间线
│   ├── meta/                 # 模式与工作流说明（非领域知识库正文）
│   ├── concepts/             # 领域概念（来自 raw/）
│   ├── entities/
│   ├── sources/              # 按 raw 路径对应的摄入摘要
│   ├── comparisons/
│   └── maps/                 # 领域地图/导航（可选）
└── assets/                   # 可选；也可放在 raw/ 下
```

**代理阅读顺序**

- **做领域内容**：[Wiki 索引](../index.md) → `sources/` → `entities/` / `concepts/` → 跟随链接。
- **做运维/模式**：[Wiki 索引](../index.md) → `meta/`（本目录）。

相关：[三层架构](three-layer-architecture.md)、[本 wiki 仓库](this-wiki-repository.md)。
