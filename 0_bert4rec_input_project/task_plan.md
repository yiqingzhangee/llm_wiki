# BERT4Rec 输入张量 Wiki 更新计划

## 目标

核验 BERT4Rec 原论文与代表实现中的数据处理和输入张量格式，并将训练、验证/测试、推理阶段的张量契约整理进中文 Wiki。

## 约束

- 优先使用原论文、作者参考实现和主流开源实现。
- 区分原始交互数据、预处理样本和模型输入张量。
- 区分 TensorFlow 稀疏 masked-LM 标签与 PyTorch 常见密集标签格式。
- 不把某个第三方实现的字段名误写成 BERT4Rec 统一标准。
- 不修改 `raw/`；完成后更新索引、追加日志并执行 Git 同步。

## 阶段

### 阶段 1：现有 Wiki 与来源审计
- **状态**：已完成
- **产出**：可复用页面、缺口与整合位置。

### 阶段 2：一手证据核验
- **状态**：已完成
- **产出**：原论文训练目标、作者代码数据管线和张量字段。

### 阶段 3：实现差异核验
- **状态**：已完成
- **产出**：TensorFlow 与 PyTorch 格式对照、字段边界。

### 阶段 4：Wiki 深度整合
- **状态**：已完成
- **产出**：概念页、关联页面、索引和日志。

### 阶段 5：验证与 Git 同步
- **状态**：进行中
- **产出**：链接、格式、边界检查和远程同步。

## 关键决策

| 决策 | 理由 | 日期 |
|---|---|---|
| 新建独立输入张量页面 | 输入契约可被推荐、事件序列和模型实现页面复用 | 2026-10-28 |
| 以“语义等价、字段不统一”为主线 | 原论文定义任务，具体张量键由实现决定 | 2026-10-28 |

## 错误日志

| 错误 | 尝试次数 | 解决方案 |
|---|---:|---|
| RecBole clone 命令在尚不存在的 `/tmp/recbole` 目录作为 workdir，工具内部失败 | 1 | 使用 RecBole 官方在线源码核验；若仍需 clone，从已存在的 `/tmp` 发起 |
| 全库 Wikilink 检查命中既有缺链 | 1 | 改为检查本次新增页面及新增链接，避免顺手修改无关历史问题 |
| ACM DOI 对 `curl` 返回 403，RecBole 文档在批量探测中超时 | 1 | 论文以 arXiv/作者代码交叉核验；RecBole 已通过浏览工具取得完整官方源码，不重复相同探测 |

## 修改文件

- `0_bert4rec_input_project/task_plan.md`
- `0_bert4rec_input_project/findings.md`
- `0_bert4rec_input_project/progress.md`
- `wiki/sources/applications/来源-BERT4Rec原论文与参考实现-2019.md`
- `wiki/concepts/applications/BERT4Rec数据处理与输入张量.md`
- `wiki/concepts/applications/Qwen异构特征接入与事件序列建模.md`
- `wiki/concepts/applications/LLM推荐系统的分层改造与评估.md`
- `wiki/maps/研究主题导航.md`
- `wiki/sources/_index.md`
- `wiki/index.md`
- `wiki/log.md`
