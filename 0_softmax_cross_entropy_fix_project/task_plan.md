# Softmax 与交叉熵数值修复调研计划

## 目标

核验 arXiv:2610.31642 的论文、代码、实验与框架状态，并将 Softmax Collapse、Softmax Cancellation、诊断方法和工程修复路径织入中文 Wiki。

## 约束

- 优先使用论文、作者代码、PyTorch/JAX 源码与官方 issue/PR。
- 将论文结论、作者声明、第三方复现和工程推断分开。
- 不把尚未合并的 PR 表述为框架已经修复。
- 不修改 `raw/`；网络来源直接记录在 Wiki 来源页。
- 完成 Wiki 后更新索引、追加日志并执行 Git 同步。

## 阶段

### 阶段 1：现有知识盘点
- **状态**：已完成
- **产出**：相关 Wiki 页面与潜在整合点。

### 阶段 2：一手证据核验
- **状态**：已完成
- **产出**：论文元数据、两个 bug 的数学机制、实验与代码。

### 阶段 3：框架状态核验
- **状态**：已完成
- **产出**：PyTorch/JAX 当前实现、issue/PR 与版本边界。

### 阶段 4：Wiki 深度整合
- **状态**：已完成
- **产出**：来源页、概念页、工程诊断清单、索引和日志。

### 阶段 5：验证与 Git 同步
- **状态**：进行中
- **产出**：链接、引用、边界检查及远程同步。

## 关键决策

| 决策 | 理由 | 日期 |
|---|---|---|
| 单独建立数值稳定性概念页 | 两个 bug 横跨分类、蒸馏与低精度训练，不应只附属于单篇论文摘要 | 2026-10-28 |
| 框架状态按查询日期冻结 | PyTorch/JAX 合入状态会变化 | 2026-10-28 |
| Gemini 分工按作者公开声明记录 | 该信息出现在第一作者发布帖，但不在论文正文或致谢中 | 2026-10-28 |
| 框架状态写为“尚未完整合入” | PyTorch 语义修复仍是 draft PR，JAX HEAD 仍为标准归约公式 | 2026-10-28 |

## 错误日志

| 错误 | 尝试次数 | 解决方案 |
|---|---:|---|

## 修改文件

- `0_softmax_cross_entropy_fix_project/task_plan.md`
- `0_softmax_cross_entropy_fix_project/findings.md`
- `0_softmax_cross_entropy_fix_project/progress.md`
- `wiki/concepts/reasoning-training/Softmax与交叉熵的浮点数值稳定性.md`
- `wiki/sources/reasoning-training/来源-A-Fix-for-Softmax-and-Cross-Entropy-2026.md`
- `wiki/concepts/reasoning-training/知识蒸馏.md`
- `wiki/concepts/reasoning-training/参数高效微调与对齐训练.md`
- `wiki/sources/_index.md`
- `wiki/index.md`
- `wiki/log.md`
