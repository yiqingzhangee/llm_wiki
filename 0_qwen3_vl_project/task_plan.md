# Qwen3-VL 统计特征编码器调研计划

## 目标

针对数值和枚举型特征接入 Qwen3-VL-4B，选出低成本验证、常用稳健和高创新性三类编码器，并将结论织入中文 Wiki。

## 约束

- 三个候选先输出统一语义的统计连续向量；独立 projector 再按需输出统一数量、统一维度的 soft token，避免接入方式干扰比较。
- 所有数据处理遵守预测时点可用性、实体隔离和先切分后拟合。
- 快速验证不修改 Qwen3-VL 主干结构；复杂方案只有在简单方案出现稳定缺口后升级。
- 不把独立表格预测 benchmark 直接等价为多模态编码效果。
- 不修改 `raw/`，不提交 `assets/`。

## 阶段

### 阶段 1：本地资料盘点
- **状态**：已完成
- **产出**：已有路线、证据契约和候选清单。

### 阶段 2：Qwen3-VL 接口核验
- **状态**：已完成
- **产出**：模型 hidden size、视觉 token 注入方式、统计 token 接入边界。

### 阶段 3：候选调研与决策
- **状态**：已完成
- **产出**：三类编码器的结构、成本、风险、实验门禁和推荐顺序。

### 阶段 4：Wiki 内化
- **状态**：已完成
- **产出**：专题对比页、相关概念页更新、索引与日志。

### 阶段 5：验证
- **状态**：已完成
- **产出**：引用、链接、Markdown 与变更边界检查。

## 关键决策

| 决策 | 理由 | 日期 |
|---|---|---|
| 三个方案共享连续向量 + 可选 projector 接口 | 保证比较只改变编码器 | 2026-10-26 |
| 快速验证默认冻结 Qwen3-VL 主干 | 降低训练和回滚成本 | 2026-10-26 |
| 快速验证选 MLP 连续向量 | 固定 schema 下实现最少，不引入字段间注意力和查询压缩 | 2026-10-26 |
| 常用方案选 FT-Transformer | 特征 token 与 Qwen soft-token 接口天然一致，公开实现和经验较成熟 | 2026-10-26 |
| 创新方案选 PLE-FT + 查询压缩 | 同时增强数值非线性、证据元数据和固定 token 预算 | 2026-10-26 |

## 错误日志

| 错误 | 尝试次数 | 解决方案 |
|---|---:|---|

## 修改文件

- `0_qwen3_vl_project/task_plan.md`
- `0_qwen3_vl_project/findings.md`
- `0_qwen3_vl_project/progress.md`
- `wiki/sources/applications/来源-Qwen3-VL统计特征编码器调研-2026.md`
- `wiki/comparisons/applications/Qwen3-VL统计特征编码器选型-2026.md`
- `wiki/concepts/applications/Qwen异构特征接入与事件序列建模.md`
- `wiki/concepts/applications/LLM统计特征编码的证据仲裁.md`
- `wiki/sources/_index.md`
- `wiki/index.md`
- `wiki/log.md`
