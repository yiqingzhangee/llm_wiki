# 来源：FOUNDv2 统一量化用户 Tokenizer（2026）

> 原始来源：[`raw/transcripts/LLM-based User Representation Foundation Models/paper_zh_11_FOUNDv2.md`](../../../raw/transcripts/LLM-based%20User%20Representation%20Foundation%20Models/paper_zh_11_FOUNDv2.md)。对应论文：*FOUNDv2: Learning Unified User Quantized Tokenizers for User Representation*，arXiv:2508.00956v3；原始阅读稿标注 KDD 2026，书目信息应以论文元数据复核。
>
> 关联概念：[[LLM用户表征基础模型与场景适配]]、[[LLM推荐系统的分层改造与评估]]。
>
> 一手核验：论文 v3 标注 KDD 2026；官方仓库 [chuanhe1999/FOUNDv2](https://github.com/chuanhe1999/FOUNDv2) 公开 MRQ-VAE 量化核心代码，未包含完整工业数据与全链路训练系统。

## 核心结论

FOUNDv2 将多源用户数据先投影到共享语义空间，再以 MRQ-VAE 的“共享 + 源特定”码本离散化成用户 token，并以 token、时间窗口与语义三级目标训练。其目标是用可控的离散容量换取更长历史、较低存储/算力和跨源可组合性。

## 技术机制

- 文本/行为源由 embedding 模型编码，表格源保留专用 MLP 通道；共享语义空间不意味着放弃结构化输入的专门建模。
- MRQ-VAE 分别保存跨源共性和来源专属模式；多级残差量化提供压缩—保真折中。
- 多尺度预训练同时约束细粒度行为转移、长周期窗口与语义对齐，避免把用户压成不可解释的单块静态向量。

## 工程边界

原始阅读稿报告内存、训练和历史窗口改善，但这些数字受数据、码本、模型及任务协议约束，不能当作通用成本承诺。量化应以重建/碰撞、下游效用、长尾用户和增量更新共同验收；它与物品 Semantic ID 相邻但对象不同：这里压缩的是**用户历史表征**，不是生成候选目录。

在推荐系统中，该方法属于表征与索引层：可为召回或排序提供特征，却不自动解决物品合法性、实时排序、业务策略与线上因果评估。

与公开方案的横向位置见 [[多模态行为序列与表格编码器对比-2026]]。
