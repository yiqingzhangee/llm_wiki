# 来源：MultiModalPFN 多模态表格学习（2026）

## 来源信息

- 论文：[MultiModalPFN: Extending Prior-Data Fitted Networks for Multimodal Tabular Learning](https://arxiv.org/abs/2602.20223)
- 会议：CVPR 2026
- 官方代码：[too-z/MultiModalPFN](https://github.com/too-z/MultiModalPFN)
- 来源等级：论文与官方仓库，按 **E3/E4** 使用

## 架构

MultiModalPFN 以 TabPFN 为主干，将图像和文本转成与表格特征兼容的 token：

1. 表格分支使用冻结的 TabPFN v2 encoder；
2. 图像分支取 DINOv2 ViT-B/14 的 `[CLS]`；
3. 文本分支取 ELECTRA 的 `[CLS]`；
4. Multi-head Gated MLP 将一个非表格向量扩展为多个投影 token；
5. Cross-Attention Pooler 再将这些 token 压缩为固定、平衡的集合；
6. 多模态 token 与表格 embedding 一起进入 TabPFN backbone。

TabPFN 块内把注意力拆为 feature attention 与 sample attention，分别建模同一样本内的列关系和同一特征跨样本的关系。模型还用 attention imbalance 分析处理模态 token 数差异。

## 发布边界

仓库包含实验代码，但使用说明要求用户自行放置 TabPFN v2 checkpoint 和数据。代码公开不等于提供了可直接部署的完整权重包。

## 关联页面

- [[多模态行为序列与表格编码器对比-2026]]
- [[表格数据深度学习]]
- [[来源-TableGPT2表格数据多模态接入-2026]]
