# 来源：Qwen3-VL 统计特征编码器调研（2026）

> 调研日期：2026-10-26
>
> 对应选型：[[Qwen3-VL统计特征编码器选型-2026]]
>
> 关联概念：[[Qwen异构特征接入与事件序列建模]]、[[LLM统计特征编码的证据仲裁]]、[[表格数据深度学习]]

## 调研问题

目标是把数值和枚举型统计特征编码后接入 Qwen3-VL-4B，同时保留文本、图像与视频输入。调研分别回答：

1. Qwen3-VL-4B 是否存在可复用的连续 embedding 接口；
2. 哪种编码器适合最低成本验证、成熟常用实现和高创新性实验；
3. 独立表格预测结果能否直接推出“接入 Qwen3-VL 后最优”。

## 核心裁决

**没有公开证据证明某个统计特征编码器在接入 Qwen3-VL-4B 后普适最优。** 现有论文主要评估独立表格预测、视觉连接器或整表理解，任务、数据、token 预算和冻结策略均不同。因此，“最佳编码器”当前属于 **证据不足**，必须在同一 Qwen 主干、统计 token 数和训练预算下实测。

工程上可形成三个清晰候选：

| 目标 | 候选 | 选择依据 | 证据边界 |
|---|---|---|---|
| 最低开发成本 | 字段语义/ID/取值加法融合 + 两层 MLP | 先输出一个连续向量；Qwen projector 完全解耦 | 只验证连续通道是否有增益，不擅长显式建模字段交互 |
| 成熟常用代表 | FT-Transformer | 每列一个 feature token，和 LLM token 接口自然匹配；公开实现和工程经验丰富 | “常用代表”不等于最新表格 benchmark 最优 |
| 创新性最足 | PLE/周期数值编码 + 治理元数据 + 浅层 FT + learned-query 压缩 | 同时处理数值非线性、字段交互、证据状态和固定 token 预算 | 是跨论文组件组合假设，尚无 Qwen3-VL 直接 benchmark |

如果唯一目标是结构化表格预测准确率，而不是让 Qwen 使用统计信息，应同时保留 CatBoost/LightGBM 和 TabM 等专用模型；它们可能优于深度表格 Transformer，但不直接解决多模态 soft-token 接入。

## Qwen3-VL 官方接口证据

### 模型维度

Qwen3-VL-4B-Instruct 官方配置显示：

- 文本主干 `hidden_size=2560`，36 层，32 个 attention heads，8 个 KV heads；
- 视觉编码器 `hidden_size=1024`；
- 视觉 patch merger 的 `out_hidden_size=2560`，与文本主干维度对齐。

这说明只有 Qwen 适配器的最终输出需要对齐 2560。统计编码器本身可先输出较小的 `[batch,d_stat]` 连续向量；需要接入 Qwen 时，再由独立 projector 产生 `[batch,M,2560]`。最低成本方案取 `M=1`。1024 是视觉塔内部维度，不是统计 token 接入语言主干的目标维度。

### 官方前向链路

Transformers 的 Qwen3-VL 实现先建立文本 embedding，再编码图像/视频，以 placeholder mask 将视觉表示写入对应位置，随后计算 M-RoPE 位置并进入语言模型。文本模型允许 `input_ids` 或 `inputs_embeds` 二选一。

由此得到的工程边界是：

- 可用独立统计占位 token，在 wrapper 中把其 embedding 替换为统计编码结果；
- 统计 token 与视觉 token 并存时，应先依据原始 `input_ids`、多模态 token 类型和视觉网格计算位置，再替换统计 embedding，并显式传入 `position_ids`；
- 不应把统计向量伪装为图片，因为这会引入无意义的视觉分块和空间位置假设；
- 标准 Transformers 训练 `forward()` 可由 wrapper 实现，但官方 vLLM、SGLang 和 OpenAI 风格服务示例没有声明自定义统计 soft-token 字段；生产推理需要另行验证或实现插件。

## 表格编码器证据

### FT-Transformer

FT-Transformer 将每个数值或类别字段变成 token，再用 Transformer 建模字段间交互。数值字段通常由标量乘可学习向量并加偏置得到，类别字段使用 embedding；`[CLS]` 表示用于最终预测。其优势不是“Transformer 一定优于树模型”，而是输出保持字段粒度，便于继续池化或压缩成 LLM soft token。

### 数值 embedding

《On Embeddings for Numerical Features in Tabular Deep Learning》系统研究分段线性编码和周期性编码，显示更强的数值 embedding 可提升多种深度表格模型。对本项目的可迁移结论是：当简单标准化标量不足时，可升级为 PLE 或周期编码；论文不证明这种升级接入 Qwen 后必然有效。

### 更新的表格基线

TabM 通过高效参数集成提升表格深度模型表现；TabArena 则显示现代方法的排名依赖数据集和预算，GBDT 仍是重要强基线。这些结果削弱了“FT-Transformer 是独立表格预测普适最优”的说法，但不否定它作为字段 token 编码器的接口优势。

## 固定 token 压缩的邻近证据

- BLIP-2 的 Q-Former 用一组可学习 query 从冻结视觉编码器提取固定数量表示，形成接入冻结 LLM 的信息瓶颈。
- Flamingo 的 Perceiver Resampler 把变长视觉特征压缩为固定数量 token。
- TableGPT2 使用面向整表的 semantic table encoder，并生成紧凑列表示，但其任务更偏整表理解和商业智能。

这些工作支持“learned query 可控制 token 预算”的结构动机，却没有直接证明字段 token 经查询压缩后接入 Qwen3-VL 的效果。因此，查询压缩只进入创新候选，不应作为首个 PoC 的默认复杂度。

## 主要来源

### 官方资料

- [Qwen3-VL 官方仓库](https://github.com/QwenLM/Qwen3-VL)
- [Qwen3-VL-4B-Instruct 模型卡](https://huggingface.co/Qwen/Qwen3-VL-4B-Instruct)
- [Qwen3-VL-4B-Instruct 配置](https://huggingface.co/Qwen/Qwen3-VL-4B-Instruct/blob/main/config.json)
- [Transformers Qwen3-VL 实现](https://github.com/huggingface/transformers/blob/main/src/transformers/models/qwen3_vl/modeling_qwen3_vl.py)

### 论文

- [Revisiting Deep Learning Models for Tabular Data（FT-Transformer）](https://arxiv.org/abs/2106.11959)
- [On Embeddings for Numerical Features in Tabular Deep Learning](https://arxiv.org/abs/2203.05556)
- [TabM: Advancing Tabular Deep Learning with Parameter-Efficient Ensembling](https://arxiv.org/abs/2410.24210)
- [TabArena: A Living Benchmark for Machine Learning on Tabular Data](https://arxiv.org/abs/2506.16791)
- [BLIP-2: Bootstrapping Language-Image Pre-training](https://arxiv.org/abs/2301.12597)
- [Flamingo: a Visual Language Model for Few-Shot Learning](https://arxiv.org/abs/2204.14198)
- [TableGPT2: A Large Multimodal Model with Tabular Data Integration](https://arxiv.org/abs/2411.02059)

## 未决问题

- 自定义统计占位在 `generate()` 首步使用 `inputs_embeds` 后，KV cache 与后续 token 准备是否在目标 Transformers 版本中完全兼容；
- vLLM/SGLang 是否可通过现有 embedding 输入能力接入，还是必须注册自定义模型；
- 单 token 是否出现稳定容量瓶颈；只有出现瓶颈后才比较多 token、字段池化与 learned-query 压缩；
- 统计编码器收益是否来自真实字段结构，需通过数值置换、类别置换、去 missing mask 和去时间元数据验证。
