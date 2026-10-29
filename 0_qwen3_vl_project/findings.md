# Qwen3-VL 统计特征编码器调研发现

## 本地证据

- `wiki/concepts/applications/Qwen异构特征接入与事件序列建模.md` 已定义从文本化、软 token 到交叉注意力的渐进接入路线。
- `wiki/comparisons/multimodal/异构数据接入大模型的四档方案对比-2026.md` 要求先保留强基线，并以同预算晚融合和专用模型作为门禁。
- `wiki/concepts/applications/LLM统计特征编码的证据仲裁.md` 要求统计值携带单位、时间窗、缺失、来源和复算状态。
- FT-Transformer 是天然输出字段 token 的表格编码器，但更新 benchmark 不支持其作为独立表格预测的普遍最优模型。

## 外部证据

### Qwen3-VL 官方边界

- 官方 Qwen3-VL 仓库要求 `transformers>=4.57.0`，标准处理链只声明文本、图像和视频三类输入。
- 官方 chat template 使用 `<|vision_start|><|image_pad|><|vision_end|>` 与 `<|video_pad|>` 占位；视觉编码结果在模型内部替换对应占位位置。
- Qwen3-VL-4B-Instruct 是 Dense 4B 模型，官方模型卡提供 Transformers、vLLM 与 SGLang推理入口，但标准 OpenAI 风格服务接口没有自定义统计模态字段。
- 官方配置显示文本 hidden size 为 2560、36 层、32 个 attention heads 和 8 个 KV heads；视觉 encoder hidden size 为 1024，patch merger 输出 2560，与文本 hidden size 对齐。
- Transformers 的文本模型允许 `input_ids` 或 `inputs_embeds` 二选一；直接传 `inputs_embeds` 时需要调用方自行保证视觉 token、位置和 DeepStack 视觉特征链路正确。
- `Qwen3VLModel.forward` 的顺序是：建立文本 `inputs_embeds`、编码图像/视频、用 placeholder mask 执行 `masked_scatter`、计算 M-RoPE `position_ids`、再进入 language model。
- 当只传 `inputs_embeds` 且同时有视觉输入时，源码不能从统计替换后的 embedding 可靠重建完整 M-RoPE；wrapper 应先用原始 `input_ids`、`mm_token_type_ids` 和视觉网格计算 `position_ids`，再替换统计占位 embedding，并显式把 `position_ids` 传回模型。
- **工程边界**：训练 forward 可在不 fork 主干的 wrapper 中实现；`generate()`、vLLM 和 SGLang 需要额外验证或自定义模型插件，不能把离线 forward 成功等价为生产服务兼容。
- **影响**：本地训练/验证应增加独立统计占位 token，并由 wrapper 在 embedding 序列中替换；不要伪装成图片，也不要假设标准 vLLM API 能直接承载自定义 soft token。

### 固定长度查询压缩的外部锚点

- BLIP-2 的 Q-Former 使用一组可学习 query 从冻结视觉 encoder 中提取固定数量表示，作为冻结 LLM 前的信息瓶颈。
- Flamingo 的 Perceiver Resampler 将变长视觉特征压缩成固定数量 token；原论文消融报告其优于相同参数预算的 MLP 和 vanilla Transformer。
- TableGPT2 为整表设计 semantic table encoder，并按列输出紧凑 embedding，但目标更偏整表理解与 BI，不直接证明单行统计向量接入 Qwen3-VL 的效果。
- **影响**：创新方案可把 Q-Former/Perceiver 的“固定查询压缩”迁移到字段 token，但应标为架构组合假设，不能宣称已有直接 benchmark 支持。

## 候选

1. 低成本：类别 embedding + 数值标准化 + 两层 MLP → 固定数量 soft token。
2. 常用稳健：FT-Transformer → 查询/池化 → soft token。
3. 创新方案：PLE/周期数值嵌入 + 治理元数据 token + FT-Transformer + 查询压缩器。

## 待核验

- `generate()` 首步传 `inputs_embeds` 后的缓存与后续 token 准备是否兼容自定义统计占位。
- vLLM/SGLang 是否存在无需自定义插件的 soft-token 输入接口；当前官方示例未提供。
- 三种编码器在目标冻结样本上的实际收益与时延。

## 最终裁决

- 没有 Qwen3-VL-4B 直接比较统计编码器的公开 benchmark，不能给出普适“最好”结论。
- 最低工程成本选择 MLP soft-token，用于验证连续统计通道本身。
- 成熟常用代表选择 FT-Transformer，依据是字段 token 接口契合与公开实现成熟，而非独立表格预测普适最优。
- 创新候选选择 PLE/周期数值 embedding + 治理元数据 + 浅层 FT + learned-query 压缩。
- 若目标只是表格预测准确率，应优先比较 CatBoost/LightGBM/TabM 和 late fusion，不应预设接入 Qwen 更优。
