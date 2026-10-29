# BERT4Rec 输入张量研究发现

## 待核验

- 原论文的 Cloze 训练样本生成、掩码比例、最大序列长度与评估构造。
- 作者参考实现的 TFRecord 字段、shape、dtype 和 padding 约定。
- 常见 PyTorch 实现的 `input_ids`、`attention_mask`、`labels` 语义。
- 推理时末尾 `[MASK]` 的输入与候选打分方式。
- `user_id`、时间戳、文本与其他辅助特征是否属于原始模型输入。

## 已确认

- 现有 Wiki 没有 BERT4Rec 专页，也没有其输入张量契约。
- 可关联页面为 `Qwen异构特征接入与事件序列建模.md` 与 `LLM推荐系统的分层改造与评估.md`。
- 原论文发表于 CIKM 2019，核心训练任务是随机掩码行为序列中的 item，并同时利用左右上下文预测被掩码 item。
- 现有 `raw/` 中没有直接的 BERT4Rec 原论文或实现；本次需要使用网络一手来源。
- 现有事件序列页面偏向 Qwen/GRU 接入和工程选型，不能替代 BERT4Rec 的离散 item 序列与 MLM 张量契约。
- 原论文把每个用户的交互按时间排序为 item ID 序列；超过最大长度 `N` 时只保留最近 `N` 个 item。输入表示是 item embedding 与可学习 position embedding 之和。
- 原论文训练时随机选择比例 `ρ` 的 item 作为 Cloze 目标；测试时在历史末尾追加 `[MASK]`，用该位置的最终 hidden state 预测下一 item。为缩小训练/测试差异，训练中还加入只掩码最后 item 的样本。
- 原论文实验采用 leave-one-out：最后 item 为测试标签、倒数第二个为验证标签，其余为训练历史；`N=200` 用于 ML-1m/ML-20m，`N=50` 用于 Beauty/Steam。这些是论文实验配置，不是模型接口的固定常数。
- 作者 TensorFlow 参考实现将样本序列化为 TFRecord：`input_ids:[L]`、`input_mask:[L]`、`masked_lm_positions:[M]`、`masked_lm_ids:[M]`、`masked_lm_weights:[M]`，另有 `info:[1]` 保存用户编号。
- 参考实现右侧补 0 到 `L=max_seq_length`；`input_mask` 对真实 token 为 1、padding 为 0。掩码位置/标签补 0 到 `M=max_predictions_per_seq`，对应 `masked_lm_weights` 补 0.0。
- `info` 只用于评估 bookkeeping，没有传入 `BertModel`；模型实际接收 `input_ids` 与 `input_mask`，`token_type_ids=None`。
- 参考实现的 item ID 从 1 开始，0 专用于 padding；`[MASK]` 是词表中的额外 ID。特殊 token 的具体 ID 由词表构建决定，不能假定 `[MASK]` 等于某个固定常数。
- 作者代码默认脚本把 `mask_prob=1.0`，即选中的预测位置总是替换成 `[MASK]`；代码也支持保留原 item 或随机 item 的分支。不能把注释中的 BERT 80/10/10 直接写成所有官方实验的实际配置。
- 原始明细文件每行是 `user_id item_id`，实现假设同一用户的行已按交互顺序排列；`data_partition` 本身不读取时间戳或重新排序。
- 作者实现生成训练 TFRecord 前把验证 item 合并回历史，测试样本由“训练历史 + 验证 item + 测试 item”构成，再把最后的测试 item 替换为 `[MASK]`。因此测试 `masked_lm_ids[0]` 是 held-out 测试 item。
- 模型只在 `masked_lm_positions` 指定的位置 gather hidden state，输出 logits 的逻辑 shape 是 `[B*M, |vocab|]`；`masked_lm_weights` 排除补齐的伪目标。
- 原论文明确说明没有 segment embedding，并将类别、价格、演员等 item side features 与显式 user component 留作未来工作；原始 BERT4Rec 核心模型输入不是 `user_id`、时间戳或文本描述。
- 现代 PyTorch 生态没有单一统一字段标准：可用与输入等长的 dense `labels:[B,L]`，在非预测位置写 `ignore_index`（常见为 `-100`）；也可像作者 TensorFlow 代码一样传 `positions/ids/weights`，或像 RecBole/Transformers4Rec 一样由 `item_seq` 和 mask schema 在模型/任务层构造目标。
- RecBole 的 BERT4Rec 使用 `masked_item_seq:[B,L]`、`masked_index:[B,M]`、`pos_items:[B,M]`，BPR 模式另有 `neg_items:[B,M]`；模型只 gather 掩码位置的 hidden state。无效目标通过 `masked_index > 0` 的权重排除，而不是 `labels=-100`。
- RecBole 推理从 `item_seq:[B,L]` 与 `item_seq_len:[B]` 重建末尾 mask，输出全量候选分数 `[B,item_num]`；这证明 PyTorch 实现可以不暴露 `attention_mask` 字段，而由 `item_seq != padding_idx` 内部构造。
- Transformers4Rec 也把 `item_ids:[B,L]` 作为源输入，在 masking task 内部返回 boolean schema 与 targets；训练随机掩码，评估默认只预测最后 item。
- Hugging Face `DataCollatorForLanguageModeling` 是通用 NLP MLM 组件而非 BERT4Rec 标准；它支持 dense input/labels 和 `-100` 忽略语义，可作为自定义 PyTorch BERT4Rec 的实现方式，但必须明确是适配方案。
- Hugging Face 官方源码确认：collator 先复制输入为 `labels:[B,L]`，再执行 `labels[~masked_indices] = -100`，即只在选中位置计算损失。默认替换比例是 80% `[MASK]`、10% 随机 token、10% 保留原 token，但可配置为 100% `[MASK]`。
- 作者 TFRecord 的整数字段在磁盘中为 `int64`，读取后统一转为 `int32`；`masked_lm_weights` 保持 `float32`。PyTorch 通常使用 `torch.long`（int64）索引，dtype 也不是跨框架固定契约。
- `input_mask:[B,L]` 在模型内部广播成双向 attention mask `[B,L,L]`；它只屏蔽被 attend 的 padding 位置，不是 SASRec/GPT 的因果上三角 mask。

## 证据缺口

- 原论文与主流实现的核心语义已经核验；剩余差异应作为实现选择呈现，不再寻找不存在的统一字段标准。
