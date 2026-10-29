# BERT4Rec 数据处理与输入张量

> 核心来源：[[来源-BERT4Rec原论文与参考实现-2019]]

## 核心结论

BERT4Rec 接收**按时间排序的离散 item ID 序列**。训练时随机遮住部分 item，再用双向注意力恢复它们。`input_ids`、`masked_lm_positions` 和 `labels` 是不同实现对这套训练语义的编码。

数据处理分为三层：

1. 原始交互记录；
2. 截断、padding、mask 后的训练样本；
3. 实际传给模型与损失函数的张量。

## 记号

| 符号 | 含义 |
|---|---|
| `B` | batch size |
| `L` | 最大序列长度 |
| `M` | 每个样本最多预测的位置数 |
| `V` | 模型词表大小，包含 item 与 padding / mask 等特殊 ID |
| `H` | hidden size |

## 从原始数据到样本

### 原始记录

最少需要：

```text
user_id, item_id, event_time
```

离线处理顺序：

```text
按 user 分组
→ 按 event_time 排序
→ 按时间划分 train / validation / test
→ 截取最近 L 个 item
→ 映射到连续 item ID
→ 生成 mask 与标签
→ padding / batch
```

`event_time` 对无泄漏排序和切分是必要的，但原始 BERT4Rec 模型不直接接收时间戳。

## 训练张量：作者 TensorFlow 格式

| 张量 | shape | dtype | 内容 |
|---|---:|---|---|
| `input_ids` | `[B,L]` | 整数 | 已替换部分位置的 item ID 序列，含 `[MASK]` 与 padding |
| `input_mask` | `[B,L]` | 0/1 整数 | 非 padding 为 1，padding 为 0 |
| `masked_lm_positions` | `[B,M]` | 整数 | 被选为预测目标的位置 |
| `masked_lm_ids` | `[B,M]` | 整数 | 这些位置原来的 item ID |
| `masked_lm_weights` | `[B,M]` | 浮点 | 有效目标为 1，补齐目标为 0 |
| `info` | `[B,1]` | 整数 | 用户编号元数据，不进入模型主干 |

示例设 `L=8`、`M=3`、`PAD=0`、`MASK=1001`：

```text
原序列:                 [12, 7, 31, 9, 44]
选中位置:               [1, 3]

input_ids:              [12, 1001, 31, 1001, 44, 0, 0, 0]
input_mask:             [ 1,    1,  1,    1,  1, 0, 0, 0]
masked_lm_positions:    [ 1,    3,  0]
masked_lm_ids:          [ 7,    9,  0]
masked_lm_weights:      [1.0,  1.0, 0.0]
```

第三个 position 的 0 只是目标数组 padding；`masked_lm_weights=0` 保证它不参与损失。

## 训练张量：PyTorch 密集标签格式

自定义 PyTorch 实现可以把标签展开到每个序列位置：

| 张量 | shape | 推荐 dtype | 内容 |
|---|---:|---|---|
| `input_ids` | `[B,L]` | `torch.long` | 已 mask 的 item 序列 |
| `attention_mask` | `[B,L]` | `bool` 或整数 | 非 padding 为真/1 |
| `labels` | `[B,L]` | `torch.long` | 目标位置为原 item ID，其余为 `-100` |

与上例等价的 dense labels：

```text
input_ids:      [12, 1001, 31, 1001, 44,   0,   0,   0]
attention_mask: [ 1,    1,  1,    1,  1,   0,   0,   0]
labels:         [-100,  7, -100,   9, -100, -100, -100, -100]
```

损失可写为：

```python
loss = cross_entropy(
    logits.reshape(-1, vocab_size),
    labels.reshape(-1),
    ignore_index=-100,
)
```

这是 Hugging Face 风格的工程适配，不是 BERT4Rec 原论文规定的唯一接口。

## 训练张量：RecBole 风格

RecBole 保留稀疏目标，但字段名不同：

| 张量 | shape | 语义 |
|---|---:|---|
| `masked_item_seq` | `[B,L]` | 已 mask 的 item 序列 |
| `masked_index` | `[B,M]` | 目标位置 |
| `pos_items` | `[B,M]` | 原始 item ID |
| `neg_items` | `[B,M]` | BPR loss 使用的负 item |

某个实现没有 `labels` 或 `attention_mask` 字段，不代表输入不完整。它可能在模型内部根据 padding ID 与目标位置构造这些信息。

## 模型内部 shape

典型 forward 路径：

```text
input_ids                     [B,L]
item_embedding                [B,L,H]
position_embedding            [B,L,H]
双向 Transformer 输出         [B,L,H]
gather masked positions       [B,M,H]
item logits                   [B,M,V]
```

作者 TensorFlow 实现为计算损失，会把 logits 展平为 `[B*M,V]`。padding mask 在内部广播为 `[B,L,L]` 的双向 attention mask，**不是因果 mask**。

## 验证、测试与在线推理

### Leave-one-out

原论文实验按用户序列：

```text
[训练历史..., validation_item, test_item]
```

- 验证：用更早历史预测 `validation_item`；
- 测试：把 `validation_item` 纳入历史，预测 `test_item`；
- 任何样本都不能使用其预测时点之后的交互。

### 推理张量

推理时在历史末尾放置 `[MASK]`：

```text
history:        [12, 7, 31, 9, 44]
input_ids:      [12, 7, 31, 9, 44, 1001, 0, 0]
attention_mask: [ 1, 1,  1, 1,  1,    1, 0, 0]
predict_index:  5
```

取 `predict_index` 的 hidden state，与所有候选 item embedding 计算分数：

```text
hidden_at_mask [B,H]
item_table     [V,H]
scores         [B,V]
```

线上还要过滤 padding、`[MASK]`、不可推荐 item 和库存或政策不合格的 item。是否过滤已交互 item 取决于产品规则。

## 原始 BERT4Rec 不直接包含的输入

| 字段 | 原始模型是否直接接收 | 说明 |
|---|---|---|
| `user_id` | 否 | 作者代码只用于评估 bookkeeping |
| 时间戳 | 否 | 用于排序、切分；不进入主干 |
| 文本描述 | 否 | 原始模型只使用 item ID |
| 类别、价格、图像 | 否 | 论文列为未来可扩展的 side features |
| `token_type_ids` | 否 | 单序列任务，没有 segment embedding |
| 因果 attention mask | 否 | 编码器使用双向注意力，仅屏蔽 padding |

若把事件类型映射为 item ID，可以复用 BERT4Rec 的张量契约；加入时间间隔、payload 或文本描述后，应明确这是 BERT4Rec 扩展，而不是原始模型输入。

## 推荐的项目接口

若自行实现 PyTorch 版本，优先采用简单、容易单测的 dense 接口：

```python
batch = {
    "input_ids": LongTensor[B, L],
    "attention_mask": BoolTensor[B, L],
    "labels": LongTensor[B, L],  # 非目标位置为 -100
}
```

将 `user_id`、原始时间、样本截止时间保留为 batch metadata，用于切分审计、评估和问题回溯，但不要未经设计就加到 item embedding。

推理接口建议显式返回：

```python
{
    "input_ids": LongTensor[B, L],
    "attention_mask": BoolTensor[B, L],
    "prediction_index": LongTensor[B],
}
```

显式返回 `prediction_index`，可以避免依赖“最后一列一定是 mask”的假设。

## 常见错误

- 先构造滑窗再随机切分，导致同一用户的未来 item 泄漏到训练；
- `labels` 填入 `[MASK]` ID，而不是被遮住的原 item ID；
- padding 位置参与损失；
- 把双向 padding mask 写成因果 mask，退化为另一类序列模型；
- 推理时忘记把末尾 `[MASK]` 计入 `attention_mask`；
- item、padding、mask 共用或越界 ID；
- 把论文实验的 `L=50/200`、某个 `ρ` 或 80/10/10 当成固定标准；
- 候选打分时把特殊 token 当成可推荐 item；
- 只有 `input_ids` 形状测试，没有验证 mask、标签与 cutoff 的语义。

## 最小验收

1. 同一原序列能转换成稀疏标签与 dense 标签，两种 loss 数值一致；
2. 非目标位置改变 logits 不影响 loss；
3. padding 数量变化不影响有效目标 loss；
4. 推理 `[MASK]` 位于最后有效位置，输出 shape 为 `[B,V]`；
5. 任一训练样本中的最大时间不晚于其 `prediction_cutoff`；
6. item ID、padding ID、mask ID 和输出候选范围互不冲突。

## 关联页面

- [[来源-BERT4Rec原论文与参考实现-2019]]
- [[事件序列编码器的自监督联合预训练]]
- [[LLM推荐系统的分层改造与评估]]
- [[Qwen异构特征接入与事件序列建模]]
- [[自回归掩码、解码与生成评测]]

## 主要来源

- [BERT4Rec 原论文](https://arxiv.org/abs/1904.06690v2)
- [作者 TensorFlow 参考实现](https://github.com/FeiSun/BERT4Rec)
- [RecBole BERT4Rec](https://recbole.io/docs/_modules/recbole/model/sequential_recommender/bert4rec.html)
- [Transformers4Rec MaskedLanguageModeling](https://nvidia-merlin.github.io/Transformers4Rec/v0.1.6/api/transformers4rec.torch.html)
- [Hugging Face DataCollatorForLanguageModeling](https://huggingface.co/docs/transformers/main_classes/data_collator#transformers.DataCollatorForLanguageModeling)
