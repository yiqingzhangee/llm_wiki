# 来源：BERT4Rec 原论文与参考实现（2019）

## 来源信息

- 论文：Fei Sun 等，*BERT4Rec: Sequential Recommendation with Bidirectional Encoder Representations from Transformer*
- 会议：CIKM 2019
- 论文：[arXiv:1904.06690v2](https://arxiv.org/abs/1904.06690v2)、[ACM DOI](https://doi.org/10.1145/3357384.3357895)
- 作者参考实现：[FeiSun/BERT4Rec](https://github.com/FeiSun/BERT4Rec)，核验 commit `615eaf2`
- PyTorch 对照：[RecBole BERT4Rec](https://recbole.io/docs/_modules/recbole/model/sequential_recommender/bert4rec.html)
- 任务层对照：[Transformers4Rec MaskedLanguageModeling](https://nvidia-merlin.github.io/Transformers4Rec/v0.1.6/api/transformers4rec.torch.html)
- 密集标签对照：[Hugging Face DataCollatorForLanguageModeling](https://huggingface.co/docs/transformers/main_classes/data_collator#transformers.DataCollatorForLanguageModeling)
- 核验日期：2026-10-28

## 证据层级

- CIKM 论文与作者实现用于定义 BERT4Rec 的训练目标、数据划分和原始 TensorFlow 字段，按 **E3/E4** 使用。
- RecBole、Transformers4Rec 和 Hugging Face 官方源码用于说明现代 PyTorch 生态的实现差异，按 **E4 工程证据**使用。
- 不存在统一的跨框架字段标准；稳定的是“离散 item 序列 + 双向 padding mask + 只在选中位置计算 Cloze 损失”的语义契约。

## 原论文定义的任务

原始交互按用户分组并按时间排序为 item ID 序列。训练时随机选择比例 `ρ` 的位置，将对应 item 替换为 `[MASK]`，利用左右上下文恢复原 item。序列超过最大长度 `N` 时只保留最近 `N` 个 item。

测试阶段在已知历史末尾追加 `[MASK]`，用该位置的最终 hidden state 对 item 集合打分，以预测下一 item。论文还加入“只掩码最后 item”的训练样本，缩小随机 Cloze 训练与 next-item 推理之间的目标差异。

论文实验使用 leave-one-out：

- 每个用户最后一个 item 为测试标签；
- 倒数第二个 item 为验证标签；
- 更早的 item 构成训练历史；
- ML-1m、ML-20m 使用 `N=200`，Beauty、Steam 使用 `N=50`。

这些长度与 mask 比例是实验配置，不是 BERT4Rec API 的固定常数。

## 作者 TensorFlow 参考实现

### 原始明细

输入文本每行是：

```text
user_id item_id
```

实现按文件中的出现顺序累积同一用户的 item；`data_partition` 不读取时间戳，也不执行排序。因此生产数据必须在进入该函数前确保顺序正确。

### TFRecord 字段

单样本写入下列字段，batch 后增加首维 `B`：

| 字段 | batch shape | 读取后 dtype | 语义 |
|---|---:|---|---|
| `info` | `[B,1]` | `int32` | 用户编号，仅用于评估 bookkeeping，不传入 BERT 主干 |
| `input_ids` | `[B,L]` | `int32` | item、`[MASK]` 与 padding ID |
| `input_mask` | `[B,L]` | `int32` | 真实 token 为 1，右侧 padding 为 0 |
| `masked_lm_positions` | `[B,M]` | `int32` | 需要预测的序列位置 |
| `masked_lm_ids` | `[B,M]` | `int32` | 被替换位置的原始 item ID |
| `masked_lm_weights` | `[B,M]` | `float32` | 有效目标为 1.0，目标 padding 为 0.0 |

其中：

- `L = max_seq_length`；
- `M = max_predictions_per_seq`；
- item ID 从 1 开始；
- 0 用于 padding；
- `[MASK]` 是词表中的额外 ID，具体数值由词表构建决定。

TFRecord 只能保存 `int64` 整数字段，读取函数随后统一转为 `int32`。这只是 TensorFlow 1.x 实现细节；PyTorch embedding 索引通常使用 `torch.long`。

### 损失路径

模型先产生 `sequence_output:[B,L,H]`，再根据 `masked_lm_positions` gather 得到 `[B,M,H]`，计算 item logits `[B,M,V]`。代码内部将前两维展平为 `[B*M,V]`，并以 `masked_lm_weights` 排除补齐的伪目标。

`input_mask:[B,L]` 会广播为双向 attention mask `[B,L,L]`。它屏蔽 padding key，不是 SASRec 或 GPT 使用的因果上三角 mask。

## 实现差异

### 作者代码的替换策略

作者脚本默认 `mask_prob=1.0`，选中的目标位置全部替换为 `[MASK]`。代码还保留“原 item 不变”或“换成随机 item”的分支，但默认实验脚本并非固定采用 BERT NLP 常见的 80/10/10。

### RecBole

RecBole 使用：

- `masked_item_seq:[B,L]`
- `masked_index:[B,M]`
- `pos_items:[B,M]`
- BPR 模式下的 `neg_items:[B,M]`

无效目标由 `masked_index > 0` 的权重排除。推理时由 `item_seq:[B,L]` 与 `item_seq_len:[B]` 在最后有效位置重建 mask，并输出 `[B,item_num]` 的全量候选分数。

### PyTorch 密集标签

自定义 PyTorch 实现也可采用 Hugging Face 风格：

- `input_ids:[B,L]`
- `attention_mask:[B,L]`
- `labels:[B,L]`

`labels` 在被选中的位置保存原 item ID，其余位置写 `-100`，交给 `CrossEntropyLoss(ignore_index=-100)` 忽略。该格式语义上等价于稀疏 `positions/ids/weights`，但它是实现选择，不是原论文规定的字段名。

## 原始模型不直接接收什么

原始 BERT4Rec 主干只建模 item ID 与 position：

- `user_id` 不进入模型；作者代码中的 `info` 只用于评估；
- 时间戳用于离线排序，不作为模型张量；
- 没有文本描述、类别、价格等 item side features；
- 没有 segment embedding，`token_type_ids=None`。

论文把 item side features 与显式 user component 列为未来方向。后续扩展可以加入这些信息，但应明确已偏离原始输入契约。

## 证据边界

- 原论文描述模型语义，作者代码提供具体字段；二者不能反推所有现代实现必须使用同名张量。
- `N`、`ρ`、padding 方向、替换比例和目标表达都可因数据与框架变化。
- next-item 测试需要严格的时间切分。先构造全序列再随机切分会泄漏未来交互。

## 关联页面

- [[BERT4Rec数据处理与输入张量]]
- [[LLM推荐系统的分层改造与评估]]
- [[Qwen异构特征接入与事件序列建模]]
- [[自回归掩码、解码与生成评测]]
