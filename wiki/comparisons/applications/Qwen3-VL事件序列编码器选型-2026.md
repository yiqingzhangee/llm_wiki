# Qwen3-VL 事件序列编码器选型（2026）

> 来源依据：[[来源-Qwen3-VL事件序列编码器调研-2026]]
>
> 统一接口：[[Qwen3-VL统计特征编码器选型-2026]]、[[Qwen异构特征接入与事件序列建模]]

## 一句话结论

约 8 步事件序列的首个工程版本使用：**固定描述离线编码 + 事件 ID/时间/payload 融合 + 单层 GRU + 最后有效状态读出**。主输出是 `[B,128]` 连续向量；只有接入 Qwen3-VL 时，才用独立 MLP 生成一个 `[B,1,2560]` soft token。

## 最小输入输出契约

| 项目 | 首版约束 |
|---|---|
| 事件描述 | 固定英文描述离线编码并版本化缓存 |
| 事件类型 | 可训练 ID embedding，保留任务特定语义 |
| 时间输入 | `log1p(delta_t)`、`log1p(age_t)`；顺序由 GRU 保留 |
| 动态属性 | 可选数值/枚举 payload，经小型 MLP/embedding 编码 |
| 序列编码器输出 | `sequence_vector:[B,128]` |
| 可选 Qwen 输出 | `sequence_token:[B,1,2560]` |
| padding | 按真实长度读取最后一个有效 hidden state |
| 第一阶段训练 | 冻结 Qwen，只训练事件融合层、GRU、projector 和任务头 |

## 最小结构

对第 `t` 个事件：

```text
s_t = Linear(description_cache[event_type_t])
i_t = EventIdEmbedding(event_type_t)
τ_t = Linear([log1p(delta_t), log1p(age_t)])
p_t = OptionalPayloadEncoder(payload_t)

event_t = LayerNorm(s_t + i_t + τ_t + p_t)
```

序列编码与读出：

```text
hidden_1...hidden_L = GRU(event_1...event_L)
sequence_vector = hidden_last_valid                 # [B,128]
```

连续向量可直接接分类头或 late fusion。Qwen 接入是独立可选步骤：

```text
sequence_token = tanh(g) * MLP_project(sequence_vector)  # [B,1,2560]
```

`g` 初始化为 0。首版不使用 attention pooling、learned query、Transformer、Mamba 或 TPP。

## 为什么这是最简单的合理实现

| 候选 | 代码与调试 | 顺序/变长 | 首版结论 |
|---|---|---|---|
| 文本模板 | 最低 | 由 Qwen 文本处理 | 必须保留的零编码器基线 |
| 固定长度展开 + MLP | 很低 | padding 与长度变化脆弱 | 不作为默认 |
| 单层 GRU | 低 | 原生支持 | 连续向量首选 |
| 浅层 Transformer | 中 | 需位置、mask 和时间 bias | GRU 达标后再比较 |
| Mamba/HSTU/TPP | 高 | 面向长序列或事件生成 | 当前不做 |

单层 GRU 不是组件数量绝对最少，而是在保留顺序、支持变长和减少边界代码之后，总工程成本最低。

## 与统计编码器的统一接口

两条链路保持对称：

```text
统计字段 → 字段级加法融合 → 两层 MLP → stat_vector [B,256]
事件序列 → 事件级加法融合 → 单层 GRU → sequence_vector [B,128]
```

连续向量先独立验收，可用于专用分类头和 late fusion。需要接入 Qwen 时分别投影：

```text
stat_vector     → stat_projector     → stat_token [B,1,2560]
sequence_vector → sequence_projector → sequence_token [B,1,2560]
```

首版保留两个独立占位 token，便于分别关闭统计或事件通道。不要先把两个向量混成一个 token，否则无法判断哪条模态有效。

## 最小消融

1. 事件文本模板；
2. GRU 连续向量直接分类；
3. Qwen + 单个事件 soft token；
4. 去掉描述语义；
5. 描述在事件类型间随机错配；
6. 去掉 `delta_t` 或 `age_t`；
7. 顺序打乱；
8. 统计 token、事件 token 和双 token 三组比较。

只有单层 GRU 在时间外、实体外切片稳定超过模板、聚合统计和直接基线后，才比较浅层时间感知 Transformer；Mamba、HSTU、TPP 和查询压缩不进入首轮。

若后续出现标签稀缺、多任务复用或离线向量库需求，可以再评估 [[事件序列编码器的自监督联合预训练]]。该路线以对比目标约束序列级表示，以掩码恢复目标保留事件信息。它是通过强基线后的训练升级，不改变首版优先使用监督 GRU 的结论。

## 训练目标成熟度门禁

模型能完成前向、计算损失并更新参数，只说明训练链路连通。若损失面向固定合成目标，它不等同于下一事件预测、掩码恢复或其他正式自监督任务。

进入骨干比较前，应先核对：

1. 目标标签是否由真实序列按预测时点构造；
2. 目标事件是否从输入与派生时间特征中排除；
3. 训练头、损失、样本清单和 cutoff 是否版本化；
4. GRU 与 Transformer 是否使用相同输入、划分、目标和读出；
5. 是否已有时间外探针或下游结果，而非只有训练 loss。

完整合同与增强风险见 [[来源-序列编码器训练合同与骨干对照内部分析-已脱敏-2026]]。

## 工程停止条件

- GRU 向量未超过事件无序聚合或文本模板；
- 顺序打乱、时间特征移除或描述错配后表现几乎不变；
- 单 token 接入没有超过连续向量 late fusion；
- 自定义 embedding 使生成或服务链路不稳定；
- 收益不足以覆盖 Qwen 推理成本。

## 数值一致性门禁

GRU 上线前除任务指标外，还要比较同一样本在单条与典型 batch shape 下的 `sequence_vector`。记录并分别控制：

```python
torch.backends.cuda.matmul.allow_tf32
torch.backends.cudnn.allow_tf32
```

两者互不替代。H20、PyTorch 2.7.1、CUDA 12.6 的一次内部对照中，单条与双条 batch 的偏差首次出现在 GRU；关闭 cuDNN TF32 后，最大绝对误差从 `2.38e-4`、`1.59e-4` 降至 `1.79e-7`、`1.49e-7`。这项结果用于增加工程门禁，不代表所有环境都必须关闭 TF32。诊断过程与证据边界见 [[GRU单条与Batch推理的cuDNN-TF32一致性]]。

验收至少覆盖单步、长序列、不等长 padding、checkpoint 回读和最终 vector 回读；同 batch 重复一致不能替代跨 batch shape 检查。
