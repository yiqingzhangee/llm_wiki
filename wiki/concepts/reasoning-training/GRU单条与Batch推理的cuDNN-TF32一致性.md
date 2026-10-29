# GRU 单条与 Batch 推理的 cuDNN TF32 一致性

> 诊断来源：[[来源-GRU单条与Batch推理一致性内部诊断-2026]]
>
> 应用背景：[[Qwen3-VL事件序列编码器选型-2026]]、[[Qwen异构特征接入与事件序列建模]]

## 问题定义

同一模型权重和同一份样本，分别以单条和 batch 形式送入模型时，输出不一定逐位相同。batch shape 可能改变底层 kernel、运算顺序或精度路径，因此“同一 batch 重复执行完全一致”只能排除运行间随机漂移，不能证明单条与 batch 数值等价。

一次限定环境的诊断给出了具体实例：

| 项目 | 观测 |
|---|---|
| 环境 | H20、PyTorch 2.7.1、CUDA 12.6 |
| 输出 | 每条样本 128 维 |
| 比较容差 | `atol=1e-5, rtol=1e-4` |
| 超标维度 | 两条样本分别为 96、88 |
| 最大绝对误差 | `2.38e-4`、`1.59e-4` |
| 重复性 | 同一 batch 重复执行时，各层输出完全一致 |

这里的一致性失败不是“结果每次都变”，而是不同 batch shape 下出现了稳定、可复现的数值偏差。

## 逐层定位

诊断按输入到输出逐层比较：

1. 语义、时间和次数输入完全一致；
2. 投影层输出在原容差内；
3. LayerNorm 输出在原容差内；
4. 首次超标出现在 GRU；
5. 将同一份归一化结果直接送入 GRU，偏差仍然出现；
6. 两条样本都只有一个有效时间步。

前五步把充分原因收缩到 GRU 计算路径；第六步排除了“长序列递归逐步放大误差”作为本次现象的必要解释。

## TF32 对照

基线只关闭了普通 CUDA matmul 的 TF32：

```python
torch.backends.cuda.matmul.allow_tf32 = False
```

但 cuDNN 仍可独立使用 TF32：

```python
torch.backends.cudnn.allow_tf32 = True
```

在权重、输入和容差不变的情况下，仅关闭 cuDNN TF32：

```python
torch.backends.cudnn.allow_tf32 = False
```

两条样本的最大绝对误差降至 `1.79e-7` 和 `1.49e-7`，全部通过 `atol=1e-5, rtol=1e-4`。六组对照整体支持：在本次 H20、PyTorch 2.7.1、CUDA 12.6 环境中，单条与 batch 的差异由 GRU 的 cuDNN TF32 计算路径触发。

这不是所有 GRU、GPU 或 CUDA 版本的普遍结论。当前实验没有测量具体使用了哪个 cuDNN kernel，也没有把误差定位到 reset、update 或 new gate。

## 工程修复

Sequence smoke 训练入口已在模型构建和优化器更新前关闭 cuDNN TF32，并覆盖 HDFS 与 NAS 两条输入路径。设置应在任何可能触发模型计算的步骤之前执行，且不能假定关闭 matmul TF32 会连带关闭 cuDNN TF32。

本地回归结果为：

- `720 passed, 1 skipped`；
- Ruff 通过；
- 代码尚未提交或推送。

## 一致性验收矩阵

| 维度 | 最低检查 |
|---|---|
| batch shape | 单条、双条及线上典型 batch |
| 序列长度 | 单步、长序列、不等长序列及 padding |
| 精度配置 | 记录 matmul TF32 与 cuDNN TF32 两个开关 |
| 软件环境 | 记录 PyTorch、CUDA、cuDNN 和驱动版本 |
| 数据类型 | 记录输入、权重、自动混合精度与累加 dtype |
| 持久化链路 | 训练后 checkpoint 回读与最终 vector 回读 |
| 重复性 | 同 shape 重复执行与跨 shape 对比都要检查 |

尚未完成的验证包括：重新训练后的 H20 完整 smoke、checkpoint/vector 回读，以及更长和不等长序列。完成这些检查前，只能认定局部回归和直接 GRU 对照通过，不能认定完整训练—保存—加载—推理链路已经闭合。

## 可复用判断

- `torch.backends.cuda.matmul.allow_tf32=False` 与 `torch.backends.cudnn.allow_tf32=False` 是两个独立控制面。
- 确定性和跨 batch shape 数值一致性是不同问题，应分别测试。
- 首次超标层比最终向量差异更适合定位问题；应保存逐层输入输出，而不是只比较末端 embedding。
- 短序列也可能因 kernel 或精度路径产生 batch 相关差异，不能把 GRU 偏差默认归因于递归累积。
- 历史任务若没有记录精度开关，只能保留“与当前机制一致”的判断，不能追认完全相同的根因。
