# Softmax 与交叉熵的浮点数值稳定性

> 核心来源：[[来源-A-Fix-for-Softmax-and-Cross-Entropy-2026]]

## 结论

“先减最大 logit”只保证 softmax 不容易指数溢出，并不保证 loss 和梯度在高置信度或师生接近时仍准确。需要分别防止：

1. **Softmax Collapse**：目标梯度被舍入为零，但非目标梯度仍非零；
2. **Softmax Cancellation**：两个近似概率相减产生不应存在的残余梯度。

共同原则不是“全部换高精度”，而是直接计算所需的小量，避免先形成 `1 + 极小数` 或先舍入两个近似值再相减。

## 两类失效

| 问题 | 典型条件 | 错误表现 | 最强诊断不变量 |
|---|---|---|---|
| Softmax Collapse | hard target 已高度置信，目标 margin 超过精度悬崖 | 目标 loss/梯度为 0，非目标梯度仍为正 | `sum(grad_logits) ≈ 0` 被破坏 |
| Softmax Cancellation | soft target 中学生 logits 接近教师 logits | 理论梯度趋近 0，实际停在噪声地板；相等时仍可非零 | `z_student == z_teacher` 时梯度应严格为 0 |

精度悬崖是 dtype 与归约路径的函数。论文给出的典型 margin 是 bf16 约 5.5、fp32 约 16.6、fp64 约 36.7，不能把这些值当成所有硬件、编译器和 kernel 的绝对常数。

## 为什么梯度必须零和

给所有 logits 加同一个常数不会改变 softmax，因此交叉熵沿全 1 方向的导数应为零：

\[
\sum_k \frac{\partial \mathcal L}{\partial z_k}=0.
\]

这既是数学不变量，也是便宜的运行时诊断信号。若高置信度样本上长期明显偏离零，应进一步检查：

- target gradient 是否被置零；
- 非 target 梯度是否仍处于可表示范围；
- loss、softmax 和 backward 是否使用不同精度或不同 kernel；
- 编译后路径是否绕过了自定义实现。

注意：接近机器误差的微小偏差并不能单独证明 Collapse；应结合 margin、各梯度分量和高精度参考判断。

## 工程排查清单

### hard target 分类

记录或抽样计算：

- 正确类 margin；
- `loss == 0` 的样本比例；
- 目标梯度为 0 但非目标梯度绝对和大于 0 的比例；
- 每个样本的 `abs(sum(grad_logits))`；
- loss spike 前后 Adam/AdamW 的二阶矩与参数步长。

最小压力测试可用 fp32 logits `[9, -9, -9]`、target `0`。论文报告标准 JAX/Optax 路径返回 0 loss 和非零和梯度，而修复路径返回约 `3.046e-8` 的 loss 与零和梯度。

### soft-target 蒸馏

构造三档测试：

1. `z_student == z_teacher`，梯度必须为 0；
2. 学生 logits 加一个接近 dtype 分辨率的小扰动，与 fp64 参考比较方向和大小；
3. 真实 batch 上扫描师生 logit gap 与梯度噪声地板的关系。

只观察 KL 值不够；Adam 可能把很小的残余梯度放大成可见的参数漂移。

## 修复与回退策略

| 优先级 | 方案 | 适用性 | 限制 |
|---|---|---|---|
| 1 | 使用框架已正式发布且覆盖当前后端的稳定实现 | 长期默认 | 截至 2026-10-28，PyTorch/JAX 尚未完整合入 |
| 2 | hard target 用“排除目标项的 log-sum-exp + softplus” | 分类、高置信度训练 | 需覆盖权重、平滑、ignore index、编译和后端语义 |
| 3 | soft target 直接接收教师 logits，用 `expm1`/`log1p` 构造差异 | 蒸馏、KL 正则 | 仅保存教师概率时无法完整恢复 logit 差 |
| 4 | 临时把 loss 计算提升到 fp32/fp64 | 快速缓解低精度问题 | 只推迟悬崖，增加成本，不是根治 |
| 5 | 梯度零和投影 | 应急实验 | 论文实验显示它不能恢复全部丢失信息，不能替代稳定公式 |

## 验收门禁

采用自定义实现前，至少验证：

- forward 与 fp64 参考在普通、极端、并列最大值、`inf/NaN` 输入上的一致性；
- 一阶梯度零和，以及有限差分/高精度参考；
- label smoothing、soft labels、class weights、ignore index 与各 reduction；
- bf16、fp16、fp32、fp64；
- eager、编译模式、CPU、GPU 和实际部署后端；
- JVP、二阶梯度和分布式归约，如训练链路确实使用；
- 训练吞吐、显存和 kernel 回退。

不要因为 eager CPU 单测通过，就推断 `torch.compile`、MPS/XPU 或自定义 fused CE 同样通过。

## 对训练现象的解释边界

- **可支持**：在论文的 modular arithmetic 设置中，Collapse 解释了精度悬崖附近的 Slingshot，并且稳定公式消除了尖峰。
- **不可直接支持**：所有大模型 loss spike、所有 grokking、所有蒸馏退化都由这两个问题导致。
- **可迁移判断**：高置信度分类、低精度训练、教师学生很接近、Adam/AdamW 放大小梯度时，排查优先级应提高。
- **仍需实验**：长尾分类是否更常触发取决于其 margin 分布和损失设计，不能仅由类别不平衡直接推出。

## 框架跟踪

截至 2026-10-28：

- PyTorch 有开放 issue 和未合并的 CPU/CUDA draft PR，尚不能称为官方已修复；
- JAX HEAD 仍使用标准归约公式；
- 需要按具体版本、后端和编译路径冻结结论，而不是只写框架名。

最新细节与链接见 [[来源-A-Fix-for-Softmax-and-Cross-Entropy-2026#框架状态（截至 2026-10-28）]]。

## 关联页面

- [[知识蒸馏]]：Cancellation 是 soft-target 蒸馏的直接风险。
- [[参数高效微调与对齐训练]]：训练稳定性不仅取决于优化算法，也取决于 loss kernel。
- [[证据分级与引用规范]]：框架状态需要官方源码和 PR 支撑。
