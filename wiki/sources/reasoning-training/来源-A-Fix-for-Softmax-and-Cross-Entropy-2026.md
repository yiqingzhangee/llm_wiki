# 来源：A Fix for Softmax and Cross-Entropy（2026）

## 来源信息

- 论文：Jonathan T. Barron、Dan Gnanapragasam、Michael Tschannen，*A Fix for Softmax and Cross-Entropy*
- 机构：Google DeepMind
- 当前版本：[arXiv:2610.31642v2](https://arxiv.org/abs/2610.31642v2)，2026-10-26 修订
- 作者 HTML 版：[jonbarron.info/softmaxfix](https://jonbarron.info/softmaxfix/)
- 作者发布说明：[Jon Barron 的公开帖子](https://x.com/jon_barron/status/2113707875290042665)
- 框架跟踪：[PyTorch issue #201986](https://github.com/pytorch/pytorch/issues/201986)、[PyTorch draft PR #202049](https://github.com/pytorch/pytorch/pull/202049)、[JAX issue #40384](https://github.com/jax-ml/jax/issues/40384)、[JAX issue #40386](https://github.com/jax-ml/jax/issues/40386)
- 核验日期：2026-10-28

## 证据层级

- 论文为作者报告的 arXiv 预印本，作为机制、公式和实验的一手来源，按 **E3** 使用，但尚不能替代同行评审和大规模训练复现。
- PyTorch/JAX 仓库、issue、PR 与源码用于判断框架状态，属于 **E4** 工程一手证据。
- Gemini 分工来自第一作者发布帖，属于可归因的**作者公开声明**；它不在论文正文、TeX 或致谢中，不能与技术实验结论混为一类证据。

## 核心问题

标准实现先减去最大 logit 以防指数溢出，但这只解决了 overflow，没有解决高置信度和近相等概率下的有效位损失。

### Softmax Collapse

对 hard target \(t\)，定义 margin：

\[
m=z_t-\log\sum_{j\ne t}e^{z_j}.
\]

标准分母包含 \(1+e^{-m}\)。当 \(e^{-m}\) 小到不能改变浮点数 `1.0` 时：

- 目标概率舍入为 1，目标梯度 \(q_t-1\) 变成 0；
- 非目标梯度仍在 0 附近直接表示为正小数；
- 原本应满足的 \(\sum_k g_k=0\) 被破坏。

论文报告的 margin 精度悬崖约为 bf16 的 5.5、fp32 的 16.6、fp64 的 36.7。这不是指数下溢；fp32 的指数小量可继续表示到约 margin 88。

### Softmax Cancellation

soft-target 蒸馏通常分别计算教师概率 \(p=\operatorname{softmax}(z_T)\) 与学生概率 \(q=\operatorname{softmax}(z_S)\)，再用 \(q-p\) 得到梯度。学生与教师接近时，这会：

- 先把两个近似值分别舍入，再相减，丢失原始 logit 差；
- 因 forward softmax 与 backward log-softmax 的运算路径不同，在 \(z_S=z_T\) 时也可能留下非零梯度。

## 修复方法

### hard target

将最大项或目标项从指数和中排除，单独计算非目标 margin：

\[
\mathcal L(z,t)=\operatorname{softplus}(-m).
\]

这样不再先形成 `1 + 极小数`。目标 logit 只通过 \(z_j-z_t\) 进入损失，自动微分会把目标梯度计算为非目标梯度之和的相反数，从而保持零和不变量。

### soft target

损失函数需要接收教师 logits，而不只是已舍入的教师概率。论文从 \(\delta=z_S-z_T\) 出发，在教师分布下中心化，再用 `expm1` 和 `log1p` 在对数空间构造 \(q-p\)。v2 还在差异较大时回退到普通 \(q-p\)，避免 `0 * inf`，最后显式消除残余求和误差。

论文正文给出 JAX 参考实现，但没有独立代码仓库。三个 hard-target 函数与 soft-target 自定义梯度函数合计超过“十几行”；“约十几行”只适合粗略描述其中一个函数。

## 实验结果

### Slingshot 与 grokking

- 在 `a + b mod 31`、带权重衰减的 1 层 Transformer 上，修复版跨过 fp32 精度悬崖而不产生周期性 loss spike，5 个随机种子均更快 grok。
- 在 `a^2 + b^2 mod 97`、无权重衰减设置中，标准交叉熵到达 fp32 悬崖后测试准确率从 61% 降到 23%，修复版继续升至 85%。
- 四个 modular arithmetic 任务上，fp32 修复版与 fp64 标准版走势一致并达到 100% 测试准确率。

论文提出的机制是：Collapse 使目标梯度消失，Adam 的二阶矩随之衰减；梯度重新出现时，被很小的分母放大为大步更新。修复保留连续的小梯度后，尖峰消失。

这支持“论文测试设置中的 Slingshot 由 Collapse 导致”，但不应扩张为“任何训练 loss spike 都由它导致”。

### soft-target 蒸馏

学生从教师的同一组 fp32 权重初始化，理论梯度应为零：

- 标准 bf16 soft-target CE 的残余梯度约为 \(10^{-3}\)，学生明显漂移并损失准确率；
- 标准 fp32 出现 KL 尖峰和约 \(10^{-6}\) 的地板；
- 修复版在 bf16、fp32、fp64 下均保持 KL 为 0 和 100% 准确率。

这是刻意构造的“学生等于教师”压力测试，不等同于普通远距离师生蒸馏的总体收益评估。

## Gemini 协作声明

第一作者在发布帖中声明：

- Gemini Flash 3.7 首先发现并修复问题；
- Gemini Flash 3.8 完成论文中的多数实验；
- Gemini 4 Argon 写出论文与图表初稿；
- 作者随后反复复核内容、用旧论文提供文风样本，并最终人工改写。

论文 v1/v2 正文和致谢没有记录这段分工。因此，本页只把它作为可归因的研究过程说明，不将其作为模型能力的独立评测证据。

这里还存在一处归因张力：论文 v2 明确写 Softmax Collapse 由 Prieto et al.（2025）先识别，而作者发布帖使用了“this bug was first found and fixed by Gemini Flash 3.7”的表述。在没有进一步澄清前，更稳妥的理解是：Gemini 在本项目中发现并形成了完整修复，不能据该帖子否定更早文献已识别 Collapse。

## 框架状态（截至 2026-10-28）

| 框架 | 状态 | 证据与边界 |
|---|---|---|
| PyTorch | 尚未正式合入论文修复 | `#201986` 仍 open；`#202049` 是未合并的 draft PR。HEAD 只合入了 helper 重构 `#202240`，仍使用包含最大项的指数和 |
| JAX | HEAD 尚未采用论文公式 | 2026-10-27 HEAD 仍计算 `log(sum(exp(shifted)))`，softmax 仍对包含 `exp(0)=1` 的总和归一化 |
| Optax | 未发现论文编号或术语对应的官方 issue/PR | 论文示例表明当时的 integer-label CE 仍受影响；未检索到后续合入证据 |

PyTorch 草稿 PR 也只先覆盖 CPU/CUDA 的 Collapse 路径；MPS、XPU、稀疏实现、decomposition/refs、`torch.compile` 等被明确列为未覆盖，Softmax Cancellation 也不是该 PR 的完整目标。

## 可迁移边界

- 主要实验证据来自小型 Transformer/MLP 的 modular arithmetic grokking，而不是标准大规模预训练。
- 论文明确承认，极端 margin 和几乎相同的师生 logits 在常规大规模训练中并不常见。
- fp32 能把精度悬崖推后，但不能消除同类结构；单纯升精度不是数学路径上的根治。
- 线上采用自定义 loss 前仍需验证 label smoothing、ignore index、class weight、分布式归约、编译器、后端、二阶梯度和性能。

## 关联页面

- [[Softmax与交叉熵的浮点数值稳定性]]
- [[知识蒸馏]]
- [[参数高效微调与对齐训练]]
- [[证据分级与引用规范]]
