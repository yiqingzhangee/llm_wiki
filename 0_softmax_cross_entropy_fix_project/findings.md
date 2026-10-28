# Softmax 与交叉熵数值修复调研发现

## 待核验主张

- arXiv:2610.31642 的标题、作者、机构与发布时间。
- Softmax Collapse 与 Softmax Cancellation 的正式定义和触发条件。
- Gemini 模型参与发现、实验与写作的作者声明。
- `1 + ε` 浮点结构、梯度和为零不变量、Slingshot 的因果关系。
- JAX 参考实现及 PyTorch/JAX 官方合入状态。

## 已确认

- 现有 Wiki 没有独立的 softmax、log-softmax 或交叉熵浮点稳定性页面。
- 直接关联页面为 `知识蒸馏.md` 与 `参数高效微调与对齐训练.md`。
- 论文来源应归入 `wiki/sources/reasoning-training/`，机制与工程门禁应归入独立概念页。
- arXiv:2610.31642 的标题为 *A Fix for Softmax and Cross-Entropy*，作者为 Jonathan T. Barron、Dan Gnanapragasam、Michael Tschannen，三人机构均为 Google DeepMind。
- arXiv API 在 2026-10-28 返回 v2（更新于 2026-10-26），而部分摘要/HTML入口仍缓存为 v1（提交于 2026-10-22）；最终来源页应以 v2 为当前版本并注明版本时间。
- arXiv 摘要明确区分两个问题：Softmax Collapse 会在目标 logit 显著占优时把其梯度舍入为零，使 logit 梯度之和不再为零；Softmax Cancellation 会在学生与教师 logits 接近时产生非零梯度，甚至二者相等时仍然如此。
- 摘要声称修复可消除先前观察到的 Slingshot、加快带权重衰减的 grokking、避免无权重衰减时的准确率崩溃，并阻止蒸馏中的学生偏离教师；具体实验边界仍需阅读全文核验。
- 论文给出的精度悬崖约为：bf16 的 margin 5.5、fp32 的 16.6、fp64 的 36.7；这来自 `1 + exp(-m)` 中小项被舍入，而非指数下溢（后者约在 margin 88）。
- hard-target 修复把目标类排除出 log-sum-exp，计算 margin 后返回 `softplus(-m)`；自动微分由非目标 logit 差值产生目标梯度，因此保持梯度和为零。
- Softmax Cancellation 的修复不是简单地对两个 softmax 结果相减，而是直接接收教师 logits：令 `delta = z_S - z_T`，在教师分布下中心化后用 `log1p`/`expm1` 构造 `log(q/p)` 和 `q-p`，再投影掉残余求和误差；学生与教师 logits 相等时梯度严格为零。
- v2 的 soft-target 实现新增了 `|s| >= 1` 时回退到 `q-p` 的保护，避免教师概率舍入为零同时 `expm1` 溢出导致 `0 * inf`；因此引用代码时应以 v2 为准。
- v1/v2 TeX 均未出现 Gemini、Flash、Argon、语言模型生成或 AI 辅助写作声明，致谢只列出七位人类顾问。论文也未提供 GitHub/代码仓库链接，参考实现直接嵌在正文。
- 但作者 Jonathan T. Barron 在论文发布帖中公开声明：Gemini Flash 3.7 首先发现并修复该 bug，Flash 3.8 完成论文多数实验，Gemini 4 Argon 写出论文和图表初稿；作者随后反复挑错、用其旧论文做风格重写，并最终人工改写。该信息应标为“作者公开声明”，不能写成论文正文已证明的技术结果。
- 归因存在张力：v2 正文称 Softmax Collapse 由 Prieto et al.（2025）先识别，作者帖子又称 “this bug was first found and fixed by Gemini Flash 3.7”。在未获澄清前，只能稳健表述为 Gemini 在该项目中发现并给出完整修复，不能据此抹去更早文献的识别优先权。
- 作者 Jonathan T. Barron 的个人主页将工作概括为：grokking 中的 slingshot 由 cross-entropy/softmax 各实现共有的数值 bug 引起，且另一个 bug 影响蒸馏；主页未提供框架合入状态或 Gemini 分工声明。
- PyTorch issue `#201986`（2026-10-25 创建）请求实现论文方案，截至 2026-10-28 仍为 open、triaged、无 milestone。
- PyTorch 草稿 PR `#202049`（2026-10-26 创建）实现 CPU/CUDA `log_softmax` 的 Softmax Collapse 修复，截至 2026-10-28 未合并；PR 明确列出 MPS、XPU、Vulkan、Metal、稀疏实现、decomposition/refs、`torch.compile` 等尚未覆盖，因此即使该 PR 后续合入，也不能泛化为全部后端和 Softmax Cancellation 均已修复。
- JAX issue `#40384`（`log_softmax`）与 `#40386`（`softmax`）在 JAX 0.10.1 上复现了 float64 可表示梯度分量被置零、VJP 不再零和的问题；二者于 2026-10-09 以 duplicate 关闭。关闭原因指向的主 issue 和实际修复状态仍需核验，不能把 “closed” 等同于 “fixed”。
- JAX 仓库 2026-10-27 HEAD（`70b48fe`）源码仍计算 `shifted_logsumexp = log(sum(exp(shifted)))`；`softmax` 也仍以包含最大项 `exp(0)=1` 的总和为分母，未采用论文的“排除最大项 + softplus/log1p”路径。按源码判断，截至 2026-10-28 未合入论文修复。
- PyTorch 仓库 2026-10-28 HEAD（`b8d6389`）已合入准备性重构 `#202240`（抽取 `log_softmax` helper），但 helper 仍直接求 `sum_j exp(x_j-max)`；语义修复 `#202049` 仍在独立 PR ref，未进入 HEAD。
- “JAX 参考实现约十几行”只能粗略描述单个修复函数：hard-target 的 `log_softmax`、`softmax`、`cross_entropy` 以及 soft-target 函数合计超过十几行。
- Slingshot 因果链由论文实验支撑：越过精度悬崖后目标梯度消失、Adam 二阶矩衰减，样本回到悬崖下时恢复的梯度被小分母放大；修复后目标梯度连续存在，尖峰消失。
- 论文实验边界主要是 modular arithmetic grokking 小模型：1 层 Transformer 或 2 层 MLP，而非标准大规模预训练。论文自己明确称这些极端 margin 和近零师生差在常规大规模训练中不常见。
- 蒸馏实验从与教师相同的 fp32 权重初始化学生；标准 soft-target CE 在 bf16 产生约 `1e-3` 残余梯度并损害准确率，在 fp32 产生 KL 尖峰及约 `1e-6` 地板，修复版在三种精度下保持 KL=0 与 100% 准确率。
- 以论文术语直接检索 PyTorch/JAX 官方仓库未找到结果，不能据此断言“未修复”；需继续检查论文代码链接、框架源码和提交记录。

## 证据缺口

- 论文没有大规模预训练实验，也没有独立代码仓库；当前只能引用正文参考实现和作者项目页。
- JAX 两个 issue 以 duplicate 关闭，但普通页面未暴露其主 issue；当前结论改由 2026-10-27 HEAD 源码直接支撑。
- PyTorch/JAX 合入状态具有强时效性，后续使用时需按版本、后端和编译路径重新检查。
