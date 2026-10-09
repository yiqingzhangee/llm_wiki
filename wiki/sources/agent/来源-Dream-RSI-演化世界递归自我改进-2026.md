# 来源：Dream-RSI——以演化世界做递归自我改进（2026）

> 原始来源：[`raw/articles/RSI/papers_Dream-RSI_2609.14858v1.pdf`](../../../raw/articles/RSI/papers_Dream-RSI_2609.14858v1.pdf)（Google / UMD / UVA，共 17 位作者，通讯 xidongwu@google.com、zzhangx@google.com；© 2026 Google）。项目：github.com/zhengkid/Dream-RSI、dream-rsi.com。
>
> 关联概念：[[递归自我改进与Harness自演化]]、[[Agent记忆系统与自我演进]]、[[来源-RSI综述-Preprints-2026]]。

## 一句话主张

把历次在线探索产生的 **discovery history 组织成可回放的 discovery tree / replay simulator**，让一个昂贵的在线探索 $\pi_t$ 支撑成千上万次跑在离线“做梦”(dreaming) 里的廉价探索策略评估与改进，从而闭合递归自我改进（RSI）的元层循环。

## 解决的核心瓶颈

- **元级反馈延迟且昂贵**：评估的不只是某个候选动作，而是“一整条探索策略会如何影响后续发现过程”，需要经历很多 proposal-evaluation 循环，反馈慢、成本高。
- **元策略空间巨大**：新提出的策略常表现很差，需要试很多替代，而每个试错都要完整的在线 rollout 才看得到结果。
- **结论**：在 exploration 层面做自我改进极贵，难以高效闭环。

**核心直觉**：完整的 discovery history 本身就是一个"world model"——它天然记录了决策树结构与对应的代码执行结果。把它当 replay simulator 用，评估新策略时只需"读回已存结果"，不必重跑底层 discovery agent 或 evaluator（作者将 replay simulator 与 worlds 视为同义词）。

## 三阶段框架（Figure 1）

1. **Online Explore（在线探索）**：当前 exploration policy 指导 coding agent 扩展 discovery tree，记录执行轨迹。
2. **Construct Replay Simulator（构建回放模拟器）**：把已记录的 discovery tree 转成可复用的 simulator pool。
3. **Dreaming-based Policy Improvement（基于做梦的策略改进）**：在离线 dreaming 环境里生成、评估大量候选策略，把表现最好的版本再部署回在线探索，形成闭环 RSI loop。

**关键边界**：Dream-RSI **不修改底层 coding agent**，只在外面加一个轻量 orchestration layer 控制 branching、并行探索与 stopping；**只有 exploration-policy 代码被更新**，底层模型、评估器和执行接口保持不变。

## 机制要点（形式化）

- **Discovery tree**:一个 rollout 逐步构建有根树。节点 $v$ 记录从根到该节点的演化历史快照、生成 artifact、evaluation diagnostics 与分数 $s_v$（越大越好）。
- **原子动作** `CONTINUE(v)`：恢复 $v$ 的 workspace → 生成并评估一个新子节点。从根继续 = 开新分支；从非根叶子继续 = 在现有分支末端 refine/repair。
- **单步决策**：从叶节点集合选一批 `C ∈ A(T;W)`（受最大并行 worker 数 $W$ 约束）执行；选空批则 rollout 终止。转移是随机的（同一 workspace 可能产生不同结果）。
- **在线 rollout**：最多 $K_1$ 个决策轮，产生的最终树 $T_t$ 追加进历史 $H_t$。
- **离线评估（replay）**：历史固定；策略版本 $\pi_t^0=\pi_t$，经 LLM 策略开发 agent 依 replay 轨迹/反馈迭代出 $\pi_t^1…\pi_t^{M-1}$；每个版本在所有已记录树 $T_i$ 上重放、从根开始，最多 $K_2$ 轮。新策略不会继承先前评估暴露的子树。
- **Replay 评分**（平衡发现质量/执行成本/并行度）：
  $$v_i^m = \underbrace{\max_{v\in T_i^{m,*}} s_v}_{\text{discovery quality}}- \underbrace{\beta_1 N_i^m}_{\text{execution cost}}+ \underbrace{\beta_2 \frac{N_i^m}{\max\{1,k_i^{m,*}\}}}_{\text{parallelism bonus}}$$
  其中 $N_i^m$ 为重放暴露的非根节点数。第三个奖励项激励策略"批量提交有价值的 continuation"，而不是顺序逐个执行。
- **策略选择**：评估分 = 固定历史上的平均重放分；选出最好的版本部署到下一轮，且在固定历史平均分上保证不比当前策略差。

## 实验结果（3 个科学发现域，8 个任务）

- **算法工程（Lasso regularization path）**：下游 6 个 held-out 数据集上，Gemini-3.1-Pro 平均运行时间 3587.1→2931.0 ms（agent calls 550→317）；Gemini-3.7-Flash 2516.7→2350.6 ms（3200→1879）。仍优于 sklearn 与 glmnet，且 discovery 计算量远小于 SimpleTES 的 51,200 generations。
- **数学优化（Sum Diff / Autocorrelation / Circle Packing，10 rounds）**：Sum-Diff 1.145427（优于 SimpleTES 与固定探索）、Circle Packing 2.635983（追平最强基线）、Autocorrelation 1.456375；SimpleTES 的同任务 SOTA 需 51,200 generations，而本方法仅 ~1,000。
- **GPU kernel（KernelBench：VGG16/LayerNorm/ConvDiv/ConvMax）**：VGG16 用 **2.43×**、LayerNorm 用 **1.79× 更少的行世代**达到相当性能；ConvDiv **2.09×**、ConvMax **1.44× 更高分**（相似预算）。

## 与既有结论的连接

- 对 [[递归自我改进与Harness自演化]]：Dream-RSI 是本仓库 RSI 四环中"**经验整合 → 改改进机制本身**"这一档的强正样本——它把改进对象放在 exploration-policy（元层 orchestration），并给出 replayable / off-policy 这一把"延迟且昂贵的元反馈"变便宜的机制。呼应既有结论"当前系统尚未实现稳定开放式 RSI"——Dream-RSI 的收益是**递归轮内、任务内**的，迁移与长期稳定性仍需独立核验。
- 对 [[Agent记忆系统与自我演进]]：把"历史到底存成什么"进一步推向"历史本身作为可执行的世界模型/replay simulator"，而非只作记忆或上下文。
- **concept 归属**：模型无关、面向 long-horizon discovery 的 exploration 层改进——与"权重更新"和"Harness 工程"都不同，是第三种改进基底：**经验重放驱动的元策略优化**。

## 冲突与不确定性

1. 结果高度依赖 Gemini CLI / 模型能力（Gemini-3.1-Pro、3.7-Flash）；跨模型迁移未充分证明。
2. Replay 评分权重 $\beta_1,\beta_2$ 与 $K_1,K_2,M,W$ 等超参对结论敏感度未展开。
3. "保证不比当前策略差"仅在**固定历史平均分**上成立，不保证对新世界泛化。
4. 精确数值、软件版本与统计显著性应回 arXiv 2609.14858 原文核验；本页仅采其报告结论。
