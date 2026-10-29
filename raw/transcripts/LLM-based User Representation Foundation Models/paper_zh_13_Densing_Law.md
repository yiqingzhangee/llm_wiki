# Behavioral Densing Law：亿级用户表征的 Tokenization 规模律 · 中文技术精读稿

> **原文**：Towards a Densing Law for User Representation Learning at Billion-Scale Capacity
> **arXiv**：2608.23392v1（Aug 2026）
> **作者**：Bin Dou¹*, Junru Zhang¹²*, Zhaoyi Yuan¹²*, Wuliang Huang¹, Letian Gong¹, **Baokun Wang¹†**, **Huan Li²†**, Yu Cheng¹, Weiqiang Wang¹
> **单位**：DeepFind Team, Ant Group（¹） / 浙江大学（²）；* 共同一作；† 共同通信作者
>
> **本文档说明**：本人对论文的中文技术阅读稿，覆盖 Abstract + Introduction + Preliminary + Raw Scaling Wall + Densing Gains + Behavioral Densing Law + Adaptive Length Gated Network 七大高价值章节。保留原文公式编号；术语给中英对照。若需要精确表达请对照原文 PDF (`pdfs/2608.23392.pdf`)。

---

## Abstract 摘要

现实工业场景下用户表征学习通常沿三条轴 scale：**用户数量、行为序列长度、模型规模**。但现有方法有两大问题：

1. **原始数据在亿级容量下遇到瓶颈**——继续叠加原始文本行为的收益急剧衰减，可通过 tokenization 缓解；
2. **缺乏 tokenization 配置随数据规模变化的定量分析**。

作者据此提出 **User Behavioral Densing Law**（用户行为致密律），描述数据规模与"最小充分 tokenization 容量"之间的定量关系。工作包含四步：

1. 在亿级支付宝数据集上做 raw vs. tokenized 的 scaling pilot study，揭示 raw data scaling wall 及 tokenization 的突破效果；
2. 从理论与实验两方面推导最小充分 tokenization 容量的量化 scaling 模式；
3. 基于该定律提出新的 tokenization 方法 **ALGN（Adaptive Length Gated Network）**；
4. 在多个数据源、tokenization 方法、下游任务上验证 Densing Law 的普适性与可靠性，且 ALGN 在性能与效率上均超越现有 baseline。

---

## 1. Introduction 引言

**传统 pipeline**：深度用户表征已成个性化平台的基础，将用户行为编码为统一 embedding 后同时驱动推荐、CTR/CVR 预估、检索、自然语言用户定向。工业系统通常沿三个方向做规模扩张：更多用户、更长历史窗、更大 encoder。但因用户行为高度受"日常习惯 + 重复模式"支配——**数据量增长远快于新的、与任务相关信息的增长**，出现严重的边际收益递减。

据此作者提出两个关键定义（Definitions 1.1 & 1.2）：

- **Raw Behavioral Scaling Wall（原始行为规模墙）**：单纯增加原始行为序列长度、用户数量或模型规模，下游收益严格递减，最终出现 **loss-quality dissociation**（预训练 loss 继续下降但下游精度不再提升）。
- **Behavioral Densing（行为致密化）**：把长的原始历史转成紧凑的表征，保留任务相关差异、抑制冗余，使每个"有效输入单元"承载更多下游相关信号。

论文进一步给出致密化的三条基本原则：
1. **Bounded Raw Gains** — 冗余占主导时边际效用下降；
2. **Density Driven Scaling** — 更稠密表征把饱和边界外推；
3. **Minimal Sufficient Capacity** — 最优表征只分配"刚好够"用来保留任务相关信息的容量。

**方法实例化**：作者用 **RQ-VAE 残差量化**把重复事件历史压成紧凑离散码序列作为 density operator；同时观察到用户信息在时间上分布不均，进一步提出**熵驱动的变长 tokenization**，把更多 code 分给高熵段、少分给可预测常规段，逼近"最小充分表征容量"。

**核心贡献**（Introduction 末尾三条）：
1. 诊断"scaling saturation"与 tokenization 效果——发现"数据信息密度"而非"模型参数量"才是主要瓶颈；
2. 提出**Predictive tokenization configuration recipe**——用 Behavioral Densing Law 从数据规模 + 少量轻量统计估算所需 tokenization 容量；
3. 提出 **ALGN**——按 quantization residual 与 expression uncertainty 自适应分配残差深度，从密度角度更高效利用离散表征空间。

---

## 2. Related Work（简要）

Table 1 从「Raw Data / Tokenized Data / Capacity Configuration / Capacity Allocation」四维覆盖度总结相关工作，本文是唯一四维全覆盖的。相关方向包括：
- **User data scaling**：Ardalani 等的规模律研究、长历史序列建模（SIM、UBR4CTR、TWIN）；
- **Behavioral tokenization**：VQ-VAE / RQ-VAE / TIGER / U²QT；
- **Adaptive quantization**：Huijben、Seo、Chae、Kusupati 的可变量化深度、matroshka embedding、自适应计算。

现有方法通常固定 tokenization 配置，未回答"数据规模变化时 codebook 应如何选择"的问题。

---

## 3. Preliminary

### 3.1 记号

对亿级平台（如支付宝），用户 $n$ 的行为按日切分：
$$u_n = \{u_n^1, \dots, u_n^D\},\quad u_n^d = \{B_n^d, S_n^d, M_n^d\}$$

分别对应 **PayBill**、**Super Position Model (SPM)**、**MiniProgram**。每种模态可用文本表示，$T$ 是文本 token 字典，$L_n^d(\cdot)$ 是各源事件数。

**关键统计**（Table 2）：
- $N$：用户数，1 亿 – 20 亿；
- $D$：行为序列长度，30–270 天；
- $P$：user embedding 模型规模，0.05B – 0.4B；
- 每用户每日事件 0–10 条（中位数 6 条）、行为文本 token 数约 2200；
- 测试集：Classification 50 任务、Text Retrieval 22 任务、U2U Retrieval 22 任务，每任务约 50 万用户。

### 3.2 预训练框架

采用 Behavior-Text Contrastive Learning 双塔（Fig. 2，沿用 FOUND）：

1. 序列切成历史段 $u_n^{past} = (u_n^1, \dots, u_n^m)$ 与未来段 $u_n^{future} = (u_n^{m+1}, \dots, u_n^D)$；
2. 历史段经 Transformer encoder $f_\theta$ 得到 $e_n^b \in \mathbb{R}^d$；
3. 未来段经模板 $Q$ 展开成描述文本（例："The user purchased {items} amounting more than {num} dollars with {status} payment"）；由 LoRA 微调的 LLM embedding 模型 $g_\phi$（Qwen3-Embedding）得到 $e_n^q$，base 冻结、LoRA rank 16；
4. 用 InfoNCE 对齐两 embedding（Eq. 1）：
   $$\mathcal{L}_{CP} = -\frac{1}{B}\sum_{i=1}^B \log \frac{\exp(\text{sim}(e_i^b, e_i^q)/\tau)}{\sum_j \exp(\text{sim}(e_i^b, e_j^q)/\tau)}$$
   其中 $\text{sim}$ 是余弦相似度，$\tau$ 为可学温度。

### 3.3 下游评测协议（Fig. 3）

三类任务全部走线性探测：
1. **Classification**（Eq. 2）：冻结 encoder，训线性头 $Wf_\theta^*(u_n) + b$；指标 AUC、KS、Accuracy。
2. **Text-based Retrieval**（Eq. 3）：零样本用户定向；文本 query embedding 与用户 embedding 计算 cosine 相似度筛选；指标 AUC、Precision、Recall。
3. **U2U Retrieval**（Eq. 4-5）：few-shot 场景，用 seed 用户学 learnable prompt $P$，再作为 query 后缀执行式 (3) 的检索；指标同上。

---

## 4. Scaling Law Analysis：Raw Data Scaling Wall

作者在 $(N, D, M)$ 三条轴上系统扫描，得到三条 Takeaway：

### 4.1 用户数量维度 $N$

固定 $D=180$ 天、encoder ≈0.1B，$N$ 从 0.01B → 0.1B：
- 0.01B → 0.03B：AUC / KS / Acc 明显上升——低数据下新增用户带来行为多样性；
- 0.03B → 0.1B：曲线趋平——**重复行为主导**，新增用户几乎不带来新任务相关信息。

**Takeaway 1**：只有在行为多样性未被充分覆盖之前，增加用户才能改善表征；饱和后主要是冗余。

### 4.2 时间窗维度 $D$

固定 $N=0.1B$，$D$ 从 30 → 120 天：
- 30 → 60 天：显著提升——近期历史帮助 encoder 学到更完整偏好；
- 60 → 120 天：几乎无提升；Appendix Fig. 12 显示 contrastive alignment loss 收敛到几乎相同的最终值。

**Takeaway 2**：长历史主要引入重复事件（例：账单周期、习惯性消费）。

### 4.3 模型容量维度 $P$（Loss-Quality Dissociation，Fig. 5）

固定 $N=0.1B$、$D=180$ 天（已在饱和区），把 encoder 从 0.05B → 0.4B：
- **训练 loss 单调下降**，说明更大模型确实更能拟合预训练目标；
- **下游 AUC / KS / Accuracy 在 0.2B 之后基本不变**——0.2B → 0.4B 几乎为零收益。

**Takeaway 3**：更大 encoder 只是在冗余行为细节上拟合更精，并没有抽取到更多任务相关信号；**信息密度**才是关键。

### 4.4 结论与动机

关键 tipping point：$N \le 0.03B,\ D \le 60,\ P \le 0.2B$。超过这条边界后 raw scaling 边际收益急剧衰减。作者据此论证：**必须从"量的扩张"转向"密度的扩张"**——即把长且冗余的原始序列先转成致密离散 token。

---

## 5. Densing Gains from Tokenization

### 5.1 RQ-VAE 行为 tokenization（Fig. 6，Eq. 6-7）

三处工业适配：
1. 量化对象从图像 latent 换成多源 PayBill 行为 embedding；
2. **解耦 token 长度 $H$ 与原始时间跨度 $D$**：长历史压缩成固定 token 数；
3. 把 tokenizer 用作"密度算子"（density operator）而不是仅做重构。

流程（Eq. 6）：
- 序列 $X_n = (x_{n,1}, \dots, x_{n,L_n})$；先经局部聚合 → $Z_n = (z_{n,1}, \dots, z_{n,H})$，$H \ll L_n$；
- 对每个 $z_{n,h}$ 做 $M$ 层残差量化：
  $$r^{(1)}_{n,h} = z_{n,h},\ \ k^{(m)}_{n,h} = \arg\min_k \|r^{(m)}_{n,h} - c^{(m)}_k\|_2^2,\ \ r^{(m+1)}_{n,h} = r^{(m)}_{n,h} - c^{(m)}_{k^{(m)}_{n,h}},$$
  $$\hat z_{n,h} = \sum_{m=1}^M c^{(m)}_{k^{(m)}_{n,h}},\quad t_{n,h} = (k^{(1)}_{n,h}, \dots, k^{(M)}_{n,h}).$$

层次含义：早期 codebook 捕捉稳定的宏观语义（消费类别、出行、支付场景），后期 codebook 精修细粒度差异（商户偏好、金额微变化）。

训练目标（Eq. 7）为重构损失 + commitment loss。tokenizer 训练完后**冻结**，把下游 encoder 训在离散 token 上，用同样的 contrastive alignment，从而把"下游收益"归因给密度提升本身。

### 5.2 Tokenized vs. Raw 对比（Fig. 7-8）

- **时间维度**：$D$ 短时 raw 与 tokenized 差异不大；约 $D \approx 64$ 天开始 tokenized 曲线超越 raw，随 $D$ 变大差距扩大。
- **用户维度**：约 $N \approx 1.2 \times 10^7$ 之后 tokenized 显著优于 raw。
- **计算成本维度**：Fig. 8 显示 tokenized 在相同 PFLOPs 下三类任务（Classification / Text Retrieval / U2U）AUC 更高。

**tokenization gain**（Eq. 8）：$\Delta(s) = P_{tok}(s) - P_{raw}(s)$，在饱和区显著为正——tokenization 不只是缩短序列长度，而是**在保留任务相关差异的同时抑制重复**，提升有效信息密度。

### 5.3 局限与下一步

固定长度 tokenization 对所有行为段一视同仁，忽略行为异质性；后续章节引入 ALGN 做**实例级 capacity 分配**。

---

## 6. Behavioral Densing Law 行为致密律

### 6.1 密度定义与 Pareto Optimal（Eq. 9-10）

- 用 $s = (s_1, \dots, s_r)$ 表示广义行为规模状态（人群、时间跨度、模态数、场景多样性等）；
- 令 $C(\phi)$ 是致密表征 $\phi$ 的实际成本，$\hat C_{raw}(s)$ 为 raw sequence 达到同样下游效用所需成本；

**最优行为密度**：
$$\rho^*(s) = \frac{\hat C_{raw}(s)}{C^*(s)} \tag{9}$$

**Pareto Optimal**（Eq. 10）：给定 tradeoff 参数 $\lambda$：
$$\phi^*(\lambda, s) = \arg\max_{\phi \in \Phi} [U(\phi, s) - \lambda C(\phi)]$$

$\lambda$ 大 → 偏好低成本；$\lambda$ 小 → 允许更高容量。**目标不是最大压缩，而是在给定规模下找 utility-cost 最佳权衡**。

### 6.2 最小充分 tokenization 容量（Eq. 11-13）

对 RQ-VAE，配置 $\kappa = (K, M, d, H)$（codebook 大小、残差深度、code 维度、token 长度）：
$$C_{tok}(\kappa) = HM \log K + \eta HM d \tag{11}$$

即 discrete code capacity + continuous embedding cost 的加权和。

由于下游性能有随机噪声，作者用鲁棒约束形式：
$$U^*(N,D) = \max_\kappa U(q_\kappa, N, D),$$
$$\kappa^*(N,D) = \arg\min_\kappa C_{tok}(\kappa)\ \text{s.t.}\ U(q_\kappa, N, D) \ge U^*(N,D) - \epsilon \tag{13}$$

**"选最小的、性能与最优点差 ≤ε 的配置"**。

### 6.3 幂律形式（Eq. 14-22）

在冗余主导区间，utility frontier 近似
$$U(C, s) = U_\infty(s) - a(s) C^{-b}, \tag{14}$$
$U_\infty$ 为最大可达 utility；$a(s)$ 描述剩余 gap；$b>0$ 描述陡峭度。

代入 Pareto 目标解 first-order condition，得
$$C^*(\lambda, s) = \left(\frac{a(s)b}{\lambda}\right)^{1/(b+1)}. \tag{17}$$

若假设 $a(s) = a_0 \prod_i (s_i / s_{i,0})^{\gamma_i}$，取对数得**行为致密律的操作形式**：
$$\ln C^*(s) = \beta + \sum_i \alpha_i \ln(s_i / s_{i,0}) \tag{19}$$

其中 $\alpha_i = \gamma_i / (b+1)$。在 $s = (N, D)$ 时：
$$\ln C^*(N, D) = \beta + \alpha_N \ln(N/N_0) + \alpha_D \ln(D/D_0). \tag{20}$$

进一步给出斜率与数据/方法性质的关系：
$$\alpha_i = f_i(\mathcal{U}_d, \mathcal{E}_\phi) \tag{22}$$
- $\mathcal{U}_d$：intra-source uniqueness（数据多样性），本文用 Qwen3-Embedding 上的 mean k-NN cosine distance 度量（Eq. 23）；
- $\mathcal{E}_\phi$：tokenization 方法的表达空间冗余（VQ-VAE 用完整 SID，每个相似表征都要独占位置；RQ-VAE 允许组合复用，冗余更低；SARQ 进一步减少无效组合）。

### 6.4 实证验证（Fig. 9）

在三种 tokenization（RQ-VAE / VQ-VAE / SARQ）× 三种数据源（PayBill / SPM / MiniProgram）× 三类任务（Classification / Text Retrieval / U2U Retrieval）上，对 $\ln C^*$ vs $\ln s$ 做拟合，观察到四条规律：

1. **log-log 上近似线性**：所有子图数据点落在一条直线附近；
2. **相同 tokenizer + 数据源下三个任务斜率几乎一致**（在数值误差内），说明规律与任务无关；
3. **斜率随 tokenization 方法变化**：VQ-VAE > RQ-VAE > SARQ；SARQ 表达冗余最低；
4. **斜率与 intra-source uniqueness 成正比**：三个数据源斜率比 ≈ $1:1.15:0.92$，与 $\mathcal{U}_d$ 比 $1:1.07:0.96$ 匹配到 $10^{-2}$ 数值误差，甚至 $\alpha_i \propto \mathcal{U}_d^2$。

**指导意义**：给定 tokenizer 与数据源，可用 Eq. 21 + Eq. 22 + Eq. 23（$\mathcal{U}_d$ 计算）直接预测所需 codebook 容量。

---

## 7. ALGN：自适应长度门控 RQ-VAE

Densing Law 指出"最小充分容量"应随输入而变，而定长 tokenizer 忽略了行为异质性——有些时段包含丰富多元行为、需要更深残差；有些时段几乎是重复日常、短码即可。

**ALGN 设计**（Fig. 10）：
- 在每一层量化后判断是否再进一步。用两个可观测代理估计边际效用：
  1. **残差范数**：$R_l = \|r_{l-1} - e_{l,m}\|_2$（Eq. 24），越大表示尚未被编码的信息越多；
  2. **累积不确定度**：$E_l = -\sum_{k=1}^l \log p(m_k \mid m_{<k})$（Eq. 25），越大表示当前 code path 越模糊。
- **门控概率**（Eq. 26）：
  $$g_l = \sigma\big(\text{sp}(w_R) R_l + \text{sp}(w_E) E_l - \text{sp}(w_C) c_l + b\big)$$
  $c_l$ 是本层激活成本（如 code 长度或层号），$\text{sp}(\cdot)$ 为 Softplus 保正。**残差大或不确定度高时倾向继续；成本高时倾向停止**。
- **激活长度**（Eq. 27）：$L_{act} = \min\{l : g_l \le \theta\}$。

由此得到变长 semantic ID：高信息段获得更多 code，常规段用较少 code。

### 实验结论（论文正文与附录）

- 与固定长度 RQ-VAE 在 matched budget 下相比，ALGN 在 AUC / KS / Accuracy 上更优或平均 token 数更少（Fig. 9 中 SARQ/ALGN 曲线斜率最低，说明单位 token 承载最多信息）；
- 在 PayBill/SPM/MiniProgram 三源上均展示 Pareto 优势。

---

## 8. Conclusion 与工业价值

作者贡献总结：
- 揭示并量化了亿级用户表征学习中的 **Raw Behavioral Scaling Wall**；
- 提出 **User Behavioral Densing Law** 并给出 log-log 幂律形式（Eq. 19-22），可由数据规模 + 少量统计量估算最小充分 tokenization 容量；
- 提出 **ALGN** 变长 tokenization 方法，兼顾性能与效率。

**工业价值**：在给定预算下，先估 $\mathcal{U}_d$ + 选定 tokenizer，即可从 Densing Law 直接决定 codebook 大小、残差深度、token 长度，不再依赖手工 grid search。对亿级用户量的支付宝表征系统尤为重要——既降训练开销，又降存储开销，同时突破 60 天窗口的历史极限。

---

## 附：术语中英对照速查

| 英文 | 中文 |
|---|---|
| Behavioral Densing / Densing Law | 行为致密化 / 致密律 |
| Raw Behavioral Scaling Wall | 原始行为规模墙 |
| Loss-Quality Dissociation | 损失-质量脱耦 |
| Minimal Sufficient Capacity | 最小充分容量 |
| Density Operator | 密度算子 |
| Semantic ID (SID) | 语义 ID |
| Adaptive Length Gated Network (ALGN) | 自适应长度门控网络 |
| Residual Quantization | 残差量化 |
| Codebook | 码本 |
| Intra-source Uniqueness ($\mathcal{U}_d$) | 源内独特度 |
| Expression Redundancy ($\mathcal{E}_\phi$) | 表达冗余 |
| Pareto Optimal | 帕累托最优 |
| Linear Probing | 线性探测 |
| U2U Retrieval | 用户到用户检索 |
