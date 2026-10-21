# 面向用户表征的 Decoder-Only LLM Attention Masking 重思考 · 中文技术精读稿

> **原文**：How Do Decoder-Only LLMs Perceive Users? Rethinking Attention Masking for User Representation Learning
> **arXiv**：2602.10622v1（Feb 2026）
> **作者**：Jiahao Yuan¹²†, Yike Xu¹†, Jinyong Wen¹, **Baokun Wang¹\*（通信作者）**, Yang Chen¹, Xiaotong Lin¹, Wuliang Huang¹, Ziyi Gao¹, Xing Fu¹, Yu Cheng¹, Weiqiang Wang¹
> **单位**：DeepFind Team, Ant Group（¹）+ 华东师范大学（²，Jiahao 实习）
> **代码**：`https://github.com/JhCircle/Deepfind-GGSM`
>
> **本文档说明**：本人对论文的中文技术阅读稿，覆盖 Abstract + Introduction + Training Data + Method (GG-SM) + Experiments + Main Results + Analysis + Conclusion 共八个高价值段落。原文公式编号保留；术语给中英对照。若需精确原文请查阅 PDF (`pdfs/2602.10622.pdf`)。

---

## Abstract 摘要

Decoder-only 大模型正越来越多地被用作用户行为编码器，但**注意力掩码策略对 user embedding 质量的影响**至今缺乏系统研究。作者在统一对比学习框架下、基于亿级真实支付宝数据，系统对比 **causal / hybrid / bidirectional** 三种 attention mask；进一步提出 **Gradient-Guided Soft Masking (GG-SM)**——基于梯度信息的 pre-warmup，先注入"未来 token 重要性"，再由线性调度器过渡到全双向。

在 9 个覆盖 User Prediction / Behavior Preference / Marketing Sensitivity 的工业级 benchmark 上，GG-SM 相较 causal、hybrid、scheduler-only baseline 都能得到更稳定的训练与更高质量的 bidirectional 表征，且与 decoder-only 预训练保持兼容。

**核心结论**：在把 decoder-only LLM 用作 user encoder 时，**masking 设计与训练过渡路径**同等重要。

---

## 1. Introduction 引言

**问题背景**：用户 embedding 需要把文本画像、交互历史、表格属性整合成紧凑表征，广泛用于数字营销、推荐、个性化系统。当前主流是自监督对比学习，通过（i）同序列上下文一致性、或（ii）跨会话一致性来对齐增强视图。

**编码器路线的取舍**：
- **Bidirectional PLM**（BERT、T5）用全自注意力天然适合表征，但**静态、批处理**，需要一次拿到全上下文——难以支持**流式、增量**的实时场景；
- **Decoder-only LLM** 支持 autoregressive 交互，近年在 user modeling 上通过对比目标微调也证明有效；但其**默认 causal mask** 是否是最优表征结构，未被充分研究。

Adaptation 时 mask 可选三条 recipe：
1. **Causal**：标准单向；
2. **Bidirectional**：全自注意力（LLM2Vec、KaLM 等）；
3. **Hybrid**：user segment 内部双向、后续 token 仍因果。

作者主张**从 causal 到 bidirectional 的过渡本身**才是核心难点，据此提出 GG-SM 作为 warm-up 机制。

**贡献**：
1. 统一对比框架下第一次系统对比三类 mask；
2. 证明"训练过渡路径"是优化稳定性与表征质量的关键；
3. 提出 GG-SM，在 9 个 benchmark 上取得更强的双向表征。

---

## 2. Related Work（简要）

- **LLM for User Embedding**：BERT4Rec、FOUND 用双向；Qwen3-Embedding、InstructUE 用 causal + 对比/指令目标；Conan 引入渐进 scheduler。但缺乏严格对照实验对比三类 mask。
- **Synthetic Data for User Embedding**：早期靠启发式增强/伪标签；近期用 GPT-4 生成行为轨迹或用户意图，但依赖闭源 API 存在成本 / 复现性问题。小型开源模型合成 fidelity 又不够。作者提出**training-free synthesis**——用 base LLM 探测 hard-to-align QA pair，再用 chain-of-thought 归纳规律迭代 prompt。

---

## 3. Training Data 训练数据

沿用 FOUND / InstructUE，两类对齐数据：

- $D_{behavior} = \{(u_i, b_i)\}$：rule-based 未来行为对；
- $D_{qa} = \{(u_i \oplus q_i, a_i)\}$：LLM 合成 QA 对。

用户 $u_i = \{Bill_i, Mini_i, Spm_i, App_i, Search_i, Tabular_i\}$，来自过去 90 天多模态记录。

### 3.1 规则化行为轨迹数据集

左塔编码过去 3 个月原始交互序列；右塔对随后 1 个月按 action type 或 temporal bin 聚合，再随机抽样一个代表性子集作为未来预测目标。左右塔在训练中对齐。

### 3.2 LLM 合成 QA 数据集（Post-Rule Improvement 机制）

启发来自"hard negative 能增强 embedding 学习"，把该思路推到**hard positive**——用预生成的困难 QA 对当 anchor：

**Step (1) 合成 & 校准集** — 用 Qwen-Max 生成 diverse user understanding scenarios 作 seed pool $\text{Pool}$；对每个用户 $u_i$：
$$\text{seed}_{top10}(u_i, \mathcal{P}) = \text{LLM}(u_i, \text{Pool}, P_{retrieve}) \tag{1}$$
$$D_c = \{\text{LLM}(\text{seed}_{top10}, u_i, P_{qa})\}_{j=1}^{1000} \tag{2}$$
从而得到 1000 条 $(u_i \oplus q_i, a_i)$ 校准集。

**Step (2) Alignment Difficulty Probing** — 用强 embedding 模型 Emb（Qwen3-Embedding-0.6B）计算难度分：
$$S_d = 1 - \text{Sim}(\text{Emb}(u_i \oplus q_i), \text{Emb}(a_i)) \tag{3}$$
过滤保留 $S_d \ge T_{filter}$ 的困难样本 $D_{hard}$。

**Step (3) Inductive Feature Completion** — 用 Qwen-Max 从 $D_{hard}$ 中归纳共同规律 $P_{rule}$，融进 $P_{qa}$。

**Step (4) Scaling & 后修** — 用优化后的 $P_{qa}$ 规模化生成，配以标准化的模态分隔符与 special `<USER>` token；再做后验重写保证语义清晰。

---

## 4. LLMs as Encoders：Training Recipe

### 4.1 训练架构（Fig. 1）

沿 FOUND 双塔框架，处理 user+query：
- 每种模态由 modality-specific encoder 处理，经 lightweight adapter 投影到 LLM embedding 空间；
- 答案 $a_i$ 走同一个 decoder-only LLM $\mathcal{M}$（另一塔），两塔共享 backbone 但独立编码；
- 输入模板（附录 A）：`<bill>...</bill>` 等模态包裹 + optional instruction + `<USER>` 特殊 token；user embedding 取 `<USER>` 的最后一层 hidden state（Eq. 12），answer embedding 取 `<EOS>` 的最后一层 hidden state（Eq. 14），并做 L2 norm。

### 4.2 Gradient-Guided Soft Masking（GG-SM）核心方法

**目标**：让 causal LLM 具备双向理解力，同时保持训练稳定。作者在 Conan 的 causal-to-bidirectional scheduler 前，加一段**梯度引导 warmup**。

设 warmup 步数 $T_{warm}$、总步数 $T_{total}$。对长度 $L$ 序列的 hidden states $H = [h_1, \dots, h_L]$，定义 soft mask $M^{soft}(t) \in \mathbb{R}^{L \times L}$：
$$M^{soft}_{ij}(t) = \begin{cases} 0 & j \le i \\ \log w_{ij}(t) & j > i \end{cases} \tag{4}$$
$$w_{ij}(t) = \begin{cases} \sigma(\|\nabla_{h_j}\mathcal{L}\|) & t < T_{warm} \\ (1-\alpha_t)\sigma(\|\nabla_{h_j}\mathcal{L}_{warm}\|) + \alpha_t & T_{warm} \le t < T_{total} \end{cases} \tag{5}$$

其中 $\alpha_t = (t-T_{warm})/(T_{total}-T_{warm}) \in [0,1]$，$\sigma$ 是 sigmoid，$\mathcal{L}_{warm}$ 是 warmup 结束时的 loss。

**直觉**：
- **Warmup 阶段** ($t < T_{warm}$)：未来 token 的可见权重由**当前梯度范数** $\|\nabla_{h_j}\mathcal{L}\|$ 决定——**对 loss 影响大的 token 获得更高可见度**。这是"梯度自证"的 token 重要性。
- **Warmup 结束**：把梯度导出的软权重**冻结**成 $\log w_{ij}$。
- **Scheduler 阶段** ($T_{warm} \le t < T_{total}$)：用 $\alpha_t$ 线性从"冻结软 mask"过渡到"完全双向"($w_{ij} = 1$)。
- **推理时**：完全双向 mask，最大化上下文整合。

这样把 causal LLM 的 token-level 预训练与 sentence-level 表征学习平滑衔接（Conan 的思想 + 梯度先验）。

### 4.3 训练目标（InfoNCE + Margin Mask）

对比损失（Eq. 6）：
$$\mathcal{L}_{cl} = -\frac{1}{B}\sum_{i=1}^B \log \frac{e^{s(\hat u_i, \hat a_i^+)/\tau}}{Z_i},$$
$$Z_i = e^{s(\hat u_i, \hat a_i^+)/\tau} + \sum_{j \ne i} m_{ij}\big[e^{s(\hat u_i, \hat a_j)/\tau} + e^{s(\hat u_i, \hat u_j)/\tau} + e^{s(\hat a_i, \hat a_j)/\tau}\big] \tag{7}$$

**三类负样本**：user–answer 跨对、user–user、answer–answer。**Margin mask**（Eq. 8）：
$$m_{ij} = \begin{cases} 0 & s_{ij} > s(\hat u_i, \hat a_i^+) + c_{margin} \\ 1 & \text{otherwise} \end{cases}$$

避免把"其实很相似"的样本错误当作负样本。

---

## 5. Experiments 实验

### 5.1 模型与实现

- **数据管道**：Qwen3-30B-A3B 为效率驱动的合成 LLM；
- **模态编码器**：gte-base-zh 分别编码 Bill / Mini / SPM / App / Search / Tabular；
- **Backbone**：Qwen2.5-0.5B-Instruct，采用 contrastive learning + 不同 mask 策略微调；
- **训练**：global batch size 2048，70k step，AdamW lr $2 \times 10^{-4}$ + cosine decay，LoRA rank=64、$\alpha=64$；
- **硬件**：64×A100-80G 训练，单卡 A100-80G 推理。

### 5.2 Baseline 与任务

Baseline 涵盖三类：
1. **通用文本 embedding**：Qwen2.5-0.5B-Instruct（未微调 oracle）、Qwen3-Embedding-0.6B、Llama-embed-nemotron-8B、KaLM-Embedding-Gemma3-12B；
2. **User embedding**：MSDP、One4all (U-MLP)、CPC、FOUND、InstructUE；
3. **Mask 训练变体**：
   - Causal（沿用原 causal mask）；
   - Hybrid：(a) gradient-guided soft masking 用左塔梯度（不含 warmup）；(b) MLP-driven attention opening；(c) global CLS token；
   - Bidirectional：(a) 直接双向；(b) scheduler；(c) GG-SM（作者提出）。

### 5.3 评测任务（Table 1，9 个 Alipay 场景）

三大域：
- **User Prediction**：Concert Click / User Log-in / MAU Loss（≈50w 样本/任务）；
- **Behavior Preference**：Public Transit / Consumption Power / Food Interest / Movie Interest；
- **Marketing Sensitivity**：Achievement / Physical。

**指标**：AUC（linear probing）。

---

## 6. Main Results 主要结果（Table 2 + Fig. 2）

**核心结论**：GG-SM 平均 AUC **0.7745**，是全部 baseline 中最高。

**对比 general embedding**（参数量 8-12B）：
- Llama-embed-nemotron: 0.7357
- KaLM-Embedding: 0.7156
- **GG-SM (Qwen2.5-0.5B) 高出 ≈4%**

作者指出：**在稀疏、非语言型行为日志上，8B+ 通用 embedding 引入的是"多余噪声"而不是"有用先验"**；GG-SM 通过梯度校准注意力，比原始 scaling 更高效对齐 LLM 潜空间与领域行为结构。

**对比 user embedding 系列**：
- MSDP / One4all / CPC：局部特征匹配好，但跨域迁移弱（如 Food / Movie 表现差）；
- FOUND: 0.7690；InstructUE: 0.7728；
- **GG-SM: 0.7745**，把 LLM 的上下文先验和注意力机制视为**可演化的 bottleneck**，比"把 LLM 当静态抽取器"的方案更能建长距依赖。

**Mask 变体内部比较**：
| Mask | Avg AUC |
|---|---|
| Causal | 0.7709 |
| Hybrid (mask/gq/mlp) | 0.7706-0.7718 |
| Bidirectional (直接) | 0.7721 |
| Scheduler (Conan) | 0.7733 |
| **GG-SM (Ours)** | **0.7745** |

结论：**双向优于 causal；scheduler-only 已提升；GG-SM 通过梯度 warmup 进一步领先**。GG-SM 提升不仅体现在整体 AUC，也体现在 Concert (0.5767)、User (0.8438)、Food (0.8554) 等多个具体任务上均取得最优。

**Marketing Sensitivity 域**（Achiev.、Physical）——**GG-SM 达到峰值 AUC**，作者认为这是因为 latent intent 最难捕捉的场景恰好最需要动态、任务相关的注意力权重。

---

## 7. 分析（Fig. 3 + 附录 A.4）

### 7.1 梯度引导的注意力演化 (efficacy)

- **Causal**（oracle）过于受限；
- **Hybrid** 引入额外参数但难与预训练 backbone 对齐，收益边缘；
- **GG-SM** 用瞬时梯度范数作为 token 重要性动态信号，让模型"不仅看到更多 token，而是学会在双向适配的早期阶段优先关注最有信息的 token"。Fig. 3 显示 GG-SM 相比 scheduler-only，训练 loss 更平滑、更快收敛。

### 7.2 域鲁棒性与迁移性

- 传统对比模型跨域方差大（在高频行为任务好，在 sensitivity 任务差）；
- **GG-SM 三大域一致领先**——尤其在 latent intent 最难的 Marketing Sensitivity 上；
- 表明**模型内部的学习压力（梯度）比人工构造的数据增强/架构改动更能捕捉可迁移的用户特性**。

### 7.3 Mask 变体的形式化定义（附录 A.4）

- **Causal**：$M^{causal}_{ij} = 0\ \text{if}\ j \le i,\ -\infty\ \text{otherwise}$；
- **Hybrid**：user segment 内部双向、后续 causal；三种打开上三角方式：gradient / MLP / global query；
- **Bidirectional**：
  - 直接：$M^{bi}_{ij} = 0$；
  - Scheduler：$M(t)$ 允许 attention 到 $\min(i + \Delta(e), L)$，$\Delta(e)$ 随 epoch 线性/cosine 增长；
  - **GG-SM (Ours)**：warmup 用梯度 mask，然后线性插值到全双向（式 15）：
    $$w_{ij}(t) = (1 - \alpha_t)\sigma(\|\nabla_{h_j}\mathcal{L}_{warm}\|) + \alpha_t,\ \alpha_t = \frac{t-T_{warm}}{T_{total}-T_{warm}}$$

---

## 8. Conclusion 结论

本工作从注意力掩码视角重新审视了 decoder-only LLM 用户表征学习，第一次在统一对比框架 + 9 个真实工业任务上系统对比 causal / hybrid / bidirectional 三条 recipe，得到两条主要论断：

1. **过渡路径**（从 causal 到 bidirectional 的路径）比最终 mask 形式更关键；
2. **GG-SM**（先用梯度做 pre-warmup 再线性调度到全双向）既能保持 decoder-only 预训练的兼容性，又能获得更强的双向表征。

**未来方向**：hybrid masking 的渐进过渡仍是有价值的研究方向；作者开源了实现（GitHub：JhCircle/Deepfind-GGSM）鼓励社区推进。

---

## 附：术语中英对照速查

| 英文 | 中文 |
|---|---|
| Attention Masking | 注意力掩码 |
| Causal / Bidirectional / Hybrid Mask | 因果 / 双向 / 混合掩码 |
| Gradient-Guided Soft Masking (GG-SM) | 梯度引导软掩码 |
| Warmup + Scheduler | 预热 + 线性调度 |
| Decoder-Only LLM | 仅解码器大模型 |
| InfoNCE Contrastive Loss | InfoNCE 对比损失 |
| Margin Mask / False Negative Filter | 边距过滤 / 伪负样本过滤 |
| Hard Positive | 困难正样本 |
| Alignment Difficulty | 对齐难度 |
| Chain-of-Thought (CoT) Reasoning | 思维链推理 |
| Linear Probing | 线性探测 |
| User Prediction / Behavior Preference / Marketing Sensitivity | 用户预测 / 行为偏好 / 营销敏感度 |
| Loss-Quality Dissociation | 损失-质量脱耦 |
