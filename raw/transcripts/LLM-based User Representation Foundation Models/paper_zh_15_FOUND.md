# FOUND：可迁移、可预测的用户定向基础模型 · 中文技术精读稿

> **原文**：Transferable and Forecastable User Targeting Foundation Model
> **arXiv**：2412.12468v2（Dec 2024 → Feb 2025）
> **会议**：ACM WWW Companion 2025 (Sydney, Australia)
> **DOI**：10.1145/3701716.3715266
> **作者**：Bin Dou\*†, **Baokun Wang‡（通信作者）**, Yun Zhu\*§, Xiaotong Lin, Yike Xu, Xiaorui Huang, Yang Chen, Yun Liu, Shaoshuai Han, Yongchao Liu, Tianyi Zhang, Yu Cheng, Weiqiang Wang, Chuntao Hong
> **单位**：Ant Group（Bin Dou、Yun Zhu 分别与西安交通大学、浙江大学有 joint 隶属）；\* 实习共同一作；‡ 通信作者
>
> **本文档说明**：本人对论文的中文技术阅读稿，覆盖 Abstract + Introduction + Preliminaries + Method + Experiments + Ablation + Conclusion 七大高价值章节。原文公式编号保留；术语给中英对照。若需精确原文请对照 PDF (`pdfs/2412.12468.pdf`)。

---

## Abstract 摘要

**用户定向（User Targeting）**——从候选池中挑出满足特定营销需求的用户——是数字营销的关键任务，但现有方法有两大瓶颈：
1. **跨域 / 跨场景迁移与泛化性差**；
2. **在真实工业场景下缺乏 forecastability**（前瞻预测能力）。

作者提出 **FOUND**：一个工业级、可迁移、可预测的用户定向基础模型。
- **跨域迁移**：整合支付宝多源异构数据（PayBill、SPM、MiniProgram、Tabular、Search Text），通过对比预训练与"一句话定向需求"对齐；
- **预测性 (Forecastability)**：**用户描述文本基于未来行为构造**，而用户表征基于历史信息——从而模型天然学到"用历史预测未来"的能力。

在真实支付宝多域 benchmark 上显著超过 baseline，并自 2024 年起在支付宝用户选择平台部署，覆盖多个业务域。

---

## 1. Introduction 引言

**背景**：用户定向是数字营销的核心——把精准的内容 / 消息 / 商品推荐给对的人。

**现有方法的两条路线**：
- **Domain-specific experts**：只能在单一场景运行，跨场景迁移弱；且大多依赖有标签数据训练下游分类头，无法适配 zero-shot 或低资源场景；
- **LLM-based methods**（如 ARALLM）：借助 LLM 的通用世界知识提升泛化，但在**工业多源数据（支付记录、行为模式）**上受限——LLM 对长尾知识理解不够，难以整合多模态时序 + 个人数据。

**两大挑战总结**：
1. **迁移与泛化差**；
2. **前瞻预测能力不足**。

**FOUND 方法概要**：
- 集成支付宝五类异构数据：PayBill 交易、SPM 位置模型日志、MiniProgram 描述、用户信息表（消费记录、金融违规等）、Search Text；
- 收集并整理**可预测的 user-text pair**，用对比学习跨模态对齐；
- **两阶段预训练**：
  - Stage I: **Self-Supervised User Modeling**——从异构多场景数据抽通用 embedding；
  - Stage II: **User-Text Alignment**——多模态融合 + 与 LoRA 微调 LLM 对齐；
- **下游 targeting 两种策略**：
  - **Zero-shot Transfer**：直接输入文本需求，按 user-text 相似度选人；
  - **Few-shot Prompt-tuning**：用少量种子用户学习 learnable prompt 增强需求描述。

**核心贡献**：
1. 工业级、可迁移、可预测的用户定向基础模型；
2. 用于用户模型预训练的新自监督任务 + 精心构造的 user-text pair；
3. 大量实验证明跨域、跨场景一致优势。

---

## 2. Related Work（简要）

### 2.1 多模态预训练

从 CLIP、ALIGN 到 BLIP、BLIP-2、LLaVA，把对比 + 生成目标结合到统一 Transformer 里，涵盖 vision-language 大模型；在推荐领域有 MGCL、MMSSL、MMCPR 等把多模态融进用户-物品交互；在图领域有 ConGrat、G2P2、GraphCLIP。

**FOUND 差异**：整合的多源数据更复杂，因此设计**两阶段流程**（先自监督预训练，再对齐 well-curated data pair）。

### 2.2 用户建模与定向

用户建模学习可迁移用户模式，用户定向是基于用户模型选人。相关工作包括 Ni 等的 attention-RNN 多任务、PeterRec、mobile app usage embedding、以及 behavioral consistency loss。这些工作在 domain scenario 覆盖上仍属早期。

---

## 3. Preliminaries

### 3.1 记号

用户行为序列数据：
$$\mathcal{V} = \{\{B_n\}_{n=1}^N, \{S_n\}_{n=1}^N, \{M_n\}_{n=1}^N\}$$
- $B_n \in \mathcal{T}^{L_n(B)}$：PayBill 原始文本；
- $S_n$：SPM 记录文本；
- $M_n$：MiniProgram 描述文本；
- $T \in \mathbb{R}^{N \times F \times D}$：Tabular，$F$ 特征数、$D$ 特征维度；
- $R \in \mathcal{T}^{L_n(R)}$：Search Text；
- $Q \in \mathcal{T}^{L(Q)}$：一句话定向需求。

### 3.2 User Targeting 目标定义

预训练目标（Eq. 1）：
$$g_{\theta^\star} = \arg\min \mathbb{E}_{V_n^s \in \mathcal{V}^s}\mathcal{L}_{pretrain}(g_\theta; V_n^s)$$

**Zero-shot Transfer**（Eq. 2）：$p_i = \arg\max_{U_i \in \mathcal{U}} P_{\theta^\star}(U_i \mid Q_i)$

**Few-shot Prompt-tuning**（Eq. 3）：
$$g_{\theta'} \in \arg\max_\theta \mathbb{E}_{(Q_i, U_i) \in I_{t\mid tr}} P_\theta(\hat U_i = U_i \mid Q_i)$$
$I_{t \mid tr}$ 是有监督样本，$\hat U_i$ 是预测的用户组。

---

## 4. Method 方法

### 4.1 Forecastable User-Text Pair Preparation

从支付宝全量账单、页面点击、小程序浏览与订单、主页搜索等收集数据，构造用户行为的自然语言描述。**关键点**：**描述基于用户未来行为**（下一时段），从而让模型天然获得预测能力。

user-text pair 形式（Eq. 4）：
$$\mathcal{A} = \{V_{i, t_1}, T_{i, t_1}, R_{i, t_1}; Q_{i, t_2}\}_{i=1}^N$$
$t_1$ 是过去时段（用户历史）、$t_2$ 是未来时段（描述基础）。

**5 类模板**：Purchase / Browsing / Search / Click / Payment Channel。示例：

> Template for purchase category: The user purchased **{items}** amounting more than **{num}** dollars with **{status}** payment.

例："The user purchased gold, gold bars, etc., amounting to more than \$10,000 with successful payment."

### 4.2 Self-Supervised User Modeling (Stage I)

#### 4.2.1 User Behavioral Sequence Encoding

对用户 $i$，行为 embedding：
$$e_i^{(\mathcal{V})} = \mathcal{P}(g_\theta(B_i, M_i, S_i)) \tag{5}$$

$g_\theta = g_\mu \circ g_\nu$：
- $g_\nu$：文本编码器 ALBERT，把序列文本先转 embedding；
- $g_\mu$：时序编码器（GRU / self-attention）；
- $\mathcal{P}$：跨时间窗口的均值池化。

记 $z = g_\nu(\cdot)$，$c = g_\mu(z)$。

**Pre-training Task I — Forecasting & Contrastive Learning**（沿 CPC，Eq. 6）：
$$\mathcal{L}_{CL} = -\frac{1}{Kk}\sum_{t=1}^K \sum_{i=1}^k \log \frac{\exp(s(c_t, z_{t+i}))}{\sum_{j=1}^K \exp(s(c_t, z_{t+j}))}$$

$c_t$ 是聚合了时序信息的表征，$z_{t+i}$ 是未来时刻的 embedding，$s$ 是余弦相似度，$K$ 是总长度，$k$ 是正样本窗口宽度。

**Pre-training Task II — Cyclic Regularization**（Fig. 2b）：账单与小程序浏览常呈**周期性**，加 KL 正则强制同一用户在多个时间窗表征相近（Eq. 7）：
$$\mathcal{L}_{UB} = \mathcal{L}_{CL} + \lambda_{cyc} \text{KL}(c_t \| c_{t+T})$$

#### 4.2.2 Tabular Encoding

采用 TabTransformer 类似的架构 $h_\phi$（Eq. 8）：
- 类别特征 $T_{cat}$：Column Embedding + Transformer；
- 连续特征 $T_{cont}$：MLP 后送同一 Transformer。

**自监督任务**：MLM + RTD。
- MLM（Eq. 9）：$\mathcal{L}_{MLM} = \sum_{m \in \mathcal{M}} \text{CE}(\hat{\tilde T}_m, \tilde T_m)$；
- RTD（Eq. 10）：$\mathcal{L}_{RTD} = \sum_{r \in \mathcal{R}} \text{BCE}(\hat{\tilde T}_r, \tilde T_r)$，各列独立分类器；
- 总损失（Eq. 11）：$\mathcal{L}_{Tab} = \lambda_{MLM}\mathcal{L}_{MLM} + \lambda_{RTD}\mathcal{L}_{RTD}$。

#### 4.2.3 Text Encoding

Search Text 用 ALBERT 做文本编码器 $f_\omega$（Eq. 12）：$e_i^{(r)} = f_\omega(R_i)$。

### 4.3 User-Text Alignment (Stage II)

#### 4.3.1 Attention-based Feature Fusion

跨模态融合（Eq. 13）：
$$e_i^f = \text{CA}(e_i^{(\mathcal{V})}, e_i^{(tab)}, e_i^{(r)})$$

CA 是 cross-attention。

#### 4.3.2 Contrastive User-Text Pre-training

沿用 CLIP 思路。用 LoRA 微调 LLM (QWEN2-1.8B) 编码 future 描述：
$$e_i^q = \text{LLM}(\mathcal{F}(B_{i, t_2}, M_{i, t_2}, S_{i, t_2})) \tag{14}$$

用 InfoNCE 对齐 $e_{i, t_1}^f$ 与 $e_{i, t_2}^q$：
$$\mathcal{L}_{CP} = -\frac{1}{B}\sum_{i=1}^B \log \frac{\exp(s(e_{i, t_1}^f, e_{i, t_2}^q))}{\sum_{j=1}^B \exp(s(e_{i, t_1}^f, e_{j, t_2}^q))} \tag{15}$$

**这就是"用历史预测未来"的核心机制**——user embedding 基于历史生成、text embedding 基于未来行为构造，两者对齐。

### 4.4 User Targeting Downstream

#### 4.4.1 Zero-shot Transfer

预训练完成后直接部署到目标数据；输入一句话需求即可选人。为让非专家用户语言也能被理解，作者加了 **Query Rewriting** 模块（QWEN2-14B 两阶段微调）：
- **Stage 1**：用支付宝多源数据合成 multi-round conversation 注入领域知识；
- **Stage 2**：挖掘不同人的 circling intentions（`<Query - Answers>`）再微调，让 LLM 把自由文本重写成对齐预训练模板的表述。

#### 4.4.2 Few-shot Prompt Tuning

低资源场景下用**种子用户**做 prompt learning——在文本输入前加 learnable prompt token。为处理**hard-negative**（例如 3C 数码维修工 vs 3C 爱好者行为相似），引入**负种子用户** $U^-$，用 triplet loss（Eq. 16）：
$$\mathcal{L}_{Tri}(e^{f+}, e^{f-}, e^q) = \sum \max(0, \|e^q - e^{f+}\|_2 - \|e^q - e^{f-}\|_2 + \alpha)$$

$\alpha$ 控制正负边距。

---

## 5. Experiments 实验

### 5.1 实验设置

**数据集**：
- 预训练：约 5 亿 user-text pair；
- 测试基准：13 个场景，覆盖 Account Security、Marketing、Recommendation；每场景 30 万 – 3000 万样本。
- 主要评测 5 个（Table 1）：
  1. Account Theft Risk（290 万）；
  2. APP Click（33 万）；
  3. APP Activity（74 万）；
  4. Willingness for Phone Renting（28 万）；
  5. Willingness for Credit Facilities（644 万）。

**Baseline**：
- **ARALLM**（GPT-3.5 预训练 + ChatGLM2-6B / QWEN2-7B 微调版）；
- **U-MLP One4all**：One4all 表征 + MLP decoder；
- **MSDP**：Multi-scale Stochastic Distribution Prediction；
- **单模态基线**：TabTransformer + MLM/RTD（tabular）、QWEN2-7B（text）。

**指标**：Targeting 用 Accuracy / Precision / Recall；用户表征用 linear probing AUC / KS。

**实现**：user embedding 维度 1024；ALBERT 编码文本，6 层 Transformer 做时序编码；tabular 用 TabTransformer 配置；跨模态融合用 5 层 cross-attention；LLM 用 QWEN2-1.8B + LoRA；AdamW + cosine decay，lr=4e-4；16×A100 训练、单卡测试。

### 5.2 主要结果

#### Zero-shot User Targeting（Table 2）

FOUND 在 5 个 benchmark 上全面领先，平均比 ARALLM-ft(QWEN2-7B) **Accuracy +10% / Precision +15% / Recall +10%**。示例：
- Test #1 Account Theft Risk：**88.6 / 83.7 / 94.1** vs. ARALLM 83.2 / 76.0 / 91.8；
- Test #5 Credit Facilities：**84.3 / 77.3 / 80.0** vs. 79.3 / 68.3 / 82.1。

#### Few-shot User Targeting（Table 3，10 seed 用户）

对比 U-MLP One4all、ARALLM-ft(ChatGLM2 / QWEN2)：
- Test #1：**95.6 / 93.8 / 97.4**（U-MLP 91.0 / 86.8 / 93.2）；
- Test #4：**84.3 / 80.0 / 88.9**（U-MLP 76.6 / 70.7 / 83.8）。
- 平均比 baseline **Accuracy +5% / Precision +8% / Recall +5%**。

Fig. 3 的 t-SNE 显示：few-shot prompt embedding 更靠近正种子用户、远离 hard-negative 用户。

#### User Representation Evaluation（Linear Probing, Table 4）

FOUND 融合表征比各单模态基线都强：
- 比 User Behavioral Sequence + MSDP AUC **+4% / KS +2%**；
- 比 Tabular AUC **+12% / KS +13%**；
- 比 Search Text AUC **+11% / KS +10%**；
- 比 Concatenation 融合也一致更优。

**关键对比示例**（Test #5）：FOUND **AUC 98.9 / KS 90.7** vs Concatenation 96.1 / 86.4。Fig. 4 的 t-SNE 显示同消费场景用户聚成同色簇。

### 5.3 消融实验（Table 5）

- **w/o User Behavioral Sequence**：性能下降最严重（Test #1 Accuracy 88.6 → 82.6），**行为序列是最关键模态**（贡献 Accuracy/Precision/Recall 各 6%+）；
- **w/o Tabular**：小幅下降 (~2-3%)；
- **w/o Search Text**：小幅下降；
- **w/o Attention-based Fusion**（用 concatenation 替换）：性能下降约 1% / 2% / 1%，验证 cross-attention 融合的价值；
- **Full** 版本在所有指标上一致最佳。

---

## 6. Conclusion 结论

FOUND 是首个**"一句话选人"**的工业级用户定向基础模型，用两阶段预训练把异构多源数据映射到与自然语言对齐的潜空间；同时通过"历史用户表征 + 未来行为文本"的对偶设计天然获得预测能力。

**部署与工业价值**：
- 自 2024 年起在支付宝用户选择平台部署；
- "Used Car Selling Preferences" 用户 CTR **+30%**；
- "Video Watching in Alipay" 用户 CTR **+13.2%**；
- 覆盖 Genshin/Honkai/Eggy Party 等游戏、Jay Chou/迪士尼等 IP、多个城市 One-Day Travel、金融理财等 25 个广告场景（Fig. 6）——Precision 从 ARALLM ~0.13 提升到 FOUND ~0.30–0.77。

**局限与未来工作**：
1. 跨平台泛化未充分验证——计划扩展到图像 / 视频 / 知识图谱等更多模态；
2. 训练成本仍高——两阶段策略相比 from-scratch 已降本，但仍需进一步优化。

---

## 附：术语中英对照速查

| 英文 | 中文 |
|---|---|
| User Targeting | 用户定向 |
| Forecastability | 前瞻预测能力 |
| Foundation Model | 基础模型 |
| Two-Stage Pre-training | 两阶段预训练 |
| Self-Supervised User Modeling | 自监督用户建模 |
| User-Text Alignment | 用户-文本对齐 |
| Contrastive Predictive Coding (CPC) | 对比预测编码 |
| Cyclic Regularization | 周期性正则化 |
| Masked Language Modeling (MLM) | 掩码语言建模 |
| Replaced Token Detection (RTD) | 替换 token 检测 |
| Cross-Attention Feature Fusion | 交叉注意力特征融合 |
| Query Rewriting | 需求重写 |
| Zero-shot Transfer | 零样本迁移 |
| Few-shot Prompt Tuning | 少样本提示微调 |
| Hard-Negative Sample | 困难负样本 |
| Triplet Loss | 三元组损失 |
| Linear Probing | 线性探测 |
| CTR (Click-Through Rate) | 点击通过率 |
| PayBill / MiniProgram / SPM | 支付账单 / 小程序 / SPM 位置模型 |
| ALBERT / QWEN2 / GPT-3.5 / ChatGLM2 | 各基线 LLM 名称 |
