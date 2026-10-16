# InstructUE：指令感知的用户嵌入基础模型 · 中文技术精读稿

> **原文**：Instruction-aware User Embedding via Synergistic Language and Representation Modeling
> **arXiv**：2510.11016v1（Oct 2025）
> **作者**：Ziyi Gao\*, Yike Xu\*, Jiahao Yuan, **Baokun Wang†（通信作者）**, Jinyong Wen, Xiaotong Lin, Yun Liu, Xing Fu, Yu Cheng, Yongchao Liu, Weiqiang Wang, Zhongle Xie
> **单位**：Ant Group + Zhejiang University；\* 共同一作；† 通信作者
>
> **本文档说明**：本人对论文的中文技术阅读稿，覆盖 Abstract + Introduction + Method + Experiments + Ablation + Conclusion 六个高价值章节。原文公式编号保留；术语给中英对照。若需精确表达请对照原文 PDF (`pdfs/2510.11016.pdf`)。

---

## Abstract 摘要

论文提出 **InstructUE**——首个基于 LLM 的**指令感知（instruction-aware）用户嵌入基础模型**。核心思想：用自然语言指令**动态调制**用户表征，同一用户可以在不同下游任务下产出不同 embedding。

四项主要设计：
1. **多编码器 + 轻量 adapter 架构**：为 6 类异构数据源分别编码，保留各自结构特征；
2. **UserQA 数据集**：一份大规模、领域无关的合成数据，用来桥接语言空间与表征空间；
3. **对比-自回归联合训练框架**：autoregressive 目标注入领域知识，contrastive 目标做 user-text embedding 对齐，实现"指令引导的降噪"；
4. **Few-shot cluster-based supervised instruction tuning**：把有限带标签样本变成"改良指令"用来 fine-tune。

在真实工业场景（用户预测、营销、推荐）上显著超过现有 baseline。

---

## 1. Introduction 引言

**背景**：用户表征模型的目标是把大规模多源数据（文本画像、交互历史、表格属性）编码成表达力强的 embedding，服务数字营销、推荐、金融等。传统对比学习通过序列内上下文一致性、跨会话一致性等自监督目标学表征，但当行为**越来越序列化、越来越 context 敏感**时，简单的历史压缩式表征已经不够——**有效表征必须能预测未来行为**，这需要更强的序列建模与语义整合能力。

**LLM 路线的两个关键问题**：
- **模态失配（Modality Mismatch）**：用户行为是**稀疏、符号化**序列；而 LLM 预训练面对的是**稠密自然语言**。直接用 LLM 编码行为存在语义 gap。
- **静态 embedding 的僵化**：现有 LLM-based user embedding（BERT4Rec、FOUND、Qwen3-Embedding 等）通常产出**静态、任务无关**的 embedding，无法根据下游指令调整。缺乏指令引导时，模型很容易受**噪声或无关行为**干扰。

**关键洞察**：**指令的质量**极大影响 LLM 用户表征质量——不精确的指令会让 embedding 降级。

**方法核心**：把用户表征重构为**指令条件化过程**（instruction-conditioned process）——用自然语言指令灵活引导 embedding 生成。具体做法：
1. 6 类异构数据源各自的 modality-specific encoder + 轻量 adapter 投影到 LLM 空间，避免"暴力拼接"造成的模态冲突与上下文溢出；
2. 通过 **UserQA** 数据集与 **对比-自回归联合训练**桥接语言空间与表征空间：autoregressive 注知识、contrastive 做指令引导的降噪；
3. **Few-shot cluster-based instruction tuning**：仅用少量下游标签就能优化指令 token，把业务知识"编"到指令里。

**核心贡献**：
1. 首个工业级、基于 LLM 的**指令感知用户嵌入基础模型**；
2. Contrastive-Autoregressive 联合训练框架 + few-shot 指令 tuning；
3. 多域实验证明 InstructUE 在迁移性 / 泛化性上均领先。

---

## 2. Related Work 相关工作（简要）

- **Text Embedding**：早期监督式 BERT 派 → SimCSE 无监督对比 → LLM2Vec、InBedder、Qwen3-Embedding 等 decoder-only embedder。启发本文——**指令驱动 + 生成范式**能提升迁移性。
- **User Representation**：DUPN、MSDP（RNN + 预测目标）→ 跨会话对比（Lin et al., Zou et al.）→ 基础模型 FOUND。但仍是"用户一份静态 embedding"。InstructUE 引入指令感知，打破静态假设。
- **LLM for User Modeling**：BERT4Rec、RecBole、语义丰富方法（Hou 等）——大多依赖 next-item / masked-action 浅目标，跨域泛化有限。InstructUE 用指令条件 + 自回归-对比策略同时注知识、抑噪声。

---

## 3. Method 方法

### 3.1 问题形式化

把 instruction-aware user embedding 看作**条件嵌入任务**：
$$u_{i,k} = f(X_i, I_k) \in \mathbb{R}^d$$

$X_i = \{B_i, M_i, S_i, R_i, T_i\}$ 是用户 $i$ 的多模态画像：
- $B_i$：PayBill 交易序列；
- $M_i$：Mini Program 交互；
- $S_i$：SPM path；
- $R_i$：homepage search queries；
- $T_i \in \mathbb{R}^{F \times D}$：$F$ 维表格特征，$D$ 维 embedding。

$I_k \in \mathcal{I}$ 是自然语言指令。

**双模态推理**：
- **有指令**：$u_{i,k} = f(X_i, I_k)$——只保留与 $I_k$ 语义相关的行为；
- **无指令**：$u_i = f(X_i)$——holistic encoder，跨模态融合成通用表征。

自然语言指令因此成为**潜空间的软控制器**，让同一个模型既能生成通用表征，又能在业务场景下做精细语义引导。

### 3.2 UserQA：合成指令-响应数据

每条样本是三元组 $(X_i, I_k, A_k)$：$X_i$ 是用户画像，$I_k$ 是自然语言指令，$A_k$ 是"该用户在该指令下应生成的答案"。

**输入格式**（Fig. 3）：把各模态包裹在 `<bill>...</bill>`、`<minipro>...</minipro>` 等语义分隔 token 中；可选的 instruction 紧随其后；最后附特殊 `<USER>` token 表示"从这里抽 unified user embedding"。

数据合成沿两条互补轴：

**(a) Mine stable user attributes** — 从历史多源数据中挖掘稳定属性，让 LLM 生成 instruction-answer pair。示例：频繁高端餐厅 PayBill → Q: "该用户的餐饮偏好？" → A: "该用户偏爱高端餐饮场所"。→ 模型学到从复杂输入抽稳定语义画像。

**(b) Forward-looking business-relevant instructions** — 基于用户未来行为构造前瞻性指令。示例：外卖记录 → Q: "预测该用户下月外卖详细开销" → A: LLM 生成合理预测（基于时序趋势）。→ 模型学到预测型表征。

**通用表征训练**：设 $I_k = \emptyset$，鼓励模型对完整画像做综合编码。

### 3.3 Architecture 架构

**语言处理**：以预训练 LLM 为主干（Qwen2.5-0.5B-Instruct）；每种文本类数据源经**独立的 user encoder + adapter**投影到 LLM embedding 空间。文中 user encoder 用 BERT-based 模型（gte-base），adapter 用单层 MLP；数值型 tabular 只经 adapter。

**多模态输入**（Fig. 2）：各模态 token 序列被压缩到统一 LLM 空间；末尾追加 `<USER>` token，其最后一层 hidden state 即 unified user embedding。

### 3.4 对比-自回归联合训练策略

**自回归目标（Autoregressive Language Learning）** — 用 teacher forcing 预测下一 token：
$$\mathcal{L}_{lm} = -\sum_{t=1}^T \log p(y_t \mid y_{<t}, x_i, s_i) \tag{1}$$

两大好处：(i) 保留 LLM 生成能力，通过预测 response token 分布注入领域知识；(ii) 依据 InBedder 观察——**语义相似文本在同一指令下产出相似 response，从而 embedding 更相似**——把相似用户在同一指令下拉近，进而增强表征。

**对比学习** — 把 $u_i$ 与 $a_i$ 对齐，鼓励相似用户 embedding 聚类。InfoNCE 损失（Eq. 2-3）：
$$\mathcal{L}_{cl} = -\frac{1}{B}\sum_{i=1}^B \log \frac{e^{s(\hat u_i, \hat a_i^+)/\tau}}{Z_i}$$
$$Z_i = e^{s(\hat u_i, \hat a_i^+)/\tau} + \sum_{j \ne i} m_{ij}\big[e^{s(\hat u_i, \hat a_j)/\tau} + e^{s(\hat u_i, \hat u_j)/\tau} + e^{s(\hat a_i, \hat a_j)/\tau}\big]$$

**同侧负样本 + 假负样本 mask**（Eq. 4）：
$$m_{ij} = \begin{cases} 0 & s_{ij} > s(\hat u_i, \hat a_i^+) + c_{margin} \\ 1 & \text{otherwise} \end{cases}$$

同侧（user-user、answer-answer）负样本能加强"不同指令下的 user 表征"与"不同 answer 表征"间的区分度。

**总目标**：
$$\mathcal{L} = \alpha_{lm} \mathcal{L}_{lm} + \alpha_{cl} \mathcal{L}_{cl} \tag{5}$$

**互补机制**：
- **Autoregressive**：token 级、sequence-level 语义理解；
- **Contrastive**：instance 级、判别性表征结构。

两者在不同粒度上协同，共同构建更泛化的用户理解。

### 3.5 Instruction Precision Matters（指令质量至关重要）

**观察**：不精确指令 → 潜空间聚类模糊 → 下游表现下降。例如"产品偏好"这类概念在模型理解与实际业务定义之间存在语义 gap。

**Few-shot Cluster-based Instruction Tuning**（Eq. 6）：只 fine-tune 若干可学 instruction token，冻结 LLM 主干与 encoder。用 supervised contrastive with class prototypes：
$$\mathcal{L}_{pt} = -\frac{1}{B}\sum_{i=1}^B \log \frac{e^{s(\hat u_i, \hat p_{l_i})/\tau}}{\sum_k e^{s(\hat u_i, \hat p_k)/\tau}},\quad p_k = \frac{1}{|C_k|}\sum_{i \in C_k} u_i$$

$p_{l_i}$ 是标签 $l_i$ 对应类的原型（该类内 user embedding 均值）。把用户 embedding 拉近正确类原型、推离其他类原型——**"业务知识被编码进 instruction token"**，无需重训 backbone。

---

## 4. Experiments 实验

### 4.1 数据与设置

- **训练集**：UserQA 2 亿样本；
- **测试**：6 个场景（Table 1），每个 100 万样本，覆盖：
  1. User Prediction — Ant Forest Click
  2. User Prediction — Game Top-up
  3. Basic Attributes — Public Transit
  4. Basic Attributes — Purchasing Power
  5. Recommendation & Marketing — Preference For Food
  6. Recommendation & Marketing — Preference For Film
- **评测**：Linear probing，每个 test set 50% fit + 50% eval，非重叠；
- **指标**：AUC (%) + KS；
- **Backbone**：Qwen2.5-0.5B-Instruct；user encoder：gte-base。

### 4.2 主结果（Table 2）

**三种推理模式**：
- $\text{InstructUE}_{Universal}$：无指令 → 通用 embedding；
- $\text{InstructUE}_{Instruction-manual}$：人工撰写指令；
- $\text{InstructUE}_{Instruction-tuned}$：Few-shot supervised 微调后的指令 token。

**核心结论**：
- $\text{InstructUE}_{Instruction-tuned}$ 在 6 个任务中 5 个取得 SOTA；
- **Purchasing Power** AUC = **95.69**（+1.14 vs FOUND 94.55）；
- **Food Preferences** AUC = **85.57**（vs FOUND 84.69）；
- **Film Preferences** AUC = **86.45**（比 FOUND 高 1.01 个百分点）；
- 全面胜过特化 user embedding（U-MLP、MSDP、CPC）和通用 LLM embedding（Qwen3-Embedding-0.6B），验证了"LLM 世界知识 + 多源行为微调"的价值。

### 4.3 指令的影响（Fig. 5 t-SNE）

- **Universal embedding**：所有场景样本形成**单一无差别云**；
- **Instruction-aware embedding**：分成**6 个清晰的、互相分离的簇**，每簇对应一个下游任务。

指令**重组表征空间**——增强类内凝聚、类间分离。

**性能对应结论**：
- Game Top-up 场景加人工指令 AUC 从 97.92 → 98.12；
- 但 Ant Forest Click / Purchasing Power 这类**业务定义复杂/模糊**的场景，人工指令反而使性能下降（不精确指令误导模型丢弃有用信号）；
- **Instruction-tuned** 通过 few-shot 优化把 Food 从 83.80 → 85.57、Purchasing Power 从 94.50 → 95.69，弥合了"模型语义 vs 业务语义"的 gap。

### 4.4 消融实验（Table 3）

**模态维度**：
- **Text Only**（把所有数据都当 plain text 交给 LLM）：Food 84.35 / Film 85.43，弱于用 modality-specific encoder 的 w/o Tabular（85.12 / 85.53）——证明**模态专属编码优于纯文本拼接**；
- **w/o Sequence**（只用 tabular）：Ant Forest 97.14 → 94.95，显著下降；
- **w/o Tabular**：多数任务小幅下降；
- 两类模态互补，都有正贡献。

**训练目标维度**：
- **w/o Contrastive**（只 autoregressive）：Purchasing Power 95.57 → 93.63；
- **w/o NTP**（只 contrastive）：Purchasing Power 95.31（下降但更温和）；
- 结论：**contrastive 是主结构、autoregressive 是知识注入 + 稳定项**；两者互相强化。

**表征选择维度**：
- **Avg Pooling**（所有 token 平均）：略低于用 `<USER>` token 的默认方案；
- 原因：causal 架构下最后一个 token 天然是 summary 位置，用它比平均更能利用 decoder 的信息累积特性。

---

## 5. Conclusion 结论

作者提出 **InstructUE**——首个指令感知的 LLM-based user embedding foundation model：
- 用 LLM 的语义能力生成**动态、任务专属**的用户 embedding；
- 核心创新：**Contrastive-Autoregressive 联合训练**，同时对齐语言空间与表征空间；
- Few-shot instruction tuning 是把业务语义"注入"指令 token 的高效手段。

**实验表明**：InstructUE 在多域一致胜过 SOTA；**指令质量至关重要**，few-shot 优化是关键杠杆。

---

## 附：术语中英对照速查

| 英文 | 中文 |
|---|---|
| Instruction-aware Embedding | 指令感知嵌入 |
| Instruction-conditioned Process | 指令条件化过程 |
| UserQA Dataset | 用户 QA 数据集 |
| Contrastive-Autoregressive Training | 对比-自回归联合训练 |
| Autoregressive Language Learning (NTP) | 自回归语言建模 / 下一 token 预测 |
| InfoNCE Contrastive Loss | InfoNCE 对比损失 |
| Same-side Negative Sample | 同侧负样本 |
| False Negative Mask | 假负样本掩码 |
| Prototypical Contrastive Loss | 原型对比损失 |
| Few-shot Cluster-based Instruction Tuning | 少样本聚类指令微调 |
| Modality-specific Encoder | 模态专属编码器 |
| Adapter | 适配器 |
| Linear Probing | 线性探测 |
| AUC / KS | ROC 曲线下面积 / Kolmogorov-Smirnov 分离度 |
| t-SNE | t-SNE 二维可视化 |
| PayBill / Mini Program / SPM | 支付账单 / 小程序 / SPM 位置模型 |
