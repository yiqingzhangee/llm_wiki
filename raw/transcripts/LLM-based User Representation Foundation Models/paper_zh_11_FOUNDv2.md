# FOUNDv2：面向用户表征的统一量化 Tokenizer · 中文技术精读稿

> **原文**：FOUNDv2: Learning Unified User Quantized Tokenizers for User Representation
> **arXiv**：2508.00956v3（Jun 2026）
> **会议**：KDD 2026（DOI 10.1145/3770855.3818413）
> **许可**：Creative Commons BY-NC-ND 4.0
> **作者**：Chuan He, Yang Chen, Bin Dou（共同一作），Wuliang Huang，**Baokun Wang（通信作者）**，Yongchao Liu, Xing Fu, Yu Cheng, Chuntao Hong, Weiqiang Wang, Zhongle Xie, Jiajun Zheng, Xin-Wei Yao
> **单位**：Ant Group + 浙江工业大学 + 浙江大学
> **代码**：`https://github.com/chuanhe1999/FOUNDv2`
>
> **本文档说明**：这是本人对论文的中文技术阅读稿，覆盖 Abstract + Introduction + Method + Main Results + Conclusion 五个高价值章节，涉及原文技术论点、公式（保留原编号）与实验数据。术语保留英文并给中文对照。若需精确表达，请对照原文 PDF（`pdfs/2508.00956.pdf`）。

---

## Abstract 摘要

用户表征学习是大规模 Web 平台个性化服务的基础。传统连续 embedding 方法面临三大挑战：
1. 缺乏统一的多源数据整合范式；
2. 信息密度低，存储开销大；
3. 缺乏多尺度建模粒度。

作者提出 **FOUNDv2**，以 **Unified User Quantized Tokenizer (U²QT)** 为核心的综合用户表征方案，通过两阶段架构把异构用户数据转成标准化离散 token 空间：
1. **紧凑特征表示抽取**：用预训练 embedding 模型（Qwen3-Embedding）把多源数据映射到共享语义空间；
2. **多视角 RQ-VAE（MRQ-VAE）**：用「共享 + 源特定」层次 codebook 把连续 embedding 离散化为存储高效的 token。

进一步设计**多尺度对齐 pretext 任务**：token 级、window 级、语义级，让 U²QT 兼具细粒度行为依赖建模与宏观时序周期性建模。实验表明 FOUNDv2 在多个 benchmark 上一致优于任务专属 baseline，同时显著降低存储与算力开销。已在支付宝多场景规模化部署验证效果。

---

## 1. Introduction 引言

### 1.1 工业用户建模的挑战

现代 Web 平台（如支付宝）产生海量多源异构数据：文本画像、行为序列、结构化表格属性。把这些多元输入蒸馏成鲁棒、通用的用户表征既是技术挑战也是战略必需。当前主流路线从简单统计特征演进到复杂的**late-fusion 范式**：
- **User2Vec**：源专属 embedding pooling；
- **MFN、MSSR**：引入注意力捕获跨源交互；
- **One4all、MSDP**：大规模预训练与序列建模；
- **FOUND**：把用户行为对齐到自然语言，实现零样本迁移。

作者认为这些 SOTA 都停留在"分开编码后融合"架构：模块化虽简单，但**每个数据源初始是隔离处理**，天然限制了模型捕捉深层跨模态协同的能力。

### 1.2 三大系统性瓶颈

作者指出主流 late-fusion 范式已到瓶颈：

**(i) 异构数据缺乏统一表征**：传统方法用专属编码器 + 简单拼接聚合，把不同模态当作孤立实体，无法捕捉跨源协同与复杂依赖。

**(ii) 高数据容量但低信息密度**：实验发现——在固定 embedding 大小下增加行为的时间跨度会带来显著性能提升，说明更稠密的表征本质上更有表达力，这与 LLM 的 densing law（Xiao et al. 2025）一致。但现有方法依赖高维稀疏连续向量，形成瓶颈，阻碍平台向长期用户历史扩展。

**(iii) 缺乏多尺度建模粒度**：传统连续向量把用户表征当作单块静态 embedding，缺乏内部离散结构，很难从数学上刻画不同时间尺度上的用户动态。缺少统一 tokenizer 使模型无法弥合"微观事件转移"与"宏观兴趣演化"之间的 gap。

### 1.3 FOUNDv2 方案与贡献

作者提出以 U²QT 为核心的统一用户表征方案，两阶段架构：

**第一阶段**：用预训练 embedding 模型把文本描述、序列行为、表格属性投影到共享语义空间，消解多源分裂同时保留语义细节。

**第二阶段**：**MRQ-VAE（Multi-View Residual Quantized VAE）**——用「共享 + 源特定」的层次 codebook 把连续 embedding 离散化为 token；层次设计让模型同时捕捉跨域共性与源专属行为模式。

**量化收益**：
- 相较 FOUND，内存占用降低 **30×**，训练加速 **3.5×**；
- 在相同 GPU 显存下，可管理历史窗口从 60 天扩展到 180 天（+200%）；
- 显著增强长期用户动态建模能力。

在此之上引入**多尺度 pretext**：token 级未来行为对齐、window 级长周期性建模、CLIP-style 语义对齐，让 tokenizer 兼具时间智能与预测能力。

**四项核心贡献**：
1. 提出 FOUNDv2 及 MRQ-VAE 技术——据作者所知，首个把多源异构用户数据转成标准化离散 token 空间的框架；
2. 相对 SOTA 内存 −30×、训练 +3.5×，时序上下文 +200%；
3. 设计 token / window / 语义三级 pretext 任务，赋予 tokenizer 多面时间智能；
4. 在真实多任务电商 benchmark 与工业环境中一致优于 SOTA。

---

## 2. Related Work 相关工作（简要）

- **量化语言（Quantitative Language / Semantic ID）**：在 VLLM 和生成推荐（如 MMGRec、MQL4GRec）中兴起；视觉侧 SEED、UniTok、SemHiTok 展示多 codebook / 层次结构能解耦语义与像素。
- **通用用户表征**：早期用 DUPN、PeterRec、SUMN 等做多任务学习与冷启动迁移；近期 Conure、TERACON 用持续学习缓解灾难性遗忘；最新以 LLM 为核心的 NoteLLM 和 HLLM 把行为序列当"语言"输入。

---

## 3. Preliminaries 记号与问题定义

用户行为序列数据记作
$$\mathcal{V} = \{(B_n, S_n, M_n, A_n)\}_{n=1}^{N}$$
分别对应 PayBill $B$、Super Position Model $S$、MiniProgram $M$、App $A$。$B_n \in \mathcal{T}^{L_n^{(B)}}$（$\mathcal{T}$ 是 token 字典），其余同理。

表格数据 $T \in \mathbb{R}^{N \times F \times D}$，搜索文本 $R_n \in \mathcal{T}^{L_n^{(R)}}$。每个用户可表示为 $X_n = (V_n, T_n, R_n) \in \mathcal{X}$。

**问题定义**：学习统一用户表征 $U$，即最小化预训练目标
$$g_{\theta^*} = \arg\min_\theta\ \mathbb{E}_{X_n^s \in \mathcal{X}^s}\ \mathcal{L}_{pre}(g_\theta; X_n^s) \tag{Eq. 1}$$

得到 $U = g_{\theta^*}(X)$ 后，下游任务
$$\hat y_n^i = \mathcal{P}_{\omega^i}(U_n) \tag{Eq. 2}$$

是典型的两阶段范式：先产生用户表征，再喂给任务专属模型。

---

## 4. Methodology 方法

FOUNDv2 由三步组成（Fig. 2）：多源编码 → 量化 tokenizer → 多尺度对齐预训练。

### 4.1 多源编码（§4.1）

传统方法对每种数据源独立编码后再融合，跨源语义 gap 难以填补。作者用 LLM 把多源数据映射到共享语言空间：用 **Qwen3 Embedding** 模型，把 query 与源数据拼接成 prompt 后取 `[EOS]` 隐藏态作为源级 embedding。表格因 LLM 对结构化数据表现次优，改用 MLP：
$$\hat H_n^{(V)} = \text{LLM}(Query, V_n, [EOS]),$$
$$\hat H_n^{(T)} = \text{MLP}(T_n),$$
$$\hat H_n^{(R)} = \text{LLM}(Query, R_n, [EOS]) \tag{Eq. 3}$$

每个源被压缩为 $\hat H_n^{(X)} \in \mathbb{R}^{1 \times d}$，把很长的上下文浓缩成单个 embedding。

### 4.2 量化用户 Tokenizer（MRQ-VAE, §4.2）

**动机**：把连续表征压到离散空间以节省存储和算力，同时显式建模用户数据的异构多源特性。

**流程**：对每个源 $x \in \{V, T, R\}$，先经共享 MLP $\mathcal{M}$ 和 mean pooling $\mathcal{P}$ 得到
$$\bar H_n^{(x)} = \mathcal{M} \circ \mathcal{P}(\hat H_n^{(x)}) \in \mathbb{R}^{d_c}$$

再送入 MRQ-VAE 编码为潜表征 $z_n^{(x)}$，接着做**两阶段层次量化**：

- **共享 codebook 阶段**（$L_c$ 层）：使用共享 codebook $\mathcal{C}^{l,(S)} = \{v_k^{l,(S)}\}_{k=1}^{K}$ 捕捉跨源语义共性；
- **源特定 codebook 阶段**（$L_u$ 层）：使用源专属 codebook $\mathcal{C}^{l,(x)}$ 细化源专属模式。

第 $l$ 层的残差量化过程：
$$c_n^l = \arg\min_k \|r_n^l - v_k^l\|_2^2, \tag{Eq. 4}$$
$$r_n^{l+1} = r_n^l - v_{c_n^l}^l, \tag{Eq. 5}$$

初始残差 $r_n^0 = z_n$。经过全部 $L = L_c + L_u$ 层后，量化表征为
$$\hat z_n^{(x)} = \sum_{l=1}^{L_c} v_{c_n^l}^{l,(S)} + \sum_{l=L_c+1}^{L_c+L_u} v_{c_n^l}^{l,(x)} \tag{Eq. 6}$$

这种设计把「共性知识（共享 codebook）」与「源专属模式（专属 codebook）」显式解耦。

**解码与训练**：$\hat z_n^{(x)}$ 送入源专属 MLP decoder 重构原始 Qwen3 embedding。损失包含重构与残差量化两部分：
$$\mathcal{L}_{re} = \|\hat X_n - X_n\|_2^2 \tag{Eq. 7}$$
$$\mathcal{L}_{rq} = \sum_{l=1}^{L} \Big[ \|\text{sg}[r_n^l] - v_{c_n^l}^l\|_2^2 + \alpha\|r_n^l - \text{sg}[v_{c_n^l}^l]\|_2^2 \Big] \tag{Eq. 8}$$

其中 $\text{sg}[\cdot]$ 为 stop-gradient，$\alpha$ 是系数。总损失 $\mathcal{L}_{re} + \mathcal{L}_{rq}$ 同时保证重建准确性与 codebook 向量的紧凑贴合。

**Token 冲突回退机制**：由于多用户可能被分配同一组 token（<0.5% 案例），作者引入
$$(e_1, e_2, \dots, e_n) = \arg\text{sort}_{engage}(U_1, U_2, \dots, U_n) \tag{Eq. 9}$$

用与业务相关的排序函数给冲突 token 分配唯一 special token，确保下游可识别性。最终一个用户由领域 token + special token 组成，例如
$$U_n = \{\langle s_n^1\rangle, \dots, \langle t_n^2\rangle, \langle s_n^1\rangle, \dots, \langle r_n^2\rangle, \langle s_n^1\rangle, \dots, \langle v_n^2\rangle, \langle e_n^1\rangle\}.$$

### 4.3 多尺度对齐 Pretext 任务（§4.3）

用户行为的复杂动态需要多粒度建模，作者设计三级对齐：

#### 4.3.1 Token 级对齐（局部依赖）

对分成离散 token 的行为序列 $Z = \{z_1, z_2, \dots, z_K\}$，采用自监督预测损失：
$$\mathcal{L}_{token} = \sum_{k=1}^{K-1} \omega_k \big(1 - \varphi(h_k, T_\theta(z_{k+1}))\big) \tag{Eq. 10}$$
$$\omega_k = 1 + k/\tau_t \tag{Eq. 11}$$

$h_k$ 是 Transformer 对第 $k+1$ 个 token 的预测表征，$T_\theta(z_{k+1})$ 是预训练 codebook 中对应的 ground-truth；$\varphi$ 为余弦相似度；位置感知权重 $\omega_k$ 给近端 token 更高重要性。

#### 4.3.2 Window 级对齐（宏观周期）

把长期轨迹划分为连续时间窗口 $\mathcal{W} = \{W_1, \dots, W_N\}$。为捕捉重复行为 motif，用 InfoNCE 拉近时序相关窗口的表征：
$$\mathcal{L}_{win} = -\frac{1}{B}\sum_{i=1}^{B} \log \frac{\exp(s(h_{W_i^a}, h_{W_i^b})/\tau_w)}{\sum_{j=1}^{B} \exp(s(h_{W_i^a}, h_{W_j^b})/\tau_w)} \tag{Eq. 12}$$

$h_{W_i^a}$ 与 $h_{W_i^b}$ 是同一用户呈现周期性相似的两个窗口聚合表征（例如不同周同一天）。这让 U²QT 内化"超越即时行为"的长程周期性。

#### 4.3.3 语义级对齐（CLIP-style）

用 template $\mathcal{F}$ 把 $t_2$ 时刻的未来行为（购买物品 $B$、金额 $S$、支付状态 $M$）转成结构化自然语言叙述，用 LoRA 微调后的 LLM 编码为语义 embedding：
$$e_i^{nl} = \text{LLM}\big(\mathcal{F}(B_{i,t_2}, S_{i,t_2}, M_{i,t_2})\big) \tag{Eq. 13}$$

用 InfoNCE 把融合 tokenizer 表征 $e_{i,t_1}^{f}$ 与未来语义 embedding $e_{i,t_2}^{nl}$ 对齐：
$$\mathcal{L}_{sem} = -\frac{1}{B}\sum_{i=1}^{B} \log \frac{\exp(s(e_{i,t_1}^f, e_{i,t_2}^{nl})/\tau_s)}{\sum_{j=1}^{B} \exp(s(e_{i,t_1}^f, e_{j,t_2}^{nl})/\tau_s)} \tag{Eq. 14}$$

**总预训练目标**：
$$\mathcal{L}_{total} = \lambda_1 \mathcal{L}_{token} + \lambda_2 \mathcal{L}_{win} + \lambda_3 \mathcal{L}_{sem} \tag{Eq. 15}$$

作者默认 $\lambda_1 = 0.2,\ \lambda_2 = 0.4,\ \lambda_3 = 0.4$。

---

## 5. Experiments 实验

### 5.1 实验设置

- **数据**：约 2000 万异构多场景用户记录做预训练；测试集覆盖 3 大领域 4 个任务，每任务约 50 万记录：
  1. Willingness for Takeout（外卖偏好）；
  2. Purchasing Power（购买力）；
  3. Money Laundering（洗钱风险）；
  4. Preference for Games（游戏偏好）。
- **Baseline**：MSDP、One4all、FOUND；此外接下游 CTR 模型 MaskNet 做推荐场景验证。
- **实现**：Qwen3-Embedding-0.6B 生成 1024 维源 embedding；MRQ-VAE 用 4 层共享 codebook + 每个数据源 2 层专属 codebook，每层 1024 项 × 256 维；$\lambda_1/\lambda_2/\lambda_3 = 0.2/0.4/0.4$；AdamW 学习率 1e-3；16 张 PPU-810E（每卡 100 GB 显存）。
- **指标**：AUC、KS（分类任务）；AUC、HR（推荐任务）。

### 5.2 主结果（RQ1，Table 2）

FOUNDv2 在四个下游任务上一致超过 baseline，平均 AUC/KS 比 FOUND 分别提升约 3% / 9%。

**关键对比**：
- 把预训练 tokenizer 作为**补充特征**接到 One4all / MSDP 上，One4all+UT 相对原版 AUC +1.2% / KS +3.0%；MSDP+UT AUC +0.8% / KS +2.3%；
- 完整 FOUNDv2 (MRQ-VAE) 在所有四个 test 集上都优于 vanilla RQ-VAE 变体（Table 2 最后两行），验证「共享 + 源专属」层次架构的必要性。

### 5.3 消融实验（RQ2）

**Codebook 设计**（Table 3）：`2 shared + 2 specific` (256×128) 平均 AUC 0.7662 > `4 shared`（0.7578）> `4 specific`（0.7606）。继续扩容到 `4 shared + 2 specific` (256×128) → 0.7677，再扩到 (1024×256) → **0.7746**。扩容持续带来收益。

**多尺度 pretext**（Table 5）：单加 semantic 对齐 → 0.7679；加 window → 0.7707；加 token → 0.7708；三级全开 (S+T+W) → **0.7732**；再把序列长度翻倍 → 0.7746。三级互补且随长序列进一步提升。

**多源数据消融**（Table 4）：从 All Sources 出发，逐个移除数据源都会带来一致下降；其中删除 Bill 对下游 Takeout / Purchasing Power 影响最大（Test #1 AUC 0.8399 → 0.6200），因为账单数据直接编码消费行为。

### 5.4 效率分析（RQ3）

**存储与训练效率**（Table 6）：2000 万用户 90 天数据（原始 240 GB）——
- FOUND：训练 ≈7 小时，表征存储 240 GB；
- FOUNDv2：训练 ≈2 小时（**3.5×**），存储 8.2 GB（**30×** 压缩）。

**信息密度**（Table 7）：固定 batch size 256——
- FOUND 60 天：显存 ≈60 GB，AUC 0.7613 / KS 0.4259；
- FOUNDv2 60 天：显存 ≈3 GB，AUC 0.7633 / KS 0.4291；
- FOUNDv2 90 天：显存 ≈5 GB，AUC 0.7732 / KS 0.4489；
- FOUNDv2 180 天：显存 ≈10 GB，AUC 0.7746 / KS 0.4538。

打破了「序列长度 vs 显存」的取舍，验证 densing 原则：**更稠密表征在有限预算下更具表达力**。

### 5.5 Codebook 分析（RQ4）

**利用率**（Fig. 5）：
- Layer 1 共享 codebook 使用率低——学到少量跨域紧凑模式；
- Layer 2 与源专属 codebook 使用率显著提升——把共性精炼成多元细粒度表征；
- **Tabular** 域利用率最低（数值型语义多样性天然较小），与 App / Bill 等文本型形成对比。

**多源可视化**（Fig. 4）：
- `2 shared + 2 specific`：各领域形成明显分离簇，类内紧凑、类间分明；
- `4 shared`：簇边界模糊、多个域（如 Search 与 SPM）混叠。
- 定性证据强力支持共享 + 专属混合设计。

### 5.6 线上 A/B（RQ5）

**支付宝生态多场景**：
- Scan-to-Pay：定价成本 **−4.71%**；
- Tap-to-Pay：优惠券成本 **−15%**；
- 广告召回策略：以 FOUNDv2 表征替换 OmniRec，
  - Ad CPM **+0.56%**；
  - 广告消费 **+0.50%**；
  - 广告驱动交易金额 **+0.72%**；
  - 总客户价值 **+3.13%**。

---

## 6. Conclusion 结论

作者提出 FOUNDv2——以 U²QT 为核心的综合用户表征框架：
1. 用 Qwen3 embedding 把异构行为编码到语言空间实现多源统一；
2. 用 MRQ-VAE 将连续表征压缩为离散 token，兼顾存储/算力效率；
3. 多尺度对齐 pretext 赋予 tokenizer 时序智能。

大量实验证明 FOUNDv2 具备强表征能力、显著效率优势与灵活的跨任务泛化。工业部署（支付宝 10 亿用户量级）采用增量更新：每天只对当日活跃用户 (4–5 亿) 处理新增行为并追加 4 个 token，避免对历史数据重复计算，大幅降低日推理延迟与资源消耗。

---

## 附：术语中英对照速查

| 英文 | 中文 |
|---|---|
| Unified User Quantized Tokenizer (U²QT) | 统一用户量化 tokenizer |
| Multi-View RQ-VAE (MRQ-VAE) | 多视角残差量化 VAE |
| Shared / Source-Specific Codebook | 共享 / 源特定码本 |
| Residual Quantization | 残差量化 |
| Stop-Gradient (sg) | 停止梯度 |
| Densing Law | 致密律 |
| Token / Window / Semantic Alignment | Token / 窗口 / 语义级对齐 |
| Multi-Scale Pretext Task | 多尺度前置任务 |
| Codebook Utilization | 码本利用率 |
| Linear Probing | 线性探测 |
| AUC / KS / HR | ROC 曲线下面积 / KS / 命中率 |
| PEPNet / OmniRec | 支付宝推荐系统 baseline 模型 |
| Late Fusion / Early Fusion | 后融合 / 前融合 |
