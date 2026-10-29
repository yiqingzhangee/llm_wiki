# Query as Anchor：基于大语言模型的场景自适应用户表征 · 中文技术精读稿

> **原文**：Query as Anchor: Scenario-Adaptive User Representation via Large Language Model
> **arXiv**：2602.14492v2（Feb 2026）
> **会议**：KDD 2026（拟收录）
> **作者**：Jiahao Yuan, Yike Xu, Jinyong Wen, **Baokun Wang（通信作者）**, Ziyi Gao, Xiaotong Lin, Yun Liu, Xing Fu, Yu Cheng, Yongchao Liu, Weiqiang Wang, Zhongle Xie
> **单位**：Ant Group, East China Normal University, Zhejiang University
> **代码**：`https://github.com/JhCircle/Q-Anchor`
>
> **本文档说明**：这是本人对论文的中文技术阅读稿，覆盖 Abstract + Introduction + Method + Main Results + Conclusion 五个高价值章节，涉及原文的技术论点、公式（保留原编号）与实验数据。术语保留英文并给中文对照；关键定义以短引用形式给出并标注引用来源。若需要精确表达，请对照原文 PDF（`pdfs/2602.14492.pdf`）。

---

## Abstract 摘要

论文指出，工业级用户表征学习需要同时兼顾**通用鲁棒性**与**任务敏感性**，但现有方法主要产出**静态、与任务无关**的 embedding，很难在统一向量空间中协调下游多个场景的差异化需求；同时，异构多源数据本身带有噪声和模态冲突，会进一步降低表征质量。

据此，作者提出 **Query-as-Anchor** 框架，把用户建模从「静态编码」转向「动态、query 感知的合成」：

1. 构建工业级预训练数据集 **UserU**，将多模态行为序列与用户理解语义显式对齐；
2. **Q-Anchor Embedding 架构**：把层次化 coarse-to-fine 编码器嵌入双塔 LLM，采用「对比 + 自回归」联合优化目标，产出 query 感知的用户表征；
3. **Cluster-based Soft Prompt Tuning**：桥接通用预训练与业务专属逻辑，在潜空间中形成有区分度的类别结构；
4. **KV-cache 加速部署**：将 query 锚点固定在序列末端，共享用户前缀 KV，多场景推理近乎无额外延迟。

在支付宝 **10 个工业级 benchmark** 上取得一致 SOTA，两个真实业务场景的线上 A/B 均验证了实际效果。

---

## 1. Introduction 引言

### 1.1 用户表征的工业背景

用户表征学习是现代工业智能系统的底座，广泛支撑推荐、数字营销、风险管理等应用。一个用户往往同时活跃于多个业务场景，每个场景有不同的行为模式与决策目标，因此需要既**跨任务可迁移**、又能**适配场景上下文**的表征。当前工业系统一般把大规模异构用户信号（文本画像、交互历史、结构化属性等）压缩成紧凑 embedding 供下游使用。

### 1.2 现有方法的不足

作者按技术路线分类分析：

1. **从零训练或基于 PLM 编码器的对比学习方法**：擅长捕捉历史行为规律，但**天然回顾性**（retrospective），缺乏对下游决策场景的自适应能力。工业系统被迫为每类应用维护一套用户模型，系统复杂度、部署成本、维护成本都很高，且跨域泛化仍是难点。
2. **基于 LLM 作为用户编码器的方法**：通过在行为序列上微调、或引入用户/物品语义来增强序列与语义理解，但现实中的用户行为**稀疏、符号化、异质性强**，与 LLM 预训练时的稠密语言中心数据存在显著分布 gap，限制了 LLM 生成自适应、前瞻性表征的能力。
3. **Instruction-aware embedding**：只提供有限的任务条件化，仍不足以刻画工业用户数据的复杂、动态与噪声特性。

据此，作者总结出三大关键挑战：

- **(1) 场景适配性与跨任务泛化受限**：静态 embedding 无法灵活支持信用评估、营销优化、风险管理等各自需要场景特定用户理解的任务。
- **(2) 模态与语义 gap**：行为日志的符号稀疏性与异构结构，与语言中心的预训练不对齐，制约了 LLM-based 用户表征的跨域泛化。
- **(3) 大规模异构数据整合**：工业用户数据不同场景下的相关性差异极大，需要机制选择性地关注场景相关信号、抑制噪声、压缩大规模行为，同时不能引入过高推理开销。

### 1.3 论文贡献

论文提出把工业级预训练与高效场景适配整合在一起的用户表征框架，核心是 **Query-as-Anchor**——一种 query 条件化机制，把用户行为编码与场景目标解耦。四条主贡献：

1. 构建 **UserU** 工业级预训练数据集，通过「未来行为预测 + QA 式监督」把异构行为与用户理解语义对齐；
2. 提出 **Query-as-Anchor** 框架，在不同下游决策上下文下把同一行为画像**重新锚定**（re-anchor），产生场景自适应 embedding，支持跨业务复用；
3. 引入 **Soft Prompt Tuning + KV-cache 加速**，无需重训 backbone 即可完成场景专属化与低延迟多场景推理；
4. 通过大量离线实验 + 大规模线上 A/B 验证，在用户活跃、风险、营销三大类场景上都取得一致提升，同时显著降低系统复杂度。

---

## 2. Related Work 相关工作（简要）

- **LLM for User Embedding**：从 BERT4Rec、FOUND 等早期基于 masked / contrastive 目标微调 LM 得到静态 embedding，到 Qwen3-Embedding、KaLM-Embedding 等 instruction-aware 变体；但通用语料预训练与稀疏符号化用户行为间的结构 / 语义 gap 仍未闭合。Query-as-Anchor 通过 query 感知的动态 embedding 桥接该 gap。
- **Synthetic Data for User Embedding**：早期靠启发式增强与伪标签；近期用 LLM 直接生成行为、意图、交互序列。作者指出**缺乏大规模、专门为用户 embedding 预训练设计的数据集**是关键瓶颈，UserU 即为此而生，采用「规则化未来行为预测 + QA 理解任务」双组件。

---

## 3. Methodology 方法

作者用户 $u_i = \{Bill_i, Mini_i, Spm_i, App_i, Search_i, Tabular_i\} \in \mathcal{U}$ 定义了一个用户过去 90 天的**多模态交互画像**，其中：

- $Bill_i$：PayBill 交易；
- $Mini_i$：小程序交互；
- $Spm_i$：SPM 导航路径；
- $App_i$：App 级列表；
- $Search_i$：站内搜索 query；
- $Tabular_i \in \mathbb{R}^{1 \times F}$：$F$ 维结构化特征。

该统一画像同时作为 UserU 数据集（§3.1）与 Query-as-Anchor 框架（§3.2）的输入。为高效迁移，进一步用 soft prompt tuning 做下游适配。

### 3.1 UserU 预训练数据集

UserU 由两部分组成：

**(a) Behavior-based Interaction Dataset $D_{future}$**
- 构建方式：对每个用户 $i$，用多模态日志组装 3 个月行为画像 $u_i$，把随后一段时间的交互按时间桶 + 动作类别聚合，选择频率/多样性感知的动作子集作为目标 $a_i^{\text{future}}$；
- 采用固定模板 query $q^{\text{future}}$（例如"用户下一个周期最可能的行为是什么？"），形成训练对：
  $$D_{future} = \{(u_i \oplus q^{\text{future}},\ a_i^{\text{future}})\}_{i=1}^{N},$$
  其中 $\oplus$ 表示上下文拼接。目的是让 embedding 捕捉时间规律并编码近未来行为的预测信号。

**(b) Synthetic Query-Answer Dataset $D_{uqa}$**
- 用**自反思式合成 (self-reflect synthetic)** 生成通用「用户-query-answer」对齐数据。灵感来自 cold-start 策略：
  1. 用 Qwen-Max 生成 72 个生活相关的用户理解主题（金融规划、健康管理等），作为种子池 $\mathcal{P}$；
  2. 对每个用户 $u_i$，让 LLM 从 $\mathcal{P}$ 中检索 Top-10 最相关主题，并把每个主题落地成基于 $u_i$ 的自然 query $q_i$；
  3. 以 $(u_i, q_i)$ 为条件，让 LLM 生成回答 $a_i$；
  4. **反思步骤**：LLM 复核初稿 $a_i$ 与 $u_i$ 是否一致，改写不被支持或不一致的表述。
- 得到 $D_{uqa} = \{(u_i \oplus q_i,\ a_i)\}_{i=1}^{M}$。

### 3.2 Query as Anchor 主框架

#### 3.2.1 分层 coarse-to-fine 用户编码器

为协调多源行为信号的稀疏性与 LLM 需要的稠密语义输入，作者设计层次编码，把原始事件精炼到多粒度表征空间。

对每个模态 $m \in \{Bill, Mini, SPM, App, Search, Tabular\}$：

1. **事件级 embedding**：原始事件序列先由编码器映射为 $\{h_{m,t}\}$，再由 modality-specific event adapter 精炼得到
   $$z_{m,t}^{(evt)} = \text{MLP}_m^{(evt)}\big(\text{LayerNorm}(h_{m,t})\big) \tag{Eq. 1}$$
   以保留每个动作的细粒度原子特征。
2. **模态级 embedding**：对同模态事件做 mean pooling 得到摘要向量 $\bar z_m^{(evt)}$，再经 shared modal adapter 得到统一模态表征
   $$z_m^{(mdl)} = \text{MLP}^{(mdl)}\big(\text{LayerNorm}(\bar z_m^{(evt)})\big).$$
3. **用户级 embedding**：把所有模态向量交给 user adapter，得到全局用户表征 $z^{(usr)}$。
4. **最终输入 token**：把三个层级按结构拼接：
   $$e_i = \Big[ z^{(usr)}\ ;\ \{z_m^{(mdl)}\}_{m}\ ;\ \{\bar z_m^{(evt)}\}_{m}^{K_m} \Big] \tag{Eq. 2}$$
   其中 $K_m$ 是模态 $m$ 保留的事件 token 数量。这样 LLM 可以按 query 动态选择关注细粒度事件还是高层摘要。

输入模板（Fig. 3）：先给出多源用户数据说明，插入 hierarchical token $e_i$，再接可选 instruction，最后放特殊 `<USER_EMB>` token 表示"从这里抽取统一用户 embedding"。

#### 3.2.2 Q-Anchor 双塔预训练架构（Fig. 2）

- **Anchor Tower**：吃 hierarchical token $e_i$，把自然语言 query $q_i$ 作为尾部**语义锚点**追加进来。把 query 放在序列末端有两个含义：
  1. LLM 主干变成 query 感知的聚合器，从 $e_i$ 的潜空间选择性抽取 intent 相关特征；
  2. 昂贵的 hierarchical prefix $e_i$ 只算一次，KV-cache 保存后可以对不同 query 反复使用，产出场景自适应 embedding，边际延迟接近 0。

  产出 anchor 表征：$u_{i,q} = \text{LLM}_{anc}(e_i, q_i)$。

- **Semantic Tower**（不对称）：把答案 $a_i$ 编码成稠密向量 $v_{a_i} = \text{LLM}_{sem}(a_i)$，作为用户行为合成 / 意图建模的高保真语言目标。

- **两塔参数共享**：确保行为特征与语义标签映射到同一潜空间。

**联合对比 + 生成优化**（Joint Contrastive–Generative Optimization）：

**(a) Query 条件化 InfoNCE 对比损失**：
$$\mathcal{L}_{cl} = -\frac{1}{B}\sum_{i=1}^{B} \log \frac{\exp\big(\text{sim}(u_{i,q}, v_{a_i})/\tau\big)}{Z_i} \tag{Eq. 3}$$
$$Z_i = \exp\big(\text{sim}(u_{i,q}, v_{a_i})/\tau\big) + \sum_{j\neq i} m_{ij}\Big[\dots\Big] \tag{Eq. 4}$$

分母 $Z_i$ 由**三种负样本**组成：user–answer 跨对 (`u_{i,q}` vs `v_{a_j}`)、user–user (`u_{i,q}` vs `u_{j,q}`)、answer–answer (`v_{a_i}` vs `v_{a_j}`)。$\text{sim}(\cdot)$ 是余弦相似度，$\tau$ 是温度，$B$ 是 batch size。

**(b) Margin-based mask 去伪负样本**（Eq. 5）：只有当候选负 $j$ 的 user embedding 或 answer embedding 与 $u_{i,q}$ 的相似度过高（超过正样本相似度 + $c_{\text{margin}}$）时才被剔除，避免将真实相关样本错误当作负样本。

**(c) Next-Token Prediction 辅助目标**（Eq. 6）：
$$\mathcal{L}_{ntp} = -\sum_{t=1}^{T} \log P(y_t \mid y_{<t}, e_i, q_i)$$

要求 anchor tower 能自回归重建目标答案 $a_i$，让 embedding 同时具备细粒度语言语义。

**(d) 总损失**：$\mathcal{L}_{total} = \mathcal{L}_{cl} + \mathcal{L}_{ntp}$。前者保证全局判别性，后者防止表征坍塌 (representation collapse) 并增加语义密度。

#### 3.2.3 Cluster-based Soft Prompt Tuning

为进一步弥合「通用用户理解」与「专属业务逻辑」的语义 gap，采用后训练适配：**冻结 LLM 主干与 coarse-to-fine 用户编码器**，只训练一组可学习的 prompt token 以及类原型 $\{p_k\}_{k=1}^{K}$，损失采用**原型对比损失**：
$$\mathcal{L}_{pt} = -\frac{1}{B}\sum_{i=1}^{B} \log \frac{\exp\big(u_{i,q}^\top p_{y_i}/\tau\big)}{\sum_{k=1}^{K} \exp\big(u_{i,q}^\top p_k / \tau\big)} \tag{Eq. 7}$$

其中 $y_i$ 是样本类别标签。这样把用户 embedding 拉向其类别中心、推离无关簇，从而在潜空间形成有判别力的聚类结构，避免全参数微调带来的灾难性遗忘，同时保持 Query-as-Anchor 范式的部署效率。

#### 3.2.4 KV-Cache 加速的多场景推理

如 Fig. 4 所示：给定用户 $i$，hierarchical prefix $e_i$ 只编码一次得到共享 KV-cache 并作为持久性语义前缀保存；对下游 query 集合 $\{q_1, \dots, q_n\}$ 顺序处理并**复用同一份缓存**——每次只算 query 后缀增量隐藏态，摊销复杂度为 $O(L_{q_j})$，全部查询为 $O(\sum L_{q_j})$。因此同一份用户表征可以高效重新锚定到多个业务场景，几乎不引入额外延迟，非常适合支付宝级的高吞吐服务。

---

## 4. Experiments 实验设置

- **主干**：Qwen2.5-0.5B-Instruct（decoder-only）；异构行为由 modality-specific `gte-base` 编码器映射为稠密 embedding，再聚合后送入主干。
- **训练**：50k step，全局 batch size 2048，AdamW，初始 lr $2\times10^{-4}$，cosine 衰减；LoRA rank 64、$\alpha=32$；用户表征固定为 128 维。
- **硬件**：预训练 64 张 A100-80G 数据并行；推理 1 张 A100-80G；soft prompt tuning 用 6 个 learnable token，1 张 A100-80G。
- **Baseline 分两类**：
  1. **通用文本 embedding**：Qwen2.5-0.5B-Instruct（无用户微调）、Qwen3-Embedding-0.6B、Llama-Embed-Nemotron-8B、KaLM-Embedding-Gemma3-12B；
  2. **用户表征模型**：MSDP、One4all、CPC、FOUND。
- **评测**：支付宝生产系统的 10 个二分类任务，分成三大域：
  - **User Engagement**：Active、Concert、Login、Forest；
  - **Risk**：Fraud、Money（洗钱）；
  - **Marketing Sensitivity**：Takeout、Brand、Promo、Value。
  每个任务约 50 万测试样本。
- **指标**：AUC、KS（Kolmogorov-Smirnov）；表征采用线性探测（linear probing）评估。

---

## 5. Main Results 主要结果

### 5.1 深入性能分析（Table 2 + Fig. 5）

**核心结论**：Q-Anchor（Prompt Tuned）在 10 个 benchmark 上全部取得最好成绩，平均 **AUC = 0.8225 / KS = 0.5267**。

- 相对最强通用 embedding baseline **Llama-Embed-Nemotron-8B (AUC 0.7488 / KS 0.3805)**：
  - AUC 提升 +0.0737（+9.84%）；
  - KS 提升 +0.1462（+38.4%）。
- 在 User Engagement / Risk / Marketing 三个域上一致占优。

作者由此得出：**工业用户建模的关键瓶颈不是语义容量，而是表征对齐**——通用文本 embedding 难以协调稀疏、符号化、多源的行为日志；Q-Anchor 通过 UserU 上的分层行为编码，把异构事件映射到 query 相关信号，不需要巨型参数化。

**跨域鲁棒性**：同一个编码器在无任务专属架构的情况下适应三大异构域：
- Risk / Money：Q-Anchor（Prompt Tuned）AUC=0.9439，优于 FOUND (0.9235)、MSDP (0.8746)，说明 anchor query 能压制高熵交易噪声并放大决策相关模式；
- Marketing / Brand：AUC 从 Base 的 0.7979 提升到 Prompt Tuned 的 0.8535，表明模型不仅跨域迁移，还能适配细粒度决策边界。

**数据 vs. 模型的 scaling 分析**（Fig. 6）：
- **数据 scaling**（20.48M → 102.4M 预训练对）：Avg AUC 0.8029 → 0.8105，Avg KS 0.4895 → 0.5044，一致提升；
- **模型 scaling**（0.5B → 3B）：**非单调**——0.5B 最优 (0.8105/0.5044)，1.5B 与 3B 无增益、部分场景（Brand、Promo）反而下降。
- Appendix Fig. 11 显示 3B 模型平均梯度比 0.5B 显著变小（0.028 vs 0.082），说明大模型在 sentence-level embedding 目标上优化景观变平，与已有 embedding scaling 观察一致。
- 结论：**embedding 质量更依赖预训练数据规模而非模型规模**；论文默认配置为 0.5B + 50k step。

**Prompt Tuning 的 scaling**（Fig. 8）：
- Token 数从 1 → 16：性能在 6 个 token 附近达到饱和 (0.8225/0.5267)；再增加意义不大；
- Training step 100 → 500：平滑上升，500 step 最好。默认配置 **6 个 prompt token + 500 step**。

**Prompt tuning 作为场景乘子且可解释**（Fig. 7）：
- Takeout Interest 场景中，模型对 Bill 模态的注意力上升 +26.0%（对应"餐饮购买力"信号）；
- Ant Forest 场景中 SPM 模态注意力上升 +6.4%（对应导航密集使用）；
- Fig. 9 的 t-SNE 显示 prompt-tuned 表征形成更清晰的场景一致聚类；PCA 可视化在 Appendix C.4 中给出同样结论。
- 这表明 Query-as-Anchor 起到"语义透镜"作用，用最少参数更新引导表征聚焦于场景相关的多模态证据。

### 5.2 消融实验（Table 3）

维持 backbone、数据与训练预算不变，逐项去除组件：

1. **模态 token**：去除 user token 或 modal token 或两者，Avg AUC 分别从 0.8104 降到 0.8086 / 0.8088 / 0.8065；对 modality-sensitive 场景（如 Brand）下降最明显（0.7979 → 0.7819）。说明显式 user/modal 标注注入了轻量结构先验，提升了跨模态聚合。
2. **训练目标**：
   - 去掉对比学习 (`w/o Contrastive`)：Avg AUC 从 0.8104 骤降至 0.7667，Brand 从 0.7979 → 0.6512，Money 0.9382 → 0.9091——**对比对齐是主导信号**；
   - 去掉 margin-mask filter (`w/o Filter`)：AUC 0.8047，一致下降；
   - 去掉 NTP (`w/o NTP`)：0.8061，同样下降但更温和。
   - 结论：**对比学习定义几何结构，filter 与 NTP 稳定训练并增强鲁棒性**。
3. **预训练**（`w/o pretrain`，只做 prompt tuning）：Avg AUC 0.8225 → 0.7781；KS 0.5267 → 0.4679（−11.2%）。Money 场景 AUC 0.9439 → 0.8153。预训练不是简单的优化辅助，而是提取鲁棒行为先验的**基础要求**。

### 5.3 线上 A/B 验证

**场景 I：IVR 备用金外呼**（Interactive Voice Response Cash-Reserve Outreach）
- Q-Anchor 用于优化外呼时机决策；
- **提款率（drawdown rate）+12.5%**；
- 人均在贷余额 +5.3%；
- 备用金产品访问率 +4.2%；
- 提款页访问 +17.7%。

**场景 II：授信逾期风险识别**
- Q-Anchor embedding 接入风控评分链路，业务关键 **KS +1.96%**。

**支付宝生产部署**
- 日级 embedding 刷新，服务数亿用户，100 张 L20 GPU 集群；
- 多场景服务通过共享前缀 KV-cache 分摊主干开销，新增一个场景只需追加 query suffix 的计算（对应约 1 张额外 L20 即可维持相同 SLA），避免为每个场景复刻整套 100-GPU 管线。

---

## 6. Conclusion 结论

Q-Anchor 是一个统一的工业级用户表征框架，弥合了「稀疏、噪声、异构行为日志」与「LLM 级语义表征」之间的 gap。三条主贡献：

1. **UserU**：耦合规则化未来行为监督与经反思校验的 LLM 合成用户 QA，为 embedding 学习注入时序动态与语义理解；
2. **Query-as-Anchor**：层次化 coarse-to-fine 编码器 + query 条件化对齐范式，把多源信号蒸馏成任务自适应 embedding；
3. **Semantic re-anchoring**：通过轻量 soft prompt 把模态重新锚定到下游决策边界。

在 10 个真实 benchmark 与 2 个线上 A/B 上取得 SOTA，同时保持可解释、可复用、低成本的部署特性。

**未来工作**：论文发现"Scaling Paradox"——1.5B–3B 主干出现梯度衰减与性能停滞，与 [Jiang et al. 2024] 的观察一致。作者将探索梯度恢复、自适应参数调优等手段，把 scaling 收益延伸到更大模型。当前部署继续以 0.5B 为主力。

---

## 附：术语中英对照速查

| 英文 | 中文 |
|---|---|
| User Representation | 用户表征 |
| Query-as-Anchor | Query 作为锚点 |
| Hierarchical Coarse-to-Fine Encoder | 分层由粗到细编码器 |
| Anchor Tower / Semantic Tower | 锚点塔 / 语义塔 |
| InfoNCE Contrastive Loss | InfoNCE 对比损失 |
| Margin-Based Mask | 边距掩码 |
| Next-Token Prediction (NTP) | 下一 token 预测 |
| Soft Prompt Tuning | 软提示微调 |
| Prototypical Contrastive Loss | 原型对比损失 |
| KV-Cache Sharing | KV 缓存共享 |
| Linear Probing | 线性探测 |
| AUC / KS | ROC 曲线下面积 / Kolmogorov-Smirnov 分离度 |
| Scaling Paradox | 规模悖论 |
