# Qwen3-VL 事件序列编码器调研发现

## 本地证据

- `raw/transcripts/多模态数据统一嵌入大模型调研/事件序列与统计特征接入 Qwen 微调可行性与工程方案.md` 记录当前单样本约 8 步事件序列。
- 事件项至少包含 code、顺序、时间戳和真实时间间隔；只有 `available_at <= prediction_cutoff` 的事件可进入样本。
- 现有 Wiki 要求先比较 Dummy、LR/LightGBM、GRU、确定性文本模板，再升级 soft token 或深融合。
- 当前 Qwen3-VL 统计特征方案已固定 `[B,M,2560]` 接口、描述离线预编码、冻结主干和 zero gate，可直接作为序列编码器的连接器契约。
- 短八步序列削弱了长序列高效架构的必要性；任何 Mamba/SSM 或深 Transformer 方案必须证明相对 GRU/浅层 Transformer 的增益。

## 外部证据

### 第一轮：基础序列模型与连续时间

- GRU4Rec（arXiv:1511.06939）是将 GRU 用于 session-based recommendation 的早期代表，支持 GRU 作为成熟低成本序列基线；其原任务和损失不直接证明风控事件分类效果。
- SASRec 将交互历史作为 token 序列并使用 causal self-attention；后续研究仍把它视为顺序推荐的强而简单基线。
- TiSASRec（WSDM 2020，DOI:10.1145/3336191.3371786）同时编码绝对位置和事件间时间间隔，直接支持“顺序位置不能替代真实时间间隔”的设计。先前候选 `arXiv:2002.00741` 实为 Déjà vu 上下文化时间注意力，不是 TiSASRec。
- Transformer Hawkes Process（ICML 2020）用 self-attention 建模事件依赖，并联合事件类型与发生时间；只有需要预测下一事件时间或生成强度时，TPP 目标才有直接价值。
- Mamba4Rec（2024）将 selective state-space model 引入顺序推荐，主要动机是长序列效率与噪声选择；对当前约 8 步序列没有天然优势。

### 第一轮裁决

- `[CONFIRMED]` 事件顺序和不规则时间间隔应分开编码；GRU、时间感知 attention 和 TPP 提供三类独立技术路径。
- `[MAJORITY]` Transformer/SASRec 类架构是成熟的字段交互与长程依赖方案，但不保证短序列优于 GRU。
- `[INSUFFICIENT DATA]` 没有直接 benchmark 比较这些编码器作为 Qwen3-VL soft-token 前端的效果。

### 第二轮：时间表示、长序列架构与反证

- Time2Vec（arXiv:1907.05321）提供模型无关的可学习时间向量，同时包含非周期与周期分量；它是时间特征候选，不证明优于 log-bucket 或相对时间 bias。
- T-LSTM（KDD 2017）按时间间隔衰减短期记忆、保留长期记忆，说明不规则间隔可直接进入 recurrent state；其临床数据归纳偏置不能无条件迁移。
- HSTU（arXiv:2402.17152）面向高基数、非平稳、流式推荐和超长历史，作者报告相对 Transformer 的质量与效率收益；其规模、目标和当前八步事件分类差异巨大，不适合作为低成本默认。
- Mamba/SSM 研究的主要优势集中在长程依赖与线性扩展；后续工作也报告长序列上的差距更明显。这构成当前短序列场景的反证：创新性高，但必要性低。
- 文本 embedding 可作为事件/item 语义初始化或适配器输入，但仍需 ID/residual embedding 吸收任务特定差异；不能把语义相似直接当作行为作用相同。

### 第二轮裁决

- `[CONFIRMED]` 当前必须至少同时编码顺序位置、相邻间隔 `Δt` 和距预测时点 `age`，三者语义不同。
- `[MAJORITY]` 对约 8 步序列，浅层 GRU 和浅层时间感知 Transformer 比 HSTU/Mamba 更符合成本与归因目标。
- `[SINGLE-SOURCE]` HSTU 和 Mamba 系列的作者结果只能作为邻近证据，不能转化为 EviState 效果预期。

### 第三轮：固定描述、异构事件与迁移

- UniSRec（KDD 2022，arXiv:2206.05941）使用 item 描述文本学习可迁移序列表征，并以轻量 adaptor 对齐文本表示；支持“固定事件描述离线编码 + 小适配器”，但 item 推荐不等于风控事件分类。
- RecFormer（arXiv:2305.13731）把 item 表达为 key-value 属性并联合学习语言与顺序模式；它支持语义和顺序联合建模，也显示逐事件展开文本会显著增加 token 成本。
- LLM-ESR（arXiv:2405.20646）使用降维后的 LLM item embedding 初始化协同 embedding，并增加两层 adapter；支持保留任务特定 ID/residual embedding，而非只依赖文本语义。
- 2026 年 SOHET 和金融多模态事件基础模型预印本都采用“事件内异构属性融合 + 事件间时序编码”的层次结构；这一拆分与 EviState 的事件类型、动态属性、时间间隔契合，但当前主要是作者报告，需标为单一来源/预印本。
- TPP-LLM-Embedding（arXiv:2410.14043）联合事件类型、时间和文本描述做序列检索，说明文本语义可与连续时间共存；其检索目标不证明分类 soft token 最优。

### 第三轮裁决

- `[CONFIRMED]` 固定事件描述应作为事件类型语义先验，并保留可训练 ID/residual embedding；只有描述向量会混淆语义近似但业务作用不同的事件。
- `[MAJORITY]` 应采用两级编码：先融合单事件的 type/description/time/payload，再编码事件间顺序。
- `[INSUFFICIENT DATA]` 没有公开结果直接比较 GRU、时间感知 Transformer 和 SSM 作为 Qwen3-VL soft-token 连接器。

### 原始来源核验

- GRU4Rec：arXiv:1511.06939，短 session 场景的 RNN/GRU 代表。
- SASRec：arXiv:1808.09781，ICDM 2018，causal self-attention 顺序推荐。
- TiSASRec：WSDM 2020，DOI:10.1145/3336191.3371786，无需借用错误的 arXiv ID。
- Transformer Hawkes Process：arXiv:2002.09291，事件类型/时间联合建模。
- UniSRec：arXiv:2206.05941，KDD 2022，描述文本 + 轻量 adaptor + 跨域迁移。
- RecFormer：arXiv:2305.13731，KDD 2023，key-value 文本属性与序列联合建模。
- LLM-ESR：arXiv:2405.20646，NeurIPS 2024 Spotlight，离线 LLM 语义 embedding 增强传统序列模型且不增加 LLM 在线推理。
- Mamba4Rec：arXiv:2403.03900，长行为序列效率动机。
- HSTU：arXiv:2402.17152，ICML 2024，8192 长度和超大规模推荐；不对应短八步成本条件。
- TPP-Embedding：arXiv:2410.14043，事件文本与时间联合检索。
- SOHET：arXiv:2606.21356，2026 预印本，异构事件内编码 + 序列级 Transformer，在专有欺诈任务和 EBES 上给出作者结果。
- 金融多模态事件基础模型：arXiv:2607.09955，2026 预印本，统一多源事件流并以 next-event prediction 预训练。

## 候选

初步候选：

1. 低成本：事件描述语义 + 时间特征 + 1 层 GRU，直接读取最后一个有效 hidden state 作为连续向量；可选两层 MLP 投影为一个 Qwen soft token。
2. 主流稳健：时间感知 Transformer（SASRec/TiSASRec 类）+ 固定查询压缩。
3. 创新：语义条件事件 token + 连续时间/多尺度间隔 + Mamba/SSM 或 TPP 辅助目标 + 查询压缩。

### 最小工程方案

对事件 `t`：

`event_t = EventMLP([description_embedding[type_t], trainable_id_embedding[type_t], log1p(delta_t), log1p(age_t), optional_payload_t])`

然后：

1. `GRU(event_1 ... event_L) -> hidden_1 ... hidden_L`
2. `sequence_vector = hidden_last_valid`，形状 `[B,128]`
3. 可选 `token = zero_gate * Projector(sequence_vector)`，形状 `[B,1,2560]`

该方案不使用 attention pooling、learned query、Transformer、Mamba 或 TPP。文本模板是不开发自定义编码器时的绝对最低成本基线；单层 GRU 是需要连续序列表征时的最低成本合理方案。

## 待核验

- EviState 事件项的动态字段、序列长度分布和截断策略。
- 目标任务是否更依赖最近事件、重复模式、绝对时间还是长程间隔。
- 目标训练与推理栈对变长 soft token 和 mask 的支持。
