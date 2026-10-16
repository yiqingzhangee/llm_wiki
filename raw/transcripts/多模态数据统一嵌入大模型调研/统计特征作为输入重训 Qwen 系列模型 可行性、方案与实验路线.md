# 统计特征作为输入重训 Qwen 系列模型：可行性、方案与实验路线

> 🎯 **核心判断**
>
> 1. **文本化只能作为基线**：列名与统计值序列化成本最低，适合先测语义增益；连续精度、长序列与异常值场景需要独立数值表示。
> 2. **融合越深，改造成本越高**：建议按“强非大模型基线→文本序列化→连续编码→软嵌入→中间层交互”逐级升级，任何一级未证明增益都不进入下一级。
> 3. **Qwen 的软嵌入路线具备接口可行性**：Qwen2、Qwen3 的公开实现支持 `inputs_embeds`，Qwen 多模态模型也采用“外部编码器＋投影器＋语言主干”范式；但接口可接入不等于目标业务有效。[[2]](https://github.com/huggingface/transformers/blob/main/src/transformers/models/qwen2/modeling_qwen2.py) [[3]](https://github.com/huggingface/transformers/blob/main/src/transformers/models/qwen3/modeling_qwen3.py) [[11]](https://arxiv.org/abs/2502.13923)
> 4. **现有证据支持“可以试”，尚不足以支持“应当重训”**：Qwen 直接证据集中于文本化时序、小范围推荐和时序软嵌入；跨领域研究同时给出正向结果与“移除大模型后更好”的反证。
> 5. **分阶段训练优于一步全量解冻**：先冻结 Qwen 对齐编码器与投影器，再做参数高效任务微调；只有在稳定增益已出现且学习曲线仍上升时，才进入局部或全量联合训练。
> 6. **PoC 的首要任务是证伪必要性**：候选方案必须在时间外、实体外及二者交集上，以同等调参预算稳定超过最佳 GBDT、深度表格模型、晚融合和文本化 Qwen；否则停止扩模。
> 7. **统计忠实性必须由确定性计算兜底**：聚合、排序、比率、置信区间与因果估计应由受测代码或统计模块生成，Qwen 负责语义理解、字段映射与解释。
> 8. **推荐从 Qwen2.5-7B-Instruct 或 Qwen3-8B 做双基座小规模试验**：先锁定 Transformers 版本、序列协议和许可证，再以两周量级 PoC 决定继续、降级为晚融合，或回到专用模型。

## 一、问题定义：文本化只能作为基线

> **本节要点**：所谓“统计特征作为输入”至少包含文本表示、连续编码、软 token 和中间层融合四种不同问题。若不先固定任务、切分与业务门槛，后续比较会把接口可行性误当成模型收益。

### 1.1 四种输入含义

统计特征可包括连续值、类别统计、滑动窗口聚合、分布摘要、缺失模式、时间位置和结构化元数据。对 Qwen 而言，输入方式决定模型实际接收到的是语言符号、连续向量，还是另一编码器的表示。

- **文本序列化**：把特征写成“列名：值；单位；窗口”等确定性模板，再由原 tokenizer 编码。优势是无模型改造，缺点是数字被拆成多个子词、精度与长度不可控。
- **连续数值 token**：对每个数值做分段线性、周期或 Fourier 编码，再投影为一个或少量向量；保留数值邻近关系，但需自行处理尺度、单位与缺失。
- **软嵌入前缀或交错 token**：编码器把整行、窗口或 patch 压成若干向量，与文本 token 一起送入 Qwen；这是本报告建议优先验证的结构化增强。
- **中间层交互**：通过 cross-attention、门控或分层注入让结构化表示与语言隐状态多次交互。表达力更强，代码侵入、版本耦合与训练成本也最高。

研究已表明，简单的数值表示改进本身就可能让 MLP 接近或超过复杂注意力模型；因此“统计特征经编码后有效”不能自动归因于 Qwen。[[16]](https://arxiv.org/abs/2203.05556v4)

### 1.2 成功标准与适用任务

| 任务形态 | Qwen 可能提供的增量 | 首选对照 | 主要判据 |
|---|---|---|---|
| 固定 schema 分类/回归 | 通常有限，除非列名和文本描述携带强语义 | CatBoost、LightGBM/XGBoost、MLP、FT-Transformer | 时间外/实体外主指标、成本与校准 |
| 表格＋长文本 | 语义字段映射、跨字段解释、弱监督迁移 | 单模态强模型＋晚融合 | 联合方案相对单源与晚融合的净增益 |
| 时序＋事件文本 | 把事件语义与数值窗口联合建模 | PatchTST、DLinear、PAttn 或目标域时序模型 | 不同预测跨度、漂移窗口与事件缺失切片 |
| 统计问答/报告生成 | 意图理解、字段映射、结果解释 | 代码执行代理＋确定性统计引擎 | 数值正确率、执行成功率与可追溯性 |

PoC 的成功必须同时满足效果、忠实性、泛化与效率四类门槛。单一随机切分上的平均分提升，只能证明候选配置在该切分有效，不能证明 Qwen 具备可上线的结构化归纳能力。

---

## 二、方案分层：融合越深，改造成本越高

> **本节要点**：逐级升级的目的在于识别真正产生增益的组件。推荐把 L0—L3 作为 PoC 主线，把中间层注入和超长上下文训练留到证据充分之后。

### 2.1 从文本序列化到特征投影

#### 统计特征注入 Qwen：先立基线，再验证软嵌入

纵轴代表模型侵入度与联合训练成本；越往上，越需要证明跨模态交互的额外价值。

- **低改造基线**：确定性“列名：值”序列化＋原生 ChatML＋LoRA/SFT，用于最快建立收益下限与错误样本集。
- **推荐 PoC 路线**：连续特征编码器＋projector → `inputs_embeds`；先冻结 Qwen 对齐，再按证据渐进解冻。

**五层方案全景（由深到浅）**

1. **L4 深融合：Cross-attention / 分层注入**
   - 定位：只在互补信息与交互增益已被消融证明后升级。
   - 输入机制：Q-Former、跨注意力或多层 hidden-state 注入。
   - 优势假设：处理长序列、多尺度与复杂模态交互。
   - 主要风险：版本耦合、训练不稳、显存与时延显著上升。
   - 进入条件：简单投影已稳定胜出，且直接拼接仍限制质量。
2. **L3 软嵌入：编码器＋projector → Qwen 隐空间（推荐 PoC）**
   - 定位：接口改造可控，能保留连续值精度并压缩序列。
   - 张量协议：投影末维对齐基座 hidden size；Qwen2.5-7B 为 3584。
   - 序列拼装：软 token 与文本交错；同步构造 mask、position 与 `labels=-100`。
   - 训练起点：先冻结 Qwen，只训练编码器/投影器完成表示对齐。
   - 必做消融：单源、直接拼接、MLP 投影、无 LLM 与晚融合。
3. **L2 连续编码：PLE / Fourier / patch 表示**
   - 定位：先解决数值表示、尺度与 token 膨胀，再连接语言模型。
   - 连续值：PLE 或周期编码保留邻近关系与动态范围。
   - 时序样本：窗口切片与 patch 压缩，避免逐值文本化。
   - 缺失与单位：显式编码缺失、单位、采样时间与字段身份。
   - 独立价值：先与 MLP、ResNet、FT-Transformer 公平比较。
4. **L1 文本化：确定性“列名：值”序列化（低改造基线）**
   - 定位：不改模型接口，最快验证语义表头与少样本价值。
   - 输入格式：固定模板保留字段名、单位、值与缺失标记。
   - 训练方式：原生 tokenizer＋ChatML，优先 LoRA/SFT。
   - 适用场景：窄表、低样本、列名有语义、上下文预算充足。
   - 失效信号：BPE 膨胀、精度损失、宽表截断或数值忠实性不足。
5. **L0 强基线：专用表格/时序模型与晚融合**
   - 定位：不是陪跑项，而是决定是否值得引入 Qwen 的门槛。
   - 表格任务：CatBoost、XGBoost、TabPFN、调优 MLP/ResNet。
   - 时序任务：PatchTST、DLinear 与等参数浅层注意力。
   - 融合基线：单模态模型独立训练，决策层聚合分数。
   - 公平口径：同切分、同预算、同早停，同时报告效果与效率。

**结论**：改造越深，交互容量越强，但证据、训练与维护成本同步上升。起点应先跑 L0 与低改造基线，主实验再推进推荐 PoC。

*图 1：统计特征注入越深入，潜在表达力与工程风险同步上升（来源：Qwen 接口、数值编码与软嵌入研究综合）*

最关键的决策不是选择哪种 projector，而是确认任务是否需要语言主干参与。固定 schema 且标签充足时，树模型和经过调优的表格深度模型仍是上线候选；只有列名、单位、描述或事件文本带来专用模型难以利用的互补信息，Qwen 路线才有明确假设。典型中等规模表格数据上，树模型优势仍相当稳健。[[25]](https://arxiv.org/abs/2207.08815)

### 2.2 五类输入融合路线

| 层级 | 实现 | 适用条件 | 主要风险 | 进入下一层条件 |
|---|---|---|---|---|
| L0 | 强非大模型基线 | 所有任务必做 | 调参预算不公平会制造虚假领先 | 确认语义或跨模态信息尚未被利用 |
| L1 | 确定性“列名：值”文本化 | 低样本、语义表头、改造预算低 | 数字拆词、序列膨胀、生成式改写失真 | 稳定超过无语义列名和纯值模板 |
| L2 | PLE/周期/Fourier 连续编码 | 连续精度、插值、动态范围重要 | 尺度和单位不一致、OOD 外推失真 | 超过原始值、分箱和文本数字表示 |
| L3 | 编码器＋MLP/Q-Former＋软 token | 多字段、窗口或 patch 需要压缩 | 软 token 数量与信息瓶颈不匹配 | 超过同编码器＋非 Qwen 头及直接拼接 |
| L4 | cross-attention/分层注入 | 存在已验证的深层跨模态交互 | 侵入主干、版本耦合、训练不稳定 | 浅融合已达平台且交互消融支持继续 |

TabLLM 显示“列名—值”模板在零样本阶段较稳，但有 8 个训练样本后格式差异已明显收敛；复杂生成式改写还可能漏字段或添加不存在的信息。[[14]](https://arxiv.org/abs/2210.10723v2) 这意味着 L1 应做成确定性数据管道，而非另用大模型生成输入文本。

若数值精度和 token 预算成为瓶颈，xVal 等研究说明单 token 连续表示具有潜在效率优势；但 xVal 是从头训练并配专用数值头的科学模型，其结果只能作为 L2 的跨领域依据，不能直接外推为 Qwen 适配收益。[[17]](https://arxiv.org/abs/2310.02989v2)

---

## 三、Qwen 适配：软嵌入路线最稳妥

> **本节要点**：Qwen 的公开接口足以完成输入层软嵌入 PoC，且官方视觉模型提供了架构类比。首轮应尽量不改 Transformer 主干，把不确定性集中在特征编码器、投影器和序列拼装。

### 3.1 模型、接口与训练栈

**接口级直接证据明确。** Hugging Face 的 Qwen2 与 Qwen3 CausalLM 前向接口都接受 `inputs_embeds`，并要求它与 `input_ids` 二选一；这允许在嵌入层之后拼入外部统计表示。[[2]](https://github.com/huggingface/transformers/blob/main/src/transformers/models/qwen2/modeling_qwen2.py) [[3]](https://github.com/huggingface/transformers/blob/main/src/transformers/models/qwen3/modeling_qwen3.py) 该证据只证明张量可进入主干，生成缓存、流式解码、beam search 和混合精度仍需固定版本实测。

首轮建议并行使用两个 7B/8B 级基座：Qwen2.5-7B-Instruct 的隐藏维度为 3584，Qwen3-8B 为 4096，投影层输出必须分别对齐。[[4]](https://huggingface.co/Qwen/Qwen2.5-7B-Instruct/raw/main/config.json) [[5]](https://huggingface.co/Qwen/Qwen3-8B/raw/main/config.json) 两者当前模型卡均标示 Apache-2.0；实际使用时仍应锁定仓库 commit 并保存许可证快照。[[7]](https://huggingface.co/Qwen/Qwen2.5-7B-Instruct) [[8]](https://huggingface.co/Qwen/Qwen3-8B)

Qwen 使用 ChatML 轮次边界。外部向量不经过 tokenizer，因此需要自行构造 attention mask、position ids 和标签掩码；统计特征位置的 labels 设为 -100，只对目标回答 token 计算自回归损失。轮次边界 token 应保留，Qwen3 的 thinking 模式则需在数据模板和评测中显式固定。[[6]](https://qwen.readthedocs.io/en/latest/getting_started/concepts.html)

### 3.2 编码器、投影层与序列协议

建议的最小结构为：数值标准化与缺失编码 → 字段级编码器 → 固定数量软 token → 两层 MLP 投影到 Qwen hidden size → 与系统提示、字段说明和任务指令拼接。若输入是时间窗口，可先 patch 化再压缩；若是单行统计特征，可按字段或字段组生成 token。

| 组件 | 首选实现 | 必须消融 | 单元测试 |
|---|---|---|---|
| 连续值编码 | 标准化＋PLE 或 Fourier；显式单位 embedding | 原始值、分箱、文本数字 | 平移/缩放、极值、NaN、无穷值 |
| 类别/统计元数据 | 类别 embedding＋字段 ID＋窗口 ID | 去字段 ID、打乱列名 | 未知类别、schema 增删、字段重排 |
| 压缩器 | MLP 起步；信息密集时试 Q-Former | 均值池化、直接拼接、等参数注意力 | 变长输入、全缺失、mask 一致性 |
| 注入位置 | 前缀软 token；有局部对齐需求再试交错 | 前缀、交错、只用文本 | 位置编号、KV cache、批内变长 |
| 输出 | 分类/回归用专用头；解释用 CausalLM | 只用生成答案、只用专用头 | 数值范围、单位、格式与拒识 |

Qwen2.5-VL 的视觉编码器经两层 MLP merger 投影到语言模型 hidden dimension；Qwen3-VL 延续相同基本范式，并在更深层加入视觉特征。[[11]](https://arxiv.org/abs/2502.13923) [[12]](https://arxiv.org/abs/2511.21631v2) 这为“外部特征编码后作为 token 注入 Qwen”提供官方架构旁证，但视觉模型还包含专用位置编码和大规模多阶段训练，不能据此认定只训练 projector 即可达到目标质量。

---

## 四、证据判断：方向可行，直接证据仍薄

> **本节要点**：Qwen 直接证据能够覆盖接口、官方投影范式和少数任务结果，却没有覆盖通用工业表格的公平端到端比较。跨领域证据可帮助设计方案，不能替代目标数据上的强基线与消融。

### 4.1 Qwen 直接证据

| 证据 | 支持的结论 | 边界 | PoC 用法 |
|---|---|---|---|
| Qwen2/Qwen3 源码接口 | 外部 embedding 可进入解码主干 | 不证明任务效果和生成兼容性 | 建立接口、mask 与梯度门禁 |
| Qwen2.5-VL/Qwen3-VL | 编码器＋merger＋LLM 是官方采用的形态 | 视觉训练规模和位置机制不可直接迁移 | 借鉴维度对齐与冻结后解冻流程 |
| Qwen2.5 小模型时序文本微调 | 离散化数字后可学习基础趋势解释 | 180 条合成训练、20 条测试，外推性弱 | 作为 L1 低成本 sanity check |
| TsLLM＋Qwen2.5-7B | patch encoder、cross-attention adapter 与占位 token 替换可训练 | 约 30B token 训练；成本远高于普通 PoC | 借鉴三阶段训练与 adapter 消融 |
| Qwen3 推荐数值融合 | 连续编码、查询压缩和显式交互可能优于直接拼接 | 仅三类 Amazon 数据、价格单一连续特征 | 作为 L3 的机制候选，不作通用结论 |

《Towards Interpretable Time Series Foundation Models》对 Qwen2.5-0.5B/1.5B 的结果说明小模型可通过后训练学习合成时序的趋势、噪声和极值描述；作者同时依赖缩放、取整和补零来规避 tokenizer 的数值表示问题。[[13]](https://arxiv.org/abs/2507.07439) 该研究支持文本化基线可行，也揭示了精度折中的来源。

更直接的 TsLLM 以 Qwen2.5-7B 为基座，用 patch encoder 与 cross-attention adapter 产生时序表示，再替换占位 token 的 embedding；其三阶段训练先对齐 adapter，再联合训练，最后执行只对答案计算损失的 SFT。[[10]](https://arxiv.org/abs/2510.01111v2) 这是软嵌入路线的强直接证据，但训练量级意味着本项目应验证机制，不应复刻规模。

### 4.2 跨领域间接证据与反证

**表格证据偏向低样本语义利用。** TabLLM 在 9 个公开数据集的零样本和极少样本区间表现突出，但数据集列数、输入长度与人工可读字段限制了外推。[[14]](https://arxiv.org/abs/2210.10723v2) TabuLa-8B 在大规模表格语料上专门训练后获得显著提升，原始 Llama-3-8B 却低于随机猜测；这表明收益可能依赖大量结构化预训练，而非通用语言能力自然迁移。[[15]](https://arxiv.org/abs/2406.12031v2)

**数值表示证据支持独立编码器。** 分段线性与周期编码在多种非 Qwen 架构上改善数值特征学习，xVal 则展示了连续 token 的效率与误差优势。[[16]](https://arxiv.org/abs/2203.05556v4) [[17]](https://arxiv.org/abs/2310.02989v2) 它们说明“不要把浮点数完全交给普通 BPE”具有合理性；能否与现成 Qwen 无损结合仍是工程假设。

**时间序列证据存在关键冲突。** Time-LLM 报告冻结语言模型并做重编程可以获得预测提升。[[19]](https://arxiv.org/abs/2310.01728v2) NeurIPS 2024 的复现与消融覆盖 13 个数据集，移除大模型或换成浅层注意力后，在多数汇总项上反而优于原方法，并显示训练成本相差多个数量级。[[20]](https://arxiv.org/abs/2406.16964v2) 因而 PoC 必须包含“无 Qwen 同构网络”和“随机初始化同构主干”，否则无法归因语言预训练。

**统计推理证据要求工具兜底。** QRData 中 GPT-4 加代码解释器的最佳准确率仍为 57.9%，纯自然语言思维链在数值题上最高不足 10%。[[21]](https://arxiv.org/abs/2402.17644v1) NumericBench 与 TableBench 也显示数字检索、复杂表格和噪声条件下存在明显缺口。[[22]](https://arxiv.org/abs/2502.11075v2) [[23]](https://arxiv.org/abs/2408.09174v1) 生产系统应让模型解释确定性结果，避免让生成文本成为唯一数值来源。

---

## 五、训练设计：分阶段优于一步全训

> **本节要点**：先把表示空间对齐，再验证任务增益，最后才考虑联合解冻。每阶段都设置退出门槛，可避免用更大训练量掩盖结构无效。

### 5.1 对齐、任务微调与联合训练

#### 分阶段训练与评测：基线给下限，PoC 证明增量

两条路线共用数据切分、训练预算和冻结测试集；只有推荐 PoC 稳定越过门槛，才增加解冻深度。

1. **共同起点：特征与泄漏审计**
   - 锁定 schema、单位、缺失语义、标签成熟期与实体边界。
   - 先切分再拟合插补/标准化。
   - 冻结时间外、实体外及二者交集测试集。
2. **分路并行训练 A：低改造基线——确定性序列化＋原生 Qwen 微调**
   - **B1 模板固化**：“列名：值｜单位｜缺失”确定性展开，保留 ChatML 边界。
   - **B2 LoRA / SFT**：先小模型、短序列；仅回答 token 计损失，固定随机种子。
   - **B3 基线评测**：对比 GBDT、深度表格/时序模型与晚融合，记录质量—成本前沿。
3. **分路并行训练 B：推荐 PoC——连续编码器＋projector＋渐进解冻**
   - **P1 编码与压缩**：连续值用 PLE/Fourier，时序用 patch；显式编码字段、时间与缺失。
   - **P2 阶段 1，只做对齐**：冻结 Qwen，仅训练编码器/投影器；验证维度、mask、position 与生成链路。
   - **P3 阶段 2，任务适配**：保持主干冻结或仅加 LoRA；只对目标回答计损失，加入单源/无 LLM 消融。
   - **P4 阶段 3，有限解冻**：仅当 P2/P3 稳定胜出，逐层开放高层；配旧数据回放或正则防遗忘。
4. **统一验收：上线门禁**
   - 增益、忠实性、鲁棒性、效率必须同时通过。
   - 若 PoC 未在预注册切片和多随机种子下稳定超过最佳低改造/专用模型基线，或任一硬门禁失败，则停止扩模并回到特征、标签或任务定义。
   - 验收维度：
     - **稳定增益**：主指标＋置信区间，多种子/多切片。
     - **数值忠实**：单位、缺失、极值、确定性复算。
     - **泛化鲁棒**：时间外、实体外、漂移/OOD/校准。
     - **工程可用**：p95/p99、显存、吞吐、成本、回滚。

**决策闭环**：门禁失败，保留低改造基线或专用模型；门禁通过，才进入更深融合与规模化训练。

*图 2：训练流程以门禁串联，对齐失败或增益不稳定即停止升级（来源：Qwen 直接证据与迁移学习方法综合）*

| 阶段 | 可训练参数 | 目标 | 退出条件 |
|---|---|---|---|
| S0 接口校验 | 无；或仅随机 projector | 验证拼接、mask、位置、生成、梯度与批处理 | 任一张量/缓存测试失败 |
| S1 表示对齐 | 编码器＋projector；冻结 Qwen | 重构统计摘要、字段识别、对比学习或辅助任务 | 验证损失不下降或简单 MLP projector 无法过拟合小批次 |
| S2 任务微调 | projector＋Qwen LoRA；必要时任务头 | 在固定预算下超过 L0/L1 与无 Qwen 对照 | 时间外/实体外增益不稳定 |
| S3 联合训练 | 渐进解冻高层，最后才考虑全参 | 突破 S2 平台并保持通用能力 | 遗忘、成本或数值忠实性超阈值 |

TsLLM 和 Qwen3-VL 都采用先冻结主干、后续再联合训练的顺序，支持把 projector-only 视为稳定冷启动。[[10]](https://arxiv.org/abs/2510.01111v2) [[12]](https://arxiv.org/abs/2511.21631v2) 渐进解冻与分层学习率可参考 ULMFiT，但该方法原本针对 LSTM 文本迁移，应用到 Qwen 时必须重新做冻结日程消融。[[31]](https://arxiv.org/abs/1801.06146)

### 5.2 损失、采样与参数高效微调

- 主任务为分类/回归时，优先使用专用头损失；生成解释使用自回归损失，统计软 token 和提示位置统一屏蔽标签。
- 辅助目标只选择可验证表示质量的任务，如字段遮蔽恢复、窗口统计重构、同实体跨窗口对比；每个辅助目标都需单独消融。
- 多任务先做单任务、等权联合和动态权重三组。GradNorm 可作为梯度冲突明显时的候选，不能替代逐任务报告。[[32]](https://arxiv.org/abs/1711.02257)
- 样本按任务、实体、时间和缺失模式分层采样；避免让高频实体或短窗口主导训练。长序列按长度分桶，记录软 token 与文本 token 的共同预算。
- 首轮 LoRA 可覆盖 attention 与 MLP 的线性层，同时保留仅 `q_proj`/`v_proj` 的低成本对照；官方 MS-SWIFT 支持 LoRA、QLoRA、全参和分布式训练，但示例超参只适合作为起点。[[9]](https://qwen.readthedocs.io/en/latest/training/ms_swift.html)
- 通用能力保持采用旧任务小样本回放、冻结层对照和必要时 EWC/蒸馏；EWC 证据来自持续学习研究，迁移到本任务需重新选择正则强度。[[33]](https://arxiv.org/abs/1612.00796)

---

## 六、评测体系：先证明增益，再验证迁移

> **本节要点**：可执行 PoC 由五道门禁组成。所有方案复用同一不可变切分、搜索预算、早停规则和种子；最终结论依据置信区间、关键切片与成本的联合结果。

### 6.1 基线、消融与数据切分

最小基线集合包括：线性/逻辑回归、CatBoost、LightGBM 或 XGBoost、MLP/ResNet-like、FT-Transformer、文本化 Qwen、编码器＋非 Qwen 任务头、单模态模型晚融合，以及候选软嵌入 Qwen。FT-Transformer 研究显示 ResNet-like 是常被忽略的强深度基线，且 GBDT 与深度模型之间不存在普遍优胜者。[[24]](https://arxiv.org/abs/2106.11959)

重复实体必须实体互斥切分；有时间因果关系时必须前训后测，并用 gap 覆盖标签成熟期与特征窗口。[[26]](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GroupKFold.html) [[27]](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.TimeSeriesSplit.html) 若业务同时要求未来时间和新实体泛化，应分别报告时间外、实体外及二者交集；独立测试集不得参与模型、阈值、特征或早停选择。[[34]](https://arxiv.org/abs/2006.16189)

| 消融轴 | 最小配置 | 回答的问题 | 保持不变 |
|---|---|---|---|
| 输入 | 无统计特征、文本化、连续编码、软 token | 统计特征及其表示是否有效 | 样本、主干、训练步数 |
| 主干 | 预训练 Qwen、随机初始化同构网络、无 Qwen 轻量网络 | 语言预训练是否必要 | 编码器、投影器、输入长度 |
| 融合 | 直接拼接、MLP、Q-Former、显式交互 | 收益来自容量还是交互 | 参数量或计算预算 |
| 训练 | 全冻结、LoRA、渐进解冻、全参 | 适配深度是否值得 | 优化器、数据与早停 |
| 鲁棒性 | 字段重排、缺失、噪声、单位变化、时间漂移 | 模型是否依赖伪相关 | 扰动强度与评测样本 |

### 6.2 PoC 门禁与停止条件

下表给出可直接执行的默认门禁。业务方应在看测试集结果前把“最小业务增益 Δ、延迟 SLO、单位成本上限、关键切片容忍度”写入实验登记；若暂时没有业务阈值，可用表中的建议值完成技术 PoC，但不得据此直接上线。

| 门禁 | 通过标准 | 失败动作 | 证据产物 |
|---|---|---|---|
| G0 数据 | 时间窗、实体和标签成熟期审计通过；预处理仅在训练折拟合；测试样本 ID 冻结 | 停止建模，修复数据与切分 | 数据卡、泄漏检查、切分清单 |
| G1 接口 | 固定版本下前向、反向、生成、KV cache、混合精度和变长批均通过；projector 梯度非零 | 不进入效果试验 | 单元测试与最小复现脚本 |
| G2 表示 | 编码器在训练集小批次可过拟合；验证集辅助任务显著优于随机/常数基线 | 回退编码、尺度或 mask 设计 | 表示探针与学习曲线 |
| G3 增益 | 3 个以上种子；主指标相对最佳强基线达到预注册 Δ，95% 置信区间下界大于 0；时间外、实体外和交集方向一致 | 停止扩模；优先保留强基线或晚融合 | 逐种子结果、置信区间、切片表 |
| G4 忠实性 | 确定性统计测试集计算结果 100% 可复算；单位/范围违规为 0；概率任务校准与拒识达到预注册阈值 | 改为工具计算＋Qwen 解释，或终止生成式输出 | 失败样本、执行轨迹、校准图 |
| G5 效率 | p95/p99 延迟、峰值显存、吞吐和单位成本满足 SLO；增益—成本位于 Pareto 前沿 | 量化/蒸馏一次；仍失败则降级 | 同硬件压测与成本账单 |

建议的技术 PoC 默认 Δ 为：分类 AUC/PR-AUC 或排序 NDCG 相对提升至少 2%，回归误差相对下降至少 3%；这些数值仅用于没有业务阈值时筛除微小波动。最终门槛应由误判成本和上线收益决定。概率输出还应报告 ECE、NLL 或 Brier，并在独立校准集上比较温度缩放等方法。[[29]](https://arxiv.org/abs/1706.04599)

**立即停止条件**：

- 无 Qwen 轻量网络或随机初始化同构网络在同预算下达到或超过候选方案，说明语言预训练没有被证明必要。
- 提升仅出现在随机切分、单一随机种子或单一群体，时间外/实体外交集没有同向收益。
- 学习曲线随数据量或训练预算已平台化，外推上限仍达不到预注册 Δ。[[35]](https://arxiv.org/abs/2201.12150)
- 依靠增加软 token 或上下文长度获得的增益不足以覆盖显存、延迟和单位请求成本。
- 关键统计结果无法确定性复算，或单位、缺失、异常值、schema 变化测试持续失败。
- 出现无法解释的切分落差、疑似泄漏、关键切片显著退化、校准/OOD 拒识不达标，或回滚链路未就绪。

---

## 七、风险边界：数值忠实性是硬约束

> **本节要点**：结构化输入的主要风险来自数据口径、切分、尺度和线上漂移，模型幻觉只是其中一部分。生产设计应允许结构化分支独立降级，并保留每个统计结果的计算轨迹。

### 7.1 泄漏、漂移与缺失值

> ⏱️ **时间与实体泄漏**
>
> 窗口跨过预测时点、同一实体进入训练和测试，都会制造不可复现的提升。标签成熟期必须纳入 gap。

> 📏 **尺度与缺失**
>
> scaler、imputer、目标编码和特征选择只能在训练折拟合；真缺失、延迟到达与解析失败应使用不同状态。

> 📉 **分布与 schema 漂移**
>
> 监控字段范围、缺失率、关键分布、预测分布和拒识率；有延迟标签后再用真实性能判断是否重训。

所有预处理都应封装进版本化 Pipeline，确保训练、验证与线上使用同一变换；先切分后拟合是防泄漏的基本要求。[[28]](https://scikit-learn.org/stable/common_pitfalls.html) OOD 检测可保留最大 softmax 概率作为简单基线，但 softmax 值不能直接解释为可靠置信度，需用 AUROC/AUPR 和明确正类定义评估拒识。[[30]](https://arxiv.org/abs/1610.02136)

### 7.2 何时不应重训 Qwen

- 输入几乎全是固定 schema 数值/类别列，标签充足，输出仅需分类或回归，且 GBDT 已达到业务目标。
- 跨模态信息可由各自强模型独立预测，配对样本少、模态经常缺失或部署需要独立降级；此时优先晚融合。
- 任务要求精确聚合、统计检验、金额核算或因果估计；此时计算引擎应作为主路径，Qwen 仅承担解释。
- 特征窗口过长且压缩后丢失关键信息，逐值文本化又超出序列预算；应优先专用时序编码器或检索式摘要。
- 业务没有可冻结的时间/实体切分、最小增益和成本阈值；在评价口径未确定前，扩大模型只会增加不可解释性。

选择不重训 Qwen 并不等同于放弃语言能力。可将 Qwen 放在上层，读取专用模型输出、统计工具结果与证据引用，生成自然语言解释；这一路径保留语义体验，同时把数值正确性留在可复算模块。

---

## 八、综合判断：先做小规模消融

> 📌 **综合判断**：统计特征接入并重训 Qwen 在工程上可行，推荐以输入层软嵌入作为首个结构化增强；是否值得进入生产，必须由强基线、无 Qwen 消融和统计忠实性门禁共同决定。
>
> 该结论来自三条证据链：Qwen2/Qwen3 已提供软嵌入接口，官方多模态模型验证了外部编码器投影范式，TsLLM 等工作给出 Qwen 上的直接任务证据；与此同时，表格与时序研究显示收益高度依赖专门训练、语义表头和评测协议，且移除大模型后更好的反证不能忽略。因此最合理的投入方式是逐级证伪，而非直接全参重训。
>
> **残余风险**：
>
> 1. 目标数据、标签延迟、实体定义、SLO 与误判成本尚未给出，当前门禁中的数值只能作为技术 PoC 默认值。
> 2. Qwen 直接证据尚未覆盖通用工业宽表、跨 schema 连续回归和长期漂移，最终结论必须由目标数据复验。

### 8.1 推荐 PoC 配置

| 项目 | 首选 | 对照 | 上限控制 |
|---|---|---|---|
| 基座 | Qwen2.5-7B-Instruct、Qwen3-8B | 同尺寸文本化 Qwen | 首轮不使用更大模型 |
| 编码器 | 字段 embedding＋PLE/Fourier＋2 层 Transformer/MLP | 原始值、分箱、文本数字 | 总新增参数先控制在基座的 1% 内 |
| 软 token | 8/16/32 三档；前缀注入 | 均值池化、直接拼接 | 总序列先控制在 4096 内 |
| 适配 | S1 冻结 Qwen；S2 LoRA | 无 Qwen 头、随机主干 | 未过 G3 不做全参 |
| 种子与切分 | 至少 3 种子；时间外、实体外、交集 | 随机切分只作诊断 | 测试集仅揭盲一次 |

### 8.2 两周实验路线

1. **第 1—2 天**：冻结任务定义、特征时点、实体、主指标、Δ、SLO 与测试样本 ID；完成 G0。
2. **第 3—4 天**：跑通 GBDT、MLP/ResNet-like、FT-Transformer、文本化 Qwen 与晚融合；建立效果—成本前沿。
3. **第 5 天**：固定 Transformers commit，完成 `inputs_embeds`、mask、labels、KV cache 和变长批测试；完成 G1。
4. **第 6—7 天**：训练连续编码器与 projector，执行小批次过拟合、辅助任务和字段扰动；完成 G2。
5. **第 8—10 天**：冻结 Qwen 对齐后进行 LoRA 任务微调；并行跑无 Qwen、随机主干、直接拼接和文本化消融。
6. **第 11—12 天**：完成时间外、实体外、交集、缺失、单位、异常值、schema 重排与漂移评测；执行 G3/G4。
7. **第 13 天**：在同硬件测 p50/p95/p99、吞吐、峰值显存和单位成本；执行 G5。
8. **第 14 天**：按门禁作出继续 LoRA/局部解冻、降级晚融合、或停止 Qwen 路线的单一决策，并冻结所有失败样本与复现实验。

---

## 待验证与口径说明

> 🔍 **待验证与口径说明**（当前结论用于方案预研，不作生产效果承诺）：
>
> - Qwen2.5/Qwen3 的 `generate(inputs_embeds=...)` 在目标固定版本、流式生成、beam search、KV cache 和混合精度下的行为需通过 G1 实测。
> - 目标任务的输入模态、样本量、标签成熟期、实体定义、SLO 与误判成本未提供，Δ、延迟和成本阈值需在 PoC 启动前替换默认值。
> - TsLLM 的公开代码、数据许可证与复现状态本轮未核验；报告仅使用论文方法与结果，不把其工程可复现性作为既定事实。[[10]](https://arxiv.org/abs/2510.01111v2)
> - TabPFN/TabICL 等表格基础模型的具体版本、许可证、最大样本/特征约束和目标硬件表现未纳入本轮，可在确定数据规模后补充。
> - 漂移、OOD 和校准阈值不存在跨业务通用常数，需用历史稳定窗口、已知事故窗口与 shadow 流量标定。
> - 本轮为控制调研范围，未展开模型压缩、量化部署、数据合成和隐私计算；只有 G3/G4 通过后，这些方向才具有投入优先级。

---

## 参考文献

- [1] Qwen Team, Qwen2.5 Technical Report, 2025-01-03. https://arxiv.org/abs/2412.15115
- [2] Hugging Face Transformers, modeling_qwen2.py, 2026-09-11 访问. https://github.com/huggingface/transformers/blob/main/src/transformers/models/qwen2/modeling_qwen2.py
- [3] Hugging Face Transformers, modeling_qwen3.py, 2026-09-11 访问. https://github.com/huggingface/transformers/blob/main/src/transformers/models/qwen3/modeling_qwen3.py
- [4] Qwen, Qwen2.5-7B-Instruct config.json, 2026-09-11 访问. https://huggingface.co/Qwen/Qwen2.5-7B-Instruct/raw/main/config.json
- [5] Qwen, Qwen3-8B config.json, 2026-09-11 访问. https://huggingface.co/Qwen/Qwen3-8B/raw/main/config.json
- [6] Qwen Documentation, Key Concepts: Tokens, Tokenization and Chat Templates, 2026-09-11 访问. https://qwen.readthedocs.io/en/latest/getting_started/concepts.html
- [7] Qwen, Qwen2.5-7B-Instruct Model Card, 2026-09-11 访问. https://huggingface.co/Qwen/Qwen2.5-7B-Instruct
- [8] Qwen, Qwen3-8B Model Card, 2026-09-11 访问. https://huggingface.co/Qwen/Qwen3-8B
- [9] Qwen Documentation, MS-SWIFT Training Guide, 2026-09-11 访问. https://qwen.readthedocs.io/en/latest/training/ms_swift.html
- [10] TsLLM Authors, Towards Interpretable Time Series Foundation Models / TsLLM method evidence, 2026-03-14. https://arxiv.org/abs/2510.01111v2
- [11] Qwen Team, Qwen2.5-VL Technical Report, 2025-02-19. https://arxiv.org/abs/2502.13923
- [12] Qwen Team, Qwen3-VL Technical Report, 2025-11-27. https://arxiv.org/abs/2511.21631v2
- [13] Authors, Towards Interpretable Time Series Foundation Models, 2025-07-10. https://arxiv.org/abs/2507.07439
- [14] Hegselmann et al., TabLLM: Few-shot Classification of Tabular Data with Large Language Models, 2023-03-17. https://arxiv.org/abs/2210.10723v2
- [15] Authors, TabuLa-8B, 2024-11-20. https://arxiv.org/abs/2406.12031v2
- [16] Gorishniy et al., On Embeddings for Numerical Features in Tabular Deep Learning, 2023-10-26. https://arxiv.org/abs/2203.05556v4
- [17] Golkar et al., xVal: A Continuous Number Encoding for Large Language Models, 2024-12-15. https://arxiv.org/abs/2310.02989v2
- [18] Authors, Soft-token Numerical Fusion with Qwen3 for Sequential Recommendation, 2026-08-01. https://arxiv.org/abs/2607.10016v3
- [19] Jin et al., Time-LLM: Time Series Forecasting by Reprogramming Large Language Models, 2024-01-29. https://arxiv.org/abs/2310.01728v2
- [20] Tan et al., Are Language Models Actually Useful for Time Series Forecasting?, 2024-10-26. https://arxiv.org/abs/2406.16964v2
- [21] Authors, QRData: A Relational Data Benchmark for Quantitative Reasoning, 2024-02-27. https://arxiv.org/abs/2402.17644v1
- [22] Authors, NumericBench, 2025-06-03. https://arxiv.org/abs/2502.11075v2
- [23] Authors, TableBench, 2024-08-17. https://arxiv.org/abs/2408.09174v1
- [24] Gorishniy et al., Revisiting Deep Learning Models for Tabular Data, 2023-10-26. https://arxiv.org/abs/2106.11959
- [25] Grinsztajn et al., Why do tree-based models still outperform deep learning on typical tabular data?, 2022-07-18. https://arxiv.org/abs/2207.08815
- [26] scikit-learn, GroupKFold Documentation, 2026-09-11 访问. https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GroupKFold.html
- [27] scikit-learn, TimeSeriesSplit Documentation, 2026-09-11 访问. https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.TimeSeriesSplit.html
- [28] scikit-learn, Common Pitfalls and Recommended Practices, 2026-09-11 访问. https://scikit-learn.org/stable/common_pitfalls.html
- [29] Guo et al., On Calibration of Modern Neural Networks, 2017-08-03. https://arxiv.org/abs/1706.04599
- [30] Hendrycks and Gimpel, A Baseline for Detecting Misclassified and Out-of-Distribution Examples, 2018-10-03. https://arxiv.org/abs/1610.02136
- [31] Howard and Ruder, Universal Language Model Fine-tuning for Text Classification, 2018-05-23. https://arxiv.org/abs/1801.06146
- [32] Chen et al., GradNorm: Gradient Normalization for Adaptive Loss Balancing in Deep Multitask Networks, 2018-06-12. https://arxiv.org/abs/1711.02257
- [33] Kirkpatrick et al., Overcoming Catastrophic Forgetting in Neural Networks, 2017-01-25. https://arxiv.org/abs/1612.00796
- [34] Walsh et al., DOME: Recommendations for supervised machine learning validation in biology, 2021-01-07. https://arxiv.org/abs/2006.16189
- [35] Viering and Loog, The Shape of Learning Curves: a Review, 2025-01-28. https://arxiv.org/abs/2201.12150
