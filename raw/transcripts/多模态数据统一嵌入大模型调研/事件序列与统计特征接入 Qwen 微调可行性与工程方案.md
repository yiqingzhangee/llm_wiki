# 事件序列与统计特征接入 Qwen：微调可行性与工程方案

**报告日期：** 2026 年 9 月 11 日　**面向：** 模型、数据与在线服务工程团队　**结论性质：** 公开证据支持下的可行性预研

> **🎯 核心判断**
>
> 1. **可行，但不宜从结构改造起步**：先证明事件顺序和统计特征各自提供稳定增益。
> 2. **首选数值适配器加 LoRA**：连续特征进入少量软 token，兼顾表达能力与改造成本。
> 3. **强基线决定项目是否成立**：LightGBM 与小型 GRU 必须和 Qwen 同切分、同预算比较。
> 4. **上线由最差桶与成本裁决**：平均指标提升不能覆盖泄漏、失校准、漂移或延迟超标。

---

## 一、问题定义：先对齐语义与尺度

> **本节要点**：当前任务同时包含离散事件顺序与连续统计量。工程上应先固定预测时点、输出任务和可见字段，再讨论 Qwen 的训练方式。

### 1.1 输入、输出与术语边界

按现有材料，单个样本含一条约 8 步的事件序列，以及与该序列对应的一组统计特征。事件 code 若有稳定业务语义，可映射为短文本标签；若只是匿名编号，Qwen 的语言先验难以直接发挥作用，模型主要学习的是顺序共现。统计特征则包含尺度、单位、缺失和极值，其信息几何不能只靠自然语言描述来保证。

本报告将用户表述中的“重新 Qwen 模型”解释为“对 Qwen 进行再训练或微调”。四类动作需要严格区分：

- **提示词输入**：把事件和统计量确定性序列化后送入现有 Qwen，不更新参数；适合零样本验证与格式敏感性测试。
- **参数高效微调**：以 LoRA 或 QLoRA 更新 Qwen 内部少量低秩参数，输入仍是 token；LoRA 可合并部署，但不会自动获得连续数值通道。LoRA 原论文展示了显著降低可训练参数与训练显存的能力，具体收益依模型和任务而变 [3](https://arxiv.org/abs/2106.09685)。
- **适配器融合**：新增数值或时序编码器，把统计特征映射成软 token、前缀键值或侧路 memory，再选择是否叠加 LoRA；新增模块需单独保存、训练和服务。
- **结构改造**：修改 tokenizer、embedding、输出头，或在 Qwen 层间插入交叉注意力；表达能力更强，训练稳定性、权重兼容和在线部署成本也最高。

输出任务尚未锁定。分类或风险评分要求概率校准；下一事件预测关注 Acc@k、F1 与排序质量；连续值预测使用 MAE、MSE 或概率损失；自然语言解释还需事实一致性评测。任务头不同，公开论文的可迁移性也不同。

### 1.2 成功标准与硬约束

| 决策项 | 必须满足 | 验证口径 | 未满足时处理 |
|:---:|---|---|---|
| 数据正确性 | 预测时点后信息零进入 | 按时间先切分，再构窗与拟合预处理 | 整批结果作废，重建数据集 |
| 增量价值 | 稳定超过强基线 | 至少 3 个滚动时间折与实体簇置信区间 | 保留 GBDT 或 GRU，不升级 Qwen |
| 概率可靠性 | 高风险区间可校准 | Brier、log loss、ECE 与可靠性图 | 独立校准或输出拒答 |
| 鲁棒性 | 缺失、错位、噪声下可控 | 最差桶退化与回退触发率 | 回退到规则、GBDT 或 GRU |
| 在线成本 | 满足既定延迟与日成本 SLO | 真实硬件 p50/p95/p99、吞吐和显存 | 缩小模型、减少软 token 或停止上线 |

*表 1：成功标准先按硬门槛判断，再比较模型上限*

Qwen 官方训练文档已覆盖 LoRA、全参监督微调及多种分布式训练框架，但示例显存与吞吐仅适用于对应模型、长度和硬件，不能直接作为本项目容量承诺 [2](https://qwen.readthedocs.io/en/latest/training/ms_swift.html)。首轮应锁定具体 Qwen 版本、精度、最大长度和推理引擎，避免把版本差异误认为架构差异。

---

## 二、方案空间：四档改造逐级增加

> **本节要点**：提示词、参数高效微调、适配器融合与结构改造解决的问题不同。推荐路径是先验证信息价值，再增加连续表示能力，最后才扩大模型结构。

### 2.1 文本序列化与结构化提示

最低成本方案是使用固定模板，将事件写成“时间、事件类型、关键属性”的有序列表，将统计量写成字段名、单位、值和缺失标记。LLM-Mob 证明离散位置事件可通过提示直接用于下一位置预测，但其闭集位置 ID 任务并未覆盖高维连续统计量 [8](https://arxiv.org/abs/2308.15197)。TabLLM 则显示，人工确定性模板在多项表格任务的零样本和少样本区间具有竞争力，复杂自动描述反而会遗漏或虚构字段 [9](https://arxiv.org/abs/2210.10723)。

该路线的价值是建立可复现的 Qwen 基线，并验证“加上统计特征是否有用”。格式必须版本化：字段顺序、数值精度、空格、分隔符、单位和截断策略均进入配置。LLMTime 观察到不同 tokenizer 对数字空格规则的最优选择相反，说明格式属于模型机制变量 [7](https://arxiv.org/abs/2310.07820)。

### 2.2 数值编码器与软提示融合

推荐的中间路线将标准化后的统计向量、缺失掩码和时间间隔送入小型多层感知机，输出 K 个与 Qwen 隐藏维度一致的软 token。软 token 与事件文本 embedding 拼接；融合投影最后一层或门控参数零初始化，使训练起点近似原 Qwen。Prefix-Tuning 已验证冻结语言模型并训练连续虚拟 token 的可行性 [5](https://arxiv.org/abs/2101.00190)，LLaMA-Adapter 的零初始化注意力门则提供了渐进注入外部信号的稳定化依据 [6](https://arxiv.org/abs/2303.16199)。

**图 1：事件序列与统计特征接入 Qwen 的分层架构**

推荐主线是保留原生文本通道，以轻量数值适配器生成少量条件 token，再用零初始化门控渐进注入。架构自下而上如下：

| 层级 | 作用 | 关键内容 |
|---|---|---|
| 数据层 L0 | 样本契约与预处理 | 事件序列包含 code、顺序、时间戳和真实间隔；统计特征保留量纲、范围、缺失率和更新时间；先切分再构窗；批处理同步返回文本 mask、feature mask 与 labels |
| 编码层 L1 | 双路表示 | 事件序列通过确定性模板进入文本 token，保留顺序、位置和 Δt；统计特征经标准化、裁剪和小型 MLP，形成带 feature mask 的连续表示 |
| 融合层 L2 | 对齐、拼接与零初始化门控 | 数值表示投影到 Qwen hidden size，压缩为少量 soft tokens，与文本 embedding 拼接并同步扩展 attention mask；zero gate 从零附近起步 |
| 主干层 L3 | Qwen 语言主干 | 原生自注意力统一处理文本 token 与 soft tokens；首轮冻结主干，必要时训练 all-linear LoRA；输入前缀和 padding 不计生成损失 |
| 任务层 L4 | 任务输出与解释 | 支持分类或风险评分、下一事件或数值预测、解释与证据摘要；概率任务独立校准 |

*图 1：输入侧适配器把统计特征压缩为少量条件 token，Qwen 主干保持可控改造。越接近数据契约，越决定信息是否真实可用。*

这一路线与 LoRA 可组合但概念不同：适配器负责建立连续特征通道，LoRA 负责调整 Qwen 对该通道和事件文本的使用方式。冻结 Qwen、只训练适配器可降低遗忘；若输出语言行为或任务边界变化明显，再启用 all-linear LoRA。Qwen 官方生态已有 LoRA 训练路径 [2](https://qwen.readthedocs.io/en/latest/training/ms_swift.html)，自定义适配器则仍需单独实现模型类与服务封装。

### 2.3 交叉注意力与时序编码器

当模型需要在生成每个 token 时动态检索不同统计字段，可让 Qwen hidden state 作 query、数值 memory token 作 key/value，并在少数高层插入零门控交叉注意力。Flamingo 和 BLIP-2 已分别证明冻结语言模型与外部编码器之间可通过桥接模块建立跨模态交互 [28](https://arxiv.org/abs/2204.14198) [27](https://arxiv.org/abs/2301.12597)；迁移到统计特征属于工程推断，尚无 Qwen 官方实现。

交叉注意力比输入拼接更有容量，但会改变 forward、KV 管理、权重保存和推理算子。首个 PoC 只在少数高层插入，且必须保留“不插交叉注意力、仅输入软 token”的同预算对照。

### 2.4 离散化、基础模型与工具增强

离散数值 token 适合生成式概率预测。Chronos 将归一化序列量化到 4096 个 token，以标准交叉熵预训练，在其时序预测基准上取得较强域内与零样本结果 [11](https://arxiv.org/abs/2403.07815)。代价是量化误差、区间边界和新 embedding/head；xVal 与 NumGPT 进一步表明，显式数值表示能提高 token 效率或数值任务表现，同时会改变输入或输出接口 [12](https://arxiv.org/abs/2310.02989) [13](https://arxiv.org/abs/2109.03137)。

若目标主要是精确计算或阈值判断，可把统计计算保留在确定性服务中，让 Qwen 调用工具并解释结果。该备选路线牺牲端到端统一性，却能把数值正确性和审计边界留在可测试代码里，适合高风险场景。

---

## 三、融合机制：统计量应参与条件推理

> **本节要点**：稳定融合依赖三件事：时间一致的数据契约、可控的表示对齐，以及只让正确参数接收正确梯度。

### 3.1 数据到 Qwen 的端到端链路

一条样本在预测时点 t 截止：事件侧保留有序 code、时间间隔和可见属性；统计侧只用 t 之前可计算的聚合量。两路分别编码后，在输入层或高层融合，再由 Qwen 产生分类、排序、回归或解释输出。Transformers 的数据整理器支持动态 padding 和以 -100 忽略标签位置，但自定义连续前缀还需同步扩展 attention mask、position ids 与 feature mask [29](https://huggingface.co/docs/transformers/main/en/main_classes/data_collator)。

**图 2：从数据契约到 Qwen 输出的端到端链路**

| 步骤 | 处理 | 输出或控制点 |
|:---:|---|---|
| 1 | **无泄漏样本构造**：按时间先切分，再在各分区内构窗；锁定预测时点可见字段 | 事件序列、统计向量、缺失标记、标签 |
| 2 | **双路编码**：事件侧使用确定性模板与 tokenizer；数值侧标准化、裁剪并经小型 MLP 编码 | 保留顺序、Δt、量纲和 missing mask |
| 3 | **对齐与门控融合**：把连续表示投影为少量 soft tokens；零初始化门控后与文本 embedding 拼接 | attention、position 和 loss mask 同步扩展 |
| 4 | **Qwen 条件推理**：首轮冻结 Qwen；如需改变语言行为，再叠加 LoRA 联合训练 | 梯度优先流向 side MLP、投影层、gate 和 LoRA |
| 5 | **任务头输出**：按目标选择分类、回归、下一事件或自然语言解释，并独立完成校准 | 以强基线、缺失/OOD、延迟和成本消融验收 |

*图 2：先锁定时点与双路编码，再进入对齐、融合和任务输出。若融合层没有稳定优于“无特征 / 乱序特征”对照，应回退文本化基线，而非继续加深结构。*

指令微调只在 assistant 输出 token 上计算语言损失；system、user、padding 和连续前缀位置均屏蔽。若让模型复现输入模板，训练损失会被字段抄写占据，模型行为也更容易偏移。

### 3.2 对齐、门控与缺失值

连续特征先按训练窗口拟合标准化参数，并按业务含义分组投影。每个值至少带三类信息：标准化数值、是否缺失、距最近观测的时间。缺失值只填 0 会混淆“真实为 0”和“未知”，必须有独立 mask。

适配器输出可加 LayerNorm，再通过零初始化门控残差注入：$h' = h + \tanh(g)\operatorname{Proj}(z)$。训练开始时 $g = 0$，主干行为近似原模型；随后由任务损失决定注入强度。对齐预训练可先使用统计量重建、事件—统计对比学习或掩码恢复，再进入监督任务。TEST 采用多种对比目标把时序表示对齐到冻结语言模型 embedding 空间，说明“先对齐、后任务训练”具备公开实验依据 [15](https://arxiv.org/abs/2308.08241)。

### 3.3 训练目标与梯度路径

联合目标可写为 $L = \lambda_{task}L_{task} + \lambda_{align}L_{align} + \lambda_{cal}L_{cal} + \lambda_{reg}L_{reg}$。其中任务损失按输出类型选择交叉熵、排序损失或回归损失；对齐损失约束事件与统计表示；校准项只在概率任务使用；正则项控制门控幅度和适配器漂移。

- 阶段 A：Qwen 冻结，只训练数值编码器、投影与门控，确认连续通道能收敛。
- 阶段 B：启用 Qwen LoRA，适配器学习率高于 LoRA；保留梯度裁剪、warmup 和 bf16 计算。
- 阶段 C：仅在证据充分时解冻少量归一化层、embedding 或高层 block；全参训练不作为默认路径。

Time-LLM 通过 patch 重编程和统计文本前缀连接冻结 LLM，论文报告可训练桥接参数少于 6.6M，约占 Llama-7B 的 0.2%；推理仍需承载完整 7B 主干 [14](https://arxiv.org/abs/2310.01728)。因此“训练参数少”不能替代总显存、吞吐与端到端延迟测量。

---

## 四、证据比较：尚无通用最优结构

> **本节要点**：公开研究支持多种混合输入方式的可行性，却没有在同一 Qwen、同一数据与同一预算下给出直接胜者。选型仍须依赖本地强基线和消融。

### 4.1 代表性研究与工程范式

| 范式 | 代表研究 | 已证明的能力 | 对本任务的启示 | 主要边界 |
|:---:|:---:|---|---|---|
| 文本化 | LLMTime、TabLLM | 数字串预测、表格少样本分类可行 | 最快建立 Qwen 基线 | 格式、精度、长度敏感 |
| 事件提示 | LLM-Mob | 离散移动事件可做下一位置预测 | 事件 code 可先模板化 | 闭集 ID，连续特征证据不足 |
| 离散 token | Chronos | 统一为生成式概率预测 | 适合大规模时序预训练 | 量化误差与新词表成本 |
| 连续编码 | xVal、NumGPT | 更显式保留数值结构 | 支持紧凑数值表示 | 需改 embedding 或输出头 |
| 软提示 | TEST、Time-LLM | 冻结 LLM 下可完成对齐与下游任务 | 最接近推荐适配器路线 | 跨任务结果不可直接横比 |
| 对齐加 LoRA | LLM4TS | 两阶段时序对齐与预测微调 | 支持适配器后叠加 PEFT | GPT-2 结果不能直接外推 Qwen |

*表 2：代表研究证明“可行”，未证明本任务上的统一胜者（来源：[7](https://arxiv.org/abs/2310.07820)、[11](https://arxiv.org/abs/2403.07815)、[14](https://arxiv.org/abs/2310.01728)、[16](https://arxiv.org/abs/2308.08469)）*

公开结果必须按任务解释。TabLLM 在极少样本时具备优势，但样本增加后逻辑回归或 LightGBM 会在部分数据集反超 [9](https://arxiv.org/abs/2210.10723)；Chronos 的证据主要来自单变量预测 [11](https://arxiv.org/abs/2403.07815)；面向时序推理的编码器方案在部分分类任务超过 GPT-4o，却明确未验证时序生成与预测 [17](https://arxiv.org/abs/2409.11376)。这些结果支持路线筛选，不能代替本地结果。

### 4.2 效果、成本与适用边界

**图 3：四类 Qwen 融合路线的相对取舍**

| 路线 | 效果潜力 | 低实施成本 | 训练稳定性 | 可解释性 | 部署兼容性 |
|---|:---:|:---:|:---:|:---:|:---:|
| 文本化 | 2 | 5 | 4 | 4 | 5 |
| 离散化 | 4 | 2 | 3 | 3 | 2 |
| 适配器融合 | 4 | 3 | 4 | 4 | 3 |
| 深度结构改造 | 5 | 1 | 2 | 3 | 1 |

*图 3：四条路线的相对工程评分。采用 1–5 级，越高越优；“低实施成本”分数越高表示成本越低。评分依据公开论文机制、Qwen/PEFT/vLLM 工程兼容性与风险证据归纳，仅用于选型讨论，不代表统一数据集、同预算条件下的 benchmark 结果。*

文本化在稳定性、成本和部署兼容性上占优，适合作为第一基线；适配器融合在效果潜力与工程代价之间较平衡；离散化更适合大规模生成式时序预训练；结构改造只在复杂交互确有证据时合理。评分属于基于论文与部署约束的工程判断，真正的排序由目标数据上的同预算实验确定。

Qwen 也不应垄断候选集。LightGBM 适合固定 8 步序列衍生出的计数、转移和窗口统计，训练与解释成本低 [21](https://lightgbm.readthedocs.io/en/stable/Features.html)。统一 TPP 实验中，GRU 历史编码器在多项时间和 mark 指标上总体优于 self-attention，为“短序列先测 GRU”提供直接反证 [20](https://arxiv.org/abs/2306.17066)。DLinear 在多项长时序基准上击败更复杂的 Transformer，也说明复杂度不能作为性能先验 [22](https://arxiv.org/abs/2205.13504)。

---

## 五、推荐方案：三阶段控制改造风险

> **本节要点**：首轮目标是确认信息增益，第二轮确认连续融合价值，第三轮才验证深层交互。任一阶段未过门槛即停止升级。

### 5.1 第一阶段：文本化与强基线

**周期建议：1–2 周。** 先冻结样本时点、字段字典、单位、缺失语义和标签成熟期；按时间切分后分别构窗。并行训练常数先验、规则、逻辑回归、LightGBM/XGBoost、小型 GRU、Qwen 纯提示和 Qwen 文本 LoRA。所有模型共用样本、切分、特征可见性和评测脚本。

- 事件模板：保留顺序、事件名称、时间间隔与未知 code 标记。
- 统计模板：固定字段顺序、单位、精度、裁剪边界和缺失标记。
- Qwen 训练：优先 LoRA；显存不足时评估 QLoRA。QLoRA 以冻结 4-bit 基座反传至 LoRA，可大幅降低特定实验的显存占用，但不保证在目标 Qwen 上训练更快或效果等价 [4](https://arxiv.org/abs/2305.14314)。

阶段出口：Qwen 文本 LoRA 在多个滚动时间折上稳定优于无特征 Qwen，并且至少在“解释质量、跨任务复用或小样本”一项展现 GBDT/GRU 无法替代的价值。若只在平均分类指标上接近强基线，保留简单模型更合理。

### 5.2 第二阶段：数值适配器融合

**周期建议：2–4 周。** 实现“标准化与 mask → 小型 MLP → 4–16 个软 token → 零门控 → Qwen”，先冻结 Qwen，随后叠加 all-linear LoRA。K 的范围、投影深度、门控粒度和特征分组进入消融；训练与服务同时记录额外参数、显存和延迟。

部署建议保留两个制品：Qwen LoRA adapter 与数值适配器权重。标准 vLLM 可按请求加载 LoRA adapter [30](https://docs.vllm.ai/en/stable/features/lora/)，但自定义软 token 或 side encoder 需要验证 prompt embedding 接口、批处理和缓存行为。若目标推理栈无法稳定承载，备选是把连续特征离线量化后恢复为文本前缀。

### 5.3 第三阶段：条件式结构升级

**周期建议：4–8 周，满足触发条件后启动。** 只有当软 token 出现可复现的容量瓶颈，例如复杂问题需要反复检索不同字段，才在少数高层插入零门控交叉注意力。若任务是连续时间到达与事件类型联合预测，另开 marked temporal point process（标记时间点过程）分支；若有效长度扩展到数百或更长且注意力成为瓶颈，再评估 Transformer 或选择性状态空间模型。

Mamba 的线性长度扩展和高吞吐来自长序列场景；固定 8 步输入没有足够理由承担其编译链与数值稳定性成本 [23](https://arxiv.org/abs/2312.00752)。长上下文扩窗也应独立评估，不能和特征融合捆绑。注意力长度从 2048 增至 8192 时，核心计算量近似增加 16 倍，扩窗会迅速改变成本边界。

| 阶段 | 主要产物 | 比较对象 | 进入条件 | 停止条件 |
|:---:|---|---|---|---|
| P0 数据与基线 | 无泄漏切分、GBDT、GRU、Qwen 文本基线 | 无特征、事件单路、统计单路、双路文本 | 数据契约通过审计 | Qwen 无稳定增量价值 |
| P1 适配器融合 | 软 token 适配器、LoRA、服务原型 | 文本 LoRA、适配器冻结主干、适配器加 LoRA | P0 证明双路信息有效 | 增益不足以覆盖部署成本 |
| P2 结构升级 | 稀疏高层交叉注意力或专项模型 | 输入软 token 与同参数量侧路 | 出现可复现容量瓶颈 | 最差桶、延迟或稳定性退化 |

*表 3：每阶段都以明确证据决定是否增加模型复杂度*

---

## 六、验证设计：消融与压力测试裁决

> **本节要点**：有效验证要同时回答“信息是否有用、顺序是否有用、融合是否有用、成本是否可接受”，并把时间泄漏与概率失准排除在外。

### 6.1 数据切分、基线与消融

主协议使用至少 3 个滚动 origin 或 expanding-window 外推折，每折满足 train 早于 validation、validation 早于 test，并在边界设置覆盖最大标签预测期和特征回看重叠的 gap。最终最近时间窗只作锁定测试。TimeSeriesSplit 保持时间顺序并支持 gap [18](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.TimeSeriesSplit.html)；先构造重叠窗口再随机切分会引入未来信息，一项控制实验报告泄漏配置的 RMSE 偏差最高达 20.5%，该数值仅用于说明风险量级 [19](https://arxiv.org/abs/2512.06932)。

| 实验组 | 配置 | 回答的问题 | 最低要求 |
|:---:|---|---|---|
| 强基线 | 常数/规则、逻辑回归、LightGBM、GRU | 复杂模型是否提供真实增益 | 同样本、同切分、同预算报告 |
| Qwen 基线 | 零样本提示、文本 LoRA、去统计特征 | 微调与统计信息各贡献多少 | 固定模板、tokenizer 与长度 |
| 序列消融 | L=1/2/4/8、打乱、删除、交换、未知 code | 顺序和历史长度是否有贡献 | 固定参数量和随机种子 |
| 统计消融 | 统计单路、随机置换、无 mask、无时间间隔 | 连续特征与缺失语义是否被使用 | 逐组增删，避免联合变化 |
| 融合消融 | 文本拼接、软 token、零门控、交叉注意力 | 结构增益是否超过参数量效应 | 参数量匹配并单列推理成本 |
| 泄漏对照 | clean 与 intentional-leaky 双管线 | 提升是否来自切分或预处理泄漏 | clean 为唯一有效结果 |

*表 4：强基线与消融矩阵把模型增益拆到可归因层*

所有预处理只在训练窗拟合，包括归一化、插补、事件字典、目标编码和聚合阈值。同一实体或强相关群组需要做 group 隔离；标签成熟期不足或会回溯修订的样本不得进入评估。

### 6.2 指标、鲁棒性与上线门槛

分类与风险评分同时报告 ROC-AUC、PR-AUC、log loss、Brier、ECE 和业务成本曲线；下一事件预测报告 Acc@k、F1 与 nDCG；连续预测报告 MAE/MSE 与分位数损失。Marked TPP 研究建议把时间与事件类型的 NLL、校准和 F1 分开，避免总损失掩盖单项失败 [20](https://arxiv.org/abs/2306.17066)。校准器只能在独立 calibration window 拟合；样本较少时优先 sigmoid，isotonic 在校准样本远少于 1000 时容易过拟合 [26](https://scikit-learn.org/stable/modules/generated/sklearn.calibration.CalibratedClassifierCV.html)。

压力测试覆盖连续值加噪、削峰、时间错位、整通道缺失、随机点缺失，以及事件删除、插入、交换和未知 code。外生变量研究表明，通道缺失和错位会显著拖累预测，而整通道 dropout 可改善多类污染下的鲁棒性 [24](https://arxiv.org/abs/2607.05452)。漂移测试按时间、实体、地域、新 code、缺失率、采样间隔和长度分桶，并报告最差桶、相对退化与检测延迟 [25](https://arxiv.org/abs/1707.07821)。

**建议上线门槛**需在试验前写入评审单，避免看到结果后调整标准：

1. 全部滚动时间折相对强基线的主指标方向一致，实体簇 bootstrap 的 95% 置信区间下界高于预先约定的可接受退化线。
2. 泄漏审计零项未关闭；离线与在线特征定义、单位、缺失语义和版本映射一致率为 100%。
3. 校准后的 Brier、log loss 与 ECE 不劣于强基线，高风险阈值同时满足精确率、召回率、业务容量和单位成本约束。
4. 预定义缺失、错位、噪声与分布外压测中，最差桶不突破业务风险预算；超限可自动回退或拒答。
5. 目标硬件上的端到端 p99、峰值显存、吞吐与单日成本满足 SLO，且包含预处理、适配器和后处理。

---

## 七、风险控制：先防泄漏与失校准

> **本节要点**：主要失败通常来自数据时点、数值表示、分布变化和服务兼容，而非 Qwen 参数量不足。风险应有可观测信号和自动回退路径。

### 7.1 主要失败模式

| 风险 | 说明 |
|---|---|
| ⚠️ **时间泄漏** | 切分前构窗、全量拟合归一化或标签回溯，会制造不可上线的离线增益。 |
| ⚠️ **数值幻觉** | 数字分词、单位变化和极值可诱发伪相关；生成概率不能直接视为风险概率。 |
| ⚠️ **分布漂移** | 新事件 code、统计口径变化和缺失率上升会使适配器离开训练分布。 |
| ⚠️ **服务不兼容** | LoRA 可走标准服务链路，自定义 side encoder 与交叉注意力需要额外运行时。 |

xVal 在科学数值序列实验中观察到文本数值编码的伪相关 [12](https://arxiv.org/abs/2310.02989)；Table Meets LLM 也显示模型对尺寸、单元格定位等基础结构问题仍会出错 [10](https://arxiv.org/abs/2305.13062)。因此输入模板要由程序生成并通过 schema 校验，不能让模型自由改写原始字段。

### 7.2 缓解措施与备选路线

- 数据侧维护 feature-as-of-time 清单、血缘与版本号；训练、回放和在线使用同一特征函数。
- 表示侧固定 tokenizer 与模板版本，记录截断率、clipping 率、未知 code 率和单位变更。
- 训练侧使用零初始化门控、分组学习率、梯度裁剪和短到长课程；每次升级只改变一个核心变量。
- 监控侧同时看主指标、校准、最差桶、漂移检测、延迟和回退率；模型与校准器绑定版本。
- 回退侧保留 GBDT/GRU 评分与规则路径；高风险计算交给确定性工具，Qwen 只负责语义解释。

备选路线应在同一服务协议下返回统一结构，包含预测值、置信或校准概率、证据字段、模型版本和回退原因。这样 Qwen 路线即使停止升级，数据契约和评测资产仍可复用。

---

## 八、综合判断：先适配融合，再决定深改

> **📌 综合判断**：建议采用“强基线 → 文本 LoRA → 数值适配器加 LoRA”的主线，暂不直接改造 Qwen 主干。
>
> 这一结论来自三组证据：确定性文本模板已经能承载事件与表格信息；软提示和桥接模块已在多类任务证明可行；短序列上存在 GBDT、GRU 和线性模型击败复杂注意力模型的反例。结构升级应由同预算消融触发。
>
> **残余风险 / 反向信号：**
>
> 1. 若事件 code 无可读语义且统计特征已被 GBDT 充分利用，Qwen 的边际价值可能很小。
> 2. 若目标要求严格概率与极低延迟，专用模型或确定性工具可能长期优于生成模型。

### 8.1 决策顺序

工程决策应按可逆性推进。P0 的数据契约、强基线和文本模板成本低、可快速否证；P1 的输入侧适配器增加有限自定义模块，仍可回退到原生 Qwen；P2 的交叉注意力、词表或输出头改造会扩大训练和服务耦合，只在 P1 明确暴露容量瓶颈时投入。

最终“上线”条件不是 Qwen 在某一平均指标上获胜，而是它在时间外推、校准、最差桶和目标硬件成本上同时通过门槛，并且提供简单模型无法替代的语义价值。否则，将 Qwen 保留为解释层，把预测交给 GBDT 或 GRU。

### 8.2 最强反对意见

#### 8.2.1 八步序列没有深模型空间

**反对意见：** 只有 8 步离散行为，复杂模型缺少足够的长程结构可学习。**回应：** 该质疑成立，因此 P0 必须包含位置化 GBDT 与小型 GRU，并做 L=1/2/4/8 和顺序打乱消融。若 Qwen 不能证明顺序或语义增益，项目应停在简单模型；TPP 统一实验中 GRU 优于 self-attention 的结果强化了这一退出条件 [20](https://arxiv.org/abs/2306.17066)。

#### 8.2.2 LoRA 已足够，无需适配器

**反对意见：** 把统计量文本化后做 LoRA 已能解决问题，新增数值通道只会抬高维护成本。**回应：** 文本 LoRA 确实是必须击败的基线；适配器仅在数值精度、长度、缺失模式或尺度几何成为瓶颈时继续。通过“文本 LoRA vs 软 token 冻结主干 vs 软 token 加 LoRA”的三组同预算实验，可以把连续表示价值与参数更新价值拆开。

#### 8.2.3 离线提升可能全是泄漏

**反对意见：** 序列窗口和聚合统计极易包含未来信息，任何小幅提升都不可信。**回应：** 该风险不能靠模型设计缓解，只能靠先切分后构窗、gap、标签成熟期、实体隔离与 intentional-leaky 对照处理。泄漏审计有一项未关闭，所有模型均不得进入影子流量。

#### 8.2.4 大模型成本难以回收

**反对意见：** 即使适配器只训练少量参数，推理仍需完整 Qwen，成本可能高于收益。**回应：** 评审采用端到端 p99、吞吐、峰值显存和单日成本，不以可训练参数比例替代服务成本。若语义解释可异步生成，可采用“专用模型实时评分 + Qwen 按需解释”；若解释也无业务增量，则不部署 Qwen。

---

## 待验证事项

> **🔍 待验证与口径说明**（以下事项会改变方案选择，需在 PoC 前确认）：
>
> - 目标任务是分类、风险评分、下一事件、连续值预测还是解释生成；由业务负责人确认主指标、误报/漏报成本和标签成熟期。
> - 事件 code 是否有稳定语义、是否带真实时间戳，以及序列是否固定为 8 步；由数据负责人输出字段字典和样例。
> - 统计特征维度、单位、缺失率、极值范围、更新时间和回溯修订规则；由特征平台负责人完成 as-of-time 审计。
> - 目标 Qwen 版本、PEFT target module、推理引擎和软 prompt 接口兼容性；Qwen 官方公开训练页仍主要提供特定版本示例 [2](https://qwen.readthedocs.io/en/latest/training/ms_swift.html)。
> - 实际 GPU、batch、上下文长度、日调用量和 p99 SLO 未提供；所有显存与吞吐数字必须在目标硬件复测。
> - 公开研究没有在同一 Qwen、同一数据切分和同一预算下比较四条路线；本地消融是最终决策依据。
> - 开发者生态、长期运维人力与安全合规评估本轮因优先保证核心技术决策暂未展开，可在 P1 通过后补充。

---

## 参考文献

1. [Qwen Team, *Qwen Documentation*, 2026-09-11 访问](https://qwen.readthedocs.io/en/latest/)
2. [Qwen Team, *MS-SWIFT Training Guide*, 2026-09-11 访问](https://qwen.readthedocs.io/en/latest/training/ms_swift.html)
3. [Hu et al., *LoRA: Low-Rank Adaptation of Large Language Models*, 2021](https://arxiv.org/abs/2106.09685)
4. [Dettmers et al., *QLoRA: Efficient Finetuning of Quantized LLMs*, 2023](https://arxiv.org/abs/2305.14314)
5. [Li and Liang, *Prefix-Tuning: Optimizing Continuous Prompts for Generation*, 2021](https://arxiv.org/abs/2101.00190)
6. [Zhang et al., *LLaMA-Adapter: Efficient Fine-tuning of Language Models with Zero-init Attention*, 2023/2024](https://arxiv.org/abs/2303.16199)
7. [Gruver et al., *Large Language Models Are Zero-Shot Time Series Forecasters*, 2023](https://arxiv.org/abs/2310.07820)
8. [Wang et al., *Where Would I Go Next? Large Language Models as Human Mobility Predictors*, 2023](https://arxiv.org/abs/2308.15197)
9. [Hegselmann et al., *TabLLM: Few-shot Classification of Tabular Data with Large Language Models*, 2023](https://arxiv.org/abs/2210.10723)
10. [Sui et al., *Table Meets LLM: Can Large Language Models Understand Structured Table Data?*, 2023/2024](https://arxiv.org/abs/2305.13062)
11. [Ansari et al., *Chronos: Learning the Language of Time Series*, 2024](https://arxiv.org/abs/2403.07815)
12. [Golkar et al., *xVal: A Continuous Number Encoding for Large Language Models*, 2024](https://arxiv.org/abs/2310.02989)
13. [Jin et al., *NumGPT: Improving Numeracy Ability of Generative Pre-trained Models*, 2022](https://arxiv.org/abs/2109.03137)
14. [Jin et al., *Time-LLM: Time Series Forecasting by Reprogramming Large Language Models*, 2024](https://arxiv.org/abs/2310.01728)
15. [Sun et al., *TEST: Text Prototype Aligned Embedding to Activate LLM's Ability for Time Series*, 2023](https://arxiv.org/abs/2308.08241)
16. [Chang et al., *LLM4TS: Aligning Pre-Trained LLMs as Data-Efficient Time-Series Forecasters*, 2023](https://arxiv.org/abs/2308.08469)
17. [Mirchandani et al., *Towards Time-Series Reasoning with LLMs*, 2024](https://arxiv.org/abs/2409.11376)
18. [scikit-learn, *TimeSeriesSplit Documentation*, 2026-09-11 访问](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.TimeSeriesSplit.html)
19. [Cao et al., *Data Leakage in Time Series Forecasting*, 2025](https://arxiv.org/abs/2512.06932)
20. [Bosser and Ben Taieb, *On the Predictive Accuracy of Neural Temporal Point Process Models*, 2023](https://arxiv.org/abs/2306.17066)
21. [Microsoft, *LightGBM Features*, 2026-09-11 访问](https://lightgbm.readthedocs.io/en/stable/Features.html)
22. [Zeng et al., *Are Transformers Effective for Time Series Forecasting?*, 2023](https://arxiv.org/abs/2205.13504)
23. [Gu and Dao, *Mamba: Linear-Time Sequence Modeling with Selective State Spaces*, 2024](https://arxiv.org/abs/2312.00752)
24. [Ghosh et al., *Robust Time Series Forecasting with Exogenous Channel Dropout*, 2026](https://arxiv.org/abs/2607.05452)
25. [Lu et al., *Learning under Concept Drift: A Review*, 2019](https://arxiv.org/abs/1707.07821)
26. [scikit-learn, *CalibratedClassifierCV Documentation*, 2026-09-11 访问](https://scikit-learn.org/stable/modules/generated/sklearn.calibration.CalibratedClassifierCV.html)
27. [Li et al., *BLIP-2: Bootstrapping Language-Image Pre-training with Frozen Image Encoders and Large Language Models*, 2023](https://arxiv.org/abs/2301.12597)
28. [Alayrac et al., *Flamingo: a Visual Language Model for Few-Shot Learning*, 2022](https://arxiv.org/abs/2204.14198)
29. [Hugging Face, *Data Collator Documentation*, 2026-09-11 访问](https://huggingface.co/docs/transformers/main/en/main_classes/data_collator)
30. [vLLM, *LoRA Adapters Documentation*, 2026-09-11 访问](https://docs.vllm.ai/en/stable/features/lora/)
