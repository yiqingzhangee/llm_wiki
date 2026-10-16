# LLM for Recommendation 向风控模型迁移：机制、架构与验证路径

> 🎯 **核心判断**
>
> 1. **问题边界**：推荐与风控共享表征问题，但不能共享效用函数。[[1]](https://arxiv.org/abs/2210.10309)
> 2. **迁移机制**：优先迁移语义、时序、检索和受约束决策能力。[[2]](https://arxiv.org/abs/2307.02046v6)
> 3. **目标架构**：大模型离线生产知识，小模型在线评分并由规则执行。[[3]](https://research.google/blog/distilling-step-by-step-outperforming-larger-language-models-with-less-training-data-and-smaller-model-sizes/)
> 4. **方案选择**：混合架构更适合高影响风控，端到端大模型暂不宜进入主决策链。
> 5. **验证路径**：先完成回放与影子评估，再逐步扩大自动决策权限。

**调研口径**：截至 2026 年 9 月 11 日，覆盖交易、账号与内容等通用在线风控。本文讨论技术迁移与工程落地，不绑定单一业务；公开论文中的离线指标均按其原始数据集理解，不直接视为线上收益承诺。

## 一、问题边界：推荐与风控同源但目标不同

> **本节要点**：两类系统都从实体、事件、上下文和关系中学习表示，但推荐优化相关性，风控优化损失约束下的识别与处置。可迁移的是建模机制，目标函数、标签和决策责任必须重建。

### 1.1 LLM for Recommendation 的技术范围

LLM for Recommendation（大语言模型用于推荐）并非单一模型形态。公开综述把它归纳为五类路径：语言模型充当语义特征编码器、在推荐数据上继续预训练、全参数或参数高效微调、通过提示完成零样本或少样本推荐，以及直接生成物品标识或自然语言结果。输入也不限于文本，还包括按时间排序的交互序列、图结构与多模态信息。[[4]](https://arxiv.org/abs/2302.03735v3) [[2]](https://arxiv.org/abs/2307.02046v6)

工程上应把这些路径分为三层。第一层是离线增强：大模型生成画像、标签或向量，现有召回与排序系统继续承担在线决策。P4R 即以大模型生成物品画像，再把语义向量与图卷积网络学习的协同表示对齐。[[5]](https://arxiv.org/abs/2409.16674v3) 第二层是受约束推理：模型读取检索结果、候选集合或指令，完成重排、解释与摘要。第三层才是生成式推荐：模型自回归生成目录内的离散语义标识，并由后处理校验实体、权限和业务规则。

这一区分决定迁移风险。离线增强改变的是特征质量，受约束推理改变的是证据使用方式，生成式决策则同时改变候选空间、评分方式与执行边界。三者不能用同一套“采用大模型”的结论概括。

### 1.2 风控任务的额外硬约束

风控把模型分数转化为审核、拒绝、冻结、上报或放行。错误会产生直接财损、用户误伤和监管责任，因此有四项高于推荐系统的硬约束。

#### 📝 标签稀缺且延迟

> 未拒付、未投诉或未结案不等于正常；时间切分与实体隔离是最低要求。[[6]](https://arxiv.org/abs/2308.02793v2)

#### 💡 错误成本不对称

> 漏放与误杀对应不同损失，单一 AUC、F1 或准确率无法决定动作阈值。[[7]](https://arxiv.org/abs/2401.16458v3)

#### 💡 攻击者主动适应

> 欺诈者会规避、投毒并借正常关系伪装，历史相关性会随对抗改变。[[8]](https://arxiv.org/abs/2005.00625v3)

#### 📋 决策必须可追溯

> 模型版本、证据、阈值与最终动作均需留痕，高影响动作还需人类监督。

CMT 的公开实验说明了“机制可迁移、口径不可照搬”：其双序列 Transformer 与图表示融合后，在微信众包欺诈数据上的 AUC/KS 为 0.9014/0.6624，高于异构图编码器基线的 0.8682/0.5905；但约 680 万用户中只有 53,660 个具人工标签，论文还采用随机 8:1:1 划分。该结果支持序列与关系融合的价值，却不能替代时间外验证、概率校准与线上延迟测试。[[6]](https://arxiv.org/abs/2308.02793v2)

---

## 二、迁移机制：语义与序列能力最值得借鉴

> **本节要点**：迁移优先级应由“信号增量与可控性”决定。语义表征和时间序列最适合先进入生产实验，检索增强适合证据与调查场景，开放式生成仅承担解释和辅助。

### 2.1 可迁移的四类机制

> **图示说明（原文为交互式 HTML5 图块，未转换为图片）**：LLM4Rec 四类能力向风控迁移的分层全景：以语义理解为底层输入增强，逐层叠加序列建模、检索增强与受控决策辅助；越向上越接近高影响动作，也越需要校准、规则和人工复核。

*图 1：四类能力的迁移价值随在线决策权限上升而递减（来源：LLM4Rec 综述* [*[2]*](https://arxiv.org/abs/2307.02046v6)*、P4R* [*[5]*](https://arxiv.org/abs/2409.16674v3)*、FraudTransformer* [*[9]*](https://arxiv.org/abs/2509.23712v2)*）*

**语义表征与时间序列是两条首选路径。**商户描述、申诉文本、设备说明、规则文本和内容素材可由语言或多模态编码器转成语义特征，再与金额、频次、设备关系和历史风险等结构化特征融合。贷款研究中，BERT 文本风险分数与 XGBoost 融合后，AUC 从 0.6575 升至 0.6644；增益有限但经论文报告具有统计显著性，符合“增强成熟模型”的预期。[[7]](https://arxiv.org/abs/2401.16458v3)

时间序列迁移需要保留事件间隔。FraudTransformer 同时加入绝对时间或事件间隔编码与位置编码，消融结果显示移除任一部分都会弱于完整模型。[[9]](https://arxiv.org/abs/2509.23712v2) 这比单纯把交易按顺序输入 Transformer 更贴合风控：同样的金额与地点组合，在十分钟内密集发生和在半年内分散发生，风险含义不同。

检索增强适合把规则、历史案件、实体关系和政策版本作为外部证据。它允许知识独立更新与撤回，但必须限制候选范围，并校验规则或案件 ID、时效、权限与证据一致性。生成式能力则宜输出结构化理由、案件摘要或处置建议，最终分数和动作继续交给可校准模型、确定性规则与人工复核。

### 2.2 从用户兴趣到风险意图的映射

| 建模维度 | 推荐系统 | 风控系统 | 迁移结论 |
|-|-|-|-|
| 基本单元 | 用户—物品—交互 | 实体/账号—事件/设备—动作 | 事件 token 化可直接借鉴 |
| 学习目标 | 相关性、排序与长期效用 | 预期损失、风险排序与处置约束 | 共享表示层，重建损失函数 |
| 标签 | 点击、停留、购买等稠密弱反馈 | 确认欺诈、拒付、申诉与未决案件 | 未标注样本不能视作负例 |
| 时间 | 偏好漂移与下一物品预测 | 攻击节奏、延迟标签与概念漂移 | 加入时间间隔并做时间外评估 |
| 关系图 | 常假设相邻实体相似 | 欺诈节点会连接正常节点伪装 | 必须建模异配与关系不一致 |
| 输出 | 候选分数、Top-K 或解释 | 风险概率、原因码与处置建议 | 分数需校准，动作与生成分离 |
| 探索成本 | 少量低相关曝光通常可逆 | 误放和误杀可能产生高损失 | 采用回放、影子与可回滚灰度 |

*表 1：推荐与风控共享建模骨架，差异集中在监督语义与决策代价（来源：推荐多目标综述* [*[1]*](https://arxiv.org/abs/2210.10309)*、异常检测综述* [*[10]*](https://arxiv.org/abs/2202.07787v1)*）*

图模型尤其需要重新审视归纳偏置。YelpChi 多关系图中，部分关系下邻居同标签比例不足 10%；GraphConsis 通过上下文嵌入、相似邻居采样和关系注意力处理伪装，在其 80% 训练设置下取得 AUC 0.7428，而普通 GraphSAGE 为 0.5169。[[8]](https://arxiv.org/abs/2005.00625v3) 风控图应把“不一致”视为信号，不能默认相邻即相似。

多模态和通用 LLM 的角色也应收窄。CALM 覆盖 9 个信用与风险数据集，但 GPT-4 在 Credit Card Fraud 上 MCC 为 0.331，在 ccFraud 上仅 0.113；同一模型跨数据集波动很大，且部分开源模型会退化为近乎全预测同一类别。[[11]](https://arxiv.org/abs/2310.00566v3) 这类能力适合冷启动、语义取证和分析员辅助，尚不足以证明其可直接替换成熟风控模型。

---

## 三、目标架构：大模型离线增强，小模型在线决策

> **本节要点**：推荐系统中的“多阶段”思想应保留。教师模型、检索系统、在线评分器和策略引擎分别承担知识生产、证据获取、风险估计与动作控制，避免一个生成模型同时拥有推理与执行权限。

### 3.1 离线教师与知识生产链

> **图示说明（原文为交互式 HTML5 图块，未转换为图片）**：风控离线教师与知识生产闭环：原始行为与文本经治理后交给大模型教师生成特征、标签和理由，经过规则与人工质检蒸馏到在线小模型，再由线上反馈和漂移监控驱动下一轮数据与模型更新。

*图 2：大模型在离线侧生产标签、特征与解释，在线侧由小模型和规则闭环（来源：Google 蒸馏方法* [*[3]*](https://research.google/blog/distilling-step-by-step-outperforming-larger-language-models-with-less-training-data-and-smaller-model-sizes/)*、OWASP 数据治理建议* [*[12]*](https://github.com/GenAI-Security-Project/GenAI-LLM-Top10/blob/main/2026/final/LLM05_DataModelPoisoning.md)*）*

离线教师最适合处理难以直接结构化的材料：案件描述、申诉记录、规则文本、设备说明、商户资料和内容证据。教师可生成候选标签、语义向量、原因码和难例说明；进入学生训练前，必须经过规则一致性、人工抽检、多次生成一致性与时间可用性过滤。Google 的 Distilling step-by-step 证明，自然语言理由与标签联合监督可显著压缩学生模型，但其结果来自四个自然语言处理基准，只能支持“专项蒸馏值得试验”，不能预设风控收益。[[3]](https://research.google/blog/distilling-step-by-step-outperforming-larger-language-models-with-less-training-data-and-smaller-model-sizes/)

数据链应保留来源、授权状态、事件时间、教师版本、提示词版本、过滤结果与学生训练批次。原始敏感文本进入教师前做最小化与去标识；蒸馏样本、向量、检查点和适配器也进入删除与审计范围。OWASP 指出，敏感样本可能通过蒸馏转移到派生模型，向量还存在反演与跨租户泄露风险。[[13]](https://github.com/GenAI-Security-Project/GenAI-LLM-Top10/blob/main/2026/final/LLM02_SensitiveInformationDisclosure.md) [[14]](https://github.com/GenAI-Security-Project/GenAI-LLM-Top10/blob/main/2026/final/LLM09_VectorAndEmbeddingWeaknesses.md)

### 3.2 在线召回、排序与校准链

![在线风控主干网络结构示意图](diagrams/2026-09-11T122456/diagram.png)

*图 3：在线风控以轻量多任务网络完成实时评分，LLM 仅在离线侧提供语义特征、弱标签与蒸馏监督；策略引擎独立消费结构化输出。*

**在线主链建议保持确定性边界：**身份与权限校验后，规则、向量或图检索负责召回候选信号，时间感知序列模型与结构化模型完成评分，再经校准器把分数映射为风险概率或成本区间；策略引擎依据阈值、额度、司法辖区和人工复核规则决定动作。模型只输出结构化字段，不直接持有冻结、拒付或状态变更凭证。

RAG（检索增强生成）不宜默认进入每笔硬实时交易。它增加检索往返、向量计算、重排和输入 token，延迟与成本取决于召回规模。[[15]](https://learn.microsoft.com/en-us/azure/foundry/concepts/retrieval-augmented-generation) 更稳妥的分工是：实时链检索短小、闭集的规则或相似案例特征；案件调查、申诉分析和规则解释可调用更完整的 RAG。跨实体、多跳和全局关系问题才启用图谱检索，局部事实优先向量或混合检索。Microsoft 对向量 RAG 与 GraphRAG 的公开比较也支持这一边界。[[16]](https://www.microsoft.com/en-us/research/blog/lazygraphrag-setting-a-new-standard-for-quality-and-cost/)

观测面至少记录数据、模型、索引、提示词和阈值版本，以及检索 ID、分阶段时延、分数、校准结果和最终动作。原始提示词、检索片段与推理轨迹默认脱敏或不落盘。发布面同时保留上一版学生模型、上一版索引、静态保守规则和人工兜底，允许先禁用图谱或大模型升级，再回退整版。

---

## 四、方案比较：混合架构优于端到端大模型

> **本节要点**：选型应先过合规、数据驻留、尾延迟和可回滚四项硬门槛。通过后再比较质量与成本；按这一顺序，混合架构是默认路线，纯蒸馏适合稳定闭集任务，在线大模型级联只适合作为受控兜底。

### 4.1 三类候选路线及适用边界

| 路线 | 质量与新鲜度 | 延迟与成本 | 治理与主要风险 | 适用判断 |
|-|-|-|-|-|
| 现有基线 | 结构化信号成熟；语义与长序列覆盖有限 | 最可预测；已有容量经验 | 解释、审计与回滚机制通常最完整 | 所有新路线必须相对它证明增益 |
| A 纯离线蒸馏 | 窄域稳定任务较强；知识刷新慢 | 一次小模型调用，尾延迟最可控 | 敏感知识可能固化进权重，删除成本高 | 稳定闭集、硬实时、本地部署 |
| B 混合架构 | 结构化模型保底；语义与检索提供增量 | 增加检索开销，但可按场景关闭 | 需管理权限、索引、引用与数据删除 | **默认推荐**：语义教师、案件辅助与规则检索 |
| C 在线级联 | 困难样本可升级；依赖路由校准 | 平均成本可降，升级请求尾延迟最高 | 多模型、多供应商扩大攻击与审计面 | 非硬实时、长尾难度大且允许人工确认 |
| 端到端大模型 | 语义强；概率校准与稳定性证据不足 | 成本、吞吐和尾延迟不确定 | 生成错误可能直接转化为高影响动作 | **暂不进入主决策链** |

*表 2：路线 B 兼顾增量与可控性，路线 A、C 分别承担快速路径和质量兜底（来源：蒸馏* [*[3]*](https://research.google/blog/distilling-step-by-step-outperforming-larger-language-models-with-less-training-data-and-smaller-model-sizes/)*、RAG* [*[17]*](https://learn.microsoft.com/en-us/azure/architecture/ai-ml/guide/rag/rag-solution-design-and-evaluation-guide)*、级联* [*[18]*](https://arxiv.org/abs/2305.05176v1)*）*

这里的“默认推荐”有明确条件：结构化基线继续保留，RAG 和大模型均可独立降级，执行动作由策略引擎控制，敏感数据不跨越既定边界。若任一条件不满足，路线 B 应退回离线特征实验，而非带着治理缺口进入线上。

### 4.2 延迟、成本与治理权衡

公开数字只能给出可行性上界。Google 报告 770M T5 在 ANLI 上使用全量数据的 80% 即超过 few-shot 540B PaLM，显示专项蒸馏可大幅减小模型体量。[[3]](https://research.google/blog/distilling-step-by-step-outperforming-larger-language-models-with-less-training-data-and-smaller-model-sizes/) FrugalGPT 在特定学术数据集与 2023 年 API 价格下，匹配质量时最高节省 98% 推理成本；串行级联的升级请求仍会累加小模型、评分器和大模型时延。[[18]](https://arxiv.org/abs/2305.05176v1) 2026 年 UCCI 在 75,000 条生产命名实体识别查询上报告成本下降 31%，ECE（期望校准误差）从 0.12 降至 0.03，但该结果来自单一任务与硬件环境。[[19]](https://arxiv.org/abs/2605.18796v1)

因此，成本指标应计算“每个正确且可接受决策”的端到端成本，纳入教师标注、训练摊销、向量更新、检索重排、升级比例和人工审核。延迟则分别记录规则、召回、重排、学生、路由与教师阶段的 P50/P95/P99，不能用平均值掩盖升级请求的尾部抖动。

**三项最强反对意见需要在立项前得到回应：**

1. 成熟 GBDT、图模型和规则系统已能覆盖大部分结构化风险，新增 LLM 可能只带来微小 AUC 增益。回应方式是把试验限定在文本、多模态、冷启动与长序列等现有基线的已知薄弱切片，并要求增益覆盖新增运维成本。
2. RAG 会引入提示注入、越权检索和索引中毒。回应方式是检索前执行文档或块级权限校验，把检索内容视为不可信输入，并由确定性策略校验输出。[[20]](https://github.com/GenAI-Security-Project/GenAI-LLM-Top10/blob/main/2026/final/LLM01_PromptInjection.md)
3. 离线蒸馏仍可能复制教师错误和敏感信息。回应方式是使用人工黄金集、教师分歧与错误切片筛选训练样本，并对数据、适配器和检查点建立版本、签名与可验证删除。

这些反对意见不会推翻混合路线，却限定了它的适用范围：LLM 必须证明对特定缺口有增量，并接受与现有模型相同甚至更严格的审计和回滚要求。

---

## 五、验证路径：先证明增益，再扩大决策权限

> **本节要点**：PoC 的成功标准应同时覆盖业务损失、概率校准、尾延迟、成本、检索忠实度和安全后果。实验从时间外离线评估开始，经历史回放、影子流量与小范围灰度逐级推进。

### 5.1 PoC 指标与实验设计

建议同时保留六组对照：现有生产基线、时间感知序列模型、基线加 LLM 语义特征、离线蒸馏学生、混合架构，以及仅用于能力上界的教师模型。所有组统一数据时间窗、实体隔离、硬件、并发、缓存和输入字段；教师不能读取目标时点之后形成的案件结论。

| 阶段 | 核心问题 | 主要指标 | 建议门槛 | 失败处置 |
|-|-|-|-|-|
| 离线留出 | 新机制是否提供真实增量 | 固定误报率召回、PRAUC、MCC、预期损失、关键切片最差值 | 高风险切片无显著劣化；95% CI 下界不低于基线 -2 个百分点 | 停止上线，回到特征与标签设计 |
| 校准评估 | 分数能否映射动作成本 | ECE、Brier、风险—覆盖曲线、阈值稳定性 | ECE≤0.05；关键切片单独达标 | 重校准或降为排序信号 |
| 历史回放 | 是否存在未来泄漏与容量问题 | 时间外损失、标签成熟度、吞吐、阶段 P95/P99 | 满足现行业务 SLO；无未来信息 | 缩短链路或转为离线特征 |
| 影子流量 | 线上分布与离线是否一致 | 分布漂移、学生—教师分歧、升级率、检索覆盖 | 稳定跨过完整标签延迟周期 | 冻结扩量，重新采样校准 |
| 小流量灰度 | 业务收益能否覆盖成本 | 净损失、误杀申诉、审核量、单位正确决策成本 | 质量达标后，成本相对教师方案下降至少 30% | 切回基线或禁用升级路径 |
| 安全评审 | 攻击能否产生外部副作用 | 注入成功率、越权召回、跨租户命中、敏感信息泄露 | 严重越权、泄露与未授权动作均为 0 | 立即回滚并保全审计证据 |

*表 3：PoC 必须把效果、校准、工程与安全放在同一门禁中（来源：RAG 评估指南* [*[17]*](https://learn.microsoft.com/en-us/azure/architecture/ai-ml/guide/rag/rag-solution-design-and-evaluation-guide)*、RAGAS* [*[21]*](https://arxiv.org/abs/2309.15217v2)*、UCCI* [*[19]*](https://arxiv.org/abs/2605.18796v1)*）*

表中的 -2 个百分点、ECE 0.05 和成本下降 30% 是项目初始假设，不是行业标准。最终阈值要按事件率、漏放与误杀成本、业务服务等级目标及统计功效重新计算。在线 A/B 在风控场景中也不应直接复制推荐探索；优先采用历史回放、影子和冠军—挑战者机制，高影响动作保留人工确认。

### 5.2 对抗、漂移与合规风险

#### 💡 对抗与中毒

> - 检索文档可携带间接提示注入
> - 正常邻居可能稀释欺诈信号
> - 规则、索引和模型需独立回滚

#### 📉 标签与漂移

> - 监测标签成熟度与开放类别
> - 按时间窗跟踪 ECE 与升级率
> - 新设备、商户和攻击手法单列切片

#### 🔐 隐私与合规

> - 检索前执行租户与块级授权
> - 限制提示词、日志与缓存字段
> - 删除覆盖文档、向量与派生模型

开放类别、延迟标签与概念漂移是风控序列模型的常态，OpenDrift 的公开研究即围绕在线支付欺诈中的三类问题展开。[[22]](https://ieeexplore.ieee.org/abstract/document/10248325/) 监控不能只看输入统计量，还要跟踪固定误报率召回、人工失败率、学生—教师分歧、校准误差、升级率和检索无答案率。触发回滚的条件应与发布门槛对称。

治理可采用 NIST AI RMF 的全生命周期框架组织风险登记、评测证据与发布审批。[[23]](https://www.nist.gov/publications/artificial-intelligence-risk-management-framework-generative-artificial-intelligence) 若处理欧盟个人数据或进入适用的高风险系统范围，还需逐项评估 GDPR 的数据最小化、默认保护和安全措施，以及 EU AI Act 的风险管理、日志、人类监督、准确性、稳健性与网络安全要求。[[24]](https://eur-lex.europa.eu/eli/reg/2016/679/oj) [[25]](https://eur-lex.europa.eu/eli/2024/1689/oj) 具体适用性须由法务与数据保护负责人确认。

---

## 六、综合判断：应从语义教师与序列模型切入

> 📌 **综合判断**：LLM for Recommendation 向风控迁移具有工程可行性，但第一阶段只应迁移表征、时序与检索机制。
>
> 推荐的语义—行为融合证据，加上欺诈序列和关系模型的直接结果，说明大模型能够补足非结构化信息与长程行为信号；标签稀缺、代价不对称、概率校准和主动对抗又决定了在线动作仍需由小模型、规则与人工闭环控制。
>
> 残余风险 / 反向信号：
>
> 1. 现有公开研究多为离线、自报或单机构结果，跨业务增益和长期稳定性尚未得到充分验证。
> 2. RAG、蒸馏与在线级联分别新增索引、派生模型和路由器三类治理对象，运维复杂度可能抵消模型收益。

**建议从两个独立 PoC 启动。**其一，使用大模型离线提取案件、申诉、商户与规则文本特征，验证其对现有 GBDT、图模型或序列模型的边际增益；其二，在不改变现网动作策略的前提下，引入时间间隔编码和长序列预训练，验证固定误报率下的召回与校准。两个 PoC 均可逆，且能分别回答“语义是否有增量”和“序列机制是否有增量”。

进入第二阶段的条件是：时间外高风险切片非劣、校准与尾延迟达标、严重安全后果为零、数据删除与回滚演练通过。此后可把 RAG 扩展到案件调查、规则解释和申诉辅助；只有在这些条件持续成立，且人工复核证明结构化输出可靠时，才评估更高权限的生成式决策。

会改变本结论的证据也很明确：如果现有结构化基线在目标薄弱切片上已经充分覆盖，LLM 特征未产生净损失改善，或治理成本超过节省的审核与损失成本，则应停止扩展；如果端到端模型在时间外、对抗和跨机构测试中持续满足校准、SLO 与审计门槛，才有理由重新评估其在线角色。

---

## 待验证事项

> 🔍 **待验证与口径说明**（以下事项影响方案参数，不改变当前的分阶段建议）：
>
> - 目标业务的现网 QPS、P95/P99 服务等级目标、单案审核成本、漏放与误杀成本矩阵尚未提供；上线阈值需由业务、风控和基础设施团队共同校准。
> - 交易、账号与内容风控的标签成熟周期、实体隔离规则及可用文本字段不同；本报告未给出单一通用 token schema。
> - FraudTransformer 等工业数据不公开，跨机构泛化、阈值稳定性与线上成本仍待独立复现。[[9]](https://arxiv.org/abs/2509.23712v2)
> - RecoAtlas、CRAB 与 UCCI 属较新预印本，其工业可复现性、代码完整度和独立实验仍待验证。[[26]](https://arxiv.org/abs/2605.18805v1)[[27]](https://arxiv.org/abs/2604.05113v1)[[19]](https://arxiv.org/abs/2605.18796v1)
> - 数据地域、敏感个人信息类型和 EU AI Act 高风险分类未确定，GDPR 与行业监管的适用范围需法务或数据保护负责人确认。
> - 本轮为优先回答机制与可行性，未展开具体云厂商、推理芯片和向量数据库选型，也未估算绝对样本量、灰度比例与删除时限；这些参数应在取得现网基线后补充。

---

## 参考文献

- [1] Ekstrand 等, Recommender Systems Through the Lens of Decision Theory, 2022-10-19. https://arxiv.org/abs/2210.10309
- [2] Fan 等, Recommender Systems in the Era of Large Language Models, 2024-04-29. https://arxiv.org/abs/2307.02046v6
- [3] Google Research, Distilling step-by-step: Outperforming larger language models with less training data and smaller model sizes, 2023-09-21. https://research.google/blog/distilling-step-by-step-outperforming-larger-language-models-with-less-training-data-and-smaller-model-sizes/
- [4] Liu 等, Pre-train, Prompt and Recommendation: A Comprehensive Survey of Language Modelling Paradigm Adaptations in Recommender Systems, 2023-09-12. https://arxiv.org/abs/2302.03735v3
- [5] P4R, A Prompting-Based Representation Learning Method for Recommendation with Large Language Models, 2024-10-01. https://arxiv.org/abs/2409.16674v3
- [6] CMT, Contrastive Multi-view Transformer for Fraud Detection, 2024-04-04. https://arxiv.org/abs/2308.02793v2
- [7] Zhang 等, Enhancing Credit Risk Assessment with BERT Risk Scores and XGBoost, 2025-03-23. https://arxiv.org/abs/2401.16458v3
- [8] Liu 等, Alleviating the Inconsistency Problem of Applying Graph Neural Network to Fraud Detection, 2020-07-02. https://arxiv.org/abs/2005.00625v3
- [9] FraudTransformer, Time-Aware GPT for Transaction Fraud Detection, 2025-10-28. https://arxiv.org/abs/2509.23712v2
- [10] Pang 等, Deep Learning for Anomaly Detection: A Review, 2022-02-15. https://arxiv.org/abs/2202.07787v1
- [11] CALM, A Multi-task Benchmark for Comprehensive Assessment of Language Models in Financial Risk, 2024-02-18. https://arxiv.org/abs/2310.00566v3
- [12] OWASP GenAI Security Project, Data and Model Poisoning, 2026-08-04. https://github.com/GenAI-Security-Project/GenAI-LLM-Top10/blob/main/2026/final/LLM05_DataModelPoisoning.md
- [13] OWASP GenAI Security Project, Sensitive Information Disclosure, 2026-08-04. https://github.com/GenAI-Security-Project/GenAI-LLM-Top10/blob/main/2026/final/LLM02_SensitiveInformationDisclosure.md
- [14] OWASP GenAI Security Project, Vector and Embedding Weaknesses, 2026-08-04. https://github.com/GenAI-Security-Project/GenAI-LLM-Top10/blob/main/2026/final/LLM09_VectorAndEmbeddingWeaknesses.md
- [15] Microsoft Azure, Retrieval-augmented generation concepts, 2026-08-21. https://learn.microsoft.com/en-us/azure/foundry/concepts/retrieval-augmented-generation
- [16] Microsoft Research, LazyGraphRAG: Setting a new standard for quality and cost, 2024-11-25. https://www.microsoft.com/en-us/research/blog/lazygraphrag-setting-a-new-standard-for-quality-and-cost/
- [17] Microsoft Azure Architecture Center, RAG solution design and evaluation guide, 2026-07-01. https://learn.microsoft.com/en-us/azure/architecture/ai-ml/guide/rag/rag-solution-design-and-evaluation-guide
- [18] Chen 等, FrugalGPT: How to Use Large Language Models While Reducing Cost and Improving Performance, 2023-05-09. https://arxiv.org/abs/2305.05176v1
- [19] UCCI, Uncertainty-Calibrated Cascaded Inference, 2026-05-11. https://arxiv.org/abs/2605.18796v1
- [20] OWASP GenAI Security Project, Prompt Injection, 2026-08-04. https://github.com/GenAI-Security-Project/GenAI-LLM-Top10/blob/main/2026/final/LLM01_PromptInjection.md
- [21] Es 等, RAGAS: Automated Evaluation of Retrieval Augmented Generation, 2025-04-28. https://arxiv.org/abs/2309.15217v2
- [22] OpenDrift, Open-set Concept Drift Adaptation for Online Payment Fraud Detection, 2023. https://ieeexplore.ieee.org/abstract/document/10248325/
- [23] NIST, Artificial Intelligence Risk Management Framework: Generative Artificial Intelligence Profile, 2024-07-26. https://www.nist.gov/publications/artificial-intelligence-risk-management-framework-generative-artificial-intelligence
- [24] European Union, General Data Protection Regulation, 2016-04-27. https://eur-lex.europa.eu/eli/reg/2016/679/oj
- [25] European Union, Artificial Intelligence Act, 2024-06-13. https://eur-lex.europa.eu/eli/2024/1689/oj
- [26] RecoAtlas, From Semantic Plausibility to Set-Level Utility in LLM Recommendation Agents, 2026-05-11. https://arxiv.org/abs/2605.18805v1
- [27] CRAB, Codebook Rebalancing for Bias Mitigation in Generative Recommendation, 2026-04-06. https://arxiv.org/abs/2604.05113v1
- [28] Huang 等, Graph Neural Networks for Financial Anti-Money Laundering, 2023-07-25. https://arxiv.org/abs/2307.13499v1
