# LLM For Recommendation：技术范式、证据边界与落地路径

> 🎯 **核心判断**
>
> 1. **定义边界决定技术判断**：LLM 可作为语义特征、推荐决策器或交互编排器；生成式推荐、推荐基础模型与推荐智能体彼此交叉，却对应不同的系统责任。
> 2. **核心收益来自语义迁移与任务统一**：冷启动、复杂条件理解、自然语言交互和解释生成已显示新增价值；协同关系、时序变化和目录约束仍需专门模块。
> 3. **实证证据支持局部增益，尚不支持普遍替代**：微调或融合协同信号的方案通常更有竞争力；纯零样本 LLM 在传统准确率任务上不稳定，结果高度依赖候选集、负采样、污染与提示协议。
> 4. **工程边界决定价值能否兑现**：生产收益依赖少算、复用、异步、压缩以及确定性控制面，模型效果必须与尾时延、总成本、合法性和长期反馈共同验收。
> 5. **落地应从增强式改造开始**：先部署离线语义增强、教师蒸馏或受限候选重排，再按已验证的业务瓶颈进入混合式，最后才考虑生成式重构。

研究口径：仅使用公开信源，可外发；资料截至 2026-09-10。本文将“LLM For Recommendation”简称为 LLM4Rec，即大语言模型用于推荐。论文数字均保留数据集、任务、候选协议和作者自报边界，不把单一离线结果或特定业务 A/B 测试外推为普遍结论。

## 一、定义边界：LLM 改造推荐包含三类角色

### 1.1 “LLM 用于推荐”的范围

广义 LLM4Rec 指将经大规模语料预训练的 Transformer 语言模型引入推荐系统，利用文本语义、外部知识、迁移和生成能力改善用户—物品建模。按模型角色，可归纳为“LLM 表征 + 推荐系统”“LLM 生成中间 token + 推荐系统”“LLM 直接承担推荐任务”三类[[1]](https://arxiv.org/abs/2305.19860v5)。这一口径同时涵盖 BERT 类判别式模型与 GPT 类生成式模型，明显宽于“让聊天模型直接给出推荐”。

生成式推荐由输出机制界定：模型直接生成物品标识、有序列表或推荐相关内容，而非逐一计算每个候选的排序分数[[2]](https://arxiv.org/abs/2309.01157v2)。因此，专用序列模型也可属于生成式推荐；使用 LLM 生成画像或解释，也不代表系统已经成为生成式推荐器。

### 1.2 三类目标与两条系统边界

- **💡 增强语义**
  从标题、描述、评论和对话中提取特征，缓解数据稀疏与冷启动。
- **🔄 统一任务**
  把评分、序列推荐、解释与搜索改写为统一的序列学习问题。
- **💬 改善交互**
  以自然语言理解意图、澄清约束，并调用专业推荐工具。

第一条边界涉及模型外延。推荐基础模型还包括视觉、音频与多模态基础模型；仅在交互 ID 上扩展规模的 Large Recommendation Model，也不自动属于 LLM4Rec[[3]](https://arxiv.org/abs/2504.16420v1)。第二条边界涉及系统责任。对话推荐是交互形态，推荐智能体是编排角色。InteRecAgent 由 LLM 负责意图理解、规划和反思，信息查询、物品召回与排序仍由专业工具完成[[9]](https://arxiv.org/abs/2308.16505v3)。

---

## 二、演进脉络：研究重心转向统一生成与治理

> **本节要点**：研究路径从文本语义增强推进到任务统一、推荐对齐、语义物品标识和智能体编排；截至 2026 年，前沿问题已经转向码本质量、偏差、受约束生成与在线效率。

### 2.1 从语义编码到统一生成

> **原图说明：** LLM 推荐从文本语义增强、任务语言化，到语义标识生成、统一推荐基础模型，再到工程治理的关键转折时间轴。

*图 1：LLM 推荐从语义增强走向统一生成与治理（来源：*[*[4]*](https://arxiv.org/abs/2203.13366v7)*、*[*[7]*](https://arxiv.org/abs/2305.05065v3)*、*[*[3]*](https://arxiv.org/abs/2504.16420v1)*）*

P5 在 2022 年把评分预测、序列推荐、解释生成、评论摘要和直接推荐统一为文本到文本任务，证明异构推荐任务可以共享一个条件语言建模接口[[4]](https://arxiv.org/abs/2203.13366v7)。同期 M6-Rec 将工业召回、排序、解释、内容生产和对话统一为语言理解或生成任务，并通过轻量适配、后期交互、提前退出、剪枝与量化控制开销[[5]](https://arxiv.org/abs/2205.08084v2)。

2023 年形成两条关键分支。TALLRec 用推荐指令数据和 LoRA（低秩适配）弥合通用语言能力与推荐目标之间的差距[[6]](https://arxiv.org/abs/2305.00447v3)；TIGER 将物品编码为 Semantic ID（语义物品标识），把序列推荐改写为自回归生成与检索[[7]](https://arxiv.org/abs/2305.05065v3)。前者解决任务对齐，后者改变候选产生机制。

### 2.2 技术转折背后的驱动力

第一项驱动力是语义与协同信号互补。原子物品 ID 能高效表达站内交互，却缺少开放语义；文本模型拥有世界知识，却不天然理解本站用户行为。LLaRA 因而把顺序推荐器学习到的行为 embedding 作为独立模态，经投影器接入 LLM，并与文本 token 联合建模[[10]](https://arxiv.org/abs/2312.02445v4)。

第二项驱动力是减少多阶段目标错配。OneRec 将语义 ID、列表生成和迭代偏好优化合入单一系统[[13]](https://arxiv.org/abs/2502.18965v1)；SynerGen 在同一 decoder-only 主干中联合搜索、推荐、召回和排序[[17]](https://arxiv.org/abs/2509.21777v1)。这些系统仍保留任务头、掩码、奖励模型与约束，统一主干不意味着自由文本模型接管全部决策。

第三项驱动力来自治理压力。CRAB 报告语义码本中的高频 token 会继承并放大流行度偏差，并通过码本重平衡和层次语义正则进行纠偏[[14]](https://arxiv.org/abs/2604.05113v1)。研究问题由“能否生成”推进到“生成机制是否稳定、公平且可控制”。

---

## 三、技术全景：LLM 可进入链路但责任不同

> **本节要点**：LLM 的角色横跨数据、表征、召回、排序、生成与交互；部署位置越接近实时主链路，对时延、候选合法性和协同信号保真度的要求越高。

### 3.1 模型、数据与系统分层

> **原图说明：** LLM 介入推荐系统的数据知识、表征索引、召回排序、交互编排和治理评测五层全景。

*图 2：LLM 介入推荐系统的分层位置（来源：*[*[3]*](https://arxiv.org/abs/2504.16420v1)*、*[*[9]*](https://arxiv.org/abs/2308.16505v3)*）*

分层的价值在于隔离能力与责任。离线数据层可生成物品摘要、用户画像、标签或训练样本；表征层可将文本映射为连续向量；召回与排序层可生成 Semantic ID、重排候选或联合优化；交互层可负责对话、计划和工具调用。越靠近用户实时请求，越需要确定性校验、超时回退和结构化评分。

同一系统通常混合多个角色。BIGRec 先让 LLM 在语言空间生成候选，再将输出映射到真实物品空间，并注入流行度与协同过滤信息[[8]](https://arxiv.org/abs/2308.08434v2)。该设计把开放语义与可服务目录分开处理，可降低虚构物品直接进入结果集的风险。

### 3.2 三种主流系统范式

| 范式 | 主要作用 | 适用条件 | 主要边界 |
|-|-|-|-|
| 特征增强 | 生成语义特征、画像与训练数据 | 既有链路成熟；实时约束严格 | 语义信号可能与协同目标解耦 |
| 生成式推荐 | 生成物品标识、候选或有序列表 | 目录可稳定编码；可实施约束解码 | 码本碰撞、无效物品、解码延迟与偏差 |
| 智能体推荐 | 理解意图并编排查询、召回、排序工具 | 高价值交互入口；允许多轮状态管理 | 调用成本、错误传播、状态一致性与复现性 |

*表 1：三种系统范式的职责与边界（综合* [*[3]*](https://arxiv.org/abs/2504.16420v1) *与* [*[2]*](https://arxiv.org/abs/2309.01157v2)*）*

三条路线可以同时存在于同一架构。较稳健的生产分工是：LLM 在离线层或交互层发挥语义和自然语言优势，专用推荐模型承担大规模行为建模与低时延服务，约束层负责目录、库存、政策与安全。只有多阶段目标错配和维护复杂度已经成为主要瓶颈时，生成式统一才值得承担更高迁移成本。

---

## 四、核心机制：增益伴随新的信息损失

> **本节要点**：LLM4Rec 的增益主要来自开放语义迁移、跨任务共享和可生成物品索引；相应代价是协同信号稀释、离散化误差、前缀剪枝损失和反馈偏差。

### 4.1 语义理解与开放世界知识

文本语义最适合解决行为数据不足。P5 通过多任务个性化提示支持未见模板与新物品迁移，但其零样本设定仍包含已见用户，不能外推为全新用户、全新目录和任意任务均可靠[[4]](https://arxiv.org/abs/2203.13366v7)。TALLRec 在电影和图书偏好预测中发现，未经推荐对齐的通用 LLM 会拒答或偏向正例；少量推荐样本与参数高效微调能够改善这一问题[[6]](https://arxiv.org/abs/2305.00447v3)。

行为与语言双空间融合更符合推荐任务实际。LLaRA 将外部顺序推荐器的 item-ID embedding 投影到 LLM 输入空间，再与物品文本拼接；课程式训练先适配文本，再逐步引入行为 token[[10]](https://arxiv.org/abs/2312.02445v4)。整体看，开放世界知识补足“物品是什么”，而“用户在本站何时会选择它”仍依赖协同与时序信号。

### 4.2 物品离散化与序列生成

> **原图说明：** 从物品内容编码、残差量化生成语义物品标识，到行为序列训练、受约束解码和真实物品落库的生成式推荐链路。

*图 3：语义物品标识驱动的生成式推荐链路（来源：*[*[7]*](https://arxiv.org/abs/2305.05065v3)*、*[*[11]*](https://arxiv.org/abs/2311.09049v4)*）*

Semantic ID 把每个物品编码为多个离散 token。TIGER 先用内容编码器得到连续向量，再用残差量化变分自编码器逐层量化；相似物品可共享前缀，模型据此在物品间共享参数，并自回归预测下一物品的完整编码[[7]](https://arxiv.org/abs/2305.05065v3)。LC-Rec 使用均匀分布约束处理末层冲突，使索引兼顾共享与唯一性[[11]](https://arxiv.org/abs/2311.09049v4)。

这条路线把语义编码、唯一标识与搜索剪枝压在同一 token 序列上。2026 年的一项诊断在三个 Amazon 域、八种 Semantic ID 构造上发现，编码邻域平均只恢复原内容编码器前十邻居的 32.2%；最终 semantic token 后，held-out 目标仍属可行候选的比例为 29.9%[[15]](https://arxiv.org/abs/2607.24995v1)。这些数值受目录和实验设置限制，却揭示了稳定机制：早期 token 误差会通过前缀剪枝不可逆地淘汰候选。

理论分析进一步指出，token 接口若不足以恢复物品排序就会损失信息；若可以精确恢复，其最终排序表达力不超过直接 item-level scoring[[16]](https://arxiv.org/abs/2602.07847v2)。Semantic ID 更应在参数共享、索引压缩和冷启动迁移上证明价值，不能预设其必然提升排序上限。

### 4.3 提示、微调与偏好对齐

- **📝 提示与候选重排**
  适合低成本原型；分数校准、位置偏差与正例偏置需单独验证。
- **🔧 监督微调与 LoRA**
  以推荐指令对齐任务；收益取决于领域、标签与候选协议。
- **💡 偏好优化**
  可优化列表级目标；奖励模型、曝光偏差与自生成负例会进入闭环。

推荐中的“对齐”需要区分语义—协同对齐与人类偏好对齐。LC-Rec 的 alignment 指物品索引、语言语义和行为关系的联合学习[[11]](https://arxiv.org/abs/2311.09049v4)；OneRec 从模型生成列表中构造 chosen/rejected 对，以奖励模型配合迭代 DPO（直接偏好优化）训练 session 级列表[[13]](https://arxiv.org/abs/2502.18965v1)。线上一次曝光通常无法同时观察反事实结果，因此推荐偏好对齐必须审计奖励偏差、曝光偏差与自举漂移。

---

## 五、实证证据：语义优势未转化为普遍胜出

> **本节要点**：同协议证据显示，微调或融合协同信号的 LLM 更有竞争力，纯零样本方案在传统准确率上通常不稳定。候选池、负采样、数据污染、提示和模型版本足以改变结论；公开 benchmark 只能筛选方案，线上 A/B 才能确认特定业务价值。

### 5.1 公开结果支持哪些结论

| 证据 | 任务与协议 | 主要结果 | 可支持的判断 |
|-|-|-|-|
| P5 | Amazon/Yelp，多任务 text-to-text | 多项序列推荐与文本生成指标优于论文基线 | 统一预训练可共享任务知识；不等同于通用聊天 LLM 零样本领先 |
| LLMRec | Amazon Beauty，五类推荐任务 | 原始通用 LLM 在准确性任务上整体弱于专门模型，语言任务更有优势 | 语言生成质量与推荐准确率需要分轴评价 |
| LLMRank | MovieLens-1M/Amazon Games，1 正＋19 负重排 | 较大模型通常更强，但存在候选位置与流行度偏差 | 支持小候选池零样本重排，不支持全库召回结论 |
| Beyond Utility | 四个公开域，约千名用户、1 正＋19 负 | 传统模型在多数标准准确率比较中更强；冷启动和知识丰富域可能受益 | LLM 优势依赖场景，幻觉与偏差需同时测量 |
| RecBench | H&M、MIND、MicroLens 等，点式与列表式任务 | 部分设置准确率提高，但推理明显更慢；LLM 特征注入 DLRM 可取得更好效用—延迟折中 | 混合方案通常比 LLM-only 更符合生产约束 |
| RecBench+ | 约 3 万条复杂电影/图书查询 | 明确条件较容易；隐含或错误条件更难，用户历史有时会干扰严格条件 | 复杂条件理解是新增能力，但需与个性化冲突分开评估 |
| LEADRE | 腾讯展示广告，20% 流量线上 A/B | 视频号 GMV +1.57%，朋友圈 GMV +1.17%；召回结果作排序特征另报 +1.43% | 证明特定工业链路可产生业务增量；不能跨产品复制收益预期 |

*表 2：代表性评估显示“任务与协议决定结论”（来源：*[*[4]*](https://arxiv.org/abs/2203.13366)*、*[*[31]*](https://arxiv.org/abs/2308.12241)*、*[*[32]*](https://arxiv.org/abs/2305.08845)*、*[*[33]*](https://arxiv.org/abs/2411.00331)*、*[*[34]*](https://arxiv.org/abs/2503.05493)*、*[*[35]*](https://arxiv.org/abs/2503.09382)*、*[*[38]*](https://arxiv.org/abs/2411.13789)*）*

这组证据形成三层结论。其一，P5 证明统一推荐任务可行，LLMRec 则显示通用 LLM 的语言能力不会自动转化为传统推荐准确率[[31]](https://arxiv.org/abs/2308.12241)。其二，LLMRank 与 Beyond Utility 的 20 项候选池实验主要测量重排能力；结果不能解释为百万级目录上的全库检索[[32]](https://arxiv.org/abs/2305.08845)[[33]](https://arxiv.org/abs/2411.00331)。其三，RecBench 与 LEADRE 共同支持混合架构：让 LLM 提供语义表征或受控召回，传统模块保留效率与约束，往往比单独追求模型规模更现实[[34]](https://arxiv.org/abs/2503.05493)[[38]](https://arxiv.org/abs/2411.13789)。

AgentRecBench 在 Amazon、Goodreads 与 Yelp 的文本环境中评估记忆、工具和多轮行动，结构化工作流通常优于简单智能体；但每轮仍常使用 1 正＋19 负的 20 项候选，HR 不能完整反映行动成本、错误恢复和长期用户价值[[36]](https://arxiv.org/abs/2505.19623)。因此，智能体 benchmark 证明的是编排能力的局部增益，不是开放环境中的长期业务效果。

### 5.2 候选、污染与提示足以改变排名

候选集是实验变量。1 正＋19 负协议计算成本低，却把全库检索降成小候选池重排；均匀负采样还可能在高流行度不均衡数据上放大偏差。通用推荐评估研究表明，按流行度加权抽样可缓解部分问题，但全库评价也会把未观测项视为负例，没有单一方案对所有目标都无偏[[39]](https://arxiv.org/abs/2001.08959)。合理协议应双报全库与多种采样结果，并把上游候选生成器 recall 与 LLM 重排 NDCG 分开。

数据泄漏是一级风险。可控实验通过 LoRA 注入同域或域外交互数据，发现同域泄漏会抬高“见过”样本的成绩，域外污染则可能降低表现；融合协同过滤的架构比纯 LLM 推荐更稳健[[37]](https://arxiv.org/abs/2602.13626)。这意味着“世界知识带来冷启动收益”和“benchmark 已进入预训练数据”难以仅凭最终分数区分。最低要求是严格时间切分、固定目录快照、去重，并分别报告疑似 seen/unseen、热门/长尾与冷启动切片。

提示与模型版本也应作为超参数披露。任务描述、few-shot 样例、历史截断、候选顺序、温度、解析器和重试规则都会改变结果。LLMRank 已观察到候选位置偏差，并用随机排列、多次调用和聚合缓解[[32]](https://arxiv.org/abs/2305.08845)。报告单次最佳提示会把选择过程隐藏在结果之外；至少应使用三个语义等价模板、三至五个候选排列和多个随机种子，报告均值与置信区间。

### 5.3 评估应形成四层证据栈

| 评估层 | 核心指标 | 回答的问题 | 主要盲点 |
|-|-|-|-|
| 离线效用 | AUC/GAUC、HR/Recall、NDCG、MRR | 能否重现历史偏好与排序 | 受曝光、负采样和候选规模影响 |
| 生成可靠性 | 目录合法率、条件命中、事实一致性、幻觉率 | 输出是否真实、可服务且满足条件 | BLEU/ROUGE 仅反映词面重叠 |
| 鲁棒与治理 | 提示/顺序稳定性、长尾、覆盖、多样性、群体公平 | 效果是否随输入扰动或群体变化 | 指标定义依目录与业务目标而变 |
| 系统与在线 | p50/p95/p99、吞吐、成本、失败率、CTR/CVR/GMV/留存 | 增量是否在真实链路兑现 | 短周期 A/B 难识别长期反馈 |

*表 3：LLM4Rec 需要效用、生成、治理与系统联合评估（综合公开 benchmark 与评估研究）*

传统指标与生成指标不能互相替代。HR/NDCG/AUC 衡量离线行为重现；BLEU/ROUGE 主要衡量参考文本的词面重叠，不能证明事实正确、个性化或目录合法。复杂助手任务可进一步报告推荐项 Precision/Recall、条件匹配率和应拒绝时仍推荐的失败率[[35]](https://arxiv.org/abs/2503.09382)。

LLM-as-judge 可扩展解释相关性、条件满足和成对偏好的评估规模，但需要先在人类金标上校准。可迁移的方法包括盲化模型身份、随机左右位置、重复采样，报告 Gwet’s AC2、κ、Spearman 或 Kendall 一致性，并对多重比较做校正[[41]](https://arxiv.org/abs/2504.17202)。关键业务结论仍应保留人工评审和线上实验。

DaisyRec 2.0 对推荐研究的复核把数据版本、预处理、切分、负采样、基线、指标与调参识别为会改变排名的“hyper-factors”[[40]](https://arxiv.org/abs/2206.10848)。LLM4Rec 又增加了提示、候选顺序、模型快照和解析规则。由此看，最低可报告协议应同时披露任务估计目标、数据与时间切分、污染审计、候选生成全过程、预算公平基线、多维指标、统计区间和最终在线 A/B。

---

## 六、工程边界：成本与控制面决定可用性

> **本节要点**：高成本能力应离线化、异步化或缓存化；任何开放生成进入在线链路，都必须处于候选约束、事实校验、安全审核和确定性回退之内。

### 6.1 在线链路中的部署位置

> **原图说明：** 生产推荐系统从离线大模型语义增强，到在线候选生成、轻量排序、受控生成和校验回退的低风险部署链路。

*图 4：离线增强、候选生成、轻量排序与受控生成的部署链路（来源：*[*[23]*](https://arxiv.org/abs/2410.23180v1)*、*[*[27]*](https://arxiv.org/abs/2606.25496v1)*）*

部署应遵循“重能力远离同步请求、关键约束留在确定性链路”的原则。ReasoningRec 用大模型合成解释与画像，再监督小模型同时完成偏好预测和理由生成[[23]](https://arxiv.org/abs/2410.23180v1)；PreBERT 将预训练知识蒸馏到两层、隐藏维度 256 的 Mini-BERT，形成离线教师与在线学生的职责划分[[21]](https://arxiv.org/abs/2409.14810v1)。

开放文本输出尤其需要后验控制。VideolandGPT 即使被要求只从候选池推荐，关闭后处理护栏后，仍有超过 22% 的任务出现用户最终选中但平台不可用的标题[[22]](https://arxiv.org/abs/2309.03645v1)。提示约束不能替代目录校验。生产链路应依次执行受限候选或约束解码、事实与库存校验、安全审核、超时回退和日志审计。

### 6.2 成本、延迟与可靠性

| 约束 | 主要成因 | 工程手段 | 验收指标 |
|-|-|-|-|
| 计算与显存 | 大模型参数、长上下文、KV 缓存 | 冻结、蒸馏、低比特权重与 KV 量化 | GPU-hour、峰值显存、效果损失 |
| 尾时延与吞吐 | 自回归解码、长历史、重复编码 | 交叉注意力、请求级复用、用户 sketch 缓存 | p50/p95/p99、QPS、缓存命中率 |
| 新鲜度 | 画像、缓存、码本和目录更新不同步 | 版本化键、TTL、增量索引、双轨校验 | 陈旧率、更新时延、SID 迁移成功率 |
| 可用性 | 超时、无效结果、依赖服务故障 | 缓存结果、近邻回退、传统模型兜底 | 回退率、错误率、故障隔离时间 |

*表 4：工程约束需要效果、时延与总成本联合验收*

成本优化的共同结构是少算、复用、异步和压缩。抖音 10K 长序列方案通过单查询交叉注意力把序列复杂度从 O(L²) 降到 O(L)，并以请求级批处理复用用户侧编码；论文报告 L=2K 时端到端带宽最高降低 84%、吞吐提升 2.2 倍[[19]](https://arxiv.org/abs/2511.06077v3)。SequenceO1 将 100K 历史压缩为固定大小 sketch，并以 TTL 和版本化键管理缓存新鲜度；其作者报告抖音时长提升 1.4999%、点踩下降 6.9829%，但该预印本发布于本报告截止日前两天[[18]](https://arxiv.org/abs/2609.08443v1)。

量化必须针对真实瓶颈。WKVQuant 指出自回归解码通常受权重和 KV cache 的内存访问限制；在论文设置中，LLaMA-2-13B、batch 16、长度 2048 的解码内存由 FP16 的 52.2GB 降至 W4KV4 的 13.1GB[[20]](https://arxiv.org/abs/2402.12065v2)。这属于通用 LLM 的系统证据，迁移到推荐仍需重新测量端到端时延、吞吐与业务指标。

### 6.3 隐私、公平与解释风险

隐私风险来自画像、对话和反馈日志。用户选择与反馈可能泄露敏感信息；数据最小化、加密、差分隐私和保留期限是基本控制，但保护强度也会改变可观测用户模型并影响协同效果[[24]](https://arxiv.org/abs/2109.06473v1)。验收时应同时观察泄露风险、覆盖与效果损失以及跨用户外部性。

公平需要先明确对象和定义。等曝光与按相关性获得等机会对应不同业务目标，Gini、Entropy、Jain、FSat 等指标各有适用条件；例如总推荐槽位小于物品数时，FSat 不应直接使用[[25]](https://arxiv.org/abs/2311.01013v1)。LangFair 可在业务 prompt 与推荐输出上评估反事实相似性、毒性、刻板印象和分配伤害，但阈值仍需按行业定义[[26]](https://arxiv.org/abs/2501.03112v1)。

解释的可读性与忠实度必须分开。ReasoningRec 将合成理由定位为用户真实动机的近似，并承认候选检索和重排影响仍是开放问题[[23]](https://arxiv.org/abs/2410.23180v1)。因此，解释适合作为受证据约束的辅助信息，不能作为排序模型真实因果依据的证明。

---

## 七、落地路径：先增强，再按瓶颈推进

> **本节要点**：路线选择应以可逆性和硬门槛为先。增强式改造快速验证语义价值；混合式解决明确瓶颈；生成式重构仅在索引、合法性、缓存、安全和回退均闭环后推进。

### 7.1 三阶段实施路径

- **💡 阶段一｜增强式**
  - 离线画像与标签
  - 教师蒸馏与冻结特征
  - 候选内重排、解释与对话
- **💡 阶段二｜混合式**
  - 语义与协同双路融合
  - 替换单一召回或排序模块
  - 联合训练、缓存与长序列
- **💡 阶段三｜生成式重构**
  - 语义 ID 与版本治理
  - 约束解码与目录校验
  - 近线内容生成与回退

阶段一验证语义价值，同时让传统系统保留候选和业务规则的控制权。推荐顺序为：离线批量生成画像、标签或教师软目标；训练轻量学生或冻结语义特征；进入影子流量；再试小流量候选重排、解释和对话。若蒸馏后增益无法保留，或 p99 与单位请求成本超预算，LLM 应继续停留在离线链路。

阶段二只替换已经明确的瓶颈。可先将冻结语义 embedding 与协同特征融合，再替换单一召回或排序模块，最后评估联合训练与长序列缓存。SynerGen 的消融显示，去除协同 embedding 后工业数据集搜索 R@300 从 72.96 降至 61.59[[17]](https://arxiv.org/abs/2509.21777v1)，语言语义无法无条件替代行为信号。

阶段三需要最高证据门槛。语义 ID 的碰撞、利用率、版本迁移和增量更新必须先受控；随后增加约束解码与候选合法性校验，再部署近线内容池、缓存与近邻回退。LLMTreeRec 通过类目树分层召回，在其设置中减少 85% 输入 token，并报告华为线上 A/B 测试[[28]](https://arxiv.org/abs/2404.00702v3)。结构化缩减候选空间应成为大目录生成式方案的前置条件。

### 7.2 场景化选型与验收指标

| 场景 | 优先路线 | 必须通过的门槛 | 核心验收 |
|-|-|-|-|
| 首页、广告等高 QPS 主链路 | 增强式 | 效果经蒸馏或冻结后保留；具备传统模型回退 | AUC/UAUC、p99、QPS、总成本 |
| 搜索、精选列表、对话入口 | 增强式或混合式 | 候选受限；事实与库存校验；故障隔离 | NDCG/MRR、越界率、满意度、回退率 |
| 冷启动与内容丰富目录 | 混合式 | 语义与协同消融能解释收益；目录更新可控 | 冷物品召回、覆盖率、码本稳定性 |
| 可缓存的个性化内容供给 | 生成式重构 | 近线生产、安全审核、缓存与近邻回退闭环 | 人工胜率、安全违规率、收入与生成成本 |

*表 5：选型应先通过硬门槛，再比较增量收益*

决策中必须保留“维持现状”作为基线。增强式改造可逆、迁移成本低，可以快速灰度；混合式需要逐层消融，确认收益来自语义、长序列或联合目标中的哪一项；生成式重构具有索引与数据契约迁移成本，应要求更长观察期和完整回滚方案。RaG 将实时兴趣推理、近线视频生成和实时服务解耦，在缓存未命中时使用内容相近结果回退[[27]](https://arxiv.org/abs/2606.25496v1)，体现了慢生成能力进入生产系统的合理边界。

---

## 八、综合判断：短期增量改造优于全链路替代

> 📌 **综合判断**：截至 2026-09-10，LLM4Rec 已具备可验证的局部与系统价值；最稳妥的产业路径仍是增强式优先、混合式按瓶颈推进、生成式有条件重构。
>
> 这一判断由四条证据链共同推出：语义与少样本研究证明 LLM 能补足冷启动和自然语言条件理解；同协议 benchmark 显示纯零样本 LLM 未形成传统准确率上的普遍优势；Semantic ID 与统一模型证明推荐可被重新表述为生成问题；工业案例同时表明，真实收益依赖长序列结构、协同特征、缓存、约束校验、近线解耦和回退。
>
> 残余风险与反向信号：
>
> 1. 公开工业结果多为作者自报，绝对时延、QPS、硬件规模、单位成本与独立复现披露不足。
> 2. 码本压缩、前缀剪枝、奖励模型与闭源模型污染会引入新的信息损失和反馈偏差。

高吞吐、强时延约束的首页与广告主链路，优先选择离线语义增强、蒸馏小模型和结构化长序列模型。中低吞吐的搜索、精选列表和对话入口，可试候选内 LLM 重排、解释与工具编排。只有当内容可缓存、目录可稳定编码、合法性与安全校验完备、增量收入覆盖生成成本时，才应推进内容或全链路生成式重构。

统一模型的反向证据值得重视：多阶段系统确有目标错配和维护冗余，OneRec、SynerGen 与 RaG 已展示统一路线的可行性。然而这些案例仍使用语义 ID、任务头、协同 embedding、缓存、奖励模型、约束和回退。统一表示与目标值得继续投资，工程控制面不会随模型统一而消失。

模型与硬件进步可能持续降低在线推理价格，但目录合法性、时序协同信号、隐私、公平与长期反馈属于独立问题。推荐策略转为生成式优先，需要三个可观测条件同时成立：同等总成本与 p99 下稳定超过现有基线；目录更新和受约束解码形成成熟工具链；跨业务、跨时间的独立线上复现显著增加。

---

## 待验证与研究缺口

> 🔍 **待验证与口径说明**（用于判断证据边界，不作精确承诺）：
>
> - LMRecSys 2021 的独立论文链接未在本轮信源中确认，演进脉络仅按生成式推荐综述的参考文献回溯[[2]](https://arxiv.org/abs/2309.01157v2)。
> - OneRec 的标题与 arXiv 标识已由后续文献交叉确认，其结果仍属于作者论文口径[[13]](https://arxiv.org/abs/2502.18965v1)。
> - 2026 年 Semantic ID 诊断结论来自当年预印本，32.2%、29.9% 等数值需在企业自有目录、编码器、beam size 与时间切分下复测[[15]](https://arxiv.org/abs/2607.24995v1)。
> - SequenceO1 于 2026-09-08 发布，距离本报告截止日仅两天，正式同行评议版本尚待核验[[18]](https://arxiv.org/abs/2609.08443v1)。
> - 通用 LLM 推荐的绝对 p50/p95/p99、QPS、GPU 配置与千次请求成本，以及多数工业结果的独立复现，公开材料仍不足。
> - 成员推断、属性推断、训练数据泄露和提示日志治理的量化证据，本轮受时间预算与公开实证可得性限制未展开，可作为后续专项研究。
> - 长期满意度、供给侧公平和反馈回路仍缺少跨平台、跨周期的统一评估；短期 CTR 或 GMV 增益不能替代长期生态验证。

---

## 参考文献

- [1] Fan 等, A Survey on Large Language Models for Recommendation, 2024-06-18. [https://arxiv.org/abs/2305.19860v5](https://arxiv.org/abs/2305.19860v5)
- [2] Lin 等, A Survey on Large Language Models for Generative Recommendation, 2024-03-23. [https://arxiv.org/abs/2309.01157v2](https://arxiv.org/abs/2309.01157v2)
- [3] Foundation Models for Recommender Systems: A Survey and New Perspectives, 2025-04-23. [https://arxiv.org/abs/2504.16420v1](https://arxiv.org/abs/2504.16420v1)
- [4] Geng 等, Recommendation as Language Processing (P5), 2022. [https://arxiv.org/abs/2203.13366](https://arxiv.org/abs/2203.13366)
- [5] Cui 等, M6-Rec, 2022-05-19. [https://arxiv.org/abs/2205.08084v2](https://arxiv.org/abs/2205.08084v2)
- [6] Bao 等, TALLRec, 2023-10-17. [https://arxiv.org/abs/2305.00447v3](https://arxiv.org/abs/2305.00447v3)
- [7] Rajput 等, Recommender Systems with Generative Retrieval (TIGER), 2023-11-03. [https://arxiv.org/abs/2305.05065v3](https://arxiv.org/abs/2305.05065v3)
- [8] Bao 等, BIGRec, 2023-12-31. [https://arxiv.org/abs/2308.08434v2](https://arxiv.org/abs/2308.08434v2)
- [9] Huang 等, InteRecAgent, 2024-01-30. [https://arxiv.org/abs/2308.16505v3](https://arxiv.org/abs/2308.16505v3)
- [10] LLaRA: Large Language-Recommendation Assistant, 2024-05-04. [https://arxiv.org/abs/2312.02445v4](https://arxiv.org/abs/2312.02445v4)
- [11] Zheng 等, LC-Rec, 2024-04-19. [https://arxiv.org/abs/2311.09049v4](https://arxiv.org/abs/2311.09049v4)
- [12] Zhai 等, Actions Speak Louder than Words: Trillion-Parameter Sequential Transducers for Generative Recommendations, 2024-05-06. [https://arxiv.org/abs/2402.17152v3](https://arxiv.org/abs/2402.17152v3)
- [13] OneRec: Unifying Retrieve and Rank with Generative Recommender, 2025-02-26. [https://arxiv.org/abs/2502.18965v1](https://arxiv.org/abs/2502.18965v1)
- [14] CRAB: Mitigating Popularity Bias in Generative Recommendation, 2026-04-06. [https://arxiv.org/abs/2604.05113v1](https://arxiv.org/abs/2604.05113v1)
- [15] Semantic-ID Generative Recommendation Diagnosis, 2026-07-27. [https://arxiv.org/abs/2607.24995v1](https://arxiv.org/abs/2607.24995v1)
- [16] On the Expressivity of Semantic-ID Generative Recommendation, 2026-08-16. [https://arxiv.org/abs/2602.07847v2](https://arxiv.org/abs/2602.07847v2)
- [17] SynerGen: Unified Generative Search and Recommendation, 2025-09-26. [https://arxiv.org/abs/2509.21777v1](https://arxiv.org/abs/2509.21777v1)
- [18] SequenceO1: Long-Sequence Modeling for Industrial Recommendation, 2026-09-08. [https://arxiv.org/abs/2609.08443v1](https://arxiv.org/abs/2609.08443v1)
- [19] Industrial 10K-History Recommendation with Target-to-History Attention, 2026-05-19. [https://arxiv.org/abs/2511.06077v3](https://arxiv.org/abs/2511.06077v3)
- [20] WKVQuant: Quantizing Weight and Key/Value Cache for Large Language Models, 2024-02-20. [https://arxiv.org/abs/2402.12065v2](https://arxiv.org/abs/2402.12065v2)
- [21] PreBERT and Mini-BERT for Sequential Recommendation, 2024-09-23. [https://arxiv.org/abs/2409.14810v1](https://arxiv.org/abs/2409.14810v1)
- [22] VideolandGPT: Recommender Systems with Large Language Models, 2023-09-07. [https://arxiv.org/abs/2309.03645v1](https://arxiv.org/abs/2309.03645v1)
- [23] ReasoningRec, 2024-10-30. [https://arxiv.org/abs/2410.23180v1](https://arxiv.org/abs/2410.23180v1)
- [24] Privacy-Aware Recommender Systems, 2021-09-14. [https://arxiv.org/abs/2109.06473v1](https://arxiv.org/abs/2109.06473v1)
- [25] Individual Item Fairness Measures for Recommender Systems, 2023-11-02. [https://arxiv.org/abs/2311.01013v1](https://arxiv.org/abs/2311.01013v1)
- [26] LangFair: A Python Package for Assessing Bias and Fairness in Large Language Model Use Cases, 2025-01-06. [https://arxiv.org/abs/2501.03112v1](https://arxiv.org/abs/2501.03112v1)
- [27] RaG: Recommendation as Generation for Personalized Video Generation, 2026-06-24. [https://arxiv.org/abs/2606.25496v1](https://arxiv.org/abs/2606.25496v1)
- [28] LLMTreeRec, 2024-12-24. [https://arxiv.org/abs/2404.00702v3](https://arxiv.org/abs/2404.00702v3)
- [29] Semantic-ID Generative Recommendation under Temporal Cold Start, 2026-07-23. [https://arxiv.org/abs/2607.21101v1](https://arxiv.org/abs/2607.21101v1)
- [30] Conversational Recommender System Breakdown and User Simulation, 2024-05-23. [https://arxiv.org/abs/2405.14249v1](https://arxiv.org/abs/2405.14249v1)
- [31] Liu 等, LLMRec: Benchmarking Large Language Models on Recommendation Task, 2023. [https://arxiv.org/abs/2308.12241](https://arxiv.org/abs/2308.12241)
- [32] Hou 等, Large Language Models are Zero-Shot Rankers for Recommender Systems, 2023. [https://arxiv.org/abs/2305.08845](https://arxiv.org/abs/2305.08845)
- [33] Dai 等, Beyond Utility: Evaluating LLM as Recommender, 2024. [https://arxiv.org/abs/2411.00331](https://arxiv.org/abs/2411.00331)
- [34] Benchmarking LLMs in Recommendation Tasks: A Comparative Evaluation with Conventional Recommenders, 2025. [https://arxiv.org/abs/2503.05493](https://arxiv.org/abs/2503.05493)
- [35] Towards Next-Generation Recommender Systems: A Benchmark for Personalized Recommendation Assistant with LLMs, 2025. [https://arxiv.org/abs/2503.09382](https://arxiv.org/abs/2503.09382)
- [36] AgentRecBench: Benchmarking LLM Agent-based Personalized Recommender Systems, 2025. [https://arxiv.org/abs/2505.19623](https://arxiv.org/abs/2505.19623)
- [37] Benchmark Leakage Trap: Can We Trust LLM-based Recommendation?, 2026. [https://arxiv.org/abs/2602.13626](https://arxiv.org/abs/2602.13626)
- [38] LEADRE: Multi-Faceted Knowledge Enhanced LLM Empowered Display Advertisement Recommender System, 2024/2025. [https://arxiv.org/abs/2411.13789](https://arxiv.org/abs/2411.13789)
- [39] Krichene 与 Rendle, Candidate Set Sampling for Evaluating Top-N Recommendation, 2020. [https://arxiv.org/abs/2001.08959](https://arxiv.org/abs/2001.08959)
- [40] DaisyRec 2.0: Benchmarking Recommendation for Rigorous Evaluation, 2022. [https://arxiv.org/abs/2206.10848](https://arxiv.org/abs/2206.10848)
- [41] LLM-as-a-Judge: Rapid Evaluation of Legal Document Recommendation for RAG, 2025. [https://arxiv.org/abs/2504.17202](https://arxiv.org/abs/2504.17202)
