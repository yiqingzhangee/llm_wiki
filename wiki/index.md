# Wiki 索引

**如何使用本索引**

- **Meta / 运维** — 本仓库与代理工作流如何组织（`CLAUDE.md`、摄入/查询/Lint）。**不是**来自 `raw/` 的主题知识。
- **领域知识** — 从 `raw/` 提炼的概念、实体、来源摘要与对比（论文、代码镜像等）。

流程说明见 [Wiki 运维](meta/wiki-operations.md) 与仓库根目录 [`CLAUDE.md`](../CLAUDE.md)。

---

## Meta / 运维（工作流与仓库结构）

- [meta/README.md](meta/README.md) — **为何首批页面不像「知识库正文」**；`meta/` 与领域目录的分工
- [meta/three-layer-architecture.md](meta/three-layer-architecture.md) — `raw/` / `wiki/` / `CLAUDE.md` 三层
- [meta/wiki-operations.md](meta/wiki-operations.md) — 摄入、查询、Lint
- [meta/repository-and-wiki-structure.md](meta/repository-and-wiki-structure.md) — 目录树与阅读顺序
- [meta/raw-sources-vs-wiki-layer.md](meta/raw-sources-vs-wiki-layer.md) — 原始层与 wiki 层
- [meta/this-wiki-repository.md](meta/this-wiki-repository.md) — 本仓库与 `raw/` 输入清单（元说明）
- [meta/schema-claude-md.md](meta/schema-claude-md.md) — 根目录 `CLAUDE.md` 摘要
- [meta/证据分级与引用规范.md](meta/证据分级与引用规范.md) — `wiki/` 证据分层（E0-E4）与结论引用约束

## Meta / 模式说明（中文）

- [CLAUDE.zh.md](CLAUDE.zh.md) — 根目录 `CLAUDE.md` 的中文说明

---

## 领域知识 — 概念（Concepts）

- [concepts/agent-ai-and-multimodal-agents.md](concepts/agent-ai-and-multimodal-agents.md) — Agent AI / 多模态智能体（MAA）（来自 arXiv 2401.03568 综述摄入）
- [concepts/推理评测污染与捷径风险.md](concepts/推理评测污染与捷径风险.md) — 区分 benchmark contamination 与 reasoning shortcuts 的评估风险框架
- [concepts/LLM推理结构与自纠错框架.md](concepts/LLM推理结构与自纠错框架.md) — 从搜索、辩论、回滚、代码执行到案例推理的结构化纠错图谱
- [concepts/表格数据深度学习.md](concepts/表格数据深度学习.md) — 表格数据深度学习的问题空间、代表模型与边界
- [concepts/TabLLM.md](concepts/TabLLM.md) — TabLLM 方法要点、适用场景与风险
- [concepts/Agent框架选型与工程维度.md](concepts/Agent框架选型与工程维度.md) — Agent 框架选型中的分层能力与工程治理维度
- [concepts/多Agent系统研究趋势与工程映射.md](concepts/多Agent系统研究趋势与工程映射.md) — 连接多Agent论文趋势与工程落地能力的映射框架
- [concepts/流量风控技能栈与Agent治理.md](concepts/流量风控技能栈与Agent治理.md) — 从传统反作弊扩展到 Agent 调用链治理的分层技能栈（含对抗样本分层防御小节）
- [concepts/Agent任务分流：工作流与智能体边界.md](concepts/Agent任务分流：工作流与智能体边界.md) — 用复杂度/价值/可执行性/错误成本四问划分 workflow 与 agent 的适用边界
- [concepts/Stable-Diffusion微调技术矩阵.md](concepts/Stable-Diffusion微调技术矩阵.md) — SD 微调方法全景对比（LoRA/DreamBooth/ControlNet/IP-Adapter 等）与选型决策
- [concepts/对抗样本攻击原理与实践.md](concepts/对抗样本攻击原理与实践.md) — VLM 对抗攻击三类方法（迁移型/多尺度/扩散生成）的 ASR 实测与防御量化
- [concepts/风控工程实战案例-ByteDance.md](concepts/风控工程实战案例-ByteDance.md) — 字节电商+内容风控一线工程复盘（双 Runtime Agent、闪帧治理、团伙挖掘、统一大盘）
- [concepts/可信图神经网络.md](concepts/可信图神经网络.md) — 从图学习视角整理鲁棒性/可解释性/隐私/公平/问责/环境福祉六维治理框架
- [concepts/Agent记忆系统与自我演进.md](concepts/Agent记忆系统与自我演进.md) — Foundation Agent Memory 三维分类法（基底/认知机制/主体），记忆从被动存储到可学习策略的自我演进
- [concepts/时间序列LLM智能体分类法.md](concepts/时间序列LLM智能体分类法.md) — 时序 Agent 问题驱动四分类 + 架构/工具/记忆三维度（异常检测子域方法谱）
- [concepts/多模态嵌入与检索基座.md](concepts/多模态嵌入与检索基座.md) — CLIP→MLLM 隐藏状态→通用多模态嵌入模型族演进与 WeMM 两阶段训练范式

## 领域知识 — 实体（Entities）

- [entities/langgraph.md](entities/langgraph.md) — LangGraph 框架
- [entities/gstack.md](entities/gstack.md) — GStack（Claude Code 技能栈）
- [entities/GSM1k.md](entities/GSM1k.md) — 小学算术污染检测对照基准
- [entities/rsbench.md](entities/rsbench.md) — Neuro-Symbolic 推理捷径评测基准套件

## 领域知识 — 来源（Sources，对应 `raw/`）

- [sources/_index.md](sources/_index.md) — 来源总表
- [sources/paper-agent-ai-survey-2401-03568.md](sources/paper-agent-ai-survey-2401-03568.md) — Agent AI 综述（`raw/articles/`）
- [sources/repo-claude-code-best-practice.md](sources/repo-claude-code-best-practice.md) — Claude Code 最佳实践仓库
- [sources/repo-langgraph.md](sources/repo-langgraph.md) — LangGraph 仓库
- [sources/repo-gstack.md](sources/repo-gstack.md) — GStack 仓库
- [sources/来源-Kimi-Agent-自主持续优化研究报告.md](sources/来源-Kimi-Agent-自主持续优化研究报告.md) — Kimi Agent 自主持续优化研究报告（`raw/transcripts/`）
- [sources/来源-表格数据Transformer模型综述与实践.md](sources/来源-表格数据Transformer模型综述与实践.md) — 表格数据 Transformer 模型综述与实践（`raw/transcripts/`）
- [sources/来源-搜索归档-Agent框架-2026-06-20.md](sources/来源-搜索归档-Agent框架-2026-06-20.md) — `raw/search/` 批次归档的 Agent 框架来源汇总与证据分层
- [sources/来源-搜索归档-Claude-Code-Minimal-Stack-2026-06-20.md](sources/来源-搜索归档-Claude-Code-Minimal-Stack-2026-06-20.md) — Claude Code minimal stack 搜索归档的证据总结与使用边界
- [sources/来源-搜索归档-多Agent-arXiv-2026-06-20.md](sources/来源-搜索归档-多Agent-arXiv-2026-06-20.md) — 多Agent arXiv 近期论文归档与趋势提要
- [sources/来源-搜索归档-流量风控skills-2026-06-20.md](sources/来源-搜索归档-流量风控skills-2026-06-20.md) — 流量风控技能与 Agent 治理搜索归档的证据分层总结
- [sources/来源-搜索归档-Agent风控情报Watch-2026-06-21.md](sources/来源-搜索归档-Agent风控情报Watch-2026-06-21.md) — Agent+风控定时情报批次的证据质量分层与平台偏移说明
- [sources/来源-搜索归档-Agent风控情报Watch-2026-06-27.md](sources/来源-搜索归档-Agent风控情报Watch-2026-06-27.md) — 06-27 情报批次：行动链风险治理、Agent 双重角色、Berkeley/Microsoft/Oracle 治理框架与平台治理信号
- [sources/来源-构建Agent的四个问题-2026-06-21.md](sources/来源-构建Agent的四个问题-2026-06-21.md) — “复杂度/价值/可执行性/错误成本”四问框架的来源归档与边界说明
- [sources/来源-TCE部署ClaudeCodeAgent操作清单-2026-06-21.md](sources/来源-TCE部署ClaudeCodeAgent操作清单-2026-06-21.md) — TCE + 飞书 + Claude Code 的企业内网部署清单与运维要点
- [sources/来源-基于控制论的内容风控系统全流程方案-2026-06-21.md](sources/来源-基于控制论的内容风控系统全流程方案-2026-06-21.md) — 将内容风控建模为传感器-控制器-执行器-反馈回路的闭环方案
- [sources/来源-Qwen2.5与Qwen3微调深度调研-2026-06-21.md](sources/来源-Qwen2.5与Qwen3微调深度调研-2026-06-21.md) — Qwen2.5 到 Qwen3 微调范式演进与框架选型对照
- [sources/来源-AIAgent常见框架深度介绍-2026-06-21.md](sources/来源-AIAgent常见框架深度介绍-2026-06-21.md) — Agent 框架全景综述与“确定性编排+局部自主”趋势判断
- [sources/来源-内容风控与机审算法面试题详解-2026-06-21.md](sources/来源-内容风控与机审算法面试题详解-2026-06-21.md) — 内容机审算法与系统设计题解的方法清单与边界说明
- [sources/来源-AIAgent面试题详解-2026-06-21.md](sources/来源-AIAgent面试题详解-2026-06-21.md) — Agent 面试高频问题的工程化知识框架（闭环执行、记忆分层、工具治理）
- [sources/来源-对抗样本生成与检测调研实验-2026-06-21.md](sources/来源-对抗样本生成与检测调研实验-2026-06-21.md) — VLM 对抗攻击家族与分层防御实验总结（含视觉编码瓶颈诊断）
- [sources/来源-Agent时代的K8S与AgentOS形态设想-2026-06-21.md](sources/来源-Agent时代的K8S与AgentOS形态设想-2026-06-21.md) — 以 K8S 类比系统化解释 AgentOS 的运行时分层、调度与治理边界
- [sources/来源-AIAgent学习与智能诊断落地路线图-2026-06-21.md](sources/来源-AIAgent学习与智能诊断落地路线图-2026-06-21.md) — 面向 SRE 智能诊断的 30 天学习与工程化落地路线（含评测与护栏）
- [sources/来源-论文调研与对抗检测方向笔记-2026-06-21.md](sources/来源-论文调研与对抗检测方向笔记-2026-06-21.md) — 小目标检测、时序聚合与对抗样本线索的阶段性研究笔记
- [sources/来源-经典Agent综述论文翻译第二批-2026-06-21.md](sources/来源-经典Agent综述论文翻译第二批-2026-06-21.md) — 第二批经典 Agent 综述导读（工具学习/增强模型/多智能体/规划/个人Agent）
- [sources/来源-LLM自主智能体综述-Renmin-2025.md](sources/来源-LLM自主智能体综述-Renmin-2025.md) — 人大高瓴 AI 学院综述：画像-记忆-规划-行动四模块统一框架

- [sources/来源-CoALA-语言代理认知架构-Princeton.md](sources/来源-CoALA-语言代理认知架构-Princeton.md) — 普林斯顿 CoALA：LLM 作为概率产生式系统的认知架构框架
- [sources/来源-AgentAI-多模态交互综述-Stanford-Microsoft.md](sources/来源-AgentAI-多模态交互综述-Stanford-Microsoft.md) — Stanford/Microsoft：跨现实多模态 Agent AI 综述（完整翻译版）
- [sources/来源-GUI-Agent综述-LLM-Brained-2025.md](sources/来源-GUI-Agent综述-LLM-Brained-2025.md) — LLM-Brained GUI Agent 综述：从历史演进到构建菜谱（500+ 参考文献）
- [sources/来源-raw-transcripts增量批次-2026-06-21.md](sources/来源-raw-transcripts增量批次-2026-06-21.md) — 本轮 transcripts 增量来源编目与后续拆解导航（Agent/风控/微调）
- [sources/来源-小学算术基准污染检验-GSM1k-2024.md](sources/来源-小学算术基准污染检验-GSM1k-2024.md) — GSM8k 对照新测集揭示部分模型评测污染风险
- [sources/来源-Neuro-Symbolic推理捷径基准-rsbench-2024.md](sources/来源-Neuro-Symbolic推理捷径基准-rsbench-2024.md) — 用 rsbench 评估“标签正确但概念错误”的推理捷径问题
- [sources/来源-DS-Agent-案例推理驱动自动化数据科学-ICML2024.md](sources/来源-DS-Agent-案例推理驱动自动化数据科学-ICML2024.md) — CBR 驱动的数据科学 Agent 闭环
- [sources/来源-多智能体辩论提升事实性与推理-ICML2024.md](sources/来源-多智能体辩论提升事实性与推理-ICML2024.md) — 通过多轮辩论提升事实性与推理稳健性
- [sources/来源-LATS-统一推理行动规划-ICML2024.md](sources/来源-LATS-统一推理行动规划-ICML2024.md) — 统一 reasoning/acting/planning 的 MCTS Agent 框架
- [sources/来源-Chain-of-Code-代码模拟混合推理-ICML2024.md](sources/来源-Chain-of-Code-代码模拟混合推理-ICML2024.md) — 代码执行与 LM 模拟的混合推理范式
- [sources/来源-Thought-Rollback-自适应回滚推理-ICML2024.md](sources/来源-Thought-Rollback-自适应回滚推理-ICML2024.md) — 回滚驱动的自适应推理路径修复
- [sources/来源-SELF-DISCOVER-自组装推理结构-NeurIPS2024.md](sources/来源-SELF-DISCOVER-自组装推理结构-NeurIPS2024.md) — 任务级自动发现并执行推理结构
- [sources/来源-MR-Ben-元推理基准-System2-NeurIPS2024.md](sources/来源-MR-Ben-元推理基准-System2-NeurIPS2024.md) — 用错误定位/错因解释评估 LLM 的元推理能力
- [sources/来源-Buffer-of-Thoughts-思维模板缓冲-NeurIPS2024.md](sources/来源-Buffer-of-Thoughts-思维模板缓冲-NeurIPS2024.md) — 通过可复用思维模板缓冲兼顾推理效果与成本
- [sources/来源-GLoRe-全局局部联合修正-ICML2024.md](sources/来源-GLoRe-全局局部联合修正-ICML2024.md) — 结合 ORM/SORM 的全局与局部修正协同框架
- [sources/来源-Reverse-Curriculum-RL-逆课程推理强化-ICML2024.md](sources/来源-Reverse-Curriculum-RL-逆课程推理强化-ICML2024.md) — 从中间状态回退的逆课程强化推理方法
- [sources/来源-AdaProp-自适应传播路径-KDD2024.md](sources/来源-AdaProp-自适应传播路径-KDD2024.md) — 查询相关的 KG 传播路径自适应采样
- [sources/来源-NRN-实体与数值联合推理-KDD2024.md](sources/来源-NRN-实体与数值联合推理-KDD2024.md) — 将实体与数值分相位编码的复杂查询推理框架
- [sources/来源-可信图神经网络综述-2022.md](sources/来源-可信图神经网络综述-2022.md) — 可信 GNN 六维框架导读：鲁棒性、可解释性、隐私、公平性、问责与环境福祉
- [sources/来源-Stable-Diffusion微调技术深度调研-2026-06-23.md](sources/来源-Stable-Diffusion微调技术深度调研-2026-06-23.md) — SD 微调技术全景调研（LoRA/DreamBooth/ControlNet/IP-Adapter 等 9 种方法 + 5 大框架对比）
- [sources/来源-Stable-Diffusion架构代表性工作与原理深度解析-2026-06-23.md](sources/来源-Stable-Diffusion架构代表性工作与原理深度解析-2026-06-23.md) — SD 架构演进深度解析（LDM→SDXL→SD3/MMDiT→FLUX，含关键代码实现）
- [sources/来源-视频透明图层提取技术方案-2026-06-23.md](sources/来源-视频透明图层提取技术方案-2026-06-23.md) — 视频抠图/层分解完整技术方案（RVM/MODNet/MatAnyone + 生成式前沿）
- [sources/来源-对抗图像样本生成原理与实践-2026-06-23.md](sources/来源-对抗图像样本生成原理与实践-2026-06-23.md) — 对抗攻击实践（MPCAttack 92% ASR + 扩散净化防御 + 线上部署代码）
- [sources/来源-风控工程实战案例面试问题答案-2026-06-23.md](sources/来源-风控工程实战案例面试问题答案-2026-06-23.md) — 字节电商+TikTok 风控面试复盘（双 Runtime Agent、闪帧治理、团伙挖掘、统一大盘架构）
- [sources/来源-LLM智能体规划综述-USTC-Huawei-2026-06-23.md](sources/来源-LLM智能体规划综述-USTC-Huawei-2026-06-23.md) — USTC/华为：LLM 智能体规划五大方向（分解/选择/外部/反思/记忆）系统综述
- [sources/来源-个人LLM智能体综述-Tsinghua-AIR-2026-06-23.md](sources/来源-个人LLM智能体综述-Tsinghua-AIR-2026-06-23.md) — 清华AIR：个人LLM智能体体系结构、L1-L5智能等级与边-云协同专家共识
- [sources/来源-工具学习与基础模型综述-Tsinghua-2026-06-23.md](sources/来源-工具学习与基础模型综述-Tsinghua-2026-06-23.md) — 清华大学主导：基于基础模型的工具学习统一框架（控制器-工具-环境-感知器）及其认知起源、训练策略与安全挑战
- [sources/来源-增强语言模型综述-ALM-MetaAI-2026-06-23.md](sources/来源-增强语言模型综述-ALM-MetaAI-2026-06-23.md) — Meta AI（Yann LeCun 参与）：增强语言模型 ALM 综述，推理+工具双重增强打破纯统计语言建模范式
- [sources/来源-LLM多智能体综述-NotreDame-2026-06-23.md](sources/来源-LLM多智能体综述-NotreDame-2026-06-23.md) — 圣母大学主导：LLM 多智能体系统四维度框架（接口/画像/通信/能力获取）与问题求解+世界模拟双模式全景
- [sources/来源-Agent记忆综述-2026.md](sources/来源-Agent记忆综述-2026.md) — Foundation Agent Memory 综述：基底/认知机制/主体三维分类法与自我演进开放挑战（`raw/articles/agent_memory/`）
- [sources/来源-视频审核与电商风控GUI-Agent批次-2026.md](sources/来源-视频审核与电商风控GUI-Agent批次-2026.md) — UNIVID/IPS/VideoModerator 审核演进 + RISK/RiskWebWorld 电商风控 GUI Agent 框架与基准
- [sources/来源-时间序列LLM智能体综述-2026.md](sources/来源-时间序列LLM智能体综述-2026.md) — 时序 LLM Agent 综述四分类 + AD-Agent/ARGOS/CALM/SAGE/AgentFM 等子论文（`raw/articles/time_series_agent/`）
- [sources/来源-AD-Agent多Agent异常检测-2025.md](sources/来源-AD-Agent多Agent异常检测-2025.md) — 7 Agent + 短/长期记忆，自然语言→可执行 AD 流水线，集成 PyOD/PyGOD/TSLib，BSD-2 开源
- [sources/来源-ARGOS自主规则时序异常检测-2025.md](sources/来源-ARGOS自主规则时序异常检测-2025.md) — LLM 训练期生成可解释可复现 Python 规则，三 Agent + 模型融合，KPI +9.5% F1
- [sources/来源-CALM流式异常检测LLM裁判-2025.md](sources/来源-CALM流式异常检测LLM裁判-2025.md) — Apache Beam + TimesFM，LLM-as-a-Judge 漂移检测 + 闭环持续微调
- [sources/来源-SAGE专家式多Agent时序诊断-2026.md](sources/来源-SAGE专家式多Agent时序诊断-2026.md) — 5 阶段管线 + 4 族 9 类分析器 + 合成 ICL（k-medoids+DTW），LLM 打分 temperature=0
- [sources/来源-WeMM多模态嵌入技术报告-2026.md](sources/来源-WeMM多模态嵌入技术报告-2026.md) — WeChat 通用多模态嵌入模型族（2B/4B/9B）两阶段训练，登顶 MMEB-v2

## 领域知识 — 对比（Comparisons）

- [comparisons/claude-code-curated-docs-vs-gstack.md](comparisons/claude-code-curated-docs-vs-gstack.md) — 两套 Claude Code 资料/工作流对照
- [comparisons/表格建模路线对比-TabLLM-vs-树模型-vs-Transformer.md](comparisons/表格建模路线对比-TabLLM-vs-树模型-vs-Transformer.md) — TabLLM、树模型与表格 Transformer 的能力/代价对照
- [comparisons/主流Agent框架对比-2026-06-20.md](comparisons/主流Agent框架对比-2026-06-20.md) — 基于 `raw/search/` 批次的主流 Agent 框架工程维度对照（含可打分占位表）
- [comparisons/多Agent论文方向对比-2026H1-arXiv.md](comparisons/多Agent论文方向对比-2026H1-arXiv.md) — 2026H1 多Agent arXiv 论文方向分层与工程关联
- [comparisons/传统流量反作弊与Agent流量治理对比.md](comparisons/传统流量反作弊与Agent流量治理对比.md) — 传统广告反作弊与 Agent 调用链治理的差异边界（含可打分占位表）
- [comparisons/工作流与Agent任务分流对比-2026-06-21.md](comparisons/工作流与Agent任务分流对比-2026-06-21.md) — 基于四问框架比较 workflow 与 agent 的适用条件与治理代价
- [comparisons/评测污染与推理捷径风险对照-2026-06-22.md](comparisons/评测污染与推理捷径风险对照-2026-06-22.md) — 对照 benchmark contamination 与 reasoning shortcuts 的风险来源与治理手段
- [comparisons/推理框架六法对照-2026-06-22.md](comparisons/推理框架六法对照-2026-06-22.md) — 对照十二类推理/修正/评测路径的结构、纠错信号与成本边界
- [comparisons/时序异常检测方法对照-2026-08-31.md](comparisons/时序异常检测方法对照-2026-08-31.md) — ARGOS/CALM/SAGE/AD-Agent 在 LLM 介入程度、可解释性、可复现性、自主性、开源上的设计取舍与风控治理同构
- [comparisons/多模态嵌入基座对比-2026-08-31.md](comparisons/多模态嵌入基座对比-2026-08-31.md) — WeMM vs CLIP/Gemini Embedding 的演进路线、规模-性能前沿与选型建议（证据层级不均，Gemini/CLIP 为 E3 外部参照）

## 领域知识 — 地图（Maps）

- [maps/README.md](maps/README.md) — 主题导航图放置说明（占位；可随摄入增加领域地图）
- [maps/研究主题导航.md](maps/研究主题导航.md) — 现有主题关系总览（蒸馏/多语/协同/治理/表格）
- [maps/表格建模选型决策树.md](maps/表格建模选型决策树.md) — 按样本量、预算与线上约束选择建模路线
- [maps/多Agent研究与工程落地地图.md](maps/多Agent研究与工程落地地图.md) — 连接多Agent论文趋势与框架工程实践的主题地图
- [maps/研究主题导航.md](maps/研究主题导航.md) — 主题关系总览，已补入 Agent 记忆 / 时序 Agent / 多模态嵌入 / 视频审核与电商风控 GUI Agent 批次及其跨域连接

## 运维

- [log.md](log.md) — 仅追加的操作时间线
