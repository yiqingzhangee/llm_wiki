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
- [meta/证据分级与引用规范.md](meta/%E8%AF%81%E6%8D%AE%E5%88%86%E7%BA%A7%E4%B8%8E%E5%BC%95%E7%94%A8%E8%A7%84%E8%8C%83.md) — `wiki/` 证据分层（E0-E4）与结论引用约束

## Meta / 模式说明（中文）

- [CLAUDE.zh.md](CLAUDE.zh.md) — 根目录 `CLAUDE.md` 的中文说明

---

## 领域知识 — 概念（Concepts）

- [concepts/agent-ai-and-multimodal-agents.md](concepts/agent/agent-ai-and-multimodal-agents.md) — Agent AI / 多模态智能体（MAA）（来自 arXiv 2401.03568 综述摄入）
- [concepts/推理评测污染与捷径风险.md](concepts/reasoning-training/%E6%8E%A8%E7%90%86%E8%AF%84%E6%B5%8B%E6%B1%A1%E6%9F%93%E4%B8%8E%E6%8D%B7%E5%BE%84%E9%A3%8E%E9%99%A9.md) — 区分 benchmark contamination 与 reasoning shortcuts 的评估风险框架
- [concepts/LLM推理结构与自纠错框架.md](concepts/reasoning-training/LLM%E6%8E%A8%E7%90%86%E7%BB%93%E6%9E%84%E4%B8%8E%E8%87%AA%E7%BA%A0%E9%94%99%E6%A1%86%E6%9E%B6.md) — 从搜索、辩论、回滚、代码执行到案例推理的结构化纠错图谱
- [concepts/自回归掩码、解码与生成评测.md](concepts/reasoning-training/%E8%87%AA%E5%9B%9E%E5%BD%92%E6%8E%A9%E7%A0%81%E3%80%81%E8%A7%A3%E7%A0%81%E4%B8%8E%E7%94%9F%E6%88%90%E8%AF%84%E6%B5%8B.md) — 区分因果掩码、贪婪/采样/搜索解码与 pass@1 的结构、推理与评测层次
- [concepts/表格数据深度学习.md](concepts/applications/%E8%A1%A8%E6%A0%BC%E6%95%B0%E6%8D%AE%E6%B7%B1%E5%BA%A6%E5%AD%A6%E4%B9%A0.md) — 表格数据深度学习的问题空间、代表模型与边界
- [concepts/TabLLM.md](concepts/applications/TabLLM.md) — TabLLM 方法要点、适用场景与风险
- [concepts/Agent框架选型与工程维度.md](concepts/agent/Agent%E6%A1%86%E6%9E%B6%E9%80%89%E5%9E%8B%E4%B8%8E%E5%B7%A5%E7%A8%8B%E7%BB%B4%E5%BA%A6.md) — Agent 框架选型中的分层能力与工程治理维度
- [concepts/多Agent系统研究趋势与工程映射.md](concepts/agent/%E5%A4%9AAgent%E7%B3%BB%E7%BB%9F%E7%A0%94%E7%A9%B6%E8%B6%8B%E5%8A%BF%E4%B8%8E%E5%B7%A5%E7%A8%8B%E6%98%A0%E5%B0%84.md) — 连接多Agent论文趋势与工程落地能力的映射框架
- [concepts/流量风控技能栈与Agent治理.md](concepts/applications/%E6%B5%81%E9%87%8F%E9%A3%8E%E6%8E%A7%E6%8A%80%E8%83%BD%E6%A0%88%E4%B8%8EAgent%E6%B2%BB%E7%90%86.md) — 从传统反作弊扩展到 Agent 调用链治理的分层技能栈（含对抗样本分层防御小节）
- [concepts/Agent任务分流：工作流与智能体边界.md](concepts/agent/Agent%E4%BB%BB%E5%8A%A1%E5%88%86%E6%B5%81%EF%BC%9A%E5%B7%A5%E4%BD%9C%E6%B5%81%E4%B8%8E%E6%99%BA%E8%83%BD%E4%BD%93%E8%BE%B9%E7%95%8C.md) — 用复杂度/价值/可执行性/错误成本四问划分 workflow 与 agent 的适用边界
- [concepts/Stable-Diffusion微调技术矩阵.md](concepts/multimodal/Stable-Diffusion%E5%BE%AE%E8%B0%83%E6%8A%80%E6%9C%AF%E7%9F%A9%E9%98%B5.md) — SD 微调方法全景对比（LoRA/DreamBooth/ControlNet/IP-Adapter 等）与选型决策
- [concepts/对抗样本攻击原理与实践.md](concepts/multimodal/%E5%AF%B9%E6%8A%97%E6%A0%B7%E6%9C%AC%E6%94%BB%E5%87%BB%E5%8E%9F%E7%90%86%E4%B8%8E%E5%AE%9E%E8%B7%B5.md) — VLM 对抗攻击三类方法（迁移型/多尺度/扩散生成）的 ASR 实测与防御量化
- [concepts/风控工程实战案例-ByteDance.md](concepts/applications/%E9%A3%8E%E6%8E%A7%E5%B7%A5%E7%A8%8B%E5%AE%9E%E6%88%98%E6%A1%88%E4%BE%8B-ByteDance.md) — 字节电商+内容风控一线工程复盘（双 Runtime Agent、闪帧治理、团伙挖掘、统一大盘）
- [concepts/可信图神经网络.md](concepts/graph/%E5%8F%AF%E4%BF%A1%E5%9B%BE%E7%A5%9E%E7%BB%8F%E7%BD%91%E7%BB%9C.md) — 从图学习视角整理鲁棒性/可解释性/隐私/公平/问责/环境福祉六维治理框架
- [concepts/Agent记忆系统与自我演进.md](concepts/agent/Agent%E8%AE%B0%E5%BF%86%E7%B3%BB%E7%BB%9F%E4%B8%8E%E8%87%AA%E6%88%91%E6%BC%94%E8%BF%9B.md) — Foundation Agent Memory 三维分类法（基底/认知机制/主体），记忆从被动存储到可学习策略的自我演进
- [concepts/时间序列LLM智能体分类法.md](concepts/applications/%E6%97%B6%E9%97%B4%E5%BA%8F%E5%88%97LLM%E6%99%BA%E8%83%BD%E4%BD%93%E5%88%86%E7%B1%BB%E6%B3%95.md) — 时序 Agent 问题驱动四分类 + 架构/工具/记忆三维度（异常检测子域方法谱）
- [concepts/多模态嵌入与检索基座.md](concepts/multimodal/%E5%A4%9A%E6%A8%A1%E6%80%81%E5%B5%8C%E5%85%A5%E4%B8%8E%E6%A3%80%E7%B4%A2%E5%9F%BA%E5%BA%A7.md) — CLIP→MLLM 隐藏状态→通用多模态嵌入模型族演进与 WeMM 两阶段训练范式
- [concepts/知识图谱驱动的深度研究Agent.md](concepts/agent/%E7%9F%A5%E8%AF%86%E5%9B%BE%E8%B0%B1%E9%A9%B1%E5%8A%A8%E7%9A%84%E6%B7%B1%E5%BA%A6%E7%A0%94%E7%A9%B6Agent.md) — 以证据图谱、统一 KG 工具层与可执行分析环境组织 deep research 的 Agent 形态
- [concepts/多模态模型视觉头与风控部署边界.md](concepts/multimodal/%E5%A4%9A%E6%A8%A1%E6%80%81%E6%A8%A1%E5%9E%8B%E8%A7%86%E8%A7%89%E5%A4%B4%E4%B8%8E%E9%A3%8E%E6%8E%A7%E9%83%A8%E7%BD%B2%E8%BE%B9%E7%95%8C.md) — 把视觉头能力、整模型表现与风控系统 KPI 三层拆开，明确 VLM 在风控中的语义证据层定位
- [concepts/参数高效微调与对齐训练.md](concepts/reasoning-training/%E5%8F%82%E6%95%B0%E9%AB%98%E6%95%88%E5%BE%AE%E8%B0%83%E4%B8%8E%E5%AF%B9%E9%BD%90%E8%AE%AD%E7%BB%83.md) — 把 LoRA/QLoRA、DPO、GRPO 放回同一训练栈，区分“参数怎么省”与“行为怎么对齐”
- [concepts/递归自我改进与Harness自演化.md](concepts/agent/%E9%80%92%E5%BD%92%E8%87%AA%E6%88%91%E6%94%B9%E8%BF%9B%E4%B8%8EHarness%E8%87%AA%E6%BC%94%E5%8C%96.md) — 将模糊目标、Harness、经验整合和独立验证拆成 RSI 四环，明确当前证据的迁移边界
- [concepts/LLM推荐系统的分层改造与评估.md](concepts/applications/LLM%E6%8E%A8%E8%8D%90%E7%B3%BB%E7%BB%9F%E7%9A%84%E5%88%86%E5%B1%82%E6%94%B9%E9%80%A0%E4%B8%8E%E8%AF%84%E4%BC%B0.md) — 按语义增强、受约束生成与 Agent 编排组织 LLM4Rec，并给出四层验收口径
- [concepts/LLM用户表征基础模型与场景适配.md](concepts/applications/LLM%E7%94%A8%E6%88%B7%E8%A1%A8%E5%BE%81%E5%9F%BA%E7%A1%80%E6%A8%A1%E5%9E%8B%E4%B8%8E%E5%9C%BA%E6%99%AF%E9%80%82%E9%85%8D.md) — 汇总未来对齐、量化长历史、条件化表示与掩码适配，并明确其在推荐等系统中的表征层边界
- [concepts/多模态结构保真与统一对齐.md](concepts/multimodal/%E5%A4%9A%E6%A8%A1%E6%80%81%E7%BB%93%E6%9E%84%E4%BF%9D%E7%9C%9F%E4%B8%8E%E7%BB%9F%E4%B8%80%E5%AF%B9%E9%BD%90.md) — 区分共享接口与原始表示统一，按数值、时间、空间与拓扑结构组织边—环—整体评测
- [concepts/Qwen异构特征接入与事件序列建模.md](concepts/applications/Qwen%E5%BC%82%E6%9E%84%E7%89%B9%E5%BE%81%E6%8E%A5%E5%85%A5%E4%B8%8E%E4%BA%8B%E4%BB%B6%E5%BA%8F%E5%88%97%E5%BB%BA%E6%A8%A1.md) — 以无泄漏契约、强基线与分阶段门禁组织事件、统计特征接入 Qwen 的路线

## 领域知识 — 实体（Entities）

- [entities/langgraph.md](entities/langgraph.md) — LangGraph 框架
- [entities/gstack.md](entities/gstack.md) — GStack（Claude Code 技能栈）
- [entities/GSM1k.md](entities/GSM1k.md) — 小学算术污染检测对照基准
- [entities/rsbench.md](entities/rsbench.md) — Neuro-Symbolic 推理捷径评测基准套件

## 领域知识 — 来源（Sources，对应 `raw/`）

- [sources/_index.md](sources/_index.md) — 来源总表
- [sources/paper-agent-ai-survey-2401-03568.md](sources/agent/paper-agent-ai-survey-2401-03568.md) — Agent AI 综述（`raw/articles/`）
- [sources/repo-claude-code-best-practice.md](sources/code/repo-claude-code-best-practice.md) — Claude Code 最佳实践仓库
- [sources/repo-langgraph.md](sources/code/repo-langgraph.md) — LangGraph 仓库
- [sources/repo-gstack.md](sources/code/repo-gstack.md) — GStack 仓库
- [sources/来源-Kimi-Agent-自主持续优化研究报告.md](sources/agent/%E6%9D%A5%E6%BA%90-Kimi-Agent-%E8%87%AA%E4%B8%BB%E6%8C%81%E7%BB%AD%E4%BC%98%E5%8C%96%E7%A0%94%E7%A9%B6%E6%8A%A5%E5%91%8A.md) — Kimi Agent 自主持续优化研究报告（`raw/transcripts/`）
- [sources/来源-表格数据Transformer模型综述与实践.md](sources/applications/%E6%9D%A5%E6%BA%90-%E8%A1%A8%E6%A0%BC%E6%95%B0%E6%8D%AETransformer%E6%A8%A1%E5%9E%8B%E7%BB%BC%E8%BF%B0%E4%B8%8E%E5%AE%9E%E8%B7%B5.md) — 表格数据 Transformer 模型综述与实践（`raw/transcripts/`）
- [sources/来源-搜索归档-Agent框架-2026-06-20.md](sources/agent/%E6%9D%A5%E6%BA%90-%E6%90%9C%E7%B4%A2%E5%BD%92%E6%A1%A3-Agent%E6%A1%86%E6%9E%B6-2026-06-20.md) — `raw/search/` 批次归档的 Agent 框架来源汇总与证据分层
- [sources/来源-搜索归档-Claude-Code-Minimal-Stack-2026-06-20.md](sources/agent/%E6%9D%A5%E6%BA%90-%E6%90%9C%E7%B4%A2%E5%BD%92%E6%A1%A3-Claude-Code-Minimal-Stack-2026-06-20.md) — Claude Code minimal stack 搜索归档的证据总结与使用边界
- [sources/来源-搜索归档-多Agent-arXiv-2026-06-20.md](sources/agent/%E6%9D%A5%E6%BA%90-%E6%90%9C%E7%B4%A2%E5%BD%92%E6%A1%A3-%E5%A4%9AAgent-arXiv-2026-06-20.md) — 多Agent arXiv 近期论文归档与趋势提要
- [sources/来源-搜索归档-流量风控skills-2026-06-20.md](sources/applications/%E6%9D%A5%E6%BA%90-%E6%90%9C%E7%B4%A2%E5%BD%92%E6%A1%A3-%E6%B5%81%E9%87%8F%E9%A3%8E%E6%8E%A7skills-2026-06-20.md) — 流量风控技能与 Agent 治理搜索归档的证据分层总结
- [sources/来源-搜索归档-Agent风控情报Watch-2026-06-21.md](sources/applications/%E6%9D%A5%E6%BA%90-%E6%90%9C%E7%B4%A2%E5%BD%92%E6%A1%A3-Agent%E9%A3%8E%E6%8E%A7%E6%83%85%E6%8A%A5Watch-2026-06-21.md) — Agent+风控定时情报批次的证据质量分层与平台偏移说明
- [sources/来源-搜索归档-Agent风控情报Watch-2026-06-27.md](sources/applications/%E6%9D%A5%E6%BA%90-%E6%90%9C%E7%B4%A2%E5%BD%92%E6%A1%A3-Agent%E9%A3%8E%E6%8E%A7%E6%83%85%E6%8A%A5Watch-2026-06-27.md) — 06-27 情报批次：行动链风险治理、Agent 双重角色、Berkeley/Microsoft/Oracle 治理框架与平台治理信号
- [sources/来源-构建Agent的四个问题-2026-06-21.md](sources/agent/%E6%9D%A5%E6%BA%90-%E6%9E%84%E5%BB%BAAgent%E7%9A%84%E5%9B%9B%E4%B8%AA%E9%97%AE%E9%A2%98-2026-06-21.md) — “复杂度/价值/可执行性/错误成本”四问框架的来源归档与边界说明
- [sources/来源-TCE部署ClaudeCodeAgent操作清单-2026-06-21.md](sources/agent/%E6%9D%A5%E6%BA%90-TCE%E9%83%A8%E7%BD%B2ClaudeCodeAgent%E6%93%8D%E4%BD%9C%E6%B8%85%E5%8D%95-2026-06-21.md) — TCE + 飞书 + Claude Code 的企业内网部署清单与运维要点
- [sources/来源-基于控制论的内容风控系统全流程方案-2026-06-21.md](sources/applications/%E6%9D%A5%E6%BA%90-%E5%9F%BA%E4%BA%8E%E6%8E%A7%E5%88%B6%E8%AE%BA%E7%9A%84%E5%86%85%E5%AE%B9%E9%A3%8E%E6%8E%A7%E7%B3%BB%E7%BB%9F%E5%85%A8%E6%B5%81%E7%A8%8B%E6%96%B9%E6%A1%88-2026-06-21.md) — 将内容风控建模为传感器-控制器-执行器-反馈回路的闭环方案
- [sources/来源-Qwen2.5与Qwen3微调深度调研-2026-06-21.md](sources/reasoning-training/%E6%9D%A5%E6%BA%90-Qwen2.5%E4%B8%8EQwen3%E5%BE%AE%E8%B0%83%E6%B7%B1%E5%BA%A6%E8%B0%83%E7%A0%94-2026-06-21.md) — Qwen2.5 到 Qwen3 微调范式演进与框架选型对照
- [sources/来源-AIAgent常见框架深度介绍-2026-06-21.md](sources/agent/%E6%9D%A5%E6%BA%90-AIAgent%E5%B8%B8%E8%A7%81%E6%A1%86%E6%9E%B6%E6%B7%B1%E5%BA%A6%E4%BB%8B%E7%BB%8D-2026-06-21.md) — Agent 框架全景综述与“确定性编排+局部自主”趋势判断
- [sources/来源-内容风控与机审算法面试题详解-2026-06-21.md](sources/applications/%E6%9D%A5%E6%BA%90-%E5%86%85%E5%AE%B9%E9%A3%8E%E6%8E%A7%E4%B8%8E%E6%9C%BA%E5%AE%A1%E7%AE%97%E6%B3%95%E9%9D%A2%E8%AF%95%E9%A2%98%E8%AF%A6%E8%A7%A3-2026-06-21.md) — 内容机审算法与系统设计题解的方法清单与边界说明
- [sources/来源-AIAgent面试题详解-2026-06-21.md](sources/agent/%E6%9D%A5%E6%BA%90-AIAgent%E9%9D%A2%E8%AF%95%E9%A2%98%E8%AF%A6%E8%A7%A3-2026-06-21.md) — Agent 面试高频问题的工程化知识框架（闭环执行、记忆分层、工具治理）
- [sources/来源-对抗样本生成与检测调研实验-2026-06-21.md](sources/multimodal/%E6%9D%A5%E6%BA%90-%E5%AF%B9%E6%8A%97%E6%A0%B7%E6%9C%AC%E7%94%9F%E6%88%90%E4%B8%8E%E6%A3%80%E6%B5%8B%E8%B0%83%E7%A0%94%E5%AE%9E%E9%AA%8C-2026-06-21.md) — VLM 对抗攻击家族与分层防御实验总结（含视觉编码瓶颈诊断）
- [sources/来源-Agent时代的K8S与AgentOS形态设想-2026-06-21.md](sources/agent/%E6%9D%A5%E6%BA%90-Agent%E6%97%B6%E4%BB%A3%E7%9A%84K8S%E4%B8%8EAgentOS%E5%BD%A2%E6%80%81%E8%AE%BE%E6%83%B3-2026-06-21.md) — 以 K8S 类比系统化解释 AgentOS 的运行时分层、调度与治理边界
- [sources/来源-AIAgent学习与智能诊断落地路线图-2026-06-21.md](sources/agent/%E6%9D%A5%E6%BA%90-AIAgent%E5%AD%A6%E4%B9%A0%E4%B8%8E%E6%99%BA%E8%83%BD%E8%AF%8A%E6%96%AD%E8%90%BD%E5%9C%B0%E8%B7%AF%E7%BA%BF%E5%9B%BE-2026-06-21.md) — 面向 SRE 智能诊断的 30 天学习与工程化落地路线（含评测与护栏）
- [sources/来源-论文调研与对抗检测方向笔记-2026-06-21.md](sources/applications/%E6%9D%A5%E6%BA%90-%E8%AE%BA%E6%96%87%E8%B0%83%E7%A0%94%E4%B8%8E%E5%AF%B9%E6%8A%97%E6%A3%80%E6%B5%8B%E6%96%B9%E5%90%91%E7%AC%94%E8%AE%B0-2026-06-21.md) — 小目标检测、时序聚合与对抗样本线索的阶段性研究笔记
- [sources/来源-经典Agent综述论文翻译第二批-2026-06-21.md](sources/batches/%E6%9D%A5%E6%BA%90-%E7%BB%8F%E5%85%B8Agent%E7%BB%BC%E8%BF%B0%E8%AE%BA%E6%96%87%E7%BF%BB%E8%AF%91%E7%AC%AC%E4%BA%8C%E6%89%B9-2026-06-21.md) — 第二批经典 Agent 综述导读（工具学习/增强模型/多智能体/规划/个人Agent）
- [sources/来源-LLM自主智能体综述-Renmin-2025.md](sources/agent/%E6%9D%A5%E6%BA%90-LLM%E8%87%AA%E4%B8%BB%E6%99%BA%E8%83%BD%E4%BD%93%E7%BB%BC%E8%BF%B0-Renmin-2025.md) — 人大高瓴 AI 学院综述：画像-记忆-规划-行动四模块统一框架

- [sources/来源-CoALA-语言代理认知架构-Princeton.md](sources/agent/%E6%9D%A5%E6%BA%90-CoALA-%E8%AF%AD%E8%A8%80%E4%BB%A3%E7%90%86%E8%AE%A4%E7%9F%A5%E6%9E%B6%E6%9E%84-Princeton.md) — 普林斯顿 CoALA：LLM 作为概率产生式系统的认知架构框架
- [sources/来源-AgentAI-多模态交互综述-Stanford-Microsoft.md](sources/agent/%E6%9D%A5%E6%BA%90-AgentAI-%E5%A4%9A%E6%A8%A1%E6%80%81%E4%BA%A4%E4%BA%92%E7%BB%BC%E8%BF%B0-Stanford-Microsoft.md) — Stanford/Microsoft：跨现实多模态 Agent AI 综述（完整翻译版）
- [sources/来源-GUI-Agent综述-LLM-Brained-2025.md](sources/agent/%E6%9D%A5%E6%BA%90-GUI-Agent%E7%BB%BC%E8%BF%B0-LLM-Brained-2025.md) — LLM-Brained GUI Agent 综述：从历史演进到构建菜谱（500+ 参考文献）
- [sources/来源-raw-transcripts增量批次-2026-06-21.md](sources/batches/%E6%9D%A5%E6%BA%90-raw-transcripts%E5%A2%9E%E9%87%8F%E6%89%B9%E6%AC%A1-2026-06-21.md) — 本轮 transcripts 增量来源编目与后续拆解导航（Agent/风控/微调）
- [sources/来源-小学算术基准污染检验-GSM1k-2024.md](sources/reasoning-training/%E6%9D%A5%E6%BA%90-%E5%B0%8F%E5%AD%A6%E7%AE%97%E6%9C%AF%E5%9F%BA%E5%87%86%E6%B1%A1%E6%9F%93%E6%A3%80%E9%AA%8C-GSM1k-2024.md) — GSM8k 对照新测集揭示部分模型评测污染风险
- [sources/来源-Neuro-Symbolic推理捷径基准-rsbench-2024.md](sources/reasoning-training/%E6%9D%A5%E6%BA%90-Neuro-Symbolic%E6%8E%A8%E7%90%86%E6%8D%B7%E5%BE%84%E5%9F%BA%E5%87%86-rsbench-2024.md) — 用 rsbench 评估“标签正确但概念错误”的推理捷径问题
- [sources/来源-DS-Agent-案例推理驱动自动化数据科学-ICML2024.md](sources/reasoning-training/%E6%9D%A5%E6%BA%90-DS-Agent-%E6%A1%88%E4%BE%8B%E6%8E%A8%E7%90%86%E9%A9%B1%E5%8A%A8%E8%87%AA%E5%8A%A8%E5%8C%96%E6%95%B0%E6%8D%AE%E7%A7%91%E5%AD%A6-ICML2024.md) — CBR 驱动的数据科学 Agent 闭环
- [sources/来源-多智能体辩论提升事实性与推理-ICML2024.md](sources/reasoning-training/%E6%9D%A5%E6%BA%90-%E5%A4%9A%E6%99%BA%E8%83%BD%E4%BD%93%E8%BE%A9%E8%AE%BA%E6%8F%90%E5%8D%87%E4%BA%8B%E5%AE%9E%E6%80%A7%E4%B8%8E%E6%8E%A8%E7%90%86-ICML2024.md) — 通过多轮辩论提升事实性与推理稳健性
- [sources/来源-LATS-统一推理行动规划-ICML2024.md](sources/reasoning-training/%E6%9D%A5%E6%BA%90-LATS-%E7%BB%9F%E4%B8%80%E6%8E%A8%E7%90%86%E8%A1%8C%E5%8A%A8%E8%A7%84%E5%88%92-ICML2024.md) — 统一 reasoning/acting/planning 的 MCTS Agent 框架
- [sources/来源-Chain-of-Code-代码模拟混合推理-ICML2024.md](sources/reasoning-training/%E6%9D%A5%E6%BA%90-Chain-of-Code-%E4%BB%A3%E7%A0%81%E6%A8%A1%E6%8B%9F%E6%B7%B7%E5%90%88%E6%8E%A8%E7%90%86-ICML2024.md) — 代码执行与 LM 模拟的混合推理范式
- [sources/来源-Thought-Rollback-自适应回滚推理-ICML2024.md](sources/reasoning-training/%E6%9D%A5%E6%BA%90-Thought-Rollback-%E8%87%AA%E9%80%82%E5%BA%94%E5%9B%9E%E6%BB%9A%E6%8E%A8%E7%90%86-ICML2024.md) — 回滚驱动的自适应推理路径修复
- [sources/来源-SELF-DISCOVER-自组装推理结构-NeurIPS2024.md](sources/reasoning-training/%E6%9D%A5%E6%BA%90-SELF-DISCOVER-%E8%87%AA%E7%BB%84%E8%A3%85%E6%8E%A8%E7%90%86%E7%BB%93%E6%9E%84-NeurIPS2024.md) — 任务级自动发现并执行推理结构
- [sources/来源-MR-Ben-元推理基准-System2-NeurIPS2024.md](sources/reasoning-training/%E6%9D%A5%E6%BA%90-MR-Ben-%E5%85%83%E6%8E%A8%E7%90%86%E5%9F%BA%E5%87%86-System2-NeurIPS2024.md) — 用错误定位/错因解释评估 LLM 的元推理能力
- [sources/来源-Buffer-of-Thoughts-思维模板缓冲-NeurIPS2024.md](sources/reasoning-training/%E6%9D%A5%E6%BA%90-Buffer-of-Thoughts-%E6%80%9D%E7%BB%B4%E6%A8%A1%E6%9D%BF%E7%BC%93%E5%86%B2-NeurIPS2024.md) — 通过可复用思维模板缓冲兼顾推理效果与成本
- [sources/来源-GLoRe-全局局部联合修正-ICML2024.md](sources/reasoning-training/%E6%9D%A5%E6%BA%90-GLoRe-%E5%85%A8%E5%B1%80%E5%B1%80%E9%83%A8%E8%81%94%E5%90%88%E4%BF%AE%E6%AD%A3-ICML2024.md) — 结合 ORM/SORM 的全局与局部修正协同框架
- [sources/来源-Reverse-Curriculum-RL-逆课程推理强化-ICML2024.md](sources/reasoning-training/%E6%9D%A5%E6%BA%90-Reverse-Curriculum-RL-%E9%80%86%E8%AF%BE%E7%A8%8B%E6%8E%A8%E7%90%86%E5%BC%BA%E5%8C%96-ICML2024.md) — 从中间状态回退的逆课程强化推理方法
- [sources/来源-AdaProp-自适应传播路径-KDD2024.md](sources/reasoning-training/%E6%9D%A5%E6%BA%90-AdaProp-%E8%87%AA%E9%80%82%E5%BA%94%E4%BC%A0%E6%92%AD%E8%B7%AF%E5%BE%84-KDD2024.md) — 查询相关的 KG 传播路径自适应采样
- [sources/来源-NRN-实体与数值联合推理-KDD2024.md](sources/reasoning-training/%E6%9D%A5%E6%BA%90-NRN-%E5%AE%9E%E4%BD%93%E4%B8%8E%E6%95%B0%E5%80%BC%E8%81%94%E5%90%88%E6%8E%A8%E7%90%86-KDD2024.md) — 将实体与数值分相位编码的复杂查询推理框架
- [sources/来源-可信图神经网络综述-2022.md](sources/applications/%E6%9D%A5%E6%BA%90-%E5%8F%AF%E4%BF%A1%E5%9B%BE%E7%A5%9E%E7%BB%8F%E7%BD%91%E7%BB%9C%E7%BB%BC%E8%BF%B0-2022.md) — 可信 GNN 六维框架导读：鲁棒性、可解释性、隐私、公平性、问责与环境福祉
- [sources/来源-Stable-Diffusion微调技术深度调研-2026-06-23.md](sources/multimodal/%E6%9D%A5%E6%BA%90-Stable-Diffusion%E5%BE%AE%E8%B0%83%E6%8A%80%E6%9C%AF%E6%B7%B1%E5%BA%A6%E8%B0%83%E7%A0%94-2026-06-23.md) — SD 微调技术全景调研（LoRA/DreamBooth/ControlNet/IP-Adapter 等 9 种方法 + 5 大框架对比）
- [sources/来源-Stable-Diffusion架构代表性工作与原理深度解析-2026-06-23.md](sources/multimodal/%E6%9D%A5%E6%BA%90-Stable-Diffusion%E6%9E%B6%E6%9E%84%E4%BB%A3%E8%A1%A8%E6%80%A7%E5%B7%A5%E4%BD%9C%E4%B8%8E%E5%8E%9F%E7%90%86%E6%B7%B1%E5%BA%A6%E8%A7%A3%E6%9E%90-2026-06-23.md) — SD 架构演进深度解析（LDM→SDXL→SD3/MMDiT→FLUX，含关键代码实现）
- [sources/来源-视频透明图层提取技术方案-2026-06-23.md](sources/multimodal/%E6%9D%A5%E6%BA%90-%E8%A7%86%E9%A2%91%E9%80%8F%E6%98%8E%E5%9B%BE%E5%B1%82%E6%8F%90%E5%8F%96%E6%8A%80%E6%9C%AF%E6%96%B9%E6%A1%88-2026-06-23.md) — 视频抠图/层分解完整技术方案（RVM/MODNet/MatAnyone + 生成式前沿）
- [sources/来源-对抗图像样本生成原理与实践-2026-06-23.md](sources/multimodal/%E6%9D%A5%E6%BA%90-%E5%AF%B9%E6%8A%97%E5%9B%BE%E5%83%8F%E6%A0%B7%E6%9C%AC%E7%94%9F%E6%88%90%E5%8E%9F%E7%90%86%E4%B8%8E%E5%AE%9E%E8%B7%B5-2026-06-23.md) — 对抗攻击实践（MPCAttack 92% ASR + 扩散净化防御 + 线上部署代码）
- [sources/来源-风控工程实战案例面试问题答案-2026-06-23.md](sources/applications/%E6%9D%A5%E6%BA%90-%E9%A3%8E%E6%8E%A7%E5%B7%A5%E7%A8%8B%E5%AE%9E%E6%88%98%E6%A1%88%E4%BE%8B%E9%9D%A2%E8%AF%95%E9%97%AE%E9%A2%98%E7%AD%94%E6%A1%88-2026-06-23.md) — 字节电商+TikTok 风控面试复盘（双 Runtime Agent、闪帧治理、团伙挖掘、统一大盘架构）
- [sources/来源-LLM智能体规划综述-USTC-Huawei-2026-06-23.md](sources/agent/%E6%9D%A5%E6%BA%90-LLM%E6%99%BA%E8%83%BD%E4%BD%93%E8%A7%84%E5%88%92%E7%BB%BC%E8%BF%B0-USTC-Huawei-2026-06-23.md) — USTC/华为：LLM 智能体规划五大方向（分解/选择/外部/反思/记忆）系统综述
- [sources/来源-个人LLM智能体综述-Tsinghua-AIR-2026-06-23.md](sources/agent/%E6%9D%A5%E6%BA%90-%E4%B8%AA%E4%BA%BALLM%E6%99%BA%E8%83%BD%E4%BD%93%E7%BB%BC%E8%BF%B0-Tsinghua-AIR-2026-06-23.md) — 清华AIR：个人LLM智能体体系结构、L1-L5智能等级与边-云协同专家共识
- [sources/来源-工具学习与基础模型综述-Tsinghua-2026-06-23.md](sources/agent/%E6%9D%A5%E6%BA%90-%E5%B7%A5%E5%85%B7%E5%AD%A6%E4%B9%A0%E4%B8%8E%E5%9F%BA%E7%A1%80%E6%A8%A1%E5%9E%8B%E7%BB%BC%E8%BF%B0-Tsinghua-2026-06-23.md) — 清华大学主导：基于基础模型的工具学习统一框架（控制器-工具-环境-感知器）及其认知起源、训练策略与安全挑战
- [sources/来源-增强语言模型综述-ALM-MetaAI-2026-06-23.md](sources/language/%E6%9D%A5%E6%BA%90-%E5%A2%9E%E5%BC%BA%E8%AF%AD%E8%A8%80%E6%A8%A1%E5%9E%8B%E7%BB%BC%E8%BF%B0-ALM-MetaAI-2026-06-23.md) — Meta AI（Yann LeCun 参与）：增强语言模型 ALM 综述，推理+工具双重增强打破纯统计语言建模范式
- [sources/来源-LLM多智能体综述-NotreDame-2026-06-23.md](sources/agent/%E6%9D%A5%E6%BA%90-LLM%E5%A4%9A%E6%99%BA%E8%83%BD%E4%BD%93%E7%BB%BC%E8%BF%B0-NotreDame-2026-06-23.md) — 圣母大学主导：LLM 多智能体系统四维度框架（接口/画像/通信/能力获取）与问题求解+世界模拟双模式全景
- [sources/来源-Agent记忆综述-2026.md](sources/agent/%E6%9D%A5%E6%BA%90-Agent%E8%AE%B0%E5%BF%86%E7%BB%BC%E8%BF%B0-2026.md) — Foundation Agent Memory 综述：基底/认知机制/主体三维分类法与自我演进开放挑战（`raw/articles/agent_memory/`）
- [sources/来源-视频审核与电商风控GUI-Agent批次-2026.md](sources/multimodal/%E6%9D%A5%E6%BA%90-%E8%A7%86%E9%A2%91%E5%AE%A1%E6%A0%B8%E4%B8%8E%E7%94%B5%E5%95%86%E9%A3%8E%E6%8E%A7GUI-Agent%E6%89%B9%E6%AC%A1-2026.md) — UNIVID/IPS/VideoModerator 审核演进 + RISK/RiskWebWorld 电商风控 GUI Agent 框架与基准
- [sources/来源-时间序列LLM智能体综述-2026.md](sources/applications/%E6%9D%A5%E6%BA%90-%E6%97%B6%E9%97%B4%E5%BA%8F%E5%88%97LLM%E6%99%BA%E8%83%BD%E4%BD%93%E7%BB%BC%E8%BF%B0-2026.md) — 时序 LLM Agent 综述四分类 + AD-Agent/ARGOS/CALM/SAGE/AgentFM 等子论文（`raw/articles/time_series_agent/`）
- [sources/来源-AD-Agent多Agent异常检测-2025.md](sources/agent/%E6%9D%A5%E6%BA%90-AD-Agent%E5%A4%9AAgent%E5%BC%82%E5%B8%B8%E6%A3%80%E6%B5%8B-2025.md) — 7 Agent + 短/长期记忆，自然语言→可执行 AD 流水线，集成 PyOD/PyGOD/TSLib，BSD-2 开源
- [sources/来源-ARGOS自主规则时序异常检测-2025.md](sources/applications/%E6%9D%A5%E6%BA%90-ARGOS%E8%87%AA%E4%B8%BB%E8%A7%84%E5%88%99%E6%97%B6%E5%BA%8F%E5%BC%82%E5%B8%B8%E6%A3%80%E6%B5%8B-2025.md) — LLM 训练期生成可解释可复现 Python 规则，三 Agent + 模型融合，KPI +9.5% F1
- [sources/来源-CALM流式异常检测LLM裁判-2025.md](sources/applications/%E6%9D%A5%E6%BA%90-CALM%E6%B5%81%E5%BC%8F%E5%BC%82%E5%B8%B8%E6%A3%80%E6%B5%8BLLM%E8%A3%81%E5%88%A4-2025.md) — Apache Beam + TimesFM，LLM-as-a-Judge 漂移检测 + 闭环持续微调
- [sources/来源-SAGE专家式多Agent时序诊断-2026.md](sources/applications/%E6%9D%A5%E6%BA%90-SAGE%E4%B8%93%E5%AE%B6%E5%BC%8F%E5%A4%9AAgent%E6%97%B6%E5%BA%8F%E8%AF%8A%E6%96%AD-2026.md) — 5 阶段管线 + 4 族 9 类分析器 + 合成 ICL（k-medoids+DTW），LLM 打分 temperature=0
- [sources/来源-AgentFM-数据库故障管理Agent-2025.md](sources/agent/%E6%9D%A5%E6%BA%90-AgentFM-%E6%95%B0%E6%8D%AE%E5%BA%93%E6%95%85%E9%9A%9C%E7%AE%A1%E7%90%86Agent-2025.md) — 分布式数据库角色感知故障管理，meta-agent 编排 system/data/task 三类 agent，IoTDB preliminary 验证
- [sources/来源-IoT异常检测LLM-Agent-2025.md](sources/applications/%E6%9D%A5%E6%BA%90-IoT%E5%BC%82%E5%B8%B8%E6%A3%80%E6%B5%8BLLM-Agent-2025.md) — 关键 IoT 基础设施 LLM+XAI 混合异常检测，语义记忆缓冲 + SHAP/LIME 可解释，仿真 95.4% vs 82.1%
- [sources/来源-WeMM多模态嵌入技术报告-2026.md](sources/multimodal/%E6%9D%A5%E6%BA%90-WeMM%E5%A4%9A%E6%A8%A1%E6%80%81%E5%B5%8C%E5%85%A5%E6%8A%80%E6%9C%AF%E6%8A%A5%E5%91%8A-2026.md) — WeChat 通用多模态嵌入模型族（2B/4B/9B）两阶段训练，登顶 MMEB-v2
- [sources/来源-RQ-VAE-残差量化自回归图像生成-2022.md](sources/multimodal/%E6%9D%A5%E6%BA%90-RQ-VAE-%E6%AE%8B%E5%B7%AE%E9%87%8F%E5%8C%96%E8%87%AA%E5%9B%9E%E5%BD%92%E5%9B%BE%E5%83%8F%E7%94%9F%E6%88%90-2022.md) — RQ 把连续表征折叠成短离散码栈的自回归源头，作为物品/用户 Semantic ID 序列化的方法学来源
- [sources/来源-DeepEvidence-生物医学知识图谱深度研究-2026.md](sources/agent/%E6%9D%A5%E6%BA%90-DeepEvidence-%E7%94%9F%E7%89%A9%E5%8C%BB%E5%AD%A6%E7%9F%A5%E8%AF%86%E5%9B%BE%E8%B0%B1%E6%B7%B1%E5%BA%A6%E7%A0%94%E7%A9%B6-2026.md) — 生物医学 deep research 的 KG 驱动多 Agent 框架：BFRS/DFRS、证据图谱与统一 API 工具层
- [sources/来源-开源多模态模型ViT视觉头与风控落地选型报告-2026.md](sources/multimodal/%E6%9D%A5%E6%BA%90-%E5%BC%80%E6%BA%90%E5%A4%9A%E6%A8%A1%E6%80%81%E6%A8%A1%E5%9E%8BViT%E8%A7%86%E8%A7%89%E5%A4%B4%E4%B8%8E%E9%A3%8E%E6%8E%A7%E8%90%BD%E5%9C%B0%E9%80%89%E5%9E%8B%E6%8A%A5%E5%91%8A-2026.md) — 开放 VLM 视觉路线、评测陷阱与风控部署边界：哪些场景适合试点，哪些不宜让 VLM 主裁决
- [sources/来源-图像对抗攻防答辩文档-2026.md](sources/multimodal/%E6%9D%A5%E6%BA%90-%E5%9B%BE%E5%83%8F%E5%AF%B9%E6%8A%97%E6%94%BB%E9%98%B2%E7%AD%94%E8%BE%A9%E6%96%87%E6%A1%A3-2026.md) — 图像对抗攻防项目复盘：把 Illusion/对抗样本失效拆成“视觉头看不见”与“决策层判断错”，并落到增强审核触发器链路
- [sources/来源-LoRA低秩适配原理-2026.md](sources/reasoning-training/%E6%9D%A5%E6%BA%90-LoRA%E4%BD%8E%E7%A7%A9%E9%80%82%E9%85%8D%E5%8E%9F%E7%90%86-2026.md) — LoRA 最小原理骨架：冻结主干，只训练低秩增量 `A/B`，合并部署时几乎不增加推理时延
- [sources/来源-GRPO分组相对策略优化原理-2026.md](sources/reasoning-training/%E6%9D%A5%E6%BA%90-GRPO%E5%88%86%E7%BB%84%E7%9B%B8%E5%AF%B9%E7%AD%96%E7%95%A5%E4%BC%98%E5%8C%96%E5%8E%9F%E7%90%86-2026.md) — 去掉 value head、用组内相对奖励替代优势估计的推理/对齐训练方法说明
- [sources/来源-递归自我改进与Harness自演化-2026.md](sources/agent/%E6%9D%A5%E6%BA%90-%E9%80%92%E5%BD%92%E8%87%AA%E6%88%91%E6%94%B9%E8%BF%9B%E4%B8%8EHarness%E8%87%AA%E6%BC%94%E5%8C%96-2026.md) — RSI、Harness 工程及 ASPIRE/HarnessDev/S3Gym 的批次综合，聚焦可迁移改进的证据边界
- [sources/来源-Dream-RSI-演化世界递归自我改进-2026.md](sources/agent/%E6%9D%A5%E6%BA%90-Dream-RSI-%E6%BC%94%E5%8C%96%E4%B8%96%E7%95%8C%E9%80%92%E5%BD%92%E8%87%AA%E6%88%91%E6%94%B9%E8%BF%9B-2026.md) — 把 discovery history 做成可回放世界，用离线"做梦"廉价评估/优化探索策略，闭合元层 RSI 循环
- [sources/来源-LLM推荐技术范式与落地边界-2026.md](sources/applications/%E6%9D%A5%E6%BA%90-LLM%E6%8E%A8%E8%8D%90%E6%8A%80%E6%9C%AF%E8%8C%83%E5%BC%8F%E4%B8%8E%E8%90%BD%E5%9C%B0%E8%BE%B9%E7%95%8C-2026.md) — LLM4Rec 的语义增强、生成式推荐、智能体编排及其评测和生产约束
- [sources/来源-FOUND可迁移可预测用户定向基础模型-2025.md](sources/applications/%E6%9D%A5%E6%BA%90-FOUND%E5%8F%AF%E8%BF%81%E7%A7%BB%E5%8F%AF%E9%A2%84%E6%B5%8B%E7%94%A8%E6%88%B7%E5%AE%9A%E5%90%91%E5%9F%BA%E7%A1%80%E6%A8%A1%E5%9E%8B-2025.md) — 以未来行为文本对齐历史多源画像的可迁移用户定向基座
- [sources/来源-FOUNDv2统一量化用户Tokenizer-2026.md](sources/applications/%E6%9D%A5%E6%BA%90-FOUNDv2%E7%BB%9F%E4%B8%80%E9%87%8F%E5%8C%96%E7%94%A8%E6%88%B7Tokenizer-2026.md) — 以共享/源特定码本和多尺度预训练压缩多源用户历史
- [sources/来源-Query-as-Anchor场景自适应用户表征-2026.md](sources/applications/%E6%9D%A5%E6%BA%90-Query-as-Anchor%E5%9C%BA%E6%99%AF%E8%87%AA%E9%80%82%E5%BA%94%E7%94%A8%E6%88%B7%E8%A1%A8%E5%BE%81-2026.md) — 以尾部 query 重锚定用户表示，并复用用户前缀 KV cache
- [sources/来源-用户行为致密律与自适应Token化-2026.md](sources/applications/%E6%9D%A5%E6%BA%90-%E7%94%A8%E6%88%B7%E8%A1%8C%E4%B8%BA%E8%87%B4%E5%AF%86%E5%BE%8B%E4%B8%8E%E8%87%AA%E9%80%82%E5%BA%94Token%E5%8C%96-2026.md) — 解释原始行为规模墙、最小充分容量与自适应离散化
- [sources/来源-DecoderOnly用户表征注意力掩码-2026.md](sources/applications/%E6%9D%A5%E6%BA%90-DecoderOnly%E7%94%A8%E6%88%B7%E8%A1%A8%E5%BE%81%E6%B3%A8%E6%84%8F%E5%8A%9B%E6%8E%A9%E7%A0%81-2026.md) — 把 decoder-only 主干作为用户编码器时的掩码适配与 GG-SM
- [sources/来源-InstructUE指令感知用户嵌入-2025.md](sources/applications/%E6%9D%A5%E6%BA%90-InstructUE%E6%8C%87%E4%BB%A4%E6%84%9F%E7%9F%A5%E7%94%A8%E6%88%B7%E5%B5%8C%E5%85%A5-2025.md) — 以指令动态调制用户表示、结合 UserQA 与少样本指令调优
- [sources/来源-TokenMinds-预训练用户Token-2026.md](sources/applications/%E6%9D%A5%E6%BA%90-TokenMinds-%E9%A2%84%E8%AE%AD%E7%BB%83%E7%94%A8%E6%88%B7Token-2026.md) — 用 LLM encoder-decoder + 共享 SID 词表产出离散用户 token 与稠密 embedding，跨长/短视频统一，YouTube 全量上线
- [sources/来源-五模态数据统一对齐技术调研-2026.md](sources/multimodal/%E6%9D%A5%E6%BA%90-%E4%BA%94%E6%A8%A1%E6%80%81%E6%95%B0%E6%8D%AE%E7%BB%9F%E4%B8%80%E5%AF%B9%E9%BD%90%E6%8A%80%E6%9C%AF%E8%B0%83%E7%A0%94-2026.md) — 五模态系统以结构保真、原生编码和任务级接口为主的调研综合
- [sources/来源-RecFormer-P5-M6Rec推荐数据文本化-2026.md](sources/applications/%E6%9D%A5%E6%BA%90-RecFormer-P5-M6Rec%E6%8E%A8%E8%8D%90%E6%95%B0%E6%8D%AE%E6%96%87%E6%9C%AC%E5%8C%96-2026.md) — 区分输入表示、任务接口与输出形态三种推荐“统一”
- [sources/来源-Qwen事件序列分类最小PoC-2026.md](sources/applications/%E6%9D%A5%E6%BA%90-Qwen%E4%BA%8B%E4%BB%B6%E5%BA%8F%E5%88%97%E5%88%86%E7%B1%BB%E6%9C%80%E5%B0%8FPoC-2026.md) — 事件序列分类的无泄漏 PoC、强基线和停止条件
- [sources/来源-Qwen事件序列与统计特征融合方案-2026.md](sources/applications/%E6%9D%A5%E6%BA%90-Qwen%E4%BA%8B%E4%BB%B6%E5%BA%8F%E5%88%97%E4%B8%8E%E7%BB%9F%E8%AE%A1%E7%89%B9%E5%BE%81%E8%9E%8D%E5%90%88%E6%96%B9%E6%A1%88-2026.md) — 从文本化、软 token 到深融合的 Qwen 异构输入工程门禁
- [sources/来源-LLM推荐向风控迁移机制与验证路径-2026.md](sources/applications/%E6%9D%A5%E6%BA%90-LLM%E6%8E%A8%E8%8D%90%E5%90%91%E9%A3%8E%E6%8E%A7%E8%BF%81%E7%A7%BB%E6%9C%BA%E5%88%B6%E4%B8%8E%E9%AA%8C%E8%AF%81%E8%B7%AF%E5%BE%84-2026.md) — 推荐机制向风控的受控迁移：离线教师、在线校准和策略执行分离
- [sources/来源-多模态统一嵌入调研批次-2026.md](sources/applications/%E6%9D%A5%E6%BA%90-%E5%A4%9A%E6%A8%A1%E6%80%81%E7%BB%9F%E4%B8%80%E5%B5%8C%E5%85%A5%E8%B0%83%E7%A0%94%E6%89%B9%E6%AC%A1-2026.md) — 本轮多模态结构对齐、推荐文本化、Qwen 融合与风控迁移的专题导航

## 领域知识 — 对比（Comparisons）

- [comparisons/claude-code-curated-docs-vs-gstack.md](comparisons/agent/claude-code-curated-docs-vs-gstack.md) — 两套 Claude Code 资料/工作流对照
- [comparisons/表格建模路线对比-TabLLM-vs-树模型-vs-Transformer.md](comparisons/applications/%E8%A1%A8%E6%A0%BC%E5%BB%BA%E6%A8%A1%E8%B7%AF%E7%BA%BF%E5%AF%B9%E6%AF%94-TabLLM-vs-%E6%A0%91%E6%A8%A1%E5%9E%8B-vs-Transformer.md) — TabLLM、树模型与表格 Transformer 的能力/代价对照
- [comparisons/主流Agent框架对比-2026-06-20.md](comparisons/agent/%E4%B8%BB%E6%B5%81Agent%E6%A1%86%E6%9E%B6%E5%AF%B9%E6%AF%94-2026-06-20.md) — 基于 `raw/search/` 批次的主流 Agent 框架工程维度对照（含可打分占位表）
- [comparisons/多Agent论文方向对比-2026H1-arXiv.md](comparisons/agent/%E5%A4%9AAgent%E8%AE%BA%E6%96%87%E6%96%B9%E5%90%91%E5%AF%B9%E6%AF%94-2026H1-arXiv.md) — 2026H1 多Agent arXiv 论文方向分层与工程关联
- [comparisons/传统流量反作弊与Agent流量治理对比.md](comparisons/agent/%E4%BC%A0%E7%BB%9F%E6%B5%81%E9%87%8F%E5%8F%8D%E4%BD%9C%E5%BC%8A%E4%B8%8EAgent%E6%B5%81%E9%87%8F%E6%B2%BB%E7%90%86%E5%AF%B9%E6%AF%94.md) — 传统广告反作弊与 Agent 调用链治理的差异边界（含可打分占位表）
- [comparisons/工作流与Agent任务分流对比-2026-06-21.md](comparisons/agent/%E5%B7%A5%E4%BD%9C%E6%B5%81%E4%B8%8EAgent%E4%BB%BB%E5%8A%A1%E5%88%86%E6%B5%81%E5%AF%B9%E6%AF%94-2026-06-21.md) — 基于四问框架比较 workflow 与 agent 的适用条件与治理代价
- [comparisons/评测污染与推理捷径风险对照-2026-06-22.md](comparisons/reasoning-training/%E8%AF%84%E6%B5%8B%E6%B1%A1%E6%9F%93%E4%B8%8E%E6%8E%A8%E7%90%86%E6%8D%B7%E5%BE%84%E9%A3%8E%E9%99%A9%E5%AF%B9%E7%85%A7-2026-06-22.md) — 对照 benchmark contamination 与 reasoning shortcuts 的风险来源与治理手段
- [comparisons/推理框架六法对照-2026-06-22.md](comparisons/reasoning-training/%E6%8E%A8%E7%90%86%E6%A1%86%E6%9E%B6%E5%85%AD%E6%B3%95%E5%AF%B9%E7%85%A7-2026-06-22.md) — 对照十二类推理/修正/评测路径的结构、纠错信号与成本边界
- [comparisons/时序异常检测方法对照-2026-08-31.md](comparisons/applications/%E6%97%B6%E5%BA%8F%E5%BC%82%E5%B8%B8%E6%A3%80%E6%B5%8B%E6%96%B9%E6%B3%95%E5%AF%B9%E7%85%A7-2026-08-31.md) — ARGOS/CALM/SAGE/AD-Agent 在 LLM 介入程度、可解释性、可复现性、自主性、开源上的设计取舍与风控治理同构
- [comparisons/多模态嵌入基座对比-2026-08-31.md](comparisons/multimodal/%E5%A4%9A%E6%A8%A1%E6%80%81%E5%B5%8C%E5%85%A5%E5%9F%BA%E5%BA%A7%E5%AF%B9%E6%AF%94-2026-08-31.md) — WeMM vs CLIP/Gemini Embedding 的演进路线、规模-性能前沿与选型建议（证据层级不均，Gemini/CLIP 为 E3 外部参照）
- [comparisons/图像对抗防御链路对照-2026-09-02.md](comparisons/multimodal/%E5%9B%BE%E5%83%8F%E5%AF%B9%E6%8A%97%E9%98%B2%E5%BE%A1%E9%93%BE%E8%B7%AF%E5%AF%B9%E7%85%A7-2026-09-02.md) — 把 SMSP / SegFormer / 对抗检测触发器 / 重模型二审 / 最终裁决器放回真实审核链路，比较各自解决的失效与上线定位
- [comparisons/多Agent论文方向对比-2026H1-arXiv.md](comparisons/agent/%E5%A4%9AAgent%E8%AE%BA%E6%96%87%E6%96%B9%E5%90%91%E5%AF%B9%E6%AF%94-2026H1-arXiv.md) — 增补 DeepEvidence 作为“共享证据图谱记忆 + 统一 KG 工具层 + 代码沙箱”的垂域多 Agent 样本
- [comparisons/参数高效微调与对齐训练方法对照-2026-09-03.md](comparisons/reasoning-training/%E5%8F%82%E6%95%B0%E9%AB%98%E6%95%88%E5%BE%AE%E8%B0%83%E4%B8%8E%E5%AF%B9%E9%BD%90%E8%AE%AD%E7%BB%83%E6%96%B9%E6%B3%95%E5%AF%B9%E7%85%A7-2026-09-03.md) — 把 SFT、LoRA/QLoRA、DPO、GRPO 放回同一训练栈，比较各自解决的问题、成本与误用风险
- [comparisons/LLM推荐三种系统范式对比-2026-09-11.md](comparisons/applications/LLM%E6%8E%A8%E8%8D%90%E4%B8%89%E7%A7%8D%E7%B3%BB%E7%BB%9F%E8%8C%83%E5%BC%8F%E5%AF%B9%E6%AF%94-2026-09-11.md) — 对照特征增强、受约束生成式推荐与推荐智能体的职责、风险和上线门槛
- [comparisons/异构数据接入大模型的四档方案对比-2026.md](comparisons/multimodal/%E5%BC%82%E6%9E%84%E6%95%B0%E6%8D%AE%E6%8E%A5%E5%85%A5%E5%A4%A7%E6%A8%A1%E5%9E%8B%E7%9A%84%E5%9B%9B%E6%A1%A3%E6%96%B9%E6%A1%88%E5%AF%B9%E6%AF%94-2026.md) — 对照文本化、软 token、独立编码器投影与端到端深融合的门禁和停止条件

## 领域知识 — 地图（Maps）

- [maps/README.md](maps/README.md) — 主题导航图放置说明（占位；可随摄入增加领域地图）
- [maps/研究主题导航.md](maps/%E7%A0%94%E7%A9%B6%E4%B8%BB%E9%A2%98%E5%AF%BC%E8%88%AA.md) — 现有主题关系总览（蒸馏/多语/协同/治理/表格）
- [maps/表格建模选型决策树.md](maps/%E8%A1%A8%E6%A0%BC%E5%BB%BA%E6%A8%A1%E9%80%89%E5%9E%8B%E5%86%B3%E7%AD%96%E6%A0%91.md) — 按样本量、预算与线上约束选择建模路线
- [maps/多Agent研究与工程落地地图.md](maps/%E5%A4%9AAgent%E7%A0%94%E7%A9%B6%E4%B8%8E%E5%B7%A5%E7%A8%8B%E8%90%BD%E5%9C%B0%E5%9C%B0%E5%9B%BE.md) — 连接多Agent论文趋势与框架工程实践的主题地图
- [maps/研究主题导航.md](maps/%E7%A0%94%E7%A9%B6%E4%B8%BB%E9%A2%98%E5%AF%BC%E8%88%AA.md) — 主题关系总览，已补入 Agent 记忆 / 时序 Agent / 多模态嵌入 / 视频审核与电商风控 GUI Agent 批次 / 知识图谱驱动 deep research，以及 PEFT-对齐训练分层节点

## 运维

- [log.md](log.md) — 仅追加的操作时间线
