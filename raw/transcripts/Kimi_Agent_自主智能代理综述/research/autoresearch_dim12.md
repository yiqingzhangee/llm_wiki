# 维度12：中国市场与国产方案

## 目录
1. [执行摘要](#1-执行摘要)
2. [通义DeepResearch（阿里巴巴）](#2-通义deepresearch阿里巴巴)
3. [Kimi Researcher（Moonshot AI）](#3-kimi-researchermoonshot-ai)
4. [字节跳动DeerFlow](#4-字节跳动deerflow)
5. [DeepSeek系列](#5-deepseek系列)
6. [其他国产方案](#6-其他国产方案)
7. [中国监管与生态](#7-中国监管与生态)
8. [综合比较与国际定位](#8-综合比较与国际定位)
9. [反面观点与争议](#9-反面观点与争议)
10. [关键引用来源汇总](#10-关键引用来源汇总)

---

## 1. 执行摘要

中国在自主研究Agent（Autonomous Research Agent）领域已形成完整的"模型-框架-产品-监管"四层生态体系。以阿里巴巴Tongyi DeepResearch、Moonshot AI Kimi-Researcher、字节跳动DeerFlow 2.0、智谱AutoGLM沉思为代表的国产方案，在深度研究基准（BrowseComp、GAIA、HLE）上已达到或超越OpenAI和Google的同类系统。与此同时，DeepSeek系列通过开源策略和极致性价比重塑了全球AI市场格局。中国的监管环境以备案制为核心，强调算法透明度和内容标识，形成了与美国、欧洲不同的治理路径。

---

## 2. 通义DeepResearch（阿里巴巴）

### 2.1 技术架构

Tongyi DeepResearch由通义实验室（Tongyi Lab）开发，基于**Qwen3-30B-A3B**基座模型构建。其核心架构特征包括：

- **稀疏混合专家（Sparse MoE）架构**：总参数量305亿，每Token仅激活33亿参数，实现小模型推理成本与大模型容量的平衡 [^471^]
- **上下文窗口**：128K tokens，支持长程信息收集与证据累积 [^859^]
- **端到端Agent训练**：并非简单的聊天LLM，而是作为完整Agent从零开始训练 [^163^]

### 2.2 训练方法

训练流程分为三个阶段，形成闭环数据飞轮：

**阶段1：Agentic Continual Pre-training (CPT)**
大规模合成轨迹数据，涵盖多步问题-推理-行动-决策过程，部分轨迹长达128K tokens。通过全自动数据合成管道生成多样化研究问题和计划，无需昂贵的人工标注 [^859^]。

**阶段2：Agentic SFT Cold-Start**
使用ReAct和IterResearch格式的轨迹数据进行监督微调，实现模式一致的规划和工具使用 [^875^]。

**阶段3：On-Policy Reinforcement Learning**
采用严格在线策略（on-policy）的强化学习方法，基于**定制的Group Relative Policy Optimization (GRPO)**框架：
- Token-level policy gradients（token级策略梯度）
- Leave-one-out advantage estimation（留一法优势估计）
- Selective filtering of negative samples（负样本选择性过滤），用于稳定非平稳Web环境下的训练 [^471^] [^859^]

### 2.3 推理范式

Tongyi DeepResearch支持两种推理范式：

1. **ReAct Mode**：原生推理-行动循环，无需提示词工程，用于严格评估模型的核心内在能力 [^859^]
2. **IterResearch-based "Heavy" Mode**：通过测试时扩展（test-time scaling）解锁最大性能上限。针对多步研究任务，将其解构为一系列研究回合，每轮中Agent仅使用上一轮最重要的输出重建精简工作空间 [^871^]

### 2.4 基准表现

Tongyi DeepResearch在2025年9月发布的基准测试中取得了开源深度研究Agent的SOTA成绩 [^471^]：

| Benchmark | Tongyi DeepResearch (30B-A3B) | OpenAI o3 | DeepSeek-V3.1 | Kimi K2 |
|-----------|------------------------------|-----------|---------------|---------|
| Humanity's Last Exam | **32.9** | 24.9 | 29.8 | 18.1 |
| BrowseComp | **43.4** | 49.7 | 30.0 | 14.1 |
| BrowseComp-ZH | **46.7** | 58.1 | 49.2 | 28.8 |
| GAIA | **70.9** | — | 63.1 | 57.7 |
| xbench-DeepSearch | **75.0** | 67.0 | 71.0 | 50.0 |
| WebWalkerQA | **72.2** | 71.7 | 61.2 | 63.0 |
| FRAMES | **90.6** | 84.0 | 83.7 | 72.0 |
| xbench-DeepSearch-2510 | 55.0 | — | — | — |

**关键发现**：Tongyi DeepResearch在HLE、GAIA、xbench-DeepSearch、WebWalkerQA、FRAMES等多个基准上超越了OpenAI o3，且在BrowseComp上超过了DeepSeek-V3.1。这些成果以仅33亿激活参数实现，展示了卓越的效率优势 [^471^]。

### 2.5 开源策略

- 模型权重、训练代码、推理脚本全部开源，托管于Hugging Face和ModelScope [^857^] [^873^]
- GitHub仓库包含完整的复现脚本、工具实现和提示配置 [^471^]
- API定价：输入$0.09/M tokens，输出$0.40/M tokens（OpenRouter平台）[^810^]

```
Claim: Tongyi DeepResearch在HLE基准上达到32.9分，超越OpenAI o3（24.9分）和DeepSeek-V3.1（29.8分），且仅以30B总参数/3.3B激活参数实现 [^471^]
Source: arXiv - Tongyi DeepResearch Technical Report
URL: https://arxiv.org/html/2510.24701v1
Date: 2025-10-28
Excerpt: "Tongyi DeepResearch reaches 32.9 on Humanity's Last Exam, 43.4 on BrowseComp, 46.7 on BrowseComp-ZH, 72.2 on WebWalkerQA, 70.9 on GAIA, 75.0 on xbench-DeepSearch, 90.6 on FRAMES and 55.0 on xbench-DeepSearch-2510, outperforming strong baselines such as OpenAI-o3 and Deepseek-V3.1"
Context: 官方技术报告，对比了LLM-based ReAct Agent和DeepResearch Agent两类基线
Confidence: high
```

---

## 3. Kimi Researcher（Moonshot AI）

### 3.1 产品定位

Kimi-Researcher是Moonshot AI（月之暗面）推出的首款自主Agent产品，于**2025年6月21日**开启小范围灰度测试 [^853^] [^854^]。其核心定位：

- **端到端自主强化学习训练**：基于end-to-end agentic RL技术训练的新一代Agent模型 [^855^]
- **深度研究专用**：专为深度研究任务而生的Agent产品，擅长多轮搜索与推理 [^854^]
- **可追溯报告**：可生成易追溯的万字报告 [^855^]

### 3.2 技术底座：Kimi K2

Kimi-Researcher基于Kimi K2模型构建，Kimi K2是Moonshot AI于2025年7月发布的开源基础模型 [^861^]：

- **架构**：1万亿参数MoE架构，每Token激活约320亿参数 [^805^]
- **长文本能力**：超长上下文窗口是Kimi的核心差异化优势，通过Kimi Delta Attention (KDA)和上下文缓存技术管理内存开销 [^805^]
- **Agent Swarm**：Kimi K2.5支持同时生成和协调最多100个并行AI Agent，可将复杂多阶段任务执行时间缩短最多4.5倍 [^805^]

### 3.3 基准表现

Kimi-Researcher在HLE（Humanity's Last Exam）基准测试中Pass@1得分率达**26.9%**，超越了Google和OpenAI的同类产品 [^864^]。在其他深度研究基准如xbench-DeepSearch上得分69.0，FRAMES得分78.8 [^471^]。

### 3.4 开源与未来计划

Moonshot AI计划未来开源Kimi-Researcher的基础预训练模型及强化学习后的模型，展示端到端智能体强化学习的潜力 [^854^]。Kimi K2基础模型本身已开源，训练成本据称为460万美元 [^864^]。

### 3.5 差异化定位

相比通义DeepResearch的技术导向，Kimi Researcher更强调：
- **用户体验**：生成可追溯、可验证的研究报告
- **长文本原生优势**：利用Moonshot在长上下文领域的技术积累
- **端到端RL**：完全基于强化学习训练Agent行为，而非通过提示工程

```
Claim: Kimi-Researcher基于端到端自主强化学习训练，在HLE基准上Pass@1达26.9%，超越Google和OpenAI产品 [^864^]
Source: Readhub / ITBear 科技资讯
URL: https://readhub.cn/topic/8kPhoDUVwT6
Date: 2025-06-23
Excerpt: "月之暗面推出Kimi研究功能，核心为Kimi-Researcher智能体，擅长多轮搜索与推理。该功能在「人类终极考验」中超越谷歌和OpenAI产品，Pass@1得分率达26.9%"
Context: 多家科技媒体集中报道
Confidence: high
```

---

## 4. 字节跳动DeerFlow

### 4.1 产品演进

DeerFlow（Deep Exploration and Efficient Research Flow）经历了从研究框架到SuperAgent Harness的范式转变 [^798^]：

- **v1时代**：专注于深度研究任务的框架——用户提问、搜索网络、综合、回答 [^800^]
- **v2.0时代**：2026年2月27日开源的完全重写版本，与v1无共用代码。定位升级为"SuperAgent Harness"——开箱即用、高度可扩展的AI Agent运行时基础设施 [^800^] [^801^]

### 4.2 技术架构

DeerFlow 2.0基于以下核心技术栈构建 [^801^]：

| 技术栈 | 作用 |
|--------|------|
| **LangGraph** | 多Agent编排与工作流管理 |
| **LangChain** | LLM交互与Chains |
| **Docker** | 隔离执行环境（Sandbox） |
| **Python** | 后端核心语言 |
| **Node.js 22+** | 前端服务 |

### 4.3 核心能力

**SuperAgent Harness定位** [^798^] [^800^]：
- **子Agent编排**：主Agent可动态生成子Agent，每个子Agent拥有独立的作用域上下文、工具和终止条件。子Agent尽可能并行运行，返回结构化结果后由主Agent综合输出
- **持久记忆**：支持长期记忆管理
- **沙箱执行**：提供隔离的Docker容器，包含完整文件系统、bash终端，Agent可实际运行代码而非仅生成建议
- **可扩展Skills**：通过Markdown文件定义工作流、最佳实践和支持资源。Skills按需渐进加载，保持上下文窗口精简
- **多通道集成**：支持Telegram、Slack、飞书/Lark等IM渠道接收任务 [^798^]

### 4.4 社区反响

- 2026年2月28日登上**GitHub Trending #1** [^799^] [^801^]
- 发布后24小时内积累超过35,000 stars [^806^] [^808^]
- 一个月内超过39,000 stars和4,600 forks [^806^]
- AI研究者Brian Roemmele评价其为"paradigm shift"，称其"绝对碾压我们测试过的任何产品" [^806^]

### 4.5 与豆包生态的协同

字节跳动在Agent领域的布局是多点并进的 [^840^]：
- **豆包大模型**：2025年6月推出豆包1.6系列，国内首个支持256K上下文的思考模型，成本较前代骤降63%
- **Agent TARS**：2025年3月开源的多模态AI智能体框架，专注视觉理解与工具集成
- **扣子空间（Coze Space）**：2025年4月推出的通用型AI Agent开发与协同办公平台，支持探索/规划双模式、专家Agent生态、MCP扩展 [^869^] [^833^]
- **豆包"深入研究"功能**：2025年6月30日上线，类似Agent能力，输入指令后几分钟生成报告，支持可视化网页和报告文档两种方式 [^834^]

```
Claim: DeerFlow 2.0在发布后24小时内登上GitHub Trending第一名，一个月内积累39,000+ stars [^806^]
Source: TechBuddies / deeplearning.ai The Batch
URL: https://www.techbuddies.io/2026/03/25/deerflow-2-0-bytedances-open-source-superagent-harness-and-its-enterprise-tradeoffs/
Date: 2026-03-25
Excerpt: "Within a month, the repository had accumulated more than 39,000 stars and 4,600 forks on GitHub — unusually fast traction even by AI tooling standards."
Context: 多因素驱动：媒体报道、意见领袖背书、企业级定位
Confidence: high
```

---

## 5. DeepSeek系列

### 5.1 DeepSeek-V3.1 / V3.2 / V4

DeepSeek-V3.1是DeepSeek的混合推理与Agent过渡版本，关键特性包括 [^804^] [^873^]：

- **混合推理**：一个模型同时支持思考（reasoning）和非思考（non-thinking）模式
- **Agent行为增强**：更强的Agent行为和改进的工具使用性能
- **Function Calling**：非推理模式下支持结构化JSON输出和严格函数调用 [^877^]
- **上下文窗口**：128K tokens [^877^]

**DeepSeek-V3.1-Terminus**改进 [^873^]：
- BrowseComp从30.0提升至38.5
- SimpleQA从93.4提升至96.8
- SWE Verified从66.0提升至68.4
- 函数调用能力显著增强，更适合CrewAI等Agent框架

**DeepSeek-V4系列**（2026年4月发布）[^844^]：
- V4-Pro：1.6T总参数，49B激活参数，1M上下文
- V4-Flash：284B总参数，13B激活参数，更低成本API使用
- 华为昇腾全系列产品已全面适配V4-Flash和V4-Pro

### 5.2 DeepSeek-Prover-V2

DeepSeek-Prover-V2是专门用于Lean 4形式化定理证明的开源模型 [^844^] [^855^]：

- **架构基础**：基于DeepSeek-V3-Base-671B进行监督微调 [^844^]
- **训练方法**：
  - 专家迭代（Expert Iteration）：当前最优证明器策略生成证明尝试，成功尝试经Lean验证后纳入SFT数据集训练改进模型
  - 强化学习用于子目标分解（subgoal decomposition）
  - 递归定理证明管道：将复杂定理分解为可管理的组件 [^855^]
- **规模**：提供7B和671B两种参数版本 [^845^]
- **基准表现** [^840^] [^894^]：
  - MiniF2F-test：88.9%通过率
  - PutnamBench：49/658问题解决
  - ProverBench（新基准）：包含325个形式化问题，包括15道AIME竞赛题，成功解决6道 [^855^]

### 5.3 DeepSeekMath-V2

DeepSeekMath-V2于2025年11月发布，聚焦**自可验证数学推理**（Self-Verifiable Mathematical Reasoning）[^886^] [^887^]：

- **核心创新**：训练准确的LLM-based验证器（verifier）作为奖励模型，激励证明生成器在最终定稿前识别并解决自身证明中的问题
- **验证-生成差距维护**：随着生成器变强，自动扩展验证计算以标记新的难验证证明，创建训练数据进一步改进验证器
- **基准表现** [^886^]：
  - IMO 2025：金牌级别表现
  - CMO 2024：金牌级别表现
  - Putnam 2024：118/120（接近完美）

### 5.4 DeepSeek的开源影响力

DeepSeek通过开源策略重塑了全球AI格局 [^829^] [^831^]：
- 2025年11月中国AI模型全球市场份额约15%，一年前仅1%
- 在92个主要模型的日语评分中，DeepSeek排名第9，开源模型全球第一 [^829^]
- 日本企业开发的前10个模型中，有6个基于DeepSeek或Qwen开发 [^831^]
- DeepSeek-R1的发布引发全球对低成本高性能AI的关注，迫使OpenAI和百度调整定价策略 [^834^]

```
Claim: DeepSeek-Prover-V2在MiniF2F-test上达到88.9%通过率，在PutnamBench上解决49/658问题，并开源了7B和671B两种规模 [^844^] [^855^]
Source: arXiv / InfoQ
URL: https://arxiv.org/html/2504.21801v2 / https://www.infoq.com/news/2025/05/deepseek-prover-v2-formal-proof/
Date: 2025-05-12
Excerpt: "DeepSeek-Prover-V2-671B demonstrates superior performance across all evaluation metrics, successfully solving 28.31% of problems in the full dataset and 56.00% of problems in the Lite subset with 32 samples."
Context: 形式化数学推理领域的SOTA开源模型
Confidence: high
```

---

## 6. 其他国产方案

### 6.1 智谱AutoGLM沉思

智谱AI于**2025年3月31日**在中关村论坛发布AutoGLM沉思，定位为"全球首个集深度研究与实际操作能力于一体的Agent" [^822^] [^832^]。

**技术演进路径** [^821^] [^827^]：
1. **GLM-4-Air-0414基座模型**：320亿参数，专为智能体任务优化
2. **GLM-Z1-Air推理模型**：消费级显卡可运行的深度思考模型
3. **GLM-Z1-Rumination沉思模型**：通过强化学习训练，提升结合工具使用完成长程推理的能力
4. **AutoGLM执行模型**：GUI阅读能力和设备操作能力

**核心能力** [^822^]：
- 深度思考：模拟人类面对复杂问题时的推理与决策
- 感知世界：像人一样获取并理解环境信息
- 工具使用：像人一样调用和操作工具，完成复杂任务
- 免费且不限量开放

**AutoGLM开源版**（2025年12月）[^816^] [^824^]：
- 开源核心AI Agent模型，具备"Phone Use"能力
- 通过"视觉理解+智能规划+ADB控制"架构实现自动化操作
- 可稳定完成外卖点单、机票预订等数十步复杂操作
- Device Use基准测试表现优于ChatGPT Agent、UI-TARS-1.5和Claude Sonnet 4 [^816^]

### 6.2 百度文心一言

百度在深度研究领域的布局包括 [^820^] [^834^]：

- **深度搜索功能**：2025年2月上线，定位为"专家级内容回复"，支持思考规划、工具调用、多场景多模态输入输出 [^820^]
- **文心4.5开源**：2025年6月30日开源，是中国主要大模型厂商的重要开源举措 [^834^]
- **文心一言全面免费**：2025年4月1日起PC端和移动端全面免费 [^834^]
- **心响App**：2025年4月发布的移动端Agent产品，聚焦通过自然语言交互帮助用户实现复杂任务拆解 [^833^]

### 6.3 华为盘古

华为盘古大模型采用**B端行业导向**策略，2025年6月发布盘古5.5 [^836^] [^838^] [^842^]：

**核心升级** [^841^]：
- NLP模型升级为718B深度思考MoE模型，推理效率提升8倍
- 视觉模型推出300亿参数MoE架构，为业界最大视觉模型
- 多模态模型为智能驾驶、具身智能机器人训练构建数字物理空间

**研究Agent相关能力** [^838^]：
- **盘古DeepDiver**：开放域信息获取Agent，在网页搜索、常识问答等应用中，能让盘古7B大模型实现接近DeepSeek-R1的效果。可在5分钟内完成超10跳复杂问答并生成万字以上专业调研报告
- **行业Thinking大模型**：发布盘古医学、金融、政务、工业、汽车五个具备深度思考能力的行业模型

**具身智能** [^836^]：
- 盘古大模型5.0能让机器人完成10步以上复杂任务规划
- CloudRobo具身智能平台整合盘古多模态能力与边缘计算
- 已在30多个行业、500多个场景中落地 [^842^]

### 6.4 MiniMax-M1

MiniMax于2025年6月发布MiniMax-M1，被称为"世界上第一个开源的大规模混合架构推理模型" [^838^]：
- 整个强化学习阶段仅使用512块英伟达H800 GPU，耗时三周，成本53.5万美元
- 开源性能比肩DeepSeek-R1，算力成本仅380万人民币

### 6.5 MindDeepResearch（旷视/第三方）

MindDR-v1.5-30B-A3B在2026年4月的技术报告中展示了与Tongyi DeepResearch竞争的性能 [^809^]：
- BrowseComp-ZH: 45.7（超越Tongyi-DR的43.2）
- BrowseComp: 42.8（超越Tongyi-DR的40.7）
- GAIA-DS: 70.9（与Tongyi-DR持平）

```
Claim: 智谱AutoGLM沉思是全球首个集深度研究与实际操作于一体的Agent，核心链路模型于2025年4月14日开源 [^822^]
Source: 证券日报 / 智谱官网
URL: https://news.10jqka.com.cn/20250331/c667121467.shtml
Date: 2025-03-31
Excerpt: "这款全新智能体不仅具备深度研究能力（Deep Research），还能实现实际操作（Operator），可以像人类一样打开并浏览网页，完成从数据检索、分析到生成报告"
Context: 2025中关村论坛正式发布
Confidence: high
```

---

## 7. 中国监管与生态

### 7.1 备案制度

中国建立了以**备案制度**为核心的生成式AI多层次治理格局 [^826^] [^828^]：

**法律法规框架** [^817^] [^823^]：
| 文件名称 | 施行时间 | 核心要求 |
|---------|---------|---------|
| 互联网信息服务算法推荐管理规定 | 2022.3.1 | 算法备案 |
| 互联网信息服务深度合成管理规定 | 2023.1.10 | 深度合成服务标识 |
| 生成式人工智能服务管理暂行办法 | 2023.8.15 | 生成式AI服务备案 |
| 人工智能生成合成内容标识办法 | 2025.9.1 | 显式+隐式双重标识 |
| 人工智能科技伦理审查与服务办法（试行）| 2026.3 | 科技伦理审查 |

**备案现状** [^828^]：
- 截至2025年12月31日，累计有**748款**生成式AI服务完成备案
- 435款生成式AI应用或功能完成登记
- 2025年全年新增446款完成备案，330款完成登记

**未备案的法律后果** [^817^]：
- 依据《网络安全法》《数据安全法》《个人信息保护法》等上位法处罚
- 未规定的由主管部门给予警告、通报批评、责令改正
- 拒不改正或情节严重的，可责令暂停服务
- 构成违反治安管理或犯罪的，依法追究刑事责任

### 7.2 数据与隐私合规

**个人信息保护** [^836^]：
- 中国通过《个人信息保护法》确立以"告知-同意"为核心的严格数据处理规则
- 强调特定场景下的"单独同意"和贯穿始终的"最小必要"原则
- 对需要持续、动态处理大量个人信息的AI Agent提出极高合规要求

**算法透明度** [^836^]：
- 中国建立了明确的算法备案制度
- 具有舆论属性或社会动员能力的服务必须提交算法原理、训练数据、模型等核心信息
- 为闭源商业Agent和基于开源模型二次开发的Agent带来不同合规挑战

### 7.3 AI生成内容标识

《人工智能生成合成内容标识办法》于**2025年9月1日**正式施行 [^867^] [^875^]：

**核心要求** [^866^] [^869^]：
- **显式标识**：在内容页面插入文字、声音或图形等可被用户明显感知的标识
- **隐式标识**：在文件元数据中嵌入包含服务提供者名称/编码、内容编号等信息的不可见标识
- **传播平台责任**：应用上架或上线审核时需核验"是否含AI生成功能"和"是否具备完整标识"

**对Agent的影响** [^879^]：
- Agent生成的研究报告、摘要等内容需添加标识
- 平台在内容上架时需进行审核，对未标识或疑似生成内容添加风险提示
- 服务提供者需在用户协议中明确说明标识方法

### 7.4 开源生态

**全球开源格局变化** [^832^]：
- 截至2026年3月，阿里云Qwen系列占全球开源模型下载量的**50%以上**，累计下载量接近10亿次
- 2026年2月单月，Qwen在Hugging Face平台产生约1.536亿次下载，超过Meta、DeepSeek等八个主要发布方总和
- 中国开源模型月下载占比达到41%，美国36.5%，中国首次超越美国 [^832^]

**中国开源第一梯队** [^845^]：
- 第一梯队：DeepSeek和Qwen
- 第二梯队：字节跳动豆包、智谱GLM
- 趋势：强调成本效益的MoE架构、混合推理模式和行业特定适应

**Qwen生态规模** [^832^] [^847^]：
- 已开源超过400款模型
- 累计衍生模型突破20万个
- 全球开发者生态规模已超越Meta Llama系列

```
Claim: 截至2026年3月，Qwen系列占全球开源模型下载量50%以上，中国开源模型月下载占比41%首次超越美国36.5% [^832^]
Source: Interconnects AI / 199IT
URL: https://www.199it.com/archives/1821424.html
Date: 2026-04-12
Excerpt: "仅2026年2月单月，Qwen模型在Hugging Face平台产生了约1.536亿次下载量，超过了包括Meta、DeepSeek和OpenAI在内的另外八个主要模型发布方的下载量总和。"
Context: ATOM报告多维度分析
Confidence: high
```

---

## 8. 综合比较与国际定位

### 8.1 国产研究Agent对比矩阵

| 维度 | Tongyi DeepResearch | Kimi Researcher | DeerFlow 2.0 | AutoGLM沉思 | 盘古DeepDiver |
|------|--------------------|-----------------|--------------|-------------|--------------|
| **开发方** | 阿里通义实验室 | Moonshot AI | 字节跳动 | 智谱AI | 华为云 |
| **基座模型** | Qwen3-30B-A3B | Kimi K2系列 | 模型无关 | GLM-4-Air | 盘古7B |
| **参数规模** | 30B总/3.3B激活 | 1T总/32B激活 | 模型无关 | 32B | 7B |
| **开源程度** | 全开源（模型+代码） | 计划开源 | 全开源（框架） | 核心链路开源 | 闭源 |
| **核心定位** | 深度研究Agent | 深度研究Agent | SuperAgent Harness | 研究+操作 | 开放域信息获取 |
| **架构特色** | MoE + 端到端RL | MoE + 长上下文 | LangGraph + Docker沙箱 | 多模型协同 | 渐进式RL |
| **多Agent** | IterResearch并行 | Agent Swarm (100个) | 子Agent并行 | 单Agent执行 | 单Agent |
| **HLE分数** | **32.9** | 26.9 | — | — | — |
| **发布时间** | 2025.09 | 2025.06 | 2026.02 | 2025.03 | 2025.06 |

### 8.2 与国际竞品的对比

**深度研究基准竞争** [^471^] [^889^]：
- 在BrowseComp上，Tongyi DeepResearch Heavy变体可达69%，与OpenAI DeepResearch和o3相当
- GLM-4.5 Heavy作为开源系统，在多个基准上与OpenAI DeepResearch和o3持平
- 通过测试时扩展（test-time scaling），开源中国模型可实现与商业前沿系统相当的性能

**效率优势** [^471^]：
- Tongyi DeepResearch以33亿激活参数实现了超越o3（估计数千亿参数）的性能
- DeepSeek-V3.1以671B总/37B激活参数在工具使用基准上与GPT-5竞争 [^862^]

### 8.3 全球生态互操作性

**技术层面** [^862^] [^877^]：
- 中国模型普遍支持OpenAI-compatible API格式
- DeepSeek-V3.1 API支持Anthropic API格式，可与Claude工作流无缝集成 [^877^]
- DeerFlow支持任何通过LangChain providers访问的模型，包括OpenAI、Anthropic、Google、DeepSeek、Doubao等 [^798^]

**生态影响** [^831^]：
- 日本国立情报学研究所的LLM-jp项目使用Qwen整理学习数据
- 日本企业开发的前10个模型中有6个基于DeepSeek或Qwen
- Meta于2026年放弃开源路线，转向闭源旗舰模型Muse Spark，而中国继续扩大开源优势 [^832^]

**芯片适配** [^844^]：
- 华为昇腾全系列产品全面适配DeepSeek-V4
- 双方"芯模技术紧密协同"，从模型研发阶段就同步推进适配

---

## 9. 反面观点与争议

### 9.1 能力局限性

**ARC-AGI-2表现差距** [^833^]：
- 中国产开源权重模型在ARC-AGI-2基准上落后前沿闭源模型（如Grok、Claude）超过20%
- 这暴露了模型在分布外泛化和不确定性下高风险决策的弱点

**工具使用缺陷** [^863^]：
- DeepSeek-V3.2和DeepSeek-R1在KernelCraft基准上完全失败，根本原因在于工具调用能力不足
- 分析显示DeepSeek-V3.2在首次write_code调用后返回finish_reason="stop"，在未调用run_evaluation前终止Agent循环

### 9.2 监管挑战

**备案范围模糊** [^829^]：
- "舆论属性或者社会动员能力"的前置要件缺乏明确定义
- 企业内部研发测试与对外公众服务的边界不清

**跨境合规成本** [^866^]：
- 不同国家法规差异导致合规成本上升
- 个人用户使用开源模型生成内容难以追溯标识

### 9.3 地缘政治因素

**芯片出口管制影响** [^845^]：
- 美国出口管制限制高端芯片获取，推动中国AI实验室在高效训练方面创新（如FP8精度、异构MoE）
- 但也可能限制了中国模型在最大规模训练上的竞争力

**数据访问障碍** [^833^]：
- 地缘政治紧张导致的数据访问障碍，可能影响中国模型的多样化数据集微调和泛化能力

### 9.4 商业化挑战

**C端竞争激烈** [^847^]：
- 字节跳动豆包AI凭借抖音生态流量，在月活与下载量上超越DeepSeek
- 百模大战已进入"中场战事"，从技术竞争转向生态、流量和产品的综合博弈

**Manus收费争议** [^821^]：
- AI Agent产品Manus Starter每月39美元、Pro每月199美元的定价引发争议
- 智谱AutoGLM沉思选择免费且不限量开放的策略作为差异化

---

## 10. 关键引用来源汇总

| # | 来源 | URL | 日期 | 权威性 |
|---|------|-----|------|--------|
| 471 | Tongyi DeepResearch Technical Report (arXiv) | https://arxiv.org/html/2510.24701v1 | 2025-10-28 | 学术(S) |
| 798 | DeerFlow 2.0 GitHub | https://github.com/deer-flow-bytedance/deer-flow-installer | 2026-03-27 | 官方(S) |
| 801 | DeerFlow 2.0 掘金分析 | https://juejin.cn/post/7620307193842352162 | 2026-03-23 | 技术博客(B) |
| 804 | DeepSeek Research Guide | https://chat-deep.ai/research/ | 2026-04-24 | 行业(B) |
| 805 | Kimi K2.5长上下文分析 | https://futuretechdiaries.com/... | 2026-01-30 | 行业(B) |
| 806 | DeerFlow企业级分析 | https://www.techbuddies.io/... | 2026-03-25 | 行业(B) |
| 809 | MindDeepResearch (arXiv) | https://arxiv.org/html/2604.14518v2 | 2026-04-17 | 学术(S) |
| 810 | OpenRouter Tongyi DeepResearch | https://openrouter.ai/compare/... | 2025-09-01 | 平台(B) |
| 816 | 智谱AutoGLM官方 | https://www.zhipuai.cn/en/research/145 | 2025-12-07 | 官方(S) |
| 820 | 百度文心深度搜索 | https://blog.csdn.net/... | 2025-02-14 | 技术博客(B) |
| 822 | 智谱AutoGLM沉思发布 | https://news.10jqka.com.cn/... | 2025-03-31 | 财经媒体(A) |
| 828 | 国家网信办备案公告 | https://www.cac.gov.cn/... | 2026-01-09 | 政府(S) |
| 829 | 日经：中国AI全球份额15% | https://www.199it.com/... | 2026-01-20 | 媒体(B) |
| 832 | Interconnects AI：Qwen占50%下载 | https://www.199it.com/... | 2026-04-12 | 机构(B) |
| 833 | ARC-AGI-2中国模型分析 | https://blockchain.news/... | 2026-03-02 | 媒体(B) |
| 834 | 豆包上线"深入研究" | https://www.cls.cn/detail/2071329 | 2025-06-30 | 财经媒体(A) |
| 836 | 华为盘古B端应用 | https://www-file.huawei.com/... | 2024 | 华为官方(A) |
| 838 | 华为盘古5.5发布 | https://pdf.dfcfw.com/... | 2025-06-23 | 券商研报(A) |
| 844 | DeepSeek V4信号 | https://finance.sina.cn/... | 2026-04-24 | 财经媒体(A) |
| 845 | 中国开源厂商排名 | https://developer.volcengine.com/... | 2025-08-19 | 开发者社区(A) |
| 853 | Kimi-Researcher开启内测 | https://readhub.cn/topic/8kLlqN0wsHO | 2026-01-28 | 聚合平台(B) |
| 855 | Kimi-Researcher深度研究 | https://www.aizws.net/news/... | 2026-01-25 | 社区(NA) |
| 859 | Tongyi DeepResearch GitHub | https://gitee.com/cherrute/DeepResearch | 2025-12-30 | 镜像(S) |
| 862 | Agentic Tool Use Benchmark | https://arxiv.org/html/2508.16260v2 | 2025-10-11 | 学术(S) |
| 864 | Kimi-Researcher HLE表现 | https://readhub.cn/topic/8kPhoDUVwT6 | 2025-11-08 | 聚合平台(B) |
| 867 | AI生成内容标识办法 | https://mp.weixin.qq.com/... | 2025-09-01 | 行业协会(A) |
| 869 | 扣子空间 | https://toolin.ai/tools/coze-space | 2025-10-16 | 工具目录(NA) |
| 871 | Tongyi DeepResearch原理 | https://www.cnblogs.com/... | 2025-10-01 | 技术博客(B) |
| 873 | DeepSeek-V3.1-Terminus | https://sambanova.ai/blog/... | 2025-09-30 | 企业博客(B) |
| 875 | Tongyi DeepResearch训练 | https://webdevwizardyt.com/... | 2025-09-19 | 技术博客(B) |
| 883 | PaperScope基准 | https://arxiv.org/html/2604.11307v1 | 2026-04-13 | 学术(S) |
| 886 | DeepSeekMath-V2 | https://arxiv.org/abs/2511.22570 | 2025-11-27 | 学术(S) |
| 889 | Test-Time Scaling | https://arxiv.org/html/2510.06135v1 | 2025-08-07 | 学术(S) |

---

*报告完成日期：2026年4月*
*覆盖信息截止日期：2026年4月*
*搜索执行次数：超过20次独立查询*
