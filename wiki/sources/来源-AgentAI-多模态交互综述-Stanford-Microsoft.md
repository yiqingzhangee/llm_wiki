# 来源：Agent AI——多模态交互前沿综述（Stanford/Microsoft）

- **原始路径**：[`raw/transcripts/output_paper4_translation.md`](../../raw/transcripts/output_paper4_translation.md)
- **收录日期**：2026-06-23
- **类型**：论文翻译（综述/预印本）
- **证据等级**：E2（学术预印本，来自 Stanford、Microsoft Research、UCLA、UW、Microsoft Gaming 联合团队，预印本状态）
- **核心作者**：Zane Durante (Stanford), Qiuyuan Huang (Microsoft Research), Li Fei-Fei (Stanford), Jianfeng Gao (Microsoft Research) 等

## 摘要

本文提出了“Agent AI”的总体概念：一类能够感知多模态视觉刺激、语言输入和环境数据，并产生有意义的具身行动的交互式系统。不同于纯文本 LLM Agent 研究，本文的核心视角是**跨现实**（cross-reality）——Agent AI 的训练使用跨现实数据（物理世界 + 虚拟世界），训练后可同时部署于物理和虚拟环境。论文覆盖了 Agent AI 集成（与大型基础模型的关系、幻觉/偏见/隐私/可解释性/监管等挑战）、Agent Transformer 范式、学习方法（RL、IL、上下文学习、智能体系统中的优化）、智能体分类（具身、仿真、生成式、知识推理、神经符号）、以及游戏/机器人学/医疗健康/多模态 NLP 四大应用领域。这是一篇将 Agent 研究从 NLP 社区扩展至计算机视觉、机器人学和 HCI 社区的关键桥梁文献。

## 关键主张

1. **Agent AI 的正式定义**：感知视觉刺激、语言输入及其他环境落地数据，并产生有意义的具身行动的交互式系统。训练目标结合外部知识、多感官输入与人类反馈，核心是“下一具身动作预测”——将基础模型的预测能力从 token 空间扩展到动作空间。

2. **跨现实（Cross-Reality）训练框架**：Agent AI 使用跨现实数据进行训练——来自物理世界（机器人传感器）和虚拟世界（游戏引擎、仿真器）的数据可以在统一框架中联合使用。训练完成的 Agent 可跨物理和虚拟环境部署，这一“现实无关”（reality-agnostic）的特性是其与传统专项 AI 的本质区别。

3. **整体论立场**：论文明确批评 AI 领域因过度还原论而分化为众多子领域的历史路径，主张回到亚里士多德的整体论（Aristotelian Holism）和“目的因”（Final Cause）的系统设计——即先回答“系统为何存在”，再设计其架构。LLM 和 VLM 的革命为这种整体式 AI 创造了新的可能。

4. **Agent Transformer 范式**：论文提出了具体的技术方案——将 Agent 系统形式化为特殊的 Transformer 架构，在预训练和微调阶段引入动作相关的训练目标，使模型不仅能理解多模态输入，还能直接输出可执行的动作序列。

5. **无限智能体（Infinite Agent）**：一种知识迁移策略——从通用基础模型（如 GPT-X、DALL-E）中将记忆和知识迁移到新的领域或场景，用于物理或虚拟世界中的场景理解、生成与交互式编辑。以 RoboGen 为案例展示了“任务提出 → 环境生成 → 技能学习”的自主循环。

6. **幻觉问题与落地缓解**：明确主张在落地（grounded）环境中开发具有智能体能力的 AI 系统可以缓解大型基础模型的幻觉问题——当模型需要为物理上可执行的动作负责时，其生成的环境不符输出会自然受到约束。

7. **伦理排行榜（Ethical Leaderboard）**：除了技术性能，论文还提出了“伦理排行榜”的概念，将偏见与包容性、数据隐私、可解释性、监管合规等伦理维度纳入 Agent AI 的评价体系。

## 与现有 Wiki 的关联

- 关联概念：[[../concepts/agent-ai-and-multimodal-agents.md|Agent AI 与多模态智能体]] — 本文是该概念页的核心来源（已有摄入），本页是对同一论文的更全面的翻译版来源归档，可补充游戏和医疗应用领域的细节。
- 关联概念：[[../concepts/多Agent系统研究趋势与工程映射.md|多Agent系统研究趋势与工程映射]] — 本文中的 CuisineWorld 多智能体游戏数据集为该概念页提供了具体的方法论锚点。
- 关联来源：[[paper-agent-ai-survey-2401-03568.md|Agent AI 综述（原始摄入）]] — 本页是基于全文翻译的更完整版本，补充了 Agent Transformer 技术方案、跨现实训练框架和各应用领域的实验细节。
- 关联来源：[[来源-CoALA-语言代理认知架构-Princeton.md|CoALA 认知架构]] — CoALA 的“落地动作”分类（物理环境/对话/数字环境）与 Agent AI 综述的跨现实训练框架形成理论-实践的互补关系。

## 待验证 / 局限性

- 本文为预印本（A PREPRINT），尚未经过正式同行评议，部分主张（如 Agent Transformer 的具体架构设计）可能在与评审互动后修改。
- 翻译覆盖了原文约前 10 节的内容，后续附录（GPT-4V 在各游戏中的具体提示词细节等）未完整翻译，对游戏实验的复现性价值有限。
- 论文中大量实验基于闭源模型（GPT-4V），可复现性存在天然限制。
