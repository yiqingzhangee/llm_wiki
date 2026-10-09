# 来源：视频审核与电商风控 GUI Agent 批次（2021–2026）

- **原始路径**（`raw/articles/video_moderation/`）：
  - [`2026-UNIVID Unified Vision-Language Model for Video Moderation.pdf`](../../raw/articles/video_moderation/2026-UNIVID%20Unified%20Vision-Language%20Model%20for%20Video%20Moderation.pdf)（含 `_dual_智谱4Flash` / `_translated_智谱4Flash` 版本）
  - [`2026-ACL-Industry-IPS In-Prompt Process Supervision for Short Video Content Moderation.pdf`](../../raw/articles/video_moderation/2026-ACL-Industry-IPS%20In-Prompt%20Process%20Supervision%20for%20Short%20Video%20Content%20Moderation.pdf)（含 `Dual` / `ZH` 版本）
  - [`2026-RiskWebWorld ... GUI Agents in E-commerce Risk Management.pdf`](../../raw/articles/video_moderation/2026-RiskWebWorld%20A%20Realistic%20Interactive%20Benchmark%20for%20GUI%20Agents%20in%20E-commerce%20Risk%20Management.pdf)（含 `Dual` / `ZH` 版本）
  - [`2025-RISK A Framework for GUI Agents in E-commerce Risk Management.pdf`](../../raw/articles/video_moderation/2025-RISK%20A%20Framework%20for%20GUI%20Agents%20in%20E-commerce%20Risk%20Management.pdf)（含 `Dual` / `ZH` 版本）
  - [`2021-VideoModerator A Risk-aware Framework ... .pdf`](../../raw/articles/video_moderation/2021-VideoModerator%20A%20Risk-aware%20Framework%20for%20Multimodal%20Video%20Moderation%20in%20E-Commerce.pdf)（含 `2109.03479v1_dual` / `_translated_智谱4Flash` 版本）
- **类型**：产业/学术论文批次（VLM 审核 + 过程监督 + GUI Agent 风控基准/框架）
- **同步日期**：2026-08-31
- **证据等级**：**E2**（含中文译文版 PDF；论文多为产业系统报告，部分量化为 beta/内部数据，需按各论文口径对待）

## 一句话摘要

这批来源沿两条主线展开：**(A) 短视频内容审核**从 2021 年的"风险感知可视化人机协作"（VideoModerator）演进到 2026 年的"统一 VLM 基座 + 过程监督微调"（UNIVID、IPS）；**(B) 电商风控 GUI Agent** 从训练框架（RISK）走向逼真交互基准（RiskWebWorld）。二者共同指向**"用基础模型把碎片化黑箱审核/风控系统，重构为可解释、可治理、可规模化"**的产业趋势。

## 各来源关键主张

### A1. UNIVID — 统一视觉语言模型用于视频审核（字节跳动，2026）

- **问题**：全球规模视频审核需要细粒度多模态推理、可解释决策、可规模化支撑下游执行；传统系统依赖碎片化黑箱分类器，难维护且不透明。
- **方案**：提出 **UNIVID**——面向视频审核的统一视觉语言基础模型（基于 **LLaVA-OneVision**，LLM 用 **Mistral-v0.3-7B**）。与标准生成/分类模型不同，UNIVID 产出语义丰富的描述，作为**可解释的中间表示**，支撑人工决策复用与多任务复用。
- **三阶段级联流程**：
  1. **风险过滤器（RiskFilter）**：低延迟高吞吐筛全量视频流，用 UNIVID 生成 caption/OCR/标题/轻量策略标签，融合共享嵌入接小 MLP，按策略类决策树路由高风险视频。
  2. **UNIVID-Lite + UNIVID-RAG**：轻量下游模型 + 检索增强，提升预测与召回。
  3. **趋势治理**：缓存 UNIVID 嵌入 + 自适应检索，应对新兴风险。
- **训练**：三阶段（模态对齐预训练 → 图像 caption/VQA 指令微调 → 高质量 caption 继续微调），32×H100 约 120 小时；用 GPT-4o 生成 caption/VQA 并做"事实修正 + 政策对齐"人工闭环。
- **量化（beta/内部口径）**：违规漏检相对下降约 **42.7%**、另一指标提升约 **37.0%**；下游支撑 1000+ 策略定义模型；自称首个大规模视频审核视觉基础模型。

### A2. IPS — 短视频内容审核的即时提示处理监督（TikTok，ACL 2026 Industry）

- **问题**：MLLM 在短视频审核有效，但现有方法对"过程级监督"关注不足，复杂策略判断（如是否非原创、是否受保护内容与用户编辑区分）需要结构化推理与过程一致性。
- **方案**：**IPS（In-Prompt Process Supervision）框架**——把人工标注的**逻辑判据组成**显式化为提示内多轮问答，每个辅助问题以 `<ans>` 标记结尾，取该标记对应的隐藏状态过 MLP 预测辅助答案与最终标签。
- **关键设计**：辅助标签作为**中间指导而非最终结果的严格决定因素**，因此对轻微不准确不敏感；可用通用 MLLM 自动生成辅助标签（与人工一致率仅 75% 仍抗噪）。
- **评测**：在 **MM-Soc**、**HSD** 及小类任务（UCC、ANSA）上验证。

### A3. VideoModerator — 电商多模态视频审核的风险感知框架（2021，arXiv 2109.03479）

- **早期工作**：电商视频审核强调**交易相关风险与场景特征**，含视觉/语音/字幕/商品/直播语境等多模态信息。
- **方案**：以**风险感知**为核心，结合多模态学习识别并可视化电商视频风险，构建含三种视频类型的**交互可视化界面**，服务审核员人机协作。
- **定位**：偏向"可理解、可视化、可交互"的人机协作审核系统，而非纯分类模型——是 UNIVID/IPS"可解释中间表示"思路的早期形态。

### B1. RISK — 电商风控 GUI Agent 框架（蚂蚁国际 + 复旦，2025，arXiv 2509.21982）

- **问题**：电商风控需多步结构化 GUI 交互，半结构化网页内容深藏于动态子页/交互表单/DOM，现有方法难泛化。
- **三组件**：
  - **RISK-Data**：浏览器交互采集，含 8,492 单步 + 2,386 多步轨迹，配截图/HTML DOM/任务提示/底层动作。
  - **RISK-Bench**：802 单步 + 320 多步任务，按难度分级，步空间 320 类。
  - **RISK-R1**：基于 **GRPO** 的强化微调，多级奖励（输出级 / 步级 / 多步轨迹级 / 任务难度加权）。
- **量化**：**RISK-R1-7B** 单步 +6.8%、多步 +8.8%，仅用 SOTA 基线 7.2% 参数，线上评测任务成功率 **70.5%**，领域 SOTA。开源 `github.com/RenqiChen/RISK`。

### B2. RiskWebWorld — 电商风控 GUI Agent 逼真交互基准（蚂蚁国际，2026，arXiv 2604.13531）

- **问题**：GUI Agent 在通用浏览器任务强，但在高风险领域专业场景效用弱；现有交互基准缺乏领域相关性与真实性。
- **方案**：高保真交互式电商风控环境 + 基准，覆盖 **8 业务域 / 300+ 网页 / 1,513 任务**，Gymnasium 兼容、支持 RL 训练。
- **贡献**：构建逼真多步风控分析任务、规模化环境基础设施、全面评测并识别失败模式、演示 RL 可提升开源模型（如 +16.2%）。
- **与 RISK 关系**：RISK 给训练框架与数据，RiskWebWorld 给更逼真的评测环境与任务规模——同团队、同方向递进。

## 与现有 wiki 的关系

- 支撑更新 [[短视频内容审核与多模态治理]]：UNIVID/IPS 把 KuaiMod 的"判例法 + VLM + CoT"思路推进到"统一 VLM 基座 + 过程监督微调"，VideoModerator 补上 2021 起点，形成审核系统**可解释中间表示**演进脉络。
- 与 [[流量风控技能栈与Agent治理]] 强呼应：RISK/RiskWebWorld 是"传统反作弊扩展到 Agent 调用链治理"的**电商风控 GUI Agent**具体实例——风控工作流正从规则/小模型迁移到 GUI Agent + RL 微调。
- 与 [[Agent任务分流：工作流与智能体边界]] 类比：RISK 的多级奖励、RiskWebWorld 的多步任务，正是"workflow 编排 + agent 决策"边界的产业样本。
- 与 [[对抗样本攻击原理与实践]] 关联：UNIVID 明确提到多语言 OCR、对抗性视觉攻击（如分屏效果）作为审核挑战。

## 冲突与不确定性

- **量化口径**：UNIVID 的 42.7%/37.0% 为 beta/内部仿真口径，非公开 benchmark；RISK 的 70.5% 为线上评测；IPS 的 MM-Soc/HSD 为学术基准——三者不可直接横比。
- **VideoModerator（2021）与 UNIVID（2026）的"可解释"语义不同**：前者指可视化人机协作，后者指 VLM 生成的语义描述作为中间表示——演进中"可解释"内涵已迁移，对照时勿混为一谈。
- 多版 PDF（dual/ZH/translated）由智谱4Flash 生成，可能存在翻译偏差；关键术语以英文原名为准。
- RISK 与 RiskWebWorld 同团队但论文各自独立，数据集/任务统计口径（8,492 vs 1,513）分属训练数据与评测基准，勿混用。

## 反向链接

- [[短视频内容审核与多模态治理]]
- [[流量风控技能栈与Agent治理]]
- [[Agent任务分流：工作流与智能体边界]]
- [[对抗样本攻击原理与实践]]
- [[研究主题导航]]
