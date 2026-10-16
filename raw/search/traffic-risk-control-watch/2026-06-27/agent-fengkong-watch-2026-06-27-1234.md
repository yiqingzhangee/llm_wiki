# Agent + 风控情报归档

- 查询时间：2026-06-27 12:34 +0800
- 归档时间：2026-06-27 12:34 +0800
- 查询词：
  - `agent 风控 论文`
  - `agent risk control paper`
  - `AI agent fraud detection`
  - `LLM agent 风险治理`
  - `B站 agent 风控`
  - `小红书 agent 风控`
  - `微信公众号 agent 风控`
  - `social media agent governance`
- 查询路径：AnySearch `academic` / `social_media` / `general`
- 说明：本次已先调用 `/anysearch-skill` 对学术、社媒与通用网页做并行拉取；`social_media` 路径对 B站/小红书/微信公众号 的直接命中有限，实际高相关结果主要来自微信搜狗、通用网页、官方文档与论文解读站。

## 来源类别统计

- academic：38 条原始结果
- social_media：40 条原始结果
- general：80 条原始结果（分两批完成，因 `batch_search` 单次最多 5 个 query）
- 去重后纳入归档：24 条

### 平台分布（归档结果）

- 论文 / 论文解读 / 白皮书：6
- 官方文档 / 官方指南：5
- 技术博客 / 开发者社区：8
- 社交媒体 / 社区 / 平台治理报道：5

## 结果清单

| 标题 | URL | 平台 | 时间 | 一句摘要 |
|---|---|---|---|---|
| Quantifying Trust: Financial Risk Management for Trustworthy AI Agents | https://arxiv.org/abs/2604.03976 | arXiv | 2026 | 将金融风险管理语言迁移到可信 AI Agent，强调从模型内生属性转向资产、权限与运行暴露面的风险量化。 |
| SafeAgent: Safeguarding LLM Agents via an Automated Risk Simulator | https://aclanthology.org/2026.acl-long.1501.pdf | ACL 2026 | 2026 | 用自动化风险模拟器评估多轮、工具调用式 LLM Agent 的动态安全风险。 |
| SAGE: An LLM-driven Self Reflective Agentic Framework for Fraud Detection | https://arxiv.org/html/2606.08146 | arXiv | 2026 | 针对支付/电商/通信欺诈，强调个体级准确率、类不平衡鲁棒性与风险经理可解释性。 |
| RISK: A Framework for GUI Agents in E-commerce Risk Management | https://papernotes.org/ACL2026/llm_safety/risk_a_framework_for_gui_agents_in_e-commerce_risk_management/ | Paper Notes / ACL 2026 解读 | 2026 | 面向电商风控 GUI Agent，提供数据集、基准与强化微调框架，强调训练-部署一致性。 |
| Agentic AI Risk-Management Standards Profile | https://cltc.berkeley.edu/publication/agentic-ai-risk-profile/ | UC Berkeley CLTC | 2026-02 | 基于 NIST AI RMF 提出 agentic AI 风险治理 profile，覆盖单 Agent 与多 Agent 系统。 |
| Agentic AI Risk-Management Standards Profile | https://papers.ssrn.com/sol3/papers.cfm?abstract_id=6276998 | SSRN | 2026 | CLTC 报告的论文版入口，可作为正式学术/政策引用。 |
| 部署 Multi-Agent AI Fraud Detection System on OCI | https://docs.oracle.com/en/solutions/ai-fraud-detection/index.html | Oracle 官方文档 | 2025-06 | 提供基于 OCI 的多 Agent 反欺诈架构，含 orchestrator、data retrieval agent 与 fraud analyzer agent。 |
| 治理并保障整个组织中的 AI 代理安全 | https://learn.microsoft.com/zh-cn/azure/cloud-adoption-framework/ai-agents/governance-security-across-organization | Microsoft Learn | 2026 | 从组织级问责、代理注册表、身份、最小权限、DLP、事件响应等角度给出 Agent 治理基线。 |
| Agent Governance Toolkit README | https://microsoft.github.io/agent-governance-toolkit/i18n/README.zh-CN/ | Microsoft Open Source | 2026 | 强调运行时策略执行、零信任身份、执行沙箱与可靠性工程，覆盖 OWASP Agentic 风险。 |
| Introducing the Agent Governance Toolkit | https://opensource.microsoft.com/blog/2026/04/02/introducing-the-agent-governance-toolkit-open-source-runtime-security-for-ai-agents/ | Microsoft Open Source Blog | 2026-04-02 | 宣布开源运行时治理工具包，主张“治理代理做什么”，而非只看模型输出。 |
| Deploy a multi-agent AI fraud detection system on OCI | https://docs.oracle.com/en/solutions/ai-fraud-detection/index.html | Oracle 官方方案 | 2025-06 | 将多 Agent 协同引入反欺诈调查流程，用 supervisor + retrieval + analyzer 模拟调查团队。 |
| AI agents for fraud detection and investigation: how they work and what to evaluate | https://www.unit21.ai/blog/ai-agents-for-fraud-detection-and-investigation-how-they-work-and-what-to-evaluate | Unit21 博客 | 2026 | 从告警量、人力瓶颈与误杀成本出发，讨论如何评估欺诈检测 Agent。 |
| Blinded by the Agent: How AI Agents Are Disrupting Fraud Detection | https://transmitsecurity.com/blog/blinded-by-the-agent-how-ai-agents-are-disrupting-fraud-detection | Transmit Security | 2026 | 讨论攻击者如何利用合法 AI Agent 掩蔽欺诈行为，提示“agent 既是防守方也是攻击媒介”。 |
| Agentic风控：Flink+Fluss+大模型构建Agent全链路风险感知与实时告警 | https://segmentfault.com/a/1190000047735923 | SegmentFault | 2026 | 以实时计算栈构建风控 Agent，强调规则硬编码向智能感知告警演进。 |
| 金融反欺诈领域大模型Agent与工作流编排技术深度研究报告 | https://developer.cloud.tencent.com.cn/article/2614836 | 腾讯云开发者社区 | 2026 | 聚焦金融反欺诈里的多智能体协作、图式工作流编排、记忆与 RAG。 |
| AI Agent安全风险治理：从访问控制到行动链管控的范式升级 | https://cloud.tencent.com/developer/article/2694982 | 腾讯云开发者社区 | 2026 | 强调 Agent 风险从“访问控制”升级为“连续行动链治理”。 |
| AI Agent Harness Engineering 在金融风控场景下的实战避坑指南 | https://gitcode.csdn.net/69d8fd6b0a2f6a37c59ea0f0.html | GitCode / CSDN | 2026 | 用 harness engineering 解释风控 Agent 在权限、上下文、反馈闭环上的工程边界。 |
| 金融风控智能体：Agent + 大模型实现实时反欺诈监测 | https://cloud.tencent.com/developer/article/2504291 | 腾讯云开发者社区 | 2026 | 以金融交易反欺诈为例，讨论 Agent 与大模型结合的实时监测路径。 |
| 银行业探索“人工智能+” AI Agent重塑金融新生态 | https://www.cfca.com.cn/20250321/200000790.html | CFCA | 2025-03-21 | 从银行视角讨论 Agent 在风险、运营、服务环节的产业化趋势。 |
| 率先向AI托管账号出手，为什么小红书要打响第一枪？ | https://36kr.com/p/3720962630417024 | 36氪 | 2026-03-10 | 报道小红书明确打击 AI 托管运营账号，说明内容社区已把 Agent 滥用纳入平台治理。 |
| 小红书平台风控严格，尽可能采用有头模式head · Issue #188 | https://github.com/dreammis/social-auto-upload/issues/188 | GitHub Issue | 2026-03-28 | 开发者直接记录小红书风控严格、自动化易触发限制，反映一线工具侧约束。 |
| 小红书社区反作弊探索与实践 | https://zhuanlan.zhihu.com/p/599625620 | 知乎专栏 | 未标明 | 从社区反作弊角度拆解小红书风控/黑灰产对抗逻辑。 |
| 安全防护 | 微信开放文档 | https://developers.weixin.qq.com/miniprogram/security/gateway/config/security.html | 微信开放文档 | 未标明 | 微信官方安全网关提供恶意注册、营销作弊、群控养号等风险识别能力。 |
| 等待就是最好的进攻：为什么微信可能是中国 AI Agent 时代最大的超级入口 | https://mp.weixin.qq.com/s?src=11&timestamp=1782534893&ver=6807&signature=bFKxqkQCFDT9Kcv2DPehL-W6*i5I-0fHZzZvIK0RFYf8nWCs-JzcwJywF*qI9BGeOD8du1ZJ43dswlaXqCtv9TnrSrZICyPVKhfJxsBxI2dz7RJvBwHiQ-2KiJjUChLI&new=1 | 微信公众号 | 2026（页面摘要） | 指出 Agent 深入客服/交易/支付后，风控与人工接管机制会成为平台级核心约束。 |
| Is anyone actually enforcing AI governance, or just writing policies? | https://www.reddit.com/r/AI_Agents/comments/1t70lnk/is_anyone_actually_enforcing_ai_governance_or/ | Reddit | 未标明 | 社区讨论企业 AI governance 常停留在文档层，而非真正落到运行时控制。 |

## 高价值来源

> 优先标准：官方文档 / 权威研究机构 / 正式论文 / 高信息密度且可复用的架构或治理材料。

1. **UC Berkeley CLTC — Agentic AI Risk-Management Standards Profile**
   - https://cltc.berkeley.edu/publication/agentic-ai-risk-profile/
   - 价值：有明确治理框架、术语和 NIST AI RMF 对齐，适合进入 wiki 的“治理框架”层。
2. **Microsoft Learn — 治理和保护 AI 代理**
   - https://learn.microsoft.com/zh-cn/azure/cloud-adoption-framework/ai-agents/governance-security-across-organization
   - 价值：组织级控制面、身份、DLP、最小权限、事件响应与开发标准完整。
3. **Microsoft Agent Governance Toolkit**
   - https://microsoft.github.io/agent-governance-toolkit/i18n/README.zh-CN/
   - 价值：运行时治理导向，直接对接工具调用、资源访问与代理间通信。
4. **Oracle — Multi-Agent AI Fraud Detection System on OCI**
   - https://docs.oracle.com/en/solutions/ai-fraud-detection/index.html
   - 价值：多 Agent 反欺诈架构清晰，适合作为“工程参考实现”证据。
5. **arXiv — Quantifying Trust: Financial Risk Management for Trustworthy AI Agents**
   - https://arxiv.org/abs/2604.03976
   - 价值：把可信 Agent 与金融风险管理语言连接，适合做概念桥接。
6. **arXiv — SAGE: An LLM-driven Self Reflective Agentic Framework for Fraud Detection**
   - https://arxiv.org/html/2606.08146
   - 价值：直接命中 fraud detection 场景，兼顾类不平衡与可解释性。
7. **ACL 2026 — SafeAgent**
   - https://aclanthology.org/2026.acl-long.1501.pdf
   - 价值：多轮工具调用 Agent 的自动化风险模拟评测框架。
8. **ACL 2026 解读 — RISK GUI Agent Framework**
   - https://papernotes.org/ACL2026/llm_safety/risk_a_framework_for_gui_agents_in_e-commerce_risk_management/
   - 价值：电商 GUI 风控 Agent 非常贴近“行动型 Agent + 风控”主题。
9. **微信开放文档 — 安全防护**
   - https://developers.weixin.qq.com/miniprogram/security/gateway/config/security.html
   - 价值：官方平台风险控制能力说明，适合作为平台治理一手证据。
10. **36氪 — 小红书打击 AI 托管账号**
    - https://36kr.com/p/3720962630417024
    - 价值：虽非官方原文，但对“小红书已把 Agent 滥用纳入治理”提供高密度媒体证据。

## 待复核清单

1. **大量微信搜狗结果**（`mp.weixin.qq.com`）
   - 问题：能拿到标题/摘要，但不少页面缺发布时间、作者身份不清，且内容营销化明显。
2. **B站 / 小红书 自动化工具推广文**
   - 问题：很多结果在卖工具或夸大“低风控”，证据偏弱，容易混入规避平台治理话术。
3. **知乎 / CSDN / GitCode 二手解读**
   - 问题：适合作线索，不宜直接当最终结论，需要回溯原论文或官方材料。
4. **Reddit 社区讨论**
   - 问题：适合观察行业感知与痛点，但不属于权威证据。
5. **小红书 / 微信 / B站 平台风控实操贴**
   - 问题：部分内容带明显“反风控”倾向，只能作为平台约束存在的旁证，不能作为推荐实践。

## 无效 / 异常记录

- `general` 路径第一次 `batch_search` 失败：**原因**为 AnySearch `batch_search supports a maximum of 5 queries`，后续已拆成两批重试成功。
- `extract` 对 `36kr` 页面失败：`all extract proxies exhausted; direct connection disabled`，因此该条目仅依据搜索摘要归档，列入“待复核边缘”。
- `social_media` 路径对 B站 / 小红书 / 微信公众号 的原生垂直支持有限；本次主要通过 `sogou_wechatmp`、`reddit_post` 和 general 搜索补足。

## 初步观察

1. **主题重心正在从“模型风险”转向“行动链风险”**：Microsoft、CLTC、腾讯云等都强调对工具调用、权限边界、持续观察与事件响应的治理。
2. **风控场景下的 Agent 已出现双重角色**：既有“Agent 作为风控系统”去做反欺诈，也有“Agent 作为攻击/绕过媒介”去放大平台风险。
3. **社交媒体平台已经开始正面处理 Agent 滥用**：小红书对 AI 托管账号的治理最明确；微信侧更多以安全网关和风险能力形式出现；B站线索更多来自反爬/风控拦截与开发者实践。
4. **学术信号开始细化到子任务**：如 GUI 风控 Agent、自动化风险模拟器、自反式欺诈检测框架，说明该方向开始从泛“Agent 安全”走向垂直领域专用方法。
