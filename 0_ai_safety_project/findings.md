# AI Safety 调研发现

## 本地证据

### 1. Agent 工具调用安全已有可直接启动的场景

- `wiki/concepts/agent/Agent对抗安全：从briccli攻击面出发的调研与启动方案.md` 已枚举直接/间接提示注入、工具滥用、数据外泄、权限提升、记忆投毒与技能注入。
- `raw/papers/.../AgentPoison...md` 提供 RAG/长期记忆投毒的一手论文材料与可复现实验入口。
- `raw/code/gstack/docs/designs/ML_PROMPT_INJECTION_KILLER.md` 提供分层防御、检测器、输入归一化、canary 与调用白名单的工程设计。
- `wiki/comparisons/agent/传统流量反作弊与Agent流量治理对比.md` 已定义会话/调用链级风险和审计指标。
- **影响**：近期主项目可直接从沙箱 benchmark 与运行时策略层起步，不必先训练新基础模型。

### 2. 多模态对抗已形成“感知恢复—触发—精审—裁决”链路

- `wiki/comparisons/multimodal/图像对抗防御链路对照-2026-09-02.md` 已区分视觉头盲区与决策层盲区，并提出对抗检测触发器优先于直接主裁决。
- `wiki/concepts/multimodal/多模态模型视觉头与风控部署边界.md` 已拆分视觉头、整模型和风控系统三层指标。
- `wiki/concepts/multimodal/短视频内容审核与多模态治理.md` 已覆盖 VLM 审核、过程监督、RAG 与人机协作。
- **影响**：第二项目应研究“多模态注入/规避样本的安全路由”，而不是再做一个孤立的图像分类器。

### 3. EviState 与证据仲裁可成为统一安全状态层

- `0_evi_state_project/evistate_training_framework.md` 已定义六类证据槽、时间 cutoff、稳定 evidence ID、修订/撤回、缺失原因与离线训练约束。
- `wiki/concepts/applications/LLM统计特征编码的证据仲裁.md` 已定义数值、时间窗、来源版本、复算状态、冲突类型与行动授权边界。
- `wiki/sources/applications/来源-文本与数值冲突下的LLM证据仲裁-2026.md` 提供文本、数值、时效、可靠性与工具预测冲突的受控基准锚点。
- **影响**：第三项目应把安全从单次检测升级为“长期证据状态 + 冲突仲裁 + 可回放决策”，并与前两个项目共享事件和审计契约。

### 4. 本地能力画像

| 能力 | 现有资产 | 可转化成果 |
|---|---|---|
| Agent 工程 | 工具、技能、记忆、Harness、调用链治理 | Agent 安全 benchmark 与 runtime policy |
| 风控系统 | 识别—行为—处置—治理闭环 | 高影响行动授权与回放评测 |
| 多模态 | VLM、视频审核、图像对抗、结构恢复 | 多模态 prompt injection / evasion 路由 |
| 图与时序 | 可信 GNN、时序 Agent、EviState | 长期风险状态与攻击链检测 |
| 证据工程 | 来源分级、冲突仲裁、证据图谱 | 可审计 AI Safety evaluation |

## 外部证据

### 1. Agent 安全已有可复用基准，但必须做自适应与影响加权评测

- AgentDojo 提供 4 个有状态环境、97 个正常任务、74 个工具和 629 个安全测试，可同时用确定性状态检查评估正常任务效用与攻击目标是否达成。其论文基线中，正常任务完成率本身低于 66%，部署检测器后攻击成功率可降至 8%，说明安全收益必须与任务效用一起报告。
- NIST CAISI 基于 AgentDojo 扩展了远程代码执行、批量数据外泄和自动钓鱼场景。针对模型做自适应红队后，攻击成功率由基线攻击的 11% 上升至 81%；允许重复尝试时，五类攻击的平均成功率由 57% 上升至 80%。
- CaMeL 不依赖模型“自觉拒绝”，而是在系统层分离可信查询中的控制流与不可信数据，并用 capability 约束数据流；作者报告在 AgentDojo 上有 67% 的任务可获得可证明安全性。
- NIST 2026 AI Agent Standards Initiative 将 Agent 身份、认证、授权、安全评测和互操作列为战略方向；NCCoE 概念项目同时强调 identification、authorization、auditing、non-repudiation 和 prompt injection 控制。
- **影响**：近期项目应复用 AgentDojo 的“正常任务 + 攻击任务 + 环境状态”结构，但增加风险影响权重、自适应攻击、重复尝试和 capability/HITL 策略，而不是只训练提示注入分类器。

### 2. 多模态安全需要把“看见攻击”与“阻止危险行动”连起来

- VPI-Bench 提供 5 类仿真网站上的 306 个动态测试，攻击指令直接嵌入渲染界面；作者报告部分平台上 CUA 与 BUA 的攻击成功率分别最高达到 51% 和 100%，单靠 system prompt 防御提升有限。
- EVADE-Bench 提供中文电商场景中的 2,833 条文本和 13,961 张图片，覆盖 6 类违规规避；其任务同时要求规则理解、OCR/视觉理解和规避意图识别。作者对 26 个模型的评测显示，当前模型仍会系统性误判隐喻、裁剪和复杂规则样本。
- 本地链路已经区分“视觉头看不见”和“看见后决策错误”，与 VPI-Bench 的行动劫持和 EVADE-Bench 的内容规避形成互补。
- **影响**：第二项目不应追求一个统一违规分类器，而应评测路由策略能否把样本送往结构恢复、OCR、VLM 二审、规则或人工，并最终降低高风险漏放与越权行动。

### 3. 长期记忆安全的攻击证据强，统一状态评测仍是空缺

- AgentPoison 在自动驾驶、知识问答和医疗 Agent 上投毒长期记忆或 RAG；作者报告在投毒率低于 0.1% 时，平均攻击成功率不低于 80%，正常性能下降不超过 1%。
- PoisonedRAG 在百万级知识库中针对每个目标问题注入 5 条恶意文本，作者报告攻击成功率可达 90%，且所测防御不足以阻断攻击。
- HoH 以 96,124 个问题和 219,463 篇文档评测过期证据；作者报告即使同时检索到当前信息，过期信息也可使主流模型性能至少下降 20%。
- RAGuard 包含 2,648 个事实核查 claim 和 16,331 篇支持、误导或无关文档；其基线中，加入误导检索后所有受测 RAG 系统均劣于无检索基线。
- Microsoft 的 Zero Trust 攻击目录和 OWASP Agent Memory Guard 均将 memory write governance、schema-bound memory、provenance、versioning、rollback、isolation 与 telemetry 作为核心控制。OWASP 项目仍处 incubator 阶段，不能把其实现成熟度当作已验证结论。
- TREC 2025 RAG Track 已把完整性、归因验证和证据一致性纳入多层评测，但并不直接覆盖跨会话投毒、修订/撤回和行动授权。
- **影响**：第三项目的研究缺口不是再证明“RAG 会被投毒”，而是建立带时间、来源、修订、撤回、冲突和回滚的长期证据状态评测，并验证这些控制是否能降低持续攻击而不破坏正常记忆效用。

### 4. 证据冲突与适用边界

- AgentDojo 原论文的攻击成功率低于 25%，而 NIST 自适应红队可达到 81%；二者不是互相否定，而是说明静态已知攻击会显著低估有针对性的攻击者。
- VPI-Bench 的视觉行动劫持、EVADE-Bench 的内容规避和本地图像像素攻击使用不同目标与指标，不能拼成统一 SOTA 排名。
- AgentPoison 与 PoisonedRAG 是目标化投毒攻击；HoH 与 RAGuard 是过期或误导证据压力测试。它们共同支持长期证据风险，但不证明同一个防御可覆盖全部失效。
- NIST、Microsoft 和 OWASP 材料主要提供治理与控制方向；具体有效性仍需在本项目的沙箱 benchmark 中实测。

### 5. 一手来源

- AgentDojo：https://arxiv.org/abs/2406.13352
- NIST Agent Hijacking Evaluation：https://www.nist.gov/news-events/news/2025/01/technical-blog-strengthening-ai-agent-hijacking-evaluations
- NIST AI Agent Standards Initiative：https://www.nist.gov/artificial-intelligence/ai-agent-standards-initiative
- CaMeL：https://arxiv.org/abs/2503.18813
- VPI-Bench：https://arxiv.org/abs/2506.02456
- EVADE-Bench：https://arxiv.org/abs/2505.17654
- AgentPoison：https://papers.neurips.cc/paper_files/paper/2024/hash/eb113910e9c3f6242541c1652e30dfd6-Abstract-Conference.html
- PoisonedRAG：https://www.usenix.org/conference/usenixsecurity25/presentation/zou-poisonedrag
- HoH：https://arxiv.org/abs/2503.04800
- RAGuard：https://arxiv.org/abs/2502.16101
- TREC 2025 RAG Track：https://pages.nist.gov/trec-browser/trec34/rag/proceedings/
- Microsoft AI Memory / Context Poisoning：https://learn.microsoft.com/en-us/security/zero-trust/catalog-ai-attack-techniques/ai-memory-context-poisoning
- OWASP Agent Memory Guard：https://owasp.org/www-project-agent-memory-guard/

## 项目候选

1. **BRIC-SafeAct**：Agent 工具调用红队基准与运行时策略网关，0—3 个月主项目。
2. **MM-SafeRoute**：多模态注入与规避内容的分层安全路由，3—6 个月扩展项目。
3. **EviGuard**：长期 Agent 的证据状态、记忆投毒与冲突回放，6—12 个月研究项目。

三者不是同时启动：后一个项目复用前一个项目的攻击场景、事件格式、审计日志与评测工具。

## 证据缺口与不确定性

- `briccli` 各域的写权限、沙箱和可用测试账号尚未确认；首期必须使用 mock tool 或只读 dry-run。
- 本地多模态对抗结论来自不同数据和指标，不能直接组合成统一 SOTA 排名。
- EviState 当前是 deterministic fixture，未接真实冻结样本、正式 target encoder 与成熟标签。
- 现有 Agent 安全草稿包含 2026 工作，需回到论文/官方页面逐条复核后再作为长期路线依据。
