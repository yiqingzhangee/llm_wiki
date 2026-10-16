# 来源：Query as Anchor 场景自适应用户表征（2026）

> 原始来源：[`raw/transcripts/LLM-based User Representation Foundation Models/paper_zh_12_Query_as_Anchor.md`](../../../raw/transcripts/LLM-based%20User%20Representation%20Foundation%20Models/paper_zh_12_Query_as_Anchor.md)。对应论文：*Query as Anchor: Scenario-Adaptive User Representation via Large Language Model*，arXiv:2602.14492v2；原始阅读稿标注 KDD 2026 拟收录。
>
> 关联概念：[[LLM用户表征基础模型与场景适配]]、[[LLM推荐系统的分层改造与评估]]。

## 核心结论

Query as Anchor 不再为每个用户输出唯一静态 embedding，而是把业务 query 放在用户 token 序列尾部作为语义锚点，使同一历史画像按不同场景重新聚焦。它追求跨任务复用与场景敏感性的平衡，而不是直接生成推荐列表。

## 方法与服务接口

- **UserU 预训练数据**同时使用未来行为预测与合成 QA，让用户历史与可解释的用户理解语义对齐。
- **分层 coarse-to-fine 编码**同时暴露用户级、模态级和部分事件级表示，允许 query 按需选择粒度并抑制无关信号。
- **对比 + 自回归训练**结合判别式对齐与语言语义注入；少样本时用 cluster-based soft prompt 适配业务。
- query 位于可缓存的用户前缀之后，因而可复用用户历史的 KV cache，降低同一用户服务多个场景时的边际开销。

## 证据边界

原始阅读稿的离线与支付宝线上 A/B 结果是特定场景、数据治理和缓存实现下的证据。query 条件化并不自动保证 query 合法、权限隔离、候选覆盖或端到端推荐提升；必须测试提示漂移、缓存命中/失效、p95/p99、无关行为干扰以及不同人群上的公平性。

它与 [[来源-InstructUE指令感知用户嵌入-2025]] 都是条件化表征，但前者以运行时 query 重新锚定并强调 KV-cache 复用，后者以指令和少样本指令调优为主要控制接口。