# 来源：InstructUE 指令感知用户嵌入（2025）

> 原始来源：[`raw/transcripts/LLM-based User Representation Foundation Models/paper_zh_16_InstructUE.md`](../../../raw/transcripts/LLM-based%20User%20Representation%20Foundation%20Models/paper_zh_16_InstructUE.md)。对应论文：*Instruction-aware User Embedding via Synergistic Language and Representation Modeling*，arXiv:2510.11016v1。
>
> 关联概念：[[LLM用户表征基础模型与场景适配]]、[[LLM推荐系统的分层改造与评估]]。

## 核心结论

InstructUE 将用户 embedding 表达为 $f(X_i, I_k)$：同一用户的异构历史 $X_i$ 在不同自然语言指令 $I_k$ 下可产生不同表征。指令是潜空间的软控制器，意在降低静态统一向量受到无关行为干扰的问题。

## 方法

1. 六类数据源各自经模态专属 encoder 与轻量 adapter 投影到 LLM 空间，避免将稀疏行为、表格和文本直接拼接后丢失结构。
2. UserQA 同时构造稳定属性问答与由未来行为得到的前瞻性问答；自回归目标注入语言/领域语义，对比目标建立可检索的用户—答案结构。
3. 少样本 cluster-based instruction tuning 用业务标签改善指令 token，使指令从泛化描述变为场景条件。

## 证据边界

指令条件化的收益依赖指令质量、合成 QA 的忠实性、标签聚类和下游任务定义；不能把论文的用户预测、营销和推荐结果外推为任意个性化系统都适用。上线还须约束指令权限、检查输入中的敏感属性与提示注入、监控任务切换时的表征漂移。

与 [[来源-Query-as-Anchor场景自适应用户表征-2026]] 的差异在于：InstructUE 的主要控制接口是指令及其少样本调优；Query as Anchor 的主要接口是尾部 query 重锚定和用户前缀 KV-cache 复用。