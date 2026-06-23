# 分节详解：A Survey on LLMs with Multilingualism（Huang et al., 2025）

> 对应来源页：[[来源-多语言大语言模型综述-2025-Huang]] · 原始：`raw/articles/MinerU_markdown_2025-A_Survey_on_Large_Language_Models_with_Multilingualism-_Recent_Advances_and_New_Frontiers_2043711215071019008.md`

## §1 Introduction

从 LLM 多语能力与**语言公平**动机出发，指出多语研究仍碎片化；承诺提供结构化分类（Fig.2）、多视角综述、挑战与解法、数据集与评测、未来方向及维护文献库。

## §2 Preliminary

### §2.1 Multilingual Models

定义多语模型目标：单模型多语言、跨语迁移、降低多模型维护成本。

### §2.2 Pre-Trained Language Models

Transformer 三代架构（编码器/解码器/编解码）与规模化趋势，以及 TLM 等多语预训练目标。

### §2.3 Multilingual Paradigm Transition

从「Pre-train, Fine-tune」到「Pre-train, Prompt, Predict」在多语设定下的含义变化；小模型时代任务型 mPLM 与当下 decoder-only LLM 的对比（配合 Fig.3）。

## §3 Large Language Models with Multilingual Capability（训练）

### §3.1 Training from Scratch

从零训练多语 LLM 的数据、词表与算力考量。

### §3.2 Continual Training

在已有模型上继续预训练/对齐以增强特定语言或领域。

### §3.3 Limitations and Future Directions on Training Paradigm

数据不平衡、低资源语言、三阶段（预训练-SFT-RLHF）各阶段的多语失败模式（与 Fig.1 呼应）。

## §4 Multilingual Inference Strategies（推理侧）

### §4.1 Direct Inference in Multilingual Models

直接以目标语言提示在单模型内推理。

### §4.2 Pre-Translation Inference

先译成高资源语言再推理的流水线及其利弊。

### §4.3 Multilingual CoT

跨语思维链、推理语言选择等。

### §4.4 Code-Switching

语码转换现象及其在提示与生成中的利用与风险。

### §4.5 Multilingual Retrieval Augmented Generation

多语检索器、文档语言混合与 RAG 管线。

### §4.6 Limitations and Future Directions on Inference Strategies

延迟、检索质量、跨语 grounding 等限制。

## §5 Multilingual Information Retrieval

### §5.1 Synthetic Training Data

合成查询-文档对等训练数据。

### §5.2 Multilingual Retrievers

多语稠密/稀疏检索模型与对齐。

### §5.3 Multilingual Rerankers

跨语重排序。

### §5.4 Challenges and Future Directions on Multilingual IR

低资源查询、域偏移与评测。

## §6 Security of Multilingual Large Language Models

### §6.1 Attack Methods

#### §6.1.1–6.1.3

坐标梯度越狱、提示越狱、**多语越狱**（利用非英语绕过安全对齐）。

### §6.2 Defense Methods

#### §6.2.1–6.2.2

开源与闭源模型不同的防御策略。

### §6.3 Limitations and Future Directions on LLM Security

#### §6.3.1–6.3.2

针对多语能力的攻击面与鲁棒多语对齐。

## §7 Multi-Domain LLMs in Multilingual Scenarios

### §7.1 Medical Domain

多语医疗 LLM 与数据稀缺、合规问题。

### §7.2 Legal Domain

多语法律文本与风险。

### §7.3 Limitations and Future Directions on Multi-Domain

非英语领域语料不足、评测标准化等。

## §8 Multilingual Data Resource

预训练、指令、评测等数据资源盘点（与原文表格/列表配合）。

## §9 Multilingual Benchmark and Evaluation

基准与评测协议；跨语可比性。

## §10 Bias and Fairness

### §10.1 Bias Categories

语言、文化、性别等偏见类型。

### §10.2 Multilingual Debias

去偏方法综述。

### §10.3 Limitations and Discussion

公平与性能张力。

## §11 Conclusion and Future Directions

收束全文并列出优先研究方向。

---

## 对照入口

- [[多语言双综述分节映射与对比]] · [[分节详解-大模型时代多语言研究综述-CCL-2024]]
