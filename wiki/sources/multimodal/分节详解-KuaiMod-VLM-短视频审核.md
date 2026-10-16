# 分节详解：VLM as Policy — KuaiMod（短视频内容审核）

> 对应来源页：[[来源-KuaiMod-VLM-短视频审核]] · 原始：`raw/articles/MinerU_markdown_VLM_as_Policy-_Common-Law_Content_Moderation_Framework_for_Short_Video_Platform_2043710410561568768.md`  
> 本文**非综述**，章节为单篇系统论文结构；与六篇综述对比时见 [[七篇文献体例与覆盖面对照]]。

## §1 INTRODUCTION

短视频平台规模与有害内容风险；传统「大陆法系」式固定规则 + 人工审核的偏差与成本；自动化方法的准确率与多模态理解不足；工业规则更新慢。引入**判例法**隐喻：**KuaiMod** 以案例驱动、可快速迭代；三组件为数据构建、离线适配、在线部署与精炼；汇报基准、离线实验与线上 A/B。

## §2 RELATED WORKS

### §2.1 Content Moderation for Online Platform

毒性内容、众包审核、规则与浅层分类器、工业系统（Perspective、Twitter、阿里等）及其在多模态与实时性上的局限。

### §2.2 Language Models for Content Moderation

从 BERT/GPT 微调检测到 GPT-4 零样本审核；**VLM** 用于多模态审核的进展与泛化、实时性瓶颈。

## §3 BENCHMARK

### §3.1 Task Definition for Video Moderation

视频有害性判定任务形式、输入输出与评价目标。

### §3.2 KuaiMod Taxonomy

违规类目与标签体系（taxonomy）。

### §3.3 KuaiMod Benchmark

422 类有害视频等规模与构造说明、开源地址（见原文）。

## §4 METHODOLOGY

### §4.1 Overview of YuanQi Model

#### §4.1.1 Architecture & Modules

平台自研 **YuanQi VLM** 结构与子模块。

#### §4.1.2 Training

预训练与通用对齐阶段要点。

#### §4.1.3 Performance & Application

基座能力与应用场景。

### §4.2 Construction of Training Data

#### §4.2.1 Data Collection and Annotation

真实用户反馈与审核标注流程。

#### §4.2.2 Generate State-Transition Data with Tag2CoT and CoT2Tag

标签↔思维链互转，构造带理由的监督数据。

### §4.3 Stage I: Offline Adaptation Training

#### §4.3.1 Large-scale SFT

大规模监督微调使 VLM 适应审核判例风格。

#### §4.3.2 Mistake-oriented DPO

针对错误样本的 DPO 式偏好优化。

### §4.4 Stage II: Online Deployment & Refinement

#### §4.4.1 RLUF Definition for KuaiMod Online Refinement

在线强化/反馈学习框架定义（RLUF）。

#### §4.4.2 RLUF Data construction

在线反馈如何形成训练对。

#### §4.4.3 Online RLUF training

持续精炼策略频率与流程（与「日更」叙事对应）。

## §5 OFFLINE EVALUATION

### §5.1 Experimental Setup

#### §5.1.1–5.1.4

数据集划分、指标、基线（含 LLM/VLM）、实现细节。

### §5.2 Evaluation Result

主表结果与相对基线提升。

### §5.3 Analysis Experiments

#### §5.3.1 Ablation study

各模块贡献。

## §6 ONLINE APPLICATIONS

### §6.1 Deployment Details

线上Serving与规模。

### §6.2 Performance of A/B test

#### §6.2.1 Comprehensive Ecosystem Governance

举报率下降等治理指标。

#### §6.2.2 Personalized Recommendation Enhancement

推荐场景 DAU、使用时长等。

## §7 CONCLUSION

贡献回顾与局限。

## Appendix（附录，若阅读）

指令模板、案例研究、伦理声明等（原文 A 节）。

---

## 对照入口

- [[七篇文献体例与覆盖面对照]] · [[短视频内容审核与多模态治理]]
