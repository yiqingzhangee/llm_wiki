# 来源：DeepEvidence——生物医学知识图谱深度研究（2026）

- **原始路径**：
  - [`raw/articles/2026-DeepEvidence Empowering Biomedical Discovery with Deep Knowledge Graph Research.pdf`](../../../raw/articles/2026-DeepEvidence%20Empowering%20Biomedical%20Discovery%20with%20Deep%20Knowledge%20Graph%20Research.pdf)
  - [`raw/articles/DeepEvidence：以深度知识图谱研究赋能生物医学发现（中文译本）.md`](../../../raw/articles/DeepEvidence%EF%BC%9A%E4%BB%A5%E6%B7%B1%E5%BA%A6%E7%9F%A5%E8%AF%86%E5%9B%BE%E8%B0%B1%E7%A0%94%E7%A9%B6%E8%B5%8B%E8%83%BD%E7%94%9F%E7%89%A9%E5%8C%BB%E5%AD%A6%E5%8F%91%E7%8E%B0%EF%BC%88%E4%B8%AD%E6%96%87%E8%AF%91%E6%9C%AC%EF%BC%89.md)
- **类型**：系统论文（生物医学深度研究 Agent / 知识图谱驱动多 Agent 框架）
- **同步日期**：2026-09-02
- **证据等级**：**E2**（系统设计 + 多基准实验 + 四类生物医学任务验证；定量结果为作者报告口径，泛化与专家级上限仍有限）

## 一句话摘要

**DeepEvidence** 把“deep research”从通用网页检索推进到**异构生物医学知识图谱上的多 Agent 深度研究**：由编排器调度 **BFRS（广度优先研究）** 与 **DFRS（深度优先研究）** 两类研究智能体，在统一 KG 工具与代码执行沙箱中逐步构建**证据图谱**，以支撑药物发现、临床前机制推理、临床试验设计和循证医学更新等任务。

## 核心架构

### 1. 编排器 + 双研究子智能体

- **编排器（orchestrator）**：接收研究问题，决定 BFRS / DFRS 的执行顺序、研究目标与预算，并维护共享证据图谱。
- **BFRS**：做跨多图谱的一跳广覆盖搜索，快速找候选实体、关系与桥接点。
- **DFRS**：沿引文链、实体关系和机制链路做多跳深挖，补足系统性文献综述与证据聚合所需的深层证据。

作者明确将其称为 **BFRS / DFRS** 而非传统 BFS / DFS，因为智能体会自主决定下一步动作，而不是按固定图遍历规则机械展开。

### 2. 证据图谱作为一等记忆

系统不会把中间结果只堆在自由文本上下文里，而是把论文、药物、基因、通路、试验及其关系持续写入**证据图谱**：

- 节点：实体、关系、具体观察与来源证据。
- 用途：支撑后续研究优先级排序、跨图谱桥接、冲突证据并存与可追溯审查。
- 价值：把“研究过程”外化为可检查、可复用、可扩展的结构化记忆。

这使 DeepEvidence 更像“知识图谱驱动的研究工作台”，而不只是带搜索的问答系统。

### 3. 统一生物医学 KG 工具 + 代码执行沙箱

系统集成 17 个生物医学知识库 / API，包括 BioThings、ChEMBL、ClinicalTrials.gov、GO、HPO、KEGG、OpenFDA、Open Targets、ProteinAtlas、PubChem、PubTator、PubMed、Reactome、UMLS、UniProt 等，并提供统一搜索工具：

- `search_genes_unified`
- `search_drugs_unified`
- `search_targets_unified`
- `search_pathways_unified`
- `search_compounds_unified`

同时，智能体通过**代码执行沙箱**直接写 Python 做 API 调用、DataFrame 清洗、统计分析、CSV 持久化与后续筛选，而不是受限于静态工具返回。这一点把“工具调用”升级为“可编程研究环境”。

## 主要实验结果

### 开放基准

在 4 个生物医学研究基准上，DeepEvidence 均优于对照的 GPT-5、Sonnet-4.5、Biomni、ToolUniverse：

| 基准 | DeepEvidence | 其他代表值 |
|---|---:|---:|
| HLE-Medicine | **40.0%** | Biomni 20.0%，GPT-5 3.3% |
| LabBench-LitQA2 | **80.0%** | GPT-5 48.0%，Biomni 32.0% |
| SuperGPQA-Medicine-Hard | **47.1%** | Sonnet-4.5 43.6%，Biomni 40.7% |
| TrialPanorama-EvidenceQA | **96.0%** | Sonnet-4.5 88.0%，Biomni 84.0% |

### 生物医学发现生命周期四阶段任务

1. **药物发现 / 靶点识别**：准确率 **68%**，高于 Biomni 56%、ToolUniverse 40%、通用 LLM 42%。
2. **临床前机制推理**：准确率 **72%**，高于 Biomni / ToolUniverse 的 44% 与通用 LLM 的 52%。
3. **体内代谢通量反应预测**：准确率 **80%**，高于 Biomni 68%、通用 LLM 60%、ToolUniverse 52%。
4. **临床试验开发**：
   - 样本量估计：**68%**（Biomni 20%，ToolUniverse 32%，LLM 24%）
   - 给药方案设计：**52%**（Biomni 36%，ToolUniverse 28%，LLM 20%）
   - 替代终点发现：F1 **73.3%**（Biomni 60.9%，ToolUniverse 61.9%，LLM 58.7%）
5. **循证医学更新 / 证据缺口发现**：
   - 缺口检出率：**90.0%**
   - Recall@30：**44.14%**
   明显高于 Biomni、ToolUniverse 与具 PubMed 搜索能力的 LLM。

## 方法论价值

### 1. 把 Deep Research 从“互联网网页”推进到“异构知识图谱”

论文的核心主张不是“再做一个研究 Agent”，而是指出：在生物医学里，仅依赖网页与纯文本语料不足以支撑真正高质量的研究，必须让智能体显式接入并跨越**文献、基因、通路、疾病、药物、临床试验**等异构知识图谱。

### 2. 证据图谱记忆比纯上下文堆叠更适合长程研究

与 [[Agent记忆系统与自我演进]] 的“外部结构化记忆”路线强呼应：DeepEvidence 用证据图谱把实体、关系、冲突和来源保留为结构化对象，天然适合长流程、多跳、多轮修正的研究任务。

### 3. 协作结构与工具环境同等重要

系统性能不仅来自“多 Agent”，而是来自以下三者组合：

- **协作拓扑**：编排器 + BFRS/DFRS 的角色分工；
- **统一工具层**：跨 KG 检索与关系查询；
- **可执行研究环境**：代码沙箱处理真实数据。

这与本仓库既有“协作拓扑 → 工程编排 / 状态管理”的主线一致，但 DeepEvidence 更进一步强调**证据记忆**与**领域知识基座**。

## 与现有 wiki 的关系

- **新增概念页 [[知识图谱驱动的深度研究Agent]]**：本论文是该概念页的直接来源，补入“证据图谱 + BFRS/DFRS + 统一 KG 工具 + 代码沙箱”的完整框架。
- **补强 [[多Agent系统研究趋势与工程映射]]**：DeepEvidence 是“层级编排 + 专职研究子 Agent + 共享结构化记忆”的高质量垂域实例，能把抽象的协作拓扑映射到医学研究任务。
- **补强 [[多Agent论文方向对比-2026H1-arXiv]]**：可把“编排与系统结构”从摘要级讨论推进到具体系统能力：证据图谱、桥接实体、统一 KG 搜索、研究预算控制。
- **与 [[Agent记忆系统与自我演进]] 互证**：证据图谱是一种外部结构化记忆；冲突发现并存、来源可追踪，是“可信记忆”的工程例子。
- **与 [[时间序列LLM智能体分类法]] 形成跨域对照**：两者都不把 LLM 当静态编码器，而当作会选工具、会记忆、会规划的 Agent；差异在于 DeepEvidence 强调跨异构 KG 的知识桥接，时序 Agent 更强调与统计模型/分析器协作。

## 冲突与不确定性

- **领域口径不可横比**：论文中的高分来自生物医学专门任务与知识库生态，不能直接外推为“通用 deep research”结论。
- **与通用网页型 deep research 的对照不完全公平**：对照系统虽然共享底层 LLM，但工具环境和知识源不同；优势中既有“架构改进”，也有“领域资源更强”的成分。
- **作者自承仍未达专家级**：系统覆盖的知识图谱仍不完整，对专有/内部图谱、缺失字段、不稳定接口的适应能力仍待增强。
- **raw 同时包含 PDF 与中文译本**：本页主要依据中文译本归纳；若后续出现术语歧义，应以英文 PDF 原文为准。

## 反向链接

- [[知识图谱驱动的深度研究Agent]]
- [[多Agent系统研究趋势与工程映射]]
- [[多Agent论文方向对比-2026H1-arXiv]]
- [[Agent记忆系统与自我演进]]
- [[时间序列LLM智能体分类法]]
- [[研究主题导航]]
