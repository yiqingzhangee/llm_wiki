# Transformer 效率技术对照-2026

> 关联概念：[[Transformer效率演进]]。数据取自各 [[Transformer效率演进]] 与论文原文；**一行为一项技术**。

## 核心对照表

| 技术 | 年份 | 机构 | 改进环节 | 核心手段 | 主要量化收益 | 关键取舍 |
|---|---|---|---|---|---|---|
| Transformer | 2017 | Google | — | 全量 self-attention（基线） | 并able、翻译 SOTA | $O(n^2)$ 起点 |
| Sparse Gated MoE | 2017 | Google Brain | ⑦ FFN | 门控路由 top-k 专家 | 参数 ×E、算力 ×k | 负载均衡 / 通信 |
| Universal Transformer | 2018 | Google/DeepMind | 整体 | 权重共享循环 + ACT | 深度 ×T、参数不变 | 长上下文扩展性 |
| 线性注意力（Transformers are RNNs） | 2020 | Idiap/EPFL | ③④ | 核特征 + 定长状态 $S_t$ | $O(n^2)\to O(n)$ | 朴素近似精度不足 |
| MLA（DeepSeek-V2） | 2024 | DeepSeek | ② 投影/KV | 低秩隐向量压缩 KV | KV −93.3%、吞吐 +5.76× | 需带宽还原 KV |
| NSA（Native Sparse Attention） | 2025.02 | DeepSeek | ③④ | 压缩+选择+滑窗三支 | 64k 三阶段显著加速 | 近似挑块 |
| MoBA | 2025.02 | Moonshot（月之暗面） | ③④ | MoE 路由→块级 top-k | 全量↔稀疏平滑切换 | 门控打分启发 |
| KDA（Kimi Linear） | 2025.10 | Moonshot（月之暗面） | ③④ | 定长状态 + 细门控 + 约束 DPLR | 1M 上下文吞吐 +6×、KV −75% | 混合比例/超参敏感 |
| DeepSeek-V4 CSA/HCA | 2026.04 | DeepSeek | ② KV | 4合1 稀疏 + 128合1 全量 + 最近 128 原 token | 1M 上下文 KV ~2%、单 token FLOPs 至 27% | FP4 KV 保真 |
| MiniMax MSA | 2026.06 | MiniMax | ③④ | GQA + 每组 top-16 块 + KV-outer | 1M 每 token 注意力 −28.4×、14.2×/7.6× 加速 | 挑块保真 / H800 测量 |

## 演进主线速览

- **注意力打分**：$(2017\text{ 全量})\to 2020\text{ 线性}\to 2025\text{(MoBA/NSA/DSA)}\to 2026\text{ MSA}$：从"算全部"走向"定长状态"或"只算一部分块"。
- **KV 压缩**：$\text{MLA}(2024)\to\text{CSA/HCA}(2026)$：从无损低秩压 KV，走向"块级压缩 + 稀疏/全量 + 保留近期原 token"。
- **整体结构**：$\text{MoE}(2017)\to\text{权重循环}(2018)\to\text{CED}(2026)$：参数/算力解耦与整体架构重排。
- **2026 主流形态**：KDA（3 KDA:1 MLA）与 DeepSeek-V4（HCA 前层 + CSA/HCA 交替）均为**混合**：局部精确 + 全局压缩，而非单一技术。

## 选型建议

- **要极低 KV 内存 + 百万上下文**：优先 DeepSeek-V4（CSA/HCA）或 MiniMax MSA（都有生产级部署与/或开源权重）。
- **要完全替换全量注意力且少降质量**：KDA（Kimi Linear）是目前"相同配方下全面超全注意力"的最强单一证据，但依赖其混合配方。
- **想 drop-in 低频顾虑**：MoBA 的"全量↔稀疏无缝切换"适合渐进上线；NSA 侧重可端到端训练 + 硬件对齐。
- **追根溯源**：三者分别对应的经典源头是 MoE（Sparse Gated MoE）、线性注意力（Transformers are RNNs）、MLA。

## 冲突与不确定性

- 各论文报告收益的**硬件/配置/任务口径不同**（H800、FLOPS vs KV vs 吞吐 vs 墙钟），不能横向拼成统一排行榜。
- "稀疏/近似挑块"和"FP4 低比特 KV"的长程精确检索保真仍是开放质疑；在线/长程端可靠性应回原文与后续版本核验。
