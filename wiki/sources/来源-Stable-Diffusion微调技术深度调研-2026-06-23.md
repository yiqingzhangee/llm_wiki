# 来源：Stable Diffusion 视觉大模型微调技术深度调研

- **原始路径**：[`raw/transcripts/Stable Diffusion 视觉大模型微调技术深度调研.md`](../../raw/transcripts/Stable%20Diffusion%20视觉大模型微调技术深度调研.md)
- **收录日期**：2026-06-23
- **类型**：技术调研
- **证据等级**：E2（技术调研综述，含大量社区最佳实践与引用文献，但部分数据来自社区文章而非原始论文）

## 摘要

该调研系统梳理了 Stable Diffusion（SD）微调技术从全参数训练到参数高效适配的完整演进图谱，覆盖 LoRA、DreamBooth、Textual Inversion、ControlNet、IP-Adapter 等 9 种代表性方法，并对比了 Diffusers、Kohya_ss、SimpleTuner 等主流训练框架。核心结论：当前 SD 微调生态已具备在消费级硬件（12-24GB VRAM）上完成工业级模型定制的能力，LoRA 系方法为效率核心、ControlNet/IP-Adapter 为控制支柱。

## 关键主张

1. **LoRA 是当前最主流的轻量化微调方案**：通过将权重更新量分解为两个低秩矩阵 `BA`，仅需训练 1% 参数量即可实现风格迁移、角色一致性等任务，文件体积 10-200MB，推理时可合并回原点（零额外开销）。
2. **DreamBooth 解决"主体驱动生成"**：通过稀有词绑定与先验保留损失防止语言漂移，仅需 5-15 张主体图 + 200-1000 张正则化图即可学习特定对象。
3. **Textual Inversion 完全不修改模型权重**：在嵌入空间中搜索新概念的向量表示，文件仅 KB 级别，但表达能力受限于预训练模型的语义覆盖。
4. **ControlNet 通过"零卷积连接"实现安全的空间控制**：克隆一份 U-Net Encoder 副本，通过初始化为零的 1x1 卷积渐进式注入控制信号，训练起始输出与原模型完全一致，避免破坏预训练特征。
5. **IP-Adapter 实现"图像作为提示词"**：通过解耦交叉注意力，为图像特征独立设置 K/V 投影，文本与图像提示互不干扰。
6. **2023-2025 前沿方向**：DoRA 将权重分解为幅度+方向独立优化；OFT 学习正交变换保持超球面能量守恒；LyCORIS 系列将低秩分解扩展到卷积层与 Kronecker/Hadamard 积。
7. **数据集质量远重于数量**：30 张精挑图片效果优于 200 张低质/重复图；SD 1.5 推荐 512px、SDXL/Flux/SD3 推荐 1024px 分辨率。
8. **超参数关键经验**：SD 1.5 LoRA 学习率 1e-4、SDXL 1e-5；简单概念 rank 8-16、复杂风格 rank 64-128；Alpha = Rank/2 为社区共识。
9. **框架选型指南**：Diffusers 适合算法研究、Kohya_ss 适合专业精调、FluxGYM 适合初学者、SimpleTuner 适合多 GPU 大模型场景。
10. **评估指标演进**：从传统 FID/CLIP Score 向 PickScore/HPS 等基于人类偏好的自动评估转变。

## 与现有 Wiki 的关联

- 与 [[来源-基于 Stable Diffusion 架构的视觉大模型代表性工作与原理深度解析-2026-06-23]] 互补：后者偏重架构原理（LDM→SDXL→SD3→FLUX），本页偏重微调方法论。
- 与 [[Stable-Diffusion微调技术矩阵]] 形成"技术原理 → 方法对比"关系。
- 与 [[来源-Qwen2.5与Qwen3微调深度调研-2026-06-21]] 平行对照：分别覆盖视觉模型与语言模型的微调生态演进。

## 反向链接

- [[Stable-Diffusion微调技术矩阵]]
- [[来源-基于 Stable Diffusion 架构的视觉大模型代表性工作与原理深度解析-2026-06-23]]
