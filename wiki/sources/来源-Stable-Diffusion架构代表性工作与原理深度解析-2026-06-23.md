# 来源：基于 Stable Diffusion 架构的视觉大模型代表性工作与原理深度解析

- **原始路径**：[`raw/transcripts/基于 Stable Diffusion 架构的视觉大模型代表性工作与原理深度解析.md`](../../raw/transcripts/基于%20Stable%20Diffusion%20架构的视觉大模型代表性工作与原理深度解析.md)
- **收录日期**：2026-06-23
- **类型**：技术解析
- **证据等级**：E2（系统性架构解析，含伪代码实现与公式推导，但部分解释来自二手资料）

## 摘要

该文档对 2022-2024 年间基于 Stable Diffusion 架构的 8 个代表性工作进行了深度技术剖析，涵盖 LDM 奠基范式、SD 2.x 的 v-prediction 升级、SDXL 的双编码器与微条件化、ControlNet 的零卷积设计、IP-Adapter 的解耦注意力、LoRA 低秩分解、SD3 的 MMDiT 架构革命以及 FLUX 的 12B 混合流 Transformer，每条均配有关键代码解读。

## 关键主张

1. **LDM 将扩散过程转移到潜空间**：VAE 将图像压缩 8 倍（512x512x3 → 64x64x4），计算量降低约 64 倍，是三组件架构（VAE + U-Net + CLIP）的核心效率保障。
2. **SD 2.x 的 v-prediction 替代 epsilon-prediction**：速度预测 `v_t = alpha_t * epsilon - sigma_t * z_0` 在高噪声水平下数值更稳定，梯度方差更小。
3. **SDXL 三大核心改进**：双文本编码器（CLIP-L + OpenCLIP-G 拼接 2048 维）、微条件化（6 维向量记录原始/裁剪/目标尺寸）、Base + Refiner 两阶段流水线。
4. **ControlNet 的零卷积设计是稳定训练的关键**：通过初始化为 0 的 1x1 卷积连接副本与主路，训练开始时 ControlNet 输出为零，控制信号渐进注入。
5. **IP-Adapter 的"解耦交叉注意力"**：文本与图像各自独立 K-V 投影，公式为 `Z_new = Attn(Q, K_text, V_text) + lambda * Attn(Q, K_img, V_img)`，仅训练图像投影模块与解耦注意力层。
6. **SD3 的 MMDiT 是架构范式革新**：彻底摒弃 U-Net，引入双流联合注意力 Transformer，图像与文本对等信息交换；使用整流流（Rectified Flow）替代离散噪声调度，理论上可单步生成。
7. **FLUX 的双流→单流混合设计**：前 19 层双流保持模态独立性，后 38 层单流深度融合；引入 2D RoPE 天然支持任意分辨率与长宽比。
8. **SVD 三阶段训练范式**：图像预训练（建立空间能力）→ 视频预训练（学习运动规律）→ 高清微调（细化质量），并通过时序卷积/时序注意力感知帧间关系。
9. **技术趋势判断**：Transformer 取代 U-Net 已成定局、整流流替代 DDPM、RoPE 解锁分辨率自由度、蒸馏技术实现 1-4 步生成。
10. **代码解读覆盖所有关键环节**：VAE 编解码、交叉注意力、DDIM 采样、双编码器融合、零卷积、LoRA 合并、MMDiT Block、单流块 + RoPE 应用。

## 与现有 Wiki 的关联

- 与 [[来源-Stable-Diffusion微调技术深度调研-2026-06-23]] 互补：本页提供架构/原理层面的地基知识，后者提供微调/应用层面方法论。
- 为 [[Stable-Diffusion微调技术矩阵]] 中"哪些架构支持哪些微调方法"提供架构上下文。
- 可与 Agent/风控领域形成"视觉模型能力边界"的交叉参考。

## 反向链接

- [[Stable-Diffusion微调技术矩阵]]
- [[来源-Stable-Diffusion微调技术深度调研-2026-06-23]]
