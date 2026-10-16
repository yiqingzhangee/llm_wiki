# 基于 Stable Diffusion 架构的视觉大模型代表性工作与原理深度解析

## 概述与技术演进路线

基于 Stable Diffusion 架构的视觉大模型在 2022\-2024 年间经历了从 U\-Net 到 Transformer 的范式跃迁，形成了覆盖图像生成、可控编辑、视频合成的完整技术谱系。本文对其中 **8 个代表性工作** 进行深度技术剖析，涵盖核心原理与关键代码实现。

**技术演进主线：** LDM \(2022\) → SD 1\.x/2\.x → SDXL \(2023\) → ControlNet \(2023\) → IP\-Adapter \(2023\) → LoRA → SD3/MMDiT \(2024\) → FLUX \(2024\) → SVD \(视频\)

|**模型**|**发布时间/团队**|**核心架构**|**关键创新**|**定位**|
|---|---|---|---|---|
|LDM / SD 1\.x|2022 / CompVis \+ Stability AI|VAE \+ U\-Net \+ CLIP|潜空间扩散|基础范式建立|
|SD 2\.x|2022 / Stability AI|VAE \+ U\-Net \+ OpenCLIP|v\-prediction, 768px|质量提升|
|SDXL|2023 / Stability AI|双编码器 \+ 2\.6B U\-Net|微条件化, Refiner|产业级生成|
|ControlNet|2023 / Zhang et al\.|冻结U\-Net \+ 可训练副本|零卷积空间控制|精准可控生成|
|IP\-Adapter|2023 / Tencent AI Lab|解耦交叉注意力|图像提示词适配|多模态条件注入|
|LoRA|2021\(原始\) / 社区广泛应用|低秩矩阵分解|高效参数微调|个性化定制|
|SD3|2024 / Stability AI|MMDiT \(多模态扩散Transformer\)|联合注意力 \+ 整流流|架构范式革新|
|FLUX|2024 / Black Forest Labs|12B 混合流 Transformer|双流\+单流 \+ RoPE|当前SOTA|

---

## 一、潜在扩散模型 \(LDM\) — 奠基性范式

### 1\.1 为什么在潜空间工作

传统扩散模型（DDPM）直接在像素空间操作。对于 $512 \\times 512$ 的图像，U\-Net 需要处理 $786432$ 维的输入，计算代价极为昂贵。LDM 的核心贡献在于将扩散过程转移到低维潜空间[\[1\]](https://openaccess.thecvf.com/content/CVPR2022/papers/Rombach_High-Resolution_Image_Synthesis_With_Latent_Diffusion_Models_CVPR_2022_paper.pdf)。

**像素空间扩散的瓶颈：**

- 计算复杂度 $O(H \times W)$ 直接作用于全分辨率

- 单张 512px 图像推理需要数十 GB 显存

- 大部分计算花在感知上无意义的高频细节

**LDM 的解决方案：**

- 利用 VAE 预先将图像压缩 $8\\times$

- $512 \\times 512 \\times 3 \\rightarrow 64 \\times 64 \\times 4$

- 扩散过程在压缩后的语义空间执行

- 计算量降低约 $64\\times$

### 1\.2 三组件架构

LDM 由三个核心模块构成[\[2\]](https://arxiv.org/abs/2112.10752)：

**1\) 变分自编码器 \(VAE\)** — 感知压缩器

VAE 的 Encoder $E$ 将图像投影到潜空间，Decoder $D$ 负责还原。压缩后的表示保留了图像的语义结构，但去除了高频噪声[\[3\]](https://aiwiki.ai/wiki/latent_diffusion)。

**2\) U\-Net 去噪网络** — 核心生成引擎

U\-Net 在潜空间中迭代预测并去除噪声。其内部包含 ResNet 残差块（捕捉局部特征）和 Cross\-Attention 交叉注意力块（引入文本条件）[\[4\]](https://blog.csdn.net/qq_34941290/article/details/157694897)。

**3\) 条件编码器** — 语义桥梁

CLIP 文本编码器 $\tau_\theta$ 将自然语言提示词映射为连续嵌入向量，通过交叉注意力机制注入 U\-Net[\[5\]](https://deepwiki.com/CompVis/stable-diffusion/4.4-text-conditioning)。

### 1\.3 扩散过程数学原理

**前向过程（加噪）：**$z_t = \sqrt{\bar{\alpha}_t}\, z_0 + \sqrt{1 - \bar{\alpha}_t}\, \epsilon, \quad \epsilon \sim \mathcal{N}(0, I)$

**训练目标（噪声预测）：**$\mathcal{L}_{LDM} = \mathbb{E}_{z \sim E(x),\, \epsilon,\, t}\left[\|\epsilon - \epsilon_\theta(z_t, t, \tau_\theta(y))\|_2^2\right]$

模型学习在给定时间步 $t$ 和文本条件 $y$ 下，从含噪潜变量 $z_t$ 中预测所加噪声 $\epsilon$[\[2\]](https://arxiv.org/abs/2112.10752)。

### 1\.4 关键代码解读

#### VAE 编码/解码

```Python
# VAE 编码：将图像压缩到潜空间
def encode(vae, image):
    # image: [B, 3, 512, 512] → 潜变量分布
    posterior = vae.encode(image).latent_dist
    latents = posterior.sample()
    # 缩放因子 0.18215 确保潜变量标准差 ≈ 1
    return latents * 0.18215  # [B, 4, 64, 64]

# VAE 解码：从潜空间还原图像
def decode(vae, latents):
    latents = latents / 0.18215
    image = vae.decode(latents).sample  # [B, 3, 512, 512]
    return image

```

缩放因子 `0.18215` 是根据训练数据统计得出的，目的是将 VAE 输出的潜变量规范化到标准正态分布附近，使扩散过程的噪声调度更加稳定[\[6\]](https://blog.csdn.net/qq_35899016/article/details/149697368)。

#### U\-Net 交叉注意力机制

```Python
class CrossAttention(nn.Module):
    """文本引导图像生成的核心桥梁"""
    def __init__(self, query_dim, context_dim, inner_dim, num_heads):
        super().__init__()
        self.scale = (inner_dim // num_heads) ** -0.5
        self.to_q = nn.Linear(query_dim, inner_dim, bias=False)
        self.to_k = nn.Linear(context_dim, inner_dim, bias=False)
        self.to_v = nn.Linear(context_dim, inner_dim, bias=False)

    def forward(self, x, context):
        # x: U-Net 中间特征 [B, HW, C]
        # context: CLIP 文本嵌入 [B, 77, 768]
        q = self.to_q(x)
        k = self.to_k(context)   # 文本信息成为 Key
        v = self.to_v(context)   # 文本信息成为 Value
        
        # 注意力权重计算
        sim = torch.einsum('b i d, b j d -> b i j', q, k) * self.scale
        attn = sim.softmax(dim=-1)
        
        # 加权聚合：文本语义融入图像特征
        out = torch.einsum('b i j, b j d -> b i d', attn, v)
        return out

```

这里 Query 来自 U\-Net 的图像特征，Key/Value 来自文本编码器——每个空间位置都可以"查询"最相关的文本语义[\[7\]](https://blog.csdn.net/gitblog_01128/article/details/152100877)。

#### DDIM 采样循环（推理核心）

```Python
@torch.no_grad()
def sample_loop(unet, scheduler, latents, text_embeddings, guidance_scale=7.5):
    for t in scheduler.timesteps:
        # 无分类器引导 (CFG)：同时计算有条件/无条件预测
        latent_input = torch.cat([latents] * 2)
        noise_pred = unet(latent_input, t, 
                         encoder_hidden_states=text_embeddings).sample
        
        # 引导公式：增强文本对应方向的生成
        noise_uncond, noise_text = noise_pred.chunk(2)
        noise_pred = noise_uncond + guidance_scale * (noise_text - noise_uncond)
        
        # 调度器执行单步去噪
        latents = scheduler.step(noise_pred, t, latents).prev_sample
    return latents

```

`guidance_scale` 控制生成与文本的一致程度，典型值 7\.5 在质量与多样性之间取得平衡[\[8\]](https://blog.csdn.net/weixin_46763552/article/details/148669723)。

---

## 二、Stable Diffusion 2\.x — v\-prediction 与高分辨率

### 2\.1 与 SD 1\.5 的关键差异

|**特性**|**SD 1\.5**|**SD 2\.0/2\.1**|
|---|---|---|
|文本编码器|OpenAI CLIP ViT\-L/14 \(768维\)|OpenCLIP ViT\-H/14 \(1024维\)|
|训练分辨率|512 × 512|768 × 768|
|预测范式|$\epsilon$\-prediction \(噪声预测\)|v\-prediction \(速度预测\)|
|训练数据|LAION\-2B \(过滤较宽松\)|LAION\-5B \(NSFW严格过滤\)|

### 2\.2 v\-prediction 的数学含义

SD 2\.x 引入的 v\-prediction 定义速度变量[\[9\]](https://arxiv.org/html/2407.03297v1)：$v_t \equiv \alpha_t \epsilon - \sigma_t z_0$

相比 $\epsilon$\-prediction，v\-prediction 在高噪声水平（$t$ 接近 1）时数值更稳定，梯度方差更小，有助于模型学习高频结构[\[10\]](https://blog.csdn.net/julac/article/details/143204324)。

---

## 三、SDXL — 产业级文本到图像生成

### 3\.1 架构跃升

SDXL 是 Stability AI 于 2023 年推出的第二代模型，参数量从 SD 1\.5 的 8\.6 亿跃升至 **26 亿** \(Base\) \+ **66 亿** \(Refiner\)[\[11\]](https://www.reddit.com/r/StableDiffusion/comments/15fj5k9/sdxl_10_a_semitechnical_introductionsummary_for/)。

**三大核心改进：**

**1\) 双文本编码器系统**

同时使用 CLIP ViT\-L/14 \(768维\) 和 OpenCLIP ViT\-bigG/14 \(1280维\)。特征在最后一维拼接得到 2048 维向量，作为交叉注意力的上下文输入[\[12\]](https://www.xta0.me/2025/01/20/GenAI-Stable-Diffusion-SDXL.html)：$\tau_\theta(y) = \text{Concat}(\tau_{\text{CLIP-L}}(y)_{[-2]},\; \tau_{\text{OpenCLIP-G}}(y)_{[-2]})$

**2\) 微条件化 \(Micro\-Conditioning\)**

训练时将图像的原始尺寸、裁剪坐标、目标尺寸作为额外条件注入模型，避免因数据增强引入的分布偏差[\[13\]](https://arxiv.org/pdf/2307.01952)。

**3\) Base \+ Refiner 两阶段流水线**

Base 模型完成主体结构生成，Refiner 在最后 20% 步骤细化纹理。

### 3\.2 关键代码解读

#### 双编码器特征融合

```Python
# 从两个编码器获取隐藏状态
# CLIP-L: [batch, 77, 768], OpenCLIP-G: [batch, 77, 1280]
prompt_embeds = torch.cat([
    text_encoder_1(prompt).hidden_states[-2],  # 倒数第二层
    text_encoder_2(prompt).hidden_states[-2]
], dim=-1)  # → [batch, 77, 2048]

# 池化输出用于全局向量条件化
pooled_embeds = text_encoder_2(prompt).pooler_output  # [batch, 1280]

```

取倒数第二层（而非最后一层）是因为 CLIP 最后一层专为对比学习优化，倒数第二层包含更丰富的语义信息[\[12\]](https://www.xta0.me/2025/01/20/GenAI-Stable-Diffusion-SDXL.html)。

#### 微条件化实现

```Python
def compute_time_ids(original_size, crops_coords_top_left, target_size):
    """构造6维微条件向量: (原高, 原宽, 裁剪y, 裁剪x, 目标高, 目标宽)"""
    add_time_ids = list(original_size + crops_coords_top_left + target_size)
    return torch.tensor([add_time_ids], dtype=torch.float32)

# 推理时：告诉模型"这是一张1024x1024的完整图像"
time_ids = compute_time_ids(
    original_size=(1024, 1024),
    crops_coords_top_left=(0, 0),   # 无裁剪
    target_size=(1024, 1024)
)
# time_ids 与 pooled_embeds 一起注入 U-Net 的时间步嵌入

```

推理时设置 `crops_coords=(0,0)` 可以避免模型生成"被裁剪"风格的图像[\[14\]](https://blog.csdn.net/LittleNyima/article/details/150503020)。

---

## 四、ControlNet — 零卷积空间控制

### 4\.1 设计哲学

ControlNet 解决了一个关键问题：**如何在不破坏预训练模型生成能力的前提下，添加精确的空间控制？**[\[15\]](https://arxiv.org/abs/2302.05543)

其解决方案极为优雅：

- **冻结原模型**：SD U\-Net 权重完全锁定

- **创建可训练副本**：克隆 U\-Net 编码器部分

- **零卷积连接**：通过初始化为 0 的卷积层连接两者

### 4\.2 零卷积的数学直觉

设 $\mathcal{F}$ 为原 U\-Net 层，$\mathcal{Z}$ 为零卷积操作：$y = \mathcal{F}(x;\Theta) + \mathcal{Z}(\mathcal{F}(x + \mathcal{Z}(c);\Theta_{copy}))$

训练开始时 $\mathcal{Z}=0$，ControlNet 输出为零，不干扰原模型。随着训练推进，控制信号被**渐进式**注入[\[16\]](https://disassemble-channel.com/controlnet/)。

### 4\.3 关键代码解读

#### 零卷积层

```Python
def zero_module(module):
    """将模块所有参数初始化为零"""
    for p in module.parameters():
        p.detach().zero_()
    return module

# 创建零卷积：1×1 卷积，权重和偏置均为 0
zero_conv = zero_module(nn.Conv2d(channels, channels, kernel_size=1))
# 训练初期，zero_conv(x) ≡ 0，保证安全启动

```

这个设计看似简单，却是 ControlNet 能够稳定训练的关键保障——它创造了一个"安全的学习起点"[\[17\]](https://blog.csdn.net/gitblog_01124/article/details/152098649)。

#### ControlNet 推理流程

```Python
# Step 1: ControlNet 处理控制图像（如 Canny 边缘图）
down_residuals, mid_residual = controlnet(
    noisy_latents,
    timestep,
    encoder_hidden_states=prompt_embeds,
    controlnet_cond=canny_image  # 控制条件输入
)

# Step 2: 将控制残差注入冻结的 U-Net
noise_pred = unet(
    noisy_latents,
    timestep,
    encoder_hidden_states=prompt_embeds,
    down_block_additional_residuals=down_residuals,  # 编码器跳跃连接
    mid_block_additional_residual=mid_residual       # 中间块
)

```

ControlNet 的输出通过跳跃连接累加到 U\-Net 各层级，实现了多尺度的空间控制[\[18\]](https://blog.csdn.net/weixin_54338498/article/details/134489927)。

---

## 五、IP\-Adapter — 图像提示词解耦注入

### 5\.1 解耦交叉注意力

IP\-Adapter 的核心洞察：文本和图像是两种本质不同的条件，不应共享同一组 Key/Value 投影权重[\[19\]](https://huggingface.co/papers?q=IP-Adapter)。

**解耦公式：**$Z_{new} = \text{Attn}(Q, K_{text}, V_{text}) + \lambda \cdot \text{Attn}(Q, K'_{image}, V'_{image})$

其中 $\lambda$ 为图像影响力权重（推理时可调），$K'$ 和 $V'$ 来自独立的图像投影层[\[20\]](https://kimjy99.github.io/%EB%85%BC%EB%AC%B8%EB%A6%AC%EB%B7%B0/ip-adapter/)。

### 5\.2 架构：最小侵入式设计

**IP\-Adapter 仅训练两个轻量组件：**

1. 图像投影模块 — 将 CLIP 图像特征映射为 cross\-attention 可用的 Token

2. 解耦注意力层 — 为图像特征独立设置的 K/V 投影

原始 U\-Net 和 CLIP 文本编码器完全冻结。

### 5\.3 关键代码解读

#### 图像投影模块

```Python
class ImageProjection(nn.Module):
    """将 CLIP 图像特征映射为可注入 cross-attention 的 Token 序列"""
    def __init__(self, clip_embed_dim=1024, cross_attention_dim=768, num_tokens=4):
        super().__init__()
        self.num_tokens = num_tokens
        # 单个线性层投影到 num_tokens 个 Token
        self.proj = nn.Linear(clip_embed_dim, num_tokens * cross_attention_dim)
        self.norm = nn.LayerNorm(cross_attention_dim)

    def forward(self, image_embeds):
        # image_embeds: [B, 1024] (CLIP ViT-H 全局特征)
        projected = self.proj(image_embeds)             # [B, 4*768]
        projected = projected.view(-1, self.num_tokens, -1)  # [B, 4, 768]
        return self.norm(projected)  # 4 个 image tokens

```

`num_tokens=4` 意味着一张参考图像被编码为 4 个"虚拟文本 Token"，与 77 个文本 Token 在不同的注意力头中被分别查询[\[21\]](https://diffengine.readthedocs.io/en/stable/_modules/diffengine/models/archs/ip_adapter.html)。

#### 解耦注意力前向传播

```Python
class IPAttnProcessor(nn.Module):
    """替换 U-Net 中的注意力处理器，实现解耦注入"""
    def __init__(self, hidden_size, cross_attention_dim, num_tokens):
        super().__init__()
        # 图像特征独立的 K/V 投影（不与文本共享）
        self.to_k_ip = nn.Linear(cross_attention_dim, hidden_size, bias=False)
        self.to_v_ip = nn.Linear(cross_attention_dim, hidden_size, bias=False)

    def forward(self, attn, hidden_states, encoder_hidden_states, ip_hidden_states, scale=1.0):
        # 标准文本注意力（保持原模型行为）
        q = attn.to_q(hidden_states)
        k = attn.to_k(encoder_hidden_states)
        v = attn.to_v(encoder_hidden_states)
        text_out = scaled_dot_product_attention(q, k, v)
        
        # 图像注意力（解耦路径）
        k_ip = self.to_k_ip(ip_hidden_states)
        v_ip = self.to_v_ip(ip_hidden_states)
        ip_out = scaled_dot_product_attention(q, k_ip, v_ip)
        
        # 加权融合
        return text_out + scale * ip_out

```

---

## 六、LoRA — 低秩高效微调

### 6\.1 低秩分解原理

对于预训练权重 $W \in \mathbb{R}^{d \times k}$，LoRA 不直接更新 $W$，而是学习一个低秩增量[\[22\]](https://deepwiki.com/kohya-ss/sd-scripts/8-lora-networks)：$W' = W + \Delta W = W + B \cdot A$其中 $B \in \mathbb{R}^{d \times r}$，$A \in \mathbb{R}^{r \times k}$，秩 $r \ll \min(d,k)$。

**参数效率示例：** SD U\-Net 某注意力层 $W \in \mathbb{R}^{768 \times 768}$，原始参数量 = 589,824。使用 $r=4$ 的 LoRA：$(768 \times 4) + (4 \times 768) = 6144$，仅为原始的 **1%**。

### 6\.2 在 SD 中的应用位置

LoRA 通常应用于 U\-Net 的注意力投影层：`to_q`, `to_k`, `to_v`, `to_out`，有时也包括文本编码器的线性层[\[23\]](https://github.com/bmaltais/kohya_ss/blob/master/docs/LoRA/options.md)。

### 6\.3 关键代码解读

```Python
class LoRALayer(nn.Module):
    """低秩适配层"""
    def __init__(self, original_layer, rank=4, alpha=1.0):
        super().__init__()
        in_dim = original_layer.in_features
        out_dim = original_layer.out_features
        self.rank = rank
        self.scale = alpha / rank  # 缩放因子控制 LoRA 影响强度
        
        # A 矩阵：降维 (Kaiming 初始化)
        self.lora_down = nn.Linear(in_dim, rank, bias=False)
        nn.init.kaiming_uniform_(self.lora_down.weight, a=math.sqrt(5))
        
        # B 矩阵：升维 (零初始化 → 确保训练初始 ΔW=0)
        self.lora_up = nn.Linear(rank, out_dim, bias=False)
        nn.init.zeros_(self.lora_up.weight)

    def forward(self, x):
        # ΔW·x = B(Ax) · scale
        return self.lora_up(self.lora_down(x)) * self.scale

```

**B 矩阵零初始化的意义：** 训练起始时 $\Delta W = B \cdot A = 0$，模型行为与原始预训练完全一致。这保证了训练过程的平稳启动——LoRA 和 ControlNet 的零卷积采用了相同的设计哲学。

#### 权重合并（推理加速）

```Python
def merge_lora_weights(base_layer, lora_layer):
    """推理时将 LoRA 权重合并到基础模型，无额外计算开销"""
    merged_weight = base_layer.weight.data + (
        lora_layer.lora_up.weight @ lora_layer.lora_down.weight
    ) * lora_layer.scale
    base_layer.weight.data = merged_weight

```

合并后推理速度与原模型完全相同，这是 LoRA 的独特优势[\[24\]](https://theneuralbase.com/stable-diffusion/learn/intermediate/merging-loras/)。

---

## 七、Stable Diffusion 3 — MMDiT 架构革命

### 7\.1 从 U\-Net 到 Transformer 的范式转移

SD3 彻底摒弃了 U\-Net 架构，引入 **多模态扩散 Transformer \(MMDiT\)**。其核心论文为《Scaling Rectified Flow Transformers for High\-Resolution Image Synthesis》\(Esser et al\., 2024\)[\[25\]](https://arxiv.org/html/2403.03206v1)。

**MMDiT 的根本性创新**：文本和图像不再是"主从关系"（Cross\-Attention 中文本单向注入图像），而是**对等的双流**，通过联合注意力实现双向信息交换[\[26\]](https://iclr-blogposts.github.io/2026/blog/2026/diffusion-architecture-evolution/)。

### 7\.2 整流流 \(Rectified Flow\) 理论

SD3 放弃了离散噪声调度，采用连续时间整流流公式[\[27\]](https://arxiv.org/abs/2403.03206)：

**线性插值路径：**$x_t = t \cdot x_1 + (1-t) \cdot x_0$

**速度预测目标：**$\mathcal{L} = \mathbb{E}_{t, x_0, x_1}\left[\|v_\theta(x_t, t) - (x_1 - x_0)\|^2\right]$

整流流学习从纯噪声 $x_0$ 到数据 $x_1$ 的直线路径，理论上允许单步生成（实践中 20\-50 步即可获得高质量结果）[\[28\]](https://stabilityai-public-packages.s3.us-west-2.amazonaws.com/Stable+Diffusion+3+Paper.pdf?utm_source=ainews&utm_medium=email&utm_campaign=ainews-not-much-happened-today)。

### 7\.3 三重文本编码器

|**编码器**|**作用**|**注入方式**|
|---|---|---|
|CLIP ViT\-L/14|通用语义编码|隐藏层 → 联合上下文|
|OpenCLIP ViT\-bigG/14|视觉增强编码 \+ 全局调制|隐藏层 → 联合上下文；池化输出 → AdaLN|
|T5\-XXL \(4\.7B参数\)|复杂逻辑与长文本理解|隐藏层 → 联合上下文|

三者的隐藏层经线性投影对齐维度后拼接为统一的 context 序列[\[29\]](https://www.reddit.com/r/comfyui/comments/1den3f3/can_anyone_explain_sd3_clip_l_clip_g_and_t5xxl/)。

### 7\.4 关键代码解读

#### MMDiT Block（联合注意力）

```Python
class MMDiTBlock(nn.Module):
    """双流联合注意力 Transformer 块"""
    def __init__(self, dim, context_dim, num_heads):
        super().__init__()
        # 两个模态各自拥有独立的规范化和投影权重
        self.norm_img = AdaLayerNorm(dim)
        self.norm_txt = AdaLayerNorm(context_dim)
        
        # 独立的 Q/K/V 投影
        self.img_qkv = nn.Linear(dim, dim * 3)
        self.txt_qkv = nn.Linear(context_dim, dim * 3)
        
        # 独立的前馈网络
        self.ff_img = FeedForward(dim)
        self.ff_txt = FeedForward(context_dim)

    def forward(self, img_tokens, txt_tokens, timestep_emb):
        # 自适应层归一化（时间步调制）
        img_mod = self.norm_img(img_tokens, timestep_emb)
        txt_mod = self.norm_txt(txt_tokens, timestep_emb)
        
        # 分别计算 Q/K/V
        q_img, k_img, v_img = self.img_qkv(img_mod).chunk(3, dim=-1)
        q_txt, k_txt, v_txt = self.txt_qkv(txt_mod).chunk(3, dim=-1)
        
        # 核心：联合注意力 — 拼接后统一计算
        q = torch.cat([q_img, q_txt], dim=1)  # 序列维度拼接
        k = torch.cat([k_img, k_txt], dim=1)
        v = torch.cat([v_img, v_txt], dim=1)
        
        attn_out = scaled_dot_product_attention(q, k, v)
        
        # 拆分回各自模态
        img_attn, txt_attn = attn_out.split(
            [img_tokens.shape[1], txt_tokens.shape[1]], dim=1
        )
        
        # 残差连接 + 独立 FFN
        img_tokens = img_tokens + img_attn
        txt_tokens = txt_tokens + txt_attn
        img_tokens = img_tokens + self.ff_img(img_tokens)
        txt_tokens = txt_tokens + self.ff_txt(txt_tokens)
        
        return img_tokens, txt_tokens

```

**关键设计要素：**

- 图像和文本各有独立权重（适应不同模态分布）

- 联合注意力中，图像 Token 可以 attend 到文本 Token（获得语义指导），文本 Token 也可以 attend 到图像 Token（获得视觉反馈）

- 这种双向信息流是 SD3 文本遵循能力大幅提升的根本原因[\[30\]](https://blog.csdn.net/weixin_41446370/article/details/139651650)

#### Patch Embedding

```Python
class PatchEmbed(nn.Module):
    """将 VAE 潜变量转化为 Transformer Token 序列"""
    def __init__(self, in_channels=16, embed_dim=1536, patch_size=2):
        super().__init__()
        # 2x2 卷积将每个 patch 映射为一个 token
        self.proj = nn.Conv2d(in_channels, embed_dim, 
                             kernel_size=patch_size, stride=patch_size)

    def forward(self, latents):
        # latents: [B, 16, H, W] (SD3 使用 16 通道 VAE)
        x = self.proj(latents)        # [B, 1536, H/2, W/2]
        x = x.flatten(2).transpose(1, 2)  # [B, (H/2)*(W/2), 1536]
        return x  # Token 序列，可输入 Transformer

```

SD3 使用 16 通道 VAE（相比 SD1\.x 的 4 通道），压缩比从 $8\\times$ 提升到 $16\\times$，输入到 Transformer 的 Token 序列长度大幅减少[\[31\]](https://www.cnblogs.com/huggingface/p/18253377)。

---

## 八、FLUX — 12B 参数的当前 SOTA

### 8\.1 架构设计哲学

FLUX 由 Black Forest Labs 开发（核心成员为 Stable Diffusion 原始团队），参数量达到 **120 亿 \(12B\)**[\[32\]](https://deepwiki.com/black-forest-labs/flux/4.1-flux-model-architecture)。

其独特的混合架构设计：

- **前 19 层：双流块 \(Double\-Stream\)** — 类似 SD3 的 MMDiT，保持模态独立性

- **后 38 层：单流块 \(Single\-Stream\)** — 将图像和文本合并为统一序列处理[\[33\]](https://deepwiki.com/leejet/stable-diffusion.cpp/6.1.4-flux-models)

**双流 → 单流的设计逻辑：** 模型浅层需要分别理解两种模态的特征（独立权重更高效），深层则需要两种模态的深度融合（统一处理更彻底）。这一设计在保持参数效率的同时最大化了跨模态理解能力[\[34\]](https://www.mindstudio.ai/blog/what-is-flux-1-1-pro/)。

### 8\.2 旋转位置编码 \(RoPE\)

FLUX 引入 2D 旋转位置编码替代固定位置嵌入[\[35\]](https://blog.csdn.net/Misdirection_XG/article/details/148406413)：$\begin{pmatrix} q'_d \\ q'_{d+1} \end{pmatrix} = \begin{pmatrix} \cos\theta & -\sin\theta \\ \sin\theta & \cos\theta \end{pmatrix}\begin{pmatrix} q_d \\ q_{d+1}\end{pmatrix}$

RoPE 的核心优势：**天然支持任意分辨率和长宽比**，无需像 SD3 那样在固定网格上重新训练。

### 8\.3 模型变体

- **FLUX\.1\-pro**: 商用全功能版本

- **FLUX\.1\-dev**: 开源研究版本，保留 Classifier\-Free Guidance

- **FLUX\.1\-schnell**: 经引导蒸馏 \(Guidance Distillation\)，1\-4 步即可生成高质量图像[\[36\]](https://lumenfall.ai/models/black-forest-labs/flux.1-dev/about)

### 8\.4 关键代码解读

#### 单流块实现

```Python
class FLUXSingleStreamBlock(nn.Module):
    """FLUX 后38层：图像和文本合并为统一序列"""
    def __init__(self, dim, num_heads):
        super().__init__()
        self.norm = AdaLayerNorm(dim)
        self.linear1 = nn.Linear(dim, dim * 3)  # 统一的 QKV 投影
        self.attention = FlashAttention(num_heads)
        self.linear2 = nn.Linear(dim, dim)

    def forward(self, x, vec, pe):
        # x: [B, N_txt + N_img, D] — 拼接后的统一序列
        # vec: 时间步/引导调制向量
        # pe: (cos, sin) 旋转位置参数
        
        # 调制 + 统一 QKV 投影
        modulated = self.norm(x, vec)
        qkv = self.linear1(modulated)
        q, k, v = qkv.chunk(3, dim=-1)
        
        # 应用 RoPE
        q = apply_rope(q, pe)
        k = apply_rope(k, pe)
        
        # 标准自注意力（文本和图像在同一序列中自由交互）
        attn_out = self.attention(q, k, v)
        x = x + self.linear2(attn_out)
        return x

```

#### RoPE 应用

```Python
def apply_rope(x, freqs):
    """对查询/键向量施加旋转位置编码"""
    cos, sin = freqs  # 预计算的频率参数
    # 将特征分为偶数/奇数维度对
    x_even = x[..., 0::2]
    x_odd = x[..., 1::2]
    # 旋转操作
    rotated_even = x_even * cos - x_odd * sin
    rotated_odd = x_even * sin + x_odd * cos
    return torch.stack([rotated_even, rotated_odd], dim=-1).flatten(-2)

```

RoPE 将位置信息编码为旋转角度，使注意力分数自然地反映 Token 间的相对空间距离[\[35\]](https://blog.csdn.net/Misdirection_XG/article/details/148406413)。

---

## 九、Stable Video Diffusion \(SVD\) — 从图像到视频

### 9\.1 时序维度扩展策略

SVD 在 SD 的 U\-Net 中每个空间层（Spatial Layer）之后插入**时序层 \(Temporal Layer\)**，使模型感知帧间关系[\[37\]](https://www.themoonlight.io/en/review/enhanced-creativity-and-ideation-through-stable-video-synthesis)。

**两种时序算子：**

- **时序卷积 \(Temporal Conv\)**：沿时间轴的 1D 卷积，学习帧间局部运动

- **时序注意力 \(Temporal Attention\)**：每帧像素关注其他帧的对应位置，确保全局时序一致性

### 9\.2 三阶段训练范式

**SVD 的训练秘诀：**

1. **图像预训练** — 在高质量图像上建立强大的空间生成能力

2. **视频预训练** — 在海量低质量视频上学习运动规律

3. **高质量视频微调** — 在精选高清视频上细化质量

### 9\.3 微条件控制

```Python
# SVD 推理接口
frames = pipeline(
    image=reference_image,    # 条件参考帧
    fps=7,                    # 帧率条件
    motion_bucket_id=127,     # 运动强度 (0=静止, 255=剧烈)
    noise_aug_strength=0.02,  # 参考图像增强噪声
    num_frames=25             # 生成帧数
).frames

```

`motion_bucket_id` 通过离散化的运动强度桶实现对运动幅度的精确控制[\[38\]](https://blog.csdn.net/gitblog_00266/article/details/156015593)。

---

## 技术总结与发展趋势

**架构演进的核心矛盾：**

- 生成质量 vs 推理效率

- 模态融合深度 vs 训练稳定性

- 参数规模 vs 可部署性

**已被验证的设计原则：**

- 潜空间压缩（LDM 范式）是必要的效率保障

- 零初始化连接确保安全的增量学习

- 低秩近似可以在 1% 参数量下完成有效适配

**2024\-2025 趋势预判：**

- Transformer 取代 U\-Net 已成定局 \(SD3, FLUX\)

- 整流流/流匹配替代 DDPM 离散调度

- RoPE 解锁分辨率自由度

- 蒸馏技术使 1\-4 步生成成为现实

- 视频/3D 生成从图像模型自然扩展

# 参考资料

[\[](https://openaccess.thecvf.com/content/CVPR2022/papers/Rombach_High-Resolution_Image_Synthesis_With_Latent_Diffusion_Models_CVPR_2022_paper.pdf)[1\] High\-Resolution Image Synthesis with Latent Diffusion Models Robin Rombach1 ∗](https://openaccess.thecvf.com/content/CVPR2022/papers/Rombach_High-Resolution_Image_Synthesis_With_Latent_Diffusion_Models_CVPR_2022_paper.pdf)

[\[](https://arxiv.org/abs/2112.10752)[2\] \[](https://arxiv.org/abs/2112.10752)[2112\.10752\] High\-Resolution Image Synthesis with Latent Diffusion Models](https://arxiv.org/abs/2112.10752)

[\[](https://aiwiki.ai/wiki/latent_diffusion)[3\] Latent diffusion model](https://aiwiki.ai/wiki/latent_diffusion)

[\[](https://blog.csdn.net/qq_34941290/article/details/157694897)[4\] 深入解析Stable Diffusion基石——潜在扩散模型\(LDMs\)\-CSDN博客](https://blog.csdn.net/qq_34941290/article/details/157694897)

[\[](https://deepwiki.com/CompVis/stable-diffusion/4.4-text-conditioning)[5\] Text Conditioning](https://deepwiki.com/CompVis/stable-diffusion/4.4-text-conditioning)

[\[](https://blog.csdn.net/qq_35899016/article/details/149697368)[6\] 第二十二章:Stable Diffusion的“引擎”:揭秘Latent Diffusion的巧妙架构\_](https://blog.csdn.net/qq_35899016/article/details/149697368)[latent diffusion 架\-CSDN博客](https://blog.csdn.net/qq_35899016/article/details/149697368)

[\[](https://blog.csdn.net/gitblog_01128/article/details/152100877)[7\] stable\-diffusion条件网络架构:交叉注意力机制解析\-CSDN博客](https://blog.csdn.net/gitblog_01128/article/details/152100877)

[\[](https://blog.csdn.net/weixin_46763552/article/details/148669723)[8\] 扩散生成基础原理\(三\)——DDIM、LDM与diffusers代码实践\_](https://blog.csdn.net/weixin_46763552/article/details/148669723)[ddim diffuser源码解析\-CSDN博客](https://blog.csdn.net/weixin_46763552/article/details/148669723)

[\[](https://arxiv.org/html/2407.03297v1)[9\] Improved Noise Schedule for Diffusion Training](https://arxiv.org/html/2407.03297v1)

[\[](https://blog.csdn.net/julac/article/details/143204324)[10\] 『扩散模型』参数化之v\-prediction\-CSDN博客](https://blog.csdn.net/julac/article/details/143204324)

[\[](https://www.reddit.com/r/StableDiffusion/comments/15fj5k9/sdxl_10_a_semitechnical_introductionsummary_for/)[11\] r/StableDiffusion on Reddit: SDXL 1\.0: a semi\-technical introduction/summary for beginners](https://www.reddit.com/r/StableDiffusion/comments/15fj5k9/sdxl_10_a_semitechnical_introductionsummary_for/)

[\[](https://www.xta0.me/2025/01/20/GenAI-Stable-Diffusion-SDXL.html)[12\] The SDXL Model Pipeline](https://www.xta0.me/2025/01/20/GenAI-Stable-Diffusion-SDXL.html)

[\[](https://arxiv.org/pdf/2307.01952)[13\] SDXL: Improving Latent Diffusion Models for High\-Resolution Image Synthesis](https://arxiv.org/pdf/2307.01952)

[\[](https://blog.csdn.net/LittleNyima/article/details/150503020)[14\] 【笔记】扩散模型\(一一\):Stable Diffusion XL 理论与实现\-CSDN博客](https://blog.csdn.net/LittleNyima/article/details/150503020)

[\[](https://arxiv.org/abs/2302.05543)[15\] \[](https://arxiv.org/abs/2302.05543)[2302\.05543\] Adding Conditional Control to Text\-to\-Image Diffusion Models](https://arxiv.org/abs/2302.05543)

[\[](https://disassemble-channel.com/controlnet/)[16\] ControlNetの仕組み — Zero Convolutionと条件付き画像生成](https://disassemble-channel.com/controlnet/)

[\[](https://blog.csdn.net/gitblog_01124/article/details/152098649)[17\] ControlNet零卷积原理:训练初期为何不会破坏基础模型\-CSDN博客](https://blog.csdn.net/gitblog_01124/article/details/152098649)

[\[](https://blog.csdn.net/weixin_54338498/article/details/134489927)[18\] ControlNet原理\+代码解析\-CSDN博客](https://blog.csdn.net/weixin_54338498/article/details/134489927)

[\[](https://huggingface.co/papers?q=IP-Adapter)[19\] Daily Papers \- Hugging Face](https://huggingface.co/papers?q=IP-Adapter)

[\[](https://kimjy99.github.io/%EB%85%BC%EB%AC%B8%EB%A6%AC%EB%B7%B0/ip-adapter/)[20\] \[](https://kimjy99.github.io/%EB%85%BC%EB%AC%B8%EB%A6%AC%EB%B7%B0/ip-adapter/)[논문리뷰\] IP\-Adapter: Text Compatible Image Prompt Adapter for Text\-to\-Image Diffusion Models](https://kimjy99.github.io/%EB%85%BC%EB%AC%B8%EB%A6%AC%EB%B7%B0/ip-adapter/)

[\[](https://diffengine.readthedocs.io/en/stable/_modules/diffengine/models/archs/ip_adapter.html)[21\] Source code for diffengine\.models\.archs\.ip\_](https://diffengine.readthedocs.io/en/stable/_modules/diffengine/models/archs/ip_adapter.html)[adapter](https://diffengine.readthedocs.io/en/stable/_modules/diffengine/models/archs/ip_adapter.html)

[\[](https://deepwiki.com/kohya-ss/sd-scripts/8-lora-networks)[22\] LoRA Networks](https://deepwiki.com/kohya-ss/sd-scripts/8-lora-networks)

[\[](https://github.com/bmaltais/kohya_ss/blob/master/docs/LoRA/options.md)[23\] kohya\_](https://github.com/bmaltais/kohya_ss/blob/master/docs/LoRA/options.md)[ss/docs/LoRA/options\.md at master · bmaltais/kohya\_](https://github.com/bmaltais/kohya_ss/blob/master/docs/LoRA/options.md)[ss](https://github.com/bmaltais/kohya_ss/blob/master/docs/LoRA/options.md)

[\[](https://theneuralbase.com/stable-diffusion/learn/intermediate/merging-loras/)[24\] Merging LoRAs](https://theneuralbase.com/stable-diffusion/learn/intermediate/merging-loras/)

[\[](https://arxiv.org/html/2403.03206v1)[25\] Scaling Rectified Flow Transformers for High\-Resolution Image Synthesis](https://arxiv.org/html/2403.03206v1)

[\[](https://iclr-blogposts.github.io/2026/blog/2026/diffusion-architecture-evolution/)[26\] From U\-Nets to DiTs: The Architectural Evolution of Text\-to\-Image Diffusion Models \(2021–2025\) \| ICLR Blogposts 2026](https://iclr-blogposts.github.io/2026/blog/2026/diffusion-architecture-evolution/)

[\[](https://arxiv.org/abs/2403.03206)[27\] \[](https://arxiv.org/abs/2403.03206)[2403\.03206\] Scaling Rectified Flow Transformers for High\-Resolution Image Synthesis](https://arxiv.org/abs/2403.03206)

[\[](https://stabilityai-public-packages.s3.us-west-2.amazonaws.com/Stable+Diffusion+3+Paper.pdf?utm_source=ainews&utm_medium=email&utm_campaign=ainews-not-much-happened-today)[28\] Scaling Rectified Flow Transformers for High\-Resolution Image Synthesis](https://stabilityai-public-packages.s3.us-west-2.amazonaws.com/Stable+Diffusion+3+Paper.pdf?utm_source=ainews&utm_medium=email&utm_campaign=ainews-not-much-happened-today)

[\[](https://www.reddit.com/r/comfyui/comments/1den3f3/can_anyone_explain_sd3_clip_l_clip_g_and_t5xxl/)[29\] r/comfyui on Reddit: Can anyone explain SD3 clip\_](https://www.reddit.com/r/comfyui/comments/1den3f3/can_anyone_explain_sd3_clip_l_clip_g_and_t5xxl/)[l clip\_](https://www.reddit.com/r/comfyui/comments/1den3f3/can_anyone_explain_sd3_clip_l_clip_g_and_t5xxl/)[g and t5xxl text encoding?](https://www.reddit.com/r/comfyui/comments/1den3f3/can_anyone_explain_sd3_clip_l_clip_g_and_t5xxl/)

[\[](https://blog.csdn.net/weixin_41446370/article/details/139651650)[30\] 【全网瞩目】最强文生图模型，Stable Diffusion 3技术报告解禁\_](https://blog.csdn.net/weixin_41446370/article/details/139651650)[mmdit\-CSDN博客](https://blog.csdn.net/weixin_41446370/article/details/139651650)

[\[](https://www.cnblogs.com/huggingface/p/18253377)[31\] 欢迎 Stable Diffusion 3 加入 🧨 Diffusers \- HuggingFace \- 博客园](https://www.cnblogs.com/huggingface/p/18253377)

[\[](https://deepwiki.com/black-forest-labs/flux/4.1-flux-model-architecture)[32\] FLUX Model Architecture](https://deepwiki.com/black-forest-labs/flux/4.1-flux-model-architecture)

[\[](https://deepwiki.com/leejet/stable-diffusion.cpp/6.1.4-flux-models)[33\] Flux Models](https://deepwiki.com/leejet/stable-diffusion.cpp/6.1.4-flux-models)

[\[](https://www.mindstudio.ai/blog/what-is-flux-1-1-pro/)[34\] What Is FLUX 1\.1 Pro? The Flagship Image Model from Black Forest Labs](https://www.mindstudio.ai/blog/what-is-flux-1-1-pro/)

[\[](https://blog.csdn.net/Misdirection_XG/article/details/148406413)[35\] Stable Diffusion教程 「精神续作」FLUX\.1 源码深度前瞻解读\_](https://blog.csdn.net/Misdirection_XG/article/details/148406413)[combinedtimestepguidancetextprojembeddings\-CSDN博客](https://blog.csdn.net/Misdirection_XG/article/details/148406413)

[\[](https://lumenfall.ai/models/black-forest-labs/flux.1-dev/about)[36\] FLUX\.1 \[](https://lumenfall.ai/models/black-forest-labs/flux.1-dev/about)[dev\]](https://lumenfall.ai/models/black-forest-labs/flux.1-dev/about)

[\[](https://www.themoonlight.io/en/review/enhanced-creativity-and-ideation-through-stable-video-synthesis)[37\] \[](https://www.themoonlight.io/en/review/enhanced-creativity-and-ideation-through-stable-video-synthesis)[Literature Review\] Enhanced Creativity and Ideation through Stable Video Synthesis](https://www.themoonlight.io/en/review/enhanced-creativity-and-ideation-through-stable-video-synthesis)

[\[](https://blog.csdn.net/gitblog_00266/article/details/156015593)[38\] 从新手到专家:Stability AI视频生成参数调优全攻略\-CSDN博客](https://blog.csdn.net/gitblog_00266/article/details/156015593)

