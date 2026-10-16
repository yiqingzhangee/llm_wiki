# Stable Diffusion 视觉大模型微调技术深度调研

## 研究概述

Stable Diffusion（SD）架构已成为当前开源视觉生成模型的基石。围绕该架构的微调技术从最初的全参数训练，演进为以 **参数高效适配** 为核心、**结构化条件控制** 为补充的成熟技术矩阵。本报告系统梳理了 SD 微调的代表性工作原理、主流框架工具及实战注意事项，旨在为研发者提供从理论到落地的完整参考。

**核心结论：** 当前 SD 微调生态以 LoRA/DoRA 为效率核心、ControlNet/IP\-Adapter 为控制支柱、DreamBooth 为主体学习基准，已具备在消费级硬件（12\-24GB VRAM）上完成工业级模型定制的能力。

## 全景总结表

|**方法**|**核心原理**|**修改参数/层**|**数据需求**|**最低显存**|**关键论文**|**典型用例**|
|---|---|---|---|---|---|---|
|**LoRA**|低秩分解：$\Delta W = BA$，仅训练两个小矩阵|U\-Net / Text Encoder 的 Attention 层 \(Q, K, V, O\)|15\-100 张|8GB \(SD1\.5\) / 12GB \(SDXL\)|Hu et al\., 2021|风格迁移、角色一致性、画风学习|
|**DreamBooth**|稀有词绑定 \+ 先验保留损失防止语言漂移|全 U\-Net（或结合 LoRA）|5\-15 张主体 \+ 200\-1000 正则化图|16\-24GB|Ruiz et al\., 2023|个人肖像、宠物、产品定制|
|**Textual Inversion**|在嵌入空间搜索代表新概念的向量，不修改模型权重|仅 Text Encoder 词嵌入（新增向量）|3\-10 张|8GB|Gal et al\., 2022|轻量概念注入、纹理学习|
|**ControlNet**|可训练 Encoder 副本 \+ 零卷积连接，提供空间控制|独立训练的 Encoder 副本支路|数千\-数万张（条件\-图像对）|12\-24GB|Zhang et al\., 2023|姿态控制、深度引导、边缘生成|
|**IP\-Adapter**|解耦交叉注意力：图像/文本各自独立 K\-V 投影|新增图像交叉注意力层的投影矩阵|大规模图文对|12\-16GB|Ye et al\., 2023|图像风格参考、内容迁移|
|**DoRA**|权重分解为幅度\+方向，分别优化逼近全微调效果|Attention 层（方向用 LoRA，幅度独立优化）|同 LoRA|同 LoRA|Liu et al\., 2024|高保真风格学习、复杂概念|
|**OFT**|学习正交变换矩阵，保持超球面能量守恒|Attention 层（正交约束）|同 LoRA|同 LoRA|Qiu et al\., 2023|需严格保留原模型分布的场景|
|**LyCORIS 系列**|LoCon/LoHa/LoKr：扩展到卷积层、Hadamard/Kronecker 积分解|Attention \+ Conv 层|同 LoRA|8\-16GB|KohakuBlueleaf, 2023|高表达力风格、复杂纹理|
|**全微调**|更新模型所有参数，实现领域级迁移|全 U\-Net（所有层）|数万张以上|24GB\+|—|医学影像、遥感、垂直领域大规模迁移|

### 微调框架选型速查

|**框架**|**核心定位**|**最低显存**|**目标用户**|**GitHub**|
|---|---|---|---|---|
|**Diffusers**|工业级标准库，PEFT 集成，模块化 API|8\-12GB|开发者/研究者|[huggingface/diffusers](https://github.com/huggingface/diffusers)|
|**Kohya\_****ss**|社区标准，分桶/噪声偏移，LyCORIS 原生支持|8GB|进阶创作者|[kohya\-ss/sd\-scripts](https://github.com/kohya-ss/sd-scripts)|
|**SimpleTuner**|多 GPU \+ DeepSpeed，Flux/SD3 优先|10GB \(量化\)|高性能需求|[bghira/SimpleTuner](https://github.com/bghira/SimpleTuner)|
|**OneTrainer**|全能 GUI，Tensorboard 集成，自动备份|12GB|进阶用户|[Nerogar/OneTrainer](https://github.com/Nerogar/OneTrainer)|
|**AI\-Toolkit / FluxGYM**|Flux 训练事实标准，极简操作|12\-16GB|初学者/Flux 用户|[ostris/ai\-toolkit](https://github.com/ostris/ai-toolkit)|

### 微调注意事项速查

|**关注维度**|**核心要点**|**推荐配置/做法**|
|---|---|---|
|**数据集质量**|质量远重于数量；需去噪、去重、构图多样|LoRA 15\-40 张精选图；使用 Bucketing 保留多长宽比|
|**学习率**|过高导致过拟合/崩溃，过低收敛缓慢|SD1\.5: 1e\-4; SDXL: 1e\-5; Text Encoder: 5e\-6|
|**Rank/Alpha**|Rank 控制容量，Alpha 控制缩放|简单概念 8\-16；复杂风格 64\-128；Alpha = Rank/2|
|**过拟合预防**|训练步数过多 / 数据过少时最常出现|定期采样检查；使用 Cosine 调度器 \+ Warmup|
|**显存优化**|bf16 混合精度 \+ 梯度检查点 \+ Flash Attention|大模型加 FP8 量化；梯度累积 2\-4 步|
|**高级技巧**|Min\-SNR \(gamma=5\.0\) 加速收敛；Noise Offset \(0\.1\) 改善动态范围|LoRA 合并用 ZipLoRA；评估用 PickScore/HPS|

---

## 第一部分：代表性微调方法与技术原理

### 1\. LoRA — 低秩自适应微调

LoRA（Low\-Rank Adaptation）是目前应用最广泛的轻量化微调技术，其设计哲学基于一个关键假设：模型在特定任务上的权重更新具有"低本征维度"[\[1\]](https://pics.iamzhaokun.com/doc/report/LoRA-Driven%20Anime%20Style%20Generation.pdf)。

**数学原理：**

对于预训练权重矩阵 $W_0 \in \mathbb{R}^{d \times k}$，LoRA 将更新量 $\Delta W$ 分解为两个低秩矩阵的乘积：

$\Delta W = BA, \quad A \in \mathbb{R}^{r \times k},\ B \in \mathbb{R}^{d \times r},\ r \ll \min(d, k)$

前向传播变为：$h = W_0 x + BAx$

训练时 $W_0$ 被冻结，仅优化 $A$（高斯初始化）和 $B$（零初始化，确保训练初始 $\Delta W = 0$）。

**作用层：** 主要应用于 U\-Net 和 Text Encoder 的 Attention 层（Q、K、V、O 投影矩阵）。

**核心优势**

- 文件体积极小（10MB\-200MB）

- 推理时可合并回原权重，无额外开销

- 支持多个 LoRA 叠加使用

- 训练仅需 8\-12GB VRAM（SD 1\.5）

**局限性**

- 秩（Rank）选择需要经验：过低无法捕捉复杂风格

- 对 Attention 层以外的参数覆盖有限

- 多 LoRA 叠加时可能产生冲突

**关键论文：** **LoRA: Low\-Rank Adaptation of Large Language Models**, Hu et al\., 2021[\[1\]](https://pics.iamzhaokun.com/doc/report/LoRA-Driven%20Anime%20Style%20Generation.pdf)

**典型用例：** 风格迁移、角色一致性生成、特定画风学习

---

### 2\. DreamBooth — 主体驱动生成

DreamBooth 是实现"主体驱动生成"（Subject\-Driven Generation）的标杆方法，通过少量图片让模型精确学习特定对象的视觉特征[\[2\]](https://arxiv.org/pdf/2409.15812?)。

**技术原理：**

1. **稀有标识符绑定（Rare Token Identifier）：** 使用词库中罕见的词（如 `sks`）与目标概念绑定，避免与已有语义冲突。

2. **先验保留损失（Prior Preservation Loss, PPL）：** 在训练新概念的同时，引入同类概念的生成图像作为正则化，防止"语言漂移"——即学会特定概念后忘记该类别的一般性表征。

$\mathcal{L} = \mathbb{E}[\|\epsilon - \epsilon_\theta(x_t, c)\|^2] + \lambda \mathbb{E}[\|\epsilon - \epsilon_\theta(x_{pr}, c_{pr})\|^2]$

其中 $x_{pr}$ 是由冻结预训练模型生成的类别正则化图像。

**修改范围：** 全 U\-Net 参数（或结合 LoRA 仅修改低秩分解部分）。

**适用场景：** 个人肖像一致性生成、特定产品视觉、宠物形象定制。

**数据需求：** 仅需 5\-15 张目标主体图片 \+ 200\-1000 张正则化类别图像[\[3\]](https://www.edureka.co/blog/fine-tuning-stable-diffusion/)。

**关键论文：** **DreamBooth: Fine Tuning Text\-to\-Image Diffusion Models for Subject\-Driven Generation**, Ruiz et al\., 2023[\[2\]](https://arxiv.org/pdf/2409.15812?)

---

### 3\. Textual Inversion — 嵌入空间逆向搜索

Textual Inversion 的独特之处在于**完全不修改模型权重**，而是在 Text Encoder 的嵌入空间中寻找能代表新概念的向量[\[4\]](https://blog.csdn.net/gitblog_02580/article/details/149626495)。

**原理：** 在词嵌入空间中优化一个新的向量 $v^*$，使得当输入特定占位词（如 `<my-concept>`）时，模型的去噪过程能够还原目标概念。本质上是对预训练模型语义空间的逆向搜索。

**优势**

- 文件极其轻量（KB 级别）

- 完全不破坏原模型性能

- 与所有推理管线兼容

**局限**

- 表达能力受限于预训练模型的原始语义覆盖

- 无法学习模型从未"见过"的极端画风

- 对复杂概念的还原精度不如 LoRA

**关键论文：** **An Image is Worth One Word: Personalizing Text\-to\-Image Generation using Textual Inversion**, Gal et al\., 2022[\[4\]](https://blog.csdn.net/gitblog_02580/article/details/149626495)

---

### 4\. ControlNet — 像素级空间结构控制

ControlNet 为扩散模型引入了像素级的空间结构控制能力，是条件控制领域的里程碑工作[\[5\]](https://www.arxiv.org/pdf/2302.05543)。

**架构设计：**

1. **可训练编码器副本：** 复制一份 U\-Net 的 Encoder 支路用于学习控制条件（姿态骨架、深度图、Canny 边缘等）。

2. **零卷积连接（Zero Convolution）：** 使用权重和偏置均初始化为零的 $1 \\times 1$ 卷积层，连接副本输出与原始 U\-Net 主路。

$y = \mathcal{F}(x; \Theta) + \mathcal{Z}(\mathcal{F}(x + c;\ \Theta_{copy});\ \Theta_z)$

初始时 $\Theta_z = 0$，保证训练起始输出与原模型完全一致，避免有害梯度破坏预训练特征。

**核心价值：** 训练完成后，条件控制支路可以独立发布，用户无需替换基础模型即可获得精确的空间控制能力。

**关键论文：** **Adding Conditional Control to Text\-to\-Image Diffusion Models**, Zhang \& Agrawala, 2023[\[5\]](https://www.arxiv.org/pdf/2302.05543)

---

### 5\. IP\-Adapter — 图像提示词适配器

IP\-Adapter 实现了直接使用**图像作为提示词**来引导生成，核心创新在于"解耦交叉注意力"机制[\[6\]](https://kimjy99.github.io/%EB%85%BC%EB%AC%B8%EB%A6%AC%EB%B7%B0/ip-adapter/)。

**解耦交叉注意力（Decoupled Cross\-Attention）：**

传统方法将图像和文本特征简单拼接，易导致文本提示失效。IP\-Adapter 为图像特征独立增加了一套 Key\-Value 投影：

$Z_{new} = \text{Attention}(Q, K_{text}, V_{text}) + \lambda \cdot \text{Attention}(Q, K_{img}, V_{img})$

图像特征由 CLIP Image Encoder 提取，经投影网络映射到交叉注意力空间。$\lambda$ 控制图像提示的影响强度。

**优势：** 文本和图像提示词互不干扰，可灵活调节各自权重。

**关键论文：** **IP\-Adapter: Text Compatible Image Prompt Adapter for Text\-to\-Image Diffusion Models**, Ye et al\., 2023[\[7\]](https://arxiv.org/abs/2308.06721)

---

### 6\. Hypernetworks — 辅助权重生成网络

Hypernetworks 训练一个小型辅助网络，接收中间特征并输出针对 Attention 权重的修正偏移量。在 SD 早期（2022\-2023 初）较为流行，但目前已被 LoRA 系列方法基本取代，原因在于其训练稳定性和可解释性相对较弱[\[8\]](https://openreview.net/pdf/e7f78cd9d99baba90eb3c4cebc126bbd5992e227.pdf)。

---

### 7\. 全参数微调 \(Full Fine\-tuning\)

更新模型的所有参数（通常为整个 U\-Net），适用于大规模领域迁移场景——例如将通用模型转变为专业的医学影像或卫星遥感生成模型[\[9\]](https://civitai.com/articles/5084/fine-tuning-ponydiffusionsdxl-without-lora-the-old-fashioned-way)。

**风险提示：** 全微调极易发生灾难性遗忘，显存需求通常 24GB 起步。除非拥有大规模领域数据集（数万张以上）和充足算力，否则不建议使用。

---

### 8\. SDXL 架构的特殊微调考量

SDXL 相较于 SD 1\.5 有显著架构升级，微调时须关注以下差异[\[10\]](https://aiwiki.ai/wiki/sdxl)：

- **双文本编码器：** 同时使用 CLIP ViT\-L 和 OpenCLIP ViT\-bigG，微调时需处理两组 Token 序列

- **池化嵌入注入：** OpenCLIP 的全局池化输出直接注入 U\-Net 时间嵌入层，提升提示词遵循度[\[11\]](https://blog.csdn.net/qq_34941290/article/details/157696229)

- **多维条件增强：** 包含 `original_size`、`target_size`、`crop_coords` 等元数据，训练数据需记录这些信息以维持生成质量[\[12\]](https://deepwiki.com/sdbds/sd-scripts/4-sdxl-training)

- **学习率需求：** 通常需要比 SD 1\.5 更低的学习率（约 $1 \\times 10^\{\-5\}$）

---

### 9\. 2023\-2025 前沿微调方法

#### DoRA — 权重解耦低秩适应 \(2024\)

将预训练权重 $W$ 分解为**幅度（Magnitude）** $m$ 和**方向（Direction）** $V$，通过独立调整两者来逼近全参数微调的学习能力，同时保持参数高效性[\[13\]](https://arxiv.org/pdf/2402.09353)。

$W' = m \cdot \frac{V + \Delta V}{\|V + \Delta V\|_c}$

其中 $\Delta V$ 由 LoRA 提供。DoRA 的训练稳定性和最终效果均优于标准 LoRA。

#### OFT — 正交微调 \(2023\)

通过学习正交变换矩阵 $R$（$W_{new} = R \cdot W_0$）来微调模型。正交变换保持了神经元间的余弦相似度（超球面能量守恒），能极好地保护预训练模型的原始生成分布[\[14\]](https://arxiv.org/html/2306.07280)。

#### LyCORIS 系列

一个集成了多种高表达力算法的项目框架[\[15\]](https://github.com/KohakuBlueleaf/LyCORIS)：

- **LoCon：** 将 LoRA 扩展到卷积层

- **LoHa：** 基于 Hadamard 积的低秩分解

- **LoKr：** 基于 Kronecker 积的分解，在相同参数量下具有更高的表达上限

---

## 第二部分：微调框架与代码库

### 核心框架对比

|**框架**|**定位与核心特性**|**最低显存**|**目标用户**|**GitHub 仓库**|
|---|---|---|---|---|
|**Hugging Face Diffusers**|工业级训练脚本，深度集成 PEFT，模块化设计|8\-12GB \(SD1\.5\)|开发者/研究者|[diffusers](https://github.com/huggingface/diffusers)|
|**Kohya\_****ss / sd\-scripts**|社区标准工具，多分辨率分桶，原生 LyCORIS 支持|8GB \(SD1\.5\)|进阶创作者|[sd\-scripts](https://github.com/kohya-ss/sd-scripts)|
|**SimpleTuner**|多 GPU \+ DeepSpeed，新架构优先\(Flux/SD3\)|10GB \(量化\)|高性能需求者|[SimpleTuner](https://github.com/bghira/SimpleTuner)|
|**OneTrainer**|全能集成 GUI，自动备份，集成 Tensorboard|12GB|追求易用性的进阶用户|[OneTrainer](https://github.com/Nerogar/OneTrainer)|
|**AI\-Toolkit / FluxGYM**|Flux 训练事实标准，极简操作|12\-16GB|Flux 用户/初学者|[ai\-toolkit](https://github.com/ostris/ai-toolkit)|
|**EveryDream2**|侧重全微调，精细数据集管理|24GB|专业创作者|[EveryDream2](https://github.com/victorchall/EveryDream2trainer)|

---

### 1\. Hugging Face Diffusers

作为开源扩散模型的"标准库"，Diffusers 提供了最规范的微调脚本和 API[\[16\]](https://huggingface.co/docs/diffusers/en/training/lora)。

**核心能力：**

- 支持 LoRA、DreamBooth、Textual Inversion、ControlNet、InstructPix2Pix 及一致性模型（LCM）微调

- 深度集成 `peft` 库，支持 AdaLoRA、IA3 等前沿适配方法[\[17\]](https://huggingface.co/docs/peft/main/en/task_guides/dreambooth_lora)

- 支持 4\-bit/8\-bit 量化训练以降低显存[\[18\]](https://apatero.com/blog/fluxgym-alternatives-lora-training-guide-2025)

**适用场景：** 算法研究、自定义训练流水线开发、与 HuggingFace Hub 生态协同。

---

### 2\. Kohya\_ss / sd\-scripts

社区中使用最广泛的微调工具，引入了多项关键技术创新[\[19\]](https://github.com/kohya-ss/sd-scripts)：

**技术亮点：**

- 多分辨率分桶（Aspect Ratio Bucketing）

- 噪声偏移（Noise Offset）

- 原生支持 LyCORIS 全系列方法（LoCon, LoHa, LoKr, DyLoRA）[\[20\]](https://blog.csdn.net/gitblog_00469/article/details/160499995)

- 极其细致的参数调节：层权重调整、优化器切换、分步训练策略

**生态：** 拥有最庞大的社区教程和预设配置，2024 年也是最早适配 Flux\.1 的工具之一[\[21\]](https://mintlify.wiki/kohya-ss/sd-scripts/training/loha-lokr)。

---

### 3\. SimpleTuner

专为超大规模模型和多 GPU 环境设计的高性能训练器[\[22\]](https://pypi.org/project/simpletuner/2.2.1/)。

- 深度集成 DeepSpeed 分布式训练

- 对 Flux\.1/2、SD3、PixArt\-Sigma 提供深度优化

- 支持 int2/int4 量化，Flux 训练最低可降至 10GB VRAM[\[23\]](https://docs.simpletuner.io/quickstart/FLUX/)

- 训练速度比 Kohya 快约 15\-25%[\[24\]](https://www.apatero.com/blog/simpletuner-flux2-training-complete-tutorial-2025)

---

### 4\. 数据集预处理工具链

|**工具**|**功能**|**适用架构**|
|---|---|---|
|WD14 Tagger|自动生成 Danbooru 风格标签（如 `1girl, long hair`）|SD 1\.5 / 二次元模型|
|BLIP2 / JoyCaption|生成自然语言长描述|SDXL / Flux / SD3|
|Tag Editor \(WebUI 插件\)|可视化批量标签编辑与管理|通用|
|Supermerger|分层合并（MBW）与 XYZ 对比|通用模型合并|

---

## 第三部分：微调注意事项与最佳实践

### 1\. 数据集准备 — 质量决定上限

**核心原则：** 30 张精细挑选、去噪、构图优秀的图片，效果远好于 200 张包含低质或重复内容的图片[\[25\]](https://artificial-intelligence-wiki.com/best-ai-tools/ai-image-and-video-generators/lora-training-dataset-preparation/)。

**规模建议：**

- **LoRA（角色/物体）：** 15\-40 张高质量图像[\[26\]](https://www.multic.com/guides/ja/sdxl-lora-guide/)

- **LoRA（艺术风格）：** 50\-100\+ 张涵盖不同主题的图片[\[27\]](https://www.artificial-intelligence-wiki.com/ai-for-content-creators/ai-image-and-video-generation/lora-training-for-custom-models/)

- **DreamBooth：** 5\-15 张目标主体 \+ 200\-1000 张正则化图像[\[3\]](https://www.edureka.co/blog/fine-tuning-stable-diffusion/)

**分辨率策略：**

- SD 1\.5 → 512×512 或 768×768

- SDXL / Flux / SD3\.5 → 1024×1024[\[28\]](https://docs.seaart.ai/guide-1/3-advanced-guide/3-2-lora-training-advance/image-training)

- 使用 **Bucketing**（多分辨率分桶）保留非正方形图片的构图信息[\[29\]](https://botmonster.com/posts/how-to-fine-tune-stable-diffusion-xl-2-with-lora/)

**标注策略选择：**

- **标签式（Tagging）：** WD14 Tagger，适合 SD 1\.5 和二次元模型。标签顺序越靠前，权重越高[\[30\]](https://digitaldivas.ai/2025/03/22/creating-stunning-ai-influencers-the-complete-stable-diffusion-guide-for-1-5-sdxl-and-flux/)

- **自然语言描述：** BLIP2 / JoyCaption / LLaVA，适合 SDXL 及更新架构。能更好表达复杂动作和空间关系[\[31\]](https://artificial-intelligence-wiki.com/best-ai-tools/ai-image-and-video-generators/stable-diffusion-xl-tutorial/)

- **触发词必备：** 训练特定概念时，在 Prompt 开头添加唯一触发词（如 `artstyle_xyz`）

---

### 2\. 关键超参数配置

#### 学习率 \(Learning Rate\)

- **LoRA \(SD 1\.5\)：** $1 \\times 10^\{\-4\}$ 至 $5 \\times 10^\{\-5\}$[\[32\]](https://how-to-l.com/lora-learning-method/)

- **LoRA \(SDXL\)：** $1 \\times 10^\{\-5\}$ 左右[\[33\]](https://velog.io/@strangekim/Stable-Diffusion-%EB%A7%88%EC%8A%A4%ED%84%B0-%EA%B0%80%EC%9D%B4%EB%93%9C-%EA%B0%9C%EB%85%90%EB%B6%80%ED%84%B0-%EC%8B%A4%EC%A0%84-%ED%8A%9C%EB%8B%9D%EA%B9%8C%EC%A7%80)

- **分层策略：** U\-Net LR = $1 \\times 10^\{\-4\}$, Text Encoder LR = $5 \\times 10^\{\-6\}$ 或更低

#### LoRA Rank 与 Alpha 选择

|**场景复杂度**|**推荐 Rank**|**说明**|
|---|---|---|
|简单概念/物体|8\-16|足以捕捉单一概念的视觉特征|
|复杂面部/多姿态|32|需要更多参数容量表达面部细节|
|复杂艺术风格|64\-128|画风包含大量纹理和色彩模式|

**Alpha 设置：** 社区共识为 Alpha = Rank/2 或 Alpha = Rank。实际影响力遵循 $\alpha/r$ 缩放[\[34\]](https://www.mindstudio.ai/blog/what-is-sdxl-lora-custom-styles)。

#### 调度器与优化器

- **Cosine \+ 5\-10% Warmup：** 最成熟的组合，训练初期平滑加载梯度，后期逐渐衰减[\[35\]](https://huggingface.co/blog/lora)

- **Prodigy / Adafactor：** 2024 年流行的自适应优化器，自动寻找最佳学习率

---

### 3\. 常见失败模式及应对

|**失败模式**|**症状表现**|**解决方案**|
|---|---|---|
|**过拟合**|生成图与训练集几乎相同，无法改变服装/姿势|降低 LR、减少 Epochs、增加正则化图、调低 LoRA 权重[\[36\]](https://blog.csdn.net/gitblog_02208/article/details/151412801)|
|**灾难性遗忘**|训练角色后无法画其他主题|降低 Text Encoder 训练步数、增加正则化图像[\[37\]](https://zeroentropy.dev/concepts/catastrophic-forgetting/)|
|**模式崩塌**|所有生成图构图和光影千篇一律|提高数据多样性、降低 LR、检查 Loss 曲线[\[38\]](https://arxiv.org/html/2406.00505)|
|**色彩溢出**|目标概念的颜色泄漏到不相关区域|在标注中明确区分颜色属性、增加背景多样性|

---

### 4\. 硬件与训练效率优化

#### 显存需求概览

- **SD 1\.5 LoRA：** 最低 8GB VRAM

- **SDXL LoRA：** 推荐 12GB\+（开启 xFormers \+ 梯度检查点）[\[39\]](https://vrlatech.com/stable-diffusion-lora-training-hardware-requirements/)

- **Flux / SD3\.5：** 推荐 24GB VRAM；量化为 FP8 后可在 16GB 运行[\[40\]](https://blog.csdn.net/weixin_33308579/article/details/155685327)

#### 关键优化技术

- **混合精度训练：** 必须开启 `fp16` 或 `bf16`（RTX 30 系列及以上首选 `bf16`）[\[41\]](https://dataoorts.com/vram-requirements-for-fine-tuning-llms-ai-models/)

- **FP8 量化训练：** 针对 SDXL/Flux 等大模型，显著降低显存压力[\[42\]](https://qiita.com/gitpullpull/items/539644f62a319323dd03)

- **Flash Attention 2：** 大幅提高注意力计算效率[\[31\]](https://artificial-intelligence-wiki.com/best-ai-tools/ai-image-and-video-generators/stable-diffusion-xl-tutorial/)

- **梯度检查点（Gradient Checkpointing）：** 以时间换空间，减少中间激活值的显存占用

- **梯度累积：** 当 Batch Size 受限时，设置累积步数 2\-4 模拟大批量训练[\[35\]](https://huggingface.co/blog/lora)

---

### 5\. 高级训练技巧

#### Min\-SNR Weighting — 信噪比加权损失

通过动态调整不同信噪比下的损失权重，解决扩散模型训练中各时间步梯度冲突的问题。社区公认最佳 `snr_gamma = 5.0`，能显著加快收敛并改善对比度[\[43\]](https://softwaremill.com/speed-up-your-diffusion-model-training-with-min-snr/)。

#### Noise Offset — 噪声偏移

标准扩散训练假设噪声均值为 0，导致模型难以生成纯黑或极亮画面。开启 `noise_offset = 0.1` 让模型学会高动态范围[\[44\]](https://docs.dreamlook.ai/advanced-features/offset-noise)。

#### LoRA 合并策略

- **ZipLoRA \(2024\)：** 允许将主体 LoRA 和风格 LoRA 完美合并而互不干扰[\[45\]](https://ar5iv.labs.arxiv.org/html/2311.13600)

- **多概念联合训练：** 在标注中交替使用不同触发词，使单个 LoRA 支持多角色生成[\[46\]](https://github.org/adobe-research/custom-diffusion)

---

### 6\. 评估与迭代策略

**训练期评估：**

- 每 N 步保存 Checkpoint 并生成固定 Prompt 的预览图

- 优秀 LoRA 应在权重 0\.6\-1\.0 区间表现稳定

- 出现过饱和或颗粒感 → 过拟合信号

**量化评估指标：**

- **FID：** 评估图像真实度（数值越低越好）

- **CLIP Score：** 评估生成图与 Prompt 的语义一致性[\[47\]](https://cs231n.stanford.edu/papers/text_file_840421827-CS231N_Final_Report.pdf)

- **PickScore / HPS：** 基于人类偏好数据集的自动评估，比 FID 更符合人类审美[\[48\]](https://ar5iv.labs.arxiv.org/html/2305.01569)

---

### 7\. 许可协议与合规

- **CreativeML Open RAIL\-M（SD 1\.5 / SDXL）：** 允许商用和模型修改，但禁止生成有害内容[\[49\]](https://terms.law/ai-output-rights/stable-diffusion/)

- **Flux 许可：** Pro 版具有商业限制，使用前须确认开发者授权条款[\[50\]](https://corp.aicu.ai/ja/stability-ai-20250709?hsLang=ja)

- **数据合规：** 微调者对训练数据的合法性负责，需确保不侵犯版权

---

## 总结与选型建议

**选型快速指引：**

- **个人创作者入门** → FluxGYM / OneTrainer，12\-16GB 显卡即可

- **专业精调需求** → Kohya\_ss，提供最细粒度的参数控制

- **算法研究与定制** → Diffusers，代码规范、与 HF 生态无缝集成

- **超大模型/多卡** → SimpleTuner，训练速度和显存效率最优

2024\-2025 年的技术趋势表明，SD 微调正向三个方向加速演进：

1. **更低的硬件门槛：** FP8/int4 量化使消费级显卡也能训练大模型

2. **更强的表达能力：** DoRA、LyCORIS 系列在参数效率与学习能力间持续优化

3. **更完善的工具链：** 从数据标注到训练评估的全流程 GUI 化，降低了入门壁垒

对于研发团队而言，建议从 LoRA 起步快速验证概念，再根据实际需求逐步探索 DoRA、ControlNet 等进阶方案，最终形成适合自身业务场景的微调 Pipeline。

## 参考资料

[\[](https://pics.iamzhaokun.com/doc/report/LoRA-Driven%20Anime%20Style%20Generation.pdf)[1\] LoRA\-Driven Anime Style Generation: A Comparative Study of Lightweight Fine\-Tuning Techniques](https://pics.iamzhaokun.com/doc/report/LoRA-Driven%20Anime%20Style%20Generation.pdf)

[\[](https://arxiv.org/pdf/2409.15812?)[2\] Aided design of bridge aesthetics based on Stable Diffusion fine\-tuning](https://arxiv.org/pdf/2409.15812?)

[\[](https://www.edureka.co/blog/fine-tuning-stable-diffusion/)[3\] Fine\-Tuning Stable Diffusion: A Complete Guide \(2025\)](https://www.edureka.co/blog/fine-tuning-stable-diffusion/)

[\[](https://blog.csdn.net/gitblog_02580/article/details/149626495)[4\] 2025最强Stable Diffusion微调指南:从官方推荐到工业级部署的全流程解密\-CSDN博客](https://blog.csdn.net/gitblog_02580/article/details/149626495)

[\[](https://www.arxiv.org/pdf/2302.05543)[5\] Adding Conditional Control to Text\-to\-Image Diffusion Models](https://www.arxiv.org/pdf/2302.05543)

[\[](https://kimjy99.github.io/%EB%85%BC%EB%AC%B8%EB%A6%AC%EB%B7%B0/ip-adapter/)[6\] \[](https://kimjy99.github.io/%EB%85%BC%EB%AC%B8%EB%A6%AC%EB%B7%B0/ip-adapter/)[논문리뷰\] IP\-Adapter: Text Compatible Image Prompt Adapter for Text\-to\-Image Diffusion Models](https://kimjy99.github.io/%EB%85%BC%EB%AC%B8%EB%A6%AC%EB%B7%B0/ip-adapter/)

[\[](https://arxiv.org/abs/2308.06721)[7\] \[](https://arxiv.org/abs/2308.06721)[2308\.06721\] IP\-Adapter: Text Compatible Image Prompt Adapter for Text\-to\-Image Diffusion Models](https://arxiv.org/abs/2308.06721)

[\[](https://openreview.net/pdf/e7f78cd9d99baba90eb3c4cebc126bbd5992e227.pdf)[8\] Hypernetworks for image recontextualization Maciej Zieba](https://openreview.net/pdf/e7f78cd9d99baba90eb3c4cebc126bbd5992e227.pdf)

[\[](https://civitai.com/articles/5084/fine-tuning-ponydiffusionsdxl-without-lora-the-old-fashioned-way)[9\] Fine\-tuning PonyDiffusion/SDXL without LoRA the old\-fashioned way \| Civitai](https://civitai.com/articles/5084/fine-tuning-ponydiffusionsdxl-without-lora-the-old-fashioned-way)

[\[](https://aiwiki.ai/wiki/sdxl)[10\] SDXL \(Stable Diffusion XL\)](https://aiwiki.ai/wiki/sdxl)

[\[](https://blog.csdn.net/qq_34941290/article/details/157696229)[11\] 深入解析 Stable Diffusion XL\(SDXL\):改进潜在扩散模型，高分辨率合成突破\_](https://blog.csdn.net/qq_34941290/article/details/157696229)[双文本编码器策略\-CSDN博客](https://blog.csdn.net/qq_34941290/article/details/157696229)

[\[](https://deepwiki.com/sdbds/sd-scripts/4-sdxl-training)[12\] SDXL Training \| sdbds/sd\-scripts \| DeepWiki](https://deepwiki.com/sdbds/sd-scripts/4-sdxl-training)

[\[](https://arxiv.org/pdf/2402.09353)[13\] DoRA: Weight\-Decomposed Low\-Rank Adaptation](https://arxiv.org/pdf/2402.09353)

[\[](https://arxiv.org/html/2306.07280)[14\] Controlling Text\-to\-Image Diffusion by Orthogonal Finetuning](https://arxiv.org/html/2306.07280)

[\[](https://github.com/KohakuBlueleaf/LyCORIS)[15\] GitHub \- KohakuBlueleaf/LyCORIS: Lora beYond Conventional methods, Other Rank adaptation Implementations for Stable diffusion\. · GitHub](https://github.com/KohakuBlueleaf/LyCORIS)

[\[](https://huggingface.co/docs/diffusers/en/training/lora)[16\] LoRA · Hugging Face](https://huggingface.co/docs/diffusers/en/training/lora)

[\[](https://huggingface.co/docs/peft/main/en/task_guides/dreambooth_lora)[17\] DreamBooth fine\-tuning with LoRA · Hugging Face](https://huggingface.co/docs/peft/main/en/task_guides/dreambooth_lora)

[\[](https://apatero.com/blog/fluxgym-alternatives-lora-training-guide-2025)[18\] FluxGYM Alternatives: Best LoRA Training Tools for Flux Models 2025](https://apatero.com/blog/fluxgym-alternatives-lora-training-guide-2025)

[\[](https://github.com/kohya-ss/sd-scripts)[19\] GitHub \- kohya\-ss/sd\-scripts · GitHub](https://github.com/kohya-ss/sd-scripts)

[\[](https://blog.csdn.net/gitblog_00469/article/details/160499995)[20\] Kohya\_](https://blog.csdn.net/gitblog_00469/article/details/160499995)[SS扩散模型训练框架深度解析:从原理到实战应用\-CSDN博客](https://blog.csdn.net/gitblog_00469/article/details/160499995)

[\[](https://mintlify.wiki/kohya-ss/sd-scripts/training/loha-lokr)[21\] LoHa and LoKr](https://mintlify.wiki/kohya-ss/sd-scripts/training/loha-lokr)

[\[](https://pypi.org/project/simpletuner/2.2.1/)[22\] simpletuner 2\.2\.1](https://pypi.org/project/simpletuner/2.2.1/)

[\[](https://docs.simpletuner.io/quickstart/FLUX/)[23\] Flux\[](https://docs.simpletuner.io/quickstart/FLUX/)[dev\] / Flux\[](https://docs.simpletuner.io/quickstart/FLUX/)[schnell\] Quickstart](https://docs.simpletuner.io/quickstart/FLUX/)

[\[](https://www.apatero.com/blog/simpletuner-flux2-training-complete-tutorial-2025)[24\] SimpleTuner Flux\.2 Training: Complete Tutorial 2025](https://www.apatero.com/blog/simpletuner-flux2-training-complete-tutorial-2025)

[\[](https://artificial-intelligence-wiki.com/best-ai-tools/ai-image-and-video-generators/lora-training-dataset-preparation/)[25\] LoRA Training Dataset Preparation](https://artificial-intelligence-wiki.com/best-ai-tools/ai-image-and-video-generators/lora-training-dataset-preparation/)

[\[](https://www.multic.com/guides/ja/sdxl-lora-guide/)[26\] SDXL LoRAガイド: モデルの微調整](https://www.multic.com/guides/ja/sdxl-lora-guide/)

[\[](https://www.artificial-intelligence-wiki.com/ai-for-content-creators/ai-image-and-video-generation/lora-training-for-custom-models/)[27\] LoRA Training for Custom Models](https://www.artificial-intelligence-wiki.com/ai-for-content-creators/ai-image-and-video-generation/lora-training-for-custom-models/)

[\[](https://docs.seaart.ai/guide-1/3-advanced-guide/3-2-lora-training-advance/image-training)[28\] Image Training \| SeaArt Guide](https://docs.seaart.ai/guide-1/3-advanced-guide/3-2-lora-training-advance/image-training)

[\[](https://botmonster.com/posts/how-to-fine-tune-stable-diffusion-xl-2-with-lora/)[29\] SDXL 2\.0 LoRA: 50\-300 MB Adapters on 12 GB VRAM](https://botmonster.com/posts/how-to-fine-tune-stable-diffusion-xl-2-with-lora/)

[\[](https://digitaldivas.ai/2025/03/22/creating-stunning-ai-influencers-the-complete-stable-diffusion-guide-for-1-5-sdxl-and-flux/)[30\] Creating Stunning AI Influencers: The Complete Stable Diffusion Guide for 1\.5, SDXL, and FLUX \- Digital Divas](https://digitaldivas.ai/2025/03/22/creating-stunning-ai-influencers-the-complete-stable-diffusion-guide-for-1-5-sdxl-and-flux/)

[\[](https://artificial-intelligence-wiki.com/best-ai-tools/ai-image-and-video-generators/stable-diffusion-xl-tutorial/)[31\] Stable Diffusion XL Tutorial](https://artificial-intelligence-wiki.com/best-ai-tools/ai-image-and-video-generators/stable-diffusion-xl-tutorial/)

[\[](https://how-to-l.com/lora-learning-method/)[32\] 2025年最新版LoRA学習方法を徹底解説！Stable Diffusionで自作モデルを成功させる手順](https://how-to-l.com/lora-learning-method/)

[\[](https://velog.io/@strangekim/Stable-Diffusion-%EB%A7%88%EC%8A%A4%ED%84%B0-%EA%B0%80%EC%9D%B4%EB%93%9C-%EA%B0%9C%EB%85%90%EB%B6%80%ED%84%B0-%EC%8B%A4%EC%A0%84-%ED%8A%9C%EB%8B%9D%EA%B9%8C%EC%A7%80)[33\] Stable Diffusion 마스터 가이드: 개념부터 실전 튜닝까지](https://velog.io/@strangekim/Stable-Diffusion-%EB%A7%88%EC%8A%A4%ED%84%B0-%EA%B0%80%EC%9D%B4%EB%93%9C-%EA%B0%9C%EB%85%90%EB%B6%80%ED%84%B0-%EC%8B%A4%EC%A0%84-%ED%8A%9C%EB%8B%9D%EA%B9%8C%EC%A7%80)

[\[](https://www.mindstudio.ai/blog/what-is-sdxl-lora-custom-styles)[34\] What Is SDXL LoRA? Custom Fine\-Tuned Styles for Stable Diffusion](https://www.mindstudio.ai/blog/what-is-sdxl-lora-custom-styles)

[\[](https://huggingface.co/blog/lora)[35\] Using LoRA for Efficient Stable Diffusion Fine\-Tuning](https://huggingface.co/blog/lora)

[\[](https://blog.csdn.net/gitblog_02208/article/details/151412801)[36\] 7天精通Stable Diffusion微调:从零基础到商业级模型定制指南\-CSDN博客](https://blog.csdn.net/gitblog_02208/article/details/151412801)

[\[](https://zeroentropy.dev/concepts/catastrophic-forgetting/)[37\] Catastrophic forgetting: when fine\-tuning erases base skills](https://zeroentropy.dev/concepts/catastrophic-forgetting/)

[\[](https://arxiv.org/html/2406.00505)[38\] Improving Text Generation on Images with Synthetic Captions](https://arxiv.org/html/2406.00505)

[\[](https://vrlatech.com/stable-diffusion-lora-training-hardware-requirements/)[39\] Hardware Requirements for Training Your Own Stable Diffusion LoRA in 2026 \- VRLA Tech](https://vrlatech.com/stable-diffusion-lora-training-hardware-requirements/)

[\[](https://blog.csdn.net/weixin_33308579/article/details/155685327)[40\] Stable Diffusion 3\.5 FP8模型支持LoRA微调扩展功能\-CSDN博客](https://blog.csdn.net/weixin_33308579/article/details/155685327)

[\[](https://dataoorts.com/vram-requirements-for-fine-tuning-llms-ai-models/)[41\] A Guide to vRAM requirements for fine\-tuning LLM \& AI models](https://dataoorts.com/vram-requirements-for-fine-tuning-llms-ai-models/)

[\[](https://qiita.com/gitpullpull/items/539644f62a319323dd03)[42\] SDXLをfp8で学習する](https://qiita.com/gitpullpull/items/539644f62a319323dd03)

[\[](https://softwaremill.com/speed-up-your-diffusion-model-training-with-min-snr/)[43\] Speed up your diffusion model training with Min\-SNR \| SoftwareMill](https://softwaremill.com/speed-up-your-diffusion-model-training-with-min-snr/)

[\[](https://docs.dreamlook.ai/advanced-features/offset-noise)[44\] Untitled](https://docs.dreamlook.ai/advanced-features/offset-noise)

[\[](https://ar5iv.labs.arxiv.org/html/2311.13600)[45\] ZipLoRA: Any Subject in Any Style by Effectively Merging LoRAs](https://ar5iv.labs.arxiv.org/html/2311.13600)

[\[](https://github.org/adobe-research/custom-diffusion)[46\] Custom Diffusion](https://github.org/adobe-research/custom-diffusion)

[\[](https://cs231n.stanford.edu/papers/text_file_840421827-CS231N_Final_Report.pdf)[47\] Aligning Text\-to\-Image Diffusion Models using Human Utility Optimization and Low\-Rank Adaptation](https://cs231n.stanford.edu/papers/text_file_840421827-CS231N_Final_Report.pdf)

[\[](https://ar5iv.labs.arxiv.org/html/2305.01569)[48\] Pick\-a\-Pic: An Open Dataset of User Preferences for Text\-to\-Image Generation](https://ar5iv.labs.arxiv.org/html/2305.01569)

[\[](https://terms.law/ai-output-rights/stable-diffusion/)[49\] Stable Diffusion Commercial License \& Output Rights 2026 \| Terms\.Law](https://terms.law/ai-output-rights/stable-diffusion/)

[\[](https://corp.aicu.ai/ja/stability-ai-20250709?hsLang=ja)[50\] Stability AI が利用規約を改訂！この3年間で何が変わったのか？7分野で詳細解説](https://corp.aicu.ai/ja/stability-ai-20250709?hsLang=ja)

