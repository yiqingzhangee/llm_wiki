# 开源多模态模型 ViT 视觉头与风控落地选型报告

**核心判断**

1. **视觉头决定可见信息上限**：分辨率、切片、位置编码与视觉 token 压缩影响小字和局部异常能否进入语言模型，但整模型与风控链路共同决定最终效果。

2. **高分辨率与动态预算已成主线**：原生可变分辨率、动态切片和自适应压缩分别优化布局保真、兼容性与成本，没有跨任务通吃的视觉路线。

3. **榜单领先不能直接指导采购**：提示模板、推理模式、像素预算和评测框架可显著改变结果，必须在相同业务输入与部署条件下比较。

4. **生产线上量化 KPI 稀缺**：公开证据集中在内容审核与金融文档处理，且多为厂商自报；支付交易反欺诈几乎没有可归因于 ViT 的线上指标。

5. **建议按风控子场景小步验证**：内容语义审核、文档一致性核验可优先试点；身份真实性与交易裁决应由专模、规则、图模型及人工掌握最终权限。

## 一、评估边界：视觉头定上限，整链路定收益

> **本节要点**：ViT 视觉头只能决定输入信息如何被保留和编码，无法单独解释整模型准确率，更不能直接解释线上拦截、误伤或损失变化。
> 
> 

### 1\.1 “ViT 视觉头”的定义与因果边界

本文所称“ViT 视觉头”，包括图像预处理、视觉编码器、位置编码、切片或原生分辨率策略，以及送入语言模型前的连接器与视觉 token 压缩。它首先影响三个中间变量：小字与局部目标是否仍可辨认、二维布局与跨区域关系是否保留、视觉序列以多大计算成本进入语言模型。固定低分辨率缩放会造成形变和模糊，高分辨率 ViT 的计算量与 token 数又会快速增长；LLaVA\-UHD 的受控实验说明，切片与压缩可同时改变 TextVQA、幻觉代理指标和推理开销，但这些结果仍只属于特定模型与公开基准。[\[1\]](https://arxiv.org/abs/2403.11703v1)

**因果边界必须分三层理解。**

- 视觉头层：可对“输入细节是否被编码、视觉 token 数与视觉编码时延”作直接归因。

- 整模型层：最终回答还受连接器训练、语言模型规模、指令数据、推理模式、提示词、工具调用与解码设置影响；只能在受控消融中判断视觉头的边际贡献。

- 风控系统层：线上 Recall、FPR、申诉率、损失与人工复核量，还受 OCR、检测、规则、图模型、阈值、人审和流量分布影响，不能从通用榜单倒推出 ViT 的业务因果贡献。

因此，本报告把“视觉头性能”“整模型效果”“系统业务结果”分开陈述。若只有整模型横评而没有替换视觉头的消融，不把差值归因于视觉编码器；若只有生产案例而没有组件消融，也不把线上收益归因于 ViT。

### 1\.2 性能、效率与风控价值的评估口径

NIST 要求准确性与稳健性指标绑定现实测试集、部署条件和数据分段；对视觉风控，至少应按场景、语言、清晰度、长宽比、压缩或翻拍、字号、设备、渠道、新旧攻击分桶。总体均值无法替代高风险子群和尾部表现。[\[2\]](https://airc.nist.gov/airmf-resources/airmf/3-sec-characteristics/)

|评估层|应看指标|可支持的结论|不可越过的边界|
|---|---|---|---|
|视觉头|有效分辨率、切片数、视觉 token、编码时延、局部目标/OCR 消融|输入保真与编码成本|不能单独解释整模型或线上收益|
|整模型|MMMU、MMVet、OCRBench、DocVQA、业务离线集、TTFT、显存|给定版本与推理口径下的能力和效率|不能直接等同生产 Recall、FPR 或 ROI|
|风控系统|Recall、Precision、FPR、金额加权召回、人审率、P99、单位真阳性成本|整链路在目标阈值下的净收益|无组件消融时不能归因于 ViT|
|证据等级|线上 A/B ＞ 生产验收 ＞ 业务离线集 ＞ 通用 benchmark|按证据强度确定置信度|厂商自报、内部集和研究基准均非独立审计|

*表 1：视觉头、整模型与风控系统必须使用不同的归因口径*

**证据边界：**截至 2026 年 9 月 2 日，公开、可复算且能直接归因于开源 ViT 多模态模型的生产线上量化 KPI 稀缺。现有线上材料常缺样本量、时间窗口、基线、阈值、置信区间或组件消融；故报告对技术机制给出较高置信度，对模型横比给出中等置信度，对线上业务增益仅给出初步或场景限定判断。

---

## 二、视觉头全景：差异集中在编码器、分辨率与 token 预算

> **本节要点**：截至 2026 年 9 月 2 日，主流开放 VLM 的视觉头已从“固定方形视觉塔 \+ 简单 MLP”分化为动态切片、原生可变分辨率、多视觉塔、时空 patch 与自适应压缩等路线。决定工程成本的核心变量是有效像素、视觉编码器计算量、进入语言模型的 token 数，以及这些预算在图片、页面与视频帧之间如何分配。
> 
> 

### 2\.1 组成与术语边界

本章所称视觉头，覆盖预处理与采样、视觉编码器、位置或时间编码、连接器与压缩，以及视觉特征注入语言模型的方式；“视觉编码器”仅指其中承担图像或视频特征提取的网络。视觉头相同也不意味着整模型能力相同，训练数据、语言模型、对齐阶段、提示与解码设置仍会改变结果。

*图 1：视觉头由输入、编码、压缩、位置时间编码与语言注入五层共同构成（来源：*[*\[4\]*](https://arxiv.org/abs/2502.13923)*、*[*\[6\]*](https://arxiv.org/abs/2508.18265)*）*

**“所有”的范围定义如下：**本章尽可能覆盖有公开权重、官方模型卡或技术报告，且截至采集日仍具代表性或活跃维护的模型家族；同一家族的小版本合并介绍，旧代、低活跃、资料缺失和仅开放权重的情况单列标注。它不宣称穷尽全部 GitHub 项目、论文原型、私有模型、未公开视觉配置的检查点或社区衍生微调版。

|层级|需要确认|常见方案|主要成本|易混淆点|
|---|---|---|---|---|
|输入与采样|缩放、补边、切片、帧率、最大像素|固定方形、AnyRes、动态 tile、原生比例|预处理、tile 或帧数量|整图分辨率不等于单 tile 输入|
|视觉编码器|骨干、patch、注意力、冻结范围|CLIP、SigLIP、InternViT、AIMv2、自研 ViT|视觉 FLOPs 与峰值显存|预训练塔不代表训练时始终冻结|
|连接与压缩|输出 token、压缩位置、是否可学习|MLP、pixel shuffle、Perceiver、视觉 tokenizer|语言模型预填充与 KV 缓存|压缩率不能直接等同端到端加速|
|位置与时间|二维位置、tile 顺序、时间戳、帧间合并|2D RoPE、M\-RoPE、3D RoPE、V2PE|长上下文与实现复杂度|支持多帧不等于具备专门时序编码器|
|语言注入|串接、交叉注意力、多层注入|前缀 token、gated cross\-attention、DeepStack|LLM 层间通信与显存流量|整模型增益不能只归因于视觉塔|

*表 2：比较视觉头时应把五层变量分开记录*

### 2\.2 CLIP／SigLIP 固定视觉塔与切片路线

这一路线的共同点是复用成熟的固定尺寸视觉塔，再由外部切片器增加有效像素，由连接器控制送入语言模型的序列长度。优势是权重成熟、实现路径清晰；代价是 tile 数近似线性增加视觉编码成本，跨 tile 布局还需要全局缩略图、位置标签或额外训练补偿。

|家族|编码器与输入|连接器／压缩|视频|工程含义|
|---|---|---|---|---|
|LLaVA 1\.5／NeXT|CLIP ViT\-L/14\-336；1\.5 补方，NeXT 以 336 tile 做 AnyRes|两层 MLP；NeXT 主要靠增加 tile 扩容|图像主线；Video\-LLaVA 另用 LanguageBind|基线简单，但高分辨率成本随 tile 增长 [\[44\]](https://arxiv.org/abs/2310.03744)|
|LLaVA\-OneVision|SigLIP\-SO400M，384×384 tile；Higher AnyRes|两层 MLP；单 tile 729 token，视频主设置每帧 196 token|统一单图、多图、视频训练|多场景统一，但视觉预算仍需按媒体类型限制 [\[45\]](https://arxiv.org/abs/2408.03326)|
|LLaVA\-UHD|CLIP\-L/14\-336；可变 slices 加空间布局|Perceiver 将每 slice 576→64 token|v1 以图像为主|高像素覆盖与固定每片预算兼得；需锁定 v1／v2／v3 版本 [\[1\]](https://arxiv.org/abs/2403.11703v1)|
|Idefics2／3|SigLIP\-SO400M；保比例输入或固定子图切分|Idefics2 Perceiver 64 token／视图；Idefics3 2×2 空间压缩为 169 token／子图|多图强；原生视频披露有限|预算可预测，但密集 OCR 要验证固定压缩瓶颈 [\[47\]](https://huggingface.co/HuggingFaceM4/idefics2-8b)[\[48\]](https://arxiv.org/abs/2408.12637)|
|SmolVLM|效率优化 SigLIP；384px 子图|每子图 81 token|后续版本支持视频，具体检查点核验|适合小模型与端侧，细节任务需提高 crop 或回退专用 OCR [\[49\]](https://huggingface.co/blog/smolvlm)|
|Phi Vision|Phi\-3\.5 用 CLIP\-336 动态多 crop；Phi\-4 Multimodal 用 SigLIP\-448；Reasoning Vision 用 SigLIP2 NaFlex|MLP；Phi\-3\.5 约 144 token／crop，Reasoning Vision 最多 3600 patches|多帧或中融合，依版本而异|家族跨三种输入路线，不能用单一“Phi 视觉头”概括 [\[50\]](https://huggingface.co/microsoft/Phi-3.5-vision-instruct)[\[51\]](https://huggingface.co/microsoft/Phi-4-reasoning-vision-15B)|
|PaliGemma／Gemma 3|SigLIP；PaliGemma 2 为 224／448／896 三档，Gemma 3 固定 896 并可 Pan\-and\-Scan|线性投影或固定 soft\-token 压缩；Gemma 3 每图 256 token|PaliGemma 偏短视频迁移；Gemma 3 非视频优先|静态 shape 易部署，高分辨率 crop 会增加总 token [\[52\]](https://arxiv.org/abs/2412.03555)[\[53\]](https://arxiv.org/abs/2503.19786)|
|MiniCPM\-V 2\.6／4\.0／4\.5|SigLIP／SigLIP2；自适应 448px 切片|每片 64\-query Perceiver；4\.5 用统一 3D\-Resampler|4\.5 联合时空重采样|端侧与长视频预算突出，但高压缩需回归小字与瞬时事件 [\[7\]](https://arxiv.org/abs/2509.18154)|

*表 3：固定视觉塔通过切片与压缩扩展高分辨率，部署成本主要由 tile 数决定*

### 2\.3 InternViT／EVA／自研 ViT 路线

专用或多塔视觉编码器的目标是补足 CLIP／SigLIP 的局限：InternViT扩大视觉容量，AIMv2、EVA\-CLIP 与自研 ViT强化高分辨率或多模态训练，多塔方案则组合语义、纹理与密集定位特征。其代价集中在视觉侧参数、显存、特征对齐和部署依赖。

|家族|视觉编码器|输入与连接方式|工程影响|
|---|---|---|---|
|InternVL2—3\.5|InternViT\-300M 或 6B|448px 动态 tile \+ thumbnail；pixel unshuffle 1024→256 token／tile；两层 MLP|细节能力强，但视觉塔可占显著算力；3\.5 以 ViR 在 256／64 token 两档路由 [\[55\]](https://arxiv.org/abs/2412.05271)[\[6\]](https://arxiv.org/abs/2508.18265)|
|NVLM\-D|冻结 InternViT\-6B\-448|最多 6 tile 加 thumbnail；pixel shuffle \+ 两层 MLP|视觉效果与预算稳定，但 72B 语言骨干和 6B 视觉塔部署昂贵，公开权重非商业 [\[58\]](https://arxiv.org/abs/2409.11402)|
|Ovis2|AIMv2\-Huge\-patch14\-448|固定分区；hidden stride=2；Visual Tokenizer \+ 视觉嵌入表|将视觉嵌入结构化对齐到词表空间；密集图表的跨分区结构仍是限制 [\[56\]](https://arxiv.org/abs/2405.20797)|
|GLM\-4\.5V／4\.6V|AIMv2\-Huge 初始化的 24 层 ViT|可变分辨率；3D patch convolution \+ 2×2 merge \+ MLP|图像、视频共用前端且有时间下采样；需以业务 FPS 验证短事件召回 [\[39\]](https://arxiv.org/abs/2507.01006)|
|DeepSeek\-VL／VL2|VL1 为 SigLIP\-L \+ SAM\-B 双塔；VL2 为高分辨率动态切图视觉前端|双塔细节融合或动态切图后接 MoE 语言骨干|VL1 兼顾语义与局部，双塔增加成本；VL2 已接替 VL1，完整压缩细节以版本配置为准 [\[61\]](https://github.com/deepseek-ai/DeepSeek-VL2)|
|CogVLM／CogVLM2|EVA\-CLIP 系视觉塔|视觉专家参数注入注意力与 FFN；图像最高 1344px，Video 版加时间戳|融合深、迁移成本高于外置 MLP；商业使用和上游许可需登记核验 [\[62\]](https://arxiv.org/abs/2408.16500)|
|Eagle／Eagle2|多视觉编码器；9B 公开版为 SigLIP \+ ConvNeXt|预对齐后拼接互补视觉 token|多塔提高表征覆盖，也增加视觉 FLOPs、显存与预对齐复杂度 [\[64\]](https://huggingface.co/nvidia/Eagle2-9B)|
|Cambrian\-1|SigLIP、CLIP、DINOv2、OpenCLIP ConvNeXt 四塔|Spatial Vision Aggregator 以固定数量 token 聚合多尺度特征|适合视觉中心研究；生产部署要承担四塔依赖，项目截至采集日低活跃 [\[65\]](https://arxiv.org/abs/2406.16860)|
|Bunny／Mini\-Gemini|Bunny 可插拔 EVA\-CLIP／SigLIP；Mini\-Gemini 另加高分辨率编码器|S2\-Wrapper 或低分辨率 query 检索高分辨率特征|能在较少语言 token 下补细节；两项目均低活跃，依赖和许可需锁定 [\[67\]](https://huggingface.co/BAAI/Bunny-v1_1-Llama-3-8B-V)[\[68\]](https://arxiv.org/abs/2403.18814)|

*表 4：专用大视觉塔和多塔提高表征容量，同时把成本前移到视觉编码阶段*

### 2\.4 原生动态分辨率与 NaViT 路线

原生动态分辨率直接编码可变长宽的 patch 网格，减少固定 tile 的拼接边界；NaViT 进一步通过 sequence packing 提高可变尺寸训练利用率。部署侧的新难点是 batch 内 token 方差、位置编码、注意力复杂度与像素上限，因而“原生”并不意味着成本不受控。[\[72\]](https://arxiv.org/abs/2307.06304)

|家族|原生输入与位置|压缩／注入|部署边界|
|---|---|---|---|
|Qwen2\-VL|动态分辨率 ViT；2D RoPE，视频用时高宽 M\-RoPE|2×2 patch merge，经两层 MLP 接入语言模型|视觉 token 随像素变化；3D temporal patch=2 兼顾图像与视频 [\[54\]](https://arxiv.org/abs/2409.12191)|
|Qwen2\.5\-VL|从头训练动态 ViT；窗口注意力配少量全局层|延续 2×2 merge；时间位置对齐绝对时间|窗口化降低高分辨率 ViT 成本，但 P99 仍受最大像素与帧数控制 [\[4\]](https://arxiv.org/abs/2502.13923)|
|Qwen3\-VL|SigLIP2 初始化后继续动态训练；2D RoPE \+ 插值绝对位置|2×2 merge；DeepStack 从中间层注入 LLM 早期层|多层注入补充纹理与语义，但增加层间特征流和实现复杂度 [\[5\]](https://arxiv.org/abs/2511.21631)|
|Ovis2\.5|SigLIP2\-SO400M 初始化的 NaViT；RoPE \+ 保留绝对位置|hidden stride=2；Visual Tokenizer／VET|避免 Ovis2 固定分区的全局结构损失；窗口与位置实现需使用官方配置 [\[8\]](https://arxiv.org/abs/2508.11737)|
|Pixtral 12B|自研 Pixtral\-ViT，patch16，RoPE\-2D，保持原始宽高比|视觉 token 随图像面积变化后直接进入多模态解码器|布局保真好；需用 128K 总上下文限制多图和高分辨率成本 [\[9\]](https://arxiv.org/abs/2410.07073)|
|LLaVA\-OneVision\-1\.5|RICE\-ViT，2D RoPE 与区域感知注意力|相邻 2×2 patch 拼接后经两层 MLP|从固定 SigLIP tile 转为原生分辨率；4B／8B 权重和训练数据已公开 [\[46\]](https://arxiv.org/abs/2509.23661)|
|Phi\-4 Reasoning Vision|SigLIP2\-SO400M NaFlex，可变分辨率|两层 GELU MLP，最多 3600 visual patches|明确上限便于容量规划；上下文、预填充和细节质量需联合测量 [\[51\]](https://huggingface.co/microsoft/Phi-4-reasoning-vision-15B)|
|Gemma 4|可变宽高比；不同变体使用 150M／550M 视觉塔，12B Unified 无独立编码器|70—1120 五档视觉 token 预算；Unified 直接投影 RGB patch|预算旋钮清晰，但各变体视觉结构不同，不能跨型号外推 [\[10\]](https://ai.google.dev/gemma/docs/core/model_card_4)|

*表 5：原生动态路线减少切片边界，但必须显式管理可变视觉 token 与批处理方差*

### 2\.5 视频时空编码与多帧压缩路线

视频能力需要同时核对帧采样、单帧分辨率、时间 patch、时间位置编码、跨帧压缩与最大帧数。仅把多张帧图串入上下文属于 frame\-wise 路线；3D patchify、时空 resampler、时间戳对齐或跟踪训练才提供更明确的时间建模机制。

*图 2：视觉输入从固定单图、动态切片演进到时空 patch 与自适应预算（来源：*[*\[54\]*](https://arxiv.org/abs/2409.12191)*、*[*\[7\]*](https://arxiv.org/abs/2509.18154)*）*

|路线／家族|时间处理|压缩与公开边界|适用与风险|
|---|---|---|---|
|Qwen\-VL／GLM\-V|2 帧 temporal patch；Qwen 动态 FPS 与时间对齐，GLM 加 3D RoPE 和文本时间索引|空间均为 2×2 merge；token 随帧数和像素增长|适合时间定位；高速短事件仍受采样率与 stride 限制 [\[4\]](https://arxiv.org/abs/2502.13923)[\[39\]](https://arxiv.org/abs/2507.01006)|
|MiniCPM\-V 4\.5|相邻帧打包后做统一 3D\-Resampler|论文特定设置最高报告 96× 视频 token 压缩|长视频与端侧有优势；高压缩下的瞬时事件、计数和顺序需复测 [\[7\]](https://arxiv.org/abs/2509.18154)|
|Molmo2|默认 2 fps；常规最多 128 帧，长上下文训练可到 384 帧|视频用 3×3 window 池化，再共享 MLP|公开支持视频问答、指向与跟踪；视频数据权利仍需核验 [\[40\]](https://arxiv.org/abs/2601.10611)|
|Aria／Gemma 4|均以帧序列为主；Gemma 4 官方示例 1 fps、最长 60 秒|Aria 用固定 query 压缩；Gemma 4 用可选每帧预算|预算容易配置，但专门运动建模弱于 3D tubelet 路线 [\[59\]](https://arxiv.org/abs/2410.05993)[\[10\]](https://ai.google.dev/gemma/docs/core/model_card_4)|
|mPLUG\-Owl3／Eagle2|长视觉序列或统一图像、多图、视频输入|Hyper Attention 或多塔特征；精确时空压缩率依检查点|适合研究型多媒体输入；厂商效率数字与非商业权重需分别核验 [\[63\]](https://huggingface.co/mPLUG/mPLUG-Owl3-7B-240728)[\[64\]](https://huggingface.co/nvidia/Eagle2-9B)|
|VILA／NVILA|先提高空间／时间分辨率，再压缩视觉 token|scale\-then\-compress；支持 PyTorch、TensorRT\-LLM、TinyChat|长视频部署工具较全，但公开 NVILA\-8B 权重为非商业研究许可 [\[66\]](https://arxiv.org/abs/2412.04468)|
|LLaMA\-VID|每帧构造 context token 与 content token|每帧 2 token，面向小时级视频|预算极低，细节损失风险高；项目疑似停更且检查点限研究用途 [\[69\]](https://arxiv.org/abs/2311.17043)|
|LongVA|把长文本上下文能力迁移到帧序列|官方称可处理 2000 帧或超过 200K 视觉 token|适合长视频检索研究；显存、吞吐和非商业许可限制生产采用 [\[70\]](https://huggingface.co/lmms-lab/LongVA-7B)|

*表 6：视频路线的首要比较对象是时间覆盖与总 token 预算，而非视觉塔名称*

### 2\.6 连接器、视觉 token 压缩与多层注入

连接器的直接职责是把视觉特征变换到语言嵌入空间；它是否同时压缩 token，决定语言模型预填充与 KV 缓存成本。简单 MLP 保留更多局部证据，规则重排提供稳定压缩，learned resampler 以固定 query 建立信息瓶颈，动态剪枝与路由则把成本变为输入相关。跨模型比较时必须固定视觉塔、像素预算、LLM 与训练数据，否则只能评价整套路线。

|机制|代表家族|token 行为|工程取舍|
|---|---|---|---|
|线性／两层 MLP|LLaVA、PaliGemma、Phi、GLM\-V|通常逐 token 映射；若无前置 merge 则不压缩|实现与训练最简单，细节保留较好；LLM prefill 随 patch 或 tile 增长 [\[44\]](https://arxiv.org/abs/2310.03744)|
|2×2 merge／pixel unshuffle|Qwen、InternVL、NVLM、Idefics3、Aya|空间 token 稳定降至 1／4，信息折入通道|预算确定、硬件友好；精细位置由后续层解码，过度叠加会伤小字|
|Perceiver／query resampler|LLaVA\-UHD、Idefics2、MiniCPM\-V、Aria|每图或每片输出固定 64／128／256 等 query|多图和长上下文成本可预测；密集实体、计数与小字存在固定容量瓶颈|
|视觉 tokenizer／VET|Ovis／Ovis2\.5|预测视觉词表分布，再对可学习嵌入表加权|视觉与文本嵌入结构更一致；实现专有，迁移与量化工具链需验证 [\[56\]](https://arxiv.org/abs/2405.20797)|
|多层特征融合|Molmo、Qwen3\-VL、Dense Connector 类实现|读取 ViT 中间层；可在相同输出长度补充浅层纹理|可能改善细粒度证据，但增加特征缓存、投影与层间通信 [\[57\]](https://arxiv.org/abs/2409.17146)|
|多塔聚合|DeepSeek\-VL、Eagle、Cambrian、Mini\-Gemini|先并联编码，再拼接、查询或空间聚合|语义与局部特征互补；视觉侧 FLOPs、显存和依赖成倍增加|
|动态路由／早期压缩|InternVL3\.5、MiniCPM\-V 4\.6|按 tile 或任务在不同压缩档位间选择|可降低平均成本；P99、路由稳定性和高风险样本误压缩必须单测 [\[6\]](https://arxiv.org/abs/2508.18265)[\[42\]](https://huggingface.co/openbmb/MiniCPM-V-4.6)|

*表 7：连接器的核心取舍是局部证据保真、固定预算与输入相关效率*

### 2\.7 模型家族目录与场景选择

下表按“代表性公开版本”给出可查阅目录。开放状态只描述公开代码、权重与模型卡，不替代对基座、数据、商用、分发、地域和行业条款的法律审查；同一家族不同检查点可能使用不同许可。

|模型家族|代表视觉路线|视频|开放／维护状态|优先场景与工程提示|
|---|---|---|---|---|
|Qwen\-VL|动态 ViT、2×2 merge、M\-RoPE；Qwen3 加 DeepStack|✅ 原生|代码与多尺寸权重公开；8B 检查点 Apache\-2\.0 [\[41\]](https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct)|文档、视频、通用理解；严格限制最大像素与帧数|
|InternVL|InternViT \+ 动态 448 tile \+ pixel unshuffle；3\.5 加视觉路由|✅ 逐帧|活跃，多尺寸权重；3\.5 8B 为 Apache\-2\.0|高分辨率文档与细粒度；视觉塔和 tile 上限影响显存|
|MiniCPM\-V|SigLIP2 \+ 自适应切片；Perceiver、3D\-Resampler 或 ViT 内早压缩|✅ 原生|权重、量化和端侧工具公开；许可逐检查点核验 [\[42\]](https://huggingface.co/openbmb/MiniCPM-V-4.6)|移动端、OCR、长视频；高压缩回归小字与瞬时事件|
|Ovis|AIMv2 或 NaViT \+ Visual Tokenizer／VET|🟡 有限披露|Ovis2\.5 权重与代码公开，仓库 Apache\-2\.0|原生比例与结构化视觉嵌入；视频采样细节待核验|
|GLM\-V|AIMv2\-Huge \+ 3D patch \+ 2×2 merge \+ 3D\-RoPE|✅ 原生|4\.6V／Flash 权重与代码公开，模型卡 MIT|视频时刻、文档与工具调用；按目标 FPS 测短事件|
|LLaVA|CLIP／SigLIP 固定塔 \+ AnyRes／UHD；新代 RICE\-ViT 原生分辨率|✅ 分支支持|生态大；OneVision 及 1\.5 权重公开，许可依检查点|研究基线与定制；分支众多，先锁版本、数据和预处理|
|Idefics／SmolVLM|SigLIP \+ Perceiver 或 pixel shuffle，固定小 token 预算|🟡 依版本|模型、数据配方和工具公开；多为 Apache\-2\.0，父模型另核|多图、小模型、端侧；密集 OCR 评估压缩瓶颈|
|Phi Vision|CLIP 多 crop、SigLIP\-448 或 SigLIP2 NaFlex \+ MLP|✅ 依版本|官方权重与 remote code 公开，多数模型卡 MIT|轻中型部署与推理；不同代视觉路线差异很大|
|Pixtral|自研 ViT、patch16、RoPE\-2D、原生宽高比|❌ 未声明|12B Apache\-2\.0；Large 为研究许可且已 deprecated|静态高分辨率、多图；Large 不宜新集成|
|PaliGemma／Gemma|固定 SigLIP 档位、Pan\-and\-Scan；Gemma 4 可变预算／encoder\-free|🟡 帧序列|PaliGemma 2／Gemma 3 用 Gemma Terms；Gemma 4 Apache\-2\.0|任务微调、可控预算与轻量部署；按型号核视觉结构|
|Molmo|CLIP／SigLIP2，多 crop，多层特征 \+ 注意力池化|✅ Molmo2|权重 Apache\-2\.0；Molmo2 部分训练数据含非商业限制 [\[57\]](https://arxiv.org/abs/2409.17146)|指向、跟踪、视觉解释；商业使用先审数据权利|
|NVLM|冻结 InternViT\-6B \+ 动态 tile \+ pixel shuffle|❌ 未声明|公开重点为 NVLM\-D\-72B；CC\-BY\-NC\-4\.0|大模型研究对照；高成本且不适合直接商业部署|
|Aria|SigLIP 初始化 ViT \+ query cross\-attention 固定压缩|✅ 帧输入|权重 Apache\-2\.0|多图、视频与长上下文；厂商速度口径需复测|
|Aya Vision|SigLIP2 \+ tile／thumbnail \+ pixel shuffle \+ SwiGLU MLP|❌ 未声明|开放权重研究发布，CC\-BY\-NC\-4\.0 [\[60\]](https://arxiv.org/abs/2505.08751)|多语言图像理解；商用硬门槛不通过|
|DeepSeek\-VL|SigLIP \+ SAM 双塔；VL2 转动态高分辨率|🟡 待核验|VL2 已接替 VL1；代码 MIT，模型许可明确支持商业使用|细节与语义兼顾；双塔成本及 VL2 压缩细节需实测|
|CogVLM|EVA\-CLIP \+ 视觉专家参数；Video 版加时间戳|✅ 专版|权重与代码公开；商业使用须登记并遵守基座条款|GUI、定位与视频；深融合提高改造和维护成本|
|mPLUG\-Owl|Owl2 模态自适应模块；Owl3 Hyper Attention|✅ Owl3|Owl2 旧代；Owl3 低频维护，7B 检查点 Apache\-2\.0|长视觉序列；作者加速数据需按目标硬件复测|
|Eagle|多视觉编码器预对齐与 token 拼接|✅ Eagle2|家族仍更新；Eagle2\-9B 权重 CC\-BY\-NC\-4\.0|多塔研究与视频；视觉成本高、公开权重非商业|
|Cambrian|四视觉塔 \+ Spatial Vision Aggregator|🟡 非主线|代码 Apache\-2\.0；权重继承基座许可，项目低活跃|视觉中心基准与多塔消融；生产依赖复杂|
|VILA／NVILA|高空间／时间分辨率后压缩，支持多图与长视频|✅ 原生|代码 Apache\-2\.0；NVILA\-8B 权重 CC\-BY\-NC\-SA|长视频和 NVIDIA 部署栈；商业许可不通过|
|Bunny|可插拔 EVA\-CLIP／SigLIP \+ S2\-Wrapper|❌ 非主线|权重与代码公开；2024 年后低活跃|轻量研究基线；部署前锁定依赖与上游许可|
|Mini\-Gemini|低分辨率 query 检索额外高分辨率特征|❌ 非主线|代码公开；数据／检查点多为 CC\-BY\-NC，疑似停更|少 token 高细节研究；维护与许可风险较高|
|LLaMA\-VID／LongVA|每帧 2 token 或超长视觉序列迁移|✅ 长视频|均为开放权重；非商业许可，低频维护或疑似停更|小时级视频研究；细节召回、显存和长期维护需验证|

*表 8：主流开放 VLM 家族的视觉路线、视频能力与开放状态目录（截至 2026 年 9 月 2 日）*

**选型应先按场景设硬门槛，再在幸存家族中比较：**

- 文档、OCR 与小目标：优先 Qwen、InternVL、Ovis2\.5、Pixtral、Gemma 4 高预算或 MiniCPM\-V，先固定有效像素和最大视觉 token，再测字段召回与 P99。

- 视频与时刻定位：优先 Qwen、GLM\-V、Molmo2、MiniCPM\-V 4\.5；长视频研究可加入 NVILA、LLaMA\-VID、LongVA，但非商业许可或维护状态可能直接淘汰。

- 移动端与低时延风控：优先固定或分桶分辨率、64—256 token 的确定性压缩与小模型；MiniCPM\-V、SmolVLM、Phi 可进入短名单，动态无限切片和输入相关剪枝需先验证最坏时延。

- 多塔与高视觉容量：Eagle、Cambrian、DeepSeek\-VL、Mini\-Gemini 适合作为上限或消融对照；若视觉侧资源受限，不宜仅凭通用榜单承担多编码器成本。

仍无法统一确认的项目包括：部分检查点在训练各阶段是否冻结视觉塔；Ovis2／2\.5、MiniCPM\-V 4\.6、DeepSeek\-VL2 的完整视频采样或压缩细节；Pixtral Large 是否完整继承 12B 的视觉配置；以及所有家族逐数据集的商用兼容性。相应字段应保留“未公开／待核验”，不得由同家族其他版本推断补齐。

---

## 三、性能证据：榜单领先不等于业务最优

> **本节要点**：同一来源的横向表只能形成候选短名单，无法形成生产结论；任务分数、视觉预算与系统效率必须在同一部署条件下联合比较。
> 
> 

### 3\.1 通用与细粒度视觉基准

*图 2：同一模型方汇总表中的代表性公开基准排名，分数仅用于短名单（来源：*[*\[7\]*](https://arxiv.org/abs/2509.18154)*）*

该组数据的价值在于同表口径相对一致，而非证明绝对领先。MiniCPM\-V 4\.5 在作者汇总的八项 OpenCompass 平均分中为 77\.0，但部分任务使用长推理或多次平均；相对 Qwen2\.5\-VL\-72B，它仅在 OCRBench 更高，其余多项更低。综合平均会掩盖业务真正关心的局部差异。[\[7\]](https://arxiv.org/abs/2509.18154)

独立框架同样不能消除全部差异。VLMEvalKit 采用生成式推理、答案抽取器和特定主榜归一方法，并明确不能保证复现第三方论文数字；同一 InternVL3\-78B 在官方页面与统一复跑中的 OCRBench、MMVet 仍有小幅差异。[\[13\]](https://arxiv.org/abs/2407.11691)

|模型与模式|MMMU|OCRBench|DocVQA|ChartQA|口径边界|
|---|---|---|---|---|---|
|MiniCPM\-V 4\.5 hybrid|67\.7|890|94\.7|87\.4|部分任务为 reasoning，模型方汇总|
|Qwen2\.5\-VL\-7B|58\.6|864|95\.7|87\.3|non\-thinking，同一汇总表|
|Qwen2\.5\-VL\-72B|68\.2|882|96\.4|89\.5|non\-thinking，规模不可与 8B 等成本比较|
|InternVL3\-8B|62\.7|880|92\.7|86\.6|动态分辨率，non\-thinking|
|GLM\-4\.1V\-9B thinking|68\.0|842|93\.4|87\.1|thinking，生成成本不同|

*表 2：同表数据仍含推理模式与规模差异，不可直接形成总排名（来源：*[*\[7\]*](https://arxiv.org/abs/2509.18154)*）*

### 3\.2 OCR、文档、视频与效率表现

参数规模与局部任务并非单调关系：Qwen2\.5\-VL\-7B 的 TextVQA 为 84\.9，高于同系列 72B 的 83\.5；Qwen3\-VL\-8B 的 Thinking 模式提高 MMMU，却低于 Instruct 模式的 OCRBench。模型选择必须锁定 checkpoint 与模式，不能把各列最高分拼成一个“理想模型”。[\[4\]](https://arxiv.org/abs/2502.13923) [\[5\]](https://arxiv.org/abs/2511.21631)

FastVLM 提供了较清晰的受控效率证据：在 M1 MacBook Pro 上，FastViTHD 从 256 分辨率、16 token 提高到 1024 分辨率、256 token 时，视觉编码时延从 10\.1 毫秒升至 235\.6 毫秒，TextVQA 和 DocVQA 同时改善；这证明高分辨率存在明确收益和成本，但端侧实验不能外推到 A100 服务集群。[\[12\]](https://arxiv.org/abs/2412.13303)

|比较维度|必须固定|必须同时报告|常见误判|
|---|---|---|---|
|图像质量|原图、像素上下限、tile、提示与解码|字段召回、FPR、视觉 token|只报 OCRBench 或平均准确率|
|服务效率|GPU、引擎、量化、batch、并发、输出长度|P50/P95 TTFT、端到端 P99、req/s、峰值显存|把离线总耗时当单请求延迟|
|视频|时长、FPS、帧数、分辨率、字幕|短事件召回、时间定位、每视频成本|只比较同名 Video\-MME 分数|
|量化|权重格式、视觉塔精度、KV 配置|量化前后业务质量与稳定性|把“可运行”视为达到 SLA|

*表 3：模型性能比较需将任务质量与系统成本放在同一实验设计内*

提示词敏感性研究在多个模型和基准上观察到最高约 15% 的准确率偏差；MMMU\-Pro 又显示，过滤文本捷径并增加 vision\-only 设置后，各模型相较原 MMMU 明显下降。榜单更适合发现能力候选，业务回归集才适合决定阈值与上线。[\[14\]](https://arxiv.org/abs/2509.03986) [\[15\]](https://arxiv.org/abs/2409.02813)

---

## 四、方案比较：按风控约束选择模型族

> **本节要点**：先用数据驻留、时延、吞吐、输入形态和可审计性做硬门槛，再按子场景比较视觉路线；模型族仅形成候选，不产生无条件推荐。
> 
> 

### 4\.1 主流开源模型族画像

*图 3：代表性开放权重模型族在视觉路线、许可与部署定位上的差异（来源：各官方技术报告与模型卡）*

Qwen3\-VL、InternVL3\.5 提供从小模型到集群级模型的规模梯度，适合构建同族精度—成本曲线；MiniCPM\-V 4\.6 提供多种量化与移动端路径，适合隐私敏感或资源受限部署；GLM\-4\.6V 同时提供 9B Flash 与 106B\-A12B；Ovis2\.5、Pixtral 更突出原生分辨率与二维结构。许可审查仍应分别覆盖代码、权重、训练数据和下游分发，不能只看仓库顶层标签。[\[41\]](https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct) [\[42\]](https://huggingface.co/openbmb/MiniCPM-V-4.6)

### 4\.2 决定性差异与子场景适用条件

|风控子场景|适用条件|优先候选路线|硬门槛与组合|不宜采用的条件|
|---|---|---|---|---|
|静态内容审核|图文隐含语义较多，政策可结构化，允许复核|专用视觉 Guard；通用候选用 Qwen/InternVL 做二审|按政策子类校准 Recall/FPR；OCR 与规则并联|要求单模型自动封禁且不可申诉|
|视频广告合规|需联合画面、字幕、语音与时间语义|时序定位优先 Qwen/GLM；吞吐优先 MiniCPM；研究型跟踪可测 Molmo2|固定 FPS、帧数和每帧 token；语音转写单独计成本|短瞬时违规不得漏检却只做均匀低帧率采样|
|证件字段与文档理解|长图、小字、表格与跨页一致性是主要难点|Qwen、InternVL、Ovis、Gemma 4 高视觉预算；端侧可测 MiniCPM|专用 OCR 输出字段、坐标与置信度；VLM 做语义核验|要求 VLM 自由文本直接写入核心字段|
|身份真实性|需识别局部篡改、活体、全息或动态安全特征|专用检测、活体与视频防伪为主；VLM 仅辅助解释|静态与动态攻击分测；高影响结果必须人审或强规则|以通用 OCR/DocVQA 分数替代防伪评测|
|退款与商品证据|需要理解多图、损伤描述与订单上下文|高分辨率通用 VLM 做语义二审，伪造检测专模做真实性信号|同时约束伪造 TPR 与真实图 TNR；保留申诉|仅因低 FPR 就自动放行或拒赔|
|合同与票据风控|关注算术、字段、跨页和外部记录一致性|文档强项模型 \+ OCR \+ 计算工具 \+ 数据库核验|关键字段级 Recall、证据定位与审计日志|把语言模型解释当作真实性证明|
|支付交易反欺诈|只有当案件含截图、票据或沟通材料时才引入视觉模型|交易/行为/图模型与规则为主，VLM 处理非结构化附件|金额加权召回、FPR、P99 与决策可追溯|纯交易结构化信号充分，却为追逐多模态榜单增加 ViT|

*表 4：模型族选择必须绑定风控子场景、硬门槛与最终裁决边界*

上述建议体现三个决定性差异。第一，文档与长图优先考察二维位置、有效视觉预算和切片上限；第二，视频优先考察时间编码、采样与 token 总量；第三，边端场景优先考察量化、数据驻留和失败降级。参数量只有在硬门槛通过后才进入比较。

**最强反对意见及回应：**

- 大型模型可能凭更强推理统一多个专模。回应：公开文档篡改和退款证据研究仍显示细微伪造低召回，且更大的生成模型增加时延、成本和攻击面，当前不支持取消专模。

- 专用 Guard 可直接替代通用 VLM。回应：Guard 在固定政策分类上更易校准，但跨文档、开放语义和新型规则仍需要通用理解层；两者承担的职责不同。

- 直接使用榜单第一可降低选型成本。回应：像素预算、Thinking、工具调用和评测器足以改变排序，生产成本与 FPR 又未进入总榜，短期省下的评测成本会转化为上线风险。

---

## 五、风控成果：公开案例集中于内容与文档

> **本节要点**：多模态模型已进入内容审核、商品检查和金融文档流程，但可量化成果多为厂商或客户自报；支付交易反欺诈的直接视觉证据仍接近空白。
> 
> 

### 5\.1 内容安全、审核与合规

#### 5\.1\.1 场景边界：多模态模型负责语义证据，不独立承担合规裁决

内容审核的目标是判断“给定内容在特定政策、地区、对象和传播语境下是否应被处置”。模型输入因此不能限于单张图片：广告还需联合文案、关键词、视频、落地页与账户线索；直播电商还需联合语音转写、屏幕文字、商品信息、主体资质和历史风险。多模态大模型适合识别跨模态含义、长尾场景与政策相关证据，但主体身份、行政许可、地域规则、交易真实性和高影响处置仍应由结构化数据、确定性规则、专用模型及人工审核共同完成。Google Ads 的公开链路和 BLM\-Guard 的研究均支持将多模态能力放在政策理解与复核层，而非赋予其单模型最终裁决权。[\[17\]](https://arxiv.org/html/2412.16215v1) [\[21\]](https://arxiv.org/html/2602.18193v2)

**“效果”必须按三种口径分开解释。**

- **线上 KPI**：反映规则、模型、账户信号、人工与申诉共同作用后的生产结果，不能归因于单一多模态模型。

- **业务离线集**：用于检验接近业务分布的分类能力、模态增益与成本，仍不能等同真实线上误伤率或拦截率。

- **公开 benchmark**：适合比较模型能力与类别短板；其类别体系、阈值、语言和样本分布与业务不同，不能直接作为上线门槛。

据此，内容安全试点应限定在“语义理解、风险分层、证据归纳与低风险自动化”范围内。封禁账户、永久下架、直播中断等高影响动作需要可重放证据、确定性策略和人工升级通道。

#### 5\.1\.2 任务全景：审核对象从单图扩展到内容、主体与交易上下文

审核任务应围绕业务对象拆分，而不能沿用笼统的 safe/unsafe 二分类。公开 Guard 模型已覆盖图像、图文提示和多模态回答，但单图、英语优化或有限政策类别仍是常见边界。[\[18\]](https://arxiv.org/html/2504.01081v2) [\[19\]](https://github.com/meta-llama/PurpleLlama/blob/main/Llama-Guard3/11B-vision/MODEL_CARD.md)

|任务域|必须联合的输入|多模态模型适合承担|需由其他组件约束|
|---|---|---|---|
|静态图像|原图、OCR、图像来源、自然图或合成图标记|风险分类、严重度判断、违规对象定位、政策理由生成|哈希库、已知违禁物检测、阈值校准与高风险人审|
|图文与对话|图像、图中文字、标题、用户意图、模型回答|指代消歧、图文矛盾或互补、上下文危害判断|提示注入隔离、输出结构校验、工具权限控制|
|视频与直播|关键帧、缩略图、ASR、画面文字、音频、时间位置|跨帧事件识别、风险片段定位、多模态证据归纳|采样策略、实时专模、流式时延控制、人工巡查|
|广告与电商|素材、文案、落地页、商品、价格、主体与地区|夸大宣称、冒充、商品与展示内容一致性、开放语义长尾|资质、价格规则、地域政策、账户与交易真实性|

*表 4A：内容审核任务应按输入形态、模型职责与确定性约束共同定义*

图像 Guard 可作为高吞吐风险评分器；通用视觉语言模型更适合处理图文隐含语义和开放政策复核；视频则额外引入采样、时序和多语言转写成本。Zefr 的业务风格离线研究同时使用画面、缩略图、关联文本与自动转写，说明视频审核收益来自多源证据联合，而非单帧分类分数。[\[20\]](https://arxiv.org/html/2508.05527v1)

#### 5\.1\.3 系统链路：低成本预筛与语义复核分层协同

*图 3A：内容进入、低成本预筛、多模态语义复核、策略裁决、人工升级与申诉反馈构成闭环（来源：*[*\[17\]*](https://arxiv.org/html/2412.16215v1)*、*[*\[20\]*](https://arxiv.org/html/2508.05527v1)*）*

链路的关键是按风险和成本逐级缩小处理集合。第一层完成文件安全检查、去重、OCR、语音转写、关键帧采样和已知规则命中；第二层由低成本专模或共嵌入召回高疑似内容；第三层才调用多模态模型，结合政策文本输出类别、置信度、证据位置与理由；第四层的策略引擎联合主体、商品、地域和历史信号，形成放行、限流、拦截或升级建议。Google Ads 已公开“快速召回—细粒度模型复核—低置信人工审核”的生产形态，证明分层架构可行，但其整体结果不能拆算为某个模型的独立贡献。[\[17\]](https://arxiv.org/html/2412.16215v1)

人工审核应集中处理低置信、证据冲突、新政策、高影响动作与用户申诉。审核结论回流时需区分标签修正、政策变更、模型错误和输入质量问题，并保留原始内容哈希、抽帧与转写参数、模型和提示版本、阈值、规则命中、处置及申诉改判记录。这样才能判断问题来自模型、策略还是数据漂移。

#### 5\.1\.4 效果证据：生产结果、业务离线集与公开基准不能混算

现有公开证据足以支持“多模态模型可进入审核链路”，尚不足以支持“单模型可替代完整审核系统”。最接近生产的 Google Ads 案例证明分层流程能够规模化处理违规广告；可复算数字则主要来自业务离线集和模型内部测试集，适用范围更窄。

|证据与等级|公开结果|可支持的判断|不可外推|
|---|---|---|---|
|Google Ads<br>生产链路|共嵌入召回、语言模型复核与人工升级进入生产；烟草图像业务离线 Precision 为 90\.8%|多模态语义复核可嵌入大规模广告政策执行|未披露绝对线上 Recall、FPR、总流量及组件消融|
|ShieldGemma 2<br>厂商内部集|色情、危险、暴力 F1 分别为 88\.6、93\.7、85\.0|小型专用 Guard 可形成可调阈值的图像审核基线|三类、英语与合成测试集结果不能代表全政策生产分布|
|Llama Guard 3 Vision<br>厂商内部集|prompt 分类 Recall 0\.623，response 分类 Recall 0\.916；任务变化带来显著差异|输入审核与回答审核必须分别校准阈值|一次一图、英语优化，不能代表直播和多图长上下文|
|Zefr<br>业务风格离线集|1500 条视频上，最佳受测模型整体 F1 为 0\.91，人工为 0\.98|多模态首筛可降低人工负担，人审仍保有明显质量优势|三类风险、固定样本与最长前 60 秒不能代表全量平台流量|
|BLM\-Guard<br>任务专项研究|将多模态广告合规作为独立政策适配任务|广告合规需要超出通用色情、暴力分类的专项建模|论文离线结果不能直接转化为线上拦截率或申诉率|

*表 4B：审核效果证据应按生产、业务离线与研究基准分级解释（来源：*[*\[17\]*](https://arxiv.org/html/2412.16215v1)*、*[*\[18\]*](https://arxiv.org/html/2504.01081v2)*、*[*\[19\]*](https://github.com/meta-llama/PurpleLlama/blob/main/Llama-Guard3/11B-vision/MODEL_CARD.md)*、*[*\[20\]*](https://arxiv.org/html/2508.05527v1)*、*[*\[21\]*](https://arxiv.org/html/2602.18193v2)*）*

总体 F1 还会掩盖类别差异。Llama Guard 3 Vision 的不同危害类别表现跨度明显，ShieldGemma 2 在外部重标数据上也无法沿用内部集的统一口径；因此验收必须同时报告类别级 Recall、Precision、FPR、FNR 和样本量。Zefr 实验中多模态版本整体优于文本版本，但人工 F1 仍为 0\.98，支持“自动首筛与语义复核、人工作最终保障”的组合，而不支持取消人工。[\[18\]](https://arxiv.org/html/2504.01081v2) [\[19\]](https://github.com/meta-llama/PurpleLlama/blob/main/Llama-Guard3/11B-vision/MODEL_CARD.md) [\[20\]](https://arxiv.org/html/2508.05527v1)

#### 5\.1\.5 失败模式：攻击、语境与分布漂移会同时制造漏放和误伤

|失败模式|典型表现|业务后果|必要测试|
|---|---|---|---|
|类别与阈值失配|色情、暴力等显性类别较强，仇恨、自残、隐喻或边界政策较弱|总体分数合格，高风险细类仍持续漏放|逐政策类别混淆矩阵与阈值曲线|
|OCR 与组合语义|文字或图片单独安全，组合后形成违规含义；小字、变形字和遮挡降低识别率|图文梗、暗语、联系方式和规避词漏检|原图、OCR、联合输入与多尺度消融|
|长视频与稀疏事件|均匀低帧率采样错过短时违规，摘要又稀释时间证据|直播中短暂展示或口播无法及时处置|短事件召回、时间定位误差与峰值时延|
|语言与文化偏移|英语优化模型面对方言、隐喻、地区政策和文化语境时校准失效|地区间误伤不均，统一全球阈值产生系统偏差|按语言、地区、文化语境和政策版本分层|
|对抗与提示注入|微小图像扰动、隐藏文字或图片内恶意指令诱导护栏误判|攻击者绕过审核，或使模型越权调用工具|扰动、编码、转写、注入与输出结构红队|
|生产漂移|新型 AIGC、压缩转码、政策调整和流量结构变化偏离离线集|上线后 Recall 与 FPR 同时恶化|时间外样本、渠道分桶与持续抽检|

*表 4C：内容审核失败既来自模型能力，也来自输入、策略与生产分布*

对抗证据尤其不应被当作边缘问题。Llama Guard 3 Vision 的公开白盒实验显示，PGD 图像扰动会显著提高有害输入被判安全的比例；OWASP 同时将多模态提示注入列为可通过图像、文档等非文本载体进入系统的攻击面。生产系统必须把图片视为不可信输入，隔离内容理解与工具执行权限，并让结构非法、疑似注入或证据不一致的结果直接降级到规则或人工。[\[43\]](https://arxiv.org/html/2411.10414v1) [\[31\]](https://genai.owasp.org/llmrisk/llm01-prompt-injection/)

模型卡声明的边界同样应进入测试设计：ShieldGemma 2 的英语与三类政策范围、Llama Guard 3 Vision 的一次一图与英语优化，都会限制其对中文图文、长视频和直播的直接适用性。[\[18\]](https://arxiv.org/html/2504.01081v2) [\[19\]](https://github.com/meta-llama/PurpleLlama/blob/main/Llama-Guard3/11B-vision/MODEL_CARD.md)

#### 5\.1\.6 上线建议：以业务红线验收，并保留人工与回滚

建议采用“离线回放—影子流量—小流量辅助—受限自动化”四阶段路径。首轮仅验证多模态模型相对现网规则和专模的增量价值；影子阶段锁定模型、提示、视觉预算和政策版本，观测真实分布下的类别召回、误伤、时延与成本；小流量阶段先开放风险排序和人工辅助；只有低风险、证据充分、申诉可逆的动作达到稳定门槛后，才允许受限自动处置。

|门槛维度|核心指标|验收方式|未达标处置|
|---|---|---|---|
|风险效果|类别级 Recall、Precision、FPR、FNR、AUPRC|在目标业务阈值下对比现网，并单列高危类别与时间外样本|调整阈值、补专模或仅保留人工排序|
|用户影响|误伤率、申诉率、改判率、地区与人群差异|按语言、地区、内容类型和创作者层级分桶|停止自动处置，回退复核或提示|
|运营产能|人工复核率、队列 P95、单量处理时长|在峰值流量和新政策期间验证审核 SLA|收紧送审范围或扩充审核产能|
|系统效率|端到端 P95/P99、超时率、吞吐、单位请求成本|分别记录 OCR、抽帧、Guard、多模态复核和规则耗时|回退低成本专模、降低视觉预算或异步处理|
|安全审计|攻击成功率、证据定位率、日志完整率、可重放率|覆盖隐藏字、扰动、提示注入、结构非法与版本回滚演练|禁止高影响自动动作并切回安全规则|

*表 4D：内容审核上线门槛需同时覆盖效果、用户、运营、效率与安全*

阈值应设置为“自动拦截、人工复核、自动放行”三个区间，并在本地业务集上校准。ShieldGemma 2 可输出 Yes/No 概率，为三区间分流提供了实现基础，但公开模型卡没有给出可直接复用的业务阈值。[\[18\]](https://arxiv.org/html/2504.01081v2) 高影响动作还应满足三项硬约束：证据可定位、决策可重放、用户可申诉；任一条件缺失时，模型只能提供辅助判断。

**上线结论：**静态图文和广告合规可优先开展受控试点，专用 Guard 负责高吞吐主筛，通用多模态模型处理隐含语义与二审；视频和直播宜先做离线与影子验证，重点约束短事件召回、P99 与人审产能；任何场景均不应以公开 benchmark 总分替代本地政策集、攻击集和真实流量验收。Zefr 的多模态增益与人工质量差距共同表明，近期最稳健的目标是提高分流效率和审核一致性，而非追求完全无人审核。[\[20\]](https://arxiv.org/html/2508.05527v1)

### 5\.2 身份核验、交易反欺诈与文档风控

|证据|公开结果|可证明|不可证明|
|---|---|---|---|
|Sun Finance|字段抽取 79\.7%→90\.8%；单文档成本降 91%；处理至 5 秒内|组合式 IDV 管线有运营价值|不能归因于单一 VLM；样本量与独立审计未披露|
|Inscribe|人工审核约 30 分钟降至 90 秒内；客户自报避免潜在损失|多模型与像素取证组合可加速文档审核|反事实损失、Precision/Recall 与 ViT 贡献未披露|
|FraudBench|MLLM 平均真实图 TNR 0\.947，伪损伤 TPR 仅 0\.197|通用 MLLM 对合成退款证据存在系统性漏检|不能外推平台线上欺诈率|
|证件防伪研究|局部专模与动态防伪各有优势，未见攻击仍明显退化|静态整图 VLM 不能覆盖全部真实性信号|AUC 不等同线上欺诈通过率|
|Nubank 试点|联合交易记录、客户沟通与提交文档识别异常|支付案件存在多模态辅助空间|未披露 Recall、FPR、损失下降或审核时长|

*表 5：公开文档与身份案例能证明流程价值，但难以归因于 ViT*

Sun Finance 的组合管线同时使用 Bedrock、Textract 与 Rekognition；其欺诈管线 Accuracy 81%、Recall 59%，对新模式的背景相似度 Recall 仅 16\.7%。这既说明组合系统可提升字段处理，也说明新型攻击泛化仍弱。[\[22\]](https://aws.amazon.com/blogs/machine-learning/sun-finance-automates-id-extraction-and-fraud-detection-with-generative-ai-on-aws/)

Inscribe 把多个语言模型与像素取证、模板匹配专模结合，官方客户案例报告审核时间显著缩短；该证据支持“语义理解 \+ 专用取证”的系统模式，但没有给出可复算的分类指标或 ViT 消融。[\[23\]](https://aws.amazon.com/blogs/machine-learning/how-inscribe-uses-amazon-bedrock-to-stop-document-fraud-in-seconds/)

退款证据形成更强的反向信号：FraudBench 中通用 MLLM 对真实图较少误伤，却漏掉大多数合成伪损伤；某些专模提高伪造召回后又严重误伤真实图。身份核验也存在类似边界，FLiD 的局部字段专模与动态证件攻击研究均表明，局部像素和时间变化需要专门采集与检测。[\[28\]](https://arxiv.org/html/2605.08820v1) [\[24\]](https://arxiv.org/html/2605.09089v2) [\[25\]](https://arxiv.org/html/2607.06466v1)

**生产 KPI 稀缺的具体含义：**内容与文档领域能找到少量部署、处理时长、成本或“潜在损失”口径，但多数缺少线上 A/B、独立审计和组件消融；合同篡改缺少公开生产 Precision/Recall；支付交易反欺诈的公开成果主要来自交易、行为、图结构或对话模型，不能作为 ViT 视觉头的证据。Nubank 仅披露试点方向，尚无可复算收益。[\[30\]](https://openai.com/index/nubank/)

---

## 六、落地路径：模型理解，规则与专模裁决

> **本节要点**：稳健方案应把输入质量、结构化证据、开放语义、确定性策略与人工复核分层，并以影子流量和小流量金丝雀逐步验证。
> 
> 

### 6\.1 分层风控链路与人工复核

*图 4：从预处理到闭环监控的分层风控链路（来源：*[*\[32\]*](https://docs.cloud.google.com/vision/docs/ocr)*、*[*\[33\]*](https://docs.stripe.com/radar/rules)*）*

OCR 应保留文字、字段、坐标、置信度与文档质量，检测专模输出对象、区域和伪造信号，VLM 负责跨区域、跨模态语义归纳。规则层再结合账户、设备、金额、地区、历史行为与合规约束，输出放行、挑战、复核或拦截。该分工使生成式解释无法绕过确定性控制，也使每个最终动作可以追溯。[\[32\]](https://docs.cloud.google.com/vision/docs/ocr) [\[33\]](https://docs.stripe.com/radar/rules)

低质输入、证据冲突、疑似注入、模型低置信和高影响动作进入人工队列；人工结果回流为标签、规则修订与再训练数据。生成文本不应被自动视为忠实解释，审计日志至少保留原图哈希、预处理参数、OCR/检测结果、模型与提示版本、规则命中、阈值和最终处置。

### 6\.2 PoC 指标、压力测试与上线门槛

|维度|主指标|必要分桶与测试|建议门槛|
|---|---|---|---|
|风险效果|Recall、金额加权 Recall|Precision、FPR、AuPRC、攻击类型、时间外样本|相对现网有统计显著增益且不突破 FPR 红线|
|用户影响|正常用户误伤率|申诉率、改判率、挑战成功率、人群与地区|高风险子群无不可接受退化|
|运营|人工复核率|单量时长、队列 P95、抽检准确率|审核产能与 SLA 同时满足|
|效率|端到端 P95/P99|OCR、ViT、LLM、图查询与规则分段埋点|峰值并发与降级状态均达标|
|成本|每千请求总成本|GPU/API、人工、误杀、漏放、存储与网络|按目标 Recall/FPR 比较整链路净收益|
|稳定性|成功率、超时率|限流、解析失败、schema、一致率、版本回归|连续观测窗口达标并完成故障演练|
|安全审计|攻击成功率、证据一致率|扰动、隐藏字、提示注入、证据定位、可重放|高影响动作零越权，失败可回退|

*表 6：PoC 以业务阈值下的整链路结果为准，不以通用榜单验收*

Recall、Precision 与 FPR 必须同时报告，因为阈值提高通常增加 Precision、降低 Recall；准确率在极度不平衡的风控数据中可能失真。成本还应增加“每检出一个真实风险的成本”和“误杀、漏放损失”，避免裸 API 单价主导选型。[\[34\]](https://docs.aws.amazon.com/frauddetector/latest/ug/training-performance-metrics.html)

时延必须采用高分位数。Google SRE 建议用 P99/P99\.9 观察可行的最差体验，P50 只代表典型体验；视觉 token、OCR 队列、图查询和生成长度均可能制造尾部瓶颈。[\[35\]](https://sre.google/sre-book/service-level-objectives/)

### 6\.3 灰度、回滚与失效安全

1. 离线基线：冻结 checkpoint、提示、像素与切片参数，用时间外样本比较“现网”“现网\+OCR/检测”“现网\+VLM”“完整链路”，形成组件消融。

2. 影子流量：复制真实请求但不影响用户，只比较分桶效果、调用错误、时延、成本与人工队列。公开云服务的影子测试模式可作为实现参照。[\[36\]](https://docs.aws.amazon.com/sagemaker/latest/dg/shadow-tests.html)

3. 小流量金丝雀：新旧版本分流，从提示和排序等可逆动作开始，再按门槛逐级放量。[\[37\]](https://docs.cloud.google.com/deploy/docs/deployment-strategies/canary)

4. 受限裁决：低风险、证据充分且规则硬门槛通过后才开放自动处置；冻结、拒付、封禁等高影响动作保留人工批准或确定性规则。

失效安全应预先定义：VLM 超时或输出结构非法时回退 OCR、检测与现网规则；图服务不可用时回退本地特征与规则；疑似注入时丢弃生成结论并进入安全规则或人工。高风险动作不得因模型不可用而默认放行。NIST 亦要求组织准备非 AI 替代、断开与停用机制。[\[38\]](https://airc.nist.gov/airmf-resources/airmf/5-sec-core/)

---

## 七、综合判断：有条件采用，按子场景小步验证

**综合判断**：开放权重多模态模型在风控中“有条件可行”，适合承担语义理解与候选证据层，不宜直接替换核心裁决链。

这一结论来自三段证据链：视觉头能提高输入保真却伴随 token 与时延成本；整模型榜单受模式和评测口径影响；少量生产案例证明流程价值，但公开量化 KPI 难以归因于 ViT，且文档伪造、退款证据与对抗样本仍显示明显漏报。

残余风险 / 反向信号：

1. 同硬件、同像素预算、同量化和同业务集的公开横测仍不足。

2. 业务可接受的 Recall、FPR、P99、人审率与成本阈值尚未提供。

3. 新型伪造、提示注入与分布漂移会持续削弱静态离线结论。

**按子场景给出最终选择：**

|子场景|结论|成立条件|优先动作|
|---|---|---|---|
|静态内容与广告审核|✅ 优先试点|政策可结构化、允许人审、可按类别校准阈值|专用 Guard 做主筛，通用 VLM 处理隐含语义和二审|
|视频内容与品牌安全|🟡 受限试点|能承担多模态预处理成本，并能覆盖短事件与多语言|以固定 FPS/帧数做端到端压力测试，保留人审|
|文档抽取与一致性核验|✅ 优先试点|OCR 为事实层，VLM 仅做跨页语义与异常解释|在票据、截图和长文档上比较高视觉预算模型|
|证件真实性与活体|🔴 不作主裁决|仅作为辅助解释，动态防伪、专模与人工仍在链路|先建设局部篡改、全息与未见攻击测试集|
|退款与商品证据|🟡 二审可用|同时控制伪造召回与真实用户误伤，支持申诉|伪造检测专模 \+ VLM 多图语义 \+ 规则联合验证|
|合同与金融票据|🟡 组合采用|关键字段可定位，算术与外部数据可确定性核验|优先验证字段漏检、跨页冲突和审计可重放|
|纯支付交易反欺诈|🔴 暂不引入 ViT|只有案件含文档、截图或沟通材料时才增加视觉层|优先交易、行为、图模型和规则，VLM 处理附件|

*表 7：选型结论随风控子场景改变，不存在统一“最佳 ViT”*

采购或自研决策应以整链路净收益为准：增量拦截损失减去新增误杀损失、推理与人工成本、稳定性及合规风险成本。若主要痛点是小字与长图，先做输入保真、OCR 和高分辨率消融；若主要痛点是团伙关联，优先图模型；若主要痛点是不可审计，优先规则与决策编排；只有在开放语义长尾成为瓶颈时，VLM 才是首要新增组件。

---

## 待验证事项

**待验证与口径说明**（以下缺口会影响采购与上线结论）：

- 生产线上量化 KPI 稀缺：合同核验和视觉支付反欺诈缺少公开、可复算的生产 Precision/Recall、损失与人审收益；Google Ads、AWS 客户案例也缺少完整基线、样本量或独立审计。

- ViT 头与整模型因果尚未充分分离：多数公开结果比较不同模型全栈，需在同一语言模型、训练数据与推理配置下替换视觉头或做输入/压缩消融。

- 统一效率横测缺失：Qwen、InternVL、MiniCPM、GLM、Ovis 等仍需在同 GPU、引擎、量化、分辨率、并发和输出长度下复测 TTFT、P99、吞吐与峰值显存。

- 业务阈值未提供：可接受 Recall、FPR、正常用户误伤率、P99、人工复核率与单位成本必须由风险偏好、损失函数和审核产能反推。

- 量化与许可仍需逐项审查：跨后端量化后的业务精度缺少统一证据；代码、权重、训练数据与下游分发条款不可合并判断。

- 攻击与漂移需本域验证：公开对抗数字不能迁移为生产攻击率，应覆盖业务语言、图像质量、设备、地区、未见伪造和提示注入。

---

## 参考文献

- \[1\] LLaVA\-UHD, An LMM Perceiving Any Aspect Ratio and High\-Resolution Images, 2024\. https://arxiv\.org/abs/2403\.11703v1

- \[2\] NIST, AI RMF—Characteristics of Trustworthy AI Systems, 2023\. https://airc\.nist\.gov/airmf\-resources/airmf/3\-sec\-characteristics/

- \[3\] Google Research, Sigmoid Loss for Language Image Pre\-Training, 2023\. https://arxiv\.org/abs/2303\.15343

- \[4\] Qwen Team, Qwen2\.5\-VL Technical Report, 2025\. https://arxiv\.org/abs/2502\.13923

- \[5\] Qwen Team, Qwen3\-VL Technical Report, 2025\. https://arxiv\.org/abs/2511\.21631

- \[6\] OpenGVLab, InternVL3\.5 Technical Report, 2025\. https://arxiv\.org/abs/2508\.18265

- \[7\] OpenBMB, MiniCPM\-V 4\.5 Technical Report, 2025\. https://arxiv\.org/abs/2509\.18154

- \[8\] Ovis Team, Ovis2\.5 Technical Report, 2025\. https://arxiv\.org/abs/2508\.11737

- \[9\] Mistral AI, Pixtral 12B, 2024\. https://arxiv\.org/abs/2410\.07073

- \[10\] Google, Gemma 4 Model Card, 2026\. https://ai\.google\.dev/gemma/docs/core/model\_card\_4?hl=zh\-cn

- \[11\] UniPruneBench, A Unified Benchmark for Visual Token Pruning, 2025\. https://arxiv\.org/abs/2511\.02650

- \[12\] Apple, FastVLM: Efficient Vision Encoding for Vision Language Models, 2025\. https://arxiv\.org/abs/2412\.13303

- \[13\] OpenCompass, VLMEvalKit: An Open\-Source Toolkit for Evaluating Large Multi\-Modality Models, 2025\. https://arxiv\.org/abs/2407\.11691

- \[14\] Prompt Sensitivity in Multimodal Evaluation, 2025\. https://arxiv\.org/abs/2509\.03986

- \[15\] MMMU\-Pro: A More Robust Multi\-discipline Multimodal Understanding Benchmark, 2025\. https://arxiv\.org/abs/2409\.02813

- \[16\] Multi\-Modal Data Contamination Detection, 2025\. https://arxiv\.org/abs/2411\.03823

- \[17\] Google Ads, A Multimodal Framework for Scalable Image Policy Enforcement, 2024\. https://arxiv\.org/html/2412\.16215v1

- \[18\] Google, ShieldGemma 2, 2025\. https://arxiv\.org/html/2504\.01081v2

- \[19\] Meta, Llama Guard 3 Vision Model Card, 2024\. https://github\.com/meta\-llama/PurpleLlama/blob/main/Llama\-Guard3/11B\-vision/MODEL\_CARD\.md

- \[20\] Multimodal Models for Video Brand Safety, 2025\. https://arxiv\.org/html/2508\.05527v1

- \[21\] BLM\-Guard: Multimodal Ad Compliance, 2026\. https://arxiv\.org/html/2602\.18193v2

- \[22\] AWS, Sun Finance Automates ID Extraction and Fraud Detection, 2026\. https://aws\.amazon\.com/blogs/machine\-learning/sun\-finance\-automates\-id\-extraction\-and\-fraud\-detection\-with\-generative\-ai\-on\-aws/

- \[23\] AWS, How Inscribe Uses Amazon Bedrock to Stop Document Fraud, 2026\. https://aws\.amazon\.com/blogs/machine\-learning/how\-inscribe\-uses\-amazon\-bedrock\-to\-stop\-document\-fraud\-in\-seconds/

- \[24\] FLiD: Field\-Level Identity Document Forgery Detection, 2026\. https://arxiv\.org/html/2605\.09089v2

- \[25\] MIDV\-DynAttack: Dynamic Identity Document Attacks, 2026\. https://arxiv\.org/html/2607\.06466v1

- \[26\] GPT4o\-Receipt: AI\-Generated Receipt Detection, 2026\. https://arxiv\.org/html/2603\.11442v2

- \[27\] Multimodal Large Language Models for Receipt Tampering Detection, 2025\. https://arxiv\.org/html/2508\.11021v1

- \[28\] FraudBench: Multimodal Refund Fraud Detection, 2026\. https://arxiv\.org/html/2605\.08820v1

- \[29\] Amazon, Catalog Phrase Grounding for Product\-Brand Matching, 2023\. https://arxiv\.org/html/2308\.16354v1

- \[30\] OpenAI, Nubank Customer Story, accessed 2026\-09\-02\. https://openai\.com/index/nubank/

- \[31\] OWASP, LLM01:2025 Prompt Injection, 2025\. https://genai\.owasp\.org/llmrisk/llm01\-prompt\-injection/

- \[32\] Google Cloud, Detect and Extract Text from Images, accessed 2026\-09\-02\. https://docs\.cloud\.google\.com/vision/docs/ocr

- \[33\] Stripe, Radar Rules, accessed 2026\-09\-02\. https://docs\.stripe\.com/radar/rules

- \[34\] AWS, Fraud Detector Model Performance Metrics, accessed 2026\-09\-02\. https://docs\.aws\.amazon\.com/frauddetector/latest/ug/training\-performance\-metrics\.html

- \[35\] Google SRE, Service Level Objectives, accessed 2026\-09\-02\. https://sre\.google/sre\-book/service\-level\-objectives/

- \[36\] AWS, Shadow Tests for SageMaker Endpoints, accessed 2026\-09\-02\. https://docs\.aws\.amazon\.com/sagemaker/latest/dg/shadow\-tests\.html

- \[37\] Google Cloud, Canary Deployment Strategy, accessed 2026\-09\-02\. https://docs\.cloud\.google\.com/deploy/docs/deployment\-strategies/canary

- \[38\] NIST, AI RMF Core, 2023\. https://airc\.nist\.gov/airmf\-resources/airmf/5\-sec\-core/

- \[39\] GLM\-V Team, GLM\-4\.6V Technical Report, 2026\. https://arxiv\.org/abs/2507\.01006

- \[40\] Allen Institute for AI, Molmo2 Technical Report, 2026\. https://arxiv\.org/abs/2601\.10611

- \[41\] Qwen Team, Qwen3\-VL\-8B\-Instruct Model Card, 2025\. https://huggingface\.co/Qwen/Qwen3\-VL\-8B\-Instruct

- \[42\] OpenBMB, MiniCPM\-V 4\.6 Model Card, 2026\. https://huggingface\.co/openbmb/MiniCPM\-V\-4\.6

- \[43\] Multimodal Safety Attacks on Vision\-Language Guard Models, 2024\. https://arxiv\.org/html/2411\.10414v1

- \[44\] LLaVA Team, Improved Baselines with Visual Instruction Tuning, 2023\. https://arxiv\.org/abs/2310\.03744

- \[45\] LLaVA Team, LLaVA\-OneVision: Easy Visual Task Transfer, 2024\. https://arxiv\.org/abs/2408\.03326

- \[46\] LLaVA Team, LLaVA\-OneVision\-1\.5 Technical Report, 2025\. https://arxiv\.org/abs/2509\.23661

- \[47\] Hugging Face, Idefics2\-8B Model Card, 2024\. https://huggingface\.co/HuggingFaceM4/idefics2\-8b

- \[48\] Hugging Face, Idefics3 Technical Report, 2024\. https://arxiv\.org/abs/2408\.12637

- \[49\] Hugging Face, SmolVLM: Redefining Small and Efficient Multimodal Models, 2024\. https://huggingface\.co/blog/smolvlm

- \[50\] Microsoft, Phi\-3\.5\-vision\-instruct Model Card, 2024\. https://huggingface\.co/microsoft/Phi\-3\.5\-vision\-instruct

- \[51\] Microsoft, Phi\-4\-reasoning\-vision\-15B Model Card, 2025\. https://huggingface\.co/microsoft/Phi\-4\-reasoning\-vision\-15B

- \[52\] Google, PaliGemma 2: A Family of Versatile VLMs for Transfer, 2024\. https://arxiv\.org/abs/2412\.03555

- \[53\] Google DeepMind, Gemma 3 Technical Report, 2025\. https://arxiv\.org/abs/2503\.19786

- \[54\] Qwen Team, Qwen2\-VL Technical Report, 2024\. https://arxiv\.org/abs/2409\.12191

- \[55\] OpenGVLab, InternVL 2\.5: Expanding Performance Boundaries of Open\-Source Multimodal Models, 2024\. https://arxiv\.org/abs/2412\.05271

- \[56\] AIDC\-AI, Ovis: Structural Embedding Alignment for Multimodal Large Language Model, 2024\. https://arxiv\.org/abs/2405\.20797

- \[57\] Allen Institute for AI, Molmo and PixMo: Open Weights and Open Data for State\-of\-the\-Art Multimodal Models, 2024\. https://arxiv\.org/abs/2409\.17146

- \[58\] NVIDIA, NVLM: Open Frontier\-Class Multimodal LLMs, 2024\. https://arxiv\.org/abs/2409\.11402

- \[59\] Rhymes AI, Aria: An Open Multimodal Native Mixture\-of\-Experts Model, 2024\. https://arxiv\.org/abs/2410\.05993

- \[60\] Cohere Labs, Aya Vision: Advancing the Frontier of Multilingual Multimodality, 2025\. https://arxiv\.org/abs/2505\.08751

- \[61\] DeepSeek AI, DeepSeek\-VL2 Official Repository, 2024\. https://github\.com/deepseek\-ai/DeepSeek\-VL2

- \[62\] Zhipu AI, CogVLM2: Visual Language Models for Image and Video Understanding, 2024\. https://arxiv\.org/abs/2408\.16500

- \[63\] X\-PLUG, mPLUG\-Owl3\-7B Model Card, 2024\. https://huggingface\.co/mPLUG/mPLUG\-Owl3\-7B\-240728

- \[64\] NVIDIA, Eagle2\-9B Model Card, 2025\. https://huggingface\.co/nvidia/Eagle2\-9B

- \[65\] NYU VisionX, Cambrian\-1: A Fully Open, Vision\-Centric Exploration of Multimodal LLMs, 2024\. https://arxiv\.org/abs/2406\.16860

- \[66\] NVIDIA, NVILA: Efficient Frontier Visual Language Models, 2024\. https://arxiv\.org/abs/2412\.04468

- \[67\] BAAI, Bunny\-v1\.1\-Llama\-3\-8B\-V Model Card, 2024\. https://huggingface\.co/BAAI/Bunny\-v1\_1\-Llama\-3\-8B\-V

- \[68\] JIA\-Lab, Mini\-Gemini: Mining the Potential of Multi\-modality Vision Language Models, 2024\. https://arxiv\.org/abs/2403\.18814

- \[69\] JIA\-Lab, LLaMA\-VID: An Image Is Worth 2 Tokens in Large Language Models, 2023\. https://arxiv\.org/abs/2311\.17043

- \[70\] LMMS\-Lab, LongVA\-7B Model Card, 2024\. https://huggingface\.co/lmms\-lab/LongVA\-7B

- \[71\] Google Research, SigLIP 2: Multilingual Vision\-Language Encoders with Improved Semantic Understanding, Localization, and Dense Features, 2025\. https://arxiv\.org/abs/2502\.14786

- \[72\] Google Research, Patch n' Pack: NaViT, a Vision Transformer for Any Aspect Ratio and Resolution, 2023\. https://arxiv\.org/abs/2307\.06304

