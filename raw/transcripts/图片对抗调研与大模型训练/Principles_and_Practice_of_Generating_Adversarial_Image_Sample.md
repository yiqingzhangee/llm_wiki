## 一、对抗攻击

从攻击者的角度，用对抗思维理解 VLM 在真实业务场景中面临的主要风险。

对抗攻击，一开始是针对CNN的分类任务，攻击者对输入样本加入很小、通常人眼难以察觉的扰动，使模型产生错误判断。

  [图片]
  [图片]

核心原理是：

1. 模型学习到的用于分类的流形，相比人类感知的流行，在维度上更高，同时也更脆弱。

2. 假设模型正在判断一张猫的图片。模型内部可能更依赖一些局部统计特征，比如纹理、边缘、颜色分布。攻击者就利用这一点，在图片上加入非常细微的扰动，让模型内部提取到的特征发生偏移。

为了应用到理解能力更强的VLM上，存在难点：

- VLM视觉编码器通常更强

- 语言模型会进行语义修正（比如猫->沙发，VLM可能会输出「猫坐在沙发上」）

- 攻击目标更复杂（NTP）

### 1.1 迁移型黑盒攻击

迁移型攻击是目前对 VLM 最主流、最有效的黑盒攻击范式。其核心在于一个关键现象——**对抗攻击的可迁移性（Transferability）**：为一个模型（白盒代理）精心构造的对抗样本，往往能在不经任何修改的情况下，成功欺骗其他结构与参数完全不同的模型（黑盒目标）。

这种攻击的本质是在高维特征空间中进行“欺骗”。攻击者通过优化一个目标函数，使得对抗样本 `x_adv` 在代理模型 `f_surrogate` 上的特征表示，与一个良性目标图像 `x_tgt` 的特征表示尽可能接近，同时保证其与原始图像 `x` 的像素差异（扰动 `δ`）在人眼难以察觉的范围内。

其简化的目标函数可以表示为：$\max_{\delta} \cos(f_{\mathrm{surrogate}}(x+\delta), f_{\mathrm{surrogate}}(x_{\mathrm{tgt}})) \quad \mathrm{s.t.} \quad \|\delta\|_p \leq \epsilon$

![board_MQk7wf9gFhJAttbMcVKcHylOnBg](board_MQk7wf9gFhJAttbMcVKcHylOnBg.svg)

> **要点/Takeaway**：迁移型攻击的核心是利用代理模型，在特征空间上“模仿”良性样本，从而欺骗目标黑盒模型。

#### 在代理模型上优化

这类攻击的要点是代理模型的选择和损失函数的设计。

[图片]

| **方法** | **代理模型 / 特征来源** | **损失函数 / 优化目标** | **关键设计** | link |
| --- | --- | --- | --- | --- |
| **M-Attack** | CLIP-B/16 + CLIP-B/32 + CLIP-Laion-G/14（纯视觉 ensemble，3 个 CLIP image encoder） | image-image 余弦相似度，只拉近 adv 与 target 图特征（ensemble 内取平均） | 解决“传统 transfer attack 对闭源 MLLM 不够强”的问题，提出一个很强的简单基线：随机局部裁剪 + CLIP ensemble，让扰动学到更清晰的局部语义。 | https://github.com/VILA-Lab/M-Attack |
| **FOA-Attack** | 3 个 CLIP（B/16 + B/32 + Laion-G/14）    和M-Attack一样 | 两部分：① global feature loss：基于 cosine similarity，使 adversarial sample 的全局 [CLS] 特征接近 target sample；② local clustering optimal transport loss：先对局部 patch tokens 聚类，再用 OT 对齐 adversarial 与 target 的局部特征分布。 | 解决 M-Attack / 早期方法主要依赖全局或粗粒度特征的问题；在全局 [CLS] 对齐之外，进一步引入 patch-level 局部特征、聚类和最优传输，做更细粒度的局部特征分布对齐。 | https://arxiv.org/abs/2505.21494 |
| **V-Attack** | 主要使用 3 个 CLIP（B/16 + B/32 + Laion-G/14），但它攻击的不是普通 patch token (X)，而是 Transformer attention 中的 Value features (V)。 | patch token × text embedding 的余弦相似度——推离 source 文本、拉近 target 文本（不用 target 图）；只在文本注意力高的 patch 上计算（自适应 mask） | 不再主要操作 patch token，而是转向 Transformer attention 里的 Value features，用文本提示定位 source/target 概念，实现局部概念级操纵。 | https://arxiv.org/abs/2511.20223 |
| **MPCAttack** | 跨范式代理模型：cross-modal alignment，如 CLIP；multi-modal understanding，如 InternVL / InternVL3；visual self-supervised learning，如 DINOv2。聚合三种学习范式的 surrogate features。 | 用 InternVL3 给 source/target 生成 caption，把图特征和文本特征加权融合； 然后把 5 个 backbone 特征拼成一个超长向量，让 adv 接近 target 向量、远离 source  向量 | 解决单一代理范式带来的表示偏差和迁移瓶颈；不只用 CLIP/CLIP ensemble，而是聚合 CLIP、InternVL、DINOv2 等不同学习范式特征，并协同优化 | https://arxiv.org/abs/2603.04846 |

在100张**seed1.8**分类为**色情违规**的图像上进行实验。

| **方法** | **<font color="gray">M-Attack</font>** | **FOA-Attack** | **V-Attack** | **MPCAttack** |
| --- | --- | --- | --- | --- |
| ASR | **<font color="gray">42</font>** | **63** | **69** | **92** |
| time/s·张 | 150 | 100 | 60 | 80 |
| [图片] 目标图片 | [图片] <font color="gray">慢且asr一般</font> | [图片] 效果还行 | [图片] | [图片] 效果不错，较明显 |

> **攻击效果：MPCAttack > V-Attack > FOA-Attack > M-Attack**

在tcc上部署MPC了Attack

```yaml
name: feed_health.adversarial_mpcattack
version: 0.0.0

inputs:
  - payload

outputs: 
  - exist_resp
  - adv_b64
  - meta_out

cache:
  psm: toutiao.redis.mlf_core_cache
  ttl: 600
  type: redis
  hash: true
ops:
  - type: LaplaceMatx
    config:
      psm: "bric.feed_health.adversarial_mpcattack.service.sg1,bric.feed_health.adversarial_mpcattack.my,bric.feed_health.adversarial_mpcattack.service.my2,bric.feed_health.adversarial_mpcattack.service.my3"
      cluster: "default,default,default,default"
      model_name: MPCAttack
      connect_timeout: 1000
      rpc_timeout: 100000
      inputs:
        - input
      outputs: 
        - adv_image
        - metadata
    inputs:
      - payload
    outputs:
      - adv_b64
      - meta_out

  - expr:
      exist_resp: cond(adv_b64 != nil, 1, 0)
    force: true
```

```python
import base64
import json
from pathlib import Path
from laplace import Client

client = Client(
    "**sd://bric.feed_health.adversarial_mpcattack?idc=my&cluster=default**",
    transport="ttheader",
    timeout=100000,
)

def img_to_b64(path: str) -> str:
    return base64.b64encode(Path(path).read_bytes()).decode("utf-8")


SRC_PATH = "dataset/big_ansa_unique_4010_200/0eb50e45b1d93763f594938b589f92da.jpg"
TGT_PATH = "dataset/benign_100/rtneg__videos__0017_o8oMoLPaQQHARfXAQRbAJPz8UxejChIHfgDWK5.png"

# 整包 payload：图 + 任意攻击的超参
payload = {
    "source_image": img_to_b64(SRC_PATH),
    "target_image": img_to_b64(TGT_PATH),
}

# 计算图 inputs: [payload]，所以这里只发一个字段 payload
res = client.matx_inference(
    "MPCAttack",
    {
        "input": [json.dumps(payload, ensure_ascii=False).encode("utf-8")]
    }
)

print(res)
# 计算图 outputs: adv_b64 / meta_out
adv_b64 = res.output_bytes_lists["adv_image"][0].decode("utf-8")
meta = json.loads(res.output_bytes_lists["metadata"][0].decode("utf-8"))

from io import BytesIO
from PIL import Image
import matplotlib.pyplot as plt

img = Image.open(BytesIO(base64.b64decode(adv_b64)))
img.save("generated.png") 

plt.imshow(img)
plt.axis("off")
plt.show()

print("metadata:", json.dumps(meta, ensure_ascii=False, indent=2))
```

gce调用：由于

```python
import base64
import json
import requests
from pathlib import Path
from io import BytesIO
from PIL import Image

# 1. 图片转base64（和你原逻辑完全一致）
def img_to_b64(path: str) -> str:
    return base64.b64encode(Path(path).read_bytes()).decode("utf-8")

# 2. 配置参数（直接改路径即可）
SRC_PATH = "/Users/bytedance/Documents/zhangzhenfei.26/ANSA_minipic_detection/nsfw/0CALPwr.jpg"
TGT_PATH = "/Users/bytedance/Documents/zhangzhenfei.26/ANSA_minipic_detection/datasets/real_time/benign_100/rtneg__videos__0165_oEgC5EWypFvt3iU7ADBerxDdFg9QkIfR3EYTq4.png"
API_URL = "https://da46tj66.sg-fn.tiktok-row.net/run_algo_graph"

inner = {                                                                                                                                      
    "source_image": img_to_b64(SRC_PATH),                                                                                                      
    "target_image": img_to_b64(TGT_PATH),                                                                                                      
}                                                                                                                                              
                                                                                                                                                
payload = {             
    "payload": json.dumps(inner, ensure_ascii=False),                                                                                                                          
    "graph": "tcc://bric.bayes.algo_engine:creation_health:feed_health.adversarial_mpcattack",                                                 
    "mode": "1"                                                                                                                                                                                                
}   

try:
    response = requests.post(
        API_URL,
        headers={"Content-Type": "application/json"},
        json=payload, 
        timeout=120000
    )
    print(response)
    response.raise_for_status() 
    result = response.json()
except Exception as e:
    print(f"请求失败: {str(e)}")
    if 'response' in locals():
        print(f"错误响应内容: {response.text}")
    exit(1)
print(result)
        
# 5. 处理结果（和你原逻辑完全一致）
adv_b64 = result["adv_image"]
meta = json.loads(result["metadata"])

# 保存生成的对抗样本
img = Image.open(BytesIO(base64.b64decode(adv_b64)))
img.save("generated_adv.png")
print("对抗样本已保存为 generated_adv.png")
print("返回元数据:", json.dumps(meta, ensure_ascii=False, indent=2))
```

### 1.2 多尺度攻击：利用人与机器的“视觉差异”

与在像素级别添加噪声的迁移攻击不同，多尺度攻击利用了人眼与计算机视觉模型在不同观察尺度和频率下的感知差异。这类攻击生成的图像通常在一种尺度（如近看或高分辨率）下呈现为无害内容，但在另一尺度（如远看或低分辨率）下则暴露出其真实意图。

**要点/Takeaway**：视错图攻击利用了模型在不同频率和分辨率下的感知差异，实现了更高层次的语义伪装。

#### [smuggleBench](https://arxiv.org/pdf/2604.06950)

把攻击分为3类。

- 感知难度类（Perceptual Difficulty）

用一些策略的方法，包括低对比度文字、微小字体、复杂背景融合、旋转/变形文字等，利用视觉感知阈值差异使AI模型无法识别人类可读内容。

- AI illusion类

生成视觉双关图，即同一图像对人类与AI呈现不同语义层。

- 推理封锁（Reasoning Blockade）

即使AI成功识别文字，也通过语义混淆（如隐语、缩写、反讽语境）阻止模型理解违规语义。该路径专门针对具备OCR能力的模型。

[图片]


#### [Hateful Illusions](https://arxiv.org/pdf/2507.22617)

把仇恨言论、仇恨符号隐藏到ai illusion里。输入是一张二值图，输出ai Illusion图片

**作者模型失效的根本原因**:论文通过语义分析和注意力机制可视化发现，失败的根源在于当前模型广泛采用的 **CLIP-ViT** 视觉编码器。该架构倾向于关注图像的**局部、表层的细节**（如建筑物、树木），而**忽略了由这些细节在更高维度上构成的、需要全局感知的隐藏信息**。

[图片]

**有效的缓解措施**:

- **图像变换**: `高斯模糊 (Gaussian blur)` + `直方图均衡化 (Histogram equalization)` 的组合被证明是提升检测准确率最有效的预处理方法。它能帮助模型忽略局部细节，聚焦于整体轮廓。

- **训练策略**: 在训练专门的分类器时，`完全微调的提示学习 (Prompt Learning (F))` 方法效果最好，在测试集上达到了 **0.938** 的高准确率。训练优化两个用于分类的soft prompt 向量，然后测试的时候计算和harmful/safe的prompt 哪个相似度高。

  [图片]
  [图片]


#### PTDiffusion

- 利用图像傅里叶变换中的相位谱（Phase Spectrum）来解耦和迁移图像结构。把参考图像反向加噪，取中间特征的相位谱作为结构信息

输入图 → latent inversion → 文本/无条件引导 → 解码输出图。类似风格转换，区别是细粒度上和原图差距更大。

[图片]

| 原始图片 | 对抗图片/目标类别 |  |  |  | VLM 检测label |
| --- | --- | --- | --- | --- | --- |
| [图片] | [图片] moss | [图片] rock | [图片] fog | [图片] sunset | 无色情 |

这个方法的相关参数不太好调，对于不同的输入图片，其所需要的强度不同。可能要先对图片进行一定的预处理

#### Resolution Attack

希望生成两种含义的图片，满足下面的要求。

1. 在高分辨率下，图像必须看起来自然，无明显伪影，并忠实地传达第一种语义

2. 低分辨率版本不仅要能误导分类器，使其输出第二种语义（如“狗”），而且在人类看来也必须令人信服地呈现第二种语义。

方法是最终用于下一步去噪的噪声 `ε` 由两部分组成：低频部分来自 `ε_L`，高频部分来自 `ε_H`。这样，图像的宏观结构由低分辨率语义主导，而细节纹理则由高分辨率语义填充。并且在去噪的不同阶段关注不同的目标。如果是I2I，则采用 深度图 (depth map) 和 软边缘图 (softedge image)，类似于ControlNet的机制提取语义。

| [图片] | [图片] | [图片] |

可以通过下采样去检测。

### 1.3 用文生图模型生成对抗图像

近些年，随着文生图模型的逐渐成熟，出现了利用文生图模型生成对抗图片的方法。

传统的**定向对抗攻击**（Targeted Adversarial Attacks）虽然精准，但生成攻击样本时，需要明确指定一个目标类别或文本描述作为监督信号，这限制了攻击的灵活性和可扩展性。例如，为 1000 个目标类别生成攻击样本，就需要重复 1000 次训练过程。希望利用生图模型，构建一个可扩展、灵活且高效的定向对抗攻击框架，使其能够将任意一张图片，转化为针对任意目标文本描述的对抗样本，并能成功攻击多种不同的黑盒 VLMs？

#### **AnyAttack**

AnyAttack 的核心思想是训练一个**解码器（Decoder）作为噪声生成器 F**，它能根据输入图像的特征，生成特定的对抗噪声 `δ`。当这个噪声 `δ` 添加到一个随机图像 `xr` 上时，得到的对抗图像 `x_adv = xr + δ` 在 VLM 的特征空间中，会与原始目标图像 `x` 的特征高度相似，从而欺骗 VLM 将 `x_adv` 识别为 `x` 的内容。

![board_TcKLwAaQmhjHz7bpH7zcaypxnHg](board_TcKLwAaQmhjHz7bpH7zcaypxnHg.svg)

更符合端到端的对抗图像生成的场景

#### **TSAttack**

- Stage 1：类似resolution attack，用扩散模型在采样过程中引入“参考图低频（高斯模糊） + 当前生成高频”的频率混合，生成外观自然但语义被扰动的图像。 

- Stage 2（可学习的残差扰动）：训练一个轻量 `U-Net` 输出受约束的小幅残差，使对抗图在多个 CLIP 视觉特征空间里与输入图显著偏离，提升迁移性与稳定性。

| **方法** | AnyAttack | TSAttack |
| --- | --- | --- |
| ASR | 83 | 87 |
| 用有害黄图生成100张，测试结果 | [图片] | [图片] |


## 不同场景下的绕过路径

这些攻击方法对线上业务的实际影响因场景而异。我们需要为不同策略线绘制风险地图，评估其被绕过的可能性与潜在危害。

| **策略风险线** | **主要攻击路径** | **与常规审核的异同** | example |
| --- | --- | --- | --- |
| **色情内容 (ANSA)** | 迁移型攻击（如 MPC-Attack）通过添加纹理扰动，使 VLM 无法识别裸露特征；视错图攻击将色情内容隐藏在图像的细节或特定频域中。 | 与常规的视觉伪装（如遮挡、拼接）不同，对抗攻击在像素层面是全局性的，更难通过局部 patch-based 的规则来检测。 |  |
| **欺诈导流 (Scam)** | 视错图/分辨率攻击可将导流信息（如二维码、联系方式）伪装成风景、美食等无害图像；迁移攻击也可通过扰动降低 VLM 对导流文字或标志的识别能力。 | 相比于传统的 OCR 对抗（如文字扭曲、背景干扰），这类攻击直接在图像语义层面进行伪装，使得基于文本识别的防御策略失效。 |  |
| **赌博宣传 (Gambling)** | 与欺诈导流类似，赌博画面（如赌博游戏界面）可以用迁移/文生图生成对抗攻击 赌博文本使用类似scam的ai illusion隐藏 | 这类攻击往往与 AIGC 生成的复杂背景结合，比简单的 Logo 遮挡或 P 图更具迷惑性，需要模型具备更强的多尺度理解能力。 |  |

**要点/Takeaway**：对抗攻击对不同业务线构成差异化威胁，其中色情和导流场景风险最高，其绕过方式比传统伪装更隐蔽。

## 二、实验与观察：从数据看攻击的有效性与边界

除了宏观的 ASR 数据，实验过程中还涌现出一些值得关注的现象。

**观察一：目标图像决定了对抗样本的“多样性”**

对于迁移型攻击（如 M-Attack, FOA-Attack 等），攻击过程需要一张良性的“目标图像”来引导特征对齐。实验发现，**使用不同的良性目标图片，可以生成视觉风格完全不同的对抗样本**。这意味着，我们仅需少量违规样本和大量无关的良性图片，就能低成本、大规模地生成无穷无尽、不重样的对抗样本库。

**要点/Takeaway**：目标图像的引入，为对抗样本的批量化、多样化生成提供了可能，极大降低了攻击成本。


## 三、防御调研与启发

### 4.1 相关防御技术

#### Randomized Smoothing (RS)：用噪声对抗噪声

随机平滑是简洁且理论基础扎实的防御方法之一。其核心思想非常直观：**在输入图像上注入少量随机噪声，然后对多次加噪后的预测结果进行投票，取多数票作为最终输出**。这相当于对模型的决策边界进行“平滑”处理，从而“抹掉”攻击者精心构造的、微小的对抗性扰动。

其平滑后的分类器 `g(x)` 可概念化为：$
g(x) = \underset{k}{\operatorname{argmax}} P(f(x+\xi)=k), \quad \text{其中} \quad \xi \sim \mathcal{N}(0, \sigma^2 I)$

**要点/Takeaway**：随机平滑通过引入噪声来破坏对抗扰动的结构，即便简化为单次加噪，也是一种有效的低成本防御。

[图片]

#### Diffusion-based Purification

扩散模型不仅能生成图像，还能“修复”图像。基于扩散的净化（Diffusion-based Purification）利用了这一点，其流程分为两步：

1. **短暂加噪**：将输入的对抗样本进行一个短暂的正向扩散（加噪），用随机噪声“冲刷”掉对抗伪影。

2. **反向去噪**：利用扩散模型强大的生成先验，从加噪后的图像中恢复出对应的“干净”图像。

实验表明，该方法对于消除迁移攻击带来的**视觉伪影**（如 TS-Attack 的纹理）效果显著，能有效提升后续 VLM 的识别准确率。

**要点/Takeaway**：扩散模型净化能有效“洗掉”对抗样本中的视觉伪影，恢复图像的原始语义。

  [图片]
  [图片]
  [图片]

| attack | asr/% | asr_rs/%（n=4） | **asr_rs/%（n=2）** | **asr_rs/%（n=1）** | asr_prune/% | prune+rs/%（n=1） |
| --- | --- | --- | --- | --- | --- | --- |
| M-attack | 58 | 35 | **33** | **36** | 31 | 25 |
| MPC-Attack | 92 | 70 | / | 72 | 34 | 32 |
