# Adversarial Example Generation and Detection: Survey and Experiments // 对抗样本生成与检测：调研与实验

## 对抗方法汇总

目前调研复现的攻击类型可归纳为 4 大攻击家族：语义伪装、像素表征、结构变换、通道渲染

|类别|方法|核心思路|
|---|---|---|
|Semantic camouflage语义伪装|风格迁移 / Text Illusion / Resolution Attack / |在视觉表象上做语义伪装，让人眼与模型读到不同信息|
|Pixel representation像素表征|AnyAttack / MPCAttack|直接在像素与特征空间上优化扰动，使输出图像向目标语义或目标样本靠拢。|
|Structural transformation结构变换|Collage Layout / Poisson Blend / Occlusion Patch / Imaging Degradation / Geometric Robustness / 低对比度文字|通过拼贴、融合、遮挡、拍摄退化和几何变换，改变内容显著性与鲁棒性表现。|
|Channel rendering通道渲染|AlphaDog / Phantom Tank|操纵 RGBA / alpha 通道，让不同显示条件下呈现不同内容。|

目标：不追求让生成样本与线上样本在整体分布上完全一致。相反，我更希望通过构造更具挑战性、覆盖更多攻击变化形式的样本，分析VLM审核对抗失败的原因，提升模型对不同类别和不同对抗方法的通用防御性能。

也就是说，生成样本与线上样本存在局部重合，验证了其现实有效性；而生成样本相对更难、更丰富、数量更多，则保证了其在鲁棒性增强和泛化防御训练中的价值。

### 一、语义伪装

#### 风格迁移

- AdaIN

    - **视觉风格**体现在通道统计量上。一张图经过 CNN 后得到多个channel的特征图，对每个通道的特征做归一化，然后用另外一张风格图的 mean/std 重新缩放，就能把**内容图**的空间结构保留住，但纹理、笔触、色调会更接近目标风格。

![Image](https://internal-api-drive-stream-sg.larkoffice.com/space/api/box/stream/download/authcode/?code=MzE1NzExMGFiYTQ2ZjBjNzdmYzQ4ZWExZTgwYmI2MjhfYjk4ZGExYTQ3ZTNmYjhhODFmOTliNmJkMTYwMTEzYzJfSUQ6NzY0NzA1ODI5NjYxMjE2MzI5OF8xNzgyMDQ2OTMxOjE3ODIxMzMzMzFfVjM)

![Image](https://internal-api-drive-stream-sg.larkoffice.com/space/api/box/stream/download/authcode/?code=ZGY0NzA5N2Q2YjRmNWE1YzNmZWM2NThmYmFmZGFkMmJfZjM3YWZlMTE2OWY0MTg0YWJmOTZjZDc3Mzg1NGU5ZGFfSUQ6NzY0NzA1ODExODY4NTMzMTE2Nl8xNzgyMDQ2OTMwOjE3ODIxMzMzMzBfVjM)

![Image](https://internal-api-drive-stream-sg.larkoffice.com/space/api/box/stream/download/authcode/?code=OGNmYzc1OTE1MzVjNzBmZTA0MTM5YWQ3MGE1OTQwNmJfM2FjZDcxYzk2NGY1Yzg5YTAwNjc0YjYyMWMxMGM5YTNfSUQ6NzY0NzA1OTA2MTQ0NzU0NDU0OV8xNzgyMDQ2OTMwOjE3ODIxMzMzMzBfVjM)

- CAST

    - 训了一个专门的风格迁移模型，把内容图和风格图映射到模型内部，再输出目标风格图像。与 AdaIN 相比，它更像端到端的风格编辑器。

![Image](https://internal-api-drive-stream-sg.larkoffice.com/space/api/box/stream/download/authcode/?code=ZGM4OGFiMjFmZWE5YmQ4Njc2MmY5YzM2MWM0NTk0OTlfMmZiODgwYzMxNjQxN2FkOGQ1YTMwZDdhYWNmZThmZTJfSUQ6NzY0NzA2Mzk1NjU4MTkyODY2OF8xNzgyMDQ2OTMwOjE3ODIxMzMzMzBfVjM)

![Image](https://internal-api-drive-stream-sg.larkoffice.com/space/api/box/stream/download/authcode/?code=MGQ0ZmRmN2ExNjY5OGQ2NTQyNjBlYjJhYjEyZDFiYTBfZjE3N2FkOGExZGIzN2QzOTFlN2JiYzU4NzMwOGM0ZmRfSUQ6NzY0NzA2MzkwMDY5NzA1NDkzOV8xNzgyMDQ2OTMwOjE3ODIxMzMzMzBfVjM)

![Image](https://internal-api-drive-stream-sg.larkoffice.com/space/api/box/stream/download/authcode/?code=M2JiZWJiYjI3YzFjMDg2YjFmNGZjMDA4NWI3MTM2YjBfZjYxNTI4Yjk1ZTE5ZTYzMWRiM2UwYTExZjczNzFkODVfSUQ6NzY0NzA2NDM2NTY2OTQwNDM4Nl8xNzgyMDQ2OTMxOjE3ODIxMzMzMzFfVjM)



- 其他具体方法……

#### Text Illusion

- controlnet图生图：把文本内容先转成二值控制图，再借助 ControlNet / 扩散模型把这些控制痕迹伪装成自然场景纹理。

![Image](https://internal-api-drive-stream-sg.larkoffice.com/space/api/box/stream/download/authcode/?code=NTg3NjJjYTk2ZWYxM2Q5NDdhZjRmYTk5N2ZjMmVhMjFfNTY1ZTA5YjdjYmQ0OWQ4N2FiYzJjZjBjNTlhODA1ZjBfSUQ6NzY0NzA2NzAwMDY3ODMyMTg4Nl8xNzgyMDQ2OTMwOjE3ODIxMzMzMzBfVjM)

![Image](https://internal-api-drive-stream-sg.larkoffice.com/space/api/box/stream/download/authcode/?code=NjhlNDU0ZmQ1Mjc4YjE0MGUxNzIzYWVlNmY4NjI3ZjhfOGM5ODNmZGU4ZmYxMDFkMjkwNTNlNDQzZjliOTg3MGVfSUQ6NzY0NzA2NzA5MjMwMjQ5OTU1Ml8xNzgyMDQ2OTMxOjE3ODIxMzMzMzFfVjM)

![Image](https://internal-api-drive-stream-sg.larkoffice.com/space/api/box/stream/download/authcode/?code=ZmY4MmRjMmZhMDI0NjA3Yzc5YTA3YmNjZjFkNDliZDFfZjg2NTZmM2EzNzRjZjdmNzJmZDVhODRmODNmMjUwMWZfSUQ6NzY0NzA2NzQxNDcyNzAwMzg3NF8xNzgyMDQ2OTMwOjE3ODIxMzMzMzBfVjM)

- 其他生成方法：不同生图模型、不同control图生成策略……

#### Resolution Attack（双 prompt 分辨率攻击）

- 原理：把两套 prompt 分别绑定到不同的生成阶段：让一套 prompt 在低分辨率阶段主导整体语义，另一套 prompt 在高分辨率阶段主导细节修饰。这样最终图像的“远看语义”和“近看细节”可能对应不同意图。

![Image](https://internal-api-drive-stream-sg.larkoffice.com/space/api/box/stream/download/authcode/?code=NzI1NTVkZmYwMzUwZWI0MTk0MGUzM2U5NjMzMmY5YmNfMGYyZjliMzE1MmYxN2VlOTZlM2YwMmFlZDRmODNhNmNfSUQ6NzY0NzA3MjE5MzA0MTYzMjk5N18xNzgyMDQ2OTMwOjE3ODIxMzMzMzBfVjM)

![Image](https://internal-api-drive-stream-sg.larkoffice.com/space/api/box/stream/download/authcode/?code=OWVlMjFjZWJkM2I2MTI3OWY0MjE0MjYyYzhjMmJhNDVfNDg3NDU3ZDhlZGIzODg0ZDcwODM5NzNlNzJkYjljNDlfSUQ6NzY0NzA3MjMxMTAyMDUyMzIzMV8xNzgyMDQ2OTMwOjE3ODIxMzMzMzBfVjM)

- 生成相对不稳定，到底算不算违规可能需要人工判断



### 二、像素级攻击

直接在图像像素或模型特征空间上做目标驱动优化，让输出图在视觉上尽量接近原图、但在模型空间更接近攻击目标。

#### AnyAttack

- 端到端对抗噪声生成：给定一张 clean image 和一张 target image，先把目标图编码到语义空间，再用专门训练的的 decoder 生成一段扰动，把这段扰动限制在小幅度范围内后叠加回原图。目标是让输出图在模型看来更像 target，而在人眼看来仍接近 clean。

![Image](https://internal-api-drive-stream-sg.larkoffice.com/space/api/box/stream/download/authcode/?code=NzE0YzliY2M1Y2ViYjRlZjlkNDE5N2I4OWQ0ZjUyYjhfYWNkYTdhYzcyNjExZjM4YTZkMmFjNGY4ZjQwYzc1ODlfSUQ6NzY0NzA3NTEwNjQyMTI5Njg2NF8xNzgyMDQ2OTMwOjE3ODIxMzMzMzBfVjM)

![Image](https://internal-api-drive-stream-sg.larkoffice.com/space/api/box/stream/download/authcode/?code=OTBhMWQ0NWZjNDRiOWU0OWIzNWE5MWQwMGQ5ZTgyZjdfMGUzY2JhMGFkZjVkZGJmN2M0ZmVlNThkYzI2N2U0NmVfSUQ6NzY0NzA3NTEzNjc3NjMxMDQ5NF8xNzgyMDQ2OTMxOjE3ODIxMzMzMzFfVjM)

- 只需要输入target语义，生成对抗图效率相对MPCAttack高（A100 ～20qps）

#### MPCAttack

使用多个代理模型（CLIP、Dinov2、InternVL）建模的表示作为优化目标，把对抗图拉向 target、远离 source

![Image](https://internal-api-drive-stream-sg.larkoffice.com/space/api/box/stream/download/authcode/?code=YjI4Y2JiODBiNDQ0ZDg1NmQ3NWIzNjk0MWE1NmE5YWJfYWE5MjhiNGQ2YThhODRmODAxMmU3YWJlYWQ3NzBiNTVfSUQ6NzY0NzA4MDE1NTExNDAxNjQ4Ml8xNzgyMDQ2OTMwOjE3ODIxMzMzMzBfVjM)

![Image](https://internal-api-drive-stream-sg.larkoffice.com/space/api/box/stream/download/authcode/?code=MGRlZjRmNGExYzAyZjU3MTI0MTliYzgxMTdiYThhMjZfODM2OTI1NTc4NjVlZTExMDNjMmM4ZWFmMGJmYWFkZGRfSUQ6NzY0NzA4MDE4MTE4NjIwMzM1NV8xNzgyMDQ2OTMwOjE3ODIxMzMzMzBfVjM)

- 攻击效果不错，但生成相对慢

### 三、结构化变换

#### Collage Layout（拼贴埋点）

小图合成，把把 risky 图后，再以角落、边缘或随机位置的方式贴到 safe 图上。位置可选择 corner / edge / random。

#### Poisson Blend（泊松融合）

通过泊松融合在边界上做更自然的亮度/梯度过渡，使 risky 图块嵌入 safe 图时不容易出现“贴纸边缘”，得到边缘更自然的拼接图。这是一种更强调局部视觉自然度的合成方式。调用 `cv2.seamlessClone` 实现

#### Occlusion Patch（遮挡贴片）

人为加入网格、迷宫或规则性遮挡，让风险内容只部分可见，降低显著性。

![Image](https://internal-api-drive-stream-sg.larkoffice.com/space/api/box/stream/download/authcode/?code=M2VjYjQwZDhhZWRhOTRmNzIyOTUwMTZkZGRiZTE5ZDhfNWRjYzg2N2NjNmRkM2ZiZWFhYzYzZTQ5NGE3Y2QyY2NfSUQ6NzY0NzA4MzgzODkyMTI0ODQ3OF8xNzgyMDQ2OTMxOjE3ODIxMzMzMzFfVjM)

![Image](https://internal-api-drive-stream-sg.larkoffice.com/space/api/box/stream/download/authcode/?code=YzM2MzgwM2JhMTA0ZWQyZjJkYTkwZDI5NzllNmE0MGFfNDk4MzBmMjQxY2Y1N2U3ZTM5OWIzMDRjZjM2YzUwNGNfSUQ6NzY0NzA4MzkwNzQzNDc0NTU2Nl8xNzgyMDQ2OTMxOjE3ODIxMzMzMzFfVjM)

![Image](https://internal-api-drive-stream-sg.larkoffice.com/space/api/box/stream/download/authcode/?code=NzdmYzIwMDAwYmFjYjcwMWY5ODllZDI3NTVlMGI2NGVfMzJjY2Y0NmNmMTg3OTZhZTY3NGIzNTdiOTcyODZmMzlfSUQ6NzY0NzA4Mzk2MTI1MjAyNDAyN18xNzgyMDQ2OTMxOjE3ODIxMzMzMzFfVjM)

#### Imaging Degradation（成像退化）  

#### Geometric Transformation（几何变换）

![Image](https://internal-api-drive-stream-sg.larkoffice.com/space/api/box/stream/download/authcode/?code=M2MzMzkzYWY4NzVkZDMwMjZlNzZkMTEwMjk5M2VmN2FfOTZkNDM2YzI1ZTA3NDkxYmI0OWJjYmE0M2E2NTdkYzdfSUQ6NzY0NzA4NDI2MzkzNzk4NjI2OF8xNzgyMDQ2OTMxOjE3ODIxMzMzMzFfVjM)

![Image](https://internal-api-drive-stream-sg.larkoffice.com/space/api/box/stream/download/authcode/?code=YTQ1Nzc3YmI1NWU0OGM0NzQ4NWVkYWEyMDEzM2I4MjFfOTUxNGExYjlhNjJkNzBjZjJiM2YyYzMyYjAxYTJiYzBfSUQ6NzY0NzA4NDc5OTA1MzQ4Mzc0OV8xNzgyMDQ2OTMxOjE3ODIxMzMzMzFfVjM)

![Image](https://internal-api-drive-stream-sg.larkoffice.com/space/api/box/stream/download/authcode/?code=N2FkYzQwZGE5MTFkYjE2ZWUxZGNhZGIyMjc0OWI3MThfOTI5NjM5MTc4ZGU2YTVhM2E5Mzc0MmVlNGNjOTk4Y2VfSUQ6NzY0NzA4NDg4ODk2NTAwOTExNV8xNzgyMDQ2OTMxOjE3ODIxMzMzMzFfVjM)

![Image](https://internal-api-drive-stream-sg.larkoffice.com/space/api/box/stream/download/authcode/?code=YTA2MzgzMGUzZjYwMjg4NmM0ZTcwMWExMjdkMWFhODlfNjA1NTQ0OGUzYzliZTVmYzA1ZTA0MThkNmY4ZmJkNGVfSUQ6NzY0NzA4NDYxMTg3Njg1MTQyMF8xNzgyMDQ2OTMxOjE3ODIxMzMzMzFfVjM)

- 攻击方法相对简单，但ASR不低

### 四、通道渲染

这类场景比较受限：评论区场景相对可行，视频抽帧场景下比较难

#### AlphaDog

同时准备两张图：一张更偏“模型可见”，一张更偏“人眼可见”。然后通过求解 alpha 通道，把两者压进一个 RGBA 图里。不同渲染背景或查看方式下，这张图会呈现出不同的视觉效果。

![Image](https://internal-api-drive-stream-sg.larkoffice.com/space/api/box/stream/download/authcode/?code=ZGE4ODU1Mzc4YTA3YmU5N2JhNWMwY2RjMjdhZDAwNTNfYzY5OTM3MTI0MWU4NDIyNTU3YzdmZmNkNmE3ZTEwY2NfSUQ6NzY0NzA5NDI4MTU2NzcwMjc1MF8xNzgyMDQ2OTMwOjE3ODIxMzMzMzBfVjM)



![Image](https://internal-api-drive-stream-sg.larkoffice.com/space/api/box/stream/download/authcode/?code=ZGNlMTNkYjRjZDNlZmNmNzczNTE3Mjc4NjUzZTg1YTZfOGM5NzkyYzA4ZDFmOGVjMmEwZWZjMjYwZDM4NDc4NjhfSUQ6NzY0NzA5NDQyNjE0MTQyOTQ2N18xNzgyMDQ2OTMwOjE3ODIxMzMzMzBfVjM)



![Image](https://internal-api-drive-stream-sg.larkoffice.com/space/api/box/stream/download/authcode/?code=MDNhZjFiMTM0ODBkYzI1MDgzMTgxYzBhMDc1OTlkZTJfM2Q0NjRiNDRhNzEwNWNiNjAwYzA0MjgwNmY4NjAwYjFfSUQ6NzY0NzA5NDQ4MTg1ODMwMTY2Nl8xNzgyMDQ2OTMwOjE3ODIxMzMzMzBfVjM)

#### Phantom Tank 

原理拆解：Phantom Tank 是经典的“白底看到 A、黑底看到 B”思路。做法是把 cover 图和 hidden 图编码进一个 RGBA 文件中，让不同背景色下的混合结果不同，因此同一张图在不同环境里会显示出两套内容。

![Image](https://internal-api-drive-stream-sg.larkoffice.com/space/api/box/stream/download/authcode/?code=ZjJmMjlkNTg0MzdjZDE4ZGQ5OWVhZWVlN2U2ZWNhZWRfOTQwNGI4NDIxMjE3ZDczMDY2MjI3YmE5NjZmZGJjZGZfSUQ6NzY0NzE0NzcyMzk0ODA4NDk1OV8xNzgyMDQ2OTMwOjE3ODIxMzMzMzBfVjM)

![Image](https://internal-api-drive-stream-sg.larkoffice.com/space/api/box/stream/download/authcode/?code=Yjk1YWVhODI4Y2JkZGZmMmIzZTVmYzJiM2QxMzViN2FfM2I1ZjA3MzU3ZDI3ZTE2NGI2ZDVjMjcyMTBmMzYzNjNfSUQ6NzY0NzE0NzcyNjU5NzE2NDc2NV8xNzgyMDQ2OTMwOjE3ODIxMzMzMzBfVjM)

![Image](https://internal-api-drive-stream-sg.larkoffice.com/space/api/box/stream/download/authcode/?code=ZjZlMzQ1YzNhOTk2MjNkYmY1ZmZkZDYwMDU3YzgxY2RfOGE3MjFmODhmZmZhNTU4ZWIyOTBiYWI0MTY0ZjljNTVfSUQ6NzY0NzE0NzcyNzgyNTk5NzU0MV8xNzgyMDQ2OTMwOjE3ODIxMzMzMzBfVjM)

相对局限在评论区、头像等场景

## 测评

- 测评图片数

|**攻击方法**|**telegram\_text\_ocr**|**gambling\_image\_asr**|**ansa\_image\_asr**|**redirection\_vlm\_asr**|
|---|---|---|---|---|
|AlphaDog|​\-|130|68|​\-|
|幻影坦克|​\-|130|68|​\-|
|AnyAttack|​\-|130|68|​\-|
|MPCAttack|​\-|130|68|​\-|
|文本生图|465|​\-|​\-|550|
|低对比文本嵌入|1000|​\-|​\-|1000|
|AdaIN 风格迁移|​\-|2600|1360|​\-|
|CAST 风格迁移|​\-|884|864|​\-|
|几何变换|​\-|1170|612|​\-|
|图像降质|​\-|1040|544|​\-|
|拼贴埋点|​\-|520|272|​\-|
|遮挡埋点|​\-|520|272|​\-|
|Poisson 无缝融合|​\-|520|272|​\-|

- ASR统计结果，颜色越深表示该任务对该方法越脆弱

![Image](https://internal-api-drive-stream-sg.larkoffice.com/space/api/box/stream/download/authcode/?code=ZDQzZjA0NTUyMzI0NDc0NmU1MWYwZTY4NTk0ZDdmNjRfNjA1NDhjYmJkMzgyODliNjJlZGZjMGJiYzI2NTM2YjhfSUQ6NzY0NzA5MjA4MDgyNjQ4NjQ5MV8xNzgyMDQ2OTMwOjE3ODIxMzMzMzBfVjM)

![Image](https://internal-api-drive-stream-sg.larkoffice.com/space/api/box/stream/download/authcode/?code=MTRiZDc3YmUxMTIxOTJiY2NlODlhZTg0OTZlMDVhODRfMTZlYmViMDlmN2NkMmE2ZDM4ZGQwNDMxMWI1ZjJlZDZfSUQ6NzY0NzA5NTM0NDg1MzUyMDA5Ml8xNzgyMDQ2OTMwOjE3ODIxMzMzMzBfVjM)

- 趋势基本相同，存在少量差距。

- 本地部署的qwen3vl\-4b 比 Qwen2\.5vl\-sft 检测更强



## All in one



### 方法分类、示例

|大类|子方法|一句话总结|VLM 能力缺失方向|例子|
|---|---|---|---|---|
|Semantic camouflage 语义伪装<br>|AdaIN 风格迁移|基于 instance normalization 将 risky 图迁移到另一种视觉风格，在保留主体结构的同时削弱原始风险语义。|**跨域泛化不足**|![Image](https://internal-api-drive-stream-sg.larkoffice.com/space/api/box/stream/download/authcode/?code=YmUyNmU1ZmFlMjA2NDgzNWQxMWYzZjRkYTdkZjk5MWRfNDk1ZDNkMjAwYzNlOGEwMmY0MzM1NjVjMDIzYjYwMTJfSUQ6NzY0NDkwMjE3MDY5MjY4NTUzNF8xNzgyMDQ2OTMwOjE3ODIxMzMzMzBfVjM)<br>|
||CAST 风格迁移<br>|使用内容感知风格迁移生成更自然的风格化 risky 图，测试模型在风格域偏移下的审核稳定性。|**跨域泛化不足**|![Image](https://internal-api-drive-stream-sg.larkoffice.com/space/api/box/stream/download/authcode/?code=MGM4MzVmNmM1MWUzZmZjMjViZDM2ZGFmYjM1ZmEwZTdfNTViNzY2YWNjYmIzYjFkMzFlZTVjNWVlMzZkNThlMjlfSUQ6NzY0NDkwMTk4NDMyNzMzOTc0MF8xNzgyMDQ2OTMxOjE3ODIxMzMzMzFfVjM)|
||Text illusion 文本生图|将 CSV 中的文本转为二值文本图，再通过 IFTG / ControlNet / SD 流水线生成带隐藏文本结构的 illusion 图。|**低频全局结构感知不足**：模型过度关注局部高频纹理和表层场景，无法像人类远看/缩小/眯眼那样识别隐藏文字。|![Image](https://internal-api-drive-stream-sg.larkoffice.com/space/api/box/stream/download/authcode/?code=OTFkMDcwMDEzNzg4ZjRhNzE2NDY3ZmQyNGY4Yzg0ZDFfOTIzMzFiZjkxZWE5NmNhN2M0MzljNDYwMTIzYzkwYWZfSUQ6NzY0NDkwMjY4NzE1MDAwMTg5M18xNzgyMDQ2OTMxOjE3ODIxMzMzMzFfVjM)|
||Resolution Attack, RA|通过双 prompt 生成高低分辨率下语义不同的图像，prompt1 表示高分辨率显式语义，prompt2 表示低分辨率隐含语义。|**低频全局结构感知不足**：模型在原图、高分辨率、低分辨率或缩略图下感知到的语义不一致，缺少跨尺度验证机制。|![Image](https://internal-api-drive-stream-sg.larkoffice.com/space/api/box/stream/download/authcode/?code=YmVkN2UxNjM3NTFmNzY2NTIzNzYzYmM5MTZjOTg2ZGFfMjk2YjgwNGIzZjY0ZjkyY2I4ZDZmYjhmZjA0NjJlNDZfSUQ6NzY0NDkwMzg0MzY0MTM5Njk2MF8xNzgyMDQ2OTMxOjE3ODIxMzMzMzFfVjM)|
|Pixel representation 像素表征级攻击|AnyAttack<br>|对 clean 图注入 target 图引导的像素级对抗扰动，使图像在人眼语义基本不变的情况下偏移模型视觉表征。|**视觉表征稳定性不足**：人眼看到的语义不变，但视觉 encoder embedding 已经偏离原风险语义。|![Image](https://internal-api-drive-stream-sg.larkoffice.com/space/api/box/stream/download/authcode/?code=NWM0ZTE5MWY3ODIwMzhjNTlkZGI2NzRiNjRmYThhMWFfNTY4ZjRiYmE1YTZhNWE2NTQ2YmIxZTFkYTU5MjJmODRfSUQ6NzY0NDkwNDQ2MzE5NDU1ODE4MV8xNzgyMDQ2OTMxOjE3ODIxMzMzMzFfVjM)|
||MPCAttack|通过多代理特征对齐，将源图特征向目标图特征迁移，诱导目标 VLM/VRM 漏检。|**视觉表征稳定性不足**：人眼看到的语义不变，但视觉 encoder embedding 已经偏离原风险语义。|![Image](https://internal-api-drive-stream-sg.larkoffice.com/space/api/box/stream/download/authcode/?code=YjU3ZWVmOGUyMDM5ZDgxM2U2ZWJmNzk4YTYwZTM3MjFfNWZmMzAzMGE5YzhkYmQ1MjgxNzM5MDA4MjNhYTc2YTJfSUQ6NzY0NDkwNjk3ODE1MzY0NzgzOV8xNzgyMDQ2OTMwOjE3ODIxMzMzMzBfVjM)|
|Structural transformation 结构化变换<br>|Collage layout 拼贴埋点|将 risky 图缩小后嵌入 safe 图，支持 corner / edge / random / large 等布局，测试小目标风险感知能力。|**低对比文本/弱视觉证据感知不足**：模型对低对比度、小尺寸、局部嵌入的文本风险不敏感，尤其容易被自然纹理背景掩盖。|![Image](https://internal-api-drive-stream-sg.larkoffice.com/space/api/box/stream/download/authcode/?code=N2Q3MWExNGQ0MDA5OWFmMTA2MWNjZmM2Yjc1YmJiZGZfMTU5ZTQxNjc0MDQzNzc3OGRjMTlkOTgxZGMxZmE3NzlfSUQ6NzY0NDkwNzA5MDQzMDkyMjQ2M18xNzgyMDQ2OTMxOjE3ODIxMzMzMzFfVjM)|
||Poisson blend 泊松无缝融合|使用 OpenCV cv2\.seamlessClone 将 risky 图无缝融合进 safe 图，使边缘梯度被抹除、主体更自然融入复杂背景。|**低对比文本/弱视觉证据感知不足\+前景\-背景分离能力不足 **|![Image](https://internal-api-drive-stream-sg.larkoffice.com/space/api/box/stream/download/authcode/?code=MGIwMzM5ZDY2MmQxMWM3MTQxYTA3MzFmMDdiMmUzYjVfMDMzMTY3ZjA2NDUxN2Q2YzViYTQ1NGRjMzQxZWU1MzVfSUQ6NzY0NDkwNzMzODYwMzY5NTgzOF8xNzgyMDQ2OTMwOjE3ODIxMzMzMzBfVjM)|
||Occlusion patch 遮挡埋点<br>|先对 risky 图施加网格、迷宫状黑色遮挡，再嵌入 safe 图，削弱底层风险语义。|**遮挡下语义恢复不足 \+ 高频纹理干扰鲁棒性弱**|![Image](https://internal-api-drive-stream-sg.larkoffice.com/space/api/box/stream/download/authcode/?code=MTEyYTNlYmZhYzYzNjM5ZTVkN2VmYjdkNWFmNjI1MmZfMzM5MmFiNTM5ZjU2OGQzZDhiMDY2ZWY5ZjI5MWFkMjdfSUQ6NzY0NDkwNzIzNTUyMjA4ODY2OV8xNzgyMDQ2OTMxOjE3ODIxMzMzMzFfVjM)|
||Imaging degradation 图像降质|通过 JPEG 压缩、缩放、模糊、中值滤波、posterize 等方式降低图像质量。|**跨域泛化性不足**|![Image](https://internal-api-drive-stream-sg.larkoffice.com/space/api/box/stream/download/authcode/?code=ZmJmNWE1ZjMxNzZlM2NmY2M3OGYxNmYwMDliNzE1MTBfNWE4NTQ4MWExNTU1NDBhNDUxZmNkZDQ1NmU3Zjg0ZTZfSUQ6NzY0NDkwNzYxMjM3MzQ4NzMyM18xNzgyMDQ2OTMwOjE3ODIxMzMzMzBfVjM)|
||Geometric robustness 几何变换<br>|对图像施加旋转、缩放、错切、透视等保持语义的几何扰动。|**跨域泛化性不足**<br>|![Image](https://internal-api-drive-stream-sg.larkoffice.com/space/api/box/stream/download/authcode/?code=NWVjOGUzMzU5YjBiNmQ0NGVmMDliMDY3MmI0MzU3OGNfM2E3MmM3YTkyYWY5Nzc4YmYzMTNlNzRlNDM4NGM5OTdfSUQ6NzY0NDkwNzUyNzE2NzA1MzUzMV8xNzgyMDQ2OTMwOjE3ODIxMzMzMzBfVjM)|
||Low\-contrast text embedding 低对比文本嵌入|在自然背景的低方差区域中，以低对比度、低分辨率方式嵌入文本，并叠加轻微几何噪声进行隐藏。|**低对比文本/弱视觉证据感知不足**：模型对低对比度、小尺寸、局部嵌入的文本风险不敏感，尤其容易被自然纹理背景掩盖。|![Image](https://internal-api-drive-stream-sg.larkoffice.com/space/api/box/stream/download/authcode/?code=ZWM2MWU2NzQ4NjM0ZGVhMDk4ZjFiNDlmYjhmZWI1YWZfNTMwMmY1MDdiMzZmNTlmMzI2YzBkYTg4ZWRkNGRkYmFfSUQ6NzY0NzA2MTE0Mzc4MTE2NjgxNF8xNzgyMDQ2OTMwOjE3ODIxMzMzMzBfVjM)|
|Channel rendering 通道渲染攻击|AlphaDog|通过求解 alpha 通道，让 RGBA 图像在丢弃 alpha 与白底合成时分别呈现不同图像。<br>|**输入解析/渲染一致性不足**：模型读取图像的方式和用户实际看到的客户端渲染结果不一致。|![Image](https://internal-api-drive-stream-sg.larkoffice.com/space/api/box/stream/download/authcode/?code=MDdiOTBjOWUyZjA3N2VlZThhMGQ2NzgyZmFlNDIyOWVfN2ZlMmZlMjZjYzZiNGI1NTlhYTlkNzcwZDMxMjJlYzJfSUQ6NzY0NDkwODU3MTU5MDY5MjU4MV8xNzgyMDQ2OTMxOjE3ODIxMzMzMzFfVjM)|
||Phantom Tank 幻影坦克|将 cover / hidden 两张灰度图编码进同一张 RGBA，使白底显示 cover、黑底显示 hidden。|**输入解析/渲染一致性不足**<br>|![Image](https://internal-api-drive-stream-sg.larkoffice.com/space/api/box/stream/download/authcode/?code=YzhlZTAzNTRjYmUzMmViZDg0NGExY2U0OTEyZWYyMDFfNzQwMzQzNzU4OGU1MzhhOGQ1MDBjM2I1NDEwZDZjODlfSUQ6NzY0NDkxMjM1NDg0NDY0MzA0Ml8xNzgyMDQ2OTMwOjE3ODIxMzMzMzBfVjM)<br>![Image](https://internal-api-drive-stream-sg.larkoffice.com/space/api/box/stream/download/authcode/?code=Y2E3MTZiNTBjNThlOWI3NTcyZjQ2ODdiNWE0NzAzNDdfNTMwZDUxNGUxNmY3OWYzNDg1YWU4ODk2ODk2ZjQwMDBfSUQ6NzY0NDkxMjAwOTIwNjE4OTc4OV8xNzgyMDQ2OTMwOjE3ODIxMzMzMzBfVjM)<br>![Image](https://internal-api-drive-stream-sg.larkoffice.com/space/api/box/stream/download/authcode/?code=OWNiYWIwNWZmOWExZTAzNmUzZDkwYWFkMWE3MDE4ODRfNDlhYjRjZjNkMTY3NTMwYjQwMWFjZGUwNzAyYTQ2Y2FfSUQ6NzY0NDkxMTg5MDMxMDMxOTgzOF8xNzgyMDQ2OTMxOjE3ODIxMzMzMzFfVjM)|

## 注意力原因分析

### Q1：常规prompt作为探针，能否注意到目标视觉token？



|大类|子方法|例子|Attention 能注意到相关视觉token|可视化|
|---|---|---|---|---|
|Semantic camouflage 语义伪装<br>|AdaIN 风格迁移|![Image](https://internal-api-drive-stream-sg.larkoffice.com/space/api/box/stream/download/authcode/?code=MzdkYThkNTU2NmM3ZTg5MWE3MWZkODIzZTBhZTY1M2NfZWJkNWJhNDhmOThlMDY2YjgyMDNhY2ExMjAzYTAyOGRfSUQ6NzY0NzA5NTg4MzI4MDA0MzczOV8xNzgyMDQ2OTMxOjE3ODIxMzMzMzFfVjM)<br>|✅|![Image](https://internal-api-drive-stream-sg.larkoffice.com/space/api/box/stream/download/authcode/?code=Njk4NTk1NGI0YjA5NDdlNWQ5NzEyMjI1MDBkMjU2MTVfOTE3MjMxYWM0MzQyYzVjNDg5ZjdiODJiMzkxMmZiMTJfSUQ6NzY0NzA5NTg4MTAyNzg5OTA5OV8xNzgyMDQ2OTMxOjE3ODIxMzMzMzFfVjM)<br>![Image](https://internal-api-drive-stream-sg.larkoffice.com/space/api/box/stream/download/authcode/?code=YTIyNDI3MWQ4ZWZmOTg3NjU2ZWVmZGVlOWYyZTQ3ODdfNjAxZjhjMDNmY2QyMzM2OGRhOTE4YzNkMTQ0NzA3YTlfSUQ6NzY0NzA5NTg4MTc2NjAzMTA3MV8xNzgyMDQ2OTMwOjE3ODIxMzMzMzBfVjM)|
||CAST 风格迁移<br>|![Image](https://internal-api-drive-stream-sg.larkoffice.com/space/api/box/stream/download/authcode/?code=ODc2NjM4M2UyNTEyNDdhMTc5YmY0NzBlZGEwNGE1OGVfMmQzOTNlMzlmZjM4NWYwN2U2NmFkZTUwY2VhM2IxNzZfSUQ6NzY0NzA5NTg4MzA3NDQ1NzMwOV8xNzgyMDQ2OTMwOjE3ODIxMzMzMzBfVjM)|✅|![Image](https://internal-api-drive-stream-sg.larkoffice.com/space/api/box/stream/download/authcode/?code=OWIxZTlhMTUxNTUzYjE1ODNiMDE4MjUyM2EzNGZiNDZfYTViZDE4ZWYzNzQ3NTFjYmQ4YTQyMWIxYmYyNmQzODNfSUQ6NzY0NzA5NTg4MTgwMTM3MTM1OF8xNzgyMDQ2OTMxOjE3ODIxMzMzMzFfVjM)<br>![Image](https://internal-api-drive-stream-sg.larkoffice.com/space/api/box/stream/download/authcode/?code=YWJjMzVlYjI4ODg0OTg0NTU1YmNlMzIyNGExM2I2ODRfZWU4OWE2ZTVlZjJiMDM0ZGE3ZjhiNmUwOWJmOTFlMTlfSUQ6NzY0NzA5NTg3OTM5MjMwMDc2NV8xNzgyMDQ2OTMxOjE3ODIxMzMzMzFfVjM)|
||Text illusion 文本生图|![Image](https://internal-api-drive-stream-sg.larkoffice.com/space/api/box/stream/download/authcode/?code=YzMxYmI1NzQ1NTRlZmQzNTZlZjAwZWIzMTExYTRiMGFfYzI3NGU0NmExYzVmZjJiZTg3ZDViOGEyNGZlZjk5NjZfSUQ6NzY0NzA5NTg4MDA2NzczMTE2N18xNzgyMDQ2OTMxOjE3ODIxMzMzMzFfVjM)|❌|![Image](https://internal-api-drive-stream-sg.larkoffice.com/space/api/box/stream/download/authcode/?code=MGRiNjg0MjBhZDE5ZjFkYjM2YWE1OTAwZDAyMjRhM2FfNzA4ZDNiZTBlZDhmNDVhYWFhNjc2MTEzZDhlNTg4YjNfSUQ6NzY0NzA5NTg4MTAyNzkzMTg2N18xNzgyMDQ2OTMwOjE3ODIxMzMzMzBfVjM)<br>![Image](https://internal-api-drive-stream-sg.larkoffice.com/space/api/box/stream/download/authcode/?code=MmMyNDk1NWYzYjEzNDFiMzQyMDVmMzQzMWEwZjdiNTVfYzA4YjNjMTFiOGEyNjcwOTZmZjU3ZjU1N2ZhYjgwMjFfSUQ6NzY0NzA5NTg4MjEzOTM0MDUxMV8xNzgyMDQ2OTMwOjE3ODIxMzMzMzBfVjM)|
||Resolution Attack, RA|![Image](https://internal-api-drive-stream-sg.larkoffice.com/space/api/box/stream/download/authcode/?code=OTY1NTE5ZjdiNmM0YjhmOGM1NTNhOGMxY2NjZjkxNjRfYWJhZjBjYzFjZGZhMWZkNDMzNGJkNDk1ZTQ4YzM0MGRfSUQ6NzY0NzA5NTg4MDAzODQ2OTM0MV8xNzgyMDQ2OTMxOjE3ODIxMzMzMzFfVjM)|❌<br>|![Image](https://internal-api-drive-stream-sg.larkoffice.com/space/api/box/stream/download/authcode/?code=NGI4M2IxNGQ0NzAyYWQwZmM1NDJkMzM0ZTg2ZGY3YzVfYTY0YzA0MzVlZjQ0NTllMWQ3OGNiYzU2MWQzODM3MWZfSUQ6NzY0NzA5NTg4MzI3OTk5NDU4N18xNzgyMDQ2OTMxOjE3ODIxMzMzMzFfVjM)<br>![Image](https://internal-api-drive-stream-sg.larkoffice.com/space/api/box/stream/download/authcode/?code=OTU4NWNjZDhkNDNiZTlmOTNmYjRmNGRkMWY4ZmQwZWZfMmI5Yzg0MTFhYWIyOWY3MjAzZmI1MWYwOGI2ZDQ5NDhfSUQ6NzY0NzA5NTg4MjkyMzQ2MjM2M18xNzgyMDQ2OTMwOjE3ODIxMzMzMzBfVjM)|
|Pixel representation 像素表征级攻击|AnyAttack|![Image](https://internal-api-drive-stream-sg.larkoffice.com/space/api/box/stream/download/authcode/?code=ZDk1Y2I4NWI0MDZmODdkMzM5NTA3Y2Y2NDkxZTNjYjhfYTExNDk4MzgzZjdhOTgyNjgyYzIzNGU2ZmI3NGRjYTVfSUQ6NzY0NzA5NTg4MDAzODQzNjU3M18xNzgyMDQ2OTMxOjE3ODIxMzMzMzFfVjM)|✅<br>|![Image](https://internal-api-drive-stream-sg.larkoffice.com/space/api/box/stream/download/authcode/?code=YmI2N2ZjZGMyZGYxM2U2ZTFkY2ViODk2M2MyZjM3NmJfZWJhOTFkMGU1ZmU3Yzc2NmVkNDhkOTA0ZjBmMjc2ZDJfSUQ6NzY0NzA5NTg4MjkyMzYwOTgxOV8xNzgyMDQ2OTMwOjE3ODIxMzMzMzBfVjM)<br>![Image](https://internal-api-drive-stream-sg.larkoffice.com/space/api/box/stream/download/authcode/?code=NDJhZjE3YWZkZmI3NmYzNjU0MWQ3ZDVmOTkzZGY3MDZfOTI4YzIzZjBhMTk3YzBjYmJkZWRmY2FkODY2Y2Q5ZWFfSUQ6NzY0NzA5NTg4MjEzOTI3NDk3NV8xNzgyMDQ2OTMwOjE3ODIxMzMzMzBfVjM)<br>|
||MPCAttack|![Image](https://internal-api-drive-stream-sg.larkoffice.com/space/api/box/stream/download/authcode/?code=YjAxM2M5YTdkYjFjNDJhNDJmM2EzM2NiNjVkNTY5Y2NfNzA2MjE3OGIyMWMyNDU5ZmE3YzFiMzQxYmRkYmI1MGFfSUQ6NzY0NzA5NTg4MzI3OTk0NTQzNV8xNzgyMDQ2OTMxOjE3ODIxMzMzMzFfVjM)|✅ <br>大部分能探测到；部分攻击特别强，直接探测不到了|![Image](https://internal-api-drive-stream-sg.larkoffice.com/space/api/box/stream/download/authcode/?code=YjVjNmI2MWYxMWViMWI0ZmRmNDQzM2IyYWRlZjBiOWZfNzVmYWFmZjA3Mjg3OWE5NzIwMWQ0NmNjYmM5MDMyNzhfSUQ6NzY0NzA5NTg3OTI0NDY0ODE1Nl8xNzgyMDQ2OTMwOjE3ODIxMzMzMzBfVjM)<br>![Image](https://internal-api-drive-stream-sg.larkoffice.com/space/api/box/stream/download/authcode/?code=MWJmNGYwMWQ2NDhjZWM2YzA0ZmMzNWI5MDI0NmM2MzFfYTFiZGU3YzM2ODViOGFiYTdiMmRjZmY2NzI1ZGJlN2FfSUQ6NzY0NzA5NTg4MTE3NDk3ODI2N18xNzgyMDQ2OTMxOjE3ODIxMzMzMzFfVjM)<br>![Image](https://internal-api-drive-stream-sg.larkoffice.com/space/api/box/stream/download/authcode/?code=NzU2YTgxNTc1ODNkY2VjNTY4ZGM0YTlmODdjNGFhMmJfNWI3MzQ1NjE0ODE5NTIzMjEwYzFlYTA1YTNjNmY3YzZfSUQ6NzY0NzA5NTg4MjAxNTI4MDg2NF8xNzgyMDQ2OTMwOjE3ODIxMzMzMzBfVjM)<br>|
|Structural transformation 结构化变换|Collage layout 拼贴埋点|![Image](https://internal-api-drive-stream-sg.larkoffice.com/space/api/box/stream/download/authcode/?code=MWYwYzQ0NjZlNDQ2ZmQ5MDMxMDM4OTM0NDQzNDFmZjhfYjA3MGRjZDE3MjI3NDkxMzhjMmI3YWU0YWQwNzk2MTZfSUQ6NzY0NzA5NTg4MzUzMTU4NzI5NF8xNzgyMDQ2OTMwOjE3ODIxMzMzMzBfVjM)|✅|![Image](https://internal-api-drive-stream-sg.larkoffice.com/space/api/box/stream/download/authcode/?code=ZGRjMTM5YWQ4OGI5Y2EzNmVkNGQxZDI1MDFjN2Y0ZjBfNTYzMDA2YzM1Njc2MWM0ZDBjNzRlMTExNzU2MzgyY2ZfSUQ6NzY0NzA5NTg4MTg2MjQzNDUzM18xNzgyMDQ2OTMxOjE3ODIxMzMzMzFfVjM)<br>![Image](https://internal-api-drive-stream-sg.larkoffice.com/space/api/box/stream/download/authcode/?code=ZjUyMmQ5ZjVkZjM1ODUxNTZlMDI4ZDU2YTY0ZDczNzdfYzUwZjA0M2MyOGM1Zjg5MzFiY2ZhYTFjMGM1NzlkNGFfSUQ6NzY0NzA5NTg3OTI0NDYxNTM4OF8xNzgyMDQ2OTMwOjE3ODIxMzMzMzBfVjM)|
||Poisson blend 泊松无缝融合|![Image](https://internal-api-drive-stream-sg.larkoffice.com/space/api/box/stream/download/authcode/?code=OWMwYjBhODY3ZDcwOGYyNWJjNTU4M2ZhODFiN2VjMWVfNTg0NWM2MmZlNGIzY2E0MWNlYTlkNTZiYTNkYTU5OTBfSUQ6NzY0NzA5NTg4MTE3NTA2MDE4N18xNzgyMDQ2OTMwOjE3ODIxMzMzMzBfVjM)|✅|![Image](https://internal-api-drive-stream-sg.larkoffice.com/space/api/box/stream/download/authcode/?code=MDFkYmZhZTA0MjQ4YWYzYjcyNGE2YzE3N2M2N2E1OTlfNmM5ZjIzOTI1MjRkZmE3ZGQzN2IyZTI2NDkwNmI4NjBfSUQ6NzY0NzA5NTg4MjkyMzQ0NTk3OV8xNzgyMDQ2OTMwOjE3ODIxMzMzMzBfVjM)<br>![Image](https://internal-api-drive-stream-sg.larkoffice.com/space/api/box/stream/download/authcode/?code=MTQ5OTU2NGQ0Y2U1ZTE5OWJiYzk3YWY2OTkzNjg1ZTJfMjdkZDllNmY3Mzc3MmYxMmNiNzUwMzM0MTY3NjkwMWZfSUQ6NzY0NzA5NTg3OTgxNTkwOTA5M18xNzgyMDQ2OTMwOjE3ODIxMzMzMzBfVjM)|
||Occlusion patch 遮挡埋点<br>|![Image](https://internal-api-drive-stream-sg.larkoffice.com/space/api/box/stream/download/authcode/?code=NmEzZTM4YWIxNzk4MDhmYjNmZjFiOTkxNjkxMzE2ODJfMDc5ZDFhY2VkZGY1ODNhMTM2MDg5OTBlMTM4ZDY2OWVfSUQ6NzY0NzA5NTg4MTAyNzk4MTAyNF8xNzgyMDQ2OTMwOjE3ODIxMzMzMzBfVjM)|✅|![Image](https://internal-api-drive-stream-sg.larkoffice.com/space/api/box/stream/download/authcode/?code=MGMxOTQyZTZmYjM5MDgxYzMxYzRhMWM3YzJlNmQ1MDJfNzQxNTZjOWUyNDBiNmZjMTE1N2ExOTMzODJmYjM5YjlfSUQ6NzY0NzA5NTg4MjQ3NDY3MTgzOV8xNzgyMDQ2OTMxOjE3ODIxMzMzMzFfVjM)<br>![Image](https://internal-api-drive-stream-sg.larkoffice.com/space/api/box/stream/download/authcode/?code=ZWU3YzhkMjMzNjM4ZDI0NjA4M2NmMDI4ZjVlZjEwMjZfNDhkNTRiYmViMWJhODk0MDJhNzUwYzQ5MjRlMjZkZTNfSUQ6NzY0NzA5NTg4MTQ2NDMzNjA5Nl8xNzgyMDQ2OTMwOjE3ODIxMzMzMzBfVjM)|
||Imaging degradation 图像降质|![Image](https://internal-api-drive-stream-sg.larkoffice.com/space/api/box/stream/download/authcode/?code=NjI3ZGM0NDFlNzIwMDY4MzJlNTM2NDQxOTExYzJmZDZfNDVjOGY4OWNlZDQ3M2JjNWRmNWFhZWU0OGJiMzU2NWJfSUQ6NzY0NzA5NTg4MDAzODQ1Mjk1N18xNzgyMDQ2OTMwOjE3ODIxMzMzMzBfVjM)|✅|![Image](https://internal-api-drive-stream-sg.larkoffice.com/space/api/box/stream/download/authcode/?code=YzQxNjRlYTY4NmI2NzhkOGEwM2E4ZDY5YTc2ZDQ2MTRfNWQwZGExOGU3ZjFlYjAwMTVmYjA0MzYzODI2NzI1MDRfSUQ6NzY0NzA5NTg4MjkyMzQ3ODc0N18xNzgyMDQ2OTMwOjE3ODIxMzMzMzBfVjM)<br>![Image](https://internal-api-drive-stream-sg.larkoffice.com/space/api/box/stream/download/authcode/?code=M2YzYzM0ODNkZGQ2ZGQzNzliZDZmMDJhYmY1ODlmYWJfMDIxMDdkZjZiMzNkMzU4ZWYwM2ZmYTI2YWRmZmJhNWZfSUQ6NzY0NzA5NTg4MjIwNjIzNjM4NF8xNzgyMDQ2OTMwOjE3ODIxMzMzMzBfVjM)|
||Geometric robustness 几何变换|![Image](https://internal-api-drive-stream-sg.larkoffice.com/space/api/box/stream/download/authcode/?code=MGNkMTAxMmQxZDY5ZWMzOGZiNGIyMmM2Yjg4YThmOWFfOGQ1ZTVhNDAyYjllZmE4YjdiY2IwODQxYmU2ZGYxZjRfSUQ6NzY0NzA5NTg4MjkyMzQyOTU5NV8xNzgyMDQ2OTMxOjE3ODIxMzMzMzFfVjM)|✅|![Image](https://internal-api-drive-stream-sg.larkoffice.com/space/api/box/stream/download/authcode/?code=YzUyYzI3NDAwNWYzZTY5YzA2YTRhOGM4ZTM2YTQxOWFfNzQ5NDc5YzcxOTg0MDRhMzU2YzNlODY2ZmY4NDM1MjhfSUQ6NzY0NzA5NTg4MzEzMzE5Mzk1NF8xNzgyMDQ2OTMwOjE3ODIxMzMzMzBfVjM)<br>![Image](https://internal-api-drive-stream-sg.larkoffice.com/space/api/box/stream/download/authcode/?code=N2NiOGQ1NDA0MTRhZGE5NjI0NWUzNDI2Mjc1ZWMxMzBfMjE2Mzc1ZGJiNmUyZTEyMTVmMjEwYjliNjlhMzNjZDBfSUQ6NzY0NzA5NTg4MjIwNjIwMzYxNl8xNzgyMDQ2OTMxOjE3ODIxMzMzMzFfVjM)|
||Low\-contrast text embedding 低对比文本嵌入|![Image](https://internal-api-drive-stream-sg.larkoffice.com/space/api/box/stream/download/authcode/?code=NzVlNmNkMmZlYmJmNTYwYTgyNzVmNTU0YTVlYjZjMzBfYmUwMDU3ZmNkODlmZGY3ZGQ1ZDJiYmFmOGEyYjA4N2VfSUQ6NzY0NzA5NTg3OTc4ODMwMjA0NF8xNzgyMDQ2OTMwOjE3ODIxMzMzMzBfVjM)|✅<br>|![Image](https://internal-api-drive-stream-sg.larkoffice.com/space/api/box/stream/download/authcode/?code=MjNhM2NmNTJhMzhlZjk3ZmJjNjAwNjEyNmFjODFmMDFfZDE3YmQzNGQ4MjViNjUwNjZlMDc2YzFiNWUzZjZlN2VfSUQ6NzY0NzA5NTg4MzA3NDUzOTIyOV8xNzgyMDQ2OTMxOjE3ODIxMzMzMzFfVjM)<br>![Image](https://internal-api-drive-stream-sg.larkoffice.com/space/api/box/stream/download/authcode/?code=ZTg5NDRiYzkzZmYxNDI5MWQzNmQwZGJlNzhkNDUwMWVfNjg2ZTVjNTMwMDA2NzEyZWIxZTdjMDg5Zjc4YTEwZmJfSUQ6NzY0NzA5NTg3OTc4ODIyMDEyNF8xNzgyMDQ2OTMwOjE3ODIxMzMzMzBfVjM)|
|Channel rendering 通道渲染攻击|AlphaDog|![Image](https://internal-api-drive-stream-sg.larkoffice.com/space/api/box/stream/download/authcode/?code=Mzg0NDJkOTdjZDM3MmYwODExNzg3M2Y2NmU2M2RhMjdfZGM0N2Y3NTgyNjY3MTA0ZTFiYTNlNjY1ODAwMDZkZDFfSUQ6NzY0NzA5NTg4MDA2NzgxMzA4N18xNzgyMDQ2OTMxOjE3ODIxMzMzMzFfVjM)|❌|![Image](https://internal-api-drive-stream-sg.larkoffice.com/space/api/box/stream/download/authcode/?code=ZTMzZWVjNDEwZDkwZjJiNWNiMDdlNWU1MzMzMzc4YjFfNmU0ZGE5MDJmMzJhN2YzMDdiYWY1N2E3YzJlNWVlNjBfSUQ6NzY0NzA5NTg3OTgxNTg5MjcwOV8xNzgyMDQ2OTMwOjE3ODIxMzMzMzBfVjM)|
||Phantom Tank 幻影坦克|![Image](https://internal-api-drive-stream-sg.larkoffice.com/space/api/box/stream/download/authcode/?code=ZDYxZjRhYjM1NWZkMDQxNTA2YmIzYzFkZGY2NWYxYTJfZmQ1NjFkNjE5Y2Y0MDYxMzU0YTlmYTY1NDllYjBjOTNfSUQ6NzY0NzA5NTg3ODc4MDAzMDY5MF8xNzgyMDQ2OTMxOjE3ODIxMzMzMzFfVjM)<br>![Image](https://internal-api-drive-stream-sg.larkoffice.com/space/api/box/stream/download/authcode/?code=ZmNhMDRkY2IyMDM0ZTNhYzliMDRmYmY1MjYyMmY0YTdfY2M4NmNiZDg0ZjcwZDc5MTg3ZDYyYmI0ZmExYmM3ZmVfSUQ6NzY0NzA5NTg4MTgwMTM4Nzc0Ml8xNzgyMDQ2OTMxOjE3ODIxMzMzMzFfVjM)<br>![Image](https://internal-api-drive-stream-sg.larkoffice.com/space/api/box/stream/download/authcode/?code=NzJmODg1NGJhMTcxZGY4NWNiYmI0MTgxMDRjMGQ2MzRfMTFiNmViNzc2YzRmMDg4YjFjM2QwNDcxZDRlZmYxMTVfSUQ6NzY0NzA5NTg4MTc2NjA5NjYwN18xNzgyMDQ2OTMxOjE3ODIxMzMzMzFfVjM)<br>|❌<br>|![Image](https://internal-api-drive-stream-sg.larkoffice.com/space/api/box/stream/download/authcode/?code=YThjN2I4NTc5ODNiNjkzMGQ0YmY2NGUwM2IyNDY5ZmVfZjQzYmM0MzdkMDBlMGY0NTNlMzA3MzFiMjdhZjllNjVfSUQ6NzY0NzA5NTg4MDAzODUzNDg3N18xNzgyMDQ2OTMwOjE3ODIxMzMzMzBfVjM)<br>![Image](https://internal-api-drive-stream-sg.larkoffice.com/space/api/box/stream/download/authcode/?code=NzkxNTI5MDRhZmI5Yjg0M2IzMTIxMmU0ZjRhY2I5MTZfMjcwMTljZTA0MzQ2NTZhYTliZGMwMWFiNjEyYTdkMjJfSUQ6NzY0NzA5NTg4MjA0ODczNjk5N18xNzgyMDQ2OTMxOjE3ODIxMzMzMzFfVjM)|

- attention无法注意到相关的视觉token：Text illusion 文本生图、Channel rendering 和比较强的MPCAttack

- 其他的都是能通过attention探测到，但VLM用prompt识别失败，可能后面针对这些类去做一些防御

### Q2：审核prompt的answer token能否检测到视觉token？

|大类|子方法|例子|探针token的Attention 能注意到相关视觉token|Answer token的 Attention 能注意到相关视觉token|可视化|
|---|---|---|---|---|---|
|Semantic camouflage 语义伪装|AdaIN 风格迁移|![Image](https://internal-api-drive-stream-sg.larkoffice.com/space/api/box/stream/download/authcode/?code=YTIyZmZkOGQ2NWZjZGFkNTU5ZDFjYTkzZmEwOTU4OGVfNTE0ODE5Y2FjZjQzN2ExMWRmMzkzNWMxMzk0OTBhNWNfSUQ6NzY0NzA5Nzg3NDczMzMyMTk1MV8xNzgyMDQ2OTMxOjE3ODIxMzMzMzFfVjM)<br>|✅|❌|![Image](https://internal-api-drive-stream-sg.larkoffice.com/space/api/box/stream/download/authcode/?code=N2Y5MWExMWIxODAxMTk0ZjgwOGVmN2Q2ODI5YzM5NWVfOTllYTQ5NWJjOGY4YzQ1NjA5MWVkODQ3ZjExNjBiYjdfSUQ6NzY0NzA5OTQxNDU0NDc4MTAyMl8xNzgyMDQ2OTMwOjE3ODIxMzMzMzBfVjM)<br>![Image](https://internal-api-drive-stream-sg.larkoffice.com/space/api/box/stream/download/authcode/?code=MWIxODU5MzQ5YjZiNmVlYTFjYmNlMWY0MjIzODhjZGNfMjc5NGMwMmE4MmExZTFhNjRkYzZmMDQ1NzQzNzQwMjFfSUQ6NzY0NzA5OTU2Mzg2MTQ5NTUxNV8xNzgyMDQ2OTMwOjE3ODIxMzMzMzBfVjM)|
||CAST 风格迁移<br>|![Image](https://internal-api-drive-stream-sg.larkoffice.com/space/api/box/stream/download/authcode/?code=OTViYTQ0ZTRiOWQ4NmQ0NTEwM2JlMWE3MzhlYjgxMWJfZmZjYzA3Mjk3OTZkZmVkOGFkYjNlNTkxNjZkNzY3N2FfSUQ6NzY0NzA5Nzg3NDczMzQwMzg3MV8xNzgyMDQ2OTMwOjE3ODIxMzMzMzBfVjM)|✅|❌|![Image](https://internal-api-drive-stream-sg.larkoffice.com/space/api/box/stream/download/authcode/?code=OWQxYzg3OGJjZWU1ODlhYjFjMDYwM2Y3ZDhjMTk5MWZfNTQ4ZmY2ODcxMzU2YjZjNTE2OGQxOGM0ZmQxOTY1NzFfSUQ6NzY0NzA5OTYxNTQwNTY1NzgyNl8xNzgyMDQ2OTMwOjE3ODIxMzMzMzBfVjM)<br>![Image](https://internal-api-drive-stream-sg.larkoffice.com/space/api/box/stream/download/authcode/?code=ZGMwOWIxZDA2NWExYzEwMTVjZTI4NzUwOTg5YTFhY2RfMjg0NmYzOWFhMzYyY2JhMDQ3YWUyNDc4ZTJlYzQ4MTNfSUQ6NzY0NzA5OTY1Mjc3NjY5MzQ3MF8xNzgyMDQ2OTMwOjE3ODIxMzMzMzBfVjM)|
||Text illusion 文本生图|![Image](https://internal-api-drive-stream-sg.larkoffice.com/space/api/box/stream/download/authcode/?code=M2UwMmNmMDQ5Njk0YzVmOGZiYzk1ZTU0NmYyZDdkMzdfMTUzYzM4YWQ0ZDQ3MWZlNTM2ZTNhMTdjMWJiY2E5NDJfSUQ6NzY0NzA5Nzg3NDg3ODQwMjI3MF8xNzgyMDQ2OTMxOjE3ODIxMzMzMzFfVjM)|❌|❌|![Image](https://internal-api-drive-stream-sg.larkoffice.com/space/api/box/stream/download/authcode/?code=Yzc1YTE1YmRmOGVjNzIwYTIzODQwY2JlNDYwYmJiNWNfNzAyNzExYTE4YjRlYjdmNjRlMzlhOGMwN2U4OThlZDRfSUQ6NzY0NzA5OTg2NDM2NDA3Njc2N18xNzgyMDQ2OTMxOjE3ODIxMzMzMzFfVjM)<br>![Image](https://internal-api-drive-stream-sg.larkoffice.com/space/api/box/stream/download/authcode/?code=OWE5YTIxZDEyMzhkMThkMmE1NGZkYzc4M2IwMGUwNTRfZDY3YzE5NWNlYzllMDUwNDY2ZjdkZTNkMDdlYzZmNGZfSUQ6NzY0NzA5OTg5MDQ4ODg4OTA1NF8xNzgyMDQ2OTMxOjE3ODIxMzMzMzFfVjM)|
||Resolution Attack, RA|![Image](https://internal-api-drive-stream-sg.larkoffice.com/space/api/box/stream/download/authcode/?code=ZjY3OGY2YzRiMDg3MDUwYmE1N2I5NWE3ZDdmMjU0Y2JfMjlhMzRiZDQ0NmNkYjhlNGUxMzkwYzhhZGFiNmVhNWFfSUQ6NzY0NzA5Nzg3ODIzMzUxNzc5Ml8xNzgyMDQ2OTMxOjE3ODIxMzMzMzFfVjM)|❌<br>|❌||
|Pixel representation 像素表征级攻击|AnyAttack|![Image](https://internal-api-drive-stream-sg.larkoffice.com/space/api/box/stream/download/authcode/?code=ZWViOTA5N2VlNDM2ZWQ2ZGE0NmU0NGM4ODI5ZjA1NDBfYmFhNTNjODIyN2Y2NzE2YzYzNGRlZjdhN2I1NmJkYjZfSUQ6NzY0NzA5Nzg3Njk1MDUzNTkwNF8xNzgyMDQ2OTMwOjE3ODIxMzMzMzBfVjM)|✅<br>|❌||
||MPCAttack|![Image](https://internal-api-drive-stream-sg.larkoffice.com/space/api/box/stream/download/authcode/?code=MDcyYmNmODI1YjdhY2YxMWMxNWZiM2Q1M2E2MDQxZTJfM2FmODYwNjg1MTliYjBhNWFhZTUzNDU2OWRkNzlkY2VfSUQ6NzY0NzA5Nzg3NDcyNzM3NDU2NV8xNzgyMDQ2OTMwOjE3ODIxMzMzMzBfVjM)|✅ / ❌<br>部分攻击特别强，直接探测不到了；大部分能探测到|❌|![Image](https://internal-api-drive-stream-sg.larkoffice.com/space/api/box/stream/download/authcode/?code=OTE5MzVlNTMzYjU2ZDNjN2U3YWJiZTZmODRiMTNlMjFfZTE2NmFhM2VjYzdjYTdhYjMyNTllZmM3ZTAxODJhMmZfSUQ6NzY0NzEwMTI1NTIxMDc0OTY2Ml8xNzgyMDQ2OTMxOjE3ODIxMzMzMzFfVjM)<br>![Image](https://internal-api-drive-stream-sg.larkoffice.com/space/api/box/stream/download/authcode/?code=YjI2ZWZmY2NmODkwZGEwMTcyMDg0ZGQxZTc2YmY5ZGJfNzk0MzE0N2YwYmM0NDY2NGRiZjBiMGMxOTM0OGJkOWNfSUQ6NzY0NzEwMTM0MzAwOTY4OTMxNF8xNzgyMDQ2OTMwOjE3ODIxMzMzMzBfVjM)|
|Structural transformation 结构化变换|Collage layout 拼贴埋点|![Image](https://internal-api-drive-stream-sg.larkoffice.com/space/api/box/stream/download/authcode/?code=OTIzMDU3ODRjZDFjNzFmMjdhNTRkYjA3NTkzZjY1N2NfODgwODJjNGVjZTc2N2M2MjU0OGNjNmJhNzNlMzM4NGRfSUQ6NzY0NzA5Nzg3OTE3NDg3Njg5MV8xNzgyMDQ2OTMwOjE3ODIxMzMzMzBfVjM)|✅|❌|![Image](https://internal-api-drive-stream-sg.larkoffice.com/space/api/box/stream/download/authcode/?code=NmIwZTkwYTJhMTQ0OGE1ODNlM2VjN2U4ODkxZWM5ZDJfMzQ1ZDJmZjJkNTA4ZjRhMjRmNjY1MjY5OThhMzQ5ZGZfSUQ6NzY0NzEwMTkxNzcwOTI3NDg0Nl8xNzgyMDQ2OTMwOjE3ODIxMzMzMzBfVjM)<br>![Image](https://internal-api-drive-stream-sg.larkoffice.com/space/api/box/stream/download/authcode/?code=OTMwNTZlYzliZWU4OGJjYjdkZmI2ZDRmYmI3OWIzNjZfNGIyYWI5NTlkM2Q2ZjdiZjlhOTgzZDg2OGI4MzBlZWJfSUQ6NzY0NzEwMjAzNTc2NzkyMjQwMF8xNzgyMDQ2OTMwOjE3ODIxMzMzMzBfVjM)|
||Poisson blend 泊松无缝融合|![Image](https://internal-api-drive-stream-sg.larkoffice.com/space/api/box/stream/download/authcode/?code=ZTg1NmY5MDJlZjFmZmQ0NmVmYWMzMWE1ODhmOWNiMDFfMTQxM2Y3NDNhM2E1ZWYzMmEwZTI2OTI5NmFlZjRiNTZfSUQ6NzY0NzA5Nzg3NDYzMTA1MzAyOV8xNzgyMDQ2OTMxOjE3ODIxMzMzMzFfVjM)|✅|❌|![Image](https://internal-api-drive-stream-sg.larkoffice.com/space/api/box/stream/download/authcode/?code=NTEwNmNlM2I2YTIwZWJhNzc3YmFkZGVlMDVjMThhOGFfYTRiMzg2MTMwNTVlNWE4NWUyN2UwNzI4ZWM5YjBjY2FfSUQ6NzY0NzEwMjQwMTgxMTM2OTY5Nl8xNzgyMDQ2OTMxOjE3ODIxMzMzMzFfVjM)<br>![Image](https://internal-api-drive-stream-sg.larkoffice.com/space/api/box/stream/download/authcode/?code=NWZlNjZmYzgyNTcxOWYzMjAxMjE5YmEwODI3MTY4NWZfOTg2NmM0OWY3NjljN2ZkOWIyOTkxYTA1OGRlMzFlODBfSUQ6NzY0NzEwMjU4MTk0NjE0MjQzMV8xNzgyMDQ2OTMxOjE3ODIxMzMzMzFfVjM)<br>|
||Occlusion patch 遮挡埋点<br>|![Image](https://internal-api-drive-stream-sg.larkoffice.com/space/api/box/stream/download/authcode/?code=MDcyOTlhNDcwZjQ0ZjY3MWIxNDlhYWJkNTU3ZTYyZmFfZWMyZDQ2NTM2MzhiODk5ODgxZmNkNjg0NjM4ZTY3MDNfSUQ6NzY0NzA5Nzg3OTE3NDkyNjA0M18xNzgyMDQ2OTMxOjE3ODIxMzMzMzFfVjM)|✅|❌|![Image](https://internal-api-drive-stream-sg.larkoffice.com/space/api/box/stream/download/authcode/?code=N2NkYzgyMDUyMjA4ZGE1ODE4ZGY5ZjFlMDAyODliM2RfZDM3MDY2MjVlZGJiMmQ0YmQ5NmFlOGE1MWQ0YmEzZTRfSUQ6NzY0NzEwMjE2MjE2NjE0MDYzNl8xNzgyMDQ2OTMxOjE3ODIxMzMzMzFfVjM)<br>![Image](https://internal-api-drive-stream-sg.larkoffice.com/space/api/box/stream/download/authcode/?code=YWExMWE0ZWJiYWIyY2ZlNThmNDg3MGU2YjNjNjgwNzhfZGRmOWEwMGVhMDY3MmNkZTk5OWYzZDIxODMxYjBkMGJfSUQ6NzY0NzEwMjI2OTUzOTgxNTEzM18xNzgyMDQ2OTMwOjE3ODIxMzMzMzBfVjM)|
||Imaging degradation 图像降质|![Image](https://internal-api-drive-stream-sg.larkoffice.com/space/api/box/stream/download/authcode/?code=YWVmNGI0NjQ1ZTA3YmFiNzlhNGJiMDExMjAyNzA1NjhfNjAwMjMzMzQyNGE5ZWYzMWI0M2ExOTEzMDdiOTIyYTZfSUQ6NzY0NzA5Nzg3NDY2NDQ0MzYxNF8xNzgyMDQ2OTMxOjE3ODIxMzMzMzFfVjM)|✅|❌|![Image](https://internal-api-drive-stream-sg.larkoffice.com/space/api/box/stream/download/authcode/?code=YzYyYmU5NjdhMGU3NzQ4NDFlODBmZDUzNWE4N2YyNDFfOWRhZGJkYzk1ZmY0NzNmZDQ0MDgyMzI0YzRkYjViODFfSUQ6NzY0NzEwMjkxMjkxMDAzNjY5OV8xNzgyMDQ2OTMwOjE3ODIxMzMzMzBfVjM)<br>![Image](https://internal-api-drive-stream-sg.larkoffice.com/space/api/box/stream/download/authcode/?code=M2FiMTE0MmIxNzFlMDg4YTI3ZWQzZjBkNWEyNTZlY2JfMTQxMzM3NTZkNjA2MDBmNmRkNDg4YWY2ZThhMGFjNjZfSUQ6NzY0NzEwMzA2MDAyNjg4NzkwMl8xNzgyMDQ2OTMwOjE3ODIxMzMzMzBfVjM)|
||Geometric robustness 几何变换|![Image](https://internal-api-drive-stream-sg.larkoffice.com/space/api/box/stream/download/authcode/?code=MWFjYzIwMGJiMWJlMmYwYjVlMTAxMDMxM2FhM2FmM2NfNWY2MjliM2VhMjgxM2ZlM2IzZDU3NjFhZTM5ZTcwODlfSUQ6NzY0NzA5Nzg3NjM5ODMxMzE4Ml8xNzgyMDQ2OTMwOjE3ODIxMzMzMzBfVjM)|✅|❌|![Image](https://internal-api-drive-stream-sg.larkoffice.com/space/api/box/stream/download/authcode/?code=NzJlMjBhZjY4Mzc0ZmQ5OWIyZThmNDc0YjJlYjM4NmFfOGVhNDI0ZGJlNGZiODQyMjFhNTlkNjFmNzE2YjZjNzhfSUQ6NzY0NzEwMzE0NDgwMzYxODUyM18xNzgyMDQ2OTMxOjE3ODIxMzMzMzFfVjM)<br>![Image](https://internal-api-drive-stream-sg.larkoffice.com/space/api/box/stream/download/authcode/?code=ZjNlY2Q3OGViZDY3NmEwZWRjMGY2OTk1Njc0NTIxMWJfMmQ4Y2RmNjU0MDg0N2M0NTliZmQ1NjQzZjE5ZjVhOTRfSUQ6NzY0NzEwMzE5MTkwMTIxMjM4Ml8xNzgyMDQ2OTMxOjE3ODIxMzMzMzFfVjM)<br>|
||Low\-contrast text embedding 低对比文本嵌入|![Image](https://internal-api-drive-stream-sg.larkoffice.com/space/api/box/stream/download/authcode/?code=NTc4ZDNkNjhkMTAwNTNjNjg3NjMzMzBkMGY4ZDNlNzhfYWM2NjRmZjQ5OWUxZDJmMDQxZTJjMTk2YjZkNzQ2ZjFfSUQ6NzY0NzA5Nzg3NTA3NDY2NjIwOF8xNzgyMDQ2OTMxOjE3ODIxMzMzMzFfVjM)|✅<br>|❌|![Image](https://internal-api-drive-stream-sg.larkoffice.com/space/api/box/stream/download/authcode/?code=YmY3YjI2YjQ5NGE3ODQwOTRkOTQwNWNiNjE2YjczY2RfNDRiNjVlOTcyNjBkYTNkODFjMjI3MDhmYjVkZGQwYjZfSUQ6NzY0NzEwMzUzNzcxNzI4NDU3NF8xNzgyMDQ2OTMxOjE3ODIxMzMzMzFfVjM)<br>![Image](https://internal-api-drive-stream-sg.larkoffice.com/space/api/box/stream/download/authcode/?code=MjNjYjM5MWU2N2Q1OWU2NTU5OGUzZTBlNTFiYmIwNWRfOTg4NDIzNTBjZDQ2ZTU5NzA3ZDVhNjE0YWQ2MTZmNDBfSUQ6NzY0NzEwMzY3MDM2NjUyMzEwMl8xNzgyMDQ2OTMwOjE3ODIxMzMzMzBfVjM)|
|Channel rendering 通道渲染攻击|AlphaDog|![Image](https://internal-api-drive-stream-sg.larkoffice.com/space/api/box/stream/download/authcode/?code=MDYyMjE1MzdkMzZmN2Y1NjM3ZTEzZWZmNmVhNjFlZDNfMWFkZWFmNzQ0ODRmOGJhODcwMjgxMGQ1MzZkMDkzMTJfSUQ6NzY0NzA5Nzg3NjM5ODMyOTU2Nl8xNzgyMDQ2OTMxOjE3ODIxMzMzMzFfVjM)|❌|❌|\-|
||Phantom Tank 幻影坦克|![Image](https://internal-api-drive-stream-sg.larkoffice.com/space/api/box/stream/download/authcode/?code=YzJkODI3NTM1MTJjYzMyYTlmZTViYjAyOWRiNzQwOWNfMWE2ODdjOWRiMDAwNmE4ODM2N2NkY2M0ZmVmOTg0N2RfSUQ6NzY0NzA5Nzg3NjU0NzYwNDE5N18xNzgyMDQ2OTMwOjE3ODIxMzMzMzBfVjM)<br>![Image](https://internal-api-drive-stream-sg.larkoffice.com/space/api/box/stream/download/authcode/?code=Nzc0MWQ4MGY1ZDQwZWVlYWQwMzkzOTEyMmY4YTBmN2ZfMjk4MGVmODI0NTk4MzY0ODBmMjI4Njc2NjE1ZWIxYjlfSUQ6NzY0NzA5Nzg3NDU4NDcwMjY4OF8xNzgyMDQ2OTMxOjE3ODIxMzMzMzFfVjM)<br>![Image](https://internal-api-drive-stream-sg.larkoffice.com/space/api/box/stream/download/authcode/?code=ODFhOWU0MThjNjE0NmRkYmJjNjUxZGJhMTNmOWJmZDlfMzE5ODc2MmQ3MGVkNWVmZDMxNzAxZWUxMzU2YTMxNzNfSUQ6NzY0NzA5Nzg3NjM5ODQ0NDI1NF8xNzgyMDQ2OTMxOjE3ODIxMzMzMzFfVjM)|✅ / ❌<br>部分特别强的会探测不到|❌|![Image](https://internal-api-drive-stream-sg.larkoffice.com/space/api/box/stream/download/authcode/?code=MTUwY2ZmNmI5NjJkZTkzNDdmYzI0MTA5YzliZWIzNjRfMjExZjAwZjIwOGEzYTcxZjQ5ZWE5NGJmNGM3NTRkZGRfSUQ6NzY0NzEwMzk3NTAwMDExNjk2MF8xNzgyMDQ2OTMxOjE3ODIxMzMzMzFfVjM)<br>![Image](https://internal-api-drive-stream-sg.larkoffice.com/space/api/box/stream/download/authcode/?code=YTczYThlN2Q2YzY0NWRhMDNmMDEzOTc2NWEyZTIxZTBfZDczYjg0ZWY4NmMzZDgxMWEyZGFhYWVhMTAzYTNkMGVfSUQ6NzY0NzEwNDE2Njc4ODc3OTc0MV8xNzgyMDQ2OTMwOjE3ODIxMzMzMzBfVjM)|

- 一些caught的例子。

![Image](https://internal-api-drive-stream-sg.larkoffice.com/space/api/box/stream/download/authcode/?code=ODE4NmE4N2FiZTkxMmE2OWFhOWZkNGI3ZDI1OGQyZjBfY2JhZDE5MTUxMjJhY2RlMTQ3ZmFkOTg1MTUyYjNjNzhfSUQ6NzY0NzM2MTI3MDEzOTgwMTMxMV8xNzgyMDQ2OTMxOjE3ODIxMzMzMzFfVjM)

![Image](https://internal-api-drive-stream-sg.larkoffice.com/space/api/box/stream/download/authcode/?code=Y2I2MGE1MWQzNjU2NTYxMTY5YWRmYjE4N2JjM2I3ZTVfZjE4ZjFjYzA4MjVkMDc1YTQzMDczNjlhMDNjNGE5YjhfSUQ6NzY0NzM2MTI5Nzg1NTg5MzIxMl8xNzgyMDQ2OTMxOjE3ODIxMzMzMzFfVjM)



### Q3：不同模型的attention可视化

- 采用3个不同模型：InternVL3\-8B、R\-4B、Sail\-VL\-1\.6\-8B

|模型|核心定位|视觉编码器|LLM 骨干|视觉连接方式|分辨率/token策略|
|---|---|---|---|---|---|
|**InternVL3\-8B**|通用强 VLM，偏全能|InternViT 系列|Qwen2\.5\-7B|ViT\-MLP\-LLM|448 tile \+ dynamic resolution \+ pixel unshuffle|
|**SAIL\-VL\-1\.6\-8B**|数据工程驱动的小/中型高性能 VLM|SAILViT\-Huge|Qwen2\.5\-7B<br>|2\-layer MLP|448×448×N \+ 2×2 token merge|
|**Qwen3\-VL\-4B**|新一代原生多模态，长上下文/视频/空间/OCR|Qwen3\-VL vision encoder|Qwen3\-VL text 4B|原生 VL 架构<br>|MRoPE \+ DeepStack \+ spatial merge|

- InternVL3和Qwen3\-VL\-4B的表现基本类似；SAIL\-VL\-1\.6\-8B的注意力提取结果不直观，分散在各个地方

    - InternVL3

![Image](https://internal-api-drive-stream-sg.larkoffice.com/space/api/box/stream/download/authcode/?code=ZTg0Mjk2NWY3ODhkYjhiNDZlMWNmYWM5NGZlZTZkYmVfYjc4ZTViYzFhNDkyNDYyZGI5ZjJhODUyOGEwOWFlOTJfSUQ6NzY1MDM3NzIwODY5NDk4NDQyMV8xNzgyMDQ2OTMwOjE3ODIxMzMzMzBfVjM)

![Image](https://internal-api-drive-stream-sg.larkoffice.com/space/api/box/stream/download/authcode/?code=YmQxMmVhODM1ZDJkNGM0ZTc5N2FjN2U1MDVlODM2MzBfNGY5MDRkMDYzNzJkMjM3Y2RlNzdhNjU4NThhN2MyNjdfSUQ6NzY1MDM3NzAzNTYyOTYxMjc2N18xNzgyMDQ2OTMwOjE3ODIxMzMzMzBfVjM)

    - SAIL\-VL\-1\.6\-8B

![Image](https://internal-api-drive-stream-sg.larkoffice.com/space/api/box/stream/download/authcode/?code=NWMyOGIzYTU2ZTc0OTM1YmEyYTViY2UzZDRkY2YzZGVfYWUzMWM4ZjY3YWViZWRhMTQ1YjVmYjg3NzBiOTczZmVfSUQ6NzY1MDM3NjcxNTkyMzA5OTM1OF8xNzgyMDQ2OTMwOjE3ODIxMzMzMzBfVjM)

![Image](https://internal-api-drive-stream-sg.larkoffice.com/space/api/box/stream/download/authcode/?code=NzlhMjBhNjFiNTljYjJkM2NkM2Q4NmUzZGY2ZmViYjRfODkwMWZlYTgxMzg1NTk1NGQyMzc5NjZjZTY4NzljMmZfSUQ6NzY1MDM3Njc1MTU5OTg0OTE4Nl8xNzgyMDQ2OTMxOjE3ODIxMzMzMzFfVjM)

可能是提取代码不适配模型架构的问题

### Q4：不同层

- 当前的attention探针，是选择的中间层（对于qwen3\-vl\-4b，是16\-21层聚合），调研到的论文说是因为多模态大模型存在明显的层级表征分工：浅层更偏向模态内的局部感知与低级视觉表征保留，中间层承担视觉—语言对齐与跨模态证据绑定，深层则逐渐转向高层语义抽象、指令遵循与答案生成。因此，中间层的注意力分布通常更适合作为视觉证据定位和跨模态 grounding 的探针。

![Image](https://internal-api-drive-stream-sg.larkoffice.com/space/api/box/stream/download/authcode/?code=NzBmMmY0M2QwMjliMDhlNjYzMzY3NTBiZmFlZjFjYTlfNWMwYjM2YzhjNjYxMTdmMjkwMmVjZmQzZmY2MGQ3OWRfSUQ6NzY0OTk3MzcwOTQ2NjE0NDQ3OF8xNzgyMDQ2OTMwOjE3ODIxMzMzMzBfVjM)

![Image](https://internal-api-drive-stream-sg.larkoffice.com/space/api/box/stream/download/authcode/?code=NGY3ZGZmMzYyMDNlMmE2MGQ3ODNiMGNhMWYwZjM2ZTFfOWU0ZDM0NmU1NGY0YzEyOWU5M2NjYjBlMTRhOTRiZDZfSUQ6NzY1MDM1MDAxNjgwMzk1MDMwM18xNzgyMDQ2OTMxOjE3ODIxMzMzMzFfVjM)

## 防御

### 针对视觉头看不见：AI illusion类

**Takeaway：** AI Illusion 攻击之所以能绕过 VLM 审核，本质是"高分辨率视觉冗余淹没了全局结构信息"。有效的防御要先让模型"看得到"——通过多尺度预处理恢复被遮蔽的视觉证据，再交由 VLM 做语义判断。

- 该类攻击asr高，现有模型防御不足，且无法探测到对应的视觉token。

#### [Hateful illusion](https://arxiv.org/pdf/2507.22617)

**本文精华：** VLM 识别 AI Illusion 的失败不是"模型不够智能"，而是架构层面的感知瓶颈。核心公式：`隐藏信息不可见 = 高分辨率视觉冗余 + ViT 静态编码 + 训练数据缺失`。类比日常工作：你让一个只看过正常合同的审计员去发现隐写术伪造的假合同，他再聪明也看不出来。

![Image](https://internal-api-drive-stream-sg.larkoffice.com/space/api/box/stream/download/authcode/?code=MWUxY2FlNDI0NmQwYmFmZWZhZTUxMTIyOGFiZmFhZmNfNzAzODlmN2Y3YjBmOTZlZjYzZDNjZDZkNDM5MWYyYWFfSUQ6NzY0NzE0OTg5OTY2MTg0MDA5M18xNzgyMDQ2OTMwOjE3ODIxMzMzMzBfVjM)

![Image](https://internal-api-drive-stream-sg.larkoffice.com/space/api/box/stream/download/authcode/?code=MTYyZmQ2NDRlNzk2MDE3ZDJlZTMyNmRmMjA5ODQ1YTRfNmNlMTdlY2YyZThhMTgyMjI2ZmE3MDJmMjE4YmMwMDZfSUQ6NzY0NzE1MDI0OTc3NzIyMTM0NF8xNzgyMDQ2OTMxOjE3ODIxMzMzMzFfVjM)



#### [SemVink](https://arxiv.org/pdf/2506.02803)

- 现有 VLM 不是“不懂隐藏内容是什么”，而是缺少人类那种 zoom\-out / squint / 多尺度观察能力；它们在高分辨率输入下被局部纹理和冗余 visual tokens 干扰，导致看不到全局隐藏结构。简单把图缩小到 32–128px（太多像素看不到隐藏，太小了又会导致细节合并），反而能让模型看到隐藏内容。

- 观察：VLM 识别 AI illusion / hidden content 的失败，很大程度上不是因为隐藏内容不存在，而是因为高分辨率图像产生了大量冗余、重复的 image tokens，这些 token 会淹没真正有用的全局结构信息。

![Image](https://internal-api-drive-stream-sg.larkoffice.com/space/api/box/stream/download/authcode/?code=N2ZlY2FjZmI0NTUyMmQwN2IzYTMyNGQ0MWEwOWViM2ZfMDhiYjI3ZTM0ZjhhOWQ4Mzc1NDNlMGM2NzNlMWZjMWFfSUQ6NzY0NzE5NDYwNTEzOTEzNjIyMV8xNzgyMDQ2OTMxOjE3ODIxMzMzMzFfVjM)

#### [Smugglebench](https://arxiv.org/pdf/2604.06950) 

涵盖多种文字类型的对抗攻击，其中能大批量生成的是低分辨率和ai illusion类

![Image](https://internal-api-drive-stream-sg.larkoffice.com/space/api/box/stream/download/authcode/?code=ZjI2ODZhNjMxMjhjZmU2M2I2N2I3Njc3Y2Y2YWQyOGFfYmM0M2NiOWI4MGU5ZjhjYjUyMTAxNDA4OTJmMDBjZjVfSUQ6NzY0NzE5NTE3NTE2NDM2NjU1OF8xNzgyMDQ2OTMxOjE3ODIxMzMzMzFfVjM)

- CoT：在 Qwen3\-VL\-235B 上测试 CoT vs Standard Prompt：

|指标|Standard Prompt|CoT Prompt|变化|
|---|---|---|---|
|ASR（攻击成功率）↓|57\.9%|50\.7%|\-7\.2 pts|
|TER（文本提取率）↑|15\.5%|17\.2%|\+1\.7 pts|
|FPR（误报率）↓|1\.5%|4\.2%（×3）|\+2\.7 pts|
|AI Illusion 子类 ASR|99\.0%|99\.0%（无改善）|0|

**结论：** CoT 对 AI Illusion 子类几乎完全无效，同时引入了 over\-sensitivity 问题。

- SFT 全参数微调

在 Qwen2\.5\-VL\-7B 上用 1700 对抗样本 \+ 1700 安全样本进行 Full Fine\-tuning：

|指标|微调前|微调后|变化|
|---|---|---|---|
|ASR ↓|87\.0%|49\.5%|**\-37\.5 pts**|
|FPR ↓|1\.6%|8\.2%（×5）|\+6\.6 pts|

ASR 下降 37\.5 个点看似巨大，但 FPR 从 1\.6% 涨到 8\.2%（5 倍），在大规模审核场景中意味着海量误伤。且 train/test 同分布，泛化性存疑。

#### SMSP

- 涵盖AI illusion和Low\-contrast text embedding两类

- 从attention角度分析，发现对于illusion / low contrast图像，模型的attention权重更多集中在背景；通过一些预处理，模型的注意力重新被拉回目标text。

    - 高频信息过滤和rescale

![Image](https://internal-api-drive-stream-sg.larkoffice.com/space/api/box/stream/download/authcode/?code=ZDE2YTM1Y2M4ZjEzZDI4ZDc2MDJhNzNkZjZlOTVjZjFfODIxMzJhNDRlYmRkMGVmMGFlYmY5OWQ2MDFkNGU4Y2VfSUQ6NzY0NzE5Nzc2MzEyMTIyMTM0Ml8xNzgyMDQ2OTMxOjE3ODIxMzMzMzFfVjM)

![Image](https://internal-api-drive-stream-sg.larkoffice.com/space/api/box/stream/download/authcode/?code=OTM3MTRkMjVkY2RiYmI1MWJjYjhlNzY2YmFjZGM5MWZfMGJlNGQ2YzhkZmE0NzI1YWE1NjcwYWFlODY4MTJjOWFfSUQ6NzY0NzE5Nzc2NDA0ODE2MjUzM18xNzgyMDQ2OTMxOjE3ODIxMzMzMzFfVjM)

- 方法：Multi\-Scale Strategy 多尺度策略。单一处理强度不够，因为隐藏 pattern 有大有小。

    

![Image](https://internal-api-drive-stream-sg.larkoffice.com/space/api/box/stream/download/authcode/?code=NGMxOGE5YWRhN2U2NWE5NDUzNmY0M2M4NzRkNjAwZjNfNmZkNWIyN2QzYzhjYmJjZmM2YzZhMzUxMmVjNTk3OTRfSUQ6NzY0NzE5Nzc2NDUwNTM0MTY2OV8xNzgyMDQ2OTMxOjE3ODIxMzMzMzFfVjM)

    - SMSP 不只生成一个处理图，而是生成多个不同处理强度的 variants。参数按照几何级数插值，从强处理到弱处理排列。最后把这些处理图和原图**一起**作为一个 tuple 输入给 MLLM。

        ```Plain Text
        输入图像 I
        ↓
        Perception Module(强) → I1
        Perception Module(中强) → I2
        Perception Module(中弱) → I3
        Perception Module(弱) → I4
        原图 I0 保留
        ↓
        [I0, I1, I2, I3, I4] 一起送给 VLM
        ```

        - 保留原图是为了不损害模型在普通 OOD/VQA 任务上的原有能力，因为强处理可能会丢掉正常视觉细节。

![Image](https://internal-api-drive-stream-sg.larkoffice.com/space/api/box/stream/download/authcode/?code=YjBiZWJlMjM0NTI1ZTE3MzdkYTYxNGZlYjIzNGYwNDhfZjJkMzhkOWZmZDhkNjNiODhhZmVlNzUwYTBmMzRiZGJfSUQ6NzY0NzE5NzY4NzE5NTkzMDMzNF8xNzgyMDQ2OTMwOjE3ODIxMzMzMzBfVjM)

#### 我的实验：preprocessing

- 我在 Qwen3\-VL\-4B 和 8B 上进行了多种 prompting 策略的对比实验：

|策略|p\_detected|p\_correct@50|p\_correct@90|
|---|---|---|---|
|Direct Prompting（直接问）|0\.24|0\.01|0\.00|
|Follow\-up Hint（给文本提示）|0\.06|0\.01|0\.00|
|Prompt Engineering|0\.21|0\.01|0\.00|
|**Zoom\-out 32px**|**0\.75**|**0\.50**|**0\.11**|
|Gaussian Blur \+ Hist EQ|**0\.95**|**0\.7**|**0\.25**|
|**SMSP FFT \(l=0\.024, s=128\)**|**0\.80**|**0\.65**|**0\.21**|

![Image](https://internal-api-drive-stream-sg.larkoffice.com/space/api/box/stream/download/authcode/?code=ODRiNDM4ZGVmZGE5NzZmNDJkNmY4NDgxYTc3ZWUyYTJfYTA2MjUxNzIxMDEzYTIxM2FkZThlNGVhMTQ2M2M1MzNfSUQ6NzY0NzE5NjA3NzA3MzkxMTUxOF8xNzgyMDQ2OTMxOjE3ODIxMzMzMzFfVjM)

![Image](https://internal-api-drive-stream-sg.larkoffice.com/space/api/box/stream/download/authcode/?code=N2Y0YTY5YTkxNjZjZGQ4NmI2NWQ1YWM3YWFiNzI1NWVfZWM4MGZmMDU0MzNjNTI3ZDgyNjU1YjFkOTMyMTBkMmNfSUQ6NzY0NzE5NjIxMDI1MTU1MDQzNF8xNzgyMDQ2OTMwOjE3ODIxMzMzMzBfVjM)

- **关键发现：** 无论怎么优化 Prompt，模型对 Illusion 的识别率始终极低（detected \< 25%）。但简单地把图片缩小或做频域处理，识别率就能从 24% 飙升到 80%。这有力证明了：**瓶颈在视觉编码，而非语言推理。**

```Python
# 图像预处理

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageFilter, ImageOps

TAKEAWAY_IMAGE_PATH = "/path/to/image.jpg"  # 改成图片路径 


# 参数
BEST_SMSP_RADIUS_RATIO = 0.02449
BEST_SMSP_RESIZE_SHORT_SIDE = 128
BEST_GAUSSIAN_BLUR_RADIUS = 2.0
BEST_GAUSSIAN_RESIZE_TO = (128, 128)

# resize
def resize_short_side(img: Image.Image, short_side: int) -> Image.Image:
    w, h = img.size
    if min(w, h) == 0:
        return img
    if w <= h:
        new_w = short_side
        new_h = max(1, round(h * short_side / w))
    else:
        new_h = short_side
        new_w = max(1, round(w * short_side / h))
    return img.resize((new_w, new_h), Image.BICUBIC)

# blur then histogram equalization
def apply_blur_then_equalize(
    img: Image.Image,
    resize_to: tuple[int, int] = BEST_GAUSSIAN_RESIZE_TO,
    blur_radius: float = BEST_GAUSSIAN_BLUR_RADIUS,
) -> Image.Image:
    resized = img.resize(resize_to, Image.BICUBIC)
    blurred = resized.filter(ImageFilter.GaussianBlur(radius=blur_radius))
    return ImageOps.equalize(blurred)


# fft lowpass filter gray image
def fft_lowpass_gray(img: Image.Image, radius_ratio: float) -> Image.Image:
    gray = np.asarray(img.convert('L'), dtype=np.float32)
    h, w = gray.shape
    f = np.fft.fft2(gray)
    f_shift = np.fft.fftshift(f)
    cy, cx = h // 2, w // 2
    yy, xx = np.ogrid[:h, :w]
    radius = max(1, int(min(h, w) * float(radius_ratio)))
    mask = ((yy - cy) ** 2 + (xx - cx) ** 2) <= radius ** 2
    filtered = f_shift * mask
    inv = np.fft.ifft2(np.fft.ifftshift(filtered))
    mag = np.abs(inv)
    mag = (mag - mag.min()) / (mag.max() - mag.min() + 1e-8) * 255.0
    return Image.fromarray(mag.astype(np.uint8)).convert('RGB')


def apply_smsp_high_frequency_filter(
    img: Image.Image,
    radius_ratio: float = BEST_SMSP_RADIUS_RATIO,
    resize_short_side_to: int = BEST_SMSP_RESIZE_SHORT_SIDE,
    canvas_color: int = 255,
) -> Image.Image:
    base = fft_lowpass_gray(img, radius_ratio=radius_ratio)
    small = resize_short_side(base, int(resize_short_side_to))
    canvas = Image.new('RGB', img.size, (canvas_color, canvas_color, canvas_color))
    x = max(0, (canvas.size[0] - small.size[0]) // 2)
    y = max(0, (canvas.size[1] - small.size[1]) // 2)
    canvas.paste(small, (x, y))
    return canvas


def main(image_path=None):
    image_path = Path(image_path) 
    orig_img = Image.open(image_path).convert('RGB')
    
    # 【method1】 blur_hist_eq
    blur_hist_eq_img = apply_blur_then_equalize(orig_img)   
    
    # 【method2】SMSP 
    smsp_img = apply_smsp_high_frequency_filter(
        orig_img,
        radius_ratio=BEST_SMSP_RADIUS_RATIO,
        resize_short_side_to=BEST_SMSP_RESIZE_SHORT_SIDE,
    )

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    panels = [
        ('original', orig_img),
        ('gaussian_blur_hist_eq', blur_hist_eq_img),
        (f'smsp_fft_l{BEST_SMSP_RADIUS_RATIO}_s{BEST_SMSP_RESIZE_SHORT_SIDE}', smsp_img),
    ]
    for ax, (title, panel_img) in zip(axes, panels):
        ax.imshow(panel_img)
        ax.set_title(title)
        ax.axis('off')
    plt.suptitle(str(image_path))
    plt.tight_layout()
    plt.show()

    return {
        'image_path': image_path,
        'original': orig_img,
        'gaussian_blur_hist_eq': blur_hist_eq_img,
        'smsp_fft_l0.02449_s128': smsp_img,
    }


outputs = main(TAKEAWAY_IMAGE_PATH)
outputs.keys()
 
```

#### 我的实验：隐藏结构恢复/逆向

**Takeaway：** 不依赖 VLM 直接检测，而是训练一个轻量分割模型（U\-Net/SegFormer），将 Illusion 图中隐藏的二值化文字结构"恢复"出来，再交给 VLM 做文字识别和风险判断。

最终版本 SegFormer \+ 多尺度 TTA 在真实对抗数据上 CDR@50 达到 93\.7%。

##### 6\.1 核心思路

将问题形式化为**像素级二值分割**：

$f_\theta(I_{illusion}) = M_{binary} \quad \text{其中} \quad I_{illusion} \in \mathbb{R}^{H \times W \times 3}, \quad M_{binary} \in \{0,1\}^{H \times W}$

类比：这就像一个"逆向工程"过程——我们知道 Illusion 图是从 binary mask 生成的（ControlNet），那就训练一个模型来做**逆推**：从生成后的图反推出原始 mask。

##### 6\.2 数据构造（零标注成本）

**关键优势：** 训练数据不需要人工标注——生成 Illusion 图时本身就有原始 binary condition，天然构成训练对。

数据构造流程：

1. **渲染 Binary Mask**：文字 → 512×512 二值图

2. **Mask → Illusion**：Binary Mask 作为 ControlNet 条件，生成自然图像

3. **获得训练对**：`illusion_0001.png ↔ mask_0001.png`

关键改进点（V2 数据）：

- **掩码颜色反转**：白底黑字 ↔ 黑底白字随机切换（最关键改进）

- **几何增强**：随机旋转 ±30°、仿射、透视扭曲

- **字体库扩充**：覆盖无衬线体/粗体/斜体/等宽/手写/哥特体

- **Hard Negative**：自然河流、树枝、迷宫、斑马线、普通文字图等

##### 6\.3 模型演进

|版本|模型|关键改动|真实数据 CDR@50|
|---|---|---|---|
|V0|U\-Net \(ResNet34\)|Baseline，白底黑字训练|0\.6%（几乎失败）分析原因是训练数据只有白底黑字没有黑底白字|
|V1|U\-Net \(ResNet34\)|黑白反转 \+ Soft Mask \(GaussianBlur σ=1\.5\)|59\.5% \~ 89\.0%|
|V2|**SegFormer\-B1**|全局 Attention \+ 多尺度 TTA|**93\.7%**|

##### 6\.4 为什么 SegFormer 优于 U\-Net？

**U\-Net 的特点：**

- Encoder\-Decoder \+ Skip Connection

- 优势：同时利用低层纹理和高层语义

- 局限：**局部边缘恢复强，但全局结构感知弱**

- 适合：规整的隐藏字（训练分布内）

**SegFormer 的特点：**

- **MiT Encoder** \(4\-stage 多尺度特征\) \+ MLP Decoder

- 优势：**全局 Attention 天然擅长长距离依赖**

- 额外收益：多尺度 TTA 进一步提升鲁棒性

- 适合：装饰性、材质化、连笔等复杂真实数据

##### 6\.5 Loss 设计

$L = \text{BCE}(p, y) + \text{Dice}(p, y)$

- **BCE Loss**：关心每个像素分类正确性

- **Dice Loss**：关心整体 mask 重叠度，有效缓解前景少/背景多的类别不平衡

配合 Soft GT（σ=1\.5 Gaussian Blur），容忍 Diffusion 弱引导导致的轻微偏移；训练时 size 在 \[384, 512, 640, 768\] 中随机抽，crop 也随机，让 encoder 见过不同尺度后更鲁棒；推理时也用tta

##### 6\.6 结果

|model|dataset|n|CDR@50|CDR@80|CDR@100|ratio\_mean|
|---|---|---|---|---|---|---|
|v0\_unet|synth\_test\_v0|105|0\.9619|0\.8667|0\.619|0\.9188|
|v0\_unet|synth\_test\_v1|105|0\.4381|0\.2667|0\.1333|0\.4306|
|v0\_unet|real\_kb2\.fun|100|0|0|0|0\.0779|
|v0\_unet|real\_keybet\.com|158|0\.0063|0|0|0\.096|

|model|dataset|n|CDR@50|CDR@80|CDR@100|ratio\_mean|
|---|---|---|---|---|---|---|
|v1\_unet|synth\_test\_v0|105|0\.9429|0\.7143|0\.3905|0\.8504|
|v1\_unet|synth\_test\_v1|105|0\.8762|0\.6857|0\.4|0\.8067|
|v1\_unet|real\_kb2\.fun|100|0\.89|0\.25|0\.19|0\.7006|
|v1\_unet|real\_keybet\.com|158|0\.5949|0\.4747|0\.3038|0\.6103|

|model|dataset|n|CDR@50|CDR@80|CDR@100|ratio\_mean|
|---|---|---|---|---|---|---|
|v2\_segformer\+tta|synth\_test\_v0|105|0\.9238|0\.6952|0\.4571|0\.8314|
|v2\_segformer\+tta|synth\_test\_v1|105|0\.8571|0\.6381|0\.3619|0\.7718|
|v2\_segformer\+tta|real\_kb2\.fun|100|0\.93|0\.93|0\.93|0\.9628|
|v2\_segformer\+tta|real\_keybet\.com|158|0\.9367|0\.7911|0\.7342|0\.9064|

- 训练时 size 在 \[384, 512, 640, 768\] 中随机抽，crop 也随机，让 encoder 见过不同尺度后更鲁棒；推理时也用tta

    - no\-tta

    - 有tta

        |        model|        dataset|        n|        CDR@50|        CDR@80|        CDR@100|        ratio\_mean|
        |---|---|---|---|---|---|---|
        |        v2\_segformer\+tta|        synth\_test\_v0|        105|        0\.9238|        0\.6952|        0\.4571|        0\.8314|
        |        v2\_segformer\+tta|        synth\_test\_v1|        105|        0\.8571|        0\.6381|        0\.3619|        0\.7718|
        |        v2\_segformer\+tta|        real\_kb2\.fun<br>        （real time）|        100|        0\.93|        0\.93|        0\.93|        0\.9628|
        |        v2\_segformer\+tta|        real\_keybet\.com（real time）|        158|        0\.9367|        0\.7911|        0\.7342|        0\.9064|

![Image](https://internal-api-drive-stream-sg.larkoffice.com/space/api/box/stream/download/authcode/?code=YmMzMjJjYTQwOGZmNzczNGMyM2UzNDQyZTJkOTJhY2NfNGNhMzhlMmVlYzA3MDY1Njk5ZTYwOGNlNGIzZTY3OWNfSUQ6NzY0NzE5ODgzOTAyNzUzNTU4OV8xNzgyMDQ2OTMxOjE3ODIxMzMzMzFfVjM)

![Image](https://internal-api-drive-stream-sg.larkoffice.com/space/api/box/stream/download/authcode/?code=M2UxZDBkZjFiMjAxNzk1Y2JmMWE5YzVhZTg1NDQ3MDhfNWViZjU1MDJjMzI0YWM5ZDhmNTZkOWU4ZjkyMThjY2ZfSUQ6NzY0NzE5ODg0MDY3MTY4NjM3M18xNzgyMDQ2OTMxOjE3ODIxMzMzMzFfVjM)

![Image](https://internal-api-drive-stream-sg.larkoffice.com/space/api/box/stream/download/authcode/?code=YmZlZmVmYzY5ZDM4YWYwZTgxMDY3NDBiMTY5NWM1NmVfODFlN2U1YmU5NzVmODg1ZTlmYmIyYjBmYTFlNTgxZjhfSUQ6NzY0NzE5ODg0MjAyNzkwNDczNl8xNzgyMDQ2OTMxOjE3ODIxMzMzMzFfVjM)

![Image](https://internal-api-drive-stream-sg.larkoffice.com/space/api/box/stream/download/authcode/?code=MjRhMjMwMzlhMmQxZjcyMzEzNDc2NzgzZGVhYTFlODhfYWJkOGQzMTg3Yjk3ZjJjNDI2NWI4NWY5NGE1ZTliODRfSUQ6NzY0NzE5ODg0MjQ4NTEwMDI1Ml8xNzgyMDQ2OTMxOjE3ODIxMzMzMzFfVjM)

![Image](https://internal-api-drive-stream-sg.larkoffice.com/space/api/box/stream/download/authcode/?code=YzFmYWJjMjA1YTQ0ZmNiOGM3M2I5NzEwZTQ2OTYwYzdfMjU2NzRmNzkzNzZkMDUzM2NkM2FlNmMyNzdlZmQzZGRfSUQ6NzY0NzE5ODg0MDAyOTc2MTI0Nl8xNzgyMDQ2OTMxOjE3ODIxMzMzMzFfVjM)

![Image](https://internal-api-drive-stream-sg.larkoffice.com/space/api/box/stream/download/authcode/?code=M2Q5NDY1Y2ZiZjQzMWRkNDllNTljZjM0ODliMDAxMzZfY2UwNTlmY2Q4N2I3MmM0YWI4MmNiNjQ0NzczZmE1MjlfSUQ6NzY0NzE5ODgzODczODI3NjA2M18xNzgyMDQ2OTMwOjE3ODIxMzMzMzBfVjM)

    - 还有一类效果不好的是，外面一圈黑框的

![Image](https://internal-api-drive-stream-sg.larkoffice.com/space/api/box/stream/download/authcode/?code=MmM1OWRkYWZjYWU1N2I1MzVhYjNjNGU2NzVkMzUwMjNfOGY2YjJjOTMwZjU1NTU2NDA0ZTkxNWMyODAwNDBkYWRfSUQ6NzY0NzE5ODgzOTAyNzU1MTk3M18xNzgyMDQ2OTMwOjE3ODIxMzMzMzBfVjM)

![Image](https://internal-api-drive-stream-sg.larkoffice.com/space/api/box/stream/download/authcode/?code=NTc2ZGY5MmY0NjIyY2NhOGJlZDQ0ZGRlNjE1ZWViNjFfY2U5NDc1OTA1MDI1YzdlNTA0OTAzYzc2Y2YzNTEzY2VfSUQ6NzY0NzE5ODg0MjA4NjU1OTQ1Nl8xNzgyMDQ2OTMxOjE3ODIxMzMzMzFfVjM)

    - 另一方面是训练集和测试集存在一批bad case，可能会影响模型训练和评估

![Image](https://internal-api-drive-stream-sg.larkoffice.com/space/api/box/stream/download/authcode/?code=YTlkMzQ3NWE5YTA2NGE0MDFkZjZjMWNiYzExYmY5ZDNfMGMzODEzNDMwZDQ2MWM4ZjcyNjdiYmNiNmVkZTgxNTNfSUQ6NzY0NzE5ODg0MTQyNjQzMTcwOV8xNzgyMDQ2OTMxOjE3ODIxMzMzMzFfVjM)

![Image](https://internal-api-drive-stream-sg.larkoffice.com/space/api/box/stream/download/authcode/?code=MDEyZWRiZjJiOTA2ZTkwNTkyM2FmZGFhMzExZTdmOWJfMGFiOGE3NTA4OGIzZmZhNzQwM2JlOGM3ODU2NTVjODlfSUQ6NzY0NzE5ODg0MDc1MTIzMDY4N18xNzgyMDQ2OTMxOjE3ODIxMzMzMzFfVjM)

![Image](https://internal-api-drive-stream-sg.larkoffice.com/space/api/box/stream/download/authcode/?code=YTVhYzFjNzU4MTIzMjU1MzczNjg5MWNkOThkYWRjNzlfMWViMmQ1NzI4MDNiOGRhZWU2OGM1MWU5N2Q0ODM5N2JfSUQ6NzY0NzE5ODg0MDgxOTk0NTE4MF8xNzgyMDQ2OTMwOjE3ODIxMzMzMzBfVjM)

![Image](https://internal-api-drive-stream-sg.larkoffice.com/space/api/box/stream/download/authcode/?code=MTczMjNhMjAxOTc1Mzg3OTFlZTg1YTk4MjI0NTUwOWNfODRmYTBlYjdhMjU5ZmQ5ODg1OTIxY2FmNWFlMjk4ODNfSUQ6NzY0NzE5ODgzODc3MTY2NjY1Nl8xNzgyMDQ2OTMxOjE3ODIxMzMzMzFfVjM)

##### 6\.7 多尺度 TTA 

推理时将同一张图缩放到 \[0\.75, 1\.0, 1\.25, 1\.5\] 四个尺度，分别 forward → 上采样回同一分辨率 → 求 mean → 后处理。

效果对比（真实 kb2\.fun 数据）：

|配置|CDR@50|CDR@80|CDR@100|
|---|---|---|---|
|SegFormer 单尺度|0\.97|0\.51|0\.46|
|**SegFormer \+ 多尺度 TTA**|**0\.93**|**0\.93**|**0\.93**|

**关键发现：** 多尺度 TTA 让 CDR@80 和 CDR@100 从 0\.51/0\.46 跃升到 0\.93/0\.93，几乎完全消除了"检测到了但读不准"的问题。代价仅是推理时间 ×4，对审核场景完全可接受。

##### 6\.8 工程指标

- **模型大小**：

- **GPU 推理**：QPS ～ 22\.39

- **CPU 推理**：

#### 后续实验\-

瓶颈其实是来源于数据质量欠佳，通过上面的阐述，可以发现现在的数据有一部分人眼也看不清楚，这样的话其实就会让模型学到很多噪声。

- 但是人工看可能比较花时间，我采用自动化标注的方法去做一个清洗，具体来说，将训练和测试集中，通过blur\+histgram， fft高频过滤\+resize的方法进行预处理，然后将经过VLM提取少于30%文字的样本给去掉

![Image](https://internal-api-drive-stream-sg.larkoffice.com/space/api/box/stream/download/authcode/?code=MjVkZjU0ZGI4OGQ4NjU0NWQ5NDFhN2I0ZjNhMzgzM2ZfOWU0YWE2NjVmNjNhOTIzYjgzOGY4MGIyYTk1YWEyYTRfSUQ6NzY0ODk1NjIwMzkzNTY5ODY1NF8xNzgyMDQ2OTMxOjE3ODIxMzMzMzFfVjM)

![Image](https://internal-api-drive-stream-sg.larkoffice.com/space/api/box/stream/download/authcode/?code=OWY1Mjk2MzRkZDZhOGYxYzUyOWM5Yjc2MGRkYTA1YmZfNDZmMzNkMmFlZWQwNDI5Nzg1NDdhNTdhMTY4MDU5NjBfSUQ6NzY0ODk1NzIwODU3OTI4MDYwNF8xNzgyMDQ2OTMwOjE3ODIxMzMzMzBfVjM)

- 工程总结

```Bash
## 1. 脚本本身在做什么                                                                                                                                      
                                                                                                                                               
  • 输入：一张 RGB 图                                                                                                                                         
  • 输出：一张 H×W 的灰度概率图（隐藏文字像素的概率，0–255）                                                                                                  
  • 流程：                                                                                                                                                    
    1. load_model(...) 从 .pt ckpt 读出 arch / encoder / size 恢复模型                      2. infer_prob_patchify(...) 做多尺度 patch 切图推理                                      3. save_prob_image(...) 把概率图存成 PNG                                                                                                                  
                                                                                                                                                              
                                                                                                                                                              
  默认 ckpt 写死成 checkpoints_v1_segformer_b1_qwen_clean_20260607_125736/segformer_best.pt（详见 extract_hidden_text_image.py:34）。                         
                                                                                                                                                              
  ## 2. 模型                                                                                                                                                  
                                                                                                                                                              
  属于「二分类语义分割」：每个像素预测属于隐藏文字 / 不属于。                                                                                                 
                                                                                                                                                              
  • 框架：segmentation_models_pytorch (smp) https://github.com/qubvel/segmentation_models.pytorch                                                             
  • 架构（默认走的那条线，extract_hidden_text_image.py:43-46 + train.py:24-31）：                                                                             
    • SegFormer（smp.Segformer），编码器 mit_b1                                                                                                               
    • 训练时也支持 fallback 到 Unet + resnet34                                                                                                                
  • 输入：3×H×W，ImageNet mean/std 归一化（tta.py:20-21）                                                                                                     
  • 输出：1×H×W logits，sigmoid 后变成 0–1 概率                                                                                                               
                                                                                                                                                              
  为什么这套：                                                                                                                                                
                                                                                                                                                              
  • SegFormer 多分辨率 token + lightweight all-MLP decoder，在小目标 / 细笔画分割上比纯 CNN UNet 更稳                                                         
  • mit_b1 大小适中，可以 384/512/640 多尺度训练 + patchify 推理时不会爆显存                                                                                  
                                                                                                                                                              
  ## 3. 推理方法：patchify 多尺度 + ROI                                                                                                                       
                                                                                                                                                              
  核心在 tta.py，分两条线：                                                                                                                                   
                                                                                                                                                              
  1. 整图 TTA：infer_prob_tta(...)                                                                                                                            
    • 把整图 LongestMaxSize 到不同 size（0.75/1.0/1.25/1.5 × base_size），各推一次再对齐平均                                                                  
    • 适合「全局结构」，但有时会把细笔画糊掉                                                                                                                  
  2. patchify 多尺度（脚本默认走这条，tta.py:161）：                                                                                                          
    • 原图按 1.0/1.5/2.0 缩放                                                                                                                                 
    • 每个尺度下，以 short side 为边长切正方 patch，step_ratio=1.0 表示无重叠                                                                                 
    • 每个 patch 单独跑模型 → sigmoid → resize 回 patch 大小                                                                                                  
    • 重叠区做权重平均，再 resize 回原图坐标                                                                                                                  
    • 多尺度结果再平均                                                                                                                                        
                                                                                                                                                              
                                                                                                                                                              
  附带 detect_main_roi（Canny + 连通块 bbox），处理「手机截屏 + illusion 主体」这种长画布。                                                                   
                                                                                                                                                              
  ## 4. 数据 / GT 的门道（dataset.py）                                                                                                                        
                                                                                                                                                              
  这是这个项目能 work 的关键：                                                                                                                                
                                                                                                                                                              
  • 输入对：image_path（带隐藏文字的图）+ mask_path（白底黑字的控制图）                                                                                       
  • mask 来源 control_source 分两类：                                                                                                                         
    • "binary"：原始白底黑字 PNG，直接 <128 阈值二值化                                                                                                        
    • "iftg:*"：经过 gaussian_dim / pixel_shadow / tilt_blur / light_rotate 变换的 jpg，用 denoise_mask 处理：medianBlur → 硬阈值 <80 → MORPH_OPEN 去 jpg 散点
  → MORPH_CLOSE 连断裂笔画                                                                                                                                    
    • 注意：iftg 不能用 Otsu（pixel_shadow 的灰色阴影会被当前景），所以坚持硬阈值（dataset.py:72-81）                                                         
  • mask 与 image 分辨率经常不一致 → 先 resize 再去噪（dataset.py:122-127）                                                                                   
  • mask soft blur：训练时把 0/1 GT 做 GaussianBlur(sigma=mask_sigma)，默认配置用了 1.5，给 diffusion 弱引导导致的像素偏移留容差（dataset.py:131-136）——      
  这是个不太显眼但很关键的 trick                                                                                                                              
                                                                                                                                                              
  ## 5. 训练的门道（train.py + 文档）                                                                                                                         
                                                                                                                                                              
  主要套路：                                                                                                                                                  
                                                                                                                                                              
  1. Loss 组合：BCEWithLogitsLoss + smp.losses.DiceLoss(mode="binary", from_logits=True)（train.py:180-181），BCE 管像素级稳态、Dice 管小目标 IoU             
  2. Optimizer：AdamW(lr=1e-4, wd=1e-4) + CosineAnnealingLR(T_max=epochs)（train.py:182-185）                                                                 
  3. AMP 默认开：torch.cuda.amp.GradScaler，省显存                                                                                                            
  4. 多尺度训练（multiscale training）：--train-sizes 384,512,640，每个 sample 随机选一个 size（dataset.py:117）；这是当前 ckpt 的关键升级点                  
  5. batch 内动态 pad：pad_batch_collate(size_divisor=32) 把 batch 内不同形状统一 pad 到 32 倍数（SegFormer 必需，train.py:78-93）                            
  6. augmentation（仅训练，dataset.py:38-49）：                                                                                                               
    • LongestMaxSize + PadIfNeeded（保比例 + 填黑边）                                                                                                         
    • HorizontalFlip(0.5)、RandomBrightnessContrast(0.3)、GaussianBlur(0.2)、ImageCompression(0.3)                                                            
    • 注意：没有竖翻 / 90° 旋转，文字方向不能被破坏                                                                                                           
  7. 指标 & 选权重：每 epoch 算 dice/iou（阈值 0.5），按 best val_dice 存 *_best.pt，每个 ckpt 还存                                                           
  arch/encoder/size/train_sizes/mask_sigma，方便推理脚本不带配置直接 load                                                                                     
  8. resume：支持 --resume，epochs 解释为「再训多少轮」，并继承 best dice                                                                                     
  9. 数据清洗这条暗线（PATCHIFY_MODEL_GUIDE.md）：                                                                                                            
    • 用本地 qwen3-vl-4b VLM + 三种预处理（gaussian_blur_hist_eq / zoom_out_128 / smsp_fft_l0.02449_s128）去读 GT 隐藏字                                      
    • 与 GT 字符相似度对比，>= 阈值才保留                                                                                                                     
    • 阈值 0.5 太狠（保留 60.3%、hate_slangs 类被砍光），收紧到 0.3 才合理（保留 69.16%）                                                                     
    • 当前 ckpt = clean_0.3 数据 + multiscale training 训出来的                                                                                               
                                                                                                                                                              
                                                                                                                                                              
  ## 6. 当前推荐配置（一句话）                                                                                                                                
                                                                                                                                                              
    SegFormer-B1（mit_b1）+ BCE+Dice + AdamW + AMP + 多尺度训练（384/512/640）+ mask_sigma=1.5，在 qwen3-vl-4b 清洗到 0.3 阈值的数据上训 50                   
  epoch；推理时关掉整图 TTA、只走 patchify 1.0/1.5/2.0，对真实截屏（特别是 keybet.com）效果最好。                                                             
                          
```

### 针对视觉头能看见，但判断错

也就是模型已经具备弱感知、但决策不稳定的攻击类型，考虑用现在的攻击pipeline构造训练数据，继承原违规图的label，做SFT。相当于教会模型：同一个风险内容经过小图嵌入、遮挡、降质、风格迁移、几何变换、扰动后，标签不应该变。



