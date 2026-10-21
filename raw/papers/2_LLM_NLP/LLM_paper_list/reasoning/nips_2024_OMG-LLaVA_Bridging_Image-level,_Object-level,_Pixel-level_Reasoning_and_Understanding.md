# **OMG-LLaVA : Bridging Image-level, Object-level, Pixel-level Reasoning and Understanding** 

**Tao Zhang**[1] **, Xiangtai Li**[2] _[,]_[4] _[†]_ **, Hao Fei**[2] **, Haobo Yuan**[3] **, Shengqiong Wu**[2] **, Shunping Ji**[1] **, Chen Change Loy**[3] **, Shuicheng Yan**[2] 

1Wuhan University 2Skywork AI 3S-Lab, NTU 4Bytedance **Project page:** `https://lxtgh.github.io/project/omg_llava` _Email: xiangtai94@gmail.com and zhang_tao@whu.edu.cn_ 

## **Abstract** 

Current universal segmentation methods demonstrate strong capabilities in pixellevel image and video understanding. However, they lack reasoning abilities and cannot be controlled via text instructions. In contrast, large vision-language multimodal models exhibit powerful vision-based conversation and reasoning capabilities but lack pixel-level understanding and have difficulty accepting visual prompts for flexible user interaction. This paper proposes OMG-LLaVA, a new and elegant framework combining powerful pixel-level vision understanding with reasoning abilities. It can accept various visual and text prompts for flexible user interaction. Specifically, we use a universal segmentation method as the visual encoder, integrating image information, perception priors, and visual prompts into visual tokens provided to the LLM. The LLM is responsible for understanding the user’s text instructions and providing text responses and pixel-level segmentation results based on the visual information. We propose perception prior embedding to better integrate perception priors with image features. OMG-LLaVA achieves image-level, object-level, and pixel-level reasoning and understanding in a single model, matching or surpassing the performance of specialized methods on multiple benchmarks. Rather than using LLM to connect each specialist, our work aims at end-to-end training on one encoder, one decoder, and one LLM. The code and model have been released for further research. 

## **1 Introduction** 

With the development of transformer models [92; 6; 91; 38; 69; 90; 62; 124; 47; 85; 10; 19; 56], recent works in both natural language processing (NLP) and computer vision raise one common trend: adopting one unified model to solve multiple tasks. For example, large language models (LLMs) [91; 38; 90] adopt scale-up models to solve multiple NLP tasks and achieve better results than previous expert models. In vision, we have also seen a similar trend [19; 56; 98; 97; 45; 110], adopting one model to solve multiple tasks or sub-tasks, including detection, segmentation, video analysis, low-level vision, pose estimations, and more tasks. Different methods adopt different transformer designs, including visual-in-context learning [97; 98], unified decoder [19; 56], and unified tokenizer [84; 16; 56]. In summary, benefiting from the _scalability_ and _flexibility_ of the transformer, adopting one model for all tasks has made a great progress [19; 69; 70; 68; 124; 86; 85]. 

Work done when Tao Zhang is an intern at Skywork AI. _[†]_ : Project leader. Corresponding author: Xiangtai Li and Shunping Ji. 

38th Conference on Neural Information Processing Systems (NeurIPS 2024). 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_OMG-LLaVA_Bridging_Image-level,_Object-level,_Pixel-level_Reasoning_and_Understanding_images/nips_2024_OMG-LLaVA_Bridging_Image-level,_Object-level,_Pixel-level_Reasoning_and_Understanding.pdf-0002-00.png)


**----- Start of picture text -----**<br>
Grounded Conversation Generation Multiple Visual Prompts Description Image-level Conversation<br>A zebra is  Instruction: What is the number on<br>standing on the  this man's shirt?<br>grass. The zebra  Instruction: Please  Answer: The number on the man's<br>is drinking from a puddle. describe the <Region><Region> and . shirt is 11.Instruction: What sport is this man playing?<br>The image features a  Answer: The man is  Answer: The man is playing baseball.<br>red double-decker bus.  swinging a bat. Instruction: Why is there a red stain<br>This bus is parked on a  on this man's pants?<br>road. The bus is  Answer: The red stain on the man's<br>positioned in front of  pants could be due to several reasons.<br>trees, with the sky  One possibility is that the stain is dirt<br>Reasoning Segmentationoverhead. Visual Prompt-based ConversationInstruction: Please describe the  or mud from the playing field, which can easily get on the player's pants when they slide<br><Region>.Answer: A man, dressed in a  or dive into the dirt while playing baseball.<br>Please segment what  green shirt, is sitting on the<br>helps the person to  floor. He is holding a dog in his<br>stand on the ocean. arms, showing affection and<br>Referring Segmentation care.Instruction: Please describe what the <Region> is standing  Instruction: How many scissors on the table?Answer: Three.<br>on? Instruction: What the color of<br>Please segment the train that is furthest from the camera. Answer: The man is standing on a rug.Instruction: Please segment what is playing with <mark>. the middle scissor?Answer: It is black.<br>Answer: It is [SEG].<br>Iteractive Segmentation Semantic Segmentation Segmentation and Captioning Image-level Caption<br>1. A small, young lamb,  In the image, a woman is standing next to a<br>sitting on the grass in a field zebra, holding its reins. She is also petting the<br>2. A large, adult sheep,  zebra, showing affection towards the animal.<br>standing in the grassy area of  The scene takes place in a zoo, with the<br>the field  woman standing in front of a tree. The zebra<br>3. A large grass filed. is standing on dirt, and the woman appears to<br>Instance Segmentation Panoptic Segmentation Region-level Caption be enjoying her interaction with the animal.<br>The image features a white pickup truck parked on<br>A book is placed on top  a street. The truck is a large, full-size vehicle, and<br>of a laptop. It is titled  it is parked in front of a residential home. The<br>"Europe". The book is  truck is positioned on the street, and it appears<br>positioned on the right  to be parked in front of the home's driveway. The<br>side of the laptop. truck is also parked next to a curb, which is a<br>common feature on streets in many cities.<br>Pixel-Level Object-Level Image-Level<br>Reasoning<br>Understanding<br>**----- End of picture text -----**<br>


Figure 1: The comprehensive capabilities of OMG-LLaVA. OMG-LLaVA can handle a variety of pixel-level, object-level, and image-level understanding and reasoning tasks. 

Meanwhile, by combining vision models and language models [69; 70; 68; 62; 63; 105], research on multi-modal models also adopts transformer-based design. One representative work, LLaVA [69; 70; 68], treats visual tokens as the inputs of LLMs and makes LLMs understand visual contents. Several works adopt similar designs [3; 13; 62; 18; 25], and all of them are termed Multi-modal Large Language Models (MLLMs). After that, most research focuses on improving MLLM benchmarks in various ways, including increasing data sizes [14; 18; 68] and enhancing the visual encoders [129; 24; 18] and visual resolutions [108; 18; 63; 25]. However, LLaVA-like models cannot output precise location information since they only carry out image-level analysis. Thus, recent works [124; 131; 11; 80; 13; 117; 126; 85; 65] try to fill this gaps by adding extra detection models for object level analysis, mask decoder for pixel-level analysis, visual prompts, and also propose task-specific instruction tuning with various datasets. By providing extra detection data and a decoder, the updated MLLMs can perform localization output. However, these models [133; 94; 47] are specifically tuned on specific tasks, losing the ability of LLaVA for image level analysis, such as caption and visual question answering. Meanwhile, several works [124; 47; 85; 76] adopt LLMs as agents to collaborate with various visual models or generation models. Despite the works being simple and effective, the inference and parameter costs are huge due to the multiple visual encoders and decoders. Moreover, there are no specific designs for task unification. 

Motivated by the previous analysis, we ask one essential question: Can we bridge image-level, object-level, and pixel-level tasks into one MLLM model with only one LLM, one visual encoder, and one visual decoder? Back to the universal perception models, we can leverage these models to help us build a stronger MLLM to unify three-level inputs, including image, object, and pixel levels. In particular, we adopt OMG-Seg [56] as our universal perception model due to its simplicity and effectiveness in various segmentation tasks. 

In this work, we present OMG-LLaVA, an elegant MLLM that bridges image-level, object-level, and pixel-level reasoning and understanding tasks in one model. We preserve the basic pixel-level segmentation ability of OMG-Seg by freezing the visual encoder and decoder, as shown in the bottom left of Fig. 1. Since the LLM processes text input, OMG-LLaVA can also perform referring segmentation, reasoning segmentation, and grounded conversation and generation, shown in the top left of Fig. 1. Moreover, as shown in Fig. 1, with the help of LLMs, OMG-LLaVA can also perform 

2 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_OMG-LLaVA_Bridging_Image-level,_Object-level,_Pixel-level_Reasoning_and_Understanding_images/nips_2024_OMG-LLaVA_Bridging_Image-level,_Object-level,_Pixel-level_Reasoning_and_Understanding.pdf-0003-00.png)


**----- Start of picture text -----**<br>
Text Response Text ResponseLLM Perception Response Text Response Perception Response Text ResponseLLM Perception Response Text ResponseLLM<br>EncoderImageImage LLMInstructionText  EncoderImageImage ExtractorPromptsFeature Region Visual  InstructionText  PerceptionModule EncoderImageImage LLMInstructionText  PerceptionModule EncoderImageImage ExtractorPromptsFeature Visual Region  InstructionText  ImagePerception UniversalModulePromptsVisual  InstructionText<br>(a) (b) (c) (d) (e)<br>**----- End of picture text -----**<br>


Figure 2: Summary of Current MLLM Architectures: (a) MLLMs with only image-level capability, including [69; 70; 68; 63], etc., (b) MLLMs with object-level capability, including [124; 85], (c) MLLMs with pixel-level capability, including [47; 86], etc., (d) MLLMs with both object-level and pixel-level capabilities but with a very complex system, such as [85], (e) OMG-LLaVA’s architecture, which possesses an elegant and simple design while having image-level, object-level, and pixel-level capabilities. 

image-level understanding as LLaVA, including caption and conversation, where most MLLMs for grounding lose such ability. In addition, OMG-LLaVA also supports the visual prompts as inputs, which results in object level understanding, such as visual prompt-based conversation and region-level captions. We achieve all these abilities using one LLM, one encoder, and one decoder. 

In particular, to better encode the visual segmentation outputs, we propose a perception prior embedding module to absorb the object queries into object-centric visual tokens, which are the inputs of LLMs. We present a unified instruction formation strategy, which lets the model accept visual images, texts, and visual prompts as inputs and generate the response of text, segmentation tokens, segmentation masks, and labels. Following the LLaVA [69], we adopt pretraining and instruct tuning pipelines. Extensive experiments show the effectiveness of our components and training strategy. In addition to visual segmentation, OMG-LLaVA can also achieve good enough performance on 6 datasets, including COCO panoptic segmentation, VIPSeg video panoptic segmentation, refCOCO, refCOCO+, refCOCOg referring expression segmentation, GranDf grounded conversation generation, and refCOCOg region caption datasets. We hope our research can inspire the research on MLLM design in a more elegant way for the community. 

## **2 Related Work** 

**Multimodal Large Language Models.** Early multimodal models [51] explore better fusion strategies, various feature extractors, and different meta-architectures. Most works focus on single tasks, such as caption and VQA. With the development of the large language models [6; 91; 38], recent works [50; 3; 90; 69; 17] mainly explore building an instruction-tuning pipeline for multiple multimodal benchmarks [37; 72; 60; 30]. LLaVA [69; 68; 67; 104; 134] is one earlier work that treats visual features as tokens. After that, several works [124] explore visual cues to enhance the visual inputs of LLaVA. On the other hand, several works [127; 125; 86; 129; 24; 25; 64; 128; 81; 36; 47] add extra components to adapt LLaVA for visual grounding, detection, segmentation, and video analysis. In particular, several works explore language-driven grounding and segmentation. However, these works are all trained with a specific purpose. We aim to build the simplest model to unify segmentation, instruction tuning, and prompt-driven segmentation in one model. To the best of our knowledge, we are the first model to achieve this goal. 

**Unified Segmentation Models.** The vision transformers [10; 26; 77; 92] have led to research interest in universal segmentation. Recent works [93; 19; 121; 54; 21; 115; 113; 71; 89; 122; 120; 137; 109; 52; 139] have developed mask classification architectures with an end-to-end set prediction approach, outperforming previous specialized models [12; 44; 34; 55; 32; 58; 138] in both image, video and generalization segmentation tasks [43; 59; 57]. In particular, several works explore open-world segmentation, including entity segmentation [83; 82], open-vocabulary segmentation [123; 103]. Meanwhile, several works [56; 39; 109; 110; 31; 2] adopt one model with shared parameters to perform various segmentation tasks. One recent work, OMG-Seg [56], first unifies image, video, open-vocabulary, and interactive segmentation in one simple model. However, all of these works focus on visual segmentation and cannot generate interactive text and visual prompts, like MLLMs. Our work builds such a bridge to align MLLMs, visual segmentation, and prompt-driven segmentation models from joint co-training and model sharing, which serves as a new baseline for this field. 

**Language-driven Location and Segmentation.** Early works [118; 66; 42; 23; 102; 133] in this direction mainly define the various language-driven tasks, including referring segmentation and 

3 

Table 1: Comparison of capabilities of different models. We include several representative methods here. Our OMG-LLaVA offers the most comprehensive capabilities, encompassing image-level, object-level, and pixel-level understanding and reasoning. Compared to [85; 33], OMG-LLaVA features an elegant and simple system architecture with only a single visual encoder. 

|Method|Visual<br>Encoder|Image-level<br>Caption<br>Conversation|Object-level<br>Visual Prompts<br>Caption<br>Conversation|Pixel-level<br>Universal Seg<br>RES<br>GCG|
|---|---|---|---|---|
|LLAVA [69]<br>MiniGPT4 [140]<br>mPLUG-Owl [116]<br>LLaMA-Adapter [130]<br>Mini-Gemini [63]<br>InternVL 1.5 [18]<br>VisionLLM [95]<br>Shikra [13]<br>Kosmos-2 [80]<br>GPT4RoI [131]<br>Ferret [117]<br>Osprey [124]<br>SPHINX-V [65]<br>LISA [47]<br>GLAMM [85]<br>Groundhog [132]<br>AnyRef [33]<br>PixelLM [86]<br>GSVA [107]<br>Groma [76]<br>VIP-LLaVA [8]<br>PSALM [133]<br>LaSagnA [100]<br>OMG-Seg [56]|1<br>1<br>1<br>1<br>2<br>1<br>1<br>1<br>1<br>1<br>1<br>1<br>1<br>2<br>2<br>4<br>2<br>1<br>2<br>1<br>1<br>1<br>2<br>1|✓<br>✓<br>✓<br>✓<br>✓<br>✓<br>✓<br>✓<br>✓<br>✓<br>✓<br>✓<br>✓<br>✓<br>✓<br>✓<br>✓<br>✓<br>✓<br>✓<br>✓<br>✓<br>✓<br>✓<br>✓<br>✓<br>✓<br>✓<br>✓<br>✓<br>✓<br>✓<br>✓<br>✓<br>✓<br>✓<br>✓<br>✓<br>✓<br>✓<br>✓<br>✓|Point & Box<br>✓<br>✓<br>Box<br>✓<br>✓<br>Box<br>✓<br>✓<br>Point & Box & Mask<br>✓<br>✓<br>Mask<br>✓<br>✓<br>Point & Box & Mask<br>✓<br>✓<br>Box<br>✓<br>✓<br>Point & Box & Mask<br>✓<br>✓<br>Box<br>✓<br>✓<br>Box<br>✓<br>✓<br>Point & Box & Mask<br>✓<br>✓<br>Point & Box & Mask<br>Point|✓<br>✓<br>✓<br>✓<br>✓<br>✓<br>✓<br>✓<br>✓<br>✓<br>✓<br>✓<br>✓<br>✓|
|OMG-LLaVA|1|✓<br>✓|Point & Box & Mask<br>✓<br>✓|✓<br>✓<br>✓|



referring localization. Most works [29; 5; 114; 75; 101; 103] design effective fusion modules to achieve better performance. Meanwhile, several works [53; 101; 106; 47; 85; 124; 79] explore more complex language-driven tasks from various aspects, including robustness, reasoning, and region-level caption. LISA [112] involves reasoning-based segmentation. Then, GLaMM [85] annotates a new dataset and proposes region-level caption and segmentation tasks. Meanwhile, several works [28; 70] use LLMs as agents to assign different visual experts. In contrast to these works, our method is a more elegant baseline, which contains **only** one visual encoder, one LLM, and one decoder. 

**Visual Prompts.** With the prompting ability of LLMs, several works [98; 97; 4; 136; 88; 49; 79] also explore visual prompting methods in vision. According to the design and purposes, these works can be divided into different aspects, including learnable tokens [136], mask-visual-modeling for different tasks [98; 27; 96], and various visual prompting encoders for visual outputs [97; 99; 123; 45]. Our OMG-LLaVa also supports visual prompts for better interaction with the user’s inputs, showing the potential for product purposes. 

## **3 Methodology** 

## **3.1 Task Unification** 

**Motivation and Our Goals.** The LLMs unify most NLP tasks as token generation tasks and exhibit strong reasoning and instruction-following capabilities. As shown in Fig. 2 (a), LLaVAlike models [69; 68; 67; 108; 63; 129; 24; 25; 18; 62] further introduce visual tokens into LLMs, enabling LLMs to understand visual information and perform visual-based reasoning. However, they cannot accomplish fine-grained visual tasks like object-level and pixel-level understanding and reasoning. As shown in Fig. 2 (b), [124; 131; 11; 80; 13; 117] introduce region-level visual embeddings, allowing LLMs to achieve object-level understanding and reasoning tasks. However, these models rely on complex region embedding extraction designs. In addition, most cannot perform pixel-level understanding tasks. Thus, as shown in Fig. 2 (c),[47; 86; 33] introduce segmentation tokens, enabling LLMs to output segmentation masks and thus handle pixel-level understanding and reasoning tasks. Nonetheless, they require a large segmentation module, such as SAM [45], making the system highly redundant. As shown in Fig. 2 (d), GLAMM [85] combines the above pipelines to handle object-level and pixel-level tasks. However, this significantly increases the system’s _complexity_ and _redundancy_ . Additionally, GLAMM relies on explicit instructions from the user, **losing** the perception ability to handle basic pixel-level understanding tasks such as instance segmentation, semantic segmentation, panoptic segmentation, and interactive segmentation. 

4 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_OMG-LLaVA_Bridging_Image-level,_Object-level,_Pixel-level_Reasoning_and_Understanding_images/nips_2024_OMG-LLaVA_Bridging_Image-level,_Object-level,_Pixel-level_Reasoning_and_Understanding.pdf-0005-00.png)


**----- Start of picture text -----**<br>
Answer: It’s a wooden <p> pier </p> [SEG], surrounded by the <p> lake </p> [SEG] and the <p><br>forest </p> [SEG]. A blue <p> sky </p> [SEG] is above the pier and the <p> mountains </p> [SEG].<br>Decode<br>Perception Prior<br>Embedding<br>LLM<br>OMG Decoder<br>Text Projector<br>Visual Projector<br>Tokenize<br>Image<br>Question: <Image>\n  Encoder<br>Please describe the   Visual Prompts<br>Visual TokensPixel-centric Object-centricVisual Tokens OMG-Seg <Region>with interleaved segmentation masks. and respond  (Point, Box and Mask)<br>Image<br>**----- End of picture text -----**<br>


Figure 3: The Overview of OMG-LLaVA. OMG-LLaVA consists of OMG-Seg and LLM. OMG-Seg tokenizes the image into pixel-centric visual tokens, the detected objects, and inputs visual prompts into object-centric visual tokens. Additionally, the [SEG] token output by LLM is decoded by OMG-Seg into segmentation masks. OMG-Seg remains frozen at all stages. 

In this paper, we focus on addressing all the challenges above in a more simple yet elegant way. Our OMG-LLaVA unifies image-level (such as image caption and image-based conversation), object-level (such as region caption and visual prompt-based conversation), and pixel-level (such as universal segmentation, referring segmentation, reasoning segmentation, and grounded conversation generation) visual understanding and reasoning tasks into token-to-token generation. The framework follows a simple and elegant system design, including only one visual perception module and one large language model. 

**Unified View of Different Tasks.** We model various tasks as the token-to-token generation to bridge the gap between image-level, object-level, and pixel-level understanding and reasoning. To support these tasks, we define three types of tokens: text tokens _Tt_ , pixel-centric visual tokens _Tpv_ , and object-centric visual tokens _Tov_ . Text tokens encode textual information. Pixel-centric visual tokens represent dense image features, providing the LLM with comprehensive image information. Object-centric visual tokens encode the features of specified objects, offering the LLM object-centric information, and can be easily decoded into segmentation masks. 

Then, all the tasks can be unified as: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_OMG-LLaVA_Bridging_Image-level,_Object-level,_Pixel-level_Reasoning_and_Understanding_images/nips_2024_OMG-LLaVA_Bridging_Image-level,_Object-level,_Pixel-level_Reasoning_and_Understanding.pdf-0005-05.png)


For example, in the classic image-level understanding task, i.e., image caption, a text response _Tt[out]_ is generated based on text instruction _Tt[in]_ and image features _Tpv[in]_[.][In the object-level understanding] task, region captioning, the text response _Tt[out]_ is generated based on text instruction _Tt[in]_[,][image] features _Tpv[in]_[, and specified object-centric visual tokens] _[ T][ in] ov_[.][The pixel-level reasoning task, referring] segmentation, involves generating object-centric visual tokens _Tov[out]_ based on text instruction _Tt[in]_ and image features _Tpv[in]_[.][Additionally, OMG-LLaVA can support various mixed-level tasks, such as] providing grounded descriptions around specified objects. 

Pixel-centric visual tokens can be obtained by tokenizing images using a CLIP backbone as the tokenizer. However, object-centric visual tokens require encoding object information to be easily decoded into segmentation masks. Therefore, methods like mask pooling in Osprey [124] and ROI pooling in GLaMM [85] fail to meet these requirements. We found that a universal perception decoder can meet all the requirements. Thus, we chose the OMG-Seg decoder [56] as the object-centric tokenizer due to its comprehensive capabilities. 

## **3.2 OMG-LLaVA Framework** 

The framework of OMG-LLaVA is shown in Fig. 2 (e). OMG-LLaVA comprises a large language model (LLM) and a _frozen_ universal perception module. The universal perception module encodes images and visual prompts from users into pixel-centric and object-centric visual tokens. It obtains object-centric visual tokens output by the LLM into explicit segmentation mask responses. The LLM accepts text instruction tokens and pixel-centric and object-centric visual tokens from the universal perception module as inputs and then outputs text responses along with object-centric visual tokens. The detailed architecture of OMG-LLaVA is illustrated in Fig. 3. The universal perception module 

5 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_OMG-LLaVA_Bridging_Image-level,_Object-level,_Pixel-level_Reasoning_and_Understanding_images/nips_2024_OMG-LLaVA_Bridging_Image-level,_Object-level,_Pixel-level_Reasoning_and_Understanding.pdf-0006-00.png)


**----- Start of picture text -----**<br>
Attention Mask for Point, Box, and Mask Prompts<br>OMG Decoder<br>Point prompt Center Mask prompt<br>Visual Prompts Encode Box prompt Encode<br>(Point, Box and Mask)<br>Encode Q Mask  Mask<br>Predict Predict<br>Visual Prompts Queries<br>Crop<br>Learnable Queries<br>K&V Object<br>Queries Attention Mask Attention Mask Attention Mask<br>Image Feature<br>Self-Attention Self-Attention<br>Masked Cross-Attention Masked Cross-Attention<br>**----- End of picture text -----**<br>


Figure 4: The Architecture of the OMG Decoder. A simple attention mask generation strategy enables the OMG decoder to encode point, box, and mask prompts. 

comprises an image encoder, an OMG decoder [56], and a non-trainable perception prior embedding component. 

**Image Encoder.** To maximize the perception capabilities of the universal perception module, we use the ConvNeXt-L [73]-based CLIP [84] model as the image encoder and employ a high image resolution (1024 _×_ 1024). However, the large image resolution results in excessive visual tokens input into the LLM, leading to significantly higher computational costs than using lower-resolution images (such as 224 _×_ 224 or 336 _×_ 336). We address this issue by utilizing the lowest resolution image features (32 _×_ downsampling). Additionally, we use the pixel shuffle operator to further reduce the image features’ resolution. Ultimately, the downsampling factor for the image features used to generate visual tokens is 64, meaning that a 1024 _×_ 1024 image produces 256 visual tokens. 

**OMG Decoder.** We utilize the OMG decoder [56] to generate object-centric visual tokens, furnishing the LLM with information regarding the primary objects in the image and those mentioned by the user’s input visual prompts. As shown on the left side of Fig. 4, the OMG decoder comprises masked cross-attention [19] and self-attention layers. The OMG decoder’s input includes a set of learnable object queries [20; 19; 10] for automatically capturing all objects of interest and visual prompt queries derived from encoded input visual prompts [45]. The visual prompt queries and learnable object queries are collectively termed object queries. The OMG decoder probes feature for object queries from the image features by employing masked cross-attention and models relationships between objects through self-attention. The object queries can be decoded into segmentation masks and object categories via a simple FFN layer. With the OMG decoder, OMG-LLaVA can efficiently tokenize object information into object-centric visual tokens, thereby equipping the LLM with information about objects in the image and those referenced by the user. 

The OMG decoder can accept point prompts as input. While box and mask prompts can be easily converted into point prompts, this crude conversion significantly loses prompt information, complicating the explicit encoding of the user’s intent. To address this, we can impose constraints on the attention masks of the masked cross-attention layers based on the visual prompt to precisely encode the object information referenced by the prompt. As depicted on the right side of Fig. 4, we utilize the box coordinates to define attention masks for all pixel features outside the box for box prompts. Similarly, we directly employ the provided object mask to generate attention masks for mask prompts. With this straightforward attention mask modification strategy, OMG-LLaVA can accurately capture the user’s visual prompts, encompassing point, box, and mask prompts. 

## **Perception Prior Embed-** 

**ding.** We find that directly combining a frozen perception module with LLM doesn’t perform well, as also observed in LISA [47]. To retain the full capabilities of the universal perception module, OMG-LLaVA doesn’t fine-tune the perception module to adapt to the output of the large language model. Instead, we propose 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_OMG-LLaVA_Bridging_Image-level,_Object-level,_Pixel-level_Reasoning_and_Understanding_images/nips_2024_OMG-LLaVA_Bridging_Image-level,_Object-level,_Pixel-level_Reasoning_and_Understanding.pdf-0006-08.png)


**----- Start of picture text -----**<br>
Filter background queries<br>(��, ��) (��, �)<br>Predcit<br>C<br>Object Queries Visual Tokens<br>Mask Score Query<br>Embedding<br>Image Feature<br>**----- End of picture text -----**<br>


Figure 5: The process of the perception prior embedding strategy. The perception prior embedding strategy integrates object queries into image features based on segmentation prior. 

6 

a perception prior embed- 

ding strategy to tackle this challenge. Fig. 5 illustrates the perception prior embedding strategy. 

First, we fuse the image features _F ∈_ R _[HW][ ×][C]_ outputted by the image encoder with the object queries _Q ∈_ R _[N][q][×][C]_ outputted by the OMG decoder _D_ . Specifically, we utilize the segmentation mask _M ∈_ R _[N][q][×][HW]_ obtained from the object queries and the corresponding confidence score _S ∈_ R[1] _[×][N][q]_ to derive a mask score _MS ∈_ R _[HW][ ×][N][q]_ for each pixel for the object queries: 

_MS_ = _Softmax_ ( _M ⊙S, dim_ = _−_ 1) (2) 

Then, we compute a weighted average of the object queries _Q_ based on the mask score _MS_ and obtain the corresponding weighted object queries for each pixel. Pixel-centric visual tokens _Tpv_ are obtained by adding the weighted object queries to the image features _F_ : 

_Tpv_ = _MS · Q_ + _F_ (3) 

Additionally, we treat the foreground object queries as object-centric visual tokens _Tov_ . The objectcentric visual tokens _Tov_ are concatenated with the pixel-centric visual tokens _Tpv_ to form the visual tokens _Tv_ = ( _Tpv, Tov_ ), which are input to the LLM to provide rich perception prior information. 

**Visual Projector and Text Projector.** Following [69], we use an MLP as the visual projector, which is responsible for mapping visual tokens to the LLM’s text embedding space. Since our visual tokens consist of pixel-centric and object-centric tokens, the visual projector comprises two MLPs, each handling one type of visual token separately. Inspired by [47; 85], we also use a simple MLP to map the LLM output’s hidden states of the [SEG] token to the visual space. 

**Instruction Formulation.** OMG-LLaVA can accept **visual** input, **text** input, and **visual prompt** input and output text responses and segmentation token, segmentation masks and labels. Thus, it can handle tasks such as image captioning, image-based conversation, region captioning, visual prompt-based conversation, referring segmentation, reasoning segmentation, grounded conversation, etc. We use a unified instruction formulation to support these functionalities. As shown in Fig. 3, there are three special tokens: <Image>, <Region>, and [SEG]. Before being fed into the LLM, the <Image> token is replaced by visual tokens _Tv_ , and the <Region> token can be replaced by any object-centric visual token encoded by the visual prompt. The [SEG] token in the LLM’s output is sent to the frozen OMG decoder to be decoded into a segmentation mask. 

## **3.3 Training and Testing Setup** 

**Training.** Following LLaVA [69], our OMG-LLaVA performs two-stage training: pretraining and instruction tuning. During the pretraining stage, the perception model and LLM are frozen, and only the visual and text projectors can be tuned. In addition to the text regression loss, we apply regularization penalties to the visual projector _Pv_ and text projector _Pt_ to preserve object-centric information as much as possible. 

_Lpretrain_ = _Ltext_ + _Lreg, Lreg_ = ( _Tov −Pt_ ( _Pv_ ( _Tov_ )))[2] (4) 

During instruction tuning, in addition to finetuning the visual projector and text projector, we use LoRA [35] to finetune the LLM. Following [85; 56], besides the text regression loss, we apply cross-entropy loss and dice loss [78] to supervise the segmentation mask decoded by the [SEG] token. 

_Lintruction_ = _Ltext_ + _Lmask, Lmask_ = _αLCE_ + _βLDICE_ (5) 

**Testing.** The image-level, object-level, and pixel-level understanding and reasoning tasks can all be encompassed within the Eq. 3.1 paradigm. During the inference stage, we encode the necessary task requirements, such as text prompts, visual prompts, and image features, into tokens to input into the LLM. The output tokens of LLM are then decoded into text responses and segmentation mask responses according to the task definition. We refer the readers to check the more details in the appendix. 

## **4 Experiment** 

**Dataset Setup.** During the pretraining stage, we use the LLaVA pretraining dataset [69] to perform visual-text alignment, following LLaVA. The instruction tuning process of OMG-LLaVA involves a 

7 

Table 2: The comprehensive comparison of OMG-LLaVA and other MLLMs regarding pixel-level and objectlevel understanding and reasoning capability and performance. "-" indicates that the method does not handle this task. _†_ indicates that the method used the GranD dataset [85] for pretraining, which is significantly larger than the datasets used by other methods. 

|Method|Visual<br>Encoder Num|COCO<br>VIPseg<br>refCOCO<br>refCOCO+<br>GCG<br>refCOCOg(C)<br>PQ<br>VPQ<br>cIoU<br>cIoU<br>METEOR<br>AP50<br>METEOR|
|---|---|---|
|OSprey [124]<br>LISA [47]<br>NeXT-Chat [126]<br>LaSagnA [100]<br>GSVA [107]<br>AnyRef [33]<br>GLaMM† [85]<br>PixelLM[86]|1<br>2<br>2<br>2<br>2<br>2<br>2<br>1|-<br>-<br>-<br>-<br>-<br>-<br>16.6<br>-<br>-<br>74.1<br>62.4<br>13.0<br>25.2<br>-<br>-<br>-<br>74.7<br>65.1<br>-<br>-<br>12.0<br>-<br>-<br>76.8<br>66.4<br>-<br>-<br>-<br>-<br>-<br>76.4<br>64.5<br>-<br>-<br>-<br>-<br>74.1<br>64.1<br>-<br>-<br>16.2<br>-<br>-<br>79.5<br>72.6<br>15.2<br>28.9<br>15.7<br>-<br>-<br>73.0<br>66.3<br>-<br>-<br>-|
|OMG-LLaVA|1|53.8<br>49.8<br>78.0<br>69.1<br>14.9<br>29.9<br>15.3|



Table 3: Performance on referring expression segmentation datasets. The evaluation metric is cIoU. "ft" indicates finetuning on the referring expression datasets. 

|Method<br>Freeze<br>Visual<br>Decoder<br>Encoder|refCOCO<br>Val<br>TestA<br>TestB|refCOCO+<br>Val<br>TestA<br>TestB|refCOCOg<br>Val<br>Test|
|---|---|---|---|
|LISA [47]<br>_×_<br>2<br>LISA(ft) [47]<br>_×_<br>2<br>PixelLM [86]<br>_×_<br>1<br>GSVA(ft) [107]<br>_×_<br>2|74.1<br>76.5<br>71.1<br>74.9<br>79.1<br>72.3<br>73.0<br>76.5<br>68.2<br>77.2<br>78.9<br>73.5|62.4<br>67.4<br>56.5<br>65.1<br>70.8<br>58.1<br>66.3<br>71.7<br>58.3<br>65.9<br>69.6<br>59.8|66.4<br>68.5<br>67.9<br>70.6<br>69.3<br>70.5<br>72.7<br>73.3|
|OMG-LLaVA<br>✓<br>1<br>OMG-LLaVA(ft)<br>_×_<br>1<br>OMG-LLaVA(ft)<br>✓<br>1|75.6<br>77.7<br>71.2<br>78.0<br>80.3<br>74.1<br>77.2<br>79.8<br>74.1|65.6<br>69.7<br>58.9<br>69.1<br>73.1<br>63.0<br>68.7<br>73.0<br>61.6|70.7<br>70.2<br>72.9<br>72.9<br>71.7<br>71.9|



Table 4: Performance on grounded conversation generation datasets. “ft” indicates finetuning on the GranDf [85] dataset. _†_ indicates that the method used the GranD dataset [85] for pretraining. 

|Methods<br>ft<br>Visual<br>Encoder|Val<br>METEOR<br>CIDEr<br>AP50<br>mIOU|Test<br>METEOR<br>CIDEr<br>AP50<br>mIOU|
|---|---|---|
|Kosmos-2 [80]<br>✓<br>1<br>LISA [47]<br>✓<br>2<br>GLaMM† [85]<br>✓<br>2|16.1<br>27.6<br>17.1<br>55.6<br>13.0<br>33.9<br>25.2<br>62.0<br>15.2<br>43.1<br>28.9<br>65.8|15.8<br>27.2<br>17.2<br>56.8<br>12.9<br>32.2<br>24.8<br>61.7<br>14.6<br>37.9<br>27.2<br>64.6|
|OMG-LLaVA<br>_×_<br>1<br>OMG-LLaVA<br>✓<br>1|13.8<br>36.2<br>26.9<br>64.6<br>14.9<br>41.2<br>29.9<br>65.5|13.5<br>33.1<br>26.1<br>62.8<br>14.5<br>38.5<br>28.6<br>64.7|



Table 5: Ablation study on RES and GCG datasets. 

|Methods|refCOCO<br>refCOCO+<br>refCOCOg<br>cIoU<br>gIoU<br>cIoU<br>gIoU<br>cIoU<br>gIoU|GCG<br>METEOR<br>mIoU|
|---|---|---|
|Baseline (M0)<br>+ Perception prior embedding (M1)<br>+ Object query input (M2)|58.7<br>61.0<br>52.6<br>55.0<br>55.8<br>58.1<br>72.5<br>74.3<br>63.2<br>65.4<br>67.8<br>70.6<br>74.4<br>75.9<br>64.4<br>66.2<br>68.5<br>71.5|13.2<br>51.0<br>13.6<br>62.1<br>13.8<br>63.6|



diverse range of tasks and datasets. For image-level understanding and reasoning tasks, we use the LLaVA dataset [69; 70; 68], which includes 665K descriptions, reasoning, and conversation data. For object-level understanding and reasoning, we use the object-level description and conversation data from the Osprey dataset [124] and the object-level point-prompt data from the MDVP dataset [65], which contain approximately 74K and 200K data, respectively. For pixel-level understanding and reasoning, we use the referring segmentation datasets, including refCOCO, refCOCO+ [40], refCOCOg [119], and refClef, totaling 74K data. Additionally, semantic segmentation datasets, including ADE20k [135] and COCO-stuff [7], totaling 26K data, and the grounded conversation generation dataset GranDf [85], containing 200K data, are used. 

**Implementation Details.** We use the pre-trained ConvNext-L [73] OMG-Seg [56] as the universal perception module and InterLM2-7B [9] as the LLM for OMG-LLaVA. We adopt xtuner codebase [22] to build our model and data pipeline. The image is resized to 1024 _×_ 1024. During the pretraining stage, only the visual projector and text projector are trained, with an initial learning rate set to 1e-3. During the instruction tuning stage, the initial learning rate is set to 2e-4, with only the 

8 

perception model kept frozen, and the LLM is fine-tuned using LoRA [35]. The maximum sequence length in the LLM is set to 2,048. All training is conducted on four NVIDIA A800 GPUs with 80GB of memory. The pretraining stage and instruction tuning stage took 7 hours and 48 hours, respectively. 

## **4.1 Main Results** 

**Comparison with MLLMs.** OMG-LLaVA is comprehensively compared with current MLLMs with perception capabilities, and the results are shown in Tab. 2. OMG-LLaVA demonstrates the most comprehensive capabilities. It achieves performance comparable to the SOTA in referring segmentation, grounded conversation generation, and region captioning. Additionally, OMG-LLaVA retains basic segmentation ability, enabling it to handle universal image and video segmentation tasks. Compared to other MLLMs, OMG-LLaVA features a simple and elegant system design, incorporating only a single visual encoder. 

**Referring Expression Segmentation.** We evaluate OMG-LLaVA on refCOCO, refCOCO+, and refCOCOg, with the results shown in Tab. 3. OMG-LLaVA outperforms LISA [47] by 1.5 cIoU, 3.2 cIoU, and 4.3 cIoU on the validation sets of refCOCO, refCOCO+, and refCOCOg, respectively, while keeping the OMG decoder frozen and using only a single visual encoder. When we unfreeze the OMG decoder and finetune OMG-LLaVA on the referring expression segmentation task, OMG-LLaVA achieves 78.0, 69.1, and 72.9 cIoU on refCOCO, refCOCO+, and refCOCOg, respectively, surpassing LISA by 3.1, 4.0, and 5.0 cIoU. Compared to PixelLM [86], OMG-LLaVA shows performance improvements of 5.0 cIoU and 3.6 cIoU on refCOCO and refCOCOg, respectively. 

**Grounded Conversation Generation.** Grounded conversation generation is a comprehensive and complex task that involves both image-level and pixel-level understanding and reasoning. MLLMs need to have the ability to provide fine-grained image descriptions and pixel-level understanding, linking the objects in the image captions to the corresponding segmentation masks. As shown in Tab. 4, when trained with comparable data, OMG-LLaVA surpasses LISA [47] by 1.9 METEOR and 7.3 CIDEr in image description ability. In terms of pixel understanding, OMG-LLaVA also outperforms LISA by 4.7 AP50 and 3.5 mIoU, even though LISA uses SAM and finetunes its segmentation decoder. Despite GLaMM [85] using much more training data than OMG-LLaVA, OMG-LLaVA demonstrates comparable pixel-understanding capabilities, outperforming GLaMM with 0.6 CIDEr, 1.4 AP50 and 0.1 mIoU on the test set. 

## **4.2 Ablation and Analysis** 

**Ablation Study.** We conduct ablation studies on referring expression segmentation and grounded conversation generation datasets, with all training and testing settings consistent with the main experiments. We use a simple combination of OMG-Seg [56] and LLaVA [69] as our baseline, similar to LISA [47], where the [SEG] tokens output by the LLM were input into OMG-Seg to obtain segmentation masks, with OMG-Seg kept frozen. 

As shown in Tab. 5, the baseline performed poorly on the RES datasets. Similarly, it exhibited low segmentation quality on the GCG dataset. This is because the LLM did not acquire any segmentation priors and needed to generate segmentation queries based on image features and adapt them to the input of the frozen perception module, which is a challenging task. When using our proposed perception prior embedding strategy, OMG-LLaVA exhibits performance gains of 13.8 cIoU, 10.6 cIoU, and 11.7 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_OMG-LLaVA_Bridging_Image-level,_Object-level,_Pixel-level_Reasoning_and_Understanding_images/nips_2024_OMG-LLaVA_Bridging_Image-level,_Object-level,_Pixel-level_Reasoning_and_Understanding.pdf-0009-08.png)


**----- Start of picture text -----**<br>
Please describe the image and respond with interleaved segmentation masks<br>A horse jumping over a hurdle A horse is jumping over a hurdle A womanhorse in a competition is riding a brown<br>Please segment the hurdle<br>Please segment the left fence<br>**----- End of picture text -----**<br>


Figure 6: Visualization of the effectiveness of the proposed strategies. The left part shows the baseline (M0 in Tab. 5), the middle part shows the model with perception prior embedding (M1 in Tab. 5), and the right part shows the model with both perception prior embedding and object query input (M2 in Tab. 5). 

cIoU on refCOCO, refCOCO+, and refCOCOg, respectively. Additionally, the perception prior embedding strategy also brings a performance improvement of 11.1 mIoU on the GCG dataset and a 

9 

slight improvement in image description capability (0.4 METEOR). When foreground object queries were provided to the LLM, OMG-LLaVA further improved its performance by 1.9 cIoU on refCOCO and 1.5 mIoU on GCG. 

We conducted a visualization analysis of the proposed strategies. As shown in the left part of Fig. 6, the simple baseline has poor capability in associating text and segmentation, which is the crucial reason for its poor performance on RES. When using our proposed perception prior embedding strategy, the object query and pixel features are explicitly integrated according to the perception prior, resulting in significantly enhanced text-segmentation association capability. By adopting the object query input strategy, the quality of some challenging segmentation cases, such as the lower right corner of the fence in Fig 6, slightly improves. 

**Qualitative Results.** We provide visualization results of OMG-LLaVA on multiple image-level, object-level, and pixel-level tasks in Fig. 1. Additional qualitative visualization results or comparable visual results for referring expression segmentation and grounded conversation generation are presented in the appendix. 

## **5 Conclusion** 

We present a new MLLM, OMG-LLaVA, which bridges image-level, object-level, and pixel-level understanding and reasoning in one model. Our method only contains one image encoder, one LLM, and one decoder. With proposed perception prior embedding and unified task instruction tuning, OMG-LLaVA can perform over 8 different multi-modal learning tasks, as well as preserving the visual perception ability of the OMG-Seg baseline. Our method can achieve comparable results compared with previous combined works with much fewer trainable parameters and computation costs. We hope our work can inspire the community to rethink the design of the MLLM meta-architecture to minimize the model components and maximize the MLLM’s functionalities. 

**Acknowledgment.** This work was supported by the National Natural Science Foundation of China (grant No. 42171430). It is also partially supported under the RIE2020 Industry Alignment Fund Industry Collaboration Projects (IAF-ICP) Funding Initiative, as well as cash and in-kind contributions from the industry partner(s). 

## **References** 

- [1] Marah Abdin, Sam Ade Jacobs, Ammar Ahmad Awan, Jyoti Aneja, Ahmed Awadallah, Hany Awadalla, Nguyen Bach, Amit Bahree, Arash Bakhtiari, Harkirat Behl, et al. Phi-3 technical report: A highly capable language model locally on your phone. _arXiv preprint arXiv:2404.14219_ , 2024. 16, 17 

- [2] Ali Athar, Alexander Hermans, Jonathon Luiten, Deva Ramanan, and Bastian Leibe. Tarvis: A unified architecture for target-based video segmentation. In _CVPR_ , 2023. 3 

- [3] Jinze Bai, Shuai Bai, Shusheng Yang, Shijie Wang, Sinan Tan, Peng Wang, Junyang Lin, Chang Zhou, and Jingren Zhou. Qwen-vl: A versatile vision-language model for understanding, localization, text reading, and beyond. _arXiv preprint arXiv:2308.12966_ , 2023. 2, 3 

- [4] Amir Bar, Yossi Gandelsman, Trevor Darrell, Amir Globerson, and Alexei Efros. Visual prompting via image inpainting. In _NeurIPS_ , 2022. 4 

- [5] Adam Botach, Evgenii Zheltonozhskii, and Chaim Baskin. End-to-end referring video object segmentation with multimodal transformers. In _CVPR_ , 2022. 4 

- [6] Tom Brown, Benjamin Mann, Nick Ryder, Melanie Subbiah, Jared D Kaplan, Prafulla Dhariwal, Arvind Neelakantan, Pranav Shyam, Girish Sastry, Amanda Askell, et al. Language models are few-shot learners. In _NeurIPS_ , 2020. 1, 3 

- [7] Holger Caesar, Jasper Uijlings, and Vittorio Ferrari. Coco-stuff: Thing and stuff classes in context. In _Proceedings of the IEEE conference on computer vision and pattern recognition_ , pages 1209–1218, 2018. 8 

- [8] Mu Cai, Haotian Liu, Siva Karthik Mustikovela, Gregory P. Meyer, Yuning Chai, Dennis Park, and Yong Jae Lee. Making large multimodal models understand arbitrary visual prompts. In _CVPR_ , 2024. 4 

- [9] Zheng Cai, Maosong Cao, Haojiong Chen, Kai Chen, Keyu Chen, Xin Chen, Xun Chen, Zehui Chen, Zhi Chen, Pei Chu, et al. Internlm2 technical report. _arXiv preprint arXiv:2403.17297_ , 2024. 8, 16, 17 

- [10] Nicolas Carion, Francisco Massa, Gabriel Synnaeve, Nicolas Usunier, Alexander Kirillov, and Sergey Zagoruyko. End-to-end object detection with transformers. In _ECCV_ , 2020. 1, 3, 6 

- [11] Chi Chen, Ruoyu Qin, Fuwen Luo, Xiaoyue Mi, Peng Li, Maosong Sun, and Yang Liu. Position-enhanced visual instruction tuning for multimodal large language models. _arXiv preprint arXiv:2308.13437_ , 2023. 2, 4 

10 

- [12] Kai Chen, Jiangmiao Pang, Jiaqi Wang, Yu Xiong, Xiaoxiao Li, Shuyang Sun, Wansen Feng, Ziwei Liu, Jianping Shi, Wanli Ouyang, Chen Change Loy, and Dahua Lin. Hybrid task cascade for instance segmentation. In _CVPR_ , 2019. 3 

- [13] Keqin Chen, Zhao Zhang, Weili Zeng, Richong Zhang, Feng Zhu, and Rui Zhao. Shikra: Unleashing multimodal llm’s referential dialogue magic. _arXiv preprint arXiv:2306.15195_ , 2023. 2, 4 

- [14] Lin Chen, Jisong Li, Xiaoyi Dong, Pan Zhang, Conghui He, Jiaqi Wang, Feng Zhao, and Dahua Lin. Sharegpt4v: Improving large multi-modal models with better captions. _arXiv preprint arXiv:2311.12793_ , 2023. 2 

- [15] Lin Chen, Jinsong Li, Xiaoyi Dong, Pan Zhang, Yuhang Zang, Zehui Chen, Haodong Duan, Jiaqi Wang, Yu Qiao, Dahua Lin, et al. Are we on the right way for evaluating large vision-language models? _arXiv preprint arXiv:2403.20330_ , 2024. 17 

- [16] Ting Chen, Saurabh Saxena, Lala Li, David J Fleet, and Geoffrey Hinton. Pix2seq: A language modeling framework for object detection. _ICLR_ , 2022. 1 

- [17] Wei-Ge Chen, Irina Spiridonova, Jianwei Yang, Jianfeng Gao, and Chunyuan Li. Llava-interactive: An all-in-one demo for image chat, segmentation, generation and editing. _arXiv preprint arXiv:2311.00571_ , 2023. 3 

- [18] Zhe Chen, Weiyun Wang, Hao Tian, Shenglong Ye, Zhangwei Gao, Erfei Cui, Wenwen Tong, Kongzhi Hu, Jiapeng Luo, Zheng Ma, et al. How far are we to gpt-4v? closing the gap to commercial multimodal models with open-source suites. _arXiv preprint arXiv:2404.16821_ , 2024. 2, 4 

- [19] Bowen Cheng, Ishan Misra, Alexander G. Schwing, Alexander Kirillov, and Rohit Girdhar. Maskedattention mask transformer for universal image segmentation. In _CVPR_ , 2022. 1, 3, 6 

- [20] Bowen Cheng, Alexander G. Schwing, and Alexander Kirillov. Per-pixel classification is not all you need for semantic segmentation. In _NeurIPS_ , 2021. 6 

- [21] Yangming Cheng, Liulei Li, Yuanyou Xu, Xiaodi Li, Zongxin Yang, Wenguan Wang, and Yi Yang. Segment and track anything. _arXiv preprint arXiv:2305.06558_ , 2023. 3 

- [22] XTuner Contributors. Xtuner: A toolkit for efficiently fine-tuning llm. `https://github.com/ InternLM/xtuner` , 2023. 8 

- [23] Henghui Ding, Chang Liu, Shuting He, Xudong Jiang, and Chen Change Loy. Mevis: A large-scale benchmark for video segmentation with motion expressions. In _ICCV_ , 2023. 3 

- [24] Xiaoyi Dong, Pan Zhang, Yuhang Zang, Yuhang Cao, Bin Wang, Linke Ouyang, Xilin Wei, Songyang Zhang, Haodong Duan, Maosong Cao, Wenwei Zhang, Yining Li, Hang Yan, Yang Gao, Xinyue Zhang, Wei Li, Jingwen Li, Kai Chen, Conghui He, Xingcheng Zhang, Yu Qiao, Dahua Lin, and Jiaqi Wang. Internlm-xcomposer2: Mastering free-form text-image composition and comprehension in vision-language large model. _arXiv preprint arXiv:2401.16420_ , 2024. 2, 3, 4 

- [25] Xiaoyi Dong, Pan Zhang, Yuhang Zang, Yuhang Cao, Bin Wang, Linke Ouyang, Songyang Zhang, Haodong Duan, Wenwei Zhang, Yining Li, Hang Yan, Yang Gao, Zhe Chen, Xinyue Zhang, Wei Li, Jingwen Li, Wenhai Wang, Kai Chen, Conghui He, Xingcheng Zhang, Jifeng Dai, Yu Qiao, Dahua Lin, and Jiaqi Wang. Internlm-xcomposer2-4khd: A pioneering large vision-language model handling resolutions from 336 pixels to 4k hd. _arXiv preprint arXiv:2404.06512_ , 2024. 2, 3, 4 

- [26] Alexey Dosovitskiy, Lucas Beyer, Alexander Kolesnikov, Dirk Weissenborn, Xiaohua Zhai, Thomas Unterthiner, Mostafa Dehghani, Matthias Minderer, Georg Heigold, Sylvain Gelly, et al. An image is worth 16x16 words: Transformers for image recognition at scale. _ICLR_ , 2021. 3 

- [27] Zhongbin Fang, Xiangtai Li, Xia Li, Joachim M Buhmann, Chen Change Loy, and Mengyuan Liu. Explore in-context learning for 3d point cloud understanding. _NeurIPS_ , 2024. 4 

- [28] Hao Fei, Shengqiong Wu, Hanwang Zhang, Tat-Seng Chua, and Shuicheng Yan. Vitron: A unified pixel-level vision llm for understanding, generating, segmenting, editing. _arxiv preprint_ , 2024. 4 

- [29] Guang Feng, Zhiwei Hu, Lihe Zhang, and Huchuan Lu. Encoder fusion network with co-attention embedding for referring image segmentation. In _CVPR_ , 2021. 4 

- [30] Chaoyou Fu, Peixian Chen, Yunhang Shen, Yulei Qin, Mengdan Zhang, Xu Lin, Jinrui Yang, Xiawu Zheng, Ke Li, Xing Sun, Yunsheng Wu, and Rongrong Ji. Mme: A comprehensive evaluation benchmark for multimodal large language models. _arXiv preprint arXiv:2306.13394_ , 2023. 3, 16, 17 

- [31] Xiuye Gu, Yin Cui, Jonathan Huang, Abdullah Rashwan, Xuan Yang, Xingyi Zhou, Golnaz Ghiasi, Weicheng Kuo, Huizhong Chen, Liang-Chieh Chen, et al. Dataseg: Taming a universal multi-dataset multi-task segmentation model. _arXiv preprint arXiv:2306.01736_ , 2023. 3 

- [32] Meng-Hao Guo, Cheng-Ze Lu, Qibin Hou, Zhengning Liu, Ming-Ming Cheng, and Shi-Min Hu. Segnext: Rethinking convolutional attention design for semantic segmentation. _NeurIPS_ , 2022. 3 

- [33] Junwen He, Yifan Wang, Lijun Wang, Huchuan Lu, Jun-Yan He, Jin-Peng Lan, Bin Luo, and Xuansong Xie. Multi-modal instruction tuned llms with fine-grained visual perception. _arXiv preprint arXiv:2403.02969_ , 2024. 4, 8 

- [34] Kaiming He, Georgia Gkioxari, Piotr Dollár, and Ross Girshick. Mask r-cnn. In _ICCV_ , 2017. 3 

- [35] Edward J Hu, Yelong Shen, Phillip Wallis, Zeyuan Allen-Zhu, Yuanzhi Li, Shean Wang, Lu Wang, and Weizhu Chen. Lora: Low-rank adaptation of large language models. _arXiv preprint arXiv:2106.09685_ , 2021. 7, 9 

- [36] Kuan-Chih Huang, Xiangtai Li, Lu Qi, Shuicheng Yan, and Ming-Hsuan Yang. Reason3d: Searching and reasoning 3d segmentation via large language model. _arXiv preprint arXiv:2405.17427_ , 2024. 3 

- [37] Drew A Hudson and Christopher D Manning. Gqa: A new dataset for real-world visual reasoning and compositional question answering. In _CVPR_ , 2019. 3 

11 

- [38] Louis Martin Hugo Touvron, Kevin Stone, Peter Albert, Amjad Almahairi, Yasmine Babaei, Nikolay Bashlykov, Soumya Batra, Prajjwal Bhargava, Shruti Bhosale, and et al. Llama 2: Open foundation and fine-tuned chat models. _arXiv:2307.09288_ , 2023. 1, 3 

- [39] Jitesh Jain, Jiachen Li, MangTik Chiu, Ali Hassani, Nikita Orlov, and Humphrey Shi. OneFormer: One Transformer to Rule Universal Image Segmentation. _CVPR_ , 2023. 3 

- [40] Sahar Kazemzadeh, Vicente Ordonez, Mark Matten, and Tamara Berg. Referitgame: Referring to objects in photographs of natural scenes. In _EMNLP_ , pages 787–798, 2014. 8 

- [41] Aniruddha Kembhavi, Mike Salvato, Eric Kolve, Minjoon Seo, Hannaneh Hajishirzi, and Ali Farhadi. A diagram is worth a dozen images. In _ECCV_ , pages 235–251. Springer, 2016. 16, 17 

- [42] Anna Khoreva, Anna Rohrbach, and Bernt Schiele. Video object segmentation with language referring expressions. In _ACCV_ , 2018. 3 

- [43] Dahun Kim, Jun Xie, Huiyu Wang, Siyuan Qiao, Qihang Yu, Hong-Seok Kim, Hartwig Adam, In So Kweon, and Liang-Chieh Chen. Tubeformer-deeplab: Video mask transformer. In _CVPR_ , 2022. 3 

- [44] Alexander Kirillov, Kaiming He, Ross Girshick, Carsten Rother, and Piotr Dollár. Panoptic segmentation. In _CVPR_ , 2019. 3 

- [45] Alexander Kirillov, Eric Mintun, Nikhila Ravi, Hanzi Mao, Chloe Rolland, Laura Gustafson, Tete Xiao, Spencer Whitehead, Alexander C Berg, Wan-Yen Lo, et al. Segment anything. _ICCV_ , 2023. 1, 4, 6 

- [46] Alina Kuznetsova, Hassan Rom, Neil Alldrin, Jasper Uijlings, Ivan Krasin, Jordi Pont-Tuset, Shahab Kamali, Stefan Popov, Matteo Malloci, Alexander Kolesnikov, et al. The open images dataset v4: Unified image classification, object detection, and visual relationship detection at scale. _IJCV_ , 2020. 19 

- [47] Xin Lai, Zhuotao Tian, Yukang Chen, Yanwei Li, Yuhui Yuan, Shu Liu, and Jiaya Jia. Lisa: Reasoning segmentation via large language model. _arXiv preprint arXiv:2308.00692_ , 2023. 1, 2, 3, 4, 6, 7, 8, 9, 16, 17, 18, 19 

- [48] Bohao Li, Rui Wang, Guangzhi Wang, Yuying Ge, Yixiao Ge, and Ying Shan. Seed-bench: Benchmarking multimodal llms with generative comprehension. _arXiv preprint arXiv:2307.16125_ , 2023. 16, 17 

- [49] Feng Li, Qing Jiang, Hao Zhang, Tianhe Ren, Shilong Liu, Xueyan Zou, Huaizhe Xu, Hongyang Li, Chunyuan Li, Jianwei Yang, et al. Visual in-context prompting. _arXiv preprint arXiv:2311.13601_ , 2023. 4 

- [50] Junnan Li, Dongxu Li, Silvio Savarese, and Steven Hoi. Blip-2: Bootstrapping language-image pretraining with frozen image encoders and large language models, 2023. 3 

- [51] Junnan Li, Dongxu Li, Caiming Xiong, and Steven Hoi. Blip: Bootstrapping language-image pre-training for unified vision-language understanding and generation. In _ICML_ , 2022. 3 

- [52] Xiangtai Li, Henghui Ding, Wenwei Zhang, Haobo Yuan, Guangliang Cheng, Pang Jiangmiao, Kai Chen, Ziwei Liu, and Chen Change Loy. Transformer-based visual segmentation: A survey. _arXiv pre-print_ , 2023. 3 

- [53] Xiang Li, Jinglu Wang, Xiaohao Xu, Xiao Li, Bhiksha Raj, and Yan Lu. Robust referring video object segmentation with cyclic structural consensus. In _ICCV_ , 2023. 4 

- [54] Xiangtai Li, Shilin Xu, Yibo Yang, Guangliang Cheng, Yunhai Tong, and Dacheng Tao. Panopticpartformer: Learning a unified model for panoptic part segmentation. In _ECCV_ , 2022. 3 

- [55] Xiangtai Li, Ansheng You, Zeping Zhu, Houlong Zhao, Maoke Yang, Kuiyuan Yang, and Yunhai Tong. Semantic flow for fast and accurate scene parsing. In _ECCV_ , 2020. 3 

- [56] Xiangtai Li, Haobo Yuan, Wei Li, Henghui Ding, Size Wu, Wenwei Zhang, Yining Li, Kai Chen, and Chen Change Loy. Omg-seg: Is one model good enough for all segmentation? _CVPR_ , 2024. 1, 2, 3, 4, 5, 6, 7, 8, 9, 19 

- [57] Xiangtai Li, Haobo Yuan, Wenwei Zhang, Guangliang Cheng, Jiangmiao Pang, and Chen Change Loy. Tube-link: A flexible cross tube baseline for universal video segmentation. In _ICCV_ , 2023. 3 

- [58] Xiangtai Li, Li Zhang, Guangliang Cheng, Kuiyuan Yang, Yunhai Tong, Xiatian Zhu, and Tao Xiang. Global aggregation then local distribution for scene parsing. _IEEE TIP_ , 2021. 3 

- [59] Xiangtai Li, Wenwei Zhang, Jiangmiao Pang, Kai Chen, Guangliang Cheng, Yunhai Tong, and Chen Change Loy. Video k-net: A simple, strong, and unified baseline for video segmentation. In _CVPR_ , 2022. 3 

- [60] Yifan Li, Yifan Du, Kun Zhou, Jinpeng Wang, Wayne Xin Zhao, and Ji-Rong Wen. Evaluating object hallucination in large vision-language models. _EMNLP_ , 2023. 3 

- [61] Yifan Li, Yifan Du, Kun Zhou, Jinpeng Wang, Wayne Xin Zhao, and Ji-Rong Wen. Evaluating object hallucination in large vision-language models. 2023. 16, 17 

- [62] Yanwei Li, Chengyao Wang, and Jiaya Jia. Llama-vid: An image is worth 2 tokens in large language models. _arXiv preprint arXiv:2311.17043_ , 2023. 1, 2, 4 

- [63] Yanwei Li, Yuechen Zhang, Chengyao Wang, Zhisheng Zhong, Yixin Chen, Ruihang Chu, Shaoteng Liu, and Jiaya Jia. Mini-gemini: Mining the potential of multi-modality vision language models. _arXiv preprint arXiv:2403.18814_ , 2024. 2, 3, 4 

- [64] Bin Lin, Bin Zhu, Yang Ye, Munan Ning, Peng Jin, and Li Yuan. Video-llava: Learning united visual representation by alignment before projection. _arXiv preprint arXiv:2311.10122_ , 2023. 3 

- [65] Weifeng Lin, Xinyu Wei, Ruichuan An, Peng Gao, Bocheng Zou, Yulin Luo, Siyuan Huang, Shanghang Zhang, and Hongsheng Li. Draw-and-understand: Leveraging visual prompts to enable mllms to comprehend what you want. _arXiv preprint arXiv:2403.20271_ , 2024. 2, 4, 8 

- [66] Chang Liu, Henghui Ding, and Xudong Jiang. GRES: Generalized referring expression segmentation. In _CVPR_ , 2023. 3 

12 

- [67] Haotian Liu, Chunyuan Li, Yuheng Li, and Yong Jae Lee. Improved baselines with visual instruction tuning, 2023. 3, 4, 16, 17 

- [68] Haotian Liu, Chunyuan Li, Yuheng Li, Bo Li, Yuanhan Zhang, Sheng Shen, and Yong Jae Lee. Llava-next: Improved reasoning, ocr, and world knowledge, January 2024. 1, 2, 3, 4, 8 

- [69] Haotian Liu, Chunyuan Li, Qingyang Wu, and Yong Jae Lee. Visual instruction tuning. In _NeurIPS_ , 2023. 1, 2, 3, 4, 7, 8, 9 

- [70] Shilong Liu, Hao Cheng, Haotian Liu, Hao Zhang, Feng Li, Tianhe Ren, Xueyan Zou, Jianwei Yang, Hang Su, Jun Zhu, et al. Llava-plus: Learning to use tools for creating multimodal agents. _arXiv preprint arXiv:2311.05437_ , 2023. 1, 2, 3, 4, 8 

- [71] Shilong Liu, Zhaoyang Zeng, Tianhe Ren, Feng Li, Hao Zhang, Jie Yang, Chunyuan Li, Jianwei Yang, Hang Su, Jun Zhu, et al. Grounding dino: Marrying dino with grounded pre-training for open-set object detection. _arXiv preprint arXiv:2303.05499_ , 2023. 3 

- [72] Yuan Liu, Haodong Duan, Yuanhan Zhang, Bo Li, Songyang Zhang, Wangbo Zhao, Yike Yuan, Jiaqi Wang, Conghui He, Ziwei Liu, et al. Mmbench: Is your multi-modal model an all-around player? _arXiv preprint arXiv:2307.06281_ , 2023. 3, 16, 17 

- [73] Zhuang Liu, Hanzi Mao, Chao-Yuan Wu, Christoph Feichtenhofer, Trevor Darrell, and Saining Xie. A convnet for the 2020s. _CVPR_ , 2022. 6, 8 

- [74] Pan Lu, Swaroop Mishra, Tony Xia, Liang Qiu, Kai-Wei Chang, Song-Chun Zhu, Oyvind Tafjord, Peter Clark, and Ashwin Kalyan. Learn to explain: Multimodal reasoning via thought chains for science question answering. In _The 36th Conference on Neural Information Processing Systems (NeurIPS)_ , 2022. 17 

- [75] Gen Luo, Yiyi Zhou, Xiaoshuai Sun, Liujuan Cao, Chenglin Wu, Cheng Deng, and Rongrong Ji. Multitask collaborative network for joint referring expression comprehension and segmentation. In _CVPR_ , 2020. 4 

- [76] Chuofan Ma, Yi Jiang, Jiannan Wu, Zehuan Yuan, and Xiaojuan Qi. Groma: Localized visual tokenization for grounding multimodal large language models. _arXiv preprint arXiv:2404.13013_ , 2024. 2, 4 

- [77] Sachin Mehta and Mohammad Rastegari. Mobilevit: Light-weight, general-purpose, and mobile-friendly vision transformer. In _ICLR_ , 2022. 3 

- [78] Fausto Milletari, Nassir Navab, and Seyed-Ahmad Ahmadi. V-net: Fully convolutional neural networks for volumetric medical image segmentation. In _3DV_ , 2016. 7 

- [79] Ting Pan, Lulu Tang, Xinlong Wang, and Shiguang Shan. Tokenize anything via prompting. _arXiv preprint arXiv:2312.09128_ , 2023. 4 

- [80] Zhiliang Peng, Wenhui Wang, Li Dong, Yaru Hao, Shaohan Huang, Shuming Ma, and Furu Wei. Kosmos2: Grounding multimodal large language models to the world. _arXiv preprint arXiv:2306.14824_ , 2023. 2, 4, 8 

- [81] Lu Qi, Yi-Wen Chen, Lehan Yang, Tiancheng Shen, Xiangtai Li, Weidong Guo, Yu Xu, and MingHsuan Yang. Generalizable entity grounding via assistance of large language model. _arXiv preprint arXiv:2402.02555_ , 2024. 3 

- [82] Lu Qi, Jason Kuen, Tiancheng Shen, Jiuxiang Gu, Weidong Guo, Jiaya Jia, Zhe Lin, and Ming-Hsuan Yang. High-quality entity segmentation. In _ICCV_ , 2023. 3 

- [83] Lu Qi, Jason Kuen, Yi Wang, Jiuxiang Gu, Hengshuang Zhao, Philip Torr, Zhe Lin, and Jiaya Jia. Open world entity segmentation. _TPAMI_ , 2022. 3 

- [84] Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, et al. Learning transferable visual models from natural language supervision. In _ICML_ , 2021. 1, 6 

- [85] Hanoona Rasheed, Muhammad Maaz, Sahal Shaji, Abdelrahman Shaker, Salman Khan, Hisham Cholakkal, Rao M. Anwer, Eric Xing, Ming-Hsuan Yang, and Fahad S. Khan. Glamm: Pixel grounding large multimodal model. _CVPR_ , 2024. 1, 2, 3, 4, 5, 7, 8, 9, 16, 17, 18, 19 

- [86] Zhongwei Ren, Zhicheng Huang, Yunchao Wei, Yao Zhao, Dongmei Fu, Jiashi Feng, and Xiaojie Jin. Pixellm: Pixel reasoning with large multimodal model. _arXiv preprint arXiv:2312.02228_ , 2023. 1, 3, 4, 8, 9, 16 

- [87] Shuai Shao, Zeming Li, Tianyuan Zhang, Chao Peng, Gang Yu, Xiangyu Zhang, Jing Li, and Jian Sun. Objects365: A large-scale, high-quality dataset for object detection. In _CVPR_ , 2019. 19 

- [88] Quan Sun, Yufeng Cui, Xiaosong Zhang, Fan Zhang, Qiying Yu, Zhengxiong Luo, Yueze Wang, Yongming Rao, Jingjing Liu, Tiejun Huang, et al. Generative multimodal models are in-context learners. _arXiv:2312.13286_ , 2023. 4 

- [89] Shuyang Sun, Weijun Wang, Andrew Howard, Qihang Yu, Philip Torr, and Liang-Chieh Chen. Remax: Relaxing for better training on efficient panoptic segmentation. _NeurIPS_ , 2023. 3 

- [90] Gemini Team, Rohan Anil, Sebastian Borgeaud, Yonghui Wu, Jean-Baptiste Alayrac, Jiahui Yu, Radu Soricut, Johan Schalkwyk, Andrew M Dai, Anja Hauth, et al. Gemini: a family of highly capable multimodal models. _arXiv preprint arXiv:2312.11805_ , 2023. 1, 3 

- [91] Hugo Touvron, Thibaut Lavril, Gautier Izacard, Xavier Martinet, Marie-Anne Lachaux, Timothée Lacroix, Baptiste Rozière, Naman Goyal, Eric Hambro, Faisal Azhar, Aurelien Rodriguez, Armand Joulin, Edouard Grave, and Guillaume Lample. Llama: Open and efficient foundation language models. _arXiv:2302.13971_ , 2023. 1, 3 

- [92] Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez, Lukasz Kaiser, and Illia Polosukhin. Attention is all you need. In _NIPS_ , 2017. 1, 3 

13 

- [93] Huiyu Wang, Yukun Zhu, Hartwig Adam, Alan Yuille, and Liang-Chieh Chen. Max-deeplab: End-to-end panoptic segmentation with mask transformers. In _CVPR_ , 2021. 3 

- [94] Junchi Wang and Lei Ke. Llm-seg: Bridging image segmentation and large language model reasoning. _arXiv preprint arXiv:2404.08767_ , 2024. 2 

- [95] Wenhai Wang, Zhe Chen, Xiaokang Chen, Jiannan Wu, Xizhou Zhu, Gang Zeng, Ping Luo, Tong Lu, Jie Zhou, Yu Qiao, and Jifeng Dai. Visionllm: Large language model is also an open-ended decoder for vision-centric tasks. _arXiv preprint arXiv:2305.11175_ , 2023. 4 

- [96] Xinshun Wang, Zhongbin Fang, Xia Li, Xiangtai Li, and Mengyuan Liu. Skeleton-in-context: Unified skeleton sequence modeling with in-context learning. _CVPR_ , 2024. 4 

- [97] Xinlong Wang, Wen Wang, Yue Cao, Chunhua Shen, and Tiejun Huang. Images speak in images: A generalist painter for in-context visual learning. In _CVPR_ , 2023. 1, 4 

- [98] Xinlong Wang, Xiaosong Zhang, Yue Cao, Wen Wang, Chunhua Shen, and Tiejun Huang. Seggpt: Segmenting everything in context. In _ICCV_ , 2023. 1, 4 

- [99] Zhendong Wang, Yifan Jiang, Yadong Lu, Yelong Shen, Pengcheng He, Weizhu Chen, Zhangyang Wang, and Mingyuan Zhou. In-context learning unlocked for diffusion models. _arXiv preprint arXiv:2305.01115_ , 2023. 4 

- [100] Cong Wei, Haoxian Tan, Yujie Zhong, Yujiu Yang, and Lin Ma. Lasagna: Language-based segmentation assistant for complex queries. _arXiv preprint arXiv:2404.08506_ , 2024. 4, 8, 16 

- [101] Jianzong Wu, Xiangtai Li, Xia Li, Henghui Ding, Yunhai Tong, and Dacheng Tao. Towards robust referring image segmentation. _IEEE-TIP_ , 2024. 4 

- [102] Jianzong Wu, Xiangtai Li, Chenyang Si, Shangchen Zhou, Jingkang Yang, Jiangning Zhang, Yining Li, Kai Chen, Yunhai Tong, Ziwei Liu, et al. Towards language-driven video inpainting via multimodal large language models. _CVPR_ , 2024. 3 

- [103] Jianzong Wu, Xiangtai Li, Shilin Xu, Haobo Yuan, Henghui Ding, Yibo Yang, Xia Li, Jiangning Zhang, Yunhai Tong, Xudong Jiang, Bernard Ghanem, and Dacheng Tao. Towards open vocabulary learning: A survey. _arXiv pre-print_ , 2023. 3, 4 

- [104] Shengqiong Wu, Hao Fei, Xiangtai Li, Jiayi Ji, Hanwang Zhang, Tat-Seng Chua, and Shuicheng Yan. Towards semantic equivalence of tokenization in multimodal llm. _arXiv preprint arXiv:2406.05127_ , 2024. 3 

- [105] Shengqiong Wu, Hao Fei, Leigang Qu, Wei Ji, and Tat-Seng Chua. Next-gpt: Any-to-any multimodal llm. _arXiv preprint arXiv:2309.05519_ , 2023. 2 

- [106] Tsung-Han Wu, Giscard Biamby, David Chan, Lisa Dunlap, Ritwik Gupta, Xudong Wang, Joseph E Gonzalez, and Trevor Darrell. See, say, and segment: Teaching lmms to overcome false premises. _arXiv preprint arXiv:2312.08366_ , 2023. 4 

- [107] Zhuofan Xia, Dongchen Han, Yizeng Han, Xuran Pan, Shiji Song, and Gao Huang. Gsva: Generalized segmentation via multimodal large language models. _arXiv preprint arXiv:2312.10103_ , 2023. 4, 8 

- [108] Ruyi Xu, Yuan Yao, Zonghao Guo, Junbo Cui, Zanlin Ni, Chunjiang Ge, Tat-Seng Chua, Zhiyuan Liu, Maosong Sun, and Gao Huang. Llava-uhd: an lmm perceiving any aspect ratio and high-resolution images. _arXiv preprint arXiv:2403.11703_ , 2024. 2, 4 

- [109] Shilin Xu, Haobo Yuan, Qingyu Shi, Lu Qi, Jingbo Wang, Yibo Yang, Yining Li, Kai Chen, Yunhai Tong, Bernard Ghanem, Xiangtai Li, and Ming-Hsuan Yang. Rap-sam: Towards real-time all-purpose segment anything. _arXiv preprint_ , 2024. 3 

- [110] Bin Yan, Yi Jiang, Jiannan Wu, Dong Wang, Zehuan Yuan, Ping Luo, and Huchuan Lu. Universal instance perception as object discovery and retrieval. In _CVPR_ , 2023. 1, 3 

- [111] An Yang, Baosong Yang, Binyuan Hui, Bo Zheng, Bowen Yu, Chang Zhou, Chengpeng Li, Chengyuan Li, Dayiheng Liu, Fei Huang, Guanting Dong, Haoran Wei, Huan Lin, Jialong Tang, Jialin Wang, Jian Yang, Jianhong Tu, Jianwei Zhang, Jianxin Ma, Jin Xu, Jingren Zhou, Jinze Bai, Jinzheng He, Junyang Lin, Kai Dang, Keming Lu, Keqin Chen, Kexin Yang, Mei Li, Mingfeng Xue, Na Ni, Pei Zhang, Peng Wang, Ru Peng, Rui Men, Ruize Gao, Runji Lin, Shijie Wang, Shuai Bai, Sinan Tan, Tianhang Zhu, Tianhao Li, Tianyu Liu, Wenbin Ge, Xiaodong Deng, Xiaohuan Zhou, Xingzhang Ren, Xinyu Zhang, Xipin Wei, Xuancheng Ren, Yang Fan, Yang Yao, Yichang Zhang, Yu Wan, Yunfei Chu, Yuqiong Liu, Zeyu Cui, Zhenru Zhang, and Zhihao Fan. Qwen2 technical report. _arXiv preprint arXiv:2407.10671_ , 2024. 16, 17 

- [112] Senqiao Yang, Tianyuan Qu, Xin Lai, Zhuotao Tian, Bohao Peng, Shu Liu, and Jiaya Jia. An improved baseline for reasoning segmentation with large language model. _arXiv preprint arXiv:2312.17240_ , 2023. 4 

- [113] Zongxin Yang, Jiaxu Miao, Yunchao Wei, Wenguan Wang, Xiaohan Wang, and Yi Yang. Scalable video object segmentation with identification mechanism. _TPAMI_ , 2024. 3 

- [114] Zhao Yang, Jiaqi Wang, Yansong Tang, Kai Chen, Hengshuang Zhao, and Philip HS Torr. Lavt: Languageaware vision transformer for referring image segmentation. In _CVPR_ , 2022. 4 

- [115] Zongxin Yang, Yunchao Wei, and Yi Yang. Collaborative video object segmentation by multi-scale foreground-background integration. _TPAMI_ , 2021. 3 

- [116] Qinghao Ye, Haiyang Xu, Guohai Xu, Jiabo Ye, Ming Yan, Yiyang Zhou, Junyang Wang, Anwen Hu, Pengcheng Shi, Yaya Shi, Chenliang Li, Yuanhong Xu, Hehong Chen, Junfeng Tian, Qi Qian, Ji Zhang, Fei Huang, and Jingren Zhou. mplug-owl: Modularization empowers large language models with multimodality, 2023. 4 

14 

- [117] Haoxuan You, Haotian Zhang, Zhe Gan, Xianzhi Du, Bowen Zhang, Zirui Wang, Liangliang Cao, Shih-Fu Chang, and Yinfei Yang. Ferret: Refer and ground anything anywhere at any granularity. _ICLR_ , 2024. 2, 4 

- [118] Licheng Yu, Zhe Lin, Xiaohui Shen, Jimei Yang, Xin Lu, Mohit Bansal, and Tamara L Berg. Mattnet: Modular attention network for referring expression comprehension. In _CVPR_ , 2018. 3 

- [119] Licheng Yu, Patrick Poirson, Shan Yang, Alexander C Berg, and Tamara L Berg. Modeling context in referring expressions. In _ECCV_ , pages 69–85. Springer, 2016. 8 

- [120] Qihang Yu, Huiyu Wang, Dahun Kim, Siyuan Qiao, Maxwell Collins, Yukun Zhu, Hartwig Adam, Alan Yuille, and Liang-Chieh Chen. Cmt-deeplab: Clustering mask transformers for panoptic segmentation. In _CVPR_ , 2022. 3 

- [121] Qihang Yu, Huiyu Wang, Siyuan Qiao, Maxwell Collins, Yukun Zhu, Hartwig Adam, Alan Yuille, and Liang-Chieh Chen. k-means mask transformer. In _ECCV_ , 2022. 3 

- [122] Qihang Yu, Huiyu Wang, Siyuan Qiao, Maxwell Collins, Yukun Zhu, Hartwig Adam, Alan Yuille, and Liang-Chieh Chen. k-means Mask Transformer. In _ECCV_ , 2022. 3 

- [123] Haobo Yuan, Xiangtai Li, Chong Zhou, Yining Li, Kai Chen, and Chen Change Loy. Open-vocabulary sam: Segment and recognize twenty-thousand classes interactively. _arXiv preprint_ , 2024. 3, 4 

- [124] Yuqian Yuan, Wentong Li, Jian Liu, Dongqi Tang, Xinjie Luo, Chi Qin, Lei Zhang, and Jianke Zhu. Osprey: Pixel understanding with visual instruction tuning. _arXiv preprint arXiv:2312.10032_ , 2023. 1, 2, 3, 4, 5, 8 

- [125] Yuhang Zang, Wei Li, Jun Han, Kaiyang Zhou, and Chen Change Loy. Contextual object detection with multimodal large language models. _arXiv preprint arXiv:2305.18279_ , 2023. 3 

- [126] Ao Zhang, Liming Zhao, Chen-Wei Xie, Yun Zheng, Wei Ji, and Tat-Seng Chua. Next-chat: An lmm for chat, detection and segmentation. _arXiv preprint arXiv:2311.04498_ , 2023. 2, 8 

- [127] Hao Zhang, Hongyang Li, Feng Li, Tianhe Ren, Xueyan Zou, Shilong Liu, Shijia Huang, Jianfeng Gao, Lei Zhang, Chunyuan Li, and Jianwei Yang. Llava-grounding: Grounded visual chat with large multimodal models, 2023. 3 

- [128] Hang Zhang, Xin Li, and Lidong Bing. Video-llama: An instruction-tuned audio-visual language model for video understanding. _arXiv preprint arXiv:2306.02858_ , 2023. 3 

- [129] Pan Zhang, Xiaoyi Dong, Bin Wang, Yuhang Cao, Chao Xu, Linke Ouyang, Zhiyuan Zhao, Shuangrui Ding, Songyang Zhang, Haodong Duan, Wenwei Zhang, Hang Yan, Xinyue Zhang, Wei Li, Jingwen Li, Kai Chen, Conghui He, Xingcheng Zhang, Yu Qiao, Dahua Lin, and Jiaqi Wang. Internlm-xcomposer: A vision-language large model for advanced text-image comprehension and composition. _arXiv preprint arXiv:2309.15112_ , 2023. 2, 3, 4 

- [130] Renrui Zhang, Jiaming Han, Chris Liu, Peng Gao, Aojun Zhou, Xiangfei Hu, Shilin Yan, Pan Lu, Hongsheng Li, and Yu Qiao. Llama-adapter: Efficient fine-tuning of language models with zero-init attention. _arXiv preprint arXiv:2303.16199_ , 2023. 4 

- [131] Shilong Zhang, Peize Sun, Shoufa Chen, Min Xiao, Wenqi Shao, Wenwei Zhang, Kai Chen, and Ping Luo. Gpt4roi: Instruction tuning large language model on region-of-interest. _arXiv preprint arXiv:2307.03601_ , 2023. 2, 4 

- [132] Yichi Zhang, Ziqiao Ma, Xiaofeng Gao, Suhaila Shakiah, Qiaozi Gao, and Joyce Chai. Groundhog: Grounding large language models to holistic segmentation. In _CVPR_ , pages 14227–14238, 2024. 4 

- [133] Zheng Zhang, Yeyao Ma, Enming Zhang, and Xiang Bai. Psalm: Pixelwise segmentation with large multi-modal model. _arXiv preprint arXiv:2403.14598_ , 2024. 2, 3, 4 

- [134] Xiangyu Zhao, Xiangtai Li, Haodong Duan, Haian Huang, Yining Li, Kai Chen, and Hua Yang. Towards semantic equivalence of tokenization in multimodal llm. _arXiv preprint_ , 2024. 3 

- [135] Bolei Zhou, Hang Zhao, Xavier Puig, Sanja Fidler, Adela Barriuso, and Antonio Torralba. Semantic understanding of scenes through the ADE20K dataset. _CVPR_ , 2017. 8 

- [136] Kaiyang Zhou, Jingkang Yang, Chen Change Loy, and Ziwei Liu. Learning to prompt for vision-language models. In _IJCV_ , 2022. 4 

- [137] Qianyu Zhou, Zhengyang Feng, Qiqi Gu, Jiangmiao Pang, Guangliang Cheng, Xuequan Lu, Jianping Shi, and Lizhuang Ma. Context-aware mixup for domain adaptive semantic segmentation. _IEEE TCSVT_ , 2023. 3 

- [138] Qianyu Zhou, Xiangtai Li, Lu He, Yibo Yang, Guangliang Cheng, Yunhai Tong, Lizhuang Ma, and Dacheng Tao. Transvod: End-to-end video object detection with spatial-temporal transformers. _TPAMI_ , 2022. 3 

- [139] Yikang Zhou, Tao Zhang, Shunping Ji, Shuicheng Yan, and Xiangtai Li. Dvis-daq: Improving video segmentation via dynamic anchor queries. _arXiv preprint arXiv:2404.00086_ , 2024. 3 

- [140] Deyao Zhu, Jun Chen, Xiaoqian Shen, Xiang Li, and Mohamed Elhoseiny. Minigpt-4: Enhancing vision-language understanding with advanced large language models. _arXiv preprint arXiv:2304.10592_ , 2023. 4 

15 

Table 6: Performance on image-level benchmarks. 

|Method<br>MME [30]<br>MMBench [72]<br>SEED-Bench [48]<br>POPE [61]<br>AI2D [41]|Method<br>MME [30]<br>MMBench [72]<br>SEED-Bench [48]<br>POPE [61]<br>AI2D [41]|
|---|---|
|Trainingonlywith LLaVA dataset||
|LLaVA 1.5 [67]<br>OMG-LLaVA|1422/267<br>68.5<br>65.9<br>86.7<br>56.6<br>1448/282<br>67.5<br>68.9<br>89.7<br>61.7|
|Co-trainingwith LLaVA dataset and segmentation datasets||
|LISA [47]<br>PixelLM [86]<br>LaSagnA [100]<br>GLaMM [85]<br>OMG-LLaVA|1/1<br>0.4<br>-<br>0.0<br>0.0<br>309/135<br>17.4<br>-<br>0.0<br>0.0<br>0/0<br>0.0<br>-<br>0.0<br>0.0<br>14/9<br>36.8<br>-<br>0.94<br>28.2<br>1177/235<br>47.9<br>56.5<br>80.0<br>42.9|



Table 7: Performance with different LLMs. 

|LLM|refCOCO<br>refCOCO+<br>CIoU<br>GIoU<br>CIoU<br>GIoU|MME<br>MMBench<br>SEED-Bench<br>POPE<br>AI2D<br>MMstar<br>SQA<br>perception<br>reasoning|
|---|---|---|
|Phi3-3.8B [1]<br>InternLM2-7B [9]<br>Qwen2-7B [111]|76.5<br>78.0<br>67.8<br>70.0<br>76.3<br>77.8<br>67.7<br>69.9<br>76.7<br>78.2<br>69.1<br>71.2|1291.6<br>265.0<br>59.6<br>60.6<br>86.7<br>56.9<br>37.1<br>64.7<br>1177.1<br>235.4<br>47.9<br>56.5<br>80.0<br>42.9<br>33.1<br>57.8<br>1215.7<br>251.1<br>62.8<br>60.7<br>84.3<br>52.6<br>37.2<br>66.4|



## **A Appendix** 

**Overview.** In this appendix, we will first give more implementation and training details of our method. Then, we present more detailed ablation studies on several component designs. Next, we present more detailed visualization results. In the end, we discuss the limitations and future work. 

## **A.1 More Implementation Details** 

**Pre-training.** Following LLaVA, OMG-LLaVA first performs pre-training to learn the projector that projects visual tokens into the text space. During the pre-training stage, we freeze the visual encoder, OMG head, and LLM to train the visual projector for projecting visual tokens into the text space and to train the text projector for restoring the projected object-centric visual tokens to the segmentation embedding. The training data used in the pre-training stage is the same as that used in LLaVA. In this stage, the OMG-LLaVA is trained for 1 epoch. The batch size is 256, with 32 per GPU, and the learning rate is 0.001. 

**Supervised fine-tuning.** During the instruction tuning stage, we freeze the visual encoder and OMG head, finetune the LLM using LoRA, and fully finetune the text and visual projectors. We train OMG-LLaVA for 1 epoch on all instruction tuning datasets, including the LLaVA instruction tuning dataset, referring expression segmentation datasets, semantic segmentation datasets, grounded conversation generation datasets, mask-based visual prompt datasets, and point-based visual prompt datasets. The batch size is 128, with 16 per GPU, and the learning rate is 2e-4. 

**Inference details for each task.** OMG-LLaVA generates answers token by token during the inference stage based on the given question. We use a fixed template for the referring expression segmentation task to create the question: “ _Please segment {EXPRESSION} in this image._ " In rare cases where OMG-LLaVA does not predict the [SEG] token, we use an empty mask as the segmentation result. We use the fixed question for the grounded conversation generation task: “ _Could you please give me a detailed description of the image? Please respond with interleaved segmentation masks for the corresponding parts of the answer._ " For other tasks, we remove special tokens such as <p>, </p>, and [SEG] from OMG-LLaVA’s responses to ensure the answers contain only text. 

## **A.2 More Experiment Results.** 

**Evaluation results on image-level benchmarks.** We evaluate OMG-LLaVA on several image-level benchmarks, including MME [30], MMBench [72], SEED-Bench [48], POPE [61], and AI2D [41] benchmarks. The evaluation results are shown in Tab. 6. When jointly Co-training with imagelevel and pixel-level datasets, OMG-LLaVA achieves 1412, 47.9, 46.5, 80.0 and 42.9 on MME, MMBench, SEED-Bench, POPE and AI2D benchmarks, respectively. Compared with GLaMM [85], PixelLM [86], and LISA [47], OMG-LLaVA demonstrates significant performance improvement. 

16 

Table 8: Ablation study of projector for object-centric visual tokens. 

|Methods|Cross Attn.<br>Individual|refCOCO<br>refCOCO+<br>refCOCOg<br>cIoU<br>gIoU<br>cIoU<br>gIoU<br>cIoU<br>gIoU|refCOCOg(C)<br>METEOR|
|---|---|---|---|
|Baseline (M0)<br>M1<br>M2|✓<br>✓|74.5<br>75.9<br>63.6<br>65.9<br>68.7<br>71.0<br>72.3<br>73.7<br>60.6<br>63.0<br>66.5<br>69.2<br>72.3<br>74.1<br>60.8<br>63.5<br>65.4<br>68.6|13.6<br>13.2<br>13.1|



Table 9: Ablation study on answer format of segmentation-based tasks. The first row represents the RES task using the fixed answer: “ _Sure, it is [SEG]._ " and the GCG task using “ _<p> Expression </p> [SEG]_ ." The second row represents the segmentation tasks’ answer format being unified as “ _<p> Expression </p> [SEG]_ . 

|Format|refCOCO<br>refCOCO+<br>refCOCOg<br>cIoU<br>gIoU<br>cIoU<br>gIoU<br>cIoU<br>gIoU|refCOCO<br>refCOCO+<br>refCOCOg<br>cIoU<br>gIoU<br>cIoU<br>gIoU<br>cIoU<br>gIoU|GCG<br>METEOR<br>AP50<br>mIOU|
|---|---|---|---|
|It is [SEG].<br><p> Expression </p> [SEG].|75.5<br>76.5<br>65.8<br>67.8<br>70.6<br>72.3<br>75.6<br>76.8<br>65.6<br>67.6<br>70.7<br>72.6||13.8<br>27.3<br>64.4<br>13.8<br>26.9<br>64.6|
|Table 10: Ablation study on segmentation embeddings.||||
|refCOCO<br>refCOCO+<br>refCOCOg<br>cIoU<br>gIoU<br>cIoU<br>gIoU<br>cIoU<br>gIoU||||
|||||
|Last layer’s hidden state<br>Mean of all layers’ hidden states<br>Concatenate of all layers’ hidden states||74.3<br>75.5<br>64.5<br>66.4<br>70.0<br>71.9<br>74.3<br>75.8<br>64.5<br>66.4<br>68.7<br>71.4<br>70.0<br>71.1<br>61.8<br>63.6<br>62.3<br>64.4||



When training only with the LLaVA [67] dataset, OMG-LLaVA achieves 1730, 67.5, 68.9, 89.7, and 61.7 on MME, MMBench, SEED-Bench, POPE, and AI2D benchmarks. OMG-LLaVA outperforms LLaVA-1.5 [67] with 41, 3.0, 3.0, 5.1 on MME, SEED-Bench, POPE, and AI2D benchmarks with the same training data. 

**Performance with diverse LLMs.** We construct the OMG-LLaVA using diverse LLMs. The performance is shown in Tab. 7. In addition to InternLM2 [9], we have tried using PHI-3.8b [1] and Qwen2-7B [111], which achieved better performance on pixel-level and image-level benchmarks than InternLM2. When using the stronger Qwen2-7B, OMG-LLaVA achieves 76.7 cIoU and 69.1 cIoU on RefCOCO and RefCOCO+ benchmarks, and 1466.8, 62.8, 60.7, 84.3, 52.6, 37.2, and 66.4 on MME [30], MMBench [72], SEED-Bench [48], POPE [61], AI2D [41], MMstar [15] and SQA [74] benchmarks. 

## **A.3 More Detailed Ablation Studies.** 

**Projector for object-centric visual tokens.** We conducted ablation experiments on the vision projector. The results are shown in Tab. 8. We use a simple MLP projector as the baseline for object-centric visual tokens. When we added a cross-attention layer to the projector, performance on segmentation and visual prompt-based tasks decreased. This is because the introduction of the cross-attention layer caused the object-centric visual tokens to incorporate too many pixel-centric visual tokens, leading to interference with the object information. Furthermore, when the projector for object-centric visual tokens generated from visual prompt input and object queries is not shared, performance declines on segmentation and visual prompt-based tasks. Therefore, a shared MLP projector can effectively project object-centric visual tokens into the text space. 

**Answer format for segmentation-based tasks.** In LISA [47], the response for the referring expression segmentation task is fixed as “ _Sure, it is [SEG]._ " However, this fixed answer may interfere with the instruction-following ability of the LLM, leading it to respond with “ _Sure, it is [SEG]._ " for new instructions. In GLaMM [85], for the grounded conversation generation task, the response is typically “ _<p> Expression </p> [SEG]_ ." Since the “ _Expression_ " is flexible and variable, the LLM is less likely to overfit to a fixed response. 

We conduct ablation experiments on the answer format for segmentation tasks, and the results are shown in Tab. 9. We find that unifying the answer format for segmentation tasks (including RES and GCG) as “ _<p> Expression </p> [SEG]_ " yields better performance. This more flexible answer 

17 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_OMG-LLaVA_Bridging_Image-level,_Object-level,_Pixel-level_Reasoning_and_Understanding_images/nips_2024_OMG-LLaVA_Bridging_Image-level,_Object-level,_Pixel-level_Reasoning_and_Understanding.pdf-0018-00.png)


**----- Start of picture text -----**<br>
the table<br>Express: the grass the sand the smallest chair<br>beside the black sofa<br>Image<br>LISA<br>GLaMM<br>Ours<br>**----- End of picture text -----**<br>


Figure 7: Qualitative comparison on the referring expression segmentation task. LISA uses the 13B LLM, while GLaMM and our proposed OMG-LLaVA use the 7B LLM. 

format not only achieves better performance in the referring expression segmentation task compared to the fixed answer but also avoids the damage to the LLM’s instruction-following ability. 

**Segmentation embeddings.** We conduct ablation experiments on the generation strategy of segmentation embedding, and the results are shown in Tab. 10. We explore whether the hidden states of the intermediate layers corresponding to the [SEG] token are helpful for segmentation. Compared to using the hidden states of the last layer of the [SEG] token as the segmentation embedding, using the mean of the hidden states from all layers as the segmentation embedding resulted in negligible improvement on refCOCO but led to a significant performance drop on the more challenging refCOCOg. Concatenating the hidden states from all layers of the [SEG] token as the segmentation embedding resulted in a significant performance drop across all RES tasks. Therefore, the hidden state of the last layer already contains sufficient features to generate the segmentation mask, and introducing hidden states from other intermediate layers does not yield better segmentation results. 

## **A.4 More Visualization Results** 

**Qualitative comparison with SOTA methods.** We conduct qualitative comparisons and analyses on various tasks, including referring expression segmentation, grounded conversation generation, and image-based conversation, against the SOTA methods LISA [47] and GLaMM [85]. Fig. 7 shows the visualization results of the RES task for LISA, GLaMM, and our proposed OMG-LLaVA. OMGLLaVA demonstrates a more stable segmentation performance than LISA and GLaMM. Additionally, OMG-LLaVA exhibits better image and text understanding capabilities than LISA (13B) and GLaMM, as illustrated in the fourth column with the example of " _the smallest chair_ ". 

18 

Fig. 8 shows the visualization results of the GCG task for GLaMM [85] and OMG-LLaVA. Our proposed OMG-LLaVA provides more detailed and accurate descriptions of the scene, such as “ _lighthouse_ " and “ _bear_ ." Additionally, OMG-LLaVA demonstrates more stable segmentation capabilities, as seen in the “ _mountain_ " in the bottom-right corner image. 

Fig. 9 shows the visualization results of the visual prompt-based description task for GLaMM [85] and OMG-LLaVA. Compared to GLaMM, OMG-LLaVA supports more flexible visual prompts, including point, box, and mask prompts. Additionally, OMG-LLaVA can generate more detailed object captions and demonstrate a more accurate image understanding. 

Fig. 10 shows the visualization results of the image-based conversation task for LISA [47], GLaMM [85], and OMG-LLaVA. Compared to LISA and GLaMM, OMG-LLaVA has stronger instruction-following ability. For example, when answering the question, “ _What is the number on the jersey of the athlete squatting on the ground?_ " both LISA and GLaMM incorrectly segmented “ _the jersey of the athlete squatting on the ground_ ." Compared to GLaMM, OMG-LLaVA can provide more detailed and accurate answers to user questions. Compared to LISA, OMG-LLaVA demonstrates stronger scene understanding and reasoning abilities. For instance, in question 3 of Fig. 10, LISA gave an utterly incorrect answer despite using a larger LLM (13B). 

**Visualization results of RES.** We provide additional visualization results of OMG-LLaVA on the RES task in Fig. 11. OMG-LLaVA demonstrates a strong understanding of spatial relationships and human actions, enabling it to accurately and reliably segment the specified objects based on these descriptions. Furthermore, even without training on any reasoning segmentation data, OMG-LLaVA exhibits the ability to perform reasoning segmentation. As shown in Fig. 12, OMG-LLaVA can infer the target based on the question and accurately segment the corresponding object. 

**Visualization results of GCG.** As depicted in Fig. 13, our method performs well on the grounded conversation generation task. OMG-LLaVA demonstrates strong scene understanding and object segmentation capabilities. Although some objects are overlooked, this is due to the omission of many objects in the image captions of the Grandf dataset. We believe that using higher-quality data for training would result in even better performance for OMG-LLaVA. 

**Visualization results of visual prompts-based description.** Fig. 14 shows more visualization results for the visual prompt-based description task. OMG-LLaVA supports input of point, box, and maskbased visual prompts and provides detailed descriptions. These descriptions include information about the objects and their relationships with other objects in the scene. 

## **A.5 Limitation and Future Work Discussion** 

**Limitations of OMG-LLaVA.** Although OMG-LLaVA achieves image-level, object-level, and pixellevel capabilities with a concise and elegant architecture, much room still exists for improvement. Firstly, joint training with pixel-level understanding data often leads to decreased image-level capability, a phenomenon widely observed in LISA [47] and GLaMM [85]. This challenge could be addressed by organizing the data to eliminate this conflict. Secondly, due to the lack of multi-granularity segmentation capability in OMG-Seg, OMG-LLaVA cannot perform part-level segmentation. This challenge could be addressed using a more powerful and universal perception module by adding part-level visual inputs. 

**Future Works.** Several future directions can be explored with our new meta-architecture. We list two potential directions, including video and more instruction-tuning data. Although OMG-Seg [56] can acquire the video inputs, OMG-LLaVA still cannot perform pixel-level spatial-temporal reasoning. This is due to the lack of such datasets. Moreover, more instruction-tuning data involve more localization outputs, and multiple round conversations can be used to build a stronger MLLM model. For example, we plan to use full GLaMM datasets [85] and more detection datasets [46; 87] for joint co-training as future work if more computation resources are available. 

19 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_OMG-LLaVA_Bridging_Image-level,_Object-level,_Pixel-level_Reasoning_and_Understanding_images/nips_2024_OMG-LLaVA_Bridging_Image-level,_Object-level,_Pixel-level_Reasoning_and_Understanding.pdf-0020-00.png)


**----- Start of picture text -----**<br>
Image<br>GLaMM<br>A black truck is driving on a  The image features a The image showcases a living room with<br>dirt road. The truck is also  building standing on  a chair and a potted plant placed on a<br>parked on the same dirt road.  grass . The sky is  rug . A window is situated beside the<br>The sky is visible over the  overarching the  chair and a wall . The rug is positioned<br>truck. building. on the wooden floor.<br>Ours<br>A black Jeep is parked on a  The image  The image showcases a living room with a large<br>dirt road, surrounded by a  features a  window and a wooden floor with rug. A potted<br>vast sandy landscape. The  lighthouse with  plant is placed on the wooden floor. A chair is<br>scene is set in a desert  a sky overhead.  positioned beside the potted plant. The chair is<br>environment with a clear blue  The sky is also  also situated on the wooden floor. The wooden<br>sky visible. seen over grass. floor is connected to the wall and the window<br>and the door is embedded in the wall.<br>**----- End of picture text -----**<br>



![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_OMG-LLaVA_Bridging_Image-level,_Object-level,_Pixel-level_Reasoning_and_Understanding_images/nips_2024_OMG-LLaVA_Bridging_Image-level,_Object-level,_Pixel-level_Reasoning_and_Understanding.pdf-0020-01.png)


**----- Start of picture text -----**<br>
Image<br>GLaMM<br>The image features a dog  The image showcases a book<br>standing on grass . The dog is  and a cabinet placed on a rug .  A woman in a<br>positioned in front of a  The cabinet is situated beside  black shirt and<br>mountain . The grass is  a wall . A light is also visible,  black pants is<br>attached to the mountain. positioned beside the book. walking in the<br>mountains .<br>Ours<br>The image shows a bear The image showcases a vintage, wooden  A woman stands in<br>standing on grass. The bear is  cabinet with a built-in bookshelf. The  front of a waterfall in<br>positioned in  cabine is positioned on a rug, and a  a mountainous region,<br>front of a mountain. The sky light fixture is hanging from the  surrounded by tall<br>is visible over the mountain. ceiling . The cabinet is also situated  snowy cliffs and<br>beside a table, which is also placed on  greenery, creating a<br>the rug. Book is on the table. The rug  beautiful and serene<br>is on the wooden floor. scene.<br>**----- End of picture text -----**<br>


Figure 8: Qualitative comparison on the grounded conversation generation task. 

20 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_OMG-LLaVA_Bridging_Image-level,_Object-level,_Pixel-level_Reasoning_and_Understanding_images/nips_2024_OMG-LLaVA_Bridging_Image-level,_Object-level,_Pixel-level_Reasoning_and_Understanding.pdf-0021-00.png)


**----- Start of picture text -----**<br>
OMG-LLaVA GlaMM<br>A coffee cup is prominently positioned in the image.  a cup of coffee<br>It's a large cup with a brown handle, and it's placed<br>on a saucer. The cup is filled with a warm beverage,<br>likely coffee, and it's the main focus of the scene.<br>A small, round pastry is located in the middle of the  a macaroon<br>plate. It's the second pastry from the left and is the  on the left<br>second row from the bottom. of the plate<br>A small monkey is seen in<br>the image, hanging from a<br>monkey<br>branch. The monkey appears<br>hanging<br>to be a baby, possibly a baby<br>from tree<br>monkey, and is positioned<br>branch<br>towards the left side of the<br>image.<br>A large, leafy tree is situated in<br>the foreground of the image. It  a tree in<br>appears to be a mature tree,  front of a<br>possibly a pine tree, and is located  house<br>near a house. The tree is the main<br>focus of this region, and it stands<br>out against the backdrop of the<br>house.<br>A silver car is parked on the<br>side of the road, occupying a<br>significant portion of the image.  a grey car<br>The car's color is white, and it  parked on<br>appears to be a sedan. The car  the side of<br>is positioned in front of a blue  the road<br>house, which is also visible in<br>the scene.<br>**----- End of picture text -----**<br>


Figure 9: Qualitative comparison on the visual prompt-based description task. 

21 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_OMG-LLaVA_Bridging_Image-level,_Object-level,_Pixel-level_Reasoning_and_Understanding_images/nips_2024_OMG-LLaVA_Bridging_Image-level,_Object-level,_Pixel-level_Reasoning_and_Understanding.pdf-0022-00.png)


**----- Start of picture text -----**<br>
Could you please give me a<br>detailed description of the image?<br>OMG-LLaVA<br>The image captures a baseball game in progress, with a batter swinging a bat at a pitch. The batter is<br>wearing a baseball uniform, and the catcher is positioned behind him, ready to catch the ball.<br>There are several other people in the scene, including a few more players on the field and a couple of<br>spectators. Some of the spectators are standing closer to the field, while others are further away. A<br>bench is also visible in the background, likely for players to rest or spectators to sit.<br>GLaMM<br>A baseball player holding a bat on a field.<br>The image captures a baseball game in progress, with a batter standing at the plate, holding a baseball<br>bat. The batter is wearing a uniform and appears to be preparing for a swing. There are several people  LISA(13B)<br>in the scene, including teammates and opponents, all focused on the game. A total of six people can be<br>seen in the image, with some standing close to the batter and others positioned further away. A<br>baseball glove is visible in the scene, likely held by one of the players. The atmosphere is lively, with<br>everyone engaged in the game.<br>What is the number on the jersey of the athlete squatting on the ground?<br>OMG-LLaVA<br>The number on the jersey of the baseball player squatting on the ground is 52.<br>GLaMM<br>Sure, [SEG].<br>Sure, the segmentation result is [SEG]. LISA(13B)<br>Why is the athlete in the middle squatting?<br>OMG-LLaVA<br>The athlete in the middle squatting is likely in the process of catching a ball. In baseball, when a<br>player is on base, the opposing team tries to score runs by hitting the ball and running around the<br>bases. The catcher's role is to receive the ball thrown by the pitcher and attempt to tag out the<br>base runners, preventing them from advancing. In this image, the catcher is squatting in front of the<br>batter, ready to catch the ball if the batter misses or if the ball is hit into the dirt.<br>GLaMM<br>The athlete in the middle squatting is the catcher, who is preparing to catch the ball thrown<br>by the pitcher.<br>The baseball player in the middle is squatting because he is preparing to hit the ball during the game.<br>As a batter, his position is crucial for making contact with the pitched ball and potentially scoring runs  LISA(13B)<br>for his team. By squatting, the player lowers his center of gravity, increases his stability, and improves<br>his ability to react quickly to the pitch. This position also allows the player to generate more power in<br>his swing, which can result in a stronger hit and potentially longer distances for the ball to be hit.<br>**----- End of picture text -----**<br>


Figure 10: Qualitative comparison on the image-based conversation task. 

22 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_OMG-LLaVA_Bridging_Image-level,_Object-level,_Pixel-level_Reasoning_and_Understanding_images/nips_2024_OMG-LLaVA_Bridging_Image-level,_Object-level,_Pixel-level_Reasoning_and_Understanding.pdf-0023-00.png)


**----- Start of picture text -----**<br>
Please segment who is<br>holding the baseball bat.<br>Please segment the fence.<br>**----- End of picture text -----**<br>



![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_OMG-LLaVA_Bridging_Image-level,_Object-level,_Pixel-level_Reasoning_and_Understanding_images/nips_2024_OMG-LLaVA_Bridging_Image-level,_Object-level,_Pixel-level_Reasoning_and_Understanding.pdf-0023-01.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_OMG-LLaVA_Bridging_Image-level,_Object-level,_Pixel-level_Reasoning_and_Understanding_images/nips_2024_OMG-LLaVA_Bridging_Image-level,_Object-level,_Pixel-level_Reasoning_and_Understanding.pdf-0023-02.png)


**----- Start of picture text -----**<br>
Please segment out the pizza<br>that is closer to the man.<br>**----- End of picture text -----**<br>



![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_OMG-LLaVA_Bridging_Image-level,_Object-level,_Pixel-level_Reasoning_and_Understanding_images/nips_2024_OMG-LLaVA_Bridging_Image-level,_Object-level,_Pixel-level_Reasoning_and_Understanding.pdf-0023-03.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_OMG-LLaVA_Bridging_Image-level,_Object-level,_Pixel-level_Reasoning_and_Understanding_images/nips_2024_OMG-LLaVA_Bridging_Image-level,_Object-level,_Pixel-level_Reasoning_and_Understanding.pdf-0023-04.png)


**----- Start of picture text -----**<br>
Please segment the person  Please segment the person<br>raising the arm. with red hat.<br>Please segment the tree in the<br>Please segment the right girl.<br>background.<br>Please segment the middle  Please segment out the<br>beer bottle. person holding a fork.<br>**----- End of picture text -----**<br>


Figure 11: More visualization results of referring expression segmentation. 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_OMG-LLaVA_Bridging_Image-level,_Object-level,_Pixel-level_Reasoning_and_Understanding_images/nips_2024_OMG-LLaVA_Bridging_Image-level,_Object-level,_Pixel-level_Reasoning_and_Understanding.pdf-0023-06.png)


**----- Start of picture text -----**<br>
Please segment out what is<br>separating the spectators<br>from the athletes.<br>Please segment out what is<br>helping the woman block<br>the sunlight.<br>**----- End of picture text -----**<br>



![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_OMG-LLaVA_Bridging_Image-level,_Object-level,_Pixel-level_Reasoning_and_Understanding_images/nips_2024_OMG-LLaVA_Bridging_Image-level,_Object-level,_Pixel-level_Reasoning_and_Understanding.pdf-0023-07.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_OMG-LLaVA_Bridging_Image-level,_Object-level,_Pixel-level_Reasoning_and_Understanding_images/nips_2024_OMG-LLaVA_Bridging_Image-level,_Object-level,_Pixel-level_Reasoning_and_Understanding.pdf-0023-08.png)


**----- Start of picture text -----**<br>
Please segment out what the  Please segment out what<br>man is most likely to eat to fill  this woman will use to hit<br>his stomach. the tennis ball.<br>Please segment out the<br>Please segment out what can  food with the most fiber.<br>help the little boy feel cooler.<br>**----- End of picture text -----**<br>


Figure 12: Visualization results of reasoning segmentation. 

23 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_OMG-LLaVA_Bridging_Image-level,_Object-level,_Pixel-level_Reasoning_and_Understanding_images/nips_2024_OMG-LLaVA_Bridging_Image-level,_Object-level,_Pixel-level_Reasoning_and_Understanding.pdf-0024-00.png)


The image shows a zebra standing on dirt. A bird is also present on the dirt. The dirt is attached to a rock. 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_OMG-LLaVA_Bridging_Image-level,_Object-level,_Pixel-level_Reasoning_and_Understanding_images/nips_2024_OMG-LLaVA_Bridging_Image-level,_Object-level,_Pixel-level_Reasoning_and_Understanding.pdf-0024-02.png)


The image shows a cake placed on a dining table. The cake is attached to a bowl. 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_OMG-LLaVA_Bridging_Image-level,_Object-level,_Pixel-level_Reasoning_and_Understanding_images/nips_2024_OMG-LLaVA_Bridging_Image-level,_Object-level,_Pixel-level_Reasoning_and_Understanding.pdf-0024-04.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_OMG-LLaVA_Bridging_Image-level,_Object-level,_Pixel-level_Reasoning_and_Understanding_images/nips_2024_OMG-LLaVA_Bridging_Image-level,_Object-level,_Pixel-level_Reasoning_and_Understanding.pdf-0024-05.png)


The image shows a bench A skateboarder is grinding situated on snow. The bench is his board on a ramp located in front of a tree. The sky is visible over the tree. 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_OMG-LLaVA_Bridging_Image-level,_Object-level,_Pixel-level_Reasoning_and_Understanding_images/nips_2024_OMG-LLaVA_Bridging_Image-level,_Object-level,_Pixel-level_Reasoning_and_Understanding.pdf-0024-07.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_OMG-LLaVA_Bridging_Image-level,_Object-level,_Pixel-level_Reasoning_and_Understanding_images/nips_2024_OMG-LLaVA_Bridging_Image-level,_Object-level,_Pixel-level_Reasoning_and_Understanding.pdf-0024-08.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_OMG-LLaVA_Bridging_Image-level,_Object-level,_Pixel-level_Reasoning_and_Understanding_images/nips_2024_OMG-LLaVA_Bridging_Image-level,_Object-level,_Pixel-level_Reasoning_and_Understanding.pdf-0024-09.png)


**----- Start of picture text -----**<br>
a giraffe standing behind another<br>giraffe. a giraffe standing in front  A boya soccer ball in a blue uniform kicks<br>of another giraffe<br>**----- End of picture text -----**<br>


Figure 13: More visualization results of grounded conversation generation. 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_OMG-LLaVA_Bridging_Image-level,_Object-level,_Pixel-level_Reasoning_and_Understanding_images/nips_2024_OMG-LLaVA_Bridging_Image-level,_Object-level,_Pixel-level_Reasoning_and_Understanding.pdf-0024-11.png)


**----- Start of picture text -----**<br>
Point-prompt Box-prompt Mask-prompt<br>The purple umbrella in the<br>The sheep in the image is<br>image is a large umbrella,  The athlete wearing a blue<br>providing protection from  helmet is the batter in the  a large, wooly animal with a thick coat of wool. It is<br>the falling snow. It is held  baseball game. He is in the<br>lying down on the grass,<br>by the man, who is walking  process of swinging the bat,<br>appearing to be relaxed<br>alongside the woman. The  attempting to hit the ball that is  and comfortable. The<br>umbrella is likely  coming towards him. The batter<br>sheep is<br>waterproof, allowing them  is likely focused on the ball's<br>being sheared by a man,<br>to stay dry while they walk  trajectory and timing his swing to<br>through the snow. make contact with it. who is carefully removing<br>the wool from its body.<br>**----- End of picture text -----**<br>


Figure 14: More visualization results of grounded conversation generation. 

24 

