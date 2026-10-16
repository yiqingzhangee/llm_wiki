## **Mastering Text-to-Image Diffusion: Recaptioning, Planning, and Generating with Multimodal LLMs** 

**Ling Yang**[* 1] **Zhaochen Yu**[* 1] **Chenlin Meng**[2 3] **Minkai Xu**[2] **Stefano Ermon**[2] **Bin Cui**[1] https://github.com/YangLing0818/RPG-DiffusionMaster 

## **Abstract** 

Diffusion models have exhibit exceptional performance in text-to-image generation and editing. However, existing methods often face challenges when handling complex text prompts that involve multiple objects with multiple attributes and relationships. In this paper, we propose a brand new _training-free_ text-to-image generation/editing framework, namely _**Recaption, Plan and Generate (RPG)**_ , harnessing the powerful chain-of-thought reasoning ability of multimodal LLMs to enhance the compositionality of text-toimage diffusion models. Our approach employs the MLLM as a global planner to decompose the process of generating complex images into multiple simpler generation tasks within subregions. We propose _complementary regional diffusion_ to enable region-wise compositional generation. Furthermore, we integrate text-guided image generation and editing within the proposed RPG in a closed-loop fashion, thereby enhancing generalization ability. Extensive experiments demonstrate our RPG outperforms state-of-the-art textto-image diffusion models, including DALL-E 3 and SDXL, particularly in multi-category object composition and text-image semantic alignment. Notably, our RPG framework exhibits wide compatibility with various MLLM architectures (e.g., MiniGPT-4) and diffusion backbones (e.g., ControlNet). Our code is available at https://github.com/YangLing0818/RPGDiffusionMaster 

*Equal contribution 1Peking University, China 2Stanford University, USA[3] Pika Labs, USA. Correspondence to: Ling Yang _<_ yangling0818@163.com _>_ . 

> _Proceedings of the 41[st] International Conference on Machine Learning_ , Vienna, Austria. PMLR 235, 2024. Copyright 2024 by the author(s). 


![](raw/papers/2_LLM_NLP/LLM_paper_list/caption/icml_2024_Mastering_Text-to-Image_Diffusion_Recaptioning,_Planning,_and_Generating_with_Multimodal_LLMs_images/icml_2024_Mastering_Text-to-Image_Diffusion_Recaptioning,_Planning,_and_Generating_with_Multimodal_LLMs.pdf-0001-07.png)


**----- Start of picture text -----**<br>
a. Text  b. Text  c. Text  LLM<br>Prompt Prompt Prompt<br>Conditioning Layout/Attention Mask Layout<br>Diffusion Models Diffusion Models Diffusion Models<br>d. Text<br>Prompt MLLM  Set of Subprompts MLLM<br>Recaptioning CoT Planning<br>Generated<br>Image Assgin Subprompts<br>to Subregions<br>Optional Self-Refinement<br>RPG (Ours) Complementary Regional Diffusion<br>**----- End of picture text -----**<br>


_Figure 1._ Architecture comparison between (a) text-conditional diffusion models (Ramesh et al., 2022), (b) layout/attention-based diffusion models (Feng et al., 2022; Cao et al., 2023), (c) LLMgrounded diffusion models (Lian et al., 2023) and (d) our RPG. 

## **1. Introduction** 

Recent advancements in diffusion models (Sohl-Dickstein et al., 2015; Dhariwal & Nichol, 2021; Song et al., 2020; Yang et al., 2023c) have significantly improve the synthesis results of text-to-image models, such as Imagen (Saharia et al., 2022), DALL-E 2/3 (Ramesh et al., 2022; Betker et al., 2023) and SDXL (Podell et al., 2023). However, despite their remarkable capabilities in synthesizing realistic images consistent with text prompts, most diffusion models usually struggle to accurately follow some complex prompts (Feng et al., 2022; Lian et al., 2023; Liu et al., 2022; Bar-Tal et al., 2023), which require the model to compose objects with different attributes and relationships into a single image. 

Some works introduce additional layouts/boxes (Li et al., 2023b; Xie et al., 2023; Yang et al., 2023e; Qu et al., 2023; Chen et al., 2024; Wu et al., 2023b; Lian et al., 2023) as conditions or leveraging prompt-aware attention guidance (Feng et al., 2022; Chefer et al., 2023; Wang et al., 2023) to improve compositional text-to-image synthesis. For example, StructureDiffusion (Feng et al., 2022) incorporates linguistic structures into the guided generation process by manipulating cross-attention maps in diffusion models. GLIGEN (Li et al., 2023b) designs trainable gated self-attention layers to incorporate spatial inputs, such as bounding boxes, while freezing the weights of original diffusion model. 

1 

**Mastering Text-to-Image Diffusion: Recaptioning, Planning, and Generating with Multimodal LLMs** 


![](raw/papers/2_LLM_NLP/LLM_paper_list/caption/icml_2024_Mastering_Text-to-Image_Diffusion_Recaptioning,_Planning,_and_Generating_with_Multimodal_LLMs_images/icml_2024_Mastering_Text-to-Image_Diffusion_Recaptioning,_Planning,_and_Generating_with_Multimodal_LLMs.pdf-0002-01.png)


**----- Start of picture text -----**<br>
SDXL RPG (Ours)<br>DALL-E 3<br>**----- End of picture text -----**<br>


**Prompt:** A beautiful landscape with a river in the middle, the left of the river is **in the evening** and **in the winter** with a big iceberg and a small village while some people are skiing on the river and some people are skating, the right of the river is **in the summer** with a **volcano in the morning** and a small village while some people are playing. 


![](raw/papers/2_LLM_NLP/LLM_paper_list/caption/icml_2024_Mastering_Text-to-Image_Diffusion_Recaptioning,_Planning,_and_Generating_with_Multimodal_LLMs_images/icml_2024_Mastering_Text-to-Image_Diffusion_Recaptioning,_Planning,_and_Generating_with_Multimodal_LLMs.pdf-0002-03.png)


**----- Start of picture text -----**<br>
SDXL RPG (Ours)<br>DALL-E 3<br>Prompt: A green twintail girl in orange dress is sitting on the sofa while a messy desk in under a big window on the left, while a<br>lively aquarium is on the top right of the sofa , realistic style.<br>**----- End of picture text -----**<br>



![](raw/papers/2_LLM_NLP/LLM_paper_list/caption/icml_2024_Mastering_Text-to-Image_Diffusion_Recaptioning,_Planning,_and_Generating_with_Multimodal_LLMs_images/icml_2024_Mastering_Text-to-Image_Diffusion_Recaptioning,_Planning,_and_Generating_with_Multimodal_LLMs.pdf-0002-04.png)


**----- Start of picture text -----**<br>
SDXL RPG (Ours) SDXL RPG (Ours)<br>DALL-E 3 DALL-E 3<br>Left Prompt:  A Chinese general wearing a crown, with whiskers and  golden Chinese  style armor, standing with a  majestic dragon head on<br>his chest , symbolizing his strength, wearing  black and gold boots . His appearance exudes a sense of authority, wisdom, and an unyielding<br>spirit , embodying the ideal ancient Chinese hero.<br>Right Prompt : This painting is a quintessential example of ancient Chinese ink art , At the top of the painting , towering mountains shrouded<br>in  mist rise majestically . The mountains‘ craggy peaks are sketched with fine , precise lines , typical of traditional Chinese ink art.  A slender<br>swirling mists , meandering waterfall begins its descent here , its water appearing almost ethereal amidst the soft. In the middle section,  the<br>waterfall cascades energetically  , creating a dynamic contrast with the serene mountains above . Lush pine trees  , rendered with graceful ,<br>flowing brush strokes , flank the waterfall. These trees appear to dance with the rhythm of the water , adding a vibrant life to the scene. At<br>the bottom , the waterfall concludes its journey in a tranquil pool. The water's surface is calm , reflecting the surrounding nature and the sky<br>above. Here ,  delicate flowers  and  small shrubs  are  depicted along the water's edge  , symbolizing peace and harmony with nature.<br>**----- End of picture text -----**<br>


_Figure 2._ Compared to SDXL (Podell et al., 2023) and DALL-E 3 (Betker et al., 2023), our proposed RPG exhibits a superior ability to convey intricate and compositional text prompts within generated images ( **colored text denotes critical part** ). 

2 

**Mastering Text-to-Image Diffusion: Recaptioning, Planning, and Generating with Multimodal LLMs** 


![](raw/papers/2_LLM_NLP/LLM_paper_list/caption/icml_2024_Mastering_Text-to-Image_Diffusion_Recaptioning,_Planning,_and_Generating_with_Multimodal_LLMs_images/icml_2024_Mastering_Text-to-Image_Diffusion_Recaptioning,_Planning,_and_Generating_with_Multimodal_LLMs.pdf-0003-01.png)


**----- Start of picture text -----**<br>
Pose ControlNet RPG + ControlNet<br>**----- End of picture text -----**<br>



![](raw/papers/2_LLM_NLP/LLM_paper_list/caption/icml_2024_Mastering_Text-to-Image_Diffusion_Recaptioning,_Planning,_and_Generating_with_Multimodal_LLMs_images/icml_2024_Mastering_Text-to-Image_Diffusion_Recaptioning,_Planning,_and_Generating_with_Multimodal_LLMs.pdf-0003-02.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/caption/icml_2024_Mastering_Text-to-Image_Diffusion_Recaptioning,_Planning,_and_Generating_with_Multimodal_LLMs_images/icml_2024_Mastering_Text-to-Image_Diffusion_Recaptioning,_Planning,_and_Generating_with_Multimodal_LLMs.pdf-0003-03.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/caption/icml_2024_Mastering_Text-to-Image_Diffusion_Recaptioning,_Planning,_and_Generating_with_Multimodal_LLMs_images/icml_2024_Mastering_Text-to-Image_Diffusion_Recaptioning,_Planning,_and_Generating_with_Multimodal_LLMs.pdf-0003-04.png)


**User input** : A beautiful black hair girl with her eyes closed in champagne long sleeved formal dress standing in her bright room with delicate blue vases with pink roses on the left and some white roses, filled with upgraded growth all around on the right 

**Base prompt** : A beautiful girl standing in her bright room **Region 0** : A delicate blue vases with pink roses **Region 1** : A beautiful black hair girl with her eyes closed in champagne long sleeved formal dress **Region 2:** Some white roses, filled with upgraded growth all around 


![](raw/papers/2_LLM_NLP/LLM_paper_list/caption/icml_2024_Mastering_Text-to-Image_Diffusion_Recaptioning,_Planning,_and_Generating_with_Multimodal_LLMs_images/icml_2024_Mastering_Text-to-Image_Diffusion_Recaptioning,_Planning,_and_Generating_with_Multimodal_LLMs.pdf-0003-07.png)


**----- Start of picture text -----**<br>
Depth ControlNet RPG + ControlNet<br>**----- End of picture text -----**<br>



![](raw/papers/2_LLM_NLP/LLM_paper_list/caption/icml_2024_Mastering_Text-to-Image_Diffusion_Recaptioning,_Planning,_and_Generating_with_Multimodal_LLMs_images/icml_2024_Mastering_Text-to-Image_Diffusion_Recaptioning,_Planning,_and_Generating_with_Multimodal_LLMs.pdf-0003-08.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/caption/icml_2024_Mastering_Text-to-Image_Diffusion_Recaptioning,_Planning,_and_Generating_with_Multimodal_LLMs_images/icml_2024_Mastering_Text-to-Image_Diffusion_Recaptioning,_Planning,_and_Generating_with_Multimodal_LLMs.pdf-0003-09.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/caption/icml_2024_Mastering_Text-to-Image_Diffusion_Recaptioning,_Planning,_and_Generating_with_Multimodal_LLMs_images/icml_2024_Mastering_Text-to-Image_Diffusion_Recaptioning,_Planning,_and_Generating_with_Multimodal_LLMs.pdf-0003-10.png)


**User input** : Under the clear starry sky, clear river water flows in the mountains, and the lavender flower sea in front of me dances with the wind, a peaceful, beautiful, and harmonious atmosphere 

**Base prompt** : Under the clear starry sky, clear river water flows in the mountains, and the lavender flower sea in front of me dances with the wind, A peaceful, beautiful, and harmonious atmosphere **Region 0:** Clear starry sky, many stars twinkling in the night sky **Region 1:** The blue stream flows through the valleys covered with colorful flowers ,reflecting the starry sky **Region 2:** The sea of lavender dancing in the wind, 


![](raw/papers/2_LLM_NLP/LLM_paper_list/caption/icml_2024_Mastering_Text-to-Image_Diffusion_Recaptioning,_Planning,_and_Generating_with_Multimodal_LLMs_images/icml_2024_Mastering_Text-to-Image_Diffusion_Recaptioning,_Planning,_and_Generating_with_Multimodal_LLMs.pdf-0003-13.png)


**----- Start of picture text -----**<br>
Canny Edge<br>**----- End of picture text -----**<br>



![](raw/papers/2_LLM_NLP/LLM_paper_list/caption/icml_2024_Mastering_Text-to-Image_Diffusion_Recaptioning,_Planning,_and_Generating_with_Multimodal_LLMs_images/icml_2024_Mastering_Text-to-Image_Diffusion_Recaptioning,_Planning,_and_Generating_with_Multimodal_LLMs.pdf-0003-14.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/caption/icml_2024_Mastering_Text-to-Image_Diffusion_Recaptioning,_Planning,_and_Generating_with_Multimodal_LLMs_images/icml_2024_Mastering_Text-to-Image_Diffusion_Recaptioning,_Planning,_and_Generating_with_Multimodal_LLMs.pdf-0003-15.png)


**----- Start of picture text -----**<br>
ControlNet<br>**----- End of picture text -----**<br>



![](raw/papers/2_LLM_NLP/LLM_paper_list/caption/icml_2024_Mastering_Text-to-Image_Diffusion_Recaptioning,_Planning,_and_Generating_with_Multimodal_LLMs_images/icml_2024_Mastering_Text-to-Image_Diffusion_Recaptioning,_Planning,_and_Generating_with_Multimodal_LLMs.pdf-0003-16.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/caption/icml_2024_Mastering_Text-to-Image_Diffusion_Recaptioning,_Planning,_and_Generating_with_Multimodal_LLMs_images/icml_2024_Mastering_Text-to-Image_Diffusion_Recaptioning,_Planning,_and_Generating_with_Multimodal_LLMs.pdf-0003-17.png)


**----- Start of picture text -----**<br>
RPG + ControlNet<br>**----- End of picture text -----**<br>



![](raw/papers/2_LLM_NLP/LLM_paper_list/caption/icml_2024_Mastering_Text-to-Image_Diffusion_Recaptioning,_Planning,_and_Generating_with_Multimodal_LLMs_images/icml_2024_Mastering_Text-to-Image_Diffusion_Recaptioning,_Planning,_and_Generating_with_Multimodal_LLMs.pdf-0003-18.png)


**User input** : From left to right, an ancient Chinese city in spring, summer, autumn and winter in four different regions 

**Region 0** : The city in spring where every where is lively and colorful with pink flowers mountains **Region 1:** The city with buildings with green rooves in summer, where a lake is beautiful 

**Region 2:** The city in autumn with buildings with orange rooves, where every where is light orange and golden **Region 3:** The city in winter with trees covered with snow and blue rooves covered with snow, where every is white and covered with snow 

_Figure 3._ Our RPG framework can extend text-to-image generation with more conditions (e.g., pose, depth and canny edge) by utilizing ControlNet (Zhang et al., 2023a). Compared to original ControlNet, RPG significantly improves its prompt understanding by decomposing ”user input” into the combination of base prompt and subprompts, and further enhance its compositional semantic alignment of generated images by performing region-wise diffusion generation (in Section 2.2). 

3 

**Mastering Text-to-Image Diffusion: Recaptioning, Planning, and Generating with Multimodal LLMs** 

Another potential solution is to leverage image understanding feedback (Huang et al., 2023a; Xu et al., 2023; Sun et al., 2023a; Fang et al., 2023) for refining diffusion generation. For instance, GORS (Huang et al., 2023a) finetunes a pretrained text-to-image model with generated images that highly align with the compositional prompts, where the finetuning loss is weighted by the text-image alignment reward. Inspired by the reinforcement learning from human feedback (RLHF) (Ouyang et al., 2022; Stiennon et al., 2020) in natural language processing, ImageReward (Xu et al., 2023) builds a general-purpose reward model to improve text-to-image models in aligning with human preference. 

Despite some improvements achieved by these methods, there are still two main limitations in the context of compositional/complex image generation: (i) existing layoutbased or attention-based methods can only provide rough and suboptimal spatial guidance, and struggle to deal with overlapped objects (Cao et al., 2023; Hertz et al., 2022; Lian et al., 2023) ; (ii) feedback-based methods require to collect high-quality feedback and incur additional training costs. 

To address these limitations, we introduce a new _trainingfree_ text-to-image generation framework, namely _**Recaption, Plan and Generate (RPG)**_ , unleashing the impressive reasoning ability of multimodal LLMs to enhance the compositionality and controllability of diffusion models. We propose three core strategies in RPG: 

**Multimodal Recaptioning.** We specialize in transforming text prompts into highly descriptive ones, offering informative augmented prompt comprehension and semantic alignment in diffusion models. We use LLMs to decompose the text prompt into distinct subprompts, and recaption them with more detailed descriptions. We use MLLMs to automatically recaption input image for identifying the semantic discrepancies between generated images and target prompt. 

**Chain-of-Thought Planning.** In a pioneering approach, we partition the image space into complementary subregions and assign different subprompts to each subregion, breaking down compositional generation tasks into multiple simpler subtasks. Thoughtfully crafting task instructions and in-context examples, we harness the powerful chainof-thought reasoning capabilities of MLLMs (Zhang et al., 2023d) for efficient region division. By analyzing the recaptioned intermediate results, we generate detailed rationales and precise instructions for subsequent image compositions. 

**Complementary Regional Diffusion.** Based on the planned non-overlapping subregions and their respective prompts, we propose _complementary regional diffusion_ to enhance the flexibility and precision of compositional textto-image generation. Specifically, we independently generate image content guided by subprompts within designated rectangle subregion, and subsequently merge them spatially 

in a _resize-and-concatenate_ approach. This region-specific diffusion effectively addresses the challenge of conflicting overlapped image contents. Furthermore, we extend this framework to accommodate editing tasks by employing contour-based regional diffusion, enabling precise manipulation of inconsistent regions targeted for modification. 

This new RPG framework can unify both text-guided image generation and editing tasks in a closed-loop fashion. We compare our RPG framework with previous work in Figure 1 and summarize our main contributions as follows: 

- We propose a new training-free text-to-image generation framework, namely _Recaption, Plan and Generate (RPG)_ , to improve the composibility and controllability of diffusion models to the fullest extent. 

- RPG is the first to utilize MLLMs as both _multimodal recaptioner and CoT planner_ to reason out more informative instructions for steering diffusion models. 

- We propose _complementary regional diffusion_ to enable extreme collaboration with MLLMs for compositional image generation and precise image editing. 

- Our RPG framework is user-friendly, and can be generalized to different MLLM architectures (e.g., MiniGPT4) and diffusion backbones (e.g., ControlNet). 

- Extensive qualitative and quantitative comparisons with previous SOTA methods, such as SDXL, DALL-E 3 and InstructPix2Pix, demonstrate our superior textguided image generation/editing ability. 

## **2. Method** 

## **2.1. Overview of Proposed RPG** 

In this section, we introduce our novel training-free framework - **R** ecaption, **P** lan and **G** enerate ( **RPG** ). We delineate three fundamental strategies of our RPG in text-to-image generation (Section 2.2), as depicted in Figure 4. Specifically, given a complex text prompt that includes multiple entities and relationships, we leverage (multimodal) LLMs to _recaption_ the prompt by decomposing it into a base prompt and highly descriptive subprompts. Subsequently, we utilize multimodal CoT planning to allocate the split (sub)prompts to complementary regions along the spatial axes. Building upon these assignments, we introduce _complementary regional diffusion_ to independently generate image latents and aggregate them in each sampling step. 

Our RPG framework exhibits versatility by extending its application to text-guided image editing with minimal adjustments, as exemplified in Section 2.3. For instance, in the recaptioning phase, we utilize MLLMs to analyze the paired target prompt and source image, which results in informative 

4 

**Mastering Text-to-Image Diffusion: Recaptioning, Planning, and Generating with Multimodal LLMs** 


![](raw/papers/2_LLM_NLP/LLM_paper_list/caption/icml_2024_Mastering_Text-to-Image_Diffusion_Recaptioning,_Planning,_and_Generating_with_Multimodal_LLMs_images/icml_2024_Mastering_Text-to-Image_Diffusion_Recaptioning,_Planning,_and_Generating_with_Multimodal_LLMs.pdf-0005-01.png)


**----- Start of picture text -----**<br>
Recaption-Plan-Generate (RPG) for<br>Complementary Regional Diffusion<br>Text-to-Image Generation<br>User Text (Base) Prompt : A moon Two owls<br>A nobleman is standing  (0.8) (0.8)<br>under a bright moon while<br>two owls are on a big oak. Rationale Generation Base Prompt A<br>## Task objectives: Generate complementary subregions based on the recaptioned subpromptsand assign them to subregions. (0.2) nobleman(0.8) A big oak(0.8)<br>MLLMs [Instructions + In-context examples]<br>Weighted sum for each sampling step<br>[Question]:<br>12 Identify key phrases: 、、A noble manA bright moon 3. Two owls, perched side by side ……2. A bright moon, full and radiant …… Subprompts :  1. A nobleman, regal and ….. Step=5 Step=10 Step=20 Step=30<br>34、、Two owlsA big oak Let’s think step by step (trigger CoT reasoning) 4. A big oak, towering and robust ……<br>Rationale:   Here we have 4 subprompts, thus total<br>4 regions. First, we take the nobleman and his<br>MLLMs castle as foreground and background, we place<br>Chain-of-Thought them in lower left of the image. Next, bright moon should in the sky so we assign the moon to the  Region-wise<br>Planning higher left corner……  (Coarse grained area division) Generation<br>Text-to-Text Recaptioning<br>1. A nobleman, regal and distinguished, with a<br>sharp gaze, dressed in a velvet doublet,  CoT Reasoning<br>standing proudly in his ancestral castle A moon Two owls<br>2.3. A bright moon, full and radiant, casting a silver glow over a tranquil lake, serene and majestic in the night sky.Two owls, perched side by side, wise and  Therefore, the region split ratio should be: 1,1,1;2,2,3 （Subregion Planning Column Mode ） Step=40<br>mysterious, with piercing eyes, on an ancient,  [Assign subprompts to subregions]<br>4. gnarled branch under the starlit skyA big oak, towering and robust, its sprawling  Region 0Region 1：： Two owls, perched side by side ……A bright moon, full and  radiant…… A nobleman Big oak<br>branches a testament to centuries, leaves  Region 2： A nobleman, regal and …..<br>whispering stories in the gentle breeze. Region 3： A big oak, towering and robust  ……<br>Stage 1: Recaption Stage 2: Plan Stage 3: Generate<br>**----- End of picture text -----**<br>


_Figure 4._ Overview of our RPG framework for text-to-image generation. 

multimodal feedback that captures their cross-modal semantic discrepancies. In multimodal CoT planning, we generate a step-by-step edit plan and produce _precise contours_ for our regional diffusion. Furthermore, we demonstrate the ability to execute our RPG workflow in a closed-loop manner for progressive self-refinement, as showcased in Section 2.3. This approach combines precise contour-based editing with complementary regional diffusion generation. 


![](raw/papers/2_LLM_NLP/LLM_paper_list/caption/icml_2024_Mastering_Text-to-Image_Diffusion_Recaptioning,_Planning,_and_Generating_with_Multimodal_LLMs_images/icml_2024_Mastering_Text-to-Image_Diffusion_Recaptioning,_Planning,_and_Generating_with_Multimodal_LLMs.pdf-0005-04.png)


**----- Start of picture text -----**<br>
Example 2 1:1<br>Split ratio:  1,1,1; 2,1,1<br>1.  First, we split the image into two rows where horizontal  0 1<br>split ratio is 1:2<br>2. 3. vertical split ratio is  Finally, we split the second row into two columns, whereSecond, we split the first row into two columns, where1:1, and we get region 0 and region 1 1 1:2 2 3<br>vertical split ratio is also 1:1, and we get  region 2  and  region 3<br>Rules:  We start splitting from the top left corner of the image, and<br>the region is numbered from the top left region to right, row by row. 3 1:1<br>**----- End of picture text -----**<br>


_Figure 5._ An illustrative example for region division. 

## **2.2. Text-to-image Generation** 

**Prompt Recaptioning** Let _y[c]_ be a complex user prompt which includes multiple entities with different attributes and relationships. We use MLLMs to identify the key phrases in _y[c]_ to obtain subpormpts denoted as: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/caption/icml_2024_Mastering_Text-to-Image_Diffusion_Recaptioning,_Planning,_and_Generating_with_Multimodal_LLMs_images/icml_2024_Mastering_Text-to-Image_Diffusion_Recaptioning,_Planning,_and_Generating_with_Multimodal_LLMs.pdf-0005-08.png)


where _n_ denotes the number of key phrases. Inspired by DALL-E 3 (Betker et al., 2023), which uses pre-trained **image-to-text** (I2T) caption models to generate descriptive prompts for images, and construct new datasets with highquality image-text pairs. In contrast, we leverage the impressive language understanding and reasoning abilities of LLMs and use the LLM as the **text-to-text** (T2T) captioner to further _recaption_ each subprompt with more informative detailed descriptions: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/caption/icml_2024_Mastering_Text-to-Image_Diffusion_Recaptioning,_Planning,_and_Generating_with_Multimodal_LLMs_images/icml_2024_Mastering_Text-to-Image_Diffusion_Recaptioning,_Planning,_and_Generating_with_Multimodal_LLMs.pdf-0005-10.png)


In this way, we can produce denser fine-grained details for each subprompt in order to effectively improve the fidelity of generated image, and reduce the semantic discrepancy between prompt and image. 

**CoT Planning for Region Division** Based on the recaptioned subprompts, we leverage the powerful multimodal chain-of-thought (CoT) reasoning ability of LLMs (Zhang et al., 2023d) to plan the compositions of final image content for diffusion models. Concretely, we divide image space _H × W_ into several _complementary regions_ , and assign each augmented subprompt _y_ ˆ _[i]_ to specific region _R[i]_ : 


![](raw/papers/2_LLM_NLP/LLM_paper_list/caption/icml_2024_Mastering_Text-to-Image_Diffusion_Recaptioning,_Planning,_and_Generating_with_Multimodal_LLMs_images/icml_2024_Mastering_Text-to-Image_Diffusion_Recaptioning,_Planning,_and_Generating_with_Multimodal_LLMs.pdf-0005-13.png)


In order to produce meaningful and accurate subregions, we need to carefully specify two components for planning region divisions: (i) region parameters: we define that rows are separated by ”;” and each column is denoted by a series of numbers separated by commas (e.g., ”1,1,1”). To be specific , we first use ”;” to split an image into different rows, then within each row, we use commas to split a row into different regions, see Figure 5 for better comprehension; (ii) region-wise task specifications to instruct MLLMs: we utilize the CoT reasoning of MLLMs with some designed in-context examples to reason out the plan of region division. We here provide a simplified template of our instructions and in-context examples: 

5 

**Mastering Text-to-Image Diffusion: Recaptioning, Planning, and Generating with Multimodal LLMs** 


![](raw/papers/2_LLM_NLP/LLM_paper_list/caption/icml_2024_Mastering_Text-to-Image_Diffusion_Recaptioning,_Planning,_and_Generating_with_Multimodal_LLMs_images/icml_2024_Mastering_Text-to-Image_Diffusion_Recaptioning,_Planning,_and_Generating_with_Multimodal_LLMs.pdf-0006-01.png)


**----- Start of picture text -----**<br>
Parallel Subprompt-Latent Cross Attention<br>Decompose complex text prompt  A cat Some leaves<br>into simpler subprompts<br>Attention maps Denoised latent<br>Text prompt:   An old man is telling<br>Intermediate  Query a story to a little girl while a cat is<br>listening as well on the tree.<br>latent latent<br>Token  Keys  (from text prompt) Token  Values  (From text prompt)<br>× M1 M2 …… × ……<br>Decomposed Prompt:<br>Region 0: a cat is on the tree ……<br>Region 1: An old man is telling a story .…..<br>Region 2: Some leaves …….. M3 M4 An old man A little girl<br>Region 3: A little girl ……<br>Subregion Latent Concatenation<br>Latent with base prompt<br>Concatenate Compositional<br>× Base ratio latent<br>Resize<br>1-<br>Base<br>× ratio<br>Concatenated latent<br>**----- End of picture text -----**<br>


_Figure 6._ The demonstration of each sampling step in our **Complementary Regional Diffusion** . 

## _**1.Task instruction**_ 

_You are an smart region planner for image. You should use split ratio to specify the split method of the image, and then recaption each subregion prompts with more descriptive prompts while maintaining the original meaning._ 

## _**2.Multi-modal split tutorial**_ 

_......_ 

## _**3. In-context examples**_ 

_**User Prompt:** A girl with white ponytail and black dress are chatting with a blonde curly hair girl in a white dress in a cafe._ 

_# Key pharses extraction and Recaption_ 

- _# Split ratio Planning_ 

- _# Composition Logic_ 

- _# Aesthetic Considerations:_ 

- _# Final output_ 

- _**4.Trigger CoT reasoning ability of MLLMs**_ 

- _**User Prompt:** An old man with his dog is looking at a parrot on the tree._ 

_Reasoning: Let’s think step by step......_ 

entity, we treat the different parts of this entity as different entities (e.g., A green hair twintail in red blouse , wearing blue skirt. = _⇒_ green hair twintail, red blouse, blue skirt). 

**Complementary Regional Diffusion** Recent works (Liu et al., 2022; Wang et al., 2023; Chefer et al., 2023; Feng et al., 2022) have adjusted cross-attention masks or layouts to facilitate compositional generation. However, these approaches predominantly rely on simply stacking latents, leading to conflicts and ambiguous results in overlapped regions. To address this issue, as depicted in Figure 6, we introduce a novel approach called complementary regional diffusion for region-wise generation and image composition. We extract non-overlapping complementary rectangular regions and apply a **resize-and-concatenate** post-processing step to achieve high-quality compositional generation. Additionally, we enhance coherence by combining the base prompt with recaptioned subprompts to reinforce the conjunction of each generated region and maintain overall image coherence (detailed ablation study in Section 4). This can be represented as: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/caption/icml_2024_Mastering_Text-to-Image_Diffusion_Recaptioning,_Planning,_and_Generating_with_Multimodal_LLMs_images/icml_2024_Mastering_Text-to-Image_Diffusion_Recaptioning,_Planning,_and_Generating_with_Multimodal_LLMs.pdf-0006-19.png)


To facilitating inferring the region for each subprompt, we adhere to two key principles in designing in-context example and generating informative rationales: (i) the objects with same class name (e.g., five apples) will be separately assign to different regions to ensure the numeric accuracy; (ii) If the prompt focuses more on the appearance of a specific 

where _s_ is a fixed random seed, CRD is the abbreviation for complementary regional diffusion. 

More concretely, we construct a prompt batch with base prompt _y_[base] = _y[c]_ and the recaptioned subprompts: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/caption/icml_2024_Mastering_Text-to-Image_Diffusion_Recaptioning,_Planning,_and_Generating_with_Multimodal_LLMs_images/icml_2024_Mastering_Text-to-Image_Diffusion_Recaptioning,_Planning,_and_Generating_with_Multimodal_LLMs.pdf-0006-23.png)


6 

**Mastering Text-to-Image Diffusion: Recaptioning, Planning, and Generating with Multimodal LLMs** 


![](raw/papers/2_LLM_NLP/LLM_paper_list/caption/icml_2024_Mastering_Text-to-Image_Diffusion_Recaptioning,_Planning,_and_Generating_with_Multimodal_LLMs_images/icml_2024_Mastering_Text-to-Image_Diffusion_Recaptioning,_Planning,_and_Generating_with_Multimodal_LLMs.pdf-0007-01.png)


**----- Start of picture text -----**<br>
Six patterned mugs User Text Prompt , arranged in  : Another Loop or Exit Mask and<br>two columns, on a marble surface,<br>and a rose in the vase on the left. Inpainting<br>Source/Generated Image<br>Stage 1: Recaption Multi-Round Editing with RPG Edited Image Precise ContoursDetect<br>Task Instruction +<br>In-context Examples +<br>User Text Prompt Stage 3: Generate<br>Image-to-Text Recaptioning<br>## Task objectives: Analyze user text prompt to identify key<br>phrases and attributes,  utilizing visual reasoning on the  Region-wise Editing<br>image to detect inconsistency<br>[Task instruction+ In-context examples ]<br>[Question] Stage 2:  Plan<br>User text prompt:  Six patterned mugs, arranged in two columns,<br>on a marble surface, and a rose in the vase on the left. Task description:  Analyze the prompt and  recaption the image,  Editing Planning:<br>find inconsistency between the prompt and the recaption prompt,  Rationale Generation：<br>make an edit plan, Let’s think step by step (trigger CoT reasoning): 1. More mugs staggered （delete）<br>Recaption the image:  Eight  patterned mugs, six mugs arranged in  2. More roses exist (delete)<br>two columns on a marble surface, and  two roses  in the  two mugs  3. There is no vase visible in the image (replace or add)<br>on the left lower corner.  Chain-of-Thought Edit plan:<br>Planning According to the inconsistency,<br>(i)  Firstly, we delete the mug on the right down corner along with its reflection image.<br>(ii)  Next, we delete the rose in the first column from left to right along with its reflection image.<br>(iii)  Finally, here is only one inconsistency, we should replace the mug in the left lower corner<br>with a vase along with its reflection image.<br>**----- End of picture text -----**<br>


_Figure 7._ RPG unifies text-guided image generation and editing in a closed-loop approach. 

In each timestep, we deliver the prompt batch into the denoising network and manipulate the cross-attention layers to generate different latents _{_ _**z** t[i] −_ 1 _[}][n] i_ =0[and] _**[ z]** t_[base] _−_ 1[in] _[ parallel]_[,] as demonstrated in Figure 6. We formulate this process as: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/caption/icml_2024_Mastering_Text-to-Image_Diffusion_Recaptioning,_Planning,_and_Generating_with_Multimodal_LLMs_images/icml_2024_Mastering_Text-to-Image_Diffusion_Recaptioning,_Planning,_and_Generating_with_Multimodal_LLMs.pdf-0007-04.png)


where image latent _**z** t_ is the query and each subprompt _y_ ˆ _[i]_ works as a key and value. _WQ, WK, WV_ are linear projections and _d_ is the latent projection dimension of the keys and queries. Then, we shall proceed with resizing and concatenating the generated latents _{_ _**z** t[i] −_ 1 _[}][n] i_ =0[,][according][to] their assigned region numbers (from 0 to _n_ ) and respective proportions. Here we denote each resized latent as: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/caption/icml_2024_Mastering_Text-to-Image_Diffusion_Recaptioning,_Planning,_and_Generating_with_Multimodal_LLMs_images/icml_2024_Mastering_Text-to-Image_Diffusion_Recaptioning,_Planning,_and_Generating_with_Multimodal_LLMs.pdf-0007-06.png)


where _h, w_ are the height and the width of its assigned region _R[i]_ . We directly concatenate them along the spatial axes: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/caption/icml_2024_Mastering_Text-to-Image_Diffusion_Recaptioning,_Planning,_and_Generating_with_Multimodal_LLMs_images/icml_2024_Mastering_Text-to-Image_Diffusion_Recaptioning,_Planning,_and_Generating_with_Multimodal_LLMs.pdf-0007-08.png)


To ensure a coherent transition in the boundaries of different regions and a harmonious fusion between the background and the entities within each region, we use the weighted sum of the _base latents_ _**z** t_[base] _−_ 1[and the] _[ concatenated latent]_ 

_**z** t_[cat] _−_ 1[to produce the final denoising output:] 


![](raw/papers/2_LLM_NLP/LLM_paper_list/caption/icml_2024_Mastering_Text-to-Image_Diffusion_Recaptioning,_Planning,_and_Generating_with_Multimodal_LLMs_images/icml_2024_Mastering_Text-to-Image_Diffusion_Recaptioning,_Planning,_and_Generating_with_Multimodal_LLMs.pdf-0007-11.png)


Here _β_ is used to achieve a suitable balance between human aesthetic perception and alignment with the complex text prompt of the generated image. It is worth noting that complementary regional diffusion can generalize to arbitrary diffusion backbones including SDXL (Podell et al., 2023), ConPreDiff (Yang et al., 2023b) and ControlNet (Zhang et al., 2023a), which will be evaluated in Section 3.1. 

## **2.3. Text-Guided Image Editing** 

**Image Recaptioning** Our RPG can also generalize to textguided image editing tasks as illustrated in Figure 7. In recaptioning stage, RPG adopts MLLMs as a captioner to recaption the source image, and leverage its powerful reasoning ability to identify the fine-grained semantic discrepancies between the image and target prompt. We directly analyze how the input image _**x**_ aligns with the target prompt _y_[tar] . Specifically, we identify the key entities in _**x**_ and _y_[tar] : 


![](raw/papers/2_LLM_NLP/LLM_paper_list/caption/icml_2024_Mastering_Text-to-Image_Diffusion_Recaptioning,_Planning,_and_Generating_with_Multimodal_LLMs_images/icml_2024_Mastering_Text-to-Image_Diffusion_Recaptioning,_Planning,_and_Generating_with_Multimodal_LLMs.pdf-0007-15.png)


Then we utilize MLLMs (e.g., GPT4 (OpenAI, 2023), Gemini Pro (Team et al., 2023)) to check the differences between 

7 

**Mastering Text-to-Image Diffusion: Recaptioning, Planning, and Generating with Multimodal LLMs** 


![](raw/papers/2_LLM_NLP/LLM_paper_list/caption/icml_2024_Mastering_Text-to-Image_Diffusion_Recaptioning,_Planning,_and_Generating_with_Multimodal_LLMs_images/icml_2024_Mastering_Text-to-Image_Diffusion_Recaptioning,_Planning,_and_Generating_with_Multimodal_LLMs.pdf-0008-01.png)


**----- Start of picture text -----**<br>
Attribute Binding Numeric Accuracy Complex Relationship<br>SDXL<br>DALL-E 3<br>LMD+<br>RPG<br>(Ours)<br>A  green  Two hardcover  Two women are  Two teddy bears  Three Four orange  Two apples on  A girl with  A white<br>twintail hair  books,  one red  negotiating a  sitting on the  sunflowers in  cats and  four left and an  green twintail twintail girl<br>girl wearing a  open  and  one  deal in a  white carpet  two vases  blue balloons  ipad on the  is standing in  in orange dress<br>white shirt  green  along  meeting room,  surrounded by  while  three floating above  right on the  her room, her  is sitting on<br>and skirt  with  a red pen  the woman  on the  three toy cars butterflies  them bed, where the  desk on the  the sofa,  a<br>printed with  and a pair of  left  with  blonde  are flying sheet printed  left is messy  messy desk  is<br>green apple red  glasses blunt bangs is  with apple and the  on the left<br>lying on the  in  white suit ,  aquarium on  while a wide<br>carpet the woman  on the  the top of the  window is above<br>right with  white  orange sofa on  the desk  and<br>curly hair  is in  the right is  the aquarium on<br>black suit . lively, Disney  the  top of the<br>style. sofa is lively,<br>anime style<br>**----- End of picture text -----**<br>


_Figure 8._ Qualitative comparison between our RPG and SOTA text-to-image models (SDXL (Podell et al., 2023) and DALL-E 3 (Betker et al., 2023)), and LLM-grounded diffusion model LMD+ (Lian et al., 2023). 

_{y[i] }[n] i_ =0[and] _[ {][e][i][}][m] i_ =0[regarding numeric accuracy, attribute] binding and object relationships. The resulting multimodal understanding feedback would be delivered to MLLMs for reason out editing plans. 

**CoT Planning for Editing** Based on the captured semantic discrepancies between prompt and image, RPG triggers the CoT reasoning ability of MLLMs with highquality filtered in-context examples, which involves manually designed step-by-step editing cases such as entity missing/redundancy, attribute mismatch, ambiguous relationships. Here, in our RPG, we introduce three main edit operations for dealing with these issues: addition Add(), deletion Del(), modification Mod(). Take the multimodal feedback as the grounding context, RPG plans out a series of editing instructions. An example Plan( _y_[tar] _,_ _**x**_ ) can be denoted as a composed operation list: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/caption/icml_2024_Mastering_Text-to-Image_Diffusion_Recaptioning,_Planning,_and_Generating_with_Multimodal_LLMs_images/icml_2024_Mastering_Text-to-Image_Diffusion_Recaptioning,_Planning,_and_Generating_with_Multimodal_LLMs.pdf-0008-05.png)


where _i, j, k <_ = _n,_ length(Plan( _y_[tar] _, x_[0] )) = _L_ . In this way, we are able to decompose original complex editing task into simpler editing tasks for more accurate results. 

**Contour-based Regional Diffusion** To collaborate more effectively with CoT-planned editing instructions, we generalize our complementary regional diffusion to text-guided 

editing. We locate and mask the target contour associated with the editing instruction (Kirillov et al., 2023), and apply diffusion-based inpainting (Rombach et al., 2022) to edit the target contour region according to the planned operation list Plan( _y_[tar] _,_ _**x**_ ). Compared to traditional methods that utilize cross-attention map swap or replacement (Hertz et al., 2022; Cao et al., 2023) for editing, our **mask-and-inpainting** method powered by CoT planning enables more accurate and complex editing operations (i.e., addition, deletion and modification). 

**Multi-Round Editing for Closed-Loop Refinement** Our text-guided image editing workflow is adaptable for a closedloop self-refined text-to-image generation, which combines the contour-based editing with complementary regional diffusion generation. We could conduct multi-round closedloop RPG workflow controlled by MLLMs to progressively refine the generated image for aligning closely with the target text prompt. Considering the time efficiency, we set a maximum number of rounds to avoid being trapped in the closed-loop procedure. Based on this closed-loop paradigm, we can unify text-guided generation and editing in our RPG, providing more practical framework for the community. 

8 

**Mastering Text-to-Image Diffusion: Recaptioning, Planning, and Generating with Multimodal LLMs** 

_Table 1._ Evaluation results on T2I-CompBench. RPG consistently demonstrates best performance regarding attribute binding, object relationships, and complex compositions. We denote the best score in blue , and the second-best score in green . The baseline data is quoted from Chen et al. (2023a). 

|**Model**|**Attribute Binding**<br>**Color**_↑_<br>**Shape**_↑_<br>**Texture**_↑_|**Object Relationship**<br>**Complex**_↑_<br>**Spatial**_↑_<br>**Non-Spatial**_↑_|
|---|---|---|
|Stable Diffusion v1.4 (Rombach et al.,2022)<br>Stable Diffusion v2 (Rombach et al.,2022)<br>Composable Diffusion (Liu et al.,2022)<br>Structured Diffusion (Feng et al.,2022)<br>Attn-Exct v2 (Chefer et al.,2023)<br>GORS (Huang et al.,2023a)<br>DALL-E 2 (Ramesh et al.,2022)<br>SDXL (Betker et al.,2023)<br>PixArt-_α_(Chen et al.,2023a)<br>ConPreDiff (Yang et al.,2023b)|0.3765<br>0.3576<br>0.4156<br>0.5065<br>0.4221<br>0.4922<br>0.4063<br>0.3299<br>0.3645<br>0.4990<br>0.4218<br>0.4900<br>0.6400<br>0.4517<br>0.5963<br>0.6603<br>0.4785<br>0.6287<br>0.5750<br>0.5464<br>0.6374<br>0.6369<br>0.5408<br>0.5637<br>0.6886<br>0.5582<br>0.7044|0.1246<br>0.3079<br>0.3080<br>0.1342<br>0.3096<br>0.3386<br>0.0800<br>0.2980<br>0.2898<br>0.1386<br>0.3111<br>0.3355<br>0.1455<br>0.3109<br>0.3401<br>0.1815<br>0.3193<br>0.3328<br>0.1283<br>0.3043<br>0.3696<br>0.2032<br>0.3110<br>0.4091<br>0.2082<br>0.3179<br>0.4117|
||0.7019<br>0.5637<br>0.7021|0.2362<br>0.3195<br>0.4184|
|**RPG (Ours)**|0.8335<br>0.6801<br>0.8129|0.4547<br>0.3462<br>0.5408|



## **3. Experiments** 

## **3.1. Text-to-Image Generation** 

**Implementation Details** Our RPG is general and extensible, we can incorporate **arbitrary** MLLM architectures and diffusion backbones[123] into the framework. In our experiment, we choose GPT-4 (OpenAI, 2023) as the recaptioner and CoT planner, and use SDXL (Podell et al., 2023) as the base diffusion backbone to build our RPG framework. Concretely, in order to trigger the CoT planning ability of MLLMs, we carefully design task-aware template and highquality in-context examples to conduct few-shot prompting. _**Base prompt**_ and its weighted hyperparameter _**base ratio**_ are critical in our regional diffusion, we have provide further analysis in Figure 16. When the user prompt includes the entities with same class (e.g., two women, four boys), we need to set higher base ratio to highlight these distinct identities. On the contrary, when user prompt includes the the entities with different class name (e.g., ceramic vase and glass vase), we need lower base ratio to avoid the confusion between the base prompt and subprompts. 

**Main Results** We compare with previous SOTA text-toimage models DALL-E 3 (Betker et al., 2023), SDXL and LMD+ (Lian et al., 2023) in three main compositional scenarios: **(i) Attribute Binding** . Each text prompt in this scenario has multiple attributes that bind to different entities. **(ii) Numeric Accuracy** . Each text prompt in this scenario has multiple entities sharing the same class name, the number of each entity should be greater than or equal to two. **(iii) Complex Relationship** . Each text prompt in this scenario has multiple entities with different attributes and relationships (e.g., spatial and non-spational). As demonstrated 

- 1https://github.com/CompVis/stable-diffusion 

- 2https://github.com/huggingface/diffusers 

Hierarchical Regional Diffusion 


![](raw/papers/2_LLM_NLP/LLM_paper_list/caption/icml_2024_Mastering_Text-to-Image_Diffusion_Recaptioning,_Planning,_and_Generating_with_Multimodal_LLMs_images/icml_2024_Mastering_Text-to-Image_Diffusion_Recaptioning,_Planning,_and_Generating_with_Multimodal_LLMs.pdf-0009-10.png)


**----- Start of picture text -----**<br>
0 1<br>0 0 1<br>2 3<br>Two girls are chatting in a  Base prompt: Base prompt:<br>Chinese restaurant , the one on  Two girls are chatting in a Chinese  Two girls are chatting in a Chinese<br>the left is a blonde long hair girl  restaurant ,  restaurant<br>with blue and white sailor shirt  ,  Region 0 :a blonde long hair girl  Region 0:  a blonde long hair girl<br>black, the one on the right is a  with blue and white sailor shirt, Region 1:  A black double bun hair girl,<br>black double bun hair girl , green  Region 1 :a black double bun hair  green cheongsam printed with flowers<br>cheongsam printed with flowers girl , green cheongsam printed with  Region 2:   a blue and white sailor shirt,<br>flowers Region 3:  Green cheongsam printed<br>with flowers<br>0 0 1<br>2<br>0 1 3<br>4<br>2 5<br>A large bookshelf with five floors  Base prompt : A large bookshelf with  Base prompt : A large bookshelf with three floors<br>and six compartments with lively  three floors and six compartments  and six compartments with books, lively aquarium,<br>aquarium on the top left,and  with books, lively aquarium, and  and plant in terrarium<br>plant in terrarium on the top  plant in terrarium  Region 0:  small lively aquarium with goldfish and<br>right,the Books with ancient kraft  Region 0:  small lively aquarium with  sea weed in the compartment of the bookshelf,<br>paper covers in the second and  goldfish and sea weed in the  Region 1:  delicate flowers in the terrarium in the<br>third floors,newly printed books  compartment of the bookshelf,  compartment of the bookshelf<br>including red and blue books in  delicate flowers in the terrarium in  Region 2:  Books with ancient kraft paper covers in<br>the fourth and fifth floors. the compartment of the bookshelf  the compartment of the bookshelf<br>Region 1:  Books with ancient kraft  Region 3:  Books with ancient kraft paper covers in<br>paper covers in the compartment of  the compartment of the bookshelf<br>the bookshelf  Region 4:  Some new books including red covers and<br>Region 2:  Some new books including  blue covers in the compartment of the bookshelf<br>red covers and blue covers in the  Region 4:  Some new books including red covers and<br>compartment of the bookshelf  blue covers in the compartment of the bookshelf<br>**----- End of picture text -----**<br>


_Figure 9._ Demonstration of our hierarchical regional diffusion. Diffusion with more hierarchies can produce more satisfying results. 

> 3https://github.com/hako-mikan/sd-webui-regional-prompter 

9 

**Mastering Text-to-Image Diffusion: Recaptioning, Planning, and Generating with Multimodal LLMs** 


![](raw/papers/2_LLM_NLP/LLM_paper_list/caption/icml_2024_Mastering_Text-to-Image_Diffusion_Recaptioning,_Planning,_and_Generating_with_Multimodal_LLMs_images/icml_2024_Mastering_Text-to-Image_Diffusion_Recaptioning,_Planning,_and_Generating_with_Multimodal_LLMs.pdf-0010-01.png)


**----- Start of picture text -----**<br>
Llama 2-13B Vicuna-33B MiniGPT-4 Llama 2-70B GPT-4<br>Text prompt:  A girl with<br>white ponytail and black<br>dress are chatting with a<br>blonde curly hair girl in a<br>white dress in a café.<br>Text prompt:  A cute<br>ragdoll cat is standing<br>on the table in the<br>middle, and beautiful<br>teapot on the right and<br>a tea cup on the left.<br>**----- End of picture text -----**<br>


_Figure 10._ Generalizing RPG to different (multimodal) LLM architectures, including Llama 2 (Touvron et al., 2023b), Vicuna (Chiang et al., 2023) and MiniGPT-4 (Zhu et al., 2023). 

in Table 1, our RPG is significantly superior to previous models in all three scenarios, and achieves remarkable level of both fidelity and precision in aligning with text prompt. We observe that SDXL and DALL-E 3 have poor generation performance regarding numeric accuracy and complex relationship. In contrast, our RPG can effectively plan out precise number of subregions, and utilize proposed complementary regional diffusion to accomplish compositional generation. Compared to LMD+ (Lian et al., 2023), a LLMgrounded layout-based text-to-image diffusion model, our RPG demonstrates both enhanced semantic expression capabilities and image fidelity. We attribute this to our CoT planning and complementary regional diffusion. For quantitative results, we assess the text-image alignment of our method in a comprehensive benchmark, T2I-Compbench (Huang et al., 2023a), which is utilized to evaluate the compositional text-to-image generation capability. In Table 1, we consistently achieve best performance among all methods proposed for both general text-to-image generation and compositional generation, including SOTA model ConPreDiff (Yang et al., 2023b). 

**Hierarchical Regional Diffusion** We can extend our regional diffusion to a hierarchical format by splitting certain subregion to smaller subregions. As illustrated in Figure 9, when we increase the hierarchies of our region split, RPG can achieve a significant improvement in text-to-image generation. This promising result reveals that our complementary regional diffusion provides a new perspective for handling complex generation tasks and has the potential to generate arbitrarily compositional images. 

**Generalizing to Various LLMs and Diffusion Backbones** Our RPG framework is of great generalization ability, and can be easily generalized to various (M)LLM architectures (in Figure 10) and diffusion backbones (in Figure 11). We 

observe that both LLM and diffusion architectures can influence the generation results. We also generalize RPG to ControlNet (Zhang et al., 2023a) for incorporating more conditional modalities. As demonstrated in Figure 3, our RPG can significantly improve the composibility of original ControlNet in both image fidelity and textual semantic alignment. 


![](raw/papers/2_LLM_NLP/LLM_paper_list/caption/icml_2024_Mastering_Text-to-Image_Diffusion_Recaptioning,_Planning,_and_Generating_with_Multimodal_LLMs_images/icml_2024_Mastering_Text-to-Image_Diffusion_Recaptioning,_Planning,_and_Generating_with_Multimodal_LLMs.pdf-0010-07.png)


**----- Start of picture text -----**<br>
RPG  with SD v2.1 RPG  with ConPreDiff<br>Text prompt:  In a fantasy world, two beautiful girls are<br>wandering in the city mall chat and eating where the stall is<br>on both side and the big castle is behind<br>**----- End of picture text -----**<br>



![](raw/papers/2_LLM_NLP/LLM_paper_list/caption/icml_2024_Mastering_Text-to-Image_Diffusion_Recaptioning,_Planning,_and_Generating_with_Multimodal_LLMs_images/icml_2024_Mastering_Text-to-Image_Diffusion_Recaptioning,_Planning,_and_Generating_with_Multimodal_LLMs.pdf-0010-08.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/caption/icml_2024_Mastering_Text-to-Image_Diffusion_Recaptioning,_Planning,_and_Generating_with_Multimodal_LLMs_images/icml_2024_Mastering_Text-to-Image_Diffusion_Recaptioning,_Planning,_and_Generating_with_Multimodal_LLMs.pdf-0010-09.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/caption/icml_2024_Mastering_Text-to-Image_Diffusion_Recaptioning,_Planning,_and_Generating_with_Multimodal_LLMs_images/icml_2024_Mastering_Text-to-Image_Diffusion_Recaptioning,_Planning,_and_Generating_with_Multimodal_LLMs.pdf-0010-10.png)


**----- Start of picture text -----**<br>
Text prompt:  The scene is framed to focus on the interaction<br>between Snow White and the dwarfs,with the warm interior<br>and the cozy fire suggesting a safe haven from the dark forest<br>outside. The cottage feels lived-in and welcoming,a perfect<br>setting for an evening of storytelling and camaraderie.<br>**----- End of picture text -----**<br>


_Figure 11._ Generalizing RPG to different diffusion backbones, Stable Diffusion v2.1 (Rombach et al., 2022) and recent SOTA diffusion model ConPreDiff (Yang et al., 2023b). 

10 

**Mastering Text-to-Image Diffusion: Recaptioning, Planning, and Generating with Multimodal LLMs** 


![](raw/papers/2_LLM_NLP/LLM_paper_list/caption/icml_2024_Mastering_Text-to-Image_Diffusion_Recaptioning,_Planning,_and_Generating_with_Multimodal_LLMs_images/icml_2024_Mastering_Text-to-Image_Diffusion_Recaptioning,_Planning,_and_Generating_with_Multimodal_LLMs.pdf-0011-01.png)


**----- Start of picture text -----**<br>
Source<br>Image<br>Prompt2Pro<br>mpt<br>Instruct<br>Pix2Pix<br>MasaCtrl<br>RPG<br>(Ours)<br>Edit task: Edit task: Edit task:<br>Blue compute -> green  Apple on the book ->  Black cat -> delete<br>computer delete Blue shirt -> Cyan<br>Green chair -> wooden  Blue book -> blue book  shirt<br>chair printed with apple<br>Wrist -> wrist with Rolex<br>**----- End of picture text -----**<br>



![](raw/papers/2_LLM_NLP/LLM_paper_list/caption/icml_2024_Mastering_Text-to-Image_Diffusion_Recaptioning,_Planning,_and_Generating_with_Multimodal_LLMs_images/icml_2024_Mastering_Text-to-Image_Diffusion_Recaptioning,_Planning,_and_Generating_with_Multimodal_LLMs.pdf-0011-02.png)


**----- Start of picture text -----**<br>
Multi-Round Text-to-Image Editing<br>Source/Generated Image Round 1 Round 2<br>Editing prompt:  A green twintail girl with white shirt and pink dress<br>Editing prompt:  In a fantasy world, two beautiful girls are wandering in the city mall chat and eating<br>where the stall is on both side and the big castle is behind<br>Editing prompt:  A brave knight is coming to crusade a giant big dragon in lightning and thunder<br>**----- End of picture text -----**<br>


_Figure 13._ Multi-round text-guided image editing with our RPG framework. 

back and CoT planning. We also find that RPG is able to achieve satisfying editing results within 3 rounds. 

## **4. Model Analysis** 

_Figure 12._ Qualitative comparison in text-guided image editing. We outperform previous powerful methods including Prompt2Prompt (Hertz et al., 2022), InstructPix2Pix (Brooks et al., 2023) and MasaCtrl (Cao et al., 2023). 

## **3.2. Text-Guided Image Editing** 

**Qualitative Results** In the qualitative comparison of textguided image editing, we choose some strong baseline methods, including Prompt2Prompt (Hertz et al., 2022), InstructPix2Pix (Brooks et al., 2023) and MasaCtrl (Cao et al., 2023). Prompt2Prompt and MasaCtrl conduct editing mainly through text-grounded cross-attention swap or replacement, InstructPix2Pix aims to learn a model that can follow human instructions. As presented in Figure 12, RPG produces more precise editing results than previous methods, and our mask-and-inpainting editing strategy can also perfectly preserve the semantic structure of source image. 

**Multi-Round Editing** We conduct multi-round editing to evaluate the self-refinement with our RPG framework in Figure 13. We conclude that the self-refinement based on RPG can significantly improve precision, demonstrating the effectiveness of our recaptioning-based multimodal feed- 

**Effect of Recaptioning** We conduct ablation study about the recaptioning, and show the result in Figure 14. From the result, we observe that without recaptioning, the model tends to ignore some key words in the generated images. Our recaptioning can describe these key words with highinformative and denser details, thus generating more delicate and precise images. 

**Effect of CoT Planning** In the ablation study about CoT planning, as demonstrated in Figure 15, we observe that the model without CoT planning fail to parse and convey complex relationships from text prompt. In contrast, our CoT planning can help the model better identify fine-grained attributes and relationships from text prompt, and express them through a more realistic planned composition. 

**Effect of Base Prompt** In RPG, we leverage the generated latent from base prompt in diffusion models to improve the coherence of image compositions. Here we conduct more analysis on it in Figure 16. From the results, we find that the proper ratio of base prompt can benefit the conjunction of different subregions, enabling more natural composition. Another finding is that excessive base ratio may result in 

11 

**Mastering Text-to-Image Diffusion: Recaptioning, Planning, and Generating with Multimodal LLMs** 


![](raw/papers/2_LLM_NLP/LLM_paper_list/caption/icml_2024_Mastering_Text-to-Image_Diffusion_Recaptioning,_Planning,_and_Generating_with_Multimodal_LLMs_images/icml_2024_Mastering_Text-to-Image_Diffusion_Recaptioning,_Planning,_and_Generating_with_Multimodal_LLMs.pdf-0012-01.png)


**----- Start of picture text -----**<br>
Without  Without<br>Recaptioning Recaptioning<br>Base prompt:  A man and his dog is  Base prompt : A nobleman standing in front of<br>looking at a parrot on the tree his castle while two owls on a big oak at night<br>Region 0 : An old man with his dog Region 0:  A bright moon.<br>Region 1:  A big tree Region 1 : two owls<br>Region 2:  A parrot Region 2:  A nobleman<br>Region 3:  A big oak<br>With  With<br>Recaptioning Recaptioning<br>**----- End of picture text -----**<br>



![](raw/papers/2_LLM_NLP/LLM_paper_list/caption/icml_2024_Mastering_Text-to-Image_Diffusion_Recaptioning,_Planning,_and_Generating_with_Multimodal_LLMs_images/icml_2024_Mastering_Text-to-Image_Diffusion_Recaptioning,_Planning,_and_Generating_with_Multimodal_LLMs.pdf-0012-02.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/caption/icml_2024_Mastering_Text-to-Image_Diffusion_Recaptioning,_Planning,_and_Generating_with_Multimodal_LLMs_images/icml_2024_Mastering_Text-to-Image_Diffusion_Recaptioning,_Planning,_and_Generating_with_Multimodal_LLMs.pdf-0012-03.png)


**Base prompt:** A nobleman standing in front of his castle while two owls on a big oak at night **Region 0:** A bright moon, full and radiant, casting a silver glow over a tranquil lake, serene and majestic in the night sky. **Region 1:** Two owls, perched side by side, wise and mysterious, with piercing eyes, on an ancient, gnarled branch under the starlit sky **Region 2:** A nobleman, regal and distinguished, with a sharp gaze, dressed in a velvet doublet, standing proudly in his ancestral castle **Region 3:** A big oak, towering and robust, its sprawling branches a testament to centuries, leaves whispering stories in the gentle breeze. 

**Base prompt:** A man and his dog is looking at a parrot on the tree **Region 0:** A contemplative man stands beside his loyal dog, both gazing upward with a sense of wonder, the bond between them palpable and heartwarming. **Region 1:** A robust tree stands tall, its branches a cradle for wildlife, leaves whispering stories of the forest, a symbol of life and growth. **Region 2:** A vibrant parrot perches alertly , its feather is smooth like blue silk, its sharp eyes scanning the world from its lofty vantage point. 

_Figure 14._ Ablation study of recaptioning in RPG. 

undesirable results because of the confusion between the base prompt and regional prompt. 


![](raw/papers/2_LLM_NLP/LLM_paper_list/caption/icml_2024_Mastering_Text-to-Image_Diffusion_Recaptioning,_Planning,_and_Generating_with_Multimodal_LLMs_images/icml_2024_Mastering_Text-to-Image_Diffusion_Recaptioning,_Planning,_and_Generating_with_Multimodal_LLMs.pdf-0012-08.png)


**----- Start of picture text -----**<br>
Without  With<br>CoT Planning CoT Planning<br>Prompt:  Two beautiful Chinese girls wearing cheongsams are<br>drinking tea in the tea room, and a Chinese Landscape<br>Painting is hanging on the wall, the girl on the left is black<br>ponytail in red cheongsam, the girl on the right is white<br>ponytail in orange cheongsam<br>**----- End of picture text -----**<br>



![](raw/papers/2_LLM_NLP/LLM_paper_list/caption/icml_2024_Mastering_Text-to-Image_Diffusion_Recaptioning,_Planning,_and_Generating_with_Multimodal_LLMs_images/icml_2024_Mastering_Text-to-Image_Diffusion_Recaptioning,_Planning,_and_Generating_with_Multimodal_LLMs.pdf-0012-09.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/caption/icml_2024_Mastering_Text-to-Image_Diffusion_Recaptioning,_Planning,_and_Generating_with_Multimodal_LLMs_images/icml_2024_Mastering_Text-to-Image_Diffusion_Recaptioning,_Planning,_and_Generating_with_Multimodal_LLMs.pdf-0012-10.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/caption/icml_2024_Mastering_Text-to-Image_Diffusion_Recaptioning,_Planning,_and_Generating_with_Multimodal_LLMs_images/icml_2024_Mastering_Text-to-Image_Diffusion_Recaptioning,_Planning,_and_Generating_with_Multimodal_LLMs.pdf-0012-11.png)


**----- Start of picture text -----**<br>
Prompt:  A close up view of a child admiring the fireworks on<br>the sky at night by the lake while crowds of people are going<br>to the temple fair in Chinese lunar new year.<br>**----- End of picture text -----**<br>


_Figure 15._ Ablation study of CoT planning in RPG. 

move the diffusion process from pixel space to latent space for balancing algorithm efficiency and image quality. Recent advancements in text-to-image diffusion models , such as SDXL (Podell et al., 2023), ContextDiff (Yang et al., 2024b) and DALL-E 3 (Betker et al., 2023), further improve both quality and alignment from different perspectives. Despite their tremendous success, generating high-fidelity images with complex prompt is still challenging (Betker et al., 2023; Huang et al., 2023a). This problem is exacerbated when dealing with compositional descriptions involving spatial relationships, attribute binding and numeric awareness. In this paper, we aim to address this issue by incorporating the powerful CoT reasoning ability of MLLMs into text-to-image diffusion models. 

## **5. Related Work** 

**Text-Guided Diffusion Models** Diffusion models (SohlDickstein et al., 2015; Song & Ermon, 2019; Ho et al., 2020; Song & Ermon, 2020; Song et al., 2020; Yang et al., 2024a) are a promising class of generative models, and Dhariwal & Nichol (2021) have demonstrated the superior image synthesis quality of diffusion model over generative adversarial networks (GANs) (Reed et al., 2016; Creswell et al., 2018). GLIDE (Nichol et al., 2021) and Imagen (Saharia et al., 2022) focus on the text-guided image synthesis, leveraging pre-trained CLIP model (Radford et al., 2021; Raffel et al., 2020) in the image sampling process to improve the semantic alignment between text prompt and generated image. Latent Diffusion Models (LDMs) (Rombach et al., 2022) 

**Compositional Diffusion Generation** Recent researches aim to improve compositional ability of text-to-image diffusion models. Some approaches mainly introduce additional modules into diffusion models in training (Li et al., 2023b; Avrahami et al., 2023; Zhang et al., 2023a; Mou et al., 2023; Yang et al., 2023e; Huang et al., 2023b;a). For example, GLIGEN (Li et al., 2023b) and ReCo (Yang et al., 2023e) design position-aware adapters on top of the diffusion models for spatially-conditioned image generation. T2I-Adapter and ControlNet (Zhang et al., 2023a; Mou et al., 2023) specify some high-level features of images for controlling semantic structures (Zhang et al., 2023b). These methods, however, result in additional training and inference costs. Training-free methods aim to steer diffusion models through 

12 

**Mastering Text-to-Image Diffusion: Recaptioning, Planning, and Generating with Multimodal LLMs** 


![](raw/papers/2_LLM_NLP/LLM_paper_list/caption/icml_2024_Mastering_Text-to-Image_Diffusion_Recaptioning,_Planning,_and_Generating_with_Multimodal_LLMs_images/icml_2024_Mastering_Text-to-Image_Diffusion_Recaptioning,_Planning,_and_Generating_with_Multimodal_LLMs.pdf-0013-01.png)


**----- Start of picture text -----**<br>
No Base Prompt Base Ratio=0.2 Base Ratio=0.4 Base Ratio=0.5<br>Two women are negotiating a deal in a meeting room, the woman  on the<br>left  with  blonde blunt bangs is in  white suit , the woman  on the right<br>with  white curly hair  is in  black suit .<br>No Base Prompt Base Ratio=0.2 Base Ratio=0.4 Base Ratio=0.5<br>A white ceramic vase contains a sunflower and a butterfly is flying on<br>the left , and a glass vase contains a pink rose on the right<br>**----- End of picture text -----**<br>


_Figure 16._ Ablation study of base prompt in complementary regional diffusion. 

manipulating latent or cross-attention maps according to spatial or semantic constraints during inference stages (Feng et al., 2022; Liu et al., 2022; Hertz et al., 2022; Cao et al., 2023; Chen et al., 2024; Chefer et al., 2023). Composable Diffusion (Liu et al., 2022) decomposes a compositional prompt into smaller sub-prompts to generate distinct latents and combines them with a score function. Chen et al. (2024) and Lian et al. (2023) utilize the bounding boxes (layouts) to propagate gradients back to the latent and enable the model to manipulate the cross-attention maps towards specific regions. Other methods apply Gaussian kernels (Chefer et al., 2023) or incorporate linguistic features (Feng et al., 2022; Rassin et al., 2023) to manipulate the cross-attention maps. Nevertheless, such manipulation-based methods can only make rough controls, and often lead to unsatisfied compositional generation results, especially when dealing with overlapped objects (Lian et al., 2023; Cao et al., 2023). Hence, we introduce an effective _training-free complementary regional diffusion model_ , grounded by MLLMs, to progressively refine image compositions with more precise control in the sampling process. 

**Multimodal LLMs for Image Generation** Large Language Models (LLMs) (ChatGPT, 2022; Chung et al., 2022; Zhang et al., 2022; Iyer et al., 2022; Workshop et al., 2022; Muennighoff et al., 2022; Zeng et al., 2022; Taylor et al., 2022; Chowdhery et al., 2023; Chen et al., 2023b; Zhu et al., 2023; Touvron et al., 2023a; Yang et al., 2023a; Li et al., 2023a) have profoundly impacted the AI community. Leading examples like ChatGPT (ChatGPT, 2022) have showcased the advanced language comprehension and reasoning skills through techniques such as instruction tuning (Ouyang et al., 2022; Li et al., 2023c; Zhang et al., 2023c; Liu et al., 2023). Further, Multimodal Large language Models (MLLMs), (Koh et al., 2023; Yu et al., 2023; 

Sun et al., 2023b; Dong et al., 2023; Fu et al., 2023; Pan et al., 2023; Wu et al., 2023a; Zou et al., 2023; Yang et al., 2023d; Gupta & Kembhavi, 2023; Sur´ıs et al., 2023) integrate LLMs with vision models to extend their impressive abilities from language tasks to vision tasks, including image understanding, reasoning and synthesis. The collaboration between LLMs (ChatGPT, 2022; OpenAI, 2023) and diffusion models (Ramesh et al., 2022; Betker et al., 2023) can significantly improve the text-image alignment as well as the quality of generated images (Yu et al., 2023; Chen et al., 2023b; Dong et al., 2023; Wu et al., 2023b; Feng et al., 2023; Pan et al., 2023). For instance, GILL (Koh et al., 2023) can condition on arbitrarily interleaved image and text inputs to synthesize coherent image outputs, and Emu (Sun et al., 2023b) stands out as a generalist multimodal interface for both image-to-text and text-to-image tasks. Recently, LMD (Lian et al., 2023) utilizes LLMs to enhance the compositional generation of diffusion models by generating images grounded on bounding box layouts from the LLM (Li et al., 2023b). However, existing works mainly incorporate the LLM as a simple plug-in component into diffusion models, or simply take the LLM as a layout generator to control image compositions. In contrast, we utilize MLLMs to plan out image compositions for diffusion models where MLLMs serves as a global task planner in both _region-based_ generation and editing process. 

## **6. Conclusion** 

In this paper, aiming to address the challenges of complex or compositional text-to-image generation, we propose a SOTA training-free framework RPG, harnessing MLLMs to master diffusion models. In RPG, we propose complementary regional diffusion models to collaborate with our designed MLLM-based recaptioner and planner. Furthermore, our RPG can unify text-guided imgae generation and editing in a closed-loop approach, and is capable of generalizing to any MLLM architectures and diffusion backbones. For future work, we will continue to improve this new framework for incorporating more complex modalities as input condition, and extend it to more realistic applications. 

## **References** 

Avrahami, O., Hayes, T., Gafni, O., Gupta, S., Taigman, Y., Parikh, D., Lischinski, D., Fried, O., and Yin, X. Spatext: Spatio-textual representation for controllable image generation. In _Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition_ , pp. 18370–18380, 2023. 

Bar-Tal, O., Yariv, L., Lipman, Y., and Dekel, T. Multidiffusion: Fusing diffusion paths for controlled image generation. _arXiv preprint arXiv:2302.08113_ , 2023. 

13 

**Mastering Text-to-Image Diffusion: Recaptioning, Planning, and Generating with Multimodal LLMs** 

- Betker, J., Goh, G., Jing, L., Brooks, T., Wang, J., Li, L., Ouyang, L., Zhuang, J., Lee, J., Guo, Y., et al. Improving image generation with better captions. _Computer Science. https://cdn. openai. com/papers/dall-e-3. pdf_ , 2023. 

- Brooks, T., Holynski, A., and Efros, A. A. Instructpix2pix: Learning to follow image editing instructions. In _Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition_ , pp. 18392–18402, 2023. 

- Cao, M., Wang, X., Qi, Z., Shan, Y., Qie, X., and Zheng, Y. Masactrl: Tuning-free mutual self-attention control for consistent image synthesis and editing. _arXiv preprint arXiv:2304.08465_ , 2023. 

- ChatGPT, I. Introducing chatgpt, 2022. 

- Chefer, H., Alaluf, Y., Vinker, Y., Wolf, L., and Cohen-Or, D. Attend-and-excite: Attention-based semantic guidance for text-to-image diffusion models. _ACM Transactions on Graphics (TOG)_ , 42(4):1–10, 2023. 

- Chen, J., Yu, J., Ge, C., Yao, L., Xie, E., Wu, Y., Wang, Z., Kwok, J., Luo, P., Lu, H., et al. Pixart-alpha: Fast training of diffusion transformer for photorealistic text-to-image synthesis. _arXiv preprint arXiv:2310.00426_ , 2023a. 

- Chen, M., Laina, I., and Vedaldi, A. Training-free layout control with cross-attention guidance. In _Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision_ , pp. 5343–5353, 2024. 

- Chen, W.-G., Spiridonova, I., Yang, J., Gao, J., and Li, C. Llava-interactive: An all-in-one demo for image chat, segmentation, generation and editing. _arXiv preprint arXiv:2311.00571_ , 2023b. 

- Chiang, W.-L., Li, Z., Lin, Z., Sheng, Y., Wu, Z., Zhang, H., Zheng, L., Zhuang, S., Zhuang, Y., Gonzalez, J. E., et al. Vicuna: An open-source chatbot impressing gpt-4 with 90%* chatgpt quality. _See https://vicuna. lmsys. org (accessed 14 April 2023)_ , 2023. 

- Chowdhery, A., Narang, S., Devlin, J., Bosma, M., Mishra, G., Roberts, A., Barham, P., Chung, H. W., Sutton, C., Gehrmann, S., et al. Palm: Scaling language modeling with pathways. _Journal of Machine Learning Research_ , 24(240):1–113, 2023. 

- Chung, H. W., Hou, L., Longpre, S., Zoph, B., Tay, Y., Fedus, W., Li, Y., Wang, X., Dehghani, M., Brahma, S., et al. Scaling instruction-finetuned language models. _arXiv preprint arXiv:2210.11416_ , 2022. 

- Creswell, A., White, T., Dumoulin, V., Arulkumaran, K., Sengupta, B., and Bharath, A. A. Generative adversarial networks: An overview. _IEEE signal processing magazine_ , 35(1):53–65, 2018. 

- Dhariwal, P. and Nichol, A. Diffusion models beat gans on image synthesis. _Advances in neural information processing systems_ , 34:8780–8794, 2021. 

- Dong, R., Han, C., Peng, Y., Qi, Z., Ge, Z., Yang, J., Zhao, L., Sun, J., Zhou, H., Wei, H., et al. Dreamllm: Synergistic multimodal comprehension and creation. _arXiv preprint arXiv:2309.11499_ , 2023. 

- Fang, G., Jiang, Z., Han, J., Lu, G., Xu, H., and Liang, X. Boosting text-to-image diffusion models with fine-grained semantic rewards. _arXiv preprint arXiv:2305.19599_ , 2023. 

- Feng, W., He, X., Fu, T.-J., Jampani, V., Akula, A. R., Narayana, P., Basu, S., Wang, X. E., and Wang, W. Y. Training-free structured diffusion guidance for compositional text-to-image synthesis. In _The Eleventh International Conference on Learning Representations_ , 2022. 

- Feng, W., Zhu, W., Fu, T.-j., Jampani, V., Akula, A., He, X., Basu, S., Wang, X. E., and Wang, W. Y. Layoutgpt: Compositional visual planning and generation with large language models. _arXiv preprint arXiv:2305.15393_ , 2023. 

- Fu, T.-J., Hu, W., Du, X., Wang, W. Y., Yang, Y., and Gan, Z. Guiding instruction-based image editing via multimodal large language models. _arXiv preprint arXiv:2309.17102_ , 2023. 

- Gupta, T. and Kembhavi, A. Visual programming: Compositional visual reasoning without training. In _Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition_ , pp. 14953–14962, 2023. 

- Hertz, A., Mokady, R., Tenenbaum, J., Aberman, K., Pritch, Y., and Cohen-Or, D. Prompt-to-prompt image editing with cross attention control. _arXiv preprint arXiv:2208.01626_ , 2022. 

- Ho, J., Jain, A., and Abbeel, P. Denoising diffusion probabilistic models. _Advances in neural information processing systems_ , 33:6840–6851, 2020. 

- Huang, K., Sun, K., Xie, E., Li, Z., and Liu, X. T2icompbench: A comprehensive benchmark for open-world compositional text-to-image generation. _arXiv preprint arXiv:2307.06350_ , 2023a. 

- Huang, L., Chen, D., Liu, Y., Shen, Y., Zhao, D., and Zhou, J. Composer: Creative and controllable image synthesis with composable conditions. _arXiv preprint arXiv:2302.09778_ , 2023b. 

- Iyer, S., Lin, X. V., Pasunuru, R., Mihaylov, T., Simig, D., Yu, P., Shuster, K., Wang, T., Liu, Q., Koura, P. S., et al. Opt-iml: Scaling language model instruction meta 

14 

**Mastering Text-to-Image Diffusion: Recaptioning, Planning, and Generating with Multimodal LLMs** 

- learning through the lens of generalization. _arXiv preprint arXiv:2212.12017_ , 2022. 

- Kirillov, A., Mintun, E., Ravi, N., Mao, H., Rolland, C., Gustafson, L., Xiao, T., Whitehead, S., Berg, A. C., Lo, W.-Y., et al. Segment anything. _arXiv preprint arXiv:2304.02643_ , 2023. 

- Koh, J. Y., Fried, D., and Salakhutdinov, R. Generating images with multimodal language models. _arXiv preprint arXiv:2305.17216_ , 2023. 

- Li, J., Li, D., Savarese, S., and Hoi, S. Blip-2: Bootstrapping language-image pre-training with frozen image encoders and large language models. _arXiv preprint arXiv:2301.12597_ , 2023a. 

- Li, Y., Liu, H., Wu, Q., Mu, F., Yang, J., Gao, J., Li, C., and Lee, Y. J. Gligen: Open-set grounded text-to-image generation. In _Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition_ , pp. 22511– 22521, 2023b. 

- Li, Y., Zhang, C., Yu, G., Wang, Z., Fu, B., Lin, G., Shen, C., Chen, L., and Wei, Y. Stablellava: Enhanced visual instruction tuning with synthesized image-dialogue data. _arXiv preprint arXiv:2308.10253_ , 2023c. 

- Lian, L., Li, B., Yala, A., and Darrell, T. Llm-grounded diffusion: Enhancing prompt understanding of text-to-image diffusion models with large language models. _arXiv preprint arXiv:2305.13655_ , 2023. 

- Liu, H., Li, C., Wu, Q., and Lee, Y. J. Visual instruction tuning. _arXiv preprint arXiv:2304.08485_ , 2023. 

- Liu, N., Li, S., Du, Y., Torralba, A., and Tenenbaum, J. B. Compositional visual generation with composable diffusion models. In _European Conference on Computer Vision_ , pp. 423–439. Springer, 2022. 

- Mou, C., Wang, X., Xie, L., Zhang, J., Qi, Z., Shan, Y., and Qie, X. T2i-adapter: Learning adapters to dig out more controllable ability for text-to-image diffusion models. _arXiv preprint arXiv:2302.08453_ , 2023. 

- Muennighoff, N., Wang, T., Sutawika, L., Roberts, A., Biderman, S., Scao, T. L., Bari, M. S., Shen, S., Yong, Z.-X., Schoelkopf, H., et al. Crosslingual generalization through multitask finetuning. _arXiv preprint arXiv:2211.01786_ , 2022. 

- Nichol, A., Dhariwal, P., Ramesh, A., Shyam, P., Mishkin, P., McGrew, B., Sutskever, I., and Chen, M. Glide: Towards photorealistic image generation and editing with text-guided diffusion models. _arXiv preprint arXiv:2112.10741_ , 2021. 

- OpenAI, R. Gpt-4 technical report. arxiv 2303.08774. _View in Article_ , 2:3, 2023. 

- Ouyang, L., Wu, J., Jiang, X., Almeida, D., Wainwright, C., Mishkin, P., Zhang, C., Agarwal, S., Slama, K., Ray, A., et al. Training language models to follow instructions with human feedback. _Advances in Neural Information Processing Systems_ , 35:27730–27744, 2022. 

- Pan, X., Dong, L., Huang, S., Peng, Z., Chen, W., and Wei, F. Kosmos-g: Generating images in context with multimodal large language models. _arXiv preprint arXiv:2310.02992_ , 2023. 

- Podell, D., English, Z., Lacey, K., Blattmann, A., Dockhorn, T., Muller,¨ J., Penna, J., and Rombach, R. Sdxl: Improving latent diffusion models for high-resolution image synthesis. _arXiv preprint arXiv:2307.01952_ , 2023. 

- Qu, L., Wu, S., Fei, H., Nie, L., and Chua, T.-S. Layoutllmt2i: Eliciting layout guidance from llm for text-to-image generation. In _Proceedings of the 31st ACM International Conference on Multimedia_ , pp. 643–654, 2023. 

- Radford, A., Kim, J. W., Hallacy, C., Ramesh, A., Goh, G., Agarwal, S., Sastry, G., Askell, A., Mishkin, P., Clark, J., et al. Learning transferable visual models from natural language supervision. In _International conference on machine learning_ , pp. 8748–8763. PMLR, 2021. 

- Raffel, C., Shazeer, N., Roberts, A., Lee, K., Narang, S., Matena, M., Zhou, Y., Li, W., and Liu, P. J. Exploring the limits of transfer learning with a unified text-to-text transformer. _The Journal of Machine Learning Research_ , 21(1):5485–5551, 2020. 

- Ramesh, A., Dhariwal, P., Nichol, A., Chu, C., and Chen, M. Hierarchical text-conditional image generation with clip latents. _arXiv preprint arXiv:2204.06125_ , 1(2):3, 2022. 

- Rassin, R., Hirsch, E., Glickman, D., Ravfogel, S., Goldberg, Y., and Chechik, G. Linguistic binding in diffusion models: Enhancing attribute correspondence through attention map alignment. _arXiv preprint arXiv:2306.08877_ , 2023. 

- Reed, S., Akata, Z., Yan, X., Logeswaran, L., Schiele, B., and Lee, H. Generative adversarial text to image synthesis. In _International conference on machine learning_ , pp. 1060–1069. PMLR, 2016. 

- Rombach, R., Blattmann, A., Lorenz, D., Esser, P., and Ommer, B. High-resolution image synthesis with latent diffusion models. In _Proceedings of the IEEE/CVF conference on computer vision and pattern recognition_ , pp. 10684–10695, 2022. 

15 

**Mastering Text-to-Image Diffusion: Recaptioning, Planning, and Generating with Multimodal LLMs** 

- Saharia, C., Chan, W., Saxena, S., Li, L., Whang, J., Denton, E. L., Ghasemipour, K., Gontijo Lopes, R., Karagol Ayan, B., Salimans, T., et al. Photorealistic text-to-image diffusion models with deep language understanding. _Advances in Neural Information Processing Systems_ , 35: 36479–36494, 2022. 

- Sohl-Dickstein, J., Weiss, E., Maheswaranathan, N., and Ganguli, S. Deep unsupervised learning using nonequilibrium thermodynamics. In _International conference on machine learning_ , pp. 2256–2265. PMLR, 2015. 

- Song, Y. and Ermon, S. Generative modeling by estimating gradients of the data distribution. _Advances in neural information processing systems_ , 32, 2019. 

- Song, Y. and Ermon, S. Improved techniques for training score-based generative models. _Advances in neural information processing systems_ , 33:12438–12448, 2020. 

- Song, Y., Sohl-Dickstein, J., Kingma, D. P., Kumar, A., Ermon, S., and Poole, B. Score-based generative modeling through stochastic differential equations. _arXiv preprint arXiv:2011.13456_ , 2020. 

- Stiennon, N., Ouyang, L., Wu, J., Ziegler, D., Lowe, R., Voss, C., Radford, A., Amodei, D., and Christiano, P. F. Learning to summarize with human feedback. _Advances in Neural Information Processing Systems_ , 33: 3008–3021, 2020. 

- Sun, J., Fu, D., Hu, Y., Wang, S., Rassin, R., Juan, D.C., Alon, D., Herrmann, C., van Steenkiste, S., Krishna, R., et al. Dreamsync: Aligning text-to-image generation with image understanding feedback. _arXiv preprint arXiv:2311.17946_ , 2023a. 

- Sun, Q., Yu, Q., Cui, Y., Zhang, F., Zhang, X., Wang, Y., Gao, H., Liu, J., Huang, T., and Wang, X. Generative pretraining in multimodality. _arXiv preprint arXiv:2307.05222_ , 2023b. 

- Sur´ıs, D., Menon, S., and Vondrick, C. Vipergpt: Visual inference via python execution for reasoning. _arXiv preprint arXiv:2303.08128_ , 2023. 

- Taylor, R., Kardas, M., Cucurull, G., Scialom, T., Hartshorn, A., Saravia, E., Poulton, A., Kerkez, V., and Stojnic, R. Galactica: A large language model for science. _arXiv preprint arXiv:2211.09085_ , 2022. 

- Team, G., Anil, R., Borgeaud, S., Wu, Y., Alayrac, J.-B., Yu, J., Soricut, R., Schalkwyk, J., Dai, A. M., Hauth, A., et al. Gemini: a family of highly capable multimodal models. _arXiv preprint arXiv:2312.11805_ , 2023. 

- Touvron, H., Lavril, T., Izacard, G., Martinet, X., Lachaux, M.-A., Lacroix, T., Roziere, B., Goyal, N., Hambro, E.,` 

- Azhar, F., et al. Llama: Open and efficient foundation language models. _arXiv preprint arXiv:2302.13971_ , 2023a. 

- Touvron, H., Martin, L., Stone, K., Albert, P., Almahairi, A., Babaei, Y., Bashlykov, N., Batra, S., Bhargava, P., Bhosale, S., et al. Llama 2: Open foundation and finetuned chat models. _arXiv preprint arXiv:2307.09288_ , 2023b. 

- Wang, R., Chen, Z., Chen, C., Ma, J., Lu, H., and Lin, X. Compositional text-to-image synthesis with attention map control of diffusion models. _arXiv preprint arXiv:2305.13921_ , 2023. 

- Workshop, B., Scao, T. L., Fan, A., Akiki, C., Pavlick, E., Ilic, S., Hesslow, D., Castagn´ e, R., Luccioni, A. S., Yvon,´ F., et al. Bloom: A 176b-parameter open-access multilingual language model. _arXiv preprint arXiv:2211.05100_ , 2022. 

- Wu, C., Yin, S., Qi, W., Wang, X., Tang, Z., and Duan, N. Visual chatgpt: Talking, drawing and editing with visual foundation models. _arXiv preprint arXiv:2303.04671_ , 2023a. 

- Wu, T.-H., Lian, L., Gonzalez, J. E., Li, B., and Darrell, T. Self-correcting llm-controlled diffusion models. _arXiv preprint arXiv:2311.16090_ , 2023b. 

- Xie, J., Li, Y., Huang, Y., Liu, H., Zhang, W., Zheng, Y., and Shou, M. Z. Boxdiff: Text-to-image synthesis with training-free box-constrained diffusion. In _Proceedings of the IEEE/CVF International Conference on Computer Vision_ , pp. 7452–7461, 2023. 

- Xu, J., Liu, X., Wu, Y., Tong, Y., Li, Q., Ding, M., Tang, J., and Dong, Y. Imagereward: Learning and evaluating human preferences for text-to-image generation. _arXiv preprint arXiv:2304.05977_ , 2023. 

- Yang, A., Xiao, B., Wang, B., Zhang, B., Bian, C., Yin, C., Lv, C., Pan, D., Wang, D., Yan, D., et al. Baichuan 2: Open large-scale language models. _arXiv preprint arXiv:2309.10305_ , 2023a. 

- Yang, L., Liu, J., Hong, S., Zhang, Z., Huang, Z., Cai, Z., Zhang, W., and Bin, C. Improving diffusion-based image synthesis with context prediction. In _Thirty-seventh Conference on Neural Information Processing Systems_ , 2023b. 

- Yang, L., Zhang, Z., Song, Y., Hong, S., Xu, R., Zhao, Y., Zhang, W., Cui, B., and Yang, M.-H. Diffusion models: A comprehensive survey of methods and applications. _ACM Computing Surveys_ , 56(4):1–39, 2023c. 

16 

**Mastering Text-to-Image Diffusion: Recaptioning, Planning, and Generating with Multimodal LLMs** 

- Yang, L., Qian, H., Zhang, Z., Liu, J., and Cui, B. Structureguided adversarial training of diffusion models. In _Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition_ , 2024a. 

- Yang, L., Zhang, Z., Yu, Z., Liu, J., Xu, M., Ermon, S., and CUI, B. Cross-modal contextualized diffusion models for text-guided visual generation and editing. In _International Conference on Learning Representations_ , 2024b. 

   - Zou, X., Dou, Z.-Y., Yang, J., Gan, Z., Li, L., Li, C., Dai, X., Behl, H., Wang, J., Yuan, L., et al. Generalized decoding for pixel, image, and language. In _Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition_ , pp. 15116–15127, 2023. 

- Yang, Z., Li, L., Wang, J., Lin, K., Azarnasab, E., Ahmed, F., Liu, Z., Liu, C., Zeng, M., and Wang, L. Mm-react: Prompting chatgpt for multimodal reasoning and action. _arXiv preprint arXiv:2303.11381_ , 2023d. 

- Yang, Z., Wang, J., Gan, Z., Li, L., Lin, K., Wu, C., Duan, N., Liu, Z., Liu, C., Zeng, M., et al. Reco: Regioncontrolled text-to-image generation. In _Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition_ , pp. 14246–14255, 2023e. 

- Yu, L., Shi, B., Pasunuru, R., Muller, B., Golovneva, O., Wang, T., Babu, A., Tang, B., Karrer, B., Sheynin, S., et al. Scaling autoregressive multi-modal models: Pretraining and instruction tuning. _arXiv preprint arXiv:2309.02591_ , 2023. 

- Zeng, A., Liu, X., Du, Z., Wang, Z., Lai, H., Ding, M., Yang, Z., Xu, Y., Zheng, W., Xia, X., et al. Glm-130b: An open bilingual pre-trained model. _arXiv preprint arXiv:2210.02414_ , 2022. 

- Zhang, L., Rao, A., and Agrawala, M. Adding conditional control to text-to-image diffusion models. In _Proceedings of the IEEE/CVF International Conference on Computer Vision_ , pp. 3836–3847, 2023a. 

- Zhang, S., Roller, S., Goyal, N., Artetxe, M., Chen, M., Chen, S., Dewan, C., Diab, M., Li, X., Lin, X. V., et al. Opt: Open pre-trained transformer language models. _arXiv preprint arXiv:2205.01068_ , 2022. 

- Zhang, T., Zhang, Y., Vineet, V., Joshi, N., and Wang, X. Controllable text-to-image generation with gpt-4. _arXiv preprint arXiv:2305.18583_ , 2023b. 

- Zhang, Y., Zhang, R., Gu, J., Zhou, Y., Lipka, N., Yang, D., and Sun, T. Enhanced visual instruction tuning for textrich image understanding. In _NeurIPS 2023 Workshop on Instruction Tuning and Instruction Following_ , 2023c. 

- Zhang, Z., Zhang, A., Li, M., Zhao, H., Karypis, G., and Smola, A. Multimodal chain-of-thought reasoning in language models. _arXiv preprint arXiv:2302.00923_ , 2023d. 

- Zhu, D., Chen, J., Shen, X., Li, X., and Elhoseiny, M. Minigpt-4: Enhancing vision-language understanding with advanced large language models. _arXiv preprint arXiv:2304.10592_ , 2023. 

17 

