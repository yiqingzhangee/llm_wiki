Published as a conference paper at ICLR 2024 

## - FINE TUNING MULTIMODAL LLMS TO FOLLOW - ZERO SHOT DEMONSTRATIVE INSTRUCTIONS 

**Juncheng Li**[1, 2] _[∗]_ **Kaihang Pan**[1] _[∗]_ **Zhiqi Ge**[1] _[∗]_ **Minghe Gao**[1] _[∗]_ **Wei Ji**[2] **Wenqiao Zhang**[1] **Tat-Seng Chua**[2] **Siliang Tang**[1] _[†]_ **Hanwang Zhang**[3] **Yueting Zhuang**[1] _[†]_ 1Zhejiang University, 2National University of Singapore, 3Nanyang Technological University 

## ABSTRACT 

Recent advancements in Multimodal Large Language Models (MLLMs) have been utilizing Visual Prompt Generators (VPGs) to convert visual features into tokens that LLMs can recognize. This is achieved by training the VPGs on millions of image-caption pairs, where the VPG-generated tokens of images are fed into a frozen LLM to generate the corresponding captions. However, this image-captioning based training objective inherently biases the VPG to concentrate solely on the primary visual contents sufficient for caption generation, often neglecting other visual details. This shortcoming results in MLLMs’ underperformance in comprehending demonstrative instructions consisting of multiple, interleaved, and multimodal instructions that demonstrate the required context to complete a task. To address this issue, we introduce a generic and lightweight Visual Prompt Generator Complete module (VPG-C), which can infer and complete the missing details essential for comprehending demonstrative instructions. Further, we propose a synthetic discriminative training strategy to fine-tune VPG-C, eliminating the need for supervised demonstrative instructions. As for evaluation, we build DEMON, a comprehensive benchmark for demonstrative instruction understanding. Synthetically trained with the proposed strategy, VPG-C achieves significantly stronger zero-shot performance across all tasks of DEMON. Further evaluation on the MME and OwlEval benchmarks also demonstrate the superiority of VPG-C. The code and models are available at https://github.com/DCDmllm/Cheetah. 

## 1 INTRODUCTION 

Recent advances in Multimodal Large Language Models (MLLMs) (Li et al., 2023c; Liu et al., 2023; Zhu et al., 2023a) have exhibited promising capabilities in processing single-image instructions, such as producing detailed image descriptions and answering questions about the image. However, they fall short in **demonstrative instructions** consisting of multiple, interleaved, and multimodal instructions that demonstrate the required context to complete a task. For instance, the instruction in Figure 1 contains interleaved visual and textual context, requiring the model to determine the authenticity of the milk in the second image based on the official image provided in the first. 

An MLLM should at least have the following two capabilities to comprehend demonstrative instructions effectively: 

**1) Not just the primary subject:** Beyond focusing on the primary visual content, it should be able to meticulously discern the details within the demonstrations. These details, complementing the primary content, play a crucial role in semantically connecting the instructions. A case in point is Figure 1, wherein accurate discernment relies on recognizing the logo detail on a milk carton. 

**2) Reasoning-aware details:** How to decide what details are complementary to the reasoning? We expect that an MLLM may “think twice”, that is, given a preliminary reasoning using the primary contents, it would know what additional contents are needed as complementary details. For example, in Figure 1, after preliminary reasoning, the model should re-attend details such as the logo and 

> _∗_ Equal Contribution. _†_ Corresponding Authors. 

1 

Published as a conference paper at ICLR 2024 


![](raw/papers/2_LLM_NLP/LLM_paper_list/zero-shot/iclr_2024_Fine-tuning_Multimodal_LLMs_to_Follow_Zero-shot_Demonstrative_Instructions_images/iclr_2024_Fine-tuning_Multimodal_LLMs_to_Follow_Zero-shot_Demonstrative_Instructions.pdf-0002-01.png)


**----- Start of picture text -----**<br>
Demonstrative Instruction<br>Compared to the genuine milk displayed in                             , please examine the packaging design of  my purchased milk shown in                         . Is the purchased milk  genuine?<br>InstructBLIP-Answer:  Yes, the purchased milk is genuine. Ours-Answer:  No, the cow icon within the red diamond-shaped box on the purchased milk packaging is<br>noticeably different from that of the genuine milk.<br>Visualized attention maps of the InstructBLIP’s VPG. Stage1: initial attention maps Stage2: attention maps after preliminary reasoning<br>**----- End of picture text -----**<br>


Figure 1: An example of InstructBLIP (Dai et al., 2023) and our MLLM enhanced by VPG-C. 

brand name on the milk carton, thereby discerning its authenticity. However, to follow zero-shot demonstrative instructions, this “reasoning-aware” capability should be acquired without the need for supervised demonstrative instructions. 

Unfortunately, we find that the reason why existing MLLMs are not effective in demonstrative instructions is due to the lack of the above capabilities. More specifically, the crux lies in the Visual Prompt Generator (VPG) in MLLMs. VPG, such as Q-former (Li et al., 2023c) and Resampler (Alayrac et al., 2022), translates visual features into tokens recognizable by LLMs, and the translation is trained on millions of image-caption pairs by feeding the VPG-generated tokens of images into a frozen LLM which generates the corresponding captions. However, this image captioning training strategy inevitably introduces the inductive bias that VPG only focuses on the primary visual contents which are just enough for the captioning task, but tends to omit other visual details. For example in Figure 1, the averaged attention map of InstructBLIP (Dai et al., 2023) (Figure 1) shows a dominant focus on the primary contents, neglecting the logo detail, which is however the key to answering the question. 

To this end, we propose a lightweight Visual _Residual Visual Details_ Prompt Generator Complete module (VPG-C), ··· which can infer and complete the missing deVisual tailstive instructionsessential for(Sectioncomprehending2.1). As showndemonstra-FigEncoderImage GeneratorPrompt ··· ⨁ Output ure 2, 1) VPG-C first derives the instruction··· _Guidance_ ×[𝐿] ×[𝐿] specific guidance by intercepting the interme2 **LLM** 2 diate LLM’s output of the primary contents extracted by a conventional VPG, and then Figure 2: An overview of VPG-C. 2) guides the VPG to recover the missing visual residual details. Finally, 3) these residual details are then seamlessly reintegrated into the intermediate LLM’s layer via a skip connection. Together with the original intermediate output, VPG-C is expected to provide an improved comprehension of the demonstration instructions. Yet, VPG-C is not ready to follow zero-shot demonstrative instructions because the “Guide” step requires fine-tuning to specialize in missing detail recovery. Therefore, we propose a synthetic discriminative training strategy to fine-tune VPG-C, without the need for the expensive data collection of “detail-caption” pairs (Section 2.2). 

To evaluate VPG-C and diagnose existing MLLMs, we build DEMON, a comprehensive benchmark for demonstrative instruction understanding, covering 31 diverse tasks across 7 categories, as shown in Figure 4 (Section 3). Systematic evaluation on DEMON confirms the limitation of existing MLLMs in demonstrative instructions. Without additional demonstrative instruction data, the lightweight VPG-C module can be effectively tuned by the synthetic training strategy in several hours with a single A100 GPU. While computation- and data- efficient, VPG-C significantly outperforms existing MLLMs on the DEMON benchmark. Zero-shot evaluation on other multimodal instruction benchmarks (Fu et al., 2023; Ye et al., 2023) also indicates considerable improvement by VPG-C. 

## 2 METHOD 

## 2.1 VISUAL PROMPT GENERATOR COMPLETE 

As illustrated in Figure 2, VPG-C is built upon the frozen LLM (Vicuna-7B (Chiang et al., 2023)) and vision encoder (EVA-CLIP (Fang et al., 2023)). We adopt the widely used Q-Former from BLIP- 

2 

Published as a conference paper at ICLR 2024 

2 (Li et al., 2023c) as our visual prompt generator. VPG-C first uses the intermediate output of the LLM to infer instruction-specific guidance. This then assists the VPG in attending to the missing visual details from the images. By merging these residual details back via a skip connection, VPG-C achieves a thorough grasp of the demonstrative instruction. 

Given a demonstrative instruction, we first adopt the Q-former to generate general visual prompts for each image in the instruction. Q-former takes a fixed number of _K_ query vectors to interact with image features by several cross-attention layers, and the output query representations are used as visual prompts, which are inserted into the position of their corresponding images in the instruction. We denote the input instruction for the language decoder as _H_[0] = _{_ **h**[0] 1 _[,]_ **[ h]**[0] 2 _[, ...,]_ **[ v]** 11[0] _[, ...,]_ **[ v]** 1[0] _K[, ...,]_ **[ h]** _i_[0] _[, ...,]_ **[ v]** _j_[0] 1 _[, ...,]_ **[ v]** _jK_[0] _[, ...,]_ **[ h]** _N_[0] _[}]_[,][where] **[h]**[0] _i_[represents][the] _[i]_[-th][text][token] and _Vj_[0][=] _[ {]_ **[v]** _j_[0] 1 _[, ...,]_ **[ v]** _jK_[0] _[}]_[ represents the] _[ K]_[ visual prompts for the] _[ j]_[-th interleaved image.][Taking the] instruction as input to the _L_ -layer language decoder, we then extract the hidden representation of the last input token **h** _[L/] N_[2] at the _[L]_ 2[-th layer, which can fully perceive the whole multimodal context] during the first _[L]_ 2[layers][and][contains][comprehensive][instruction-aware][semantics.][Next,][we][infer] the instruction-specific guidance **g** from **h** _[L/] N_[2] via a linear projection layer: **g** = **Linear** ( **h** _[L/] N_[2][)][.] 

After obtaining the instruction-specific guidance from the language decoder, we compose it with a new set of learnable queries: **g** + _Q_ , where _Q ∈_ **R** _[K][×][d]_ and **g** is added to each query of _Q_ . Then, we reuse the same Q-former with the above conditionally generated queries to attend to residual visual details, thus obtaining the visual prompts _V j_ = ~~_{_~~ **v** _j_ 1 _, ...,_ **v** _jK}_ for each image _j_ , which contains the complementary details missed by the original visual prompts. Finally, the transformed _V j_ is reintegrated with the corresponding original intermediate representations of _Vj[L/]_[2] , via skip connection: _V_[˜] _j[L/]_[2] = _Vj[L/]_[2] + **Linear** ( _V j_ ), which is taken as the input to the ( _[L]_ 2[+ 1)][-th layer.] 

**Efficient training.** Our VPG-C module is parameter-efficient as the Q-former is frozen and only a set of query embeddings and two linear projection layers need to be fine-tuned, which only account for **0.09% (** _∼_ **6.3M)** of the entire model. To stabilize the training process (Zhang & Agrawala, 2023), we initialize the linear projection layers with zeros. Thus, at the early training stage, the input to the ( _[L]_ 2[+ 1)][-th layer can be converted to:] _[V]_[˜] _j[L/]_[2] = _Vj[L/]_[2] , which will not cause any influence on LLMs. 

**Analysis on inserting VPG-C in the intermediate layer (** _[L]_ 2 **[):][1) Guidance generation.]**[Previous] studies have shown that features provided by the intermediate layer may suffice to preliminarily understand the given input samples (Xin et al., 2020) and can serve as guidance hints to improve training (Romero et al., 2014). Thus, generating guidance in the intermediate layer allows the model to form a preliminary understanding of the given instruction. Generating guidance too early could be problematic, as the model might not have gathered sufficient contextual information to generate effective guidance. Conversely, generating guidance too late could result in the model’s attention being narrowly focused on what it perceives as the final answer, hindering its ability to guide the Q-former to extract relevant details from the images. Therefore, placing the guidance generation step in the intermediate layer strikes a balance. **2) Detail reintegration.** Intermediate-layer reintegration of residual visual details preserves prior knowledge and allows subsequent layers to integrate new information effectively. Reintegrating residual details too early in the pipeline might overwrite important context, while reintegrating it too late could limit the impact on the model’s reasoning. Therefore, the intermediate layer offers a strategic position for residual details reintegration, enabling the model to reason effectively and arrive at the correct answers by leveraging the complemented visual residual details. We further provide **quantitative analysis** in Section 4.4. 

## 2.2 SYNTHETIC DISCRIMINATIVE TRAINING STRATEGY 

The proposed training strategy diagnoses the areas initially ignored by Q-former according to its cross-attention maps between the queries and the image features, and generates a synthetic image by performing several types of editing on the ignored areas. Then, an inter-image discriminative task is formulated as describing the subtle difference between the original and the synthetic images. Considering the edits are performed in the mostly ignored areas, VPG-C is challenged to recover the missing details to describe the difference. An overview is illustrated in Figure 3. 

**Editing target identification.** The Q-former takes the queries to interact with frozen image features through several cross-attention layers and uses the output query representations as the visual 

3 

Published as a conference paper at ICLR 2024 


![](raw/papers/2_LLM_NLP/LLM_paper_list/zero-shot/iclr_2024_Fine-tuning_Multimodal_LLMs_to_Follow_Zero-shot_Demonstrative_Instructions_images/iclr_2024_Fine-tuning_Multimodal_LLMs_to_Follow_Zero-shot_Demonstrative_Instructions.pdf-0004-01.png)


**----- Start of picture text -----**<br>
Deleting Object Modifying Object Instruction:<br>Generate suitable  Describe the difference between the images.<br>editing description 1 2<br>Segment<br>Anything Blended<br>Diffusion<br>Input Image Segment Map “ remove…” “ red…”<br>Original Image Edited Image<br>Adding Object Swaping Object<br>ignored MLLM<br>Q-Former areas<br>Response:<br>In the first image, the car on the road is yellow,<br>Attention Map Editing Target whereas in the second image, it is red.<br>“ stones…” “ horse…”<br>Cross-Attention Guided Synthetic Image Editing Inter-Image Discriminative Training<br>**----- End of picture text -----**<br>


Figure 3: Pipeline demonstration of synthetic discriminative training strategy for VPG-C. 

prompts. Therefore, the cross-attention maps between queries and image features reflect the interest of queries. We average the cross-attention maps across all layers and all queries to obtain the global cross-attention map _A_ , where the value _Aij_ indicates the significance degree of the corresponding image feature by the original task-agnostic Q-former queries. After that, we employ the advanced vision foundation model (Kirillov et al., 2023) to obtain all the objects with segmentation masks in the image. Then, the significance degree of each object Φ( _oi_ ) is computed based on the crossattention map _A_ with RoIAlign (He et al., 2017), where we average the values of _A_ within the mask _mi_ by interpolation. Φ( _oi_ ) reflects what degree the visual features of object _oi_ is extracted by the Q-former. Thus, we select the most ignored objects based on the Φ( _oi_ ) value. 

**Editing description generation.** We define four types of editing: _modifying objects, swapping objects, deleting objects, and adding objects._ Given the selected object, we instruct ChatGPT (OpenAI, 2023a) to generate a suitable editing description that is in harmony with the context, where ChatGPT is prompted with the corresponding image caption and detailed object information ( _i.e.,_ labels, positions). For adding objects, we only select BACKGROUND objects to add objects. 

**Synthetic image generation.** After obtaining the editing description, we generate the synthetic image using a text-to-image latent diffusion model ( _i.e.,_ Blended Diffusion (Avrahami et al., 2022)). Blended Diffusion performs local editing on the image according to the target object mask and the editing description, thus rendering the synthetic image. To ensure quality, we filter the edited images using CLIP similarity (Radford et al., 2021b). 

**Inter-image discriminative training.** Given the original image and the synthetic image pair, along with the task instruction ( _“Describe the difference between the images”_ ), the inter-image discriminative training task is defined as generating sentences to describe the subtle difference between the images. We convert the editing description to acquire the ground-truth sentences. 

## 3 DEMON BENCHMARK 

**Data format.** All task instances are transformed into a unified instruction-response form for zeroshot evaluation. Formally, each instance in DEMON consists of the following components: 

- Task_Instruction: provides a complete natural language definition of a given task, including the input/output format and the task objective. 

- Task_Instance: is a concrete instance of a given task that consists of demonstrative imagetext sequential context ( _e.g._ , visually-rich textbooks, specific questions about the context). 

- Response: represents the target output in natural language for a given task instruction and task instance. For classification tasks, we convert the class labels as options into the instruction and ask the model to output the option index in natural language as the response. 

Without any specific emphasis, we use the term “instruction” to refer to the combination of Task_Instruction and Task_Instance. For each task, we manually design 10 Task_Instruction templates in natural language to increase the diversity. 

**Task collection and categorization.** To comprehensively benchmark the demonstrative instruction following ability, we extensively gather a wide variety of multimodal datasets from different fields and scenarios. As illustrated in Figure 4, DEMON has three important properties: **1) Demonstrative vision-language context,** all the instructions contain sequences of inter-related images and texts, such as storyboards with scripts, and textbooks with diagrams. **2) Diverse forms of complex** 

4 

Published as a conference paper at ICLR 2024 


![](raw/papers/2_LLM_NLP/LLM_paper_list/zero-shot/iclr_2024_Fine-tuning_Multimodal_LLMs_to_Follow_Zero-shot_Demonstrative_Instructions_images/iclr_2024_Fine-tuning_Multimodal_LLMs_to_Follow_Zero-shot_Demonstrative_Instructions.pdf-0005-01.png)


**----- Start of picture text -----**<br>
Category : Visual Storytelling Category : Multi-Modal Dialogue<br>Instruction:  Given the stories paired with the first several images, can you finish the story based on the last image? Instructions & Responses :<br>1 2 3 4 Folks of all ages come out  5 As an intelligent assistant, your ultimate objective is to<br>to peruse the crafts for sale. place a soap container on a counter . Start by<br>carefully observing your surroundings and making<br>decisions step by step.<br>The local parish holds a craft show each year. Lots of folks come out and set up tables to sell their crafts. Some of these crafts are very uniqueand take a lot of talent to make. Folks of all ages come out to peruse the crafts for sale. 1 What should be done next?<br>Response:  Some of the crafters even dress up in unique costumes as part of their selling act.<br>Category : Visual Relation Inference Category : Text-Rich Images QA Turn left and walk to the right sink.<br>Instruction: What modifications can be observed  Instruction:  Please analyze the data in these slides and accordingly answer the proposed question. 2 What’s the next step?<br>between these two pictures?  1 2<br>1 2<br>Open the bottom left cabinet and<br>take out the closest soap container.<br>3 OK. What follows?<br>Response: Question: How many profiles are viewed every day on the social<br>There are three people in the photo on the left and none in the photo on  website that has over 350 million members and where the average user has 240 connections? Take a step left.<br>the right. Response:  Over 25 million. …<br>Category : Multi-Image Reasoning Category : Multi-Modal Cloze<br>Instruction: Instruction:  With a sequence of comic panels provided, could you<br>Are these three pictures of clothes similar in their style?  identify the most relevant next panel from the available choices?<br>1 2 3 1 2 3<br>A B C D<br>Response:<br>Yes. They are in similar style. Response:  Panel C.<br>Category : Knowledge Grounded QA<br>Instruction:  Provided with a series of diagrams from a textbook, your responsibility is to correctly answer the following question.<br>1 2 Folks of all ages come out to peruse the crafts for sale.3 The graph shows an oxygendissociation curve for A. a decrease in carbon dioxide, increase inpH, or a decrease in temperature<br>hemoglobin. Based on the B. an increase in carbon dioxide, increase in<br>graph, what would likely pH, or a decrease in temperature<br>cause the curve to shift to C. a decrease in carbon dioxide, decrease in<br>the left, as shown by the pH, or a decrease in temperature<br>dotted plot line? D. a decrease in carbon dioxide, increase in<br>pH, or an increase in temperature<br>Response:  A. a decrease in carbon dioxide, increase in pH, or a decrease in temperature.<br>**----- End of picture text -----**<br>


Figure 4: Demonstrations and task taxonomy of the proposed DEMON benchmark. 

**instructions,** the instructions range from designing panels for comics, to discovering differences between surveillance images, and to conversational embodied tasks. **3) Vast range of instructionfollowing scenarios,** the benchmark covers multiple practical scenarios, including cartoons, industrial visuals, driving recordings, recipes, _etc_ . 

**Evaluation protocols.** Thanks to the unified task format of DEMON, all tasks can be evaluated in a zero-shot manner. For the open-ended generation tasks, we adopt _ROUGE-L_ for evaluation. For the tasks that require the models to output option indexes, we take _Accuracy_ as the evaluation metric. While well-formated options are provided, we empirically observe that many MLLMs struggle to strictly follow instructions to output the option indexes but generate free-form text. Thus, when models do not exactly output the required options, we match their outputs to one of the given options based on the TF-IDF distance, which we find is more robust than model-based methods (OpenAI, 2023a; Reimers & Gurevych, 2019). Since we explore a large number of tasks, we take a maximum of 500 instances per task for evaluation efficiency and exclude several datasets that are difficult to obtain and are subject to strict copyright restrictions (referred to as DEMON-Core). Meanwhile, we report the full version of the benchmark to facilitate future research on large-scale multimodal instruction tuning (referred to as DEMON-Full). Unless specifically stated, we use DEMON to refer to DEMON-Core in the following. 

Table 1: Detailed statistics of DEMON benchmark. 

||Tasks|Scenarios|Images|Instructions|Avg.|Images / Instruction Avg. Words / Instruction|Images / Instruction Avg. Words / Instruction|
|---|---|---|---|---|---|---|---|
|DEMON-Core|29|19|62.81K|18.18K||3.46|92.69|
|DEMON-Full|31|20|1.77M|477.72K||3.70|97.58|



**Benchmark analysis.** Table 1 details the statistics. DEMON benchmark covers 31 tasks of 7 categories across 20 scenarios. In total, DEMON-Full includes 477.72K instruction-response pairs, 

5 

Published as a conference paper at ICLR 2024 

Table 2: Average results of zero-shot evaluation on each task category of DEMON Benchmark. 

||Multimodal<br>Visual<br>Visual Relation<br>Multimodal<br>Knowledge<br>Text-Rich<br>Multi-Image|
|---|---|
||Dialogue<br>Storytelling<br>Inference<br>Cloze<br>GroundedQA<br>ImagesQA<br>Reasoning|
|BLIP-2 (Li et al., 2023c)<br>InstructBLIP (Dai et al., 2023)<br>LLaMA-Adapter V2 (Gao et al., 2023)<br>LLaVA (Liu et al., 2023)<br>MiniGPT-4 (Zhu et al., 2023a)<br>mPLUG-Owl (Ye et al., 2023)<br>OpenFlamingo (Awadalla et al., 2023)<br>Otter(Li et al.,2023a)|26.12<br>21.31<br>10.67<br>17.94<br>39.23<br>33.53<br>39.65<br>33.58<br>24.41<br>11.49<br>21.20<br>47.40<br>44.40<br>48.55<br>14.22<br>17.57<br>13.51<br>18.00<br>44.80<br>32.00<br>44.03<br>7.79<br>10.70<br>8.27<br>15.85<br>36.20<br>28.33<br>41.53<br>13.69<br>17.07<br>7.95<br>16.60<br>30.27<br>26.40<br>43.50<br>12.67<br>19.33<br>5.40<br>16.25<br>33.27<br>32.47<br>42.50<br>16.88<br>24.22<br>13.85<br>21.65<br>32.00<br>30.60<br>41.63<br>15.37<br>15.57<br>11.39<br>16.00<br>41.67<br>27.73<br>43.85|
|**VPG-C**|**37.50**<br>**25.20**<br>**25.90**<br>**22.15**<br>**48.60**<br>**44.93**<br>**50.28**|



serving as a large-scale benchmark for demonstrative instruction following. On average, each instruction contains 3.70 images and 97.58 words. Please refer to Appendix B for more details. 

## 4 EXPERIMENTS 

## 4.1 ZERO-SHOT EVALUATION ON DEMON BENCHMARK 

**Comparison with advanced MLLMs.** In this section, we conduct comprehensive evaluation of our VPG-C and the recent advanced MLLMs on the proposed DEMON benchmark. For all methods, we choose versions with parameter sizes less than 10B. Please refer to Appendix D, F for details. The average results of each task category are summarized in Table 2, which indicates the following. 

- VPG-C consistently outperforms existing models by a large margin across all categories, which demonstrates the stronger generalizability to follow such complicated demonstrative instructions. 

- While previous works mainly fine-tune on massive multimodal instruction data, VPG-C still achieves the highest performance using synthetic training data with much lower computation cost. This validates the effectiveness of the proposed VPG-C module and its synthetic training strategy. 

- Compared with previous works that fine-tune the large-scale language decoder or visual encoder ( _i.e.,_ LLaVA, mPLUG-Owl), our model only tunes the lightweight VPG-C module with 6.3M parameters and achieves significant performance gain. 

- VPG-C exhibits significant superiority in several challenging tasks. For instance, VPG-C surpasses SOTA methods by 3.92% on multimodal dialogue, which requires models to effectively associate the interleaved images mentioned in different turns of the dialogue. 

**Innovative findings.** The extensive evaluation on DEMON benchmark reveals several key findings. 

- **Poor performance on demonstrative instructions.** While several models ( _e.g.,_ OpenFlamingo, Otter, mPLUG-owl) have been trained on interleaved vision-language data, such as mmc4 (Zhu et al., 2023b), they still struggle to perform well on the demonstraive instructions. We suppose that while mmc4 contains sequences of interleaved images as context, the web-crawled images are often weakly related. In contrast, the images and text in demonstrative instructions are highly related, requiring models to deeply associate them to comprehend the task intents. 

- **Limited instruction following ability.** Despite existing vision-language models leveraging stateof-the-art LLMs, which have demonstrated impressive ability in following language instructions, this competence seems to falter when dealing with complex multimodal instructions. For instance, when tasked with selecting the correct answer from a choice list given the context of images and texts, we observed some models inclining more towards describing the contents of the images instead of addressing the posed questions. This is perceived as a deficiency in the image-text alignment training process, to which we attribute the discrepancy. 

- **Failing to process image-choice questions.** When dealing with multimodal cloze tasks, all models are limited to processing instructions that involve images as options. We hope future work to utilize the new benchmark to make progress on this type of demonstrative instructions. 

## 4.2 ZERO-SHOT EVALUATION ON MME BENCHMARK 

We evaluate our VPG-C on the concurrently proposed MME benchmark (Fu et al., 2023) to further illustrate its strong generalizability to follow a diverse range of single-image instructions. MME benchmark measures both perception and cognition abilities on a total of 14 subtasks. We report the averaged results of perception tasks and cognition tasks in Table 3, respectively. While we 

6 

Published as a conference paper at ICLR 2024 

Table 3: Zero-shot evaluation of perception and cognition abilities on MME benchmark. 

||BLIP-2<br>InstructBLIP<br>LA-V2<br>LLaVA<br>MiniGPT-4<br>mPLUG-Owl<br>Otter|**VPG-C**|
|---|---|---|
|Perception<br>Cognition|1293.84<br>1212.82<br>972.67<br>502.82<br>866.57<br>967.34<br>1292.26<br>290.00<br>291.79<br>248.93<br>214.64<br>292.14<br>276.07<br>306.43|**1299.24**<br>**321.07**|



do not use massive multimodal instruction data to fine-tune VPG-C, VPG-C still achieves superior performance, compared with the supervised instruction-tuned models. This indicates our method effectively overcomes the inherent limitation of VPGs and the completed residual details are also essential for single-image instructions. Please refer to Appendix E for detailed results. 

## 4.3 HUMAN EVALUATION ON GENERAL-PURPOSE LANGUAGE GENERATION 

We further conduct human evaluation on the OwlEval benchmark (Ye et al., 2023), which contains 82 openended questions including advertisement and poem creation, diagram and flowchart comprehension, and teaching, _etc._ Specifically, we recruit 8 well-educated people to rank the randomly shuffled responses from VPG-C, MiniGPT-4, mPLUG-Owl, OpenFlamingo, and InstructBLIP. The scores range from 1 to 5 (5 means best) and are allowed to be equal for comparable instances. As shown in Figure 5, VPG-C also demonstrates better open-ended language generation ability in various practical cases. 


![](raw/papers/2_LLM_NLP/LLM_paper_list/zero-shot/iclr_2024_Fine-tuning_Multimodal_LLMs_to_Follow_Zero-shot_Demonstrative_Instructions_images/iclr_2024_Fine-tuning_Multimodal_LLMs_to_Follow_Zero-shot_Demonstrative_Instructions.pdf-0007-06.png)


**----- Start of picture text -----**<br>
4.19<br>3.62<br>3.24<br>2.53<br>1.69<br>OpenFlamingo InstructBLIP MiniGPT-4 mPLUG-Owl VPG-C<br>Figure 5: Human evaluation.<br>**----- End of picture text -----**<br>


## 4.4 IN-DEPTH ANALYSIS 

**Effectiveness of individual components.** We investigate the effectiveness of each component in Table 4. We start with the backbone model that uses the Q-former to generate visual prompts. **1)** Instead of applying VPG-C to capture missing details, we first attempt a simple heuristic-based method that directly extracts the less attended visual features according to the cross-attention maps of Q-former and reintegrates them to the intermediate layer of the LLM as ours. We fine-tune a linear layer before reintegrating with 0.5 million image-caption pairs. The results of Row 2 show that such a sample heuristic can bring some improvement. This validates _the importance of re-extracting missing visual features from images for comprehending demonstrative instructions_ . **2)** Then, we replace it with VPG-C and train it only using the image-caption pairs without synthetic training. The results of Row 3 demonstrate that _VPG-C can more accurately complete the required missing details by leveraging the intermediate inference results of the LLM._ **3)** However, solely using common image-caption data can not fully unleash the power of VPG-C. Comparing Row 3 and Row 4, we observe a significant improvement for all tasks, indicating that _the proposed synthetic discriminative training strategy can methodically empower VPG-C to extract missing visual details._ 

**VPG-C can better guide VPGs.** Since InstructBLIP can perform conditional visual feature extraction, we implement a variant version that concatenates its initially generated answer with the instruction as condition to re-extract features. The initial generated answer serves as an additional heuristic from the LLM for guiding feature extraction. Then, the newly extracted visual prompts are used to re-generate answers. For a fair comparison, we provide a zero-shot version (Row 5) and a fine-tuned version (Row 6) using synthetic training as ours. As shown in Table 4, directly using synthetic data and inferred answers as heuristic conditions fails to yield a notable improvement. In contrast, VPG-C can better guide the VPG to complete the missing visual details by intercepting the intermediate representations of the LLM. Further, VPG-C is more computation-efficient as it only requires one full forward pass of the LLM, while the InstructBLIP variants require twice. 

**VPG-C works well on various language backbones.** Table 4 also validates that our approach can well cooperate with language backbones of _different families (LLaMA2) and sizes (Vicuna-13B)_ . 

**VPG-C can be implemented with very simple VPG.** As a generic method, VPG-C can be implemented with different VPGs. Beyond the widely used Q-former that is composed of multiple Transformer blocks, we further probe the effectiveness of VPG-C with a simpler VPG, _i.e.,_ Linear Projection, as used in LLaVA (please refer to Appendix C for implementation details). Table 4 Row 7 shows promising results. VPG-C can also significantly bolster the performance of a simple 

7 

Published as a conference paper at ICLR 2024 

Table 4: Ablation results on DEMON Benchmark. 

||Multimodal<br>Visual<br>Visual Relation<br>Multimodal<br>Knowledge<br>Text-Rich<br>Multi-Image|
|---|---|
||Dialogue<br>Storytelling<br>Inference<br>Cloze<br>GroundedQA<br>ImagesQA<br>Reasoning|
|1<br>Backbone<br>2<br>+Heuristic Details<br>3<br>+VPG-C<br>4<br>+Synthetic Training|25.65<br>21.72<br>9.33<br>17.06<br>37.21<br>32.42<br>41.30<br>28.13<br>22.76<br>12.69<br>18.81<br>38.75<br>34.14<br>43.26<br>31.76<br>23.62<br>19.12<br>20.09<br>42.53<br>39.68<br>46.71<br>37.50<br>25.20<br>25.90<br>22.15<br>48.60<br>44.93<br>50.28|
|InstructBLIP<br><br>|33.58<br>24.41<br>11.48<br>21.20<br>47.40<br>44.40<br>48.55<br><br><br><br><br><br><br>|
|5<br>+Answer Condition<br>6<br>+Synthetic Training|32.10<br>23.76<br>11.02<br>21.86<br>47.94<br>42.08<br>49.01<br>31.76<br>24.32<br>12.78<br>19.87<br>46.58<br>42.36<br>49.82|
|LLaVA<br>|7.79<br>10.70<br>8.27<br>15.85<br>36.20<br>28.33<br>41.53<br><br><br><br><br><br><br>|
|7<br>Linear VPG|16.43<br>19.48<br>14.75<br>18.54<br>41.32<br>36.87<br>46.02|
|8<br>VPG-C-LLaMA2-7B<br>9<br>VPG-C-Vicuna-13B|42.70<br>24.76<br>25.50<br>22.95<br>51.00<br>44.93<br>48.68<br>38.14<br>26.59<br>27.15<br>27.15<br>52.93<br>49.33<br>53.65|



linear VPG, verifying the transferability of VPG-C. It is promising to adapt our generic VPG-C and corresponding low-resource synthetic training strategy to different VPGs in the future. 

**Analysis on the inserted layer of VPG-C.** We investigate the impact of inserting VPG-C into different layers of LLMs. We report the averaged accuracy for multiplechoice tasks and averaged ROUGE-L for open-ended generation tasks in Figure 6. We observe that the performance is low when we insert VPG-C too early ( _i.e.,_ 4, 8) as the model might not have gathered sufficient contextual information to generate effective guidance. Meanwhile, inserting VPG-C too late ( _i.e.,_ 24, 28) degenerates the performance. We speculate this is due to the generated guidance being too concentrated and there not being enough layers to integrate the residual details. 

|15<br>20<br>25<br>30<br>35<br>40<br>45<br>5||Accuracy<br>~~ROUGE-L~~|
|---|---|---|
||||
||||
||||
||||
||||
||||
|||4<br>8<br>12<br>16<br>20<br>24<br>28<br>Insertion Layer of VPG-C|
|Figure 6: Performance on DEMON with<br>different insertion layers.<br>Table 5: Effciency analysis of<br>synthetic training.<br>Accuracy<br>ROUGE-L<br>16K<br>38.93<br>25.67<br>32K<br>39.62<br>27.38<br>48K<br>40.45<br>28.81<br>64K<br>**41.49**<br>**29.53**<br>80K<br>41.62<br>29.73<br>96K<br>40.12<br>28.31<br>ynthetic<br> tasks in<br>missing<br>demon-<br>f differ-<br>le 5, the<br>creased<br>|||



**Synthetic training is data-efficient.** Since our proposed synthetic Table 5: Efficiency analysis of training strategy can construct challenging discriminative tasks in synthetic training. a targeted manner, enhancing VPG-C’s ability to complete missing Accuracy ROUGE-L details, it avoids the need for a large amount of supervised demon16K 38.93 25.67 32K 39.62 27.38 strative instruction data. We further investigate the impact of differ48K 40.45 28.81 ent numbers of synthetic training data. As illustrated in Table 5, the 64K **41.49 29.53** performance keeps increasing when the number of data is increased 80K96K 41.6240.12 29.7328.31 from 16K to 64K. Beyond this, escalating the data count from 64K to 80K yields only marginal enhancement. Further amplification of data eventually triggers a performance dip as excessive data leads to model overfitting to the synthetic training task. 

**Image order sensitivity.** The order of interleaved images in demonstrative instructions is pivotal for the compositional semantics of the instruction. Intuitively, altering the order of images within a demonstrative instruction can significantly shift its semantics. Consequently, variations in model performance can reveal the model’s sensitivity to the instruction semantics. An ideal model should keenly capture changes in instruction semantics. Therefore, we visualize the performance variations of models by randomly shuffling the order of interleaved images within the demonstrative instructions. Ac- 


![](raw/papers/2_LLM_NLP/LLM_paper_list/zero-shot/iclr_2024_Fine-tuning_Multimodal_LLMs_to_Follow_Zero-shot_Demonstrative_Instructions_images/iclr_2024_Fine-tuning_Multimodal_LLMs_to_Follow_Zero-shot_Demonstrative_Instructions.pdf-0008-08.png)


**----- Start of picture text -----**<br>
20<br>15<br>10<br>5<br>0<br>Multi Visual Visual Multi Knowledge Text Multi<br>Modal Story Relation Modal Grounded Rich Image<br>Dialogue Telling Inference Cloze QA ImagesReasoning<br>QA<br>Figure 7: Analysis on image order sensitivity.<br>VPG-C<br>Mini-GPT4<br>Open<br>Flamingo<br>InstructBlip<br>**----- End of picture text -----**<br>


cording to Figure 7, we surprisingly find that SOTA models are less sensitive to the image order. In contrast, VPG-C can keenly capture the semantic changes caused by the shuffled image order. Particularly, our performance varies dramatically in multimodal dialogue, as the order of images within these tasks is closely intertwined with the dialogue content. 

**Qualitative examples.** As illustrated in Figure 8, VPG-C demonstrates strong abilities to perform reasoning over complicated demonstrative instructions. For instance, in **(a)** , VPG-C can keenly identify the connections between the images and thereby infer the reason that causes this unusual phenomenon. In **(b, c)** , VPG-C exhibits the ability to comprehend absurd objects through multimodal conversations with humans. In **(d, e)** , VPG-C can reasonably infer the relations among 

8 

Published as a conference paper at ICLR 2024 


![](raw/papers/2_LLM_NLP/LLM_paper_list/zero-shot/iclr_2024_Fine-tuning_Multimodal_LLMs_to_Follow_Zero-shot_Demonstrative_Instructions_images/iclr_2024_Fine-tuning_Multimodal_LLMs_to_Follow_Zero-shot_Demonstrative_Instructions.pdf-0009-01.png)


**----- Start of picture text -----**<br>
What is it and how to use it？ What is this in picture?<br>Please explain the interesting point based on the three pictures. This is an image of a camera lens. It is a type of camera lens that is designed to capture images and videos. It is typically used in combination with a camera body to form a camera system. The lens is designed to focus light onto a  This is an image of a fish. The fish is black and silvery, with a small mouth and eyes. It is lying on its side, with its tail and fins visible. The fish is about 5.51 inches long and 12.20 inches wide.<br>film or digital sensor, which captures the image.<br>The third image reveals some peculiar tanned circles<br>on a men‘s feet, perfectly aligned with the holes in  Is it still the fish you thought it was now?<br>the sandals shown in the second image. It might be<br>associated with his long-time outdoor activities since the first image indicates that the sun is intense.  Are you sure this is an image of a camera lens? No, I don't think it's a fish now, there are many colored pens and a fish-like case in this picture,<br>Therefore, the amusing sunburn occurs when the men’s feet are exposed to prolonged sunlight. Due to the sandals partially blocking the UV rays, the skin away from the holes remains untanned while the skin under the holes gets tanned. However, the sunlight  I’m sorry. The image shows a large silver cup with a handle, featuring a lens-like design. The cup is made of stainless steel and there is also a spoon resting inside the cup.Can this lens take photos? these pens should be packed in a pencil case, so this should be a black and white fish-looking pencil case.What else can this pencil case be used for besides holding pens?<br>still reaches the skin between the holes, resulting in the tanned circles. (a) No, it's just designed to look like a camera lens, but it's not a real camera lens. (b) It can also be used to hold stickers, erasers, or other stationery, but its fish-shaped appearance is too deceptive! (c)<br>What changes have occurred to this wall in the pictures over time, and what does this set of  between these two pictures?What is the connection<br>pictures hope to show us through these changes? The first image is a gray T-shirt with Muggle text<br>This set of pictures mainly shows the change of a  printed on it, while the second image is Google's logo.<br>wall. In the beginning, the wall is a metal wall with only one line of writing, which has a silver shine and is very clean. In the third picture, some rust marks appear on the wall. A man stops by and stares at the wall. In the third image, there are  The set of images is a creative advertisement for What do these four pictures want to convey to us? The connection between these two images is that their font and color scheme are very similar.What makes this set of pictures look interesting?<br>some cars and a rusted wall with the outline of a knife. The last picture is already a rusted wall and the outline of an unrusted knife. It demonstrates the stable quality of the knife by showing that it will not rust after a long time. (d) animal adoption, which encourages people to take a note with contact information that also serves as a cage bar for the animals behind it. As people take the notes, animals are more likely to be adopted and freed from their cages. (e) Because the word 'muggle' uses the same font and color scheme as Google's logo, and their pronunciation is very similar. Its humor lies in using a homophonic stem and making the person wearing this dress look like a muggle. (f)<br>**----- End of picture text -----**<br>


Figure 8: Qualitative examples generated by our VPG-C-Vicuna-7B model. 

the images and understand the metaphorical implications they want to convey. In Appendix G, we provide more practical examples as well as comparisons with other MLLMs, where we find that baseline models fail to correctly associate multiple images and comprehend demonstrative context. 

## 5 RELATED WORK 

MLLMs (Yin et al., 2023) aim to serve as a general-purpose assistant to perform various visionlanguage tasks by free-text generation. Flamingo (Alayrac et al., 2022) and BLIP-2 (Li et al., 2023c) bridge LLMs with powerful pre-trained visual encoders and demonstrate strong zero-shot ability by aligning visual features with LLMs. Follow-up works of LLaVA (Liu et al., 2023), MiniGPT4 (Zhu et al., 2023a), InstructBLIP (Dai et al., 2023), Hallucidoctor (Yu et al., 2024), mPLUGOwl (Ye et al., 2023) propose to fine-tune MLLMs with multimodal instruction tuning data. To effectively benchmark the recent progress in MLLMs, concurrent works of LVLM-eHub (Xu et al., 2023) and MME Benchmark (Fu et al., 2023) are proposed, while they mainly focus on instructions that only involve a single image with limited instruction diversity. In this paper, we propose the first demonstrative instruction-following benchmark, covering various tasks of diverse scenarios. Further, we propose a lightweight and generic VPG-C module to address the inherent limitation of current VPGs. Our VPG-C is efficiently tuned by our synthetic discriminative training strategy, which demonstrates powerful potentials of text-to-image diffusion models (He et al., 2022; Lin et al., 2023; Prabhu et al., 2023; Bansal & Grover, 2023; Yu et al., 2023b) to facilitate vision-language understanding (Radford et al., 2021b; Jia et al., 2021; Li et al., 2022b). 

## 6 CONCLUSION 

In this paper, we propose VPG-C, a generic and parameter-efficient approach that infers and completes the missing visual details for MLLMs to comprehend demonstrative instructions with interleaved multimodal context. Meanwhile, we present a synthetic discriminative training strategy to fine-tune VPG-C, eliminating the need for supervised demonstrative instruction data. To foster the research on demonstrative instruction understanding, we build DEMON, a comprehensive benchmark for multimodal large language models, consisting of 31 tasks with complicated vision-language demonstrative context, covering a wide range of scenarios. Through synthetic training, VPG-C showcases notable zero-shot performance on the DEMON benchmark. Its superior performance on other established benchmarks like MME and OwlEval further underscores its effectiveness. 

**Acknowledgment.** This work was supported by the NSFC (No. 62272411), Key Research and Development Projects in Zhejiang Province (No. 2024C01106), the National Key Research and Development Project of China (2018AAA0101900), the Tencent WeChat Rhino-Bird Special Research Program (Tencent WXG-FR-2023-10), and Research funding from FinVolution Group. 

9 

Published as a conference paper at ICLR 2024 

## REFERENCES 

- Jean-Baptiste Alayrac, Jeff Donahue, Pauline Luc, Antoine Miech, Iain Barr, Yana Hasson, Karel Lenc, Arthur Mensch, Katherine Millican, Malcolm Reynolds, et al. Flamingo: a visual language model for few-shot learning. _Advances in Neural Information Processing Systems_ , 35:23716– 23736, 2022. 

- Stanislaw Antol, Aishwarya Agrawal, Jiasen Lu, Margaret Mitchell, Dhruv Batra, C Lawrence Zitnick, and Devi Parikh. Vqa: Visual question answering. In _Proceedings of the IEEE international conference on computer vision_ , pp. 2425–2433, 2015. 

- Omri Avrahami, Dani Lischinski, and Ohad Fried. Blended diffusion for text-driven editing of natural images. In _Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition_ , pp. 18208–18218, 2022. 

- Anas Awadalla, Irena Gao, Joshua Gardner, Jack Hessel, Yusuf Hanafy, Wanrong Zhu, Kalyani Marathe, Yonatan Bitton, Samir Gadre, Jenia Jitsev, Simon Kornblith, Pang Wei Koh, Gabriel Ilharco, Mitchell Wortsman, and Ludwig Schmidt. Openflamingo, March 2023. URL https: //doi.org/10.5281/zenodo.7733589. 

- Haoping Bai, Shancong Mou, Tatiana Likhomanenko, Ramazan Gokberk Cinbis, Oncel Tuzel, Ping Huang, Jiulong Shan, Jianjun Shi, and Meng Cao. Vision datasets: A benchmark for vision-based industrial inspection. _arXiv preprint arXiv:2306.07890_ , 2023. 

- Ankan Bansal, Yuting Zhang, and Rama Chellappa. Visual question answering on image sets. In _Computer Vision–ECCV 2020: 16th European Conference, Glasgow, UK, August 23–28, 2020, Proceedings, Part XXI 16_ , pp. 51–67. Springer, 2020. 

- Hritik Bansal and Aditya Grover. Leaving reality to imagination: Robust classification via generated datasets. _arXiv preprint arXiv:2302.02503_ , 2023. 

- Nilavra Bhattacharya, Qing Li, and Danna Gurari. Why does a visual question have different answers? In _Proceedings of the IEEE/CVF International Conference on Computer Vision_ , pp. 4271–4280, 2019. 

- Holger Caesar, Varun Bankiti, Alex H Lang, Sourabh Vora, Venice Erin Liong, Qiang Xu, Anush Krishnan, Yu Pan, Giancarlo Baldan, and Oscar Beijbom. nuscenes: A multimodal dataset for autonomous driving. In _Proceedings of the IEEE/CVF conference on computer vision and pattern recognition_ , pp. 11621–11631, 2020. 

- Yingshan Chang, Mridu Narang, Hisami Suzuki, Guihong Cao, Jianfeng Gao, and Yonatan Bisk. Webqa: Multihop and multimodal qa. In _Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition_ , pp. 16495–16504, 2022. 

- Wei-Lin Chiang, Zhuohan Li, Zi Lin, Ying Sheng, Zhanghao Wu, Hao Zhang, Lianmin Zheng, Siyuan Zhuang, Yonghao Zhuang, Joseph E Gonzalez, et al. Vicuna: An open-source chatbot impressing gpt-4 with 90% chatgpt quality, 2023. URL https://vicuna.lmsys.org. 

- Hyung Won Chung, Le Hou, Shayne Longpre, Barret Zoph, Yi Tay, William Fedus, Eric Li, Xuezhi Wang, Mostafa Dehghani, Siddhartha Brahma, et al. Scaling instruction-finetuned language models. _arXiv preprint arXiv:2210.11416_ , 2022. 

- Wenliang Dai, Junnan Li, Dongxu Li, Anthony Meng Huat Tiong, Junqi Zhao, Weisheng Wang, Boyang Li, Pascale Fung, and Steven Hoi. Instructblip: Towards general-purpose vision-language models with instruction tuning. _arXiv preprint arXiv:2305.06500_ , 2023. 

- Yuxin Fang, Wen Wang, Binhui Xie, Quan Sun, Ledell Wu, Xinggang Wang, Tiejun Huang, Xinlong Wang, and Yue Cao. Eva: Exploring the limits of masked visual representation learning at scale. In _Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition_ , pp. 19358–19369, 2023. 

- Maxwell Forbes, Christine Kaeser-Chen, Piyush Sharma, and Serge Belongie. Neural naturalist: generating fine-grained image comparisons. _arXiv preprint arXiv:1909.04101_ , 2019. 

10 

Published as a conference paper at ICLR 2024 

Chaoyou Fu, Peixian Chen, Yunhang Shen, Yulei Qin, Mengdan Zhang, Xu Lin, Zhenyu Qiu, Wei Lin, Jinrui Yang, Xiawu Zheng, et al. Mme: A comprehensive evaluation benchmark for multimodal large language models. _arXiv preprint arXiv:2306.13394_ , 2023. 

Peng Gao, Jiaming Han, Renrui Zhang, Ziyi Lin, Shijie Geng, Aojun Zhou, Wei Zhang, Pan Lu, Conghui He, Xiangyu Yue, et al. Llama-adapter v2: Parameter-efficient visual instruction model. _arXiv preprint arXiv:2304.15010_ , 2023. 

- Tanmay Gupta, Dustin Schwenk, Ali Farhadi, Derek Hoiem, and Aniruddha Kembhavi. Imagine this! scripts to compositions to videos. In _Proceedings of the European conference on computer vision (ECCV)_ , pp. 598–613, 2018. 

- Xintong Han, Zuxuan Wu, Phoenix X Huang, Xiao Zhang, Menglong Zhu, Yuan Li, Yang Zhao, and Larry S Davis. Automatic spatially-aware fashion concept discovery. In _Proceedings of the IEEE international conference on computer vision_ , pp. 1463–1471, 2017. 

- Darryl Hannan, Akshay Jain, and Mohit Bansal. Manymodalqa: Modality disambiguation and qa over diverse inputs. In _Proceedings of the AAAI Conference on Artificial Intelligence_ , volume 34, pp. 7879–7886, 2020. 

- Kaiming He, Georgia Gkioxari, Piotr Doll´ar, and Ross Girshick. Mask r-cnn. In _Proceedings of the IEEE international conference on computer vision_ , pp. 2961–2969, 2017. 

- Ruifei He, Shuyang Sun, Xin Yu, Chuhui Xue, Wenqing Zhang, Philip Torr, Song Bai, and Xiaojuan Qi. Is synthetic data from generative models ready for image recognition? _arXiv preprint arXiv:2210.07574_ , 2022. 

- Mehrdad Hosseinzadeh and Yang Wang. Image change captioning by learning from an auxiliary task. In _Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition_ , pp. 2725–2734, 2021. 

- Edward J Hu, Yelong Shen, Phillip Wallis, Zeyuan Allen-Zhu, Yuanzhi Li, Shean Wang, Lu Wang, and Weizhu Chen. Lora: Low-rank adaptation of large language models. _arXiv preprint arXiv:2106.09685_ , 2021. 

- Ting-Hao Huang, Francis Ferraro, Nasrin Mostafazadeh, Ishan Misra, Aishwarya Agrawal, Jacob Devlin, Ross Girshick, Xiaodong He, Pushmeet Kohli, Dhruv Batra, et al. Visual storytelling. In _Proceedings of the 2016 conference of the North American chapter of the association for computational linguistics: Human language technologies_ , pp. 1233–1239, 2016. 

- Phillip Isola, Joseph J Lim, and Edward H Adelson. Discovering states and transformations in image collections. In _Proceedings of the IEEE conference on computer vision and pattern recognition_ , pp. 1383–1391, 2015. 

- Mohit Iyyer, Varun Manjunatha, Anupam Guha, Yogarshi Vyas, Jordan Boyd-Graber, Hal Daume, and Larry S Davis. The amazing mysteries of the gutter: Drawing inferences between panels in comic book narratives. In _Proceedings of the IEEE Conference on Computer Vision and Pattern recognition_ , pp. 7186–7195, 2017. 

- Harsh Jhamtani and Taylor Berg-Kirkpatrick. Learning to describe differences between pairs of similar images. _arXiv preprint arXiv:1808.10584_ , 2018. 

- Chao Jia, Yinfei Yang, Ye Xia, Yi-Ting Chen, Zarana Parekh, Hieu Pham, Quoc Le, Yun-Hsuan Sung, Zhen Li, and Tom Duerig. Scaling up visual and vision-language representation learning with noisy text supervision. In _International conference on machine learning_ , pp. 4904–4916. PMLR, 2021. 

- Aniruddha Kembhavi, Minjoon Seo, Dustin Schwenk, Jonghyun Choi, Ali Farhadi, and Hannaneh Hajishirzi. Are you smarter than a sixth grader? textbook question answering for multimodal machine comprehension. In _Proceedings of the IEEE Conference on Computer Vision and Pattern recognition_ , pp. 4999–5007, 2017. 

11 

Published as a conference paper at ICLR 2024 

- Alexander Kirillov, Eric Mintun, Nikhila Ravi, Hanzi Mao, Chloe Rolland, Laura Gustafson, Tete Xiao, Spencer Whitehead, Alexander C Berg, Wan-Yen Lo, et al. Segment anything. _arXiv preprint arXiv:2304.02643_ , 2023. 

- Bo Li, Yuanhan Zhang, Liangyu Chen, Jinghao Wang, Fanyi Pu, Jingkang Yang, Chunyuan Li, and Ziwei Liu. Mimic-it: Multi-modal in-context instruction tuning. _arXiv preprint arXiv:2306.05425_ , 2023a. 

Dongxu Li, Junnan Li, Hung Le, Guangsen Wang, Silvio Savarese, and Steven CH Hoi. Lavis: A library for language-vision intelligence. _arXiv preprint arXiv:2209.09019_ , 2022a. 

- Juncheng Li, Xin Wang, Siliang Tang, Haizhou Shi, Fei Wu, Yueting Zhuang, and William Yang Wang. Unsupervised reinforcement learning of transferable meta-skills for embodied navigation. In _Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition_ , pp. 12123–12132, 2020. 

- Juncheng Li, Xin He, Longhui Wei, Long Qian, Linchao Zhu, Lingxi Xie, Yueting Zhuang, Qi Tian, and Siliang Tang. Fine-grained semantically aligned vision-language pre-training. _Advances in neural information processing systems_ , 35:7290–7303, 2022b. 

- Juncheng Li, Junlin Xie, Long Qian, Linchao Zhu, Siliang Tang, Fei Wu, Yi Yang, Yueting Zhuang, and Xin Eric Wang. Compositional temporal grounding with structured variational cross-graph correspondence learning. In _Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition_ , pp. 3032–3041, 2022c. 

- Juncheng Li, Minghe Gao, Longhui Wei, Siliang Tang, Wenqiao Zhang, Mengze Li, Wei Ji, Qi Tian, Tat-Seng Chua, and Yueting Zhuang. Gradient-regulated meta-prompt learning for generalizable vision-language models. 2023b. 

- Junnan Li, Dongxu Li, Silvio Savarese, and Steven Hoi. Blip-2: Bootstrapping languageimage pre-training with frozen image encoders and large language models. _arXiv preprint arXiv:2301.12597_ , 2023c. 

- Mengze Li, Tianbao Wang, Haoyu Zhang, Shengyu Zhang, Zhou Zhao, Jiaxu Miao, Wenqiao Zhang, Wenming Tan, Jin Wang, Peng Wang, et al. End-to-end modeling via information tree for oneshot natural language spatial video grounding. In _Proceedings of the 60th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)_ , pp. 8707–8717, 2022d. 

- Mengze Li, Haoyu Zhang, Juncheng Li, Zhou Zhao, Wenqiao Zhang, Shengyu Zhang, Shiliang Pu, Yueting Zhuang, and Fei Wu. Unsupervised domain adaptation for video object grounding with cascaded debiasing learning. In _Proceedings of the 31st ACM International Conference on Multimedia_ , pp. 3807–3816, 2023d. 

- Yitong Li, Zhe Gan, Yelong Shen, Jingjing Liu, Yu Cheng, Yuexin Wu, Lawrence Carin, David Carlson, and Jianfeng Gao. Storygan: A sequential conditional gan for story visualization. In _Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition_ , pp. 6329–6338, 2019. 

- Yongqi Li, Wenjie Li, and Liqiang Nie. Mmcoqa: Conversational question answering over text, tables, and images. In _Proceedings of the 60th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)_ , pp. 4220–4231, 2022e. 

- Shaobo Lin, Kun Wang, Xingyu Zeng, and Rui Zhao. Explore the power of synthetic data on few-shot object detection. In _Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition_ , pp. 638–647, 2023. 

- Haotian Liu, Chunyuan Li, Qingyang Wu, and Yong Jae Lee. Visual instruction tuning. _arXiv preprint arXiv:2304.08485_ , 2023. 

- Adyasha Maharana, Darryl Hannan, and Mohit Bansal. Storydall-e: Adapting pretrained text-toimage transformers for story continuation. In _Computer Vision–ECCV 2022: 17th European Conference, Tel Aviv, Israel, October 23–27, 2022, Proceedings, Part XXXVII_ , pp. 70–87. Springer, 2022. 

12 

Published as a conference paper at ICLR 2024 

Minesh Mathew, Dimosthenis Karatzas, and CV Jawahar. Docvqa: A dataset for vqa on document images. In _Proceedings of the IEEE/CVF winter conference on applications of computer vision_ , pp. 2200–2209, 2021. 

- Anand Mishra, Shashank Shekhar, Ajeet Kumar Singh, and Anirban Chakraborty. Ocr-vqa: Visual question answering by reading text in images. In _2019 international conference on document analysis and recognition (ICDAR)_ , pp. 947–952. IEEE, 2019. 

OpenAI. Chatgpt: A language model for conversational ai. Technical report, OpenAI, 2023a. URL https://www.openai.com/research/chatgpt. 

OpenAI. Gpt-4 technical report. _arXiv:2303.08774_ , 2023b. 

- Viraj Prabhu, Sriram Yenamandra, Prithvijit Chattopadhyay, and Judy Hoffman. Lance: Stresstesting visual models by generating language-guided counterfactual images. _arXiv preprint arXiv:2305.19164_ , 2023. 

- Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, et al. Learning transferable visual models from natural language supervision. In _International conference on machine learning_ , pp. 8748–8763. PMLR, 2021a. 

- Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, et al. Learning transferable visual models from natural language supervision. In _International conference on machine learning_ , pp. 8748–8763. PMLR, 2021b. 

- Hareesh Ravi, Kushal Kafle, Scott Cohen, Jonathan Brandt, and Mubbasir Kapadia. Aesop: Abstract encoding of stories, objects, and pictures. In _Proceedings of the IEEE/CVF International Conference on Computer Vision_ , pp. 2052–2063, 2021. 

- Nils Reimers and Iryna Gurevych. Sentence-BERT: Sentence embeddings using Siamese BERTnetworks. In _Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing and the 9th International Joint Conference on Natural Language Processing (EMNLPIJCNLP)_ , pp. 3982–3992, 2019. 

- Adriana Romero, Nicolas Ballas, Samira Ebrahimi Kahou, Antoine Chassang, Carlo Gatta, and Yoshua Bengio. Fitnets: Hints for thin deep nets. _arXiv preprint arXiv:1412.6550_ , 2014. 

- Piyush Sharma, Nan Ding, Sebastian Goodman, and Radu Soricut. Conceptual captions: A cleaned, hypernymed, image alt-text dataset for automatic image captioning. In _Proceedings of the 56th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)_ , pp. 2556–2565, 2018. 

- Mohit Shridhar, Jesse Thomason, Daniel Gordon, Yonatan Bisk, Winson Han, Roozbeh Mottaghi, Luke Zettlemoyer, and Dieter Fox. Alfred: A benchmark for interpreting grounded instructions for everyday tasks. In _Proceedings of the IEEE/CVF conference on computer vision and pattern recognition_ , pp. 10740–10749, 2020. 

- Alane Suhr, Stephanie Zhou, Ally Zhang, Iris Zhang, Huajun Bai, and Yoav Artzi. A corpus for reasoning about natural language grounded in photographs. _arXiv preprint arXiv:1811.00491_ , 2018. 

- Alon Talmor, Ori Yoran, Amnon Catav, Dan Lahav, Yizhong Wang, Akari Asai, Gabriel Ilharco, Hannaneh Hajishirzi, and Jonathan Berant. Multimodalqa: Complex question answering over text, tables and images. _arXiv preprint arXiv:2104.06039_ , 2021. 

- Hao Tan, Franck Dernoncourt, Zhe Lin, Trung Bui, and Mohit Bansal. Expressing visual relationships via language. _arXiv preprint arXiv:1906.07689_ , 2019. 

- Ryota Tanaka, Kyosuke Nishida, Kosuke Nishida, Taku Hasegawa, Itsumi Saito, and Kuniko Saito. Slidevqa: A dataset for document visual question answering on multiple images. _arXiv preprint arXiv:2301.04883_ , 2023. 

13 

Published as a conference paper at ICLR 2024 

- Hugo Touvron, Thibaut Lavril, Gautier Izacard, Xavier Martinet, Marie-Anne Lachaux, Timoth´ee Lacroix, Baptiste Rozi`ere, Naman Goyal, Eric Hambro, Faisal Azhar, et al. Llama: Open and efficient foundation language models. _arXiv:2302.13971_ , 2023a. 

- Hugo Touvron, Louis Martin, Kevin Stone, Peter Albert, Amjad Almahairi, Yasmine Babaei, Nikolay Bashlykov, Soumya Batra, Prajjwal Bhargava, Shruti Bhosale, Dan Bikel, Lukas Blecher, Cristian Canton Ferrer, Moya Chen, Guillem Cucurull, David Esiobu, Jude Fernandes, Jeremy Fu, Wenyin Fu, Brian Fuller, Cynthia Gao, Vedanuj Goswami, Naman Goyal, Anthony Hartshorn, Saghar Hosseini, Rui Hou, Hakan Inan, Marcin Kardas, Viktor Kerkez, Madian Khabsa, Isabel Kloumann, Artem Korenev, Punit Singh Koura, Marie-Anne Lachaux, Thibaut Lavril, Jenya Lee, Diana Liskovich, Yinghai Lu, Yuning Mao, Xavier Martinet, Todor Mihaylov, Pushkar Mishra, Igor Molybog, Yixin Nie, Andrew Poulton, Jeremy Reizenstein, Rashi Rungta, Kalyan Saladi, Alan Schelten, Ruan Silva, Eric Michael Smith, Ranjan Subramanian, Xiaoqing Ellen Tan, Binh Tang, Ross Taylor, Adina Williams, Jian Xiang Kuan, Puxin Xu, Zheng Yan, Iliyan Zarov, Yuchen Zhang, Angela Fan, Melanie Kambadur, Sharan Narang, Aurelien Rodriguez, Robert Stojnic, Sergey Edunov, and Thomas Scialom. Llama 2: Open foundation and fine-tuned chat models. _arXiv:2307.09288_ , 2023b. 

- Fei Xia, Amir R Zamir, Zhiyang He, Alexander Sax, Jitendra Malik, and Silvio Savarese. Gibson env: Real-world perception for embodied agents. In _Proceedings of the IEEE conference on computer vision and pattern recognition_ , pp. 9068–9079, 2018. 

- Ji Xin, Raphael Tang, Jaejun Lee, Yaoliang Yu, and Jimmy Lin. Deebert: Dynamic early exiting for accelerating bert inference. _arXiv preprint arXiv:2004.12993_ , 2020. 

- Kelvin Xu, Jimmy Ba, Ryan Kiros, Kyunghyun Cho, Aaron Courville, Ruslan Salakhudinov, Rich Zemel, and Yoshua Bengio. Show, attend and tell: Neural image caption generation with visual attention. In _International conference on machine learning_ , pp. 2048–2057. PMLR, 2015. 

- Peng Xu, Wenqi Shao, Kaipeng Zhang, Peng Gao, Shuo Liu, Meng Lei, Fanqing Meng, Siyuan Huang, Yu Qiao, and Ping Luo. Lvlm-ehub: A comprehensive evaluation benchmark for large vision-language models. _arXiv preprint arXiv:2306.09265_ , 2023. 

- Semih Yagcioglu, Aykut Erdem, Erkut Erdem, and Nazli Ikizler-Cinbis. Recipeqa: A challenge dataset for multimodal comprehension of cooking recipes. _arXiv preprint arXiv:1809.00812_ , 2018. 

- Qinghao Ye, Haiyang Xu, Guohai Xu, Jiabo Ye, Ming Yan, Yiyang Zhou, Junyang Wang, Anwen Hu, Pengcheng Shi, Yaya Shi, et al. mplug-owl: Modularization empowers large language models with multimodality. _arXiv preprint arXiv:2304.14178_ , 2023. 

- Shukang Yin, Chaoyou Fu, Sirui Zhao, Ke Li, Xing Sun, Tong Xu, and Enhong Chen. A survey on multimodal large language models. _arXiv preprint arXiv:2306.13549_ , 2023. 

- Qifan Yu, Juncheng Li, Yu Wu, Siliang Tang, Wei Ji, and Yueting Zhuang. Visually-prompted language model for fine-grained scene graph generation in an open world. In _Proceedings of the IEEE/CVF International Conference on Computer Vision_ , pp. 21560–21571, 2023a. 

- Qifan Yu, Juncheng Li, Wentao Ye, Siliang Tang, and Yueting Zhuang. Interactive data synthesis for systematic vision adaptation via llms-aigcs collaboration. _arXiv preprint arXiv:2305.12799_ , 2023b. 

- Qifan Yu, Juncheng Li, Longhui Wei, Liang Pang, Wentao Ye, Bosheng Qin, Siliang Tang, Qi Tian, and Yueting Zhuang. Hallucidoctor: Mitigating hallucinatory toxicity in visual instruction data. In _Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition_ , 2024. 

- Lvmin Zhang and Maneesh Agrawala. Adding conditional control to text-to-image diffusion models. _arXiv preprint arXiv:2302.05543_ , 2023. 

- Wenqiao Zhang, Siliang Tang, Yanpeng Cao, Shiliang Pu, Fei Wu, and Yueting Zhuang. Frame augmented alternating attention network for video question answering. _IEEE Transactions on Multimedia_ , 22(4):1032–1041, 2019. 

14 

Published as a conference paper at ICLR 2024 

Wenqiao Zhang, Haochen Shi, Jiannan Guo, Shengyu Zhang, Qingpeng Cai, Juncheng Li, Sihui Luo, and Yueting Zhuang. Magic: Multimodal relational graph adversarial inference for diverse and unpaired text-based image captioning. In _Proceedings of the AAAI Conference on Artificial Intelligence_ , volume 36, pp. 3335–3343, 2022. 

Deyao Zhu, Jun Chen, Xiaoqian Shen, Xiang Li, and Mohamed Elhoseiny. Minigpt-4: Enhancing vision-language understanding with advanced large language models. _arXiv preprint arXiv:2304.10592_ , 2023a. 

Wanrong Zhu, Jack Hessel, Anas Awadalla, Samir Yitzhak Gadre, Jesse Dodge, Alex Fang, Youngjae Yu, Ludwig Schmidt, William Yang Wang, and Yejin Choi. Multimodal c4: An open, billionscale corpus of images interleaved with text. _arXiv preprint arXiv:2304.06939_ , 2023b. 

15 

Published as a conference paper at ICLR 2024 

## A OVERVIEW 

## In this appendix, we present: 

- Detailed information of the proposed DEMON benchmark (Section B). 

- Implementation details of our VPG-C (Section C). 

- Implementation details of existing MLLMs on the DEMON benchmark (Section D). 

- Detailed zero-shot performance on MME benchmark (Section E). 

- Detailed zero-shot performance on DEMON benchmark (Section F). 

- Qualitative comparison with existing MLLMs (Section G). 

## B BENCHMARK DETAILS 

|Task|Scenario|Dataset|Metirc|
|---|---|---|---|
|**Multimodal Dialogue**||||
|Conversational Embodied Dialogue|Embodied|ALFRED (Shridhar et al., 2020)|ROUGE-L|
|Multimodal Dialogue|Conversation|MMCoQA(Li et al.,2022e)|ROUGE-L|
|**Visual Storytelling**||||
|Animated Story Completion|Cartoon|AESOP (Ravi et al., 2021)|ROUGE-L|
|Animated Story Completion|Cartoon|PororoSV (Li et al., 2019)|ROUGE-L|
|Animated Story Completion|Cartoon|FlintstonesSV (Gupta et al., 2018)|ROUGE-L|
|Sequential Photo Storytelling|Album|VIST (Huang et al., 2016)|ROUGE-L|
|Sequential Photo Storytelling|Cartoon|DiDeMoSV(Maharana et al.,2022)|ROUGE-L|
|**Visual Relation Inference**||||
|Visual Change Captioning|Surveillance|Spot-the-Diff (Jhamtani & Berg-Kirkpatrick, 2018)|ROUGE-L|
|Visual Change Captioning|Synthetic|CLEVR-Change (Hosseinzadeh & Wang, 2021)|ROUGE-L|
|Visual Relationship Expressing|General|IEdit (Tan et al., 2019)|ROUGE-L|
|Subtle Difference Expressing|Fine-Grained|Birds-to-Words(Forbes et al.,2019)|ROUGE-L|
|**Multimodal Cloze**||||
|Comic Dialogue Identifcation|Cartoon|COMICS-Dialogue (Iyyer et al., 2017)|Accuracy|
|Comic Panel Identifcation|Cartoon|COMICS-Panel (Iyyer et al., 2017)|Accuracy|
|Recipe Completion|Recipe|RecipeQA-TextCloze (Yagcioglu et al., 2018)|Accuracy|
|Visual StepCloze|Recipe|RecipeQA-VisualCloze(Yagcioglu et al.,2018)|Accuracy|
|**Knowledge Grounded QA**||||
|Webpage QA|Webpage|WebQA (Chang et al., 2022)|Accuracy|
|Textbook QA|Textbook|TQA (Kembhavi et al., 2017)|Accuracy|
|Complex Multimodal QA|Wikipedia|MMQA (Talmor et al., 2021)|Accuracy|
|Complex MultimodalQA*|Wikipedia|MANYMODALQA(Hannan et al.,2020)|Accuracy|
|**Text-Rich Images QA**||||
|Slide QA|Slide|SlideVQA (Tanaka et al., 2023)|Accuracy|
|OCR QA|Book Cover|OCR-VQA (Mishra et al., 2019)|Accuracy|
|DocumentQA|Document Image|DocVQA(Mathew et al.,2021)|Accuracy|
|**Multi-Image Reasoning**||||
|Image-Set QA*|Indoor Egocentric|Gibson (Bansal et al., 2020; Xia et al., 2018)|Accuracy|
|Image-Set QA|Driving Recording|nuScenes (Bansal et al., 2020; Caesar et al., 2020)|Accuracy|
|Industrial Inspection|Industrial|VISION (Bai et al., 2023)|Accuracy|
|Fashion QA|Fashion|Fashion200K (Han et al., 2017)|Accuracy|
|Property Coherence|General|MIT-States-PropertyCoherence (Isola et al., 2015)|Accuracy|
|State Transformation Coherence|General|MIT-States-StateCoherence (Isola et al., 2015)|Accuracy|
|Visual Step Matching|Recipe|RecipeQA-ImageCoherence (Yagcioglu et al., 2018)|Accuracy|
|Multi-Image Visual Entailment|General|NLVR2 (Suhr et al., 2018)|Accuracy|
|AmbiguityAnalysis|Mobile Photo|VizWiz(Bhattacharya et al.,2019)|Accuracy|



Table 6: Summary of the demonstrative instruction-following tasks in DEMON benchmark. * indicates the tasks that are not included in DEMON-Core. 

## C IMPLEMENTATION DETAILS 

**Model.** We choose ViT-G/14 from EVA-CLIP (Fang et al., 2023) as our visual encoder and pretrained Q-former from BLIP-2 without instruction tuning as the task-agnostic visual prompt generator. For the large language model, we implement three versions: LLaMA2-7B (Touvron et al., 

16 

Published as a conference paper at ICLR 2024 


![](raw/papers/2_LLM_NLP/LLM_paper_list/zero-shot/iclr_2024_Fine-tuning_Multimodal_LLMs_to_Follow_Zero-shot_Demonstrative_Instructions_images/iclr_2024_Fine-tuning_Multimodal_LLMs_to_Follow_Zero-shot_Demonstrative_Instructions.pdf-0017-01.png)


**----- Start of picture text -----**<br>
Step1: Reasoning with Task-Agnostic Visual Prompts (VP) Step2: Visual Prompt Generator Complete (VPG-C) Step3: Reasoning with<br>Frozen Queries Learnable Queries Instruction-Specific VP<br>··· ··· + Linear    ℎ$(/#      ℎ$(/#<br>② Residual Details Extraction ① Guidance Generation  𝑣 𝑣'&'! ...... (/#(/# 𝑣$𝑣$ ...... '!'&(/#(/#<br>Image Q-Former * Q-Former ... ...<br>Encoder  ℎ%(/#  ℎ%(/#<br>... ... ... ...<br>③ reintegration  𝑣'& ... (/# 𝑣$!& ... (/#<br>𝑣̅!! ... 𝑣̅!& ... 𝑣̅'! ... 𝑣̅'& Linear +  𝑣'!(/# 𝑣$!!(/#<br>Interleaved Vision-Language Instruction Instruction-Specific Visual Prompts ... ...<br> ℎ!"  ℎ#" ...  𝑣!!" ...  𝑣!&" ...  ℎ%" ...  𝑣'!" ...  𝑣'&" ...  ℎ$"  ℎ#(/#  ℎ#(/#<br> ℎ text token  𝑣 visual prompt * the reused Q-Former takes the   ℎ!(/#  ℎ!(/#<br>same input from the image encoder<br>Language Response<br>LLM Layer<br>LLM Layer  1 LLM Layer  L/2 LLM Layer  L<br>L/2 + 1<br>**----- End of picture text -----**<br>


Figure 9: Detailed framework of our MLLM enhanced with VPG-C. 

2023b), Vicuna-7B (Chiang et al., 2023), Vicuna-13B, with 32, 32, 48 Transformer layers, respectively. We derive instruction-specific conditions from the 16th / 24th layer and re-inject the conditional visual knowledge into the 17th / 25th layer. Furthermore, we provide detailed framework of our MLLM enhanced with VPG-C in Figure 9. 

**Choice of Q-former.** Recently, InstructBLIP (Dai et al., 2023) proposes to take the instruction as additional input to the Q-former and fine-tune the Q-former to extract visual features according to instructions using 16M multimodal instruction tuning data. While achieving outstanding performance on in-domain tasks, a recent study (Xu et al., 2023) indicates that fine-tuning on massive in-domain data severely undermines its generalizability on open-world scenarios. Instead of directly relying on the Q-former to achieve task-specific feature extraction by massive instruction tuning, we aim to utilize the sophisticated reasoning ability of LLMs to guide the Q-former to conditionally attend to residual visual details. Thus, we use the Q-former without instruction data tuning from BLIP-2 (Li et al., 2023c), which extracts the task-agnostic primary visual contents at the first time. 

**Training.** We implement VPG-C in LAVIS library (Li et al., 2022a). We keep the visual backbone, visual prompt generator, and the language model frozen, and tune the VPG-C module using the proposed training strategy. Since BLIP-2 models do not include pre-trained Q-former that matches Vicuna and LLaMA2, we reuse the Q-former that matches FlanT5-XXL and fine-tune the last linear projection layer with 5 million image-text pairs to align it with Vicuna/LLaMA2. All the tunable parameters of our VPG-C module are a set of query embeddings and two linear projection layers, which only accounts for 0.09% ( _∼_ 6.3M) of the entire model. As for synthetic training, we select about 30k images from CC3M (Sharma et al., 2018) that contain significantly ignored objects and perform different types of editing on them. Totally, we generate approximately 64k synthetic images with suitable modifications. To stablize the training and avoid overfitting, we use 500k imagecaption pairs from CC3M to jointly train the VPG-C module. We tune the VPG-C module for 18k steps using a batch size of 24 for synthetic training and 64 for image captioning, which takes about 7 hours to complete with a single A100 GPU. Additionally, we adopt the AdamW optimizer with _β_ = (0 _._ 9 _,_ 0 _._ 999), and set the learning rate and weight decay to 0.00002 and 0.05, respectively. We warm up the training with 2k warm-up steps, followed by a learning rate decay mechanism with the cosine schedule. 

**Implementation of VPG-C with the linear VPG.** As a generic method, VPG-C can be implemented with different VPGs. Beyond widely used Q-former that is composed of multiple Transformer blocks, we further probe the effectiveness of VPG-C with a simpler VPG, i.e., Linear Projection, as used in LLaVA (Liu et al., 2023). LLaVA trains a simple linear layer as the VPG to connect image features into the word embedding space. To implement VPG-C with the linear VPG, we first linearly project the generated guidance **g** and then take it as a filter to perform element-wise Hadamard product with the visual features _X[I]_ from the image encoder: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/zero-shot/iclr_2024_Fine-tuning_Multimodal_LLMs_to_Follow_Zero-shot_Demonstrative_Instructions_images/iclr_2024_Fine-tuning_Multimodal_LLMs_to_Follow_Zero-shot_Demonstrative_Instructions.pdf-0017-07.png)


where _W_ 1 and _W_ 2 are linear projection matrixes, 1 _[T]_ is the transpose of an all-ones vector, and _⊙_ represents Hadamard product. The output _V_ represents the newly-extracted missing visual details according to the inferred guidance. And _V_ is reintegrated into the LLM in the same manner. 

17 

Published as a conference paper at ICLR 2024 

## D MODEL DETAILS IN DEMON BENCHMARK 

Recent advancements in LLMs (OpenAI, 2023a;b) have heralded significant achievements across various domains. Inspired by this success, many MLLMs (Li et al., 2023c; Liu et al., 2023; Zhu et al., 2023a; Alayrac et al., 2022; Ye et al., 2023; Gao et al., 2023; Li et al., 2023a) have been proposed to foster generalist vision-language reasoning (Xu et al., 2015; Li et al., 2023b; 2020; Yu et al., 2023a; Li et al., 2022d; Zhang et al., 2022; Li et al., 2022c; 2023d; Zhang et al., 2019; Antol et al., 2015). In our experiments, we conducted comparisons with some of the most recent and representative MLLMs in the following. 

- **LLaVA** (Liu et al., 2023) establishes a connection between the visual encoder ViT-L/14 from CLIP (Radford et al., 2021a) and the language decoder LLaMA (Touvron et al., 2023a), utilizing a lightweight, fully-connected (FC) layer. Initially, the system trains this FC layer using 595K image-text pairs, while keeping both the visual encoder and LLM static. Following this, LLaVA fine-tunes both the FC layer and LLM using a dataset comprising 158K instructional vision-language pairs. The tested version is “LLaVA-7B-v0”. 

- **LLaMA-Adapter V2** (Gao et al., 2023) stands as a model of parameter efficiency within the realm of visual instruction. Despite maintaining the visual encoder (ViT-L/14) and the LLM in a static state, LA-V2 distributes the instruction-following capacity of the entire LLaMA system via bias-tuning. This method allows for the refinement of scale, bias, norm, and prompt parameters on diverse data sets. These include 200M image captioning data, 158K visual instruction-following data, and an additional 52K language instructionfollowing data, the latter of which was assembled by GPT-4 (OpenAI, 2023b). The tested version is “LLaVA-7B”. 

- **MiniGPT-4** (Zhu et al., 2023a) bridges the gap between the visual encoder and text encoder using a fully-connected (FC) layer. Initially, this model trains the FC layer on a dataset comprised of 5M image-text pairs before fine-tuning it on 3.5K instructional visionlanguage data. Notwithstanding its simplicity, MiniGPT-4 requires the loading of a pretrained vision encoder from BLIP2, as well as a Vicuna LLM (Chiang et al., 2023). The tested version is “minigpt4-aligned-with-vicuna7b”. 

- **BLIP2** (Li et al., 2023c) employs a dual-stage strategy to seamlessly bridge the modality gap, utilizing a lean Q-Former pre-trained on 129 million image-text pairs. The initial stage kick-starts the learning process of vision-language representation, leveraging a frozen image encoder, the ViT-g/14 from EVA-CLIP (Fang et al., 2023). Subsequently, the second stage harnesses a frozen LLM, the FlanT5 (Chung et al., 2022), to initiate the visionto-language generative learning. This innovative strategy effectively facilitates zero-shot instructed image-to-text generation. The tested version is “blip2-pretrain-flant5xl”. 

- **mPLUG-Owl** (Ye et al., 2023) introduces a visual abstractor, fundamentally close the Perceiver Resampler in Flamingo (Alayrac et al., 2022), as a bridge between the pre-trained visual encoder ViT-L/14 and the LLM (LLaMA (Touvron et al., 2023a)). This model adopts a two-stage fine-tuning procedure. In the initial phase, both the visual encoder and the visual abstractor undergo comprehensive fine-tuning using a dataset of 204M image-text pairs. Subsequently, in the second phase, mPLUG-Owl applies the 158K LLaVA-Instruct dataset to fine-tune the pre-trained LLM in a parameter-efficient manner through the use of LoRA (Hu et al., 2021). The tested version is “mplug-owl-llama-7b”. 

- **Otter** (Li et al., 2023a) is a multimodal model that applies in-context instruction tuning based on OpenFlamingo (Alayrac et al., 2022). This model integrates a LLaMA-7B (Touvron et al., 2023a) language encoder and a CLIP ViT-L/14. While the visual and text encoders remain static, Otter refines an additional 1.3 billion parameters. These parameters are derived from adaptation modules and are trained using 158K instruction-following data. The tested version is “OTTER-Image-LLaMA7B-LA-InContext”. 

- **InstructBLIP** (Dai et al., 2023) originates from a pre-trained BLIP-2 model, which consists of a ViT-g/14 image encoder, a Vicuna, and a Q-Former to act as the bridge between these two components. During the process of vision-language instruction tuning, only the Q-Former undergoes fine-tuning, with the training process leveraging data from 13 distinct visual question-answering datasets. The tested version is “blip2-vicuna-instruct-7b”. 

18 

Published as a conference paper at ICLR 2024 

- **OpenFlamingo** (Alayrac et al., 2022; Awadalla et al., 2023) represents one of the pioneering efforts to incorporate Language Model Learning (LLMs) into the domain of visionlanguage pretraining. To optimize its conditioning on visual features, Flamingo strategically integrates a number of gated cross-attention dense blocks amidst the layers of the pre-trained language encoder. OpenFlamingo offers an open-source rendition of this advanced model. The tested version is “llama-7b”. 

The DEMON benchmark predominantly features interleaved vision-language instructions, distinguishing it from the traditional single-image datasets. While our innovative method, VPG-C, along with OpenFlamingo and MiniGPT-4, inherently accommodates interleaved image-text sequences, other models like BLIP-2, InstructBlip, LLaVA, mPLUG-Owl, Otter, and LLaMA-Adapter V2 do not. For these, we employed a strategy where we concatenate the embeddings of all images. This approach can be analogized to treating images as frames within a video. To maintain the positional context of each image in an interleaved image-text instruction, we explicitly indicate the location of each image within the context. 

## E DETAILED ZERO-SHOT PERFORMANCE ON MME BENCHMARK 

In this section, we report the detailed performance on the 14 subtasks of MME benchmark in Table 7. 

Table 7: Detailed zero-shot performance on MME benchmark. 

||Table 7: Detailed zero-shot performance on MME benchmark.||
|---|---|---|
||BLIP-2<br>InstructBLIP<br>LA-V2<br>LLaVA<br>MiniGPT-4<br>mPLUG-Owl<br>Otter|VPG-C|
|Existence<br>Count<br>Position<br>Color<br>Poster<br>Celebrity<br>Scene<br>Landmark<br>Artwork<br>OCR|160.00<br>185.00<br>120.00<br>50.00<br>115.00<br>120.00<br>195.00<br>135.00<br>143.33<br>50.00<br>50.00<br>123.33<br>88.33<br>50.00<br>73.33<br>66.67<br>48.33<br>50.00<br>81.67<br>50.00<br>86.67<br>148.33<br>153.33<br>75.00<br>55.00<br>110.00<br>55.00<br>113.33<br>141.84<br>123.81<br>99.66<br>50.00<br>55.78<br>136.05<br>138.78<br>105.59<br>101.18<br>86.18<br>48.82<br>65.29<br>100.29<br>172.65<br>145.25<br>153.00<br>148.50<br>50.00<br>95.75<br>135.50<br>158.75<br>138.00<br>79.75<br>150.25<br>50.00<br>69.00<br>159.25<br>137.25<br>136.50<br>134.25<br>69.75<br>49.00<br>55.75<br>96.25<br>129.00<br>110.00<br>72.50<br>125.00<br>50.00<br>95.00<br>65.00<br>72.50|180.00<br>96.67<br>80.00<br>116.67<br>147.28<br>164.12<br>156.00<br>145.00<br>113.50<br>100.00|
|Perception|1293.84<br>1212.82<br>972.67<br>502.82<br>866.57<br>967.34<br>1292.26|1299.24|
|Commonsense<br>Numerical<br>Text Translation<br>Code Reasoning|110.00<br>129.29<br>81.43<br>57.14<br>72.14<br>78.57<br>106.43<br>40.00<br>40.00<br>62.50<br>50.00<br>55.00<br>60.00<br>72.50<br>65.00<br>65.00<br>50.00<br>57.50<br>55.00<br>80.00<br>57.50<br>75.00<br>57.50<br>55.00<br>50.00<br>110.00<br>57.50<br>70.00|98.57<br>77.50<br>57.50<br>87.50|
|Cognition|290.00<br>291.79<br>248.93<br>214.64<br>292.14<br>276.07<br>306.43|321.07|



## F DETAILED ZERO-SHOT PERFORMANCE ON DEMON BENCHMARK 

Table 8: Zero-shot evaluation on multimodal dialogue. 

||Conversational Embodied Dialogue<br>Multimodal Dialogue|
|---|---|
|BLIP-2<br>InstructBLIP<br>LLaMA-Adapter V2<br>LLaVA<br>MiniGPT-4<br>mPLUG-Owl<br>OpenFlamingo<br>Otter|16.75<br>35.49<br>18.07<br>49.09<br>19.04<br>9.40<br>10.19<br>5.39<br>16.82<br>10.57<br>11.07<br>14.27<br>24.27<br>9.49<br>16.06<br>14.68|
|VPG-C-LLaMA2-7B<br>VPG-C-Vicuna-7B<br>VPG-C-Vicuna-13B|48.31<br>37.04<br>41.02<br>33.99<br>42.25<br>34.02|



19 

Published as a conference paper at ICLR 2024 

Table 9: Zero-shot evaluation on visual storytelling. 

||Table 9: Zero-shot evaluation on visual storytelling.|
|---|---|
||Animated Story<br>Animated Story<br>Animated Story<br>Sequential Photo<br>Sequential Photo<br>Completion-AESOP<br>Completion-PororoSV<br>Completion-FlintstonesSV<br>Storytelling-VIST<br>Storytelling-DiDeMoSV|
|BLIP-2<br>InstructBLIP<br>LLaMA-Adapter V2<br>LLaVA<br>MiniGPT-4<br>mPLUG-Owl<br>OpenFlamingo<br>Otter|21.64<br>26.24<br>29.61<br>13.16<br>24.2<br>18.80<br>28.20<br>33.32<br>16.92<br>24.80<br>18.01<br>20.15<br>24.22<br>10.89<br>14.57<br>13.56<br>11.44<br>12.77<br>8.00<br>7.71<br>12.23<br>16.00<br>26.48<br>14.82<br>15.81<br>18.28<br>20.49<br>32.12<br>10.82<br>14.94<br>23.32<br>32.35<br>37.79<br>15.14<br>12.50<br>13.94<br>17.52<br>22.21<br>9.96<br>14.23|
|VPG-C-LLaMA2-7B<br>VPG-C-Vicuna-7B<br>VPG-C-Vicuna-13B|19.98<br>28.67<br>38.14<br>16.95<br>20.05<br>19.93<br>28.36<br>39.19<br>17.34<br>21.27<br>20.53<br>29.81<br>41.32<br>19.04<br>22.26|



Table 10: Zero-shot evaluation on visual relation inference. 

||Visual Change Captioning<br>Visual Change Captioning<br>Visual Relationship<br>Subtle Difference<br>-Spot-the-Diff<br>-CLEVR-Change<br>Expressing<br>Expressing|
|---|---|
|BLIP-2<br>InstructBLIP<br>LLaMA-Adapter V2<br>LLaVA<br>MiniGPT-4<br>mPLUG-Owl<br>OpenFlamingo<br>Otter|17.48<br>3.21<br>12.37<br>9.62<br>19.71<br>4.61<br>10.70<br>10.92<br>16.72<br>15.52<br>7.88<br>13.92<br>8.50<br>8.76<br>6.72<br>9.11<br>7.50<br>7.49<br>7.84<br>8.97<br>6.06<br>1.46<br>6.22<br>7.86<br>13.01<br>11.90<br>12.57<br>17.90<br>12.69<br>11.63<br>8.85<br>12.38|
|VPG-C-LLaMA2-7B<br>VPG-C-Vicuna-7B<br>VPG-C-Vicuna-13B|21.02<br>42.05<br>14.10<br>24.81<br>20.01<br>41.60<br>16.35<br>25.64<br>21.56<br>40.67<br>20.27<br>26.08|



Table 11: Zero-shot evaluation on multimodal cloze. 

||Table 11: Zero-shot evaluation on multimodal cloze.|
|---|---|
||Comic Dialogue Identifcation<br>Comic Panel Identifcation1<br>Recipe Completion<br>Visual StepCloze1|
|BLIP-2<br>InstructBLIP<br>LLaMA-Adapter V2<br>LLaVA<br>MiniGPT-4<br>mPLUG-Owl<br>OpenFlamingo<br>Otter|39.70<br>0.00<br>30.46<br>1.60<br>40.60<br>0.00<br>27.40<br>16.80<br>24.40<br>0.40<br>38.20<br>9.00<br>30.60<br>0.00<br>32.80<br>0.00<br>33.00<br>1.00<br>31.60<br>0.80<br>36.60<br>0.00<br>27.60<br>0.80<br>38.40<br>1.20<br>29.00<br>18.00<br>29.00<br>0.00<br>35.00<br>0.00|
|VPG-C-LLaMA2-7B<br>VPG-C-Vicuna-7B<br>VPG-C-Vicuna-13B|36.80<br>1.80<br>51.80<br>1.40<br>39.20<br>3.60<br>30.40<br>15.40<br>42.20<br>8.20<br>39.80<br>18.40|



> 1 For tasks with images as options, only responses that begin with the correct answer will be evaluated as correct. 

Table 12: Zero-shot evaluation on knowledge grounded QA. 

||WebpageQA<br>TextbookQA<br>Complex MultimodalQA|
|---|---|
|BLIP-2<br>InstructBLIP<br>LLaMA-Adapter V2<br>LLaVA<br>MiniGPT-4<br>mPLUG-Owl<br>OpenFlamingo<br>Otter|47.60<br>29.73<br>40.36<br>45.20<br>30.20<br>66.80<br>44.60<br>46.00<br>43.80<br>39.40<br>39.60<br>29.60<br>27.40<br>28.60<br>34.80<br>34.20<br>30.00<br>35.60<br>37.80<br>32.40<br>25.80<br>45.00<br>39.00<br>41.00|
|VPG-C-LLaMA2-7B<br>VPG-C-Vicuna-7B<br>VPG-C-Vicuna-13B|49.40<br>42.40<br>61.20<br>50.00<br>33.40<br>62.40<br>50.60<br>43.40<br>64.80|



20 

Published as a conference paper at ICLR 2024 

Table 13: Zero-shot evaluation on text-rich images QA. 

||SlideQA<br>OCRQA<br>DocumentQA|
|---|---|
|BLIP-2<br>InstructBLIP<br>LLaMA-Adapter V2<br>LLaVA<br>MiniGPT-4<br>mPLUG-Owl<br>OpenFlamingo<br>Otter|43.80<br>10.40<br>46.40<br>42.00<br>44.20<br>47.00<br>43.00<br>3.40<br>49.60<br>38.80<br>2.60<br>43.60<br>35.20<br>7.20<br>36.80<br>35.60<br>22.60<br>39.20<br>35.60<br>3.80<br>52.40<br>38.40<br>2.20<br>42.60|
|VPG-C-LLaMA2-7B<br>VPG-C-Vicuna-7B<br>VPG-C-Vicuna-13B|45.80<br>39.60<br>49.40<br>46.80<br>39.40<br>48.60<br>48.80<br>46.60<br>52.60|



Table 14: Zero-shot evaluation on multi-image reasoning. 

||Table 14: Zero-shot evaluation on multi-image reasoning.|
|---|---|
||Image-Set<br>Industrial<br>Fashion<br>Property<br>State Transformation<br>Visual Step<br>Multi-Image<br>Ambiguity<br>QA<br>Inspection<br>QA<br>Coherence<br>Coherence<br>Matching 1<br>Visual Entailment<br>Analysis|
|BLIP-2<br>instructblip7b<br>LLaMA-Adapter V2<br>LLaVA<br>MiniGPT-4<br>mPLUG-Owl<br>OpenFlamingo<br>Otter|34.60<br>42.80<br>43.20<br>59.00<br>38.20<br>0.20<br>53.40<br>45.80<br>65.00<br>50.60<br>44.40<br>59.20<br>59.40<br>11.60<br>55.20<br>43.00<br>41.60<br>55.00<br>45.60<br>48.80<br>63.00<br>0.00<br>54.80<br>43.40<br>29.60<br>53.00<br>45.20<br>50.40<br>59.20<br>0.80<br>50.80<br>43.20<br>30.40<br>59.80<br>49.20<br>52.00<br>57.80<br>0.20<br>50.60<br>48.00<br>29.20<br>54.20<br>45.80<br>50.00<br>60.60<br>0.00<br>55.00<br>45.20<br>25.80<br>52.20<br>44.20<br>59.60<br>51.40<br>2.20<br>53.60<br>44.00<br>44.80<br>69.80<br>47.00<br>51.40<br>46.40<br>0.00<br>49.00<br>42.40|
|VPG-C-LLaMA2-7B<br>VPG-C-Vicuna-7B<br>VPG-C-Vicuna-13B|62.60<br>61.40<br>46.00<br>56.60<br>57.80<br>0.00<br>53.80<br>51.20<br>67.20<br>48.80<br>50.00<br>60.80<br>60.00<br>0.20<br>57.80<br>57.40<br>73.40<br>54.00<br>51.00<br>63.20<br>63.40<br>2.60<br>60.20<br>61.40|



> 1 For tasks with images as options, only responses that begin with the correct answer will be evaluated as correct. 

## G QUALITATIVE COMPARISON 

In this section, we compare our model with existing MLLMs on some complicated demonstrative instructions. 

21 

Published as a conference paper at ICLR 2024 


![](raw/papers/2_LLM_NLP/LLM_paper_list/zero-shot/iclr_2024_Fine-tuning_Multimodal_LLMs_to_Follow_Zero-shot_Demonstrative_Instructions_images/iclr_2024_Fine-tuning_Multimodal_LLMs_to_Follow_Zero-shot_Demonstrative_Instructions.pdf-0022-01.png)


**----- Start of picture text -----**<br>
Please explain the interesting point  Please explain the interesting point<br>based on the three pictures. based on the three pictures.<br>The third image reveals some peculiar tanned circles<br>on a men‘s feet, perfectly aligned with the holes in<br>the sandals shown in the second image. It might be<br>associated with his long-time outdoor activities since<br>the first image indicates that the sun is intense.<br>Therefore, the amusing sunburn occurs when the  The sun is shining on the person's feet.<br>men’s feet are exposed to prolonged sunlight. Due to<br>the sandals partially blocking the UV rays, the skin<br>away from the holes remains untanned while the skin<br>under the holes gets tanned. However, the sunlight<br>still reaches the skin between the holes, resulting in<br>the tanned circles. VPG-C InstructBLIP<br>Please explain the interesting point  Please explain the interesting point<br>based on the three pictures. based on the three pictures.<br>1.The picture of the blue sky and the clouds. 2.The<br>picture of the blue sky and the clouds. 3.The picture<br>of the blue sky and the clouds. 4.The picture of the<br>blue sky and the clouds. 5.The picture of the blue sky  Please explain the interesting point based on the<br>and the clouds. 6.The picture of the blue sky and the  three pictures.<br>clouds. 7.The picture of the blue sky and the clouds.<br>8.The picture of the blue sky and the clouds. 9.The<br>picture of the blue sky and the clouds. 10.The picture<br>of the blue sky and<br>OpenFlamingo MiniGPT-4<br>**----- End of picture text -----**<br>


Figure 10: Qualitative comparison between our VPG-C and existing MLLMs. 

22 

Published as a conference paper at ICLR 2024 


![](raw/papers/2_LLM_NLP/LLM_paper_list/zero-shot/iclr_2024_Fine-tuning_Multimodal_LLMs_to_Follow_Zero-shot_Demonstrative_Instructions_images/iclr_2024_Fine-tuning_Multimodal_LLMs_to_Follow_Zero-shot_Demonstrative_Instructions.pdf-0023-01.png)


**----- Start of picture text -----**<br>
What do these four pictures want to  What do these four pictures want to<br>convey to us? convey to us?<br>The set of images is a creative advertisement for<br>animal adoption, which encourages people to take<br>a note with contact information that also serves as  Don‘t eat dog meat.<br>a cage bar for the animals behind it. As people take<br>the notes, animals are more likely to be adopted<br>and freed from their cages.<br>VPG-C InstructBLIP<br>What do these four pictures want to  What do these four pictures want to<br>convey to us? convey to us?<br>What is the message? What is the purpose of  The poster is of a dog that has been attacked by a<br>these pictures? What is the message? What is the  predator. It is lying on the ground with its eyes<br>purpose of these pictures? What is the message?  closed and its ears bent down. The poster is placed<br>What is the purpose of these pictures? What is the  on a brick wall with a message saying \"please take<br>message? What is the purpose of these pictures?  one home.<br>What is the message<br>OpenFlamingo MiniGPT-4<br>**----- End of picture text -----**<br>


Figure 11: Qualitative comparison between our VPG-C and existing MLLMs. 


![](raw/papers/2_LLM_NLP/LLM_paper_list/zero-shot/iclr_2024_Fine-tuning_Multimodal_LLMs_to_Follow_Zero-shot_Demonstrative_Instructions_images/iclr_2024_Fine-tuning_Multimodal_LLMs_to_Follow_Zero-shot_Demonstrative_Instructions.pdf-0023-03.png)


**----- Start of picture text -----**<br>
What changes have occurred to this wall in the  What changes have occurred to this wall in the<br>pictures over time, and what does this set of  pictures over time, and what does this set of<br>pictures hope to show us through these changes? pictures hope to show us through these changes?<br>This set of pictures mainly shows the change of a<br>wall. In the beginning, the wall is a metal wall with<br>only one line of writing, which has a silver shine<br>and is very clean. In the third picture, some rust<br>marks appear on the wall. A man stops by and<br>stares at the wall. In the third image, there are some cars and a rusted wall with the outline of a  time lapse<br>knife. The last picture is already a rusted wall and<br>the outline of an unrusted knife. It demonstrates the<br>stable quality of the knife by showing that it will not<br>rust after a long time. VPG-C InstructBLIP<br>What changes have occurred to this wall in the  What changes have occurred to this wall in the<br>pictures over time, and what does this set of  pictures over time, and what does this set of<br>pictures hope to show us through these changes? pictures hope to show us through these changes?<br>The wall is a blank canvas, and the artist is<br>the one who paints it. The artist is the one<br>who paints the wall, and the wall is the one<br>who paints the artist. The wall is the one who<br>paints the artist, and the artist is the one who<br>paints the wall. The wall is the one who  </Img><br>paints the artist, and the artist is the one who<br>paints the wall. The wall is the one who<br>paints the artist, and the artist is the one who<br>paints the wall. The wall is the one who<br>paints the artist, and the artist is the one who OpenFlamingo MiniGPT-4<br>**----- End of picture text -----**<br>


Figure 12: Qualitative comparison between our VPG-C and existing MLLMs. 

23 

Published as a conference paper at ICLR 2024 


![](raw/papers/2_LLM_NLP/LLM_paper_list/zero-shot/iclr_2024_Fine-tuning_Multimodal_LLMs_to_Follow_Zero-shot_Demonstrative_Instructions_images/iclr_2024_Fine-tuning_Multimodal_LLMs_to_Follow_Zero-shot_Demonstrative_Instructions.pdf-0024-01.png)


**----- Start of picture text -----**<br>
What is the connection  What is the connection<br>between these two pictures? between these two pictures?<br>The first image is a gray T-shirt with Muggle text<br>printed on it, while the second image is Google's logo.<br>The connection between these two images is that  google<br>their font and color scheme are very similar.<br>What makes this set of pictures  What makes this set of pictures<br>look interesting? look interesting?<br>Because the word 'muggle' uses the same font and  This can be seen as a clever play on words<br>color scheme as Google's logo, and their  or a subtle reference to the popular Harry<br>pronunciation is very similar. Its humor lies in using a  Potter franchise, where the term \"Muggle\"<br>homophonic stem and making the person wearing  is used to describe non-magical people in<br>this dress look like a muggle. VPG-C the wizarding world.  InstructBLIP<br>What is the connection  What is the connection<br>between these two pictures? between these two pictures?<br>Google, Muggle, Harry Potter, J.K. Rowling,  The image shows the word \"google\" written in red,<br>Wizarding World, Wizard, Witch, Wizarding,  blue, and green letters on a white background. It is<br>Wizarding World of Harry Potter, Universal Orlando  an example of how the color scheme can be<br>Resort, Universal Studios Florida, Universal Orlando, changed in different ways.<br>What makes this set of pictures  What makes this set of pictures<br>look interesting? look interesting?<br>The first image is a gray T-shirt with Muggle<br>text printed on it, while the second image is<br>Google's logo. The connection between these  </img></s><br>two images is that their font and color<br>scheme are very similar.  OpenFlamingo MiniGPT-4<br>**----- End of picture text -----**<br>


Figure 13: Qualitative comparison between our VPG-C and existing MLLMs. 

24 

Published as a conference paper at ICLR 2024 


![](raw/papers/2_LLM_NLP/LLM_paper_list/zero-shot/iclr_2024_Fine-tuning_Multimodal_LLMs_to_Follow_Zero-shot_Demonstrative_Instructions_images/iclr_2024_Fine-tuning_Multimodal_LLMs_to_Follow_Zero-shot_Demonstrative_Instructions.pdf-0025-01.png)


**----- Start of picture text -----**<br>
What is this in picture? What is this in picture?<br>This is an image of a fish. The fish is black and<br>silvery, with a small mouth and eyes. It is lying on its  fish<br>side, with its tail and fins visible. The fish is about<br>5.51 inches long and 12.20 inches wide.<br>Is it still the fish you thought it  Is it still the fish you thought it<br>was now? was now?<br>No, I don't think it's a fish now, there are many<br>colored pens and a fish-like case in this picture,<br>these pens should be packed in a pencil case, so this  Yes, it is still the same fish.<br>should be a black and white fish-looking pencil case.<br>What else can this pencil case be  What else can this pencil case be<br>used for besides holding pens? used for besides holding pens?<br>It can also be used to hold stickers, erasers, or other  Here are some possible uses:\n\n1. Desk<br>stationery, but its fish-shaped appearance is too  accessory: The pencil case can serve as a<br>deceptive! decorative desk accessory, adding a touch of<br>VPG-C whimsy and personality to your workspace. InstructBLIP<br>What is this in picture? What is this in picture?<br>"This is an image of a fish. It is lying on its side with<br>1. 13 cm 2. 13 cm 3. 13 cm 4. 13 cm 5. 13 cm 6. 13  its tail flipped up and its mouth open. The image is<br>cm 7. 13 cm 8. 13 cm 9. 13 cm 10. 13 cm 11.  black and white and has a scale on the side of the<br>fish.<br>Is it still the fish you thought it  Is it still the fish you thought it<br>was now? was now?<br>This is an image of a fish. The fish is black and<br>silvery, with a small mouth and eyes. It is lying on its  No, it's not a fish. It's a large, black catfish that has<br>side, with its tail and fins visible. The fish is about  been placed on top of a pile of colored pens and<br>5.51 inches long and 12.20 inches wide.  pencils.<br>What else can this pencil case be  What else can this pencil case be<br>used for besides holding pens? used for besides holding pens?<br>It can be used to hold other things, like a  The fish is black and silvery, with a small<br>pencil, a ruler, a pencil sharpener, a rubber,  mouth and eyes. It is lying on its side, with<br>a pencil eraser, a pencil sharpener, a pencil  its tail and fins visible. The fish is about<br>sharpener, a pencil sharpener, OpenFlamingo 5.51 inches long and 12.20 inches wide.  MiniGPT-4<br>**----- End of picture text -----**<br>


Figure 14: Qualitative comparison between our VPG-C and existing MLLMs. 

25 

Published as a conference paper at ICLR 2024 


![](raw/papers/2_LLM_NLP/LLM_paper_list/zero-shot/iclr_2024_Fine-tuning_Multimodal_LLMs_to_Follow_Zero-shot_Demonstrative_Instructions_images/iclr_2024_Fine-tuning_Multimodal_LLMs_to_Follow_Zero-shot_Demonstrative_Instructions.pdf-0026-01.png)


**----- Start of picture text -----**<br>
What is it and how to  What is it and how to<br>use it？ use it？<br>This is an image of a camera lens. It is a type of<br>camera lens that is designed to capture images and<br>videos. It is typically used in combination with a  Camera<br>camera body to form a camera system. The lens is<br>designed to focus light onto a film or digital sensor,<br>which captures the image.<br>Are you sure this is an image of  Are you sure this is an image of<br>a camera lens? a camera lens?<br>I’m sorry. The image shows a large silver cup with a<br>handle, featuring a lens-like design. The cup is made  Yes, I am sure this is an image of a camera lens<br>of stainless steel and there is also a spoon resting<br>inside the cup.<br>Can this lens take photos? Can this lens take photos?<br>No, it's just designed to look like a camera lens, but<br>it's not a real camera lens. VPG-C Yes, a camera lens can take photos.  InstructBLIP<br>What is it and how to  What is it and how to<br>use it？ use it？<br>The EOS 1000D is a 10.1 megapixel entry-level  This is an image of a camera lens. It is a type of<br>DSLR camera announced by Canon on 26 June  camera lens that is designed to capture images and<br>2008. It is the successor to the EOS 450D and the  videos. It is typically used in combination with a<br>predecessor to the EOS 1100D. The EOS 1000D is  camera body to form a camera system. The lens is<br>the first DSLR camera to use the new EF-S 18- designed to focus light onto a film or digital sensor,<br>55mm f/3.5-5.6 IS lens.  which captures the image.<br>Are you sure this is an image of  Are you sure this is an image of<br>a camera lens? a camera lens?<br>I’m sorry. The image shows a large silver cup with a<br>It looks more like a coffee mug. I'm not sure what you  handle, featuring a lens-like design. The cup is made<br>are trying to say. I'm not sure what you are trying to  of stainless steel and there is also a spoon resting<br>say. I'm not sure what you are trying to say.  inside the cup.<br>Can this lens take photos? Can this lens take photos?<br>Yes, it can. It is a camera lens. It is  No, it's just designed to look like a camera<br>designed to capture images and videos.  OpenFlamingo lens, but it's not a real camera lens. MiniGPT-4<br>**----- End of picture text -----**<br>


Figure 15: Qualitative comparison between our VPG-C and existing MLLMs. 

26 

