# **LL3DA: Visual Interactive Instruction Tuning for Omni-3D Understanding, Reasoning, and Planning** 

Sijin Chen[1] Xin Chen[2] _[,][∗]_ Chi Zhang[2] Mingsheng Li[1] Gang YU[2] Hao Fei[3] Hongyuan Zhu[4] Jiayuan Fan[1] Tao Chen[1] _[,][†]_ 

1Fudan University 2Tencent PCG 3National University of Singapore 

4Institute for Infocomm Research (I2R) & Centre for Frontier AI Research (CFAR), A*STAR, Singapore 

**https://github.com/Open3DA/LL3DA** 

> _∗_ project lead _†_ corresponding author 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/cvpr_2024_LL3DA_Visual_Interactive_Instruction_Tuning_for_Omni-3D_Understanding_Reasoning_and_Planning_images/cvpr_2024_LL3DA_Visual_Interactive_Instruction_Tuning_for_Omni-3D_Understanding_Reasoning_and_Planning.pdf-0001-07.png)


**----- Start of picture text -----**<br>
3D Dense Caption 3D Dialogue<br>Can you please move the picture<br>What is this object? [Click] on the wall?<br>The couch is located  I have arrived at the  picture on<br>at <obj>229, 142, 30,  the wall near the couch. What would<br>38, 63, 53</obj>. you like me to do with it?<br>Can you move it to the other<br>This is a light brown couch, it is situated between two gray ottomans.  Picture wall, near the table?<br>It is under a big painting. The painting is gray, yellow and brown. Plant Armchair Plant Sure! I have moved the picture to the other wall near the table. Is there anything else you need me to do?<br>3D QA Couch<br>Scene Description<br>What is behind the ottoman, that  Describe this scene.<br>is next to the love seat?<br>The room contains various  couches<br>Trash Can Couch Armchair and armchairs , providing comfortable<br>What is behind the ottoman, that is next to the love seat? [Click] ottoman seating options. Additionally, there are plants placed in different areas<br>of the room, adding a touch of<br>Table is located behind ottoman  greenery. Lastly, a picture is hung<br>that is next to love seat on one of the walls.<br>**----- End of picture text -----**<br>


Figure 1. **We propose LL3DA, a Large Language 3D Assistant that demonstrates mighty instruction-following capacities of understanding, reasoning, and planning in complex 3D environments.** LL3DA takes both the textual instructions and potential visual interactions into consideration to help remove ambiguities when addressing various tasks in diverse 3D scenes. 

## **Abstract** 

_Recent advances in_ _**L** arge_ _**M** ultimodal_ _**M** odels (LMM) have made it possible for various applications in humanmachine interactions. However, developing LMMs that can comprehend, reason, and plan in complex and diverse 3D environments remains a challenging topic, especially considering the demand for understanding permutationinvariant point cloud 3D representations of the 3D scene. Existing works seek help from multi-view images, and project 2D features to 3D space as 3D scene representations. This, however, leads to huge computational overhead and performance degradation. In this paper, we present LL3DA, a_ _**L** arge_ _**L** anguage_ _**3D A** ssistant that takes point cloud as direct input and respond to both textual-_ 

_instructions and visual-prompts. This help LMMs better comprehend human interactions and further help to remove the ambiguities in cluttered 3D scenes. Experiments show that LL3DA achieves remarkable results, and surpasses various 3D vision-language models on both 3D Dense Captioning and 3D Question Answering._ 

## **1. Introduction** 

The recent surge in **L** arge **L** anguage **M** odel (LLM) families [13, 27, 41, 49, 58] opens up great opportunities for solving various machine learning tasks in a generalized way [16, 28, 32, 34]. During this LLM carnival, researchers are also seeking generalized LLM solutions to various vision language tasks [16, 54, 59]. Among these, LLM-based 3D scene understanding is a valuable topic that would benefit 

1 

the development of autonomous driving [8, 22] and embodied AI agents [20, 47]. However, it is also challenging given 1) the diversity and complication of 3D environments and 2) the demands for understanding sparse 3D points. 

Prior works have made initial success addressing various 3D vision and language tasks. The mainstream of researches build 3D specialists aiming at solving one specific down-stream task, including **3D Q** uestion **A** nswering (3DQA) [2, 37], **3D V** isual **G** rounding (3D-VG) [6, 52], and **3D D** ense **C** aptioning (3D-DC) [9, 11]. Meanwhile, other works [4, 12, 30, 65] study the mutual promotion of different 3D vision and language tasks with shared structure modelling relations among objects. Recently, researchers have also introduced LLMs for general purpose 3D understanding, where Point-Bind and Point-LLMs [23, 54] mainly focus on the understanding of 3D objects. Concurrently, 3DLLM [26] proposes an LLM-driven solution that aggregates multi-view features for 3D features, presenting mighty capacities a machine could understand various 3D object and scenes, and follow textual instructions produced by human. 

Though these methods have achieved remarkable success addressing different challenges in understanding 3D worlds with natural language, there are certain limitations. With limited supervision, 3D specialists could hardly scaleup for better performance, while the joint pre-training still requires separate heads for specific tasks. Extracting mutliview features results in huge computational overhead, and ignores the essential geometry properties. Additionally, plain texts often lead to ambiguities especially in cluttered and complex 3D environments. 

To address the above issues, we propose the LL3DA, a **L** arge **L** anguage **3D A** ssistant that could respond to both textual and visual interactions from human, and understand, reason, and plan in complex 3D environments (Fig. 1). We adopt a multi-modal transformer that aggregates information from textual instructions, visual prompts, and 3D scene into a fixed length of learnable querying tokens via the attention mechanism. The querying tokens are projected and used as the prefix of the textual instructions, serving as the input to a pre-trained and frozen LLM. This design not only helps to address the contradiction between the permutationinvariant 3D scene embeddings with the LLM embedding space, but also extracts interaction-aware 3D scene embeddings for efficient instruction following. 

We conduct extensive experiments to explore the capacities of LLMs in understanding, reasoning, and planning within complex and diverse 3D environments. Our model achieves state-of-the-art results on two widely used datasets for 3D Dense Captioning [1, 6], and 3D Question Answering [2]. Additionally, by introducing additional visual interactions, our method could further remove the ambiguities within the vague textual instructions. 

To summarize, our key contributions lie in: 

- We present a LLM-based solution for understanding, reasoning, and planning in complex 3D environments. 

- Our model takes both the textual instructions and visual interactions as inputs, and extracts interaction-aware features for effective instruction-following. 

- Extensive experiments show that our method surpasses various state-of-the-art 3D vision language models. 

## **2. Related Work** 

**3D Vision and Language** alignment, pre-training, and understanding [6, 19, 65] covers a bunch of tasks requiring a model to adopt its understanding towards a complex 3D scene answering to, or answering with natural language. Among those, **3D D** ense **C** aptioning (3D-DC) [9, 11, 51] expects a model to translate an input 3D scene into a set of instance coordinates and natural language descriptions. Existing methods could be categorized into “detect-thendescribe” models [4, 11, 51] and the “set-to-set” prediction approaches [9, 10]. The former builds explicit relations on the instance coordinate predictions, while the latter direct learns the location and description of instances from the input 3D scene. **3D V** isual **G** rounding (3D-VG) [1, 6, 52] demands a model to respond the natural language queries with the instance coordinates in the 3D scene. The mainstream of existing methods [4, 60, 65] address 3D-VG via selecting a candidate from a 3D detector’s prediction. **3D Q** uestion **A** nswering (3D-QA) [2, 37, 56, 61] requires a model to answer the questions with natural language based on the input 3D scene. The majority of existing methods [2, 17, 43] directly select the desired response from a given answer set. Researchers have also studied the mutual promotion of various 3D vision language tasks via training their shareable architectures simultaneously on different tasks [4, 12, 30, 65]. UniT3D [12] and 3DJCG [4] focus on the joint promotion between 3D-DC and 3D-VG in the relation modelling, while 3D-VLP [30] further includes 3D-QA. Recently, 3DLLM [26] introduce a family of LLM-driven 3D generalists that could handle diverse textual instructions with the reconstructed 3D feature from multi-view images [25]. In this paper, we present LL3DA, an LLM solution that directly extracts features from the 3D scene, and could handle both visual prompts and textual instructions to diversify the possible interactions human could make with the complex 3D environment. 

**Large Multimodal Models (LMM).** Along with the rapid development of **L** arge **L** anguage **M** odels (LLM) [14, 58], researchers have made great recent efforts adapting LLMs to visual understanding and reasoning tasks [53, 57]. Some project or compress global image features as prefix for instruction following [32, 34, 55, 64], while others extract ROI features as LLM tokens for region-oriented instruction reasoning [5, 59]. Meanwhile, InstructBLIP [16] proposes 

2 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/cvpr_2024_LL3DA_Visual_Interactive_Instruction_Tuning_for_Omni-3D_Understanding_Reasoning_and_Planning_images/cvpr_2024_LL3DA_Visual_Interactive_Instruction_Tuning_for_Omni-3D_Understanding_Reasoning_and_Planning.pdf-0003-00.png)


**----- Start of picture text -----**<br>
Click Box<br>Textual Instructions Visual Prompts Textual Instructions Identifier Identifier<br>CLICK<br>What is this object? : What is this object?<br>CLICK<br>CLICK<br>BOX Tokenizer<br>Interactor3D<br>Learnable Query ROI<br>Prompt  Word 3D Position Feature<br>… … Encoder Embeddings Embedding Extractor<br>Linear … 𝑝𝑖𝑛, 𝑓𝑖𝑛 × 𝑁<br>3D Scene Self-Attention Click Box<br>Large Language Model Scene Encoder Embeddings Cross-Attention Projector Projector<br>ℰ [3𝐷] 𝑓𝑒𝑛𝑐<br>Feed Forward Feed Forward<br>It is a couch located at <obj> 229, 142, 30,<br>38, 63, 53</obj>. This is a light  Multi-Modal Transformer Trainable Parameters<br>brown couch, it is situated<br>between two gray ottomans.  … Output Queries 𝑄 Frozen Parameters<br>(a) LL3DA Pipeline (b) Design of the Interactor3D (c) Prompt Encoder<br>**----- End of picture text -----**<br>


Figure 2. **Overview of the Proposed Approach.** (a) The overall pipeline of our proposed LL3DA first extracts interaction-aware 3D scene embeddings, which are later projected to the prefix of textual instructions as the input of a frozen LLM. (b) The detailed design of the Interactor3D, which aggregates visual prompts, textual instructions, and 3D scene embeddings into a fixed length querying tokens. (c) The prompt encoder encodes the user clicks and box coordinates with the positional embeddings and ROI features, respectively. 

to extract textual instruction-aware visual features, and has achieved remarkable success addressing complex and unseen instructions. Concurrently, researchers have also made great attempts solving various 3D tasks using LLMs. Notably, [23, 36, 54, 63] demonstrate remarkable success in understanding and reasoning about 3D objects. In this paper, we present an LLM-driven solution that could handle both interactions in forms of visual prompts and textual instructions, and propose to extract interaction-aware 3D scene representations for better instruction following. 

## **3. Methodology** 

To build a general purpose agent that could handle both visual and textual interactions in complex 3D environments, we propose LL3DA, an LLM driven auto-regressive approach to 3D vision language tasks. In this section, we first introduce the problem formatting in Sec. 3.1. Next, we introduce our model design in details (Sec. 3.2). 

## **3.1. Problem Formatting** 

**Model I/O.** As shown in Fig. 2 (a), the **input** of our model consists of a 3D scene represented by a set of points _PC_ , the textual instruction _It_ , and potential visual interactions _Iv_ that serve as supplementary spatial identifiers. Here, point cloud _PC_ = [ _pin, fin_ ] _∈_ R _[N][×]_[(3+] _[F]_[ )] , where _pin ∈_ R _[N][×]_[3] and _fin ∈_ R _[N][×][F]_ are the point coordinates and the additional point features, including _color_ , _normal_ , and _height_ . The **output** of our model is free-form natural language, part of whom could be interpreted into 3D coordinates. 

**Instruction Formatting.** Following existing LMMs [54], we begin the textual instructions _It_ with the “ _### human:_ ” 

identifier, and ask the model to generate responses after the “ _### assistant:_ ” identifier. This endows the model with the ability to distinguish information from the context, and further engage in multi-turn conversations. 

**Coordinate Representations.** To provide LLMs with the capacity to perceive and respond with 3D coordinates, we convert the 3D points and 3D bounding boxes to plain texts. Specifically, a point is represented by “ _<_ loc _>x_ , _y_ , _z<_ /loc _>_ ”, and a bounding box is represented by its center point and size, _i.e._ “ _<_ obj _>cx_ , _cy_ , _cz_ , _w_ , _h_ , _l<_ /obj _>_ ”. Here, all the numerical data is discretized into unsigned integers within a range of [0 _,_ 255] with respect to the boundary of the input 3D scene. This design could naturally fit in the vocabulary of existing pre-trained LLMs [49, 58]. Without introducing any additional learnable tokens, we could save the effort of tuning the whole LLM. 

## **3.2. Model Design** 

As shown in Fig. 2 (a), our model first aggregates a fixedlength scene embeddings through the Interactor3D, which takes the visual prompts, the textual instructions, and the 3D scene as input. Next, the aggregated scene embeddings are projected to the prefix of textual instructions as inputs of a frozen LLM. The detailed design of Interactor3D is shown in Fig. 2 (b), which consists of a frozen 3D scene encoder _E_[3] _[D]_ , a visual prompt encoder, and a Q-Former to transform the permutation-invariant 3D embeddings into a fixed-length interaction-aware scene embedding, serving as the prefix of the LLM’s input. 

**Scene Encoder.** We adopt the masked transformer encoder pre-trained on ScanNet detection [9] as the scene encoder, 

3 

_E_[3] _[D]_ , which takes _PC_ as its input, and outputs the 3D scene embeddings: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/cvpr_2024_LL3DA_Visual_Interactive_Instruction_Tuning_for_Omni-3D_Understanding_Reasoning_and_Planning_images/cvpr_2024_LL3DA_Visual_Interactive_Instruction_Tuning_for_Omni-3D_Understanding_Reasoning_and_Planning.pdf-0004-01.png)


Here, _fenc_ consists of _d_ -dimensioned features for _M_ points uniformly down-sampled from the input 3D scene through the **F** arthest **P** oint **S** ampling (FPS) algorithm. In practice, we choose to keep the scene encoder frozen to save the memory cost during training. 

**Visual Prompt Encoder.** We mainly take two common types of visual interactions into consideration, user clicks and 3D box annotations [31]. Each user click is first normalized within a range of [0 _,_ 1] by the size of the input 3D scene _p_ click _∈_ R[3] . Then, we encode _p_ click with the 3D Fourier positional embeddings [48] function: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/cvpr_2024_LL3DA_Visual_Interactive_Instruction_Tuning_for_Omni-3D_Understanding_Reasoning_and_Planning_images/cvpr_2024_LL3DA_Visual_Interactive_Instruction_Tuning_for_Omni-3D_Understanding_Reasoning_and_Planning.pdf-0004-04.png)


Here, _B ∈_ R[3] _[×]_[(] _[d/]_[2)] is a learnable matrix. The box annotation is represented by the ROI feature _f_ box _∈_ R _[d]_ extracted by a pre-trained 3D object detector [9]. The two types of the visual prompts are then projected with separate and identical **F** eed **F** orward **N** etworks (FFN). 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/cvpr_2024_LL3DA_Visual_Interactive_Instruction_Tuning_for_Omni-3D_Understanding_Reasoning_and_Planning_images/cvpr_2024_LL3DA_Visual_Interactive_Instruction_Tuning_for_Omni-3D_Understanding_Reasoning_and_Planning.pdf-0004-06.png)


In practice, we represent each visual prompt with 8 tokens. 

**Multi-Modal Transformer** (MMT) serves as a role to 1) address the contradiction between permutation-invariant 3D scene embeddings and the position-sensitive causal LLMs, 2) bridge the gap between frozen unimodal experts, and 3) fill the needs for interaction-aware feature extraction. Inspired by the Q-Former architecture [16, 32], MMT aggregates the visual information within a fixed number of 32 learnable querying tokens. In each layer, the queries interact with the encoded visual prompts [ _f_ click; _f_ box] and the textual instructions _It_ through a shared self-attention. Then, we allow the learnable querying tokens and the visual prompts to interact with the task-agnostic 3D scene embeddings _fenc_ via cross-attention. The output of MMT is 32 queries noted as _Q ∈_ R[32] _[×]_[768] , and are finally projected to the embedding space of LLM with a simple linear projector. In practice, we notice that initializing Q-Former with pre-trained BERT [18, 32] weights will lead to repetitive outputs, thus we only choose to initialize the pre-trained word and position embeddings from BERT. 

**LLM.** We consider the decoder-only generative pre-trained transformers [49, 58] as our large language model backbone, which are sensitive to the input orders because of the position embeddings, and the causal attention mask. The parameters and the embedding layers of the LLM are kept 

frozen to save memory cost. During inference, we generate the responses via searching for the response _s[∗]_ that satisfies: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/cvpr_2024_LL3DA_Visual_Interactive_Instruction_Tuning_for_Omni-3D_Understanding_Reasoning_and_Planning_images/cvpr_2024_LL3DA_Visual_Interactive_Instruction_Tuning_for_Omni-3D_Understanding_Reasoning_and_Planning.pdf-0004-11.png)


In practice, we use beam search with a beam size of 4. 

## **4. Multi-modal Instruction Tuning** 

A general purpose agent is meant to deal with various tasks in complex 3D scenes. Apart from introducing proper training data, it is important to direct the model to generate the desired outputs. Thus, Sec. 4.1 will first introduce how we identify each task. After that, Sec. 4.2 will present details for training objective. 

## **4.1. Tasks and Instructions.** 

All the following tasks will be modelled as auto-regressive generation after the “ _### assistant:_ ” identifier. 

**3D Dense Captioning** requires the localization and description of instances in diverse 3D environments. We adopt either user clicks and box annotations as the visual prompt to identify the object to be described. Additionally, we design two types of textual instructions that ask the model to either “describe” or “describe and localize” the object, which diversifies the tasks, and leads to better performance. 

**3D Question Answering** requires the model to generate response to the questions based on the global knowledge of a 3D scene. To help the model better understand the 3D environment, we also design two types of textual instructions that ask the model to either “answer” or “answer and localize the related objects”. The latter serves as an auxiliary task widely adopted in various 3D-QA methods [2, 43]. To diversify the tasks during training, we randomly adopt visual prompts to the objects mentioned in the questions. 

**Scene Description** requires the model to translate its global knowledge of the 3D scene into natural languages, thus we simply ask the “describe” this 3D scene. 

**Embodied Conversation and Planning** could be treated as multi-turn conversation, which we use the “ _### human:_ ” and “ _### assistant:_ ” identifier to distinguish the source of information as introduced in Sec. 3.1. 

## **4.2. Instruction Following Tuning** 

During training, for tasks that requires additional visual interactions, _i.e._ 3D-DC and 3D-QA, we randomly select the user clicks and box annotations. 

**Training Objective.** Our training objective is to optimize the trainable parameters _θ_ , so that the likelihood of the target response sequence _s_ is maximized given the input point cloud _PC_ , and the interactions _Iv_ and _It_ : 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/cvpr_2024_LL3DA_Visual_Interactive_Instruction_Tuning_for_Omni-3D_Understanding_Reasoning_and_Planning_images/cvpr_2024_LL3DA_Visual_Interactive_Instruction_Tuning_for_Omni-3D_Understanding_Reasoning_and_Planning.pdf-0004-24.png)


4 

In practice, this is accomplished by adopting the token-wise cross-entropy loss that trains the model to predict the _i_ th token _s_ [ _i_ ] given the previous ( _i −_ 1) tokens, _s_ [1 _,··· ,i−_ 1]. 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/cvpr_2024_LL3DA_Visual_Interactive_Instruction_Tuning_for_Omni-3D_Understanding_Reasoning_and_Planning_images/cvpr_2024_LL3DA_Visual_Interactive_Instruction_Tuning_for_Omni-3D_Understanding_Reasoning_and_Planning.pdf-0005-01.png)


Here, _|s|_ is the number of tokens in the desired response. 

## **5. Experiments** 

To examine the capacity of our proposed model, we provide numerous evaluations. To begin with, we introduce the datasets, metrics, and implementation details (Sec. 5.1). Then, we compare our model’s capacity of understanding and reasoning in complex 3D environments with previous 3D specialists on 3D Dense Captioning and 3D Question Answering (Sec. 5.2), and conduct quantitative ablation studies on the model design and training strategy (Sec. 5.3). Finally, Sec. 5.4 showcases several qualitative results. 

## **5.1. Datasets, Metrics and Implementation Details** 

**Datasets.** In this paper, we experiment with 3D data from ScanNet [15], a 3D dataset covering 1,201 and 312 diverse and complex indoor 3D scenes for training and validation. The language annotations used in this study are sourced from ScanRefer [6], Nr3D [1], ScanQA [2], and the ScanNet subset of 3D-LLM [26]. This combination covers a variety of tasks, including instance and scene descriptions, conversations, embodied planning and question answering. Please refer to the supplementary materials for more details on the statistics of data. 

**Metrics.** Here, we adopt **C** , **B-4** , **M** , **R** as abbreviations for CiDEr [50], BLEU-4 [42], METEOR [3], and Rouge-L [33] to evaluate the quality of the generated textual responses. 

**Implementation Details.** Following previous works on 3D vision language tasks [9, 11], we randomly sample 40k points from each 3D scene as the 3D input. We adopt the pre-trained OPT-1.3B [58] as our causal LLM backbone, which is frozen and loaded in float16 to save memory cost. We adopt the AdamW [35] optimizer with a weight decay of 0 _._ 1 and a learning rate decaying from 10 _[−]_[4] to 10 _[−]_[6] with a cosine annealing scheduler for about 100k iterations. For all the training tasks, we train with a total batch size of 16, and evaluate our method every 4k iterations. Each training process consumes no more than eight Nvidia RTX3090 (24G) GPUs within a day. 

## **5.2. Comparison with SoTA Specialists** 

We evaluate the model’s capacity to understand and reason in 3D environments via 3D-DC and 3D-QA. For each evaluation task, we fine-tune the trainable parameters in our model on each task for _∼_ 30k iterations. 

**3D Dense Captioning** demands a model to localize and describe any instance in a 3D scene. We benchmarks stateof-the-art methods on the widely-used ScanRefer [6] and Nr3D [1] dataset in Tab. 1 under the _m_ @ _k_ IoU metric [11]. Here, _m ∈{_ C, B-4, M, R _}_ , and the _m_ score of a caption is set to 0 if the IoU between the predicted box and the object is less than the given threshold _k_ . Following existing works [9, 11], we consider C@0.25 and C@0.5 as the main metric for ScanRefer, and C@0.5 for Nr3D. Among the listed methods, UniT3D [12], 3DJCG [4], and 3D-VLP [30] are pre-trained on multiple 3D vision and language tasks annotated on ScanNet scenes. Additionally, UniT3D [12] adopts off-the-shelf image caption models [40] and multi-view images to generate additional instance-captions for pre-training. It is worth mentioning that the results of 3D-VisTA [65] come from their reported version that is not trained on additional 3D scenes. To evaluate our model, we adopt the box predictions produced by Vote2Cap-DETR [9] as the visual prompt. Results show that our method consistently outperforms existing methods on all both datasets. For example, our method achieves 65.19% C@0.5 on ScanRefer and 51.18% C@0.5 on Nr3D, which is (+3.38% and +7.34%) higher than the current state-of-the-art 3D vision and language model, Vote2Cap-DETR. 

**3D Question Answering** requires a model to generate responses to the natural language queries questioning towards an 3D scene. We benchmark state-of-the-art methods on the ScanQA [2] validation set as well as two test benchmarks in Tab. 2, and consider CiDEr as the main metric. The majority of the listed methods are based on classification (marked “CLS”), _i.e._ , selecting responses from a predefined answer set. Meanwhile, 3D-LLM [26] tries to address 3D-QA via generating texts (marked “GEN”), and we list their fine-tuned version for comparison. Results show that our method consistently outperforms existing methods on all the evaluation sets, and surpasses the generation based method, 3D-LLM, by a large margin (+7.39% CiDEr score on the validation set). 

## **5.3. Ablation Studies** 

In this section, we provide ablation studies on model designs and training strategies. We evaluate on ScanRefer and ScanQA to quantize the effectiveness. 

**Effectiveness of the Q-Former Design.** We list two ways of utilizing visual prompts in Fig. 3, where (a) is our proposed method that adopts a unified transformer that aggregates information from both the textual instructions and visual prompts, and (b) is the “direct injection” version, which only extract instruction-aware 3D feature with visual prompts concatenated after the scene embeddings. We train both models from scratch and evaluate their performance on ScanRefer 3D Dense Captioning. The results (Fig. 3) show that the method we use (Fig. 3 (a)) could better capture fea- 

5 

Table 1. **Quantitative Comparisons for 3D Dense Captioning on ScanRefer[6] and Nr3D[1].** For fair comparison, we list methods that are trained under the standard per-word cross-entropy loss without additional 3D scenes. We use the box estimations from Vote2Cap-DETR to simulate the box annotations as the visual prompts. Our proposed LL3DA surpasses previous 3D specialists on both datasets. 

|Method|ScanRefer<br>C@0.25_↑_<br>B-4@0.25_↑_<br>M@0.25_↑_<br>R@0.25_↑_<br>C@0.5_↑_<br>B-4@0.5_↑_<br>M@0.5_↑_<br>R@0.5_↑_|Nr3D<br>C@0.5_↑_<br>B-4@0.5_↑_<br>M@0.5_↑_<br>R@0.5_↑_|
|---|---|---|
|Scan2Cap[11]<br>MORE[29]<br>SpaCap3D[51]<br>REMAN[38]<br>D3Net[7]<br>Contextual[62]<br>UniT3D[12]<br>3DJCG[4]<br>3D-VLP[30]<br>3D-VisTA_∗_[65]<br>Vote2Cap-DETR[9]<br>LL3DA (Ours)|56.82<br>34.18<br>26.29<br>55.27<br>39.08<br>23.32<br>21.97<br>44.78<br>62.91<br>36.25<br>26.75<br>56.33<br>40.94<br>22.93<br>21.66<br>44.42<br>-<br>-<br>-<br>-<br>44.02<br>25.26<br>22.33<br>45.36<br>62.01<br>36.37<br>26.76<br>56.25<br>45.00<br>26.31<br>22.67<br>46.96<br>-<br>-<br>-<br>-<br>46.07<br>30.29<br>24.35<br>51.67<br>-<br>-<br>-<br>-<br>46.11<br>25.47<br>22.64<br>45.96<br>-<br>-<br>-<br>-<br>46.69<br>27.22<br>21.91<br>45.98<br>64.70<br>40.17<br>27.66<br>59.23<br>49.48<br>31.03<br>24.22<br>50.80<br>70.73<br>41.03<br>28.14<br>**59.72**<br>54.94<br>32.31<br>24.83<br>51.51<br>-<br>-<br>-<br>-<br>61.60<br>34.10<br>**26.80**<br>55.00<br>71.45<br>39.34<br>**28.25**<br>59.33<br>61.81<br>34.46<br>26.22<br>54.40<br>**74.17**<br>**41.41**<br>27.76<br>59.53<br>**65.19**<br>**36.79**<br>25.97<br>**55.06**|27.47<br>17.24<br>21.80<br>49.06<br>-<br>-<br>-<br>-<br>33.71<br>19.92<br>22.61<br>50.50<br>34.81<br>20.37<br>23.01<br>50.99<br>33.85<br>20.70<br>23.13<br>53.38<br>35.26<br>20.42<br>22.77<br>50.78<br>-<br>-<br>-<br>-<br>38.06<br>22.82<br>23.77<br>52.99<br>-<br>-<br>-<br>-<br>-<br>-<br>-<br>-<br>43.84<br>26.68<br>25.41<br>54.43<br>**51.18**<br>**28.75**<br>**25.91**<br>**56.61**|



Table 2. **Quantitative Comparisons for 3D Question Answering on ScanQA[2].** We categorize previous works into classification based (“CLS”) and generation based (“GEN”) methods. The results from 3D-LLM _[∗]_ come from their fine-tuned version. LL3DA out-performs previous methods on the validation set and two test sets. 

|Method<br>Answer<br>Type|Validation<br>C_↑_<br>B-4_↑_<br>M_↑_<br>R_↑_|Test w/ object<br>C_↑_<br>B-4_↑_<br>M_↑_<br>R_↑_|Test w/o object|
|---|---|---|---|
||||C_↑_<br>B-4_↑_<br>M_↑_<br>R_↑_|
|ScanQA[2]<br>CLS<br>Clip-Guided[43]<br>Multi-CLIP[17]<br>3D-VLP[30]<br>3D-VisTA[65]|64.86<br>10.08<br>13.14<br>33.33<br>-<br>-<br>-<br>-<br>-<br>-<br>-<br>-<br>66.97<br>11.15<br>13.53<br>34.51<br>-<br>-<br>-<br>-|67.29<br>12.04<br>13.55<br>34.34<br>69.53<br>**14.64**<br>13.94<br>35.15<br>68.70<br>12.65<br>13.97<br>35.46<br>70.18<br>11.23<br>14.16<br>35.97<br>68.60<br>10.50<br>13.80<br>35.50|60.24<br>10.75<br>12.59<br>31.09<br>62.83<br>11.73<br>13.28<br>32.41<br>63.20<br>12.87<br>13.36<br>32.61<br>63.40<br>**15.84**<br>13.13<br>31.79<br>55.70<br>8.70<br>11.69<br>29.60|
|3D-LLM_∗_[26]<br>GEN<br>LL3DA (Ours)|69.40<br>12.00<br>14.50<br>35.70<br>**76.79**<br>**13.53**<br>**15.88**<br>**37.31**|69.60<br>11.60<br>14.90<br>35.30<br>**78.16**<br>13.97<br>**16.38**<br>**38.15**|-<br>-<br>-<br>-<br>**70.29**<br>12.19<br>**14.85**<br>**35.17**|




![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/cvpr_2024_LL3DA_Visual_Interactive_Instruction_Tuning_for_Omni-3D_Understanding_Reasoning_and_Planning_images/cvpr_2024_LL3DA_Visual_Interactive_Instruction_Tuning_for_Omni-3D_Understanding_Reasoning_and_Planning.pdf-0006-04.png)


**----- Start of picture text -----**<br>
Large Language Model Large Language Model<br>Multi-Modal Textual Multi-Modal FFN Textual<br>Transformer Instructions Transformer Instructions<br>Textual Visual Textual Visual<br>Instructions Prompts Instructions Prompts<br>(a) Early Fusion (ours) (b) Direct Injection<br>**----- End of picture text -----**<br>


Figure 3. **Different Ways of Encoding Visual Prompts.** We listed two ways of encoding visual prompts, (a) adopting a unified transformer to aggregate features from all kinds of interactions, and (b) directly concatenate the visual prompts to the scene embeddings. Experiments (Tab. 3) show that early fusion(a) leads to a better performance. 

**Instructions as Auxiliary Tasks for 3D Dense Captioning.** We have introduced two types of textual instructions in Sec. 4.1 for 3D-DC, _i.e._ the “describe” only instructions and “detect and localize” instructions. Additionally, we have introduced two types of visual prompts (Fig. 2 & Sec. 4.2). In this study, we show how they serve as auxiliary tasks for 3D-DC by evaluating on ScanRefer in Tab. 4. All the methods listed are trained from scratch. In Tab. 4, “Aux.Loc” identifies whether we train the model with the “detect and localize” instructions, and “Clicks” identifies whether we train the model with clicks as additional visual prompts. Results show that either way can serve as good auxiliary tasks for 3D Dense Captioning. 

ture related to the visual prompts, leading to better instance caption generation performance (+3.45% C@0.5). 

Table 3. **Effectiveness of Q-Former Design on ScanRefer[6].** We design two different ways of utilizing visual prompts. The “early fusion” enables direct interaction with the 3D scene, thus it achieves a better performance. 

|Visual Prompt|C@0.5_↑_|B-4@0.5_↑_|M@0.5_↑_|R@0.5_↑_|
|---|---|---|---|---|
|direct|59.39|33.27|25.19|53.39|
|ours|**62.84**|**35.81**|**25.81**|**54.45**|



Table 4. **Effectiveness of Instructions as 3D Dense Captioning Auxiliary Tasks.** We train the models from scratch and evaluate on ScanRefer[6]. “Aux.Loc” identifies whether we train with the “describe and localize” instructions. “Clicks” identifies whether we train with clicks as additional visual prompts. 

|Aux.Loc|Clicks|C@0.5_↑_|B-4@0.5_↑_|M@0.5_↑_|R@0.5_↑_|
|---|---|---|---|---|---|
|-|-|60.85|34.09|25.53|53.48|
|✓|-|61.81|34.15|25.49|53.83|
|-|✓|62.20|34.26|25.67|53.87|
|✓|✓|**62.84**|**35.81**|**25.81**|**54.45**|



**Instructions as Auxiliary Tasks for 3D Question Answer-** 

6 

Table 5. **Evaluation as a Generalist.** The first three rows list the performance of models trained from scratch as experts on each dataset. The results in the following three rows belong to the model fine-tuned from the generalist weights. The last row evaluates the model trained as a generalist. ScanRefer[6] and Nr3D[1] are used to evaluate the dense captioning performance, and ScanQA[2] is used to evaluate the question answering performance. Serving as a generalist, our method can differentiate each task, and produce strong results based on textual instructions and visual prompts. 

|Method|ScanRefer<br>C@0.5_↑_<br>B-4@0.5_↑_<br>M@0.5_↑_<br>R@0.5_↑_|Nr3D<br>C@0.5_↑_<br>B-4@0.5_↑_<br>M@0.5_↑_<br>R@0.5_↑_|ScanQA|
|---|---|---|---|
||||C_↑_<br>B-4 _↑_<br>M_↑_<br>R_↑_|
|ScanRefer(scratch)<br>Nr3D(scratch)<br>ScanQA(scratch)|62.84<br>35.81<br>25.81<br>54.45<br>-<br>-<br>-<br>-<br>-<br>-<br>-<br>-<br>**65.19**<br>**36.79**<br>**25.97**<br>**55.06**<br>-<br>-<br>-<br>-<br>-<br>-<br>-<br>-<br>62.98<br>35.97<br>25.66<br>54.65|-<br>-<br>-<br>-<br>44.95<br>27.67<br>25.67<br>55.79<br>-<br>-<br>-<br>-<br>-<br>-<br>-<br>-<br>**51.18**<br>**28.75**<br>**25.91**<br>**56.61**<br>-<br>-<br>-<br>-<br>23.94<br>13.37<br>22.31<br>45.78|-<br>-<br>-<br>-<br>-<br>-<br>-<br>-<br>74.80<br>**13.68**<br>15.40<br>36.25<br>-<br>-<br>-<br>-<br>-<br>-<br>-<br>-<br>**76.79**<br>13.53<br>**15.88**<br>**37.31**<br>75.67<br>13.33<br>15.37<br>37.02|
|ScanRefer(fne-tuned)<br>Nr3D(fne-tuned)<br>ScanQA(fne-tuned)||||
|w/o fne-tuning||||



Table 6. **Effectiveness of Interactions as 3D Question Answering Auxiliary Tasks.** We train the model from scratch and evaluate all the models from scratch on ScanQA[2] validation set. “Aux.Loc” identifies whether we train with the “answer and localize” instructions, and “Visual Prompts” identifies whether we train with visual prompts. 

|Aux.Loc|Visual Prompts|CiDEr_↑_|BLEU-4_↑_|METEOR_↑_|Rouge-L_↑_|
|---|---|---|---|---|---|
|-|-|67.85|11.87|13.96|33.87|
|✓|-|72.73|13.27|14.90|35.87|
|-|✓|68.09|12.59|14.20|33.71|
|✓|✓|**74.80**|**13.68**|**15.40**|**36.25**|



**ing.** We have made similar study on the effectiveness of treating “answer and localize” instructions and additional visual prompts as auxiliary tasks for 3D-QA on ScanQA [2] validation set in Tab. 6. The listed methods are evaluated without any visual interactions for fair comparison. Results show that the additional textual instructions and visual prompts improve the task diversity and further improve the performance on 3D Question Answering. 

**Performance as a Generalist.** To examine whether our method could distinguish different tasks given the textual instructions and visual prompts introduced in Sec. 4.1, we evaluate our model on different tasks with the same set of weights in Tab. 5 with several baseline methods. The first three rows list the performance our proposed models trained on each dataset from scratch as generalists. The following three rows list the performance of the model fine-tuned on each dataset. The last row lists the performance of our model as a generalist. Results show that our model could distinguish 3D-DC and 3D-QA given the textual instructions and visual prompts, providing strong results (62.98% C@0.5 on ScanRefer, 75.67% CiDEr on ScanQA). However, the generalist model achieves poor results on Nr3D [1], which is because we did not try to differentiate between Nr3D and ScanRefer during training as these two datasets are used for the same task in the first place. There is also an interesting observation that although we did not differentiate between the two datasets for 3D-DC, and the training 

sample sizes of the two models were similar, the model still tend to achieve high scores on ScanRefer (62.98% C@0.5). We are also excited to see that the weights of the generalist model could serve as a strong initialization of weights for fine-tuning. For example, the fine-tuned model on ScanRefer could achieve 65.19% C@0.5, which is +2.35% higher than the model trained from scratch. 

**Importance of Textual Instructions.** We further conduct study to see whether the instructions are necessary in 3D Dense Captioning in Tab. 7. The first row is our baseline method that directly generates the captions based on visual prompts without any textual instructions, and the second row is our method that is trained with textual instructions introduced in Sec. 4.1. Both methods are trained from scratch for fair comparison. We notice that since the LLM is frozen, certain textual instructions are beneficial to generating results in specific domains/tasks. 

Table 7. **Effectiveness of Instructions on 3D Dense Captioning.** We perform experiments on ScanRefer[6]. The baseline method directly generates the captions given the input 3D scene and visual prompts without any textual instructions. 

|Instructions|C@0.5_↑_|B-4@0.5_↑_|M@0.5_↑_|R@0.5_↑_|
|---|---|---|---|---|
|-|60.20|34.79|25.40|54.03|
|✓|**62.84**|**35.81**|**25.81**|**54.45**|



**Clicks for Better Question Answering.** One major challenge of answering questions in complex 3D environments is the vague identification of objects with plain texts. Therefore, we try to click on some of the related objects along with the textual instructions during evaluation, and see how it could affect the generated answers on the ScanQA validation set in Tab. 8. Results show that this technique would remove the ambiguities, and further improve the quality of the answers (+6.12% C). This illustrates the importance of visual interaction in complex 3D environments. 

7 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/cvpr_2024_LL3DA_Visual_Interactive_Instruction_Tuning_for_Omni-3D_Understanding_Reasoning_and_Planning_images/cvpr_2024_LL3DA_Visual_Interactive_Instruction_Tuning_for_Omni-3D_Understanding_Reasoning_and_Planning.pdf-0008-00.png)


**----- Start of picture text -----**<br>
3D Dense Captioning 3D Embodied Planning<br>Describe this object in the<br>3D scene.<br>This is a brown armchair. It is<br>to the right of the couch.<br>SpaCap3D : This is a chair with no arms.<br>It is next to a desk.<br>Vote2Cap-DETR : This is a brown armchair.<br>It is to the right of the table.<br>Ground Truth : The armchair is facing the<br>desk. The armchair is located to the right<br>of the couch, in the middle of the room. I want to clean the kitchen. What<br>should I do?<br>3D Question Answering<br>1. pick up any trash off the floor. 2.<br>Open the kitchen shelf. 3. Clean the<br>How many chairs are behind  kitchen sink with a cleaning solution.<br>the long table? 4. Clean the kitchen sink with a paper<br>towel.<br>2 chairs<br>I want to clean the kitchen. I have done<br>+ Visual Prompts these things: 1. Turn on the lamp on the<br>table. 2. Open the kitchen cabinet and<br>take out cleaning supplies. 3. Clean the<br>countertops with a cloth. 4. Clean the<br>How many chairs are behind  stove and microwave. What should I do<br>CLICK the long table? [Clicks] next?<br>3 chairs are behind long<br>table Clean the refrigerator and cabinets<br>3D Scene Description<br>Describe this 3D scene.<br>This room is a kitchen with a stove, stovetop, cabinets, refrigerator,<br>oven, cooler, recycling bins, microwave, water heater, soap dish, trash can,<br>a sink, and a refrigerator. There are multiple walls scattered in the room,<br>creating a separate space for dining and cooking. The door leads to the<br>kitchen. Another door and several windows hang around the area, providing a<br>large view of the floor. As well as additional doors, walls, windows,<br>doorstopper caps, kitchen items, sink and fridge, the food preparation and<br>storage areas are visible.<br>**----- End of picture text -----**<br>


Figure 4. **Qualitative Results.** We provide several visualization results on various 3D vision and language tasks in diverse 3D environments (living room, classroom, kitchen, and bedroom). Red highlights the wrong answer. 

Table 8. **Test Time Visual Interactions for Question Answering on ScanQA[2].** The model achieves better performance on the question answering when we add visual prompts to some of the related objects along with the text instructions during evaluation. 

|Visual Prompts|CiDEr_↑_|BLEU-4_↑_|METEOR_↑_|Rouge-L_↑_|
|---|---|---|---|---|
|-|76.79|**13.53**|15.88|37.31|
|✓|**82.91**|11.80|**16.74**|**39.97**|



## **5.4. Qualitative Results** 

We present several visualization results on different tasks in Fig. 4 to show our model’s capacities in understanding, rea- 

soning, and planning in different 3D environments. To prevent repetition when generating long sequences, we combine the top- _k_ [21] and top- _p_ [24] sampling strategy with _k_ = 50 and _p_ = 0 _._ 95. 

## **6. Conclusions** 

In this paper, we present LL3DA, a large language 3D assistant that could take both textual- and visual- interactions from human for understanding, reasoning, and planning in complex 3D environments. Our model directly encodes 3D point cloud for scene representations, and aggregates information from scenes and human interactions with the atten- 

8 

tion mechanism. The visual interactions could remove the ambiguities in cluttered 3D environments, showing mighty instruction-following capacities. Experimental results show that our method could achieve remarkable results on various 3D vision-language benchmarks. We hope that our approach could inspire further designs and training strategies for large 3D language models. In future studies, we anticipate that the availability of high-quality and diverse 3D vision and language annotations will enhance the model’s reasoning and planning capabilities. 

## **References** 

- [1] Panos Achlioptas, Ahmed Abdelreheem, Fei Xia, Mohamed Elhoseiny, and Leonidas Guibas. Referit3d: Neural listeners for fine-grained 3d object identification in real-world scenes. In _European Conference on Computer Vision_ , pages 422– 440. Springer, 2020. 2, 5, 6, 7, 19 

- [2] Daichi Azuma, Taiki Miyanishi, Shuhei Kurita, and Motoaki Kawanabe. Scanqa: 3d question answering for spatial scene understanding. In _proceedings of the IEEE/CVF conference on computer vision and pattern recognition_ , pages 19129– 19139, 2022. 2, 4, 5, 6, 7, 8, 19 

- [3] Satanjeev Banerjee and Alon Lavie. Meteor: An automatic metric for mt evaluation with improved correlation with human judgments. In _Proceedings of the acl workshop on intrinsic and extrinsic evaluation measures for machine translation and/or summarization_ , pages 65–72, 2005. 5, 15 

- [4] Daigang Cai, Lichen Zhao, Jing Zhang, Lu Sheng, and Dong Xu. 3djcg: A unified framework for joint dense captioning and visual grounding on 3d point clouds. In _Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition_ , pages 16464–16473, 2022. 2, 5, 6 

- [5] Chi Chen, Ruoyu Qin, Fuwen Luo, Xiaoyue Mi, Peng Li, Maosong Sun, and Yang Liu. Position-enhanced visual instruction tuning for multimodal large language models. _arXiv preprint arXiv:2308.13437_ , 2023. 2 

- [6] Dave Zhenyu Chen, Angel X Chang, and Matthias Nießner. Scanrefer: 3d object localization in rgb-d scans using natural language. In _European Conference on Computer Vision_ , pages 202–221. Springer, 2020. 2, 5, 6, 7, 19 

- [7] Dave Zhenyu Chen, Qirui Wu, Matthias Nießner, and Angel X Chang. D3net: A speaker-listener architecture for semi-supervised dense captioning and visual grounding in rgb-d scans. _arXiv preprint arXiv:2112.01551_ , 2021. 6 

- [8] Long Chen, Oleg Sinavski, Jan H¨unermann, Alice Karnsund, Andrew James Willmott, Danny Birch, Daniel Maund, and Jamie Shotton. Driving with llms: Fusing object-level vector modality for explainable autonomous driving. _arXiv preprint arXiv:2310.01957_ , 2023. 2 

- [9] Sijin Chen, Hongyuan Zhu, Xin Chen, Yinjie Lei, Gang Yu, and Tao Chen. End-to-end 3d dense captioning with vote2cap-detr. In _Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition_ , pages 11124– 11133, 2023. 2, 3, 4, 5, 6, 17, 18 

- [10] Sijin Chen, Hongyuan Zhu, Mingsheng Li, Xin Chen, Peng Guo, Yinjie Lei, Gang Yu, Taihao Li, and Tao 

   - Chen. Vote2cap-detr++: Decoupling localization and describing for end-to-end 3d dense captioning. _arXiv preprint arXiv:2309.02999_ , 2023. 2 

- [11] Zhenyu Chen, Ali Gholami, Matthias Nießner, and Angel X Chang. Scan2cap: Context-aware dense captioning in rgb-d scans. In _Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition_ , pages 3193–3203, 2021. 2, 5, 6, 17 

- [12] Zhenyu Chen, Ronghang Hu, Xinlei Chen, Matthias Nießner, and Angel X Chang. Unit3d: A unified transformer for 3d dense captioning and visual grounding. In _Proceedings of the IEEE/CVF International Conference on Computer Vision_ , pages 18109–18119, 2023. 2, 5, 6 

- [13] Hyung Won Chung, Le Hou, Shayne Longpre, Barret Zoph, Yi Tay, William Fedus, Eric Li, Xuezhi Wang, Mostafa Dehghani, Siddhartha Brahma, et al. Scaling instruction-finetuned language models. _arXiv preprint arXiv:2210.11416_ , 2022. 1 

- [14] Hyung Won Chung, Le Hou, Shayne Longpre, Barret Zoph, Yi Tay, William Fedus, Yunxuan Li, Xuezhi Wang, Mostafa Dehghani, Siddhartha Brahma, et al. Scaling instruction-finetuned language models. _arXiv preprint arXiv:2210.11416_ , 2022. 2 

- [15] Angela Dai, Angel X Chang, Manolis Savva, Maciej Halber, Thomas Funkhouser, and Matthias Nießner. Scannet: Richly-annotated 3d reconstructions of indoor scenes. In _Proceedings of the IEEE conference on computer vision and pattern recognition_ , pages 5828–5839, 2017. 5, 17, 19 

- [16] Wenliang Dai, Junnan Li, Dongxu Li, Anthony Meng Huat Tiong, Junqi Zhao, Weisheng Wang, Boyang Li, Pascale Fung, and Steven Hoi. Instructblip: Towards generalpurpose vision-language models with instruction tuning, 2023. 1, 2, 4 

- [17] Alexandros Delitzas, Maria Parelli, Nikolas Hars, Georgios Vlassis, Sotirios Anagnostidis, Gregor Bachmann, and Thomas Hofmann. Multi-clip: Contrastive vision-language pre-training for question answering tasks in 3d scenes. _arXiv preprint arXiv:2306.02329_ , 2023. 2, 6 

- [18] Jacob Devlin, Ming-Wei Chang, Kenton Lee, and Kristina Toutanova. Bert: Pre-training of deep bidirectional transformers for language understanding. _arXiv preprint arXiv:1810.04805_ , 2018. 4, 20 

- [19] Runyu Ding, Jihan Yang, Chuhui Xue, Wenqing Zhang, Song Bai, and Xiaojuan Qi. Pla: Language-driven openvocabulary 3d scene understanding. In _Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition_ , pages 7010–7019, 2023. 2 

- [20] Danny Driess, Fei Xia, Mehdi SM Sajjadi, Corey Lynch, Aakanksha Chowdhery, Brian Ichter, Ayzaan Wahid, Jonathan Tompson, Quan Vuong, Tianhe Yu, et al. Palme: An embodied multimodal language model. _arXiv preprint arXiv:2303.03378_ , 2023. 2 

- [21] Angela Fan, Mike Lewis, and Yann Dauphin. Hierarchical neural story generation. _arXiv preprint arXiv:1805.04833_ , 2018. 8 

- [22] Daocheng Fu, Xin Li, Licheng Wen, Min Dou, Pinlong Cai, Botian Shi, and Yu Qiao. Drive like a human: Rethink- 

9 

ing autonomous driving with large language models. _arXiv preprint arXiv:2307.07162_ , 2023. 2 

- [23] Ziyu Guo, Renrui Zhang, Xiangyang Zhu, Yiwen Tang, Xianzheng Ma, Jiaming Han, Kexin Chen, Peng Gao, Xianzhi Li, Hongsheng Li, et al. Point-bind & point-llm: Aligning point cloud with multi-modality for 3d understanding, generation, and instruction following. _arXiv preprint arXiv:2309.00615_ , 2023. 2, 3 

- [24] Ari Holtzman, Jan Buys, Li Du, Maxwell Forbes, and Yejin Choi. The curious case of neural text degeneration. _arXiv preprint arXiv:1904.09751_ , 2019. 8 

- [25] Yining Hong, Chunru Lin, Yilun Du, Zhenfang Chen, Joshua B Tenenbaum, and Chuang Gan. 3d concept learning and reasoning from multi-view images. In _Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition_ , pages 9202–9212, 2023. 2 

- [26] Yining Hong, Haoyu Zhen, Peihao Chen, Shuhong Zheng, Yilun Du, Zhenfang Chen, and Chuang Gan. 3d-llm: Injecting the 3d world into large language models. _arXiv preprint arXiv:2307.12981_ , 2023. 2, 5, 6, 15, 16, 19 

- [27] Srinivasan Iyer, Xi Victoria Lin, Ramakanth Pasunuru, Todor Mihaylov, Daniel Simig, Ping Yu, Kurt Shuster, Tianlu Wang, Qing Liu, Punit Singh Koura, et al. Opt-iml: Scaling language model instruction meta learning through the lens of generalization. _arXiv preprint arXiv:2212.12017_ , 2022. 1 

- [28] Biao Jiang, Xin Chen, Wen Liu, Jingyi Yu, Gang Yu, and Tao Chen. Motiongpt: Human motion as a foreign language. _arXiv preprint arXiv:2306.14795_ , 2023. 1 

- [29] Yang Jiao, Shaoxiang Chen, Zequn Jie, Jingjing Chen, Lin Ma, and Yu-Gang Jiang. More: Multi-order relation mining for dense captioning in 3d scenes. _arXiv preprint arXiv:2203.05203_ , 2022. 6 

- [30] Zhao Jin, Munawar Hayat, Yuwei Yang, Yulan Guo, and Yinjie Lei. Context-aware alignment and mutual masking for 3dlanguage pre-training. In _Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition_ , pages 10984–10994, 2023. 2, 5, 6 

- [31] Alexander Kirillov, Eric Mintun, Nikhila Ravi, Hanzi Mao, Chloe Rolland, Laura Gustafson, Tete Xiao, Spencer Whitehead, Alexander C. Berg, Wan-Yen Lo, Piotr Doll´ar, and Ross Girshick. Segment anything. _arXiv:2304.02643_ , 2023. 4 

- [32] Junnan Li, Dongxu Li, Silvio Savarese, and Steven Hoi. Blip-2: Bootstrapping language-image pre-training with frozen image encoders and large language models. _arXiv preprint arXiv:2301.12597_ , 2023. 1, 2, 4 

- [33] Chin-Yew Lin. Rouge: A package for automatic evaluation of summaries. In _Text summarization branches out_ , pages 74–81, 2004. 5, 15 

- [34] Haotian Liu, Chunyuan Li, Qingyang Wu, and Yong Jae Lee. Visual instruction tuning. _arXiv preprint arXiv:2304.08485_ , 2023. 1, 2 

- [35] Ilya Loshchilov and Frank Hutter. Decoupled weight decay regularization. _arXiv preprint arXiv:1711.05101_ , 2017. 5 

- [36] Tiange Luo, Chris Rockwell, Honglak Lee, and Justin Johnson. Scalable 3d captioning with pretrained models. _arXiv preprint arXiv:2306.07279_ , 2023. 3 

- [37] Xiaojian Ma, Silong Yong, Zilong Zheng, Qing Li, Yitao Liang, Song-Chun Zhu, and Siyuan Huang. Sqa3d: Situated question answering in 3d scenes. _arXiv preprint arXiv:2210.07474_ , 2022. 2 

- [38] Aihua Mao, Zhi Yang, Wanxin Chen, Ran Yi, and Yong-jin Liu. Complete 3d relationships extraction modality alignment network for 3d dense captioning. _IEEE Transactions on Visualization and Computer Graphics_ , 2023. 6 

- [39] Ishan Misra, Rohit Girdhar, and Armand Joulin. An end-toend transformer model for 3d object detection. In _Proceedings of the IEEE/CVF International Conference on Computer Vision_ , pages 2906–2917, 2021. 20 

- [40] Ron Mokady, Amir Hertz, and Amit H Bermano. Clipcap: Clip prefix for image captioning. _arXiv preprint arXiv:2111.09734_ , 2021. 5 

- [41] OpenAI. Gpt-4 technical report, 2023. 1 

- [42] Kishore Papineni, Salim Roukos, Todd Ward, and Wei-Jing Zhu. Bleu: a method for automatic evaluation of machine translation. In _Proceedings of the 40th annual meeting of the Association for Computational Linguistics_ , pages 311–318, 2002. 5, 15 

- [43] Maria Parelli, Alexandros Delitzas, Nikolas Hars, Georgios Vlassis, Sotirios Anagnostidis, Gregor Bachmann, and Thomas Hofmann. Clip-guided vision-language pre-training for question answering in 3d scenes. In _Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition_ , pages 5606–5611, 2023. 2, 4, 6 

- [44] Charles Ruizhongtai Qi, Li Yi, Hao Su, and Leonidas J Guibas. Pointnet++: Deep hierarchical feature learning on point sets in a metric space. _Advances in neural information processing systems_ , 30, 2017. 20 

- [45] Charles R Qi, Or Litany, Kaiming He, and Leonidas J Guibas. Deep hough voting for 3d object detection in point clouds. In _proceedings of the IEEE/CVF International Conference on Computer Vision_ , pages 9277–9286, 2019. 17 

- [46] David Rozenberszki, Or Litany, and Angela Dai. Languagegrounded indoor 3d semantic segmentation in the wild. In _Proceedings of the European Conference on Computer Vision (ECCV)_ , 2022. 17 

- [47] Chan Hee Song, Jiaman Wu, Clayton Washington, Brian M Sadler, Wei-Lun Chao, and Yu Su. Llm-planner: Few-shot grounded planning for embodied agents with large language models. In _Proceedings of the IEEE/CVF International Conference on Computer Vision_ , pages 2998–3009, 2023. 2 

- [48] Matthew Tancik, Pratul Srinivasan, Ben Mildenhall, Sara Fridovich-Keil, Nithin Raghavan, Utkarsh Singhal, Ravi Ramamoorthi, Jonathan Barron, and Ren Ng. Fourier features let networks learn high frequency functions in low dimensional domains. _Advances in Neural Information Processing Systems_ , 33:7537–7547, 2020. 4 

- [49] Hugo Touvron, Louis Martin, Kevin Stone, Peter Albert, Amjad Almahairi, Yasmine Babaei, Nikolay Bashlykov, Soumya Batra, Prajjwal Bhargava, Shruti Bhosale, et al. Llama 2: Open foundation and fine-tuned chat models. _arXiv preprint arXiv:2307.09288_ , 2023. 1, 3, 4, 16 

- [50] Ramakrishna Vedantam, C Lawrence Zitnick, and Devi Parikh. Cider: Consensus-based image description evaluation. In _Proceedings of the IEEE conference on computer_ 

10 

_vision and pattern recognition_ , pages 4566–4575, 2015. 5, 15 

- [51] Heng Wang, Chaoyi Zhang, Jianhui Yu, and Weidong Cai. Spatiality-guided transformer for 3d dense captioning on point clouds. _arXiv preprint arXiv:2204.10688_ , 2022. 2, 6 

   - [65] Ziyu Zhu, Xiaojian Ma, Yixin Chen, Zhidong Deng, Siyuan Huang, and Qing Li. 3d-vista: Pre-trained transformer for 3d vision and text alignment. In _Proceedings of the IEEE/CVF International Conference on Computer Vision_ , pages 2911– 2921, 2023. 2, 5, 6 

- [52] Yanmin Wu, Xinhua Cheng, Renrui Zhang, Zesen Cheng, and Jian Zhang. Eda: Explicit text-decoupling and dense alignment for 3d visual grounding. In _Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition_ , pages 19231–19242, 2023. 2 

- [53] Peng Xu, Xiatian Zhu, and David A Clifton. Multimodal learning with transformers: A survey. _IEEE Transactions on Pattern Analysis and Machine Intelligence_ , 2023. 2 

- [54] Runsen Xu, Xiaolong Wang, Tai Wang, Yilun Chen, Jiangmiao Pang, and Dahua Lin. Pointllm: Empowering large language models to understand point clouds. _arXiv preprint arXiv:2308.16911_ , 2023. 1, 2, 3 

- [55] Qinghao Ye, Haiyang Xu, Guohai Xu, Jiabo Ye, Ming Yan, Yiyang Zhou, Junyang Wang, Anwen Hu, Pengcheng Shi, Yaya Shi, et al. mplug-owl: Modularization empowers large language models with multimodality. _arXiv preprint arXiv:2304.14178_ , 2023. 2 

- [56] Shuquan Ye, Dongdong Chen, Songfang Han, and Jing Liao. 3d question answering. _IEEE Transactions on Visualization and Computer Graphics_ , 2022. 2 

- [57] Shukang Yin, Chaoyou Fu, Sirui Zhao, Ke Li, Xing Sun, Tong Xu, and Enhong Chen. A survey on multimodal large language models. _arXiv preprint arXiv:2306.13549_ , 2023. 2 

- [58] Susan Zhang, Stephen Roller, Naman Goyal, Mikel Artetxe, Moya Chen, Shuohui Chen, Christopher Dewan, Mona Diab, Xian Li, Xi Victoria Lin, et al. Opt: Open pre-trained transformer language models. _arXiv preprint arXiv:2205.01068_ , 2022. 1, 2, 3, 4, 5, 16 

- [59] Shilong Zhang, Peize Sun, Shoufa Chen, Min Xiao, Wenqi Shao, Wenwei Zhang, Kai Chen, and Ping Luo. Gpt4roi: Instruction tuning large language model on region-of-interest. _arXiv preprint arXiv:2307.03601_ , 2023. 1, 2 

- [60] Lichen Zhao, Daigang Cai, Lu Sheng, and Dong Xu. 3dvgtransformer: Relation modeling for visual grounding on point clouds. In _Proceedings of the IEEE/CVF International Conference on Computer Vision_ , pages 2928–2937, 2021. 2 

- [61] Lichen Zhao, Daigang Cai, Jing Zhang, Lu Sheng, Dong Xu, Rui Zheng, Yinjie Zhao, Lipeng Wang, and Xibo Fan. Towards explainable 3d grounded visual question answering: A new benchmark and strong baseline. _IEEE Transactions on Circuits and Systems for Video Technology_ , 2022. 2 

- [62] Yufeng Zhong, Long Xu, Jiebo Luo, and Lin Ma. Contextual modeling for 3d dense captioning on point clouds. _arXiv preprint arXiv:2210.03925_ , 2022. 6 

- [63] Qiang Zhou, Chaohui Yu, Shaofeng Zhang, Sitong Wu, Zhibing Wang, and Fan Wang. Regionblip: A unified multimodal pre-training framework for holistic and regional comprehension, 2023. 3 

- [64] Deyao Zhu, Jun Chen, Xiaoqian Shen, Xiang Li, and Mohamed Elhoseiny. Minigpt-4: Enhancing vision-language understanding with advanced large language models. _arXiv preprint arXiv:2304.10592_ , 2023. 2 

11 

## **Supplementary Material** 

The supplementary material consists of more visualization results (Sec. A), quantitative evaluations on scene descriptions, embodied dialogue, and embodied planning (Sec. B), additional studies on 3D coordinates generated by LL3DA (Sec. C), data statistics as well as training samples (Sec. D), additional implementation details (Sec. E). 

**Codes and Pre-trained weights** will be fully released. We provide example code files along with this file, including the training and evaluation process for different tasks. 

**Video.** We also provide video clips for more illustrative demonstrations. 

## **A. More Visualizations** 

We present additional visualization results on Scene Descriptions (Fig. 5), 3D Dense Captioning, 3D Question Answering, Embodied Dialogue (Fig. 6), and Embodied Planning (Fig. 7). 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/cvpr_2024_LL3DA_Visual_Interactive_Instruction_Tuning_for_Omni-3D_Understanding_Reasoning_and_Planning_images/cvpr_2024_LL3DA_Visual_Interactive_Instruction_Tuning_for_Omni-3D_Understanding_Reasoning_and_Planning.pdf-0012-06.png)


**----- Start of picture text -----**<br>
Chair 3<br>Table 2<br>Chair 2<br>Laptop<br>Television Chair 4<br>Chair 1 Table 1<br>Describe this scene.<br>Describe this scene.<br>The room is a cozy space that accommodates various activities<br>and features. Throughout the room, there is a door and other<br>objects, providing convenient entrances and exits. Various  The room is spacious with walls surrounding its entire<br>objects are present, including  a laptop, a microwave, a lamp, a  boundaries. There are a couple of armchairs scattered<br>coffee mug, a trash can, and a coffee maker . Some of the objects  within the room, with two in different areas. There are<br>are located on shelves or on windowsills, adding to the room’s  four chairs in the room, along with two tables and two<br>decoration. The room’s walls are well-laid out, creating a  shelves. There are two chairs in different areas, with<br>bright and organized setting. Several objects are scattered  two near each other. Additionally, there is a doorframe,<br>throughout, creating a sense of excitement or adventure. A  tv a door, and a coffee table in the room. Further, there<br>hangs from a wall , providing entertainment and a view. A coffee  is another chair, a table, and a cabinet. There are<br>maker is placed at a table, providing a convenient and  several books scattered around the room, including a<br>convenient beverage option. Another type of structure is a wall.  book, a cup, a magazine, a magazine box, and a paper<br>A wall is a common feature in some arrangements of homes,  towel. There are also three bags in the room with one<br>businesses, or institutions.  being a tabletop .<br>**----- End of picture text -----**<br>


Figure 5. **Qualitative Results on Scene Descriptions.** We highlight some of the phrases in the generated scene descriptions mentioning the instances in the 3D environment. 

12 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/cvpr_2024_LL3DA_Visual_Interactive_Instruction_Tuning_for_Omni-3D_Understanding_Reasoning_and_Planning_images/cvpr_2024_LL3DA_Visual_Interactive_Instruction_Tuning_for_Omni-3D_Understanding_Reasoning_and_Planning.pdf-0013-00.png)


**----- Start of picture text -----**<br>
Describe this object in the 3D  Describe this object in the 3D  Describe this object in the 3D<br>scene. scene. scene.<br>This is a brown armchair. It is to  This is a brown bookshelf. It is to  This is a white sink. It is to the<br>the right of another brown armchair. the right of the door. left of the chair.<br>SpaCap3D: This is a brown leather chair. It  SpaCap3D: The bookshelf is to the left of  SpaCap3D: This is a black music stand. It<br>is next to a bookshelf. the door. It is to the left of the door. is to the left of a table.<br>Vote2Cap-DETR: The chair is to the left of  Vote2Cap-DETR: This is a clear plastic bag.  Vote2Cap-DETR: This is a white sink. It is<br>the table. It is to the right of the chair. It is to the left of the door. to the right of the room.<br>Ground Truth: This is a brown armchair. It  Ground Truth: On the right you can see the  Ground Truth: There is a water fountain on<br>is back-to-back with another armchair and is  edge of a tall bookcase. It appears to be  the wall. It is to the left of the other<br>facing a table. open-backed like a library shelf. water fountain.<br>What color is the chair on the  What is in the corner by the<br>right of the chalk board? whiteboard?<br>black chair<br>+ Visual Prompts + Visual Prompts<br>What color is the chair on the<br>CLICK right of the chalk board? CLICK What is in the corner by the<br>[Click] CLICK whiteboard? [Click]<br>brown office chair and table<br>Wow, this room seems quite<br>organized. What do you<br>notice about it?<br>Wow, this room has quite a few objects! I see some  Yes, it does. I see a desk<br>walls, desks, chairs, and even a blackboard. in the center of the room.<br>Yes, it seems like it. I see the walls are very  Oh, I see. And there are two doors in this room.<br>well placed. They keep the room tidy and separated. One of them is close to the window, and the<br>other one is on the opposite side of the room.<br>Yes, I see that too. There are a total of eight<br>chairs and twenty-three seats. Yes, they are located at different spots. One of<br>them is located on the right side of the room, and<br>That's quite a lot of seating options. I also noticed a  the other one is closer to the window.<br>few desks, one of which is placed near the blackboard.<br>Yes, it definitely gives the room a better flow. I also<br>Absolutely. It looks like the chairs are placed around  see some walls outlining the perimeter of the room. They<br>the desks and the seats are scattered throughout the room. seem to be evenly distributed.<br>Definitely. This room must be versatile. It's  Yep, this room is quite open and flexible. It makes<br>perfect for both students and professors. for a more flexible and comfortable work space.<br>**----- End of picture text -----**<br>


Figure 6. **More Qualitative Results on 3D Dense Captioning (upper), 3D Question Answering (middle), and Embodied Dialogue (lower).** Red highlights the wrong answer. 

13 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/cvpr_2024_LL3DA_Visual_Interactive_Instruction_Tuning_for_Omni-3D_Understanding_Reasoning_and_Planning_images/cvpr_2024_LL3DA_Visual_Interactive_Instruction_Tuning_for_Omni-3D_Understanding_Reasoning_and_Planning.pdf-0014-00.png)


**----- Start of picture text -----**<br>
I want to organize workspace. What should I do? I want to decorate the living room. What should I do?<br>1. arrange work chairs around a table  1. place the cabinet on the floor<br>2. place a monitor on the table  2. place the table at one end of the room<br>3. place a keyboard and mouse near the monitor 3. arrange the chairs around the table<br>4. put the file cabinet next to the table 4. hang the picture on the wall to the left of the table<br>5. connect the monitor to the computer via the cable 5. hang the other picture on the other wall to the right<br>6. place a recycling bin near the table of the table<br>7. adjust the table until it is comfortable for the user 6. hang the fourth picture on the same wall to the other<br>8. arrange the objects on the table such as files or cups side of the table<br>9. organize the office tools and equipment in the cabinet  7. hang the 5th picture on the floor to the right<br>on the table 8. ensure the room is comfortable and clean<br>I want to organize workspace. I have done these things:<br>1. sit on the chair at the desk  I want to decorate the living room. I have done these things:<br>2.3. place the laptop on the deskarrange the backpack next to the chair.  What should I do next?1.2. place the couch against the wallposition the coffee table in front of the couch.<br>What should I do next?<br>position the monitor on the desk place the end table in the corner<br>I want to set up a workstation. What should I do?" I want to take a shower. What should I do?<br>1. go to the desk  1. close the bathroom door<br>2. sit at the desk  2. turn on the shower<br>3. place the monitor on the desk  3. open the shower curtain<br>4. set up the mouse and keyboard 4. adjust the temperature<br>5. position the telephone next to the desk  5. wash body with soap<br>6. connect the laptop to the computer  6. turn off the shower<br>7. connect the power cable to the laptop  7. dry body with towel<br>8. place the office chair near the desk  8. close the shower curtain<br>9. put the backpack on the desk<br>10.organize the desk area with files and folders"<br>I want to take a shower. I have done these things:<br>I want to set up a workstation. I have done these things:  1. enter the bathroom through the door<br>1. place the coffee table in the desired location on the  2. close the door behind you<br>floor. 3. go to the bathtub<br>2. position the keyboard piano on top of the coffee table 4. draw the shower curtain.<br>What should I do next? what should I do next?":<br>place the chair near the coffee table for seating. turn on the water<br>**----- End of picture text -----**<br>


Figure 7. **Qualitative Results on Embodied Planning.** The planning results generated by LL3DA are consistent with common sense knowledge. Additionally, LL3DA could not only make plans directly, but also generate feedbacks based on the things have been done. 

14 

## **B. More Evaluations** 

Since 3D-LLM [26] has not yet released the validation set they used for evaluation, we make our own split to quantify our method’s performance on: 1) scene descriptions, 2) embodied dialogue, and 3) embodied planning. 

**Data Splits.** In practice, we set the scenes with ids less than 600 as the training set, and the rest as the validation set. The data statistics of our split are listed in Tab. 9. It is worth mentioning that: 

- The multi-turn embodied dialogues are further decomposed into 8,490 and 1,222 lines of data samples following the procedure in Fig. 8 for training and validation. 

- The embodied planning data is also decomposed into 16,972 and 2,282 lines of data samples for training and validation following the procedure in Fig. 9. 

As mentioned in Sec. 4.1 in the main paper, we distinguishes the source of texts with the “### human:” and “### assistant:” identifier. 

**Metrics.** Similar to the main paper, we evaluate the generated natural language responses under the CiDEr [50], BLEU [42], METEOR [3], and Rouge-L [33] metrics in Tab. 10. 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/cvpr_2024_LL3DA_Visual_Interactive_Instruction_Tuning_for_Omni-3D_Understanding_Reasoning_and_Planning_images/cvpr_2024_LL3DA_Visual_Interactive_Instruction_Tuning_for_Omni-3D_Understanding_Reasoning_and_Planning.pdf-0015-07.png)


**----- Start of picture text -----**<br>
Training Sample 1<br>Can you hand me the remote?  Input: "### human: Can you hand me the remote? ### assistant:",<br>Multi-Turn<br>Sure! Where is it? Dialogue Output: "Sure! Where is it?"<br>Decomposition<br>It's on the ledge in the bathroom.  Training Sample 2<br>Input: "### human: Can you hand me the remote? ### assistant: Sure!<br>Alright. I have retrieved the remote  Where is it? ### human: It's on the ledge in the bathroom. ### assistant:",<br>from the ledge. Here you go. Is there  Output: "Alright. I have retrieved the remote from the ledge. Here you<br>anything else you need?<br>go. Is there anything else you need?"<br>**----- End of picture text -----**<br>


Figure 8. **The Multi-Turn Dialogue Decomposition Pipeline.** For one sample of a _n_ -turn dialogue data ( _n ≥_ 1), we decompose it into _n_ lines of training samples, and distinguish the source of data with the “### human:” and “### assistant:” identifier. 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/cvpr_2024_LL3DA_Visual_Interactive_Instruction_Tuning_for_Omni-3D_Understanding_Reasoning_and_Planning_images/cvpr_2024_LL3DA_Visual_Interactive_Instruction_Tuning_for_Omni-3D_Understanding_Reasoning_and_Planning.pdf-0015-09.png)


**----- Start of picture text -----**<br>
Training Sample 𝑘 (𝑘= 1 ⋯𝑛)<br>Input: "### human: I want to organize the office supplies. I have done<br>I want to organize the office supplies. I  these things:<br>have done these things:  1 .  xxxx<br>1 …  .  xxxx Embodied …  𝑘. xxxx.<br>𝑛. xxxx.  Planning what should I do next? ### assistant:",<br>what should I do next? Decomposition Output: {the  𝑘+ 1 -th issue}<br>take out the boxes Training Sample 𝑛+ 1<br>Input: "### human: I want to organize the office supplies. What should<br>I do? ### assistant:",<br>Output: " 1 .  xxxx<br>…<br>𝑛. xxxx<br>𝑛+ 1. Take out the boxes."<br>**----- End of picture text -----**<br>


Figure 9. **The Embodied Planning Decomposition Pipeline.** For one sample with given _n_ -tasks ( _n ≥_ 1), we decompose it into ( _n_ + 1) lines of training and validation samples. 

Table 9. **Details about the Training-Validation Split of the ScanNet Part of 3D-LLM [26].** We list the number of natural language annotations (“# annotations”) and number of scenes (“# scenes”) in each split for each task. 

|Split|Scene Description<br># annotations<br># scenes|Embodied Planning<br># annotations<br># scenes|Embodied Dialogue|
|---|---|---|---|
||||# annotations<br># scenes|
|Train<br>Validation|912<br>456<br>121<br>61|1,636<br>449<br>228<br>61|2,592<br>456<br>363<br>61|
|Total|1,033<br>517|1,864<br>510|2,955<br>517|



15 

Table 10. **Quantitative Comparisons on Scene Description, Embodied Dialogue and Embodied Planning.** We evaluate our method with different sampling strategies on the ScanNet part of 3D-LLM [26]. We manually set the scenes with ids less than 600 as the training set, and the rest as the validation set. To prevent repetition, we adopt the _n_ -gram repetition penalty so that no _n_ -gram appears twice ( _n_ = 4). 

|Task|Method<br>BLEU-1_↑_<br>BLEU-2_↑_<br>BLEU-3_↑_<br>BLEU-4_↑_<br>CiDEr_↑_<br>METEOR_↑_<br>Rouge-L_↑_|
|---|---|
|Scene<br>Description|**_Zero-Shot:_**<br>OPT-1.3B [58]<br>15.79<br>6.10<br>2.07<br>0.84<br>0.00<br>8.40<br>11.70<br>OPT-2.7B [58]<br>19.97<br>7.59<br>2.14<br>0.00<br>0.11<br>6.60<br>12.32<br>OPT-6.7B [58]<br>24.40<br>9.79<br>3.62<br>1.13<br>0.06<br>8.99<br>16.96<br>LLAMA-7B[49]<br>19.26<br>7.69<br>2.79<br>0.92<br>0.20<br>7.00<br>12.31<br>**_Ours:_**<br>top-_k_& top-_p_<br>43.02<br>26.70<br>15.97<br>8.97<br>0.96<br>14.65<br>24.84<br>greedy decoding<br>29.15<br>20.51<br>13.99<br>9.38<br>1.44<br>12.83<br>24.62<br>beam search<br>29.94<br>21.56<br>14.93<br>10.02<br>1.32<br>12.31<br>27.08|
|Embodied<br>Dialogue|**_Zero-Shot:_**<br>OPT-1.3B [58]<br>2.44<br>1.05<br>0.46<br>0.23<br>0.31<br>5.62<br>4.83<br>OPT-2.7B [58]<br>3.88<br>1.56<br>0.73<br>0.39<br>0.38<br>7.38<br>6.28<br>OPT-6.7B [58]<br>3.59<br>1.65<br>0.81<br>0.43<br>0.25<br>6.88<br>6.16<br>LLAMA-7B[49]<br>4.08<br>1.80<br>0.90<br>0.50<br>0.27<br>7.81<br>6.68<br>**_Ours:_**<br>top-_k_& top-_p_<br>41.38<br>32.59<br>27.47<br>23.95<br>190.01<br>23.50<br>40.61<br>greedy decoding<br>48.08<br>39.59<br>34.37<br>30.70<br>251.78<br>27.01<br>47.57<br>beam search<br>48.14<br>39.83<br>34.83<br>31.32<br>260.07<br>27.21<br>47.69|
|Embodied<br>Planning|**_Zero-Shot:_**<br>OPT-1.3B [58]<br>1.26<br>0.59<br>0.26<br>0.13<br>0.16<br>0.24<br>3.56<br>OPT-2.7B [58]<br>2.02<br>0.99<br>0.49<br>0.26<br>0.10<br>3.59<br>4.35<br>OPT-6.7B [58]<br>2.03<br>1.06<br>0.53<br>0.28<br>0.00<br>3.65<br>3.94<br>LLAMA-7B[49]<br>2.24<br>1.13<br>0.55<br>0.29<br>0.04<br>3.53<br>4.71<br>**_Ours:_**<br>top-_k_& top-_p_<br>40.72<br>27.18<br>18.64<br>12.95<br>128.80<br>17.05<br>39.25<br>greedy decoding<br>38.81<br>27.58<br>20.10<br>14.66<br>186.13<br>19.60<br>45.34<br>beam search<br>45.07<br>33.04<br>24.96<br>19.15<br>196.78<br>19.87<br>45.58|



**Quantitative Comparisons.** As shown in Tab. 10, we perform zero-shot evaluation on the frozen LLMs [49, 58] with the top- _k_ and top- _p_ sampling strategy as the baseline, and evaluate our method with different sampling strategies. Among all the listed results, we adopt the _n_ -gram penalty that restricts no _n_ -gram appears twice with _n_ = 4. We also set _k_ = 50 and _p_ = 0 _._ 95 for top- _k_ and top- _p_ sampling, and a beam size = 4 for beam search. From the table, one could see that our method could generate high-quality responses, and out-performs language-only LLMs. 

16 

## **C. Additional Studies: Generating 3D Bounding Boxes with LLM** 

**Quantitative Evaluation on 3D Open-Vocabulary Detection with LLM-Generated Boxes.** We treat 3D Open-Vocabulary Detection as the 3D Dense Captioning problem, and simulate the click prompts using the spatial location of vote queries [9]. The textual instruction is set as “ _### human: what is this object? ### assistant:_ ”. To obtain localization results, we reconstruct 3D bounding boxes from the “ _<_ obj _>cx_ , _cy_ , _cz_ , _w_ , _h_ , _l<_ /obj _>_ ” in the generated texts, and also decode the object category prediction from the texts. In Tab. 13, we present performances under the ScanNet vocabulary [15], and the ScanNet200 vocabulary [46]. Per-class mAP results on ScanNet could be found in Tab. 12. Results show that LL3DA could produce high-quality bounding boxes comparable to 3D detectors under the threshold IoU=0.5. 

Table 11. **Quantitative Comparisons on Open-Vocabulary Detection with LL3DA.** We treat 3D Detection as the 3D Dense Captioning problem. We simulate the click prompt with the vote queries [9], and decode the “ _<_ obj _>cx_ , _cy_ , _cz_ , _w_ , _h_ , _l<_ /obj _>_ ” and category names from the generated texts. Results show that our method is able to produce comparable to the 3D detector specialist when IoU=0.5. 

|Method|ScanNet Vocabulary<br>IoU=0.25<br>IoU=0.5<br>mAP_↑_<br>AR_↑_<br>mAP_↑_<br>AR_↑_|ScanNet Vocabulary<br>IoU=0.25<br>IoU=0.5<br>mAP_↑_<br>AR_↑_<br>mAP_↑_<br>AR_↑_|ScanNet200 Vocabulary|ScanNet200 Vocabulary|
|---|---|---|---|---|
||IoU=0.25<br>mAP_↑_<br>AR_↑_||IoU=0.25<br>mAP_↑_<br>AR_↑_|IoU=0.5|
|||||mAP_↑_<br>AR_↑_|
|VoteNet [45]<br>Ours|57.17<br>81.18<br>48.94<br>65.15|31.50<br>50.08<br>32.48<br>49.46|-<br>-<br>7.40<br>12.10|-<br>-<br>5.20<br>9.04|



Table 12. **Per-class AP under IoU threshold of 0.25 and 0.5 on ScanNet validation set.** 

|IoU|bathtub<br>bed<br>bookshelf<br>cabinet<br>chair<br>counter<br>curtain<br>desk<br>door<br>others<br>picture<br>refrigerator<br>shower curtain<br>sink<br>sofa<br>table<br>toilet<br>window|
|---|---|
|0.25<br>0.50|83.33<br>62.16<br>10.63<br>43.13<br>80.83<br>55.63<br>40.20<br>42.75<br>37.17<br>27.03<br>17.11<br>24.96<br>62.92<br>56.72<br>64.30<br>52.80<br>87.73<br>31.46<br>68.33<br>53.24<br>6.50<br>19.81<br>72.15<br>23.21<br>23.55<br>25.81<br>18.32<br>13.54<br>6.13<br>18.00<br>35.42<br>18.02<br>51.02<br>45.72<br>76.93<br>8.98|



**Quantitative Evaluation on 3D Dense Captioning with LLM-Generated Boxes.** We evaluate the quality of the 3D bounding boxes generated by LL3DA on 3D Dense Captioning. Following existing works [11], we evaluate with the _m_ @ _k_ IoU metric, where _m_ could be one of CiDEr, BLEU-4, METEOR, Rouge-L: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/cvpr_2024_LL3DA_Visual_Interactive_Instruction_Tuning_for_Omni-3D_Understanding_Reasoning_and_Planning_images/cvpr_2024_LL3DA_Visual_Interactive_Instruction_Tuning_for_Omni-3D_Understanding_Reasoning_and_Planning.pdf-0017-07.png)


Here, _N_ is the total size of annotated instances, _c_ pred and _c_ gt are the predicted caption and the ground truth human annotations for this instance, _b_[3] pred _[D]_[and] _[ b]_[3] gt _[D]_ are the predicted 3D bounding box and the ground truth 3D box annotation, and I _{_ condition _}_ is the identification function that equals 1 if the condition meets, and 0 otherwise. 

Our baseline method is listed in Row 1 (Tab. 13). The baseline method is also the LL3DA model, but it only generates instance captions in response to the box prompts generated by the 3D detector [9] and the textual instruction “ _### human: describe this object in the given 3D scene. ### assistant:_ ”. The results listed in the second row of Tab. 13 come from reconstructing 3D bounding boxes from decoding the “ _<_ obj _>cx_ , _cy_ , _cz_ , _w_ , _h_ , _l<_ /obj _>_ ” in the generated texts. We use “ _### human: given the 3D scene, localize and describe this object. ### assistant:_ ” as the textual instruction. Though there is still gap between the bounding boxes generated by LL3DA and 3D localizer specialists, LL3DA could generate reasonable bounding box estimations. One can also refer to Fig. 10 for more visualization details. 

Table 13. **3D Dense Captioning Performance with 3D Bounding Boxes Generated by LL3DA.** Though there is still gap between LL3DA and 3D specialists for object localization, LL3DA could generate reasonable bounding box estimations. 

|Test Time<br>Localization|IoU = 0.25<br>C@0.25_↑_<br>B-4@0.25_↑_<br>M@0.25_↑_<br>R@0.25_↑_|IoU = 0.5|
|---|---|---|
|||C@0.5_↑_<br>B-4@0.5_↑_<br>M@0.5_↑_<br>R@0.5_↑_|
|3D Detector [9]<br>LL3DA|74.17<br>41.41<br>27.76<br>59.53<br>62.90<br>35.14<br>26.70<br>55.62|65.19<br>36.79<br>25.97<br>55.06<br>51.12<br>29.27<br>24.18<br>49.51|



**Qualitative Results of LL3DA’s Box Predictions.** We adopt the spatial location of vote queries to simulate the user click (256 queries per scene), and ask the model to generate 3D bounding boxes in forms of “ _<_ obj _>cx_ , _cy_ , _cz_ , _w_ , _h_ , _l<_ /obj _>_ ” as introduced in the main paper. Then, we reconstruct the 3D bounding boxes, and visualize them in in Fig. 10. The visualization results show that LL3DA could produce tight bounding boxes close to the objects in diverse and complex 3D environments. 

17 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/cvpr_2024_LL3DA_Visual_Interactive_Instruction_Tuning_for_Omni-3D_Understanding_Reasoning_and_Planning_images/cvpr_2024_LL3DA_Visual_Interactive_Instruction_Tuning_for_Omni-3D_Understanding_Reasoning_and_Planning.pdf-0018-00.png)


**----- Start of picture text -----**<br>
Input 3D Scene VoteNet LL3DA Ground Truth Annotation<br>**----- End of picture text -----**<br>



![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/cvpr_2024_LL3DA_Visual_Interactive_Instruction_Tuning_for_Omni-3D_Understanding_Reasoning_and_Planning_images/cvpr_2024_LL3DA_Visual_Interactive_Instruction_Tuning_for_Omni-3D_Understanding_Reasoning_and_Planning.pdf-0018-01.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/cvpr_2024_LL3DA_Visual_Interactive_Instruction_Tuning_for_Omni-3D_Understanding_Reasoning_and_Planning_images/cvpr_2024_LL3DA_Visual_Interactive_Instruction_Tuning_for_Omni-3D_Understanding_Reasoning_and_Planning.pdf-0018-02.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/cvpr_2024_LL3DA_Visual_Interactive_Instruction_Tuning_for_Omni-3D_Understanding_Reasoning_and_Planning_images/cvpr_2024_LL3DA_Visual_Interactive_Instruction_Tuning_for_Omni-3D_Understanding_Reasoning_and_Planning.pdf-0018-03.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/cvpr_2024_LL3DA_Visual_Interactive_Instruction_Tuning_for_Omni-3D_Understanding_Reasoning_and_Planning_images/cvpr_2024_LL3DA_Visual_Interactive_Instruction_Tuning_for_Omni-3D_Understanding_Reasoning_and_Planning.pdf-0018-04.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/cvpr_2024_LL3DA_Visual_Interactive_Instruction_Tuning_for_Omni-3D_Understanding_Reasoning_and_Planning_images/cvpr_2024_LL3DA_Visual_Interactive_Instruction_Tuning_for_Omni-3D_Understanding_Reasoning_and_Planning.pdf-0018-05.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/cvpr_2024_LL3DA_Visual_Interactive_Instruction_Tuning_for_Omni-3D_Understanding_Reasoning_and_Planning_images/cvpr_2024_LL3DA_Visual_Interactive_Instruction_Tuning_for_Omni-3D_Understanding_Reasoning_and_Planning.pdf-0018-06.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/cvpr_2024_LL3DA_Visual_Interactive_Instruction_Tuning_for_Omni-3D_Understanding_Reasoning_and_Planning_images/cvpr_2024_LL3DA_Visual_Interactive_Instruction_Tuning_for_Omni-3D_Understanding_Reasoning_and_Planning.pdf-0018-07.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/cvpr_2024_LL3DA_Visual_Interactive_Instruction_Tuning_for_Omni-3D_Understanding_Reasoning_and_Planning_images/cvpr_2024_LL3DA_Visual_Interactive_Instruction_Tuning_for_Omni-3D_Understanding_Reasoning_and_Planning.pdf-0018-08.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/cvpr_2024_LL3DA_Visual_Interactive_Instruction_Tuning_for_Omni-3D_Understanding_Reasoning_and_Planning_images/cvpr_2024_LL3DA_Visual_Interactive_Instruction_Tuning_for_Omni-3D_Understanding_Reasoning_and_Planning.pdf-0018-09.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/cvpr_2024_LL3DA_Visual_Interactive_Instruction_Tuning_for_Omni-3D_Understanding_Reasoning_and_Planning_images/cvpr_2024_LL3DA_Visual_Interactive_Instruction_Tuning_for_Omni-3D_Understanding_Reasoning_and_Planning.pdf-0018-10.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/cvpr_2024_LL3DA_Visual_Interactive_Instruction_Tuning_for_Omni-3D_Understanding_Reasoning_and_Planning_images/cvpr_2024_LL3DA_Visual_Interactive_Instruction_Tuning_for_Omni-3D_Understanding_Reasoning_and_Planning.pdf-0018-11.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/cvpr_2024_LL3DA_Visual_Interactive_Instruction_Tuning_for_Omni-3D_Understanding_Reasoning_and_Planning_images/cvpr_2024_LL3DA_Visual_Interactive_Instruction_Tuning_for_Omni-3D_Understanding_Reasoning_and_Planning.pdf-0018-12.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/cvpr_2024_LL3DA_Visual_Interactive_Instruction_Tuning_for_Omni-3D_Understanding_Reasoning_and_Planning_images/cvpr_2024_LL3DA_Visual_Interactive_Instruction_Tuning_for_Omni-3D_Understanding_Reasoning_and_Planning.pdf-0018-13.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/cvpr_2024_LL3DA_Visual_Interactive_Instruction_Tuning_for_Omni-3D_Understanding_Reasoning_and_Planning_images/cvpr_2024_LL3DA_Visual_Interactive_Instruction_Tuning_for_Omni-3D_Understanding_Reasoning_and_Planning.pdf-0018-14.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/cvpr_2024_LL3DA_Visual_Interactive_Instruction_Tuning_for_Omni-3D_Understanding_Reasoning_and_Planning_images/cvpr_2024_LL3DA_Visual_Interactive_Instruction_Tuning_for_Omni-3D_Understanding_Reasoning_and_Planning.pdf-0018-15.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/cvpr_2024_LL3DA_Visual_Interactive_Instruction_Tuning_for_Omni-3D_Understanding_Reasoning_and_Planning_images/cvpr_2024_LL3DA_Visual_Interactive_Instruction_Tuning_for_Omni-3D_Understanding_Reasoning_and_Planning.pdf-0018-16.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/cvpr_2024_LL3DA_Visual_Interactive_Instruction_Tuning_for_Omni-3D_Understanding_Reasoning_and_Planning_images/cvpr_2024_LL3DA_Visual_Interactive_Instruction_Tuning_for_Omni-3D_Understanding_Reasoning_and_Planning.pdf-0018-17.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/cvpr_2024_LL3DA_Visual_Interactive_Instruction_Tuning_for_Omni-3D_Understanding_Reasoning_and_Planning_images/cvpr_2024_LL3DA_Visual_Interactive_Instruction_Tuning_for_Omni-3D_Understanding_Reasoning_and_Planning.pdf-0018-18.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/cvpr_2024_LL3DA_Visual_Interactive_Instruction_Tuning_for_Omni-3D_Understanding_Reasoning_and_Planning_images/cvpr_2024_LL3DA_Visual_Interactive_Instruction_Tuning_for_Omni-3D_Understanding_Reasoning_and_Planning.pdf-0018-19.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/cvpr_2024_LL3DA_Visual_Interactive_Instruction_Tuning_for_Omni-3D_Understanding_Reasoning_and_Planning_images/cvpr_2024_LL3DA_Visual_Interactive_Instruction_Tuning_for_Omni-3D_Understanding_Reasoning_and_Planning.pdf-0018-20.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/cvpr_2024_LL3DA_Visual_Interactive_Instruction_Tuning_for_Omni-3D_Understanding_Reasoning_and_Planning_images/cvpr_2024_LL3DA_Visual_Interactive_Instruction_Tuning_for_Omni-3D_Understanding_Reasoning_and_Planning.pdf-0018-21.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/cvpr_2024_LL3DA_Visual_Interactive_Instruction_Tuning_for_Omni-3D_Understanding_Reasoning_and_Planning_images/cvpr_2024_LL3DA_Visual_Interactive_Instruction_Tuning_for_Omni-3D_Understanding_Reasoning_and_Planning.pdf-0018-22.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/cvpr_2024_LL3DA_Visual_Interactive_Instruction_Tuning_for_Omni-3D_Understanding_Reasoning_and_Planning_images/cvpr_2024_LL3DA_Visual_Interactive_Instruction_Tuning_for_Omni-3D_Understanding_Reasoning_and_Planning.pdf-0018-23.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/cvpr_2024_LL3DA_Visual_Interactive_Instruction_Tuning_for_Omni-3D_Understanding_Reasoning_and_Planning_images/cvpr_2024_LL3DA_Visual_Interactive_Instruction_Tuning_for_Omni-3D_Understanding_Reasoning_and_Planning.pdf-0018-24.png)


Figure 10. **Visualization of the Bounding Boxes.** We simulate the user click with the spatial position of vote queries proposed in Vote2Cap-DETR [9]. The bounding boxes from LL3DA are reconstructed via decoding the “ _<_ obj _>cx_ , _cy_ , _cz_ , _w_ , _h_ , _l<_ /obj _>_ ” from the generated texts. The reconstructed bounding boxes are tight to the objects in the 3D scene. 

18 

## **D. Data and Instructions** 

In this section, we provide statistic details on the 3D vision and language datasets we train the model (Sec. D.1) as well as the several textual instruction samples for different tasks (Sec. D.2). 

## **D.1. Data Statistics** 

Our method is trained on the following datasets: 

**ScanNet [15]** is a 3D indoor dataset covering diverse 3D environments, including apartments, living rooms, kitchens, bedrooms, and so on. ScanNet splits the dataset into 1201, 312, and 100 scenes for training, validation, and testing, repectively. **ScanRefer [6]** is a 3D vision-language dataset, which contains 36,665 free-form natural language descriptions on 7,875 objects from 562 scenes for training, and 9,508 descriptions on 2,068 objects from 141 scenes from ScanNet [15] for evaluation. **Nr3D** [1] is a 3D vision-language dataset with 32,919 free-form natural language descriptions on 4,664 objects from 511 scenes for training, and 8,584 descriptions on 1,214 objects from 130 scenes from ScanNet [15] for evaluation. 

**ScanQA [2]** is a 3D vision-language dataset. The training set of ScanQA consists of 25,563 question-answer pairs on 562 scenes from ScanNet training set. ScanQA further splits 141 unique 3D scenes from ScanNet validation set into two sets: 

- The validation set contains 4,675 question-answer pairs on 71 out of 141 validation scenes. 

• The “test w/ object” set (“test set with ScanNet object annotations”) contains 4,976 questions on the rest 70 scenes. Additionally, ScanQA annotates 6,149 questions on 97 scenes on the ScanNet test set as the “test w/o object” set (“test set without ScanNet object annotations”). 

**3D-LLM [26].** The ScanNet subset of 3D-LLM covers 1) 1,033 descriptions on 517 scenes, 2) 1,864 lines of embodied task planning on 510 scenes, and 3) 2,955 lines of multi-turn embodied dialogues on 517 scenes. All of the annotated scenes come from the training set of ScanNet. The evaluation details could be found in Sec. B. 

## **D.2. Training Samples** 

We list a couple of the training samples for different tasks in Tab. 14. The special tokens “[Caption]”, “[Question]”, “[Answer]”, “[Box]”, and “[Response]” will be replaced with the language annotations and 3D coordinates. 

Table 14. **Data Samples Used to Train LL3DA.** We list a couple of data samples used to train LL3DA for each task. 

|Task Name|Text Instructions|Visual Interactions|Expected Output|
|---|---|---|---|
||“### human: describe this object in the given 3D scene. ### assistant:|[Click] or [Box]|[Caption]|
|3D Dense Captioning|“### human: given the 3D scene, localize and describe this object. ###|[Click] or [Box]|the object is localized at [Box],|
||assistant:”||[Caption]|
||“### human: given the 3D scene, answer the question: “[Question]”|Optional [Clicks]|[Answer].|
|3D Question Answering|### assistant:”<br>“### human: answer the question: ”[Question]” with the related object|Optional [Clicks]|the related objects are localized at|
||locations in the input 3D scene. ### assistant:”||[Box]. the answer is: [Answer].|
|Scene Description|“### human: describe this 3D scene### assistant:”|-|[Caption]|
||“### human: I want to prepare and cook a meal in the kitchen. I have|-|open the refrigerator|
||done these things: 1. go to the refrigerator. What should I do next? ###|||
|Embodied Planning|assistant:”|||
||“### human: I want to set up a home offce workspace. What should I<br>do? ### assistant:”|-|1. place a desk against the wall<br>2. position a chair at the desk...|
|Embodied Dialogue|“### human: ... ### assistant:”<br>“### human: ... ### assistant: ... ### human: ... ### assistant:”|-<br>-|[Response]<br>[Response]|



19 

## **E. Additional Implementation Details** 

**Scene Encoder.** As introduced in the main paper, we adopt the pre-trained masked transformer encoder [39] as the scene encoder. The masked transformer encoder first tokenizes the input point cloud _PC_ into 2,048 points tokens uniformly scattered in the 3D scene with a set-abstraction layer [44]. Following that are three cascaded transformer encoder blocks with masking radius of 0 _._ 16, 0 _._ 64, 1 _._ 44, respectively. There is another set-abstraction layer between the first two transformer blocks, which further down-samples the encoded tokens into 1,024 point tokens. The output feature of the scene encoder is _fenc_ = R[1] _[,]_[024] _[×]_[256] , representing a 256 dimensioned feature for each of the 1,024 point tokens. 

**Multi-Modal Transformer.** The feature dimension of the Multi-Model Transformer is 768. We use up to 12 heads for each attention layer, and a total of six transformer layers in practice. As mentioned in the main paper, we choose to load the pre-trained word embeddings and positional embeddings from BERT [18]. The total vocabulary size is 30,522, and the number of position embeddings is 512. 

**Trainable Parameters.** The total number of trainable parameters is about 111M, which is less than 10% of the parameters in the frozen LLM backbone (OPT-1.3B). 

20 

