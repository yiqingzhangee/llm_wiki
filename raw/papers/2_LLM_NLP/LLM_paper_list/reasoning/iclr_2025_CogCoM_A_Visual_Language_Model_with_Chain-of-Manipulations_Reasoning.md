## **CogCoM: Train Large Vision-Language Models Diving into Details through Chain of Manipulations** 

**Ji Qi** _[⋄†]_ **, Ming Ding** _[†]_ **, Weihan Wang** _[⋄†]_ **, Yushi Bai** _[⋄†]_ **, Qingsong Lv** _[†]_ **, Wenyi Hong** _[⋄†]_ **Bin Xu** _[⋄∗]_ **, Lei Hou** _[⋄]_ **, Juanzi Li** _[⋄]_ **, Yuxiao Dong** _[⋄]_ **, Jie Tang** _[⋄∗] ⋄_ Tsinghua University _†_ Zhipu AI 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_CogCoM_A_Visual_Language_Model_with_Chain-of-Manipulations_Reasoning_images/iclr_2025_CogCoM_A_Visual_Language_Model_with_Chain-of-Manipulations_Reasoning.pdf-0001-03.png)


**----- Start of picture text -----**<br>
Find regions using GROUNDING(road signs) to get  No, there is a sign<br>[[3,4,4,6],[12,2,15,5]] … Zoom into [3,4,4,6] cropped … showing one way.<br>VLM with Chain of Manipulations<br>Can I turn left on<br>this road?<br>Answer with CoM.<br>Q : What is the brand of the plane? Q : How many wheels of the truck can be seen? Q : What time does this photo show?<br>CogCoM : Using GROUNDING(the plane) to find  CogCoM : Using GROUNDING(all the individual  CogCoM : Applying GROUNDING(the clock) …,<br>planes, and there are two regions [[255, 217, 312,  wheels of the truck) to locate all the wheels in this  resulting in the region of [[814,531,940,1062]].<br>245], [193, 113, 481, 208]] that can be found. picture, resulting in the boxes of [[56, 256, 96, 513<br>], [105, 240, 144, 480], …] Use CropZoomIn([[814,531,940,1062<br>With the most notable one, using GROUNDING( ]],2) to crop … and then zoom in it by<br>the logo in [[193, 113, 481, 208]]) to find the logo,  Based on the identified boxes of the wheels, the  two times … re-input this new image.<br>resulting in [[246,142,307,163]]. number of visible wheels of the truck is ten.<br>Using ReadTime(the clock in the new<br>Using CropZoomIn([[246, 142, 3 image) to re-cognize the time on the<br>07,163]], 3) to get a new image. clock, and the result is 1:45.<br>The brand name is AirFrance.<br>Detailed Recognition Counting Objects Read Time<br>O<br>A B<br>Q : In what year did the fatalities peak? Q : What color is the umbrella the men are standing  Q : In circle  O  with radius of 5.0, the length of chord<br>CogCoM : Firstly, using GROUNDING(fatalities  under? AB  is 8.0, then the distance from the  O  to  AB  is?<br>values) to find the death toll, and the values at the  CogCoM : Determine if there is an umbrella present. CogCoM : Draw a straight line with LINE([[(54, 43<br>position [[116, 114, 913, 755]] are 221, 213, 212,  ), (13,70)]]) to obtain a new image after drawing.<br>214, 199, 194, 193, 193, 195, 190, 183, 173, 168,  There are no men standing together; there is only<br>160, 155, 151, 141, 121, 124, 118, and 85. The  one person visible. With the new image, set O perpe<br>maximum number is 221. The person in the image is not standing but appears  -ndicular to AB at point C. SinceC is the midpoint of AB, making O<br>Draw a line using LINE[[([374, to be in a throwing stance. AC 4.0 long. In the right triangle A B<br>71), (374, 466])] to obtain new Since there is no umbrella present, its color cannot  AOC, OA=5.0, AC=4.0. Accordi<br>image after drawing. be determined. -ng to the Pythagorean theorem,<br>The corresponding year is 2008. OC=3.0. So the answer is 3.0.<br>Read Figures De-hallucination Math Geometry<br>Manipulation creation Manipulation result (same color for referential relation)<br>**----- End of picture text -----**<br>


Figure 1: CogCoM solves various visual problems with Chain of Manipulations mechanism. Note that the CoM reasoning generates evidential and explainable steps, without involving external tools. 

- Work done when JQ, WW, YB, and WH interned at Zhipu AI. 

> _∗_ Corresponding authors: BX and JT (jietang@tsinghua.edu.cn) 

Preprint. Under review. 

## **Abstract** 

Vision-Language Models (VLMs) have demonstrated their broad effectiveness thanks to extensive training in aligning visual instructions to responses. However, such training of conclusive alignment leads models to ignore essential visual reasoning, further resulting in failures in meticulous visual problems and unfaithful responses. Drawing inspiration from human cognition in solving visual problems ( _e.g._ , _marking_ , _zoom in_ ), this paper introduces **Chain of Manipulations** , a mechanism that enables VLMs to solve problems step-by-step with evidence. After training, models can solve various visual problems by eliciting intrinsic manipulations ( _e.g., grounding_ , _zoom in_ ) with results ( _e.g., boxes_ , _image_ ) actively without involving external tools, while also allowing users to trace error causes. We study the roadmap to implement this mechanism, including (1) a flexible design of manipulations upon extensive analysis, (2) an efficient automated data generation pipeline, (3) a compatible VLM architecture capable of multi-turn multi-image, and (4) a model training process for versatile capabilities. With the design, we also manually annotate 6K high-quality samples for the challenging graphical mathematical problems. Our trained model, **CogCoM** , equipped with this mechanism with 17B parameters achieves state-of-the-art performance across 9 benchmarks from 4 categories, demonstrating the effectiveness while preserving the interpretability. Our code, model weights, and collected data will be publicly available. 

## **1 Introduction** 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_CogCoM_A_Visual_Language_Model_with_Chain-of-Manipulations_Reasoning_images/iclr_2025_CogCoM_A_Visual_Language_Model_with_Chain-of-Manipulations_Reasoning.pdf-0002-03.png)


**----- Start of picture text -----**<br>
First, use GROUNDING(man) to find regions [[4,5,7,12], [28,6,30,9]] …<br>Then, use GROUNDING(pillar near [4,5,7,12]) to find [[8,0,17,13]] …<br>Zoom into the cropped region [[9,7,11,9]] by two times and re-input …<br>No Smoking Based on the new image, the text written is QUICK DEPOSIT.<br>���<br>���<br>Large Language Model Large Language Model<br>Large Language Model<br>Image tokens pillar in front of the man?What is written on the  Image tokens pillar in front of the man?What is written on the<br>Image tokens<br>**----- End of picture text -----**<br>


Figure 2: In comparison with existing VLMs, CogCoM performs the multiple steps of evidential reasoning with chain of manipulations (CoM) to achieve the faithful answer to visual scene. 

Benefiting from the advantage of Large Language Models (LLMs) in broad world knowledge, large Vision Language Models (VLMs) (Alayrac et al., 2022; Wang et al., 2023b) that are further trained to understand visual inputs have demonstrated viabilities on broad multimodal scenarios, such as visual question answering (Liu et al., 2023b), visual grounding (Peng et al., 2023), optical character recognition (Zhang et al., 2023b). The research employing VLMs as foundation models (Bai et al., 2023; Sun et al., 2023b; Wang et al., 2023b) usually involves two main stages of training, where the first stage develops intrinsic visual understanding ability through exposure to massive imagecaption pairs, and the second stage endows the models with problem-solving capabilities through the instruction tuning. 

However, existing tuning methods train models to respond to instructions with conclusive language responses upon visual inputs, which leads models to ignore the essential intermediate visual reasoning and further results in failures in meticulous visual problems, unfaithful responses, and even hallucinations. For example in the left subplot of Figure 2, we test the top-performing model CogVLM (Wang et al., 2023b) about the details in the image ( _i.e., texts written on a pillar_ ), and it directly responds an incorrect answer ( _i.e., NO SMOKING_ ), most likely from bias to visual or linguistic priors ( _i.e., typical scenes with a pillar in office_ ). The absence of the essential reasoning on the visual scene may lead to a rash response (Hwang et al., 2023). 

2 

Humans solve problems regarding visual details by marking or processing the given images for convenience and rigor, which we refer to as manipulations. For example, we find targets by sequentially locating references, and concentrate on subtle details by zooming into a corresponding region. Most of VLMs have developed numerous intrinsic capabilities ( _e.g.,_ grounding boxes, recognizing texts) during the first stage of training. By further imitating the fundamental human behaviours ( _e.g.,_ cropping, zoom in), models have the potential to perform this cognitive reasoning process. Three major obstacles in eliciting VLMs with such reasoning are (1) flexible definitions of manipulations covering most visual problems, (2) an efficient data collection pipeline capable of producing abundant training data, and (3) a multi-turn multi-image VLM structure compatible with existing models. 

Inspired by the human cognition in solving visual problems, we introduce **Chain of Manipulations (CoM)** , a mechanism that enables VLMs to solve problems step-by-step with evidence, with each step potentially involving a manipulation on the visual input and its corresponding result, both generated by the model to facilitate the success and fidelity. This paper studies a complete roadmap with manipulations design, data collection, model architecture and training process for training general VLMs with this mechanism. We first formally design 6 basic manipulations upon the pilot experiments, which are capable of handling diverse visual problems. Next, we propose a cascading data generation pipeline based on reliable large language models ( _e.g.,_ LLMs, the linguistic annotators) and visual foundational models ( _e.g.,_ VFMs, the visual annotators), which can automatically produce abundant error-free training data. We collect 70K CoM samples with this pipeline. We then devise a multi-turn multi-image model architecture compatible with typical VLMs structures. Based on a data recipe incorporating the curated corpus, we finally train a general VLM equipped with CoM reasoning mechanism, named CogCoM, which possesses capabilities of chat, captioning, grounding and reasoning. Additionally, benefiting from the expressive capability of the proposed mechanism, we further manually annotated 6K high-quality samples of graphical mathematical problems, each accompanied by a CoM reasoning process, to advance the research of VLMs in solving challenging mathematical problems. 

We conduct extensive experiments on 9 benchmarks from 4 categories, including TextVQA (Singh et al., 2019), ST-VQA (Biten et al., 2019), TallyVQA (Acharya et al., 2019), and GQA Hudson & Manning (2019) for detailed visual question answering, RefCOCO (Yu et al., 2016), RefCOCO+(Yu et al., 2016), and RefCOCOg (Mao et al., 2016) for visual grounding, POPE (Li et al., 2023c) for hallucination validation, and MM-Vet (Yu et al., 2023b) for general multimodal ability. Our model achieves up to 9.0 and 1.09 accuracy improvement on the detailed VQA and grounding benchmarks, respectively, and the superior performance on the general multimodal benchmark. The results demonstrate the effectiveness of the mechanism while maintaining the interpretability of outputs. 

## **2 Terminology** 

We first conduct pilot experiments to investigate the possible manipulations capable of handling diverse visual problems. 

Specifically, given a question about an image, we prompt the advanced large language model, GPT-4, to generate solving steps by optionally utilizing possible actions on the image that facilitate problem-solving. We conduct this experiment on 170K questions from TextVQA, a dataset requiring detailed reasoning and recognition on images. To ensure the stability, we manually write 4 demonstrations as priors, The detailed statistics are available at Appendix C.3. 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_CogCoM_A_Visual_Language_Model_with_Chain-of-Manipulations_Reasoning_images/iclr_2025_CogCoM_A_Visual_Language_Model_with_Chain-of-Manipulations_Reasoning.pdf-0003-06.png)


**----- Start of picture text -----**<br>
OCR<br>50000 GROUNDING<br>ZOOMIN<br>40000<br>30000<br>20000<br>10000<br>0<br>read, extract, ... locate, look for ...  zoom in, focus on ...<br>COUNTING<br>1000 CALCULATE<br>LINE<br>800<br>600<br>400<br>200<br>0<br>count, number, ... calculate, assess, ... draw, add, ...<br>#Frequencies<br>#Frequencies<br>**----- End of picture text -----**<br>


We utilize the StanfordCoreNLP toolkit to extract verb phrases referring to the actions, and the distribution of frequencies is shown in Figure 3. Through result analysis, we find that most of the actions can be mapped to 6 fundamental manipulations on images: _OCR_ , _Grounding_ , _CropZoomIn_ , _Counting_ , _Calculate_ , and _Line_ . 

Figure 3: Distribution of the generated 465 actions base on GPT-4, mapped into 6 manipulations. 

3 

Based on the observation, we formally predefine a set of 6 manipulations, which can either be developed from pre-training or be learned from fine-tuning with the imitation to human behaviors: _M ⊆_ { _OCR_ ( _tgt_ ) _→ txt_ , _Grounding_ ( _tgt_ ) _→ bbx_ , _Counting_ ( _tgt_ ) _→ num_ , _Calculate_ ( _tgt_ ) _→ num_ , _CropZoomIn_ ( _bbx, x_ ) _→ img_ , _Line_ ( _pts_ ) _→ img_ }, where the parameters or results _tgt, txt, bbx, num, x, img, pts_ refer to the bounding boxes, zoom ratio, image, target description, numbers, texts, and points, respectively. In addition to the predefined manipulations, we also allow trained models to create new manipulations during inference to facilitate problem-solving. We empirically find that more complicated goals can be derived from these fundamental manipulations. 

We then define the **standard CoM data structure** to streamline the subsequent data construction and validation process. Given a question _Q_ about an initial input image _I_ 0, a VLM equipped with chain of manipulations mechanism solves the problem to achieve final answer as _VLM_ _**ς**_ ( _A, C|I_ 0 _, Q_ ), where _**ς**_ refers to the reasoning chain with evidence, 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_CogCoM_A_Visual_Language_Model_with_Chain-of-Manipulations_Reasoning_images/iclr_2025_CogCoM_A_Visual_Language_Model_with_Chain-of-Manipulations_Reasoning.pdf-0004-02.png)


where _C_ = ( _ci, c_ 2 _, ..., c|C|_ ) refers to the free-form textual descriptions incorporating manipulation names _fi_ and corresponding results from utilizing _fi_ . This definition explicitly declares the symbolic execution process, while also being compatible with linguistic reasoning steps. Based on this definition, we can clearly construct standard CoM samples that incorporating the manipulation executions and linguistic steps with evidence. After the data construction, we can utilize a simple method to convert the standard CoM samples to the **compatible VQA samples** . 

## **3 Data Collection** 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_CogCoM_A_Visual_Language_Model_with_Chain-of-Manipulations_Reasoning_images/iclr_2025_CogCoM_A_Visual_Language_Model_with_Chain-of-Manipulations_Reasoning.pdf-0004-05.png)


**----- Start of picture text -----**<br>
VQA Sample Linguistic Annotation Visual Annotation Compatible VQA<br>Q : What is written on the pillar in front of man in black top? A : QUICK DEPOSIT Generate solving steps for a given question, by optionally using chain of manipulations  GroundingGrounding (( tgttgt 21))  ! ! bbx bbx 21<br>from      or newly created ones. M Grounding ( tgt 3)  ! bbx 3<br>OCR ( img 1)  ! txt 1<br>I 0 bbx 2<br>Linguistic Annotators (LLMs) Visual Annotators (VFMs)<br>bbx 1 bbx 3 /I 1 bbx 1 A 0 Using Grounding(locate .., the position is at         .the man in black shirt bbx 1 ) to  [4,5,7,12] Q [28,6,30,9] (   ,     ,      ) Q I 0 A 0<br>txt 1 Using Grounding(the pillar in front of the man at         ) to find .., the resultant region is         . bbx 1 bbx 2 [8,0,17,13] …<br>Using Grounding(the text written on the pillar  [9,7,11,9]<br>in         ) to find .., the resultant region is         . bbx 2 bbx 3<br>Using CropZoomIn(         ,  bbx 3 4) to crop it on        ,  bbx 3 DFS ����������������������<br>and zoom in it by 4 times to get new image     . I 1<br>QUICK DEPOSIT<br>A 1 Using OCR(     )  recognize the text written on the image     , the result is  “QUICK DEPOSIT”. I 1 I 1 A (   ,     ,      ) Q ¯ I 1 A 1<br>**----- End of picture text -----**<br>


Figure 4: A cascading data generation pipeline that automatically produces standard CoM samples. Given an original VQA sample, the linguistic annotator (LLMs) taught with usage of manipulations (prompt) is first asked to provide solving steps for the question _Q_ , and the visual foundational models (VFMs) are then engaged to replace the manipulations results, followed by a final traversal on the tree branched by the possible manipulation results to find positive paths terminating to the answer _A_ . 

In this section, we first introduces the automated data generation pipeline (illustrated in Figure 4), that employs reliable LLMs as linguistic annotators and VFMs as the visual annotators to produce error-free CoM samples upon prevalent VQA corpus, and then present the manual annotation of high-quality CoM samples for the challenging graphical mathematical problems. 

## **3.1 Automated Data Generation** 

Given a general corpus _D_ = _{_ ( _I, Q, A_ ) _}_ consisting of triplet samples of images with corresponding visual question-answer pairs, our automated data generation pipeline consists of a linguistic annotator and several visual annotators according to the manipulations. For a question _Q_ in each sample, we first engage the linguistic annotator to generate manipulations-assisted solving steps with the CoM format ( _fi, ci_ ), where the corresponding results of the instantiated manipulation executions are set with variables as placeholders. In this paper, we adopt GPT-4 (OpenAI, 2023a), a large language 

4 

model with reliable language understanding and generation abilities as the linguistic annotator. We design a comprehensive prompt including the task requirements, usage of manipulations, and output data format, and further manually annotate 5 demonstrations for a stable generation. The detailed implementations are available at Appendix C.4. 

We then employ essential visual annotators to supply the results of manipulations requested in the solving steps by exactly performing the corresponding manipulations. By empirically analyzing the manipulations from both predefined set and newly created ones (refers to Appendix C.3 for a detailed statistics), we reveal the _Grounding_ and _OCR_ are two fundamental manipulations, and most of the others can be consequently derived ( _e.g., CropZoomIn_ along a region of box, _Counting_ upon recognized boxes, and _Calculate_ for the recognized formula). Therefore, we employ two visual foundational models, GroundingDINO (Liu et al., 2023c) and PaddleOCR (Du et al., 2020), and develop the implementations of these manipulations[3] . The execution of the manipulations will transform the sequential reasoning steps into a **tree** _T_ , as the input of current manipulation _f_ 1( _xa_ ) may rely on one of the multiple results of previous manipulation _f_ 2 _→_ ( _xb, xc_ ), _i.e._ , _xa_ rely on _xb_ ( _e.g.,_ step 2 for finding pillars in Figure 5). We then perform a traversal on each produced tree with Depth First Search (DFS) to find all positive paths _{Pi|Pi ∈T , i_ = 1 _,_ 2 _, ...}_ that can terminate with the final answer _A_ from the result of the last manipulation. Based on this method, the generated CoM samples with positive paths are guaranteed to be error-free. We implement this pipeline on 3 existing datasets that require detailed recognition or objects counting, TextVQA (Singh et al., 2019), ST-VQA (Biten et al., 2019), and TDIUC (Shrestha et al., 2019), to build 70K CoM samples[4] . The designed prompt, a generated example with linguistic and visual results, and detailed algorithm illustration are available at AppendixC.1. 

## **3.2 Human Annotation** 

The analysis from Fig.1 of AlphaGeometry (Trinh et al., 2024) shows that outputting auxiliary lines in linguistic reasoning process helps LLMs to solve complex geometry problems. Benefiting from the expressive capability of CoM structure, we have also manually annotated high-quality CoM samples for the graphical mathematical problems to facilitate VLMs in solving this challenging scenario. Similar to the automated pipeline, we engage 10 human experts as the linguistic annotators and visual annotators, where each expert is asked to annotate the linguistic solving steps and the use of manipulations, as well as the results of manipulations on images. We perform this annotation on the MathVista (Lu et al., 2023) and ChartQA (Masry et al., 2022), which include geometric and chart math problems, resulting in the collection of 6K high-quality CoM math samples. 

Finally, we adapt the CoM samples to be compatible with VQA-style training samples. For each CoM sample including _n_ images from manipulations outputs ( _I_ 0 _, Q, C_ 0 _, I_ 1 _, C_ 1 _, ..., In, A_ ), we convert it into a multi-turn VQA sample segmented by the images [( _I_ 0 _, Q, C_ 0) _,_ ( _I_ 1 _, Q, C_[¯] 1) _, ...,_ ( _In, Q, A_[¯] )], where _Ci_ represents the intermediate steps between _Ii_ and _Ii_ +1, and _Q_[¯] is a simple prompt asking model to answer question based on history. This transformation converts CoM samples into multi-turn VQA samples that are compatible with existing VLMs training data. The detailed statistics of the data generation are available at Appendix C.3. 

## **4 Model Training** 

## **4.1 Architecture** 

We use the same model architecture as CogVLM (Wang et al., 2023b), a general VLM approach that involves four fundamental components: (1) a Visual Encoder, (2) an MLP Adapter, (3) an LLM Backbone, and (4) a Visual Expert Module, for a reliable multimodal understanding. Concretely, the pre-trained EVA2-CLIP-E (Sun et al., 2023a) with 4B parameters and Vicuna-7B-v1.5 (Chiang et al., 2023) are adopted as the visual encoder and LLM backbone, respectively. A two-layer MLP (SwiGLU (Shazeer, 2020)) is further engaged to map the output of the visual encoder into the linguistic space of the LLM backbone. The visual expert module adds the vision-specific weights into the attention layer and feed-forward layer of each block in the LLM backbone, resulting in a total of 6.5B additional parameters for the deep fusion of modalities. 

> 3We simply implement the _CropZoomIn_ referring to human behaviors with a local code interpreter. 

> 4The success rate of GPT-4 to achieve the positive paths is 0.3555. 

5 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_CogCoM_A_Visual_Language_Model_with_Chain-of-Manipulations_Reasoning_images/iclr_2025_CogCoM_A_Visual_Language_Model_with_Chain-of-Manipulations_Reasoning.pdf-0006-00.png)


**----- Start of picture text -----**<br>
Response_0<br>Response_1<br>���<br>��� Stage 1: Pre-training Data structure:<br>Large Language Model - Foundational visual ������������� - {(image, caption)}<br>KV-memory LLM - Grounded ������������� - {(image, prompt, response)}<br>(Image_0, Prompt_0)<br>Stage 2: Fine-tuning Data structure<br>Projector Prompt_1 - Multimodal problems-solving - {(image, prompt, response)}<br>ViT<br>Image_1 Chat, Captioning, Grounding, Reasoning<br>**----- End of picture text -----**<br>


Figure 5: **Left** : A compatible VLM architecture capable of multi-turn multi-image understanding. **Right** : An effective training process to develop a general VLM with versatile capabilities. 

Based on this general architecture, we develop a memory-based multi-turn multi-image VLM approach. Specifically, for a multi-turn VQA sample [( _It, Qt, At_ ) _|t_ = 1 _,_ 2 _, ..._ ], where _At_ refers to _Ct_ in CoM, we keep the accumulated KV memories of each layer in the LLM backbone throughout these turns. And at each turn _t_ in training and inference, we calculate the attention function _att_ as: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_CogCoM_A_Visual_Language_Model_with_Chain-of-Manipulations_Reasoning_images/iclr_2025_CogCoM_A_Visual_Language_Model_with_Chain-of-Manipulations_Reasoning.pdf-0006-03.png)


where _**Q** t ∈_ R _[s][×][d]_ is query representation of current layer, and the _**K**[′] t[,]_ _**[ V]**[′] t[∈]_[R][(] _[s][×][t]_[)] _[×][d]_[ refer to the] concatenation of accumulated representations and will be further truncated if the sequence length _s × t_ is greater than a predefined threshold. At _t >_ 0, the new image _It_ will be cropped from _It−_ 1 and amplified with the Bicubic Interpolation (Keys, 1981). 

## **4.2 Training** 

The proposed CogCoM-17B relies on two main stages of training, to develop the capabilities of general multimodal task-solving as well as the visual reasoning. 

**First Stage Pre-Training** This stage consists of two ordinal sub-phases of training for foundational visual understanding and grounded generation. Following the pre-training of CogVLM (Wang et al., 2023b), we first train model on 1.5B image-text pairs cleaned from the LAION-2B (Schuhmann et al., 2022) and COYO-700M (Byeon et al., 2022) with 120,000 iterations and batch size of 8,192. We then train model on 40M grounded image-question-answer triples cleaned from LAION-115M (Li et al., 2023b) with 60,000 iterations and batch size of 1,024, where each noun phrase in the answer is followed by a list of coordinates [[ _x_ 0 _, y_ 0 _, x_ 1 _, y_ 1] _, ..._ ][5] referring the phrase to the grounded objects in the image. Both phases adopt the next token prediction objective, and train the 6.5B parameters of visual experts. 

**Second Stage Alignment** This stage further trains the model to align with human preferences on solving practical visual problems. We fuse the produced CoM data with 3 types of corpus, including MultiInstruct (Xu et al., 2022), LLaVAR (Zhang et al., 2023b), and ShareGPT4V (Chen et al., 2023c), referring the abilities of instruction-following, texts-recognizing, and detailed-captioning. This fusion results in a total of 570K ( _I, Q, A_ ) samples, where the answer _A_ in CoM data consists of multiple turns. For the training data of CoM, we randomly prepend a lunching prompt[6] _P[M]_ to questions _Q_ = _P[M]_ + _Q_ asking models to optionally use manipulations for the adaption of explicitly eliciting. We empirically show that the model can effectively learn the evidential visual reasoning by ingesting this portion of CoM data. We train model with 14,000 iterations and a batch size of 160, where the learning rate reaches 10 _[−]_[5] after 280 steps of warm-up and then decays linearly. The parameters of 6.5B visual experts are trained with the objective of next token prediction. These two stages of training result in our standard version of CogCoM involving both chat and reasoning capabilities. More training details are available at Appendix D.2. 

> 5 _xi, yi ∈_ [000 _,_ 999] refer to the normalized pixel coordinates. 

> 6See Appendix D.1 for examples. 

6 

## **5 Experiment** 

To quantitatively validate the suitability and efficiency of the proposed method, we conduct experiments on 9 benchmarks corresponding to 4 categories of multimodal capabilities, as well as on a newly constructed testbed that includes the evidential reasoning paths with a keypoints-aware metric. Following previous works, we train two generalist versions of CogCoM for adapting to the different scenarios of Visual Question Answering and Visual Grounding, and evaluate the standard version with a qualitative analysis (Hwang et al., 2023). We also evaluate the time complexity. 

- **Detailed Visual Question Answering.** This task involves models to perform detailed reasoning or recognition on images. We use 4 prominent benchmarks including, GQA (Hudson & Manning, 2019), TextVQA (Singh et al., 2019), ST-VQA (Biten et al., 2019), and TallyVQA (Acharya et al., 2019). 

- **Visual Grounding.** Visual grounding evaluates the crucial abilities of VLMs on meticulous position understanding. We evaluate our model on 3 standard benchmarks, RefCOCO (Yu et al., 2016), RefCOCO+ (Yu et al., 2016), and RefCOCOg (Mao et al., 2016). 

- **General Multimodal Capabilities & Hallucination.** We also evaluate on a general multimodal benchmark, MM-Vet (Yu et al., 2023b), and a hallucination detection benchmark POPE (Li et al., 2023c), to investigate the helpfulness of visual reasoning. 

## **5.1 Experiments on Detailed VQA** 

VLMs have demonstrated the well-known superiority in visual scenes with salient content understanding. We evaluate the effectiveness of CogCoM on VQAs on detailed understanding, which typically require models to perform multiple actions ( _find, read_ ) or multiple reasoning steps ( _recognizing and then calculating_ ). Following previous studies (Wang et al., 2023b), we train our model obtained from the first-phase of stage-1 on a mixture of data, including an instruction corpus of MultiInstruct, 13 publicly available VQA datasets (only using training set), a newly created VQA dataset built through promoting GPT-4V (OpenAI, 2023b) for image-oriented question-answer generation, and the automatically generated 70K CoM corpus. This training results in a generalist VQA model incorporating CoM reasoning. For all existing VQA tasks, we directly prompt CogCoM with given questions and examine the correctness of outputted answers. 

|**Type**|**Model**|**GQA**<br>test-balanced|**TallyVQA**<br>simple<br>complex|**TallyVQA**<br>simple<br>complex|**TextVQA**<br>test|**ST-VQA**<br>test|
|---|---|---|---|---|---|---|
||Flamingo (Alayrac et al., 2022)|-|-|-|54.1|-|
|Generalist|GIT (Wang et al., 2022a)<br>GI2 (Wang et al., 2022a)|-<br>-|-<br>-|-<br>-|59.8<br>67.3|-<br>-|
||BLIP-2 (Li et al., 2023b)|44.7_†_|-|-|-|21.7|
||InstructBLIP (Dai et al., 2023)|49.5_†_|-|-|-|50.7_†_|
||Qwen-VL (Bai et al., 2023)|59.3|-|-|63.8|-|
||CogVLM (Wang et al., 2023b)|65.2|79.8|68.0|69.7|61.0|
||**CogCoM**|**71.7**|**84.0**|**70.1**|**71.1**|**70.0**|
|Specialist||72.1|86.0|75.6|71.4|86.0|
|SOTAs||(CFR)|(PaLI-X)|(PaLI-X)|(PaLI-X)|(SMoLA)|



Table 1: Performance on Visual Question Answering benchmarks, where the results labeled with _†_ refer to the few-shot setting. CogCoM achieves SOTA across the board, and demonstrates the effectiveness on the visual reasoning and scene texts recognition benchmarks. 

## **5.1.1 GQA, TextVQA, ST-VQA, TallyVQA** 

**Settings** GQA is a compositional VQA benchmark with diverse reasoning questions coming from semantic functional programs. TallyVQA is an objects counting benchmark with human-annotated complex counting questions involving challenging non-zero counterparts. TextVQA and ST-VQA are two texts understanding benchmarks requiring models to answer questions through textual cues on images. We use the official evaluation scripts for GQA and TallyVQA, which calculate the accuracy score by the Exact Matching (EM) between model predictions and answers. For TextVQA and ST-VQA, we submit our model predictions to the official online websites for calculating the accuracy with VQA Score metric (Antol et al., 2015). 

7 

**Results** As the results shown in Table 2, CogCoM achieves the state-of-the-art performance in comparison with all generalist models, and achieves significant improvements over the baseline model. Specifically, compared to the baseline model, our model achieves up to 5.97 and 9.0 percentage points improvement on the benchmarks that requires complex reasoning and detailed recognition, respectively. On GQA and TextVQA, CogCoM also obtains comparable results with the large-scale specialist SOTAs. This result demonstrates the effectiveness of the proposed approach in solving details recognition problem. 

## **5.1.2 Experiments for Reasoning Accuracy and Time Complexity** 

Due to the lack of resource, we build CoM-test, a benchmark with evidential reasoning chains on the TextVQA test set based on the proposed data generation pipeline, and also introduce a keypointsaware metric to validate the correctness of reasoning paths (see Appendix C.3 for detailed statistics). We also evaluate the time complexity for model generation on a held-out benchmark, MM-Vet. 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_CogCoM_A_Visual_Language_Model_with_Chain-of-Manipulations_Reasoning_images/iclr_2025_CogCoM_A_Visual_Language_Model_with_Chain-of-Manipulations_Reasoning.pdf-0008-03.png)


**----- Start of picture text -----**<br>
CogCoM / answer<br>60.00 CogCoM / explanation<br>55.59 350 CogCoM/TokensCogVLM/Tokens CogCoM/TimeCogVLM/Time 346.1 20.0<br>300 292.5 17.5<br>275.4<br>262.9<br>40.00 250 234.6 15.0<br>12.5<br>30.00 200 190.9 189.8<br>10.0<br>157.4<br>150 141.7<br>20.00 7.5<br>100 5.0<br>71.4<br>7.272.15 Zero shot / answer 50 18.0 24.1 22.8 42.1 33.0 48.9 2.5<br>0 2k 4k 6k 8k 0 0.0<br>Training Steps 1 2 3 4 5 6 7 8<br>Figure 6: Left : Results on a reasoning testbed CoM-test shows CogCoM achieves satisfactory<br>performance with only 70K training data and 2K steps. Right : Results on MM-Vet shows that<br>CogCoM produces comprehensive reasoning content without incurring excessive time overhead.<br>#Tokens<br>Accuracy Score<br>#Time (seconds)<br>**----- End of picture text -----**<br>


**Reasoning Accuracy** To validate the correctness of execution and results of manipulations in reasoning paths, we introduce a keypoints-aware evaluation metric that concentrates on these contents and their order. Concretely, given a predicted chain-answer pair ( _C[′] , A[′]_ ) and the ground truth pair ( _C, A_ ), we first extract the keypoints ( _i.e.,_ the name, parameters, and results of manipulations) in _A[′] , A_ to form two lists, and then discretize these two lists into _K[′]_ and _K_ based on a bagof-words composed of all keypoints. Then, we calculate the normalized Levenshtein Distance _sK_ = _Levenshtein_ ( _K[′] , K_ ) _/N_ as the manipulation score. We also compute the BLEU (Papineni et al., 2002) score _sC_ = BLEU( _C[′] , C_ ) as the paragraph score. Finally, a weighted average of these two scores serves as the ultimate reasoning score s _acc_ = (0 _._ 6 _× sK_ + 0 _._ 4 _× sC_ ) _/_ 2. 

We train our first-stage model only using the 70K automated CoM data without other supervision for qualitatively evaluate the effectiveness of chains, and the results are shown in the left subplot of Figure 6. We find that by training with the CoM chains, our model can swiftly achieve the satisfactory performance of 48.41 accuracy score with 2k training steps, and obtain the optimal result of 55.59 with 8K steps. Additionally, the explanation scores gradually improve along with the model performance, indicating that successful reasoning steps contribute to the achieving of final answer. 

**Time Complexity** We also evaluate the time complexity and average length of tokens during model reasoning on a held-out test set, MM-Vet. Specifically, we run CogCoM and the baseline model on all 218 questions, and record the time overhead as well as the average number of outputted tokens (using the Vicuna-7B-v1.5 tokenizer). We divide the 218 samples into 8 intervals based on the time expenditure for each sample and calculate the average values of the time complexity and the number of tokens for each interval, with the results presented in the right subplot of Figure 6. 

From the results we find that compared to baseline model, CogCoM produces information-intensive reasoning content ( _e.g.,_ detection boxes, auxiliary lines) without incurring infeasible time overhead. For example, without quantitive optimization, CogCoM outputs 262.9 informative tokens in approximately 9 seconds. With the advantages in long-context optimization techniques (Hooper et al., 2024), we believe that it is crucial for models to produce informative content and accurate responses. 

8 

## **5.2 Experiments on Visual Grounding** 

The task of visual grounding requires models to precisely provide the corresponding coordinates of regions in an image based on the given target description. Following the existing work (Wang et al., 2023b), we train our model obtained by the first stage on a mixture of datasets, including an instruction corpus MultiInstruct, a high-quality grounded VQA corpus introduced in CogVLM, and the 70K CoM data. This training results in a generalist grounding model that is excelling at visual grounding while capable of reasoning. For all benchmarks, we prompt CogOM in a chat manner to ask the model to provide grounded coordinates, such as “ _Where is ⟨expr⟩ answer in [x0,y0,x1,y1] format._ ", where the _⟨expr⟩_ refers to the target expression. We use the standard metric, that considers a prediction as correct when the intersection-over-union (IoU) between boxes is greater than 0.5. 

|**Type**<br>**Model**|**RefCOCO**|**RefCOCO+**|**RefCOCOg**<br>val<br>test|
|---|---|---|---|
||val<br>test-A<br>test-B|val<br>test-A<br>test-B||
|Generalist<br>OFA-L* (Wang et al., 2022b)<br>79.96<br>83.67<br>76.39<br>Shikra-7B (Chen et al., 2023b)<br>87.01<br>90.61<br>80.24<br>Shikra-13B (Chen et al., 2023b)<br>87.83<br>91.11<br>81.81<br>Qwen-VL (Bai et al., 2023)<br>89.36<br>92.26<br>85.34<br>CogVLM (Wang et al., 2023b)<br>**92.51**<br>93.95<br>88.73<br>**CogCoM**<br>92.34<br>**94.57**<br>**89.15**||68.29<br>76.00<br>61.75<br>81.60<br>87.36<br>72.12<br>82.89<br>87.79<br>74.41<br>83.12<br>88.25<br>77.21<br>87.52<br>91.81<br>81.43<br>**88.19**<br>**92.80**<br>**82.08**|67.57<br>67.58<br>82.27<br>82.19<br>82.64<br>83.16<br>85.58<br>85.48<br>**89.46**<br>90.09<br>89.32<br>**90.45**|
|Specialist<br>SOTAs<br>92.64<br>(UNINEXT)<br>94.33<br>(UNINEXT)<br>91.46<br>(UNINEXT)||88.77<br>(ONE-PEACE)<br>92.21<br>(ONE-PEACE)<br>83.23<br>(ONE-PEACE)|89.22<br>(ONE-PEACE)<br>89.37<br>(UNINEXT-H)|



Table 2: Results on VG benchmarks, where the specialist SOTAs are quoted from (Bai et al., 2023). 

**Results** As shown in Figure 2, CogCoM achieves the best performance in 6 out of all 8 sub-sets. Based on the training with a mixture of broad capabilities, this result indicates that our model exhibits a superior grounding abilities while offers potential to solve a variety of tasks. 

## **5.3 Experiments on General Multimodal Evaluation and Hallucination Examination** 

We further examine the general multimodal capabilities, and the hallucination issue. We use the generalist VQA model and obtain model predictions by directly asking the original questions in benchmarks. We use the challenging adversarial version and official evaluation scripts for POPE. 

|**Method**|**LLM**|**MM-Vet**|**POPE**_adv_|
|---|---|---|---|
|InstructBLIP (Dai et al., 2023)|Vicuna-13B|25.6|77.3|
|LLaVA (Liu et al., 2023b)|LLaMA2-7B|28.1|66.3|
|DreamLLM (Dong et al., 2023)|Vicuna-7B|35.9|76.5|
|LLaVA-1.5 (Liu et al., 2023a)|Vicuna-13B|36.3|84.5|
|CogVLM (Wang et al., 2023b)|Vicuna-7B|45.5_†_|87.2|
|**CogCoM**|Vicuna-7B|**46.1**|**87.8**|



Table 3: Evaluation results on the general and hallucination assessment benchmarks. 

**Results** As shown in Table 3, we can see that CogCoM improves the performance by 0.6 points compared to the baseline model on MM-Vet, and achieves the superior performance on POPE which is in consistent with the baseline model. This result suggests that out model maintains superior reasoning capabilities while preserving effectiveness in general multimodal tasks, and simultaneously exhibits lower hallucination. 

## **6 Conclusion** 

This paper studies the problems presented by the conclusive alignment training of VLMs, and proposes a mechanism, Chain of Manipulations (CoM), that enables VLMs to solve problems stepby-step by actively manipulating visual inputs as evidence. We realize this methodology by proposing (1) a flexible data structure, (2) an efficient data generation framework capable of producing abundant samples, (3) a memory-based architecture compatible with existing VLMs, and (4) a training process for versatile capabilities. We also annotate 6K graphical math samples with reasoning chains to facilitate the advancement of VLMs in solving mathematical problems. Experiments on 9 public benchmarks show that our trained 17B general VLM can produce informative reasoning content while achieving superior performance on diverse multimodal problems. 

9 

## **References** 

Acharya, M., Kafle, K., and Kanan, C. Tallyqa: Answering complex counting questions. In _Proceedings of the AAAI conference on artificial intelligence_ , 2019. 

- Alayrac, J.-B., Donahue, J., Luc, P., Miech, A., Barr, I., Hasson, Y., Lenc, K., Mensch, A., Millican, K., Reynolds, M., et al. Flamingo: a visual language model for few-shot learning. _Advances in Neural Information Processing Systems_ , 2022. 

- Antol, S., Agrawal, A., Lu, J., Mitchell, M., Batra, D., Zitnick, C. L., and Parikh, D. Vqa: Visual question answering. In _Proceedings of the IEEE international conference on computer vision_ , 2015. 

- Awadalla, A., Gao, I., Gardner, J., Hessel, J., Hanafy, Y., Zhu, W., Marathe, K., Bitton, Y., Gadre, S., Sagawa, S., et al. Openflamingo: An open-source framework for training large autoregressive vision-language models. _arXiv preprint_ , 2023. 

- Bai, J., Bai, S., Yang, S., Wang, S., Tan, S., Wang, P., Lin, J., Zhou, C., and Zhou, J. Qwen-vl: A frontier large vision-language model with versatile abilities. _arXiv preprint_ , 2023. 

- Biten, A. F., Tito, R., Mafla, A., Gomez, L., Rusinol, M., Valveny, E., Jawahar, C., and Karatzas, D. Scene text visual question answering. In _Proceedings of the IEEE/CVF international conference on computer vision_ , 2019. 

- Byeon, M., Park, B., Kim, H., Lee, S., Baek, W., and Kim, S. Coyo-700m: Image-text pair dataset, 2022. 

- Changpinyo, S., Sharma, P., Ding, N., and Soricut, R. Conceptual 12m: Pushing web-scale image-text pre-training to recognize long-tail visual concepts. In _Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition_ , 2021. 

- Chen, D., Liu, J., Dai, W., and Wang, B. Visual instruction tuning with polite flamingo. _arXiv preprint arXiv:2307.01003_ , 2023a. 

- Chen, K., Zhang, Z., Zeng, W., Zhang, R., Zhu, F., and Zhao, R. Shikra: Unleashing multimodal llm’s referential dialogue magic. _arXiv preprint_ , 2023b. 

- Chen, L., Li, J., Dong, X., Zhang, P., He, C., Wang, J., Zhao, F., and Lin, D. Sharegpt4v: Improving large multi-modal models with better captions. _arXiv preprint arXiv:2311.12793_ , 2023c. 

- Chen, X., Wang, X., Changpinyo, S., Piergiovanni, A., Padlewski, P., Salz, D., Goodman, S., Grycner, A., Mustafa, B., Beyer, L., et al. Pali: A jointly-scaled multilingual language-image model. In _The Eleventh International Conference on Learning Representations_ , 2022. 

- Chen, X., Djolonga, J., Padlewski, P., Mustafa, B., Changpinyo, S., Wu, J., Ruiz, C. R., Goodman, S., Wang, X., Tay, Y., et al. Pali-x: On scaling up a multilingual vision and language model. _arXiv preprint_ , 2023d. 

- Chiang, W.-L., Li, Z., Lin, Z., Sheng, Y., Wu, Z., Zhang, H., Zheng, L., Zhuang, S., Zhuang, Y., Gonzalez, J. E., et al. Vicuna: An open-source chatbot impressing gpt-4 with 90%* chatgpt quality. _See https://vicuna. lmsys. org (accessed 14 April 2023)_ , 2023. 

- Dai, W., Li, J., Li, D., Tiong, A., Zhao, J., Wang, W., Li, B., Fung, P., and Hoi, S. Instructblip: Towards general-purpose vision-language models with instruction tuning. arxiv 2023. _arXiv preprint arXiv:2305.06500_ . 

- Dai, W., Li, J., Li, D., Tiong, A., Zhao, J., Wang, W., Li, B., Fung, P., and Hoi, S. Instructblip: Towards general-purpose vision-language models with instruction tuning. arxiv 2023. _arXiv preprint_ , 2023. 

- Dong, R., Han, C., Peng, Y., Qi, Z., Ge, Z., Yang, J., Zhao, L., Sun, J., Zhou, H., Wei, H., et al. Dreamllm: Synergistic multimodal comprehension and creation. _arXiv preprint arXiv:2309.11499_ , 2023. 

10 

- Du, Y., Li, C., Guo, R., Yin, X., Liu, W., Zhou, J., Bai, Y., Yu, Z., Yang, Y., Dang, Q., et al. Pp-ocr: A practical ultra lightweight ocr system. _arXiv preprint arXiv:2009.09941_ , 2020. 

- Hong, W., Wang, W., Lv, Q., Xu, J., Yu, W., Ji, J., Wang, Y., Wang, Z., Dong, Y., Ding, M., et al. Cogagent: A visual language model for gui agents. _arXiv preprint arXiv:2312.08914_ , 2023. 

- Hooper, C., Kim, S., Mohammadzadeh, H., Mahoney, M. W., Shao, Y. S., Keutzer, K., and Gholami, A. Kvquant: Towards 10 million context length llm inference with kv cache quantization. _arXiv preprint arXiv:2401.18079_ , 2024. 

- Huang, K.-H., Zhou, M., Chan, H. P., Fung, Y. R., Wang, Z., Zhang, L., Chang, S.-F., and Ji, H. Do lvlms understand charts? analyzing and correcting factual errors in chart captioning. _arXiv preprint arXiv:2312.10160_ , 2023a. 

- Huang, S., Dong, L., Wang, W., Hao, Y., Singhal, S., Ma, S., Lv, T., Cui, L., Mohammed, O. K., Liu, Q., et al. Language is not all you need: Aligning perception with language models. _arXiv preprint_ , 2023b. 

- Hudson, D. A. and Manning, C. D. Gqa: A new dataset for real-world visual reasoning and compositional question answering. In _Proceedings of the IEEE/CVF conference on computer vision and pattern recognition_ , 2019. 

- Hwang, A., Head, A., and Callison-Burch, C. Grounded intuition of gpt-vision’s abilities with scientific images. _arXiv preprint_ , 2023. 

- Jia, C., Yang, Y., Xia, Y., Chen, Y.-T., Parekh, Z., Pham, H., Le, Q., Sung, Y.-H., Li, Z., and Duerig, T. Scaling up visual and vision-language representation learning with noisy text supervision. In _International conference on machine learning_ , 2021. 

- Keys, R. Cubic convolution interpolation for digital image processing. _IEEE transactions on acoustics, speech, and signal processing_ , 1981. 

- Krishna, R., Zhu, Y., Groth, O., Johnson, J., Hata, K., Kravitz, J., Chen, S., Kalantidis, Y., Li, L.-J., Shamma, D. A., et al. Visual genome: Connecting language and vision using crowdsourced dense image annotations. _International journal of computer vision_ , 2017. 

- Li, B., Zhang, Y., Chen, L., Wang, J., Yang, J., and Liu, Z. Otter: A multi-modal model with in-context instruction tuning. _arXiv preprint arXiv:2305.03726_ , 2023a. 

- Li, J., Li, D., Savarese, S., and Hoi, S. Blip-2: Bootstrapping language-image pre-training with frozen image encoders and large language models. _arXiv preprint_ , 2023b. 

- Li, Y., Du, Y., Zhou, K., Wang, J., Zhao, W. X., and Wen, J.-R. Evaluating object hallucination in large vision-language models. _arXiv preprint arXiv:2305.10355_ , 2023c. 

- Li, Y., Zhang, C., Yu, G., Wang, Z., Fu, B., Lin, G., Shen, C., Chen, L., and Wei, Y. Stablellava: Enhanced visual instruction tuning with synthesized image-dialogue data. _arXiv preprint arXiv:2308.10253_ , 2023d. 

- Lin, T.-Y., Maire, M., Belongie, S., Hays, J., Perona, P., Ramanan, D., Dollár, P., and Zitnick, C. L. Microsoft coco: Common objects in context. In _Computer Vision–ECCV 2014: 13th European Conference, Zurich, Switzerland, September 6-12, 2014, Proceedings, Part V 13_ , 2014. 

- Liu, H., Li, C., Li, Y., and Lee, Y. J. Improved baselines with visual instruction tuning. _arXiv preprint arXiv:2310.03744_ , 2023a. 

- Liu, H., Li, C., Wu, Q., and Lee, Y. J. Visual instruction tuning. _arXiv preprint_ , 2023b. 

- Liu, S., Zeng, Z., Ren, T., Li, F., Zhang, H., Yang, J., Li, C., Yang, J., Su, H., Zhu, J., et al. Grounding dino: Marrying dino with grounded pre-training for open-set object detection. _arXiv preprint arXiv:2303.05499_ , 2023c. 

- Lu, P., Bansal, H., Xia, T., Liu, J., Li, C., Hajishirzi, H., Cheng, H., Chang, K.-W., Galley, M., and Gao, J. Mathvista: Evaluating mathematical reasoning of foundation models in visual contexts. _arXiv preprint arXiv:2310.02255_ , 2023. 

11 

- Mao, J., Huang, J., Toshev, A., Camburu, O., Yuille, A. L., and Murphy, K. Generation and comprehension of unambiguous object descriptions. In _Proceedings of the IEEE conference on computer vision and pattern recognition_ , 2016. 

- Masry, A., Long, D. X., Tan, J. Q., Joty, S., and Hoque, E. Chartqa: A benchmark for question answering about charts with visual and logical reasoning. _arXiv preprint arXiv:2203.10244_ , 2022. 

- OpenAI, R. Gpt-4 technical report. arxiv 2303.08774. _View in Article_ , 2023a. 

- OpenAI, R. Gpt-4v(ision) system card. _Citekey: gptvision._ , 2023b. 

- Ordonez, V., Kulkarni, G., and Berg, T. Im2text: Describing images using 1 million captioned photographs. _Advances in neural information processing systems_ , 2011. 

- Papineni, K., Roukos, S., Ward, T., and Zhu, W.-J. Bleu: a method for automatic evaluation of machine translation. In _Proceedings of the 40th annual meeting of the Association for Computational Linguistics_ , 2002. 

- Peng, Z., Wang, W., Dong, L., Hao, Y., Huang, S., Ma, S., and Wei, F. Kosmos-2: Grounding multimodal large language models to the world. _arXiv preprint_ , 2023. 

- Schuhmann, C., Vencu, R., Beaumont, R., Kaczmarczyk, R., Mullis, C., Katta, A., Coombes, T., Jitsev, J., and Komatsuzaki, A. Laion-400m: Open dataset of clip-filtered 400 million image-text pairs. _arXiv preprint arXiv:2111.02114_ , 2021. 

- Schuhmann, C., Beaumont, R., Vencu, R., Gordon, C., Wightman, R., Cherti, M., Coombes, T., Katta, A., Mullis, C., Wortsman, M., et al. Laion-5b: An open large-scale dataset for training next generation image-text models. _Advances in Neural Information Processing Systems_ , 2022. 

- Sharma, P., Ding, N., Goodman, S., and Soricut, R. Conceptual captions: A cleaned, hypernymed, image alt-text dataset for automatic image captioning. In _Proceedings of the 56th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)_ , 2018. 

- Shazeer, N. Glu variants improve transformer. _arXiv preprint arXiv:2002.05202_ , 2020. 

- Shrestha, R., Kafle, K., and Kanan, C. Answer them all! toward universal visual question answering models. In _Proceedings of the IEEE/CVF conference on computer vision and pattern recognition_ , 2019. 

- Singh, A., Natarajan, V., Shah, M., Jiang, Y., Chen, X., Batra, D., Parikh, D., and Rohrbach, M. Towards vqa models that can read. In _Proceedings of the IEEE/CVF conference on computer vision and pattern recognition_ , 2019. 

- Sun, Q., Fang, Y., Wu, L., Wang, X., and Cao, Y. Eva-clip: Improved training techniques for clip at scale. _arXiv preprint arXiv:2303.15389_ , 2023a. 

- Sun, Q., Yu, Q., Cui, Y., Zhang, F., Zhang, X., Wang, Y., Gao, H., Liu, J., Huang, T., and Wang, X. Generative pretraining in multimodality. _arXiv preprint_ , 2023b. 

- Trinh, T. H., Wu, Y., Le, Q. V., He, H., and Luong, T. Solving olympiad geometry without human demonstrations. _Nature_ , 625(7995):476–482, 2024. 

- Wang, J., Yang, Z., Hu, X., Li, L., Lin, K., Gan, Z., Liu, Z., Liu, C., and Wang, L. Git: A generative image-to-text transformer for vision and language. _arXiv preprint arXiv:2205.14100_ , 2022a. 

- Wang, P., Yang, A., Men, R., Lin, J., Bai, S., Li, Z., Ma, J., Zhou, C., Zhou, J., and Yang, H. Ofa: Unifying architectures, tasks, and modalities through a simple sequence-to-sequence learning framework. 2022b. 

- Wang, W., Chen, Z., Chen, X., Wu, J., Zhu, X., Zeng, G., Luo, P., Lu, T., Zhou, J., Qiao, Y., et al. Visionllm: Large language model is also an open-ended decoder for vision-centric tasks. _arXiv preprint_ , 2023a. 

- Wang, W., Lv, Q., Yu, W., Hong, W., Qi, J., Wang, Y., Ji, J., Yang, Z., Zhao, L., Song, X., et al. Cogvlm: Visual expert for pretrained language models. _arXiv preprint_ , 2023b. 

12 

- Wang, Y., Kordi, Y., Mishra, S., Liu, A., Smith, N. A., Khashabi, D., and Hajishirzi, H. Self-instruct: Aligning language model with self generated instructions. _arXiv preprint arXiv:2212.10560_ , 2022c. 

- Wu, P. and Xie, S. V*: Guided visual search as a core mechanism in multimodal llms. _arXiv preprint arXiv:2312.14135_ , 2023. 

- Xu, Z., Shen, Y., and Huang, L. Multiinstruct: Improving multi-modal zero-shot learning via instruction tuning. _arXiv preprint arXiv:2212.10773_ , 2022. 

- Yin, S., Fu, C., Zhao, S., Xu, T., Wang, H., Sui, D., Shen, Y., Li, K., Sun, X., and Chen, E. Woodpecker: Hallucination correction for multimodal large language models. _arXiv preprint arXiv:2310.16045_ , 2023. 

- Yu, L., Poirson, P., Yang, S., Berg, A. C., and Berg, T. L. Modeling context in referring expressions. In _Computer Vision–ECCV 2016: 14th European Conference, Amsterdam, The Netherlands, October 11-14, 2016, Proceedings, Part II 14_ , 2016. 

- Yu, Q., Li, J., Wei, L., Pang, L., Ye, W., Qin, B., Tang, S., Tian, Q., and Zhuang, Y. Hallucidoctor: Mitigating hallucinatory toxicity in visual instruction data. _arXiv preprint arXiv:2311.13614_ , 2023a. 

- Yu, W., Yang, Z., Li, L., Wang, J., Lin, K., Liu, Z., Wang, X., and Wang, L. Mm-vet: Evaluating large multimodal models for integrated capabilities. _arXiv preprint arXiv:2308.02490_ , 2023b. 

- Zeng, Y., Zhang, H., Zheng, J., Xia, J., Wei, G., Wei, Y., Zhang, Y., and Kong, T. What matters in training a gpt4-style language model with multimodal inputs? _arXiv preprint arXiv:2307.02469_ , 2023. 

- Zhang, S., Sun, P., Chen, S., Xiao, M., Shao, W., Zhang, W., Chen, K., and Luo, P. Gpt4roi: Instruction tuning large language model on region-of-interest. _arXiv preprint arXiv:2307.03601_ , 2023a. 

- Zhang, Y., Zhang, R., Gu, J., Zhou, Y., Lipka, N., Yang, D., and Sun, T. Llavar: Enhanced visual instruction tuning for text-rich image understanding. _arXiv preprint_ , 2023b. 

13 

## **A Related Works** 

## **A.1 Large Vision-Langauge Models as Foundations** 

Most of LVLMs rely on the training on publicly available image-caption pairs, including ALIGN (Jia et al., 2021), MSCOCO (Lin et al., 2014), VG Krishna et al. (2017), CC3M Sharma et al. (2018), CC12M (Changpinyo et al., 2021), SBU (Ordonez et al., 2011), LAION2B (Schuhmann et al., 2022), LAION400M Schuhmann et al. (2021). Starting from Flamingo (Alayrac et al., 2022), a series of LVLMs have focused on training the adaptation layers to align the visual representation to the frozen LLMs on a mixture of image-text pairs with the above corpus, including BLIP2 Li et al. (2023b), KOSMOS Huang et al. (2023b), and OpenFlamingo (Awadalla et al., 2023). Inspired by success of instruction tuning in LLMs (Wang et al., 2022c), a line of works have devoted efforts to build visionoriented instruction-answer pairs through GPT4 and train models for imitation, such as LLAVA (Liu et al., 2023b), Otter (Li et al., 2023a), VisionLLM (Wang et al., 2023a), MultiInstruct (Xu et al., 2022), Lynx (Zeng et al., 2023), InstructBLIP (Dai et al.), CleverFlamingo (Chen et al., 2023a) and StableLLaVA (Li et al., 2023d). Recently, researchers have proven the efficiency of developing LVLMs with two stages of training, the first stage of abundant pretraining on image-caption pairs and the second stage of alignment on image-question-answer triples, such as PALI (Chen et al., 2022), PaLI-X (Chen et al., 2023d), Qwen-VL (Bai et al., 2023), and CogVLM Wang et al. (2023b). 

## **A.2 Large Vision-Language Models with Reasoning** 

To further enhance the ability of LVLMs in solving high-level visual problems, research focusing on various aspects of reasoning is attracting broad attention. We simply divide existing studies into tree broad categories. The first line of research focus on enhance train models with a mastery of cross-modal grounded reasoning, where grounded instruction-following supervision is build through public visual grounding dataset or GPT4-V for training, including KOSMOS-2 (Peng et al., 2023), Shikra (Chen et al., 2023b), and GPT4ROI (Zhang et al., 2023a). The second aspect of efforts have been devoted into promoting models to understand artificial visual scenes, such as figures, charts, and receipts. These studies includes CogAgent (Hong et al., 2023) and CHARTVE (Huang et al., 2023a). Some other studies address the crucial problem of hallucination in LVLMs with counterfactual or interpretable reasoning (Yu et al., 2023a; Yin et al., 2023). V* (Wu & Xie, 2023) also contributes efforts to enhance the details recognition of VLMs based the LLM-guided searching process. 

## **B Limitation and Impact** 

Though we try to develop an accurate and robust framework that engages remarkable LLM to provide basic solving steps, adopts reliable visual tools to obtain visual contents, and then acquires feasible paths based on traversal, there are still limitations in our methodology that we hope to improve in the future. First, We find that the diversity of linguistic solving steps is insufficient, and the inaccuracy of visual tools ( _e.g.,_ the rough granularity of grounding boxes, OCR failures on slant letters) will lead to a large amount of negative paths (effectively utilizing these paths would beneficial). We suggest to promote these limitations with dedicate prompts and improved visual tools. Second, our current model re-input the manipulated images with a set of hard prompts, which may bring speed losses. This is expected to be improved by implementing the physical manipuations into the calculations in vector space. This work presents a general visual reasoning mechanism that alleviate the problems caused by existing conclusion-alignment training for VLMs, introduces a data production framework involving LLMs and visual tools as reliable annotators, and devises a memory-based compatible VLM architecture. We expect this work to bring three benefits to the community. First, the proposed visual reasoning mechanism may push the progress of VLMs in solving complex visual problems. Second, the introduced data production framework may be applied to widespread training scenarios to promote the development of current data-driven machine learning. Third, we hope that the memory-based architecture will be helpful for VLMs in multi-turn long contexts. 

14 

## **C Details of Data Production** 

In this section, we further introduce the details of CoM data production, with the overall algorithm of a pseudo code, an example of the solving steps generation with LLM and corresponding guideline, an example of the reasoning chains completion with visual tools. We also list the details of data statistics for the synthesised training data as well as the evaluation data of CoM-test, followed by a limitation analysis for the current data production method. 

## **C.1 Algorithm for the Automate Data Generation Pipeline** 

We provide the pseudocode of the CoM synthesis algorithm to clearly explain the process of data generation, thereby facilitating understanding and reproduction 1. 

**Algorithm 1** Synthesising Chain of Manipulations 

 _Manipulations_ : _{fi_ : _x → y | fi ∈M}_ 1: **Define:** _Linguistic Annotator_ : Ψ _L //We use GPT_ 4 _in this work_   _V isual Annotator_ : Ψ _V //We use PaddleOCR and GroundingDINO in this work_ 

2: **Input:** Image _I_ , Question _Q_ , Answer _A_ 

3: // Linguistic Annotation 

4: Prompt Ψ _L_ with guideline _P[L]_ to generate reasoning steps: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_CogCoM_A_Visual_Language_Model_with_Chain-of-Manipulations_Reasoning_images/iclr_2025_CogCoM_A_Visual_Language_Model_with_Chain-of-Manipulations_Reasoning.pdf-0015-09.png)


5: Define tree _T_ 

6: **for** _i_ = 1 **to** _|ς|_ **do** 7: Extract _xi, yi_ instantiated with _fi_ in _stepi_ 8: Extract referential boxes _B_ from _xi_ 9: **for** _b_ in _B_ **do** 10: Leverage Ψ _V_ to acquire corresponding visual content _yi[′]_[= Ψ(] _[x][i][|][I, b]_[)][, and apply] _[ y][i]_[to] tree 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_CogCoM_A_Visual_Language_Model_with_Chain-of-Manipulations_Reasoning_images/iclr_2025_CogCoM_A_Visual_Language_Model_with_Chain-of-Manipulations_Reasoning.pdf-0015-12.png)


11: **end for** 12: **end for** 

13: Traverse _T_ to obtain positive chains that leads to given answer with terminal return 

[ _ς_ 1 _, ς_ 2 _, ..._ ] = _DFS_ ( _T |A_ ) (5) 

- 14: Return [ _ς_ 1 _, ς_ 2 _, ..._ ] 

## **C.2 The CoM-test Benchmark and Evaluation Metric** 

To measure the correctness of CoM chains, we introduce a **keypoints-aware metric** . The intuition is that we care about the key elements including actions ( _i.e.,_ manipulation name), targets ( _i.e.,_ manipulation input), and visual contents ( _i.e.,_ manipulation returns) of each step in the path, as well as the logical execution order of manipulations. Given a pair of chain-answer annotation ( _c, a_ ) and corresponding model prediction ( _c[′] , a[′]_ ), we first sequentially extract the key elements from _c_ and _c[′]_ to construct two ordered lists, and then replace the elements in the lists with their fixed indices in a Bag-of-Elements _E_ = _c ∪ c[′]_ to result in lists of _k_ and _k[′]_ . We thus calculate the score as the normalized Levenshtein Distance _sc_ = _Levenshtein_ ( _k, k[′]_ ) _/N_ between the two lists, where _N_ is the maximum length between _k_ and _k[′]_ . We adopt this simple discretization strategy with low time complexity to concentrate on the key points as well as the solving order. We further consider the linguistic matching of paragraphs by calculating the BLEU (Papineni et al., 2002) score between two chains _sp_ = BLEU( _c, c[′]_ ), and the final sore is a weighted combination as _acc_ = (0 _._ 6 _× sc_ + 0 _._ 4 _× sp_ ) _/_ 2. 

15 

## **C.3 Data Statistics** 

We develop a strategy to extract predicate phrases based constituency parsing with StandordCoreNLP, in which we extract verb, conjunction-connected verb phrase, preposition-connected verb phrase. 

Besides the standard CoM data incorporating manipulations with explicit visual evidences, the proposed data synthesising framework is compatible of producing implicit visual reasoning steps _step[′] i_[= (] _[desc][i]_[)][ without involving the manipulations.][We thereby also build this partial CoM data on] the corpus consisting of absurd visual questions ( _i.e.,_ asking unanswerable questions based on the given image) to further resist the toxic hallucinations. Specifically, given an image _I_ with a question _Q_ ,we prompt GPT-4V (OpenAI, 2023b) to solve the question step-by-step to acquire the reasoning chains. 

|**Data Source**|**#QAs**|**#Chains**|**#Steps/Chain**|**#Manipulations Types/Chain**|
|---|---|---|---|---|
|TextVQA (Biten et al., 2019)|10782|13766|2.93|2.41|
|ST-VQA (Singh et al., 2019)|4814|3959|2.88|2.43|
|TDIUC-count (Shrestha et al., 2019)|53547|54523|2.35|0.74|
|TDIUC-absurd(Shrestha et al., 2019)|11677|11677|4.09|-|
|CoM-test|4609|8612|3.26|2.18|



Table 4: Detailed statistics the the training data and evaluation data synthesised with CoM production. 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_CogCoM_A_Visual_Language_Model_with_Chain-of-Manipulations_Reasoning_images/iclr_2025_CogCoM_A_Visual_Language_Model_with_Chain-of-Manipulations_Reasoning.pdf-0016-05.png)


**----- Start of picture text -----**<br>
0.06<br>0.05<br>0.04<br>0.03<br>0.02<br>0.01<br>0.00<br>locatereadidentifydeterminelook forobserveanalyzeexaminecountzoom incomparefocus onfind usezoomextractinterpretrecognizesearch forzoom onnotelook atresearchunderstandprovidemakelistcheckenhanceconductmeasurecheck formatchanswerconsultconvertconfirmdescribeinspectcalculatelookopenstatenote downreferconsidergatherestimatetranslaterecord<br>**----- End of picture text -----**<br>


Figure 7: Distribution of the top-50 generated manipulations out of total 465 based on 4-shot prompting, where the _first three bars_ are scaled with 20% for a smooth visualization of all data. 

16 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_CogCoM_A_Visual_Language_Model_with_Chain-of-Manipulations_Reasoning_images/iclr_2025_CogCoM_A_Visual_Language_Model_with_Chain-of-Manipulations_Reasoning.pdf-0017-00.png)


**----- Start of picture text -----**<br>
OCR_i(tgt)->txt_i: i-th OCR manipulation, that recognize the natural texts written on the target `tgt`, and return the recognized texts `txt_i`.<br>calculate(tgt)->res_i: i-th calculate manipulation, that calculate the formula specified by the target `tgt` in current image, and return<br>the calculation result `res_i`.<br>grounding_i(tgt)->bbx_i:  i-th grounding manipulation, that locates the object(s) specified by the target noun phrase `tgt` in current<br>image, and return the resulting bounding box(es) as `bbx_i` where each box is represented by the top-left<br>and bottom-right coordinates.<br>crop_and_zoomin_i(bbx)->img_i:  i-th crop_and_zoomin manipulation which is useful to identify small and subtle objects in image, that first crops<br>the current image using the box `bbx` defined by the top-left and bottom-right coordinates, and then zoom in<br>the cropped image by two times and finally return the resulting image `img_i`.<br>You are a visual assistant capable of generating solving steps for image-oriented visual questions. In each step, you can optionally use a<br>manipulation to operate the image, which can be used to acquire specific information from the image or to acquire the processed new image<br>(please be aware that these manipulations will not actually be performed when you generate the solving steps). The manipulation can be one from<br>the predefined ones, or can be a new one you create yourself (should there indeed be a need), where the predefined manipulations with their<br>descriptions are listed below:\n\n{MANIPULATIONS}.\n\nGiven a question Q abount an image, please generate a series of essential solving<br>steps, where the output of each step is a tuple consisting of a Manipulation (leave it to None if the current step doesn't involve any manipulation)<br>and a Description: (1) Manipulation f(x)->y, that is the manipulation `f` targeting `x` to obtain specific information or image `y`; (2) Description,<br>which is a sentence describing the current solving step.\n Please adhere the following format: given an input of 'Q: xxx. The essential solving<br>steps are: ', the output should like 'Step 1: (Manipulation, Description); Step 2: (Manipulation, Description); ...'.\n\n There are several examples:\n<br>{DEMONSTRATIONS}\n\n Q: {QUESTION} The essential solving steps are:<br>Manipulations Definition and Linguistic Annotation Guideline<br>INPUTS Question : What number police station is on the building?<br>Step 1: grounding_1(the building)->bbx_1, Locate the building in the image and return the bounding box of the building as `bbx_1`.<br>OUTPUTSL Step 2: (grounding_2(number police station on               the building in box `bbx_1`)->bbx_2, Identify the number of the police station on the building in box `bbx_1` and return the bounding box of the number as `bbx_2`. AnnotationLinguistic<br>Step 3: (OCR_1(number in region `bbx_2`) Recognize the number in the region `bbx_2` and return the recognized number as `txt_1`.<br>               ->txt_1,<br>Question : What number police station is on the building?<br>Answer : 43<br>INPUTS<br>OUTPUTSL<br>Visual<br>bbx_1 bbx_1 bbx_2 bbx_1 bbx_2 Annotation<br>OUTPUTSV L txt_1<br>Locate the building in the image and return  Identify the number of the police station on  Recognize the number in the region `bbx_2`<br>the bounding box of the building as `bbx_1`. the building in box `bbx_1` and return the  and return the recognized number as `txt_1`.<br>bounding box of the number as `bbx_2`.<br>An Example to show the linguistic annotation results and Visual annotation results<br>MANIPULATIONS<br>REQUIREMENTS (PROMPT)<br>**----- End of picture text -----**<br>


Figure 8: An example shows the configuration, inputs, outputs of the linguistic annotation and visual annotation. 

## **C.4 Details of the Linguistic/Visual Annotations** 

In this work, we adopt the GPT4-turbo as the linguistic annotator for generating problems solving steps, and the API call was conducted during the period of 2023.9 - 2023.12. For the visual annotators, we leverage the the currently best-performing tools, GroundingDINO and PaddleOCR, to acquire all visual contents requested by the manipulations. For a clear description to the production setting and results, we illustrate the guiding prompt, and an example-based linguistic annotation results as well as the visual annotation results in Figure 8. 

## **C.5 Limitation Analysis for the Data Production** 

For the implemented data framework, we engage the remarkable LLM to provide basic solving steps, adopt two reliable visual tools ( _i.e.,_ GroundingDINO and PaddleOCR) to acquire corresponding visual contents, and then perform the traversal to achieve feasible reasoning paths, which ensures the correctness and robustness of data synthesizing. However, we also find that there are three major limitations caused by the employed models and could be improved in future: 

17 

- The lack of diversity in linguistic reasoning steps. The 5-shot prompting to the GPT-4 gains a stable solving steps, but it also results in the descriptions for executing manipulations or general thinking are similar. We suggest that this can be addressed by employing diversified prompts or requirements. 

- The inaccuracy of visual tools. We find that there are a considerable amount of negative paths caused by the failures of visual tools, such as the rough granularity of bounding boxes and the error recognition of slated letters or long sentences. This issue can be relieved by improving the semantic understanding capabilities of visual tools. 

## **D Details of Training** 

## **D.1 Launching Prompts** 

- Please solve the problem gradually via a chain of manipulations, where in each step you can selectively adopt one of the following manipulations GROUNDING(a phrase) _→_ boxes, OCR(an image or a region) _→_ texts, CROP_AND_ZOOMIN(a region on given image) _→_ new_image, CALCULATE(a computable target) _→_ numbers, or invent a new manipulation, if that seems helpful. {QUESTION} 

- Please tackle a given question in a stepbystep manner. For each step one of the following manipulations (depicted as Name(Input) _→_ Retrun) can be optionally used: GROUNDING(a phrase) _→_ boxes, OCR(an image or a region) _→_ texts, CROP_AND_ZOOMIN(a region on given image) _→_ new_image, CALCULATE(a computable target) _→_ numbers, or develop a new manipulation yourself (if it is indeed required). {QUESTION} 

- Please go through the question incrementally with chain of manipulations (optionally use manipulation when needed) such as GROUNDING(a phrase) _→_ boxes, OCR(an image or a region) _→_ texts, CROP_AND_ZOOMIN(a region on given image) _→_ new_image, CALCULATE(a computable target) _→_ numbers, and create a new manipulation if necessary. {QUESTION} 

## **D.2 Training settings** 

|**Parameters**|**Stage1-1**|**State1-2**|**Stage-2**|
|---|---|---|---|
|Hardware Environment|3,840 A100xdays|256 A100xdays|160 A100xdays|
|Objective|next token prediction|next token prediction|next token prediction|
|Images|1.5B|40M|576K|
|Batch size|8192|1024|160|
|Iterations|120,000|60000|14000|
|Optimizer|AdamW|AdamW|AdamW|
|Learning rate|1e-4|1e-5|1e-5|
|Warm up steps|7200|1200|280|
|Trainable weights|6.5B visual expert|6.5B visual expert|6.5B visual expert|



Table 5: Training details of all stages. 

## **E Details of Qualitative Analysis** 

## **E.1 Qualitative Analysis** 

We investigate the evidential reasoning capability of CogCoM on scenarios that requires different types of meticulous reasoning, including recognizing textual details, reading time, understanding charts and counting objects. The results are shown in Figure 1. The first case demonstrates that CogCoM finds the region corresponding to the plane logo through two steps of grounding and then achieves the answer based on zooming in the cropped region. The second case illustrates the ability of CogCoM in reading time, by locating the device that displays time and then transforming the time into words based on the read_timne manipulation. In the forth example, CogCoM first identifies all 

18 

visible truck wheels through grounding, and then leverages the counting manipulation to consolidate the total number. 

## **E.2 Representative Cases** 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_CogCoM_A_Visual_Language_Model_with_Chain-of-Manipulations_Reasoning_images/iclr_2025_CogCoM_A_Visual_Language_Model_with_Chain-of-Manipulations_Reasoning.pdf-0019-02.png)


_**CoM**_ **:** Given a question, please solve the question step-by-step with a chain of manipulations, where in each step you can optionally use one of the following manipulations (defined as Name(Input)->Return) on the image: GROUNDING(a phrase)->boxes, OCR(an image or a region)->texts, CROP_AND_ZOOMIN(a region on given image)->new_image, CALCULATE(a computable target)->numbers, or a new manipulation created by yourself (if it is indeed needed). _Question:_ What is written on the traffic sign near the bus? _Model:_ Using GROUNDING(the traffic sign near the bus) to locate the traffic sign near the bus in current image, which is _[[084,157,926,703;676,146,740,273]]_ . Utilizing OCR(texts in region _[[084,157,926,703]]_ ) to read the texts written on the traffic sign within region [[084,157,926,703]], resulting Taxi. So ultimately, the conclusive answer to the question in discussion is Taxi. 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_CogCoM_A_Visual_Language_Model_with_Chain-of-Manipulations_Reasoning_images/iclr_2025_CogCoM_A_Visual_Language_Model_with_Chain-of-Manipulations_Reasoning.pdf-0019-04.png)


**----- Start of picture text -----**<br>
Visual Grounding:<br>Question:  Find the region in image that "a traffic sign" describes.<br>Model:  [[678,145,739,270]<br>Grounded Captioning:<br>Question:  The goal of this task is to generate description for one part of the image. The part is specified by [[678,145,739,270]].<br>Model: A no tax sign<br>Image Captioning:<br>Question:  In this task, you are given an image and you will need to generate some text to describe it. Please give me a concise answer.<br>Answer: Taxi bus with rainbow colors on side.<br>Multi Choice:<br>Question:  Select the color that does not appear in the rainbow sign of this bus. (1) red. (2) yellow. (3) pink. (4) black. Please give me a direct answer.<br>Answer: black<br>Detailed Captioning:<br> Question:  Explain the visual content of the image in great detail.<br>Answer: The image captures a scene on a city street, where a white bus with red and blue accents is parked. The bus bears the text "Trafalgar" in bold red<br>letters across its side, indicating it's likely part of a tourist or travel company. The bus is positioned next to a traffic sign that reads "TAXI", suggesting it might<br>be waiting for passengers or ready to embark on a journey. In the background, there are buildings and trees, adding depth to the urban landscape. The overall<br>impression is one of a typical day in an urban setting, with a bus waiting for its passengers amidst the hustle and bustle of city life.<br>**----- End of picture text -----**<br>


Figure 9: CogCoM demonstrates the flexible capabilities for adapting to different multimodal scenarios, including evidential visual reasoning, Visual Grounding, Grounded Captioning, Image Captioning, Multi Choice, and Detailed Captioning. 

19 

