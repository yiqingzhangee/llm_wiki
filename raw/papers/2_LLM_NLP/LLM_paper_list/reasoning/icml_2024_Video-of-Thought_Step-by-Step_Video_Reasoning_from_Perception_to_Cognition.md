_**Video-of-Thought**_ **: Step-by-Step Video Reasoning from Perception to Cognition** 

**Hao Fei**[1] **Shengqiong Wu**[1] **Wei Ji**[1] **Hanwang Zhang**[2] **Meishan Zhang**[3] **Mong Li Lee**[1] **Wynne Hsu**[1] 

## **Abstract** 

Existing research of video understanding still struggles to achieve in-depth comprehension and reasoning in complex videos, primarily due to the under-exploration of two key bottlenecks: _fine-grained spatial-temporal perceptive understanding_ and _cognitive-level video scene comprehension_ . This paper bridges the gap by presenting a novel solution. We first introduce a novel video Multimodal Large Language Model (MLLM), **MotionEpic** , which achieves fine-grained pixel-level spatial-temporal video grounding by integrating video spatial-temporal scene graph (STSG) representation. Building upon MotionEpic, we then develop a Video-ofThought ( **VoT** ) reasoning framework. VoT inherits the Chain-of-Thought (CoT) core, breaking down a complex task into simpler and manageable sub-problems, and addressing them step-bystep from a low-level pixel perception to highlevel cognitive interpretation. Extensive experiments across various complex video QA benchmarks demonstrate that our overall framework strikingly boosts existing state-of-the-art. To our knowledge, this is the first attempt at successfully implementing the CoT technique for achieving human-level video reasoning, where we show great potential in extending it to a wider range of video understanding scenarios. Project is open at https://haofei.vip/VoT. 

## **1. Introduction** 

Enabling learning models to accurately interpret video data is one of the most paramount goals in the relevant community. In the current research, while there has been extensive exploration into building models for video action and dynamics recognition (Lei et al., 2018; Bertasius 

1National University of Singapore, Singapore 2Nanyang Technological University, Singapore[3] Harbin Institute of Technology (Shenzhen), China. Correspondence to: Meishan Zhang _<_ zhangmeishan@hit.edu.cn _>_ . 

_Proceedings of the 41[st] International Conference on Machine Learning_ , Vienna, Austria. PMLR 235, 2024. Copyright 2024 by the author(s). 

**Question:** What will happen to _the red oil tanker truck_ ? 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/icml_2024_Video-of-Thought_Step-by-Step_Video_Reasoning_from_Perception_to_Cognition_images/icml_2024_Video-of-Thought_Step-by-Step_Video_Reasoning_from_Perception_to_Cognition.pdf-0001-10.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/icml_2024_Video-of-Thought_Step-by-Step_Video_Reasoning_from_Perception_to_Cognition_images/icml_2024_Video-of-Thought_Step-by-Step_Video_Reasoning_from_Perception_to_Cognition.pdf-0001-11.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/icml_2024_Video-of-Thought_Step-by-Step_Video_Reasoning_from_Perception_to_Cognition_images/icml_2024_Video-of-Thought_Step-by-Step_Video_Reasoning_from_Perception_to_Cognition.pdf-0001-12.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/icml_2024_Video-of-Thought_Step-by-Step_Video_Reasoning_from_Perception_to_Cognition_images/icml_2024_Video-of-Thought_Step-by-Step_Video_Reasoning_from_Perception_to_Cognition.pdf-0001-13.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/icml_2024_Video-of-Thought_Step-by-Step_Video_Reasoning_from_Perception_to_Cognition_images/icml_2024_Video-of-Thought_Step-by-Step_Video_Reasoning_from_Perception_to_Cognition.pdf-0001-14.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/icml_2024_Video-of-Thought_Step-by-Step_Video_Reasoning_from_Perception_to_Cognition_images/icml_2024_Video-of-Thought_Step-by-Step_Video_Reasoning_from_Perception_to_Cognition.pdf-0001-15.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/icml_2024_Video-of-Thought_Step-by-Step_Video_Reasoning_from_Perception_to_Cognition_images/icml_2024_Video-of-Thought_Step-by-Step_Video_Reasoning_from_Perception_to_Cognition.pdf-0001-16.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/icml_2024_Video-of-Thought_Step-by-Step_Video_Reasoning_from_Perception_to_Cognition_images/icml_2024_Video-of-Thought_Step-by-Step_Video_Reasoning_from_Perception_to_Cognition.pdf-0001-17.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/icml_2024_Video-of-Thought_Step-by-Step_Video_Reasoning_from_Perception_to_Cognition_images/icml_2024_Video-of-Thought_Step-by-Step_Video_Reasoning_from_Perception_to_Cognition.pdf-0001-18.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/icml_2024_Video-of-Thought_Step-by-Step_Video_Reasoning_from_Perception_to_Cognition_images/icml_2024_Video-of-Thought_Step-by-Step_Video_Reasoning_from_Perception_to_Cognition.pdf-0001-19.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/icml_2024_Video-of-Thought_Step-by-Step_Video_Reasoning_from_Perception_to_Cognition_images/icml_2024_Video-of-Thought_Step-by-Step_Video_Reasoning_from_Perception_to_Cognition.pdf-0001-20.png)


**----- Start of picture text -----**<br>
2 Tracking objects temporally<br>blue truck red oil truck car<br>1 Determining the target fine-grainedly collided<br>Oil is a highly flammable<br>Answer: substance.  A collision<br>could lead to a rupture in<br>Being on fire  the tanker's storage, and if<br>and explosion. there's an ignition source,...Commonsense red oil truck car<br>4 In-depth reasoning with commonsense 3 Understanding action scene semantics<br>**----- End of picture text -----**<br>


Figure 1: Human-like video reasoning intuitively follows a multi-step procedure, from lower-level perceptive finegrained pixel grounding and tracking, to higher-level cognitive action scene semantics understanding. 

et al., 2021), mostly they fall prey to the type of straightforward perceptual-level understanding, i.e., for simple videos (Zolfaghari et al., 2018; Lin et al., 2019). And there remains a significant gap in research concerning comprehending and reasoning about complex videos in depth, an imperative capability urgently needed in real-world applications. Compared to shallow video perception, reasoning about complex videos poses greater challenges: it demands not only an intricate understanding of the video’s spatiotemporal characteristics (Caballero et al., 2017), but also a profound grasp of the underlying implications behind pixels. 

Drawing from human cognition patterns, we mark that reasoning about videos, especially for the complex ones, requires superior mastery in two points: perceptual capability of pixel understanding and cognitive ability for semantic understanding. **Firstly** , to achieve precise content perception, a fine-grained perceptive pixel understanding of the video movement is necessary. Most existing video understanding approaches focus on instance or patch-level analysis (Yuan et al., 2021; Neimark et al., 2021), lacking the precision for detailed granular control and accurate object-level recognition or tracking, let alone in-depth video comprehension. **Secondly** , profound reasoning demands cognitive capabilities allowing reasonable explanation and even causal imagination, i.e., with a reservoir of commonsense knowledge to link video pixels to the factual world. For example, understanding that jumping from a height can cause fractures, or that colliding with a tanker truck can cause an explosion. 

1 

**Video-of-Thought: Step-by-Step Video Reasoning from Perception to Cognition** 

**Most importantly** , for humans, video reasoning is not an instantaneous process but follows a multi-hop procedure from lower level to higher level. This often involves first identifying specific targets, like a “red oil truck” (cf. Fig. 1) in the video frames, then tracking and analyzing its temporal behaviors and interactions with the environment to deduce the scene semantics, and finally, integrating factual commonsense to formulate a cognitively coherent response. 

Recently, the community of MLLMs has seen rapid advancement, exhibiting formidable data understanding and reasoning capabilities, among which video MLLMs have been extensively developed, such as Video-LLaMA (Zhang et al., 2023a), Video-ChatGPT (Maaz et al., 2023), and Video-LLaVA (Lin et al., 2023). Simultaneously, there is a growing interest in integrating CoT prompting technique (Wei et al., 2022) to augment the reasoning capabilities of LLMs. CoT works by intuitively breaking down a complex problem into a chain of simpler and more manageable sub-problems, facilitating a human-like reasoning process. While this technique has flourished in language understanding tasks extensively (Wang et al., 2022a), unfortunately, a CoT-based reasoning framework specifically tailored for video input with video MLLMs is yet under-explored. 

To this end, this paper is dedicated to devising a solution that enables human-like complex video reasoning. We first propose the integration of a STSG representation (Ji et al., 2020), modeling both the input video and its STSG representation, where fine-grained spatial-temporal features are carefully integrated and modeled. To implement this, we introduce a novel video LLM, named **MotionEpic** (cf. Fig. 2), which, based on a similar architecture as existing video MLLMs, supports not only video input but also the encoding, understanding and generation of STSGs. To enable MotionEpic with fine-grained pixel-level spatial-temporal grounding between videos and STSGs, we also investigate various distinct video-STSG training objects. STSG annotations are used during the grounding-aware tuning phase, while in the subsequent stage, the system is learned to autonomously parse STSG, and thus supports STSG-free inference and reasoning for downstream tasks. 

Building upon MotionEpic, we next design a novel reasoning framework, named Video-of-Thought ( **VoT** ), cf. Fig. 4. Inheriting the key spirit of CoT, VoT breaks down the raw intricate video reasoning problem into a chain of simpler sub-problems, and solves them one by one sequentially. These sub-questions follow a progression from lower to higher level, i.e., starting with pixel grounding for a precise understanding of target content, and then accurately interpreting corresponding semantic signals. 1 Given an input video and a question, VoT identifies the possible target(s) involved in the question to observe. 2 The system then grounds the temporal tracklet(s), which serves as supporting 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/icml_2024_Video-of-Thought_Step-by-Step_Video_Reasoning_from_Perception_to_Cognition_images/icml_2024_Video-of-Thought_Step-by-Step_Video_Reasoning_from_Perception_to_Cognition.pdf-0002-05.png)


**----- Start of picture text -----**<br>
Textual Response Grounded (partial) STSG Expression<br>Large Language Model Lora<br>Projector<br>Text Prompt Encoder STSG Encoder<br>Video Spatial-temporal Scene Graph<br>(STSG)<br>Recycling<br>**----- End of picture text -----**<br>


Figure 2: Overview of the MotionEpic video MLLM. 

evidence/rationale for content perception in subsequent analysis. 3 Combined with factual commonsense, VoT next interprets the target object’s trajectory and its interactions with neighboring scenes to thoroughly understand the action dynamics and semantics. 4 With in-depth understanding of the target actions in the video, we then carefully examine each optional answer with commonsense knowledge, where the final result is output after ranking those candidates. 5 Finally, VoT performs verification for the answer from both pixel grounding perception and commonsense cognition perspectives, ensuring the most factually accurate result. 

Our experiments mainly focus on video Question Answering (QA), a representative task reliant on in-depth video reasoning. We evaluate our system across 8 complex video QA benchmarks, where it strikingly boosts the current performances in both fine-tuning and zero-shot settings by very clear margins, establishing a series of new states of the arts. We further conduct in-depth analyses of MotionEpic’s capabilities in video grounding, and probe the video reasoning ability of VoT framework, providing insights into how the framework advances. To summarize, this work contributes in multiple aspects: 

- proposing the first video Chain-of-Thought reasoning framework, VoT, which decomposes raw complex problems into a chain of sub-problems, and reasons through multiple steps from low to high levels, enabling not only pixel perceptive recognition but also semantic cognitive understanding of videos. 

- contributing a novel video MLLM, MotionEpic, which supports fine-grained pixel-level spatial-temporal video grounding via STSG encoding and generation. 

- empirically setting new state-of-the-art (SoTA) performances in a range of video QA benchmarks that require intricate reasoning capability. 

## **2. Related Work** 

A key objective in the intelligence community is the understanding of various modalities of data. Currently, with the advent of LLMs such as ChatGPT (OpenAI, 2022a), we 

2 

**Video-of-Thought: Step-by-Step Video Reasoning from Perception to Cognition** 

have attained unprecedented language reasoning capabilities, on par with the human level. This is largely due to the vast repository of commonsense knowledge and semantic understanding capabilities inherent in LLMs, enabling provide plausible causal explanations and even engage in imaginative reasoning. Particularly, the integration of the recent trending CoT technology, which deconstructs a problem into its constituent parts and provides rationale at each step, has made the reasoning process more reliable. As for image understanding, the rapid development of MLLMs, e.g., LLaVA (Liu et al., 2023), GPT-4V (OpenAI, 2022b), has also nearly achieved substantial comprehension ability. However, unlike language and images, video understanding or reasoning presents a dual challenge of static spatial and temporal dynamics. 

Historically, earlier video understanding research efforts predominantly learn neural models over small-size in-domain training datasets (Zolfaghari et al., 2018; Lin et al., 2019). However, these ‘small’ models are limited to relatively superficial levels of perception, lacking the depth of humanlevel cognition. As a result, previous methods were mostly confined to the shallow understanding of simple videos, such as identifying contents and movements within a video. Unlike simple video comprehension, which relies mainly on perceptive abilities, understanding complex videos necessitates deeper cognitive reasoning, such as _explaining why certain actions occur in a video or hypothesizing potential outcomes_ . Although MLLMs supporting video data have been developed (Li et al., 2023c; Zhang et al., 2023a; Wu et al., 2023c), offering greater video understanding capabilities than smaller models, the research into penetrating beyond the perceptual surface of videos to deeply understand the implied semantic content and perform cognitive-level reasoning is still insufficiently explored. We observe that current video MLLMs either fail to achieve fine-grained spatial-temporal understanding of videos, or do not fully leverage the rich commonsense knowledge and causal reasoning inherent in LLMs for enhanced cognitive-level comprehension. To enable MLLMs with spatiotemporal modeling, we consider employing the dynamic video scene graph representation (Ji et al., 2020). SGs (Johnson et al., 2018) are characterized by highly structured graph representations (Fei et al., 2022), which intrinsically depict the underlying semantic implications of the data, and thus have been extensively integrated into a wide range of downstream crossmodal tasks (Zhao et al., 2023b; Wu et al., 2024; Fei et al., 2023b; Wu et al., 2023b;a), especially in video modeling (Zhao et al., 2023a; Fei et al., 2023c; 2024). 

Meanwhile, recent advancements in CoT technology have made significant strides in enhancing the reasoning capabilities of LLMs (Wei et al., 2022; Zhang et al., 2022; Fei et al., 2023a; Zheng et al., 2024). While there are efforts enhancing multimodal reasoning with multimodal CoT (Lu et al., 

2022; Zhang et al., 2023c), we still note a lack of research specifically focused on integrating CoT into video scenarios to establish a powerful video reasoning framework. To bridge this gap, this paper takes the initiative and introduces the concept of Video-of-Thought. Unlike the original CoT approach that attempts to improve outputs with a simple “ _Let’s Think Step By Step_ ” prompt (Wei et al., 2022), we implement a more genuine thought chain. We encourage MLLM to first decompose the original problem into a series of more manageable sub-solutions before the model initiates reasoning, following the human-cognitive procedure from low-level pixel grounding and understanding to high-level cognitive semantic meaning inference, ultimately achieving human-level video understanding and reasoning capabilities. 

## **3. MotionEpic: Fine-grained Spatial-temporal Grounded Video MLLM** 

In this section, we describe the MotionEpic video MLLM, and elaborate on how the STSGs are integrated, as well as the fine-grained spatial-temporal grounding-aware tuning. 

## **3.1. Architecture Briefing** 

Fig. 2 presents a schematic overview of MotionEpic, where MotionEpic takes as input three sources: text prompt, video, and STSG representation of video. We follow the most common practice, and employ the Vicuna-7B (v1.5) (Chiang et al., 2023) as the backbone LLM. To perceive video input, we adopt the ViT-L/14 encoder (Dosovitskiy et al., 2020) and Q-Former projector (Li et al., 2023a). We also design MotionEpic to support the STSG signal, where we retrofit the Graph Transformer (Dwivedi & Bresson, 2020) with recurrent propagation to encode the multi-frame STSG information. 

## **3.2. Integrating STSG Representation** 

By definition (Ji et al., 2020), an STSG consists of a sequence of single SGs corresponding to all video frames, with each SG comprising triplets in the video frame, i.e., ‘ _subject_ ’-‘ _predicate_ ’-‘ _object_ ’, where ‘ _subject_ ’ and ‘ _object_ ’ refer to two visual proposals (RoIs) that are connected with the ‘ _predicate_ ’ relationship. STSG intuitively depicts the underlying core semantics representations of videos while filtering the less-informative background information, aiding the perceptive understanding of videos (Zhao et al., 2023a). Also, such fine-grained structural feature helps effectively model the compositional spatiotemporal semantics. 

In our practice, we slightly retrofit the vanilla STSG definition to meet the demand in our reasoning framework. Since a video has redundant temporal contents across frames, we first evenly sample the frames (with a proper sampling rate), which can effectively reduce computation costs. We denote 

3 

**Video-of-Thought: Step-by-Step Video Reasoning from Perception to Cognition** 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/icml_2024_Video-of-Thought_Step-by-Step_Video_Reasoning_from_Perception_to_Cognition_images/icml_2024_Video-of-Thought_Step-by-Step_Video_Reasoning_from_Perception_to_Cognition.pdf-0004-01.png)


**----- Start of picture text -----**<br>
Predicate edge Temporal coreference edge<br>house man-1 on<br>beside s treet beside s treet street beside street<br>on the lefton the right house on man on man-2 on<br>car-1 car-2 car behind car behind car<br>... ...<br>Frame k-1 Frame k Frame k+1 Frame k+2<br>MotionEpic<br>Frame k-1 : {<br>Objects: [{"car-1": [0.0,13.4,7.0,8.1]}, {"car-2": [27.0,12.9,9.0, 4.0]},<br>    {"house": [15.0, 0.0,34.5,17.0]}, {"street": [9.0,13.5,16.5,15.0]}],<br>Triplets: [("car-1", "on the left", "street"), ("car-2", "on the right", "street"), ("house",<br>     "beside", "street")]<br>}<br>...<br>Frame k+2 : {<br>Objects: [{"car": [9.0,2.4,33.7,25.6]}, {"man-1": [1.8,16.4,4.2,10.0]},<br>    {"man-2": [43.0,14.4, 6.2,12.0]}, {"street": [17.9,0.0,10.2,10.0]}],<br>Triplets: [("man-1", "beside", "car"), ("car", "on", "street"), ("man-2", "behind", "car"),...]<br>}<br>...<br>Temporal Coreference Edge : [<br>{ Frame k-1    k : [("house", "house"), ("street", "street")]},<br>] ... Grounded (partial) STSG Expression<br>**----- End of picture text -----**<br>


Figure 3: The STSG expression generated by MotionEpic, with its corresponding structural STSG illustration. 

each single SG at _k_ -th frame as _Gk_ =( _Vk_ ; _Ek_ ), where _Vk_ is a list of nodes, i.e., object proposal, and _Ek_ is a list of predicate edges. For each object proposal _vk,i_ we record the category label _ci_ , the proposal’s neural representation _fi_ , and the bounding box (bbox) annotation _bi_ =( _x, y, w, h_ ), i.e., the 2D coordinate in the image. Thus, each _vk,i_ =( _ci, fi, bi_ ) _k_ . All nodes (i.e., _vk,i_ and _vk,j_ ) are connected with edges _ek,i,j_ . To enhance the connectivity of STSG, we further create a type of _temporal coreference edges_ across each single-frame SG, where the same objects are linked together with timepersistent edges, _ek−_ 1 _→k_ , mimicking the ‘tracking’ process. 

MotionEpic achieves fine-grained spatial-temporal video grounding by simultaneously understanding and generating STSGs. After full tuning (cf. _§_ 3.3), MotionEpic can directly output (partial) STSG based on input video (with text prompts), essentially grounding the specific portions of the video content as indicated in the input prompts. In Fig. 3 we illustrate how the generated STSG expression corresponds to the structural STSG. Further, the output STSG serving as the rationale will be recycled in the system, i.e., repurposed as the input for the subsequent round. 

## **3.3. Fine-grained Video-Scene Grounding-aware Tuning** 

Intuitively, we expect our system to perform video reasoning for downstream tasks without relying on any external STSG annotations, i.e., STSG-free inference. This requires an accurate spatial-temporal grounding between videos and STSGs. To this end, we carry out tuning for MotionEpic such that it is learned to autonomously parse STSG according to input instructions. The grounding-aware tuning is performed based on video-STSG pairs. We design various training objectives, which can be further divided into 

coarse-grained and fine-grained levels: 

- **1) Enhancing coarse-grained correspondence** : 

   - _L_ 1: predicting if the overall input video and STSG are paired. 

   - _L_ 2: given a video, generating the whole STSG (expression) of the video. 

- **2) Enhancing fine-grained correspondence** : 

   - _L_ 3: given a video and action description(s), outputting the corresponding object tracklet(s), i.e., a partial STSG. 

   - _L_ 4: given a video and key object(s), describing the corresponding temporal action(s) in textual response, and outputting the corresponding object tracklet(s). 

   - _L_ 5: given a video and a bbox of a certain frame’s object, outputting the object label, as well as the corresponding tracklet. 

For each learning objective, we wrap up the inputs with instruction-tuning (Liu et al., 2023) style question-answer pairs, being consistent with the following downstream inference. Overall, except for the STSG encoder and video projector, the video encoder and the backbone LLM are kept frozen throughout all the learning stages. To tune the LLM, we leverage LoRA (Hu et al., 2022) to enable a small subset of parameters to be updated. 

Before conducting the above grounding-level tuning, we perform conventional video pre-training on Webvid, which serves as the important warming starting for the following video understanding tuning, Despite aligning the encoding modules with LLM, there remains a gap towards the goal of enabling the overall system to faithfully follow and understand users’ instructions and generate the desired outputs. To address this, further instruction tuning is necessary. After the grounding-level tuning, we utilized existing video instruction tuning data for instruction tuning of the model, which includes the dataset from VideoChat (Li et al., 2023c) and Video-ChatGPT (Maaz et al., 2023) 

## **4. Video-of-Thought Reasoning Framework** 

Based on MotionEpic, we now perform video reasoning with VoT. Different from the vanilla CoT with one straightforward prompt, i.e., “ _Let’s think step by step_ ”, VoT divides the raw problem into much smaller and finer-grained subproblems. We consider an exact paradigm of task decomposition, which encompasses five chained steps, following a process from low-level perceptive pixel grounding to highlevel cognitive semantic comprehension. In Fig. 4 we illustrate the overall VoT framework. 

## ▶ **Step-1: Task Definition and Target Identification** 

First, MotionEpic is fed with the raw video along with the text prompt of the task definition, format, and raw question, all of which serve as the background foundation information 

4 

**Video-of-Thought: Step-by-Step Video Reasoning from Perception to Cognition** 

Step-1: Task Definition and Target Identification You are an expert ... answer a question based on the given video. For the question, several candidate answers ... 

Given the question: [ What is the relationship between the white truck and this neighborhood? A. Transportation  B. Buildings  C. Clean Services  D. Entertainment Facilities ] 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/icml_2024_Video-of-Thought_Step-by-Step_Video_Reasoning_from_Perception_to_Cognition_images/icml_2024_Video-of-Thought_Step-by-Step_Video_Reasoning_from_Perception_to_Cognition.pdf-0005-03.png)


**----- Start of picture text -----**<br>
What are the possible targets of the         mainly mentioned or<br>involved?<br>The involved targets are [ the white truck ], [ the neighborhood ]<br>Step-2: Object Tracking<br>Provide the tracklet of involved [ the neighborhood ] and [ the<br>white truck ] by outputting the corresponding partial<br>expression in the          .<br>The partial               in tracking [ the neighborhood ] and [ the<br>white truck ] is [ Frame 1 : {Objects: [{"car-1": [0.0,13.4,7.0,8.1]}, ...],<br>Triplets: [("car-1","on the left", "street"), ...]}...].<br>Step-3: Action Analyzing<br>Combining all possible related commonsense, analyze the<br>motion behavior based on the [ th e  w h ite  tr uc k ] and [  th e<br>neighborhood ] and the neighbor scenes within                .<br>Describing the action and the implication.<br>[ The two men are driving the white truck into a neighborhood, and<br>pour the garbage from the roadside trash cans into the white truck.<br>According to commonsense, the white car is used for collecting<br>rubbish... ]<br>Step-4: Question Answering via Ranking<br>A B C D<br>For the question [ What is the relationship between the white truck<br>and this neighborhood? ], given a candidate answer [ A.<br>Transportation ], please based on the [ The two men are driving the<br>white truck... ] combined with commonsense, score the rationality<br>of this answer with a 1-10 scale, and also output the rationale.<br>Based on the observations and commonsense, the rationality of<br>the answer [ A. Transportation ] is 5. While the white truck is<br>indeed moving through ... but rather the collection of garbage ...<br>Ranking<br> Based on the rationality, the ranking results in descend order is<br>[ C. Clean Service ] [ A. Transportation ] [ D. Entertainment Facilities ]  [ B.<br>Buildings ]. So, the final answer is [ C. Clean Service ].<br>Step-5: Answer Verification<br>Given the        , and the raw question [ What is ... ], now you need<br>to verify the previous answer by 1) checking the pixel grounding<br>information             if the answer [ C. Clean Service ] aligns with<br>the facts presented in the video from a perception standpoint; 2)<br>determining from a cognition perspective if the commonsense<br>implications inherent in the answer contradict any of the main<br>[ The two men... ]. Output the verification result with rationale.<br>Certainly. 1. Grounding information: the collection action of<br>garbage, which aligns ... 2. Commonsense verification: driving a<br>white truck into  a neighborhood is associated with waste<br>management ...  Overall, the answer [ C. Clean Service ] is correct.<br>Ouput<br>Ouput<br>Ouput<br>Input<br>Ouput<br>Ouput<br>Ouput<br>Input<br>Input<br>Input<br>**----- End of picture text -----**<br>


Figure 4: An illustrative view of VoT framework. The complete I/O and prompts are detailed in Appendix. 

of the reasoning. As the initial phase, we expect to identify the target within the video that requires analysis, which is a crucial prerequisite for determining the subsequent in-depth reasoning. It is noteworthy that sometimes the question may 

explicitly include targets visible in the video, or implicitly involve related targets. Therefore, we proceed to prompt the model, to infer from the question what the target object(s) involved or related to in the video might be: 

## Given the question [Question], what are the possible targets of the mainly mentioned or involved? 

After this step, all the possible [Target] involved in the question will be confirmed. 

## ▶ **Step-2: Object Tracking** 

In the second step, we aim to further ground the object’s full spatial-temporal characteristics, i.e., to track the target’s trajectory. We note that grounding the targets’ temporal tracking is pivotal for pursuing fine-grained video understanding, as only accurately perceiving the behaviors in the video can ensure that the subsequent cognitive-level understanding is meaningful. In this work, we leverage the STSG for the temporal grounding, rather than directly tracking the original video frames. Such semantic representation carried by STSG is highly concise, ensuring that the tracking of the video’s target is more accurate and reliable. Also notably, object tracking and pixel grounding based on the STSG can effectively mitigate the hallucination issues (Zhang et al., 2023b) inherent in existing MLLMs. 

Having performed grounding-aware tuning, MotionEpic develops the full capability to ground from object to (partial) STSG. Therefore, we directly prompt the model with: 

## Provide the tracklet of involved [Target] by outputting the corresponding partial expression. 

The yielded grounded [Target Tracklet] of STSG will serve as low-level evidence (i.e., supporting rationale) for the next step of behavior analysis. 

## ▶ **Step-3: Action Analyzing** 

In this step, VoT further analyze the corresponding actions and behaviors by integrating the target tracking in STSG. For an accurate understanding of the target object’s motion, merely observing the target itself is insufficient. This process should also reference the higher-order neighbor nodes within the STSG representation, interacting targets with their neighboring scenes. On the other hand, directly inferring actions from video pixels alone is still inadequate, as interpretations based solely on pixel information often remain superficial. Therefore, we further prompt the model to consider more potentially relevant commonsense knowledge, allowing the model to connect video pixel observations with the factual world, achieving a more in-depth understanding of the video. Given that MLLMs possess the necessary repository of commonsense knowledge via extensive pretraining, all that is required is to properly prompt the model: 

5 

**Video-of-Thought: Step-by-Step Video Reasoning from Perception to Cognition** 

Combining all possible related commonsense, analyze the motion behavior based on the [Target Tracklet] and the neighbor scenes within . Describing the action observations and implications. 

This step yields the target action’s [Observation and Implication]. 

## ▶ **Step-4: Question Answering via Ranking** 

Having established an in-depth understanding of the target actions in the video, we can now consider answering the original question. We contemplate a multiple-choice QA format, where multiple candidate answers are provided.[1] Inspired by the human pattern of answering multi-choice questions, we also consider a ranking mechanism to determine the final answer. Specifically, for each candidate answer, we prompt the model to score its likelihood (from 1 to 10) in conjunction with commonsense knowledge, and provide a corresponding rationale: 

For the question [Question], given a candidate answer [Answer], please based on the action’s [Observation and Implication] combined with commonsense, score the rationality of this answer with a 1-10 scale, and also output the rationale. 

We then rank the scores of all options and select the most optimal answer [Answer]. 

## ▶ **Step-5: Answer Verification** 

Given that complex video task often involves intricate questions and answers, and the entire reasoning process encompasses lengthy chained steps, it is essential to verify the answer provided in the previous step. Our basic idea to verification is that, assuming that answer A is correct, we retrospectively evaluate whether the answer results in contradictions with the input question and video in two aspects: 1) First, check the pixel grounding information if it aligns with the facts presented in the video from a perception standpoint. 2) On the other hand, prompt the model again from a cognition perspective to determine if the commonsense implications inherent in the answer contradict any of the main observations inferred in the 3- _rd_ reasoning step. 

Given the , and the raw question [Question], now you need to verify the previous answer by 

1) checking the pixel grounding information if the answer [Answer] aligns with the facts presented in the video from a perception standpoint; 

2) determining from a cognition perspective if the commonsense implications inherent in the answer contradict any of the main [Observations] inferred in the 3- _rd_ reasoning step. Output the verification result with rationale. 

If any inconsistencies are found in perception and cognition perspectives, we record the corresponding rationale, and reexecute the 4- _th_ step to reselect the answer. This approach ensures that the final outcome is the most factually accurate. 

## **5. Experiments** 

## **5.1. Settings** 

**Task and Data.** While in theory all video understanding tasks could benefit from our reasoning framework, we mainly focus on the most representative task, video QA. For fine-tuning setting, we adopt 6 benchmarks characterizing complex video QA where advanced video abilities, e.g., explanation, causality, foresight and imagination are required: VLEP (Lei et al., 2020), STAR (Wu et al., 2021), IntentQA (Li et al., 2023b), Social-IQ (Zadeh et al., 2019), CausalVidQA (Li et al., 2022a) and NExT-QA (Xiao et al., 2021). For zero-shot setting, we further consider using MSR-VTT (Xu et al., 2016) and ActivityNet (Heilbron et al., 2015) datasets. All datasets come with their own splitting, and we follow the prior practice without modification. 

**Grounding-aware Tuning Corpus.** To construct the video-STSG pairs, we leverage the Action Genome data (Ji et al., 2020), which contains 10K high-quality manual annotated STSGs of videos. To enrich the data amount, we also use part of WebVid-10M videos (Bain et al., 2021), where we select 350K videos with clear actions, and parse the STSGs via SoTA parser (Li et al., 2022b). 

**Implementations.** MotionEpic uses the Vicuna-7B (v1.5)[2] as the backbone LLM. We adopt the ViT-L/14[3] as the video encoder, and use the Q-Former[4] as the projector. All the modules take the default configurations without much modification. For our recurrent Graph Transformer encoding STSGs, we take a 6-layer architecture with 768-d hidden sizes. The object neural representation _fi_ is obtained via CLIP, which will be used as node embedding initiation. The text tokenizer is sourced from LLaMA, with approximately 32,000 classes. For each video, we uniformly sample certain frames with a sampling rate of 8 fps for fine-grained reasoning. We note that too large sampling rate introduces noises (i.e., redundant frames) and huge computation costs, while too small one will cause important information loss. Here we use the 8 fps, as in our preliminary study we verified that it helps achieve the best trade-off. For the fine-tuning setting of end tasks, we will tune the MotionEpic based on the training set using the setting as prior baselines, i.e., data split and evaluation methods. For the zero-shot setting, we will directly perform video QA without using the in-domain 

2https://huggingface.co/lmsys/vicuna-7b-v1.5 

1Note that for open-ended QA, we consider prompting the model to output multiple distinct optional answers, such that we unify different types of QA formats into a multi-choice format. For the open-ended QA format, we detail processing and prompt methods in Appendix. 

3https://huggingface.co/sentence-transformers/ clip-ViT-L-14 

> 4https://huggingface.co/spaces/Salesforce/ BLIP2 

6 

**Video-of-Thought: Step-by-Step Video Reasoning from Perception to Cognition** 

Table 1: Results on four VideoQA datasets. STAR data includes four subsets: Interaction (Int.), Sequence (Seq.), Prediction (Pre.), Feasibility (Fea.). The best scores of baselines are underlined, and the new best results are **bold** . 

|**Model**<br>**VLEP**|**STAR**<br>**IntentQA**<br>Int. Seq. Pre. Fea.|**STAR**<br>**IntentQA**<br>Int. Seq. Pre. Fea.|**STAR**<br>**IntentQA**<br>Int. Seq. Pre. Fea.|**Social-IQ**<br>2-Way 4-Way|
|---|---|---|---|---|
||Int. Seq.||||
|_•_**SoTA baselines**<br>InternVideo<br>63.9<br>LLaMA-VQA<br>71.0<br>VLAP<br>69.6<br>SeViLA<br>68.9<br>VideoChat<br>62.0<br>Video-LLaVA<br>65.8|62.7 65.6 <br>66.2 67.9 <br>70.0<br>70.4<br>63.7 70.4 <br>63.2 66.8 <br>64.3 67.0|54.9 51.9<br> 57.2 52.7<br>65.9<br>62.2<br> 63.1 62.4<br> 54.1 49.6<br> 56.5 50.1|-<br>-<br>-<br>-<br>59.3<br>62.5|-<br>-<br>-<br>-<br>-<br>-<br>-<br>-<br>67.7<br>37.8<br>68.9<br>39.2|
|_•_**CoT**<br>Video-LLaVA<br>65.7<br>Video-LLaVA+STSG<br>67.0<br>MotionEpic<br>68.2|65.0 67.7 <br>65.9 68.9 <br>66.8 69.6|57.8 52.0<br> 58.7 53.7<br> 60.6 57.4|63.2<br>64.9<br>66.1|69.5<br>40.4<br>70.4<br>41.7<br>71.7<br>43.0|
|_•_**VoT**|||||
|MotionEpic<br>**73.4**|**71.5**<br>**72.6**|**66.6**<br>**62.7**|**70.8**|**72.8**<br>**45.0**|



training set. All trainings are conducted on 16 NVIDIA A100 GPUs. 

**Baselines and Evaluations.** The evaluations are compared with recent SoTA baselines of these complex video QA datasets, including InternVideo (Wang et al., 2022b), LLaMA-VQA (Ko et al., 2023), VLAP (Wang et al., 2023), SeViLA (Yu et al., 2023), TranSTR (Li et al., 2023f) and HiTeA (Ye et al., 2023). The results are faithfully copied from their papers. Also we reimplement current video MLLMs, including VideoChat2 (Li et al., 2023e), VideoLLaMA (Zhang et al., 2023a), Video-ChatGPT (Maaz et al., 2023), VideoChat (Li et al., 2023d) and Video-LLaVA (Lin et al., 2023). For fairness, we compare MotionEpic with these video MLLMs in a vanilla CoT setting. Further, we also implement the Video-LLaVA by integrating the STSG features. We adopt accuracy as the main metric of the QA task performance, following the prior practice. 

## **5.2. Main Performance on Video QA Reasoning** 

In Table 1, 2 and 3 we present the main results of different systems. Overall, our MotionEpic under the VoT reasoning framework has boosted all the SoTA baselines by very large margins consistently. Beyond enhanced performance, we further gain some key observations. First, by observing Video-LLaVA without/with CoT prompting, we see that the improvement from CoT for video reasoning could be quite limited. Further, by comparing Video-LLaVA without/with STSG integration, we notice that the structural fine-grained STSG features play a positive role in understanding videos. Third, by comparing Video-LLaVA+STSG with our MotionEpic under the same CoT, it is clear that the implicit integration of the scene graph features is quite superior to the explicit integration. Also, even our MotionEpic with vanilla 

Table 2: Results on Causal-VidQA data. D: Description, E: Explanation, P: Prediction, C: Counterfactual. 

|**Model**<br>**Acc@D Acc@E**|**Model**<br>**Acc@D Acc@E**|**Acc@P**|**Acc@C**<br>A<br>R<br>AR|**Acc@C**<br>A<br>R<br>AR|
|---|---|---|---|---|
|||A<br>R<br>AR|A||
|_•_**SoTA baselines**<br>TranSTR<br>Video-LLaMA<br>VideoChat<br>Video-ChatGPT<br>Video-LLaVA|73.6<br>75.8<br>69.2<br>71.0<br>72.9<br>73.9<br>73.1<br>75.1<br>73.7<br>74.4|65.1 65.0 48.9 <br>63.6 62.4 44.4 <br>65.2 63.1 45.9 <br>66.0 63.9 46.0 <br>67.6 65.4 47.7|68.6 <br> 65.4 <br> 66.0 <br> 67.8 <br> 68.0|65.3 50.3<br> 60.1 45.0<br> 62.7 45.8<br> 63.6 50.0<br> 64.9 51.5|
|_•_**CoT**<br>Video-LLaVA<br>Video-LLaVA+STSG<br>MotionEpic|74.2<br>74.8<br>75.7<br>75.9<br>78.5<br>77.2|68.0 65.7 48.1 <br>68.9 67.2 50.0 <br>70.1<br>70.8<br>52.4|70.3 <br> 70.7 <br>71.2|65.7 52.9<br> 67.2 53.6<br>69.1<br>55.0|
|_•_**VoT**|||||
|MotionEpic|**81.2**<br>**83.0**|**74.3**<br>**73.7**<br>**54.7**|**74.5**|**73.8**<br>**58.6**|



Table 3: Results on NExT-QA data. 

|**Model**<br>**Acc@All**<br>**Acc@C**<br>**Acc@T**<br>**Acc@D**|**Model**<br>**Acc@All**<br>**Acc@C**<br>**Acc@T**<br>**Acc@D**|
|---|---|
|_•_**SoTA baselines**<br>InternVideo<br>63.2<br>HiTeA<br>63.1<br>LLaMA-VQA<br>72.0<br>SeViLA<br>73.8<br>VLAP<br>75.5<br>Video-LLaMA<br>60.6<br>VideoChat<br>61.8<br>Video-ChatGPT<br>64.4<br>Video-LLaVA<br>66.3|62.5<br>58.5<br>75.8<br>62.4<br>58.3<br>75.6<br>72.7<br>69.2<br>75.8<br>73.8<br>67.0<br>81.8<br>74.9<br>72.3<br>82.1<br>59.2<br>57.4<br>72.3<br>63.5<br>61.5<br>74.6<br>66.9<br>64.1<br>75.7<br>67.7<br>63.8<br>75.9|
|_•_**CoT**<br>Video-LLaVA<br>67.7<br>Video-LLaVA+STSG<br>68.0<br>MotionEpic<br>72.2|69.0<br>65.9<br>76.5<br>71.6<br>67.6<br>78.9<br>73.4<br>69.1<br>80.7|
|_•_**VoT**||
|MotionEpic<br>**76.0**|**75.8**<br>**74.6**<br>**83.3**|



CoT we beat the SoTA methods on certain datasets. Lastly, observing the MotionEpic under CoT and our proposed VoT, we see there are huge performance gaps in between consistently on all reasoning scenarios and tasks, indicating the great potential of our proposed video reasoning framework. 

## **5.3. Zero-shot Performance** 

We then examine the performance in zero-shot setting. Table 4 presents the comparisons. In general, we can notice that CoT exhibits stronger improvements than direct prompting methods under the zero-shot scenario, compared with the scenario of the above fine-tuning. Notedly, the improvements by VoT over the CoT become larger and clearer under the zero-shot setting. The enhancements on these two complex video QA tasks are clearer than those on the comparatively simpler tasks, i.e., MSR-VTT and ActivityNet. This is largely because the latter datasets more tend to perceptive understanding (e.g., describing _what’s in video_ ), rather than cognitive understanding (e.g., explanation, foresight 

7 

**Video-of-Thought: Step-by-Step Video Reasoning from Perception to Cognition** 

Table 4: Zero-shot Video QA results. Verify-G/C: verification in terms of Grounding and Commonsense perspectives. 

|**Model**<br>**MSR-VTT ActivityNet NExT-QA STAR** **_AVG._**|**Model**<br>**MSR-VTT ActivityNet NExT-QA STAR** **_AVG._**|
|---|---|
|_•_**Zero-shot SoTA baselines**<br>InternVideo<br>-<br>-|49.1<br>41.6<br>-|
|Video-LLaMA<br>49.6<br>21.4<br>VideoChat<br>52.0<br>26.5<br>Video-ChatGPT<br>54.3<br>35.2<br>Video-LLaVA<br>59.2<br>45.3<br>VideoChat2<br>54.1<br>49.1|43.5<br>36.4<br>37.7<br>52.8<br>45.0<br>44.1<br>53.0<br>48.7<br>47.8<br>57.3<br>50.6<br>53.1<br>61.7<br>59.0<br>56.0|
|_•_**CoT**<br>Video-LLaVA<br>60.0<br>46.9<br>Video-LLaVA+STSG<br>61.5<br>48.4<br>MotionEpic<br>63.1<br>50.0|59.5<br>52.0<br>54.6<br>60.6<br>52.7<br>55.8<br>61.9<br>56.5<br>57.8|
|_•_**VoT**||
|MotionEpic<br>**66.2**<br>**54.6**|**66.5**<br>**61.7**<br>**62.3**|
|w/o Verify-G<br>63.6<br>51.4<br>w/o Verify-C<br>65.1<br>53.4|62.0<br>59.1<br>59.0<br>62.8<br>58.8<br>60.1|




![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/icml_2024_Video-of-Thought_Step-by-Step_Video_Reasoning_from_Perception_to_Cognition_images/icml_2024_Video-of-Thought_Step-by-Step_Video_Reasoning_from_Perception_to_Cognition.pdf-0008-03.png)


**----- Start of picture text -----**<br>
GPS-Net STTran AP-Net MotionEpic Human<br>95 100<br>47<br>85 45 . 5 94<br>75 44 88<br>Bbox SG Triplets Action Grounding<br>(%)<br>(%) (%)<br>IoU IoU<br>R@10<br>**----- End of picture text -----**<br>


Figure 5: MotionEpic performance on object grounding, scene graph triplet classification, and action grounding. 

or imagination). Further, we cancel the verification mechanism (at 6- _th_ ) of either the pixel grounding perspective or the commonsense perspective. We see that on MSRVTT and ActivityNet, the perceptive-level pixel grounding verification is more crucial than the commonsense cognitive verification. For those complex videos, both types of verifications are pivotal. 

## **5.4. Analyses on MotionEpic Video MLLM** 

**Probing Video Grounding Ability.** To evaluate how well our MotionEpic is capable of video grounding, we here perform the probing test. Specifically, we evaluate the performance of MotionEpic on STSG parsing on the Action Genome test set, by comparing with SoTA DSG parsers: GPS-Net (Lin et al., 2020), STTran (Cong et al., 2021) and AP-Net (Li et al., 2022b). We measure three aspects: 1) the object grounding (bbox detection), 2) SG triplet classification (categories of entities, and relation predicates between entities), and 3) temporal action grounding (the start and end times of actions). Fig. 5 illustrates the results, where we see that MotionEpic achieves very competitive performance on par with SoTA parser, even with human-level performance. This reveals that MotionEpic shows reliable capability in providing video grounding information to support the subsequent in-depth video reasoning. 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/icml_2024_Video-of-Thought_Step-by-Step_Video_Reasoning_from_Perception_to_Cognition_images/icml_2024_Video-of-Thought_Step-by-Step_Video_Reasoning_from_Perception_to_Cognition.pdf-0008-08.png)


**----- Start of picture text -----**<br>
All w/o L 1 w/o L 2 w/o L 3 w/o L 4 w/o L 5<br>62<br>76<br>59.6 75.5<br>58 71 -3.9<br>54 -4.5 -5.5 -6.4 66 -7.9 -8.<br>-10.9 9<br>50 -8.5 -9.7 61 -15.9<br>ActivityNet NExT-QA<br>Figure 6: Performance (zero-shot) drop of MotionEpic after<br>ablating different grounding-aware tuning items.<br>CoT VoT<br>Data Human<br>Video-LLaVA MotionEpic MotionEpic<br>Causal-VidQA 32.4 56.8 74.3 80.6<br>Social-IQ 22.3 40.1 61.4 72.7<br>MotionEpic (VoT) Video-LLaVA (CoT)<br>75 76.2 Causal-VidQA 68.4 83.5<br>54.5<br>50 65.9% 33.3 51.6% 63.1%<br>25 47.8% 22.1 16.2% 27.9%<br>0<br>100 Social-IQ 90.3<br>77.2<br>75 65.4 70.3<br>50 32.8 52.0 38.6% 50.0% 59.9% 54.8% 66.2%<br>25 28.3%<br>0<br>Object Action Long Action Action<br>Video<br>Recognition Temporality Number Semantics Commonsense<br>(%)<br>Acc<br>(%)<br>Rate<br>Error<br>**----- End of picture text -----**<br>


Figure 7: **Above Table** : human evaluation of video QA. **Below Figure** : error rate under various specific categories. 

**Influence of Various Grounding-aware Tuning Strategies.** We further study the impacts/contributions of different grounding-aware tuning objectives introduced in _§_ 3.3. We design five groups of ablations where each tuning goes without one item, and the resulting model performs zeroshot end task. The results on two datasets are shown in Fig. 6, where different items come with varied impacts, indicating the importance of video-STSG grounding finetuning. Notably, the lack of _L_ 2 and _L_ 4 result in the greatest degradation. This is intuitive, as these two objectives are directly associated with the subsequent reasoning process, i.e., understanding STSG from video, and generating (partial) STSG given objects. 

## **5.5. Analyses on VoT Reasoning Framework** 

**Reasoning Ability Breaking-down.** Previously, we validated the overall stronger performance of the VoT reasoning system through extensive experimentation. Here, we aim to provide a more in-depth analysis of VoT. First, we select 200 hard instances each from the Causal-VidQA and SocialIQ test sets, and then compare the performance of VideoLLaVA and MotionEpic under CoT and VoT frameworks, respectively. Also we conduct human evaluation on this subset to gauge its difficulty level. The results, in the above table of Fig. 7, show that MotionEpic with VoT reasoning 

8 

**Video-of-Thought: Step-by-Step Video Reasoning from Perception to Cognition** 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/icml_2024_Video-of-Thought_Step-by-Step_Video_Reasoning_from_Perception_to_Cognition_images/icml_2024_Video-of-Thought_Step-by-Step_Video_Reasoning_from_Perception_to_Cognition.pdf-0009-01.png)


**----- Start of picture text -----**<br>
Question:  What is the least likely category for the animal in this video?<br>A. Police Dog                B. Competitive Animal                      C. Circus Performer                     D. Companion Pet                   E. Search and Rescue DogVideo-LLaVA Video-ChatGPT MotionEpic<br>Frame 1 Frame 2 Frame 3 Frame 4 Frame 5 Frame 6 Frame 7 Frame 8<br>Step-1:  The involved target is [ dog ]. Step-4:<br>The video depicting professional  The companion dog is to support<br>Step-2:  The partial STSG in tracking [ dog ] is: training and complex actions  companionship and emotional<br>lie on grass run on grass dog on grass suggests it might be a police dog performing daily training ... The  support to their owners rather tan engaging in specialized tasks ...   ...<br>lie ondog stand ongrass behinddog manon in front ofdog manon ... hurdle-1behind behind manon rationality of the answer [ Dog ] is 2.  A. Police  The answer [has a coherence score of 8.  D. Companion Pet ]<br>behind man hurdle hurdle hurdle-2 After ranking the rationale score, the final answer is [ D. Companion Pet ].<br>Frame 1 Frame 2 Frame 3 Frame 7<br>Step-5: Let' verify the [ D. Companion Pet ] based on visual perception ..<br>Step-3: According to the video scene and STSG, the dog is crossing  1. Pixel Grounding Information Check: Based on the video scene, it<br>multiple hurdles with the dog being visible both before and after the  depicts a training ground with a dog, so the answer is fitting.<br>hurdles. The accompanying man is observed providing instructions to      2. Commonsense Check: Observing the dog's energetic behavior during<br>guide the dog through the obstacles... Drawing on factual commonsense  training aligns with the common understanding that companion pet are less<br>understanding, it might be inferred that the man is a trainer who is  likely to undergo such training, supporting the chose answer.<br>imparting various commands and training the dog on a grassy field.     Conclusion: The answer [ D. Companion Pet ] is supported both by ...<br>**----- End of picture text -----**<br>


Figure 8: Visualization of qualitative example showcasing how our VoT framework achieves successful video reasoning. 

framework achieves quite exceptional results, comparable even to human performance. We further summarize the error cases and analyze differences in the 6 most frequent categories of errors. As seen in the below part of the figure, MotionEpic (with VoT) significantly reduces the error rate of Video-LLaVA (with CoT), especially in terms of action semantics and commonsense understanding. 

**Video Reasoning Visualization.** Finally, we present a case study to aid an intuitive understanding of the superiority of our system. We randomly select an instance where our model gives the correct answer. As shown in Fig. 8, the video displays a complex scene, and the given question is abstract and complex, not directly answerable through the mere perception of the video itself. However, our MotionEpic provides the correct answer, while the other two baselines err. At the content perception level, VoT ensures accurate and robust understanding through STSG-based video grounding, preventing hallucination, i.e., it correctly interprets that the animal is a dog, then infers from commonsense that the scene involves a trainer training the dog. Then, at the cognitive level, it analyzes each option to determine the best answer. Through further verification, the result aligns with both the video content and factual commonsense understanding. Overall, the entire reasoning greatly improves the accuracy at each step through problem decomposition, while ensuring an explainable process decision rationale. In the Appendix, we show more qualitative visualizations of examples. 

## **6. Conclusion** 

In this work, we for the first time introduce an innovative solution for complex video reasoning, the Video-of-Thought (VoT) framework. To accomplish the reasoning framework, also a novel video MLLM, MotionEpic, is proposed. Mo- 

tionEpic achieves fine-grained pixel-level spatial-temporal video grounding by adeptly integrating video STSG representation. With MotionEpic, the VoT framework resolves the intricate video task by skillfully dissecting it into manageable sub-problems, tackling them sequentially from lowlevel pixel perception to advanced cognitive interpretation. Our experiments across various complex video QA benchmarks have not only proven the efficacy of our approach but have also boosted the existing state-of-the-art standards. Overall, this work marks a substantial contribution to the video modeling community, paving the way for more nuanced, human-level analysis in the relevant community. 

## **Acknowledgements** 

This research is supported by the National Research Foundation Singapore under its AI Singapore Programme (Award Number: AISG-GC-2019-001-2A). This work is supported by CCF-Baidu Open Fund, and the National Natural Science Foundation of China (NSFC) Grant (No. 62336008). 

## **Statement of Potential Broader Impact** 

This paper aims to construct a robust, human-level video understanding and reasoning framework. The system must be built upon existing LLM to realize its full potential. Potential implications include substantial energy consumption during LLM system training, leading to environmental degradation, and the necessity for more extensive data corpora for training. Moreover, due to the powerful video reasoning and comprehension capabilities, there exists the potential for malicious actors to exploit this framework for nefarious intents, posing a societal threat. Consequently, the release of this framework necessitates the establishment of specific licensing mechanisms to ensure responsible deployment and mitigate potential misuse. 

9 

**Video-of-Thought: Step-by-Step Video Reasoning from Perception to Cognition** 

## **References** 

- Bain, M., Nagrani, A., Varol, G., and Zisserman, A. Frozen in time: A joint video and image encoder for end-to-end retrieval. In _Proceedings of the IEEE/CVF International Conference on Computer Vision_ , pp. 1728–1738, 2021. 

- Bertasius, G., Wang, H., and Torresani, L. Is space-time attention all you need for video understanding? In _ICML_ , volume 2, pp. 4, 2021. 

- Caballero, J., Ledig, C., Aitken, A., Acosta, A., Totz, J., Wang, Z., and Shi, W. Real-time video super-resolution with spatio-temporal networks and motion compensation. In _Proceedings of the IEEE conference on computer vision and pattern recognition_ , pp. 4778–4787, 2017. 

- Chiang, W.-L., Li, Z., Lin, Z., Sheng, Y., Wu, Z., Zhang, H., Zheng, L., Zhuang, S., Zhuang, Y., Gonzalez, J. E., Stoica, I., and Xing, E. P. Vicuna: An open-source chatbot impressing gpt-4 with 902023. 

- Cong, Y., Liao, W., Ackermann, H., Rosenhahn, B., and Yang, M. Y. Spatial-temporal transformer for dynamic scene graph generation. In _Proceedings of the IEEE/CVF international conference on computer vision_ , pp. 16372– 16382, 2021. 

- Dosovitskiy, A., Beyer, L., Kolesnikov, A., Weissenborn, D., Zhai, X., Unterthiner, T., Dehghani, M., Minderer, M., Heigold, G., Gelly, S., et al. An image is worth 16x16 words: Transformers for image recognition at scale. In _International Conference on Learning Representations_ , 2020. 

- Dwivedi, V. P. and Bresson, X. A generalization of transformer networks to graphs. _arXiv preprint arXiv:2012.09699_ , 2020. 

- Fei, H., Wu, S., Ren, Y., and Zhang, M. Matching structure for dual learning. In _Proceedings of the International Conference on Machine Learning, ICML_ , pp. 6373–6391, 2022. 

- Fei, H., Li, B., Liu, Q., Bing, L., Li, F., and Chua, T.S. Reasoning implicit sentiment with chain-of-thought prompting. _arXiv preprint arXiv:2305.11255_ , 2023a. 

- Fei, H., Liu, Q., Zhang, M., Zhang, M., and Chua, T.-S. Scene graph as pivoting: Inference-time image-free unsupervised multimodal machine translation with visual scene hallucination. In _Proceedings of the 61st Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)_ , pp. 5980–5994, 2023b. 

- Fei, H., Wu, S., Ji, W., Zhang, H., and Chua, T.-S. Empowering dynamics-aware text-to-video diffusion with large language models. _arXiv preprint arXiv:2308.13812_ , 2023c. 

- Fei, H., Wu, S., Zhang, M., Zhang, M., Chua, T.-S., and Yan, S. Enhancing video-language representations with structural spatio-temporal alignment. _IEEE Transactions on Pattern Analysis and Machine Intelligence_ , 2024. 

- Heilbron, F. C., Escorcia, V., Ghanem, B., and Niebles, J. C. Activitynet: A large-scale video benchmark for human activity understanding. In _Proceedings of the CVPR_ , pp. 961–970, 2015. 

- Hu, E. J., Shen, Y., Wallis, P., Allen-Zhu, Z., Li, Y., Wang, S., Wang, L., and Chen, W. Lora: Low-rank adaptation of large language models. In _Proceedings of the ICLR_ , 2022. 

- Ji, J., Krishna, R., Fei-Fei, L., and Niebles, J. C. Action genome: Actions as compositions of spatio-temporal scene graphs. In _Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition_ , pp. 10236–10247, 2020. 

- Johnson, J., Gupta, A., and Fei-Fei, L. Image generation from scene graphs. In _Proceedings of the CVPR_ , pp. 1219–1228, 2018. 

- Ko, D., Lee, J., Kang, W.-Y., Roh, B., and Kim, H. Large language models are temporal and causal reasoners for video question answering. In _Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing_ , pp. 4300–4316, 2023. 

- Lei, J., Yu, L., Bansal, M., and Berg, T. Tvqa: Localized, compositional video question answering. In _Proceedings of the 2018 Conference on Empirical Methods in Natural Language Processing_ , pp. 1369–1379, 2018. 

- Lei, J., Yu, L., Berg, T., and Bansal, M. What is more likely to happen next? video-and-language future event prediction. In _Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing (EMNLP)_ , pp. 8769–8784, 2020. 

- Li, J., Niu, L., and Zhang, L. From representation to reasoning: Towards both evidence and commonsense reasoning for video question-answering. In _Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition_ , pp. 21273–21282, 2022a. 

- Li, J., Li, D., Savarese, S., and Hoi, S. C. H. BLIP-2: bootstrapping language-image pre-training with frozen image encoders and large language models. In _Proceedings of the ICML_ , pp. 19730–19742, 2023a. 

- Li, J., Wei, P., Han, W., and Fan, L. Intentqa: Context-aware video intent reasoning. In _Proceedings of the IEEE/CVF International Conference on Computer Vision_ , pp. 11963– 11974, 2023b. 

10 

**Video-of-Thought: Step-by-Step Video Reasoning from Perception to Cognition** 

- Li, K., He, Y., Wang, Y., Li, Y., Wang, W., Luo, P., Wang, Y., Wang, L., and Qiao, Y. Videochat: Chat-centric video understanding. _CoRR_ , abs/2305.06355, 2023c. 

- Li, K., He, Y., Wang, Y., Li, Y., Wang, W., Luo, P., Wang, Y., Wang, L., and Qiao, Y. Videochat: Chat-centric video understanding. _arXiv preprint arXiv:2305.06355_ , 2023d. 

- Li, K., Wang, Y., He, Y., Li, Y., Wang, Y., Liu, Y., Wang, Z., Xu, J., Chen, G., Luo, P., et al. Mvbench: A comprehensive multi-modal video understanding benchmark. _arXiv preprint arXiv:2311.17005_ , 2023e. 

- Li, Y., Yang, X., and Xu, C. Dynamic scene graph generation via anticipatory pre-training. In _Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition_ , pp. 13874–13883, 2022b. 

- Li, Y., Xiao, J., Feng, C., Wang, X., and Chua, T.-S. Discovering spatio-temporal rationales for video question answering. In _Proceedings of the IEEE/CVF International Conference on Computer Vision_ , pp. 13869–13878, 2023f. 

- Lin, B., Ye, Y., Zhu, B., Cui, J., Ning, M., Jin, P., and Yuan, L. Video-llava: Learning united visual representation by alignment before projection. _CoRR_ , abs/2311.10122, 2023. 

- Lin, J., Gan, C., and Han, S. Tsm: Temporal shift module for efficient video understanding. In _Proceedings of the IEEE/CVF international conference on computer vision_ , pp. 7083–7093, 2019. 

- Lin, X., Ding, C., Zeng, J., and Tao, D. Gps-net: Graph property sensing network for scene graph generation. In _Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition_ , pp. 3746–3753, 2020. 

- Liu, H., Li, C., Wu, Q., and Lee, Y. J. Visual instruction tuning. _CoRR_ , abs/2304.08485, 2023. 

- Lu, P., Mishra, S., Xia, T., Qiu, L., Chang, K.-W., Zhu, S.-C., Tafjord, O., Clark, P., and Kalyan, A. Learn to explain: Multimodal reasoning via thought chains for science question answering. _Advances in Neural Information Processing Systems_ , 35:2507–2521, 2022. 

- Maaz, M., Rasheed, H. A., Khan, S. H., and Khan, F. S. Video-chatgpt: Towards detailed video understanding via large vision and language models. _CoRR_ , abs/2306.05424, 2023. 

- Neimark, D., Bar, O., Zohar, M., and Asselmann, D. Video transformer network. In _Proceedings of the IEEE/CVF international conference on computer vision_ , pp. 3163– 3172, 2021. 

OpenAI. Introducing chatgpt. 2022a. 

OpenAI. Gpt-4 technical report. 2022b. 

- Wang, X., Wei, J., Schuurmans, D., Le, Q., Chi, E., Narang, S., Chowdhery, A., and Zhou, D. Self-consistency improves chain of thought reasoning in language models. _arXiv preprint arXiv:2203.11171_ , 2022a. 

- Wang, X., Liang, J., Wang, C.-K., Deng, K., Lou, Y., Lin, M., and Yang, S. Vlap: Efficient video-language alignment via frame prompting and distilling for video question answering. _arXiv preprint arXiv:2312.08367_ , 2023. 

- Wang, Y., Li, K., Li, Y., He, Y., Huang, B., Zhao, Z., Zhang, H., Xu, J., Liu, Y., Wang, Z., et al. Internvideo: General video foundation models via generative and discriminative learning. _arXiv preprint arXiv:2212.03191_ , 2022b. 

- Wei, J., Wang, X., Schuurmans, D., Bosma, M., Xia, F., Chi, E., Le, Q. V., Zhou, D., et al. Chain-of-thought prompting elicits reasoning in large language models. _Advances in Neural Information Processing Systems_ , 35: 24824–24837, 2022. 

- Wu, B., Yu, S., Chen, Z., Tenenbaum, J. B., and Gan, C. Star: A benchmark for situated reasoning in real-world videos. In _Annual Conference on Neural Information Processing Systems_ , 2021. 

- Wu, S., Fei, H., Cao, Y., Bing, L., and Chua, T.-S. Information screening whilst exploiting! multimodal relation extraction with feature denoising and multimodal topic modeling. In _Proceedings of the 61st Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)_ , pp. 14734–14751, 2023a. 

- Wu, S., Fei, H., Ji, W., and Chua, T.-S. Cross2StrA: Unpaired cross-lingual image captioning with cross-lingual cross-modal structure-pivoted alignment. In _Proceedings of the 61st Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)_ , pp. 2593–2608, 2023b. 

- Wu, S., Fei, H., Qu, L., Ji, W., and Chua, T.-S. Nextgpt: Any-to-any multimodal llm. _CoRR_ , abs/2309.05519, 2023c. 

- Wu, S., Fei, H., Zhang, H., and Chua, T.-S. Imagine that! abstract-to-intricate text-to-image synthesis with scene graph hallucination diffusion. _Advances in Neural Information Processing Systems_ , 36, 2024. 

- Xiao, J., Shang, X., Yao, A., and Chua, T. Next-qa: Next phase of question-answering to explaining temporal actions. In _Proceedings of the CVPR_ , pp. 9777–9786, 2021. 

11 

**Video-of-Thought: Step-by-Step Video Reasoning from Perception to Cognition** 

- Xu, J., Mei, T., Yao, T., and Rui, Y. MSR-VTT: A large video description dataset for bridging video and language. In _Proceedings of the CVPR_ , pp. 5288–5296, 2016. 

- Ye, Q., Xu, G., Yan, M., Xu, H., Qian, Q., Zhang, J., and Huang, F. Hitea: Hierarchical temporal-aware videolanguage pre-training. In _Proceedings of the IEEE/CVF International Conference on Computer Vision_ , pp. 15405– 15416, 2023. 

   - with graph-of-thought. In _Proceedings of the AAAI Conference on Artificial Intelligence_ , volume 38, pp. 19688– 19696, 2024. 

   - Zolfaghari, M., Singh, K., and Brox, T. Eco: Efficient convolutional network for online video understanding. In _Proceedings of the European conference on computer vision (ECCV)_ , pp. 695–712, 2018. 

- Yu, S., Cho, J., Yadav, P., and Bansal, M. Self-chained image-language model for video localization and question answering. _arXiv preprint arXiv:2305.06988_ , 2023. 

- Yuan, L., Chen, Y., Wang, T., Yu, W., Shi, Y., Jiang, Z.H., Tay, F. E., Feng, J., and Yan, S. Tokens-to-token vit: Training vision transformers from scratch on imagenet. In _Proceedings of the IEEE/CVF international conference on computer vision_ , pp. 558–567, 2021. 

- Zadeh, A., Chan, M., Liang, P. P., Tong, E., and Morency, L.-P. Social-iq: A question answering benchmark for artificial social intelligence. In _Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition_ , pp. 8807–8817, 2019. 

- Zhang, H., Li, X., and Bing, L. Video-llama: An instructiontuned audio-visual language model for video understanding. _CoRR_ , abs/2306.02858, 2023a. 

- Zhang, Y., Li, Y., Cui, L., Cai, D., Liu, L., Fu, T., Huang, X., Zhao, E., Zhang, Y., Chen, Y., et al. Siren’s song in the ai ocean: A survey on hallucination in large language models. _arXiv preprint arXiv:2309.01219_ , 2023b. 

- Zhang, Z., Zhang, A., Li, M., and Smola, A. Automatic chain of thought prompting in large language models. _arXiv preprint arXiv:2210.03493_ , 2022. 

- Zhang, Z., Zhang, A., Li, M., Zhao, H., Karypis, G., and Smola, A. Multimodal chain-of-thought reasoning in language models. _arXiv preprint arXiv:2302.00923_ , 2023c. 

- Zhao, Y., Fei, H., Cao, Y., Li, B., Zhang, M., Wei, J., Zhang, M., and Chua, T.-S. Constructing holistic spatio-temporal scene graph for video semantic role labeling. In _Proceedings of the 31st ACM International Conference on Multimedia_ , pp. 5281–5291, 2023a. 

- Zhao, Y., Fei, H., Ji, W., Wei, J., Zhang, M., Zhang, M., and Chua, T.-S. Generating visual spatial description via holistic 3D scene understanding. In _Proceedings of the 61st Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)_ , pp. 7960–7977, 2023b. 

- Zheng, L., Fei, H., Li, F., Li, B., Liao, L., Ji, D., and Teng, C. Reverse multi-choice dialogue commonsense inference 

12 

**Video-of-Thought: Step-by-Step Video Reasoning from Perception to Cognition** 

## **A. More Configuration Details** 

## **A.1. Detailed Prompt Construction and System I/O** 

Here, we provide detailed prompts as well as their inputs and outputs, for each step of the VoT reasoning framework. 

▶ **Step-1:** If the raw question is a multi-choice question, the prompt for Step-1 should be: 

Step-1: Task Definition and Target Identification for Multi-choice Question 

▶ **Input:** 

_<_ Task Definition _>_ 

Now you are an expert in analyzing video data, and you should answer a question based on the given video. For the question, several candidate answers are provided, where you need to choose [the most suitable option — all possible correct option(s)]. 

_<_ /Task Definition _>_ 

_<_ Input Video _> <_ /Input Video _>_ 

_<_ Question _>_ Given the question: [What is the relationship between the white truck and this neighborhood? A. Transportation B. Buildings C. Clean Services D. Entertainment Facilities], what are the possible targets of the [Video] mainly mentioned or involved? _<_ /Question _>_ 

▶ **Output:** 

The involved targets are [the white truck], [the neighborhood] 

Otherwise for the open-ended format, the prompt will be: 

## Step-1: Task Definition and Target Identification for Open-ended Question 

▶ **Input:** 

_<_ Task Definition _>_ 

Now you are an expert in analyzing video data, and you should answer a question based on the given video. For the question, you should answer in an open-ended format. _<_ /Task Definition _>_ 

_<_ Video _> <_ /Video _>_ 

_<_ Question _>_ Given the question: [What is the relationship between the white truck and this neighborhood?] what are the possible targets of the [Video] mainly mentioned or involved? _<_ /Question _>_ 

▶ **Output:** 

The involved targets are [the white truck], [the neighborhood]. 

▶ **Step-2:** The detailed prompt for Step-2 is shown as follows: 

13 

**Video-of-Thought: Step-by-Step Video Reasoning from Perception to Cognition** 

Step-2: Object Tracking 

▶ **Input:** 

_<_ Question _>_ 

Provide the tracklet of involved [the neighborhood] and [the white truck] by outputting the corresponding partial expression in the [STSG]. _<_ /Question _>_ 

▶ **Output:** 

The partial STSG in tracking [the neighborhood] and [the white truck] is [Frame 1: Objects: [”car-1”: [0.0,13.4,7.0,8.1], ...], Triplets: [(”car-1”,”on the left”, ”street”), ...]...]. 

▶ **Step-3:** The detailed prompt for Step-3 is shown as follows: 

Step-3: Action Analyzing 

▶ **Input:** 

_<_ Question _>_ 

Combining all possible related commonsense, analyze the motion behavior based on the [the white truck] and [the neighborhood] and the neighbor scenes within [STSG]. Describing the action observations and implications. _<_ /Question _>_ 

_<_ STSG _>_ Frame 1: _{_ Objects: [ _{_ ”car-1”: [0.0,13.4,7.0,8.1] _}_ , ...], Triplets: [(”car-1”, ”on the left”, ”street”), ...] _}_ ... _<_ /STSG _>_ 

▶ **Output:** 

The two men are driving the white truck into a neighborhood, and pour the garbage from the roadside trash cans into the white truck. According to commonsense, the white car is used for collecting rubbish... 

▶ **Step-4:** When the raw question is an open-ended QA, we consider prompting the model to output multiple distinct optional answers, such that we unify all QA problems into the Multi-choice type: 

## Step-4-Pre: Transforming Open-ended Question Answering into Multi-choice one 

▶ **Input:** _<_ Question _>_ For the question [What is the relationship between the white truck and this neighborhood?], please based on the action’s [The two men are driving the white truck into a neighborhood...] combined with commonsense, output 4 distinct optional answers with the rationality score of this answer with a 1-10 scale. _<_ /Question _>_ 

## ▶ **Output:** 

Answer A: While the white truck is indeed moving through ... but rather the collection of garbage ... Answer B: ... 

14 

**Video-of-Thought: Step-by-Step Video Reasoning from Perception to Cognition** 

Given the multiple-choice question, we first prompt the model to score its likelihood (from 1 to 10) in conjunction with commonsense knowledge, and provide a corresponding rationale for each candidate answer. Then, we consider a ranking mechanism to determine the final answer. 

## Step-4-A: Multi-choice Question Answering via Ranking 

## ▶ **Input:** 

_<_ Question for Answer A _>_ For the question [What is the relationship between the white truck and this neighborhood? A. Transportation B. Buildings C. Clean Services D. Entertainment Facilities], given a candidate answer [A. Transportation], please based on the action’s [The two men are driving the white truck into a neighborhood...] combined with commonsense, score the rationality of this answer with a 1-10 scale, and also output the rationale. _<_ /Question for Answer A _>_ 

_<_ Question for Answer B _>_ For the question [What is the relationship between the white truck and this neighborhood? A. Transportation B. Buildings C. Clean Services D. Entertainment Facilities], given a candidate answer [B. Buildings], please based on the action’s [The two men are driving the white truck into a neighborhood...] combined with commonsense, score the rationality of this answer with a 1-10 scale, and also output the rationale. _<_ /Question for Answer B _>_ 

## _<_ Question for Answer C _>_ 

For the question [What is the relationship between the white truck and this neighborhood? A. Transportation B. Buildings C. Clean Services D. Entertainment Facilities], given a candidate answer [C. Clean Services], please based on the action’s [The two men are driving the white truck into a neighborhood...] combined with commonsense, score the rationality of this answer with a 1-10 scale, and also output the rationale. _<_ /Question for Answer C _>_ 

_<_ Question for Answer D _>_ For the question [What is the relationship between the white truck and this neighborhood? A. Transportation B. Buildings C. Clean Services D. Entertainment Facilities], given a candidate answer [D. Entertainment Facilities], please based on the action’s [The two men are driving the white truck into a neighborhood...] combined with commonsense, score the rationality of this answer with a 1-10 scale, and also output the rationale. _<_ /Question for Answer D _>_ 

## ▶ **Output:** 

Based on the observations and commonsense, the rationality of the answer [A. Transportation] is 5. While the white truck is indeed moving through ... but rather the collection of garbage ... 

15 

**Video-of-Thought: Step-by-Step Video Reasoning from Perception to Cognition** 

Step-4-B: Multi-choice Question Answering via Ranking 

## ▶ **Input:** 

_<_ Question for Ranking _>_ 

Now, we know the rationale score of the answer [A. Transportation] is 5, ... Please rank the candidate the answer based on the rationale score of each candidate’s answer. 

_<_ /Question for Ranking _>_ 

▶ **Output:** 

Based on the rationality, the ranking results in descending order is [C. Clean Service] [A. Transportation] [D. Entertainment Facilities] [B. Buildings]. So, the final answer is [C. Clean Service]. 

## ▶ **Step-5:** The detailed prompt for Step-5 is shown as follows: 

## Step-5: Answer Verification 

▶ **Input:** 

_<_ Video _> <_ /Video _>_ 

_<_ Question _>_ 

Given the [Video], and the raw question [What is the relationship between the white truck and this neighborhood? A. Transportation B. Buildings C. Clean Services D. Entertainment Facilities], now you need to verify the previous answer by 

1) checking the pixel grounding information if the answer [C. Clean Service] aligns with the facts presented in the video from a perception standpoint; 

2) determining from a cognition perspective if the commonsense implications inherent in the answer [C. Clean Service] contradict any of the main [Observations] inferred in the 3- _rd_ reasoning step. Output the verification result with rationale. 

_<_ /Question _>_ 

## ▶ **Output:** 

Certainly. Let’s verify the answer [C. Clean Service]: 

1. Grounding information: the collection action of garbage, which aligns ... 

2. Commonsense verification: driving a white truck into a neighborhood is associated with waste management ... Overall, the answer [C. Clean Service] is correct. 

Note that the above demonstration of the entire prompting process is just one example. In practice, we simply need to replace the example instance with any other QA instance. It is also worth noting that the VoT system is not limited to video QA; any other video tasks and scenarios requiring reasoning can be adapted to this framework by transforming the original task into a QA format. 

## **A.2. Implementations** 

MotionEpic uses the Vicuna-7B (v1.5)[5] as the backbone LLM. We adopt the ViT-L/14[6] as the video encoder, and use the Q-Former[7] as the projector. All the modules take the default configurations without much modification. For our Recurrent Graph Transformer, we take a 6-layer architecture with 768-d hidden sizes. The text tokenizer is sourced from LLaMA, with approximately 32,000 classes. For each video, we uniformly sample certain frames with a sampling rate of 8 fps for fine-grained reasoning. We note that too large sampling rate introduces noises (i.e., redundant frames) and huge computation cost, while too small one will cause important information loss. Here we use the 8 fps, as in our preliminary study we 

> 5https://huggingface.co/lmsys/vicuna-7b-v1.5 

> 6https://huggingface.co/sentence-transformers/clip-ViT-L-14 

> 7https://huggingface.co/spaces/Salesforce/BLIP2 

16 

**Video-of-Thought: Step-by-Step Video Reasoning from Perception to Cognition** 

verified that it helps achieve the best trade-off. For the fine-tuning setting of end tasks, we will tune the MotionEpic based on the training set using the setting as prior baselines, i.e., data split and evaluation methods. For the zero-shot setting, we will directly perform video QA without using the in-domain training set. 

## **B. More Qualitative Visualizations** 

Finally, we provide two sets of cases for qualitative analyses. We observe that different Video QA datasets exhibit varying biases. Some datasets lean more towards content recognition, relying heavily on perceptual abilities without necessitating much cognitive understanding; others are more inclined towards cognitive-level comprehension, such as physical, cultural or humanities knowledge, where the video content itself is relatively straightforward. We consider cases from both these perspectives. 

Fig. 9 presents two sets of QA cases at the video perception level. For the first question, which requires counting the number of people in the video, it is observed that both baselines provided incorrect answers. However, thanks to our MotionEpic’s utilization of the STSG structured representation, it can accurately ground the number of objects, thereby providing the correct result. In the second case, a straightforward understanding of the temporal information in the video suffices to answer the question. It is shown that both MotionEpic and Video-LLaVA answered correctly. 

Fig. 10 showcases two cases at the cognitive level. For the first case, the question “Where does this scene take place?” can be answered by understanding the scene’s content and combining it with common sense to conclude: a supermarket. For the second case, merely observing the video content “a woman holds a crab with a stick” makes it challenging to grasp the implicit intention. However, integrating some cultural commonsense, it can be understood that the girl is releasing the crab back into the sea. 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/icml_2024_Video-of-Thought_Step-by-Step_Video_Reasoning_from_Perception_to_Cognition_images/icml_2024_Video-of-Thought_Step-by-Step_Video_Reasoning_from_Perception_to_Cognition.pdf-0017-06.png)


**----- Start of picture text -----**<br>
Question (a):  How many people are wearing white clothes?<br>Video-ChatGPT<br>A. Two                                  B. Three                                    C. Five                          D. Six        Video-LLaVA MotionEpic<br>Question (b):  What was the little boy doing before taking the gift?<br>A. Placing a box on the sofa                          Video-LLaVAMotionEpic<br>B. Searching for other gifts<br>C. Communicating with a woman Video-ChatGPT<br>D. Playing beside the sofa<br>**----- End of picture text -----**<br>



![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/icml_2024_Video-of-Thought_Step-by-Step_Video_Reasoning_from_Perception_to_Cognition_images/icml_2024_Video-of-Thought_Step-by-Step_Video_Reasoning_from_Perception_to_Cognition.pdf-0017-07.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/icml_2024_Video-of-Thought_Step-by-Step_Video_Reasoning_from_Perception_to_Cognition_images/icml_2024_Video-of-Thought_Step-by-Step_Video_Reasoning_from_Perception_to_Cognition.pdf-0017-08.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/icml_2024_Video-of-Thought_Step-by-Step_Video_Reasoning_from_Perception_to_Cognition_images/icml_2024_Video-of-Thought_Step-by-Step_Video_Reasoning_from_Perception_to_Cognition.pdf-0017-09.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/icml_2024_Video-of-Thought_Step-by-Step_Video_Reasoning_from_Perception_to_Cognition_images/icml_2024_Video-of-Thought_Step-by-Step_Video_Reasoning_from_Perception_to_Cognition.pdf-0017-10.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/icml_2024_Video-of-Thought_Step-by-Step_Video_Reasoning_from_Perception_to_Cognition_images/icml_2024_Video-of-Thought_Step-by-Step_Video_Reasoning_from_Perception_to_Cognition.pdf-0017-11.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/icml_2024_Video-of-Thought_Step-by-Step_Video_Reasoning_from_Perception_to_Cognition_images/icml_2024_Video-of-Thought_Step-by-Step_Video_Reasoning_from_Perception_to_Cognition.pdf-0017-12.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/icml_2024_Video-of-Thought_Step-by-Step_Video_Reasoning_from_Perception_to_Cognition_images/icml_2024_Video-of-Thought_Step-by-Step_Video_Reasoning_from_Perception_to_Cognition.pdf-0017-13.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/icml_2024_Video-of-Thought_Step-by-Step_Video_Reasoning_from_Perception_to_Cognition_images/icml_2024_Video-of-Thought_Step-by-Step_Video_Reasoning_from_Perception_to_Cognition.pdf-0017-14.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/icml_2024_Video-of-Thought_Step-by-Step_Video_Reasoning_from_Perception_to_Cognition_images/icml_2024_Video-of-Thought_Step-by-Step_Video_Reasoning_from_Perception_to_Cognition.pdf-0017-15.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/icml_2024_Video-of-Thought_Step-by-Step_Video_Reasoning_from_Perception_to_Cognition_images/icml_2024_Video-of-Thought_Step-by-Step_Video_Reasoning_from_Perception_to_Cognition.pdf-0017-16.png)


Figure 9: Qualitative examples of perception-level reasoning. The correct answer is marked with a green checkmark, and the wrong answer is marked with a red cross. 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/icml_2024_Video-of-Thought_Step-by-Step_Video_Reasoning_from_Perception_to_Cognition_images/icml_2024_Video-of-Thought_Step-by-Step_Video_Reasoning_from_Perception_to_Cognition.pdf-0017-18.png)


**----- Start of picture text -----**<br>
Question (a):  Where does this scene take place? Video-ChatGPT<br>A. Supermarket                         B. Amusement Park                      C. Gargen                     D. Campus          MotionEpicVideo-LLaVA<br>Question (b):  What is the woman likely to do next?<br>A. Release the crab back into the sea                         MotionEpic<br>B. Take the crab home for a pet Video-ChatGPT<br>C. Use the stick to explore other marine life on the beach<br>D. Capture the moment with crab and share it on social media       Video-LLaVA<br>**----- End of picture text -----**<br>



![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/icml_2024_Video-of-Thought_Step-by-Step_Video_Reasoning_from_Perception_to_Cognition_images/icml_2024_Video-of-Thought_Step-by-Step_Video_Reasoning_from_Perception_to_Cognition.pdf-0017-19.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/icml_2024_Video-of-Thought_Step-by-Step_Video_Reasoning_from_Perception_to_Cognition_images/icml_2024_Video-of-Thought_Step-by-Step_Video_Reasoning_from_Perception_to_Cognition.pdf-0017-20.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/icml_2024_Video-of-Thought_Step-by-Step_Video_Reasoning_from_Perception_to_Cognition_images/icml_2024_Video-of-Thought_Step-by-Step_Video_Reasoning_from_Perception_to_Cognition.pdf-0017-21.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/icml_2024_Video-of-Thought_Step-by-Step_Video_Reasoning_from_Perception_to_Cognition_images/icml_2024_Video-of-Thought_Step-by-Step_Video_Reasoning_from_Perception_to_Cognition.pdf-0017-22.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/icml_2024_Video-of-Thought_Step-by-Step_Video_Reasoning_from_Perception_to_Cognition_images/icml_2024_Video-of-Thought_Step-by-Step_Video_Reasoning_from_Perception_to_Cognition.pdf-0017-23.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/icml_2024_Video-of-Thought_Step-by-Step_Video_Reasoning_from_Perception_to_Cognition_images/icml_2024_Video-of-Thought_Step-by-Step_Video_Reasoning_from_Perception_to_Cognition.pdf-0017-24.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/icml_2024_Video-of-Thought_Step-by-Step_Video_Reasoning_from_Perception_to_Cognition_images/icml_2024_Video-of-Thought_Step-by-Step_Video_Reasoning_from_Perception_to_Cognition.pdf-0017-25.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/icml_2024_Video-of-Thought_Step-by-Step_Video_Reasoning_from_Perception_to_Cognition_images/icml_2024_Video-of-Thought_Step-by-Step_Video_Reasoning_from_Perception_to_Cognition.pdf-0017-26.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/icml_2024_Video-of-Thought_Step-by-Step_Video_Reasoning_from_Perception_to_Cognition_images/icml_2024_Video-of-Thought_Step-by-Step_Video_Reasoning_from_Perception_to_Cognition.pdf-0017-27.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/icml_2024_Video-of-Thought_Step-by-Step_Video_Reasoning_from_Perception_to_Cognition_images/icml_2024_Video-of-Thought_Step-by-Step_Video_Reasoning_from_Perception_to_Cognition.pdf-0017-28.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/icml_2024_Video-of-Thought_Step-by-Step_Video_Reasoning_from_Perception_to_Cognition_images/icml_2024_Video-of-Thought_Step-by-Step_Video_Reasoning_from_Perception_to_Cognition.pdf-0017-29.png)


Figure 10: Qualitative examples of cognitive-level reasoning. 

17 

