# **DIBS: Enhancing Dense Video Captioning with Unlabeled Videos via Pseudo Boundary Enrichment and Online Refinement** 

Hao Wu[1,2][ ♣] , Huabin Liu[2,3][ ♣] , Yu Qiao[2] , Xiao Sun[2][ �] 

1 2 University of Science and Technology of China Shanghai Artificial Intelligence Laboratory 

3 Shanghai Jiao Tong University 

wuhao@mail.ustc.edu.cn, huabinliu@sjtu.edu.cn, _{_ qiaoyu, sunxiao _}_ @pjlab.org.cn 

## **Abstract** 

_We present_ _**D** ive_ _**I** nto the_ _**B** oundarie_ _**S** (_ _**DIBS** ), a novel pretraining framework for dense video captioning (DVC), that elaborates on improving the quality of the generated event captions and their associated pseudo event boundaries from unlabeled videos. By leveraging the capabilities of diverse large language models (LLMs), we generate rich DVC-oriented caption candidates and optimize the corresponding pseudo boundaries under several meticulously designed objectives, considering diversity, eventcentricity, temporal ordering, and coherence. Moreover, we further introduce a novel online boundary refinement strategy that iteratively improves the quality of pseudo boundaries during training. Comprehensive experiments have been conducted to examine the effectiveness of the proposed technique components. By leveraging a substantial amount of unlabeled video data, such as HowTo100M [16], we achieve a remarkable advancement on standard DVC datasets like YouCook2 [31] and ActivityNet [13]. We outperform the previous state-of-the-art Vid2Seq [27] across a majority of metrics, achieving this with just 0.4% of the unlabeled video data used for pre-training by Vid2Seq._ 

## **1. Introduction** 

Dense video captioning (DVC), a challenging task in video understanding, involves the temporal localization and captioning of all events within an untrimmed video [12]. Compared to standard video captioning that generates a single caption for a short video clip [4, 26, 30], the complexity of dense captioning significantly increases as it requires localizing multiple events in long-term video sequences and much more detailed captioning. 

Particularly, the event boundary is paramount in dense 

> ♣ Co-first authors. Work done as interns at Shanghai AI Laboratory. 

> � Corresponding author. 

video captioning. It provides precise event localization, ensuring the generated captions are accurate, coherent, and contextually relevant. Unfortunately, data containing precise event boundaries is rare and expensive to annotate. This scarcity poses a substantial performance bottleneck. 

Pioneering efforts have been made to address data shortage challenges in DVC. Notably, several weakly supervised approaches [3, 6] have endeavored to approximate fully supervised performance without relying on all the boundary annotations provided by existing datasets, thereby circumventing the need for such annotations. However, these methods take a conservative approach to the problem by carefully re-designing the DVC training or testing framework, aiming to theoretically develop self-sufficient techniques and reduce the reliance on precise boundary annotations. They have not yet substantially incorporated largerscale data for training purposes to drive improved performance. Instead, we take a direct approach to address the fundamental data scarcity issues. Specifically, we introduce an effective method for generating and enhancing pseudo boundaries and harnessing the capabilities of LLMs to produce higher-quality coherent captions. Given the enhanced captions, their corresponding boundaries are generated and further optimized using a carefully designed unified metric and optimization algorithm, ultimately achieving optimal quality. Moreover, we deploy this approach on a substantial volume of unannotated, large-scale video data, effectively bolstering the training data for the DVC task and thereby resulting in a notable performance improvement. 

Similar to our motivation, Vid2Seq [27] also emphasizes the utilization of large-scale unlabeled video data for model training to boost performance. They first propose a singlestage DVC framework that collectively predicts all event captions and corresponding temporal boundaries by generating a single sequence of discrete tokens. Then, this framework is pre-trained on numerous unlabeled narrated videos, where the target event captions and boundaries are obtained directly from the text and timestamps of subtitles 


![](raw/papers/2_LLM_NLP/LLM_paper_list/caption/cvpr_2024_DIBS_Enhancing_Dense_Video_Captioning_with_Unlabeled_Videos_via_Pseudo_Boundary_Enrichment_and_Online_Refinement_images/cvpr_2024_DIBS_Enhancing_Dense_Video_Captioning_with_Unlabeled_Videos_via_Pseudo_Boundary_Enrichment_and_Online_Refinement.pdf-0002-00.png)


Figure 1. Comparison between Vid2Seq [27] and ours. (a) Raw subtitles extracted from videos. (b) Vid2Seq [27] directly converts the raw text and timestamps of subtitles into pseudo event boundaries and captions for pretraining. (c) Our proposed framework utilizes LLMs to generate rich and accurate captions for events from raw narrations. Subsequently, corresponding pseudo boundaries can be generated using these captions through our devised optimization algorithm (cf. Sec. 3.2). 

(as shown in Figure 1(b)). Nevertheless, these raw subtitles usually comprise dialogues, personal reflections, or superfluous background details, and are often temporally misaligned with the visual stream, which introduces significant noise into the event learning. Therefore, while Vid2Seq successfully integrates a larger scale of data for DVC, the effectiveness and utilization efficiency of its large-scale video data still need to be improved. 

To this end, we present **DIBS** , a novel DVC framework that generates accurate event captions along with pseudo boundaries from unlabeled videos to enable effective pretraining for DVC. Specifically, DIBS harnesses diverse offthe-shelf LLMs, leveraging their proficiency in text processing to produce coherent and rich caption candidates for events from unlabeled videos. Moreover, it generates corresponding pseudo event boundaries for each caption, and further optimizes each boundary with a meticulously designed algorithm considering multiple metrics such as diversity, event-centricity, temporal ordering, and coherence. Furthermore, considering that the generated pseudo boundary is still imperfect and includes noises (e.g., background segments), we propose a novel training strategy with online boundary refinement. This strategy aims to iteratively refine and improve the pseudo boundaries during the training phase. Additionally, we seamlessly integrate our ap- 

proach with state-of-the-art DVC training frameworks such as PDVC [25], achieving significantly improved results on extensive benchmarks. 

## **2. Related Work** 

## **2.1. Dense Video Captioning** 

Dense video captioning entails event localization and captioning tasks in long-form videos. Early approaches to this problem typically employed a two-stage ”detect-thendescribe” framework. Within this framework, previous methods [10, 12, 17, 28] focused extensively on improving event representation. For instance, HCN [28] observed that contextual modeling could significantly enhance event captioning performance, while [10, 11] incorporated the audio modality to produce a more robust representation. 

Traditionally, the separation between event localization and captioning posed a gap in methods. However, recent approaches [5, 14, 27, 32, 33] strive for joint learning. MT [32] connected captioning loss and proposal boundaries via a differential masking mechanism for their mutual optimization. PDVC [25], inspired by DETR [2], framed the task as set prediction, enabling simultaneous optimization of both tasks. In contrast, SGR [5] proposed a top-down framework, generating paragraphs before assigning event descriptions to video segments. E2ESG [33] tackled dense video captioning as a unified sequence-to-sequence task using a multimodal Transformer, predicting event locations and captions concurrently. 

Despite their advancements, current DVC algorithms still rely on comprehensive annotations for events, particularly precise event boundaries. This requirement restricts the utilization of large-scale datasets to enhance DVC performance. Recently, Vid2Seq [27] made a groundbreaking attempt by leveraging unlabeled narrated videos for DVC pre-training. However, it directly converts the text and timestamps from narrations into pseudo event captions and boundaries for training, which introduces considerable noise (e.g., backgrounds) into the learning targets. This operation limits the potential benefits of utilizing extensive narrated video datasets for DVC. 

## **2.2. Weakly-supervised Dense Video Captioning** 

Research in weakly-supervised dense video captioning has gained traction due to its potential to bypass the tedious task of annotating precise event boundaries in lengthy videos. WSDEV [6] introduced a cyclical system tackling caption generation and sentence localization as dual tasks. WLT [20] extended this by incorporating audio inputs for improved event captioning. Recently, EC-SL [3] introduced a concept learner to enhance the sentence localizer. However, these methods primarily focus on reducing reliance on precise boundary annotations without incorporating un- 

labeled data for training or significantly improving performance compared to fully supervised approaches. 

## **2.3. Video-Language Pretraining** 

Large-scale video-text pretraining has proven highly effective in diverse video applications like video retrieval, recognition, and non-dense video captioning. UniVL [15], for instance, employed multi-task pretraining on an instructional dataset for video retrieval. All-in-one [24] extended this approach with an end-to-end video-language model, incorporating multiple video datasets to support various downstream tasks like video question answering. 

However, few studies have focused on extensive pretraining for dense video captioning due to the demanding nature of event annotation in these tasks. UEDVC [29] introduced a pretraining task on ActivityNet to enhance DVC performance on the same dataset. Vid2Seq [27] collected and used numerous narrated videos for pretraining in dense video captioning. However, it directly translates raw narrations and timestamps into pseudo event captions and boundaries for DVC pretraining, leading to significant noise and reduced effectiveness in utilizing large-scale narration video data. In contrast, our approach introduces a new pipeline to extract rich and accurate event captions and pseudo boundaries, preserving the wealth of information in large-scale videos for DVC. 

## **3. Approach** 

Given an input video **V** comprising _M_ frames **f** , DVC aims to temporally localize and describe events within the video using natural language. In particular, DVC must predict timestamped boundaries **b** around key moments and generate descriptive captions **c** for each segmented event, resulting in a triplet ({ **f** _m_ ∣ _m_ ∈ _M_ } _,_ { **b** _n_ ∣ _n_ ∈ _N_ } _,_ { **c** _n_ ∣ _n_ ∈ _N_ }), where _N_ represents the number of events. 

Therefore, large-scale pretraining for DVC requires an extensive collection of video data paired with textual descriptions like subtitles or speech-to-text transcriptions, represented as ( **V** _,_ **C** ). However, this data lacks precise boundary information crucial for determining event quantity, positions, and durations. Additionally, the textual descriptions may not consistently align with the requirements of dense captioning, needing a coherent, sequentially narrated description based on pivotal events. 

Previous methods, like Vid2Seq [27], tackled this issue by segmenting subtitles into boundaries and captions. However, these subtitles often contain dialogues, musings, reflections, or irrelevant details, leading to discrepancies between the intended content and the subtitles. This mismatch resulted in inaccuracies in automatically generated captions, misrepresenting the depicted events in the video. 

|Method|Vid2Seq [27]|Ours|
|---|---|---|
|_N_<br>**c**<br>**b**|#Sentences in subtitle<br>Noisy raw sentences in subtitle<br>Timestamps of subtitle|#Events reinterpreted by diverse LLMs<br>Rich, event-centric and sequential<br>Our pseudo boundaries generator in Sec.3.2|



Table 1. Comparison of event caption generation methods between Vid2Seq [27] and our approach in terms of ‘ **b** ’, ‘ **c** ’, and ‘ _N_ ’. 

## **3.1. Prompting LLMs for DVC-Oriented Captions** 

To address the aforementioned issue more effectively, we aim to utilize the capabilities of off-the-shelf LLMs [21, 22] to enrich **C** for the DVC task. LLMs are recognized for their proficiency in text processing, showcasing a remarkable capacity to generate rich and contextually accurate captions, particularly when provided with carefully crafted prompts. **Diverse Off-the-shelf LLMs** Our exploration into diverse LLMs includes both open-source models like LLAMA2 [22] and InternLM [21], as well as closed-source APIbased models such as ChatGPT and Claude. Leveraging diverse LLMs enables us to evaluate different backbones and datasets, tapping into the distinct capabilities of each model. Tailoring carefully selected prompts to the unique characteristics of each LLM, whether open-source or APIbased, enables us to extract accurate and contextually relevant event captions. Integrating a diverse range of LLMs illustrates the adaptability and effectiveness of our methodology, highlighting its versatility across different LLMs and ensuring the extraction of high-quality event captions. 

**Leveraging Sparse Ground Truth Captions as Prompts for Prompt Generation** Initially, we aim to extract the event events descriptions { **c** _n_ ∣ _n_ ∈ _N_ } from the subtitles **C** . To achieve this, we employ a circle-prompting strategy, initially providing a small set of ground truth event captions as hints and querying an LLM for prompts that can generate similar results. We subsequently conduct iterative testing of prompts and captions for better results and manually correct LLM errors in the loop to ensure that our prompts generate rich, accurate, concise, coherent, and event-centric captions. This iterative process seeks to achieve a balanced compromise between precision and conciseness. An example prompt we generate is specified as follows: 

_Task: Extract concise and action-oriented steps or instructions from the video subtitles, focusing solely on the sequential process. Each step should be presented as a single sentence with clear actions. Exclude any steps that are not directly related to the actions in the video. Generate the steps directly without repeating the original text._ 

Consequently, the original subtitles are transformed into logically structured and temporally coherent description of events. Table 1 and Figure 1 present a comparison between our method and Vid2Seq [27] in generating event captions, accompanied by an example. Utilizing various LLMs and prompts, we generate multiple candidate event captions. These candidates, introduced in the next section 

during the pseudo boundary generation process, are optimized with boundary generation. 

## **3.2. Optimization of Pseudo Boundaries** 

After generating event captions { **c** _n_ ∣ _n_ ∈ _N_ }, the subsequent challenge is to ascertain the temporal boundaries { **b** _n_ ∣ _n_ ∈ _N_ } of the corresponding events. In this section, we introduce how to obtain and optimize the corresponding boundaries for each event caption. The optimization comprises two main objectives: first, maximizing alignment between the event caption **c** _n_ and the video clip **f** [ **b** _n_ ]; and second, ensuring that the temporal order relationships between the boundaries reflect those between event captions. 

**Vision-Language Similarity Matrix** For semantic alignment between { **c** _n_ ∣ _n_ ∈ _N_ } and { **f** _m_ ∣ _m_ ∈ _M_ }, we adopt a bottom-up optimization strategy. Initially, a pre-trained vision-language (VL) model, denoted as ( _MV , ML_ ) for vision and language feature extractors, is employed to calculate similarities between individual frames in the video and each caption. This yields a similarity matrix **S** that signifies the associations between frames and captions. 

|**Algorithm 1**Pseudo BoundaryGeneration|**Algorithm 1**Pseudo BoundaryGeneration||
|---|---|---|
|1:|**Input:** event captions{**c**_n_∣_n_∈_N_}; similarity|matrix**S**; user-|
||defned top-_k_parameter_K_, total iterations_Q_.||
|2:|**Initialization:** {**b**0<br>_n_∣_n_∈_N_}←Divide the entire video**V**into||
||_N_ equal segments.||
|3:|**for**each caption**c**_n_ in{**c**_n_∣_n_∈_N_}**do**||
|4:|**for**each iteration_t_in_t_∈_Q_**do**||
|5:|collect top-ˆ_k_frames{_T_ˆ_k,n_∣ˆ_k_ ∈_K_}around**b**_t_<br>_n_.||
|6:|**for**each ˆ_k_in_K_ **do**||
|7:|_D_ˆ_k,n_ =∑<br>_K_<br>_i_=1 ∣_Ti,n_−_T_ˆ_k,n_∣||
|8:|**end for**||
|9:|ˆ_k_∗←ˆ_k_with minimum_D_ˆ_k,n._||
|10:|new boundary center←_T_ˆ_k_∗_,n_||
|11:|new boundary size←2∗_α_∗_stdn._||
|12:|update boundary←**b**_t_<br>_n_ in Eq.3||
|13:|record loss←_Lt_ in Eq.4.||
|14:|**end for**||
|15:|select iter with minimum loss_q_ =argmin|_Lt, t_∈_Q_|
||_t_||
|16:|determine the pseudo boundary for caption**c**_n_: **b**_n_ ←**b**_q_<br>_n_.||
|17:|determine the loss for captions : _L_←_Lq_.||
|18:|**end for**||
|19:|**Output:** Generated pseudo boundaries{**b**_n_∣_n_|∈_N_}and loss|
||_L_for input event captions{**c**_n_∣_n_∈_N_}.||




![](raw/papers/2_LLM_NLP/LLM_paper_list/caption/cvpr_2024_DIBS_Enhancing_Dense_Video_Captioning_with_Unlabeled_Videos_via_Pseudo_Boundary_Enrichment_and_Online_Refinement_images/cvpr_2024_DIBS_Enhancing_Dense_Video_Captioning_with_Unlabeled_Videos_via_Pseudo_Boundary_Enrichment_and_Online_Refinement.pdf-0004-05.png)


_Note that when employing a video-language model, a short video clip centered around a frame is utilized to represent this frame._ For implementation, image-language models such as CLIP [18] and video-language models like UniVL [15] are utilized. In the experiments, we aggregate scores from multiple vision-language models, averaging them to yield a more robust similarity matrix. 

_Discussion: the domain gap and noisy detection issue._ Evidently, the quality of the similarity matrix significantly depends on the VL models’ quality and the domain gap between the dataset used for VL model training and the DVC dataset. On the one hand, this underscores the importance of employing diverse LLMs to generate rich caption candidates. On the other hand, despite these concerted efforts, the similarity matrix still exhibits notable detection noise and, in some instances, false positives. 

## **Caption-Aware Pseudo Boundary Generation with Soft** 

**Time Constraints** Leveraging the similarity matrix **S** , we convert the event localization problem into the identification of the optimal _N_ frames in { **f** _m_ ∣ _m_ ∈ _M_ } corresponding to captions { **c** _n_ ∣ _n_ ∈ _N_ }. This process should adhere to the temporal order specified in { **c** _n_ ∣ _n_ ∈ _N_ }, resulting in a drop dynamic time warping (Drop-DTW) [7] problem. We denote this baseline method as the **Drop-DTW Baseline** . 

However, due to the noisy detection issue discussed above, the effectiveness of the direct Drop-DTW Baseline method is constrained. This ineffectiveness arises from several reasons: 1) **Multimodal Responses:** Higher response 

positions often exhibit multiple peaks, with their distribution in the video lacking concentration. 2) **Non-Uniform Event Coverage:** Events do not exhibit a preference for a uniformly distributed coverage across the majority of the entire video. 3) **Strict Temporal Ordering Constraints:** the hard global temporal ordering constraints in Drop-DTW not only slow down the optimization process significantly but also impede individual captions from effectively finding the best-matching video intervals; 4) **Duration Determination:** Additionally, determining the duration of events lacks a straightforward, intuitive method. 

To address these issues, we propose a caption-aware pseudo boundary generation algorithm with soft time constraints, as illustrated in Algorithm 1. To encourage events cover most of the video length _evenly_ , we initialize the boundaries **b**[0] by dividing the video into _N_ equal segments. Subsequently, we iteratively optimize the boundaries, as illustrated in Figure 2. For the _n_ th caption, in its _t_ th iteration, we first collect a _local_ set of top- _k_[ˆ] frames around **b** _n[t]_[−][1] (the boundary range in the previous iteration, starting from **b**[0] _n_[)][with][the][highest][similarity][scores][to] **[c]** _n_[.][From][these] _[k]_[ˆ] positions, we select the new boundary center. For each of the _k_[ˆ] frames, we calculate the total temporal distance from this frame to the other _k_[ˆ] − 1 frames. The frame with the minimum total distance indicates that these top _k_[ˆ] frames are more concentrated around it, thus setting it as the new boundary center, demoted as _Tk_ ˆ∗ _,n_ . The size of the boundary is determined by calculating the standard deviation with respect to the new boundary center as shown in Equation 2. 


![](raw/papers/2_LLM_NLP/LLM_paper_list/caption/cvpr_2024_DIBS_Enhancing_Dense_Video_Captioning_with_Unlabeled_Videos_via_Pseudo_Boundary_Enrichment_and_Online_Refinement_images/cvpr_2024_DIBS_Enhancing_Dense_Video_Captioning_with_Unlabeled_Videos_via_Pseudo_Boundary_Enrichment_and_Online_Refinement.pdf-0005-00.png)


**----- Start of picture text -----**<br>
Prior Range of Boundary<br>Video Frames New Range of Boundary<br>Event-1 𝑇<br>Event-2<br>Event-3<br>Event-4<br>Event-5<br>Event-6<br>**----- End of picture text -----**<br>


Figure 2. Prior range division and frame processing in pseudo boundary generation. Frames showcasing top- _k_ global similarity to Event-3 are indicated by deep green points, contrasting with lighter green points denoting similarity outside the top- _k_ range. The blue box delineates the evenly divided prior range, while the orange box signifies the adjusted range after an iteration. The pseudo boundary generation process is outlined in Algorithm 1. 


![](raw/papers/2_LLM_NLP/LLM_paper_list/caption/cvpr_2024_DIBS_Enhancing_Dense_Video_Captioning_with_Unlabeled_Videos_via_Pseudo_Boundary_Enrichment_and_Online_Refinement_images/cvpr_2024_DIBS_Enhancing_Dense_Video_Captioning_with_Unlabeled_Videos_via_Pseudo_Boundary_Enrichment_and_Online_Refinement.pdf-0005-02.png)


To this end, we gain a coarse boundary **b** _coarse,n_ and we further determine the actual boundary **b** _[t] n_[of][the][boundary] by selecting the minimum and maximum values of _k_ frames within the new coarse boundary as depicted in Equation 3: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/caption/cvpr_2024_DIBS_Enhancing_Dense_Video_Captioning_with_Unlabeled_Videos_via_Pseudo_Boundary_Enrichment_and_Online_Refinement_images/cvpr_2024_DIBS_Enhancing_Dense_Video_Captioning_with_Unlabeled_Videos_via_Pseudo_Boundary_Enrichment_and_Online_Refinement.pdf-0005-04.png)


where _α_ is a hyperparameter determining the size of the coarse boundary. After each iteration, we define a loss function to evaluate the quality of the generated boundary, as shown in Equation 4: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/caption/cvpr_2024_DIBS_Enhancing_Dense_Video_Captioning_with_Unlabeled_Videos_via_Pseudo_Boundary_Enrichment_and_Online_Refinement_images/cvpr_2024_DIBS_Enhancing_Dense_Video_Captioning_with_Unlabeled_Videos_via_Pseudo_Boundary_Enrichment_and_Online_Refinement.pdf-0005-06.png)


Here, the loss is the total distance weighted by the similarity score between the frame and caption. The function _Dis_ (⋅) measures the distance between a frame _f_ and the boundary **b** , where **b** _s_ and **b** _e_ represent the start and end of the boundary respectively. The function _Dis_ ( _f,_ **b** ) is positive when _f_ is within the boundary range and negative otherwise. We then collect the local top- _k_ frames with the highest similarity scores to the _n_ th caption around the new boundary and proceed to the next iteration. Finally, we select the boundary with the minimum loss value as the final optimized boundary. Refer to Algorithm 1 for further details on generating the pseudo boundaries. 

## **3.3. Training with Online Boundary Refinement** 

Although we can employ various fully supervised approaches [25, 29, 33] to train a DVC model using gener- 


![](raw/papers/2_LLM_NLP/LLM_paper_list/caption/cvpr_2024_DIBS_Enhancing_Dense_Video_Captioning_with_Unlabeled_Videos_via_Pseudo_Boundary_Enrichment_and_Online_Refinement_images/cvpr_2024_DIBS_Enhancing_Dense_Video_Captioning_with_Unlabeled_Videos_via_Pseudo_Boundary_Enrichment_and_Online_Refinement.pdf-0005-10.png)


**----- Start of picture text -----**<br>
Proposal<br>sampling<br>Event<br>Event<br>Inaccurate Bag head score Selected<br>Pseudo  Proposals<br>Proposal<br>Boundary score Merge<br>Cap  Caption<br>head score Refined<br>Matched  “Add water to the incubator” Pseudo<br>Stage-1 queries Boundary<br>Proposal<br>sampling Event<br>Event<br>Refined Bag head score Selected<br>Pseudo  Proposal  Proposals<br>Boundary score Merge<br>Cap  Caption<br>head score Refined<br>Matched  “Add water to the incubator” Pseudo<br>Stage-2 queries Boundary<br>… Final refined<br>Stage-K<br>Pseudo Boundary<br>**----- End of picture text -----**<br>


Figure 3. The process of online pseudo boundary refinement. 

ated event captions and pseudo boundaries, these boundaries might be imperfect, encompassing inaccuracies like background elements. Utilizing them directly as ground truth during DVC pretraining could mislead the event learning process, especially in event localization. To counter this, we introduce an online strategy for refining pseudo boundaries, seamlessly integrating it with cutting-edge DVC training frameworks like PDVC [25]. 

**Background Review of PDVC** PDVC [25] is an effective DVC framework employing parallel decoding and set prediction principles. It starts by utilizing a pre-trained video feature extractor and a transformer encoder to derive a sequence of frame-level features. Then, employing _M_ learnable event queries { **q** _i_ } _[M] i_ =1[,][it][employs][a][transformer][de-] coder along with three prediction heads (localization head, caption head, and event counter) to simultaneously predict _M_ boundaries, _M_ captions, and event count. During inference, the model selects the top detected events by ranking captioning and localization scores, without using nonmaximum suppression (NMS) [9]. 

**Online Pseudo Boundary Refinement** We refine pseudo boundaries by augmenting their quantity and conducting quality evaluations. As shown in Figure 3, consider a pseudo boundary **b** = ( _t, d_ ) of caption **c** , we employ a standard jitter augmentation on the boundary duration _d_ and the boundary center _t_ with jitter ratio _r_ 1 and _r_ 2 respectively, obtaining an augmented set of proposals, denoted as _B_ = { **b**[ˆ] _u_ } _u[U]_ =1[(] _[U]_[proposals][in][total][and] **[b]**[is][contained][in] _B_ as well), representing potentially superior event segments for caption **c** . Then, following PDVC, we adopt the Hungarian matching method [2] between all proposals { **b**[ˆ] _u_ } _u[U]_ =1 and query embeddings { **q** _i_ } _i[M]_ =1[to link each proposal with a] 

specific query, yielding { **Q** _u_ } _u[U]_ =1[(] **[Q]** _u_[=] **[ q]** _i_[, if the] _[ i]_[th query] is linked to the _u_ th proposal). The linked query serves as the proxy feature for the proposal. 

Utilizing the augmented set of proposals and their proxy features, we evaluate their quality online during training. PDVC offers an event classification head **h** _[e]_ and a caption scoring head **h** _[c]_ , both using the query embeddings as input and producing confidence scores. Formally, 


![](raw/papers/2_LLM_NLP/LLM_paper_list/caption/cvpr_2024_DIBS_Enhancing_Dense_Video_Captioning_with_Unlabeled_Videos_via_Pseudo_Boundary_Enrichment_and_Online_Refinement_images/cvpr_2024_DIBS_Enhancing_Dense_Video_Captioning_with_Unlabeled_Videos_via_Pseudo_Boundary_Enrichment_and_Online_Refinement.pdf-0006-02.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/caption/cvpr_2024_DIBS_Enhancing_Dense_Video_Captioning_with_Unlabeled_Videos_via_Pseudo_Boundary_Enrichment_and_Online_Refinement_images/cvpr_2024_DIBS_Enhancing_Dense_Video_Captioning_with_Unlabeled_Videos_via_Pseudo_Boundary_Enrichment_and_Online_Refinement.pdf-0006-03.png)


Note that the event score **p** _[e] u_[is][normalized][with][the][Sig-] moid activation while the caption score **p** _[c] u_[is][normalized] with the Softmax activation across all proposals in _B_ . The final assessment score **p** _u_ for the proposal **b**[ˆ] _u_ is calculated as **p** _u_ = **p** _[e] u_[+] **[ p]** _[c] u_[.][A higher proposal score means this pro-] posal better represents the boundary of caption **c** . 

Finally, we select the top- _K_ proposals in { **b**[ˆ] _u_ } _u[U]_ =1[with] the highest scores and compute a weighted average of their boundaries as the final refined boundary. 


![](raw/papers/2_LLM_NLP/LLM_paper_list/caption/cvpr_2024_DIBS_Enhancing_Dense_Video_Captioning_with_Unlabeled_Videos_via_Pseudo_Boundary_Enrichment_and_Online_Refinement_images/cvpr_2024_DIBS_Enhancing_Dense_Video_Captioning_with_Unlabeled_Videos_via_Pseudo_Boundary_Enrichment_and_Online_Refinement.pdf-0006-06.png)


The refined pseudo boundary **b** _ref_ replaces the original boundary **b** for subsequent training stages. This iterative process, shown in Figure 3, refines each pseudo boundary in multiple stages, resulting in increasingly accurate boundaries. Our implementation defaults to a 2-stage refinement process to derive the final boundary. Importantly, boundary refinement doesn’t impact model inference, incurring no additional computational overhead during this phase. 

## **4. Experiments** 

## **4.1. Experimental Setting** 

**Datasets** Our experiments encompass two prominent datasets commonly employed for dense video captioning: YouCook2 [31] and ActivityNet Captions [13]. For pretraining, we leverage a subset of the HowTo100M [16] dataset, specifically focusing on cooking videos, amounting to approximately 56,000 videos. 

**Implementation Details** In our setup, we uniformly sample video frames at 1 FPS and adjust them to a fixed count denoted as _F_ . For YouCook2, _F_ is set to 200, and for ActivityNet Captions, it’s 100. We utilize pretrained visionlanguage models like CLIP [18] and UniVL [15] to extract frame-level features across all datasets. The model undergoes a two-stage training process: initially, a 10-epoch pretraining on a subset of HowTo100M, followed by a 20epoch fine-tuning phase on each target dataset. Notably, to 

address the domain gap between the HowTo100M dataset and the target dataset, we augment the training data by incorporating the target dataset using pseudo-boundaries during pretraining. Our model architecture mirrors PDVC [25], incorporating a transformer encoder, transformer decoder, and three prediction heads. 

**Evaluation Metrics** We employ standard captioning metrics: METEOR [1] (M) for semantic similarity, and CIDEr [23] (C) for human judgment correlation. For event localization evaluation, we utilize average precision (Pre.) and recall (Rec.) at Intersection over Union(IoU) thresholds of 0.3, 0.5, 0.7 and 0.9, along with the overall F1 score for boundary prediction. These metrics are computed using the official ActivityNet challenge toolbox. Additionally, we incorporate the SODA ~~c~~ [8] (S) metric, which provides a joint evaluation of caption quality and localization accuracy. 

## **4.2. Comparison with State-of-the-art Methods** 

In Table 2, we compare our captioning performance against state-of-the-art approaches. We evaluate our model under two settings: weakly supervised and fully supervised paradigms. In the weakly supervised setting, our model is trained directly on the target dataset using ground truth captions and generated pseudo boundaries, without using any ground truth boundaries. It can be observed that this strategy achieves comparable performance compared to prior weakly supervised DVC methods, demonstrating the effectiveness of our pseudo boundary generation and refinement. Regarding the standard fully-supervised setting, our approach outperforms PDVC [25] with the same backbone architecture after pretraining. This illustrates the significant benefits of pretraining for DVC. Particularly, on YouCook2 and ActivityNet, our models with UniVL and CLIP backbones surpass previous methods across multiple metrics. Notably, Vid2Seq incorporates additional speech cues during inference, providing a richer multi-modal context. Besides, Vid2Seq uses much larger-scale pretraining data (15M videos) and captioning models (T5 [19]) compared to our DIBS. These factors likely explain some performance differences relative to Vid2Seq, particularly with the CLIP backbone. However, using the UniVL backbone, our model can surpass Vid2Seq on YouCook2 by a significant margin. On ActivityNet, our CLIP-based approach also achieves state-of-the-art results. 

In Table 3, we see that our pretraining consistently enhances the event localization, enabling our DIBS to outperform previous methods on YouCook2 by a great margin. However, on ActivityNet, our approach lags behind PDVC and UEDVC [29] while outperforming Vid2Seq. This disparity may be due to the domain gap between instructional and activity videos, impacting the localization of less common events despite improvements in captioning. 

|Training|Method|Settings<br>Pretrain<br>Backbone|YouCook2<br>M<br>C<br>S|ActivityNet<br>M<br>C<br>S||||||
|---|---|---|---|---|---|---|---|---|---|
|Weakly-<br>Supervised<br>WLT [20]<br>∅<br>TSN<br>-<br>-<br>-<br>4.93<br>13.79<br>-<br>WS-DEC [6]†<br>∅<br>C3D<br>-<br>-<br>-<br>6.30<br>18.77<br>-<br>EC-SL [3]†<br>∅<br>C3D<br>-<br>-<br>-<br>7.49<br>21.21<br>-<br>DIBS (Ours)<br>∅<br>CLIP<br>4.63<br>22.05<br>4.03<br>7.33<br>15.86<br>4.67<br>DIBS (Ours)<br>∅<br>UniVL<br>5.90<br>29.62<br>4.96<br>6.76<br>13.75<br>4.26||||||Method|Settings<br>Pretrain<br>Backbone|YouCook2<br>Rec.<br>Pre.|ActivityNet<br>Rec.<br>Pre.|
|||||||PDVC [25]<br>PDVC [25]†<br>PDVC [25]†<br>UEDVC [29]<br>E2ESG [33]<br>Vid2Seq [27]<br>**DIBS (Ours)**<br>**DIBS (Ours)**|∅<br>TSN<br>∅<br>CLIP<br>∅<br>UniVL<br>676k<br>TSN<br>∅<br>C3D<br>15M<br>CLIP<br>56k<br>CLIP<br>56k<br>UniVL|-<br>-<br>21.76<br>31.92<br>29.66<br>42.04<br>-<br>-<br>3.49<br>25.00<br>27.90<br>27.80<br>26.24<br>39.18<br>**30.80**<br>**45.13**|55.42<br>58.07<br>50.82<br>55.73<br>53.21<br>59.46<br>**59.00**<br>**60.32**<br>-<br>-<br>52.70<br>53.90<br>53.14<br>58.31<br>53.02<br>58.39|
|Fully-<br>Supervised|PDVC [25]<br>PDVC [25]†<br>PDVC [25]†<br>UEDVC [29]<br>E2ESG [33]<br>Vid2Seq [27]<br>**DIBS (Ours)**<br>**DIBS (Ours)**|∅<br>TSN<br>∅<br>CLIP<br>∅<br>UniVL<br>676k<br>TSN<br>∅<br>C3D<br>15M<br>CLIP<br>56k<br>CLIP<br>56k<br>UniVL|4.74<br>22.71<br>4.42<br>5.47<br>28.37<br>5.00<br>7.87<br>46.02<br>6.87<br>2.18<br>8.37<br>3.34<br>3.49<br>25.00<br>-<br>9.30<br>47.10<br>7.90<br>7.51<br>44.44<br>6.39<br>**9.41**<br>**59.35**<br>**7.97**|7.96<br>28.96<br>5.44<br>8.31<br>30.11<br>5.63<br>8.24<br>28.21<br>5.43<br>-<br>-<br>5.49<br>-<br>-<br>-<br>8.50<br>30.10<br>5.80<br>**8.93**<br>**31.89**<br>**5.85**<br>8.25<br>28.85<br>5.35||||||
|||||||||||



Table 3. Performance of event localization on YouCook2 and ActivityNet.[†] Results are obtained from our implementation with official codebase. 

Table 2. Performance of caption generation on YouCook2 and ActivityNet. † Results are obtained from our implementation with official codebase. 

|Settings<br>Feature<br>Boundary<br>Caption|Metrics<br>M<br>C<br>S<br>Rec.<br>Pre.|
|---|---|
|1.<br>CLIP<br>✗<br>✗<br>2.<br>CLIP<br>✗<br>Cost<br>3.<br>CLIP<br>✗<br>Sim<br>4.<br>CLIP<br>✓<br>✗|2.32<br>9.07<br>3.40<br>18.47<br>25.01<br>1.68<br>7.09<br>2.79<br>14.19<br>16.63<br>2.18<br>8.37<br>3.34<br>**20.77**<br>24.70<br>**4.62**<br>**21.93**<br>**3.73**<br>15.48<br>**28.82**|
|1.<br>UniVL<br>✗<br>✗<br>2.<br>UniVL<br>✗<br>Cost<br>3.<br>UniVL<br>✗<br>Sim<br>4.<br>UniVL<br>✓<br>✗|2.66<br>9.97<br>3.21<br>15.62<br>25.18<br>1.99<br>8.49<br>3.15<br>18.75<br>21.08<br>2.27<br>9.16<br>3.16<br>15.83<br>24.88<br>**5.88**<br>**28.04**<br>**4.47**<br>**19.72**<br>**35.43**|



In summary, our proposed pretraining approach substantially boosts event localization on well-matched instructional datasets like YouCook2. Regarding general activity localization, additional techniques may be necessary to transfer specific knowledge from the instructional videos due to the clear domain gap. Therefore, taking careful consideration of domain similarity is critical when leveraging unlabeled videos for DVC pretraining. 

## **4.3. Ablation Study** 

Table 4. Performance comparison of event localization and caption generation on YouCook2 with and without pseudo boundaries. 

**Effects of Pseudo Boundary** We examine the impact of pseudo boundaries on YouCook2, comparing against a baseline where ground truth event boundaries are omitted. When utilizing pseudo boundaries, matching between proposals and captions primarily relies on the Generalized IoU (GIoU) cost. To provide a comprehensive comparison, we construct two additional settings, both incorporating caption information similar to our pseudo boundary. The first setting introduces an additional caption cost for matching, replacing the GIoU cost, while the second setting incorporates a caption-proposal similarity cost in place of the GIoU cost. In Table 4, we present a comprehensive comparison of model performance on event localization and caption generation using CLIP and UniVL as backbones. The results demonstrate a significant improvement in model performance when employing pseudo boundaries, whereas the other two settings exhibit inferior performance compared to the baseline at most metrics. This corroborates the effectiveness of generating pseudo boundaries using captions as a powerful method for leveraging caption information in event localization tasks. 

|Dataset|Features<br>STC|M<br>C<br>S<br>Rec.<br>Pre.|
|---|---|---|
|YouCook2|CLIP<br>✗<br>CLIP<br>✓|3.92<br>17.05<br>3.57<br>14.29<br>25.87<br>**4.62**<br>**21.93**<br>**3.73**<br>**15.48**<br>**28.82**|
||UniVL<br>✗<br>UniVL<br>✓|5.43<br>25.57<br>4.47<br>18.05<br>32.83<br>**5.88**<br>**28.04**<br>**4.47**<br>**19.72**<br>**35.43**|
|ActivityNet|CLIP<br>✗<br>CLIP<br>✓|6.60<br>15.23<br>3.87<br>35.31<br>57.68<br>**7.23**<br>**16.63**<br>**4.68**<br>**42.44**<br>**65.46**|
||UniVL<br>✗<br>UniVL<br>✓|5.88<br>13.60<br>3.68<br>35.72<br>55.14<br>**6.75**<br>**14.13**<br>**4.21**<br>**42.24**<br>**64.75**|



Table 5. Comparison of model performance with and without soft time constraints. STC denotes soft time constraints. 

troducing soft time constraints notably enhances model performance in event localization and caption generation. This improvement underscores the efficacy of soft time constraints in generating higher-quality pseudo boundaries. The sequential nature of events in videos highlights the significance of maintaining temporal order, even in scenarios with potential event overlap, as observed in ActivityNet. 

**Effects of Soft Time Constraints** We assess the impact of soft time constraints on pseudo boundaries by excluding ground truth boundaries from both the YouCook2 and ActivityNet. Our method without soft time constraints performs global video iteration for each caption. Table 5 delineates model performance with and without soft time constraints, employing CLIP and UniVL backbones. In- 

**Effects of Boundary Refinement** To assess the impact of boundary refinement, we employ two distinct settings on the YouCook2 and ActivityNet Caption datasets. In the first setting, we examine performance by omitting the ground truth boundary, utilizing pseudo boundaries both with and without refinement. In the second setting, we perform pretrain- 

|Dataset|Pretrain<br>Refne|M<br>C<br>S<br>Rec.<br>Pre.|
|---|---|---|
|YouCook2|✗<br>✗<br>5.88<br>28.04<br>4.47<br>19.72<br>35.43<br>✗<br>✓<br>**5.90**<br>**29.62**<br>**4.96**<br>**30.80**<br>**45.13**||
||✓<br>✗<br>✓<br>✓|9.04<br>54.98<br>**8.13**<br>**31.55**<br>44.95<br>**9.41**<br>**59.35**<br>7.97<br>30.80<br>**45.13**|
|ActivityNet|✗<br>✗<br>7.23<br>**16.63**<br>**4.68**<br>**42.44**<br>65.46<br>✗<br>✓<br>**7.33**<br>15.86<br>4.67<br>42.21<br>**65.79**||
||✓<br>✗<br>✓<br>✓|**8.94**<br>30.49<br>5.39<br>51.45<br>57.49<br>8.93<br>**31.89**<br>**5.85**<br>**53.14**<br>**58.31**|



Table 6. Comparison between models with and without pseudo boundary refinement. 

|Dataset|Pretrain<br>Features|M<br>C<br>S<br>Rec.<br>Pre.|
|---|---|---|
|YouCook2|✗<br>CLIP<br>✓<br>CLIP<br>✗<br>UniVL<br>✓<br>UniVL|5.47<br>28.37<br>5.00<br>21.76<br>31.92<br>7.51<br>44.44<br>6.39<br>26.24<br>39.18<br>7.87<br>46.02<br>6.87<br>29.66<br>42.04<br>**9.41**<br>**59.35**<br>**7.97**<br>**30.80**<br>**45.13**|
|ActivityNet|✗<br>CLIP<br>✓<br>CLIP<br>✗<br>UniVL<br>✓<br>UniVL|8.31<br>30.11<br>5.63<br>50.82<br>55.73<br>**8.93**<br>**31.89**<br>**5.85**<br>53.14<br>58.31<br>8.24<br>28.21<br>5.43<br>**53.21**<br>**59.46**<br>8.25<br>28.85<br>5.35<br>53.02<br>58.39|



Table 7. Comparative analysis of model performance with and without pretraining on YouCook2 and ActivityNet Datasets. 

ing on the HowTo100M subset with and without refinement, followed by fine-tuning the pretrained models on target datasets like YouCook2 and ActivityNet within our study. In Table 6, we present a comprehensive model performance comparison with and without refinement. Refinement enhances model performance on localization and caption metrics across both YouCook2 and ActivityNet datasets. However, while most metrics improve, a few experience a slight drop, and this discrepancy varies between the YouCook2 and ActivityNet datasets. We attribute this observation to the domain gap between the two datasets. 

**Effect of Pretraining** To examine pretraining effects, we conducted a comprehensive comparison of model performance with and without pretraining using CLIP and UniVL backbones on YouCook2 and ActivityNet datasets. Table 7 details performance for event localization and caption generation. It indicates improved caption generation on both datasets after pretraining, yet stable improvements in event localization are not consistently observed, especially on ActivityNet. The enrichment of pseudo captions from large-scale unlabeled videos, generated by LLMs, likely contributes to enhanced caption performance. Despite employing diverse strategies to improve the quality of pseudo boundaries, the substantial distance from ground truth boundaries persists, possibly explaining the limited impact of pretraining on event localization. We also examined the impact of varying pretraining data amounts, please refer to our supplementary materials. 

|Dataset|Pretrain<br>FT Data|M<br>C<br>S<br>Rec.<br>Pre.<br>F1|
|---|---|---|
|YouCook2|✗<br>100%<br>✓<br>25%<br>✓<br>50%<br>✓<br>75%<br>✓<br>100%|7.87<br>46.02<br>6.87<br>29.66<br>42.04<br>34.78<br>7.81<br>46.69<br>7.17<br>28.93<br>41.08<br>33.95<br>8.60<br>55.73<br>7.84<br>30.14<br>42.73<br>35.34<br>9.11<br>59.09<br>7.86<br>29.97<br>44.54<br>35.83<br>**9.41**<br>**59.35**<br>**7.97**<br>**30.80**<br>**45.13**<br>**36.61**|
|ActivityNet|✗<br>100%<br>✓<br>25%<br>✓<br>50%<br>✓<br>75%<br>✓<br>100%|8.31<br>30.11<br>5.63<br>50.82<br>55.73<br>53.16<br>8.66<br>28.24<br>5.01<br>51.08<br>56.24<br>53.54<br>8.56<br>30.09<br>5.39<br>51.96<br>56.26<br>54.02<br>8.83<br>30.80<br>5.55<br>52.90<br>57.19<br>54.96<br>**8.93**<br>**31.89**<br>**5.85**<br>**53.14**<br>**58.31**<br>**55.61**|



Table 8. Few-shot performance fine-tuning on partial data. “FT data” represents the percentage of data used for fine-tuning. 

**Few-shot Dense Video Captioning** In Table 8, we depict the model’s performance concerning varying proportions of fine-tuning data on YouCook2 and ActivityNet datasets. The results show a direct association with the amount of fine-tuning data used. Notably, even with a fraction of the complete training set post-pretraining, the model surpasses the performance of the setting without pretraining and using the entire training set. This distinction is particularly prominent on YouCook2, where achieving superior performance with only half of the target training set is feasible. Conversely, ActivityNet requires more training data to achieve comparable advancements. 

**Backbone Influence in Dense Video Captioning** Table 7 compares CLIP and UniVL features, both with and without pretraining. UniVL demonstrates better performance on YouCook2, possibly due to its pretraining on instructional videos, narrowing the domain gap. In contrast, CLIP excels on ActivityNet, leveraging its strong generalization abilities. Our findings emphasize the importance of selecting backbones pretrained on tasks aligned with the dataset. UniVL suits instructional videos like YouCook2, while CLIP’s broader generalization is advantageous for diverse datasets like ActivityNet, highlighting the need for a tailored approach based on dataset characteristics and backbone pretraining specifics in dense video captioning tasks. 

## **5. Conclusion** 

We introduce **DIBS** , a novel pretraining framework for DVC that improves the quality of pseudo event boundaries and captions derived from large-scale unlabeled videos. Leveraging LLMs, we generate and optimize pseudo boundaries and captions simultaneously, emphasizing objectives like diversity and coherence. We also propose an online boundary refinement strategy to further improve the quality of pseudo boundaries. Extensive experiments validate the effectiveness of DIBS. 

**Acknowledgments** This work is partially supported by the National Key R&D Program of China (NO.2022ZD0160104). 

## **References** 

- [1] Satanjeev Banerjee and Alon Lavie. Meteor: An automatic metric for mt evaluation with improved correlation with human judgments. In _Proceedings of the acl workshop on intrinsic and extrinsic evaluation measures for machine translation and/or summarization_ , pages 65–72, 2005. 6 

- [2] Nicolas Carion, Francisco Massa, Gabriel Synnaeve, Nicolas Usunier, Alexander Kirillov, and Sergey Zagoruyko. End-toend object detection with transformers. In _European conference on computer vision_ , pages 213–229. Springer, 2020. 2, 5 

- [3] Shaoxiang Chen and Yu-Gang Jiang. Towards bridging event captioner and sentence localizer for weakly supervised dense event captioning. In _Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition_ , pages 8425–8435, 2021. 1, 2, 7 

- [4] Yangyu Chen, Shuhui Wang, Weigang Zhang, and Qingming Huang. Less is more: Picking informative frames for video captioning. In _Proceedings of the European conference on computer vision (ECCV)_ , pages 358–373, 2018. 1 

- [5] Chaorui Deng, Shizhe Chen, Da Chen, Yuan He, and Qi Wu. Sketch, ground, and refine: Top-down dense video captioning. In _Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition_ , pages 234–243, 2021. 2 

- [6] Xuguang Duan, Wenbing Huang, Chuang Gan, Jingdong Wang, Wenwu Zhu, and Junzhou Huang. Weakly supervised dense event captioning in videos. _Advances in Neural Information Processing Systems_ , 31, 2018. 1, 2, 7 

- [7] Mikita Dvornik, Isma Hadji, Konstantinos G Derpanis, Animesh Garg, and Allan Jepson. Drop-dtw: Aligning common signal between sequences while dropping outliers. _Advances in Neural Information Processing Systems_ , 34:13782–13793, 2021. 4 

- [8] Soichiro Fujita, Tsutomu Hirao, Hidetaka Kamigaito, Manabu Okumura, and Masaaki Nagata. Soda: Story oriented dense video captioning evaluation framework. In _Computer Vision–ECCV 2020: 16th European Conference, Glasgow, UK, August 23–28, 2020, Proceedings, Part VI 16_ , pages 517–531. Springer, 2020. 6 

- [9] Ross Girshick. Fast r-cnn. In _Proceedings of the IEEE International Conference on Computer Vision_ , pages 1440–1448, 2015. 5 

- [10] Vladimir Iashin and Esa Rahtu. A better use of audio-visual cues: Dense video captioning with bi-modal transformer. _arXiv preprint arXiv:2005.08271_ , 2020. 2 

- [11] Vladimir Iashin and Esa Rahtu. Multi-modal dense video captioning. In _Proceedings of the IEEE/CVF conference on computer vision and pattern recognition workshops_ , pages 958–959, 2020. 2 

- [12] Ranjay Krishna, Kenji Hata, Frederic Ren, Li Fei-Fei, and Juan Carlos Niebles. Dense-captioning events in videos. In _Proceedings of the IEEE international conference on computer vision_ , pages 706–715, 2017. 1, 2 

- [13] Ranjay Krishna, Kenji Hata, Frederic Ren, Li Fei-Fei, and Juan Carlos Niebles. Dense-captioning events in videos. In 

_International Conference on Computer Vision (ICCV)_ , pages 706–715, 2017. 1, 6, 11 

- [14] Yehao Li, Ting Yao, Yingwei Pan, Hongyang Chao, and Tao Mei. Jointly localizing and describing events for dense video captioning. In _Proceedings of the IEEE conference on computer vision and pattern recognition_ , pages 7492–7500, 2018. 2 

- [15] Huaishao Luo, Lei Ji, Botian Shi, Haoyang Huang, Nan Duan, Tianrui Li, Jason Li, Taroon Bharti, and Ming Zhou. Univl: A unified video and language pre-training model for multimodal understanding and generation. _arXiv preprint arXiv:2002.06353_ , 2020. 3, 4, 6 

- [16] Antoine Miech, Dimitri Zhukov, Jean-Baptiste Alayrac, Makarand Tapaswi, Ivan Laptev, and Josef Sivic. Howto100m: Learning a text-video embedding by watching hundred million narrated video clips. In _Proceedings of the IEEE/CVF International Conference on Computer Vision_ , pages 2630–2640, 2019. 1, 6, 11 

- [17] Jonghwan Mun, Linjie Yang, Zhou Ren, Ning Xu, and Bohyung Han. Streamlined dense video captioning. In _Proceedings of the IEEE/CVF conference on computer vision and pattern recognition_ , pages 6588–6597, 2019. 2 

- [18] Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, et al. Learning transferable visual models from natural language supervision. In _International conference on machine learning_ , pages 8748–8763. PMLR, 2021. 4, 6 

- [19] Colin Raffel, Noam Shazeer, Adam Roberts, Katherine Lee, Sharan Narang, Michael Matena, Yanqi Zhou, Wei Li, and Peter J Liu. Exploring the limits of transfer learning with a unified text-to-text transformer. _Journal of machine learning research_ , 21(140):1–67, 2020. 6 

- [20] Tanzila Rahman, Bicheng Xu, and Leonid Sigal. Watch, listen and tell: Multi-modal weakly supervised dense event captioning. In _Proceedings of the IEEE/CVF international conference on computer vision_ , pages 8908–8917, 2019. 2, 7 

- [21] InternLM Team. Internlm: A multilingual language model with progressively enhanced capabilities. https:// github.com/InternLM/InternLM, 2023. 3 

- [22] Hugo Touvron, Louis Martin, Kevin Stone, Peter Albert, Amjad Almahairi, Yasmine Babaei, Nikolay Bashlykov, Soumya Batra, Prajjwal Bhargava, Shruti Bhosale, et al. Llama 2: Open foundation and fine-tuned chat models. _arXiv preprint arXiv:2307.09288_ , 2023. 3, 11 

- [23] Ramakrishna Vedantam, C Lawrence Zitnick, and Devi Parikh. Cider: Consensus-based image description evaluation. In _Proceedings of the IEEE conference on computer vision and pattern recognition_ , pages 4566–4575, 2015. 6 

- [24] Jinpeng Wang, Yixiao Ge, Rui Yan, Yuying Ge, Kevin Qinghong Lin, Satoshi Tsutsui, Xudong Lin, Guanyu Cai, Jianping Wu, Ying Shan, et al. All in one: Exploring unified video-language pre-training. In _Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition_ , pages 6598–6608, 2023. 3 

- [25] Teng Wang, Ruimao Zhang, Zhichao Lu, Feng Zheng, Ran Cheng, and Ping Luo. End-to-end dense video captioning 

with parallel decoding. In _Proceedings of the IEEE/CVF International Conference on Computer Vision_ , pages 6847– 6857, 2021. 2, 5, 6, 7 

- [26] Ning Xu, An-An Liu, Yongkang Wong, Yongdong Zhang, Weizhi Nie, Yuting Su, and Mohan Kankanhalli. Dualstream recurrent neural network for video captioning. _IEEE Transactions on Circuits and Systems for Video Technology_ , 29(8):2482–2493, 2018. 1 

- [27] Antoine Yang, Arsha Nagrani, Paul Hongsuck Seo, Antoine Miech, Jordi Pont-Tuset, Ivan Laptev, Josef Sivic, and Cordelia Schmid. Vid2seq: Large-scale pretraining of a visual language model for dense video captioning. In _Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition_ , pages 10714–10726, 2023. 1, 2, 3, 7 

- [28] Dali Yang and Chun Yuan. Hierarchical context encoding for events captioning in videos. In _2018 25th IEEE International Conference on Image Processing (ICIP)_ , pages 1288–1292. IEEE, 2018. 2 

- [29] Qi Zhang, Yuqing Song, and Qin Jin. Unifying event detection and captioning as sequence generation via pre-training. In _European Conference on Computer Vision_ , pages 363– 379. Springer, 2022. 3, 5, 6, 7 

- [30] Ziqi Zhang, Yaya Shi, Chunfeng Yuan, Bing Li, Peijin Wang, Weiming Hu, and Zheng-Jun Zha. Object relational graph with teacher-recommended learning for video captioning. In _Proceedings of the IEEE/CVF conference on computer vision and pattern recognition_ , pages 13278–13288, 2020. 1 

- [31] Luowei Zhou, Chenliang Xu, and Jason J Corso. Towards automatic learning of procedures from web instructional videos. In _AAAI Conference on Artificial Intelligence_ , pages 7590–7598, 2018. 1, 6, 11 

- [32] Luowei Zhou, Yingbo Zhou, Jason J Corso, Richard Socher, and Caiming Xiong. End-to-end dense video captioning with masked transformer. In _Proceedings of the IEEE conference on computer vision and pattern recognition_ , pages 8739– 8748, 2018. 2 

- [33] Wanrong Zhu, Bo Pang, Ashish V Thapliyal, William Yang Wang, and Radu Soricut. End-to-end dense video captioning as sequence generation. _arXiv preprint arXiv:2204.08121_ , 2022. 2, 5, 7 

## **6. Selection Process for Pretraining Subset: Focused on Cooking Videos** 

For our pretraining process, we carefully selected a subset of HowTo100M [16] videos, leveraging available metadata to curate a tailored dataset. Focusing on cooking-related content, we filtered videos based on the ”Recipes” category ~~2~~ tag, ensuring relevance to recipe-based tasks. Within this subset, we further refined our selection by prioritizing videos within the top 30 rankings. This selection strategy aimed to capture cooking-related content, given its sequential nature and specificity in depicting step-by-step instructions. Cooking videos often present a structured flow, mirroring the requirements for dense video captioning tasks. Their explicit actions and clear events make them ideal for training models to capture and describe such events accurately. Additionally, these instructional videos offer rich visual and textual information, providing diverse examples pivotal for robust model training and the generation of precise captions for subsequent tasks. 

## **7. Prompt Engineering** 

To streamline the process of deriving step-by-step event sequences from video subtitles, we utilize LLMs like LLAMA2 [22] and ChatGPT to generate prompts. Initially, our requests for prompts, such as ’ _Task: write a prompt to send to LLM to extract the ordered steps of a recipe video subtitle. Example steps should resemble ”coat the chicken pieces in the flour”_ ,’ resulted in step-by-step event captions that did not align with the concise, action-oriented format found in YouCook2 [31] captions. 

To enhance this extraction process further, we manually refine the prompts to mirror the succinct nature of YouCook2 captions. Specifically, we emphasize extra regularization highlighting single-sentence clarity, exclusion of events unrelated to video actions, and emphasis on the sequential process. Additionally, after the instruction prompt, we append the original subtitle text followed by ” _\_ nSteps:” to extract the step-by-step events without extra irrelevant words. The complete input for LLM should resemble ” _Prompt_ [ _original subtitle_ ] _\nSteps:_ ”. This fine-tuning aims to closely align the extracted events with the concise and action-focused nature of YouCook2 captions, potentially shedding light on why pretraining shows better performance in YouCook2 [31] compared to ActivityNet [13]. Some potential prompt candidates are detailed in Table 9. 

## **8. Additional ablation studies** 

**Effects of Pretraining Data Variation** Our comparative experiments in Table 10 exploring varying pretraining data proportions on YouCook2 and ActivityNet reveal a notable trend. Increasing the pretraining data volume significantly boosts model performance on both event localization 

and caption generation. Strikingly, we witness a substantial performance jump on YouCook2 compared to ActivityNet. This suggests the instructional cooking focus within YouCook2 derives greater benefit from more pretraining data. The cooking domain aligns closely with the pretraining data, enabling more efficient learning. In contrast, ActivityNet’s broader activity spectrum may need more diverse pretraining data to show similar improvements. This highlights the pivotal role of dataset characteristics and domain relevance in pretraining efficacy for dense video captioning. 

**Pseudo Boundary Generation** In our comparative study across different pseudo boundary generation schemes on the YouCook2 dataset, we evaluate the performance of models using uniform boundaries, Drop-DTW boundaries, and our proposed generation scheme in Table 11. The uniform boundary approach, dividing the entire video into segments based on the number of captions, results in moderate performance. Surprisingly, the Drop-DTW method exhibits higher caption metric performance than uniform boundaries but notably underperforms in localization metrics. This discrepancy suggests that the noise present in the similarity matrix might hinder the accurate generation of boundaries using Drop-DTW. Conversely, our proposed scheme showcases superior performance in both localization and caption metrics, indicating the efficacy of our method in mitigating the impact of noise and improving overall performance in dense video captioning tasks. 

**Few-shot Model Performance** In our investigation of few-shot dense video captioning, we analyze model performance with varying amounts of fine-tuning data on the YouCook2 dataset, as detailed in Table 12. Notably, as expected, models exhibit improved performance as more finetuning data is utilized, showcasing a positive correlation between data volume and model performance. Moreover, our comparison between models with and without pretraining reveals a substantial performance boost in pretrainingenabled models across all fine-tuning data proportions. Intriguingly, models with pretraining showcase significantly superior performance even with just half the fine-tuning data, surpassing the performance of models without pretraining. These results underscore the remarkable effectiveness of pretraining and our pseudo annotation generation method in enhancing model capabilities for few-shot dense video captioning tasks. 

**Effects of top-** _k_[ˆ] **in Pseudo Boundary Generation** In our investigation of top- _k_[ˆ] values’ impact on pseudo boundary generation and subsequent model performance on YouCook2 and ActivityNet, detailed in Table 13, contrasting trends emerge between the two datasets. Surprisingly, 

|**Index**|**Prompt Description**|
|---|---|
|||
|1|Task: Extract ordered steps from video subtitles, focusing on sequential process and action-oriented instructions.|
|||
|2|Task: Please extract concise and action-oriented steps from video subtitles, ensuring a clear sequential arrange-<br>ment of events.|
|||
|3|Task: Generate steps from subtitles in a single-sentence, emphasizing clear actions, excluding non-action-related<br>events.|
|||
|4|Task: Extract concise and action-oriented steps or instructions from the video subtitles, focusing solely on the<br>sequential process. Each step should be presented as a single sentence with clear actions.|
|||
|5|Task: Extract concise and action-oriented steps or instructions from the video subtitles, focusing solely on the<br>sequential process. Each step should be presented as a single sentence with clear actions. Exclude any steps that<br>are not directly related to the actions in the video.|



Table 9. Examples of enhanced prompts for extracting step-by-step events from video subtitles. 

|Dataset|Data|M<br>C<br>S<br>Rec.<br>Pre.<br>F1|
|---|---|---|
|YouCook2|0%<br>25%<br>50%<br>75%<br>100%|7.87<br>46.02<br>6.87<br>29.66<br>42.04<br>34.78<br>8.64<br>52.47<br>7.89<br>**31.87**<br>43.25<br>36.70<br>9.14<br>55.42<br>7.37<br>29.02<br>44.87<br>35.25<br>9.39<br>**60.95**<br>**8.17**<br>31.27<br>44.61<br>**36.77**<br>**9.41**<br>59.35<br>7.97<br>30.80<br>**45.13**<br>36.61|
|ActivityNet|0%<br>25%<br>50%<br>75%<br>100%|8.31<br>30.11<br>5.63<br>50.82<br>55.73<br>53.16<br>8.85<br>30.11<br>5.63<br>51.18<br>56.29<br>55.60<br>8.72<br>30.00<br>5.70<br>53.29<br>57.93<br>55.51<br>8.84<br>**32.75**<br>**5.90**<br>**53.86**<br>56.67<br>55.23<br>**8.93**<br>31.89<br>5.85<br>53.14<br>**58.31**<br>**55.61**|



Table 10. Impact of varying pretraining data on dense video captioning performance on YouCook2 and ActivityNet. 

|Boundary|M<br>C<br>S<br>Rec.<br>Pre.<br>F1|
|---|---|
|Uniform<br>Drop-DTW<br>Ours w/o STC<br>Ours w/ STC|2.95<br>15.16<br>4.29<br>19.4<br>19.88<br>19.64<br>3.21<br>17.37<br>3.86<br>14.09<br>17.26<br>15.51<br>5.43<br>25.57<br>4.47<br>18.05<br>32.83<br>23.29<br>**5.88**<br>**28.04**<br>**4.47**<br>**19.72**<br>**35.43**<br>**25.34**|



Table 11. Comparative analysis of pseudo boundary generation schemes for dense video captioning on YouCook2. 

|Pretrain<br>FT Data|M<br>C<br>S<br>Rec.<br>Pre.<br>F1|
|---|---|
|✗<br>25%<br>✓<br>25%<br>✗<br>50%<br>✓<br>50%<br>✗<br>75%<br>✓<br>75%<br>✗<br>100%<br>✓<br>100%|5.06<br>22.39<br>4.45<br>23.94<br>36.41<br>28.89<br>7.81<br>46.69<br>7.17<br>28.93<br>41.08<br>33.95<br>6.45<br>34.30<br>5.44<br>25.18<br>39.75<br>30.83<br>8.60<br>55.73<br>7.84<br>30.14<br>42.73<br>35.35<br>7.26<br>40.49<br>6.31<br>27.81<br>40.60<br>33.01<br>9.11<br>59.09<br>7.86<br>29.97<br>44.54<br>35.83<br>7.87<br>46.02<br>6.87<br>29.66<br>42.04<br>34.78<br>**9.41**<br>**59.35**<br>**7.97**<br>**30.80**<br>**45.13**<br>**36.61**|



Table 12. Impact of fine-tuning data and pretraining on few-shot dense video captioning performance in YouCook2. “FT data” represents the percentage of data used for fine-tuning. 

|Dataset|ˆ_k_|M<br>C<br>S<br>Rec.<br>Pre.<br>F1|
|---|---|---|
|YouCook2<br>15<br>**5.91**<br>**30.89**<br>**5.00**<br>**20.56**<br>**35.92**<br>**26.15**<br>20<br>5.79<br>27.04<br>4.72<br>18.99<br>35.89<br>24.84<br>25<br>5.35<br>25.86<br>4.71<br>17.8<br>33.24<br>23.18<br>30<br>4.62<br>21.88<br>4.48<br>17.26<br>30.05<br>21.93|||
|ActivityNet|15<br>20<br>25<br>30|7.21<br>**18.40**<br>**4.76**<br>**43.96**<br>61.50<br>51.27<br>7.30<br>17.52<br>4.69<br>43.46<br>63.68<br>51.66<br>7.32<br>17.25<br>4.71<br>43.22<br>64.74<br>**51.84**<br>**7.45**<br>16.50<br>4.71<br>42.29<br>**65.68**<br>51.45|



Table 13. Impact of top- _k_[ˆ] values on model performance in YouCook2 and ActivityNet. 

YouCook2 displays improved performance as the top- _k_[ˆ] value decreases, while ActivityNet exhibits a contrasting trend, showing deteriorating results with smaller top- _k_[ˆ] values for partial metrics. Our pseudo boundary generation scheme operates such that a larger top- _k_[ˆ] value leads to an inclusion of more frames, potentially unrelated to the current event caption. We hypothesize that YouCook2, known for its diverse events, likely has a higher frequency of events, thereby resulting in shorter event durations compared to ActivityNet. This difference in event duration might explain the disparate impact of top- _k_[ˆ] values on model performance between the two datasets. 

**Effects of Refinement Hyperparameters.** In Table 14, our exploration of the impact of the number of selected proposals for merging new boundaries (3, 4, and 5 proposals) reveals an intriguing trend across both YouCook2 and ActivityNet. As the number of proposals increases, we observe a trade-off in metrics. Fewer proposals tend to select boundaries closely aligned with the current caption, suggesting a focus on relevance, while a larger count encompasses a more diverse selection, albeit with some proposals less related to the ongoing caption. This dual impact leads to a 

|Dataset|_K_|M<br>C<br>S<br>Rec.<br>Pre.<br>F1|
|---|---|---|
|YouCook2|3<br>4<br>5|5.90<br>29.62<br>4.96<br>20.44<br>**36.38**<br>**26.17**<br>5.89<br>29.28<br>4.85<br>19.85<br>35.67<br>25.51<br>**5.91**<br>**30.89**<br>**5.00**<br>**20.56**<br>35.92<br>26.15|
|ActivityNet|3<br>4<br>5|7.33<br>15.86<br>4.67<br>42.21<br>**65.79**<br>51.43<br>**7.45**<br>**16.50**<br>**4.71**<br>42.29<br>65.68<br>51.45<br>7.35<br>16.27<br>4.68<br>**42.77**<br>65.54<br>**51.76**|



Table 14. Impact of selected proposals on boundary merging in YouCook2 and ActivityNet. 

|Dataset|Stages|M<br>C<br>S<br>Rec.<br>Pre.<br>F1|
|---|---|---|
|YouCook2|1<br>2<br>3|**5.95**<br>**32.08**<br>4.78<br>19.39<br>35.79<br>25.15<br>5.90<br>29.62<br>4.96<br>**20.44**<br>**36.38**<br>**26.17**<br>5.74<br>28.95<br>**4.98**<br>20.29<br>34.72<br>25.61|
|ActivityNet|1<br>2<br>3|7.34<br>15.93<br>4.65<br>**43.10**<br>65.5<br>**51.99**<br>7.33<br>15.86<br>**4.67**<br>42.21<br>**65.79**<br>51.43<br>**7.37**<br>**16.04**<br>4.65<br>42.40<br>65.39<br>51.44|



Table 15. Impact of refinement stages on model performance in YouCook2 and ActivityNet. 

compromise in model performance, highlighting the detrimental effects when the number of proposals deviates too far from an optimal range. 

In Table 15, we examine the influence of refinement stages (1, 2, and 3 stages) and observe distinct trends across both YouCook2 and ActivityNet. On YouCook2, we observe a trend of declining caption performance with increasing refinement stages, accompanied by a marginal decrease in localization metrics with larger stage counts. Similarly, ActivityNet exhibits varying metric shifts with increased stages, suggesting that an excessive number of refinement stages may not consistently enhance performance and could lead to fluctuations in metric outcomes. This nuanced relationship emphasizes that excessive refinement may not always yield improved results and could elongate the training process. 


![](raw/papers/2_LLM_NLP/LLM_paper_list/caption/cvpr_2024_DIBS_Enhancing_Dense_Video_Captioning_with_Unlabeled_Videos_via_Pseudo_Boundary_Enrichment_and_Online_Refinement_images/cvpr_2024_DIBS_Enhancing_Dense_Video_Captioning_with_Unlabeled_Videos_via_Pseudo_Boundary_Enrichment_and_Online_Refinement.pdf-0013-06.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/caption/cvpr_2024_DIBS_Enhancing_Dense_Video_Captioning_with_Unlabeled_Videos_via_Pseudo_Boundary_Enrichment_and_Online_Refinement_images/cvpr_2024_DIBS_Enhancing_Dense_Video_Captioning_with_Unlabeled_Videos_via_Pseudo_Boundary_Enrichment_and_Online_Refinement.pdf-0013-07.png)


**----- Start of picture text -----**<br>
(a) Without pretraining<br>(b) With pretraining<br>**----- End of picture text -----**<br>


Figure 4. Word cloud of predicted captions from our DIBS w/ and w/o pretraining. A larger word indicates a higher frequency of its occurrence in the predicted captions. 

**Qualitative Results** Finally, to further prove the effectiveness of our method, we present some qualitative results of DIBS on YouCook2 and ActivityNet datasets in Figure 5 and Figure 6, respectively. We can see from these figures that our DIBS could predict accurate event boundaries along with rich captions. 

## **8.1. Visualization Results** 

**Does pretraining lead to diverse captioning?** In Figure 4, we compare word clouds of predicted captions from DIBS model with and without our proposed pretraining. It can be seen that predicted captions after pretraining (Figure 4b) show a richer vocabulary and broader concepts than those without pretraining (Figure 4a), highlighting our method’s success in diversifying caption generation. The variation in word size and diversity after pretraining suggests significant improvements in the model’s nuanced understanding, supporting our hypothesis on the benefits of pretraining for rich and diverse dense video captioning. 


![](raw/papers/2_LLM_NLP/LLM_paper_list/caption/cvpr_2024_DIBS_Enhancing_Dense_Video_Captioning_with_Unlabeled_Videos_via_Pseudo_Boundary_Enrichment_and_Online_Refinement_images/cvpr_2024_DIBS_Enhancing_Dense_Video_Captioning_with_Unlabeled_Videos_via_Pseudo_Boundary_Enrichment_and_Online_Refinement.pdf-0014-00.png)


**----- Start of picture text -----**<br>
Input<br>Video<br>When the Add<br>Grind<br>Ground Add tomato chili, garlic chili, marsala powder and salt to the chicken and mix. shallots into a  Cook the chicken in a pan with oil. chicken issoft, add onion paste  water and cook  Add coriander leaves and stir.<br>Truth paste. and mix. more.<br>DIBS Add some spices and  Add the oil and  Cook until the  Add  Garnish with green<br>coriander powder and salt to  chicken to the pan and  chicken is  some  onions and<br>(Ours) the pan and mix. cook. golden brown. water. coriander leaves.<br>Input<br>Video<br>Cut the meat and<br>Add soy sauce and<br>Ground Add minced garlic minced onion brown sugar  sesame oil and Add ribs and mix. Grill the ribs. warp them with<br>and black pepper and mix. rice in a lettuce<br>Truth mix. leaf.<br>Roast the<br>DIBS Grate onion and  Add the sauce to  Dip the ribs in  Place the meat  ribs in<br>(Ours) garlic. the pan. the marinade. on the pan. the pan.<br>**----- End of picture text -----**<br>


Figure 5. Visualization of qualitative results of our DIBS model. Video examples are selected from the validation set of YouCook2. 


![](raw/papers/2_LLM_NLP/LLM_paper_list/caption/cvpr_2024_DIBS_Enhancing_Dense_Video_Captioning_with_Unlabeled_Videos_via_Pseudo_Boundary_Enrichment_and_Online_Refinement_images/cvpr_2024_DIBS_Enhancing_Dense_Video_Captioning_with_Unlabeled_Videos_via_Pseudo_Boundary_Enrichment_and_Online_Refinement.pdf-0014-02.png)


**----- Start of picture text -----**<br>
Input<br>Video<br>Ground A large group of people are seen playing a lacrosse game with one another as the camera  The girls continue to play with one another and end by<br>Truth pans around them running up and down the field. walking way off the field.<br>The girls continue playing with one another and ends with the camera<br>panning around the field.<br>DIBS A group of people are plying lacrosse on a field.<br>(Ours) A group of people are seen running around a field with a large group of<br>people playing lacrosse.<br>Input<br>Video<br>Ground Two groups of people are seen pulling on a rope and playing a game of tug of<br>Truth war with one another.<br>The men continue playing the game and end by walking away from one another.<br>A group of people are seen standing around a large field with one another<br>DIBS and a man holding a rope.<br>The man then begins to pull the rope while the man continues to speak.<br>(Ours)<br>The men continue to pull the rope and one another man walks away in the end.<br>**----- End of picture text -----**<br>


Figure 6. Visualization of qualitative results of our DIBS model. Video examples are selected from the validation set of ActivityNet. 

