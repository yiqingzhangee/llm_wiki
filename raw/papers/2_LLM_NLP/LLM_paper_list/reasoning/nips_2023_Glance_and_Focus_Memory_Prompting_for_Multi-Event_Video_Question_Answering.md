# **Glance and Focus: Memory Prompting for Multi-Event Video Question Answering** 

**Ziyi Bai**[1] _[,]_[2] **, Ruiping Wang**[1] _[,]_[2] **, Xilin Chen**[1] _[,]_[2] 

1Key Laboratory of Intelligent Information Processing of Chinese Academy of Sciences (CAS), Institute of Computing Technology, CAS, Beijing, 100190, China 

2University of Chinese Academy of Sciences, Beijing, 100049, China _ziyi.bai@vipl.ict.ac.cn_ , _{wangruiping, xlchen}@ict.ac.cn_ 

## **Abstract** 

Video Question Answering (VideoQA) has emerged as a vital tool to evaluate agents’ ability to understand human daily behaviors. Despite the recent success of large vision language models in many multi-modal tasks, complex situation reasoning over videos involving multiple human-object interaction events still remains challenging. In contrast, humans can easily tackle it by using a series of episode memories as _anchors_ to quickly locate question-related key moments for reasoning. To mimic this effective reasoning strategy, we propose the GlanceFocus model. One simple way is to apply an action detection model to predict a set of actions as key memories. However, these actions within a closed set vocabulary are hard to generalize to various video domains. Instead of that, we train an Encoder-Decoder to generate a set of dynamic event memories at the glancing stage. Apart from using supervised bipartite matching to obtain the event memories, we further design an unsupervised memory generation method to get rid of dependence on event annotations. Next, at the focusing stage, these event memories act as a bridge to establish the correlation between the questions with high-level event concepts and low-level lengthy video content. Given the question, the model first focuses on the generated key event memory, then focuses on the most relevant moment for reasoning through our designed multi-level crossattention mechanism. We conduct extensive experiments on four Multi-Event VideoQA benchmarks including STAR, EgoTaskQA, AGQA, and NExT-QA. Our proposed model achieves state-of-the-art results, surpassing current large models in various challenging reasoning tasks. The code and models are available at `https://github.com/ByZ0e/Glance-Focus` . 

## **1 Introduction** 

For intelligent agent that can assist humans in completing daily-life tasks, complex event reasoning ability is particularly important. Multiple Event Video Question Answering (Multi-Event VideoQA) task [14, 42, 16, 43] has recently been proposed. These benchmarks require agents to reason about complex videos depicting multiple human-object interactions (i.e., events or situations). Furthermore, a wide scope of event-centric questions are involved[14], ranging from event semantic descriptions to event sequencing and dependencies to comprehensively evaluate an agent’s ability to understand human behavior. 

Recently, the emerged large vision-language models [1, 3, 8, 25, 47, 50] have demonstrated impressive generalization capabilities in VideoQA tasks. By leveraging pre-training on large-scale cross-modal pairs, these models can establish correlations between questions and videos to find answers effectively. Such approaches proves effective for short videos containing single or few events with one-step 

37th Conference on Neural Information Processing Systems (NeurIPS 2023). 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2023_Glance_and_Focus_Memory_Prompting_for_Multi-Event_Video_Question_Answering_images/nips_2023_Glance_and_Focus_Memory_Prompting_for_Multi-Event_Video_Question_Answering.pdf-0002-00.png)


**----- Start of picture text -----**<br>
𝑣! 𝑣" 𝑣#$& 𝑣#$% 𝑣#$" 𝑣#$! 𝑣# 𝑣! 𝑣" 𝑣#$& 𝑣#$% 𝑣#$" 𝑣#$! 𝑣#<br>... ...<br>Episode  before before before<br>Q : What did the person do to the blanket before  Memory<br>taking the clothes?  Event 1 Event N-2 Event N-1 Event N<br>Glance: Generating Episode Memory<br>𝑥# 𝑥#<br>𝒗𝑻"𝟒 𝒗𝟏 𝒗𝑻"𝟒 𝒗𝟏<br>𝒗𝑻"𝟑 video 𝒗𝟐 𝒗𝑻"𝟑 video 𝒗𝟐<br>person beforetaking ？ 𝒗𝑻"𝟏𝒗𝑻"𝟐 𝒗𝑻 before 𝒗𝑻"𝟏 𝒗𝑻"𝟐 𝒗𝑻<br>question person taking<br>question<br>the<br>clothes the<br>A : Took. clothes A : Put Down.<br>𝑥" 𝑥"<br>Q : What ... before<br>Q : What ... before  … A : Took. taking the clothes?  A : Put Down.<br>taking the clothes?<br>Focus: Reasoning with Memory Prompt<br>(a) VideoQA through cross-modality<br>correspondence modeling (b) VideoQA through Glance-Focus<br>taking the<br>clothes<br>putting down<br>the blanket<br>before<br>**----- End of picture text -----**<br>


Figure 1: Comparison of different VideoQA solutions. The cross-modal common feature space is visualized. (a) Cross-modal correspondence modeling. Weak correspondence between the question with high-level event concepts and low-level lengthy video leads to a wrong answer. (b) Human-like Glance-Focus Strategy. Glancing stage: a set of episode memories are generated. Focusing stage: memory prompts are used as anchors to quickly locate the keyframes for reasoning. 

questions like “ _What is someone doing?_ ”. However, Multi-Event VideoQA poses distinct challenge. As shown in Figure 1(a), due to the longer video duration and more events involved, it is challenging for the model to effectively find the visual clues from the lengthy video content. Additionally, the event-centric questions often include high-level event concepts while the semantics of video frames are relatively low-level, making it even more difficult to directly locate key moments related to the question. 

This is very similar to the difficulties we humans encounter when addressing reading comprehension tasks, where we must quickly extract question-related context from long paragraphs. To ensure fast and effective reasoning, it is common for humans to employ a strategy of first glancing through the story, summarizing the storyline in our mind, and then using this episode memory to quickly focus on the relevant context when answering questions. 

We mimic such a reasoning strategy to tackle the challenging Multi-Event VideoQA task with our proposed Glance-Focus model. It comprises two stages: glancing and focusing. The purpose of the glancing stage is to generate a set of episode memories that can efficiently summarize the multiple events occurring in the video. A naive way is to directly train an action detection model to obtain a set of actions. However, this would result in unnecessary computation, and most importantly, it may not be practical for encapsulating all possible actions in the real world using closed vocabularies. 

Instead, we adopt a more flexible approach that automatically obtains a set of dynamic event memories using a Transformer Encoder-Decoder architecture [39]. For datasets with closed event vocabularies, we employ bipartite matching between the extracted memories and ground truth labels to explicitly obtain event memories. Apart from supervised event matching approach, we additionally design an unsupervised event generation way to get rid of the event-level annotations of the video. We impose individual certainty and global diversity constraints during event generation to ensure they can effectively summarize the video content. 

Finally, since these generated memories are still independent events, we reorganize them into complete event memory prompts by arranging them in sequence order with temporal information remaining. As shown in Figure 1(b), since these memory prompts have high-level event semantics, they play an anchor role in the multi-modal common feature space to help quickly locate the related video content according to the question. Such as the example in Figure 1(b), we design a multi-level cross 

2 

attention mechanism to first focus on the event “putting down the blanket” before “taking the clothes” according to the question, then focus on the related video content for answer prediction. 

We conduct experiments on various recently released Multi-Event VideoQA datasets, STAR [42], AGQA [14], EgoTaskQA[16] and NExT-QA[43]. The experimental results demonstrate that our proposed Glance-Focus model achieves state-of-the-art performance on all four datasets. We further verify the effectiveness of our approach via model analysis, showing the outstanding performance of key components in our method. Moreover, both quantitative and qualitative results affirm that our proposed memory prompts play a significant _anchor_ role in multi-event reasoning by bridging the gap between questions and videos. 

## **2 Related Works** 

## **2.1 Video Question Answering** 

Early attempts[11, 49, 6, 51, 41] have made progress in improving complex spatial-temporal contextual information modeling for VideoQA tasks. They usually extract the appearance and motion features of video units (such as clips or frames) separately and then let them interact with the question to find an answer. To capture question-related spatial and temporal clues, co-memory and co-attention mechanisms are proposed by MACN[11] and HME[6] respectively. B2A[35] uses questions as a bridge to connect the appearance and motion features to find the answers. These holistic models are effective at aggregating spatial and temporal context, making them well-suited for general one-step video reasoning tasks. However, they tend to encounter challenges when confronted with more intricate compositional questions[14]. 

Therefore many studies[23, 15, 30, 5, 44, 45] have delved into the exploration of hierarchical modeling strategies to address the challenges of multi-grained video content representation. HCRN[23] proposes using reusable units stacked hierarchically to gradually extract frame-level to video-level features conditioned by motion and question. Graph-based methods such as L-GCN[15], HAIR[30], and VGT[45] utilize extra object detection models, like Faster R-CNN[37], to explicitly incorporate object-level features into the model. They further organize the objects into a unified graph for fine-grained reasoning. Although these hierarchical structures contain multi-grained clues, few of them explicitly model event-level structure of videos. This limitation poses a challenge in effectively handling high-level tasks, including event prediction, explanation, and counterfactual reasoning. 

## **2.2 Video-Language Pre-trained Models** 

Recently, Transformers-based large video-language models (VLMs) [52, 25, 2, 8, 47, 50, 12, 1, 3, 40] have accomplished comparable or superior results on downstream tasks like VideoQA. These models harness the power of large-scale image or video-text pairs collected from the Internet to pre-train cross-modal correspondence models. They commonly employ proxy tasks like masked language modeling and masked frame modeling [25, 50, 40] or contrastive learning [47, 2] during pre-training. Some models like ActBERT[52] and BridgeFormer[12] focus on multi-grained semantics grounding by explicitly learning the alignment of objects and actions across modalities. However, applying these VLMs directly to Multi-Event VideoQA tasks is non-trivial. These models often necessitate sparse sampling of lengthy videos, leading to a notable loss of video semantics, and thus suffer from weak correspondence between questions involving high-level event concepts and intricate video content[9, 16]. To tackle this challenge, some retrieval-based cross-modal methods [48, 19, 36] have been proposed recently to efficiently extract the keyframes of the video for reasoning. Nonetheless, the use of fixed keyframes may pose difficulties when dealing with multi-grained questions. Instead, we propose to employ the dynamic and adaptable event memories for representing lengthy videos. 

## **2.3 Multi-Event Modeling** 

Multi-event modeling is an essential ability for household robots to assist humans in completing complex tasks. However, explicit multi-event modeling still needs to be explored for VideoQA task. TSEA[9] proposes a heuristic method to extract event-level features of videos. Several studies targeting at multi-activity recognition and detection task[15, 34, 33] employ graph structures to model complex relationships among multiple activities. P-GCN[15] extracts activity proposals from videos and utilizes graph models to integrate context. Ego-Topo[33] organizes egocentric videos into several 

3 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2023_Glance_and_Focus_Memory_Prompting_for_Multi-Event_Video_Question_Answering_images/nips_2023_Glance_and_Focus_Memory_Prompting_for_Multi-Event_Video_Question_Answering.pdf-0004-00.png)


**----- Start of picture text -----**<br>
data flow<br>... attention A. put down<br>m" m$ m! m#<br>c) predict answer<br>... Transformer Decoder Transformer Decoder ...<br>memory  a) focus on memory answer<br>query query<br>... ... ... ...<br>b) focus on frame<br>Transformer Encoder Transformer Encoder<br>... ... ... ...<br>... N-2 N-1 what ...  the clothes ?<br>1.Glancing Stage 2.Focusing Stage<br>memory prompting<br>**----- End of picture text -----**<br>


Figure 2: The complex multi-event reasoning task is structured into two stages: Glancing and Focusing. At the Glancing stage, a set of unordered event memories are obtained by aggregating relevant input video frames. At the Focusing stage, these event memories are reorganized according to their sequential order. The resulting memory prompts act as bridges, allowing model progressively focus on the related video frames for answer prediction using multi-level cross-attention mechanism. 

zones and learns the links between these zones for scene affordances and future action prediction. Bridge-prompt[27] leverages the continuity of the natural language to concatenate multiple discrete events by constructing a series of text prompts. Similar to the multi-activity detection task, our goal is to obtain a set of high-level key event from the video. However, directly training an action detection model requires additional event-level annotations, which may not always be available. Furthermore, it is impractical to use a closed set of actions to encompass all possible events in the real-world. By contrast, our proposed approach learns dynamic event memories from videos through either unsupervised generation or supervised set-matching methods, depending on the availability of event labels during the training process. 

## **3 Approach: The Glance-Focus Model** 

We first give an overview of our model in Section 3.1. Next, we describe in detail the two stages of our model, the Glancing stage and the Focusing stage in Section 3.2 and Section 3.3 respectively. Finally, in Section 3.4 we illustrate the overall loss functions. 

## **3.1 Overview** 

Given a video clip _V_ and a question _q_ , a VideoQA model predicts the answer _y_ , formulated as follows: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2023_Glance_and_Focus_Memory_Prompting_for_Multi-Event_Video_Question_Answering_images/nips_2023_Glance_and_Focus_Memory_Prompting_for_Multi-Event_Video_Question_Answering.pdf-0004-07.png)


where _y_ is the predicted answer chosen from the candidate answers denoted as _A_ , i.e., answer vocabulary for open-ended questions or choices for multi-choice questions. _θ_ is the trainable parameters of a VideoQA model _F_ . 

We propose a novel Glance-Focus VideoQA model with the Transformer Encoder-Decoder framework[39]. As shown in Figure 2, the model answers the question with two stages: 1) At the Glancing stage, a set of episode memories are queried from the input video; 2) At the focusing stage, the model reorganizes the memories into memory prompts and they will serve as anchors to localize the question-related video content by multi-level focusing. 

**Input representations.** For the input multi-event video of length _T_ , we use a pretrained visual backbone to extract its features **X** _∈_ R _[T][ ×][D]_ with dimension _D_ . The video features are fed into the model both in glancing and focusing stages. For questions, following [31], we tokenize them using Byte-Pair Encoding (BPE) tokenization, and embed them with pretrained RoBERTa language 

4 

model[31] to get textual tokens **w** = _{w_ 1 _, ..., wL}_ with question length _L_ . Then they are projected into the same dimension feature space as video features with a projector _G_ ( _._ ) to get question embeddings **L** = _G_ ( **w** ) _∈_ R _[L][×][D]_ . The question embeddings are fed into the model only at focusing stage. 

## **3.2 Glancing Stage** 

It is challenging for the VideoQA model to directly find answers from lengthy videos containing multiple complex events. To efficiently establish the corresponding relationships between the highlevel event concepts of the questions and the low-level video unit features, we propose employing a quick glance strategy that effectively extracts high-level episode memories from the video to guide the subsequent reasoning. 

To obtain the episode memory, a direct approach would be training an action detection model to extract a set of actions from the video. However, this may not be applicable in certain cases where video data lacks event annotations. Moreover, it is impractical to encompass all possible events in the real world with a close-set vocabulary. To address these issues, we propose a more flexible and practical solution which can automatically learn a set of dynamic memories **M** = _{_ **m1** _, ...,_ **mN** _} ∈_ R _[N][×][D]_ to summarize the key _N_ events occurring in the videos. Given a group of randomly initialized memory queries, we train a Transformer encoder-decoder[39] to update them by gradually aggregating the related encoded video features **X** through cross-attention interactions. Specifically, we design both unsupervised generation and supervised matching approaches depending on whether event-level annotations are available. 

**Unsupervised event memory generation.** In this case, we employ the idea of information maximization[21, 29] from the unsupervised learning to generate a set of event memories. Our goal is to generate a collection of event memories. To ensure that this set comprehensively captures all the crucial events within the video, we impose two vital constraints. 1) **Individual Certainty** . Each individual event is encouraged to be distinctive, thereby discouraging individual frames from forming separate and independent events. 2) **Global Diversity** . We aim to maintain diversity within the event set, preventing all video frames from being clustered into only one single event. 

Assuming that there are _C_ types of events that will occur in the video. For individual certainty, we suppose higher classification confidence score. Namely, the ideal category distribution is similar to one-hot encoding[21]. Specifically, a classifer header _fc_ of a linear layer with softmax is used to ˆ predict the event classification logits **p** _m_ = _fc_ ( **m** ). The individual certainty loss is denoted as 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2023_Glance_and_Focus_Memory_Prompting_for_Multi-Event_Video_Question_Answering_images/nips_2023_Glance_and_Focus_Memory_Prompting_for_Multi-Event_Video_Question_Answering.pdf-0005-06.png)


To ensure global diversity, we consider both the diversity of semantics and temporal spans within the event set. To promote semantic diversity, we assign category labels evenly across all memories. Namely, in contrast to the above one-hot classification logits of each event, we aim for an even distribution. The following _Lcls_ is minimized to constitute the semantic diversity loss 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2023_Glance_and_Focus_Memory_Prompting_for_Multi-Event_Video_Question_Answering_images/nips_2023_Glance_and_Focus_Memory_Prompting_for_Multi-Event_Video_Question_Answering.pdf-0005-08.png)


where ˆ **p** = E **m** _∈_ **M** [ _fc_ ( **m** )] is the mean output embedding of the all event memories of a video sample. Next, to ensure temporal span diversity, we pursue a reduction in the temporal overlap of all events. We use a temporal header _ft_ of 3-layer FFN with ReLU to predict the normalized temporal center ˆ ˆ coordinate _δ_[ˆ] and width _w_ of events, ˆ _τ_ = _ft_ ( **m** ) = [ _δ,_[ˆ] _w_ ] _∈_ [0 _,_ 1][2] . The temporal IoU loss formulated as, 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2023_Glance_and_Focus_Memory_Prompting_for_Multi-Event_Video_Question_Answering_images/nips_2023_Glance_and_Focus_Memory_Prompting_for_Multi-Event_Video_Question_Answering.pdf-0005-10.png)


where IoU(.) is the normalized temporal IoU[24] given the paired temporal prediction of events. The overall global diversity loss is the weighted sum of _Lcls_ and _Liou_ 

5 

_Ldiv_ ( _fc_ ; _ft_ ; **M** ) = _λclsLcls_ + _λiouLiou._ 

(5) 

**Supervised event memory matching.** For datasets with ground-truth event-level labels, we similarly use the set prediction via bipartite matching [4, 24] to extract event memories. Since the output of the Transformer is unordered, we do not have a one-to-one correspondence between the predicted event memories and the ground-truth events. Following [4, 24], we use the Hungarian algorithm to find the optimal bipartite matching results. Given the padding ground-truth event labels _E_ = _{ei}[N] i_ =1[,] _ei_ = _{ci, τi_ = [ _δi, wi_ ] _}_ , where _c_ and _τ_ are class labels and the temporal labels respectively, the matching cost _Cmatch_ is defined as: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2023_Glance_and_Focus_Memory_Prompting_for_Multi-Event_Video_Question_Answering_images/nips_2023_Glance_and_Focus_Memory_Prompting_for_Multi-Event_Video_Question_Answering.pdf-0006-03.png)


where _π ∈_ Π _N_ is the permutation of _N_ event memories. Then the goal is to find the optimal _π_ to minimize the matching cost. 

**Memory prompting.** At the glancing stage, the output is a set of unordered event memories **M** = _{_ **m1** _, ..._ **mN** _}_ . We reorganize them into **M**[�] in the sequential order according to their temporal center coordinate _{δi}_ . The temporal positional embedding **P** _t ∈{_ Φ( _i_ ) _|i ∈_ [1 _, N_ ] _}_ is added to the corresponding events to indicate their temporal positions in the video. And they will serve as the memory prompts during multi-level attention at the following focusing stage. 

## **3.3 Focusing Stage** 

It is difficult to directly associate the high-level event semantics in the questions with corresponding low-level videos. Thus the event memories extracted through video glancing serve as useful anchors to prompt the model quickly locate event segments related to the questions. Video features **X** , memory prompts **M**[�] + **P** _t_ , and question embeddings **L** are concatenated as input. The Transformer Encoder embeds these embeddings from different sources into a joint space [ **X** _enc,_ **M** _enc,_ **L** _enc_ ]. 

**Multi-level cross attentions.** Given a set of answer queries **a** _∈_ R _[N][×][D]_ , the Transformer Decoder exploits multi-level cross attention to replace the original multi-head cross-attention to guide the model gradually focus on the question-related visual clues in the video. 

a) _Focus on memory._ Given questions with high-level event semantics, we first use question embeddings as queries to interact with memory prompts to guide the model focus on the corresponding event memories **M** _foc_ : 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2023_Glance_and_Focus_Memory_Prompting_for_Multi-Event_Video_Question_Answering_images/nips_2023_Glance_and_Focus_Memory_Prompting_for_Multi-Event_Video_Question_Answering.pdf-0006-10.png)


b) _Focus on frame._ Then the selected event memories prompt the model to localize the corresponding video frames **X** _foc_ as visual clues for specific reasoning: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2023_Glance_and_Focus_Memory_Prompting_for_Multi-Event_Video_Question_Answering_images/nips_2023_Glance_and_Focus_Memory_Prompting_for_Multi-Event_Video_Question_Answering.pdf-0006-12.png)


c) _Predict the answer._ Finally, the answer queries find the answers **ˆa** through reasoning across the focused video frames: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2023_Glance_and_Focus_Memory_Prompting_for_Multi-Event_Video_Question_Answering_images/nips_2023_Glance_and_Focus_Memory_Prompting_for_Multi-Event_Video_Question_Answering.pdf-0006-14.png)


Similarly to classifier header _fc_ , an answer classifier header _fa_ of a linear layer with softmax is used ˆ to predict the answer label from the last token of answer queries _y_ = _fa_ ( **ˆa** ). 

## **3.4 Loss Functions** 

The first part of loss functions is the answer classification cross-entropy loss _Lqa_ . The overall unsupervised loss is defined as a linear combination: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2023_Glance_and_Focus_Memory_Prompting_for_Multi-Event_Video_Question_Answering_images/nips_2023_Glance_and_Focus_Memory_Prompting_for_Multi-Event_Video_Question_Answering.pdf-0006-18.png)


6 

Under the situation with ground-truth event-level labels, we turn to use them for supervision by minimizing the event classification cross-entropy loss _Lcls_ and the temporal span _L_ 1 loss _LL_ 1 between the matching memories with the ground-truth events: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2023_Glance_and_Focus_Memory_Prompting_for_Multi-Event_Video_Question_Answering_images/nips_2023_Glance_and_Focus_Memory_Prompting_for_Multi-Event_Video_Question_Answering.pdf-0007-01.png)


## **4 Evaluation** 

## **4.1 Datasets** 

We evaluate our model on various recently proposed Multi-Event VideoQA datasets. The comparison of them is shown in Table 1. Note that we use the v2 version of the AGQA dataset, which has more balanced distributions, as the dataset creator recommended. We choose these benchmarks for a broad coverage of: **Video content** . Including household and general activities, _etc_ . **Video duration** . The average video length ranges from 12 seconds to 44 seconds. **Event complexity** . All these videos contain multiple events or actions, with the average event number ranging from 2.7 to 8.8. **Event complexity** . The number of event categories varies from 50 to 793. 

Table 1: Comparison of the Multi-Event VideoQA benchmarks. 

|Datasets|# Avg.|Video|#Event|#Videos|# QAs|
|---|---|---|---|---|---|
||Events/Video|Length|Classes|||
|STAR[42]|2.7|12s|157|22K|60K|
|AGQA v2[14]|6.8|30s|157|9.7K|2.27M|
|EgoTaskQA[16]|5|25s|793|2K|40K|
|NExT-QA[43]|8.8|44s|50|5.4K|52K|



## **4.2 Implementation Details** 

For each benchmark, we follow standard protocols outlined by prior works for data processing, metrics, and settings to ensure fair comparisons. For video feature extraction, we use frozen S3D model [46] pre-trained on HowTo100M[32] on STAR dataset and C3D[38] feature on EgoTaskQA dataset following[47, 16]. We also try Faster-RCNN model[37] pretrained on the Visual Genome[22] and CLIP(ViT-B/32) for more challenging AGQA and NExT-QA datasets. We employ a standard 2-layer, 8-head Transformer Encoder-Decoder with hidden size _D_ of 512 as the backbone for our Glance-Focus model. According to the statistics of common event categories, the query number _N_ is set to 10 since the maximum average number of events per video in the current dataset is approximately 8.8. And the event class number _C_ is set to the ground truth overall class number in the dataset or 100 if it is unknown by default. For training details, we use dropout of 0.1, and initialize model weights using Xavier init[13]. Adam optimizer[20] is used with a learning rate of 5e-6 to optimize model parameters. All experiments are conducted on an NVIDIA GeForce RTX 3090Ti GPU. More details can be found in the supplementary material. 

## **4.3 Comparison with State-of-the-arts** 

We conduct a comprehensive comparison with current state-of-the-arts (SOTA) VideoQA methods, including traditional spatial-temporal models and large vision language models. We compare our Glance-Focus model (shorted as GF model in the following) with them on all four datasets. The results are shown in Table 2, 3, 4 and 5 respectively. Overall, our proposed GF model achieves SOTA results on all benchmarks, and obtains significant advantages in both the STAR and the EgoTaskQA dataset. We will analyze different datasets one by one as follows. Note that in Table 2-5 the red numbers highlight the significant gains compared to prior SOTA results. 

For **STAR** dataset, our proposed model obtains excellent reasoning performance shown in Table 2. When facing all four different types of questions, GF significantly surpasses the previous SOTAs, especially on the Sequence category. Such results reflect that our model has strong temporal contextual modeling ability. What’s more, note that our model trained without ground-truth event labels, _i.e._ , GF(uns), has achieved comparable results with the one trained under supervision, _i.e._ , GF(sup). It indicates that our proposed unsupervised memory generation in the Glance stage can effectively extract key events with strong discriminability. 

7 

Table 2: QA accuracies of SOTA methods on STAR test set (* indicates the val set results). 

|Model|Interaction|Sequence|Prediction|Feasibility|Mean|
|---|---|---|---|---|---|
|ClipBERT[25]|39.81|43.59|32.34|31.42|36.70|
|RESERVE-B[50]|44.80|42.40|38.80|36.20|40.50|
|Flamingo-9B[1]|-|-|-|-|43.40|
|AIO[40]*|47.53|50.81|47.75|44.08|47.54|
|Temp[ATP][3]*|50.63|52.87|49.36|40.61|48.37|
|MIST[10]*|55.59|54.23|54.24|44.48|51.13|
|GF(uns)|51.91|**63.06**(+8.83)|**54.89**|45.57|53.86(+2.73)|
|GF(sup)|**56.10**|61.27(+7.04)|52.65|**45.74**|**53.94**(+2.81)|



We also evaluate our model in the normal split of the **EgoTaskQA** dataset. From Table 3, we can find that our model outperforms SOTA methods on almost all question types. Our unsupervised GF model has strong performance especially on the intent and explanatory question types. These questions evaluate the understanding of the humans daily activities, which are all necessary for building the household robots. And the model trained under supervision achieves strong performance especially on object and predictive question types, which indicates that by explicitly modeling the events happening in the video, the model can obtain stronger event semantic understanding ability, namely, to recognize the interacted objects and to predict the next action. 

Table 3: QA accuracies of SOTA methods on EgoTaskQA normal split. 

|Category|VisualBERT[26]<br>ClipBERT[25]<br>HCRN[23]|GF(uns)<br>GF(sup)|
|---|---|---|
|Scope<br>world<br>intent<br>multi-agent|39.73<br>42.15<br>44.27<br>44.51<br>40.94<br>49.77<br>26.29<br>27.63<br>31.36|44.65<br>**46.53**<br>**52.07**(+2.30)<br>51.88<br>33.17<br>**34.31**|
|Type<br>descriptive<br>predictive<br>counterfactual<br>explanatory|41.99<br>38.45<br>43.48<br>30.37<br>31.50<br>36.56<br>41.99<br>46.75<br>**48.00**<br>37.42<br>42.39<br>40.60|44.48<br>**46.68**(+3.20)<br>39.00<br>**40.36**(+3.80)<br>46.66<br>46.80<br>**42.45**<br>42.18|
|Semantic<br>action<br>object<br>state<br>change|15.02<br>22.91<br>14.92<br>23.26<br>21.80<br>45.31<br>59.20<br>54.36<br>68.28<br>68.27<br>66.58<br>67.38|15.68<br>**15.68**<br>45.97<br>**50.98**(+5.67)<br>70.44<br>**71.23**<br>68.69<br>**69.71**|
|Overall<br>open<br>binary<br>all|24.62<br>27.70<br>30.23<br>68.08<br>67.52<br>69.42<br>37.93<br>39.87<br>42.20|31.26<br>**32.79**(+2.56)<br>69.80<br>**70.25**<br>43.06<br>**44.27**(+2.07)|



As for **AGQA** dataset, from Table 4, our model also achieves the SOTA result with object-level video features, _i.e._ , GF(sup) with Faster R-CNN[37] backbone. And the one trained with S3D features also gains comparable results. We show several representative reasoning types of the dataset. Our model obtains better results on some challenging question types, such as obj-rel composition questions and temporal superlative questions like “ _the longest..._ ”. Overall, our model shows more clear advantage over SOTA on the challenging Open questions, compared to the relatively simple Binary questions for which the model could gain a 50% accuracy rate by blind guessing. The complete breakdown results on all kinds of question types and a more thorough analysis can be found in the Supplementary Material. 

Table 4: QA accuracies of SOTA methods on AGQA v2 test set. 

|Method|HME[6]<br>HCRN[23]<br>AIO[40]|Temp[3]<br>MIST[10]|GF(uns)<br>GF(uns)<br>GF(sup)|
|---|---|---|---|
|Backbone|C3D<br>C3D<br>ViT|CLIP<br>CLIP|S3D<br>FasterRCNN<br>FasterRCNN|
|obj-rel<br>superlative<br>sequencing<br>exist<br>duration<br>act. recog.|37.42<br>40.33<br>48.34<br>33.21<br>33.55<br>37.53<br>49.77<br>49.70<br>49.61<br>49.96<br>50.01<br>50.81<br>47.03<br>43.84<br>45.36<br>5.43<br>5.52<br>18.97|50.15<br>51.68<br>39.78<br>42.05<br>48.25<br>**67.24**<br>51.79<br>**60.33**<br>49.59<br>**54.62**<br>18.96<br>19.69|52.93<br>54.31<br>**54.96**(+3.28)<br>42.78<br>42.90<br>**44.62**(+2.57)<br>53.03<br>51.97<br>53.24<br>58.31<br>58.08<br>59.13<br>50.86<br>52.02<br>52.80<br>**22.08**(+2.39)<br>16.38<br>14.17|
|open<br>binary<br>all|31.01<br>36.34<br>-<br>48.91<br>47.97<br>-<br>39.89<br>42.11<br>48.59|-<br>50.56<br>-<br>**58.28**<br>49.79<br>54.39|53.06<br>55.66<br>**56.07**(+5.51)<br>53.61<br>53.52<br>54.17<br>53.33<br>54.59<br>**55.08**|



8 

We also evaluate our model on the challenging **NExT-QA** dataset based on different video backbones. Note that the NExT-QA dataset does not provide event annotations. Therefore, we exploit the proposed unsupervised memory generation in the Glance stage. As shown in Table 5, without event-level supervision or applying any extra action detection model, the unsupervised GF models with the same backbone as other methods still achieve SOTA performances. Such results further justify the strong multi-event reasoning ability of our method, considering that NExT-QA is currently the most complex Multi-Event VideoQA benchmark with longer video duration (44s) and richer events (8.8 actions per video). 

What’s more, we also report the breakdown results by the different question types. According to the results, GF has strong event reasoning ability especially on Descriptive (Acc@D) questions, improving the accuracy by 3.61%, and also on event relationship reasoning, achieving 2.31% accuracy gain on Causal (Acc@C) questions. As recommended in ATP[3], we additionally report results on its proposed ATP-hard subset of the NExT-QA validation set. This subset filters out those “easy” questions that can be answered with a single frame. The results in Table 5 show that our GF models significantly surpass HGA[17] models on this subset by improving 5.22% accuracy overall. The above results indicate that compared to HGA and other methods that directly model the correspondence between question and long video through co-attention, our method using event memory as an intermediate bridge to establish such correlation for reasoning is more efficient. 

Table 5: Accuracy (%) comparison on NExT-QA validation set and the ATP-hard subset[3]. Acc@C, T, D, denote accuracy for Causal, Temporal, and Descriptive questions respectively. 

|Methods<br>Backbone|Methods<br>Backbone|val<br>val (ATP-hard subset)<br>Acc@C<br>Acc@T<br>Acc@D<br>Acc@All<br>Acc@C<br>Acc@T<br>Acc@All|val (ATP-hard subset)|
|---|---|---|---|
||||Acc@C<br>Acc@T<br>Acc@All|
|ATP[3]<br>VQA-T[47]<br>IGV*[28]<br>Temp[ATP][3]<br>HGA[17]<br>MIST[10]|CLIP<br>S3D<br>C3D<br>CLIP<br>C3D<br>CLIP|48.30<br>49.30<br>65.00<br>49.20<br>49.60<br>51.49<br>63.19<br>52.32<br>48.56<br>51.67<br>59.64<br>51.34<br>53.10<br>50.20<br>66.80<br>54.30<br>46.26<br>50.74<br>59.33<br>49.74<br>54.62<br>56.64<br>66.92<br>57.18|19.60<br>22.60<br>20.20<br>-<br>-<br>-<br>-<br>-<br>-<br>38.40<br>36.50<br>38.80<br>43.30<br>45.30<br>44.10<br>-<br>-<br>-|
|GF(uns)<br>GF(uns)<br>GF(uns)|S3D<br>C3D<br>CLIP|51.78<br>54.71<br>63.96<br>54.62(+2.30)<br>53.59<br>55.71<br>66.80<br>56.33(+4.99)<br>**56.93**(+2.31)<br>**57.07**<br>**70.53**(+3.61)<br>**58.83**|48.28<br>49.95<br>48.96<br>48.50<br>49.52<br>48.92<br>**48.65**(+5.35)<br>**50.27**(+4.97)<br>**49.32**(+5.22)|



## **4.4 Model Analysis** 

In this section, we evaluate the effectiveness of our proposed modules by comparing different variants of the GF models and various baselines. We first introduce the baselines and GF variants as follows. **Baselines** : 1) **MDETR** [18] which utilizes same 2-layer Transformer Encoder-Decorder architecture. 2) An **Oracle** model with ground-truth event labels concatenated to the questions as textual input as an upper-bound model. **GF Variants** : 1) **Glance-only** . Model which uses standard cross-attention to replace the multi-level attention strategy. 2) **Detect-Focus** . Apply a frozen action detection model, such as SlowFast[7], to replace the Glance stage of the model. We try to keep the model parameters consistent. 

**Effect of memory prompting.** Firstly, the performance of the Oracle model significantly improves compared to the baseline model. It convinced us that event memories do play an important prompting role in the Multi-Event VideoQA task. Besides, it can be found that with event memory generation at the glancing stage, the Glance-only model outperforms the MDETR baseline and the model using a frozen action detection (Detect-Focus) with a large gap on both datasets. Such results reflect the effectiveness of our proposed memory-prompting strategy. Compared to additionally fine-tuning an action detection model, our proposed Glance strategy can adaptively generate key event/action features of the long videos without any annotations. 

What’s more, we evaluate the effectiveness of different loss functions that are designed during the event memory extraction in Table 7. We carefully evaluate all pairs of losses on STAR, EgoTaskQA and AGQA datasets. Firstly, the model with complete losses achieves the best results on both unsupervised and supervised settings of all datasets. It indicates that each loss plays its role in generating the event memories. Secondly, under the unsupervised setting, the _Liou_ shows an 

9 

important role in controlling the temporal prediction of the events. For supervised settings, the _Lcls_ shows a more important role for the event semantic supervision helps cross-modal grounding. Please see more discussions on the loss functions in the Supplementary Material. 

**Effect of multi-level attention.** From Table 6, compared to the Glance-only model using standard cross-attention, the GlanceFocus model with our designed multi-level attention shows stronger reasoning ability on both STAR and EgoTaskQA datasets. Taking an example from STAR dataset, we further visualize the attention weights in different levels in Figure 3, the darker the color, the greater the weight. The attention map in the first row shows the correspondence of the question embeddings with the generated memories. In this level, the model focuses on the most related event “ _Throwing a towel_ ”. The attention map in the second row is the correspondence between the focused event and the 

Table 6: Module ablations on STAR and EgoTaskQA datasets. 

|Model|STAR<br>EgoTaskQ|
|---|---|
|MDETR<br>Detect-Focus<br>Glance-only<br>Glance-Focus<br>Oracle(Upper)|46.26<br>39.68<br>49.85<br>42.62<br>53.12<br>42.70<br>**53.94**<br>**44.27**<br>55.19<br>49.52|



video embeddings. At this level, the model correctly attends to the most related keyframes for answer prediction. 

Table 7: Ablation study of the loss functions on all datasets. 

|Uns/Sup|_Luns_<br>_cls_<br>_Liou_<br>_Lcert_|_Lsup_<br>_cls_<br>_LL_1|AGQA<br>STAR<br>EgoTaskQA|
|---|---|---|---|
|Uns<br>Uns<br>Uns<br>Uns|✓<br>✓<br>✓<br>✓<br>✓<br>✓<br>✓<br>✓<br>✓||54.12<br>53.62<br>42.69<br>52.89<br>51.35<br>41.77<br>54.27<br>53.72<br>42.85<br>**54.59**<br>**53.86**<br>**43.06**|
|Sup<br>Sup<br>Sup||✓<br>✓<br>✓<br>✓|54.26<br>53.69<br>43.87<br>53.19<br>53.03<br>42.64<br>**55.08**<br>**53.94**<br>**44.27**|




![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2023_Glance_and_Focus_Memory_Prompting_for_Multi-Event_Video_Question_Answering_images/nips_2023_Glance_and_Focus_Memory_Prompting_for_Multi-Event_Video_Question_Answering.pdf-0010-07.png)


**----- Start of picture text -----**<br>
Video:<br>[event;GT: Someone is eating somethingPutting some food somewhere Washing something with a towel Throwing a towel/s  Watching/Looking outside of a window 𝑡<br>(center, somewhere<br>width)] (0.2023, 0.4047) (0.2093, 0.4186) (0.3744, 0.2558) (0.4279, 0.2233) (0.7395, 0.6605)<br>Pred: (0.2413, 0.3042)Holding a dish Putting some food somewhere(0.2594, 0.4277) Washing something with a towel(0.3801, 0.2693) Throwing a towel/s somewhere(0.4612, 0.2537) Holding a phone/camera(0.7805, 0.6305)<br>Which<br>objectdid<br>the<br>personthrow<br>after<br>they<br>downput<br>the<br>food?<br>Holding a dish Putting some food somewhere Washing something with a towel Throwing a towel/s somewhere Holding a phone/camera<br>frame 1 frame 𝑇<br>Question: Which object did the person throw after they put down the food?  Predicted Answer: The towel.<br>Throwing<br>a towel/s<br>(𝑀<br>somewhere!"#)<br>**----- End of picture text -----**<br>


Figure 3: Visualization of the event memory generation and the multi-level attention map. 

## **5 Conclusion** 

Our proposed Glance-Focus model provides an innovative approach to the challenging Multi-Event VideoQA task. With a set of event memories as prompts, the model can efficiently establish the correspondence between high-level questions and low-level video frames. Our experiments has shown that event memories play a crucial role in locating question-related visual clues for reasoning. The proposed Glance-Focus strategy can obtain a complete and compact representation of videos without additional annotations, which can be extended to more video comprehension tasks, such as video captioning/summarization. We hope this study can inspire more future research in the field of Multi-Event Video Understanding. 

10 

## **Acknowledgements** 

This work is partially supported by National Key R&D Program of China No. 2021ZD0111901, and Natural Science Foundation of China under contracts Nos. U21B2025, U19B2036. 

## **References** 

- [1] Jean-Baptiste Alayrac, Jeff Donahue, Pauline Luc, Antoine Miech, Iain Barr, Yana Hasson, Karel Lenc, Arthur Mensch, Katie Millican, Malcolm Reynolds, et al. Flamingo: a visual language model for few-shot learning. _Advances in Neural Information Processing Systems_ , 2022. 

- [2] Max Bain, Arsha Nagrani, Gül Varol, and Andrew Zisserman. Frozen in time: A joint video and image encoder for end-to-end retrieval. In _Proceedings of the IEEE/CVF International Conference on Computer Vision_ , pages 1728–1738, 2021. 

- [3] Shyamal Buch, Cristóbal Eyzaguirre, Adrien Gaidon, Jiajun Wu, Li Fei-Fei, and Juan Carlos Niebles. Revisiting the “video” in video-language understanding. In _Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition_ , pages 2917–2927, 2022. 

- [4] Nicolas Carion, Francisco Massa, Gabriel Synnaeve, Nicolas Usunier, Alexander Kirillov, and Sergey Zagoruyko. End-to-end object detection with transformers. In _Proceedings of the European Conference on Computer Vision_ , pages 213–229, 2020. 

- [5] Long Hoang Dang, Thao Minh Le, Vuong Le, and Truyen Tran. Hierarchical object-oriented spatiotemporal reasoning for video question answering. _International Joint Conference on Artificial Intelligence_ , 2021. 

- [6] Chenyou Fan, Xiaofan Zhang, Shu Zhang, Wensheng Wang, Chi Zhang, and Heng Huang. Heterogeneous memory enhanced multimodal attention model for video question answering. In _Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition_ , pages 1999–2007, 2019. 

- [7] Christoph Feichtenhofer, Haoqi Fan, Jitendra Malik, and Kaiming He. Slowfast networks for video recognition. In _Proceedings of the IEEE/CVF International Conference on Computer Vision_ , pages 6202–6211, 2019. 

- [8] Tsu-Jui Fu, Linjie Li, Zhe Gan, Kevin Lin, William Yang Wang, Lijuan Wang, and Zicheng Liu. Violet: Endto-end video-language transformers with masked visual-token modeling. _arXiv preprint arXiv:2111.12681_ , 2021. 

- [9] Difei Gao, Ruiping Wang, Ziyi Bai, and Xilin Chen. Env-qa: a video question answering benchmark for comprehensive understanding of dynamic environments. In _Proceedings of the IEEE/CVF International Conference on Computer Vision_ , pages 1675–1685, 2021. 

- [10] Difei Gao, Luowei Zhou, Lei Ji, Linchao Zhu, Yi Yang, and Mike Zheng Shou. Mist: Multi-modal iterative spatial-temporal transformer for long-form video question answering. _arXiv preprint arXiv:2212.09522_ , 2022. 

- [11] Jiyang Gao, Runzhou Ge, Kan Chen, and Ram Nevatia. Motion-appearance co-memory networks for video question answering. In _Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition_ , pages 6576–6585, 2018. 

- [12] Yuying Ge, Yixiao Ge, Xihui Liu, Dian Li, Ying Shan, Xiaohu Qie, and Ping Luo. Bridgeformer: Bridging video-text retrieval with multiple choice questions. _Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition_ , 2022. 

- [13] Xavier Glorot and Yoshua Bengio. Understanding the difficulty of training deep feedforward neural networks. In _Proceedings of the 13th International Conference on Artificial Intelligence and Statistics_ , pages 249–256. JMLR Workshop and Conference Proceedings, 2010. 

- [14] Madeleine Grunde-McLaughlin, Ranjay Krishna, and Maneesh Agrawala. Agqa: A benchmark for compositional spatio-temporal reasoning. In _Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition_ , pages 11287–11297, 2021. 

- [15] Deng Huang, Peihao Chen, Runhao Zeng, Qing Du, Mingkui Tan, and Chuang Gan. Location-aware graph convolutional networks for video question answering. In _Proceedings of the AAAI Conference on Artificial Intelligence_ , pages 11021–11028, 2020. 

- [16] Baoxiong Jia, Ting Lei, Song-Chun Zhu, and Siyuan Huang. Egotaskqa: Understanding human tasks in egocentric videos. _Advances in Neural Information Processing Systems_ , 2022. 

- [17] Pin Jiang and Yahong Han. Reasoning with heterogeneous graph alignment for video question answering. In _Proceedings of the AAAI Conference on Artificial Intelligence_ , pages 11109–11116, 2020. 

- [18] Aishwarya Kamath, Mannat Singh, Yann LeCun, Gabriel Synnaeve, Ishan Misra, and Nicolas Carion. Mdetr-modulated detection for end-to-end multi-modal understanding. In _Proceedings of the IEEE/CVF International Conference on Computer Vision_ , pages 1780–1790, 2021. 

- [19] Sungdong Kim, Jin-Hwa Kim, Jiyoung Lee, and Minjoon Seo. Semi-parametric video-grounded text generation. _arXiv preprint arXiv:2301.11507_ , 2023. 

- [20] Diederik P Kingma and Jimmy Ba. Adam: A method for stochastic optimization. _International Conference on Learning Representations_ , 2015. 

- [21] Andreas Krause, Pietro Perona, and Ryan Gomes. Discriminative clustering by regularized information maximization. _Advances in Neural Information Processing Systems_ , 23, 2010. 

11 

- [22] Ranjay Krishna, Yuke Zhu, Oliver Groth, Justin Johnson, Kenji Hata, Joshua Kravitz, Stephanie Chen, Yannis Kalantidis, Li-Jia Li, David A Shamma, et al. Visual genome: Connecting language and vision using crowdsourced dense image annotations. _International Journal of Computer Vision_ , 123:32–73, 2017. 

- [23] Thao Minh Le, Vuong Le, Svetha Venkatesh, and Truyen Tran. Hierarchical conditional relation networks for video question answering. In _Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition_ , pages 9972–9981, 2020. 

- [24] Jie Lei, Tamara L Berg, and Mohit Bansal. Detecting moments and highlights in videos via natural language queries. _Advances in Neural Information Processing Systems_ , 34:11846–11858, 2021. 

- [25] Jie Lei, Linjie Li, Luowei Zhou, Zhe Gan, Tamara L Berg, Mohit Bansal, and Jingjing Liu. Less is more: Clipbert for video-and-language learning via sparse sampling. In _Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition_ , pages 7331–7341, 2021. 

- [26] Liunian Harold Li, Mark Yatskar, Da Yin, Cho-Jui Hsieh, and Kai-Wei Chang. Visualbert: A simple and performant baseline for vision and language. _arXiv preprint arXiv:1908.03557_ , 2019. 

- [27] Muheng Li, Lei Chen, Yueqi Duan, Zhilan Hu, Jianjiang Feng, Jie Zhou, and Jiwen Lu. Bridge-prompt: Towards ordinal action understanding in instructional videos. In _Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition_ , pages 19880–19889, 2022. 

- [28] Yicong Li, Xiang Wang, Junbin Xiao, Wei Ji, and Tat-Seng Chua. Invariant grounding for video question answering. In _Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition_ , pages 2928–2937, 2022. 

- [29] Jian Liang, Dapeng Hu, and Jiashi Feng. Do we really need to access the source data? source hypothesis transfer for unsupervised domain adaptation. In _International Conference on Machine Learning_ , pages 6028–6039, 2020. 

- [30] Fei Liu, Jing Liu, Weining Wang, and Hanqing Lu. Hair: Hierarchical visual-semantic relational reasoning for video question answering. In _Proceedings of the IEEE/CVF International Conference on Computer Vision_ , pages 1698–1707, 2021. 

- [31] Yinhan Liu, Myle Ott, Naman Goyal, Jingfei Du, Mandar Joshi, Danqi Chen, Omer Levy, Mike Lewis, Luke Zettlemoyer, and Veselin Stoyanov. Roberta: A robustly optimized bert pretraining approach. _arXiv preprint arXiv:1907.11692_ , 2019. 

- [32] Antoine Miech, Dimitri Zhukov, Jean-Baptiste Alayrac, Makarand Tapaswi, Ivan Laptev, and Josef Sivic. Howto100m: Learning a text-video embedding by watching hundred million narrated video clips. In _Proceedings of the IEEE/CVF International Conference on Computer Vision_ , pages 2630–2640, 2019. 

- [33] Tushar Nagarajan, Yanghao Li, Christoph Feichtenhofer, and Kristen Grauman. Ego-topo: Environment affordances from egocentric video. In _Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition_ , pages 163–172, 2020. 

- [34] Megha Nawhal and Greg Mori. Activity graph transformer for temporal action localization. _arXiv preprint arXiv:2101.08540_ , 2021. 

- [35] Jungin Park, Jiyoung Lee, and Kwanghoon Sohn. Bridge to answer: Structure-aware graph interaction network for video question answering. In _Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition_ , pages 15526–15535, 2021. 

- [36] Tianwen Qian, Ran Cui, Jingjing Chen, Pai Peng, Xiaowei Guo, and Yu-Gang Jiang. Locate before answering: Answer guided question localization for video question answering. _IEEE Transactions on Multimedia_ , 2023. 

- [37] Shaoqing Ren, Kaiming He, Ross Girshick, and Jian Sun. Faster r-cnn: Towards real-time object detection with region proposal networks. _Advances in Neural Information Processing Systems_ , pages 91–99, 2015. 

- [38] Du Tran, Lubomir Bourdev, Rob Fergus, Lorenzo Torresani, and Manohar Paluri. Learning spatiotemporal features with 3d convolutional networks. In _Proceedings of the IEEE/CVF International Conference on Computer Vision_ , pages 4489–4497, 2015. 

- [39] Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez, Łukasz Kaiser, and Illia Polosukhin. Attention is all you need. _Advances in Neural Information Processing Systems_ , 30, 2017. 

- [40] Alex Jinpeng Wang, Yixiao Ge, Rui Yan, Yuying Ge, Xudong Lin, Guanyu Cai, Jianping Wu, Ying Shan, Xiaohu Qie, and Mike Zheng Shou. All in one: Exploring unified video-language pre-training. _Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition_ , 2022. 

- [41] Jianyu Wang, Bing-Kun Bao, and Changsheng Xu. Dualvgr: A dual-visual graph reasoning unit for video question answering. _IEEE Transactions on Multimedia_ , pages 3369–3380, 2021. 

- [42] Bo Wu, Shoubin Yu, Zhenfang Chen, Joshua B Tenenbaum, and Chuang Gan. Star: A benchmark for situated reasoning in real-world videos. In _Advances in Neural Information Processing Systems_ , 2021. 

- [43] Junbin Xiao, Xindi Shang, Angela Yao, and Tat-Seng Chua. Next-qa: Next phase of question-answering to explaining temporal actions. In _Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition_ , pages 9777–9786, 2021. 

- [44] Junbin Xiao, Angela Yao, Zhiyuan Liu, Yicong Li, Wei Ji, and Tat-Seng Chua. Video as conditional graph hierarchy for multi-granular question answering. Proceedings of the AAAI Conference on Artificial Intelligence, 2022. 

- [45] Junbin Xiao, Pan Zhou, Tat-Seng Chua, and Shuicheng Yan. Video graph transformer for video question answering. In _Proceedings of the European Conference on Computer Vision_ , pages 39–58, 2022. 

- [46] Saining Xie, Chen Sun, Jonathan Huang, Zhuowen Tu, and Kevin Murphy. Rethinking spatiotemporal feature learning: Speed-accuracy trade-offs in video classification. In _Proceedings of the European_ 

12 

_Conference on Computer Vision_ , pages 305–321, 2018. 

- [47] Antoine Yang, Antoine Miech, Josef Sivic, Ivan Laptev, and Cordelia Schmid. Just ask: Learning to answer questions from millions of narrated videos. In _Proceedings of the IEEE/CVF International Conference on Computer Vision_ , pages 1686–1697, 2021. 

- [48] Shoubin Yu, Jaemin Cho, Prateek Yadav, and Mohit Bansal. Self-chained image-language model for video localization and question answering. _arXiv preprint arXiv:2305.06988_ , 2023. 

- [49] Zhou Yu, Jun Yu, Yuhao Cui, Dacheng Tao, and Qi Tian. Deep modular co-attention networks for visual question answering. In _Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition_ , pages 6281–6290, 2019. 

- [50] Rowan Zellers, Jiasen Lu, Ximing Lu, Youngjae Yu, Yanpeng Zhao, Mohammadreza Salehi, Aditya Kusupati, Jack Hessel, Ali Farhadi, and Yejin Choi. Merlot reserve: Multimodal neural script knowledge through vision and language and sound. In _Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition_ , 2022. 

- [51] Zhou Zhao, Zhu Zhang, Shuwen Xiao, Zhenxin Xiao, Xiaohui Yan, Jun Yu, Deng Cai, and Fei Wu. Long-form video question answering via dynamic hierarchical reinforced networks. _IEEE Transactions on Image Processing_ , pages 5939–5952, 2019. 

- [52] Linchao Zhu and Yi Yang. Actbert: Learning global-local video-text representations. In _Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition_ , pages 8746–8755, 2020. 

13 

