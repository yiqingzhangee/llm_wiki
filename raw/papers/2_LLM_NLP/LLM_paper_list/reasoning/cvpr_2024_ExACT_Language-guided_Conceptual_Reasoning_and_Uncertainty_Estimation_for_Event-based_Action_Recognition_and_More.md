# **ExACT: Language-guided Conceptual Reasoning and Uncertainty Estimation for Event-based Action Recognition and More** 

Jiazhou Zhou[1] Xu Zheng[1] Yuanhuiyi Lyu[1] Lin Wang[1] _[,]_[2][*] 1AI Thrust, HKUST(GZ) 2Dept. of CSE, HKUST 

jiazhouzhou@hkust-gz.edu.cn, yuanhuiyilv@hkust-gz.edu.cn, zhengxu128@gmail.com, linwang@ust.hk Project Page: https://vlislab22.github.io/ExACT/ 

## **Abstract** 

_Event cameras have recently been shown beneficial for practical vision tasks, such as action recognition, thanks to their high temporal resolution, power efficiency, and reduced privacy concerns. However, current research is hindered by 1) the difficulty in processing events because of their prolonged duration and dynamic actions with complex and ambiguous semantics and 2) the redundant action depiction of the event frame representation with fixed stacks. We find language naturally conveys abundant semantic information, rendering it stunningly superior in reducing semantic uncertainty. In light of this, we propose ExACT, a novel approach that, for the first time, tackles event-based action recognition from a cross-modal conceptualizing perspective. Our ExACT brings two technical contributions. Firstly, we propose an adaptive fine-grained event (AFE) representation to adaptively filter out the repeated events for the stationary objects while preserving dynamic ones. This subtly enhances the performance of ExACT without extra computational cost. Then, we propose a conceptual reasoning-based uncertainty estimation module, which simulates the recognition process to enrich the semantic representation. In particular, conceptual reasoning builds the temporal relation based on the action semantics, and uncertainty estimation tackles the semantic uncertainty of actions based on the distributional representation. Experiments show that our ExACT achieves superior recognition accuracy of 94.83%(+2.23%), 90.10%(+37.47%) and 67.24% on PAF, HARDVS and our SeAct datasets respectively._ 

## **1. Introduction** 

Action recognition is a crucial vision task with many applications, such as robot navigation [33, 43], and abnormal human behavior recognition [25, 35]. Many framebased learning approaches have been presented, leading to impressive performance improvements [24, 43]. How- 

- *Corresponding author. 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/cvpr_2024_ExACT_Language-guided_Conceptual_Reasoning_and_Uncertainty_Estimation_for_Event-based_Action_Recognition_and_More_images/cvpr_2024_ExACT_Language-guided_Conceptual_Reasoning_and_Uncertainty_Estimation_for_Event-based_Action_Recognition_and_More.pdf-0001-09.png)


**----- Start of picture text -----**<br>
(a) Dynamic Action:  Stationary Object:<br>Comparison Sit down Dragonfly<br>1) Duration: Avg. 5s Prolonged Avg. 0.1s Short<br>2) Semantic: Ambiguous  Limited<br>&Complex<br>(b) Our AEF representation VS. Previous event frame rep.<br>t t<br>(c) Conceptual  Contrastive  Uncertainty<br>Reasoning  Alignment Estimation<br>Squatting?<br>Sitting?<br>Temporal Relation Standing?<br>**----- End of picture text -----**<br>


Figure 1. **(a)** Unlike stationary objects, _e.g_ ., `‘Dragonfly’` with short duration (0.1s) and limited semantics, dynamic actions like `‘Sit down’` have the prolonged duration (5s) with ambiguous and complex semantics. **(b)** Compared with previous event representation, stacking events with fixed counts, we adaptively filter out events recording stationary actions while preserving dynamic ones; **(c)** We introduce language guidance to stimulate the recognition process, particularly focusing on conceptually reasoning temporal relations and estimating uncertain semantics. 

ever, these methods may not be ideal solutions for powerconstrained scenarios, _e.g_ ., surveillance [3, 24]. RGB cameras also degrade due to environment bias [36] like motion blur and lighting variations. Moreover, frame-based cameras trigger considerable privacy concerns as they directly capture users’ appearance. 

Recently, bio-inspired event cameras are gaining popularity [1, 14, 27], which ignore the background and only record moving objects. This leads to sensing efficiency and resilience to rapid motion and illumination changes with low power consumption. Also, event cameras mostly reflect objects’ edges, which alleviates users’ privacy concerns like skin color and gender. Owing to these advantages, eventbased action recognition offers more pragmatic solutions for real-world implementations. This has inspired research endeavors [14, 27, 36, 39, 49, 54, 55] in event-based action recognition areas with plausible performance. 

However, the above methods are deficient for two reasons: 1) They have a limited capacity for recognizing a large number of different human actions, as demonstrated by experiments on the HARDVS [48] dataset with 300 categories in Tab. 1. This probably arises from the complex and ambiguous semantics induced by the dynamic actions and prolonged duration (around 5s [48]), compared to objects with short duration (around 0.1s [48]) and limited semantics. For example, as shown in Fig. 1 (a), `‘Dragonfly’` _vs._ `‘Sit down’` . 2) The lack of _tailored event representation_ as the raw events are directly stacked into event frames with fixed stacks, resulting in event frames with either overlapped or vague edge information depicting the same action, see Fig. 3 and Fig. 1 (b). 

Recent advancements in vision-language models (VLMs) [6, 11, 37] have pioneered ideas that incorporate semantic concepts across text and vision modalities, aiming at simulating the human processes of conceptualization and reasoning [19, 41, 42]. _The key insight is that language naturally conveys inherent semantic richness, which can be beneficial for modeling semantic uncertainty and establishing complex semantic relations_ . Inspired by this, we introduce language as guidance for event-based action recognition. As the first exploration, the research hurdles include: 1) How to represent events to depict dynamic actions in detail without redundant event frames? 2) How to integrate text embeddings with event embeddings to help reason complex semantics of dynamic actions and reduce semantic uncertainty? 

To this end, we propose a novel ExACT framework to tackle event-based action recognition from a cross-modal conceptual reasoning perspective, as depicted in Fig. 1 (b) and (c). To address the first challenge, an Adaptive Finegrained Event (AFE) representation (Sec. 3.1) is inspired by the ‘overlapped action regions’ observed in Fig. 3. These regions indicate an excessive stack of events in one frame, which is inevitable for previous frame-based representations with fixed stacks. Differently, our AFE recursively and offline find the dividing line of different actions based on the overlapped regions. It eliminates repeated events and preserves dynamic actions, thus enhancing model performance without extra computational costs (Tab. 2). 

For the second challenge, we propose a novel Conceptual Reasoning-based Uncertainty Estimation (CRUE) module (Sec. 3.3) to simulate the action recognition process of humans. Concretely, CRUE initially establishes the temporal relation of event frames by leveraging the text embeddings to reason each frame’s semantics and then obtain the fused event embeddings. Subsequently, CRUE converts event and text embeddings from discrete representation to distributional representation, where the distribution variance quantifies semantic uncertainty. In this way, our proposed CRUE module establishes a semantic-abundant 

and uncertainty-aware embedding space to enhance model performance (Tab. 3). 

Meanwhile, as existing datasets solely provide categorylevel labels, we propose the SeAct dataset, consisting of 58 categories of actions, with semantic-abundant captionlevel labels. Our dataset serves as the first dataset for eventtext action recognition (67.24% accuracy). We also conduct extensive experiments to show that our ExACT outperforms previous methods , _e.g_ ., [14, 26] on the public datasets, PAF [34] 94.83% accuracy (+2.23%) and especially on HARDVS [48] 90.10% accuracy (+37.47%) by a large margin. Beyond action recognition, our ExACT can be flexibly applied to the event-text retrieval task. 

In summary, our main contributions are: (I) We propose the ExACT– the first framework utilizing language guidance for event-based action recognition; (II) We propose the CRUE module to mimic human action recognition, creating a rich, uncertainty-aware cross-modal embedding space for action recognition. Also, our AFE representation adaptively filters redundant events, yielding effective representation for dynamic actions. (III) We introduce the SeAct dataset with detailed text captions for evaluating the recognizing ability of actions composed of multiple sub-actions with different semantics. Extensive experiments demonstrate the superiority of our ExACT framework on our SeAct dataset and public datasets. 

## **2. Related Works** 

**Event-based Action Recognition** methods can be categorized into the event-only and event-other-modality methods. For event-only frameworks, the most widespread techniques involve stacking the event stream into compact frames, followed by the utilization of off-the-shelf Convolutional Neural Networks (CNNs) [14] or Vision Transformers (ViTs) [39, 55] for effective feature extraction. This approach currently exhibits state-of-the-art performance owing to the excellent CNN/ViT backbones performance. Meanwhile, considering the asynchronous characteristics of event data, the research community has explored the applicability of bio-inspired Spiking Neural Networks (SNNs) [27] and the spatiotemporal capabilities of Graph Convolutional Networks (GCNs) [54] to more aptly resonate with the unique structure of event data. However, these approaches have yielded suboptimal performance and exhibited limited adaptability, partly due to the specialized hardware requisites inherent to SNNs. 

Concurrently, there have been endeavors to integrate event data with additional modalities. For example, incorporating the abundance of color and texture information in RGB [49] data with event information or utilizing the supplementary motion knowledge in optical flow [36]. In summary, most of these approaches rely on dense consecutive event frames, inevitably resulting in redundant frames with 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/cvpr_2024_ExACT_Language-guided_Conceptual_Reasoning_and_Uncertainty_Estimation_for_Event-based_Action_Recognition_and_More_images/cvpr_2024_ExACT_Language-guided_Conceptual_Reasoning_and_Uncertainty_Estimation_for_Event-based_Action_Recognition_and_More.pdf-0003-00.png)


**----- Start of picture text -----**<br>
Adaptive Fine-grained  t 𝑓𝑒 …<br>Event Representation Event<br>Event stream Sec. 3.1 T Sampled Event Frames Encoder<br>T  Conceptual<br>Trainable Event [ CLS ]  Reasoning-based<br>Frozen A series of photos recoding human  𝑓𝑡ℎ Embedding Uncertainty Estimation<br>action for<br>Sit Down.  Hand-crafted Prompt 𝑃ℎ Text  𝑓𝑡 ModuleSec. 3.3  Embedding SpaceJoint Event-Text<br>Label Encoder<br>𝑉1 𝑉2 𝑉3 … 𝑉𝑀 𝑓 𝑡 𝑙 Text [ CLS ]<br>Learnable Prompt 𝑃l Embedding<br>**----- End of picture text -----**<br>


Figure 2. **Overall framework of our proposed ExACT framework.** It consists of four components: (1) the AFE representation recursively eliminates repeated events and generates event frames depicting dynamic actions (Sec. 3.1); (2) the event encoder and (3) the text encoder, responsible for the event and text embedding, respectively (Sec. 3.2); (4) the CRUE module simulates the action recognition process to establish the complex semantic relations for sub-actions and reduce the semantic uncertainty. (Sec. 3.3) 

overlapped actions and uncertain semantics. To represent events to depict detailed dynamic actions, the AFE representation is proposed to sample event frames adaptively without introducing extra computation costs. 

**Vision-laguage Models (VLMs)** Recently, there has been a growing interest in large-scale pre-trained VLMs [6, 11] for multimodal representations. Inspired by it, several pioneering works [7, 52, 58] have investigated the potential for transferring VLMs’ capability to the event modality, thereby revitalizing the best performance of objection recognition. In addition, the remarkable zero-shot capability of VLMs has motivated researchers to explore eventbased label-free [7] or zero (few)-shot applications [58], thus addressing the scarcity of high-quality event datasets. Nevertheless, prior event-text methods focus on recognizing objects with limited semantics but fail to recognize events recording actions with prolonged time duration and complex and ambiguous semantics. Therefore, Our ExACT aims at enhancing event-based action recognition from a cross-modal conceptual reasoning perspective. 

## **3. The Proposed ExACT Framework** 

**Overview.** An overview of our ExACT framework is depicted in Fig. 2. _The key idea of ExACT is to introduce language as guidance for estimating semantic uncertainty and establishing semantic relations for event-based action recognition._ The following subsections explain the technical details of 1) the proposed Adaptive Fine-grained Event (AFE) representation (Sec. 3.1); 2) the event encoder and the text encoder (Sec. 3.2); 3) the Conceptual Reasoningbased Uncertainty Estimation (CRUE) module (Sec. 3.3). Besides, in Sec. 3.5, we introduce our proposed **s** emanticabundant **e** vent-based **act** ion recognition ( **SeAct** ) dataset as the first dataset for event-text action recognition. 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/cvpr_2024_ExACT_Language-guided_Conceptual_Reasoning_and_Uncertainty_Estimation_for_Event-based_Action_Recognition_and_More_images/cvpr_2024_ExACT_Language-guided_Conceptual_Reasoning_and_Uncertainty_Estimation_for_Event-based_Action_Recognition_and_More.pdf-0003-06.png)


**----- Start of picture text -----**<br>
× ×<br>× × × t<br>𝐼10 𝐼11<br>Event Stream  𝐸00<br>t t<br>Event Stream 𝐸10 Event Stream 𝐸11<br>Repetitive frames …… Diff Map:𝐼10 −𝐼11 ……<br>R ≥Δ && 𝑁𝑠𝑢𝑏 ≥𝑁min ?<br>❓<br>M Times Recursion<br>The observed  t<br>‘overlapped  𝐼𝑀0 𝐼𝑀1 𝐼𝑀2 𝐼𝑀3 𝐼𝑀4 𝐼𝑀5<br>actions regions’ √ Informative frames with dynamic actions<br>(a) Study Case (b) Algorithm Flow<br>Motivating<br>**----- End of picture text -----**<br>


Figure 3. **(a)** Unlike existing methods often lead to repetitive event frames, our AFE representation adaptively filters out repetitive events for the same action based on the observed overlapped action regions; **(b)** Illustration of the AFE representation. 

## **3.1. The AFE Representation** 

Most event-based action recognition models [14, 39, 55] primarily rely on event frame representations [31, 56], which is compatible with off-the-shelf CNN/ViT backbones. For these models, the event stream is spatially integrated into frames with fixed event counts or time duration [57], as shown in Fig. 1 (b). However, the high temporal resolution of event data inevitably leads to a profusion of repetitive event frames displaying the same stationary objects (refer to Fig. 3 (a) blue circles). Consequently, it is arduous for such representation to depict dynamic actions. To this end, we seek to answer the following question: _Can we adaptively filter out repetitive events for stationary objects while preserving events recording the dynamic actions?_ 

Accordingly, we visualize the previous event frame representation [57] in Fig. 3 (a). A salient observation is the ‘overlapped action region’, as marked by the red square, 

which is a byproduct of the frame transformation process where consecutive actions overlap due to excessive event stacks. This ‘overlapped action region’ thus serves as a pivotal indicator for inappropriate event sampling intervals. In light of this, we propose the AFE representation. 

Specifically, to find the most appropriate dividing line of different actions based on the ‘overlapped action regions’, we adopt the classic _binary search_ and implement it _offline and recursively_ with efficient _O_ ( _logn_ ) algorithm complexity. As illustrated in Fig. 3 (b), the search algorithm can be seen as finding leaf nodes of a binary tree. To begin with, we equally divide the event stream _E_ 0[0][(root) into two sub-] streams _E_ 1[0][(node)][and] _[E]_ 1[1][(node)][and][thus][generate][their] corresponding event count images _I_ 1[0][and] _[I]_ 1[1][.][Then,][we] subtract _I_ 1[0][with] _[I]_ 1[1][to][obtain][the][difference][map.][Next,][to] measure the proportion of overlapped actions based on the difference map, we define a factor named deference rate _R_ , given by: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/cvpr_2024_ExACT_Language-guided_Conceptual_Reasoning_and_Uncertainty_Estimation_for_Event-based_Action_Recognition_and_More_images/cvpr_2024_ExACT_Language-guided_Conceptual_Reasoning_and_Uncertainty_Estimation_for_Event-based_Action_Recognition_and_More.pdf-0004-02.png)


where the _sum_ ( _._ ) and _abs_ ( _._ ) functions indicate the event counts and absolute value operations, respectively. Intuitively, the high value of _R_ indicates a high probability of stacking events recording two different actions into one frame. In this case, we need to divide the event sub-stream recursively. 

For the recursive algorithm, the boundary conditions are important. In our cases, if the difference rate _R_ is higher than the lowest sampling threshold ∆, we repeat the above division process until it’s lower than ∆ or the event count number of the sub-stream _Nsub_ is less than the minimum aggregating event count number _Nmin_ . Note that hyperparameters _Nmin_ and ∆ vary with different datasets. _More discussions about selecting Nmin and_ ∆ _refer to the Supplmat._ After the above searching process with _M_ times recursion, we finally obtain a series of fine-grained event frames _IM[T]_[with] _[ T]_[event frames (all leaf nodes).] 

## **3.2. Feature Encoding** 

With the AFE representation, the event stream is processed into a series of fine-grained event frames _IM[T]_[.][Then,][the] event encoder and text encoder from the pre-trained eventtext model of [58] are utilized to establish a unified eventtext embedding space. 

**Event Encoder.** As shown in Fig. 2, it inputs an RGB event frame _IM[i][∈]_[R] _[H][×][W][ ×]_[3] _[, i]_[=][1] _[,]_[ 2] _[, ..., T]_[of][the][spatial][size] _H × W_ and outputs the event embedding _fe[i]_[.][Given] _[ T]_[event] frames, the event encoder processes _T_ times and generates the event [ _CLS_ ] embeddings _fe ∈_ R _[T][ ×][H][e][×][W][e][×][N][e]_ . 

**Text Encoder.** It takes two different kinds of text prompts as input: **1)** the hand-crafted text prompt `‘A series of photos recording action for` [ _CLASS_ ] `.’` , where [ _CLASS_ ] represents the category name. After encoding, each word is converted into the _Dp_ -dimension word 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/cvpr_2024_ExACT_Language-guided_Conceptual_Reasoning_and_Uncertainty_Estimation_for_Event-based_Action_Recognition_and_More_images/cvpr_2024_ExACT_Language-guided_Conceptual_Reasoning_and_Uncertainty_Estimation_for_Event-based_Action_Recognition_and_More.pdf-0004-09.png)


**----- Start of picture text -----**<br>
Event Text<br>𝑇<br>𝑓𝑒 MLP MLP 𝑓𝑡 Distributional<br>Representation<br>𝑇<br>𝑓𝑒𝑝 𝑓𝑡𝑝<br>Softmax<br>Discrete<br>… Representation<br>Conceptual<br>Reasoning<br>𝑓𝑒𝑓𝑢𝑠𝑒 Weights<br>Joint Event-Text<br>Embedding Space<br>𝑁𝑒 𝜇𝑒, 𝜎𝑒 [2] 𝑁𝑡 𝜇𝑡, 𝜎𝑡2<br>Cross<br>Multiplication<br>𝑓𝑠𝑎𝑚𝑝𝑙𝑒,𝑒𝑛 𝑓𝑠𝑎𝑚𝑝𝑙𝑒,𝑡𝑛 Point<br>Multiplication<br>Conceptual Reasoning-based Fusion<br>Uncertainty Estimation<br>**----- End of picture text -----**<br>


Figure 4. **The proposed CRUE module** consists of 1) conceptual reasoning for frame fusion based on the temporal relation among events and 2) uncertainty estimation of sub-actions for both text and event embeddings utilizing distributional representation. 

embedding and formed into the final text token _Ph_ **2)** the learnable text prompt _Pl_ = [ _P_ 1 _, P_ 2 _, ..., Pn, PCLASS._ ], where _Pi, i_ = 1 _,_ 2 _, ..., nl_ , is a random initialized parameter with _Dp_ dimensions; _nl_ denotes the number of the learnable text prompts; _PCLASS_ represents the encoded word embeddings of [ _CLASS_ ] and [ _._ ] means the concatenation operation. Then, the text encoder transforms hand-crafted text prompt _Ph_ and learnable text prompt _Pl_ into corresponding text embeddings _ft[h]_[and] _[ f] t[ l]_[. We finally obtain the text [] _[CLS]_[]] embeddings _ft_ by averaging _ft[h]_[and] _[ f] t[ l]_[.] 

## **3.3. CRUE Module** 

Previous event-based action recognition methods [14, 26] fail to consider the following aspects: **1) Temporal Relation** : Unlike stationary objects, dynamic actions unfold over time. The event data’s temporal information is vital for understanding the meaning of an action. For instance, in Fig. 1, `‘Sit down’` and `‘Stand up’` involve similar sub-actions with the converse temporal occurrence, namely `‘Standing’` _→_ `‘Squatting’` _→_ `‘Sitting’` _vs._ `‘Sitting’` _→_ `‘Squatting’` _→_ `‘Standing’` , thus resulting in different semantics. **2) Semantic Uncertainty** : Actions, comprising various sub-actions, present greater semantic complexity than that of stationary objects. Take `‘Sit down’` as an example. It includes sub-actions: `‘Standing’` , `‘Squatting’` , and `‘Sitting’` . Each sub-action owns a specific meaning. Thus, using any subaction is _insufficient_ and _uncertain_ to express the meaning of entire action `‘Sit down’` . Motivated by these two aspects, we propose the CRUE module to emulate the action recognition process of humans through the proposed conceptual reasoning-based fusion and uncertainty estimation. 

**Conceptual Reasoning-based Fusion:** Unlike the simple average fusion of event frames [14, 36, 39, 55], the proposed CRUE module leverages the text embeddings to guide the event frame fusion. Specifically, as depicted in Fig. 4, given the event [ _CLS_ ] embeddings _fe_ and text [ _CLS_ ] embeddings _ft_ , a two-layer MLP network is employed for dimension projection. In this way, we can obtain the projected event embeddings _fe[p]_[and text embeddings] _[ f] t[ p]_[.][Then,] we multiply _fe[p]_[and] _[ f] t[ p]_[followed by the softmax function to] generate the conceptual reasoning weights. Next, the conceptual reasoning weights are multiplied with the original projected event embeddings _fe[p]_[to][obtain][the][fused][event] embeddings _fe[fuse]_ . Intuitively, the conceptual reasoning weights are used as semantic weights generated based on event frames’ temporal sequence for the frame fusion. 

**Uncertainty Estimation:** Semantic uncertainty refers to the obtained messages that tend to present multiple targets [19]. To model the semantic uncertainty of actions, we borrow the idea of distributional representation applied both in Natural Language Processing (NLP) [2, 8, 47] and Computer Vision (CV) [5, 41, 42]. Unlike the methods that extract features as the discrete representation, we utilize the probability distribution encoder [19] for events and text embedding. Thus, the semantic uncertainty can be quantified by the variance of the probability distribution. 

Specifically, as shown in Fig. 4, the fused event [ _CLS_ ] embeddings _fe[fuse] ∈_ R _[H][e][×][W][e][×][N][e]_ are equally split into _fe,[fuse]_ 1 and _fe,[fuse]_ 2 at the channel dimension [19]. Then, _fe,[fuse]_ 1 and _fe,[fuse]_ 2 are fed into two standard self-attention modules [46]. Next, we can predict a mean vector _µe ∈_ R _[H][e][×][W][e][×][N][e]_ and a variance vector _σe ∈_ R _[H][e][×][W][e][×][N][e]_ . Here, _µe_ and _σe_ are the estimated parameters of the multivariate Gaussian distribution _fe[fuse] ∼ Ne_ ( _µe, σe_[2][)][.][We] conduct the same operation on the text _CLS_ embeddings _ft_ to estimate its corresponding multivariate Gaussian distribution _ft[p][∼][N][t]_[(] _[µ][t][, σ] t_[2][)][.][Overall, the above operations can] be formulated as follows: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/cvpr_2024_ExACT_Language-guided_Conceptual_Reasoning_and_Uncertainty_Estimation_for_Event-based_Action_Recognition_and_More_images/cvpr_2024_ExACT_Language-guided_Conceptual_Reasoning_and_Uncertainty_Estimation_for_Event-based_Action_Recognition_and_More.pdf-0005-03.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/cvpr_2024_ExACT_Language-guided_Conceptual_Reasoning_and_Uncertainty_Estimation_for_Event-based_Action_Recognition_and_More_images/cvpr_2024_ExACT_Language-guided_Conceptual_Reasoning_and_Uncertainty_Estimation_for_Event-based_Action_Recognition_and_More.pdf-0005-04.png)


where _i_ = _e, t_ denotes the event and text embeddings respectively; _Att_ represents the self-attention module and [ _._ ] indicates the concatenation operation. 

With estimated distributional representation _Ni_ ( _µi, σi_[2][)] _[,] i_ = _t, e_ , we now quantify the semantic uncertainty of event and text embeddings. Then, we sample arbitrary discrete representation by employing the re-parameterization method [23] to ensure smooth back-propagation. That is, we first sample a random noise _δ ∼ N_ (0 _, I_ ) from the standard normal distribution rather than sample directly from _fi ∼ Ni_ ( _µi, σi_[2][)] _[, i]_[ =] _[ t, e]_[.][Next, we obtain the sampled dis-] crete embeddings _fsample,i[n]_[by] _[ f] sample,i[ n]_[=] _[ µ][i]_[ +] _[ δσ][i]_[, where] 

_n_ = 1 _,_ 2 _, ..., N_ and _N_ is a hyper-parameter denoting the number of sampled discrete embeddings. (More discussion about _N_ refer to Sec. 4.3.) The obtained _fsample,i[n]_[observes] the estimated distribution _Ni_ , and thus it can be used for estimating semantic uncertainty. 

However, the above random sampling process increases the complexity of training, especially given the spatial sparsity of event data. To accelerate model convergence, we introduce the _Smooth L1_ loss [15] between the normalized _CLS_ embeddings _fi, i_ = _t, e_ and the sampled embeddings _fsample,i[n][, i]_[ =] _[ t, e]_[:] 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/cvpr_2024_ExACT_Language-guided_Conceptual_Reasoning_and_Uncertainty_Estimation_for_Event-based_Action_Recognition_and_More_images/cvpr_2024_ExACT_Language-guided_Conceptual_Reasoning_and_Uncertainty_Estimation_for_Event-based_Action_Recognition_and_More.pdf-0005-09.png)


where _n_ = 1 _,_ 2 _, ..., N_ and _N_ is a hyper-parameter denoting the number of sampled discrete embeddings, _mean_ ( _._ ) and _std_ ( _._ ) denote the mean and variance of the input embeddings. Besides, to reduce the semantic uncertainty of distributional representation, we introduce regularization loss: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/cvpr_2024_ExACT_Language-guided_Conceptual_Reasoning_and_Uncertainty_Estimation_for_Event-based_Action_Recognition_and_More_images/cvpr_2024_ExACT_Language-guided_Conceptual_Reasoning_and_Uncertainty_Estimation_for_Event-based_Action_Recognition_and_More.pdf-0005-11.png)


where _sum_ ( _._ ) and _abs_ ( _._ ) indicate the summation of input embeddings and absolute value operation, respectively. Note that the experiment results show the final _Lreg_ is higher than zero when the model converges. This indicates the model doesn’t degenerate from distributional representation into discrete representation as the variances are greater than zero. 

## **3.4. Training Objectives** 

To establish a joint event-text representation space for action recognition, we utilize the contrastive loss _L_ ( _fb_[1] _[, f] b_[ 2][)] between two modal embeddings _fb_[1][and] _[ f] b_[ 2][as follows:] 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/cvpr_2024_ExACT_Language-guided_Conceptual_Reasoning_and_Uncertainty_Estimation_for_Event-based_Action_Recognition_and_More_images/cvpr_2024_ExACT_Language-guided_Conceptual_Reasoning_and_Uncertainty_Estimation_for_Event-based_Action_Recognition_and_More.pdf-0005-15.png)


where _τ_ is the temperature coefficient, _B_ represents the size of the mini-batch, _b_ and _b_ denote the _b_ -th and the _b_ -th data among the mini-batch. 

We calculate the contrastive loss among all sampled event embeddings _fsample,e[n]_[and text embeddings] _[ f] sample,t[ n]_[.] Finally, the whole training objective is composed of the contrastive loss, the _Smooth L1_ loss, and regularization loss combined with different rate hyperparameters: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/cvpr_2024_ExACT_Language-guided_Conceptual_Reasoning_and_Uncertainty_Estimation_for_Event-based_Action_Recognition_and_More_images/cvpr_2024_ExACT_Language-guided_Conceptual_Reasoning_and_Uncertainty_Estimation_for_Event-based_Action_Recognition_and_More.pdf-0005-18.png)


where we set the default values of _α_ , _β_ and _θ_ to 1 after considering their numerical range. 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/cvpr_2024_ExACT_Language-guided_Conceptual_Reasoning_and_Uncertainty_Estimation_for_Event-based_Action_Recognition_and_More_images/cvpr_2024_ExACT_Language-guided_Conceptual_Reasoning_and_Uncertainty_Estimation_for_Event-based_Action_Recognition_and_More.pdf-0006-00.png)


**----- Start of picture text -----**<br>
t<br>sit down: Human action for sit down refers to a person<br>lowering their body to rest in a seated position.<br>t<br>catch and throw a ball: Human action for catching and<br>throwing a ball involves coordination of motor skills,<br>timing, and visual perception.<br>t<br>vomit: Human action for vomit refers to the act of<br>forcefully expelling stomach contents through the mouth.<br>t<br>handshake: A handshake is a human action symbolizing<br>agreement, friendship, respect, or conclusion of a deal.<br>Only<br>Body-motion<br>Interaction<br>Human-object<br>Health-care  Monitoring<br>Interaction<br>Human-human<br>**----- End of picture text -----**<br>


Figure 5. Examples of our SeAct dataset. 

## **3.5. SeAct Dataset** 

Since the category-level labels provided in the previous event action dataset [1, 4, 34, 48] only use several words to describe each action, they fail to stimulate the ability of our ExACT framework to process complex language information. To this end, we propose the **first** semanticabundant SeAct dataset for event-text action recognition, where the detailed caption-level label of each action is provided. SeAct is collected with a DAVIS346 event camera whose resolution is 346 × 260. It contains 58 actions under four themes, as presented in Fig. 5. Each action is accompanied by an action caption of less than 30 words generated by GPT-4 [38] to enrich the semantic space of the original action labels. We split 80% and 20% of each category for training and testing (validating), respectively. We believe our SeAct dataset will be a better evaluation platform for event-text action recognition and inspire more relevant research in the future. _Please refer to the Supplmat. for more dataset introduction._ 

## **4. Experiments** 

## **4.1. Dataset and Experimental Settings** 

**Dataset** In this work, four datasets are adopted for the evaluation of our proposed model, including PAF [34], HARDVS [48], DVS128 Gesture [1] and our newly proposed SeAct. PAF [34], also named Action Recognition, is a dataset recorded indoors, containing ten action categories with 450 recordings. HARDVS [48] is a recently released dataset for event-based action recognition, currently having the largest action classes, namely 107,646 recordings for 300 action categories. Both of the above two datasets have an average time duration of 5 seconds with 346 × 260 resolution [48]. DVS128 Gesture [1] is collected using a DVS128 camera with 128 × 128 resolution, dividing into 11 hand and arm gestures. _Refer to Sec. 3.5 for the introduction of our SeAct dataset._ 

**Experimental Settings** In the AFE representation, the low- 

|Dataset<br>Model|Accuracy (%)|
|---|---|
||Top-1<br>Top-5|
|PAF<br>HMAX SNN [53]<br>STCA [16]<br>Motion SNN [27]<br>MST [51]<br>Swin-T (BN) [51]<br>EV-ACT [14]<br>**ExACT (Ours)**|55.00<br>-<br>71.20<br>78.10<br>-<br>88.21<br>-<br>90.14<br>-<br>92.60<br>-<br>**94.83**<br>**98.28**|
|HARDVS<br>X3D [12]<br>SlowFast [13]<br>ACTION-Net [50]<br>R2Plus1D [45]<br>ResNet18 [17]<br>TAM [28]<br>C3D [44]<br>ESTF [48]<br>Video-SwinTrans [29]<br>TSM [26]<br>**ExACT (Ours)**|45.82<br>52.33<br>46.54<br>54.76<br>46.85<br>56.19<br>49.06<br>56.43<br>49.20<br>56.09<br>50.41<br>57.99<br>50.52<br>56.14<br>51.22<br>57.53<br>51.91<br>59.11<br>52.63<br>60.56<br>**90.10**<br>**96.69**|
|DVS128 Gesture<br>Time-surfaces [32]<br>SNN eRBP [20]<br>Slayer [40]<br>DECOLLE [21]<br>EvT [39]<br>TBR [18]<br>EventTransAct [9]<br>**ExACT (Ours)**|90.62<br>-<br>92.70<br>-<br>93.64<br>-<br>95.54<br>-<br>96.20<br>-<br>97.73<br>-<br>97.92<br>-<br>**98.86**<br>**98.86**|
|SeAct<br>EventTransAct [9]<br>EvT [39]<br>**ExACT-category**<br>**ExACT-caption**|57.81<br>64.22<br>61.30<br>67.81<br>66.07<br>70.54<br>**67.24**<br>**75.00**|



Table 1. An overall comparison with SoTA models for event-based action recognition task on the PAF, HARDVS, DVS128 Gesture, and our proposed SeAct dataset. The best scores are in bold, and the second scores are underlined. ‘ExACT-category’ and ‘ExACTcaption’ represent that ExACT is trained with category-level and caption-level labels, respectively. 

est sampling rate ∆ is set as 50%, 40%, and 40% while the minimum aggregating event number _Nmin_ is chosen at 100,000, 150,000, and 100,000 on the PAF, HARDVS our SeAct datasets, respectively. The event encoder and text encoder of the event-image-text model ECLIP [58] are utilized for feature encoding. The number of sampled discrete embeddings _N_ is set to 5 based on the hyperparameter search. The initial learning rates are set to 1 _e −_ 5 with Adam optimizer [22] and weight decay equal to 2 _e −_ 4. CosineAnnealingLR [30] learning rate schedule is employed with a minimum learning rate of 1e-6. Our model is trained for 100 epochs for PAF and SeAct datasets, and 25 epochs for HARDVS. _Please refer to the Supplmat. for additional experimental settings._ 

## **4.2. Comparison with SOTA Methods** 

As shown in Tab. 1, our proposed ExACT demonstrates superior performance on the PAF and HARDVS datasets. Specifically, ExACT brings +2.23% improvements on the PAF dataset with ten classes compared with SOTA re- 

|Sample strategy<br>Frame number<br>Aggregated method|Accuracy|
|---|---|
||Top-1<br>Top-5|
|Event histogram<br>3122<br>every 80,000 events<br>2720<br>every 90,000 events<br>2405<br>every 100,000 events|89.29<br>91.07<br>94.21<br>97.14<br>93.10<br>95.24|
|Event voxel [10]<br>3122<br>every 80,000 events<br>2720<br>every 90,000 events<br>2405<br>every 100,000 events|88.88<br>90.46<br>92.45<br>94.89<br>90.85<br>92.31|
|TBR [18]<br>2758<br>every 2000 ms events|92.79<br>95.16|
|AFE (Ours)<br>2894<br>adaptive events number|**94.83**<br>**98.28**|



Table 2. The comparison of AFE representation with previous event frame representation. 

|Method|Accuracy|
|---|---|
||Top-1<br>∆<br>Top-5<br>∆|
|Contrastive<br>+ CR<br>+ CR, UE|92.86<br>-<br>94.64<br>-<br>93.64<br>+0.78<br>96.43<br>+1.79<br>**94.83**<br>+1.97<br>**98.28**<br>+3.64|



Table 3. The ablation study of the CRUE module, where Contrastive, CR, and UE denote the contrastive learning loss function, proposed conceptual reasoning-based fusion and uncertainty estimation, respectively. 

|Method|Accuracy|
|---|---|
||Top-1<br>∆<br>Top-5<br>∆|
|Sum<br>Mean Pooling<br>CR|89.14<br>-<br>90.97<br>-<br>92.52<br>+3.38<br>95.04<br>+5.07<br>**94.83**<br>+5.69<br>**98.28**<br>+8.31|



Table 4. Impact of CR operation proposed in the CRUE module. 

sults. Notably, ExACT brings remarkable +37.47% improvements on the HARDVS with 300 classes, showcasing ExACT’s excellent potential in classifying complex and diverse actions. Moreover, upon evaluation using our proposed SeAct dataset, ExACT exhibits a 67.24% Top-1 and 75.00% Top-5 recognition accuracy in real-world scenarios involving 58 dynamic actions with caption-level labels. These results demonstrate the effectiveness of our ExACT framework in action recognition. 

## **4.3. Ablation Studies** 

In this section, we ablate the key components of ExACT, training objectives, and important hyper-parameters to explore their effectiveness. Unless otherwise stated, experiments are performed on the PAF dataset. 

**Effectiveness of the AFE representation.** Tab. 2 shows that, with a comparatively lower sampled number of 2816, our AFE representation obtains the best accuracy of 94.83%. This result indicates that our AFE representation can achieve better performance by filtering out the repetitive frames portraying the same action. 

**CRUE module** _**vs.**_ **Contrastive learning.** As shown in Tab. 3, the baseline model is trained using the contrastive learning loss, which is widely adopted in previous methods [52, 58]. Results indicate that the Concep- 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/cvpr_2024_ExACT_Language-guided_Conceptual_Reasoning_and_Uncertainty_Estimation_for_Event-based_Action_Recognition_and_More_images/cvpr_2024_ExACT_Language-guided_Conceptual_Reasoning_and_Uncertainty_Estimation_for_Event-based_Action_Recognition_and_More.pdf-0007-11.png)


**----- Start of picture text -----**<br>
Accuracy<br>Training Method<br>Top-1 ∆ Top-5 ∆<br>Lcontrastive 92.86 - 94.64 -<br>+  Lreg 93.96 +1.10 95.82 +1.18<br>+  LsmoothL 1 94.41 +1.55 97.89 +3.25<br>+  Lreg ,  LsmoothL 1 94.83 +1.97 98.28 +3.46<br>Table 5. Impact of different training objectives.<br>100<br>98.28 98.28<br>98<br>95.83<br>96<br>94.83 94.83<br>94.34<br>93.94<br>94<br>92.43<br>92 Top-1<br>Top-5<br>90<br>2 4 6 8<br>Number of sampled discrete embeddings, N<br>）<br>Accuracy (%<br>**----- End of picture text -----**<br>


Figure 6. Impact of different numbers of sampled discrete embeddings proposed in the CRUE Module. 

tual Reasoning-based fusion (CR) and CR with Uncertainty Estimation (UE) lead to an increase in accuracy of +0.78% and +1.97%, respectively. This implies that the CRUE module enhances the model’s ability to comprehend actions by conceptual reasoning-based fusing event frames based on their temporal relations and estimating the uncertainty of action semantics during training, thus achieving better performance than simply employing contrastive learning. 

**Conceptual Reasoning-based fusion (CR)** _**vs.**_ **Other frame fusion methods** To evaluate the effectiveness of the CR. We compare the CR with two other designs, namely, the sum and average pooling of all event frames, as shown in Tab. 3. The results demonstrate the effectiveness of our proposed CR, as it improves recognition accuracy by +5.69% and +2.31% when compared to the sum operation and the mean pooling operation, respectively. 

**Performance comparison of different training objectives.** Tab. 5 shows the impact of different training objectives on model performance. Model trained using the contrastive learning loss _Lcontrastive_ (Eq. 6) exhibits the lowest performance compared to other combinations of training objectives. Both the _Lreg_ (Eq. 5) and _LsmoothL_ 1 (Eq. 4) enhance the model performance, bringing +1.10% and +1.55% accuracy improvements respectively. The combination of all training objectives brings the largest performance improvement of +1.97%, which further validates the effectiveness of the proposed CRUE module. 

**Impact of the different number of point sampling.** As shown in Fig. 6, we explore the effect of the number of sampled discrete embeddings _N_ . We find that as _N_ increases from 1 to 4, the accuracy increases from 92.43% to 94.83%. When _N_ increases from 4 to 8, the accuracy re- 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/cvpr_2024_ExACT_Language-guided_Conceptual_Reasoning_and_Uncertainty_Estimation_for_Event-based_Action_Recognition_and_More_images/cvpr_2024_ExACT_Language-guided_Conceptual_Reasoning_and_Uncertainty_Estimation_for_Event-based_Action_Recognition_and_More.pdf-0008-00.png)


**----- Start of picture text -----**<br>
(a) (b) (c)<br>In Place Hip Kick Hip Kick Jump Standing Right Leg Lift<br>Alternate Front  Hip Up Kick Jump Standing Left Leg Lift<br>Kick Jump<br>**----- End of picture text -----**<br>


Figure 7. t-SNE visualization of event embeddings on the HARDVS dataset. (a) Before training; (b) Training without CRUE module; (c) Training with CRUE module. 

mains constant, indicating that increasing _N_ yields diminishing performance accuracy improvement. Intuitively, this comes from the fact that distributional representation introduces disturbance during training while more sampled discrete embeddings mitigate this disturbance. Consequently, we set the hyperparameter _N_ as 5 during training. 

**The t-SNE visualization of event embeddings for the CRUE module.** We select 144 event examples belonging to six categories from the HARDVS dataset. These categories include a simple pair with different semantics, as well as two challenging pairs with similar semantics, namely `‘Hip Kick Jump’` _vs._ `‘Hip Up Kick Jump’` and `‘Standing Right Leg Lift’` _vs._ `‘"Standing Left Leg Lift"’` . Fig. 7 displays the alteration of event embedding distribution before training and training with/without the CRUE module. Comparing those hard pairs in Fig. 7 (b) and (c), we can find that event embeddings in different categories are widely distributed while our proposed CRUE module advances in distinguishing those hard pairs with similar semantic meanings. Intuitively, visualization results prove that the CRUE module enhances the model’s performance by expressing the uncertainty of semantics, especially for those with similar semantics. _See more results in Supplmat._ 

## **4.4. Extension to Other Tasks** 

In this section, we transfer our EXACT model to both textto-event and event-to-text retrieval tasks. Retrieval refers to searching and retrieving required data that matches a given query. The query can be of different modalities like events or texts. It has several practical applications, such as matching a real-world abnormal action with its corresponding event data in surveillance scenarios. For action retrieval using EXACT, we feed the query and events/texts into corresponding encoders to obtain their embeddings. Next, we calculate the similarity score between the query embedding and event/text embeddings, then select those events/texts with the highest similarity scores as the output. All retrieval experiments are conducted on our SeAct dataset. _Refer to Supplmat. for more retrieving results._ 

**Text-to-event action retrieval** In Fig. 8 (a), we present the Top-3 retrieved event streams utilizing the action 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/cvpr_2024_ExACT_Language-guided_Conceptual_Reasoning_and_Uncertainty_Estimation_for_Event-based_Action_Recognition_and_More_images/cvpr_2024_ExACT_Language-guided_Conceptual_Reasoning_and_Uncertainty_Estimation_for_Event-based_Action_Recognition_and_More.pdf-0008-07.png)


**----- Start of picture text -----**<br>
Text query:  Falling down: Falling down refers to the involuntary<br>action of losing balance and suddenly collapsing to the ground.<br>(1) t<br>(2)<br>t<br>(3)<br>t<br>(a) Text-to-event<br>Event<br>Query:  t<br>(1)   Hurdle start: Human action for hurdle start refers to the<br>physical movements a person undertakes to begin a hurdle race .<br>(2)   Long jump: Human action for long jump involves running,<br>leaping, and landing to achieve maximum horizontal distance.<br>(3)   Running: Running is a human action involving swift movement<br>on foot where both feet leave the ground simultaneously.<br>(b) Event-to-text<br>**----- End of picture text -----**<br>


Figure 8. Action retrieval results. 

`‘Falling down’` with captions. All retrieved event data exhibit a high degree of similarity to the input text query, providing further evidence of EXACT’s effectiveness. 

**Event-to-text action retrieval** In Fig. 8 (b), we display the Top-3 retrieved text captions queried by the event stream recording action `‘Hurdle start’` . The retrieved captions include actions of `‘Hurdle start’` , `‘Long jump’` and `‘Running’` , which share several sub-actions of similar semantics, proving the effectiveness of ExACT. 

## **5. Conclusion and Future Work** 

We presented the ExACT, as the first exploration utilizing language guidance for event-based action recognition. We proposed a CRUE module to simulate the action recognition of human beings, especially focusing on conceptual reasoning of the temporal relations among event frames and estimating the uncertainty of actions. Besides, we proposed the AFE representation, which adaptively eliminated repetitive events to generate detailed event frames for dynamic actions. To evaluate models’ understanding of the complex semantics of actions involving multiple sub-actions with different semantics, we presented the SeAct dataset with semantic-abundant action captions as the first benchmark for event-text action recognition. Our ExACT framework achieved SOTA results on the PAF and HARDVA datasets and achieved plausible performance on our SeAct dataset. Furthermore, we extended ExACT to event-text retrieval tasks, proving its flexible transferability. 

**Future Work:** In the future, we will enhance the conceptual reasoning and uncertainty estimation module in various event vision tasks and experiments on larger event action datasets with semantic-abundant caption-level labels. 

**Acknowledgement** This paper is supported by the National Natural Science Foundation of China (NSF) under Grant No. NSFC22FYT45 and the Guangzhou City, University and Enterprise Joint Fund under Grant No.SL2022A03J01278. 

## **References** 

- [1] Arnon Amir, Brian Taba, David Berg, Timothy Melano, Jeffrey McKinstry, Carmelo Di Nolfo, Tapan Nayak, Alexander Andreopoulos, Guillaume Garreau, Marcela Mendoza, et al. A low power, fully event-based gesture recognition system. In _Proceedings of the IEEE conference on computer vision and pattern recognition_ , pages 7243–7252, 2017. 1, 6 

- [2] Ben Athiwaratkun and Andrew Gordon Wilson. Multimodal word distributions. _arXiv preprint arXiv:1704.08424_ , 2017. 5 

- [3] Djamila Romaissa Beddiar, Brahim Nini, Mohammad Sabokrou, and Abdenour Hadid. Vision-based human activity recognition: a survey. _Multimedia Tools and Applications_ , 79(41-42):30509–30555, 2020. 1 

- [4] Yin Bi, Aaron Chadha, Alhabib Abbas, Eirina Bourtsoulatze, and Yiannis Andreopoulos. Graph-based spatio-temporal feature learning for neuromorphic vision sensing. _IEEE Transactions on Image Processing_ , 29:9084–9098, 2020. 6 

- [5] Jie Chang, Zhonghao Lan, Changmao Cheng, and Yichen Wei. Data uncertainty learning in face recognition. In _Proceedings of the IEEE/CVF conference on computer vision and pattern recognition_ , pages 5710–5719, 2020. 5 

- [6] Fei-Long Chen, Du-Zhen Zhang, Ming-Lun Han, Xiu-Yi Chen, Jing Shi, Shuang Xu, and Bo Xu. Vlp: A survey on vision-language pre-training. _Machine Intelligence Research_ , 20(1):38–56, 2023. 2, 3 

- [7] Hoonhee Cho, Hyeonseong Kim, Yujeong Chae, and KukJin Yoon. Label-free event-based object recognition via joint learning with image reconstruction from events. In _Proceedings of the IEEE/CVF International Conference on Computer Vision_ , pages 19866–19877, 2023. 3 

- [8] Shib Dasgupta, Michael Boratko, Dongxu Zhang, Luke Vilnis, Xiang Li, and Andrew McCallum. Improving local identifiability in probabilistic box embeddings. _Advances in Neural Information Processing Systems_ , 33:182–192, 2020. 5 

- [9] Tristan de Blegiers, Ishan Rajendrakumar Dave, Adeel Yousaf, and Mubarak Shah. Eventtransact: A video transformer-based framework for event-camera based action recognition. In _2023 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS)_ , pages 1–7. IEEE, 2023. 6 

- [10] Yongjian Deng, Hao Chen, Hai Liu, and Youfu Li. A voxel graph cnn for object classification with event cameras. In _Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition_ , pages 1172–1181, 2022. 7 

- [11] Yifan Du, Zikang Liu, Junyi Li, and Wayne Xin Zhao. A survey of vision-language pre-trained models. _arXiv preprint arXiv:2202.10936_ , 2022. 2, 3 

- [12] Christoph Feichtenhofer. X3d: Expanding architectures for efficient video recognition. In _Proceedings of the IEEE/CVF conference on computer vision and pattern recognition_ , pages 203–213, 2020. 6 

- [13] Christoph Feichtenhofer, Haoqi Fan, Jitendra Malik, and Kaiming He. Slowfast networks for video recognition. In _Proceedings of the IEEE/CVF international conference on computer vision_ , pages 6202–6211, 2019. 6 

- [14] Yue Gao, Jiaxuan Lu, Siqi Li, Nan Ma, Shaoyi Du, Yipeng Li, and Qionghai Dai. Action recognition and benchmark using event cameras. _IEEE Transactions on Pattern Analysis and Machine Intelligence_ , 2023. 1, 2, 3, 4, 5, 6 

- [15] Ross Girshick. Fast r-cnn. In _Proceedings of the IEEE international conference on computer vision_ , pages 1440–1448, 2015. 5 

- [16] Pengjie Gu, Rong Xiao, Gang Pan, and Huajin Tang. Stca: Spatio-temporal credit assignment with delayed feedback in deep spiking neural networks. In _IJCAI_ , pages 1366–1372, 2019. 6 

- [17] Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun. Deep residual learning for image recognition. In _Proceedings of the IEEE conference on computer vision and pattern recognition_ , pages 770–778, 2016. 6 

- [18] Simone Undri Innocenti, Federico Becattini, Federico Pernici, and Alberto Del Bimbo. Temporal binary representation for event-based action recognition. In _2020 25th International Conference on Pattern Recognition (ICPR)_ , pages 10426–10432. IEEE, 2021. 6, 7 

- [19] Yatai Ji, Junjie Wang, Yuan Gong, Lin Zhang, Yanru Zhu, Hongfa Wang, Jiaxing Zhang, Tetsuya Sakai, and Yujiu Yang. Map: Multimodal uncertainty-aware vision-language pre-training model. In _Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition_ , pages 23262–23271, 2023. 2, 5 

- [20] Jacques Kaiser, Alexander Friedrich, J Tieck, Daniel Reichard, Arne Roennau, Emre Neftci, and R¨udiger Dillmann. Embodied neuromorphic vision with event-driven random backpropagation. _arXiv preprint arXiv:1904.04805_ , 2019. 6 

- [21] Jacques Kaiser, Hesham Mostafa, and Emre Neftci. Synaptic plasticity dynamics for deep continuous local learning (decolle). _Frontiers in Neuroscience_ , 14:515306, 2020. 6 

- [22] Diederik P Kingma and Jimmy Ba. Adam: A method for stochastic optimization. _arXiv preprint arXiv:1412.6980_ , 2014. 6 

- [23] Diederik P Kingma and Max Welling. Auto-encoding variational bayes. _arXiv preprint arXiv:1312.6114_ , 2013. 5 

- [24] Yu Kong and Yun Fu. Human action recognition and prediction: A survey. _International Journal of Computer Vision_ , 130(5):1366–1401, 2022. 1 

- [25] Athanasios Lentzas and Dimitris Vrakas. Non-intrusive human activity recognition and abnormal behavior detection on elderly people: A review. _Artificial Intelligence Review_ , 53 (3):1975–2021, 2020. 1 

- [26] Ji Lin, Chuang Gan, and Song Han. Tsm: Temporal shift module for efficient video understanding. In _Proceedings of the IEEE/CVF international conference on computer vision_ , pages 7083–7093, 2019. 2, 4, 6 

- [27] Qianhui Liu, Dong Xing, Huajin Tang, De Ma, and Gang Pan. Event-based action recognition using motion information and spiking neural networks. In _IJCAI_ , pages 1743– 1749, 2021. 1, 2, 6 

- [28] Zhaoyang Liu, Limin Wang, Wayne Wu, Chen Qian, and Tong Lu. Tam: Temporal adaptive module for video recognition. In _Proceedings of the IEEE/CVF international conference on computer vision_ , pages 13708–13718, 2021. 6 

- [29] Ze Liu, Jia Ning, Yue Cao, Yixuan Wei, Zheng Zhang, Stephen Lin, and Han Hu. Video swin transformer. In _Proceedings of the IEEE/CVF conference on computer vision and pattern recognition_ , pages 3202–3211, 2022. 6 

- [30] Ilya Loshchilov and Frank Hutter. Sgdr: Stochastic gradient descent with warm restarts. _arXiv preprint arXiv:1608.03983_ , 2016. 6 

- [31] Jacques Manderscheid, Amos Sironi, Nicolas Bourdis, Davide Migliore, and Vincent Lepetit. Speed invariant time surface for learning to detect corner points with event-based cameras. In _Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition_ , pages 10245– 10254, 2019. 3 

- [32] Jean-Matthieu Maro, Sio-Hoi Ieng, and Ryad Benosman. Event-based gesture recognition with dynamic background suppression using smartphone computational capabilities. _Frontiers in neuroscience_ , 14:501775, 2020. 6 

- [33] Christoforos Mavrogiannis, Francesca Baldini, Allan Wang, Dapeng Zhao, Pete Trautman, Aaron Steinfeld, and Jean Oh. Core challenges of social robot navigation: A survey. _ACM Transactions on Human-Robot Interaction_ , 12(3):1– 39, 2023. 1 

- [34] Shu Miao, Guang Chen, Xiangyu Ning, Yang Zi, Kejia Ren, Zhenshan Bing, and Alois Knoll. Neuromorphic vision datasets for pedestrian detection, action recognition, and fall detection. _Frontiers in neurorobotics_ , 13:38, 2019. 2, 6 

- [35] Preksha Pareek and Ankit Thakkar. A survey on video-based human action recognition: recent updates, datasets, challenges, and applications. _Artificial Intelligence Review_ , 54: 2259–2322, 2021. 1 

- [36] Chiara Plizzari, Mirco Planamente, Gabriele Goletto, Marco Cannici, Emanuele Gusso, Matteo Matteucci, and Barbara Caputo. E2 (go) motion: Motion augmented event stream for egocentric action recognition. In _Proceedings of the IEEE/CVF conference on computer vision and pattern recognition_ , pages 19935–19947, 2022. 1, 2, 5 

- [37] Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, et al. Learning transferable visual models from natural language supervision. In _International conference on machine learning_ , pages 8748–8763. PMLR, 2021. 2 

- [38] Partha Pratim Ray. Chatgpt: A comprehensive review on background, applications, key challenges, bias, ethics, limitations and future scope. _Internet of Things and CyberPhysical Systems_ , 2023. 6 

- [39] Alberto Sabater, Luis Montesano, and Ana C Murillo. Event transformer. a sparse-aware solution for efficient event data processing. In _Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition_ , pages 2677– 2686, 2022. 1, 2, 3, 5, 6 

- [40] Sumit B Shrestha and Garrick Orchard. Slayer: Spike layer error reassignment in time. _Advances in neural information processing systems_ , 31, 2018. 6 

- [41] Yukun Su, Guosheng Lin, and Qingyao Wu. Self-supervised 3d skeleton action representation learning with motion consistency and continuity. In _Proceedings of the IEEE/CVF_ 

   - _international conference on computer vision_ , pages 13328– 13338, 2021. 2, 5 

- [42] Jennifer J Sun, Jiaping Zhao, Liang-Chieh Chen, Florian Schroff, Hartwig Adam, and Ting Liu. View-invariant probabilistic embedding for human pose. In _Computer Vision– ECCV 2020: 16th European Conference, Glasgow, UK, August 23–28, 2020, Proceedings, Part V 16_ , pages 53–70. Springer, 2020. 2, 5 

- [43] Zehua Sun, Qiuhong Ke, Hossein Rahmani, Mohammed Bennamoun, Gang Wang, and Jun Liu. Human action recognition from various data modalities: A review. _IEEE transactions on pattern analysis and machine intelligence_ , 2022. 1 

- [44] Du Tran, Lubomir Bourdev, Rob Fergus, Lorenzo Torresani, and Manohar Paluri. Learning spatiotemporal features with 3d convolutional networks. In _Proceedings of the IEEE international conference on computer vision_ , pages 4489–4497, 2015. 6 

- [45] Du Tran, Heng Wang, Lorenzo Torresani, Jamie Ray, Yann LeCun, and Manohar Paluri. A closer look at spatiotemporal convolutions for action recognition. In _Proceedings of the IEEE conference on Computer Vision and Pattern Recognition_ , pages 6450–6459, 2018. 6 

- [46] Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez, Łukasz Kaiser, and Illia Polosukhin. Attention is all you need. _Advances in neural information processing systems_ , 30, 2017. 5 

- [47] Luke Vilnis and Andrew McCallum. Word representations via gaussian embedding. _arXiv preprint arXiv:1412.6623_ , 2014. 5 

- [48] Xiao Wang, Zongzhen Wu, Bo Jiang, Zhimin Bao, Lin Zhu, Guoqi Li, Yaowei Wang, and Yonghong Tian. Hardvs: Revisiting human activity recognition with dynamic vision sensors. _arXiv preprint arXiv:2211.09648_ , 2022. 2, 6 

- [49] Xiao Wang, Zongzhen Wu, Yao Rong, Lin Zhu, Bo Jiang, Jin Tang, and Yonghong Tian. Sstformer: Bridging spiking neural network and memory support transformer for frameevent based recognition. _arXiv preprint arXiv:2308.04369_ , 2023. 1, 2 

- [50] Zhengwei Wang, Qi She, and Aljosa Smolic. Action-net: Multipath excitation for action recognition. In _Proceedings of the IEEE/CVF conference on computer vision and pattern recognition_ , pages 13214–13223, 2021. 6 

- [51] Ziqing Wang, Yuetong Fang, Jiahang Cao, Qiang Zhang, Zhongrui Wang, and Renjing Xu. Masked spiking transformer. In _Proceedings of the IEEE/CVF International Conference on Computer Vision_ , pages 1761–1771, 2023. 6 

- [52] Ziyi Wu, Xudong Liu, and Igor Gilitschenski. Eventclip: Adapting clip for event-based object recognition. _arXiv preprint arXiv:2306.06354_ , 2023. 3, 7 

- [53] Rong Xiao, Huajin Tang, Yuhao Ma, Rui Yan, and Garrick Orchard. An event-driven categorization model for aer image sensors using multispike encoding and learning. _IEEE transactions on neural networks and learning systems_ , 31 (9):3649–3657, 2019. 6 

- [54] Bochen Xie, Yongjian Deng, Zhanpeng Shao, Hai Liu, and Youfu Li. Vmv-gcn: Volumetric multi-view based graph cnn 

for event stream classification. _IEEE Robotics and Automation Letters_ , 7(2):1976–1983, 2022. 1, 2 

- [55] Bochen Xie, Yongjian Deng, Zhanpeng Shao, Hai Liu, Qingsong Xu, and Youfu Li. Event voxel set transformer for spatiotemporal representation learning on event streams. _arXiv preprint arXiv:2303.03856_ , 2023. 1, 2, 3, 5 

- [56] Fang Xu, Lei Yu, Bishan Wang, Wen Yang, Gui-Song Xia, Xu Jia, Zhendong Qiao, and Jianzhuang Liu. Motion deblurring with real events. In _Proceedings of the IEEE/CVF International Conference on Computer Vision_ , pages 2583–2592, 2021. 3 

- [57] Xu Zheng, Yexin Liu, Yunfan Lu, Tongyan Hua, Tianbo Pan, Weiming Zhang, Dacheng Tao, and Lin Wang. Deep learning for event-based vision: A comprehensive survey and benchmarks. _arXiv preprint arXiv:2302.08890_ , 2023. 3 

- [58] Jiazhou Zhou, Xu Zheng, Yuanhuiyi Lyu, and Lin Wang. E- clip: Towards label-efficient event-based open-world understanding by clip. _arXiv preprint arXiv:2308.03135_ , 2023. 3, 4, 6, 7 

