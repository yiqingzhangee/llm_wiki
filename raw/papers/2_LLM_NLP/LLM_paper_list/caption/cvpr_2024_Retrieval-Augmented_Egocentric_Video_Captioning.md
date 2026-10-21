## **Retrieval-Augmented Egocentric Video Captioning** 

Jilan Xu[1] _[,]_[5] Yifei Huang[2] _[,]_[5] Junlin Hou[1] Guo Chen[3] _[,]_[5] Yuejie Zhang[1][*] Rui Feng[1] Weidi Xie[4] _[,]_[5] 1Fudan University 2The University of Tokyo 3Nanjing University 4Shanghai Jiao Tong University 5Shanghai AI Laboratory 

## **Abstract** 

_Understanding human actions from videos of first-person view poses significant challenges. Most prior approaches explore representation learning on egocentric videos only, while overlooking the potential benefit of exploiting existing large-scale third-person videos. In this paper, (1) we develop EgoInstructor, a retrieval-augmented multimodal captioning model that automatically retrieves semantically relevant third-person instructional videos to enhance the video captioning of egocentric videos. (2) For training the cross-view retrieval module, we devise an automatic pipeline to discover ego-exo video pairs from distinct largescale egocentric and exocentric datasets. (3) We train the cross-view retrieval module with a novel EgoExoNCE loss that pulls egocentric and exocentric video features closer by aligning them to shared text features that describe similar actions. (4) Through extensive experiments, our cross-view retrieval module demonstrates superior performance across seven benchmarks. Regarding egocentric video captioning, EgoInstructor exhibits significant improvements by leveraging third-person videos as references._ 

## **1. Introduction** 

Recently, egocentric video understanding is getting increasing attention in the vision community, for example, on action recognition [5, 18, 40, 59], detection [66, 77], videotext retrieval [43, 78], grounding [14, 43, 54] and gaze estimation [35, 39]. As egocentric videos are normally recorded from a “first-person” view, reflecting activities engaging with the environment, it plays a paramount role in deploying vision models into real-world scenarios, such as robotics [33] and augmented reality [47]. 

As preliminary baseline solutions for egocentric video understanding, early works have directly trained models on the egocentric videos [18, 39, 40, 73], however, such approaches are limited by dataset scale and have overlooked the benefit of exploiting the large corpus of third-person (ex- 

> *Corresponding author 


![](raw/papers/2_LLM_NLP/LLM_paper_list/caption/cvpr_2024_Retrieval-Augmented_Egocentric_Video_Captioning_images/cvpr_2024_Retrieval-Augmented_Egocentric_Video_Captioning.pdf-0001-09.png)


**----- Start of picture text -----**<br>
Cross-view Pretraining from Pseudo Paired Ego-Exo Videos<br>#C C cuts the cucumber<br>After splitting, let’s slice<br>the cucumber thinly<br>Cross-view Retriever<br>Toast the buttered side of the bread<br>Multimodal #C C flips the<br>Generator bread and toasts<br>the other side<br>Prompt : Describe the<br>action in the video Spread the garlic butter on the bread<br>Retrieval-augmented Egocentric Video Captioning<br>**----- End of picture text -----**<br>


Figure 1. EgoInstructor is a retrieval-augmented multimodal captioning model that retrieves relevant exocentric videos as references to generate the caption of egocentric videos. The cross-view retrieval ability is enabled by training on automatically constructed large-scale pseudo paired ego-exo videos. 

ocentric) videos. To leverage the information in exocentric videos, some researchers have tried to directly transfer models learnt from third-person videos [31, 41, 42, 60]. Due to the discrepancy in terms of recording perspective, camera motion, video continuity, and domain shift, the representations learnt from exocentric videos may not be optimal for egocentric videos. Another line of research involves the joint learning of egocentric and exocentric video representations [3, 24, 59, 71, 72]. However, training these models would require time-synchronised ego- and exo-video pairs collected in the same environment [19, 34, 58, 59]. 

In this paper, we take inspiration from the fact that humans are born with the ability of learning by observation, _i.e._ , the ability of observing third-view demonstrations and subsequently generalising towards egocentric perspective in different environment [25, 51, 56]. For a more specific example, people often seek answers to “How To” questions and watch third-person instructional videos before attempting tasks they are unfamiliar with, _e.g. how to make a cheesebeef sandwich_ . When engaging in the task, people also recall key procedures, visual demonstrations and 

1 

detailed instructions from instructional videos. We explore retrieval-augmented egocentric video captioning, an alternative way for transferring knowledge from exocentric videos to enhance egocentric video understanding. 

Specifically, as shown in Fig. 1, our goal is to learn a unified ego-exo representation space that allows for explicitly retrieving relevant third-view videos available on the Internet ( _e.g._ from HowTo100M [46]), to assist video captioning in the egocentric perspective. This is achieved by first introducing an automated pipeline to generate pseudo ego-exo video pairs at scale by aligning the captions that describe similar actions. Subsequently, the paired videos are employed to train a cross-view retrieval module via a novel EgoExoNCE loss, which aligns both egocentric and exocentric video features with shared text features describing similar action semantics. With this cross-view retrieval capability, we develop a retrieval-augmented egocentric video captioning model by leveraging the retrieved exocentric videos as references for caption generation. 

As a consequence, we evaluate the cross-view retrieval module on seven benchmarks, _e.g._ , EK100 Multi-instance Retrieval [18], Ego4d Multiple Choice Questions [21], YouCook2 video-text retrieval [79] and CharadesEgo crossview video retrieval [59], all of which demonstrate superior performance. Regarding egocentric video captioning on Ego4d [21], we show the benefits of leveraging semantically related exocentric instructional videos. 

## **2. Related Work** 

**Egocentric video understanding.** The unique viewpoint of egocentric videos poses a broad range of challenges in human activity analysis [11], including action recognition and detection [18, 31, 42, 66, 77], human pose estimation [29, 49], gaze estimation [26, 39, 73], and captioning [30, 48]. Recently, the introduction of the Ego4d [21] dataset has prompted a series of studies for representation learning in egocentric videos [5, 43, 76, 78]. For instance, EgoVLP [43] performed contrastive learning over paired egocentric videos and narrations. LaViLa [78] learnt egocentric video-language representations by using pseudolabelled egocentric videos. Despite the progress in egocentric video understanding, these models lack enough generalisation ability on exocentric videos. In contrast, our crossview retrieval module is designed to learn a unified ego-exo video-language representation space. 

**Ego-exo video understanding.** Ego-exo video understanding models involve both egocentric and exocentric videos to solve a variety of vision tasks [2, 3, 20, 24, 42, 59, 69, 71]. The majority of these approaches explore shared visual clues from synchronous egocentric and exocentric data recorded simultaneously [20, 58, 59], which poses additional difficulty and cost in data collection. Subsequently, 

versatile methods are developed without relying on paired egocentric and exocentric videos. EgoExo [42] transferred exocentric pre-trained model to egocentric videos by mining egocentric pseudo labels. AE2 [72] temporally aligned the representations of the unpaired egocentric and exocentric videos via Dynamic Time Warping [9]. Sum-L [67] created cross-view pseudo pairs from egocentric dataset [18, 59] and exocentric dataset [28] using the ground-truth semantic category labels. In contrast, our method learns joint ego-exo representations on unpaired egocentric [21] and exocentric [46] datasets by leveraging captions/narrations and moves beyond vision-only [42, 67, 72] tasks to videolanguage understanding. 

**Retrieval-augmented models.** Retrieval-augmented language models in the NLP community aim at retrieving external knowledge to enhance performance in various NLP tasks [4, 12, 22, 36, 53]. Motivated by these approaches, recent progresses in vision-language domain also retrieve semantically related samples to improve the performance, including image recognition [27, 45], captioning [55, 57], visual question answering [15, 74], image generation [16, 75] and vision-language pre-training [70]. While most of these efforts primarily focus on image domain and employ a pretrained CLIP [52] model for cross-modal retrieval, such approaches are sub-optimal for video data, especially in the case between egocentric and exocentric views. In contrast, our approach involves training a cross-view retrieval module tailored for retrieving exocentric videos for egocentric video understanding. 

## **3. Methodology** 

The overall architecture of our proposed EgoInstructor is shown in Fig. 2. Given an egocentric video _v_[ego] , our model first retrieves _K_ semantically relevant third-view instructional videos and associated texts with the cross-view retrieval, _i.e._ , _{x_[exo] 1 _[, . . . , x]_[exo] _K[}]_[ = Φ][retrieve][(] _[v]_[ego][)][, where] _[ x] i_[exo] = ( _vi_[exo] _, t_[exo] _i_[)][includes][both][video][and][associated][text][narra-] tions. Then, the multimodal captioning model takes both the ego-video and retrieved samples to generate the text description, _t_[ego] = Ψgenerate( _v_[ego] _, {x_[exo] 1 _[, . . . , x]_[exo] _K[}]_[)][.][In the fol-] lowing sections, we start by describing the procedure for training the cross-view retrieval module with unpaired data (Sec. 3.1), then followed by the full retrieval-augmented multimodal captioning model (Sec. 3.2). 

## **3.1. Cross-view Visual Representation Alignment** 

Our goal in this step is to train a cross-view retrieval module that associates the ego- and exo-videos sharing the same semantics, using videos with only caption or narrations, _i.e._ , without manual annotation for pairing ego-exo videos. Specifically, we adopt **egocentric** videos from Ego4D [21] with 4M manually labelled captions, and **exocentric** videos 

2 


![](raw/papers/2_LLM_NLP/LLM_paper_list/caption/cvpr_2024_Retrieval-Augmented_Egocentric_Video_Captioning_images/cvpr_2024_Retrieval-Augmented_Egocentric_Video_Captioning.pdf-0003-00.png)


**----- Start of picture text -----**<br>
#C C turns the bread over and toasts the other side<br>Text Decoder<br>Perceiver<br>LLM block<br>Resampler<br>Gated Cross Attention<br>Egocentric video ···<br>LLM block<br>Perceiver<br>Resampler<br>Gated Cross Attention<br>Ego-Video →Exo-Text Perceiver <video> Press the meat and flip the bread. <eoc><video> Toast the bread until it’s golden. <eoc><br>Ego-Video →Exo-Video Resampler <video><br>**----- End of picture text -----**<br>


Figure 2. An overview of our EgoInstructor. Given an egocentric video, we first retrieve relevant exocentric instructional videos using a frozen cross-view retrieval module pre-trained on pseudo ego-exo pairs generated automatically. The multimodal captioning model (consisting of a visual encoder, a perceiver resampler, and a text decoder.) takes the egocentric video and the retrieved videos and captions as references, and generates the caption of the ego-video. 

from HowTo100M [46], containing detailed instructions for performing daily procedural activities. 

## **3.1.1 Architecture Detail** 

**Egocentric visual-language representation.** For egocentric video clips, we extract representations via a CLIPstyle dual encoder architecture, adopting a frozen VideoMAE [62] as the visual encoder, _zi_[ego] = _f_[ego] ( _vi_[ego] ) _∈_ R _[T][ ×][d]_ with _T_ frames; and a BERT model [32] as the text encoder _u_[ego] _i_ = _f_[text] ( _t_[ego] _i_[)] _[ ∈]_[R] _[d]_[.][Both visual and text encoders have] been pre-trained on Ego4d video-text pairs. 

**Exocentric visual-language representation.** For exocentric video clip and its instruction from HowTo100M dataset, the visual representation is computed using a frozen InternVideo [68] video encoder. This encoder is a Uniformer [38] model pre-trained on large corpus of exocentric video-text pairs [68], _zi_[exo] = _f_[exo] ( _vi_[exo] ) _∈_ R _[T][ ×][d]_ . The text features are also computed with the same BERT encoder used for ego-text processing, _u_[exo] _i_ = _f_[text] ( _t_[exo] _i_[)] _[ ∈]_[R] _[d]_[.] 

**Cross-view video encoder.** With computed visuallanguage representations, we train a cross-view video encoder _ϕ_ ( _·_ ) that takes either egocentric or exocentric video feature as input, and maps them into an embedding space, such that the videos with similar semantics are close together regardless of the shooting perspective. We obtain ˆ ˆ _zi_[ego] = _ϕ_ ( _zi_[ego] ) _∈_ R _[d]_ and _zi_[exo] = _ϕ_ ( _zi_[exo] ) _∈_ R _[d]_ with the temporal dimension being aggregated with average pooling. The cross-view retrieval module is shown in Fig. 3. 


![](raw/papers/2_LLM_NLP/LLM_paper_list/caption/cvpr_2024_Retrieval-Augmented_Egocentric_Video_Captioning_images/cvpr_2024_Retrieval-Augmented_Egocentric_Video_Captioning.pdf-0003-07.png)


**----- Start of picture text -----**<br>
Egocentric Cross-view<br>Video Encoder Video Encoder<br>shared<br>Exocentric Cross-view<br>Video Encoder Video Encoder<br>𝓛𝑬𝒈𝒐𝑬𝒙𝒐𝑵𝑪𝑬<br>Turn the bread over<br>and toast the other side<br>Toast the bread until<br>it’s golden<br>**----- End of picture text -----**<br>


Figure 3. Our cross-view retrieval module trained via EgoExoNCE loss. We keep the egocentric and exocentric video encoders frozen and train the cross-view video encoder and text encoder. 

## **3.1.2 Automatic Ego-Exo Pair Generation** 

For training the described cross-view retrieval module, we propose a scalable approach that exploits natural languages as a bridge to pair videos from different viewpoints, _i.e._ , grouping egocentric and exocentric videos if their corresponding language narrations and instructions are highly relevant. In practice, unlike egocentric videos from Ego4D that have manually annotated captions, instructions for HowTo100M videos are normally acquired from ASR transcripts [8]. This incurs two issues: (i) the transcript may not be of descriptive style, containing redundancy or ambiguity, as depicted in Fig. 4 (left); (ii) the transcript may not be well aligned with the visual signals. There might be greetings from the speaker or inaccuracies in timing, _e.g._ , describing the action after performing it, as discussed in [23]. 

**Caption refinement via large language model.** We transform the ASR transcript into similar style as those of Ego4D captions with a large language model (LLM). Specifically, given an instructional video from HowTo100M, _i.e._ , _x_[exo] = 

3 


![](raw/papers/2_LLM_NLP/LLM_paper_list/caption/cvpr_2024_Retrieval-Augmented_Egocentric_Video_Captioning_images/cvpr_2024_Retrieval-Augmented_Egocentric_Video_Captioning.pdf-0004-00.png)


**----- Start of picture text -----**<br>
Context-aware Caption Refinement Cross-view Pair Construction<br>𝒔 - 𝜶 𝒔 𝒆 𝒆 + 𝜶<br>Today, I’m gonna share<br>0:08 it with you all of our Introduce the steps to make #C C cuts the cucumber<br>0:10 secrets for making cheesesteak sandwiches<br>cheesesteak sandwiches<br>Large<br>Language #C C closes the tap<br>Model<br>Let’s start by toasting<br>3:15 the buttered side of the #C C turns the bread over<br>3:19 bread on medium heat, Toast the bread until it’s golden and toasts the other side<br>just until it’s golden. Ego-exo pair<br>···<br>**----- End of picture text -----**<br>



![](raw/papers/2_LLM_NLP/LLM_paper_list/caption/cvpr_2024_Retrieval-Augmented_Egocentric_Video_Captioning_images/cvpr_2024_Retrieval-Augmented_Egocentric_Video_Captioning.pdf-0004-01.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/caption/cvpr_2024_Retrieval-Augmented_Egocentric_Video_Captioning_images/cvpr_2024_Retrieval-Augmented_Egocentric_Video_Captioning.pdf-0004-02.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/caption/cvpr_2024_Retrieval-Augmented_Egocentric_Video_Captioning_images/cvpr_2024_Retrieval-Augmented_Egocentric_Video_Captioning.pdf-0004-03.png)


Figure 4. An illustration of context-aware caption refinement (left) and cross-view pair construction (right). The ASR transcripts of instructional videos are concatenated and refined by a LLM to match the descriptive style of manually labelled captions in Ego4d. We construct the ego-exo pairs by choosing the ego and exo captions that describe the similar action (e.g., toast the bread). 

_{_ ( _vj, tj, sj, ej_ ) _}[M] j_ =1[,][where][each][narration] _[t][j]_[is][annotated] with a start time _sj_ and end time _ej_ , we refine the ASR transcript by prompting the LLM to only capture the main actions in the transcript and output descriptive captions: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/caption/cvpr_2024_Retrieval-Augmented_Egocentric_Video_Captioning_images/cvpr_2024_Retrieval-Augmented_Egocentric_Video_Captioning.pdf-0004-06.png)


where **Prompt** is composed of task instructions, a list of 10 caption rules followed by example cases. Please refer to **Supplementary** for the exact prompt. Compared with clipwise processing of each transcript individually, video-level refinement enables the LLM to infer the long-term goal of the activity within the context, and manages to replace the pronouns with specific objects deduced from other transcripts for better visual-text alignment. Additionally, we extend the temporal boundary of the narration to [ _s − α, e_ + _α_ ] by _α_ seconds, to mitigate the temporal shift problem. As shown in Fig. 4 (middle), the ASR transcripts are effectively transformed into shorter yet more informative sentences. 

**Pairing ego-exo videos via language alignment.** In addition to refining the captions of HowTo100M videos, we also remove the character indicator in Ego4D narrations, _e.g._ , the _‘#C’_ in captions. Till this point, the egocentric and exocentric narrations share similar textual formats, we then establish pairings via a two-step procedure. _First_ , we conduct an initial categorisation based on the daily activity scenarios, such as food and entertaining, crafts, gardening, _etc_ . Egocentric and exocentric videos belonging to the same scenario are grouped together; _Second_ , we extract nouns and verbs using Spacy [64] from each sentence as they collaboratively determine the semantics of the activity in the video. For each egocentric video, the exocentric video(s) with the highest overlaps in terms of nouns and verbs are selected as the paired sample(s). Note that, we have also experimented by computing sentence similarities directly, however, it is sub-optimal than explicitly comparing entities. 

**Long-form video-text pairs construction.** Till here, we have obtained paired ego-exo short video clips, where the duration of video clips are determined by the timestamp 

of the captions. As both Ego4d and HowTo100M consist of long videos lasting minutes, training on short videotext pairs may not fully exploit the information in long videos. To address this, we propose a simple strategy to better utilise long-form information. In specific, as the narrations/captions are dense, we create ‘fake’ long-form video-text pairs by cropping longer video clips and summarising their accompanied narrations into one sentence via the LLM. These pseudo long-form pairs are integrated into the original data for training, and pairing long-form ego-exo video pairs can be executed similarly as before. 

## **3.1.3 Retrieval Module Training and Inference** 

For training cross-view retrieval, we introduce a novel contrastive loss, termed as EgoExoNCE loss. Specifically, the proposed loss incorporates three variants of positive pairs: (1) the original video-text pairs (ˆ _zi_[ego] _, u_[ego] _i_[)][and] (ˆ _zi_[exo] _, u_[exo] _i_[)][ as adopted in the InfoNCE loss [][52][];][(2) cross-] view pairs mined via language alignment, _i.e._ , (ˆ _zi_[ego] _, u_[exo] _i_[)] and (ˆ _zi_[exo] _, u_[ego] _i_[)][,][as both] _[ u]_[ego] _i_ and _u_[exo] _i_ represent similar actions in the video; (3) all pairs that share at least one noun or verb _regardless of views_ . The remaining pairs in each mini-batch are considered as negative pairs. 

At training time, assume we have _B_ samples in each mini-batch, the positive pair set for the _i_ -th sample is thus defined as: 

_Pi_ = _{_ (ˆ _zi_[ego] _[, u]_[ego] _i_[)] _[,]_[ (ˆ] _[z] i_[exo] _[, u]_[exo] _i_[)] _[} ∪{]_[(ˆ] _[z] i_[ego] _[, u] i_[exo][)] _[,]_[ (ˆ] _[z] i_[exo] _[, u]_[ego] _i_[)] _[}∪] {_ (ˆ _zi_[ego] _[, u][c] j_[)] _[,]_[ (ˆ] _[z] i_[exo] _[, u][c] j_[)] _[|N]_[(] _[t] i[c]_[)] _[ ∩N]_[(] _[t][c] j_[)] _[ ̸]_[=] _[ ∅][,][ V]_[(] _[t] i[c]_[)] _[ ∩V]_[(] _[t][c] j_[)] _[ ̸]_[=] _[ ∅}][,]_ 

where _j ∈B, c ∈{_ ego _,_ exo _}_ , _N_ ( _t_ ) and _V_ ( _t_ ) represent the sets of nouns and verbs in text _t_ . The video-to-text EgoExoNCE loss _L_[v2t] EgoExoNCE[can be calculated as:] 


![](raw/papers/2_LLM_NLP/LLM_paper_list/caption/cvpr_2024_Retrieval-Augmented_Egocentric_Video_Captioning_images/cvpr_2024_Retrieval-Augmented_Egocentric_Video_Captioning.pdf-0004-16.png)


4 

where _⟨z, u⟩_ = exp( _z_[T] _u/τ_ ) and _τ_ is the temperature. The total loss _L_ EgoExoNCE is the sum of video-to-text loss and symmetrical text-to-video loss. Note that, despite the training procedure only encourages cross-view, cross-modal retrieval, in fact, the ego-exo video embeddings are also implicitly pulled closer, by aligning to the shared semantics of the actor’s action. 

At inference time, given an egocentric video and a candidate set of exocentric videos with associated texts, we first calculate egocentric video representation ( _z_ ˆ[ego] ) and exocentric video and text representations (ˆ _z_[exo] _, u_[exo] ). Then, for the egocentric video, we choose the exocentric sample with the highest averaged similarity, considering both (1) ego-video to exo-video similarity and (2) ego-video to exo-text similarity, which is defined as: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/caption/cvpr_2024_Retrieval-Augmented_Egocentric_Video_Captioning_images/cvpr_2024_Retrieval-Augmented_Egocentric_Video_Captioning.pdf-0005-02.png)


## **3.2. Retrieval-augmented Captioning** 

In this section, we detail the proposed retrieval-augmented model for egocentric video captioning. In addition to the target video as input, retrieved instructional videos and associated texts are also incorporated as conditions for generating captions. Next, we present three key modules in our captioning model, as shown in Fig. 2. 

**Visual encoder.** We adopt a frozen CLIP vision transformer encoder that takes the concatenation of exo- and ego-videos _v_ = [ _v_ 1[exo][;] _[ ...]_[;] _[ v] K_[exo][;] _[ v]_[ego][]] _[∈]_[R][(] _[K]_[+1)] _[×][T][ ×][C][×][H][×][W]_[as][in-] put, tokenises each frame into _N_ image patches, and compute dense visual representation per frame, _z_ = Φv( _v_ ) _∈_ R[(] _[K]_[+1)] _[×][T][ ×][N][×][d]_ . Here, [ _·_ ] refers to concatenation. 

**Perceiver resampler** serves as a bottleneck module, that models the correspondence among the visual features and squeezes the visual features to a bottleneck feature with fixed length _L_ , typically _L ≪ T × N_ . This is achieved with Transformer Decoder using fixed-length of learnable query tokens _q ∈_ R _[L][×][d]_ , that cross-attend to the visual features, _i.e._ , _q_ ˆ = Φp( _q,_ [ _z_ + **pe** ; _q_ ]) _∈_ R[(] _[K]_[+1)] _[×][L][×][d]_ . Here, **pe** denotes temporal positional embedding. 

**Text decoder** is a visually conditioned LLM. It takes captions of the retrieved samples as input prompt and crossattends to visual features for caption generation. 

The text input prompt **t** p is formed by stacking captions for all exocentric videos, which is denoted as: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/caption/cvpr_2024_Retrieval-Augmented_Egocentric_Video_Captioning_images/cvpr_2024_Retrieval-Augmented_Egocentric_Video_Captioning.pdf-0005-09.png)


where _⟨_ video _⟩_ and _⟨_ eoc _⟩_ are special media tokens representing the video and end-of-chunk in text format, respectively. 

In particular, several gated cross-attention modules [1, 6] are inserted between the original LLM blocks. Compared 

|Dataset|Evaluation Metrics|#Len(s)|#Num|
|---|---|---|---|
|_Egocentric benchmark_||||
|EK100 MIR [18]|mAP, nDCG|3.7|9668|
|EgoMCQ [21]|inter-/intra-video acc.|34.2|39751|
|SummMCQ [21]|inter-video acc.|181.6|1614|
|_Exocentric benchmark_||||
|YouCook2-Clip [79]|R@1, R@5, R@10|19.7|3350|
|YouCook2-Video [79]|R@1, R@5, R@10|212.4|436|
|_Ego-Exo benchmark_||||
|CharadesEgo [59]|Ego2Exo/Exo2Ego Avg. R@1/@5/@10|22.7|145|
|EgoLearner-MCQ|Ego2Exo/Exo2Ego acc.|5.1|1951|
|_Video Captioning_||||
|Ego4d cooking [21]|BELU-4, METEOR, ROUGE-L, CIDER|0.7|7161|
|EgoLearner|BELU-4, METEOR, ROUGE-L, CIDER|7.4|1089|



Table 1. Datasets for evaluation. We list evaluation metrics, average clip/video length (#Len), and the number of test samples. 

with the standard LLM block, the gated cross-attention module exhibits two key variations: (1) the cross-attention layer takes text features as query and visual features as keys and values, enabling visually conditioned generation. Note that, each text token only cross-attends to the visual feature of its corresponding video. (2) A tanh gating mechanism is added. The output of cross-attention layer and feedforward-network in each gated cross-attention block is multiplied by a tanh layer with zero-initialisation. This ensures that the decoder is initially equivalent to the original LLM, thereby improving the training stability. The captioning model is trained by optimising the standard cross-entropy loss for each predicted token. 

## **4. Experiments** 

## **4.1. Experimental Setups** 

**Training datasets.** For egocentric videos, we use the public Ego4d dataset [21], consisting of 3670 hours of daily activity videos. Following prior works [43, 78], videos belonging to the validation and test sets are excluded during training, resulting in 4.3M video-text pairs for training [78]. For exocentric videos, we adopt the large-scale instructional video dataset HowTo100M [46], containing more than 1M videos from YouTube. We use the annotation from TAN [23] including 30M video and ASR transcript pairs with better temporally aligned narration timestamps. For training the video captioning model, we use the Ego4d [21] cooking subset, containing 0.55M video clips with manually labelled captions. All video clips are obtained based on the timestamps of the captions/narrations. 

**Downstream benchmarks.** We evaluate the zero-shot transfer ability of our **retrieval module** on the validation/test set using benchmark datasets outlined in Table 1. Our evaluation includes 7 retrieval benchmarks that cover both egocentric [18, 21, 59] and exocentric videos [59, 79], ranging from short-term to long-term. Notably, CharadesEgo and an in-house dataset, termed as EgoLearner, with ground-truth ego-exo pairs are adopted to evaluate model’s 

5 

|ID<br>Loss<br>Caption<br>View|Ego-centric Benchmark|Ego-centric Benchmark|Exo-centric Benchmark|Ego-Exo Benchmark<br>EgoLearner<br>CharadesEgo<br>Ego2Exo<br>Exo2Ego<br>Ego2Exo<br>Exo2Ego<br>29.4<br>22.6<br>6.1<br>7.4<br>24.6<br>21.9<br>1.8<br>4.1|Ego-Exo Benchmark<br>EgoLearner<br>CharadesEgo<br>Ego2Exo<br>Exo2Ego<br>Ego2Exo<br>Exo2Ego<br>29.4<br>22.6<br>6.1<br>7.4<br>24.6<br>21.9<br>1.8<br>4.1|
|---|---|---|---|---|---|
||EK100-MIR|EgoMCQ|YouCook2 Clip_→_Text|EgoLearner||
||mAP<br>nDCG|Inter Acc.<br>Intra Acc.|R@1<br>R@5<br>R@10|Ego2Exo<br>Exo2Ego||
|1<br>InfoNCE<br>-<br>Ego<br>2<br>InfoNCE<br>ASR<br>Exo|29.9<br>33.9<br>6.7<br>12.3|91.5<br>54.6<br>35.0<br>22.6|0.1<br>0.5<br>0.6<br>21.7<br>48.1<br>61.2|29.4<br>22.6<br>24.6<br>21.9||
|3<br>InfoNCE<br>ASR<br>Ego+Exo<br>4<br>InfoNCE<br>Refned Cap.<br>Ego+Exo|28.1<br>**32.1**<br>**28.4**<br>32.0|91.5<br>51.8<br>**92.3**<br>**53.2**|21.3<br>45.4<br>57.3<br>**23.3**<br>**49.2**<br>**60.7**|38.7<br>29.9<br>**46.8**<br>**30.1**|**18.3**<br>7.6<br>17.8<br>**9.8**|
|5<br>EgoNCE<br>ASR<br>Ego+Exo<br>6<br>EgoExoNCE<br>ASR<br>Ego+Exo<br>7<br>EgoExoNCE<br>Refned Cap.<br>Ego+Exo|29.0<br>33.1<br>28.6<br>**33.2**<br>**29.2**<br>33.1|91.5<br>52.5<br>91.9<br>51.4<br>**92.8**<br>**54.0**|21.4<br>45.0<br>57.6<br>21.1<br>45.6<br>57.7<br>**23.7**<br>**49.4**<br>**61.8**|39.2<br>30.9<br>41.3<br>**31.3**<br>**46.4**<br>30.5|19.0<br>8.7<br>20.8<br>13.6<br>**22.6**<br>**15.2**|



Table 2. Ablation study on the exocentric instructional video textual descriptions (original ASR transcripts vs. Refined Caption) and the cross-view alignment loss (InfoNCE [52] vs. EgoNCE [43] vs. EgoExoNCE). We report metrics on short-term benchmarks. The baseline model trained on ego-only/exo-only data (ID-1, ID-2) fails on exo/ego benchmarks due to the discrepancy between views. 

video-to-video retrieval ability. The retrieval evaluations include video-text, video-video retrieval, and multiple choice questions (MCQ). Specifically, MCQ involves retrieving the matched video from 5 candidates from a query. The query is the associated text in EgoMCQ and SummMCQ. While in EgoLearner, the video from the other view serves as the query. We report the R@1, R@5 and R@10 for retrieval tasks and Top-1 accuracy for MCQ tasks. For **video captioning** , we sample 7161 cooking video clips from the Ego4d [21] validation set. We also choose videos from EgoLearner which contains 1089 video clips with finegrained video captions. Please refer to the **Supplementary** for details on each dataset. 

**Training and inference pipeline.** To begin with, we take egocentric videos from Ego4d and exocentric videos from HowTo100M to construct pseudo ego-exo pairs offline, which are subsequently used to train the cross-view retrieval module. Next, we take the videos from the cooking subset of Ego4d, along with the paired exocentric videos and captions, to train the retrieval-augmented captioning model. At inference stage, given an egocentric video, we retrieve relevant exocentric videos via the cross-view retrieval module, and use them as references to generate the ego-caption. 

**Implementation details.** In the retrieval module, the egocentric video encoder is a VideoMAE-L [62] model pretrained on Ego4d. The exocentric video encoder is a UniformerV2 [38] model with InternVideo [68] pre-trained weights on third-person video datasets. Both video encoders output a 768-d feature vector for an 8-frame input. Following [23, 46], we get one feature per second by decoding video at 8 fps. To match the average duration of Ego4d and HowTo100M, we randomly sample 4 frame features as input to the cross-view video encoder by default. In terms of pseudo long-form video-text pairs, we crop long video clips that last between 60 and 300 seconds, covering 20 narrations. We employ Stable-Vicuna-v1.5 [17], an advanced instruction-tuned LLM for both long-form text summarisation and caption refinement. The cross-view video encoder consists of four Transformer encoder layers [65]. 

The text encoder is a BERT model [32]. The retrieval module is trained for 5 epochs using the AdamW optimizer with a learning rate of 3 _×_ 10 _[−]_[5] , and the batch size is set to 4096. 

For retrieval-augmented captioning model, the visual encoder is CLIP-L/14 and the perceiver resampler consists of 6-layer transformer blocks with 64 learnable query tokens. We adopt two variants of language models, LLaMA7B [63] and MPT-1B [61]. For fast convergence, we initialise the captioning model with Otter [37] pre-trained weights. We train the model for 3 epochs with an initial learning rate of 1 _×_ 10 _[−]_[5] and cosine learning rate decay. We set batch size to 8/32 for LLaMA-/MPT-based models. 

## **4.2. Experimental Results** 

In this section, we study the effect of each component for cross-view retrieval and retrieval-augmented captioning. 

## **4.2.1 Results on Cross-view Retrieval** 

We evaluate the effect of caption refinement, EgoExoNCE loss, and different egocentric video encoders in this section. Additional ablations can be found in the Supplementary. 

**Effect of caption refinement.** We show comparison between our refined exocentric video caption with the original ASR transcript across multiple losses in Table 2 (ID3 vs. ID-4, ID-6 vs. ID-7). The refined caption leads to consistent improvements on YouCook2 exocentric retrieval, indicating better visual-text alignment. Surprisingly, we observe 1% _∼_ 2% improvement on egocentric benchmark, EgoMCQ. We conjecture that the performance gain is from the alignment between egocentric and exocentric caption formats achieved through caption refinement. This alignment allows egocentric videos to align with broader semantically rich captions, thereby enhancing the representation of egocentric video-text relations. On cross-view benchmark, _i.e._ , EgoLearner video retrieval, the accuracy consistently improves in the Ego2Exo setting regardless of the used losses, as semantically similar ego- and exo-videos are better aligned via similar captions. In the Exo2Ego setting, 

6 

|Method<br>View<br>Backbone||Ego-centric Benchmark||Exo-centri|c Benchmark|Ego-Exo|Benchmark<br>CharadesEgo<br>Ego2Exo<br>Exo2Ego<br>-<br>-|
|---|---|---|---|---|---|---|---|
||EK100-MIR<br>mAP<br>nDCG|EgoMCQ<br>Inter Acc.<br>Intra Acc.|SummMCQ<br>Acc|YC2-Clip<br>R@1<br>R@5|YC2-Video<br>R@1<br>R@5|EgoLearner<br>Ego2Exo<br>Exo2Ego||
|HierVL [5]<br>Ego<br>FIT|18.9<br>24.7|90.5<br>52.4|95.4_†_|-<br>-|-<br>-|-<br>-||
|CLIP [52]<br>Ego+Exo<br>ViT-L<br>CLIP + Ours<br>Ego+Exo<br>ViT-L|18.4<br>24.0<br>**19.4**<br>**24.5**|**88.7**<br>34.6<br>88.3<br>**35.0**|92.0<br>**93.5**|20.2<br>44.0<br>**23.6**<br>**52.8**|36.0<br>76.4<br>**58.0**<br>**90.4**|48.9<br>41.5<br>**49.0**<br>**42.3**|50.1<br>49.0<br>**59.5**<br>**59.1**|
|EgoVLP [43]<br>Ego+Exo<br>TSF-B<br>EgoVLP + Ours<br>Ego+Exo<br>TSF-B|24.7<br>28.4<br>**25.7**<br>**29.1**|91.2<br>45.7<br>**91.9**<br>**47.2**|88.8<br>**91.6**|20.2<br>44.8<br>**22.7**<br>**50.5**|39.6<br>78.0<br>**59.4**<br>**91.3**|52.0<br>43.8<br>**53.2**<br>**46.8**|**21.7**<br>22.9<br>21.2<br>**29.2**|
|LaViLa [78]<br>Ego+Exo<br>TSF-L<br>LaViLa + Ours<br>Ego+Exo<br>TSF-L|24.3<br>28.2<br>**25.7**<br>**29.0**|90.8<br>45.3<br>**91.2**<br>**46.5**|89.9<br>**94.2**|19.7<br>44.5<br>**23.2**<br>**51.2**|37.6<br>76.6<br>**61.0**<br>**89.7**|52.6<br>**43.3**<br>**53.7**<br>41.8|**37.1**<br>25.9<br>35.5<br>**33.6**|
|InternVideo [68]<br>Ego+Exo<br>UF-L<br>InternVideo + Ours<br>Ego+Exo<br>UF-L|21.4<br>25.1<br>**25.7**<br>**27.6**|88.9<br>38.7<br>**90.3**<br>**39.6**|88.6<br>**93.4**|20.4<br>45.4<br>**23.6**<br>**52.1**|34.8<br>73.8<br>**59.6**<br>**87.6**|50.1<br>42.2<br>**52.8**<br>**48.3**|63.9<br>62.0<br>**79.8**<br>**70.3**|
|VideoMAE [62]<br>Ego+Exo<br>ViT-L<br>VideoMAE + Ours<br>Ego+Exo<br>ViT-L|28.1<br>32.1<br>**31.6**<br>**34.9**|91.5<br>51.8<br>**92.7**<br>**54.2**|87.1<br>**90.3**|21.3<br>45.4<br>**25.5**<br>**51.8**|37.6<br>75.4<br>**73.4**<br>**96.8**|38.7<br>29.9<br>**44.0**<br>**30.4**|18.3<br>7.6<br>**25.1**<br>**19.3**|



Table 3. Effect of our cross-view visual representation alignment over various egocentric video encoders, while fixing the exocentric video encoder as InternVideo [68]. We report clip/video to text retrieval results on YouCook-Clip/YouCook-Video. _†_ indicates different test data (still unavailable), which is not fairly comparable. FIT, TSF and UF refer to Frozen in Time [7], TimeSformer [10] and UniformerV2 [38]. 

however, the model exhibits minimal improvement, probably due to the fact that the egocentric videos in EgoLearner are primarily collected in a similar environment, making it challenging to distinguish these ego-videos. 

**Effect of EgoExoNCE loss.** To investigate the impact of different loss functions on aligning egocentric and exocentric videos, we compared our EgoExoNCE with two commonly used loss functions, namely, InfoNCE [52, 78] and EgoNCE [43]. InfoNCE aligns each video with its paired caption without considering cross-view information. EgoNCE extends this by incorporating videos performing similar actions as positive samples, while different actions in the same video as negative samples. 

As shown in Table 2, EgoNCE improves InfoNCE on egocentric benchmarks (ID-3 vs. ID-5), _i.e._ , EK100-MIR and EgoMCQ, however, only marginal improvement is observed on cross-view benchmarks. In contrast, EgoExoNCE boosts the accuracy of InfoNCE by an average of 2.0% and 4.3% on cross-view benchmarks (ID-3 vs. ID-6), _i.e._ , EgoLearner and CharadesEgo, respectively. The results can be further improved by using refined captions (ID-4 vs. ID7). This indicates the superiority of our EgoExoNCE loss in leveraging captions to mine cross-view samples, thereby enhancing the model’s video alignment ability. In summary, our cross-view visual alignment outperforms the baseline on all benchmarks (ID-3 vs. ID-7). 

**Analysis on egocentric video encoder.** We extract egocentric video features using various pre-trained encoders, including CLIP [52], EgoVLP [43], LaViLa [78], InternVideo [68] and VideoMAE [62], while ensuring the consistency of exocentric video features extracted via InternVideo [68]. The results are shown in Table 3. 

All the baseline models are trained on both original egocentric and exocentric data via InfoNCE. By adding our cross-view visual representation alignment strategies, _i.e.,_ training on refined paired data with EgoExoNCE loss, 

all models exhibit consistent improvements on most per- 

formance metrics. Specifically, on cross-view evaluation benchmarks, decent improvement can be seen on CLIP and InternVideo. For instance, our model improves the baseline InternVideo model by 15.9% and 7.7% on CharadesEgo. 

## **4.2.2 Results on Retrieval-augmented Captioning** 

**Effect of retrieved exocentric videos.** Table 4 shows the comparison results on the impact of retrieved samples. We vary the number of exocentric videos _K_ from 0 to 8, which is referred to as _K_ -shot evaluation [1]. In the 0-shot scenario, where no exocentric videos are used, both LLaMAand MPT-based captioning models achieve relatively low captioning performance, with the MPT-based model achieving only a 0.468 CIDER score. In contrast, by leveraging only one exocentric video as reference, the model demonstrates a significant improvement across all metrics. The captioning performance gradually improves as the number of retrieved samples increases, indicating that the multimodal captioning model successfully leverages relevant visual and textual information from exocentric videos. This observation aligns with prior studies on few-shot learning [1, 13, 44, 74]. Compared with refined caption, using the original ASR transcript as the text input prompt leads to a slight decrease in performance for LLaMA-based captioning model (CIDER 1.665 vs. 1.768). We also re-implement a prior egocentric video captioning model on our training set, _i.e.,_ LaViLA-GPT2 [78]. Our retrieval-augmented captioning model outperforms LaViLA by a large margin. 

**Performance on EgoLearner.** We evaluate the generalisation ability of EgoInstructor by directly transferring the model trained on Ego4d cooking subset to EgoLearner cooking videos. As seen from Table 5, in the 0-shot scenario, our captioning models fail to achieve satisfactory results, which can be attributed to the discrepancy between the video and caption distribution of two datasets. In the 1-shot scenario, a significant boost can be observed, demonstrat- 

7 


![](raw/papers/2_LLM_NLP/LLM_paper_list/caption/cvpr_2024_Retrieval-Augmented_Egocentric_Video_Captioning_images/cvpr_2024_Retrieval-Augmented_Egocentric_Video_Captioning.pdf-0008-00.png)


**----- Start of picture text -----**<br>
Retrieved third-person instructional videos Retrieved third-person instructional videos<br>1. ASR:  Now i'm going to take some of the roasted red peppers  1. ASR:  Then you can transfer it into an airtight container and<br>out of the jar.   again it'll last about a week on your countertop.<br>1. Refined:  Take roasted red peppers from the jar. 1. Refined:  Transfer the mixture to an airtight container and<br>store for up to a week.<br>2. ASR:  I'll show you how to chop it up and i've got three serrano<br>peppers. 2. ASR:  You just leave the bowl on the counter.<br>2. Refined:  Show how to chop and use three serrano peppers. 2. Refined:  Leave the bowl on the counter.<br>GT Caption:  #C C takes the pepper on the chop board.  GT Caption:  #C C drops the bowl on the counter.<br>GPT4-V:  The man is chopping vegetables on a cutting board.  GPT4-V:  The man is placing chopped vegetables into a bowl.<br>Generated (w/o exo):  #C C puts down the knife.  Generated (w/o exo):  #C C puts the knife on the chopping<br>Generated (w/ exo):  #C C picks up another pepper from the  board.<br>cutting board. Generated (w/ exo):  #C C puts the bowl in the countertop.<br>**----- End of picture text -----**<br>


Figure 5. **Visualisation results.** For each egocentric video, we show two retrieved third-person instructional videos and their original ASR/refined captions. By leveraging retrieved exocentric samples, the generated captions capture correct actions and interacting objects. 

|Model|Shot|BELU-4|METEOR|ROUGE-L|CIDER|
|---|---|---|---|---|---|
|LaViLA-GPT2 [78]|0|0.131|0.287|0.478|0.751|
|LaViLA-GPT2-XL|0|0.170|0.305|0.514|1.030|
|LaViLA-LLaMA7B|0|0.174|0.313|0.516|1.089|
|_Ours_||||||
|LLaMA-7B|0|0.125|0.276|0.458|0.541|
|LLaMA-7B-ASR|1|0.277|0.384|0.583|1.665|
|LLaMA-7B|1|**0.284**|**0.387**|**0.589**|**1.738**|
|MPT-1B|0|0.120|0.279|0.447|0.468|
|MPT-1B|1|0.322|0.400|0.612|1.973|
|MPT-1B|2|0.325|0.402|0.618|2.054|
|MPT-1B|4|**0.326**|0.404|0.620|2.070|
|MPT-1B|8|0.320|**0.407**|**0.622**|**2.107**|



Table 4. Video captioning results on Ego4d validation set (cooking subset). Shot refers to the number of exocentric videos. 

ing that EgoInstructor effectively transfers knowledge from retrieved exocentric videos to assist ego-video captioning, achieving fast adaptation from Ego4d to EgoLearner. In addition, we study two variants of retrieved samples: (1) random selection of exocentric videos and (2) groundtruth paired exocentric videos. Table 5 reveals that using randomly selected exocentric videos yields similar performance to 0-shot captioning, indicating that random samples do not provide useful semantics for EgoInstructor. In contrast, samples retrieved by our cross-view retriever lead to a large gain of captioning performance, and using GT paired exo-videos performs even better. 

|Model|Shot|BELU-4|METEOR|ROUGE-L|CIDER|
|---|---|---|---|---|---|
|LaViLA-GPT2 [78]|0|0.016|0.069|0.175|0.085|
|LaViLA-GPT2-XL|0|0.017|0.069|0.175|0.078|
|LaViLA-LLaMA7B|0|0.015|0.082|0.173|0.085|
|_Ours_||||||
|LLaMA-7B|0|0.005|0.071|0.140|0.082|
|LLaMA-7B|1|0.024|0.120|0.226|0.204|
|LLaMA-7B (random)|1|0.008|0.071|0.173|0.052|
|LLaMA-7B (GT)|1|**0.029**|**0.134**|**0.240**|**0.251**|
|MPT-1B|0|0.006|0.070|0.156|0.065|
|MPT-1B|1|0.024|0.130|0.241|0.245|
|MPT-1B (random)|1|0.010|0.071|0.182|0.054|
|MPT-1B (GT)|1|**0.032**|**0.153**|**0.261**|**0.327**|



Table 5. Zero-shot and one-shot video captioning results on EgoLearner. All the models are trained on Ego4d cooking subset and directly transferred to EgoLearner. 

describes fine-grained objects in the video, such as “pepper” or “bowl”. In contrast, the captioning model without thirdperson video references generates less accurate descriptions. These visualisation results demonstrate the effectiveness of our EgoInstructor in leveraging third-person videos to enhance egocentric video understanding. We have also experimented with the GPT4-Vision model [50]. Though GPT4-V excels at detecting objects in the video, the generated caption is not precise due to the incorrect perception of interacting objects in egocentric videos. 

## **5. Conclusion** 

## **4.2.3 Qualitative Results** 

As shown in Fig. 5, we showcase the retrieved third-person videos relevant to the given egocentric videos. As can be seen from examples, the retrieved samples are generally associated with actions or objects present in the egocentric videos. With the assistance of these third-person videos and corresponding captions, the captioning model accurately 

In this paper, we consider the problem of egocentric video captioning, and present a retrieval-augmented model, named EgoInstructor. It aims to retrieve semantically similar third-person instructional videos as references and generate captions for egocentric videos. We design a crossview visual representation alignment for retrieval, where we train a cross-view retrieval module on automatically cu- 

8 

rated ego- and exo-video pairs. We propose a novel EgoExoNCE loss for training retrieval module, which aligns egoand exo-video features to shared text features representing similar action semantics. Across seven benchmarks, our retrieval module demonstrates strong zero-shot crossview and cross-modal retrieval ability. With the capability of cross-view retrieval, quantitative and qualitative results show the benefits of leveraging exocentric videos to enhance egocentric video captioning. 

## **References** 

- [1] Jean-Baptiste Alayrac, Jeff Donahue, Pauline Luc, Antoine Miech, Iain Barr, Yana Hasson, Karel Lenc, Arthur Mensch, Katherine Millican, Malcolm Reynolds, et al. Flamingo: a visual language model for few-shot learning. _Advances in Neural Information Processing Systems_ , 35:23716–23736, 2022. 5, 7 

- [2] Shervin Ardeshir and Ali Borji. Ego2top: Matching viewers in egocentric and top-view videos. In _Proceedings of the European Conference on Computer Vision_ , pages 253–268. Springer, 2016. 2 

- [3] Shervin Ardeshir and Ali Borji. An exocentric look at egocentric actions and vice versa. _Computer Vision and Image Understanding_ , 171:61–68, 2018. 1, 2 

- [4] Akari Asai, Sewon Min, Zexuan Zhong, and Danqi Chen. Retrieval-based language models and applications. In _Proceedings of the 61st Annual Meeting of the Association for Computational Linguistics (Volume 6: Tutorial Abstracts)_ , pages 41–46, 2023. 2 

- [5] Kumar Ashutosh, Rohit Girdhar, Lorenzo Torresani, and Kristen Grauman. Hiervl: Learning hierarchical videolanguage embeddings. In _Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition_ , pages 23066–23078, 2023. 1, 2, 7 

- [6] Anas Awadalla, Irena Gao, Josh Gardner, Jack Hessel, Yusuf Hanafy, Wanrong Zhu, Kalyani Marathe, Yonatan Bitton, Samir Gadre, Shiori Sagawa, et al. Openflamingo: An opensource framework for training large autoregressive visionlanguage models. _arXiv preprint arXiv:2308.01390_ , 2023. 5 

- [7] Max Bain, Arsha Nagrani, G¨ul Varol, and Andrew Zisserman. Frozen in time: A joint video and image encoder for end-to-end retrieval. In _Proceedings of the IEEE/CVF International Conference on Computer Vision_ , pages 1728–1738, 2021. 7 

- [8] Max Bain, Jaesung Huh, Tengda Han, and Andrew Zisserman. Whisperx: Time-accurate speech transcription of longform audio. _INTERSPEECH 2023_ , 2023. 3 

- [9] Donald J Berndt and James Clifford. Using dynamic time warping to find patterns in time series. In _Proceedings of the 3rd International Conference on Knowledge Discovery and Data Mining_ , pages 359–370, 1994. 2 

- [10] Gedas Bertasius, Heng Wang, and Lorenzo Torresani. Is space-time attention all you need for video understanding? In _International Conference on Machine Learning_ , page 4, 2021. 7 

- [11] Alejandro Betancourt, Pietro Morerio, Carlo S Regazzoni, and Matthias Rauterberg. The evolution of first person vision methods: A survey. _IEEE Transactions on Circuits and Systems for Video Technology_ , 25(5):744–760, 2015. 2 

- [12] Sebastian Borgeaud, Arthur Mensch, Jordan Hoffmann, Trevor Cai, Eliza Rutherford, Katie Millican, George Bm Van Den Driessche, Jean-Baptiste Lespiau, Bogdan Damoc, Aidan Clark, et al. Improving language models by retrieving from trillions of tokens. In _International Conference on Machine Learning_ , pages 2206–2240. PMLR, 2022. 2 

- [13] Tom Brown, Benjamin Mann, Nick Ryder, Melanie Subbiah, Jared D Kaplan, Prafulla Dhariwal, Arvind Neelakantan, Pranav Shyam, Girish Sastry, Amanda Askell, et al. Language models are few-shot learners. _Advances in Neural Information Processing Systems_ , 33:1877–1901, 2020. 7 

- [14] Guo Chen, Sen Xing, Zhe Chen, Yi Wang, Kunchang Li, Yizhuo Li, Yi Liu, Jiahao Wang, Yin-Dong Zheng, Bingkun Huang, et al. Internvideo-ego4d: A pack of champion solutions to ego4d challenges. _arXiv preprint arXiv:2211.09529_ , 2022. 1 

- [15] Wenhu Chen, Hexiang Hu, Xi Chen, Pat Verga, and William W Cohen. Murag: Multimodal retrieval-augmented generator for open question answering over images and text. _Proceedings of the 2022 Conference on Empirical Methods in Natural Language Processing_ , 2022. 2 

- [16] Wenhu Chen, Hexiang Hu, Chitwan Saharia, and William W Cohen. Re-imagen: Retrieval-augmented text-to-image generator. _arXiv preprint arXiv:2209.14491_ , 2022. 2 

- [17] Wei-Lin Chiang, Zhuohan Li, Zi Lin, Ying Sheng, Zhanghao Wu, Hao Zhang, Lianmin Zheng, Siyuan Zhuang, Yonghao Zhuang, Joseph E. Gonzalez, Ion Stoica, and Eric P. Xing. Vicuna: An open-source chatbot impressing gpt-4 with 90 6 

- [18] Dima Damen, Hazel Doughty, Giovanni Maria Farinella, Antonino Furnari, Evangelos Kazakos, Jian Ma, Davide Moltisanti, Jonathan Munro, Toby Perrett, Will Price, et al. Rescaling egocentric vision: Collection, pipeline and challenges for epic-kitchens-100. _International Journal of Computer Vision_ , pages 1–23, 2022. 1, 2, 5 

- [19] Fernando De la Torre, Jessica Hodgins, Adam Bargteil, Xavier Martin, Justin Macey, Alex Collado, and Pep Beltran. Guide to the carnegie mellon university multimodal activity (cmu-mmac) database. 2009. 1 

- [20] Chenyou Fan, Jangwon Lee, Mingze Xu, Krishna Kumar Singh, Yong Jae Lee, David J Crandall, and Michael S Ryoo. Identifying first-person camera wearers in thirdperson videos. In _Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition_ , pages 5125– 5133, 2017. 2 

- [21] Kristen Grauman, Andrew Westbury, Eugene Byrne, Zachary Chavis, Antonino Furnari, Rohit Girdhar, Jackson Hamburger, Hao Jiang, Miao Liu, Xingyu Liu, et al. Ego4d: Around the world in 3,000 hours of egocentric video. In _Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition_ , pages 18995–19012, 2022. 2, 5, 6 

- [22] Kelvin Guu, Kenton Lee, Zora Tung, Panupong Pasupat, and Mingwei Chang. Retrieval augmented language model pretraining. In _International Conference on Machine Learning_ , pages 3929–3938. PMLR, 2020. 2 

9 

- [23] Tengda Han, Weidi Xie, and Andrew Zisserman. Temporal alignment networks for long-term video. In _Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition_ , pages 2906–2916, 2022. 3, 5, 6 

- [24] Hsuan-I Ho, Wei-Chen Chiu, and Yu-Chiang Frank Wang. Summarizing first-person videos from third persons’ points of view. In _Proceedings of the European Conference on Computer Vision_ , pages 70–85, 2018. 1, 2 

- [25] Nicola J Hodges, A Mark Williams, Spencer J Hayes, and Gavin Breslin. What is modelled during observational learning? _Journal of Sports Sciences_ , 25(5):531–545, 2007. 1 

- [26] Yifei Huang, Minjie Cai, Zhenqiang Li, and Yoichi Sato. Predicting gaze in egocentric video by learning taskdependent attention transition. In _Proceedings of the European Conference on Computer Vision_ , pages 754–769, 2018. 2 

- [27] Ahmet Iscen, Mathilde Caron, Alireza Fathi, and Cordelia Schmid. Retrieval-enhanced contrastive vision-text models. _arXiv preprint arXiv:2306.07196_ , 2023. 2 

- [28] Baoxiong Jia, Yixin Chen, Siyuan Huang, Yixin Zhu, and Song-Chun Zhu. Lemma: A multi-view dataset for le arning multi-agent multi-task activities. In _Proceedings of the European Conference on Computer Vision_ , pages 767–786. Springer, 2020. 2 

- [29] Hao Jiang and Kristen Grauman. Seeing invisible poses: Estimating 3d body pose from egocentric video. In _2017 IEEE Conference on Computer Vision and Pattern Recognition_ , pages 3501–3509. IEEE, 2017. 2 

- [30] Soo-Han Kang and Ji-Hyeong Han. Video captioning based on both egocentric and exocentric views of robot vision for human-robot interaction. _International Journal of Social Robotics_ , pages 1–11, 2021. 2 

- [31] Evangelos Kazakos, Arsha Nagrani, Andrew Zisserman, and Dima Damen. Epic-fusion: Audio-visual temporal binding for egocentric action recognition. In _Proceedings of the IEEE/CVF International Conference on Computer Vision_ , pages 5492–5501, 2019. 1, 2 

- [32] Jacob Devlin Ming-Wei Chang Kenton and Lee Kristina Toutanova. Bert: Pre-training of deep bidirectional transformers for language understanding. In _Proceedings of North American Chapter of the Association for Computational Linguistics-HLT_ , page 2, 2019. 3, 6 

- [33] Ashish Kumar, Saurabh Gupta, and Jitendra Malik. Learning navigation subroutines from egocentric videos. In _Conference on Robot Learning_ , pages 617–626. PMLR, 2020. 1 

- [34] Taein Kwon, Bugra Tekin, Jan St¨uhmer, Federica Bogo, and Marc Pollefeys. H2o: Two hands manipulating objects for first person interaction recognition. In _Proceedings of the IEEE/CVF International Conference on Computer Vision_ , pages 10138–10148, 2021. 1 

- [35] Bolin Lai, Miao Liu, Fiona Ryan, and James M Rehg. In the eye of transformer: Global–local correlation for egocentric gaze estimation and beyond. _International Journal of Computer Vision_ , pages 1–18, 2023. 1 

- [36] Patrick Lewis, Ethan Perez, Aleksandra Piktus, Fabio Petroni, Vladimir Karpukhin, Naman Goyal, Heinrich 

K¨uttler, Mike Lewis, Wen-tau Yih, Tim Rockt¨aschel, et al. Retrieval-augmented generation for knowledge-intensive nlp tasks. _Advances in Neural Information Processing Systems_ , 33:9459–9474, 2020. 2 

- [37] Bo Li, Yuanhan Zhang, Liangyu Chen, Jinghao Wang, Jingkang Yang, and Ziwei Liu. Otter: A multi-modal model with in-context instruction tuning. _arXiv preprint arXiv:2305.03726_ , 2023. 6 

- [38] Kunchang Li, Yali Wang, Yinan He, Yizhuo Li, Yi Wang, Limin Wang, and Yu Qiao. Uniformerv2: Unlocking the potential of image vits for video understanding. In _Proceedings of the IEEE/CVF International Conference on Computer Vision_ , pages 1632–1643, 2023. 3, 6, 7 

- [39] Yin Li, Alireza Fathi, and James M Rehg. Learning to predict gaze in egocentric video. In _Proceedings of the IEEE International Conference on Computer Vision_ , pages 3216–3223, 2013. 1, 2 

- [40] Yin Li, Zhefan Ye, and James M Rehg. Delving into egocentric actions. In _Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition_ , pages 287–295, 2015. 1 

- [41] Yin Li, Miao Liu, and James M Rehg. In the eye of beholder: Joint learning of gaze and actions in first person video. In _Proceedings of the European Conference on Computer Vision_ , pages 619–635, 2018. 1 

- [42] Yanghao Li, Tushar Nagarajan, Bo Xiong, and Kristen Grauman. Ego-exo: Transferring visual representations from third-person to first-person videos. In _Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition_ , pages 6943–6953, 2021. 1, 2 

- [43] Kevin Qinghong Lin, Jinpeng Wang, Mattia Soldan, Michael Wray, Rui Yan, Eric Z XU, Difei Gao, Rong-Cheng Tu, Wenzhe Zhao, Weijie Kong, et al. Egocentric video-language pretraining. _Advances in Neural Information Processing Systems_ , 35:7575–7586, 2022. 1, 2, 5, 6, 7 

- [44] Jiachang Liu, Dinghan Shen, Yizhe Zhang, Bill Dolan, Lawrence Carin, and Weizhu Chen. What makes good in-context examples for gpt-3? _arXiv preprint arXiv:2101.06804_ , 2021. 7 

- [45] Alexander Long, Wei Yin, Thalaiyasingam Ajanthan, Vu Nguyen, Pulak Purkait, Ravi Garg, Alan Blair, Chunhua Shen, and Anton van den Hengel. Retrieval augmented classification for long-tail visual recognition. In _Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition_ , pages 6959–6969, 2022. 2 

- [46] Antoine Miech, Dimitri Zhukov, Jean-Baptiste Alayrac, Makarand Tapaswi, Ivan Laptev, and Josef Sivic. Howto100m: Learning a text-video embedding by watching hundred million narrated video clips. In _Proceedings of the IEEE/CVF International Conference on Computer Vision_ , pages 2630–2640, 2019. 2, 3, 5, 6 

- [47] Tushar Nagarajan, Yanghao Li, Christoph Feichtenhofer, and Kristen Grauman. Ego-topo: Environment affordances from egocentric video. In _Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition_ , pages 163–172, 2020. 1 

- [48] Katsuyuki Nakamura, Hiroki Ohashi, and Mitsuhiro Okada. Sensor-augmented egocentric-video captioning with dy- 

10 

namic modal attention. In _Proceedings of the 29th ACM International Conference on Multimedia_ , pages 4220–4229, 2021. 2 

- [49] Evonne Ng, Donglai Xiang, Hanbyul Joo, and Kristen Grauman. You2me: Inferring body pose in egocentric video via first and second person interactions. In _Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition_ , pages 9890–9900, 2020. 2 

- [50] OpenAI. Gpt-4 technical report, 2023. 8 

- [51] David Premack and Guy Woodruff. Does the chimpanzee have a theory of mind? _Behavioral and Brain Sciences_ , 1 (4):515–526, 1978. 1 

- [52] Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, et al. Learning transferable visual models from natural language supervision. In _International Conference on Machine Learning_ , pages 8748–8763. PMLR, 2021. 2, 4, 6, 7 

- [53] Ori Ram, Yoav Levine, Itay Dalmedigos, Dor Muhlgay, Amnon Shashua, Kevin Leyton-Brown, and Yoav Shoham. In-context retrieval-augmented language models. _arXiv preprint arXiv:2302.00083_ , 2023. 2 

- [54] Santhosh Kumar Ramakrishnan, Ziad Al-Halah, and Kristen Grauman. Naq: Leveraging narrations as queries to supervise episodic memory. In _Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition_ , pages 6694–6703, 2023. 1 

- [55] Rita Ramos, Bruno Martins, Desmond Elliott, and Yova Kementchedjhieva. Smallcap: lightweight image captioning prompted with retrieval augmentation. In _Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition_ , pages 2840–2849, 2023. 2 

- [56] Giacomo Rizzolatti and Laila Craighero. The mirror-neuron system. _Annu. Rev. Neurosci._ , 27:169–192, 2004. 1 

- [57] Sara Sarto, Marcella Cornia, Lorenzo Baraldi, and Rita Cucchiara. Retrieval-augmented transformer for image captioning. In _Proceedings of the 19th International Conference on Content-based Multimedia Indexing_ , pages 1–7, 2022. 2 

- [58] Fadime Sener, Dibyadip Chatterjee, Daniel Shelepov, Kun He, Dipika Singhania, Robert Wang, and Angela Yao. Assembly101: A large-scale multi-view video dataset for understanding procedural activities. In _Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition_ , pages 21096–21106, 2022. 1, 2 

- [59] Gunnar A Sigurdsson, Abhinav Gupta, Cordelia Schmid, Ali Farhadi, and Karteek Alahari. Actor and observer: Joint modeling of first and third-person videos. In _Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition_ , pages 7396–7404, 2018. 1, 2, 5 

- [60] Swathikiran Sudhakaran, Sergio Escalera, and Oswald Lanz. Lsta: Long short-term attention for egocentric action recognition. In _Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition_ , pages 9954–9963, 2019. 1 

- [61] MosaicML NLP Team. Introducing mpt-7b: A new standard for open-source, commercially usable llms, 2023. Accessed: 2023-05-05. 6 

- [62] Zhan Tong, Yibing Song, Jue Wang, and Limin Wang. Videomae: Masked autoencoders are data-efficient learners for self-supervised video pre-training. _Advances in Neural Information Processing Systems_ , 35:10078–10093, 2022. 3, 6, 7 

- [63] Hugo Touvron, Thibaut Lavril, Gautier Izacard, Xavier Martinet, Marie-Anne Lachaux, Timoth´ee Lacroix, Baptiste Rozi`ere, Naman Goyal, Eric Hambro, Faisal Azhar, Aurelien Rodriguez, Armand Joulin, Edouard Grave, and Guillaume Lample. Llama: Open and efficient foundation language models. _arXiv preprint arXiv:2302.13971_ , 2023. 6 

- [64] Yuli Vasiliev. _Natural language processing with Python and spaCy: A practical introduction_ . No Starch Press, 2020. 4 

- [65] Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez, Łukasz Kaiser, and Illia Polosukhin. Attention is all you need. _Advances in Neural Information Processing Systems_ , 30, 2017. 6 

- [66] Huiyu Wang, Mitesh Kumar Singh, and Lorenzo Torresani. Ego-only: Egocentric action detection without exocentric transferring. In _Proceedings of the IEEE/CVF International Conference on Computer Vision_ , pages 5250–5261, 2023. 1, 2 

- [67] Qitong Wang, Long Zhao, Liangzhe Yuan, Ting Liu, and Xi Peng. Learning from semantic alignment between unpaired multiviews for egocentric video recognition. In _Proceedings of the IEEE/CVF International Conference on Computer Vision_ , pages 3307–3317, 2023. 2 

- [68] Yi Wang, Kunchang Li, Yizhuo Li, Yinan He, Bingkun Huang, Zhiyu Zhao, Hongjie Zhang, Jilan Xu, Yi Liu, Zun Wang, et al. Internvideo: General video foundation models via generative and discriminative learning. _arXiv preprint arXiv:2212.03191_ , 2022. 3, 6, 7 

- [69] Xinxiao Wu, Han Wang, Cuiwei Liu, and Yunde Jia. Crossview action recognition over heterogeneous feature spaces. In _Proceedings of the IEEE International Conference on Computer Vision_ , pages 609–616, 2013. 2 

- [70] Hu Xu, Gargi Ghosh, Po-Yao Huang, Dmytro Okhonko, Armen Aghajanyan, Florian Metze, Luke Zettlemoyer, and Christoph Feichtenhofer. Videoclip: Contrastive pre-training for zero-shot video-text understanding. _Proceedings of the 2021 Conference on Empirical Methods in Natural Language Processing_ , 2021. 2 

- [71] Mingze Xu, Chenyou Fan, Yuchen Wang, Michael S Ryoo, and David J Crandall. Joint person segmentation and identification in synchronized first-and third-person videos. In _Proceedings of the European Conference on Computer Vision_ , pages 637–652, 2018. 1, 2 

- [72] Zihui Xue and Kristen Grauman. Learning fine-grained view-invariant representations from unpaired ego-exo videos via temporal alignment. _Advances in Neural Information Processing Systems_ , 2023. 1, 2 

- [73] Kentaro Yamada, Yusuke Sugano, Takahiro Okabe, Yoichi Sato, Akihiro Sugimoto, and Kazuo Hiraki. Attention prediction in egocentric video using motion and visual saliency. In _Advances in Image and Video Technology: 5th Pacific Rim Symposium, PSIVT 2011, Gwangju, South Korea, November 20-23, 2011, Proceedings, Part I 5_ , pages 277–288. Springer, 2012. 1, 2 

11 

- [74] Zhengyuan Yang, Zhe Gan, Jianfeng Wang, Xiaowei Hu, Yumao Lu, Zicheng Liu, and Lijuan Wang. An empirical study of gpt-3 for few-shot knowledge-based vqa. In _Proceedings of the AAAI Conference on Artificial Intelligence_ , pages 3081–3089, 2022. 2, 7 

- [75] Michihiro Yasunaga, Armen Aghajanyan, Weijia Shi, Richard James, Jure Leskovec, Percy Liang, Mike Lewis, Luke Zettlemoyer, and Wen-tau Yih. Retrieval-augmented multimodal language modeling. 2023. 2 

- [76] Chuhan Zhang, Ankush Gputa, and Andrew Zisserman. Helping hands: An object-aware ego-centric video recognition model. In _International Conference on Computer Vision_ , 2023. 2 

- [77] Chen-Lin Zhang, Jianxin Wu, and Yin Li. Actionformer: Localizing moments of actions with transformers. In _Proceedings of the European Conference on Computer Vision_ , pages 492–510. Springer, 2022. 1, 2 

- [78] Yue Zhao, Ishan Misra, Philipp Kr¨ahenb¨uhl, and Rohit Girdhar. Learning video representations from large language models. In _Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition_ , pages 6586– 6597, 2023. 1, 2, 5, 7, 8 

- [79] Luowei Zhou, Chenliang Xu, and Jason Corso. Towards automatic learning of procedures from web instructional videos. In _Proceedings of the AAAI Conference on Artificial Intelligence_ , 2018. 2, 5 

12 

