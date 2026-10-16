Published as a conference paper at ICLR 2024 

# - LOOK, REMEMBER AND REASON: GROUNDED REA SONING IN VIDEOS WITH LANGUAGE MODELS 

**Apratim Bhattacharyya, Sunny Panchal, Mingu Lee, Reza Pourreza, Pulkit Madan, Roland Memisevic** 

Qualcomm AI Research _[∗]_ 

## ABSTRACT 

Multi-modal language models (LM) have recently shown promising performance in high-level reasoning tasks on videos. However, existing methods still fall short in tasks like causal or compositional spatiotemporal reasoning over actions, in which model predictions need to be grounded in fine-grained low-level details, such as object motions and object interactions. In this work, we propose training an LM end-to-end on low-level surrogate tasks, including object detection, re-identification, and tracking, to endow the model with the required low-level visual capabilities. We show that a two-stream video encoder with spatiotemporal attention is effective at capturing the required static and motion-based cues in the video. By leveraging the LM’s ability to perform the low-level surrogate tasks, we can cast reasoning in videos as the three-step process of _Look, Remember, Reason_ , wherein visual information is extracted using low-level visual skills step-by-step and then integrated to arrive at a final answer. We demonstrate the effectiveness of our framework on diverse visual reasoning tasks from the ACRE, CATER, Something-Else and STAR datasets. Our approach is trainable end-to-end and surpasses state-of-the-art task-specific methods across tasks by a large margin. 

## 1 INTRODUCTION 

Autoregressive language models (LMs) have shown impressive results on reasoning tasks such as on grade school math problems (Cobbe et al., 2021) and even on LSAT (OpenAI, 2023). Most language models designed for these problems, however, are trained on only textual data. Since many real-world scenarios require reasoning over heterogeneous sensory inputs, _e.g_ ., visual cues. multi-modal LMs for images or videos have recently gained traction (Alayrac et al., 2022; Koh et al., 2023; Zhang et al., 2023b; Maaz et al., 2023; Li et al., 2023b;b; Maaz et al., 2023). The focus of current multi-modal LMs for videos have primarily been on high-level question answering and instruction following (Maaz et al., 2023; Li et al., 2023b; Alayrac et al., 2022). However, many visual reasoning problems in videos require grounding in fine-grained low-level information, _i.e_ ., recognizing objects, and understanding their spatiotemporal interactions. For example, as shown in Fig. 1, recognition of the high-level compositional action requires a fine-grained understanding of low-level details of motion and interactions between objects. The ability of multi-modal LMs to perform visual reasoning tasks such as compositional action recognition in videos (Materzynska et al., 2020) that require a combination of low-level skills with high-level reasoning has not yet been explored. 

To enable multi-modal LMs to solve such reasoning problems we propose our _Look, Remember and Reason (LRR)_ multi-modal LM. Our LRR model architecture extracts dense spatiotemporal features from each input frame. This is accomplished using a two-stream attention-based architecture that captures low-level spatial and temporal details (Simonyan & Zisserman, 2014) at each input video frame using top-down cross-attention. Our multi-modal LRR model is grounded to relevant low-level visual information in the scene by stochastically introducing low-level surrogate tasks during training, including object recognition, re-identification, and tracking, at randomly selected time-steps ( _c.f_ . Fig. 1). We keep these grounded spatiotemporal features, which include low-level visual details, in the working memory, _i.e_ ., “remembered” within the context window of the LM. This allows the 

> _∗_ Qualcomm AI Research is an initiative of Qualcomm Technologies, Inc. 

1 

Published as a conference paper at ICLR 2024 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2024_Look,_Remember_and_Reason_Grounded_Reasoning_in_Videos_with_Language_Models_images/iclr_2024_Look,_Remember_and_Reason_Grounded_Reasoning_in_Videos_with_Language_Models.pdf-0002-01.png)


**----- Start of picture text -----**<br>
Video Frames<br>Surrogate Task for Grounding<br>Object Tracking<br>Visual Reasoning Solution<br>Problem Language Model  Pushing [something]<br>Example Task:  Compositional  from right to left<br>Action Recognition<br>**----- End of picture text -----**<br>


Figure 1. Our Look, Remember, Reason (LRR) model ‘looks’ at the video frames to extract relevant low-level information, _e.g_ ., object motion and interactions, supervised with surrogate tasks like object tracking only during training. It ‘remembers’ the information from intermediate steps and ‘reasons’ using the aggregated information. 

model to combine the low-level visual features with high-level inferences to “reason” and generate the final responses using our LRR framework as shown in Fig. 1. 

Our main contributions are: (i) We highlight the importance of grounding for visual reasoning in multi-modal LMs and propose a novel Look, Remember, and Reason framework to this end to instill the required low-level visual skills in the model using surrogate tasks during training; (ii) We introduce a two-stream video encoder that captures both the scene structure and object motions, crucial for learning the low-level skills; (iii) We demonstrate the effectiveness of our approach on ACRE (Zhang et al., 2021), CATER (Girdhar & Ramanan, 2020), and the real-world Something-Else (Materzynska et al., 2020) and STAR (Wu et al., 2021) datasets. Our approach outperforms the prior state-of-the-art, that is based on highly task-specific architectures, by a large margin—highlighting how our general-purpose LRR model can perform varied and complex spatiotemporal reasoning tasks in videos including causal, compositional and situated reasoning. 

## 2 RELATED WORK 

**Multi-modal language models.** The use of auto-regressive language models with adapters to process visual inputs is an active area of research. Recently proposed models include Pix2seq (Chen et al., 2022), which is trained to extract low-level visual information from images. Unlike our work, extracting low-level visual information is a goal in itself in that work not a surrogate task, and the model processes images not videos. Other image-based multi-modal LMs include ViperGPT (Surís et al., 2023), VisProg (Gupta & Kembhavi, 2022), and Chameleon (Lu et al., 2023). Similarly, PaLM-E (Driess et al., 2023) provides images and text as interleaved multi-modal latent vectors, allowing it to process multiple images within any part of a sentence which serves as an input to the LM where the model is trained end-to-end. LLaMA-Adapter (Zhang et al., 2023a) introduces an adapter layer to enable multi-modal inputs with the LLaMA model Touvron et al. (2023). FROMAGe (Koh et al., 2023) freezes the language model and fine-tunes the input and output linear layers to encode multi-modal interactions. InstructBLIP (Dai et al., 2023) improves high-level instruction following abilities using instruction fine-tuning. Qwen-VL (Bai et al., 2023) and Kosmos-2 (Peng et al., 2023) propose grounded multi-model LMs that can localize referring expressions in images. In contrast to our work, all these approaches process images not videos. 

Video-based multi-modal LMs include Video-ChatGPT (Maaz et al., 2023), VideoChat (Li et al., 2023b), and Valley (Luo et al., 2023). They focus on question answering and instruction following using video. Flamingo (Alayrac et al., 2022), by using a Perceiver as input, is also able to ingest video data, and similarly is trained to infer high-level concepts. In our work, in contrast, we study visual reasoning tasks that require low-level understanding in the form of _e.g_ ., motion, actions, and interactions, in addition to high-level reasoning. We show how grounding high-level concepts in these low-level representations greatly improves video-based reasoning tasks. 

**Spatiotemporal video grounding.** Recent work on grounding focuses on spatially localizing an object in an image or in a video given a referring expression (Huang et al., 2018; Shi et al., 2019; Vasudevan et al., 2018; Li et al., 2023c). Deng et al. (2021); Yang et al. (2020); Luo et al. (2020); Yang et al. (2019); Kamath et al. (2021) propose end-to-end approaches for the spatial localization in images. For video, Chen et al. (2018; 2019); Zeng et al. (2020); Zhang et al. (2019; 2020); 

2 

Published as a conference paper at ICLR 2024 

He et al. (2019); Jang et al. (2023) focus on temporal localization given a natural language query. Methods like STVGBert (Su et al., 2021) and STCAT (Jin et al., 2022) perform spatiotemporal localization of a natural language query using a transformer-based model. Yang et al. (2022) focuses on action localization given a natural language query. Xiao et al. (2023) propose video grounding using Gaussian mask optimization. In contrast to these methods, the approach we introduce performs visual grounding in a pre-trained LM through end-to-end training using surrogate tasks. We show that this makes it possible to instill low-level visual capabilities as required for solving reasoning problems. 

**Attention-based models and visual reasoning.** Attention-based models have been studied extensively for visual reasoning Ding et al. (2021); Hu et al. (2017); Hudson & Manning (2018); Kamath et al. (2021); Mahajan & Roth (2020); Santoro et al. (2017). Recent advances include an object-centric encoder and a transformer reasoning module to solve RPM-like benchmarks (Mondal et al., 2023), multi-hop feature modulation (Strub et al., 2018) and cascaded modulation networks (Yao et al., 2018) that use a multi-step comprehension process, neural interpreters (Rahaman et al., 2021) that factorize inference in a self-attention network and ALANS learner (Zhang et al., 2022a) that combines abstract algebra and representation theory. Calibrating concepts and operations Li et al. (2021) enables neural symbolic models to capture underlying data characteristics and perform hierarchical inference. Again, the key difference in our work is that we instill the ability to extract object-centric information within the LM itself, using surrogate tasks, instead of resorting to separate object detection modules. 

## 3 LOOK, REMEMBER, REASON 

To allow for visual reasoning in videos using language models, we propose a novel _Look, Remember, Reason_ framework which grounds the model to the low-level structual and motion information in the visual input required for solving high-level reasoning tasks. Grounding is enabled through low-level visual surrogate tasks, _e.g_ . object recognition, tracking and re-identification. In the following, we first describe our auto-regressive LRR architecture including details of our two-stream video encoder, followed by our training pipeline with details of our surrogate tasks. 

## 3.1 AUTO-REGRESSIVE PIPELINE 

Our LRR model parameterized by _θ_ , as shown in Fig. 2, is based on a pre-trained LM backbone with a two-stream auto-regressive video encoder. Our LRR model receives the visual reasoning problem **Q** and, an interleaved sequence _I_ of video frames **V** = ( **v** 1 _, . . . ,_ **v** _Tv_ ) and tokenized text **S** = ( **s** 1 _, . . . ,_ **s** _Ts_ ) as input. **S** consists of low-level visual surrogate tasks and the answer indicated by <taskname> and <answer> respectively. The input video frame sequence **V** is encoded by our twostream video encoder. The LM backbone receives as input **Q** , and the interleaved sequence of encoded video frames **V** whose positions are indicated with <frame> special tokens and **S** . Conditioned on **Q** and **V** , we train our LRR model by maximizing log-likelihood of the text sequence **S** as, 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2024_Look,_Remember_and_Reason_Grounded_Reasoning_in_Videos_with_Language_Models_images/iclr_2024_Look,_Remember_and_Reason_Grounded_Reasoning_in_Videos_with_Language_Models.pdf-0003-07.png)


where, ( **v** 1 _, . . . ,_ **v** _tv_ ) is the interleaved video frame sub-sequence up to the text token **s** _ts_ . In Section 3.4, we describe the surrogate tasks included in **S** . The parameters of the LM backbone are initialized from pre-trained LMs, allowing us to exploit their existing high-level reasoning capabilities. We use LMs from the OPT family (Zhang et al., 2022b), but verified that similar performance can be achieved using other pre-trained models (Gao et al., 2021; Scao et al., 2022). While LM backbones are trained only on text, visual reasoning relies on information from the visual domain **V** . Therefore, to ground our LRR model, visual information in **V** needs to be mapped to the text-based representation space of the LM. The key challenge here is that in contrast to text tokens, videos are highly information dense. To address this, we develop a two-stream video encoder to extract information relevant for the visual reasoning problem at hand, such that responses to the visual reasoning problems are grounded in the fine-grained low-level details of the video. 

3 

Published as a conference paper at ICLR 2024 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2024_Look,_Remember_and_Reason_Grounded_Reasoning_in_Videos_with_Language_Models_images/iclr_2024_Look,_Remember_and_Reason_Grounded_Reasoning_in_Videos_with_Language_Models.pdf-0004-01.png)


**----- Start of picture text -----**<br>
v ( tv−⌧ ) v ( tv−⌧ +1) … v tv v Tv<br>Temporal Attention<br>Spatial Attention<br>z [0] tv<br>Temp-Attn Temp-Attn Temp-Attn z [1] tv Temp-Attn<br>Spatial-Attn Spatial-Attn Spatial-Attn Spatial-Attn<br>Temp-Attn Temp-Attn Temp-Attn z [L] tv Temp-Attn<br>At training time<br><frame> <frame> <frame> <taskname> <frame> <answer><br>Self-Attn Self-Attn Self-Attn h [1] tv Self-Attn Self-Attn Self-Attn<br>Cross-Attn Cross-Attn Cross-Attn Self-Attn Cross-Attn Self-Attn<br>…<br>Self-Attn Self-Attn Self-Attn h [M] tv Self-Attn Self-Attn Self-Attn<br>Surrogate Task s 1 Answer s ts<br>Encoder<br>blocks<br>L<br>Two-stream<br>Backbone blocks<br>LM M<br>{<br>{<br>**----- End of picture text -----**<br>


Figure 2. The architecture of our LRR model, highlighting the use of interleaved top-down cross-attention layers in between self-attention layers higher up in the hierarchy. 

## 3.2 TWO-STREAM VIDEO ENCODER 

Our autoregressive two-stream video encoder exploits divided space-time attention (Bertasius et al., 2021) and generates for each input video frame patch based embeddings that capture relevant lowlevel spatio-temporal information. Spatial attention captures structural information, _e.g_ ., object identities. Temporal attention autoregressively captures object motion and interaction information using the previous _τ_ frames as a buffer. As a first step, it converts each input video frame **v** _tv_ into _P_ flattened patches **v** _tv_ = _{_ **v** (1 _,tv_ ) _, . . . ,_ **v** ( _P,tv_ ) _}_ of size 16 _×_ 16 (Dosovitskiy et al., 2021) as shown in Fig. 2. Our LRR model applies a linear transformation (LN) on the input patches to generate the initial patch embeddings **z**[0] _tv_[=] _[ {]_ **[z]**[0] (1 _,tv_ ) _[, . . . ,]_ **[ z]**[0] ( _P,tv_ ) _[}]_[ as,] 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2024_Look,_Remember_and_Reason_Grounded_Reasoning_in_Videos_with_Language_Models_images/iclr_2024_Look,_Remember_and_Reason_Grounded_Reasoning_in_Videos_with_Language_Models.pdf-0004-05.png)


The spatiotemporal positional embedding **em** ( _p,tv_ ) is added to each patch to aid our video encoder extract spatiotemporal features from each patch in the input video frame. Next, these patch embeddings are processed by the _L_ blocks of our video encoder where at each block spatial or temporal attention is applied. At every block _ℓ ∈{_ 1 _, . . . , L}_ of our model, the embeddings are linearly mapped to produce the query, keys, and values required for the space- and time-based attention operations, 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2024_Look,_Remember_and_Reason_Grounded_Reasoning_in_Videos_with_Language_Models_images/iclr_2024_Look,_Remember_and_Reason_Grounded_Reasoning_in_Videos_with_Language_Models.pdf-0004-07.png)


In case of temporal attention, for the patch embedding **z** _[ℓ]_ ( _p,tv_ )[the attention is computed over patch] embeddings at the same spatial position _p_ in the _τ_ previous input video frames, **z** _[ℓ]_ ( _p,_ ( _tv−τ_ ))[to] **[ z]** _[ℓ]_ ( _p,tv_ )[.] For the patch highlighted in red in Fig. 2 the patches where temporal attention is applied in the previous _τ_ input frames is highlighted in cyan. The temporal attention vector ( _αT_ ) _[ℓ]_ ( _p,tv_ )[is given by,] 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2024_Look,_Remember_and_Reason_Grounded_Reasoning_in_Videos_with_Language_Models_images/iclr_2024_Look,_Remember_and_Reason_Grounded_Reasoning_in_Videos_with_Language_Models.pdf-0004-09.png)


where _dm_ is the dimensionality of the key, queries and values. In contrast, the spatial attention vector, ( _αS_ ) _[ℓ]_ ( _p,tv_ )[, is calculated over all patch embeddings of the current input video frame, highlighted in] green in Fig. 2, 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2024_Look,_Remember_and_Reason_Grounded_Reasoning_in_Videos_with_Language_Models_images/iclr_2024_Look,_Remember_and_Reason_Grounded_Reasoning_in_Videos_with_Language_Models.pdf-0004-11.png)


The patch embeddings **z** _[ℓ]_ ( _p,t_[+1] _v_ )[are the weighted sums of the values] **[ v]** ( _[ℓ] p,tv_ )[computed using the attention] weights ( _αT_ ) _[ℓ]_ ( _p,tv_ )[or][ (] _[α][S]_[)] _[ℓ]_ ( _p,tv_ )[in case of temporal or spatial attention, respectively.][Finally, the] patch embeddings **z** _[L] tv_[,][containing][localized][information][of][the][scene][structure][and][motion,][are] obtained as output from the last block _L_ at time-step _tv_ of the video encoder. 

4 

Published as a conference paper at ICLR 2024 

## 3.3 LM BACKBONE WITH TOP-DOWN CROSS ATTENTION 

Visual information encoded by the two-stream video encoder is mapped top-down to the LM backbone at positions indicated by the <frame> token at time-steps _tv_ (Fig. 2) using cross attention layers. This top-down mechanism with cross attention layers, trained using surrogate tasks (Section 3.4), helps us ground the LM backbone to the relevant low-level information from the input video frames. 

We modify the backbone LM architecture by inserting cross attention (CROSS-ATTN) layers (Alayrac et al., 2022) between self-attention (SELF-ATTN) layers (Fig. 2). The output patch embeddings from our video encoder **z** _[L] tv_[=] _[ {]_ **[z]** _[L]_ (1 _,tv_ ) _[, . . . ,]_ **[ z]** _[L]_ ( _p,tv_ ) _[}]_[ are transformed using multi-layer perceptrons] (MLP) at every cross-attention layer which help deal with the domain shift between the visual and textual domains. Next, we exploit the rich hierarchical representations in the hidden states **h** to _tv_ “look”= _{_ **h**[1] _t_ at _v[, . . . ,]_ the visual **[ h]** _t[M] v[}]_[of] input[the] and[LM] extract[at][timestep] visual _[t][v]_ information[,][where] _[M]_[is] in[the] a top-down[number][of] fashion[self-attention] for grounding.[layers,] Specifically, we use the representation **h**[ˆ] _[k] tv_[after the application of the][ (] _[k]_[ + 1)][th][ self-attention layer in] the LM. The hidden representation **h**[ˆ] _[k] tv_[encodes global semantics about the visual reasoning problem] **Q** and previous video frames upto _tv_ . It serves as the query vector after a linear transformation and the visual features **z** _[L] tv_[serves as the keys and values of the cross attention layer respectively,] 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2024_Look,_Remember_and_Reason_Grounded_Reasoning_in_Videos_with_Language_Models_images/iclr_2024_Look,_Remember_and_Reason_Grounded_Reasoning_in_Videos_with_Language_Models.pdf-0005-04.png)


where FFN is a feedforward layer (Vaswani et al., 2017).global semantics, it allows the LM backbone to extract relevant low-level spatiotemporal informationAs the hidden representation **h** _[k] tv_[encodes] (Eq. (8)) and is implicitly "remembered" within the context window of the LM. This information canin ˆ **z** _tv_ (Eq. (7)), when trained using our surrogate tasks. This information added to hidden states **h** _[k] tv_[+1] be aggregated by the LM to “reason” and arrive at the final answer. 

## 3.4 GROUNDING THROUGH SURROGATE TASKS 

To ground our LRR model to the relevant low-level information in the visual input, we utilize surrogate tasks during training. This is illustrated in Fig. 2 where the model is prompted using special tokens (<taskname>) to solve a surrogate task. We consider tasks like object recognition, tracking and re-identification as low-level. From Fig. 1, the ability to recognize the performed action rests upon grounding to the motion and interactions of the hand and the stapler. This makes recognition and tracking constituent low-level capabilities for recognizing the (high-level) compositional action. 

Our LRR model is highly flexible and can be prompted to solve a wide range of low-level surrogate tasks. Here, we focus primarily on the tasks of object recognition, localization, re-identification and tracking. These tasks are fundamental to solving a range of visual reasoning problems, and the requisite ground truth can be readily obtained using off-the-shelf vision models (Ren et al., 2017; Tang et al., 2017; Ye et al., 2022). Moreover, these tasks can be encoded as text to be processed by the LM backbone in the following general format: “<taskname> _object id_ 1 _, object class_ 1 _, object bounding box_ 1 _; . . . ; object idn, object classn, object bounding boxn_ ”, where there are _n_ objects in the scene. The <taskname> special token prompts the model to solve the surrogate task. We use <detect>, <re-identify> and <track> for detection, re-identification, and tracking surrogate tasks respectively. The _object id_ is an integer that is assigned by the model based on the spatiotemporal order of appearance and is crucial for re-identification and tracking as same object instance should be assigned the same id across video frames. The bounding box is described as a 4-tuple of the x and y coordinates of the upper left and lower right corners. We provide additional details in the experimental section (Section 4). 

Our LRR model is prompted to solve these surrogate tasks at randomly selected time steps during training. Random prompting forgoes the need to include surrogate tasks during inference time, leading to faster inference. Random prompting also benefits training efficiency in the case of long video sequences. To train our LRR model using maximum likelihood as described in Eq. (1). We construct the token sequence **S** in _I_ that includes surrogate tasks for each training example as follows: 

5 

Published as a conference paper at ICLR 2024 

Table 1. Evaluation results on the ACRE dataset, where, D.R. – Direct evidence, I.D. – Indirect evidence, S.O. – Screened-off, and B.B. – Backward Blocked subsets ( _[∗]_ represents results tested by ourselves.). 

|Model|Compositional<br>Systematic<br>All<br>D.R.<br>I.D.<br>S.O.<br>B.B.<br>All<br>D.R.<br>I.D.<br>S.O.<br>B.B|Compositional<br>Systematic<br>All<br>D.R.<br>I.D.<br>S.O.<br>B.B.<br>All<br>D.R.<br>I.D.<br>S.O.<br>B.B|
|---|---|---|
||||
|NS-OPT (Zhang et al., 2021)<br>IV-CL_†_ (Sun et al., 2023)<br>ALOE (Ding et al., 2021)<br>OpenFlamingo_∗_(Awadalla et al., 2023)|69.0<br>92.5<br>76.0<br>88.3<br>13.4<br>93.2<br>-<br>-<br>-<br>-<br>91.7<br>97.1<br>90.8<br>96.8<br>78.8<br>38.2<br>42.6<br>49.5<br>9.9<br>47.6|67.4<br>94.7<br>88.3<br>82.7<br>16.0<br>92.6<br>-<br>-<br>-<br>-<br>93.9<br>97.1<br>71.2<br>98.9<br>94.4<br>38.6<br>36.5<br>25.8<br>13.7<br>67.6|
|LRR (Ours)<br>**98.2**<br>**99.9**<br>**92.8**<br>**99.2**<br>**97.4**<br>**99.2**<br>**99.9**<br>**97.0**<br>**99.9**<br>**98.8**|||
||||
|LRR (w/o Surrogate tasks)<br>LRR (w/o “Two-stream” encoder)<br>LRR (from scratch)|38.1<br>38.4<br>30.2<br>26.2<br>50.0<br>92.2<br>98.6<br>77.7<br>96.2<br>85.8<br>87.7<br>96.9<br>57.7<br>91.1<br>84.8|36.5<br>35.2<br>28.1<br>20.0<br>55.3<br>92.8<br>98.5<br>76.2<br>96.8<br>89.4<br>88.0<br>96.3<br>58.5<br>91.5<br>86.9|



|||Contex|t Trials||||Query|Query|
|---|---|---|---|---|---|---|---|---|
||||||||||
|<re-identify><br>_1,medium_<br>_gray_<br>_rub_<br>_ber_<br>_cylin_<br>_der; 2,_<br>_medium_<br>_green_<br>_rubber_<br>_cylinder._<br><blicket>_on._<br>_. . ._||<re-identify><br>_gray_<br>_rub_<br>_ber_||<re-identify><br>_5,_<br>_medium_<br>_blue_<br>_metal_<br>_sphere;_<br>_6,_<br>_medium_<br>_purple_<br>_rubber_<br>_cylinder;_<br>_3,_<br>_medium_<br>_gray_<br>_rub_<br>_ber_<br>_sphere._<br><blicket>_off._|||<re-identify><br>_1,_<br>_medium_<br>_gray_<br>_rub_<br>_ber_<br>_cylin_<br>_der;_<br>_5,_<br>_medium_<br>_blue_<br>_metal_<br>_sphere._|_1,_<br>_medium_|
||||||||<answer>_no._||



Figure 3. Example solutions to surrogate tasks generated by our LRR model on ACRE. Re-identified objects across context trials are underlined in the same color. 

We first randomly select a subset of video frames ( **v** _t_ 1 _, . . . ,_ **v** _tk_ ) _∈_ **V** . We then update the token sequence **S** to include the surrogate tasks at these randomly selected time-steps, the beginning of which is marked by a task specific <taskname> special token, _e.g_ ., <detect>, <re-identify> and <track>. The surrogate task itself is added in **S** as described above. 

## 4 EXPERIMENTS 

We evaluate on visual reasoning tasks from: ACRE (Zhang et al., 2021), Something-Else Materzynska et al. (2020), CATER (Girdhar & Ramanan, 2020) and STAR (Wu et al., 2021). 

**Models and training details.** We fine-tune the pre-trained LM backbone along with the video encoder and cross-attention layers in our LRR model as the visual reasoning problems considered here are challenging and cannot be accurately solved by prompting state of the art LMs such as GPT-4 (Gendron et al., 2023). We focus on the OPT family of LMs Zhang et al. (2022b), particularly OPT-125M/350M/1.3B. We use 4 Nvidia A100 GPUs. Additional details in the appendix. 

## 4.1 ACRE 

The ACRE dataset (Zhang et al., 2021) evaluates the performance of vision systems on the problem of causal induction. Specifically, the dataset focuses on the problem of causal discovery using “blicket” detection tests, originally administered to children, using a blicket detector which activates when a blicket object is placed on it. The experiment involves a series of context trials, in which various (combinations of) objects are placed on the blicket detector, and subjects are shown whether the detector is activated. They are then asked which objects or (novel) combinations of objects would activate the machine. The ACRE dataset contains 6 context trials per example and 4 “blicket” detection tests based on the images. 

**Surrogate tasks.** The key low-level visual challenge in the ACRE dataset is to associate query objects to the context trials to detect whether the blicket machine is activated. Therefore, we consider the surrogate tasks of object recognition and re-identification. The solution to the surrogate task can be generated by the backbone LM in the following format: “<re-identify> _object id_ 1 _, object class_ 1 _; . . . ; object idn, object classn_ ” as shown in Fig. 3. We also found it helpful to introduce an additional surrogate task to identify the state of the blicket machine: “<blicket> _on/off_ ”. These surrogate tasks are introduced randomly during training with a probability of 30% after each context trial or query. 

6 

Published as a conference paper at ICLR 2024 

Table 2. Evaluation on the Something-Else dataset ( _[∗]_ represents results tested by ourselves.). 

|Table 2. Evaluation on the Something-Else dataset(_∗_represents results|tested byourselves.).|tested byourselves.).|
|---|---|---|
|Method|Base<br>Compositional<br>Top-1<br>Top-5<br>Top-1<br>Top-5||
||||
|STIN + OIE + NL (Materzynska et al., 2020)<br>Video-ChatGPT_∗_(Maaz et al., 2023)|78.1<br>94.5<br>52.6<br>75.8|56.2<br>81.3<br>38.6<br>67.8|
|LRR (Ours)<br>**80.2**<br>**96.1**<br>**62.0**<br>**86.3**|||
||||
|LRR (w/o Surrogate tasks)<br>LRR (w/o Two-stream encoder)|71.3<br>89.6<br>73.2<br>90.4|50.1<br>70.8<br>53.6<br>76.1|




![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2024_Look,_Remember_and_Reason_Grounded_Reasoning_in_Videos_with_Language_Models_images/iclr_2024_Look,_Remember_and_Reason_Grounded_Reasoning_in_Videos_with_Language_Models.pdf-0007-03.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2024_Look,_Remember_and_Reason_Grounded_Reasoning_in_Videos_with_Language_Models_images/iclr_2024_Look,_Remember_and_Reason_Grounded_Reasoning_in_Videos_with_Language_Models.pdf-0007-04.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2024_Look,_Remember_and_Reason_Grounded_Reasoning_in_Videos_with_Language_Models_images/iclr_2024_Look,_Remember_and_Reason_Grounded_Reasoning_in_Videos_with_Language_Models.pdf-0007-05.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2024_Look,_Remember_and_Reason_Grounded_Reasoning_in_Videos_with_Language_Models_images/iclr_2024_Look,_Remember_and_Reason_Grounded_Reasoning_in_Videos_with_Language_Models.pdf-0007-06.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2024_Look,_Remember_and_Reason_Grounded_Reasoning_in_Videos_with_Language_Models_images/iclr_2024_Look,_Remember_and_Reason_Grounded_Reasoning_in_Videos_with_Language_Models.pdf-0007-07.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2024_Look,_Remember_and_Reason_Grounded_Reasoning_in_Videos_with_Language_Models_images/iclr_2024_Look,_Remember_and_Reason_Grounded_Reasoning_in_Videos_with_Language_Models.pdf-0007-08.png)


Figure 4. Example solutions to surrogate task tracking generated by our LRR model on Something-Else. Bounding boxes belonging to the same track are highlighted using the same color. 

**Baselines and evaluation.** We base our LRR models on the OPT-125M backbone. We compare to the state of the art in Table 1, including the (powerful transformer based) multi-modal LLM OpenFlamingo (3B-mosaicml/mpt-1b-redpajama-200b-dolly; Awadalla et al. (2023)) as it has shown success in reasoning problems involving multiple images and videos. To highlight the importance of our rationale generation process, we consider a baseline without the surrogate re-identification task: LRR (w/o Surrogate tasks). Although there is no motion or object interactions in ACRE, we still found it helpful to use our two-stream encoder as our temporal attention mechanism can be used for re-identification of (previously observed) objects. To highlight this, we also include a LRR (w/o Two-stream encoder) ablation that uses a plain single stream ViT encoder. Finally, to highlight the importance of using pre-trained LM backbones we include a LRR model that is trained from scratch with the same OPT-125M LM architecture. 

Our LRR model outperforms the state of the art powerful transformer based ALOE (Ding et al., 2021), by a large margin of 6.5% and 5.3% on the compositional and systematic splits respectively. This shows the advantage of our end-to-end LRR model with surrogate tasks over explicit object centric input representations used by ALOE. Similarly, we observe a significant performance gain of 60% over the powerful transformer based multi-modal LLM OpenFlamingo (Awadalla et al., 2023) which highlights the importance of our LRR framework to ground our model to the relevant low-level details – highlighted by the weak performance of the LRR (w/o Surrogate tasks) ablation. We also see that our two-stream encoder improves performance over the single-stream model. The weak performance of the LRR (from scratch) ablation shows that it is crucial to start from a pre-trained LM backbone to exploit its high level reasoning abilities. Finally, although the results in Table 1 use the OPT-125M backbone in our LRR model, we obtain an identical 98.2% and 99.0% accuracy on the compositional and systematic splits respectively with an OPT-1.3B backbone, indicating that our _look, remember, reason_ framework is applicable across LM backbone sizes. 

## 4.2 SOMETHING-ELSE 

The complex real-world Something-Else dataset (Materzynska et al., 2020) focuses on compositional action recognition. It is based on the Something-Something dataset (Goyal et al., 2017), which it extends to measure compositional generalization. The compositional split of the dataset breaks each action down into a combination of a verb, a subject, and one or more objects. This makes it possible to benchmark performance on novel verb and object combinations which are not present in the training set, requiring models to have a fine-grained understanding of motion instead of merely correlating actions with object types. 

7 

Published as a conference paper at ICLR 2024 

Table 3. Evaluation on the CATER dataset ( _[†]_ results reported only for static camera). 

|Method|Static Camera<br>Moving Camera<br>Top-1(_↑_)<br>Top-5(_↑_)<br>L1(grid;_↓_)<br>Top-1(_↑_)<br>Top-5(_↑_)<br>L1(grid;_↓_)|Static Camera<br>Moving Camera<br>Top-1(_↑_)<br>Top-5(_↑_)<br>L1(grid;_↓_)<br>Top-1(_↑_)<br>Top-5(_↑_)<br>L1(grid;_↓_)|
|---|---|---|
||||
|R3D + NL LSTM Girdhar & Ramanan (2020)<br>ALOE Ding et al. (2021)|46.2<br>69.9<br>1.5<br>74.0<br>94.0<br>0.44|38.6<br>70.2<br>1.5<br>59.7<br>90.1<br>0.69|
||||
|IV-CL_†_ (Sun et al., 2023)<br>OPNet_†_ (Shamsian et al., 2020)<br>Hopper_†_ (Zhou et al., 2021)<br>TFC V3D_†_ (Zhang, 2022)<br>LRR (Ours)|70.1<br>88.3<br>-|-<br>-<br>-|
||74.8<br>-<br>0.54<br>73.2<br>93.8<br>0.85<br>79.7<br>95.5<br>0.47<br>**84.1**<br>**97.2**<br>**0.34**|-<br>-<br>-<br>-<br>-<br>-<br>-<br>-<br>-<br>**80.4**<br>**96.7**<br>**0.42**|
||||
|LRR (w/o Surrogate tasks)<br>LRR (w/o Two-stream encoder)|68.5<br>88.7<br>0.65<br>81.4<br>97.2<br>0.44|62.7<br>86.7<br>0.77<br>75.6<br>96.6<br>0.53|
|13,<br>18,<br>18,<br>21<br>21,<br>33,<br>3,<br>21<br>21,<br>33,<br>3,<br>21<br>14,<br>32,<br>3,<br>14|||



Figure 5. Example answers to the tracking surrogate task generated by our LRR model on CATER. Our LRR model is prompted with the “<track>” special token to solve the tracking surrogate task at randomly selected time-steps during training. Object tracks are over the 6 _×_ 6 grid on the surface and are highlighted in color. 

**Surrogate tasks.** In this task, we employ tracking as a surrogate task to support the model’s ability to capture motion and object interactions. Note that, as object classes are not important for compositional action recognition, the solution to the surrogate task can be generated in the following format: “<track> _object id_ 1 _, object bounding box_ 1 _; . . . ; object idn, object bounding boxn_ ” as shown in Fig. 4. Surrogate tasks are introduced randomly during training with a probability of 30% after each input video frame. 

**Baselines and evaluation.** We base our LRR model on the OPT-125M LM backbone and compare to several baselines and ablations in Table 2. We report results on both the base split and the compositional split with novel action-object combinations. We consider the state-of-the-art STIN + OIE + NL models from Materzynska et al. (2020) along with the state-of-the-art powerful transformer based multi-modal LLM Video-ChatGPT (Maaz et al., 2023) model as baselines. The Video-ChatGPT model uses the 7B parameter Vicuna model (Chiang et al., 2023) as an LM backbone and a CLIP (Radford et al., 2021) based video encoder. To highlight the importance of our surrogate tasks, we consider an ablation without the surrogate tracking task: LRR (w/o Surrogate Tasks). To highlight the importance of our two-stream video encoder we consider an ablation with a single stream ViT model that cannot capture motion features (w/o Two-stream encoder). 

Our LRR model also outperforms the STIN + OIE + NL baseline by 2.1% and 5.8% Top-1 accuracy on the base and compositional split respectively, which highlights the reasoning ability of grounded LM based architectures. The results also show that the performance of the state of the art multi-modal LM: Video-ChatGPT (finetuned), lags very significantly by 23.4% behind our LRR model. This is because of the CLIP based video encoder in Video-ChatGPT is not well suited for capturing motion features and due to a lack of grounding. The importance of motion features is underscored by the lacking performance of the plain ViT based (w/o Two-stream encoder) ablation: a Top-1 accuracy drop of 8.4% on the compositional split. The importance of grounding is highlighted through our ablation without surrogate tasks: a Top-1 accuracy drop of 11.9% on the compositional split. We illustrate successful tracking of objects in complex real-world scenarios by our LRR model in Fig. 4 and in Appendix B. Note that, although the results in Table 2 uses the OPT-125M backbone, we obtain an identical 61.3% Top-1 and 85.9% Top-5 accuracy on the compositional split with an OPT-1.3B backbone, indicating that our _look, remember, reason_ framework is applicable across LM sizes. 

## 4.3 CATER 

The CATER (Compositional Actions and TEmporal Reasoning) dataset is designed to test the ability to recognize compositions of object movements that require long-term temporal reasoning. Similar to Ding et al. (2021), we focus on the hardest task from the CATER dataset, _i.e_ ., adversarial target 

8 

Published as a conference paper at ICLR 2024 

Table 4. Evaluation of our LRR model on STAR (validation set). 

|Method|Int.<br>Seq.<br>Pre.<br>Fea.|Overall_↑_|
|---|---|---|
||||
|Internvideo (Wang et al., 2022)<br>BLIP-2 (Li et al., 2023a)<br>SeViLA (Yu et al., 2023)|62.7<br>65.6<br>54.9<br>51.9<br>65.4<br>69.0<br>59.7<br>54.2<br>63.7<br>70.4<br>63.1<br>62.4|58.7<br>62.0<br>64.9|
||||
|LRR (Ours)|**73.7**<br>**71.0**<br>**71.3**<br>**65.1**|**70.5**|
||||
|LRR (w/o Surrogate tasks)|54.5<br>48.7<br>44.3<br>45.5|48.2|



tracking under occlusion and containment. This task amounts to predicting the position of a special object, referred to as "snitch”, at the end of each video sequence. This is challenging as the position of the “snitch” can be occluded or (recursively) contained within other objects at the end of the sequence. This task is posed as a classification problem over a 6 _×_ 6 grid. There are two splits of the dataset: static and moving camera – the latter of which is more challenging as the grid positions can only be predicted by capturing long-term spatiotemporal correlations. 

**Surrogate tasks.** We employ (multi-target) tracking as a surrogate task for which the solution can be expressed in the following format: “<track> _object id_ 1 _, object grid position_ 1 _; . . . ; object idn, object grid positionn_ ” as shown in Fig. 5. The tracking task includes the medium and large cones in the scene, as these objects can occlude the snitch. We use grid positions instead of bounding boxes (unlike Something-Else) because the final goal is to predict the grid position of the snitch. Surrogate tasks are introduced randomly during training with a probability of 30% after each input frame. 

**Baselines and evaluation.** Our LRR model uses the OPT-125M backbone. We report the results in Table 3. To highlight the importance of our surrogate tasks, we consider an ablation without the surrogate tracking task (w/o Surrogate tasks) and to highlight the importance of our Two-stream video encoder we include an ablation with single-stream ViT encoder (w/o Two-stream Encoder). We train jointly on both the static and moving camera splits, similar to ALOE Ding et al. (2021). 

The state-of-the-art powerful transformer-based IV-CL (Sun et al., 2023), OPNet (Shamsian et al., 2020), Hopper (Zhou et al., 2021), TFC V3D Depthwise (Zhang, 2022) and Loci (Traub et al., 2023) report results only on the static camera split. Loci (Traub et al., 2023) reports an impressive 90.7% accuracy on the static camera split, but it is not applicable to the moving camera split due to its static background and camera model. Our LRR model outperforms TFC V3D Depthwise (Zhang, 2022) model on the static camera by 4.4% and ALOE (Ding et al., 2021) on the challenging moving camera split by 20.7%. The large performance gain over the LRR (w/o Surrogate tasks) baseline shows the advantage of surrogate tracking tasks, without which the model is not grounded to the motion of the cones and hence fails in cases where the snitch is contained by the cones. Qualitative examples in Fig. 5 illustrates that our model is able to successfully track objects in cases of recursive containment and is robust to moving cameras. Finally, the performance advantage over the LRR (w/o Two-stream encoder) confirms that our LRR model is able to better capture the motion of the objects. 

## 4.4 STAR 

STAR (Wu et al., 2021) is a situated spatio-temporal reasoning benchmark, consisting of questions built upon real-world videos associated with human actions and interactions. The STAR benchmark thus requires an understanding of low-level human motion, actions and object interactions. However, the STAR dataset does not contain dense object annotations, _e.g_ ., object tracks, unlike the CATER and Something-Else datasets. It only contains sparse spatio-temporal scene graphs which cover selected keyframes. We introduce object recognition surrogate tasks to localize objects on these keyframes based on the scene graphs. Furthermore, we jointly train our LRR model to recognize actions from the Kinetics (Kay et al., 2017) and Moments in Time (Monfort et al., 2020) dataset, as well as on surrogate tracking tasks from the Something-Else dataset (see Section 4.2) to improve the understanding of object interactions. We additionally regularize on text data. 

Our LRR model with an OPT-350M LM backbone achieves 70.5% overall accuracy and significantly outperforms powerful trasnformer-based state of the art models, such as SeViLA (Yu et al., 2023), Internvideo (Wang et al., 2022) and BLIP-2 (Li et al., 2023a) as shown in Table 4. Following SeViLA (Yu et al., 2023), we report results on the validation set. Note that methods such as SeViLA are trained on a much larger set of image/video - text pairs and, use the 4.1B parameter BLIP as a video 

9 

Published as a conference paper at ICLR 2024 

encoder and the 3B parameter Flan-T5 XL as an language model ( _c.f_ . Section 4.4 in Yu et al. (2023)). In contrast, our LRR model contains fewer parameters and is trained on a much smaller training set. Curicially, our training set includes carefully selected surrogate tasks to endow the model with the requisite low-level visual capabilities. To illustrate this, we include a ablation of our LRR model without any surrogate tasks (w/o Surrogate tasks). 

In addition to the results on the STAR validation set presented above we also evaluate our LRR model on the STAR challenge leaderboard. The results can be found: here, where our LRR model is ranked 1[st] as of January 2024. 

## 5 CONCLUSION 

We show that off-the-shelf LMs can solve complex visual reasoning tasks on videos using our LRR framework. In this framework, LMs equipped with a two-stream video encoder are supervised and grounded using surrogate tasks. Grounding ensures that the LM can utilize relevant low-level visual cues from in the input video to make predictions. Grounding predictions to low-level visual cues combined with the high-level reasoning ability of the LM is the key to the success of the model. Our LRR model outperforms the state of the art by 6.5% and 5.3% on the compositional and systematic splits of the ACRE dataset; by 5.8% Top-1 accuracy on the compositional split of the Something-Else dataset; by 4.4% Top-1 accuracy on static camera and 20.7% Top-1 accuracy on moving camera splits of the CATER dataset; and by 5.6% overall accuracy on STAR. 

## REFERENCES 

- Jean-Baptiste Alayrac, Jeff Donahue, Pauline Luc, Antoine Miech, Iain Barr, Yana Hasson, Karel Lenc, Arthur Mensch, Katherine Millican, Malcolm Reynolds, Roman Ring, Eliza Rutherford, Serkan Cabi, Tengda Han, Zhitao Gong, Sina Samangooei, Marianne Monteiro, Jacob L. Menick, Sebastian Borgeaud, Andy Brock, Aida Nematzadeh, Sahand Sharifzadeh, Mikolaj Binkowski, Ricardo Barreira, Oriol Vinyals, Andrew Zisserman, and Karén Simonyan. Flamingo: a visual language model for few-shot learning. In _NeurIPS_ , 2022. 

- Anas Awadalla, Irena Gao, Josh Gardner, Jack Hessel, Yusuf Hanafy, Wanrong Zhu, Kalyani Marathe, Yonatan Bitton, Samir Yitzhak Gadre, Shiori Sagawa, Jenia Jitsev, Simon Kornblith, Pang Wei Koh, Gabriel Ilharco, Mitchell Wortsman, and Ludwig Schmidt. Openflamingo: An open-source framework for training large autoregressive vision-language models. _CoRR_ , abs/2308.01390, 2023. 

- Jinze Bai, Shuai Bai, Shusheng Yang, Shijie Wang, Sinan Tan, Peng Wang, Junyang Lin, Chang Zhou, and Jingren Zhou. Qwen-vl: A frontier large vision-language model with versatile abilities. _CoRR_ , abs/2308.12966, 2023. 

- Gedas Bertasius, Heng Wang, and Lorenzo Torresani. Is space-time attention all you need for video understanding? In Marina Meila and Tong Zhang (eds.), _ICML_ , 2021. 

- Jingyuan Chen, Xinpeng Chen, Lin Ma, Zequn Jie, and Tat-Seng Chua. Temporally grounding natural sentence in video. In Ellen Riloff, David Chiang, Julia Hockenmaier, and Jun’ichi Tsujii (eds.), _ACL_ , 2018. 

- Jingyuan Chen, Lin Ma, Xinpeng Chen, Zequn Jie, and Jiebo Luo. Localizing natural language in videos. In _AAAI_ , 2019. 

- Ting Chen, Saurabh Saxena, Lala Li, Tsung-Yi Lin, David J. Fleet, and Geoffrey E. Hinton. A unified sequence interface for vision tasks. In _NeurIPS_ , 2022. 

- Wei-Lin Chiang, Zhuohan Li, Zi Lin, Ying Sheng, Zhanghao Wu, Hao Zhang, Lianmin Zheng, Siyuan Zhuang, Yonghao Zhuang, Joseph E. Gonzalez, Ion Stoica, and Eric P. Xing. Vicuna: An open-source chatbot impressing gpt-4 with 90%* chatgpt quality, March 2023. URL https: //lmsys.org/blog/2023-03-30-vicuna/. 

- Karl Cobbe, Vineet Kosaraju, Mohammad Bavarian, Jacob Hilton, Reiichiro Nakano, Christopher Hesse, and John Schulman. Training verifiers to solve math word problems. _CoRR_ , abs/2110.14168, 2021. 

10 

Published as a conference paper at ICLR 2024 

Ekin D. Cubuk, Barret Zoph, Jonathon Shlens, and Quoc V. Le. Randaugment: Practical automated data augmentation with a reduced search space. In _CVPR Workshops_ , 2020. 

- Wenliang Dai, Junnan Li, Dongxu Li, Anthony Meng Huat Tiong, Junqi Zhao, Weisheng Wang, Boyang Li, Pascale Fung, and Steven C. H. Hoi. Instructblip: Towards general-purpose visionlanguage models with instruction tuning. In _NeurIPS_ , 2023. 

- Jiajun Deng, Zhengyuan Yang, Tianlang Chen, Wengang Zhou, and Houqiang Li. Transvg: End-toend visual grounding with transformers. In _ICCV_ , 2021. 

- David Ding, Felix Hill, Adam Santoro, Malcolm Reynolds, and Matt M. Botvinick. Attention over learned object embeddings enables complex visual reasoning. In _NeurIPS_ , 2021. 

- Alexey Dosovitskiy, Lucas Beyer, Alexander Kolesnikov, Dirk Weissenborn, Xiaohua Zhai, Thomas Unterthiner, Mostafa Dehghani, Matthias Minderer, Georg Heigold, Sylvain Gelly, Jakob Uszkoreit, and Neil Houlsby. An image is worth 16x16 words: Transformers for image recognition at scale. In _ICLR_ , 2021. 

- Danny Driess, Fei Xia, Mehdi S. M. Sajjadi, Corey Lynch, Aakanksha Chowdhery, Brian Ichter, Ayzaan Wahid, Jonathan Tompson, Quan Vuong, Tianhe Yu, Wenlong Huang, Yevgen Chebotar, Pierre Sermanet, Daniel Duckworth, Sergey Levine, Vincent Vanhoucke, Karol Hausman, Marc Toussaint, Klaus Greff, Andy Zeng, Igor Mordatch, and Pete Florence. Palm-e: An embodied multimodal language model. _CoRR_ , abs/2303.03378, 2023. 

- Leo Gao, Stella Biderman, Sid Black, Laurence Golding, Travis Hoppe, Charles Foster, Jason Phang, Horace He, Anish Thite, Noa Nabeshima, Shawn Presser, and Connor Leahy. The pile: An 800gb dataset of diverse text for language modeling. _CoRR_ , abs/2101.00027, 2021. 

- Gaël Gendron, Qiming Bao, Michael Witbrock, and Gillian Dobbie. Large language models are not abstract reasoners. _CoRR_ , abs/2305.19555, 2023. 

Rohit Girdhar and Deva Ramanan. CATER: A diagnostic dataset for compositional actions & temporal reasoning. In _ICLR_ , 2020. 

- Raghav Goyal, Samira Ebrahimi Kahou, Vincent Michalski, Joanna Materzynska, Susanne Westphal, Heuna Kim, Valentin Haenel, Ingo Fründ, Peter Yianilos, Moritz Mueller-Freitag, Florian Hoppe, Christian Thurau, Ingo Bax, and Roland Memisevic. The "something something" video database for learning and evaluating visual common sense. In _ICCV_ , 2017. 

- Tanmay Gupta and Aniruddha Kembhavi. Visual programming: Compositional visual reasoning without training. _CoRR_ , abs/2211.11559, 2022. 

- Dongliang He, Xiang Zhao, Jizhou Huang, Fu Li, Xiao Liu, and Shilei Wen. Read, watch, and move: Reinforcement learning for temporally grounding natural language descriptions in videos. In _AAAI_ , 2019. 

- Ronghang Hu, Jacob Andreas, Marcus Rohrbach, Trevor Darrell, and Kate Saenko. Learning to reason: End-to-end module networks for visual question answering. In _ICCV_ , 2017. 

- De-An Huang, Shyamal Buch, Lucio M. Dery, Animesh Garg, Li Fei-Fei, and Juan Carlos Niebles. Finding "it": Weakly-supervised reference-aware visual grounding in instructional videos. In _CVPR_ , 2018. 

- Drew A. Hudson and Christopher D. Manning. Compositional attention networks for machine reasoning. In _ICLR_ , 2018. 

- Jinhyun Jang, Jungin Park, Jin Kim, Hyeongjun Kwon, and Kwanghoon Sohn. Knowing where to focus: Event-aware transformer for video grounding. _CoRR_ , abs/2308.06947, 2023. 

- Yang Jin, Yongzhi Li, Zehuan Yuan, and Yadong Mu. Embracing consistency: A one-stage approach for spatio-temporal video grounding. In _NeurIPS_ , 2022. 

- Aishwarya Kamath, Mannat Singh, Yann LeCun, Gabriel Synnaeve, Ishan Misra, and Nicolas Carion. MDETR - modulated detection for end-to-end multi-modal understanding. In _ICCV_ , 2021. 

11 

Published as a conference paper at ICLR 2024 

Will Kay, João Carreira, Karen Simonyan, Brian Zhang, Chloe Hillier, Sudheendra Vijayanarasimhan, Fabio Viola, Tim Green, Trevor Back, Paul Natsev, Mustafa Suleyman, and Andrew Zisserman. The kinetics human action video dataset. _CoRR_ , abs/1705.06950, 2017. 

- Jing Yu Koh, Ruslan Salakhutdinov, and Daniel Fried. Grounding language models to images for multimodal generation. _CoRR_ , abs/2301.13823, 2023. 

Junnan Li, Dongxu Li, Silvio Savarese, and Steven C. H. Hoi. BLIP-2: bootstrapping language-image pre-training with frozen image encoders and large language models. In _ICML_ , 2023a. 

- Kunchang Li, Yinan He, Yi Wang, Yizhuo Li, Wenhai Wang, Ping Luo, Yali Wang, Limin Wang, and Yu Qiao. Videochat: Chat-centric video understanding. _CoRR_ , abs/2305.06355, 2023b. 

- Mengze Li, Han Wang, Wenqiao Zhang, Jiaxu Miao, Zhou Zhao, Shengyu Zhang, Wei Ji, and Fei Wu. WINNER: weakly-supervised hierarchical decomposition and alignment for spatio-temporal video grounding. In _CVPR_ , 2023c. 

- Zhuowan Li, Elias Stengel-Eskin, Yixiao Zhang, Cihang Xie, Quan Tran, Benjamin Van Durme, and Alan L. Yuille. Calibrating concepts and operations: Towards symbolic reasoning on real images. In _ICCV_ , 2021. 

Ilya Loshchilov and Frank Hutter. Decoupled weight decay regularization. In _ICLR_ , 2019. 

- Pan Lu, Baolin Peng, Hao Cheng, Michel Galley, Kai-Wei Chang, Ying Nian Wu, Song-Chun Zhu, and Jianfeng Gao. Chameleon: Plug-and-play compositional reasoning with large language models. _CoRR_ , abs/2304.09842, 2023. 

- Gen Luo, Yiyi Zhou, Xiaoshuai Sun, Liujuan Cao, Chenglin Wu, Cheng Deng, and Rongrong Ji. Multi-task collaborative network for joint referring expression comprehension and segmentation. In _CVPR_ , 2020. 

- Ruipu Luo, Ziwang Zhao, Min Yang, Junwei Dong, Minghui Qiu, Pengcheng Lu, Tao Wang, and Zhongyu Wei. Valley: Video assistant with large language model enhanced ability. _CoRR_ , abs/2306.07207, 2023. 

- Muhammad Maaz, Hanoona Abdul Rasheed, Salman H. Khan, and Fahad Shahbaz Khan. Videochatgpt: Towards detailed video understanding via large vision and language models. _CoRR_ , abs/2306.05424, 2023. 

Shweta Mahajan and Stefan Roth. Diverse image captioning with context-object split latent spaces. In _NeurIPS_ , 2020. 

- Joanna Materzynska, Tete Xiao, Roei Herzig, Huijuan Xu, Xiaolong Wang, and Trevor Darrell. Something-else: Compositional action recognition with spatial-temporal interaction networks. In _CVPR_ , 2020. 

- Shanka Subhra Mondal, Taylor Webb, and Jonathan D. Cohen. Learning to reason over visual objects. _CoRR_ , abs/2303.02260, 2023. 

- Mathew Monfort, Carl Vondrick, Aude Oliva, Alex Andonian, Bolei Zhou, Kandan Ramakrishnan, Sarah Adel Bargal, Tom Yan, Lisa M. Brown, Quanfu Fan, and Dan Gutfreund. Moments in time dataset: One million videos for event understanding. _IEEE Trans. Pattern Anal. Mach. Intell._ , 42 (2):502–508, 2020. 

OpenAI. GPT-4 technical report. _CoRR_ , abs/2303.08774, 2023. 

- Zhiliang Peng, Wenhui Wang, Li Dong, Yaru Hao, Shaohan Huang, Shuming Ma, and Furu Wei. Kosmos-2: Grounding multimodal large language models to the world. _CoRR_ , abs/2306.14824, 2023. 

- Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, Gretchen Krueger, and Ilya Sutskever. Learning transferable visual models from natural language supervision. In _ICML_ , 2021. 

12 

Published as a conference paper at ICLR 2024 

- Nasim Rahaman, Muhammad Waleed Gondal, Shruti Joshi, Peter V. Gehler, Yoshua Bengio, Francesco Locatello, and Bernhard Schölkopf. Dynamic inference with neural interpreters. In _NeurIPS_ , 2021. 

- Shaoqing Ren, Kaiming He, Ross B. Girshick, and Jian Sun. Faster R-CNN: towards real-time object detection with region proposal networks. _IEEE Trans. Pattern Anal. Mach. Intell._ , 39(6): 1137–1149, 2017. 

- Adam Santoro, David Raposo, David G. T. Barrett, Mateusz Malinowski, Razvan Pascanu, Peter W. Battaglia, and Tim Lillicrap. A simple neural network module for relational reasoning. In _NeurIPS_ , 2017. 

- Teven Le Scao, Angela Fan, Christopher Akiki, Ellie Pavlick, Suzana Ilic, Daniel Hesslow, Roman Castagné, Alexandra Sasha Luccioni, François Yvon, Matthias Gallé, Jonathan Tow, Alexander M. Rush, Stella Biderman, Albert Webson, Pawan Sasanka Ammanamanchi, Thomas Wang, Benoît Sagot, Niklas Muennighoff, Albert Villanova del Moral, Olatunji Ruwase, Rachel Bawden, Stas Bekman, Angelina McMillan-Major, Iz Beltagy, Huu Nguyen, Lucile Saulnier, Samson Tan, Pedro Ortiz Suarez, Victor Sanh, Hugo Laurençon, Yacine Jernite, Julien Launay, Margaret Mitchell, Colin Raffel, Aaron Gokaslan, Adi Simhi, Aitor Soroa, Alham Fikri Aji, Amit Alfassy, Anna Rogers, Ariel Kreisberg Nitzav, Canwen Xu, Chenghao Mou, Chris Emezue, Christopher Klamm, Colin Leong, Daniel van Strien, David Ifeoluwa Adelani, and et al. BLOOM: A 176bparameter open-access multilingual language model. _CoRR_ , abs/2211.05100, 2022. 

- Aviv Shamsian, Ofri Kleinfeld, Amir Globerson, and Gal Chechik. Learning object permanence from video. In _ECCV_ , 2020. 

- Jing Shi, Jia Xu, Boqing Gong, and Chenliang Xu. Not all frames are equal: Weakly-supervised video grounding wit contextual similarity and visual clustering losses. In _CVPR_ , 2019. 

- Karen Simonyan and Andrew Zisserman. Two-stream convolutional networks for action recognition in videos. In Zoubin Ghahramani, Max Welling, Corinna Cortes, Neil D. Lawrence, and Kilian Q. Weinberger (eds.), _NeurIPS_ , 2014. 

- Florian Strub, Mathieu Seurin, Ethan Perez, Harm de Vries, Jérémie Mary, Philippe Preux, Aaron C. Courville, and Olivier Pietquin. Visual reasoning with multi-hop feature modulation. In _ECCV_ , 2018. 

- Rui Su, Qian Yu, and Dong Xu. Stvgbert: A visual-linguistic transformer based framework for spatio-temporal video grounding. In _ICCV_ , 2021. 

- Chen Sun, Calvin Luo, Xingyi Zhou, Anurag Arnab, and Cordelia Schmid. Does visual pretraining help end-to-end reasoning? _CoRR_ , abs/2307.08506, 2023. 

- Dídac Surís, Sachit Menon, and Carl Vondrick. Vipergpt: Visual inference via python execution for reasoning. _CoRR_ , abs/2303.08128, 2023. 

- Siyu Tang, Mykhaylo Andriluka, Bjoern Andres, and Bernt Schiele. Multiple people tracking by lifted multicut and person re-identification. In _CVPR_ , 2017. 

- Hugo Touvron, Thibaut Lavril, Gautier Izacard, Xavier Martinet, Marie-Anne Lachaux, Timothée Lacroix, Baptiste Rozière, Naman Goyal, Eric Hambro, Faisal Azhar, Aurélien Rodriguez, Armand Joulin, Edouard Grave, and Guillaume Lample. Llama: Open and efficient foundation language models. _CoRR_ , abs/2302.13971, 2023. 

- Manuel Traub, Sebastian Otte, Tobias Menge, Matthias Karlbauer, Jannik Thümmel, and Martin V. Butz. Learning what and where - unsupervised disentangling location and identity tracking. In _ICLR_ , 2023. 

- Arun Balajee Vasudevan, Dengxin Dai, and Luc Van Gool. Object referring in videos with language and human gaze. In _CVPR_ , 2018. 

- Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Lukasz Kaiser, and Illia Polosukhin. Attention is all you need. In _NeurIPS_ , 2017. 

13 

Published as a conference paper at ICLR 2024 

- Yi Wang, Kunchang Li, Yizhuo Li, Yinan He, Bingkun Huang, Zhiyu Zhao, Hongjie Zhang, Jilan Xu, Yi Liu, Zun Wang, Sen Xing, Guo Chen, Junting Pan, Jiashuo Yu, Yali Wang, Limin Wang, and Yu Qiao. Internvideo: General video foundation models via generative and discriminative learning. _CoRR_ , abs/2212.03191, 2022. 

- Jason Wei, Xuezhi Wang, Dale Schuurmans, Maarten Bosma, brian ichter, Fei Xia, Ed H. Chi, Quoc V Le, and Denny Zhou. Chain of thought prompting elicits reasoning in large language models. In _NeurIPS_ , 2022. 

- Bo Wu, Shoubin Yu, Zhenfang Chen, Josh Tenenbaum, and Chuang Gan. STAR: A benchmark for situated reasoning in real-world videos. In Joaquin Vanschoren and Sai-Kit Yeung (eds.), _NeurIPS_ , 2021. 

- Junbin Xiao, Angela Yao, Yicong Li, and Tat-Seng Chua. Can I trust your answer? visually grounded video question answering. _CoRR_ , abs/2309.01327, 2023. 

- Antoine Yang, Antoine Miech, Josef Sivic, Ivan Laptev, and Cordelia Schmid. Tubedetr: Spatiotemporal video grounding with transformers. In _CVPR_ , 2022. 

- Zhengyuan Yang, Boqing Gong, Liwei Wang, Wenbing Huang, Dong Yu, and Jiebo Luo. A fast and accurate one-stage approach to visual grounding. In _ICCV_ , 2019. 

- Zhengyuan Yang, Tianlang Chen, Liwei Wang, and Jiebo Luo. Improving one-stage visual grounding by recursive sub-query construction. In Andrea Vedaldi, Horst Bischof, Thomas Brox, and Jan-Michael Frahm (eds.), _ECCV_ , 2020. 

- Yiqun Yao, Jiaming Xu, Feng Wang, and Bo Xu. Cascaded mutual modulation for visual reasoning. In _EMNLP_ , 2018. 

- Mang Ye, Jianbing Shen, Gaojie Lin, Tao Xiang, Ling Shao, and Steven C. H. Hoi. Deep learning for person re-identification: A survey and outlook. _IEEE Trans. Pattern Anal. Mach. Intell._ , 44(6): 2872–2893, 2022. 

- Shoubin Yu, Jaemin Cho, Prateek Yadav, and Mohit Bansal. Self-chained image-language model for video localization and question answering. _CoRR_ , abs/2305.06988, 2023. 

- Runhao Zeng, Haoming Xu, Wenbing Huang, Peihao Chen, Mingkui Tan, and Chuang Gan. Dense regression network for video grounding. In _CVPR_ , 2020. 

- Chi Zhang, Baoxiong Jia, Mark Edmonds, Song-Chun Zhu, and Yixin Zhu. ACRE: abstract causal reasoning beyond covariation. In _CVPR_ , 2021. 

- Chi Zhang, Sirui Xie, Baoxiong Jia, Ying Nian Wu, Song-Chun Zhu, and Yixin Zhu. Learning algebraic representation for systematic generalization in abstract reasoning. In _ECCV_ , 2022a. 

- Da Zhang, Xiyang Dai, Xin Wang, Yuan-Fang Wang, and Larry S. Davis. MAN: moment alignment network for natural language moment retrieval via iterative graph adjustment. In _CVPR_ , 2019. 

- Hao Zhang, Aixin Sun, Wei Jing, and Joey Tianyi Zhou. Span-based localizing network for natural language video localization. In Dan Jurafsky, Joyce Chai, Natalie Schluter, and Joel R. Tetreault (eds.), _ACL_ , 2020. 

- Renrui Zhang, Jiaming Han, Aojun Zhou, Xiangfei Hu, Shilin Yan, Pan Lu, Hongsheng Li, Peng Gao, and Yu Qiao. Llama-adapter: Efficient fine-tuning of language models with zero-init attention. _CoRR_ , abs/2303.16199, 2023a. 

- Shiwen Zhang. Tfcnet: Temporal fully connected networks for static unbiased temporal reasoning. _CoRR_ , 2022. 

- Susan Zhang, Stephen Roller, Naman Goyal, Mikel Artetxe, Moya Chen, Shuohui Chen, Christopher Dewan, Mona T. Diab, Xian Li, Xi Victoria Lin, Todor Mihaylov, Myle Ott, Sam Shleifer, Kurt Shuster, Daniel Simig, Punit Singh Koura, Anjali Sridhar, Tianlu Wang, and Luke Zettlemoyer. OPT: open pre-trained transformer language models. _CoRR_ , abs/2205.01068, 2022b. 

14 

Published as a conference paper at ICLR 2024 

Zhuosheng Zhang, Aston Zhang, Mu Li, Hai Zhao, George Karypis, and Alex Smola. Multimodal chain-of-thought reasoning in language models. _CoRR_ , abs/2302.00923, 2023b. 

Honglu Zhou, Asim Kadav, Farley Lai, Alexandru Niculescu-Mizil, Martin Renqiang Min, Mubbasir Kapadia, and Hans Peter Graf. Hopper: Multi-hop transformer for spatiotemporal reasoning. In _ICLR_ , 2021. 

15 

Published as a conference paper at ICLR 2024 

## APPENDIX 

## A OVERVIEW 

Here we provide, 1. Results on a “generalist” LRR model. 2. Surrogate tasks and overfitting. 3. Additional training details including hyper-parameters used for evaluation in Section 4 of the main paper across ACRE, Something-Else and CATER datasets. 4. Additional experiments to compare our LRR approach to “Chain-of-thought” (Wei et al., 2022) based approaches. 5. Additional qualitative examples across ACRE, Something-Else and CATER datasets. 

## A.1 GENERALIST MODEL 

The results in the main paper show that we can leverage the reasoning capabilities of pre-trained LMs to create “specialist” models with state of the art performance on a variety of visual reasoning tasks using end-to-end learning. Next, as both the input modality and the acquired visual skills are generic and in principle applicable across tasks, we train the model with OPT-125M backbone jointly on the two video-based reasoning tasks discussed above, CATER and Something-Else.[1] This is a step towards developing a grounded “generalist” for performing reasoning in videos. Despite commonalities in underlying low-level features, the diverse nature of reasoning objectives in the two tasks makes this highly challenging. As shown in Table 5, our LRR (Joint) model outperforms even the dataset specific fine-tuned state of the art models (SOTA) (Materzynska et al., 2020; Zhang, 2022; Ding et al., 2021) on these highly diverse visual reasoning tasks. 

Table 5. Evaluation of our LRR model trained jointly on CATER and Something-Else. 

|Method|Something-Else|CATER: Static Camera|CATER: Moving Camera|
|---|---|---|---|
||Top 1_↑_<br>Top 5_↑_|Top 1_↑_<br>Top 5_↑_|Top 1_↑_<br>Top 5_↑_|
|SOTA (_c.f_. Tables 2 and 3)<br>LRR (Joint, Ours)|56.2<br>81.3<br>**61.1**<br>**85.4**|79.7<br>95.5<br>**80.6**<br>**97.2**|59.7<br>90.1<br>**73.7**<br>**95.6**|



## B SURROGATE TASKS AND OVERFITTING 

To highlight that our surrogate tasks prevent our LRR model from overfitting, we report training and test accuracy of our LRR model trained with and without surrogate tasks on the compositional split of Something Else and on the moving camera split of CATER in Table 6. 

The results show that our surrogate tasks clearly prevent overfitting as the model is grounded to the fine-grained low-level details and is thus able to better “understand” the task at hand. 

## C ACRE 

**Additional training details.** We trained our LRR model with the OPT-125M and OPT-1.3B backbone until convergence ( _∼_ 500k iterations) with a batch size of 4. We use the AdamW optimizer (Loshchilov & Hutter, 2019) with a learning rate of 1 _×_ 10 _[−]_[5] , _β_ 1 = 0 _._ 9 _, β_ 2 = 0 _._ 95 and _λ_ (weight decay) = 0 _._ 1 and gradient clipping with a norm of 1 _._ 0. 

**Comparsion to “Chain-of-thought”.** As described in Section 3.4 of the main paper, we prompt our LRR model to solve certain surrogate tasks at randomly selected time steps. It is also possible to include surrogate tasks after every time-step. In this case, this would resemble a “Chain of Thought” Wei et al. (2022) like process, where the final answer would depend upon the surrogate tasks solved at inference time. However, this is very inefficient and thus impractical, especially for long video sequences. We compare both of these approaches in Table 7. LRR (Random 30%, Ours) where we prompt our LRR model with the OPT-125M backbone with a probability of 30% an input video frame to solve surrogate tasks _only_ during training. LRR (Every frame) where the model is prompted to solve surrogate tasks after every frame during training and inference. Note, that in case of the LRR (Every frame) model it is necessary to solve surrogate tasks during inference as the (autoregressive LM) model always sees such tasks during training. We see from the results in Table 7, that the performance of LRR (Random 30% during training, Ours) is comparable with LRR (Every frame) 

> 1 It may be possible to include datasets like ACRE by treating images as still videos, but we leave this for future work. 

16 

Published as a conference paper at ICLR 2024 

Table 6. Results highlighting that our LRR model prevents overfitting. 

|Method|Something-Else (Top 1_↑_)<br>Train<br>Test|CATER: Moving Camera (Top 1_↑_)|
|---|---|---|
|||Train<br>Test|
|LRR (w/o Surrogate tasks)<br>LRR (Ours)|92.6<br>50.1<br>87.3<br>62.0|99.4<br>62.7<br>89.7<br>80.4|



Table 7. Evaluation of our random prompting strategy on ACRE _c.f_ . Table 2 in the main paper. 

|Method|Compositional_↑_|Systematic_↑_|Inference Speed (msec)_↑_|
|---|---|---|---|
|||||
|ALOE (Ding et al., 2021)|91.7|93.9|-|
|LRR (Random 30% during training, Ours)<br>98.2<br>99.2<br>**61**<br>LRR (Every frame)<br>**99.3**<br>**99.5**<br>1415||||



while the inference speed is order of magnitudes faster. We report the inference speed in milliseconds on a single Nvidia A100 GPU. This is because our random prompting during training distills the relevant low-level information into the hidden states of the LM backbone at a feature level and thus does not require solving surrogate tasks at inference time in a COT like fashion. 

**Additional qualitative examples.** We include additional qualitative examples in Fig. 6 highlighting surrogate tasks. These examples illustrate that using our surrogate re-identification task, our LRR model can aggregate information from multiple context trials to arrive at the final answer – following the paradigm of “Look, Remember, Reason”. 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2024_Look,_Remember_and_Reason_Grounded_Reasoning_in_Videos_with_Language_Models_images/iclr_2024_Look,_Remember_and_Reason_Grounded_Reasoning_in_Videos_with_Language_Models.pdf-0017-07.png)


**----- Start of picture text -----**<br>
Context Trials Query<br><re-identify> 1,medium <re-identify> 1,medium<br><re-identify> 1,medium purple rubber cube; <re-identify> 2, medium purple rubber cube; 2,<br>purple rubber cube . . . 2, medium cyan rubber . . . cyan rubber cylinder. medium cyan rubber cylin-<br><blicket>  on. cylinder. <blicket>  off. der<br><blicket>  on. <answer>  yes.<br><re-identify><blicket> p4cube;metalmetalur,mediumplecubecube.5 on.,metalmedium; brown6,medium3, mediumsppmetalurhere;redple . . . <re-identify><blicket> p3smetalpur,mediumhere;plecube off. . metal6p,urmedium5p,lemediummetalcube;red . . . <re-identify> redp3srubbrownpur,mediumhere;pbermetallemetalsp7here;cube,metalmediumpcube.ur;p645le,,,mediummediummediumgmetalcube;reen <re-identify><answer> pmediumsmetalgpreenurhere;plerubcube;metal maybe.ber6p,ursmediumppcube;75here.le,, mediummediummetalred3,<br><blicket>  on.<br><blicket><re-identify> gray rubber on.cube.3, medium . . . <re-identify> gmediumcube;rubrayberrubcube;5ber,purmediumpcube;le36,,mediummediumrubcyberan4, . . . <re-identify> p6crubyur,mediumlinberpledercube.; 5rub,brownmediumber4, mediummetalcube;cyan <re-identify> cmedium4cube;y, anmediumrub6g,rabermediumpyurrubcube;p5le,bermediumrubbrowncube;ber3,<br><blicket> brown metal on.cylinder  . <blicket>  off. <answer> metal cylin yes.der.<br>**----- End of picture text -----**<br>


Figure 6. Example solutions to surrogate tasks generated by our LRR model on ACRE. Re-identified objects across context trials are underlined in the same color. 

## D SOMETHING-ELSE 

**Additional training details.** We trained our LRR model with the OPT-125M and OPT-1.3B backbone until convergence ( _∼_ 700k iterations) with a batch size of 4. We use the AdamW opti- 

17 

Published as a conference paper at ICLR 2024 

mizer (Loshchilov & Hutter, 2019) with a learning rate of 1 _×_ 10 _[−]_[5] , _β_ 1 = 0 _._ 9 _, β_ 2 = 0 _._ 95 and _λ_ (weight decay) = 0 _._ 1 and gradient clipping with a norm of 1 _._ 0. We also employ random data augmentation using RandAugment (Cubuk et al., 2020) with a magnitude of 15 and label smoothing with _ϵ_ = 0 _._ 3, where applicable (for fairness). 

**Additional qualitative examples.** The qualitative examples in Fig. 4 show that our LRR model can deal with visually complex real-world scenarios such as severe occlusions, _e.g_ ., the silver ball in Fig. 4 (bottom row) is occluded by the hand. Here in Fig. 7, we include additional qualitative examples further highlighting that our LRR model can deal with visually complex real-world scenarios such as severe deformations, _e.g_ ., the piece of paper in Fig. 7 second row which is torn into two pieces; severe changes in appearance, _e.g_ ., the pen in Fig. 7 third row, the teal case in the fifth row; motion blur in Fig. 7 last row. 

Finally, these examples illustrate that using our surrogate tracking task, our LRR model can aggregate information from multiple context trials to arrive at the final answer – following the paradigm of “Look, Remember, Reason”. 

## E CATER 

**Additional training details.** We trained our LRR model with the OPT-125M backbone until convergence ( _∼_ 600k iterations) with a batch size of 4. We use the AdamW optimizer (Loshchilov & Hutter, 2019) with a learning rate of 1 _×_ 10 _[−]_[5] , _β_ 1 = 0 _._ 9 _, β_ 2 = 0 _._ 95 and _λ_ (weight decay) = 0 _._ 1. We use gradient clipping with a norm of 1 _._ 0. 

**Additional qualitative examples.** We include additional qualitative examples including both the static and moving camera splits in Fig. 8, highlighting the surrogate tracking task. We see that our LRR model can successfully deal with containment in both static camera (rows 1-3; Fig. 8) and moving camera (row 4-5; Fig. 8) settings, due to our rationale that explicitly tracks cones and the snitch. Note that the example in row 5 in Fig. 8 from the moving camera split is especially challenging due to recursive containment. 

18 

Published as a conference paper at ICLR 2024 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2024_Look,_Remember_and_Reason_Grounded_Reasoning_in_Videos_with_Language_Models_images/iclr_2024_Look,_Remember_and_Reason_Grounded_Reasoning_in_Videos_with_Language_Models.pdf-0019-01.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2024_Look,_Remember_and_Reason_Grounded_Reasoning_in_Videos_with_Language_Models_images/iclr_2024_Look,_Remember_and_Reason_Grounded_Reasoning_in_Videos_with_Language_Models.pdf-0019-02.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2024_Look,_Remember_and_Reason_Grounded_Reasoning_in_Videos_with_Language_Models_images/iclr_2024_Look,_Remember_and_Reason_Grounded_Reasoning_in_Videos_with_Language_Models.pdf-0019-03.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2024_Look,_Remember_and_Reason_Grounded_Reasoning_in_Videos_with_Language_Models_images/iclr_2024_Look,_Remember_and_Reason_Grounded_Reasoning_in_Videos_with_Language_Models.pdf-0019-04.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2024_Look,_Remember_and_Reason_Grounded_Reasoning_in_Videos_with_Language_Models_images/iclr_2024_Look,_Remember_and_Reason_Grounded_Reasoning_in_Videos_with_Language_Models.pdf-0019-05.png)


**----- Start of picture text -----**<br>
Answer: Pretending or failing to wipe [something] off of [something].<br>Answer: Tearing [something] into two pieces.<br>Answer: Pushing [something] so that it almost falls off but doesn’t.<br>Answer: Pushing [something] from right to left.<br>Answer: Poking [something] so that it falls over.<br>Answer: Pushing [something] with [something].<br>Answer: Putting [something] next to [something].<br>**----- End of picture text -----**<br>


Figure 7. Example solutions to surrogate tasks generated by our LRR model on Something-Else, highlighting object tracks and bounding boxes 

19 

Published as a conference paper at ICLR 2024 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2024_Look,_Remember_and_Reason_Grounded_Reasoning_in_Videos_with_Language_Models_images/iclr_2024_Look,_Remember_and_Reason_Grounded_Reasoning_in_Videos_with_Language_Models.pdf-0020-01.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2024_Look,_Remember_and_Reason_Grounded_Reasoning_in_Videos_with_Language_Models_images/iclr_2024_Look,_Remember_and_Reason_Grounded_Reasoning_in_Videos_with_Language_Models.pdf-0020-02.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2024_Look,_Remember_and_Reason_Grounded_Reasoning_in_Videos_with_Language_Models_images/iclr_2024_Look,_Remember_and_Reason_Grounded_Reasoning_in_Videos_with_Language_Models.pdf-0020-03.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2024_Look,_Remember_and_Reason_Grounded_Reasoning_in_Videos_with_Language_Models_images/iclr_2024_Look,_Remember_and_Reason_Grounded_Reasoning_in_Videos_with_Language_Models.pdf-0020-04.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2024_Look,_Remember_and_Reason_Grounded_Reasoning_in_Videos_with_Language_Models_images/iclr_2024_Look,_Remember_and_Reason_Grounded_Reasoning_in_Videos_with_Language_Models.pdf-0020-05.png)


**----- Start of picture text -----**<br>
2, 29, 17 35, 20, 20 20, 20, 20 12, 3, 20<br>9, 34, 32 32, 3, 32 19, 0, 19 7, 2, 19<br>28, 12, 15, 1 1, 18, 29, 1 9, 7, 23, 9 34, 6, 23, 34<br>4, 20, 29, 9 23, 9, 22, 11 23, 9, 22, 11 15, 4, 11, 11<br>22, 10, 22 22, 22, 22 17, 17, 17 17, 29, 17<br>**----- End of picture text -----**<br>


Figure 8. Example solution to the tracking surrogate tasks generated by our LRR model on CATER. We show the predicted grid locations of the cones and the snitch below. 

20 

