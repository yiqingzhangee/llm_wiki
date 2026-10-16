
![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/cvpr_2024_PixelLM_Pixel_Reasoning_with_Large_Multimodal_Model_images/cvpr_2024_PixelLM_Pixel_Reasoning_with_Large_Multimodal_Model.pdf-0001-01.png)


## **PixelLM: Pixel Reasoning with Large Multimodal Model** 

Zhongwei Ren[1][*] , Zhicheng Huang[2][*] , Yunchao Wei[1] _[,]_[4] _[†]_ , Yao Zhao[1] _[,]_[4] , Dongmei Fu[2] , Jiashi Feng[3] , Xiaojie Jin[3][*] _[†‡]_ 

1Beijing Jiaotong University 2University of Science and Technology Beijing 

3ByteDance Inc. 4Peng Cheng Laboratory 

https://pixellm.github.io/ 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/cvpr_2024_PixelLM_Pixel_Reasoning_with_Large_Multimodal_Model_images/cvpr_2024_PixelLM_Pixel_Reasoning_with_Large_Multimodal_Model.pdf-0001-07.png)


Figure 1. PixelLM is an effective and efficient LMM for pixel-level reasoning and understanding. We show its visualization results in following scenarios: 1) Multi-target reasoning segmentation; 2) Instance-level segmentation tied with text description; 3) Multi-referring segmentation; 4) Conversation 

## **Abstract** 

_While large multimodal models (LMMs) have achieved remarkable progress, generating pixel-level masks for image reasoning tasks involving multiple open-world targets remains a challenge. To bridge this gap, we introduce_ _**PixelLM** , an effective and efficient LMM for pixel-level reasoning and understanding. Central to PixelLM is a novel, lightweight pixel decoder and a comprehensive segmentation codebook. The decoder efficiently produces masks from the hidden embeddings of the codebook tokens, which encode detailed target-relevant information. With this design, PixelLM harmonizes with the structure of popular LMMs and avoids the need for additional costly segmen-_ 

> *Equal contribution. Work done when Zhongwei and Zhicheng interned at ByteDance Inc. _[†]_ Correspondence to Xiaojie Jin _<_ jinxiaojie@ bytedance.com _>_ and Yunchao Wei _<_ yunchao.wei@bjtu.edu.cn _>_ . _‡_ Project lead. 

_tation models. Furthermore, we propose a target refinement loss to enhance the model’s ability to differentiate between multiple targets, leading to substantially improved mask quality. To advance research in this area, we construct MUSE, a high-quality multi-target reasoning segmentation benchmark. PixelLM excels across various pixellevel image reasoning and understanding tasks, outperforming well-established methods in multiple benchmarks, including MUSE, single- and multi-referring segmentation. Comprehensive ablations confirm the efficacy of each proposed component. All code, models, and datasets will be publicly available._ 

## **1. Introduction** 

Built upon the success of Large Language Models (LLMs) [6, 24, 25, 37], large multimodal models (LMMs) 

have significantly enhanced high-level visual perception and user interaction experiences [2, 15, 17, 41]. Yet, most of them generate textual descriptions for global images or regions, with limited capability for pixel-level responses like object masks. This research gap limits the practical application of multimodal systems in fine-grained tasks, such as image editing, autonomous driving, and robotics. 

Recent work [14] explores using LLMs to produce object masks in a novel reasoning segmentation task, which is more challenging and flexible for real-world applications. In contrast to traditional segmentation which explicitly specifies objects (e.g., “orange”), reasoning segmentation requires complex reasoning for more intricate instructions (e.g., “the fruit high in Vitamin-C”), which aligns well with the capabilities of LMMs. However, this method has two major drawbacks: 1) it cannot handle tasks involving multiple target objects, which are indispensable in realworld scenarios, and 2) it depends on a pre-trained image segmentation model like SAM [13]. This reliance incurs substantial computational demands and confines the overall model’s performance to the capability of segmentation model, consequently impeding the model’s potential to enhance its performance through further training scaling up. 

In this paper, we introduce **PixelLM** , an effective and efficient LMM for pixel-level reasoning and understanding. PixelLM proficiently handles tasks with an arbitrary number of open-set targets and diverse reasoning complexities. Its design maintains the fundamental structure of LMMs while avoiding additional, costly segmentation models, enhancing both efficiency and transferability to diverse applications. Fig. 1 showcases PixelLM’s capabilities in diverse segmentation tasks, producing high-quality target masks. 

At the core of PixelLM is a novel pixel decoder and a segmentation codebook. The codebook contains learnable tokens that encode contexts and knowledge pertinent to targets referencing at different visual scales. The pixel decoder then produces target masks based on the hidden embeddings from the codebook tokens in conjunction with image features. Thanks to this design, PixelLM can generate highquality masks without external segmentation models, significantly boosting its efficiency. Furthermore, we propose a target refinement loss to enhance the model’s capability of differentiating between multiple targets, thus further improving the mask quality. 

To facilitate model training and evaluation in this research area, we construct MUSE, a comprehensive multitarget reasoning segmentation dataset. Utilizing a GPT4V [25]-aided data curation pipeline, we create 246k question-answer pairs, covering 0.9 million instances. Our extensive ablation studies confirm the dataset’s effectiveness in cultivating the model’s pixel reasoning capabilities. 

PixelLM’s effectiveness is clearly demonstrated across a variety of benchmarks. It achieves state-of-the-art perfor- 

mance on benchmarks including MUSE, single- and multireferring segmentation, significantly outperforming baseline models. Notably, in comparison to models that rely on external segmentation models, such as [14], PixelLM reduces computational costs by up to 50% while either maintaining or improving performance. 

In summary, our contributions are: 

- We present PixelLM, a novel LMM for pixel-level reasoning and understanding. It handles tasks with diverse reasoning complexities, maintaining high efficiency. 

- We construct MUSE, a high-quality multi-target reasoning segmentation dataset, facilitating model training and evaluation in future research. 

- PixelLM achieves state-of-the-art results across a diverse range of benchmarks, clearly demonstrating its superior efficacy and efficiency compared to competing methods. 

## **2. Related Work** 

## **2.1. Large Multimodal Models** 

Large multimodal models (LMMs) have significantly enhanced the performance of tasks requiring the understanding of diverse modalities. These models generally fall into two categories based on their use of large language models (LLMs). 

The first category [21, 31, 36] involves models trained from scratch or employing relatively smaller language models like BERT for text processing. They typically utilize a blend of contrastive and generative objectives to address a range of multimodal tasks. However, their limited language understanding capacity often hinders their performance in tasks that demand extensive world knowledge and reasoning abilities. 

The advent of LLMs has spurred a new direction in LMM development, where LLMs are augmented with multimodal comprehension capabilities [26, 29, 30, 38]. Common approaches in this framework involve integrating adapters to align visual and textual representations within LLMs. Notable examples include BLIP-2 [15], Flamingo [1], MiniGPT-4 [41], llama-adapter [10, 37], LLaVA [17], InstructBLIP [8], InternGPT [19], and QwenVL [2]. While these methods have shown improved performance in vision-language tasks through instructional tuning, their primary limitation lies in generating only textual outputs about the entire image, thus constraining their utility in tasks requiring more granular, region-level or pixellevel understanding. 

## **2.2. Fine-Grained LMMs** 

In many practical applications, understanding visual inputs at a more detailed level is crucial, such as specific regions or even at the pixel level. Several methods have been proposed to endow LLMs with this fine-grained un- 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/cvpr_2024_PixelLM_Pixel_Reasoning_with_Large_Multimodal_Model_images/cvpr_2024_PixelLM_Pixel_Reasoning_with_Large_Multimodal_Model.pdf-0003-00.png)


Figure 2. Overview of the proposed PixelLM model architecture. (Left) Overall architecture. (Right) The proposed lightweight pixel decoder. Trainable LoRA parameters are incorporated into the LLM. All parameters except those for the CLIP encoder and LLM are trainable. 

derstanding capability. GPT4RoI [39], Shikra [5], VisionLLM [32], Kosmos-2 [27], InternGPT [20], and Ferret [35] offer grounding capabilities to specified image regions, typically encoding location coordinates as tokens for integration with LLMs. Unlike these methods, which lack the ability to generate pixel-wise masks, LISA [14] integrates SAM with LLMs for segmentation tasks. Moreover, LISA explores the use of LMMs for complex instruction reasoning, which differs from traditional tasks that rely on explicit human instructions for object or category identification. However, LISA is constrained to handling single targets in images, and the incorporation of SAM adds significant computational overhead. In contrast, our goal is to develop an efficient LMM capable of pixel-level image reasoning and understanding, accommodating a varied number of targets and diverse reasoning complexities. 

## **3. Method** 

We first outline the framework in Sec. 3.1, elucidating two key designs: pixel decoder and segmentation codebook. Training objectives are introduced in Sec. 3.2. 

## **3.1. Model Design** 

**Framework overview** As depicted in Fig. 2, PixelLM features a streamlined architecture, comprising four main parts: _i_ ) a pre-trained CLIP-ViT vision encoder _I_ which aligns with text, _ii_ ) a large language model _F_ , _iii_ ) a lightweight pixel decoder _D_ and _iv_ ) a segmentation codebook _C_ seg. PixelLM processes image _x_ img and query text _x_ txt, yielding interleaved text description and corresponding masks for varied number of targets. 

While components _i_ ) and _ii_ ) adhere to well-established LMM architectures for ensuring compatibility, the pixel decoder and segmentation codebook are pivotal in empowering LMMs with mask generation capabilities across diverse 

scenarios. We utilize the segmentation codebook to encode target-relevant information into the embeddings of the codebook tokens, which the pixel decoder then transforms in conjunction with image features into precise masks. In the following, we detail their designs. 

**Segmentation codebook** Aiming to enrich the encoding of target-specific information and thereby facilitate the generation of high-quality masks, we devise a comprehensive segmentation codebook. This codebook includes various groups of tokens, each representing different levels of granularity or scales in visual concepts, tailored to meet the demands of segmentation tasks. For example, proficient segmentation requires comprehension of both semantic categories and nuanced geometric shapes. These elements are typically represented at distinct network layers. In response, our single codebook integrates diverse visual information, effectively capturing both semantic and geometric aspects necessary for accurate segmentation. 

Specifically, the codebook consists of multiple token groups, each corresponding to a semantic scale of visual features from the image encoder. Formally, we define _C_ seg = � _c[ℓ] n[∈]_[R] _[d]_[�] _[N,L] n_ =1 _,ℓ_ =1[,][where] _[L]_[and] _[N]_[denote][the] number of visual scales and tokens per group, respectively, and _d_ represents the hidden dimension in LMMs. For clarity, we first set _N_ = 1 and introduce how the codebook tokens are integrated within the LMMs to encode requisite information for target mask generation. 

For an input image _x_ img, the vision encoder _I_ extracts a spectrum of multi-scale visual features _I_ img = _{I_ img _[ℓ][}] ℓ[L]_ =1 from _I_ ( _x_ img), comprising _L_ visual features output at select layers of _I_ . The output of the final layer, _I_ img _[L]_[,][en-] capsulates global image information and is transformed into the language space via a vision-to-language projection layer _pV →T_ . Simultaneously, a vision-to-decoder projection _pV →D_ transforms all _I_ img features, resulting in _f_ img = 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/cvpr_2024_PixelLM_Pixel_Reasoning_with_Large_Multimodal_Model_images/cvpr_2024_PixelLM_Pixel_Reasoning_with_Large_Multimodal_Model.pdf-0004-00.png)


Figure 3. The segmentation codebook example comprises two scales with two tokens each. Each attention map results from the interaction between one token and its corresponding image feature in the decoder. The first two rows depict the token fusion mechanism, while the final row demonstrates a failure case arising from the utilization of only one token. 

_L_ � _f_ img _[ℓ]_[=] _[ p][V][ →][D]_[(] _[I]_ img _[ℓ]_[)] � _ℓ_ =1[.] The codebook tokens, combined with the input image and text, are then processed by the LLM to generate interleaved response _y_ res in an autoregressive way: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/cvpr_2024_PixelLM_Pixel_Reasoning_with_Large_Multimodal_Model_images/cvpr_2024_PixelLM_Pixel_Reasoning_with_Large_Multimodal_Model.pdf-0004-03.png)


To help understand this process, consider an example of text query “Segment the apple on the left”. Then, the output _y_ res contains _L_ tokens of _C_ seg: “The apple is _c_[1] _, . . . , c[L]_ ”. The corresponding hidden embeddings (i.e. the output of last layer of _F_ ) of _C_ seg are represented as _h_ = � _h[ℓ]_[�] _[L] ℓ_ =1[,][which][are][inputs][to][the][pixel][decoder] _[D]_ alongside image features _f_ img for mask generation. 

We then explain the rationale behind setting _N >_ 1. As depicted in Fig. 3, scenarios involving multiple targets or increased complexity reveal that a single token may not adequately capture the full scope of target semantics, despite the LLM providing precise textual responses. To bolster the model’s capacity for interpreting complex reasoning scenarios, we introduce multiple tokens within each scale group and perform a token fusion operation, denoted _N_ as _c[ℓ]_ = � _c[ℓ] n_ � _n_ =1[.][Before][the][decoding][process,][a][linear] projection layer _ϕ_ is utilized to convert the hidden states of these grouped tokens into a unified form, represented as _h[ℓ]_ = _ϕ_ ( _h[ℓ]_ 1 _[, . . . , h][ℓ] N_[)][.][Fig.][3][showcases][the][application][of] multiple tokens within each group. The post-decoding attention map visualizations demonstrate that individual tokens contribute distinct, yet complementary information, leading to a more effective mask generation than is possible with a single token approach. 

**Pixel decoder** We design a novel and lightweight pixel decoder _D_ to adeptly harness the multi-scale features from the vision encoder. This decoder is tasked with learning the transformation of these features, in conjunction with the hidden embeddings from _C_ seg, into precise segmentation masks. Such a design obviates the need for extra costly segmentation models like SAM [14], thus significantly improving efficiency. 

As depicted in the right panel of Fig. 2, _D_ consists of _L_ attention blocks � _Attn[ℓ]_[�] _[L] ℓ_ =1[, each corresponding to dis-] tinct scales of image features and the codebook. For each targeted mask generation, _D_ sequentially produces a mask score map _m[ℓ]_ at each scale _ℓ_ , which then directs the model’s attention to regions of higher relevance in the subsequent scale _ℓ −_ 1. This strategy works by guiding the model to focus on areas with high confidence scores in _m[ℓ]_ , thereby facilitating more accurate mask generation. _fimg[ℓ][′]_[=] � _ffimgimg L[ℓ][⊙]_[(] _[σ]_[(] _[m][ℓ]_[+1][) + 1)] _ℓ< Lℓ_ = _L_ (1) _m[ℓ]_ = _Attn[ℓ]_ ( _h[ℓ] , fimg[ℓ][′]_[)] _[.]_ 

where _fimg[ℓ][′]_[is the modulated feature at scale] _[ ℓ]_[,] _[ σ]_[ is sigmoid] function and _⊙_ is element-wise multiplication. Finally, we learn the weighting factors _γ_ = [ _γ[ℓ]_ ] _[L] ℓ_ =1[to][combine][mask] maps at all scales to get the final segmentation result: _M_[ˆ] = � _Lℓ_ =1 _[γ][ℓ][m][ℓ]_[where] _[|][γ][|]_[=][1][.][The][detailed][structure][of][the] decoder is provided in Supp. A. 

## **3.2. Traning Objectives** 

**Target refinement loss** As the number of targets increases, the likelihood of the model encountering confusion and producing overlapping masks intensifies. To mitigate this issue, we introduce a target refinement loss. This simple yet effective strategy focuses on unclear pixels where multiple targets are predicted together. This helps the model in clearly identifying and learning different targets. 

Denote the mask predictions as _M_ ˆ _k ∈_ R _[H][×][W]_[ �] _[K]_ � _k_ =1 whereshape of _K_ mask.is the total _M_ ˆ _ki_ number _∈_ R representsof targets,the _H_ binaryand _W_ valueare theof each pixel. Then we define a map _A_ to assign increased weights to regions predicting multiple targets: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/cvpr_2024_PixelLM_Pixel_Reasoning_with_Large_Multimodal_Model_images/cvpr_2024_PixelLM_Pixel_Reasoning_with_Large_Multimodal_Model.pdf-0004-12.png)


where _α_ is a hyper-parameter. The weighted loss is computed against the ground-truth _Mk_ for each mask as follows: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/cvpr_2024_PixelLM_Pixel_Reasoning_with_Large_Multimodal_Model_images/cvpr_2024_PixelLM_Pixel_Reasoning_with_Large_Multimodal_Model.pdf-0004-14.png)


where _LBCE_ is per-pixel binary cross-entropy loss. 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/cvpr_2024_PixelLM_Pixel_Reasoning_with_Large_Multimodal_Model_images/cvpr_2024_PixelLM_Pixel_Reasoning_with_Large_Multimodal_Model.pdf-0005-00.png)


Figure 4. The left panel illustrates the prompt employed in our GPT-4V data generation pipeline. The right panel showcases an example of the generated data. 

**Overall loss** The model is trained end-to-end using an autoregressive cross-entropy loss _Ltxt_ for text generation (note the tokens from the segmentation codebook are also included in the loss calculation), a DICE loss _Ldice_ and a target refinement loss _Lref_ for mask generation. The overall objective _L_ constitutes the weighted sum of these losses, calibrated by _λref_ and _λdice_ : 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/cvpr_2024_PixelLM_Pixel_Reasoning_with_Large_Multimodal_Model_images/cvpr_2024_PixelLM_Pixel_Reasoning_with_Large_Multimodal_Model.pdf-0005-03.png)


## **4. Multi-target Reasoning Segmentation** 

Our objective is to develop a LLM capable of handling tasks involving an arbitrary number of open-set targets and diverse reasoning complexities. A primary challenge is the absence of a suitable dataset for model training. We review existing public datasets and identify critical limitations: 1) Inadequate detail and object representation in segmentation masks; 2) A lack of question-answer pairs featuring complex reasoning and a varied number of objectives. 

To address these issues, we introduce a structured annotation pipeline for constructing the multi-target reasoning segmentation (MUSE) data. Examples of MUSE are shown in Fig. 1. MUSE stands out with its open-set concepts, detailed object descriptions, complex multi-target questionanswer pairs, and instance-level mask annotations. In the next, we elaborate on the construction of MUSE. 

## **4.1. MUSE Dataset** 

Two public datasets, RefCOCO series [12] and ReasonSeg dataset [14], are pertinent to our goal. RefCOCO guides segmentation with explicit target object names, e.g. “orange”, lacking more complicated instructions, e.g. “the fruit high in Vitamin-C”. Besides, they also fall short in offering multi-target question-answer pairs with target descriptions directly connected to segmentation masks, which 

however is a common requirement in real-world scenarios, like “How to make fruit salad?” 

To this end, a total of 910k high-quality instance segmentation masks are selected from the LVIS dataset, along with detailed textual descriptions based on image content. Utilizing these instances, we construct 246k question-answer pairs, averaging 3.7 targets per answer. This dataset is then divided into three splits: train, val, and test, containing 239k, 2.8k, and 4.3k question-answer pairs, respectively. The test split comprises two parts: the number of targets involved in the question are less or more than three. Please see Supp. C.2 for more analysis of MUSE. 

## **4.2. Dataset Generation Pipeline** 

We first try to use GPT-4 to construct our dataset: using LLAVA [17] for image captioning and then GPT-4 for generating questions about multiple image regions. We utilize images with pre-existing mask annotations to reduce annotation costs. The image caption, manually selected object names, and bounding box coordinates from the image are input into GPT-4 to facilitate answer selection and question formulation. However, due to the inability to directly perceive image content, the content of question-answer pairs generated by this method is often confined to the caption’s description, significantly limiting the diversity of the data. 

To address these limitations, the pipeline is refined in two key ways. First, we convert to the more advanced GPT4V which shows strong capabilities in understanding visual contents [25]. This model has been instrumental in generating more nuanced and naturalistic questions. Additionally, we implement a more dynamic approach for answer generation. Specifically, we feed all the instance category names and corresponding bounding box coordinates in the image to GPT-4V. Using carefully crafted prompts, GPT-4V autonomously selects instances to construct question-answer 

pairs relevant to the image content. An example of such a prompt is illustrated in Fig. 4. More details of data filtering and the data generation pipeline are provided in Supp. C. 

## **4.3. Evaluation** 

To evaluate our task, we focus on three main aspects: _i_ ) the generation of natural text descriptions aligned with corresponding object masks, _ii_ ) the accuracy of the match between object masks and text descriptions, and _iii_ ) the quality of the masks. Since we focus on the mask quality of each target, we do not evaluate image-level captioning. 

The evaluation process for each question involves four steps: **First** , we match the predicted masks with groundtruth masks based on mask IoU scores using bipartite matching, similar to DETR [4]. Any unassigned predictions or ground-truths are assigned an empty mask. **Second** , we replace the position of the mask in the generated text with the corresponding ground-truth object description. For example, as shown in Fig. 2, the text might read: “Sit in the kayak (a red kayak parked on the beach), propel it forward using the paddle (a double-bladed paddle on the kayak)”. The content in brackets is a ground-truth description. **Third** , this modified text prompts GPT-3.5 to score each prediction from 1 to 10, with higher scores indicating better quality, and unassigned predictions receiving a score of 0. **Fourth** , the final score for each prediction is the product of the GPT and IoU scores, from which we calculate the gIoU and cIoU metrics. For more evaluation details, please refer to Supp. A. 

## **5. Experiment** 

In this section, we first present the implementation details and then show the comparison results on benchmarks. Finally, we ablate on the key components in PixelLM. 

## **5.1. Implementation Details** 

We use pre-trained multimodal model from LlaVA-7B and LlaVA-llama2-13B, with LoRA adopted for efficient finetuning. The vision encoder uses a fixed CLIP-ViT-L/14336 model, modified with linearly interpolated position embeddings for processing 448x resolution images. The trainable parts of our model include the pixel decoder _D_ , LoRA parameters, segmentation codebook _C_ seg, the vision-tolanguage and vision-to-decoder projection layers _pV →T_ and _pV →D_ . To facilitate task evaluation, we convert refCOCO series datasets into multi-referring segmentation datasets (detailed in Supp. C.3). Training involves random sampling from multi-referring segmentation datasets (ADE20K [40], COCO-Stuff [3], LVIS-PACO [28], refCOCO series [12], _∼_ 200k images in total), VQA data (LLAVA-150k), and MUSE. The training steps follow LISA [14], requiring approximately 1.5/2 days on 8 A100 GPUs for the 7B/13B model. 

## **5.2. Benchmarks and Baselines** 

**Benchmarks** We evaluate PixelLM on three benchmarks with a varied number of targets: MUSE, multi-referring segmentation and the conventional referring segmentation (refCOCO series). The former two involve multiple targets and the last one focuses on a single object. Through this evaluation, we validate the versatility of PixelLM in diverse mask generation tasks. For the MUSE benchmark, we use the GPT-based evaluation metric as detailed in Sec. 4.3. In multi-referring segmentation, we formulate multi-target queries from the refCOCO dataset’s annotations, asking models to segment 3 to 6 objects per image. These queries are structured as “Please segment the _<_ objects _>_ ”, with _<_ objects _>_ being comma-separated object descriptors. The responses involve generating masks corresponding to the listed objects’ sequence. For refCOCO series, we follow previous methods to calculate the gIoU and cIoU scores. **Baselines** To our best knowledge, PixelLM is the first to handle complex pixel reasoning tasks involving multiple targets. To demonstrate the effectiveness of PixelLM, we establish four strong baselines for comparative analysis on the aforementioned benchmarks. Three baselines evolve from LISA [14], which is the most relevant work with PixelLM, and one additional non-LLM based model. 

- LISA [14]: The original model, designed for single-target segmentation, employs SAM for mask generation. 

- LISArec: To address the limits of LISA, this variant employs a two-step approach: first identify target objects in text form and then ask LISA to segment them one by one. Specifically, LLAVA-13B extracts noun phrase responses from multi-target queries using the prompt: “ _<_ question _>_ , Please only use noun phrases in answer”, which are then input to LISA for segmentation. 

- LISAaug: An augmented version of LISA, adapted to generate masks for multiple targets and additionally trained with the MUSE dataset. 

- SEEM [43]: A state-of-the-art image segmentation model, SEEM is limited to producing segmentation masks only. Hence, its evaluation is confined to mask quality assessment across each benchmark. 

## **5.3. Results on MUSE** 

Tab. 1 compares PixelLM with competing methods on the multi-target reasoning segmentation task. For SEEM and the original LISA, since they can only generate a single segmentation mask given query text, we report the IoU between prediction and the ground truth which merges all target masks. Although the text description for the target is ignored, thus leading to a simpler setting when compared with ours, we can observe their performance is still inferior, which demonstrates the challenges of this task. Although LISArec improves upon LISA by simplifying the generation task by first generating the target for multi-target queries, 

|Method|w/o<br>SAM|TFLOPs|Val|Test|Test|Test|
|---|---|---|---|---|---|---|
||||overall|few targets|many targets|overall|
||||gIoU<br>cIoU|gIoU<br>cIoU|gIoU<br>cIoU|gIoU<br>cIoU|
|SEEM[43]|�|0.43|13.6<br>16.2|23.6<br>24.9|8.5<br>13.2|11.7<br>15.7|
|LISA-7B [14]<br>LISA-7Brec<br>LISA-7Baug<br>PixelLM-7B_†_<br>PixelLM-7B|�<br>�<br>�<br>�<br>�|7.16<br>7.16<br>7.16<br>**3.57**<br>**3.57**|18.8<br>29.0<br>24.5<br>31.1<br>42.0<br>46.1<br>39.9<br>48.0<br>**42.6**<br>**50.7**|24.7<br>36.5<br>30.0<br>30.9<br>43.5<br>52.0<br>43.1<br>56.7<br>**44.6**<br>**59.2**|9.6<br>24.5<br>12.4<br>23.2<br>37.7<br>42.3<br>36.0<br>38.2<br>**37.7**<br>**42.8**|12.8<br>27.1<br>16.2<br>24.8<br>38.9<br>44.4<br>37.5<br>42.2<br>**39.2**<br>**46.3**|
|LISA-Llama2-13B [14]<br>LISA-Llama2-13Baug<br>PixelLM-Llama2-13B_†_<br>PixelLM-Llama2-13B|�<br>�<br>�<br>�|10.24<br>10.24<br>**6.65**<br>**6.65**|20.4<br>29.2<br>43.6<br>50.2<br>43.0<br>51.7<br>**44.8**<br>**54.1**|27.5<br>38.5<br>44.7<br>60.0<br>44.8<br>61.6<br>**45.2**<br>**62.9**|10.9<br>25.6<br>41.2<br>**47.9**<br>39.3<br>44.6<br>**41.5**<br>47.6|14.4<br>28.4<br>41.9<br>50.5<br>40.5<br>48.2<br>**42.3**<br>**51.0**|



Table 1. **Comparison on MUSE benchmark.** _[†]_ denotes PixelLM w/o using the token fusion mechanism and target refinement loss. 

|Method|w/o<br>SAM|MrefCOCO|MrefCOCO+|MrefCOCOg|
|---|---|---|---|---|
|||val<br>testA<br>testB|val<br>testA<br>testB|val(U) test(U)|
|LISA [14]<br>LISAaug<br>PixelLM_†_<br>PixelLM|�<br>�<br>�<br>�|36.7<br>38.3<br>36.4<br>68.9<br>70.8<br>66.3<br>70.3<br>74.2<br>66.2<br>**72.7**<br>**76.2**<br>**68.1**|34.0<br>36.3<br>32.1<br>59.8<br>62.2<br>54.1<br>64.4<br>69.6<br>57.0<br>**65.7**<br>**71.3**<br>**57.7**|34.5<br>36.2<br>62.3<br>63.9<br>64.0<br>67.0<br>**65.8**<br>**67.7**|



Table 2. **Results on the multi-referring segmentation benchmark.** The meaning of _[†]_ is the same as Tab. 1. 

|**mark.** The me|anin|g of _†_ is the same|as Tab.1.||
|---|---|---|---|---|
|Method|w/o<br>SAM|refCOCO|refCOCO+|refCOCOg|
|||val<br>testA<br>testB|val<br>testA<br>testB|val(U) test(U)|
|MCN [22]<br>VLT [9]<br>CRIS [33]<br>LAVT [34]<br>ReLA [16]<br>X-Decoder [42]<br>SEEM [43]|�<br>�<br>�<br>�<br>�<br>�<br>�|62.4<br>64.2<br>59.7<br>67.5<br>70.5<br>65.2<br>70.5<br>73.2<br>66.1<br>72.7<br>75.8<br>68.8<br>73.8<br>76.5<br>70.2<br>-<br>-<br>-<br>-<br>-<br>-|50.6<br>55.0<br>44.7<br>56.3<br>61.0<br>50.1<br>62.3<br>68.1<br>53.7<br>62.1<br>68.4<br>55.1<br>66.0<br>71.0<br>57.7<br>-<br>-<br>-<br>-<br>-<br>-|49.2<br>49.4<br>55.0<br>57.7<br>59.9<br>60.4<br>61.2<br>62.1<br>65.0<br>66.0<br>64.6<br>-<br>65.7<br>-|
|LISA [14]<br>LISAaug<br>PixelLM|�<br>�<br>�|**74.1**<br>**76.5**<br>**71.1**<br>74.0<br>76.3<br>70.4<br>73.0<br>**76.5**<br>68.2|62.4<br>67.4<br>56.5<br>62.5<br>66.3<br>56.0<br>**66.3**<br>**71.7**<br>**58.3**|66.4<br>68.5<br>67.0<br>69.1<br>**69.3**<br>**70.5**|



Table 3. **Results on the referring segmentation benchmark.** 

it still largely lags behind our models. This highlights the importance of end-to-end training on the task. By training on MUSE, the performance of the augmented LISA LISAaug further improves on the dataset, which validates the benefits of our dataset in training an LMM with strong pixel reasoning capabilities. 

PixelLM stands out in both efficiency and performance. Under the fair setting involving equivalent training data and LMM sizes, PixelLM-7B and PixelLM-Llama2-13B consistently outperform their counterparts (LISAaug and LISALlama2-13Baug, respectively) across almost all evaluated metrics. Notably, PixelLM eschews additional segmentation modules, thereby conferring substantial efficiency advantages. In terms of computational efficiency, PixelLM achieves a remarkable reduction in TFlops by 50% and 35% for the 7B and 13B model sizes, respectively, compared to LISA. Interestingly, while PixelLM benefits from model scaling up: PixelLM-Llama2-13B boosts upon the PixelLM with a 7B LMM, it still maintains smaller TFlops (6.65 vs. 

7.16) than LISAaug which includes an extra SAM model. Furthermore, we conduct an ablation study to ascertain the impact of the proposed token fusion mechanism and target refinement loss. PixelLM, devoid of these two components, is denoted as PixelLM _[†]_ . The results clearly demonstrate that the complete PixelLM configuration significantly outperforms PixelLM _[†]_ , thereby affirming the contribution of these components to the model’s efficacy. This trend is consistent in the 13B model variant as well. Detailed ablation analyses of each component are presented in Sec. 5.5. 

## **5.4. Results on Referring Segmentation** 

Tab. 2 presents the results on the multi-target referring segmentation dataset, wherein PixelLM demonstrates superior performance across all data splits, significantly outperforming LISA and its enhanced variant, LISAaug. Notably, PixelLM exhibits enhanced efficiency compared to the LISA models, as evidenced by its lower TFlops (as detailed in Tab. 1). The superiority of the token fusion and target refinement loss components within PixelLM is further corroborated through the comparison with PixelLM _[†]_ . 

We also compare results on the conventional singletarget refCOCO series dataset. Despite the dataset’s focus on single target segmentation, PixelLM attains commendable performance. As shown in Tab. 3, PixelLM achieves the highest scores on most metrics, particularly excelling in the more challenging refCOCO+/refCOCOg datasets. 

## **5.5. Ablation Study** 

**Number of scales.** Tab. 4a shows the effects of increasing the number scales in PixelLM by adding more feature layers and corresponding codebook tokens to the decoder. We observe notable gains by adding just one extra scale (the 2nd row). The gain diminishes with more scales. Using a larger laion-CLIP-H can further improves performance. 

**Token fusion mechanism** In this ablation, the number of tokens in a group is 3. Table 4c shows that applying token fusion achieves up to a 2.4% and 2.2% increase in cIoU on MUSE val and refCOCOg, respectively. This demonstrates its benefits for both multi-target and single-target tasks. 

|Architecture|MUSE Val<br>gIoU<br>cIoU|refCOCOg<br>val(U)<br>test(U)|
|---|---|---|
|baseline<br>40.1<br>47.2<br>64.3<br>65.6<br>+ 2 scale feature-fusion<br>42.6<br>50.7<br>69.3<br>70.5<br>+ 3 scale feature-fusion<br>42.3<br>51.4<br>69.8<br>70.4<br>+ w/ CLIP-H<br>**47.8**<br>**55.2**<br>**72.6**<br>**73.0**<br>(a) Different number of scales and larger vision encoder.|||



|Token<br>fusion<br>Target<br>refnement loss|MUSE Val<br>gIoU<br>cIoU|refCOCOg<br>val(U)<br>test(U)|
|---|---|---|
|39.9<br>48.0<br>68.0<br>68.3<br>�<br>41.5<br>50.1<br>69.3<br>70.5<br>�<br>40.7<br>48.6<br>68.0<br>68.3<br>�<br>�<br>**42.6**<br>**50.7**<br>**69.3**<br>**70.5**<br>(c) Token fusion mechanism and target refnement loss.|||



|Token number_N_<br>per-group|Token number_N_<br>per-group|MUSE Val<br>gIoU<br>cIoU|MUSE Val<br>gIoU<br>cIoU|refCOCOg<br>val(U)<br>test(U)|
|---|---|---|---|---|
||1<br>40.7<br>48.6<br>68.0<br>68.3<br>2<br>42.0<br>49.5<br>69.0<br>69.9<br>3<br>42.6<br>**50.7**<br>**69.3**<br>70.5<br>4<br>**43.0**<br>50.3<br>69.1<br>**70.6**<br>(b) Different token number in each group.<br>Data<br>generator<br>Data<br>amount<br>MUSE Val<br>gIoU<br>cIoU<br>GPT-4<br>30k<br>30.0<br>35.7<br>GPT-4V<br>30k<br>35.6<br>40.0<br>100k<br>38.9<br>45.7<br>200k<br>**42.6**<br>**50.7**||||
||GPT-4||30k||
||GPT-4V||30k<br>100k<br>200k||



(d) Different data generators and number of question-answer pairs. 

Table 4. **Ablations.** We conduct all experiments based on our 7B model. 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/cvpr_2024_PixelLM_Pixel_Reasoning_with_Large_Multimodal_Model_images/cvpr_2024_PixelLM_Pixel_Reasoning_with_Large_Multimodal_Model.pdf-0008-05.png)


Figure 5. **Comparison between PixelLM and PixelLM** _[†]_ (w/o token fusion mechanism and target refinement loss.) 

**Token number in each scale** Tab. 4b explores how changing the number of tokens per group (denoted as _N_ in Section 3.1) affects performance. _N_ = 1 corresponds to no token fusion. Increasing _N_ to two and three consistently boosts performance, yielding increases of 0.9% to 2.1% in cIoU and 1.3% to 2.1% in gIoU. However, further increases yield only slight improvements in metrics. 

**Target refinement loss** As Table 4c shows, target refinement loss results in a 0.8% improvement in gIoU and 0.6% in cIoU on MUSE val. This loss, designed for scenarios with multiple targets, does not affect performance in refCOCOg. Combining the loss with token fusion can lead to up to a 2.7% improvement in cIoU on MUSE. Fig. 5 compares the use (third column) and non-use (second column) 

of these two designs, wherein the former shows higher mask quality and better target completeness than the former. 

**MUSE enhances pixel reasoning.** To further verify this, We test LISA both with and without MUSE training on ReasonSeg, a single-object reasoning dataset proposed by LISA. Training with MUSE markedly improves LISA’s performance on ReasonSeg val (gIoU: 52.9 _→_ 58.0, cIoU: 54.0 _→_ 59.1), underscoring MUSE’s effectiveness in enhancing pixel reasoning. 

**GPT-4V** _**vs**_ **. GPT-4** To showcase GPT-4V’s advantage over GPT-4 in data generation, we create 30,000 extra multitarget question-answer pairs for a balanced comparison. As one can see from Tab. 4d, GPT-4V-generated data shows superior performance in the MUSE benchmark, with up to 7.0% gIoU and 8.6% cIoU gains using the same data amount. Additionally, increasing MUSE training data volume leads to consistent performance improvements, indicating that enhancing the MUSE dataset allows PixelLM to gradually improve its general pixel-level understanding. 

## **6. Conclusion** 

In this study, we introduce PixelLM, an effective and efficient LMM for pixel-level image reasoning and understanding. Benefiting from novel designs, PixelLM is adept at producing high-quality masks for a variety of tasks. Moreover, we construct a comprehensive multi-target reasoning segmentation benchmark to bolster this research area. Through extensive experiments, PixelLM achieves promising results across various benchmarks. Future endeavors will focus on the expansion and enhancement of PixelLM’s capabilities. 

## **Acknowledgments** 

This research was funded by the Fundamental Research Funds for the Central Universities (2023JBZD003), and the National NSF of China (No.U23A20314). 

## **References** 

- [1] Jean-Baptiste Alayrac, Jeff Donahue, Pauline Luc, Antoine Miech, Iain Barr, Yana Hasson, Karel Lenc, Arthur Mensch, Katherine Millican, Malcolm Reynolds, et al. Flamingo: a visual language model for few-shot learning. _Advances in Neural Information Processing Systems_ , 35:23716–23736, 2022. 2 

- [2] Jinze Bai, Shuai Bai, Shusheng Yang, Shijie Wang, Sinan Tan, Peng Wang, Junyang Lin, Chang Zhou, and Jingren Zhou. Qwen-vl: A versatile vision-language model for understanding, localization, text reading, and beyond. _arXiv preprint arXiv:2308.12966_ , 2023. 2 

- [3] Holger Caesar, Jasper Uijlings, and Vittorio Ferrari. Cocostuff: Thing and stuff classes in context. In _Computer Vision and Pattern Recognition_ , pages 1209–1218, 2018. 6 

- [4] Nicolas Carion, Francisco Massa, Gabriel Synnaeve, Nicolas Usunier, Alexander Kirillov, and Sergey Zagoruyko. Endto-end object detection with transformers. pages 213–229, 2020. 6, 11 

- [5] Keqin Chen, Zhao Zhang, Weili Zeng, Richong Zhang, Feng Zhu, and Rui Zhao. Shikra: Unleashing multimodal llm’s referential dialogue magic. _arXiv preprint arXiv:2306.15195_ , 2023. 3 

- [6] Wei-Lin Chiang, Zhuohan Li, Zi Lin, Ying Sheng, Zhanghao Wu, Hao Zhang, Lianmin Zheng, Siyuan Zhuang, Yonghao Zhuang, Joseph E Gonzalez, et al. Vicuna: An open-source chatbot impressing gpt-4 with 90% chatgpt quality, 2023. 1 

- [7] Kevin Clark, Minh-Thang Luong, Quoc V. Le, and Christopher D. Manning. Electra: Pre-training text encoders as discriminators rather than generators. In _ICLR_ , 2020. 11 

- [8] Wenliang Dai, Junnan Li, Dongxu Li, Anthony Meng Huat Tiong, Junqi Zhao, Weisheng Wang, Boyang Li, Pascale Fung, and Steven Hoi. Instructblip: Towards generalpurpose vision-language models with instruction tuning, 2023. 2 

- [9] Henghui Ding, Chang Liu, Suchen Wang, and Xudong Jiang. Vision-language transformer and query generation for referring segmentation. In _International Conference on Computer Vision_ , pages 16321–16330, 2021. 7 

- [10] Peng Gao, Jiaming Han, Renrui Zhang, Ziyi Lin, Shijie Geng, Aojun Zhou, Wei Zhang, Pan Lu, Conghui He, Xiangyu Yue, et al. Llama-adapter v2: Parameter-efficient visual instruction model. _arXiv preprint arXiv:2304.15010_ , 2023. 2 

- [11] Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun. Deep residual learning for image recognition, 2015. 12 

- [12] Sahar Kazemzadeh, Vicente Ordonez, Mark Matten, and Tamara Berg. Referitgame: Referring to objects in photographs of natural scenes. In _Empirical Methods in Natural Language Processing_ , pages 787–798, 2014. 5, 6 

- [13] Alexander Kirillov, Eric Mintun, Nikhila Ravi, Hanzi Mao, Chloe Rolland, Laura Gustafson, Tete Xiao, Spencer Whitehead, Alexander C Berg, Wan-Yen Lo, et al. Segment anything. _arXiv preprint arXiv:2304.02643_ , 2023. 2 

- [14] Xin Lai, Zhuotao Tian, Yukang Chen, Yanwei Li, Yuhui Yuan, Shu Liu, and Jiaya Jia. Lisa: Reasoning segmentation 

via large language model. _arXiv preprint arXiv:2308.00692_ , 2023. 2, 3, 4, 5, 6, 7 

- [15] Junnan Li, Dongxu Li, Silvio Savarese, and Steven Hoi. Blip-2: Bootstrapping language-image pre-training with frozen image encoders and large language models. _arXiv preprint arXiv:2301.12597_ , 2023. 2 

- [16] Chang Liu, Henghui Ding, and Xudong Jiang. Gres: Generalized referring expression segmentation. In _Computer Vision and Pattern Recognition_ , pages 23592–23601, 2023. 7 

- [17] Haotian Liu, Chunyuan Li, Qingyang Wu, and Yong Jae Lee. Visual instruction tuning, 2023. 2, 5, 11 

- [18] Ze Liu, Yutong Lin, Yue Cao, Han Hu, Yixuan Wei, Zheng Zhang, Stephen Lin, and Baining Guo. Swin transformer: Hierarchical vision transformer using shifted windows, 2021. 12 

- [19] Zhaoyang Liu, Yinan He, Wenhai Wang, Weiyun Wang, Yi Wang, Shoufa Chen, Qinglong Zhang, Zeqiang Lai, Yang Yang, Qingyun Li, Jiashuo Yu, et al. Interngpt: Solving vision-centric tasks by interacting with chatgpt beyond language. _arXiv preprint arXiv:2305.05662_ , 2023. 2 

- [20] Zhaoyang Liu, Yinan He, Wenhai Wang, Weiyun Wang, Yi Wang, Shoufa Chen, Qinglong Zhang, Zeqiang Lai, Yang Yang, Qingyun Li, et al. Interngpt: Solving vision-centric tasks by interacting with chatgpt beyond language. _arXiv preprint arXiv:2305.05662_ , 2023. 3 

- [21] Jiasen Lu, Christopher Clark, Rowan Zellers, Roozbeh Mottaghi, and Aniruddha Kembhavi. Unified-io: A unified model for vision, language, and multi-modal tasks. In _International Conference on Learning Representations_ , 2022. 2 

- [22] Gen Luo, Yiyi Zhou, Xiaoshuai Sun, Liujuan Cao, Chenglin Wu, Cheng Deng, and Rongrong Ji. Multi-task collaborative network for joint referring expression comprehension and segmentation. In _Computer Vision and Pattern Recognition_ , pages 10034–10043, 2020. 7 

- [23] Microsoft. Deepspeed. Technical report, Microsoft, 2023. 11 

- [24] OpenAI. Chatgpt: A language model for conversational ai. Technical report, OpenAI, 2023. 1 

- [25] OpenAI. Gpt-4 technical report, 2023. 1, 2, 5 

- [26] Baolin Peng, Chunyuan Li, Pengcheng He, Michel Galley, and Jianfeng Gao. Instruction tuning with gpt-4. _arXiv preprint arXiv:2304.03277_ , 2023. 2 

- [27] Zhiliang Peng, Wenhui Wang, Li Dong, Yaru Hao, Shaohan Huang, Shuming Ma, and Furu Wei. Kosmos-2: Grounding multimodal large language models to the world. _arXiv preprint arXiv:2306.14824_ , 2023. 3 

- [28] Vignesh Ramanathan, Anmol Kalia, Vladan Petrovic, Yi Wen, Baixue Zheng, Baishan Guo, Rui Wang, Aaron Marquez, Rama Kovvuri, Abhishek Kadian, et al. Paco: Parts and attributes of common objects. In _Computer Vision and Pattern Recognition_ , pages 7141–7151, 2023. 6 

- [29] Rohan Taori, Ishaan Gulrajani, Tianyi Zhang, Yann Dubois, Xuechen Li, Carlos Guestrin, Percy Liang, and Tatsunori B Hashimoto. Stanford alpaca: An instruction-following llama model, 2023. 2 

- [30] Hugo Touvron, Thibaut Lavril, Gautier Izacard, Xavier Martinet, Marie-Anne Lachaux, Timoth´ee Lacroix, Baptiste Rozi`ere, Naman Goyal, Eric Hambro, Faisal Azhar, et al. Llama: Open and efficient foundation language models. _arXiv preprint arXiv:2302.13971_ , 2023. 2, 11 

- [31] Peng Wang, An Yang, Rui Men, Junyang Lin, Shuai Bai, Zhikang Li, Jianxin Ma, Chang Zhou, Jingren Zhou, and Hongxia Yang. Ofa: Unifying architectures, tasks, and modalities through a simple sequence-to-sequence learning framework. In _International Conference on Machine Learning_ , pages 23318–23340. PMLR, 2022. 2 

Yuan, et al. Generalized decoding for pixel, image, and language. In _Computer Vision and Pattern Recognition_ , pages 15116–15127, 2023. 7 

   - [43] Xueyan Zou, Jianwei Yang, Hao Zhang, Feng Li, Linjie Li, Jianfeng Gao, and Yong Jae Lee. Segment everything everywhere all at once. _Advances in Neural Information Processing Systems_ , 2023. 6, 7 

- [32] Wenhai Wang, Zhe Chen, Xiaokang Chen, Jiannan Wu, Xizhou Zhu, Gang Zeng, Ping Luo, Tong Lu, Jie Zhou, Yu Qiao, et al. Visionllm: Large language model is also an open-ended decoder for vision-centric tasks. _arXiv preprint arXiv:2305.11175_ , 2023. 3 

- [33] Zhaoqing Wang, Yu Lu, Qiang Li, Xunqiang Tao, Yandong Guo, Mingming Gong, and Tongliang Liu. Cris: Clip-driven referring image segmentation. In _Computer Vision and Pattern Recognition_ , pages 11686–11695, 2022. 7 

- [34] Zhao Yang, Jiaqi Wang, Yansong Tang, Kai Chen, Hengshuang Zhao, and Philip HS Torr. Lavt: Language-aware vision transformer for referring image segmentation. In _Computer Vision and Pattern Recognition_ , pages 18155–18165, 2022. 7 

- [35] Haoxuan You, Haotian Zhang, Zhe Gan, Xianzhi Du, Bowen Zhang, Zirui Wang, Liangliang Cao, Shih-Fu Chang, and Yinfei Yang. Ferret: Refer and ground anything anywhere at any granularity. _arXiv preprint arXiv:2310.07704_ , 2023. 3 

- [36] Jiahui Yu, Zirui Wang, Vijay Vasudevan, Legg Yeung, Mojtaba Seyedhosseini, and Yonghui Wu. Coca: Contrastive captioners are image-text foundation models. _arXiv preprint arXiv:2205.01917_ , 2022. 2 

- [37] Renrui Zhang, Jiaming Han, Aojun Zhou, Xiangfei Hu, Shilin Yan, Pan Lu, Hongsheng Li, Peng Gao, and Yu Qiao. Llama-adapter: Efficient fine-tuning of language models with zero-init attention. _arXiv preprint arXiv:2303.16199_ , 2023. 1, 2 

- [38] Susan Zhang, Stephen Roller, Naman Goyal, Mikel Artetxe, Moya Chen, Shuohui Chen, Christopher Dewan, Mona Diab, Xian Li, Xi Victoria Lin, et al. Opt: Open pre-trained transformer language models. _arXiv preprint arXiv:2205.01068_ , 2022. 2 

- [39] Shilong Zhang, Peize Sun, Shoufa Chen, Min Xiao, Wenqi Shao, Wenwei Zhang, Kai Chen, and Ping Luo. Gpt4roi: Instruction tuning large language model on region-of-interest. _arXiv preprint arXiv:2307.03601_ , 2023. 3 

- [40] Bolei Zhou, Hang Zhao, Xavier Puig, Tete Xiao, Sanja Fidler, Adela Barriuso, and Antonio Torralba. Semantic understanding of scenes through the ade20k dataset. _International Journal of Computer Vision_ , 127(3):302–321, 2019. 6 

- [41] Deyao Zhu, Jun Chen, Xiaoqian Shen, Xiang Li, and Mohamed Elhoseiny. Minigpt-4: Enhancing vision-language understanding with advanced large language models. _arXiv preprint arXiv:2304.10592_ , 2023. 2 

- [42] Xueyan Zou, Zi-Yi Dou, Jianwei Yang, Zhe Gan, Linjie Li, Chunyuan Li, Xiyang Dai, Harkirat Behl, Jianfeng Wang, Lu 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/cvpr_2024_PixelLM_Pixel_Reasoning_with_Large_Multimodal_Model_images/cvpr_2024_PixelLM_Pixel_Reasoning_with_Large_Multimodal_Model.pdf-0011-00.png)


## **PixelLM: Pixel Reasoning with Large Multimodal Model** Supplementary Material 

In this supplementary material, we first detail the training configuration, decoder structure, flops calculation, and MUSE evaluation process in Sec.A. We then present an additional experimental analysis of our segmentation codebook and decoder in Sec.B. Furthermore, we offer a more comprehensive analysis of MUSE and the multi-referring segmentation dataset in Sec. C. 

## **A. Implementation Details** 

**Training details.** We give the detailed training configuration in Tab. A.1, and we do _not_ use color jittering, drop path or gradient clip. The gradient accumulation step is set to 10. **Structure of pixel decoder.** The decoder can be divided into three parts based on their functions: _i_ ) the attention block for each scale; _ii_ ) using the output mask from one scale to modulate the features in the next scale; _iii_ ) the fusion of masks from all scales to obtain the final results. We present the PyTorch-style pseudocodes for the overall decoder and each part in Alg. 1. 

**Calculation of TFLOPs.** We compare models’ TFlops (trillion floating-point operations per second) in Tab. 1. The calculation follows the formula in [7] and the script inDeepSpeed [23]. Since the flops for LMMs vary with the generated token length, we standardize it at 512. This length aligns with the common default used by [17, 30] and suffices to accommodate dozens of target objects. 

**Details of the evaluation metric.** Sec. 4.3 provides a concise overview of the MUSE evaluation pipeline. In this section, we delve into a more formal and detailed explanation of its design. Let us denote by _M_ = _{Mg}[G] g_ =1[the ground] truth set of _G_ objects, and _M_[ˆ] = _{M_[ˆ] _k}[K] k_ =1[the set of] _[ K]_[ pre-] dictions. Motivated by [4], assuming _K_ is not equal to _G_ , we use ∅ (no objects) to pad the smaller set and both sets finally have size _P_ = max( _G, K_ ). 

**(1)** We find a bipartite matching between these two sets by searching for a permutation of _P_ elements, _σ ∈_ S _P_ , with the lowest cost: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/cvpr_2024_PixelLM_Pixel_Reasoning_with_Large_Multimodal_Model_images/cvpr_2024_PixelLM_Pixel_Reasoning_with_Large_Multimodal_Model.pdf-0011-08.png)


where _Lmatch_ ( _Mi, Mσ_[ˆ] ( _i_ )) is a pairwise matching cost between ground truth _Mi_ and a prediction with index _σ_ ( _i_ ). We compute this optimal assignment efficiently with the Hungarian algorithm. We define _Lmatch_ ( _Mi, Mσ_[ˆ] ( _i_ )) as _Lbce_ ( _Mi, Mσ_[ˆ] ( _i_ )) + _Ldice_ ( _Mi, Mσ_[ˆ] ( _i_ )). **(2)** Based on the matching results, we modify the generated response _yres_ to _yres[′]_[:][since each] _M_[ˆ] _i_ originates from a seg- 

|aterial||
|---|---|
|confg|value|
|optimizer<br>base learning rate<br>weight decay<br>optimizer monmentum<br>batch size<br>learning rate schedule<br>warmup iterations<br>augmentations<br>_α_<br>_λref_<br>_λdice_|AdamW<br>3.0e-4<br>0<br>_β_1,_β_2=0.9,0.95<br>16<br>WarmipDecayLR<br>100<br>None<br>2.0<br>2.0<br>0.5|



Table A.1. **Training settings.** 

||Table A.|1. **Training settings.**||
|---|---|---|---|
|Codebook<br>design|MUSE Val<br>gIoU<br>cIoU|refCOCO+<br>val<br>testA<br>testB|refCOCOg<br>val(U)<br>test(U)|
|_N_<br>_N × L_|41.0<br>48.3<br>**42.6**<br>**50.7**|64.0<br>69.8<br>57.5<br>**66.3**<br>**71.7**<br>**58.3**|67.9<br>68.4<br>**69.3**<br>**70.5**|



Table A.2. **Sharing tokens across** _L_ **scales.** The first row corresponds to results of sharing tokens across all feature scales. 

|Selected layers|MUSE Val<br>gIoU<br>cIoU|refCOCO+<br>val<br>testA<br>testB|refCOCOg<br>val(U)<br>test(U)|
|---|---|---|---|
|baseline<br>20, 17, 14<br>20, 17<br>23, 14, 10<br>23, 14, 20<br>23, 14|40.1<br>47.2<br>42.0<br>51.5<br>41.2<br>49.1<br>41.8<br>48.0<br>42.3<br>**51.4**<br>**42.6**<br>50.7|61.1<br>65.4<br>54.7<br>66.0<br>71.4<br>58.5<br>65.9<br>71.2<br>58.0<br>65.1<br>68.3<br>57.9<br>66.0<br>71.5<br>**58.5**<br>**66.3**<br>**71.7**<br>58.3|64.3<br>65.6<br>68.8<br>70.6<br>68.0<br>66.4<br>67.5<br>68.0<br>**69.8**<br>70.4<br>69.3<br>**70.5**|



Table A.3. **Multi-scale layer selection.** CLIP-ViT consists of 24 layers. “Baseline” only uses the penultimate (i.e., the 23rd) layer. 

mentation token sequence in _yres_ , we replace each sequence with the GPT-generated description of _Mi_ . 

**(3)** We use a carefully designed prompt for GPT-3.5 to assign a score _si_ to each _M_[ˆ] _i_ in the answer in a single step. An example of this methodology is depicted in Fig. A.1. The empty predictions are directly scored with 0. 

The above three steps assess the model’s capability to generate outputs where masks are intertwined with text descriptions and evaluate how accurately these masks correspond to their respective text descriptions. Then we evaluate the quality of the masks. 

**(4)** The final IoU of each prediction is: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/cvpr_2024_PixelLM_Pixel_Reasoning_with_Large_Multimodal_Model_images/cvpr_2024_PixelLM_Pixel_Reasoning_with_Large_Multimodal_Model.pdf-0011-20.png)


And the final IoU _img_ of each image is: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/cvpr_2024_PixelLM_Pixel_Reasoning_with_Large_Multimodal_Model_images/cvpr_2024_PixelLM_Pixel_Reasoning_with_Large_Multimodal_Model.pdf-0011-22.png)


**Algorithm 1:** Pseudo codes of our pixel decoder. 

||#|Inputs: f<br>~~i~~mg:image features from L scales|Inputs: f<br>~~i~~mg:image features from L scales|Inputs: f<br>~~i~~mg:image features from L scales|Inputs: f<br>~~i~~mg:image features from L scales|[L,C,H,W];|
|---|---|---|---|---|---|---|
|||h|~~s~~eg:segmentation tokens for L scales [L,|||C];|
||#|Variables: lev<br>~~t~~oken: Learnable embeddings||||for L scales|
|||[L,C]; out<br>token:Learnable embeddings [N,|||C];||
|||image<br>pe:position embedding of image features; gamma:mask|||||
|||weighting factors [L]|||||
||#|Functions: SelfAttention();CrossAttention();MLP();|||||
|||up||~~s~~cale(); down<br>scale();|||
|**1 **|**def **||**feature**<br>~~**u**~~**pdate(f, mask):**||||
|||||# f:image feature of one scale.[CxHxW]|||
|||||# mask:output from the attention block|above f.[Cx4Hx4W]||
|**2**||||mask = down<br>~~s~~cale(mask, size=(HxW));# mask: [H,W]|||
|**3**||||f = f * (sigmoid(mask) + 1)# update feature|||
|**4**||||return f|||
|**5 **|**def **||**attention**<br>~~**b**~~**lock(h, f, l):**||||
|||||# h, f:segmentation token and image feature of one|||
|||||scale.[C], [CxHxW]|||
|||||# l:scale index|||
|**6**||||token = cat([out<br>~~t~~oken, h], dim=0) + lev||~~t~~oken[l];|
|||||# token:[N+1,C]|||
|**7**||||attn<br>~~o~~ut = SelfAttention(q=k=v=token);#|self attention||
|||||output:[N+1,C]|||
|**8**||||token = norm(token + attn<br>~~o~~ut)|||
|**9**||||key = f + image<br>pe;# key:[C,H,W]|||
|**10**||||attn<br>~~o~~ut = CrossAttention(q=token, k=key, v=f);# cross|||
|||||attention output:[N+1,C]|||
|**11**||||token = norm(token + attn<br>~~o~~ut + MLP(token)) ;# update|||
|||||token:[N+1,C]|||
|**12**||||attn<br>~~o~~ut = CrossAttention(q=key, k=token,|||
|||||v=token);# cross attention output:[C,H,W]|||
|**13**||||f = norm(f + attn<br>~~o~~ut);# update feature:[C,H,W]|||
|**14**||||f<br>up = up<br>scale(f)# f<br>~~u~~p:Cx4Hx4W|||
|**15**||||token = MLP(token)# token:[N+1,C]|||
|**16**||||mask = token @ f# mask:[N+1,4H,4W]|||
|**17**||||mask = mean(mask, dim=0)# mask:[4H,4W]|||
|**18**||||return mask|||
|**19 **|**def **||**mask**<br>**fusion(mask**<br>~~**l**~~**ist):**||||
|||||# masks:a list of masks from each scale|||
|**20**||||final<br>mask = zeros(4Hx4W);# initialize|empty mask||
|**21**||||**for l, m in enumerate(mask**<br>~~**l**~~**ist):**|||
|**22**||||final<br>~~m~~ask = final<br>~~m~~ask + gamma[l]|*|m|
|**23**||||return final<br>~~m~~ask|||
||#|Main Function|||||
|**24 **|**def **||**pixel**<br>~~**d**~~**ecoder(f**<br>~~**i**~~**mg, h**<br>~~**s**~~**eg):**||||
|||||# masks:a list of mask from each scale|||
|**25**||||mask<br>~~l~~ist = []|||
|**26**||||**for l, (f, h) in enumerate(zip(f**<br>~~**i**~~**mg, h**|~~**s**~~**eg)):**||
|**27**||||**if l < L-1:**|||
|**28**||||f = feature<br>~~u~~pdate(f, mask)|||
|**29**||||;# update features after the scale|L||
|**30**||||mask = attention<br>~~b~~lock(h, f, l)|||
|**31**||||mask<br>~~l~~ist.append(mask)|||
|**32**||||final<br>mask = mask<br>~~f~~usion(mask<br>~~l~~ist)|||
|**33**||||return final<br>~~m~~ask|||



Based on the IoU scores, we can calculate gIoU and cIoU metrics by referring segmentation dataset. 

## **B. More Ablative Experiments.** 

**Multi-scale tokens sharing.** To clearly demonstrate the necessity of our multi-scale segmentation tokens in the codebook _C_ seg, we continue using _L_ multi-scale features, but reduce the original codebook shape from _N × L_ to _N_ (recall that _N_ is the number of tokens in each scale group). This reduction resulted in the remaining _N_ tokens being identi- 

cal (i.e., shared) across all _L_ scales. As Tab. A.2 shows, using a dedicated segmentation token for each scale yields better performance. 

**Multi-scale layer selection.** We experiment with different layer selections as presented in Tab. A.3. The CLIP-ViT-L used in PixelLM consists of 24 layers. To reduce the computational cost of the decoder, we avoid selecting features from all layers of ViT. Instead, we follow the multi-scale feature selection ratio from prior work [11, 18] and consider other selection options. Our experiments reveal that selecting features from layers before the middle does not yield benefits for our task (the fourth row in Tab. A.3). Therefore, we mainly focus on selections from the middle and rear layers. The results show that using features from layers 14 and 23 leads to the best outcomes. 

## **C. More Details about MUSE** 

## **C.1. Data Filtering** 

**GPT-4V filtering.** Although GPT-4V can efficiently understand image content, there are still failure cases in the generated data, which can be summarized in the following two points: 

- Questions are vague and open to multiple interpretations. For example, the question “What should I take with me on my outing?” is extremely vague because “outing” can mean a wide array of activities. 

- Answers that omit semantically equivalent instances. For instance, a question might ask about “choosing a fruit for a snack”, but the answer may only suggest “an apple”, ignoring other fruit visible in the image. 

Therefore, it is necessary to employ a stringent sample filtering process to guarantee the quality of the data. Toward this goal, we develop a GPT-4V assisted data filtering pipeline. This pipeline operates by prompting GPT-4V to evaluate all initial question-answer pairs, based on identified common failure modes. Pairs classified by GPT-4V as falling within these failure categories are removed. This procedure effectively excludes approximately 20% of the preliminary data. The specific prompts used in this process are detailed in Fig. A.1. 

**Human verification.** To ensure high quality in the generated question-answer pairs during the evaluation stage, we further engage experienced human annotators to doublecheck our evaluation set. Our approach is driven by two primary objectives: 

- The questions should follow an intuitive and logical sequence that a person would typically think of when viewing the image. 

- The answers should correspond closely to the way a human would naturally respond to the question 

- The above filtering process effectively ensures that the questions in the MUSE dataset are sufficiently challenging for 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/cvpr_2024_PixelLM_Pixel_Reasoning_with_Large_Multimodal_Model_images/cvpr_2024_PixelLM_Pixel_Reasoning_with_Large_Multimodal_Model.pdf-0013-00.png)


Figure A.1. **Evaluation example.** The left panel illustrates the prompt employed in our GPT evaluation pipeline. The right panel showcases an example of a predicted answer alongside its corresponding modified version, as input to GPT. 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/cvpr_2024_PixelLM_Pixel_Reasoning_with_Large_Multimodal_Model_images/cvpr_2024_PixelLM_Pixel_Reasoning_with_Large_Multimodal_Model.pdf-0013-02.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/cvpr_2024_PixelLM_Pixel_Reasoning_with_Large_Multimodal_Model_images/cvpr_2024_PixelLM_Pixel_Reasoning_with_Large_Multimodal_Model.pdf-0013-03.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/cvpr_2024_PixelLM_Pixel_Reasoning_with_Large_Multimodal_Model_images/cvpr_2024_PixelLM_Pixel_Reasoning_with_Large_Multimodal_Model.pdf-0013-04.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/cvpr_2024_PixelLM_Pixel_Reasoning_with_Large_Multimodal_Model_images/cvpr_2024_PixelLM_Pixel_Reasoning_with_Large_Multimodal_Model.pdf-0013-05.png)


**----- Start of picture text -----**<br>
(a) The number of instances per category (b) Distribution of token count in instance descrip- (c) Distribution of target count in questions.<br>tions.<br>**----- End of picture text -----**<br>


Figure A.2. **Dataset statistics** . Best viewed digitally. 

reasoning, while the answers remain detailed and accurate. 

## **C.2. Dataset Statistics** 

In this section, we systematically analyze our dataset. First, our question-answer pairs are based on over 1000 categories, encompassing a wide spectrum of objects found in daily scenes. Additionally, the descriptions of objects in the answers go beyond mere category names limited to a few tokens. Instead, they offer context-specific descriptions extending to over 100 tokens. This demonstrates that our dataset is rich in perception information, crucial for realworld applications. Finally, we present the statistics regarding the number of objects involved in a data sample. 

**Category statistics.** There are over 1000 categories in MUSE from the original LVIS dataset, and 0.9 million instances with unique descriptions that vary based on the context of the question-answer pairs. Fig. A.2a shows the number of instances per category on all question-answer pairs. The distribution inherits the low-shot nature of LVIS. 

**Token count.** Fig. A.2b presents the distribution of instances by token count in their descriptions, highlighting 

a wide range that exceeds 100 tokens in the most extensive cases. These descriptions are not limited to simple category names; rather, they are substantially enriched with detailed information about each instance, encompassing aspects like appearance, attributes, and relationships with other objects, thanks to our GPT-4V-based data generation pipeline. The depth and variety of information in the dataset bolster the trained model’s generalization capabilities, enabling it to effectively address open-set questions. 

**Target count.** Fig. A.2c presents statistics on the number of targets in each question-answer pair. The average number of targets is 3.7, with the maximum number of targets in a single pair reaching up to 34. This number can cover most scenarios of target reasoning for a single image. 

## **C.3. Multi-referring Segmentation** 

As mentioned in Sec. 5.1, we transform conventional referring segmentation datasets into a multi-referring format for model training. In this subsection, we detail this process. The transformation involves selecting one to three distinct target objects from the annotations of each image. 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/cvpr_2024_PixelLM_Pixel_Reasoning_with_Large_Multimodal_Model_images/cvpr_2024_PixelLM_Pixel_Reasoning_with_Large_Multimodal_Model.pdf-0014-00.png)


Figure C.3. **Example of GPT-4 data generation.** The corresponding image is the same as in Fig. A.1. 

These objects are used to construct questions in the format: Please segment the _<_ objects _>_ in the image, where _<_ objects _>_ represents a list of comma-separated object descriptors. The response format is a list of comma-separated _<_ object _>_ is _<_ SEG _>_ , with _<_ object _>_ being the description of each object. We require that the order of predictions in the answer matches the order of object names in the question and calculate gIoU and cIoU for each prediction based on this. 

## **C.4. More Details about GPT-4 Generated Data.** 

In Sec. 5.5, we create 30,000 additional multi-target question-answer pairs to compare GPT-4 and GPT-4V. The prompts for GPT-4 adhere to similar generation principles but incorporate detailed image captions to offset the absence of visual content. Fig. C.3 demonstrates an example of our GPT-4 data generation process and its typical failures. Given that image content is not directly perceivable, conveying as detailed an image description as possible into GPT-4 is crucial to compensate for this information gap. However, this method often leads to a lack of diversity in the generated question-answer pairs (refer to the questionanswer pair at the bottom of Fig. C.3), characterized by: _i_ ) Numerous questions are composed of simple referring style sub-questions; _ii_ ) The question and answer content is limited to what the image caption describes; _iii_ ) Challenges in generating detailed object descriptions. To address these issues, it might be necessary to introduce more complex annotation details like object relationships, which significantly increases the complexity and burden of data generation. 

CVPR #626 

CVPR #626 

**CVPR 2024 Submission #626. CONFIDENTIAL REVIEW COPY. DO NOT DISTRIBUTE.** 

**001** We sincerely thank all reviewers for the positive com- **002** ments on the novelty and potential impact of our work in **003** multimodal research. For brevity in our response, we use **004** following abbreviations: Dv8c( **R1** ), 7enq( **R2** ), 3UqD( **R3** ). **005 R1- Larger test set.** We have enlarged the MUSE dataset **006** after paper submission. We follow the suggestion to test **007** PixelLM on a larger test set (MUSE _[‡]_ ) which has 15k sam- **008** ples (vs. 4.3k in MUSE). Tab. A shows that PixelLM per- **009** forms similarly on both dataset, confirming the consistency **010** of MUSE results. MUSE _[‡]_ will be made publicly available. **011 R1- refCOCO baseline.** Following LISA, we select the **012** same baselines for comparison and achieve the best results **013** on most metrics. UNINEXT’s better performance could be **014** due to its larger datasets (2M vs. 200k) and higher input res- **015** olution (800px vs. 448px). However, it may struggle with **016** complex reasoning tasks without LLMs. We will provide **017** more detailed comparisons in the final version. **018 R1- Ablations on hyperparameters and vision encoder. 019** We apply the same CE and DICE loss weights ( _λref_ and **020** _λdice_ ) as LISA and use identical hyperparameters for all **021** datasets. Tab. A (rows 2 and 3) shows that the default setting **022** yields the best results. Using a larger laion-CLIP-H (row 5 **023** in Tab. A) further improves PixelLM in all metrics. **024 R1- MUSE enhances pixel reasoning.** We test LISA both **025** with and without MUSE training on ReasonSeg, a single- **026** object reasoning dataset proposed by LISA. Training with **027** MUSE markedly improves LISA’s performance on Reason- **028** Seg val (gIoU: 52.9 _→_ 58.0, cIoU: 54.0 _→_ 59.1), underscor- **029** ing MUSE’s effectiveness in enhancing pixel reasoning. **030 R1- Use discrete tokens for mask generation.** As sug- **031** gested, we follow Pix2Seq v2 and VisionLLM, using 512 **032** tokens for coordinates and training a model to predict 128 **033** polygon points per mask. Tab. A (row 4) shows that this **034** yields worse results than PixelLM across all datasets. The **035** increased sequence length (128 vs. our 6) likely results in **036** more autoregressive prediction errors, especially for multi- **037** instance masks. Combining discrete tokens (like those from **038** VQGAN-related methods) with LLM for segmentation re- **039** mains an open challenge we aim to address in future work. **040 R1- Visualization and Miscs.** Fig. A displays LISA and **041** PixelLM’s ReasonSeg results. PixelLM more accurately lo- **042** cates objects and creates finer masks. Our decoder has 4.1M **043** parameters and requires 0.4 Gflops per mask. We will incor- **044** porate all suggestions and correct typos in the final version. **045 R2- MUSE benefits VL tasks.** To assess MUSE’s impact **046** on VL tasks, we fine-tune LLaVA with solely LLaVA-150k **047** and compare it to LLaVA trained with a mix of LLaVA- **048** 150k and MUSE’s image-text data (MUSE _[text]_ , 2:1 mix- **049** ing ratio). Testing on LLaVA-Bench, a popular benchmark **050** for image captioning, VQA and more, shows that adding **051** MUSE _[text]_ improves all metrics (Tab. B, 3rd vs 2nd rows). **052** This highlights MUSE’s effects in improving model under- **053** standing (including reducing hallucinations) and its gener- 

|Method|refCOCOg|MrefCOCOg|MUSE|MUSE_‡_|
|---|---|---|---|---|
|Default setting<br>_{λref, λdice}_=_{_1,1_}_/_{_0.5,2_}_<br>_α_= 3/4<br>w/ discrete token<br>w/ CLIP-H|69.3<br>68.8/68.6<br>69.3/69.3<br>59.1<br>**72.6**|65.8<br>65.4/65.2<br>65.9/65.7<br>46.8<br>**68.0**|46.3<br>45.8/45.7<br>46.2/46.3<br>36.2<br>**47.8**|45.9<br>45.3/45.0<br>45.6/45.7<br>32.7<br>**47.2**|



Table A. **Ablations.** Results on (M)refCOCOg and MUSE val with cIoU metrics. Default setting: _α_ =2, _λref_ =2, _λdice_ =0.5. 

|Method|Conv|Detail|Complex|All|MrefCOCOg|
|---|---|---|---|---|---|
|PixelLM<br>LLaVA<br>LLaVA w/ MUSE_text_<br>LLaVA w/ MUSE|40.2<br>52.0<br>51.8<br>**52.2**|41.8<br>55.2<br>**56.1**<br>56.0|65.2<br>73.6<br>74.0<br>**74.3**|51.7<br>62.7<br>63.1<br>**63.3**|**69.3**<br>-<br>-<br>60.8|



Table B. **Results on LLaVA-Bench.** MUSE _[text]_ means w/o using the segmentation mask/loss. All results are based on 7B models. alization across various multimodal tasks. 

alization across various multimodal tasks. **054 R2- Impact of using decoder.** Tab. B shows that training **055** with our decoder and mask targets not only endows segmen- **056** tation ability but also improves performance on LLaVA- **057** Bench (4th vs. 3rd rows). This synergy may stem from **058** both the decoder and LLM utilizing image features from **059** the same CLIP, in contrast to the separately trained ‘diffu- **060** sion+LLM’ methods mentioned by the reviewer. PixelLM, **061** employing traditional segmentation datasets for training **062** ( _∼_ 50% are refcoco, ade20k, etc.), scores lower on LLaVA- **063** Bench due to the limited text quality in these datasets, which **064** are often brief and simple. With larger, high-quality datasets **065** like MUSE, we believe both understanding and mask gen- **066** eration can be harmonized in one model. **067 R3- Codebook details.** Our codebook includes special to- **068** kens ‘[SEG1], ..., [SEGN]’ (N=6 in our experiments), added **069** to the LLM’s vocabulary. Like text tokens, we transform **070** them into hidden embeddings using an nn.Embedding **071** layer before they are input into the LLM. The LLM learns **072** to autoregressively generate these special tokens. These to- **073** kens are shared to create masks for different objects. **074 R3- Key factors for surpassing LISA.** PixelLM outper- **075** forms LISA, benefiting from both the codebook and de- **076** coder. Tab. 4 in the main paper shows that the codebook **077** boosts results both with and without using SAM as the de- **078** coder, proving the versatility of the codebook. Even with- **079** out the codebook, using our decoder alone surpasses the one **080** using SAM in cIoU (the primary metric on refCOCO), de- **081** spite being slightly behind in gIoU. This is because cIoU fa- **082** vors larger objects, which might benefit from SAM’s higher **083** input resolution (1024px). Importantly, our decoder can **084** achieve comparable performance with SAM while being **085** much more efficient, even without SAM’s large-scale pre- **086** training datasets, underscoring our decoder’s effectiveness. **087** 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/cvpr_2024_PixelLM_Pixel_Reasoning_with_Large_Multimodal_Model_images/cvpr_2024_PixelLM_Pixel_Reasoning_with_Large_Multimodal_Model.pdf-0015-09.png)


Figure A. **Comparison to LISA on ReasonSeg.** The green part in GT represents the ignored area. 

