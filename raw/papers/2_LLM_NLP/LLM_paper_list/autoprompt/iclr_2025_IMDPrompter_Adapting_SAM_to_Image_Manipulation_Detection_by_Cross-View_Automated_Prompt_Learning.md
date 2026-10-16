Published as a conference paper at ICLR 2025 

# - IMDPROMPTER: ADAPTING SAM TO IMAGE MANIP ULATION DETECTION BY CROSS-VIEW AUTOMATED PROMPT LEARNING 

**Quan Zhang**[1] **, Yuxin Qi** _[∗]_[2] _[∗]_ **, Xi Tang**[1] **, Jinwei Fang**[3] **, Xi Lin**[2] **, Ke Zhang**[1] **, Chun Yuan** _[†]_[1] _[†]_ 1Tsinghua University 2Shanghai Jiao Tong University 3University of Science and Technology of China 

_{_ zhangqua22, ke-zhang19 _}_ @mails.tsinghua.edu.cn, qiyuxin98@sjtu.edu.cn, yuanc@sz.tsinghua.edu.cn 

## ABSTRACT 

Using extensive training data from SA-1B, the Segment Anything Model (SAM) has demonstrated exceptional generalization and zero-shot capabilities, attracting widespread attention in areas such as medical image segmentation and remote sensing image segmentation. However, its performance in the field of image manipulation detection remains largely unexplored and unconfirmed. There are two main challenges in applying SAM to image manipulation detection: a) reliance on manual prompts, and b) the difficulty of single-view information in supporting cross-dataset generalization. To address these challenges, we develops a crossview prompt learning paradigm called IMDPrompter based on SAM. Benefiting from the design of automated prompts, IMDPrompter no longer relies on manual guidance, enabling automated detection and localization. Additionally, we propose components such as Cross-view Feature Perception, Optimal Prompt Selection, and Cross-View Prompt Consistency, which facilitate cross-view perceptual learning and guide SAM to generate accurate masks. Extensive experimental results from five datasets (CASIA, Columbia, Coverage, IMD2020, and NIST16) validate the effectiveness of our proposed method. 

## 1 INTRODUCTION 

With the continuous emergence of powerful editing tools, image manipulation has become unprecedentedly simple. These new opportunities have sparked the creativity of both benevolent and malicious users. In the past, organizing multimedia misinformation activities required complex skills, with attackers limited to splicing, copying, or deleting objects. Today, the rapid development of deep learning has made image editing tools more user-friendly and powerful, enabling users to quickly generate images of fictitious characters or create convincing deepfakes. Generative models can produce realistic image edits based on natural language prompts, perfectly matching inserted elements with the style and lighting of the environmentAvrahami et al. (2023); Nichol et al. (2021). 

However, the risks of these tools falling into the wrong hands are evident. In recent years, governments and funding agencies have shown increasing interest in developing forensic tools capable of addressing such attacks, particularly focusing on local image tampering that alters the semantic content of images Le et al. (2021). In response to these challenges, the fields of multimedia forensics and related sciences have rapidly expanded, proposing various methods and tools for image manipulation detection (IMD) and localization Guillaro et al. (2023). Despite significant progress in this area, the performance of state-of-the-art detectors in practical applications remains insufficient, primarily limited by several shortcomings that require further research: a) limited generalization ability; b) limited robustness; c) poor detection performance. 

To address the shortcomings of existing IMD methods, we turn our attention to foundational models Kirillov et al. (2023); Radford et al. (2021); Jia et al. (2021); Sharif Razavian et al. (2014). Thanks to 

> _∗_ These authors contributed equally. 

> _†_ Corresponding authors. 

1 

Published as a conference paper at ICLR 2025 

large-scale pretraining, foundational models like GPT-4 Achiam et al. (2023), Flamingo Alayrac et al. (2022), and SAM Kirillov et al. (2023) have made significant strides and contributed importantly to societal advancements. Among them, SAM, trained on one billion masks, demonstrates exceptional generalization ability, inspiring our research. However, using SAM for iimage manipulation detection tasks presents significant challenges: a) reliance on manual prompts: SAM’s interactive framework requires predefined prompts for input images, such as points, boxes, or masks. As a category-agnostic segmentation method, these limitations hinder SAM’s ability to achieve fully automated understanding in IMD tasks; b) single-view information struggles to support cross-dataset generalization: previous studies Zhou et al. (2020) have reported that DeepLabv2 Chen et al. (2017) trained on the CASIAv2 Dong et al. (2013) dataset performed well on the CAISAv1 Dong et al. (2013) dataset (sourced from CASIAv2) but poorly on the non-sourced COVER Wen et al. (2016) dataset. Similar behavior has also been observed with FCN Long et al. (2015) in this research. 

To activate SAM’s IMD capabilities, we propose a crossview automated prompt learning paradigm called IMDPrompter, aimed at learning how to generate prompts to enhance SAM’s functionality: a) Inspired by Battle & Gollapudi (2024) using automated prompt optimizers to improve the performance of LLMs, we constructed the automated prompt learning paradigm IMDPrompter, which generates prompt information automatically; b) To enhance SAM’s cross-dataset generalization ability, we introduced three noise views: SRM filtering, Bayer Conv, and Noiseprint, aiming to generate enhanced prompt information from multiple perspectives. To integrate information from multiple views, previous methods Zhai et al. (2023) employed a strategy of weighted averaging of the outputs from mul- 


![](raw/papers/2_LLM_NLP/LLM_paper_list/autoprompt/iclr_2025_IMDPrompter_Adapting_SAM_to_Image_Manipulation_Detection_by_Cross-View_Automated_Prompt_Learning_images/iclr_2025_IMDPrompter_Adapting_SAM_to_Image_Manipulation_Detection_by_Cross-View_Automated_Prompt_Learning.pdf-0002-03.png)


**----- Start of picture text -----**<br>
Prompts<br>1Manually generated prompts<br>RGB SAM<br>Inaccurate mask<br>RGB<br>Prompts Generator<br>      Automatically generated prompts<br>SRM SAF<br>Bayar            SAM<br>Accurate mask<br>Noiseprint<br>             Integration of semantic-agnostic information<br>**----- End of picture text -----**<br>


Figure 1: Improvements of proposed IMDPrompter over existing pipelines of baselines: (a). Automated prompt learning without the need for manual input. (b). Flexible integration of semanticagnostic information crucial for Image Manipulation Detection. 

tiple views. However, due to the inaccuracy of prompts generated from individual views, simple weighted averaging can smooth out the accurate prompts from the optimal view to suboptimal ones. To preserve the accurate prompts from the optimal view, we propose an Optimal Prompt Selection (OPS) module based on minimizing localization loss; meanwhile, we noted that ideally, the prompt information from each view should converge toward the optimal prompt. Based on this motivation, we constructed a Cross-View Prompt Consistency (CPC) constraint to achieve alignment enhancement across views. For the mask generation process, we developed an Attention-based Cross-view Feature Perception (CFP) module and a Multi-Layer Perceptron-based Prompt Mixing module (PMM) to achieve the fusion of cross-view information and the integration of multiple types of prompts sequentially. 

In summary, our main contributions are as follows. 

- We are the first to apply SAM to the field of image manipulation detection and propose an automated prompt learning paradigm, IMDPrompter, eliminating the original SAM’s reliance on manual prompts. 

- We propose modules such as Optimal Prompt Selection and Cross-View Prompt Consistency Constraint, achieving alignment enhancement across views. 

- We propose Cross-view Feature Perception and Prompt Mixing modules, achieving the fusion of cross-view information and the integration of multiple types of prompts. 

2 

Published as a conference paper at ICLR 2025 

- We demonstrate extensive results across five different image manipulation detection datasets, thoroughly validating the strong in-distribution and out-of-distribution image manipulation detection and localization capabilities of IMDPrompter. 

## 2 RELATED WORK 

**Image Manipulation Detection.** Currently, methods for image manipulation detection can be broadly categorized into two types, primarily distinguished by their recognition of manipulated artifacts. Some techniques Wu et al. (2019); Chen et al. (2021); Wu et al. (2022); Bi et al. (2019); Hu et al. (2020); Yang et al. (2020); Marra et al. (2020) rely on detecting abnormal features and often use high-pass noise filters Yang et al. (2020); Li & Huang (2019) to suppress content information. Other methods Park et al. (2018); Kwon et al. (2022); Mareen et al. (2022) attempt to detect inconsistencies in compression within tampered images, as they assume different compression Quality Factors (QFs) before and after the operation. Additionally, some researchers focus their attention on camera-based artifacts, such as model fingerprints Mareen et al. (2022); Cozzolino & Verdoliva (2019); Cozzolino et al. (2015). 

**Foundational Models.** In recent years, foundational models have sparked a tremendous transformation in the field of artificial intelligence. These models, trained on extensive datasets, have demonstrated impressive zero-shot generalization capabilities across various scenarios Kirillov et al. (2023); Radford et al. (2021); Jia et al. (2021); Sharif Razavian et al. (2014). Renowned models such as Chat-GPT Ouyang et al. (2022), GPT-4 Achiam et al. (2023), and Stable Diffusion Rombach et al. (2022) have further propelled the development of artificial intelligence, making significant contributions to human civilization and exerting considerable influence across various industries. Inspired by the success of foundational models in natural language processing (NLP), researchers have begun exploring their potential applications in computer vision. While most of these models are aimed at extracting accessible knowledge from freely available data Alayrac et al. (2022); Radford et al. (2021); Chen et al. (2023), the recent SAM model Kirillov et al. (2023) adopts an innovative approach by constructing a data engine where the model co-develops annotations with environmental datasets. SAM uniquely leverages a vast collection of masks, showcasing robust generalization capabilities. However, it was initially designed as a task-agnostic segmentation model, requiring prompts (i.e., inputs of prior points, bounding boxes, or masks), and therefore does not directly facilitate end-to-end automated segmentation perception. This paper does not delve into the design and training of foundational image manipulation detection models; instead, we explore the potential of utilizing SAM’s powerful universal segmentation capabilities for image manipulation detection and localization. Furthermore, the proposed method of learning prompts can be extended to other visual foundational models beyond SAM. 

**Prompt Learning.** In the past, machine learning tasks were primarily focused on fully supervised learning, where task-specific models were trained only on labeled instances of the target task Krizhevsky et al. (2017). However, over time, there has been a significant shift in learning paradigms, transitioning from fully supervised learning towards _pretraining and fine-tuning_ approaches for downstream tasks. This shift allows models to leverage general features acquired during pretraining Russakovsky et al. (2015); Simonyan & Zisserman (2014); He et al. (2016). More recently, with the advent of foundational models, a new paradigm has emerged known as _pretraining and prompting_ Chen et al. (2023); Lester et al. (2021); Liu et al. (2023); Zhou et al. (2022). In this paradigm, researchers no longer train models specifically for downstream tasks but instead redesign inputs using prompts to reformulate the downstream tasks to align with the original pretraining task Radford et al. (2021); Devlin et al. (2018); Radford et al. (2019). Prompting helps to reduce semantic gaps, bridge the gap between pretraining and fine-tuning, and prevent overfitting of the heads. Since the advent of GPT-3 Brown et al. (2020), prompting has evolved from traditional discrete Liu et al. (2023) and continuous prompt constructions Chen et al. (2023); Zhou et al. (2022) to large-scale model-centric contextual learning Alayrac et al. (2022), instruction tuning Liu et al. (2024); Peng et al. (2023); Gupta et al. (2022), and chaining approaches Wei et al. (2022); Wang et al. (2022a); Zhang et al. (2022). Currently, methods for constructing prompts include manual templates, heuristic-based templates, generation, fine-tuning word embeddings, and pseudo-labeling Liu et al. (2023); Wang et al. (2022b). In this paper, we propose a prompt generator for generating prompts compatible with SAM. 

3 

Published as a conference paper at ICLR 2025 

## 3 PROPOSED METHOD 


![](raw/papers/2_LLM_NLP/LLM_paper_list/autoprompt/iclr_2025_IMDPrompter_Adapting_SAM_to_Image_Manipulation_Detection_by_Cross-View_Automated_Prompt_Learning_images/iclr_2025_IMDPrompter_Adapting_SAM_to_Image_Manipulation_Detection_by_Cross-View_Automated_Prompt_Learning.pdf-0004-02.png)


**----- Start of picture text -----**<br>
Image Mask<br>Image Encoder Decoder<br>FSAM-image FSAF Fmix<br>SAF FSAF PMM<br>SRM EncoderSRM FSRM classify<br>Mopt<br>RGB EncoderRGB fRGB classify CPC<br>OPS Prompt<br>Encoder Fopt<br>CPC<br>Bayar EncoderBayar fBayar classify<br>Bopt<br>Noiseprint NoiseprintEncoder fNoiseprint classify<br>SRM: SRM Filter Bayar: Bayar Filter Noiseprint: Noiseprint++<br>**----- End of picture text -----**<br>


Figure 2: Overall framework of IMDPrompter. The prompter part consists of four views: RGB, SRM, Bayar and Noiseprint. OPS selects the optimal prediction from the four views to generate the best prompt. CPC enhances cross-view consistency. CFP achieves cross-view feature perception fusion. PMM achieves a mixture of multiple prompt information. 

As shown in Figure 2, we established the automated prompt learning paradigm IMDPrompter based on cross-view perception learning, activating the automated detection and localization capabilities of SAM. Specifically, through the optimal prompt selection module, we heuristically select the best prompts from multiple prompts. By employing cross-view prompt consistency, we achieve enhanced alignment across views. Through the cross-view feature perception module and the prompt mixing module, we successively implement cross-view feature fusion and the mixing of various prompt information, thereby enabling efficient automated image manipulation detection and localization based on SAM. 

## 3.1 MULTI-VIEW FEATURE REPRESENTATION 

Current technology Zhou et al. (2020) finds that exploring semantic information from RGB views works well for in-domain (IND) manipulation detection but performs poorly in out-of-domain (OOD) detection. Additionally, utilizing noise views to learn semantic-agnostic information can yield strong performance. Given these findings, we believe that relying solely on RGB view information is insufficient for detecting and localizing manipulations. To address this issue, we introduce information from three semantic-agnostic views: the SRM noise view, the Bayer noise view, and the Noiseprint noise view, with the process as follows: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/autoprompt/iclr_2025_IMDPrompter_Adapting_SAM_to_Image_Manipulation_Detection_by_Cross-View_Automated_Prompt_Learning_images/iclr_2025_IMDPrompter_Adapting_SAM_to_Image_Manipulation_Detection_by_Cross-View_Automated_Prompt_Learning.pdf-0004-07.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/autoprompt/iclr_2025_IMDPrompter_Adapting_SAM_to_Image_Manipulation_Detection_by_Cross-View_Automated_Prompt_Learning_images/iclr_2025_IMDPrompter_Adapting_SAM_to_Image_Manipulation_Detection_by_Cross-View_Automated_Prompt_Learning.pdf-0004-08.png)


4 

Published as a conference paper at ICLR 2025 

In the system, _I_ represents the input image, and Φsam-img denotes the SAM image encoder. _F_ sam represents the features encoded by the SAM image encoder. ΦSRM, ΦBayer and ΦNoiseprint respectively represent the SRM noise map extractor, the Bayer noise map extractor and the Noiseprint noise map extractor. ΦSeg-RGB, ΦSeg-SRM, ΦSeg-Bayer and ΦSeg-Noiseprint each represent the segmenter for the RGB view, SRM view, Bayer view and Noiseprint view, with all four segmenters having identical structures but non-shared parameters. _f_ RGB, _f_ SRM, _f_ Bayer and _f_ Noiseprint represent the features from the RGB, SRM, Bayer, and Noiseprint views, respectively. 

## 3.2 OPTIMAL PROMPT SELECTION 

In the process of generating masked prompts, we utilize the mask probability distributions from four views, _P_ RGB, _P_ SRM, _P_ Bayar, and _P_ Noiseprint, and construct an integrated mask probability distribution _P_ Ens. We then select the optimal mask probability segmentation _P_ opt based on the principle of minimizing segmentation loss. Further, we obtain the mask _M_ opt and bounding box prompts _B_ opt, which are input into SAM’s prompt encoder to obtain the prompt encoding _F_ opt. The process is as follows: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/autoprompt/iclr_2025_IMDPrompter_Adapting_SAM_to_Image_Manipulation_Detection_by_Cross-View_Automated_Prompt_Learning_images/iclr_2025_IMDPrompter_Adapting_SAM_to_Image_Manipulation_Detection_by_Cross-View_Automated_Prompt_Learning.pdf-0005-04.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/autoprompt/iclr_2025_IMDPrompter_Adapting_SAM_to_Image_Manipulation_Detection_by_Cross-View_Automated_Prompt_Learning_images/iclr_2025_IMDPrompter_Adapting_SAM_to_Image_Manipulation_Detection_by_Cross-View_Automated_Prompt_Learning.pdf-0005-05.png)


where ΦRGB _−_ CLS, ΦSRM _−_ CLS, ΦBayer _−_ CLS and ΦNoiseprint _−_ CLS represent classifiers for the RGB, SRM, Bayer, and Noiseprint views, respectively, _G_ represents the one-hot encoded mask labels, _L_ Seg is the segmentation loss function, Φmask represents the mask generation operation, Φbox is the bounding box generation operation, and Φp _−_ enc represents SAM’s prompt encoder. 

## 3.3 CROSS-VIEW PROMPT CONSISTENCY 

Ideally, the segmentation masks from the four views should be consistent with the optimal segmentation mask. Therefore, we constructed a cross-view prompt consistency enhancement loss to achieve enhanced prompt consistency across views. The CPC loss function is expressed as follows: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/autoprompt/iclr_2025_IMDPrompter_Adapting_SAM_to_Image_Manipulation_Detection_by_Cross-View_Automated_Prompt_Learning_images/iclr_2025_IMDPrompter_Adapting_SAM_to_Image_Manipulation_Detection_by_Cross-View_Automated_Prompt_Learning.pdf-0005-09.png)


## 3.4 CROSS-VIEW FEATURE PERCEPTION 

Regarding the decoding process of SAM, relying solely on RGB view information is insufficient for generating accurate segmentation masks. Therefore, we propose a cross-view feature perception (CFP) module that integrates cross-view perception features from RGB, SRM, Bayer, and Noiseprint views, providing rich feature representations for the decoding process of SAM. The entire process is illustrated in Figure 3. ΦCFP represents the cross-view feature perception module, while _F_ CFP denotes the features after cross-view perception fusion. The structure diagram of the CFP module is shown in Figure 3. 

First, the features _f_ RGB, _f_ SRM, _f_ Bayer, and _f_ Noiseprint are aligned with _F_ sam through bicubic interpolation, followed by channel concatenation with _F_ sam after passing through a 1 _×_ 1 convolution. The concatenated features then pass through two stacked convolution blocks (comprising 3 _×_ 3 convolution, instance normalization, and GELU activation), 


![](raw/papers/2_LLM_NLP/LLM_paper_list/autoprompt/iclr_2025_IMDPrompter_Adapting_SAM_to_Image_Manipulation_Detection_by_Cross-View_Automated_Prompt_Learning_images/iclr_2025_IMDPrompter_Adapting_SAM_to_Image_Manipulation_Detection_by_Cross-View_Automated_Prompt_Learning.pdf-0005-13.png)


**----- Start of picture text -----**<br>
C Concatenation Operation<br>SIGMOID<br>Dot Multiplication Operation FC<br>GELU<br>FC<br>Global Average Pool<br>Semantic Selection Attention<br>GELU<br>Instance Normalization<br>Conv3x3<br>Convolution Block   X 2<br>C<br>**----- End of picture text -----**<br>


5 

Published as a conference paper at ICLR 2025 

and the output from the semantic selection attention branch (consisting of global average pooling, a fully connected layer, GELU, and sigmoid) is weighted onto the shortcut branch to produce the integrated feature _F_ CFP. Through the CFP module, features from multiple views are effectively integrated. 


![](raw/papers/2_LLM_NLP/LLM_paper_list/autoprompt/iclr_2025_IMDPrompter_Adapting_SAM_to_Image_Manipulation_Detection_by_Cross-View_Automated_Prompt_Learning_images/iclr_2025_IMDPrompter_Adapting_SAM_to_Image_Manipulation_Detection_by_Cross-View_Automated_Prompt_Learning.pdf-0006-02.png)


## 3.5 PROMPT MIXING MODULE 

Since _F_ CFP aggregates semantically related and unrelated features, it aids the guidance of SAM for image manipulation detection. Therefore, we constructed the PMM module, which is primarily 

based on MLP, to integrate multiple types of prompt information. First, the two types of prompt inputs, _F_ CFP and _F_ opt, are concatenated, followed by an MLP layer to perform dimension transformation to align with the original SAM prompt encoder’s prompt embedding. Then, _F_ mix and _F_ CFP are input into SAM’s mask decoder to achieve the image manipulation localization process. The process is as follows: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/autoprompt/iclr_2025_IMDPrompter_Adapting_SAM_to_Image_Manipulation_Detection_by_Cross-View_Automated_Prompt_Learning_images/iclr_2025_IMDPrompter_Adapting_SAM_to_Image_Manipulation_Detection_by_Cross-View_Automated_Prompt_Learning.pdf-0006-06.png)


where ΦMLP represents the MLP layer operation, _F_ mix represents the integrated prompt embedding, and Φsam _−_ dec represents SAM’s mask decoder, with _P_ sam representing the mask probability prediction output of the mask decoder. 

## 3.6 TRAINING AND INFERENCE PIPELINE 

The training loss function of IMDPrompter includes the segmentation loss from four prompt views _L_ Seg-p, the CPC loss _L_ CPC, the SAM decoder segmentation loss _L_ Seg-sam, and the image-level prediction loss _L_ Img-level. The formula is expressed as follows: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/autoprompt/iclr_2025_IMDPrompter_Adapting_SAM_to_Image_Manipulation_Detection_by_Cross-View_Automated_Prompt_Learning_images/iclr_2025_IMDPrompter_Adapting_SAM_to_Image_Manipulation_Detection_by_Cross-View_Automated_Prompt_Learning.pdf-0006-10.png)


where 


![](raw/papers/2_LLM_NLP/LLM_paper_list/autoprompt/iclr_2025_IMDPrompter_Adapting_SAM_to_Image_Manipulation_Detection_by_Cross-View_Automated_Prompt_Learning_images/iclr_2025_IMDPrompter_Adapting_SAM_to_Image_Manipulation_Detection_by_Cross-View_Automated_Prompt_Learning.pdf-0006-12.png)


In this paper, we use Focal Loss as our _L_ Seg. 

For image-level detection, following the work of Zhai et al. (2023), we adopt an adaptive pooling based on minimizing intra-class prediction variance. The overall computational process is as follows, first using Otsu’s method Otsu et al. (1975) to find a threshold _ω_ 0 that minimizes the intra-class prediction variance: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/autoprompt/iclr_2025_IMDPrompter_Adapting_SAM_to_Image_Manipulation_Detection_by_Cross-View_Automated_Prompt_Learning_images/iclr_2025_IMDPrompter_Adapting_SAM_to_Image_Manipulation_Detection_by_Cross-View_Automated_Prompt_Learning.pdf-0006-15.png)


ˆ where var( _·_ ) denotes variance, _pi,j_ is the pixel-level response at position ( _i, j_ ). Then the image-level prediction is aggregated from pixel-level responses above the threshold and the image-level loss is: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/autoprompt/iclr_2025_IMDPrompter_Adapting_SAM_to_Image_Manipulation_Detection_by_Cross-View_Automated_Prompt_Learning_images/iclr_2025_IMDPrompter_Adapting_SAM_to_Image_Manipulation_Detection_by_Cross-View_Automated_Prompt_Learning.pdf-0006-17.png)


where _y_ ˆA is the image-level prediction, _L_ BCE denotes the binary cross-entropy loss. 

During the inference process of IMDPrompter, since there are no true labels, our OPS module defaults to selecting _P_ Ens for generating masks and bounding box prompts, with other components functioning as during the training process. 

6 

Published as a conference paper at ICLR 2025 

## 4 EXPERIMENT 

**Dataset.** Our method is trained only on the CASIAv2 dataset Dong et al. (2013). For in-distribution (IND) evaluation, we use the CASIAv1 dataset Dong et al. (2013). For out-of-distribution (OOD) evaluation, we use three datasets: Columbia Hsu & Chang (2006), Coverage Wen et al. (2016)and IMD2020 Novozamsky et al. (2020). 

**Evaluation Metrics.** For image-level manipulation detection, we report specificity, sensitivity, and their F1-score (I-F1). The area under the receiver operating characteristic curve (AUC) is also reported as a threshold-independent metric for image-level detection. For pixel-level manipulation localization, we follow previous methods Chen et al. (2021); Zhou et al. (2018b;a); Salloum et al. (2018) to compute pixel accuracy, recall, and their F1-score (P-F1) on manipulated images. The overall performance of image and pixel-level manipulation detection/localization is measured by the harmonic mean of pixel-level and image-level F1-scores Chen et al. (2021), denoted as composite F1 (C-F1), and is sensitive to lower values of P-F1 and I-F1. To ensure fair comparison, a default threshold of 0.5 is used for F1 computation unless otherwise specified. 

**Implementation Details.** In our experiments, unless otherwise specified, we consistently use the VITL backbone of SAM and employ FCN (lightweight architecture based on MobileNet Howard (2017)) as the segmentor for the three prompt views. We maintain image size at 1024×1024, consistent with the original input of the SAM model. To augment training samples, we use data augmentation techniques such as horizontal flipping and random cropping. The image encoder remains frozen during the training phase. All experiments are run on NVIDIA A6000 GPUs. For the optimization process, we train our model using the AdamW optimizer with an initial learning rate of 1e-4. We use a batch size of 4 and train for 100 epochs. We implement a linear warm-up strategy with a cosine annealing scheduler Loshchilov & Hutter (2016) to decay the learning rate. 

## 4.1 COMPARISON WITH STATE-OF-THE-ART METHOD 

Table 1: Pixel-level manipulation detection performance. Performance metric is F1-score (%). The best results for each test set are highlighted in bold and second-best values are underlined. 

|Method|Best threshold|Fixed threshold(0.5)|
|---|---|---|
||CASIA<br>COVER<br>Columbia<br>IMD<br>NIST<br>MEAN|CASIA<br>COVER<br>Columbia<br>IMD<br>NIST<br>MEAN|
|MFCN Salloum et al. (2018)<br>RGB-N Zhou et al. (2018b)<br>H-LSTM Bappy et al. (2019)<br>ManTra-Net Wu et al. (2019)<br>HP-FCN Li & Huang (2019)<br>CR-CNN Yang et al. (2020)<br>GSR-Net Zhou et al. (2018a)<br>SPAN Hu et al. (2020)<br>CAT-Net Kwon et al. (2021)<br>MVSS-Net Chen et al. (2021)<br>Trufor Guillaro et al. (2023)|54.1<br>-<br>61.2<br>-<br>-<br>-<br>40.8<br>37.9<br>-<br>-<br>-<br>-<br>20.9<br>21.3<br>14.2<br>31.0<br>46.6<br>26.80<br>69.2<br>77.2<br>70.9<br>70.5<br>45.5<br>66.66<br>21.4<br>19.9<br>47.1<br>16.9<br>36.0<br>28.26<br>66.2<br>47.0<br>70.4<br>60.0<br>42.8<br>57.28<br>57.4<br>48.9<br>62.2<br>68.7<br>45.6<br>56.56<br>68.8<br>71.8<br>77.4<br>69.6<br>68.3<br>71.18<br>57.3<br>48.5<br>77.6<br>51.7<br>59.9<br>59.00<br>75.3<br>82.4<br>70.3<br>75.7<br>73.7<br>75.48<br>83.5<br>74.1<br>82.1<br>-<br>68.8<br>-|-<br>-<br>-<br>-<br>-<br>-<br>-<br>-<br>-<br>-<br>-<br>-<br>15.4<br>16.3<br>13.0<br>19.5<br>35.4<br>19.92<br>15.5<br>28.6<br>36.4<br>18.7<br>0.0<br>19.84<br>15.4<br>0.3<br>6.7<br>11.2<br>12.1<br>9.14<br>40.5<br>29.1<br>43.6<br>26.2<br>23.8<br>32.64<br>38.7<br>28.5<br>61.3<br>24.3<br>28.3<br>36.22<br>18.4<br>17.2<br>48.7<br>17.0<br>22.1<br>24.68<br>13.6<br>12.9<br>55.5<br>5.4<br>17.9<br>21.06<br>45.2<br>45.3<br>63.8<br>26.0<br>29.2<br>41.90<br>73.7<br>60.0<br>85.9<br>-<br>39.9<br>-|
|FCN Long et al. (2015)|74.2<br>57.3<br>58.6<br>64.5<br>50.7<br>61.06|44.1<br>19.9<br>22.3<br>21.0<br>16.7<br>24.80|
|IMDPromter|**85.1**<br>**83.5**<br>**87.3**<br>**76.3**<br>**74.6**<br>**81.36**|**76.3**<br>**63.6**<br>**87.3**<br>**30.6**<br>**41.1**<br>**59.78**|
|p|+10.9<br>+26.2<br>+28.7<br>+11.8<br>+23.9<br>+20.3|+32.2<br>+43.7<br>+65.0<br>+9.6<br>+24.4<br>+34.98|



Table 2: Image-level manipulation detection performance. Decision threshold: 0.5. NIST16 is excluded as it lacks genuine images. The best results for each test set are highlighted in bold and second-best values are underlined. 

|Method|CASIA|COVER||Columbia|IMD|MEAN|
|---|---|---|---|---|---|---|
||AUC<br>Sen.<br>Spe.<br>F1|AUC<br>Sen.<br>Spe.|F1|AUC<br>Sen.<br>Spe.<br>F1|AUC<br>Sen.<br>Spe.<br>F1|AUC<br>F1|
|H-LSTM Bappy et al. (2019)<br>ManTra-Net Wu et al. (2019)<br>CR-CNN Yang et al. (2020)<br>GSR-Net Zhou et al. (2018a)<br>SPAN Hu et al. (2020)<br>CAT-Net Kwon et al. (2021)<br>MVSS-Net Chen et al. (2021)<br>Trufor Guillaro et al. (2023)|0.498<br>99.7<br>0.0<br>0.0<br>0.500<br>100.0<br>0.0<br>0.0<br>0.719<br>93.0<br>13.9<br>24.2<br>0.500<br>99.4<br>0.0<br>0.0<br>0.500<br>100.0<br>0.0<br>0.0<br>0.647<br>23.9<br>92.1<br>38.0<br>0.937<br>61.5<br>98.8<br>75.8<br>0.916<br>-<br>-<br>-|0.500<br>100.0<br>0.0<br>0.500<br>100.0<br>0.0<br>0.566<br>96.7<br>7.0<br>0.515<br>100.0<br>0.0<br>0.500<br>100.0<br>0.0<br>0.557<br>28.0<br>80.0<br>0.731<br>94.0<br>14.0<br>0.770<br>-<br>-|0.0<br>0.0<br>13.1<br>0.0<br>0.0<br>41.5<br>24.4<br>-|0.506<br>100.0<br>1.1<br>2.2<br>0.701<br>100.0<br>0.0<br>0.0<br>0.783<br>96.1<br>24.6<br>39.2<br>0.502<br>100.0<br>1.1<br>2.2<br>0.500<br>100.0<br>0.0<br>0.0<br>0.971<br>87.2<br>96.2<br>91.5<br>0.980<br>66.9<br>100.0<br>80.2<br>**0.996**<br>-<br>-<br>-|0.501<br>100.0<br>0.0<br>0.0<br>0.500<br>100.0<br>0.0<br>0.0<br>0.615<br>92.9<br>12.3<br>21.7<br>0.500<br>100.0<br>0.0<br>0.0<br>0.500<br>100.0<br>0.0<br>0.0<br>0.586<br>27.5<br>81.6<br>41.1<br>0.656<br>91.5<br>22.0<br>35.5<br>-<br>-<br>-<br>-|0.5013<br>0.55<br>0.5503<br>0.00<br>0.6708<br>24.55<br>0.5043<br>0.55<br>0.5000<br>0.00<br>0.6903<br>53.03<br>0.8260<br>53.98<br>-<br>-|
|FCN Long et al. (2015)|0.770<br>72.8<br>64.3<br>68.3|0.541<br>90.0<br>10.0|18.0|0.762<br>95.0<br>32.2<br>48.1|0.502<br>84.6<br>15.5<br>26.2|0.6438<br>40.15|
|IMDP|**0.978**<br>91.6<br>99.3<br>**77.3**|**0.796**<br>96.8<br>61.3|**70.3**|0.983<br>97.3<br>91.6<br>**93.6**|**0.671**<br>71.6<br>73.2<br>**63.7**|**0.8570**<br>**76.23**|
|rompter|+0.208<br>+18.8<br>+35.0<br>+9.0|+0.255<br>+6.8<br>+51.3|+52.3|+0.221<br>+2.3<br>+59.4<br>+45.5|+0.169<br>-13.0<br>+57.7<br>+37.5|+0.2132<br>+36.08|



7 

Published as a conference paper at ICLR 2025 


![](raw/papers/2_LLM_NLP/LLM_paper_list/autoprompt/iclr_2025_IMDPrompter_Adapting_SAM_to_Image_Manipulation_Detection_by_Cross-View_Automated_Prompt_Learning_images/iclr_2025_IMDPrompter_Adapting_SAM_to_Image_Manipulation_Detection_by_Cross-View_Automated_Prompt_Learning.pdf-0008-01.png)


**----- Start of picture text -----**<br>
Image GT ManTra-Net MVSS-Net Trufor IMDPrompter<br>**----- End of picture text -----**<br>


Figure 4: Visualization Comparison of Image Manipulation Detection and Localization Results 

Table 3: Evaluation of each component of our method in IND dataset. 

Table 4: Evaluation of each component of our method in OOD dataset. 

||CFP<br>CPC<br>OPS<br>PMM|CASIA|||CFP<br>CPC<br>OPS<br>PMM|COVER|
|---|---|---|---|---|---|---|
|||I-F1<br>P-F1<br>C-F1||||I-F1<br>P-F1<br>C-F1|
|1<br>2<br>3<br>4<br>5|✓<br>✓<br>✓<br>✓<br>✓<br>✓<br>✓<br>✓<br>✓<br>✓|70.6<br>70.3<br>70.4<br>73.2<br>73.1<br>73.1<br>74.3<br>74.3<br>74.3<br>75.1<br>76.1<br>75.6<br>77.3<br>76.3<br>76.8||6<br>7<br>8<br>9<br>10|✓<br>✓<br>✓<br>✓<br>✓<br>✓<br>✓<br>✓<br>✓<br>✓|20.4<br>39.8<br>27.0<br>49.6<br>54.3<br>51.8<br>58.6<br>57.0<br>57.8<br>65.2<br>61.4<br>63.2<br>70.3<br>63.6<br>66.8|



**Pixel-level Manipulation Detection** Table 1 illustrates the pixel-level detection performance of different models. We evaluate the F1-score under two settings: best threshold and fixed threshold (0.5). From Table 1, it is evident that IMDPrompter achieves the best performance across nearly all datasets. Under the optimal threshold setting, we achieved an average F1 Score of 81.36%. Under the fixed threshold setting, IMDPrompter achieved an average F1 Score of 59.78%, indicating that our proposed IMDPrompter demonstrates better robustness in threshold settings. 

**Image-level Manipulation Detection.** Table 2 presents the image-level performance of different models. For image-level performance, we use a default decision threshold of 0.5. Once again, IMDPrompter emerges as the top performer, leveraging our prompt learning paradigm to achieve higher specificity in most test settings, thereby reducing false positives. Additionally, IMDPrompter achieves the best average AUC and average F1-scores. The average F1-score of IMDPrompter significantly surpasses that of the second-best method, MVSS-Net, with an improvement of 22.25%. 

**Qualitative Results.** As shown in Figure 10, we visualized the image manipulation localization results of four methods: ManTra-Net, MVSS-Net, Trufor, and IMDPrompter. It is evident that IMDPrompter generates more accurate confidence maps for image manipulation localization, achieving better detection and localization. 

## 4.2 ABLATION STUDIES 

To reveal the impact of different components, we evaluated the proposed model under various settings, analyzing the distinct effects of each component on in-domain and out-of-domain datasets. As shown 

8 

Published as a conference paper at ICLR 2025 

Table 5: Ablation Study of View Combinations 

|RGB<br>Noiseprint<br>Bayar<br>SRM|CASIA|COVER|Columbia|IMD|
|---|---|---|---|---|
||I-AUC<br>I-F1<br>P-F1<br>C-F1|I-AUC<br>I-F1<br>P-F1<br>C-F1|I-AUC<br>I-F1<br>P-F1<br>C-F1|I-AUC<br>I-F1<br>P-F1<br>C-F1|
|✓<br>✓<br>✓<br>✓<br>✓<br>✓<br>✓<br>✓<br>✓<br>✓<br>✓<br>✓<br>✓|0.856<br>73.60<br>69.60<br>71.50<br>0.903<br>75.90<br>73.60<br>74.70<br>0.863<br>74.30<br>70.60<br>72.40<br>0.896<br>75.10<br>71.60<br>73.30<br>0.941<br>76.90<br>75.20<br>76.00<br>0.962<br>77.10<br>75.90<br>76.50<br>0.978<br>77.30<br>76.30<br>76.80|0.556<br>19.60<br>38.60<br>26.00<br>0.761<br>62.90<br>58.70<br>60.70<br>0.706<br>61.30<br>56.30<br>58.70<br>0.713<br>59.60<br>55.60<br>57.50<br>0.781<br>66.80<br>61.90<br>64.30<br>0.788<br>68.70<br>62.40<br>65.40<br>0.796<br>70.30<br>63.60<br>66.80|0.786<br>53.60<br>24.90<br>34.00<br>0.946<br>88.60<br>83.50<br>86.00<br>0.889<br>83.40<br>81.60<br>82.50<br>0.906<br>84.60<br>79.30<br>81.90<br>0.969<br>91.70<br>85.40<br>88.40<br>0.977<br>92.40<br>86.50<br>89.40<br>0.983<br>93.60<br>87.30<br>90.30|0.531<br>28.10<br>21.20<br>24.20<br>0.636<br>59.90<br>27.90<br>38.10<br>0.601<br>54.60<br>25.70<br>34.90<br>0.593<br>56.30<br>24.30<br>33.90<br>0.655<br>61.50<br>28.80<br>39.20<br>0.660<br>62.60<br>29.50<br>40.10<br>0.671<br>63.70<br>30.60<br>41.30|



Table 6: Ablation Study of SAM 

|Method|CASIA|COVER|Columbia|IMD|
|---|---|---|---|---|
||I-AUC<br>I-F1<br>P-F1<br>C-F1|I-AUC<br>I-F1<br>P-F1<br>C-F1|I-AUC<br>I-F1<br>P-F1<br>C-F1|I-AUC<br>I-F1<br>P-F1<br>C-F1|
|FCN<br>FCN+<br>IMDPrompter|0.770<br>68.3<br>44.1<br>53.6<br>0.873<br>73.4<br>61.3<br>66.8<br>0.978<br>77.3<br>76.3<br>76.8|0.541<br>18.0<br>19.9<br>18.9<br>0.681<br>49.2<br>43.9<br>46.4<br>0.796<br>70.3<br>63.6<br>66.8|0.762<br>48.1<br>22.3<br>30.5<br>0.871<br>72.6<br>52.4<br>60.9<br>0.983<br>93.6<br>87.3<br>90.3|0.502<br>26.2<br>21.0<br>23.3<br>0.589<br>43.1<br>25.1<br>31.7<br>0.671<br>63.7<br>30.6<br>41.3|



in Tables 3 and 4, we conducted ablation experiments on the in-domain dataset CASIA and the out-of-domain dataset COVER, respectively. 

**Baseline.** From _Experiment 1_ , it is known that our baseline method achieved performance scores of 70.6%, 70.3%, and 70.4% on the I-F1, P-F1, and C-F1 metrics respectively in the CASIA dataset. In the COVER dataset, it scored 20.4%, 39.8%, and 27.0% on the same metrics. These experiments show that using only RGB visual view information has limited generalization ability in out-of-domain datasets. 

**Impact of CFP.** From _Experiment 2_ , By implementing cross-view feature perception fusion through the CFP module, the I-F1, P-F1, and C-F1 metrics on the CASIA dataset improved by 2.6%, 2.8%, and 2.7%, respectively. Experiment VII showed that on the COVER dataset, the I-F1, P-F1, and C-F1 metrics increased by 29.2%, 14.5%, and 24.8%, respectively. This indicates that cross-view feature perception fusion can enhance performance on both in-domain and out-of-domain datasets, especially on out-of-domain datasets. 

**Impact of CPC.** From _Experiment 3_ , the inclusion of CPC, which introduced cross-view consistency enhancement, resulted in performance gains of 3.7%, 4.0%, and 3.9% on the I-F1, P-F1, C-F1 metrics respectively in the CASIA dataset. Experiment VIII showed performance gains of 38.2%, 17.2%, and 30.8% respectively on the COVER dataset, confirming that the prompt information between different views is complementary and integrative enhancement can bring objective performance improvements. 

**Impact of OPS.** The inclusion of OPS for heuristic selection of optimal prompts resulted in improvements of 4.5%, 5.8%, and 5.2% in the I-F1, P-F1, and C-F1 metrics on the CASIA dataset, respectively. Experiment IX showed gains of 44.8%, 21.6%, and 36.2% respectively on the COVER dataset, indicating that simply adding together prompts from different views is not enough to generate the best prompts, and a optimal prompts selection strategy can optimize the prompt selection process. From _Experiment 4_ , the inclusion of OPS adaptive selection of the optimal prompts resulted in performance gains of 4.5%, 5.8%, and 5.2% on the I-F1, P-F1, C-F1 metrics respectively in the CASIA dataset. 

**Impact of PMM.** From _Experiment 5_ , with the inclusion of PMM which fully integrates various types of prompt information, there was an increase in performance of 6.7%, 6.0%, and 6.4% on the I-F1, P-F1, C-F1 metrics respectively in the CASIA dataset. _Experiment 10_ showed an increase of 49.9%, 23.8%, and 39.8% respectively on the COVER dataset, further demonstrating the effectiveness of PMM. 

**Impact of View Combinations.** As shown in Table 5, the performance of using the RGB view alone does not differ significantly from that of using semantic-agnostic views such as SRM, Bayer, and Noiseprint individually on the in-domain test set CASIA. In contrast, for the out-of-domain test sets COVER, Columbia, and IMD, there is a significant performance gap between the RGB view and the semantic-agnostic views. This indicates that semantic-agnostic views play a crucial role in generalization on out-of-domain datasets. Additionally, we conducted ablation studies on various view combinations, and using both the RGB view and the Noiseprint view simultaneously resulted in substantial performance improvements. This demonstrates that the Noiseprint view serves as a critical semantic-agnostic feature. Next, we sequentially introduced the Bayer and SRM views, 

9 

Published as a conference paper at ICLR 2025 


![](raw/papers/2_LLM_NLP/LLM_paper_list/autoprompt/iclr_2025_IMDPrompter_Adapting_SAM_to_Image_Manipulation_Detection_by_Cross-View_Automated_Prompt_Learning_images/iclr_2025_IMDPrompter_Adapting_SAM_to_Image_Manipulation_Detection_by_Cross-View_Automated_Prompt_Learning.pdf-0010-01.png)


**----- Start of picture text -----**<br>
H-LSTM CR-CNN SPAN MVSS-Net<br>ManTra-Net GSR-Net CAT-Net IMDPrompter<br>80 80 80<br>60 60 60<br>40 40 40<br>20 20 20<br>0 0 0<br>100 90 80 70 60 50 100 90 80 70 60 50 100 90 80 70 60 50<br>JPEG Compression Quality JPEG Compression Quality JPEG Compression Quality<br>(a) (b) (c)<br>80 80<br>60 60 60<br>40 40 40<br>20 20<br>20<br>0 0<br>0 5 11 17 23 29 0 5 11 17 23 29 0 5 11 17 23 29<br>Gaussian Blur Kernel Size  Gaussian Blur Kernel Size Gaussian Blur Kernel Size<br>(d) (e) (f)<br>Com-F1 [%]<br>Pixel-level F1 [%] Image-level F1 [%]<br>Com-F1 [%]<br>Pixel-level F1 [%] Image-level F1 [%]<br>**----- End of picture text -----**<br>


Figure 5: Robustness Analysis Against JPEG Compression and Gaussian Blur Interference 

achieving further steady improvements across all metrics. When we combined the RGB, Noiseprint, SRM, and Bayer views, our method achieved optimal performance, validating the effectiveness of this combination. 

**Impact of SAM.** As shown in Table 6, we constructed an FCN+ model that integrates four views without using SAM, further analyzing the impact of SAM on IMD. Due to the lack of potential priors from a large-scale pre-training dataset, the performance ceiling of FCN+ is relatively low. In contrast, our IMDPrompter fully leverages these potential priors, thereby enhancing the performance ceiling 

**Impact of Quality Degradation.** As shown in Figure 5, following Dong et al. (2022), we evaluated the robustness of the models under two common image processing operations encountered during the dissemination of images on the internet, namely JPEG compression and Gaussian blur. Comparing these two operations, Gaussian blur has a more significant impact on detection performance, especially when using larger 17x17 scale convolution kernels. Compared to previous methods, our IMDPrompter exhibits better robustness. 

## 5 CONSLUSION 

In this paper, we propose a cross-view perceptual prompt learning paradigm, IMDPrompter, which applies SAM to the image manipulation detection task for the first time. We sequentially introduce components such as optimal prompt selection, cross-view prompt consistency, cross-view feature perception, and prompt mixing modules to achieve efficient and automated image manipulation detection and localization. Our IMDPrompter demonstrates significant image-level and pixel-level manipulation detection performance in both in-domain (IND) and out-of-domain (OOD) scenarios, as well as across various robustness evaluation settings. 

## REFERENCES 

> Josh Achiam, Steven Adler, Sandhini Agarwal, Lama Ahmad, Ilge Akkaya, Florencia Leoni Aleman, Diogo Almeida, Janko Altenschmidt, Sam Altman, Shyamal Anadkat, et al. Gpt-4 technical report. _arXiv preprint arXiv:2303.08774_ , 2023. 

10 

Published as a conference paper at ICLR 2025 

Jean-Baptiste Alayrac, Jeff Donahue, Pauline Luc, Antoine Miech, Iain Barr, Yana Hasson, Karel Lenc, Arthur Mensch, Katherine Millican, Malcolm Reynolds, et al. Flamingo: a visual language model for few-shot learning. _Advances in neural information processing systems_ , 35:23716–23736, 2022. 

- Omri Avrahami, Ohad Fried, and Dani Lischinski. Blended latent diffusion. _ACM transactions on graphics (TOG)_ , 42(4):1–11, 2023. 

- Jawadul H Bappy, Cody Simons, Lakshmanan Nataraj, BS Manjunath, and Amit K Roy-Chowdhury. Hybrid lstm and encoder–decoder architecture for detection of image forgeries. _IEEE transactions on image processing_ , 28(7):3286–3300, 2019. 

- Rick Battle and Teja Gollapudi. The unreasonable effectiveness of eccentric automatic prompts. _arXiv preprint arXiv:2402.10949_ , 2024. 

- Xiuli Bi, Yang Wei, Bin Xiao, and Weisheng Li. Rru-net: The ringed residual u-net for image splicing forgery detection. In _Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition Workshops_ , pp. 0–0, 2019. 

- Tom Brown, Benjamin Mann, Nick Ryder, Melanie Subbiah, Jared D Kaplan, Prafulla Dhariwal, Arvind Neelakantan, Pranav Shyam, Girish Sastry, Amanda Askell, et al. Language models are few-shot learners. _Advances in neural information processing systems_ , 33:1877–1901, 2020. 

- Keyan Chen, Xiaolong Jiang, Yao Hu, Xu Tang, Yan Gao, Jianqi Chen, and Weidi Xie. Ovarnet: Towards open-vocabulary object attribute recognition. In _Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition_ , pp. 23518–23527, 2023. 

- Liang-Chieh Chen, George Papandreou, Iasonas Kokkinos, Kevin Murphy, and Alan L Yuille. Deeplab: Semantic image segmentation with deep convolutional nets, atrous convolution, and fully connected crfs. _IEEE transactions on pattern analysis and machine intelligence_ , 40(4):834–848, 2017. 

- Xinru Chen, Chengbo Dong, Jiaqi Ji, Juan Cao, and Xirong Li. Image manipulation detection by multi-view multi-scale supervision. In _Proceedings of the IEEE/CVF International Conference on Computer Vision_ , pp. 14185–14193, 2021. 

- Davide Cozzolino and Luisa Verdoliva. Noiseprint: A cnn-based camera model fingerprint. _IEEE Transactions on Information Forensics and Security_ , 15:144–159, 2019. 

- Davide Cozzolino, Giovanni Poggi, and Luisa Verdoliva. Splicebuster: A new blind image splicing detector. In _2015 IEEE International Workshop on Information Forensics and Security (WIFS)_ , pp. 1–6. IEEE, 2015. 

- Jacob Devlin, Ming-Wei Chang, Kenton Lee, and Kristina Toutanova. Bert: Pre-training of deep bidirectional transformers for language understanding. _arXiv preprint arXiv:1810.04805_ , 2018. 

- Chengbo Dong, Xinru Chen, Ruohan Hu, Juan Cao, and Xirong Li. Mvss-net: Multi-view multi-scale supervised networks for image manipulation detection. _IEEE Transactions on Pattern Analysis and Machine Intelligence_ , 45(3):3539–3553, 2022. 

- Jing Dong, Wei Wang, and Tieniu Tan. Casia image tampering detection evaluation database. In _2013 IEEE China summit and international conference on signal and information processing_ , pp. 422–426. IEEE, 2013. 

- Fabrizio Guillaro, Davide Cozzolino, Avneesh Sud, Nicholas Dufour, and Luisa Verdoliva. Trufor: Leveraging all-round clues for trustworthy image forgery detection and localization. In _Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition_ , pp. 20606–20615, 2023. 

- Prakhar Gupta, Cathy Jiao, Yi-Ting Yeh, Shikib Mehri, Maxine Eskenazi, and Jeffrey P Bigham. Instructdial: Improving zero and few-shot generalization in dialogue through instruction tuning. _arXiv preprint arXiv:2205.12673_ , 2022. 

11 

Published as a conference paper at ICLR 2025 

- Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun. Deep residual learning for image recognition. In _Proceedings of the IEEE conference on computer vision and pattern recognition_ , pp. 770–778, 2016. 

- Andrew G Howard. Mobilenets: Efficient convolutional neural networks for mobile vision applications. _arXiv preprint arXiv:1704.04861_ , 2017. 

- Yu-Feng Hsu and Shih-Fu Chang. Detecting image splicing using geometry invariants and camera characteristics consistency. In _2006 IEEE International Conference on Multimedia and Expo_ , pp. 549–552. IEEE, 2006. 

- Xuefeng Hu, Zhihan Zhang, Zhenye Jiang, Syomantak Chaudhuri, Zhenheng Yang, and Ram Nevatia. Span: Spatial pyramid attention network for image manipulation localization. In _Computer Vision– ECCV 2020: 16th European Conference, Glasgow, UK, August 23–28, 2020, Proceedings, Part XXI 16_ , pp. 312–328. Springer, 2020. 

- Chao Jia, Yinfei Yang, Ye Xia, Yi-Ting Chen, Zarana Parekh, Hieu Pham, Quoc Le, Yun-Hsuan Sung, Zhen Li, and Tom Duerig. Scaling up visual and vision-language representation learning with noisy text supervision. In _International conference on machine learning_ , pp. 4904–4916. PMLR, 2021. 

- Alexander Kirillov, Eric Mintun, Nikhila Ravi, Hanzi Mao, Chloe Rolland, Laura Gustafson, Tete Xiao, Spencer Whitehead, Alexander C Berg, Wan-Yen Lo, et al. Segment anything. In _Proceedings of the IEEE/CVF International Conference on Computer Vision_ , pp. 4015–4026, 2023. 

- Vladimir V Kniaz, Vladimir Knyaz, and Fabio Remondino. The point where reality meets fantasy: Mixed adversarial generators for image splice detection. _Advances in neural information processing systems_ , 32, 2019. 

- Alex Krizhevsky, Ilya Sutskever, and Geoffrey E Hinton. Imagenet classification with deep convolutional neural networks. _Communications of the ACM_ , 60(6):84–90, 2017. 

- Myung-Joon Kwon, In-Jae Yu, Seung-Hun Nam, and Heung-Kyu Lee. Cat-net: Compression artifact tracing network for detection and localization of image splicing. In _Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision_ , pp. 375–384, 2021. 

- Myung-Joon Kwon, Seung-Hun Nam, In-Jae Yu, Heung-Kyu Lee, and Changick Kim. Learning jpeg compression artifacts for image manipulation detection and localization. _International Journal of Computer Vision_ , 130(8):1875–1895, 2022. 

- Trung-Nghia Le, Huy H Nguyen, Junichi Yamagishi, and Isao Echizen. Openforensics: Large-scale challenging dataset for multi-face forgery detection and segmentation in-the-wild. In _Proceedings of the IEEE/CVF international conference on computer vision_ , pp. 10117–10127, 2021. 

Brian Lester, Rami Al-Rfou, and Noah Constant. The power of scale for parameter-efficient prompt tuning. _arXiv preprint arXiv:2104.08691_ , 2021. 

- Haodong Li and Jiwu Huang. Localization of deep inpainting using high-pass fully convolutional network. In _proceedings of the IEEE/CVF international conference on computer vision_ , pp. 8301–8310, 2019. 

- Haotian Liu, Chunyuan Li, Qingyang Wu, and Yong Jae Lee. Visual instruction tuning. _Advances in neural information processing systems_ , 36, 2024. 

- Pengfei Liu, Weizhe Yuan, Jinlan Fu, Zhengbao Jiang, Hiroaki Hayashi, and Graham Neubig. Pre-train, prompt, and predict: A systematic survey of prompting methods in natural language processing. _ACM Computing Surveys_ , 55(9):1–35, 2023. 

- Jonathan Long, Evan Shelhamer, and Trevor Darrell. Fully convolutional networks for semantic segmentation. In _Proceedings of the IEEE conference on computer vision and pattern recognition_ , pp. 3431–3440, 2015. 

- Ilya Loshchilov and Frank Hutter. Sgdr: Stochastic gradient descent with warm restarts. _arXiv preprint arXiv:1608.03983_ , 2016. 

12 

Published as a conference paper at ICLR 2025 

Jun Ma, Yuting He, Feifei Li, Lin Han, Chenyu You, and Bo Wang. Segment anything in medical images. _Nature Communications_ , 15(1):654, 2024. 

- Hannes Mareen, Dante Vanden Bussche, Fabrizio Guillaro, Davide Cozzolino, Glenn Van Wallendael, Peter Lambert, and Luisa Verdoliva. Comprint: Image forgery detection and localization using compression fingerprints. In _International Conference on Pattern Recognition_ , pp. 281–299. Springer, 2022. 

- Francesco Marra, Diego Gragnaniello, Luisa Verdoliva, and Giovanni Poggi. A full-image fullresolution end-to-end-trainable cnn framework for image forgery detection. _IEEE Access_ , 8: 133488–133502, 2020. 

- Alex Nichol, Prafulla Dhariwal, Aditya Ramesh, Pranav Shyam, Pamela Mishkin, Bob McGrew, Ilya Sutskever, and Mark Chen. Glide: Towards photorealistic image generation and editing with text-guided diffusion models. _arXiv preprint arXiv:2112.10741_ , 2021. 

- Adam Novozamsky, Babak Mahdian, and Stanislav Saic. Imd2020: A large-scale annotated dataset tailored for detecting manipulated images. In _Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision Workshops_ , pp. 71–80, 2020. 

- Nobuyuki Otsu et al. A threshold selection method from gray-level histograms. _Automatica_ , 11 (285-296):23–27, 1975. 

- Long Ouyang, Jeffrey Wu, Xu Jiang, Diogo Almeida, Carroll Wainwright, Pamela Mishkin, Chong Zhang, Sandhini Agarwal, Katarina Slama, Alex Ray, et al. Training language models to follow instructions with human feedback. _Advances in neural information processing systems_ , 35:27730– 27744, 2022. 

- Jinseok Park, Donghyeon Cho, Wonhyuk Ahn, and Heung-Kyu Lee. Double jpeg detection in mixed jpeg quality factors using deep convolutional neural network. In _Proceedings of the European conference on computer vision (ECCV)_ , pp. 636–652, 2018. 

- Baolin Peng, Chunyuan Li, Pengcheng He, Michel Galley, and Jianfeng Gao. Instruction tuning with gpt-4. _arXiv preprint arXiv:2304.03277_ , 2023. 

- Alec Radford, Jeffrey Wu, Rewon Child, David Luan, Dario Amodei, Ilya Sutskever, et al. Language models are unsupervised multitask learners. _OpenAI blog_ , 1(8):9, 2019. 

- Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, et al. Learning transferable visual models from natural language supervision. In _International conference on machine learning_ , pp. 8748–8763. PMLR, 2021. 

- Yuan Rao and Jiangqun Ni. Self-supervised domain adaptation for forgery localization of jpeg compressed images. In _Proceedings of the IEEE/CVF international conference on computer vision_ , pp. 15034–15043, 2021. 

- Nikhila Ravi, Valentin Gabeur, Yuan-Ting Hu, Ronghang Hu, Chaitanya Ryali, Tengyu Ma, Haitham Khedr, Roman Radle, Chloe Rolland, Laura Gustafson, et al.¨ Sam 2: Segment anything in images and videos. _arXiv preprint arXiv:2408.00714_ , 2024. 

- Robin Rombach, Andreas Blattmann, Dominik Lorenz, Patrick Esser, and Bjorn¨ Ommer. Highresolution image synthesis with latent diffusion models. In _Proceedings of the IEEE/CVF conference on computer vision and pattern recognition_ , pp. 10684–10695, 2022. 

- Olga Russakovsky, Jia Deng, Hao Su, Jonathan Krause, Sanjeev Satheesh, Sean Ma, Zhiheng Huang, Andrej Karpathy, Aditya Khosla, Michael Bernstein, et al. Imagenet large scale visual recognition challenge. _International journal of computer vision_ , 115:211–252, 2015. 

- Ronald Salloum, Yuzhuo Ren, and C-C Jay Kuo. Image splicing localization using a multi-task fully convolutional network (mfcn). _Journal of Visual Communication and Image Representation_ , 51: 201–209, 2018. 

13 

Published as a conference paper at ICLR 2025 

Tal Shaharabany, Aviad Dahan, Raja Giryes, and Lior Wolf. Autosam: Adapting sam to medical images by overloading the prompt encoder. _arXiv preprint arXiv:2306.06370_ , 2023. 

- Ali Sharif Razavian, Hossein Azizpour, Josephine Sullivan, and Stefan Carlsson. Cnn features off-the-shelf: an astounding baseline for recognition. In _Proceedings of the IEEE conference on computer vision and pattern recognition workshops_ , pp. 806–813, 2014. 

- Karen Simonyan and Andrew Zisserman. Very deep convolutional networks for large-scale image recognition. _arXiv preprint arXiv:1409.1556_ , 2014. 

- Xuezhi Wang, Jason Wei, Dale Schuurmans, Quoc Le, Ed Chi, Sharan Narang, Aakanksha Chowdhery, and Denny Zhou. Self-consistency improves chain of thought reasoning in language models. _arXiv preprint arXiv:2203.11171_ , 2022a. 

- Zifeng Wang, Zizhao Zhang, Chen-Yu Lee, Han Zhang, Ruoxi Sun, Xiaoqi Ren, Guolong Su, Vincent Perot, Jennifer Dy, and Tomas Pfister. Learning to prompt for continual learning. In _Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition_ , pp. 139–149, 2022b. 

- Jason Wei, Xuezhi Wang, Dale Schuurmans, Maarten Bosma, Fei Xia, Ed Chi, Quoc V Le, Denny Zhou, et al. Chain-of-thought prompting elicits reasoning in large language models. _Advances in neural information processing systems_ , 35:24824–24837, 2022. 

- Bihan Wen, Ye Zhu, Ramanathan Subramanian, Tian-Tsong Ng, Xuanjing Shen, and Stefan Winkler. Coverage—a novel database for copy-move forgery detection. In _2016 IEEE international conference on image processing (ICIP)_ , pp. 161–165. IEEE, 2016. 

- Haiwei Wu, Jiantao Zhou, Jinyu Tian, Jun Liu, and Yu Qiao. Robust image forgery detection against transmission over online social networks. _IEEE Transactions on Information Forensics and Security_ , 17:443–456, 2022. 

- Junde Wu, Wei Ji, Yuanpei Liu, Huazhu Fu, Min Xu, Yanwu Xu, and Yueming Jin. Medical sam adapter: Adapting segment anything model for medical image segmentation. _arXiv preprint arXiv:2304.12620_ , 2023. 

- Yue Wu, Wael AbdAlmageed, and Premkumar Natarajan. Mantra-net: Manipulation tracing network for detection and localization of image forgeries with anomalous features. In _Proceedings of the IEEE/CVF conference on computer vision and pattern recognition_ , pp. 9543–9552, 2019. 

- Enze Xie, Wenhai Wang, Zhiding Yu, Anima Anandkumar, Jose M Alvarez, and Ping Luo. Segformer: Simple and efficient design for semantic segmentation with transformers. _Advances in neural information processing systems_ , 34:12077–12090, 2021. 

- Chao Yang, Huizhou Li, Fangting Lin, Bin Jiang, and Hao Zhao. Constrained r-cnn: A general image manipulation detection model. In _2020 IEEE International conference on multimedia and expo (ICME)_ , pp. 1–6. IEEE, 2020. 

- Chao Yang, Zhiyu Wang, Huawei Shen, Huizhou Li, and Bin Jiang. Multi-modality image manipulation detection. In _2021 IEEE International conference on multimedia and expo (ICME)_ , pp. 1–6. IEEE, 2021. 

- Yuanhao Zhai, Tianyu Luan, David Doermann, and Junsong Yuan. Towards generic image manipulation detection with weakly-supervised self-consistency learning. In _Proceedings of the IEEE/CVF International Conference on Computer Vision_ , pp. 22390–22400, 2023. 

- Kaidong Zhang and Dong Liu. Customized segment anything model for medical image segmentation. _arXiv preprint arXiv:2304.13785_ , 2023. 

- Zhuosheng Zhang, Aston Zhang, Mu Li, and Alex Smola. Automatic chain of thought prompting in large language models. _arXiv preprint arXiv:2210.03493_ , 2022. 

- Hengshuang Zhao, Jianping Shi, Xiaojuan Qi, Xiaogang Wang, and Jiaya Jia. Pyramid scene parsing network. In _Proceedings of the IEEE conference on computer vision and pattern recognition_ , pp. 2881–2890, 2017. 

14 

Published as a conference paper at ICLR 2025 

- Sixiao Zheng, Jiachen Lu, Hengshuang Zhao, Xiatian Zhu, Zekun Luo, Yabiao Wang, Yanwei Fu, Jianfeng Feng, Tao Xiang, Philip HS Torr, et al. Rethinking semantic segmentation from a sequence-to-sequence perspective with transformers. In _Proceedings of the IEEE/CVF conference on computer vision and pattern recognition_ , pp. 6881–6890, 2021. 

- Kaiyang Zhou, Jingkang Yang, Chen Change Loy, and Ziwei Liu. Learning to prompt for visionlanguage models. _International Journal of Computer Vision_ , 130(9):2337–2348, 2022. 

- Peng Zhou, B Chen, Xintong Han, Mahyar Najibi, and Larry S Davis. Generate, segment and replace: Towards generic manipulation segmentation. _arXiv preprint arXiv:1811.09729_ , 2018a. 

- Peng Zhou, Xintong Han, Vlad I Morariu, and Larry S Davis. Learning rich features for image manipulation detection. In _Proceedings of the IEEE conference on computer vision and pattern recognition_ , pp. 1053–1061, 2018b. 

- Peng Zhou, Bor-Chun Chen, Xintong Han, Mahyar Najibi, Abhinav Shrivastava, Ser-Nam Lim, and Larry Davis. Generate, segment, and refine: Towards generic manipulation segmentation. In _Proceedings of the AAAI conference on artificial intelligence_ , volume 34, pp. 13058–13065, 2020. 

- Xueyan Zou, Jianwei Yang, Hao Zhang, Feng Li, Linjie Li, Jianfeng Wang, Lijuan Wang, Jianfeng Gao, and Yong Jae Lee. Segment everything everywhere all at once. _Advances in Neural Information Processing Systems_ , 36, 2024. 

15 

Published as a conference paper at ICLR 2025 

## 6 APPENDIX 

## 6.1 LIMITATIONS AND BROADER IMPACTS 

**Limitations.** We have some limitations in our method. First, it cannot detect completely generated images. Second, training IMDPrompter requires complete pixel-level supervision. In future research, we plan to explore more efficient label utilization through weakly supervised and semi-supervised learning approaches and extend these methods to newer foundational models such as SEEMZou et al. (2024) and SAM2 Ravi et al. (2024). 

**Broader Impacts.** In recent years, the editing and manipulation of digital media have become increasingly convenient and widespread. Advances in image editing software, along with deep generative models such as Generative Adversarial Networks and diffusion models, make image manipulations, often imperceptible to the human eye, easier. These technologies are even widely used by potential malicious users. The widespread use of smartphones and social networks has also accelerated the dissemination of these manipulated media. Therefore, when these edited images are used to support false information activities or confuse news content to mislead the public, they can cause social issues and lead to a crisis of trust. Hence, IMDPrompter, as an effective iimage manipulation detection method, can significantly mitigate the problems caused by intentional image manipulation. 

## 6.2 DATASET DESCRIPTION 

In order to directly compare with state-of-the-art technologies, we trained on CASIAv2 Dong et al. (2013) and conducted extensive testing on COVER Wen et al. (2016), Columbia Hsu & Chang (2006), NIST16 Hsu & Chang (2006), CASIAv1Dong et al. (2013), and the recent IMD Novozamsky et al. (2020). 

Table 7: Details of the training set and five test sets used in our experiments. The symbol “–” indicates unavailable information. Copy-move, splicing, and inpainting operations are denoted as cpmv, spli, and inpa, respectively. Our model was trained on the CASIAv2 dataset and evaluated across all test sets. 

|**Dataset**|**Negative**|**Positive**|**cpmv**|**spli**|**inpa**|
|---|---|---|---|---|---|
|**Training**||||||
|CASIAv2 Dong et al. (2013)|7491|5063|3235|1828|0|
|**Testing**||||||
|COVER Wen et al. (2016)|100|100|100|0|0|
|Columbia Hsu & Chang (2006)|183|180|0|180|0|
|NIST16 Dong et al. (2013)|0|564|68|288|208|
|CASIAv1+ Dong et al. (2013)|800|920|459|461|0|
|IMD Novozamsky et al. (2020)|414|2010|–|–|–|



## 6.3 BASELINE METHODS 

As shown in Table 8, we have organized the technical routes of the baseline methods. To ensure fair and reproducible comparisons, we selected state-of-the-art models that meet any of the following criteria: 1) pretrained models released by the paper authors, 2) publicly available source code, and 3) adherence to a common evaluation protocol, where CASIAv2 is used for training and other public datasets for testing. Based on these criteria, we compiled a list of nine published baselines, as follows: 

- H-LSTM Bappy et al. (2019): Pretrained on a custom dataset of 65k processed images and fine-tuned on NIST16 and IEEE Forensics Challenge data. 

- ManTra-Net Wu et al. (2019): Trained on millions of processed images in a private collection. 

16 

Published as a conference paper at ICLR 2025 

Table 8: Comparison of Technical Routes of Baseline Methods 

|Method|View|Semantic Segmentation<br>Backbone|
|---|---|---|
||RGB<br>Noise<br>Fusion||
|MFCNSalloum et al. (2018)<br>RGB-NZhou et al. (2018b)<br>H-LSTMBappy et al. (2019)<br>ManTra-NetWu et al. (2019)<br>HP-FCNLi & Huang (2019)<br>GSR-NetZhou et al. (2018a)<br>CR-CNNYang et al. (2020)<br>SPANHu et al. (2020)<br>MM-NetYang et al. (2021)<br>JPEG-ComNetRao & Ni (2021)<br>CAT-NetKwon et al. (2021)<br>MVSS-NetChen et al. (2021)<br>TruFor Guillaro et al.(2023)|+<br>-<br>-<br>+<br>SRM<br>Late Fusion(Bilinear Pooling)<br>+<br>-<br>-<br>+<br>SRM+Bayar<br>Early Fusion(Feature Concatenation)<br>-<br>High-pass flters<br>-<br>+<br>-<br>-<br>-<br>Bayar<br>-<br>+<br>SRM+Bayar<br>Early Fusion(Feature Concatenation)<br>+<br>Bayar<br>Middle Fusion(Attention Guidance)<br>+<br>SRM<br>Early Fusion(Feature Concatenation)<br>+<br>DCT<br>Middle Fusion(Feature Concatenation)<br>+<br>Bayar<br>Late Fusion(Dual Attention)<br>+<br>Noiseprint<br>Late Fusion(Feature Concatenation)|FCN<br>Faster R-CNN<br>Patch-LSTM<br>Wider VGG<br>FCN<br>DeepLabv2<br>Mask R-CNN<br>Wider VGG<br>Mask R-CNN<br>Siamese FCN<br>HRNet<br>FCN<br>Segformer|
|IMDPrompter|+<br>SRM+Bayar+Noiseprint<br>Late Fusion(Optimal Prompt Selection<br>+Cross-view Prompt Consistency)|FCN+SAM|



- HP-FCN Li & Huang (2019): Trained on a private set of repaired images. 

- CR-CNN Yang et al. (2020): Trained on CASIAv2. 

- SPAN Hu et al. (2020): Trained using the same data as ManTra-Net and fine-tuned on CASIAv2. 

- CAT-Net Kwon et al. (2021): Trained on a joint dataset including CASIAv2, IMD, Fantastic Reality Kniaz et al. (2019), and self-spliced COCO. 

For models with publicly available code, we trained them using the code provided by the authors, such as GSR-Net Zhou et al. (2018a). When citing their results, we adopt the original data where appropriate and only use our retrained models when necessary. For models that follow the same evaluation protocol, such as MFCN Salloum et al. (2018) and RGB-N Zhou et al. (2018b), we directly cite results from the same team Yang et al. (2020). For fair comparison, we retrained FCN Long et al. (2015), MVSS-NetDong et al. (2022), and Trufor Guillaro et al. (2023) from scratch on CASIAv2. As previous studies rarely report their image-level performance, these models typically lack an image classification head in their implementations. To obtain a baseline for image-level predictions without modifying their models or code, we adopted the GMP strategy used in MVSS-Net. 

## 6.4 SUPPLEMENTARY EXPERIMENTS 

**Experimental Setups.** To improve reproducibility, we present the used parameters details in Table 9. 

**Overall Performance in Detection and Localization.** Table 10 and Figure 6 shows the overall performance of pixel-level and image-level manipulation detection. We use the harmonic mean of image-level detection F1 and pixel-level localization F1, denoted as C-F1, as our overall performance metric. As shown in Table 3, IMDPrompter achieves the best performance in all settings, with a performance improvement of 45.3% over the next best method. Notably, in the experiments on the COVER dataset, we achieved a performance gain of 110.7%. 

**Ablation study of the segmenters for the four prompt views:** We selected several segmenters, including FCN, PSPNet, SETR, and Segformer, for ablation experiments. As shown in the table below, the lightweight FCN based on MobileNet is sufficient to effectively guide SAM’s automated prompt learning. The introduction of more powerful segmenters like PSPNet, SETR, and Segformer did not result in a significant improvement in detection performance but instead introduced greater computational overhead. Therefore, we chose the lightweight FCN as our segmenter. 

**Computational Complexity Analysis:** As shown in Table 15, I-AUC represents Image-level AUC, I-F1 represents Image-level F1, P-F1 represents Pixel-level F1, and C-F1 represents the harmonic mean of I-F1 and P-F1. It is worth noting that the IMDPrompter mentioned in the main text is based on SAM-H, which has a high computational complexity. Therefore, we implemented a lightweight version, IMDPrompter*, based on Mobile SAM. As can be seen from Table 15, IMDPrompter achieved the best performance on almost all metrics, and IMDPrompter* achieved performance second only to IMDPrompter in nearly all metrics. 

17 

Published as a conference paper at ICLR 2025 

Table 9: Parameter Settings Details 

|**Parameter**|**Value**|
|---|---|
|**Warm-up Learning Rate Scheduler**||
|Type|LinearLR|
|Start Factor|1.00E-04|
|By Epoch|TRUE|
|Begin|0|
|End|1|
|Convert to Iter Based|TRUE|
|**Main Learning Rate Scheduler**||
|Type|CosineAnnealingLR|
|T<br>max|max<br>epochs|
|By Epoch|TRUE|
|Begin|1|
|End|max<br>epochs|
|**Data Preprocessor**||
|Mean|[123.675, 116.28, 103.53]|
|Standard Deviation|[58.395, 57.12, 57.375]|
|BGR to RGB|TRUE|
|Padding Value|0|
|Segmentation Padding Value|255|
|Size|-10,241,024|
|**Trainer**||
|Number of Things Classes|2|
|Train Batch Size per GPU|1|
|Train Number of Workers|4|
|Test Batch Size per GPU|1|
|Test Number of Workers|4|
|Epochs|100|



Table 10: Overall Detection and Localization Performance (Com-F1 Score). The best results for each test set are highlighted in bold and second-best values are underlined. 

|**Method**<br>**CASIA**<br>**COVER**<br>**Columbia**<br>**IMD**<br>**MEAN**|**Method**<br>**CASIA**<br>**COVER**<br>**Columbia**<br>**IMD**<br>**MEAN**|
|---|---|
|H-LSTM Bappy et al. (2019)<br>ManTra-Net Wu et al. (2019)<br>CR-CNN Yang et al. (2020)<br>GSR-Net Zhou et al. (2018a)<br>SPAN Hu et al. (2020)<br>CAT-Net Kwon et al. (2021)<br>MVSS-Net Chen et al. (2021)|0.0<br>0.0<br>3.8<br>0.0<br>0.94<br>0.0<br>0.0<br>0.0<br>0.0<br>0.00<br>30.3<br>18.1<br>41.3<br>23.7<br>28.35<br>0.0<br>0.0<br>4.2<br>0.0<br>1.06<br>0.0<br>0.0<br>0.0<br>0.0<br>0.00<br>20.0<br>19.7<br>69.1<br>9.5<br>29.59<br>56.6<br>31.7<br>71.1<br>30.0<br>47.36|
|FCN Long et al. (2015)|53.6<br>18.9<br>30.5<br>23.3<br>31.57|
|IMDPromter|**76.8**<br>**66.8**<br>**90.3**<br>**41.3**<br>**68.82**|
|p|+23.2<br>+47.9<br>+59.8<br>+18.0<br>+37.25|



**Comparison with SAM-based Methods:** As shown in Table 15, We supplement the performance of four SAM-based image segmentation methods: MedSAM, MedSAM-Adapter, AutoSAM, and SAMed. As we can see, these four SAM-based methods exhibit limited performance in the iimage manipulation detection task due to their reliance solely on semantically relevant information. In particular, there is a significant performance gap between them and IMDPrompter, especially on the out-of-domain test sets CVOER, Columbia, and IMD. 

**Proportion of Four Views Selected as Optimal Prompts.** As shown in Table 17, we summarize the proportion of each prompt view being selected as the optimal prompt during training. We observe that the RGB view is selected as the optimal prompt in only 13.9% of cases, indicating that 

18 

Published as a conference paper at ICLR 2025 


![](raw/papers/2_LLM_NLP/LLM_paper_list/autoprompt/iclr_2025_IMDPrompter_Adapting_SAM_to_Image_Manipulation_Detection_by_Cross-View_Automated_Prompt_Learning_images/iclr_2025_IMDPrompter_Adapting_SAM_to_Image_Manipulation_Detection_by_Cross-View_Automated_Prompt_Learning.pdf-0019-01.png)


**----- Start of picture text -----**<br>
COVER COVER COVER<br>Columbia Columbia Columbia<br>CASIA CASIA CASIA<br>IMD IMD IMD<br>MEAN MEAN MEAN<br>(a) I-F1 (b) I-AUC (c) C-F1<br>Columbia COVER Columbia COVER<br>IMD CASIA IMD CASIA CR-CNN<br>CAT-Net<br>MVSS-Net<br>FCN<br>NIST MEAN NIST MEAN IMDPrompter<br>(d) Best threshold P-F1 (e) Fixed threshold P-F1 (f)<br>**----- End of picture text -----**<br>


Figure 6: Visualization of Metric Data Comparison between IMDPrompter and State-of-the-Art (SOTA) Methods 

Table 11: Impact of JPEG Compression and Gaussian Blur on Pixel-level F1 

|Method|JPEG Compression<br>Gaussian Blurs<br>100<br>90<br>80<br>70<br>60<br>50<br>0<br>5<br>11<br>17<br>23<br>29|Gaussian Blurs|
|---|---|---|
|||0<br>5<br>11<br>17<br>23<br>29|
|H-LSTM Bappy et al. (2019)<br>ManTra-Net Wu et al. (2019)<br>CR-CNN Yang et al. (2020)<br>GSR-Net Zhou et al. (2018a)<br>SPAN Hu et al. (2020)<br>CAT-Net Kwon et al. (2021)<br>MVSS-Net Chen et al. (2021)<br>IMDPrompter|15.4<br>15.0<br>14.8<br>14.7<br>14.0<br>13.9<br>15.5<br>11.9<br>8.2<br>8.0<br>8.7<br>8.6<br>40.4<br>33.6<br>22.4<br>17.1<br>11.7<br>9.2<br>37.4<br>35.9<br>32.5<br>32.4<br>30.8<br>27.0<br>18.2<br>17.1<br>17.1<br>16.6<br>15.7<br>14.9<br>13.5<br>2.2<br>2.0<br>1.3<br>0.9<br>0.9<br>45.1<br>43.2<br>42.5<br>41.2<br>40.2<br>39.5<br>76.3<br>73.6<br>72.0<br>71.6<br>70.7<br>70.5|15.4<br>13.4<br>12.1<br>10.8<br>10.6<br>10.3<br>15.5<br>12.0<br>11.3<br>12.0<br>11.6<br>11.2<br>40.4<br>23.3<br>22.1<br>21.1<br>20.7<br>19.9<br>37.4<br>30.5<br>26.5<br>23.5<br>20.8<br>19.0<br>18.2<br>10.7<br>10.6<br>10.0<br>9.6<br>9.6<br>13.5<br>9.4<br>9.3<br>6.9<br>6.1<br>6.0<br>45.1<br>37.4<br>36.0<br>33.0<br>21.1<br>17.3<br>76.3<br>71.8<br>71.1<br>68.8<br>56.6<br>51.6|



semantically relevant features alone are insufficient to guide SAM’s prompt learning. On the other hand, the Noiseprint view is chosen as the optimal prompt in 46.3% of cases, demonstrating that the semantically irrelevant information provided by Noiseprint is the most crucial. 

**Hyperparametric Analysis** As shown in Figure 7, we conducted hyperparameter analysis on _λ_ 1, _λ_ 2, and _λ_ 3, ultimately selecting the optimal parameter configuration: _λ_ 1 = 1 _._ 0 _, λ_ 2 = 0 _._ 1 _, λ_ 3 = 1 _._ 0. 

**Performance of Generative Image Manipulation Detection:** Following the work of Trufor, we supplement the performance of IMDPrompter on CocoGlide (image tampering based on diffusion models). As shown in the table below, our IMDPrompter also achieves objective results for AIGCedited images. 

**Impact of Quality Degradation.** As shown in Table 11,12,13, following Dong et al. (2022), we evaluated the robustness of the models under two common image processing operations encountered during the dissemination of images on the internet, namely JPEG compression and Gaussian blur. Comparing these two operations, Gaussian blur has a more significant impact on detection performance, especially when using larger 17x17 scale convolution kernels. Compared to previous methods, our IMDPrompter exhibits better robustness. 

**More Visualization.** As shown in Figure 10, we have included additional visualizations. 

19 

Published as a conference paper at ICLR 2025 

Table 12: Impact of JPEG Compression and Gaussian Blur on Image-level F1 

|Method|JPEG Compression<br>100<br>90<br>80<br>70<br>60<br>50|Gaussian Blur<br>0<br>5<br>11<br>17<br>23<br>29|
|---|---|---|
|H-LSTM Bappy et al. (2019)<br>ManTra-Net Wu et al. (2019)<br>CR-CNN Yang et al. (2020)<br>GSR-Net Zhou et al. (2018a)<br>SPAN Hu et al. (2020)<br>CAT-Net Kwon et al. (2021)<br>MVSS-Net Chen et al. (2021)<br>IMDPrompter|0.0<br>0.0<br>0.0<br>0.0<br>0.0<br>0.0<br>0.0<br>0.0<br>0.0<br>0.0<br>0.0<br>0.0<br>24.3<br>25.8<br>24.0<br>22.4<br>20.0<br>18.4<br>0.0<br>0.0<br>0.0<br>0.0<br>0.0<br>0.0<br>0.0<br>0.0<br>0.0<br>0.0<br>0.0<br>0.0<br>38.0<br>17.9<br>17.2<br>15.0<br>12.4<br>11.0<br>75.9<br>76.2<br>75.7<br>75.3<br>74.8<br>73.0<br>77.3<br>77.8<br>77.2<br>76.8<br>76.1<br>74.6|0.0<br>0.0<br>0.0<br>0.0<br>0.0<br>0.0<br>0.0<br>0.0<br>0.0<br>0.0<br>0.0<br>0.0<br>24.3<br>14.3<br>13.9<br>13.7<br>12.4<br>11.4<br>0.0<br>0.0<br>0.0<br>0.0<br>0.0<br>0.0<br>0.0<br>0.0<br>0.0<br>0.0<br>0.0<br>0.0<br>38.0<br>11.9<br>11.0<br>10.5<br>10.3<br>10.7<br>75.9<br>73.8<br>72.8<br>70.7<br>65.8<br>63.4<br>77.3<br>75.9<br>74.6<br>73.1<br>69.3<br>66.6|



Table 13: Impact of JPEG Compression and Gaussian Blur on Com-F1 

|Method|JPEG Compression<br>100<br>90<br>80<br>70<br>60<br>50|Gaussian Blur<br>0<br>5<br>11<br>17<br>23<br>29|
|---|---|---|
|H-LSTM Bappy et al. (2019)<br>ManTra-Net Wu et al. (2019)<br>CR-CNN Yang et al. (2020)<br>GSR-Net Zhou et al. (2018a)<br>SPAN Hu et al. (2020)<br>CAT-Net Kwon et al. (2021)<br>MVSS-Net Chen et al. (2021)<br>IMDPrompter|0.0<br>0.0<br>0.0<br>0.0<br>0.0<br>0.0<br>0.0<br>0.0<br>0.0<br>0.0<br>0.0<br>0.0<br>30.3<br>29.2<br>23.2<br>19.4<br>14.8<br>12.3<br>0.0<br>0.0<br>0.0<br>0.0<br>0.0<br>0.0<br>0.0<br>0.0<br>0.0<br>0.0<br>0.0<br>0.0<br>19.9<br>3.9<br>3.6<br>2.3<br>1.7<br>1.7<br>56.6<br>55.1<br>54.4<br>53.2<br>52.3<br>51.2<br>76.8<br>75.6<br>74.5<br>74.1<br>73.3<br>72.5|0.0<br>0.0<br>0.0<br>0.0<br>0.0<br>0.0<br>0.0<br>0.0<br>0.0<br>0.0<br>0.0<br>0.0<br>30.3<br>17.7<br>17.0<br>16.6<br>15.5<br>14.5<br>0.0<br>0.0<br>0.0<br>0.0<br>0.0<br>0.0<br>0.0<br>0.0<br>0.0<br>0.0<br>0.0<br>0.0<br>19.9<br>10.5<br>10.1<br>8.3<br>7.7<br>7.7<br>56.6<br>49.7<br>48.2<br>45.0<br>32.0<br>27.2<br>76.8<br>73.8<br>72.8<br>70.9<br>62.3<br>58.1|



Table 14: Ablation Study of the Segmenter 

|**Params/M**<br>**GFLOPs**|**CASIA**<br>**I-AUC**<br>**I-F1**<br>**P-F1**<br>**C-F1**|**COVER**|**Columbia**<br><br>**I-AUC**<br>**I-F1**<br>**P-F1**<br>**C-F1**|**IMD**|
|---|---|---|---|---|
|||**I-AUC**<br>**I-F1**<br>**P-F1**<br>**C-F1**||**I-AUC**<br>**I-F1**<br>**P-F1**<br>**C-F1**|
|**PSPNetZhao et al. (2017)**<br>365.3<br>1610.3<br>**SETRZheng et al. (2021)**<br>1591.6<br>2131.6<br>**SegformerXie et al. (2021)**<br>536.3<br>1769.3<br>**FCNLong et al. (2015)**<br>347.6<br>1533.2|0.979<br>77.4<br>76.5<br>76.9<br>0.985<br>78.1<br>77.1<br>77.6<br>0.988<br>78.9<br>77.8<br>78.3<br>0.978<br>77.4<br>76.4<br>76.9|0.794<br>70.3<br>63.7<br>66.8<br>0.801<br>70.9<br>64.3<br>67.4<br>0.811<br>71.2<br>65.1<br>68.0<br>0.795<br>70.2<br>63.7<br>66.8|0.984<br>93.7<br>87.8<br>90.7<br>0.988<br>94.1<br>88.2<br>91.1<br>0.991<br>94.9<br>88.7<br>91.7<br>0.984<br>93.6<br>87.6<br>90.5|0.673<br>64.1<br>30.6<br>41.4<br>0.679<br>64.5<br>31.1<br>42.0<br>0.684<br>65.1<br>32.1<br>43.0<br>0.670<br>63.8<br>30.6<br>41.4|



Table 15: Computational Complexity Overhead 

|**Params/M**<br>**GFLOPs**|**CASIA**|**COVER**<br>**I-AUC**<br>**I-F1**<br>**P-F1**<br>**C-F**|**Columbia**<br>**1**<br>**I-AUC**<br>**I-F1**<br>**P-F1**<br>**C-F1**|**IMD**|
|---|---|---|---|---|
||**I-AUC**<br>**I-F1**<br>**P-F1**<br>**C-F1**|||**I-AUC**<br>**I-F1**<br>**P-F1**<br>**C-F1**|
|**ManTra-NetWu et al. (2019)**<br>4.0<br>1009.7<br>**SPANHu et al. (2020)**<br>15.4<br>30.9<br>**MVSS-NetChen et al. (2021)**<br>146.9<br>160.0<br>**TruforGuillaro et al. (2023)**<br>67.8<br>90.1<br>**IMDPrompter**<br>347.6<br>1533.2<br>**IMDPrompter***<br>85.8<br>151.3|0.500<br>0.0<br>15.5<br>0.0<br>0.500<br>0.0<br>18.4<br>0.0<br>0.937<br>75.8<br>45.2<br>56.6<br>0.916<br>-<br>44.1<br>-<br>**0.978**<br>**77.3**<br>**76.3**<br>**76.8**<br>0.951<br>76.1<br>70.3<br>73.1|0.500<br>0.0<br>28.6<br>0.0<br>0.500<br>0.0<br>17.2<br>0.0<br>0.731<br>24.4<br>45.3<br>31.<br>0.770<br>-<br>19.9<br>-<br>**0.796**<br>**70.3**<br>**63.6**<br>**66.**<br>0.779<br>64.5<br>57.7<br>60.|0.701<br>0.0<br>36.4<br>0.0<br><br>0.500<br>0.0<br>48.7<br>0.0<br>7<br>0.980<br>80.2<br>63.8<br>71.1<br>**0.996**<br>-<br>22.3<br>-<br>**8**<br>0.983<br>**93.6**<br>**87.3**<br>**90.3**<br>9<br>0.980<br>88.3<br>81.6<br>84.8|0.500<br>0.0<br>18.7<br>0.0<br>0.500<br>0.0<br>17.0<br>0.0<br>0.656<br>35.5<br>26.0<br>30.0<br>-<br>-<br>21.0<br>-<br>**0.671**<br>**63.7**<br>**30.6**<br>**41.3**<br>0.667<br>56.9<br>26.9<br>36.5|



Table 16: Comparison with SAM-based methods. 

||**CASIA**<br>**I-AUC**<br>**I-F1**<br>**P-F1**<br>**C-F1**|**COVER**<br>**I-AUC**<br>**I-F1**<br>**P-F1**<br>**C-F1**|**Columbia**<br>**I-AUC**<br>**I-F1**<br>**P-F1**<br>**C-F1**|**IMD**|
|---|---|---|---|---|
|||||**I-AUC**<br>**I-F1**<br>**P-F1**<br>**C-F1**|
|**MedSAMMa et al. (2024)**<br>**MedSAM-AdapterWu et al. (2023)**<br>**AutoSAMShaharabany et al. (2023)**<br>**SAMedZhang & Liu (2023)**<br>**IMDPrompter**<br>**IMDPrompter***|0.861<br>70.3<br>56.4<br>62.6<br>0.877<br>71.0<br>57.3<br>63.4<br>0.843<br>69.1<br>55.1<br>61.3<br>0.843<br>69.9<br>52.3<br>59.8<br>**0.978**<br>**77.3**<br>**76.3**<br>**76.8**<br>0.951<br>76.1<br>70.3<br>73.1|0.563<br>26.3<br>22.1<br>24.0<br>0.543<br>21.6<br>20.3<br>20.9<br>0.513<br>18.9<br>18.6<br>18.7<br>0.571<br>27.6<br>24.6<br>26.0<br>**0.796**<br>**70.3**<br>**63.6**<br>**66.8**<br>0.779<br>64.5<br>57.7<br>60.9|0.791<br>50.1<br>24.6<br>33.0<br>0.742<br>46.6<br>22.1<br>30.0<br>0.772<br>49.1<br>28.4<br>36.0<br>0.801<br>53.1<br>31.7<br>39.7<br>0.983<br>**93.6**<br>**87.3**<br>**90.3**<br>0.980<br>88.3<br>81.6<br>84.8|0.513<br>26.1<br>22.4<br>24.1<br>0.494<br>25.1<br>21.6<br>23.2<br>0.523<br>27.9<br>22.9<br>25.2<br>0.502<br>27.6<br>23.1<br>25.2<br>**0.671**<br>**63.7**<br>**30.6**<br>**41.3**<br>0.667<br>56.9<br>26.9<br>36.5|



Table 17: The proportion of each view selected as the optimal view 

|**Prompt View**|**RGB**|**SRM**|**Bayer**|**Noiseprint**|
|---|---|---|---|---|
|**Percent**|13.9%|18.4%|21.4%|46.3%|



20 

Published as a conference paper at ICLR 2025 

Table 18: Performance of Generative Image Manipulation Detection 

|**Method**|**CocoGlide**<br>**P-F1(best)**<br>**P-F1(fxed)**<br>**I-AUC**<br>**I-Acc**|
|---|---|
|**ManTraNetWu et al. (2019)**<br>**SPANHu et al. (2020)**<br>**MVSS-NetChen et al. (2021)**<br>**TruforGuillaro et al. (2023)**<br>**IMDPrompter**|0.673<br>0.516<br>0.778<br>0.500<br>0.35<br>0.298<br>0.475<br>0.491<br>0.642<br>0.486<br>0.654<br>0.536<br>0.72<br>0.523<br>0.752<br>0.639<br>0.746<br>0.539<br>0.781<br>0.652|




![](raw/papers/2_LLM_NLP/LLM_paper_list/autoprompt/iclr_2025_IMDPrompter_Adapting_SAM_to_Image_Manipulation_Detection_by_Cross-View_Automated_Prompt_Learning_images/iclr_2025_IMDPrompter_Adapting_SAM_to_Image_Manipulation_Detection_by_Cross-View_Automated_Prompt_Learning.pdf-0021-03.png)


Figure 7: Hyperparametric analysis of _λ_ 1, _λ_ 2, and _λ_ 3 

21 

Published as a conference paper at ICLR 2025 


![](raw/papers/2_LLM_NLP/LLM_paper_list/autoprompt/iclr_2025_IMDPrompter_Adapting_SAM_to_Image_Manipulation_Detection_by_Cross-View_Automated_Prompt_Learning_images/iclr_2025_IMDPrompter_Adapting_SAM_to_Image_Manipulation_Detection_by_Cross-View_Automated_Prompt_Learning.pdf-0022-01.png)


**----- Start of picture text -----**<br>
Image GT ManTra-Net MVSS-Net Trufor IMDPrompter<br>**----- End of picture text -----**<br>


Figure 8: Some qualitative results, compared with the state-of-the-art 

22 

Published as a conference paper at ICLR 2025 


![](raw/papers/2_LLM_NLP/LLM_paper_list/autoprompt/iclr_2025_IMDPrompter_Adapting_SAM_to_Image_Manipulation_Detection_by_Cross-View_Automated_Prompt_Learning_images/iclr_2025_IMDPrompter_Adapting_SAM_to_Image_Manipulation_Detection_by_Cross-View_Automated_Prompt_Learning.pdf-0023-01.png)


**----- Start of picture text -----**<br>
Image GT ManTra-Net MVSS-Net Trufor IMDPrompter<br>**----- End of picture text -----**<br>


Figure 9: Some qualitative results, compared with the state-of-the-art 

23 

Published as a conference paper at ICLR 2025 


![](raw/papers/2_LLM_NLP/LLM_paper_list/autoprompt/iclr_2025_IMDPrompter_Adapting_SAM_to_Image_Manipulation_Detection_by_Cross-View_Automated_Prompt_Learning_images/iclr_2025_IMDPrompter_Adapting_SAM_to_Image_Manipulation_Detection_by_Cross-View_Automated_Prompt_Learning.pdf-0024-01.png)


**----- Start of picture text -----**<br>
Image GT ManTra-Net MVSS-Net Trufor IMDPrompter<br>**----- End of picture text -----**<br>


Figure 10: Some qualitative results, compared with the state-of-the-art 

24 

