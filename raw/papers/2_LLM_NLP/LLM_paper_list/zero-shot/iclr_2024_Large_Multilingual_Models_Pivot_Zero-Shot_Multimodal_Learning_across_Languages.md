Published as a conference paper at ICLR 2024 

# LARGE MULTILINGUAL MODELS PIVOT ZERO-SHOT MULTIMODAL LEARNING ACROSS LANGUAGES 

**Jinyi Hu**[1] **Yuan Yao**[1] _[∗]_ **Chongyi Wang**[4] **Shan Wang**[4] **Yinxu Pan**[5] **Qianyu Chen**[1] **Tianyu Yu**[1] **Hanghao Wu**[5] **Yue Zhao**[2] **Haoye Zhang**[1] **Xu Han**[1] **Yankai Lin**[3] **Jiao Xue**[4] **Dahai Li**[4] **Zhiyuan Liu**[1] _[∗]_ **Maosong Sun**[1] _[∗]_ 1 Tsinghua University 2 Beijing University of Posts and Telecommunications 3 Renmin University of China 4 Zhihu Inc. 5 ModelBest Inc. hu-jy21@mails.tsinghua.edu.cn 

## ABSTRACT 

Recently there has been a significant surge in multimodal learning in terms of both image-to-text and text-to-image generation. However, the success is typically limited to English, leaving other languages largely behind. Building a competitive counterpart in other languages is highly challenging due to the low-resource nature of non-English multimodal data (i.e., lack of large-scale, high-quality image-text data). In this work, we propose MPM, an effective training paradigm for training large multimodal models in non-English languages. MPM demonstrates that **M** ultilingual language models can **P** ivot zero-shot **M** ultimodal learning across languages. Specifically, based on a strong multilingual large language model, multimodal models pretrained on English-only image-text data can well generalize to other languages in a (quasi)-zero-shot manner, even surpassing models trained on image-text data in native languages. Taking Chinese as a practice of MPM, we build large multimodal models VISCPM in image-to-text and text-toimage generation, which achieve state-of-the-art (open-source) performance in Chinese. To facilitate future research, we open-source codes and model weights at https://github.com/OpenBMB/VisCPM. 

## 1 INTRODUCTION 

With the rapid advancement of powerful models such as GPT-4 (OpenAI, 2023) and Stable Diffusion (Rombach et al., 2022) in their multimodal capabilities, large multimodal models have emerged as the latest frontier in pursuing achieving Artificial General Intelligence (AGI). Generally, the multimodal generative capabilities across images and text can be divided into two categories: (i) In the field of image-to-text generation, prominent multimodal large language models like GPT-4 (OpenAI, 2023), LLaVA (Liu et al., 2023a) and InstructBLIP (Dai et al., 2023) exhibit remarkable multimodal conversational and reasoning abilities based on images; (ii) In the field of text-to-image generation, models such as Imagen (Saharia et al., 2022) and Stable Diffusion (Rombach et al., 2022) excel in generating highly realistic and relevant images based on text prompts. These models possess exceptional capabilities in processing images and text, profoundly reshaping the landscape of multimodal AI in both academia and industry. 

However, the success of large multimodal models has mainly been achieved within the English community, while the multimodal capabilities in other non-English languages significantly trail behind. Bridging this gap is challenging due to the extensive image-text pair data requirements for training multimodal models. For instance, the pretraining of BLIP-2 (Li et al., 2023a) involves more than 100M high-quality image-text pairs, while Stable Diffusion (Rombach et al., 2022) utilizes more than 2B pairs. As a result of the paucity of such multimodal data resources in non-English languages, the progress of multimodal research in these languages remains hindered. 

To address this challenge, we propose MPM, an effective training paradigm for large multimodal models in non-English languages. MPM utilizes the **M** ultilingual language model to **P** ivot **M** ultimodal 

> _∗_ Corresponding authors. 

1 

Published as a conference paper at ICLR 2024 


![](raw/papers/2_LLM_NLP/LLM_paper_list/zero-shot/iclr_2024_Large_Multilingual_Models_Pivot_Zero-Shot_Multimodal_Learning_across_Languages_images/iclr_2024_Large_Multilingual_Models_Pivot_Zero-Shot_Multimodal_Learning_across_Languages.pdf-0002-01.png)


**----- Start of picture text -----**<br>
High-resource<br>English as Pivot Image-Text Pretrain VisCPM-Chat<br>Instruction Tuning<br>Cat Visual Encoder Image-to-Text Model<br>CPM-Bee<br>Bilingual LLM +<br>猫 Backbone LLM Large-Scale<br>Non-English Generalization Image-Text Pairs<br>Language Image-Text Pretrain VisCPM-Paint<br>Visual Decoder Text-to-Image Model<br>(a) Motivation of MPM based on the generalization of<br>multimodal alignment across languages. (b) Architecture of VISCPM.<br>Multimodal<br>Alignment Alignment<br>Multilingual<br>**----- End of picture text -----**<br>


Figure 1: Overview of the motivation and architecture of MPM and VISCPM. 

learning across languages and considers English, which contains substantial multimodal data resources, as a pivot between the visual signals and non-English languages which commonly lacking in multimodal data. MPM draws inspiration from the _Bilingual Dual-coding Theory_ (Paivio, 2014; Clark & Paivio, 1991) that argue that _visual semantics are largely language-agnostic_ . Intuitively, as portrayed in Fig. 1a, multilingual learners can effectively align the visual semantics with newly acquired language based on established multimodal and multilingual alignment. Simulating the human learning process, MPM also divides the non-English multimodal learning into two consecutive stages: multilingual alignment and multimodal alignment. The former focuses on building a multilingual model, while the latter culminates in a multimodal model spanning multiple languages. 

Specifically, for multilingual alignment, MPM harnesses a pretrained multilingual large language model (LLM) as the backbone language model, which can provide aligned representations for different languages. Next, for the multimodal alignment, MPM trains the visual modules based on the multilingual model exclusively on English image-text pairs to align English and visual semantics. Similar to how humans learn, using the multilingual model as a pivot point, the resultant multimodal model naturally acquires zero-shot multimodal capability in other non-English languages. 

Taking Chinese as a practical instance for MPM, we develop Chinese large multimodal models named VISCPM based on English-Chinese bilingual large language model CPM-Bee (Zhang et al., 2021). Notably, pretraining exclusively on English image-text pairs, the zero-shot performance of VISCPM in Chinese still surpasses that of existing Chinese multimodal models trained on image-text pairs in native Chinese. The promising performance of MPM in Chinese sheds light on its potential application in broader languages. Following the same training process, we further extend MPM to develop a multilingual multimodal conversation model supporting six languages based on the LLaMA (Touvron et al., 2023), including English, German, French, Spanish, Italian, and Portuguese. 

In summary, the contributions of this paper are as follows: (i) We propose MPM, an effective training paradigm specifically designed for non-English languages lacking multimodal resources. Researchers worldwide can utilize MPM to rapidly adapt advanced multimodal learning methods from English to their respective languages. (ii) We develop a series of Chinese large multimodal models VISCPM as a practical application of MPM, which achieves state-of-the-art performance among open-source Chinese multimodal models. (iii) We open-source the model weights of VISCPM and provide experimental details, serving as a valuable reference for other researchers. (iv) We validate the generalization capability of VISCPM in diverse languages and develop the multilingual multimodal conversation model across six languages. 

## 2 RELATED WORK 

**Image-to-text Models.** Traditional image-to-text generation models mainly focus on the task of image caption and visual question answering (Yuan et al., 2021; Yu et al., 2022a; Wang et al., 2022b). Recently, the mainstream of image-to-text has turned to multimodal LLM, focusing on rendering LLM capable of multimodal interaction with users. These models connect the visual module and LLM with perceivers, such as BLIP-2 (Li et al., 2023a) and InstructBLIP (Dai et al., 2023) or linear projectors such as LLaVA (Liu et al., 2023a) and PandaGPT (Su et al., 2023). VPGTrans (Zhang et al., 2023) explores the transferability of visual modules across LLM. Meanwhile, many efforts have been dedicated to building multimodal instruction following datasets. LLaVA (Liu et al., 2023a) 

2 

Published as a conference paper at ICLR 2024 

and MiniGPT-4 (Zhu et al., 2023) build visual content-related dialog by transferring image captions into conversation data using GPT-4. InstructBLIP (Dai et al., 2023) and M[3] IT (Li et al., 2023b) incorporate downstream vision-language datasets to construct instruction data. 

**Text-to-image Models.** In the early stages of text-to-image model development, generative adversarial networks (Zhu et al., 2019; Li et al., 2019) and auto-regressive generation (Esser et al., 2021) are popularly chosen architectures for text-to-image synthesis models (Li et al., 2019). More recently, large-scale diffusion-based text-to-image models such as DALLE-2 (Ramesh et al., 2022), Imagen (Saharia et al., 2022), and Stable Diffusion (Rombach et al., 2022) have taken center stage, demonstrating exceptional generative capabilities. 

**Multilingual Multimodal Models.** The extension of multimodal models to include more languages has become a key research focus over the past few years. Researchers have made efforts to extend the powerful image-text model CLIP (Radford et al., 2021) to handle more languages using techniques of knowledge distillation (Carlsson et al., 2022) or contrastive learning (Bianchi et al., 2021; Chen et al., 2023). Other studies have aimed to create a universal framework for multilingual vision-language pertaining and simultaneously achieve multilingual and multimodel alignment (Ni et al., 2021; Zhou et al., 2021; Zeng et al., 2023). In the era of LLMs, PaLI (Chen et al., 2022) develops a 17B multilingual language-image model based on 10B image-text pairs spanning 100 languages. YingVLM (Li et al., 2023b) shows that instruction tuning in English can generalize to other languages. MultiFusion (Bellagente et al., 2023) discover that the multilingual language model can help crosslingual transfer in text-to-image generation. In comparison, this work provides a more systematical formulation for training multilingual multimodal models and demonstrates that the zero-shot transfer performance of these models can surpass that of models trained on native-language multimodal data. 

## 3 MPM TRAINING PARADIGM 

In this section, we first present the formulation of multilingual multimodal learning and provide an overview of the training procedure of MPM. Following this, we detail the specific training procedures of MPM for both image-to-text and text-to-image generation. 

## 3.1 PROBLEM FORMULATION AND OVERVIEW 

Multimodal learning can be formulated as modeling the relationship between images, denoted as _x_ , and text, denoted as _y_ , in a target language _lt_ . In this context, the image-to-text generation, which can be roughly summarized as generating description for input images, is to learn the conditional distribution _pθ_ ( _y[l][t] |x_ ) parameterized by _θ_ ; the text-to-image generation, which is to synthesize relevant images given input text prompts, is to learn _pϕ_ ( _x|y[l][t]_ ) parameterized by _ϕ_ . 

In the vanilla setting, these conditional distributions are typically trained using image-text pairs _Dt_ = _{_ ( _xi, yi[l][t]_[)] _[}] i[N]_ =1[in the target language] _[ l][t]_[(Radford et al., 2021; Yang et al., 2022).][However,] high-quality image-text pairs are extremely scarce in most languages. To mitigate the dependency on native image-text pairs, we introduce the pivot language _lp_ , which contains abundant multimodal pairs _Dp_ = _{_ ( _xi, yi[l][p]_[)] _[}] i[M]_ =1[, where] _[ M][≫][N]_[.][Imitating the human learning mechanism that can naturally] align visual concepts with various learned languages, MPM aims to transfer visual concepts learned in the pivot language to the target language. 

MPM divides the multimodal learning process in target language _lt_ into two consecutive stages: **multilingual alignment** and **multimodal alignment** . For the multilingual alignment, MPM aims to establish the cross-lingual alignment for _lt_ and _lp_ . This is achieved by directly leveraging a pretrained multilingual LLM, denoted as _fσ_ , which can provide close hidden representations for text pair _y[l][t]_ and _y[l][p]_ with similar semantics, i.e., _fσ_ ( _y[l][t]_ ) _≈ fσ_ ( _y[l][p]_ ). For the multimodal alignment, MPM utilize the sufficient multimodal resource _Dp_ in the pivot language and optimize the image-to-text objective _pθ_ ( _y[l][p] |x_ ) and text-to-image objective _pϕ_ ( _x|y[l][p]_ ). In the following sections, we introduce the training process of **multimodal alignment** stage. It’s worth noting that MPM is agnostic to the specific model architecture and training method, which enables us to flexibly utilize existing highly effective model architectures and training techniques in each task. 

3 

Published as a conference paper at ICLR 2024 

## 3.2 IMAGE-TO-TEXT GENERATION 

In image-to-text generation, we incorporate an image encoder module _hξ_ parameterized by _ξ_ to provide visual feature **z** = _hξ_ ( _x_ ). These visual features **z** are then concatenated with the text embedding as input into the multilingual LLM. Following recent work to train multimodal conversation models (Zhu et al., 2023; Liu et al., 2023a), MPM’s training process for image-to-text generation consists of two sub-stages: Multimodal Pretraining and Instruction Tuning. 

**Multimodal Pretraining.** In this sub-stage, we pretrain the visual module to align it with LLM on a large scale of image-text pairs using the language modeling objective: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/zero-shot/iclr_2024_Large_Multilingual_Models_Pivot_Zero-Shot_Multimodal_Learning_across_Languages_images/iclr_2024_Large_Multilingual_Models_Pivot_Zero-Shot_Multimodal_Learning_across_Languages.pdf-0004-04.png)


Here, we fix the parameters of LLM ( _θ_ = _{ξ}_ ) to prevent the powerful capabilities of LLM from being influenced by short texts in the image-text pairs. 

**Instruction Tuning.** To enhance models’ capabilities in following human instructions, we conduct instruction tuning on elaborately curated multimodal instruction tuning datasets built by blending the existing multimodal instruction tuning datasets in the pivot language and their translated version in the target language. We denote this multilingual instruction tuning datasets as _Di_ = _{xk, yq,k[l][, y] a,k[l][}] k[S]_ =1[,] where _yq[l]_[is the instructions and] _[ y] a[l]_[is the response in certain language] _[ l]_[.][Both the visual module and] multilingual LLM are fine-tuned, i.e., _θ_ = _{ξ, σ}_ , by maximizing the probability of the response: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/zero-shot/iclr_2024_Large_Multilingual_Models_Pivot_Zero-Shot_Multimodal_Learning_across_Languages_images/iclr_2024_Large_Multilingual_Models_Pivot_Zero-Shot_Multimodal_Learning_across_Languages.pdf-0004-07.png)


Interestingly, we find a _quasi-zero-shot_ transfer capability of multilingual multimodal models in this scenario. If excluding the translated variant in the target language and solely performing instruction tuning using the pivot language, when given an image _x_ and a question or an instruction _yq[l][t]_[in] the target language, the resultant model responds accurately though mostly in the pivot language. This can be attributed to the close resemblance between the hidden representation of instructions in two languages provided by the multilingual LLM, i.e., _fσ_ ( _yq[l][p]_[)] _[ ≈][f] σ_[(] _[y] q[l][t]_[)][.][Consequently, we have] _pθ_ ( _ya[l][p][|][h] ξ_[(] _[x]_[)] _[, f] σ_[(] _[y] q[l][p]_[))] _[ ≈][p] θ_[(] _[y] a[l][p][|][h] ξ_[(] _[x]_[)] _[, f] σ_[(] _[y] q[l][t]_[))][.][Since both the pretraining and instruction tuning] stages employ text components solely in the pivot language, the LLM can understand the question in the target language but cannot calibrate the response in the same language. To stimulate the model to respond in the target language, MPM incorporates a small number of translated pairs in the target language during instruction tuning. In this way, MPM simultaneously improves the model’s instruction-following capability and calibrates the response language, ultimately realizing a multimodal chatbot in the target language. 

## 3.3 TEXT-TO-IMAGE GENERATION 

In the text-to-image generation, we adopt a similar architecture with Stable Diffusion (Rombach et al., 2022). It incorporates a denoising network _gδ_ with a UNet architecture (Ronneberger et al., 2015) parameterized by _δ_ as an image decoder to generate images given the input prompt. The LLM _fσ_ and image decoder _gδ_ are interconnected with cross-attention mechanism (Vaswani et al., 2017). 

Diffusion models (Ho et al., 2020; Song et al., 2020) involve learning an iterative process of denoising a Gaussian noise into data distribution. The denoise network is optimized to remove the noise of noised image _xτ_ , conditioned on the hidden states of text input provided by the LLM. The training objective is defined as follows: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/zero-shot/iclr_2024_Large_Multilingual_Models_Pivot_Zero-Shot_Multimodal_Learning_across_Languages_images/iclr_2024_Large_Multilingual_Models_Pivot_Zero-Shot_Multimodal_Learning_across_Languages.pdf-0004-12.png)


In this stage, _ϕ_ = _{δ}_ , i.e., the image decoder is trained to align with frozen LLM. In this way, when input with the unseen prompt in the target language _y[l][t]_ , the multilingual LLM _fσ_ can inherently provide a representation _fσ_ ( _y[l][t]_ ) close to the seen representation _fσ_ ( _y[l][p]_ ) of the pivot language prompt with similar semantics. Therefore, the capability of text-to-image in the target language can be seamlessly transferred from the pivot language in a zero-shot fashion, illustrated as _gδ_ ( _xτ , fσ_ ( _y[l][t]_ ) _, τ_ ) _≈ gδ_ ( _xτ , fσ_ ( _y[l][p]_ ) _, τ_ ). 

4 

Published as a conference paper at ICLR 2024 

## 4 VISCPM 

As a practice of MPM, we develop a series of large-scale Chinese multimodal models called VISCPM. We use Chinese as the target language and English as the pivot language. The Chinese-English bilingual language model CPM-Bee (Zhang et al., 2021) serves as the backbone multilingual LLM. We have two variations of the model: VISCPM-Chat for image-to-text multimodal conversation and VISCPM-Paint for text-to-image synthesis. In the following sections, we begin by providing an overview of existing multimodal datasets in Chinese and then introduce the training procedure of VISCPM-Chat and VISCPM-Paint. 

## 4.1 ARE CHINESE MULTIMODAL DATASETS ENOUGH TO TRAIN A MULTIMODAL MODEL? 

The largest publicly available multimodal dataset of native Chinese is Wukong (Gu et al., 2022), consisting of 100M image-text pair pairs. However, by visualizing the CLIP score computed with Chinese-CLIP (Yang et al., 2022), as shown in Fig. 2, and manually inspecting, as introduced in Appendix G, we discover that only a minor fraction of image-text pairs in Wukong possess semantically matched content. The poor quality of the dataset escalates existing data resource shortfalls. A straightforward method to enhance the count of Chinese image-text pairs is to translate the English image-text pairs into Chinese, which has been utilized in previous work (Wang et al., 2022a; Qiu et al., 2022). However, translation requires an external machine translation model, and translating a large-scale dataset used in pretraining consumes substantial computing resources. Also, as discussed in Sec. 


![](raw/papers/2_LLM_NLP/LLM_paper_list/zero-shot/iclr_2024_Large_Multilingual_Models_Pivot_Zero-Shot_Multimodal_Learning_across_Languages_images/iclr_2024_Large_Multilingual_Models_Pivot_Zero-Shot_Multimodal_Learning_across_Languages.pdf-0005-05.png)


**----- Start of picture text -----**<br>
1e6 Histogram of CLIP Score<br>4<br>3<br>2<br>1<br>0<br>0.00 0.05 0.10 0.15 0.20 0.25 0.30 0.35<br>CLIP Score<br>Frequency<br>**----- End of picture text -----**<br>


Figure 2: The histogram of CLIP score of 100M Chinese image-text pairs from Wukong dataset. We set 0.18 as a moderate filtering threshold. 

5.3, we practically find that incorporating translated image-text only has marginal improvement on the performance when using powerful multimodal LLM as backbone language model, which already possesses strong cross-lingual generalization capability. Based on this analysis, we argue that effectively utilizing the existing English data to achieve knowledge transfer in multimodal alignment is the key to developing a powerful Chinese large multimodal model. 

## 4.2 � VISCPM-CHAT 

VISCPM-Chat is a Chinese-English bilingual multimodal chatbot capable of responding to users’ instructions based on the input image. VISCPM-Chat utilizes the Muffin architecture (Yu et al., 2023) as the image encoder. Specifically, Muffin directly leverages a pretrained vision-language model BEiT-3 (Wang et al., 2023) as an inherent bridge module between vision and language. In the _multimodal pretraining stage_ , the visual module is trained on 100M image-text pairs to align with the frozen LLM for 180K steps. In the _instruction tuning sub-stage_ , we utilize bilingual versions of LLaVA 150K (Liu et al., 2023a) and UniMM-Chat (Yu et al., 2023), and the English part of M[3] IT (Li et al., 2023b) to fine-tune the image encoder and LLM for 80K steps. The details of these datasets are presented in Appendix B.2. Due to the _quasi-zero-shot_ phenomenon in Chinese introduced in Sec. 3.2, we incorporate certain Chinese data by translating LLaVA 150K and UniMM-Chat into Chinese using machine translation.[1] 

To demonstrate the effect of Chinese image-text pairs, we also train an additional version of VISCPMChat, which adds additional Chinese image-text pairs during pretraining, including 20M native Chinese image-text pairs filtered from 100M Wukong (Gu et al., 2022) and 20M Zero-Chinese (Xie et al., 2022) dataset based on a CLIP score threshold greater than 0.18, and 136M image-text pairs translated from Laion-COCO dataset. We name this model as VISCPM-Chat+. 

> 1We employ CPM-Bee for machine translation. CPM-Bee can obtain 41.6 BLEU score on WMT 2017 test set (Bojar et al., 2017). Except for special instructions, all translation processes in this work are carried out using the CPM-Bee model. 

5 

Published as a conference paper at ICLR 2024 

Table 1: Experimental results on LLaVA Test Set accessed by GPT-4. Con: Conversation, DD: Detailed Description, CR: Complex Reasoning, AVG: the average score of three tasks. The best/second best results are marked in **bold** and underlined, respectively. 

|Model|Model|LLM<br>Backbone||English|English|Chinese|Chinese|
|---|---|---|---|---|---|---|---|
||||Con|DD<br>CR|AVG|Con<br>DD<br>CR|AVG|
|English<br>Model|MiniGPT-4 (Zhu et al., 2023)<br>InstructBLIP (Dai et al., 2023)<br>LLaVA (Liu et al., 2023a)|Vicuna-13B<br>Vicuna-13B<br>Vicuna-13B|65.0<br>81.9<br>**89.5**|67.3<br>76.6<br>68.0<br>91.2<br>**70.4**<br>96.2|69.7<br>80.5<br>**85.6**|-<br>-<br>-<br>-<br>-<br>-<br>-<br>-<br>-|-<br>-<br>-|
|En-Zh<br>Bilingual<br>Model|mPLUG-Owl (Ye et al., 2023)<br>VisualGLM (Du et al., 2022)<br>Ziya-Visual (Wang et al., 2022a)<br>Qwen-VL-Chat(Bai et al., 2023)|BLOOMZ-7B<br>ChatGLM-6B<br>Ziya-LLaMA-13B<br>Qwen-7B|64.6<br>62.4<br>82.7<br>82.4|47.7<br>80.1<br>63.0<br>80.6<br>69.9<br>92.1<br>76.9<br>91.9|64.2<br>68.7<br>81.7<br>83.8|76.3<br>61.2<br>77.8<br>76.6<br>87.8<br>83.6<br>85.0<br>74.7<br>82.4<br>82.3<br>**93.4**<br>89.5|72.0<br>82.7<br>80.8<br>88.2|
||VISCPM-Chat<br>VISCPM-Chat+|CPM-Bee-10B<br>CPM-Bee-10B|81.4<br>80.1|69.2<br>93.1<br>67.1<br>**97.1**|81.4<br>81.5|90.0<br>87.4<br>95.0<br>**91.3**<br>90.7<br>**95.4**|90.9<br>**92.5**|



## 4.3 � VISCPM-PAINT 

VISCPM-Paint is a text-to-image synthesis model that can accept prompts in both Chinese and English. VISCPM-Paint employs the UNet in Stable Diffusion (Rombach et al., 2022) as the image decoder. To maintain the generative capability of UNet, the training process only involves the cross-attention layer of UNet and the linear transblock between the multilingual LLM and UNet. We optimize these parameters using an extensive dataset of English image-text pairs Laion-2B (Schuhmann et al., 2022) for 300K steps. 

Similar to VISCPM-Chat+, we train an additional version of VISCPM-Paint, which is fine-tuned on Chinese image-text pairs. The component of these pairs are identical to that VISCPM-Chat+ uses. We name this model as VISCPM-Paint+. 

## 5 EXPERIMENTS 

## 5.1 EVALUATION OF VISCPM-CHAT 

## 5.1.1 EVALUATION SETTING 

**Baselines** . We compare VISCPM-Chat with existing multimodal conversation models, which include the English-only models: MiniGPT-4 (Zhu et al., 2023), InstructBLIP (Dai et al., 2023), and LLaVA (Liu et al., 2023a), as well as Chinese-English bilingual models: mPLUG-Owl (Ye et al., 2023), VisualGLM (Du et al., 2022), Ziya-Visual (Wang et al., 2022a), and Qwen-VL-Chat (Bai et al., 2023). All of these Chinese-English models have performed large-scale training on both Chinese and English multimodal datasets. More details of baselines are presented in Appendix C. 

**Evaluation Benchmark** . We assess the multimodal conversational abilities of the VISCPM-Chat in both English and Chinese on the LLaVA Test Set (Liu et al., 2023a) and UniMM-Bench (Yu et al., 2023). We use CPM-Bee to translate them into Chinese and manually check the translation results. Specifically, LLaVA comprehensively evaluates the model’s multimodal conversation capability from “conversation”, “detailed description”, and “complex reasoning”. UniMM-Bench, drawn from commonly used visual question-answering datasets—including OKVQA (Marino et al., 2019), AOKVQA (Schwenk et al., 2022), GQA (Hudson & Manning, 2019), and VQAv2 (Antol et al., 2015)—is designed particularly for evaluating multimodal conversation models on the abilities associated with reasoning, commonsense, and world knowledge. Considering that multimodal conversation models commonly generate responses with complete sentences to answer the question, traditional string-matching-based metrics, such as VQAScore (Antol et al., 2015), are not suitable in this context. Therefore, for the LLaVA Test Set and UniMM-Bench, we follow Liu et al. (2023a); Yu et al. (2023) and employ GPT-4[2] to rate the model-generated responses and the reference answers. Further details of the LLaVA Test Set and UniMM-Bench are presented in Appendix D.3. 

> 2Specifically, we use the GPT-4-0314 version to evaluate the responses. 

6 

Published as a conference paper at ICLR 2024 

Table 2: Evaluation of bilingual multimodal models on UniMM-Bench in Chinese. See results in English at Table 6. 

Table 3: Results of extension to more languages based on Vicuna. 

|Model|OKVQA<br>AOKVQA<br>GQA<br>VQAv2|AVG||Lang|Con<br>DD<br>CR|AVG|
|---|---|---|---|---|---|---|
|mPLUG-Owl<br>52.8<br>55.8<br>60.8<br>56.7<br>56.5<br>VisualGLM<br>53.7<br>57.5<br>64.8<br>62.5<br>59.6<br>Ziya-Visual<br>59.2<br>58.1<br>61.9<br>59.1<br>59.6<br>Qwen-VL-Chat<br>58.5<br>58.5<br>**70.6**<br>**72.0**<br>64.9||||_en_<br>_de_<br>_fr_<br>_es_<br>_it_<br>_pt_|87.6<br>76.8<br>101.0<br>90.8<br>80.8<br>93.6<br>87.6<br>81.9<br>94.6<br>87.1<br>79.6<br>102.3<br>84.5<br>79.5<br>93.3<br>81.3<br>81.6<br>92.4|88.6<br>88.7<br>88.2<br>89.8<br>85.9<br>85.1|
|VisCPM-Chat|**62.8**<br>**64.9**<br>68.3<br>71.8|**67.0**|||||



## 5.1.2 EXPERIMENTAL RESULTS 

**Quantitative Results.** The evaluation results on the LLaVA Test Set and UniMM-Bench are presented in Table 1 and Table 2. In **Chinese** , VISCPM-Chat achieves remarkable results and outperforms all baseline models on both benchmarks. It is worth noting that, unlike VisualGLM and Qwen-VL-Chat, which leverage a substantial number of Chinese image-text pairs during pretraining, VISCPMChat does not incorporate any Chinese multimodal data in its pretraining process. Despite this, VISCPM-Chat still presents superior performance on conversational abilities, commonsense, and world knowledge. Such results strongly demonstrate the effectiveness of MPM in transferring visual knowledge from English to Chinese. In **English** , the performance of VISCPM-Chat exceeds that of mPLUG-Owl, VisualGLM, InstructBLIP, and MiniGPT-4, and is roughly on par with Ziya-Visual and Qwen-VL-Chat while remaining comparable to the strong baseline LLaVA. 

**Case Study.** In addition to acquiring broad world knowledge and commonsense, MPM training also implicitly stimulated VISCPM-Chat’s competency in transferring knowledge unique to Chinese culture. For example, as shown in Fig. 3, For instance, as illustrated in Fig. 3, VISCPM-Chat is capable of relating the Chinese classical poems present in Su Shi’s "Water Melody" according to the full moon and pavilion. Although this image-text pair deeply related to Chinese culture is not included in the training data, VISCPM-Chat successfully navigates such a nontrivial knowledge transfer through MPM training. For a more detailed analysis of cases involving VISCPMChat, please refer to Appendix B.3. 


![](raw/papers/2_LLM_NLP/LLM_paper_list/zero-shot/iclr_2024_Large_Multilingual_Models_Pivot_Zero-Shot_Multimodal_Learning_across_Languages_images/iclr_2024_Large_Multilingual_Models_Pivot_Zero-Shot_Multimodal_Learning_across_Languages.pdf-0007-07.png)


**----- Start of picture text -----**<br>
你能用苏轼的诗句描述这幅图像吗？<br>(Can you describe this image with Su Shi‘s poems?)<br>在这张照片中，月亮高高地挂在天空上，照亮了古老的中国建筑和它周围<br>的环境。这创造出一种美丽而宁静的气氛，让人想起苏轼著名的诗句：<br>“明月几时有？把酒问青天。不知天上宫阙，今夕是何年？”<br>In this photo, the moon hangs high in the sky, illuminating the ancient<br>Chinese building and its surroundings. This creates a beautiful and serene<br>atmosphere, reminiscent of Su Shi's famous poem: "When is the bright moon?<br>Ask the blue sky for wine. I don't know the palace in the sky. What year is it?")<br>**----- End of picture text -----**<br>


Figure 3: Multimodal conversation cases of VISCPM-Chat in Chinese. 

## 5.1.3 GENERALIZATION TO MORE LANGUAGES 

The remarkable results of VISCPM-Chat on Chinese of MPM illuminate the potential broader application to a more diverse set of languages. Specifically, we leverage the multilingual LLM LLaMA (Touvron et al., 2023) as the LLM backbone and consider German, French, Spanish, Portuguese, and Italian as target languages. Following the same training procedure as in VISCPMChat, we develop a multilingual multimodal chatbot proficiently supporting six languages. We begin by pretraining the visual encoder with LLaMA to achieve visual feature alignment in English image-text pairs. Next, we employ M2M-100 (Fan et al., 2021) to translate the instruction training set of LLaVA into five target languages. The original English instruction training set and the five translated sets are merged and shuffled, then used for fine-tuning the visual module and LLM. 

Table 3 presents the evaluation results for English and five target languages on LLaVA Testset. See the results on IGLUE benchmark (Bugliarello et al., 2022) in Appendix D.5. Notably, for the relatively popular languages, such as German, French, and Spanish, the average scores exceed 88. Additionally, for Italian and Portuguese, the average scores are above 85. These results are highly encouraging as they demonstrate the chatbot’s coherent and accurate responses to visual-related questions in all six languages, even though the five target languages are simply blended during instruction tuning. The results in these languages validate the generalization and robustness of MPM in building powerful multimodal models in diverse linguistic contexts. 

7 

Published as a conference paper at ICLR 2024 

Table 4: Zero-shot FID on MSCOCO dataset. 

|Model|FID_↓_|FID_↓_|
|---|---|---|
||En|Ch|
|GLIDE (Nichol et al., 2022)<br>Make-A-Scene (Gafni et al., 2022)<br>DALL·E-2 (Ramesh et al., 2022)<br>UniDiffuser (Bao et al., 2023)<br>CogView2 (Ding et al., 2022)|12.2<br>11.8<br>10.4<br>9.7<br>-|-<br>-<br>-<br>-<br>24.0|
|Stable Diffusion (Rombach et al., 2022)<br>AltDiffusion (Chen et al., 2023)<br>TaiyiDiffusion (Wang et al., 2022a)|**8.6**<br>17.2<br>-|-<br>16.1<br>15.6|
|VISCPM-Paint<br>VISCPM-Paint+|9.5<br>9.9|10.9<br>**9.6**|




![](raw/papers/2_LLM_NLP/LLM_paper_list/zero-shot/iclr_2024_Large_Multilingual_Models_Pivot_Zero-Shot_Multimodal_Learning_across_Languages_images/iclr_2024_Large_Multilingual_Models_Pivot_Zero-Shot_Multimodal_Learning_across_Languages.pdf-0008-03.png)


**----- Start of picture text -----**<br>
Overall Alignment Fidelity<br>3/5<br>4/5<br>5/5<br>40<br>30<br>20<br>10<br>0<br>VisCPM Taiyi Alt VisCPM Taiyi Alt VisCPM Taiyi Alt<br>Percent<br>**----- End of picture text -----**<br>


Figure 4: Results of human evaluation for text-toimages on Chinese Drawbench. 

## 5.2 EVALUATION OF VISCPM-PAINT 

## 5.2.1 EVALUATION SETTING 

We compare VISCPM-Paint with several strong text-to-image model, which including English-only models: GLIDE (Nichol et al., 2022), Make-A-Scene (Gafni et al., 2022), DALL·E-2 (Ramesh et al., 2022), UniDiffuser (Bao et al., 2023), and Chinese or Chinese-English bilingual text-to-image models: CogView2 (Ding et al., 2022), AltDiffusion (Chen et al., 2023) and TaiyiDiffusion (Wang et al., 2022a). We mainly compare VISCPM-Paint with AltDiffusion (Chen et al., 2023) and TaiyiDiffusion (Wang et al., 2022a). More details of baselines are presented in Appendix C. 

## 5.2.2 AUTOMATIC EVALUATION 

For the text-to-image tasks, we assess the zero-shot Frechet Inception Distance (FID) (Heusel et al., 2017) and the CLIP score (Ramesh et al., 2021) using the MSCOCO validation set (Lin et al., 2014). We sample 30K prompts from MSCOCO, and for the Chinese evaluations, the text prompts are translated from the original English captions. We maintain the same sampling configuration for VISCPM-Paint, AltDiffusion, and TaiyiDiffusion and grid search the optimal FID across eight separate classifier guidance scales. 

We present the zero-shot FID on MSCOCO validation in Table 4. In **Chinese** , VISCPM-Paint achieves the best FID performance. By solely training on English image-text pairs, VISCPM-Paint displays a significant advantage over AltDiffusion (Chen et al., 2023) and TaiyiDiffusion (Wang et al., 2022a). In **English** , the performance of VISCPM-Paint is comparable to existing powerful text-to-image models, such as Stable Diffuson (Rombach et al., 2022) and UniDiffuser (Bao et al., 2023). See more analysis of fidelity and alignment trade-offs in Appendix E.2. 

## 5.2.3 HUMAN EVALUATION 

Following previous work (Yu et al., 2022b; Chang et al., 2023), we perform a human evaluation to have a more comprehensive understanding of model performance. Due to the lack of human evaluation benchmarks in Chinese, we create a Chinese text-to-image human evaluation benchmark, named _Chinese Drawbench_ . _Chinese Drawbench_ consists of 174 prompts and evaluates text-toimage models’ proficiency across different aspects. We conduct a human evaluation involving VISCPM-Paint, AltDiffusion (Chen et al., 2023), and TaiyiDiffusion (Wang et al., 2022a) on _Chinese Drawbench_ . We ask five independent annotators to judge the best image generated by three models for each prompt. Their evaluation criteria included Overall, Alignment, and Fidelity. More details of human evaluation and _Chinese Drawbench_ are provided in Appendix E.3. 

Figure 4 presents the human evaluation results. The figure shows the preference shares of 5 evaluators for each generated image with different marginals over consensuses in three aspects. Notably, VISCPM-Paint receives the strongest preference across all three evaluation aspects. In each aspect, more than 40 percent of images generated by VISCPM-Paint are the favored choice. Impressively, in the more objective metrics of Overall and Alignment, VISCPM-Paint earns more than 20 percent 5/5 

8 

Published as a conference paper at ICLR 2024 

Table 5: Performance in Chinese with different combinations of dataset languages in each training stage. IT means instruction tuning. 

(a) Image-to-text: Evaluation results on LLaVA Test Set. (b) Text-to-image: zero-shot FID on MSCOCO. 

||||||Dataset Language|Dataset Language|FID_↓_|
|---|---|---|---|---|---|---|---|
||||||Pretrain|Finetune||
|Native Chinese|Chinese|82.4<br>81.8<br>91.7|85.5||Native<br>Chinese<br>-<br>15.1|||
|English|Chinese|84.0<br>**83.6**<br>91.2|86.3|||||
||||||English|-|10.9|
|English|Bilingual|85.4<br>81.4<br>**96.6**|88.0|||||
|English+<br>Native Chinese|Bilingual|89.1<br>82.3<br>91.4|87.8||English|Native Chinese w/o Filter|12.8|
||||||English|Native Chinese|10.7|
|English+<br>Native Chinese+<br>Translated Chinese|Bilingual|**90.3**<br>81.4<br>92.1|**88.2**|||||
||||||English|Native Chinese +<br>Translated Chinese|**9.6**|



preference. These results strongly demonstrate the superior quality of VISCPM-Paint’s generated images compared to the two baseline models. 

## 5.3 ABLATION STUDY 

We conduct the ablation study of dataset languages to investigate the impact of varied dataset language combinations on multimodal models’ performance in image-to-text and text-to-image tasks. For better efficiency, we only use LLaVA 150K for instruction tuning in the image-to-text task. The detailed configurations of each experiment are reported in Appendix F. Based on the results presented in Table 5a and Table 5b, We have the following observations: (i) _Relying solely on a native Chinese dataset is insufficient for achieving good performance in both image-to-text and text-to-image tasks._ Models trained exclusively on native Chinese datasets yield worse scores, with the image-to-text obtaining an 85.5 average score and the text-to-image model attaining a 15.1 FID. (ii) _English data plays a crucial role in improving the chat capability of the model in Chinese during the instruction tuning stage._ When the image-to-text model is pretrained on a large dataset of English but then fine-tuned using the monolingual Chinese instruction tuning dataset, its average performance experiences a decline from 88.0 to 86.3 compared to the model utilizing the bilingual instruction tuning dataset. (iii) _The filtering process applied to the native dataset is essential for the Chinese performance._ In the textto-image task, after pretraining on English data and then fine-tuning with unfiltered native Chinese multimodal pairs, the FID worsens from 10.9 in the zero-shot scenario to 12.8. (iv) _Incorporating the native Chinese dataset and translated Chinese dataset yields a marginal improvement in the model performance._ Specifically, adding native Chinese data into the pretraining stage of the image-to-text model or fine-tuning stage of the text-to-image model only results in a 0.2 variation of metrics, while further mixing translated datasets improves the FID from 10.7 to 9.6. The improvement from VISCPM-Chat to VISCPM-Chat+, shown in Table 1, also confirms the same results. We present a more systematic exploration of the influence on the response language accuracy and content quality with different percentages of the Chinese instruction tuning datasets in Appendix D.4. 

## 6 CONCLUSION 

In this work, we introduce MPM, an innovative training paradigm designed for effectively training large multimodal models in non-English languages. By utilizing a multilingual LLM as a pivotal intermediary between vision signals and target languages, MPM facilitates the efficient transfer of multimodal alignment knowledge across different languages. Based on MPM, we develop a series of open-sourced Chinese large multimodal models VISCPM, which show remarkable capability in Chinese image-to-text and text-to-image tasks. Our experimental results demonstrate that by solely relying on English multimodal data, VISCPM can achieve the SOTA performance among Chinese open-sourced multimodal models. We further scale the scope of language by constructing a versatile multimodal chatbot that supports six distinct languages. We believe that the effectiveness of MPM will contribute to the development of large multimodal models worldwide, thereby fostering the development of sophisticated multimodal models across various languages and cultures. 

9 

Published as a conference paper at ICLR 2024 

## REFERENCES 

- Stanislaw Antol, Aishwarya Agrawal, Jiasen Lu, Margaret Mitchell, Dhruv Batra, C. Lawrence Zitnick, and Devi Parikh. VQA: Visual Question Answering. In _Proceedings of ICCV_ , 2015. 

- Jinze Bai, Shuai Bai, Shusheng Yang, Shijie Wang, Sinan Tan, Peng Wang, Junyang Lin, Chang Zhou, and Jingren Zhou. Qwen-vl: A frontier large vision-language model with versatile abilities. _arXiv preprint arXiv:2308.12966_ , 2023. 

- Fan Bao, Shen Nie, Kaiwen Xue, Chongxuan Li, Shi Pu, Yaole Wang, Gang Yue, Yue Cao, Hang Su, and Jun Zhu. One transformer fits all distributions in multi-modal diffusion at scale. In Andreas Krause, Emma Brunskill, Kyunghyun Cho, Barbara Engelhardt, Sivan Sabato, and Jonathan Scarlett (eds.), _Proceedings of ICML_ , volume 202, pp. 1692–1717. PMLR, 23–29 Jul 2023. 

- Marco Bellagente, Manuel Brack, Hannah Teufel, Felix Friedrich, Björn Deiseroth, Constantin Eichenberg, Andrew Dai, Robert Baldock, Souradeep Nanda, Koen Oostermeijer, et al. Multifusion: Fusing pre-trained models for multi-lingual, multi-modal image generation. _arXiv preprint arXiv:2305.15296_ , 2023. 

- Federico Bianchi, Giuseppe Attanasio, Raphael Pisoni, Silvia Terragni, Gabriele Sarti, and Sri Lakshmi. Contrastive language-image pre-training for the italian language. _arXiv preprint arXiv:2108.08688_ , 2021. 

- Ondˇrej Bojar, Rajen Chatterjee, Christian Federmann, Yvette Graham, Barry Haddow, Shujian Huang, Matthias Huck, Philipp Koehn, Qun Liu, Varvara Logacheva, et al. Findings of the 2017 conference on machine translation (wmt17). Association for Computational Linguistics, 2017. 

- Emanuele Bugliarello, Fangyu Liu, Jonas Pfeiffer, Siva Reddy, Desmond Elliott, Edoardo Maria Ponti, and Ivan Vuli´c. IGLUE: A benchmark for transfer learning across modalities, tasks, and languages. In Kamalika Chaudhuri, Stefanie Jegelka, Le Song, Csaba Szepesvari, Gang Niu, and Sivan Sabato (eds.), _Proceedings of the 39th International Conference on Machine Learning_ , volume 162 of _Proceedings of Machine Learning Research_ , pp. 2370–2392. PMLR, 17–23 Jul 2022. URL https://proceedings.mlr.press/v162/bugliarello22a.html. 

- Fredrik Carlsson, Philipp Eisen, Faton Rekathati, and Magnus Sahlgren. Cross-lingual and multilingual clip. In _Proceedings of LREC_ , pp. 6848–6854, 2022. 

- Huiwen Chang, Han Zhang, Jarred Barber, AJ Maschinot, Jose Lezama, Lu Jiang, Ming-Hsuan Yang, Kevin Murphy, William T Freeman, Michael Rubinstein, et al. Muse: Text-to-image generation via masked generative transformers. _arXiv preprint arXiv:2301.00704_ , 2023. 

- Soravit Changpinyo, Piyush Sharma, Nan Ding, and Radu Soricut. Conceptual 12M: Pushing webscale image-text pre-training to recognize long-tail visual concepts. In _Proceedings of CVPR_ , pp. 3558–3568, June 2021. 

- Xi Chen, Xiao Wang, Soravit Changpinyo, AJ Piergiovanni, Piotr Padlewski, Daniel Salz, Sebastian Goodman, Adam Grycner, Basil Mustafa, Lucas Beyer, et al. Pali: A jointly-scaled multilingual language-image model. In _Proceedings of ICLR_ , 2022. 

- Zhongzhi Chen, Guang Liu, Bo-Wen Zhang, Qinghong Yang, and Ledell Wu. AltCLIP: Altering the language encoder in CLIP for extended language capabilities. In _Findings of ACL_ , pp. 8666–8682, Toronto, Canada, July 2023. Association for Computational Linguistics. doi: 10.18653/v1/2023. findings-acl.552. 

- Schuhmann Christoph, Köpf Andreas, Vencu Richard, Coombes Theo, and Beaumont Romain. Laion COCO: 600M synthetic captions from Laion2B-En, 2022. URL https://laion.ai/blog/ laion-coco/. 

- James M Clark and Allan Paivio. Dual coding theory and education. _Educational psychology review_ , 3:149–210, 1991. 

- Wenliang Dai, Junnan Li, Dongxu Li, Anthony Meng Huat Tiong, Junqi Zhao, Weisheng Wang, Boyang Li, Pascale Fung, and Steven Hoi. InstructBLIP: Towards general-purpose vision-language models with instruction tuning, 2023. 

10 

Published as a conference paper at ICLR 2024 

- Abhishek Das, Satwik Kottur, Khushi Gupta, Avi Singh, Deshraj Yadav, José M.F. Moura, Devi Parikh, and Dhruv Batra. Visual Dialog. In _Proceedings of CVPR_ , 2017. 

Ming Ding, Wendi Zheng, Wenyi Hong, and Jie Tang. Cogview2: Faster and better text-to-image generation via hierarchical transformers. _NeurIPS_ , 35:16890–16902, 2022. 

- Alexey Dosovitskiy, Lucas Beyer, Alexander Kolesnikov, Dirk Weissenborn, Xiaohua Zhai, Thomas Unterthiner, Mostafa Dehghani, Matthias Minderer, Georg Heigold, Sylvain Gelly, et al. An image is worth 16x16 words: Transformers for image recognition at scale. In _Proceedings of ICLR_ , 2020. 

- Zhengxiao Du, Yujie Qian, Xiao Liu, Ming Ding, Jiezhong Qiu, Zhilin Yang, and Jie Tang. Glm: General language model pretraining with autoregressive blank infilling. In _Proceedings of ACL_ , pp. 320–335, 2022. 

- Patrick Esser, Robin Rombach, and Bjorn Ommer. Taming transformers for high-resolution image synthesis. In _Proceedings of CVPR_ , pp. 12873–12883, 2021. 

- Angela Fan, Shruti Bhosale, Holger Schwenk, Zhiyi Ma, Ahmed El-Kishky, Siddharth Goyal, Mandeep Baines, Onur Celebi, Guillaume Wenzek, Vishrav Chaudhary, et al. Beyond englishcentric multilingual machine translation. _JMLR_ , 22(1):4839–4886, 2021. 

- Oran Gafni, Adam Polyak, Oron Ashual, Shelly Sheynin, Devi Parikh, and Yaniv Taigman. Make-aScene: Scene-based text-to-image generation with human priors. In _Proceedings of ECCV_ , pp. 89–106. Springer, 2022. 

- Jiaxi Gu, Xiaojun Meng, Guansong Lu, Lu Hou, Minzhe Niu, Hang Xu, Xiaodan Liang, Wei Zhang, Xin Jiang, and Chunjing Xu. Wukong: 100 million large-scale chinese cross-modal pre-training dataset and a foundation framework, 2022. 

- Martin Heusel, Hubert Ramsauer, Thomas Unterthiner, Bernhard Nessler, and Sepp Hochreiter. GANs trained by a two time-scale update rule converge to a local nash equilibrium. _NeurIPS_ , 30, 2017. 

- Jonathan Ho, Ajay Jain, and Pieter Abbeel. Denoising diffusion probabilistic models. _NeurIPS_ , 33: 6840–6851, 2020. 

- Drew A Hudson and Christopher D Manning. Gqa: A new dataset for real-world visual reasoning and compositional question answering. In _Proceedings of CVPR_ , pp. 6700–6709, 2019. 

- Gabriel Ilharco, Mitchell Wortsman, Ross Wightman, Cade Gordon, Nicholas Carlini, Rohan Taori, Achal Dave, Vaishaal Shankar, Hongseok Namkoong, John Miller, Hannaneh Hajishirzi, Ali Farhadi, and Ludwig Schmidt. Openclip, July 2021. If you use this software, please cite it as below. 

- Alexander Kirillov, Eric Mintun, Nikhila Ravi, Hanzi Mao, Chloe Rolland, Laura Gustafson, Tete Xiao, Spencer Whitehead, Alexander C. Berg, Wan-Yen Lo, Piotr Dollár, and Ross Girshick. Segment anything. _arXiv:2304.02643_ , 2023. 

Ranjay Krishna, Yuke Zhu, Oliver Groth, Justin Johnson, Kenji Hata, Joshua Kravitz, Stephanie Chen, Yannis Kalantidis, Li-Jia Li, David A Shamma, et al. Visual Genome: Connecting language and vision using crowdsourced dense image annotations. _IJCV_ , 123:32–73, 2017. 

- Bowen Li, Xiaojuan Qi, Thomas Lukasiewicz, and Philip Torr. Controllable text-to-image generation. _NeurIPS_ , 32, 2019. 

- Junnan Li, Dongxu Li, Caiming Xiong, and Steven Hoi. BLIP: Bootstrapping language-image pre-training for unified vision-language understanding and generation. In Kamalika Chaudhuri, Stefanie Jegelka, Le Song, Csaba Szepesvari, Gang Niu, and Sivan Sabato (eds.), _Proceedings of ICML_ , volume 162, pp. 12888–12900. PMLR, 17–23 Jul 2022. 

- Junnan Li, Dongxu Li, Silvio Savarese, and Steven Hoi. BLIP-2: Bootstrapping language-image pre-training with frozen image encoders and large language models. In Andreas Krause, Emma Brunskill, Kyunghyun Cho, Barbara Engelhardt, Sivan Sabato, and Jonathan Scarlett (eds.), _Proceedings of ICML_ , volume 202, pp. 19730–19742. PMLR, 23–29 Jul 2023a. 

11 

Published as a conference paper at ICLR 2024 

- Lei Li, Yuwei Yin, Shicheng Li, Liang Chen, Peiyi Wang, Shuhuai Ren, Mukai Li, Yazheng Yang, Jingjing Xu, Xu Sun, et al. M[3] IT: A large-scale dataset towards multi-modal multilingual instruction tuning. _arXiv preprint arXiv:2306.04387_ , 2023b. 

- Tsung-Yi Lin, Michael Maire, Serge Belongie, James Hays, Pietro Perona, Deva Ramanan, Piotr Dollár, and C Lawrence Zitnick. Microsoft COCO: Common objects in context. In _Proceedings of ECCV_ , pp. 740–755. Springer, 2014. 

- Fangyu Liu, Emanuele Bugliarello, Edoardo Maria Ponti, Siva Reddy, Nigel Collier, and Desmond Elliott. Visually grounded reasoning across languages and cultures. In Marie-Francine Moens, Xuanjing Huang, Lucia Specia, and Scott Wen-tau Yih (eds.), _Proceedings of the 2021 Conference on Empirical Methods in Natural Language Processing_ , pp. 10467–10485, Online and Punta Cana, Dominican Republic, November 2021. Association for Computational Linguistics. doi: 10.18653/v1/ 2021.emnlp-main.818. URL https://aclanthology.org/2021.emnlp-main.818. 

- Haotian Liu, Chunyuan Li, Qingyang Wu, and Yong Jae Lee. Visual instruction tuning. _arXiv preprint arXiv:2304.08485_ , 2023a. 

- Shanyuan Liu, Dawei Leng, and Yuhui Yin. Bridge diffusion model: bridge non-english languagenative text-to-image diffusion model with english communities. _arXiv preprint arXiv:2309.00952_ , 2023b. 

- Shilong Liu, Zhaoyang Zeng, Tianhe Ren, Feng Li, Hao Zhang, Jie Yang, Chunyuan Li, Jianwei Yang, Hang Su, Jun Zhu, et al. Grounding DINO: Marrying dino with grounded pre-training for open-set object detection. _arXiv preprint arXiv:2303.05499_ , 2023c. 

- Kenneth Marino, Mohammad Rastegari, Ali Farhadi, and Roozbeh Mottaghi. OK-VQA: A visual question answering benchmark requiring external knowledge. In _Proceedings of CVPR_ , 2019. 

- Niklas Muennighoff, Thomas Wang, Lintang Sutawika, Adam Roberts, Stella Biderman, Teven Le Scao, M Saiful Bari, Sheng Shen, Zheng-Xin Yong, Hailey Schoelkopf, Xiangru Tang, Dragomir Radev, Alham Fikri Aji, Khalid Almubarak, Samuel Albanie, Zaid Alyafeai, Albert Webson, Edward Raff, and Colin Raffel. Crosslingual generalization through multitask finetuning, 2023. 

- Minheng Ni, Haoyang Huang, Lin Su, Edward Cui, Taroon Bharti, Lijuan Wang, Dongdong Zhang, and Nan Duan. M3P: Learning universal representations via multitask multilingual multimodal pre-training. In _Proceedings of CVPR_ , pp. 3977–3986, 2021. 

- Alexander Quinn Nichol, Prafulla Dhariwal, Aditya Ramesh, Pranav Shyam, Pamela Mishkin, Bob Mcgrew, Ilya Sutskever, and Mark Chen. GLIDE: Towards photorealistic image generation and editing with text-guided diffusion models. In Kamalika Chaudhuri, Stefanie Jegelka, Le Song, Csaba Szepesvari, Gang Niu, and Sivan Sabato (eds.), _Proceedings of ICML_ , volume 162, pp. 16784–16804. PMLR, 17–23 Jul 2022. 

OpenAI. GPT-4 technical report. _arXiv preprint arXiv:2303.08774_ , 2023. 

- Allan Paivio. _Bilingual Dual Coding Theory and Memory_ , pp. 41–62. Springer New York, New York, NY, 2014. ISBN 978-1-4614-9218-4. doi: P10.1007/978-1-4614-9218-4_3. 

- Jonas Pfeiffer, Gregor Geigle, Aishwarya Kamath, Jan-Martin Steitz, Stefan Roth, Ivan Vuli´c, and Iryna Gurevych. xGQA: Cross-lingual visual question answering. In Smaranda Muresan, Preslav Nakov, and Aline Villavicencio (eds.), _Findings of the Association for Computational Linguistics: ACL 2022_ , pp. 2497–2511, Dublin, Ireland, May 2022. Association for Computational Linguistics. doi: 10.18653/v1/2022.findings-acl.196. URL https://aclanthology.org/ 2022.findings-acl.196. 

- Chen Qiu, Dan Oneat,˘a, Emanuele Bugliarello, Stella Frank, and Desmond Elliott. Multilingual multimodal learning with machine translated text. In Yoav Goldberg, Zornitsa Kozareva, and Yue Zhang (eds.), _Findings of the Association for Computational Linguistics: EMNLP 2022_ , pp. 4178–4193, Abu Dhabi, United Arab Emirates, December 2022. Association for Computational Linguistics. doi: 10.18653/v1/2022.findings-emnlp.308. URL https://aclanthology. org/2022.findings-emnlp.308. 

12 

Published as a conference paper at ICLR 2024 

- Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, et al. Learning transferable visual models from natural language supervision. In _ICML_ , pp. 8748–8763. PMLR, 2021. 

- Aditya Ramesh, Mikhail Pavlov, Gabriel Goh, Scott Gray, Chelsea Voss, Alec Radford, Mark Chen, and Ilya Sutskever. Zero-shot text-to-image generation. In Marina Meila and Tong Zhang (eds.), _Proceedings of ICML_ , volume 139, pp. 8821–8831. PMLR, 18–24 Jul 2021. 

- Aditya Ramesh, Prafulla Dhariwal, Alex Nichol, Casey Chu, and Mark Chen. Hierarchical textconditional image generation with clip latents. _arXiv preprint arXiv:2204.06125_ , 2022. 

- Robin Rombach, Andreas Blattmann, Dominik Lorenz, Patrick Esser, and Björn Ommer. Highresolution image synthesis with latent diffusion models. In _Proceedings of CVPR_ , pp. 10684–10695, 2022. 

- Olaf Ronneberger, Philipp Fischer, and Thomas Brox. U-Net: Convolutional networks for biomedical image segmentation. In _MICCAI_ , pp. 234–241. Springer, 2015. 

- Chitwan Saharia, William Chan, Saurabh Saxena, Lala Li, Jay Whang, Emily L Denton, Kamyar Ghasemipour, Raphael Gontijo Lopes, Burcu Karagol Ayan, Tim Salimans, et al. Photorealistic text-to-image diffusion models with deep language understanding. _NeurIPS_ , 35:36479–36494, 2022. 

- Christoph Schuhmann, Richard Vencu, Romain Beaumont, Robert Kaczmarczyk, Clayton Mullis, Aarush Katta, Theo Coombes, Jenia Jitsev, and Aran Komatsuzaki. Laion-400M: Open dataset of clip-filtered 400 million image-text pairs. _arXiv preprint arXiv:2111.02114_ , 2021. 

- Christoph Schuhmann, Romain Beaumont, Richard Vencu, Cade Gordon, Ross Wightman, Mehdi Cherti, Theo Coombes, Aarush Katta, Clayton Mullis, Mitchell Wortsman, et al. Laion-5B: An open large-scale dataset for training next generation image-text models. _NeurIPS_ , 35:25278–25294, 2022. 

- Dustin Schwenk, Apoorv Khandelwal, Christopher Clark, Kenneth Marino, and Roozbeh Mottaghi. A-OKVQA: A benchmark for visual question answering using world knowledge. _arXiv_ , 2022. 

- Piyush Sharma, Nan Ding, Sebastian Goodman, and Radu Soricut. Conceptual Captions: A cleaned, hypernymed, image alt-text dataset for automatic image captioning. In _Proceedings of ACL_ , pp. 2556–2565, 2018. 

- Yang Song, Jascha Sohl-Dickstein, Diederik P Kingma, Abhishek Kumar, Stefano Ermon, and Ben Poole. Score-based generative modeling through stochastic differential equations. In _Proceedings of ICLR_ , 2020. 

- Yixuan Su, Tian Lan, Huayang Li, Jialu Xu, Yan Wang, and Deng Cai. PandaGPT: One model to instruction-follow them all. _arXiv preprint arXiv:2305.16355_ , 2023. 

- Hugo Touvron, Thibaut Lavril, Gautier Izacard, Xavier Martinet, Marie-Anne Lachaux, Timothée Lacroix, Baptiste Rozière, Naman Goyal, Eric Hambro, Faisal Azhar, et al. LLaMA: Open and efficient foundation language models. _arXiv preprint arXiv:2302.13971_ , 2023. 

- Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez, Łukasz Kaiser, and Illia Polosukhin. Attention is all you need. _NeurIPS_ , 30, 2017. 

- Junjie Wang, Yuxiang Zhang, Lin Zhang, Ping Yang, Xinyu Gao, Ziwei Wu, Xiaoqun Dong, Junqing He, Jianheng Zhuo, Qi Yang, Yongfeng Huang, Xiayu Li, Yanghan Wu, Junyu Lu, Xinyu Zhu, Weifeng Chen, Ting Han, Kunhao Pan, Rui Wang, Hao Wang, Xiaojun Wu, Zhongshen Zeng, Chongpei Chen, Ruyi Gan, and Jiaxing Zhang. Fengshenbang 1.0: Being the foundation of chinese cognitive intelligence. _CoRR_ , abs/2209.02970, 2022a. 

- Peng Wang, An Yang, Rui Men, Junyang Lin, Shuai Bai, Zhikang Li, Jianxin Ma, Chang Zhou, Jingren Zhou, and Hongxia Yang. OFA: Unifying architectures, tasks, and modalities through a simple sequence-to-sequence learning framework. In Kamalika Chaudhuri, Stefanie Jegelka, Le Song, Csaba Szepesvari, Gang Niu, and Sivan Sabato (eds.), _Proceedings of ICML_ , volume 162, pp. 23318–23340. PMLR, 17–23 Jul 2022b. 

13 

Published as a conference paper at ICLR 2024 

- Wenhui Wang, Hangbo Bao, Li Dong, Johan Bjorck, Zhiliang Peng, Qiang Liu, Kriti Aggarwal, Owais Khan Mohammed, Saksham Singhal, Subhojit Som, et al. Image as a foreign language: Beit pretraining for vision and vision-language tasks. In _Proceedings of CVPR_ , pp. 19175–19186, 2023. 

- Chunyu Xie, Heng Cai, Jianfei Song, Jincheng Li, Fanjing Kong, Xiaoyu Wu, Henrique Morimitsu, Lin Yao, Dexin Wang, Xiangzheng Zhang, Dawei Leng, Xiangyang Ji, and Yafeng Deng. Zero and R2D2: A large-scale chinese cross-modal benchmark and a vision-language framework, 2022. 

- An Yang, Junshu Pan, Junyang Lin, Rui Men, Yichang Zhang, Jingren Zhou, and Chang Zhou. Chinese CLIP: Contrastive vision-language pretraining in chinese. _arXiv preprint arXiv:2211.01335_ , 2022. 

- Yukang Yang, Dongnan Gui, Yuhui Yuan, Haisong Ding, Han Hu, and Kai Chen. Glyphcontrol: Glyph conditional control for visual text generation. 2023. 

- Qinghao Ye, Haiyang Xu, Guohai Xu, Jiabo Ye, Ming Yan, Yiyang Zhou, Junyang Wang, Anwen Hu, Pengcheng Shi, Yaya Shi, Chaoya Jiang, Chenliang Li, Yuanhong Xu, Hehong Chen, Junfeng Tian, Qian Qi, Ji Zhang, and Fei Huang. mPLUG-Owl: Modularization empowers large language models with multimodality, 2023. 

- Jiahui Yu, Zirui Wang, Vijay Vasudevan, Legg Yeung, Mojtaba Seyedhosseini, and Yonghui Wu. CoCa: Contrastive captioners are image-text foundation models. _arXiv preprint arXiv:2205.01917_ , 2022a. 

- Jiahui Yu, Yuanzhong Xu, Jing Yu Koh, Thang Luong, Gunjan Baid, Zirui Wang, Vijay Vasudevan, Alexander Ku, Yinfei Yang, Burcu Karagol Ayan, Ben Hutchinson, Wei Han, Zarana Parekh, Xin Li, Han Zhang, Jason Baldridge, and Yonghui Wu. Scaling autoregressive models for content-rich text-to-image generation. _TMLR_ , 2022b. ISSN 2835-8856. Featured Certification. 

- Tianyu Yu, Jinyi Hu, Yuan Yao, Haoye Zhang, Yue Zhao, Chongyi Wang, Shan Wang, Yinxu Pan, Jiao Xue, Dahai Li, Zhiyuan Liu, Haotao Zheng, and Maosong Sun. Reformulating vision-language foundation models and datasets towards universal multimodal assistants. _arXiv_ , 2023. 

- Lu Yuan, Dongdong Chen, Yi-Ling Chen, Noel Codella, Xiyang Dai, Jianfeng Gao, Houdong Hu, Xuedong Huang, Boxin Li, Chunyuan Li, et al. Florence: A new foundation model for computer vision. _arXiv preprint arXiv:2111.11432_ , 2021. 

- Yan Zeng, Wangchunshu Zhou, Ao Luo, Ziming Cheng, and Xinsong Zhang. Cross-view language modeling: Towards unified cross-lingual cross-modal pre-training. In Anna Rogers, Jordan BoydGraber, and Naoaki Okazaki (eds.), _Proceedings of the 61st Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)_ , pp. 5731–5746, Toronto, Canada, July 2023. Association for Computational Linguistics. doi: 10.18653/v1/2023.acl-long.315. URL https://aclanthology.org/2023.acl-long.315. 

- Ao Zhang, Hao Fei, Yuan Yao, Wei Ji, Li Li, Zhiyuan Liu, and Tat-Seng Chua. Transfer visual prompt generator across llms. _arXiv preprint arXiv:2305.01278_ , 2023. 

- Zhengyan Zhang, Xu Han, Hao Zhou, Pei Ke, Yuxian Gu, Deming Ye, Yujia Qin, Yusheng Su, Haozhe Ji, Jian Guan, et al. CPM: A large-scale generative chinese pre-trained language model. _AI Open_ , 2:93–99, 2021. 

- Mingyang Zhou, Luowei Zhou, Shuohang Wang, Yu Cheng, Linjie Li, Zhou Yu, and Jingjing Liu. UC2: Universal cross-lingual cross-modal vision-and-language pre-training. In _Proceedings of CVPR_ , pp. 4155–4165, 2021. 

- Deyao Zhu, Jun Chen, Xiaoqian Shen, Xiang Li, and Mohamed Elhoseiny. Minigpt-4: Enhancing vision-language understanding with advanced large language models. _arXiv preprint arXiv:2304.10592_ , 2023. 

- Minfeng Zhu, Pingbo Pan, Wei Chen, and Yi Yang. DM-GAN: Dynamic memory generative adversarial networks for text-to-image synthesis. In _Proceedings of CVPR_ , pp. 5802–5810, 2019. 

14 

Published as a conference paper at ICLR 2024 

## APPENDIX 

## A CONTRIBUTIONS 

The authors’ contributions can be outlined as follows: 

In the preparation of the project, Jinyi Hu and Yuan Yao design the model architecture. Xu Han, Yankai Lin, Jiao Xue, Dahai Li, Zhiyuan Liu, and Maosong Sun offer invaluable guidance in refining the model’s architecture. Shan Wang, Chongyi Wang, and Jinyi Hu take charge of collecting and processing the extensive multimodal dataset required for pretraining. Additionally, Hanghao Wu, Yue Zhao, Haoye Zhang, and Yuan Yao collaborate on constructing the instruction tuning data. Jinyi Hu, Chongyi Wang, Tianyu Yu, Qianyu Chen, and Shan Wang jointly implement the training codebase. 

In model training, Chongyi Wang, Tianyu Yu, and Yinxu Pan babysit the VISCPM-Chat training; Jinyi Hu, Shan Wang and Qianyu Chen take care of the VISCPM-Paint training. 

In model evaluation, Jinyi Hu and Yuan Yao design the evaluation framework. Chongyi Wang and Tianyu Yu evaluate VISCPM-Chat; Jinyi Hu and Shan Wang execute the automatic evaluation of VISCPM-Paint and organize the human evaluation of VISCPM-Paint. 

In paper writing, Jinyi Hu and Yuan Yao write the main paper; Yankai Lin, Zhiyuan Liu, and Maosong Sun provide suggestions to polish the writing. 

For public usability, Jinyi Hu, Yinxu Pan, Chongyi Wang, Shan Wang, and Yuan Yao promote the open-source of VISCPM; Yinxu Pan develops the online demo and API of VISCPM; Chongyi Wang and Yinxu Pan implement the low-resource inference of VISCPM. 

Throughout the project, Xu Han, Yankai Lin, Jiao Xue, Dahai Li, Zhiyuan Liu, and Maosong Sun provide invaluable technical guidance and advice. 

## B DATASET 

## B.1 PRETRAINING DATASET 

**COCO** (Lin et al., 2014): The COCO dataset is a meticulously compiled image caption dataset that encompasses everyday scenes with common objects. It includes a training set of 118,287 images, each complemented by five unique captions. For each pass through the dataset, we randomly select one caption, resulting in 591,435 image-text pairs, five times the total number of images. 

**Visual Genome** (Krishna et al., 2017) Visual Genome stands as a meticulously labeled image dataset, enriched with detailed annotations for objects. Our employed training set features approximately 100K images, each accompanied by an average of 8 unique captions. 

**CC3M** (Sharma et al., 2018) Also known as Conceptual Captions, the CC3M dataset contains approximately 3.3M image-text pairs. CC3M is collected from the web and carefully processed to achieve the cleanliness and informativeness of captions. Due to some broken image links, the final successfully downloaded dataset contains approximately 2.8M pairs. 

**CC12M** (Changpinyo et al., 2021) The CC12M dataset is an extension of the CC3M (Sharma et al., 2018), hosting nearly 12M image-text pairs. A more relaxed collection pipeline was used for its creation. After downloading, our dataset comprises approximately 6M pairs. 

**Laion2B** (Schuhmann et al., 2022) Laion-2B is a massive dataset populated with image data sourced from publicly accessible areas of the internet. Our successful download yielded a vast collection of around 1.3B images. 

**Laion-COCO** (Christoph et al., 2022) Laion-COCO is a subset of the Laion-2B dataset which includes 600M image entries. These images have been captioned using the BLIP (Li et al., 2022) to generate high-quality descriptions mimicking the MS COCO style. 

**Wukong** (Gu et al., 2022) Wukong is a 100M image-text pair in Chinese, where the images are filtered according to the image size, and the text is filtered according to its language, length, and frequency. 

15 

Published as a conference paper at ICLR 2024 

Table 6: Evaluation of bilingual multimodal chat models on UniMM-Bench in English. 

|Model|OKVQA<br>AOKVQA<br>GQA<br>VQAv2|AVG|
|---|---|---|
|mPLUG-Owl<br>VisualGLM<br>Ziya-Visual<br>Qwen-VL-Chat|66.7<br>62.5<br>63.0<br>66.0<br>57.6<br>62.8<br>58.1<br>63.9<br>66.1<br>68.7<br>67.4<br>67.3<br>**71.4**<br>**77.9**<br>68.6<br>**77.9**|64.6<br>60.6<br>67.4<br>**74.0**|
|VISCPM-Chat|65.4<br>75.5<br>**71.5**<br>76.6|**72.3**|



Table 7: Summary of multimodal chat models’ architecture, parameters, and training data. Here, the size of mPlug-Owl’s training data and the proportion of Chinese and English in the Ziya-Visual’s training data are reported in their model cards. 

|Model|Visual Module<br>LLM|Training Data|
|---|---|---|
|mPlug-Owl<br>VisualGLM<br>Ziya-Visual<br>Qwen-VL-Chat<br>VISCPM-Chat|ViT-L/14 (0.3B)<br>BLOOMZ-7B<br>Q-Former (1.6B)<br>ChatGLM-6B<br>Q-Former (1.1B)<br>Ziya-LLaMA-13B-v1<br>ViT-bigG (1.9B)<br>Qwen-7B<br>Muffn (0.7B)<br>CPM-Bee-10B|-<br>English: 300M; Chinese 30M<br>20M<br>English: 1.1B; Chinese: 300M<br>English: 140M; Chinese: 1M|



**Zero** (Xie et al., 2022) Sourced from a search engine, the Zero dataset comprises 20M images and corresponding textual descriptions, selected from a pool of 5B image-text pairs based on user click-through rate. 

## B.2 INSTRUCTION TUNING DATASET 

**LLaVA-Instruct-150K** (Liu et al., 2023a) LLaVA-Instruct-150K is a set of multimodal instructionfollowing data generated by GPT4. In its creation, image captions and corresponding bounding boxes are harnessed to encode images into textual sequences for the text-only GPT4 model. LLaVAInstruct-150K divides the dataset into three types: Conversation, Detailed Description, and Complex Reasoning. For each type, manually designed examples are incorporated into prompts for in-context learning of GPT4. 

**M**[3] **IT** (Li et al., 2023b) The M[3] IT dataset is a collection of large-scale multimodal instruction tuning datasets curated by leveraging downstream datasets covering diverse vision-language tasks, including captioning, reasoning, and visual question-answering. The datasets have been reformulated into a unified image-text schema. Certain constructed instances, derived from key datasets, are translated into a total of 100 languages. It is important to note that we do not proceed with additional translations for non-Chinese portions of the dataset. It’s worth noting that we do not proceed with additional translations for non-Chinese portions of the dataset. 

**UniMM-Chat** (Yu et al., 2023) UniMM-Chat is a bespoke knowledge-intensive multimodal conversational dataset. UniMM-Chat utilizes images from the MSCOCO (Lin et al., 2014) and associated labeled datasets as the foundation for data generation. The related datasets include VQAv2 (Antol et al., 2015), Visual Dialogue (Das et al., 2017), OKVQA (Marino et al., 2019), and AOKVQA (Schwenk et al., 2022). The initial step involves merging the annotation information across different datasets for the same image, including pairs of questions and answers, dialogues, rationale information, and image captions. The varied text annotations provide a diversified view of the image content, enabling the comprehensive interpretation of the image by ChatGPT. 

Table 8: Summary of text-to-image models’ architecture and training data. 

|Model|Visual Module<br>Text Encoder|Training Data|
|---|---|---|
|AltDiffusion<br>TaiyiDiffusion<br>VISCPM-Paint|UNet<br>AltCLIP<br>UNet<br>Taiyi-CLIP-RoBERTa<br>UNet<br>CPM-Bee|-<br>English: 0; Chinese: 20M<br>English: 1.3B; Chinese: 0|



16 

Published as a conference paper at ICLR 2024 

Table 9: Desription and examples of different categories in Chinese Drawbench 

|Category|Description|Example|
|---|---|---|
|Relation|Ability to generate objects with specific interaction<br>relationship between them|“挂在⾐架上的⼀个帽⼦”<br>(A hat hanging on a hanger.)|
|Compositional|Ability to multiply kinds of objects|“⼀个盘⼦上放有⻩⾊的梨和紫⾊的苹果”<br>(There are yellow pears and purple apples ona plate.)|
|Attribute|Ability to accurately generate objects with given<br>attributes, such as size, color, and action.|“⼀件蓝⾊的天鹅绒晚礼服”<br>(A blue velvet evening gown.)|
|Counterintuitive|Ability to generate unusual scene against common<br>sense|“穿着宇航服的李⽩开宇宙⻜船”<br>(Li Bai flies a spaceship in a space suit.)|
|Rare Word|Ability to understand rare expression|窗台上的⼀盆剑兰<br>(Agladiolus on the windowsill.)|
|Count|Ability to generate given numbers of objects|三块涂有⻩油的⾯包<br>(Three pieces of bread with butter.)|
|Long Input|Ability to understand long and complex input|平静的湖⾯上，船夫划着船桨在湖⾯上划过，泛起涟漪，⼀只⻦<br>⻜过⽔⾯，抓起来⼀只⻥<br>(On the calm lake, the boatman rowed his paddle across the lake,<br>rippling, a birdflew overthe waterand caught afish.)|
|Chinese Culture|Ability to generate scene related to Chinese Culture|海上⽣明⽉，天涯共此时<br>(As the bright moon shines over the sea,<br>From far away you share this moment with me.)|
||||



Table 10: Details of image-text datasets used in VISCPM-Chat’s pretraining, where “bil” means the mixed version of English and translated Chinese. 

|Datasets|Size|Weight|Epoch|
|---|---|---|---|
|VISCPM-Chat||||
|COCO + Visual Genome|626K|12.50%|27.60|
|CC3M + CC12M|8.4M|25.00%|16.40|
|Laion-COCO|390M|62.50%|0.35|
|VISCPM-Chat+||||
|COCO + VG|626K|12.50%|15.33|
|CC12M|5.6M|25.00%|3.43|
|Zero + Wukong|20M|12.50%|0.48|
|CC3M-bil|5.6M|12.50%|1.72|
|Laion-COCO-bil|780M|37.50%|0.04|



## B.3 CASE STUDY 

Fig. 7 and Fig. 8 vividly depict VISCPM-Chat’s capabilities, embodying wide-ranging global knowledge assimilation. As illustrated in the two case studies in Fig. 7, VISCPM-Chat can identify the Mona Lisa painting adapted in a surreal style and recognize a stained map of New York City, further interpreting its real-world meaning. Demonstrating a well-balanced capacity for ChineseEnglish multi-modal conversation and robust text recognition skills, as exemplified in Fig. 8, VISCPM-Chat can engage in fluent multimodal conversations on varying topics in English and effectively identify the words “Starbucks”, “Avengers: Endgame”, and its release date, “April 26th”, from visual inputs. 

## C DETAILS OF BASELINES 

The details of existing Chinese-English bilingual multimodal chat models are summarized in Table 7 and introduced as follows: 

- mPLUG-Owl (Ye et al., 2023): mPLUG-Owl consists of a vision encoder, a vision abstractor module, and BLOOMZ (Muennighoff et al., 2023) as language model backbone. It is trained on LAION-400M, COYO, CC3M, and MSCOCO. 

17 

Published as a conference paper at ICLR 2024 

Table 11: Performance of MPM on IGLUE benchmark. We compare MPM with mUNITER, xUNITER (Liu et al., 2021), UC[2] (Zhou et al., 2021) and M[3] P (Ni et al., 2021). 

|Model|xNLI|xNLI|xNLI|xNLI|xGQA|xGQA|xGQA|xGQA|xGQA|xGQA|xGQA|
|---|---|---|---|---|---|---|---|---|---|---|---|
||ARB|SPA|FRA|RUS|BEN|DEU|IND|KOR|POR|RUS|CMN|
|mUNITER<br>xUNITER<br>UC2<br>M3P|46.7<br>52.0<br>**56.2**<br>55.2|57.0<br>58.9<br>57.5<br>58.9|59.4<br>63.3<br>69.5<br>56.4|51.7<br>59.7<br>64.9<br>62.5|3.1<br>10.8<br>**20.0**<br>18.6|24.0<br>34.8<br>42.9<br>33.4|9.4<br>33.7<br>28.7<br>32.5|4.2<br>12.1<br>21.4<br>25.1|13.7<br>22.1<br>30.4<br>31.4|8.5<br>18.8<br>31.0<br>27.5|7.0<br>19.6<br>31.2<br>28.7|
|MPM|55.8|**75.9**|**78.1**|**75.0**|6.6|**51.0**|**37.1**|**36.9**|**49.8**|**47.9**|**46.1**|



- VisualGLM (Du et al., 2022): VisualGLM employs Q-Former as image encoder and ChatGLM-6B (Du et al., 2022) as language model backbone. VisualGLM-6B’s pretraining incorporates 30M high-quality Chinese image-text pairs and 300M filtered English image-text pairs. In the fine-tuning phase, VisualGLM is trained on long VQA data. 

- Ziya-Visual (Wang et al., 2022a): Ziya-Visual leverage Q-Former (Li et al., 2023a) as image encoder and Ziya-LLaMA-13B-v1 as language model backbone. They utilize 20M Chinese imagetext pair, which is built by cleaning high-quality data from open source data, translating English datasets, and extracting coarse-grained information from captions using BLIP (Li et al., 2022) and Grounded SAM (Kirillov et al., 2023; Liu et al., 2023c). 

- Qwen-VL-Chat (Bai et al., 2023): Qwen-VL-Chat utilizes a large ViT (Dosovitskiy et al., 2020) with 1.9B parameters initialized from Openclip’s bigG (Ilharco et al., 2021) as image encoder and Qwen-7B as language model backbone. The image encoder and language model are connected by a one-layer cross-attention module augmented with positional encodings. Qwen-VL-Chat p 

The details of existing Chinese-English bilingual text-to-image models are summarized in Table 8 and introduced as follows: 

- AltDiffusion (Chen et al., 2023): AltDiffusion is a Chinese-English bilingual text-to-image model based on Stable Diffusion and bilingual vision-language encoder AltClip (Chen et al., 2023). The training data are collected form Laion (Schuhmann et al., 2021). 

- TaiyiDiffusion (Wang et al., 2022a): TaiyiDiffusion is a Chinese text-to-image model that adapts a Chinese text encoder into Stable Diffusion. The visual part is frozen during training. The training datasets include 20M filtered Chinese image-text pairs. 

## D DETAILS OF VISCPM-CHAT 

## D.1 MULTIMODAL PRETRAINING 

In the visual feature alignment pretraining, we train the visual encoder on a mix of image-text pair datasets, including CC3M (Sharma et al., 2018), CC12M (Changpinyo et al., 2021), COCO (Lin et al., 2014), Visual Genome (Krishna et al., 2017), and Laion-COCO (Christoph et al., 2022). We train VISCPM-Chat and VISCPM-Chat+ for 180K and 480K steps, respectively, with a batch size of 768 and a learning rate of 1e-5. The distribution of each dataset used in pretraining for VISCPM-Chat and VISCPM-Chat+ are shown in Table 10. Only the parameters of the visual module are optimized in this stage. 

## D.2 INSTRUCTION TUNING 

During the instruction tuning phase, the optimization process involves the visual module and the LLM. We train VISCPM-Chat on the mixed dataset from LLaVA-Instruct-150K, UniMM-Chat, and M[3] IT, where LLaVA-Instruct-150K and UniMM-Chat are Chinese-English bilingual versions. The instruction tuning lasts for 80K steps with a batch size of 64. The training configuration of VISCPM-Chat+ is identical to VISCPM-Chat in this stage. In the inference phase, VISCPM-Chat uses beam search decoding with a beam size of 3. 

18 

Published as a conference paper at ICLR 2024 


![](raw/papers/2_LLM_NLP/LLM_paper_list/zero-shot/iclr_2024_Large_Multilingual_Models_Pivot_Zero-Shot_Multimodal_Learning_across_Languages_images/iclr_2024_Large_Multilingual_Models_Pivot_Zero-Shot_Multimodal_Learning_across_Languages.pdf-0019-01.png)


**----- Start of picture text -----**<br>
Comparison of Response Language Accuracy and Content Quality<br>100.0 88<br>97.5<br>87<br>95.0<br>92.5 86<br>90.0<br>85<br>87.5<br>85.0<br>84<br>82.5<br>83<br>Percent of Translated Chinese Data (%)<br>0.1 0.5 1.0 10.0 25.0 50.0 75.0100.0<br>Response Content Quality<br>Response Language Accuracy (%)<br>**----- End of picture text -----**<br>


Figure 5: The trend of response language accuracy and content quality when using different percentages of translated Chinese data during instruction tuning. 

## D.3 EVALUATION BENCHMARK 

**LLaVA Test Set** (Liu et al., 2023a) LLaVA Test Set consists of 90 instances, each containing an image, a question, and an answer, which comprehensively evaluates the model’s performance in conversation, detailed description, and complex reasoning. Following LLaVA (Liu et al., 2023a), we use GPT-4[3] to rate the model-generated answers and reference answers in a range of 1-10. We norm the average score of each instance to 1-100. 

**UniMM-Bench** (Yu et al., 2023) UniMM-Bench consists of 400 test instances, uniformly sampled from 4 commonly used VQA benchmarks, including OKVQA (Marino et al., 2019), AOKVQA (Schwenk et al., 2022), GQA (Hudson & Manning, 2019), and VQAv2 (Antol et al., 2015), whose annotations has undergone meticulous check. In the traditional evaluation of visual question answering, the metrics are to compute whether the model-generated answers are exactly matched to the reference answers. However, considering that LLMs typically generate responses with one complete sentence to answer the question, UniMM-Bench is built to comprehensively evaluate multimodal chat models’ capabilities in the context of visual question-answering, including reasoning, commonsense, grounding, and world knowledge (Schwenk et al., 2022). Given the cost of using GPT-4 for evaluation, 400 test instances are sampled from reliable VQA benchmarks. 

## D.4 THE TREND OF USING DIFFERENT NUMBER OF CHINESE TRANSLATED DATA 

To investigate the influence of varying amounts of Chinese data on the model’s ability to respond in the target language during the instruction tuning, surpassing the quasi-zero-shot transfer threshold, we systematically varied the percentages of translated Chinese data. Our analysis, depicted in Fig 5, elucidates the trends in both response language accuracy and content quality as a function of the Chinese data percentage utilized during instruction tuning. Remarkably, introducing a mere 0.1% of translated Chinese data suffices to achieve over 80% accuracy in response language accuracy, signifying the model’s high sensitivity to even minimal language-specific data inputs. Elevating this percentage to 1% results in a scenario where all responses are rendered in Chinese. Regarding content quality, our findings suggest that a range between 25% to 50% of translated Chinese data represents an optimal balance, yielding high-quality content responses. We leave the exploration of determining a more precise optimal proportion of translated Chinese data for maximizing both response content quality in future work. 

## D.5 ADDITIONAL RESULTS ON IGLUE 

We evaluate the effectiveness of MPM in multilingual image understanding on two benchmarks of IGLUE: xVNLI (Bugliarello et al., 2022) and xGQA (Pfeiffer et al., 2022). We choose our model 

> 3We use GPT-4 0314 version to evaluate the answers. 

19 

Published as a conference paper at ICLR 2024 


![](raw/papers/2_LLM_NLP/LLM_paper_list/zero-shot/iclr_2024_Large_Multilingual_Models_Pivot_Zero-Shot_Multimodal_Learning_across_Languages_images/iclr_2024_Large_Multilingual_Models_Pivot_Zero-Shot_Multimodal_Learning_across_Languages.pdf-0020-01.png)


**----- Start of picture text -----**<br>
cfg-scales: [1.5 2.0 3.0 4.0 5.0 6.0 7.0 8.0]<br>TaiyiDiffusion<br>24 AltDiffusion<br>VisCPM-Paint<br>22 VisCPM-Paint+<br>20<br>18<br>16<br>14<br>12<br>10<br>20 21 22 23 24<br>CLIP Score (ViT-L/14)<br>FID Score (30K)<br>**----- End of picture text -----**<br>


Figure 6: The curves of FID _↓ vs_ CLIP score _↑_ with different classifier guidance scales. 

pre-trained exclusively on English image-text pairs with LLaMA as the backbone language model, fine-tune it with IGLUE’s English training set, and conduct a zero-shot evaluation following the official setting of IGLUE. 

As Table 11 shows, the performance of MPM demonstrates a notable improvement over baseline models. Specifically, when using Chinese-English bilingual LLM CPM-Bee as the backbone language model, MPM attains a 56.05 score on xGQA for CMN (Chinese Mandarin). Based on the above results, we can observe that for languages where multilingual LLMs demonstrate proficiency, such as Spanish, French, Russian, and German, the multilingual capability can be effectively generalized to achieve strong multimodal capabilities in the target language. 

## E DETAILS OF VISCPM-PAINT 

## E.1 TRAINING AND SAMPLING 

VISCPM-Paint consists of a UNet, initialized from Stable Diffusion, and a bilingual LLM CPM-Bee (Zhang et al., 2021). We pretrain VISCPM-Paint on English image-text pairs dataset Laion-2B (Schuhmann et al., 2022) for 300k steps with batch size 4096. The 300K step consists of 200K-step training on 256 _×_ 256 resolutions and 100K training 512 _×_ 512 resolutions. We only optimize the cross-attention layer of UNet and the linear transblock between LLM and UNet. Linear transblock consists of a linear layer and a layer-norm layer, where the linear layer converts the dimension of LLM’s hidden states into UNet’s dimension. After pretraining on Laion-2B, we continue to train VISCPM-Paint+ on 20M filtered native Chinese image-text pairs and 136M translated image-text pairs in Laion-COCO (Christoph et al., 2022). We sample the generated image with DDPM scheduler (Ho et al., 2020) with 50 steps. For each prompt, we run three models to generate four images and select the one with the highest CLIP score evaluated by Chinese-CLIP (Yang et al., 2022). 

## E.2 TRADE OFF OF FIDELITY AND ALIGNMENT 

To visualize the trade-off between fidelity and alignment, Fig. 6 plots the curves of the FID against the CLIP score under different classifier guidance scales. The results indicate VISCPM-Paint and VISCPM-Paint+ deliver good overall performance regarding the balance between image quality and semantic alignment. While the CLIP score of AltDiffusion slightly outperforms that of VISCPMPaint and VISCPM-Paint+, AltDiffusion’s FID falls significantly short. The sub-optimal quality of generated images will affect the practical use, as shown in the human evaluation below. 

20 

Published as a conference paper at ICLR 2024 

## E.3 HUMAN EVALUATION 

To better understand the model performance on text-to-image generation in Chinese, we create a comprehensive set of prompts in Chinese, named _Chinese Drawbench_ . _Chinese Drawbench_ contains 174 prompts and consists of 8 categories, including Relation, Compositional, Attribute, Counterintuitive, Rare Word, Count, Long Input, and Chinese Culture. The descriptions and examples of each category are shown in Table 9. During the construction of _Chinese Drawbench_ , we asked 7 annotators to create 25 prompts per person conditioned on the given categories. To diversify the prompts, we also restrain the field of 20 prompts ranging from 9 common classes, such as animal, plants, food, space, etc, and leave five prompts with no restrain. After this, we double-check the prompts and polish them minor. 

We invite 5 independent human evaluators to assess the performance of VISCPM-Paint, TaiyiDiffusion (Wang et al., 2022a), and AltDiffusion on _Chinese Drawbench_ . For each instance, three random shuffled images and the input prompt are shown to the human evaluators. We ask them to select the best one among three images from 3 aspects: Overall, Alignment, and Fidelity. Alignment measures the consistency between the generated image and the input prompt. Fidelity measures the clarity, aesthetics appeal, and object realism of the image. Overall measures the comprehensive quality of generated images. Figure 10 provides detailed results of each category of _Chinese DrawBench_ . 

## F DETAIL OF ABLATION STUDY 

We introduce the detailed experimental configuration in the ablation study. 

For image-to-text tasks shown in Table 5a, the native Chinese dataset consists of 20M image-text pairs filtered from Wukong (Gu et al., 2022) and Zero (Xie et al., 2022). For the “Pretrain” column in Table 5a, “Native Chinese” stands for the 20M native Chinese image-text pairs filtered from Wukong (Gu et al., 2022) and Zero (Xie et al., 2022). “English” corresponds to the dataset used in VISCPM-Chat, i.e., COCO, Visual Genome, CC3M, CC12M, and Laion-COCO. “Translated Chinese” consists of 390M Chinese image-text pairs whose captions are translated from English in Laion-COCO. In instruction tuning, i.e., supervised fine-tuning (SFT), we only utilize LLaVA and its translated Chinese version to reduce the computation time in the ablation study. Specifically, “Chinese” in the “SFT” column of Table 5a means translated Chinese LLaVA instruction tuning dataset. “Bilingual” means the mixed version of English and translated Chinese LLaVa. 

For text-to-image tasks shown in Table 5b, “English” stands for Laion-2B. The other setting is the same as the image-to-text tasks introduced above. 

## G DETAIL OF CLIP SCORE 

We use Chinese-CLIP (Yang et al., 2022) to compute the CLIP score in Chinese image-text pairs. Chinese CLIP models have been used in several Chinese text-to-image models for data filtering and text encoding (Wang et al., 2022a; Liu et al., 2023b; Yang et al., 2023). We referred to Taiyidiffusion’s criteria for data volume and filtering standards (including CLIP score) in training Chinese text-toimage models. We manually check the caption in the Wukong dataset. We observe that the overall quality of text content in Wukong is unsatisfactory. Here are some findings about low CLIP scores: 

- Broken Images: Some image URLs are broken or redirected to the same default image for downloading. 

- Unrelated captions: Some content of captions are totally unrelated to image content. 

- Incomplete sentences: Some captions are truncated or missing punctuation. 

21 

Published as a conference paper at ICLR 2024 


![](raw/papers/2_LLM_NLP/LLM_paper_list/zero-shot/iclr_2024_Large_Multilingual_Models_Pivot_Zero-Shot_Multimodal_Learning_across_Languages_images/iclr_2024_Large_Multilingual_Models_Pivot_Zero-Shot_Multimodal_Learning_across_Languages.pdf-0022-01.png)


**----- Start of picture text -----**<br>
你怎么看这幅画？<br>(Please appreciate this painting)<br>这幅画描绘了一只猫的肖像，它有长长的胡须和大眼睛。这只猫直勾勾地盯着观众看，给人一种好奇或顽皮的感觉。在背景中可<br>以看到一个绿色的墙，为画面增添了一抹色彩。整幅作品具有艺术性与趣味性，捕捉到了猫咪独特而富有表现力的特征。<br>(This painting depicts a portrait of a cat with long whiskers and large eyes. The cat is staring straight at the viewer, giving it a<br>sense of curiosity or playfulness. A green wall can be seen in the background, adding a splash of color to the picture. The<br>whole composition is artistic and playful, capturing the unique and expressive characteristics of cats.)<br>这只猫的形象和哪副世界名画中的人物形象有关？为什么?<br>(The image of this cat is related to the characters in<br>which world famous painting? Why?)<br>这只猫的形象与达芬奇名画《蒙娜丽莎》中女人的形象相似，因为它被描绘成一个长着大眼睛、有胡须和头发的女人。这幅画以<br>其神秘而富有表现力的微信著称，而这只猫似乎也在做同样的表情——睁大了眼睛直勾勾地盯着观众。这种相似性可能是由于两<br>者都具有独特的面部特征或表现出一种类似于达芬奇艺术作品中所捕捉到的情感和氛围。<br>(The image of the cat is similar to the image of the woman in Leonardo da Vinci's famous painting "Mona Lisa", as it is depicted<br>as a woman with big eyes, beard and hair. The painting is known for its enigmatic and expressive wechat, and the cat seems to<br>be doing the same—staring wide-eyed and intently at the viewer. This similarity could be due to both having distinctive facial<br>features or displaying an emotion and atmosphere similar to those captured in da Vinci's artwork.)<br>地图中画的是哪个城市？<br>(Which city is drawn on the map?)<br>这是地图显示的是纽约市。<br>(This is a map of New York City.)<br>图中不同的色块是什么含义？<br>(What do the different color blocks in the picture mean?)<br>图中不同的颜色块代表了纽约市各个区域，每个区域都有自己独特的颜色。这些色彩编码系统有助于人们快速、容易地识别和区<br>分城市内不同的街区或社区。通过使用这种编码方式，居民和游客可以很容易地找到他们想要去的地方并更好地理解这座城市的<br>布局与组织结构。 图中不同颜色分别代表什么？<br>(The different colored blocks in the map representWhat dothethevariousdifferentboroughscolors inof New York City, each with its own unique color.<br>These color-coding systems help people quicklythe picture represent?and easily identify and distinguish between different neighborhoods or<br>neighborhoods within a city. By using this coding method, residents and tourists can easily find where they want to go and<br>better understand the layout and organization of the city.)<br>**----- End of picture text -----**<br>


Figure 7: Multimodal conversation cases of VISCPM-Chat in Chinese. VISCPM-Chat can recognize artwork like the Mona Lisa, albeit adapted in a surrealistic style, and a detailed, stained map of New York City. 

22 

Published as a conference paper at ICLR 2024 


![](raw/papers/2_LLM_NLP/LLM_paper_list/zero-shot/iclr_2024_Large_Multilingual_Models_Pivot_Zero-Shot_Multimodal_Learning_across_Languages_images/iclr_2024_Large_Multilingual_Models_Pivot_Zero-Shot_Multimodal_Learning_across_Languages.pdf-0023-01.png)


**----- Start of picture text -----**<br>
Which movie is this poster from and which characters<br>appear in the poster?<br>This poster is for the movie "Avengers: Endgame”, which is set to release on April<br>26th. The characters featured in the poster are Thor, Hulk, Iron Man, Captain America,<br>Black Widow, and Hawkeye.<br>Who about the villain?<br>The villain in the movie is Thanos, a powerful and dangerous character from the<br>Avengers Movie.<br>What is this place in the picture?<br>The place in the picture is a Starbucks coffee shop, with a large green and white<br>sign on its side that reads "Starbucks”.<br>What can I buy here?<br>In the image, there is a Starbucks coffee shop sign hanging on the side of a<br>building. This suggests that one can buy beverages, such as coffee or tea, at this<br>location. The presence of the Starbucks logo also indicates that they offer various<br>products and services related coffee, including coffee-related merchandise like<br>mugs, T-shirts, and other accessories.<br>**----- End of picture text -----**<br>


Figure 8: Multimodal conversation cases of VISCPM-Chat in English. VISCPM-Chat can conduct fluent multimodal conversations on different topics in English and recognize the words “Starbucks”, “Avengers: Endgame” and its release date “April 26th” in the image. 

23 

Published as a conference paper at ICLR 2024 


![](raw/papers/2_LLM_NLP/LLM_paper_list/zero-shot/iclr_2024_Large_Multilingual_Models_Pivot_Zero-Shot_Multimodal_Learning_across_Languages_images/iclr_2024_Large_Multilingual_Models_Pivot_Zero-Shot_Multimodal_Learning_across_Languages.pdf-0024-01.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/zero-shot/iclr_2024_Large_Multilingual_Models_Pivot_Zero-Shot_Multimodal_Learning_across_Languages_images/iclr_2024_Large_Multilingual_Models_Pivot_Zero-Shot_Multimodal_Learning_across_Languages.pdf-0024-02.png)


Figure 9: Generated images of VISCPM-Paint. 

24 

Published as a conference paper at ICLR 2024 


![](raw/papers/2_LLM_NLP/LLM_paper_list/zero-shot/iclr_2024_Large_Multilingual_Models_Pivot_Zero-Shot_Multimodal_Learning_across_Languages_images/iclr_2024_Large_Multilingual_Models_Pivot_Zero-Shot_Multimodal_Learning_across_Languages.pdf-0025-01.png)


**----- Start of picture text -----**<br>
Alt Taiyi VisCPM<br>100<br>80<br>60<br>40<br>20<br>0<br>(a) Overall<br>100<br>80<br>60<br>40<br>20<br>0<br>(b) Alignment<br>100<br>80<br>60<br>40<br>20<br>0<br>(c) Fidelity<br>Relation Compositional Attribute Counterintuitive Rare Word Count Long Input Chinese Culture<br>Relation Compositional Attribute Counterintuitive Rare Word Count Long Input Chinese Culture<br>Relation Compositional Attribute Counterintuitive Rare Word Count Long Input Chinese Culture<br>Percent<br>Percent<br>Percent<br>**----- End of picture text -----**<br>


Figure 10: Human preference rate on Chinese Drawbench in different kinds of aspects. 

25 

