## **MultiCapCLIP: Auto-Encoding Prompts for Zero-Shot Multilingual Visual Captioning** 

**Bang Yang[1,2]** _[∗]_ **, Fenglin Liu[3] , Xian Wu** _[∗]_ **[4] , Yaowei Wang[2] , Xu Sun[5]** _[†]_ **, and Yuexian Zou[1]** _[†]_ 1ADSPLAB, School of ECE, Peking University 2Peng Cheng Laboratory 3University of Oxford 4Tencent Jarvis Lab 5School of Computer Science, Peking University {yangbang, zouyx}@pku.edu.cn; fenglin.liu@eng.ox.ac.uk 

## **Abstract** 

Supervised visual captioning models typically require a large scale of images or videos paired with descriptions in a specific language (i.e., the vision-caption pairs) for training. However, collecting and labeling large-scale datasets is time-consuming and expensive for many scenarios and languages. Therefore, sufficient labeled pairs are usually not available. To deal with the label shortage problem, we present a simple yet effective zero-shot approach MultiCapCLIP that can generate visual captions for different scenarios and languages without any labeled vision-caption pairs of downstream datasets. In the training stage, MultiCapCLIP only requires text data for input. Then it conducts two main steps: 1) retrieving concept prompts that preserve the corresponding domain knowledge of new scenarios; 2) autoencoding the prompts to learn writing styles to output captions in a desired language. In the testing stage, MultiCapCLIP instead takes visual data as input directly to retrieve the concept prompts to generate the final visual descriptions. The extensive experiments on image and video captioning across four benchmarks and four languages (i.e., English, Chinese, German, and French) confirm the effectiveness of our approach. Compared with state-of-theart zero-shot and weakly-supervised methods, our method achieves 4.8% and 21.5% absolute improvements in terms of BLEU@4 and CIDEr metrics. Our code is available at https: //github.com/yangbang18/MultiCapCLIP. 

## **1 Introduction** 

Visual captioning targets to first 1) understand the information of visual inputs, which are typically videos or images, and then 2) produces a corresponding textual sentence describing the visual objects/attributes/relationships. Visual captioning has drawn remarkable attention from natural language 

> _∗_ Equal contribution. 

> _†_ Corresponding authors. 

processing and computer vision fields due to its wide applications, e.g., cross-modal retrieval (Luo et al., 2022; Cheng et al., 2023b) and helping the visually impaired (Çaylı et al., 2021). Currently, visual captioning models based on the encoderdecoder framework (Huang et al., 2020; Liu et al., 2020; Yang et al., 2021; Zhang et al., 2021; Hu et al., 2022; Lin et al., 2022) have achieved tremendous progress in advancing the state-of-the-art. These models are usually trained with full supervision and rely on large-scale humanly-annotated training data (i.e., vision-caption pairs), which needs expensive labeling work. In particular, when it comes to Non-English caption systems, it is challenging to collect and label sufficient visioncaption pairs in a timely manner, which prevents such encoder-decoder models from rapid deployment in different scenarios and languages. 

To deal with the shortage of labeled pairs, we propose the MultiCapCLIP - a prompt-based natural language auto-encoder. As shown in Figure 1, MultiCapCLIP only requires textual input for training, and it can conduct zero-shot multilingual visual captioning, including image and video captioning. Therefore, MultiCapCLIP can deal with the situation where the labeled vision-caption pairs are missing. MultiCapCLIP is particularly suitable for new scenarios and languages, improving the practical value of visual captioning. 

To implement MultiCapCLIP, we first adopt a pre-trained vision-language model, i.e., CLIP (Radford et al., 2021), as our backbone. CLIP has shown success in correlating the visual and textual modalities into the same latent space (vision-language embedding space) (Tewel et al., 2022b; Su et al., 2022; Zeng et al., 2023). We observe two critical issues for zero-shot visual captioning: the understanding of domain visual knowledge (e.g., objects, attributes, and relationships) and the generation of descriptive sentences in a specific writing style and language. Therefore, we propose a prompt-based 

auto-encoder, which introduces the visual concept prompts _P_ to preserve the corresponding domain knowledge and writing styles of zero-shot visual captioning. During training, given the text-only data, we train the model by reconstructing the caption _S_ in the _S →P → S_ auto-encoding pipeline. Since the auto-encoding process reconstructs the same input sentence, the model training needs only unlabeled text data. In the reconstruction process, the model is able to preserve the necessary domain knowledge and the writing styles of visual captioning (Wang et al., 2016; Tschannen et al., 2018). During inference, we can directly take the vision input _V_ as queries to retrieve the domain knowledge preserved in the visual concept prompts and finally rely on the learned writing styles in a specific language in the text decoder to generate visual descriptions in the _V →P → S_ pipeline. 

Meanwhile, to further bridge the modality gap between the visual and textual data (Liang et al., 2022), we introduce an augmentation method, including input augmentation and feature augmentation, which can boost the robustness of the model and in turn improve the performance of zeroshot visual captioning. The experiments on four benchmark datasets, i.e., MS-COCO (Chen et al., 2015), MSR-VTT (Xu et al., 2016), VATEX (Wang et al., 2019), and Multi30K (Elliott et al., 2016), show that our approach can accurately and dataefficiently generate visual captions in English, Chinese, German, and French. 

Overall, our main contributions are as follows: 

- We propose a simple yet effective approach MultiCapCLIP that requires no downstream labeled data to make the first attempt for zeroshot multilingual visual captioning. 

- MultiCapCLIP first introduces visual concept prompts to preserve the domain knowledge and then auto-encodes them to learn the writing styles of captioning. After text-only training, our approach can shift from text-to-text generation to vision-to-text generation. 

- The out-of-domain and in-domain experiments on image and video captioning across different languages show that our approach trained on text-only data significantly outperforms previous zero-shot/weakly-supervised methods trained on unpaired or partial labeled visual and textual data, setting new state-ofthe-art zero-shot performance. 

## **2 Approach** 

In this section, we first give a brief review of CLIP, whose vision-language embedding space lays a foundation for our approach. Next, we introduce the framework of the proposed MultiCapCLIP, followed by two key components: concept prompts and textual augmentations. 

## **2.1 A Brief Review of CLIP** 

CLIP uses two independent encoders to process image and text input separately and then bridges the gap between modalities with contrastive learning. The image encoder _ϕv_ ( _·_ ) can be a convolutional neural network like ResNet (He et al., 2016) or a vision Transformer like ViT (Dosovitskiy et al., 2021), and it extracts a feature vector for each input image. The text encoder _ϕt_ ( _·_ ) is based on Transformer (Vaswani et al., 2017), and it outputs a vector representation of the input text. By training two encoders on 400M imagetext data with noisy correspondences under the InfoNCE objective (Oord et al., 2018), CLIP learns a powerful vision-language embedding space that measures image-text similarity well and enables open-vocabulary classification. In this paper, we re-purpose CLIP for zero-shot multilingual visual captioning and always keep _ϕv_ ( _·_ ) and _ϕt_ ( _·_ ) frozen. 

## **2.2 Overview of MultiCapCLIP** 

As shown in Figure 1, MultiCapCLIP consists of visual and textual encoders from CLIP and a trainable Multilingual Language Model (MLM). MultiCapCLIP supports English text[1] , images or videos as inputs and can produce output in desired language. Specifically, we implement MLM with a stack of Transformer decoder blocks, each of which comprises a masked self-attention layer, a crossattention layer, and a feed-forward layer. Moreover, we add explicit signals in the embedding layer to indicate which language to be generated. 

Let denote the text input as _S_ , the vision input as _V_ , and concept prompts as _P_ . Unlike typical visual captioning models that are trained on a vision-text dataset, MultiCapCLIP relies on a text dataset and follows the _S → P → S_ auto-encoding pipeline during training. Based on the semantic alignment characteristic of CLIP’s feature space, MultiCapCLIP uses the _V → P → S_ pipeline for visual captioning during inference. We extend MultiCapCLIP to support multilingual text generation by us- 

1The training corpora for CLIP is mainly in English. 


![](raw/papers/2_LLM_NLP/LLM_paper_list/caption/acl_2023_MultiCapCLIP_Auto-Encoding_Prompts_for_Zero-Shot_Multilingual_Visual_Captioning_images/acl_2023_MultiCapCLIP_Auto-Encoding_Prompts_for_Zero-Shot_Multilingual_Visual_Captioning.pdf-0003-00.png)


**----- Start of picture text -----**<br>
Training: Text Input Vision-Language Embedding Space Multilingual Language Model<br>A young girl sings with a microphone on the stage EncoderText retrieval Concept Prompts A young girl sings with a microphone on the stage<br>Training %(#)<br>Inference: Vision Input Inference &! DecoderText or<br>!(#)<br>Embedding<br>Image 0 … Position 一位年轻女孩<br>Encoder en/zh … Language 在舞台上用麦<br>Vision Embedding Text Embedding [BOS] … Token 克风唱歌<br>%′(#)<br>Augmentation Noisy Text Embedding<br>**----- End of picture text -----**<br>


Figure 1: Illustration of MultiCapCLIP. It comprises of a frozen CLIP-like vision-language pre-trained model and a trainable multilingual language model. During training, MultiCapCLIP only requires text data and aims to produce a reconstruction/translation result based on the text feature _ft_ of the source input and its relevant concept prompts (§ 2.3). During inference, MultiCapCLIP replaces _ft_ with the feature(s) of an image or a video for captioning. 

ing parallel corpora with ( _S, T_ ) pairs, where _T_ denotes the target text in a desired language. In such a case, MultiCapCLIP follows the _S/V → P → T_ translation pipeline. 

In the following, we will detail how to extract and leverage _P_ in Section 2.3. Then in Section 2.4, we will introduce an augmentation method to improve the training of MultiCapCLIP. 

## **2.3 Decoding with Concept Prompts** 

A set of visual concepts is a good embodiment of domain visual knowledge because a visual concept (e.g., “a young girl”) manifest as the explicit clue in the vision input. Given a pure text dataset, we use the spaCy toolkit[2] to extract noun phrases and reserve the most frequent 1,000 noun phrases as visual concepts, which are first embedded into a prompt template “{concept}”[3] and then fed into the CLIP’s text encoder _ϕt_ ( _·_ ) to extract l2-normalized concept features _C_ = _{c_ 1 _, . . . , c_ 1000 _}_ . 

During training, given the text input _S_ , we first encode it into a global feature _ft_ : 


![](raw/papers/2_LLM_NLP/LLM_paper_list/caption/acl_2023_MultiCapCLIP_Auto-Encoding_Prompts_for_Zero-Shot_Multilingual_Visual_Captioning_images/acl_2023_MultiCapCLIP_Auto-Encoding_Prompts_for_Zero-Shot_Multilingual_Visual_Captioning.pdf-0003-07.png)


where Norm( _·_ ) denotes L2 normalization. Next, we calculate the dot product of _ft_ and _C_ to measure cosine similarities, based on which we obtain _soft_ concept prompts _P_ , a subset of _C_ that includes _K_ concept features most semantically similar to _ft_ . Assuming that the dimension of vectors outputted by CLIP is _d_ , _P_ is in the shape of _K ∗ d_ . To prompt MLM, we prefix embeddings of the target text _S_ 

> 2https://spacy.io 

> 3The simplest prompt template “{concept}” produced better performance than other templates like “a concept of {concept}” in our preliminary experiments. 

with _P_ to obtain the final input embeddings _E_ : 


![](raw/papers/2_LLM_NLP/LLM_paper_list/caption/acl_2023_MultiCapCLIP_Auto-Encoding_Prompts_for_Zero-Shot_Multilingual_Visual_Captioning_images/acl_2023_MultiCapCLIP_Auto-Encoding_Prompts_for_Zero-Shot_Multilingual_Visual_Captioning.pdf-0003-12.png)


where _ω_ ( _·_ ) is implemented as a fully connected layer followed by a layer normalization (LN) (Ba et al., 2016), and _e_ ( _·_ ) denotes the summation of position, language, and token embeddings for each _si ∈ S_ , followed by LN. The prompt sequence generated by _ω_ ( _P_ ) and token sequence generated by _e_ ( _S_ ) are concatenated together and sent to the text decoder of MLM to regenerate the input sequence _S_ . Considering that _ft_ may contain information supplementary to _P_ , we do not discard _ft_ . We first feed the projected feature _f_ = _ω[′]_ ( _ft_ ), where _ω[′]_ ( _·_ ) has the same structure as _ω_ ( _·_ ) but shares no parameters, into the text decoder of MLM. Then we calculate the cross attention between _f_ and _E_ . We train the model with a cross-entropy loss: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/caption/acl_2023_MultiCapCLIP_Auto-Encoding_Prompts_for_Zero-Shot_Multilingual_Visual_Captioning_images/acl_2023_MultiCapCLIP_Auto-Encoding_Prompts_for_Zero-Shot_Multilingual_Visual_Captioning.pdf-0003-14.png)


where _pθ_ ( _·_ ) is MLM’s predicted distribution over a vocabulary and _θ_ denotes all trainable parameters. 

During inference, we process the vision input _V_ in a similar manner, except that we use CLIP’s image encoder _ϕv_ ( _·_ ) to obtain _V_ ’s vector representation _fv_ and obtain relevant concept prompts _P_ based on (averaged) image-concept similarities. Given the previously generated text _S<i_ , the prediction of the next token is based on the following probability distribution: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/caption/acl_2023_MultiCapCLIP_Auto-Encoding_Prompts_for_Zero-Shot_Multilingual_Visual_Captioning_images/acl_2023_MultiCapCLIP_Auto-Encoding_Prompts_for_Zero-Shot_Multilingual_Visual_Captioning.pdf-0003-17.png)


## **2.4 Training with Augmentations** 

Our MultiCapCLIP’s ability of shifting text-to-text generation to vision-to-text generation is built on 

||**MS-COCO**|**MSR-VTT**|**VATEX**|
|---|---|---|---|
|Vision Type|Image|Video|Video|
|Training size|113,287|6,513|25,006|
|Validation size|5,000|497|1,393|
|Testing size|5,000|2,990|1,500|
|# Captions|616,767|200,000|278,990|
|Avg. Length|10.6|9.3|12.3|
|Target Language|English|English|Chinese|



Table 1: Statistics of datasets used in main experiments. 

the assumption that the paired vision-text data is well aligned in the vision-language embedding space of CLIP. However, Liang et al. (2022) demonstrated that there exists _modality gap_ in CLIP-like models and such gap has a significant impact on model generalization ability. To this end, inspired by _denoising_ auto-encoders (Vincent et al., 2008), we propose to train MultiCapCLIP with augmented text features _ft[′]_[.][Here we consider both the input] augmentation (IA) and the feature augmentation (FA). Specifically, IA replaces the source text _S_ with a semantically similar one _S[′]_ to obtain _ft[′]_[:] 


![](raw/papers/2_LLM_NLP/LLM_paper_list/caption/acl_2023_MultiCapCLIP_Auto-Encoding_Prompts_for_Zero-Shot_Multilingual_Visual_Captioning_images/acl_2023_MultiCapCLIP_Auto-Encoding_Prompts_for_Zero-Shot_Multilingual_Visual_Captioning.pdf-0004-03.png)


where _S[′] ∼_ X _S_ and X _S_ = _{S, S_ 1 _[′][, . . . , S] N[′] −_ 1 _[}]_[ de-] notes the candidate set of _S_ . For simplicity, we use _ϕt_ ( _·_ ) to measure similarities among text in the dataset and select the most similar _N −_ 1 text to construct X _S_ for each _S_ . Since we sample text from X _S_ with uniform probability, there will be 1 _/N_ probability that the input text keeps unchanged. As for FA, we follow Li et al. (2021) to add Gaussian noise _n ∼N_ (0 _, ϵ_ ) into text features. Hence, Eq. (5) can be further extended to: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/caption/acl_2023_MultiCapCLIP_Auto-Encoding_Prompts_for_Zero-Shot_Multilingual_Visual_Captioning_images/acl_2023_MultiCapCLIP_Auto-Encoding_Prompts_for_Zero-Shot_Multilingual_Visual_Captioning.pdf-0004-05.png)


During training, we replace _ft_ in Eq. (1) with _ft[′]_ in Eq. (6) to encourage the model to learn more robust latent representations. 

## **3 Main Experiments** 

In this section, we first introduce the datasets, metrics, settings of the experiments; Then, we provide the out-of-domain and in-domain results of our approach for zero-shot visual captioning. 

## **3.1 Experimental Setups** 

**Datasets.** As shown in Table 1, we use three benchmark datasets under CC BY 4.0 licence in 

|**Settings and**<br>**Languages**<br>**Training Data + Prompts**<br>**(****_Text-only data_)**<br>**Testing Data**|**Settings and**<br>**Languages**<br>**Training Data + Prompts**<br>**(****_Text-only data_)**<br>**Testing Data**|
|---|---|
|Out-of-<br>Domain|English<br>MSR-VTT<br>MS-COCO<br>MS-COCO<br>MSR-VTT|
||Chinese<br>MSR-VTT-CN<br>VATEX|
|In-<br>Domain|English<br>MS-COCO<br>MS-COCO<br>MSR-VTT<br>MSR-VTT|
||Chinese<br>VATEX<br>VATEX|



Table 2: Training and testing data used for different experimental settings. We adopt two English captioning datasets: MS-COCO (Chen et al., 2015), MSR-VTT (Xu et al., 2016) and two Chinese captioning datasets: MSR-VTT-CN (Wang et al., 2022b), and VATEX (Wang et al., 2019), to conduct the main experiments. 

this section: MS-COCO (Chen et al., 2015), MSRVTT (Xu et al., 2016), and VATEX (Wang et al., 2019). We apply the Karpathy and Fei-Fei’s (2015) split to MS-COCO and follow the official split of MSR-VTT for English captioning. Besides, VATEX is a multilingual video captioning dataset that contains parallel English-Chinese captions. We use it for Chinese captioning[4] . In Section 4, we will further use the Multi30K dataset (Elliott et al., 2016) for German and French caption generation. 

**Metrics.** Following the common practice in the literature, we report BLEU@4 (Papineni et al., 2002), METEOR (Banerjee and Lavie, 2005), ROUGE-L (Lin, 2004) and CIDEr (Vedantam et al., 2015) for video captioning, and additionally measure SPICE (Anderson et al., 2016) for image captioning. All metrics are computed by Microsoft COCO Evaluation Server[5] (Chen et al., 2015). 

**Settings** As shown in Table 2, we conduct the out-of-domain and in-domain experiments. 1) _Outof-Domain Experiments_ are performed by training the model on the text-only data of A dataset, and then evaluating on the B dataset. 2) _In-Domain Experiments_ are conducted by training the model on the text-only data of A dataset, and then evaluating on the A dataset. 

> 4The official splits of VATEX is 25,991: 3,000: 6,000. However, some video clips are no longer available, resulting in the splits 25,006: 2,893: 5,792. Besides, VATEX does not provide Chinese captions of the test set (https://eric-xw. github.io/vatex-website). We construct new validation and test sets from the original validation set. 

5For non-English captioning, we do not report METEOR and SPICE metrics because their implementations consider synonym matching and named entity recognition in English. 

|**Settings Methods**<br>**Pre-trained Backbone**<br>**Training Data**<br>Vision<br>Text<br>|**Settings Methods**<br>**Pre-trained Backbone**<br>**Training Data**<br>Vision<br>Text<br>|**MS-COCO (English)**<br>B@4<br>M<br>R-L<br>C<br>S<br>|**MSR-VTT (English)**<br> <br>B@4<br>M<br>R-L<br>C<br>|**VATEX (Chinese)**<br>B@4 R-L<br>C|
|---|---|---|---|---|
|Weakly-<br>Supervised<br>UIC_†_(2019)<br>Inception + Faster R-CNN<br>IC-SME_†_(2019)<br>ResNet + Faster R-CNN<br>R2M_†_(2020)<br>Faster R-CNN<br>TSGAN_†_(2021)<br>Faster R-CNN<br>SGM_†_(2021)<br>Inception + Faster R-CNN|~~_√_~~<br>~~_√_~~<br>_√_<br>_√_<br>_√_<br>_√_<br>_√_<br>_√_<br>_√_<br>_√_|5.6<br>12.4 28.7 28.6 8.1<br>6.5<br>12.9 35.1 22.7<br>-<br>6.4<br>13.0 31.3 29.0 **9.1**<br>6.9<br>13.0 32.3 28.9 8.3<br>6.3<br>14.0 34.5 **31.9** 8.6|-<br>-<br>-<br>-<br>-<br>-<br>-<br>-<br>-<br>-<br>-<br>-<br>-<br>-<br>-<br>-<br>-<br>-<br>-<br>-|-<br>-<br>-<br>-<br>-<br>-<br>-<br>-<br>-<br>-<br>-<br>-<br>-<br>-<br>-|
|Zero-Shot<br>ZeroCap (2022b)<br>CLIP + GPT<br>MAGIC (2022)<br>CLIP + GPT<br>EPT (2022a)<br>CLIP + GPT<br>ZS-CapCLIP_∗_(2021)<br>CLIP<br>MultiCapCLIP (**Ours**)<br>CLIP|✗<br>✗<br>✗<br>_√_<br>✗<br>✗<br>✗<br>_√_|2.6<br>11.5<br>-<br>14.6 5.5<br>5.2<br>12.5 30.7 18.3 5.7<br>-<br>-<br>-<br>-<br>-<br>3.4<br>13.0 27.6 12.2 6.2|2.3<br>12.9 30.4<br>5.8<br>5.5<br>13.3 35.4<br>7.4<br>3.0<br>14.6 27.7 11.3<br>4.0<br>15.0 31.0<br>5.0|-<br>-<br>-<br>-<br>-<br>-<br>-<br>-<br>-<br>2.8<br>25.8<br>2.0|
||✗<br>_√_|**9.7**<br>**15.8**<br>**37.6**<br>30.2<br>8.9|**13.3**<br>**19.5**<br>**43.3**<br>**15.5**|**8.4**<br>**31.2**<br>**6.2**|



Table 3: **Out-of-domain visual captioning results** . B@4, M, R-L, C, and S are short for BLEU@4, METEOR, ROUGE-L, CIDEr, and SPICE, respectively. _†_ : Training with a corpus with more than 2.3M sentences. _[∗]_ denotes our implementations. All previous works can not deal with multilingual zero-shot captioning. Our approach achieves the best results on most metrics across three datasets. 

**Baselines** Since previous works can not generate zero-shot multilingual visual captions directly, we implement a zero-shot CLIP-based model: _ZSCapCLIP_ , which is trained on text-only data with the same architecture as our MultiCapCLIP but without our proposed concept prompts and text augmentations. To observe the gap between zero-shot and fully-supervised methods, We also implement _CapCLIP_ trained on vision-caption pairs. 

**Implementations.** Following the previous works in zero-shot captioning (Tewel et al., 2022b; Su et al., 2022; Zeng et al., 2023), we adopt the CLIP (ViT-B/16 variant) (Radford et al., 2021) as our image encoder and text encoder, and adopt a randomly initialized Transformer-BASE (Vaswani et al., 2017) as our language decoder. We adopt the same vocabulary as BERT / multilingual BERT (Devlin et al., 2019) for English / non-English captioning. We use the Jieba toolkit[6] to segment Chinese sentences. We select the hyperparameter _K_ from values _{_ 4 _,_ 8 _,_ 16 _,_ 32 _}_ , _N_ from values _{_ 5 _,_ 10 _,_ 20 _}_ and _ϵ_ from values _{_ 0 _._ 01 _,_ 0 _._ 1 _,_ 1 _._ 0 _}_ according to the CIDEr performance on the validation split, and set _K_ = 16, _N_ = 5, _ϵ_ = 0 _._ 01 for all datasets and settings except that _ϵ_ = 0 _._ 1 for in-domain experiments on MS-COCO. During training, we apply label smoothing (Szegedy et al., 2016) of 0.1, use batches of 32 samples and AdamW (Loshchilov and Hutter, 2019) with L2 weight decay of 0.01 to train models for 10 epochs. We set the learning rate fixed to 1e-4 with 10% warm-up iterations when training on text-only data. During inference, we use beam search with a beam size of 3 to generate captions. 

6https://github.com/fxsjy/jieba 

## **3.2 Out-of-Domain Results** 

In this section, we evaluate the zero-shot multilingual captioning performance of our approach under out-of-domain settings. We can notice from Table 3 that our zero-shot model MultiCapCLIP achieves competitive performance on three datasets across English and Chinese. Although SGM (Honda et al., 2021) and R[2] M (Guo et al., 2020) perform better than our model on CIDEr and SPICE metrics on MS-COCO, they require the large-scale image datasets for training and use a larger training corpus (2.3M sentences) than ours (130K sentences). While the previous methods do not target nonEnglish caption generation, our MultiCapCLIP gains obvious relative improvements against the CapCLIP on VATEX Chinese captioning. The outof-domain results show that our approach is able to generate multilingual visual captions without any labeled vision-caption data, which could have the potential to promote the application of visual captioning for low-resource language applications. 

## **3.3 In-Domain Results** 

For comparisons, we further consider state-of-theart fully-supervised and large-scale pre-trained models and models under the _unpaired_ setting, i.e., both vision and text data of the target dataset are utilized for training independently, leaving their pairing annotations unused. As shown in Table 4, our approach significantly outperforms previous unpaired/zero-shot competitors by up to 4.8% BLEU@4, 3.9% ROUGE-L, and 21.5% CIDEr scores in MS-COCO English captioning. When it comes to MSR-VTT English captioning and VATEX Chinese captioning, our MultiCapCLIP surpasses ZS-CapCLIP by a large margin under the CIDEr metric, e.g., an absolute improvement of 

|**Settings**<br>**Methods**|**MS-COCO (English)**<br>B@4<br>M<br>R-L<br>C<br>S|**MSR-VTT (English)**<br>B@4<br>M<br>R-L<br>C|**VATEX (Chinese)**<br>B@4<br>R-L<br>C|
|---|---|---|---|
|Fully-<br>Supervised<br>VATEX (2019)<br>MAD+SAP (2020)<br>Oscarbase (2020)<br>OpenBook (2021)<br>ClipCap (2021)<br>CapCLIP_∗_(2021)<br>CaMEL (2022)<br>LEMONbase (2022)<br>SwinBERT (2022)<br>CLIP-DCD (2022a)<br>MV-GPT (2022)<br>GIT (2022a)|-<br>-<br>-<br>-<br>-<br>37.0<br>28.1<br>57.2<br>117.3<br>21.3<br>36.5<br>30.3<br>-<br>123.7<br>23.1<br>-<br>-<br>-<br>-<br>-<br>33.5<br>27.5<br>-<br>113.1<br>21.1<br>32.3<br>27.7<br>55.4<br>109.5<br>20.7<br>39.1<br>29.4<br>58.5<br>125.7<br>22.2<br>40.3<br>30.2<br>-<br>133.3<br>23.3<br>-<br>-<br>-<br>-<br>-<br>-<br>-<br>-<br>-<br>-<br>-<br>-<br>-<br>-<br>-<br>44.1<br>31.5<br>-<br>144.8<br>24.7|-<br>-<br>-<br>-<br>41.3<br>28.3<br>61.4<br>48.5<br>-<br>-<br>-<br>-<br>42.8<br>29.3<br>61.7<br>52.9<br>-<br>-<br>-<br>-<br>42.9<br>29.8<br>62.3<br>54.5<br>-<br>-<br>-<br>-<br>-<br>-<br>-<br>-<br>45.4<br>30.6<br>64.1<br>55.9<br>48.2<br>31.3<br>64.8<br>58.7<br>48.9<br>38.7<br>64.0<br>60.0<br>53.8<br>32.9<br>67.7<br>73.9|23.4<br>46.0<br>39.4<br>-<br>-<br>-<br>-<br>-<br>-<br>-<br>-<br>-<br>28.3_∗_<br>49.5_∗_<br>51.3_∗_<br>29.7<br>49.8<br>51.0<br>-<br>-<br>-<br>-<br>-<br>-<br>-<br>-<br>-<br>-<br>-<br>-<br>-<br>-<br>-<br>-<br>-<br>-|
|Weakly-<br>Supervised<br>UIC (2019)<br>IC-SME (2019)<br>Graph-Align (2019)<br>IGGAN (2020)<br>TSGAN (2021)<br>USGAE (2022c)<br>SCS (2022)|18.6<br>17.9<br>43.1<br>54.9<br>11.1<br>19.3<br>20.2<br>45.0<br>61.8<br>12.9<br>21.5<br>20.9<br>47.2<br>69.5<br>15.0<br>21.9<br>21.1<br>46.5<br>64.0<br>14.5<br>18.9<br>18.2<br>43.3<br>55.2<br>11.3<br>17.1<br>19.1<br>43.8<br>55.1<br>12.8<br>22.8<br>21.4<br>47.7<br>74.7<br>15.1|-<br>-<br>-<br>-<br>-<br>-<br>-<br>-<br>-<br>-<br>-<br>-<br>-<br>-<br>-<br>-<br>-<br>-<br>-<br>-<br>-<br>-<br>-<br>-<br>-<br>-<br>-<br>-|-<br>-<br>-<br>-<br>-<br>-<br>-<br>-<br>-<br>-<br>-<br>-<br>-<br>-<br>-<br>-<br>-<br>-<br>-<br>-<br>-|
|Zero-<br>Shot<br>MAGIC (2022)<br>ZS-CapCLIP_∗_(2021)<br>MultiCapCLIP (**Ours**)|12.9<br>17.4<br>39.9<br>49.3<br>11.3<br>6.1<br>15.8<br>33.0<br>27.3<br>9.3|-<br>-<br>-<br>-<br>8.6<br>19.8<br>37.3<br>11.1|-<br>-<br>-<br>21.2<br>45.0<br>31.8|
||**27.6**<br>**25.2**<br>**51.6**<br>**96.2**<br>**18.5**|**22.0**<br>**24.4**<br>**50.2**<br>**33.6**|**22.8**<br>**46.0**<br>**38.2**|



Table 4: **In-domain visual captioning results** . Fully-supervised and large-scale pre-trained models are included for comparisons. _[∗]_ denotes our implementations. Our MultiCapCLIP significantly outperforms previous zeroshot/weakly-supervised models, but still suffers from performance gaps compared with fully-supervised counterparts. 


![](raw/papers/2_LLM_NLP/LLM_paper_list/caption/acl_2023_MultiCapCLIP_Auto-Encoding_Prompts_for_Zero-Shot_Multilingual_Visual_Captioning_images/acl_2023_MultiCapCLIP_Auto-Encoding_Prompts_for_Zero-Shot_Multilingual_Visual_Captioning.pdf-0006-02.png)


Figure 2: Results of out-of-domain and in-domain experiments with respect to different ratios of training data. The margins in different ratios are shown with the polyline. Our method consistently surpasses CapCLIP, and the fewer the vision-caption pairs, the larger the margins. 

22.5% on MSR-VTT. These results prove the effectiveness of MultiCapCLIP in zero-shot multilingual visual captioning. Nevertheless, there still exists performance gaps between MultiCapCLIP trained on text-only data and existing state-of-the-art fullysupervised models trained on full vision-text data. 

## **4 Analysis** 

In this section, we conduct several analyses to better understand our approach. 

## **4.1 Semi-Supervised Visual Captioning** 

To further prove the effectiveness of our approach, we fine-tune MultiCapCLIP with partial labeled vision-caption data of downstream datasets. To this end, in Figure 2, we evaluate the performance 

of MultiCapCLIP with respect to the increasing amount of labeled data. Specifically, we randomly sample a small portion of training images/videos and use the resulting vision-caption pairs for finetuning. We repeat this process by three times and report the average performance. For a fair comparison, we also train CapCLIP (Section 3.1) with the same amount of pairs. As we can see in Figure 2, for both in-domain or a out-of-domain corpus, MultiCapCLIP consistently outperforms CapCLIP with different ratios of training data. It is worth noting that the fewer the labeled vision-caption pairs, the larger the margins. In detail, under the extremely low label setting, e.g., 0.1% of paired data on MSR-VTT (only 6 videos), our approach under the in-domain setting significantly surpasses the 

|**Setting**|**Component**<br>**_K_**<br>**Concept Type**<br>CP<br>IA<br>FA|**Out-of-Domain Setting**<br>B@4<br>M<br>R-L<br>C<br>S|**In-Domain Setting**<br>B@4<br>M<br>R-L<br>C<br>S|
|---|---|---|---|
|Base Model|-<br>-|3.4<br>13.0<br>27.6<br>12.2<br>6.2|6.1<br>15.8<br>33.0<br>27.3<br>9.3|
|(a)<br>(b)<br>(c)<br>(d)<br>(e)<br>(f)|~~_√_~~<br>16<br>Noun<br>_√_<br>-<br>-<br>_√_<br>-<br>-<br>_√_<br>_√_<br>-<br>-|5.7<br>15.2<br>32.4<br>18.1<br>8.6<br>7.4<br>15.3<br>34.1<br>23.8<br>8.5<br>5.5<br>14.3<br>32.1<br>15.3<br>7.3<br>7.4<br>15.7<br>34.4<br>23.9<br>**9.2**|7.7<br>17.6<br>35.9<br>40.6<br>12.3<br>15.1<br>20.5<br>42.2<br>57.6<br>14.5<br>26.1<br>25.2<br>51.2<br>91.5<br>18.3<br>26.6<br>25.2<br>51.3<br>92.5<br>18.4<br>27.7<br>25.2<br>51.9<br>94.6<br>18.4<br>27.5<br>25.1<br>51.6<br>95.0<br>18.3|
||~~_√_~~<br>~~_√_~~<br>~~_√_~~<br>4<br>Noun<br>_√_<br>_√_<br>_√_<br>8<br>Noun|8.2<br>15.1<br>35.9<br>27.7<br>8.1<br>8.1<br>15.6<br>35.3<br>29.3<br>8.5||
|(g) Full Model|_√_<br>_√_<br>_√_<br>16<br>Noun<br><br><br>|**9.7**<br>15.8<br>**37.6**<br>**30.2**<br>8.9|27.6<br>**25.2**<br>51.6<br>**96.2**<br>**18.5**|
|(h)<br>(i)<br>(j)|_√_<br>_√_<br>_√_<br>32<br>Noun|9.1<br>**16.2**<br>37.1<br>30.1<br>9.1|**28.4**<br>**25.2**<br>**51.9**<br>95.7<br>**18.5**<br>27.8<br>**25.2**<br>**51.9**<br>93.2<br>18.3<br>27.1<br>25.1<br>51.4<br>94.3<br>18.5|
||~~_√_~~<br>~~_√_~~<br>~~_√_~~<br>16<br>Verb<br>_√_<br>_√_<br>_√_<br>16<br>Noun + Verb|7.0<br>15.0<br>34.1<br>21.1<br>7.1<br>9.2<br>15.7<br>37.0<br>28.4<br>8.6||



Table 5: Quantitative analysis of the proposed MultiCapCLIP, which includes visual concept prompts (CP), input augmentation (IA), and feature augmentation (FA). We conduct the ablation study on the MS-COCO dataset under out-of-domain and in-domain settings. The full model denotes our proposed MultiCapCLIP. 

CapCLIP by 23.1% absolute BLEU@4 score. It further proves the effectiveness of our approach, which can relax the reliance on the vision-caption annotations. We can make use of available unpaired text-only data as a solid basis for multilingual visual captioning tasks. 

## **4.2 Quantitative Analysis** 

In this section, we analyze the contributions of each component in our approach. 

**Ablation Study** We conduct the ablation study on the out-of-domain and in-domain settings using MS-COCO dataset (Chen et al., 2015). As shown in Table 5, each component in our proposed approach can boost the performance over all metrics, verifying our arguments and the effectiveness of our approach. In particular, setting (a) shows that the introduced prompts can improve the base model with absolute gains up to 5.9% and 13.3% CIDEr scores under out-of-domain and in-domain settings, respectively. Settings (b,c) show that either input augmentation (IA) or feature augmentation (FA) respectively boost performance, indicating the importance of bridging the modality gap between the visual and textual data and in turn, boosting the robustness of the model and improving the performance of zero-shot visual captioning. Moreover, by comparing the results of (b) and (c), we observe that FA brings more improvements under the in-domain setting whereas IA is better under the out-of-domain setting. This indicates that structure noises are more suitable to bridge the modality gap between vision and text data from the same domain. From another perspective, we need a more com- 

plex feature adaptation method for out-of-domain transfer. Since the IA and FA can improve the performance from different perspectives, as shown in setting (d), combining them can lead to the most prominent improvement across all metrics. Moreover, compared with (d), our full model in the setting (g) can still gain improvements under most metrics, especially the CIDEr metric, showing that concept prompts benefit visual captioning by generating more accurate details. 

**Effect of** _K_ As shown in Table 5 (e-h), when we set the number of prompts _K_ to 16, the model substantially performs the best. For other _K_ values, when _K <_ 16, the performance is improved with an increasing _K_ due to more adequate guidance signals to the model. However, when _K >_ 16, we can observe saturated or impaired captioning performance, possibly because retrieving more prompts do not include additional useful clues and introduce irrelevant noises to the model. 

**Concept Type** Other than prompting the model with noun phrases (Section 2.3), we also consider the effect of verbs. As shown in Table 5, setting (g) surpasses settings (i) and (j) at most cases, i.e., using verb-based prompts degrades performance. We speculate the reason is that the vision-language model we used (i.e., CLIP) can recognize salient objects more accurately than human actions. 

## **4.3 Robustness Analysis: Extensions to More Non-English Languages** 

We adopt the Multi30K dataset (Elliott et al., 2016) to further evaluate in-domain performance on German and French image captioning. As shown in 


![](raw/papers/2_LLM_NLP/LLM_paper_list/caption/acl_2023_MultiCapCLIP_Auto-Encoding_Prompts_for_Zero-Shot_Multilingual_Visual_Captioning_images/acl_2023_MultiCapCLIP_Auto-Encoding_Prompts_for_Zero-Shot_Multilingual_Visual_Captioning.pdf-0008-00.png)


**----- Start of picture text -----**<br>
GT a child digging in the sand with a toy shovel GT two men are boxing each other in a ring GT a man is repairing a tire and its rim<br>CapCLIP a little boy is playing with a toy CapCLIP two men are  wrestling  each other CapCLIP a man is working on a  car<br>Ours a little boy playing with a toy in the  sand Ours a boxing match  is shown and  a man hits a punching bag . Ours a man working on a  wheel  of a  truck<br>(Zero-Shot) a sandy beach the sand a little kid a toddler (Zero-Shot) a punching bag a match a competition  the rope (Zero-Shot) a car part a truck a trailer<br>Ours 一个小男孩在(a little boy was playing with  沙滩 上拿着一个 a shovel 小铲子 on the 玩耍 beach ) Ours 拳击场上 ( In the ring , ,  两个拳击手two boxers  are fighting in a 正在进行 拳击比赛boxing match ) Ours 一个穿着蓝色衣服的男人正在(a man in blue is  repairing a tire ) 修理一个轮胎<br>(Zero-Shot) a shovel the sand a young toddler a beach (Zero-Shot) a match a referee a competition video  (Zero-Shot) a tire video a wheel a workshop<br>(a) (b) (c)<br>GT a man riding a red moped stopped at a stop light GT a baseball player swinging his bat at a baseball.<br>CapCLIP a baseball player holding a bat next to home plate<br>CapCLIP a man riding on the back of a motorcycle down a street<br>(Zero-Shot) Ours a man a scooter wearing a helmet  a biker a motorcycleis riding a mopeda helmet (Zero-Shot) Ours a baseball player hitting the ball in the fielda baseball game baseball gameplay a player<br>一名棒球运动员 挥动 棒球棒击球， 接球手接住了<br>一个 戴着头盔 的男人正在路上骑着一辆 平衡车<br>(Zero-Shot) Ours (A man a scooter with a helmet  a motorcycleis riding  a unicycle a balance car  a segway on the road) (Zero-Shot) Ours 棒球 the catcher catches the ball a baseball game(a baseball player baseball swings ) a baseball field his bat to hit the ball, and a swing<br>(d) (e)<br>**----- End of picture text -----**<br>


Figure 3: Captioning comparisons between ground-truths (GT), CapCLIP trained with full in-domain vision-text pairs, and our zero-shot MultiCapCLIP trained with out-of-domain corpora. We emphasize **accurate** and _**wrong**_ keywords and highlight reasonable and _noisy_ concepts used for prompting. Our approach can generate plausible visual descriptions in English and Chinese without the need of vision-caption pairs. 

|**Setting**|**German**<br>B@4<br>R-L<br>C|**French**<br>B@4<br>R-L<br>C|
|---|---|---|
|Supervised|20.0<br>45.7<br>55.8|7.1<br>28.0<br>54.0|
|ZS-Base Model|3.8<br>27.7<br>10.7|2.6<br>19.4<br>20.4|
|ZS-Full Model|**13.3**<br>**38.3**<br>**36.7**|**5.2**<br>**23.9**<br>**40.5**|



Table 6: In-domain performance on German and French image captioning. ZS is short for “Zero-Shot”. 

Table 6, our full model again outperforms the base model by a large margin, proving the effectiveness of concept prompts and text augmentations. 

## **4.4 Qualitative Analysis** 

In this section, we give some visualization results and examples to better understand our approach. 

**Visualization** To verify the effect of our method on representation learning, we use t-SNE (van der Maaten and Hinton, 2008) to visualize the features produced by ZS-CapCLIP and our MultiCapCLIP in Figure 4, which shows that our approach can bridge the modality gap between visual and textual inputs during training and obtain a blended distribution, leading to a more robust shift from text-to-text generation to vision-to-text generation. 

**Examples** In Figure 3, we compare our model trained with out-of-domain corpora with CapCLIP trained on full in-domain supervision. As we can see, our model can generate accurate keywords, e.g., “sand” in (a), “tire” in (c), and “helmet” in 


![](raw/papers/2_LLM_NLP/LLM_paper_list/caption/acl_2023_MultiCapCLIP_Auto-Encoding_Prompts_for_Zero-Shot_Multilingual_Visual_Captioning_images/acl_2023_MultiCapCLIP_Auto-Encoding_Prompts_for_Zero-Shot_Multilingual_Visual_Captioning.pdf-0008-09.png)


Figure 4: T-SNE visualization (van der Maaten and Hinton, 2008) of image and text embeddings produced by (a) ZS-CapCLIP and (b) our MultiCapCLIP during training. We plot the scatter diagrams for 200 imagecaption pairs. Our approach can effectively bridge the gap between the vision and text modalities. 

(d), which can be attributed to the useful clues of concept prompts. However, there exist noises in the retrieved concepts in some cases, e.g., “a punching bag” in (b), misleading the model to produce wrong details. Besides, in (e), we can observe how the training corpus affect the writing style of the model: the corpus of a video caption dataset (VATEX) makes the model focus on the temporal evolution of events, resulting in a speculative description “the catcher catches the ball”. Overall, our approach can be a solid basis for zero-shot multilingual visual captioning. It requires no vision-caption pairs but generates plausible visual descriptions. 

## **5 Related Works** 

The related works are introduced from zero-shot learning and visual captioning. 

**Zero-shot Learning** Adapting models to novel tasks with limited labeled data is an important research topic toward general intelligence (Griffiths et al., 2019). Contrastive pre-training is an effective technique to achieve this goal and has revolutionized multimodal research (Hou et al., 2021; Gan et al., 2022; Jin et al., 2022; Cheng et al., 2023a). Specifically for the vision-language field, models such as CLIP (Radford et al., 2021) and ALIGN (Jia et al., 2021) learn a shared multimodal embedding space from large-scale noisy image-text pairs, leading to an impressive zero-shot performance on tasks like image classification and vision-text retrieval (Zhang et al., 2022; Luo et al., 2022). Nevertheless, employing CLIP-like models in low-data vision-grounded text generation (i.e., visual captioning) remains challenging. 

**Visual Captioning** As a key vision-language task, visual captioning has achieved tremendous progress under the encoder-decoder framework (Xu et al., 2015) and the “pre-training and fine-tuning” paradigm. Yet, typical visual captioning methods require curated datasets of numerous images or videos paired with descriptions in a specific language, which are costly to collect. To this end, some weakly-supervised approaches are proposed (Feng et al., 2019; Guo et al., 2020; Honda et al., 2021; Ben et al., 2022). These methods require disjoint vision and text data for training and rely on a pre-trained object detector like Faster R-CNN (Ren et al., 2015) to construct weak supervision signals. However, the detectors they use are limited to a pre-defined set of categories. Recently, several works integrate CLIP with large language models (LLMs) like GPT (Radford et al., 2019; Brown et al., 2020) for zero-shot visual captioning (Tewel et al., 2022b; Su et al., 2022; Liu et al., 2022; Zeng et al., 2023). Although effective, these methods suffer from over-parameterization of large LLMs. We instead train a lightweight decoder from scratch. Besides, some concurrent works address zero-shot visual captioning by training CLIP with text-only data (Nukrai et al., 2022; Gu et al., 2022; Li et al., 2023; Yang et al., 2023). What differentiates our work from them is that we consider visual concept prompts that perverse domain visual knowledge. 

## **6 Conclusions** 

We have presented a data-efficient method dubbed MultiCapCLIP to re-purpose CLIP-like visionlanguage pre-trained models for zero-shot multilin- 

gual visual captioning. Our approach reduces the reliance on labeled vision-caption pairs of downstream datasets by auto-encoding concept prompts on text-only data. Extensive experiments on four datasets and four languages confirm the effectiveness of our approach, which can be a solid basis for visual captioning in low-data regimes and lowresource languages. 

## **Limitations** 

Although the proposed MultiCapCLIP can generate multilingual zero-shot visual captions without any labeled vision-caption training pairs. We still need the independent set of text for training/translating, which may still be difficult to collect for some lowresource languages. This might be alleviated in the future with techniques such as knowledge distillation from publicly-available pre-trained models, e.g., BERT (Devlin et al., 2019). Besides, our approach uses CLIP to measure text-text similarities for retrieving concept prompts and conducting input augmentation during training. Considering that CLIP is optimized by image-text global contrast (Radford et al., 2021) and intra-modal retrieval of such a model is not as well as its cross-modal retrieval (Jia et al., 2021), an improvement direction of our approach is using a vision-language pre-trained model that measures intra-modal and inter-modal semantic similarities well (Yang et al., 2022b). 

## **Ethics Statement** 

We conduct the experiments on public datasets, which are exclusively about natural images, videos, and captions. These datasets have been carefully pre-processed for the academic study purpose, and therefore do not contain any information that names or uniquely identifies individual people or offensive content. It is noteworthy that our approach inherits the drawback of the pre-trained backbone, i.e., CLIP, which has demonstrated that improper class design used for prompting may raise unwanted biases (Radford et al., 2021). Therefore, careful examination is needed before employing our approach in real-world scenarios to avoid prejudices. 

## **Acknowledgements** 

This paper was partially supported by NSFC (No: 62176008) and Shenzhen Science & Technology Research Program (No: GXWD2020123116580700720200814115301001). 

## **References** 

- Peter Anderson, Basura Fernando, Mark Johnson, and Stephen Gould. 2016. Spice: Semantic propositional image caption evaluation. In _European conference on computer vision_ , pages 382–398. Springer. 

- Jimmy Lei Ba, Jamie Ryan Kiros, and Geoffrey E Hinton. 2016. Layer normalization. _arXiv preprint arXiv:1607.06450_ . 

- Satanjeev Banerjee and Alon Lavie. 2005. METEOR: An automatic metric for MT evaluation with improved correlation with human judgments. In _Proceedings of the ACL Workshop on Intrinsic and Extrinsic Evaluation Measures for Machine Translation and/or Summarization_ , pages 65–72, Ann Arbor, Michigan. Association for Computational Linguistics. 

- Manuele Barraco, Matteo Stefanini, Marcella Cornia, Silvia Cascianelli, Lorenzo Baraldi, and Rita Cucchiara. 2022. Camel: Mean teacher learning for image captioning. In _2022 26th International Conference on Pattern Recognition (ICPR)_ , pages 4087– 4094. IEEE. 

- Huixia Ben, Yingwei Pan, Yehao Li, Ting Yao, Richang Hong, Meng Wang, and Tao Mei. 2022. Unpaired image captioning with semantic-constrained selflearning. _IEEE Transactions on Multimedia_ , 24:904– 916. 

- Tom Brown, Benjamin Mann, Nick Ryder, Melanie Subbiah, Jared D Kaplan, Prafulla Dhariwal, Arvind Neelakantan, Pranav Shyam, Girish Sastry, Amanda Askell, et al. 2020. Language models are few-shot learners. _Advances in neural information processing systems_ , 33:1877–1901. 

- Shan Cao, Gaoyun An, Zhenxing Zheng, and Qiuqi Ruan. 2020. Interactions guided generative adversarial network for unsupervised image captioning. _Neurocomputing_ , 417:419–431. 

- Özkan Çaylı, Burak Makav, Volkan Kılıç, and Aytu˘g Onan. 2021. Mobile application based automatic caption generation for visually impaired. In _Intelligent and Fuzzy Techniques: Smart and Innovative Solutions: Proceedings of the INFUS 2020 Conference, Istanbul, Turkey, July 21-23, 2020_ , pages 1532–1539. Springer. 

- Xinlei Chen, Hao Fang, Tsung-Yi Lin, Ramakrishna Vedantam, Saurabh Gupta, Piotr Dollár, and C Lawrence Zitnick. 2015. Microsoft coco captions: Data collection and evaluation server. _arXiv preprint arXiv:1504.00325_ . 

- Xuxin Cheng, Bowen Cao, Qichen Ye, Zhihong Zhu, Hongxiang Li, and Yuexian Zou. 2023a. Ml-lmcl: Mutual learning and large-margin contrastive learning for improving asr robustness in spoken language understanding. In _Findings of the 61st Annual Meeting of the Association for Computational Linguistics_ . 

- Xuxin Cheng, Zhihong Zhu, Hongxiang Li, Yaowei Li, and Yuexian Zou. 2023b. Ssvmr: Saliency-based selftraining for video-music retrieval. In _ICASSP 20232023 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)_ , pages 1–5. IEEE. 

- Jacob Devlin, Ming-Wei Chang, Kenton Lee, and Kristina Toutanova. 2019. BERT: Pre-training of deep bidirectional transformers for language understanding. In _Proceedings of the 2019 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, Volume 1 (Long and Short Papers)_ , pages 4171–4186, Minneapolis, Minnesota. Association for Computational Linguistics. 

- Alexey Dosovitskiy, Lucas Beyer, Alexander Kolesnikov, Dirk Weissenborn, Xiaohua Zhai, Thomas Unterthiner, Mostafa Dehghani, Matthias Minderer, Georg Heigold, Sylvain Gelly, Jakob Uszkoreit, and Neil Houlsby. 2021. An image is worth 16x16 words: Transformers for image recognition at scale. In _International Conference on Learning Representations_ . 

- Desmond Elliott, Stella Frank, Khalil Sima’an, and Lucia Specia. 2016. Multi30K: Multilingual EnglishGerman image descriptions. In _Proceedings of the 5th Workshop on Vision and Language_ , pages 70– 74, Berlin, Germany. Association for Computational Linguistics. 

- Yang Feng, Lin Ma, Wei Liu, and Jiebo Luo. 2019. Unsupervised image captioning. In _Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition_ , pages 4125–4134. 

- Zhe Gan, Linjie Li, Chunyuan Li, Lijuan Wang, Zicheng Liu, Jianfeng Gao, et al. 2022. Vision-language pretraining: Basics, recent advances, and future trends. _Foundations and Trends® in Computer Graphics and Vision_ , 14(3–4):163–352. 

- Thomas L Griffiths, Frederick Callaway, Michael B Chang, Erin Grant, Paul M Krueger, and Falk Lieder. 2019. Doing more with less: meta-reasoning and meta-learning in humans and machines. _Current Opinion in Behavioral Sciences_ , 29:24–30. 

- Jiuxiang Gu, Shafiq Joty, Jianfei Cai, Handong Zhao, Xu Yang, and Gang Wang. 2019. Unpaired image captioning via scene graph alignments. In _Proceedings of the IEEE/CVF International Conference on Computer Vision_ , pages 10323–10332. 

- Sophia Gu, Christopher Clark, and Aniruddha Kembhavi. 2022. I can’t believe there’s no images! learning visual tasks using only language data. _arXiv preprint arXiv:2211.09778_ . 

- Dan Guo, Yang Wang, Peipei Song, and Meng Wang. 2020. Recurrent relational memory network for unsupervised image captioning. In _Proceedings of the_ 

_Twenty-Ninth International Conference on International Joint Conferences on Artificial Intelligence_ , pages 920–926. 

- Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun. 2016. Deep residual learning for image recognition. In _2016 IEEE Conference on Computer Vision and Pattern Recognition_ , pages 770–778. 

- Ukyo Honda, Yoshitaka Ushiku, Atsushi Hashimoto, Taro Watanabe, and Yuji Matsumoto. 2021. Removing word-level spurious alignment between images and pseudo-captions in unsupervised image captioning. In _Proceedings of the 16th Conference of the European Chapter of the Association for Computational Linguistics: Main Volume_ , pages 3692–3702, Online. Association for Computational Linguistics. 

- Ji Hou, Benjamin Graham, Matthias Nießner, and Saining Xie. 2021. Exploring data-efficient 3d scene understanding with contrastive scene contexts. In _Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition_ , pages 15587–15597. 

- Xiaowei Hu, Zhe Gan, Jianfeng Wang, Zhengyuan Yang, Zicheng Liu, Yumao Lu, and Lijuan Wang. 2022. Scaling up vision-language pre-training for image captioning. In _Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition_ , pages 17980–17989. 

- Yiqing Huang, Jiansheng Chen, Wanli Ouyang, Weitao Wan, and Youze Xue. 2020. Image captioning with end-to-end attribute detection and subsequent attributes prediction. _IEEE Transactions on Image processing_ , 29:4013–4026. 

- Chao Jia, Yinfei Yang, Ye Xia, Yi-Ting Chen, Zarana Parekh, Hieu Pham, Quoc Le, Yun-Hsuan Sung, Zhen Li, and Tom Duerig. 2021. Scaling up visual and vision-language representation learning with noisy text supervision. In _International Conference on Machine Learning_ , pages 4904–4916. PMLR. 

- Peng Jin, Jinfa Huang, Fenglin Liu, Xian Wu, Shen Ge, Guoli Song, David Clifton, and Jie Chen. 2022. Expectation-maximization contrastive learning for compact video-and-language representations. _Advances in Neural Information Processing Systems_ , 35:30291–30306. 

- Andrej Karpathy and Li Fei-Fei. 2015. Deep visualsemantic alignments for generating image descriptions. In _Proceedings of the IEEE conference on computer vision and pattern recognition_ , pages 3128– 3137. 

- Iro Laina, Christian Rupprecht, and Nassir Navab. 2019. Towards unsupervised image captioning with shared multimodal embeddings. In _Proceedings of the IEEE/CVF International Conference on Computer Vision_ , pages 7414–7424. 

- Pan Li, Da Li, Wei Li, Shaogang Gong, Yanwei Fu, and Timothy M Hospedales. 2021. A simple feature 

augmentation for domain generalization. In _Proceedings of the IEEE/CVF International Conference on Computer Vision_ , pages 8886–8895. 

- Wei Li, Linchao Zhu, Longyin Wen, and Yi Yang. 2023. Decap: Decoding CLIP latents for zero-shot captioning via text-only training. In _International Conference on Learning Representations_ . 

- Xiujun Li, Xi Yin, Chunyuan Li, Pengchuan Zhang, Xiaowei Hu, Lei Zhang, Lijuan Wang, Houdong Hu, Li Dong, Furu Wei, et al. 2020. Oscar: Objectsemantics aligned pre-training for vision-language tasks. In _European Conference on Computer Vision_ , pages 121–137. Springer. 

- Weixin Liang, Yuhui Zhang, Yongchan Kwon, Serena Yeung, and James Zou. 2022. Mind the gap: Understanding the modality gap in multi-modal contrastive representation learning. In _Advances in Neural Information Processing Systems_ . 

- Chin-Yew Lin. 2004. ROUGE: A package for automatic evaluation of summaries. In _Text Summarization Branches Out_ , pages 74–81, Barcelona, Spain. Association for Computational Linguistics. 

- Kevin Lin, Linjie Li, Chung-Ching Lin, Faisal Ahmed, Zhe Gan, Zicheng Liu, Yumao Lu, and Lijuan Wang. 2022. Swinbert: End-to-end transformers with sparse attention for video captioning. In _Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition_ , pages 17949–17958. 

- Fenglin Liu, Xuancheng Ren, Xian Wu, Shen Ge, Wei Fan, Yuexian Zou, and Xu Sun. 2020. Prophet attention: Predicting attention with future attention. _Advances in Neural Information Processing Systems_ , 33:1865–1876. 

- Fenglin Liu, Xian Wu, Chenyu You, Shen Ge, Yuexian Zou, and Xu Sun. 2022. Aligning source visual and target language domains for unpaired video captioning. _IEEE Transactions on Pattern Analysis and Machine Intelligence_ , 44(12):9255–9268. 

- Ilya Loshchilov and Frank Hutter. 2019. Decoupled weight decay regularization. In _International Conference on Learning Representations_ . 

- Huaishao Luo, Lei Ji, Ming Zhong, Yang Chen, Wen Lei, Nan Duan, and Tianrui Li. 2022. Clip4clip: An empirical study of clip for end to end video clip retrieval and captioning. _Neurocomputing_ , 508:293– 304. 

- Ron Mokady, Amir Hertz, and Amit H Bermano. 2021. Clipcap: Clip prefix for image captioning. _arXiv preprint arXiv:2111.09734_ . 

- David Nukrai, Ron Mokady, and Amir Globerson. 2022. Text-only training for image captioning using noiseinjected clip. _arXiv preprint arXiv:2211.00575_ . 

- Aaron van den Oord, Yazhe Li, and Oriol Vinyals. 2018. Representation learning with contrastive predictive coding. _arXiv preprint arXiv:1807.03748_ . 

- Kishore Papineni, Salim Roukos, Todd Ward, and WeiJing Zhu. 2002. Bleu: a method for automatic evaluation of machine translation. In _Proceedings of the 40th Annual Meeting of the Association for Computational Linguistics_ , pages 311–318, Philadelphia, Pennsylvania, USA. Association for Computational Linguistics. 

- Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, et al. 2021. Learning transferable visual models from natural language supervision. In _International Conference on Machine Learning_ , pages 8748–8763. PMLR. 

- Alec Radford, Jeffrey Wu, Rewon Child, David Luan, Dario Amodei, Ilya Sutskever, et al. 2019. Language models are unsupervised multitask learners. _OpenAI blog_ , 1(8):9. 

- Shaoqing Ren, Kaiming He, Ross Girshick, and Jian Sun. 2015. Faster r-cnn: Towards real-time object detection with region proposal networks. _Advances in neural information processing systems_ , 28. 

- Paul Hongsuck Seo, Arsha Nagrani, Anurag Arnab, and Cordelia Schmid. 2022. End-to-end generative pretraining for multimodal video captioning. In _Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition_ , pages 17959–17968. 

- Yixuan Su, Tian Lan, Yahui Liu, Fangyu Liu, Dani Yogatama, Yan Wang, Lingpeng Kong, and Nigel Collier. 2022. Language models can see: Plugging visual controls in text generation. _arXiv preprint arXiv:2205.02655_ . 

- Christian Szegedy, Vincent Vanhoucke, Sergey Ioffe, Jon Shlens, and Zbigniew Wojna. 2016. Rethinking the inception architecture for computer vision. In _Proceedings of the IEEE conference on computer vision and pattern recognition_ , pages 2818–2826. 

- Yoad Tewel, Yoav Shalev, Roy Nadler, Idan Schwartz, and Lior Wolf. 2022a. Zero-shot video captioning with evolving pseudo-tokens. _arXiv preprint arXiv:2207.11100_ . 

- Yoad Tewel, Yoav Shalev, Idan Schwartz, and Lior Wolf. 2022b. Zerocap: Zero-shot image-to-text generation for visual-semantic arithmetic. In _Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition_ , pages 17918–17928. 

- Michael Tschannen, Olivier Bachem, and Mario Lucic. 2018. Recent advances in autoencoder-based representation learning. _arXiv preprint arXiv:1812.05069_ . 

- Laurens van der Maaten and Geoffrey Hinton. 2008. Visualizing data using t-sne. _Journal of Machine Learning Research_ , 9(86):2579–2605. 

- Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Lukasz Kaiser, and Illia Polosukhin. 2017. Attention is all 

you need. In _Advances in Neural Information Processing Systems_ , pages 5998–6008. 

- Ramakrishna Vedantam, C Lawrence Zitnick, and Devi Parikh. 2015. Cider: Consensus-based image description evaluation. In _Proc. IEEE Conf. Comput. Vis. Pattern Recog. (CVPR)_ , pages 4566–4575. 

- Pascal Vincent, Hugo Larochelle, Yoshua Bengio, and Pierre-Antoine Manzagol. 2008. Extracting and composing robust features with denoising autoencoders. In _Proceedings of the 25th international conference on Machine learning_ , pages 1096–1103. 

- Jianfeng Wang, Zhengyuan Yang, Xiaowei Hu, Linjie Li, Kevin Lin, Zhe Gan, Zicheng Liu, Ce Liu, and Lijuan Wang. 2022a. GIT: A generative image-to-text transformer for vision and language. _Transactions on Machine Learning Research_ . 

- Xin Wang, Jiawei Wu, Junkun Chen, Lei Li, YuanFang Wang, and William Yang Wang. 2019. Vatex: A large-scale, high-quality multilingual dataset for video-and-language research. In _Proceedings of the International Conference on Computer Vision_ , pages 4581–4591. 

- Yabing Wang, Jianfeng Dong, Tianxiang Liang, Minsong Zhang, Rui Cai, and Xun Wang. 2022b. Crosslingual cross-modal retrieval with noise-robust learning. In _Proceedings of the 30th ACM International Conference on Multimedia_ , pages 422–433. 

- Yasi Wang, Hongxun Yao, and Sicheng Zhao. 2016. Auto-encoder based dimensionality reduction. _Neurocomputing_ , 184:232–242. 

- Jun Xu, Tao Mei, Ting Yao, and Yong Rui. 2016. Msrvtt: A large video description dataset for bridging video and language. In _Proceedings of the IEEE conference on computer vision and pattern recognition_ . 

- Kelvin Xu, Jimmy Ba, Ryan Kiros, Kyunghyun Cho, Aaron Courville, Ruslan Salakhudinov, Rich Zemel, and Yoshua Bengio. 2015. Show, attend and tell: Neural image caption generation with visual attention. In _International conference on machine learning_ , pages 2048–2057. PMLR. 

- Bang Yang, Fenglin Liu, Yuexian Zou, Xian Wu, Yaowei Wang, and David A Clifton. 2023. Zeronlg: Aligning and autoencoding domains for zero-shot multimodal and multilingual natural language generation. _arXiv preprint arXiv:2303.06458_ . 

- Bang Yang, Tong Zhang, and Yuexian Zou. 2022a. Clip meets video captioning: Concept-aware representation learning does matter. In _Chinese Conference on Pattern Recognition and Computer Vision (PRCV)_ , pages 368–381. Springer. 

- Bang Yang, Yuexian Zou, Fenglin Liu, and Can Zhang. 2021. Non-autoregressive coarse-to-fine video captioning. In _Proceedings of the AAAI Conference on Artificial Intelligence_ , volume 35, pages 3119–3127. 

- Jinyu Yang, Jiali Duan, Son Tran, Yi Xu, Sampath Chanda, Liqun Chen, Belinda Zeng, Trishul Chilimbi, and Junzhou Huang. 2022b. Vision-language pretraining with triple contrastive learning. In _Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition_ , pages 15671–15680. 

- Xu Yang, Hanwang Zhang, and Jianfei Cai. 2022c. Auto-encoding and distilling scene graphs for image captioning. _IEEE Transactions on Pattern Analysis and Machine Intelligence_ , 44(5):2313–2327. 

- Andy Zeng, Maria Attarian, brian ichter, Krzysztof Marcin Choromanski, Adrian Wong, Stefan Welker, Federico Tombari, Aveek Purohit, Michael S Ryoo, Vikas Sindhwani, Johnny Lee, Vincent Vanhoucke, and Pete Florence. 2023. Socratic models: Composing zero-shot multimodal reasoning with language. In _International Conference on Learning Representations_ . 

- Renrui Zhang, Wei Zhang, Rongyao Fang, Peng Gao, Kunchang Li, Jifeng Dai, Yu Qiao, and Hongsheng Li. 2022. Tip-adapter: Training-free adaption of CLIP for few-shot classification. In _European Conference on Computer Vision_ , volume 13695, pages 493–510. Springer. 

- Ziqi Zhang, Zhongang Qi, Chunfeng Yuan, Ying Shan, Bing Li, Ying Deng, and Weiming Hu. 2021. Openbook video captioning with retrieve-copy-generate network. In _Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition_ , pages 9837–9846. 

- Yucheng Zhou, Wei Tao, and Wenqiang Zhang. 2021. Triple sequence generative adversarial nets for unsupervised image captioning. In _ICASSP 2021-2021 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)_ , pages 7598–7602. IEEE. 

