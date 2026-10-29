# **SyCoCa: Symmetrizing Contrastive Captioners with Attentive Masking for Multimodal Alignment** 

Ziping Ma,[*] Furong Xu, Jian Liu, Ming Yang, Qingpei Guo[†] Ant Group 

maziping.im@gmail.com, booyoungxu.xfr@antgroup.com, rex.lj@antgroup.com 

m.yang@antgroup.com, qingpei.gqp@antgroup.com 

## **Abstract** 

_Multimodal alignment between language and vision is the fundamental topic in current vision-language model research. Contrastive Captioners (CoCa), as a representative method, integrates Contrastive Language-Image Pretraining (CLIP) and Image Caption (IC) into a unified framework, resulting in impressive results. CLIP imposes a bidirectional constraints on global representation of entire images and sentences. Although IC conducts an unidirectional image-to-text generation on local representation, it lacks any constraint on local text-to-image reconstruction, which limits the ability to understand images at a fine-grained level when aligned with texts. To achieve multimodal alignment from both global and local perspectives, this paper proposes Symmetrizing Contrastive Captioners (SyCoCa), which introduces bidirectional interactions on images and texts across the global and local representation levels. Specifically, we expand a Text-Guided Masked Image Modeling (TG-MIM) head based on ITC and IC heads. The improved SyCoCa can further leverage textual cues to reconstruct contextual images and visual cues to predict textual contents. When implementing bidirectional local interactions, the local contents of images tend to be cluttered or unrelated to their textual descriptions. Thus, we employ an attentive masking strategy to select effective image patches for interaction. Extensive experiments on five vision-language tasks, including image-text retrieval, image-captioning, visual question answering, and zero-shot/finetuned image classification, validate the effectiveness of our proposed method._ 

## **1. Introduction** 

In recent years, the dramatic progress of multimodal alignment between vision and language has reshaped computer 

> *Work done during an internship with Ant Group. 

> †Corresponding author. 

vision research in some sense. The availability of largescale datasets and powerful computational resources has led to many seminal works and impressive results in this field. Most methods use contrastive objective to constrain global representation between modalities as shown in Figure 1 (a), such as CLIP [41, 58], ALBEF [29], mPLUG [28, 57]. BEiT-v3 [53] treats image as a foreign language and uses a mask-then-predict strategy for pre-training. This pretraining method requires separate fine-tuning for each downstream task and the pretrained model cannot be directly used across tasks. CoCa [61] combines Image-Text Contrastive (ITC) and Image Caption (IC), resulting in a pretrained model that can be directly used for both retrieval and IC tasks. It is a promising architecture and works well on multiple vision-language tasks. 

However, in terms of local interaction between image patches and text tokens, the IC head in CoCa only utilizes the visual cues to generate textual descriptions, yet disregarding the visual context reconstruction from textual cues. From vision pre-training works, such as SimMIM [56] and MAE [19], we learn that the image reconstruction can learn a strong content representation. Therefore, in the multimodal scenario, when text cue is introduced into the image reconstruction task and local interaction is performed, the representation of texts and images can be unified into one space, thereby further enhancing multimodal alignment. 

In this paper, we propose a novel framework called Symmetrizing Contrastive Captioners (SyCoCa) that incorporates both local image-to-text generation and text-to-image reconstruction in addition to global constractive objective. In addition to ITC head and IC head, we introduce a textguided masked image modeling (TG-MIM) head. The difference between CoCa and our method is illustrated in Figure 1. CoCa only achieved undirectional interaction between image and text as shown in Figure 1 (b). Further, in Figure 1 (c), our TG-MIM introduces text for image reconstruction, which enhances the fine-grained representation ability of images. In this way, SyCoCa has the bidirectional global and local interactions between modalities. 

1 


![](raw/papers/2_LLM_NLP/LLM_paper_list/caption/icml_2024_SyCoCa_Symmetrizing_Contrastive_Captioners_with_Attentive_Masking_for_Multimodal_Alignment_images/icml_2024_SyCoCa_Symmetrizing_Contrastive_Captioners_with_Attentive_Masking_for_Multimodal_Alignment.pdf-0002-00.png)


**----- Start of picture text -----**<br>
Text-guided<br>Captioning Captioning<br>modeling<br>Text Decoder Image Decoder Text Decoder<br>Contrasting Contrasting Contrasting<br>Image Encoder Text Encoder Image Encoder Text Encoder Image Encoder Text Encoder<br>image text image text image text<br>(a) Cross-modal Alignment (CLIP) (b) Cross-modal Alignment & (c) Cross-modal Alignment &<br>Unidirectional Interaction (CoCa) Bidirectional Interaction (SyCoCa)<br>**----- End of picture text -----**<br>


Figure 1. Comparison of the pipelines in vision-language pretraining frameworks. (a) CLIP: aligning global features across modalities through contrastive learning. (b) CoCa: introducing image captioning to construct unidirectional fine-grained interaction. (c) Our SyCoCa: bidirectional local interation with attentive masking to enhance comprehensive cross-modal understanding. 

During the actual multimodal alignment process, although an image is worth one thousand words, people seldom write one thousand words to describe the image content. Instead, text descriptions or image captions are often highly abstract just focusing on the main character, object, or event in an image. For example, the caption of a picture about the family dinner of Thanksgiving would rarely delineate the furniture in the room in details. Moreover, even short text descriptions may imply rich and abundant image contents. _e.g.,_ . _a football match_ may well imply a vivid scene of a grass field, multiple palyers, and croweded audience, etc. Therefore, it is necessary to choose appropriate local regions and representation for alignment. To select effective image patches for text, we employ an attentive masking strategy. Specifically, we calculate the similarity between image tokens and text tokens to determine the relevance of image patches. For the IC task, the most pertinent image patches are selected to aid in generating textual descriptions by considering their semantic similarity with text descriptions. In contrast, we choose the least relevant image patches with the text on the TG-MIM task, aiming to leverage the text tokens to assist in recovering image content. Extensive experiments demonstrate the effectiveness of our proposed method on five downstream tasks. 

In summary, the main contributions of our work are listed as follows: 

- We first propose a symmetrizing contrastive captioners for multimodal alignment, which improves the understanding between images and sentences from a global and local perspective. 

- To promote bidirectional local interaction, we adopt an attentive masking strategy to select appropriate image patches for IC and TG-MIM heads respectively. 

- Thorough experiments validate that our proposed _SyCoCa_ outperforms CoCa on several downstream tasks, _e.g._ image-text retrieval, image-captioning, visual question answering and image classification. For example, we 

obtained +5.1%/3.7% in mTR/mIR on Flicker-30k compared to CoCa in image-text retrieval tasks. 

## **2. Related Work** 

**Vision-Language Pretraining.** In recent years, there has been tremendous progress in multimodal alignment, especially between vision and language. Extensive researchers have dedicated their efforts to exploring vision-language pretraining. Early works [8, 49, 67] prefixed a pretrained object detection modules to extract visual representations, which were then aligned with textual representations to achieve multimodal alignment. Later efforts focused on training multimodal transformers from scratch, such as ViLT [25] and VLMo [3]. Pre-training of a foundation model on gigantic data with tremendous computation led to the breakthrough of multimodal alignment of images and texts. CLIP [41] and ALIGN [23] trained a dual-encoder on large-scale noisy image-text pairs using contrastive loss, obtaining generic image and text representations for crossmodal alignment and zero-shot image classification tasks. Florence [63] used a unified contrastive learning for various vision and image-text benchmarks. BLIP [31] trained a Q-former to align a frozen vision encoder and language encoder. BEiT-v3 [53] treated image as a foreign language, and used mask-then-predict strategy for pre-training. To enhance local interaction of images and texts, CoCa [61] introduced a decoder for image caption based on CLIP. Our SyCoCa further achieves bidirectional local interactions and enhances fine-grained alignment capabilities between modalities. 

**Masked Image Modeling.** Beyond contrastive learning, masked image modeling (MIM) [7, 13, 13, 39, 50] emerges as another promising technique in vision pre-training. Recently, iGPT [7], ViT [15] and BEiT [2] recalled this learning approach on the modern vision transformers, which show great potential in representation learning by introduc- 

2 

ing special designs, such as clustering on pixels [7], prediction of mean color [15], and tokenization via an additional dVAE network with a block-wise masking strategy [2]. SimMIM [56] predicted RGB values of raw pixels by direct regression. MAE [19] used an asymmetric encoderdecoder architecture, where the encoder operates only on the visible subset of patches, then a lightweight decoder reconstructs the original image from the latent representation and mask tokens. 

Inspired by MAE, we propose a TG-MIM head to improve fine-grained multimodal alignment. The major differences of TG-MIM against MAE include: 1) the text cues are leveraged in TG-MIM to guide image reconstruction; 2) an attentive masking scheme is used in TG-MIM to select appropriate patches; and 3) TG-MIM targets on improving multi-modal alignment, instead of learning vision representation as in MAE. 

## **3. Method** 

We are interested in leveraging bidirectional cross-modal interaction to learn fine-grained visual and textual representations within an aligned latent space. Contrastive Captioner (CoCa) proposes to incorporate an image captioning task alongside the contrastive task, which learns finegrained representation and establishes a stronger alignment between modalities than CLIP. It is worth noting that the modality interaction in CoCa is unidirectional and asymmetric, focusing solely on genetating text from images. 

To enhance vision-language alignment, we introduce a novel framework called SyCoCa, which expands the existing CoCa model by incorporating bidirectional local interaction with attentive masking. Our framework comprises of three objectives: (i) image-text contrasting (ITC), (ii) image captioning (IC), and (iii) text-guided masked image modeling (TG-MIM). Figure 2 shows the overall framework of SyCoCa. 

The goal of SyCoCa is to further explore the potential of cross-modal interactive prediction tasks. Building upon CoCa, SyCoCa incorporates TG-MIM to establish predictions from text to image, aiming to compensate for the inherent asymmetry in CoCa. This establishes bidirectional prediction between modalities, promoting finegrained vision-language understanding. Furthermore, we design a novel attentive masking procedure that enables the bidirectional cross-modal interactions to focus on different regions of images, guided by the accompanying texts. 

## **3.1. Model Architecture** 

As shown in Figure 2, the overall model architecture consists of four key components: the image encoder, text encoder, (text-to-)image decoder, and (image-to-)text decoder. Here, we provide a detailed explanation to each ones. 

**Image Encoder.** We employ vision transformer as the image encoder to model an input image. The image encoder takes image patches as input and encodes them into a sequence of embeddings _{v[cls] , v_[1] _, ..., v[P] }_ where each embedding corresponds to a specific image patch. Additionally, an extra [ `CLS` ] embedding is included to provide a global representation of the image. 

**Text Encoder.** As the text encoder, we adopt a causal masked transformer encoder to model text inputs. This encoder takes the simple BPE tokenized input text and converts it into a sequence vector represented as _{w_[1] _, ..., w[S] , w[cls] }_ , in which the embedding of the [ `CLS` ] token summarizes the global text feature. The purpose of adopting causal attention mask is to prevent any potential information leakage from future tokens to past tokens during the text encoding process. 

**Image / Text Decoder.** To further capture the interaction between image-text pairs, we use an image decoder and a text decoder. Each decoder utilizes cross-attention transformer modules to deeply fuse image and text information, enabling bidirectional local interaction. The crossmodality multi-head attention module uses the representations of one modality as the query and the other modality’s representations as the key and value. More specifically, the image decoder predicts the pixel values of masked image patches, leveraging cross-attention to incorporate relevant text. Similarly, the text decoder predicts the logits of the next token, utilizing cross-attention to integrate relevant image information. This bidirectional fusion mechanism ensures a comprehensive mutual understanding of the imagetext pairs. 

## **3.2. Training objectives** 

We start from an image-text pair dataset consisting of pairs ( _Ii, Ti_ ), where _Ii_ represents an image and _Ti_ is the corresponding text caption. The image encoder _EI_ and text encoder _ET_ are responsible for encoding the image and text, respectively, yielding embeddings: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/caption/icml_2024_SyCoCa_Symmetrizing_Contrastive_Captioners_with_Attentive_Masking_for_Multimodal_Alignment_images/icml_2024_SyCoCa_Symmetrizing_Contrastive_Captioners_with_Attentive_Masking_for_Multimodal_Alignment.pdf-0003-13.png)


_P_ refers to the number of patches, and _S_ is the length of tokenized text sequence. Here, we present the applied training objectives in details. 

**Image Text Contrasting (ITC).** In the training process, we consider a batch of _N_ image-text pairs _{Ii, Ti}[N] i_ =1[and] their corresponding global representations _{vi[cls][, w] i[cls][}][N] i_ =1[.] It is assumed that these representations have been normalized to have a unit _ℓ_ 2 norm. The contrasive objective is 

3 


![](raw/papers/2_LLM_NLP/LLM_paper_list/caption/icml_2024_SyCoCa_Symmetrizing_Contrastive_Captioners_with_Attentive_Masking_for_Multimodal_Alignment_images/icml_2024_SyCoCa_Symmetrizing_Contrastive_Captioners_with_Attentive_Masking_for_Multimodal_Alignment.pdf-0004-00.png)


**----- Start of picture text -----**<br>
/ TG-MIM IC<br>high / low-attn mask a dog is running on the grass [/s]<br>Attentive  Text-to-Image Image-to-Text<br>Masking Decoder Decoder<br>ITC Q K, V Q K, V<br>text CLS image CLS text embeddings<br>Causal Text Encoder Image Encoder<br>[s] a dog is running on the grass [CLS]<br>“A dog is running on the grass.”<br>text image masked images<br>**----- End of picture text -----**<br>


Figure 2. The detailed illustration of our proposed method. The framework of our method consist of four modules: an image encoder, a (causal) text encoder, a (text-to-)image decoder, a (image-to-)text decoder. Our method focuses on three pretraining objectives: image-text contrasting (ITC), text-guided masked image modeling (TG-MIM) and image captioning (IC). 

designed to align the image and text representations: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/caption/icml_2024_SyCoCa_Symmetrizing_Contrastive_Captioners_with_Attentive_Masking_for_Multimodal_Alignment_images/icml_2024_SyCoCa_Symmetrizing_Contrastive_Captioners_with_Attentive_Masking_for_Multimodal_Alignment.pdf-0004-03.png)


where _⟨·, ·⟩_ refers to the inner product, and _τ_ is the temperature to scale the logits. The contrastive objective that pulls the representations of paired image-text close together while pushing apart unmatched pairs, promoting alignment in a shared semantic space. 

**Image Captioning.** Image captioning objective requires the model to generate tokenized texts _Ti_ with precise details in an autoregressive manner, compared with ITC treating the inputs as single entities. We use the image encoder _EI_ to encode image representations and train the text decoder _DT_ to maximize the conditional likelihood of the text _Ti_ by utilizing forward autoregressive factorization: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/caption/icml_2024_SyCoCa_Symmetrizing_Contrastive_Captioners_with_Attentive_Masking_for_Multimodal_Alignment_images/icml_2024_SyCoCa_Symmetrizing_Contrastive_Captioners_with_Attentive_Masking_for_Multimodal_Alignment.pdf-0004-06.png)


The text decoder is trained using parallel computation for enhanced learning efficiency. This training objective enables the model to learn fine-grained representations with strong alignment through cross-modal prediction, fostering the acquisition of joint-modality representations useful for various multimodal understanding and generation tasks. 

Nonetheless, this interaction is unidirectional and asymmetrical, primarily focusing on understanding visual elements in language, without exploiting the comprehension of textual elements in the visual context. 

**Text-Guided Masked Image Modeling.** To address the inherent imbalance in cross-modal interaction within the image captioning task, we design a novel training objective called text-guided masked image modeling. Our SyCoCa framework utilizes the image decoder, denoted as _DI_ , to reconstruct the masked image by predicting the pixel values for each masked patch. The output linear projection and reshape process is similar to that of the MAE. For a given image _I_ = _{p_[1] _, ..., p[P] }_ consisting of _P_ patches, we employ patch-wise masking using a masking map _M_ = _{m_[1] _, ..., m[P] }_ , where _m[i] ∈{_ 0 _,_ 1 _}_ indicates whether the patch is masked. The TG-MIM loss is computed as the L1 loss between the masked patches and their reconstructed counterparts in the pixel space: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/caption/icml_2024_SyCoCa_Symmetrizing_Contrastive_Captioners_with_Attentive_Masking_for_Multimodal_Alignment_images/icml_2024_SyCoCa_Symmetrizing_Contrastive_Captioners_with_Attentive_Masking_for_Multimodal_Alignment.pdf-0004-10.png)


where _p_ ˆ _[i]_ refers to the reconstructed patches. Note that the TG-MIM objective, proposed in this paper, distinguishes itself from the image-pretraining objective MIM by dedicated efforts on the guidance offered by the cross-attended text representation. The text-guided modeling focuses on the interaction between visual and textual representations, with the goal of enhancing the fine-grained understanding of textual elements within the visual context through reconstruction process. The introduction of this objective com- 

4 

plements the uni-directional prediction of image captioning. 

## **3.3. Attentive Masking** 

Intuitively, the reconstruction of image regions that reveal the same semantics or show highly related contents to the text shall be a more effective and robust cross-modal interaction, rather than reconstructing those irrelevant image regions. Therefore, we suggest generating masking maps that eliminate image patches that are relevant to language captions, and subsequently reconstruct these patches based on the text captions. 

A straightforward idea to evalaute the correlation is by measuring the similarity between each patch embedding _v[i]_ and the global text embedding _w[cls]_ . However, _w[cls]_ provides a coarse-grained summary of the entire sentence. To mask the vision elements mentioned in the caption as accurately as possible, we propose to calculate the token-wise maximum similarity between image and text embeddings. This method is effective in modeling fine-grained crossmodal similarities in a previous study [60]. Formally, for the _i_ -th embedding _v[i]_ of an image, we calculate its similarities with all text embeddings _{w_[1] _, ..., w[S] }_ and select the largest similarity 


![](raw/papers/2_LLM_NLP/LLM_paper_list/caption/icml_2024_SyCoCa_Symmetrizing_Contrastive_Captioners_with_Attentive_Masking_for_Multimodal_Alignment_images/icml_2024_SyCoCa_Symmetrizing_Contrastive_Captioners_with_Attentive_Masking_for_Multimodal_Alignment.pdf-0005-04.png)


to represent its correlation with the text. 

Image patches to be masked are selected based on their scores _{s_[1] _, ..., s[P] }_ . To ensure stable training in TG-MIM, a fixed ratio _rh_ of top scoring patches (represented by the _high attn mask_ in Figure 2) in the input image are masked. Conversely, in the caption task, the patches with the lowest scores (represented by the _low attn mask_ in Figure 2) in the input image are masked at a fixed ratio _rl_ . The purpose is to enhance cross-modal interaction by encouraging the model to prioritize visually salient regions that play a significant role in understanding and representing the image for caption generation. 

## **3.4. Overall objective** 

Finally, we pretrain SyCoCa with all these losses combined: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/caption/icml_2024_SyCoCa_Symmetrizing_Contrastive_Captioners_with_Attentive_Masking_for_Multimodal_Alignment_images/icml_2024_SyCoCa_Symmetrizing_Contrastive_Captioners_with_Attentive_Masking_for_Multimodal_Alignment.pdf-0005-09.png)


where _λIC_ and _λT M_ are the hyper-parameters weighting between IC and TG-MIM. All the modules of SyCoCa are trained from scratch. 

## **4. Experiments** 

To demonstrate the effectiveness of our proposed SyCoCa, we conduct extensive experiments on 5 downstream tasks. Initially, we present the experimental setup, including model architecture, pretraining datasets, downstream tasks, 

and implementation details. Subsequently, we compare SyCoCa with the baseline model, CoCa, on image-text retrieval, image classification, image captioning and visual question answering tasks. Finally, we perform a series of ablation studies to further analyze and evaluate our model. 

## **4.1. Experimental Setup** 

**Model Architecture.** To ensure a fair comparison, we conduct our experiment using the open-source implementation of CoCa[1] . Both SyCoCa and the baseline model utilize the same CoCa-Base configuration for the image encoder, text encoder, and text decoder. However, in SyCoCa, we introduce a new image decoder that shares the same architecture as the text decoder yet with minor modifications. Specifically, we replace the token-prediction head with a pixel-prediction head. All model parameters are initialized using a Gaussian distribution with a mean of 0 and a standard deviation of 0.01, allowing the training process to start from scratch. 

**Pretraining Data.** We use the Conceptual Captions 12M [6] (CC12M) dataset with 12 million image-caption pairs, as the multi-modal pretraining data for all models. Although this dataset is smaller in scale compared to the large custom datasets employed by the state-of-the-art pretraining models (such as 400M pairs in CLIP [41] and 3B pairs in CoCa [61]), it is a good match to the computation resources available to us. Moreover, CC12M has been widely adopted for benchmark evaluations in various studies on vision-language pretraining [17, 32, 36, 47, 64]. **Downstream Tasks.** We evaluate SyCoCa on 3 downstream vision-language tasks and 2 classification tasks, including image-text retrieval, image captioning, visual question answering, zero-shot image classification, and finetuned image classification. 

**Implementation Details.** We conduct our model training on two machines, each equipped with 8 NVIDIA A100 GPUs, for a total of 20 epochs. The batch size during training is set to 2048, and the resolution of pretraining images is set to 224 _×_ 224. We use the AdamW optimizer [35] with an initial learning rate of 1 _e −_ 4. The learning rate schedule follows a cosine decay, including a warm-up period of 5000 steps. In terms of hyperparameters, we simply set _λIC_ = 2 following CoCa and _λT M_ = 1. The masking ratios _rh_ and _rl_ are both empirically set to 50%. 

## **4.2. Results on Downstream Vision-Language Tasks** 

We compare the performance of SyCoCa and CoCa on downstream vision-language tasks, including image-text retrieval, image captioning, and vision question answering. 

**Image-Text Retrieval.** In this task, the models are required to find the sample that best matches the input 

> 1https://github.com/mlfoundations/open_clip 

5 

||Flickr30K<br>Image_→_Text<br>Text_→_Image<br>R@1<br>R@5<br>R@10<br>R@1<br>R@5<br>R@10|Flickr30K<br>Image_→_Text<br>Text_→_Image<br>R@1<br>R@5<br>R@10<br>R@1<br>R@5<br>R@10|MSCOCO<br>Image_→_Text<br>Text_→_Image<br>R@1<br>R@5<br>R@10<br>R@1<br>R@5<br>R@10|MSCOCO<br>Image_→_Text<br>Text_→_Image<br>R@1<br>R@5<br>R@10<br>R@1<br>R@5<br>R@10|
|---|---|---|---|---|
||Image_→_Text||Image_→_Text||
||R@1<br>R@5<br>R@10||R@1<br>R@5<br>R@10||
|CoCa<br>Ours|44.2<br>72.4<br>81.7<br>**46.6**<br>**76.3**<br>**83.6**|31.3<br>59.7<br>70.8<br>**35.9**<br>**64.6**<br>**75.6**|16.3<br>37.1<br>48.9<br>**18.7**<br>**41.0**<br>**53.6**|15.3<br>35.8<br>47.0<br>**17.2**<br>**39.7**<br>**51.6**|
|%Gains|+5.4<br>+5.4<br>+2.3|+14.7<br>+8.2<br>+6.8|+14.7<br>+10.5<br>+9.6|+12.4<br>+10.9<br>+9.8|



Table 1. Zero-shot image-text retrieval evaluation results on Flickr30K and MSCOCO dataset. 

||Image Captioning<br>MSCOCO<br>NoCaps-Test<br>B@4<br>M<br>C<br>S<br>B@4<br>M<br>C<br>S|Image Captioning<br>MSCOCO<br>NoCaps-Test<br>B@4<br>M<br>C<br>S<br>B@4<br>M<br>C<br>S|vision question answering<br>VQA<br>val<br>test-dev<br>test|
|---|---|---|---|
||MSCOCO<br>B@4<br>M<br>C<br>S|||
|CoCa<br>Ours|21.3<br>22.6<br>71.1<br>16.2<br>**22.4**<br>**23.6**<br>**75.6**<br>**16.8**|12.7<br>22.4<br>53.4<br>10.4<br>**12.9**<br>**22.8**<br>**54.8**<br>**10.5**|43.0<br>39.1<br>39.6<br>**46.9**<br>**42.6**<br>**42.9**|
|%Gains|+5.2<br>+4.4<br>+6.3<br>+3.7|+1.6<br>+1.8<br>+2.6<br>+1.0|+9.1<br>+9.0<br>+8.3|



Table 2. Comparison with CoCa on image captioning (MSCOCO, NoCaps) and vision question answering (VQA). B: BLEU@4. M: METEOR. C: CIDEr. S: SPICE. 

across modalities without finetuning. We conduct evaluation on standard image-text retrieval datasets, namely Flickr30K [40] and MSCOCO [34], and report the results in 1. We can find that SyCoCa consistently outperforms CoCa showcasing the gains in the range of 5%-15% for R@1. 

**Image Captioning.** In this task, the models are required to generate textual descriptions for input images. We finetune both SyCoCa and CoCa using cross-entropy loss on the MSCOCO Captioning [34] dataset. Subsequently, we report the BLEU@4, METEOR, CIDEr, and SPICE scores on the Karparthy test split of MSCOCO, as well as the test split of the NoCaps [1] dataset. The results shown in Table 2 demonstrate that SyCoCa outperforms CoCa across all metrics. Specifically, SyCoCa improves by 4%-6% on MSCOCO and 1%-3% on NoCaps compared to CoCa. 

**Vision Question Answering.** In this task, the models are required to predict an answer based on both an image and a question. To apply SyCoCa and CoCa to this task, we modify the text encoder to a multi-modal decoder, which allows for the fusion of image representation and question input. We adapt the text decoder to generate answers in an autoregressive manner, derived from the output of the multi-modal decoder. To finetune the models, we utilize the VQA [18] dataset. During inference, we constrain the decoder to generate answers only from a set of 3,128 candidate answers to make a fair comparison. The results depicted in Table 2 clearly indicate that SyCoCa surpasses CoCa in all cases, where SyCoCa achieves remarkable improvements of 8%9% on the validation, test-dev, and test splits of VQA. 

Our evaluation on downstream vision-language tasks has 

confirmed the advancements achieved by SyCoCa in terms of multi-modal alignment and cross-modal understanding. On the one hand, our zero-shot retrieval experiments have demonstrated the importance of bidirectional mutual interaction in enhancing the performance of CoCa in terms of modality alignment. On the other hand, the experimental results presented in Table 2 highlights the critical role played by the bidirectional interaction mechanism in fostering mutual understanding and capturing fine-grained elements across different modalities. 

## **4.3. Results on Image Classification Tasks** 

We conduct a comparison between CoCa and SyCoCa regarding their classification performance on both zero-shot and fine-tuned image classification tasks. For zero-shot image classification, we evaluate the models on 8 coarsegrained image classification datasets that include common categories like airplanes and dogs. The results are presented in Table 3. SyCoCa outperforms CoCa in 7 out of 8 cases, resulting in an average accuracy improvement of 2.5%. These findings highlight the effectiveness of bidirectional understanding in bridging the gap between visual and textual representations. 

In case of the fine-tuned image classification tasks, we use 7 fine-grained image classification datasets that encompass subcategories within a specific categorie, such as different breeds of dogs [24] like Border Collie and Golden Retriever. During the fine-tuning process, we employ linear probing to gauge the image encoders’ capability to discern intricate details within images. We report our results in Table 4. The results indicate that SyCoCa outperforms CoCa 

6 


![](raw/papers/2_LLM_NLP/LLM_paper_list/caption/icml_2024_SyCoCa_Symmetrizing_Contrastive_Captioners_with_Attentive_Masking_for_Multimodal_Alignment_images/icml_2024_SyCoCa_Symmetrizing_Contrastive_Captioners_with_Attentive_Masking_for_Multimodal_Alignment.pdf-0007-00.png)


**----- Start of picture text -----**<br>
A few pieces of luggage sitting on top of<br>A man is holding a cell phone in front of a mountain. a chair in an airport. The hull of a boat that is producing a wake<br>Image 1 CoCa SyCoCa Image 2 CoCa SyCoCa Image 3 CoCa SyCoCa<br>A wooden park bench with a remote control on top of it. That small plane looks like it is ready for takeoff. A bird that is playing in the snow.<br>Image 4 CoCa SyCoCa Image 5 CoCa SyCoCa Image 6 CoCa SyCoCa<br>**----- End of picture text -----**<br>


Figure 3. Qualitative analysis of the proposed SyCoCa. We visualize the attention localization map of the first convolution layer in image encoder by the toolkit Grad-CAM. 

||IN-1K [12]<br>IN-V2 [42]<br>IN-A [21]<br>IN-R [20]<br>C-10 [26]<br>C-100 [26]<br>STL-10 [11]<br>Caltech [16]|Avg.|
|---|---|---|
|CoCa<br>Ours|26.1<br>22.4<br>6.5<br>31.7<br>**65.9**<br>29.7<br>88.1<br>64.5<br>**27.8**<br>**23.9**<br>**7.0**<br>**32.3**<br>62.0<br>**29.8**<br>**89.2**<br>**65.6**|41.9<br>**42.2**|
|%Gains|+6.5<br>+6.7<br>+7.7<br>+1.9<br>-5.9<br>+0.3<br>+1.2<br>+1.7|+2.5|



Table 3. Zero-shot image classification evaluation results on 8 coarse-grained datasets: ImageNet, ImageNet-V2, ImageNet-A, ImageNetR, CIFAR-10, CIFAR-100, STL-10 and Caltech101. 

||DTD [10]<br>Dogs [24]<br>CUB [51]<br>Pets [38]<br>Flowers[37]<br>MNIST [27]<br>Food [4]|Avg.|
|---|---|---|
|CoCa<br>Ours|58.1<br>49.0<br>40.6<br>**72.7**<br>69.1<br>93.3<br>69.1<br>**60.5**<br>**51.2**<br>**41.3**<br>70.0<br>**70.6**<br>**94.5**<br>**71.1**|64.6<br>**65.6**|
|%Gains|+4.1<br>+4.5<br>+1.7<br>-3.7<br>+2.2<br>+1.3<br>+2.9|+1.9|



Table 4. Fine-tuned image classification evaluation results on 7 fine-grained datasets: DTD, Oxford-IIIT Pet, Stanford Dogs, CUB-200, Flowers102, MNIST, Food101. Both models are transfered using linear probing. 

in 6 out of 7 datasets, resulting in an average performance improvement of 2%. This demonstrates the effectiveness of incorporating bidirectional prediction to enhance the understanding and differentiation of fine-grained elements within the visual representation domain. 

## **4.4. Qualitive Analysis of SyCoCa** 

To obtain a intuitive comprehension of the advantages of SyCoCa, we use Grad-CAM [44], a commonly used ”visual explanation” toolkit, to generate attention location maps for the first convolutional layer in the image encoder. As shown in Figure 3, compared with CoCa, the improved version SyCoCa can capture fine-grained visual elements related to informative words in text. For instance, in the case of Image 1, SyCoCa exhibits a focused attention towards regions that are pertinent to the words ” _holding_ ”, ” _cell phone_ ” and ” _mountain_ ” whereas CoCa neglects these essential details. In contrast, in the case of Image 2, SyCoCa successfully identifies the objects ” _luggage_ ” and ” _chair_ ”, as well as the scenario ” _airport_ ”, whereas CoCa falls short in recognizing 

these elements. This enhanced performance is attributed to the fine-grained cross-modal understanding ability enabled by bidirectional local interaction and attentive masking, which proves advantageous for downstream tasks such as image captioning and vision language answering. 

## **4.5. Ablation Study** 

We conduct a series of experiments to evalute the impact of training objectives and hyper-parameter settings of SyCoCa. Due to limited computing resources, we train the models on Conceptual Captions 3M (CC3M) [45], which is a small dataset consisting of filtered image-text pairs. This dataset has been widely used for evaluations of visionlanguage pretraining [14, 54, 59, 68]. 

**objective Analysis.** To analyze the impact of individual training objectives in SyCoCa, we conduct a series of experiments. We compare the performance of different variants on image-text retrieval and classification tasks, as presented in Table 5. We can observe that: 

• By incorporating bidirectional cross-modal interaction 

7 

|ITC<br>IC<br>MIM<br>TG-MIM<br>RM<br>AM|ITC<br>IC<br>MIM<br>TG-MIM<br>RM<br>AM|zero-shot retrieval<br>Flickr30K<br>MSCOCO<br>mTR<br>mIR<br>mTR<br>mIR|zero-shot retrieval<br>Flickr30K<br>MSCOCO<br>mTR<br>mIR<br>mTR<br>mIR|zero-shot classifcation<br>IN-1K<br>C-10<br>STL-10<br>Caltech<br>Avg.|zero-shot classifcation<br>IN-1K<br>C-10<br>STL-10<br>Caltech<br>Avg.|
|---|---|---|---|---|---|
|||Flickr30K||||
|||mTR<br>mIR||||
||✓|32.0<br>24.4|13.5<br>14.8|10.7<br>41.0<br>68.4<br>36.7|30.2|
|CoCa|✓<br>✓|37.5<br>28.7|16.1<br>16.0|10.4<br>39.6<br>69.2<br>37.8|31.9|
||✓<br>✓<br>✓<br>✓<br>✓<br>✓<br>✓<br>✓<br>✓<br>✓<br>✓<br>✓<br>✓<br>✓<br>✓|41.1<br>31.4<br>36.0<br>27.1<br>37.7<br>27.8<br>35.8<br>26.4|17.5<br>18.3<br>15.8<br>16.3<br>15.2<br>16.2<br>14.4<br>15.2|11.4<br>40.0<br>71.7<br>39.1<br>11.8<br>46.6<br>71.5<br>38.5<br>11.6<br>43.5<br>72.1<br>35.9<br>**11.9**<br>44.8<br>71.4<br>39.2|33.8<br>33.0<br>32.5<br>32.4|
|Ours|✓<br>✓<br>✓<br>✓|**42.6**<br>**32.4**|**18.3**<br>**18.4**|11.6<br>**51.1**<br>**73.6**<br>**39.5**|**35.9 (+12.5%)**|



Table 5. Ablation study on training objectives. ITC: image-text contrasting IC: image captioning. MIM: masked image modeling. TGMIM: text-guided masked image modeling. RM: random masking for MIM and TG-MIM. AM: attentive masking for MIM and TG-MIM. mIR/mTR refers to the corresponding mean value of R@1, R@5 and R@10. 

|_rl_<br>_rh_|zero-sho|t retrieval|zero-shot classifcation|Avg.|
|---|---|---|---|---|
||Flickr30K|MSCOCO|IN-1K<br>C-10<br>STL-10<br>Caltech||
||mTR<br>mIR|mTR<br>mIR|||
|25%<br>75%|40.1<br>31.0|17.5<br>17.8|11.1<br>49.7<br>73.3<br>39.4|35.0|
|(a)<br>50%<br>50%<br><br>|42.6<br>32.4|18.3<br>18.4|11.6<br>51.1<br>73.6<br>39.5|35.9|
|75%<br>25%|40.7<br>31.1|17.5<br>17.3|11.1<br>47.6<br>72.2<br>39.8|34.7|
|25%<br>50%|42.5<br>32.4|18.3<br>18.3|11.1<br>48.0<br>73.9<br>39.4|35.5|
|(b)<br>50%<br>50%<br><br>|42.6<br>32.4|18.3<br>18.4|11.6<br>51.1<br>73.6<br>39.5|35.9<br>|
|75%<br>50%|42.8<br>32.6|18.4<br>18.4|11.9<br>48.3<br>72.7<br>40.1|35.6|
|50%<br>25%|42.4<br>32.3|18.3<br>18.3|11.5<br>49.9<br>74.1<br>36.5|35.4|
|(c)<br>50%<br>50%<br><br>|42.6<br>32.4|18.3<br>18.4|11.6<br>51.1<br>73.6<br>39.5|35.9|
|50%<br>75%|43.2<br>33.3|18.6<br>18.8|11.7<br>51.2<br>73.1<br>39.0|36.1|



|_λT M_|zero-sho|t retrieval<br>MSCOCO<br>mTR<br>mIR|zero-shot classifcation<br>IN-1K<br>C-10<br>STL-10<br>Caltech<br>Avg.|zero-shot classifcation<br>IN-1K<br>C-10<br>STL-10<br>Caltech<br>Avg.|
|---|---|---|---|---|
||Flickr30K||||
||mTR<br>mIR||||
|0_._1<br>0_._5|42.1<br>**32.4**<br>**42.8**<br>32.2|**18.3**<br>18.2<br>18.2<br>18.2|11.3<br>46.5<br>73.2<br>39.8<br>11.5<br>46.1<br>**73.6**<br>38.6|35.2<br>35.1|
|1_._0|42.6<br>**32.4**|**18.3**<br>**18.4**|**11.6**<br>**51.1**<br>**73.6**<br>39.5|**35.9**|
|2_._0|40.5<br>31.3|17.7<br>17.8|11.3<br>47.3<br>72.2<br>**40.0**|34.8|



Table 7. Comparison of different weights for the TG-MIM objective. mIR/mTR refers to the corresponding mean value of R@1, R@5 and R@10. 

Table 6. Comparison of different masking ratios. mIR/mTR refers to the corresponding mean value of R@1, R@5 and R@10. 

SyCoCa is not sensitive to changes in the mask ratio. 

tasks namely IC and TG-MIM, our proposed SyCoCa achieves an average improvement of 12 _._ 5% compared to CoCa on CC3M dataset. 

- When applying a random masking mechanism, the performance of TG-MIM slightly decreases compared to MIM. That’s because the randomly masked patches may not be relevant to the text description, thereby introducing additional noise in the modal interaction and consequently affecting the performance of TG-MIM. 

- Introducing TG-MIM alone, without the presence of AM, has little impact on performance. This is because attention masking is essential for TG-MIM to rely on text information in order to reconstruct highly correlated visual areas, thereby enhancing the cross-modal interaction between the vision and text modalities. 

**Impact of Masking Ratio.** We train SyCoCa with different high/low masking ratios to evaluate their impact on model performance. Our evaluation is designed from three aspects: (a) dividing all image patches into high or low attentive masking, (b) maintaining a fixed low masking ratio while varying the high masking ratio, and (c) vice versa. The results are presented in Table 6. From Table 6 (a), we can observe that dividing the image patches evenly yields better results compared to biased partitioning, as it balances the visual information used in objectives in terms of image captioning and Tg-MIM. Moreover, in Table 6 (b)-(c) 

**Training objective Weights.** We also investigate the impact of different weights for the TG-MIM, by varying the weights _λT M_ assigned to the TG-MIM loss while keeping other objectives’ weights fixed. The results in Table 7 indicate that within the specific range of [0 _._ 1 _,_ 1 _._ 0], the variation of _λT M_ has minimal impact on the performance. We select _λT M_ = 1 _._ 0 for training our model. 

## **5. Conclusion** 

In this paper, we introduce a novel vision-language pretraining method called SyCoCa, which aims to further enhance multi-modal alignment. Our approach focuses on improving the fine-grained understanding between vision and language modalities by introducing the text-guided masked image modeling (TG-MIM) training objective. By incorporating the TG-MIM training objective into the CoCa framework, we establish bidirectional local interaction, which leads to a precise and fine-grained alignment between the vision and language modalities. Additionally, we propose a new attentive masking approach for TG-MIM, which selectively masks image patches that have strong correlations with the text caption. By focusing on these highly relevant patches, we enhance the cross-modal interaction and significantly improve overall performance. Through extensive experiments on five vision-language tasks, we demonstrate the effectiveness and generalization ability of our SyCoCa. 

8 

## **References** 

- [1] Harsh Agrawal, Karan Desai, Yufei Wang, Xinlei Chen, Rishabh Jain, Mark Johnson, Dhruv Batra, Devi Parikh, Stefan Lee, and Peter Anderson. Nocaps: Novel object captioning at scale. In _ICCV_ , pages 8948–8957, 2019. 6, 12 

- [2] Hangbo Bao, Li Dong, Songhao Piao, and Furu Wei. Beit: Bert pre-training of image transformers. _arXiv preprint arXiv:2106.08254_ , 2021. 2, 3 

- [3] Hangbo Bao, Wenhui Wang, Li Dong, Qiang Liu, Owais Khan Mohammed, Kriti Aggarwal, Subhojit Som, Songhao Piao, and Furu Wei. Vlmo: Unified vision-language pre-training with mixture-of-modality-experts. In _NeurIPS_ , pages 32897–32912, 2022. 2 

- [4] Lukas Bossard, Matthieu Guillaumin, and Luc Van Gool. Food-101–mining discriminative components with random forests. In _ECCV_ , pages 446–461, 2014. 7 

- [5] Minwoo Byeon, Beomhee Park, Haecheon Kim, Sungjun Lee, Woonhyuk Baek, and Saehoon Kim. Coyo-700m: Image-text pair dataset. https://github.com/ kakaobrain/coyo-dataset, 2022. 11 

- [6] Soravit Changpinyo, Piyush Sharma, Nan Ding, and Radu Soricut. Conceptual 12m: Pushing web-scale image-text pretraining to recognize long-tail visual concepts. In _CVPR_ , pages 3558–3568, 2021. 5 

- [7] Mark Chen, Alec Radford, Rewon Child, Jeffrey Wu, Heewoo Jun, David Luan, and Ilya Sutskever. Generative pretraining from pixels. In _ICML_ , pages 1691–1703. PMLR, 2020. 2, 3 

- [8] Yen-Chun Chen, Linjie Li, Licheng Yu, Ahmed El Kholy, Faisal Ahmed, Zhe Gan, Yu Cheng, and Jingjing Liu. Uniter: Universal image-text representation learning. In _ECCV_ , pages 104–120, 2020. 2 

- [9] Mehdi Cherti, Romain Beaumont, Ross Wightman, Mitchell Wortsman, Gabriel Ilharco, Cade Gordon, Christoph Schuhmann, Ludwig Schmidt, and Jenia Jitsev. Reproducible scaling laws for contrastive language-image learning. _arXiv preprint arXiv:2212.07143_ , 2022. 12 

- [10] Mircea Cimpoi, Subhransu Maji, Iasonas Kokkinos, Sammy Mohamed, and Andrea Vedaldi. Describing textures in the wild. In _CVPR_ , pages 3606–3613, 2014. 7 

- [11] Adam Coates, Andrew Ng, and Honglak Lee. An analysis of single-layer networks in unsupervised feature learning. In _AISTATS_ , pages 215–223, 2011. 7 

- [12] Jia Deng, Wei Dong, Richard Socher, Li-Jia Li, Kai Li, and Li Fei-Fei. Imagenet: A large-scale hierarchical image database. In _CVPR_ , pages 248–255, 2009. 7 

- [13] Carl Doersch, Abhinav Gupta, and Alexei A Efros. Unsupervised visual representation learning by context prediction. In _ICCV_ , pages 1422–1430, 2015. 2 

- [14] Xiaoyi Dong, Jianmin Bao, Yinglin Zheng, Ting Zhang, Dongdong Chen, Hao Yang, Ming Zeng, Weiming Zhang, Lu Yuan, Dong Chen, et al. Maskclip: Masked selfdistillation advances contrastive language-image pretraining. In _CVPR_ , pages 10995–11005, 2023. 7 

- [15] Alexey Dosovitskiy, Lucas Beyer, Alexander Kolesnikov, Dirk Weissenborn, Xiaohua Zhai, Thomas Unterthiner, 

Mostafa Dehghani, Matthias Minderer, Georg Heigold, Sylvain Gelly, et al. An image is worth 16x16 words: Transformers for image recognition at scale. _arXiv preprint arXiv:2010.11929_ , 2020. 2, 3 

- [16] Li Fei-Fei, Rob Fergus, and Pietro Perona. Learning generative visual models from few training examples: An incremental bayesian approach tested on 101 object categories. In _CVPRW_ , pages 178–178, 2004. 7 

- [17] Shashank Goel, Hritik Bansal, Sumit Bhatia, Ryan Rossi, Vishwa Vinay, and Aditya Grover. Cyclip: Cyclic contrastive language-image pretraining. In _NeurIPS_ , pages 6704–6719, 2022. 5 

- [18] Yash Goyal, Tejas Khot, Douglas Summers-Stay, Dhruv Batra, and Devi Parikh. Making the v in vqa matter: Elevating the role of image understanding in visual question answering. In _CVPR_ , pages 6904–6913, 2017. 6 

- [19] Kaiming He, Xinlei Chen, Saining Xie, Yanghao Li, Piotr Doll´ar, and Ross Girshick. Masked autoencoders are scalable vision learners. In _CVPR_ , pages 16000–16009, 2022. 1, 3 

- [20] Dan Hendrycks, Steven Basart, Norman Mu, Saurav Kadavath, Frank Wang, Evan Dorundo, Rahul Desai, Tyler Zhu, Samyak Parajuli, Mike Guo, et al. The many faces of robustness: A critical analysis of out-of-distribution generalization. In _ICCV_ , pages 8340–8349, 2021. 7 

- [21] Dan Hendrycks, Kevin Zhao, Steven Basart, Jacob Steinhardt, and Dawn Song. Natural adversarial examples. In _CVPR_ , pages 15262–15271, 2021. 7 

- [22] Xiaowei Hu, Zhe Gan, Jianfeng Wang, Zhengyuan Yang, Zicheng Liu, Yumao Lu, and Lijuan Wang. Scaling up vision-language pre-training for image captioning. In _CVPR_ , pages 17980–17989, 2022. 12 

- [23] Chao Jia, Yinfei Yang, Ye Xia, Yi-Ting Chen, Zarana Parekh, Hieu Pham, Quoc Le, Yun-Hsuan Sung, Zhen Li, and Tom Duerig. Scaling up visual and vision-language representation learning with noisy text supervision. In _ICML_ , pages 4904– 4916, 2021. 2, 11, 12 

- [24] Aditya Khosla, Nityananda Jayadevaprakash, Bangpeng Yao, and Li Fei-Fei. Novel dataset for fine-grained image categorization. In _CVPRW_ , 2011. 6, 7 

- [25] Wonjae Kim, Bokyung Son, and Ildoo Kim. Vilt: Visionand-language transformer without convolution or region supervision. In _ICML_ , pages 5583–5594, 2021. 2 

- [26] Alex Krizhevsky, Geoffrey Hinton, et al. Learning multiple layers of features from tiny images. 2009. 7 

- [27] Yann LeCun, Corinna Cortes, and CJ Burges. Mnist handwritten digit database. _ATT Labs [Online]. Available: http://yann.lecun.com/exdb/mnist_ , 2, 2010. 7 

- [28] Chenliang Li, Haiyang Xu, Junfeng Tian, Wei Wang, Ming Yan, Bin Bi, Jiabo Ye, Hehong Chen, Guohai Xu, Zheng Cao, et al. mplug: Effective and efficient vision-language learning by cross-modal skip-connections. _arXiv preprint arXiv:2205.12005_ , 2022. 1 

- [29] Junnan Li, Ramprasaath Selvaraju, Akhilesh Gotmare, Shafiq Joty, Caiming Xiong, and Steven Chu Hong Hoi. Align before fuse: Vision and language representation learning with momentum distillation. In _NeurIPS_ , pages 9694– 9705, 2021. 1 

9 

- [30] Junnan Li, Dongxu Li, Caiming Xiong, and Steven Hoi. Blip: Bootstrapping language-image pre-training for unified vision-language understanding and generation. In _ICML_ , pages 12888–12900, 2022. 12 

- [31] Junnan Li, Dongxu Li, Silvio Savarese, and Steven Hoi. Blip-2: Bootstrapping language-image pre-training with frozen image encoders and large language models. _arXiv preprint arXiv:2301.12597_ , 2023. 2 

- [32] Liunian Harold Li, Pengchuan Zhang, Haotian Zhang, Jianwei Yang, Chunyuan Li, Yiwu Zhong, Lijuan Wang, Lu Yuan, Lei Zhang, Jenq-Neng Hwang, et al. Grounded language-image pre-training. In _CVPR_ , pages 10965–10975, 2022. 5 

- [33] Xianhang Li, Zeyu Wang, and Cihang Xie. Clipa-v2: Scaling clip training with 81.1 _arXiv preprint arXiv:2306.15658_ , 2023. 12 

- [34] Tsung-Yi Lin, Michael Maire, Serge Belongie, James Hays, Pietro Perona, Deva Ramanan, Piotr Doll´ar, and C Lawrence Zitnick. Microsoft coco: Common objects in context. In _ECCV_ , pages 740–755, 2014. 6, 12 

- [35] Ilya Loshchilov and Frank Hutter. Decoupled weight decay regularization. _arXiv preprint arXiv:1711.05101_ , 2017. 5 

- [36] Norman Mu, Alexander Kirillov, David Wagner, and Saining Xie. Slip: Self-supervision meets language-image pretraining. In _ECCV_ , pages 529–544, 2022. 5 

- [37] Maria-Elena Nilsback and Andrew Zisserman. Automated flower classification over a large number of classes. In _ICVGIP_ , pages 722–729, 2008. 7 

- [38] Omkar M Parkhi, Andrea Vedaldi, Andrew Zisserman, and CV Jawahar. Cats and dogs. In _CVPR_ , pages 3498–3505, 2012. 7 

- [39] Deepak Pathak, Philipp Krahenbuhl, Jeff Donahue, Trevor Darrell, and Alexei A Efros. Context encoders: Feature learning by inpainting. In _CVPR_ , pages 2536–2544, 2016. 2 

- [40] Bryan A Plummer, Liwei Wang, Chris M Cervantes, Juan C Caicedo, Julia Hockenmaier, and Svetlana Lazebnik. Flickr30k entities: Collecting region-to-phrase correspondences for richer image-to-sentence models. In _ICCV_ , pages 2641–2649, 2015. 6, 12 

- [41] Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, et al. Learning transferable visual models from natural language supervision. In _ICML_ , pages 8748–8763. PMLR, 2021. 1, 2, 5, 12 

- [42] Benjamin Recht, Rebecca Roelofs, Ludwig Schmidt, and Vaishaal Shankar. Do imagenet classifiers generalize to imagenet? In _ICML_ , pages 5389–5400, 2019. 7 

- [43] Christoph Schuhmann, Romain Beaumont, Richard Vencu, Cade Gordon, Ross Wightman, Mehdi Cherti, Theo Coombes, Aarush Katta, Clayton Mullis, Mitchell Wortsman, et al. Laion-5b: An open large-scale dataset for training next generation image-text models. _NIPS_ , 35:25278–25294, 2022. 11 

- [44] Ramprasaath R Selvaraju, Michael Cogswell, Abhishek Das, Ramakrishna Vedantam, Devi Parikh, and Dhruv Batra. 

Grad-cam: Visual explanations from deep networks via gradient-based localization. In _ICCV_ , pages 618–626, 2017. 7 

- [45] Piyush Sharma, Nan Ding, Sebastian Goodman, and Radu Soricut. Conceptual captions: A cleaned, hypernymed, image alt-text dataset for automatic image captioning. In _Proceedings of the 56th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)_ , pages 2556–2565, 2018. 7 

- [46] Sheng Shen, Liunian Harold Li, Hao Tan, Mohit Bansal, Anna Rohrbach, Kai-Wei Chang, Zhewei Yao, and Kurt Keutzer. How much can clip benefit visionand-language tasks? _arXiv preprint arXiv:2107.06383_ , 3, 2021. 12 

- [47] Amanpreet Singh, Ronghang Hu, Vedanuj Goswami, Guillaume Couairon, Wojciech Galuba, Marcus Rohrbach, and Douwe Kiela. Flava: A foundational language and vision alignment model. In _CVPR_ , pages 15638–15650, 2022. 5 

- [48] Quan Sun, Yuxin Fang, Ledell Wu, Xinlong Wang, and Yue Cao. Eva-clip: Improved training techniques for clip at scale. _arXiv preprint arXiv:2303.15389_ , 2023. 12 

- [49] Hao Tan and Mohit Bansal. Lxmert: Learning crossmodality encoder representations from transformers. _arXiv preprint arXiv:1908.07490_ , 2019. 2 

- [50] Trieu H Trinh, Minh-Thang Luong, and Quoc V Le. Selfie: Self-supervised pretraining for image embedding. _arXiv preprint arXiv:1906.02940_ , 2019. 2 

- [51] Catherine Wah, Steve Branson, Peter Welinder, Pietro Perona, and Serge Belongie. The caltech-ucsd birds-200-2011 dataset. 2011. 7 

- [52] Wenhui Wang, Hangbo Bao, Li Dong, Johan Bjorck, Zhiliang Peng, Qiang Liu, Kriti Aggarwal, Owais Khan Mohammed, Saksham Singhal, Subhojit Som, et al. Image as a foreign language: Beit pretraining for all vision and visionlanguage tasks. _arXiv preprint arXiv:2208.10442_ , 2022. 12 

- [53] Wenhui Wang, Hangbo Bao, Li Dong, Johan Bjorck, Zhiliang Peng, Qiang Liu, Kriti Aggarwal, Owais Khan Mohammed, Saksham Singhal, Subhojit Som, et al. Image as a foreign language: Beit pretraining for vision and visionlanguage tasks. In _CVPR_ , pages 19175–19186, 2023. 1, 2 

- [54] Weihan Wang, Zhen Yang, Bin Xu, Juanzi Li, and Yankui Sun. Vilta: Enhancing vision-language pre-training through textual augmentation. In _ICCV_ , pages 3158–3169, 2023. 7 

- [55] Zirui Wang, Jiahui Yu, Adams Wei Yu, Zihang Dai, Yulia Tsvetkov, and Yuan Cao. Simvlm: Simple visual language model pretraining with weak supervision. _arXiv preprint arXiv:2108.10904_ , 2021. 12 

- [56] Zhenda Xie, Zheng Zhang, Yue Cao, Yutong Lin, Jianmin Bao, Zhuliang Yao, Qi Dai, and Han Hu. Simmim: A simple framework for masked image modeling. In _CVPR_ , pages 9653–9663, 2022. 1, 3 

- [57] Haiyang Xu, Qinghao Ye, Ming Yan, Yaya Shi, Jiabo Ye, Yuanhong Xu, Chenliang Li, Bin Bi, Qi Qian, Wei Wang, et al. mplug-2: A modularized multi-modal foundation model across text, image and video. _arXiv preprint arXiv:2302.00402_ , 2023. 1 

- [58] An Yang, Junshu Pan, Junyang Lin, Rui Men, Yichang Zhang, Jingren Zhou, and Chang Zhou. Chinese clip: 

10 

Contrastive vision-language pretraining in chinese. _arXiv preprint arXiv:2211.01335_ , 2022. 1 

- [59] Jianwei Yang, Chunyuan Li, Pengchuan Zhang, Bin Xiao, Ce Liu, Lu Yuan, and Jianfeng Gao. Unified contrastive learning in image-text-label space. In _CVPR_ , pages 19163–19173, 2022. 7 

- [60] Lewei Yao, Runhui Huang, Lu Hou, Guansong Lu, Minzhe Niu, Hang Xu, Xiaodan Liang, Zhenguo Li, Xin Jiang, and Chunjing Xu. Filip: Fine-grained interactive language-image pre-training. _arXiv preprint arXiv:2111.07783_ , 2021. 5, 12 

- [61] Jiahui Yu, Zirui Wang, Vijay Vasudevan, Legg Yeung, Mojtaba Seyedhosseini, and Yonghui Wu. Coca: Contrastive captioners are image-text foundation models. _arXiv preprint arXiv:2205.01917_ , 2022. 1, 2, 5 

- [62] Jiahui Yu, Zirui Wang, Vijay Vasudevan, Legg Yeung, Mojtaba Seyedhosseini, and Yonghui Wu. Coca: Contrastive captioners are image-text foundation models. _arXiv preprint arXiv:2205.01917_ , 2022. 12 

- [63] Lu Yuan, Dongdong Chen, Yi-Ling Chen, Noel Codella, Xiyang Dai, Jianfeng Gao, Houdong Hu, Xuedong Huang, Boxin Li, Chunyuan Li, et al. Florence: A new foundation model for computer vision. _arXiv preprint arXiv:2111.11432_ , 2021. 2, 12 

- [64] Xiaohua Zhai, Xiao Wang, Basil Mustafa, Andreas Steiner, Daniel Keysers, Alexander Kolesnikov, and Lucas Beyer. Lit: Zero-shot transfer with locked-image text tuning. In _CVPR_ , pages 18123–18133, 2022. 5 

- [65] Xiaohua Zhai, Xiao Wang, Basil Mustafa, Andreas Steiner, Daniel Keysers, Alexander Kolesnikov, and Lucas Beyer. Lit: Zero-shot transfer with locked-image text tuning. In _CVPR_ , pages 18123–18133, 2022. 12 

- [66] Xiaohua Zhai, Basil Mustafa, Alexander Kolesnikov, and Lucas Beyer. Sigmoid loss for language image pre-training. _arXiv preprint arXiv:2303.15343_ , 2023. 12 

- [67] Pengchuan Zhang, Xiujun Li, Xiaowei Hu, Jianwei Yang, Lei Zhang, Lijuan Wang, Yejin Choi, and Jianfeng Gao. Vinvl: Revisiting visual representations in vision-language models. In _CVPR_ , pages 5579–5588, 2021. 2, 12 

- [68] Yiwu Zhong, Jianwei Yang, Pengchuan Zhang, Chunyuan Li, Noel Codella, Liunian Harold Li, Luowei Zhou, Xiyang Dai, Lu Yuan, Yin Li, et al. Regionclip: Region-based language-image pretraining. In _CVPR_ , pages 16793–16803, 2022. 7 

## **Appendix: Comparison with other methods** 

To verify the effectiveness of our proposed Text-Guided Masked Image Modeling (TG-MIM) and attentive masking (AM) strategy, the experiments are conducted under the same settings of CoCa and SyCoCa in the main paper. Especially for the training dataset, we used CC12M, a small dataset with 12 million image-text pairs. As we all know, the amount of training dataset has a crucial impact on model performance. To eliminate data factors and reflect the superiority of our algorithm, we use a large-scale image-text pairs dataset for pre-training. Specifically, we collect Laion-2B [43] and COYO-700M [5]. In addition, the number of parameters will also affect the model performance. To train the collected datasets, we use a total parameter size of 930M. Tab 8 summarizes the pre-training model configurations of the related methods. The comparison results between our SyCoCa and other methods are shown in Tab 9 and Tab 10. 

**Image-Text Retrieval** . For image-text retrieval task, the comparison results are shown in Tab 9. Our SyCoCa ahieves the best results in terms of text-to-image retrieval on the Flickr30K dataset, as well as text-to-image/imageto-text retrieval on the MSCOCO dataset. It is worth noting that, SyCoCa outperforms ALIGN [23] in 11 out of 12 cases, despite both models having similar model parameter sizes and pre-training dataset sizes. In addition, even though some models (such as CoCa, BEiT-3, and SigLIP) employ larger parameter sizes and datasets, our SyCoCa still achieves comparable results. For instance, the mean text-to-image/image-to-text recall results for SyCoCa are 96 _._ 0 _/_ 90 _._ 5 _/_ **82.3** _/_ **69.8** , while for CoCa, they are **97.3** _/_ **91.3** _/_ 81 _._ 4 _/_ 69 _._ 1. 

**Image Caption** . For image caption task, the comparison results are shown in Tab 10. On MSCOCO dataset, our SyCoCa is slightly lower than the best method on BLEU@4 and SPICE, but outperforms CoCa on all metrics. On NoCaps dataset, our method achieves the best performance, which shows that our attentive mask strategy has certain advantages in the caption task. 

11 

|Model|total #param.|precision|dataset|image size|
|---|---|---|---|---|
|CLIP [41]|430M|fp16|WIT-400M|3362|
|OpenCLIP [9]|430M|bf16|LAION-2B|2242|
|ALIGN [23]|820M|-|ALIGN-1.8B|2892|
|FILIP [60]|430M|fp16|FILIP-340M|2242|
|Florence [63]|890M|-|FLD-900M|2242|
|CoCa [62]|2.1B|-|JFT-3B+ALIGN-1.8B|2882|
|CoCa-Large [62]|790M|-|JFT-3B+ALIGN-1.8B|2882|
|BEiT-3 [52]|1.9B|-|15M images+160GB documents+21M pairs|2242|
|EVA-CLIP [48]|430M|-|LAION-2B+COYO-700M|2242|
|SigLIP [66]|430M|-|WebLI_†_|2242|
|CLIPA-v2 [33]|1.0B|-|DataComp-1B|3362|
|LiT [65]|1.3B|-|LiT-4B|2882|
|SyCoCa (ours)|930M|bf16|LAION-2B+COYO-700M|2242|



Table 8. Pre-training model configurations. _[†]_ WebLI is a private multilingual dataset conducted by Google, which consists of 10 billion images and 12 billion alt-texts. 

|Model|Flickr30K|Flickr30K|Flickr30K|Flickr30K|MSCOCO<br>Image_→_Text<br>Text_→_Image<br>R@1<br>R@5<br>R@10<br>R@1<br>R@5<br>R@10|MSCOCO<br>Image_→_Text<br>Text_→_Image<br>R@1<br>R@5<br>R@10<br>R@1<br>R@5<br>R@10|
|---|---|---|---|---|---|---|
||Image_→_Text||Text_→_Image||||
||R@1|R@5<br>R@10|R@1<br>R@5<br>R@10||||
|CLIP [41]<br>OpenCLIP [9]<br>ALIGN [23]<br>FILIP [60]<br>Florence [63]<br>CoCa [62]<br>BEiT-3 [52]<br>EVA-CLIP [48]<br>SigLIP [66]<br>CLIPA-v2 [33]|88.0<br>88.7<br>88.6<br>89.8<br>**90.9**<br>92.5<br>94.9<br>89.7<br>-<br>89.1|98.7<br>99.4<br>98.4<br>99.2<br>98.7<br>99.7<br>**99.2**<br>**99.8**<br>99.1<br>-<br>99.5<br>99.9<br>99.9<br>100.0<br>98.6<br>99.2<br>-<br>-<br>-<br>-|68.7<br>90.6<br>75.0<br>92.5<br>75.7<br>93.8<br>75.0<br>93.4<br>76.7<br>93.6<br>80.4<br>95.7<br>81.5<br>95.6<br>77.3<br>93.6<br>-<br>-<br>73.0<br>-|95.2<br>95.6<br>96.8<br>96.3<br>-<br>97.7<br>97.8<br>96.8<br>-<br>-|58.4<br>81.5<br>88.1<br>62.1<br>83.4<br>90.3<br>58.6<br>83.0<br>89.7<br>61.3<br>84.3<br>90.4<br>64.7<br>85.9<br>-<br>66.3<br>86.2<br>91.8<br>-<br>-<br>-<br>63.7<br>84.3<br>90.4<br>70.2<br>-<br>-<br>64.1<br>-<br>-|37.8<br>62.4<br>72.2<br>46.1<br>70.7<br>79.4<br>45.6<br>69.8<br>78.6<br>45.9<br>70.6<br>79.3<br>47.2<br>71.4<br>-<br>51.2<br>74.2<br>82.0<br>-<br>-<br>-<br>47.5<br>71.2<br>79.7<br>52.0<br>-<br>-<br>46.3<br>-<br>-|
|SyCoCa (ours)|89.2|99.1<br>99.6|**78.7**<br>**95.4**|**97.4**|**67.2**<br>**87.5**<br>**92.1**|**50.7**<br>**75.7**<br>**82.9**|



Table 9. Zero-shot image-text retrieval comparisons on Flickr30K [40] and MSCOCO [34]. Results of models that use significantly larger parameter sizes or dataset sizes are indicated in gray. 

|Model|MSCOCO<br>B@4<br>M<br>C<br>S|MSCOCO<br>B@4<br>M<br>C<br>S|NoCaps<br>val<br>test<br>C<br>S<br>C<br>S|NoCaps<br>val<br>test<br>C<br>S<br>C<br>S|NoCaps<br>val<br>test<br>C<br>S<br>C<br>S|
|---|---|---|---|---|---|
||B@4<br>M||val|||
||||C<br>S|||
|CLIP-ViL [46]<br>BLIP [30]<br>VinVL [67]<br>SimVLM [55]<br>LEMON [22]<br>CoCa [62]|40.2<br>29.7<br>40.4<br>-<br>41.0<br>31.1<br>40.6<br>33.7<br>**41.5**<br>30.8<br>40.9<br>33.9|134.2<br>23.8<br>136.7<br>-<br>140.9<br>**25.4**<br>143.3<br>**25.4**<br>139.1<br>24.1<br>143.6<br>24.7|-<br>-<br>113.2<br>14.8<br>105.1<br>14.4<br>112.2<br>-<br>117.3<br>15.0<br>122.4<br>15.5||-<br>-<br>-<br>-<br>103.7<br>14.4<br>110.3<br>14.5<br>114.3<br>14.9<br>120.6<br>15.5|
|SyCoCa (ours)|41.4<br>**34.1**|**143.7**<br>25.3|**122.6**<br>**15.8**||**121.1**<br>**15.6**|



Table 10. Image Captioning comparisons on MSCOCO [34] and NoCaps [1]. B@4: BLEU@4, M: METEOR, C: CIDEr, S: SPICE. 

12 

