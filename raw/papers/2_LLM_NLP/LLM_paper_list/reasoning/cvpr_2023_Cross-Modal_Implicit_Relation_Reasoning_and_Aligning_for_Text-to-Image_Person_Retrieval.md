# **Cross-Modal Implicit Relation Reasoning and Aligning for Text-to-Image Person Retrieval** 

Ding Jiang[1] , Mang Ye[1] _[,]_[2][*] 

1National Engineering Research Center for Multimedia Software, Institute of Artificial Intelligence, Hubei Key Laboratory of Multimedia and Network Communication Engineering, School of Computer Science, Wuhan University, Wuhan, China 

2 Hubei Luojia Laboratory, Wuhan, China 

https://github.com/anosorae/IRRA 

## **Abstract** 

_Text-to-image person retrieval aims to identify the target person based on a given textual description query. The primary challenge is to learn the mapping of visual and textual modalities into a common latent space. Prior works have attempted to address this challenge by leveraging separately pre-trained unimodal models to extract visual and textual features. However, these approaches lack the necessary underlying alignment capabilities required to match multimodal data effectively. Besides, these works use prior information to explore explicit part alignments, which may lead to the distortion of intra-modality information. To alleviate these issues, we present IRRA: a cross-modal Implicit Relation Reasoning and Aligning framework that learns relations between local visual-textual tokens and enhances global image-text matching without requiring additional prior supervision. Specifically, we first design an Implicit Relation Reasoning module in a masked language modeling paradigm. This achieves cross-modal interaction by integrating the visual cues into the textual tokens with a cross-modal multimodal interaction encoder. Secondly, to globally align the visual and textual embeddings, Similarity Distribution Matching is proposed to minimize the KL divergence between image-text similarity distributions and the normalized label matching distributions. The proposed method achieves new state-of-the-art results on all three public datasets, with a notable margin of about 3%-9% for Rank-1 accuracy compared to prior methods._ 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/cvpr_2023_Cross-Modal_Implicit_Relation_Reasoning_and_Aligning_for_Text-to-Image_Person_Retrieval_images/cvpr_2023_Cross-Modal_Implicit_Relation_Reasoning_and_Aligning_for_Text-to-Image_Person_Retrieval.pdf-0001-08.png)


**----- Start of picture text -----**<br>
A woman in a<br>Gray pair of<br>Image Text shorts, a pair of<br>Encoder Encoder Gray shoes and a<br>white purse<br>around her waist.<br>(a) Early global matching paradigm<br>A woman in a<br>Gray pair of<br>Image Text shorts,  a pair of<br>Encoder Encoder Gray shoes and a<br>white purse<br>around her waist.<br>(b) Existing explicit local matching paradigm<br>A woman in a<br>Gray pair of<br>Image Text shorts, a pair of<br>Encoder Encoder Gray shoes and a<br>[MASK] purse<br>around her waist.<br>white<br>(c) Our implicit relation reasoning aided matching paradigm<br>Alignment Global Image Feature Global Text Feature<br>Attention Local Image Feature Local Text Feature Masked Token<br>**----- End of picture text -----**<br>


Figure 1. Evolution of text-to-image person retrieval paradigms. (a) Early global-matching method directly align global image and text embeddings. (b) Recent local-matching method, explicitly extract and align local image and text embeddings. (c) Our implicit relation reasoning method, implicitly reasoning the relation among all local tokens to better align global image and text embeddings. 

## **1. Introduction** 

Text-to-image person retrieval aims to retrieve a personof-interest from a large image gallery that best matches the 

> *Corresponding Author: Mang Ye (yemang@whu.edu.cn) 

text description query [30], which is a sub-task of both image-text retrieval [26, 33, 42] and image-based person reidentification (Re-ID) [15,32,45]. Textual descriptions provide a natural and relatively comprehensive way to describe a person’s attributes, and are more easily accessible than images. Text-to-image person retrieval thus received increas- 

1 

ing attention in recent years, benefiting a variety of applications from personal photo album search to public security. 

However, text-to-image person retrieval remains a challenging task due to significant intra-identity variations and modality heterogeneity between vision and language. The former challenge stems from the fact that visual appearances of an identity differ based on pose, viewpoint, illumination, and other factors, while textual description varies by arbitrary descriptive order and textual ambiguity. The latter challenge is the primary issue in cross-modal tasks and is caused by inherent representation discrepancies between vision and language. To tackle above two challenges, the core research problem in text-to-image person retrieval is to explore better ways to extract discriminative feature representations and to design better cross-modal matching methods to align images and texts into a joint embedding space. Early _global-matching_ methods [53, 54] aligned images and texts into a joint embedding space by designing cross-modal matching loss functions (Fig. 1 (a)). Typically, these approaches learned cross-modal alignments by using matching losses only at the end of the network, failing to achieve sufficient modality interaction in middle-level layers, which are crucial to bridge the feature-level modality gap. Therefore, some later methods [5, 7, 21, 46] introduced the practice of _local-matching_ by building the correspondence between the body parts and the textual entities (Fig. 1 (b)). Although this local matching strategy benefits retrieval performance, it introduces unavoidable noise and uncertainty in the retrieval process. Besides, the strategy requires extracting and storing multiple local part representations of images and texts, computing pairwise similarity between all those representations during inference. These resource-demanding properties limit their applicability for practical large-scale scenarios. 

In this paper, we present IRRA: a _cross-modal Implicit Relation Reasoning and Aligning_ framework, which performs global alignment with the aid of cross-modal implicit local relation learning. Unlike previous methods that heavily rely on explicit fine-grained local alignment, our approach implicitly utilizes fine-grained information to enhance global alignment without requiring any additional supervision and inference costs (Fig. 1 (c)). Specifically, we design an Implicit Relation Reasoning module that effectively builds relations between visual and textual representations through self- and cross-attention mechanisms. This fused representation is then utilized to perform masked language modeling (MLM) task to achieve effective implicit inter-modal and intra-modal fine-grained relation learning. MLM is generally utilized during the pre-training stage of vision-language pre-training (VLP) [6, 9, 27, 31, 41]. In this work, we make the first attempt to demonstrate the effectiveness of MLM in downstream fine-tuning tasks. Our main innovation is the design of a multimodal interaction 

encoder that can efficiently fuse visual and textual representations, align cross-modal fine-grained features through the MLM task. This design helps the backbone network to extract more discriminative global image-text representations without requiring additional supervision. 

To guide the image-text matching, commonly used loss functions include ranking loss and cross-modal projection matching (CMPM) [53] loss. Compared to ranking loss, the CMPM loss does not require the selection of specific triplets or margin parameter tuning. It exhibits great stability with varying batch sizes, making it widely used in textto-image person retrieval [5, 39, 50]. However, we found that the projection in CMPM can be regarded as a variable weight that adjusts the distribution of softmax output logits, similar to the temperature parameter [17] for knowledge distillation. Nevertheless, limited by the varying projection length, CMPM therefore cannot precisely control the projection probability distribution, making it difficult to focus on hard-negative samples during model updates. To explore more effective cross-modal matching objective, we further propose an image-text similarity distribution matching (SDM) loss. The SDM loss minimizes the KL divergence between the normalized image-text similarity score distributions and the normalized ground truth label matching distributions. Additionally, we introduce a temperature hyperparameter to precisely control the similarity distribution compactness, which enables the model updates focus on hard-negative samples and effectively enlarges the variance between non-matching pairs and the correlation between matching pairs. 

To address the limitations of separate pre-trained models on unimodal datasets, we leverage the Contrastive Language-Image Pre-training (CLIP) [35] as the initialization of our model. CLIP is pre-trained with abundant image-text pairs and has powerful underlying cross-modal alignment capabilities. Some previous approaches [13, 50] have either frozen some part of parameters or introduced only CLIP’s image encoder, which resulted in their inability to fully exploit CLIP’s powerful capabilities in image-text matching. With the proposed IRRA, we successfully transfer the powerful knowledge directly from the pre-trained full CLIP model and continue to learn fine-grained crossmodal implicit local relations on text-to-image person retrieval datasets. In addition, compared to many recent methods [5, 38, 50], IRRA is more efficient as it computes only one global image-text pair similarity score in the inference stage. The main contributions can be summarized as follows: 

- We propose IRRA to implicitly utilize fine-grained interaction to enhance the global alignment without requiring any additional supervision and inference cost. 

- We introduce a new cross-modal matching loss named 

2 

image-text similarity distribution matching (SDM) loss. It directly minimizes the KL divergence between image-text similarity distributions and the normalized label matching distributions. 

- We demonstrate that the full CLIP model can be applied to text-to-image person retrieval and can outperform existing state-of-the-art methods with straightforward fine-tuning. Moreover, our proposed IRR module enables fine-grained image-text relation learning, allowing IRRA to learn more discriminative image-text representations. 

- Extensive experiments on three public benchmark datasets, _i.e_ ., CUHK-PEDES [30], ICFG-PEDES [7] and RSTPReid [55] show that IRRA consistently outperforms the state-of-the-arts by a large margin. 

## **2. Related work** 

**Text-to-image Person Retrieval** was first introduced by Li _et al_ . [30], who proposed the first benchmark dataset, CUHK-PEDES [30]. The main challenge is how to efficiently align image and text features into a joint embedding space for fast retrieval. Early works [2,29,30] utilized VGG [40] and LSTM [18] to learn representations for visualtextual modalities and then aligned them using a matching loss. Later works [4, 36, 53] improved the feature extraction backbone with ResNet50/101 [14] and BERT [22], as well as designed novel cross-modal matching losses to align global image-text features in a joint embedding space. More recent works [5, 46, 47, 49, 55] extensively employs additional local feature learning branches that explicitly exploit human segmentation, body parts, color information, and text phrases. There is also some works [7, 10, 38, 51] that implicitly performs local feature learning through attentional mechanisms. However, while these approaches have been shown to provide better retrieval results than using only global features, they also introduce additional computational complexity during inference when computing image-text similarity. The aforementioned works all use backbones pre-trained separately with unimodal data to extract visual and textual features, and then perform cross-modal alignment without exploiting the great crossmodal alignment capabilities of recently promising visionlanguage pre-training models. Han _et al_ . [13] first introduced a CLIP model for text-to-image person retrieval using a momentum contrastive learning framework to transfer the knowledge learned from large-scale generic imagetext pairs. Later, Yan _et al_ . [50] proposed a CLIP-driven fine-grain information excavation framework to transfer the knowledge of CLIP. However, they failed in directly transferring the original aligned CLIP dual-encoder to text-toimage person retrieval. In this work, we demonstrate that the CLIP model can be easily transferred to text-to-image 

person retrieval and propose the IRRA to learn more discriminative image-text embeddings. 

**Vision-Language Pre-training** aims to learn the semantic correspondence between vision and language modalities by pre-training on large-scale image-text pairs. Inspired by the success of Transformer-based [44] language model pretraining (such as BERT) [22] and Vision Transformer (ViT) [8], Vision-Language Pre-training (VLP) has emerged as the prevailing paradigm in learning multimodal representations, demonstrating strong results on downstream tasks such as image captioning [3], image-text retrieval [25] and visual question answering [1]. Existing work on VLP can be categorized into two types: single-stream and dualstream, depending on their model structure. In singlestream models [6, 23, 41], text and visual features are concatenated and then fed into a single transformer encoder. Although this architecture is more parameter-efficient as it uses the same set of parameters for both modalities, it has a slow retrieval speed during the inference stage because it needs to predict the similarity score of all possible imagetext pairs. On the other hand, dual-stream models [9,20,35] use two separate encoders to extract the text and visual features independently. These two transformer encoders do not share parameters. While achieving remarkable performance on image-text retrieval tasks, dual-stream modals lack the ability to model complex interactions between vision and language for other vision-language understanding tasks. 

## **3. Method** 

In this section, we present our proposed IRRA framework. The overview of IRRA is illustrated in Fig. 2 and the details are discussed in the following subsections. 

## **3.1. Feature Extraction Dual-Encoder** 

Previous works in text-to-image person retrieval typically utilize image and text encoders that are pre-trained separately on unimodal datasets. Inspired by the partial success of transferring knowledge from CLIP to text-image person retrieval [13], we directly initialize our IRRA with the full CLIP image and text encoder to enhance its underlying cross-modal alignment capabilities. 

**Image Encoder.** Given an input image _I ∈ R[H][×][W][ ×][C]_ , a CLIP pre-trained ViT model is adopted to obtain the image embedding. We first split _I_ into a sequence of _N_ = _H × W/P_[2] fixed-sized non-overlapping patches, where _P_ denotes the patch size, and then map the patch sequence to 1D tokens _{fi[v][}|][N] i_ =1[by][a][trainable][linear][pro-] jection. With injection of positional embedding and extra [CLS] token, the sequence of tokens _{fcls[v][, f]_ 1 _[ v][, ..., f] N[ v][}]_[are] input into L-layer transformer blocks to model correlations of each patch. Finally, a linear projection is adopted to map _fcls[v]_[to the joint image-text embedding space, which serves] as global image representation. 

3 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/cvpr_2023_Cross-Modal_Implicit_Relation_Reasoning_and_Aligning_for_Text-to-Image_Person_Retrieval_images/cvpr_2023_Cross-Modal_Implicit_Relation_Reasoning_and_Aligning_for_Text-to-Image_Person_Retrieval.pdf-0004-00.png)


**----- Start of picture text -----**<br>
Image Encoder<br>Image Input 12× [CLS]<br>Embedding<br>Text-Image Similarities KL<br>Cosine Similarity<br>Image Embedding<br>Similarity Score<br>True Match Labels<br>Similarity Distribution Matching<br>Multimodal Interaction Encoder<br>4× [MASK] Embedding<br>V<br>A woman in a Gray pair of<br>shorts, a pair of Gray shoes and  K gray<br>a white purse around her waist. Text Embedding<br>Random<br>Mask A woman in a  [MASK]  pair of  shoes<br>shorts , a pair of Gray  [MASK]  and<br>a white purse  [MASK]  her waist. around<br>MLM Text Embedding<br>Q<br>Text Input 12× Masked Language Modeling<br>Text Encoder<br>Implicit Relation Reasoning<br>Self Attention Feed Forward<br>SDM IRR<br>ID Loss<br>Self Attention Feed Forward Cross Attention Self Attention Feed Forward MLM Head<br>**----- End of picture text -----**<br>


Figure 2. **Overview of the proposed IRRA framework.** It consists of a dual-stream feature extraction backbone and three representation learning branches, _i.e_ . Implicit Relation Reasoning (IRR), Similarity Distribution Matching (SDM) and Identity Identification (ID loss). IRR aims to implicitly utilize fine-grained information to learn a discriminative global representation. SDM minimizes the KL divergence between image-text similarity score distributions and true label matching distributions, which can effectively enlarges the variance between non-matching pairs and the correlation between matching pairs. Additionally, we adopt ID loss to aggregate the feature representations of the same identity, further improving the retrieval performance. IRRA is trained end-to-end with these three tasks, and it computes only one global image-text similarity score, making it computationally efficient. Modules connected by dashed lines will be removed during inference stage. 

**Text Encoder.** For an input text _T_ , we directly use the CLIP text encoder to extract the text representation, which is a Transformer [44] modified by Radford _et al_ . [35]. Following CLIP, the lower-cased byte pair encoding (BPE) with a 49152 vocab size [37] is firstly employed to tokenize the input text description. The text description is bracketed with [SOS] and [EOS] tokens to indicate the start and end of sequence. Then the tokenized text _{fsos[t][, f]_ 1 _[ t][, ...f] eos[ t][}]_ are fed into the transformer and exploit correlations of each patch by masked self-attention. Finally, the highest layer of the transformer at the [EOS] token _feos[t]_[is][linearly][pro-] jected into the image-text joint embedding space to obtain the global text representation. 

## **3.2. Implicit Relation Reasoning** 

To fully exploit fine-grained information, it is crucial to bridge the significant modality gap between vision and language. While most existing methods do so by explicitly aligning local features between images and text, this paper introduces a novel approach. Specifically, we use MLM to implicitly mine fine-grained relations and learn discriminative global features. 

**Masked Language Modeling.** Masked language modeling (MLM) was initially proposed by Taylor [43] in 1953, 

it became widely known when the BERT model adapted it as a novel pre-training task. In this work, We utilize MLM to predict masked textual tokens not only by the rest of unmasked textual tokens but also by the visual tokens. Similar to the analysis of Fu _et al_ . [11] in pure language pretraining, MLM optimizes two properties: (1) the alignment of image and text contextualized representations with the static embeddings of masked textual tokens, and (2) the uniformity of static embeddings in the joint embedding space. In the alignment property, sampled embeddings of masked textual tokens serve as an anchor to align images and text contextualized representations, as illustrated in Fig. 3. We find that such a local anchor is essential for modeling local dependencies and can implicitly utilize fine-grained local information for global feature alignment. 

**Multimodal Interaction Encoder.** To achieve full interaction between image and text modalities, We design an efficient multimodal interaction encoder to fuse the image and text embeddings, compared to two other popular multimodal interaction modules [9, 16], our design is more computationally efficient, as illustrated in Fig. 4. The multimodal interaction encoder consists of a multi-head cross attention (MCA) layer and 4-layer transformer blocks. Given an input text description _T_ , we randomly mask out the text 

4 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/cvpr_2023_Cross-Modal_Implicit_Relation_Reasoning_and_Aligning_for_Text-to-Image_Person_Retrieval_images/cvpr_2023_Cross-Modal_Implicit_Relation_Reasoning_and_Aligning_for_Text-to-Image_Person_Retrieval.pdf-0005-00.png)


**----- Start of picture text -----**<br>
white shoulder  bag<br>carrying a bag<br>bag<br>Static Embedding<br>Image Embedding<br>holding a black bag Text Embedding<br>Alignment<br>Boundary<br>**----- End of picture text -----**<br>


Figure 3. Illustration of the MLM objective. MLM uses static embedding of masked textual tokens as local fine-grained keys to align image and text contextualized representations in the same context. 

tokens with a probability of 15% and replace them with the special token [MASK]. Following BERT, the replacements are 10% random tokens, 10% unchanged, and 80% [MASK]. The masked text is defined as _T_[ˆ] , and fed into the Text Transformer as described in Sec. 3.1. Then the last hidden states _{h[t] i_[ˆ] _[}][L] i_ =1[and] _[ {][h][v] i[}][N] i_ =1[of the text transformer and] the vision transformer are fed into the multimodal interaction encoder jointly. In order to fuse image and masked text representations more effectively, the masked text representation _{h[t] i_[ˆ] _[}][L] i_ =1[served as query(] _[Q]_[), and the image represen-] tation _{h[v] i[}][N] i_ =1[are served as key(] _[K]_[) and value(] _[V]_[).][The full] interaction between image and masked text representations can be achieved by: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/cvpr_2023_Cross-Modal_Implicit_Relation_Reasoning_and_Aligning_for_Text-to-Image_Person_Retrieval_images/cvpr_2023_Cross-Modal_Implicit_Relation_Reasoning_and_Aligning_for_Text-to-Image_Person_Retrieval.pdf-0005-03.png)


where _{h[m] i[}][L] i_ =1[is][the][fused][image][and][masked][text][con-] textualized representations, _L_ is the length of input textual tokens, _LN_ ( _·_ ) denotes Layer Normalization, the _MCA_ ( _·_ ) is the multi-head cross attention and can be realized by: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/cvpr_2023_Cross-Modal_Implicit_Relation_Reasoning_and_Aligning_for_Text-to-Image_Person_Retrieval_images/cvpr_2023_Cross-Modal_Implicit_Relation_Reasoning_and_Aligning_for_Text-to-Image_Person_Retrieval.pdf-0005-05.png)


where _d_ is the embedding dimension of masked tokens. 

For each masked position _{h[m] i_ : _i ∈M}[L] i_ =1[,][we][use][a] multi-layer perception (MLP) classifier to predict the probability of the corresponding original tokens _{m[i] j[}][|V|] j_ =1[=] _MLP_ ( _h[m] i_[)][.][The IRR objective can be formulated as:] 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/cvpr_2023_Cross-Modal_Implicit_Relation_Reasoning_and_Aligning_for_Text-to-Image_Person_Retrieval_images/cvpr_2023_Cross-Modal_Implicit_Relation_Reasoning_and_Aligning_for_Text-to-Image_Person_Retrieval.pdf-0005-08.png)


where _M_ denotes the set of masked text tokens and _|V|_ is the size of vocabulary _V_ . _m[i]_ is predicted token probability distribution and _y[i]_ is a one-hot vocabulary distribution where the ground-truth token has a probability of 1. 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/cvpr_2023_Cross-Modal_Implicit_Relation_Reasoning_and_Aligning_for_Text-to-Image_Person_Retrieval_images/cvpr_2023_Cross-Modal_Implicit_Relation_Reasoning_and_Aligning_for_Text-to-Image_Person_Retrieval.pdf-0005-10.png)


**----- Start of picture text -----**<br>
Feedforward Feedforward Feedforward M× Feedforward<br>M× Cross-Attn Cross-Attn M× Self-Attn<br>�� �� �� �� �� �� Self-Attn<br>�� Self-Attn�� �� �� Self-Attn�� �� �� ��/� ��/� �� ��/� ��/� �� Cross-Attn�� ��<br>Textual  Visual  Textual  Visual<br>Textual Feature Visual Feature Feature Feature Feature Feature<br>(a) Co-attention (b) Merged attention (c) Ours<br>**----- End of picture text -----**<br>


Figure 4. **Illustration of our multimodal interaction encoder and two other popular interaction modules.** (a) Co-attention, textual and visual features are fed into separate transformer blocks with self-attn and cross-attn independently to enable cross-modal interaction. (b) Merged attention, textual and visual features are concatenated together and then fed into a single transformer block. (c) Our multimodal interaction encoder, textual and visual features are first fused by a cross-attn layer and then fed into a single transformer block. 

## **3.3. Similarity Distribution Matching** 

We introduce a novel cross modal matching loss termed as Similarity Distribution Matching (SDM), which incorporates the cosine similarity distributions of the _N ×N_ imagetext pairs embeddings into KL divergence to associate the representations across different modalities. 

Given a mini-batch of N image-text pairs, for each image global representation _fi[v]_[,][we][construct][a][set][of][image-] text representation pairs as _{_ ( _fi[v][, f] j[ t]_[)] _[, y][i,j][}][N] j_ =1[,][where] _[y][i,j]_ is a true matching label, _yi,j_ = 1 means that ( _fi[v][, f] j[ t]_[)][is][a] matched pair from the same identity, while _yi,j_ = 0 indicates the unmatched pair. Let _sim_ ( **u** _,_ **v** ) = **u** _[⊤]_ **v** _/∥_ **u** _∥∥_ **v** _∥_ denotes the dot product between _L_ 2 normalized **u** and **v** ( _i.e_ . cosine similarity). Then the probability of matching pairs can be simply calculated with the following softmax function: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/cvpr_2023_Cross-Modal_Implicit_Relation_Reasoning_and_Aligning_for_Text-to-Image_Person_Retrieval_images/cvpr_2023_Cross-Modal_Implicit_Relation_Reasoning_and_Aligning_for_Text-to-Image_Person_Retrieval.pdf-0005-15.png)


where _τ_ is a temperature hyperparameter which controls the probability distribution peaks. The matching probability _pi,j_ can be viewed as the proportion of the cosine similarity score between _fi[v]_[and] _[ f] j[ t]_[to the sum of cosine similarity] score between _fi[v]_[and] _[{][f] j[ t][}][N] j_ =1[in][a][mini-batch.][Then][the] SDM loss from image to text in a mini-batch is computed by : 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/cvpr_2023_Cross-Modal_Implicit_Relation_Reasoning_and_Aligning_for_Text-to-Image_Person_Retrieval_images/cvpr_2023_Cross-Modal_Implicit_Relation_Reasoning_and_Aligning_for_Text-to-Image_Person_Retrieval.pdf-0005-17.png)


where _ϵ_ is a small number to avoid numerical problems, and _qi,j_ = _yi,j/_[�] _[N] k_ =1 _[y][i,k]_[is the true matching probability.] 

Symmetrically, the SDM loss from text to image _Lt_ 2 _i_ can be formulated by exchanging _f[v]_ and _f[t]_ in Eq.(4) (5), and the bi-directional SDM loss is calculated by: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/cvpr_2023_Cross-Modal_Implicit_Relation_Reasoning_and_Aligning_for_Text-to-Image_Person_Retrieval_images/cvpr_2023_Cross-Modal_Implicit_Relation_Reasoning_and_Aligning_for_Text-to-Image_Person_Retrieval.pdf-0005-20.png)


5 

|Method|Type<br>Ref<br>Image Enc.<br>Text Enc.|Rank-1<br>Rank-5<br>Rank-10<br>mAP<br>mINP|Rank-1<br>Rank-5<br>Rank-10<br>mAP<br>mINP|
|---|---|---|---|
|CMPM/C [53]<br>L<br>ECCV18<br>RN50<br>LSTM<br>49.37<br>-<br>TIMAM [36]<br>G<br>ICCV19<br>RN101<br>BERT<br>54.51<br>77.56<br>ViTAA [46]<br>L<br>ECCV20<br>RN50<br>LSTM<br>54.92<br>75.18<br>NAFS [12]<br>L<br>arXiv21<br>RN50<br>BERT<br>59.36<br>79.13<br>DSSL [55]<br>L<br>MM21<br>RN50<br>BERT<br>59.98<br>80.41<br>SSAN [7]<br>L<br>arXiv21<br>RN50<br>LSTM<br>61.37<br>80.15<br>LapsCore [49]<br>L<br>ICCV21<br>RN50<br>BERT<br>63.40<br>-<br>ISANet [51]<br>L<br>arXiv22<br>RN50<br>LSTM<br>63.92<br>82.15<br>LBUL [48]<br>L<br>MM22<br>RN50<br>BERT<br>64.04<br>82.66<br>Han et al. [13]<br>G<br>BMVC21<br>CLIP-RN101<br>CLIP-Xformer<br>64.08<br>81.73<br>SAF [28]<br>L<br>ICASSP22<br>ViT-Base<br>BERT<br>64.13<br>82.62<br>TIPCB [5]<br>L<br>Neuro22<br>RN50<br>BERT<br>64.26<br>83.19<br>CAIBC [47]<br>L<br>MM22<br>RN50<br>BERT<br>64.43<br>82.87<br>AXM-Net [10]<br>L<br>MM22<br>RN50<br>BERT<br>64.44<br>80.52<br>LGUR [38]<br>L<br>MM22<br>DeiT-Small<br>BERT<br>65.25<br>83.12<br>IVT [39]<br>G<br>ECCVW22<br>ViT-Base<br>BERT<br>65.59<br>83.11<br>CFine[50]<br>L<br>arXiv22<br>CLIP-ViT<br>BERT<br>69.57<br>85.93|||79.27<br>-<br>-<br>79.27<br>-<br>-<br>82.90<br>51.60<br>-<br>86.00<br>54.07<br>-<br>87.56<br>-<br>-<br>86.73<br>-<br>-<br>87.80<br>-<br>-<br>87.69<br>-<br>-<br>87.22<br>-<br>-<br>88.19<br>60.08<br>-<br>88.40<br>-<br>-<br>89.10<br>-<br>88.37<br>-<br>-<br>86.77<br>58.73<br>-<br>89.00<br>-<br>-<br>89.21<br>-<br>-<br>91.15<br>-<br>-|
|**Baseline (CLIP-RN50)**<br>**Baseline (CLIP-RN101)**<br>**Baseline (CLIP-ViT-B/16)**<br>**IRRA (Ours)**|G<br>-<br>CLIP-RN50<br>CLIP-Xformer<br>G<br>-<br>CLIP-RN101<br>CLIP-Xformer<br>G<br>-<br>CLIP-ViT<br>CLIP-Xformer<br>G<br>-<br>CLIP-ViT<br>CLIP-Xformer|57.26<br>78.57<br>60.27<br>80.88<br>68.19<br>86.47<br>**73.38**<br>**89.93**|85.58<br>50.88<br>34.44<br>87.88<br>53.93<br>37.54<br>91.47<br>61.12<br>44.86<br>**93.71**<br>**66.13**<br>**50.24**|



Table 1. Performance comparisons with state-of-the-art methods on CUHK-PEDES dataset. Results are ordered based on the Rank-1 accuracy. “G” and “L” in “Type” column stand for global-matching/local-matching method. 

**Optimization.** As mentioned previously, the main objective of IRRA is to improve the learning of global imagetext representations in joint embedding space. To achieve this goal, the commonly utilized ID loss [54] is also adopted along with SDM loss and IRR loss to optimize IRRA. The ID loss is a softmax loss which classifies an image or text into distinct groups based on their identities. It explicitly considers the intra-modal distance and ensures that feature representations of the same image/text group are closely clustered together in the joint embedding space. 

IRRA is trained in an end-to-end manner and the overall optimization objective for training is defined as: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/cvpr_2023_Cross-Modal_Implicit_Relation_Reasoning_and_Aligning_for_Text-to-Image_Person_Retrieval_images/cvpr_2023_Cross-Modal_Implicit_Relation_Reasoning_and_Aligning_for_Text-to-Image_Person_Retrieval.pdf-0006-04.png)


## **4. Experiments** 

We extensively evaluate our method on three challenging text-to-image person retrieval datasets. 

**CUHK-PEDES** [30] is the first dataset dedicated to textto-image person retrieval, which contains 40,206 images and 80,412 textual descriptions for 13,003 identities. Following the official data split, the training set consists of 11,003 identities, 34,054 images and 68,108 textual descriptions. The validation set and test set contain 3,078 and 3,074 images, 6158 and 6156 textual descriptions, respectively, and both of them have 1,000 identities. 

**ICFG-PEDES** [7] contains a total of 54,522 images for 4,102 identities. Each image has only one corresponding textual description. The dataset is divided into a training set and a test set, the former comprises 34,674 image-text pairs of 3,102 identities, while the latter contains 19,848 imagetext pairs for the remaining 1,000 identities. 

**RSTPReid** [55] contains 20505 images of 4,101 identities from 15 cameras. Each identity has 5 corresponding images taken by different cameras and each image is annotated with 2 textual descriptions. Following the official data split, the training, validation and test set contain 3701, 200 and 200 identities respectively. 

**Evaluation Metrics.** We adopt the popular Rank- _k_ metrics ( _k_ =1,5,10) as the primary evaluation metrics. Rank- _k_ reports the probability of finding at least one matching person image within the top-k candidate list when given a textual description as a query. In addition, for a comprehensive evaluation, we also adopt the mean Average Precision (mAP) and mean Inverse Negative Penalty(mINP) [52] as another retrieval criterion. The higher Rank- _k_ , mAP and mINP indicates better performance. 

**Implementation Details.** IRRA consists of a pre-trained image encoder, _i.e_ ., CLIP-ViT-B/16, a pre-trained text encoder, _i.e_ ., CLIP text Transformer, and a random-initialized multimodal interaction encoder. For each layer of the multimodal interaction encoder, the hidden size and number of heads are set to 512 and 8. During training, random horizontally flipping, random crop with padding, and random erasing are employed for image data augmentation. All input images are resized to 384 _×_ 128. The maximum length of the textual token sequence _L_ is set to 77. Our model is trained with Adam optimizer [24] for 60 epochs with a learning rate initialized to 1 _×_ 10 _[−]_[5] and cosine learning rate decay. At the beginning, we spend 5 warm-up epochs linearly increasing the learning rate from 1 _×_ 10 _[−]_[6] to 1 _×_ 10 _[−]_[5] . For random-initialized modules, we set the initial learning rate to 5 _×_ 10 _[−]_[5] . The temperature parameter _τ_ in SDM 

6 

loss is set to 0.02. This work is supported by Huawei MindSpore [19]. We perform our experiments on a single RTX3090 24GB GPU. 

## **4.1. Comparison with State-of-the-Art Methods** 

In this section, we present comparison results with stateof-the-art methods on three public benchmark datasets. Note that the Baseline models in Tab. 1 2 and 3 denotes different CLIP models fine-tuned with the original CLIP loss (InfoNCE [34]). 

**Performance Comparisons on CUHK-PEDES** We first evaluate the proposed method on the most common benchmark, CUHK-PEDES. As shown in Tab. 1, IRRA outperforms all state-of-the-art methods, achieving 73.38% Rank-1 accuracy and 66.13% mAP respectively. It is worth noting that our directly fine-tuned CLIP Baseline has already achieved the recent state-of-the-art method CFine [50], with Rank-1 accuracy and mAP reaching 68.19% and 86.47% respectively. In Tab. 1, we annotate the feature extraction backbones (”Image Enc.” and ”Text Enc.” column) employed by each method, and it is evident that there is a growing demand of powerful feature extraction backbone for text-to-image person retrieval, with transformer-based backbone becoming progressively dominant. 

**Performance Comparisons on ICFG-PEDES** The experimental results on the ICFG-PEDES dataset are reported in Tab. 2. The Baseline can achieve comparable results to recent state-of-the-art methods, with 56.74%, 75.72% and 82.26% on Rank-1, Rank-5 and Rank-10, respectively. Moreover, our proposed IRRA achieves 63.46%, 80.24% and 85.82% on these metrics, which exceed the recent stateof-the-art local-matching method Cfine [50] by a large margin, _i.e_ ., +2.63%, +3.69% and +3.4%. It is worth noting that the mINP [52] metric on ICFG-PEDES is relatively low, which indicates the inferior capability of IRRA to find the hardest matching samples. 

**Performance Comparisons on RSTPReid** We also report our experimental results on the newly released RSTPReid dataset in Tab. 3. Our proposed IRRA dramatically surpass the recent global-matching method IVT [39] by +13.5%, +11.3% and +9.4% on Rank-1, Rank-5 and Rank10, respectively. Compared with the recent local-matching method Cfine [50], IRRA also achieves considerable performance gains, with the rise of +9.65%, +8.8% and +6.6% on Rank-1, Rank-5 and Rank-10, respectively. 

In summary, our IRRA consistently achieves the best performance for all metrics on all three benchmark datasets. This demonstrates the generalization and robustness of our proposed method. 

## **4.2. Ablation Study** 

In this subsection, we analyze the effectiveness of each component in the IRRA framework. Here, we adopt the 

|Method|Type|Rank-1<br>Rank-5<br>Rank-10<br>mAP<br>mINP|
|---|---|---|
|Dual Path [54]<br>CMPM/C [53]<br>ViTAA [46]<br>SSAN [7]<br>IVT [39]<br>ISANet [51]<br>CFine[50]|G<br>L<br>L<br>L<br>G<br>L<br>L|38.99<br>59.44<br>68.41<br>-<br>-<br>43.51<br>65.44<br>74.26<br>-<br>-<br>50.98<br>68.79<br>75.78<br>-<br>-<br>54.23<br>72.63<br>79.53<br>-<br>-<br>56.04<br>73.60<br>80.22<br>-<br>-<br>57.73<br>75.42<br>81.72<br>-<br>-<br>60.83<br>76.55<br>82.42<br>-<br>-|
|**Baseline (CLIP-RN50)**<br>**Baseline (CLIP-RN101)**<br>**Baseline (CLIP-ViT-B/16)**<br>**IRRA (Ours)**|G<br>G<br>G<br>G|41.46<br>63.68<br>73.04<br>21.00<br>2.46<br>44.09<br>66.27<br>74.75<br>22.59<br>2.84<br>56.74<br>75.72<br>82.26<br>31.84<br>5.03<br>**63.46**<br>**80.25**<br>**85.82**<br>**38.06**<br>**7.93**|



Table 2. Performance comparisons with state-of-the-art methods on ICFG-PEDES dataset. 

|Method|Type|Rank-1<br>Rank-5<br>Rank-10<br>mAP<br>mINP|
|---|---|---|
|DSSL [55]<br>SSAN [7]<br>LBUL [48]<br>IVT [39]<br>CFine[50]|G<br>L<br>L<br>G<br>L|39.05<br>62.60<br>73.95<br>-<br>-<br>43.50<br>67.80<br>77.15<br>-<br>-<br>45.55<br>68.20<br>77.85<br>-<br>-<br>46.70<br>70.00<br>78.80<br>-<br>-<br>50.55<br>72.50<br>81.60<br>-<br>-|
|**Baseline (CLIP-RN50)**<br>**Baseline (CLIP-RN101)**<br>**Baseline (CLIP-ViT-B/16)**<br>**IRRA (Ours)**|G<br>G<br>G<br>G|41.40<br>68.55<br>77.95<br>31.51<br>12.71<br>43.45<br>67.75<br>78.40<br>29.91<br>11.18<br>54.05<br>80.70<br>88.00<br>43.41<br>22.31<br>**60.20**<br>**81.30**<br>**88.20**<br>**47.17**<br>**25.28**|



Table 3. Performance comparisons with state-of-the-art methods on RSTPReid dataset. 

CLIP-ViT-B/16 model fine-tuned with InfoNCE loss as the Baseline to facilitate the ablation study. 

**Ablations on proposed components** To fully demonstrate the impact of different components in IRRA, we conduct a comprehensive empirical analysis on three public datasets ( _i.e_ ., CUHK-PEDES [30], ICFG-PEDES [7] and RSTPReid [55]). The Rank-1, Rank-5, Rank-10 accuracies (%) are reported in Tab. 4. 

IRR learns local relations through MLM task which can be easily integrated with other transformer-based methods to facilitate fine-grained cross-modal alignment. The efficacy of IRR is revealed via the experimental results of No.0 _vs_ . No.4, No.2 _vs_ . No.6 and No.5 _vs_ . No.7. Merely adding the IRR to Baseline improves the Rank-1 accuracy by 3.04%, 4.22% and 3.85% on the three datasets, respectively. The above results clearly show that IRR module are beneficial for cross-modal matching. 

To demonstrate the effectiveness of our proposed similarity distribution matching (SDM) loss, we compare it with the commonly used cross-modal projection matching (CMPM) loss [53] (No.1 _vs_ . No.2) on the three public datasets, the SDM loss promotes the Rank-1 accuracy of the CMPM loss by 11.11%, 6.62%, and 2.2%, respectively. Besides, replace the original InfoNCE loss with the commonly used CMPM loss (No.0 _vs_ . No.1) does not improve the performance on text-to-image person retrieval, yet it leads to performance degradation. In contrast, the SDM loss promotes the Rank-1 accuracy of the Baseline by 2.23%, 3.71%, and 3.15% on three datasets, respectively. These results demonstrate that the proposed SDM loss well aligns the features representations between the two modalities. In 

7 

|No.|Methods|Components|CUHK-PEDES|CUHK-PEDES|ICFG-PEDES|RSTPReid|
|---|---|---|---|---|---|---|
|||SDM<br>_Lid_<br>IRR|Rank-1<br>Rank-5<br>Rank-10||Rank-1<br>Rank-5<br>Rank-10|Rank-1<br>Rank-5<br>Rank-10|
|0<br>1<br>2<br>3<br>4<br>5<br>6|Baseline<br>+_Lcmpm_[53]<br>+SDM<br>+_Lid_<br>+IRR<br>+SDM+_Lid_<br>+SDM +IRR|✓<br>✓<br>✓<br>✓<br>✓<br>✓<br>✓|68.19<br>59.31<br>70.42<br>65.33<br>71.23<br>70.52<br>72.81|86.47<br>91.47<br>79.66<br>86.11<br>86.73<br>92.04<br>84.05<br>90.33<br>88.89<br>93.24<br>87.59<br>92.12<br>89.31<br>93.39|56.74<br>75.72<br>82.26<br>53.83<br>72.20<br>79.02<br>60.45<br>77.88<br>83.86<br>53.38<br>72.70<br>79.70<br>60.96<br>79.02<br>84.90<br>61.03<br>78.26<br>83.89<br>63.27<br>80.10<br>85.77|54.05<br>80.70<br>88.00<br>55.40<br>77.70<br>85.25<br>57.20<br>79.90<br>88.10<br>54.15<br>76.65<br>85.00<br>57.90<br>80.85<br>88.50<br>58.65<br>80.70<br>87.05<br>59.25<br>79.70<br>88.00|
|7|IRRA|✓<br>✓<br>✓|**73.38**|**89.83**<br>**93.71**|**63.46**<br>**80.25**<br>**85.82**|**60.20**<br>**81.30**<br>**88.20**|



Table 4. Ablation study on each component of IRRA on CUHK-PEDES, ICFG-PEDES and RSTPReid. 

|Method|Param(M)|Time(ms)|Rank-1|Rank-5|Rank-10|
|---|---|---|---|---|---|
|_Co-attn_|33.62|24.30|73.28|89.04|93.44|
|_Merged attn_|**12.61**|19.20|73.21|89.18|93.70|
|Ours|13.66|**6.42**|**73.38**|**89.83**|**93.71**|



Table 5. Comparisons between different Multimodal Interaction Module of IRRA on CUHK-PEDES. 

addition, the experimental results of No.2 _vs_ . No.5 and No.6 _vs_ . No.7 demonstrate the effectiveness of the ID loss. 

**Analysis of the Multimodal Interaction Encoder** To demonstrate the advantages of our proposed Multimodal Interaction Module, we compare it with two other popular multimodal interaction modules in Tab. 5. The Multimodal Interaction Module in IRR is a computationally efficient operation to fuse multimodal features, building the connection between the two modalities. We extensively compare it with _Co-attn_ and _Merged attn_ under our proposed IRRA setting, and observe slight but consistent performance gain on all Rank- _k_ metrics. Notably, our major advantage is the computational efficiency. 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/cvpr_2023_Cross-Modal_Implicit_Relation_Reasoning_and_Aligning_for_Text-to-Image_Person_Retrieval_images/cvpr_2023_Cross-Modal_Implicit_Relation_Reasoning_and_Aligning_for_Text-to-Image_Person_Retrieval.pdf-0008-06.png)


**----- Start of picture text -----**<br>
A man wearing a<br>white and gray stripe<br>shirt, a pair of green<br>pants and a pair of<br>shoes.<br>The man is crossing<br>the street wearing a<br>white shirt, black<br>pants and carrying a<br>black book bag.<br>The woman wears a<br>teal t-shirt,  black<br>pants, and blue tennis<br>shoes. She wears<br>glasses and carries a<br>red backpack.<br>**----- End of picture text -----**<br>


Figure 5. Comparison of top-10 retrieved results on CUHKPEDES between Baseline (the first row) and IRRA (the second row) for each text query. The image corresponding to query text., matched and mismatched images are marked with black, green and red rectangles, respectively. 

## **4.3. Qualitative Results** 

Fig. 5 compares the top-10 retrieval results from the Baseline and our proposed IRRA. As the figure shows, IRRA achieves much more accurate retrieval results and obtains accurate retrieval results when Baseline fails to retrieve them. This is mainly due to the Implicit Relation Reasoning (IRR) modules we designed, which fully exploit fine-grained discriminative clues to distinguish different pedestrians. This is illustrated in the orange highlighted text and image regions box in Fig. 5. Moreover, We found that our model only learns the semantic information of the word-level but unable to understand the semantics of the phrase-level in the description text, which leads to the distortion of semantic information. This is because we only masked random single tokens in MLM, and did not perform phrase-level masking. We plan to address this issue in the future. 

## **5. Conclusion** 

In this paper, we introduce a cross modal implicit relation reasoning and aligning framework(IRRA) to learn discriminative global image-text representations. To achieve full cross-modal interaction, we propose an Implicit Relation Reasoning module that exploits MLM to mine finegrained relations between visual and textual tokens. We further propose a Similarity Distribution Matching loss to effectively enlarge the variance between non-matching pairs and the correlation between matching pairs. These modules collaborate to align images and text into a joint embedding space. Significant performance gains on three popular benchmarks datasets prove the superiority and effectiveness of our proposed IRRA framework. We believe that the CLIP-based approach will be the future trend for text-toimage person retrieval. 

**Acknowledgement.** This work is partially supported by the Key Research and Development Program of Hubei Province (2021BAA187), National Natural Science Foundation of China under Grant (62176188), Zhejiang lab (NO.2022NF0AB01), the Special Fund of Hubei Luojia Laboratory (220100015) and CAAI-Huawei MindSpore Open Fund. 

8 

## **References** 

- [1] Stanislaw Antol, Aishwarya Agrawal, Jiasen Lu, Margaret Mitchell, Dhruv Batra, C Lawrence Zitnick, and Devi Parikh. Vqa: Visual question answering. In _Proceedings of the IEEE international conference on computer vision_ , pages 2425– 2433, 2015. 3 

- [2] Tianlang Chen, Chenliang Xu, and Jiebo Luo. Improving text-based person search by spatial matching and adaptive threshold. In _2018 IEEE Winter Conference on Applications of Computer Vision (WACV)_ , pages 1879–1887. IEEE, 2018. 3 

- [3] Xinlei Chen, Hao Fang, Tsung-Yi Lin, Ramakrishna Vedantam, Saurabh Gupta, Piotr Doll´ar, and C Lawrence Zitnick. Microsoft coco captions: Data collection and evaluation server. _arXiv preprint arXiv:1504.00325_ , 2015. 3 

- [4] Yucheng Chen, Rui Huang, Hong Chang, Chuanqi Tan, Tao Xue, and Bingpeng Ma. Cross-modal knowledge adaptation for language-based person search. _IEEE Transactions on Image Processing_ , 30:4057–4069, 2021. 3 

- [5] Yuhao Chen, Guoqing Zhang, Yujiang Lu, Zhenxing Wang, and Yuhui Zheng. Tipcb: A simple but effective part-based convolutional baseline for text-based person search. _Neurocomputing_ , 494:171–181, 2022. 2, 3, 6 

- [6] Yen-Chun Chen, Linjie Li, Licheng Yu, Ahmed El Kholy, Faisal Ahmed, Zhe Gan, Yu Cheng, and Jingjing Liu. Uniter: Universal image-text representation learning. In _European conference on computer vision_ , pages 104–120. Springer, 2020. 2, 3 

- [7] Zefeng Ding, Changxing Ding, Zhiyin Shao, and Dacheng Tao. Semantically self-aligned network for text-toimage part-aware person re-identification. _arXiv preprint arXiv:2107.12666_ , 2021. 2, 3, 6, 7 

- [8] Alexey Dosovitskiy, Lucas Beyer, Alexander Kolesnikov, Dirk Weissenborn, Xiaohua Zhai, Thomas Unterthiner, Mostafa Dehghani, Matthias Minderer, Georg Heigold, Sylvain Gelly, et al. An image is worth 16x16 words: Transformers for image recognition at scale. In _International Conference on Learning Representations_ , 2020. 3 

- [9] Zi-Yi Dou, Yichong Xu, Zhe Gan, Jianfeng Wang, Shuohang Wang, Lijuan Wang, Chenguang Zhu, Pengchuan Zhang, Lu Yuan, Nanyun Peng, et al. An empirical study of training end-to-end vision-and-language transformers. In _Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition_ , pages 18166–18176, 2022. 2, 3, 4 

- [10] Ammarah Farooq, Muhammad Awais, Josef Kittler, and Syed Safwan Khalid. Axm-net: Implicit cross-modal feature alignment for person re-identification. 36(4):4477–4485, 2022. 3, 6 

- [11] Zhiyi Fu, Wangchunshu Zhou, Jingjing Xu, Hao Zhou, and Lei Li. Contextual representation learning beyond masked language modeling. In _Proceedings of the 60th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)_ , pages 2701–2714, 2022. 4 

- [12] Chenyang Gao, Guanyu Cai, Xinyang Jiang, Feng Zheng, Jun Zhang, Yifei Gong, Pai Peng, Xiaowei Guo, and Xing Sun. Contextual non-local alignment over full-scale rep- 

   - resentation for text-based person search. _arXiv preprint arXiv:2101.03036_ , 2021. 6 

- [13] Xiao Han, Sen He, Li Zhang, and Tao Xiang. Textbased person search with limited data. _arXiv preprint arXiv:2110.10807_ , 2021. 2, 3, 6 

- [14] Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun. Deep residual learning for image recognition. In _Proceedings of the IEEE conference on computer vision and pattern recognition_ , pages 770–778, 2016. 3 

- [15] Shuting He, Hao Luo, Pichao Wang, Fan Wang, Hao Li, and Wei Jiang. Transreid: Transformer-based object reidentification. In _Proceedings of the IEEE/CVF international conference on computer vision_ , pages 15013–15022, 2021. 1 

- [16] Lisa Anne Hendricks, John Mellor, Rosalia Schneider, JeanBaptiste Alayrac, and Aida Nematzadeh. Decoupling the role of data, attention, and losses in multimodal transformers. _Transactions of the Association for Computational Linguistics_ , 9:570–585, 2021. 4 

- [17] Geoffrey Hinton, Oriol Vinyals, Jeff Dean, et al. Distilling the knowledge in a neural network. 2 

- [18] Sepp Hochreiter and J¨urgen Schmidhuber. Long short-term memory. _Neural computation_ , 9(8):1735–1780, 1997. 3 

- [19] Huawei. Mindspore, https://www.mindspore.cn/, 2020. 7 

- [20] Chao Jia, Yinfei Yang, Ye Xia, Yi-Ting Chen, Zarana Parekh, Hieu Pham, Quoc Le, Yun-Hsuan Sung, Zhen Li, and Tom Duerig. Scaling up visual and vision-language representation learning with noisy text supervision. In _International Conference on Machine Learning_ , pages 4904–4916. PMLR, 2021. 3 

- [21] Ya Jing, Chenyang Si, Junbo Wang, Wei Wang, Liang Wang, and Tieniu Tan. Pose-guided multi-granularity attention network for text-based person search. In _Proceedings of the AAAI Conference on Artificial Intelligence_ , volume 34, pages 11189–11196, 2020. 2 

- [22] Jacob Devlin Ming-Wei Chang Kenton and Lee Kristina Toutanova. Bert: Pre-training of deep bidirectional transformers for language understanding. In _Proceedings of NAACL-HLT_ , pages 4171–4186, 2019. 3 

- [23] Wonjae Kim, Bokyung Son, and Ildoo Kim. Vilt: Visionand-language transformer without convolution or region supervision. In _International Conference on Machine Learning_ , pages 5583–5594. PMLR, 2021. 3 

- [24] Diederik P Kingma and Jimmy Ba. Adam: A method for stochastic optimization. In _ICLR (Poster)_ , 2015. 6 

- [25] Ryan Kiros, Ruslan Salakhutdinov, and Richard S Zemel. Unifying visual-semantic embeddings with multimodal neural language models. _arXiv preprint arXiv:1411.2539_ , 2014. 3 

- [26] Jie Lei, Xinlei Chen, Ning Zhang, Mengjiao Wang, Mohit Bansal, Tamara L Berg, and Licheng Yu. Loopitr: Combining dual and cross encoder architectures for image-text retrieval. _arXiv preprint arXiv:2203.05465_ , 2022. 1 

- [27] Junnan Li, Ramprasaath Selvaraju, Akhilesh Gotmare, Shafiq Joty, Caiming Xiong, and Steven Chu Hong Hoi. Align before fuse: Vision and language representation learning with momentum distillation. _Advances in neural information processing systems_ , 34:9694–9705, 2021. 2 

9 

- [28] Shiping Li, Min Cao, and Min Zhang. Learning semanticaligned feature representation for text-based person search. In _ICASSP 2022-2022 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)_ , pages 2724–2728. IEEE, 2022. 6 

- [29] Shuang Li, Tong Xiao, Hongsheng Li, Wei Yang, and Xiaogang Wang. Identity-aware textual-visual matching with latent co-attention. In _Proceedings of the IEEE International Conference on Computer Vision_ , pages 1890–1899, 2017. 3 

- [30] Shuang Li, Tong Xiao, Hongsheng Li, Bolei Zhou, Dayu Yue, and Xiaogang Wang. Person search with natural language description. In _Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition_ , pages 1970– 1979, 2017. 1, 3, 6, 7 

- [31] Jiasen Lu, Dhruv Batra, Devi Parikh, and Stefan Lee. Vilbert: Pretraining task-agnostic visiolinguistic representations for vision-and-language tasks. _Advances in neural information processing systems_ , 32, 2019. 2 

- [32] Hao Luo, Youzhi Gu, Xingyu Liao, Shenqi Lai, and Wei Jiang. Bag of tricks and a strong baseline for deep person re-identification. In _Proceedings of the IEEE/CVF conference on computer vision and pattern recognition workshops_ , pages 0–0, 2019. 1 

- [33] Antoine Miech, Jean-Baptiste Alayrac, Ivan Laptev, Josef Sivic, and Andrew Zisserman. Thinking fast and slow: Efficient text-to-visual retrieval with transformers. In _Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition_ , pages 9826–9836, 2021. 1 

- [34] Aaron van den Oord, Yazhe Li, and Oriol Vinyals. Representation learning with contrastive predictive coding. _arXiv preprint arXiv:1807.03748_ , 2018. 7 

- [35] Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, et al. Learning transferable visual models from natural language supervision. In _International Conference on Machine Learning_ , pages 8748–8763. PMLR, 2021. 2, 3, 4 

- [36] Nikolaos Sarafianos, Xiang Xu, and Ioannis A Kakadiaris. Adversarial representation learning for text-to-image matching. In _Proceedings of the IEEE/CVF international conference on computer vision_ , pages 5814–5824, 2019. 3, 6 

- [37] Rico Sennrich, Barry Haddow, and Alexandra Birch. Neural machine translation of rare words with subword units. _arXiv preprint arXiv:1508.07909_ , 2015. 4 

- [38] Zhiyin Shao, Xinyu Zhang, Meng Fang, Zhifeng Lin, Jian Wang, and Changxing Ding. Learning granularity-unified representations for text-to-image person re-identification. _arXiv preprint arXiv:2207.07802_ , 2022. 2, 3, 6 

- [39] Xiujun Shu, Wei Wen, Haoqian Wu, Keyu Chen, Yiran Song, Ruizhi Qiao, Bo Ren, and Xiao Wang. See finer, see more: Implicit modality alignment for text-based person retrieval. _arXiv preprint arXiv:2208.08608_ , 2022. 2, 6, 7 

- [40] Karen Simonyan and Andrew Zisserman. Very deep convolutional networks for large-scale image recognition. _arXiv preprint arXiv:1409.1556_ , 2014. 3 

- [41] Weijie Su, Xizhou Zhu, Yue Cao, Bin Li, Lewei Lu, Furu Wei, and Jifeng Dai. Vl-bert: Pre-training of generic visual- 

linguistic representations. _arXiv preprint arXiv:1908.08530_ , 2019. 2, 3 

- [42] Siqi Sun, Yen-Chun Chen, Linjie Li, Shuohang Wang, Yuwei Fang, and Jingjing Liu. Lightningdot: Pre-training visualsemantic embeddings for real-time image-text retrieval. In _Proceedings of the 2021 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies_ , pages 982–997, 2021. 1 

- [43] Wilson L Taylor. “cloze procedure”: A new tool for measuring readability. _Journalism quarterly_ , 30(4):415–433, 1953. 4 

- [44] Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez, Łukasz Kaiser, and Illia Polosukhin. Attention is all you need. _Advances in neural information processing systems_ , 30, 2017. 3, 4 

- [45] Haochen Wang, Jiayi Shen, Yongtuo Liu, Yan Gao, and Efstratios Gavves. Nformer: Robust person re-identification with neighbor transformer. In _Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition_ , pages 7297–7307, 2022. 1 

- [46] Zhe Wang, Zhiyuan Fang, Jun Wang, and Yezhou Yang. Vitaa: Visual-textual attributes alignment in person search by natural language. In _European Conference on Computer Vision_ , pages 402–420. Springer, 2020. 2, 3, 6, 7 

- [47] Zijie Wang, Aichun Zhu, Jingyi Xue, Xili Wan, Chao Liu, Tian Wang, and Yifeng Li. Caibc: Capturing all-round information beyond color for text-based person retrieval. _arXiv preprint arXiv:2209.05773_ , 2022. 3, 6 

- [48] Zijie Wang, Aichun Zhu, Jingyi Xue, Xili Wan, Chao Liu, Tian Wang, and Yifeng Li. Look before you leap: Improving text-based person retrieval by learning a consistent crossmodal common manifold. In _Proceedings of the 30th ACM International Conference on Multimedia_ , pages 1984–1992, 2022. 6, 7 

- [49] Yushuang Wu, Zizheng Yan, Xiaoguang Han, Guanbin Li, Changqing Zou, and Shuguang Cui. Lapscore: Languageguided person search via color reasoning. In _Proceedings of the IEEE/CVF International Conference on Computer Vision_ , pages 1624–1633, 2021. 3, 6 

- [50] Shuanglin Yan, Neng Dong, Liyan Zhang, and Jinhui Tang. Clip-driven fine-grained text-image person re-identification. _arXiv preprint arXiv:2210.10276_ , 2022. 2, 3, 6, 7 

- [51] Shuanglin Yan, Hao Tang, Liyan Zhang, and Jinhui Tang. Image-specific information suppression and implicit local alignment for text-based person search. _arXiv preprint arXiv:2208.14365_ , 2022. 3, 6, 7 

- [52] Mang Ye, Jianbing Shen, Gaojie Lin, Tao Xiang, Ling Shao, and Steven CH Hoi. Deep learning for person reidentification: A survey and outlook. _IEEE transactions on pattern analysis and machine intelligence_ , 44(6):2872–2893, 2021. 6, 7 

- [53] Ying Zhang and Huchuan Lu. Deep cross-modal projection learning for image-text matching. In _Proceedings of the European conference on computer vision (ECCV)_ , pages 686– 701, 2018. 2, 3, 6, 7, 8 

- [54] Zhedong Zheng, Liang Zheng, Michael Garrett, Yi Yang, Mingliang Xu, and Yi-Dong Shen. Dual-path convolutional 

10 

image-text embeddings with instance loss. _ACM Transactions on Multimedia Computing, Communications, and Applications (TOMM)_ , 16(2):1–23, 2020. 2, 6, 7 

- [55] Aichun Zhu, Zijie Wang, Yifeng Li, Xili Wan, Jing Jin, Tian Wang, Fangqiang Hu, and Gang Hua. Dssl: Deep surroundings-person separation learning for text-based person retrieval. In _Proceedings of the 29th ACM International Conference on Multimedia_ , pages 209–217, 2021. 3, 6, 7 

11 

