Published as a conference paper at ICLR 2024 

## - IMAGE2SENTENCE BASED ASYMMETRIC ZERO SHOT COMPOSED IMAGE RETRIEVAL 

**Yongchao Du**[1] **, Min Wang**[2] _[∗]_ **, Wengang Zhou**[1] _[,]_[2] _[∗]_ **, Shuping Hui**[1] **, Houqiang Li**[1] _[,]_[2] 1CAS Key Laboratory of Technology in GIPAS, University of Science and Technology of China 2Institute of Artificial Intelligence, Hefei Comprehensive National Science Center _{_ ycdu2020,huisp _}_ @mail.ustc.edu.cn, wangmin@iai.ustc.edu.cn, _{_ zhwg,lihq _}_ @ustc.edu.cn 

## ABSTRACT 

The task of composed image retrieval (CIR) aims to retrieve images based on the query image and the text describing the users’ intent. Existing methods have made great progress with the advanced large vision-language (VL) model in CIR task, however, they generally suffer from two main issues: lack of labeled triplets for model training and difficulty of deployment on resource-restricted environments when deploying the large vision-language model. To tackle the above problems, we propose Image2Sentence based Asymmetric zero-shot composed image retrieval (ISA), which takes advantage of the VL model and only relies on unlabeled images for composition learning. In the framework, we propose a new adaptive token learner that maps an image to a sentence in the word embedding space of VL model. The sentence adaptively captures discriminative visual information and is further integrated with the text modifier. An asymmetric structure is devised for flexible deployment, in which the lightweight model is adopted for the query side while the large VL model is deployed on the gallery side. The global contrastive distillation and the local alignment regularization are adopted for the alignment between the light model and the VL model for CIR task. Our experiments demonstrate that the proposed ISA could better cope with the real retrieval scenarios and further improve retrieval accuracy and efficiency. 

## 1 INTRODUCTION 

Image retrieval is a crucial task for computer vision, and plays a fundamental role in a variety of applications, _e.g._ , product retrieval Dong et al. (2022); Zhan et al. (2021); Liu et al. (2016); Zhou et al. (2010), landmark retrieval Radenovi´c et al. (2018), and person re-identification Zheng et al. (2016); Gheissari et al. (2006). Composed image retrieval (CIR) is presented as a more challenging task that requires accurate understanding of the users’ intentions. CIR allows multi-modal queries and aims to retrieve images that are both visually similar to the given query image and match with the text modifier, which enables users to conduct more precise and flexible search. However, to learn the composition ability, it requires labeled triplets of reference image, target image and text modifier Liu et al. (2021); Wu et al. (2021), which are time-consuming and expensive to collect. 

To eliminate the limitation of data scarcity, zero-shot composed image retrieval (ZSCIR) is proposed in Saito et al. (2023); Baldrati et al. (2023a), which converts the CIR task to text-to-image retrieval by mapping the image into a single word and combining it with text modifier. However, they all adopt a symmetric retrieval setting based on CLIP models Radford et al. (2021), _i.e._ , using large models to extract both query and database features, and fail to consider the real-life application where the model may be deployed on a resource-limited platform such as mobile device, etc. Moreover, the mapping is implemented by an MLP that projects image feature to a single word vector, _i.e._ , representing image as a single word in the word embedding space. This may be insufficient as a faithful representation for an image, and the heterogeneity of word embedding space and visual feature space may further dampen the discriminative information of image feature after projection, thus constraining its potential. 

> _∗_ Corresponding authors: Min Wang and Wengang Zhou 

1 

Published as a conference paper at ICLR 2024 

To address the above issues, we propose Image2Sentence based Asymmetric ZSCIR that utilizes different models to extract query and database features. Specifically, a lightweight model is adopted for the query side, which is usually user’s terminal device with limited computation resources; the large VL model is employed for the gallery side, which is usually deployed on a cloud platform. Asymmetric structures often suffer from two core issues: inferior representation capability and misalignment between the lightweight model and the larger model. To this end, we propose an adaptive token learner to realize the alignment between the lightweight visual model and the large visual language model, which could further improve the representation discrimination ability of the lightweight model. Specifically, the lightweight model extracts the visual feature map of an input image, then the adaptive token learner extracts visual tokens from visual feature map, and maps them to a more informative sentence in the word embedding space. By adaptively focusing on more distinctive visual patterns, our adaptive token learner serves as a semantic filter that automatically filters out trivial background noise and preserves discriminative visual information. The mapped sentence is learned to cope with the visual feature from the large visual encoder under the global contrastive distillation (GCD), as a constraint to build up semantic alignment between the large model and the lightweight model. Therefore during inference, the mapped sentence is directly concatenated with the modifier text as a caption to search for images. To better learn the image2sentence mapping and retain more visual information, we further leverage the local alignment regularization (LAR) to help the mapped sentence capture more distinctive visual concepts, and to be closer to the real caption that faithfully describes the image. Our asymmetric framework allows for more flexible deployment while also enhancing performance compared with symmetric setting, and outperforms the state-of-the-art methods on three benchmarks. 

## 2 RELATED WORK 

## 2.1 COMPOSED IMAGE RETRIEVAL 

CIR is the task of retrieving images that correspond to the reference image with a modifier text, and has been investigated in several domains, _e.g._ , natural scene Baldrati et al. (2023a); Liu et al. (2021), fashion domain Dodds et al. (2022); Lee et al. (2021); Wu et al. (2021), and synthetic domain Gu et al. (2023); Vo et al. (2019). Supervised learning serves as a stepping stone in the progression of CIR, and is first introduced in Vo et al. (2019), in which meticulously annotated (reference image, text modifier, target image) triplets are adopted for training. Supervised methods mainly follow the pipeline that merges the features of reference image and text modifier to align with the target image feature, as a result, their innovations generally revolve around jointly capturing the information from both visual and textual modality Vo et al. (2019); Lee et al. (2021); Chen et al. (2020). Moreover, some methods further improve the performance by adopting multimodal foundation model Baldrati et al. (2021; 2022a;b; 2023b); Levy et al. (2023) or incorporating pretraining-finetuning strategy Dodds et al. (2022); Han et al. (2022; 2023). 

However, the above supervised methods heavily rely on the triplet training dataset, which involves great difficulty in collecting and labeling in large scale. A new task of zero-shot composed image retrieval (ZSCIR) Saito et al. (2023) is proposed to liberate the model learning from meticulous triplet-labeled data, and relies only on image-text pairs. Saito et al. (2023) propose to use MLP to convert CLIP visual feature to a single word in the word embedding space, and adopt a cycle contrastive loss for training; Baldrati et al. (2023a) propose a two-stage framework including text inversion and distillation to convert image into text domain. However, the above methods rely on symmetric retrieval, in which heavy models are utilized to extract features for both query and database, and are difficult to deploy in practical retrieval application scenarios. We propose a ZSCIR framework, which enables flexible deployment on resource-limited scenarios by adopting the lightweight model. Different from previous works, instead of converting image to a single word, we propose the adaptive token learner to filter out more discriminative visual information, which is further mapped to a more informative sentence. Apart from global contrastive loss, a dense local alignment regularization is applied on mapped sentences to preserve the visual information during conversion. Our work shows its superiority in both retrieval accuracy and deployment flexibility. 

## 2.2 ASYMMETRIC RETRIEVAL 

In recent times, the adoption of expansive deep models has greatly improved the multi-modal comprehension ability Jia et al. (2021); Li et al. (2021; 2022; 2023); Radford et al. (2021). However, 

2 

Published as a conference paper at ICLR 2024 


![](raw/papers/2_LLM_NLP/LLM_paper_list/zero-shot/iclr_2024_Image2Sentence_based_Asymmetrical_Zero-shot_Composed_Image_Retrieval_images/iclr_2024_Image2Sentence_based_Asymmetrical_Zero-shot_Composed_Image_Retrieval.pdf-0003-01.png)


**----- Start of picture text -----**<br>
Training framework Adaptive token learner<br>𝒕𝒕 𝒗𝒗 𝑨𝑨= [𝒂𝒂 [(1)] , 𝒂𝒂 [(2)] ,···,𝒂𝒂 [(𝐿𝐿)] ]<br>GCD loss<br>Adaptive<br>token leaner LAR loss 𝒁𝒁𝒐𝒐𝒖𝒖𝒕𝒕 𝒁𝒁𝒓𝒓<br>𝑭𝑭 [𝐿] 𝑭𝑭 [𝑣𝑣]<br>Large  Large      Large<br>EncoderLight EncoderText  Matching Encoder EncoderVisual  𝑭𝑭 [𝐿] H<br>⊗ L<br>W<br>A  photo of C 𝒁𝒁= [𝒛𝒛1,𝒛𝒛2,···, 𝒛𝒛𝐿𝐿] 𝒁𝒁𝒄𝒄 L<br>𝑼𝑼= [𝒖𝒖1, 𝒖𝒖2,···,𝒖𝒖𝐿𝐿] …<br>Augment [𝒇𝒇𝐿ℎ,𝑤𝑤]1,1𝐻𝐻,𝑊𝑊 𝑼𝑼= [𝒖𝒖1,𝒖𝒖2,···, 𝒖𝒖𝐿𝐿]<br>𝒓𝒓𝑸𝑸<br>Conv2d<br>𝒓𝒓𝑲𝑲<br>𝒓𝒓𝑽𝑽 Self-attention feed forward Add & norm &<br>𝒄𝒄𝑸𝑸<br>𝒄𝒄𝑲𝑲<br>𝒄𝒄𝑽𝑽 Cross-attention feed forward Add & norm &<br>Flatten<br>**----- End of picture text -----**<br>


Figure 1: **Left** : our training framework. Large visual encoder, large text encoder and matching encoder of pre-trained foundation model are frozen during training. The framework is trained with global contrastive distillation (GCD) and local alignment regularization (LAR). **Right** : adaptive token learner. Adaptive token learner learns to adaptively select salient visual patterns and map them to a sentence in the word embedding space of text encoder with attention mechanism. 

large pre-trained models usually consume more resources in computing and memory, which is not desirable for some resource-constrained scenarios such as image retrieval. To tackle this inconvenience, recently many works Budnik & Avrithis (2021); Wu et al. (2022a); Duggal et al. (2021); Wu et al. (2022b) have been proposed to research into asymmetric image retrieval, in which the query (user) side adopts a lightweight model such as Howard et al. (2017; 2019); Liu et al. (2023); Mehta & Rastegari (2021); Sandler et al. (2018); Tan & Le (2019) while the gallery (server) side takes a large model. To our knowledge, our proposed framework is the first work to adopt asymmetric retrieval in CIR task, and could even outperform its symmetric counterpart. 

## 3 METHOD 

The overview of our framework is shown in Figure 1. Our framework integrates a pre-trained visionlanguage (VL) model, comprising a large visual encoder and a large text encoder. These encoders are specifically designed to align visual and textual modalities within a unified feature space. Additionally, a matching encoder is incorporated to enhance image-text matching capabilities, which ensures fine-grained alignment and interpretation between different modalities. We freeze the parameter of VL model to maintain the alignment between visual and textual modality during training. Given an image _**I**_ , a lightweight visual encoder is applied to extract visual feature map _**F**[LE]_ of augmented _**I**_ , and a new adaptive token learner is utilized to map the visual feature map to a series of sentence tokens _**U**_ . _**U**_ is then concatenated with a prompt such as “a photo of” to form a full sentence, which is fed into the large text encoder to obtain textual feature _**t**_ . Large visual encoder is utilized to extract the image feature _**v**_ and feature map _**F**[v]_ of _**I**_ . Global contrastive distillation is applied to align _**v**_ and _**t**_ while local alignment regularization is applied on _**U**_ and _**F[v]**_ . 

## 3.1 LIGHT VISUAL ENCODER AND ADAPTIVE TOKEN LEARNER 

As shown in Figure 1, we apply data augmentation to input image _**I**_ and feed it into the light visual encoder, to extract visual feature map _**F**[LE] ∈_ R _[H][×][W][ ×][C]_ . It is a common understanding that light models are less discriminative compared with the large models. Therefore, we introduce an innovative adaptive token learner. Instead of representing an image as a single word, our method conceptualizes the image as a sentence, which significantly enhances visual discrimination within the word embedding space. Each image element is treated as an individual narrative component, allowing for a richer and more detailed representation in the embedding space. Specifically, we map each pixel _**f** h,w[LE]_[of] _**[ F]**[ LE]_[to one of] _[ L]_[ semantic groups using convolutions, and generate] _[ L]_[ attention] maps _**A**_ = _{_ _**a**_[(1)] _, · · · ,_ _**a**_[(] _[L]_[)] _}_ conditioned on the _**F**[LE]_ . The spatial attention maps are computed as 


![](raw/papers/2_LLM_NLP/LLM_paper_list/zero-shot/iclr_2024_Image2Sentence_based_Asymmetrical_Zero-shot_Composed_Image_Retrieval_images/iclr_2024_Image2Sentence_based_Asymmetrical_Zero-shot_Composed_Image_Retrieval.pdf-0003-08.png)


where _**W**_ = [ _**w**_[(1)] _, · · · ,_ _**w**_[(] _[L]_[)] ] _∈_ R _[L][×][C]_ are the parameters of the _L_ convolutions. Spatial attention is applied to adaptively discover discriminative visual patterns, and obtain _L_ visual tokens _**Z**_ = [ _**z**_[(1)] _, · · · ,_ _**z**_[(] _[L]_[)] ] 

3 

Published as a conference paper at ICLR 2024 


![](raw/papers/2_LLM_NLP/LLM_paper_list/zero-shot/iclr_2024_Image2Sentence_based_Asymmetrical_Zero-shot_Composed_Image_Retrieval_images/iclr_2024_Image2Sentence_based_Asymmetrical_Zero-shot_Composed_Image_Retrieval.pdf-0004-01.png)


Since each visual token is learned separately without considering their mutual relationship, selfattention is utilized to model the interaction among visual tokens to obtain the refined tokens _**Z** r_ 


![](raw/papers/2_LLM_NLP/LLM_paper_list/zero-shot/iclr_2024_Image2Sentence_based_Asymmetrical_Zero-shot_Composed_Image_Retrieval_images/iclr_2024_Image2Sentence_based_Asymmetrical_Zero-shot_Composed_Image_Retrieval.pdf-0004-03.png)


where _**Q** r,_ _**K** r,_ _**V** r_ are the parameters of query, key and value, and _**W** r_ 1 _,_ _**W** r_ 2 are the parameters of feed-forward network with _σ_ ( _·_ ) activation. To obtain the _L_ sentence tokens that involve filtered visual semantics, we adopt the cross-attention to adaptively select the salient visual patterns from the feature map _**F**[LE]_ . _**F**[LE]_ is first flattened as a sequence _**F**[LE][′]_ = [ _**f**_ 1 _[LE] ,_ 1 _[,][ · · ·][,]_ _**[ f]**[ LE] H,W_[]] _[∈]_[R] _[HW][ ×][C]_[,] and the sentence tokens are calculated as 


![](raw/papers/2_LLM_NLP/LLM_paper_list/zero-shot/iclr_2024_Image2Sentence_based_Asymmetrical_Zero-shot_Composed_Image_Retrieval_images/iclr_2024_Image2Sentence_based_Asymmetrical_Zero-shot_Composed_Image_Retrieval.pdf-0004-05.png)


where _**Q** c,_ _**K** c,_ _**V** c_ are the parameters of query, key and value, and _**W** c_ 1 _,_ _**W** c_ 2 are the parameters of feed-forward network with _σ_ ( _·_ ) activation. _**W** p ∈_ R _[C][×][d][w]_ are the parameters of fc layer to project the output from feed-forward network to _L_ sentence tokens _**U**_ = [ _**u**_[(1)] _, · · · ,_ _**u**_[(] _[L]_[)] ] _∈_ R _[L][×][d][w]_ , where _dw_ is the dimension of word embedding space. In sum, the adaptive token learner incorporates the spatial attention mechanism to filter out noisy information by selecting more prominent visual patterns, and map them to the word embedding space as a sentence. 

## 3.2 LEARNING WITH PRE-TRAINED FOUNDATION MODEL 

Given sentence tokens _**U**_ , we concatenate them with the predefined prompt to form a full sentence, and feed it into the large text encoder to obtain the textual feature _**t**_ . During training, _**t**_ is constrained to be similar to the corresponding image feature _**v**_ extracted from the large visual encoder within batch _B_ . The normalized image-to-text and text-to-image similarity are calculated as 


![](raw/papers/2_LLM_NLP/LLM_paper_list/zero-shot/iclr_2024_Image2Sentence_based_Asymmetrical_Zero-shot_Composed_Image_Retrieval_images/iclr_2024_Image2Sentence_based_Asymmetrical_Zero-shot_Composed_Image_Retrieval.pdf-0004-09.png)


where _τ_ denotes the temperature parameter. Suppose _**y**[I]_[2] _[T] ,_ _**y**[T]_[ 2] _[I]_ denote the ground-truth one-hot similarity, the global contrastive distillation is computed as the cross-entropy _H_ between _**p**_ and _**y**_ 


![](raw/papers/2_LLM_NLP/LLM_paper_list/zero-shot/iclr_2024_Image2Sentence_based_Asymmetrical_Zero-shot_Composed_Image_Retrieval_images/iclr_2024_Image2Sentence_based_Asymmetrical_Zero-shot_Composed_Image_Retrieval.pdf-0004-11.png)


To further enhance the semantic alignment of sentence tokens with visual information, local alignment regularization is applied to harness sentence tokens to be more like the caption that faithfully describes the image content in the word embedding space. Suppose that the word embedding space is shared across the text encoder and matching encoder, therefore the local alignment regularization will help to generate better sentence tokens, and improve the global semantic alignment between _**t**_ and _**v**_ . Specifically, we feed sentence tokens together with visual feature map _**F**[v]_ into the matching encoder to obtain the probability _**p**[LAR]_ . The local alignment regularization is calculated as 


![](raw/papers/2_LLM_NLP/LLM_paper_list/zero-shot/iclr_2024_Image2Sentence_based_Asymmetrical_Zero-shot_Composed_Image_Retrieval_images/iclr_2024_Image2Sentence_based_Asymmetrical_Zero-shot_Composed_Image_Retrieval.pdf-0004-13.png)


where _**y**[LAR]_ denotes the binary one-hot vector that represents the paired label. Therefore, the final loss could be computed as 


![](raw/papers/2_LLM_NLP/LLM_paper_list/zero-shot/iclr_2024_Image2Sentence_based_Asymmetrical_Zero-shot_Composed_Image_Retrieval_images/iclr_2024_Image2Sentence_based_Asymmetrical_Zero-shot_Composed_Image_Retrieval.pdf-0004-15.png)


With the knowledge from the pre-trained foundation model, our framework could acquire composition ability by training on unlabeled images. 

4 

Published as a conference paper at ICLR 2024 


![](raw/papers/2_LLM_NLP/LLM_paper_list/zero-shot/iclr_2024_Image2Sentence_based_Asymmetrical_Zero-shot_Composed_Image_Retrieval_images/iclr_2024_Image2Sentence_based_Asymmetrical_Zero-shot_Composed_Image_Retrieval.pdf-0005-01.png)


**----- Start of picture text -----**<br>
Feedback<br>Image + Text User Large<br>Text<br>is on grass and  A  photo of                               Upload Encoder retrieval<br>has one cat lying  that is on grass and has one<br>down on it cat lying down on it<br>Database<br>image set<br>Large<br>EncoderLight  Mobile device Cloud server EncoderVisual<br>token leaner Adaptive<br>**----- End of picture text -----**<br>


Figure 2: The inference workflow of our framework. The light encoder and the adaptive token learner are deployed on mobile device, to extract sentence tokens from the query image and concatenate them with the prompt and text modifier as the composed query. Pre-trained foundation model is deployed on the cloud server, and the large text encoder receives the uploaded composed query to extract the text feature for retrieval, and the large visual encoder is used to extract features of database images. The retrieval results are returned to the user side. 

## 3.3 INFERENCE 

In practical application scenarios, the memory and computing resources should be included in our consideration. Therefore, as shown in Figure 2, the lightweight visual encoder and the adaptive token learner are deployed on mobile devices, while the large visual and text encoder are deployed on the server side. During the inference phase, the user inputs a query image along with a text modifier. The system then converts this query image into sentence tokens, a process that encapsulates the visual content into a textual format. This conversion is pivotal for the subsequent retrieval process, as it enables the system to interpret and process the visual data in a linguistically contextualized manner. Specifically, we concatenate the sentence tokens and pre-defined prompts together with text modifier as the composed query: [prompt] _**u**_ 1 _,_ _**u**_ 2 _, · · · ,_ _**u** L_ that [text modifier]. It is uploaded to the cloud server to retrieve the images from the database. The large text encoder is utilized to extract the composed query feature during online inference, and the large visual encoder is only utilized to extract image features from the database offline. 

## 4 EXPERIMENTS 

In this section, we conduct a series of experiments to evaluate our ISA on the task of zero-shot composed image retrieval. Three datasets are utilized to verify the effectiveness of our method, including CIRR Liu et al. (2021), FashionIQ Wu et al. (2021) and CIRCO Baldrati et al. (2023a). 

## 4.1 IMPLEMENTATION DETAIL 

We adopt the BLIP model Li et al. (2022) pre-trained on 129M image-text pairs as the multi-modal foundation model. CC3M Sharma et al. (2018) is adopted as the training set in our framework. Since BLIP visual encoder is frozen during training, image feature _**v**_ and **FLOPS(G) PARAM(M)** visual feature map _**F**[v]_ could be ex- **Query model Gallery model** ABS % ABS % tracted and stored in advance. For BLIP VE BLIP VE 16.86 100.0 87.0 100.0 the adaptive token learner, the token EfficientNet B0 0.43 2.6 5.3 6.1 EfficientNet B2 0.72 4.3 8.5 9.7 length _L_ is set as 6, and two hidden EfficientViT M2 BLIP VE 0.21 1.2 5.0 5.7 dimensions of feed-forward are set as MobileNet V2 0.35 2.1 3.5 4.0 256 and 512, respectively. AdamW MobileViT V2 1.42 8.4 5.5 6.3 optimizer with 3e-4 learning rate is adopted, and the framework is trained Table 1: FLOPS and parameters of different light models for 20 epochs with 5 epochs of linear with adaptive token learner in our work, absolute (ABS) warm-up and 15 epochs of cosine anand relative (%) to the BLIP visual encoder (BLIP VE). nealing. The proposed method is implemented on open source Pytorch framework on a server with 4 NVIDIA GeForce RTX 3090 GPU, with the batch size as 320. 

5 

Published as a conference paper at ICLR 2024 

We first compare with three simple baselines which directly extract the visual and text features using large foundation model: (i) Image-only, using only visual features for both query and database; (ii) Text-only, using textual features for query and visual features for database; (iii) Image + Text, simply merging the visual and textual features with addition for query, while using the visual features for database. For the state-of-the-art ZSCIR methods, Pic2word Saito et al. (2023)and SEARLE Baldrati et al. (2023a) are included. Since Pic2word and SEARLE are all based on ViT-L/14 CLIP model, we re-implement their methods on BLIP with the open-source codes for fair comparison. Two lightweight models are mainly used as the lightweight visual encoder for our ISA: EfficientNet B2 Tan & Le (2019) and EfficientViT M2 Liu et al. (2023), as representatives of two main categories of deep architectures (CNN and transformer). Since CLIP models do not include the pre-trained matching encoder that share the word embedding space with the text encoder, we only report our results based on BLIP. We also compare ISA with a symmetric baseline, _i.e._ , frozen BLIP visual encoder is adopted to obtain the visual feature map, followed by the same adaptive token learner to obtain the sentence tokens. 

|**Gallery model**<br>**Query model**<br>**Method**|**Recall@K**<br>**Avg**<br>R@1<br>R@5<br>R@10<br>R@50|
|---|---|
|CLIP<br>ViT-L/14<br>CLIP VE<br>Image-only<br>-<br>Text-only<br>CLIP VE<br>Image + Text<br>CLIP VE<br>Pic2Word Saito et al. (2023)<br>CLIP VE<br>SEARLE Baldrati et al. (2023a)|7.40<br>23.60<br>34.00<br>57.40<br>30.60<br>20.90<br>44.80<br>56.70<br>79.10<br>50.36<br>12.40<br>36.20<br>49.10<br>78.20<br>43.97<br>23.90<br>51.70<br>65.30<br>87.80<br>57.17<br>24.22<br>52.41<br>66.29<br>88.63<br>57.88|
|BLIP<br>BLIP VE<br>Image-only<br>-<br>Text-only<br>BLIP VE<br>Image + Text<br>BLIP VE<br>Pic2word Saito et al. (2023)<br>BLIP VE<br>SEARLE Baldrati et al. (2023a)<br>BLIP VE<br>**Ours(Sym)**<br>EffcientNet B2<br>**Ours(Asy)**<br>EffcientViT M2<br>**Ours(Asy)**|7.52<br>21.83<br>31.21<br>52.94<br>28.37<br>26.48<br>50.46<br>61.22<br>82.26<br>55.10<br>8.19<br>24.19<br>34.41<br>57.81<br>31.15<br>26.70<br>53.16<br>64.10<br>84.36<br>57.08<br>29.27<br>54.86<br>66.57<br>86.16<br>59.21<br>29.68<br>58.72<br>70.79<br>90.33<br>62.38<br>**30.84**<br>**61.06**<br>**73.57**<br>**92.43**<br>**64.48**<br>29.63<br>58.99<br>71.37<br>91.47<br>62.87|



Table 2: Results of zero-shot composed image retrieval on CIRR test set. The best result for each column is shown in bold fonts. Sym: use BLIP visual models to both extract features for query and database images. Asy: use different visual models to extract features for query and database images. 

|**Gallery model**<br>**Query model**<br>**Method**|**mAP@K**<br>**Avg**<br>mAP@5<br>mAP@10<br>mAP@25<br>mAP@50|
|---|---|
|CLIP<br>ViT-L/14<br>CLIP VE<br>Image-only<br>-<br>Text-only<br>CLIP VE<br>Image + Text<br>CLIP VE<br>Pic2Word Saito et al. (2023)<br>CLIP VE<br>SEARLE Baldrati et al. (2023a)|1.69<br>2.18<br>2.78<br>3.22<br>2.47<br>2.67<br>3.00<br>3.38<br>3.69<br>3.19<br>3.60<br>4.40<br>5.59<br>6.21<br>4.95<br>8.43<br>9.11<br>10.15<br>10.73<br>9.61<br>10.18<br>11.03<br>12.72<br>13.67<br>11.90|
|BLIP<br>BLIP VE<br>Image-only<br>-<br>Text-only<br>BLIP VE<br>Image + Text<br>BLIP VE<br>Pic2word Saito et al. (2023)<br>BLIP VE<br>SEARLE Baldrati et al. (2023a)<br>BLIP VE<br>**Ours(Sym)**<br>EffcientNet B2<br>**Ours(Asy)**<br>EffcientViT M2<br>**Ours(Asy)**|1.28<br>1.48<br>1.88<br>2.13<br>1.69<br>4.01<br>4.37<br>4.79<br>5.13<br>4.58<br>1.39<br>1.72<br>2.22<br>2.51<br>1.96<br>8.69<br>9.36<br>10.40<br>10.99<br>9.86<br>10.65<br>11.34<br>12.40<br>13.02<br>11.85<br>9.67<br>10.32<br>11.26<br>11.61<br>10.72<br>**11.33**<br>**12.25**<br>**13.42**<br>**13.97**<br>**12.74**<br>9.82<br>10.50<br>11.61<br>12.09<br>11.01|



Table 3: Results of zero-shot composed image retrieval on CIRCO test set. The best result for each column is shown in bold fonts. Sym: use BLIP visual models to both extract features for query and database images. Asy: use different visual models to extract features for query and database images. 

## 4.2 COMPARISON WITH THE STATE-OF-THE-ART METHODS 

We compare different methods on the test set of CIRR and CIRCO while on the validation set of FashionIQ, and the final results are computed as the average of 3 runs on different random seeds for training. As shown in Table 1, the lightweight models are more efficient in parameters and computational complexity compared with BLIP visual encoder. The retrieval performance on three benchmark datasets is reported in Table 2, 3, 4, respectively. Our method could outperform simple baselines (Image-only, Text-only and Image + Text) by a large margin. From the results of Pic2word and SEARLE, we find that the methods based on BLIP are generally better than those based on CLIP for ZSCIR. Our EfficientNet B2-based ISA could outperform Pic2word and SEARLE on three benchmark datasets, while EfficientViT M2-based ISA achieves better results on FashionIQ and CIRR and is comparable with SEARLE on CIRCO. Compared with our symmetric retrieval in 

6 

Published as a conference paper at ICLR 2024 

|**Gallery model**<br>**Query model**<br>**Method**|**Dress**<br>**Shirt**<br>**Top&tee**<br>**Average**|
|---|---|
||R@10<br>R@50<br>R@10<br>R@50<br>R@10<br>R@50<br>R@10<br>R@50|
|CLIP<br>ViT-L/14<br>CLIP VE<br>Image-only<br>-<br>Text-only<br>CLIP VE<br>Image + Text<br>CLIP VE<br>Pic2Word Saito et al. (2023)<br>CLIP VE<br>SEARLE Baldrati et al. (2023a)|5.40<br>13.90<br>9.90<br>20.80<br>8.30<br>17.70<br>7.90<br>17.50<br>13.60<br>29.70<br>18.90<br>31.80<br>19.30<br>37.00<br>17.30<br>32.90<br>16.30<br>33.60<br>21.00<br>34.50<br>22.20<br>39.00<br>19.80<br>35.70<br>20.00<br>40.20<br>26.20<br>43.60<br>27.90<br>47.40<br>24.70<br>43.70<br>20.32<br>43.18<br>27.43<br>45.68<br>29.32<br>50.17<br>25.69<br>46.34|
|BLIP<br>BLIP VE<br>Image-only<br>-<br>Text-only<br>BLIP VE<br>Image + Text<br>BLIP VE<br>Pic2word Saito et al. (2023)<br>BLIP VE<br>SEARLE Baldrati et al. (2023a)<br>BLIP VE<br>**Ours(Sym)**<br>EffcientNet B2<br>**Ours(Asy)**<br>EffcientViT M2<br>**Ours(Asy)**|4.07<br>11.35<br>7.07<br>16.39<br>6.88<br>13.67<br>6.01<br>13.67<br>18.10<br>35.05<br>23.45<br>39.21<br>26.06<br>44.93<br>22.54<br>39.73<br>5.16<br>13.58<br>8.15<br>18.89<br>7.80<br>16.62<br>7.03<br>16.36<br>21.35<br>42.68<br>27.51<br>46.01<br>29.12<br>49.33<br>25.99<br>46.00<br>22.11<br>41.79<br>29.72<br>48.53<br>31.03<br>52.37<br>27.62<br>47.56<br>24.69<br>43.88<br>**30.79**<br>**50.05**<br>**33.91**<br>53.65<br>**29.79**<br>49.19<br>25.33<br>**46.26**<br>30.03<br>48.58<br>33.45<br>53.80<br>29.60<br>**49.54**<br>**25.48**<br>45.51<br>29.64<br>48.68<br>32.94<br>**54.31**<br>29.35<br>49.50|



Table 4: Results of zero-shot composed image retrieval on FashionIQ validation set. The best result for each column is shown in bold fonts. Sym: use BLIP visual model to both extract features for query and database. Asy: use different visual models to extract features for query and database. 

Ours (Sym), our ISA could achieve superior performance, which could be attributed to ISA containing more training parameters compared with Ours (Sym) (lightweight model + adaptive token learner v.s. adaptive token learner) during training. Meanwhile, ISA consumes much less computation resources during online retrieval and is more flexible to deploy for real application scenarios, demonstrating the effectiveness of our framework. 

## 4.3 ABLATION STUDY 

For ablation study, we mainly adopt the validation set of CIRR and CIRCO for evaluation, and the results for FashionIQ are included in the appendix. 

## 4.3.1 TRAINING WITH DIFFERENT MAPPING NETWORK AND NUMBER OF TOKENS 

To verify the effectiveness of our adaptive token learner, we compare it with a variant that adopts MLP as the mapping network, which maps the image feature to a single word following Baldrati et al. (2023a); Saito et al. (2023). The results are shown in Table 5, which demonstrate the consistent superiority for both EfficientNet B2 and EfficientViT M2. These results prove the effectiveness of our adaptive token learner, which maps the visual feature map to multiple sentence tokens per image, and helps to adaptively select more discriminative visual patterns from the query image to generate more informative sentence tokens. It is notable that the variant MLP (single token) could be seen as Pic2word with LAR and outperform Pic2word, thus proving the effectiveness of LAR. 

|**Query model**<br>**Mapping network**|**CIRR**<br>**Avg**<br>R@1<br>R@5<br>R@10<br>R@50|**CIRCO**<br>**Avg**<br>mAP@5<br>mAP@10<br>mAP@25<br>mAP@50|
|---|---|---|
|EffcientNet B2<br>ATL<br>EffcientNet B2<br>MLP (single token)|**31.48**<br>**63.42**<br>**77.27**<br>**93.16**<br>**66.33**<br>30.18<br>59.46<br>71.30<br>89.33<br>62.57|**13.19**<br>**13.83**<br>**15.20**<br>**15.85**<br>**14.52**<br>8.28<br>8.67<br>9.62<br>10.00<br>9.14|
|EffcientViT M2<br>ATL<br>EffcientViT M2<br>MLP (single token)|31.26<br>62.31<br>75.70<br>92.38<br>65.41<br>30.32<br>60.25<br>71.49<br>89.57<br>62.90|12.24<br>12.75<br>13.80<br>14.27<br>13.27<br>7.19<br>7.73<br>8.47<br>9.09<br>8.12|



Table 5: Results of different mapping networks with BLIP on CIRR and CIRCO validation set. ATL denotes the proposed adaptive token learner. The best results are shown in bold fonts. 

For our adaptive token learner, we show the performance of our method with different token lengths _L_ on three datasets in Figure 3. We average the recall or precision values for each dataset to obtain the overall evaluation criterion. We find that the average performance improves with the increase of token length for both lightweight visual encoders, however excessive large token length leads to performance degradation. We visualize the attention map of adaptive token learner and the retrieval results in Figure 4, and discover that with excessive large token lengths, some sentence tokens may tend to focus on the background or the trivial patterns, which would likely introduce noise to degrade the retrieval performance. 

## 4.3.2 TRAIN WITH DIFFERENT QUERY MODELS 

In order to investigate the influence of different lightweight encoders, we conduct experiments with different light visual models and pre-training parameters, as shown in Table 6. We could see that EfficientNet B2 shows the best overall performance in CNN, while EfficientViT M2 is proved to be the best in transformer. This could be attributed to the superior network design and larger capacity of 

7 

Published as a conference paper at ICLR 2024 


![](raw/papers/2_LLM_NLP/LLM_paper_list/zero-shot/iclr_2024_Image2Sentence_based_Asymmetrical_Zero-shot_Composed_Image_Retrieval_images/iclr_2024_Image2Sentence_based_Asymmetrical_Zero-shot_Composed_Image_Retrieval.pdf-0008-01.png)


**----- Start of picture text -----**<br>
FashionIQ CIRR CIRCO<br>68 16<br>40<br>14<br>66<br>38 12<br>64<br>10<br>36<br>ENT B2 ENT B 2 ENT B2<br>EVT M2 62 EVT M2 8 EVT M2<br>34<br>0 5 10 0 5 10 0 5 10<br>token length of sentence token length of sentence token length of sentence<br>Average scores Average scores Average scores<br>**----- End of picture text -----**<br>


Figure 3: The average and standard deviation of performance scores on three validation sets. 


![](raw/papers/2_LLM_NLP/LLM_paper_list/zero-shot/iclr_2024_Image2Sentence_based_Asymmetrical_Zero-shot_Composed_Image_Retrieval_images/iclr_2024_Image2Sentence_based_Asymmetrical_Zero-shot_Composed_Image_Retrieval.pdf-0008-03.png)


**----- Start of picture text -----**<br>
L= 6, attention map L= 6, attention map<br>Retrieval results Retrieval results<br>has a dog of  is shot with<br>a different  L= 10, attention map the same  L= 10, attention map<br>breed and  style and<br>shows a  most of the<br>jolly roger benches are<br>Retrieval results empty Retrieval results<br>**----- End of picture text -----**<br>


Figure 4: Attention map of sentence tokens for different token lengths and retrieval results. Query image and text modifier are on the left side. Ground-truth target images are marked in green boxes. 

parameters. Moreover, better initialization weights, _i.e._ , adopting weights pre-trained on ImageNet Deng et al. (2009), could obtain better performance compared with random initialization. This reveals the possibility of adopting different lightweight models to flexibly meet various efficiency and retrieval accuracy requirements, and improve the performance with better pre-trained weights. 

|**Query model**|**CIRR**<br>**Avg**<br>R@1<br>R@5<br>R@10<br>R@50|**CIRCO**<br>**Avg**<br>mAP@5<br>mAP@10<br>mAP@25<br>mAP@50|
|---|---|---|
|EffcientNet B0<br>EffcientNet B0*<br>EffcientNet B2<br>EffcientNet B2*<br>EffcientViT M2<br>EffcientViT M2*<br>MobileNet V2<br>MobileNet V2*<br>MobileViT V2<br>MobileViT V2*|**32.02**<br>**64.57**<br>76.12<br>92.29<br>66.25<br>27.82<br>58.38<br>71.16<br>90.07<br>61.86<br>31.48<br>63.42<br>**77.27**<br>**93.16**<br>**66.33**<br>29.47<br>60.13<br>74.05<br>91.27<br>63.73<br>31.26<br>62.31<br>75.70<br>92.38<br>65.41<br>28.56<br>59.46<br>71.11<br>89.38<br>62.13<br>30.50<br>63.51<br>74.27<br>91.18<br>64.87<br>29.47<br>60.14<br>72.10<br>90.07<br>62.95<br>27.63<br>58.66<br>70.85<br>88.68<br>61.46<br>27.02<br>57.92<br>69.33<br>88.96<br>60.81|11.11<br>12.04<br>13.14<br>13.67<br>12.49<br>9.5<br>9.96<br>10.6<br>11.02<br>10.27<br>**13.19**<br>**13.83**<br>**15.20**<br>**15.85**<br>**14.52**<br>11.16<br>11.57<br>12.56<br>13.13<br>12.11<br>12.24<br>12.75<br>13.80<br>14.27<br>13.27<br>10.19<br>10.79<br>11.48<br>11.94<br>11.10<br>10.50<br>10.82<br>12.09<br>12.44<br>11.46<br>8.55<br>9.16<br>10.04<br>10.39<br>9.54<br>9.44<br>9.68<br>10.49<br>10.96<br>10.14<br>8.41<br>8.77<br>9.54<br>9.97<br>9.17|



Table 6: Results of different query models with BLIP on CIRR/CIRCO validation set. * denotes adopting random initialization for light query model. The best results are shown in bold fonts. 

## 4.3.3 SYMMETRIC AND ASYMMETRIC RETRIEVAL 

To verify the effectiveness of asymmetric framework, we conduct additional experiments on symmetric setting, which adopts the same model to extract both query and database features. Specifically, for the symmetric retrieval in the second, 4-th and 6-th rows of Table 7, query and gallery models are both updated during training. Therefore, the network should both learn to generate the image feature, and map the image to the sentence tokens in the word embedding space. We find that they would drastically degrade the performance. This could be attributed to the confusion of multitask learning for image feature and sentence tokens, since image feature space and word embedding space are heterogeneous semantic spaces. Moreover, we also carry out additional experiments that adopting BLIP VE as both query model and gallery model, and only query model is updated during 

8 

Published as a conference paper at ICLR 2024 

training to avoid the multi-task confusion. Therefore, BLIP VE is only trained to generate sentence tokens. As revealed in the result of BLIP VE _†_ , there is no breaking down for learning, however it is still outperformed by EfficientNet B2-based ISA. This could be explained by the fact that BLIP VE is much heavier than light models as shown in Table 1, and could be trained only with reduced batch size, thus degrading the effectiveness of contrastive and LAR loss. The above results reveal the importance of pre-trained alignment for different modalities in ZSCIR, but the BLIP visual encoder could be replaced with lightweight model for efficiency. 

|**Query model**<br>**Gallery model**|**CIRR**<br>**Avg**<br>R@1<br>R@5<br>R@10<br>R@50|**CIRCO**<br>**Avg**<br>mAP@5<br>mAP@10<br>mAP@25<br>mAP@50|
|---|---|---|
|EffcientNet B2<br>BLIP VE<br>EffcientNet B2<br>EffcientNet B2<br>EffcientViT M2<br>BLIP VE<br>EffcientViT M2<br>EffcientViT M2<br>BLIP VE_†_<br>BLIP VE<br>BLIP VE<br>BLIP VE|**31.48**<br>**63.42**<br>**77.27**<br>**93.16**<br>**66.33**<br>0.61<br>0.74<br>1.02<br>3.97<br>1.59<br>31.26<br>62.31<br>75.70<br>92.38<br>65.41<br>0.29<br>0.37<br>0.79<br>2.68<br>1.03<br>30.93<br>62.09<br>74.67<br>92.90<br>65.15<br>0.49<br>0.60<br>0.83<br>2.96<br>1.22|**13.19**<br>**13.83**<br>**15.20**<br>**15.85**<br>**14.52**<br>0.00<br>0.02<br>0.08<br>0.14<br>0.06<br>12.24<br>12.75<br>13.80<br>14.27<br>13.27<br>0.00<br>0.01<br>0.04<br>0.11<br>0.04<br>12.60<br>13.20<br>14.56<br>15.07<br>13.86<br>0.02<br>0.03<br>0.08<br>0.15<br>0.07|



Table 7: Comparison with symmetric retrieval on the validation set of CIRR and CIRCO. BLIP VE denotes the BLIP visual encoder. _†_ denotes updating the query BLIP VE while preserving the pre-trained BLIP VE in gallery side during training for maintaining the multi-modality alignment. The best results are shown in bold fonts. 

## 4.3.4 LEARNING WITH DIFFERENT LOSS CONSTRAINTS 

In our framework, we mainly apply global contrastive distillation (GCD) and local alignment regularization (LAR) for learning. We verify the losses by dropping each of them separately. To further validate LAR, we compare it with the text regularization, _i.e._ , LLM TR proposed in Baldrati et al. (2023a). Specifically, LLM TR is realized following the pipeline that utilizes a classification model to label each training image and an LLM to do sentence-making. A cosine loss is applied to constrain the sentence tokens to be similar to the LLM-made sentence as text regularization. As reported in Table 8, it is obvious that both losses are crucial for training, since dropping either of them would lead to consistent worse performance. Text regularization from LLM-made sentences would be helpful but only restricted to CIRCO, however it needs extra classification model and LLM model which leads to a much more complex pipeline. Our LAR is simple but more effective on three benchmarks. 

|**Query model**<br>**Variants**||**CIRR**|**Avg**<br>R@50|**CIRCO**<br>**Avg**<br>mAP@5<br>mAP@10<br>mAP@25<br>mAP@50|
|---|---|---|---|---|
|||R@1<br>R@5<br>R@10|||
|EffcientNet B2<br>GCD + LAR<br>GCD<br>LAR<br>GCD + LLM TR||**31.48**<br>**63.42**<br>**77.27**<br>29.86<br>60.18<br>73.90<br>27.02<br>56.67<br>68.91<br>29.43<br>60.18<br>73.21|**93.16**<br>**66.33**<br>91.13<br>63.77<br>88.18<br>60.20<br>91.32<br>63.54|**13.19**<br>**13.83**<br>**15.20**<br>**15.85**<br>**14.52**<br>10.46<br>10.98<br>11.88<br>12.41<br>11.43<br>7.56<br>7.68<br>8.45<br>8.94<br>8.16<br>11.07<br>11.35<br>12.59<br>13.17<br>12.05|
|EffcientViT M2<br>GCD + LAR<br>GCD<br>LAR<br>GCD + LLM TR||31.26<br>62.31<br>75.70<br>26.87<br>55.58<br>69.00<br>24.35<br>50.49<br>63.26<br>27.61<br>56.49<br>70.03|92.38<br>65.41<br>89.38<br>60.01<br>84.00<br>55.53<br>89.76<br>60.97|12.24<br>12.75<br>13.80<br>14.27<br>13.27<br>9.48<br>9.84<br>10.95<br>11.38<br>10.41<br>5.75<br>6.00<br>6.98<br>7.26<br>6.50<br>10.18<br>10.69<br>11.48<br>12.04<br>11.10|



Table 8: Results of different losses on CIRR and CIRCO validation set. GCD: global contrastive distillation; LAR: local alignment regularization; LLM TR: text regularization with the sentences made by LLM. The best and second-best results are highlighted in bold and underlined fonts, respectively. 

## 5 CONCLUSION 

In this paper, we focus on a new but more practical scenarios of asymmetric zero-shot composed image retrieval, which eliminates the requirement of meticulously-labeled training triplets and is flexible for deployment on a resource-constrained platform. Specifically, lightweight model is adopted for the query side while large visual encoder of foundation model is adopted for the gallery side. Adaptive token learner is proposed as the semantic filter to automatically select discriminative semantic groups from image and map them to a sentence, maintaining the fidelity of visual semantics within the word embedding space. By directly concatenating the mapped sentence with text modifier, we are able to compose the multi-modal query to achieve composed retrieval. Global contrastive distillation is applied to align the light model with large foundation model, and local alignment regularization is further utilized to help the mapped sentence better align with the real caption that faithfully describes the image. Extensive experimental results demonstrate that our framework could both achieve better performance and improve retrieval efficiency with deployment flexibility. 

9 

Published as a conference paper at ICLR 2024 

## 6 ACKNOWLEDGEMENT 

This work was supported by the National Natural Science Foundation of China under Contract 62102128 and 62021001. It was also supported by the GPU cluster built by MCC Lab of Information Science and Technology Institution of USTC. 

## REFERENCES 

- Alberto Baldrati, Marco Bertini, Tiberio Uricchio, and Alberto Del Bimbo. Conditioned image retrieval for fashion using contrastive learning and clip-based features. In _ACM Multimedia Asia_ , pp. 1–5. 2021. 

- Alberto Baldrati, Marco Bertini, Tiberio Uricchio, and Alberto Del Bimbo. Conditioned and composed image retrieval combining and partially fine-tuning clip-based features. In _Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition_ , pp. 4959–4968, 2022a. 

- Alberto Baldrati, Marco Bertini, Tiberio Uricchio, and Alberto Del Bimbo. Effective conditioned and composed image retrieval combining clip-based features. In _Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition_ , pp. 21466–21474, 2022b. 

- Alberto Baldrati, Lorenzo Agnolucci, Marco Bertini, and Alberto Del Bimbo. Zero-shot composed image retrieval with textual inversion. _arXiv preprint arXiv:2303.15247_ , 2023a. 

- Alberto Baldrati, Marco Bertini, Tiberio Uricchio, and Alberto del Bimbo. Composed image retrieval using contrastive learning and task-oriented clip-based features. _arXiv preprint arXiv:2308.11485_ , 2023b. 

- Mateusz Budnik and Yannis Avrithis. Asymmetric metric learning for knowledge transfer. In _Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition_ , pp. 8228– 8238, 2021. 

- Yanbei Chen, Shaogang Gong, and Loris Bazzani. Image search with text feedback by visiolinguistic attention learning. In _Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition_ , pp. 3001–3011, 2020. 

- Ginger Delmas, Rafael S Rezende, Gabriela Csurka, and Diane Larlus. Artemis: Attention-based retrieval with text-explicit matching and implicit similarity. In _International Conference on Learning Representations_ , 2021. 

- Jia Deng, Wei Dong, Richard Socher, Li-Jia Li, Kai Li, and Li Fei-Fei. Imagenet: A large-scale hierarchical image database. In _Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition_ , pp. 248–255. Ieee, 2009. 

- Eric Dodds, Jack Culpepper, and Gaurav Srivastava. Training and challenging models for textguided fashion image retrieval. _arXiv preprint arXiv:2204.11004_ , 2022. 

- Xiao Dong, Xunlin Zhan, Yangxin Wu, Yunchao Wei, Michael C Kampffmeyer, Xiaoyong Wei, Minlong Lu, Yaowei Wang, and Xiaodan Liang. M5product: Self-harmonized contrastive learning for e-commercial multi-modal pretraining. In _Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition_ , pp. 21252–21262, 2022. 

- Rahul Duggal, Hao Zhou, Shuo Yang, Yuanjun Xiong, Wei Xia, Zhuowen Tu, and Stefano Soatto. Compatibility-aware heterogeneous visual search. In _Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition_ , pp. 10723–10732, 2021. 

- Niloofar Gheissari, Thomas B Sebastian, and Richard Hartley. Person reidentification using spatiotemporal appearance. In _Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition_ , volume 2, pp. 1528–1535. IEEE, 2006. 

- Geonmo Gu, Sanghyuk Chun, Wonjae Kim, HeeJae Jun, Yoohoon Kang, and Sangdoo Yun. Compodiff: Versatile composed image retrieval with latent diffusion. _arXiv preprint arXiv:2303.11916_ , 2023. 

10 

Published as a conference paper at ICLR 2024 

- Xiao Han, Licheng Yu, Xiatian Zhu, Li Zhang, Yi-Zhe Song, and Tao Xiang. Fashionvil: Fashionfocused vision-and-language representation learning. In _Proceedings of the European Conference on Computer Vision_ , pp. 634–651. Springer, 2022. 

- Xiao Han, Xiatian Zhu, Licheng Yu, Li Zhang, Yi-Zhe Song, and Tao Xiang. Fame-vil: Multitasking vision-language model for heterogeneous fashion tasks. In _Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition_ , pp. 2669–2680, 2023. 

- Andrew Howard, Mark Sandler, Grace Chu, Liang-Chieh Chen, Bo Chen, Mingxing Tan, Weijun Wang, Yukun Zhu, Ruoming Pang, Vijay Vasudevan, et al. Searching for mobilenetv3. In _Proceedings of the IEEE/CVF International Conference on Computer Vision_ , pp. 1314–1324, 2019. 

- Andrew G Howard, Menglong Zhu, Bo Chen, Dmitry Kalenichenko, Weijun Wang, Tobias Weyand, Marco Andreetto, and Hartwig Adam. Mobilenets: Efficient convolutional neural networks for mobile vision applications. _arXiv preprint arXiv:1704.04861_ , 2017. 

- Chao Jia, Yinfei Yang, Ye Xia, Yi-Ting Chen, Zarana Parekh, Hieu Pham, Quoc Le, Yun-Hsuan Sung, Zhen Li, and Tom Duerig. Scaling up visual and vision-language representation learning with noisy text supervision. In _International Conference on Machine Learning_ , pp. 4904–4916. PMLR, 2021. 

- Seungmin Lee, Dongwan Kim, and Bohyung Han. Cosmo: Content-style modulation for image retrieval with text feedback. In _Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition_ , pp. 802–812, 2021. 

Matan Levy, Rami Ben-Ari, Nir Darshan, and Dani Lischinski. Data roaming and early fusion for composed image retrieval. _arXiv preprint arXiv:2303.09429_ , 2023. 

- Junnan Li, Ramprasaath Selvaraju, Akhilesh Gotmare, Shafiq Joty, Caiming Xiong, and Steven Chu Hong Hoi. Align before fuse: Vision and language representation learning with momentum distillation. _Advances in Neural Information Processing Systems_ , 34:9694–9705, 2021. 

- Junnan Li, Dongxu Li, Caiming Xiong, and Steven Hoi. Blip: Bootstrapping language-image pretraining for unified vision-language understanding and generation. In _International Conference on Machine Learning_ , pp. 12888–12900. PMLR, 2022. 

- Junnan Li, Dongxu Li, Silvio Savarese, and Steven Hoi. Blip-2: Bootstrapping languageimage pre-training with frozen image encoders and large language models. _arXiv preprint arXiv:2301.12597_ , 2023. 

- Victor Weixin Liang, Yuhui Zhang, Yongchan Kwon, Serena Yeung, and James Y Zou. Mind the gap: Understanding the modality gap in multi-modal contrastive representation learning. _Advances in Neural Information Processing Systems_ , 35:17612–17625, 2022. 

- Xinyu Liu, Houwen Peng, Ningxin Zheng, Yuqing Yang, Han Hu, and Yixuan Yuan. Efficientvit: Memory efficient vision transformer with cascaded group attention. In _Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition_ , pp. 14420–14430, 2023. 

- Zheyuan Liu, Cristian Rodriguez-Opazo, Damien Teney, and Stephen Gould. Image retrieval on real-life images with pre-trained vision-and-language models. In _Proceedings of the IEEE/CVF International Conference on Computer Vision_ , pp. 2125–2134, 2021. 

- Ziwei Liu, Ping Luo, Shi Qiu, Xiaogang Wang, and Xiaoou Tang. Deepfashion: Powering robust clothes recognition and retrieval with rich annotations. In _Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition_ , pp. 1096–1104, 2016. 

- Sachin Mehta and Mohammad Rastegari. Mobilevit: light-weight, general-purpose, and mobilefriendly vision transformer. _arXiv preprint arXiv:2110.02178_ , 2021. 

- Filip Radenovi´c, Giorgos Tolias, and Ondˇrej Chum. Fine-tuning cnn image retrieval with no human annotation. _IEEE Transactions on Pattern Analysis and Machine Intelligence_ , 41(7):1655–1668, 2018. 

11 

Published as a conference paper at ICLR 2024 

- Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, et al. Learning transferable visual models from natural language supervision. In _International Conference on Machine Learning_ , pp. 8748–8763. PMLR, 2021. 

- Kuniaki Saito, Kihyuk Sohn, Xiang Zhang, Chun-Liang Li, Chen-Yu Lee, Kate Saenko, and Tomas Pfister. Pic2word: Mapping pictures to words for zero-shot composed image retrieval. In _Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition_ , pp. 19305– 19314, 2023. 

- Mark Sandler, Andrew Howard, Menglong Zhu, Andrey Zhmoginov, and Liang-Chieh Chen. Mobilenetv2: Inverted residuals and linear bottlenecks. In _Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition_ , pp. 4510–4520, 2018. 

- Piyush Sharma, Nan Ding, Sebastian Goodman, and Radu Soricut. Conceptual captions: A cleaned, hypernymed, image alt-text dataset for automatic image captioning. In _Proceedings of the 56th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)_ , pp. 2556–2565, 2018. 

- Mingxing Tan and Quoc Le. Efficientnet: Rethinking model scaling for convolutional neural networks. In _International Conference on Machine Learning_ , pp. 6105–6114. PMLR, 2019. 

- Nam Vo, Lu Jiang, Chen Sun, Kevin Murphy, Li-Jia Li, Li Fei-Fei, and James Hays. Composing text and image for image retrieval-an empirical odyssey. In _Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition_ , pp. 6439–6448, 2019. 

- Hui Wu, Yupeng Gao, Xiaoxiao Guo, Ziad Al-Halah, Steven Rennie, Kristen Grauman, and Rogerio Feris. Fashion iq: A new dataset towards retrieving images by natural language feedback. In _Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition_ , pp. 11307–11317, 2021. 

- Hui Wu, Min Wang, Wengang Zhou, and Houqiang Li. A general rank preserving framework for asymmetric image retrieval. In _The Eleventh International Conference on Learning Representations_ , 2022a. 

- Hui Wu, Min Wang, Wengang Zhou, Houqiang Li, and Qi Tian. Contextual similarity distillation for asymmetric image retrieval. In _Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition_ , pp. 9489–9498, 2022b. 

- Xunlin Zhan, Yangxin Wu, Xiao Dong, Yunchao Wei, Minlong Lu, Yichi Zhang, Hang Xu, and Xiaodan Liang. Product1m: Towards weakly supervised instance-level product retrieval via crossmodal pretraining. In _Proceedings of the IEEE/CVF International Conference on Computer Vision_ , pp. 11782–11791, 2021. 

- Liang Zheng, Yi Yang, and Alexander G Hauptmann. Person re-identification: Past, present and future. _arXiv preprint arXiv:1610.02984_ , 2016. 

Wengang Zhou, Yijuan Lu, Houqiang Li, Yibing Song, and Qi Tian. Spatial coding for large scale partial-duplicate web image search. In _Proceedings of the 18th ACM International Conference on Multimedia_ , pp. 511–520, 2010. 

## A APPENDIX 

In the appendix, we first present detailed descriptions for the three evaluation datasets. 

FashionIQ Wu et al. (2021) is a fashion domain dataset, and it includes 77684 fashion images which are divided into three different categories: Dress, Toptee, and Shirt. The training set comprises 18000 triplets, and in total 46609 images. The validation set includes 15537 images and 6017 triplets. Each query image in the validation set has only one target image. It also has a test set but without ground-truth labels, thus the test set is not used for evaluation. 

12 

Published as a conference paper at ICLR 2024 

CIRR Liu et al. (2021) is a dataset of natural domain, and includes 21552 real-life images. It involves 36554 triplets, and are randomly assigned in 80% for training, 10% for validation and 10% for test. Each query image in the validation and test set has only one target image. 

CIRCO Baldrati et al. (2023a) is collected from MSCOCO and consists of a validation set and a test set, which includes 220 and 800 query images, respectively. Each query image has a text modifier and more than one target images. The database involves 120k images. 

We report the comparison with the state-of-the-art methods on CIRR and CIRCO validation sets in Table 9 and 10, respectively. Accompanied with Table 2 and 3, our method demonstrates consistent superiority over the simple baselines (Image-only, Text-only and Image + Text) and the state-of-theart methods of Pic2word and SEARLE. 

|**Gallery model**<br>**Query model**<br>**Method**|**Recall@K**<br>**Average**<br>R@1<br>R@5<br>R@10<br>R@50|
|---|---|
|CLIP<br>ViT-L/14<br>Image-only<br>Text-only<br>Image + Text<br>Pic2Word<br>SEARLE|7.06<br>25.52<br>35.64<br>60.18<br>32.10<br>21.19<br>46.14<br>57.43<br>78.74<br>50.88<br>10.79<br>34.37<br>47.05<br>75.27<br>41.87<br>22.54<br>51.69<br>64.76<br>86.19<br>56.30<br>25.07<br>55.20<br>68.52<br>90.24<br>59.76|
|BLIP<br>Image-only<br>Text-only<br>Image + Text<br>Pic2word<br>SEARLE<br>Ours(Sym)<br>EffcientNet B2<br>Ours(Asy)<br>EffcientViT M2<br>Ours(Asy)|6.72<br>23.13<br>32.82<br>55.66<br>29.58<br>25.38<br>51.02<br>61.76<br>81.68<br>54.96<br>7.58<br>25.90<br>36.28<br>61.13<br>32.72<br>27.89<br>58.15<br>69.42<br>87.53<br>60.75<br>28.03<br>60.74<br>71.41<br>88.82<br>62.25<br>30.21<br>61.52<br>74.09<br>90.81<br>64.16<br>31.48<br>63.42<br>**77.27**<br>**93.16**<br>**66.33**<br>31.26<br>62.31<br>75.70<br>92.38<br>65.41|



Table 9: Results of zero-shot composed image retrieval on CIRR validation set. 

|**Gallery model**<br>**Query model**<br>**Method**|**mAP@K**<br>**Average**<br>mAP@5<br>mAP@10<br>mAP@25<br>mAP@50|**mAP@K**<br>**Average**<br>mAP@5<br>mAP@10<br>mAP@25<br>mAP@50|
|---|---|---|
|CLIP<br>ViT-L/14<br>Image-only<br>Text-only<br>Image + Text<br>Pic2Word<br>SEARLE|1.85<br>2.78<br>4.71<br>7.50<br>11.02|2.24<br>3.11<br>3.57<br>2.69<br>3.06<br>3.72<br>3.95<br>3.38<br>5.74<br>6.88<br>7.44<br>6.19<br>8.31<br>9.45<br>10.11<br>8.84<br>12.12<br>13.62<br>14.43<br>12.80|
|BLIP<br>Image-only<br>Text-only<br>Image + Text<br>Pic2Word<br>SEARLE<br>Ours(Sym)<br>EffcientNet B2<br>Ours(Asy)<br>EffcientViT M2<br>Ours(Asy)|1.86<br>5.32<br>2.04<br>7.91<br>11.52<br>11.03<br>**13.19**<br>12.24|2.18<br>2.71<br>2.97<br>2.43<br>5.76<br>6.04<br>6.29<br>5.85<br>2.63<br>3.18<br>3.49<br>2.84<br>8.27<br>9.05<br>9.56<br>8.70<br>11.74<br>12.91<br>13.56<br>12.43<br>12.01<br>13.17<br>13.81<br>12.51<br>**13.83**<br>**15.20**<br>**15.85**<br>**14.52**<br>12.75<br>13.80<br>14.27<br>13.27|



Table 10: Results of zero-shot composed image retrieval on CIRCO validation set. 

We also report the results of ablation study on FashionIQ dataset. 

Table 11 shows adopting different mapping networks during training, which demonstrates consistent superiority for both EfficientNet B2 and EfficientViT M2. 

Table 12 shows utilizing different query models for ZSCIR, which shows the superiority of lightweight models with pre-training initialization. 

Table 13 shows the comparison of symmetric and asymmetric retrieval on FashionIQ dataset, and shows consistent superiority of our asymmetric structure. 

13 

Published as a conference paper at ICLR 2024 

|**Query model**<br>**Mapping network**|**Dress**<br>**Shirt**<br>**Top&tee**<br>**Average**<br>R@10<br>R@50<br>R@10<br>R@50<br>R@10<br>R@50<br>R@10<br>R@50|
|---|---|
|EffcientNet B2<br>transformer<br>EffcientNet B2<br>MLP (single token)|25.33<br>**46.26**<br>**30.03**<br>48.58<br>**33.45**<br>53.80<br>**29.60**<br>**49.54**<br>23.65<br>43.83<br>28.46<br>47.74<br>31.16<br>51.96<br>27.76<br>47.84|
|EffcientViT M2<br>transformer<br>EffcientViT M2<br>MLP (single token)|**25.48**<br>45.51<br>29.64<br>**48.68**<br>32.94<br>**54.31**<br>29.35<br>49.50<br>24.39<br>44.27<br>29.05<br>47.45<br>30.75<br>53.19<br>28.06<br>48.30|



Table 11: Results of different mapping networks with BLIP on FashionIQ validation set. The best results are shown in bold fonts. 

|**Query model**|**Dress**<br>**Shirt**<br>**Top&tee**<br>**Average**|
|---|---|
||R@10<br>R@50<br>R@10<br>R@50<br>R@10<br>R@50<br>R@10<br>R@50|
|EffcientNet B0<br>EffcientNet B0*<br>EffcientNet B2<br>EffcientNet B2*<br>EffcientViT M2<br>EffcientViT M2*<br>MobileNet V2<br>MobileNet V2*<br>MobileViT V2<br>MobileViT V2*|25.19<br>45.56<br>28.95<br>48.04<br>32.53<br>52.78<br>28.89<br>48.79<br>23.05<br>44.17<br>27.43<br>45.14<br>29.98<br>51.76<br>26.82<br>47.02<br>25.33<br>**46.26**<br>**30.03**<br>48.58<br>**33.45**<br>53.80<br>**29.60**<br>**49.54**<br>25.04<br>44.22<br>28.31<br>46.76<br>31.26<br>53.24<br>28.20<br>48.07<br>**25.48**<br>45.51<br>29.64<br>**48.68**<br>32.94<br>**54.31**<br>29.35<br>49.50<br>24.14<br>43.63<br>27.63<br>45.58<br>31.72<br>53.24<br>27.83<br>47.48<br>24.34<br>45.51<br>28.95<br>47.35<br>32.02<br>54.11<br>28.44<br>48.99<br>24.59<br>44.47<br>27.48<br>45.78<br>31.51<br>53.19<br>27.86<br>47.81<br>22.16<br>40.46<br>26.35<br>43.23<br>27.84<br>49.01<br>25.45<br>44.23<br>21.52<br>39.41<br>25.66<br>42.89<br>27.54<br>48.70<br>24.91<br>43.67|



Table 12: Results of different query models with BLIP on FashionIQ validation set. * denotes adopting random initialization for light query model. The best results are shown in bold fonts. 

Table 14 demonstrates the influence of different losses on FashionIQ, which further verifies the effectiveness of GCD and LAR losses. 

To better understand the performance of zero-shot methods, we compare our methods with the supervised methods that adopt the labeled-triplets for training, including CIRPLANT Liu et al. (2021), ARTEMIS Delmas et al. (2021), and CLIP4Cir Baldrati et al. (2022b). We report the results on CIRR and fashionIQ, Since CIRCO does not include training set so that there is no supervised results for it. From Table 15 and 16, our method could surpass CIRPLANT and ALTEMIS, which shows the effectiveness of leveraging pre-trained BLIP model for zero-shot learning. Although there is still gap compared with CLIP4Cir, our method is getting closer with the supervised state-of-the-art method in contrast to previous zero-shot methods, while achieving deployment flexibility with the asymmetric framework. 

We visualize the results of the attention map of sentence tokens with the text modifier, and to compare them with the spatial attention of adaptive token learner together with the retrieval results. Some good retrieval results are shown in Figure 5. We could see that there are some tokens that attend to both concepts of query images and concepts of text modifier. In the up-right example, the third and 

|**Query model**<br>**Gallery model**|**Dress**<br>**Shirt**<br>**Toptee**<br>**Average**|
|---|---|
||R@10<br>R@50<br>R@10<br>R@50<br>R@10<br>R@50<br>R@10<br>R@50|
|EffcientNet B2<br>BLIP VE<br>EffcientNet B2<br>BLIP VE<br>EffcientViT M2<br>BLIP VE<br>EffcientViT M2<br>BLIP VE<br>BLIP VE_†_<br>BLIP VE<br>BLIP VE<br>BLIP VE|25.33<br>**46.26**<br>30.03<br>48.58<br>33.45<br>53.80<br>29.60<br>49.54<br>0.5<br>2.08<br>0.64<br>2.01<br>0.87<br>1.63<br>0.67<br>1.91<br>**25.48**<br>45.51<br>29.64<br>48.68<br>32.94<br>54.31<br>29.35<br>49.50<br>0.69<br>1.59<br>1.08<br>2.16<br>1.22<br>2.04<br>1.00<br>1.93<br>24.89<br>45.36<br>**30.18**<br>**49.07**<br>**35.65**<br>**55.12**<br>**30.24**<br>**49.85**<br>2.13<br>5.7<br>1.86<br>4.47<br>2.4<br>6.12<br>2.13<br>5.43|



Table 13: Comparison with symmetric retrieval on the validation set of FashionIQ. BLIP VE denotes the BLIP visual encoder. _†_ denotes updating the query BLIP VE while preserving the pre-trained BLIP VE in gallery side during training for maintaining the multi-modality alignment. The best results are shown in bold fonts.. 

14 

Published as a conference paper at ICLR 2024 

|**Query model**<br>**Variants**<br>EffcientNet B2<br>GCD + LAR<br>GCD<br>LAR<br>GCD + LLM TR<br>EffcientViT M2<br>GCD + LAR<br>GCD<br>LAR<br>GCD + LLM TR|**Dress**<br>**Shirt**<br>**Top&tee**<br>**Average**<br>R@10<br>R@50<br>R@10<br>R@50<br>R@10<br>R@50<br>R@10<br>R@50|
|---|---|
||25.33<br>**46.26**<br>**30.03**<br>48.58<br>**33.45**<br>53.80<br>**29.60**<br>**49.54**<br>23.90<br>45.66<br>27.33<br>45.14<br>31.31<br>52.12<br>27.51<br>47.64<br>22.76<br>42.69<br>26.15<br>44.16<br>29.68<br>50.59<br>26.20<br>45.81<br>24.19<br>45.17<br>27.92<br>44.90<br>31.06<br>51.56<br>27.72<br>47.21|
||**25.48**<br>45.51<br>29.64<br>**48.68**<br>32.94<br>**54.31**<br>29.35<br>49.50<br>23.35<br>44.03<br>26.79<br>44.80<br>30.65<br>50.99<br>26.93<br>46.61<br>13.73<br>28.46<br>21.49<br>38.13<br>24.68<br>43.50<br>19.97<br>36.70<br>23.80<br>44.03<br>26.94<br>45.00<br>30.49<br>50.94<br>27.08<br>46.66|



Table 14: Results of different losses on FashionIQ validation set. GCD: global contrastive distillation; LAR: local alignment regularization; LLM TR: text regularization that adopts the sentences made by LLM. The best and second-best results are highlighted in bold and underlined fonts, respectively. 

|**Supervision**<br>**Method**|||**Recall@K**|**Recall@K**|**Avg**<br>R@50|
|---|---|---|---|---|---|
|||R@1|R@5|R@10||
|Supervised<br>CIRPLANT<br>ALTEMIS<br>CLIP4Cir||19.60<br>16.96<br>**33.60**|52.60<br>46.10<br>**65.40**|68.40<br>61.31<br>**77.40**|92.40<br>58.25<br>87.73<br>53.03<br>**95.20**<br>**67.90**|
|Zero-shot<br>Pic2Word(CLIP)<br>SEARLE(CLIP)<br>Pic2word(BLIP)<br>SEARLE(BLIP)<br>Ours(Asy, EN B2)<br>Ours(Asy, EVT M2)||23.90<br>24.22<br>26.70<br>29.27<br>30.84<br>29.63|51.70<br>52.41<br>53.16<br>54.86<br>61.06<br>58.99|65.30<br>66.29<br>64.10<br>66.57<br>73.57<br>71.37|87.80<br>57.17<br>88.63<br>57.88<br>84.36<br>57.08<br>86.16<br>59.21<br>92.43<br>64.48<br>91.47<br>62.87|



Table 15: Comparsion with supervised methods on CIRR test set. The best and second best results are shown in bold and underline fonts, respectively. 

the 6-th tokens focus on “cat” and “fruits” respectively, and the 4-th and 6-th are paying attention to the concepts in the text modifier, _i.e._ , “vegetables” and “bench”, respectively. These examples could extract distinctive visual concepts and interact with the text modifier correspondingly, thus achieving good retrieval performance. 

We also present some failure cases in Figure 6. The failure retrieval could be attributed to the wrong associated concept, rare words, and abstract concepts. For the example in the middle, although the 4-th token catches the concept of “woman” and attends to the “phone”, there is difficulty associating “them” in the text modifier with “dog”, _i.e._ , the 6-th attends to “them” but fail to catch the concept 

|**Supervision**<br>**Method**|**Dress**<br>**Shirt**<br>**Top&tee**<br>**Average**|
|---|---|
||R@10<br>R@50<br>R@10<br>R@50<br>R@10<br>R@50<br>R@10<br>R@50|
|Supervised<br>CIRPLANT<br>ALTEMIS<br>CLIP4Cir|17.45<br>40.41<br>17.53<br>38.81<br>21.64<br>45.38<br>18.87<br>41.53<br>27.20<br>52.40<br>21.80<br>43.60<br>29.20<br>54.80<br>26.10<br>50.30<br>**30.30**<br>**54.50**<br>**37.20**<br>**55.80**<br>**39.20**<br>**61.30**<br>**35.60**<br>**57.20**|
|Zero-shot<br>Pic2Word(CLIP)<br>SEARLE(CLIP)<br>Pic2word(BLIP)<br>SEARLE(BLIP)<br>Ours(Asy, EN B2)<br>Ours(Asy, EVT M2)|20.00<br>40.20<br>26.20<br>43.60<br>27.90<br>47.40<br>24.70<br>43.70<br>20.32<br>43.18<br>27.43<br>45.68<br>29.32<br>50.17<br>25.69<br>46.34<br>21.35<br>42.68<br>27.51<br>46.01<br>29.12<br>49.33<br>25.99<br>46.00<br>22.11<br>41.79<br>29.72<br>48.53<br>31.03<br>52.37<br>27.62<br>47.56<br>25.33<br>46.26<br>30.03<br>48.58<br>33.45<br>53.80<br>29.60<br>49.54<br>25.48<br>45.51<br>29.64<br>48.68<br>32.94<br>54.31<br>29.35<br>49.50|



Table 16: Comparsion with supervised methods on FashionIQ validation set. The best and second best results are shown in bold and underline fonts, respectively. 

15 

Published as a conference paper at ICLR 2024 

of “dog” in the visual feature map. For the example in the bottom, it is difficult for the model to grasp the abstract concept “pattern” and “less of a V-neck”, thus leading to wrong retrieval results. 

We present more visualization comparison of different token lengths in Figure 7 and 8. Generally, to achieve accurate retrieval, the pseudo sentence tokens need to accomplish two aspects: extracting effective visual information and interacting with textual information. When the token length is small, there would be drop of performance due to the lack of capacity to extract sufficient visual information. As shown in Figure 7, query model with small token length performs worse even if the attention maps are reasonable. When the token length increases for too long, on one hand, some tokens may focus on trivial and noisy visual patterns, as shown in Figure 4; on the other hand, these tokens may interact incorrectly with the text modifier, and may impact the correct token-text interaction. As shown in Figure 8, the excessive tokens associate the visual pattern with incorrect text information, potentially interfering with the interaction of multi-modal information. Even if there are accurate associations between other sentence tokens and text modifier, this interference would still degrade the retrieval accuracy. 

To verify the efficiency of our framework, we report the latency on both query side and cloud side on CIRCO vailidation set as shown in Table 17. For query side, we report the inference latency for different query models. We find that lightweight encoders are generally twice faster than large visual encoder (BLIP VE). For cloud side, since the gallery models are the same (BLIP TE & VE), the retrieval latency is the same. 

To further verify the effectiveness of adaptive token learner, we carry out experiments for other three variants of mapping networks: 

(I) simple tokenizer (ST): only using the block described in Eq. (1) and Eq. (2) in section 3.1 in the manuscript. 

(II) adaptive token learner without cross-attention (ATL w/o CA): including the blocks described in Eq. (1), (2), (3). 

(III) adaptive token learner without cross-attention (ATL w/o SA): including the blocks described in Eq. (1), (2), (4). 

We report the results on EfficientNet B2 as below. From the table, it could be seen that with more components added to the mapping network, the performance gradually improves on three benchmark datasets. The simple tokenizer divides the semantics in different groups, which preserves more visual information than MLP. The self-attention models the mutual relationship between different visual tokens and cross-attention filters out noisy information, enhancing the pseudo sentence tokens to be more robust and refined visual representations in word embedding space. Therefore, the full version of adaptive token learner (ATL) achieves the best performance. 

|**Query model**|**Query inference**<br>**latency (ms, perquery)**|**Gallery model**|**Retrieval latency**<br>**(ms, perquery)**|**Total retrieval latency**|
|---|---|---|---|---|
|EffcientNet B0|3.07|BLIP VE|3.58|6.65|
|EffcientNet B2|3.24|BLIP VE|3.58|6.82|
|EffcientViT M2|3.27|BLIP VE|3.58|6.85|
|MobileNet V2|3.03|BLIP VE|3.58|6.61|
|MobileViT V2|3.40|BLIP VE|3.58|6.98|
|BLIP VE|6.53|BLIP VE|3.58|10.11|
|CLIP ViT-L/14|24.44|CLIP ViT-L/14|3.60|15.38|



Table 17: Inference latency on query side and gallery side for different query and gallery models. 

16 

Published as a conference paper at ICLR 2024 

|**Mapping network**|**Components**<br>**Recall@K**<br>**Avg**<br>ST<br>SA<br>CA<br>R@1<br>R@5<br>R@10<br>R@50|
|---|---|
|MLP<br>ST<br>ATL w/o CA<br>ATL w/o SA<br>ATL|30.18<br>59.46<br>71.30<br>89.33<br>62.57<br>✓<br>29.94<br>61.68<br>73.71<br>91.92<br>64.31<br>✓<br>✓<br>29.85<br>61.61<br>74.48<br>91.53<br>64.37<br>✓<br>✓<br>30.35<br>62.13<br>75.95<br>91.89<br>65.08<br>✓<br>✓<br>✓<br>**31.48**<br>**63.42**<br>**77.27**<br>**93.16**<br>**66.33**|



Table 18: Ablation study on variants of mapping networks on CIRR validation set. Best results are shown in bold fonts. 

|**Mapping network**|**Components**<br>**mAP@K**<br>**Avg**<br>ST<br>SA<br>CA<br>mAP@1<br>mAP@5<br>mAP@10<br>mAP@50|
|---|---|
|MLP<br>ST<br>ATL w/o CA<br>ATL w/o SA<br>ATL|8.28<br>8.67<br>9.62<br>10.00<br>9.14<br>✓<br>9.89<br>10.33<br>11.38<br>11.79<br>10.85<br>✓<br>✓<br>10.46<br>10.94<br>12.03<br>12.52<br>11.49<br>✓<br>✓<br>11.52<br>12.15<br>13.83<br>13.58<br>12.77<br>✓<br>✓<br>✓<br>**13.19**<br>**13.83**<br>**15.20**<br>**15.85**<br>**14.52**|



Table 19: Ablation study on variants of mapping networks on CIRCO validation set. Best results are shown in bold fonts. 

|**Mapping network**|**Components**<br>**Dress**<br>**Shirt**<br>**Top&tee**<br>**Average**|
|---|---|
||ST<br>SA<br>CA<br>R@10<br>R@50<br>R@10<br>R@50<br>R@10<br>R@50<br>R@10<br>R@50|
|MLP<br>ST<br>ATL w/o CA<br>ATL w/o SA<br>ATL|23.65<br>43.83<br>28.46<br>47.74<br>31.16<br>51.96<br>27.76<br>47.84<br>✓<br>23.45<br>43.58<br>28.80<br>45.78<br>31.57<br>52.32<br>27.94<br>47.23<br>✓<br>✓<br>24.00<br>45.86<br>27.33<br>45.29<br>31.62<br>52.27<br>27.65<br>47.80<br>✓<br>✓<br>24.15<br>**46.71**<br>28.16<br>46.22<br>32.46<br>52.37<br>28.26<br>48.43<br>✓<br>✓<br>✓<br>**25.33**<br>46.26<br>**30.03**<br>**48.58**<br>**33.45**<br>**53.80**<br>**29.60**<br>**49.54**|



Table 20: Ablation study on variants of mapping networks on FashionIQ. Best results are shown in bold fonts. 

17 

Published as a conference paper at ICLR 2024 


![](raw/papers/2_LLM_NLP/LLM_paper_list/zero-shot/iclr_2024_Image2Sentence_based_Asymmetrical_Zero-shot_Composed_Image_Retrieval_images/iclr_2024_Image2Sentence_based_Asymmetrical_Zero-shot_Composed_Image_Retrieval.pdf-0018-01.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/zero-shot/iclr_2024_Image2Sentence_based_Asymmetrical_Zero-shot_Composed_Image_Retrieval_images/iclr_2024_Image2Sentence_based_Asymmetrical_Zero-shot_Composed_Image_Retrieval.pdf-0018-02.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/zero-shot/iclr_2024_Image2Sentence_based_Asymmetrical_Zero-shot_Composed_Image_Retrieval_images/iclr_2024_Image2Sentence_based_Asymmetrical_Zero-shot_Composed_Image_Retrieval.pdf-0018-03.png)


**----- Start of picture text -----**<br>
has two of  # 5 # 6<br>them and a<br>beer instead<br>of a coke<br>has more<br># 5 # 3<br>boxes and<br>they are on<br>the grass<br>has a  # 6 # 3<br>bench and<br>vegetables<br>instead of<br>fruits<br>is black  # 3 # 4<br>classy dress<br>with white<br>collar and is<br>more formal<br>**----- End of picture text -----**<br>


Figure 5: Some good retrieval examples and the attention map of sentence tokens on visual feature map and with modifier text. Ground-truth target images are marked in green boxes. 

18 

Published as a conference paper at ICLR 2024 


![](raw/papers/2_LLM_NLP/LLM_paper_list/zero-shot/iclr_2024_Image2Sentence_based_Asymmetrical_Zero-shot_Composed_Image_Retrieval_images/iclr_2024_Image2Sentence_based_Asymmetrical_Zero-shot_Composed_Image_Retrieval.pdf-0019-01.png)


**----- Start of picture text -----**<br>
Is redder  # 3 # 4<br>and has<br>more<br>garnishes<br>has only<br># 4 # 6<br>one of them,<br>the woman<br>is talking on<br>the phone<br>Is less<br>revealing  # 6 # 5<br>and has a<br>pattern and<br>less of a V-<br>neck<br>**----- End of picture text -----**<br>


Figure 6: Some failure retrieval case and the attention map of sentence tokens on visual feature map and with modifier text. Ground-truth target images are marked in green boxes. 

19 

Published as a conference paper at ICLR 2024 


![](raw/papers/2_LLM_NLP/LLM_paper_list/zero-shot/iclr_2024_Image2Sentence_based_Asymmetrical_Zero-shot_Composed_Image_Retrieval_images/iclr_2024_Image2Sentence_based_Asymmetrical_Zero-shot_Composed_Image_Retrieval.pdf-0020-01.png)


**----- Start of picture text -----**<br>
L= 6, retrieval results & attention map<br># 3 # 4<br>Is only one<br>and the<br>photo has a<br>body of  L= 2, retrieval results & attention map<br>water in the<br>background<br># 1 # 2<br>L= 6, retrieval results & attention map<br># 5 # 6<br>Is green, has<br>a different<br>shape and is<br>seen from a<br>closer look L= 2, retrieval results & attention map<br># 1 # 2<br>**----- End of picture text -----**<br>


Figure 7: More comparison results of different token lengths (2 v.s. 6). Query image and text modifier are shown on the left, and the attention map of sentence tokens with text modifiers and on visual feature map are shown on the right. Ground-truth target images are marked in green boxes. 

20 

Published as a conference paper at ICLR 2024 


![](raw/papers/2_LLM_NLP/LLM_paper_list/zero-shot/iclr_2024_Image2Sentence_based_Asymmetrical_Zero-shot_Composed_Image_Retrieval_images/iclr_2024_Image2Sentence_based_Asymmetrical_Zero-shot_Composed_Image_Retrieval.pdf-0021-01.png)


**----- Start of picture text -----**<br>
L= 6, retrieval results & attention map<br># 6 # 4<br>Has a red<br>color and is  L= 10, retrieval results & attention map<br>only one, the<br>photo has<br>rocks in the<br>background<br># 4 # 5<br>L= 6, retrieval results & attention map<br># 5 # 6<br>Is green, has  L= 10, retrieval results & attention map<br>a different<br>shape and is<br>seen from a<br>closer look<br># 2 # 7<br>**----- End of picture text -----**<br>


Figure 8: More comparison results of different token lengths (6 v.s. 10). Query image and text modifier are shown on the left, and the attention map of sentence tokens with text modifiers and on visual feature map are shown on the right. Ground-truth target images are marked in green boxes. 

21 

