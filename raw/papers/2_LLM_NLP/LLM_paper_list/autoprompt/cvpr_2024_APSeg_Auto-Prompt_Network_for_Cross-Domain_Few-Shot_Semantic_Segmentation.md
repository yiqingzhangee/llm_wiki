# **APSeg: Auto-Prompt Network for Cross-Domain Few-Shot Semantic Segmentation** 

Weizhao He[1] _[†]_ , Yang Zhang[1] _[†∗]_ , Wei Zhuo[3] _[∗]_ , Linlin Shen[1] _[,]_[2] _[,]_[3] , Jiaqi Yang[3] _[,]_[4] , Songhe Deng[1] , Liang Sun[1] 1Computer Vision Institute, School of Computer Science & Software Engineering, Shenzhen University 2Shenzhen Institute of Artificial Intelligence and Robotics for Society 3National Engineering Laboratory for Big Data System Computing Technology, Shenzhen University 

4School of Computer Science, University of Nottingham, China 

_{_ heweizhao, dengsonghe, sunliang _}_ 2022@email.szu.edu.cn _{_ yangzhang, weizhuo, llshen _}_ @szu.edu.cn jiaqi.yang2@nottingham.edu.cn 

## **Abstract** 

## **1. Introduction** 

_Few-shot semantic segmentation (FSS) endeavors to segment unseen classes with only a few labeled samples. Current FSS methods are commonly built on the assumption that their training and application scenarios share similar domains, and their performances degrade significantly while applied to a distinct domain. To this end, we propose to leverage the cutting-edge foundation model, the Segment Anything Model (SAM), for generalization enhancement. The SAM however performs unsatisfactorily on domains that are distinct from its training data, which primarily comprise natural scene images, and it does not support automatic segmentation of specific semantics due to its interactive prompting mechanism. In our work, we introduce APSeg, a novel auto-prompt network for cross-domain fewshot semantic segmentation (CD-FSS), which is designed to be auto-prompted for guiding cross-domain segmentation. Specifically, we propose a Dual Prototype Anchor Transformation (DPAT) module that fuses pseudo query prototypes extracted based on cycle-consistency with support prototypes, allowing features to be transformed into a more stable domain-agnostic space. Additionally, a Meta Prompt Generator (MPG) module is introduced to automatically generate prompt embeddings, eliminating the need for manual visual prompts. We build an efficient model which can be applied directly to target domains without fine-tuning. Extensive experiments on four cross-domain datasets show that our model outperforms the state-of-the-art CD-FSS method by 5.24% and 3.10% in average accuracy on 1-shot and 5-shot settings, respectively._ 

> _†_ Equal Contribution: Weizhao He and Yang Zhang 

> _∗_ Corresponding Author: Yang Zhang and Wei Zhuo 


![](raw/papers/2_LLM_NLP/LLM_paper_list/autoprompt/cvpr_2024_APSeg_Auto-Prompt_Network_for_Cross-Domain_Few-Shot_Semantic_Segmentation_images/cvpr_2024_APSeg_Auto-Prompt_Network_for_Cross-Domain_Few-Shot_Semantic_Segmentation.pdf-0001-10.png)


**----- Start of picture text -----**<br>
Source  D ataset Target  D atasets<br>Pascal VOC FSS-1000 ISIC<br>transfer<br>Chest X-ray Deepglobe<br>(a)<br>support image prediction label<br>Sampling Point SAM<br>query image ImageSAM PromptVisual MaskBox EncoderPrompt  MaskSAM<br>Encoder Decoder<br>(b)<br>support image prediction label<br>Meta<br>query image ImageSAM TransformationFeature GenerationPrompt MaskSAM<br>Encoder Decoder<br>(c)<br>**----- End of picture text -----**<br>


Figure 1. (a) In CD-FSS tasks, training (source) and testing (target) datasets come from different domains, and categories in the testing dataset are unseen during the training phase. (b) The framework of PerSAM [53], an existing one-shot segmentation method based on SAM. (C) The framework of our proposed APSeg. 

Current deep neural networks [7, 8, 25, 45] depend heavily on extensive annotated data to attain satisfactory performance. Data annotation is a time-consuming task that requires significant human resources, especially for dense pixel-wise annotation for segmentation tasks [8, 20, 23]. The few-shot semantic segmentation (FSS) [33] is therefore introduced to close this gap, aiming to produce pixel-level predictions for a novel category with only a few labeled samples. Although existing FSS methods [20, 26, 30, 38] 

have achieved significant progress, they are commonly built on the assumption that their training and test images are from the same domain. When applied to a different domain, their performance decreases dramatically [21, 37, 42, 43]. The cross-domain generalizability is thus significant and necessary. FSS models typically require a large amount of base data for training. However, images for some tasks, such as cancer diagnosis and remote-sensing analysis, are scarce and challenging to obtain, making training a powerful model directly on their own data nearly impossible. In our work, we aim to transfer knowledge from easily accessible natural domains to data-scarce domains, as shown in Fig. 1(a). We believe that the transfer ability of general models trained on large-scale natural scene datasets will benefit domains that own only a handful of data. 

To accomplish the cross-domain few-shot semantic segmentation (CD-FSS) task, PATNet [21] utilizes support prototypes for computing a transformation matrix, facilitating the conversion of domain-specific features into domainagnostic ones. In addition, PATNet has a further fine-tuning process using the target domain data during the testing phase. Built on [21], RestNet [16] introduces a unified attention module to enhance query and support features prior to transformation. Residual connections are integrated to fuse features before and after the transformation, preserving important information from the original space. Furthermore, RestNet achieves better segmentation results by predicting twice. These works, however, are inefficient in application due to an additional finetuning process or double predictions. In addition, their backbone pretrained on ImageNet [32] limits their performance. 

To this end, we attempt to take advantage of recent achievements on the foundation model, the Segment Anything Model (SAM) [19] to assist CD-FSS. With more than one billion masks under its training, SAM exhibits strong feature extraction and generalization abilities. However, recent studies [6, 9, 27, 39] have reported that applying SAM directly to new domains often yields subpar performance on zero-shot segmentation tasks, particularly when the data distribution significantly differs from the natural domain data used in SAM training. In addition, its interactive framework necessitates manual visual prompts, such as points or boxes, for precise segmentation, which restricts its capability for full automation. Furthermore, some recent works [6, 22, 48] demonstrate SAM is sensitive to manual visual prompts. Even slight deviations in provided prompts can remarkedly affect segmentation accuracy. As shown in Fig. 1(b), PerSAM presents unsatisfactory performance due to its inability to extract high-quality visual prompts. 

In our work, we propose an auto-prompt network for cross-domain few-shot semantic segmentation (APSeg), which builds a novel end-to-end framework that efficiently adapts SAM to CD-FSS tasks for accurate segmentation. 

As shown in Fig. 1(c), the core of our framework is feature transformation and meta prompt generation. In particular, for feature transformation, we propose a Dual Prototype Anchor Transformation (DPAT) module to extract pseudo query prototypes based on cycle-consistency between support and query features. By fusing the pseudo query prototypes with the support prototypes, the computation of the transformation matrix can incorporate information from both support and query samples, which facilitates transforming input features into a more resilient domainagnostic feature space. Combined with DPAT, the potential of SAM in cross-domain scenarios can be unleashed. In addition, for automatic prompt-embedding generation, we introduce a Meta Prompt Generator (MPG) module via a meta-learning procedure. Rather than relying on manual visual prompts such as point and box prompts, MPG leverages support features to guide the generation of meta prompt embeddings associated with target objects to substitute the output of the SAM’s prompt encoder. With MPG, our method is robust and general for automatic segmentation. Our main contributions are summarized as follows: 

- We propose a novel model that integrates a dual prototype anchor transformation (DPAT) module and a meta prompt generator (MPG) module for efficiently adapting SAM to CD-FSS tasks. 

- The DPAT module is proposed for cross-domain feature transformation, which integrates support prototypes and pseudo query prototypes and transforms input features into a stable domain-agnostic space. 

- The MPG module is introduced to generate prompt embeddings through meta-learning to establish a fully automatic framework for segmentation. 

- Extensive experiments on four cross-domain datasets demonstrate that our model outperforms the state-of-theart CD-FSS method by 5.24% and 3.10% in average accuracy on 1-shot and 5-shot settings, respectively. Especially, we attain 17.49% (1-shot) and 14.30% (5-shot) improvements on the Chest X-ray dataset. 

In our method, the parameters of SAM’s image encoder and mask decoder are frozen, and only a few parameters are trainable. Notably, our trained model can directly achieve promising results when applied to target domains without fine-tuning or multiple rounds of inference. 

## **2. Related Work** 

## **2.1. Few-shot Segmentation** 

The goal of few-shot semantic segmentation (FSS) is to segment new classes with a few annotated examples. Current FSS methods are commonly based on meta-learning, which can be largely grouped into two types: prototypebased methods d[4, 41, 47, 54] and matching-based methods [20, 30, 44, 49]. Motivated by PrototypicalNet [36] 


![](raw/papers/2_LLM_NLP/LLM_paper_list/autoprompt/cvpr_2024_APSeg_Auto-Prompt_Network_for_Cross-Domain_Few-Shot_Semantic_Segmentation_images/cvpr_2024_APSeg_Auto-Prompt_Network_for_Cross-Domain_Few-Shot_Semantic_Segmentation.pdf-0003-00.png)


**----- Start of picture text -----**<br>
Feature extraction Cross-domain feature transformation Meta prompt generation Mask prediction<br>support  transformed support<br>Dense<br>Embeddings<br>Image Mask APSeg<br>Encoder query transformed query  Decoder<br>Sparse<br>Embeddings<br>Training Testing<br>Phase Phase<br>Tuning Frozen Matrix Multiplication DPAT Dual Prototype Anchor Transformation MPG Meta Prompt Generator<br>DPAT MPG<br>**----- End of picture text -----**<br>


Figure 2. The overall architecture of our proposed APSeg in a 1-way 1-shot example. After obtaining the multi-layer features of support and query images, DPAT is employed to transform the domain-specific features into domain-agnostic ones through linear transformation. Then, the transformed features are passed into MPG to generate prompt embeddings. At last, the mask decoder takes the prompt embeddings and the transformed high-level query feature as input to make a prediction for the query image. In the testing phase, the trained model is directly applied to complete meta-testing in the target domain. 

for few-shot learning, the prevalent FSS models utilize prototypes for specific-class representation. Recent works [23, 50] point out that a single prototype has a limitation to cover all regions of an object, especially for pixel-wise dense segmentation tasks. To alleviate this problem, methods of [23, 50] use expectation-maximization and cluster algorithms to generate multiple prototypes for different parts of the objects. Compared with prototype-based methods, matching-based ones [28, 38, 52] are designed to extract dense correspondences between query images and support annotations, harnessing pixel-level features to augment the support context with more intricate details. These methods, however, only focus on segmenting new categories from the same domain and fail to generalize unseen domains. 

## **2.2. Cross-domain Few-shot Segmentation** 

In contrast to the traditional FSS setting, CD-FSS necessitates that models refrain from accessing target data during the training process. Furthermore, the data distribution and labeling space in the test phase differ from those in the training phase. This is a more realistic setting. PATNet [21] proposes a feature transformation module which aims to convert domain-specific features into domain-agnostic features. In addition, the target domain data are desired to be utilized to fine-tune the model during the testing phase. Notably, PATNet outperforms current FSS methods on its established benchmark. RestNet [16] utilizes a lightweight attention module to enhance pre-transformation features and merge post-transformation features through residual connections to maintain the key information in the original domain. Meanwhile, a mask prediction strategy is introduced to mitigate the issue of overfitting to support samples in FSS and facilitates the model in a gradual acquisi- 

tion of cross-domain segmentation capabilities. However, existing methods still utilize classical classification models [15, 35] as feature extractors with limited feature extraction capabilities compared to existing visual foundation models [19, 29, 31]. The performance of cross-domain segmentation is limited. Moreover, either additional training or multiple inferences is required when predicting masks, which makes the inference process complex and inefficient. 

## **2.3. Segment Anything Model** 

The Segment Anything Model (SAM) [19], pretrained on massive amounts of labeled data, first introduced a foundation model for image segmentation. SAM relies on explicit points and bounding boxes at precise locations for accurate segmentation [10, 39]. Therefore, extensive manual guidance or a specialist detector is often required, leading to a complex multi-stage pipeline [40]. SAM cannot achieve automatic segmentation for specific semantics. To address this, PerSAM [53] proposes automatic sampling of visual prompts, and some other methods suggest directly generating prompt embeddings [6, 48]. Inspired by these works, we propose an automatic prompting method in a meta-learning manner to adapt SAM to CD-FSS. 

## **3. Method** 

## **3.1. Problem Definition** 

For CD-FSS, a source domain ( _Xs, Ys_ ) and a target domain ( _Xt, Yt_ ) exist, where the input distribution of the source domain and target domain are different, and their label space has no intersection as well, i.e., _Xs_ = _Xt_ and _Ys ∩ Yt_ = _∅_ . Here _X_ denotes input distribution, and _Y_ denotes the label space. In our work, we train and test our 


![](raw/papers/2_LLM_NLP/LLM_paper_list/autoprompt/cvpr_2024_APSeg_Auto-Prompt_Network_for_Cross-Domain_Few-Shot_Semantic_Segmentation_images/cvpr_2024_APSeg_Auto-Prompt_Network_for_Cross-Domain_Few-Shot_Semantic_Segmentation.pdf-0004-00.png)


**----- Start of picture text -----**<br>
foreground matching background matching<br>support image query image support image query image<br>+ +<br>+<br>+<br>+<br>+<br>forward matching backward matching + foreground points background points<br>match correctly<br>match failed<br>**----- End of picture text -----**<br>


Figure 3. A visual example of a support-query pair to perform cycle-consistent selection (CCS). 

model in a meta-learning episodic manner following [21], and our model is only trained on the source domain and has no access to the target data. Each episode data consists of a support set _S_ and a query set _Q_ with a specific category. The support set _S_ = (I[s] _i[,]_[ M][s] _i_[)] _[K] i_ =1[contains] _[ K]_[image-mask pairs,] where I[s] _i_[denotes][the] _[i]_[-th][support][image][and][M][s] _i_[denotes] the corresponding binary mask. Similarly, the query set is defined as _Q_ = (I[q] _i[,]_[ M][q] _i_[)] _[K] i_ =1[.][To][train][our][model,][support] sets and images from query sets are used as model inputs to predict the query masks. To assess the trained model’s performance, we test it on a support set and a query set from the target domain. 

## **3.2. Method Overview** 

Our target is to train a general model on natural domains with rich annotations and transfer the knowledge to target domains with limited labeled data. As illustrated in Fig. 2, our proposed APSeg consists of two key modules: the Dual Prototype Anchor Transformation (DPAT) module and the Meta Prompt Generator (MPG) module. Specifically, given the support image I[s] and query image I[q] , the SAM image encoder is used to extract multi-level features from different layers. DPAT module is then employed to map support and query features to a new domain-agnostic space, facilitating the rapid adaptation of subsequent modules for previously unseen domains. Next, we introduce the MPG module whose task is to generate sparse and dense prompt embeddings through the meta-learning procedure for the SAM’s mask decoder by utilizing the transformed features. At last, the generated prompt embeddings and the transformed high-level query features are passed into the mask decoder for target mask prediction. 

## **3.3. Dual Prototype Anchor Transformation** 

To map support and query features into a new domainagnostic space, we introduce a Dual Prototype Anchor Transformation (DPAT) module, as shown in Fig. 4. Previous method [21] solely relies on a set of support proto- 


![](raw/papers/2_LLM_NLP/LLM_paper_list/autoprompt/cvpr_2024_APSeg_Auto-Prompt_Network_for_Cross-Domain_Few-Shot_Semantic_Segmentation_images/cvpr_2024_APSeg_Auto-Prompt_Network_for_Cross-Domain_Few-Shot_Semantic_Segmentation.pdf-0004-07.png)


**----- Start of picture text -----**<br>
Dual Prototype Anchor Transformation<br>1 1 1 …… 1 1 1<br>MAP ...  ...<br>Anchor Layer<br>A<br>CCS<br>**----- End of picture text -----**<br>


Figure 4. The specific implementation of the Dual Prototype Anchor Transformation module. Pseudo foreground and background prototypes of query are extracted through cycle-consistent selection (CCS). The extracted pseudo prototypes are then fused with support prototypes to calculate the transformation matrix **W** with an anchor layer. 

types and anchor layers to compute a transformation matrix. However, support prototypes cannot well represent complete information of a category due to intra-class variance. Therefore, we propose to enhance the support prototype set with query prototypes for better feature transformation. Specifically, our proposed DPAT consists of two procedures: dual prototype enhancement and crossdomain feature transformation. Inspired by [52], we propose cycle-consistent selection (CCS) to extract both the query foreground and background prototypes in the absence of query masks, which is used to enhance the support prototypes. Based on these enhanced prototypes which represent a category and its surroundings, an effective transformation matrix can be computed with a learnable domain-agnostic module. The transformation matrix is then applied to support and query features for cross-domain feature transformation. 

**Dual Prototype Enhancement** Representative prototypes are significant for our cross-domain transformation. To this end, we build a cycling-examine procedure that reasons on both the query foregrounds and backgrounds to augment support prototypes. The process is shown in Fig. 3. We conduct forward matching to attain query features that have the highest similarity with the support foregrounds. We then use these identified forward-matched query features to re-locate corresponding support features reversely. If the located support features through reverse matching fall within support foregrounds, the identified query features will be averaged and used to derive the foreground prototypes. The background prototypes are obtained with the same process. Finally, support prototypes and query prototypes are fused by addition. 

Specifically, given a support image I[s] and a query image I[q] , we initially employ a shared-weight image encoder to extract their multi-layer feature maps _{_ f _l_[s] _[}]_[3] _l_ =1[and] _[{]_[f] _l_[q] _[}]_[3] _l_ =1 respectively, where f _l_[s][,][ f] _l_[q] _[∈]_[R] _[c][l][×][h][×][w]_[.] _[ c][l][, h, w]_[ are the chan-] nel dimension, height, and width of the feature map. DPAT 

takes support features _{_ f _l_[s] _[}]_[3] _l_ =1[,][query][features] _[{]_[f] _l_[q] _[}]_[3] _l_ =1[and] support mask M[s] as input. To simplify the notations, f[s] and f[q] are used to represent any one of _{_ f _l_[s] _[}]_[3] _l_ =1[and] _[ {]_[f] _l_[q] _[}]_[3] _l_ =1[. We] then employ masked average pooling (MAP) on the support feature to obtain the foreground and background prototype, denoted as **p**[s] fg[and] **[ p]**[s] bg[.][By concatenating] **[ p]**[s] fg[and] **[ p]**[s] bg[, we] get the support prototype matrix **P**[s] = � **p**[s] fg _[,]_ **[ p]**[s] bg�. Considering that the query mask cannot be accessed during training, we extract pseudo query prototypes through CCS. First, the support foreground feature is obtained by multiplying the support mask M[s] with the support feature f[s] . Next, the similarity between the support foreground feature and the query feature is calculated. For each element of the support foreground feature, we search for the element in the query feature map with the highest similarity score, and acquire the matched position set _i_[s] _[→]_[q] from support to query as follows, 


![](raw/papers/2_LLM_NLP/LLM_paper_list/autoprompt/cvpr_2024_APSeg_Auto-Prompt_Network_for_Cross-Domain_Few-Shot_Semantic_Segmentation_images/cvpr_2024_APSeg_Auto-Prompt_Network_for_Cross-Domain_Few-Shot_Semantic_Segmentation.pdf-0005-01.png)


where sim( _·, ·_ ) is a cosine function. Based on _i_[s] _[→]_[q] , the matched query feature f _i_[q][s] _[→]_[q][can be extracted.][Similarly, the] matched positions _j_[s] _[←]_[q] from query to support can obtained as below, 


![](raw/papers/2_LLM_NLP/LLM_paper_list/autoprompt/cvpr_2024_APSeg_Auto-Prompt_Network_for_Cross-Domain_Few-Shot_Semantic_Segmentation_images/cvpr_2024_APSeg_Auto-Prompt_Network_for_Cross-Domain_Few-Shot_Semantic_Segmentation.pdf-0005-03.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/autoprompt/cvpr_2024_APSeg_Auto-Prompt_Network_for_Cross-Domain_Few-Shot_Semantic_Segmentation_images/cvpr_2024_APSeg_Auto-Prompt_Network_for_Cross-Domain_Few-Shot_Semantic_Segmentation.pdf-0005-04.png)


where _j_[s] _[←]_[q] is the positions on support that have the most similar features with the corresponding reference query features f _i_[q][s] _[→]_[q][.] If the position in the matched position set _j_[s] _[←]_[q] does not fall in the support mask M[s] , we filter out thetion setposition _i′_ s _→_ qfrom. According to the set _i_[s] _[→]_[q] and obtain the _i′_ s _→_ finalq, the correspond-matched posiing features are extracted from f[q] and averaged to obtain the pseudo foreground query prototype **p**[q] fg[.][The][pseudo] background query prototype **p**[q] bg[can][be][obtained][through] a similar process, resulting in the query prototype matrix **P**[q] = � **p**[q] fg _[,]_ **[ p]**[q] bg�. Finally, a mixed prototype matrix can be obtained by **P**[m] = **P**[s] + **P**[q] . 

**Cross-domain Feature Transformation** Features of the same class yield similar results when they are transformed in the same way. Support features and query features are transformed into a domain-agnostic space using the same transformation matrix **W** to avoid the detrimental impact caused by domain shift. Given the weight matrix of an anchor layer **A** , the definition of a transformation matrix is as follows, 


![](raw/papers/2_LLM_NLP/LLM_paper_list/autoprompt/cvpr_2024_APSeg_Auto-Prompt_Network_for_Cross-Domain_Few-Shot_Semantic_Segmentation_images/cvpr_2024_APSeg_Auto-Prompt_Network_for_Cross-Domain_Few-Shot_Semantic_Segmentation.pdf-0005-07.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/autoprompt/cvpr_2024_APSeg_Auto-Prompt_Network_for_Cross-Domain_Few-Shot_Semantic_Segmentation_images/cvpr_2024_APSeg_Auto-Prompt_Network_for_Cross-Domain_Few-Shot_Semantic_Segmentation.pdf-0005-08.png)


**----- Start of picture text -----**<br>
Meta Prompt Generator<br>C<br>PE<br>Transformer<br>C<br>Decoder<br>C<br>PE Positional Encoding<br>Conv MAP Linear<br>Conv MLP<br>PMG Conv FEM Conv<br>**----- End of picture text -----**<br>


Figure 5. The specific implementation of the Meta Prompt Generator module. PE indicates a positional encoding. 

where **P**[m] = � _∥_ **pp** mf[m] fgg _[∥][,] ∥_ **pp**[m] b[m] bgg _[∥]_ �, **A** = � _∥_ **aa** ffgg _∥[,] ∥_ **aa** bbgg _∥_ � and **a** is the anchor vector which is independent of the input. Different from previous work [21], we leverage the mixed prototype matrix that incorporates both support and query feature information during the computation process. Since the prototype **P**[m] is a non-square matrix, the generalized inverse [1] of **P**[m] is calculated with **P**[m+] = _{_ **P**[m] **[T] P**[m] _}[−]_[1] **P**[m] **[T]** . Therefore, the transformation matrix is calculated as **W** = **AP**[m+] . In our work, We have two different anchor layers for mid-level features _{_ f1[s] _[,]_[ f] 2[s] _[,]_[ f] 1[q] _[,]_[ f] 2[q] _[}]_[ and high-level features] _{_ f3[s] _[,]_[ f] 3[q] _[}]_[.][Finally, we can efficiently map support and query] features to a stable, domain-agnostic space by multiplying them with **W** . 

Objects and things of even the same class can differ in shape and appearance. Due to limited samples of supports, it is intrinsically challenging to represent all the variance within objects and things of a class. Through the doublecheck procedure for both foreground and background regions, our proposed DPAT can effectively mitigate the challenges aroused by intra-class variances and generate a more stable transformation matrix for cross-domain feature transformation. 

## **3.4. Meta Prompt Generator** 

To construct an end-to-end fully automated SAM-based segmentation framework for CD-FSS, we utilize the features of support-query pairs to directly generate prompt embeddings. In particular, a meta prompt generator (MPG) module is designed to obtain both sparse and dense embeddings simultaneously, as shown in Fig. 5. Different from [4] that uses single support embeddings without alignment, our new pipeline extends to leverage multiple support embeddings and integrates feature alignment. Our design comprehensively takes account of intra-class variance and the multiple prompts coming along with the support embed- 

dings enhance the segmentation. For clarity, we refer to the process of generating sparse embeddings and dense embeddings as sparse path and dense path respectively. In this way, our method eliminates the need for external manual visual prompts, such as points or boxes. 

**Sparse path.** In this process, the query features and several support embeddings augmented from the support prototypes are utilized to generate sparse embeddings through a transformer decoder [5], which then replace the original sparse embeddings in SAM. First, we concatenate the transformed mid-level support features[ˆ] f1[s] _[∈]_[R] _[c]_[1] _[×][h][×][w]_[and] ˆf2[s] _[∈]_[R] _[c]_[2] _[×][h][×][w]_[along the channel dimension, and then per-] form dimension reduction through a convolution layer to obtain[ˆ] f[s] _∈_ R _[c][r][×][h][×][w]_ . In the same way, we can also get ˆf[q] _∈_ R _[c] r[×][h][×][w]_ , 


![](raw/papers/2_LLM_NLP/LLM_paper_list/autoprompt/cvpr_2024_APSeg_Auto-Prompt_Network_for_Cross-Domain_Few-Shot_Semantic_Segmentation_images/cvpr_2024_APSeg_Auto-Prompt_Network_for_Cross-Domain_Few-Shot_Semantic_Segmentation.pdf-0006-02.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/autoprompt/cvpr_2024_APSeg_Auto-Prompt_Network_for_Cross-Domain_Few-Shot_Semantic_Segmentation_images/cvpr_2024_APSeg_Auto-Prompt_Network_for_Cross-Domain_Few-Shot_Semantic_Segmentation.pdf-0006-03.png)


where _Fconv_ means performing a 1 _×_ 1 convolution followed by a ReLU activation function and _⊕_ denotes the concatenation operation in channel dimension. 

Next, we take[ˆ] f[s] and M[s] as input and apply MAP to obtain the foreground class prototypes **ˆp** _[s] ∈_ R _[c][r]_ . Then, a linear layer maps the **ˆp** _[s]_ to multiple augmented support embeddings **E**[aug] _∈_ R _[k][×][c][r]_ , 


![](raw/papers/2_LLM_NLP/LLM_paper_list/autoprompt/cvpr_2024_APSeg_Auto-Prompt_Network_for_Cross-Domain_Few-Shot_Semantic_Segmentation_images/cvpr_2024_APSeg_Auto-Prompt_Network_for_Cross-Domain_Few-Shot_Semantic_Segmentation.pdf-0006-06.png)


where _k_ denotes the number of the augmented support embeddings. Here we generate several embeddings instead of only a single embedding for simulating multiple point prompts. 

To supplement positional information, learnable position encodings are applied to **E**[aug] and fixed sine-cosine positional encodings are applied to[ˆ] f[q] . We then input them into the transformer decoder and its output is further processed by a two-layer MLP to increase the channel dimensions, which yields **E**[ˆ][aug] _∈_ R _[k][×][c][o]_ . 


![](raw/papers/2_LLM_NLP/LLM_paper_list/autoprompt/cvpr_2024_APSeg_Auto-Prompt_Network_for_Cross-Domain_Few-Shot_Semantic_Segmentation_images/cvpr_2024_APSeg_Auto-Prompt_Network_for_Cross-Domain_Few-Shot_Semantic_Segmentation.pdf-0006-09.png)


To align the generated sparse embeddings with those produced by SAM’s prompt encoder, the sine function is employed to generate the final sparse embeddings **E**[spa] _∈_ R _[k][×][c][o]_ following [6]. 


![](raw/papers/2_LLM_NLP/LLM_paper_list/autoprompt/cvpr_2024_APSeg_Auto-Prompt_Network_for_Cross-Domain_Few-Shot_Semantic_Segmentation_images/cvpr_2024_APSeg_Auto-Prompt_Network_for_Cross-Domain_Few-Shot_Semantic_Segmentation.pdf-0006-11.png)


**Dense path.** In this process, the dense embeddings are modulated by query features and support prototypes.[ˆ] f3[s][,][ ˆf] 3[q][and][M][s][are][first][passed][to][prior][mask][genera-] tion (PMG) module [38] to generate a prior mask M[pr] _∈_ R[1] _[×][h][×][w]_ . After concatenating **ˆp**[s] ,[ˆ] f[q] and M[pr] , we perform a 1 _×_ 1 convolution for dimension reduction to obtain 

ˆf[pr] _∈_ R _[c] r[×][h][×][w]_ . The output is then passed into the feature enhancement (FEM) module [38] to get[ˆ] f[fem] .[ˆ] f[fem] is further processed by a 1 _×_ 1 convolution layer to increase the channel dimensions to obtain **E**[den] _∈_ R _[c][o][×][h][×][w]_ . 

## **3.5. Training Loss** 

In the training of APSeg, we employ a Dice loss function, computed between the predicted mask M[ˆ] and the corresponding ground truth query mask M[q] . The loss function, denoted as _L_ , is expressed as: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/autoprompt/cvpr_2024_APSeg_Auto-Prompt_Network_for_Cross-Domain_Few-Shot_Semantic_Segmentation_images/cvpr_2024_APSeg_Auto-Prompt_Network_for_Cross-Domain_Few-Shot_Semantic_Segmentation.pdf-0006-16.png)


Here, _n_ represents the total number of training episodes in each batch, and DICE signifies the Dice loss function. The function _I_ serves as an interpolation function, ensuring that M[ˆ] shares the same spatial size as M[q] . 

## **4. Experiment** 

**Datasets.** Following the previous approach [21], we use PASCAL VOC 2012 [13] with SBD [14] augmentation as training dataset and then evaluate the trained model on Chest X-ray [3, 17], ISIC [11], FSS-1000 [24] and Deepglobe [12] respectively. 

**Metric and Evaluation.** We use the mean intersectionover-union (mIoU) as the evaluation metric, which is the same as the previous method. We take the mean-IoU of 5 runs [28] with different random seeds for each test. For all datasets except FSS-1000, each run has 1200 tasks. Every run of FSS-1000 has 2400 tasks. 

**Implementation Details.** In our experiments, we employ the base version of the SAM and keep it frozen during training. To be consistent with the original input to SAM, we set spatial sizes of both support and query images to 1024 _×_ 1024. For the SAM image encoder, we utilize feature maps derived from the mid-level features output by the 5[th] and 8[th] transformer blocks, in addition to the high-level features obtained from the final output of the image encoder. Concerning the DPAT module, we create two anchor layers dedicated to mid-level and high-level features, each configured with channel numbers set to 768 and 256, respectively. For the MPG module, the number of feature channels after dimension reduction is set to 64. Additionally, the number of sparse embeddings and dense embeddings channels output by MPG is set to 256. We implement the model in PyTorch and utilize the Adam [18] optimizer with a learning rate of 1e _−_ 3. 

## **4.1. Comparison with State-of-the-Arts** 

We compare our method against existing CNN-based and SAM-based approaches for CD-FSS. As shown in 


![](raw/papers/2_LLM_NLP/LLM_paper_list/autoprompt/cvpr_2024_APSeg_Auto-Prompt_Network_for_Cross-Domain_Few-Shot_Semantic_Segmentation_images/cvpr_2024_APSeg_Auto-Prompt_Network_for_Cross-Domain_Few-Shot_Semantic_Segmentation.pdf-0007-00.png)


**----- Start of picture text -----**<br>
support image ground truth prediction support image ground truth prediction<br>Chest X-ray<br>ISIC<br>FSS-1000<br>Deepglobe<br>**----- End of picture text -----**<br>


Figure 6. Qualitative results of APSeg in 1-way 1-shot segmentation on CD-FSS. Note that the model is trained on PASCAL VOC. Support labels are overlaid in blue. The prediction and ground truth of query images are in green and red respectively. Best viewed in color. 

|Methods|Backbone|Chest X-ray|Chest X-ray|ISIC|ISIC|FSS-1000|FSS-1000|Deepglobe|Deepglobe|Average|Average|
|---|---|---|---|---|---|---|---|---|---|---|---|
|||1-shot|5-shot|1-shot|5-shot|1-shot|5-shot|1-shot|5-shot|1-shot|5-shot|
|||Few-shot Semantic Segmentation Methods||||||||||
|AMP [34]<br>PGNet [50]<br>PANet [41]<br>CaNet [51]<br>RPMMs [46]<br>PFENet [38]<br>RePRI [2]<br>HSNet [28]<br>PerSAM[53]|VGG-16<br>Res-50<br>Res-50<br>Res-50<br>Res-50<br>Res-50<br>Res-50<br>Res-50<br>ViT-base|51.23<br>33.95<br>57.75<br>28.35<br>30.11<br>27.22<br>65.08<br>51.88<br>29.95|53.04<br>27.96<br>69.31<br>28.62<br>30.82<br>27.57<br>65.48<br>54.36<br>30.05|28.42<br>21.86<br>25.29<br>25.16<br>18.02<br>23.50<br>23.27<br>31.20<br>23.27|30.41<br>21.25<br>33.99<br>28.22<br>20.04<br>23.83<br>26.23<br>35.10<br>25.33|57.18<br>62.42<br>69.15<br>70.67<br>65.12<br>70.87<br>70.96<br>77.53<br>60.92|59.24<br>62.74<br>71.68<br>72.03<br>67.06<br>70.52<br>74.23<br>80.99<br>66.53|37.61<br>10.73<br>36.55<br>22.32<br>12.99<br>16.88<br>25.03<br>29.65<br>36.08|40.61<br>12.36<br>**45.43**<br>23.07<br>13.47<br>18.01<br>27.41<br>35.08<br>40.65|43.61<br>32.24<br>47.19<br>36.63<br>31.56<br>34.62<br>46.09<br>47.57<br>37.56|45.83<br>31.08<br>55.10<br>37.99<br>32.85<br>34.98<br>48.34<br>51.38<br>40.64|
|||Cross-domain Few-shot Semantic Segmentation Methods||||||||||
|PATNet [21]<br>PATNet_‡_ [21]<br>RestNet [16]<br>**APSeg(ours)**|Res-50<br>ViT-base<br>Res-50<br>ViT-base|66.61<br>76.43<br>70.43<br>**84.10**|70.20<br>-<br>73.69<br>**84.50**|41.16<br>44.25<br>42.25<br>**45.43**|53.58<br>-<br>51.10<br>**53.98**|78.59<br>72.03<br>**81.53**<br>79.71|81.23<br>-<br>**84.89**<br>81.90|**37.89**<br>22.37<br>-<br>35.94|42.97<br>-<br>-<br>39.98|56.06<br>53.77<br>-<br>**61.30**|61.99<br>-<br>-<br>**65.09**|



Table 1. Comparison with previous FSS and CD-FSS methods under 1-way 1-shot and 1-way 5-shot settings. All the methods are trained on the source dataset and tested on the CD-FSS benchmark. The method marked with _‡_ is implemented by ourselves. 

Tab. 1, the results demonstrate the superiority of the proposed method in this challenging task. Specifically, our approach exhibits improvements of 5.40% and 3.10% compared to the PATNet [21], under 1-shot and 5-shot settings, respectively. Moreover, APSeg surpasses the SAM-based method PerSAM [53] by 23.74% and 24.45%, affirming the effectiveness of our approach. Notably, our method showcases significantly superior performance compared to the current methods when confronted with large domain gaps between the testing and training datasets. Specifically, compared with PATNet, our method achieves improvements of 17.49% and 14.30% on the chest X-ray dataset under the 1-shot and 5-shot settings, respectively. Similarly, on the ISIC dataset, improvements of 4.27% are observed for the 

1-shot setting. Furthermore, our automatic prompting approach significantly surpasses the manual prompting strategy proposed by [53] in cross-domain performance. For a fair comparison, we also implement PATNet based on the SAM’s image encoder. The model is trained with the same input image size, and test-time fine-tuning is not employed. The results demonstrate that APSeg still maintains a significant superiority. Qualitative results, illustrated in Fig. 6, validate that our proposed method attains substantial improvements in generalization performance in the presence of large domain gap while maintaining considerable accuracy with minor domain shift. More visualization results are provided in the supplementary materials. 

## **4.2. Ablation study** 

**Components analysis.** We assess the effectiveness of our proposed DPAT module and MPG module by using the 1- shot mIoU averaged on four datasets. To establish a baseline model, we first remove the DPAT module and the sparse path in the MPG module and then replace the final layer of the dense path with a 1 _×_ 1 convolution layer for predicting segmentation masks. Tab. 2 illustrates the impact of each component on model performance. Overall, the incorporation of the two components suggested in this paper enhances the baseline by 18.44%. In the second row, MPG leverages the segmentation capabilities of SAM by autonomously generating semantic-aware prompt embeddings, eliminating the need for manual prompts and improving the baseline by 2.80%. Upon combining MPG and DPAT, DPAT unleashes the segmentation capabilities of SAM in cross-domain scenarios. It achieves this by transforming input features into a more stable domain-agnostic feature space, resulting in a significant performance improvement of 15.46% compared to the second row. More analysis and discussion about DPAT are shown in supplementary materials. 

|mentary materials.|mentary materials.|mentary materials.|mentary materials.|mentary materials.|mentary materials.|
|---|---|---|---|---|---|
|Method<br>Chest X-ray<br>ISIC<br>FSS-1000<br>Deepglobe<br>Average||||||
|Baseline<br>Baseline + MPG<br>Baseline + MPG + DPAT|28.80<br>32.62<br>84.10|44.55<br>36.67<br>45.43|77.90<br>79.77<br>79.71|20.19<br>34.31<br>35.94|42.86<br>45.84<br>61.30|



Table 2. Ablation study on key components of APSeg on CD-FSS. Results are averaged over four datasets for 1-shot. 

|Sparse|Dense|1-shot mIoU|
|---|---|---|
|✓<br>✓|✓<br>✓|41.52<br>44.09<br>45.43|



Table 3. Ablation study on the choice of different types of prompt embeddings in MPG on ISIC. 

**Meta Prompt Generator.** Tab. 3 shows the impact of the main components in the MPG, namely sparse embeddings and dense embeddings. We show the results for three combination scenarios: using only sparse embeddings, only dense embeddings, and both. Our observation reveals that better performance can be attained when combining sparse and dense embeddings. This emphasizes the significance of leveraging both types of embeddings to harness the segmentation capabilities of SAM in cross-domain scenarios. **The number of feature channels.** After cross-domain feature transformation, we fuse the transformed mid-level features and perform dimension reduction, which can reduce the number of learnable parameters and avoid the trained model overfitting the training dataset. Observation in Tab. 4 shows that reducing the number of channels to 64 allows our method to achieve better CD-FSS performance with only a small number of additional learnable parameters. 

|dim|1-shot mIoU|# Learnable Params|
|---|---|---|
|64<br>128<br>256|61.30<br>60.03<br>58.65|0.7 M<br>2.3 M<br>8.4 M|



Table 4. Ablation study on different output feature channels. Results are averaged over four datasets for 1-shot. **The number of sparse embeddings.** Our proposed MPG introduces an automatic generation mechanism for prompt embeddings, replacing SAM’s original method of obtaining them through manual prompts fed into the prompt encoder. Employing more manual visual prompts has been shown to enhance SAM’s interactive segmentation performance. Therefore, Tab. 5 shows the association between the number of sparse embeddings generated by MPG and the performance of cross-domain few-shot segmentation with APSeg. Notably, generating 4 sparse embeddings results in a 0.55% improvement in 1-shot scenarios. However, as indicated in the third row, continuing to generate more prompt embeddings may lead to a decline in performance. 

|y lead to a|decline in perfo|
|---|---|
|num|1-shot mIoU|
|1<br>4<br>8|60.75<br>61.30<br>60.65|



Table 5. Ablation study on the different number of sparse embeddings. Results are averaged over four datasets for 1-shot. 

## **5. Conclusion** 

In this paper, we introduce APSeg, an auto-prompt method for guiding SAM to complete CD-FSS tasks. To achieve fully automatic segmentation based on SAM and release the segmentation capability of SAM in cross-domain scenarios, we propose the Meta Prompt Generator (MPG) module and Dual Prototype Anchor Transformation (DPAT) module to achieve this goal. By fusing the extracted pseudo query prototypes with support prototypes, DPAT enables domain-specific input features to be more stably converted into domain-agnostic features, significantly improving cross-domain generalization capabilities. In addition, MPG generates semantic-aware prompt embeddings with meta-learning, promoting the construction of a fully automatic CD-FSS framework based on SAM. Combining DPAT and MPG, extensive experimental results show that our APSeg achieves a new state-of-the-art in CD-FSS. 

## **6. Acknowledgements** 

This work is supported by the National Natural Science Foundation of China under Grant 62176163, 62306183 and 82261138629; the Science and Technology Foundation of Guangdong Province under Grant 2021A1515012303, 2024A1515010194 and 2023A1515010688; the Science and Technology Foundation of Shenzhen under Grant JCYJ20210324094602007 and JCYJ20220531101412030; and Shenzhen University Startup Funding. 

## **References** 

- [1] Adi Ben-Israel and Thomas NE Greville. _Generalized inverses: theory and applications_ . Springer Science & Business Media, 2003. 

- [2] Malik Boudiaf, Hoel Kervadec, Ziko Imtiaz Masud, Pablo Piantanida, Ismail Ben Ayed, and Jose Dolz. Few-shot segmentation without meta-learning: A good transductive inference is all you need? In _Proceedings of the IEEE/CVF conference on computer vision and pattern recognition_ , pages 13979–13988, 2021. 

- [3] Sema Candemir, Stefan Jaeger, Kannappan Palaniappan, Jonathan P Musco, Rahul K Singh, Zhiyun Xue, Alexandros Karargyris, Sameer Antani, George Thoma, and Clement J McDonald. Lung segmentation in chest radiographs using anatomical atlases with nonrigid registration. _IEEE transactions on medical imaging_ , 33(2):577–590, 2013. 

- [4] Leilei Cao, Yibo Guo, Ye Yuan, and Qiangguo Jin. Prototype as query for few shot semantic segmentation. _arXiv preprint arXiv:2211.14764_ , 2022. 

- [5] Nicolas Carion, Francisco Massa, Gabriel Synnaeve, Nicolas Usunier, Alexander Kirillov, and Sergey Zagoruyko. End-toend object detection with transformers. In _European conference on computer vision_ , pages 213–229. Springer, 2020. 

- [6] Keyan Chen, Chenyang Liu, Hao Chen, Haotian Zhang, Wenyuan Li, Zhengxia Zou, and Zhenwei Shi. Rsprompter: Learning to prompt for remote sensing instance segmentation based on visual foundation model. _IEEE Transactions on Geoscience and Remote Sensing_ , 2024. 

- [7] Liang-Chieh Chen, George Papandreou, Iasonas Kokkinos, Kevin Murphy, and Alan L Yuille. Deeplab: Semantic image segmentation with deep convolutional nets, atrous convolution, and fully connected crfs. _IEEE transactions on pattern analysis and machine intelligence_ , 40(4):834–848, 2017. 

- [8] Liang-Chieh Chen, Yukun Zhu, George Papandreou, Florian Schroff, and Hartwig Adam. Encoder-decoder with atrous separable convolution for semantic image segmentation. In _Proceedings of the European conference on computer vision (ECCV)_ , pages 801–818, 2018. 

- [9] Tianrun Chen, Lanyun Zhu, Chaotao Deng, Runlong Cao, Yan Wang, Shangzhan Zhang, Zejian Li, Lingyun Sun, Ying Zang, and Papa Mao. Sam-adapter: Adapting segment anything in underperformed scenes. In _Proceedings of the IEEE/CVF International Conference on Computer Vision_ , pages 3367– 3375, 2023. 

- [10] Dongjie Cheng, Ziyuan Qin, Zekun Jiang, Shaoting Zhang, Qicheng Lao, and Kang Li. Sam on medical images: A comprehensive study on three prompt modes. _arXiv preprint arXiv:2305.00035_ , 2023. 

- [11] Noel Codella, Veronica Rotemberg, Philipp Tschandl, M Emre Celebi, Stephen Dusza, David Gutman, Brian Helba, Aadi Kalloo, Konstantinos Liopyris, Michael Marchetti, et al. Skin lesion analysis toward melanoma detection 2018: A challenge hosted by the international skin imaging collaboration (isic). _arXiv preprint arXiv:1902.03368_ , 2019. 

- [12] Ilke Demir, Krzysztof Koperski, David Lindenbaum, Guan Pang, Jing Huang, Saikat Basu, Forest Hughes, Devis Tuia, and Ramesh Raskar. Deepglobe 2018: A challenge to parse 

   - the earth through satellite images. In _Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition Workshops_ , pages 172–181, 2018. 

- [13] Mark Everingham, Luc Van Gool, Christopher KI Williams, John Winn, and Andrew Zisserman. The pascal visual object classes (voc) challenge. _International journal of computer vision_ , 88:303–338, 2010. 

- [14] Bharath Hariharan, Pablo Arbel´aez, Lubomir Bourdev, Subhransu Maji, and Jitendra Malik. Semantic contours from inverse detectors. In _2011 international conference on computer vision_ , pages 991–998. IEEE, 2011. 

- [15] Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun. Deep residual learning for image recognition. In _Proceedings of the IEEE conference on computer vision and pattern recognition_ , pages 770–778, 2016. 

- [16] Xinyang Huang, Chuang Zhu, and Wenkai Chen. Restnet: Boosting cross-domain few-shot segmentation with residual transformation network. _arXiv preprint arXiv:2308.13469_ , 2023. 

- [17] Stefan Jaeger, Alexandros Karargyris, Sema Candemir, Les Folio, Jenifer Siegelman, Fiona Callaghan, Zhiyun Xue, Kannappan Palaniappan, Rahul K Singh, Sameer Antani, et al. Automatic tuberculosis screening using chest radiographs. _IEEE transactions on medical imaging_ , 33(2):233–245, 2013. 

- [18] Diederik P Kingma and Jimmy Ba. Adam: A method for stochastic optimization. _arXiv preprint arXiv:1412.6980_ , 2014. 

- [19] Alexander Kirillov, Eric Mintun, Nikhila Ravi, Hanzi Mao, Chloe Rolland, Laura Gustafson, Tete Xiao, Spencer Whitehead, Alexander C Berg, Wan-Yen Lo, et al. Segment anything. In _Proceedings of the IEEE/CVF International Conference on Computer Vision_ , pages 4015–4026, 2023. 

- [20] Chunbo Lang, Gong Cheng, Binfei Tu, and Junwei Han. Learning what not to segment: A new perspective on few-shot segmentation. In _Proceedings of the IEEE/CVF conference on computer vision and pattern recognition_ , pages 8057–8067, 2022. 

- [21] Shuo Lei, Xuchao Zhang, Jianfeng He, Fanglan Chen, Bowen Du, and Chang-Tien Lu. Cross-domain few-shot semantic segmentation. In _European Conference on Computer Vision_ , pages 73–90. Springer, 2022. 

- [22] Chengyin Li, Prashant Khanduri, Yao Qiang, Rafi Ibn Sultan, Indrin Chetty, and Dongxiao Zhu. Auto-prompting sam for mobile friendly 3d medical image segmentation. _arXiv preprint arXiv:2308.14936_ , 2023. 

- [23] Gen Li, Varun Jampani, Laura Sevilla-Lara, Deqing Sun, Jonghyun Kim, and Joongkyu Kim. Adaptive prototype learning and allocation for few-shot segmentation. In _Proceedings of the IEEE/CVF conference on computer vision and pattern recognition_ , pages 8334–8343, 2021. 

- [24] Xiang Li, Tianhan Wei, Yau Pun Chen, Yu-Wing Tai, and Chi-Keung Tang. Fss-1000: A 1000-class dataset for few-shot segmentation. In _Proceedings of the IEEE/CVF conference on computer vision and pattern recognition_ , pages 2869–2878, 2020. 

- [25] Jonathan Long, Evan Shelhamer, and Trevor Darrell. Fully convolutional networks for semantic segmentation. In _Pro-_ 

_ceedings of the IEEE conference on computer vision and pattern recognition_ , pages 3431–3440, 2015. 

- [26] Zhihe Lu, Sen He, Xiatian Zhu, Li Zhang, Yi-Zhe Song, and Tao Xiang. Simpler is better: Few-shot semantic segmentation with classifier weight transformer. In _Proceedings of the IEEE/CVF International Conference on Computer Vision_ , pages 8741–8750, 2021. 

- [27] Maciej A Mazurowski, Haoyu Dong, Hanxue Gu, Jichen Yang, Nicholas Konz, and Yixin Zhang. Segment anything model for medical image analysis: an experimental study. _Medical Image Analysis_ , 89:102918, 2023. 

- [28] Juhong Min, Dahyun Kang, and Minsu Cho. Hypercorrelation squeeze for few-shot segmentation. In _Proceedings of the IEEE/CVF international conference on computer vision_ , pages 6941–6952, 2021. 

- [29] Maxime Oquab, Timoth´ee Darcet, Th´eo Moutakanni, Huy Vo, Marc Szafraniec, Vasil Khalidov, Pierre Fernandez, Daniel Haziza, Francisco Massa, Alaaeldin El-Nouby, et al. Dinov2: Learning robust visual features without supervision. _arXiv preprint arXiv:2304.07193_ , 2023. 

- [30] Bohao Peng, Zhuotao Tian, Xiaoyang Wu, Chengyao Wang, Shu Liu, Jingyong Su, and Jiaya Jia. Hierarchical dense correlation distillation for few-shot segmentation. In _Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition_ , pages 23641–23651, 2023. 

- [31] Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, et al. Learning transferable visual models from natural language supervision. In _International conference on machine learning_ , pages 8748–8763. PMLR, 2021. 

- [32] Olga Russakovsky, Jia Deng, Hao Su, Jonathan Krause, Sanjeev Satheesh, Sean Ma, Zhiheng Huang, Andrej Karpathy, Aditya Khosla, Michael Bernstein, et al. Imagenet large scale visual recognition challenge. _International journal of computer vision_ , 115:211–252, 2015. 

- [33] Amirreza Shaban, Shray Bansal, Zhen Liu, Irfan Essa, and Byron Boots. One-shot learning for semantic segmentation. _arXiv preprint arXiv:1709.03410_ , 2017. 

- [34] Mennatullah Siam, Boris N Oreshkin, and Martin Jagersand. Amp: Adaptive masked proxies for few-shot segmentation. In _Proceedings of the IEEE/CVF International Conference on Computer Vision_ , pages 5249–5258, 2019. 

- [35] Karen Simonyan and Andrew Zisserman. Very deep convolutional networks for large-scale image recognition. _arXiv preprint arXiv:1409.1556_ , 2014. 

- [36] Jake Snell, Kevin Swersky, and Richard Zemel. Prototypical networks for few-shot learning. _Advances in neural information processing systems_ , 30, 2017. 

- [37] Antonio Tavera, Fabio Cermelli, Carlo Masone, and Barbara Caputo. Pixel-by-pixel cross-domain alignment for fewshot semantic segmentation. In _Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision_ , pages 1626–1635, 2022. 

- [38] Zhuotao Tian, Hengshuang Zhao, Michelle Shu, Zhicheng Yang, Ruiyu Li, and Jiaya Jia. Prior guided feature enrichment network for few-shot segmentation. _IEEE transactions on_ 

_pattern analysis and machine intelligence_ , 44(2):1050–1065, 2020. 

- [39] Tassilo Wald, Saikat Roy, Gregor Koehler, Nico Disch, Maximilian Rouven Rokuss, Julius Holzschuh, David Zimmerer, and Klaus Maier-Hein. Sam. md: Zero-shot medical image segmentation capabilities of the segment anything model. In _Medical Imaging with Deep Learning, short paper track_ , 2023. 

- [40] An Wang, Mobarakol Islam, Mengya Xu, Yang Zhang, and Hongliang Ren. Sam meets robotic surgery: An empirical study on generalization, robustness and adaptation. In _International Conference on Medical Image Computing and Computer-Assisted Intervention_ , pages 234–244. Springer, 2023. 

- [41] Kaixin Wang, Jun Hao Liew, Yingtian Zou, Daquan Zhou, and Jiashi Feng. Panet: Few-shot image semantic segmentation with prototype alignment. In _proceedings of the IEEE/CVF international conference on computer vision_ , pages 9197–9206, 2019. 

- [42] Wenjian Wang, Lijuan Duan, Yuxi Wang, Qing En, Junsong Fan, and Zhaoxiang Zhang. Remember the difference: Crossdomain few-shot semantic segmentation via meta-memory transfer. In _Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition_ , pages 7065–7074, 2022. 

- [43] Yixin Wang, Zhe Xu, Jiang Tian, Jie Luo, Zhongchao Shi, Yang Zhang, Jianping Fan, and Zhiqiang He. Cross-domain few-shot learning for rare-disease skin lesion segmentation. In _ICASSP 2022-2022 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)_ , pages 1086– 1090. IEEE, 2022. 

- [44] Zhonghua Wu, Xiangxi Shi, Guosheng Lin, and Jianfei Cai. Learning meta-class memory for few-shot semantic segmentation. In _Proceedings of the IEEE/CVF International Conference on Computer Vision_ , pages 517–526, 2021. 

- [45] Enze Xie, Wenhai Wang, Zhiding Yu, Anima Anandkumar, Jose M Alvarez, and Ping Luo. Segformer: Simple and efficient design for semantic segmentation with transformers. _Advances in Neural Information Processing Systems_ , 34:12077– 12090, 2021. 

- [46] Boyu Yang, Chang Liu, Bohao Li, Jianbin Jiao, and Qixiang Ye. Prototype mixture models for few-shot semantic segmentation. In _Computer Vision–ECCV 2020: 16th European Conference, Glasgow, UK, August 23–28, 2020, Proceedings, Part VIII 16_ , pages 763–778. Springer, 2020. 

- [47] Lihe Yang, Wei Zhuo, Lei Qi, Yinghuan Shi, and Yang Gao. Mining latent classes for few-shot segmentation. In _Proceedings of the IEEE/CVF international conference on computer vision_ , pages 8721–8730, 2021. 

- [48] Wenxi Yue, Jing Zhang, Kun Hu, Yong Xia, Jiebo Luo, and Zhiyong Wang. Surgicalsam: Efficient class promptable surgical instrument segmentation. _arXiv preprint arXiv:2308.08746_ , 2023. 

- [49] Bingfeng Zhang, Jimin Xiao, and Terry Qin. Self-guided and cross-guided learning for few-shot segmentation. In _Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition_ , pages 8312–8321, 2021. 

- [50] Chi Zhang, Guosheng Lin, Fayao Liu, Jiushuang Guo, Qingyao Wu, and Rui Yao. Pyramid graph networks with connection attentions for region-based one-shot semantic segmentation. In _Proceedings of the IEEE/CVF International Conference on Computer Vision_ , pages 9587–9595, 2019. 

- [51] Chi Zhang, Guosheng Lin, Fayao Liu, Rui Yao, and Chunhua Shen. Canet: Class-agnostic segmentation networks with iterative refinement and attentive few-shot learning. In _Proceedings of the IEEE/CVF conference on computer vision and pattern recognition_ , pages 5217–5226, 2019. 

- [52] Gengwei Zhang, Guoliang Kang, Yi Yang, and Yunchao Wei. Few-shot segmentation via cycle-consistent transformer. _Advances in Neural Information Processing Systems_ , 34: 21984–21996, 2021. 

- [53] Renrui Zhang, Zhengkai Jiang, Ziyu Guo, Shilin Yan, Junting Pan, Hao Dong, Peng Gao, and Hongsheng Li. Personalize segment anything model with one shot. _arXiv preprint arXiv:2305.03048_ , 2023. 

- [54] Xiaolin Zhang, Yunchao Wei, Yi Yang, and Thomas S Huang. Sg-one: Similarity guidance network for one-shot semantic segmentation. _IEEE transactions on cybernetics_ , 50 (9):3855–3865, 2020. 

## **APSeg: Auto-Prompt Network for Cross-Domain Few-Shot Semantic Segmentation** 

## Supplementary Material 

## **1. Experiment** 

## **1.1. Datasets** 

**FSS-1000.** FSS-1000 [9] is a natural scenario dataset containing 1000 class categories with 10 samples per category. The evaluation procedure is conducted on 2400 randomly sampled support-query pairs. 

**Chest X-ray.** Chest X-ray [1, 7] is an X-ray image dataset of 566 images collected from 58 cases with manifestations of tuberculosis and 80 normal cases. 

**ISIC.** ISIC [4] is a skin lesion image dataset from the ISIC2018 challenge. Following the previous approach [8], the evaluation procedure is conducted on the training set, which includes 2596 images and the corresponding annotations. **Deepglobe.** Deepglobe [5] is a remote sensing image dataset that can be used for land cover segmentation. The dataset contains 7 categories: areas of urban, agriculture, rangeland, forest, water, barrel, and unknown. Following the previous method [8], we filter the unknown class in the training set and chunk the images to obtain 5666 images. We report the test results on the processed training set. 

## **1.2. Implementation Details** 

In the meta prompt generator (MPG) module, the spatial size of the feature enrichment module (FEM) is set to _{_ 60 _,_ 30 _,_ 15 _,_ 8 _}_ , maintaining consistency with PFENet [10]. The transformer decoder [3] block consists of a selfattention mechanism, a cross-attention mechanism, and a feed-forward network. Its configuration is in line with Protoformer [2]. 

## **1.3. Ablation Study** 

**Effect of the cycle consistent selection.** In the dual prototype anchor transformation (DPAT) module, pseudo query prototypes are extracted through cycle-consistent selection (CSS) to enhance the feature transformation process. To further validate the effectiveness of CCS, we explore an alternative method for extracting pseudo query prototypes. Analogous to ResNet [6], we first obtain the coarse prediction mask of the query image and then perform the MAP operation to obtain query prototypes. This method is referred to as prediction mask-based MAP (PM-MAP). We conduct an experiment to evaluate the model without CCS, with CCS, and with PM-MAP, respectively, to better analyze the contribution of our CCS. The results in Tab. 1 show that CCS achieves better performance, with an average improvement of 2.67% mIoU on four datasets on the 1-shot 

|Method|1-shot mIoU|
|---|---|
|w/o CCS<br>w/ CCS<br>w/ PM-MAP|58.63<br>61.30<br>59.05|



Table 1. Ablation studies of cycle-consistent selection (CSS) in dual prototype anchor transformation (DPAT) module. PM-MAP means the prediction mask-based MAP method. The results are averaged over four datasets under the 1-shot setting. 

setting. This indicates that CCS can extract reliable query prototypes to enhance the support prototypes, allowing features to be transformed into a more stable domain-agnostic space. Qualitative results on the effectiveness of CSS are provided in Fig. 1. 

## **1.4. Additional Analysis** 

PerSAM [11] is a few-shot segmentation method based on SAM. To enable automatic segmentation, PerSAM extracts point prompts based on cosine similarity measure and obtains box/mask prompts from the coarse predictions. The final prediction results are produced under the guidance of three types of visual prompts. The performance of PerSAM is far inferior to our proposed APSeg and PATNet [8] due to the inability to extract precise prompts. In contrast, our APSeg introduces MPG and DPAT, avoiding reliance on precise visual prompts and achieving competitive results in cross-domain scenarios. Qualitative results are provided in Fig. 2. It can be observed that our method outperforms PerSAM by a large margin, which validates the effectiveness of our automatic way of generating prompt embeddings. In addition, we provide more qualitative segmentation results of our proposed method on four datasets in Fig. 3. 

## **References** 

- [1] Sema Candemir, Stefan Jaeger, Kannappan Palaniappan, Jonathan P Musco, Rahul K Singh, Zhiyun Xue, Alexandros Karargyris, Sameer Antani, George Thoma, and Clement J McDonald. Lung segmentation in chest radiographs using anatomical atlases with nonrigid registration. _IEEE transactions on medical imaging_ , 33(2):577–590, 2013. 

- [2] Leilei Cao, Yibo Guo, Ye Yuan, and Qiangguo Jin. Prototype as query for few shot semantic segmentation. _arXiv preprint arXiv:2211.14764_ , 2022. 

- [3] Nicolas Carion, Francisco Massa, Gabriel Synnaeve, Nicolas Usunier, Alexander Kirillov, and Sergey Zagoruyko. End-toend object detection with transformers. In _European conference on computer vision_ , pages 213–229. Springer, 2020. 

- [4] Noel Codella, Veronica Rotemberg, Philipp Tschandl, 

M Emre Celebi, Stephen Dusza, David Gutman, Brian Helba, Aadi Kalloo, Konstantinos Liopyris, Michael Marchetti, et al. Skin lesion analysis toward melanoma detection 2018: A challenge hosted by the international skin imaging collaboration (isic). _arXiv preprint arXiv:1902.03368_ , 2019. 

- [5] Ilke Demir, Krzysztof Koperski, David Lindenbaum, Guan Pang, Jing Huang, Saikat Basu, Forest Hughes, Devis Tuia, and Ramesh Raskar. Deepglobe 2018: A challenge to parse the earth through satellite images. In _Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition Workshops_ , pages 172–181, 2018. 

- [6] Xinyang Huang, Chuang Zhu, and Wenkai Chen. Restnet: Boosting cross-domain few-shot segmentation with residual transformation network. _arXiv preprint arXiv:2308.13469_ , 2023. 

- [7] Stefan Jaeger, Alexandros Karargyris, Sema Candemir, Les Folio, Jenifer Siegelman, Fiona Callaghan, Zhiyun Xue, Kannappan Palaniappan, Rahul K Singh, Sameer Antani, et al. Automatic tuberculosis screening using chest radiographs. _IEEE transactions on medical imaging_ , 33(2):233– 245, 2013. 

- [8] Shuo Lei, Xuchao Zhang, Jianfeng He, Fanglan Chen, Bowen Du, and Chang-Tien Lu. Cross-domain few-shot semantic segmentation. In _European Conference on Computer Vision_ , pages 73–90. Springer, 2022. 

- [9] Xiang Li, Tianhan Wei, Yau Pun Chen, Yu-Wing Tai, and Chi-Keung Tang. Fss-1000: A 1000-class dataset for fewshot segmentation. In _Proceedings of the IEEE/CVF conference on computer vision and pattern recognition_ , pages 2869–2878, 2020. 

- [10] Zhuotao Tian, Hengshuang Zhao, Michelle Shu, Zhicheng Yang, Ruiyu Li, and Jiaya Jia. Prior guided feature enrichment network for few-shot segmentation. _IEEE transactions on pattern analysis and machine intelligence_ , 44(2):1050– 1065, 2020. 

- [11] Renrui Zhang, Zhengkai Jiang, Ziyu Guo, Shilin Yan, Junting Pan, Hao Dong, Peng Gao, and Hongsheng Li. Personalize segment anything model with one shot. _arXiv preprint arXiv:2305.03048_ , 2023. 


![](raw/papers/2_LLM_NLP/LLM_paper_list/autoprompt/cvpr_2024_APSeg_Auto-Prompt_Network_for_Cross-Domain_Few-Shot_Semantic_Segmentation_images/cvpr_2024_APSeg_Auto-Prompt_Network_for_Cross-Domain_Few-Shot_Semantic_Segmentation.pdf-0014-00.png)


**----- Start of picture text -----**<br>
support image ground truth w/o CCS w/ CCS support image ground truth w/o CCS w/ CCS<br>Chest X-ray<br>ISIC<br>FSS-1000<br>Deepglobe<br>**----- End of picture text -----**<br>


Figure 1. Visual comparison of segmentation results with and without cycle-consistent selection (CCS) in dual prototype anchor transformation (DPAT) module under the 1-shot setting. 


![](raw/papers/2_LLM_NLP/LLM_paper_list/autoprompt/cvpr_2024_APSeg_Auto-Prompt_Network_for_Cross-Domain_Few-Shot_Semantic_Segmentation_images/cvpr_2024_APSeg_Auto-Prompt_Network_for_Cross-Domain_Few-Shot_Semantic_Segmentation.pdf-0014-02.png)


**----- Start of picture text -----**<br>
support image ground truth PerSAM APSeg support image ground truth PerSAM APSeg<br>Chest X-ray<br>ISIC<br>FSS-1000<br>Deepglobe<br>**----- End of picture text -----**<br>


Figure 2. Visual Comparison Results between APSeg and PerSAM in four target datasets under the 1-shot setting. 


![](raw/papers/2_LLM_NLP/LLM_paper_list/autoprompt/cvpr_2024_APSeg_Auto-Prompt_Network_for_Cross-Domain_Few-Shot_Semantic_Segmentation_images/cvpr_2024_APSeg_Auto-Prompt_Network_for_Cross-Domain_Few-Shot_Semantic_Segmentation.pdf-0015-00.png)


**----- Start of picture text -----**<br>
support image ground truth prediction support image ground truth prediction<br>Chest X-ray<br>ISIC<br>FSS-1000<br>Deepglobe<br>**----- End of picture text -----**<br>


Figure 3. More qualitative segmentation results of our proposed APSeg in four target datasets under the 1-shot setting. 

