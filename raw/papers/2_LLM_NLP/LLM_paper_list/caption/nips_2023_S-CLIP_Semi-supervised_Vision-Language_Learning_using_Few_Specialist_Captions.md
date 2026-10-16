# **S-CLIP: Semi-supervised Vision-Language Learning using Few Specialist Captions** 

**Sangwoo Mo**[1] _[,]_[2] **Minkyu Kim**[1] _[,]_[3] **Kyungmin Lee**[1] **Jinwoo Shin**[1] 1KAIST 2University of Michigan 3KRAFTON 

## **Abstract** 

Vision-language models, such as contrastive language-image pre-training (CLIP), have demonstrated impressive results in natural image domains. However, these models often struggle when applied to specialized domains like remote sensing, and adapting to such domains is challenging due to the limited number of image-text pairs available for training. To address this, we propose S-CLIP, a semi-supervised learning method for training CLIP that utilizes additional unpaired images. S- CLIP employs two pseudo-labeling strategies specifically designed for contrastive learning and the language modality. The caption-level pseudo-label is given by a combination of captions of paired images, obtained by solving an optimal transport problem between unpaired and paired images. The keyword-level pseudo-label is given by a keyword in the caption of the nearest paired image, trained through partial label learning that assumes a candidate set of labels for supervision instead of the exact one. By combining these objectives, S-CLIP significantly enhances the training of CLIP using only a few image-text pairs, as demonstrated in various specialist domains, including remote sensing, fashion, scientific figures, and comics. For instance, S-CLIP improves CLIP by 10% for zero-shot classification and 4% for image-text retrieval on the remote sensing benchmark, matching the performance of supervised CLIP while using three times fewer image-text pairs.[1] 

## **1 Introduction** 

Pre-trained vision-language models have achieved remarkable success, providing a foundation for numerous downstream tasks [1]. However, these models often struggle when applied to specialized domains such as remote sensing or medical imaging [2]. This is because the models are trained on web-crawled data that may not fully capture the diversity and complexity of these domains [3, 4]. To address this issue, previous research has focused on constructing large-scale pre-training datasets for each domain [5, 6]. However, annotating caption for each specialized domain can be expensive and time-consuming, which limits their applicability across various domains. 

Several studies have attempted to reduce the number of image-text pairs used for vision-language pre-training. However, these approaches often rely on additional information, such as pre-trained object detectors [7–9], or class-annotated images [10], which may not be applicable to specialized domains. Other approaches leverage self-supervised learning to utilize unpaired data [11–14], but they do not fully exploit the information provided by image-text pairs. Research on leveraging a few image-text pairs to improve vision-language pre-training remains underexplored. 

In various areas of machine learning, particularly in image classification, semi-supervised learning [15] is a popular approach for training a model using limited annotations. This technique typically relies on pseudo-labeling [16], which predicts labels for unlabeled data by propagating information 

> *Work done at KAIST. Corresponds to: `swmo@umich.edu` 

> 1Code: `https://github.com/alinlab/s-clip` 

37th Conference on Neural Information Processing Systems (NeurIPS 2023). 


![](raw/papers/2_LLM_NLP/LLM_paper_list/caption/nips_2023_S-CLIP_Semi-supervised_Vision-Language_Learning_using_Few_Specialist_Captions_images/nips_2023_S-CLIP_Semi-supervised_Vision-Language_Learning_using_Few_Specialist_Captions.pdf-0002-00.png)


**----- Start of picture text -----**<br>
Nearest image Zero-shot accuracy (%) 66.9<br>in the batch on RSICD-CLS (+8.6)<br>55.6<br>58.3 (-2.7)<br>Their captions are different!<br>(naive pseudo-labeling fails) 45.3<br>“ two tennis courts are in two sides of a  and some  roadbuildings ”  “the a  swimming poolhuge rectangle and  eight tennis courts , contains a  two basketball courtsbaseball field ” , CLIP CLIP-FT Hard-PL S-CLIP (ours)<br>(a) Problem of naive pseudo-labeling (use the nearest caption) (b) Comparison of models<br>**----- End of picture text -----**<br>


Figure 1: **Motivation.** (a) Images and captions are from the remote sensing dataset, RSICD. Unlike class labels, captions vary widely across images, making the naive pseudo-labeling of assigning the nearest caption ineffective. (b) Zero-shot accuracy on the RSICD-CLS dataset. Bracket indicates the gap from CLIP fine-tuned on remote sensing datasets, called CLIP-FT. The original CLIP struggles in this specialized domain, and CLIP-FT gives a large gain. As discussed in (a), naive pseudo-labeling (Hard-PL) harms CLIP-FT, while our S-CLIP provides substantial improvement. 

from labeled data and refining the model using them. Combining this approach with self-supervised learning [17] has significantly improved classifiers using only a few class labels [18–22]. 

A natural question is whether semi-supervised learning, particularly pseudo-labeling, can improve vision-language pre-training. However, the techniques from image classification cannot be directly applied to vision-language models since captions are diverse and often uniquely associated with each image. Thus, the naive pseudo-labeling approach of assigning the caption of the nearest labeled image[2] (Hard-PL) [16] can mislead the models. Figure 1a demonstrates that even visually similar images often have different captions, although they may share keywords such as “tennis court.” As a result, Figure 1b shows that Hard-PL harms the performance of CLIP fine-tuning. This observation motivates the proper design of pseudo-labeling method that effectively utilizes captions. 

**Contribution.** We propose S-CLIP, a novel semi-supervised learning method for vision-language pretraining, particularly CLIP [23]. S-CLIP introduces two novel pseudo-labeling methods specifically designed for contrastive learning and the language modality, as illustrated in Figure 2: 

- _Caption-level pseudo-label._ We assume that the semantics of an unlabeled image can be expressed as a combination of those of labeled images. To achieve this, we define the caption-level pseudolabel as a probability distribution over the labeled images, which guides the relationship of an unlabeled image with its corresponding captions. Specifically, the pseudo-labels are obtained by solving an optimal transport [24, 25] problem between the unlabeled and labeled images. This approach prevents pseudo-labels from collapsing to a few captions and ensures robust training, particularly when dealing with distribution shifts between unlabeled and labeled images. 

- _Keyword-level pseudo-label._ We assume that an unlabeled image shares keywords with visually similar images, even if their full captions are not identical. Therefore, we define the keyword-level pseudo-label as one of the keywords in the nearest labeled image to an unlabeled image. This approach creates a candidate set of target keywords instead of a single exact one, and the training can be formulated as a partial label learning [26] problem. This loss helps the model understand the local components of unlabeled images, leveraging the structure of language. 

We note that both pseudo-labels are complementary. Caption-level pseudo-label helps the model understand the global structure of language, which is more effective for image-text retrieval. Keywordlevel pseudo-label helps the model understand the local phrases, which is more effective for zero-shot classification. Combining them achieves the best of both worlds (Section 5.5). 

We demonstrate the effectiveness of our method in various specialist domains with limited available image-text pairs, including remote sensing [27], fashion [28], scientific figures [29], and comics [30] domains. In the remote sensing domain, S-CLIP outperforms CLIP fine-tuning and semi-supervised learning competitors in five zero-shot classification and six image-text retrieval tasks. For instance, S-CLIP improves zero-shot accuracy on WHU-RS19 [31] by 10.4% and image-to-text retrieval R@5 

2We use the terms “paired” and “labeled” interchangeably, as well as “unpaired” and “unlabeled,” depending on the context of vision-language pre-training or semi-supervised learning. 

2 


![](raw/papers/2_LLM_NLP/LLM_paper_list/caption/nips_2023_S-CLIP_Semi-supervised_Vision-Language_Learning_using_Few_Specialist_Captions_images/nips_2023_S-CLIP_Semi-supervised_Vision-Language_Learning_using_Few_Specialist_Captions.pdf-0003-00.png)


**----- Start of picture text -----**<br>
Pseudo-label = (from visually similar images in the batch) Combination of the captions Pseudo-label = (from the caption of the Candidate set of keywords  nearest image)<br>0.4  × 0.3  ×<br>huge rectangle<br>= + ... ∈ baseball field<br>swimming pool<br>“… tennis courts …”  “… buildings …”  tennis courts<br>(a) Caption-level pseudo-label (b) Keyword-level pseudo-label<br>**----- End of picture text -----**<br>


Figure 2: **Method overview.** Conceptual illustration of our proposed S-CLIP, utilizing (a) captionlevel and (b) keyword-level pseudo-labels. In (a), the pseudo-label is given by a probability distribution over the captions, obtained by solving an optimal transport problem between unlabeled and labeled images. In (b), the pseudo-label is given by the candidate set of keywords from the caption of the nearest labeled image, which is learned by the partial label learning algorithm. 

on UCM [32] by 4.4% compared to CLIP fine-tuning. S-CLIP remains robust even when unlabeled images are drawn from a different dataset. In the comics domain, the Simpsons [30] dataset only includes 800 image-text pairs. Therefore, we incorporate the Simpsons Characters [33] dataset, which consists of 30,000 images without captions. Here, S-CLIP improves the text _→_ image retrieval R@1 of CLIP from 11.8% to 15.8%, providing a relative gain of 33%. 

## **2 Related work** 

**Vision-language pre-training (VLP).** Pre-training vision-language models has achieved remarkable success, providing a foundation for many downstream tasks [1]. Numerous approaches have been proposed, including reconstructing masked inputs [34–38], learning a joint embedding of vision and language modalities [23, 39–42], generating language descriptions from image inputs [43–45], and connecting pre-trained unimodal image and language models [46–48]. CLIP [23] is a representative model that learns a joint embedding between vision and language modalities. CLIP has demonstrated effectiveness in various applications, including zero-shot classification [23], image-text retrieval [49], open-vocabulary object segmentation [50, 51], and out-of-distribution detection [52]. 

**VLP for specialist domains.** Despite their success in natural image domains, VLP models often struggle when applied to specialist domains like remote sensing [2]. This is because they are trained on web-crawled data that may not capture the diversity and complexity of these domains [3, 4]. Data in these fields is often scarce, has limited accessibility, and requires expert annotations, making it difficult to obtain the image-text pairs needed for VLP training. While attempts have been made to train VLP models for specialized domains, including medical [10, 53–59], fashion [60–63], and remote sensing [64], previous research has mainly focused on constructing large-scale pre-training datasets for each domain [5, 6], which can be expensive and time-consuming. 

**VLP with limited pairs.** Several studies trained VLP models with a limited number of image-text pairs. Hsu et al. [65] used adversarial domain adaptation to regularize unpaired data embeddings similar to paired data. However, this approach can cause problems when the unpaired data is drawn from a different distribution. MedCLIP [10] utilized images with class labels to reduce the need for captions. Nevertheless, it still requires class labels, limiting the use of large unlabeled data. Other studies explored unsupervised or semi-supervised VLP training in image domains [7–9]. However, they rely on a pre-trained object detector to align detected objects with keywords in captions, making them unsuitable for specialist domains. Other works leverage self-supervised learning on each modality [11–14], but do not fully exploit the information provided by image-text pairs. In contrast, our method leverages image-text pairs and is generally applicable without extra information. 

**Semi-supervised learning.** Semi-supervised learning [15] aims to train a model using a small labeled dataset and a large unlabeled dataset. Techniques for semi-supervised learning can be categorized into two types: pseudo-labeling (or label propagation) [16, 66], which utilizes information from the small labeled dataset, and self-supervised learning [17], which learns representations from the unlabeled dataset. Combining these techniques has achieved remarkable success in semi-supervised learning, particularly for training classifiers with few class labels [18–22]. Prior works on robust semi-supervised learning focus on scenarios where the distributions of unlabeled and labeled data 

3 

shift [67–69]. This issue naturally arises in vision-language pre-training due to the uniqueness of captions for each image. Our method addresses this issue using soft labels and performs robustly even when the unlabeled images are from a different dataset than the labeled images. 

## **3 Background** 

Before we demonstrate our semi-supervised vision-language pre-training method, we provide a brief overview on contrastive language-image pre-training and semi-supervised learning. 

## **3.1 Contrastive language-image pre-training** 

Contrastive language-image pre-training (CLIP) [23] is a popular vision-language model that learns a joint embedding space connecting images _x_ and texts _y_ .[3] CLIP is trained using a batch of paired images and texts _{xi, yi}[N] i_ =1[through contrastive learning, where those from the same pair are treated] as positive and different pairs as negative. The training objective of CLIP involves two classification tasks: predicting a text given an image _p_ ( _y|x_ ) and predicting an image given a text _p_ ( _x|y_ ). Labels are assigned to the _N_ samples in the batch, corresponding to their pairs as the target. The embeddings are normalized to have a unit norm, and CLIP predicts the label using a softmax function with a temperature parameter _τ >_ 0, applied to the cosine similarity of the embeddings. Specifically, we denote the softmax classifier _στ_ as a function of input and target embeddings. This allows us to represent _p_ ( _y|x_ ) = _στ_ ( _x, {yi}[N] i_ =1[)] _[ ∈]_[R] _[N]_[and] _[ p]_[(] _[x][|][y]_[) =] _[ σ][τ]_[(] _[y,][ {][x][i][}][N] i_ =1[)] _[ ∈]_[R] _[N]_[, which] denote the class probabilities of CLIP. These probabilities are calculated as follows: _p_ ( _y_ = _yi|x_ ) = exp( _x · yi/τ_ ) _/_ ([�] _[N] j_ =1[exp(] _[x][ ·][ y][j][/τ]_[))][.][The CLIP model minimizes the following objective:] 


![](raw/papers/2_LLM_NLP/LLM_paper_list/caption/nips_2023_S-CLIP_Semi-supervised_Vision-Language_Learning_using_Few_Specialist_Captions_images/nips_2023_S-CLIP_Semi-supervised_Vision-Language_Learning_using_Few_Specialist_Captions.pdf-0004-05.png)


where _H_ denotes the cross-entropy loss and **e** _i ∈_ R _[N]_ is a one-hot vector with the _i_ -th element being one. The CLIP model can be used for zero-shot classification or image-text retrieval by computing cosine similarity in the embedding space. For example, in zero-shot classification, the text embedding of the class prompt that is closest to an image embedding can be identified. 

## **3.2 Semi-supervised learning** 

Semi-supervised learning (Semi-SL) [15] aims to train a model using a small number of labeled data _{xi, yi}[N] i_ =1[and a large number of unlabeled data] _[ {][u][i][}][M] i_ =1[, i.e., when] _[ M][≫][N]_[.][Various Semi-] SL methods have been proposed to enhance neural network performance, particularly for image classification [19–22].[4] These methods typically rely on pseudo-labels [16], obtained by predicting classes for unlabeled data using a model. The model is then refined by minimizing the cross-entropy loss _M_ 1 � _Mi_ =1 _[H]_[(] _[p]_[(] _[y][|][u][i]_[)] _[, q][i]_[)][ with respect to the prediction] _[ p]_[(] _[y][|][u][i]_[)] _[ ∈]_[R] _[C]_[and pseudo-label] _[ q][i][∈]_[R] _[C]_ for unlabeled data _ui_ . Pseudo-labels can take the form of a hard label, represented by a one-hot vector identifying the class with the highest probability [16], or a soft label that calibrates this hard label [21]. Pseudo-labels minimize prediction entropy by sharpening specific classes [70]. 

**Semi-SL for CLIP.** Assigning pseudo-labels in CLIP poses challenges due to the unique captions associated with each image. The assumption that unlabeled data belongs to the same class as labeled data does not hold here. Therefore, we propose a novel Semi-SL method for CLIP. 

## **4 S-CLIP: Semi-supervised Vision-Language Pre-training** 

We extend the training of CLIP to incorporate unpaired images _{ui}[M] i_ =1[alongside the image-text] pairs _{xi, yi}[N] i_ =1[.][To achieve this, we introduce S-CLIP, which integrates two novel pseudo-labeling] methods considering contrastive learning and the language modality. The conceptual illustrations of the proposed pseudo-labels and training objectives are shown in Figure 2 and Figure 3, respectively. 

> 3We use the notations _x_ and _y_ to represent image and text embeddings computed by the CLIP encoders. However, for simplicity, we often omit the term "embeddings" when referring to them. 

> 4In image classification, the targets _y_ are represented by one-hot vectors of size _C_ for _C_ classes. 

4 


![](raw/papers/2_LLM_NLP/LLM_paper_list/caption/nips_2023_S-CLIP_Semi-supervised_Vision-Language_Learning_using_Few_Specialist_Captions_images/nips_2023_S-CLIP_Semi-supervised_Vision-Language_Learning_using_Few_Specialist_Captions.pdf-0005-00.png)


**----- Start of picture text -----**<br>
Captions Captions Keywords<br>𝑦! 𝑦" ⋯ 𝑦# 𝑦! 𝑦" ⋯ 𝑦# 𝑘! 𝑘" ⋯ 𝑘#<br>𝑥! 𝑢! 𝑢!<br>𝑥" 𝑢" 𝑢"<br>𝑥# 𝑢$ 𝑢%<br>One-hot labels Soft labels Sparse soft labels<br>(a) CLIP loss  L CLIP (b) Caption loss  L caption (c) Keyword loss  L keyword<br>(■ (■ (■<br>⋯ ⋯ ⋯<br>Paired images from OT plan<br>from true pairs) Unpaired images )Γ Unpaired images from keyword set K<br>$)<br>**----- End of picture text -----**<br>


Figure 3: **Training objective.** Conceptual illustrations of the training objectives of S-CLIP: (a) is the original CLIP loss in Eq. (1) using paired images and texts; (b) and (c) are our proposed caption-level and keyword-level pseudo-label losses in Eq. (3) and Eq. (4), respectively, applied to unpaired images. Caption-level pseudo-labels are soft labels obtained from the optimal transport (OT) plan Γ, while keyword-level pseudo-labels are sparse soft labels obtained from the candidate keyword set K _u_ . 

## **4.1 Caption-level pseudo-label** 

Although the semantics of an unlabeled image _u_ may not precisely match any captions _yi_ , visually similar images often share similar semantics. Therefore, the caption for unlabeled image _u_ can be expressed as a combination of the captions _{yi}_ . To achieve this, we use caption-level pseudo-labels, which represent a probability distribution over the captions. These pseudo-labels serve as the target for the _N_ -way classification problem in contrastive learning, with a batch size of _N_ . Figure 2a demonstrates how caption-level pseudo-label for an unlabeled image is obtained. 

The pseudo-labels are derived from the relationship between unlabeled and labeled images, formulated as an optimal transport (OT) [24] problem. We define the cost function C _∈_ R _[M][×][N]_ , where _M_ and _N_ represent the number of unlabeled and labeled images, respectively. The cost is given by the negative cosine similarity between the normalized embeddings of the unlabeled image _ui_ and the labeled image _xj_ , i.e., C _ij_ = 1 _− ui · xj_ . Then, we solve following entropic regularized OT [25] problem: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/caption/nips_2023_S-CLIP_Semi-supervised_Vision-Language_Learning_using_Few_Specialist_Captions_images/nips_2023_S-CLIP_Semi-supervised_Vision-Language_Learning_using_Few_Specialist_Captions.pdf-0005-05.png)


Here, Π( **p** _,_ **q** ) represents a set of transportation plans that satisfy the flow constraint. The sums of flows from sources and to sinks match the vectors **p** _∈_ R _[M]_ +[and] **[ q]** _[ ∈]_[R] _[N]_ +[, respectively, with both] **[ p]**[ and] **q** adding up to one. We set them as uniform probabilities, and they worked in our experiments, even when unlabeled and labeled data have distribution shifts. **1** _N_ denotes the all-one vector of dimension _N_ . The entropic regularizer _H_ (Γ) = _−_[�] _i,j_[Γ] _[ij]_[ log Γ] _[ij]_[with scale] _[ λ]_[ smooths the transportation plan,] enabling an efficient solution using the Sinkhorn-Knopp [25] algorithm. 

Once we obtain the transportation plan Γ, we define the pseudo-label for an unlabeled image _ui_ by normalizing its plan to sum up to one. Specifically, the _j_ -th element of the pseudo-label _qi ∈_ R _[N]_ is computed as Γ _ij/_[�] _j_[Γ] _[ij]_[.][This pseudo-label is then utilized to guide the probability distribution over] _N_ captions, where the prediction _p_ ( _y|ui_ ) _∈_ R _[N]_ is determined by a softmax classifier _στ_ ( _ui, {yi}[N] i_ =1[)][.] The loss function for the caption-level pseudo-label aims to minimize the following: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/caption/nips_2023_S-CLIP_Semi-supervised_Vision-Language_Learning_using_Few_Specialist_Captions_images/nips_2023_S-CLIP_Semi-supervised_Vision-Language_Learning_using_Few_Specialist_Captions.pdf-0005-08.png)


The prediction _p_ ( _y|ui_ ) relates the unlabeled image _ui_ to the text embeddings _y_ , while the pseudolabel _qi_ relates it to the image embeddings _x_ . Consequently, the proposed loss function aligns the relationships between image and text embeddings, as suggested in supervised CLIP [71–73]. In our experiments, we compare the usage of text and image embeddings for pseudo-label computation, and using image embeddings outperformed using text embeddings due to their robustness in novel specialist domains. We also explore computing pseudo-embeddings [9], but it yields inferior results as synthesizing pseudo-embeddings is more challenging than inferring pseudo-labels. 

5 

**Effect of OT.** Optimal transport balances the flow between unlabeled and labeled images, ensuring robust pseudo-label assignments without concentration on a few labels. In vision-language tasks, diverse and imbalanced captions often cause image embeddings to focus on a few nearest captions, resulting in the collapse of embeddings. This challenge is magnified when employing semi-supervised learning with distribution shifts [67]. We empirically validate the detrimental effect of naive pseudolabeling approaches, especially in the presence of distribution shifts. 

Compared to soft-nearest neighbor pseudo-labels (Soft-PL) [21], our method achieves balance through Sinkhorn iterations. Specifically, it reverts to Soft-PL when the iteration count is zero. In our experiments, we use 10 iterations, and increasing the iteration count has only a marginal impact on performance. The computational cost is negligible as the process occurs in a low-dimensional embedding space. We set the scale of the entropic regularizer to match the softmax temperature, _λ_ = _τ_ . This adjustment scales the cost function by 1 _/τ_ , aligning it with the cosine similarity scale learned by CLIP. More information on the OT and Soft-PL relationship is in Appendix A. 

**OT for vision-language models.** Several works have utilized OT for vision-language models, but for problems other than semi-supervised learning. We discuss these works in Appendix B. 

## **4.2 Keyword-level pseudo-label** 

We propose a keyword-level pseudo-labeling approach to overcome the limitations of caption-level pseudo-labels in capturing the meanings of words in captions. By leveraging the compositionality of language, we assume that visually similar images may share keywords, even if the entire captions differ. This enables us to guide unlabeled images using keywords from their nearest neighbors. It is important to note that an image may have multiple keywords, and visually similar images share only a subset. This aligns with the partial label learning (PLL) [26] problem, where the target label is unknown, and we have access to a candidate set of labels. For instance, in Figure 1a, the image shares the keyword “tennis court” with its nearest neighbor but not “baseball field.” Consequently, the nearest neighbor provides a candidate set of keywords, as illustrated in Figure 2b. 

To identify candidate keywords for an unlabeled image _u_ , we assume a pre-defined set of keywords with corresponding embeddings K = _{ki}[K] i_ =1[.][These keywords can be obtained from class names, if] available, or extracted by using algorithms like YAKE [74] applied to captions. Given an unlabeled image _u_ , we find the nearest labeled image _x_ using the OT assignments from the previous section. Let _{c_ 1 _, . . . , cl}_ be the indices of keywords present in labeled image _x_ , denoted as K _u_ = _{kc_ 1 _, . . . , kcl }_ . This forms the candidate set of target keywords for _u_ . While we simply use the keywords explicitly from the caption, some additional steps can be taken to infer synonyms or related concepts. 

In the context of partial label learning, we assume that the ground-truth target label for the unlabeled image _u_ belongs to the candidate set K _u_ . PLL aims to minimize the loss by selecting the minimum value among candidate labels, min _k∈_ K _u H_ ( _p_ ( _k|u_ ) _, k_ ), rather than specifying a single target label. However, relying on a single minimum can be risky. Previous works have addressed this by using soft labels for candidates [75, 76]. This involves minimizing _H_ ( _p_ ( _k|u_ ) _, q_ ), where _q ∈_ R _[K]_ is a soft label with positive values for keywords in the candidate set and zero for others. 

Similarly, we define the pseudo-label _qi_ for the unlabeled image _ui_ based on the similarity between the embeddings of _ui_ and the keywords _kj_ in the candidate set K _u_ . Specifically, the _j_ -th element of the pseudo-label _qi_ is calculated using a softmax function _στ_ ( _ui,_ K _u_ ) for _k ∈_ K _u_ and is set to zero otherwise. This pseudo-label is then utilized to guide the probability distribution over _K_ keywords, where the prediction _p_ ( _k|u_ ) _∈_ R _[K]_ is determined by a softmax classifier _στ_ ( _ui,_ K). The loss function for the keyword-level pseudo-label aims to minimize the following: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/caption/nips_2023_S-CLIP_Semi-supervised_Vision-Language_Learning_using_Few_Specialist_Captions_images/nips_2023_S-CLIP_Semi-supervised_Vision-Language_Learning_using_Few_Specialist_Captions.pdf-0006-08.png)


Both the prediction _p_ ( _k|ui_ ) and the pseudo-label _qi_ are related to the unlabeled image and keywords. However, the restricted candidate set effectively guides the learning process through PLL. 

**Training objective.** We apply the CLIP loss to paired images and texts and the proposed pseudo-label losses to unpaired images. The pseudo-label losses are halved to match the scale of the image part in the CLIP loss. In summary, our final training objective becomes _L_ CLIP +[1] _/_ 2 ( _L_ caption + _L_ keyword). 

6 

Table 1: Zero-shot classification results on remote sensing datasets. We compare the original CLIP, supervised CLIP fine-tuned on labeled data (L), and semi-supervised methods that utilize unlabeled data sampled from the same (L=U) or different (L=U) distribution as the labeled data. Parentheses indicate the performance gap from the supervised CLIP, where values highlighted in green indicate gaps larger than one. Bolds denote the best results among the semi-supervised methods within the same setups. S-CLIP consistently improves zero-shot accuracy, even when the unlabeled data has a distribution shift. In contrast, naive pseudo-labeling often harms the accuracy. 

|Method|Data|RSICD-CLS|UCM-CLS|WHU-RS19|RSSCN7|AID|
|---|---|---|---|---|---|---|
|CLIP (original)|-|45.3|50.5|65.5|58.9|47.8|
|CLIP (fne-tune)|L|58.3_±_0.3|63.5_±_3.4|76.5_±_3.2|61.9_±_1.2|63.1_±_1.3|
|Hard-PL [16]||56.6_±_3.5( -1.7)|61.6_±_2.2( -1.9)|78.1_±_2.5(+1.6)|63.9_±_2.1(+2.0)|63.2_±_2.6(+0.1)|
|Soft-PL [21]|L=U|62.5_±_0.8(+4.2)|65.7_±_2.7(+2.2)|83.7_±_2.7(+7.2)|65.7_±_0.6(+3.8)|68.0_±_0.7(+4.9)|
|S-CLIP (ours)||**66.9**_±_1.7(+8.6)|**66.7**_±_1.6(+3.2)|**86.9**_±_2.0(+10.4)|**66.2**_±_1.1(+4.3)|**73.0**_±_0.3(+9.9)|
|Hard-PL [16]||55.8_±_1.2( -2.5)|61.0_±_6.2( -2.5)|76.3_±_1.1( -0.2)|62.5_±_2.7(+0.6)|62.3_±_1.4( -0.8)|
|Soft-PL [21]|L=U|56.1_±_3.5( -2.2)|62.4_±_1.8( -1.1)|79.5_±_3.6(+3.0)|63.3_±_2.6(+1.4)|62.4_±_2.1( -0.7)|
|S-CLIP (ours)||**65.3**_±_2.1(+7.0)|**66.0**_±_0.4(+2.5)|**86.3**_±_1.1(+9.8)|**68.3**_±_2.2(+6.4)|**70.8**_±_2.5(+7.7)|



## **5 Experiments** 

We demonstrate the effectiveness of our method in various specialist domains with limited available image-text pairs, including remote sensing, fashion, scientific figures, and comics domains. 

**Setup.** We use a pre-trained CLIP model on natural images [23], called CLIP (original). However, this model often struggles in specialist domains. Therefore, we fine-tune the model with domain-specific image-text pairs, called CLIP (fine-tune). We compare this supervised model with semi-supervised methods using unpaired images. We use a batch size of 64 per GPU, with a total of 4 GPUs. To ensure fair GPU memory usage in semi-supervised learning, we employ 32 image-caption pairs and 32 unpaired images for each mini-batch. Models are evaluated on zero-shot classification and image-text retrieval tasks, measuring Top-1 classification accuracy (%) and recall at K (R@K). We report the average and standard deviation across three random seeds. We follow the training recipe of OpenClip [77] if not specified. Additional experimental details are in Appendix C. 

## **5.1 Baselines** 

Our proposed S-CLIP includes two novel pseudo-labeling methods. Thus, we mainly compare S- CLIP with other pseudo-labeling approaches, which are extensions of semi-supervised classification techniques. Our baselines, Hard-PL and Soft-PL, extend the hard (top-1) and soft nearest neighbor approaches for pseudo-label assignment. We then utilize these methods as caption-level pseudo-labels, similar to Eq. (3). The specific forms of each method are as follows. 

**Hard-PL.** Given an unlabeled image _ui_ , Hard-PL identifies the top-1 nearest labeled image[5] _xi∗_ based on their cosine similarities, calculated as _i[∗]_ = arg max _j_ ( _ui · xj_ ). Subsequently, the pseudolabel of _ui_ is determined by the corresponding caption, denoted as _qi_ = _yi∗_ . This pseudo-label serves as the target for predicting _p_ ( _y|ui_ ), and the loss is computed as[�] _i[H]_[(] _[p]_[(] _[y][|][u][i]_[)] _[, q][i]_[)][.] 

**Soft-PL.** Soft-PL calibrates the pseudo-label of Hard-PL by considering all the relations between labeled and unlabeled images, known as soft nearest neighbor [21], instead of selecting only the top-1 nearest image. For an unlabeled image _ui_ and a set of labeled images _{xi}_ , these relations are determined by applying softmax to their cosine similarities. The pseudo-label is defined as _qi_ = _στ_ ( _ui, {xi}[N] i_ =1[)] _[ ∈]_[R] _[N]_[, where the pseudo-label has a higher value close to one for visually similar] images. We use the same softmax temperature _τ_ as CLIP, which is used to compute the similarities between the labeled image and text embeddings. The loss is computed as[�] _i[H]_[(] _[p]_[(] _[y][|][u][i]_[)] _[, q][i]_[)][.] 

## **5.2 Remote sensing datasets** 

We train vision-language models using the union of RSICD [27], UCM [32], and Sydney [78], named RS-ALL, following the setup of [64]. For semi-supervised learning, we subsample 10% of image-text 

5We also tested the version directly finding the nearest text for both Hard-PL and Soft-PL. However, the similarities in visual embeddings give more robust results, as discussed in the ablation study. 

7 

Table 2: Image-text retrieval results on remote sensing datasets, following the same setup of Table 1. Bolds denote the best results among the semi-supervised methods within the same setups. S-CLIP performs best in most cases, except the supervised baseline works best for text _→_ image retrieval on the Sydney dataset. Naive pseudo-labeling also helps, but the gain is often unstable. 

|Method<br>Data|Image_→_text R@5<br>RSICD<br>UCM<br>Sydney|Text_→_image R@5<br>RSICD<br>UCM<br>Sydney|
|---|---|---|
|CLIP (original)<br>-|9.4<br>34.3<br>36.2|10.1<br>24.8<br>51.7|
|CLIP (fne-tune)<br>L|15.4_±_1.7<br>41.3_±_1.8<br>47.1_±_6.5|15.1_±_1.0<br>40.9_±_1.6<br>56.1_±_2.4|
|Hard-PL [16]<br>L=U<br>Soft-PL [21]<br>S-CLIP (ours)|16.1_±_0.2<br>40.8_±_2.9<br>43.1_±_3.0<br>17.0_±_0.9<br>43.2_±_3.9<br>42.0_±_4.3<br>**18.4**_±_0.6<br>**45.7**_±_1.4<br>**50.0**_±_3.0|15.7_±_0.7<br>40.5_±_3.0<br>47.7_±_5.3<br>16.5_±_0.1<br>42.9_±_3.3<br>50.2_±_4.9<br>**16.8**_±_1.2<br>**43.5**_±_1.5<br>**55.1**_±_2.0|
|Hard-PL [16]<br>L=U<br>Soft-PL [21]<br>S-CLIP (ours)|15.8_±_1.0<br>42.1_±_5.7<br>47.1_±_4.0<br>16.5_±_1.1<br>40.2_±_3.2<br>46.6_±_4.6<br>**17.1**_±_0.8<br>**43.5**_±_3.5<br>**48.9**_±_2.6|15.3_±_0.4<br>42.1_±_3.1<br>50.0_±_1.7<br>15.4_±_0.2<br>40.8_±_0.5<br>**54.0**_±_4.0<br>**15.8**_±_0.7<br>**42.5**_±_1.5<br>52.3_±_1.0|



pairs as labeled data (L), while the remaining 90% of images (L=U) or unlabeled images from the RESISC45 [79] dataset (L=U) served as unlabeled data. In Appendix D, we demonstrate that our method consistently gives improvements over CLIP across various setups. These include different neural architectures, varied ratios of image-text pairs, and even scenarios where we use all image-text pairs from RS-ALL in conjunction with unlabeled images from RESISC45. 

For zero-shot classification, we use the validation sets of the classification versions of the RSICD and UCM datasets, denoted as RSCID-CLS and UCM-CLS, respectively. To evaluate the generalization abilities, we test models on unseen datasets such as WHU-RS19 [31], RSSCN7 [80], and AID [81]. For image-text retrieval, we use the validation sets of the RSICD, UCM, and Sydney datasets. 

**Zero-shot classification.** Table 1 presents the zero-shot classification results. S-CLIP consistently and significantly outperforms all supervised CLIP fine-tuning and semi-supervised methods. One can make several observations. Firstly, fine-tuning CLIP improves performance over the original CLIP, highlighting the importance of adapting models to the target specialist domain. Secondly, our proposed pseudo-labeling techniques are crucial, as naive pseudo-labeling often harms accuracy. Thirdly, semi-supervised learning methods improve performance on unseen datasets, such as RSSCN7, demonstrating the usefulness of observing more unlabeled images in generalization. Lastly, our method is robust to the distribution shifts (L=U) setup, which leverages an external source of unlabeled images, a common scenario in practice. In contrast, Soft-PL improves performance in the same distribution (L=U) setup but fails on the distribution shifts scenario, confirming the effectiveness of optimal transport in balancing the assignments of pseudo-labels. 

**Image-text retrieval.** Table 2 presents the image-text retrieval results. S-CLIP consistently improves both image _→_ text and text _→_ image retrieval, except in one case where the supervised baseline performs the best. This confirms that S-CLIP also benefits fine language understanding for retrieval. The trend is consistent across evaluation metrics, as seen in the R@1 results in the appendix. 

## **5.3 Fashion datasets** 

We train vision-language models using the union of Fashion200k [28], FashionGen [82], and Polyvore Outfits [83] datasets. We subsample 10% of the image-text pairs as labeled data and the remaining 90% of images as unlabeled data. We evaluate the zero-shot accuracy on the validation sets of all three datasets, considering both super-class and sub-class accuracies for Fashion200k and FashionGen. The class names in Polyvore match the same level of super-class as those in the other datasets. 

Table 3 presents the zero-shot classification results. S-CLIP consistently outperforms all supervised and semi-supervised methods. The improvement is more significant for super-class classification but less pronounced for sub-class classification and image-text retrieval. This is because the current pseudo-labeling approach assumes that the semantics of unlabeled image can be represented by the captions in a batch. However, this assumption may not hold true for fine-grained semantics, as certain fine-grained captions may be missing from the batch. Therefore, our method could be further enhanced by increasing the batch size or incorporating a queue of caption embeddings [84, 85]. 

8 

Table 3: Zero-shot classification results on fashion datasets. Parentheses indicate the performance gap from the supervised CLIP, where values highlighted in green indicate gaps larger than one. Bolds denote the best results. S-CLIP consistently outperforms all supervised and semi-supervised methods. 

|Method|Fashion200k<br>Super-class<br>Sub-class|FashionGen<br>Super-class<br>Sub-class|Polyvore<br>Class|
|---|---|---|---|
|CLIP (original)|73.4<br>29.3|35.9<br>22.1|58.4|
|CLIP (fne-tune)|79.4_±_1.5<br>38.4_±_1.5|41.5_±_1.4<br>33.1_±_1.9|64.2_±_3.1|
|Hard-PL [16]<br>Soft-PL [21]<br>S-CLIP (ours)|74.6_±_2.7( -4.8)<br>35.0_±_2.6( -3.4)<br>80.0_±_1.1(+0.6)<br>37.1_±_0.7( -1.3)<br>**82.0**_±_2.8(+2.6)<br>**39.5**_±_1.1(+1.1)|36.6_±_3.9( -4.9)<br>28.8_±_0.8( -4.3)<br>45.5_±_3.2(+4.0)<br>33.2_±_0.3(+0.1)<br>**55.3**_±_2.6(+13.8)<br>**38.7**_±_1.9(+5.6)|65.5_±_0.7(+1.3)<br>73.6_±_7.9(+9.4)<br>**74.6**_±_1.8(+10.4)|



Table 4: Image-text retrieval results on SciCap (scientific figures) and Simpsons (comics) datasets. We train the model on each dataset and evaluate on their validation sets. Bolds denote the best results. S-CLIP consistently outperforms all supervised and semi-supervised methods. 

|Method|SciCap<br>Image_→_text<br>Text_→_image<br>R@1<br>R@5<br>R@1<br>R@5|Simpsons<br>Image_→_text<br>Text_→_image<br>R@1<br>R@5<br>R@1<br>R@5|
|---|---|---|
|CLIP (original)|9.2<br>14.7<br>9.0<br>14.5|15.8<br>40.8<br>10.5<br>27.6|
|CLIP (fne-tune)|10.5_±_0.2<br>17.9_±_0.2<br>10.2_±_0.1<br>17.2_±_0.2|16.7_±_2.0<br>41.4_±_1.1<br>11.8_±_2.3<br>41.7_±_1.5|
|Hard-PL [16]<br>Soft-PL [21]<br>S-CLIP (ours)|9.7_±_0.3<br>16.8_±_0.6<br>9.2_±_0.6<br>16.2_±_0.7<br>9.1_±_0.5<br>15.5_±_1.0<br>9.7_±_0.3<br>16.8_±_0.5<br>**10.9**_±_0.3<br>**18.8**_±_0.3<br>**11.1**_±_0.3<br>**18.8**_±_0.4|15.4_±_1.5<br>41.7_±_3.3<br>12.7_±_1.5<br>40.8_±_2.3<br>15.4_±_2.0<br>39.9_±_2.7<br>12.9_±_1.9<br>36.4_±_2.7<br>**18.4**_±_2.3<br>**43.9**_±_1.5<br>**15.8**_±_1.3<br>**42.1**_±_0.1|



## **5.4 More captioning datasets** 

We conduct experiments on two more specialist domains, namely science figures and comics. For this purpose, we use the SciCap [29] and Simpsons [30] datasets. We train the models on each dataset and evaluate their image-text retrieval performance on the validation sets. The results are presented in Table 4, and detailed discussions can be found in the following paragraphs. 

**SciCap.** We subsample 10% of the image-text pairs as labeled data and the remaining 90% of images as unlabeled data. S-CLIP improves upon CLIP, even for the scientific figures domain, which exhibits a substantial gap from the natural image domain that CLIP is pre-trained on. This confirms the practical impact of S-CLIP in extending the applicability of CLIP to various specialist domains. 

**Simpsons.** The Simpsons dataset contains only 800 image-text pairs, so we incorporate unlabeled images from another dataset called Simpsons Characters [33]. S-CLIP significantly improves CLIP, even in the presence of distribution shifts between unlabeled and labeled images. For example, it boosts the text _→_ image retrieval R@1 from 11.8% to 15.8% (+33%). This confirms the practical impact of S-CLIP in specialist domains with limited available image-text pairs. 

## **5.5 Ablation studies** 

We conduct an ablation study on the design choices for S-CLIP. The results are reported in Table 5, which compares the average values of zero-shot accuracy (across five tasks in Table 1) and image-text retrieval R@1 (across six tasks in Table 2) for each design. The observations are as follows: 

- (a) _Training objective._ Caption-level pseudo-labels are more beneficial for image-text retrieval, while keyword-level pseudo-labels are better for zero-shot classification. Intuitively, keyword-level pseudo-labels help the model understand specific words, while caption-level pseudo-labels aid in fine-grained language comprehension. Combining both produces the best results. 

- (b) _Sinkhorn iteration._ Sinkhorn iterations significantly improve Soft-PL (= Iter. 0) by balancing pseudo-labels. However, it saturates after 10 iterations, so we have used it as the default in our experiments. Increasing the number of iterations ensures a more even distribution of pseudolabels. For example, the pseudo-labels from Soft-PL often collapse to some index in the batch. Sinkhorn iterations effectively regularize this behavior, reducing the maximum confidence of 

9 

Table 5: Ablation study on design choices. We report the average values of zero-shot accuracy (left) and image _↔_ text retrieval (right). Bolds denote the best results. See Section 5.5 for discussions. 

|(a) Training objective<br>CLIP (fne-tune)<br>64.7<br>9.3<br>+ Caption-level<br>69.6<br>10.5<br>+ Keyword-level<br>70.0<br>9.9<br>+ Both<br>**71.9**<br>**10.6**<br>(d) Pseudo-target<br>Pseudo-embed<br>68.8<br>9.2<br>Pseudo-label<br>**71.9**<br>**10.6**|(b) Sinkhorn iteration<br>Iter. 0<br>70.6<br>9.2<br>Iter. 1<br>71.4<br>9.4<br>Iter. 5<br>71.6<br>10.2<br>Iter. 10<br>**71.9**<br>**10.6**<br>(e) Embedding for OT<br>Text<br>71.6<br>9.7<br>Image<br>**71.9**<br>**10.6**|(c) Choice of keywords|
|---|---|---|
|||YAKE-100<br>70.0<br>9.8<br>YAKE-200<br>**72.2**<br>9.3<br>YAKE-300<br>70.6<br>10.2<br>Class name<br>71.9<br>**10.6**|
|||(f) Partial label loss<br>Hard-max<br>66.1<br>10.2<br>Soft-max<br>**71.9**<br>**10.6**|



= prediction from 100% to 40%. This regularization is more important in the L U setup, where Soft-PL often harms the performance, while ours consistently provides improvement. 

- (c) _Choice of keywords._ We compare class names and YAKE- _K_ , which implies that _K_ keywords are extracted using the YAKE [74] algorithm. Both types of keywords work well, although class names provide more consistent improvement. YAKE provides both nouns, such as “building,” and adjectives like “green,” “many,” or “large,” in contrast to the class names, which consist only of nouns. This enables YAKE to understand diverse semantics beyond object recognition, making keyword-level pseudo-labeling applicable to any domain. This applicability is further confirmed on the SciCap and Simpsons datasets, where class names are unavailable. 

- (d) _Pseudo-target._ Given a pseudo-label _qi ∈_ R _[N]_ for an unlabeled image _ui_ , one can synthesize a pseudo-embedding _zi_ =[�] _j[q][ij][·][ y][j][∈]_[R] _[d]_[,][where] _[q][ij]_[denotes][the] _[j]_[-th][element][of] _[q][i]_[,][and] _d_ is the dimension of the embeddings _yj_ . Then, one can consider _{ui, zi}_ as paired data and apply contrastive learning to it in addition to the true pairs _{xi, yi}_ . However, we found that pseudo-embedding is less effective than pseudo-labeling. This is because accurately estimating the pseudo-embedding is harder than assigning soft probabilities to the true embeddings. Additionally, the pseudo-embeddings lie on an interpolation of true embeddings, which can introduce confusing negatives and potentially harm the effectiveness of contrastive learning. 

- (e) _Embedding for OT._ We compute pseudo-labels by examining the relationship between unlabeled images and either labeled images or labeled text directly. The relationship with labeled images proves to be more effective in assigning pseudo-labels. Specifically, using text embeddings directly also works reasonably well in the L=U setup. However, it often fails entirely in the L=U setup. This is because the corresponding captions for unlabeled images are unknown. In contrast, visual similarities can be robustly computed even when the images are unseen. 

- (f) _Partial label loss._ We use the soft label given by _στ_ ( _ui,_ K _u_ ) as the target for PLL. We also test the hard version, using the top-1 similar keyword from the candidate set, i.e., _i[∗]_ = arg max _j_ ( _ui · kj_ ). However, it performs worse, aligning with the observations from prior PLL literature. 

We provide additional ablation studies in Appendix E, including results in the general image domain, additional pseudo-labeling baselines, qualitative examples of pseudo-labels, training curves, regularized fine-tuning approaches, selection of mini-batches during training, additional baselines using keyword information, and statistics of keywords in captions. 

## **6 Conclusion** 

We propose S-CLIP, a semi-supervised extension of CLIP that leverages unpaired images to train in specialist domains with limited image-text pairs. We demonstrate its superiority across various domains such as remote sensing, fashion, scientific figures, and comics. We hope that S-CLIP expands the applicability of CLIP. Limitations and broader impacts are discussed in Appendix F. 

## **Acknowledgements** 

This work was supported by Institute of Information & communications Technology Planning & Evaluation (IITP) grant funded by the Korea government (MSIT) (No.2019-0-00075, Artificial 

10 

Intelligence Graduate School Program (KAIST); No.2021-0-02068, Artificial Intelligence Innovation Hub; No.2022-0-00959, Few-shot Learning of Casual Inference in Vision and Language for Decision Making). 

## **References** 

- [1] R. Bommasani, D. A. Hudson, E. Adeli, R. Altman, S. Arora, S. von Arx, M. S. Bernstein, J. Bohg, A. Bosselut, E. Brunskill, et al. On the opportunities and risks of foundation models. _arXiv preprint arXiv:2108.07258_ , 2021. 

- [2] Y. Kim, S. Mo, M. Kim, K. Lee, J. Lee, and J. Shin. Bias-to-text: Debiasing unknown visual biases through language interpretation. _arXiv preprint arXiv:2301.11104_ , 2023. 

- [3] P. Sharma, N. Ding, S. Goodman, and R. Soricut. Conceptual captions: A cleaned, hypernymed, image alt-text dataset for automatic image captioning. In _Annual Conference of the Association for Computational Linguistics_ , 2018. 

- [4] C. Schuhmann, R. Beaumont, R. Vencu, C. Gordon, R. Wightman, M. Cherti, T. Coombes, A. Katta, C. Mullis, M. Wortsman, et al. Laion-5b: An open large-scale dataset for training next generation image-text models. _arXiv preprint arXiv:2210.08402_ , 2022. 

- [5] A. E. Johnson, T. J. Pollard, S. J. Berkowitz, N. R. Greenbaum, M. P. Lungren, C.-y. Deng, R. G. Mark, and S. Horng. Mimic-cxr, a de-identified publicly available database of chest radiographs with free-text reports. _Nature scientific data_ , 2019. 

- [6] X. Yang, H. Zhang, D. Jin, Y. Liu, C.-H. Wu, J. Tan, D. Xie, J. Wang, and X. Wang. Fashion captioning: Towards generating accurate descriptions with semantic rewards. In _European Conference on Computer Vision_ , 2020. 

- [7] L. H. Li, H. You, Z. Wang, A. Zareian, S.-F. Chang, and K.-W. Chang. Unsupervised visionand-language pre-training without parallel images and captions. In _Annual Conference of the North American Chapter of the Association for Computational Linguistics_ , 2021. 

- [8] M. Zhou, L. Yu, A. Singh, M. Wang, Z. Yu, and N. Zhang. Unsupervised vision-and-language pre-training via retrieval-based multi-granular alignment. In _Conference on Computer Vision and Pattern Recognition_ , 2022. 

- [9] T.-J. J. Wang, J. Laaksonen, T. Langer, H. Arponen, and T. E. Bishop. Learning by hallucinating: Vision-language pre-training with weak supervision. In _Winter Conference on Applications of Computer Vision_ , 2023. 

- [10] Z. Wang, Z. Wu, D. Agarwal, and J. Sun. Medclip: Contrastive learning from unpaired medical images and text. In _Conference on Empirical Methods in Natural Language Processing_ , 2022. 

- [11] Y. Li, F. Liang, L. Zhao, Y. Cui, W. Ouyang, J. Shao, F. Yu, and J. Yan. Supervision exists everywhere: A data efficient contrastive language-image pre-training paradigm. In _International Conference on Learning Representations_ , 2022. 

- [12] N. Mu, A. Kirillov, D. Wagner, and S. Xie. Slip: Self-supervision meets language-image pre-training. In _European Conference on Computer Vision_ , 2022. 

- [13] E. Segal, B. Bogin, and J. Berant. Training vision-language models with less bimodal supervision. In _Automated Knowledge Base Construction_ , 2022. 

- [14] R. Windsor, A. Jamaludin, T. Kadir, and A. Zisserman. Vision-language modelling for radiological imaging and reports in the low data regime. _arXiv preprint arXiv:2303.17644_ , 2023. 

- [15] O. Chapelle, B. Scholkopf, and A. Zien. Semi-supervised learning. 2006. _Cambridge, Massachusettes: The MIT Press View Article_ , 2, 2006. 

- [16] D.-H. Lee et al. Pseudo-label: The simple and efficient semi-supervised learning method for deep neural networks. In _Workshop on challenges in representation learning, ICML_ , 2013. 

- [17] X. Liu, F. Zhang, Z. Hou, L. Mian, Z. Wang, J. Zhang, and J. Tang. Self-supervised learning: Generative or contrastive. _IEEE Transactions on Knowledge and Data Engineering_ , 2021. 

- [18] D. Berthelot, N. Carlini, I. Goodfellow, N. Papernot, A. Oliver, and C. A. Raffel. Mixmatch: A holistic approach to semi-supervised learning. In _Neural Information Processing Systems_ , 2019. 

11 

- [19] K. Sohn, D. Berthelot, N. Carlini, Z. Zhang, H. Zhang, C. A. Raffel, E. D. Cubuk, A. Kurakin, and C.-L. Li. Fixmatch: Simplifying semi-supervised learning with consistency and confidence. In _Neural Information Processing Systems_ , 2020. 

- [20] Q. Xie, M.-T. Luong, E. Hovy, and Q. V. Le. Self-training with noisy student improves imagenet classification. In _Conference on Computer Vision and Pattern Recognition_ , 2020. 

- [21] M. Assran, M. Caron, I. Misra, P. Bojanowski, A. Joulin, N. Ballas, and M. Rabbat. Semisupervised learning of visual features by non-parametrically predicting view assignments with support samples. In _International Conference on Computer Vision_ , 2021. 

- [22] M. Bošnjak, P. H. Richemond, N. Tomasev, F. Strub, J. C. Walker, F. Hill, L. H. Buesing, R. Pascanu, C. Blundell, and J. Mitrovic. Semppl: Predicting pseudo-labels for better contrastive representations. In _International Conference on Learning Representations_ , 2023. 

- [23] A. Radford, J. W. Kim, C. Hallacy, A. Ramesh, G. Goh, S. Agarwal, G. Sastry, A. Askell, P. Mishkin, J. Clark, et al. Learning transferable visual models from natural language supervision. In _International Conference on Machine Learning_ , 2021. 

- [24] C. Villani et al. _Optimal transport: old and new_ . Springer, 2009. 

- [25] M. Cuturi. Sinkhorn distances: Lightspeed computation of optimal transport. In _Neural Information Processing Systems_ , 2013. 

- [26] T. Cour, B. Sapp, and B. Taskar. Learning from partial labels. _Journal of Machine Learning Research_ , 2011. 

- [27] X. Lu, B. Wang, X. Zheng, and X. Li. Exploring models and data for remote sensing image caption generation. _IEEE Transactions on Geoscience and Remote Sensing_ , 2017. 

- [28] X. Han, Z. Wu, P. X. Huang, X. Zhang, M. Zhu, Y. Li, Y. Zhao, and L. S. Davis. Automatic spatially-aware fashion concept discovery. In _Proceedings of the IEEE international conference on computer vision_ , pages 1463–1471, 2017. 

- [29] T.-Y. Hsu, C. L. Giles, and T.-H. Huang. Scicap: Generating captions for scientific figures. In _Findings of the Association for Computational Linguistics: EMNLP_ , 2021. 

- [30] D. Adler. Simpsons blip captions. `https://huggingface.co/datasets/Norod78/ simpsons-blip-captions` , 2023. 

- [31] G.-S. Xia, W. Yang, J. Delon, Y. Gousseau, H. Sun, and H. Maître. Structural high-resolution satellite image indexing. 2009. 

- [32] Y. Yang and S. Newsam. Bag-of-visual-words and spatial extensions for land-use classification. In _SIGSPATIAL international conference on advances in geographic information systems_ , 2010. 

- [33] A. Attia. The simpsons characters data. `https://www.kaggle.com/datasets/ alexattia/the-simpsons-characters-dataset` , 2018. 

- [34] L. H. Li, M. Yatskar, D. Yin, C.-J. Hsieh, and K.-W. Chang. Visualbert: A simple and performant baseline for vision and language. In _Annual Conference of the Association for Computational Linguistics_ , 2020. 

- [35] Y.-C. Chen, L. Li, L. Yu, A. El Kholy, F. Ahmed, Z. Gan, Y. Cheng, and J. Liu. Uniter: Universal image-text representation learning. In _European Conference on Computer Vision_ , 2020. 

- [36] W. Kim, B. Son, and I. Kim. Vilt: Vision-and-language transformer without convolution or region supervision. In _International Conference on Machine Learning_ , 2021. 

- [37] Z. Wang, J. Yu, A. W. Yu, Z. Dai, Y. Tsvetkov, and Y. Cao. Simvlm: Simple visual language model pretraining with weak supervision. _arXiv preprint arXiv:2108.10904_ , 2021. 

- [38] W. Wang, H. Bao, L. Dong, J. Bjorck, Z. Peng, Q. Liu, K. Aggarwal, O. K. Mohammed, S. Singhal, S. Som, et al. Image as a foreign language: Beit pretraining for all vision and vision-language tasks. _arXiv preprint arXiv:2208.10442_ , 2022. 

- [39] C. Jia, Y. Yang, Y. Xia, Y.-T. Chen, Z. Parekh, H. Pham, Q. Le, Y.-H. Sung, Z. Li, and T. Duerig. Scaling up visual and vision-language representation learning with noisy text supervision. In _International Conference on Machine Learning_ , 2021. 

12 

- [40] H. Pham, Z. Dai, G. Ghiasi, H. Liu, A. W. Yu, M.-T. Luong, M. Tan, and Q. V. Le. Combined scaling for zero-shot transfer learning. _arXiv preprint arXiv:2111.10050_ , 2021. 

- [41] X. Zhai, X. Wang, B. Mustafa, A. Steiner, D. Keysers, A. Kolesnikov, and L. Beyer. Lit: Zero-shot transfer with locked-image text tuning. In _Conference on Computer Vision and Pattern Recognition_ , 2022. 

- [42] Y. Li, H. Fan, R. Hu, C. Feichtenhofer, and K. He. Scaling language-image pre-training via masking. _arXiv preprint arXiv:2212.00794_ , 2022. 

- [43] K. Desai and J. Johnson. Virtex: Learning visual representations from textual annotations. In _Conference on Computer Vision and Pattern Recognition_ , 2021. 

- [44] J. Li, D. Li, C. Xiong, and S. Hoi. Blip: Bootstrapping language-image pre-training for unified vision-language understanding and generation. In _International Conference on Machine Learning_ , 2022. 

- [45] J. Yu, Z. Wang, V. Vasudevan, L. Yeung, M. Seyedhosseini, and Y. Wu. Coca: Contrastive captioners are image-text foundation models. _Transactions on Machine Learning Research_ , 2022. 

- [46] M. Tsimpoukelli, J. L. Menick, S. Cabi, S. Eslami, O. Vinyals, and F. Hill. Multimodal few-shot learning with frozen language models. In _Neural Information Processing Systems_ , 2021. 

- [47] J.-B. Alayrac, J. Donahue, P. Luc, A. Miech, I. Barr, Y. Hasson, K. Lenc, A. Mensch, K. Millican, M. Reynolds, et al. Flamingo: a visual language model for few-shot learning. In _Neural Information Processing Systems_ , 2022. 

- [48] J. Li, D. Li, S. Savarese, and S. Hoi. Blip-2: Bootstrapping language-image pre-training with frozen image encoders and large language models. _arXiv preprint arXiv:2301.12597_ , 2023. 

- [49] H. Luo, L. Ji, M. Zhong, Y. Chen, W. Lei, N. Duan, and T. Li. Clip4clip: An empirical study of clip for end to end video clip retrieval and captioning. _Neurocomputing_ , 508:293–304, 2022. 

- [50] X. Gu, T.-Y. Lin, W. Kuo, and Y. Cui. Open-vocabulary object detection via vision and language knowledge distillation. In _International Conference on Learning Representations_ , 2022. 

- [51] M. Xu, Z. Zhang, F. Wei, Y. Lin, Y. Cao, H. Hu, and X. Bai. A simple baseline for openvocabulary semantic segmentation with pre-trained vision-language model. In _European Conference on Computer Vision_ , 2022. 

- [52] J. Jeong, Y. Zou, T. Kim, D. Zhang, A. Ravichandran, and O. Dabeer. Winclip: Zero-/few-shot anomaly classification and segmentation. In _Conference on Computer Vision and Pattern Recognition_ , 2023. 

- [53] X. Wang, Y. Peng, L. Lu, Z. Lu, and R. M. Summers. Tienet: Text-image embedding network for common thorax disease classification and reporting in chest x-rays. In _Conference on Computer Vision and Pattern Recognition_ , 2018. 

- [54] Y. Zhang, H. Jiang, Y. Miura, C. D. Manning, and C. P. Langlotz. Contrastive learning of medical visual representations from paired images and text. In _Machine Learning for Healthcare Conference_ , 2022. 

- [55] S.-C. Huang, L. Shen, M. P. Lungren, and S. Yeung. Gloria: A multimodal global-local representation learning framework for label-efficient medical image recognition. In _International Conference on Computer Vision_ , 2021. 

- [56] R. Liao, D. Moyer, M. Cha, K. Quigley, S. Berkowitz, S. Horng, P. Golland, and W. M. Wells. Multimodal representation learning via maximization of local mutual information. In _Medical Image Computing and Computer Assisted Intervention_ , 2021. 

- [57] B. Boecking, N. Usuyama, S. Bannur, D. C. Castro, A. Schwaighofer, S. Hyland, M. Wetscherek, T. Naumann, A. Nori, J. Alvarez-Valle, et al. Making the most of text semantics to improve biomedical vision–language processing. In _European Conference on Computer Vision_ , 2022. 

- [58] P. Müller, G. Kaissis, C. Zou, and D. Rueckert. Joint learning of localized representations from medical images and reports. In _European Conference on Computer Vision_ , 2022. 

13 

- [59] S. Zhang, Y. Xu, N. Usuyama, J. Bagga, R. Tinn, S. Preston, R. Rao, M. Wei, N. Valluri, C. Wong, et al. Large-scale domain-specific pretraining for biomedical vision-language processing. _arXiv preprint arXiv:2303.00915_ , 2023. 

- [60] D. Gao, L. Jin, B. Chen, M. Qiu, P. Li, Y. Wei, Y. Hu, and H. Wang. Fashionbert: Text and image matching with adaptive loss for cross-modal retrieval. In _International ACM SIGIR Conference on Research and Development in Information Retrieval_ , 2020. 

- [61] M. Zhuge, D. Gao, D.-P. Fan, L. Jin, B. Chen, H. Zhou, M. Qiu, and L. Shao. Kaleido-bert: Vision-language pre-training on fashion domain. In _Conference on Computer Vision and Pattern Recognition_ , 2021. 

- [62] S. Goenka, Z. Zheng, A. Jaiswal, R. Chada, Y. Wu, V. Hedau, and P. Natarajan. Fashionvlp: Vision language transformer for fashion retrieval with feedback. In _Conference on Computer Vision and Pattern Recognition_ , 2022. 

- [63] X. Han, L. Yu, X. Zhu, L. Zhang, Y.-Z. Song, and T. Xiang. Fashionvil: Fashion-focused vision-and-language representation learning. In _European Conference on Computer Vision_ , 2022. 

- [64] A. Arampacha, V. Dev, G. Venkatesh, B. Mayank, G. Ritobrata, and P. Sujit. Fine tuning clip with remote sensing (satellite) images and captions. `https://huggingface.co/blog/ fine-tune-clip-rsicd` , 2021. 

- [65] T.-M. H. Hsu, W.-H. Weng, W. Boag, M. McDermott, and P. Szolovits. Unsupervised multimodal representation learning across medical images and reports. In _NeurIPS Machine Learning for Health Workshop_ , 2018. 

- [66] Y. Bengio, O. Delalleau, and N. Le Roux. label propagation and quadratic criterion. 2006. 

- [67] A. Oliver, A. Odena, C. A. Raffel, E. D. Cubuk, and I. Goodfellow. Realistic evaluation of deep semi-supervised learning algorithms. In _Neural Information Processing Systems_ , 2018. 

- [68] J. Park, S. Yun, J. Jeong, and J. Shin. Opencos: Contrastive semi-supervised learning for handling open-set unlabeled data. _arXiv preprint arXiv:2107.08943_ , 2021. 

- [69] S. Mo, J.-C. Su, C.-Y. Ma, M. Assran, I. Misra, L. Yu, and S. Bell. Ropaws: Robust semisupervised representation learning from uncurated data. In _International Conference on Learning Representations_ , 2023. 

- [70] Y. Grandvalet and Y. Bengio. Semi-supervised learning by entropy minimization. In _Neural Information Processing Systems_ , 2004. 

- [71] J. Zhou, L. Dong, Z. Gan, L. Wang, and F. Wei. Non-contrastive learning meets language-image pre-training. _arXiv preprint arXiv:2210.09304_ , 2022. 

- [72] S. Goel, H. Bansal, S. Bhatia, R. Rossi, V. Vinay, and A. Grover. Cyclip: Cyclic contrastive language-image pretraining. In _Neural Information Processing Systems_ , 2022. 

- [73] Y. Gao, J. Liu, Z. Xu, T. Wu, W. Liu, J. Yang, K. Li, and X. Sun. Softclip: Softer cross-modal alignment makes clip stronger. _arXiv preprint arXiv:2303.17561_ , 2023. 

- [74] R. Campos, V. Mangaravite, A. Pasquali, A. Jorge, C. Nunes, and A. Jatowt. Yake! keyword extraction from single documents using multiple local features. _Information Sciences_ , 509: 257–289, 2020. 

- [75] J. Lv, M. Xu, L. Feng, G. Niu, X. Geng, and M. Sugiyama. Progressive identification of true labels for partial-label learning. In _International Conference on Machine Learning_ , 2020. 

- [76] H. Wang, R. Xiao, Y. Li, L. Feng, G. Niu, G. Chen, and J. Zhao. Pico: Contrastive label disambiguation for partial label learning. In _International Conference on Learning Representations_ , 2022. 

- [77] G. Ilharco, M. Wortsman, R. Wightman, C. Gordon, N. Carlini, R. Taori, A. Dave, V. Shankar, H. Namkoong, J. Miller, H. Hajishirzi, A. Farhadi, and L. Schmidt. Openclip, 2021. 

- [78] F. Zhang, B. Du, and L. Zhang. Saliency-guided unsupervised feature learning for scene classification. _IEEE transactions on Geoscience and Remote Sensing_ , 2014. 

- [79] G. Cheng, J. Han, and X. Lu. Remote sensing image scene classification: Benchmark and state of the art. _Proceedings of the IEEE_ , 2017. 

14 

- [80] Q. Zou, L. Ni, T. Zhang, and Q. Wang. Deep learning based feature selection for remote sensing scene classification. _IEEE Geoscience and Remote Sensing Letters_ , 2015. 

- [81] G.-S. Xia, J. Hu, F. Hu, B. Shi, X. Bai, Y. Zhong, L. Zhang, and X. Lu. Aid: A benchmark data set for performance evaluation of aerial scene classification. _IEEE Transactions on Geoscience and Remote Sensing_ , 2017. 

- [82] N. Rostamzadeh, S. Hosseini, T. Boquet, W. Stokowiec, Y. Zhang, C. Jauvin, and C. Pal. Fashion-gen: The generative fashion dataset and challenge. _arXiv preprint arXiv:1806.08317_ , 2018. 

- [83] M. I. Vasileva, B. A. Plummer, K. Dusad, S. Rajpal, R. Kumar, and D. Forsyth. Learning type-aware embeddings for fashion compatibility. In _Proceedings of the European conference on computer vision (ECCV)_ , pages 390–405, 2018. 

- [84] T. Chen, S. Kornblith, M. Norouzi, and G. Hinton. A simple framework for contrastive learning of visual representations. In _International Conference on Machine Learning_ , 2020. 

- [85] K. He, H. Fan, Y. Wu, S. Xie, and R. Girshick. Momentum contrast for unsupervised visual representation learning. In _Conference on Computer Vision and Pattern Recognition_ , 2020. 

- [86] S. Yuan, K. Bai, L. Chen, Y. Zhang, C. Tao, C. Li, G. Wang, R. Henao, and L. Carin. Weakly supervised cross-domain alignment with optimal transport. In _British Machine Vision Conference_ , 2020. 

- [87] G. Chen, W. Yao, X. Song, X. Li, Y. Rao, and K. Zhang. Prompt learning with optimal transport for vision-language models. In _International Conference on Learning Representations_ , 2023. 

- [88] K. Kim, Y. Oh, and J. C. Ye. Zegot: Zero-shot segmentation through optimal transport of text prompts. _arXiv preprint arXiv:2301.12171_ , 2023. 

- [89] T. Gupta, A. Vahdat, G. Chechik, X. Yang, J. Kautz, and D. Hoiem. Contrastive learning for weakly supervised phrase grounding. In _European Conference on Computer Vision_ , 2020. 

- [90] B. Wu, R. Cheng, P. Zhang, P. Vajda, and J. E. Gonzalez. Data efficient language-supervised zero-shot recognition with optimal transport distillation. In _International Conference on Learning Representations_ , 2022. 

- [91] F. Taherkhani, A. Dabouei, S. Soleymani, J. Dawson, and N. M. Nasrabadi. Transporting labels via hierarchical optimal transport for semi-supervised learning. In _European Conference on Computer Vision_ , 2020. 

- [92] F. Taherkhani, A. Dabouei, S. Soleymani, J. Dawson, and N. M. Nasrabadi. Self-supervised wasserstein pseudo-labeling for semi-supervised image classification. In _Conference on Computer Vision and Pattern Recognition_ , 2021. 

- [93] D. Dwibedi, Y. Aytar, J. Tompson, P. Sermanet, and A. Zisserman. With a little help from my friends: Nearest-neighbor contrastive learning of visual representations. In _International Conference on Computer Vision_ , 2021. 

- [94] M. Zheng, F. Wang, S. You, C. Qian, C. Zhang, X. Wang, and C. Xu. Weakly supervised contrastive learning. In _International Conference on Computer Vision_ , 2021. 

- [95] M. Azabou, M. G. Azar, R. Liu, C.-H. Lin, E. C. Johnson, K. Bhaskaran-Nair, M. Dabagia, B. Avila-Pires, L. Kitchell, K. B. Hengen, et al. Mine your own view: Self-supervised learning through across-sample prediction. _arXiv preprint arXiv:2102.10106_ , 2021. 

- [96] P. Khosla, P. Teterwak, C. Wang, A. Sarna, Y. Tian, P. Isola, A. Maschinot, C. Liu, and D. Krishnan. Supervised contrastive learning. In _Neural Information Processing Systems_ , 2020. 

- [97] M. Assran, N. Ballas, L. Castrejon, and M. Rabbat. Supervision accelerates pre-training in contrastive semi-supervised learning of visual representations. _arXiv preprint arXiv:2006.10803_ , 2020. 

- [98] B. Qu, X. Li, D. Tao, and X. Lu. Deep semantic understanding of high resolution remote sensing image. In _International conference on computer, information and telecommunication systems_ , 2016. 

- [99] K. He, X. Zhang, S. Ren, and J. Sun. Deep residual learning for image recognition. In _Conference on Computer Vision and Pattern Recognition_ , 2016. 

15 

- [100] A. Dosovitskiy, L. Beyer, A. Kolesnikov, D. Weissenborn, X. Zhai, T. Unterthiner, M. Dehghani, M. Minderer, G. Heigold, S. Gelly, et al. An image is worth 16x16 words: Transformers for image recognition at scale. In _International Conference on Learning Representations_ , 2020. 

- [101] T.-Y. Lin, M. Maire, S. Belongie, J. Hays, P. Perona, D. Ramanan, P. Dollár, and C. L. Zitnick. Microsoft coco: Common objects in context. In _European Conference on Computer Vision_ , 2014. 

16 

## **A Relationship between OT and Soft-PL** 

The entropic-regularized optimal transport problem described in Eq. (2) can be efficiently solved using the Sinkhorn-Knopp [25] algorithm. For our case, the cost function C _∈_ R _[M][×][N]_ is given by the negative cosine similarity: C _ij_ = 1 _− ui · xj_ . The Sinkhorn-Knopp algorithm proceeds as follows: first, it computes the exponentiated cost matrix K = exp( _−_[C] _λ_[)][, where] _[ λ]_[ denotes the scale] of the entropic regularizer. Then, the algorithm iteratively updates two vectors: **u** _∈_ R _[M][×]_[1] and **v** _∈_ R[1] _[×][N]_ , which are initialized with uniform probability. The transportation plan Γ can be estimated using the exponentiated cost matrix K and the current vectors **u** and **v** . Specifically, it is given by Γ = **u** _⊙_ K _⊙_ **v** , where _⊙_ denotes element-wise multiplication. 

At the initialization zero, the transportation plan Γ is obtained by normalizing the exponentiated cost matrix K since the vectors **u** and **v** are uniform. Consequently, our caption-level pseudo-label _qi_ is given by _qi_ = K _ij/_[�] _j_[K] _[ij]_[, which is equivalent to the Soft-PL] _[ σ][λ]_[(] _[u][i][, x][i]_[)][ except for the softmax] temperature, which is denoted as _λ_ . This observation motivates us to set _λ_ = _τ_ , matching the same temperature as CLIP, for computing the relation between embeddings. From this perspective, our proposed OT approach enhances the robustness of Soft-PL in pseudo-label assignment, especially in cases with distribution shifts between unlabeled and labeled images. Our ablation study demonstrates that increasing the number of Sinkhorn iterations improves the training process. After initialization, the vectors **u** and **v** are updated to align with the exponentiated cost matrix K and satisfy the flow constraints **p** and **q** . Specifically, given the current vectors **u**[(] _[t]_[)] and **v**[(] _[t]_[)] , the next iteration of vectors is obtained as **u**[(] _[t]_[+1)] = K **vp**[(] _[t]_[)][and] **[ v]**[(] _[t]_[+1)][=] K _[⊤]_ **uq**[(] _[t]_[+1)][.][This iterative process] converges to the optimal solution. Our ablation study demonstrates that a small number of iterations is sufficiently effective, and we employ 10 iterations in our experiments. 

## **B Additional related work** 

**Optimal transport for vision-language models.** Several prior works have applied optimal transport to vision-language models. However, most of them focus on transportation between an image and a caption, connecting objects and phrases [35, 36, 86–88]. This is known as weakly supervised phrase grounding [89], and it assumes a supervised setting with image-text pairs. OTTER [90] calculates transportation between images and text, which aligns with our approach. However, they also assume a supervised setting and focus on calibrating the one-hot target of CLIP. To our knowledge, our work is the first to apply optimal transport for semi-supervised CLIP training. 

**Optimal transport for semi-supervised learning.** Some prior works have utilized optimal transport for semi-supervised learning, specifically in image classification [91, 92]. These studies demonstrate the benefits of optimal transport in pseudo-labeling class labels. However, we argue that OT plays a more critical role in vision-language models due to inherent distribution shifts between labeled and unlabeled image captions. Our experiments reveal that naive pseudo-labeling often has detrimental effects, while OT significantly improves semi-supervised CLIP training. 

**Partial label learning.** Partial label learning (PLL) [26] relaxes the classification problem by using a candidate set as the target instead of the exact one. This involves optimizing for the best candidate within the set. Instead of determining the hard assignment of the candidate, PRODEN [75] computes soft labels for the elements in the set to smooth the optimization process. PiCO [76] further enhances this approach by incorporating contrastive learning. Our work extends the techniques of PLL to a novel problem: keyword-level pseudo-labeling for vision-language models. 

**Label propagation in contrastive learning.** The concept of pseudo-labeling (or label propagation) is also applied in image-only contrastive learning. In addition to using the same image with different augmentations [84, 85], some studies explore the utilization of neighboring images as positive pairs [93–95]. Similar concept can be extended to supervised [96] and semi-supervised [22, 97] contrastive learning, where images sharing the same class label are considered positive pairs. 

17 

## **C Experimental details** 

## **C.1 Dataset details** 

**Remote sensing.** We use the union of RSICD [27], UCM [32], and Sydney [78] datasets for training. The captions for the UCM and Sydney datasets were annotated by Qu et al. [98]. Figure 4 displays the example images and captions, while Table 6 presents the statistics of each dataset. Since an image may have multiple associated captions, we randomly select one for each iteration. 


![](raw/papers/2_LLM_NLP/LLM_paper_list/caption/nips_2023_S-CLIP_Semi-supervised_Vision-Language_Learning_using_Few_Specialist_Captions_images/nips_2023_S-CLIP_Semi-supervised_Vision-Language_Learning_using_Few_Specialist_Captions.pdf-0018-03.png)


**----- Start of picture text -----**<br>
“a playground with eight basketball  “many houses arranged neatly and  “a residential area with houses<br>fields near it is surrounded by many  some roads crossed them with some  arranged neatly and some roads go<br>green trees” plants in the roadside” through this area”<br>(a) RSICD (b) UCM (c) Sydney<br>**----- End of picture text -----**<br>


Figure 4: Examples of image-text pairs in the remote sensing domain. 

Table 6: Statistics of image-text paired datasets in the remote sensing domain. 

|# of pairs|RSICD<br>UCM<br>Sydney|
|---|---|
||8,734<br>1,680<br>497|



We use the validation sets of RSCID-CLS [27], UCM-CLS [32], WHU-RS19 [31], RSSCN7 [80], and AID [81] datasets for zero-shot classification. RSCID-CLS and UCM-CLS are the classification versions of RSCID and UCM, respectively. Table 7 presents the statistics of each dataset. 

Table 7: Statistics of classification datasets in the remote sensing domain. 

|# of images<br># of classes|RSICD-CLS<br>UCM-CLS<br>WHU-RS19<br>RSSCN7<br>AID|
|---|---|
||1,094<br>2,100<br>1,005<br>2,800<br>10,000<br>31<br>21<br>19<br>7<br>30|



**Fashion.** We use the union of Fashion200k [28], FashionGen [82], and Polyvore Outfits [83] datasets for training. Figure 5 displays the example images and captions, while Table 8 presents the statistics of each dataset. Since an item may have multiple associated views of images corresponding to the caption, we randomly select one view for each iteration. For captions, we concatenate the titles and descriptions from FashionGen and Polyvore, and use the sentences from Fashion200k. 


![](raw/papers/2_LLM_NLP/LLM_paper_list/caption/nips_2023_S-CLIP_Semi-supervised_Vision-Language_Learning_using_Few_Specialist_Captions_images/nips_2023_S-CLIP_Semi-supervised_Vision-Language_Learning_using_Few_Specialist_Captions.pdf-0018-11.png)


**----- Start of picture text -----**<br>
“grey classic u-neck t-shirt. short  “off-white striped flared pants.<br>“white printed silk-jersey wrap dress  sleeve t-shirt in heather grey. tonal  multicoloured cotton and silk striped<br>midnight blue” logo flag at u-neck collar.” flared pants from off-white.”<br>(a) Fashion200k (b) FashionGen (c) Polyvore<br>**----- End of picture text -----**<br>


Figure 5: Examples of image-text pairs in the fashion domain. 

18 

Table 8: Statistics of image-text paired datasets in the fashion domain. 

|# of pairs|Fashion200k<br>FashionGen<br>Polyvore|
|---|---|
||61,753<br>60,147<br>71,967|



We use the validation sets of Fashion200k, FashionGen, and Polyvore datasets for zero-shot classification. Table 9 presents the statistics of each dataset. For Fashion200k and FashionGen, we report both the super-class and the sub-class. The class names in Polyvore match the same level of the super-class as those in the other datasets. For example, the super-class contains class names such as “dress,” while the sub-class contains class names such as “casual and day dress.” 

Table 9: Statistics of classification datasets in the fashion domain. 

|# of images<br># of classes|Fashion200k<br>FashionGen<br>Polyvore<br>Super-class<br>Sub-class<br>Super-class<br>Sub-class<br>Class|
|---|---|
||29,785<br>32,528<br>14,657<br>5<br>31<br>48<br>121<br>11|



**Scientific figures.** We use the SciCap [29] dataset for training. Figure 6 displays the example images and captions, while Table 10 presents the statistics of each dataset. We use figures that do not have subfigures, denoted as the “SciCap-No-Subfig-Img” subset, for simplicity. 

**Comics.** We use the Simpsons [30] dataset as labeled data and the Simpsons Character [33] dataset as unlabeled data. Figure 6 displays the example images and captions, while Table 10 presents the statistics of each dataset. We use 90% of pairs from the Simpsons dataset as the training split and the remaining 10% as the validation split. Since the training split contains only 720 image-text pairs, it is necessary to incorporate images from different datasets like Simpsons Characters. 


![](raw/papers/2_LLM_NLP/LLM_paper_list/caption/nips_2023_S-CLIP_Semi-supervised_Vision-Language_Learning_using_Few_Specialist_Captions_images/nips_2023_S-CLIP_Semi-supervised_Vision-Language_Learning_using_Few_Specialist_Captions.pdf-0019-07.png)


**----- Start of picture text -----**<br>
“Fig. 2. variable 2-D initial probe<br>distribution used for CFO runs  “simpsons wearing headphones and<br>reported here” looking like a guy”<br>(a) SciCap (b) Simpsons<br>**----- End of picture text -----**<br>


Figure 6: Examples of image-text pairs in the scientific figure and comics domains. 

Table 10: Statistics of image-text paired datasets in the scientific figure and comics domains. 

|# of pairs|SciCap<br>Simpsons|
|---|---|
||106,834<br>720|



19 

## **C.2 Implementation details** 

**Training.** We utilize the OpenCLIP [77] library to implement our algorithms. By default, we follow the original training configuration. We use a batch size of 64 per GPU, with a total of 4 GPUs. The learning rate is set to 5e-5, and we apply the default cosine learning rate scheduling with a warmup period of the first 10 steps. We train all models until the performance saturate, which can vary over the number of image-text pairs. Specifically, for remote sensing, we train models for 25 epochs, for fashion 10 epochs, for scientific figures 5 epochs, and for comics datasets 10 epochs. We run three trials and report the average and standard deviation for all experiments. 

We add the supervised CLIP loss and pseudo-label losses for semi-supervised methods. We halve the scale of pseudo-label losses to match the scale of the image part in the CLIP loss. We set the scale of the entropic regularizer to be the same as the softmax temperature of CLIP, denoted as = _τ_ , and set the number of Sinkhorn iterations to 10. Note that our method does not introduce any additional hyperparameters, except for the selection of keyword candidate sets discussed below. 

**Keywords.** We use the available class names for remote sensing and fashion datasets while extracting the keywords using the YAKE [74] algorithm from the captions for the SciCap and Simpsons datasets. For remote sensing, we combine the class names from RSICD-CLS and UCM-CLS to obtain 45 keywords, of which around 30 words appeared in the captions. For fashion, we combine the superclass names from Fashion200k and FashionGen to obtain 56 keywords, which are more effective than sub-class names since the captions often do not contain the exact sub-class names. 

Our ablation study presents that the keywords extracted by YAKE are also effective, even outperforming the class names in some cases. In the case of remote sensing, the YAKE keywords contain not only nouns like “building” but also adjectives like “green,” “many,” or “large,” which help the model understand the fine-grained information of the scene. Moreover, we utilize YAKE-100 keywords for SciCap and Simpsons when class names are unavailable, and this approach has proven to be effective. Thus, we believe that a reasonable choice of keywords would firmly benefit our method. 

**Computation.** The computation bottleneck mostly lies in computing embeddings rather than the loss function. Thus, all methods require a similar amount of time for a single training iteration, except for S-CLIP, which takes longer due to the additional forwarding of keywords. Training duration varies depending on the number of image-text pairs, taking a few minutes to a few hours. Semi-supervised methods take twice as long as supervised CLIP since we draw batches of labeled and unlabeled data with an even size, necessitating twice as many iterations for the same epoch. 

**Evaluation.** We use the evaluation code provided by OpenCLIP. For zero-shot classification, we use the default template: “a photo of a [class].” We use this template for fashion datasets but use the template “an aerial photograph of a [class]” for remote sensing, following the setup of [64]. 

20 

## **D Additional remote sensing results** 

## **D.1 Different neural architectures** 

Table 11 presents the zero-shot classification results using the ResNet [99] and ViT [100] models. S-CLIP significantly improves the supervised baseline in all considered scenarios. Note that the larger model shows better overall accuracy, for both the original and fine-tuned CLIP. Here, S-CLIP gives reliable improvements when scaling up to larger models. 

Table 11: Zero-shot classification results on remote sensing datasets using different neural architectures. Parentheses indicate the performance gap from the supervised CLIP, and bolds denote the best results within each architecture. S-CLIP give consistent improvements, scaling to larger networks. 

|Model|Method|RSICD-CLS|UCM-CLS|WHU-RS19|RSSCN7|AID|
|---|---|---|---|---|---|---|
||CLIP (original)|45.3|50.5|65.5|58.9|47.8|
|ResNet-50|CLIP (fne-tune)|58.3_±_0.3|63.5_±_3.4|76.5_±_3.2|61.9_±_1.2|63.1_±_1.3|
||S-CLIP (ours)|**66.9**_±_1.7(+8.6)|**66.7**_±_1.6(+3.2)|**86.9**_±_2.0(+10.4)|**66.2**_±_1.1(+4.3)|**73.0**_±_0.3(+9.9)|
||CLIP (original)|55.8|58.6|76.4|62.1|55.7|
|ViT-B/32|CLIP (fne-tune)|72.3_±_1.2|76.8_±_3.0|90.4_±_0.6|68.9_±_2.2|78.9_±_0.5|
||S-CLIP (ours)|**77.5**_±_1.0(+5.2)|**78.3**_±_3.6(+1.5)|**93.2**_±_1.9(+2.8)|**69.6**_±_0.8(+0.7)|**83.7**_±_1.8(+4.8)|
||CLIP (original)|58.9|60.1|80.9|69.4|59.6|
|ViT-B/16|CLIP (fne-tune)|72.4_±_0.6|78.7_±_2.7|90.0_±_1.4|71.9_±_3.8|77.9_±_0.9|
||S-CLIP (ours)|**79.5**_±_1.5(+7.1)|**82.3**_±_0.4(+3.6)|**93.9**_±_2.1(+2.1)|**76.3**_±_0.3(+4.4)|**85.2**_±_0.9(+7.3)|



## **D.2 Different image-text pair ratios** 

Table 12 presents the zero-shot classification results for different image-text pair ratios. Using only 10% of pairs (2 _×_ fewer than CLIP), S-CLIP achieves performance comparable to CLIP using 20% of pairs. Moreover, with 30% of pairs (3 _×_ fewer than CLIP), S-CLIP matches the performance of CLIP using 100% of pairs and even outperforms it, possibly due to the keyword-level pseudo-labeling. 

Table 12: Zero-shot classification results on remote sensing datasets using different image-text pair ratios. Parentheses indicate the performance gap from the supervised CLIP, and bolds denote the best results within each pair ratio. S-CLIP gives consistent improvements, reducing required pairs. 

|Method|Ratio|RSICD-CLS|UCM-CLS|WHU-RS19|RSSCN7|AID|
|---|---|---|---|---|---|---|
|CLIP (original)|0%|45.3|50.5|65.5|58.9|47.8|
|CLIP (fne-tune)<br>S-CLIP (ours)|10%|58.3_±_0.3<br>**66.9**_±_1.7(+8.6)|63.5_±_3.4<br>**66.7**_±_1.6(+3.2)|76.5_±_3.2<br>**86.9**_±_2.0(+10.4)|61.9_±_1.2<br>**66.2**_±_1.1(+4.3)|63.1_±_1.3<br>**73.0**_±_0.3(+9.9)|
|CLIP (fne-tune)<br>S-CLIP (ours)|20%|66.2_±_1.1<br>**77.6**_±_1.1(+11.4)|69.7_±_0.2<br>**72.0**_±_0.7(+2.3)|82.8_±_1.5<br>**90.7**_±_1.9(+7.9)|65.6_±_3.4<br>**67.9**_±_2.6(+2.3)|71.8_±_1.1<br>**83.7**_±_0.7(+11.9)|
|CLIP (fne-tune)<br>S-CLIP (ours)|30%|70.8_±_1.6<br>**80.1**_±_1.7(+9.3)|70.7_±_2.3<br>**74.0**_±_1.1(+3.3)|85.4_±_1.7<br>**94.8**_±_1.2(+9.4)|63.6_±_1.5<br>**69.6**_±_2.8(+6.0)|75.7_±_0.8<br>**87.8**_±_0.5(+12.1)|
|CLIP (fne-tune)|100%|77.5_±_1.5|76.5_±_0.5|93.9_±_0.9|71.2_±_0.3|83.6_±_1.6|



21 

## **D.3 Using all image-text pairs** 

We used 10% of the image-text pairs as labeled data to compare two scenarios, L=U and L=U, in the main paper. Having demonstrated the effectiveness of S-CLIP in the L=U scenario, the next question is whether it would also be helpful when using all paired data and extra unlabeled data. Table 13 presents the results of using the entire RS-ALL dataset as labeled data and RESISC45 as the unlabeled data. S-CLIP is shown to be effective even in this challenging scenario. 

Table 13: Zero-shot classification results on remote sensing datasets using all captioned images and extra unlabeled images, where the unlabeled data is from a different distribution than the labeled data. S-CLIP also provides improvements in this most challenging scenario. 

|Model|Method|RSICD-CLS|UCM-CLS|WHU-RS19|RSSCN7|AID|
|---|---|---|---|---|---|---|
|ResNet-50|CLIP (fne-tune)<br>S-CLIP (ours)|77.5_±_1.5<br>**81.1**_±_0.2(+3.6)|76.5_±_0.5<br>**77.7**_±_1.4(+1.2)|93.9_±_0.9<br>**95.8**_±_0.9(+1.9)|71.2_±_0.3<br>**72.3**_±_2.8(+1.1)|83.6_±_1.6<br>**87.7**_±_0.4(+4.1)|
|ViT-B/32|CLIP (fne-tune)<br>S-CLIP (ours)|85.3_±_1.0<br>**87.0**_±_0.6(+1.7)|87.3_±_0.7<br>**88.6**_±_1.4(+1.3)|95.8_±_1.0<br>**97.2**_±_0.4(+1.4)|75.7_±_1.4<br>**76.2**_±_0.2(+0.5)|92.0_±_0.5<br>**92.5**_±_0.8(+0.5)|
|ViT-B/16|CLIP (fne-tune)<br>S-CLIP (ours)|87.0_±_0.3<br>**87.4**_±_1.2(+0.4)|88.5_±_0.6<br>**88.9**_±_1.6(+0.4)|96.7_±_0.1<br>**97.3**_±_0.4(+0.6)|78.4_±_0.8<br>**79.1**_±_1.0(+0.7)|89.2_±_1.4<br>**93.1**_±_1.4(+3.9)|



## **D.4 Image-text retrieval R@1** 

Table 14 presents the R@1 results of image-text retrieval, complementing Table 2 in the main paper which shows the R@5 results. S-CLIP outperforms the semi-supervised methods in most cases, except the supervised baseline works best for image _→_ text retrieval on the UCM dataset. 

Table 14: Image-text retrieval results on remote sensing datasets, following the same setup of Table 1. Bolds denote the best results among the semi-supervised methods within the same setups. S-CLIP performs the best in most cases, similar to the observation in Table 2. 

|Method<br>Data|Image_→_text R@1<br>RSICD<br>UCM<br>Sydney|Text_→_image R@1<br>RSICD<br>UCM<br>Sydney|
|---|---|---|
|CLIP (original)<br>-|2.1<br>7.1<br>10.3|2.2<br>7.1<br>20.7|
|CLIP (fne-tune)<br>L|3.6_±_0.2<br>10.6_±_2.1<br>13.2_±_2.0|3.7_±_0.3<br>10.0_±_0.5<br>14.4_±_2.0|
|Hard-PL [16]<br>L=U<br>Soft-PL [21]<br>S-CLIP (ours)|3.8_±_0.1<br>9.7_±_2.7<br>14.4_±_1.0<br>4.0_±_0.3<br>11.0_±_0.8<br>11.5_±_2.0<br>**4.2**_±_0.1<br>**11.6**_±_1.1<br>**14.9**_±_1.0|3.9_±_0.7<br>9.2_±_1.8<br>10.9_±_1.0<br>**4.2**_±_0.4<br>9.8_±_1.5<br>16.1_±_4.3<br>**4.2**_±_0.6<br>**11.1**_±_0.5<br>**17.8**_±_1.0|
|Hard-PL [16]<br>L=U<br>Soft-PL [21]<br>S-CLIP (ours)|3.8_±_0.3<br>8.1_±_1.0<br>13.4_±_1.2<br>3.8_±_0.4<br>8.4_±_1.2<br>10.9_±_1.0<br>**4.2**_±_0.6<br>**9.8**_±_0.6<br>**13.8**_±_0.2|3.4_±_0.2<br>8.9_±_1.4<br>12.6_±_1.0<br>3.5_±_0.3<br>9.4_±_1.0<br>14.9_±_1.0<br>**3.9**_±_1.1<br>**10.8**_±_1.5<br>**17.8**_±_5.0|



22 

## **E Additional ablation studies** 

## **E.1 General image domain (COCO) results** 

Table 15 presents results on the COCO [101] dataset. We observed that fine-tuning models using limited image-caption pairs degrades performance since the original CLIP already performs well. Therefore, we selected a subset of the “sports” category where the original CLIP performs relatively weakly. Following other setups, we used 10% of the data as labeled data and the rest as unlabeled data. We kept the training configuration consistent with our paper but ran 10 epochs to ensure model convergence. S-CLIP model outperforms others, even in this general image domain. 

Table 15: Image-text retrieval results for the COCO sports category, where the original CLIP performs relatively weakly. S-CLIP also performs well in the general image domain. 

|Method|Image-to-text retrieval|Text-to-image retrieval|
|---|---|---|
|CLIP (original)|40.30|38.17|
|CLIP (fne-tune)|46.59|47.76|
|Hard-PL|48.72|47.12|
|Soft-PL|48.72|47.23|
|S-CLIP (ours)|**49.79**|**48.40**|



## **E.2 Additional pseudo-labeling baselines** 

Table 16 presents a comparison with pseudo-labeling baselines adapted from the sate-of-the-art semisupervised image classification methods. Specifically, we employ the pseudo-labeling techniques of SemPPL [22] and RoPAWS [69]. SemPPL uses the mode of k-NN predictions. In our case, since captions are unique for each image, we assign a uniform probability over k-NN samples, which we refer to as “k-NN” pseudo-labeling. On the other hand, RoPAWS utilizes an analytic fixed-point solution for label propagation, which we refer to as “LabProp” pseudo-labeling. We used these baselines for caption-level pseudo-labeling and compare them with OT. We do not used keyword-level pseudo-labeling for a fair comparison. OT outperforms other pseudo-labeling approaches. 

Table 16: Comparison with pseudo-labeling approaches adopted from the SOTA semi-supervised image classification methods, following the setup of Table 5. OT performs the best. 

|Method|Zero-shot accuracy|Image-text retrieval|
|---|---|---|
|k-NN (k=5)|61.1|8.8|
|LabProp|68.2|9.4|
|OT (ours)|**69.6**|**10.5**|



## **E.3 Qualitative examples of pseudo-labels** 

Figure 7 and Figure 8 provide qualitative examples of caption-level and keyword-level pseudo-labels, respectively. Optimal transport (OT) matches visually and semantically similar labeled images for unlabeled queries, providing meaningful caption-level pseudo-labels. In addition, the nearest labeled images share overlapping keywords, providing meaningful keyword-level pseudo-labels. 

23 


![](raw/papers/2_LLM_NLP/LLM_paper_list/caption/nips_2023_S-CLIP_Semi-supervised_Vision-Language_Learning_using_Few_Specialist_Captions_images/nips_2023_S-CLIP_Semi-supervised_Vision-Language_Learning_using_Few_Specialist_Captions.pdf-0024-00.png)


**----- Start of picture text -----**<br>
Query Nearest Labeled Images (Top-5)<br>around a building with a “many green trees are swimming pool” “a forest surrounded a house” “there are two houses near the road  behind which is a forest” “the river is very long and has a very wide area” many sparse green trees swimming pool is near “a building with a and a road” “many green trees are in a forest near a pond”<br>“this is a port, many  “many boats are near a  “many green trees are  “a lawn is between the  “some boats are in a  “many boats are in a<br>ships, road and land” port with many buildings” near a port with many boats” road and the port” port near a piece of water” port near a wharf”<br>“many buildings and  “a large number of  “the residential buildings  “houses of different  “on both sides of the  “many green trees are<br>dense residential area”green trees are in two sides of a road in a  factories have been built in the factory” on both sides of the road are neatly arranged” colors are lied up orderly along the winding roads” road were densely brown houses” around many buildings with many cars in an industrial area”<br>**----- End of picture text -----**<br>


Figure 7: **Qualitative examples of caption-level pseudo-labels.** Optimal transport (OT) can identify visually and semantically similar images, including those representing forests, ports, and dense buildings. As a result, caption-level pseudo-labels offer accurate semantics for unlabeled images. 


![](raw/papers/2_LLM_NLP/LLM_paper_list/caption/nips_2023_S-CLIP_Semi-supervised_Vision-Language_Learning_using_Few_Specialist_Captions_images/nips_2023_S-CLIP_Semi-supervised_Vision-Language_Learning_using_Few_Specialist_Captions.pdf-0024-02.png)


**----- Start of picture text -----**<br>
“this is a  bridge , several  “a trees and a  park  with many green  pond  is  “many  buildings  are  “four  baseball fields  are  “the  industrial  area is  “many  buildings  and<br>cars, grassland and a  river ” buildings surrounded by many green trees and  and a road” near a football field” stadium  with a  surrounded by many green trees and  meadows ” advanced ide the very modern and  land ” bare  green trees are around a  commercial crossroads in a  area”<br>with some green trees in “a  bridge two sides of it” is on a  river “many green trees and a large surrounded by many  pondbuildings  are in a ” park around a “many green trees are  parking lotstadium  near it” with a  “the is surrounded by lush trees and  baseball fieldbuildings  field ” “many green trees are in an  industrialbuildings  area” and  commercial the “the trapezoid  playground  is next to ”<br>Query<br>Nearest<br>Labeled Images<br>**----- End of picture text -----**<br>


Figure 8: **Qualitative examples of keyword-level pseudo-labels.** The images closest to the query, as found by OT, often share the keywords in their captions (column 1-2). Even in cases where the images do not share all the keywords, they still have some overlapping keywords (column 3-6). 

24 

## **E.4 Training curves** 

Figure 9 presents the training curves showing the loss and evaluation metrics. The validation accuracy for zero-shot classification on RSICD-CLS and image _→_ text retrieval on RS-ALL is reported. The trends are similar in other datasets. Supervised CLIP is prone to overfitting, as indicated by the CLIP validation loss, but this is regularized by Hard-PL and Soft-PL. Our proposed S-CLIP yields the greatest benefits, with superior classification and retrieval performance. 


![](raw/papers/2_LLM_NLP/LLM_paper_list/caption/nips_2023_S-CLIP_Semi-supervised_Vision-Language_Learning_using_Few_Specialist_Captions_images/nips_2023_S-CLIP_Semi-supervised_Vision-Language_Learning_using_Few_Specialist_Captions.pdf-0025-02.png)


**----- Start of picture text -----**<br>
70 6<br>4.75 CLIP (base)<br>Hard-PL<br>4.50 Soft-PL 60 4<br>4.25 S-CLIP (ours) CLIP (base) CLIP (base)<br>4.00 50 Hard-PL Soft-PL 2 HSoft-PLard-PL<br>3.75 S-CLIP (ours) S-CLIP (ours)<br>40 0<br>0 5 10 15 20 25 5 10 15 20 25 5 10 15 20 25<br>(a) CLIP validation loss (b) Zero-shot classification (c) Image-text retrieval<br>**----- End of picture text -----**<br>


Figure 9: Trends in loss and evaluation metrics during training. S-CLIP regularizes overfitting, as shown by the CLIP validation loss. As a result, it performs the best classification and retrieval. 

## **E.5 Locked image or text tuning** 

Table 17 presents the results of fine-tuning after locking (freezing weights) either the image or text encoders, as suggested by LiT [41]. In our method, locking either encoder leads to a degradation in performance. This occurs because both the image and text data from the specialist domain are unseen, requiring the model to learn this novel information. In contrast, in the base CLIP, locking the text encoder yields similar results to having no lock.[6] This is due to the base CLIP suffering from overfitting and inadequately learning the text information. These results provide additional evidence that pseudo-labeling with S-CLIP improves the language understanding of models. 

Table 17: Ablation study on locked image or text encoders. We report the average values of zero-shot accuracy (left) and image _↔_ text retrieval (right). Locking either the image or text encoder degrades performance, as it prevents the model learning new information from specialist domains. 

|CLIP (fne-tune)<br>S-CLIP (ours)|Locked image<br>48.7<br>8.6<br>63.1<br>8.7|Locked text<br>64.8<br>9.4<br>67.1<br>10.2|No lock|
|---|---|---|---|
||||64.7<br>9.3<br>**71.9**<br>**10.6**|



## **E.6 Selection of mini-batches during training** 

Table 18 presents the effect of batch selection by applying different strategies to S-CLIP. Using a smaller batch size (16 _×_ 4 = 64) instead of the original (64 _×_ 4 = 256) significantly decreased the performance. However, collecting similar images helped mitigate this drop. We simplified the process by arranging batches based on their indices, sorted over visual similarities, and used them without shuffling. This method, “sorted batch,” resulted in a reasonable gain. Exploring a more effective active batch sampling strategy would be an interesting future direction. 

Table 18: Ablation study on the mini-batch sampling strategies, following the setup of Table 5. Using a larger and sorted batch improves performance by providing more informative pseudo-labels. 

|Batch size|Zero-shot accuracy|Image-text retrieval|
|---|---|---|
|64 (random batch)|65.8|8.3|
|64 (sorted batch)|67.3|8.9|
|256 (random batch)|**71.9**|**10.6**|



> 6The reported average values may appear similar, but each dataset has distinct values. 

25 

## **E.7 Baselines using keyword information** 

Table 19 presents the results compared to the baselines using keyword information. Specifically, we incorporated two losses: (a) keyword loss on labeled data using multi-label classification, and (b) keyword loss on unlabeled data using our pseudo-labeling approach (nearest image is given by OT). To assess the impact of both losses, we compared CLIP-FT, CLIP-FT+(a), and CLIP-FT+(a)+(b). For the ablation study, we did not apply caption-level pseudo-labeling. Both (a) and (b) contribute to the overall performance, confirming that our proposed pseudo-labeling loss relies not only on keyword information but also on exploiting information from unlabeled data. 

Table 19: Comparison to the baselines using keyword information, following the setup of Table 5. Both keyword information and semi-supervised learning contribute to the final performance. 

|Method|Zero-shot accuracy|Image-text retrieval|
|---|---|---|
|CLIP-FT|64.7|9.3|
|+ keyword info.|67.3|9.7|
|+ semi-supervised|**70.6**|**10.1**|



## **E.8 Keyword statistics in captions** 

We trained our models using captioned datasets, and the class names were derived from the classification counterparts of the dataset. Consequently, captions often contain multiple class names. For instance, 45.8% of captions in the RSICD dataset contain more than two class names. The remaining 50% of captions have only a single class name, which results in our keyword-level loss being reduced to a classification loss. It is worth noting that we developed a method to handle scenarios where class names are unavailable. By utilizing YAKE-200 keywords, we found that 94.8% of captions contain more than two keywords, confirming the necessity of partial label learning. 

## **F Limitations and broader impacts** 

**Limitations.** Our pseudo-labeling approach is effective for zero-shot classification but less effective for image-text retrieval, which requires fine-grained language understanding. This is because the current pseudo-labeling approach assumes that captions in a batch can capture the semantics of unlabeled images, which may not hold true for fine-grained contexts. To address this issue, one can increase the batch size [84] or incorporate a queue of caption embeddings [85]. 

In addition, the current OT formulation of caption-level pseudo-labels assumes uniform sink and source constraints. However, this assumption can be refined in the case of distribution shifts. For example, techniques from robust semi-supervised learning [68, 69] can be utilized to estimate the in-domain-ness and subsequently regularize pseudo-labeling based on this estimation. 

Lastly, keyword-level pseudo-labeling could be enhanced. The current method relies on the exact inclusion of keywords in the caption to form the candidate set. One can relax this criterion to include synonyms or related concepts by incorporating additional inference steps. In addition, the extracted keywords from captions may vary in their levels. Integrating a word hierarchy to account for different levels of keywords would be an interesting future research direction. 

**Broader impacts.** Our paper aims to broaden the applicability of CLIP to specialist domains. However, this broader scope may present challenges when using the model with domains that include harmful content. These challenges arise from the data itself, rather than the model itself. Therefore, it is crucial to have adequate data regularization to effectively address these concerns. 

26 

