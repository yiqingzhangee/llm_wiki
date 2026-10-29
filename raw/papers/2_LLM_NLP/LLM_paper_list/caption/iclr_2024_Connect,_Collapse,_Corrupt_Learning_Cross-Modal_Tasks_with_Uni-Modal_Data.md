Published as a conference paper at ICLR 2024 

# _Connect, Collapse, Corrupt:_ LEARNING CROSS-MODAL TASKS WITH UNI-MODAL DATA 

**Yuhui Zhang** _[∗]_ **, Elaine Sui** _[∗]_ **, Serena Yeung-Levy** Stanford University 

_{_ yuhuiz, esui, syyeung _}_ @cs.stanford.edu 

## ABSTRACT 

Building cross-modal applications is challenging due to limited paired multimodal data. Recent works have shown that leveraging a pre-trained multi-modal contrastive representation space enables cross-modal tasks to be learned from unimodal data. This is based on the assumption that contrastive optimization makes embeddings from different modalities interchangeable. However, this assumption is under-explored due to the poorly understood geometry of the multi-modal contrastive space, where a modality gap exists. In our study, we provide a theoretical explanation of this space’s geometry and introduce a three-step method, _C_[3] (Connect, Collapse, Corrupt), to bridge the modality gap, enhancing the interchangeability of embeddings. Our _C_[3] method significantly improves cross-modal learning from uni-modal data, achieving state-of-the-art results on zero-shot image / audio / video captioning and text-to-image generation. 

## 1 INTRODUCTION 

Building cross-modal applications often face a critical challenge: the scarcity of paired multi-modal data. Largescale multi-modal contrastive learning has emerged as a promising approach to address this gap. Pioneering works like CLIP (Radford et al., 2021) and ImageBind (Girdhar et al., 2023) offer publicly available representation spaces learned from contrasting millions of web-based images and texts. These spaces that align embeddings of concepts across different modalities pave the way for leveraging abundant uni-modal data in cross-modal tasks. For instance, instead of training an image captioning system on image embeddings, one could train it on caption embeddings. Likewise, text-to-image generation models can utilize image embeddings as opposed to text. During crossmodal inference, embeddings from the other modality are simply input into the model (Figure 1). Notably, this ap- 


![](raw/papers/2_LLM_NLP/LLM_paper_list/caption/iclr_2024_Connect,_Collapse,_Corrupt_Learning_Cross-Modal_Tasks_with_Uni-Modal_Data_images/iclr_2024_Connect,_Collapse,_Corrupt_Learning_Cross-Modal_Tasks_with_Uni-Modal_Data.pdf-0001-09.png)


**----- Start of picture text -----**<br>
x<br>② f 𝒳 ② ex<br>② ②<br>g cutecat y<br>① ①<br>cutecat ① f 𝒴 ① ey<br>y<br>① Uni-modal Training<br>② Cross-modal Evaluation<br>**----- End of picture text -----**<br>


Figure 1: Interchangeable use of embeddings enables learning cross-modal tasks with uni-modal data. 

proach has achieved impressive results in image captioning (Li et al., 2023; Tam et al., 2023; Nukrai et al., 2022) and text-to-image generation (Zhou et al., 2022c;a;b), eliminating the need for multimodal paired data. 

This aforementioned process is based on the hypothesis that image embeddings and text embeddings can be used interchangeably in the multi-modal representation space. However, the validity of this assumption is not well understood. Recent works reveal that the resulting geometry from multi-modal contrastive learning is nontrivial — corresponding image and text embeddings do not necessarily collapse to the same points in the space. Instead, there is a significant gap between embeddings from different modalities, potentially hindering the direct interchangeable use of image and text embeddings (Liang et al., 2022; Zhang et al., 2023). The lack of comprehension of the resulting geometry from contrastive learning makes it challenging to design methods to leverage this representation space and learn cross-modal tasks from uni-modal data. 

> _∗_ Equal contribution. Project page available at https://yuhui-zh15.github.io/C3-Website/. 

1 

Published as a conference paper at ICLR 2024 

In this study, we conclude that the geometry of multimodal contrastive representation space as (Figure 2): 


![](raw/papers/2_LLM_NLP/LLM_paper_list/caption/iclr_2024_Connect,_Collapse,_Corrupt_Learning_Cross-Modal_Tasks_with_Uni-Modal_Data_images/iclr_2024_Connect,_Collapse,_Corrupt_Learning_Cross-Modal_Tasks_with_Uni-Modal_Data.pdf-0002-02.png)


where _**e** x_ and _**e** y_ denote the embeddings of paired inputs from different modalities, _**c** ⊥_ is a constant vector representing the modality gap, which is orthogonal to the embedding span of _**e** x_ and _**e** y_ , and _ϵ_ is a random vector representing the alignment noise, which can be approximated by a Gaussian distribution. 


![](raw/papers/2_LLM_NLP/LLM_paper_list/caption/iclr_2024_Connect,_Collapse,_Corrupt_Learning_Cross-Modal_Tasks_with_Uni-Modal_Data_images/iclr_2024_Connect,_Collapse,_Corrupt_Learning_Cross-Modal_Tasks_with_Uni-Modal_Data.pdf-0002-04.png)


**----- Start of picture text -----**<br>
Modality 1  ex 1 ex 3<br>Embedding ex 2 ϵ c ⊥<br>ex  − ey<br>= cute large<br>Modality 2  c ⊥ +  ϵ ey cat1 small  pig ey 3<br>dog<br>Embedding ey 2<br>**----- End of picture text -----**<br>


Figure 2: Geometry of the multi-modal contrastive representation space. 

We provide a theoretical explanation of the above geometry. Specifically, the modality gap emerges during initialization as certain dimensions of image and text embeddings remain approximately constant in the embedding space, and the constants are distinct for images and texts due to separate initializations. During optimization, these constant dimensions lack a gradient that pushes them to align to the same value, leading to the preservation and orthogonality of the modality gap. Meanwhile, the alignment noise arises from the stable region produced by the contrastive loss, where points within a certain range result in a loss near zero, causing the optimization to halt. 

Based on this understanding of the geometry, we propose a simple method, _C_[3] , to improve the interchangeability of embeddings from different modalities, thereby enabling the learning of crossmodal tasks with uni-modal data. _C_[3] consists of three steps: 

1. Connect: Related concepts from different modalities are connected via multi-modal contrastive learning, resulting in a shared representation space that can allow the interchangeability of embeddings from the different modalities. However, a modality gap and alignment noise exist. 

2. Collapse: Based on its geometry, the modality gap can be closed by subtracting the embedding mean from each modality, eliminating the most dominant distributional difference between them. 

3. Corrupt: Additional noise is introduced as regularization during training to improve the performance and robustness of learning cross-modal tasks from uni-modal data, given the alignment noise from multi-modal contrastive learning. 

We demonstrate the effectiveness and broad generalization of _C_[3] on four tasks: image, audio, video captioning and text-to-image generation, and achieve state-of-the-art performance on zero-shot evaluation settings when trained solely on uni-modal data. We also provide a detailed analysis of each component that contributes to performance improvements. Our method sheds new light on the possibility of achieving cross-modal tasks under a low-data regime, and our theoretical analysis provides insight into understanding multi-modal contrastive learning. 

Our contributions are three-fold: 

1. We provide a theoretical explanation of the representation space geometry resulting from multimodal contrastive learning. 

2. Based on this geometry, we propose a simple three-step solution to enhance the interchangeability of embeddings from different modalities, improving cross-modal learning with uni-modal data. 

3. We show the effectiveness of our method on image / audio / video captioning and text-to-image generation, achieving state-of-the-art results. 

## 2 LEARNING CROSS-MODAL TASKS WITH UNI-MODAL DATA 

In this section, we present the general notion of a cross-modal task and how we can leverage unimodal data to learn such tasks. 

## 2.1 CROSS-MODAL TASK FORMULATION 

Cross-modal tasks aim to learn a model that maps inputs from one modality _X_ (e.g., images) to another modality _Y_ (e.g., texts). Given a paired multi-modal dataset _D_ = _{_ ( _x, y_ ) _∈X × Y}_ (e.g., an image-caption dataset), the task can be achieved by minimizing the empirical risk _Ld_ between ˆ the predicted target _y_ = _g_ ( _fX_ ( _x_ )) and the true target _y_ over the dataset _D_ , where _fX_ : _X →_ R _[d]_ is an encoder that maps inputs from _X_ to a _d_ -dimensional representation space, and _g_ : R _[d] →Y_ is a 

2 

Published as a conference paper at ICLR 2024 

decoder that maps outputs from the encoder to _Y_ : 


![](raw/papers/2_LLM_NLP/LLM_paper_list/caption/iclr_2024_Connect,_Collapse,_Corrupt_Learning_Cross-Modal_Tasks_with_Uni-Modal_Data_images/iclr_2024_Connect,_Collapse,_Corrupt_Learning_Cross-Modal_Tasks_with_Uni-Modal_Data.pdf-0003-02.png)


_Ld_ measures the discrepancy between the predicted and true targets, which can be mean squared error (MSE) for images and cross-entropy loss for texts. 

## 2.2 ENABLING CROSS-MODAL TASKS WITH UNI-MODAL DATA 

The need for a multi-modal paired dataset _D_ to learn cross-modal tasks is suboptimal, as collecting such datasets can be expensive and time-consuming. However, if we have a encoder _fY_ : _Y →_ R _[d]_ that maps inputs from _Y_ to the same representation space as the encoder _fX_ , i.e., _∀x, y ∈ D, fX_ ( _x_ ) = _fY_ ( _y_ ), we can train the cross-modal task using a uni-modal dataset _D[′]_ = _{y ∈Y}_ : 


![](raw/papers/2_LLM_NLP/LLM_paper_list/caption/iclr_2024_Connect,_Collapse,_Corrupt_Learning_Cross-Modal_Tasks_with_Uni-Modal_Data_images/iclr_2024_Connect,_Collapse,_Corrupt_Learning_Cross-Modal_Tasks_with_Uni-Modal_Data.pdf-0003-06.png)


Note that _fY_ should be frozen during training to maintain its embedding alignment with _fX_ . When evaluating in a cross-modal setting, we can replace _fX_ with _fY_ , thus enabling cross-modal tasks with only uni-modal training data. 

## 2.3 ESTABLISHING A SHARED REPRESENTATION SPACE 

Recent advances in multi-modal contrastive learning has enabled for encoders that map similar inputs from different modalities to a shared representation space. Specifically, given a large multimodal dataset[1] , _n_ paired inputs are randomly sampled during each iteration and the following objective is optimized (Radford et al., 2021): 


![](raw/papers/2_LLM_NLP/LLM_paper_list/caption/iclr_2024_Connect,_Collapse,_Corrupt_Learning_Cross-Modal_Tasks_with_Uni-Modal_Data_images/iclr_2024_Connect,_Collapse,_Corrupt_Learning_Cross-Modal_Tasks_with_Uni-Modal_Data.pdf-0003-10.png)


where _**e** x_ = _fX_ ( _x_ ), _**e** y_ = _fY_ ( _y_ ), and _τ_ is the temperature hyperparameter. This loss function encourages high similarity between the embeddings of the paired image-texts relative to the similarities between unpaired ones. Intuitively, after optimizing the loss, paired image and text embeddings should collapse to the same point. However, empirical results have shown a significant gap between paired embeddings (Liang et al., 2022), which prevents the direct interchangeable use of image and text embeddings. In the next section, we provide a detailed analysis of the geometry of the multi-modal contrastive representation space. 

## 3 MULTI-MODAL CONTRASTIVE REPRESENTATION SPACE GEOMETRY 

We begin by providing the following proposition that describes the geometry of the multi-modal contrastive representation space. These findings inform our general method of how to adapt unimodal data for cross-modal learning. 

**Proposition 1. (Multi-modal Contrastive Representation Space Geometry)** _Given a paired image x and text y, the relationship between the ℓ_ 2 _-normalized image embedding ex and text embedding ey obtained from multi-modal contrastive learning can be described as:_ 

_**e** x −_ _**e** y_ = _**c** ⊥_ + _**ϵ**_ 

_where_ _**c** ⊥ is a constant vector representing the modality gap and is orthogonal to the image and text embedding span, i.e., ∀_ _**e** x_ 1 _,_ _**e** x_ 2 _,_ _**c** ⊥ ·_ ( _**e** x_ 1 _−_ _**e** x_ 2) = 0 _;_ _**ϵ** ∼N_ ( **0** _, σ_[2] _**I**_ ) _is a random Gaussian vector representing the alignment noise._ 

In the following subsections, we will first explain why the modality gap _**c** ⊥_ exists before and after model optimization. Then, we will introduce the alignment noise _**ϵ**_ and its relation to the temperature parameter in contrastive loss. 

> 1Acquiring domain-specific paired multi-modal datasets can be challenging. However, several works have gathered large-scale noisy image-caption pairs from the web and made pre-trained encoders _fX_ and _fY_ available for direct use. 

3 

Published as a conference paper at ICLR 2024 

## 3.1 MODALITY GAP 

The presence of a modality gap and its orthogonality to the image and text embedding span are due to the joint effect of initialization and optimization. 

**Initialization.** The modality gap exists when randomly initializing multi-modal models, which can be explained by the dimensional collapse (Jing et al., 2022) of the representation space defined below. 

**Definition 1. (Dimensional Collapse of the Representation Space)** _Given a d-dimensional representation space_ R _[d] , we define its effective dimension de as:_ 


![](raw/papers/2_LLM_NLP/LLM_paper_list/caption/iclr_2024_Connect,_Collapse,_Corrupt_Learning_Cross-Modal_Tasks_with_Uni-Modal_Data_images/iclr_2024_Connect,_Collapse,_Corrupt_Learning_Cross-Modal_Tasks_with_Uni-Modal_Data.pdf-0004-05.png)


_where the σi’s are the singular values of the representation covariance matrix in decreasing order, and γ thresholds the minimum variance explained by the de dimensions. Dimensional collapse occurs when de ≪ d._ 

To demonstrate the dimensional collapse phenomenon, we took a randomly initialized CLIP with a _d_ = 512 representation space and fed MS-COCO images and captions as input. We obtained the corresponding image features and text features and performed SVD on the image feature and text feature covariance matrices. The distribution of the singular values is shown in Figure 3, where we can clearly see that the effective dimension of both the image Figure 3: Dimensional collapse of the CLIP and text features is small. Specifically, when representation space. Singular values obtained setting _γ_ = 0 _._ 99, the effective dimension of from SVD reveal that the effective dimension of image embeddings is _de,x_ = 25 and that of text the image and text representation space is much embeddings is _de,y_ = 230. Therefore, the efsmaller than the total number of dimensions. fective dimension of the shared representation space _de ≤ de,x_ + _de,y_ = 255. The equality holds only when all the effective dimensions of the image and text are orthogonal. 

Dimensional collapse indicates that only a small number of dimensions contribute significantly to the variance of the representation space, while the remaining dimensions can be viewed as constant. Suppose the shared representation space has a maximum effective dimension _de_ = 255. This indicates that the image and text embeddings will remain constant in the _dc_ = _d − de_ = 257 ineffective dimensions. As a result, a modality gap exists at the beginning of model optimization, as these _dc_ ineffective dimensions will be inherently different for images and texts, given random initialization. 

To verify this, we synthesize _n_ = 1 _,_ 000 image and text embeddings in _d_ = 512 space. For each image embedding, we initialize the first _de,x_ dimensions, and for each text embedding, the _de,x_ -th to ( _de,x_ + _de,y_ )-th dimensions, by randomly sampling a standard Gaussian distribution _N_ (0 _,_ 1). All the other dimensions are set to a constant value drawn from a _N_ (0 _,_ 1), where the constant for the image and text embeddings are different. Figure 4 (left) illustrates this setup by showing the variance of each dimension. This setup mimics our findings of SVD analysis on CLIP, where we set image embeddings to have _de,x_ effective dimensions and text embeddings to have _de,y_ effective dimensions, and assume these effective dimensions are fully orthogonal. We normalize the embeddings to unit length before performing our analysis. 

We observe a clear modality gap at the beginning: the _ℓ_ 2-distance between the mean of the image embeddings and the mean of the text embeddings is 1 _._ 21. When we only consider the last _dc_ ineffective dimensions, the average distance is 0 _._ 99. More discussion is in Appendix I. 

**Optimization.** The analysis above reveals that the modality gap exists at initialization, and we further analyze why optimizing for the multi-modal contrastive loss fails to close the gap. The following lemma reveals that there is no gradient in the modality gap direction, therefore the gap and its orthogonality will be preserved. 

4 

Published as a conference paper at ICLR 2024 


![](raw/papers/2_LLM_NLP/LLM_paper_list/caption/iclr_2024_Connect,_Collapse,_Corrupt_Learning_Cross-Modal_Tasks_with_Uni-Modal_Data_images/iclr_2024_Connect,_Collapse,_Corrupt_Learning_Cross-Modal_Tasks_with_Uni-Modal_Data.pdf-0005-01.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/caption/iclr_2024_Connect,_Collapse,_Corrupt_Learning_Cross-Modal_Tasks_with_Uni-Modal_Data_images/iclr_2024_Connect,_Collapse,_Corrupt_Learning_Cross-Modal_Tasks_with_Uni-Modal_Data.pdf-0005-02.png)


Figure 4: Variance of each dimension before (left) and after (right) multi-modal contrastive optimization. Our analysis reveals that gradients will only be propagated to effective dimensions and no gradient will be propagated to ineffective dimensions. Therefore, the effective dimensions are aligned while ineffective dimensions remain constant after optimization. 

**Lemma 1. (Gradients in Contrastive Optimization)** _(Proof in Appendix B) With the mild assumption of equal presence of n images and texts with p_ ( _xi_ ) = _p_ ( _yi_ ) = 1 _/n, optimizing the multi-modal contrastive loss L_ = _−_ 2[1] _n_ � _ni_ =1 � log � _nj_ exp=1[exp(] ( _**e** xi_ _**[e]** ·_ _**e**[x] yii[·]_ _**[e]** /[y] τj_ ) _[/τ]_[)][+] exp( _**e** xi ·_ _**e** yi /τ_ ) log � _nj_ =1[exp(] _**[e]**[x] j[·]_ _**[e]**[y] i[/τ]_[)] � _yields the following gradients:_ 


![](raw/papers/2_LLM_NLP/LLM_paper_list/caption/iclr_2024_Connect,_Collapse,_Corrupt_Learning_Cross-Modal_Tasks_with_Uni-Modal_Data_images/iclr_2024_Connect,_Collapse,_Corrupt_Learning_Cross-Modal_Tasks_with_Uni-Modal_Data.pdf-0005-05.png)


Lemma 1 highlights that the gradients of image embeddings during contrastive optimization are fully determined by the text embedding span, and vice versa. Due to the dimensional collapse of the image and text embedding span, the contrastive optimization process fails to propagate gradients in the direction of the ineffective dimensions, resulting in gap preservation and orthogonality to the image and text embedding span after optimization. Lemma 1 also implies that the effective dimensionality of the joint representation space remains unchanged after optimization. 

To empirically verify this, we optimize the multi-modal contrastive loss _L_ on the _n_ = 1 _,_ 000 previously synthesized image and text embeddings. We optimize for 200K steps with a learning rate 0 _._ 1, and CLIP’s initial temperature of _τ_ = 0 _._ 07. We compute the variance of each dimension pre- and post-contrastive optimization and plot the results in Figure 4 (right). From the figure, we can see that the first _de_ = 255 effective dimensions are aligned after optimization while the last _dc_ = 257 ineffective ones remain unchanged. This verifies that no gradient is propagated to the ineffective dimensions, as the variance for these dimensions remains zero after optimization. The modality gap becomes slightly smaller (0.82 compared to 0.99 before optimization) mainly due to _ℓ_ 2-regularization, where changes in the effective dimensions affect changes in the ineffective ones. 

In summary, due to dimensional collapse at model initialization, there are ineffective dimensions where image and text embeddings can be viewed as different constants, resulting in a modality gap at initialization. During optimization, these ineffective dimensions have no gradient update, and thus, the modality gap and its orthogonality are preserved after optimization. 

## 3.2 ALIGNMENT NOISE 

In this section, we explain alignment noise after multi-modal contrastive learning. This noise results from the stable region of contrastive loss demonstrated in the following lemma, where we can consider a single term in the loss given the symmetry of the added terms. 

**Lemma 2. (Stable Region Controlled by Temperature)** _(Proof in Appendix B)_ 

_We consider a single term in the multi-modal contrastive loss Li_ = _−_ log _n_ exp( _**e** xi ·_ _**e** yi /τ_ ) _[We]_ � _j_ =1[exp(] _**[e]**[x] i[·]_ _**[e]**[y] j[/τ]_[)] _[.] define the margin r_ = _**e** xi ·_ _**e** yi −_ max _j_ = _i_ _**e** xi ·_ _**e** yj as the measure of the similarity difference between the matched pair and the hardest negative pair. When r exceeds a threshold given below, Li falls below a small pre-set value δ, where we assume optimization ends:_ 


![](raw/papers/2_LLM_NLP/LLM_paper_list/caption/iclr_2024_Connect,_Collapse,_Corrupt_Learning_Cross-Modal_Tasks_with_Uni-Modal_Data_images/iclr_2024_Connect,_Collapse,_Corrupt_Learning_Cross-Modal_Tasks_with_Uni-Modal_Data.pdf-0005-13.png)


5 

Published as a conference paper at ICLR 2024 

_where o_ ( _τ_ ) _is a monotonically increasing function of temperature τ that satisfies_ 1 _< o_ ( _τ_ ) _< n. Therefore, the required margin r is monotonically increasing with τ ._ 

Lemma 2 suggests that there is a stable region of the contrastive loss. This region can be viewed as a function of temperature where region size increases as the temperature decreases, with the required margin becoming smaller. Within this stable region, the loss falls below the small value _δ_ , indicating that optimization has ended. In the extreme case where _τ →_ 0+, if _r ≥_ 0, the loss will be less than _δ_ ( _δ →_ 0+ for this case). This means that given all the _yj_ , _xi_ can end up within a region instead of a fixed point, resulting in the same zero loss. Figure 5 illustrates the stable region defined by 

|_y_−<br>_y_+<br>_y_−<br>_y_−<br>{<br>_s_+<br>{<br>max_s_−<br>Low<br>Stable region fo|_y_−<br>_y_+<br>_y_−<br>_y_−<br>{<br>_s_+<br>{<br>max_s_−<br>Low<br>Stable region fo|:_r_ =_s_+−max_s_−≥_τ_log_o_(_τ_)/(exp(_δ_) −1) ⇒ℒ≤_δ_<br>r _x_+|
|---|---|---|
||_y_−<br>_y_+<br>_y_−<br>_y_−<br>{<br>_s_+<br>{<br>max_s_−|_y_−<br>_y_+<br>_y_−<br>_y_−<br>_y_−<br>_y_−<br>_y_+<br>_y_−<br>_y_−<br>_y_−<br>_y_−|
|||High<br>Temperature|



Figure 5: Stable region (green area) of contrastive learning controlled by temperature. Within the stable region, the loss falls below a small preset value, indicating that optimization has ended. The region increases as the temperature decreases. 

the margin with regard to the temperature. Therefore, there may be a mismatch between the matched pairs in the representation space, resulting in alignment noise. 

## 3.3 SUMMARY 

|**Statistic**|**Mean Std**|
|---|---|
|_∥_**_d_**(_i_)_∥_2<br>cos(**_d_**(_i_)_,_**_d_**(_j_))<br>cos(**_d_**(_i_)_,_**_r_**(_i_)<br>_j,k_)]<br>E[**_ϵ_**(_i_)<br>_j_ ]_k_<br>cos(**_ϵ_**(_i_)<br>_j ,_**_ϵ_**(_i_)<br>_k_ )|0.83 0.01<br>0.99 0.00<br>0.00 0.06<br>0.00 0.00<br>0.00 0.10|



In Section 3.1 and 3.2, we explain the modality gap _**c** ⊥_ and alignment noise _**ϵ**_ in Proposition 1, respectively. Combining them together, we explain the geometric relation of paired embeddings _**e** x −_ _**e** y_ = _**c** ⊥_ + _**ϵ**_ . 

We verify this geometric relation using CLIP on the MS-COCO imagecos( _**ϵ**_[(] _j[i]_[)] _[,]_ _**[ ϵ]**_[(] _k[i]_[)][)] 0.00 0.10 caption dataset. We randomly group each 100 images into group _i_ , and Table 1: Statistics that define individual gap _**d**_[(] _j[i]_[)] = _**e** x_[(] _[i] j_[)] _[−]_ _**[e]** y_[(] _[i] j_[)][, group gap] _**[ d]**_[(] _[i]_[)][=][ E] _j_[[] _**[d]**_[(] _j[i]_[)][]][, image] reveals representation difference _**r** j,k_[(] _[i]_[)][=] _**[e]**_[(] _x[i] j_[)] _[−]_ _**[e]** x_[(] _[i] k_[)][,][alignment][noise] _**[ϵ]**_[(] _j[i]_[)] = _**d**_[(] _j[i]_[)] _−_ _**d**_[(] _[i]_[)] , where space geometry. _j, k_ are image or text indices. These statistics are computed in Table 1, where the first three statistics show that the modality gap _**c** ⊥_ approximates a constant vector orthogonal to the image and text embedding span, and the last two statistics show that the alignment noise _**ϵ**_ can be viewed as Gaussian noise. We provide a detailed explanation of how to interpret these statistics in Appendix C. 

This geometric analysis in this section serves as the foundation of the approach we introduce in the next section, where we develop a simple method to align the shared representation space and enable learning cross-modal tasks with uni-modal data. 

## 4 CONNECT, COLLAPSE, CORRUPT 

Proposition 1 from Section 3 reveals the geometry of the multi-modal contrastive representation space. Based on this, we propose three steps, connect, collapse, corrupt ( _C_[3] ), to align the representation space, making it possible for embeddings from different modalities to be interchangeably consumed by the decoder and thus enabling learning cross-modal tasks from uni-modal data. 

**Stage 1: Connect.** This stage establishes connections between similar concepts across different modalities. We leverage recent advances in multi-modal contrastive learning (Radford et al., 2021) and use encoders trained with this strategy to build cross-modal models. However, a modality gap and alignment noise exists after multi-modal contrastive learning, as shown in Proposition 1. 

**Stage 2: Collapse.** When directly using embeddings of different modalities as input, there is a drastic degradation in performance due to the modality gap, which causes input distributions to the decoder to differ. To address this issue, we adopt a simple approach proposed by (Zhang et al., 2023) that effectively removes the modality gap. Specifically, during training, in place of _**e** x_ , we feed in _**e**[′] x_[=] _**[e]**[x][−]_[E] _[x]_[[] _**[e]**[x]_[]][ to the decoder,][and during inference with another modality,][in place of] _**e** y_ , we feed in _**e**[′] y_[=] _**[e]**[y][−]_[E] _[y]_[[] _**[e]**[y]_[]][.][This approach collapses the modality gap, eliminating the input] distribution mismatch between the two modalities: 

> _**e**[′] x[−]_ _**[e]**[′] y_[= (] _**[e]**[x][−]_ _**[e]**[y]_[)] _[ −]_[(][E] _[x]_[[] _**[e]**[x]_[]] _[ −]_[E] _[y]_[[] _**[e]**[y]_[]) =] _**[ ϵ]**_ 

6 

Published as a conference paper at ICLR 2024 

**Stage 3: Corrupt.** After removing the modality gap, there is still alignment noise which can be approximated by a _N_ (0 _, σ_[2] _I_ ). During unsupervised training, instead of directly decoding _y_ from _**e**[′] y_[, we add explicit Gaussian noise to the input and decode] _[ y]_[from] _**[ e]**[′′] y_[=] _**[e]**[′] y_[+] _**[ ϵ]**_[ following (Nukrai] et al., 2022; Zhou et al., 2022c). By introducing this noise, the uni-modal and multi-modal training processes become similar, and the learned decoder is more robust and invariant to the small perturbation _N_ (0 _, σ_[2] _I_ ), leading to improved performance. 

Appendix Algorithm 1 summarizes the entire procedure of our proposed method, _C_[3] , that enables learning cross-modal tasks with uni-modal data. 

## 5 RESULTS 

In this section, we verify the effectiveness of our proposed method, _C_[3] , on four tasks: image captioning, audio captioning, video captioning, and text-to-image generation. We show that our method achieves state-of-the-art performances, generalizes to different modalities and contrastive embedding spaces, and is especially useful when multi-modal data are limited. 

## 5.1 IMAGE CAPTIONING 

We use the ClipCap model (Mokady et al., 2021), pairing a frozen CLIP ViT-B/32 image encoder (Radford et al., 2021) with a GPT-2 decoder (Radford et al., 2019). A lightweight MLP mapping network bridges the dimensional gap between CLIP (512- _d_ ) and GPT-2 (768- _d_ ) embeddings and also produces a prefix for GPT-2 caption generation. We train and evaluate on the MSCOCO dataset (Lin et al., 2014) using the standard split (Karpathy & Fei-Fei, 2015), comprising 113K training images and 5K each for validation and testing, with each image having 5 captions. We utilize metrics such as BLEU (Papineni et al., 2002) and ROUGE (Lin, 2004) to evaluate lexical and semantic similarity between generated and human captions. 

We first train our model for text reconstruction using the MS-COCO captions only. Following _C_[3] , we extract the text embedding from the frozen CLIP text encoder and apply the collapse operation (remove pre-computed mean) and corrupt operation (add Gaussian noise). After training, we evaluate our model in the cross-modal setting, replacing the text encoder with the CLIP image encoder and decoding captions from image embeddings. We refer to this evaluation setting as image-free zero-shot evaluation, as images are not seen during training. Additionally, we fine-tune the pretrained model on different amounts of image-caption pairs and evaluate its performance. We refer to this evaluation setting as semi-supervised evaluation. More details can be found in Appendix E. We show image-free zero-shot captioning results in Table 2 and semi-supervised captioning results in Figure 6. 

_C_[3] **achieves state-of-the-art image-free zero-shot captioning results.** As shown in Table 2, our proposed method, _C_[3] , outperforms previous state-of-the-art methods in image-free captioning. Details of the other methods can be found in Appendix Section A. Our ablation analysis demonstrates that both the collapse and corrupt components are crucial for improving cross-modal evaluation performance, as they eliminate the differences between embeddings from different modalities. Notably, the most competitive baseline, CapDec (Nukrai et al., 2022), can be viewed as an ablated version of _C_[3] , but without an analysis of why the corruption works. Our proposed method, however, provides a clear explanation based on a geometric analysis of the multi-modal embedding space, and we further improve performance by introducing a collapse step. Overall, _C_[3] represents a potential standard approach for future works that use a multi-modal contrastive embedding space. 

_C_[3] **is particularly useful in low-data regimes.** In scenarios where multi-modal data is limited, our method remains highly effective. To demonstrate this, we fine-tuned our pre-trained model on 1%, 5%, 25%, and 100% of the MS-COCO training image-text pairs, and compared our performance with a fully supervised baseline (ClipCap) and ablated models (see Figure 6). The results clearly show that _C_[3] outperforms the fully supervised baseline across all metrics, with the most significant improvements seen in low-data regimes where multi-modal paired data are limited. Thus, our method represents a promising solution for achieving cross-modal tasks in such scenarios. 

7 

Published as a conference paper at ICLR 2024 

|**Method**|**Conn. Coll. Corr.**|**BLEU-1**_↑_**BLEU-4**_↑_**METEOR**_↑_**ROUGE-L**_↑_**CIDEr**_↑_**SPICE**_↑_|**BLEU-1**_↑_**BLEU-4**_↑_**METEOR**_↑_**ROUGE-L**_↑_**CIDEr**_↑_**SPICE**_↑_|
|---|---|---|---|
||||Baselines|
|ZeroCap (2022)<br>MAGIC (2022)<br>ESPER (2022)<br>CLIPRe (2023)<br>DeCap (2023)<br>WS-ClipCap (2023)<br>WS-ClipCap-Multi (2023)<br>CapDec (2022)|✗<br>✗<br>✗<br>✗<br>✗<br>✗<br>✗<br>✗<br>✗<br>✓<br>✗<br>✗<br>✓<br>✗<br>✗<br>✓<br>✗<br>✗<br>✓<br>✗<br>✓<br>✓<br>✗<br>✓|49.8<br>7.0<br>15.4<br>31.8<br>34.5<br>-<br>56.8<br>12.9<br>17.4<br>39.9<br>49.3<br>11.3<br>-<br>21.9<br>21.9<br>-<br>78.2<br>-<br>-<br>4.6<br>13.3<br>-<br>25.6<br>9.2<br>-<br>8.9<br>17.5<br>-<br>50.6<br>13.1<br>50.3<br>9.6<br>15.2<br>37.5<br>33.7<br>8.6<br>65.5<br>22.1<br>22.2<br>48.0<br>74.6<br>14.9<br>69.2<br>26.4<br>**25.1**<br>51.8<br>91.8<br>-||
|Ours||||
|_C_1<br>_C_2<br>1<br>_C_2<br>2<br>_C_3|✓<br>✗<br>✗<br>✓<br>✓<br>✗<br>✓<br>✗<br>✓<br>✓<br>✓<br>✓|28.1<br>2.4<br>12.2<br>25.4<br>13.0<br>6.8<br>44.4<br>6.1<br>15.5<br>33.6<br>25.2<br>9.2<br>69.0<br>25.5<br>24.3<br>50.8<br>87.6<br>17.6<br>**71.0**_±_0.1<br>**27.7**_±_0.1<br>25.0_±_0.0<br>**52.0**_±_0.0<br>**93.3**_±_0.3 **18.3**_±_0.1||



Table 2: Image-free image-to-text captioning results. We achieve state-of-the-art zero-shot image captioning and our ablation shows the effectiveness of each component in our method. Baseline results are from the original paper, where some metrics were not reported. Std with 3 runs reported. 


![](raw/papers/2_LLM_NLP/LLM_paper_list/caption/iclr_2024_Connect,_Collapse,_Corrupt_Learning_Cross-Modal_Tasks_with_Uni-Modal_Data_images/iclr_2024_Connect,_Collapse,_Corrupt_Learning_Cross-Modal_Tasks_with_Uni-Modal_Data.pdf-0008-03.png)


**----- Start of picture text -----**<br>
C [3] ClipCap C [1]  (Connect) C1 [2][ (Connect & Collapse)] C2 [2][ (Connect & Corrupt)]<br>32.5 110 20<br>30.0 100<br>27.5 18<br>90<br>25.0<br>80 16<br>22.5<br>70<br>1 5 25 100 1 5 25 100 1 5 25 100<br>% of Fine-tuning Data % of Fine-tuning Data % of Fine-tuning Data<br>BLEU-4 CIDEr SPICE<br>**----- End of picture text -----**<br>


Figure 6: Image-to-text captioning results in the low data regime. When paired multi-modal data are limited, our approach that leverages uni-modal data for pre-training leads to substantial improvements compared to the purely supervised method (ClipCap). 

**Qualitative analysis of collapse and corrupt.** Both the collapse and corrupt components of our method show consistent improvements, but it is not immediately clear how they do so. To address this, we provide qualitative results in Appendix E. We can see that after collapsing, the generated captions are much more natural and fluent, as it removes the most significant distributional difference between image and text embeddings. After corrupting, the model generates more accurate and faithful text descriptions of the image. We hypothesize that adding noise makes the decoder robust to small variations in the embedding space. Therefore when evaluating in cross-modal settings, the alignment noise will not affect the prediction and thus reduces hallucination in the generated caption. 

## 5.2 TEXT-TO-IMAGE GENERATION 

We further apply our method, _C_[3] , to text-to-image generation, the reverse of image captioning. We uti- **Method Conn. Coll. Corr. FID** _↓_ **IS** _↑_ lize LAFITE (Zhou et al., 2022c), which integrates Baselines a frozen CLIP ViT-B/32 (Radford et al., 2021) as DALL-E (2021) ✗ ✗ ✗ 27.5 17.9 CogView (2021) ✗ ✗ ✗ 27.1 18.2 the text encoder and a modified StyleGAN2’s genLAFITE _G_ (2022c) ✓ ✗ ✓ 20.9 24.9 erator (Karras et al., 2020) as the trainable decoder. Ours We use the MS-COCO dataset (Lin et al., 2014) _C_[1] ✓ ✗ ✗ 29.8 22.4 with LAFITE’s official split, including 82K training andWe use40Kthevalidationstandardimages,metrics withsuch 5ascaptionsFID (Fr´echeteach. _CCC_ 12[2][2][3] ✓✓✓ ✓✓✗ ✓✓✗ 21.719.8 **19.6** 24.425.5 **26.0** Inception Distance) (Heusel et al., 2017) and IS (Inception Score) (Salimans et al., 2016) to assess the Table 3: Language-free text-to-image realism of generated images. Similar to the imgeneration results. Our method _C_[3] consisage captioning setting, we first train the model for tently outperforms the baselines. image reconstruction using images only, and then evaluate the model in the cross-modal setting by generating images from text. We show the language-free zero-shot image generation results in Table 3. Similar to image captioning, we find 

8 

Published as a conference paper at ICLR 2024 

||**Image Captioning (MS-COCO 2014)**<br>**BLEU-1**_↑_**METEOR**_↑_<br>**ROUGE-L**_↑_|**Audio Captioning (Clotho 2020)**<br>**BLEU-1**_↑_**METEOR**_↑_**ROUGE-L**_↑_|**Video Captioning (MSR-VTT 2016)**<br>**BLEU-1**_↑_**METEOR**_↑_**ROUGE-L**_↑_|
|---|---|---|---|
|_C_1<br>_C_2<br>1<br>_C_2<br>2<br>_C_3|33.5<br>12.9<br>25.8<br>53.8<br>17.4<br>38.6<br>64.4<br>22.2<br>45.8<br>**74.0**<br>**26.6**<br>**54.0**|21.8<br>17.3<br>18.1<br>26.7<br>20.0<br>21.4<br>27.6<br>20.0<br>20.6<br>**29.5**<br>**20.1**<br>**23.0**|16.7<br>12.6<br>15.2<br>25.1<br>17.8<br>23.1<br>25.2<br>18.1<br>23.8<br>**31.4**<br>**20.0**<br>**26.9**|



Table 4: Generalization of _C_[3] to other modalities, datasets, and contrastive embedding spaces. 

that our method _C_[3] **consistently outperforms the baselines** in terms of FID and IS. Our ablation study further reveals that each component of _C_[3] is useful for improving the performance. More detailed setups and qualitative comparisons can be found in Appendix F. 

## 5.3 GENERALIZATION TO OTHER MODALITIES AND EMBEDDING SPACES 

To verify the generalization of our method to other modalities, datasets, and embedding spaces, we further conduct experiments on zero-shot captioning from image, audio and video using ImageBind (Girdhar et al., 2023) embeddings. We use the same settings as image captioning with CLIP. Results are shown in Table 4. We find that _C_[3] **consistently improves baselines in all the settings** , and **using ImageBind embeddings achieves further improvements in image captioning compared to CLIP embeddings** . 

## 6 RELATED WORKS (FULL VERSION IN APPENDIX A) 

**Multi-modal contrastive learning and resulting geometry.** Multi-modal contrastive learning aims to bridge representations from different modalities, drawing similar concepts closer and distancing dissimilar ones (Radford et al., 2021). CLIP (Radford et al., 2021), ImageBind (Girdhar et al., 2023), and similar models have leveraged extensive multi-modal data to construct such representation spaces, which have been demonstrated to effectively support a range of uni-modal and multi-modal applications (Wortsman et al., 2022; Ramesh et al., 2022; Shen et al., 2022). However, the resulting geometry in the shared representation space, particularly the “modality gap”, where embeddings from different modalities are clearly separate in the shared representation space, remains under-explored (Liang et al., 2022; Zhang et al., 2023). In our work, we unify the observations from Liang et al. (2022) and Zhang et al. (2023), and contribute a formal formulation and theoretical explanation of the unique geometry resulting from multi-modal contrastive learning. 

**Learning cross-modal tasks with uni-modal data.** Given the expense of multi-modal data collection, there is a growing interest in learning cross-modal tasks using uni-modal data. Based on the assumption that contrastive optimization makes representations from different modalities interchangeable, recent works have leveraged these representation spaces and shown great success in building image captioning models with text data only (Tam et al., 2023; Li et al., 2023; Nukrai et al., 2022) and text-to-image generation models with image data only (Zhou et al., 2022c;a;b). Despite these advancements, these methods have noted the intriguing “modality gap” phenomenon and proposed different empirical methods to address this gap, such as (Tam et al., 2023)’s paraphrased decoding, (Li et al., 2023; Zhou et al., 2022a)’s memory retrieval, (Nukrai et al., 2022; Zhou et al., 2022c)’s noise addition, and (Zhou et al., 2022b)’s prior network. In our work, we first provide a theoretical analysis of the multi-modal representation space geometry. Based on the geometry, we propose a simple method that addresses the “modality gap” in a principled manner and ultimately improves performance on cross-modal tasks and outperforms these strategies. 

## 7 CONCLUSION 

In this work, we provide a theoretical explanation of the unique geometry that arises from multimodal contrastive learning. Building upon this, we present a straightforward technique, _C_[3] , which enhances the interchangeability of embeddings between modalities, enabling the creation of crossmodal applications using only uni-modal data. We demonstrate the effectiveness of our approach on image, audio, video captioning and text-to-image generation, achieving state-of-the-art performance on zero-shot evaluation settings when trained solely on uni-modal data. 

9 

Published as a conference paper at ICLR 2024 

## ACKNOWLEDGMENTS 

We thank all the reviewers for their constructive feedback. Serena Yeung-Levy is a Chan Zuckerberg Biohub – San Francisco Investigator. 

## ETHICS STATEMENT 

Drawing from a deep understanding of the multi-modal contrastive representation space, our method enables the effortless creation of multi-modal content from uni-modal data. While this has the potential to revolutionize content generation, the risks associated with generation, such as creating harmful content, persists. We strongly emphasize the importance of ethically and responsibly employing any advancements built upon our approach, ensuring that their impact enhances the betterment of our digital ecosystem. 

## REPRODUCIBILITY STATEMENT 

We provide open-source implementation of our work at https://github.com/ yuhui-zh15/C3. The implementations will enable researchers to reproduce all the experiments described here and run their own analyses on additional multi-modal models and datasets. 

## REFERENCES 

- Peter Anderson, Basura Fernando, Mark Johnson, and Stephen Gould. Spice: Semantic propositional image caption evaluation. In European Conference on Computer Vision (ECCV), 2016. 

- Satanjeev Banerjee and Alon Lavie. Meteor: An automatic metric for mt evaluation with improved correlation with human judgments. In Workshop of Annual Meeting of the Association for Computational Linguistics (ACL Workshop), 2005. 

- Ming Ding, Zhuoyi Yang, Wenyi Hong, Wendi Zheng, Chang Zhou, Da Yin, Junyang Lin, Xu Zou, Zhou Shao, Hongxia Yang, et al. Cogview: Mastering text-to-image generation via transformers. Conference on Neural Information Processing Systems (NeurIPS), 2021. 

- Konstantinos Drossos, Samuel Lipping, and Tuomas Virtanen. Clotho: An audio captioning dataset. In International Conference on Acoustics, Speech and Signal Processing (ICASSP), 2020. 

EleutherAI. Clasp: Contrastive language aminoacid sequence pretraining, 2021. URL https: //github.com/MicPie/clasp. 

- Sabri Eyuboglu, Maya Varma, Khaled Kamal Saab, Jean-Benoit Delbrouck, Christopher LeeMesser, Jared Dunnmon, James Zou, and Christopher Re. Domino: Discovering systematic errors with cross-modal embeddings. In International Conference on Learning Representations (ICLR), 2022. 

- Rohit Girdhar, Alaaeldin El-Nouby, Zhuang Liu, Mannat Singh, Kalyan Vasudev Alwala, Armand Joulin, and Ishan Misra. Imagebind: One embedding space to bind them all. In Conference on Computer Vision and Pattern Recognition (CVPR), 2023. 

- Evan Hernandez, Sarah Schwettmann, David Bau, Teona Bagashvili, Antonio Torralba, and Jacob Andreas. Natural language descriptions of deep visual features. In International Conference on Learning Representations (ICLR), 2022. 

- Martin Heusel, Hubert Ramsauer, Thomas Unterthiner, Bernhard Nessler, and Sepp Hochreiter. Gans trained by a two time-scale update rule converge to a local nash equilibrium. Conference on Neural Information Processing Systems (NeurIPS), 2017. 

- Li Jing, Pascal Vincent, Yann LeCun, and Yuandong Tian. Understanding dimensional collapse in contrastive self-supervised learning. In International Conference on Learning Representations (ICLR), 2022. 

10 

Published as a conference paper at ICLR 2024 

- Andrej Karpathy and Li Fei-Fei. Deep visual-semantic alignments for generating image descriptions. In Conference on Computer Vision and Pattern Recognition (CVPR), 2015. 

- Tero Karras, Samuli Laine, Miika Aittala, Janne Hellsten, Jaakko Lehtinen, and Timo Aila. Analyzing and improving the image quality of StyleGAN. In Conference on Computer Vision and Pattern Recognition (CVPR), 2020. 

- Wei Li, Linchao Zhu, Longyin Wen, and Yi Yang. DECAP: Decoding CLIP latents for zero-shot captioning. In International Conference on Learning Representations (ICLR), 2023. 

- Weixin Liang, Yuhui Zhang, Yongchan Kwon, Serena Yeung, and James Zou. Mind the gap: Understanding the modality gap in multi-modal contrastive representation learning. In Conference on Neural Information Processing Systems (NeurIPS), 2022. 

- Chin-Yew Lin. Rouge: A package for automatic evaluation of summaries. In Workshop of Annual Meeting of the Association for Computational Linguistics (ACL Workshop), 2004. 

- Tsung-Yi Lin, Michael Maire, Serge Belongie, James Hays, Pietro Perona, Deva Ramanan, Piotr Doll´ar, and C Lawrence Zitnick. Microsoft coco: Common objects in context. In European Conference on Computer Vision (ECCV), 2014. 

- Ilya Loshchilov and Frank Hutter. Decoupled weight decay regularization. In International Conference on Learning Representations (ICLR), 2019. 

- Ron Mokady, Amir Hertz, and Amit H Bermano. Clipcap: Clip prefix for image captioning. arXiv preprint arXiv:2111.09734, 2021. 

- David Nukrai, Ron Mokady, and Amir Globerson. Text-only training for image captioning using noise-injected CLIP. In Findings of Conference on Empirical Methods in Natural Language Processing (EMNLP Findings), 2022. 

- Kishore Papineni, Salim Roukos, Todd Ward, and Wei-Jing Zhu. Bleu: a method for automatic evaluation of machine translation. In Annual Meeting of the Association for Computational Linguistics (ACL), 2002. 

- Alec Radford, Jeff Wu, Rewon Child, David Luan, Dario Amodei, and Ilya Sutskever. Language models are unsupervised multitask learners. OpenAI, 2019. 

- Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, et al. Learning transferable visual models from natural language supervision. In International Conference on Machine Learning (ICML), 2021. 

- Aditya Ramesh, Mikhail Pavlov, Gabriel Goh, Scott Gray, Chelsea Voss, Alec Radford, Mark Chen, and Ilya Sutskever. Zero-shot text-to-image generation. In International Conference on Machine Learning (ICML), 2021. 

- Aditya Ramesh, Prafulla Dhariwal, Alex Nichol, Casey Chu, and Mark Chen. Hierarchical textconditional image generation with clip latents. arXiv preprint arXiv:2204.06125, 2022. 

- Tim Salimans, Ian Goodfellow, Wojciech Zaremba, Vicki Cheung, Alec Radford, and Xi Chen. Improved techniques for training gans. Conference on Neural Information Processing Systems (NeurIPS), 2016. 

- Piyush Sharma, Nan Ding, Sebastian Goodman, and Radu Soricut. Conceptual captions: A cleaned, hypernymed, image alt-text dataset for automatic image captioning. In Annual Meeting of the Association for Computational Linguistics (ACL), 2018. 

- Sheng Shen, Liunian Harold Li, Hao Tan, Mohit Bansal, Anna Rohrbach, Kai-Wei Chang, Zhewei Yao, and Kurt Keutzer. How much can CLIP benefit vision-and-language tasks? In International Conference on Learning Representations (ICLR), 2022. 

- Yixuan Su, Tian Lan, Yahui Liu, Fangyu Liu, Dani Yogatama, Yan Wang, Lingpeng Kong, and Nigel Collier. Language models can see: plugging visual controls in text generation. arXiv preprint arXiv:2205.02655, 2022. 

11 

Published as a conference paper at ICLR 2024 

Christian Szegedy, Vincent Vanhoucke, Sergey Ioffe, Jon Shlens, and Zbigniew Wojna. Rethinking the inception architecture for computer vision. In Conference on Computer Vision and Pattern Recognition (CVPR), 2016. 

- Derek Tam, Colin Raffel, and Mohit Bansal. Simple weakly-supervised image captioning via CLIP’s multimodal embeddings. In Workshop of AAAI Conference on Artifcial Intelligence (AAAI Workshop), 2023. 

- Yoad Tewel, Yoav Shalev, Idan Schwartz, and Lior Wolf. Zerocap: Zero-shot image-to-text generation for visual-semantic arithmetic. In Conference on Computer Vision and Pattern Recognition (CVPR), 2022. 

- Ramakrishna Vedantam, C Lawrence Zitnick, and Devi Parikh. Cider: Consensus-based image description evaluation. In Conference on Computer Vision and Pattern Recognition (CVPR), 2015. 

- Joshua Vendrow, Saachi Jain, Logan Engstrom, and Aleksander Madry. Dataset interfaces: Diagnosing model failures using controllable counterfactual generation. arXiv preprint arXiv:2302.07865, 2023. 

- Mitchell Wortsman, Gabriel Ilharco, Jong Wook Kim, Mike Li, Simon Kornblith, Rebecca Roelofs, Raphael Gontijo Lopes, Hannaneh Hajishirzi, Ali Farhadi, Hongseok Namkoong, et al. Robust fine-tuning of zero-shot models. In Conference on Computer Vision and Pattern Recognition (CVPR), 2022. 

- Hu Xu, Gargi Ghosh, Po-Yao Huang, Dmytro Okhonko, Armen Aghajanyan, Florian Metze, Luke Zettlemoyer, and Christoph Feichtenhofer. Videoclip: Contrastive pre-training for zero-shot video-text understanding. In Conference on Empirical Methods in Natural Language Processing (EMNLP), 2021. 

- Jun Xu, Tao Mei, Ting Yao, and Yong Rui. Msr-vtt: A large video description dataset for bridging video and language. In Conference on Computer Vision and Pattern Recognition (CVPR), 2016. 

- Youngjae Yu, Jiwan Chung, Heeseung Yun, Jack Hessel, JaeSung Park, Ximing Lu, Prithviraj Ammanabrolu, Rowan Zellers, Ronan Le Bras, Gunhee Kim, et al. Multimodal knowledge alignment with reinforcement learning. arXiv preprint arXiv:2205.12630, 2022. 

- Lu Yuan, Dongdong Chen, Yi-Ling Chen, Noel Codella, Xiyang Dai, Jianfeng Gao, Houdong Hu, Xuedong Huang, Boxin Li, Chunyuan Li, et al. Florence: A new foundation model for computer vision. arXiv preprint arXiv:2111.11432, 2021. 

- Yuhao Zhang, Hang Jiang, Yasuhide Miura, Christopher D Manning, and Curtis P Langlotz. Contrastive learning of medical visual representations from paired images and text. In Machine Learning for Healthcare (MLHC), 2022. 

- Yuhui Zhang, Jeff Z HaoChen, Shih-Cheng Huang, Kuan-Chieh Wang, James Zou, and Serena Yeung. Diagnosing and rectifying vision models using language. In International Conference on Learning Representations (ICLR), 2023. 

- Yufan Zhou, Chunyuan Li, Changyou Chen, Jianfeng Gao, and Jinhui Xu. Lafite2: Few-shot textto-image generation. arXiv preprint arXiv:2210.14124, 2022a. 

Yufan Zhou, Bingchen Liu, Yizhe Zhu, Xiao Yang, Changyou Chen, and Jinhui Xu. Shifted diffusion for text-to-image generation. arXiv preprint arXiv:2211.15388, 2022b. 

- Yufan Zhou, Ruiyi Zhang, Changyou Chen, Chunyuan Li, Chris Tensmeyer, Tong Yu, Jiuxiang Gu, Jinhui Xu, and Tong Sun. Lafite: Towards language-free training for text-to-image generation. In Conference on Computer Vision and Pattern Recognition (CVPR), June 2022c. 

12 

Published as a conference paper at ICLR 2024 

## OVERVIEW OF APPENDIX 

In this appendix, we supplement related works and additional details of theory and experiments. 

- In Appendix A, we provide detailed related works to contextualize our work in existing works of multi-modal contrastive learning and learning cross-modal tasks with uni-modal data. 

- In Appendix B, we provide proofs of the two lemmas used in the main paper that reveal important properties of multi-modal contrastive learning. 

- In Appendix C, we employ statistical methods to validate the proposed geometric structure on large pre-trained contrastive models. 

- In Appendix D, we summarize the _C_[3] method into an algorithm. 

- In Appendix E, we provide additional experimental details and qualitative results of image captioning. 

- In Appendix F, we provide additional experimental details and qualitative results of textto-image generation. 

- In Appendix G, we explain the importance of aligning embeddings from different modalities. 

- In Appendix H, we offer further discussions about the effectiveness of collapse vs corrupt. 

- In Appendix I, we offer further insights into dimensional collapse and connect it to the cone effect identified by Liang et al. (2022). 

## A RELATED WORKS 

**Multi-modal contrastive learning and resulting geometry.** Multi-modal contrastive learning aims to create a shared representation for different modalities by attracting similar while repelling dissimilar concepts from different modalities during the optimization process (Radford et al., 2021; Zhang et al., 2022; Xu et al., 2021; EleutherAI, 2021; Yuan et al., 2021; Girdhar et al., 2023). Recent works such as CLIP (Radford et al., 2021) have leveraged large-scale image-text data during pre-training, resulting in models that can build strong uni-modal and cross-modal applications. Connecting different modalities can be advantageous given the complementarity of different modalities. For example, connecting vision and language enables zero-shot categorization of visual objects (Radford et al., 2021; Yuan et al., 2021), explanation of model prediction errors or internal representations (Eyuboglu et al., 2022; Hernandez et al., 2022), diagnosis and rectification of vision models using language by composing different concepts (Zhang et al., 2023; Vendrow et al., 2023), and learning cross-modal tasks with uni-modal data. 

However, the geometry resulting from multi-modal contrastive learning has received limited study. Recent work (Liang et al., 2022) found that there is a clear distinction between embeddings from different modalities in the shared representation space, which is referred to as the modality gap. Liang et al. (2022) correctly attributed the gap to the joint effect of model initialization and optimization. However, they did not study the geometric property of the gap and their theory cannot explain the geometry as well. Their theory can only show there is a distributional difference between image and text embeddings. Subsequently, Zhang et al. (2023) studied the geometric properties of the modality gap and found that the gap can be empirically well approximated by a constant vector orthogonal to the image or text embedding subspace. However, this work did not provide an explanation for how this unique geometry arises. 

In our work, we unify the observations from Liang et al. (2022) and Zhang et al. (2023), and contribute a formal formulation and theoretical explanation of the unique geometry resulting from multimodal contrastive learning. Specifically, we show two important factors of the geometry of the multi-modal representation space: modality gap and alignment noise, where the modality gap is a constant vector orthogonal to the image and text embedding span, and the alignment noise can be approximated by a Gaussian distribution. The modality gap arises due to the interplay between the dimensional collapse in model initialization and the resulting collapsed gradient during optimization, whereas the alignment noise is due to the stable region of contrastive loss. These explanations provide a deeper understanding of multi-modal contrastive learning and resulting geometry. 

13 

Published as a conference paper at ICLR 2024 

Moreover, this non-trivial geometry has important implications for building applications on top of a multi-modal representation space. 

**Learning cross-modal tasks with uni-modal data.** Multi-modal data are less abundant and more expensive to collect than uni-modal data, making it ideal to learn cross-modal tasks with uni-modal data. The recent rise of multi-modal contrastive learning provides this possibility, as similar concepts from different modalities establish close connections during the optimization process. Many recent works have leveraged CLIP, which aligns images and text to achieve image-to-text captioning with text-only data and text-to-image generation with image-only data. These methods assume and explore how to interchangeably use image embeddings and text embeddings resulting from multimodal contrastive learning. They achieve better performance than prior methods that did not leverage a multi-modal representation space, outperforming ZeroCap (Tewel et al., 2022), MAGIC (Su et al., 2022), and ESPER (Yu et al., 2022) on captioning, and DALL-E (Ramesh et al., 2021) and CogView (Ding et al., 2021) on image generation. However, due to the peculiar geometry that arises from multi-modal contrastive learning, paired image embeddings and text embeddings are not collapsed to the same point, making it non-trivial to substitute one for the other. Therefore, researchers are proposing different methods to tackle this problem. 

For image-to-text captioning, WS-ClipCap (Li et al., 2023) was the first work to leverage CLIP’s text embeddings to decode text during training, finding that training to decode a paraphrased version of the text from the corresponding CLIP’s text embedding leads to significantly better performance (WS-ClipCap-Multi). That is, for datasets where a single image is paired with multiple captions, it is better to decode a caption by feeding the text embedding obtained from one of the other captions corresponding to the same image. Despite proposing the method, WS-ClipCap failed to explain why it works and that on a high level, this paraphrased decoding can be viewed as adding noise. Decap (Li et al., 2023) maintains a memory of image embeddings and converts image embeddings to text embeddings by using a weighted average of the most similar image embeddings. They then feed the decoder with the converted embedding to decode the same text. While their method outperforms the baseline CLIPRe, which directly retrieves the most similar captions based on image embeddings, it underperforms WS-ClipCap-Multi, despite proposing a more complex method to tackle the modality gap. CapDec (Nukrai et al., 2022) found that directly adding Gaussian noise to the embedding and then feeding it into the decoder leads to comparably better performance. This method corresponds exactly to the corrupt stage in our method. CapDec attributes its success to the hypothesis that adding Gaussian noise closes the modality gap, however, this intuition is inaccurate. 

For text-to-image generation, LAFITE (Zhou et al., 2022c) was the first work to use CLIP with uni-modal images only. Their approach can be seen as the inverse version of CapDec, where they add Gaussian noise to the image embedding and decode the same image. They also incorrectly suggest that adding Gaussian noise closes the modality gap. In contrast, DALLE-2 (Ramesh et al., 2022) uses a complex and heavily trained prior network to convert CLIP text embeddings to CLIP image embeddings but did not provide clear evidence of why the prior network is necessary and its effectiveness. Corgi (Zhou et al., 2022b) modifies the prior network by restricting its starting point, resulting in improved performance. 

Despite the plethora of complex tricks proposed, there is no consensus on the best approach to interchangeably use image and text embeddings, and none of them reveal fundamental reasons or perform systematic ablations. Our work unifies previous approaches by first analyzing the multi-modal representation space and how it arises. Based on the geometric relation, we provide a straightforward method to tackle embedding interchangeability and demonstrate that our method achieves state-ofthe-art results on cross-modal tasks when only training on uni-modal data. 

## B THEORY PROOF 

In this section, we provide proofs of the two lemmas used in the main paper that reveal important properties of multi-modal contrastive learning. 

14 

Published as a conference paper at ICLR 2024 

## B.1 PROOF OF LEMMA 1 

## **Lemma 3. (Gradients in Contrastive Optimization)** 

_With the mild assumption of equal presence of n images and texts with p_ ( _xi_ ) = _p_ ( _yi_ ) = 1 _/n, optimizing the multi-modal contrastive loss L_ = _−_ 2[1] _n_ � _ni_ =1 � log � _nj_ exp=1[exp(] ( _**e** xi_ _**[e]** ·_ _**e**[x] yii[·]_ _**[e]** /[y] τj_ ) _[/τ]_[)][+] exp( _**e** xi ·_ _**e** yi /τ_ ) log � _nj_ =1[exp(] _**[e]**[x] j[·]_ _**[e]**[y] i[/τ]_[)] � _yields the following gradients:_ 


![](raw/papers/2_LLM_NLP/LLM_paper_list/caption/iclr_2024_Connect,_Collapse,_Corrupt_Learning_Cross-Modal_Tasks_with_Uni-Modal_Data_images/iclr_2024_Connect,_Collapse,_Corrupt_Learning_Cross-Modal_Tasks_with_Uni-Modal_Data.pdf-0015-04.png)


_where λ_ = 1 _/_ (2 _nτ_ ) _, αxj_ = _p_ ( _xj|yi_ ) + _p_ ( _yi|xj_ ) _, αyj_ = _p_ ( _yj|xi_ ) + _p_ ( _xi|yj_ ) _, p_ ( _xi|yj_ ) = � _nk_ exp(=1[exp(] _**e** xi_ _**[e]** ·_ _**e**[x] ykj[·]_ _**[e]** /τ[y] j_ ) _[/τ]_[)] _[,][ p]_[(] _[y][i][|][x][j]_[) =] � _nk_ exp(=1[exp(] _**e** yi_ _**[e]** ·_ _**e**[y] xkj[·]_ _**[e]** /τ[x] j_ ) _[/τ]_[)] _[, and][ τ][is temperature.]_ 


![](raw/papers/2_LLM_NLP/LLM_paper_list/caption/iclr_2024_Connect,_Collapse,_Corrupt_Learning_Cross-Modal_Tasks_with_Uni-Modal_Data_images/iclr_2024_Connect,_Collapse,_Corrupt_Learning_Cross-Modal_Tasks_with_Uni-Modal_Data.pdf-0015-06.png)


Then, we can prove the lemma using chain rule: 

_∇_ _**e** xk L_ 


![](raw/papers/2_LLM_NLP/LLM_paper_list/caption/iclr_2024_Connect,_Collapse,_Corrupt_Learning_Cross-Modal_Tasks_with_Uni-Modal_Data_images/iclr_2024_Connect,_Collapse,_Corrupt_Learning_Cross-Modal_Tasks_with_Uni-Modal_Data.pdf-0015-09.png)


Similarly, we can get _∇_ _**e** yk L_ = _λ_[�] _[n] i_ =1 _[α][x] i_[(] _**[e]**[x] i[−]_ _**[e]**[x] k_[)][, which finishes the proof.] 

B.2 PROOF OF LEMMA 2 

**Lemma 4. (Stable Region Controlled by Temperature)** 

_We consider a single term in the multi-modal contrastive loss Li_ = _−_ log _n_ exp( _**e** xi ·_ _**e** yi /τ_ ) _[We]_ � _j_ =1[exp(] _**[e]**[x] i[·]_ _**[e]**[y] j[/τ]_[)] _[.]_ 

15 

Published as a conference paper at ICLR 2024 

_define the margin r_ = _**e** xi ·_ _**e** yi −_ max _j_ = _i_ _**e** xi ·_ _**e** yj as the measure of the similarity difference between the matched pair and the hardest negative pair. When r exceeds a threshold given below, Li falls below a small pre-set value δ, where we assume optimization ends:_ 


![](raw/papers/2_LLM_NLP/LLM_paper_list/caption/iclr_2024_Connect,_Collapse,_Corrupt_Learning_Cross-Modal_Tasks_with_Uni-Modal_Data_images/iclr_2024_Connect,_Collapse,_Corrupt_Learning_Cross-Modal_Tasks_with_Uni-Modal_Data.pdf-0016-02.png)


_where o_ ( _τ_ ) _is a monotonically increasing function of temperature τ that satisfies_ 1 _< o_ ( _τ_ ) _< n. Therefore, the required margin r is monotonically increasing with τ ._ 

_Proof of Lemma 2._ We first prove[�] _i_[exp(] _[t][i][/τ]_[)] _[ ≤][o]_[(] _[τ]_[) exp(max] _[i][ t][i][/τ]_[)][, where] _[ o]_[(] _[τ]_[)][ is a monoton-] ically increasing function of _τ_ that satisfies 1 _< o_ ( _τ_ ) _< n_ . Let us denote _m_ = arg max _i ti_ (no tie), we have: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/caption/iclr_2024_Connect,_Collapse,_Corrupt_Learning_Cross-Modal_Tasks_with_Uni-Modal_Data_images/iclr_2024_Connect,_Collapse,_Corrupt_Learning_Cross-Modal_Tasks_with_Uni-Modal_Data.pdf-0016-05.png)


Let us denote _o[′]_ ( _τ_ ) = 1 +[�] _i_ = _m_[exp((] _[t][i][−][t][m]_[)] _[/τ]_[)][,][since] _[∀][i]_[=] _[m, t][i][−][t][m][<]_[0][,][we][have][0] _[<]_ exp(( _ti − tm_ ) _/τ_ ) _<_ 1, so 0 _<_[�] _i_ = _m_[exp((] _[t][i][−][t][m]_[)] _[/τ]_[)] _[<][n][−]_[1][,][therefore][1] _[<][o][′]_[(] _[τ]_[)] _[<][n]_[.] Moreover, exp(( _ti − tm_ ) _/τ_ ) monotonically increases with _τ_ , therefore _o[′]_ ( _τ_ ) is a monotonically increasing function of _τ_ . We can denote _o_ ( _τ_ ) = _⌈o[′]_ ( _τ_ ) _⌉_ . 

Based on this, we have: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/caption/iclr_2024_Connect,_Collapse,_Corrupt_Learning_Cross-Modal_Tasks_with_Uni-Modal_Data_images/iclr_2024_Connect,_Collapse,_Corrupt_Learning_Cross-Modal_Tasks_with_Uni-Modal_Data.pdf-0016-08.png)


Suppose _Li ≤_ log(1 + _o_ ( _τ_ ) exp( _−r/τ_ )) _≤ δ_ , we have _r ≥ τ_ log exp( _o_ ( _δτ_ )) _−_ 1[,][which][finishes][the] proof. 

## C EMPIRICAL VERIFICATION OF GEOMETRY 

In Section 3, we established a theoretical framework for the multi-modal contrastive representation space geometry. We prove that after contrastive learning, we have _**e** x −_ _**e** y_ = _**c** ⊥_ + _**ϵ**_ , where _**e** x_ and _**e** y_ are _ℓ_ 2-normalized embeddings of a paired image _x_ and text _y_ , _**c** ⊥_ is a constant vector representing the modality gap and is orthogonal to the image and text embedding span, and _**ϵ** ∼N_ ( **0** _, σ_[2] _**I**_ ) is a random Gaussian vector representing the alignment noise. Here we employ statistical methods to validate the proposed geometric structure on large pre-trained contrastive models. Figure 7 visualizes all the definitions introduced in this section. 

To analyze the modality gap _**c** ⊥_ , we need to first find a way to isolate this vector, because it is entangled with alignment noise _**ϵ**_ in _**e** x −_ _**e** y_ . Since _**ϵ**_ is Gaussian, averaging multiple instances of _**e** x −_ _**e** y_ should neutralize this noise, and thus isolating _**c** ⊥_ . Therefore, we randomly group every 100 image-text pairs into a group _i_ . We define embedding difference for pair _j_ in group _i_ as _**d**_[(] _j[i]_[)] = _**e** x_[(] _[i] j_[)] _[−]_ _**[e]**_[(] _y[i] j_[)][.][This][difference][includes][both][the][modality][gap][and][alignment][noise.][By] computing the expected value, _**d**_[(] _[i]_[)] = E _j_ [ _**d**_[(] _j[i]_[)][]][,][we][effectively][eliminate][the][noise][component][and] leave the modality gap. 

To demonstrate that the modality gap is a constant vector, we analyze its magnitude and direction across different groups. The distribution of _∥_ _**d**_[(] _[i]_[)] _∥_ 2 (Figure 8 (Gap Length)) shows that the gap has 

16 

Published as a conference paper at ICLR 2024 

a near constant length of 0.83, while the 0.99 mean of cos( _**d**_[(] _[i]_[)] _,_ _**d**_[(] _[j]_[)] ) (Figure 8 (Gap Direction)) confirms that the gap has the same direction. These findings collectively validate that the modality gap is a constant vector. 

Next, we establish the orthogonality of the modality gap to the embedding spans. By demonstrating its orthogonality to the image embedding span, we implicitly confirm its orthogonality to the text embedding span, given that the modality gap is a constant vector. We define the embedding difference of image _j_ and _k_ as _**r** j,k_[(] _[i]_[)][=] _**[e]**_[(] _x[i] j_[)] _[−]_ _**[e]** x_[(] _[i] k_[)][.][We][observe][the][distribution][of][cos(] _**[d]**_[(] _[i]_[)] _[,]_ _**[ r]** j,k_[(] _[i]_[)][)] shown in Figure 8 (Gap Orthogonality) has zero mean with a small standard deviation 0.06, which demonstrates the gap’s orthogonality to embedding spans. 

Finally, we address the alignment noise. We define _**ϵ**_[(] _j[i]_[)] = _**d**_[(] _j[i]_[)] _−_ _**d**_[(] _[i]_[)] , which eliminates the gap and only leaves the noise. We first demonstrate its zero-mean nature E _i,j_ [ _**ϵ**_[(] _j[i]_[)][]][=] **[0]**[through][the] distribution of each dimension’s mean. We see the mean of each dimension of the noise vectors is bounded between -1e-8 and 1e-8 (Figure 8 (Noise Mean)). Furthermore, we show in Figure 8 (Noise Direction) that the distribution of cos( _**ϵ**_[(] _j[i]_[)] _[,]_ _**[ ϵ]**_[(] _k[i]_[)][)][has][zero][mean][and][small][standard][deviation][0.10,] indicating that noise has random directions as every two noises are likely to be orthogonal. These findings affirm that the alignment noise can be approximated by a Gaussian noise _N_ ( **0** _, σ_[2] _**I**_ ). 

In summary, our empirical analyses support the proposed geometric model of the multi-modal contrastive representation space, with a constant modality gap vector orthogonal to embedding spans and an alignment noise characterizable as zero-mean Gaussian. 

In Figure 9, we show that our analyses above fully apply to other modalities, datasets, and contrastive embedding spaces, such as image-caption (MS-COCO), audio-caption (Clotho), and video-caption (MSR-VTT) on ImageBind embeddings. 


![](raw/papers/2_LLM_NLP/LLM_paper_list/caption/iclr_2024_Connect,_Collapse,_Corrupt_Learning_Cross-Modal_Tasks_with_Uni-Modal_Data_images/iclr_2024_Connect,_Collapse,_Corrupt_Learning_Cross-Modal_Tasks_with_Uni-Modal_Data.pdf-0017-06.png)


**----- Start of picture text -----**<br>
Group 1 Group 2 Group 3<br>Image Modality 𝒳 ex [(1)] 2 ex [(1)] 1 ex [(1)] 4 ex [(1)] 3 ex [(2)] 1 [e] x [(2)] 2 ex [(2)] 3 ex [(2)] 4 ex [(3)] 3 ex [(3)] 2 ex [(3)] 4 ex [(3)] 1<br>ϵ 2 [(1)] ϵ 1 [(1)] ϵ 4 [(1)] ϵ 3 [(1)] r 1,4 [(3)]<br>d 2 [(1)] d 1 [(1)] d 4 [(1)] d 3 [(1)] d [(1)] d [(2)] d [(3)]<br>Text Modality 𝒴 ey [(1)] 1 ey [(1)] 2 ey [(1)] 3 ey [(1)] 4 ey [(2)] 1 ey [(2)] 2 ey [(2)] 3 ey [(2)] 4 ey [(3)] 1 ey [(3)] 2 ey [(3)] 3 ey [(3)] 4<br>ex [(] j [i] [)] embedding of image  j  in group  i d [(] [i] [)] = 𝔼 j [ dj [(] [i] [)][]] modality gap in group  i<br>ey [(] j [i] [)] embedding of text  j  in group  i rj [(] , [i] k [)] [=] [ e] x [(] j [i] [)] [−] [e] x [(] k [i] [)] [embedding difference between image ] [j] [ and image ] [k] [ in group ] [i]<br>dj [(] [i] [)] =  ex [(] j [i] [)] [−] [e] y [(] j [i] [)] embedding difference between image  j  and text  j  in group  i ϵj [(] [i] [)] =  dj [(] [i] [)] − d [(] [i] [)] alignment noise between image  j  and text  j  in group  i<br>**----- End of picture text -----**<br>


Figure 7: Visualization of the multi-modal contrastive representation space and various defnitions introduced in Appendix C. 


![](raw/papers/2_LLM_NLP/LLM_paper_list/caption/iclr_2024_Connect,_Collapse,_Corrupt_Learning_Cross-Modal_Tasks_with_Uni-Modal_Data_images/iclr_2024_Connect,_Collapse,_Corrupt_Learning_Cross-Modal_Tasks_with_Uni-Modal_Data.pdf-0017-08.png)


Figure 8: Empirical verifcation of the multi-modal contrastive representation space geometry. The modality gap approximates a constant vector, indicated by the gap length and direction distributions. The modality gap is orthogonal to the span of embeddings from two modalities, indicated by the gap orthogonality distributions. The alignment noise can be approximated by zero-mean Gaussian, indicated by the noise mean and direction distributions. 

17 

Published as a conference paper at ICLR 2024 


![](raw/papers/2_LLM_NLP/LLM_paper_list/caption/iclr_2024_Connect,_Collapse,_Corrupt_Learning_Cross-Modal_Tasks_with_Uni-Modal_Data_images/iclr_2024_Connect,_Collapse,_Corrupt_Learning_Cross-Modal_Tasks_with_Uni-Modal_Data.pdf-0018-01.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/caption/iclr_2024_Connect,_Collapse,_Corrupt_Learning_Cross-Modal_Tasks_with_Uni-Modal_Data_images/iclr_2024_Connect,_Collapse,_Corrupt_Learning_Cross-Modal_Tasks_with_Uni-Modal_Data.pdf-0018-02.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/caption/iclr_2024_Connect,_Collapse,_Corrupt_Learning_Cross-Modal_Tasks_with_Uni-Modal_Data_images/iclr_2024_Connect,_Collapse,_Corrupt_Learning_Cross-Modal_Tasks_with_Uni-Modal_Data.pdf-0018-03.png)


Figure 9: Generalization of Figure 8 to other modalities and multi-modal contrastive embedding spaces, including image-caption (MS-COCO), audio-caption (Clotho), and video-caption (MSR-VTT) on ImageBind embeddings. 

18 

Published as a conference paper at ICLR 2024 

## D _C_[3] ALGORITHM 

In this section, we summarize the _C_[3] algorithm as follows: 

**Algorithm 1** _C_[3] algorithm 

**Require:** unpaired uni-modal dataset _X_ , _Y_ , fixed encoders _fX_ : _X �→_ R _[d]_ , _fY_ : _Y �→_ R _[d]_ obtained from multi-modal contrastive learning, trainable decoder _g_ : R _[d] �→Y_ , noise level _σ_ ¯ 1 1: _**e** x ←_[�] _x∈X |X|[f][X]_[ (] _[x]_[)] ¯ 1 2: _**e** y ←_[�] _y∈Y |Y|[f][Y]_[(] _[y]_[)] 3: 4: **function** TRAIN( _y_ , _fY_ , _g_ ) 5: _**e** y ← fY_ ( _y_ ) 7:6: _y_ _**ϵ**_ ˆ _←N← g_ ( _**e**_ ( **0** _y −, σ_[2] _**e**_ ¯ _**I** y_ ) + _**ϵ**_ ) 8: **return** _L_ (ˆ _y, y_ ) 9: **end function** 10: 11: **function** TEST( _x_ , _fX_ , _g_ ) 12:13: _**e** y_ ˆ _x←←g_ ( _f_ _**e** Xx_ ( _−x_ ) _**e**_ ¯ _x_ ) 14: **return** _y_ ˆ 15: **end function** 

## E IMAGE CAPTIONING 

In this section, we provide additional experimental details and qualitative results of image captioning. 

E.1 EXPERIMENTAL SETUP 

**Model.** We employ the ClipCap model architecture (Mokady et al., 2021), which utilizes the CLIP ViT-B/32 (Radford et al., 2021) as the image encoder _fX_ , and a mapping network with a pre-trained GPT-2 (Radford et al., 2019) as the decoder _g_ . The mapping network is designed to handle the difference in embedding dimensions between CLIP (512- _d_ ) and GPT-2 (768- _d_ ) and to generate a “prefix” as input to GPT-2. It is implemented as a lightweight MLP with a single hidden layer, which transforms the CLIP embedding into prefix embeddings for GPT-2 to generate captions. During training, we fix the CLIP encoder to maintain the connection between image and text embeddings and only update the decoder, which includes the mapping network and GPT-2. 

**Data.** To train and evaluate our model, we use the MS-COCO image-caption dataset (Lin et al., 2014). We adopt the widely-used data split (Karpathy & Fei-Fei, 2015), which consists of a training set of approximately 113K images, and validation and test sets of 5K images each, where each image has 5 ground truth captions. **Evaluation.** We evaluate our model using various commonly-used image captioning metrics, including BLEU (Papineni et al., 2002), METEOR (Banerjee & Lavie, 2005), ROUGE (Lin, 2004), CIDEr (Vedantam et al., 2015), and SPICE (Anderson et al., 2016). These metrics measure the lexical and semantic similarities between the generated captions and the ground-truth captions. 

**Setup.** We train our model for text reconstruction using the MS-COCO captions only. Following the _C_[3] algorithm, we extract the text embedding from the CLIP text encoder _fY_ and apply the collapse operation (removing the pre-computed mean) and corrupt operation (adding Gaussian noise). After pre-training, we evaluate our model in the cross-modal setting. We replace the encoder with the CLIP image encoder and decode captions from image embeddings. Since no image is seen during pre-training, we refer to this evaluation setting as image-free zero-shot evaluation. Additionally, we fine-tune our model on different amounts of image-caption pairs and evaluate its performance. 

19 

Published as a conference paper at ICLR 2024 

We refer to this evaluation setting as semi-supervised evaluation. During both the pre-training and fine-tuning stages, we train the model for 10 epochs with a batch size of 40, a learning rate of 2e-5, and AdamW (Loshchilov & Hutter, 2019) optimizer with a linear warmup of 5K steps. We use early stopping on the validation set and report the test set performance. 

## E.2 QUALITATIVE EXAMPLES 

We provide qualitative results for image captioning in Appendix Figure 10, which helps us better understand the improvements of each component of _C_[3] . We observe that: 

- _C_[1] generates captions that are **highly repetitive and/or nonsensical** . 

- _C_ 1[2][and] _[ C]_ 2[2][generates captions that are] **[ more fluent, but contain some hallucinations]**[.] 

- _C_[3] generates captions that are **more correct and concise with no extraneous details** . 

## E.3 QUANTITATIVE RESULTS 

|**Method**|**Connect**<br>**Collapse**<br>**Corrupted**|**BLEU-1**<br>**BLEU-4**<br>**METEOR**<br>**ROUGE-L**<br>**CIDEr**<br>**SPICE**|
|---|---|---|
|1% Fine-tuning|||
|ClipCap<br>_C_1<br>_C_2<br>1<br>_C_2<br>2<br>_C_3 (Ours)|-<br>-<br>-<br>✓<br>✗<br>✗<br>✓<br>✓<br>✗<br>✓<br>✗<br>✓<br>✓<br>✓<br>✓|64.5<br>21.2<br>21.2<br>47.6<br>71.1<br>14.7<br>66.5<br>22.9<br>22.6<br>48.7<br>78.5<br>15.9<br>66.8<br>23.0<br>22.6<br>48.7<br>78.5<br>15.9<br>71.7<br>28.8<br>25.3<br>52.7<br>96.3<br>18.3<br>71.9<br>28.9<br>25.3<br>52.7<br>96.4<br>18.3|
|5% Fine-tuning|||
|ClipCap<br>_C_1<br>_C_2<br>1<br>_C_2<br>2<br>_C_3 (Ours)|-<br>-<br>-<br>✓<br>✗<br>✗<br>✓<br>✓<br>✗<br>✓<br>✗<br>✓<br>✓<br>✓<br>✓|70.1<br>26.4<br>24.0<br>51.3<br>89.4<br>17.3<br>70.6<br>27.0<br>24.5<br>51.6<br>91.6<br>17.8<br>70.7<br>27.1<br>24.6<br>51.7<br>92.1<br>17.9<br>72.1<br>29.3<br>25.6<br>53.1<br>98.2<br>18.7<br>72.3<br>29.4<br>25.7<br>53.1<br>98.8<br>18.8|
|25% Fine-tuning|||
|ClipCap<br>_C_1<br>_C_2<br>1<br>_C_2<br>2<br>_C_3 (Ours)|-<br>-<br>-<br>✓<br>✗<br>✗<br>✓<br>✓<br>✗<br>✓<br>✗<br>✓<br>✓<br>✓<br>✓|73.0<br>30.0<br>26.0<br>53.7<br>101.7<br>19.1<br>73.2<br>30.4<br>26.1<br>53.9<br>102.8<br>19.4<br>73.1<br>30.2<br>26.1<br>53.8<br>102.4<br>19.3<br>73.2<br>30.9<br>26.3<br>54.2<br>103.1<br>19.3<br>73.4<br>31.1<br>26.3<br>54.3<br>103.6<br>19.4|
|100% Fine-tuning|||
|ClipCap<br>_C_1<br>_C_2<br>1<br>_C_2<br>2<br>_C_3 (Ours)|-<br>-<br>-<br>✓<br>✗<br>✗<br>✓<br>✓<br>✗<br>✓<br>✗<br>✓<br>✓<br>✓<br>✓|74.0<br>31.5<br>26.8<br>54.7<br>106.6<br>19.9<br>74.6<br>32.4<br>27.2<br>55.2<br>109.7<br>20.3<br>74.6<br>32.3<br>27.1<br>55.2<br>109.2<br>20.1<br>74.1<br>32.2<br>27.1<br>55.2<br>108.8<br>20.2<br>74.0<br>32.2<br>27.1<br>55.2<br>108.9<br>20.2|



Table 5: Image-to-text captioning results in the low data regime. When paired multi-modal data are limited, our approach that leverages uni-modal data for pre-training leads to substantial improvements compared to the purely supervised method (ClipCap). This is the table version used to reproduce Figure 5 in the main paper. Results are averaged over three random seeds for 1-25% fine-tuning to reduce the effect of randomness. 

We include Appendix Table 5, which is the table version to produce Figure 5 in the main text. In this table, we compare the fully supervised ClipCap (Mokady et al., 2021) to an ablation of components in our method _C_[3] , and demonstrate the effectiveness of learning cross-modal tasks with uni-modal data. Results are averaged over three random seeds for 1-25% fine-tuning to reduce the effect of randomness. 

## F TEXT-TO-IMAGE GENERATION 

In this section, we provide additional experimental details and qualitative results of text-to-image generation. 

20 

Published as a conference paper at ICLR 2024 


![](raw/papers/2_LLM_NLP/LLM_paper_list/caption/iclr_2024_Connect,_Collapse,_Corrupt_Learning_Cross-Modal_Tasks_with_Uni-Modal_Data_images/iclr_2024_Connect,_Collapse,_Corrupt_Learning_Cross-Modal_Tasks_with_Uni-Modal_Data.pdf-0021-01.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/caption/iclr_2024_Connect,_Collapse,_Corrupt_Learning_Cross-Modal_Tasks_with_Uni-Modal_Data_images/iclr_2024_Connect,_Collapse,_Corrupt_Learning_Cross-Modal_Tasks_with_Uni-Modal_Data.pdf-0021-02.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/caption/iclr_2024_Connect,_Collapse,_Corrupt_Learning_Cross-Modal_Tasks_with_Uni-Modal_Data_images/iclr_2024_Connect,_Collapse,_Corrupt_Learning_Cross-Modal_Tasks_with_Uni-Modal_Data.pdf-0021-03.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/caption/iclr_2024_Connect,_Collapse,_Corrupt_Learning_Cross-Modal_Tasks_with_Uni-Modal_Data_images/iclr_2024_Connect,_Collapse,_Corrupt_Learning_Cross-Modal_Tasks_with_Uni-Modal_Data.pdf-0021-04.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/caption/iclr_2024_Connect,_Collapse,_Corrupt_Learning_Cross-Modal_Tasks_with_Uni-Modal_Data_images/iclr_2024_Connect,_Collapse,_Corrupt_Learning_Cross-Modal_Tasks_with_Uni-Modal_Data.pdf-0021-05.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/caption/iclr_2024_Connect,_Collapse,_Corrupt_Learning_Cross-Modal_Tasks_with_Uni-Modal_Data_images/iclr_2024_Connect,_Collapse,_Corrupt_Learning_Cross-Modal_Tasks_with_Uni-Modal_Data.pdf-0021-06.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/caption/iclr_2024_Connect,_Collapse,_Corrupt_Learning_Cross-Modal_Tasks_with_Uni-Modal_Data_images/iclr_2024_Connect,_Collapse,_Corrupt_Learning_Cross-Modal_Tasks_with_Uni-Modal_Data.pdf-0021-07.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/caption/iclr_2024_Connect,_Collapse,_Corrupt_Learning_Cross-Modal_Tasks_with_Uni-Modal_Data_images/iclr_2024_Connect,_Collapse,_Corrupt_Learning_Cross-Modal_Tasks_with_Uni-Modal_Data.pdf-0021-08.png)


- _: Pangliclicliclicliclicliclicliclicliclicliclicliclicliclicliclicliclicliclicliclicliclicliclicliclicliclicliclicliclicliclicliclicli...._ 

- _: Two sport skiers in a row look into a lake_ 

- _: A group of people in a canoe in the water._ 

- _: A group of people on a boat in the water._ 

- _Ground truth: a number of people riding on a small boat_ 

- _: A van with a red unicorn horn is driving by a van that has a photo of the photographer's grandmother's van._ 

- _: A very strange horse with a red wig is showing a van in a field of zombies._ 

- _: A horse is standing in a field with a man._ 

- _: A horse is standing in a field next to a house._ 

- _Ground truth: A woman is feeding a horse at a ranch._ 

- _: A road sign that reads \"Bus Stop, Los Angeles, California, USA\"._ 

- _: A road sign that is closed while a man is walking and working._ 

- _: A street sign is shown with a red arrow._ 

- _: A street sign is shown on a road._ 

- _Ground truth: A sign saying "END ROAD WORK" is in front of a truck on a road._ 

- _: A bus stops at the entrance to a European Union bus._ 

- _: A train is parked alongside a passenger's luggage._ 

- _: A man in a suit and tie is standing next to a train._ 

- _: A man is waiting at a train station while a train passes him._ 

- _Ground truth: A man is standing in front of a passing train_ 

_: A little league player catching a ball during the game of baseball at the Baltimore Orioles._ 

_:A kid that is hitting a ball in a baseball game that is on the field._ 

- _: A boy is getting ready to swing at a baseball._ 

- _: A boy is getting ready to hit a baseball while a catcher and umpire watch._ 

- _Ground truth: a baseball player playing catcher with an umpire behind them_ 

_: Dog sniffing the back of the black and white dog in the cobblestone._ 

- _: Dog behind a man's ear in the street._ 

- _: A street sign is shown with a red arrow._ 

- _: A dog looking at a black cat on the street._ 

- _Ground truth: A black cat is scared by a large dog._ 

_: A red and white bus is en route to the International Space Station._ 

- _: A PERSON IS DRIVING THE SEA WAVE_ 

- _: A group of people in a boat with a large white boat in the background._ 

- _: A group of people on a boat in the water._ 

- _Ground truth: A boat in the water filled with people_ 

_: A person and their luggage, resting in the courtyard of the University of Southern California, observing the Eiffel Tower._ 

_: A person on a bench in the midst of the street to take a walk with their library._ 

- _: A man sitting on a bench with a book._ 

- _: A man and woman sitting on a park bench._ 

- _Ground truth: A man and woman sitting on a park bench under a huge tree_ 

Figure 10: Qualitative examples of image-to-text captioning on the MS-COCO dataset. _C_[1] generates captions that are highly repetitive and/or nonsensical. _C_ 1[2][and] _[C]_ 2[2][generates][captions][that][are] more fluent, but contain some hallucinations. _C_[3] generates captions that are more correct and concise with no extraneous details. 

21 

Published as a conference paper at ICLR 2024 

## F.1 EXPERIMENTAL SETUP 

**Model.** We use the LAFITE (Zhou et al., 2022c) model for image generation, which uses the CLIP ViT-B/32 (Radford et al., 2021) as the text encoder _fX_ , and an adapted version of the unconditional generator from StyleGAN2 (Karras et al., 2020) as the decoder _g_ . LAFITE’s generator is adversarially trained alongside a discriminator with an additional contrastive objective to align the generator’s representation space to that of CLIP’s. As with image captioning, during training, we fix the CLIP encoder to maintain the connection between image and text embeddings, and only update the decoder, which in this case, involves updating both the generator and discriminator. 

**Data.** Same as image-to-text captioning, we train and evaluate the model on the MS-COCO dataset (Lin et al., 2014). We use the same pre-processed data and data split provided in the LAFITE official code repository, comprised of 82K training images and 40K validation images with 5 captions per image. 

**Evaluation.** We evaluate our model using the widely-used image generation metric Frechet Inception Distance (FID) (Heusel et al., 2017). This metric measure the realism of generated images by computing feature similarity between those of generated images and those of ground-truth images, where the features are derived from the pre-trained Inception-v3 model (Szegedy et al., 2016). We also report Inception Score (IS) (Salimans et al., 2016), which is similar to FID. Similarly to LAFITE, FID/IS scores are computed based on 50K generated images, using captions that are randomly sampled from the validation set. 

**Setup.** We train the model for at most 750,000 steps with a batch size of 16, a learning rate of 2.5e-3, and Adam optimizer. We initialize our model with the official pre-trained weights from LAFITE (Zhou et al., 2022c) that was trained on CC3M (Sharma et al., 2018). We report the validation set performance with the lowest FID score. 

## F.2 QUALITATIVE EXAMPLES 

We provide qualitative results for image generation in Figure 11, which helps us better understand the improvements of each component of _C_[3] . We observe that: 

- _C_[1] generates the **basic scene** that pertains to the text description, but are less photo-realistic in terms of contrast and color. 

- _C_ 1[2][and] _[ C]_ 2[[2]] 

   - 2[[2]][add more] **[ fine-grained detail]**[ despite still generating some artifacts.] 

- _C_[3] generates **sharper images** that are more **detailed with fewer artifacts** . 

## G WHY ALIGN EMBEDDING FROM DIFFERENT MODALITIES? 

If embeddings from different modalities are aligned, we can train a model on one modality and then infer on another modality, enabling us to build cross-modal applications with only uni-modal data. This is an emerging field that lacks principled approaches to be easily applied without requiring more empirical tuning. 

We added an experiment to explain this further. We train a text generator (image captioner) over CLIP’s text embeddings _x_ . During inference, we manually shift all the _x_ to _x_ + _c_ to simulate the modality gap (a constant vector orthogonal to original spans). We report the captioning performance in terms of gap distance _∥c∥_ in Table 6. We observe substantial performance drops when the gap grows, showing the need to align embeddings. 

## H EFFECTIVENESS OF COLLAPSE VS CORRUPT 

In Table 2, we observe that adding noise (i.e., corrupt, _C_ 2[2][) is more effective than closing the modality] gap (i.e., collapse, _C_ 2[1][).][We hypothesize that the greater effectiveness of] _[ C]_ 2[2][is because] _[ C]_ 2[2][has two] effects: 1) injecting noise in the span to mitigate alignment noise; 2) injecting noise in the modality gap direction to mitigate the model’s sensitivity to this gap. 

22 

Published as a conference paper at ICLR 2024 


![](raw/papers/2_LLM_NLP/LLM_paper_list/caption/iclr_2024_Connect,_Collapse,_Corrupt_Learning_Cross-Modal_Tasks_with_Uni-Modal_Data_images/iclr_2024_Connect,_Collapse,_Corrupt_Learning_Cross-Modal_Tasks_with_Uni-Modal_Data.pdf-0023-01.png)


**----- Start of picture text -----**<br>
A short train traveling through a mountainous landscape<br>Ground Truth<br>A group of cows are grazing in the grass.<br>Ground Truth<br>A small town street with old buildings<br>Ground Truth<br>A street scene with a double-decker bus on the road.<br>Ground Truth<br>A herd of sheep grazing in an open field.<br>Ground Truth<br>Large bus transporting on a open road into a city.<br>Ground Truth<br>**----- End of picture text -----**<br>


Figure 11: Qualitative examples of text-to-image generation on the MS-COCO dataset. _C_[1] generates the basic scene that pertains to the text description, but are less photo-realistic in terms of contrast and color. _C_ 1[2][and] _[ C]_ 2[2][add more fine-grained detail despite still generating some artifacts.] _C_[3] generates sharper images that are more detailed with fewer artifacts. 

23 

Published as a conference paper at ICLR 2024 

|**Gap Distance**_∥c∥_|**ROUGE-1**<br>**ROUGE-L**<br>**METEOR**|
|---|---|
|||
|0.0<br>0.2<br>0.4<br>0.6<br>0.8<br>1.0<br>1.5<br>2.0|85.5<br>81.2<br>83.1<br>76.3<br>71.5<br>73.8<br>55.8<br>50.6<br>52.5<br>40.6<br>35.9<br>36.6<br>30.5<br>26.7<br>26.5<br>24.1<br>21.0<br>20.3<br>16.6<br>14.4<br>13.8<br>13.8<br>12.1<br>11.6|



Table 6: Image captioning performance when trained on embedding _x_ and tested on _x_ + _c_ . This shows the necessity to align embeddings when training a model on one modality and then inferring on another modality. 

We have added an experiment to verify this hypothesis. When adding noise sampled from Gaussian _ϵ ∼N_ distributions,(0 _, σ_[2] _I_ ), wewecomputeremove itsits projectioncomponentoninthethegapmodalitydirectiongapasdirection. _ϵg_ = _∥ϵg·g∥_ Specifically, _∥gg∥_[,][where] _∥_ given _gg∥_[is] the modality gap direction, then we remove this projection to get a new noise _ϵ[′]_ = _ϵ − ϵg_ . We add this new noise during training and name this experiment as _C_ 2[2][(span noise only).] 

From Table 7, we see that adding noise only in the span (i.e., _C_ 2[2][(span][noise][only))][makes][its] performance much worse than adding noise to all the directions (i.e., _C_ 2[2][),][and][its][performance][is] similar to removing the modality gap (i.e., _C_ 1[2][).][Therefore, adding noise (i.e., corrupt) actually leads] to a similar improvement to removing the modality gap (i.e., collapse). The reason for the greater effectiveness of corrupt than collapse is that injecting Gaussian noise not only adds noise in the span but also to the modality gap direction. 

Given the substantial size of the modality gap, adding noise is not enough to fully diminish the gap. Therefore, adding noise and removing the gap (i.e., _C_[3] ) still enhance the overall performance. 

|**Method**|**Conn. Coll. Corr.**|**BLEU-1**_↑_**BLEU-4**_↑_**METEOR**_↑_**ROUGE-L**_↑_**CIDEr**_↑_**SPICE**_↑_|
|---|---|---|
||||
|_C_1<br>_C_2<br>1<br>_C_2<br>2 (span noise only)<br>_C_2<br>2<br>_C_3|✓<br>✗<br>✗<br>✓<br>✓<br>✗<br>✓<br>✗<br>✓<br>✓<br>✗<br>✓<br>✓<br>✓<br>✓|28.1<br>2.4<br>12.2<br>25.4<br>13.0<br>6.8<br>44.4<br>6.1<br>15.5<br>33.6<br>25.2<br>9.2<br>41.2<br>6.2<br>14.9<br>33.6<br>22.8<br>8.3<br>69.0<br>25.5<br>24.3<br>50.8<br>87.6<br>17.6<br>**71.0**<br>**27.7**<br>25.0<br>**52.0**<br>**93.3**<br>**18.3**|



Table 7: Collapse is actually as effective as corrupt. The reason for the greater effectiveness of corrupt than collapse is that injecting Gaussian noise not only adds noise in the span but also to the modality gap direction. 

## I DIMENSIONAL COLLAPSE 

Echoing our main paper, the phenomenon of dimensional collapse (Jing et al., 2022) in randomly initialized image and text encoders creates a modality gap prior to optimization that also persists after optimization. In this section, we offer further insights into dimensional collapse and demonstrate that 1) it is an inherent characteristic of deep neural networks and 2) deeper networks experience more pronounced dimensional collapse. Additionally, we establish a connection between dimensional collapse and the cone effect identified by Liang et al. (2022). 

## I.1 REAL NETWORKS 

In Figure 12, we find that the vision encoder (vision transformer) and text encoder (transformer) of the CLIP have the dimensional collapse phenomenon at initialization. Specifically, the effective dimension of the embeddings generated by the vision and text encoder is both much smaller than the full dimension, which induces a modality gap. Since the gradients of multi-modal contrastive 

24 

Published as a conference paper at ICLR 2024 

learning will only be propagated in the effective dimensions, the modality gap will be preserved after optimization. 


![](raw/papers/2_LLM_NLP/LLM_paper_list/caption/iclr_2024_Connect,_Collapse,_Corrupt_Learning_Cross-Modal_Tasks_with_Uni-Modal_Data_images/iclr_2024_Connect,_Collapse,_Corrupt_Learning_Cross-Modal_Tasks_with_Uni-Modal_Data.pdf-0025-02.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/caption/iclr_2024_Connect,_Collapse,_Corrupt_Learning_Cross-Modal_Tasks_with_Uni-Modal_Data_images/iclr_2024_Connect,_Collapse,_Corrupt_Learning_Cross-Modal_Tasks_with_Uni-Modal_Data.pdf-0025-03.png)


Figure 12: Dimensional collapse of the randomly initialized (left) and pre-trained (right) CLIP representation space. Singular values obtained from SVD reveal that the effective dimension of the image and text representation space is much smaller than the total number of dimensions. 

## I.2 SIMULATION 

To investigate the dimensional collapse phenomenon, we initialize a simple Multi-Layer Perceptron (MLP) with _n_ blocks, where each block contains a linear layer and Rectified Linear Unit (ReLU) activation. We set the dimensionality of the input space and hidden states to 512, and initialize the weights with Xavier uniform distribution and biases to zero. We initialize _N_ = 1000 input embeddings, where each dimension is sampled from a standard normal distribution _N_ (0 _,_ 1). We feed these embeddings into the MLP and extract the features after every 5 layers. We perform Singular Value Decomposition (SVD) on the feature covariance matrix to quantify the degree of dimensional collapse. 

As shown in Figure 13, our experiments reveal two key insights: 

1. Dimensional collapse is an inherent characteristic of deep neural networks and even a simple MLP exhibits this behavior. 

2. Deeper networks are more prone to dimensional collapse, resulting in a smaller effective dimension of the feature space. 


![](raw/papers/2_LLM_NLP/LLM_paper_list/caption/iclr_2024_Connect,_Collapse,_Corrupt_Learning_Cross-Modal_Tasks_with_Uni-Modal_Data_images/iclr_2024_Connect,_Collapse,_Corrupt_Learning_Cross-Modal_Tasks_with_Uni-Modal_Data.pdf-0025-10.png)


Figure 13: Simulation of dimensional collapse on a MLP ( _n_ *(ReLU(Linear))) network. The _x_ - coordinates of the black dots indicate the effective dimensions for the embeddings from each of the layers, respectively, which quantifies the extent of collapse. 

25 

Published as a conference paper at ICLR 2024 

## I.3 CONNECTION TO CONE EFFECT 

The dimensional collapse phenomenon can provide an explanation for the cone effect observed by Liang et al. (2022). They found that the cosine similarities between any two embeddings outputted by a deep neural network were significantly higher than zero. Due to the dimensional collapse, all embeddings share similar values in the ineffective dimensions, leading to high cosine similarities. The stronger the dimensional collapse, the greater the number of ineffective dimensions, and hence, a stronger cone effect. This explanation is illustrated in Figure 14, where collapsing the _z_ -axis of a 3D representation space induces a cone, leading to significantly higher cosine similarities between any two embeddings. 


![](raw/papers/2_LLM_NLP/LLM_paper_list/caption/iclr_2024_Connect,_Collapse,_Corrupt_Learning_Cross-Modal_Tasks_with_Uni-Modal_Data_images/iclr_2024_Connect,_Collapse,_Corrupt_Learning_Cross-Modal_Tasks_with_Uni-Modal_Data.pdf-0026-03.png)


**----- Start of picture text -----**<br>
z  (Collapsed)<br>y<br>x<br>**----- End of picture text -----**<br>


Figure 14: Dimensional collapse explains the cone effect of deep neural networks. When the _z_ -axis of a 3D representation space is collapsed, it results in a cone shape, where the cosine similarities between any two embeddings are significantly higher than zero. 

26 

