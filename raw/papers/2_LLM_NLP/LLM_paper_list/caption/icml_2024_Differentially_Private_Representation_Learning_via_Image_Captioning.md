**Differentially Private Representation Learning via Image Captioning** 

**Tom Sander**[* 1 2] **Yaodong Yu**[* 1 3] **Maziar Sanjabi**[1] **Alain Durmus**[2] **Yi Ma**[3] **Kamalika Chaudhuri**[1 4] **Chuan Guo**[1] 

## **Abstract** 

Differentially private (DP) machine learning is considered the gold-standard solution for training a model from sensitive data while still preserving privacy. However, a major barrier to achieving this ideal is its sub-optimal privacy-accuracy trade-off, which is particularly visible in DP representation learning. Specifically, it has been shown that under modest privacy budgets, most models learn representations that are not significantly better than hand-crafted features. In this work, we show that effective DP representation learning can be done via image captioning and scaling up to internet-scale multimodal datasets. Through a series of engineering tricks, we successfully train a DP image captioner (DP-Cap) on a 233M subset of LAION-2B _from scratch_ using a reasonable amount of computation, and obtaining unprecedented high-quality image features that can be used in a variety of downstream vision and vision-language tasks. For example, under a privacy budget of _ε_ = 8 for the LAION dataset, a linear classifier trained on top of learned DP-Cap features attains 65 _._ 8% accuracy on ImageNet-1K, considerably improving the previous SOTA of 56 _._ 5%. Our work challenges the prevailing sentiment that high-utility DP representation learning cannot be achieved by training from scratch. Code is available at https://github.com/ facebookresearch/dpcap. 

## **1. Introduction** 

Differentially private (DP; Dwork et al. (2006)) model training is an effective strategy for privacy-preserving ML on sensitive data. For most optimization-based learning algo- 

> *Equal contribution 1Meta 2CMAP, Ecole´ polytechnique 

> 3UC Berkeley 4UCSD. Correspondence to: Tom Sander _<_ tomsander@meta.com _>_ , Yaodong Yu _<_ yyu@eecs.berkeley.edu _>_ , Chuan Guo _<_ chuanguo@meta.com _>_ . 

_Proceedings of the 41[st] International Conference on Machine Learning_ , Vienna, Austria. PMLR 235, 2024. Copyright 2024 by the author(s). 

rithms, DP-SGD (Song et al., 2013; Abadi et al., 2016) can be readily applied to obtain models with rigorous DP guarantee. Regrettably, DP training has also been marred by a sub-optimal privacy-utility trade-off, with model utility severely lagging behind their non-private counterpart (Jayaraman & Evans, 2019; Tramer & Boneh, 2020; Kurakin et al., 2022). At the core of this unfavorable trade-off is the _difficulty of DP representation learning_ . Tramer & Boneh (2020) showed that when DP training from scratch under a low-to-moderate privacy budget, most models do not learn useful representations, with the quality of learned representations worse than even handcrafted features. These observations naturally lead to the research question: _“How does one learn useful representations with DP training?”_ 

One plausible reason for the failure of prior attempts at DP representation learning is the lack of training data. Indeed, DP limits the information content of each training sample via the privacy budget _ε_ , inducing a privacy-accuracysample size tradeoff; thus a substantially larger training dataset is required to extract the same amount of information to train the model. As the vast majority of prior work only utilize small to moderate scale classification datasets such as CIFAR-10 (Krizhevsky et al., 2009) and ImageNet (Deng et al., 2009b), the amount of training data is simply insufficient for learning high-quality representations under DP (Tramer & Boneh, 2020). Yu et al. (2023) made partial progress towards this through _self-supervised learning_ (SSL) on _internet-scale data_ . By training a masked autoencoder (MAE; He et al. (2022)) using DP-SGD on a 233M subset of the LAION-2B dataset (Schuhmann et al., 2022), the model learned image representations that are on-par with non-private AlexNet (Krizhevsky et al., 2012) trained on ImageNet—the first deep learning model to outperform handcrafted features and a major cornerstone for representation learning. However, the MAE objective also promotes the model to learn extraneous details in the image that may not be helpful for obtaining generalizable representations, severely limiting the potential of this approach for DP. 

We adopt a different approach of DP training via image captioning on internet-scale multimodal datasets. The reason is twofold: **1.** Text caption provides a concise summary of the training image and serves as better supervision compared to image-only SSL (Tschannen et al., 2023). Under the constraint on information content from DP, we hypothe- 

1 

**Differentially Private Representation Learning via Image Captioning** 


![](raw/papers/2_LLM_NLP/LLM_paper_list/caption/icml_2024_Differentially_Private_Representation_Learning_via_Image_Captioning_images/icml_2024_Differentially_Private_Representation_Learning_via_Image_Captioning.pdf-0002-01.png)


**----- Start of picture text -----**<br>
(a). Few-shot learning on ImageNet (b). ARO Evaluation<br>Seagull standing on the beach A brown bear is sitting Cat sitting in toilet Swans swimming in the<br>on the grass water<br>(c). Captions generated by DP-Cap<br>**----- End of picture text -----**<br>


Figure 1: (a) Few-shot ImageNet-1K linear probe accuracy comparison between DP-Cap (ours) and ViP (Yu et al., 2023) (previous SOTA). DP-Cap learns better image representations using the same training data and privacy budget, and considerably surpasses synthetic initialization ( _syn_ ). The privacy budget _ε_ is for the LAION dataset, and the linear classifiers are trained without DP. (b) Compositional understanding evaluation on the ARO benchmark (Yuksekgonul et al., 2022). DP-Cap performance is close to non-private Cap and outperforms non-private CLIP. (c) Captions generated by DP-Cap on images from the MS-COCO 2017 (Lin et al., 2015) test set. 

size that it provides substantially more efficient information extraction under DP training. **2.** Image captioning is wellaligned with the prerequisites of DP-SGD such as having an instance-separable loss. We apply this method on a 233M subset of LAION-2B to train a DP image captioning model (DP-Cap), whose learned representations surpass previous SOTA—ViP (Yu et al., 2023)—by a large margin. As depicted in Figure 1(a), our model trained with a privacy budget of _ε_ = 8 shows substantial improvements on downstream tasks compared to ViP, both trained on the same dataset. To achieve this, we also made crucial improvements to the efficiency of the DP training pipeline, **reducing the compute cost by close to 5** _×_ on the largest model. 

The image representations learned by DP-Cap also exhibit strong performance for multimodal tasks that require alignment of image and text features, the first occurrence for models trained from scratch with DP; see Figure 1(b). As a qualitative evaluation, we also use the trained DP-Cap model to caption several images from the MS-COCO 2017 (Lin et al., 2015) test set in Figure 1(c) and Appendix B.3. The resulting captions are grammatically correct and semantically coherent, while (close to) accurately describing contents of the image; this is interesting because our model has only 

been exposed to language supervision from LAION, which are far from being flawless. Our results suggest that DP training on internet-scale multimodal datasets can be a viable approach for obtaining high-utility learned representations. 

## **2. Background and Related Work** 

**Vision-language pre-training.** Many modern ML datasets such as Conceptual Captions (Changpinyo et al., 2021), LAION (Schuhmann et al., 2021) and DataComp (Gadre et al., 2023) consist of aligned image-text pairs where the image and text contain roughly similar semantic information. One can leverage the aligned nature of the training data to pre-train _vision-language models_ (VLMs) that connect the two modalities, whose representations perform more general multi-modal tasks. Contrastive learning-based techniques such as CLIP (Radford et al., 2021) and BLIP (Li et al., 2022a) are also applicable for pre-training VLMs. Doing so not only learns high-quality image and text representations but also introduces new multi-modal capabilities such as cross-modal retrieval and zero-shot prediction (Radford et al., 2021). Recent work by (Tschannen et al., 2023) shows an image captioning approach (predicting text captions from 

2 

**Differentially Private Representation Learning via Image Captioning** 

images) is a viable alternative to contrastive learning and can lead to models with robust performance. 

**Differential privacy (Dwork et al., 2006).** In the following, we denote by _M_ a randomized learning algorithm, which takes a dataset _D_ containing _N_ samples and produces a machine learning model _**θ**_ through the process _M_ ( _D_ ). A randomized mechanism _M_ is ( _ε, δ_ )-DP if, for any two adjacent datasets _D_ and _D[′]_ differing by a single sample, and for any subset _O ⊂_ **Im** ( _M_ ): 


![](raw/papers/2_LLM_NLP/LLM_paper_list/caption/icml_2024_Differentially_Private_Representation_Learning_via_Image_Captioning_images/icml_2024_Differentially_Private_Representation_Learning_via_Image_Captioning.pdf-0003-03.png)


We adopt the leave-one-out notion of adjacency in this work, _i.e._ , _D_ = _D[′] ∪{_ **x** _}_ for some sample **x** or vice versa. DP bounds the extent to which any potential adversary can infer information about the dataset _D_ after observing the algorithm’s output. In the context of ML, this implies that if we obtain the model _**θ**_ through a DP training algorithm _M_ then its training data is provably difficult to recover or infer (Balle et al., 2022; Guo et al., 2022; 2023). 

**DP-SGD** (Song et al., 2013; Abadi et al., 2016) is predominant differentially private algorithm for training deep neural networks (DNNs). At each gradient step _k_ , a batch _Bk_ is sampled where each example from the training data is chosen randomly with probability _q_ = _B/N_ , where _B_ represents the average batch size. For _C >_ 0, define the clipping function for any _X ∈_ R _[d]_ by clip _C_ ( _X_ ) = _C · X/∥X∥_ if _∥X∥≥ C_ and clip _C_ ( _X_ ) = _X_ otherwise. Given model pa� rameters _**θ** k_ , DP-SGD defines the update _**θ** k_ +1 = _**θ** k −ηk_ **g** _k_ where _ηk_ is the step size and � **g** _k_ is given by: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/caption/icml_2024_Differentially_Private_Representation_Learning_via_Image_Captioning_images/icml_2024_Differentially_Private_Representation_Learning_via_Image_Captioning.pdf-0003-06.png)


where _ℓi_ ( _**θ**_ ) is the per-sample loss function evaluated at sample **x** _i_ . We also use the term “DP-SGD” loosely to refer to the category of gradient-based optimization algorithms that operate on the noisy gradient, _e.g._ , Adam (Kingma & Ba, 2014). The privacy analysis of DP-SGD relies on composition of multiple steps. One particularly powerful analysis framework amenable to such compositions relies on a variant of DP called Renyi differential privacy (RDP) (´ Mironov, 2017). An advantage of RDP is its additive composition property, where the privacy guarantees of a sequence of mechanisms can be combined with amplification by subsampling (Wang et al., 2019) and then translated to ( _ε, δ_ )- DP (Balle et al., 2020; Gopi et al., 2021). In this work, we adopt this accounting technique. 

**Scaling up DP-SGD training.** DP training is a theoretically and empirically proven remedy against unintended training data memorization. Even models with large _ε_ ( _e.g._ , _ε_ = 100) can empirically defend against privacy attacks (Carlini et al., 2021; Guo et al., 2023). Despite its great appeal, DP training 

also carries a significant drawback of large drop in model utility (Abadi et al., 2016; Tramer & Boneh, 2020). For example, the SOTA performance on ImageNet when training from scratch with a DP guarantee of _ε_ = 8 is 39.2% (Sander et al., 2023); in comparison, the non-private performance on ImageNet when training from scratch can reach 88% (Touvron et al., 2022) or higher. This degradation in model utility also translates to poorly learned representations, as Tramer & Boneh (2020) showed that even handcrafted features can rival ones learned through DP training. 

Yu et al. (2023) made the first step towards obtaining highutility learned representations through scaling DP training. They proposed self-supervised learning (SSL) on internetscale data as a solution for the privacy-utility trade-off in DP representation learning. Among the numerous SSL algorithms, the authors observed that the reconstruction-based approach of masked autoencoder (MAE; He et al. (2022)) is compatible with the requirements of DP-SGD. By leveraging weight initialization through synthetic pre-training, the authors were able to obtain high-utility learned representations at a strict privacy budget of _ε_ = 8. Compared to ViP (Yu et al., 2023), we demonstrate that the image captioning approach (see Section 3.1) learns much better image representations by utilizing the additional text supervision. 

## **3. Approach** 

We describe in detail our approach of DP representation learning via image captioning. We first argue why image captioning is intuitively a suitable objective for obtaining better image representations via DP-SGD training (section 3.1). Then, we elucidate the technical challenges that we resolved to make DP training viable and effective for image captioning (section 3.2). 

## **3.1. DP Representation Learning via Image Captioning** 

**Why is vision-language pre-training suitable?** Given image-text aligned datasets, prior works (Radford et al., 2021; Li et al., 2022a; Tschannen et al., 2023) showed that pre-training using language supervision is an appealing option for non-private representation learning. We hypothesize that this is true for DP representation learning as well. Compared to image-only supervision, language supervision contains a more condensed summary of the image content, allowing the model to ignore irrelevant details such as background and focus on objects of interest and their relationships. This is especially helpful for DP since the model needs to extract as much useful information as possible from each sample given the privacy budget _ε_ . Captioning could thus enhance the privacy-utility-sample size trade-off in DP, considering it requires less information per sample. 

In addition, we show that vision-language pre-training supports a very large batch size, much larger than what is typi- 

3 

**Differentially Private Representation Learning via Image Captioning** 

cally used in image-only pre-training (Radford et al., 2021; Li et al., 2022a; Yu et al., 2022). This subtle aspect is in fact crucial for reducing the effective noise in DP-SGD (Li et al., 2021), which allows the model parameters to converge to a stable solution with lower training loss (see Section 3.2). 

**Vision-language pre-training via image captioning.** Perhaps the most popular approach for vision-language pretraining is contrastive language image pre-training (CLIP; Radford et al. (2021)) as well as its variants (Mu et al., 2022; Li et al., 2023). However, the contrastive loss used in these methods is not an additive function over the samples, _i.e._ , it cannot be written in the form[�] _i[ℓ][i]_[, where] _[ ℓ][i]_[depends only] on the _i_ -th sample. Thus, DP-SGD ( _cf._ equation 2) cannot be directly applied. 

Unlike contrastive learning, the image captioning approach (Sariyildiz et al., 2020; Desai & Johnson, 2021; Tschannen et al., 2023) aligns well with DP-SGD training. Specifically, an image captioner is trained to predict captions based on their corresponding images. The training objective of the image captioner for one image-text pair [ **x**[img] _,_ **z**[text] ] is to minimize over _**θ**_ := _{_ _**θ**_ enc _,_ _**θ**_ dec _}_ the following loss: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/caption/icml_2024_Differentially_Private_Representation_Learning_via_Image_Captioning_images/icml_2024_Differentially_Private_Representation_Learning_via_Image_Captioning.pdf-0004-04.png)


We use **z**[text] denotes the caption token sequence _{z_ 1[text] _[, . . . , z] T_[text] _[}]_[,][and][we][let] _**[z]**_ 1:0[text] = _∅_ . The image captioner consists of two parts: the image encoder _ψ_ ( _·_ ; _**θ**_ enc) and the text decoder _φ_ ( _·_ ; _**θ**_ dec). The rationale behind the design of equation 3 is that the image encoder maps the input image **x**[img] to an embedding vector, and the text decoder takes the image embedding _ψ_ ( **x**[img] ; _**θ**_ enc) and the first _t_ caption tokens _{z_ 1[text] _[, . . . , z] t_[text] _}_ as inputs and predicts the next caption token _zt_[text] +1[.][Both the encoder and decoder are] trained to maximize the log-likelihood of the correct next token. Equation 3 corresponds to the loss function for an image-text pair [ **x**[img] _,_ **z**[text] ]; summing over all the samples in a batch gives the complete empirical loss in an additive form, which is directly compatible with DP-SGD. 

**Interpretation of DP bound for image-text pairs.** _ε_ -DP bounds the amount of information extracted from each training sample by _ε_ . For DP-Cap, each sample is made of _{_ image + caption _}_ , while ViP (Yu et al., 2023) utilizes _{_ image _}_ only. Consequently, DP-Cap inherently offers an equivalent or better privacy guarantee for each image. To better understand this, we can consider membership inference attacks (Shokri et al., 2017) as an example. Suppose _ε_ -DP upper bounds the success rate of the membership inference attack (when given the image-text pair) against DP-Cap as less or equal to _p_ . Then the MIA success rate when given only the image can be at most _p_ since the attacker has strictly 


![](raw/papers/2_LLM_NLP/LLM_paper_list/caption/icml_2024_Differentially_Private_Representation_Learning_via_Image_Captioning_images/icml_2024_Differentially_Private_Representation_Learning_via_Image_Captioning.pdf-0004-07.png)


**----- Start of picture text -----**<br>
0.8<br>DP-Cap (random init)<br>0.7 Syn Init only<br>0.63 DP-Cap (syn init)<br>0.6<br>0.54<br>0.52<br>0.5 0.47<br>0.44 0.43 0.45<br>0.4 0.34 0.36 0.34<br>0.3 0.29<br>0.20<br>0.2<br>0.1<br>0.0<br>ImageNet Places365 Places205 iNaturalist<br>Accuracy<br>**----- End of picture text -----**<br>


Figure 2: Impact of synthetic initialization on DP-Cap. The learned image representation benefits substantially from initializing on the Shaders21K dataset. The gap between DP-Cap (random init) and DP-Cap (syn init) can be as large as 24% when evaluated using linear probing on ImageNet. 

less information. This is exactly the upper bound for the success rate of a membership inference attack against ViP. 

## **3.2. Strategy for Effective DP Training** 

Although image captioning has demonstrated impressive representation learning capabilities in the non-private regime, adapting it to DP training requires careful considerations. To obtain a useful pre-trained model, one needs to train for a sufficient number of steps under a low effective noise, both of which are at odds with obtaining a strong privacy guarantee. We detail the strategy we used to handle this trade-off when training the image captioner. 

**Sufficient number of training steps.** We address this challenge via synthetic pre-training. Image representations learned by _DP-Cap (random init)_ outperform those of _ViP (random init)_ as evidenced in table 9 in Appendix B. Interestingly, Yu et al. (2023) and Tang et al. (2024) have shown that synthetic images consisting of only textures can provide a good initialization for training DP models without any privacy risk. With this initialization in Yu et al. (2023), the reconstruction-based model can then focus on learning dataset-specific properties rather than low-level image properties such as edge detectors, therefore expending privacy budget in a more optimal manner. We adapt this technique of pre-training on the Shaders21K dataset (Baradad et al., 2022) for initialization and observe that it is even more effective with DP-Cap compared to ViP.[1] As shown in Fig. 2, our _DP-Cap (syn init)_ improves over _DP-Cap (random init)_ by more than 24% on ImageNet-1k linear probing. It also improves over synthetic initialization alone ( _Syn-init_ ) by more than 14% on ImageNet-1k linear probing, whereas the gain in ViP is smaller than 6%. 

> 1For the synthetic pre-training on Shders21k in DP-Cap, we modify the training of image captioning model such that the model is trained to predict missing image patches, which is similar to MAE (He et al., 2022). More details can be found in Appendix A.1. 

4 

**Differentially Private Representation Learning via Image Captioning** 


![](raw/papers/2_LLM_NLP/LLM_paper_list/caption/icml_2024_Differentially_Private_Representation_Learning_via_Image_Captioning_images/icml_2024_Differentially_Private_Representation_Learning_via_Image_Captioning.pdf-0005-01.png)


**----- Start of picture text -----**<br>
6.0 0.35<br>B=40,960 Reconstruction-based (ViP)<br>Captioning-based (DP-Cap)<br>5.5 B=81,920 0.30 Non-private MAE<br>B=163,840<br>5.0 0.25<br>4.5 0.20<br>4.0 0.15<br>3.5<br>(98k, 0.47) (200k, 0.513) (400k, 0.565) (1.3M, 0.728)<br>0 1000 2000 3000 4000 5000<br>(B, ) for = 8<br>DP-SGD step<br>(a) DP-Cap scales gracefully to extreme batch sizes (b) Performance improves consistently with larger batch size<br>Cross Entropy Loss ImageNet 10-shot<br>**----- End of picture text -----**<br>


Figure 3: (a) We fix the effective noise _σ/B_ = 5 _._ 6 _×_ 10 _[−]_[7] (corresponding to our (B, _σ_ ) = (1.3M, 0.728)) and show that the loss is remarkably consistent across different batch sizes, allowing us to effectively scale up batch size to improve the SNR. (b) Performance from 4 sets of parameters that provide _ε_ = 8, with constant number of steps 5708. From batch size 98k (used in ViP (Yu et al., 2023)), to our 1.3M batch size. In contrast to ViP, DP-Cap successfully leverages the better SNR and learns features that achieve substantially better 10-shot accuracy on ImageNet even compared to a **non-private** MAE (He et al., 2022) trained on the same dataset (see Appendix A.1). 

**Using extreme batch sizes to reduce effective noise.** Li et al. (2021) first showed that increasing the batch size in DP-SGD often improves the privacy-utility trade-off. This is because the effective noise added to the average gradient has magnitude _σ/B_ ( _cf._ equation 2), and that increasing _B_ , rather than decreasing _σ_ , results in better privacy guarantees according to conventional accounting techniques (Bun & Steinke, 2016; Dwork & Rothblum, 2016; Mironov et al., 2019; Sander et al., 2023). However, in supervised learning, increasing the batch size beyond a certain point may degrade performance in both private and non-private settings (Sander et al., 2024). Specifically, Sander et al. (2023) observed that when training a classifier from scratch with DP-SGD on ImageNet, at a fixed number of steps _S_ and fixed effective noise _σ/B_ , the performance decreases significantly when the batch size becomes too large: for _B ∈_ [128 _,_ 16384], a drop of 10% in top-1 accuracy was observed. 

Intriguingly, we find that vision-language pre-training on internet-scale datasets can tolerate _extreme_ batch sizes, _e.g. B_ = 1M. In Figure 3(a), we compare the loss behaviors when scaling the batch size for DP-Cap. We fix the effective noise _σ/B_ while varying the batch size. In stark contrast to the previous observation from (Sander et al., 2023), the loss trajectory is identical across different batch sizes. With this observation, we are able to successfully scale up the batch size for DP-Cap to as large as _B_ = 1 _._ 3M, achieving an effective noise of 5 _._ 6 _×_ 10 _[−]_[7] , almost 10 times smaller than the effective noise of ViP in Yu et al. (2023). Training DP-Cap under such a small effective noise allows it to ex- 

tract information from the training dataset more efficiently under the DP constraint. In Figure 3, we show that with _B_ = 1 _._ 3M, the representation learned by DP-Cap even outperforms _non-private_ MAE trained on the same dataset. From these results, we find that the DP-Cap training can handle extreme large batches much more effectively than ViP in the context of differentially private training. This advantage is likely independent of image captioning models’s inherent superiority over MAE. 

**Improving training pipeline efficiency.** DP training is known to be computationally inefficient due to factors such as per-sample gradient computation (Lee & Kifer, 2020; Li et al., 2021). Training on internet-scale datasets using extreme batch sizes further complicates this issue. For example, a naive implementation of DP-Cap with per-sample gradient computation using functorch would take approximately 61 days(!) on 128 NVIDIA V100 GPUs. Recent studies (Li et al., 2021; Bu et al., 2023) have enhanced this efficacy by enabling DP training without the necessity to store per-sample gradients. We made significant efficiency improvements to the training pipeline using two techniques: ghost norm (Li et al., 2021) and the automatic mixed precision (AMP) package in PyTorch. Combining these two techniques with DP-SGD requires careful considerations to ensure both a correct DP guarantee as well as numerical stability. We detail the implementation in Appendix A.2. 

In Figure 4 we compare the compute cost of different gradient computation methods: functorch, ghost norm and ghost norm+AMP. The number of GPU hours is estimated 

5 

**Differentially Private Representation Learning via Image Captioning** 


![](raw/papers/2_LLM_NLP/LLM_paper_list/caption/icml_2024_Differentially_Private_Representation_Learning_via_Image_Captioning_images/icml_2024_Differentially_Private_Representation_Learning_via_Image_Captioning.pdf-0006-01.png)


**----- Start of picture text -----**<br>
6000<br>functorch<br>5000<br>Ghost norm<br>4000<br>Ghost norm+AMP<br>3000<br>2000<br>1000<br>0<br>Tiny Small Base Large<br>(59M) (115M) (190M) (407M)<br>GPU Hours<br>**----- End of picture text -----**<br>


Figure 4: Number of GPU hours to train DP-Cap for a single epoch on 233M samples. For the Large model, we achieve a close to 5 _×_ reduction. 

|**Method**|**Model**<br>Tiny<br>Small<br>Base<br>Large|
|---|---|
|functorch<br>36<br>20<br>12<br>5<br>Ghost norm<br>350<br>230<br>180<br>64<br>Ghost norm+AMP<br>420<br>310<br>225<br>90||



Table 1: Max physical batch size under different persample gradient computation methods. 

on a single NVIDIA V100 GPU with 32GB memory using 100K samples. The improvement is especially notable for our largest model: Using ghost norm+AMP, we achieve a 4 _._ 7 _×_ speedup compared to functorch and 3 _._ 3 _×_ speedup compared to ghost norm alone, which amounts to a reduction from 61 days to 13 days when training for 32 epochs—a large but manageable compute cost. This improvement is due to both a more efficient forward-backward pass, as well as enabling a larger physical batch size; see Table 1. In addition, we adopt the TAN simulation framework (Sander et al., 2023) to reduce the compute cost during hyperparameter search. Due to the batch size scaling behavior depicted in Figure 3, TAN simulation is ideal for DP-Cap training and allows for rapid experimentation to identify promising methods before launching a full training run. 

## **4. Evaluation** 

We demonstrate the representation learning capabilities of DP-Cap on both vision (V) and vision-language (V-L) downstream tasks. For all evaluations, the DP-Cap model is first pre-trained using DP-SGD on a subset of LAION2B (Schuhmann et al., 2022), and then fine-tuned nonprivately on a downstream dataset. 

## **4.1. Downstream Tasks** 

**Linear probing (V).** We train a linear classifier on top of learned representations and evaluate its accuracy. We consider both full linear probing using the full downstream dataset, as well as few-shot linear probing, which subsamples the downstream dataset down to _K_ samples per class. Few-shot linear probing is especially useful for evaluating learned representations since the model must rely heavily on the generalizability of representations in order to perform well under data scarcity. 

**Zero-shot image classification (V-L)** is one of the most widely used methodologies for evaluating vision-language models (Radford et al., 2021). A strong zero-shot perfor- 

mance suggests that the image representation aligns well to text. We perform zero-shot classification using the DP-Cap image encoder and text decoder by evaluating the likelihood of captions of the form “this is a photo of a [label]”. We enumerate over different labels and predict the class that has the highest likelihood; see Section A.3.1 for full details. 

**ARO (Attribution, Relation, and Order) (V-L).** The ARO benchmark (Yuksekgonul et al., 2022) can be used to gauge the adeptness of VLMs in understanding the compositional relationship between objects and attributes. A strong performance on ARO suggests that the learned image representation encodes semantic relationships such as “the horse is eating the grass” vs. “the grass is eating the horse”. 

## **4.2. Experimental Setup** 

We present an overview of the experimental setup; refer to Appendix A for additional details. 

**Datasets.** Following the approach introduced by Yu et al. (2023), we first pre-train on the Shader21k dataset (Baradad et al., 2022) of synthetic images. We then train with DP-SGD on a subset comprising 233 million deduplicated (using SemDeDup (Abbas et al., 2023)), NSFWfiltered and face-blurred (using an approach similar to Yang et al. (2021)) image-caption pairs from the (English-only) LAION-2B dataset (Schuhmann et al., 2022). We refer to this dataset as Dedup-LAION-233M. 

We use the ImageNet-1K (Deng et al., 2009a; Russakovsky et al., 2014), CIFAR-10/100 (Krizhevsky et al., 2009), Places-365/205 (Zhou et al., 2014) and iNaturalist2021 (Van Horn et al., 2021) image classification datasets to assess the performance of learned image representations via full linear probing, few-shot linear probing, and zeroshot prediction. For vision-language tasks, we employ the Visual Genome Attribution (VGA), Visual Genome Relation (VGR), COCO-order (Lin et al., 2015) and Flickr30k (Plummer et al., 2016) datasets from the ARO benchmark (Yuksekgonul et al., 2022). Finally, we evaluate image 

6 

**Differentially Private Representation Learning via Image Captioning** 

Table 2: Linear probing evaluation on downstream classification. Results for DP-NFNet, TAN, AlexNet and SimCLR are obtained from Yu et al. (2023). For ViP (Yu et al., 2023), we train with the same privacy parameters as for DP-Cap on the deduplicated dataset. More details are given in Appendix A.1. 

||Model<br>pretraining data<br>DP?<br>ImageNet-1K<br>Places-365<br>Places-205<br>iNat-2021|Model<br>pretraining data<br>DP?<br>ImageNet-1K<br>Places-365<br>Places-205<br>iNat-2021|
|---|---|---|
||DP-NFNet<br>ImageNet-1K<br>✓<br>45.3%<br>40.1%<br>39.2%<br>28.2%<br>TAN<br>ImageNet-1K<br>✓<br>49.0%<br>40.5%<br>38.2%<br>31.7%<br>AlexNet<br>ImageNet-1K<br>✗<br>56.5%<br>39.8%<br>35.1%<br>23.7%<br>SimCLR<br>ImageNet-1K<br>✗<br>67.5%<br>46.8%<br>49.3%<br>34.8%<br>Cap<br>Dedup-LAION-233M<br>✗<br>77.5%<br>56.3%<br>63.9%<br>63.9%<br>||
||MAE<br>Dedup-LAION-233M<br>✗<br>62.5%<br>51.0%<br>54.7%<br>42.3%||
||ViP<br>Dedup-LAION-233M<br>✓<br>56.5%<br>47.7%<br>49.6%<br>38.2%||
||DP-Cap<br>Dedup-LAION-233M<br>✓|63.4%<br>51.9%<br>54.3%<br>44.5%|



Table 3: Performance of DP-Cap on zero-shot classification and compositional understanding (ARO). CLIP’s zero-shot results are obtained from Radford et al. (2021) (base model). For ARO, see Appendix A.3.2. 

|Model<br>DP?<br>Zero-shot<br>ARO<br>ImageNet-1k<br>CIFAR10<br>CIFAR100<br>VGR<br>VGA<br>COCO<br>Flickr|Model<br>DP?<br>Zero-shot<br>ARO<br>ImageNet-1k<br>CIFAR10<br>CIFAR100<br>VGR<br>VGA<br>COCO<br>Flickr|Model<br>DP?<br>Zero-shot<br>ARO<br>ImageNet-1k<br>CIFAR10<br>CIFAR100<br>VGR<br>VGA<br>COCO<br>Flickr|
|---|---|---|
||||
|Random Chance<br>-|0.1%<br>10%<br>1%|50%<br>50%<br>20%<br>20%|
||||
|CLIP<br>✗<br>62.2%<br>91.3%<br>65.1%<br>62.4%<br>62.9%<br>47.8%<br>58.0%<br>Cap<br>✗<br>25.2%<br>90.0%<br>37.4%<br>59.9%<br>87.2%<br>87.0%<br>87.4%|||
||||
|DP-Cap<br>✓|7.8%<br>54.4%<br>16.4%|58.6%<br>82.4%<br>86.6%<br>87.2%|



captioning using the MS-COCO 2017 (Lin et al., 2015) test set; result is shown in Figure 1(c) and Appendix B.3. 

**Model and training.** We use a transformer architecture (Vaswani et al., 2017) for both the encoder and the decoder of DP-Cap, where the decoder applies causal crossattention. We train DP-Cap model from scratch, including the embedding layer, with vocabulary size around 32,000. See Section A.1 and Tschannen et al. (2023) for details. For privacy accounting we use Renyi DP composi-´ tion along with privacy amplification via Poisson subsampling (Mironov et al., 2019), and convert to DP using Balle et al. (2020) through the Opacus library (Yousefpour et al., 2021), targeting _δ_ = 1 _/N_ where _N_ represents the number of training samples. We refer to the non-private counterpart of DP-Cap trained on the same dataset as “Cap”. 

## **4.3. Main Results** 

**Linear probing evaluation (V).** We assess the performance of the vision encoder on downstream tasks via linear probing. In Fig 1(a), we compare the performance of DP-Cap and ViP (Yu et al., 2023) on ImageNet-1k few-shot linear probing. DP-Cap significantly improves over ViP, with up to _×_ 2.5 better performance across different shots. In addition, we 

evaluate the full linear probing accuracy of DP-Cap, ViP and other baselines in Table 2. DP-Cap outperforms ViP and other DP models, including TAN (Sander et al., 2023) and DP-NFNet (De et al., 2022), across all tasks. DP-Cap even outperforms non-private AlexNet (Krizhevsky et al., 2012) and except on ImageNet, SimCLR (Chen et al., 2020) (both were trained on ImageNet). We provide additional results for fine-tuning on downstream datasets in Table 10 (App. B), also showing improvements over competing methods. 

**Zero-shot performance (V-L).** In the left three columns of Table 3, we evaluate the zero-shot performance of DP-Cap compared to non-private Cap and CLIP/BLIP on ImageNet-1k and CIFAR10/100. Contrastive methods such as CLIP and BLIP have demonstrated greater suitability for zero-shot prediction compared to image captioning approaches (Tschannen et al., 2023), which is evident by the disparity between the performance of Cap and CLIP/BLIP. Nevertheless, we observe that DP-Cap achieves noteworthy zero-shot classification performance that is significantly above random chance, and stands as the first DP model to do so. This accomplishment marks a promising milestone for DP training, although there remains a substantial performance gap between DP-Cap and Cap. 

7 

**Differentially Private Representation Learning via Image Captioning** 

Table 4: Ablation studies on the effect of dataset size and privacy budget _ε_ on DP-Cap (base). 

|_ε_<br>_σ_<br># Data<br># Steps<br>_B_|ImageNet-1K|ARO (V-L)|
|---|---|---|
|0-shot (V-L)<br>1-shot (V)<br>2-shot (V)<br>10-shot (V)<br>VGR<br>VGA<br>COCO<br>Flickr|||
|+_∞_<br>0<br>233M<br>60,000<br>40,960<br>25.2%<br>27.0%<br>37.2%<br>57.9%<br>59.9%<br>87.2%<br>87.0%<br>87.4%<br>8.0<br>0.728<br>233M<br>5708<br>1.3M<br>7.8%<br>10.3%<br>15.6%<br>31.8%<br>58.6%<br>82.4%<br>86.6%<br>87.2%<br>2.0<br>1.18<br>233M<br>2854<br>1.3M<br>3.2%<br>7.0%<br>10.8%<br>23.9%<br>58.5%<br>79.7%<br>85.3%<br>86.6%<br>1.0<br>1.5<br>233M<br>1427<br>1.3M<br>1.1%<br>5.2%<br>8.2%<br>19.9%<br>58.3%<br>75.6%<br>83.9%<br>85.3%<br>8.0<br>0.728<br>23M<br>5708<br>130K<br>0.7%<br>3.4%<br>5.3%<br>13.3%<br>58.3%<br>76.2%<br>84.9%<br>85.9%<br>8.0<br>0.728<br>2.3M<br>5708<br>13K<br>0.1%<br>1.8%<br>2.9%<br>8.1%<br>57.6%<br>66.4%<br>79.5%<br>82.0%|||



Table 5: Ablation studies on the effect of model size. We compare ViP and DP-Cap’s number of encoder parameters. More details about the DP-Cap models can be found in Table 6. 

|etails about the DP-Cap models can be found in Table6.|etails about the DP-Cap models can be found in Table6.|etails about the DP-Cap models can be found in Table6.|
|---|---|---|
|Model<br>Confg<br># parameters<br>ImageNet-1K (Vision)<br>ARO (Vision-Language)|||
||||
||1-shot<br>2-shot<br>5-shot<br>10-shot<br>LP|VGR<br>VGA<br>COCO<br>Flickr|
||||
|ViP<br>Base<br>86.6M|2.5%<br>4.2%<br>8.5%<br>14.3%<br>56.5%|/<br>/<br>/<br>/|
|DP-Cap<br>Tiny<br>22.0M<br>7.9%<br>12.1%<br>18.7%<br>25.2%<br>57.5%<br>58.6%<br>79.1%<br>85.7%<br>87.1%<br>DP-Cap<br>Small<br>49.0M<br>9.0%<br>14.0%<br>21.6%<br>28.9%<br>61.1%<br>59.1%<br>80.5%<br>86.0%<br>86.6%<br>DP-Cap<br>Base<br>86.6M<br>10.3%<br>15.6%<br>24.2%<br>31.8%<br>63.4%<br>58.6%<br>82.4%<br>86.6%<br>87.2%|||
|DP-Cap<br>Large<br>407.3M|11.8%<br>17.5%<br>26.2%<br>34.0%<br>65.8%|59.5%<br>80.1%<br>86.6%<br>86.5%|



**Attribution, Relation, and Order (ARO) evaluation (VL).** Contrastive-based methods such as CLIP often exhibit behavior akin to bag-of-words models (Yuksekgonul et al., 2022; Tejankar et al., 2021; Basu et al., 2023), making them less adept at performing well on the ARO benchmark. Remarkably, DP-Cap significantly outperforms _non-private_ CLIP in this context (see Fig 1(b) and Table 3), and even achieves performance close to that of _non-private_ Cap. Our result shows that DP training can be particularly effective for learning complex compositional relationships. 

## **4.4. Ablation Studies** 

We perform ablation studies on the scaling behavior of DPCap with respect to the dataset size, privacy budget and model size. In Appendix B, we show additional results on image captioning and on the impact of compute budget. 

**Scaling dataset size.** We show that dataset scaling is crucial for effectively training DP-Cap as it results in better SNR under the same privacy budget (see Figure 5). We randomly subsample 1% and 10% of the Dedup-LAION233M dataset, which is used for training our default DP-CapBase model in Table 2 (denoted by Dedup-LAION-2M and Dedup-LAION-23M). We set the batch size to _B/_ 100 for Dedup-LAION-2M and _B/_ 10 for Dedup-LAION-23M, respectively. This allows the model to be trained for the same number of steps across the different datasets, although at a much larger effective noise level. As shown in Table 4, the number of training samples is critical for achieving strong 

performance for DP-Cap models: the zero-shot performance of our model trained on 1% of the dataset achieves random zero-shot performance on ImageNet and much worse accuracy across the board on ARO. 

**Impact of the privacy budget** _ε_ **.** We also investigate the performance of DP-Cap under lower privacy budgets ( _ε_ = 1 and _ε_ = 2), employing the same batch size of 1.3 million. The outcomes of these experiments are presented in Table 4. As anticipated, the utility of our model does exhibit a decline with decreasing _ε_ . However, the performance degradation is relatively minor for the learned representation, with 10-shot ImageNet performance decreasing from 31.8% ( _ε_ = 8) to 19.9% ( _ε_ = 1). More surprisingly, the performance impact on ARO is nearly negligible. It is noteworthy that both models continue to outperform previous state-of-the-art DP models trained with _ε_ = 8 (see Figure 1). This phenomenon can be attributed to the relatively small effective noise resulting from the extreme batch size, which for _ε_ = 1 remains five times smaller than that used in (Yu et al., 2023). 

**Scaling model size.** Scaling up the model size is one of the most effective approaches for training better non-private foundation models (Brown et al., 2020; Bommasani et al., 2021; Touvron et al., 2023). However, an intuitive understanding of DP-SGD[2] suggests that scaling up model size 

> 2This is because the added noise has _L_ 2 norm _≈ σC_ ~~_√_~~ _d/B_ , where _d_ is the number of model parameters, whereas the gradient norm is constrained to _C_ regardless of model size. 

8 

**Differentially Private Representation Learning via Image Captioning** 

does not improve utility in DP training since more model parameters will lead to lower signal-to-noise ratio, especially when training large DP models from scratch. Recent work (Li et al., 2022b) demonstrates that during DP finetuning, gradients occupy a lower effective dimensional subspace, and larger networks can enhance the performance of DP-SGD. To test this hypothesis in pre-training, we train DPCap with different model sizes (Tiny, Small, Base, Large) using the same hyperparameters and evaluate their performance in Table 5,11; see Table 6 for details about different model sizes. We observe consistent improvements when scaling up the model from DP-Cap-Tiny to DP-Cap-Large. Our observation suggests that DP-Cap has strong model scaling behavior even with DP-SGD training. 

## **5. Discussion and Future Work** 

We demonstrated that DP representation learning via image captioning is viable. In particular, image captioning is an ideal objective that supports both per-sample loss and large batch training—two critical ingredients in DP-SGD. When applied to the Dedup-LAION-233M dataset, the trained model learns useful image representations for downstream tasks and exhibits strong multi-modal capabilities. 

Through our study we also identify three open problems in the general direction of DP pre-training of large-scale foundation models that are difficult to handle with existing techniques: 

1. Scaling up the batch size to extreme levels is crucial for reducing the effective noise and facilitating model convergence. Is there a fundamental reason why image captioning can support extremely large batch training? When the model does not scale well with batch size, is it due to poor optimization or is it inherent to the model and/or task? Answers to these question can help identify other successful training recipes that further improve the quality of DP-learned representations. 

2. Scaling up the model size is typically beneficial in nonprivate representation learning, but can be ineffective in DP training due to a decrease in gradient SNR. While we have observed a performance improvement from Tiny to Large, we anticipate that deterioration could occur when scaling up to a much larger model size, especially when the signal-to-noise ratio (SNR) undergoes a certain threshold. Does this threshold exist? Can this trade-off be resolved through parameter-efficient architectures? 

3. Contrastive learning offers unique advantages compared to other multi-modal learning objectives such as learning image and text representations that are semantically aligned, but is not compatible with standard DP-SGD training. What techniques can enable differentially private contrastive learning? 

## **Acknowledgements** 

This research was partially supported as a BAIR Open Research Common Project with Meta. We thank Quentin Garrido for his time and insights regarding the significance of vision backbones quality evaluations. 

## **Impact Statement** 

Our work focuses on developing privacy-preserving techniques for machine learning. Progress in this direction has significant potential for evolving the concept of privacy for AI. The purpose of our work is to push the boundary of what is achievable for representation learning through differentially private mechanisms, towards determining the technical feasibility of applying DP to tackle real-world AI privacy challenges. 

## **References** 

- Abadi, M., Chu, A., Goodfellow, I., McMahan, H. B., Mironov, I., Talwar, K., and Zhang, L. Deep learning with differential privacy. In _Proceedings of the 2016 ACM SIGSAC conference on computer and communications security_ , pp. 308–318, 2016. 

- Abbas, A., Tirumala, K., Simig, D., Ganguli, S., and Morcos, A. S. Semdedup: Data-efficient learning at webscale through semantic deduplication. _arXiv preprint arXiv:2303.09540_ , 2023. 

- Assran, M., Caron, M., Misra, I., Bojanowski, P., Bordes, F., Vincent, P., Joulin, A., Rabbat, M., and Ballas, N. Masked siamese networks for label-efficient learning. In _European Conference on Computer Vision_ , pp. 456–473. Springer, 2022. 

- Balle, B., Barthe, G., Gaboardi, M., Hsu, J., and Sato, T. Hypothesis testing interpretations and Renyi differential´ privacy. In _International Conference on Artificial Intelligence and Statistics_ , pp. 2496–2506. PMLR, 2020. 

- Balle, B., Cherubin, G., and Hayes, J. Reconstructing training data with informed adversaries. In _2022 IEEE Symposium on Security and Privacy (SP)_ , pp. 1138–1156. IEEE, 2022. 

- Baradad, M., Chen, R., Wulff, J., Wang, T., Feris, R., Torralba, A., and Isola, P. Procedural image programs for representation learning. _Advances in Neural Information Processing Systems_ , 35:6450–6462, 2022. 

- Basu, S., Sanjabi, M., Massiceti, D., Hu, S. X., and Feizi, S. Augmenting clip with improved visio-linguistic reasoning. _arXiv preprint arXiv:2307.09233_ , 2023. 

9 

**Differentially Private Representation Learning via Image Captioning** 

- Berrada, L., De, S., Shen, J. H., Hayes, J., Stanforth, R., Stutz, D., Kohli, P., Smith, S. L., and Balle, B. Unlocking accuracy and fairness in differentially private image classification. _arXiv preprint arXiv:2308.10888_ , 2023. 

- Bommasani, R., Hudson, D. A., Adeli, E., Altman, R., Arora, S., von Arx, S., Bernstein, M. S., Bohg, J., Bosselut, A., Brunskill, E., et al. On the opportunities and risks of foundation models. _arXiv preprint arXiv:2108.07258_ , 2021. 

- Brown, T. B., Mann, B., Ryder, N., Subbiah, M., Kaplan, J., Dhariwal, P., Neelakantan, A., Shyam, P., Sastry, G., Askell, A., et al. Language models are few-shot learners. _arXiv preprint arXiv:2005.14165_ , 2020. 

- Bu, Z., Wang, Y.-X., Zha, S., and Karypis, G. Differentially private optimization on large model at small cost. In _International Conference on Machine Learning_ , pp. 3192– 3218. PMLR, 2023. 

- Bun, M. and Steinke, T. Concentrated differential privacy: Simplifications, extensions, and lower bounds. In _Theory of Cryptography Conference_ , pp. 635–658. Springer, 2016. 

- Carlini, N., Chien, S., Nasr, M., Song, S., Terzis, A., and Tramer, F. Membership inference attacks from first principles. _arXiv preprint arXiv:2112.03570_ , 2021. 

- Changpinyo, S., Sharma, P., Ding, N., and Soricut, R. Conceptual 12m: Pushing web-scale image-text pre-training to recognize long-tail visual concepts. In _Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition_ , pp. 3558–3568, 2021. 

- Chen, T., Kornblith, S., Norouzi, M., and Hinton, G. E. A simple framework for contrastive learning of visual representations. _CoRR_ , abs/2002.05709, 2020. URL https://arxiv.org/abs/2002.05709. 

- De, S., Berrada, L., Hayes, J., Smith, S. L., and Balle, B. Unlocking high-accuracy differentially private image classification through scale, 2022. URL https: //arxiv.org/abs/2204.13650. 

- Deng, J., Dong, W., Socher, R., Li, L.-J., Li, K., and Fei-Fei, L. Imagenet: A large-scale hierarchical image database. In _2009 IEEE Conference on Computer Vision and Pattern Recognition_ , pp. 248–255, 2009a. doi: 10.1109/CVPR.2009.5206848. 

- Deng, J., Dong, W., Socher, R., Li, L.-J., Li, K., and Fei-Fei, L. Imagenet: A large-scale hierarchical image database. In _2009 IEEE conference on computer vision and pattern recognition_ , pp. 248–255. Ieee, 2009b. 

- Desai, K. and Johnson, J. Virtex: Learning visual representations from textual annotations. In _Proceedings of the IEEE/CVF conference on computer vision and pattern recognition_ , pp. 11162–11173, 2021. 

- Dwork, C. and Rothblum, G. N. Concentrated differential privacy. _arXiv preprint arXiv:1603.01887_ , 2016. 

- Dwork, C., McSherry, F., Nissim, K., and Smith, A. Calibrating noise to sensitivity in private data analysis. In _Proceedings of the Third Conference on Theory of Cryptography_ , pp. 265–284, 2006. 

- Gadre, S. Y., Ilharco, G., Fang, A., Hayase, J., Smyrnis, G., Nguyen, T., Marten, R., Wortsman, M., Ghosh, D., Zhang, J., et al. Datacomp: In search of the next generation of multimodal datasets. _arXiv preprint arXiv:2304.14108_ , 2023. 

- Gopi, S., Lee, Y. T., and Wutschitz, L. Numerical composition of differential privacy. _Advances in Neural Information Processing Systems_ , 34:11631–11642, 2021. 

- Guo, C., Karrer, B., Chaudhuri, K., and van der Maaten, L. Bounding training data reconstruction in private (deep) learning. In _International Conference on Machine Learning_ , pp. 8056–8071. PMLR, 2022. 

- Guo, C., Sablayrolles, A., and Sanjabi, M. Analyzing privacy leakage in machine learning via multiple hypothesis testing: A lesson from fano. In _International Conference on Machine Learning_ , pp. 11998–12011. PMLR, 2023. 

- He, K., Chen, X., Xie, S., Li, Y., Dollar, P., and Girshick,´ R. Masked autoencoders are scalable vision learners. In _Proceedings of the IEEE/CVF conference on computer vision and pattern recognition_ , pp. 16000–16009, 2022. 

- Jayaraman, B. and Evans, D. Evaluating differentially private machine learning in practice. In _28th USENIX Security Symposium (USENIX Security 19)_ , pp. 1895–1912, 2019. 

- Kingma, D. P. and Ba, J. Adam: A method for stochastic optimization. _arXiv preprint arXiv:1412.6980_ , 2014. 

- Krizhevsky, A., Hinton, G., et al. Learning multiple layers of features from tiny images. 2009. 

- Krizhevsky, A., Sutskever, I., and Hinton, G. E. Imagenet classification with deep convolutional neural networks. _Advances in neural information processing systems_ , 25, 2012. 

- Kurakin, A., Song, S., Chien, S., Geambasu, R., Terzis, A., and Thakurta, A. Toward training at imagenet scale with differential privacy. _arXiv preprint arXiv:2201.12328_ , 2022. 

10 

**Differentially Private Representation Learning via Image Captioning** 

- Lee, J. and Kifer, D. Scaling up differentially private deep learning with fast per-example gradient clipping. _arXiv preprint arXiv:2009.03106_ , 2020. 

- Li, J., Li, D., Xiong, C., and Hoi, S. Blip: Bootstrapping language-image pre-training for unified vision-language understanding and generation. In _International Conference on Machine Learning_ , pp. 12888–12900. PMLR, 2022a. 

- Li, X., Tramer, F., Liang, P., and Hashimoto, T.` Large language models can be strong differentially private learners, 2021. 

- Li, X., Liu, D., Hashimoto, T. B., Inan, H. A., Kulkarni, J., Lee, Y.-T., and Guha Thakurta, A. When does differentially private learning not suffer in high dimensions? _Advances in Neural Information Processing Systems_ , 35: 28616–28630, 2022b. 

- Li, Y., Fan, H., Hu, R., Feichtenhofer, C., and He, K. Scaling language-image pre-training via masking. In _Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition_ , pp. 23390–23400, 2023. 

- Lin, T.-Y., Maire, M., Belongie, S., Bourdev, L., Girshick, R., Hays, J., Perona, P., Ramanan, D., Zitnick, C. L., and Dollar, P.´ Microsoft coco: Common objects in context, 2015. 

- Liu, J. and Talwar, K. Private selection from private candidates. In _Proceedings of the 51st Annual ACM SIGACT Symposium on Theory of Computing (STOC)_ , pp. 298– 309, 2019. 

- Loshchilov, I. and Hutter, F. Decoupled weight decay regularization. In _International Conference on Learning Representations_ , 2018. 

- Mairal, J. Cyanure: An open-source toolbox for empirical risk minimization for python, c++, and soon more. _arXiv preprint arXiv:1912.08165_ , 2019. 

- McMahan, H. B., Ramage, D., Talwar, K., and Zhang, L. Learning differentially private recurrent language models. _arXiv preprint arXiv:1710.06963_ , 2017. 

- Mironov, I. Renyi differential privacy.´ In _2017 IEEE 30th Computer Security Foundations Symposium (CSF)_ , pp. 263–275. IEEE, 2017. 

- Mironov, I., Talwar, K., and Zhang, L. Renyi differential´ privacy of the Sampled Gaussian Mechanism. _arXiv preprint arXiv:1908.10530_ , 2019. 

- Mu, N., Kirillov, A., Wagner, D., and Xie, S. Slip: Selfsupervision meets language-image pre-training. In _European Conference on Computer Vision_ , pp. 529–544. Springer, 2022. 

- Plummer, B. A., Wang, L., Cervantes, C. M., Caicedo, J. C., Hockenmaier, J., and Lazebnik, S. Flickr30k entities: Collecting region-to-phrase correspondences for richer image-to-sentence models, 2016. 

- Radford, A., Kim, J. W., Hallacy, C., Ramesh, A., Goh, G., Agarwal, S., Sastry, G., Askell, A., Mishkin, P., Clark, J., et al. Learning transferable visual models from natural language supervision. In _International conference on machine learning_ , pp. 8748–8763. PMLR, 2021. 

- Renyi,´ A. On measures of entropy and information. In _Proceedings of the Fourth Berkeley Symposium on Mathematical Statistics and Probability, Volume 1: Contributions to the Theory of Statistics_ , volume 4, pp. 547–562. University of California Press, 1961. 

- Russakovsky, O., Deng, J., Su, H., Krause, J., Satheesh, S., Ma, S., Huang, Z., Karpathy, A., Khosla, A., Bernstein, M., Berg, A. C., and Fei-Fei, L. Imagenet large scale visual recognition challenge, 2014. URL https:// arxiv.org/abs/1409.0575. 

- Sander, T., Stock, P., and Sablayrolles, A. Tan without a burn: Scaling laws of dp-sgd. In _International Conference on Machine Learning_ , pp. 29937–29949. PMLR, 2023. 

- Sander, T., Sylvestre, M., and Durmus, A. Implicit bias in noisy-sgd: With applications to differentially private training. _arXiv preprint arXiv:2402.08344_ , 2024. 

- Sariyildiz, M. B., Perez, J., and Larlus, D. Learning visual representations with caption annotations. In _Computer Vision–ECCV 2020: 16th European Conference, Glasgow, UK, August 23–28, 2020, Proceedings, Part VIII 16_ , pp. 153–170. Springer, 2020. 

- Schuhmann, C., Vencu, R., Beaumont, R., Kaczmarczyk, R., Mullis, C., Katta, A., Coombes, T., Jitsev, J., and Komatsuzaki, A. Laion-400m: Open dataset of clipfiltered 400 million image-text pairs. _arXiv preprint arXiv:2111.02114_ , 2021. 

- Schuhmann, C., Beaumont, R., Vencu, R., Gordon, C., Wightman, R., Cherti, M., Coombes, T., Katta, A., Mullis, C., Wortsman, M., Schramowski, P., Kundurthy, S., Crowson, K., Schmidt, L., Kaczmarczyk, R., and Jitsev, J. Laion-5b: An open large-scale dataset for training next generation image-text models. _ArXiv_ , abs/2210.08402, 2022. URL https://api.semanticscholar. org/CorpusID:252917726. 

- Shokri, R., Stronati, M., Song, C., and Shmatikov, V. Membership inference attacks against machine learning models. In _2017 IEEE Symposium on Security and Privacy (S&P)_ , pp. 3–18, 2017. 

11 

**Differentially Private Representation Learning via Image Captioning** 

- Song, S., Chaudhuri, K., and Sarwate, A. D. Stochastic gradient descent with differentially private updates. In _IEEE Global Conference on Signal and Information Processing (GlobalSIP)_ , pp. 245–248, 2013. 

- Tang, X., Panda, A., Sehwag, V., and Mittal, P. Differentially private image classification by learning priors from random processes. _Advances in Neural Information Processing Systems_ , 36, 2024. 

- Tejankar, A., Sanjabi, M., Wu, B., Xie, S., Khabsa, M., Pirsiavash, H., and Firooz, H. A fistful of words: Learning transferable visual models from bag-of-words supervision. _arXiv preprint arXiv:2112.13884_ , 2021. 

- Touvron, H., Cord, M., and Jegou, H.´ Deit iii: Revenge of the vit. In _European Conference on Computer Vision_ , pp. 516–533. Springer, 2022. 

- Touvron, H., Lavril, T., Izacard, G., Martinet, X., Lachaux, M.-A., Lacroix, T., Roziere, B., Goyal, N., Hambro, E.,` Azhar, F., et al. Llama: Open and efficient foundation language models. _arXiv preprint arXiv:2302.13971_ , 2023. 

- Tramer, F. and Boneh, D. Differentially private learning needs better features (or much more data). In _International Conference on Learning Representations_ , 2020. 

   - You, Y., Gitman, I., and Ginsburg, B. Scaling SGD batch size to 32k for imagenet training. _CoRR_ , abs/1708.03888, 2017. URL http://arxiv.org/ abs/1708.03888. 

   - Yousefpour, A., Shilov, I., Sablayrolles, A., Testuggine, D., Prasad, K., Malek, M., Nguyen, J., Ghosh, S., Bharadwaj, A., Zhao, J., Cormode, G., and Mironov, I. Opacus: Userfriendly differential privacy library in PyTorch, 2021. 

   - Yu, J., Wang, Z., Vasudevan, V., Yeung, L., Seyedhosseini, M., and Wu, Y. Coca: Contrastive captioners are imagetext foundation models. _arXiv preprint arXiv:2205.01917_ , 2022. 

   - Yu, Y., Sanjabi, M., Ma, Y., Chaudhuri, K., and Guo, C. Vip: A differentially private foundation model for computer vision. _arXiv preprint arXiv:2306.08842_ , 2023. 

   - Yuksekgonul, M., Bianchi, F., Kalluri, P., Jurafsky, D., and Zou, J. When and why vision-language models behave like bags-of-words, and what to do about it? In _The Eleventh International Conference on Learning Representations_ , 2022. 

   - Zhou, B., Lapedriza, A., Xiao, J., Torralba, A., and Oliva, A. Learning deep features for scene recognition using places database. _Advances in neural information processing systems_ , 27, 2014. 

- Tschannen, M., Kumar, M., Steiner, A., Zhai, X., Houlsby, N., and Beyer, L. Image captioners are scalable vision learners too. _arXiv preprint arXiv:2306.07915_ , 2023. 

- Van Horn, G., Cole, E., Beery, S., Wilber, K., Belongie, S., and Mac Aodha, O. Benchmarking representation learning for natural world image collections. In _Proceedings of the IEEE/CVF conference on computer vision and pattern recognition_ , pp. 12884–12893, 2021. 

- Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A. N., Kaiser, Ł., and Polosukhin, I. Attention is all you need. _Advances in neural information processing systems_ , 30, 2017. 

- Vedantam, R., Zitnick, C. L., and Parikh, D. Cider: Consensus-based image description evaluation, 2015. 

- Wang, Y.-X., Balle, B., and Kasiviswanathan, S. P. Subsampled renyi´ differential privacy and analytical moments accountant. In _The 22nd International Conference on Artificial Intelligence and Statistics_ , pp. 1226–1235. PMLR, 2019. 

- Yang, K., Yau, J., Fei-Fei, L., Deng, J., and Russakovsky, O. A study of face obfuscation in imagenet. _CoRR_ , abs/2103.06191, 2021. URL https://arxiv.org/ abs/2103.06191. 

12 

**Differentially Private Representation Learning via Image Captioning** 

## **A. Implementation Details** 

## **A.1. Training Details** 

**DP accounting.** We use RDP accounting with subsampling from the Opacus library (Yousefpour et al., 2021). Let _Dα_ denote the R´enyi divergence of order _α_ (R´enyi, 1961), and let 


![](raw/papers/2_LLM_NLP/LLM_paper_list/caption/icml_2024_Differentially_Private_Representation_Learning_via_Image_Captioning_images/icml_2024_Differentially_Private_Representation_Learning_via_Image_Captioning.pdf-0013-04.png)


Then, from Mironov et al. (2019), performing _S_ steps of DP-SGD satisfies ( _ε, δ_ )-DP with: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/caption/icml_2024_Differentially_Private_Representation_Learning_via_Image_Captioning_images/icml_2024_Differentially_Private_Representation_Learning_via_Image_Captioning.pdf-0013-06.png)


The quantity _gα_ ( _σ, q_ ) can be upper bounded mathematically or derived numerically: we use the Opacus (Yousefpour et al., 2021) library for accounting in our work. 

Regarding the DP guarantee, _ε_ -DP bounds the amount of information extracted from each training sample by _ε_ . Notably, for DP-Cap, each sample is made of _{_ image + caption _}_ , while ViP utilizes _{_ image _}_ only. Consequently, DP-Cap inherently offers an equivalent or better privacy guarantee for each image. One way to see it is to note that DP provides protection against membership inference attacks (Shokri et al., 2017). Suppose _ε_ -DP upper bounds the success rate of a membership inference attack (when given the image-text pair) against DP-Cap as _≤ p_ . Then the MIA success rate when given only the image can be at most _p_ since the attacker has strictly less information. This is exactly the upper bound for the success rate of a membership inference attack against ViP. In other words, any attacker that can attack the image+caption model (such as DP-Cap) can also attack the image-only model (such as ViP). 

On the other hand, since the _{_ image+caption _}_ models utilize the caption, the privacy leakage from the text part of the image-caption pair is non-zero for _ε >_ 0. It is worth noting that in our set up since we use DP, we protect the captions with the same _ε_ -DP guarantee. Thus, the privacy protection for DP-Cap is neither strictly stronger nor strictly weaker than that for ViP, so the two privacy notions are not directly comparable. 

**Model details and task description.** We utilize a transformer architecture (Vaswani et al., 2017) DP-Cap. This captioning model uses a text decoder that generates captions in an auto-regressive manner, utilizing a full attention mechanism on the vision encoder’s output, as well as causal attention on the text. This architecture is closely aligned with the Cap architecture introduced in (Tschannen et al., 2023). See Table 6 for details about the transformer architecture for different sizes. All results utilize the _base_ model with the exception of the comparison in Table 6. 

**Hyperparameters.** Our choice of gradient clipping factor is _C_ = 1, as we did not observe any performance improvement with other values. We always use AdamW (Loshchilov & Hutter, 2018) for training. We use a learning rate of 5 _._ 12 _×_ 10 _[−]_[4] . The learning rate is kept constant across batch sizes for TAN simulations and for the performance comparison in Figure 3 as the effective noise is kept constant in these cases (Sander et al., 2023). We use a maximum length of 40 tokens to process the LAION captions. We use a linear schedule, with 40% of warm-up iterations, and 2 _×_ the entire training as decay horizon. As opposed to what was previously observed (De et al., 2022; Sander et al., 2023), the learning rate schedule played an important role for us with DP-SGD training. We use a weight decay of 0 _._ 05. These choices come from hyperparameter search using TAN simulation with our base model. Following the standard practice (Berrada et al., 2023; De et al., 2022; Li et al., 2021; Yu et al., 2023; Sander et al., 2023), we do not count hyperparameter search within our privacy budget. (Liu & Talwar, 2019) have shown that hyperparameter search might not incur observable privacy loss. 

**Pre-training DP-Cap on the synthetic dataset.** Compared to the encoder and decoder architecture design used in masked autoencoders (MAE) (He et al., 2022), the two main differences of the image captioning model used in this paper are: (1) The output of the encoder is fed into the decoder via cross-attention (Vaswani et al., 2017) in Cap; and (2) The selfattention used in the Cap decoder is causal self-attention. Similar to Yu et al. (2023), we apply the synthetic image dataset, Shaders21K (Baradad et al., 2022), to pre-train the DP-Cap model via MAE-based training. We follow most of the training setups used in ViP synthetic pre-training (Yu et al., 2023), except that we feed the output of the encoder to the decoder via cross-attention. The training loss of the synthetic pre-training in this paper is still the reconstruction loss used in MAE (He et al., 2022), and we did not leverage real-world text data for pre-training. After the model is pre-trained on Shaders21K, we change the self-attention to causal self-attention in the decoder, and replace the final layer (for pixel-wise reconstruction) of the decoder with the (randomly initialized) decoding layer for next word prediction. After making these modifications, we apply DP-SGD to pre-train our DP-Cap model with standard image captioning training objectives (see Section 3.1). 

13 

**Differentially Private Representation Learning via Image Captioning** 

Table 6: Details of transformer backbone variants used in DP-Cap. 

|Model<br>Encoder depth<br>Encoder width<br>Decoder depth<br>Decoder width<br># parameters (encoder & decoder)|Model<br>Encoder depth<br>Encoder width<br>Decoder depth<br>Decoder width<br># parameters (encoder & decoder)|Model<br>Encoder depth<br>Encoder width<br>Decoder depth<br>Decoder width<br># parameters (encoder & decoder)|
|---|---|---|
|DP-Cap-Tiny<br>12<br>384<br>6<br>384<br>59M<br>DP-Cap-Small<br>12<br>576<br>6<br>576<br>115M<br>DP-Cap-Base<br>12<br>768<br>6<br>768<br>190M|||
|DP-Cap-Large|24<br>1024<br>6<br>768|407M|



**Pre-training ViP.** To conduct a comparison with training on an identical datasets, we follow the methodology outlined in (Yu et al., 2023) to train with DP-SGD a MAE-based model, but with a change in the training data from LAION-233M to Dedup-LAION-223M, and use the same encoder’s synthetic initialization as for DP-Cap. We further examine the linear probing performance on ImageNet and observe a 2% between the original model and the one trained on the deduplicated dataset. In addition, to corroborate the observation made in Figure 3, which suggests that the MAE-based method struggles to effectively harness massive batch sizes for achieving low effective noise in DP-SGD, we also train ViP models with larger batches, up to using the exact privacy parameters employed for DP-Cap (under _ε_ = 8) with a notably large batch size of 1.3 million, and showcase the results in Table 7. For full linear probing, we observe only a small improvement over the original ViP model that was trained with batch size 98k. The success of DP-Cap is not solely attributed to its appropriate privacy parameters but is also a consequence of its remarkable ability to leverage the small effective noised induced by extremely large batch sizes. 

## **A.2. Computation cost** 

**Mixed Precision Package & Ghost Norm** DP-SGD introduces additional computational overhead compared to nonprivate training, primarily due to the computation of per-sample gradient norms. By employing the ghost norm technique of Li et al. (2021), we have successfully reduced the computational cost by up to one third with the Large Model (see Figure 4) compared to using functorch. The torch.amp package offers convenient methods for mixed precision, significantly speeding up operations like linear layers. However, it often leads to NaNs due to low precision handling of extreme values. While one can skip a step that led to NaNs in normal training, combining AMP with Ghost Norm is more complex. Ghost Norm requires two backward passes. In the first pass, per-sample gradient norms are computed. If one gradient norm is NaN, it contaminates the entire batch, leading to a NaN in the second backward pass. This issue is particularly prevalent in our setting with a batch size of 1 _._ 3M, as even a minuscule proportion of computations leading to NaNs can cause problems. To address this, we propose two solutions: 

- **Loss Scaler** : We employ a similar trick to the standard use of AMP to reduce the number of NaNs. This involves dynamically upscaling and downscaling the loss with torch.cuda.amp.GradScaler. The same factor is used before the first and the second backward, and is updated based on the outputs of the second backward only. 

- **Clipping to 0** : If any per-sample gradient norm computation results in a NaN value after the first backward, we set its clipping coefficient (the multiplicative coefficient in front of the corresponding per-sample loss for the second backward, as detailed in Li et al. (2021)) to 0 for the second backward. In this case, we do not update the loss scaling factor. 

It’s worth noting that the second solution is entirely valid for DP: instead of clipping the per-sample gradient to a norm _C_ , it clips it to 0 in cases where computation results in a NaN value. This approach effectively mitigates the issue of NaN contamination in large batches. Overall, We have successfully reduced the computational cost by a factor 5 for the Large Model compared to functorch. 

**TAN simulation** Crucially, to achieve a favorable privacy-utility trade-off, DP-SGD necessitates training with massive batches over a substantial number of steps to achieve a good privacy-utility trade-off, as elaborated in Section 3.2. All our hyperparameter search were performed using the TAN simulation (Sander et al., 2023) for one epoch on our Dedup-LAION233M. For our _ε_ = 8 models, we limited training to 32 epochs, a process that took 5 days utilizing 128 V100 GPUs for the Base model. 

14 

**Differentially Private Representation Learning via Image Captioning** 

Table 7: Set-ups for our training for ViP, MAE (He et al., 2022) and DP-Cap: ImageNet-1k linear probing. 

|Model<br>pretraining data<br>(_B_,_σ_,_S_)|ImageNet-1K<br>1-shot<br>2-shot<br>5-shot<br>10-shot<br>full|
|---|---|
|||
|ViP (Yu et al.,2023)<br>LAION-233M<br>(98k, 0.48, 6000)<br>2.5%<br>4.1%<br>8.5%<br>14.2%<br>55.7%<br>ViP<br>Dedup-LAION-233M<br>(98k, 0.474, 5708)<br>2.3%<br>3.9%<br>8.0%<br>13.6%<br>53.4%<br>ViP<br>Dedup-LAION-233M<br>(200k, 0.513, 5708)<br>2.5%<br>4.1%<br>8.3%<br>14.1%<br>54.0%<br>ViP<br>Dedup-LAION-233M<br>(400k, 0.564, 5708)<br>2.5%<br>4.2%<br>8.7%<br>14.7%<br>55.2%<br>ViP<br>Dedup-LAION-233M<br>(1.3M, 0.728, 5708)<br>2.7%<br>4.6%<br>9.4%<br>15.7%<br>56.5%<br>MAE (Non private)<br>Dedup-LAION-233M<br>(40960, 0, 40000)<br>3.4%<br>5.8%<br>11.8%<br>19.5%<br>62.5%<br>DP-Cap<br>Dedup-LAION-233M<br>(98k, 0.474, 5708)<br>4.2%<br>6.9%<br>11.4%<br>17.2%<br>50.6%<br>DP-Cap<br>Dedup-LAION-233M<br>(200k, 0.513, 5708)<br>5.6%<br>8.9%<br>14.5%<br>20.2%<br>54.2%<br>DP-Cap<br>Dedup-LAION-233M<br>(400k, 0.564, 5708)<br>7.6%<br>11.5%<br>18.5%<br>25.3%<br>59.1%<br>DP-Cap<br>Dedup-LAION-233M<br>(1.3M, 0.728, 5708)<br>10.3%<br>15.6%<br>24.2%<br>31.8%<br>63.4%||



While we have tried to reduced it as much as possible, training DP-Cap imposed a considerable energy consumption, resulting in elevated CO2 emissions. Our intention in releasing these models is to contribute to the mitigation of future carbon emissions, as the training has already been completed. 

## **A.3. Evaluation Details** 

## A.3.1. DETAILS ABOUT ZERO-SHOT IMAGE CLASSIFICATION 

While methods employing contrastive learning, such as CLIP, excel in this task, captioning methods exhibit comparatively lower performance, and with greater computational demands during evaluation. To evaluate a captioning model’s zero-shot performance, we employ two distinct strategies: 

- **Tree-based search** : We initiate caption generation with a prompt like “this is a photo of a,” and greedily select the most likely next token among those that lead to valid completions within the true label set. The process continues until an End of Sentence (EOS) token is reached. For instance, if there are only two labels starting with “car”: “car [EOS]” and “carpet [EOS]”, and the initial predicted token is “car”. Then the text decoder will predict the next token among “[EOS]” and “pet”. If, among these two, “[EOS]” is chosen, and “car [EOS]” corresponds to the true label, then the zero-shot prediction is deemed correct. 

- **Loss-based classification** : We assess, for each image, the probability of various captions that begin with “this is a photo of a [...]” where “[...]” is substituted with all feasible labels. Subsequently, we select the label that yields the most probable caption. 

The “loss-based classification” comes with significantly higher computation costs as all the different captions have to be evaluated for each image (there representations is conditional to the image). For ImageNet, it implies 1000 forwards through the decoder for each image. We thus employ the tree-based search for presenting our findings in Table 3, although its greedy character with no backtracking is not optimal. Surprisingly, our preliminary experiments suggest the tree-based search gives comparable results. 

## A.3.2. DETAILS ABOUT ARO EVALUATION 

We adhered to the protocol and code base established in (Yuksekgonul et al., 2022) for re-evaluating CLIP’s performance, and we observe slightly different results (see Table 8). For our captioning models, our approach involved computing the cross-entropy loss for all possible captions associated with each image and subsequently selecting the one with the lowest loss. 

15 

**Differentially Private Representation Learning via Image Captioning** 

Table 8: Compositional understanding (ARO): Results for CLIP (base) in Yuksekgonul et al. (2022) compared to our evaluation. 

|Model<br>ARO<br>VGR<br>VGA<br>COCO<br>Flickr|Model<br>ARO<br>VGR<br>VGA<br>COCO<br>Flickr|
|---|---|
|||
|CLIP (eval fromYuksekgonul et al.(2022))<br>CLIP (our eval)|59%<br>63%<br>46%<br>60%<br>62.4%<br>62.9%<br>47.8%<br>58.0%|



Table 9: Training from random initialization: Superiority of DP-Cap over ViP, both trained from random initialization. 

|Model|ImageNet-1K<br>1-shot<br>2-shot<br>10-shot<br>full|
|---|---|
|||
|ViP (_ε_= 8)<br>DP-Cap (_ε_= 8)|0.1%<br>1.7%<br>6.1%<br>23.9%<br>5.6%<br>8.5%<br>18.8%<br>47.0%|



Table 10: Fine-tuning evaluation on few-shot downstream classification. 

||Model<br>Aircraft<br>Caltech-101<br>CIFAR-100<br>10-shot<br>20-shot<br>30-shot<br>5-shot<br>10-shot<br>30-shot<br>5-shot<br>10-shot<br>30-shot|
|---|---|
||AlexNet<br>23.3%<br>34.4%<br>41.4%<br>64.7%<br>73.6%<br>81.4%<br>29.7%<br>36.3%<br>49.3%<br>SimCLR<br>38.8%<br>56.9%<br>64.9%<br>81.7%<br>89.1%<br>94.5%<br>49.9%<br>60.2%<br>71.8%<br>TAN<br>22.8%<br>37.9%<br>46.0%<br>49.3%<br>66.4%<br>77.9%<br>21.3%<br>27.8%<br>42.4%|
||ViP<br>31.6%<br>53.1%<br>64.3%<br>68.1%<br>79.0%<br>88.9%<br>30.7%<br>41.0%<br>57.5%|
||DP-Cap<br>37.5%<br>57.9%<br>66.7%<br>70.3%<br>81.3%<br>90.0%<br>36.3%<br>46.3%<br>62.1%|



## A.3.3. DETAILS ABOUT LINEAR PROBING AND FINE-TUNING EVALUATION. 

Few-shot linear probing is accomplished using the Cyanure library (Mairal, 2019). We use the same hyper parameters as in Assran et al. (2022). We adapted the MAE (He et al., 2022) code base for full linear probing, and we use the same hyperparameters as in (Yu et al., 2023) (extract 12 layers of the image encoder, LARS optimizer (You et al., 2017) with base learning rate of 0.1, no weight decay and batch size of 16384). 

## **B. Additional Results** 

## **B.1. Additional experiments** 

**Impact of the initialization (V).** Our synthetic initialization for DP-Cap achieves less favorable results than the one from ViP reaches 50% (Yu et al., 2023); for instance, for full linear probing on ImageNet, they achieve 44% (Figure 2) and 50% respectively. However we have demonstrated that training with DP on top of synthetic initialization leads to significantly better results for DP-Cap compared to ViP for all the metrics; see Table 2, Table 10 and Figure 1. We observe that this superiority also appears when the models are trained from random initialization: as shown in Table 9, the improvement over ViP is even larger when training without synthetic initialization. 

**Fine-tuning (V).** In Table 10, we present DP-Cap’s performance in fine-tuning for few-shot evaluation. In contrast to the linear probing results shown in Table 2, the network is completely unfrozen. Therefore, we assess DP-Cap’s capabilities primarily as a network initialization. Similarly to the linear probing results, we note a significant improvement in all metrics compared to previous DP vision backbones. Note that, similarly to linear probing comparison in Figure 1, we compare to non-private model performance which provides information about the performance gap between private models and non-private models. For fair comparison, we evaluate on the same same datasets than Yu et al. (2023). 

**Captioning task (V-L).** We evaluate the image captioning performance of DP-Cap in comparison to non-private Cap. In Fig. 1(c), we present some (randomly chosen) captions generated by DP-Cap; more examples for DP-Cap and Cap can be 

16 

**Differentially Private Representation Learning via Image Captioning** 

Table 11: Ablation studies on the effect of model size for zero-shot prediction. 

|Model<br>Confg<br># parameters<br>DP?<br>Zero-shot<br>ImageNet-1k<br>CIFAR10<br>CIFAR100|Model<br>Confg<br># parameters<br>DP?<br>Zero-shot<br>ImageNet-1k<br>CIFAR10<br>CIFAR100|
|---|---|
|||
|Random Chance<br>-<br>-<br>-|0.1%<br>10%<br>1%|
|||
|Cap<br>Base<br>86.6M<br>✗<br>25.2%<br>90.0%<br>37.4%||
|||
|DP-Cap<br>Tiny<br>22.0M<br>✓<br>DP-Cap<br>Small<br>49.0M<br>✓<br>DP-Cap<br>Base<br>86.6M<br>✓<br>DP-Cap<br>Large<br>407.3M<br>✓|5.0%<br>46.5%<br>11.1%<br>6.9%<br>53.6%<br>17.1%<br>7.8%<br>54.4%<br>16.4%<br>9.2%<br>62.1%<br>24.0%|



Table 12: Captioning evaluation on the MS-COCO test set of Cap and DP-Cap. For “fine-tuned”, the model’s decoder is fine-tuned for one epoch on the MS-COCO train set (with the image encoder frozen). 

||Model|CIDEr score<br>original<br>fne-tuned|
|---|---|---|
||Cap<br>29.9<br>79.2||
||DP-Cap|15.7<br>51.3|



found in Appendix B.3. Qualitatively, DP-Cap seems to generate reasonably good captions, similar to the ones generated by Cap. We also compare the two models quantitatively using the CIDEr metric (Vedantam et al., 2015) to evaluate the generated captions on the MS-COCO test set, and the results are summarized in the last column of Table 3. As DP-Cap and Cap are only trained on noisy captions from LAION, the CIDEr metric on MS-COCO is relatively low for both models. Moreover, despite the similar performance between DP-Cap and Cap on ARO, the gap is much more significant for the captioning evaluation. Given these results, it is plausible that even though DP-Cap attains remarkably compositional understanding capabilities, its ability to generate text is still limited. 

We also fine-tune Cap and DP-Cap’s decoders (while freezing the encoder) for one epoch on the MS-COCO train set, and assess the improvement in CIDEr scores in Table 12 to showcase the quality of the image representations and decoder initialization from the pre-training stage. The captions in Figure 1 and Appendix B.3 are generated using models that were _not_ trained on MS-COCO. 

**What can we do with more compute budget?** We restricted training the DP-Cap model for a compute budget of 32 epochs on the Dedup-LAION-233M dataset for each of our models with _ε_ = 8. To fit the privacy budget while utilizing a batch size of 1.3 million and training for 32 epochs, RDP analysis yields _σ_ = 0 _._ 728. However, we anticipate that further increasing the compute budget can yield even better models up to a certain limit: With the same _ε_ and batch size, doubling the compute to 64 epochs only necessitates a 12% increase in _σ_ . This increase enables twice as many steps to be performed with only a marginal increase in effective noise, potentially allowing the model to converge to a better solution. 

In the absence of necessary compute for running this experiment, _we partially validate this hypothesis through the Total Amount of Noise (TAN) simulation_ , training for the same number of gradient steps and with the same SNR per step, but using a _×_ 32 smaller batch size and _×_ 32 smaller _σ_ to simulate at _×_ 32 lower compute. Our results in Table 13 indicate a significant performance improvement of 5% in 10-shot accuracy on ImageNet (compared to a similar simulation of the 32 epochs training). However, increasing the budget further to 128 epochs does not seem to enhance performance compared to the 64 epoch counterpart. Intuitively, the lower gradient SNR and larger number of gradient steps have opposite effects on optimization, and pushing past the “sweet spot” of training for 64 epochs at _σ_ = 0 _._ 81 results in noisy steps that are unproductive for model convergence. To surpass the performance of the 64-epoch, 1.3-million batch size DP-Cap model, training with an even larger batch size appears necessary. We emphasize again that this result is derived through TAN simulation, and actual, compute-intensive training is required to fully validate this assertion. 

17 

**Differentially Private Representation Learning via Image Captioning** 

Table 13: TAN simulation of the impact of the compute budget on the performance at fixed _B_ . 

|_σ_<br>0.81<br>0.95|_σ_<br>0.81<br>0.95|
|---|---|
|||
|Epochs<br>Effective noise_σ/B_<br>Predicted Final loss<br>Predicted 10-shot ImageNet|64 (_×_2)<br>128 (_×_4)<br>_×_1.12<br>_×_1.32<br>_−_0.2<br>_−_0.2<br>+5%<br>+5%|




![](raw/papers/2_LLM_NLP/LLM_paper_list/caption/icml_2024_Differentially_Private_Representation_Learning_via_Image_Captioning_images/icml_2024_Differentially_Private_Representation_Learning_via_Image_Captioning.pdf-0018-03.png)


Figure 5: At fixed (B, _σ_ , S), _ε_ drastically reduces with the dataset size. 

## **B.2. More on the Impact of dataset size and privacy parameters** 

**Dataset size.** We emphasize here (again) the importance of having enough training data to achieve a good privacy-utility trade-off with DP-SGD. As depicted in Figure 5, increasing the number of training samples _N_ while keeping the same number of equivalent DP-SGD steps ( _i.e._ , keeping batch size _B_ , noise _σ_ , and number of update steps _S_ constant) considerably reduces the privacy budget _ε_ . Equivalently, having more data allows for an increase in the number of equivalent DP-SGD steps at fixed _ε_ . Similar observations were also made by Tramer & Boneh (2020); McMahan et al. (2017). The abundance of pre-training data available for training foundation models thus proves highly compatible with DP requirements. 

**Batch size and** _σ_ **.** We wish to underscore the influence of batch size and _σ_ on both the computational budget and model performance. As highlighted in Section 3.2, for a given target _ε_ , elevating _σ_ beyond 0.5 allows training for significantly more steps. In Figure 6, the blue, orange and green lines show the batch size ( _B_ ) vs. compute trade-off ( _E_ ) at a given _σ_ . The lines are monotonically decreasing with _B_ , signifying that the number of epochs _E_ decreases when increasing _B_ . When maintaining a fixed privacy budget _ε_ = 8, even a marginal increase in _σ_ from 0 _._ 48 to 0 _._ 728 (from blue to orange) translates to a remarkable increase ranging from 100 (for small batch sizes) to 100,000 (for very large batch sizes) times more gradient steps. Thus it is favorable to increase _σ_ and _B_ at the same time for better model convergence. 

Meanwhile, doing so also incurs a higher computational cost: Under a 32-epoch budget on Dedup-LAION-233M with a batch size of 1 _._ 3 million, we had to cut the red curve in Figure 6, with _σ_ = 0 _._ 728. As outlined in Section 4.4, with twice this budget, we could have raised _σ_ to 0.81 (green curve), with simulations indicating that this would have substantially improved performance. Additionally, Section 3.2 underscores that increasing the batch size is pivotal for achieving a high SNR while maintaining reasonable privacy guarantees. It is also crucial to note that at fixed _ε_ , the compute budget is inversely proportional to the batch size. Therefore, increasing the batch size is beneficial for both SNR and computational 

18 

**Differentially Private Representation Learning via Image Captioning** 


![](raw/papers/2_LLM_NLP/LLM_paper_list/caption/icml_2024_Differentially_Private_Representation_Learning_via_Image_Captioning_images/icml_2024_Differentially_Private_Representation_Learning_via_Image_Captioning.pdf-0019-01.png)


**----- Start of picture text -----**<br>
 = 0.48<br> = 0.728<br>10 [4]  = 0.81<br>B=98k (ViP)<br>B=1.3M (DP-Cap)<br>10 [3]<br>10 [2]<br>10 [1]<br>10 [0]<br>10 1<br>10 [5] 10 [6] 10 [7]<br>Batch Size (B)<br>Compute/Epochs (E)<br>**----- End of picture text -----**<br>


Figure 6: All points correspond to _ε_ = 8 for a dataset of size _N_ = 233M. At fixed _ε_ and _σ_ , the number of epochs decreases as the batch size increases. 

efficiency. However, an excessively large batch size leads to fewer epochs and consequently a very limited number of training steps, which is detrimental to the training process (in addition to the difficulties of large batch training). For optimal privacy-utility-compute trade-off, a balance must be struck between computational resources, feasible batch size, and a reasonable number of training steps. 

## **B.3. Image Caption Examples** 

In Figures 7 and 8, we show images from the MS-COCO 2017 test set and their corresponding captions generated by human annotator, Cap, and DP-Cap. Images in Figure 7 are selected randomly, whereas images in Figure 8 are randomly selected from the top 10% CIDEr score examples for DP-Cap. Qualitatively, the human-generated captions are more precise, whereas the captions generated by Cap and DP-Cap are more generic and sometimes contain factual errors. This is to be expected since Cap and DP-Cap are trained on LAION with much noisier text description and were _not_ fine-tuned on MS-COCO. Nevertheless, DP-Cap still generates grammatically correct and (mostly) semantically coherent captions for unseen images. 

19 

**Differentially Private Representation Learning via Image Captioning** 


![](raw/papers/2_LLM_NLP/LLM_paper_list/caption/icml_2024_Differentially_Private_Representation_Learning_via_Image_Captioning_images/icml_2024_Differentially_Private_Representation_Learning_via_Image_Captioning.pdf-0020-01.png)


Figure 7: Captions of randomly selected images from the MS-COCO 2017 test set. 

20 

**Differentially Private Representation Learning via Image Captioning** 


![](raw/papers/2_LLM_NLP/LLM_paper_list/caption/icml_2024_Differentially_Private_Representation_Learning_via_Image_Captioning_images/icml_2024_Differentially_Private_Representation_Learning_via_Image_Captioning.pdf-0021-01.png)


Figure 8: Captions of images (randomly picked among the top 10% CIDEr score of DP-Cap) from the MS-COCO 2017 test set. 

21 

