Published as a conference paper at ICLR 2024 

## BECLR: BATCH ENHANCED CONTRASTIVE FEW-SHOT LEARNING 

**Stylianos Poulakakis-Daktylidis**[1] **& Hadi Jamali-Rad**[1] _[,]_[2] 

1Delft University of Technology (TU Delft), The Netherlands 

2Shell Global Solutions International B.V., Amsterdam, The Netherlands 

_{_ s.p.mrpoulakakis-daktylidis, h.jamalirad _}_ @tudelft.nl 

## ABSTRACT 

Learning quickly from very few labeled samples is a fundamental attribute that separates machines and humans in the era of deep representation learning. Unsupervised few-shot learning (U-FSL) aspires to bridge this gap by discarding the reliance on annotations at training time. Intrigued by the success of contrastive learning approaches in the realm of U-FSL, we structurally approach their shortcomings in both pretraining and downstream inference stages. We propose a novel Dynamic Clustered mEmory (DyCE) module to promote a highly separable latent representation space for _enhancing positive sampling_ at the pretraining phase and infusing implicit class-level insights into unsupervised contrastive learning. We then tackle the, somehow overlooked yet critical, issue of _sample bias_ at the fewshot inference stage. We propose an iterative Optimal Transport-based distribution Alignment (OpTA) strategy and demonstrate that it efficiently addresses the problem, especially in low-shot scenarios where FSL approaches suffer the most from sample bias. We later on discuss that DyCE and OpTA are two intertwined pieces of a novel end-to-end approach (we coin as BECLR), constructively magnifying each other’s impact. We then present a suite of extensive quantitative and qualitative experimentation to corroborate that BECLR sets a new state-of-the-art across ALL existing U-FSL benchmarks (to the best of our knowledge), and significantly outperforms the best of the current baselines (codebase available at GitHub). 

## 1 INTRODUCTION 

Achieving acceptable performance in deep representation learning comes at the cost of humongous data collection, laborious annotation, and excessive supervision. As we move towards more complex downstream tasks, this becomes increasingly prohibitive; in other words, supervised representation learning simply does not scale. In stark contrast, humans can quickly learn new tasks from a handful of samples, without extensive supervision. Few-shot learning (FSL) aspires to bridge this fundamental gap between humans and machines, using a suite of approaches such as metric learning (Wang et al., 2019; Bateni et al., 2020; Yang et al., 2020), meta-learning (Finn et al., 2017; Rajeswaran et al., 2019; Rusu et al., 2018), and probabilistic learning (Iakovleva et al., 2020; Hu et al., 2019; Zhang et al., 2021). FSL has shown promis- 


![](raw/papers/2_LLM_NLP/LLM_paper_list/few-shot/iclr_2024_BECLR_Batch_Enhanced_Contrastive_Few-Shot_Learning_images/iclr_2024_BECLR_Batch_Enhanced_Contrastive_Few-Shot_Learning.pdf-0001-10.png)


Figure 1: miniImageNet (5-way, 1-shot, left) and (5-way, 5-shot, right) accuracy in the U-FSL landscape. BECLR sets a new state-of-the-art in all settings by a significant margin. 

ing results in a supervised setting so far on a number of benchmarks (Hu et al., 2022; Singh & Jamali-Rad, 2022; Hu et al., 2023b); however, the need for supervision still lingers on. This has led to the emergence of a new divide called unsupervised FSL (U-FSL). The stages of U-FSL are the same as their supervised counterparts: pretraining on a large dataset of _base_ classes followed by fast adaptation and inference to unseen few-shot tasks (of _novel_ classes). The extra challenge here is the absence of labels during pretraining. U-FSL approaches have gained an upsurge of attention most recently owing to their practical significance and close ties to self-supervised learning. 

1 

Published as a conference paper at ICLR 2024 

The goal of pretraining in U-FSL is to learn a feature extractor (a.k.a encoder) to capture the global structure of the unlabeled data. This is followed by fast adaptation of the frozen encoder to unseen tasks typically through a simple linear classifier (e.g., a logistic regression classifier). The body of literature here can be summarized into two main categories: (i) meta-learning-based pretraining where synthetic few-shot tasks resembling the downstream inference are generated for episodic training of the encoder (Hsu et al., 2018; Khodadadeh et al., 2019; 2020); (ii) (non-episodic) transferlearning approaches where pretraining boils down to learning optimal representations suitable for downstream few-shot tasks (Medina et al., 2020; Chen et al., 2021b; 2022; Jang et al., 2022). Recent studies demonstrate that (more) complex meta-learning approaches are data-inefficient (Dhillon et al., 2019; Tian et al., 2020), and that the transfer-learning-based methods outperform their metalearning counterparts. More specifically, the state-of-the-art in this space is currently occupied by approaches based on _contrastive learning_ , from the transfer-learning divide, achieving top performance across a wide variety of benchmarks (Chen et al., 2021a; Lu et al., 2022). The underlying idea of contrastive representation learning (Chen et al., 2020a; He et al., 2020) is to attract “positive” samples in the representation space while repelling “negative” ones. To efficiently materialize this, some contrastive learning approaches incorporate memory queues to alleviate the need for larger batch sizes (Zhuang et al., 2019; He et al., 2020; Dwibedi et al., 2021; Jang et al., 2022). 

**Key Idea I:** _Going beyond instance-level contrastive learning._ Operating under the unsupervised setting, contrastive FSL approaches typically enforce consistency only at the _instance level_ , where each image within the batch and its augmentations correspond to a unique class, which is an unrealistic but seemingly unavoidable assumption. The pitfall here is that potential positive samples present within the same batch might then be repelled in the representation space, hampering the overall performance. We argue that infusing a semblance of class (or membership)-level insights into the unsupervised contrastive paradigm is essential. Our key idea to address this is extending the concept of memory queues by introducing inherent membership clusters represented by dynamically updated prototypes, while circumventing the need for large batch sizes. This enables the proposed pretraining approach to sample more meaningful positive pairs owing to a novel Dynamic Clustered mEmory (DyCE) module. While maintaining a fixed memory size (same as queues), DyCE efficiently constructs and dynamically updates separable memory clusters. 

**Key Idea II:** _Addressing inherent sample bias in FSL._ The base (pretraining) and novel (inference) classes are either mutually exclusive classes of the same dataset or originate from different datasets - both scenarios are investigated in this paper (in Section 5). This distribution shift poses a significant challenge at inference time for the swift adaptation to the novel classes. This is further aggravated due to access to only a few labeled (a.k.a _support_ ) samples within the few-shot task because the support samples are typically not representative of the larger unlabeled (a.k.a _query_ ) set. We refer to this phenomenon as _sample bias_ , highlighting that it is overlooked by most (U-)FSL baselines. To address this issue, we introduce an Optimal Transport-based distribution Alignment (OpTA) add-on module within the supervised inference step. OpTA imposes no additional learnable parameters, yet efficiently aligns the representations of the labeled support and the unlabeled query sets, right before the final supervised inference step. Later on in Section 5, we demonstrate that these two novel modules (DyCE and OpTA) are actually intertwined and amplify one another. Combining these two key ideas, we propose an end-to-end U-FSL approach coined as Batch-Enhanced Contrastive LeaRning (BECLR). Our main contributions can be summarized as follows: 

- I. We introduce BECLR to structurally address two key shortcomings of the prior art in U-FSL at pretraining and inference stages. At pretraining, we propose to infuse implicit class-level insights into the contrastive learning framework through a novel dynamic clustered memory (coined as DyCE). Iterative updates through DyCE help establish a highly separable partitioned latent space, which in turn promotes more meaningful positive sampling. 

- II. We then articulate and address the inherent sample bias in (U-)FSL through a novel add-on module (coined as OpTA) at the inference stage of BECLR. We show that this strategy helps mitigate the distribution shift between query and support sets. This manifests its significant impact in low-shot scenarios where FSL approaches suffer the most from sample bias. 

- III. We perform _extensive_ experimentation to demonstrate that BECLR sets a new state-of-theart in ALL established U-FSL benchmarks; e.g. miniImageNet (see Fig. 1), tieredImageNet, CIFAR-FS, FC100, outperforming ALL existing baselines by a significant margin (up to 14%, 12%, 15%, 5 _._ 5% in 1-shot settings, respectively), to the best of our knowledge. 

2 

Published as a conference paper at ICLR 2024 

## 2 RELATED WORK 

**Self-Supervised Learning (SSL).** It has been approached from a variety of perspectives (Balestriero et al., 2023). Deep metric learning methods (Chen et al., 2020a; He et al., 2020; Caron et al., 2020; Dwibedi et al., 2021), build on the principle of a contrastive loss and encourage similarity between semantically transformed views of an image. Redundancy reduction approaches (Zbontar et al., 2021; Bardes et al., 2021) infer the relationship between inputs by analyzing their cross-covariance matrices. Self-distillation methods (Grill et al., 2020; Chen & He, 2021; Oquab et al., 2023) pass different views to two separate encoders and map one to the other via a predictor. Most of these approaches construct a contrastive setting, where a symmetric or (more commonly) asymmetric Siamese network (Koch et al., 2015) is trained with a variant of the infoNCE(Oord et al., 2018) loss. 

**Unsupervised Few-Shot Learning (U-FSL).** The objective here is to pretrain a model from a large _unlabeled_ dataset of base classes, akin to SSL, but tailored so that it can quickly generalize to unseen downstream FSL tasks. Meta-learning approaches (Lee et al., 2020; Ye et al., 2022) generate synthetic learning episodes for pretraining, which mimic downstream FSL tasks. Here, PsCo (Jang et al., 2022) utilizes a student-teacher momentum network and optimal transport for creating pseudo-supervised episodes from a memory queue. Despite its elegant form, meta-learning has been shown to be data-inefficient in U-FSL (Dhillon et al., 2019; Tian et al., 2020). On the other hand, transfer-learning approaches (Li et al., 2022; Antoniou & Storkey, 2019; Wang et al., 2022a), follow a simpler non-episodic pretraining, focused on representation quality. Notably, contrastive learning methods, such as PDA-Net (Chen et al., 2021a) and UniSiam (Lu et al., 2022), currently hold the state-of-the-art. Our proposed approach also operates within the contrastive learning premise, but also employs a dynamic clustered memory module (DyCE) for infusing membership/class-level insights within the instance-level contrastive framework. Here, SAMPTransfer (Shirekar et al., 2023) takes a different path and tries to ingrain _implicit_ global membership-level insights through message passing on a graph neural network; however, the computational burden of this approach significantly hampers its performance with (and scale-up to) deeper backbones than Conv4. 

**Sample Bias in (U-)FSL.** Part of the challenge in (U-)FSL lies in the domain difference between base and novel classes. To make matters worse, estimating class distributions only from a few support samples is inherently biased, which we refer to as _sample bias_ . To address sample bias, Chen et al. (2021a) propose to enhance the support set with additional base-class images, Xu et al. (2022) project support samples farther from the task centroid, while Yang et al. (2021) use a calibrated distribution for drawing more support samples, yet all these methods are dependent on base-class characteristics. On the other hand, Ghaffari et al. (2021); Wang et al. (2022b) utilize Firth bias reduction to alleviate the bias in the logistic classifier itself, yet are prone to over-fitting. In contrast, the proposed OpTA module requires no fine-tuning and does not depend on the pretraining dataset. 

## 3 PROBLEM STATEMENT: UNSUPERVISED FEW-SHOT LEARNING 

We follow the most commonly adopted setting in the literature (Chen et al., 2021a;b; Lu et al., 2022; Jang et al., 2022), which consists of: an unsupervised pretraining, followed by a supervised inference (a.k.a fine-tuning) strategy. Formally, we consider a large unlabeled dataset _D_ tr = _{_ _**x** i}_ of so-called “base” classes for pretraining the model. The inference phase then involves transferring the model to unseen few-shot downstream tasks _Ti_ , drawn from a smaller labeled test dataset of so-called “novel” classes _D_ tst = _{_ ( _**x** i, yi_ ) _}_ , with _yi_ denoting the label of sample _**x** i_ . Each task _Ti_ is composed of two parts [ _S, Q_ ]: (i) the support set _S_ , from which the model learns to adapt to the novel classes, and (ii) the query set _Q_ , on which the model is evaluated. The support set _S_ = _{_ _**x**[s] i[, y] i[s][}][NK] i_ =1[is constructed] by drawing _K_ labeled random samples from _N_ different classes, resulting in the so-called ( _N_ -way, _K_ -shot) setting. The query set _Q_ = _{_ _**x**[q] j[}][NQ] j_ =1[contains] _[ NQ]_[ (with] _[ Q][>][K]_[) unlabeled samples.][The] base and novel classes are mutually exclusive, i.e., the distributions of _D_ tr and _D_ tst are different. 

## 4 PROPOSED METHOD: BECLR 

## 4.1 UNSUPERVISED PRETRAINING 

We build the unsupervised pretraining strategy of BECLR following contrastive representation learning. The core idea here is to efficiently attract “positive” samples (i.e., augmentations of the same image) in the representation space, while repelling “negative” samples. However, traditional contrastive learning approaches address this at the _instance level_ , where each image within the batch 

3 

Published as a conference paper at ICLR 2024 


![](raw/papers/2_LLM_NLP/LLM_paper_list/few-shot/iclr_2024_BECLR_Batch_Enhanced_Contrastive_Few-Shot_Learning_images/iclr_2024_BECLR_Batch_Enhanced_Contrastive_Few-Shot_Learning.pdf-0004-01.png)


**----- Start of picture text -----**<br>
DyCE [S]<br>DyCE [T]<br>**----- End of picture text -----**<br>


Figure 2: Overview of the proposed pretraining framework of BECLR. Two augmented views of the batch images _**X**[{][α,β][}]_ are both passed through a student-teacher network followed by the DyCE memory module. DyCE enhances the original batch with meaningful positives and dynamically updates the memory partitions. 

has to correspond to a unique class (a statistically unrealistic assumption!). As a result, potential positives present within a batch might be treated as negatives, which can have a detrimental impact on performance. A common strategy to combat this pitfall (also to avoid prohibitively large batch sizes) is to use a memory queue (Wu et al., 2018; Zhuang et al., 2019; He et al., 2020). Exceptionally, PsCo (Jang et al., 2022) uses optimal transport to sample from a first-in-first-out memory queue and generate pseudo-supervised few-shot tasks in a meta-learning-based framework, whereas NNCLR (Dwibedi et al., 2021) uses a nearest-neighbor approach to generate more meaningful positive pairs in the contrastive loss. However, these memory queues are still oblivious to global memberships (i.e., class-level information) in the latent space. Instead, we propose to infuse membership/classlevel insights through a novel memory module (DyCE) within the pretraining phase of the proposed end-to-end approach: BECLR. Fig. 2 provides a schematic illustration of the proposed contrastive pretraining framework within BECLR, and Fig. 3 depicts the mechanics of DyCE. 

**Pretraining Strategy of BECLR.** The pretraining pipeline of BECLR is summarized in Al- **Algorithm 1:** Pretraining of BECLR gorithm 1, and a Pytorch-like pseudo-code can **Require:** ˆ _A_ , _θ_ , _ψ_ , _fθ_ , _fψ_ , _gθ_ , _gψ_ , _hθ_ , _µ_ , _m_ , DyCE( _·_ ) be found in Appendix E. Let us now walk you **12** _**XZ**[S]_ ==� _ζh[α] θ_ ( _◦_ _**X** g_ ) _θ, ζ◦[β] f_ ( _θ_ _**X**_ � _µ_ )(� _**X**_[ˆ] for)� _ζ[α] , ζ[β] ∼A_ through the algorithm. Let _ζ[a] , ζ[b] ∼A_ be two **3** _**Z**_ ˆ _[T]_ =ˆ _gψ ◦ fψ_ ( _**X**_[ˆ] ) randomly sampled data augmentations from the **4** _**Z**[S] ,_ _**Z**[T]_ = DyCE _[S]_ ( _**Z**[S]_ ) _,_ DyCE _[T]_ ( _**Z**[T]_ ) set of all available augmentations, _A_ . The mini- **5** Compute loss: _Lcontr._ using Eq. 3 on _**Z**_[ˆ] _[S] ,_ _**Z**_[ˆ] _[T]_ batch can then be denoted as _**X**_[ˆ] = [ˆ _**x** i_ ][2] _i_ =1 _[B]_[=] **6** Update: _θ ← θ −∇Lcontr., ψ ← mψ_ + (1 _− m_ ) _θ_ �[ _ζ[a]_ ( _**x** i_ )] _[B] i_ =1 _[,]_[ [] _[ζ][b]_[(] _**[x]**[i]_[)]] _[B] i_ =1�, where _B_ the original batch size (line 1, Algorithm 1). As shown in Fig. 2, we adopt a student-teacher (a.k.a Siamese) asymmetric momentum architecture similar to Grill et al. (2020); Chen & He (2021). Let _µ_ ( _·_ ) be a patch-wise masking operator, _f_ ( _·_ ) the backbone feature extractor (ResNets (He et al., 2016) in our case), and _g_ ( _·_ ), _h_ ( _·_ ) projection and prediction multi-layer perceptrons (MLPs), respectively. The teacher weight parameters ( _ψ_ ) are an exponential moving averaged ( _EMA_ ) version of the student parameters ( _θ_ ), i.e., _ψ ← mψ_ + (1 _− m_ ) _θ_ , as in Grill et al. (2020), where _m_ the momentum hyperparameter, while _θ_ are updated through stochastic gradient descent (SGD). The student and teacher representations _**Z**[S]_ and _**Z**[T]_ (both of size 2 _B × d_ , with _d_ the latent embedding dimension) can then be obtained as follows: _**Z**[S]_ = _hθ ◦ gθ ◦ fθ_ � _µ_ ( _**X**_[ˆ] )� _,_ _**Z**[T]_ = _gψ ◦ fψ_ ( _**X**_[ˆ] ) (lines 2,3). 

Upon extracting _**Z**[S]_ and _**Z**[T]_ , they are fed into the proposed dynamic memory module (DyCE), where enhanced versions of the batch representations _**Z**_[ˆ] _[S]_ , _**Z**_[ˆ] _[T]_ (both of size 2 _B_ ( _k_ + 1) _× d_ , with _k_ denoting the number of selected nearest neighbors) are generated (line 4). Finally, we apply the contrastive loss in Eq. 3 on the enhanced batch representations _**Z**_ **[ˆ]** _**[S]**_ , _**Z**_ **[ˆ]** _**[T]**_ (line 5). Upon finishing unsupervised pretraining, only the student encoder ( _fθ_ ) is kept for the subsequent inference stage. 

**Dynamic Clustered Memory (DyCE).** _How do we manage to enhance the batch with meaningful true positives in the absence of labels?_ We introduce DyCE: a novel dynamically updated clustered memory to moderate the representation space during training, while infusing a semblance of classcognizance. We demonstrate later on in Section 5 that the design choices in DyCE have a significant impact on both pretraining performance as well as the downstream few-shot classification. 

4 

Published as a conference paper at ICLR 2024 

Let us first establish some notation. We consider a memory unit _M_ capable of storing up to _M_ latent embeddings (each of size _d_ ). To accommodate clustered memberships within DyCE, we consider up to _P_ partitions (or clusters) _Pi_ in _M_ = [ _P_ 1 _, . . . , PP_ ], each of which is represented by a prototype stored in **Γ** = [ _**γ**_ 1 _, . . . ,_ _**γ** P_ ]. Each prototype _**γ** i_ (of size 1 _× d_ ) is the average of the latent embeddings stored in partition _Pi_ . As shown in Fig. 3 and in Algorithm 2, DyCE consists of two informational paths: (i) the top- _k_ neighbor selection and batch enhancement path (bottom branch of the figure), which uses the current state of _M_ and **Γ** ; (ii) the iterative memory updating via dynamic clustering path (top branch). DyCE takes as input student or teacher embeddings (we use _**Z**_ here, for brevity) and returns the enhanced versions _**Z**_[ˆ] . We also allow for an adaptation period epoch _<_ epochthr (empirically 20-50 epochs), during which path (i) is not activated and the training batch is not enhanced. To further elaborate, path (i) starts with assigning each _**z** i ∈_ _**Z**_ to its nearest (out of _P_ ) memory prototype _**γ** νi_ based on the Euclidean distance _⟨·⟩_ . _**ν**_ is a vector of indices (of size 2 _B×_ 1) that contains these prototype assignments for all batch embeddings (line 4, Algorithm 2). Next (in line 5), we se- 


![](raw/papers/2_LLM_NLP/LLM_paper_list/few-shot/iclr_2024_BECLR_Batch_Enhanced_Contrastive_Few-Shot_Learning_images/iclr_2024_BECLR_Batch_Enhanced_Contrastive_Few-Shot_Learning.pdf-0005-02.png)


**----- Start of picture text -----**<br>
DyCE<br>**----- End of picture text -----**<br>


Figure 3: Overview of the proposed dynamic clustered memory (DyCE) and its two informational paths. 

## **Algorithm 2:** DyCE 


![](raw/papers/2_LLM_NLP/LLM_paper_list/few-shot/iclr_2024_BECLR_Batch_Enhanced_Contrastive_Few-Shot_Learning_images/iclr_2024_BECLR_Batch_Enhanced_Contrastive_Few-Shot_Learning.pdf-0005-05.png)


**----- Start of picture text -----**<br>
Require: epochthr,  M ,  Γ ,  Z ,  B ,  k<br>1 if |M| = M then<br>2 / / Path (i) : top-k and batch enhancement)<br>3 if epoch  ≥ epochthr then<br>4 ν = [ νi ] [2] i =1 [B] [=] �argmin j∈ [ P  ] ⟨ z i,  γ j ⟩ �2 i =1 B<br>5 Y ˆ i ← top-k� { z i, Pνi } � , ∀i ∈ [2 B ]<br>6 Z = [ Z ,  Y 1 , . . . ,  Y 2 B ]<br>7 / / Path (ii) : iterative memory updating<br>8 Find optimal plan between  Z ,  Γ :  π [∗] ← Solve Eq. 2<br>9 Update  M  with new  Z :  M ← update( M,  π [∗] ,  Z )<br>10 Discard 2 B  oldest batch embeddings: dequeue( M )<br>11 else<br>12 Store new batch:  M ← store( M,  Z )<br>Return: Z [ˆ]<br>**----- End of picture text -----**<br>


lect the _k_ most similar memory embeddings to _**z** i_ from the memory partition corresponding to its assigned prototype ( _Pνi_ ) and store them in _**Y** i_ (of size _k ×d_ ). Finally (in line 6), all _**Y** i, ∀i ∈_ [2 _B_ ] are concatenated into the enhanced batch _**Z**_[ˆ] = [ _**Z** ,_ _**Y**_ **1** _, . . . ,_ _**Y**_ 2 _B_ ] of size _L × d_ (where _L_ = 2 _B_ ( _k_ + 1)). Path (ii) addresses the iterative memory updating by casting it into an optimal transport problem (Cuturi, 2013) given by: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/few-shot/iclr_2024_BECLR_Batch_Enhanced_Contrastive_Few-Shot_Learning_images/iclr_2024_BECLR_Batch_Enhanced_Contrastive_Few-Shot_Learning.pdf-0005-07.png)


to find a transport plan _**π**_ (out of **Π** ) mapping _**Z**_ to **Γ** . Here, _**r** ∈_ R[2] _[B]_ denotes the distribution of batch embeddings [ _**z** i_ ][2] _i_ =1 _[B]_[,] _**[ c]**[ ∈]_[R] _[P]_[is the distribution of memory cluster prototypes][ [] _**[γ]**[i]_[]] _[P] i_ =1[.][The last] two conditions in Eq. 1 enforce equipartitioning (i.e., uniform assignment) of _**Z**_ into the _P_ memory partitions/clusters. Obtaining the optimal transport plan, _**π**[⋆]_ , can then be formulated as: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/few-shot/iclr_2024_BECLR_Batch_Enhanced_Contrastive_Few-Shot_Learning_images/iclr_2024_BECLR_Batch_Enhanced_Contrastive_Few-Shot_Learning.pdf-0005-09.png)


and solved using the Sinkhorn-Knopp (Cuturi, 2013) algorithm (line 8). Here, _**D**_ is a pairwise distance matrix between the elements of _**Z**_ and **Γ** (of size 2 _B × P_ ), _⟨·⟩F_ denotes the Frobenius dot product, _ε_ is a regularisation term, and H( _·_ ) is the Shannon entropy. Next (in line 9), we add the embeddings of the current batch _**Z**_ to _M_ and use _**π**[⋆]_ for updating the partitions _Pi_ and prototypes **Γ** (using EMA for updating). Finally, we discard the 2 _B_ oldest memory embeddings (line 10). 

**Loss Function.** The popular infoNCE loss (Oord et al., 2018) is the basis of our loss function, yet recent studies (Poole et al., 2019; Song & Ermon, 2019) have shown that it is prone to high bias, when the batch size is small. To address this, we adopt a variant of infoNCE, which maximizes the same mutual information objective, but has been shown to be less biased (Lu et al., 2022): 


![](raw/papers/2_LLM_NLP/LLM_paper_list/few-shot/iclr_2024_BECLR_Batch_Enhanced_Contrastive_Few-Shot_Learning_images/iclr_2024_BECLR_Batch_Enhanced_Contrastive_Few-Shot_Learning.pdf-0005-12.png)


where _τ_ is a temperature parameter, d[ _·_ ] is the negative cosine similarity, _λ_ is a weighting hyperparameter, _L_ is the enhanced batch size and _**z** i_[+][stands for the latent embedding of the positive sample,] 

5 

Published as a conference paper at ICLR 2024 


![](raw/papers/2_LLM_NLP/LLM_paper_list/few-shot/iclr_2024_BECLR_Batch_Enhanced_Contrastive_Few-Shot_Learning_images/iclr_2024_BECLR_Batch_Enhanced_Contrastive_Few-Shot_Learning.pdf-0006-01.png)


**----- Start of picture text -----**<br>
OpTA<br>**----- End of picture text -----**<br>


Figure 4: Overview of the inference strategy of BECLR. Given a test episode, the support ( _S_ ) and query ( _Q_ ) sets are passed to the pretrained feature extractor ( _fθ_ ). OpTA aligns support prototypes and query features. 

corresponding to sample _i_ . Following (Chen & He, 2021), to further boost training performance, we pass both views through both the student and the teacher. The first term in Eq. 3 operates on positive pairs, and the second term pushes each representation away from all other batch representations. 

## 4.2 SUPERVISED INFERENCE 

Supervised inference (a.k.a fine-tuning) usually combats the distribution shift between training and test datasets. However, the limited number of support samples (in FSL tasks) at test time leads to a significant performance degradation due to the so-called _sample bias_ (Cui & Guo, 2021; Xu et al., 2022). This issue is mostly disregarded in recent state-of-the-art U-FSL baselines (Chen et al., 2021a; Lu et al., 2022; Hu et al., 2023a). As part of the inference phase of BECLR, we propose a simple, yet efficient, add-on module (coined as OpTA) for aligning the distributions of the query ( _Q_ ) and support ( _S_ ) sets, to structurally address sample bias. Notice that OpTA is not a learnable module and that there are no model updates nor any fine-tuning involved in the inference stage of BECLR. 

**Optimal Transport-based Distribution Alignment (OpTA).** Let _T_ = _S ∪Q_ be a downstream few-shot task. We first extract the support _**Z**[S]_ = _fθ_ ( _S_ ) (of size _NK × d_ ) and query _**Z**[Q]_ = _fθ_ ( _Q_ ) (of size _NQ × d_ ) embeddings and calculate the support set prototypes _**P**[S]_ (class averages of size _N × d_ ). Next, we find the optimal transport plan ( _**π**[⋆]_ ) between _**P**[S]_ and _**Z**[Q]_ using Eq. 2, with _**r** ∈_ R _[NQ]_ the distribution of _**Z**[Q]_ and _**c** ∈_ R _[N]_ the distribution of _**P**[S]_ . Finally, we use _**π**[⋆]_ to map the support set prototypes onto the region occupied by the query embeddings: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/few-shot/iclr_2024_BECLR_Batch_Enhanced_Contrastive_Few-Shot_Learning_images/iclr_2024_BECLR_Batch_Enhanced_Contrastive_Few-Shot_Learning.pdf-0006-07.png)


where _**π**_ ˆ _[⋆]_ is the normalized transport plan and _**P**_[ˆ] _[S]_ are the transported support prototypes. Finally, we fit a logistic regression classifier on _**P**_[ˆ] _[S]_ to infer on the unlabeled query set. We show in Section 5 that OpTA successfully minimizes the distribution shift (between support and query sets) and contributes to the overall significant performance margin BECLR offers against the best existing baselines. Note that we iteratively perform _δ_ consecutive passes of OpTA, where _**P**_[ˆ] _[S]_ acts as the input of the next pass. Notably, OpTA can straightforwardly be applied on top of any U-FSL approach. An overview of OpTA and the proposed inference strategy is illustrated in Fig. 4. 

**Remark:** OpTA operates on two distributions and relies on the total number of unlabeled query samples being larger than the total number of labeled support samples ( _|Q| > |S|_ ) for reasonable distribution mapping, which is also the standard convention in the U-FSL literature. That said, OpTA would still perform on imbalanced FSL tasks as long as the aforementioned condition is met. 

## 5 EXPERIMENTS 

In this section, we rigorously study the performance of the proposed approach both quantitatively as well as qualitatively by addressing the following three main questions: [ **Q1** ] How does BECLR perform against the state-of-the-art for _in-domain_ and _cross-domain_ settings? [ **Q2** ] Does DyCE affect pretraining performance by establishing _separable memory partitions_ ? [ **Q3** ] Does OpTA address the _sample bias_ via the proposed distribution alignment strategy? 

We use PyTorch (Paszke et al., 2019) for all implementations. Elaborate implementation and training details are discussed in the supplementary material, in Appendix A. 

**Benchmark Datasets.** We evaluate BECLR in terms of its in-domain performance on the two most widely adopted few-shot image classification datasets: miniImageNet (Vinyals et al., 2016) and tieredImageNet (Ren et al., 2018). Additionally, for the in-domain setting, we also evaluate on two 

6 

Published as a conference paper at ICLR 2024 

Table 1: Accuracies (in % _±_ std) on miniImageNet and tieredImageNet compared against unsupervised (Unsup.) and supervised (Sup.) baselines. Backbones: RN: Residual network. _[†]_ : denotes our reproduction. _[∗]_ : denotes extra synthetic training data used. Style: **best** and second best. 

||||**miniImageNet**|**miniImageNet**|**tieredImageNet**|**tieredImageNet**|
|---|---|---|---|---|---|---|
|**Method**|**Backbone**|**Setting**|**5-way 1-shot**|**5-way 5-shot**|**5-way 1-shot**|**5-way 5-shot**|
|SwAV_†_ (Caron et al.,2020)<br>NNCLR_†_ (Dwibedi et al.,2021)<br>CPNWCP (Wang et al.,2022a)<br>HMS (Ye et al.,2022)<br>SAMPTransfer_†_ (Shirekar et al.,2023)<br>PsCo_†_ (Jang et al.,2022)<br>UniSiam + dist (Lu et al.,2022)<br>Meta-DM + UniSiam + dist_∗_(Hu et al.,2023a)<br>MetaOptNet (Lee et al.,2019)<br>Transductive CNAPS(Bateni et al.,2022)|RN18<br>RN18<br>RN18<br>RN18<br>RN18<br>RN18<br>RN18<br>RN18|Unsup.<br>Unsup.<br>Unsup.<br>Unsup.<br>Unsup.<br>Unsup.<br>Unsup.<br>Unsup.|59.84 ± 0.52<br>63.33 ± 0.53<br>53.14 ± 0.62<br>58.20 ± 0.23<br>45.75 ± 0.77<br>47.24 ± 0.76<br>64.10 ± 0.36<br>65.64 ± 0.36|78.23 ± 0.26<br>80.75 ± 0.25<br>67.36 ± 0.5<br>75.77 ± 0.16<br>68.33 ± 0.66<br>65.48 ± 0.68<br>82.26 ± 0.25<br>83.97 ± 0.25|65.26 ± 0.53<br>65.46 ± 0.55<br>45.00 ± 0.19<br>58.42 ± 0.25<br>42.32 ± 0.75<br>54.33 ± 0.54<br>67.01 ± 0.39<br>67.11 ± 0.40|81.73 ± 0.24<br>81.40 ± 0.27<br>62.96 ± 0.19<br>75.85 ± 0.18<br>53.45 ± 0.76<br>69.73 ± 0.49<br>84.47 ± 0.28<br>84.39 ± 0.28<br>81.56 ± 0.53<br>81.80 ± 0.70|
||RN18<br>RN18|Sup.<br>Sup.|64.09 ± 0.62<br>55.60 ± 0.90|80.00 ± 0.45<br>73.10 ± 0.70|65.99 ± 0.72<br>65.90 ± 1.10||
|**BECLR (Ours)**|RN18|Unsup.|**75.74** ± 0.62|**84.93** ± 0.33|**76.44** ± 0.66|**84.85** ± 0.37|
|PDA-Net (Chen et al.,2021a)<br>UniSiam + dist (Lu et al.,2022)<br>Meta-DM + UniSiam + dist_∗_(Hu et al.,2023a)|RN50<br>RN50<br>RN50|Unsup.<br>Unsup.<br>Unsup.|63.84 ± 0.91<br>65.33 ± 0.36<br>66.68 ± 0.36|83.11 ± 0.56<br>83.22 ± 0.24<br>85.29 ± 0.23|69.01 ± 0.93<br>69.60 ± 0.38<br>69.61 ± 0.38|84.20 ± 0.69<br>86.51 ± 0.26<br>86.53 ± 0.26|
|**BECLR (Ours)**|RN50|Unsup.|**80.57** ± 0.57|**87.82** ± 0.29|**81.69** ± 0.61|**87.86** ± 0.32|



Table 3: Accuracies (in % _±_ std) on miniImageNet → CDFSL. _[†]_ : our reproduc. Style: **best** and second best. 

||**ChestX**|**ChestX**|**ISIC**|**ISIC**|**EuroSAT**|**EuroSAT**|**CropDiseases**|**CropDiseases**|
|---|---|---|---|---|---|---|---|---|
|**Method**|**5 way 5-shot**|**5 way 20-shot**|**5 way 5-shot**|**5 way 20-shot**|**5 way 5-shot**|**5 way 20-shot**|**5 way 5-shot**|**5 way 20-shot**|
|SwAV_†_ (Caron et al.,2020)<br>NNCLR_†_ (Dwibedi et al.,2021)<br>SAMPTransfer (Shirekar et al.,2023)<br>PsCo (Jang et al.,2022)<br>UniSiam + dist (Lu et al.,2022)<br>ConFeSS(Das et al.,2021)|25.70 ± 0.28<br>25.74 ± 0.41<br>26.27 ± 0.44<br>24.78 ± 0.23<br>**28.18** ± 0.45<br>27.09|30.41 ± 0.25<br>29.54 ± 0.45<br>**34.15** ± 0.50<br>27.69 ± 0.23<br>**34.58** ± 0.46<br>33.57|40.69 ± 0.34<br>38.85 ± 0.56<br>47.60 ± 0.59<br>44.00 ± 0.30<br>45.65 ± 0.58<br>**48.85**|49.03 ± 0.30<br>47.82 ± 0.53<br>**61.28** ± 0.56<br>54.59 ± 0.29<br>56.54 ± 0.5<br>60.10|84.82 ± 0.24<br>83.45 ± 0.57<br>85.55 ± 0.60<br>81.08 ± 0.35<br>86.53 ± 0.47<br>84.65|90.77 ± 0.26<br>90.80 ± 0.39<br>88.52 ± 0.50<br>87.65 ± 0.28<br>93.24 ± 0.30<br>90.40|88.64 ± 0.26<br>90.76 ± 0.57<br>91.74 ± 0.55<br>88.24 ± 0.31<br>92.05 ± 0.50<br>88.88|95.11 ± 0.21<br>95.37 ± 0.37<br>96.36 ± 0.28<br>94.95 ± 0.18<br>96.83 ± 0.27<br>95.34|
|**BECLR (Ours)**|**28.46** ± 0.23|**34.21** ± 0.25|44.48 ± 0.31|56.89 ± 0.29|**88.55** ± 0.23|**93.92** ± 0.14|**93.65** ± 0.25|**97.72** ± 0.13|



curated versions of CIFAR-100 (Krizhevsky et al., 2009) for FSL, i.e., CIFAR-FS and FC100. Next, we evaluate BECLR in cross-domain settings on the Caltech-UCSD Birds (CUB) dataset (Welinder et al., 2010) and a more recent cross-domain FSL (CDFSL) benchmark (Guo et al., 2020). For crossdomain experiments, miniImageNet is used as the pretraining (source) dataset and ChestX (Wang et al., 2017), ISIC (Codella et al., 2019), EuroSAT (Helber et al., 2019) and CropDiseases (Mohanty et al., 2016) (in Table 3) and CUB (in Table 12 in the Appendix), as the inference (target) datasets. 

## 5.1 EVALUATION RESULTS 

We report test accuracies with 95% confidence intervals over 2000 test episodes, each with _Q_ = 15 query shots per class, for all datasets, as is most commonly adopted in the literature (Chen et al., 2021a; 2022; Lu et al., 2022). The performance on miniImageNet, tieredImageNet, CIFAR-FS, FC100 and miniImageNet _→_ CUB is evaluated on (5-way, _{_ 1 _,_ 5 _}_ -shot) classification tasks, whereas for miniImageNet _→_ CDFSL we test on (5-way, _{_ 5 _,_ 20 _}_ -shot) tasks, as is customary across the literature (Guo et al., 2020; Ericsson et al., 2021). We assess BECLR’s performance against a wide variety of baselines ranging from (i) established SSL baselines (Chen et al., 2020a; Grill et al., 2020; Caron et al., 2020; Zbontar et al., 2021; Chen & He, 2021; Dwibedi et al., 2021) to (ii) state-of-theart U-FSL approaches (Chen et al., 2021a; Lu et al., 2022; Shirekar et al., 2023; Chen et al., 2022; Hu et al., 2023a; Jang et al., 2022), as well as (iii) against a set of competitive supervised baselines (Rusu et al., 2018; Gidaris et al., 2019; Lee et al., 2019; Bateni et al., 2022). 

[ **A1-a** ] **In-Domain Setting.** The results for miniImageNet and tieredImageNet in the (5-way, _{_ 1 _,_ 5 _}_ -shot) settings are reported in Table 1. Regardless of backbone depth, BECLR sets a _new state-of-the-art_ on both datasets, showing up to a 14% and 2 _._ 5% gain on miniImageNet over the prior art of U-FSL for the 1 and 5-shot settings, respectively. The results on tiered- 

Table 2: Accuracies in (% _±_ std) on CIFAR-FS and FC100 in (5-way, _{_ 1 _,_ 5 _}_ -shot). Style: **best** and second best. 

||**CIFAR-FS**|**CIFAR-FS**|**FC100**|**FC100**|
|---|---|---|---|---|
|**Method**|**1-shot**|**5-shot**|**1-shot**|**5-shot**|
|SimCLR (Chen et al.,2020a)<br>MoCo v2 (Chen et al.,2020b)<br>LF2CS (Li et al.,2022)<br>Barlow Twins (Zbontar et al.,2021)<br>HMS (Ye et al.,2022)<br>DeepEigenmaps(Chen et al.,2022)|54.56 ± 0.19<br>52.73 ± 0.20<br>55.04 ± 0.72<br>-<br>54.65 ± 0.20<br>-|71.19 ± 0.18<br>67.81 ± 0.19<br>70.62 ± 0.57<br>-<br>73.70 ± 0.18<br>-|36.20 ± 0.70<br>37.70 ± 0.70<br>37.20 ± 0.70<br>37.90 ± 0.70<br>37.88 ± 0.16<br>39.70 ± 0.70|49.90 ± 0.70<br>53.20 ± 0.70<br>52.80 ± 0.60<br>54.10 ± 0.60<br>53.68 ± 0.18<br>57.90 ± 0.70|
|**BECLR (Ours)**|**70.39** ± 0.62|**81.56** ± 0.39|**45.21** ± 0.50|**60.02** ± 0.43|



ImageNet also highlight a considerable performance margin. Interestingly, BECLR even outperforms U-FSL baselines trained with extra (synthetic) training data, sometimes distilled from deeper backbones, also the cited supervised baselines. Table 2 provides further insights on (the less commonly adopted) CIFAR-FS and FC100 benchmarks, showing a similar trend with up to 15% and 8% in 1 and 5-shot settings, respectively, for CIFAR-FS, and 5 _._ 5% and 2% for FC100. 

7 

Published as a conference paper at ICLR 2024 


![](raw/papers/2_LLM_NLP/LLM_paper_list/few-shot/iclr_2024_BECLR_Batch_Enhanced_Contrastive_Few-Shot_Learning_images/iclr_2024_BECLR_Batch_Enhanced_Contrastive_Few-Shot_Learning.pdf-0008-01.png)


Figure 5: BECLR outperforms all baselines, in terms of U-FSL performance on miniImageNet, even without OpTA. 


![](raw/papers/2_LLM_NLP/LLM_paper_list/few-shot/iclr_2024_BECLR_Batch_Enhanced_Contrastive_Few-Shot_Learning_images/iclr_2024_BECLR_Batch_Enhanced_Contrastive_Few-Shot_Learning.pdf-0008-03.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/few-shot/iclr_2024_BECLR_Batch_Enhanced_Contrastive_Few-Shot_Learning_images/iclr_2024_BECLR_Batch_Enhanced_Contrastive_Few-Shot_Learning.pdf-0008-04.png)


Figure 6: The dynamic updates of DyCE Figure 7: OpTA produces allow the memory to evolve into a highly an increasingly larger perforseparable partitioned latent space. Clusters mance boost as the pretrainare denoted by different (colors, markers). ing feature quality increases. 

[ **A1-b** ] **Cross-Domain Setting.** Following Guo et al. (2020), we pretrain on miniImageNet and evaluate on CDFSL, the results of which are summarized in Tables 3. BECLR again sets a new state-of-the-art on ChestX, EuroSAT, and CropDiseases, and remains competitive on ISIC. Notably, the data distributions of ChestX and ISIC are considerably different from that of miniImageNet. We argue that this influences the embedding quality for the downstream inference, and thus, the efficacy of OpTA in addressing sample bias. Extended versions of Tables 1-3 are found in Appendix C. 

[ **A1-c** ] **Pure Pretraining and OpTA.** To substantiate the impact of the design choices in BECLR, we compare against some of the most influential contrastive SSL approaches: SwAV, SimSiam, NNCLR, and the prior U-FSL state-of-the-art: UniSiam (Lu et al., 

Table 4: BECLR outperforms enhanced prior art with OpTA. 

|Table 4: BECLRoutperf|orms enhanced prior art withOpTA.|orms enhanced prior art withOpTA.|
|---|---|---|
||**miniImageNet**<br>**tieredImageNet**||
|**Method**|**1 shot**<br>**5 shot**|**1 shot**<br>**5 shot**|
|CPNWCP+OpTA(Wang et al.,2022a)<br>HMS+OpTA(Ye et al.,2022)<br>PsCo+OpTA(Jang et al.,2022)<br>UniSiam+OpTA (Lu et al.,2022)|60.45 ± 0.81<br>75.84 ± 0.56<br>69.85 ± 0.42<br>80.77 ± 0.35<br>52.89 ± 0.71<br>67.42 ± 0.54<br>72.54 ± 0.61<br>82.46 ± 0.32|55.05 ± 0.31<br>72.91 ± 0.26<br>71.75 ± 0.43<br>81.32 ± 0.34<br>57.46 ± 0.59<br>70.70 ± 0.45<br>73.37 ± 0.64<br>73.37 ± 0.64|
|**BECLR (Ours)**|**75.74 ± 0.62**<br>**84.93 ± 0.33**<br>**76.44 ± 0.66**<br>**84.85 ± 0.37**||



2022), in terms of pure pretraining performance, by directly evaluating the pretrained model on downstream FSL tasks (i.e., no OpTA and no fine-tuning). Fig. 5 summarizes this comparison for various network depths in the (5-way, _{_ 1 _,_ 5 _}_ -shot) settings on miniImageNet. BECLR again outperforms all U-FSL/SSL frameworks for all backbone configurations, even without OpTA. As an additional study, in Table 4 we take the opposite steps by plugging in OpTA on a suite of recent prior art in U-FSL. The results demonstrate two important points: (i) OpTA is in fact agnostic to the choice of pretraining method, having considerable impact on downstream performance, and (ii) there still exists a margin between enhanced prior art and BECLR, corroborating that it is not just OpTA that has a meaningful effect but also DyCE and our pretraining methodology. 

[ **A2** ] **Latent Memory Space Evolution.** As a qualitative demonstration, we visualize 30 memory embeddings from 25 partitions _Pi_ within DyCE for the initial (left) and final (right) state of the latent memory space ( _M_ ). The 2-D UMAP plots in Fig. 6 provide qualitative evidence of a significant improvement in terms of cluster separation, as training progresses. To quantitatively substantiate this finding, the quality of the memory clusters is also measured by the DBI score (Davies & Bouldin, 1979), with a lower DBI indicating better inter-cluster separation and intra-cluster tightness. The DBI value is significantly lower between partitions _Pi_ in the final state of _M_ , further corroborating DyCE’s ability to establish highly separable and meaningful partitions. 

[ **A3-a** ] **Impact of OpTA.** We visualize the UMAP projections for a randomly sampled (3-way, 1- shot) miniImageNet episode. Fig. 8 illustrates the original _**P**[S]_ (left) and transported _**P**_[ˆ] _[S]_ (right) support prototypes (♦), along with the query set embeddings _**Z**[Q]_ ( _•_ ) and their kernel density distributions (in contours). As can be seen, the original prototypes are highly biased and deviate from the latent query distributions. OpTA pushes the trans- 


![](raw/papers/2_LLM_NLP/LLM_paper_list/few-shot/iclr_2024_BECLR_Batch_Enhanced_Contrastive_Few-Shot_Learning_images/iclr_2024_BECLR_Batch_Enhanced_Contrastive_Few-Shot_Learning.pdf-0008-13.png)


**----- Start of picture text -----**<br>
Before OpTA After OpTA<br>**----- End of picture text -----**<br>


Figure 8: OpTA addresses sample bias, reducing the distribution shift between support and query sets. 

ported prototypes much closer to the query distributions (contour centers), effectively diminishing sample bias, resulting in significantly higher accuracy in the corresponding episode. 

[ **A3-b** ] **Relation between Pretraining and Inference.** OpTA operates under the assumption that the query embeddings _**Z**[Q]_ are representative of the actual class distributions. As such, we argue that its efficiency depends on the quality of the extracted features through the pretrained encoder. 

8 

Published as a conference paper at ICLR 2024 

Table 6: Hyperparameter ablation study for miniImageNet (5-way, 5-shot) tasks. Accuracies in (% _±_ std). 

|**Masking Ratio**<br>**Value**<br>**Accuracy**|**Output Dim. (**_d_**)**|**Neg. Loss Weight(**_λ_**)**|**# of NNs(**_k_**)**|**# of Clusters(**_P_**)**<br>**Value**<br>**Accuracy**|**Memory Size(**_M_**)**|**Memory Module Confguration**<br>**Value**<br>**Accuracy**|
|---|---|---|---|---|---|---|
||**Value**<br>**Accuracy**|**Value**<br>**Accuracy**|**Value**<br>**Accuracy**||**Value**<br>**Accuracy**||
|10%<br>86.59 ± 0.25|256<br>85.16 ± 0.26|0.0<br>85.45 ± 0.27|1<br>86.58 ± 0.27|100<br>85.27 ± 0.24|2048<br>85.38 ± 0.25<br>4096<br>86.28 ± 0.29|DyCE-FIFO<br>84.05 ± 0.39<br>DyCE-kMeans<br>85.37 ± 0.33|
|30%<br>**87.82** ± 0.29|512<br>**87.82** ± 0.29|0.1<br>**87.82** ± 0.29|3<br>**87.82** ± 0.29|200<br>**87.82** ± 0.29|||
|50%<br>83.36 ± 0.28<br>70%<br>77.70 ± 0.20|1024<br>85.93 ± 0.31|0.3<br>86.33 ± 0.29|5<br>86.79 ± 0.26|300<br>85.81 ± 0.25|8192<br>**87.82** ± 0.29|DyCE<br>**87.82** ± 0.29|
||2054<br>85.42 ± 0.34|0.5<br>85.63 ± 0.26|10<br>86.17 ± 0.28|500<br>85.45 ± 0.20|12288<br>85.84 ± 0.22||



Fig. 7 assesses this hypothesis by comparing pure pretraining (i.e., BECLR without OpTA) and downstream performance on miniImageNet for the (5-way, 1-shot) setting as training progresses. As can be seen, when the initial pretraining performance is poor, OpTA even leads to performance degradation. On the contrary, it offers an increasingly larger boost as pretraining accuracy improves. The key message here is that these two steps (pretraining and inference) are highly intertwined, further enhancing the overall performance. This notion sits at the core of the BECLR design strategy. 

## 5.2 ABLATION STUDIES 

**Main Components of BECLR.** Let us investigate the impact of sequentially adding in the four main components of BECLR’s end-to-end architecture: (i) masking, (ii) EMA teacher encoder, (iii) DyCE and OpTA. As can be seen from Table 5, when applied individually, masking degrades the performance, but when combined with EMA, it gives a slight boost (1 _._ 5%) for both _{_ 1 _,_ 5 _}_ -shot settings. 

Table 5: Ablating main components of BECLR. 

||**Masking**|**EMA Teacher**|**DyCE**|**OpTA**|**5-way 1-shot**|**5-way 5-shot**|
|---|---|---|---|---|---|---|
||-|-|-|-|63.57 ± 0.43|81.42 ± 0.28|
||✓|-|-|-|54.53 ± 0.42|68.35 ± 0.27|
||-|✓|-|-|65.02 ± 0.41|82.33 ± 0.25|
||✓<br>✓<br>✓|✓<br>✓<br>✓|-<br>✓<br>✓|-<br>-<br>✓|65.33 ± 0.44<br>67.75 ± 0.43<br>**80.57** ± 0.57|82.69 ± 0.26<br>85.53 ± 0.27<br>**87.82** ± 0.29|



DyCE and OpTA are the most crucial components contributing to the overall performance of BECLR. DyCE offers an extra 2 _._ 4% and 2 _._ 8% accuracy boost in the 1-shot and 5-shot settings, respectively, and OpTA provides another 12 _._ 8% and 2 _._ 3% performance gain, in the aforementioned settings. As discussed earlier, also illustrated in Fig. 7, the gain of OpTA is owing and proportional to the performance of DyCE. This boost is paramount in the 1-shot scenario where the sample bias is severe. 

**Other Hyperparameters.** In Table 6, we summarize the result of ablations on: (i) the masking ratio of student images, (ii) the embedding latent dimension _d_ , (iii) the loss weighting hyperparameter _λ_ (in Eq. 3), and regarding DyCE: (iv) the number of nearest neighbors selected _k_ , (v) the number of memory partitions/clusters _P_ , (vi) the size of the memory _M_ , and (vii) different memory module configurations. As can be seen, a random masking ratio of 30% yields the best performance. We find that _d_ = 512 gives the best results, which is consistent with the literature (Grill et al., 2020; Chen & He, 2021). The negative loss term turns out to be unnecessary to prevent representation collapse (as can be seen when _λ_ = 0) and _λ_ = 0 _._ 1 yields the best performance. Regarding the number of neighbors selected ( _k_ ), there seems to be a sweet spot in this setting around _k_ = 3, where increasing further would lead to inclusion of potentially false positives, and thus performance degradation. _P_ and _M_ are tunable hyperparameters (per dataset) that return the best performance at _P_ = 200 and _M_ = 8192 in this setting. Increasing _P_ , _M_ beyond a certain threshold appears to have a negative impact on cluster formation. We argue that extremely large memory would result in accumulating old embeddings, which might no longer be a good representative of their corresponding classes. Finally, we compare DyCE against two _degenerate_ versions of itself: DyCE-FIFO, where the memory has no partitions and is updated with a first-in-first-out strategy; here for incoming embeddings we pick the closest _k_ neighbors. DyCE-kMeans, where we preserve the memory structure and only replace optimal transport with kMeans (with _P_ clusters), in line 8 of Algorithm 2. The performance drop in both cases confirms the importance of the proposed mechanics within DyCE. 

## 6 CONCLUDING REMARKS AND BROADER IMPACT 

In this paper, we have articulated two key shortcomings of the prior art in U-FSL, to address each of which we have proposed a novel solution embedded within the proposed end-to-end approach, BECLR. The first angle of novelty in BECLR is its dynamic clustered memory module (coined as DyCE) to enhance positive sampling in contrastive learning. The second angle of novelty is an efficient distribution alignment strategy (called OpTA) to address the inherent sample bias in (U-)FSL. Even though tailored towards U-FSL, we believe DyCE has potential _broader impact_ on generic self-supervised learning state-of-the-art, as we already demonstrate (in Section 5) that even with OpTA unplugged, DyCE alone empowers BECLR to still outperform the likes of SwaV, SimSiam and NNCLR. OpTA, on the other hand, is an efficient add-on module, which we argue has to become an _integral part_ of all (U-)FSL approaches, especially in the more challenging low-shot scenarios. 

9 

Published as a conference paper at ICLR 2024 

## REPRODUCIBILITY STATEMENT. 

To help readers reproduce our experiments, we provide extensive descriptions of implementation details and algorithms. Architectural and training details are provided in Appendices A.1 and A.2, respectively, along with information on the applied data augmentations (Appendix A.3) and tested benchmark datasets (Appendix B.1). The algorithms for BECLR pretraining and DyCE are also provided in both algorithmic (in Algorithms 1, 2) and Pytorch-like pseudocode formats (in Algorithms 3, 4). We have taken every measure to ensure fairness in our comparisons by following the most commonly adopted pretraining and evaluation settings in the U-FSL literature in terms of: pretraining/inference benchmark datasets used for both in-domain and cross-domain experiments, pretraining data augmentations, ( _N_ -way, _K_ -shot) inference settings and number of query set images per tested episode. We also draw baseline results from their corresponding original papers and compare their performance with BECLR for identical backbone depths. For our reproductions (denoted as _[†]_ ) of SwAV and NNCLR we follow the codebase of the original work and adopt it in the U-FSL setup, by following the same augmentations, backbones, and evaluation settings as BECLR. Our codebase is also provided as part of our supplementary material, in an anonymized fashion, and will be made publicly available upon acceptance. All training and evaluation experiments are conducted on 2 A40 NVIDIA GPUs. 

## ETHICS STATEMENT. 

We have read the ICLR Code of Ethics (https://iclr.cc/public/CodeOfEthics) and ensured that this work adheres to it. All benchmark datasets and pretrained model checkpoints are publicly available and not directly subject to ethical concerns. 

## ACKNOWLEDGEMENTS. 

The authors would like to thank Marcel Reinders, Ojas Shirekar and Holger Caesar for insightful conversations and brainstorming. This work was in part supported by Shell.ai Innovation program. 

## REFERENCES 

- Antreas Antoniou and Amos Storkey. Assume, augment and learn: Unsupervised few-shot metalearning via random labels and data augmentation. _arXiv preprint arXiv:1902.09884_ , 2019. 

- Randall Balestriero, Mark Ibrahim, Vlad Sobal, Ari Morcos, Shashank Shekhar, Tom Goldstein, Florian Bordes, Adrien Bardes, Gregoire Mialon, Yuandong Tian, et al. A cookbook of selfsupervised learning. _arXiv preprint arXiv:2304.12210_ , 2023. 

- Adrien Bardes, Jean Ponce, and Yann LeCun. Vicreg: Variance-invariance-covariance regularization for self-supervised learning. In _International Conference on Learning Representations_ , 2021. 

- Peyman Bateni, Raghav Goyal, Vaden Masrani, Frank Wood, and Leonid Sigal. Improved fewshot visual classification. In _Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition_ , pp. 14493–14502, 2020. 

- Peyman Bateni, Jarred Barber, Jan-Willem Van de Meent, and Frank Wood. Enhancing few-shot image classification with unlabelled examples. In _Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision_ , pp. 2796–2805, 2022. 

- Yassir Bendou, Yuqing Hu, Raphael Lafargue, Giulia Lioi, Bastien Pasdeloup, St´ephane Pateux, and Vincent Gripon. Easy—ensemble augmented-shot-y-shaped learning: State-of-the-art fewshot classification with simple components. _Journal of Imaging_ , 8(7):179, 2022. 

- Mathilde Caron, Ishan Misra, Julien Mairal, Priya Goyal, Piotr Bojanowski, and Armand Joulin. Unsupervised learning of visual features by contrasting cluster assignments. _Advances in neural information processing systems_ , 33:9912–9924, 2020. 

- Lee Chen, Kuilin Chen, and Kuilin Chi-Guhn. Unsupervised few-shot learning via deep laplacian eigenmaps. _arXiv preprint arXiv:2210.03595_ , 2022. 

10 

Published as a conference paper at ICLR 2024 

Ting Chen, Simon Kornblith, Mohammad Norouzi, and Geoffrey Hinton. A simple framework for contrastive learning of visual representations. In _International conference on machine learning_ , pp. 1597–1607. PMLR, 2020a. 

Wei-Yu Chen, Yen-Cheng Liu, Zsolt Kira, Yu-Chiang Frank Wang, and Jia-Bin Huang. A closer look at few-shot classification. In _International Conference on Learning Representations_ , 2018. 

- Wentao Chen, Chenyang Si, Wei Wang, Liang Wang, Zilei Wang, and Tieniu Tan. Few-shot learning with part discovery and augmentation from unlabeled images. _arXiv preprint arXiv:2105.11874_ , 2021a. 

- Xinlei Chen and Kaiming He. Exploring simple siamese representation learning. In _Proceedings of the IEEE/CVF conference on computer vision and pattern recognition_ , pp. 15750–15758, 2021. 

- Xinlei Chen, Haoqi Fan, Ross Girshick, and Kaiming He. Improved baselines with momentum contrastive learning. _arXiv e-prints_ , pp. arXiv–2003, 2020b. 

- Zitian Chen, Subhransu Maji, and Erik Learned-Miller. Shot in the dark: Few-shot learning with no base-class labels. In _Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition_ , pp. 2668–2677, 2021b. 

- Noel Codella, Veronica Rotemberg, Philipp Tschandl, M Emre Celebi, Stephen Dusza, David Gutman, Brian Helba, Aadi Kalloo, Konstantinos Liopyris, Michael Marchetti, et al. Skin lesion analysis toward melanoma detection 2018: A challenge hosted by the international skin imaging collaboration (isic). _arXiv preprint arXiv:1902.03368_ , 2019. 

- Ekin D Cubuk, Barret Zoph, Jonathon Shlens, and Quoc V Le. Randaugment: Practical automated data augmentation with a reduced search space. In _Proceedings of the IEEE/CVF conference on computer vision and pattern recognition workshops_ , pp. 702–703, 2020. 

Wentao Cui and Yuhong Guo. Parameterless transductive feature re-representation for few-shot learning. In _International Conference on Machine Learning_ , pp. 2212–2221. PMLR, 2021. 

- Marco Cuturi. Sinkhorn distances: lightspeed computation of optimal transport. In _Proceedings of the 26th International Conference on Neural Information Processing Systems-Volume 2_ , pp. 2292–2300, 2013. 

- Debasmit Das, Sungrack Yun, and Fatih Porikli. Confess: A framework for single source crossdomain few-shot learning. In _International Conference on Learning Representations_ , 2021. 

- David L Davies and Donald W Bouldin. A cluster separation measure. _IEEE transactions on pattern analysis and machine intelligence_ , pp. 224–227, 1979. 

- Guneet Singh Dhillon, Pratik Chaudhari, Avinash Ravichandran, and Stefano Soatto. A baseline for few-shot image classification. In _International Conference on Learning Representations_ , 2019. 

- Debidatta Dwibedi, Yusuf Aytar, Jonathan Tompson, Pierre Sermanet, and Andrew Zisserman. With a little help from my friends: Nearest-neighbor contrastive learning of visual representations. In _Proceedings of the IEEE/CVF International Conference on Computer Vision_ , pp. 9588–9597, 2021. 

- Linus Ericsson, Henry Gouk, and Timothy M Hospedales. How well do self-supervised models transfer? In _Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition_ , pp. 5414–5423, 2021. 

- Chelsea Finn, Pieter Abbeel, and Sergey Levine. Model-agnostic meta-learning for fast adaptation of deep networks. In _International conference on machine learning_ , pp. 1126–1135. PMLR, 2017. 

- Saba Ghaffari, Ehsan Saleh, David Forsyth, and Yu-Xiong Wang. On the importance of firth bias reduction in few-shot classification. In _International Conference on Learning Representations_ , 2021. 

- Spyros Gidaris, Andrei Bursuc, Nikos Komodakis, Patrick P´erez, and Matthieu Cord. Boosting few-shot visual learning with self-supervision. In _Proceedings of the IEEE/CVF international conference on computer vision_ , pp. 8059–8068, 2019. 

11 

Published as a conference paper at ICLR 2024 

- Jean-Bastien Grill, Florian Strub, Florent Altch´e, Corentin Tallec, Pierre Richemond, Elena Buchatskaya, Carl Doersch, Bernardo Avila Pires, Zhaohan Guo, Mohammad Gheshlaghi Azar, et al. Bootstrap your own latent-a new approach to self-supervised learning. _Advances in neural information processing systems_ , 33:21271–21284, 2020. 

- Yunhui Guo, Noel C Codella, Leonid Karlinsky, James V Codella, John R Smith, Kate Saenko, Tajana Rosing, and Rogerio Feris. A broader study of cross-domain few-shot learning. In _Computer Vision–ECCV 2020: 16th European Conference, Glasgow, UK, August 23–28, 2020, Proceedings, Part XXVII 16_ , pp. 124–141. Springer, 2020. 

- Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun. Deep residual learning for image recognition. In _Proceedings of the IEEE conference on computer vision and pattern recognition_ , pp. 770–778, 2016. 

- Kaiming He, Haoqi Fan, Yuxin Wu, Saining Xie, and Ross Girshick. Momentum contrast for unsupervised visual representation learning. In _Proceedings of the IEEE/CVF conference on computer vision and pattern recognition_ , pp. 9729–9738, 2020. 

- Yangji He, Weihan Liang, Dongyang Zhao, Hong-Yu Zhou, Weifeng Ge, Yizhou Yu, and Wenqiang Zhang. Attribute surrogates learning and spectral tokens pooling in transformers for few-shot learning. In _Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition_ , pp. 9119–9129, 2022. 

- Patrick Helber, Benjamin Bischke, Andreas Dengel, and Damian Borth. Eurosat: A novel dataset and deep learning benchmark for land use and land cover classification. _IEEE Journal of Selected Topics in Applied Earth Observations and Remote Sensing_ , 12(7):2217–2226, 2019. 

- Olivier J H´enaff, Skanda Koppula, Jean-Baptiste Alayrac, Aaron Van den Oord, Oriol Vinyals, and Joao Carreira. Efficient visual pretraining with contrastive detection. In _Proceedings of the IEEE/CVF International Conference on Computer Vision_ , pp. 10086–10096, 2021. 

- Markus Hiller, Rongkai Ma, Mehrtash Harandi, and Tom Drummond. Rethinking generalization in few-shot classification. _Advances in Neural Information Processing Systems_ , 35:3582–3595, 2022. 

- Kyle Hsu, Sergey Levine, and Chelsea Finn. Unsupervised learning via meta-learning. In _International Conference on Learning Representations_ , 2018. 

- Shell Xu Hu, Pablo Garcia Moreno, Yang Xiao, Xi Shen, Guillaume Obozinski, Neil Lawrence, and Andreas Damianou. Empirical bayes transductive meta-learning with synthetic gradients. In _International Conference on Learning Representations_ , 2019. 

- Shell Xu Hu, Da Li, Jan St¨uhmer, Minyoung Kim, and Timothy M Hospedales. Pushing the limits of simple pipelines for few-shot learning: External data and fine-tuning make a difference. In _Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition_ , pp. 9068–9077, 2022. 

- Wentao Hu, Xiurong Jiang, Jiarun Liu, Yuqi Yang, and Hui Tian. Meta-dm: Applications of diffusion models on few-shot learning. _arXiv preprint arXiv:2305.08092_ , 2023a. 

- Yuqing Hu, St´ephane Pateux, and Vincent Gripon. Adaptive dimension reduction and variational inference for transductive few-shot classification. In _International Conference on Artificial Intelligence and Statistics_ , pp. 5899–5917. PMLR, 2023b. 

- Ekaterina Iakovleva, Jakob Verbeek, and Karteek Alahari. Meta-learning with shared amortized variational inference. In _International Conference on Machine Learning_ , pp. 4572–4582. PMLR, 2020. 

- Huiwon Jang, Hankook Lee, and Jinwoo Shin. Unsupervised meta-learning via few-shot pseudosupervised contrastive learning. In _The Eleventh International Conference on Learning Representations_ , 2022. 

12 

Published as a conference paper at ICLR 2024 

- Yannis Kalantidis, Mert Bulent Sariyildiz, Noe Pion, Philippe Weinzaepfel, and Diane Larlus. Hard negative mixing for contrastive learning. _Advances in Neural Information Processing Systems_ , 33:21798–21809, 2020. 

- Siavash Khodadadeh, Ladislau Boloni, and Mubarak Shah. Unsupervised meta-learning for fewshot image classification. _Advances in neural information processing systems_ , 32, 2019. 

- Siavash Khodadadeh, Sharare Zehtabian, Saeed Vahidian, Weijia Wang, Bill Lin, and Ladislau Boloni. Unsupervised meta-learning through latent-space interpolation in generative models. In _International Conference on Learning Representations_ , 2020. 

- Gregory Koch, Richard Zemel, Ruslan Salakhutdinov, et al. Siamese neural networks for one-shot image recognition. In _ICML deep learning workshop_ . Lille, 2015. 

- Alex Krizhevsky, Vinod Nair, and Geoffrey Hinton. Cifar-10 and cifar-100 datasets. _URl: https://www. cs. toronto. edu/kriz/cifar. html_ , 6(1):1, 2009. 

- Dong Bok Lee, Dongchan Min, Seanie Lee, and Sung Ju Hwang. Meta-gmvae: Mixture of gaussian vae for unsupervised meta-learning. In _International Conference on Learning Representations_ , 2020. 

- Kwonjoon Lee, Subhransu Maji, Avinash Ravichandran, and Stefano Soatto. Meta-learning with differentiable convex optimization. In _Proceedings of the IEEE/CVF conference on computer vision and pattern recognition_ , pp. 10657–10665, 2019. 

- Jianyi Li and Guizhong Liu. Trainable class prototypes for few-shot learning. _arXiv preprint arXiv:2106.10846_ , 2021. 

- Shuo Li, Fang Liu, Zehua Hao, Kaibo Zhao, and Licheng Jiao. Unsupervised few-shot image classification by learning features into clustering space. In _European Conference on Computer Vision_ , pp. 420–436. Springer, 2022. 

- Aristidis Likas, Nikos Vlassis, and Jakob J Verbeek. The global k-means clustering algorithm. _Pattern recognition_ , 36(2):451–461, 2003. 

- Yuning Lu, Liangjian Wen, Jianzhuang Liu, Yajing Liu, and Xinmei Tian. Self-supervision can be a good few-shot learner. In _European Conference on Computer Vision_ , pp. 740–758. Springer, 2022. 

- David McAllester and Karl Stratos. Formal limitations on the measurement of mutual information. In _International Conference on Artificial Intelligence and Statistics_ , pp. 875–884. PMLR, 2020. 

- Leland McInnes, John Healy, Nathaniel Saul, and Lukas Großberger. Umap: Uniform manifold approximation and projection. _Journal of Open Source Software_ , 3(29):861, 2018. 

- Carlos Medina, Arnout Devos, and Matthias Grossglauser. Self-supervised prototypical transfer learning for few-shot classification. _arXiv preprint arXiv:2006.11325_ , 2020. 

- Sharada P Mohanty, David P Hughes, and Marcel Salath´e. Using deep learning for image-based plant disease detection. _Frontiers in plant science_ , 7:1419, 2016. 

- Aaron van den Oord, Yazhe Li, and Oriol Vinyals. Representation learning with contrastive predictive coding. _arXiv preprint arXiv:1807.03748_ , 2018. 

- Maxime Oquab, Timoth´ee Darcet, Th´eo Moutakanni, Huy Vo, Marc Szafraniec, Vasil Khalidov, Pierre Fernandez, Daniel Haziza, Francisco Massa, Alaaeldin El-Nouby, et al. Dinov2: Learning robust visual features without supervision. _arXiv preprint arXiv:2304.07193_ , 2023. 

- Adam Paszke, Sam Gross, Francisco Massa, Adam Lerer, James Bradbury, Gregory Chanan, Trevor Killeen, Zeming Lin, Natalia Gimelshein, Luca Antiga, et al. Pytorch: An imperative style, highperformance deep learning library. _Advances in neural information processing systems_ , 32, 2019. 

- Ben Poole, Sherjil Ozair, Aaron Van Den Oord, Alex Alemi, and George Tucker. On variational bounds of mutual information. In _International Conference on Machine Learning_ , pp. 5171– 5180. PMLR, 2019. 

13 

Published as a conference paper at ICLR 2024 

- Aravind Rajeswaran, Chelsea Finn, Sham M Kakade, and Sergey Levine. Meta-learning with implicit gradients. _Advances in neural information processing systems_ , 32, 2019. 

- Sachin Ravi and Hugo Larochelle. Optimization as a model for few-shot learning. In _International conference on learning representations_ , 2016. 

- Mengye Ren, Eleni Triantafillou, Sachin Ravi, Jake Snell, Kevin Swersky, Joshua B Tenenbaum, Hugo Larochelle, and Richard S Zemel. Meta-learning for semi-supervised few-shot classification. In _6th International Conference on Learning Representations, ICLR 2018_ , 2018. 

- Olga Russakovsky, Jia Deng, Hao Su, Jonathan Krause, Sanjeev Satheesh, Sean Ma, Zhiheng Huang, Andrej Karpathy, Aditya Khosla, Michael Bernstein, et al. Imagenet large scale visual recognition challenge. _International journal of computer vision_ , 115:211–252, 2015. 

- Andrei A Rusu, Dushyant Rao, Jakub Sygnowski, Oriol Vinyals, Razvan Pascanu, Simon Osindero, and Raia Hadsell. Meta-learning with latent embedding optimization. In _International Conference on Learning Representations_ , 2018. 

- Ojas Shirekar and Hadi Jamali-Rad. Self-supervised class-cognizant few-shot classification. In _2022 IEEE International Conference on Image Processing (ICIP)_ , pp. 976–980. IEEE, 2022. 

- Ojas Kishorkumar Shirekar, Anuj Singh, and Hadi Jamali-Rad. Self-attention message passing for contrastive few-shot learning. In _Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision_ , pp. 5426–5436, 2023. 

- Anuj Rajeeva Singh and Hadi Jamali-Rad. Transductive decoupled variational inference for few-shot classification. _Transactions on Machine Learning Research_ , 2022. 

- Jake Snell, Kevin Swersky, and Richard Zemel. Prototypical networks for few-shot learning. _Advances in neural information processing systems_ , 30, 2017. 

- Jiaming Song and Stefano Ermon. Understanding the limitations of variational mutual information estimators. In _International Conference on Learning Representations_ , 2019. 

- Yonglong Tian, Yue Wang, Dilip Krishnan, Joshua B Tenenbaum, and Phillip Isola. Rethinking few-shot image classification: a good embedding is all you need? In _Computer Vision–ECCV 2020: 16th European Conference, Glasgow, UK, August 23–28, 2020, Proceedings, Part XIV 16_ , pp. 266–282. Springer, 2020. 

- Oriol Vinyals, Charles Blundell, Timothy Lillicrap, Daan Wierstra, et al. Matching networks for one shot learning. _Advances in neural information processing systems_ , 29, 2016. 

- Haoqing Wang and Zhi-Hong Deng. Cross-domain few-shot classification via adversarial task augmentation. _arXiv preprint arXiv:2104.14385_ , 2021. 

- Haoqing Wang, Zhi-Hong Deng, and Haoqing Wang. Contrastive prototypical network with wasserstein confidence penalty. In _European Conference on Computer Vision_ , pp. 665–682. Springer, 2022a. 

- Heng Wang, Tan Yue, Xiang Ye, Zihang He, Bohan Li, and Yong Li. Revisit finetuning strategy for few-shot learning to transfer the emdeddings. In _The Eleventh International Conference on Learning Representations_ , 2022b. 

- Xiaosong Wang, Yifan Peng, Le Lu, Zhiyong Lu, Mohammadhadi Bagheri, and Ronald M Summers. Chestx-ray8: Hospital-scale chest x-ray database and benchmarks on weakly-supervised classification and localization of common thorax diseases. In _Proceedings of the IEEE conference on computer vision and pattern recognition_ , pp. 2097–2106, 2017. 

- Yan Wang, Wei-Lun Chao, Kilian Q Weinberger, and Laurens Van Der Maaten. Simpleshot: Revisiting nearest-neighbor classification for few-shot learning. _arXiv preprint arXiv:1911.04623_ , 2019. 

- Peter Welinder, Steve Branson, Takeshi Mita, Catherine Wah, Florian Schroff, Serge Belongie, and Pietro Perona. Caltech-ucsd birds 200. 2010. 

14 

Published as a conference paper at ICLR 2024 

- Zhirong Wu, Yuanjun Xiong, Stella X Yu, and Dahua Lin. Unsupervised feature learning via nonparametric instance discrimination. In _Proceedings of the IEEE conference on computer vision and pattern recognition_ , pp. 3733–3742, 2018. 

- Hui Xu, Jiaxing Wang, Hao Li, Deqiang Ouyang, and Jie Shao. Unsupervised meta-learning for few-shot learning. _Pattern Recognition_ , 116:107951, 2021. 

- Jing Xu, Xu Luo, Xinglin Pan, Yanan Li, Wenjie Pei, and Zenglin Xu. Alleviating the sample selection bias in few-shot learning by removing projection to the centroid. _Advances in Neural Information Processing Systems_ , 35:21073–21086, 2022. 

- Ling Yang, Liangliang Li, Zilun Zhang, Xinyu Zhou, Erjin Zhou, and Yu Liu. Dpgn: Distribution propagation graph network for few-shot learning. In _Proceedings of the IEEE/CVF conference on computer vision and pattern recognition_ , pp. 13390–13399, 2020. 

- Shuo Yang, Lu Liu, and Min Xu. Free lunch for few-shot learning: Distribution calibration. In _International Conference on Learning Representations_ , 2021. 

- Han-Jia Ye, Lu Han, and De-Chuan Zhan. Revisiting unsupervised meta-learning via the characteristics of few-shot tasks. _IEEE Transactions on Pattern Analysis and Machine Intelligence_ , 45(3): 3721–3737, 2022. 

- Yang You, Igor Gitman, and Boris Ginsburg. Large batch training of convolutional networks. _arXiv preprint arXiv:1708.03888_ , 2017. 

- Jure Zbontar, Li Jing, Ishan Misra, Yann LeCun, and St´ephane Deny. Barlow twins: Self-supervised learning via redundancy reduction. In _International Conference on Machine Learning_ , pp. 12310– 12320. PMLR, 2021. 

- Hao Zhang, Feng Li, Shilong Liu, Lei Zhang, Hang Su, Jun Zhu, Lionel Ni, and Heung-Yeung Shum. Dino: Detr with improved denoising anchor boxes for end-to-end object detection. In _The Eleventh International Conference on Learning Representations_ , 2022. 

- Xueting Zhang, Debin Meng, Henry Gouk, and Timothy M Hospedales. Shallow bayesian meta learning for real-world few-shot recognition. In _Proceedings of the IEEE/CVF International Conference on Computer Vision_ , pp. 651–660, 2021. 

- Chengxu Zhuang, Alex Lin Zhai, and Daniel Yamins. Local aggregation for unsupervised learning of visual embeddings. In _Proceedings of the IEEE/CVF International Conference on Computer Vision_ , pp. 6002–6012, 2019. 

15 

Published as a conference paper at ICLR 2024 

## A IMPLEMENTATION DETAILS 

This section describes the implementation and training details of BECLR. 

## A.1 ARCHITECTURE DETAILS 

BECLR is implemented on PyTorch (Paszke et al., 2019). We use the ResNet family (He et al., 2016) for our backbone networks ( _fθ_ , _fψ_ ). The projection ( _gθ_ , _gψ_ ) and prediction ( _hθ_ ) heads are 3- and 2- layer MLPs, respectively, as in Chen & He (2021). Batch normalization (BN) and a ReLU activation function are applied to each MLP layer, except for the output layers. No ReLU activation is applied on the output layer of the projection heads ( _gθ_ , _gψ_ ), while neither BN nor a ReLU activation is applied on the output layer of the prediction head ( _hθ_ ). We use a resolution of 224 _×_ 224 for input images and a latent embedding dimension of _d_ = 512 in all models and experiments, unless otherwise stated. The DyCE memory module consists of a memory unit _M_ , initialized as a random table (of size _M × d_ ). We also maintain up to _P_ partitions in _M_ = [ _P_ 1 _, . . . , PP_ ], each of which is represented by a prototype stored in **Γ** = [ _**γ**_ 1 _, . . . ,_ _**γ** P_ ]. Prototypes _**γ** i_ are the average of the latent embeddings stored in partition _Pi_ . When training on miniImageNet, CIFAR-FS and FC100, we use a memory of size _M_ = 8192 that contains _P_ = 200 partitions and cluster prototypes ( _M_ = 40960, _P_ = 1000 for tieredImageNet). Note that both _M_ and _P_ are important hyperparameters, whose values were selected by evaluating on the validation set of each dataset for model selection. These hyperparameters would also need to be carefully tuned on an unknown unlabeled training dataset. 

## A.2 TRAINING DETAILS 

BECLR is pretrained on the training splits of miniImageNet, CIFAR-FS, FC100 and tieredImageNet. We use a batch size of _B_ = 256 images for all datasets, except for tieredImageNet ( _B_ = 512). Following Chen & He (2021), images are resized to 224 _×_ 224 for all configurations. We use the SGD optimizer with a weight decay of 10 _[−]_[4] , a momentum of 0 _._ 995, and a cosine decay schedule of the learning rate. Note that we do not require large-batch optimizers, such as LARS (You et al., 2017), or early stopping. Similarly to Lu et al. (2022), the initial learning rate is set to 0 _._ 3 for the smaller miniImageNet, CIFAR-FS, FC100 datasets and 0 _._ 1 for tieredImageNet, and we train for 400 and 200 epochs, respectively. The temperature scalar in the loss function is set to _τ_ = 2. Upon finishing unsupervised pretraining, we only keep the last epoch checkpoint of the student encoder ( _fθ_ ) for the subsequent inference stage. For the inference and downstream few-shot classification stage, we create ( _N_ -way, _K_ -shot) tasks from the validation and testing splits of each dataset for model selection and evaluation, respectively. In the inference stage, we sequentially perform up to _δ ≤_ 5 consecutive passes of OpTA, with the transported prototypes of each pass acting as the input of the next pass. The optimal value for _δ_ for each dataset and ( _N_ -way, _K_ -shot) setting is selected by evaluating on the validation dataset. 

Note that at the beginning of pretraining both the encoder representations and the memory embedding space within DyCE are highly volatile. Thus, we allow for an adaptation period epoch _<_ epochthr (empirically 20-50 epochs), during which the batch enhancement path of DyCE is not activated and the encoder is trained via standard contrastive learning (without enhancing the batch with additional positives). On the contrary, the memory updating path of DyCE is activated for every training batch from the beginning of training, allowing the memory to reach a highly separable converged state (see Fig. 6), before plugging in the batch enhancement path in the BECLR pipeline. When the memory space ( _M_ ) is full for the first time, a kmeans (Likas et al., 2003) clustering step is performed for initializing the cluster prototypes ( _**γ** i_ ) and memory partitions ( _Pi_ ). This kmeans clustering step is performed only once during training to initialize the memory prototypes, which are then dynamically updated for each training batch by the memory updating path of DyCE. 

## A.3 IMAGE AUGMENTATIONS 

The data augmentations that were applied in the pretraining stage of BECLR are showcased in Table 7. These augmentations were applied on the input images for all training datasets. The default data augmentation profile follows a common data augmentation strategy in SSL, including RandomResizedCrop (with scale in [0.2, 1.0]), random ColorJitter (Wu et al., 2018) of _{_ brightness, contrast, saturation, hue _}_ with a probability of 0 _._ 1, RandomGrayScale with a probability of 0 _._ 2, random GaussianBlur with a probability of 0 _._ 5 and a Gaussian kernel in [0 _._ 1 _,_ 2 _._ 0], and finally, RandomHorizontalFlip with a probability of 0 _._ 5. Following Lu et al. (2022), this profile can be expanded to 

16 

Published as a conference paper at ICLR 2024 

Table 7: Pytorch-like descriptions of the data augmentation profiles applied on the pretraining phase of BECLR. 

|**Data Augmentation Profle**|**Description**|
|---|---|
||RandomResizedCrop(size=224, scale=(0.2, 1))|
||RandomApply([ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1)], p=0.1)|
|**Default**|RandomGrayScale(p=0.2)|
||RandomApply([GaussianBlur([0.1, 2.0])], p=0.5)|
||RandomHorizontalFlip(p=0.5)|
||RandomResizedCrop(size=224, scale=(0.2, 1))|
||RandomApply([ColorJitter(brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1)], p=0.1)|
||RandomGrayScale(p=0.2)|
|**Strong**|RandomApply([GaussianBlur([0.1, 2.0])], p=0.5)|
||RandAugment(n=2, m=10, mstd=0.5)|
||RandomHorizontalFlip(p=0.5)|
||RandomVerticalFlip(p=0.5)|



the strong data augmentation profile, which also includes RadomVerticalFlip with a probability of 0 _._ 5 and RandAugment (Cubuk et al., 2020) with _n_ = 2 layers, a magnitude of _m_ = 10, and a noise of the standard deviation of magnitude of _mstd_ = 0 _._ 5. Unless otherwise stated, the strong data augmentation profile is applied on all training images before being passed to the backbone encoders. 

## B EXPERIMENTAL SETUP 

In this section, we provide more detailed information regarding all the few-shot benchmark datasets that were used as part of our experimental evaluation (in Section 5), along with BECLR’s training and evaluation procedures for both in-domain and cross-domain U-FSL settings, for ensuring fair comparisons with U-FSL baselines and our reproductions (for SwAV and NNCLR). 

Table 8: Overview of cross-domain few-shot benchmarks, on which BECLR is evaluated. The datasets are sorted with decreasing (distribution) domain similarity to ImageNet and miniImageNet. 

|**ImageNet similarity**|**Dataset**|**# of classes**|**# of samples**|
|---|---|---|---|
|High|CUB (Welinder et al.,2010)|200|11,788|
|Low|CropDiseases (Mohanty et al.,2016)|38|43,456|
|Low|EuroSAT (Helber et al.,2019)|10|27,000|
|Low|ISIC (Codella et al.,2019)|7|10,015|
|Low|ChestX (Wang et al.,2017)|7|25,848|



## B.1 DATASET DETAILS 

**miniImageNet.** It is a subset of ImageNet (Russakovsky et al., 2015), containing 100 classes with 600 images per class. We randomly select 64, 16, and 20 classes for training, validation, and testing, following the predominantly adopted settings of Ravi & Larochelle (2016). 

**tieredImageNet.** It is a larger subset of ImageNet, containing 608 classes and a total of 779 _,_ 165 images, grouped into 34 high-level categories, 20 (351 classes) of which are used for training, 6 (97 classes) for validation and 8 (160 classes) for testing. 

**CIFAR-FS.** It is a subset of CIFAR-100 (Krizhevsky et al., 2009), which is focused on FSL tasks by following the same sampling criteria that were used for creating miniImageNet. It contains 100 classes with 600 images per class, grouped into 64, 16, 20 classes for training, validation, and testing, respectively. The additional challenge here is the limited original image resolution of 32 _×_ 32. 

**FC100.** It is also a subset of CIFAR-100 (Krizhevsky et al., 2009) and contains the same 60000 (32 _×_ 32) images as CIFAR-FS. Here, the original 100 classes are grouped into 20 superclasses, in such a way as to minimize the information overlap between training, validation and testing classes (McAllester & Stratos, 2020). This makes this data set more demanding for (U-)FSL, since training and testing classes are highly dissimilar. The training split contains 12 superclasses (of 60 classes), while both the validation and testing splits are composed of 4 superclasses (of 20 classes). 

17 

Published as a conference paper at ICLR 2024 

**CDFSL.** It consists of four distinct datasets with decreasing domain similarity to ImageNet, and by extension miniImageNet, ranging from crop disease images in CropDiseases (Mohanty et al., 2016) and aerial satellite images in EuroSAT (Helber et al., 2019) to dermatological skin lesion images in ISIC2018 (Codella et al., 2019) and grayscale chest X-ray images in ChestX (Wang et al., 2017). 

**CUB.** It consists of 200 classes and a total of 11 _,_ 788 images, split into 100 classes for training and 50 for both validation and testing, following the split settings of Chen et al. (2018). Additional information on the cross-domain few-shot benchmarks (CUB and CDFSL) is provided in Table 8. 

## B.2 PRETRAINING AND EVALUATION PROCEDURES 

For all in-domain experiments BECLR is pretrained on the training split of the selected dataset (miniImageNet, tieredImageNet, CIFAR-FS, or FC100), followed by the subsequent inference stage on the validation and testing splits of the same dataset for model selection and final evaluation, respectively. In contrast, in the cross-domain setting BECLR is pretrained on the training split of miniImageNet and then evaluated on the validation and test splits of either CDFSL (ChestX, ISIC, EuroSAT, CropDiseases) or CUB. We report test accuracies with 95% confidence intervals over 2000 test episodes, each with 15 query shots per class, for all tested datasets, as is most commonly adopted in the literature (Chen et al., 2021a; 2022; Lu et al., 2022). The performance on miniImageNet, tieredImageNet, CIFAR-FS, FC100 and miniImageNet _→_ CUB is evaluated on (5-way, _{_ 1 _,_ 5 _}_ -shot) classification tasks, while on miniImageNet _→_ CDFSL we evaluate on (5-way, _{_ 5 _,_ 20 _}_ -shot) tasks, as is customary across the literature (Guo et al., 2020). 

We have taken every measure to ensure fairness in our comparison with U-FSL baselines and our reproductions. To do so, all compared baselines have the same pretraining and testing dataset (in both in-domain and cross-domain scenarios), follow similar data augmentation profiles as part of their pretraining and are evaluated in identical ( _N_ -way, _K_ -shot) FSL settings, on the same number of query set images, for all tested inference episodes. We also draw baseline results from their corresponding original papers and compare their performance with BECLR for identical backbone depths (roughly similar parameter count for all methods). For our reproductions (SwAV and NNCLR), we follow the codebase of the original work and adopt it in the U-FSL setup, by following the same augmentations, backbones, and evaluation settings as BECLR. 

## C EXTENDED RESULTS AND VISUALIZATIONS 

This section provides extended experimental findings, both quantitative and qualitative, complementing the experimental evaluation in Section 5 and providing further intuition on BECLR. 


![](raw/papers/2_LLM_NLP/LLM_paper_list/few-shot/iclr_2024_BECLR_Batch_Enhanced_Contrastive_Few-Shot_Learning_images/iclr_2024_BECLR_Batch_Enhanced_Contrastive_Few-Shot_Learning.pdf-0018-08.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/few-shot/iclr_2024_BECLR_Batch_Enhanced_Contrastive_Few-Shot_Learning_images/iclr_2024_BECLR_Batch_Enhanced_Contrastive_Few-Shot_Learning.pdf-0018-09.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/few-shot/iclr_2024_BECLR_Batch_Enhanced_Contrastive_Few-Shot_Learning_images/iclr_2024_BECLR_Batch_Enhanced_Contrastive_Few-Shot_Learning.pdf-0018-10.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/few-shot/iclr_2024_BECLR_Batch_Enhanced_Contrastive_Few-Shot_Learning_images/iclr_2024_BECLR_Batch_Enhanced_Contrastive_Few-Shot_Learning.pdf-0018-11.png)


**----- Start of picture text -----**<br>
(a) (b) (c)<br>**----- End of picture text -----**<br>


Figure 9: (a, b): All 3 support images ( _•_ ), for all 3 classes, get assigned to a single class-representative memory prototype (■). (c): A cluster can be split into multiple subclusters, without degrading the model performance. Colors denote the classes of the random (3-way, 3-shot) episode and the assigned memory prototypes. 

## C.1 ANALYSIS ON PSEUDOCLASS-COGNIZANCE 

As discussed in Section 5, the dynamic equipartitioned updates of DyCE facilitate the creation of a highly separable latent memory space, which is then used for sampling additional meaningful 

18 

Published as a conference paper at ICLR 2024 

positive pairs. We argue that these separable clusters and their prototypes capture a semblance of class-cognizance. As a qualitative demonstration, Fig. 9 depicts the 2-D UMAP (McInnes et al., 2018) projections of the support embeddings ( _•_ ) for 3 (Figs. 9a, 9b, 9c) randomly sampled miniImageNet (3-way, 3-shot) tasks, along with 50 randomly selected memory prototypes (□) from the last-epoch checkpoint of DyCE. Only memory prototypes assigned to support embeddings (■) are colored (with the corresponding class color). As can be seen in Figs. 9a, 9b, all 3-shot embeddings for each class are assigned to a single memory prototype, which shows that the corresponding prototype, and by extension memory cluster, is capable of capturing the latent representation of a class even without access to base class labels. Note that as the training progresses, a memory cluster can split into two or more (sub-)clusters, corresponding to the same latent training class. For example, Fig. 9c illustrates such a case where the fox class has been split between two different prototypes. This is to be expected since there is no “one-to-one” relationship between memory clusters and latent training classes (we do not assume any prior knowledge on the classes of the unlabeled pretraining dataset and the number of memory clusters is an important hyperparameter to be tuned). Notably, BECLR still treats images sampled from these two (sub-)clusters as positives, which does not degrade the model’s performance assuming that both of these (sub-)clusters are consistent (i.e., contain only fox image embeddings). 

Table 9: Extended version of Table 1. Accuracies (in % _±_ std) on miniImageNet and tieredImageNet compared against unsupervised (Unsup.) and supervised (Sup.) baselines. Encoders: RN: ResNet, Conv: convolutional blocks. _[†]_ : denotes our reproduction. _[∗]_ : extra synthetic training data. Style: **best** and second best. 

||||**miniImageNet**|**miniImageNet**|**tieredImageNet**|**tieredImageNet**|
|---|---|---|---|---|---|---|
|**Method**|**Backbone**|**Setting**|**5-way 1-shot**|**5-way 5-shot**|**5-way 1-shot**|**5-way 5-shot**|
|ProtoTransfer (Medina et al.,2020)<br>Meta-GMVAE (Lee et al.,2020)<br>C3LR (Shirekar & Jamali-Rad,2022)<br>SAMPTransfer (Shirekar et al.,2023)<br>LF2CS (Li et al.,2022)<br>CPNWCP (Wang et al.,2022a)<br>SimCLR (Chen et al.,2020a)<br>SwAV_†_ (Caron et al.,2020)<br>NNCLR_†_ (Dwibedi et al.,2021)<br>SimSiam (Chen & He,2021)<br>HMS (Ye et al.,2022)<br>Laplacian Eigenmaps (Chen et al.,2022)<br>PsCo_†_ (Jang et al.,2022)<br>UniSiam + dist (Lu et al.,2022)<br>Meta-DM + UniSiam + dist_∗_(Hu et al.,2023a)<br>MetaOptNet (Lee et al.,2019)<br>Transductive CNAPS (Bateni et al.,2022)<br>MAML (Finn et al.,2017)<br>ProtoNet(Snell et al.,2017)|Conv4<br>Conv4<br>Conv4<br>Conv4b<br>RN12<br>RN18<br>RN18<br>RN18<br>RN18<br>RN18<br>RN18<br>RN18<br>RN18<br>RN18<br>RN18|Unsup.<br>Unsup.<br>Unsup.<br>Unsup.<br>Unsup.<br>Unsup.<br>Unsup.<br>Unsup.<br>Unsup.<br>Unsup.<br>Unsup.<br>Unsup.<br>Unsup.<br>Unsup.<br>Unsup.|45.67 ± 0.79<br>42.82 ± 0.45<br>47.92 ± 1.20<br>61.02 ± 1.05<br>47.93 ± 0.19<br>53.14 ± 0.62<br>62.58 ± 0.37<br>59.84 ± 0.52<br>63.33 ± 0.53<br>62.80 ± 0.37<br>58.20 ± 0.23<br>59.47 ± 0.87<br>47.24 ± 0.76<br>64.10 ± 0.36<br>65.64 ± 0.36|62.99 ± 0.75<br>55.73 ± 0.39<br>64.81 ± 1.15<br>72.52 ± 0.68<br>66.44 ± 0.17<br>67.36 ± 0.5<br>79.66 ± 0.27<br>78.23 ± 0.26<br>80.75 ± 0.25<br>79.85 ± 0.27<br>75.77 ± 0.16<br>78.79 ± 0.58<br>65.48 ± 0.68<br>82.26 ± 0.25<br>83.97 ± 0.25|-<br>-<br>42.37 ± 0.77<br>49.10 ± 0.94<br>53.16 ± 0.66<br>45.00 ± 0.19<br>63.38 ± 0.42<br>65.26 ± 0.53<br>65.46 ± 0.55<br>64.05 ± 0.40<br>58.42 ± 0.25<br>-<br>54.33 ± 0.54<br>67.01 ± 0.39<br>67.11 ± 0.40|-<br>-<br>61.77 ± 0.25<br>65.19 ± 0.82<br>66.59 ± 0.57<br>62.96 ± 0.19<br>79.17 ± 0.34<br>81.73 ± 0.24<br>81.40 ± 0.27<br>81.40 ± 0.30<br>75.85 ± 0.18<br>-<br>69.73 ± 0.49<br>84.47 ± 0.28<br>84.39 ± 0.28<br>81.56 ± 0.53<br>81.80 ± 0.70<br>70.30 ± 1.75<br>70.30 ± 1.75|
||RN18<br>RN18<br>RN34<br>RN34|Sup.<br>Sup.<br>Sup.<br>Sup.|64.09 ± 0.62<br>55.60 ± 0.90<br>51.46 ± 0.90<br>53.90 ± 0.83|80.00 ± 0.45<br>73.10 ± 0.70<br>65.90 ± 0.79<br>74.65 ± 0.64|65.99 ± 0.72<br>65.90 ± 1.10<br>51.67 ± 1.81<br>51.67 ± 1.81||
|**BECLR (Ours)**|RN18|Unsup.|**75.74** ± 0.62|**84.93** ± 0.33|**76.44** ± 0.66|**84.85** ± 0.37|
|SwAV_†_ (Caron et al.,2020)<br>NNCLR_†_ (Dwibedi et al.,2021)<br>TrainProto (Li & Liu,2021)<br>UBC-FSL (Chen et al.,2021b)<br>PDA-Net (Chen et al.,2021a)<br>UniSiam + dist (Lu et al.,2022)<br>Meta-DM + UniSiam + dist_∗_(Hu et al.,2023a)|RN50<br>RN50<br>RN50<br>RN50<br>RN50<br>RN50<br>RN50|Unsup.<br>Unsup.<br>Unsup.<br>Unsup.<br>Unsup.<br>Unsup.<br>Unsup.|63.34 ± 0.42<br>65.42 ± 0.44<br>58.92 ± 0.91<br>56.20 ± 0.60<br>63.84 ± 0.91<br>65.33 ± 0.36<br>66.68 ± 0.36|82.76 ± 0.24<br>83.31 ± 0.21<br>73.94 ± 0.63<br>75.40 ± 0.40<br>83.11 ± 0.56<br>83.22 ± 0.24<br>85.29 ± 0.23|68.02 ± 0.52<br>69.82 ± 0.54<br>-<br>66.60 ± 0.70<br>69.01 ± 0.93<br>69.60 ± 0.38<br>69.61 ± 0.38|85.93 ± 0.33<br>86.41 ± 0.31<br>-<br>83.10 ± 0.50<br>84.20 ± 0.69<br>86.51 ± 0.26<br>86.53 ± 0.26|
|**BECLR (Ours)**|RN50|Unsup.|**80.57** ± 0.57|**87.82** ± 0.29|**81.69** ± 0.61|**87.86** ± 0.32|



Table 10: Extended version of Table 2. Accuracies in (% _±_ std) on CIFAR-FS and FC100 of different U-FSL baselines. Encoders: RN: ResNet. _[†]_ : denotes our reproduction. Style: **best** and second best. 

||||**CIFAR-FS**|**CIFAR-FS**|**FC100**|**FC100**|
|---|---|---|---|---|---|---|
|**Method**|**Backbone**|**Setting**|**5-way 1-shot**|**5-way 5-shot**|**5-way 1-shot**|**5-way 5-shot**|
|Meta-GMVAE (Lee et al.,2020)<br>SimCLR (Chen et al.,2020a)<br>MoCo v2 (Chen et al.,2020b)<br>MoCHi (Kalantidis et al.,2020)<br>BYOL (Grill et al.,2020)<br>LF2CS (Li et al.,2022)<br>CUMCA (Xu et al.,2021)<br>Barlow Twins (Zbontar et al.,2021)<br>HMS (Ye et al.,2022)<br>DeepEigenmaps(Chen et al.,2022)|RN18<br>RN18<br>RN18<br>RN18<br>RN18<br>RN18<br>RN18<br>RN18<br>RN18<br>RN18|Unsup.<br>Unsup.<br>Unsup.<br>Unsup.<br>Unsup.<br>Unsup.<br>Unsup.<br>Unsup.<br>Unsup.<br>Unsup.|-<br>54.56 ± 0.19<br>52.73 ± 0.20<br>50.42 ± 0.22<br>51.33 ± 0.21<br>55.04 ± 0.72<br>50.48 ± 0.12<br>-<br>54.65 ± 0.20<br>-|-<br>71.19 ± 0.18<br>67.81 ± 0.19<br>65.91 ± 0.20<br>66.73 ± 0.18<br>70.62 ± 0.57<br>67.83 ± 0.18<br>-<br>73.70 ± 0.18<br>-|36.30 ± 0.70<br>36.20 ± 0.70<br>37.70 ± 0.70<br>37.51 ± 0.17<br>37.20 ± 0.70<br>37.20 ± 0.70<br>33.00 ± 0.17<br>37.90 ± 0.70<br>37.88 ± 0.16<br>39.70 ± 0.70|49.70 ± 0.80<br>49.90 ± 0.70<br>53.20 ± 0.70<br>48.95 ± 0.17<br>52.80 ± 0.60<br>52.80 ± 0.60<br>47.41 ± 0.19<br>54.10 ± 0.60<br>53.68 ± 0.18<br>57.90 ± 0.70|
|**BECLR (Ours)**|RN18|Unsup.|**70.39** ± 0.62|**81.56** ± 0.39|**45.21** ± 0.50|**60.02** ± 0.43|



19 

Published as a conference paper at ICLR 2024 

Table 11: Extended version of Table 3. Accuracies (in % _±_ std) on miniImageNet → CDFSL. _[†]_ : denotes our reproduction. Style: **best** and second best. 

|reproduction. Style: **best**|andsecond best<br>.|andsecond best<br>.|||||||
|---|---|---|---|---|---|---|---|---|
||**ChestX**||**ISIC**||**EuroSAT**||**CropDiseases**||
|**Method**|**5 way 5-shot**|**5 way 20-shot**|**5 way 5-shot**|**5 way 20-shot**|**5 way 5-shot**|**5 way 20-shot**|**5 way 5-shot**|**5 way 20-shot**|
|ProtoTransfer (Medina et al.,2020)<br>BYOL (Grill et al.,2020)<br>MoCo v2 (Chen et al.,2020b)<br>SwAV_†_ (Caron et al.,2020)<br>SimCLR (Chen et al.,2020a)<br>NNCLR_†_ (Dwibedi et al.,2021)<br>C3LR (Shirekar & Jamali-Rad,2022)<br>SAMPTransfer (Shirekar et al.,2023)<br>PsCo (Jang et al.,2022)<br>UniSiam + dist (Lu et al.,2022)<br>ConFeSS (Das et al.,2021)<br>ATA(Wang& Deng,2021)|26.71 ± 0.46<br>26.39 ± 0.43<br>25.26 ± 0.44<br>25.70 ± 0.28<br>26.36 ± 0.44<br>25.74 ± 0.41<br>26.00 ± 0.41<br>26.27 ± 0.44<br>24.78 ± 0.23<br>**28.18** ± 0.45<br>27.09<br>24.43 ± 0.2|33.82 ± 0.48<br>30.71 ± 0.47<br>29.43 ± 0.45<br>30.41 ± 0.25<br>30.82 ± 0.43<br>29.54 ± 0.45<br>33.39 ± 0.47<br>34.15 ± 0.50<br>27.69 ± 0.23<br>**34.58** ± 0.46<br>33.57<br>-|45.19 ± 0.56<br>43.09 ± 0.56<br>42.60 ± 0.55<br>40.69 ± 0.34<br>43.99 ± 0.55<br>38.85 ± 0.56<br>45.93 ± 0.54<br>47.60 ± 0.59<br>44.00 ± 0.30<br>45.65 ± 0.58<br>**48.85**<br>45.83 ± 0.3|59.07 ± 0.55<br>53.76 ± 0.55<br>52.39 ± 0.49<br>49.03 ± 0.30<br>53.00 ± 0.54<br>47.82 ± 0.53<br>59.95 ± 0.53<br>**61.28** ± 0.56<br>54.59 ± 0.29<br>56.54 ± 0.5<br>60.10<br>-|75.62 ± 0.67<br>83.64 ± 0.54<br>84.15 ± 0.52<br>84.82 ± 0.24<br>82.78 ± 0.56<br>83.45 ± 0.57<br>80.32 ± 0.65<br>85.55 ± 0.60<br>81.08 ± 0.35<br>86.53 ± 0.47<br>84.65<br>83.75 ± 0.4|86.80 ± 0.42<br>89.62 ± 0.39<br>88.92 ± 0.41<br>90.77 ± 0.26<br>89.38 ± 0.40<br>90.80 ± 0.39<br>88.09 ± 0.45<br>88.52 ± 0.50<br>87.65 ± 0.28<br>93.24 ± 0.30<br>90.40<br>-|86.53 ± 0.56<br>92.71 ± 0.47<br>87.62 ± 0.60<br>88.64 ± 0.26<br>90.29 ± 0.52<br>90.76 ± 0.57<br>87.90 ± 0.55<br>91.74 ± 0.55<br>88.24 ± 0.31<br>92.05 ± 0.50<br>88.88<br>90.59 ± 0.3|95.06 ± 0.32<br>96.07 ± 0.33<br>92.12 ± 0.46<br>95.11 ± 0.21<br>94.03 ± 0.37<br>95.37 ± 0.37<br>95.38 ± 0.31<br>96.36 ± 0.28<br>94.95 ± 0.18<br>96.83 ± 0.27<br>95.34<br>-|
|**BECLR (Ours)**|**28.46** ± 0.23|**34.21** ± 0.25|44.48 ± 0.31|56.89 ± 0.29|**88.55** ± 0.23|**93.92** ± 0.14|**93.65** ± 0.25|**97.72** ± 0.13|



## C.2 IN-DOMAIN SETTING 

Here, we provide more extensive experimental results on miniImageNet, tieredImageNet, CIFAR-FS and FC100 by comparing against additional baselines. Table 9 corresponds to an extended version of Table 1 and, similarly, Table 10 is an extended version of Table 2. We assess the performance of BECLR against a wide variety of methods: from (i) established SSL baselines (Chen et al., 2020a; Caron et al., 2020; Grill et al., 2020; Kalantidis et al., 2020; Chen et al., 2020b; Zbontar et al., 2021; Xu et al., 2021; Chen & He, 2021; Dwibedi et al., 2021) to (ii) state-of-the-art U-FSL approaches (Hsu et al., 2018; Khodadadeh et al., 2019; Medina et al., 2020; Lee et al., 2020; Chen et al., 2021b; Li & Liu, 2021; Ye et al., 2022; Shirekar & Jamali-Rad, 2022; Li et al., 2022; Wang et al., 2022a; Chen et al., 2022; Lu et al., 2022; Jang et al., 2022; Shirekar et al., 2023; Hu et al., 2023a). Furthermore, we compare with a set of supervised baselines (Finn et al., 2017; Snell et al., 2017; Rusu et al., 2018; Gidaris et al., 2019; Lee et al., 2019; Bateni et al., 2022). 

## C.3 CROSS-DOMAIN SETTING 

We also provide an extended version of the Table 12: Accuracies in (% _±_ std) on miniImageNet experimental results in the miniImageNet → → CUB. _[†]_ : denotes our reproduction. Style: **best** and CDFSL cross-domain setting of Table 3, where second best. we compare against additional baselines, as **miniImageNet → CUB** seen in Table 11. Additionally, in Table 12 **Method 5-way 1-shot 5-way 5-shot** we evaluate the performance of BECLR on Meta-GMVAE (SimCLR (Chen et al.Lee et al., 2020a, 2020) ) 38.0938.25 ± 0.47± 0.49 55.6555.89 ± 0.42± 0.46 the miniImageNet → CUB cross-domain setMoCo v2 (Chen et al., 2020b) 39.29 ± 0.47 56.49 ± 0.44 ting. We compare against any existing unsuperBYOL (Grill et al., 2020) 40.63 ± 0.46 56.92 ± 0.43 vised baselines (Hsu et al., 2018; Khodadadeh SwAV _[†]_ (Caron et al., 2020) 38.34 ± 0.51 53.94 ± 0.43 NNCLR _[†]_ (Dwibedi et al., 2021) 39.37 ± 0.53 54.78 ± 0.42 et al., 2019; Chen et al., 2020a; Caron et al., Barlow Twins (Zbontar et al., 2021) 40.46 ± 0.47 57.16 ± 0.42 2020; Medina et al., 2020; Grill et al., 2020; Laplacian Eigenmaps (Chen et al., 2022) 41.08 ± 0.48 58.86 ± 0.45 Chen et al., 2020b; Zbontar et al., 2021; Ye HMS (Ye et al., 2022) 40.75 58.32 et al., 2022; Chen et al., 2022; Lu et al., 2022; PsCo (Jang et al., 2022) - 57.38 ± 0.44 **BECLR (Ours) 43.45** ± 0.50 **59.51** ± 0.46 Shirekar et al., 2023; Shirekar & Jamali-Rad, 2022; Jang et al., 2022; Lee et al., 2020), for which this more challenging cross-domain experiment has been conducted (to the best of our knowledge). 

|econd best<br>.|||
|---|---|---|
||**miniImageNet →CUB**||
|**Method**|**5-way 1-shot**|**5-way 5-shot**|
|Meta-GMVAE (Lee et al.,2020)<br>SimCLR (Chen et al.,2020a)<br>MoCo v2 (Chen et al.,2020b)<br>BYOL (Grill et al.,2020)<br>SwAV_†_ (Caron et al.,2020)<br>NNCLR_†_ (Dwibedi et al.,2021)<br>Barlow Twins (Zbontar et al.,2021)<br>Laplacian Eigenmaps (Chen et al.,2022)<br>HMS (Ye et al.,2022)<br>PsCo(Janget al.,2022)|38.09 ± 0.47<br>38.25 ± 0.49<br>39.29 ± 0.47<br>40.63 ± 0.46<br>38.34 ± 0.51<br>39.37 ± 0.53<br>40.46 ± 0.47<br>41.08 ± 0.48<br>40.75<br>-|55.65 ± 0.42<br>55.89 ± 0.46<br>56.49 ± 0.44<br>56.92 ± 0.43<br>53.94 ± 0.43<br>54.78 ± 0.42<br>57.16 ± 0.42<br>58.86 ± 0.45<br>58.32<br>57.38 ± 0.44|
|**BECLR (Ours)**|**43.45** ± 0.50|**59.51** ± 0.46|



## D COMPLEXITY ANALYSIS 

In this section, we analyze the computational and time complexity of BECLR and compare with different contrastive learning baselines, as summarized in Table 13. Neither DyCE nor OpTA introduce additional trainable parameters; thus, the total parameter count of BECLR is on par with standard Siamese architectures and dependent on the backbone configuration. BECLR utilizes a student-teacher EMA architecture, hence needs to store separate weights for 2 distinct networks, denoted as ResNet 2 _×_ , similar to BYOL (Grill et al., 2020). A batch size of 256 and 512 is used for training BECLR on miniImageNet and tieredImageNet, respectively, which again is standard practice in the U-FSL literature. However, DyCE artificially enhances the batch, on which the contrastive loss is applied, to _k_ + 1 times the size of the original, in effect slightly increasing the training time of BECLR. OpTA also introduces additional calculations in the inference time, which nevertheless results in a negligible increase in terms of the average episode inference time of BECLR. 

20 

Published as a conference paper at ICLR 2024 

Table 13: Comparison of the computational complexity of BECLR with different contrastive SSL approaches in terms of parameter count, training, and inference times. _[†]_ : denotes our reproduction. 

|**Method**|**Backbone Architecture**|**Backbone Parameter Count (M)**|**Training Time (sec/epoch)**|**Inference Time (sec/episode)**|
|---|---|---|---|---|
|SwAV_†_ (Caron et al.,2020)|ResNet-18 (1_×_)|11_._2|124_._51|0_._213|
|NNCLR_†_ (Dwibedi et al.,2021)|ResNet-18 (2_×_)|22_._4|174_._13|0_._211|
|UniSiam (Lu et al.,2022)|ResNet-18 (1_×_)|11_._2|153_._37|0_._212|
|BECLR (Ours)|ResNet-18 (2_×_)|22_._4|190_._20|0_._216|
|SwAV_†_ (Caron et al.,2020)|ResNet-50 (1_×_)|23_._5|136_._82|0_._423|
|NNCLR_†_ (Dwibedi et al.,2021)|ResNet-50 (2_×_)|47_._0|182_._33|0_._415|
|UniSiam (Lu et al.,2022)|ResNet-50 (1_×_)|23_._5|167_._71|0_._419|
|BECLR (Ours)|ResNet-50 (2_×_)|47_._0|280_._53|0_._446|



## E PSEUDOCODE 

This section includes the algorithms for the pretraining methodology of BECLR and the proposed dynamic clustered memory (DyCE) in a Pytorch-like pseudocode format. Algorithm 3 provides an overview of the pretraining stage of BECLR and is equivalent to Algorithm 1, while Algorithm 4 describes the two informational paths of DyCE and is equivalent to Algorithm 2. 

**Algorithm 3:** Unsupervised Pretraining of BECLR: PyTorch-like Pseudocode 

# _{_ f, g, h _}_ student: student backbone, projector, and predictor # _{_ f, g _}_ teacher: teacher backbone and projector # DyCE _{_ student, teacher _}_ : our dynamic clustered memory module for student and teacher paths (see Algorithm. 4) def BECLR(x): # x: a random training mini-batch of L samples x = [x1, x2] = [aug1(x), aug2(x)] # concatenate the two augmented views of x z s = h ~~s~~ tudent( g student( f ~~s~~ tudent( mask(x)))) # (2B _×_ d): extract student representations z t = g ~~t~~ eacher( f teacher(x)).detach() # (2B _×_ d): extract teacher representations z s, z ~~t~~ = DyCE ~~s~~ tudent(z s), DyCE teacher(z ~~t~~ ) # update memory via optimal transport & compute enhanced batch (2B(k+1) _×_ d) loss pos = - (z ~~s~~ _∗_ z t).sum(dim=1).mean() # compute positive loss term loss = loss pos + (matmul(z ~~s~~ , z t.T) _∗_ mask).div(temp).exp().sum(dim=1)).div(n ~~n~~ eg).mean().log() # compute final loss term loss.backward(), momentum update( student.parameters, teacher.parameters) # update student and teacher parameters 

## **Algorithm 4:** Dynamic Clustered Memory (DyCE): PyTorch-like Pseudocode 

# z: batch representations (2B _×_ d) # self.memory: memory embedding space (M _×_ d) # self.prototypes: memory partition prototypes (P _×_ d) def DyCE(self, z): if self.memory.shape[0] == M: # - - - - - **Path I** : Top-k NNs Selection and Batch Enhancement - - - - - if epoch _≥_ epoch ~~t~~ hr batch ~~p~~ rototypes = assign prototypes(z, self.prototypes) # (2B _×_ d): find nearest memory prototype for each batch embedding y mem = topk(self.memory, z, batch prototypes) # (2Bk _×_ d): find top-k NNs, from memory partition of nearest prototype z = [z, y mem] # (2B(k+1) _×_ d): concatenate batch and memory representations to create the final enhanced batch # - - - - - **Path II** : Iterative Memory Updating - - - - - opt ~~p~~ lan = sinkhorn( D(z, self.prototypes)) # get optimal assignments between batch embeddings and prototypes (Solve Eq. 2) self.update(z, opt plan) # add latest batch to memory and update memory partitions and prototypes, using the optimal assignments self.dequeue() # discard the 2B oldest memory embeddings else: self.enqueue(z) # simply store latest batch until the memory is full for the first time return z 

## F IN-DEPTH COMPARISON WITH PSCO 

In this section we perform an comparative analysis of BECLR with PsCo (Jang et al., 2022), in terms of their motivation, similarities, discrepancies, and performance. 

## F.1 MOTIVATION AND DESIGN CHOICES 

BECLR and PsCo indeed share some similarities, in that both methods utilize a student-teacher momentum architecture, a memory module of past representations, some form of contrastive loss, and optimal transport (even though for different purposes). Note that none of these components are unique to neither PsCo nor BECLR, but can be found in the overall U-FSL and SSL literature (He et al., 2020; Dwibedi et al., 2021; Lu et al., 2022; Wang et al., 2022a; Ye et al., 2022). 

21 

Published as a conference paper at ICLR 2024 

Let us now expand on their discrepancies and the unique aspects of BECLR: (i) PsCo is based on meta learning, constructing few-shot classification (i.e., _N_ -way _K_ -shot) tasks, during meta-training and relies on fine-tuning to rapidly adapt to novel tasks during meta-testing. In contrast, BECLR is a contrastive framework based on metric/transfer learning, which focuses on representation quality and relies on OpTA for transferring to novel tasks (no few-shot tasks during pretraining, no finetuning during inference/testing). (ii) PsCo utilizes a simple FIFO memory queue and is oblivious to class-level information, while BECLR maintains a clustered highly-separable (as seen in Fig. 6) latent space in DyCE, which, after an adaptation period, is used for sampling meaningful positives. (iii) PsCo applies optimal transport for creating pseudolabels (directly from the unstructured memory queue) for _N ∗K_ support embeddings, in turn used as a supervisory signal to enforce consistency between support (drawn from teacher) and query (student) embeddings of the created pseudolabeled task. In stark contrast, BECLR artificially enhances the batch with additional positives and applies an instance-level contrastive loss to enforce consistency between the original and enhanced (additional) positive pairs. After each training iteration, optimal transport is applied to update the stored clusters within DyCE in an equipartitioned fashion with embeddings from the current batch. (iv) Finally, BECLR also incorporates optimal transport (in OpTA) to align the distributions between support and query sets, during inference, which does not share similarity with the end-to-end pipeline of PsCo. 

## F.2 PERFORMANCE AND ROBUSTNESS 

We conduct an additional experiment, summarized in Table 14, to study the impact of adding components of BECLR (symmetric loss, masking, OpTA) to PsCo and the robustness of BECLR. For the purposes of this experiment, we reproduced PsCo on a deeper ResNet-18 backbone, ensuring fairness in our comparisons. Next, we modified its loss function to be symmetric (allowing both augmentations of _X_ to pass through both branches) similar to BECLR, (this is denoted as PsCo+). Next, we also add patchwise masking to PsCo (denoted as PsCo++) and as an additional step we have also added 

Table 14: Accuracies in (% _±_ std) on miniImageNet. _[†]_ : denotes our reproduction. Style: **best** and second best. 

||**Method**<br>PsCo(Jang et al.,2022)<br>PsCo_†_ (Jang et al.,2022)|**Backbone**<br>Conv5<br>RN18|**5 way 1 shot**<br>46.70 ± 0.42<br>47.24 ± 0.46|**5 way 5 shot**<br>63.26 ± 0.37<br>65.48 ± 0.38|
|---|---|---|---|---|
||PsCo+_†_ (Jang et al.,2022)<br>PsCo++_†_ (Janget al.,2022)<br>PsCow/OpTA_†_ (Jang et al.,2022)<br>PsCo+w/OpTA_†_ (Jang et al.,2022)|RN18<br>RN18<br>RN18<br>RN18|47.86 ± 0.44<br>47.58 ± 0.45<br>52.89 ± 0.61<br>54.43 ± 0.59|65.95 ± 0.37<br>65.74 ± 0.38<br>67.42 ± 0.51<br>68.31 ± 0.52|
||PsCo++w/OpTA_†_ (Janget al.,2022)|RN18|54.35 ± 0.60|68.43 ± 0.52|
||BECLR(Ours)<br>BECLR-(Ours)<br>BECLR-- (Ours)|RN18<br>RN18<br>RN18|**75.74 ± 0.62**<br>74.37 ± 0.61<br>73.65 ± 0.61|**84.93 ± 0.33**<br>84.19 ± 0.31<br>83.63 ± 0.31|
||BECLRw/oOpTA(Ours)|RN18|66.14 ± 0.43|84.32 ± 0.27|
||BECLR-w/oOpTA(Ours)<br>BECLR--w/oOpTA(Ours)|RN18<br>RN18|65.26 ± 0.41<br>64.68 ± 0.44|83.68 ± 0.25<br>83.45 ± 0.26|



our novel module OpTA on top of all these models (PsCo w/ OpTA), during inference. We observe that neither the symmetric loss nor masking offer meaningful improvements in the performance of PsCo. On the contrary, OpTA yields a significant performance boost (up to 6 _._ 5% in the 1-shot setting, in which the sample bias is most severe). This corroborates our claim that our proposed OpTA should be considered as an add-on module to every (U-)FSL approach out there. 

As an additional robustness study we take the opposite steps for BECLR, first removing the patchwise masking (denoted as BECLR-) and then also the symmetric loss (denoted as BECLR--), noticing that both slightly degrade the performance, as seen in Table 14, which confirms our design choices to include them. Similarly, we also remove OpTA from BECLR’s inference stage. The most important takeaway here is that the most degraded version of BECLR (BECLR-- w/o OpTA) still outperforms the best enhanced version of PsCo ( PsCo++ w/ OpTA), which we believe offers an additional perspective on the advantages of adopting BECLR over PsCo. 

## F.3 COMPUTATIONAL COMPLEXITY 

Finally, we also compare BECLR and PsCo in terms of model size (total number of parameters in M) and inference times (in sec/episode), when using the same ResNet-18 backbone architecture. The results are summarized in Table 15. We notice that BECLR’s total parameter count is, in fact, lower than that of PsCo. Regarding inference time, BECLR is slightly slower in comparison, but this difference is negligible in real-time inference scenarios. 

22 

Published as a conference paper at ICLR 2024 

Table 15: Comparison of the computational complexity of BECLR with PsCo (Jang et al., 2022) in terms of total model parameter count and inference times. 

|**Method**|**Backbone**|**Model Total Parameter Count (M)**|**Inference Time (sec/episode)**|
|---|---|---|---|
|PsCo (Jang et al.,2022)|RN18|30.766|0.082|
|BECLR (Ours)|RN18|24.199|0.216|



## G COMPARISON WITH SUPERVISED FSL 

In this work we have focused on unsupervised few-shot learning and demonstrated how BECLR sets a new state-of-the-art in this exciting space. In this section, as an additional study, we are also comparing BECLR with a group of most recent supervised FSL baselines, which also have access to the base-class labels in the pretraining stage. 

## G.1 PERFORMANCE EVALUATION 

In Table 16 we compare BECLR with seven recent baselines from the supervised FSL state-of-the-art (Lee et al., 2019; Bateni et al., 2022; He et al., 2022; Hiller et al., 2022; Bendou et al., 2022; Singh & Jamali-Rad, 2022; Hu et al., 2023b), in terms of their in-domain performance on miniImageNet and tieredImageNet. As can be seen, some of these supervised methods do outperform BECLR, yet these baselines are heavily engineered towards the target dataset in the in-domain setting (where the target classes still originate from the pretraining dataset). Another interesting observation here is that it turns out the top performing supervised baselines are all _transductive_ methodologies (i.e., learn from both labeled and unlabeled data at the same time), and such a transductive episodic pretraining cannot be established in a fully unsupervised pretraining strategy as in BECLR Notice that BECLR can even outperform recent inductive supervised FSL approaches, _even without access to base-class labels_ during pretraining. 

Table 16: Accuracies (in % _±_ std) on miniImageNet and tieredImageNet compared against supervised FSL baselines. Pretrainig Setting: unsupervised (Unsup.) and supervised (Sup.) pretraining. Approach: Ind.: inductive setting, Transd.: transductive setting. Style: **best** and second best. 

|inductive setting,Transd.: transductive setting. Style: **best**andsecond best<br>.||
|---|---|
|**miniImageNet**|**tieredImageNet**|
|**Method**<br>**Setting**<br>**Approach**<br>**5 way 1 shot**<br>**5 way 5 shot**|**5 way 1 shot**<br>**5 way 5 shot**|
|**BECLR (Ours)**<br>Unsup.<br>Ind.<br>**80.57 ± 0.57**<br>**87.82 ± 0.29**|**81.69 ± 0.61**<br>87.86 ± 0.32|
|MetaOptNet (Lee et al.,2019)<br>Sup.<br>Ind.<br>64.09 ± 0.62<br>80.00 ± 0.45<br>HCTransformers (He et al.,2022)<br>Sup.<br>Ind.<br>74.74 ± 0.17<br>85.66 ± 0.10<br>FewTURE (Hiller et al.,2022)<br>Sup.<br>Ind.<br>72.40 ± 0.78<br>86.38 ± 0.49<br>EASY (inductive) (Bendou et al.,2022)<br>Sup.<br>Ind.<br>70.63 ± 0.20<br>86.28 ± 0.12|65.99 ± 0.72<br>81.56 ± 0.53<br>79.67 ± 0.20<br>89.27 ± 0.13<br>76.32 ± 0.87<br>**89.96 ± 0.55**<br>74.31 ± 0.22<br>87.86 ± 0.15|
|EASY (transductive) (Bendou et al.,2022)<br>Sup.<br>Transd.<br>82.31 ± 0.24<br>88.57 ± 0.12<br>Transductive CNAPS (Bateni et al.,2022)<br>Sup.<br>Transd.<br>55.60 ± 0.90<br>73.10 ± 0.70<br>BAVARDAGE (Hu et al.,2023b)<br>Sup.<br>Transd.<br>84.80 ± 0.25<br>91.65 ± 0.10<br>TRIDENT (Singh & Jamali-Rad,2022)<br>Sup.<br>Transd.<br>**86.11 ± 0.59**<br>**95.95 ± 0.28**|83.98 ± 0.24<br>89.26 ± 0.14<br>65.90 ± 1.10<br>81.80 ± 0.70<br>85.20 ± 0.25<br>90.41 ± 0.14<br>**86.97 ± 0.50**<br>**96.57 ± 0.17**|



## G.2 ADDITIONAL INTUITION 

Let us now try to provide some high-level intuition as to why BECLR can outperform such supervised baselines. We argue that self-supervised pretraining helps generalization to the unseen classes, whereas supervised training heavily tailors the model towards the pretraining classes. This is also evidenced by their optimization objectives. In particular, supervised pretraining maximizes the mutual information _I_ ( _**Z** ,_ _**y**_ ) between representations _**Z**_ and base-class labels _**y**_ , whereas BECLR maximizes the mutual information _I_ ( _**Z**_ **1** _,_ _**Z**_ **2** ) between different augmented views _**Z**_ **1** _,_ _**Z**_ **2** of the input images _**X**_ , which is a lower bound of the mutual information _I_ ( _**Z** ,_ _**X**_ ) between representations _**Z**_ and raw data/ images _**X**_ . As such, BECLR _potentially_ has a higher capacity to learn more discriminative and generalizable features. By the way, this is not the case only for BECLR, many other unsupervised FSL approaches report similar behavior, e.g. (Chen et al., 2021a; Lu et al., 2022; Hu et al., 2023a). Next to that, pure self-supervised learning approaches also report a similar observation where they can outperform supervised counterparts due to better generalization to different downstream tasks, e.g. (H´enaff et al., 2021; Zhang et al., 2022). 

23 

Published as a conference paper at ICLR 2024 

Another notable reason why BECLR can outperform some of its supervised counterparts is our novel module OpTA specifically designed to address sample bias, a problem that both supervised and unsupervised FSL approaches suffer from and typically overlook. So comes our claim that the proposed OpTA should becomes an integral part of all (U-)FSL approaches, especially in low-shot scenarios where FSL approaches suffer from samples bias the most. 

24 

