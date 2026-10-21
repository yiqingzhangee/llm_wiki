Published as a conference paper at ICLR 2023 

## NEURAL COLLAPSE INSPIRED FEATURE-CLASSIFIER ALIGNMENT FOR FEW-SHOT CLASS INCREMENTAL LEARNING 

**Yibo Yang**[1] _[∗]_ **, Haobo Yuan**[2] _[∗]_ **, Xiangtai Li**[3] **, Zhouchen Lin**[3] _[,]_[4] _[,]_[5] _[†]_ **, Philip Torr**[6] **, Dacheng Tao**[1] 1JD Explore Academy 2School of Computer Science, Wuhan University 3National Key Lab of General AI, School of Intelligence Science and Technology, Peking University 4Institute for Artificial Intelligence, Peking University 5Peng Cheng Laboratory 6University of Oxford _†_ : corresponding author *: equal contribution 

## ABSTRACT 

Few-shot class-incremental learning (FSCIL) has been a challenging problem as only a few training samples are accessible for each novel class in the new sessions. Finetuning the backbone or adjusting the classifier prototypes trained in the prior sessions would inevitably cause a misalignment between the feature and classifier of old classes, which explains the well-known catastrophic forgetting problem. In this paper, we deal with this misalignment dilemma in FSCIL inspired by the recently discovered phenomenon named _neural collapse_ , which reveals that the last-layer features of the same class will collapse into a vertex, and the vertices of all classes are aligned with the classifier prototypes, which are formed as a simplex equiangular tight frame (ETF). It corresponds to an optimal geometric structure for classification due to the maximized Fisher Discriminant Ratio. We propose a neural collapse inspired framework for FSCIL. A group of classifier prototypes are pre-assigned as a simplex ETF for the whole label space, including the base session and all the incremental sessions. During training, the classifier prototypes are not learnable, and we adopt a novel loss function that drives the features into their corresponding prototypes. Theoretical analysis shows that our method holds the neural collapse optimality and does not break the feature-classifier alignment in an incremental fashion. Experiments on the miniImageNet, CUB-200, and CIFAR-100 datasets demonstrate that our proposed framework outperforms the state-of-the-art performances. Code address: https://github.com/NeuralCollapseApplications/FSCIL 

## 1 INTRODUCTION 

Learning incrementally and learning with few-shot data are common in the real-world implementations, and in many applications, such as robotics, the two demands emerge simultaneously. Despite the great success in a closed label space, it is still challenging for a deep learning model to learn new classes continually with only limited samples (LeCun et al., 2015). To this end, few-shot classincremental learning (FSCIL) was proposed to tackle this problem (Tao et al., 2020b). 

Compared with few-shot learning (Ravi & Larochelle, 2017; Vinyals et al., 2016), FSCIL transfers a trained model into new label spaces incrementally. It also differs from incremental learning (Cauwenberghs & Poggio, 2000; Li & Hoiem, 2017; Rebuffi et al., 2017) in that there are only a few (usually 5) samples accessible for each new class in the incremental sessions. For each session’s evaluation, the model is required to infer test images coming from all the classes that have been encountered. The base session of FSCIL contains a large label space and sufficient training samples, while each incremental session only has a few novel classes and labeled images. It poses the notorious _catastrophic forgetting_ problem (Goodfellow et al., 2013) because the novel sessions have no access to the data of the previous sessions. 

Due to the importance and difficulty, FSCIL has attracted much research attention. The initial solutions to FSCIL finetune the network on new session data with distillation schemes to reduce the forgetting of old classes (Tao et al., 2020b; Dong et al., 2021). However, the few-shot data in novel sessions can easily induce over-fitting. Following studies favor training a backbone network on the 

1 

Published as a conference paper at ICLR 2023 


![](raw/papers/2_LLM_NLP/LLM_paper_list/few-shot/iclr_2023_Neural_Collapse_Inspired_Feature-Classifier_Alignment_for_Few-Shot_Class-Incremental_Learning_images/iclr_2023_Neural_Collapse_Inspired_Feature-Classifier_Alignment_for_Few-Shot_Class-Incremental_Learning.pdf-0002-01.png)


**----- Start of picture text -----**<br>
Base Session Session 1 Session 2<br>(new)<br>(new)<br>(a) prior studies: classifier prototypes are learnable<br>(b) our solution: classifier prototypes are pre-assigned and fixed<br>current prototypes prototypes in the last session misalignment<br>**----- End of picture text -----**<br>


Figure 1: A popular choice in prior studies is to evolve the old-class prototypes via delicate design of loss or regularizer to keep them separated from novel-class prototypes, but will cause misalignment. As a comparison, we pre-assign and fix an optimal feature-classifier alignment, and then train a model towards the same neural collapse optimality in each session to avoid target conflict. 

base session as a feature extractor (Zhang et al., 2021; Hersche et al., 2022; Aky¨urek et al., 2022). For novel sessions, the backbone network is fixed and a group of novel-class prototypes (classifier vectors) are learned incrementally. But as shown in Figure 1 (a), the newly added prototypes may lie close to the old-class prototypes, which impedes the ability to discriminate between the old-class and the novel-class samples in evaluation. As a result, adjusting the classifier prototypes is always necessary for two goals: ( _i_ ) keep a sufficient distance between the old-class and the novel-class prototypes; ( _ii_ ) prevent the adjusted old-class prototypes from shifting far away from their original positions. However, the two goals rely on sophisticated loss functions or regularizers (Chen & Lee, 2021; Hersche et al., 2022; Aky¨urek et al., 2022), and are hard to attain simultaneously without qualification. Besides, as shown in Figure 1 (a), there will be a misalignment between the adjusted classifier and the fixed features of old classes. A recent study proposes to reserve feature space for novel classes to circumvent their conflict with old classes (Zhou et al., 2022a), but an optimal feature-classifier alignment is hard to be guaranteed with learnable classifier (Pernici et al., 2021). 

We point out that it is the misalignment dilemma between feature and classifier that causes the catastrophic forgetting problem of old classes. If a backbone network is finetuned in novel sessions, the features of old classes will be easily deviated from their classifier prototypes. Alternatively, when a backbone network is fixed and a group of new prototypes for novel classes are learned incrementally, the adjustment of old-class prototypes will also induce misalignments with their fixed features. In this paper, we pose and study the following question, 

_“Can we look for and pre-assign an optimal feature-classifier alignment such that the model is optimized towards the fixed optimality, so avoids conflict among sessions?”_ 

## 1.1 MOTIVATIONS AND CONTRIBUTIONS 

Neural collapse is a recently discovered phenomenon that at the terminal phase of training (after 0 training error rate), the last-layer features of the same class will collapse into a single vertex, and the vertices of all classes will be aligned with their classifier prototypes and be formed as a simplex equiangular tight frame (ETF) (Papyan et al., 2020). A simplex ETF is a geometric structure of _K_ vectors in R _[d]_ , _d ≥ K−_ 1. All vectors have the same _ℓ_ 2 norm of 1 and any pair of two different vectors has an inner product of _− K_ 1 _−_ 1[,][which][corresponds][to][the][largest][possible][angle][of] _[K]_[equiangular] vectors. Particularly when _d_ = _K −_ 1, a simplex ETF reduces to a regular simplex such as triangle and tetrahedron. It describes an optimal geometric structure for classification due to the minimized within-class variance and the maximized between-class variance (Martinez & Kak, 2001), which indicates that the Fisher Discriminant Ratio (Fisher, 1936; Rao, 1948) is maximized. Following studies aim to theoretically explain this phenomenon (Fang et al., 2021; Han et al., 2022). 

2 

Published as a conference paper at ICLR 2023 

It is expected that imperfect training condition, such as imbalance, cannot induce neural collapse and will cause deteriorated performance (Fang et al., 2021; Yang et al., 2022b). Training in an incremental fashion will also break the neural collapse optimality. Since neural collapse offers us an optimal structure where features and their classifier prototypes are aligned, we can pre-assign such a structure and learn the model towards the optimality. Inspired by this insight, in this paper, we initialize a group of classifier prototypes **W**[ˆ] ETF _∈_ R _[d][×]_[(] _[K]_[0][+] _[K][′]_[)] as a simplex ETF for the whole label space, where _K_ 0 is the number of classes in the base session and _K[′]_ is the number of classes in all the incremental sessions. As shown in Figure 1 (b), it serves as the optimization target and keeps fixed throughout all sessions training. We append a projection layer after the backbone network and store the mean latent feature of each class output by the backbone in a memory. In the training of incremental sessions, we only finetune the projection layer using a novel loss function that drives the final features towards their corresponding target prototypes. Without bells and whistles, our method achieves superior performances and relieves the catastrophic forgetting problem. 

The contributions of this paper can be summarized as follows: 

- To relieve the misalignment dilemma in FSCIL, we propose to pre-assign an optimal alignment inspired by neural collapse as a fixed target throughout the incremental learning. Our model is trained towards the same optimality to avoid optimization conflict among sessions. 

- We fix the prototypes and apply a novel loss function that only finetunes a projection layer to drive the output features into their corresponding prototypes. Theoretical and empirical analyses show that our method better holds the neural collapse optimality. 

- Experiments on miniImageNet, CIFAR-100, and CUB-200 demonstrate that our method is able to surpass the state-of-the-art performances. In particular, our method achieves an average accuracy improvement of more than 3.5% over a recent strong baseline on both miniImageNet and CIFAR-100. 

## 2 RELATED WORK 

**Few-shot class-incremental learning (FSCIL).** As a variant of class-incremental learning (CIL) (Cauwenberghs & Poggio, 2000; Li & Hoiem, 2017; Rebuffi et al., 2017), FSCIL only has a few novel-classes and training data in each incremental session (Tao et al., 2020b; Dong et al., 2021), which increases the tendency of overfitting on novel classes (Snell et al., 2017; Sung et al., 2018). Both CIL and FSCIL require a delicate balance between well adapting a model to novel classes and less forgetting of old classes (Zhao et al., 2021). A popular choice is to use meta learning (Yoon et al., 2020; Chi et al., 2022; Zhou et al., 2022b). Some studies try to make base and incremental sessions compatible via pseudo-feature (Cheraghian et al., 2021b; Zhou et al., 2022a), augmentation (Peng et al., 2022), or looking for a flat minima (Shi et al., 2021). For training in incremental sessions, the new prototypes for novel classes should be separable from the old-class prototypes. Meanwhile, the adjustment of old-class prototypes should not induce large shifts. Current studies widely rely on evolving the prototypes (Zhang et al., 2021; Zhu et al., 2021a) or sophisticated designs of loss and regularizer (Ren et al., 2019; Hou et al., 2019; Tao et al., 2020a; Joseph et al., 2022; Lu et al., 2022; Hersche et al., 2022; Chen & Lee, 2021; Aky¨urek et al., 2022; Yang et al., 2022a). However, the two goals have inherent conflict, and a tough effort to balance the loss terms is necessary. In contrast, our method pre-assigns and fixes a feature-classifier alignment as an optimality. A model is trained towards the same target in all sessions. **We only use a single loss without any regularizer** . 

**Neural collapse.** Neural collapse describes an elegant geometric structure of the last-layer feature and classifier in a well-trained model (Papyan et al., 2020). It inspires later studies to theoretically explain this phenomenon. Based on a simplified model that only considers the last-layer optimization, neural collapse is proved to be the global optimality of balanced training with the CE (Weinan & Wojtowytsch, 2020; Graf et al., 2021; Lu & Steinerberger, 2020; Fang et al., 2021; Zhu et al., 2021b; Ji et al., 2022) and the MSE (Mixon et al., 2020; Poggio & Liao, 2020; Zhou et al., 2022c; Han et al., 2022; Tirer & Bruna, 2022) loss functions. Recent studies try to induce neural collapse in imbalanced training by fixing a classifier (Yang et al., 2022b; Zhong et al., 2023) or novel loss (Xie et al., 2023). Our method is inspired by Yang et al. (2022b), but we apply the classifier in an incremental fashion. Galanti et al. (2022) show that neural collapse is still valid when transferring a model into new samples or classes. To the best of our knowledge, **we are the first to study FSCIL from the neural collapse perspective, which offers our method sound interpretability** . 

3 

Published as a conference paper at ICLR 2023 

## 3 BACKGROUND 

## 3.1 FEW-SHOT CLASS-INCREMENTAL LEARNING (FSCIL) 

In real-world applications, one often needs to adapt a model to data coming from a new label space with only a few labeled samples. FSCIL trains a model incrementally on a sequence of training datasets _{D_[(0)] _, D_[(1)] _, . . . , D_[(] _[T]_[ )] _}_ , where _D_[(] _[t]_[)] = _{_ ( _**x** i, yi_ ) _}[|D] i_ =1[(] _[t]_[)] _[|]_[,] _[ D]_[(0)][is the base session, and] _[ T]_[the] number of incremental sessions. The base session _D_[(0)] usually contains a large label space _C_[(0)] and sufficient training images for each class _c ∈C_[(0)] . In each incremental session _D_[(] _[t]_[)] , _t >_ 0, there are only a few labeled images and we have _|D_[(] _[t]_[)] _|_ = _pq_ , where _p_ is the number of classes and _q_ is the number samples per novel class, known as _p_ -way _q_ -shot. The label space _C_[(] _[t]_[)] has no overlap with any other session, _i.e., C_[(] _[t]_[)] _∩C_[(] _[t][′]_[)] = _∅_ , _∀t[′]_ = _t_ . For any incremental session _t >_ 0, we only have access to the data in _D_[(] _[t]_[)] , and the training sets of the previous sessions are not available. For evaluation in session _t_ , the test dataset comes from all the encountered classes in the previous and current sessions[1] , _i.e._ the label space of _∪[t] i_ =0 _[C]_[(] _[i]_[)][.] 

Therefore, FSCIL suffers from severe data scarcity and imbalance. It requires a model to be adaptable to novel classes, and meanwhile keep the ability on old classes. 

## 3.2 NEURAL COLLAPSE 

Neural collapse refers to a phenomenon at the terminal phase of training (after 0 training error rate) on balanced data (Papyan et al., 2020). It reveals a geometric structure formed by the last-layer 

**Definition 1 (Simplex Equiangular Tight Frame)** _A simplex Equiangular Tight Frame (ETF) refers to a matrix that is composed of K vectors in_ R _[d] and satisfies:_ 


![](raw/papers/2_LLM_NLP/LLM_paper_list/few-shot/iclr_2023_Neural_Collapse_Inspired_Feature-Classifier_Alignment_for_Few-Shot_Class-Incremental_Learning_images/iclr_2023_Neural_Collapse_Inspired_Feature-Classifier_Alignment_for_Few-Shot_Class-Incremental_Learning.pdf-0004-08.png)


_where_ **E** = [ **e** 1 _, · · · ,_ **e** _K_ ] _∈_ R _[d][×][K] ,_ **U** _∈_ R _[d][×][K] allows a rotation and satisfies_ **U** _[T]_ **U** = **I** _K,_ **I** _K is the identity matrix, and_ **1** _K is an all-ones vector._ 

_All column vectors in_ **E** _have the same ℓ_ 2 _norm and any pair has an inner produce of − K_ 1 _−_ 1 _[,]_[ i.e.,] 


![](raw/papers/2_LLM_NLP/LLM_paper_list/few-shot/iclr_2023_Neural_Collapse_Inspired_Feature-Classifier_Alignment_for_Few-Shot_Class-Incremental_Learning_images/iclr_2023_Neural_Collapse_Inspired_Feature-Classifier_Alignment_for_Few-Shot_Class-Incremental_Learning.pdf-0004-11.png)


_where δk_ 1 _,k_ 2 = 1 _when k_ 1 = _k_ 2 _, and 0 otherwise._ 

The neural collapse phenomenon includes the following four properties: 

**(NC1)** : The last-layer features of the same class will collapse into their within-class mean, _i.e.,_ the covariance Σ[(] _W[k]_[)] _[→]_ **[0]**[, where][ Σ][(] _W[k]_[)][= Avg] _i[{]_[(] _**[µ]**[k,i][ −]_ _**[µ]**[k]_[)(] _**[µ]**[k,i][ −]_ _**[µ]**[k]_[)] _[T][ }]_[,] _**[ µ]**[k,i]_[ is the feature of sample] _i_ in class _k_ , and _**µ** k_ is the within-class mean of class _k_ features; 

**(NC2)** : The within-class means of all classes centered by the global mean will converge to the vertices of a simplex ETF defined in Definition 1, _i.e.,_ _**µ**_ ˆ _k_ , 1 _≤ k ≤ K_ satisfy Eq. (2), where _**µ**_ ˆ _k_ = ( _**µ** k −_ _**µ** G_ ) _/∥_ _**µ** k −_ _**µ** G∥_ and _**µ** G_ is the global mean; 

**(NC3)** : The within-class means centered by the global mean will be aligned with (parallel to) their corresponding classifier weights, which means the classifier weights will converge to the same simˆ plex ETF, _i.e.,_ _**µ** k_ = _**w** k/∥_ _**w** k∥_ , 1 _≤ k ≤ K_ , where _**w** k_ is the classifier weight of class _k_ ; 

**(NC4)** : When **(NC1)** - **(NC3)** hold, the model prediction using logits can be simplified to the nearest class centers[2] , _i.e.,_ arg max _k⟨_ _**µ** ,_ _**w** k⟩_ = arg min _k ||_ _**µ** −_ _**µ** k||_ , where _⟨·⟩_ is the inner product operator, _**µ**_ is the last-layer feature of a sample for prediction. 

Neural collapse corresponds to an optimal feature-classifier alignment for classification due to the maximized Fisher Discriminant Ratio (between-class variance to within-class variance). 

> 1Different from task-incremental learning, we do not know which session a test sample comes from. 

> 2We omit the bias term in a linear classifier layer for simplicity. 

4 

Published as a conference paper at ICLR 2023 

## 4 METHOD 

Neural collapse tells us an optimal geometric structure for classification problems where the lastlayer features and classifier prototype of the same class are aligned, and those of different classes are **maximally separated** . However, this structure will be broken in imperfect training conditions, such as imbalanced training data (Fang et al., 2021; Yang et al., 2022b). As illustrated in Figure 1 (a), training in an incremental fashion will also break the neural collapse optimality. Inspired by this perspective, what we should do for FSCIL is to keep the neural collapse inspired feature-classifier alignment as sound as possible. Concretely, we adopt a fixed classifier and a novel loss function as described in Section 4.1 and Section 4.2, respectively. We introduce our framework for FSCIL in Section 4.3. Finally, in Section 4.4, we conduct theoretical analysis to show how our method better holds the neural collapse optimality in an incremental fashion. 

## 4.1 ETF CLASSIFIER 

Assume that the base session contains a label space of _K_ 0 classes, each incremental session has _p_ classes, and we have _T_ incremental sessions in total. The whole label space of this FSCIL problem has _K_ 0 + _K[′]_ classes, where _K[′]_ = _Tp_ , _i.e.,_ we need to learn a model that can recognize samples from _K_ 0 + _K[′]_ classes. We denote a backbone network as _f_ , and then we have _**µ**_ = _f_ ( _**x** , θf_ ), where _**µ** ∈_ R _[d]_ is the output feature of input _**x**_ , and _θf_ is the backbone network parameters. 

A popular choice in current studies learns _f_ and **W**[(0)] using the base session data, where **W**[(0)] _∈_ R _[d][×][K]_[0] is the classifier prototypes for base classes. In incremental sessions _t >_ 0, _f_ is fixed as a feature extractor and only **W**[(] _[t]_[)] _∈_ R _[d][×][p]_ for novel classes is learnable. As shown in Figure 1 (a), one need to adjust _{_ **W**[(0)] _, · · · ,_ **W**[(] _[t]_[)] _}_ via sophisticated loss or regularizer to ensure separation among these prototypes (Aky¨urek et al., 2022; Hersche et al., 2022). But it will inevitably introduce misalignment between the adjusted prototypes and the fixed features of old classes. It is an underlying reason for the catastrophic forgetting problem (Joseph et al., 2022). 

Since neural collapse describes an optimal geometric structure of the last-layer feature and classifier, we pre-assign such an optimality by fixing a learnable classifier as the structure instructed by neural collapse. Following Yang et al. (2022b), we adopt an ETF classifier that initializes a classifier as a simplex ETF and fixes it during training. The difference lies in that we use it in an incremental fashion. Concretely, we randomly initialize classifier prototypes **W**[ˆ] ETF _∈_ R _[d][×]_[(] _[K]_[0][+] _[K][′]_[)] by Eq. (1) for the whole label space, _i.e.,_ the union of classes in all session, _∪[T] i_ =0 _[C]_[(] _[i]_[)][.][We][have] _[K]_[0][=] _[|C]_[(0)] _[|]_ and _K[′]_ =[�] _[T] i_ =1 _[|C]_[(] _[i]_[)] _[|]_[ =] _[ Tp]_[.][Then any pair (] _[k]_[1] _[, k]_[2][) of classifier prototypes in] **W**[ˆ] ETF satisfies: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/few-shot/iclr_2023_Neural_Collapse_Inspired_Feature-Classifier_Alignment_for_Few-Shot_Class-Incremental_Learning_images/iclr_2023_Neural_Collapse_Inspired_Feature-Classifier_Alignment_for_Few-Shot_Class-Incremental_Learning.pdf-0005-07.png)


where **w** ˆ _k_ 1 and **w** ˆ _k_ 2 are two column vectors in **W**[ˆ] ETF. Our ETF classifier ensures that the prototypes of the whole label space have the maximal pair-wise separation. It serves as a fixed target along the incremental training to avoid conflict among sessions. We only need to learn a model whose output features are aligned with this pre-assigned structure. 

## 4.2 DOT-REGRESSION LOSS 

The gradient of cross entropy (CE) loss with respect to the last-layer feature is composed of a pull term that drives the feature into its classifier prototype of the same class, and a push term that pushes it away from the prototypes of different classes. As pointed out by Yang et al. (2022b), when the classifier prototypes are fixed as an optimality, the pull term is always accurate towards the solution, and we can drop the push gradient that may be inaccurate. Accordingly, we adopt a novel loss named dot-regression (DR) loss that can be formulated as (Yang et al., 2022b): 


![](raw/papers/2_LLM_NLP/LLM_paper_list/few-shot/iclr_2023_Neural_Collapse_Inspired_Feature-Classifier_Alignment_for_Few-Shot_Class-Incremental_Learning_images/iclr_2023_Neural_Collapse_Inspired_Feature-Classifier_Alignment_for_Few-Shot_Class-Incremental_Learning.pdf-0005-11.png)


ˆ ˆ where _**µ** i_ is the normalized feature, _i.e.,_ _**µ** i_ = _**µ** i/∥_ _**µ** i∥_ , _**µ** i_ = _f_ ( _**x** i, θf_ ), _yi_ is the label of input _**x** i_ , is an average over a batch of input **w** ˆ _yi_ is the fixed prototype in **W**[ˆ] ETF _**x** i_ for class. The gradient of Eq. _yi_ , and we have(4) with respect to _∥_ **w** ˆ _yi∥_ = 1 by Eq. _**µ**_ ˆ _i_ (3).takes the form of:The total loss 

5 

Published as a conference paper at ICLR 2023 


![](raw/papers/2_LLM_NLP/LLM_paper_list/few-shot/iclr_2023_Neural_Collapse_Inspired_Feature-Classifier_Alignment_for_Few-Shot_Class-Incremental_Learning_images/iclr_2023_Neural_Collapse_Inspired_Feature-Classifier_Alignment_for_Few-Shot_Class-Incremental_Learning.pdf-0006-01.png)


**----- Start of picture text -----**<br>
*𝐰(!<br>*𝝁)<br>for the base session  for all sessions  ℒ<br>(𝑡= 0) (0 ≤𝑡≤𝑇)<br>𝑓 𝒉$ 𝑔<br>ℓ' normalize<br>𝒟 [(")] backbone  ⋮ 𝒞 [($%&)] projection layer #𝐖%&'<br>(0 ≤𝑡≤𝑇) network<br>𝒞 [(")] forward propagation<br>backward update<br>(1 ≤𝑡≤𝑇)<br>fixed ETF classifier prototypes<br>**----- End of picture text -----**<br>


Figure 2: An illustration of our NC-FSCIL. _**h** i_ is the intermediate feature from the backbone network _f_ . _**µ**_ ˆ _i_ is the normalized output feature after the projection layer _g_ . **W**[ˆ] ETF is the ETF classifier that contains prototypes of the whole label space and serves as a fixed target throughout the incremental training. _L_ denotes the dot-regression loss function. _f_ is frozen in the incremental sessions (1 _≤ t ≤ T_ ). A small memory of old-class features is widely adopted in prior studies such as Cheraghian et al. (2021a), Chen & Lee (2021), Aky¨urek et al. (2022), and Hersche et al. (2022). 

ˆ ˆ ˆ direction _∂L/∂_ _**µ** i_ =of _−_ **w** ˆ(1 _yi −_ , whichcos ∠is(ˆ _**µ**_ a _i,_ pre-assigned **w** _yi_ )) ˆ **w** _yi_ . Ittargetis shownprototype.that theFinally,gradientthepullsconvergedfeaturefeatures _**µ** i_ towardswillthebe aligned with **W**[ˆ] ETF, and thus the geometric structure instructed by neural collapse is attained. The theoretical advantage of the DR loss has been proved in Yang et al. (2022b). In experiments, we will compare the DR loss with the CE loss to show its effectiveness in FSCIL. 

## 4.3 NC-FSCIL 

Based on the ETF classifier and the DR loss, we now introduce our neural collapse inspired framework for few-shot class-incremental learning (NC-FSCIL). As shown in Figure 2, our model is composed of two components, a backbone network _f_ and a projection layer _g_ . The backbone network _f_ takes the training data _**x** i_ as input, and outputs an intermediate feature _**h** i_ . The projection layer _g_ can be a linear transformation or an MLP block following Hersche et al. (2022); Peng et al. (2022). It projects the intermediate feature _**h** i_ into _**µ** i_ . Finally, we perform an _ℓ_ 2 normalization on _**µ** i_ to get the output feature _**µ**_ ˆ _i_ , _i.e.,_ 


![](raw/papers/2_LLM_NLP/LLM_paper_list/few-shot/iclr_2023_Neural_Collapse_Inspired_Feature-Classifier_Alignment_for_Few-Shot_Class-Incremental_Learning_images/iclr_2023_Neural_Collapse_Inspired_Feature-Classifier_Alignment_for_Few-Shot_Class-Incremental_Learning.pdf-0006-06.png)


where _θf_ and _θg_ denote the parameters of theˆ backbone network and the projection layer, respectively. We use the normalized output feature _**µ** i_ to compute error signal by Eq. (4). 

In the base session _t_ = 0, we jointly train both _f_ and _g_ using the base session data. The empirical risk to minimize in the base session can be formulated as: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/few-shot/iclr_2023_Neural_Collapse_Inspired_Feature-Classifier_Alignment_for_Few-Shot_Class-Incremental_Learning_images/iclr_2023_Neural_Collapse_Inspired_Feature-Classifier_Alignment_for_Few-Shot_Class-Incremental_Learning.pdf-0006-09.png)


where **W**[ˆ] ETF is the pre-assigned ETF classifier as introduced in Section 4.1, _L_ is the DR loss as introduced in Section 4.2, and _**µ**_ ˆ _i_ is a function of _f_ and _g_ as shown in Eq. (5). 

In each incremental session 1 _≤ t ≤ T_ , we fix the backbone network _f_ as a feature extractor, and only finetune the projection layer _g_ . As a widely adopted practice in FSCIL studies, a small memory of samples or features of old classes can be retained to relieve the overfitting on novel classes (Cheraghian et al., 2021a; Chen & Lee, 2021; Aky¨urek et al., 2022; Hersche et al., 2022). Following Hersche et al. (2022), we only keep a memory _M_[(] _[t]_[)] of the mean intermediate feature _**h** c_ for each old class _c_ . Concretely, we have, 


![](raw/papers/2_LLM_NLP/LLM_paper_list/few-shot/iclr_2023_Neural_Collapse_Inspired_Feature-Classifier_Alignment_for_Few-Shot_Class-Incremental_Learning_images/iclr_2023_Neural_Collapse_Inspired_Feature-Classifier_Alignment_for_Few-Shot_Class-Incremental_Learning.pdf-0006-12.png)


where _f_ has been fixed after the base session. Then we use _D_[(] _[t]_[)] as the input of _f_ , and _M_[(] _[t]_[)] as the input of _g_ to finetune the projection layer _g_ . The empirical risk to minimize in incremental sessions 

6 

Published as a conference paper at ICLR 2023 

can be formulated as: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/few-shot/iclr_2023_Neural_Collapse_Inspired_Feature-Classifier_Alignment_for_Few-Shot_Class-Incremental_Learning_images/iclr_2023_Neural_Collapse_Inspired_Feature-Classifier_Alignment_for_Few-Shot_Class-Incremental_Learning.pdf-0007-02.png)


where _**µ**_ ˆ _i,_ and _**µ**_ ˆ _c_ are the output features of _**x** i_ and _**h** c_ , respectively, _|D_[(] _[t]_[)] _|_ is the number of training samples in session _t_ , and we have _|M_[(] _[t]_[)] _|_ =[�] _[t] j[−]_ =0[1] _[|C]_[(] _[j]_[)] _[|]_[.] **[Thanks to our pre-assigned alignment,] we do not rely on any regularizer in our training** . 

In the evaluation of session _t_ , we predict an input _**x**_ based on the inner product between its output feature _**µ**_ ˆ and the ETF classifier prototypes: arg max _k⟨_ _**µ**_ ˆ _,_ **w** ˆ _k⟩, ∀_ 1 _≤ k ≤ K_ 0 + _K[′] ._ 

## 4.4 THEORETICAL SUPPORTS 

We perform our theoretical work based on a simplified model that drops the backbone network and only keeps the last-layer features and classifier prototypes as independent optimization variables. This simplification has been widely adopted in prior studies to facilitate analysis (Graf et al., 2021; Fang et al., 2021; Zhu et al., 2021b). We investigate the neural collapse optimality of an incremental problem of _T_ sessions with our ETF classifier. Concretely, we consider the following problem, 


![](raw/papers/2_LLM_NLP/LLM_paper_list/few-shot/iclr_2023_Neural_Collapse_Inspired_Feature-Classifier_Alignment_for_Few-Shot_Class-Incremental_Learning_images/iclr_2023_Neural_Collapse_Inspired_Feature-Classifier_Alignment_for_Few-Shot_Class-Incremental_Learning.pdf-0007-07.png)


where **m**[(] _k,i[t]_[)] _[∈]_[R] _[d]_[denotes a feature variable that belongs to the] _[ i]_[-th sample of class] _[ k]_[in session] _[ t]_[,] _nk_ is number of samples in class _k_ , _K_[(] _[t]_[)] is number of classes in session _t_ , _N_[(] _[t]_[)] is the number of samples **W** ˆ ETF _∈_ inRsession _[d][×][K]_ refers to the ETF classifier for the whole label space as introduced in Section 4.1, _t_ , _i.e., N_[(] _[t]_[)] =[�] _[K] k_ =1[(] _[t]_[)] _[n][k]_[,][and] **[M]**[(] _[t]_[)] _[∈]_[R] _[d][×][N]_[(] _[t]_[)][denotes][a][collection][of] _**[m]**_[(] _k,i[t]_[)][.] and we have _K_ =[�] _[T] t_ =0 _[K]_[(] _[t]_[)][.] _[L]_[ can be both the cross entropy and the dot regression loss functions.] 

**Theorem 1** _Let_ **M**[ˆ][(] _[t]_[)] _denote the global minimizer of Eq. (9) by optimizing the model incrementally from t_ = 0 _, and we have_ **M**[ˆ] = [ **M**[ˆ][(0)] _, · · · ,_ **M**[ˆ][(] _[T]_[ )] ] _∈_ R _[d][×]_[�] _t[T]_ =0 _[N]_[ (] _[t]_[)] _. When L in Eq. (9) is CE or DR loss, for any column vector_ **m** ˆ _k,i in_ **M**[ˆ] _whose class label is k, we have:_ 


![](raw/papers/2_LLM_NLP/LLM_paper_list/few-shot/iclr_2023_Neural_Collapse_Inspired_Feature-Classifier_Alignment_for_Few-Shot_Class-Incremental_Learning_images/iclr_2023_Neural_Collapse_Inspired_Feature-Classifier_Alignment_for_Few-Shot_Class-Incremental_Learning.pdf-0007-10.png)


_where K_ =[�] _[T] t_ =0 _[K]_[(] _[t]_[)] _[ denotes the total number of classes of the whole label space,][ δ][k,k][′]_[= 1] _[ when]_ ˆ _k_ = _k[′] and 0 otherwise, and_ **w** _k′ is the prototype of class k[′] in_ **W**[ˆ] ETF _._ 

The proof of Theorem 1 can be found in Appendix A. Eq. **M** ˆ of Eq. (9) satisfies the neural collapse condition, _i.e.,_ features of the same class collapse into a(10) indicates that the global minimizer single vertex, and the vertices of all classes are aligned with **W**[ˆ] ETF as a simplex ETF. It is shown that the feature space is equally separated by prototypes of all classes. More importantly, in problem Eq. (9), the number of classes _K_[(] _[t]_[)] among _T_ + 1 sessions and the number of samples _nk_ among _K_ classes can be imbalanced, which corresponds to the challenging demand of FSCIL. 

## 5 EXPERIMENTS 

In this section, we test our method on FSCIL benchmark datasets including miniImageNet (Russakovsky et al., 2015), CIFAR-100 (Krizhevsky et al., 2009), and CUB-200 (Wah et al., 2011), and compare it with state-of-the-art methods. We also perform ablation studies to validate the effects of ETF classifier and DR loss. Finally, we show the feature-classifier structure achieved by our method. 

## 5.1 IMPLEMENTATION DETAILS 

Please refer to Appendix B for our implementation details. 

7 

Published as a conference paper at ICLR 2023 

Table 1: Performance of FSCIL in each session on miniImageNet and comparison with other studies. The top rows list class-incremental learning and few-shot learning results implemented by Tao et al. (2020b); Zhang et al. (2021) in the FSCIL setting. “Average Acc.” is the average accuracy of all sessions. “Final Improv.” calculates the improvement of our method in the last session. * indicates that the method saves the within-class feature mean of each class for training or inference. 

|**Methods**|**Accuracy in each session (%)**_↑_<br>**Average**<br>**Final**<br>**0**<br>**1**<br>**2**<br>**3**<br>**4**<br>**5**<br>**6**<br>**7**<br>**8**<br>**Acc.**<br>**Improv.**|
|---|---|
|iCaRL (Rebuff et al., 2017)<br>NCM (Hou et al., 2019)<br>D-Cosine (Vinyals et al., 2016)|61.31<br>46.32<br>42.94<br>37.63<br>30.49<br>24.00<br>20.89<br>18.80<br>17.21<br>33.29<br>**+41.1**<br>61.31<br>47.80<br>39.30<br>31.90<br>25.70<br>21.40<br>18.70<br>17.20<br>14.17<br>30.83<br>**+44.14**<br>70.37<br>65.45<br>61.41<br>58.00<br>54.81<br>51.89<br>49.10<br>47.27<br>45.63<br>55.99<br>**+12.68**|
|*TOPIC (Tao et al., 2020b)<br>*IDLVQ (Chen & Lee, 2021)<br>Self-promoted (Zhu et al., 2021a)<br>CEC (Zhang et al., 2021)<br>*LIMIT (Zhou et al., 2022b)<br>*Regularizer (Aky¨urek et al., 2022)<br>MetaFSCIL (Chi et al., 2022)<br>*C-FSCIL (Hersche et al., 2022)<br>Data-free Replay (Liu et al., 2022)<br>*ALICE (Peng et al., 2022)|61.31<br>50.09<br>45.17<br>41.16<br>37.48<br>35.52<br>32.19<br>29.46<br>24.42<br>39.64<br>**+33.89**<br>64.77<br>59.87<br>55.93<br>52.62<br>49.88<br>47.55<br>44.83<br>43.14<br>41.84<br>51.16<br>**+16.47**<br>61.45<br>63.80<br>59.53<br>55.53<br>52.50<br>49.60<br>46.69<br>43.79<br>41.92<br>52.76<br>**+16.39**<br>72.00<br>66.83<br>62.97<br>59.43<br>56.70<br>53.73<br>51.19<br>49.24<br>47.63<br>57.75<br>**+10.68**<br>72.32<br>68.47<br>64.30<br>60.78<br>57.95<br>55.07<br>52.70<br>50.72<br>49.19<br>59.06<br>**+9.12**<br>80.37<br>74.68<br>69.39<br>65.51<br>62.38<br>59.03<br>56.36<br>53.95<br>51.73<br>63.71<br>**+6.58**<br>72.04<br>67.94<br>63.77<br>60.29<br>57.58<br>55.16<br>52.90<br>50.79<br>49.19<br>58.85<br>**+9.12**<br>76.40<br>71.14<br>66.46<br>63.29<br>60.42<br>57.46<br>54.78<br>53.11<br>51.41<br>61.61<br>**+6.90**<br>71.84<br>67.12<br>63.21<br>59.77<br>57.01<br>53.95<br>51.55<br>49.52<br>48.21<br>58.02<br>**+10.10**<br>80.60<br>70.60<br>67.40<br>64.50<br>62.50<br>60.00<br>57.80<br>56.80<br>55.70<br>63.99<br>**+2.61**|
|***NC-FSCIL (ours)**|**84.02**<br>**76.80**<br>**72.00**<br>**67.83**<br>**66.35**<br>**64.04**<br>**61.46**<br>**59.54**<br>**58.31**<br>**67.82**|
|_Improvement over ALICE_|+3.42<br>+6.20<br>+4.60<br>+3.33<br>+3.85<br>+4.04<br>+3.66<br>+2.74<br>+2.61<br>**+3.83**|



Table 2: Performance of FSCIL in each session on CIFAR-100 and comparison with other studies. The top rows list class-incremental learning and few-shot learning results implemented by Tao et al. (2020b); Zhang et al. (2021) in the FSCIL setting. “Average Acc.” is the average accuracy of all sessions. “Final Improv.” calculates the improvement of our method in the last session. 

|**Methods**|**Accuracy in each session (%)**_↑_<br>**Average**<br>**Final**<br>**0**<br>**1**<br>**2**<br>**3**<br>**4**<br>**5**<br>**6**<br>**7**<br>**8**<br>**Acc.**<br>**Improv.**|
|---|---|
|iCaRL (Rebuff et al., 2017)<br>NCM (Hou et al., 2019)<br>D-Cosine (Vinyals et al., 2016)|64.10<br>53.28<br>41.69<br>34.13<br>27.93<br>25.06<br>20.41<br>15.48<br>13.73<br>32.87<br>**+42.38**<br>64.10<br>53.05<br>43.96<br>36.97<br>31.61<br>26.73<br>21.23<br>16.78<br>13.54<br>34.22<br>**+42.57**<br>74.55<br>67.43<br>63.63<br>59.55<br>56.11<br>53.80<br>51.68<br>49.67<br>47.68<br>58.23<br>**+8.43**|
|*TOPIC (Tao et al., 2020b)<br>Self-promoted (Zhu et al., 2021a)<br>CEC (Zhang et al., 2021)<br>DSN (Yang et al., 2022a)<br>*LIMIT (Zhou et al., 2022b)<br>MetaFSCIL (Chi et al., 2022)<br>*C-FSCIL (Hersche et al., 2022)<br>Data-free Replay (Liu et al., 2022)<br>*ALICE (Peng et al., 2022)|64.10<br>55.88<br>47.07<br>45.16<br>40.11<br>36.38<br>33.96<br>31.55<br>29.37<br>42.62<br>**+26.74**<br>64.10<br>65.86<br>61.36<br>57.45<br>53.69<br>50.75<br>48.58<br>45.66<br>43.25<br>54.52<br>**+12.86**<br>73.07<br>68.88<br>65.26<br>61.19<br>58.09<br>55.57<br>53.22<br>51.34<br>49.14<br>59.53<br>**+6.97**<br>73.00<br>68.83<br>64.82<br>62.64<br>59.36<br>56.96<br>54.04<br>51.57<br>50.00<br>60.14<br>**+6.11**<br>73.81<br>72.09<br>67.87<br>63.89<br>60.70<br>57.77<br>55.67<br>53.52<br>51.23<br>61.84<br>**+4.88**<br>74.50<br>70.10<br>66.84<br>62.77<br>59.48<br>56.52<br>54.36<br>52.56<br>49.97<br>60.79<br>**+6.14**<br>77.47<br>72.40<br>67.47<br>63.25<br>59.84<br>56.95<br>54.42<br>52.47<br>50.47<br>61.64<br>**+ 5.64**<br>74.40<br>70.20<br>66.54<br>62.51<br>59.71<br>56.58<br>54.52<br>52.39<br>50.14<br>60.78<br>**+5.97**<br>79.00<br>70.50<br>67.10<br>63.40<br>61.20<br>59.20<br>58.10<br>56.30<br>54.10<br>63.21<br>**+2.01**|
|***NC-FSCIL (ours)**|**82.52**<br>**76.82**<br>**73.34**<br>**69.68**<br>**66.19**<br>**62.85**<br>**60.96**<br>**59.02**<br>**56.11**<br>**67.50**|
|_Improvement over ALICE_|+3.52<br>+6.32<br>+6.24<br>+6.28<br>+4.99<br>+3.65<br>+2.86<br>+2.72<br>+2.01<br>**+4.29**|



## 5.2 PERFORMANCE ON BENCHMARKS 

Our experiment results on minImageNet, CIFAR-100, and CUB-200 are shown in Table 1, Table 2, and Table 4 (Appendix C), respectively. We see that our method achieves the best performance in all sessions on both miniImageNet and CIFAR-100 compared with previous studies. ALICE (Peng et al., 2022) is a recent study that achieves strong performances on FSCIL. Compared with this challenging baseline, we have an improvement of 2.61% in the last session on miniImageNet, and 2.01% on CIFAR-100. We achieve an averaged accuracy improvement of more than 3.5% on both miniImageNet and CIFAR-100. Although we do not surpass ALICE in the last session on CUB-200, we still have the best average accuracy among all methods. As shown in the last rows of Table 1 and Table 2, the improvement of our method lasts and even becomes larger in the first several sessions. It indicates that our method is able to hold the superiority and relieve the forgetting of old sessions. 

## 5.3 ABLATION STUDIES 

We consider three models to validate the effects of ETF classifier and DR loss. All three models are based on the same framework introduced in Section 4.3 including the backbone network, the projection layer, and the memory module. The first model uses a learnable classifier and the CE 

8 

Published as a conference paper at ICLR 2023 

Table 3: Ablation studies on three datasets to investigate the effects of ETF classifier and DR loss. “Learnable+CE” uses a learnable classifier and the CE loss; “ETF+CE” adopts our ETF classifier with the CE loss; “ETF+DR” uses both ETF classifier and DR loss. “FINAL” refers to the accuracy of the last session; “AVERAGE” is the average accuracy of all sessions; “PD” denotes the performance drop, _i.e.,_ the accuracy difference between the first and the last sessions. 


![](raw/papers/2_LLM_NLP/LLM_paper_list/few-shot/iclr_2023_Neural_Collapse_Inspired_Feature-Classifier_Alignment_for_Few-Shot_Class-Incremental_Learning_images/iclr_2023_Neural_Collapse_Inspired_Feature-Classifier_Alignment_for_Few-Shot_Class-Incremental_Learning.pdf-0009-02.png)


**----- Start of picture text -----**<br>
Methods miniImageNet CIFAR-100 CUB-200<br>FINAL ↑ AVERAGE ↑ PD ↓ FINAL ↑ AVERAGE ↑ PD ↓ FINAL ↑ AVERAGE ↑ PD ↓<br>Learnable+CE 50.04 61.30 34.53 52.13 62.68 30.14 50.38 59.58 29.19<br>ETF+CE 56.66 68.23 28.21 54.42 64.00 27.36 56.83 65.51 23.27<br>ETF+DR 58.31 67.82 25.71 56.11 67.50 26.41 59.44 67.28 21.01<br>2 1e 2 Learnable+CE 2.0 1e 3 Learnable+CE 4 1e 2 Learnable+CE 1e 4<br>ETF+DR (Ours) ETF+DR (Ours) 3 ETF+DR (Ours) 6<br>1 1.5 2<br>1 4<br>0 1.0 0<br>1 0.5 12 20<br>2 0.0 34 2 Learnable+CE ETF+DR (Ours)<br>0 1 2 3 4 5 6 7 8 0 1 2 3 4 5 6 7 8 0 1 2 3 4 5 6 7 8 0 1 2 3 4 5 6 7 8<br>Sessions Sessions Sessions Sessions<br>(a) train (each) (b) train (accumulate) (c) test (each) (d) test (accumulate)<br>**----- End of picture text -----**<br>


Figure 3: Average cosine similarity between features and classifier prototypes of different classes, _i.e.,_ Avg _k_ = _k′{_ cos ∠( **m** _k −_ **m** _G,_ **w** _k′_ ) _}_ , where **m** _k_ is the within-class mean of class _k_ features, **m** _G_ denotes the global mean, and **w** _k′_ is the classifier prototype of class _k[′]_ . Statistics are performed among classes in each session (a and c), and all encountered classes by the current session (b and d), on train set (a and b) and test set (c and d), for models trained after each session on miniImageNet. 

loss, which is the most adopted practice. The second model only replaces the classifier with our ETF classifier and also uses the CE loss. The third model corresponds to our method using both ETF classifier and DR loss. As shown in Table 3, when a fixed ETF classifier is used, the final session accuracies are significantly better, and the performance drops get much mitigated. Adopting the DR loss is able to further moderately improve the performances. It indicates that the success of our method is largely attributed to ETF classifier and DR loss, as they pre-assign a neural collapse inspired alignment and drive a model towards the fixed optimality, respectively. 

## 5.4 FEATURE-CLASSIFIER STRUCTURE 

We check the feature-classifier alignment instructed by neural collapse using our method and “Learnable+CE” as a comparison. As shown in Figure 3, the average cosine similarities between features and classifier prototypes of different classes, _i.e.,_ Avg _k_ = _k′{_ cos ∠( **m** _k −_ **m** _G,_ **w** _k[′]_ ) _}_ , of our method are consistently lower than those of the baseline. Most values of our method are negative and close to 0, which is in line with the guidance from neural collapse as derived in Eq. (10). Particularly in Figure 3b and Figure 3d, the average cosine similarities between **m** _k −_ **m** _G_ and **w** _k′_ ( _k_ = _k[′]_ ) among all encountered classes increase fast with session for the baseline method, while ours keep relatively flat. It indicates that the baseline method reduces the feature-classifier margin of different classes as training incrementally, and our method enjoys a stable alignment. As shown in Figure 4 and Figure 5, we also calculate the average cosine similarities between feature and classifier of the same class, _i.e.,_ Avg _k{_ cos ∠( **m** _k −_ **m** _G,_ **w** _k_ ) _}_ , and the trace ratio of within-class covariance to between-class covariance, tr(Σ _W_ ) _/_ tr(Σ _B_ ). These results together support that our method better holds the feature-classifier alignment and relieves the forgetting problem. 

## 6 CONCLUSION 

In this paper, we propose to fix a learnable classifier as a geometric structure instructed by neural collapse for FSCIL. It pre-assigns an optimal feature-classifier alignment as a fixed target throughout incremental training, which avoids optimization conflict among sessions. Accordingly, a novel loss function that drives features towards this pre-assigned optimality is adopted **without** any regularizer. Both theoretical and empirical results support that our method is able to hold the alignment in an incremental fashion, and thus relieve the forgetting problem. In experiments of FSCIL, we achieve and even surpass the state-of-the-art performances on three datasets. 

9 

Published as a conference paper at ICLR 2023 

## ACKNOWLEDGMENTS 

Z. Lin was supported by National Key R&D Program of China (2022ZD0160302), the major key project of PCL, China (No. PCL2021A12), the NSF China (No. 62276004), Qualcomm, and Project 2020BD006 supported by PKU-Baidu Fund. 

## STATEMENTS 

**Ethics Statement.** Our study does NOT involve any of the potential issues such as human subject, public health, privacy, fairness, security, _etc_ . All authors of this paper confirm that they adhere to the ICLR Code of Ethics. 

**Reproducibility Statement.** For our theoretical result Theorem 1, we offer the proof in Appendix A. All datasets used in this paper are public and have been cited. Please refer to Appendix B for the dataset descriptions and the implementation details of our experiments. Our source code is released at https://github.com/NeuralCollapseApplications/FSCIL. 

## REFERENCES 

Afra Feyza Aky¨urek, Ekin Aky¨urek, Derry Wijaya, and Jacob Andreas. Subspace regularizers for few-shot class incremental learning. In _ICLR_ , 2022. 

- Francisco M Castro, Manuel J Mar´ın-Jim´enez, Nicol´as Guil, Cordelia Schmid, and Karteek Alahari. End-to-end incremental learning. In _ECCV_ , pp. 233–248, 2018. 

Gert Cauwenberghs and Tomaso Poggio. Incremental and decremental support vector machine learning. In _NeurIPS_ , volume 13, 2000. 

- Kuilin Chen and Chi-Guhn Lee. Incremental few-shot learning via vector quantization in deep embedded space. In _ICLR_ , 2021. 

- Ali Cheraghian, Shafin Rahman, Pengfei Fang, Soumava Kumar Roy, Lars Petersson, and Mehrtash Harandi. Semantic-aware knowledge distillation for few-shot class-incremental learning. In _CVPR_ , pp. 2534–2543, 2021a. 

- Ali Cheraghian, Shafin Rahman, Sameera Ramasinghe, Pengfei Fang, Christian Simon, Lars Petersson, and Mehrtash Harandi. Synthesized feature based few-shot class-incremental learning on a mixture of subspaces. In _ICCV_ , pp. 8661–8670, 2021b. 

- Zhixiang Chi, Li Gu, Huan Liu, Yang Wang, Yuanhao Yu, and Jin Tang. Metafscil: A meta-learning approach for few-shot class incremental learning. In _CVPR_ , pp. 14166–14175, 2022. 

- Songlin Dong, Xiaopeng Hong, Xiaoyu Tao, Xinyuan Chang, Xing Wei, and Yihong Gong. Fewshot class-incremental learning via relation knowledge distillation. In _AAAI_ , volume 35, pp. 1255– 1263, 2021. 

- Cong Fang, Hangfeng He, Qi Long, and Weijie J Su. Exploring deep neural networks via layerpeeled model: Minority collapse in imbalanced training. _Proceedings of the National Academy of Sciences_ , 118(43), 2021. 

- Ronald A Fisher. The use of multiple measurements in taxonomic problems. _Annals of eugenics_ , 7 (2):179–188, 1936. 

- Tomer Galanti, Andr´as Gy¨orgy, and Marcus Hutter. On the role of neural collapse in transfer learning. In _ICLR_ , 2022. 

- Ian J Goodfellow, Mehdi Mirza, Da Xiao, Aaron Courville, and Yoshua Bengio. An empirical investigation of catastrophic forgetting in gradient-based neural networks. _arXiv preprint arXiv:1312.6211_ , 2013. 

- Florian Graf, Christoph Hofer, Marc Niethammer, and Roland Kwitt. Dissecting supervised constrastive learning. In _ICML_ , pp. 3821–3830. PMLR, 2021. 

10 

Published as a conference paper at ICLR 2023 

- XY Han, Vardan Papyan, and David L Donoho. Neural collapse under mse loss: Proximity to and dynamics on the central path. In _ICLR_ , 2022. 

- Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun. Deep residual learning for image recognition. In _CVPR_ , pp. 770–778, 2016. 

- Michael Hersche, Geethan Karunaratne, Giovanni Cherubini, Luca Benini, Abu Sebastian, and Abbas Rahimi. Constrained few-shot class-incremental learning. In _CVPR_ , pp. 9057–9067, 2022. 

- Saihui Hou, Xinyu Pan, Chen Change Loy, Zilei Wang, and Dahua Lin. Learning a unified classifier incrementally via rebalancing. In _CVPR_ , pp. 831–839, 2019. 

- Wenlong Ji, Yiping Lu, Yiliang Zhang, Zhun Deng, and Weijie J Su. An unconstrained layer-peeled perspective on neural collapse. In _ICLR_ , 2022. 

- KJ Joseph, Salman Khan, Fahad Shahbaz Khan, Rao Muhammad Anwer, and Vineeth N Balasubramanian. Energy-based latent aligner for incremental learning. In _CVPR_ , pp. 7452–7461, 2022. 

- Alex Krizhevsky, Geoffrey Hinton, et al. Learning multiple layers of features from tiny images. 2009. 

- Yann LeCun, Yoshua Bengio, and Geoffrey Hinton. Deep learning. _Nature_ , 521(7553):436–444, 2015. 

- Zhizhong Li and Derek Hoiem. Learning without forgetting. _IEEE Transactions on Pattern Analysis and Machine Intelligence_ , 40(12):2935–2947, 2017. 

- Bin Liu, Yue Cao, Yutong Lin, Qi Li, Zheng Zhang, Mingsheng Long, and Han Hu. Negative margin matters: Understanding margin in few-shot classification. In _ECCV_ , pp. 438–455, 2020. 

- Huan Liu, Li Gu, Zhixiang Chi, Yang Wang, Yuanhao Yu, Jun Chen, and Jin Tang. Few-shot classincremental learning via entropy-regularized data-free replay. In _ECCV_ , 2022. 

- Bin Lu, Xiaoying Gan, Lina Yang, Weinan Zhang, Luoyi Fu, and Xinbing Wang. Geometer: Graph few-shot class-incremental learning via prototype representation. In _KDD_ , 2022. 

- Jianfeng Lu and Stefan Steinerberger. Neural collapse with cross-entropy loss. _arXiv preprint arXiv:2012.08465_ , 2020. 

- Aleix M Martinez and Avinash C Kak. Pca versus lda. _IEEE Transactions on Pattern Analysis and Machine Intelligence_ , 23(2):228–233, 2001. 

- Dustin G Mixon, Hans Parshall, and Jianzong Pi. Neural collapse with unconstrained features. _arXiv preprint arXiv:2011.11619_ , 2020. 

- Vardan Papyan, XY Han, and David L Donoho. Prevalence of neural collapse during the terminal phase of deep learning training. _Proceedings of the National Academy of Sciences_ , 117(40): 24652–24663, 2020. 

- Can Peng, Kun Zhao, Tianren Wang, Meng Li, and Brian C Lovell. Few-shot class-incremental learning from an open-set perspective. In _ECCV_ , 2022. 

- Federico Pernici, Matteo Bruni, Claudio Baecchi, Francesco Turchini, and Alberto Del Bimbo. Class-incremental learning with pre-allocated fixed classifiers. In _ICPR_ , pp. 6259–6266, 2021. 

- Tomaso Poggio and Qianli Liao. Explicit regularization and implicit bias in deep network classifiers trained with the square loss. _arXiv preprint arXiv:2101.00072_ , 2020. 

- C Radhakrishna Rao. The utilization of multiple measurements in problems of biological classification. _Journal of the Royal Statistical Society. Series B (Methodological)_ , 10(2):159–203, 1948. 

Sachin Ravi and Hugo Larochelle. Optimization as a model for few-shot learning. In _ICLR_ , 2017. 

Sylvestre-Alvise Rebuffi, Alexander Kolesnikov, Georg Sperl, and Christoph H Lampert. icarl: Incremental classifier and representation learning. In _CVPR_ , pp. 2001–2010, 2017. 

11 

Published as a conference paper at ICLR 2023 

Mengye Ren, Renjie Liao, Ethan Fetaya, and Richard Zemel. Incremental few-shot learning with attention attractor networks. In _NeurIPS_ , volume 32, 2019. 

- Olga Russakovsky, Jia Deng, Hao Su, Jonathan Krause, Sanjeev Satheesh, Sean Ma, Zhiheng Huang, Andrej Karpathy, Aditya Khosla, Michael Bernstein, et al. Imagenet large scale visual recognition challenge. _IJCV_ , 115(3):211–252, 2015. 

- Guangyuan Shi, Jiaxin Chen, Wenlong Zhang, Li-Ming Zhan, and Xiao-Ming Wu. Overcoming catastrophic forgetting in incremental few-shot learning by finding flat minima. In _NeurIPS_ , volume 34, pp. 6747–6761, 2021. 

- Jake Snell, Kevin Swersky, and Richard Zemel. Prototypical networks for few-shot learning. In _NeurIPS_ , volume 30, 2017. 

- Flood Sung, Yongxin Yang, Li Zhang, Tao Xiang, Philip HS Torr, and Timothy M Hospedales. Learning to compare: Relation network for few-shot learning. In _CVPR_ , pp. 1199–1208, 2018. 

- Xiaoyu Tao, Xinyuan Chang, Xiaopeng Hong, Xing Wei, and Yihong Gong. Topology-preserving class-incremental learning. In _ECCV_ , pp. 254–270. Springer, 2020a. 

- Xiaoyu Tao, Xiaopeng Hong, Xinyuan Chang, Songlin Dong, Xing Wei, and Yihong Gong. Fewshot class-incremental learning. In _CVPR_ , pp. 12183–12192, 2020b. 

- Tom Tirer and Joan Bruna. Extended unconstrained features model for exploring deep neural collapse. In _ICML_ , 2022. 

- Oriol Vinyals, Charles Blundell, Timothy Lillicrap, Daan Wierstra, et al. Matching networks for one shot learning. _NeurIPS_ , 29, 2016. 

- Catherine Wah, Steve Branson, Peter Welinder, Pietro Perona, and Serge Belongie. The caltech-ucsd birds-200-2011 dataset. 2011. 

- E Weinan and Stephan Wojtowytsch. On the emergence of tetrahedral symmetry in the final and penultimate layers of neural network classifiers. _arXiv preprint arXiv:2012.05420_ , 2020. 

- Liang Xie, Yibo Yang, Deng Cai, and Xiaofei He. Neural collapse inspired attraction-repulsionbalanced loss for imbalanced learning. _Neurocomputing_ , 2023. 

- Boyu Yang, Mingbao Lin, Yunxiao Zhang, Binghao Liu, Xiaodan Liang, Rongrong Ji, and Qixiang Ye. Dynamic support network for few-shot class incremental learning. _IEEE Transactions on Pattern Analysis and Machine Intelligence_ , 2022a. 

- Yibo Yang, Liang Xie, Shixiang Chen, Xiangtai Li, Zhouchen Lin, and Dacheng Tao. Do we really need a learnable classifier at the end of deep neural network? In _NeurIPS_ , 2022b. 

- Hongxu Yin, Pavlo Molchanov, Jose M Alvarez, Zhizhong Li, Arun Mallya, Derek Hoiem, Niraj K Jha, and Jan Kautz. Dreaming to distill: Data-free knowledge transfer via deepinversion. In _CVPR_ , pp. 8715–8724, 2020. 

- Sung Whan Yoon, Do-Yeon Kim, Jun Seo, and Jaekyun Moon. Xtarnet: Learning to extract taskadaptive representation for incremental few-shot learning. In _ICML_ . PMLR, 2020. 

- Chi Zhang, Yujun Cai, Guosheng Lin, and Chunhua Shen. Deepemd: Few-shot image classification with differentiable earth mover’s distance and structured classifiers. In _CVPR_ , pp. 12203–12213, 2020. 

- Chi Zhang, Nan Song, Guosheng Lin, Yun Zheng, Pan Pan, and Yinghui Xu. Few-shot incremental learning with continually evolved classifiers. In _CVPR_ , pp. 12455–12464, 2021. 

- Hanbin Zhao, Yongjian Fu, Mintong Kang, Qi Tian, Fei Wu, and Xi Li. Mgsvf: Multi-grained slow vs. fast framework for few-shot class-incremental learning. _IEEE Transactions on Pattern Analysis and Machine Intelligence_ , 2021. 

12 

Published as a conference paper at ICLR 2023 

Zhisheng Zhong, Jiequan Cui, Yibo Yang, Xiaoyang Wu, Xiaojuan Qi, Xiangyu Zhang, and Jiaya Jia. Understanding imbalanced semantic segmentation through neural collapse. _arXiv preprint arXiv:2301.01100_ , 2023. 

- Da-Wei Zhou, Fu-Yun Wang, Han-Jia Ye, Liang Ma, Shiliang Pu, and De-Chuan Zhan. Forward compatible few-shot class-incremental learning. In _CVPR_ , pp. 9046–9056, 2022a. 

- Da-Wei Zhou, Han-Jia Ye, Liang Ma, Di Xie, Shiliang Pu, and De-Chuan Zhan. Few-shot classincremental learning by sampling multi-phase tasks. _IEEE Transactions on Pattern Analysis and Machine Intelligence_ , 2022b. 

- Jinxin Zhou, Xiao Li, Tianyu Ding, Chong You, Qing Qu, and Zhihui Zhu. On the optimization landscape of neural collapse under mse loss: Global optimality with unconstrained features. _arXiv preprint arXiv:2203.01238_ , 2022c. 

- Kai Zhu, Yang Cao, Wei Zhai, Jie Cheng, and Zheng-Jun Zha. Self-promoted prototype refinement for few-shot class-incremental learning. In _CVPR_ , pp. 6801–6810, 2021a. 

Zhihui Zhu, Tianyu DING, Jinxin Zhou, Xiao Li, Chong You, Jeremias Sulam, and Qing Qu. A geometric analysis of neural collapse with unconstrained features. In _NeurIPS_ , 2021b. 

13 

Published as a conference paper at ICLR 2023 

## A APPENDIX: PROOF OF THEOREM 1 

Our proof is following Yang et al. (2022b). We consider the problem in Eq. (9), 


![](raw/papers/2_LLM_NLP/LLM_paper_list/few-shot/iclr_2023_Neural_Collapse_Inspired_Feature-Classifier_Alignment_for_Few-Shot_Class-Incremental_Learning_images/iclr_2023_Neural_Collapse_Inspired_Feature-Classifier_Alignment_for_Few-Shot_Class-Incremental_Learning.pdf-0014-03.png)


where **m**[(] _k,i[t]_[)] _[∈]_[R] _[d]_[denotes][a][feature][variable][that][belongs][to][the] _[i]_[-th][sample][of][class] _[k]_[in][session] _t_ , _nk_ is number of samples in class _k_ , _K_[(] _[t]_[)] is number of classes in session _t_ , _N_[(] _[t]_[)] is the number of samples in session **W** ˆ ETF _∈_ R _[d][×][K]_ refers to the ETF classifier for the whole label space as introduced in Section 4.1. _t_ , _i.e., N_[(] _[t]_[)] =[�] _[K] k_ =1[(] _[t]_[)] _[n][k]_[, and] **[ M]**[(] _[t]_[)] _[∈]_[R] _[d][×][N]_[(] _[t]_[)][denotes a collection of] **[ m]**[(] _k,i[t]_[)][.] We have _K_ =[�] _[T] t_ =0 _[K]_[(] _[t]_[)][and,] 


![](raw/papers/2_LLM_NLP/LLM_paper_list/few-shot/iclr_2023_Neural_Collapse_Inspired_Feature-Classifier_Alignment_for_Few-Shot_Class-Incremental_Learning_images/iclr_2023_Neural_Collapse_Inspired_Feature-Classifier_Alignment_for_Few-Shot_Class-Incremental_Learning.pdf-0014-05.png)


ˆ ˆ where **w** _k_ and **w** _k′_ are two column vectors in **W**[ˆ] ETF. From the definition of a simplex ETF in Eq. (1), we have **W**[ˆ] ETF _·_ **1** _K_ = **0** _d_ , where **1** _K_ is an all-ones vector in R _[K]_ , and **0** _d_ is an all-zeros vector in R _[d]_ . Then we have, 


![](raw/papers/2_LLM_NLP/LLM_paper_list/few-shot/iclr_2023_Neural_Collapse_Inspired_Feature-Classifier_Alignment_for_Few-Shot_Class-Incremental_Learning_images/iclr_2023_Neural_Collapse_Inspired_Feature-Classifier_Alignment_for_Few-Shot_Class-Incremental_Learning.pdf-0014-07.png)


When _L_ is the dot-regression (DR) loss in Eq. (4), it is easy to identify that _L ≥_ 0 and the equality holds if and only if **w** ˆ _k[T]_ **[m]**[(] _k,i[t]_[)][=][1][,] _[∀]_[0] _[≤][t][≤][T,]_[ 1] _[≤][k][≤][K,]_[ 1] _[≤][i][≤][n][k]_[.][Since] _[∥]_ **[w]**[ˆ] _[k][∥]_[=][1] and _∥_ **m**[(] _k,i[t]_[)] _[∥]_[2] _[≤]_[1][,][we][have] **[w]**[ˆ] _k[T]_ **[m]**[(] _k,i[t]_[)] _[≤]_[1][.][The][equality][holds][if][and][only][if] _[∥]_ **[m]**[(] _k,i[t]_[)] _[∥]_[2][=][1][and] cos ∠( ˆ **w** _k,_ **m**[(] _k,i[t]_[)][) = 1][.][Denote] **[M]**[ˆ][ = [ ˆ] **[M]**[(0)] _[,][ · · ·][,]_ **[M]**[ˆ][(] _[T]_[ )][]] _[ ∈]_[R] _[d][×]_[�] _t[T]_ =0 _[N]_[ (] _[t]_[)] as the global optimality of Eq. (9) for all sessions 0 _≤ t ≤ T_ . For any column vector **m** ˆ _k,i_ in **M**[ˆ] , we have, 


![](raw/papers/2_LLM_NLP/LLM_paper_list/few-shot/iclr_2023_Neural_Collapse_Inspired_Feature-Classifier_Alignment_for_Few-Shot_Class-Incremental_Learning_images/iclr_2023_Neural_Collapse_Inspired_Feature-Classifier_Alignment_for_Few-Shot_Class-Incremental_Learning.pdf-0014-09.png)


which concludes the proof for DR loss. 

When _L_ is the cross-entropy (CE) loss, _i.e.,_ 

where 0 _≤ t ≤ T,_ 1 _≤ k ≤ K_[(] _[t]_[)] , and 1 _≤ i ≤ nk_ . Since the problem is separable among _T_ + 1 sessions, we only analyze the _t_ -th session and omit the superscript ( _t_ ) for simplicity. The objective in Eq. (13) is the sum of an affine function and log-sum-exp functions. When **W**[ˆ] ETF is fixed, the loss is convex _w.r.t_ **m** _k,i_ with convex constraints. So, we can use the KKT condition for its global optimality. Based on Eq. (9) and Eq. (13), we have the Lagrange function, 


![](raw/papers/2_LLM_NLP/LLM_paper_list/few-shot/iclr_2023_Neural_Collapse_Inspired_Feature-Classifier_Alignment_for_Few-Shot_Class-Incremental_Learning_images/iclr_2023_Neural_Collapse_Inspired_Feature-Classifier_Alignment_for_Few-Shot_Class-Incremental_Learning.pdf-0014-13.png)


where _λk,i_ is the Lagrange multiplier. The gradient with respect to **m** _k,i_ takes the form of: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/few-shot/iclr_2023_Neural_Collapse_Inspired_Feature-Classifier_Alignment_for_Few-Shot_Class-Incremental_Learning_images/iclr_2023_Neural_Collapse_Inspired_Feature-Classifier_Alignment_for_Few-Shot_Class-Incremental_Learning.pdf-0014-15.png)


where 1 _≤ i ≤ nk_ , 1 _≤ k ≤ K_[(] _[t]_[)] , and _pj_ is the softmax probability of **m** _k,i_ for the _j_ -th class, _i.e.,_ 


![](raw/papers/2_LLM_NLP/LLM_paper_list/few-shot/iclr_2023_Neural_Collapse_Inspired_Feature-Classifier_Alignment_for_Few-Shot_Class-Incremental_Learning_images/iclr_2023_Neural_Collapse_Inspired_Feature-Classifier_Alignment_for_Few-Shot_Class-Incremental_Learning.pdf-0014-17.png)


14 

Published as a conference paper at ICLR 2023 

Since _∥_ **w** ˆ _k∥_ = 1 and _∥_ **m** _k,i∥≤_ 1, we have 0 _< pk <_ 1 _, ∀_ 1 _≤ k ≤ K_ . 

We now solve the equation _∂_ **m** _∂L_[˜] _k,i_[= 0][.][Assume that] _[ λ][k,i]_[= 0][, and then we have,] 


![](raw/papers/2_LLM_NLP/LLM_paper_list/few-shot/iclr_2023_Neural_Collapse_Inspired_Feature-Classifier_Alignment_for_Few-Shot_Class-Incremental_Learning_images/iclr_2023_Neural_Collapse_Inspired_Feature-Classifier_Alignment_for_Few-Shot_Class-Incremental_Learning.pdf-0015-03.png)


Since 1 _− pk_ =[�] _[K] j_ = _k[p][j]_[and Eq.][(11), multiplying] **[w]**[ˆ] _[k]_[by both sides of Eq.][(17), we have,] 


![](raw/papers/2_LLM_NLP/LLM_paper_list/few-shot/iclr_2023_Neural_Collapse_Inspired_Feature-Classifier_Alignment_for_Few-Shot_Class-Incremental_Learning_images/iclr_2023_Neural_Collapse_Inspired_Feature-Classifier_Alignment_for_Few-Shot_Class-Incremental_Learning.pdf-0015-05.png)


whichcondition, the global optimalitycontradicts with 0 _< pk_ **m** ˆ _<k,i_ 1.satisfies thatThen we have the other case _λk,i >_ 0. Based on the KKT 


![](raw/papers/2_LLM_NLP/LLM_paper_list/few-shot/iclr_2023_Neural_Collapse_Inspired_Feature-Classifier_Alignment_for_Few-Shot_Class-Incremental_Learning_images/iclr_2023_Neural_Collapse_Inspired_Feature-Classifier_Alignment_for_Few-Shot_Class-Incremental_Learning.pdf-0015-07.png)


_∂L_[˜] The equation _∂_ **m** ˆ _k,i_[= 0][ leads to:] 


![](raw/papers/2_LLM_NLP/LLM_paper_list/few-shot/iclr_2023_Neural_Collapse_Inspired_Feature-Classifier_Alignment_for_Few-Shot_Class-Incremental_Learning_images/iclr_2023_Neural_Collapse_Inspired_Feature-Classifier_Alignment_for_Few-Shot_Class-Incremental_Learning.pdf-0015-09.png)


ˆ Based on Eq. (11), for any _j[′]_ = _k_ , multiplying **w** _j[′]_ by both sides of Eq. (20), we have, 


![](raw/papers/2_LLM_NLP/LLM_paper_list/few-shot/iclr_2023_Neural_Collapse_Inspired_Feature-Classifier_Alignment_for_Few-Shot_Class-Incremental_Learning_images/iclr_2023_Neural_Collapse_Inspired_Feature-Classifier_Alignment_for_Few-Shot_Class-Incremental_Learning.pdf-0015-11.png)


Since _∀k ∈_ [1 _, K_ ], _pk >_ 0, we have **m** ˆ _[T] k,i_ **[w]**[ˆ] _[j][′][<]_[ 0][.][Then for any] _[ j]_[1] _[, j]_[2][=] _[ k]_[,] 


![](raw/papers/2_LLM_NLP/LLM_paper_list/few-shot/iclr_2023_Neural_Collapse_Inspired_Feature-Classifier_Alignment_for_Few-Shot_Class-Incremental_Learning_images/iclr_2023_Neural_Collapse_Inspired_Feature-Classifier_Alignment_for_Few-Shot_Class-Incremental_Learning.pdf-0015-13.png)


The function _f_ ( _x_ ) = exp( _x_ ) _/x_ is monotonically increasing when _x <_ 1. So, Eq. (22) indicates that 


![](raw/papers/2_LLM_NLP/LLM_paper_list/few-shot/iclr_2023_Neural_Collapse_Inspired_Feature-Classifier_Alignment_for_Few-Shot_Class-Incremental_Learning_images/iclr_2023_Neural_Collapse_Inspired_Feature-Classifier_Alignment_for_Few-Shot_Class-Incremental_Learning.pdf-0015-15.png)


and 


![](raw/papers/2_LLM_NLP/LLM_paper_list/few-shot/iclr_2023_Neural_Collapse_Inspired_Feature-Classifier_Alignment_for_Few-Shot_Class-Incremental_Learning_images/iclr_2023_Neural_Collapse_Inspired_Feature-Classifier_Alignment_for_Few-Shot_Class-Incremental_Learning.pdf-0015-17.png)


Multiplying **w** ˆ _k_ by both sides of Eq. (20), we have, 


![](raw/papers/2_LLM_NLP/LLM_paper_list/few-shot/iclr_2023_Neural_Collapse_Inspired_Feature-Classifier_Alignment_for_Few-Shot_Class-Incremental_Learning_images/iclr_2023_Neural_Collapse_Inspired_Feature-Classifier_Alignment_for_Few-Shot_Class-Incremental_Learning.pdf-0015-19.png)


Combing Eq. (24) and Eq. (25), we have, 


![](raw/papers/2_LLM_NLP/LLM_paper_list/few-shot/iclr_2023_Neural_Collapse_Inspired_Feature-Classifier_Alignment_for_Few-Shot_Class-Incremental_Learning_images/iclr_2023_Neural_Collapse_Inspired_Feature-Classifier_Alignment_for_Few-Shot_Class-Incremental_Learning.pdf-0015-21.png)


Based on _pj_ =[1] _K[−] −[p]_ 1 _[k][,][ ∀][j]_[=] _[ k]_[, and Eq.][(12), we can rewrite Eq.][(20) as:] 


![](raw/papers/2_LLM_NLP/LLM_paper_list/few-shot/iclr_2023_Neural_Collapse_Inspired_Feature-Classifier_Alignment_for_Few-Shot_Class-Incremental_Learning_images/iclr_2023_Neural_Collapse_Inspired_Feature-Classifier_Alignment_for_Few-Shot_Class-Incremental_Learning.pdf-0015-23.png)


ˆ ˆ ˆ ˆ _∥_ which **m** ˆ _k,i∥_ means= 1 (Eq.that(19)), we have, **m** _k,i_ is aligned with **w** _k_ , _i.e.,_ cos ∠( ˆ **m** _k,i,_ **w** _k_ ) = 1. Given that _∥_ **w** _k∥_ = 1 and 

**m** ˆ _[T] k,i_ **[w]**[ˆ] _[k]_[= 1] _[,]_ 

and Eq. (26) leads to: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/few-shot/iclr_2023_Neural_Collapse_Inspired_Feature-Classifier_Alignment_for_Few-Shot_Class-Incremental_Learning_images/iclr_2023_Neural_Collapse_Inspired_Feature-Classifier_Alignment_for_Few-Shot_Class-Incremental_Learning.pdf-0015-27.png)


Therefore, for any column vector **m** ˆ _k,i_ in **M**[ˆ] , we have, 


![](raw/papers/2_LLM_NLP/LLM_paper_list/few-shot/iclr_2023_Neural_Collapse_Inspired_Feature-Classifier_Alignment_for_Few-Shot_Class-Incremental_Learning_images/iclr_2023_Neural_Collapse_Inspired_Feature-Classifier_Alignment_for_Few-Shot_Class-Incremental_Learning.pdf-0015-29.png)


which concludes the proof for CE loss. 

15 

Published as a conference paper at ICLR 2023 

Table 4: Performance of FSCIL in each session on CUB-200 and comparison with other studies. The top rows list class-incremental learning and few-shot learning results implemented by Tao et al. (2020b); Zhang et al. (2021); Liu et al. (2022); Zhou et al. (2022a) in the FSCIL setting. “Average Acc.” is the average accuracy of all sessions. “Final Improv.” calculates the improvement of our method in the last session. 

|**Methods**|**Accuracy in each session (%)**_↑_<br>**Average**<br>**Final**<br>**0**<br>**1**<br>**2**<br>**3**<br>**4**<br>**5**<br>**6**<br>**7**<br>**8**<br>**9**<br>**10**<br>**Acc.**<br>**Improv.**|
|---|---|
|iCaRL (Rebuff et al., 2017)<br>EEIL (Castro et al., 2018)<br>NCM (Hou et al., 2019)<br>Fixed classifer (Pernici et al., 2021)<br>D-NegCosine (Liu et al., 2020)<br>D-DeepEMD (Zhang et al., 2020)<br>D-Cosine (Vinyals et al., 2016)<br>DeepInv (Yin et al., 2020)|68.68<br>52.65<br>48.61<br>44.16<br>36.62<br>29.52<br>27.83<br>26.26<br>24.01<br>23.89<br>21.16<br>36.67<br>**+38.28**<br>68.68<br>53.63<br>47.91<br>44.20<br>36.30<br>27.46<br>25.93<br>24.70<br>23.95<br>24.13<br>22.11<br>36.27<br>**+37.33**<br>68.68<br>57.12<br>44.21<br>28.78<br>26.71<br>25.66<br>24.62<br>21.52<br>20.12<br>20.06<br>19.87<br>32.49<br>**+39.57**<br>68.47<br>51.00<br>45.42<br>40.76<br>35.90<br>33.18<br>27.23<br>24.24<br>21.18<br>17.34<br>16.20<br>34.63<br>**+43.24**<br>74.96<br>70.57<br>66.62<br>61.32<br>60.09<br>56.06<br>55.03<br>52.78<br>51.50<br>50.08<br>48.47<br>58.86<br>**+10.97**<br>75.35<br>70.69<br>66.68<br>62.34<br>59.76<br>56.54<br>54.61<br>52.52<br>50.73<br>49.20<br>47.60<br>58.73<br>**+11.84**<br>75.52<br>70.95<br>66.46<br>61.20<br>60.86<br>56.88<br>55.40<br>53.49<br>51.94<br>50.93<br>49.31<br>59.36<br>**+10.13**<br>75.90<br>70.21<br>65.36<br>60.14<br>58.79<br>55.88<br>53.21<br>51.27<br>49.38<br>47.11<br>45.67<br>57.54<br>**+13.77**|
|TOPIC (Tao et al., 2020b)<br>IDLVQ (Chen & Lee, 2021)<br>SPPR (Zhu et al., 2021a)<br>Cheraghian et al. (2021b)<br>CEC (Zhang et al., 2021)<br>LIMIT (Zhou et al., 2022b)<br>MgSvF (Zhao et al., 2021)<br>MetaFSCIL (Chi et al., 2022)<br>FACT (Zhou et al., 2022a)<br>Data-free replay (Liu et al., 2022)<br>ALICE (Peng et al., 2022)|68.68<br>62.49<br>54.81<br>49.99<br>45.25<br>41.40<br>38.35<br>35.36<br>32.22<br>28.31<br>26.28<br>43.92<br>**+33.16**<br>77.37<br>74.72<br>70.28<br>67.13<br>65.34<br>63.52<br>62.10<br>61.54<br>59.04<br>58.68<br>57.81<br>65.23<br>**+1.63**<br>68.68<br>61.85<br>57.43<br>52.68<br>50.19<br>46.88<br>44.65<br>43.07<br>40.17<br>39.63<br>37.33<br>49.32<br>**+22.11**<br>68.78<br>59.37<br>59.32<br>54.96<br>52.58<br>49.81<br>48.09<br>46.32<br>44.33<br>43.43<br>43.23<br>51.84<br>**+16.21**<br>75.85<br>71.94<br>68.50<br>63.50<br>62.43<br>58.27<br>57.73<br>55.81<br>54.83<br>53.52<br>52.28<br>61.33<br>**+7.16**<br>76.32<br>74.18<br>72.68<br>69.19<br>**68.79**<br>**65.64**<br>63.57<br>62.69<br>**61.47**<br>60.44<br>58.45<br>66.67<br>**+0.99**<br>72.29<br>70.53<br>67.00<br>64.92<br>62.67<br>61.89<br>59.63<br>59.15<br>57.73<br>55.92<br>54.33<br>62.37<br>**+5.11**<br>75.9<br>72.41<br>68.78<br>64.78<br>62.96<br>59.99<br>58.3<br>56.85<br>54.78<br>53.82<br>52.64<br>61.93<br>**+6.8**<br>75.90<br>73.23<br>70.84<br>66.13<br>65.56<br>62.15<br>61.74<br>59.83<br>58.41<br>57.89<br>56.94<br>64.42<br>**+2.5**<br>75.90<br>72.14<br>68.64<br>63.76<br>62.58<br>59.11<br>57.82<br>55.89<br>54.92<br>53.58<br>52.39<br>61.52<br>**+7.05**<br>77.40<br>72.70<br>70.60<br>67.20<br>65.90<br>63.40<br>62.90<br>61.90<br>60.50<br>**60.60**<br>**60.10**<br>65.75<br>-0.66|
|**NC-FSCIL (ours)**|**80.45**<br>**75.98**<br>**72.30**<br>**70.28**<br>68.17<br>65.16<br>**64.43**<br>**63.25**<br>60.66<br>60.01<br>59.44<br>**67.28**|



Table 5: A comparison of backbone networks used in different studies. 

|Methods|miniImageNet|CIFAR-100|CUB-200|
|---|---|---|---|
|TOPIC (Tao et al., 2020b)|ResNet-18|ResNet-18|ResNet-18|
|CEC (Zhang et al., 2021)|ResNet-18|ResNet-20|ResNet-18|
|CFSCIL (Hersche et al., 2022)|ResNet-12|ResNet-12|-|
|LIMIT (Zhou et al., 2022b)|ResNet-18|ResNet-20|ResNet-18|
|ALICE (Peng et al., 2022)|ResNet-18|ResNet-18|ResNet-18|
|NC-FSCIL (ours)|ResNet-12|ResNet-12|ResNet-18|



## B APPENDIX: IMPLEMENTATION DETAILS 

**Datasets.** We conduct our experiments on three FSCIL benchmark datasets including miniImageNet (Russakovsky et al., 2015), CIFAR-100 (Krizhevsky et al., 2009), and CUB-200 (Wah et al., 2011). miniImageNet is a variant of ImageNet with an image size of 84 _×_ 84. It has 100 classes with each class containing 500 images for training and 100 images for testing. CIFAR-100 has the same number of classes and images, and the image size is 32 _×_ 32. CUB-200 is a dataset for fine-grained image classification containing 11,788 images of 200 classes in a resolution of 224 _×_ 224. There are 5,994 images for training and 5,794 images for testing. We follow the standard experimental settings in FSCIL (Tao et al., 2020b; Zhang et al., 2021). For both miniImageNet and CIFAR-100, the base session ( _t_ = 0) contains 60 classes, and a 5-way 5-shot (5 classes and 5 images per class) problem is adopted for each of the 8 incremental sessions (1 _≤ t ≤_ 8). For CUB-200, 100 classes are used in the base session, and there are 10 incremental sessions, each of which is 10-way 5-shot. 

**Architectures.** Prior studies widely adopt ResNet-12, ResNet-18, and ResNet-20 (He et al., 2016) for FSCIL experiments. As shown in Table 5, we compare the backbone networks used in different studies. For miniImageNet and CIFAR-100, we use ResNet-12 following Hersche et al. (2022). For CUB-200, we use ResNet-18 (pre-trained on ImageNet) following other studies. We adopt a two-layer MLP block as the projection layer following the practice in Peng et al. (2022). 

16 

Published as a conference paper at ICLR 2023 

**Training Details.** We adopt the standard data pre-processing and augmentation schemes including random resizing, random flipping, and color jittering (Tao et al., 2020b; Zhang et al., 2021; Peng et al., 2022). We train all models with a batchsize of 512 in the base session, and a batchsize of 64 (containing new session data and intermediate features in the memory) in each incremental session. On miniImageNet, we train for 500 epochs in the base session, and 100-170 iterations in each incremental session. The initial learning rate is 0.25 for base session, and 0.025 for incremental sessions. On CIFAR-100, we train for 200 epochs in the base session, and 50-200 iterations in each incremental session. The initial learning rate is 0.25 for both base and incremental sessions. On CUB-200, we train for 80 epochs in the base session, and 105-150 iterations in each incremental session. The initial learning rates are 0.025 and 0.05 for base session and incremental sessions, respectively. In all experiments, we adopt a cosine annealing strategy for learning rate, and use SGD with momentum as optimizer. Our code will be publicly available in the final version. 

## C APPENDIX: MORE RESULTS 

Our experimental result on CUB-200 is shown in Table 4. We achieve a better accuracy in the last session than most of the baseline methods. Although we do not surpass ALICE in the last session on CUB-200, we still have the best average accuracy among all methods. 

We also visualize the average cosine similarities between feature and classifier of the same class, _i.e.,_ Avg _k{_ cos ∠( **m** _k −_ **m** _G,_ **w** _k_ ) _}_ and the trace ratio of within-class covariance to between-class covariance, tr(Σ _W_ ) _/_ tr(Σ _B_ ). 

A higher average cos ∠( **m** _k −_ **m** _G,_ **w** _k_ ) indicates that feature centers are more closely aligned with their corresponding classifier prototypes of the same class. As shown in Figure 4, the values of our method are consistently higher than those of the baseline method. Figure 4a and Figure 4d reveal that our method has a better feature-classifier alignment in each session of the incremental training on both train and test sets. When we measure on all the encountered classes by each session in Figure 4b and Figure 4e, the metric for our method does not change obviously after the 4-th session on train set, while the metric for the baseline method keeps decreasing as training incrementally. Especially for the base session classes, our method is able to keep the metric stable on both train and test sets after the decline of the first 3-4 sessions, as shown in Figure 4c and Figure 4f. As a comparison, the baseline method cannot mitigate the deterioration. Given that the base session has the most classes, the performance on base session classes largely decides the final accuracy in the last session for FSCIL. Therefore, the superiority of our method can be attributed to our ability of keeping the feature-classifier alignment well for base session classes. 

The within-class covariance Σ _W_ and the between-class covariance Σ _B_ are defined as: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/few-shot/iclr_2023_Neural_Collapse_Inspired_Feature-Classifier_Alignment_for_Few-Shot_Class-Incremental_Learning_images/iclr_2023_Neural_Collapse_Inspired_Feature-Classifier_Alignment_for_Few-Shot_Class-Incremental_Learning.pdf-0017-07.png)


## and 


![](raw/papers/2_LLM_NLP/LLM_paper_list/few-shot/iclr_2023_Neural_Collapse_Inspired_Feature-Classifier_Alignment_for_Few-Shot_Class-Incremental_Learning_images/iclr_2023_Neural_Collapse_Inspired_Feature-Classifier_Alignment_for_Few-Shot_Class-Incremental_Learning.pdf-0017-09.png)


where **m** _k,i_ is the feature of sample _i_ in class _k_ , **m** _k_ is the within-class mean of class _k_ features, and **m** _G_ denotes the global mean of all features. A lower within-class variation with a higher betweenclass variation corresponds to a better Fisher Discriminant Ratio. As shown in Figure 5, we compare the trace ratio of within-class covariance to between-class covariance between our method and the baseline method. We observe similar patterns to Figure 4. Concretely, the trace ratio metric of our method is consistently lower that of baseline. For the base session classes, the metric of our method increases more mildly, which corroborates our ability of maintaining the performance on the old classes, and is in line with the indications from Figure 3 and Figure 4. 

17 

Published as a conference paper at ICLR 2023 


![](raw/papers/2_LLM_NLP/LLM_paper_list/few-shot/iclr_2023_Neural_Collapse_Inspired_Feature-Classifier_Alignment_for_Few-Shot_Class-Incremental_Learning_images/iclr_2023_Neural_Collapse_Inspired_Feature-Classifier_Alignment_for_Few-Shot_Class-Incremental_Learning.pdf-0018-01.png)


**----- Start of picture text -----**<br>
1.00 Learnable+CE 1.000 Learnable+CE 1.000 Learnable+CE<br>0.95 ETF+DR (Ours) 0.975 ETF+DR (Ours) 0.975 ETF+DR (Ours)<br>0.90 0.950 0.950<br>0.925<br>0.85 0.925<br>0.900<br>0.80 0.900 0.875<br>0.75 0.875 0.850<br>0.70 0.850 0.825<br>0.65 0.825 0.800<br>0 1 2 3 4 5 6 7 8 0 1 2 3 4 5 6 7 8 0 1 2 3 4 5 6 7 8<br>Sessions Sessions Sessions<br>(a) train (each) (b) train (accumulate) (c) train (base)<br>1.0 Learnable+CE 1.00 Learnable+CE 1e 1 Learnable+CE<br>ETF+DR (Ours) ETF+DR (Ours) ETF+DR (Ours)<br>0.9 0.95 9.5<br>0.8 0.90 9.0<br>0.7 0.85<br>8.5<br>0.6 0.80<br>8.0<br>0.5 0.75<br>0 1 2 3 Sessions4 5 6 7 8 0 1 2 3 Sessions4 5 6 7 8 0 1 2 3 Sessions4 5 6 7 8<br>(d) test (each) (e) test (accumulate) (f) test (base)<br>**----- End of picture text -----**<br>


Figure 4: Average cosine similarity between features and classifier prototypes of the same class, _i.e.,_ Avg _k{_ cos ∠( **m** _k −_ **m** _G,_ **w** _k_ ) _}_ , where **m** _k_ is the within-class mean of class _k_ features, **m** _G_ denotes the global mean, and **w** _k_ is the classifier prototype of class _k_ . Statistics are performed among classes in each session (a and d), all encountered classes by the current session (b and e), and only the base session classes (c and f), on train set (a, b, c) and test set (d, e, f), for models trained after each session on miniImageNet. 


![](raw/papers/2_LLM_NLP/LLM_paper_list/few-shot/iclr_2023_Neural_Collapse_Inspired_Feature-Classifier_Alignment_for_Few-Shot_Class-Incremental_Learning_images/iclr_2023_Neural_Collapse_Inspired_Feature-Classifier_Alignment_for_Few-Shot_Class-Incremental_Learning.pdf-0018-03.png)


**----- Start of picture text -----**<br>
1e 1 1e 1 1e 1<br>7 8 Learnable+CE<br>6 6 7 ETF+DR (Ours)<br>5 5 6<br>4 4 5<br>4<br>3 3<br>3<br>2 2 2<br>1 Learnable+CE 1 Learnable+CE 1<br>0 ETF+DR (Ours) 0 ETF+DR (Ours) 0<br>0 1 2 3 4 5 6 7 8 0 1 2 3 4 5 6 7 8 0 1 2 3 4 5 6 7 8<br>Sessions Sessions Sessions<br>(a) train (each) (b) train (accumulate) (c) train (base)<br>3.5 1.4 Learnable+CE Learnable+CE<br>3.0 ETF+DR (Ours) 1.0 ETF+DR (Ours)<br>1.2 0.9<br>2.5<br>1.0 0.8<br>2.0<br>0.7<br>0.8<br>1.5 0.6<br>1.0 0.6 0.5<br>Learnable+CE<br>0.5 ETF+DR (Ours) 0.4 0.4<br>0 1 2 3 4 5 6 7 8 0 1 2 3 4 5 6 7 8 0 1 2 3 4 5 6 7 8<br>Sessions Sessions Sessions<br>(d) test (each) (e) test (accumulate) (f) test (base)<br>**----- End of picture text -----**<br>


Figure 5: Trace ratio of within-class covariance to between-class covariance, _i.e.,_ tr(Σ _W_ ) _/_ tr(Σ _B_ ), where Σ _W_ is the within-class covariance, and Σ _B_ denotes the between-class covariance. Statistics are performed among classes in each session (a and d), all encountered classes by the current session (b and e), and only the base session classes (c and f), on train set (a, b, c) and test set (d, e, f), for models trained after each session on miniImageNet. 

18 

