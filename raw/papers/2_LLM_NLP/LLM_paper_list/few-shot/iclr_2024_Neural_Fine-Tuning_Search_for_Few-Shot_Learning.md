# **Neural Fine-Tuning Search for Few-Shot Learning** 

Panagiotis Eustratiadis[1] 

Łukasz Dudziak[[2]] 

> [[2]] Da Li[1,2] Timothy Hospedales[1,2] 2Samsung AI Center, Cambridge 

1University of Edinburgh 

## **Abstract** 

_In few-shot recognition, a classifier that has been trained on one set of classes is required to rapidly adapt and generalize to a disjoint, novel set of classes. To that end, recent studies have shown the efficacy of fine-tuning with carefully crafted adaptation architectures. However this raises the question of: How can one design the optimal adaptation strategy? In this paper, we study this question through the lens of neural architecture search (NAS). Given a pretrained neural network, our algorithm discovers the optimal arrangement of adapters, which layers to keep frozen and which to fine-tune. We demonstrate the generality of our NAS method by applying it to both residual networks and vision transformers and report state-of-the-art performance on Meta-Dataset and Meta-Album._ 

## **1. Introduction** 

Few-shot recognition [23, 33, 51] aims to learn novel concepts from few examples, often by rapid adaptation of a model trained on a disjoint set of labels. Many solutions adopt a meta-learning perspective [16, 25, 38, 41, 43], or train a powerful feature extractor on the source classes [44, 50] – both of which assume that the training and testing classes are drawn from the same underlying distribution e.g., handwritten characters [24], or ImageNet categories [48]. Later work considers a more realistic and challenging problem variant where a classifier should perform few-shot adaptation not only across visual categories, but also across diverse visual domains [46, 47]. In this crossdomain problem variant, customising the feature extractor to the novel domains is important, and several studies address this through dynamic feature extractors [2, 40] or ensembles of features [12, 28, 32]. Another group of studies employ simple yet effective fine-tuning strategies for adaptation [10, 21, 29, 53] that are predominantly heuristically motivated. Thus, an important question that arises from previous work is: How can one design the _optimal_ adaptation strategy? In this paper, we take a step towards answering this question. 


![](raw/papers/2_LLM_NLP/LLM_paper_list/few-shot/iclr_2024_Neural_Fine-Tuning_Search_for_Few-Shot_Learning_images/iclr_2024_Neural_Fine-Tuning_Search_for_Few-Shot_Learning.pdf-0001-10.png)


Figure 1: Our proposed supernet architecture for few-shot adaptation. The supernet contains all combinations of pretrained, fine-tuned and adapter parameters. _f_ denotes the feature extractor, which is composed of many layers, _g_ , which are the minimal unit for adaptation in our search space. The dotted lines represent possible paths that can be sampled during SPOS training. Every adaptable layer _gi[ϕ,ϕ][′][,α]_ has its own pre-trained parameters ( _ϕi ⊂ θ_ ), finetuned parameters ( _ϕ[′] i_[), and adapter parameters (] _[α][i]_[).] 

Fine-tuning approaches to few-shot adaptation must manage a trade-off between adapting a large or small number of parameters. The former allows for better adaptation, but risks overfitting on a few-shot training set. The latter reduces the risk of overfitting, but limits the capacity for adaptation to novel categories and domains. The recent PMF [21] manages this trade-off through careful tuning of learning rates while fine-tuning the entire feature extractor. TSA [29] and ETT [53] manage it by freezing the feature extractor weights, and inserting some parameter-efficient 

adaptation modules, lightweight enough to be trained in a few-shot manner. FLUTE [45] manages it through selective fine-tuning of a tiny set of FILM [36] parameters, while keeping most of them fixed. Despite this progress, the best way to manage the adaptation/generalisation tradeoff in fine-tuning approaches to few-shot learning (FSL) is still an open question. For example, which layers should be fine-tuned? What kind of adapters should be inserted, and where? While PMF, TSA, ETT, FLUTE, and others provide some intuitive recommendations, we propose a more systematic approach to answer these questions. 

In this paper, we advance the adaptation-based paradigm for FSL by developing a neural architecture search (NAS) algorithm to find the optimal adaptation architecture. Given an initial pre-trained feature extractor, our NAS determines the subset of the architecture that should be fine-tuned, as well as the subset of layers where adaptation modules should be inserted. We draw inspiration from recent work in NAS [4, 7, 8, 18, 55] that proposes revised versions of the stochastic Single-Path One-Shot (SPOS) [18] weight-sharing strategy. Specifically, given a pre-trained ResNet [19] or Vision Transformer (ViT) [11], we consider a search space defined by the inclusion or non-inclusion of task-specific adapters per layer, and the freezing or finetuning of learnable parameters per layer. Based on this search space, we construct a supernet [3] that we train by sampling a random path in each forward pass [18]. Our supernet architecture is illustrated schematically in Figure 1, where the aforementioned decisions are drawn as decision nodes ( _⋄_ ), and possible paths are marked in dotted lines. 

While the supernet training remains somewhat similar to the standard NAS approaches, the subsequent search poses new challenges due to the inherent characteristics of the FSL setting. Specifically, as cross-domain FSL considers a number of datasets including novel domains at test time, it becomes questionable whether searching for a single model – which is the prevalent paradigm in NAS [5, 27, 31, 49] – is the best choice. On the other hand, per-episode architecture selection is too slow and might overfit to the small support set. 

Motivated by these challenges, we propose a novel NAS algorithm that shortlists a small number of architecturally diverse configurations at training time, but defers the final selection until the dataset and episode is known at test time. We empirically show that this is not only computationally efficient, but also improves results noticeably, especially when only a limited amount of domains is available at training time. We term our method Neural Fine-Tuning Search (NFTS). 

NFTS defines a generic search space that is relevant to both major architecture families (i.e., convolutional networks and transformers), and the choice of which specific adapter modules to consider is a hyperparameter, rather than 

a hard constraint. In this paper, we consider using adapter modules that are currently state-of-the-art for ResNets and ViTs (TSA and ETT, respectively), but more adaptation architectures can be added to the search space. 

Our contributions are summarised as follows: (i) We provide the first systematic Auto-ML approach to finding the optimal adaptation strategy that trades-off adaptation flexibility and overfitting risk in multi-domain FSL. (ii) Our novel NFTS algorithm automatically determines which layers should be frozen or adapted, and where new adaptation parameters should be inserted for best few-shot adaptation. (iii) We advance the state-of-the-art in the well-established and challenging Meta-Dataset [46], and the more recent and diverse Meta-Album [47] benchmarks. 

## **2. Related Work** 

## **2.1. Adaptation for Few-shot Learning** 

**Gradient-Based Adaptation** Parameter-efficient adaptation modules have been previously applied for multidomain learning, and transfer learning. A seminal example of this are Residual Adapters [39], which are lightweight 1x1 convolutional filters added to ResNet blocks. They were initially proposed for multi-domain learning, but are also useful for FSL, by providing the ability to update the feature extractor while being lightweight enough to avoid severe overfitting in the few-shot regime. Task-Specific Adapters (TSA) [29] use such adapters together with a URL [28] pre-trained backbone to achieve state of the art results for CNNs on the Meta-Dataset benchmark [46]. Meanwhile, prompt [22] and prefix [30] tuning are established examples of parameter-efficient adaptation for transformer architectures for similar reasons. In FSL, Efficient Transformer Tuning (ETT) [53] apply a similar strategy to fewshot ViT adaptation using a DINO [6] pre-trained backbone. 

PMF [21], FLUTE [45] and FT [10] focus on adaptation of existing parameters without inserting new ones. To manage the adaptation/overfitting trade-off in the few-shot regime, PMF fine-tunes the whole ResNet or ViT backbone, but with carefully-managed learning rates. Meanwhile, FLUTE hand-picks a set of FILM parameters with a modified ResNet backbone for few-shot fine-tuning, while keeping the majority of the feature extractor frozen. 

All of the methods above make heuristic choices about where to place adapters within the backbone, or for which parameters to allow/disallow fine-tuning. However, as different input layers represent different features [7, 54], there is scope for making better decisions about which features to update. Furthermore, in the multi-domain setting different target datasets may benefit from different choices about which modules to update. This paper takes an Auto-ML NAS-based approach to systematically address this issue. **Feed-Forward Adaptation** The aforementioned meth- 

ods all use stochastic gradient descent to update the features during adaptation. We briefly mention CNAPS [40] and derivatives [2] as a competing line of work that use feed-forward networks to modulate the feature extraction process. However, these dynamic feature extractors are less able to generalise to completely novel domains than gradient-based methods [17], as the adaptation module itself suffers from an out of distribution problem. 

## **2.2. Neural architecture search** 

Neural Architecture Search (NAS) is a large and wellstudied topic [14] which we do not attempt to review in detail here. Mainstream NAS aims to discover new architectures that achieve high performance when training on a single dataset from scratch in a many-shot regime. To this end, research aims to develop faster search algorithms [1, 18, 31, 52], and more effective search spaces [9, 15, 37, 56]. We build upon the popular SPOS [18] family of search strategies that encapsulate the entire search space inside a supernet that is trained by sampling paths randomly, and a search algorithm then determines the optimal path. 

We develop an instantiation of the SPOS strategy for the multi-domain FSL problem. We construct a search space suited for parameter-efficient adaptation of a prior architecture to a new set of categories, and extend SPOS to learn on a suite of datasets, and efficiently generalise to novel datasets. This is different than the traditional SPOS paradigm of training and evaluating on the same dataset and same set of categories. 

While there exist some recent NAS works that try to address a similar “train once, search many times” problem efficiently [4, 26, 34, 35], naively using these approaches has two serious shortcomings: i) They assume that after the initial supernet training, subsequent searches do not involve any training (e.g., a search is only performed to consider a different FLOPs constraint while accuracy of different configurations is assumed to stay the same) and thus can be done efficiently – this is not true in the FSL setting as explained earlier. ii) Even if naively searching for each dataset at test time were computationally feasible, the fewshot nature of our setting poses a significant risk of overfitting the architecture to the small support set considered in each episode. 

## **3. Neural Fine-Tuning Search** 

## **3.1. Few-Shot Learning Background** 

Let _D_ = _{Di}[D] i_ =1[be the set of] _[ D]_[ classification domains,] and _D_[¯] = _{X, Y } ∈D_ a task containing _n_ samples along with their designated true labels _{X,_[¯] _Y_[¯] _}_ = _{xj, yj}[n] j_ =1[.] Few-shot classification is defined as the problem of learning to correctly classify a query set _Q_ = _{XQ, YQ} ∼ D_[¯] by training on a support set _S_ = _{XS , YS } ∼ D_[¯] that contains 


![](raw/papers/2_LLM_NLP/LLM_paper_list/few-shot/iclr_2024_Neural_Fine-Tuning_Search_for_Few-Shot_Learning_images/iclr_2024_Neural_Fine-Tuning_Search_for_Few-Shot_Learning.pdf-0003-08.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/few-shot/iclr_2024_Neural_Fine-Tuning_Search_for_Few-Shot_Learning_images/iclr_2024_Neural_Fine-Tuning_Search_for_Few-Shot_Learning.pdf-0003-09.png)


**----- Start of picture text -----**<br>
(a) Correlation between inclusion/non-inclusion of learnable pa-<br>rameters  α  and  ϕ [′] , and validation performance.<br>**----- End of picture text -----**<br>



![](raw/papers/2_LLM_NLP/LLM_paper_list/few-shot/iclr_2024_Neural_Fine-Tuning_Search_for_Few-Shot_Learning_images/iclr_2024_Neural_Fine-Tuning_Search_for_Few-Shot_Learning.pdf-0003-10.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/few-shot/iclr_2024_Neural_Fine-Tuning_Search_for_Few-Shot_Learning_images/iclr_2024_Neural_Fine-Tuning_Search_for_Few-Shot_Learning.pdf-0003-11.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/few-shot/iclr_2024_Neural_Fine-Tuning_Search_for_Few-Shot_Learning_images/iclr_2024_Neural_Fine-Tuning_Search_for_Few-Shot_Learning.pdf-0003-12.png)


(b) Top 3 performing paths subject to diversity constraint. 

Figure 2: Qualitative analysis of our architecture search. Fig. 2a summarises the whole search space by answering the question: _How important is to adapt (α) or fine-tune (ϕ[′] ) each block?_ The color of each square indicates the point-biserial correlation (over all searched architectures) between adapting/fine-tuning layer _gi_ and validation performance. Fig. 2b shows the top 3 performing candidates subject to a diversity constraint, after 15 generations of evolutionary search. Dark blue indicates that the layer is adapted/fine-tuned and light blue that it is not. 

very few examples. This can be achieved by finding the parameters _θ_ of a classifier _fθ_ with the objective 


![](raw/papers/2_LLM_NLP/LLM_paper_list/few-shot/iclr_2024_Neural_Fine-Tuning_Search_for_Few-Shot_Learning_images/iclr_2024_Neural_Fine-Tuning_Search_for_Few-Shot_Learning.pdf-0003-16.png)


In practice, if _θ_ is randomly initialised and trained using stochastic gradient descent on a small support set _S_ , it will overfit and fail to generalise to _Q_ . To address this issue, one can exploit knowledge transfer from some seen classes to the novel classes. Formally, each domain _D_[¯] is partitioned into two disjoint sets _D_[¯] train and _D_[¯] test, which are commonly referred to as “meta-train” and “meta-test”, respectively. The labels in these sets are also disjoint, i.e., _Y_ train _∩ Y_ test = _∅_ . In that case, _θ_ is trained by maximising the objective in Eq. 1 using the meta-train set, but the overall objective is to perform adequately when transferring knowledge to meta-test. 

The knowledge transferred from meta-train to meta-test can take various forms [20]. As discussed earlier, we aim to generalise a family of few-shot methods [21, 29, 53] where parameters _θ_ are transferred before a subset of them _ϕ ⊂ θ_ are fine-tuned; and possibly extended by attaching additional “adapter” parameters _α_ that are trained for the target task. For meta-test, Eq. 1 can therefore be rewritten as 


![](raw/papers/2_LLM_NLP/LLM_paper_list/few-shot/iclr_2024_Neural_Fine-Tuning_Search_for_Few-Shot_Learning_images/iclr_2024_Neural_Fine-Tuning_Search_for_Few-Shot_Learning.pdf-0003-19.png)


In this paper, we focus on the problem of finding the optimal adaptation strategy in terms of (i) the optimal subset 


![](raw/papers/2_LLM_NLP/LLM_paper_list/few-shot/iclr_2024_Neural_Fine-Tuning_Search_for_Few-Shot_Learning_images/iclr_2024_Neural_Fine-Tuning_Search_for_Few-Shot_Learning.pdf-0004-00.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/few-shot/iclr_2024_Neural_Fine-Tuning_Search_for_Few-Shot_Learning_images/iclr_2024_Neural_Fine-Tuning_Search_for_Few-Shot_Learning.pdf-0004-01.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/few-shot/iclr_2024_Neural_Fine-Tuning_Search_for_Few-Shot_Learning_images/iclr_2024_Neural_Fine-Tuning_Search_for_Few-Shot_Learning.pdf-0004-02.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/few-shot/iclr_2024_Neural_Fine-Tuning_Search_for_Few-Shot_Learning_images/iclr_2024_Neural_Fine-Tuning_Search_for_Few-Shot_Learning.pdf-0004-03.png)


Figure 3: Population of paths(candidate architectures) in the search space after 1, 5, and 15 generations of evolutionary search. Each dot is a 2-d TSNE projection of the binary vector representing an architecture, and its color shows the validation performance for that architecture. The supernet contains a wide variety of models in terms of validation performance, and the search algorithm converges to a well-performing population. The top 3 performing paths that are given in 2b are highlighted in the far right figure (Generation 15) in purple outline. 

of parameters _ϕ ⊂ θ_ that need to be fine-tuned, and (ii) the optimal task-specific parameters _α_ to add. 

## **3.2. Defining the search space** 

Let _gϕk_ be the minimal unit for adaptation in an architecture. We consider these to be the repeated units in contemporary deep architectures, e.g., a convolutional layer in a ResNet, or a self-attention block in a ViT. If the feature extractor _fθ_ comprises of _K_ such units with learnable parameters _ϕk_ , then we denote _θ_ =[�] _[K] k_ =1 _[ϕ][k]_[,][assuming][all] other parameters are kept fixed. For brevity in notation we will now omit the indices and refer to every such layer as _gϕ_ . Following the state-of-the-art [21, 29, 45, 53], let us also assume that task-specific adaptation can be performed either by inserting additional adapter parameters _α_ into _gϕ_ , or by fine-tuning the layer parameters _ϕ_ . 

This allows us to define the search space as two independent binary decisions per layer: (i) The inclusion or noninclusion of an adapter module attached to _gϕ_ , and (ii) the decision of whether to use the pre-trained parameters _ϕ_ , or replace them with their fine-tuned counterparts _ϕ[′]_ . The size of the search space is, therefore, (2[2] ) _[K]_ = 4 _[K]_ . For ResNets, we use the proposed adaptation architecture of TSA [29], where a residual adapter _hα_ , parameterised by _α_ , is con- 

||_gϕ,ϕ′,α_(_x_)(ResNet)|_gϕ,ϕ′,α_(_x_)(ViT)|
|---|---|---|
|_ϕ_,_−_<br>_ϕ_,_α_<br>_ϕ′_,_−_<br>_ϕ′, α_|_gϕ_(_x_)<br>_gϕ_(_x_) +_hα_(_x_)<br>_gϕ′_(_x_)<br>_gϕ′_(_x_) +_hα_(_x_)|_z_(_Aqkv_[_q_ ; _gϕ_(_x_)])<br>_z_(_Aqkv_[_q_ ; _gϕ_(_x_)] +_hα_1) +_hα_2<br>_z_(_Aqkv_[_q_ ; _gϕ′_(_x_)])<br>_z_(_Aqkv_[_q_ ; _gϕ′_(_x_)] +_hα_1) +_hα_2|



Table 1: The search space, as described in Section 3.2. When sampling a layer _gϕ,ϕ′,α_ , it can be sampled in one of the following variants: (i) _ϕ_ : fixed pre-trained parameters, no adaptation, (ii) _α_ : fixed pre-trained parameters, with adaptation, (iii) _ϕ[′]_ : fine-tuned parameters, no adaptation, (iv) _ϕ[′] , α_ fine-tuned-parameters, with adaptation. 

**Algorithm 1:** Supernet training. 

|**Input:** Supernet_fθ,α,ϕ′_. Datasets_D_. Step sizes_η_1,|
|---|
|_η_2. Path pool_P_. Prototypical loss_L_(Eq.5).|
|**Output:** Trained supernet_fθ,α,ϕ′_.|
|**repeat**|
|Sample dataset ¯_D ∼D_|
|Sample episode_S_,_Q ∼_¯_D_|
|Sample path_p ∼P_ with learnable parameters|
|_αp_,_ϕ′_<br>_p_ and frozen parameters_ϕp ⊂θ_|
|_αp ←−αp −η_1_∇αpL_(_f p_<br>_θ,α,ϕ′, S, Q_)|
|_ϕ′_<br>_p ←−ϕ′_<br>_p −η_2_∇ϕ′pL_(_f p_<br>_θ,α,ϕ′, S, Q_)|
|**until**prototypical loss converges|



nected to _gϕ_ 


![](raw/papers/2_LLM_NLP/LLM_paper_list/few-shot/iclr_2024_Neural_Fine-Tuning_Search_for_Few-Shot_Learning_images/iclr_2024_Neural_Fine-Tuning_Search_for_Few-Shot_Learning.pdf-0004-14.png)


where _x ∈_ R _[W,H,C]_ . For ViTs, we use the proposed adaptation architecture of ETT [53], where a tuneable prefix is prepended to the multi-head self-attention module _Aqkv_ , and a residual adapter is appended to both _Aqkv_ and the feed-forward module _z_ in each decoder block 


![](raw/papers/2_LLM_NLP/LLM_paper_list/few-shot/iclr_2024_Neural_Fine-Tuning_Search_for_Few-Shot_Learning_images/iclr_2024_Neural_Fine-Tuning_Search_for_Few-Shot_Learning.pdf-0004-16.png)


where _x ∈_ R _[D]_ and [ _·_ ; _·_ ] denotes the concatenation operation. Note that in the case of ViTs the adapter is not a function of the input features, but simply an added offset. 

Irrespective of the architecture, every layer _gϕ,ϕ′,α_ is parameterised by three sets of parameters, _ϕ_ , _ϕ[′]_ , and _α_ , denoting the initial parameters, fine-tuned parameters and adapter parameters respectively. Consequently, when sampling a configuration (i.e., path) from that search space, every such layer can be sampled as one of the variants listed in Table 1. 

## **3.3. Training the supernet** 

Following SPOS [18], our search space is actualised in the form of a supernet _fθ,α,ϕ′_ ; a “super” architecture that 

contains all possible architectures derived from the decisions detailed in Section 3.2. It is parameterised by: (i) _θ_ , the frozen parameters from the backbone architecture _fθ_ , (ii) _α_ , from the adapters _hα_ , and (iii) _ϕ[′]_ , from the fine-tuned parameters per layer _gϕ,ϕ′,α_ . 

We use a prototypical loss _L_ ( _f, S, Q_ ) as the core objective during supernet training and the subsequent search and fine-tuning. 


![](raw/papers/2_LLM_NLP/LLM_paper_list/few-shot/iclr_2024_Neural_Fine-Tuning_Search_for_Few-Shot_Learning_images/iclr_2024_Neural_Fine-Tuning_Search_for_Few-Shot_Learning.pdf-0005-02.png)


where _CQi_ denotes the embedding of the class centroid that corresponds to the true class of _Qi_ , and _dcos_ denotes the cosine distance. The set of class centroids _C_ is computed as the mean embeddings of support examples that belong to the same class: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/few-shot/iclr_2024_Neural_Fine-Tuning_Search_for_Few-Shot_Learning_images/iclr_2024_Neural_Fine-Tuning_Search_for_Few-Shot_Learning.pdf-0005-04.png)


where _L_ denotes the number of unique labels in _S_ . 

For supernet training specifically, let _P_ be a set of size 4 _[K]_ , enumerating all possible sequences of _K_ layers that can be sampled from the search space. Denoting a path sampled from the supernet as _fθ,α,ϕ[p][′]_[, we minimise an expectation of] the loss in Eq. 5 over multiple episodes and paths, so the final objective becomes: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/few-shot/iclr_2024_Neural_Fine-Tuning_Search_for_Few-Shot_Learning_images/iclr_2024_Neural_Fine-Tuning_Search_for_Few-Shot_Learning.pdf-0005-07.png)


Algorithm 1 summarises the supernet training algorithm in pseudocode. 

## **3.4. Searching for an optimal path** 

A supernet _fθ,α,ϕ′_ trained with the method described in Section 3.3 contains 4 _[K]_ models, intertwined via weight sharing. As explained in Section 1, our goal is to search for the best-performing one, but the main challenge is related to the fact that we do not know what data is going to be used for adaptation at test time. One extreme approach, would be to search for a single solution at training time and simply use it throughout the entire test, regardless of the potential domain shift. Another, would be to defer the search and perform it from scratch each time a new support set is given to us at test time. However, both have their shortcomings. As such, we propose a generalization of this process where searching is split into two phases – one during training, and a subsequent one during testing. 

**Meta-training time.** The search is responsible for preselecting a set of _N_ models from the entire search space. Its main purpose is to mitigate potential overfitting that can happen at test time, when only a small amount of data is 

**Algorithm 2:** Training time evolutionary search. 

**Input:** Supernet _fθ,α,ϕ′_ . Datasets _D_ . Step sizes _η_ 1, _η_ 2. Prototypical loss _L_ (Eq. 5). NCC accuracy _A_ (Eq. 11). **Output:** Optimal path _p[∗]_ . Initialise population _P_ randomly Initialise fitness of _P_ as Ψ _P ←−_ 0 **repeat** 


![](raw/papers/2_LLM_NLP/LLM_paper_list/few-shot/iclr_2024_Neural_Fine-Tuning_Search_for_Few-Shot_Learning_images/iclr_2024_Neural_Fine-Tuning_Search_for_Few-Shot_Learning.pdf-0005-14.png)


offspring _←−_ recombine the _M_ best candidates of _P_ w.r.t. Ψ _P P ←− P_ + offspring eliminate the _M_ worst candidates of _P_ w.r.t. Ψ _P_ **until** population fitness converges or max. iterations 

available, while providing enough diversity to successfully adjust the architecture to the diverse set of test domains. Formally, we search for a sequence of paths ( _p_ 1 _, p_ 2 _, ..., pN_ ) where: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/few-shot/iclr_2024_Neural_Fine-Tuning_Search_for_Few-Shot_Learning_images/iclr_2024_Neural_Fine-Tuning_Search_for_Few-Shot_Learning.pdf-0005-17.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/few-shot/iclr_2024_Neural_Fine-Tuning_Search_for_Few-Shot_Learning_images/iclr_2024_Neural_Fine-Tuning_Search_for_Few-Shot_Learning.pdf-0005-18.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/few-shot/iclr_2024_Neural_Fine-Tuning_Search_for_Few-Shot_Learning_images/iclr_2024_Neural_Fine-Tuning_Search_for_Few-Shot_Learning.pdf-0005-19.png)


where _T_ denotes a scalar threshold for the cosine distance between paths _pk_ and _pj_ , and _A_ is the classification accuracy of a nearest centroid classifier (NCC) [43], 


![](raw/papers/2_LLM_NLP/LLM_paper_list/few-shot/iclr_2024_Neural_Fine-Tuning_Search_for_Few-Shot_Learning_images/iclr_2024_Neural_Fine-Tuning_Search_for_Few-Shot_Learning.pdf-0005-21.png)


Noticeably, we measure accuracy of a solution using a query set, after fine-tuning on a separate support set (Eq. 9), then average across multiple episodes to avoid overfitting to a particular support set (Eq. 8). We also employ a diversity constraint, in the form of cosine distance between binary encodings of selected paths (Eq. 10), to allow for sufficient flexibility in the following test time search. 

To efficiently obtain sequence _{p_ 1 _, ..., pN }_ , we use evolutionary search to find points that maximise Eq. 8, and afterwards select the _N_ best performers from the evolutionary 

search history that satisfy the constraint in Eq. 10. Algorithm 2 summarises training-time search. 

**Meta-testing time.** For a given meta-test episode, we decide which one of the pre-selected _N_ models is best suited for adaptation on the given support set data. It acts as a failsafe to counteract the bias of the initial selection made at training time in cases when the support set might be particularly out-of-domain. Formally, the final path _p[∗]_ to be used in a particular episode is defined as: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/few-shot/iclr_2024_Neural_Fine-Tuning_Search_for_Few-Shot_Learning_images/iclr_2024_Neural_Fine-Tuning_Search_for_Few-Shot_Learning.pdf-0006-02.png)


Noticeably, we test each of the _N_ models by fine-tuning it on the support set (Eq. 13) and testing its performance on the same support set (Eq. 12). This is because the support set is the only source of data we have at test time and we cannot extract a disjoint validation set from it without risking the quality of the fine-tuning process. It is important to note that, while this step risks overfitting, the pre-selection of models at training time, as described previously, should already limit the subsequent search to only models that are unlikely to overfit. Since _N_ is kept small in our experiments, we use a naive grid search to find _p[∗]_ . 

This approach is a generalization of the existing NAS approaches, as it recovers both when _N_ = 1 or _N_ = 4 _[K]_ . Our claim is that intermediate values of _N_ are more likely to give us better results than any of the extremes, due to the reasons mentioned earlier. In particular, we would expect pre-selecting 1 _< N ≪_ 4 _[K]_ models to introduce reasonable overhead at test time while improving results, especially in cases when exposure to different domains might be limited at training time. In our evaluation we compare _N_ = 3 and _N_ = 1 to test this hypothesis. We do not include comparison to _N_ = 4 _[K]_ as it is computationally infeasible in our setting (performing equivalent of training time search for each test episode would require us to fine-tune _≈_ 14 _∗_ 10[6] models in total). 

## **4. Experiments** 

## **4.1. Experimental setup** 

**Evaluation on Meta-Dataset** We evaluate NFTS on the extended version of Meta-Dataset [40, 46], currently the most commonly used benchmark for few-shot classification, consisting of 13 publicly available datasets: FGVC Aircraft, CU Birds, Describable Textures (DTD), FGVCx Fungi, ImageNet, Omniglot, QuickDraw, VGG Flowers, CIFAR-10/100, MNIST, MSCOCO, and Traffic Signs. There are 2 evaluation protocols: single domain learning and multi-domain learning. In the single domain setting, only ImageNet is seen during training and meta-training, 

while in the multi-domain setting the first eight datasets are seen (FGVC Aircraft to VGG Flower). For meta-testing at least 600 episodes are sampled for each domain. 

**Evaluation on Meta-Album** Further, we evaluate NFTS on the more recently introduced Meta-Album [47]. MetaAlbum is more diverse than Meta-Dataset. We use the currently available Sets 0-2, which contain over 1000 unique labels across 30 datasets spanning 10 domains including microscopy, remote sensing, manufacturing, plant disease, character recognition and human action recognition tasks, etc. Unlike Meta-Dataset, where their default evaluation protocol is variable-way variable-shot, Meta-Album evaluation follows a 5-way variable-shot setting, where the number of shots is typically 1, 5, 10 and 20. For meta-testing, results are averaged over 1800 episodes. 

**Architectures** We employ two different backbone architectures, a ResNet-18 [19] and a ViT-small [11]. Following TSA [29], the ResNet-18 backbone is pre-trained on the seen domains with the knowledge-distillation method URL [28] and, following ETT [53], the ViT-small backbone is pre-trained on the seen portion of ImageNet with the self-supervised method DINO [6]. We consider TSA residual adapters [29, 39] for ResNet and Prefix Tuning [30, 53] adapters for ViT. This is mainly to enable direct comparison with prior work on the same base architectures that use exactly these same adapter families, without introducing new confounders in terms of mixing adapter types [29, 53]. However our framework is flexible, meaning it can accept any adapter type, or even multiple types in its search space. 

## **4.2. Comparison to state-of-the-art** 

**Meta-Dataset** The results on Meta-Dataset are shown in Table 2 and Table 3 for single-domain and multi-domain training setting respectively. We can see that NFTS obtains the best average performance across all the competitor methods for both ResNet and ViT architectures. The margins over prior state-of-the-art are often substantial for this benchmark with +1.9% over TSA in ResNet-18 single domain, +2.3% in multi-domain and +1.6% over ETT in VITsmall single domain. The increased margin in the multidomain case is intuitive, as our framework has more data with which to learn the optimal path(s). 

We re-iterate that PMF, ETT, and TSA are special cases of our search space corresponding respectively to: (i) Finetune all and include no adapters, (ii) Include ETT adapters at every layer while freezing all backbone weights and (iii) Include TSA adapters at every layer while freezing all backbone weights. We also share initial pre-trained backbones with ETT and TSA (but not PMF, as it uses a stronger pre-trained model with additional data). Thus the margins achieved over these competitors are attributable to our systematic approach to finding suitable architectures in terms of where to fine-tune and where to insert new adapter pa- 

|Method|Aircrafts<br>Birds<br>DTD<br>Fungi<br>ImageNet<br>Omniglot<br>QuickDraw<br>Flowers<br>CIFAR-10<br>CIFAR-100<br>MNIST<br>MSCOCO<br>Tr. Signs|Average|
|---|---|---|
|ResNet-18<br>FLUTE [32]<br>ProtoNet [43]<br>BOHB [42]<br>FO-MAML [46]<br>TSA [29]<br>NFTS|48.5<br>47.9<br>63.8<br>31.8<br>46.9<br>61.6<br>57.5<br>80.1<br>65.4<br>52.7<br>80.8<br>41.4<br>46.5<br>53.1<br>68.8<br>66.6<br>39.7<br>50.5<br>60.0<br>49.0<br>85.3<br>-<br>-<br>-<br>41.0<br>47.1<br>54.1<br>70.7<br>68.3<br>41.4<br>51.9<br>67.6<br>50.3<br>87.3<br>-<br>-<br>-<br>48.0<br>51.8<br>63.4<br>69.8<br>70.8<br>41.5<br>52.8<br>61.9<br>59.2<br>86.0<br>-<br>-<br>-<br>48.1<br>60.8<br>72.2<br>74.9<br>77.3<br>44.7<br>59.5<br>78.2<br>**67.6**<br>90.9<br>82.1<br>70.7<br>93.9<br>59.0<br>**82.5**<br>**74.9**<br>**76.5**<br>**81.6**<br>**50.5**<br>**62.7**<br>**80.2**<br>67.2<br>**94.5**<br>**83.0**<br>**71.5**<br>**94.0**<br>**59.7**<br>81.9|52.6<br>56.1<br>59.2<br>61.4<br>73.3<br>**75.2**|
|ViT-S<br>PMF_∗_[21]<br>ETT [53]<br>NFTS|76.8<br>85.0<br>86.6<br>54.8<br>**74.7**<br>80.7<br>71.3<br>94.6<br>-<br>-<br>-<br>**62.6**<br>**88.3**<br>79.9<br>**85.9**<br>**87.6**<br>61.8<br>67.4<br>78.1<br>71.3<br>**96.6**<br>-<br>-<br>-<br>62.3<br>85.1<br>**83.0**<br>85.5<br>**87.6**<br>**62.2**<br>71.0<br>**81.9**<br>**74.5**<br>96.0<br>79.4<br>72.6<br>95.2<br>**62.6**<br>87.9|77.5<br>77.6<br>**79.2**|



Table 2: Comparison to the state-of-the art methods on Meta-Dataset. Single domain setting – only ImageNet is seen during training and search. Reporting mean accuracy over 600 episodes. _[∗]_ Additional data used for training. 

|Method|Aircrafts<br>Birds<br>DTD<br>Fungi<br>ImageNet<br>Omniglot<br>QuickDraw<br>Flowers<br>CIFAR-10<br>CIFAR-100<br>MNIST<br>MSCOCO<br>Tr. Signs|Average|
|---|---|---|
|ResNet-18<br>CNAPS [40]<br>SCNAPS [50]<br>SUR [13]<br>URT [32]<br>FLUTE [45]<br>URL [28]<br>TSA [29]<br>NFTS|83.7<br>73.6<br>59.5<br>50.2<br>50.8<br>91.7<br>74.7<br>88.9<br>-<br>-<br>-<br>39.4<br>56.5<br>82.0<br>74.8<br>68.8<br>46.6<br>58.4<br>91.6<br>76.5<br>90.5<br>74.9<br>61.3<br>94.6<br>48.9<br>57.2<br>85.5<br>71.0<br>71.0<br>64.3<br>56.2<br>94.1<br>81.8<br>82.9<br>66.5<br>56.9<br>94.3<br>52.0<br>51.0<br>85.8<br>76.2<br>71.6<br>64.0<br>56.8<br>94.2<br>82.4<br>87.9<br>67.0<br>57.3<br>90.6<br>51.5<br>48.2<br>82.8<br>75.3<br>71.2<br>48.5<br>58.6<br>92.0<br>77.3<br>90.5<br>75.4<br>62.0<br>96.2<br>52.8<br>63.0<br>89.4<br>80.7<br>77.2<br>68.1<br>58.8<br>94.5<br>82.5<br>92.0<br>74.2<br>63.5<br>94.7<br>57.3<br>63.3<br>89.9<br>81.1<br>77.5<br>66.3<br>59.5<br>**94.9**<br>81.7<br>**92.2**<br>82.9<br>70.4<br>**96.7**<br>57.6<br>**82.8**<br>**90.1**<br>**83.8**<br>**82.3**<br>**68.4**<br>**61.4**<br>94.3<br>**82.6**<br>**92.2**<br>**83.0**<br>**75.1**<br>95.4<br>**58.8**<br>81.9|66.9<br>69.5<br>71.4<br>71.8<br>72.7<br>76.6<br>78.4<br>**80.7**|
|ViT-S<br>PMF_∗_[21]<br>NFTS|88.3<br>91.0<br>**86.6**<br>74.2<br>**74.6**<br>91.8<br>79.2<br>**94.1**<br>-<br>-<br>-<br>62.6<br>**88.9**<br>**89.1**<br>**92.5**<br>86.3<br>**75.1**<br>**74.6**<br>**92.0**<br>**80.6**<br>93.5<br>75.9<br>70.8<br>91.3<br>**62.8**<br>87.2|83.1<br>**83.4**|



Table 3: Comparison to the state-of-the art methods on Meta-Dataset. Multi-domain setting – the first 8 datasets are seen during training and search. Reporting mean accuracy over 600 episodes. _[∗]_ Additional data used for training. 

## rameters. 

**Meta-Album** The results on Meta-Album are shown in Figure 4 as a function of number of shots within the 5-way setting, following [47]. We can see that across the whole range of support set sizes, our NFTS dominates all of the well-tuned baselines from [47]. The margins are substantial, greater than 5% at 5-way/5-shot operating point, for example. This result confirms that our framework scales to even more diverse datasets and domains than those considered previously in Meta-Dataset. 

## **4.3. Ablation study** 

To analyse more precisely the role that our architecture search plays in few-shot performance, we also conduct an ablation study of our final model against four corners of our search space: (i) Initial model only, using a pre-trained feature extractor and simple NCC classifier, which loosely corresponds to SimpleShot [50], (ii) Full adaptation only, using a fixed feature extractor, which loosely corresponds to TSA [29], ETT [53], FLUTE [45], and others – depending on base architecture and choice of adapter, (iii) Fully finetuned model, which loosely corresponds to PMF [21], and (iv) Combination of full fine-tuning and adaptation. From the results in Table 4 we can see that both fine-tuning (ii), adapters (iii), and their combination (iv) give improvement on the linear readout baseline (i). However, all of them are worse than the systematically optimised adaptation architecture of NFTS. 

The ablation also compares the results using the top-1 adaptation architecture found by SPOS architecture search 

|Method|Single Domain<br>Multi-Domain|
|---|---|
|ResNet-18<br>_ϕ , −_<br>_ϕ , α_<br>_ϕ′, −_<br>_ϕ′, α_<br>NFTS-1<br>NFTS-N|67.8<br>67.8<br>70.4<br>76.5<br>70.2<br>76.3<br>70.8<br>76.9<br>73.6<br>80.1<br>75.2<br>80.7|
|ViT-S<br>_ϕ , −_<br>_ϕ , α_<br>_ϕ′, −_<br>_ϕ′, α_<br>NFTS-1<br>NFTS-N|71.8<br>71.8<br>73.8<br>77.3<br>74.0<br>77.5<br>74.4<br>78.9<br>78.7<br>83.1<br>79.2<br>83.4|



Table 4: Ablation study on Meta-Dataset comparing four special cases of the search space in terms of average accuracy: (i) _ϕ, −_ : No adaptation, no fine-tuning, (ii) _ϕ, α_ : Adapt all, (iii) _ϕ[′] , −_ : Fine-tune all, (iv) _ϕ[′] , α_ : Adapt and fine-tune all. NFTS- _{_ 1,N _}_ refer to conventional and deferred episode-wise NAS respectively. 

against our novel progressive approach that defers the final architecture selection to an episode-wise decision. Our deferred architecture selection improves on fixing the top1 architecture from meta-train, demonstrating the value of per-dataset/episode architecture selection (see also Sec 4.4). 

## **4.4. Further analysis** 

The ablation study shows quantitatively the benefit of adaptation architecture search over common fixed adaptation strategies. In this Section, we aim to analyse: What 


![](raw/papers/2_LLM_NLP/LLM_paper_list/few-shot/iclr_2024_Neural_Fine-Tuning_Search_for_Few-Shot_Learning_images/iclr_2024_Neural_Fine-Tuning_Search_for_Few-Shot_Learning.pdf-0008-00.png)


**----- Start of picture text -----**<br>
70<br>TrainFromScratch<br>FineTuning<br>65 MatchingNet<br>ProtoNet<br>60 FO-MAML<br>NFTS<br>55<br>50<br>45<br>40<br>35<br>30<br>1 5 10 20<br>Number of shots<br>Average accuracy (%)<br>**----- End of picture text -----**<br>


Figure 4: Comparison of our method against Meta-Album baselines, as reported in Fig. 2 of their paper [47]. The setting is 5-way [1, 5, 10, 20]-shot, and accuracy scores are averaged over 1800 tasks drawn from Set0, Set1 and Set2. 

kind of adaptation architecture is discovered by our NAS strategy, and how it is discovered? 

**Discovered Architectures** We first summarise results of the entire search space in terms of which layers are preferential to fine-tune or not, and which layers are preferential to insert adapters or not in Figure 2a. The blocks indicate layers (columns) and adapters/fine-tuning (rows), with the color indicating whether that architectural decision was positively (green) or negatively (red) correlated with validation performance. We can see that the result is complex, without a simple pattern, as assumed by existing work [21, 29, 53]. That said, our NAS does discover some interpretable trends. For example, adapters should be included at early/late ResNet-18 layers and not at layers 5-9. 

We next show the top three performing paths subject to diversity constraint in Figure 2b. We see that these follow the strong trends in the search space from Figure 2a. For example, they always adapt ( _α_ ) block 14 and never adapt block 9. However, otherwise they do include diverse decisions (such as whether to fine-tune ( _ϕ[′]_ ) block 15) which was not strongly indicated in Figure 2a. 

Finally, we analyse how our small set of _N_ = 3 candidate architectures in Figure 2b as used during meta-test. Recall that this small set allows us to perform an efficient minimal episode-wise NAS, including for novel datasets unseen during training. The results in Table 5 show how often each architecture is selected by held out datasets during metatest (shading), and what is the per-dataset performance using only that architecture. It shows how our approach successfully learns to select the most suitable architecture on a per-dataset basis, even for unseen datasets. This unique capability goes beyond prior work [21, 29, 53] where all 


![](raw/papers/2_LLM_NLP/LLM_paper_list/few-shot/iclr_2024_Neural_Fine-Tuning_Search_for_Few-Shot_Learning_images/iclr_2024_Neural_Fine-Tuning_Search_for_Few-Shot_Learning.pdf-0008-06.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/few-shot/iclr_2024_Neural_Fine-Tuning_Search_for_Few-Shot_Learning_images/iclr_2024_Neural_Fine-Tuning_Search_for_Few-Shot_Learning.pdf-0008-07.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/few-shot/iclr_2024_Neural_Fine-Tuning_Search_for_Few-Shot_Learning_images/iclr_2024_Neural_Fine-Tuning_Search_for_Few-Shot_Learning.pdf-0008-08.png)


|||||||||
|---|---|---|---|---|---|---|---|
|CIFAR-10<br>CIFAR-100<br>MNIST<br>MSCOCO<br>Tr. Signs|82.0<br>81.2<br>**83.3**|||||||
||**75.9**<br>75.0<br>75.1|||||||
||**95.5**<br>94.4<br>95.1|||||||
||**58.1**<br>57.8<br>56.4|||||||
||81.7<br>**82.2**<br>81.8|||||||



Table 5: How the diverse selection of architectures from Fig. 2b perform per unseen downstream domain in MetaDataset. Shading indicates episode-wise architecture selection frequency, numbers indicate accuracy using the corresponding architecture. The best dataset-wise architecture (bold) is most often selected (shading). 

domains must rely on the same adaptation strategy despite their diverse adaptation needs. 

**Path Search Process** In addition, we illustrate the path search process in Figure 3. This figure shows a 2D t- SNE projection of our 2 _K_ -dimensional architecture search space, where the dots are candidate architectures of the evolutionary search process at different iterations. The dots are colored according to their validation accuracy. From the results we can see that: The initial set of candidates is broadly dispersed and generally low performing (left), and gradually converge toward a tighter cluster of high performing candidates (right). The top 3 performing paths subject to a diversity constraint (also illustrated in Fig. 2b) are annotated in purple outline. 

**Discussion** As analysed in Section 4.3, our approach can be used in either top-1 – where each episode is a pure finetuning operation given the chosen architecture; or top-N architecture mode as discussed above – where each episode performs a mini architecture selection based on the short listed produced during evolutionary search, as well as finetuning. We remark that while the latter imposes a slightly increased cost during testing ( _N_ = 3 _×_ in practice), this is similar or less than competitors who repeat adaptation with different learning rates during testing [21] (4 _×_ cost), or exploit a backbone ensemble (8 _×_ cost) [13, 32]. 

## **5. Conclusions** 

In this paper we present NFTS, a novel neural architecture-search based approach that discovers the optimal adaptation architecture for gradient-based few-shot learning. NFTS contains several recent strong heuristic adaptation architectures as special cases within its search space, and we show that by systematic architecture search they are all outperformed, leading to a new state-of-theart on Meta-Dataset and Meta-Album. While in this paper we use a simple and coarse search space for easy and direct comparison to prior work’s hand-designed adaptation strategies, in future work we will extend this framework to include a richer range of adaptation strategies, and a finergranularity of search. 

## **References** 

- [1] Mohamed S Abdelfattah, Abhinav Mehrotra, Łukasz Dudziak, and Nicholas Donald Lane. Zero-cost proxies for lightweight NAS. In _International Conference on Learning Representations_ , 2021. 3 

- [2] Peyman Bateni, Raghav Goyal, Vaden Masrani, Frank Wood, and Leonid Sigal. Improved few-shot visual classification. In _CVPR_ , 2020. 1, 3 

- [3] Andrew Brock, Theo Lim, James M. Ritchie, and Nick Weston. SMASH: One-shot model architecture search through hypernetworks. In _ICLR_ , 2018. 2 

- [4] Han Cai, Chuang Gan, Tianzhe Wang, Zhekai Zhang, and Song Han. Once-for-all: Train one network and specialize it for efficient deployment. In _ICLR_ , 2020. 2, 3 

- [5] Han Cai, Ligeng Zhu, and Song Han. ProxylessNAS: Direct neural architecture search on target task and hardware. In _ICLR_ , 2019. 2 

- [6] Mathilde Caron, Hugo Touvron, Ishan Misra, Herv´e J´egou, Julien Mairal, Piotr Bojanowski, and Armand Joulin. Emerging properties in self-supervised vision transformers. In _ICCV_ , 2021. 2, 6 

- [7] Minghao Chen, Houwen Peng, Jianlong Fu, and Haibin Ling. Autoformer: Searching transformers for visual recognition. In _ICCV_ , 2021. 2 

- [8] Xiangxiang Chu, Bo Zhang, and Ruijun Xu. Fairnas: Rethinking evaluation fairness of weight sharing neural architecture search. In _ICCV_ , 2021. 2 

- [9] Yuanzheng Ci, Chen Lin, Ming Sun, Boyu Chen, Hongwen Zhang, and Wanli Ouyang. Evolving search space for neural architecture search. In _Proceedings of the IEEE/CVF International Conference on Computer Vision_ , pages 6659–6669, 2021. 3 

- [10] Guneet Singh Dhillon, Pratik Chaudhari, Avinash Ravichandran, and Stefano Soatto. A baseline for few-shot image classification. In _International Conference on Learning Representations_ , 2020. 1, 2 

- [11] Alexey Dosovitskiy, Lucas Beyer, Alexander Kolesnikov, Dirk Weissenborn, Xiaohua Zhai, Thomas Unterthiner, Mostafa Dehghani, Matthias Minderer, Georg Heigold, Sylvain Gelly, Jakob Uszkoreit, and Neil Houlsby. An image is worth 16x16 words: Transformers for image recognition at scale. In _ICLR_ , 2021. 2, 6 

- [12] Nikita Dvornik, Cordelia Schmid, and Julien Mairal. Selecting relevant features from a multi-domain representation for few-shot classification. In _ECCV_ , 2020. 1 

- [13] Nikita Dvornik, Cordelia Schmid, and Julien Mairal. Selecting relevant features from a universal representation for few-shot classification. In _ECCV_ , 2020. 7, 8 

- [14] Thomas Elsken, Jan Hendrik Metzen, and Frank Hutter. Neural architecture search: A survey. _Journal of Machine Learning Research_ , 20(55):1–21, 2019. 3 

- [15] Jiemin Fang, Yuzhu Sun, Qian Zhang, Yuan Li, Wenyu Liu, and Xinggang Wang. Densely connected search space for more flexible neural architecture search. In _CVPR_ , 2020. 3 

- [16] Chelsea Finn, Pieter Abbeel, and Sergey Levine. Modelagnostic meta-learning for fast adaptation of deep networks. In _ICML_ , 2017. 1 

- [17] Chelsea Finn and Sergey Levine. Meta-learning and universality: Deep representations and gradient descent can approximate any learning algorithm. In _ICLR_ , 2018. 3 

- [18] Zichao Guo, Xiangyu Zhang, Haoyuan Mu, Wen Heng, Zechun Liu, Yichen Wei, and Jian Sun. Single path one-shot neural architecture search with uniform sampling. In _ECCV_ , 2020. 2, 3, 4 

- [19] Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun. Deep residual learning for image recognition. In _CVPR_ , 2016. 2, 6 

- [20] Timothy M Hospedales, Antreas Antoniou, Paul Micaelli, and Amos J. Storkey. Meta-Learning in Neural Networks: A Survey. _IEEE Transactions on Pattern Analysis and Machine Intelligence_ , pages 1–1, 2021. 3 

- [21] Shell Xu Hu, Da Li, Jan St¨uhmer, Minyoung Kim, and Timothy M. Hospedales. Pushing the limits of simple pipelines for few-shot learning: External data and fine-tuning make a difference. In _CVPR_ , 2022. 1, 2, 3, 4, 7, 8 

- [22] Menglin Jia, Luming Tang, Bor-Chun Chen, Claire Cardie, Serge Belongie, Bharath Hariharan, and Ser-Nam Lim. Visual prompt tuning. In _ECCV_ , 2022. 2 

- [23] Brenden M. Lake, Ruslan Salakhutdinov, Jason Gross, and Joshua B. Tenenbaum. One shot learning of simple visual concepts. In _CogSci_ , 2011. 1 

- [24] Brenden M. Lake, Ruslan Salakhutdinov, and Joshua B. Tenenbaum. Human-level concept learning through probabilistic program induction. _Science_ , 2015. 1 

- [25] Kwonjoon Lee, Subhransu Maji, Avinash Ravichandran, and Stefano Soatto. Meta-learning with differentiable convex optimization. In _CVPR_ , 2019. 1 

- [26] Changlin Li, Jiefeng Peng, Liuchun Yuan, Guangrun Wang, Xiaodan Liang, Liang Lin, and Xiaojun Chang. Blockwisely supervised neural architecture search with knowledge distillation. In _CVPR_ , 2020. 3 

- [27] Guohao Li, Guocheng Qian, Itzel C Delgadillo, Matthias Muller, Ali Thabet, and Bernard Ghanem. Sgas: Sequential greedy architecture search. In _CVPR_ , 2020. 2 

- [28] Wei-Hong Li, Xialei Liu, and Hakan Bilen. Universal representation learning from multiple domains for few-shot classification. In _ICCV_ , 2021. 1, 2, 6, 7 

- [29] Wei-Hong Li, Xialei Liu, and Hakan Bilen. Cross-domain few-shot learning with task-specific adapters. In _CVPR_ , 2022. 1, 2, 3, 4, 6, 7, 8, 11 

- [30] Xiang Lisa Li and Percy Liang. Prefix-tuning: Optimizing continuous prompts for generation. In _ACL_ , 2021. 2, 6 

- [31] Hanxiao Liu, Karen Simonyan, and Yiming Yang. DARTS: Differentiable architecture search. In _International Conference on Learning Representations_ , 2019. 2, 3 

- [32] Lu Liu, William L. Hamilton, Guodong Long, Jing Jiang, and Hugo Larochelle. A universal representation transformer layer for few-shot image classification. In _ICLR_ , 2021. 1, 7, 8 

- [33] Erik G. Miller, Nicholas E. Matsakis, and Paul A. Viola. Learning from one example through shared densities on transforms. In _CVPR_ , 2000. 1 

- [34] Pavlo Molchanov, Jimmy Hall, Hongxu Yin, Jan Kautz, Nicol`o Fusi, and Arash Vahdat. LANA: latency aware network acceleration. In _ECCV_ , 2022. 3 

- [35] Bert Moons, Parham Noorzad, Andrii Skliar, Giovanni Mariani, Dushyant Mehta, Chris Lott, and Tijmen Blankevoort. Distilling optimal neural networks: Rapid search in diverse spaces. In _ICCV_ , 2021. 3 

- [36] Ethan Perez, Florian Strub, Harm de Vries, Vincent Dumoulin, and Aaron C. Courville. Film: Visual reasoning with a general conditioning layer. In _AAAI_ , 2018. 2 

- [37] Ilija Radosavovic, Justin Johnson, Saining Xie, Wan-Yen Lo, and Piotr Doll´ar. On network design spaces for visual recognition. In _ICCV_ , 2019. 3 

- [38] Sachin Ravi and Hugo Larochelle. Optimization as a model for few-shot learning. In _ICLR_ , 2017. 1 

- [39] Sylvestre-Alvise Rebuffi, Hakan Bilen, and Andrea Vedaldi. Learning multiple visual domains with residual adapters. In _NeurIPS_ , 2017. 2, 6 

- [40] James Requeima, Jonathan Gordon, John Bronskill, Sebastian Nowozin, and Richard E. Turner. Fast and flexible multi-task classification using conditional neural adaptive processes. In _NeurIPS_ , 2019. 1, 3, 6, 7 

   - [51] Yaqing Wang, Quanming Yao, James T. Kwok, and Lionel M. Ni. Generalizing from a few examples: A survey on few-shot learning. _ACM Comput. Surv._ , 2020. 1 

   - [52] Lichuan Xiang, Łukasz Dudziak, Mohamed S. Abdelfattah, Thomas Chau, Nicholas D. Lane, and Hongkai Wen. Zerocost operation scoring in differentiable architecture search. In _AAAI_ , 2023. 3 

   - [53] Chengming Xu, Siqian Yang, Yabiao Wang, Zhanxiong Wang, Yanwei Fu, and Xiangyang Xue. Exploring efficient few-shot adaptation for vision transformers. _TMLR_ , 2022. 1, 2, 3, 4, 6, 7, 8, 11 

   - [54] M. D. Zeiler and R. Fergus. Visualizing and understanding convolutional networks. In _ECCV_ , 2014. 2 

   - [55] Yuanhan Zhang, Kaiyang Zhou, and Ziwei Liu. Neural prompt search. _CoRR_ , 2022. 2 

   - [56] Daquan Zhou, Xiaojie Jin, Xiaochen Lian, Linjie Yang, Yujing Xue, Qibin Hou, and Jiashi Feng. Autospace: Neural architecture search with less human interference. In _ICCV_ , 2021. 3 

- [41] Andrei A. Rusu, Dushyant Rao, Jakub Sygnowski, Oriol Vinyals, Razvan Pascanu, Simon Osindero, and Raia Hadsell. Meta-learning with latent embedding optimization. In _ICLR_ , 2019. 1 

- [42] Tonmoy Saikia, Thomas Brox, and Cordelia Schmid. Optimized generic feature learning for few-shot classification across domains. _arXiv preprint arXiv:2001.07926_ , 2020. 7 

- [43] Jake Snell, Kevin Swersky, and Richard S. Zemel. Prototypical networks for few-shot learning. In _NeurIPS_ , 2017. 1, 5, 7 

- [44] Yonglong Tian, Yue Wang, Dilip Krishnan, Joshua B Tenenbaum, and Phillip Isola. Rethinking few-shot image classification: a good embedding is all you need? In _ECCV_ , 2020. 1 

- [45] Eleni Triantafillou, Hugo Larochelle, Richard S. Zemel, and Vincent Dumoulin. Learning a universal template for fewshot dataset generalization. In _ICML_ , 2021. 2, 4, 7 

- [46] Eleni Triantafillou, Tyler Zhu, Vincent Dumoulin, Pascal Lamblin, Utku Evci, Kelvin Xu, Ross Goroshin, Carles Gelada, Kevin Swersky, Pierre-Antoine Manzagol, and Hugo Larochelle. Meta-dataset: A dataset of datasets for learning to learn from few examples. In _ICLR_ , 2020. 1, 2, 6, 7 

- [47] Ihsan Ullah, Dustin Carrion, Sergio Escalera, Isabelle M Guyon, Mike Huisman, Felix Mohr, Jan N van Rijn, Haozhe Sun, Joaquin Vanschoren, and Phan Anh Vu. Meta-album: Multi-domain meta-dataset for few-shot image classification. In _NeurIPS Datasets and Benchmarks Track_ , 2022. 1, 2, 6, 7, 8 

- [48] Oriol Vinyals, Charles Blundell, Tim Lillicrap, Koray Kavukcuoglu, and Daan Wierstra. Matching networks for one shot learning. In _NeurIPS_ , 2016. 1 

- [49] Ruochen Wang, Minhao Cheng, Xiangning Chen, Xiaocheng Tang, and Cho-Jui Hsieh. Rethinking architecture selection in differentiable NAS. In _ICLR_ , 2021. 2 

- [50] Yan Wang, Wei-Lun Chao, Kilian Q. Weinberger, and Laurens van der Maaten. Simpleshot: Revisiting nearestneighbor classification for few-shot learning, 2019. 1, 7 

|Hyperparameter|ResNet-18<br>SDL(MD)<br>MDL(MD)<br>MDL(MA)|ViT-S<br>SDL(MD)<br>MDL(MD)|
|---|---|---|
||||
|Backbone architecture<br>Adapter architecture|URL<br>URL<br>Supervised<br>TSA<br>TSA<br>TSA|DINO<br>DINO<br>ETT<br>ETT|
|TRAIN<br>Number of episodes<br>Number of epochs<br>Optimizer<br>Learning rate<br>Learning rate schedule<br>Learning rate warmup<br>Weight decay<br>Weight decay schedule|50000<br>80000<br>20000<br>1<br>1<br>1<br>adadelta<br>adadelta<br>adadelta<br>0.05<br>0.05<br>0.05<br>-<br>-<br>-<br>-<br>-<br>-<br>0.0001<br>0.0001<br>0.0001<br>-<br>-<br>-|80000<br>160000<br>1<br>1<br>adamw<br>adamw<br>0.00007<br>0.00007<br>cosine<br>cosine<br>linear<br>linear<br>0.01<br>0.01<br>cosine<br>cosine|
|SEARCH<br>Number of episodes<br>Number of epochs<br>Optimizer<br>Learning rate<br>Weight decay<br>Initial population size<br>Top-K crossover<br>Mutation chance<br>Top-N paths<br>Diversity threshold|100<br>100<br>100<br>20<br>20<br>20<br>adadelta<br>adadelta<br>adadelta<br>0.1<br>0.1<br>0.1<br>0.0001<br>0.0001<br>0.0001<br>64<br>64<br>64<br>8<br>8<br>8<br>5%<br>5%<br>5%<br>3<br>3<br>3<br>0.4<br>0.4<br>0.4|100<br>100<br>40<br>40<br>adamw<br>adamw<br>0.000003<br>0.000003<br>0.1<br>0.1<br>64<br>64<br>8<br>8<br>5%<br>5%<br>3<br>3<br>0.2<br>0.2|
|TEST<br>Number of episodes<br>Number of epochs<br>Optimizer<br>Learning rate<br>Weight decay<br>Regulariser strength|600<br>600<br>1800<br>40<br>40<br>40<br>adadelta<br>adadelta<br>adadelta<br>0.1<br>0.1<br>0.1<br>0.0001<br>0.0001<br>0.0001<br>0.04<br>0.04<br>0.04|600<br>600<br>40<br>40<br>adamw<br>adamw<br>0.000003<br>0.000003<br>0.1<br>0.1<br>-<br>-|



Table 6: Hyperparameter setting for all experiments presented in Section 4 of the main paper. The notation is as follows: SDL=Single domain learning, MDL=Multi-domain learning, MD=Meta-Dataset, MA=Meta-Album, TRAIN=Supernet training phase, SEARCH=Evolutionary search phase, TEST=Meta-test phase. 

## **A. Hyperparameter Setting** 

Table 6 reports the hyperparameters used for all of our experiments. Note the following clarifications: 

- “Number of epochs” refers to multiple forward passes of the same episode, while “Number of episodes” refers to the number of episodes sampled in total. 

- The batch size is not mentioned, because we only conduct episodic learning, where we do not split the episode into batches, i.e., we feed the entire support and query set into our neural network architectures. 

- Learning rate warmup, where applicable, occurs for the first 10% of the episodes. 

We further specify something important: While our strongest competitors [29, 53] tune their learning rates for meta-testing (e.g., TSA uses LR=0.1 for seen domains and LR=1.0 for unseen, and ETT uses a different learning rate per downstream Meta-Dataset domain), we treat meta-testing episodes as completely unknown, and use the same hyperparameters we used on the validation set during search. 

## **B. Detailed Ablation Study** 

Tables 7 and 8 provide the exact scores per Meta-Dataset domain that are summarised in Table 4 of the main paper, for single domain and multi-domain FSL respectively. 

|Method|Aircrafts<br>Birds<br>DTD<br>Fungi<br>ImageNet<br>Omniglot<br>QuickDraw<br>Flowers<br>CIFAR-10<br>CIFAR-100<br>MNIST<br>MSCOCO<br>Tr. Signs|Average|
|---|---|---|
|ResNet-18<br>_ϕ , −_<br>_ϕ , α_<br>_ϕ′, −_<br>_ϕ′, α_<br>NFTS-1<br>NFTS-K|64.5<br>69.6<br>71.1<br>41.2<br>56.4<br>74.8<br>64.2<br>84.6<br>75.0<br>63.9<br>82.1<br>55.9<br>77.7<br>69.6<br>67.7<br>75.0<br>42.5<br>59.5<br>71.3<br>64.9<br>88.8<br>77.4<br>70.0<br>90.2<br>58.4<br>80.1<br>69.9<br>74.7<br>73.3<br>39.5<br>57.3<br>71.9<br>65.4<br>89.0<br>76.5<br>66.3<br>93.6<br>54.4<br>81.4<br>67.6<br>69.1<br>77.0<br>39.3<br>59.7<br>77.8<br>66.1<br>87.4<br>81.7<br>69.5<br>91.9<br>55.1<br>78.7<br>73.2<br>76.5<br>81.6<br>42.1<br>61.3<br>80.2<br>66.9<br>90.0<br>82.9<br>68.8<br>94.0<br>58.4<br>80.6<br>74.9<br>76.5<br>81.6<br>50.5<br>62.7<br>80.2<br>67.2<br>94.5<br>83.0<br>71.5<br>94.0<br>59.7<br>81.9|67.8<br>70.4<br>70.2<br>70.8<br>73.6<br>75.2|
|ViT-S<br>_ϕ , −_<br>_ϕ , α_<br>_ϕ′, −_<br>_ϕ′, α_<br>NFTS-1<br>NFTS-N|73.4<br>73.6<br>81.6<br>56.3<br>60.3<br>69.4<br>70.8<br>90.4<br>70.4<br>61.5<br>83.8<br>60.5<br>81.7<br>76.9<br>83.2<br>86.7<br>59.3<br>63.7<br>75.8<br>65.1<br>89.5<br>70.7<br>67.4<br>81.1<br>54.8<br>82.9<br>76.8<br>80.9<br>85.8<br>61.4<br>65.9<br>73.2<br>68.5<br>91.0<br>69.9<br>66.1<br>82.5<br>57.6<br>78.8<br>77.0<br>83.4<br>82.4<br>58.6<br>66.7<br>73.1<br>65.0<br>95.9<br>76.7<br>66.1<br>87.7<br>58.7<br>82.9<br>83.0<br>85.5<br>87.3<br>62.2<br>68.8<br>81.9<br>72.9<br>95.3<br>79.4<br>72.6<br>95.2<br>62.6<br>87.5<br>83.0<br>85.5<br>87.6<br>62.2<br>71.0<br>81.9<br>74.5<br>96.0<br>79.4<br>72.6<br>95.2<br>62.6<br>87.9|71.8<br>73.8<br>74.0<br>74.4<br>78.7<br>79.2|



Table 7: Ablation study on Meta-Dataset comparing four special cases of the search space: (i) _ϕ, −_ : No adaptation, no finetuning, (ii) _ϕ, α_ : Adapt all, (iii) _ϕ[′] , −_ : Fine-tune all, (iv) _ϕ[′] , α_ : Adapt and fine-tune all. NFTS- _{_ 1,N _}_ refer to conventional and deferred episode-wise NAS respectively. Single domain setting: Only ImageNet is seen during training and search. Reporting mean accuracy over 600 episodes. 

|Method|Aircrafts<br>Birds<br>DTD<br>Fungi<br>ImageNet<br>Omniglot<br>QuickDraw<br>Flowers<br>CIFAR-10<br>CIFAR-100<br>MNIST<br>MSCOCO<br>Tr. Signs|Average|
|---|---|---|
|ResNet-18<br>_ϕ , −_<br>_ϕ , α_<br>_ϕ′, −_<br>_ϕ′, α_<br>NFTS-1<br>NFTS-K|64.5<br>69.6<br>71.1<br>41.2<br>56.4<br>74.8<br>64.2<br>84.6<br>75.0<br>63.9<br>82.1<br>55.9<br>77.7<br>89.3<br>78.3<br>76.1<br>62.7<br>57.2<br>93.8<br>76.0<br>90.8<br>77.8<br>66.1<br>90.5<br>56.9<br>79.5<br>90.2<br>76.7<br>70.6<br>63.1<br>57.8<br>88.2<br>79.3<br>88.9<br>78.2<br>68.2<br>96.1<br>51.7<br>82.9<br>86.1<br>78.9<br>77.2<br>60.5<br>57.6<br>94.1<br>79.5<br>86.5<br>81.0<br>67.2<br>96.1<br>52.6<br>81.8<br>90.1<br>82.1<br>79.9<br>67.9<br>61.4<br>94.3<br>82.6<br>92.2<br>82.4<br>73.8<br>95.4<br>58.1<br>81.0<br>90.1<br>83.8<br>82.3<br>68.4<br>61.4<br>94.3<br>82.6<br>92.2<br>83.0<br>75.1<br>95.4<br>58.8<br>81.9|67.8<br>76.5<br>76.3<br>76.9<br>80.1<br>80.7|
|ViT-S<br>_ϕ , −_<br>_ϕ , α_<br>_ϕ′, −_<br>_ϕ′, α_<br>NFTS-1<br>NFTS-N|73.4<br>73.6<br>81.6<br>56.3<br>60.3<br>69.4<br>70.8<br>90.4<br>70.4<br>61.5<br>83.8<br>60.5<br>81.7<br>85.7<br>84.3<br>81.8<br>68.7<br>70.4<br>89.1<br>77.0<br>90.2<br>73.5<br>61.4<br>82.6<br>53.7<br>72.4<br>83.0<br>84.5<br>81.1<br>70.9<br>72.4<br>88.6<br>74.6<br>90.4<br>75.1<br>63.5<br>87.0<br>54.0<br>75.5<br>82.5<br>85.9<br>82.7<br>68.9<br>73.7<br>90.4<br>77.1<br>94.0<br>73.4<br>66.2<br>85.9<br>55.9<br>77.4<br>89.1<br>90.3<br>86.3<br>75.1<br>74.6<br>92.0<br>80.6<br>93.5<br>75.9<br>70.8<br>91.3<br>62.7<br>87.2<br>89.1<br>92.5<br>86.3<br>75.1<br>74.6<br>92.0<br>80.6<br>93.5<br>75.9<br>70.8<br>91.3<br>62.8<br>87.2|71.8<br>77.3<br>77.5<br>78.9<br>83.1<br>83.4|



Table 8: Ablation study on Meta-Dataset comparing four special cases of the search space: (i) _ϕ, −_ : No adaptation, no finetuning, (ii) _ϕ, α_ : Adapt all, (iii) _ϕ[′] , −_ : Fine-tune all, (iv) _ϕ[′] , α_ : Adapt and fine-tune all. NFTS- _{_ 1,N _}_ refer to conventional and deferred episode-wise NAS respectively. Multi-domain setting: The first 8 datasets are seen during training and search. Reporting mean accuracy over 600 episodes. 

## **C. Source code** 

The source code is available at: https://github.com/peustr/nfts-public. 

