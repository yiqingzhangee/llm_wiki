Published as a conference paper at ICLR 2024 

# COLEP: CERTIFIABLY ROBUST LEARNING-REASONING CONFORMAL PREDICTION VIA PROBABILISTIC CIRCUITS 

**Mintong Kang Nezihe Merve Gürel Linyi Li** UIUC TU Delft UIUC mintong2@illinois.edu n.m.gurel@tudelft.nl linyi2@illinois.edu 

**Bo Li** 

UChicago & UIUC bol@uchicago.edu 

## ABSTRACT 

Conformal prediction has shown spurring performance in constructing statistically rigorous prediction sets for arbitrary black-box machine learning models, assuming the data is exchangeable. However, even small adversarial perturbations during the inference can violate the exchangeability assumption, challenge the coverage guarantees, and result in a subsequent decline in prediction coverage. In this work, we propose the first certifiably robust learning-reasoning conformal prediction framework (COLEP) via probabilistic circuits, which comprises a data-driven _learning component_ that trains statistical models to learn different semantic concepts, and a _reasoning component_ that encodes knowledge and characterizes the relationships among the statistical knowledge models for logic reasoning. To achieve exact and efficient reasoning, we employ probabilistic circuits (PCs) to construct the reasoning component. Theoretically, we provide end-to-end certification of prediction coverage for COLEP under _ℓ_ 2 bounded adversarial perturbations. We also provide certified coverage considering the finite size of the calibration set. Furthermore, we prove that COLEP achieves higher prediction coverage and accuracy over a single model as long as the utilities of knowledge models are non-trivial. Empirically, we show the validity and tightness of our certified coverage, demonstrating the robust conformal prediction of COLEP on various datasets. 

## 1 INTRODUCTION 

Deep neural networks (DNNs) have demonstrated impressive achievements in different domains (He et al., 2016; Vaswani et al., 2017; Li et al., 2022b; Pan et al., 2019; Chen et al., 2018). However, as DNNs are increasingly employed in real-world applications, particularly those safety-critical ones such as autonomous driving (Xu et al., 2022a) and medical diagnosis (Kang et al., 2023a), concerns regarding their trustworthiness and reliability have emerged (Szegedy et al., 2014; Eykholt et al., 2018; Cao et al., 2021; Li et al., 2017; Wang et al., 2023). To address these concerns, it is crucial to provide the _worst-case certification_ for the model predictions (i.e., certified robustness) (Wong & Kolter, 2018; Cohen et al., 2019; Li et al., 2023), and develop techniques that enable users to statistically evaluate the _uncertainty_ linked to the model predictions (i.e., conformal prediction) (Lei et al., 2018; Solari & Djordjilovi´c, 2022). 

In particular, considering potential adversarial manipulations during test time, where a small perturbation could mislead the models to make incorrect predictions (Szegedy et al., 2014; Madry et al., 2018; Xiao et al., 2018; Cao et al., 2021; Kang et al., 2024b), different robustness certification approaches have been explored to provide the worst-case prediction guarantees for neural networks. On the other hand, conformal prediction has been studied as an effective tool for generating prediction sets that reflect the prediction uncertainty of a given black-box model, assuming that the data is exchangeable (Romano et al., 2020; Vovk et al., 2005). Such prediction _uncertainty_ has provided a probabilistic guarantee in addition to the _worst-case certification_ . However, it is unclear how such _uncertainty_ certification would perform in the worst-case adversarial environments. 

1 

Published as a conference paper at ICLR 2024 

Although the worst-case certification for conformal prediction is promising, it is challenging for purely data-driven models to achieve high **certified prediction coverage (i.e., worst-case coverage with bounded input perturbations)** . Recent studies on the learning-reasoning framework, which integrates extrinsic domain knowledge and reasoning capabilities into data-driven models, have demonstrated great success in improving the model worst-case certifications (Yang et al., 2022; Zhang et al., 2023). In this paper, we aim to bridge the _worst-case robustness certification_ and _uncertainty_ certification, and explore whether such knowledge-enabled logical reasoning could help improve the certified prediction coverage for conformal prediction. We ask: _How to efficiently integrate knowledge and reasoning capabilities into DNNs for conformal prediction? Can we prove that such knowledge and logical reasoning enabled framework would indeed achieve higher certified prediction coverage and accuracy than that of a single DNN?_ 

In this work, we first propose a certifiably robust learning-reasoning conformal prediction framework (COLEP) via probabilistic circuits (PCs). In particular, we train different data-driven DNNs in the “learning" component, whose logical relationships (e.g., a stop sign should be of octagon shape) are encoded in the “reasoning" component, which is constructed with probabilistic circuits (Darwiche, 1999; 2001) for logical reasoning as shown in Fig 1. Theoretically, we first derive the _certified coverage_ for COLEP under _ℓ_ 2 bounded perturbations. We also provide _certified coverage_ considering the finite size of the calibration set. We then prove that COLEP achieves higher certified coverage and accuracy than that of a single model as long as the knowledge models have non-trivial utilities. To our best knowledge, this is the first knowledge-enabled learning framework for conformal prediction and with provably higher certified coverage in adversarial settings. 

We have conducted extensive experiments on GTSRB, CIFAR-10, and AwA2 datasets to demonstrate the effectiveness and tightness of the certified coverage for COLEP. We show that the certified prediction coverage of COLEP is significantly higher compared with the SOTA baselines, and COLEP has weakened the tradeoff between prediction coverage and prediction set size. We perform a range of ablation studies to show the impacts of different types of knowledge. 

**Related work** Conformal prediction is a statistical tool to construct the prediction set with guaranteed prediction coverage (Vovk et al., 1999; 2005; Shafer & Vovk, 2008; Lei et al., 2013; Yang & Kuchibhotla, 2021; Solari & Djordjilovi´c, 2022; Jin et al., 2023), assuming that the data is exchangeable. Several works have explored the scenarios where the exchangeability assumption is violated (Jin et al., 2023; Barber et al., 2022; Ghosh et al., 2023; Kang et al., 2023c; 2024a). However, it is unclear how to provide the worst-case certification for such prediction coverage considering adversarial setting. Recently, Gendler et al. (Gendler et al., 2022) provided a certified method for conformal prediction by applying randomized smoothing to certify the non-conformity scores. However, such certified coverage could be loose due to the generic nature of randomized smoothing. We propose COLEP with a sensing-reasoning framework to achieve much stronger certified coverage by leveraging the knowledge-enabled logic reasoning capabilities. We provide a comprehensive literature review on certified robustness and logical reasoning in appendix B. 

## 2 PRELIMINARIES 

Suppose that we have _n_ data samples _{_ ( _Xi, Yi_ ) _}i[n]_ =1[with][features] _[X][i][∈]_[R] _[d]_[and][labels] _[Y][i][∈Y]_[:=] _{_ 1 _,_ 2 _, ..., Nc}_ . Assume that the data samples are drawn exchangeably from some unknown distribution _PXY_ . Given a desired coverage 1 _− α ∈_ (0 _,_ 1), conformal prediction methods construct a prediction set _C_[ˆ] _n,α ⊆Y_ for a new data sample ( _Xn_ +1 _, Yn_ +1) _∼ PXY_ with the guarantee of _marginal prediction coverage_ : P[ _Yn_ +1 _∈ C_[ˆ] _n,α_ ( _Xn_ +1)] _≥_ 1 _− α_ . 

In this work, we focus on the split conformal prediction setting (Lei et al., 2018; Solari & Djordjilovi´c, 2022), where the data samples are randomly partitioned into two disjoint sets: a training set _I_ tr and a calibration set _I_ cal = [ _n_ ] _\I_ tr. In here and what follows, [ _n_ ] := _{_ 1 _, · · · , n}_ . We fit a classifier _h_ ( _x_ ) : R _[d] �→Y_ to the training set _I_ tr to estimate the **conditional class probability** _πy_ ( _x_ ) := P[ _Y_ = _y|X_ = _x_ ] of _Y_ given _X_ . Using the estimated probabilities that we denote by _π_ ˆ _y_ ( _x_ ), we then compute a non-conformity score _Sπ_ ˆ _y_ ( _Xi, Yi_ ) for each sample in the calibration set _I_ cal to measure how much non-conformity each validation sample has with respect to its ground truth label. A commonly used non-conformity score for valid and adaptive coverage is introduced by (Romano et al., 2020) as: _Sπ_ ˆ _y_ ( _x, y_ ) =[�] _j∈Y[π]_[ˆ] _[j]_[(] _[x]_[)][I][[ˆ] _[π] j_[(] _[x]_[)] _[>][π]_[ˆ] _y_[(] _[x]_[)]][ + ˆ] _[π][y]_[(] _[x]_[)] _[u,]_[ where][ I][[] _[·]_[]][ is the indicator function and] _[ u]_[ is uniformly] sampled over the interval [0 _,_ 1]. Given a desired coverage 1 _− α_ , the prediction set of a new data point _Xn_ +1 is formulated as: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2024_COLEP_Certifiably_Robust_Learning-Reasoning_Conformal_Prediction_via_Probabilistic_Circuits_images/iclr_2024_COLEP_Certifiably_Robust_Learning-Reasoning_Conformal_Prediction_via_Probabilistic_Circuits.pdf-0002-08.png)


(1) 

2 

Published as a conference paper at ICLR 2024 

where _Q_ 1 _−α_ ( _{Sπ_ ˆ _y_ ( _Xi, Yi_ ) _}i∈I_ cal ) is the _⌈_ (1 _−α_ )(1+ _|I_ cal _|_ ) _⌉_ -th largest value of the set _{Sπ_ ˆ _y_ ( _Xi, Yi_ ) _}i∈I_ cal . The prediction set _C_[ˆ] _n,α_ ( _Xn_ +1) includes all the labels with a smaller non-conformity score than the (1 _− α_ )-quantile of scores in the calibration set. Since we assume the data samples are exchangeable, the marginal coverage of the prediction set _C_[ˆ] _n,α_ ( _Xn_ +1) is no less than 1 _− α_ . 

3 LEARNING-REASONING PIPELINE FOR CONFORMAL PREDICTION 

To achieve high certified prediction coverage against adversarial perturbations, we introduce the certifiably robust learning-reasoning conformal prediction framework (COLEP). For better readability, we provide structured tables for used notations through the analysis in Appendix K. 

## 3.1 OVERVIEW OF COLEP 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2024_COLEP_Certifiably_Robust_Learning-Reasoning_Conformal_Prediction_via_Probabilistic_Circuits_images/iclr_2024_COLEP_Certifiably_Robust_Learning-Reasoning_Conformal_Prediction_via_Probabilistic_Circuits.pdf-0003-05.png)


**----- Start of picture text -----**<br>
PC (Shape Knowledge)<br>Main  IsStopSignPC (Shape Knowledge) ⟹ IsOctagon (weight:  𝐰 )<br>Model B(𝜋&!)(!)<br>IsStopSignModel 𝜋"!! B(𝜋(B(𝜋&!)(!) [["]] )!))<br>B(1 −𝜋&!)B(𝜋(B(𝜋&!)(!)<br>IsOctagon 𝜋"" [[!]] B(1 −𝜋(( [["]] 𝑒)) [[#]] 𝜋"!! [["#$%&]]<br>IsRed 𝜋"" [["]] PC<br>IsDigit 𝜋"" [[#]] (Color Knowledge)PC<br>Knowledge ModelsModels … (Content Knowledge)PC<br>LearningKnowledge ModelsModels … Reasoning<br>**----- End of picture text -----**<br>


The COLEP is comprised of a data-driven _learning_ **Main IsStopSignPC (Shape Knowledge)** ⟹ **IsOctagon (weight:** 𝐰 **)** The _component_ learning and a logic-drivencomponent is equipped _reasoning component_ with a main. **IsStopSignModel** 𝜋"!! B(1 −𝜋&!)B(𝜋(B(𝜋&!)(!)[["]] ))!)) model to perform the main task of classification and _L_ knowledge models, each learns different concepts **IsOctagon** 𝜋""[[!]] B(1 −𝜋(([["]] 𝑒))[[#]] 𝜋"!![["#$%&]] from data. Following the learning component is the **IsRed** 𝜋""[["]] **PC** reasoning component, which consists of _R_ subcom- **IsDigit** 𝜋""[[#]] **(Color Knowledge)PC** ponents (e.g., PCs) responsible for encoding diversedomain knowledge and logical reasoning by charac- **LearningKnowledge ModelsModels** … **(Content Knowledge)PC** … **Reasoning** terizing the logical relationships among the learning Figure 1: Overview of COLEP. models. The COLEP framework is depicted in Fig 1. Concretely, consider an input _x ∈_ R _[d]_ (e.g., the stop sign image) and an untrusted main model (e.g., stop sign classification model) within the learning component that predicts the label of _x_ along with estimates of conditional class probabilities 

_π_ ˆ _j_ ( _x_ ) := P[ˆ _y_ = _j|X_ = _x_ ] _,_ (2) 

where _j ∈Y_ := [ _Nc_ ]. Note that the main model as a _Nc_ -class classification model can be viewed as ˆ _Nc_ binary classification models, each with success probability _πj∈_ [ _Nc_ ]( _·_ ). In Fig 1, we particularly ˆ illustrate COLEP using a binary stop sign classifier _πs_ ( _·_ ) as the main model for simplicity. Each ˆ knowledge model (e.g., octagon shape classification model _π[o]_ ( _·_ )) learns a concept from _x_ (e.g., octagon shape) and outputs its prediction denoted by _y_ ˆ[(] _[l]_[)] _∈{_ 0 _,_ 1 _}_ where _l ∈_ [ _L_ ] indexes a particular knowledge model. There can be only two possible outcomes: whether the _l_ -th knowledge model detects the concept _y_ ˆ[(] _[l]_[)] = 1 or not _y_ ˆ[(] _[l]_[)] = 0. Consistently with the notation of the main model, we denote the conditional concept probability of the _l_ -th knowledge model by 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2024_COLEP_Certifiably_Robust_Learning-Reasoning_Conformal_Prediction_via_Probabilistic_Circuits_images/iclr_2024_COLEP_Certifiably_Robust_Learning-Reasoning_Conformal_Prediction_via_Probabilistic_Circuits.pdf-0003-09.png)


(3) 

for _l ∈_ [ _L_ ] Once the learning models output their predictions along with estimated probabilities, the reasoning component accepts these estimates as input and passes them through _R_ distinct reasoning subcomponents in parallel. It then combines the probabilities at the outputs of the reasoning subcomponents to calculate the corrected estimates of conditional class probabilities by COLEP, which we denote as _π_ ˆ _j_[COLEP] ( _x_ ) for _j ∈_ [ _Nc_ ] (e.g., corrected probability of _x_ having a stop sign or not: _π_ ˆ _s_[COLEP] ( _x_ ) or 1- _π_ ˆ _s_[COLEP] ( _x_ )) and form the conformal prediction set with them. 

3.2 REASONING VIA PROBABILISTIC CIRCUITS 

We encode two types of knowledge in the reasoning component: (1) _preventive knowledge_ of the form “ _A_ = _⇒ B_ ” (e.g., “ _IsStopSign_ = _⇒ IsOctagon_ ”), and (2) _permissive knowledge_ of the form “ _B_ = _⇒ A_ ” (e.g., “ _HasContentStop_ = _⇒ IsStopSign_ ”), where _A_ represents the main task corresponding to the main model (e.g., “ _IsStopSign_ ”), _B_ as one of the _L_ knowledge models (e.g., “ _IsOctagon_ ”), and = _⇒_ denotes logical implication with a _conditional variable_ as the left operand and a _consequence variable_ on the right. In this part, we consider one particular reasoning subcomponent (i.e., one PC) and encode _H_ preventive and permissive knowledge rules _{Kh}h[H]_ =1[in it.][The] _[h][-th] knowledge rule Kh_ is assigned with weight _wh ∈_ R[+] . 

To enable exact and efficient reasoning, we need to seek for appropriate probabilistic models as the reasoning component. Recent work (Gürel et al., 2021; Yang et al., 2022) encode propositional logic rules with Markov Logic Networks (MLNs) (Richardson & Domingos, 2006), but the inference complexity is exponential regarding the number of knowledge models. Variational inference is efficient for reaonsing (Zhang et al., 2023), but induces error during reasoning. Thus, we use probabilistic circuits (PCs) to achieve better trade-offs between the model expressiveness and reasoning efficiency. 

3 

Published as a conference paper at ICLR 2024 

**Encode preventive and permissive knowledge rules with PCs** . PCs define a joint distribution over a set of random variables. PCs take an assignment of the random variables as input and output the joint probability of the assignment. Specifically, the input of PCs can be represented by an _Nc_ + _L_ sized vector of Bernoulli random variables (e.g., [ _IsStopSign_ , _IsOctagon_ , _IsRed_ , _IsDigit_ ]) with the corresponding vector of success probabilities (e.g., _π_ ˆ = [ˆ _πs,_ ˆ _π[o] ,_ ˆ _π[r] ,_ ˆ _π[d]_ ]) denoted as: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2024_COLEP_Certifiably_Robust_Learning-Reasoning_Conformal_Prediction_via_Probabilistic_Circuits_images/iclr_2024_COLEP_Certifiably_Robust_Learning-Reasoning_Conformal_Prediction_via_Probabilistic_Circuits.pdf-0004-02.png)


Note that _π_ ˆ is a concatenated vector of the estimated conditional class and concept probabilities of the main model and knowledge models, respectively (e.g., _π_ ˆ = [ˆ _πs,_ ˆ _π[o] ,_ ˆ _π[r] ,_ ˆ _π[d]_ ] with _Nc_ = 1 _, L_ = 2). Since conformal prediction is our main focus, the conditional probabilities _π_ ˆ in eq. (4) will be frequently referenced throughout the paper. To distinguish the index representing each class label _j ∈_ [ _Nc_ ], we use _j∀ ∈_ [ _Nc_ + _L_ ] to address entries of _Nc_ + _L_ sized vectors. 

The computation of PCs is based on an acyclic directed graph _G_ . Each leaf node _O_ leaf in _G_ represents a univariate distribution (e.g., _B_ (ˆ _πs_ ( _x_ )): Bernoulli distribution over _IsStopSign_ with success rate _π_ ˆ _s_ ( _x_ )). The output of a PC is typically the root node _O_ root representing the joint probability of an assignment (e.g., _p_ ( _IsStopSign_ = 1 _, IsOctagon_ = 1 _|x_ )). The graph _G_ contains internal sum nodes computing a convex combination of children nodes, and product nodes computing the product of them. 

Formally, we let the leaf node _O_ leaf[(] _[j][∀]_[)][in the computational graph] _[ G]_[be a Bernoulli distribution with] success probability _π_ ˆ _j∀_ ( _x_ ). Let _µ ∈ M_ := _{_ 0 _,_ 1 _}[N][c]_[+] _[L]_ be the index variable (not an observation) of a possible assignment for _Nc_ classes in the interest of the main model and _L_ concepts in the interest of _L_ knowledge models. We also define the factor function _F_ : _{_ 0 _,_ 1 _}[N][c]_[+] _[L] �→_ R[+] which takes as input a possible assignment _µ_ and outputs the factor value of the assignment: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2024_COLEP_Certifiably_Robust_Learning-Reasoning_Conformal_Prediction_via_Probabilistic_Circuits_images/iclr_2024_COLEP_Certifiably_Robust_Learning-Reasoning_Conformal_Prediction_via_Probabilistic_Circuits.pdf-0004-06.png)


where I[ _µ∼Kh_ ] = 1 indicates that the assignment _µ_ follows the given logic rule _Kh_ , and otherwise I[ _µ∼Kh_ ] = 0 (e.g., in the shape PC in fig. 1, _F_ ([1 _,_ 1]) = _e[w]_ , I[[1 _,_ 1] _∼K_ ] = 1, where _K_ is the rule “ _IsStopSign_ = _⇒ IsOctagon_ ” with weight _w_ ). _F_ ( _µ_ ) essentially measures how well the assignment _µ_ conforms to the specified knowledge rules. Following (Darwiche, 2002; 2003), we let sum nodes compute the logical “or” operation, and product nodes compute the logical “and” operation. We associate each assignment _µ ∈ M_ with the factor value _F_ ( _µ_ ) at the level of leaf nodes as fig. 1. Given a feasible assignment of random variables, the output of PC _O_ root computes the likelihood of the assignment (e.g., _p_ ( _IsStopSign_ = 1 _, IsOctogon_ = 1 _|x_ )). When considering the likelihood of the _j_ -th class label, we can marginalize class labels other than _j_ and formulate the class probability conditioned on input _x_ as: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2024_COLEP_Certifiably_Robust_Learning-Reasoning_Conformal_Prediction_via_Probabilistic_Circuits_images/iclr_2024_COLEP_Certifiably_Robust_Learning-Reasoning_Conformal_Prediction_via_Probabilistic_Circuits.pdf-0004-08.png)


where _T_ ( _a, b_ ) := log( _ab_ +(1 _− a_ )(1 _− b_ )) for simplicity of notations and _µj_ denotes the _j_ -th element of vector _µ_ . Note that the numerator and denominator of eq. (6) can be exactly and efficiently computed with a single forwarding pass of PCs (Hitzler & Sarker, 2022; Choi & Darwiche, 2017; Rooshenas & Lowd, 2014). We can also improve the expressiveness of the reasoning component by linearly combining outputs of _R_ PCs _π_ ˆ _j_[(] _[r]_[)] with coefficients _βr_ for the _r_ -th PC as _π_ ˆ _j_[COLEP] ( _x_ ) =[�] _r∈_ [ _R_ ] _[β][r][π]_[ˆ] _j_[(] _[r]_[)] ( _x_ ) . The core of this formulation is the mixture model involving a latent variable _r_ PC representing the PCs. In short, we can write _π_ ˆ _j_[COLEP] ( _x_ ) as the marginalized probability over the latent variable _r_ PC as _π_ ˆ _j_[COLEP] ( _x_ ) =[�] _r∈_ [ _R_ ][P][[] _[r]_[PC][=] _[ r]_[]] _[ ·]_[ ˆ] _[π] j_[(] _[r]_[)] ( _x_ ). Hence, the coefficient _βr_ for the _r_ -th PC are determined by P[ _r_ PC = _r_ ]. Although we lack direct knowledge of this probability, we can estimate it using the data by examining how frequently each PC _r_ correctly predicts the outcome across the given examples, similarly as in the estimation of prior class probabilities for Naive Bayes classifiers. **Illustrative example** . We open up the stop sign classification example in fig. 1 further as follows. The main model predicts that the image _x_ has a stop sign with probability _π_ ˆ _s_ ( _x_ ). In parallel to it, the knowledge model that detects octagon shape predicts that the image _x_ has an object with octagon ˆ “ ” shape with probability _πo_ ( _x_ ). Then we can encode the logic rule _IsStopSign_ = _⇒ IsOctagon_ with weight _w_ as fig. 1. The reasoning component can correct the output probability with encoded knowledge rules and improve the robustness of the prediction. Consider the following concrete example. Suppose that the adversarial example of a speed limit sign misleads the main classifier 

4 

Published as a conference paper at ICLR 2024 

to detect it as a stop sign with a large probability (e.g., _π_ ˆ _s_ = 0 _._ 9), but the octagon classifier is not misled and correctly predicts that the speed limit sign has a square shape (e.g., _π_ ˆ _o_ = 0 _._ 0). Then according the computation as eq. (6), the corrected probability of detecting a stop sign given the ˆ attacked speed limit image is _πs_[COLEP] = 0 _._ 9 _/_ (0 _._ 1 _e[w]_ + 0 _._ 9), which is down-weighted from the original ˆ wrong prediction _πs_ = 0 _._ 9 as we have a positive logical rule weight _w >_ 0. Therefore, the reasoning component can correct the wrong output probabilities with the knowledge rules and lead to a more robust prediction framework. 

## 3.3 CONFORMAL PREDICTION WITH COLEP 

After obtaining the corrected conditional class probabilities using COLEP in eq. (6), we move on to conformal prediction with COLEP. To begin with, we construct prediction sets for each class label within the learning component. Using eq. (1), the prediction set for the _j_ -th class, where _j ∈_ [ _Nc_ ], can be formulated as: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2024_COLEP_Certifiably_Robust_Learning-Reasoning_Conformal_Prediction_via_Probabilistic_Circuits_images/iclr_2024_COLEP_Certifiably_Robust_Learning-Reasoning_Conformal_Prediction_via_Probabilistic_Circuits.pdf-0005-04.png)


where 1 _− αj_ is the user-defined coverage level, similar to _α_ in the standard conformal prediction. The fact that element 1 is present in _C_[ˆ] _n,α_ COLEP _j j_ ( _Xn_ +1) signifies that the class label _j_ is a possible ground truth label for _Xn_ +1 and can be included in the final prediction set. That is: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2024_COLEP_Certifiably_Robust_Learning-Reasoning_Conformal_Prediction_via_Probabilistic_Circuits_images/iclr_2024_COLEP_Certifiably_Robust_Learning-Reasoning_Conformal_Prediction_via_Probabilistic_Circuits.pdf-0005-06.png)


where _α_ = max _j∈_ [ _Nc_ ] _{αj}_ . It is worth noting that if _αj_ is set to a fixed value for all _j ∈_ [ _Nc_ ], _α_ in the standard conformal prediction setting is recovered. In this work, we adopt a fixed _α_ for each _αj_ for simplicity, but note that our framework allows setting different miscoverage levels for each class _j_ , which is advantageous in addressing imbalanced classes. We defer more discussions on the flexibility of defining class-wise coverage levels to Appendix D.1. 

## 4 CERTIFIABLY ROBUST LEARNING-REASONING CONFORMAL PREDICTION 

In this section, we provide robustness certification of COLEP for conformal prediction. We start by providing the certification of the reasoning component in Section 4.1, and then move to end-to-end certified coverage as well as finite-sample certified coverage in Section 4.2. We defer all the proofs to Appendix. The overview of the certification framework is provided in fig. 4 in Appendix E. 

## 4.1 CERTIFICATION FRAMEWORK AND CERTIFICATION OF THE REASONING COMPONENT 

The inference time adversaries violate the data exchangeability assumption, compromising the guaranteed coverage. With COLEP, we aim to investigate (1) whether the guaranteed coverage can be preserved for COLEP using a prediction set that takes the perturbation bound into account, and (2) whether we can provide a worst-case coverage (a lower bound) using the standard prediction set as before in eq. (8). We provide certification for goal (1) and (2) in Thms. 2 and 3, respectively. 

The certification procedure can be decomposed into several subsequent stages. First, we **certify learning robustness** by deriving the bounds for _π_ ˆ within the learning component given the input perturbation bound _δ_ . There are several existing approaches to certify the robustness of models within this component (Cohen et al., 2019; Zhang et al., 2018; Gowal et al., 2018). In this work, we use randomized smoothing (Cohen et al., 2019) (details in Appendix C.3), and get the bound of _π_ ˆ under the perturbation. Next, we **certify reasoning robustness** by deriving the reasoning-corrected bounds using the bounds of the learning component (Thm. 1). Finally, we **certify the robustness of conformal prediction** considering the reasoning-corrected bounds (Thm. 2 and Thm. 3). 

As robustness of the learning component is performed using randomized smoothing, we directly focus on the robustness certification of reasoning component. 

**Theorem 1** (Bounds for Conditional Class Probabilities ˆ _πj_[COLEP] ( _x_ ) within the Reasoning Component) **.** _Given any input x and perturbation bound δ, we let_ [ˆ _πj∀_ ( _x_ ) _, π_ ˆ _j∀_ ( _x_ )] _be bounds for the estimated conditional class and concept probabilities by all models with j∀ ∈_ [ _Nc_ + _L_ ] _(for example, achieved via randomized smoothing). Let Vd[j][be the set of index of conditional variables in the PC except for] j ∈_ [ _Nc_ ] _and Vs[j][be that of consequence variables.][Then the bound for][ COLEP][-corrected estimate of] the conditional class probability π_ ˆ _j[COLEP] is given by:_ 

5 

Published as a conference paper at ICLR 2024 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2024_COLEP_Certifiably_Robust_Learning-Reasoning_Conformal_Prediction_via_Probabilistic_Circuits_images/iclr_2024_COLEP_Certifiably_Robust_Learning-Reasoning_Conformal_Prediction_via_Probabilistic_Circuits.pdf-0006-01.png)


_where T_ ( _a, b_ ) = log( _ab_ +(1 _− a_ )(1 _− b_ )) _. We similarly give the lower bound_ L[ˆ _πj[COLEP]_ ] _in Appendix F.1. Remarks._ Thm. 1 establishes a certification connection from the probability bound of the learning component to that of the reasoning component. Applying robustness certification methods to DNNs, we can get the probability bound of the learning component [ _π_ ˆ _, π_ ˆ] within bounded input perturbations. Thm. 1 shows that we can compute the probability bound after the reasoning component [L[ˆ _πj_[COLEP] ] _,_ U[ˆ _πj_[COLEP] ]] following eq. (9). Since Equation (9) presents an analytical formulation of the bound after the reasoning component as a function of the bounds before the reasoning component, we can directly and efficiently compute [L[ˆ _πj_[COLEP] ] _,_ U[ˆ _πj_[COLEP] ]] in evaluations. 

4.2 CERTIFIABLY ROBUST CONFORMAL PREDICTION 

In this section, we leverage the reasoning component bounds in Thm. 1 to perform calibration with a worst-case non-conformity score function for adversarial perturbations. This way, we perform certifiably robust conformal prediction in the adversary setting during the inference time. We formulate the procedure to achieve this in Thm. 2. 

**Theorem 2** (Certifiably Robust Conformal Prediction of COLEP) **.** _Consider a new test sample Xn_ +1 _drawn from PXY . For any bounded perturbation ∥ϵ∥_ 2 _≤ δ in the input space and the adversarial sample X_[˜] _n_ +1 := _Xn_ +1 + _ϵ, we have the following guaranteed marginal coverage:_ P[ _Yn_ +1 _∈ C_[ˆ] _n,α[COLEP][δ]_ ( _X_[˜] _n_ +1)] _≥_ 1 _− α_ (10) 

_if we construct the certified prediction set of COLEP where_ 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2024_COLEP_Certifiably_Robust_Learning-Reasoning_Conformal_Prediction_via_Probabilistic_Circuits_images/iclr_2024_COLEP_Certifiably_Robust_Learning-Reasoning_Conformal_Prediction_via_Probabilistic_Circuits.pdf-0006-07.png)


_Remarks._ Thm. 2 shows that the coverage guarantee of COLEP in the adversary setting is still valid (eq. (10)) if we construct the prediction set by considering the worst-case perturbation as in eq. (11). That is, the prediction set of COLEP in eq. (11) covers the ground truth of an adversarial sample with nominal level 1 _− α_ . To achieve that, we use a worst-case non-conformity score as in eq. (12) during calibration to counter the influence of adversarial sample during inference. The bound of output probability (U _δ_ [ˆ _πj_[COLEP] ] _,_ L _δ_ [ˆ _πj_[COLEP] ]) in eq. (12) can be computed by Thm. 1 to achieve end-to-end robustness certification of COLEP, which generally holds for any DNN and probabilistic circuits. We also consider the finite-sample errors of using randomized smoothing for the learning component certification as Anonymous (2023) and provided the theoretical statement and proofs in thm. 6 in Appendix F.3. 

**Theorem 3** (Certified (Worst-Case) Coverage of COLEP) **.** _Consider the new sample Xn_ +1 _drawn from PXY and adversarial sample X_[˜] _n_ +1 := _Xn_ +1 + _ϵ with any perturbation ∥ϵ∥_ 2 _≤ δ in the input space. We have:_ P[ _Yn_ +1 _∈ C_[ˆ] _n,α[COLEP]_ ( _X_[˜] _n_ +1)] _≥ τ[COLEP][cer]_ := _j_ min _∈_ [ _Nc_ ] � _τj[COLEP][cer]_ � _,_ (13) 

_where the certified (worst-case) coverage of the j-th class label τj[COLEP][cer] is formulated as: τj[COLEP][cer]_ = max � _τ_ : _Qτ_ ( _{Sπ_ ˆ _jCOLEPδ_ ( _Xi,_ I[ _Yi_ = _j_ ]) _}i∈Ical_ ) _≤ Q_ 1 _−α_ ( _{Sπ_ ˆ _jCOLEP_ ( _Xi,_ I[ _Yi_ = _j_ ]) _}i∈Ical_ )� _._ (14) 

_Remarks._ Thm. 2 constructs a certified prediction set with 1 _− α_ coverage as eq. (11) by considering the worst-case perturbation during calibration to counter the influence of test-time adversaries. In contrast, Thm. 3 targets the case when we still adopt a standard calibration process without the consideration of test-time perturbations, which would lead to a lower coverage than 1 _− α_ and Thm. 3 concretely shows the formulation of the worst-case coverage. Thm. 3 presents the formulations of the lower bound of the coverage ( _τj_[COLEP][cer] ) with the standard prediction set as eq. (8) in the adversary setting. In addition to the certified coverage, we consider finite calibration set size and certified coverage with finite sample errors in Appendix G. 

6 

Published as a conference paper at ICLR 2024 

## 5 THEORETICAL ANALYSIS OF COLEP 

In Section 5.3, we theoretically show that COLEP achieves better coverage and prediction accuracy than a single main model as long as the utility of knowledge models and rules (characterized in Section 5.1) is non-trivial. To achieve it, we provide a connection between those utilities and the effectiveness of PCs in Section 5.2. The certification overview is provided in fig. 4 in Appendix E. 

5.1 CHARACTERIZATION OF MODELS AND KNOWLEDGE RULES We consider a mixture of benign distribution _Db_ and adversarial distribution _Da_ denoted by _Dm_ := _pDb Db_ + _pDa Da_ with _pDb_ + _pDa_ = 1. We map a set of implication rules encoded in the _r_ -th PC to an undirected graph _Gr_ = ( _Vr, Er_ ), where _Vr_ = [ _Nc_ + _L_ ] corresponds to _Nc_ (binary) class labels and _L_ (binary) concept labels. There exists an edge between node _j_ 1 _∈Vr_ and node _j_ 2 _∈Vr_ iff they are connected by an implication rule _j_ 1 = _⇒ j_ 2 or _j_ 2 = _⇒ j_ 1. 

Let _Ar_ ( _j∀_ ) be the set of nodes in the connected component of node _j∀_ in graph _Gr_ . Let _Ar,d_ ( _j∀_ ) _⊆ Ar_ ( _j∀_ ) be the set of conditional variables and _Ar,s_ ( _j∀_ ) _⊆Ar_ ( _j∀_ ) be the set of consequence variables. For any sample _x_ , let _ν_ ( _x_ ) _∈_ [0 _,_ 1] _[N][c]_[+] _[L]_ be the Boolean vector of ground truth class and concepts such that _ν_ ( _x_ ) := �[I[ _y_ = _j_ ]] _j∈_ [ _Nc_ ] _,_ [I[ _y_ ( _l_ )=1]] _l∈_ [ _L_ ]�. We denote _ν_ ( _x_ ) as _ν_ when sample _x_ is clear in the context. For simplicity, we assume a fixed weight _w ∈_ R[+] for all knowledge rules in the analysis. In what follows, we characterize the prediction capability of each model (main or knowledge) and establish a relation between these characteristics and the prediction performance of COLEP. 

For each class and concept probability estimated by learning models _π_ ˆ _j∀∈_ [ _Nc_ + _L_ ] : R _[d] �→_ [0 _,_ 1] (recall eq. (4)), we have the following definition: 

P _D_ [ˆ _πj∀_ ( _x_ ) _≤_ 1 _− tj∀,D |νj∀_ = 0] _≥ zj∀,D,_ P _D_ [ˆ _πj∀_ ( _x_ ) _≥ tj∀,D |νj∀_ = 1] _≥ zj∀,D_ (15) where _tj∀,D, zj∀,D_ are parameters quantifying quality of the main model and knowledge models in predicting correct class and concept, respectively. In particular, _tj∀,D_ characterizes the threshold of confidence whereas _zj∀,D_ quantifies the probability matching this threshold over data distribution _D_ . Specifically, models with a large _tj∀,D, zj∀,D_ imply that the model outputs a small class probability when the ground truth of input is 0 and a large class probability when the ground truth of input is 1, indicating the model performs well on prediction. We define the combined utility of models within _r_ -th PC by using individual model qualities. For _j_ -th class, we have: 

_Tj,_[(] _[r] D_[)][= Π] _[j] ∀[∈A] r_[(] _[j]_[)] _[t][j] ∀[,][D][,] Zj,_[(] _[r] D_[)][= Π] _[j] ∀[∈A] r_[(] _[j]_[)] _[z][j] ∀[,][D]_ (16) 

and the utility of associated knowledge rules within _r_ -th PC for the _j_ -th class as: 

P [ _νj∀_ = 0 _, ∃j∀ ∈Ar,s_ ( _j_ ) _|νj_ = 0] _, r ∈Pd_ ( _j_ ) _Uj_[(] _[r]_[)] = �P [ _νj∀_ = 1 _, ∃j∀ ∈Ar,d_ ( _j_ ) _|νj_ = 1] _, r ∈Ps_ ( _j_ ) (17) where _Pd_ ( _j_ ) and _Ps_ ( _j_ ) are PC index sets where _j_ appears as conditional and consequence variables. _Uj_[(] _[r]_[)] quantifies the utility of knowledge rules for class _j_ . Consider the case where _j_ corresponds to a conditional variable in the _r_ -th PC. Small _Uj_[(] _[r]_[)] indicates that P [ _νj∀_ = 1 _, ∀j∀ ∈Ar,s_ ( _j_ ) _|νj_ = 0] is high, hence implication rules for _j∀_ are universal and of low utility. 

5.2 EFFECTIVENESS OF THE REASONING COMPONENT WITH PCS Here we theoretically build the connection between the utility of knowledge models _Tj_[(] _∀[r] ,_[)] _D_[,] _[ Z] j_[(] _∀[r]_[)] _,D_[and] the utility of implication rules _Uj_[(] _∀[r]_[)][with the effectiveness of reasoning components.][We quantify the] effectiveness of the reasoning component as the relative confidence in ground truth class probabilities. Specifically, we expect E � _π_ ˆ _j_[COLEP] ( _X_ ) _− π_ ˆ _j_ ( _X_ ) _|Y_ = _j_ � to be large for _j ∈_ [ _Nc_ ] as it enables COLEP to correct the prediction of the main model, especially in the adversary setting. 

**Lemma 5.1** (Effectiveness of the Reasoning Component) **.** _For any class probability within models j ∈_ [ _Nc_ ] _and data sample_ ( _X, Y_ ) _drawn from the mixture distribution Dm, we can prove that:_ 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2024_COLEP_Certifiably_Robust_Learning-Reasoning_Conformal_Prediction_via_Probabilistic_Circuits_images/iclr_2024_COLEP_Certifiably_Robust_Learning-Reasoning_Conformal_Prediction_via_Probabilistic_Circuits.pdf-0007-12.png)


7 

Published as a conference paper at ICLR 2024 

_Remarks._ Lemma 5.1 analyzes the relative confidence of the ground truth class probabilities after corrections with the reasoning component. Towards that, we consider different cases based on the utility of the models and knowledge rules described earlier and combine conditional upper bounds with probabilities. We can quantify the effectiveness of the reasoning component with _ϵj,_ 0 _, ϵj,_ 1 since non-trivial _ϵj,_ 0 _, ϵj,_ 1 can correct the confidence of ground truth class in the expected direction and benefit conformal prediction by inducing a smaller non-conformity score for an adversarial sample. 

5.3 COMPARISON BETWEEN COLEP AND SINGLE MAIN MODEL 

Here we reveal the superiority of COLEP over a single main model in terms of the certified prediction coverage and prediction accuracy. Let _**ϵ** j,_ 0 _,D_ = E( _X,Y_ ) _∼D_ [ _ϵj,_ 0 _|νj_ = 0] and _**ϵ** j,_ 1 _,D_ = E( _X,Y_ ) _∼D_ [ _ϵj,_ 1 _|νj_ = 1]. In Section 5.2, we show that the utility of models and knowledge rules is crucial in forming an understanding of COLEPand _ϵj,_ 0 _>_ 0 _, ϵj,_ 1 _>_ 0 holds. In similar spirit, _**ϵ** j,_ 0 _,D >_ 0 and _**ϵ** j,_ 1 _,D >_ 0 for _j ∈_ [ _Nc_ ] and _D ∈{Da, Db}_ in the subsequent analysis. 

**Theorem 4** (Comparison of Marginal Coverage of COLEP and Main Model) **.** _Consider the adversary setting that the calibration set Ical consists of nDb samples drawn from the benign distribution Db, while the new sample_ ( _Xn_ +1 _, Yn_ +1) _is drawn nDa times from the adversarial distribution Da. Assume that A_ (ˆ _πj, Da_ ) _<_ 0 _._ 5 _< A_ (ˆ _πj, Db_ ) _for j ∈_ [ _Nc_ ] _, where A_ (ˆ _πj, D_ ) _is the expectation of prediction accuracy of π_ ˆ _j on D. Then we have:_ 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2024_COLEP_Certifiably_Robust_Learning-Reasoning_Conformal_Prediction_via_Probabilistic_Circuits_images/iclr_2024_COLEP_Certifiably_Robust_Learning-Reasoning_Conformal_Prediction_via_Probabilistic_Circuits.pdf-0008-05.png)


_where pj_ 0 = P _Db_ [I[ _Y ̸_ = _j_ ]] _and pj_ 1 = P _Db_ [I[ _Y_ = _j_ ]] _are class probabilities on benign distribution. Remarks._ Thm. 4 shows that COLEP can achieve better marginal coverage than a single model with a high probability exponentially approaching 1. The probability increases in particular with a higher quality of models represented by _**ϵ** j,_ 1 _,Da_ , _**ϵ** j,c,Db , A_ (ˆ _πj, Db_ ). _**ϵ** j,_ 1 _,Da_ , _**ϵ** j,c,Db_ quantifies the effectiveness of the correction of the reasoning component and is positively correlated with the utility of models _Tj,_[(] _[r] D_[)] _[, Z] j,_[(] _[r] D_[)][and the utility of knowledge rules] _[ U] j_[ (] _[r]_[)] as shown in Lemma 5.1, indicating that better knowledge models and rules benefits a higher prediction coverage with COLEP. The probability also increases with lower accuracy on the adversarial distribution _A_ (ˆ _πj, Da_ ), indicating COLEP improves marginal coverage more likely in a stronger adversary setting. **Theorem 5** (Comparison of Prediction Accuracy of COLEP and Main Model) **.** _Suppose that we_ ˆ ˆ _evaluate the expected prediction accuracy of πj[COLEP]_ ( _·_ ) _and πj_ ( _·_ ) _on n samples drawn from Dm and denote the prediction accuracy as A_ (ˆ _πj[COLEP]_ ( _·_ ) _, Dm_ ) _and A_ (ˆ _πj_ ( _·_ ) _, Dm_ ) _. Then we have: A_ (ˆ _πj[COLEP]_ ( _·_ ) _, Dm_ ) _≥ A_ (ˆ _πj_ ( _·_ ) _, Dm_ ) _, w.p._ 1 _−_ � _pD_ � P _D_ [ _Y_ = _j_ ] exp � _−_ 2 _n_ ( _**ϵ** j,c,D_ )[2][�] _._ (21) _D∈{Da,Db} c∈{_ 0 _,_ 1 _}_ 

_Remarks._ Thm. 5 shows that COLEP achieves better prediction accuracy than the main model with a high probability exponentially approaching 1. The probability increases with more effectiveness of the reasoning component quantified by a large _**ϵ** j,c,D_ , which positively correlates to the utility of models _Tj,_[(] _[r] D_[)] _[, Z] j,_[(] _[r] D_[)][and the utility of knowledge rules] _[ U] j_[ (] _[r]_[)] as shown in Lemma 5.1, indicating that better knowledge models and rules benefits a higher prediction accuracy with COLEP. In Appendix I, we further show that COLEP achieves higher prediction accuracy with more useful knowledge rules. 

## 6 EXPERIMENTS 

We evaluate COLEP on certified conformal prediction in the adversarial setting on various datasets, including GTSRB (Stallkamp et al., 2012), CIFAR-10, and AwA2 (Xian et al., 2018). For fair comparisons, we use the same model architecture and parameters in COLEP and baselines CP (Romano et al., 2020) and RSCP (Gendler et al., 2022). The desired coverage is set 0 _._ 9 across evaluations. Note that we leverage randomized smoothing for learning component certification (100k Monte-Carlo sampling) and consider the finite-sample errors of RSCP following Anonymous (2023) and that of COLEP following Thm. 6. We fix weights _w_ as 1 _._ 5 and provide more details in Appendix J. **Construction of the reasoning component** . In each PC, we encode a type of implication knowledge with disjoint attributes. In GTSRB, we have a PC of shape knowledge (e.g.,“octagon”, “square”), a PC of boundary color knowledge (e.g., “red boundary”, “black boundary”), and a PC of content knowledge (e.g., “digit 50”, “turn left”). Specifically, in the PC of shape knowledge, we can encode the implication rules such as IsStopSign = _⇒_ IsOctagon (stop signs are octagon), IsSpeedLimit = _⇒_ IsSquare (speed limit signs are square). In summary, we have 3 PCs and 28 knowledge rules in 

8 

Published as a conference paper at ICLR 2024 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2024_COLEP_Certifiably_Robust_Learning-Reasoning_Conformal_Prediction_via_Probabilistic_Circuits_images/iclr_2024_COLEP_Certifiably_Robust_Learning-Reasoning_Conformal_Prediction_via_Probabilistic_Circuits.pdf-0009-01.png)


**----- Start of picture text -----**<br>
0.95 RSCP COLEP Upper Bnd. of 𝜏("#$) 0.95 RSCP COLEP Upper Bnd. of 𝜏("#$) 0.95 RSCP COLEP Upper Bnd. of 𝜏("#$)<br>0.9 0.9 0.9<br>0.85 0.85 0.85<br>0.8 0.8 0.8<br>0.75 0.75 0.75<br>0.7 0.7 0.7<br>0.65 0.65 0.65<br>𝛿= 0.125 𝛿= 0.25 𝛿= 0.50 𝛿= 0.125 𝛿= 0.25 𝛿= 0.50 𝛿= 0.125 𝛿= 0.25 𝛿= 0.50<br>(a) GTSRB (b) CIFAR-10 (c) AwA2<br>Figure 2: Comparison of certified coverage between COLEP ( τ [COLEP][cer] ) and RSCP under bounded perturbations<br>δ = 0 . 125 ,  0 . 25 ,  0 . 50 on GTSRB, CIFAR-10, and AwA2. The upper bound of certified coverage  τ (cer) is 0.9.<br>Marginal Coverage Averaged Set Size Marginal Coverage Averaged Set Size Marginal Coverage Averaged Set Size<br>1 3 1 4 1 7<br>0.98 2.5 0.95 3.5 0.98 6<br>0.96 3 0.96 5<br>0.94 2 0.9 2.5 0.94<br>0.92 0.92 4<br>1.5 0.85 2<br>0.9 0.9 3<br>0.88 1 0.8 1.5 0.88<br>0.86 1 0.86 2<br>0.84 0.5 0.75 0.5 0.84 1<br>0.82 0 0.7 0 0.82 0<br>CP RSCP COLEP CP RSCP COLEP CP RSCP COLEP CP RSCP COLEP CP RSCP COLEP CP RSCP COLEP<br>(a) GTSRB (b) CIFAR-10 (c) AwA2<br>Certified Coverage<br>**----- End of picture text -----**<br>


Figure 3: Comparison of the marginal coverage and averaged set size for CP, RSCP, and COLEP under PGD attack ( _δ_ = 0 _._ 25) on GTSRB, CIFAR-10, and AwA2. The nominal coverage level (green line) is 0.9. 

GTSRB, 3 PCs and 30 knowledge rules in CIFAR-10, and 4 PCs and 187 knowledge rules in AwA2. We also provide detailed steps of PC construction in appendix J.1. 

**Certified Coverage under Bounded Perturbations.** We evaluate the certified coverage of COLEP given a new test sample _Xn_ +1 with bounded perturbation _ϵ_ ( _|ϵ|_ 2 _< δ_ ) based on our certification in Thm. 3. Certified coverage indicates the worst-case coverage during inference in the adversary setting, hence a higher certified coverage indicates a more robust conformal prediction and tigher certification. We compare the certified coverage of COLEP with the SOTA baseline RSCP (Theorem 2 in (Gendler et al., 2022)). The comparison of certified coverage between COLEP and RSCP is provided in Figure 2. The results indicate that COLEP consistently achieves higher certified coverage under different _ℓ_ 2 norm bounded perturbations, and COLEP outperforms RSCP by a large margin under a large perturbation radius _δ_ = 0 _._ 5. Note that an obvious upper bound of the certified coverage in the adversary setting is the guaranteed coverage without an adversary, which is 0 _._ 9 in the evaluation. The closeness between the certified coverage of COLEP _τ_ (cer)[COLEP] and the upper bound of coverage guarantee _τ_ (cer) shows the robustness of COLEP and tightness of our certification. 

**Prediction Coverage and Prediction Set Size under Adversarial Attacks.** We also evaluate the marginal coverage and averaged set size of COLEP and baselines under adversarial attacks. COLEP constructs the certifiably robust prediction set following Thm. 2. We compare the results with the standard conformal prediction (CP) and the SOTA conformal prediction with randomized smoothing (RSCP). For fair comparisons, we apply PGD attack (Madry et al., 2018) with the same parameters on CP, RSCP, and COLEP. For COLEP, we consider the adaptive PGD attack against the complete learning-reasoning pipeline. The comparison of the marginal coverage and averaged set size for CP, RSCP, and COLEP under PGD attack ( _δ_ = 0 _._ 25) is provided in Figure 3. The results indicate that the marginal coverage of CP is below the nominal coverage level 0 _._ 9 under PGD attacks as data exchangeability is violated, while COLEP still achieves higher marginal coverage than the nominal level, validating the robustness of COLEP for conformal prediction in the adversary setting. Compared with RSCP, COLEP achieves both larger marginal coverage and smaller set size, demonstrating that COLEP maintains the guaranteed coverage with less inflation of the prediction set. The observation validates our theoretical analysis in Thm. 4 that COLEP can achieve better coverage than a single model with the power of kwowledge-enabled logical reasoning. We provide more evaluations of different coverage levels 1 _− α_ , different perturbation bound _δ_ , different conformal prediction baselines, and contributions of different knowledge rules in Appendix J.2. 

**Conclusion** In this paper, we present COLEP, a certifiably robust conformal prediction framework via knowledge-enabled logical reasoning. We leverage PCs for efficient reasoning and provide robustness certification. We also provide end-to-end certification for finite-sample certified coverage in the presence of adversaries and theoretically prove the advantage of COLEP over a single model. We included the discussions of limitations, broader impact, and future work in Appendix A. 

9 

Published as a conference paper at ICLR 2024 

## ACKNOWLEDGMENTS 

This work is partially supported by the National Science Foundation under grant No. 1910100, No. 2046726, No. 2229876, DARPA GARD, the National Aeronautics and Space Administration (NASA) under grant No. 80NSSC20M0229, the Alfred P. Sloan Fellowship, the Amazon research award, and the eBay research award. 

## ETHICS STATEMENT 

We do not see potential ethical issues about COLEP. In contrast, with the power of logical reasoning, COLEP is a certifiably robust conformal prediction framework against adversaries during the inference time. 

## REPRODUCIBILITY STATEMENT 

The reproducibility of COLEP span theoretical and experimental perspectives. We provide complete proofs of all the theoretical results in appendices. We provide the source codes for implementing COLEP at https://github.com/kangmintong/COLEP. 

## REFERENCES 

Milton Abramowitz and Irene A Stegun. _Handbook of mathematical functions with formulas, graphs, and mathematical tables_ , volume 55. US Government printing office, 1948. 

- Kareem Ahmed, Stefano Teso, Kai-Wei Chang, Guy Van den Broeck, and Antonio Vergari. Semantic probabilistic layers for neuro-symbolic learning. _Advances in Neural Information Processing Systems_ , 35:29944–29959, 2022. 

- Anonymous. Provably robust conformal prediction with improved efficiency. In _Submitted to The Twelfth International Conference on Learning Representations_ , 2023. URL https:// openreview.net/forum?id=BWAhEjXjeG. under review. 

- Mislav Balunovic, Maximilian Baader, Gagandeep Singh, Timon Gehr, and Martin Vechev. Certifying geometric robustness of neural networks. In _Advances in Neural Information Processing Systems_ , pp. 15287–15297, 2019. 

- Rina Foygel Barber, Emmanuel J Candes, Aaditya Ramdas, and Ryan J Tibshirani. Conformal prediction beyond exchangeability. _arXiv preprint arXiv:2202.13415_ , 2022. 

- Frank Markham Brown. _Boolean reasoning: the logic of Boolean equations_ . Courier Corporation, 2003. 

- Daniel D Caminhas. Detecting and correcting typing errors in open-domain knowledge graphs using semantic representation of entities. 2019. 

- Yulong Cao, Ningfei Wang, Chaowei Xiao, Dawei Yang, Jin Fang, Ruigang Yang, Qi Alfred Chen, Mingyan Liu, and Bo Li. Invisible for both camera and lidar: Security of multi-sensor fusion based perception in autonomous driving under physical-world attacks. In _2021 IEEE Symposium on Security and Privacy (SP)_ , pp. 176–194. IEEE, 2021. 

- Jiaoyan Chen, Xi Chen, Ian Horrocks, Erik B. Myklebust, and Ernesto Jimenez-Ruiz. Correcting knowledge base assertions. In _Proceedings of the Web Conference 2020_ , pp. 1537–1547, 2020a. 

- Qingrong Chen, Chong Xiang, Minhui Xue, Bo Li, Nikita Borisov, Dali Kaarfar, and Haojin Zhu. Differentially private data generative models. _arXiv preprint arXiv:1812.02274_ , 2018. 

- Xiaojun Chen, Shengbin Jia, and Yang Xiang. A review: Knowledge reasoning over knowledge graph. _Expert Systems with Applications_ , 141:112948, 2020b. 

- Arthur Choi and Adnan Darwiche. On relaxing determinism in arithmetic circuits. In _International Conference on Machine Learning_ , pp. 825–833. PMLR, 2017. 

10 

Published as a conference paper at ICLR 2024 

Jeremy Cohen, Elan Rosenfeld, and Zico Kolter. Certified adversarial robustness via randomized smoothing. In _international conference on machine learning_ , pp. 1310–1320. PMLR, 2019. 

- Francesco Croce and Matthias Hein. Reliable evaluation of adversarial robustness with an ensemble of diverse parameter-free attacks. In _International conference on machine learning_ , pp. 2206–2216. PMLR, 2020. 

- Adnan Darwiche. Compiling knowledge into decomposable negation normal form. In _IJCAI_ , volume 99, pp. 284–289. Citeseer, 1999. 

- Adnan Darwiche. Decomposable negation normal form. _Journal of the ACM (JACM)_ , 48(4):608–647, 2001. 

Adnan Darwiche. A logical approach to factoring belief networks. _KR_ , 2:409–420, 2002. 

Adnan Darwiche. A differential approach to inference in bayesian networks. _Journal of the ACM (JACM)_ , 50(3):280–305, 2003. 

- Kevin Eykholt, Ivan Evtimov, Earlence Fernandes, Bo Li, Amir Rahmati, Chaowei Xiao, Atul Prakash, Tadayoshi Kohno, and Dawn Song. Robust physical-world attacks on deep learning visual classification. In _Proceedings of the IEEE conference on computer vision and pattern recognition_ , pp. 1625–1634, 2018. 

- Artur d’Avila Garcez, Sebastian Bader, Howard Bowman, Luis C Lamb, Leo de Penning, BV Illuminoo, Hoifung Poon, and COPPE Gerson Zaverucha. Neural-symbolic learning and reasoning: A survey and interpretation. _Neuro-Symbolic Artificial Intelligence: The State of the Art_ , 342(1):327, 2022. 

- Asaf Gendler, Tsui-Wei Weng, Luca Daniel, and Yaniv Romano. Adversarially robust conformal prediction. In _International Conference on Learning Representations_ , 2022. URL https: //openreview.net/forum?id=9L1BsI4wP1H. 

- Subhankar Ghosh, Yuanjie Shi, Taha Belkhouja, Yan Yan, Jana Doppa, and Brian Jones. Probabilistically robust conformal prediction. In _Uncertainty in Artificial Intelligence_ , pp. 681–690. PMLR, 2023. 

- Ian J Goodfellow, Jonathon Shlens, and Christian Szegedy. Explaining and harnessing adversarial examples. _arXiv preprint arXiv:1412.6572_ , 2014. 

- Sven Gowal, Krishnamurthy Dvijotham, Robert Stanforth, Rudy Bunel, Chongli Qin, Jonathan Uesato, Relja Arandjelovic, Timothy Mann, and Pushmeet Kohli. On the effectiveness of interval bound propagation for training verifiably robust models. _arXiv preprint arXiv:1810.12715_ , 2018. 

- Nezihe Merve Gürel, Xiangyu Qi, Luka Rimanic, Ce Zhang, and Bo Li. Knowledge-enhanced machine learning pipeline against diverse adversarial attacks. In _International Conference on Machine Learning_ , 2021. 

- Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun. Deep residual learning for image recognition. In _Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR)_ , June 2016. 

- P Hitzler and MK Sarker. Tractable boolean and arithmetic circuits. _Neuro-Symbolic Artificial Intelligence: The State of the Art_ , 342:146, 2022. 

- Ying Jin, Zhimei Ren, and Emmanuel J Candès. Sensitivity analysis of individual treatment effects: A robust conformal inference approach. _Proceedings of the National Academy of Sciences_ , 120(6): e2214889120, 2023. 

- Mintong Kang, Linyi Li, Maurice Weber, Yang Liu, Ce Zhang, and Bo Li. Certifying some distributional fairness with subpopulation decomposition. _Advances in Neural Information Processing Systems_ , 35:31045–31058, 2022. 

- Mintong Kang, Bowen Li, Zengle Zhu, Yongyi Lu, Elliot K Fishman, Alan Yuille, and Zongwei Zhou. Label-assemble: Leveraging multiple datasets with partial labels. In _2023 IEEE 20th International Symposium on Biomedical Imaging (ISBI)_ , pp. 1–5. IEEE, 2023a. 

11 

Published as a conference paper at ICLR 2024 

Mintong Kang, Linyi Li, and Bo Li. Fashapley: Fast and approximated shapley based model pruning towards certifiably robust dnns. In _2023 IEEE Conference on Secure and Trustworthy Machine Learning (SaTML)_ , pp. 575–592. IEEE, 2023b. 

- Mintong Kang, Zhen Lin, Jimeng Sun, Cao Xiao, and Bo Li. Certifiably byzantine-robust federated conformal prediction. 2023c. 

- Mintong Kang, Nezihe Merve Gürel, Ning Yu, Dawn Song, and Bo Li. C-rag: Certified generation risks for retrieval-augmented language models. _arXiv preprint arXiv:2402.03181_ , 2024a. 

Mintong Kang, Dawn Song, and Bo Li. Diffattack: Evasion attacks against diffusion-based adversarial purification. _Advances in Neural Information Processing Systems_ , 36, 2024b. 

- Doga Kisa, Guy Van den Broeck, Arthur Choi, and Adnan Darwiche. Probabilistic sentential decision diagrams. In _Proceedings of the 14th international conference on principles of knowledge representation and reasoning (KR)_ , pp. 1–10, 2014. 

- Aounon Kumar, Alexander Levine, and Soheil Feizi. Policy smoothing for provably robust reinforcement learning. In _International Conference on Learning Representations_ , 2022. 

- Mathias Lecuyer, Vaggelis Atlidakis, Roxana Geambasu, Daniel Hsu, and Suman Jana. Certified robustness to adversarial examples with differential privacy. In _2019 IEEE Symposium on Security and Privacy (SP)_ , pp. 656–672. IEEE, 2019. 

Jing Lei, James Robins, and Larry Wasserman. Distribution-free prediction sets. _Journal of the American Statistical Association_ , 108(501):278–287, 2013. 

- Jing Lei, Max G’Sell, Alessandro Rinaldo, Ryan J Tibshirani, and Larry Wasserman. Distribution-free predictive inference for regression. _Journal of the American Statistical Association_ , 113(523): 1094–1111, 2018. 

- Bo Li, Kevin Roundy, Chris Gates, and Yevgeniy Vorobeychik. Large-scale identification of malicious singleton files. In _Proceedings of the seventh ACM on conference on data and application security and privacy_ , pp. 227–238, 2017. 

- Linyi Li, Zexuan Zhong, Bo Li, and Tao Xie. Robustra: Training provable robust neural networks over reference adversarial space. In _IJCAI_ , pp. 4711–4717, 2019. 

- Linyi Li, Maurice Weber, Xiaojun Xu, Luka Rimanic, Bhavya Kailkhura, Tao Xie, Ce Zhang, and Bo Li. TSS: Transformation-specific smoothing for robustness certification. In _Proceedings of the 2021 ACM SIGSAC Conference on Computer and Communications Security_ , pp. 535–557, 2021. ISBN 9781450384544. 

- Linyi Li, Jiawei Zhang, Tao Xie, and Bo Li. Double sampling randomized smoothing. In _International Conference on Machine Learning_ , 2022a. 

- Linyi Li, Tao Xie, and Bo Li. Sok: Certified robustness for deep neural networks. In _44th IEEE Symposium on Security and Privacy, SP 2023, San Francisco, CA, USA, 22-26 May 2023_ . IEEE, 2023. 

- Yujia Li, David Choi, Junyoung Chung, Nate Kushman, Julian Schrittwieser, Rémi Leblond, Tom Eccles, James Keeling, Felix Gimeno, Agustin Dal Lago, et al. Competition-level code generation with alphacode. _Science_ , 378(6624):1092–1097, 2022b. 

- Aleksander Madry, Aleksandar Makelov, Ludwig Schmidt, Dimitris Tsipras, and Adrian Vladu. Towards deep learning models resistant to adversarial attacks. In _International Conference on Learning Representations_ , 2018. URL https://openreview.net/forum?id=rJzIBfZAb. 

- Robin Manhaeve, Sebastijan Dumancic, Angelika Kimmig, Thomas Demeester, and Luc De Raedt. Deepproblog: Neural probabilistic logic programming. _Advances in neural information processing systems_ , 31, 2018. 

- Pascal Massart. The tight constant in the dvoretzky-kiefer-wolfowitz inequality. _The annals of Probability_ , pp. 1269–1283, 1990. 

12 

Published as a conference paper at ICLR 2024 

- André Melo and Heiko Paulheim. An approach to correction of erroneous links in knowledge graphs. In _CEUR Workshop Proceedings_ , volume 2065, pp. 54–57. RWTH Aachen, 2017. 

- Xinlei Pan, Chaowei Xiao, Warren He, Shuang Yang, Jian Peng, Mingjie Sun, Jinfeng Yi, Zijiang Yang, Mingyan Liu, Bo Li, et al. Characterizing attacks on deep reinforcement learning. _AISTATS_ , 2019. 

- Bhaskar Ray Chaudhury, Linyi Li, Mintong Kang, Bo Li, and Ruta Mehta. Fairness in federated learning via core-stability. _Advances in neural information processing systems_ , 35:5738–5750, 2022. 

- Matthew Richardson and Pedro Domingos. Markov logic networks. _Machine learning_ , 62:107–136, 2006. 

- Yaniv Romano, Matteo Sesia, and Emmanuel Candes. Classification with valid and adaptive coverage. In H. Larochelle, M. Ranzato, R. Hadsell, M.F. Balcan, and H. Lin (eds.), _Advances in Neural Information Processing Systems_ , volume 33, pp. 3581–3591. Curran Associates, Inc., 2020. URL https://proceedings.neurips.cc/paper/2020/file/ 244edd7e85dc81602b7615cd705545f5-Paper.pdf. 

- Amirmohammad Rooshenas and Daniel Lowd. Learning sum-product networks with direct and indirect variable interactions. In Eric P. Xing and Tony Jebara (eds.), _Proceedings of the 31st International Conference on Machine Learning_ , volume 32 of _Proceedings of Machine Learning Research_ , pp. 710–718, Bejing, China, 22–24 Jun 2014. PMLR. URL https://proceedings. mlr.press/v32/rooshenas14.html. 

- Hadi Salman, Jerry Li, Ilya Razenshteyn, Pengchuan Zhang, Huan Zhang, Sebastien Bubeck, and Greg Yang. Provably robust deep learning via adversarially trained smoothed classifiers. In _Advances in Neural Information Processing Systems_ , pp. 11289–11300, 2019. 

- Glenn Shafer and Vladimir Vovk. A tutorial on conformal prediction. _Journal of Machine Learning Research_ , 9(3), 2008. 

- Sahil Singla, Surbhi Singla, and Soheil Feizi. Improved deterministic l2 robustness on CIFAR-10 and CIFAR-100. In _International Conference on Learning Representations_ , 2022. 

- Aldo Solari and Vera Djordjilovi´c. Multi split conformal prediction. _Statistics & Probability Letters_ , 184:109395, 2022. 

- Johannes Stallkamp, Marc Schlipsing, Jan Salmen, and Christian Igel. Man vs. computer: Benchmarking machine learning algorithms for traffic sign recognition. _Neural networks_ , 32:323–332, 2012. 

- Christian Szegedy, Wojciech Zaremba, Ilya Sutskever, Joan Bruna, Dumitru Erhan, Ian J. Goodfellow, and Rob Fergus. Intriguing properties of neural networks. In Yoshua Bengio and Yann LeCun (eds.), _2nd International Conference on Learning Representations, ICLR 2014, Banff, AB, Canada, April 14-16, 2014, Conference Track Proceedings_ , 2014. 

- Yusuke Tsuzuku, Issei Sato, and Masashi Sugiyama. Lipschitz-margin training: scalable certification of perturbation invariance for deep neural networks. In _Advances in Neural Information Processing Systems_ , 2018. 

- Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez, Łukasz Kaiser, and Illia Polosukhin. Attention is all you need. _Advances in neural information processing systems_ , 30, 2017. 

- Vladimir Vovk, Alexander Gammerman, and Glenn Shafer. _Algorithmic learning in a random world_ , volume 29. Springer, 2005. 

- Volodya Vovk, Alexander Gammerman, and Craig Saunders. Machine-learning applications of algorithmic randomness. 1999. 

13 

Published as a conference paper at ICLR 2024 

Boxin Wang, Weixin Chen, Hengzhi Pei, Chulin Xie, Mintong Kang, Chenhui Zhang, Chejian Xu, Zidi Xiong, Ritik Dutta, Rylan Schaeffer, et al. Decodingtrust: A comprehensive assessment of trustworthiness in gpt models. _arXiv preprint arXiv:2306.11698_ , 2023. 

- Eric Wong and Zico Kolter. Provable defenses against adversarial examples via the convex outer adversarial polytope. In _International Conference on Machine Learning_ , pp. 5286–5295, 2018. 

- Fan Wu, Linyi Li, Zijian Huang, Yevgeniy Vorobeychik, Ding Zhao, and Bo Li. CROP: Certifying robust policies for reinforcement learning through functional smoothing. In _International Conference on Learning Representations_ , 2022a. 

- Fan Wu, Linyi Li, Chejian Xu, Huan Zhang, Bhavya Kailkhura, Krishnaram Kenthapadi, Ding Zhao, and Bo Li. Copa: Certifying robust policies for offline reinforcement learning against poisoning attacks. In _International Conference on Learning Representations_ , 2022b. 

- Yongqin Xian, Christoph H Lampert, Bernt Schiele, and Zeynep Akata. Zero-shot learning—a comprehensive evaluation of the good, the bad and the ugly. _IEEE transactions on pattern analysis and machine intelligence_ , 41(9):2251–2265, 2018. 

- Chaowei Xiao, Jun Yan Zhu, Bo Li, Warren He, Mingyan Liu, and Dawn Song. Spatially transformed adversarial examples. In _6th International Conference on Learning Representations, ICLR 2018_ , 2018. 

- Chulin Xie, Minghao Chen, Pin-Yu Chen, and Bo Li. Crfl: Certifiably robust federated learning against backdoor attacks. In _International Conference on Machine Learning_ , pp. 11372–11382. PMLR, 2021. 

- Chulin Xie, Yunhui Long, Pin-Yu Chen, and Bo Li. Uncovering the connection between differential privacy and certified robustness of federated learning against poisoning attacks. _arXiv preprint arXiv:2209.04030_ , 2022. 

- Chejian Xu, Wenhao Ding, Weijie Lyu, Zuxin Liu, Shuai Wang, Yihan He, Hanjiang Hu, Ding Zhao, and Bo Li. Safebench: A benchmarking platform for safety evaluation of autonomous vehicles. _Advances in Neural Information Processing Systems_ , 35:25667–25682, 2022a. 

- Jingyi Xu, Zilu Zhang, Tal Friedman, Yitao Liang, and Guy Broeck. A semantic loss function for deep learning with symbolic knowledge. In _International conference on machine learning_ , pp. 5502–5511. PMLR, 2018. 

- Xiaojun Xu, Linyi Li, and Bo Li. Lot: Layer-wise orthogonal training on improving l2 certified robustness. In _Advances in Neural Information Processing Systems_ , 2022b. 

- Yachong Yang and Arun Kumar Kuchibhotla. Finite-sample efficient conformal prediction. _arXiv preprint arXiv:2104.13871_ , 2021. 

- Zhuolin Yang, Zhikuan Zhao, Boxin Wang, Jiawei Zhang, Linyi Li, Hengzhi Pei, Bojan Karlaš, Ji Liu, Heng Guo, Ce Zhang, and Bo Li. Improving certified robustness via statistical learning with logical reasoning. In _Advances in Neural Information Processing Systems 35 (NeurIPS 2022)_ , 2022. 

- Kexin Yi, Jiajun Wu, Chuang Gan, Antonio Torralba, Pushmeet Kohli, and Josh Tenenbaum. Neuralsymbolic vqa: Disentangling reasoning from vision and language understanding. _Advances in neural information processing systems_ , 31, 2018. 

- Huan Zhang, Tsui-Wei Weng, Pin-Yu Chen, Cho-Jui Hsieh, and Luca Daniel. Efficient neural network robustness certification with general activation functions. In _Advances in neural information processing systems_ , pp. 4939–4948, 2018. 

- Jiawei Zhang, Linyi Li, Ce Zhang, and Bo Li. CARE: Certifiably robust learning with reasoning via variational inference. In _First IEEE Conference on Secure and Trustworthy Machine Learning_ , 2023. URL https://openreview.net/forum?id=1n6oWTTV1n. 

14 

Published as a conference paper at ICLR 2024 

## **Appendix** 

|**Appendix**|**Appendix**|||
|---|---|---|---|
|**Contents**||||
|**A **|**Discussions, Broader Impact, Limitations, and Future Work**||**16**|
|**B**|**More related work**||**17**|
|**C **|**Preliminaries**||**18**|
||C.1<br>Probabilistic Circuits|. . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .|18|
||C.2<br>Multiple PCs with coeffcients_βr_ in the combination of PCs<br>. . . . . . . . . . . .||18|
||C.3<br>Randomized Smoothing . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .||19|
|**D **|**Conformal Prediction with COLEP**||**19**|
||D.1<br>Flexibility of user-defned coverage level . . . . . . . . . . . . . . . . . . . . . . .||19|
||D.2<br>Illustrative example .|. . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .|20|
|**E**|**Overview of Certifcation**||**20**|
|**F**|**Omitted Proofs in Section**|**4**|**20**|
||F.1<br>Proof of Thm. 1 . . .|. . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .|20|
||F.2<br>Proof of Thm. 2 . . .|. . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .|23|
||F.3<br>Thm. 2 with fnite-sample errors by randomized smoothing . . . . . . . . . . . . .||24|
||F.4<br>Proof of Thm. 3 . . .|. . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .|26|
|**G **|**Finite-Sample Certifed Coverage**||**27**|
|**H **|**Omitted Proofs in Section**|**5**|**28**|
||H.1<br>Proof of Lemma 5.1 .|. . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .|28|
||H.2<br>Proof of Thm. 4 . . .|. . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .|32|
||H.3<br>Proof of Thm. 5 . . .|. . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .|34|
|**I**|**Lower Bound of**_A_(ˆ_π_**COLEP**<br>_j_|_, Dm_)|**34**|
|**J**|**Experiment**||**37**|
||J.1<br>Implementation Details . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .||37|
||J.2<br>Additional evaluation|ofCOLEP . . . . . . . . . . . . . . . . . . . . . . . . . . .|38|
|**K **|**Overview of notations**||**40**|



15 

Published as a conference paper at ICLR 2024 

## A DISCUSSIONS, BROADER IMPACT, LIMITATIONS, AND FUTURE WORK 

**Broader Impact.** In this paper, we propose a certifiably robust conformal prediction framework COLEP via knowledge-enabled logic reasoning. We prove that for any adversarial test sample with _ℓ_ 2-norm bounded perturbations, the prediction set of COLEP captures the ground truth label with a guaranteed coverage level. Considering distribution shifts during inference time and weakening the assumption of data exchangeability has been an active area of conformal prediction. COLEP demonstrates both theoretical and empirical improvements in the adversarial setting and can inspire future work to explore various types of distribution shifts with the power of logical reasoning and the certification framework of COLEP. Furthermore, robustness certification for conformal prediction can be viewed as the generalization of certified prediction robustness, which justifies a certifiably robust prediction rather than a prediction set. Considering the challenges of providing tight bounds for certified prediction robustness, we expect that COLEP can motivate the certification of prediction robustness to benefit from a statistical perspective. 

**Limitations.** A possible limitation may lie in the computational costs induced by pretraining knowledge models. The problem can be partially relieved by training knowledge models with parallelism. Moreover, we only need to pretrain them once and then can encode different types of knowledge rules based on domain knolwedge. The training cost is worthy since we theoretically and empirically show that more knowledge models benefit the robustness of COLEP a lot for conformal prediction. Another possible limitation may lie in the access to knowledge rules. For the datasets that release hierarchical information (e.g., ImageNet, CIFAR-100, Visual Genome), we can easily adapt the information to implication rules used in COLEP. We can also seek knowledge graphs (Chen et al., 2020b) with related label semantics for rule design. These approaches may be effective in assisting practitioners in designing knowledge rules, but the process is not fully automatic, and human efforts are still needed. Overall, for any knowledge system, one naturally needs domain experts to design the knowledge rules specific to that application. There is probably no universal strategy on how to aggregate knowledge for any arbitrary application, and rather application-specific constructions are needed. This is where the power as well as limitation of our framework comes from. 

**Future work.** The following directions are potentially interesting future work based on the framework of COLEP: 1) exploring different structures of knowledge representations and their effectiveness on conformal prediction performance, 2) proposing a more systematic way of designing logic rules based on domain knowledge, and 3) initiating a better understanding of the importance of different types of knowledge rules and motivating better design of rule weights accordingly. 

**Discussions.** We will consider the cases where the knowledge graph is contaminated by adversarial attackers or contains noises and misinformation in this part. The robustness of COLEP mainly originates from the knowledge rules via factor function _F_ ( _·_ ) denoting how well each assignment conforms to the specified knowledge rules. Since the knowledge rules and weights are specified by practitioners and fixed during inferences, the PCs encoding the knowledge rules together with the main model and knowledge models (learning + reasoning component in Figure 1) can be viewed as a complete model. According to the adversarial attack settings in the literature Goodfellow et al. (2014); Gendler et al. (2022), the model weights can be well protected in practical cases and the attackers can only manipulate the inputs but not the model, and thus, the PC graph (i.e., the source of side information) which is a part of the augmented model can not be manipulated by adversarial attacks. On the other hand, it is an interesting question to discuss the case when the knowledge rules may be contaminated and contain misinformation due to the neglect of designers. We can view the problem of checking knowledge rules and correcting misinformation as a pre-processing step before the employment of COLEP. The parallel module of knowledge checking and correction is well-explored by a line of research Melo & Paulheim (2017); Caminhas (2019); Chen et al. (2020a), which basically detects and corrects false logical relations in the semantic embedding space via consistency-checking techniques. It is also interesting to leverage more advanced large language models to check and correct human-designed knowledge rules. We deem that the process of knowledge-checking is independent and parallel to COLEP, which may take additional efforts before the deployment of COLEP but significantly benefits in achieving a much more robust system based on our analysis and evaluations. 

**Alternative certification** . There are two general certification directions for certifiably robust conformal prediction: (1) plugging in the standard score for for test example, and the quantile is computed 

16 

Published as a conference paper at ICLR 2024 

on the worst-case scores for each calibration point as in our paper; (2) plugging in the worst-case score for the test example and the standard score for each calibration point. We can also adapt the results with the second line based on the symmetry between _x_ and _x[′]_ is indeed an interesting point. Specifically, given perturbed _x[′]_ during test time, we can compute the bounds of output logits in the _δ_ -ball, which will also cover the clean _x_ . Since the logit bounds also hold for clean _x_ , we can compute the worst-case scores based on the bounds and finally construct the prediction set with the clean quantile value. The two lines of certification are essentially parallel. It is fundamentally due to the exchangeability of clean samples, which leads to consideration of worst-case bounds either during calibration or inference. One practical advantage of the first line of certification might be that the computation of worst-case bounds requires more runtime, and thus, computing it during the offline calibration process might be more desirable than during the online inference process. 

**Discussions of selection of rule weights** . If a knowledge rule is true with a lower probability, then the associated weight rule should be smaller, and vice versa. For example, there might be a case that a knowledge rule only holds with probability 70%. In the example, the probability is with respect to the data distribution, which means that 70% of data satisfies the rule “IsStopSign _→_ IsOctogan”. Besides, other factors such as the portions of class labels also affect the importance of corresponding rules. If one class label is pretty rare in the data distribution, then the knowledge rules associated with it clearly should be assigned a small weight, and vice versa. In summary, the importance of knowledge rules depends on the data distribution, no matter whether we select it via grid search or optimize it in a more subtle way. Therefore, we deem that a general guideline for an effective and efficient selection of rule weights is to perform a grid search on a held-out validation set, which is also essential for standard model training. However, we do not think the effectiveness of COLEP is quite sensitive to the selection of weights. From the theoretical analysis and evaluations, we can expect benefits from the knowledge rules as long as we have a positive weight _w_ for the rules. 

## B MORE RELATED WORK 

**Certified Prediction Robustness** To provide the worst-case robustness certification, a rich body of work has been proposed, including convex relaxations (Zhang et al., 2018; Li et al., 2019; Kang et al., 2023b), Lipschitz-bounded model layers (Tsuzuku et al., 2018; Singla et al., 2022; Xu et al., 2022b), and randomized smoothing (Cohen et al., 2019; Salman et al., 2019; Li et al., 2021; 2022a). However, existing work mainly focuses on certifying the _prediction robustness_ rather than the _prediction coverage_ , which is demonstrated to provide essential uncertainty quantification in conformal prediction (Jin et al., 2023). In this work, we aim to bridge worst-cast robustness certification and prediction coverage, and design learning frameworks to improve the uncertainty certification. 

**Knowledged-Enabled Logical Reasoning** Neural-symbolic methods (Ahmed et al., 2022; Garcez et al., 2022; Yi et al., 2018) use symbols to represent objects, encode relationships with logical rules, and perform logical inference on them. DeepProbLog (Manhaeve et al., 2018) and Semantic loss (Xu et al., 2018) are representatives that leverage logic circuits Brown (2003) with weighted mode counting (WMC) to integrate symbolic knowledge. _Probabilistic circuits (PCs)_ (Darwiche, 2002; 2003; Kisa et al., 2014) are another family of probabilistic models which allow exact and efficient inference. PCs permit various types of inference to be performed in linear time of the size of circuits (Hitzler & Sarker, 2022; Choi & Darwiche, 2017; Rooshenas & Lowd, 2014). 

**Worst-case Certification** . In particular, considering potential adversarial manipulations during test time, where a small perturbation could mislead the models to make incorrect predictions (Szegedy et al., 2014; Madry et al., 2018; Xiao et al., 2018; Cao et al., 2021), different robustness certification approaches have been explored to provide the worst-case prediction guarantees for neural networks. For instance, a range of deterministic and statistical _worst-case certifications_ have been proposed for a single model (Wong & Kolter, 2018; Zhang et al., 2018; Cohen et al., 2019; Lecuyer et al., 2019), reinforcement learning (Wu et al., 2022a; Kumar et al., 2022; Wu et al., 2022b), federated learning paradigms (Xie et al., 2021; 2022), and fair learning (Kang et al., 2022; Ray Chaudhury et al., 2022), under constrained perturbations (Balunovic et al., 2019; Li et al., 2021; 2022a). 

17 

Published as a conference paper at ICLR 2024 

## C PRELIMINARIES 

## C.1 PROBABILISTIC CIRCUITS 

**Definition 1** (Probabilistic Circuit) **.** A probabilistic circuit (PC) _P_ defines a joint distribution over the set of random variables **X** and can be defined as a tuple ( _G, ψ_ ), where _G_ represents an acyclic directed graph ( _V, E_ ), and _ψ_ : _V �→_ 2 **[X]** the scope function that maps each node in _V_ to a subset of **X** (i.e., scope). This scope function maps the nodes in the graph to specific subsets of random variables **X** . A leaf node _O_ leaf of _G_ computes a probability density over its scope _ψ_ ( _O_ leaf). All internal nodes of _G_ are either sum nodes ( _S_ ) or product nodes ( _P_ ). A sum node _S_ computes a convex combination of its children: _S_ =[�] _N ∈_ **ch** ( _S_ ) _[w][S,N][N]_[where] **[ ch]**[(] _[S]_[)][ denotes the children of node] _[ S]_[.][A product node] _[ P]_ computes a product of its children: _P_ =[�] _N ∈_ **ch** ( _P_ ) _[N]_[.] 

The output of PC is typically the value of the root node _O_ root. We mainly focus on the marginal probability computation inside PCs (e.g., P[ _xj_ = 1] = � _x∈_ **X** _,xj_ =1 _[O]_[root][(] _[x]_[)] _[/]_ � _x∈_ **X** _[O]_[root][(] _[x]_[)] _[dx]_[).][The] inference of marginal probability is tractable by imposing structural constraints of _smoothness_ and _decomposability_ : (1) _decomposability_ : for any product node _P ∈V_ , _ψ_ ( _v_ 1) _∩ ψ_ ( _v_ 2) = _∅_ , for _v_ 1 _, v_ 2 _∈_ **ch** ( _P_ ) _, v_ 1 = _v_ 2, and 2) _smoothness_ : for any sum node _S ∈V_ , _ψ_ ( _v_ 1) = _ψ_ ( _v_ 2), for _v_ 1 _, v_ 2 _∈_ **ch** ( _S_ ). With the constraint of decomposability, the integrals on the product node _P_ can be factorized into integrals of children nodes **ch** ( _P_ ), which can be formulated as follows: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2024_COLEP_Certifiably_Robust_Learning-Reasoning_Conformal_Prediction_via_Probabilistic_Circuits_images/iclr_2024_COLEP_Certifiably_Robust_Learning-Reasoning_Conformal_Prediction_via_Probabilistic_Circuits.pdf-0018-05.png)


With the constraint of smoothness, the integrals on the sum node _S_ can be decomposed linearly to integrals on the children nodes **ch** ( _S_ ), which can be formulated as: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2024_COLEP_Certifiably_Robust_Learning-Reasoning_Conformal_Prediction_via_Probabilistic_Circuits_images/iclr_2024_COLEP_Certifiably_Robust_Learning-Reasoning_Conformal_Prediction_via_Probabilistic_Circuits.pdf-0018-07.png)


We can see that the smoothness and decomposability of PCs enable the integral computation on parent nodes to be pushed down to that on children nodes, and thus, the integral � _x∈_ **X** _,xj_ =1 _[O]_[root][(] _[x]_[)] or � _x∈_ **X** _[O]_[root][(] _[x]_[)][can][be][computed][by][a][single][forwarding][pass][of][the][graph] _[G]_[.][In][total,][we][only] need two forwarding passes in the graph _G_ to compute the marginal probability P[ _xj_ = 1] = � _x∈_ **X** _,xj_ =1 _[O]_[root][(] _[x]_[)] _[/]_ � _x∈_ **X** _[O]_[root][(] _[x]_[)] _[dx]_[.] 

**Knowledge rules in PCs.** For the implication rule _A_ = _⇒ B_ , we define _A_ as conditional variables and _B_ as consequence variables. Suppose that the _r_ -th PC encodes a set of implication rules defined over the set of Boolean variables _Vr_ . The PC is called a homogeneous logic PC if there exists a bipartition _Vrd_ and _Vrs_ ( _Vrd ∪ Vrs_ = _V, Vrd ∩ Vrs_ = _∅_ ) such that all variables in _Vrd_ only appear as conditional variables and all variables in _Vrs_ only appears as consequence variables. We construct homogeneous logic PCs to follow the structural properties of PCs (i.e., decomposability and smoothness) and can show that such construction is intuitive in real-world applications. For example, in the road sign recognition task, we construct a PC with the knowledge of shapes. We encode a set of implication rules _A_ = _⇒ B_ where _A_ is the road sign, and _B_ is the corresponding shape (e.g., a road sign can imply an octagon shape). We can verify that such a PC is a homogeneous PC and follow the structural constraints in the computation graph _G_ , as shown in Figure 1. 

C.2 MULTIPLE PCS WITH COEFFICIENTS _βr_ IN THE COMBINATION OF PCS 

Let us consider the case with _R_ PCs. The conditional class probability of the _j_ -th class corrected by _r_ -th PC can be formulated as: 

18 

Published as a conference paper at ICLR 2024 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2024_COLEP_Certifiably_Robust_Learning-Reasoning_Conformal_Prediction_via_Probabilistic_Circuits_images/iclr_2024_COLEP_Certifiably_Robust_Learning-Reasoning_Conformal_Prediction_via_Probabilistic_Circuits.pdf-0019-01.png)


where _µj_ denotes the _j_ -th element of vector _µ_ , and _T_ ( _a, b_ ) = log( _ab_ + (1 _− a_ )(1 _− b_ )), and _Fr_ ( _µ_ ) = exp _Hh_ =1 _r[w][hr]_[I][[] _[µ][ ∼][K][hr]_[]] . _Hr_ , _Khr_ , _whr_ are the number of knowledge rules, the _h_ -th logic rule, �� � and the weight of the _h_ -th logic rule for the _r_ -th PC. Note that the numerator and denominator of eq. (24) can be exactly and efficiently computed with a single forwarding pass of PCs (Hitzler & Sarker, 2022; Choi & Darwiche, 2017; Rooshenas & Lowd, 2014). Specifically, the inference time complexity of one PC defined on graph ( _V, E_ ) with node set _V_ and edge set _E_ is _O_ ( _|V|_ ). We improve the expressiveness of the reasoning component by combining _R_ PCs with coefficients _βr_ for the _r_ -th PC. Formally, given data input _x_ , the corrected conditional class probability _π_ ˆ _j_[COLEP] can be expressed as: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2024_COLEP_Certifiably_Robust_Learning-Reasoning_Conformal_Prediction_via_Probabilistic_Circuits_images/iclr_2024_COLEP_Certifiably_Robust_Learning-Reasoning_Conformal_Prediction_via_Probabilistic_Circuits.pdf-0019-03.png)


We also note that the coefficients _βr_ correspond to the accuracy of _r_ -th PC normalized over all PC accuracies. The core of this formulation is the mixture model involving a latent variable _r_ PC representing the PCs. In short, we can write _π_ ˆ _j_[(] _[r]_[)][(] _[x]_[)][ as] _[π]_[ˆ] _j_[(] _[r]_[)][(] _[x]_[)][=][P][[] _[Y]_[=] _[y][|][X]_[=] _[x, r]_[PC][=] _[r]_[]][, and] _π_ ˆ _j_[COLEP] ( _x_ ) as the marginalized probability over the latent variable _r_ PC such as: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2024_COLEP_Certifiably_Robust_Learning-Reasoning_Conformal_Prediction_via_Probabilistic_Circuits_images/iclr_2024_COLEP_Certifiably_Robust_Learning-Reasoning_Conformal_Prediction_via_Probabilistic_Circuits.pdf-0019-05.png)


Hence, the coefficient _βr_ for the _r_ -th PC are determined by P[ _r_ PC = _r_ ]. Although we lack direct knowledge of this probability, we can estimate it using the data by examining how frequently each PC _r_ correctly predicts the outcome across the given examples, similarly as in the estimation of prior class probabilities for Naive Bayes classifiers. 

## C.3 RANDOMIZED SMOOTHING 

**Lemma C.1** (Randomized Smoothing (Cohen et al., 2019)) **.** _Considering the class conditional_ ˆ _probability πj_ : R _[d] �→_ [0 _,_ 1] _, we construct a smoothed model gj_ ( _x_ ; _σ_ ) = E _η∼N_ (0 _,σ_ 2)[ˆ _πj_ ( _x_ + _η_ )] _. With input perturbation ∥ϵ∥_ 2 _≤ δ, the smoothed model satisfies:_ 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2024_COLEP_Certifiably_Robust_Learning-Reasoning_Conformal_Prediction_via_Probabilistic_Circuits_images/iclr_2024_COLEP_Certifiably_Robust_Learning-Reasoning_Conformal_Prediction_via_Probabilistic_Circuits.pdf-0019-09.png)


_where_ Φ _is the Gaussian cumulative density function (CDF) and_ Φ _[−]_[1] _is its inverse._ 

In practice, we can estimate the expectation in _gj_ ( _x_ ; _σ_ ) by taking the average over many i.i.d. realizations of Gaussian noises following (Cohen et al., 2019). Thus, the output probability of the smoothed model can be bounded given input perturbations. Note that the specific ways of certifying the learning component are orthogonal to certifying the reasoning component and the conformal prediction procedure, and one can plug in different certification strategies for the learning component. 

## D CONFORMAL PREDICTION WITH COLEP 

## D.1 FLEXIBILITY OF USER-DEFINED COVERAGE LEVEL 

In this part, we illustrate the flexibility of COLEP to use different miscoverage levels _αj_ for different label classes _j ∈_ [ _Nc_ ] and prove the guaranteed coverage. Recall that in the standard conformal prediction setting, we are given a desired coverage 1 _− α_ and construct the prediction set of a new test sample _C_[ˆ] _n,α_ ( _Xn_ +1) with the guarantee of marginal coverage: P[ _Yn_ +1 _⊆ C_[ˆ] _n,α_ ( _Xn_ +1)] _≥_ 1 _− α_ . In COLEP, we construct the prediction set _C_[ˆ] _n,α_[COLEP] _j[j]_ ( _Xn_ +1) for the _j_ -th class ( _j ∈_ [ _Nc_ ]) as in Equation (7) 

19 

Published as a conference paper at ICLR 2024 

> and construct the final prediction set _C_[ˆ] _n,α_[COLEP] ( _Xn_ +1) as in Equation (8). Then we have the following: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2024_COLEP_Certifiably_Robust_Learning-Reasoning_Conformal_Prediction_via_Probabilistic_Circuits_images/iclr_2024_COLEP_Certifiably_Robust_Learning-Reasoning_Conformal_Prediction_via_Probabilistic_Circuits.pdf-0020-02.png)


Therefore, we conclude that as long as we set max _j∈_ [ _Nc_ ] _{αj} ≤ α_ , we have the guarantee of marginal coverage: P � _Yn_ +1 _∈ C_[ˆ] _n,α_[COLEP] ( _Xn_ +1)� _≥_ 1 _− α_ . The flexibility of COLEP to set various coverage levels for different label classes is advantageous in addressing problems related to the imbalance of classes. For example, when samples of class _j_ are extremely vulnerable to be attacked by the adversary, we can set a higher miscoverage level _αj_ for class _j_ to consider the class-specific vulnerability instead of setting a higher _α_ for all classes which may affect the performance on other classes. 

## D.2 ILLUSTRATIVE EXAMPLE 

Let us consider the example of classifying whether the given image has a stop sign in Figure 1. The main model generates an estimate _π_ ˆ( _x_ ), meaning that the image _x_ has a stop sign with probability _π_ ˆ( _x_ ). Suppose that we have a knowledge model to detect whether the image has an object of octagon shape with an estimate _π_ ˆ _o_ ( _x_ ), meaning that the image _x_ has an object of octagon shape with ˆ probability _πo_ ( _x_ ). Then we can encode the logic rule “IsStopSign = _⇒_ IsOctagon” with weight _w_ in the PC, as shown in Figure 1. The PC defines a joint distribution over the Bernoulli random variables IsStopSign and IsStopSign and computes the likelihood of an instantiation of the random variables at its output (e.g., _p_ (IsStopSign = 1 _,_ IsOctagon = 0)). The represented joint distribution in the PC is determined by the encoded logic rule with the pattern in the top PC of Figure 1. We can see that only the assignment (IsStopSign = 1 _,_ IsOctagon = 0) is not multiplied by the weight _w_ due to the contradiction of the rule. Therefore, the probability of instantiations of random variables with such contradictions will be down-weighted after the correction of the PC. In the simple PC, we formally compute the class probability as: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2024_COLEP_Certifiably_Robust_Learning-Reasoning_Conformal_Prediction_via_Probabilistic_Circuits_images/iclr_2024_COLEP_Certifiably_Robust_Learning-Reasoning_Conformal_Prediction_via_Probabilistic_Circuits.pdf-0020-06.png)


Note that the nominator and denominator can be tractably computed by a forwarding pass of the PC graph. COLEP can enlarge the probability estimate _p_ (IsStopSign = 1) when _x_ is a stop sign and reduce it when _x_ is not a stop sign by leveraging additional concept information _p_ (IsStopSign, as rigorously analyzed in Lemma 5.1. 

## E OVERVIEW OF CERTIFICATION 

We provide the overview of certification flow in Figure 4, for a better understanding the connections and differences of certifications in COLEP. 

## F OMITTED PROOFS IN SECTION 4 

F.1 PROOF OF THM. 1 

**Theorem 1** (Generalization of Thm. 1 with multiple PCs). _Given any input x and perturbation bound δ, we let_ [ _π_ ˆ _j∀_ ( _x_ ) _, π_ ˆ _j∀_ ( _x_ )] _be bounds for the estimated conditional class and concept probabilities by_ 

20 

Published as a conference paper at ICLR 2024 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2024_COLEP_Certifiably_Robust_Learning-Reasoning_Conformal_Prediction_via_Probabilistic_Circuits_images/iclr_2024_COLEP_Certifiably_Robust_Learning-Reasoning_Conformal_Prediction_via_Probabilistic_Circuits.pdf-0021-01.png)


**----- Start of picture text -----**<br>
𝜋# ["#$%&] 𝑥% (with reasoning component) achieves<br>𝜋# higher prediction coverage ["#$%&] reasoning component) (Theorem 4)𝑥% achieves  higher prediction accuracy  than 𝜋# 𝑥% (without  Certified prediction set  with the coverage guarantee 𝐶 [C] ',)"#$%&!(𝑥%)<br>than 𝜋# 𝑥% (Theorem 5) (1 −𝛼)<br>(Theorem 2)<br>with bounded perturbationAdversarial test samples 𝑥% −𝑥 ! ≤𝛿 𝑥% Main Model & Knowledge models Bound of output likelihood[𝜋# 𝑥% , 𝜋# 𝑥% ] ComponentReasoning  after the reasoning component[𝕃Bound of output likelihood 𝜋# ["#$%&] 𝑥% , 𝕌[𝜋# ["#$%&] 𝑥% ]]<br>(Achieved by robustness  (Theorem 1) Standard prediction set  with the coverage guarantee 𝐶 [C] ',)"#$%&(𝑥%)<br>certification methods (e.g.,  (1 −𝛼 ["#$%&]["#$] )<br>randomized smoothing)) (Theorem 3)<br>**----- End of picture text -----**<br>


Figure 4: Overview of prediction certification of COLEP. The certification setting is that the inference time adversaries can violate the data exchangeability assumption, compromising the guaranteed coverage. Therefore, the certification generally achieves the following three goals. (1) We can preserve the guaranteed coverage using a prediction set that takes the perturbation bound into account (Thm. 2), achieved by computing the probability bound of models before the reasoning component (by randomized smoothing) and the bound after the reasoning component (by Thm. 1). (2) We prove the worst-case coverage (a lower bound) if we use the standard prediction set as before (Thm. 3). (3) We theoretically show that COLEP can achieve a better prediction coverage (Thm. 4) and prediction accuracy (Thm. 5) than a data-driven model without the reasoning component. 

_the models with j∀ ∈_ [ _Nc_ + _L_ ] _. Let Vrd[j][be the set of index of conditional variables in the][ r][-th PC] except for j ∈_ [ _Nc_ ] _and Vrs[j][be that of consequence variables.][Then the bound for][ COLEP][-corrected] estimate of the conditional class probability π_ ˆ _j[COLEP] is given by:_ 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2024_COLEP_Certifiably_Robust_Learning-Reasoning_Conformal_Prediction_via_Probabilistic_Circuits_images/iclr_2024_COLEP_Certifiably_Robust_Learning-Reasoning_Conformal_Prediction_via_Probabilistic_Circuits.pdf-0021-04.png)


_where T_ ( _a, b_ ) = log( _ab_ + (1 _− a_ )(1 _− b_ )) _._ 

_Proof of Thm. 1._ We first provide a lemma that proves the monotonicity of the complex function with respect to the estimated condition probabilities of conditional variables and consequence variables within PCs as follows. 

**Lemma F.1** (Function Monotonicity within PCs) **.** _We define the target function as G_ ( _q, d_ ) = � _µ∈Mjd_[exp] �� _j∀∈_ [ _Nc_ + _L_ ] _\{j}[T]_[(ˆ] _[π][j] ∀_[(] _[x]_[)] _[, µ][j] ∀_[)] � _Fr_ ( _µ_ ) _, where q ∈_ [ _Nc_ + _L_ ] _\{j} and Mjd_ = _{µ ∈_ ˆ _Mconditional variable in the PC and_ : _µj_ = _d}_ ( _d ∈{_ 0 _,_ 1 _}_ ) _. Then ∂G/∂we haveπ_ ˆ _q_ ( _x∂G/∂_ ) _≥_ 0 _if class labelπq_ ( _x_ ) _≤_ 0 _if qclass corresponds to a consequencelabel q corresponds to a variable in the PC._ 

> _Proof of Lemma F.1._ monotonicity, we only need to determine the sign of the coefficients ofNote that the function _G_ ( _q, d_ ) is linear with respect to _π_ ˆ _q_ ( _x_ ) _π_ .ˆWe first consider the _q_ ( _x_ ). To explore the case where class label _q_ corresponds to a conditional variable. We can rearrange the expression of _G_ ( _q, d_ ) as follows: 

21 

Published as a conference paper at ICLR 2024 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2024_COLEP_Certifiably_Robust_Learning-Reasoning_Conformal_Prediction_via_Probabilistic_Circuits_images/iclr_2024_COLEP_Certifiably_Robust_Learning-Reasoning_Conformal_Prediction_via_Probabilistic_Circuits.pdf-0022-01.png)


Then we rearrange the coefficients of _π_ ˆ _q_ ( _x_ ) as follows: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2024_COLEP_Certifiably_Robust_Learning-Reasoning_Conformal_Prediction_via_Probabilistic_Circuits_images/iclr_2024_COLEP_Certifiably_Robust_Learning-Reasoning_Conformal_Prediction_via_Probabilistic_Circuits.pdf-0022-03.png)


where _R_ ( _µ, q_ ) : _{_ 0 _,_ 1 _}[N][c]_[+] _[L] ×_ Z[+] _�→{_ 0 _,_ 1 _}[N][c]_[+] _[L]_ rewrites the _q_ -th dimension of vector _µ_ to 1. Note that when _q_ is the index of a conditional variable, given any implication logic rule _Khr_ , we have I[ _R_ ( _µ, q_ ) _∼ Khr_ ] _≤_ I[ _µ ∼ Khr_ ]. Therefore, _Fr_ ( _R_ ( _µ, q_ )) _− Fr_ ( _µ_ ) _≤_ 0 holds and _∂G/∂π_ ˆ _q_ ( _x_ ) _≤_ 0. 

Similarly, when _q_ is the index of a consequence variable, given any implication logic rule _Khr_ , we have I[ _R_ ( _µ, q_ ) _∼ Khr_ ] _≥_ I[ _µ ∼ Khr_ ]. Therefore, _Fr_ ( _R_ ( _µ, q_ )) _− Fr_ ( _µ_ ) _≥_ 0 holds and _∂G/∂π_ ˆ _q_ ( _x_ ) _≥_ 0. 

Consider the _r_ -th PC. We first rearrange the formulation of the class conditional probability _π_ ˆ _j_[(] _[r]_[)][(] _[x]_[)] to the pattern of function _G_ ( _·_ ) and then leverage Lemma F.1 to derive the upper and lower bound. The reformulation of _π_ ˆ _j_[(] _[r]_[)][(] _[x]_[)][ is shown as the following:] 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2024_COLEP_Certifiably_Robust_Learning-Reasoning_Conformal_Prediction_via_Probabilistic_Circuits_images/iclr_2024_COLEP_Certifiably_Robust_Learning-Reasoning_Conformal_Prediction_via_Probabilistic_Circuits.pdf-0022-07.png)


Next, we focus on deriving the lower bound L[ˆ _πj_[(] _[r]_[)][(] _[x]_[)]][, and the upper bound can be derived similarly.] 

22 

Published as a conference paper at ICLR 2024 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2024_COLEP_Certifiably_Robust_Learning-Reasoning_Conformal_Prediction_via_Probabilistic_Circuits_images/iclr_2024_COLEP_Certifiably_Robust_Learning-Reasoning_Conformal_Prediction_via_Probabilistic_Circuits.pdf-0023-01.png)


In Equation (41), we relax the numerator and denominator separately and then will show how to derive the upper bound of the numerator and the lower bound of the denominator in Equation (41). We can apply Lemma F.1 for any _q ∈_ [ _Nc_ + _L_ ] _\{j}_ and get the following: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2024_COLEP_Certifiably_Robust_Learning-Reasoning_Conformal_Prediction_via_Probabilistic_Circuits_images/iclr_2024_COLEP_Certifiably_Robust_Learning-Reasoning_Conformal_Prediction_via_Probabilistic_Circuits.pdf-0023-03.png)


where _Vrd[j]_[is the set of index of conditional variables in the] _[ r]_[-th PC except for] _[ j][∈]_[[] _[N][c]_[]][ and] _[ V] rs[j]_[is] that of consequence variables. 

From Equation (41) and Equation (42), we can derive the final bound of ˆ _πj_[COLEP] ( _x_ ) by considering all the PCs: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2024_COLEP_Certifiably_Robust_Learning-Reasoning_Conformal_Prediction_via_Probabilistic_Circuits_images/iclr_2024_COLEP_Certifiably_Robust_Learning-Reasoning_Conformal_Prediction_via_Probabilistic_Circuits.pdf-0023-06.png)


F.2 PROOF OF THM. 2 

**Theorem 2** (Recall). Consider a new test sample _Xn_ +1 drawn from _PXY_ . For any bounded perturbation _∥ϵ∥_ 2 _≤ δ_ in the input space and the adversarial sample _X_[˜] _n_ +1 := _Xn_ +1 + _ϵ_ , we have the following guaranteed marginal coverage: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2024_COLEP_Certifiably_Robust_Learning-Reasoning_Conformal_Prediction_via_Probabilistic_Circuits_images/iclr_2024_COLEP_Certifiably_Robust_Learning-Reasoning_Conformal_Prediction_via_Probabilistic_Circuits.pdf-0023-09.png)


if we construct the certified prediction set of COLEP where 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2024_COLEP_Certifiably_Robust_Learning-Reasoning_Conformal_Prediction_via_Probabilistic_Circuits_images/iclr_2024_COLEP_Certifiably_Robust_Learning-Reasoning_Conformal_Prediction_via_Probabilistic_Circuits.pdf-0023-11.png)


23 

Published as a conference paper at ICLR 2024 

and _Sπ_ ˆ _j_ COLEP _δ_ ( _·, ·_ ) is a function of worst-case non-conformity score considering perturbation radius _δ_ : 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2024_COLEP_Certifiably_Robust_Learning-Reasoning_Conformal_Prediction_via_Probabilistic_Circuits_images/iclr_2024_COLEP_Certifiably_Robust_Learning-Reasoning_Conformal_Prediction_via_Probabilistic_Circuits.pdf-0024-02.png)


with U _δ_ [ˆ _πj_[COLEP] ( _x_ )] = max _|η|_ 2 _≤δ_ ˆ _πj_[COLEP] ( _x_ + _η_ ) and L _δ_ [ˆ _πj_[COLEP] ( _x_ )] = min _|η|_ 2 _≤δ_ ˆ _πj_[COLEP] ( _x_ + _η_ ). _Proof of Thm. 2._ We can upper bound the probability of miscoverage of the main task with the probability of miscoverage of label classes as follows: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2024_COLEP_Certifiably_Robust_Learning-Reasoning_Conformal_Prediction_via_Probabilistic_Circuits_images/iclr_2024_COLEP_Certifiably_Robust_Learning-Reasoning_Conformal_Prediction_via_Probabilistic_Circuits.pdf-0024-04.png)


where _C_[ˆ] _n,α,j_[COLEP] _[δ]_[( ˜] _[X][n]_[+1][)][:=] � _q[y] ∈{_ 0 _,_ 1 _}_ : _Sπ_ ˆ _j_ COLEP( _X_[˜] _n_ +1 _, y_ ) _≤ Q_ 1 _−α_ ( _Sπ_ ˆ _j_ COLEP _δ_ ( _{Xi,_ I[ _Yi_ = _j_ ] _}i∈I_ cal )� is the certified prediction set for the class label _j_ . Note that Equation (50) holds because we have _Sπ_ ˆ _j_ COLEP _δ_ ( _Xn_ +1 _,_ I[ _Yn_ +1= _j_ ]) _≥ Sπ_ ˆ _j_ COLEP( _X_[˜] _n_ +1 _,_ I[ _Yn_ +1= _j_ ]). Next, we will prove Equation (50) rigorously. We first consider the probability of miscoverage conditioned on _Yn_ +1 = _j_ . 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2024_COLEP_Certifiably_Robust_Learning-Reasoning_Conformal_Prediction_via_Probabilistic_Circuits_images/iclr_2024_COLEP_Certifiably_Robust_Learning-Reasoning_Conformal_Prediction_via_Probabilistic_Circuits.pdf-0024-06.png)


Similarly, we consider the probability of miscoverage conditioned on _Yn_ +1 = _j_ . 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2024_COLEP_Certifiably_Robust_Learning-Reasoning_Conformal_Prediction_via_Probabilistic_Circuits_images/iclr_2024_COLEP_Certifiably_Robust_Learning-Reasoning_Conformal_Prediction_via_Probabilistic_Circuits.pdf-0024-08.png)


Combining Equation (55) and Equation (59), we prove that Equation (50) holds. Finally, from Equation (51), we prove that: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2024_COLEP_Certifiably_Robust_Learning-Reasoning_Conformal_Prediction_via_Probabilistic_Circuits_images/iclr_2024_COLEP_Certifiably_Robust_Learning-Reasoning_Conformal_Prediction_via_Probabilistic_Circuits.pdf-0024-10.png)


## F.3 THM. 2 WITH FINITE-SAMPLE ERRORS BY RANDOMIZED SMOOTHING 

**Theorem 6** (Thm. 2 with finite-sample errors of learning component certification by randomized smoothing) **.** _Consider a new test sample Xn_ +1 _drawn from PXY . For any bounded perturbation_ 

24 

Published as a conference paper at ICLR 2024 

_∥ϵ∥_ 2 _≤ δ in the input space and the adversarial sample X_[˜] _n_ +1 := _Xn_ +1 + _ϵ, we have the following guaranteed marginal coverage:_ 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2024_COLEP_Certifiably_Robust_Learning-Reasoning_Conformal_Prediction_via_Probabilistic_Circuits_images/iclr_2024_COLEP_Certifiably_Robust_Learning-Reasoning_Conformal_Prediction_via_Probabilistic_Circuits.pdf-0025-02.png)


_if we construct the certified prediction set of COLEP where_ 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2024_COLEP_Certifiably_Robust_Learning-Reasoning_Conformal_Prediction_via_Probabilistic_Circuits_images/iclr_2024_COLEP_Certifiably_Robust_Learning-Reasoning_Conformal_Prediction_via_Probabilistic_Circuits.pdf-0025-04.png)


_and Sπ_ ˆ _jCOLEPδ_ ( _·, ·_ ) _is a function of worst-case non-conformity score considering perturbation radius δ:_ 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2024_COLEP_Certifiably_Robust_Learning-Reasoning_Conformal_Prediction_via_Probabilistic_Circuits_images/iclr_2024_COLEP_Certifiably_Robust_Learning-Reasoning_Conformal_Prediction_via_Probabilistic_Circuits.pdf-0025-06.png)


_where_ U _β,δ_ [ˆ _πj[COLEP]_ ( _x_ )] _and_ L _β,δ_ [ˆ _πj[COLEP]_ ( _x_ )] _are computed by plugging in the β-confidence probability bound_ [ _π_ ˆ _j∀_ ( _x_ ; _β_ ) _, π_ ˆ _j∀_ ( _x_ ; _β_ )] _into Thm. 1. The high-confidence bound_ [ _π_ ˆ _j∀_ ( _x_ ; _β_ ) _, π_ ˆ _j∀_ ( _x_ ; _β_ )] _can be obtained by randomized smoothing as the following:_ 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2024_COLEP_Certifiably_Robust_Learning-Reasoning_Conformal_Prediction_via_Probabilistic_Circuits_images/iclr_2024_COLEP_Certifiably_Robust_Learning-Reasoning_Conformal_Prediction_via_Probabilistic_Circuits.pdf-0025-08.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2024_COLEP_Certifiably_Robust_Learning-Reasoning_Conformal_Prediction_via_Probabilistic_Circuits_images/iclr_2024_COLEP_Certifiably_Robust_Learning-Reasoning_Conformal_Prediction_via_Probabilistic_Circuits.pdf-0025-09.png)


_where_ E[ˆ] _is the empirical mean of NMC Monte-Carlo samples during randomized smoothing and bHoef_ ( _β_ ) = ln 1 _/β is the Hoeffding’s finite-sample error and bBern_ ( _β_ ) = 2 _V_ ln 2 _/β_ + � 2 _NMC_ � _NMC_ 7 ln 2 _/β_ 3( _NMC −_ 1) _[is the Berstein’s finite-sample error.]_ 

_Proof of Thm. 6._ Bounding the finite-sample error of randomized smoothing similarly as Theorem 1 in Anonymous (2023), we get that for a given _j∀ ∈_ [ _Nc_ + _L_ ], we have the following: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2024_COLEP_Certifiably_Robust_Learning-Reasoning_Conformal_Prediction_via_Probabilistic_Circuits_images/iclr_2024_COLEP_Certifiably_Robust_Learning-Reasoning_Conformal_Prediction_via_Probabilistic_Circuits.pdf-0025-12.png)


We can upper bound the probability of miscoverage of the main task with the probability of miscoverage of label classes as follows: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2024_COLEP_Certifiably_Robust_Learning-Reasoning_Conformal_Prediction_via_Probabilistic_Circuits_images/iclr_2024_COLEP_Certifiably_Robust_Learning-Reasoning_Conformal_Prediction_via_Probabilistic_Circuits.pdf-0025-14.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2024_COLEP_Certifiably_Robust_Learning-Reasoning_Conformal_Prediction_via_Probabilistic_Circuits_images/iclr_2024_COLEP_Certifiably_Robust_Learning-Reasoning_Conformal_Prediction_via_Probabilistic_Circuits.pdf-0025-15.png)


where _C_[ˆ] _n,α,j_[COLEP] _[δ]_[( ˜] _[X][n]_[+1][) :=] � _q[y] ∈{_ 0 _,_ 1 _}_ : _Sπ_ ˆ _j_ COLEP( _X_[˜] _n_ +1 _, y_ ) _≤ Q_ 1 _−α_ +2 _β_ ( _Sπ_ ˆ _j_ COLEP _δ_ ( _{Xi,_ I[ _Yi_ = _j_ ] _}i∈I_ cal)� is the certified prediction set for the class label _j_ . Note that Equation (70) holds because we have _Sπ_ ˆ _j_ COLEP _δ_ ( _Xn_ +1 _,_ I[ _Yn_ +1= _j_ ]) _≥ Sπ_ ˆ _j_ COLEP( _X_[˜] _n_ +1 _,_ I[ _Yn_ +1= _j_ ]). Next, we will prove Equation (70) rigorously. We first consider 

25 

Published as a conference paper at ICLR 2024 

the probability of miscoverage conditioned on _Yn_ +1 = _j_ . 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2024_COLEP_Certifiably_Robust_Learning-Reasoning_Conformal_Prediction_via_Probabilistic_Circuits_images/iclr_2024_COLEP_Certifiably_Robust_Learning-Reasoning_Conformal_Prediction_via_Probabilistic_Circuits.pdf-0026-02.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2024_COLEP_Certifiably_Robust_Learning-Reasoning_Conformal_Prediction_via_Probabilistic_Circuits_images/iclr_2024_COLEP_Certifiably_Robust_Learning-Reasoning_Conformal_Prediction_via_Probabilistic_Circuits.pdf-0026-03.png)


Combining Equation (77) and Equation (82), we prove that Equation (70) holds. Finally, from Equation (72), we prove that: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2024_COLEP_Certifiably_Robust_Learning-Reasoning_Conformal_Prediction_via_Probabilistic_Circuits_images/iclr_2024_COLEP_Certifiably_Robust_Learning-Reasoning_Conformal_Prediction_via_Probabilistic_Circuits.pdf-0026-05.png)


## F.4 PROOF OF THM. 3 

**Theorem** _X_ ˜ _n_ +1 := _X_ **3** _n_ (Recall).+1 + _ϵ_ with any perturbationConsider the new _∥_ sample _ϵ∥_ 2 _≤Xδ_ in the input space. _n_ +1 drawn from _P_ We have: _XY_ and adversarial sample 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2024_COLEP_Certifiably_Robust_Learning-Reasoning_Conformal_Prediction_via_Probabilistic_Circuits_images/iclr_2024_COLEP_Certifiably_Robust_Learning-Reasoning_Conformal_Prediction_via_Probabilistic_Circuits.pdf-0026-08.png)


_τj_[COLEP][cer] = max � _τ_ : _Qτ_ ( _{Sπ_ ˆ _j_ COLEP _δ_ ( _Xi,_ I[ _Yi_ = _j_ ]) _}i∈I_ cal) _≤ Q_ 1 _−α_ ( _{Sπ_ ˆ _j_ COLEP( _Xi,_ I[ _Yi_ = _j_ ]) _}i∈I_ cal)� _._ (85) 

_Proof of Thm. 3._ We can bound the probability of coverage of the main task with the probability of coverage of label classes as follows: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2024_COLEP_Certifiably_Robust_Learning-Reasoning_Conformal_Prediction_via_Probabilistic_Circuits_images/iclr_2024_COLEP_Certifiably_Robust_Learning-Reasoning_Conformal_Prediction_via_Probabilistic_Circuits.pdf-0026-11.png)


26 

Published as a conference paper at ICLR 2024 

where Equation (90) holds because _Sπ_ ˆ _j_ COLEP _δ_ ( _Xn_ +1 _,_ I[ _Yn_ +1= _j_ ]) _≥ Sπ_ ˆ _j_ COLEP( _X_[˜] _n_ +1 _,_ I[ _Yn_ +1= _j_ ]) and is proved rigorously in the proof of Thm. 2 in Appendix F.2. 

## G FINITE-SAMPLE CERTIFIED COVERAGE 

In this section, we consider the finite-sample error induced by the finite size of the calibration set _I_ cal. We provide the finite-sample certified coverage in the adversary setting as follows. 

**Corollary 1** (Certified Coverage with Finite Samples) **.** _Consider the new sample Xn_ +1 _drawn from PXY and adversarial sample X_[˜] _n_ +1 := _Xn_ +1 + _ϵ with any perturbation ∥ϵ∥_ 2 _≤ δ in the input space. We have the following finite-sample certified coverage:_ 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2024_COLEP_Certifiably_Robust_Learning-Reasoning_Conformal_Prediction_via_Probabilistic_Circuits_images/iclr_2024_COLEP_Certifiably_Robust_Learning-Reasoning_Conformal_Prediction_via_Probabilistic_Circuits.pdf-0027-05.png)


_where |Ical| is the size of the calibration set in conformal prediction._ 

_Proof sketch and remarks._ Leveraging corollary 1 of (Massart, 1990) and following the sketch in (Yang & Kuchibhotla, 2021), we can prove the finite-sample error bound. Combining the finitesample error with the certified coverage bound in Thm. 3 concludes the proof. Note that the union bound is applied to multiple models in (Yang & Kuchibhotla, 2021) while we circumvent it by aggregating individual results for class labels subtly, and thus, COLEP achieves an improved sample complexity of conformal prediction compared with ensemble learning given the same tolerance of finite-sample error. 

_Proof of Corollary 1._ From Corollary 1 of (Massart, 1990), we have that for all _ϵ >_ 0, 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2024_COLEP_Certifiably_Robust_Learning-Reasoning_Conformal_Prediction_via_Probabilistic_Circuits_images/iclr_2024_COLEP_Certifiably_Robust_Learning-Reasoning_Conformal_Prediction_via_Probabilistic_Circuits.pdf-0027-09.png)


For ease of notation, we let: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2024_COLEP_Certifiably_Robust_Learning-Reasoning_Conformal_Prediction_via_Probabilistic_Circuits_images/iclr_2024_COLEP_Certifiably_Robust_Learning-Reasoning_Conformal_Prediction_via_Probabilistic_Circuits.pdf-0027-11.png)


We let _t_ = _Qτj_ COLEPcer ( _Sπ_ ˆ _j_ COLEP _δ_ ( _{Xi,_ I[ _Yi_ = _j_ ] _}i∈I_ cal ). It implies that: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2024_COLEP_Certifiably_Robust_Learning-Reasoning_Conformal_Prediction_via_Probabilistic_Circuits_images/iclr_2024_COLEP_Certifiably_Robust_Learning-Reasoning_Conformal_Prediction_via_Probabilistic_Circuits.pdf-0027-13.png)


27 

Published as a conference paper at ICLR 2024 

Then using Equation (93), we can bound E[ _W_ ] as follows: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2024_COLEP_Certifiably_Robust_Learning-Reasoning_Conformal_Prediction_via_Probabilistic_Circuits_images/iclr_2024_COLEP_Certifiably_Robust_Learning-Reasoning_Conformal_Prediction_via_Probabilistic_Circuits.pdf-0028-02.png)


Inequality (a) follows from Formula 7.1.13 of (Abramowitz & Stegun, 1948). Finally, combining the results in Thm. 3, we get: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2024_COLEP_Certifiably_Robust_Learning-Reasoning_Conformal_Prediction_via_Probabilistic_Circuits_images/iclr_2024_COLEP_Certifiably_Robust_Learning-Reasoning_Conformal_Prediction_via_Probabilistic_Circuits.pdf-0028-04.png)


## H OMITTED PROOFS IN SECTION 5 

H.1 PROOF OF LEMMA 5.1 

**Lemma 6.1** (Recall). For any class probability within models _j ∈_ [ _Nc_ ] and data sample ( _X, Y_ ) drawn from the mixture distribution _Dm_ , we can prove that: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2024_COLEP_Certifiably_Robust_Learning-Reasoning_Conformal_Prediction_via_Probabilistic_Circuits_images/iclr_2024_COLEP_Certifiably_Robust_Learning-Reasoning_Conformal_Prediction_via_Probabilistic_Circuits.pdf-0028-08.png)


where _λ_[(] _j,[r] D_[)][= (1] _[/T]_[ (] _j,[r] D_[)][+] _[ e][−][w][ −]_[1)][ and we shorthand] _[π]_[ˆ] _[j]_[(] _[X]_[)][ as] _[π]_[ˆ] _[j]_[.] 

_Proof of Lemma 5.1._ We first consider one PC and then analyze the effectiveness of _R_ linearly combined PCs. Without loss of generality, we focus on the class conditional probability by the _r_ -th PC ( _r ∈_ [ _R_ ]) for the class label _j ∈_ [ _Nc_ ]. 

Recall that we map a set of implication rules encoded in the _r_ -th PC to an undirected graph _Gr_ = ( _Vr, Er_ ), where _Vr_ = [ _Nc_ + _L_ ] corresponds to the Boolean vector of class and concept predictions in the learning stage. There exists an edge between node _j_ 1 _∈Vr_ and node _j_ 2 _∈Vr_ iff they are connected by an implication rule _j_ 1 = _⇒ j_ 2 or _j_ 2 = _⇒ j_ 1. Let _Ar_ ( _j∀_ ) be the set of nodes in the connected component of node _j∀_ except for node _j∀_ (i.e., _j∀ ∈/ Ar_ ( _j∀_ )). Let _Ar,d_ ( _j∀_ ) _⊆Ar_ ( _j∀_ ) be the set of conditional variables and _Ar,s_ ( _j∀_ ) _⊆Ar_ ( _j∀_ ) be the set of consequence variables. Denote _Mj_ be the universal set of assignments except for the _j_ -th element. Let _Pd_ ( _j_ ) and _Ps_ ( _j_ ) be the set of PC index where _j_ appears as conditional variables and consequence variables, respectively. 

We consider a mixture of benign and adversarial distributions denoted by _Dm_ , which indicates that an input _x_ is either a benign example drawn from _Db_ with probability _pDb_ := P[ _x ∼Db_ ] or an adversarial example drawn from _Da_ with probability _pDa_ := P[ _x ∼Da_ ], where _pDb_ + _pDa_ = 1. We analyze the effectiveness of PCs in four cases: case (1) _r ∈Pd_ ( _j_ ) and _Y_ = _j_ , case (2) _r ∈Pd_ ( _j_ ) and _Y_ = _j_ , case (3) _r ∈Ps_ ( _j_ ) and _Y_ = _j_ , and case (4) _r ∈Ps_ ( _j_ ) and _Y_ = _j_ . We leave the probability conditions on _Y_ = _j_ or _Y_ = _j_ in each case for simplicity and consider them in the conclusions. 

28 

Published as a conference paper at ICLR 2024 

**Case (1)** _r ∈Pd_ ( _j_ ) and _Y_ = _j_ . We can derive the class conditional probability by PCs as follows: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2024_COLEP_Certifiably_Robust_Learning-Reasoning_Conformal_Prediction_via_Probabilistic_Circuits_images/iclr_2024_COLEP_Certifiably_Robust_Learning-Reasoning_Conformal_Prediction_via_Probabilistic_Circuits.pdf-0029-02.png)


Note that since _r ∈Ps_ ( _j_ ), we have[�] _µ∈M, µj_ =0[exp] �� _j∀∈_ [ _Nc_ + _L_ ] _\{j}[T]_[(ˆ] _[π][j] ∀_[(] _[x]_[)] _[, µ][j] ∀_[)] � _Fr_ ( _µ_ ) _≥_ � _µ∈M, µj_ =1[exp] �� _j∀∈_ [ _Nc_ + _L_ ] _\{j}[T]_[(ˆ] _[π][j] ∀_[(] _[x]_[)] _[, µ][j] ∀_[)] � _Fr_ ( _µ_ ), and thus, _π_ ˆ _j_[(] _[r]_[)][(] _[x]_[)] _[≤][π]_[ˆ] _[j]_[(] _[x]_[)][holds.] Next, we analyze the gap to get a tight bound based on the distribution of the Boolean vector of the ground truth class and concepts _ν_ . 

_Case (1a)_ : If _∃j∀ ∈Ar,s_ ( _j_ ) _, νj∀_ = 0, then we have: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2024_COLEP_Certifiably_Robust_Learning-Reasoning_Conformal_Prediction_via_Probabilistic_Circuits_images/iclr_2024_COLEP_Certifiably_Robust_Learning-Reasoning_Conformal_Prediction_via_Probabilistic_Circuits.pdf-0029-05.png)


ˆ ˆ When we have models such that _πj∀_ ( _x_ ) _>_ 1 _− tj∀,D |νj∀_ = 0 or _πj∀_ ( _x_ ) _<_ 1 _− tj∀,D |νj∀_ = 1 _, ∃j∀ ∈Ar_ ( _j_ ) (with probability 1 _− Zj,_[(] _[r] D_[)][),][we][leverage][the][general][bound][in] the case, 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2024_COLEP_Certifiably_Robust_Learning-Reasoning_Conformal_Prediction_via_Probabilistic_Circuits_images/iclr_2024_COLEP_Certifiably_Robust_Learning-Reasoning_Conformal_Prediction_via_Probabilistic_Circuits.pdf-0029-07.png)


ˆ ˆ When we have models of high quality such that _πj∀_ ( _x_ ) _≤_ 1 _− tj∀,D |νj∀_ = 0 and _πj∀_ ( _x_ ) _≥_ 1 _− tj∀,D |νj∀_ = 1 _, ∀j∀ ∈Ar_ ( _j_ ) (with probability _Zj,_[(] _[r] D_[)][), we can derive a tighter bound as follows:] 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2024_COLEP_Certifiably_Robust_Learning-Reasoning_Conformal_Prediction_via_Probabilistic_Circuits_images/iclr_2024_COLEP_Certifiably_Robust_Learning-Reasoning_Conformal_Prediction_via_Probabilistic_Circuits.pdf-0029-09.png)


29 

Published as a conference paper at ICLR 2024 

In inequality (a), we plug in the lower bound of confidence of a correct prediction _tj∀,D_ and the upper bound of confidence of a wrong prediction 1 _− tj∀,D_ for label _j∀_ . 

Combining Equations (109) and (113), we get the following bound for case (1a): 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2024_COLEP_Certifiably_Robust_Learning-Reasoning_Conformal_Prediction_via_Probabilistic_Circuits_images/iclr_2024_COLEP_Certifiably_Robust_Learning-Reasoning_Conformal_Prediction_via_Probabilistic_Circuits.pdf-0030-03.png)


where _λ_[(] _j,[r] D_[)][= (1] _[/T]_[ (] _j,[r] D_[)][+] _[ e][−][w][ −]_[1)][.] _Case (1b)_ : If _∀j∀ ∈Ar,s_ ( _j_ ) _, νj∀_ = 1, then we use the general bound in the case: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2024_COLEP_Certifiably_Robust_Learning-Reasoning_Conformal_Prediction_via_Probabilistic_Circuits_images/iclr_2024_COLEP_Certifiably_Robust_Learning-Reasoning_Conformal_Prediction_via_Probabilistic_Circuits.pdf-0030-05.png)


Note that case (1a) is with probability _Uj_[(] _[r]_[)] and case (1b) is with probability 1 _− Uj_[(] _[r]_[)] . Combining Equation (114) in _case (1a)_ and Equation (115) in _case (1b)_ , we have: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2024_COLEP_Certifiably_Robust_Learning-Reasoning_Conformal_Prediction_via_Probabilistic_Circuits_images/iclr_2024_COLEP_Certifiably_Robust_Learning-Reasoning_Conformal_Prediction_via_Probabilistic_Circuits.pdf-0030-07.png)


**Case (2)** : _r ∈Pd_ ( _j_ ) and _Y_ = _j_ . Since in the _r_ -th PC _j_ is the index of a conditional variable, we have _∀j∀ ∈Ar,s_ ( _j_ ) _, νj∀_ = 1, where _ν_ is the ground truth Boolean vector. Then we have the following: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2024_COLEP_Certifiably_Robust_Learning-Reasoning_Conformal_Prediction_via_Probabilistic_Circuits_images/iclr_2024_COLEP_Certifiably_Robust_Learning-Reasoning_Conformal_Prediction_via_Probabilistic_Circuits.pdf-0030-09.png)


Then the following holds in case (2): 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2024_COLEP_Certifiably_Robust_Learning-Reasoning_Conformal_Prediction_via_Probabilistic_Circuits_images/iclr_2024_COLEP_Certifiably_Robust_Learning-Reasoning_Conformal_Prediction_via_Probabilistic_Circuits.pdf-0030-11.png)


**Case (3)** : _r ∈Ps_ ( _j_ ) and _Y_ = _j_ . As in Case (1), we similarly consider cases based on the distribution of ground truth Boolean vector _ν_ to get a tighter bound. 

_Case (3a)_ : _∃j∀ ∈Ar,d_ ( _j_ ) _, νj∀_ = 1. 

ˆ ˆ When we have models such that _πj∀_ ( _x_ ) _>_ 1 _− tj∀,D |νj∀_ = 0 or _πj∀_ ( _x_ ) _<_ 1 _− tj∀,D |νj∀_ = 1 _, ∃j∀ ∈Ar_ ( _j_ ) (with probability 1 _− Zj,_[(] _[r] D_[)][),][we][leverage][the][general][bound][in] the case, 

E[ˆ _πj_[(] _[r]_[)][(] _[x]_[)] _[|][Y]_[=] _[ j]_[]] _[ ≤]_[E][[ˆ] _[π][j]_[(] _[x]_[)] _[|][Y]_[=] _[ j]_[]] (121) 

Equation (121) holds because in case (3), _j_ corresponds to a consequence variable in the _r_ -th PC, and thus, we always have: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2024_COLEP_Certifiably_Robust_Learning-Reasoning_Conformal_Prediction_via_Probabilistic_Circuits_images/iclr_2024_COLEP_Certifiably_Robust_Learning-Reasoning_Conformal_Prediction_via_Probabilistic_Circuits.pdf-0030-17.png)


30 

Published as a conference paper at ICLR 2024 

ˆ ˆ When we have models of high quality such that _πj∀_ ( _x_ ) _≤_ 1 _− tj∀,D |νj∀_ = 0 and _πj∀_ ( _x_ ) _≥_ 1 _− tj∀,D |νj∀_ = 1 _, ∀j∀ ∈Ar_ ( _j_ ) (with probability _Zj,_[(] _[r] D_[)][), we can derive a tighter bound as follows:] 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2024_COLEP_Certifiably_Robust_Learning-Reasoning_Conformal_Prediction_via_Probabilistic_Circuits_images/iclr_2024_COLEP_Certifiably_Robust_Learning-Reasoning_Conformal_Prediction_via_Probabilistic_Circuits.pdf-0031-02.png)


Therefore, the following holds in case (3a); 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2024_COLEP_Certifiably_Robust_Learning-Reasoning_Conformal_Prediction_via_Probabilistic_Circuits_images/iclr_2024_COLEP_Certifiably_Robust_Learning-Reasoning_Conformal_Prediction_via_Probabilistic_Circuits.pdf-0031-04.png)


_Case (3b)_ : If _∀j∀ ∈Ar,d_ ( _j_ ) _, νj∀_ = 0, then we leverage the general bound: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2024_COLEP_Certifiably_Robust_Learning-Reasoning_Conformal_Prediction_via_Probabilistic_Circuits_images/iclr_2024_COLEP_Certifiably_Robust_Learning-Reasoning_Conformal_Prediction_via_Probabilistic_Circuits.pdf-0031-06.png)


Note that case (3a) is with probability _Uj_[(] _[r]_[)] and case (3b) is with probability 1 _− Uj_[(] _[r]_[)] . Combining Equations (126) and (127), the following holds: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2024_COLEP_Certifiably_Robust_Learning-Reasoning_Conformal_Prediction_via_Probabilistic_Circuits_images/iclr_2024_COLEP_Certifiably_Robust_Learning-Reasoning_Conformal_Prediction_via_Probabilistic_Circuits.pdf-0031-08.png)


**Case (4)** : _r ∈Ps_ ( _j_ ) and _Y_ = _j_ . Since _j_ is the index of a consequence variable, we have _∀j∀ ∈ Ar,d_ ( _j_ ) _, νj∀_ = 0. Then we can lower bound the coefficients as follows: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2024_COLEP_Certifiably_Robust_Learning-Reasoning_Conformal_Prediction_via_Probabilistic_Circuits_images/iclr_2024_COLEP_Certifiably_Robust_Learning-Reasoning_Conformal_Prediction_via_Probabilistic_Circuits.pdf-0031-10.png)


Thus, in case (4) we have: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2024_COLEP_Certifiably_Robust_Learning-Reasoning_Conformal_Prediction_via_Probabilistic_Circuits_images/iclr_2024_COLEP_Certifiably_Robust_Learning-Reasoning_Conformal_Prediction_via_Probabilistic_Circuits.pdf-0031-12.png)


Then we consider all _R_ PCs to get the effectiveness of COLEP on the class conditional probability. Combining Equation (116) in case (1) and Equation (132) in case (4), we finally have: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2024_COLEP_Certifiably_Robust_Learning-Reasoning_Conformal_Prediction_via_Probabilistic_Circuits_images/iclr_2024_COLEP_Certifiably_Robust_Learning-Reasoning_Conformal_Prediction_via_Probabilistic_Circuits.pdf-0031-14.png)


where 

31 

Published as a conference paper at ICLR 2024 

Combining Equation (120) in case (2) and Equation (128) in case (3), we finally have: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2024_COLEP_Certifiably_Robust_Learning-Reasoning_Conformal_Prediction_via_Probabilistic_Circuits_images/iclr_2024_COLEP_Certifiably_Robust_Learning-Reasoning_Conformal_Prediction_via_Probabilistic_Circuits.pdf-0032-02.png)


where 

## H.2 PROOF OF THM. 4 

**Theorem 4** (Recall). Consider the adversary setting that the calibration set _I_ cal consists of _nDb_ samples drawn from the benign distribution _Db_ , while the new sample ( _Xn_ +1 _, Yn_ +1) is drawn _nDa_ wheretimes from the adversarial distribution _A_ (ˆ _πj, D_ ) is the expectation of prediction accuracy of _Da_ . Assume that _A_ (ˆ _πjπ_ ˆ _, Dj_ on _a_ ) _D <_ . 0Then we have: _._ 5 _< A_ (ˆ _πj, Db_ ) for _j ∈_ [ _Nc_ ], 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2024_COLEP_Certifiably_Robust_Learning-Reasoning_Conformal_Prediction_via_Probabilistic_Circuits_images/iclr_2024_COLEP_Certifiably_Robust_Learning-Reasoning_Conformal_Prediction_via_Probabilistic_Circuits.pdf-0032-06.png)


where _pj_ 0 = P _Db_ [I[ _Y ̸_ = _j_ ]] and _pj_ 1 = P _Db_ [I[ _Y_ = _j_ ]] are class probabilities on benign distribution. 

_Proof of Thm. 4._ The following holds based on the definition of _C_[ˆ] _n,α_[COLEP] ( _·_ ) in Equation (8). 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2024_COLEP_Certifiably_Robust_Learning-Reasoning_Conformal_Prediction_via_Probabilistic_Circuits_images/iclr_2024_COLEP_Certifiably_Robust_Learning-Reasoning_Conformal_Prediction_via_Probabilistic_Circuits.pdf-0032-09.png)


Similarly, the following holds for _C_[ˆ] _n,α_ ( _·_ ). 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2024_COLEP_Certifiably_Robust_Learning-Reasoning_Conformal_Prediction_via_Probabilistic_Circuits_images/iclr_2024_COLEP_Certifiably_Robust_Learning-Reasoning_Conformal_Prediction_via_Probabilistic_Circuits.pdf-0032-11.png)


We can transform the objective of comparison of marginal prediction coverage into a comparison of non-conformity scores as follows: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2024_COLEP_Certifiably_Robust_Learning-Reasoning_Conformal_Prediction_via_Probabilistic_Circuits_images/iclr_2024_COLEP_Certifiably_Robust_Learning-Reasoning_Conformal_Prediction_via_Probabilistic_Circuits.pdf-0032-13.png)


Next, we individually bound the probability formulation in Equations (142) and (143). 

**Case (1)** : to bound P � _Sπ_ ˆ _Yn_ COLEP+1[( ˜] _[X][n]_[+1] _[,]_[ 1)] _[ ≤][S][π]_[ˆ] _[Y] n_ +1[( ˜] _[X][n]_[+1] _[,]_[ 1)] �. 

We first compute E � _Sπ_ ˆ _Yn_ COLEP+1[( ˜] _[X][n]_[+1] _[,]_[ 1)] _[ −][S][π]_[ˆ] _[Y] n_ +1[( ˜] _[X][n]_[+1] _[,]_[ 1)] � and then leverage tail bounds to derive the lower bound of the probability formulation. 

ˆ _Case (1a)_ : _πYn_ +1( _·_ ) correctly classifies the sample _X_[˜] _n_ +1 _∼Db_ (with probability _A_ (ˆ _πYn_ +1( _·_ ) _, Da_ )). Since we assume _**ϵ** j,c,D >_ 0 for _j ∈_ [ _Nc_ ], _c ∈{_ 0 _,_ 1 _}_ , and _D ∈{Da, Db}_ , _π_ ˆ _Y_[COLEP] _n_ +1[( ˜] _[X][n]_[+1][)][also] correctly classifies the sample _X_[˜] _n_ +1 _∼Db_ , and thus, we have the following: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2024_COLEP_Certifiably_Robust_Learning-Reasoning_Conformal_Prediction_via_Probabilistic_Circuits_images/iclr_2024_COLEP_Certifiably_Robust_Learning-Reasoning_Conformal_Prediction_via_Probabilistic_Circuits.pdf-0032-18.png)


32 

Published as a conference paper at ICLR 2024 

ˆ _Case (1b)_ : _πYn_ +1( _·_ )( _·_ ) wrongly classifies the sample _X_[˜] _n_ +1 (with probability 1 _− A_ (ˆ _πYn_ +1( _·_ ) _, Da_ )). We have the following: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2024_COLEP_Certifiably_Robust_Learning-Reasoning_Conformal_Prediction_via_Probabilistic_Circuits_images/iclr_2024_COLEP_Certifiably_Robust_Learning-Reasoning_Conformal_Prediction_via_Probabilistic_Circuits.pdf-0033-02.png)


Combining Equation (144) in case (1a) and Equation (145) in case (1b), we have: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2024_COLEP_Certifiably_Robust_Learning-Reasoning_Conformal_Prediction_via_Probabilistic_Circuits_images/iclr_2024_COLEP_Certifiably_Robust_Learning-Reasoning_Conformal_Prediction_via_Probabilistic_Circuits.pdf-0033-04.png)


Then from Equation (147), the following holds by applying Hoeffding’s inequality. 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2024_COLEP_Certifiably_Robust_Learning-Reasoning_Conformal_Prediction_via_Probabilistic_Circuits_images/iclr_2024_COLEP_Certifiably_Robust_Learning-Reasoning_Conformal_Prediction_via_Probabilistic_Circuits.pdf-0033-06.png)


Therefore, we have: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2024_COLEP_Certifiably_Robust_Learning-Reasoning_Conformal_Prediction_via_Probabilistic_Circuits_images/iclr_2024_COLEP_Certifiably_Robust_Learning-Reasoning_Conformal_Prediction_via_Probabilistic_Circuits.pdf-0033-08.png)


Similar to case (1), we first derive the bound for the expectation of the difference of non-conformity scores: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2024_COLEP_Certifiably_Robust_Learning-Reasoning_Conformal_Prediction_via_Probabilistic_Circuits_images/iclr_2024_COLEP_Certifiably_Robust_Learning-Reasoning_Conformal_Prediction_via_Probabilistic_Circuits.pdf-0033-10.png)


Then with Hoeffding’s inequality and the union bound, we have: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2024_COLEP_Certifiably_Robust_Learning-Reasoning_Conformal_Prediction_via_Probabilistic_Circuits_images/iclr_2024_COLEP_Certifiably_Robust_Learning-Reasoning_Conformal_Prediction_via_Probabilistic_Circuits.pdf-0033-12.png)


where _pj_ 0 = P _Db_ [I[ _Y ̸_ = _j_ ]] and _pj_ 1 = P _Db_ [I[ _Y_ = _j_ ]] are class probabilities on benign distribution. Combining Equations (149) and (152) and plugging them into Equations (142) and (143), we finally get: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2024_COLEP_Certifiably_Robust_Learning-Reasoning_Conformal_Prediction_via_Probabilistic_Circuits_images/iclr_2024_COLEP_Certifiably_Robust_Learning-Reasoning_Conformal_Prediction_via_Probabilistic_Circuits.pdf-0033-14.png)


33 

Published as a conference paper at ICLR 2024 

## H.3 PROOF OF THM. 5 

ˆ **Theorem 5** (Recall). Suppose that we evaluate the expectation of prediction accuracy of _πj_[COLEP] ( _·_ ) ˆ and _πj_ ( _·_ ) on _n_ samples drawn from _Dm_ and denote the prediction accuracy as _A_ (ˆ _πj_[COLEP] ( _·_ ) _, Dm_ ) and _A_ (ˆ _πj_ ( _·_ ) _, Dm_ ). Then we have: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2024_COLEP_Certifiably_Robust_Learning-Reasoning_Conformal_Prediction_via_Probabilistic_Circuits_images/iclr_2024_COLEP_Certifiably_Robust_Learning-Reasoning_Conformal_Prediction_via_Probabilistic_Circuits.pdf-0034-03.png)


ˆ _Proof of Thm. 5._ We can formulate the accuracy of _πj_ ( _·_ ) on _Dm_ as: 

_A_ (ˆ _πj_ ( _·_ ) _, Dm_ ) = P[ _Y_ = _j_ ]P[ˆ _πj_ ( _X_ ) _<_ 0 _._ 5 _|Y_ = _j_ ] + P[ _Y_ = _j_ ]P[ˆ _πj_ ( _X_ ) _≥_ 0 _._ 5 _|Y_ = _j_ ] (155) Similarly, _A_ (ˆ _πj_[COLEP] ( _·_ ) _, Dm_ ) can be formulated as: 

_A_ (ˆ _πj_[COLEP] ( _·_ ) _, Dm_ ) = P[ _Y_ = _j_ ]P[ˆ _πj_[COLEP] ( _X_ ) _<_ 0 _._ 5 _|Y_ = _j_ ] + P[ _Y_ = _j_ ]P[ˆ _πj_[COLEP] ( _X_ ) _≥_ 0 _._ 5 _|Y_ = _j_ ] (156) 

ˆ ˆ Let ∆ _j_ ( _·_ ) := _πj_[COLEP] ( _·_ ) _− πj_ ( _·_ ). From the results in Lemma 5.1, we have E[∆ _j_ ( _X_ ) _|Y_ = _j_ ] _≤ −_ _**ϵ** j,_ 0 _,D_ and E[∆ _j_ ( _X_ ) _|Y_ = _j_ ] _≥_ _**ϵ** j,_ 1 _,D_ . Note that we have _**ϵ** j,_ 0 _,D >_ 0 and _**ϵ** j,_ 1 _,D >_ 0 by assumption. 

Then by rearranging the terms and Hoeffding’s inequality, we can derive as follows: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2024_COLEP_Certifiably_Robust_Learning-Reasoning_Conformal_Prediction_via_Probabilistic_Circuits_images/iclr_2024_COLEP_Certifiably_Robust_Learning-Reasoning_Conformal_Prediction_via_Probabilistic_Circuits.pdf-0034-09.png)


Similarly, we consider the case conditioned on _Y_ = _j_ . 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2024_COLEP_Certifiably_Robust_Learning-Reasoning_Conformal_Prediction_via_Probabilistic_Circuits_images/iclr_2024_COLEP_Certifiably_Robust_Learning-Reasoning_Conformal_Prediction_via_Probabilistic_Circuits.pdf-0034-11.png)


Note that from Equations (160) and (164), we have the following relation: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2024_COLEP_Certifiably_Robust_Learning-Reasoning_Conformal_Prediction_via_Probabilistic_Circuits_images/iclr_2024_COLEP_Certifiably_Robust_Learning-Reasoning_Conformal_Prediction_via_Probabilistic_Circuits.pdf-0034-13.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2024_COLEP_Certifiably_Robust_Learning-Reasoning_Conformal_Prediction_via_Probabilistic_Circuits_images/iclr_2024_COLEP_Certifiably_Robust_Learning-Reasoning_Conformal_Prediction_via_Probabilistic_Circuits.pdf-0034-14.png)


Finally, by considering both the adversarial distribution and benign distribution, we can conclude that: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2024_COLEP_Certifiably_Robust_Learning-Reasoning_Conformal_Prediction_via_Probabilistic_Circuits_images/iclr_2024_COLEP_Certifiably_Robust_Learning-Reasoning_Conformal_Prediction_via_Probabilistic_Circuits.pdf-0034-16.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2024_COLEP_Certifiably_Robust_Learning-Reasoning_Conformal_Prediction_via_Probabilistic_Circuits_images/iclr_2024_COLEP_Certifiably_Robust_Learning-Reasoning_Conformal_Prediction_via_Probabilistic_Circuits.pdf-0034-17.png)


**Theorem 7** (Lower Bound of _A_ (ˆ _πj_[COLEP] _, Dm_ )) **.** _Let Ej,d[COLEP]_ = E _Dm_ [ˆ _πj[COLEP]_ ( _X_ ) _|_ I[ _Y_ = _j_ ] = _d_ ] _for d ∈{_ 0 _,_ 1 _}. Assume that the class probability computed by the r-th PC π_ ˆ _j,r[COLEP] is conditinally_ 

34 

Published as a conference paper at ICLR 2024 

_independent of π_ ˆ _j,r[COLEP] −_ 1 _[for]_[ 2] _[ ≤][r][≤][R][ and][ E] j,[COLEP]_ 0 _<_ 0 _._ 5 _< Ej,[COLEP]_ 1 _. Assume that we aggregate PCs by taking the average (i.e., βr_ = 1 _/R). Then we have:_ 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2024_COLEP_Certifiably_Robust_Learning-Reasoning_Conformal_Prediction_via_Probabilistic_Circuits_images/iclr_2024_COLEP_Certifiably_Robust_Learning-Reasoning_Conformal_Prediction_via_Probabilistic_Circuits.pdf-0035-02.png)


_Proof sketch._ We construct a sequence _{Jr}[R] r_ =1[in][a][specific][way][and][prove][that][it][is][a][super-] martingale or a sub-martingale with bounded differences conditioned based on the true label. Then we use Azuma’s inequality to bound the objective _A_ (ˆ _πj_[COLEP] _, Dm_ ) in different cases and combine them with union bound. 

_Remarks._ Lemma 5.1 indicates that better utility of knowledge models and implication rules can indicate a smaller _Ej,_[COLEP] 0 and larger _Ej,_[COLEP] 1 , which can further imply a tighter lower bound of prediction accuracy according toThm. 7. Furthermore, according to the proof, we know that _Ej,_[COLEP] 0 negatively correlates with the number of PCs _R_ and _Ej,_[COLEP] 1 has a positive correlation with _R_ , indicating that more PCs (i.e., knowledge rules) implies a better prediction accuracy. 

> _Proof of Thm. 7._ Consider _R_ PCs. We construct a sequence _{Jr}[R] r_ =1[such][that] _[J][r]_[(] _[X]_[)] = 1 _r_ � _q∈_ [ _r_ ] _[π]_[ˆ] _j_[(] _[q]_[)][(] _[X]_[)][, where] _[π]_[ˆ] _j_[(] _[q]_[)][(] _[·]_[)][ is defined as Equation (24).] 

ˆ We first consider the case conditioned on _Y_ = _j_ . Without loss of generality, assume that _πj_[(1)][(] _[X]_[)] _[ ≥] π_ ˆ _j_[(2)][(] _[X]_[)] _[≥· · ·][≥][π]_[ˆ] _j_[(] _[R]_[)] ( _X_ ). Then we can prove _{Jr}[R] r_ =1[is a super-martingale since we have the] following: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2024_COLEP_Certifiably_Robust_Learning-Reasoning_Conformal_Prediction_via_Probabilistic_Circuits_images/iclr_2024_COLEP_Certifiably_Robust_Learning-Reasoning_Conformal_Prediction_via_Probabilistic_Circuits.pdf-0035-07.png)


_{Jr}[R] r_ =1[also has bounded difference since we have the following:] 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2024_COLEP_Certifiably_Robust_Learning-Reasoning_Conformal_Prediction_via_Probabilistic_Circuits_images/iclr_2024_COLEP_Certifiably_Robust_Learning-Reasoning_Conformal_Prediction_via_Probabilistic_Circuits.pdf-0035-09.png)


Also note that we have: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2024_COLEP_Certifiably_Robust_Learning-Reasoning_Conformal_Prediction_via_Probabilistic_Circuits_images/iclr_2024_COLEP_Certifiably_Robust_Learning-Reasoning_Conformal_Prediction_via_Probabilistic_Circuits.pdf-0035-11.png)


35 

Published as a conference paper at ICLR 2024 

Therefore, _{Jr}[R] r_ =1[is conditionally independent.][Then by using Azuma’s inequality, we can derive] as follows: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2024_COLEP_Certifiably_Robust_Learning-Reasoning_Conformal_Prediction_via_Probabilistic_Circuits_images/iclr_2024_COLEP_Certifiably_Robust_Learning-Reasoning_Conformal_Prediction_via_Probabilistic_Circuits.pdf-0036-02.png)


1 Inequality ( _a_ ) holds because the inequality[�] _[R] q_ =1 _q_[2] _[<][π]_ 6[2][holds.][Finally, the following holds.] 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2024_COLEP_Certifiably_Robust_Learning-Reasoning_Conformal_Prediction_via_Probabilistic_Circuits_images/iclr_2024_COLEP_Certifiably_Robust_Learning-Reasoning_Conformal_Prediction_via_Probabilistic_Circuits.pdf-0036-04.png)


Then we consider the case conditioned on _Y_ = _j_ . Similarly, we construct a sequence _{Jr[′][}][R] r_ =1 such that _Jr[′]_[(] _[X]_[)][=][1] _r_ � _q∈_ [ _r_ ] _[π]_[ˆ] _j_[(] _[q]_[)][(] _[X]_[)][, where] _[π]_[ˆ] _j_[(] _[q]_[)][(] _[·]_[)][ is defined as Equation (24).][Without loss of] generality, assume that _π_ ˆ _j_[(1)][(] _[X]_[)] _[≤][π]_[ˆ] _j_[(2)][(] _[X]_[)] _[≤· · ·][≤][π]_[ˆ] _j_[(] _[R]_[)] ( _X_ ). Then we can prove _{Jr[′][}][R] r_ =1[is a] sub-martingale since we have the following: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2024_COLEP_Certifiably_Robust_Learning-Reasoning_Conformal_Prediction_via_Probabilistic_Circuits_images/iclr_2024_COLEP_Certifiably_Robust_Learning-Reasoning_Conformal_Prediction_via_Probabilistic_Circuits.pdf-0036-06.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2024_COLEP_Certifiably_Robust_Learning-Reasoning_Conformal_Prediction_via_Probabilistic_Circuits_images/iclr_2024_COLEP_Certifiably_Robust_Learning-Reasoning_Conformal_Prediction_via_Probabilistic_Circuits.pdf-0036-07.png)


Again, we can show that _{Jr[′][}][R] r_ =1[has bounded difference:] 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2024_COLEP_Certifiably_Robust_Learning-Reasoning_Conformal_Prediction_via_Probabilistic_Circuits_images/iclr_2024_COLEP_Certifiably_Robust_Learning-Reasoning_Conformal_Prediction_via_Probabilistic_Circuits.pdf-0036-09.png)


Also note that we have: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2024_COLEP_Certifiably_Robust_Learning-Reasoning_Conformal_Prediction_via_Probabilistic_Circuits_images/iclr_2024_COLEP_Certifiably_Robust_Learning-Reasoning_Conformal_Prediction_via_Probabilistic_Circuits.pdf-0036-11.png)


36 

Published as a conference paper at ICLR 2024 

Therefore, _{Jr[′][}][R] r_ =1[is conditionally independent.][Then by using Azuma’s inequality, we can derive] as follows: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2024_COLEP_Certifiably_Robust_Learning-Reasoning_Conformal_Prediction_via_Probabilistic_Circuits_images/iclr_2024_COLEP_Certifiably_Robust_Learning-Reasoning_Conformal_Prediction_via_Probabilistic_Circuits.pdf-0037-02.png)


Finally, the following holds. 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2024_COLEP_Certifiably_Robust_Learning-Reasoning_Conformal_Prediction_via_Probabilistic_Circuits_images/iclr_2024_COLEP_Certifiably_Robust_Learning-Reasoning_Conformal_Prediction_via_Probabilistic_Circuits.pdf-0037-04.png)


Note that we can formulate _A_ (ˆ _πj_[COLEP] _, Dm_ ) as: 

_A_ (ˆ _πj_[COLEP] _, Dm_ ) = P[ _Y_ = _j_ ]P[ˆ _πj_[COLEP] ( _X_ ) _<_ 0 _._ 5 _|Y_ = _j_ ] + P[ _Y_ = _j_ ]P[ˆ _πj_[COLEP] ( _X_ ) _>_ 0 _._ 5 _|Y_ = _j_ ] (196) 

Combining Equations (182) and (195), we have: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2024_COLEP_Certifiably_Robust_Learning-Reasoning_Conformal_Prediction_via_Probabilistic_Circuits_images/iclr_2024_COLEP_Certifiably_Robust_Learning-Reasoning_Conformal_Prediction_via_Probabilistic_Circuits.pdf-0037-08.png)


## J EXPERIMENT 

J.1 IMPLEMENTATION DETAILS 

**Dataset & Model.** We evaluate COLEP on three datasets, GTSRB, CIFAR-10, and AwA2. GTSRB consists of 12 types of German road signs, such as “Stop” and “Speed Limit 20”. CIFAR-10 includes 10 categories of animals and transportation. AwA2 contains 50 animal categories. We use GTSRBCNN architecture (Eykholt et al., 2018), ResNet-56 (He et al., 2016), and ResNet-50 (He et al., 2016) as the architecture of models on them, respectively. For fair comparisons, we use the same model architecture and training smoothing factors for models in COLEP and SOTA baselines CP and RSCP. We perform smoothed training (Cohen et al., 2019) with the same smoothing factor _σ_ for models. We use the official validation set of GTSRB including 973 samples, and randomly select 1000 samples from the test set of CIFAR-10 and AwA2 as the calibration sets for conformal prediction with the nominal level 1 _− α_ = 0 _._ 9 across evaluations. The evaluation is performed on the test set of GTSRB and the remaining samples of the test set of CIFAR-10 and GTSRB. GTSRB and CIFAR-10 contain images with the size of 32 _×_ 32. AwA2 contains images with the size of 224 _×_ 224. GTSRB and CIFAR-10 are under the MIT license. AwA2 is under the Creative Commons Attribution-NonCommercial-ShareAlike (CC BY-NC-SA) license. 

**Implementation details.** We run evaluations using three different random seeds and report the averaged results. We omit the standard deviation of the results since we observe they are of low variance due to a large sample size used for calibration and testing. All the evaluation is done on a single _A_ 6000 GPU. In the evaluation of certified coverage, we compute the smoothed score or prediction probability 100 _,_ 000 times for randomized smoothing and fix the ratio of perturbation bound and the smoothing factor during certification _δ/σcer_ as 0 _._ 5 for both RSCP and COLEP. In the evaluation of marginal coverage and set size under PGD attack, we use the attack objective of 

37 

Published as a conference paper at ICLR 2024 

cross-entropy loss for RSCP. For COLEP, the objective is the same cross-entropy loss for the main model and the binary cross-entropy loss for the knowledge models. 

**Knowledge Rules & Reasoning** . In summary, we have 3 PCs and 28 knowledge rules in GTSRB, 3 PCs and 30 knowledge rules in CIFAR-10, and 4 PCs and 187 knowledge rules in AwA2. Based on the domain knowledge in each dataset, we encode a set of implication rules into PCs. In each PC, we encode a type of implication knowledge with disjoint attributes. In GTSRB, we have a PC of shape knowledge (e.g.,“octagon”, “square”), a PC of boundary color knowledge (e.g., “red boundary”, “black boundary”), and a PC of content knowledge (e.g., “digit 50”, “Turn Left”). In CIFAR-10, we have a PC of category knowledge (e.g., “animal”, “transportation”), a PC of space knowledge (e.g., “in the sky”, “in the water”, “on the ground”), and a PC of feature knowledge (e.g., “has legs”, “has wheels”). In AwA2, we have a PC of superclass knowledge (e.g., “procyonid”, “big cat”), a PC of predation knowledge (e.g., “fish”, “meat”, “plant”), a PC of appearance (e.g., “big ears”, “small eyes”), and a PC of fur color knowledge (e.g., “green”, “gray”, “white”). 

For example, in the PC of shape knowledge in GTSRB, we can encode the implication rules related to the shape knowledge such as IsStopSign = _⇒_ IsOctagon (stop signs are octagon) with weight _w_ 1, IsSpeedLimit = _⇒_ IsSquare (speed limit signs are square) with weight _w_ 2, and IsTurnLeft = _⇒_ IsRound (turn left signs are round) with weight _w_ 3. To implement the COLEP learning-reasoning framework, we follow the steps to encode the rules into PCs. (1) We learn the main model and knowledge models implemented with DNNs to estimate a preliminary probability ˆ for _Nc_ class labels and _L_ concept labels (e.g., _p_ (IsSpeedLimit = 1) = _πsp, p_ (IsSquare = 1) = _π_ ˆ _sq_ ). (2) We compute the factor value _F_ ( _µ_ ) for any possible assignment _µ ∈{_ 0 _,_ 1 _}[N][c]_[+] _[L]_ as _F_ ( _µ_ ) = exp _{_[�] _[H] h_ =1 _[w][h]_[I][[] _[µ][∼][K] h_[]] _[}]_[ (e.g.,] _[F]_[([1] _[,]_[ 1] _[,]_[ 1] _[,]_[ 1] _[,]_[ 1] _[,]_[ 1]) =] _[ e][w]_[1][+] _[w]_[2][+] _[w]_[3][since the assignment satisfy] all three knowledge rules above). (3) We construct a three-layer PC as shown in Figure 1: (a) the bottom layer consists of leaf nodes representing Bernoulli distributions of all class and concept ˆ _π_ labelsˆ _sq_ ) _, ..., F_ and([1the _,_ 1constant _,_ 1 _,_ 1 _,_ 1 _,_ 0])factor _, F_ ([1values _,_ 1 _,_ 1 _,_ 1of _,_ 1assignments _,_ 1]) _, ..._ ), (b) the second layer consists of product nodes(e.g., _B_ (ˆ _πsp_ ) _, B_ (1 _− πsp_ ) _, B_ (ˆ _πsq_ ) _, B_ (1 _−_ computing the likelihood of each assignment by multiplying the likelihood of the instantiations of Bernoulli variables and the correction factor values (e.g., _π_ ˆ _sp × π_ ˆ _sq × ... × F_ ([1 _,_ 1 _,_ 1 _,_ 1 _,_ 1 _,_ 1])), and (c) the top layer is a sum node computing the summation of the products. (4) We linearly combine multiple PCs that encode different types of knowledge rules (e.g., 3 PCs in GTSRB: PCs of shape/color/content knowledge). We compute _βk_ based on the utility of the _k_ -th PC on the calibration set and manually set the weights of knowledge rules based on their utility. 

Codes are available at https://github.com/kangmintong/COLEP. 

## J.2 ADDITIONAL EVALUATION OF COLEP 

Table 1: Comparison of the runtime (millisecond) of COLEP per forward time between a standard model and COLEP. The evaluation is done on RTX A6000 GPUs. 

||GTSRB<br>CIFAR-10<br>AwA2|
|---|---|
|||
|standard model<br>COLEP|13.2<br>15.5<br>35.2<br>13.5<br>15.7<br>35.6|



**Visualization of the contribution of different knowledge rules** . In GTSRB, we consider the contribution of the shape knowledge rules, color knowledge rules, and content knowledge rules. In CIFAR-10, we consider the contribution of the category knowledge rules, space knowledge rules, and feature knowledge rules. More details of the meanings and examples of knowledge rules are provided in Appendix J.1. We provide the visualization of contributions of different types of knowledge rules in Figure 5. The results show that the rules related to the color knowledge in GTSRB and space knowledge in CIFAR-10 benefit the certified coverage more, but collectively applying all the well-designed knowledge rules leads to even better coverage. The effectiveness of these types of knowledge attributes to a higher utility of the knowledge models (i.e., the accuracy of concept detection). Concretely, the feature of these concepts is relatively easy to learn and can be more accurately detected (e.g., color concepts are more easily learned than shape concepts). The correlation between the knowledge model utility and effectiveness of COLEP is also theoretically analyzed and observed in Section 5. 

38 

Published as a conference paper at ICLR 2024 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2024_COLEP_Certifiably_Robust_Learning-Reasoning_Conformal_Prediction_via_Probabilistic_Circuits_images/iclr_2024_COLEP_Certifiably_Robust_Learning-Reasoning_Conformal_Prediction_via_Probabilistic_Circuits.pdf-0039-01.png)


**----- Start of picture text -----**<br>
0.06 0.07<br>0.05 0.06<br>0.05<br>0.04<br>0.04<br>0.03<br>0.03<br>0.02<br>0.02<br>0.01 0.01<br>0 0<br>Shape Color Content All Category Space Feature All<br>(a) GTSRB (b) CIFAR-10<br>**----- End of picture text -----**<br>


Figure 5: Certified coverage of COLEP under bounded perturbation _δ_ = 0 _._ 25 on GTSRB and CIFAR10 with different types of knowledge rules. 

**Runtime evaluation of COLEP** . Since the knowledge models in the learning component are simple neural networks and can perform forward passes in parallelism, the additional computational complexity of COLEP comes mainly from the overhead of the reasoning component (i.e., PC). Given a PC defined on the directed graph ( _V, E_ ), the time complexity of probabilistic inference is _O_ ( _|V|_ ), according to the details illustrated in Appendix C.1. Therefore, the additional time complexity of COLEP is _O_ ( _|V|_ ), linear to the number of nodes in PC graph. We empirically evaluate the runtime of a forward pass of COLEP and compare it with a standard model. The results in Table 1 show that COLEP only achieves a little higher runtime compared with a single standard model. 

Table 2: Marginal coverage / average set size of RSCP and COLEP under AutoAttack with _ℓ_ 2 bounded perturbation of 0 _._ 25 on GTSRB, CIFAR-10, and AwA2. The nominal level 1 _− α_ is 0.9. 

||GTSRB<br>CIFAR-10<br>AwA2|
|---|---|
|||
|RSCP<br>COLEP|0.9682 / 2.36<br>0.9102 / 3.56<br>0.923 / 5.98<br>**0.9702**/**2.13**<br>**0.9232**/**2.55**<br>**0.952**/**4.57**|



Table 3: Marginal coverage / average set size of RSCP and COLEP under PGD Attack with multiple desired coverage levels 1 _− α_ with _ℓ_ 2 bounded perturbation of 0 _._ 25 on GTSRB. 

||1_−α_= 0_._85<br>1_−α_= 0_._9<br>1_−α_= 0_._95|
|---|---|
|||
|RSCP<br>COLEP|0.9029 / 1.63<br>0.9682 / 2.36<br>0.9829 / 2.96<br>**0.9053**/**1.34**<br>**0.9702**/**2.13**<br>**0.9859**/**2.53**|



**Evaluations with different types of adversarial attacks** . To evaluate the robustness of COLEP under various types of attacks, we further consider the AutoAttack method Croce & Hein (2020), which uses an ensemble of four diverse attacks to reliably evaluate the robustness: (1) APGD-CE attack, (2) APGD-DLR attack, (3) FAB attack, and (4) square attack. APGD-CE and APGD-DLR attacks are step size-free versions of PGD attacks with cross-entropy loss and DLR loss, respectively. FAB attack optimizes the adversarial sample by minimizing the norm of the adversarial perturbations. Square attack is a query-efficient black-box attack. We evaluate the prediction coverage and averaged set size of COLEP and SOTA method RSCP with _ℓ_ 2 bounded perturbation of 0 _._ 25 on GTSRB, CIFAR-10, and AwA2. The desired coverage is 1 _− α_ = 0 _._ 9 in the evaluation. The results in Table 2 show that under diverse and stronger adversarial attacks, COLEP still achieves higher marginal coverage than the nominal level 0 _._ 9 and also demonstrates a better prediction efficiency (i.e., smaller size of prediction sets) than RSCP. 

**Evaluations with various nominal levels** . We further evaluate COLEP with multiple nominal levels of coverage 1 _− α_ . We provide the results with multiple coverage levels 1 _− α_ in Table 3. The 

39 

Published as a conference paper at ICLR 2024 

results demonstrate that for different parameters of _α_ , (1) COLEP always achieves a higher marginal coverage than the nominal level, indicating the validity of the certification in COLEP, and (2) COLEP shows a better trade-off between the prediction coverage and prediction efficiency (i.e., average set size) than the SOTA baseline RSCP. 

**Selection of weights of knowledge rules.** The weight _w_ of logical rules is specified based on the utility of the corresponding knowledge rule. In Section 6, we theoretically show that the robustness of COLEP correlates with the utility of knowledge rules regarding the uniqueness of the rule, defined in Equation (17). Specifically, a universal knowledge rule which generally holds for almost all class labels is of low quality and does not benefit the robustness and should be assigned with a low weight _w_ . For example, if all the traffic signs in the dataset are of octagon shape, then the implication rule _IsStopSign_ = _⇒ IsOctagon_ should be useless and assigned with a low weight _w_ , which also aligns with the intuition. In the empirical evaluations, since we already designed knowledge rules of high quality (i.e., good uniqueness) as shown in Section 7, we do not need to spend great efforts selecting weights for each knowledge rule individually. We assume that all the knowledge rules have the same weight _w_ and select a proper weight _w_ for all the knowledge rules via grid searching. The results inspire us to fix _w_ as 1 _._ 5, and the evaluations demonstrate that it is sufficient to achieve good robustness in different settings. One can also model the weight selection problem as a combinatorial optimization, which is non-trivial but can better unleash the power of COLEP. We leave a more intelligent and automatic approach of knowledge weight selection for future work. 

## K OVERVIEW OF NOTATIONS 

In this section, we provide overviews of used notations for a better understanding of the analysis of COLEP. Specifically, we provide an overview of notations related to models in Table 4, notations related to conformal prediction in Table 5, notations related to certification in Section 4 in Table 6, and notations related to analysis in Section 5 in Table 4. 

Table 4: Notations related to models in COLEP. 

|Notation||Meaning|
|---|---|---|
|_Nc_||number of class labels|
|_j_||_j ∈_[_Nc_]to index the output of the main model|
|_L_||number of concept labels, number of knowledge models|
|_l_||_l ∈_[_L_]to index_L_knowledge models/concept labels|
|ˆ_πj_(_x_)|the output likelihood of the main model for the_j_-th class label (_j ∈_[_Nc_]) given input_x_||
|ˆ_π_(_l_)(_x_)||the output likelihood of the_l_-th knowledge model (_l ∈_[_L_]) given input_l_|
|ˆ_π_|ˆ_π_ :=|�<br>[ˆ_πj_]_j∈_[_Nc_]_,_[ˆ_π_(_l_)]_l∈_[_L_]<br>�<br>_∈_R_Nc_+_L_, concatenated vector of class and concept probabilities|
|_j∀_|_j∀∈_[_Nc_+_L_]to index ˆ_π_(the concatenation of the output of the main model and concept models)||
|ˆ_πj∀_(_x_)|the output likelihood of the_j_-th class label (_j∀∈_[_Nc_+_L_])**before the reasoning component**||
|ˆ_π_COLEP<br>_j_<br>(_x_)<br>_{Kh}H_<br>_h_=1|the output likelihood of the_j_-th class label (_j ∈_[_Nc_])**after the reasoning component**<br>_H_ propositional logic rules||
|_wh_||the weight of the_h_-th logic rule (_h ∈_[_H_])|
|I[_·_]||indicator function|
|_µ_||_µ ∈M_ :=_{_0_,_1_}Nc_+_L_, the index variable in the feasible assignment set_M_|
|_F_(_µ_)||_F_(_µ_) = exp_{_�_H_<br>_h_=1 _wh_I[_µ∼Kh_]_}_, factor value of the assignment_µ_|
|_T_(_a, b_)||_T_(_a, b_) := log(_ab_+ (1_−a_)(1_−b_)), introduced for ease of notation|
|_O_leaf_, O_root||leaf node and root node in the PC|
|_R_||number of probabilistic circuits|
|_r_||_r ∈_[_R_]to index the_r_-th PC|
|_βr_||the coeffcient of the_r_-th PC|



40 

Published as a conference paper at ICLR 2024 

Table 5: Notations related to conformal predictions. 

|Notation||||Meaning|
|---|---|---|---|---|
|_n_||||number of calibration data samples|
|(_Xi, Yi_)|||the|_i_-th data sample with input_Xi_ and label_Yi_ (_i ∈_[_n_])|
|(_Xn_+1_, Yn_+1)||||the test sample|
|_Y_||||label space[_Nc_] :=_{_1_,_2_, ..., Nc}_|
|_y_||||_y ∈Y_ to index labels in the label space_Y_|
|_u_|||random variable draw from uniform distribution_U_(0_,_1)||
|_Q_1_−α_||||1_−α_quantile value|
|_S_ˆ_π_(_x, y_)|APS non-conformity score of sample(_x, y_)given probability estimate ˆ_π_||||
|_S_ˆ_π_COLEP<br>_j_<br>(_x, y_)<br>ˆ_Cn,α_(_x_)|APS score with ˆ_π_COLEP<br>_j_<br>(output likelihood of the_j_-th class label after the reasoning component)<br>ˆ_Cn,α_(_x_)_⊆Y_ is the conformal prediction set of input_x_**without COLEP**||||
|ˆ_C_COLEP_j_<br>_n,αj_<br>(_x_)|ˆ_C_COLEP_j_<br>_n,αj_|(_x_)||_⊆{_0_,_1_}_is the conformal prediction set of the_j_-th class label|
|ˆ_C_COLEP<br>_n,α_<br>(_x_)|ˆ_C_COLEP<br>_n,α_||(_x_)|_⊆Y_ is the conformal prediction set of input_x_**with COLEP**|
|_qy_||||_qy ∈{_0_,_1_}_to index in label space_{_0_,_1_}_|



Table 6: Notations related to certification in Section 4. 

|Notation||Meaning|
|---|---|---|
||||
|ˆ_πj∀_(_x_)|upper bound of ˆ_πj∀_(_x_)|(likelihood**without reasoning component**) with bounded perturbation|
|ˆ_πj∀_<br>(_x_)|lower bound of ˆ_πj∀_(_x_)|(likelihood**without reasoning component**) with bounded perturbation|
|U[ˆ_π_COLEP<br>_j_<br>(_x_)]|upper bound of ˆ_π_COLEP<br>_j_|(_x_)(likelihood**with reasoning component**) with bounded perturbation|
|L[ˆ_π_COLEP<br>_j_<br>(_x_)]<br>_V j_<br>_d_|lower bound of ˆ_π_COLEP<br>_j_<br>(_x_)(likelihood**with reasoning component**) with bounded perturbation<br>index set of conditional variables (left operand of rule_A_ =_⇒B_) in the PC except for_j_||
|_V j_<br>_s_|index set of conditional variables (right operand of rule_A_ =_⇒B_) in the PC except for_j_||
|_µj_||the_j_-th element of assignment_µ_|
|_ϵ_||adversarial perturbation|
|_δ_||bound of perturbations such that_∥ϵ∥_2 _≤δ_|
|˜_Xn_+1||˜_Xn_+1 =_Xn_+1+_ϵ_adversarial test sample|
|ˆ_C_COLEP_δ_<br>_n,α_<br>(˜_Xn_+1)||certifed prediction set with perturbation bound_δ_|
|_S_ˆ_π_<br>COLEP_δ_<br>_j_|worst-case non-conformity score for the_j_-th class label with bounded perturbation_δ_||
|_τ_ COLEPcer|lower bound of the coverage guarantee within bounded perturbation||
|_τ_ COLEPcer<br>_j_|lower bound of the coverage guarantee of the_j_-th class label within bounded perturbation||



Table 7: Notations related to analysis in Section 5. 

|_Db_|benign data distribution|
|---|---|
|_Da_|adversarial data distribution|
|_Dm_|mixed data distribution|
|_pD_|portion of distribution_D_in_Dm_|
|_Gr_ = (_Vr, Er_)|undirected graph encoded with rules in the_r_-th PC|
|_Ar_(_j∀_)|node set in connected components of node_j∀_in graph_Gr_|
|_Ar,d_(_j∀_)|node set of conditional variables in connected components of node_j∀_in graph_Gr_|
|_Ar,s_(_j∀_)|node set of consequence variables in connected components of node_j∀_in graph_Gr_|
|_ν_(_x_)|boolean vector of ground truth class and concepts given sample_x_as input|
|_tj∀,D_|the threshold of confdence for the_j∀_-th label|
|_zj∀,D_|probability matching this threshold for the_j∀_-th label|
|_T_ (_r_)<br>_j,D_<br>_Z_(_r_)<br>_j,D_|multiplication of_tj∀,D_ in the_r_-th PC<br>multiplication of_zj∀,D_ in the_r_-th PC|
|_Pd_(_j_)_, Ps_(_j_)|the set of PC index where_j_ appears as conditional variables and consequence variables|
|_U_ (_r_)<br>_j_|utility of associated knowledge rules within the_r_-th PC for the_j_-th class|
|_ϵj,_0,_ϵj,_1|characterization of probability correction by COLEP conditioned on ground truth label0and1|
|**_ϵ_**_j,_0_,D_|**_ϵ_**_j,_0_,D_ =E(_X,Y_)_∼D_[_ϵj,_0_|νj_ = 0], expectation of_ϵj,_0|
|**_ϵ_**_j,_1_,D_|**_ϵ_**_j,_1_,D_ =E(_X,Y_)_∼D_[_ϵj,_1_|νj_ = 1], expectation of_ϵj,_1|
|_A_(ˆ_π, D_)|accuracy of probability estimator ˆ_π_on data distribution_D_|



41 

