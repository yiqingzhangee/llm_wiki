# **Zero-shot causal learning** 

**Hamed Nilforoshan** _[∗]_[1] **Michael Moor** _[∗]_[1] **Yusuf Roohani**[2] **Yining Chen**[1] **Anja Šurina**[3] **Michihiro Yasunaga**[1] **Sara Oblak**[4] **Jure Leskovec**[1] 

1Department of Computer Science, Stanford University 

2Department of Biomedical Data Science, Stanford University 

3Department of Neuroinformatics, ETH Zürich 

4Department of Computer Science, University of Ljubljana 

Correspondence to: hamedn@cs.stanford.edu, mdmoor@cs.stanford.edu, jure@cs.stanford.edu 

## **Abstract** 

Predicting how different interventions will causally affect a specific individual is important in a variety of domains such as personalized medicine, public policy, and online marketing. There are a large number of methods to predict the effect of an existing intervention based on historical data from individuals who received it. However, in many settings it is important to predict the effects of novel interventions ( _e.g._ , a newly invented drug), which these methods do not address. Here, we consider zero-shot causal learning: predicting the personalized effects of a novel intervention. We propose CaML, a causal meta-learning framework which formulates the personalized prediction of each intervention’s effect as a task. CaML trains a single meta-model across thousands of tasks, each constructed by sampling an intervention, along with its recipients and nonrecipients. By leveraging both intervention information ( _e.g._ , a drug’s attributes) and individual features ( _e.g._ , a patient’s history), CaML is able to predict the personalized effects of novel interventions that do not exist at the time of training. Experimental results on real world datasets in large-scale medical claims and cell-line perturbations demonstrate the effectiveness of our approach. Most strikingly, CaML’s zero-shot predictions outperform even strong baselines trained directly on data from the test interventions. 

## **1 Introduction** 

Personalized predictions about how an intervention will causally affect a specific individual are important across many high impact applications in the physical, life, and social sciences. For instance, consider a doctor deciding whether or not to prescribe a drug to a patient. Depending on the patient, the same drug could either (a) cure the disease, (b) have no effect, or (c) elicit a life-threatening adverse reaction. Predicting which effect the drug will have for each patient could revolutionize healthcare by enabling personalized treatments for each patient. 

The causal inference literature formalizes this problem as conditional average treatment effects (CATE) estimation, in which the goal is to predict the effect of an intervention, conditioned on patient characteristics ( _X_ ). When natural experiment data is available, consisting of individuals who already did and did not receive an intervention, a variety of CATE estimators exist to accomplish this task [1, 2, 14, 21, 28, 32, 42, 53, 34, 66]. These methods can then predict the effect of an _existing_ intervention ( _W_ ) on a new individual ( _X[′]_ ). 

However, in many real-world applications natural experiment data is entirely unavailable, and yet CATE estimation is critical. For instance, when new drugs are discovered, or new government policies are passed, it is important to know the effect of these novel interventions on individuals and subgroups 

*Equal contribution. Preprint. Under review. 

in advance, _i.e._ , before anybody is treated. There is thus a need for methods that can predict the effect of a _novel_ intervention ( _W[′]_ ) on a new individual ( _X[′]_ ) in a zero-shot fashion, _i.e._ , without relying on _any_ historical data from individuals who received the intervention. 

Generalizing to novel interventions is especially challenging because it requires generalizing across two dimensions simultaneously: to new interventions and new individuals. This entails efficiently “aligning” newly observed interventions to the ones previously observed in the training data. A zero-shot causal learning framework thus requires drawing analogies between the properties of interventions, and linking these to the features and outcomes of individual samples. 

**Present work.** Here, we propose CaML ( **Ca** usal **M** eta- **l** earning), a general framework for training a single meta-model to estimate CATE across many interventions, including novel interventions that did not exist at the time of model training (Figure 1). Our key insight is to frame CATE estimation for each intervention as a separate meta-learning task. For each task observed during training, we sample a retrospective natural experiment of individuals who did and who did not receive the intervention. This natural experiment data is used to estimate the effect of the intervention for each individual (using any off-the-shelf CATE estimator), which serves as the training target for the task. 

In order to achieve zero-shot generalization to new interventions, we include information ( _W_ ) about the intervention ( _e.g.,_ a drug’s attributes), in the task. We then train a single meta-model which fuses intervention features with individual-level features ( _X_ ) to predict the intervention’s effect. Our approach allows us to predict the causal effect of novel interventions, _i.e._ interventions without sample-level training data, such as a newly discovered drug (Figure 1). We refer to this capability as _zero-shot causal learning_ . 

Experiments across different real-world settings show that CaML is both scalable and effective, including the application to a large-scale medical dataset featuring tens of millions of patients. Most strikingly, CaML’s zero-shot performance exceeds even strong baselines that were trained directly on data from the test interventions. We further find that CaML is capable of zero-shot generalization even under challenging conditions: when trained only on single interventions, at inference time it can accurately predict the effect of combinations of novel interventions. Finally, we explain these findings, by proving a zero-shot generalization bound. 

## **2 Related Work** 

We discuss recent work which is most closely related to zero-shot causal learning, and provide an extended discussion of other related work in Appendix B. Most CATE estimators do not address novel interventions, requiring that all considered interventions be observed during training. A notable exception is recent methods which estimate CATE for an intervention using structured information about its attributes [23, 33]. In principle, these methods can also be used for zeroshot predictions. These methods estimate CATE directly from the raw triplets ( _W, X, Y_ ), without considering natural experiments, by tailoring specific existing CATE estimators (the S-learner [42] and Robinson decomposition [53], respectively) to structured treatments. The main drawback of these approaches is that they are inflexible, _i.e._ , they are restricted to using a single estimator and are unable to take advantage of the recent advances in the broader CATE estimation literature ( _e.g._ , recently developed binary treatment estimators [14, 19, 38]). This is a limitation because any single CATE estimator can be unstable across different settings [13]. Notably, the estimators which these methods build on have already been shown to result in high bias in many domains [42, 35, 9, 14]. Likewise, we find that these methods struggle with zero-shot predictions (Section 6). CaML’s key difference from prior work is that we construct a separate task for each training intervention by synthesizing natural experiments. This allows us to (a) flexibly wrap any existing CATE estimator to obtain labels for each task, and thus take advantage of the most recent CATE estimation methods and (b) leverage meta-learning, which requires task-structured data. Consequently, CaML is able to achieve strong zero-shot performance (Section 6). 

## **3 Background: Single-intervention CATE estimation** 

Each task in the CaML framework consists of estimating conditional average treatment effects (CATEs) for a single binary treatment. In this section, we first provide background on CATE estimation under this simple case of a single treatment ( _W_ ) and outcome ( _Y_ ), and subsequently generalize it to our zero-shot setting. Under a single intervention and outcome, we consider _n_ independent observations _P_ 1 _, . . . , Pn_ drawn from a distribution _P_ . For unit _i_ = 1 _, ..., n_ , _Pi_ = 

2 


![](raw/papers/2_LLM_NLP/LLM_paper_list/zero-shot/nips_2023_Zero-shot_causal_learning_images/nips_2023_Zero-shot_causal_learning.pdf-0003-00.png)


**----- Start of picture text -----**<br>
Training                                              Testing<br>Potential Outcomes       (shown for instance 1    ) 1        2        3           4          5         (novel intervention)<br>Y(    ) observed Y(   ) unobserved X<br>W   {     }   {     }     { }    {    ,    }    { }<br>Y(      ) unobserved Y(0) unobserved Y<br>         =  Y(   ) - Y(0) ^<br>always unobserved<br>}<br>**----- End of picture text -----**<br>


Figure 1: Overview of the zero-shot causal learning problem. Each individual has features ( _X_ ), an intervention with features ( _W_ ), and an outcome ( _Y_ ). Lightning bolts (�) represent interventions ( _e.g._ drugs). The personalized effect of an intervention ( _τ_ ) is always unobserved. The goal is to predict the _τ_ for a novel intervention ( _W[′]_ ) and individual ( _X[′]_ ) that did not exist during training. 

( _Wi, Xi, Yi_ ) _∼P_ collects: a binary or continuous outcome of interest _Yi ∈Y ⊂_ R, instance features _Xi ∈X ⊂_ R _[d]_ , a treatment-assignment indicator _Wi ∈{_ 0 _,_ 1 _}_ . We use the Neyman-Rubin potential outcomes framework [31], in which _Yi_ (1) _, Yi_ (0) reflect the outcome of interest either under treatment ( _Wi_ = 1), or under control ( _Wi_ = 0), respectively. In our running medical example, _Yi_ (1) is health status if exposed to the drug, and _Yi_ (0) is health status if not exposed to the drug. Notably, the _fundamental problem of causal inference_ is that we only observe one of the two potential outcomes, as _Yi_ = _Wi · Yi_ (1) + (1 _− Wi_ ) _· Yi_ (0) (e.g. either health status with or without drug exposure can be observed for a specific individual, depending on whether they are prescribed the drug). However, it is possible to make personalized decisions by estimating treatment effects that are tailored to the attributes of individuals (based on features _X_ ). Thus, we focus on estimating _τ_ ( _x_ ), known as the conditional average treatment effect (CATE): 


![](raw/papers/2_LLM_NLP/LLM_paper_list/zero-shot/nips_2023_Zero-shot_causal_learning_images/nips_2023_Zero-shot_causal_learning.pdf-0003-03.png)


A variety of methods have been developed to estimate _τ_ ( _x_ ) from observational data [14]. These rely on standard assumptions of unconfoundedness, consistency, and overlap [50]. _Unconfoundedness_ : there are no unobserved confounders, _i.e. Yi_ (0) _, Yi_ (1) _⊥⊥Wi | Xi_ . _Consistency_ : _Yi_ = _Yi_ ( _Wi_ ), _i.e._ treatment assignment determines whether _Yi_ (1) or _Yi_ (0) is observed. _Overlap_ : Treatment assignment is nondeterministic, such that for all _x_ in support of _X_ : 0 _< P_ ( _Wi_ = 1 _| Xi_ = _x_ ) _<_ 1. 

## **4 Zero-shot causal learning** 

In many real-world settings ( _e.g._ drugs, online A/B tests) novel interventions are frequently introduced, for which no natural experiment data are available. These settings require zero-shot CATE estimates. The zero-shot CATE estimation problem extends the prior section, except the intervention variable _Wi_ is no longer binary, but rather contains rich information about the intervention: _Wi ∈W ⊂_ R _[e]_ ( _e.g._ , a drug’s chemistry), where _Wi_ = 0 corresponds to a sample that did not receive any intervention. Thus, each intervention value _w_ has its own CATE function that we seek to estimate: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/zero-shot/nips_2023_Zero-shot_causal_learning_images/nips_2023_Zero-shot_causal_learning.pdf-0003-07.png)


During training, we observe _n_ independent observations _P_ 1 _, . . . , Pn_ drawn from a distribution _P_ . Each _Pi_ = ( _Wi, Xi, Yi_ ) _∼P_ . Let _Wseen_ be set of all interventions observed during training. The zero-shot CATE estimation task consists of estimating CATE for a novel intervention that was never observed during training: 

_Problem_ 1 (Zero-shot CATE estimation) _._ **Given** _n_ training observations ( _W_ 1 _, X_ 1 _, Y_ 1) _, . . . ,_ ( _Wn, Xn, Yn_ ) drawn from _P_ containing intervention information, individual features, and outcomes... **estimate** _τw′_ ( _x_ ) for a novel intervention _w[′] ∈W/ seen_ . 

This problem formulation extends in a straightforward manner to combinations of interventions, by allowing a single intervention _Wi_ to consist of a set of intervention vectors. CaML supports combinations of interventions, as we elaborate on in Section 4.1 

**CaML overview.** We propose a novel framework for estimating CATE across multiple interventions, even including ones that were never encountered during training. Our framework consists of three 

3 


![](raw/papers/2_LLM_NLP/LLM_paper_list/zero-shot/nips_2023_Zero-shot_causal_learning_images/nips_2023_Zero-shot_causal_learning.pdf-0004-00.png)


**----- Start of picture text -----**<br>
1. Sample natural experiment  2. Estimate pseudo-outcomes (   )~  3. Predict     from individual and intervention information~<br>X       ~ ~ W ~<br>W   {     }   {     }     { }      { } ~ X Predicted    ^<br>Y    ~ Loss<br>{<br>**----- End of picture text -----**<br>


Figure 2: Visual illustration of the CaML (causal meta-learning) framework. (1) We sample a task (i.e., an intervention) and a natural experiment from the training data consisting of individuals who either received the intervention (W={�}), or did not (W={}). Each individual has features ( _X_ ) and an outcome ( _Y_ ), and the intervention also has information ( _W_ ) ( _e.g._ , a drug’s attributes). (2) For each individual we estimate the effect of the intervention on the outcome (pseudo-outcomes _τ_ ˜). (3) We predict an individual’s pseudo-outcomes _τ_ ˜ using a model that fuses _X_ and _W_ . CaML is trained by repeating this procedure across many tasks and corresponding natural experiments. 

key components (Figure 2). First, we formulate CATE estimation as a meta-learning problem in which each task corresponds to the CATE estimation for a unique intervention. A task dataset for a given intervention is constructed by sampling a natural experiment of all individuals who received the intervention, and a sample of individuals who did not. Tasks are augmented with intervention information ( _W_ ). Synthesizing these natural experiments allows us to compute a noisy CATE label _τ_ ˜ using any off-the-shelf estimator ( _τ_ ˜ is referred to as pseudo-outcomes by the causal inference literature [14]). Finally, we train a single meta-model to predict these labels using individual-level ( _X_ ) and intervention-level ( _W_ ) information, such that it is able to generalize to novel tasks, _i.e._ , estimating CATE for novel interventions. 

The CaML framework incorporates three important design considerations: _(1) Single meta-model_ . In domains such as electronic health records and online marketing, we observe that large-scale datasets contain thousands of interventions with rich feature information ( _W_ ). Instead of training a separate model for each intervention, CaML trains a single meta-model that can estimate CATE across all interventions. This approach lets us leverage shared structure across tasks and generalize to novel interventions that were not present during training. _(2) Pseudo-outcomes_ . Rather than directly modeling the response surfaces E[ _Y_ ( _w_ ) _| X_ = _x_ ] and E[ _Y_ (0) _| X_ = _x_ ], we train our model using pseudo-outcomes for each intervention. This decision is based on recent research that highlights the estimation bias when inferring CATE from direct predictions of observed outcomes [9, 42]. CaML outperforms strong baselines that meta-learn these outcomes directly, as demonstrated in our experiments (see Tables 2 and 3, rows S-learner and T-learner with meta-learning). _(3) Discrete tasks from continuous interventions_ . CaML takes advantage of the extensive literature on CATE estimation for single, binary interventions. By creating a natural experiment for each intervention, CaML taps into this literature and benefits from the high performance of recently developed nonparametric CATE estimators[14, 53, 42]. 

CaML identifies CATE for novel interventions under the assumptions that: (1) for each observed intervention _w_ , _τw_ ( _x_ ) is identifiable under the binary treatment assumptions (unconfoundedness, consistency, and overlap) in Section 3. This allows for valid training labels for each task. (2) _τw_ ( _x_ ) = _τ_ ( _w, x_ ), i.e., a global function _τ_ ( _w, x_ ) unifies all intervention-specific CATE functions, (3) _τ_ ( _w, x_ ) is continuous in _w_ . This allows the model to smoothly extrapolate the treatment effect to new interventions that are close to observed interventions in the interventions space. Lastly, (4) _W_ follows a continuous distribution. 

## **4.1 Meta-dataset** 

We formulate CATE estimation as a meta-learning problem. For this, each task refers to CATE estimation for a distinct intervention. Interventions as well as tasks in our meta-dataset are jointly indexed by _j ∈_ N with 1 _≤ j ≤ K_ , such that we can refer to the _j_ -th intervention features with _w_[(] _[j]_[)] . 

4 

We then construct a meta-dataset _D_ in the following way: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/zero-shot/nips_2023_Zero-shot_causal_learning_images/nips_2023_Zero-shot_causal_learning.pdf-0005-01.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/zero-shot/nips_2023_Zero-shot_causal_learning_images/nips_2023_Zero-shot_causal_learning.pdf-0005-02.png)


_D_[(] _[j]_[)] denotes the natural experiment dataset for task _j_ , composed of a treated group (instances which received the intervention, _i.e. Wi_ = _w_[(] _[j]_[)] ) and control group (instances which did not receive any intervention, _i.e. Wi_ = 0). Each sample _i_ represents an individual, for which the quantities ( _Xi, Yi_ ) are collected as introduced in Section 3. In practice, we down-sample both groups ( _i.e._ to 1 million samples for the treated and control groups) in our large-scale experiments. 

We augment each task dataset _D_[(] _[j]_[)] with intervention information, _w_[(] _[j]_[)] _∈_ R _[e]_ , for zero-shot generalization to new interventions [33, 16, 82, 37]. The form of _w_[(] _[j]_[)] varies with the problem domain — for text interventions, it could be a language model’s text embedding [74, 79, 55], while biomedical treatments can be represented as nodes in a knowledge graph [7, 47]. Additionally, domain-specific features, like treatment categories from an ontology, may be included in _w_[(] _[j]_[)] . To handle combinations of interventions (e.g. pairs of drugs), we aggregate the _w_ for each intervention using an order-invariant pooling operation (we used the sum operator), and sample a separate natural experiment for individuals who received the full combination. 

## **4.2 Estimating pseudo-outcomes** 

We next estimate the training targets for each task ( _i.e._ intervention) in the meta-dataset. The training target ( _τ_ ˜[(] _[j]_[)] ) is an unbiased, but noisy, estimate of CATE. More formally, for each task _j_ (which points to the natural experiment dataset for intervention _w_[(] _[j]_[)] ), we estimate _τ_ ˜[(] _[j]_[)] , where E _P_ [˜ _τ_[(] _[j]_[)] _|X_ = _x_ ] = _τw_ ( _j_ )( _x_ ). Thus, _τ_ ˜ _i_[(] _[j]_[)] denotes the target for the _i_ -th sample in the _j_ -th task (indexing will be omitted when it is clear from context). We refer to these targets as pseudo-outcomes, following prior literature [14]. For more details on pseudo-outcomes, refer to Section B in the appendix. 

CaML is agnostic to the specific choice of pseudo-outcome estimator. Thus, we assume a function _η_ ( _D_[(] _[j]_[)] ) which takes as input a task dataset _D_[(] _[j]_[)] _∈ D_ and returns a vector containing the pseudooutcomes _τ_ ˜ for each sample in the task. We extend each task dataset _D_[(] _[j]_[)] with the pseudo-outcomes, such that a sample holds the elements ( _Xi, Yi,_ ˜ _τi_ ). Our key insight is that by collecting these pseudo-outcomes across multiple tasks, and predicting them using a combination of intervention and individual information ( _W, X_ ) we can develop a CATE estimator which generalizes to novel interventions. In practice, we use the RA-learner [15] and treat pseudo-outcome estimation as a data pre-processing step (Appendix C.6). 

## **4.3 Meta-model training** 

Given _m_ outcomes of interest, our goal is then to learn a model Ψ _θ_ : R _[e] ×_ R _[d] →_ R _[m]_ that for parameters _θ_ minimizes 


![](raw/papers/2_LLM_NLP/LLM_paper_list/zero-shot/nips_2023_Zero-shot_causal_learning_images/nips_2023_Zero-shot_causal_learning.pdf-0005-10.png)


where _U_ ( _D_ ) denotes the discrete uniform distribution over the tasks of the meta-dataset _D_ , and where _L_ ( _f_ ) refers to a standard loss function between the pseudo-outcomes and the model output, _i.e._ , _L_ ( _f_ ) = (˜ _τ − f_ ( _w, x_ ))[2] . To assess whether the model generalizes to novel tasks, we partition our meta-dataset by task, into non-overlapping subsets _D_ = _D_ train _∪ D_ val _∪ D_ test. During training, Ψ _θ_ is optimized on training tasks _D_ train. We validate and test this model on _D_ val and _D_ test, which are thus unseen during training tasks. While the CaML framework is agnostic to a specific training strategy, we based our approach (Algorithm 1) on the Reptile meta-learning algorithm [51] which we find performs better compared to straightforward empirical risk minimization ( _c.f._ Section 6). For this, the objective is slightly modified to 


![](raw/papers/2_LLM_NLP/LLM_paper_list/zero-shot/nips_2023_Zero-shot_causal_learning_images/nips_2023_Zero-shot_causal_learning.pdf-0005-12.png)


where _A[k] D_[:] _[F][→F]_[represents the operator that updates a model] _[ f][∈F]_[using data sampled from] the dataset _D_ for _k_ gradient steps. This operator is defined in more detail as the ADAPT routine in 

5 

**Algorithm 1** The CaML algorithm 

**Require:** meta-dataset _D_ , meta-model Ψ _θ_ with initialized **function** ADAPT(Data _D_ , Pseudo-outcomes _τ_ ˜, parameters _θ_ , hyperparameter _k_ . Intervention features _w_ , Model Ψ _θ_ , # of Steps **for** iteration = 1 _,_ 2 _, . . . , L_ **do** _k_ ) _j ←_ SAMPLETASK() Ψ _[′] θ[←]_[Create copy of][ Ψ] _[θ] D_ treat[(] _[j]_[)] _[, D]_ ctrl[(] _[j]_[)] _[, w]_[(] _[j]_[)] _[←]_[Q][UERY][T][ASK][D][ATA][(] _[j]_[)] **for** s = 1 _,_ 2 _, . . . , k_ **do** _τ_ ˜[(] _[j]_[)] _←_ ESTIMATEPSEUDOOUTCOMES( _D_ treat[(] _[j]_[)] _[, D]_ ctrl[(] _[j]_[)][)] ComputeDraw batch of sizeloss _Ls b_ by fromfeeding _D_ . instances _θ[′] ←_ ADAPT(( _D_ treat[(] _[j]_[)] _[, D]_ ctrl[(] _[j]_[)][)] _[,]_[ ˜] _[τ]_[ (] _[j]_[)] _[, w]_[(] _[j]_[)] _[,]_[ Ψ] _[θ]_[,] _[ k]_[)] through model, conditioned on task: _gθ ←← θθ − − βgθ[′]_ {Reptile gradient}{Gradient step for meta-model Ψ _θ_ } _Ls_ =[1] _b_ � _bi_ =1[(˜] _[τ][i][ −]_[Ψ] _θ[′]_[(] _[w][i][, x][i]_[))][2] **end for** Update parameters of Ψ _[′] θ_[:] **return** Ψ _θ_ **end for** _θ ← θ − α∇Ls_ **end function** 

Algorithm 1. Note that depending on the choice of CATE estimator, this routine iterates only over treated samples of a task dataset _D_[(] _[j]_[)] (as in our experiments), or over all samples, including untreated ones. 

## **4.4 CaML architecture** 

To parameterize Ψ _θ_ , we propose a simple but effective model architecture (see Section 6): 

Ψ _θ_ ( _w, x_ ) = MLP1([ ˜ _w_ ; ˜ _x_ ]) _,_ with _x_ ˜ = MLP2( _x_ ) and _w_ ˜ = MLP3( _w_ ) _,_ (7) 

wherefeatures [ _·_ ; _x ·_ are encoded separately into dense vectors] denotes concatenation. Equation 7 shows that the intervention features _w_ ˜ and _x_ ˜, respectively. Our MLPs consist of _w_ and individual layers of the form _g_ ( _z_ ) = _z_ + ReLU(Linear( _z_ )). 

## **5 Theoretical analysis** 

We now consider zero-shot causal learning from a theoretical perspective. Under simplified assumptions, we bound the prediction error in the zero-shot setting. 

We formulate the setting as a supervised learning problem with noisy labels (pseudo-outcomes) where we learn a smooth function _f_ = Ψ( _w, x_ ) _→ τ_ among a family _F_ . We focus on _τ ∈_ R, and assume _τ ∈_ [0 _,_ 1] without loss of generality, since we can normalize _τ_ to this range. The training dataset has _n_ interventions with _m_ samples each, _i.e._ first _n i.i.d._ draws from _PW_ : _w_[(1)] _, . . . , w_[(] _[n]_[)] and then for each _w_[(] _[j]_[)] , _m i.i.d._ draws from _PX_ : _x_[(] 1 _[j]_[)] _[, . . . , x]_[(] _m[j]_[)][.] 

The main theorem quantifies the rate that combining information across different interventions helps with zero-shot performance. We prove a finite-sample generalization bound for the ERM variant of CaML. The ERM is a special case of ADAPT with _k_ = 1 that is more conducive to rigorous analysis. The advantage of Reptile over ERM is orthogonal and we refer the readers to the original discussion [52]. We assume the estimated pseudo-outcomes _τ_ ˜ during training satisfy _τ_ ˜ = _τ_ + _ξ_ where _ξ_ is an independent zero-mean noise with _|ξ| ≤ ϵ_ almost surely for some _ϵ ≥_ 0, 


![](raw/papers/2_LLM_NLP/LLM_paper_list/zero-shot/nips_2023_Zero-shot_causal_learning_images/nips_2023_Zero-shot_causal_learning.pdf-0006-11.png)


The test error is _L_ ( _f_ ) = E _W,X,τ_ [( _f_ ( _w, x_ ) _− τ_ )[2] ]. Let _f[∗]_ = min _f L_ ( _f_ ). We bound the excess loss _L_ ( _f_[ˆ] ) _− L_ ( _f[∗]_ ). Our key assumption is that interventions with similar features _W_ have similar effects in expectation. We assume that all functions in our family are smooth with respect to _W_ , i.e., _∀f ∈F,_ E _W,X ∥∂f/∂W ∥_[2] 2 _≤ β_[2] . � � 

**Theorem 1.** _Under our assumptions, with probability_ 1 _− δ,_ 


![](raw/papers/2_LLM_NLP/LLM_paper_list/zero-shot/nips_2023_Zero-shot_causal_learning_images/nips_2023_Zero-shot_causal_learning.pdf-0006-14.png)


6 

where _Rnm_ is a novel notion of zero-shot Rademacher complexity defined in equation (8); _C_ is a Poincaré constant that only depends on the distribution of _W_ . For large _n, m_ , the leading terms are the function complexity _Rnm_ ( _F_ ), and an _O_ (�1 _/n_ ) term with a numerator that scales with _β_ and (1 + _ϵ_ )[2] _/m_ . This validates our intuition that when the intervention information _W_ is more informative of the true treatment effects (smaller _β_ ), and when the estimation of _τ_ in the training dataset is more accurate, the performance is better on novel interventions. Please refer to Section A for the full proof. Compared to standard generalization bound which usually has a �1 _/n_ term, our main technical innovation involves bounding the variance by the smoothness of the function class plus Poincaré-type inequalities. When _β_ is much smaller than 1 we achieve a tighter bound. 

|Dataset|Samples|Features (_X_)|Outcome (_Y_)|Intervention<br>type|Intervention in-<br>formation (_W_)|
|---|---|---|---|---|---|
|Claims|Patients|Patient history (binned<br>counts<br>of<br>medical<br>codes)|Pancytopenia onset|Drug intake (pre-<br>scription)|Drug embedding<br>(knowledge<br>graph)|
|LINCS|Cell lines|Cancer cell encyclope-<br>dia|Expression of land-<br>mark genes (DEG)|Perturbagen<br>(small molecule)|Molecular<br>embeddings<br>(RDKit)|



Table 1: High-level overview of our two experimental settings. Details in Appendix C.1. 

## **6 Experiments** 

We explore to what extent zero-shot generalization is practical when predicting the effects of interventions. We thus design two novel evaluation settings using real-world data in domains where zero-shot CATE estimation will be highly impactful: (1) Health Insurance Claims: predicting the effect of a drug on a patient, and (2) LINCS: predicting the effect of a perturbation on a cell. We use new datasets because existing causal inference benchmarks [29, 69] focus on a single intervention. By contrast, zero-shot causal learning must be conceptualized in a multi-intervention setting. 

**Zero-shot Evaluation** . Each task corresponds to estimating CATE across different individual samples that received the same intervention. We split all tasks into meta-training/meta-validation, and a holdout meta-testing set for evaluating zero-shot predictions (Table 2, unseen drugs for Claims and Table 3, unseen molecular perturbations in LINCS). For the Claims dataset, we also consider the challenging setting of combinations of unseen drugs (Table B.3). The same patient (Claims) or cell-line (LINCS) can appear in multiple tasks (if they received different interventions at different times). Thus, to ensure a fair zero-shot evaluation, we exclude all samples who have ever received a meta-testing intervention from meta-val/meta-train. Similarly, we exclude all meta-validation patients from meta-train. Details on holdout selection are provided in Appendix C.2. 

Table 1 gives an overview of both benchmarks. In the Claims dataset, we compare zero-shot predictions with strong single-intervention baselines which cannot generalize to unseen interventions. To do so, we further split each task in meta-validation and meta-testing into a train/test (50/50) split of samples. These baselines are trained on a task’s train split, and all methods are evaluated on the test split of the meta-testing tasks. On the LINCS dataset, as each task consists of _<_ 100 cells, single-intervention baselines performed weakly and are excluded from analysis. 

**Baselines.** We compare the zero-shot performance of CaML to two distinct categories of baselines. (1) _Trained directly on test interventions_ . These are strong CATE estimators from prior work and can only be trained on a single intervention. Thus, we train a single model on each meta-testing task’s train split, and evaluate performance on its test split. This category includes T-learner [42], X-learner [42], RA-learner [14], R-learner [53], DragonNet [68], TARNet [66], and FlexTENet [15]. 

(2) _Zero-shot_ baselines are trained across all meta-training tasks and are able to incorporate intervention information ( _W_ ). These methods are thus, in principle, capable of generalizing to unseen interventions. We use GraphITE [23] and Structured Intervention Networks (SIN) [33]. We also introduce two strong baselines which learn to directly estimate potential outcomes by meta-learning across all training interventions, without using pseudo-outcomes: S-learner and T-learner with meta-learning. These extend the S-learner and T-learner from prior work [42] to incorporate intervention information 

7 

( _W_ ) in their predictions. We elaborate on implementation details of baselines in Appendix C.7. For details on hyperparameter search and fair comparison, see Appendix C.1. 

## **6.1 Setting 1: Personalized drug side effect prediction from large-scale medical claims** 

Our first setting (Claims) is to predict the increased likelihood of a life-threatening side effect caused by a drug prescription. We leverage a large-scale insurance claims dataset of over 3.5 billion claims across 30.6 million patients in the United States[1] . Each datestamped insurance claim contains a set of diagnoses (ICD-10 codes), drug prescriptions (DrugBank ID), procedures (ICD-10 codes), and laboratory results (LOINC codes). Laboratory results were categorized by whether the result was high, low, normal, abnormal (for non-continuous labs), or unknown. 

Interventions are administration of one drug ( _n_ = 745), or two drugs ( _n_ = 22 _,_ 883) prescribed in combination. Time of intervention corresponds to the _first_ day of exposure. Intervention information ( _W_ ) was generated from pre-trained drug embeddings from a large-scale biomedical knowledge graph [7] (Appendix C). We compute drug combination embeddings as the sum of the embeddings of the constituent drugs. We focus on the binary outcome ( _Y_ ) of the occurrence of the side effect pancytopenia within 90 days of intervention exposure. Pancytopenia is a deficiency across all three blood cell lines (red blood cells, white blood cells, and platelets). Pancytopenia is life-threatening, with a 10-20% mortality rate [36, 41], and is a rare side effect of many common medications [40] ( _e.g._ arthritis and cancer drugs), which in turn require intensive monitoring of the blood work. Following prior work [22], patient medical history features ( _X_ ) were constructed by time-binned counts of each unique medical code (diagnosis, procedure, lab result, drug prescription) at seven different time scales before the drug was prescribed, resulting in a total of 443,940 features. For more details, refer to Appendix C.1. 

**Metrics** We rely on best practices for evaluating CATE estimators in observational data, as established by recent work [81, 10], which recommend to assess treatment rules by comparing subgroups across different quantiles of estimated CATE. We follow the high vs. others RATE (rank-weighted average treatment effect) approach from Yadlowsky et. al [81], which computes the difference in average treatment effect (ATE) of the top _u_ percent of individuals (ranked by predicted CATE), versus all individuals (for more details, see Appendix C.1). For instance, RATE @ 0.99 is the difference between the top 1% of the samples (by estimated CATE) vs. the average treatment effect (ATE) across all samples, which we would expect to be high if the CATE estimator is accurate. Note that estimates of RATE can be negative if model predictions are inversely associated with CATE. We elaborate on the RATE computation in Appendix C.1. 

The real-world use case of our model is preventing drug prescription for a small subset of high-risk individuals. Thus, more specifically, for each task _j_ , intervention _wj_ in the meta-dataset, and metamodel Ψ _θ_ , we compute _RATE_ @ _u_ for each _u_ in [0 _._ 999 _,_ 0 _._ 998 _,_ 0 _._ 995 _,_ 0 _._ 99] across individuals who received the intervention. 

Additionally, because our meta-testing dataset consists of individuals treated with drugs known to cause pancytopenia, observational metrics of recall and precision are also a rough _proxy_ for successful CATE estimation (and highly correlated to RATE, Table 2). Thus, as secondary metrics, we also compute _Recall_ @ _u_ and _Precision_ @ _u_ for the same set of thresholds as RATE, where a positive label is defined as occurrence of pancytopenia after intervention. 

## **6.2 Setting 2: Cellular gene expression response due to perturbation** 

Our second setting (LINCS) is to predict how a cell’s gene expression ( _Y_ ) will respond to intervention from perturbagen (small molecule compound such as a drug). This is a critical problem as accurately predicting intervention response will accelerate drug-discovery. We use data for 10,325 different perturbagens from the LINCS Program [70]. Each perturbagen corresponds to a different small molecule. Molecular embeddings were generated using the RDKit featurizer [44] and used as intervention information ( _W_ ). Outcomes ( _Y_ ) of interest are post-intervention gene expression across the top-50 and top-20 differentially expressed landmark genes (DEGs) in the LINCS dataset. We did not look at all 978 genes since most do not show significant variation upon perturbation. We use 19 _,_ 221 features ( _X_ ) from the Cancer Cell Line Encyclopedia (CCLE) [20] to characterize each 

> 1Insurance company undisclosed per data use agreement. 

8 

||RATE @_u_ (_↑_)<br>0_._999 _._998 0_._995<br>0_._99|Recall @_u_ (_↑_)<br>0_._999 0_._998 0_._995 0_._99|Precision @_u_ (_↑_)<br>0_._999 0_._998 0_._995<br>0_._99<br>0.00<br>0.00<br>0.00<br>0.00<br>0.36<br>0.29<br>0.18<br>0.11<br>0.09<br>0.07<br>0.06<br>0.05<br>0.24<br>0.21<br>0.15<br>0.11<br>0.54<br>0.42<br>0.26<br>0.16<br>0.15<br>0.12<br>0.08<br>0.06<br>0.18<br>0.15<br>0.09<br>0.06<br>0.15<br>0.13<br>0.09<br>0.06<br>0.23<br>0.14<br>0.07<br>0.04<br>0.01<br>0.01<br>0.01<br>0.01<br>0.25<br>0.18<br>0.1<br>0.06<br>0.45<br>0.35<br>0.21<br>0.13<br>0.45<br>0.35<br>0.22<br>0.14<br>0.48<br>0.38<br>0.26<br>**0.16**<br>**0.54**<br>**0.43**<br>**0.26**<br>0.16<br>**Trained**<br>**on test**<br>**intervention**<br>**Zero-shot**<br>(Best in bold)<br>(Best underlined)|
|---|---|---|---|
|Random|0.00<br>0.00<br>0.00<br>0.00|0.00<br>0.00<br>0.01<br>0.00||
|T-learner|0.32<br>0.26<br>0.16<br>0.10|0.12<br>0.18<br>0.26<br>0.31||
|X-learner|0.06<br>0.05<br>0.04<br>0.03|0.02<br>0.04<br>0.08<br>0.12||
|R-learner|0.19<br>0.17<br>0.12<br>0.08|0.06<br>0.1<br>0.19<br>0.26||
|RA-learner|0.47<br>0.37<br>0.23<br>0.14|0.17<br>0.26<br>0.38<br>0.45||
|DragonNet|0.09<br>0.07<br>0.05<br>0.04|0.03<br>0.05<br>0.08<br>0.11||
|TARNet|0.15<br>0.12<br>0.07<br>0.05|0.05<br>0.08<br>0.12<br>0.14||
|FlexTENet|0.10<br>0.09<br>0.06<br>0.04|0.04<br>0.06<br>0.11<br>0.16||
|GraphITE|0.19<br>0.12<br>0.05<br>0.03|0.07<br>0.08<br>0.09<br>0.10||
|SIN|0.00<br>0.00<br>0.00<br>0.00|0.00<br>0.00<br>0.01<br>0.02||
|S-learner w/ meta-learning|0.21<br>0.16<br>0.09<br>0.05|0.08<br>0.11<br>0.15<br>0.16||
|T-learner w/ meta-learning|0.40<br>0.31<br>0.18<br>0.11|0.15<br>0.22<br>0.32<br>0.38||
|CaML - w/o meta-learning|0.39<br>0.31<br>0.18<br>0.11|0.15<br>0.22<br>0.32<br>0.39||
|CaML - w/o RA-learner|0.45<br>0.36<br>0.22<br>**0.14**|0.16<br>0.24<br>0.34<br>0.41||
|CaML (ours)|**0.48**<br>**0.38**<br>**0.23**<br>0.13|**0.18**<br>**0.27**<br>**0.38**<br>**0.45**||



Table 2: Performance results for the Claims dataset (predicting the effect of drug exposure on pancytopenia onset from patient medical history). Key findings are (1) CaML outperforms all zeroshot baselines (RATE is 18-27% higher than T-Learner w/ meta-learning, the strongest zero-shot baseline) (2) CaML performs stronger (up to 8 _×_ higher RATE values) than 6 of the 7 baselines which are trained directly on the test interventions, and performs comparably to the strongest baseline trained directly on the test interventions (RA-learner). Mean is reported across all runs; standard deviations included in (Appendix Table 4). Analogous trends hold for generalization to _pairs_ of unseen drugs (Appendix Table B.3). 

cell-line ( _n_ = 99), each of which correspond to unperturbed gene expression measured in a different lab environment using a different experimental assay. For more details, see Appendix C.1. 

**Metrics** . A key advantage of experiments on cells is that at evaluation time we can observe both _Y_ (0) and _Y_ (1) for the same cell line _X_ , through multiple experiments on clones of the same cellline in controlled lab conditions. In the LINCS dataset, _Y_ (0) is also measured for all cells which received an intervention. Thus, we can directly compute the precision in estimating heterogeneous effects (PEHE) on all treated cells in our meta-testing dataset, an established measure for CATE estimation performance analogous to mean-squared error [28] (see Appendix C.1). 

## **6.3 Key findings** 

_CaML’s zero-shot predictions outperform baselines with direct access to the target intervention._ In the medical claims setting, single intervention baselines (Tables 2, dark grey rows) are the highest performing baselines as we train them directly on the meta-test intervention. Still, CaML outperforms 6 out of 7 of these baselines (up to 8 _×_ higher RATE) and achieves comparable performance to the strongest of these baselines, the RA-learner. Furthermore, CaML strongly outperforms alternative zero-shot CATE estimators (RATE is 18-27% higher than T-Learner w/ meta-learning, the strongest zero-shot baseline). In the LINCS data, multi-intervention learners are strongest as there are only a small number of instances (cell lines) per intervention[2] . CaML outperforms both single-intervention and multi-intervention learners by drawing from both of their strengths—it allows us to use strong CATE estimation methods ( _i.e._ the RA-learner) which previously were restricted to single interventions, while sharing information across multiple interventions. 

_CaML learns to generalize from single interventions to combinations of unseen interventions (drug pairs)._ We evaluate CaML’s performance in the challenging setting of predicting the personalized effects of combinations of two drugs which are both unseen during training, while only training on interventions consisting of single drugs. CaML achieves strong performance results (see Appendix Table B.3), surpassing the best baseline trained on the test tasks, and outperforms all zero-shot baselines, across all 12 metrics. 

_Understanding CaML’s performance results._ Our ablation studies explain that CaML’s performance gains are due to (1) our meta-learning formulation and algorithm (in contrast to the w/o meta-learning 

> 2Single-task baselines excluded from Table 3: all performed similar or worse than mean baseline due to low task sample size. 

9 

||PEHE 50 DEGs(_↓_)|PEHE 20 DEGs(_↓_)|
|---|---|---|
|Mean.|3.78|4.11|
|GraphITE|3.58_±_0.023|3.82_±_0.011|
|SIN|3.78_±_0.001|4.06_±_0.001|
|S-learner w/ meta-learning|3.63_±_0.004|3.90_±_0.004|
|T-learner w/ meta-learning|3.61_±_0.007|3.85_±_0.006|
|CaML - w/o meta-learning|3.57_±_0.006|3.79_±_0.004|
|CaML - w/o RA-learner|4.28_±_0.517|4.60_±_0.413|
|CaML (ours)|**3.56**_±_0.001|**3.78**_±_0.005|



Table 3: Performance results for the LINCS dataset (predicting the effect of an unseen perturbation on the gene expression of an unseen cell-line). CaML outperforms all baselines. Improvement is largest for the 20 most differentially expressed genes, where most signal is expected. 

row, in which ERM is used to train the model), and (2) the flexible CATE estimation strategy, allowing to take advantage of recently developed CATE estimators previously restricted to single interventions (in contrast to the w/o RA-learner row, in which an alternative pseudo-outcome estimator is used). Lastly, (3) comparison to existing binary intervention CATE estimators trained separately on each meta-testing intervention (Table 2, grey rows) shows that we gain from learning from thousands interventions. See Appendix C.3 for details on ablations. 

## **7 Conclusion** 

We introduce a novel approach to predict the effects of novel interventions. CaML consistently outperforms state-of-the-art baselines, by unlocking zero-shot capability for many recently developed CATE estimation methods which were previously restricted to studying single interventions in isolation. While our study is limited to retrospective data, we plan to prospectively validate our findings. Future work includes designing new model architectures and CATE estimators which learn well under the CaML framework, as well as more generally exploring novel learning strategies that enable zero-shot causal learning. 

**Societal impacts** . In high-stakes decision-making inaccurate predictions can lead to severe consequences. It is important not to overly rely on model predictions and proactively involve domain experts, such as doctors, in the decision-making process. Additionally, it is crucial to ensure that underserved communities are not disadvantaged by errors in treatment effect estimates due to underrepresentation in the training data. This risk can be monitored by evaluating CATE estimators on underserved patient groups prior to clinical deployment. 

## **Acknowledgements** 

We thank Stefan Wager, Emma Pierson, Kexin Huang, Kaidi Cao, Yanay Rosen, Johann Gaebler, Maria Brbi´c, Kefang Dong, June Vuong for helpful conversations. H.N was supported by a Stanford Knight-Hennessy Scholarship and the National Science Foundation Graduate Research Fellowship under Grant No. DGE-1656518. M.M. was supported by DARPA N660011924033 (MCS), NIH NINDS R61 NS11865, GSK, Wu Tsai Neurosciences Institute. A.S and S.O were supported by the American Slovenian Education Foundation (ASEF) fellowship. M.Y was supported by the Microsoft Research PhD fellowship. Y.R was supported by funding from GlaxoSmithKline LLC. Y.C. was supported by Stanford Graduate Fellowship and NSF IIS 2045685. We also gratefully acknowledge the support of Stanford HAI for Google Cloud Credits, DARPA under Nos. HR00112190039 (TAMI), N660011924033 (MCS); ARO under Nos. W911NF-16-1-0342 (MURI), W911NF-16-10171 (DURIP); NSF under Nos. OAC-1835598 (CINES), OAC-1934578 (HDR), CCF-1918940 (Expeditions), NIH under No. 3U54HG010426-04S1 (HuBMAP), Stanford Data Science Initiative, Wu Tsai Neurosciences Institute, Amazon, Docomo, GSK, Hitachi, Intel, JPMorgan Chase, Juniper Networks, KDDI, NEC, and Toshiba. 

10 

## **References** 

- [1] Ahmed M Alaa and Mihaela Van Der Schaar. Bayesian inference of individualized treatment effects using multi-task gaussian processes. _Advances in neural information processing systems_ , 30, 2017. 

- [2] Susan Athey and Guido Imbens. Recursive partitioning for heterogeneous causal effects. _Proceedings of the National Academy of Sciences_ , 113(27):7353–7360, 2016. 

- [3] Yoshua Bengio, Samy Bengio, and Jocelyn Cloutier. _Learning a synaptic learning rule_ . Citeseer, 1990. 

- [4] Ioana Bica, Ahmed M Alaa, Craig Lambert, and Mihaela Van Der Schaar. From real-world patient data to individualized treatment effects using machine learning: current and future methods to address underlying challenges. _Clinical Pharmacology & Therapeutics_ , 109(1):87– 100, 2021. 

- [5] Stéphane Boucheron, Gábor Lugosi, and Pascal Massart. _Concentration inequalities: A nonasymptotic theory of independence_ . Oxford university press, 2013. 

- [6] Olivier Bousquet. A bennett concentration inequality and its application to suprema of empirical processes. _Comptes Rendus Mathematique_ , 334(6):495–500, 2002. 

- [7] Payal Chandak, Kexin Huang, and Marinka Zitnik. Building a knowledge graph to enable precision medicine. _bioRxiv_ , 2022. 

- [8] Hong-Bin Chen, Sinho Chewi, and Jonathan Niles-Weed. Dimension-free log-sobolev inequalities for mixture distributions. _Journal of Functional Analysis_ , 281(11):109236, 2021. 

- [9] Victor Chernozhukov, Denis Chetverikov, Mert Demirer, Esther Duflo, Christian Hansen, Whitney Newey, and James Robins. Double/debiased machine learning for treatment and structural parameters, 2018. 

- [10] Victor Chernozhukov, Mert Demirer, Esther Duflo, and Ivan Fernandez-Val. Generic machine learning inference on heterogeneous treatment effects in randomized experiments, with an application to immunization in india. Technical report, National Bureau of Economic Research, 2018. 

- [11] Richard K Crump, V Joseph Hotz, Guido W Imbens, and Oscar A Mitnik. Nonparametric tests for treatment effect heterogeneity. _The Review of Economics and Statistics_ , 90(3):389–405, 2008. 

- [12] Alicia Curth, David Svensson, Jim Weatherall, and Mihaela van der Schaar. Really doing great at estimating cate? a critical look at ml benchmarking practices in treatment effect estimation. In _Thirty-fifth Conference on Neural Information Processing Systems Datasets and Benchmarks Track (Round 2)_ , 2021. 

- [13] Alicia Curth and Mihaela van der Schaar. Doing great at estimating cate? on the neglected assumptions in benchmark comparisons of treatment effect estimators. _arXiv preprint arXiv:2107.13346_ , 2021. 

- [14] Alicia Curth and Mihaela van der Schaar. Nonparametric estimation of heterogeneous treatment effects: From theory to learning algorithms. In _International Conference on Artificial Intelligence and Statistics_ , pages 1810–1818. PMLR, 2021. 

- [15] Alicia Curth and Mihaela van der Schaar. On inductive biases for heterogeneous treatment effect estimation. _Advances in Neural Information Processing Systems_ , 34:15883–15894, 2021. 

- [16] Gerald DeJong and Raymond Mooney. Explanation-based learning: An alternative view. _Machine learning_ , 1986. 

- [17] Qiaonan Duan, Corey Flynn, Mario Niepel, Marc Hafner, Jeremy L Muhlich, Nicolas F Fernandez, Andrew D Rouillard, Christopher M Tan, Edward Y Chen, Todd R Golub, et al. Lincs canvas browser: interactive web app to query, browse and interrogate lincs l1000 gene expression signatures. _Nucleic acids research_ , 42(W1):W449–W460, 2014. 

- [18] Chelsea Finn, Pieter Abbeel, and Sergey Levine. Model-agnostic meta-learning for fast adaptation of deep networks. In _International conference on machine learning_ , pages 1126–1135. PMLR, 2017. 

11 

- [19] Dennis Frauen and Stefan Feuerriegel. Estimating individual treatment effects under unobserved confounding using binary instruments. _arXiv preprint arXiv:2208.08544_ , 2022. 

- [20] Mahmoud Ghandi, Franklin W Huang, Judit Jané-Valbuena, Gregory V Kryukov, Christopher C Lo, E Robert McDonald, 3rd, Jordi Barretina, Ellen T Gelfand, Craig M Bielski, Haoxin Li, Kevin Hu, Alexander Y Andreev-Drakhlin, Jaegil Kim, Julian M Hess, Brian J Haas, François Aguet, Barbara A Weir, Michael V Rothberg, Brenton R Paolella, Michael S Lawrence, Rehan Akbani, Yiling Lu, Hong L Tiv, Prafulla C Gokhale, Antoine de Weck, Ali Amin Mansour, Coyin Oh, Juliann Shih, Kevin Hadi, Yanay Rosen, Jonathan Bistline, Kavitha Venkatesan, Anupama Reddy, Dmitriy Sonkin, Manway Liu, Joseph Lehar, Joshua M Korn, Dale A Porter, Michael D Jones, Javad Golji, Giordano Caponigro, Jordan E Taylor, Caitlin M Dunning, Amanda L Creech, Allison C Warren, James M McFarland, Mahdi Zamanighomi, Audrey Kauffmann, Nicolas Stransky, Marcin Imielinski, Yosef E Maruvka, Andrew D Cherniack, Aviad Tsherniak, Francisca Vazquez, Jacob D Jaffe, Andrew A Lane, David M Weinstock, Cory M Johannessen, Michael P Morrissey, Frank Stegmeier, Robert Schlegel, William C Hahn, Gad Getz, Gordon B Mills, Jesse S Boehm, Todd R Golub, Levi A Garraway, and William R Sellers. Next-generation characterization of the cancer cell line encyclopedia. _Nature_ , 569(7757):503–508, May 2019. 

- [21] Donald P Green and Holger L Kern. Modeling heterogeneous treatment effects in survey experiments with bayesian additive regression trees. _Public opinion quarterly_ , 76(3):491–511, 2012. 

- [22] Lin Lawrence Guo, Ethan Steinberg, Scott Lanyon Fleming, Jose Posada, Joshua Lemmon, Stephen R Pfohl, Nigam Shah, Jason Fries, and Lillian Sung. Ehr foundation models improve robustness in the presence of temporal distribution shift. _medRxiv_ , 2022. 

- [23] Shonosuke Harada and Hisashi Kashima. Graphite: Estimating individual effects of graphstructured treatments. In _Proceedings of the 30th ACM International Conference on Information & Knowledge Management_ , pages 659–668, 2021. 

- [24] Negar Hassanpour and Russell Greiner. Counterfactual regression with importance sampling weights. In _IJCAI_ , pages 5880–5887, 2019. 

- [25] Negar Hassanpour and Russell Greiner. Learning disentangled representations for counterfactual regression. In _International Conference on Learning Representations_ , 2019. 

- [26] Trevor Hastie, Robert Tibshirani, Jerome H Friedman, and Jerome H Friedman. _The elements of statistical learning: data mining, inference, and prediction_ , volume 2. Springer, 2009. 

- [27] Leon Hetzel, Simon Böhm, Niki Kilbertus, Stephan Günnemann, Mohammad Lotfollahi, and Fabian Theis. Predicting single-cell perturbation responses for unseen drugs. _arXiv preprint arXiv:2204.13545_ , 2022. 

- [28] Jennifer L Hill. Bayesian nonparametric modeling for causal inference. _Journal of Computational and Graphical Statistics_ , 20(1):217–240, 2011. 

- [29] Jennifer L Hill, Jeanne Brooks-Gunn, and Jane Waldfogel. Sustained effects of high participation in an early intervention for low-birth-weight premature infants. _Developmental psychology_ , 39(4):730, 2003. 

- [30] Timothy Hospedales, Antreas Antoniou, Paul Micaelli, and Amos Storkey. Meta-learning in neural networks: A survey. _IEEE transactions on pattern analysis and machine intelligence_ , 44(9):5149–5169, 2021. 

- [31] Guido W Imbens and Donald B Rubin. _Causal inference in statistics, social, and biomedical sciences_ . Cambridge University Press, 2015. 

- [32] Fredrik Johansson, Uri Shalit, and David Sontag. Learning representations for counterfactual inference. In _International conference on machine learning_ , pages 3020–3029. PMLR, 2016. 

- [33] Jean Kaddour, Yuchen Zhu, Qi Liu, Matt J Kusner, and Ricardo Silva. Causal effect inference for structured treatments. _Advances in Neural Information Processing Systems_ , 34:24841–24854, 2021. 

- [34] Edward H Kennedy. Optimal doubly robust estimation of heterogeneous causal effects. _arXiv preprint arXiv:2004.14497_ , 2020. 

- [35] Edward H Kennedy. Towards optimal doubly robust estimation of heterogeneous causal effects (2020). _URL https://arxiv. org/abs_ , 2020. 

12 

- [36] Jitender Mohan Khunger, S Arulselvi, Uma Sharma, Sunil Ranga, and VH Talib. Pancytopenia– a clinico haematological study of 200 cases. _Indian journal of pathology & microbiology_ , 45(3):375–379, 2002. 

- [37] Pang Wei Koh, Shiori Sagawa, Henrik Marklund, Sang Michael Xie, Marvin Zhang, Akshay Balsubramani, Weihua Hu, Michihiro Yasunaga, Richard Lanas Phillips, Irena Gao, et al. Wilds: A benchmark of in-the-wild distribution shifts. In _International Conference on Machine Learning (ICML)_ , 2021. 

- [38] Andrei V Konstantinov, Stanislav R Kirpichenko, and Lev V Utkin. Heterogeneous treatment effect with trained kernels of the nadaraya-watson regression. _arXiv preprint arXiv:2207.09139_ , 2022. 

- [39] N Kostantinos. Gaussian mixtures and their applications to signal processing. _Advanced signal processing handbook: theory and implementation for radar, sonar, and medical imaging real time systems_ , pages 3–1, 2000. 

- [40] Michael Kuhn, Ivica Letunic, Lars Juhl Jensen, and Peer Bork. The sider database of drugs and side effects. _Nucleic acids research_ , 44(D1):D1075–D1079, 2016. 

- [41] R Kumar, SP Kalra, H Kumar, AC Anand, and H Madan. Pancytopenia–a six year study. _The Journal of the Association of Physicians of India_ , 49:1078–1081, 2001. 

- [42] Sören R Künzel, Jasjeet S Sekhon, Peter J Bickel, and Bin Yu. Metalearners for estimating heterogeneous treatment effects using machine learning. _Proceedings of the national academy of sciences_ , 116(10):4156–4165, 2019. 

- [43] Nikolay Kuznetsov and Alexander Nazarov. Sharp constants in the poincaré, steklov and related inequalities (a survey). _Mathematika_ , 61(2):328–344, 2015. 

- [44] Greg Landrum et al. Rdkit: Open-source cheminformatics. 2006. 

- [45] Michel Ledoux. Concentration of measure and logarithmic sobolev inequalities. In _Seminaire de probabilites XXXIII_ , pages 120–216. Springer, 1999. 

- [46] Hongzhu Li, Xiangrui Gao, and Yafeng Deng. Stargraph: A coarse-to-fine representation method for large-scale knowledge graph, 2022. 

- [47] Michelle M Li, Kexin Huang, and Marinka Zitnik. Graph representation learning in biomedicine and healthcare. _Nature Biomedical Engineering_ , pages 1–17, 2022. 

- [48] Jing Ma, Ruocheng Guo, Aidong Zhang, and Jundong Li. Multi-cause effect estimation with disentangled confounder representation. In _Proceedings of the Thirtieth International Joint Conference on Artificial Intelligence_ , 2021. 

- [49] Ron Meir and Tong Zhang. Generalization error bounds for bayesian mixture algorithms. _Journal of Machine Learning Research_ , 4(Oct):839–860, 2003. 

- [50] Stephen L Morgan and Christopher Winship. _Counterfactuals and causal inference_ . Cambridge University Press, 2015. 

- [51] Alex Nichol, Joshua Achiam, and John Schulman. On first-order meta-learning algorithms. _arXiv preprint arXiv:1803.02999_ , 2018. 

- [52] Alex Nichol and John Schulman. Reptile: a scalable metalearning algorithm. _arXiv preprint arXiv:1803.02999_ , 2(3):4, 2018. 

- [53] Xinkun Nie and Stefan Wager. Quasi-oracle estimation of heterogeneous treatment effects. _Biometrika_ , 108(2):299–319, 2021. 

- [54] Frank Nielsen and Richard Nock. On the chi square and higher-order chi distances for approximating f-divergences. _IEEE Signal Processing Letters_ , 21(1):10–13, 2013. 

- [55] Hamed Nilforoshan and Eugene Wu. Leveraging quality prediction models for automatic writing feedback. In _Twelfth International AAAI Conference on Web and Social Media_ , 2018. 

- [56] Lawrence E Payne and Hans F Weinberger. An optimal poincaré inequality for convex domains. _Archive for Rational Mechanics and Analysis_ , 5(1):286–292, 1960. 

- [57] Henri Poincaré. Sur les équations aux dérivées partielles de la physique mathématique. _American Journal of Mathematics_ , pages 211–294, 1890. 

13 

- [58] Zhaozhi Qian, Alicia Curth, and Mihaela van der Schaar. Estimating multi-cause treatment effects via single-cause perturbation. _Advances in Neural Information Processing Systems_ , 34:23754–23767, 2021. 

- [59] Aniruddh Raghu, Maithra Raghu, Samy Bengio, and Oriol Vinyals. Rapid learning or feature reuse? towards understanding the effectiveness of maml. _arXiv preprint arXiv:1909.09157_ , 2019. 

- [60] Bernardino Romera-Paredes and Philip Torr. An embarrassingly simple approach to zero-shot learning. In _International conference on machine learning_ , pages 2152–2161. PMLR, 2015. 

- [61] Yusuf Roohani, Kexin Huang, and Jure Leskovec. Gears: Predicting transcriptional outcomes of novel multi-gene perturbations. _bioRxiv_ , 2022. 

- [62] Shiv Kumar Saini, Sunny Dhamnani, Akil Arif Ibrahim, and Prithviraj Chavan. Multiple treatment effect estimation using deep generative model with task embedding. In _The World Wide Web Conference_ , pages 1601–1611, 2019. 

- [63] Tim Salimans and Durk P Kingma. Weight normalization: A simple reparameterization to accelerate training of deep neural networks. _Advances in neural information processing systems_ , 29, 2016. 

- [64] André Schlichting. Poincaré and log–sobolev inequalities for mixtures. _Entropy_ , 21(1):89, 2019. 

- [65] Jürgen Schmidhuber. _Evolutionary principles in self-referential learning, or on learning how to learn: the meta-meta-... hook_ . PhD thesis, Technische Universität München, 1987. 

- [66] Uri Shalit, Fredrik D Johansson, and David Sontag. Estimating individual treatment effect: generalization bounds and algorithms. In _International Conference on Machine Learning_ , pages 3076–3085. PMLR, 2017. 

- [67] Ankit Sharma, Garima Gupta, Ranjitha Prasad, Arnab Chatterjee, Lovekesh Vig, and Gautam Shroff. Metaci: Meta-learning for causal inference in a heterogeneous population. _arXiv preprint arXiv:1912.03960_ , 2019. 

- [68] Claudia Shi, David Blei, and Victor Veitch. Adapting neural networks for the estimation of treatment effects. _Advances in neural information processing systems_ , 32, 2019. 

- [69] Yishai Shimoni, Chen Yanover, Ehud Karavani, and Yaara Goldschmnidt. Benchmarking framework for performance-evaluation of causal inference analysis. _arXiv preprint arXiv:1802.05046_ , 2018. 

- [70] Aravind Subramanian, Rajiv Narayan, Steven M Corsello, David D Peck, Ted E Natoli, Xiaodong Lu, Joshua Gould, John F Davis, Andrew A Tubelli, Jacob K Asiedu, David L Lahr, Jodi E Hirschman, Zihan Liu, Melanie Donahue, Bina Julian, Mariya Khan, David Wadden, Ian C Smith, Daniel Lam, Arthur Liberzon, Courtney Toder, Mukta Bagul, Marek Orzechowski, Oana M Enache, Federica Piccioni, Sarah A Johnson, Nicholas J Lyons, Alice H Berger, Alykhan F Shamji, Angela N Brooks, Anita Vrcic, Corey Flynn, Jacqueline Rosains, David Y Takeda, Roger Hu, Desiree Davison, Justin Lamb, Kristin Ardlie, Larson Hogstrom, Peyton Greenside, Nathanael S Gray, Paul A Clemons, Serena Silver, Xiaoyun Wu, Wen-Ning Zhao, Willis Read-Button, Xiaohua Wu, Stephen J Haggarty, Lucienne V Ronco, Jesse S Boehm, Stuart L Schreiber, John G Doench, Joshua A Bittker, David E Root, Bang Wong, and Todd R Golub. A next generation connectivity map: L1000 platform and the first 1,000,000 profiles. _Cell_ , 171(6):1437–1452.e17, November 2017. 

- [71] Nicholas P Tatonetti, Patrick P Ye, Roxana Daneshjou, and Russ B Altman. Data-driven prediction of drug effects and interactions. _Science translational medicine_ , 4(125):125ra31– 125ra31, 2012. 

- [72] Sebastian Thrun and Lorien Pratt. _Learning to learn_ . Springer Science & Business Media, 2012. 

- [73] Vladimir Vapnik. Principles of risk minimization for learning theory. _Advances in neural information processing systems_ , 4, 1991. 

- [74] Victor Veitch, Dhanya Sridhar, and David Blei. Adapting text embeddings for causal inference. In _Conference on Uncertainty in Artificial Intelligence_ , pages 919–928. PMLR, 2020. 

- [75] Stefan Wager. Stats 361: Causal inference, 2020. 

14 

- [76] Stefan Wager and Susan Athey. Estimation and inference of heterogeneous treatment effects using random forests. _Journal of the American Statistical Association_ , 113(523):1228–1242, 2018. 

- [77] Wei Wang, Vincent W Zheng, Han Yu, and Chunyan Miao. A survey of zero-shot learning: Settings, methods, and applications. _ACM Transactions on Intelligent Systems and Technology (TIST)_ , 10(2):1–37, 2019. 

- [78] Yixin Wang and David M Blei. The blessings of multiple causes. _Journal of the American Statistical Association_ , 114(528):1574–1596, 2019. 

- [79] Galen Weld, Peter West, Maria Glenski, David Arbour, Ryan A Rossi, and Tim Althoff. Adjusting for confounders with text: Challenges and an empirical evaluation framework for causal inference. In _Proceedings of the International AAAI Conference on Web and Social Media_ , volume 16, pages 1109–1120, 2022. 

- [80] Yongqin Xian, Bernt Schiele, and Zeynep Akata. Zero-shot learning-the good, the bad and the ugly. In _Proceedings of the IEEE conference on computer vision and pattern recognition_ , pages 4582–4591, 2017. 

- [81] Steve Yadlowsky, Scott Fleming, Nigam Shah, Emma Brunskill, and Stefan Wager. Evaluating treatment prioritization rules via rank-weighted average treatment effects. _arXiv preprint arXiv:2111.07966_ , 2021. 

- [82] Michihiro Yasunaga, Hongyu Ren, Antoine Bosselut, Percy Liang, and Jure Leskovec. QAGNN: Reasoning with language models and knowledge graphs for question answering. In _North American Chapter of the Association for Computational Linguistics (NAACL)_ , 2021. 

- [83] Shing-Tung Yau. Isoperimetric constants and the first eigenvalue of a compact riemannian manifold. In _Annales Scientifiques de l’École Normale Supérieure_ , volume 8, pages 487–507, 1975. 

- [84] Jinsung Yoon, James Jordon, and Mihaela Van Der Schaar. Ganite: Estimation of individualized treatment effects using generative adversarial nets. In _International Conference on Learning Representations_ , 2018. 

- [85] Long Yu, Zhicong Luo, Huanyong Liu, Deng Lin, Hongzhu Li, and Yafeng Deng. Triplere: Knowledge graph embeddings via tripled relation vectors. _arXiv preprint arXiv:2209.08271_ , 2022. 

- [86] Yao Zhang, Alexis Bellot, and Mihaela Schaar. Learning overlapping representations for the estimation of individualized treatment effects. In _International Conference on Artificial Intelligence and Statistics_ , pages 1005–1014. PMLR, 2020. 

- [87] Marinka Zitnik, Monica Agrawal, and Jure Leskovec. Modeling polypharmacy side effects with graph convolutional networks. _Bioinformatics_ , 34(13):i457–i466, 2018. 

15 

## **A Zero-shot Rademacher complexity and Proof of Theorem 1** 

## **A.1 Problem setup and assumptions** 

Let _w ∈W ⊆_ R _[e]_ denote an intervention and _x ∈X ⊆_ R _[d]_ denote an individual that received it. Assume the outcome to predict is a scalar _y ∈_ [0 _,_ 1]. The hypothesis class is _F_ = _{f_ : ( _w, x_ ) _→ y}_ . The dataset has _n_ interventions with _m_ independent units which received each intervention, _i.e._ first _n i.i.d._ draws from _PW_ and then _m i.i.d._ draws from _PX_ for each _w_[(] _[j]_[)] . During training we have access to noisy estimate _y_ ˜ = _y_ + _ξ_ where _ξ_ is an independent noise with E _ξ_ = 0 and _|ξ| ≤ ϵ_ almost surely. We are tested directly on _y_ . 

The ERM is 


![](raw/papers/2_LLM_NLP/LLM_paper_list/zero-shot/nips_2023_Zero-shot_causal_learning_images/nips_2023_Zero-shot_causal_learning.pdf-0016-04.png)


The test error is 


![](raw/papers/2_LLM_NLP/LLM_paper_list/zero-shot/nips_2023_Zero-shot_causal_learning_images/nips_2023_Zero-shot_causal_learning.pdf-0016-06.png)


and let _f[∗]_ = min _f L_ ( _f_ ). 

We are interested in bounding the excess error _L_ ( _f_[ˆ] ) _− L_ ( _f[∗]_ ). 

Our key assumption is that interventions with similar attributes ( _w_ ) have similar effects in expectation. More concretely, we assume that all hypotheses in our family are smooth with respect to _w_ : 

## **Assumption 2.** 


![](raw/papers/2_LLM_NLP/LLM_paper_list/zero-shot/nips_2023_Zero-shot_causal_learning_images/nips_2023_Zero-shot_causal_learning.pdf-0016-11.png)


Furthermore, we assume that _PW_ satisfies a Poincaré-type inequality: **Assumption 3.** For some constant _C_ that only depends on _PW_ , for any smooth function _F_ , 

_V arw_ [ _F_ ( _w_ )] _≤ C_ E � _∥∇wF_ ( _w_ ) _∥_ 2[2] � _._ 

For example, _PW_ can be any of the following distributions: 

- Multivariate Gaussian: _w ∈_ R _[e] ∼N_ ( _µ,_ Σ) for some vector _µ ∈_ R _[e]_ and positive semidefinite matrix Σ _∈_ R _[e][×][e]_ ; 

- _w ∈_ R _[e]_ has independent coordinates; each coordinate has the symmetric exponential distribution 1 _/_ 2 _e[−|][t][|]_ for _t ∈_ R. 

- _PW_ is a mixture over base distributions satisfying Poincaré inequalities, and their pair-wise chi-squared distances are bounded. 

- _PW_ is a mixture of isotropic Gaussians in R _[e]_ . 

- _PW_ is the uniform distribution over _W ⊂_ R _[e]_ , which is open, connected, and bounded with Lipschitz boundary. 

We note that isotropic Gaussian can approximate any smooth densities in R _[e]_ [39] (since RBF kernels are universal), showing that Assumption 3 is fairly general. We define a novel notion of function complexity specialized to the zero-shot setting. Intuitively, it measure how well we can fit random labels, which is first drawing _n_ interventions and _m_ recipients for each intervention. For examples of concrete upper bound on zero-shot Rademacher complexity see section A.4. 


![](raw/papers/2_LLM_NLP/LLM_paper_list/zero-shot/nips_2023_Zero-shot_causal_learning_images/nips_2023_Zero-shot_causal_learning.pdf-0016-21.png)


where _σi[j]_[are independently randomly drawn from] _[ {−]_[1] _[,]_[ 1] _[}]_[.] 

16 

## **A.2 Formal theorem statement** 

**Theorem 4.** _Under Assumptions 2 3, with probability_ 1 _− δ,_ 


![](raw/papers/2_LLM_NLP/LLM_paper_list/zero-shot/nips_2023_Zero-shot_causal_learning_images/nips_2023_Zero-shot_causal_learning.pdf-0017-02.png)


## **A.3 Proof of the main theorem** 

˜ We define the population loss on the noisy label _L_[�] ( _f_ ) = E _w,x,y_ ˜ ( _f_ ( _w, x_ ) _− y_ )[2] . Due to independence of _ξ_ , E _w,x,y,ξ_ ( _f_ ( _w, x_ ) _− y − ξ_ )[2] = E _w,x,y_ ( _f_ ( _w, x_ ) _− y_ )[2] + E[ _ξ_[2] ] = _L_ ( _f_ ) + E[ _ξ_[2] ] for any _f_ , so _L_ ( _f_[ˆ] ) _− L_ ( _f[∗]_ ) = _L_[�] ( _f_[ˆ] ) _− L_[�] ( _f[∗]_ ). We shall focus on bounding the latter. 

We first need a lemma that bounds the supremum of an empirical process indexed by a bounded function class. 

**Lemma 5** (Theorem 2.3 of [6]) **.** _Assume that Xj are identically distributed according to P , G is a countable set of functions from X to_ R _and, and all g ∈G are P -measurable, square-integrable, and satisfy_ E[ _g_ ] = 0 _. Suppose_ sup _g∈G ∥g∥∞ ≤_ 1 _, and we denote Z_ = sup _g_ ���� _nj_ =1 _[g]_[(] _[X][j]_[)] ��� _. Suppose σ_[2] _≥_ sup _g∈G V ar_ ( _g_ ( _Xj_ )) _almost surely, the for all t ≥_ 0 _, we have_ 


![](raw/papers/2_LLM_NLP/LLM_paper_list/zero-shot/nips_2023_Zero-shot_causal_learning_images/nips_2023_Zero-shot_causal_learning.pdf-0017-07.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/zero-shot/nips_2023_Zero-shot_causal_learning_images/nips_2023_Zero-shot_causal_learning.pdf-0017-08.png)


The next lemma bounds the variance _σ_[2] in equation (9). 

**Lemma 6.** 


![](raw/papers/2_LLM_NLP/LLM_paper_list/zero-shot/nips_2023_Zero-shot_causal_learning_images/nips_2023_Zero-shot_causal_learning.pdf-0017-11.png)


_Proof of Lemma 6._ Using the law of total variance, if we write 


![](raw/papers/2_LLM_NLP/LLM_paper_list/zero-shot/nips_2023_Zero-shot_causal_learning_images/nips_2023_Zero-shot_causal_learning.pdf-0017-13.png)


then 


![](raw/papers/2_LLM_NLP/LLM_paper_list/zero-shot/nips_2023_Zero-shot_causal_learning_images/nips_2023_Zero-shot_causal_learning.pdf-0017-15.png)


To bound the first term of equation (10), we use Poincaré-type inequalities in Assumption 3. For each of the example distributions, we show that they indeed satisfy Assumption 3. 

17 

**Lemma 7.** _Each of the example distributions in Assumption 3 satisfies a Poincare-type inequality._ 

_Proof._ • When _PW_ is the uniform distribution over _W ∈_ R _[e]_ , which is open, connected, and bounded with Lipschitz boundary, we use Poincaré–Wirtinger inequality [57] on the smooth function E[ _g | w_ ]: For some constant _C_ that only depends on _PW_ , 


![](raw/papers/2_LLM_NLP/LLM_paper_list/zero-shot/nips_2023_Zero-shot_causal_learning_images/nips_2023_Zero-shot_causal_learning.pdf-0018-02.png)


_C_ is the Poincaré constant for the domain _W_ in _L_ 2 norm. It can be bounded by 1 _/λ_ 1 where _λ_ 1 is the first eigenvalue of the negative Laplacian of the manifold _W_ [83]. Many previous works study the optimal Poincaré constants for various domains [43]. For example, when _w_ is uniform over _W_ which is a bounded, convex, Lipschitz domain with diameter _d_ , _C ≤ d/π_ [56]. 

We can apply probabilistic Poincaré inequalities over non-Lebesgue measure _PW_ : 

- When _w ∼N_ ( _µ,_ Σ), we use the Gaussian Poincaré inequality (see _e.g._ Theorem 3.20 of [5] and using change of variables), 


![](raw/papers/2_LLM_NLP/LLM_paper_list/zero-shot/nips_2023_Zero-shot_causal_learning_images/nips_2023_Zero-shot_causal_learning.pdf-0018-06.png)


We apply this with _F_ ( _w_ ) = E[ _g | w_ ]. Since E[ _v[⊤] Av_ ] = E[ _Tr_ ( _v[⊤] Av_ )] = E[ _Tr_ ( _Avv[⊤]_ )] = _Tr_ ( _A_ E[ _vv[⊤]_ ]) _≤∥A∥_ 2E � _∥v∥_ 2[2] �, _V arw_ [E[ _g | w_ ]] _≤∥_ Σ _∥_ 2E � _∥∇w_ E[ _g | w_ ] _∥_ 2[2] � _,_ 

which satisfies equation (11) with _C_ = _∥_ Σ _∥_ 2. 

- When _w ∈_ R _[e]_ has independent coordinates _w_ 1 _, . . . , we_ and each coordinate has the symmetric exponential distribution 1 _/_ 2 _e[−|][t][|]_ for _t ∈_ R, we first bound a single dimension using Lemma 4.1 of [45], which says for any function _k ∈ L_[1] , 


![](raw/papers/2_LLM_NLP/LLM_paper_list/zero-shot/nips_2023_Zero-shot_causal_learning_images/nips_2023_Zero-shot_causal_learning.pdf-0018-10.png)


which, combined with the Efro-Stein inequality (Theorem 3.1 of [5]), 


![](raw/papers/2_LLM_NLP/LLM_paper_list/zero-shot/nips_2023_Zero-shot_causal_learning_images/nips_2023_Zero-shot_causal_learning.pdf-0018-12.png)


yields: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/zero-shot/nips_2023_Zero-shot_causal_learning_images/nips_2023_Zero-shot_causal_learning.pdf-0018-14.png)


which satisfies equation (11) with _C_ = 4. 

Lastly, we consider the case where _PW_ is a mixture over base distributions satisfying Poincaré inequalities. We first consider the case where the pair-wise chi-squared distances are bounded. Next, we show that mixture of isotropic Gaussians satisfies Poincaré inequality without further condition on pair-wise chi-squared distances. 

- When _{PW[q][}][q][∈Q]_[ is a family of distributions, each satisfying Poincaré inequality with con-] stant _C[q]_ , and _PW_ is any mixture over _{PW[q][}][q][∈Q]_[ with density] _[ µ]_[, let] _[ K][P]_[ (] _[µ]_[) =] _[ ess][µ]_[ sup] _q[C][q]_[,] which is an upper bound on the base Poincaré constants almost surely, and _Kχ[p]_[2][(] _[µ]_[)][=] 

E _q,q′∼µ_ [(1 + _χ_[2] ( _PW[q][||][P][ q] W[′]_[))] _[p]_[]][1] _[/p]_[, which is an upper bound on the pairwise] _[ χ]_[2][-divergence.] Using Theorem 1 of [8] we get that _PW_ satisfies Poincaré inequality with constant _C_ such that _C ≤ KP_ ( _µ_ )( _p[∗]_ + _Kχ[p]_[2][(] _[µ]_[))][where] _[p][∗]_[is][the][dual][exponent][of] _[p]_[satisfying] 1 _/p_ + 1 _/p[∗]_ = 1. 

As an example, when base distributions are from the same exponential family and the natural parameter space is affine, such as mixture of Poisson or Multinomial distributions, the pair-wise chi-squared distances are bounded (under some additional conditions) and hence the mixture satisfies Poincaré inequality. More formally, let _pθ_ ( _x_ ) = 

18 

exp � _T_ ( _x_ ) _[⊤] θ − F_ ( _θ_ ) + _k_ ( _x_ )� where _θ ∈_ Θ is the natural parameter space and _A_ ( _θ_ ) is the log partition function. Lemma 1 in [54] shows that _χ_[2] ( _pθ_ 1 _||pθ_ 2) = _e_[(] _[A]_[(2] _[θ]_[2] _[−][θ]_[1][)] _[−]_[(2] _[A]_[(] _[θ]_[2][)] _[−][A]_[(] _[θ]_[1][)))] _−_ 1 _,_ which is bounded as long as 2 _θ_ 2 _− θ_ 1 _∈_ Θ. This is satisfied for mixture of 1-D Poisson distributions which can be written as _p_ ( _w|λ_ ) = _w_ 1![exp (] _[w]_[ log] _[ λ][ −][λ]_[)][with][natural] parameter space R, and mixture of _e_ -dimensional Multinomial distributions _p_ ( _w|π_ ) = exp _⟨w,_ log _π/_ 1 _−_[�] _i[e]_ =1 _[−]_[1] _[π][i] ⟩_ + log 1 _−_[�] _[e] i_ =1 _[−]_[1] _[π][i]_ with natural parameter space � � � �� � �� _R[e][−]_[1] . When applied to Gaussian family the natural parameters are 


![](raw/papers/2_LLM_NLP/LLM_paper_list/zero-shot/nips_2023_Zero-shot_causal_learning_images/nips_2023_Zero-shot_causal_learning.pdf-0019-01.png)


Since the covariance has to be positive definite matrices, 2 _θq − θq′_ may not be a set of valid natural parameter. We deal with this in the next case. 

- When _{PW[q][}][q][∈Q]_[is][a][mixture][of][isotropic][Gaussians,][each][with][mean] _[µ][q][∈]_[R] _[e]_[and][co-] variance Σ _q_ = _σq_[2] _[I][e]_[, each satisfying Poincaré inequality with constant] _[ C][q]_[(in the single-] Gaussian case above we know that _C[q] ≤ σq_[2][),] _[ P][W]_[also satisifes Poincaré inequality.][We] prove this via induction. The key lemma is below: 

**Lemma 8** (Corollary 1 of [64]) **.** _Suppose measure p_ 0 _is absolutely continuous with respect to measure p_ 1 _, and p_ 0 _, p_ 1 _satisfy Poincaré inequality with constants C_ 0 _, C_ 1 _respectively, then for all α ∈_ [0 _,_ 1] _and β_ = 1 _− α, mixture measure p_ = _αp_ 0 + _βp_ 1 _satisfies Poincaré inequality with with C ≤_ max _{C_ 0 _, C_ 1(1 + _αχ_ 1) _} where χ_ 1 = � _ddpp_ 01 _[dp]_[0] _[ −]_[1] _[.]_ 


![](raw/papers/2_LLM_NLP/LLM_paper_list/zero-shot/nips_2023_Zero-shot_causal_learning_images/nips_2023_Zero-shot_causal_learning.pdf-0019-05.png)


The convergence condition of the above integral is 2 _σi_[2] _[≥]_[2] _[σ] j_[2][for all] _[ j][< i]_[ which is satisfied] when _σi_[2] _[≥][σ] j_[2][.] 

Next we observe that 


![](raw/papers/2_LLM_NLP/LLM_paper_list/zero-shot/nips_2023_Zero-shot_causal_learning_images/nips_2023_Zero-shot_causal_learning.pdf-0019-08.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/zero-shot/nips_2023_Zero-shot_causal_learning_images/nips_2023_Zero-shot_causal_learning.pdf-0019-09.png)


Therefore 


![](raw/papers/2_LLM_NLP/LLM_paper_list/zero-shot/nips_2023_Zero-shot_causal_learning_images/nips_2023_Zero-shot_causal_learning.pdf-0019-11.png)


19 

To bound the second term of equation (10), we use concentration of mean of _m i.i.d._ random variables. 

Conditioned on _w_[(] _[j]_[)] , each of the loss ( _f_ ( _w_[(] _[j]_[)] _, x_[(] _i[j]_[)][)] _[ −][y]_[˜] _i_[(] _[j]_[)][)][2][ are] _[ i.i.d.]_[and bounded in][ [0] _[,]_[ (1 +] _[ ϵ]_[)][2][]][.] Hence each variable has variance upper bound ((1 + _ϵ_ )[2] _−_ 0)[2] _/_ 4 = (1 + _ϵ_ )[4] _/_ 4 and the mean has variance upper bound (1 + _ϵ_ )[4] _/_ 4 _m_ . 

Therefore _V ar_ [ _g_ ] _≤_ 4(1 + _ϵ_ )[2] _Cβ_[2] + (1 + _ϵ_ )[4] _/_ 4 _m_ . 

_Proof of Theorem 4._ 


![](raw/papers/2_LLM_NLP/LLM_paper_list/zero-shot/nips_2023_Zero-shot_causal_learning_images/nips_2023_Zero-shot_causal_learning.pdf-0020-04.png)


by equation (9) and Lemma 6. 

We now show that E sup _f |L_[�] ( _f_ ) _− L_[ˆ] ( _f_ ) _| ≤_ 2(1 + _ϵ_ ) _Rnm_ ( _F_ ). This is similar to the argument for classical Rademacher complexity 


![](raw/papers/2_LLM_NLP/LLM_paper_list/zero-shot/nips_2023_Zero-shot_causal_learning_images/nips_2023_Zero-shot_causal_learning.pdf-0020-07.png)


where the first inequality uses Jensen’s inequality and convexity of _sup_ . 

Now we prove the equivalent of Talagrand’s contraction lemma to show that _Rnm_ ( _L_[�] ) _≤_ 2 _Rnm_ ( _F_ ). Note that the squared loss is 2(1 + _ϵ_ )-Lipschitz since ��� _∂_ ( _f∂f−y_ ˜)2 ��� = 2 _|f − y_ ˜ _| ≤_ 2(1 + _ϵ_ ). We use the following lemma to prove this: 

**Lemma 9** (Lemma 5 of [49]) **.** _Suppose {ϕi}, {ψi}, i_ = 1 _, . . . , N are two sets of functions on_ Θ _such that for each i an θ, θ[′] ∈_ Θ _, |ϕi_ ( _θ_ ) _− ϕi_ ( _θ[′]_ ) _| ≤|ψi_ ( _θ_ ) _− ψi_ ( _θ[′]_ ) _|. Then for all functions c:_ Θ _→_ R _,_ 


![](raw/papers/2_LLM_NLP/LLM_paper_list/zero-shot/nips_2023_Zero-shot_causal_learning_images/nips_2023_Zero-shot_causal_learning.pdf-0020-11.png)


For any set of _w, x_ , we apply Lemma 9 with Θ = _F_ , _θ_ = _f_ , _N_ = _nm_ , _ϕij_ ( _f_ ) = ( _f_ ( _w_[(] _[j]_[)] _, x_[(] _i[j]_[)][)] _[ −] y_ ˜ _i_[(] _[j]_[)][)][2][,] _[ ψ][ij]_[(] _[f]_[) = 2(1+] _[ϵ]_[)] _[f]_[(] _[w]_[(] _[j]_[)] _[, x]_[(] _i[j]_[)][)][, and] _[ c]_[(] _[θ]_[) = 0][.][Since] ��( _f − y_ ˜)2 _−_ ( _f ′ − y_ ˜)2�� _≤_ 2(1+ _ϵ_ ) _|f − f[′] |_ , so the condition for Lemma 9 hold. We take expectation over _w, x_ and divide both sides by _nm_ to get 


![](raw/papers/2_LLM_NLP/LLM_paper_list/zero-shot/nips_2023_Zero-shot_causal_learning_images/nips_2023_Zero-shot_causal_learning.pdf-0020-13.png)


which means _Rnm_ ( _L_ ) _≤_ 2(1 + _ϵ_ ) _Rnm_ ( _F_ ). Substituting this into inequality (12) finishes the proof. 

20 

## **A.4 Zero-shot Rademacher complexity bound for the linear hypothesis class** 

Consider the linear classifier _F_ = _{_ ( _w_ 1 _[⊤][w]_[ +] _[ w]_ 2 _[⊤][x]_[ :] _[ ∥][w]_[1] _[∥]_[2] _[≤][B,][ ∥][w]_[1] _[∥]_[2] _[≤][C][}]_[.][Suppose] _[ ∥][w][∥]_[2] _[≤]_[1] and _∥x∥_ 2 _≤_ 1. 


![](raw/papers/2_LLM_NLP/LLM_paper_list/zero-shot/nips_2023_Zero-shot_causal_learning_images/nips_2023_Zero-shot_causal_learning.pdf-0021-02.png)


We observe that the bound is the same as the standard Rademacher complexity for _nm_ independent samples, which is interesting. The relationship between standard and zero-shot Rademacher complexity for other function classes is an important future direction. 

## **B Extended Related Work** 

Our approach to zero-shot prediction of intervention effects is related to recent advances in heterogenous treatment effect (HTE) estimation, zero-shot learning, and meta-learning. 

## **B.1 Heterogenous treatment effect (HTE) estimation** 

**Conditional average treatment effect (CATE) estimation.** A number of approaches have been developed to predict the effect of an existing intervention on an individual or subgroup, based on historical data from individuals who received it. This problem is often referred to in the literature as heterogeneous treatment effect (HTE) estimation [26, 11], to denote that the goal is to detect heterogeneities in how individuals respond to an intervention. A more specific instance of HTE estimation, which we focus on here, is conditional average treatment effect (CATE) estimation [76, 42], in which the goal is to predict the effect of a treatment _conditioned_ on an individual’s features. A variety of methods and specific models have been developed to achieve this goal [26, 32, 21, 28, 76, 66, 1, 84, 24, 86, 25, 14, 12, 42, 34, 11, 2], and we refer to Bica et al. and Curth et al. for a detailed review of these methods [4, 14]. These methods estimate CATE for an existing intervention, based on historical data from individuals who received it and those that did not. 

While these approaches have a number of useful applications, they do not address CATE for novel interventions which did not exist during training (zero-shot). Our primary contribution is a metalearning framework to leverage these existing CATE estimators for zero-shot predictions. In the CaML framework (Figure 2), each task corresponds to predicting CATE for a single intervention. We synthesize a task by sampling a natural experiment for each intervention, and then use any existing CATE estimator to generate a noisy target label for our the task (Step 2: estimate pseudo-outcomes). We rely on pseoudo-outcome estimates as training labels because prior work has shown that training on observed outcomes directly leads to biased CATE estimates [9, 42, 34], a result which we find holds true in our experiments as well (see T-learner and S-learner w/ meta-learning in Tables 2 and 3). 

**Pseudo-outcome estimators.** Prior work has developed a variety of methods to estimate CATE pseudo-outcomes, which are noisy but unbiased estimates of CATE, such as the X-learner [42], R-learner [53], DR-learner [34], and RA-learner [14]. Moreover, the outputs of any other CATE estimation method, such as methods which directly estimate CATE via an end-to-end neural network [32, 66, 68] are an equally valid choice of pseudo-outcome. The literature on pseudo-outcome estimation is growing continuously as new estimators are being developed [19, 38]. Typically, these estimators are specific to a _single binary intervention_ , for which a set of nuisance models are trained and used to compute the pseoudo-outcomes. As such, applying meta-learning algorithms to these pseudo-outcomes requires synthesizing a natural experiment for each intervention, which corresponds to a single task in the CaML framework. 

21 

**Multi-cause estimators.** Our methods to address zero-shot CATE estimation for combinations of interventions are distinct from multi-cause estimators for combinations of binary or categorical interventions [78, 58, 62]. Recent work has shown that these methods can predict the effects of new combinations of interventions [48], when every intervention in the combination has been observed at some point during. However, these methods do not estimate CATE for novel interventions which did not exist during training. By contrast, CaML estimates CATE for zero-shot intervention combinations in which none of the interventions in the combo was ever observed during training (Appendix Table C). 

## **B.2 Zero-shot learning** 

Zero-shot learning (ZSL) has traditionally aimed to reason over new concepts and classes [80, 60] which did not exist during training time. While ZSL has primarily focused on natural language processing and computer vision [77], recent interest has been sparked in generalizing over novel interventions (zero-shot) in the biomedical domain [61, 27] in which data can be cheaply collected for hundreds or thousands of possible interventions [87, 71, 17]. However, general-purpose zero-shot causal methods have been largely unexplored. Notable exceptions include GranITE [23] and SIN [23], which each extend a specific CATE estimation [53, 42] method to incorporate intervention features ( _W_ ). However, these approaches have significant drawbacks, which we discuss in Section 2. 

## **B.3 Meta-learning** 

Meta-learning, or _learning to learn_ , aims to train models which can quickly adapt to new settings and tasks. The key idea is to enable a model to gain experience over multiple learning episodes - in which episodes typically correspond to distinct tasks - to accelerate learning in subsequent learning episodes [30]. The meta-learning literature is rich and spans multiple decades [72, 65, 63, 3], with recent interest focused on model-agnostic methods to train deep learning models to quickly adapt to new tasks [18, 59, 52]. A common focus in the meta-learning literature is few-shot learning, in which a model must adapt to a new task given a small support set of labeled examples. By contrast, we focus on the zero-shot setting, in which no such support set exists. However, we hypothesize that the typical meta-learning problem formulation and training algorithms may also improve zero-shot performance. Thus, CaML’s problem formulation and algorithm inspiration from the meta-learning literature, particularly the Reptile algorithm [52] and its application to other tasks in causal inference [67]. Our experimental results show that this meta-learning formulation improves CaML’s performance, compared to a standard multi-task learning strategy. 

## **C Experimental details** 

## **C.1 Experimental setup** 

Here, we provide more details about the experimental setup for each investigated setting. This serves to complement the high-level overview given in Table 1. Experiments were run using Google Cloud Services. Deep learning-based methods ( _i.e._ , CaML and its ablations, S-learner w/ meta-learning, T-learner w/ meta-learning, SIN, GraphITE, FlexTENET, TARNet, and DragonNet) were run on n1-highmem-64 machines with 4x NVIDIA T4 GPU devices. The remaining baselines (RA-learner, R-learner, X-learner, and T-learner) were run on n1-highmem-64 machines featuring 64 CPUs. 

**Fair comparison.** We perform hyper-parameter optimization with random search for all models, with the meta-testing dataset predetermined and held out. To avoid “hyperparameter hacking”, hyperparameters ranges are consistent between methods wherever possible, and were chosen using defaults similar to prior work [33, 23]. Choice of final model hyper-parameters was determined using performance metrics (specific to each dataset) computed on the meta-validation dataset, using the best hyper-parameters over 48 runs (6 servers x 4 NVIDIA T4 GPUs per server x 2 runs per GPU ) (Appendix C.4). All table results are computed as the mean across 8 runs of the final model with distinct random seeds. 

## **C.1.1 Claims dataset** 

**Interventions (** _W_ **)** : We consider drug prescriptions consisting of either one drug, or two drugs prescribed in combination. We observed 745 unique single drugs, and 22,883 unique drug pairs, 

22 

excluding interventions which occurred less than 500 times. Time of intervention corresponds to the _first_ day of exposure. To obtain intervention information, we generated pre-trained drug embeddings from a large-scale biomedical knowledge graph [7] (see Appendix C.5). Drugs correspond to nodes in the knowledge graph, which are linked to other nodes ( _e.g._ genes, based on the protein target of the drug). Drug combination embeddings are the sum of the embeddings for their constituent drugs. 

**Control group.** A challenge in such causal analyses of clinical settings is defining a control group. We randomly sample 5% (1.52M patients) to use as controls, with a 40/20/40 split betweem metatrain/meta-val/meta-test. When sampling a natural experiment for a given intervention, we select all patients from this control group that did not receive such an intervention. An additional challenge is defining time of intervention for the control group. It is not possible to naively sample a random date, because there are large quiet periods in the claims dataset in which no data is logged. We thus sample a date in which the control patient received _a random drug_ , and thus our measure of CATE estimates the _increase_ in side effect likelihood from the drug(s) _W_ , compared to another drug intervention chosen at random. 

**Outcome (** _Y_ **)** : We focus on the side effect pancytopenia: a deficiency across all three blood cell lines (red blood cells, white blood cells, and platelets). Pancytopenia is life-threatening, with a 1020% mortality rate [36, 41], and is a rare side effect of many common medications [40] ( _e.g._ arthritis and cancer drugs), which in turn require intensive monitoring of the blood work. Our outcome is defined as the (binary) occurrence of pancytopenia within 90 days of intervention exposure. 

**Features (** _X_ **)** : Following prior work [22], patient medical history features were constructed by time-binned counts of each unique medical code (diagnosis, procedure, lab result, drug prescription) before the drug was prescribed. In total, 443,940 features were generated from the following time bins: 0-24 hours, 24-48 hours, 2-7 days, 8-30 days, and 31-90 days, 91-365 days, and 365+ days prior. All individuals in the dataset provided by the insurance company had at least 50 unique days of claims data. 

**Metrics** : We rely on best practices for evaluating CATE estimators as established established by recent work [81, 10], which recommend to assess treatment rules by comparing subgroups across different quantiles of estimated CATE. We follow the high vs. others RATE (rank-weighted average treatment effect) approach from Yadlowsky et. al [81], which computes the difference in average treatment effect (ATE) of the top _u_ percent of individuals (ranked by predicted CATE), versus all individuals: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/zero-shot/nips_2023_Zero-shot_causal_learning_images/nips_2023_Zero-shot_causal_learning.pdf-0023-05.png)


where _S_ ( _·_ ) is a priority score which ranks samples lowest to highest predicted CATE, and _FS_ ( _·_ ) is the cumulative distribution function (CDF) of _S_ ( _Xi_ ). For instance, RATE @ 0.99 would be the difference between the top 1% of the samples (by estimated CATE) vs. the average treatment effect (ATE) across all samples, which we would expect to be high if the CATE estimator is accurate. The real-world use case of our model would be preventing drug prescription a small subset of high-risk individuals. Thus, more specifically, for each task _j_ , intervention _wj_ in the meta-dataset, and meta-model Ψ _θ_ (our priority score _S_ ( _·_ )), we compute _RATE_ @ _u_ for each _u_ in [0 _._ 999 _,_ 0 _._ 998 _,_ 0 _._ 995 _,_ 0 _._ 99] across individuals who received the intervention. 

We now summarize how to estimate RATE performance metrics for a single intervention (task). As RATE performance is calculated separately per-intervention we are concerned with a single intervention, we use the simplified notation ( _i.e. Yi_ (1) instead of _Yi_ ( _w_ )) from Section 3. Due to the fundamental problem of causal inference (we can only observe _Yi_ (0) or _Yi_ (1) for a given sample), the true RATE, as defined above, cannot be directly observed. 

We follow the method outlined in Section 2.2 and 2.4 of Yadlowsky et.� al, [81] in which we compute Γ _i_ , a (noisy but unbiased) estimate for CATE which is in turn used to estimate RATE: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/zero-shot/nips_2023_Zero-shot_causal_learning_images/nips_2023_Zero-shot_causal_learning.pdf-0023-09.png)


Our data is observational, and as such we can estimate Γ[�] _i_ using a direct non-parametric estimator [75]: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/zero-shot/nips_2023_Zero-shot_causal_learning_images/nips_2023_Zero-shot_causal_learning.pdf-0023-11.png)


23 

where _m_ ( _x, w_ ) is a model that predicts the outcome. Here _m_ ˆ ( _x, w_ ) represent nonparametric estimates of _m_ ( _x, w_ ), respectively, which we obtain by fitting a cross-fitting a model to the intervention natural experiment over 5-folds. We use random forest models for _m_ ˆ ( _x, w_ ), as they perform well (achieving _≥_ 0 _._ 90 ROC AUC across all meta-testing tasks for predicting outcomes) and are robust to choice of hyperparameters. 

RATE can then be estimated via sample-averaging estimator. Specifically, we compute the difference between the average value of Γ[�] _i_ for those in the top _u_ percent of individuals (based on our metamodel’s predictions), compared to the average Γ[�] _i_ across all individuals. For further discussion on estimating RATE, we refer readers to [81]. Note that estimates of RATE are _unbounded_ : RATE can be less than 0 (due to predictions inversely relating to CATE). 

Finally, because our meta-testing dataset consists of individuals treated with drugs _known_ in the medical literature to cause pancytopenia (identified by filtering drugs using the side effect database SIDER [40]), observational metrics of recall and precision are also a rough _proxy_ for successful CATE estimation. Thus, as secondary metrics, we also compute _Recall_ @ _u_ and _Precision_ @ _u_ for the same set of thresholds as RATE, where a positive label is defined as occurrence of pancytopenia after intervention. We find that these metrics are highly correlated to RATE in our performance results. 

**Training & Evaluation:** For each method, we ran a hyperparameter search with 48 random configurations (48 due to running 8 jobs in parallel on 6 servers each) that were drawn uniformly from a pre-defined hyperparameter search space (see Appendix C.4). Methods that can be trained on multiple tasks to then be applied to tasks unseen during training ( _i.e._ , CaML and its ablations, S-learner w/ meta-learning, T-learner w/ meta-learning, SIN, GraphITE) were trained for 24 hours (per run) on the meta-training tasks. Model selection was performed on the meta-validation tasks by maximizing the mean RATE@0 _._ 998 across meta-validation tasks. Then, the best hyperparameter configuration was used to fit 8 repetition runs across 8 different random seeds. Each repetition model was then tested on the meta-testing tasks, where for all metrics averages across the testing tasks are reported. To make the setting of multi-task models comparable with single-task models that were trained on meta-testing tasks (requiring a train and test split of each meta-testing task), the evaluation of all models was computed on the test split of the meta-testing tasks, respectively. Single-task baselines (FlexTENET, TARNet, and DragonNet, RA-learner, R-learner, X-learner, and T-learner) were given access to the meta-testing tasks during training. Specifically, model selection was performed on the meta-validation tasks, while the best hyperparameter configuration was used to train 8 repetition models (using 8 random seeds) on the train split of each meta-testing task. For the final evaluation, each single-task model that was fit on meta-testing task _i_ was tested on the test split of the same meta-testing task _i_ , and the average metrics were reported across meta-testing tasks. 

## **C.1.2 LINCS** 

**Interventions (** _W_ **)** : Interventions in the LINCS dataset consist of a single perturbagen (small molecule). For intervention information, we used the molecular embeddings for each perturbagen using the RDKit featurizer The same cell line-perturbagen combinations are tested with different perturbagen dosages and times of exposure. [44].To maintain consistency in experimental conditions while also ensuring that the dataset is sufficiently large for training a model, we filter for most frequently occurring dosage and time of exposure in the dataset, which are 10 _µM_ and 24 hours, respectively. We use data from 10,322 different perturbagens. 

**Control group.** For each perturbagen (at a given timepoint and dose), we use cell lines which did not receive that intervention as the control group. 

**Outcomes (** _Y_ **)** : We measure gene expression across the top-50 and top-20 landmark differentially expressed genes (DEGs) in the LINCS dataset. Accurately predicting in gene expression in these DEGs is most crucial to the drug discovery process. 

**Features (** _X_ **)** : We use 19 _,_ 221 features from the Cancer Cell Line Encyclopedia (CCLE) [20] to describe each cell-line, based on historical gene expression values in a different lab environment. Our dataset consisted of 99 unique cell lines (after filtering for cell-lines with CCLE features). 

**Metrics** : A key advantage of experiments on cells is that at evaluation time we can observe both _Y_ (0) and _Y_ (1) for the same cell line _X_ , through multiple experiments on clones of the same cellline in controlled lab conditions. In the LINCS dataset, _Y_ (0) is also measured for all cells which 

24 

received an intervention. Thus, we can directly compute the Precision Estimation of Heterogenous Effects (PEHE) on all treated cells in our meta-testing dataset. PEHE is a standard metric for CATE estimation performance [28], analagous to mean squared error (MSE). 


![](raw/papers/2_LLM_NLP/LLM_paper_list/zero-shot/nips_2023_Zero-shot_causal_learning_images/nips_2023_Zero-shot_causal_learning.pdf-0025-01.png)


**Training & Evaluation:** For each method, we ran a hyperparameter search with 48 random configurations (48 due to running 8 jobs in parallel on 6 servers each) that were drawn uniformly from a pre-defined hyperparameter search space (see Appendix C.4). Methods that can be trained on multiple tasks to then be applied to tasks unseen during training ( _i.e._ , CaML and its ablations, S-learner w/ meta-learning, T-learner w/ meta-learning, SIN) were trained for 12 hours (per run) on the meta-training tasks. Model selection was performed on the meta-validation tasks by minimizing the overall PEHE for the Top-20 most differentially expressed genes (DEGs) across meta-validation tasks. Then, the best hyperparameter configuration was used to fit 8 repetition runs across 8 different random seeds. Each repetition model was then tested on the meta-testing tasks, where for all metrics averages across the testing tasks are reported. 

## **C.2 Selecting holdout interventions for meta-validation and meta-testing** 

## **C.2.1 Claims.** 

In the 30.4 million patient insurance claims dataset, each intervention task in meta-train/metaval/meta-testing corresponds to a natural experiment of multiple patients, with some interventions ( _e.g._ commonly prescribed drugs) having millions of associated patients who were prescribed the drug. One challenge is that in this setting, there is overlap in subjects between the natural experiments sampled by CaML, which can lead to data leakage between training and testing. For instance, if a patient received Drug 1 (in meta-test) and Drug 2 (meta-train), they would appear in both natural experiments, resulting in data leakage. 

We take a conservative approach and exclude all patients who have ever received a meta-testing drug in their lifespan from the natural experiments for meta-val/meta-train. Similarly, we exclude all patients who received a meta-validation drug from meta-training. 

This approach means we must take great care in selecting meta-testing drugs. Specifically, we must trade off between selecting drugs that are important (covering enough patients) while not diminishing the training dataset size. For instance selecting a commonly prescribed ( _e.g._ aspirin) for meta-testing would deplete our meta-training dataset by over 50% of patients. Thus we only selected meta-test/meta-validation drugs which were prescribed to between 1,000,000 and 100K patients in our dataset, after filtering for only drugs which known to cause Pancytopenia [40] (using the SIDER database). From this subset of drugs, we randomly selected 10 meta-testing drugs and 2 meta-validation drugs, resulting in a total meta-testing/meta-validation pool of 4.1 million patients and 685K patients respectively. 

To evaluate on unseen pairs of drugs on the same hold-out test dataset, we additionally created a second pairs testing dataset from the 5 most frequently occurring combinations from the metatesting dataset. This allowed us to train a single model on the same meta-train split and evaluate on both single drug and drug pair interventions without occurrence of data leakage. Designing a larger evaluation of pairs was not possible because while pairs of drugs are commonly prescribed as intervention, each particular pair of drugs is a rare event, and accurately evaluating CATE estimation performance (for a rare outcome such as Pancytopenia) requires amassing a natural experiment with at least several thousand patients who received the same intervention. 

## **C.2.2 LINCS.** 

The goal in selecting holdout interventions for the meta-validation and meta-testing sets was to ensure that they consisted of both cell lines and tasks (small molecules) that had not been seen previously at the time of training (i.e. zero-shot on cell lines and tasks). 

Using a random data splitting approach would result in large portions (up to 50%) of the data being unused to comply with the zero-shot requirements on cell lines and tasks. One approach to tackle this 

25 

was to reserve only those tasks in the held-out sets which had been tested on the fewest cell lines. This preserved the maximum amount of data but resulted in an average of just 1 cell line per task in the meta-testing and meta-validation sets, which would not be fair to the non-zero shot baselines. 

To address these issues, we designed a new data split procedure that exploits the structure of how tasks and cell lines are paired. To do so, We first clustered tasks by the cell lines they are tested on. We then identified a set of 600 drugs that had all been tested on a shared set of roughly 20 cell lines. We divided the cell lines and tasks within this set into the meta-validation and meta-testing set, while enforcing zero-shot constraints on both. This resulted in roughly 10 cell lines per intervention in both the meta-validation and meta-testing sets, while still maintaining a reasonably large size of 11 distinct cell lines and 300 distinct tasks in both sets. All remaining tasks and cell lines were reserved for the training set. (See Table 8) 

## **C.3 Understanding CaML’s performance** 

Our comparison to CATE estimators which are restricted to single interventions (Grey, Table 2,B.3) shows that a key reason for CaML’s strong performance is the ability to joinly learn across from many intervention datasets, in order to generalize to unseen intervention. 

Additionally, in both the Claims and LINCS settings, we conduct two key ablation studies to further understand the underlying reason for CaML’s strong performance results. 

In our first ablation experiment (w/o meta-learning), we trained the CaML model without employing meta-learning, instead using the standard empirical risk minimization (ERM) technique [73]. This can be seen as a specific implementation of the CaML algorithm (refer to Algorithm 1) when _k_ = 1 [52]. The results of this experiment showed a varying degree of performance deterioration across our primary tests. In the Claims settings, we observed a decrease in the RATE performance metric by 15%-22% (refer to Table 2), while in the LINCS settings, the PEHE performance metric decreased by approximately 0.01 (see Table 3). These results indicate that the absence of meta-learning affects the model’s performance, although the impact varies depending on the specific setting. An important detail to consider is that the Claims data experiments dealt with substantially larger datasets, each comprising hundreds of thousands of patients per intervention. This extensive scale of data potentially amplifies the benefits of using meta-learning in the CaML model for the Claims dataset. The larger dataset enables the model to adapt to a given task over a larger set of iterations without reusing the same data, thereby enhancing the efficacy of meta-learning. 

Our second ablation (w/o RA-learner) assesses the sensitivity of CaML’s performance to different pseudo-outcome estimation strategies. A key aspect of CaML is _flexibility_ in choice of any pseudooutcome estimator to infer CATE, in contrast to prior work which uses specific CATE estimation strategies [23, 33]. We find that CaML performance benefits strongly from flexibility of pseudooutcome estimator choice. We assess this by using an alternative pseudo-outcome estimator. Firstly, we find that this ablation results in much noisier model training. For instance, the standard deviation in RATE across the 8 random seeds increases by 20 _×_ when using the alternative pseudo-outcome estimator in the claims setting. Moreover, the alternative pseudo-outcome estimator typyically worsens performance, decreasing RATE by up to 6% in the Claims setting , and increasing PEHE by 20%-21% in the LINCS setting (Table 3). We note that this ablation performs slightly better at the 0.99 threshold, which may be a result of the high variance in this ablation. Specific choice of alternative pseudo-outcome estimator for this ablation varies by setting. We use the R-learner [53] for Claims as it also achieves strong single task performance (Table 2, grey) on Claims data. However, R-learner is restricted to single-dimensional outcomes, and thus for LINCS (in which outcomes are 50 and 20 dimensional), we use the PW-learner instead [14]. 

## **C.4 Hyperparameter space** 

## **C.4.1 Claims dataset hyperparameter space** 

We list the hyperparameter search spaces for the medical claims dataset in the following tables. Table 9 represents the search space for CaML. The SIN baseline consists of two stages, Stage 1 and Stage 2. For the Stage 1 model, we searched the identical hyperparameter search space as for CaML (Table 9). For Stage 2, we used the hyperparameters displayed in Table 10. The search space for the GraphITE baseline is displayed in Table 11. For the S-learner and T-learner w/ meta-learning baselines, we use 

26 

the same hyperparameter space as for CaML (Table 9) with the only major difference that the these baselines predicts the outcome _Y_ instead of _τ_ ˆ. For all deep learning-based methods, we employed a batch size of 8 _,_ 192, except for GraphITE, where we were restricted to using a batch size of 512 due to larger memory requirements. Single-task neural network baselines (FlexTENet, TARNet, and DragonNet) are shown in Tables 12,13, and 14, respectively. For the remaining baselines, i.e., the model-agnostic CATE estimators, the (shared) hyperparameter search space is shown in Table 15. Finally, applied L1 regularization to the encoder layer of the customizable neural network models (that were not reused as external packages), i.e., SIN learner, GraphITE, T-learner w/ meta-learning, and S-learner w/ meta-learning, and CaML. 

## **C.4.2 LINCS hyperparameter space** 

We list the hyperparameter search spaces for LINCS in the following tables. CaMLis shown in Table 16. SIN Stage 1 used the same search space as CaML (Table 16. The search space of SIN Stage 2 is shown in Table 17. S learner and T-learner w/ meta-learning used the same search space as CaML. The search space of GraphITE is shown in Table 18. All methods that were applied to LINCS used a batch size of 20. 

## **C.5 More details on intervention information** 

Here we give more details about the intervention information used for the medical claims dataset. In order to perform zero-shot generalization, we acquired information about a specific intervention through the use of pretrained embeddings. We generated these embeddings on the Precision Medicine Knowledge Graph [7] that contains drug nodes as well as 9 other node types. We extracted embeddings for 7957 drugs from the knowledge graph. 

To extract rich neighborhood information from the knowledge graph we used Stargraph [47], which is a coarse-to-fine representation learning algorithm. StarGraph generates a subgraph for each node by sampling from its neighbor nodes (all nodes in the one-hop neighborhood) and anchor nodes (a preselected subset of nodes appearing in the multihop neighborhood). In our case the anchor nodes were the 2% of graph nodes with the highest degree. For the scoring function we used the augmented version of TripleRE [85] presented in the StarGraph article [47]. 

We performed a hyperparameter optimization to compare different models and determine the one we used to calculate our final embeddings (see Table C.5). The hyperparameter search was random with the objective of minimizing the loss function used in training on held out data. The search range for each of the parameters is displayed in C.5. Since certain parameters did not seem to influence the final score as much we decided to use them as constants and focus on optimizing the hyperparameters in the table. Therefore the number of sampled anchors was set to 20 and _u_ = 0 _._ 1 in the augmented TripleRE function, the values matching those seen in Stargraph [46]. 

Our final embeddings were 256-dimensional, the learning rate was 2e-4, the drop-ratio was 5e-3. We used the self-adversarial negative sampling loss with _γ_ = 8 and we sampled 4 neighbor nodes for each subgraph. 

To additionally evaluate the quality of the embeddings we assigned classes to drug combinations and then scored them using multiple clustering metrics. We were interested to see if embeddings of drug combinations used for similar purposes would be embedded closer together than other drug combinations. For the class label of single drugs we used the first level of the Anatomical Therapeutic Chemical (ATC) code, which represents one of the 14 anatomical or pharmacological groups. Since certain medications have more than one ATC code, we took the mode of all labels for a specific drug. For multiple drugs we combined all distinct first level values and took the mode of them as the label. We used the Silhouette metric, Calinski Harabasz index and Davies Bouldin index as well as the average classification accuracy over 10 runs of training a random forest classifier on a random sample of 80% of the dataset and evaluating on the remaining 20%. Out of all tested embeddings the hyperparameter optimized StarGraph embeddings performed best (exceeding 93% in the classification accuracy metric). 

27 

## **C.6 Pseudo-outcome estimation** 

In our experiments, we estimate pseudo-outcomes _τ_ ˜ for a given intervention _w_ using the RAlearner [14]: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/zero-shot/nips_2023_Zero-shot_causal_learning_images/nips_2023_Zero-shot_causal_learning.pdf-0028-02.png)


Furthermore, in both settings we only estimate CATE for treated individuals. We focus on treated individuals in the Claims setting because we care about the risk of an adverse event for prescribing a sick patients drugs that may cure their sickness, not the adverse event risk of prescribing healthy patients drugs (which is of less clinical interest). In the LINCS setting, we focus on treated cells as for these cell-lines _Y_ (0) is also measured from a cloned cell-line under similar laboratory conditions, which allows us to directly estimate CATE prediction performance using the PEHE metric. As we ˜ ˆ ˆ focus on treated samples, the RA-learner can be simplified to _τ_ = _Y − µ_ 0( _X_ ). We estimate _µ_ 0( _X_ ) using a random forest model in the Claims setting, whereas in the LINCS setting we use the point estimate from the untreated control cell line’s gene expression. 

## **C.7 Baselines** 

Here we provide more details on the baselines used in our experiments. 

_Trained on test task:_ These baselines leverage CATE estimators which can only be trained on a single task (typically these are the strongest baselines, when there is a large enough dataset for a single task). Thus, we train a single model for each meta-testing task on its train split, and evaluate performance on its test split. We use a number of strong baselines for CATE estimation developed by prior work including both model-agnostic and end-to-end deep learning approaches: T-learner. Specifically, we use the model-agnostic CATE estimators: [42], X-learner [42], RA-learner [14], R-learner [53]. We additionally use the end-to-end deep learning estimators DragonNet [68], TARNet [66], and FlexTENet [15], using implementations from [15]. For model-agnostic CATE estimators, we use random forest models following prior work [12, 76]. 

_Zero-shot_ . These baselines use CATE estimators which incorporate intervention information ( _W_ ) and are capable of multi-task learning. We train these baselines on all meta-training tasks. These baselines have no access to the meta-testing tasks during training. We found in preliminary experiments that in some cases, baseline models trained with vanilla ERM would not even converge. To allow for fair comparison to baselines, we allow for all zero-shot baselines to be trained using Reptile (by training using the same optimization strategy as Algorithm 1, while allowing for training with ERM by including _k_ = 1 in the hyperparameter search space). 

Firstly, we use GraphITE [23] and Structured Intervention Networks [33]. These are, to the best of our knowledge, the only methods from prior work which are (in principle) capable of zero-shot generalization. We use existing implementations provided by the authors [33]. 

Additionally, we implement two strong baselines which estimate CATE by modeling potential outcomes, rather than via pseudo-outcomes. These are variants of the S-learner and T-learner [42] with meta-learning, which use the intervention information as input, rather than one-hot encoded vectors of the different interventions—such that they also have zero-shot capability. Specifically, we train MLPs using the same architecture as CaML to estimate the response function from observed outcomes: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/zero-shot/nips_2023_Zero-shot_causal_learning_images/nips_2023_Zero-shot_causal_learning.pdf-0028-10.png)


and estimate CATE by 


![](raw/papers/2_LLM_NLP/LLM_paper_list/zero-shot/nips_2023_Zero-shot_causal_learning_images/nips_2023_Zero-shot_causal_learning.pdf-0028-12.png)


Where _w_ denotes the corresponding intervention information _w_ for an intervention, and **0** denotes a null intervention vector. In the LINCS setting, we represent **0** as a vector of zeros, whereas in the Claims setting we represent **0** as the mean embedding of all drugs (as the estimand is the increase in adverse event likelihood compared to a randomly chosen drug). The difference between the T-learner and the S-learner is that the T-learner estimates two models, one for control units and one for treated units. By contrast, the S-learner estimates a shared model across all units. 

28 

|RATE @_u_ (_↑_)<br>Recall @_u_ (_↑_)<br>Precision @_u_ (_↑_)<br>0_._999<br>_._998<br>0_._995<br>0_._99<br>0_._999<br>0_._998<br>0_._995<br>0_._99<br>0_._999<br>0_._998<br>0_._995<br>0_._99<br>Random<br>0.00_±<_0.001 0.00_±<_0.001 0.00_±<_0.001<br>0.00_±<_0.001<br>0.00_±<_0.001 0.00_±<_0.001 0.01_±<_0.001 0.00_±<_0.001<br>0.00_±<_0.001 0.00_±<_0.001 0.00_±<_0.001<br>0.00_±<_0.001<br>T-learner<br>0.32_±<_0.001<br>0.26_±<_0.001<br>0.16_±<_0.001<br>0.10_±<_0.001<br>0.12_±<_0.001<br>0.18_±<_0.001<br>0.26_±<_0.001<br>0.31_±<_0.001<br>0.36_±<_0.001<br>0.29_±<_0.001<br>0.18_±<_0.001<br>0.11_±<_0.001<br>X-learner<br>0.06_±<_0.001<br>0.05_±<_0.001<br>0.04_±<_0.001<br>0.03_±<_0.001<br>0.02_±<_0.001<br>0.04_±<_0.001<br>0.08_±<_0.001<br>0.12_±<_0.001<br>0.09_±<_0.001<br>0.07_±<_0.001<br>0.06_±<_0.001<br>0.05_±<_0.001<br>R-learner<br>0.19_±<_0.001<br>0.17_±<_0.001<br>0.12_±<_0.001<br>0.08_±<_0.001<br>0.06_±<_0.001<br>0.10_±<_0.001<br>0.19_±<_0.001<br>0.26_±<_0.001<br>0.24_±<_0.001<br>0.21_±<_0.001<br>0.15_±<_0.001<br>0.11_±<_0.001<br>RA-learner<br>0.47<br>_±_0.001<br>0.37<br>_±<_0.001<br>0.23<br>_±<_0.001<br>0.14<br>_±<_0.001<br>0.17<br>_±<_0.001<br>0.26<br>_±<_0.001<br>0.38<br>_±<_0.001<br>0.45<br>_±<_0.001<br>0.54<br>_±_0.001<br>0.42<br>_±<_0.001<br>0.26<br>_±<_0.001<br>0.16<br>_±<_0.001<br>DragonNet<br>0.09_±_0.037<br>0.07_±_0.030<br>0.05_±_0.019<br>0.04_±_0.013<br>0.02_±_0.008<br>0.04_±_0.012<br>0.07_±_0.020<br>0.10_±_0.027<br>0.12_±_0.045<br>0.10_±_0.036<br>0.07_±_0.023<br>0.05_±_0.015<br>TARNet<br>0.15_±_0.011<br>0.12_±_0.011<br>0.07_±_0.006<br>0.05_±_0.004<br>0.05_±_0.003<br>0.08_±_0.006<br>0.12_±_0.008<br>0.14_±_0.011<br>0.18_±_0.013<br>0.15_±_0.012<br>0.09_±_0.007<br>0.06_±_0.004<br>FlexTENet<br>0.10_±_0.015<br>0.09_±_0.016<br>0.06_±_0.008<br>0.04_±_0.006<br>0.04_±_0.006<br>0.07_±_0.009<br>0.12_±_0.011<br>0.17_±_0.017<br>0.12_±_0.018<br>0.11_±_0.019<br>0.08_±_0.010<br>0.06_±_0.007<br>GraphITE<br>0.19_±_0.024<br>0.12_±_0.013<br>0.05_±_0.004<br>0.03_±_0.002<br>0.07_±_0.009<br>0.08_±_0.010<br>0.09_±_0.008<br>0.10_±_0.008<br>0.23_±_0.027<br>0.14_±_0.015<br>0.07_±_0.005<br>0.04_±_0.003<br>SIN<br>0.00_±_0.002<br>0.00_±_0.001<br>0.00_±_0.001<br>0.00_±_0.001<br>0.00_±_0.001<br>0.00_±_0.001<br>0.01_±_0.001<br>0.02_±_0.002<br>0.01_±_0.002<br>0.01_±_0.001<br>0.01_±_0.001<br>0.01_±_0.001<br>S-learner w/ meta-learning<br>0.21_±_0.032<br>0.16_±_0.028<br>0.09_±_0.020<br>0.05_±_0.012<br>0.08_±_0.013<br>0.11_±_0.022<br>0.15_±_0.035<br>0.16_±_0.038<br>0.25_±_0.034<br>0.18_±_0.031<br>0.10_±_0.023<br>0.06_±_0.014<br>T-learner w/ meta-learning<br>0.40_±_0.012<br>0.31_±_0.010<br>0.18_±_0.007<br>0.11_±_0.004<br>0.15_±_0.006<br>0.22_±_0.008<br>0.32_±_0.013<br>0.38_±_0.014<br>0.45_±_0.013<br>0.35_±_0.011<br>0.21_±_0.008<br>0.13_±_0.004<br>CaML - w/o meta-learning<br>0.39_±_0.012<br>0.31_±_0.006<br>0.18_±_0.008<br>0.11_±_0.006<br>0.15_±_0.005<br>0.22_±_0.007<br>0.32_±_0.014<br>0.39_±_0.021<br>0.45_±_0.010<br>0.35_±_0.006<br>0.22_±_0.008<br>0.14_±_0.006<br>CaML - w/o RA-learner<br>0.45_±_0.058<br>0.36_±_0.066<br>0.22_±_0.067<br>**0.14**_±_0.041<br>0.16_±_0.020<br>0.24_±_0.019<br>0.35_±_0.016<br>0.41_±_0.023<br>0.51_±_0.076<br>0.41_±_0.082<br>0.26_±_0.078<br>**0.16**_±_0.048<br>CaML (ours)<br>**0.48**_±_0.010<br>**0.38**_±_0.007<br>**0.23**_±_0.003<br>0.13_±_0.002<br>**0.18**_±_0.004<br>**0.27**_±_0.005<br>**0.38**_±_0.006<br>**0.45**_±_0.010<br>**0.54**_±_0.012<br>**0.43**_±_0.008<br>**0.26**_±_0.004<br>0.16_±_0.003<br>Table 4: Performance results for the Claims dataset (predicting pancytopenia onset from drug exposure using patient medical history. This table extends Table 2 with<br>standard deviations.|RATE @_u_ (_↑_)<br>Recall @_u_ (_↑_)<br>Precision @_u_ (_↑_)<br>0_._999<br>_._998<br>0_._995<br>0_._99<br>0_._999<br>0_._998<br>0_._995<br>0_._99<br>0_._999<br>0_._998<br>0_._995<br>0_._99<br>Random<br>0.00_±<_0.001 0.00_±<_0.001 0.00_±<_0.001<br>0.00_±<_0.001<br>0.00_±<_0.001 0.00_±<_0.001 0.01_±<_0.001 0.00_±<_0.001<br>0.00_±<_0.001 0.00_±<_0.001 0.00_±<_0.001<br>0.00_±<_0.001<br>T-learner<br>0.32_±<_0.001<br>0.26_±<_0.001<br>0.16_±<_0.001<br>0.10_±<_0.001<br>0.12_±<_0.001<br>0.18_±<_0.001<br>0.26_±<_0.001<br>0.31_±<_0.001<br>0.36_±<_0.001<br>0.29_±<_0.001<br>0.18_±<_0.001<br>0.11_±<_0.001<br>X-learner<br>0.06_±<_0.001<br>0.05_±<_0.001<br>0.04_±<_0.001<br>0.03_±<_0.001<br>0.02_±<_0.001<br>0.04_±<_0.001<br>0.08_±<_0.001<br>0.12_±<_0.001<br>0.09_±<_0.001<br>0.07_±<_0.001<br>0.06_±<_0.001<br>0.05_±<_0.001<br>R-learner<br>0.19_±<_0.001<br>0.17_±<_0.001<br>0.12_±<_0.001<br>0.08_±<_0.001<br>0.06_±<_0.001<br>0.10_±<_0.001<br>0.19_±<_0.001<br>0.26_±<_0.001<br>0.24_±<_0.001<br>0.21_±<_0.001<br>0.15_±<_0.001<br>0.11_±<_0.001<br>RA-learner<br>0.47<br>_±_0.001<br>0.37<br>_±<_0.001<br>0.23<br>_±<_0.001<br>0.14<br>_±<_0.001<br>0.17<br>_±<_0.001<br>0.26<br>_±<_0.001<br>0.38<br>_±<_0.001<br>0.45<br>_±<_0.001<br>0.54<br>_±_0.001<br>0.42<br>_±<_0.001<br>0.26<br>_±<_0.001<br>0.16<br>_±<_0.001<br>DragonNet<br>0.09_±_0.037<br>0.07_±_0.030<br>0.05_±_0.019<br>0.04_±_0.013<br>0.02_±_0.008<br>0.04_±_0.012<br>0.07_±_0.020<br>0.10_±_0.027<br>0.12_±_0.045<br>0.10_±_0.036<br>0.07_±_0.023<br>0.05_±_0.015<br>TARNet<br>0.15_±_0.011<br>0.12_±_0.011<br>0.07_±_0.006<br>0.05_±_0.004<br>0.05_±_0.003<br>0.08_±_0.006<br>0.12_±_0.008<br>0.14_±_0.011<br>0.18_±_0.013<br>0.15_±_0.012<br>0.09_±_0.007<br>0.06_±_0.004<br>FlexTENet<br>0.10_±_0.015<br>0.09_±_0.016<br>0.06_±_0.008<br>0.04_±_0.006<br>0.04_±_0.006<br>0.07_±_0.009<br>0.12_±_0.011<br>0.17_±_0.017<br>0.12_±_0.018<br>0.11_±_0.019<br>0.08_±_0.010<br>0.06_±_0.007<br>GraphITE<br>0.19_±_0.024<br>0.12_±_0.013<br>0.05_±_0.004<br>0.03_±_0.002<br>0.07_±_0.009<br>0.08_±_0.010<br>0.09_±_0.008<br>0.10_±_0.008<br>0.23_±_0.027<br>0.14_±_0.015<br>0.07_±_0.005<br>0.04_±_0.003<br>SIN<br>0.00_±_0.002<br>0.00_±_0.001<br>0.00_±_0.001<br>0.00_±_0.001<br>0.00_±_0.001<br>0.00_±_0.001<br>0.01_±_0.001<br>0.02_±_0.002<br>0.01_±_0.002<br>0.01_±_0.001<br>0.01_±_0.001<br>0.01_±_0.001<br>S-learner w/ meta-learning<br>0.21_±_0.032<br>0.16_±_0.028<br>0.09_±_0.020<br>0.05_±_0.012<br>0.08_±_0.013<br>0.11_±_0.022<br>0.15_±_0.035<br>0.16_±_0.038<br>0.25_±_0.034<br>0.18_±_0.031<br>0.10_±_0.023<br>0.06_±_0.014<br>T-learner w/ meta-learning<br>0.40_±_0.012<br>0.31_±_0.010<br>0.18_±_0.007<br>0.11_±_0.004<br>0.15_±_0.006<br>0.22_±_0.008<br>0.32_±_0.013<br>0.38_±_0.014<br>0.45_±_0.013<br>0.35_±_0.011<br>0.21_±_0.008<br>0.13_±_0.004<br>CaML - w/o meta-learning<br>0.39_±_0.012<br>0.31_±_0.006<br>0.18_±_0.008<br>0.11_±_0.006<br>0.15_±_0.005<br>0.22_±_0.007<br>0.32_±_0.014<br>0.39_±_0.021<br>0.45_±_0.010<br>0.35_±_0.006<br>0.22_±_0.008<br>0.14_±_0.006<br>CaML - w/o RA-learner<br>0.45_±_0.058<br>0.36_±_0.066<br>0.22_±_0.067<br>**0.14**_±_0.041<br>0.16_±_0.020<br>0.24_±_0.019<br>0.35_±_0.016<br>0.41_±_0.023<br>0.51_±_0.076<br>0.41_±_0.082<br>0.26_±_0.078<br>**0.16**_±_0.048<br>CaML (ours)<br>**0.48**_±_0.010<br>**0.38**_±_0.007<br>**0.23**_±_0.003<br>0.13_±_0.002<br>**0.18**_±_0.004<br>**0.27**_±_0.005<br>**0.38**_±_0.006<br>**0.45**_±_0.010<br>**0.54**_±_0.012<br>**0.43**_±_0.008<br>**0.26**_±_0.004<br>0.16_±_0.003<br>Table 4: Performance results for the Claims dataset (predicting pancytopenia onset from drug exposure using patient medical history. This table extends Table 2 with<br>standard deviations.|RATE @_u_ (_↑_)<br>Recall @_u_ (_↑_)<br>Precision @_u_ (_↑_)<br>0_._999<br>_._998<br>0_._995<br>0_._99<br>0_._999<br>0_._998<br>0_._995<br>0_._99<br>0_._999<br>0_._998<br>0_._995<br>0_._99<br>Random<br>0.00_±<_0.001 0.00_±<_0.001 0.00_±<_0.001<br>0.00_±<_0.001<br>0.00_±<_0.001 0.00_±<_0.001 0.01_±<_0.001 0.00_±<_0.001<br>0.00_±<_0.001 0.00_±<_0.001 0.00_±<_0.001<br>0.00_±<_0.001<br>T-learner<br>0.32_±<_0.001<br>0.26_±<_0.001<br>0.16_±<_0.001<br>0.10_±<_0.001<br>0.12_±<_0.001<br>0.18_±<_0.001<br>0.26_±<_0.001<br>0.31_±<_0.001<br>0.36_±<_0.001<br>0.29_±<_0.001<br>0.18_±<_0.001<br>0.11_±<_0.001<br>X-learner<br>0.06_±<_0.001<br>0.05_±<_0.001<br>0.04_±<_0.001<br>0.03_±<_0.001<br>0.02_±<_0.001<br>0.04_±<_0.001<br>0.08_±<_0.001<br>0.12_±<_0.001<br>0.09_±<_0.001<br>0.07_±<_0.001<br>0.06_±<_0.001<br>0.05_±<_0.001<br>R-learner<br>0.19_±<_0.001<br>0.17_±<_0.001<br>0.12_±<_0.001<br>0.08_±<_0.001<br>0.06_±<_0.001<br>0.10_±<_0.001<br>0.19_±<_0.001<br>0.26_±<_0.001<br>0.24_±<_0.001<br>0.21_±<_0.001<br>0.15_±<_0.001<br>0.11_±<_0.001<br>RA-learner<br>0.47<br>_±_0.001<br>0.37<br>_±<_0.001<br>0.23<br>_±<_0.001<br>0.14<br>_±<_0.001<br>0.17<br>_±<_0.001<br>0.26<br>_±<_0.001<br>0.38<br>_±<_0.001<br>0.45<br>_±<_0.001<br>0.54<br>_±_0.001<br>0.42<br>_±<_0.001<br>0.26<br>_±<_0.001<br>0.16<br>_±<_0.001<br>DragonNet<br>0.09_±_0.037<br>0.07_±_0.030<br>0.05_±_0.019<br>0.04_±_0.013<br>0.02_±_0.008<br>0.04_±_0.012<br>0.07_±_0.020<br>0.10_±_0.027<br>0.12_±_0.045<br>0.10_±_0.036<br>0.07_±_0.023<br>0.05_±_0.015<br>TARNet<br>0.15_±_0.011<br>0.12_±_0.011<br>0.07_±_0.006<br>0.05_±_0.004<br>0.05_±_0.003<br>0.08_±_0.006<br>0.12_±_0.008<br>0.14_±_0.011<br>0.18_±_0.013<br>0.15_±_0.012<br>0.09_±_0.007<br>0.06_±_0.004<br>FlexTENet<br>0.10_±_0.015<br>0.09_±_0.016<br>0.06_±_0.008<br>0.04_±_0.006<br>0.04_±_0.006<br>0.07_±_0.009<br>0.12_±_0.011<br>0.17_±_0.017<br>0.12_±_0.018<br>0.11_±_0.019<br>0.08_±_0.010<br>0.06_±_0.007<br>GraphITE<br>0.19_±_0.024<br>0.12_±_0.013<br>0.05_±_0.004<br>0.03_±_0.002<br>0.07_±_0.009<br>0.08_±_0.010<br>0.09_±_0.008<br>0.10_±_0.008<br>0.23_±_0.027<br>0.14_±_0.015<br>0.07_±_0.005<br>0.04_±_0.003<br>SIN<br>0.00_±_0.002<br>0.00_±_0.001<br>0.00_±_0.001<br>0.00_±_0.001<br>0.00_±_0.001<br>0.00_±_0.001<br>0.01_±_0.001<br>0.02_±_0.002<br>0.01_±_0.002<br>0.01_±_0.001<br>0.01_±_0.001<br>0.01_±_0.001<br>S-learner w/ meta-learning<br>0.21_±_0.032<br>0.16_±_0.028<br>0.09_±_0.020<br>0.05_±_0.012<br>0.08_±_0.013<br>0.11_±_0.022<br>0.15_±_0.035<br>0.16_±_0.038<br>0.25_±_0.034<br>0.18_±_0.031<br>0.10_±_0.023<br>0.06_±_0.014<br>T-learner w/ meta-learning<br>0.40_±_0.012<br>0.31_±_0.010<br>0.18_±_0.007<br>0.11_±_0.004<br>0.15_±_0.006<br>0.22_±_0.008<br>0.32_±_0.013<br>0.38_±_0.014<br>0.45_±_0.013<br>0.35_±_0.011<br>0.21_±_0.008<br>0.13_±_0.004<br>CaML - w/o meta-learning<br>0.39_±_0.012<br>0.31_±_0.006<br>0.18_±_0.008<br>0.11_±_0.006<br>0.15_±_0.005<br>0.22_±_0.007<br>0.32_±_0.014<br>0.39_±_0.021<br>0.45_±_0.010<br>0.35_±_0.006<br>0.22_±_0.008<br>0.14_±_0.006<br>CaML - w/o RA-learner<br>0.45_±_0.058<br>0.36_±_0.066<br>0.22_±_0.067<br>**0.14**_±_0.041<br>0.16_±_0.020<br>0.24_±_0.019<br>0.35_±_0.016<br>0.41_±_0.023<br>0.51_±_0.076<br>0.41_±_0.082<br>0.26_±_0.078<br>**0.16**_±_0.048<br>CaML (ours)<br>**0.48**_±_0.010<br>**0.38**_±_0.007<br>**0.23**_±_0.003<br>0.13_±_0.002<br>**0.18**_±_0.004<br>**0.27**_±_0.005<br>**0.38**_±_0.006<br>**0.45**_±_0.010<br>**0.54**_±_0.012<br>**0.43**_±_0.008<br>**0.26**_±_0.004<br>0.16_±_0.003<br>Table 4: Performance results for the Claims dataset (predicting pancytopenia onset from drug exposure using patient medical history. This table extends Table 2 with<br>standard deviations.|RATE @_u_ (_↑_)<br>Recall @_u_ (_↑_)<br>Precision @_u_ (_↑_)<br>0_._999<br>_._998<br>0_._995<br>0_._99<br>0_._999<br>0_._998<br>0_._995<br>0_._99<br>0_._999<br>0_._998<br>0_._995<br>0_._99<br>Random<br>0.00_±<_0.001 0.00_±<_0.001 0.00_±<_0.001<br>0.00_±<_0.001<br>0.00_±<_0.001 0.00_±<_0.001 0.01_±<_0.001 0.00_±<_0.001<br>0.00_±<_0.001 0.00_±<_0.001 0.00_±<_0.001<br>0.00_±<_0.001<br>T-learner<br>0.32_±<_0.001<br>0.26_±<_0.001<br>0.16_±<_0.001<br>0.10_±<_0.001<br>0.12_±<_0.001<br>0.18_±<_0.001<br>0.26_±<_0.001<br>0.31_±<_0.001<br>0.36_±<_0.001<br>0.29_±<_0.001<br>0.18_±<_0.001<br>0.11_±<_0.001<br>X-learner<br>0.06_±<_0.001<br>0.05_±<_0.001<br>0.04_±<_0.001<br>0.03_±<_0.001<br>0.02_±<_0.001<br>0.04_±<_0.001<br>0.08_±<_0.001<br>0.12_±<_0.001<br>0.09_±<_0.001<br>0.07_±<_0.001<br>0.06_±<_0.001<br>0.05_±<_0.001<br>R-learner<br>0.19_±<_0.001<br>0.17_±<_0.001<br>0.12_±<_0.001<br>0.08_±<_0.001<br>0.06_±<_0.001<br>0.10_±<_0.001<br>0.19_±<_0.001<br>0.26_±<_0.001<br>0.24_±<_0.001<br>0.21_±<_0.001<br>0.15_±<_0.001<br>0.11_±<_0.001<br>RA-learner<br>0.47<br>_±_0.001<br>0.37<br>_±<_0.001<br>0.23<br>_±<_0.001<br>0.14<br>_±<_0.001<br>0.17<br>_±<_0.001<br>0.26<br>_±<_0.001<br>0.38<br>_±<_0.001<br>0.45<br>_±<_0.001<br>0.54<br>_±_0.001<br>0.42<br>_±<_0.001<br>0.26<br>_±<_0.001<br>0.16<br>_±<_0.001<br>DragonNet<br>0.09_±_0.037<br>0.07_±_0.030<br>0.05_±_0.019<br>0.04_±_0.013<br>0.02_±_0.008<br>0.04_±_0.012<br>0.07_±_0.020<br>0.10_±_0.027<br>0.12_±_0.045<br>0.10_±_0.036<br>0.07_±_0.023<br>0.05_±_0.015<br>TARNet<br>0.15_±_0.011<br>0.12_±_0.011<br>0.07_±_0.006<br>0.05_±_0.004<br>0.05_±_0.003<br>0.08_±_0.006<br>0.12_±_0.008<br>0.14_±_0.011<br>0.18_±_0.013<br>0.15_±_0.012<br>0.09_±_0.007<br>0.06_±_0.004<br>FlexTENet<br>0.10_±_0.015<br>0.09_±_0.016<br>0.06_±_0.008<br>0.04_±_0.006<br>0.04_±_0.006<br>0.07_±_0.009<br>0.12_±_0.011<br>0.17_±_0.017<br>0.12_±_0.018<br>0.11_±_0.019<br>0.08_±_0.010<br>0.06_±_0.007<br>GraphITE<br>0.19_±_0.024<br>0.12_±_0.013<br>0.05_±_0.004<br>0.03_±_0.002<br>0.07_±_0.009<br>0.08_±_0.010<br>0.09_±_0.008<br>0.10_±_0.008<br>0.23_±_0.027<br>0.14_±_0.015<br>0.07_±_0.005<br>0.04_±_0.003<br>SIN<br>0.00_±_0.002<br>0.00_±_0.001<br>0.00_±_0.001<br>0.00_±_0.001<br>0.00_±_0.001<br>0.00_±_0.001<br>0.01_±_0.001<br>0.02_±_0.002<br>0.01_±_0.002<br>0.01_±_0.001<br>0.01_±_0.001<br>0.01_±_0.001<br>S-learner w/ meta-learning<br>0.21_±_0.032<br>0.16_±_0.028<br>0.09_±_0.020<br>0.05_±_0.012<br>0.08_±_0.013<br>0.11_±_0.022<br>0.15_±_0.035<br>0.16_±_0.038<br>0.25_±_0.034<br>0.18_±_0.031<br>0.10_±_0.023<br>0.06_±_0.014<br>T-learner w/ meta-learning<br>0.40_±_0.012<br>0.31_±_0.010<br>0.18_±_0.007<br>0.11_±_0.004<br>0.15_±_0.006<br>0.22_±_0.008<br>0.32_±_0.013<br>0.38_±_0.014<br>0.45_±_0.013<br>0.35_±_0.011<br>0.21_±_0.008<br>0.13_±_0.004<br>CaML - w/o meta-learning<br>0.39_±_0.012<br>0.31_±_0.006<br>0.18_±_0.008<br>0.11_±_0.006<br>0.15_±_0.005<br>0.22_±_0.007<br>0.32_±_0.014<br>0.39_±_0.021<br>0.45_±_0.010<br>0.35_±_0.006<br>0.22_±_0.008<br>0.14_±_0.006<br>CaML - w/o RA-learner<br>0.45_±_0.058<br>0.36_±_0.066<br>0.22_±_0.067<br>**0.14**_±_0.041<br>0.16_±_0.020<br>0.24_±_0.019<br>0.35_±_0.016<br>0.41_±_0.023<br>0.51_±_0.076<br>0.41_±_0.082<br>0.26_±_0.078<br>**0.16**_±_0.048<br>CaML (ours)<br>**0.48**_±_0.010<br>**0.38**_±_0.007<br>**0.23**_±_0.003<br>0.13_±_0.002<br>**0.18**_±_0.004<br>**0.27**_±_0.005<br>**0.38**_±_0.006<br>**0.45**_±_0.010<br>**0.54**_±_0.012<br>**0.43**_±_0.008<br>**0.26**_±_0.004<br>0.16_±_0.003<br>Table 4: Performance results for the Claims dataset (predicting pancytopenia onset from drug exposure using patient medical history. This table extends Table 2 with<br>standard deviations.|RATE @_u_ (_↑_)<br>Recall @_u_ (_↑_)<br>Precision @_u_ (_↑_)<br>0_._999<br>_._998<br>0_._995<br>0_._99<br>0_._999<br>0_._998<br>0_._995<br>0_._99<br>0_._999<br>0_._998<br>0_._995<br>0_._99<br>Random<br>0.00_±<_0.001 0.00_±<_0.001 0.00_±<_0.001<br>0.00_±<_0.001<br>0.00_±<_0.001 0.00_±<_0.001 0.01_±<_0.001 0.00_±<_0.001<br>0.00_±<_0.001 0.00_±<_0.001 0.00_±<_0.001<br>0.00_±<_0.001<br>T-learner<br>0.32_±<_0.001<br>0.26_±<_0.001<br>0.16_±<_0.001<br>0.10_±<_0.001<br>0.12_±<_0.001<br>0.18_±<_0.001<br>0.26_±<_0.001<br>0.31_±<_0.001<br>0.36_±<_0.001<br>0.29_±<_0.001<br>0.18_±<_0.001<br>0.11_±<_0.001<br>X-learner<br>0.06_±<_0.001<br>0.05_±<_0.001<br>0.04_±<_0.001<br>0.03_±<_0.001<br>0.02_±<_0.001<br>0.04_±<_0.001<br>0.08_±<_0.001<br>0.12_±<_0.001<br>0.09_±<_0.001<br>0.07_±<_0.001<br>0.06_±<_0.001<br>0.05_±<_0.001<br>R-learner<br>0.19_±<_0.001<br>0.17_±<_0.001<br>0.12_±<_0.001<br>0.08_±<_0.001<br>0.06_±<_0.001<br>0.10_±<_0.001<br>0.19_±<_0.001<br>0.26_±<_0.001<br>0.24_±<_0.001<br>0.21_±<_0.001<br>0.15_±<_0.001<br>0.11_±<_0.001<br>RA-learner<br>0.47<br>_±_0.001<br>0.37<br>_±<_0.001<br>0.23<br>_±<_0.001<br>0.14<br>_±<_0.001<br>0.17<br>_±<_0.001<br>0.26<br>_±<_0.001<br>0.38<br>_±<_0.001<br>0.45<br>_±<_0.001<br>0.54<br>_±_0.001<br>0.42<br>_±<_0.001<br>0.26<br>_±<_0.001<br>0.16<br>_±<_0.001<br>DragonNet<br>0.09_±_0.037<br>0.07_±_0.030<br>0.05_±_0.019<br>0.04_±_0.013<br>0.02_±_0.008<br>0.04_±_0.012<br>0.07_±_0.020<br>0.10_±_0.027<br>0.12_±_0.045<br>0.10_±_0.036<br>0.07_±_0.023<br>0.05_±_0.015<br>TARNet<br>0.15_±_0.011<br>0.12_±_0.011<br>0.07_±_0.006<br>0.05_±_0.004<br>0.05_±_0.003<br>0.08_±_0.006<br>0.12_±_0.008<br>0.14_±_0.011<br>0.18_±_0.013<br>0.15_±_0.012<br>0.09_±_0.007<br>0.06_±_0.004<br>FlexTENet<br>0.10_±_0.015<br>0.09_±_0.016<br>0.06_±_0.008<br>0.04_±_0.006<br>0.04_±_0.006<br>0.07_±_0.009<br>0.12_±_0.011<br>0.17_±_0.017<br>0.12_±_0.018<br>0.11_±_0.019<br>0.08_±_0.010<br>0.06_±_0.007<br>GraphITE<br>0.19_±_0.024<br>0.12_±_0.013<br>0.05_±_0.004<br>0.03_±_0.002<br>0.07_±_0.009<br>0.08_±_0.010<br>0.09_±_0.008<br>0.10_±_0.008<br>0.23_±_0.027<br>0.14_±_0.015<br>0.07_±_0.005<br>0.04_±_0.003<br>SIN<br>0.00_±_0.002<br>0.00_±_0.001<br>0.00_±_0.001<br>0.00_±_0.001<br>0.00_±_0.001<br>0.00_±_0.001<br>0.01_±_0.001<br>0.02_±_0.002<br>0.01_±_0.002<br>0.01_±_0.001<br>0.01_±_0.001<br>0.01_±_0.001<br>S-learner w/ meta-learning<br>0.21_±_0.032<br>0.16_±_0.028<br>0.09_±_0.020<br>0.05_±_0.012<br>0.08_±_0.013<br>0.11_±_0.022<br>0.15_±_0.035<br>0.16_±_0.038<br>0.25_±_0.034<br>0.18_±_0.031<br>0.10_±_0.023<br>0.06_±_0.014<br>T-learner w/ meta-learning<br>0.40_±_0.012<br>0.31_±_0.010<br>0.18_±_0.007<br>0.11_±_0.004<br>0.15_±_0.006<br>0.22_±_0.008<br>0.32_±_0.013<br>0.38_±_0.014<br>0.45_±_0.013<br>0.35_±_0.011<br>0.21_±_0.008<br>0.13_±_0.004<br>CaML - w/o meta-learning<br>0.39_±_0.012<br>0.31_±_0.006<br>0.18_±_0.008<br>0.11_±_0.006<br>0.15_±_0.005<br>0.22_±_0.007<br>0.32_±_0.014<br>0.39_±_0.021<br>0.45_±_0.010<br>0.35_±_0.006<br>0.22_±_0.008<br>0.14_±_0.006<br>CaML - w/o RA-learner<br>0.45_±_0.058<br>0.36_±_0.066<br>0.22_±_0.067<br>**0.14**_±_0.041<br>0.16_±_0.020<br>0.24_±_0.019<br>0.35_±_0.016<br>0.41_±_0.023<br>0.51_±_0.076<br>0.41_±_0.082<br>0.26_±_0.078<br>**0.16**_±_0.048<br>CaML (ours)<br>**0.48**_±_0.010<br>**0.38**_±_0.007<br>**0.23**_±_0.003<br>0.13_±_0.002<br>**0.18**_±_0.004<br>**0.27**_±_0.005<br>**0.38**_±_0.006<br>**0.45**_±_0.010<br>**0.54**_±_0.012<br>**0.43**_±_0.008<br>**0.26**_±_0.004<br>0.16_±_0.003<br>Table 4: Performance results for the Claims dataset (predicting pancytopenia onset from drug exposure using patient medical history. This table extends Table 2 with<br>standard deviations.|
|---|---|---|---|---|
|Precision @_u_ (_↑_)<br>0_._999<br>0_._998<br>0_._995<br>0_._99|0.00_±<_0.001 0.00_±<_0.001 0.00_±<_0.001<br>0.00_±<_0.001|0.11_±<_0.001<br>0.05_±<_0.001<br>0.11_±<_0.001<br>0.16<br>_±<_0.001<br>0.05_±_0.015<br>0.06_±_0.004<br>0.06_±_0.007<br>0.04_±_0.003<br>0.01_±_0.001<br>0.06_±_0.014<br>0.13_±_0.004||0.14_±_0.006<br>**0.16**_±_0.048<br>0.16_±_0.003|
|||.18_±<_0.001<br>.06_±<_0.001<br>.15_±<_0.001<br>.26<br>_±<_0.001<br>0.07_±_0.023<br>0.09_±_0.007<br>0.08_±_0.010<br>0.07_±_0.005<br>0.01_±_0.001<br>0.10_±_0.023<br>0.21_±_0.008||0.22_±_0.008<br>0.26_±_0.078<br>**0.26**_±_0.004|
|||0<br>0<br>0<br>0|||
|||.29_±<_0.001<br>.07_±<_0.001<br>.21_±<_0.001<br>.42<br>_±<_0.001<br>0.10_±_0.036<br>0.15_±_0.012<br>0.11_±_0.019<br>0.14_±_0.015<br>0.01_±_0.001<br>0.18_±_0.031<br>0.35_±_0.011||0.35_±_0.006<br>0.41_±_0.082<br>**0.43**_±_0.008|
|||0<br>0<br>0<br>0|||
|||0.36_±<_0.001<br>0.09_±<_0.001<br>0.24_±<_0.001<br>0.54<br>_±_0.001<br>0.12_±_0.045<br>0.18_±_0.013<br>0.12_±_0.018<br>0.23_±_0.027<br>0.01_±_0.002<br>0.25_±_0.034<br>0.45_±_0.013||0.45_±_0.010<br>0.51_±_0.076<br>**0.54**_±_0.012|
|Recall @_u_ (_↑_)<br>0_._999<br>0_._998<br>0_._995<br>0_._99|0.00_±<_0.001 0.00_±<_0.001 0.01_±<_0.001 0.00_±<_0.001|.31_±<_0.001<br>.12_±<_0.001<br>.26_±<_0.001<br>.45<br>_±<_0.001<br>0.10_±_0.027<br>0.14_±_0.011<br>0.17_±_0.017<br>0.10_±_0.008<br>0.02_±_0.002<br>0.16_±_0.038<br>0.38_±_0.014||0.39_±_0.021<br>0.41_±_0.023<br>**0.45**_±_0.010|
|||0<br>0<br>0<br>0|||
|||.26_±<_0.001<br>.08_±<_0.001<br>.19_±<_0.001<br>.38<br>_±<_0.001<br>0.07_±_0.020<br>0.12_±_0.008<br>0.12_±_0.011<br>0.09_±_0.008<br>0.01_±_0.001<br>0.15_±_0.035<br>0.32_±_0.013||0.32_±_0.014<br>0.35_±_0.016<br>**0.38**_±_0.006|
|||0<br>0<br>0<br>0|||
|||.18_±<_0.001<br>.04_±<_0.001<br>.10_±<_0.001<br>.26<br>_±<_0.001<br>0.04_±_0.012<br>0.08_±_0.006<br>0.07_±_0.009<br>0.08_±_0.010<br>0.00_±_0.001<br>0.11_±_0.022<br>0.22_±_0.008||0.22_±_0.007<br>0.24_±_0.019<br>**0.27**_±_0.005|
|||0<br>0<br>0<br>0|||
|||0.12_±<_0.001<br>0.02_±<_0.001<br>0.06_±<_0.001<br>0.17<br>_±<_0.001<br>0.02_±_0.008<br>0.05_±_0.003<br>0.04_±_0.006<br>0.07_±_0.009<br>0.00_±_0.001<br>0.08_±_0.013<br>0.15_±_0.006||0.15_±_0.005<br>0.16_±_0.020<br>**0.18**_±_0.004|
|RATE @_u_ (_↑_)<br>0_._999<br>_._998<br>0_._995<br>0_._99|0.00_±<_0.001 0.00_±<_0.001 0.00_±<_0.001<br>0.00_±<_0.001|0.10_±<_0.001<br>0.03_±<_0.001<br>0.08_±<_0.001<br>0.14<br>_±<_0.001<br>0.04_±_0.013<br>0.05_±_0.004<br>0.04_±_0.006<br>0.03_±_0.002<br>0.00_±_0.001<br>0.05_±_0.012<br>0.11_±_0.004||0.11_±_0.006<br>**0.14**_±_0.041<br>0.13_±_0.002|
|||.16_±<_0.001<br>.04_±<_0.001<br>.12_±<_0.001<br>.23<br>_±<_0.001<br>0.05_±_0.019<br>0.07_±_0.006<br>0.06_±_0.008<br>0.05_±_0.004<br>0.00_±_0.001<br>0.09_±_0.020<br>0.18_±_0.007||0.18_±_0.008<br>0.22_±_0.067<br>**0.23**_±_0.003|
|||0<br>0<br>0<br>0|||
|||.26_±<_0.001<br>.05_±<_0.001<br>.17_±<_0.001<br>.37<br>_±<_0.001<br>0.07_±_0.030<br>0.12_±_0.011<br>0.09_±_0.016<br>0.12_±_0.013<br>0.00_±_0.001<br>0.16_±_0.028<br>0.31_±_0.010||0.31_±_0.006<br>0.36_±_0.066<br>**0.38**_±_0.007|
|||0<br>0<br>0<br>0|||
|||0.32_±<_0.001<br>0.06_±<_0.001<br>0.19_±<_0.001<br>0.47<br>_±_0.001<br>0.09_±_0.037<br>0.15_±_0.011<br>0.10_±_0.015<br>0.19_±_0.024<br>0.00_±_0.002<br>0.21_±_0.032<br>0.40_±_0.012||0.39_±_0.012<br>0.45_±_0.058<br>**0.48**_±_0.010|
||Random|T-learner<br>X-learner<br>R-learner<br>RA-learner<br>DragonNet<br>TARNet<br>FlexTENet<br>GraphITE<br>SIN<br>S-learner w/ meta-learning<br>T-learner w/ meta-learning||CaML - w/o meta-learning<br>CaML - w/o RA-learner<br>CaML (ours)|



29 

|RATE @_u_ (_↑_)<br>Recall @_u_ (_↑_)<br>Precision @_u_(_↑_)<br>0_._999<br>_._998<br>0_._995<br>0_._99<br>0_._999<br>0_._998<br>0_._995<br>0_._99<br>0_._999<br>0_._998<br>0_._995<br>0_._99<br>Random<br>0.00_±<_0.001<br>0.00_±<_0.001<br>0.00_±<_0.001<br>0.00_±<_0.001<br>0.0_±<_0.001<br>0.0_±<_0.001<br>0.01_±<_0.001 0.01_±<_0.001<br>0.01_±<_0.0014 0.01_±<_0.001 0.01_±<_0.001<br>0.00_±<_0.001<br>T-learner<br>0.10_±<_0.001<br>0.07_±<_0.001<br>0.05_±<_0.001<br>0.04_±<_0.001<br>0.05_±<_0.001<br>0.07_±<_0.001<br>0.11_±<_0.001<br>0.13_±<_0.001<br>0.10_±<_0.001<br>0.08_±<_0.001<br>0.06_±<_0.001<br>0.04_±<_0.001<br>X-learner<br>0.00_±<_0.001<br>-0.01_±<_0.001<br>0.00_±<_0.001<br>0.00_±<_0.001<br>0.00_±<_0.001<br>0.00_±<_0.001<br>0.01_±<_0.001<br>0.02_±<_0.001<br>0.00_±<_0.001<br>0.00_±<_0.001<br>0.00_±<_0.001<br>0.01_±<_0.001<br>R-learner<br>-0.01_±<_0.001<br>-0.01_±<_0.001<br>-0.01_±<_0.001<br>0.00_±<_0.001<br>0.00_±<_0.001<br>0.00_±<_0.001<br>0.00_±<_0.001<br>0.04_±<_0.001<br>0.00_±<_0.001<br>0.00_±<_0.001<br>0.00_±<_0.001<br>0.01_±<_0.001<br>RA-learner<br>0.28<br>_±<_0.001<br>0.26<br>_±<_0.001<br>0.17<br>_±<_0.001<br>0.10<br>_±<_0.001<br>0.10<br>_±<_0.001<br>0.19<br>_±<_0.001<br>0.30<br>_±<_0.001<br>0.37<br>_±<_0.001<br>0.30<br>_±<_0.001<br>0.28<br>_±<_0.001<br>0.18<br>_±<_0.001<br>0.11<br>_±<_0.001<br>DragonNet<br>-0.01_±_0.002<br>0.00_±_0.009<br>0.00_±_0.004<br>0.00_±_0.003<br>0.00_±<_0.001<br>0.00_±_0.003<br>0.00_±_0.005<br>0.01_±_0.009<br>0.00_±<_0.001<br>0.00_±_0.010<br>0.00_±_0.004<br>0.00_±_0.003<br>TARNet<br>0.04_±_0.046<br>0.03_±_0.030<br>0.02_±_0.013<br>0.02_±_0.012<br>0.01_±_0.011<br>0.02_±_0.015<br>0.04_±_0.013<br>0.06_±_0.029<br>0.05_±_0.046<br>0.04_±_0.032<br>0.03_±_0.013<br>0.02_±_0.012<br>FlexTENet<br>0.02_±_0.024<br>0.02_±_0.019<br>0.04_±_0.012<br>0.03_±_0.013<br>0.01_±_0.009<br>0.03_±_0.018<br>0.08_±_0.012<br>0.12_±_0.037<br>0.02_±_0.027<br>0.03_±_0.020<br>0.04_±_0.012<br>0.04_±_0.014<br>S-learner w/ meta-learning<br>0.27_±_0.173<br>0.16_±_0.118<br>0.08_±_0.052<br>0.04_±_0.030<br>0.09_±_0.055<br>0.10_±_0.070<br>0.13_±_0.084<br>0.15_±_0.090<br>0.29_±_0.180<br>0.18_±_0.123<br>0.09_±_0.055<br>0.05_±_0.032<br>T-learner w/ meta-learning<br>0.27_±_0.173<br>0.16_±_0.118<br>0.08_±_0.052<br>0.04_±_0.030<br>0.09_±_0.055<br>0.10_±_0.070<br>0.13_±_0.084<br>0.15_±_0.090<br>0.29_±_0.180<br>0.18_±_0.123<br>0.09_±_0.055<br>0.05_±_0.032<br>GraphITE<br>0.25_±_0.088<br>0.15_±_0.054<br>0.06_±_0.025<br>0.03_±_0.011<br>0.08_±_0.024<br>0.10_±_0.034<br>0.11_±_0.045<br>0.13_±_0.049<br>0.27_±_0.091<br>0.16_±_0.057<br>0.07_±_0.027<br>0.04_±_0.013<br>SIN<br>0.00_±_0.008<br>0.00_±_0.014<br>0.00_±_0.008<br>0.00_±_0.005<br>0.00_±_0.005<br>0.00_±_0.008<br>0.02_±_0.015<br>0.03_±_0.009<br>0.00_±_0.007<br>0.01_±_0.014<br>0.01_±_0.009<br>0.01_±_0.005<br>CaML - w/o meta-learning<br>0.45_±_0.070<br>**0.38**_±_0.057<br>0.21_±_0.017<br>0.13_±_0.008<br>0.19_±_0.019<br>0.28_±_0.026<br>0.38_±_0.025<br>0.45_±_0.019<br>0.49_±_0.070<br>**0.41**_±_0.057<br>0.23_±_0.017<br>0.15_±_0.008<br>CaML - w/o RA-learner<br>0.40_±_0.101<br>0.33_±_0.034<br>**0.24**_±_0.014<br>0.15_±_0.010<br>0.18_±_0.025<br>0.28_±_0.010<br>0.42_±_0.024<br>0.50_±_0.028<br>0.44_±_0.099<br>0.36_±_0.033<br>**0.26**_±_0.014<br>0.17_±_0.010<br>CaML (ours)<br>**0.47**_±_0.084<br>0.37_±_0.044<br>0.23_±_0.022<br>**0.15**_±_0.013<br>**0.20**_±_0.015<br>**0.30**_±_0.016<br>**0.43**_±_0.024<br>**0.51**_±_0.027<br>**0.51**_±_0.079<br>0.40_±_0.044<br>0.25_±_0.023<br>**0.17**_±_0.013<br>Table 5: Performance results for the medical claims dataset, in which the task is to predict the effect of a_pair_of drugs the drug on pancytopenia occurrence. Mean<br>and standard deviation between runs is reported. Single-task methods were trained on the meta-testing tasks (best model underlined). Methods that were capable of<br>training across multiple tasks were trained on meta-training tasks and applied to previously unseen meta-testing tasks (best model in bold).CaML outperforms the<br>strongest baseline that had access to testing tasks on 12 out of 12 metrics, and outperforms all zero-shot baselines. Notably, due to the small sample size for natural<br>experiments with combinations of drugs,_the RATE estimation process is very noisy_which is refected in high variability of the measured RATE. Here, the secondary<br>metrics (Recall and Precision) that are not affected, additionally assert the dominance of CaML over all baselines.|RATE @_u_ (_↑_)<br>Recall @_u_ (_↑_)<br>Precision @_u_(_↑_)<br>0_._999<br>_._998<br>0_._995<br>0_._99<br>0_._999<br>0_._998<br>0_._995<br>0_._99<br>0_._999<br>0_._998<br>0_._995<br>0_._99<br>Random<br>0.00_±<_0.001<br>0.00_±<_0.001<br>0.00_±<_0.001<br>0.00_±<_0.001<br>0.0_±<_0.001<br>0.0_±<_0.001<br>0.01_±<_0.001 0.01_±<_0.001<br>0.01_±<_0.0014 0.01_±<_0.001 0.01_±<_0.001<br>0.00_±<_0.001<br>T-learner<br>0.10_±<_0.001<br>0.07_±<_0.001<br>0.05_±<_0.001<br>0.04_±<_0.001<br>0.05_±<_0.001<br>0.07_±<_0.001<br>0.11_±<_0.001<br>0.13_±<_0.001<br>0.10_±<_0.001<br>0.08_±<_0.001<br>0.06_±<_0.001<br>0.04_±<_0.001<br>X-learner<br>0.00_±<_0.001<br>-0.01_±<_0.001<br>0.00_±<_0.001<br>0.00_±<_0.001<br>0.00_±<_0.001<br>0.00_±<_0.001<br>0.01_±<_0.001<br>0.02_±<_0.001<br>0.00_±<_0.001<br>0.00_±<_0.001<br>0.00_±<_0.001<br>0.01_±<_0.001<br>R-learner<br>-0.01_±<_0.001<br>-0.01_±<_0.001<br>-0.01_±<_0.001<br>0.00_±<_0.001<br>0.00_±<_0.001<br>0.00_±<_0.001<br>0.00_±<_0.001<br>0.04_±<_0.001<br>0.00_±<_0.001<br>0.00_±<_0.001<br>0.00_±<_0.001<br>0.01_±<_0.001<br>RA-learner<br>0.28<br>_±<_0.001<br>0.26<br>_±<_0.001<br>0.17<br>_±<_0.001<br>0.10<br>_±<_0.001<br>0.10<br>_±<_0.001<br>0.19<br>_±<_0.001<br>0.30<br>_±<_0.001<br>0.37<br>_±<_0.001<br>0.30<br>_±<_0.001<br>0.28<br>_±<_0.001<br>0.18<br>_±<_0.001<br>0.11<br>_±<_0.001<br>DragonNet<br>-0.01_±_0.002<br>0.00_±_0.009<br>0.00_±_0.004<br>0.00_±_0.003<br>0.00_±<_0.001<br>0.00_±_0.003<br>0.00_±_0.005<br>0.01_±_0.009<br>0.00_±<_0.001<br>0.00_±_0.010<br>0.00_±_0.004<br>0.00_±_0.003<br>TARNet<br>0.04_±_0.046<br>0.03_±_0.030<br>0.02_±_0.013<br>0.02_±_0.012<br>0.01_±_0.011<br>0.02_±_0.015<br>0.04_±_0.013<br>0.06_±_0.029<br>0.05_±_0.046<br>0.04_±_0.032<br>0.03_±_0.013<br>0.02_±_0.012<br>FlexTENet<br>0.02_±_0.024<br>0.02_±_0.019<br>0.04_±_0.012<br>0.03_±_0.013<br>0.01_±_0.009<br>0.03_±_0.018<br>0.08_±_0.012<br>0.12_±_0.037<br>0.02_±_0.027<br>0.03_±_0.020<br>0.04_±_0.012<br>0.04_±_0.014<br>S-learner w/ meta-learning<br>0.27_±_0.173<br>0.16_±_0.118<br>0.08_±_0.052<br>0.04_±_0.030<br>0.09_±_0.055<br>0.10_±_0.070<br>0.13_±_0.084<br>0.15_±_0.090<br>0.29_±_0.180<br>0.18_±_0.123<br>0.09_±_0.055<br>0.05_±_0.032<br>T-learner w/ meta-learning<br>0.27_±_0.173<br>0.16_±_0.118<br>0.08_±_0.052<br>0.04_±_0.030<br>0.09_±_0.055<br>0.10_±_0.070<br>0.13_±_0.084<br>0.15_±_0.090<br>0.29_±_0.180<br>0.18_±_0.123<br>0.09_±_0.055<br>0.05_±_0.032<br>GraphITE<br>0.25_±_0.088<br>0.15_±_0.054<br>0.06_±_0.025<br>0.03_±_0.011<br>0.08_±_0.024<br>0.10_±_0.034<br>0.11_±_0.045<br>0.13_±_0.049<br>0.27_±_0.091<br>0.16_±_0.057<br>0.07_±_0.027<br>0.04_±_0.013<br>SIN<br>0.00_±_0.008<br>0.00_±_0.014<br>0.00_±_0.008<br>0.00_±_0.005<br>0.00_±_0.005<br>0.00_±_0.008<br>0.02_±_0.015<br>0.03_±_0.009<br>0.00_±_0.007<br>0.01_±_0.014<br>0.01_±_0.009<br>0.01_±_0.005<br>CaML - w/o meta-learning<br>0.45_±_0.070<br>**0.38**_±_0.057<br>0.21_±_0.017<br>0.13_±_0.008<br>0.19_±_0.019<br>0.28_±_0.026<br>0.38_±_0.025<br>0.45_±_0.019<br>0.49_±_0.070<br>**0.41**_±_0.057<br>0.23_±_0.017<br>0.15_±_0.008<br>CaML - w/o RA-learner<br>0.40_±_0.101<br>0.33_±_0.034<br>**0.24**_±_0.014<br>0.15_±_0.010<br>0.18_±_0.025<br>0.28_±_0.010<br>0.42_±_0.024<br>0.50_±_0.028<br>0.44_±_0.099<br>0.36_±_0.033<br>**0.26**_±_0.014<br>0.17_±_0.010<br>CaML (ours)<br>**0.47**_±_0.084<br>0.37_±_0.044<br>0.23_±_0.022<br>**0.15**_±_0.013<br>**0.20**_±_0.015<br>**0.30**_±_0.016<br>**0.43**_±_0.024<br>**0.51**_±_0.027<br>**0.51**_±_0.079<br>0.40_±_0.044<br>0.25_±_0.023<br>**0.17**_±_0.013<br>Table 5: Performance results for the medical claims dataset, in which the task is to predict the effect of a_pair_of drugs the drug on pancytopenia occurrence. Mean<br>and standard deviation between runs is reported. Single-task methods were trained on the meta-testing tasks (best model underlined). Methods that were capable of<br>training across multiple tasks were trained on meta-training tasks and applied to previously unseen meta-testing tasks (best model in bold).CaML outperforms the<br>strongest baseline that had access to testing tasks on 12 out of 12 metrics, and outperforms all zero-shot baselines. Notably, due to the small sample size for natural<br>experiments with combinations of drugs,_the RATE estimation process is very noisy_which is refected in high variability of the measured RATE. Here, the secondary<br>metrics (Recall and Precision) that are not affected, additionally assert the dominance of CaML over all baselines.|RATE @_u_ (_↑_)<br>Recall @_u_ (_↑_)<br>Precision @_u_(_↑_)<br>0_._999<br>_._998<br>0_._995<br>0_._99<br>0_._999<br>0_._998<br>0_._995<br>0_._99<br>0_._999<br>0_._998<br>0_._995<br>0_._99<br>Random<br>0.00_±<_0.001<br>0.00_±<_0.001<br>0.00_±<_0.001<br>0.00_±<_0.001<br>0.0_±<_0.001<br>0.0_±<_0.001<br>0.01_±<_0.001 0.01_±<_0.001<br>0.01_±<_0.0014 0.01_±<_0.001 0.01_±<_0.001<br>0.00_±<_0.001<br>T-learner<br>0.10_±<_0.001<br>0.07_±<_0.001<br>0.05_±<_0.001<br>0.04_±<_0.001<br>0.05_±<_0.001<br>0.07_±<_0.001<br>0.11_±<_0.001<br>0.13_±<_0.001<br>0.10_±<_0.001<br>0.08_±<_0.001<br>0.06_±<_0.001<br>0.04_±<_0.001<br>X-learner<br>0.00_±<_0.001<br>-0.01_±<_0.001<br>0.00_±<_0.001<br>0.00_±<_0.001<br>0.00_±<_0.001<br>0.00_±<_0.001<br>0.01_±<_0.001<br>0.02_±<_0.001<br>0.00_±<_0.001<br>0.00_±<_0.001<br>0.00_±<_0.001<br>0.01_±<_0.001<br>R-learner<br>-0.01_±<_0.001<br>-0.01_±<_0.001<br>-0.01_±<_0.001<br>0.00_±<_0.001<br>0.00_±<_0.001<br>0.00_±<_0.001<br>0.00_±<_0.001<br>0.04_±<_0.001<br>0.00_±<_0.001<br>0.00_±<_0.001<br>0.00_±<_0.001<br>0.01_±<_0.001<br>RA-learner<br>0.28<br>_±<_0.001<br>0.26<br>_±<_0.001<br>0.17<br>_±<_0.001<br>0.10<br>_±<_0.001<br>0.10<br>_±<_0.001<br>0.19<br>_±<_0.001<br>0.30<br>_±<_0.001<br>0.37<br>_±<_0.001<br>0.30<br>_±<_0.001<br>0.28<br>_±<_0.001<br>0.18<br>_±<_0.001<br>0.11<br>_±<_0.001<br>DragonNet<br>-0.01_±_0.002<br>0.00_±_0.009<br>0.00_±_0.004<br>0.00_±_0.003<br>0.00_±<_0.001<br>0.00_±_0.003<br>0.00_±_0.005<br>0.01_±_0.009<br>0.00_±<_0.001<br>0.00_±_0.010<br>0.00_±_0.004<br>0.00_±_0.003<br>TARNet<br>0.04_±_0.046<br>0.03_±_0.030<br>0.02_±_0.013<br>0.02_±_0.012<br>0.01_±_0.011<br>0.02_±_0.015<br>0.04_±_0.013<br>0.06_±_0.029<br>0.05_±_0.046<br>0.04_±_0.032<br>0.03_±_0.013<br>0.02_±_0.012<br>FlexTENet<br>0.02_±_0.024<br>0.02_±_0.019<br>0.04_±_0.012<br>0.03_±_0.013<br>0.01_±_0.009<br>0.03_±_0.018<br>0.08_±_0.012<br>0.12_±_0.037<br>0.02_±_0.027<br>0.03_±_0.020<br>0.04_±_0.012<br>0.04_±_0.014<br>S-learner w/ meta-learning<br>0.27_±_0.173<br>0.16_±_0.118<br>0.08_±_0.052<br>0.04_±_0.030<br>0.09_±_0.055<br>0.10_±_0.070<br>0.13_±_0.084<br>0.15_±_0.090<br>0.29_±_0.180<br>0.18_±_0.123<br>0.09_±_0.055<br>0.05_±_0.032<br>T-learner w/ meta-learning<br>0.27_±_0.173<br>0.16_±_0.118<br>0.08_±_0.052<br>0.04_±_0.030<br>0.09_±_0.055<br>0.10_±_0.070<br>0.13_±_0.084<br>0.15_±_0.090<br>0.29_±_0.180<br>0.18_±_0.123<br>0.09_±_0.055<br>0.05_±_0.032<br>GraphITE<br>0.25_±_0.088<br>0.15_±_0.054<br>0.06_±_0.025<br>0.03_±_0.011<br>0.08_±_0.024<br>0.10_±_0.034<br>0.11_±_0.045<br>0.13_±_0.049<br>0.27_±_0.091<br>0.16_±_0.057<br>0.07_±_0.027<br>0.04_±_0.013<br>SIN<br>0.00_±_0.008<br>0.00_±_0.014<br>0.00_±_0.008<br>0.00_±_0.005<br>0.00_±_0.005<br>0.00_±_0.008<br>0.02_±_0.015<br>0.03_±_0.009<br>0.00_±_0.007<br>0.01_±_0.014<br>0.01_±_0.009<br>0.01_±_0.005<br>CaML - w/o meta-learning<br>0.45_±_0.070<br>**0.38**_±_0.057<br>0.21_±_0.017<br>0.13_±_0.008<br>0.19_±_0.019<br>0.28_±_0.026<br>0.38_±_0.025<br>0.45_±_0.019<br>0.49_±_0.070<br>**0.41**_±_0.057<br>0.23_±_0.017<br>0.15_±_0.008<br>CaML - w/o RA-learner<br>0.40_±_0.101<br>0.33_±_0.034<br>**0.24**_±_0.014<br>0.15_±_0.010<br>0.18_±_0.025<br>0.28_±_0.010<br>0.42_±_0.024<br>0.50_±_0.028<br>0.44_±_0.099<br>0.36_±_0.033<br>**0.26**_±_0.014<br>0.17_±_0.010<br>CaML (ours)<br>**0.47**_±_0.084<br>0.37_±_0.044<br>0.23_±_0.022<br>**0.15**_±_0.013<br>**0.20**_±_0.015<br>**0.30**_±_0.016<br>**0.43**_±_0.024<br>**0.51**_±_0.027<br>**0.51**_±_0.079<br>0.40_±_0.044<br>0.25_±_0.023<br>**0.17**_±_0.013<br>Table 5: Performance results for the medical claims dataset, in which the task is to predict the effect of a_pair_of drugs the drug on pancytopenia occurrence. Mean<br>and standard deviation between runs is reported. Single-task methods were trained on the meta-testing tasks (best model underlined). Methods that were capable of<br>training across multiple tasks were trained on meta-training tasks and applied to previously unseen meta-testing tasks (best model in bold).CaML outperforms the<br>strongest baseline that had access to testing tasks on 12 out of 12 metrics, and outperforms all zero-shot baselines. Notably, due to the small sample size for natural<br>experiments with combinations of drugs,_the RATE estimation process is very noisy_which is refected in high variability of the measured RATE. Here, the secondary<br>metrics (Recall and Precision) that are not affected, additionally assert the dominance of CaML over all baselines.|RATE @_u_ (_↑_)<br>Recall @_u_ (_↑_)<br>Precision @_u_(_↑_)<br>0_._999<br>_._998<br>0_._995<br>0_._99<br>0_._999<br>0_._998<br>0_._995<br>0_._99<br>0_._999<br>0_._998<br>0_._995<br>0_._99<br>Random<br>0.00_±<_0.001<br>0.00_±<_0.001<br>0.00_±<_0.001<br>0.00_±<_0.001<br>0.0_±<_0.001<br>0.0_±<_0.001<br>0.01_±<_0.001 0.01_±<_0.001<br>0.01_±<_0.0014 0.01_±<_0.001 0.01_±<_0.001<br>0.00_±<_0.001<br>T-learner<br>0.10_±<_0.001<br>0.07_±<_0.001<br>0.05_±<_0.001<br>0.04_±<_0.001<br>0.05_±<_0.001<br>0.07_±<_0.001<br>0.11_±<_0.001<br>0.13_±<_0.001<br>0.10_±<_0.001<br>0.08_±<_0.001<br>0.06_±<_0.001<br>0.04_±<_0.001<br>X-learner<br>0.00_±<_0.001<br>-0.01_±<_0.001<br>0.00_±<_0.001<br>0.00_±<_0.001<br>0.00_±<_0.001<br>0.00_±<_0.001<br>0.01_±<_0.001<br>0.02_±<_0.001<br>0.00_±<_0.001<br>0.00_±<_0.001<br>0.00_±<_0.001<br>0.01_±<_0.001<br>R-learner<br>-0.01_±<_0.001<br>-0.01_±<_0.001<br>-0.01_±<_0.001<br>0.00_±<_0.001<br>0.00_±<_0.001<br>0.00_±<_0.001<br>0.00_±<_0.001<br>0.04_±<_0.001<br>0.00_±<_0.001<br>0.00_±<_0.001<br>0.00_±<_0.001<br>0.01_±<_0.001<br>RA-learner<br>0.28<br>_±<_0.001<br>0.26<br>_±<_0.001<br>0.17<br>_±<_0.001<br>0.10<br>_±<_0.001<br>0.10<br>_±<_0.001<br>0.19<br>_±<_0.001<br>0.30<br>_±<_0.001<br>0.37<br>_±<_0.001<br>0.30<br>_±<_0.001<br>0.28<br>_±<_0.001<br>0.18<br>_±<_0.001<br>0.11<br>_±<_0.001<br>DragonNet<br>-0.01_±_0.002<br>0.00_±_0.009<br>0.00_±_0.004<br>0.00_±_0.003<br>0.00_±<_0.001<br>0.00_±_0.003<br>0.00_±_0.005<br>0.01_±_0.009<br>0.00_±<_0.001<br>0.00_±_0.010<br>0.00_±_0.004<br>0.00_±_0.003<br>TARNet<br>0.04_±_0.046<br>0.03_±_0.030<br>0.02_±_0.013<br>0.02_±_0.012<br>0.01_±_0.011<br>0.02_±_0.015<br>0.04_±_0.013<br>0.06_±_0.029<br>0.05_±_0.046<br>0.04_±_0.032<br>0.03_±_0.013<br>0.02_±_0.012<br>FlexTENet<br>0.02_±_0.024<br>0.02_±_0.019<br>0.04_±_0.012<br>0.03_±_0.013<br>0.01_±_0.009<br>0.03_±_0.018<br>0.08_±_0.012<br>0.12_±_0.037<br>0.02_±_0.027<br>0.03_±_0.020<br>0.04_±_0.012<br>0.04_±_0.014<br>S-learner w/ meta-learning<br>0.27_±_0.173<br>0.16_±_0.118<br>0.08_±_0.052<br>0.04_±_0.030<br>0.09_±_0.055<br>0.10_±_0.070<br>0.13_±_0.084<br>0.15_±_0.090<br>0.29_±_0.180<br>0.18_±_0.123<br>0.09_±_0.055<br>0.05_±_0.032<br>T-learner w/ meta-learning<br>0.27_±_0.173<br>0.16_±_0.118<br>0.08_±_0.052<br>0.04_±_0.030<br>0.09_±_0.055<br>0.10_±_0.070<br>0.13_±_0.084<br>0.15_±_0.090<br>0.29_±_0.180<br>0.18_±_0.123<br>0.09_±_0.055<br>0.05_±_0.032<br>GraphITE<br>0.25_±_0.088<br>0.15_±_0.054<br>0.06_±_0.025<br>0.03_±_0.011<br>0.08_±_0.024<br>0.10_±_0.034<br>0.11_±_0.045<br>0.13_±_0.049<br>0.27_±_0.091<br>0.16_±_0.057<br>0.07_±_0.027<br>0.04_±_0.013<br>SIN<br>0.00_±_0.008<br>0.00_±_0.014<br>0.00_±_0.008<br>0.00_±_0.005<br>0.00_±_0.005<br>0.00_±_0.008<br>0.02_±_0.015<br>0.03_±_0.009<br>0.00_±_0.007<br>0.01_±_0.014<br>0.01_±_0.009<br>0.01_±_0.005<br>CaML - w/o meta-learning<br>0.45_±_0.070<br>**0.38**_±_0.057<br>0.21_±_0.017<br>0.13_±_0.008<br>0.19_±_0.019<br>0.28_±_0.026<br>0.38_±_0.025<br>0.45_±_0.019<br>0.49_±_0.070<br>**0.41**_±_0.057<br>0.23_±_0.017<br>0.15_±_0.008<br>CaML - w/o RA-learner<br>0.40_±_0.101<br>0.33_±_0.034<br>**0.24**_±_0.014<br>0.15_±_0.010<br>0.18_±_0.025<br>0.28_±_0.010<br>0.42_±_0.024<br>0.50_±_0.028<br>0.44_±_0.099<br>0.36_±_0.033<br>**0.26**_±_0.014<br>0.17_±_0.010<br>CaML (ours)<br>**0.47**_±_0.084<br>0.37_±_0.044<br>0.23_±_0.022<br>**0.15**_±_0.013<br>**0.20**_±_0.015<br>**0.30**_±_0.016<br>**0.43**_±_0.024<br>**0.51**_±_0.027<br>**0.51**_±_0.079<br>0.40_±_0.044<br>0.25_±_0.023<br>**0.17**_±_0.013<br>Table 5: Performance results for the medical claims dataset, in which the task is to predict the effect of a_pair_of drugs the drug on pancytopenia occurrence. Mean<br>and standard deviation between runs is reported. Single-task methods were trained on the meta-testing tasks (best model underlined). Methods that were capable of<br>training across multiple tasks were trained on meta-training tasks and applied to previously unseen meta-testing tasks (best model in bold).CaML outperforms the<br>strongest baseline that had access to testing tasks on 12 out of 12 metrics, and outperforms all zero-shot baselines. Notably, due to the small sample size for natural<br>experiments with combinations of drugs,_the RATE estimation process is very noisy_which is refected in high variability of the measured RATE. Here, the secondary<br>metrics (Recall and Precision) that are not affected, additionally assert the dominance of CaML over all baselines.|RATE @_u_ (_↑_)<br>Recall @_u_ (_↑_)<br>Precision @_u_(_↑_)<br>0_._999<br>_._998<br>0_._995<br>0_._99<br>0_._999<br>0_._998<br>0_._995<br>0_._99<br>0_._999<br>0_._998<br>0_._995<br>0_._99<br>Random<br>0.00_±<_0.001<br>0.00_±<_0.001<br>0.00_±<_0.001<br>0.00_±<_0.001<br>0.0_±<_0.001<br>0.0_±<_0.001<br>0.01_±<_0.001 0.01_±<_0.001<br>0.01_±<_0.0014 0.01_±<_0.001 0.01_±<_0.001<br>0.00_±<_0.001<br>T-learner<br>0.10_±<_0.001<br>0.07_±<_0.001<br>0.05_±<_0.001<br>0.04_±<_0.001<br>0.05_±<_0.001<br>0.07_±<_0.001<br>0.11_±<_0.001<br>0.13_±<_0.001<br>0.10_±<_0.001<br>0.08_±<_0.001<br>0.06_±<_0.001<br>0.04_±<_0.001<br>X-learner<br>0.00_±<_0.001<br>-0.01_±<_0.001<br>0.00_±<_0.001<br>0.00_±<_0.001<br>0.00_±<_0.001<br>0.00_±<_0.001<br>0.01_±<_0.001<br>0.02_±<_0.001<br>0.00_±<_0.001<br>0.00_±<_0.001<br>0.00_±<_0.001<br>0.01_±<_0.001<br>R-learner<br>-0.01_±<_0.001<br>-0.01_±<_0.001<br>-0.01_±<_0.001<br>0.00_±<_0.001<br>0.00_±<_0.001<br>0.00_±<_0.001<br>0.00_±<_0.001<br>0.04_±<_0.001<br>0.00_±<_0.001<br>0.00_±<_0.001<br>0.00_±<_0.001<br>0.01_±<_0.001<br>RA-learner<br>0.28<br>_±<_0.001<br>0.26<br>_±<_0.001<br>0.17<br>_±<_0.001<br>0.10<br>_±<_0.001<br>0.10<br>_±<_0.001<br>0.19<br>_±<_0.001<br>0.30<br>_±<_0.001<br>0.37<br>_±<_0.001<br>0.30<br>_±<_0.001<br>0.28<br>_±<_0.001<br>0.18<br>_±<_0.001<br>0.11<br>_±<_0.001<br>DragonNet<br>-0.01_±_0.002<br>0.00_±_0.009<br>0.00_±_0.004<br>0.00_±_0.003<br>0.00_±<_0.001<br>0.00_±_0.003<br>0.00_±_0.005<br>0.01_±_0.009<br>0.00_±<_0.001<br>0.00_±_0.010<br>0.00_±_0.004<br>0.00_±_0.003<br>TARNet<br>0.04_±_0.046<br>0.03_±_0.030<br>0.02_±_0.013<br>0.02_±_0.012<br>0.01_±_0.011<br>0.02_±_0.015<br>0.04_±_0.013<br>0.06_±_0.029<br>0.05_±_0.046<br>0.04_±_0.032<br>0.03_±_0.013<br>0.02_±_0.012<br>FlexTENet<br>0.02_±_0.024<br>0.02_±_0.019<br>0.04_±_0.012<br>0.03_±_0.013<br>0.01_±_0.009<br>0.03_±_0.018<br>0.08_±_0.012<br>0.12_±_0.037<br>0.02_±_0.027<br>0.03_±_0.020<br>0.04_±_0.012<br>0.04_±_0.014<br>S-learner w/ meta-learning<br>0.27_±_0.173<br>0.16_±_0.118<br>0.08_±_0.052<br>0.04_±_0.030<br>0.09_±_0.055<br>0.10_±_0.070<br>0.13_±_0.084<br>0.15_±_0.090<br>0.29_±_0.180<br>0.18_±_0.123<br>0.09_±_0.055<br>0.05_±_0.032<br>T-learner w/ meta-learning<br>0.27_±_0.173<br>0.16_±_0.118<br>0.08_±_0.052<br>0.04_±_0.030<br>0.09_±_0.055<br>0.10_±_0.070<br>0.13_±_0.084<br>0.15_±_0.090<br>0.29_±_0.180<br>0.18_±_0.123<br>0.09_±_0.055<br>0.05_±_0.032<br>GraphITE<br>0.25_±_0.088<br>0.15_±_0.054<br>0.06_±_0.025<br>0.03_±_0.011<br>0.08_±_0.024<br>0.10_±_0.034<br>0.11_±_0.045<br>0.13_±_0.049<br>0.27_±_0.091<br>0.16_±_0.057<br>0.07_±_0.027<br>0.04_±_0.013<br>SIN<br>0.00_±_0.008<br>0.00_±_0.014<br>0.00_±_0.008<br>0.00_±_0.005<br>0.00_±_0.005<br>0.00_±_0.008<br>0.02_±_0.015<br>0.03_±_0.009<br>0.00_±_0.007<br>0.01_±_0.014<br>0.01_±_0.009<br>0.01_±_0.005<br>CaML - w/o meta-learning<br>0.45_±_0.070<br>**0.38**_±_0.057<br>0.21_±_0.017<br>0.13_±_0.008<br>0.19_±_0.019<br>0.28_±_0.026<br>0.38_±_0.025<br>0.45_±_0.019<br>0.49_±_0.070<br>**0.41**_±_0.057<br>0.23_±_0.017<br>0.15_±_0.008<br>CaML - w/o RA-learner<br>0.40_±_0.101<br>0.33_±_0.034<br>**0.24**_±_0.014<br>0.15_±_0.010<br>0.18_±_0.025<br>0.28_±_0.010<br>0.42_±_0.024<br>0.50_±_0.028<br>0.44_±_0.099<br>0.36_±_0.033<br>**0.26**_±_0.014<br>0.17_±_0.010<br>CaML (ours)<br>**0.47**_±_0.084<br>0.37_±_0.044<br>0.23_±_0.022<br>**0.15**_±_0.013<br>**0.20**_±_0.015<br>**0.30**_±_0.016<br>**0.43**_±_0.024<br>**0.51**_±_0.027<br>**0.51**_±_0.079<br>0.40_±_0.044<br>0.25_±_0.023<br>**0.17**_±_0.013<br>Table 5: Performance results for the medical claims dataset, in which the task is to predict the effect of a_pair_of drugs the drug on pancytopenia occurrence. Mean<br>and standard deviation between runs is reported. Single-task methods were trained on the meta-testing tasks (best model underlined). Methods that were capable of<br>training across multiple tasks were trained on meta-training tasks and applied to previously unseen meta-testing tasks (best model in bold).CaML outperforms the<br>strongest baseline that had access to testing tasks on 12 out of 12 metrics, and outperforms all zero-shot baselines. Notably, due to the small sample size for natural<br>experiments with combinations of drugs,_the RATE estimation process is very noisy_which is refected in high variability of the measured RATE. Here, the secondary<br>metrics (Recall and Precision) that are not affected, additionally assert the dominance of CaML over all baselines.|
|---|---|---|---|---|
|Precision @_u_(_↑_)<br>0_._999<br>0_._998<br>0_._995<br>0_._99|0.01_±<_0.0014 0.01_±<_0.001 0.01_±<_0.001<br>0.00_±<_0.001|0.04_±<_0.001<br>0.01_±<_0.001<br>0.01_±<_0.001<br>0.11<br>_±<_0.001<br>0.00_±_0.003<br>0.02_±_0.012<br>0.04_±_0.014<br>0.05_±_0.032<br>0.05_±_0.032<br>0.04_±_0.013<br>0.01_±_0.005||0.15_±_0.008<br>0.17_±_0.010<br>**0.17**_±_0.013|
|||.06_±<_0.001<br>.00_±<_0.001<br>.00_±<_0.001<br>.18<br>_±<_0.001<br>0.00_±_0.004<br>0.03_±_0.013<br>0.04_±_0.012<br>0.09_±_0.055<br>0.09_±_0.055<br>0.07_±_0.027<br>0.01_±_0.009||0.23_±_0.017<br>**0.26**_±_0.014<br>0.25_±_0.023|
|||0<br>0<br>0<br>0|||
|||.08_±<_0.001<br>.00_±<_0.001<br>.00_±<_0.001<br>.28<br>_±<_0.001<br>0.00_±_0.010<br>0.04_±_0.032<br>0.03_±_0.020<br>0.18_±_0.123<br>0.18_±_0.123<br>0.16_±_0.057<br>0.01_±_0.014||**0.41**_±_0.057<br>0.36_±_0.033<br>0.40_±_0.044|
|||0<br>0<br>0<br>0|||
|||0.10_±<_0.001<br>0.00_±<_0.001<br>0.00_±<_0.001<br>0.30<br>_±<_0.001<br>0.00_±<_0.001<br>0.05_±_0.046<br>0.02_±_0.027<br>0.29_±_0.180<br>0.29_±_0.180<br>0.27_±_0.091<br>0.00_±_0.007||0.49_±_0.070<br>0.44_±_0.099<br>**0.51**_±_0.079|
|Recall @_u_ (_↑_)<br>0_._999<br>0_._998<br>0_._995<br>0_._99|0.0_±<_0.001<br>0.0_±<_0.001<br>0.01_±<_0.001 0.01_±<_0.001|.13_±<_0.001<br>.02_±<_0.001<br>.04_±<_0.001<br>.37<br>_±<_0.001<br>0.01_±_0.009<br>0.06_±_0.029<br>0.12_±_0.037<br>0.15_±_0.090<br>0.15_±_0.090<br>0.13_±_0.049<br>0.03_±_0.009||0.45_±_0.019<br>0.50_±_0.028<br>**0.51**_±_0.027|
|||0<br>0<br>0<br>0|||
|||.11_±<_0.001<br>.01_±<_0.001<br>.00_±<_0.001<br>.30<br>_±<_0.001<br>0.00_±_0.005<br>0.04_±_0.013<br>0.08_±_0.012<br>0.13_±_0.084<br>0.13_±_0.084<br>0.11_±_0.045<br>0.02_±_0.015||0.38_±_0.025<br>0.42_±_0.024<br>**0.43**_±_0.024|
|||0<br>0<br>0<br>0|||
|||.07_±<_0.001<br>.00_±<_0.001<br>.00_±<_0.001<br>.19<br>_±<_0.001<br>0.00_±_0.003<br>0.02_±_0.015<br>0.03_±_0.018<br>0.10_±_0.070<br>0.10_±_0.070<br>0.10_±_0.034<br>0.00_±_0.008||0.28_±_0.026<br>0.28_±_0.010<br>**0.30**_±_0.016|
|||0<br>0<br>0<br>0|||
|||0.05_±<_0.001<br>0.00_±<_0.001<br>0.00_±<_0.001<br>0.10<br>_±<_0.001<br>0.00_±<_0.001<br>0.01_±_0.011<br>0.01_±_0.009<br>0.09_±_0.055<br>0.09_±_0.055<br>0.08_±_0.024<br>0.00_±_0.005||0.19_±_0.019<br>0.18_±_0.025<br>**0.20**_±_0.015|
|RATE @_u_ (_↑_)<br>0_._999<br>_._998<br>0_._995<br>0_._99|0.00_±<_0.001<br>0.00_±<_0.001<br>0.00_±<_0.001<br>0.00_±<_0.001|0.04_±<_0.001<br>0.00_±<_0.001<br>0.00_±<_0.001<br>0.10<br>_±<_0.001<br>0.00_±_0.003<br>0.02_±_0.012<br>0.03_±_0.013<br>0.04_±_0.030<br>0.04_±_0.030<br>0.03_±_0.011<br>0.00_±_0.005||0.13_±_0.008<br>0.15_±_0.010<br>**0.15**_±_0.013|
|||0.05_±<_0.001<br>0.00_±<_0.001<br>0.01_±<_0.001<br>0.17<br>_±<_0.001<br>0.00_±_0.004<br>0.02_±_0.013<br>0.04_±_0.012<br>0.08_±_0.052<br>0.08_±_0.052<br>0.06_±_0.025<br>0.00_±_0.008||0.21_±_0.017<br>**0.24**_±_0.014<br>0.23_±_0.022|
|||-<br>|||
|||0.07_±<_0.001<br>0.01_±<_0.001<br>0.01_±<_0.001<br>0.26<br>_±<_0.001<br>0.00_±_0.009<br>0.03_±_0.030<br>0.02_±_0.019<br>0.16_±_0.118<br>0.16_±_0.118<br>0.15_±_0.054<br>0.00_±_0.014||**0.38**_±_0.057<br>0.33_±_0.034<br>0.37_±_0.044|
|||-<br>-<br>|||
|||0.10_±<_0.001<br>0.00_±<_0.001<br>-0.01_±<_0.001<br>0.28<br>_±<_0.001<br>-0.01_±_0.002<br>0.04_±_0.046<br>0.02_±_0.024<br>0.27_±_0.173<br>0.27_±_0.173<br>0.25_±_0.088<br>0.00_±_0.008||0.45_±_0.070<br>0.40_±_0.101<br>**0.47**_±_0.084|
||Random|T-learner<br>X-learner<br>R-learner<br>RA-learner<br>DragonNet<br>TARNet<br>FlexTENet<br>S-learner w/ meta-learning<br>T-learner w/ meta-learning<br>GraphITE<br>SIN||CaML - w/o meta-learning<br>CaML - w/o RA-learner<br>CaML (ours)|



30 

||Split|# of Patients|
|---|---|---|
|Allopurinol|Test|815,921|
|Pregabalin|Test|636,995|
|Mirtazapine|Test|623,980|
|Indomethacin|Test|560,380|
|Colchicine|Test|370,397|
|Hydralazine|Test|363,070|
|Hydroxychloroquine|Test|324,750|
|Methotrexate|Test|323,387|
|Memantine|Test|306,832|
|Fentanyl|Test|261,000|
|Etodolac|Val|438,854|
|Azathioprine|Val|100,000|



Table 6: Held-out test and validation drugs for our single-drug meta-testing and meta-validation datasets for our Claims evaluation in Table 2. Drugs are unseen (excluded) during training. All drugs are known to cause pancytopenia [40] 

||Split|# of Patients|
|---|---|---|
|Allopurinol + Hydralazine|Test|7,859|
|Methotrexate + Hydroxychloroquine|Test|25,716|
|Pregabalin + Fentanyl|Test|5,424|
|Indomethacin + Colchicine|Test|42,846|
|Mirtazapine + Memantine|Test|10,215|



Table 7: Held-out test pairs of drugs for our meta-testing and meta-validation datasets in Appendix Table B.3. Both drugs are unseen (excluded) during training. All drugs are known to cause pancytopenia [40] 

|Split|# Perturbagens|# Cell-Lines|Mean #Cell Lines/Task|
|---|---|---|---|
|Meta-training|9717|77|5.79|
|Meta-validation|304|11|9.99|
|Meta-testing|301|11|10.77|



Table 8: Composition of the meta-training, meta-validation and meta-testing sets for the LINCS dataset. No cell lines or drugs (tasks) were shared across any of the splits. 

|**Hyperparameter**|**Search range**|
|---|---|
|Num. of layers|_{_2_,_4_,_6_}_|
|Dim. of hidden layers|_{_128_,_256_}_|
|Dropout|_{_0_,_0_._1_}_|
|Learning rate|_{_3_×_10_−_3_,_1_×_10_−_3_,_3_×_10_−_4_,_1_×_10_−_4_}_|
|Meta learning rate|_{_1_}_|
|Weight decay|_{_5_×_10_−_3_}_|
|Reptile k|_{_1_,_10_,_50_}_|
|L1 regularization coeffcient|_{_0_,_1_×_10_−_7_,_5_×_10_−_7_}_|



Table 9: Hyperparameter search space for **CaML** (our proposed method) on the medical claims dataset. 

31 

|**Hyperparameter**|**Search range**|
|---|---|
|Num. of como layers|_{_2_,_4_,_6_}_|
|Num. of covariate layers|_{_2_,_4_,_6_}_|
|Num. of propensity layers|_{_2_,_4_,_6_}_|
|Num. of treatment layers|_{_2_,_4_,_6_}_|
|Dim. of hidden como layers|_{_128_,_256_}_|
|Dim. of hidden covariate layers|_{_128_,_256_}_|
|Dim. of hidden treatment layers|_{_128_,_256_}_|
|Dim. of hidden propensity layers|_{_16_,_32_,_64_,_128_}_|
|Dropout|_{_0_,_0_._1_}_|
|Learning rate|_{_3_×_10_−_3_,_1_×_10_−_3_,_3_×_10_−_4_,_1_×_10_−_4_}_|
|Meta learning rate|_{_1_}_|
|Sin Weight decay|_{_0_,_5_×_10_−_3_}_|
|Pro Weight decay|_{_0_,_5_×_10_−_3_}_|
|GNN Weight decay|_{_0_,_5_×_10_−_3_}_|
|Reptile k|_{_1_,_10_,_50_}_|
|L1 regularization coeffcient|_{_0_,_1_×_10_−_7_,_5_×_10_−_7_}_|



Table 10: Hyperparameter search space for **SIN** on the medical claims dataset. The SIN model consists of two stages, Stage 1 and Stage 2. For the Stage 1 model we searched the identical hyperparameter search space as for CaML (Table 9). For Stage 2, we used the hyperparameters shown in this table. 

|**Hyperparameter**|**Search range**|
|---|---|
|Num. of covariate layers|_{_2_,_4_,_6_}_|
|Num. of treatment layers|_{_2_,_4_,_6_}_|
|Dim. of hidden treatment layers|_{_128_,_256_}_|
|Dim. of hidden covariate layers|_{_128_,_256_}_|
|Dropout|_{_0_,_0_._1_}_|
|Independence regularization coeffcient|_{_0_,_0_._01_,_0_._1_,_1_._0_}_|
|Learning rate|_{_3_×_10_−_3_,_1_×_10_−_3_,_3_×_10_−_4_,_1_×_10_−_4_}_|
|Meta learning rate|_{_1_}_|
|Weight decay|_{_5_×_10_−_3_}_|
|Reptile k|_{_1_,_10_,_50_}_|
|L1 regularization coeffcient|_{_0_,_1_×_10_−_7_,_5_×_10_−_7_}_|
|Table 11: Hyperparameter search space|for**GraphITE**on the medical claims dataset.|



|**Hyperparameter**|**Search range**|
|---|---|
|Num. of out layers|_{_1_,_2_,_4_}_|
|Num. of r layers|_{_2_,_4_,_6_}_|
|Num. units p out|_{_32_,_64_,_128_,_256_}_|
|Num. units s out|_{_32_,_64_,_128_,_256_}_|
|Num. units s r|_{_32_,_64_,_128_,_256_}_|
|Num. units p r|_{_32_,_64_,_128_,_256_}_|
|Weight decay|_{_5_×_10_−_3_}_|
|Orthogonal penalty|_{_0_,_1_×_10_−_5_,_1_×_10_−_3_,_0_._1_}_|
|Private out|_{_True, False_}_|
|Learning rate|_{_3_×_10_−_3_,_1_×_10_−_3_,_3_×_10_−_4_,_1_×_10_−_4_}_|



Table 12: Hyperparameter search space for **FlexTENet** on the medical claims dataset. 

32 

|**Hyperparameter**|**Search range**|
|---|---|
|Num. of out layers|_{_1_,_2_,_4_}_|
|Num. of r layers|_{_2_,_4_,_6_}_|
|Num. units out|_{_128_,_256_}_|
|Weight decay|_{_5_×_10_−_3_}_|
|Penalty disc|_{_0_,_1_×_10_−_3_}_|
|Learning rate|_{_3_×_10_−_3_,_1_×_10_−_3_,_3_×_10_−_4_,_1_×_10_−_4_}_|



Table 13: Hyperparameter search space for **TARNet** on the medical claims dataset. 

|**Hyperparameter**|**Search range**|
|---|---|
|Num. of out layers|_{_1_,_2_,_4_}_|
|Num. of r layers|_{_2_,_4_,_6_}_|
|Num. units r|_{_128_,_256_}_|
|Num. units out|_{_128_,_256_}_|
|Weight decay|_{_5_×_10_−_3_}_|
|Learning rate|_{_3_×_10_−_3_,_1_×_10_−_3_,_3_×_10_−_4_,_1_×_10_−_4_}_|



Table 14: Hyperparameter search space for **DragonNet** on the medical claims dataset. 

|**Hyperparameter**|**Search range**|
|---|---|
|Num. of estimators|[50_,_250]|
|Max depth|[10_,_50]|
|Min sample split|[2_,_8]|
|Criterion regress|_{_squared error, absolute error_}_|
|Criterion binary|_{_gini, entropy_}_|
|Max features|_{_sqrt, log2, auto_}_|



Table 15: Hyperparameter search space for model-agnostic CATE estimators, i.e., **R-learner** , **Xlearner** , **RA-learner** , and **T-learner** on the medical claims dataset. 

|**Hyperparameter**|**Search range**|
|---|---|
|Num. of layers|_{_2_,_4_,_6_}_|
|Dim. of hidden layers|_{_512_,_1024_}_|
|Dropout|_{_0_,_0_._1_}_|
|Learning rate|_{_3_×_10_−_3_,_1_×_10_−_3_,_3_×_10_−_4_,_1_×_10_−_4_}_|
|Meta learning rate|_{_0_._1_,_0_._5_,_0_._9_}_|
|Weight decay|_{_0_._1_}_|
|Reptile k|_{_1_,_2_,_3_}_|
|L1 regularization coeffcient|_{_0_,_1_×_10_−_7_,_5_×_10_−_7_}_|



Table 16: Hyperparameter search space for **CaML** (our proposed method) on the LINCS dataset. 

33 

|**Hyperparameter**|**Search range**|
|---|---|
|Num. of como layers|_{_2_,_4_,_6_}_|
|Num. of covariates layers|_{_2_,_4_,_6_}_|
|Num. of propensity layers|_{_2_,_4_,_6_}_|
|Num. of treatment layers|_{_2_,_4_,_6_}_|
|Dim. output|_{_128_,_256_}_|
|Dim. of hidden treatment layers|_{_128_,_256_}_|
|Dim. of hidden covariate layers|_{_128_,_256_}_|
|Dim. of hidden como layers|_{_128_,_256_}_|
|Dim. of hidden propensity layers|_{_16_,_32_,_64_,_128_}_|
|Model dim.|_{_512_,_1024_}_|
|Dropout|_{_0_,_0_._1_}_|
|Learning rate|_{_3_×_10_−_3_,_1_×_10_−_3_,_3_×_10_−_4_,_1_×_10_−_4_}_|
|Meta learning rate|_{_0_._1_,_0_._5_,_0_._9_}_|
|Sin weight decay|_{_0_._0_,_0_._005_}_|
|Pro weight decay|_{_0_._0_,_0_._005_}_|
|GNN weight decay|_{_0_._0_,_0_._005_}_|
|Weight decay|_{_0_._1_}_|
|Reptile k|_{_1_,_2_,_3_}_|
|L1 regularization coeffcient|_{_0_,_1_×_10_−_7_,_5_×_10_−_7_}_|



Table 17: Hyperparameter search space for the **SIN** baseline on the LINCS dataset. 

|**Hyperparameter**|**Search range**|
|---|---|
|Num. of covariate layers|_{_2_,_4_,_6_}_|
|Num. of treatment layers|_{_2_,_4_,_6_}_|
|Num. of layers|_{_2_,_4_,_6_}_|
|Dim. of hidden covariate layers|_{_128_,_256_}_|
|Independence regularization coeffcient|_{_0_,_0_._01_,_0_._1_,_1_._0_}_|
|Dropout|_{_0_,_0_._1_}_|
|Model dim.|_{_512_,_1024_}_|
|Learning rate|_{_3_×_10_−_3_,_1_×_10_−_3_,_3_×_10_−_4_,_1_×_10_−_4_}_|
|Meta learning rate|_{_0_._1_,_0_._5_,_0_._9_}_|
|Weight decay|_{_0_._1_}_|
|Reptile k|_{_1_,_2_,_3_}_|
|L1 regularization coeffcient|_{_0_,_1_×_10_−_7_,_5_×_10_−_7_}_|



Table 18: Hyperparameter search space for the **GraphITE** baseline on the LINCS dataset. 

|**Hyperparameter**|**Search range**|
|---|---|
|Dropout|[1e-4,1e-1]|
|Learning rate|[1e-5,1e-3]|
|Weight decay|[1e-5,1e-2]|
|Adversarial temperature|[1,10]|
|Gamma|[0,30]|
|Num. of sampled neighbors|0-10|
|Dim. of hidden layers|{ 64, 128, 256, 512}|



Table 19: The hyperparameter optimization search ranges used in the selection of the optimal model for the generation of knowledge graph node embeddings that would serve as intervention information for the medical claims dataset. 

34 

