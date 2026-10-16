Published as a conference paper at ICLR 2024 

## A HIERARCHICAL BAYESIAN MODEL FOR FEW-SHOT META LEARNING 

**Minyoung Kim**[1] **& Timothy M. Hospedales**[1] _[,]_[2] 

1Samsung AI Center Cambridge, UK mikim21@gmail.com 

2University of Edinburgh, UK t.hospedales@ed.ac.uk 

## ABSTRACT 

We propose novel model parametrisation and inference algorithm in a hierarchical Bayesian model for the few-shot meta learning problem. We consider episode-wise random variables to model episode-specific generative processes, where these local random variables are governed by a higher-level global random variable. The global variable captures information shared across episodes, while controlling how much the model needs to be adapted to new episodes in a principled Bayesian manner. Within our framework, prediction on a novel episode/task can be seen as a Bayesian inference problem. For tractable training, we need to be able to relate each local episode-specific solution to the global higher-level parameters. We propose a Normal-Inverse-Wishart model, for which establishing this localglobal relationship becomes feasible due to the approximate closed-form solutions for the local posterior distributions. The resulting algorithm is more attractive than the MAML in that it does not maintain a costly computational graph for the sequence of gradient descent steps in an episode. Our approach is also different from existing Bayesian meta learning methods in that rather than modeling a single random variable for all episodes, it leverages a hierarchical structure that exploits the local-global relationships desirable for principled Bayesian learning with many related tasks. 

## 1 INTRODUCTION 

Few-shot learning (FSL) aims to emulate the human ability to learn from few examples (Lake et al., 2015). It has received substantial and growing interest (Wang et al., 2020b) due to the need to alleviate the notoriously data intensive nature of mainstream supervised deep learning. Approaches to FSL are all based on some kind of knowledge transfer from a set of plentiful source recognition problems to the sparse data target problem of interest. Existing approaches are differentiated in terms of the assumptions they make about what is task agnostic knowledge that can be transferred from the source tasks, and what is task-specific knowledge that should be learned from the sparse target examples. For example, the seminal MAML (Finn et al., 2017) and ProtoNets (Snell et al., 2017) respectively assume that the initialisation for fine-tuning, or the feature extractor for metric-based recognition should be transferred from source categories. 

One of the most principled and systematic ways to model such sets of related problems are hierarchical Bayesian models (HBMs) (Gelman et al., 2003). The HBM paradigm is widely used in statistics, but has seen relatively less use in deep learning, due to the technical difficulty of bringing hierarchical Bayesian modelling to bear on deep learning. HBMs provide a powerful way to model a set of related problems, by assuming that each problem has its own parameters (e.g, the neural networks that recognise cat vs dog, or car vs bike), but that those problems share a common prior (the prior over such neural networks). Data-efficient learning of the target tasks is then achieved by inferring the prior based on source tasks, and using it to enhance posterior learning over the target task parameters. 

A Bayesian learning treatment of FSL would be appealing due to the overfitting resistance provided by Bayesian Occam’s razor (MacKay, 2003), as well as the ability to improve calibration of inference so that the model’s confidence is reflective of its probability of correctness — a crucial property in mission critical applications (Guo et al., 2017). However the limited attempts that have been made to exploit these tools in deep learning have either been incomplete treatments that only model a single Bayesian layer within the neural network (Zhang et al., 2021; Gordon et al., 2019), or else fail to scale up to modern neural architectures (Finn et al., 2018; Yoon et al., 2018). 

1 

Published as a conference paper at ICLR 2024 

In this paper we present the first complete hierarchical Bayesian learning algorithm for few-shot deep learning. Our algorithm efficiently learns a prior over neural networks during the meta-train phase, and efficiently learns a posterior neural network during each meta-test episode. Importantly, our learning is architecture independent. It can scale up to the state-of-the-art backbones including ViTs (Dosovitskiy et al., 2021), and works smoothly with any few-shot learning architecture – spanning simple linear decoders (Finn et al., 2017; Snell et al., 2017), to those based on sophisticated set-based decoders such as FEAT (Ye et al., 2020) and CNP(Garnelo et al., 2018)/ANP(Kim et al., 2019). We show empirically that our HBM provides improved performance and calibration in all of these cases, as well as providing clear theoretical justification. 

Our analysis also reveals novel links between seminal FSL methods such as ProtoNet (Snell et al., 2017), MAML (Finn et al., 2017), and Reptile (Nichol et al., 2018), all of which are different special cases of our framework despite their very different appearance. Interestingly, despite its close relatedness to MAML-family algorithms, our Bayesian learner admits an efficient closedform solution to the task-specific and task-agnostic updates that does not require maintaining the computational graph for reverse-mode backpropagation. This provides a novel solution to a famous meta-learning scalability bottleneck. **In summary, our contributions include** : (i) The first complete hierarchical Bayesian treatment of the few-shot deep learning problem, and associated theoretical justification. (ii) An efficient algorithmic learning solution that can scale up to modern architectures, and plug into most existing neural FSL meta-learners. (iii) Empirical results demonstrating improved accuracy and calibration performance on both classification and regression benchmarks. 

## 2 PROBLEM SETUP 

Let _p_ ( _T_ ) be the (unknown) task/episode distribution, where each task _T ∼ p_ ( _T_ ) is defined as a distribution _pT_ ( _x, y_ ) for data ( _x, y_ ) where _x_ is input and _y_ is target. For training, we have a large number of episodes, _T_ 1 _, T_ 2 _, . . . , TN ∼ P_ ( _T_ ) sampled i.i.d., but we only observe a small number of labeled samples from each episode, denoted by _Di_ = _{_ ( _x[i] j[, y] j[i]_[)] _[}][n] j_ =1 _[i][∼][p][T] i_[(] _[x, y]_[)][, where] _[ n][i]_[ =] _[|][D][i][|]_[ is the] number of samples in _Di_ . The goal of the learner, after observing the training data _D_ 1 _, . . . , DN_ from a large number of different tasks, is to build a predictor _p[∗]_ ( _y|x_ ) for novel unseen tasks _T[∗] ∼ p_ ( _T_ ). We will often abuse the notation, e.g., _i ∼T_ refers to the episode _i_ sampled, i.e., _Di ∼ pTi_ ( _x, y_ ) where _Ti ∼ p_ ( _T_ ). At the test time we are allowed to have some hints about the new test task _T[∗]_ , in the form of a few labeled examples from _T[∗]_ , also known as the _support set_ denoted by _D[∗] ∼ PT ∗_ ( _x, y_ ). 

In Bayesian perspective, the goal is to infer the posterior distribution with all training episodes and a test support set as _evidence_ , i.e., _p_ ( _y|x, D[∗] , D_ 1: _N_ ). A major computational challenge, compared to conventional Bayesian learning, is that the training episodes (evidence) may not be stored/replayed/revisited. 

## 3 MAIN APPROACH 


![](raw/papers/2_LLM_NLP/LLM_paper_list/few-shot/iclr_2024_A_Hierarchical_Bayesian_Model_for_Few-Shot_Meta_Learning_images/iclr_2024_A_Hierarchical_Bayesian_Model_for_Few-Shot_Meta_Learning.pdf-0002-07.png)


Figure 1: (a) IID episodes. (b) Individual episode. (c): FSL as Bayesian inference (grey nodes = _evidences_ , red = _target_ to infer). _D[∗]_ = support set for test episode. 

We introduce two types of latent random variables, _ϕ_ and _{θi}[N] i_ =1[.][Each episode] _[ i]_[ uses neural network weights] _[ θ][i]_[ for modeling the data] _[ D][i]_[ (] _[i]_[ = 1] _[, . . . , N]_[).] Specifically, _Di_ is generated (input _x_ given and only _p_ ( _y|x_ ) modeled) by _θi_ as in the likelihood model in (1). The variable _ϕ_ can be viewed as a globally shared variable that is responsible for linking the individual episode-wise parameters _θi_ . We assume conditionally independent and identical priors, _p_ ( _{θi}i|ϕ_ ) =[�] _i[p]_[(] _[θ][i][|][ϕ]_[)][.][Thus the prior for the latent variables][ (] _[ϕ,][ {][θ][i][}][N] i_ =1[)][ is formed in a] hierarchical manner as follows. (For background on Bayesian modeling, refer to (Murphy, 2022).) 

(Prior) _p_ ( _ϕ, θ_ 1: _N_ ) = _p_ ( _ϕ_ )[�] _[N] i_ =1 _[p]_[(] _[θ][i][|][ϕ]_[)] _[,]_ (Likelihood) _p_ ( _Di|θi_ ) =[�] ( _x,y_ ) _∈Di[p]_[(] _[y][|][x, θ][i]_[)] (1) where _p_ ( _y|x, θi_ ) is a conventional neural network model. See the graphical model in Fig. 1(a). 

Given the training data _{Di}[N] i_ =1[,][the][posterior][is] _[p]_[(] _[ϕ, θ]_[1:] _[N][|][D]_[1:] _[N]_[)] _[∝][p]_[(] _[ϕ]_[)][ �] _[N] i_ =1 _[p]_[(] _[θ][i][|][ϕ]_[)] _[p]_[(] _[D][i][|][θ][i]_[)][,] and we approximate it with variational inference. That is, _q_ ( _ϕ, θ_ 1: _N_ ; _L_ ) _≈ p_ ( _ϕ, θ_ 1: _N |D_ 1: _N_ ) where 


![](raw/papers/2_LLM_NLP/LLM_paper_list/few-shot/iclr_2024_A_Hierarchical_Bayesian_Model_for_Few-Shot_Meta_Learning_images/iclr_2024_A_Hierarchical_Bayesian_Model_for_Few-Shot_Meta_Learning.pdf-0002-12.png)


where the variational parameters _L_ consists of _L_ 0 (parameters for _q_ ( _ϕ_ )) and _{Li}[N] i_ =1[’s (parameters] of _qi_ ( _θi_ )’s for episode _i_ ). Note that although _θi_ ’s are independent across episodes under (2), they 

2 

Published as a conference paper at ICLR 2024 

are differently modeled (note the subscript _i_ in notation _qi_ ), reflecting different posterior beliefs originating from heterogeneity of episodic datasets _Di_ . 

**Normal-Inverse-Wishart (NIW) model.** We consider NIW distributions for the prior and variational posterior. First, the prior is modeled as a conjugate form of Gaussian–NIW. With _ϕ_ = ( _µ,_ Σ), _p_ ( _ϕ_ ) = _N_ ( _µ_ ; _µ_ 0 _, λ[−]_ 0[1][Σ)] _[ · IW]_[(Σ; Σ][0] _[, ν]_[0][)] _[,] p_ ( _θi|ϕ_ ) = _N_ ( _θi_ ; _µ,_ Σ) _, i_ = 1 _, . . . , N,_ (3) where Λ = _{µ_ 0 _,_ Σ0 _, λ_ 0 _, ν_ 0 _}_ is the parameters of the NIW. We do not need to pay attention to the choice of values for Λ since _p_ ( _ϕ_ ) has vanishing effect on posterior for a large amount of evidence as we will see shortly. Next, our choice of the variational density family for _q_ ( _ϕ_ ) is the NIW, mainly because it admits closed-form expressions in the ELBO function due to the conjugacy, allowing efficient local episodic optimisation, as will be shown. For _qi_ ( _θi_ )’s we adopt Gaussians. That is, 

_q_ ( _ϕ_ ; _L_ 0) := _N_ ( _µ_ ; _m_ 0 _, l_ 0 _[−]_[1][Σ)] _[ · IW]_[(Σ;] _[ V]_[0] _[, n]_[0][)] _[,] qi_ ( _θi_ ; _Li_ ) = _N_ ( _θi_ ; _mi, Vi_ ) _._ (4) So, _L_ 0 = _{m_ 0 _, V_ 0 _, l_ 0 _, n_ 0 _}_ with _V_ 0 restricted to be diagonal, and _Li_ = _{mi, Vi}_ . Learning (variational inference) amounts to finding _L_ 0 and _{Li}[N] i_ =1[that][makes][the][approximation] _q_ ( _ϕ, θ_ 1: _N_ ; _L_ ) _≈ p_ ( _ϕ, θ_ 1: _N |D_ 1: _N_ ), as tight as possible. 

**Variational inference.** The negative marginal log-likelihood (NMLL) has the following upper bound (Appendix B.1 for derivations): 

_−_ log _p_ ( _D_ 1: _N_ ) _≤_ KL( _q_ ( _ϕ_ ) _||p_ ( _ϕ_ )) +[�] _[N] i_ =1�E _qi_ ( _θi_ )[ _li_ ( _θi_ )] + E _q_ ( _ϕ_ )�KL( _qi_ ( _θi_ ) _||p_ ( _θi|ϕ_ ))�[�] (5) where _li_ ( _θi_ )= _−_ log _p_ ( _Di|θi_ ) is the negative training log-likelihood of _θi_ in episode _i_ . By dividing both sides by _N_ , the LHS naturally becomes the _effective episode-averaged NMLL − N_[1][log] _[ p]_[(] _[D]_[1:] _[N]_[)][.] The first KL term in the RHS, _N_ 1[KL][(] _[q]_[(] _[ϕ]_[)] _[||][p]_[(] _[ϕ]_[))][diminishes][for][large] _[N]_[.][Using] _N_ 1 � _Ni_ =1 _[f][i][≈]_ E _i∼T_ [ _fi_ ] for any expression _fi_ , the ELBO learning (approximately) reduces to the following: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/few-shot/iclr_2024_A_Hierarchical_Bayesian_Model_for_Few-Shot_Meta_Learning_images/iclr_2024_A_Hierarchical_Bayesian_Model_for_Few-Shot_Meta_Learning.pdf-0003-06.png)


**Local episodic optimisation (whose solution as a function of global parameters** _L_ 0 **).** Note that (6) is challenging due to a large number of optimisation variables _{Li}[N] i_ =1[and the nature of episode] sampling _i ∼T_ . Applying conventional SGD would simply fail since each _Li_ will never be updated more than once. Instead, we tackle it by finding the optimal solutions for _Li_ ’s for fixed _L_ 0, thus effectively representing the optimal solutions as functions of _L_ 0, namely _{L[∗] i_[(] _[L]_[0][)] _[}][N] i_ =1[.][Plugging the] optimal _L[∗] i_[(] _[L]_[0][)][’s back to (6) leads to the optimisation problem over] _[ L]_[0][ alone.][The idea is just like] solving: min _x,y f_ ( _x, y_ ) = min _x f_ ( _x, y[∗]_ ( _x_ )) where _y[∗]_ ( _x_ ) = arg min _y f_ ( _x, y_ ) with _x_ fixed. 

Note that when we fix _L_ 0 (i.e., fix _q_ ( _ϕ_ )), the objective (6) is completely separable over _i_ , and we can optimise individual _i_ independently. More specifically, for each _i ≥_ 1, 

min _Li_ E _qi_ ( _θi_ ; _Li_ )[ _li_ ( _θi_ )] + E _ϕ_ �KL( _qi_ ( _θi_ ; _Li_ ) _||p_ ( _θi|ϕ_ ))� (7) 

As the expected KL term in (7) admits a closed form due to NIW-Gaussian conjugacy (Appendix B.2 for derivations), we can reduce (7) to the following optimisation for _Li_ = ( _mi, Vi_ ): 


![](raw/papers/2_LLM_NLP/LLM_paper_list/few-shot/iclr_2024_A_Hierarchical_Bayesian_Model_for_Few-Shot_Meta_Learning_images/iclr_2024_A_Hierarchical_Bayesian_Model_for_Few-Shot_Meta_Learning.pdf-0003-11.png)


**Quadratic approximation of episodic loss via SGLD.** To find the closed-form solution _L[∗] i_[(] _[L]_[0][)] in (8), we make quadratic approximation of _li_ ( _θi_ )= _−_ log _p_ ( _Di|θi_ ). In general, _−_ log _p_ ( _Di|θ_ ), as a function of _θ_ , can be written as: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/few-shot/iclr_2024_A_Hierarchical_Bayesian_Model_for_Few-Shot_Meta_Learning_images/iclr_2024_A_Hierarchical_Bayesian_Model_for_Few-Shot_Meta_Learning.pdf-0003-13.png)


for some ( _mi, Ai_ ) that are constant with respect to _θ_ . One may attempt to obtain ( _mi, Ai_ ) via Laplace approximation (e.g., the minimiser of _−_ log _p_ ( _Di|θ_ ) for _mi_ and the Hessian at the minimiser for _Ai_ ). However, this involves computationally intensive Hessian computation. Instead, using the fact that the log-posterior log _p_ ( _θ|Di_ ) equals (up to constant) log _p_ ( _Di|θ_ ) when we use uninformative prior _p_ ( _θ_ ) _∝_ 1, we can obtain samples from the posterior _p_ ( _θ|Di_ ) using MCMC sampling, especially the stochastic gradient Langevin dynamics (SGLD) (Welling & Teh, 2011), and estimate sample mean and precision, which become _mi_ and _Ai_ , respectively[1] . Note that this amounts to performing 

> 1Similar approaches include the stochastic weight averaging (Izmailov et al., 2018; Maddox et al., 2019). 

3 

Published as a conference paper at ICLR 2024 

several SGD iterations (skipping a few initial for burn-in), and unlike MAML (Finn et al., 2017) no computation graph needs to be maintained since ( _mi, Ai_ ) are constant. Once we have ( _mi, Ai_ ), the optimisation (8) admits the closed-form solution (Appendix B.4 for derivations), 

_m[∗] i_[(] _[L]_[0][) = (] _Ai_ + _n_ 0 _/V_ 0) _[−]_[1] ( _Aimi_ + _n_ 0 _m_ 0 _/V_ 0) _, Vi[∗]_[(] _[L]_[0][) = (] _Ai_ + _n_ 0 _/V_ 0) _[−]_[1] _._ (10) Computation in (10) is cheap since all matrices are diagonal. 

**Final optimisation.** Plugging (10) back to (6), the final optimisation is (Appendix B.5 for details): min _L_ 0 E _i∼T_ � _fi_ ( _L_ 0) +[1] 2 _[g][i]_[(] _[L]_[0][) +] 2 _[d] l_ 0 � s.t. _fi_ ( _L_ 0) = E _ϵ∼N_ (0 _,I_ )� _li_ � _m[∗] i_[(] _[L]_[0][) +] _[ V] i[∗]_[(] _[L]_[0][)][1] _[/]_[2] _[ϵ]_ �� _, gi_ ( _L_ 0) = log _|Vi|[∗] V_[(] 0 _[L] |_[0][)] _[|]_[+] _[ n]_[0][Tr] � _Vi[∗]_[(] _[L]_[0][)] _[/V]_[0] � + _n_ 0� _m[∗] i_[(] _[L]_[0][)] _[−][m]_[0] �2 _/V_ 0 _− ψd_ � _n_ 20 � _,_ (11) 

where _ψd_ ( _·_ ) is the multivariate digamma **Algorithm 1** Our few-shot meta learning algorithm. function and _d_ = dim( _θ_ ). As _l_ 0 only appears in the term 2 _dl_ 0[,][the][optimal][value] **Initialise:for** episode _L i_ 0 = 1= _{,_ 2 _m, . . ._ 0 _, V_ 0 **do** _, n_ 0 _}_ of _q_ ( _ϕ_ ; _L_ 0) randomly. is _l_ 0 _[∗]_[=] _[ ∞]_[.][We][use][SGD][to][solve][(11),] Perform SGLD iterations on _Di_ to estimate ( _mi, Ai_ ). repeating the two steps: i) Sample _i ∼T_ ; Compute the episodic minimiser _L[∗] i_[(] _[L]_[0][)][ from (10).] ii) _L_ 0 _← L_ 0 _−η∇L_ 0� _fi_ ( _L_ 0)+[1] 2 _[g][i]_[(] _[L]_[0][)] �. Update _L_ 0 by the gradient of _fi_ ( _L_ 0)+[1] 2 _[g][i]_[(] _[L]_[0][)][ as in (11).] Note that _∇L_ 0� _fi_ ( _L_ 0) +[1] 2 _[g][i]_[(] _[L]_[0][)] � is an **end for** _unbiased_ stochastic estimate for the gra- **Output:** Learned _L_ 0. dient of the objective E _i∼T_ [ _· · ·_ ] in (11). Furthermore, our learning algorithm above (pseudocode in Alg. 1) is fully compatible with the online/batch episode sampling nature of typical FSL. After training, we obtain the learned _L_ 0, and the posterior _q_ ( _ϕ_ ; _L_ 0) will be used at the meta test time, where we show in Sec. 3.2 that this can be seen as Bayesian inference as well. 

We emphasise that our framework is completely flexible in the choice of the backbone _p_ ( _y|x, θ_ ). It could be the popular instance-based network comprised of a feature extractor and a prediction head where the latter can be either a conventional learnable readout head or the parameter-free one like the nearest centroid classifier (NCC) in ProtoNet (Snell et al., 2017), i.e., _p_ ( _D|θ_ )= _p_ ( _Q|S, θ_ ) where _D_ = _S ∪ Q_ and _p_ ( _y|x, S, θ_ ) is the NCC prediction with support _S_ . We can also adopt the set-based networks (Ye et al., 2020; Garnelo et al., 2018; Kim et al., 2019) where _p_ ( _y|x, S, θ_ ) itself is modeled by a neural net _y_ = _G_ ( _x, S_ ; _θ_ ) with input ( _x, S_ ). 

## 3.1 INTERPRETATION 

We show that our framework unifies seemingly unrelated seminal FSL algorithms into one perspective. 

**MAML (Finn et al., 2017) as a special case.** Suppose we have spiky variational densities, _Vi →_ 0 (constant). The local episodic optimisation (8) reduces to: arg min _mili_ ( _θi_ )+ _R_ ( _mi_ ) where _R_ ( _mi_ ) is the quadratic penalty of _mi_ deviating from _m_ 0. One reasonable solution is to perform a few gradient steps with loss _li_ , starting from _m_ 0 to have small penalty ( _R_ =0 initially). That is, _mi ← m_ 0 and a few steps of _mi ← mi − α∇li_ ( _mi_ ) to return _m[∗] i_[(] _[L]_[0][)][.][Plugging this into (11) and disregarding the] _gi_ term, leads to the MAML algorithm. Obviously, the main drawback is _m[∗] i_[(] _[L]_[0][)][ is a function of] _m_ 0 _∈ L_ 0 via a full computation graph of SGD steps, compared to our lightweight closed forms (10). 

**ProtoNet (Snell et al., 2017) as a special case.** With _Vi →_ 0, if we ignore the negative loglikelihood term in (8), then the optimal solution becomes _m[∗] i_[(] _[L]_[0][) =] _[ m]_[0][.][If we remove the] _[ g][i]_[ term,] we can solve (11) by simple gradient descent with _∇m_ 0( _−_ log _p_ ( _Di|m_ 0)). We then adopt the NCC head and regard _m_ 0 as sole feature extractor parameters, which becomes exactly the ProtoNet update. 

**Reptile (Nichol et al., 2018) as a special case.** Instead, if we ignore all penalty terms in (8) and follow our quadratic approximation (9) with _Vi →_ 0, then _m[∗] i_[(] _[L]_[0][) =] _mi_ . It is constant with respect to _L_ 0 = ( _m_ 0 _, V_ 0 _, n_ 0), and makes the optimisation (11) very simple: the optimal _m_ 0 is the average of _mi_ for all tasks _i_ , i.e., _m[∗]_ 0[=][E] _[i][∼T]_[[] _mi_ ] (we ignore _V_ 0 here). Note that Reptile ultimately finds the exponential smoothing of _m_[(] _i[k]_[)] over _i ∼T_ where _m_[(] _i[k]_[)] is the iterate after _k_ SGD steps for task _i_ . This can be seen as an online/running estimate of E _i∼T_ [ _mi_ ]. 

## 3.2 META TEST PREDICTION AS BAYESIAN INFERENCE 

At meta test time, we need to be able to predict the target _y[∗]_ of a novel test input _x[∗] ∼T[∗]_ sampled from the unknown distribution _T[∗] ∼ p_ ( _T_ ). In FSL, we have the test support data _D[∗]_ = _{_ ( _x, y_ ) _}∼T[∗]_ . The 

4 

Published as a conference paper at ICLR 2024 

Table 1: Three competing Bayesian models for the toy experiment. (Fig. 3 for graphical models) 

|Model I|Model II|Model III(Ours)|
|---|---|---|
|_y_ =_θ⊤_<br>_i x_+_βi_ +_ϵy_,<br>_p_(_θi, βi_) =_N_(_µ, σ_2) _∀i_|_y_ =_θ⊤x_+_β_+_ϵy_,<br>_p_(_θ, β_) =_N_(_µ, σ_2)|_y_ =_θ⊤_<br>_i x_+_βi_ +_ϵy_,<br>_p_(_ϕ_) =_N_(_m, V_)_, p_(_θi, βi|ϕ_) =_N_(_ϕ, σ_2) _∀i_|



test-time prediction can be seen as a posterior inference problem with _additional evidence_ of the support data _D[∗]_ (Fig. 1(c)). More specifically, _p_ ( _y[∗] |x[∗] , D[∗] , D_ 1: _N_ ) = � _p_ ( _y[∗] |x[∗] , θ_ ) _p_ ( _θ|D[∗] , D_ 1: _N_ ) _dθ_ . So, it boils down to _p_ ( _θ|D[∗] , D_ 1: _N_ ), the posterior given both the test support data _D[∗]_ and the entire training data _D_ 1: _N_ . Under our hierarchical model, exploiting conditional independence (Fig. 1(c)), we can link it to our trained _q_ ( _ϕ_ ) as: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/few-shot/iclr_2024_A_Hierarchical_Bayesian_Model_for_Few-Shot_Meta_Learning_images/iclr_2024_A_Hierarchical_Bayesian_Model_for_Few-Shot_Meta_Learning.pdf-0005-04.png)


where in the first approximation in (12) we disregard the impact of _D[∗]_ on the higher-level _ϕ_ given the joint evidence, i.e., _p_ ( _ϕ|D[∗] , D_ 1: _N_ ) _≈ p_ ( _ϕ|D_ 1: _N_ ), due to dominance of _D_ 1: _N_ compared to smaller _D[∗]_ . We use the delta function approximation in the last part of (12) with the mode _ϕ[∗]_ of _q_ ( _ϕ_ ), where _ϕ[∗]_ = ( _µ[∗] ,_ Σ _[∗]_ ) has a closed form _µ[∗]_ = _m_ 0 _,_ Σ _[∗]_ = _V_ 0 _/_ ( _n_ 0 + _d_ +2). 

Next, since _p_ ( _θ|D[∗] , ϕ[∗]_ ) involves difficult marginalisation _p_ ( _D[∗] |ϕ[∗]_ ) = � _p_ ( _D[∗] |θ_ ) _p_ ( _θ|ϕ[∗]_ ) _dθ_ , we adopt variational inference, introducing a tractable variational distribution _v_ ( _θ_ ) _≈ p_ ( _θ|D[∗] , ϕ[∗]_ ). With the Gaussian family as in the training time (4), i.e., _v_ ( _θ_ ) = _N_ ( _θ_ ; _m, V_ ) where ( _m, V_ ) are the variational parameters optimised by ELBO optimisation, 


![](raw/papers/2_LLM_NLP/LLM_paper_list/few-shot/iclr_2024_A_Hierarchical_Bayesian_Model_for_Few-Shot_Meta_Learning_images/iclr_2024_A_Hierarchical_Bayesian_Model_for_Few-Shot_Meta_Learning.pdf-0005-07.png)


The detailed derivations for (13) can be found in Appendix B.6. Once we have the optimised model _v_ , our predictive distribution can be approximated by the Monte-Carlo average. _p_ ( _y[∗] |x[∗] , D[∗] , D_ 1: _N_ ) _≈_ (1 _/MS_ )[�] _[M] s_ =1 _[S][p]_[(] _[y][∗][|][x][∗][, θ]_[(] _[s]_[)][)][, where] _[ θ]_[(] _[s]_[)] _[ ∼][v]_[(] _[θ]_[)][ for] _[ s]_[ = 1] _[, . . . , M][S]_[samples.][which simply requires] feed-forwarding _x[∗]_ through the sampled networks _θ_[(] _[s]_[)] and averaging. Our meta-test algorithm is also summarised in Alg. 2 (Appendix). Note that we have test-time backbone update as per (13), which can make the final _m_ deviate from the learned mean _m_ 0. Alternatively, if we drop the first term in (13), the optimal _v_ ( _θ_ ) equals _p_ ( _θ|ϕ[∗]_ ) = _N_ ( _θ_ ; _m_ 0 _, V_ 0 _/_ ( _n_ 0 + _d_ +2)). This can be seen as using the learned model _m_ 0 with some small random perturbation as a test-time backbone _θ_ . 

## 4 TOY EXPERIMENT: WHY HIERARCHICAL BAYESIAN MODEL? 

To demonstrate why our hierarchical Bayesian modelling is effective for few-shot meta learning problems, we devise a simple toy synthetic experiment as a proof of concept. We consider a multi-task (Bayesian) linear regression problem. The data pairs ( _x ∈_ R[2] _, y ∈_ R) for each episode _i_ are generated by the following process: _y_ = ( _w_ shared + _ϵw_ ) _[⊤] x_ + _bj_ ( _i_ ) + _ϵy_ where _w_ shared is the episode-agnostic shared weight vector _∀i_ , and we have episode-dependent intercept _bj_ ( _i_ ) – among the three candidates _{b_ 1 _, b_ 2 _, b_ 3 _}_ , we select _j_ ( _i_ ) _∼{_ 1 _,_ 2 _,_ 3 _}_ uniformly at random for each episode _i_ . _Please refer to Appendix. C for full details and derivations._ In this way we ensure that the resulting episodes are not only related to one another through the shared weight vector, but they are differentiated by potentially different intercepts. We sample _N_ =40 episodes for training and 10 episodes for test. Each training episode has _|Di|_ = 3 samples, and the support set at test time also has _|D∗|_ = 3 labeled samples. 

**Three competing models.** We consider three Bayesian models with different levels/degrees of flexibility and regularisation as outlined in Table 1. **Model I** has episode-wise parameters ( _θi, βi_ ), thus highly flexible. However, these parameters are all independent across episodes, hence lacking regularisation. **Model II** is a conventional (non-hierarchical) Bayesian model where a single parameter set ( _θ, β_ ) is shared across episodes, thus too much regularisation with lack of flexibility. **Model III** is our hierarchical Bayesian model which imposes balanced flexibility and regularisation – episodewise ( _θi, βi_ )’s allows high flexibility, but unlike Model I, the higher-level variable _ϕ_ regularises the episode-specific parameters, and captures the inter-episodic shared information. 

**Results.** After training (learning _µ_ for Model I and II; learning ( _m, V_ ) for our Model III), at test time, for each of 10 test episodes, we obtain the posterior means of the weights and intercept parameters E[ _θ, β|D∗, D_ 1: _N_ ] for the three models. They all admit closed-form solutions as detailed in Appendix C, and we predict the outputs of _∼_ 50 unseen test inputs. The mean absolute errors (MAE) averaged over 10 test episodes are: **Model I** = **2** _._ **87** , **Model II** = **3** _._ **13** , and **Model III (ours)** = **1** _._ **28** , clearly showing the superiority of our model to other competing methods. Fig. 2 visualises 

5 

Published as a conference paper at ICLR 2024 


![](raw/papers/2_LLM_NLP/LLM_paper_list/few-shot/iclr_2024_A_Hierarchical_Bayesian_Model_for_Few-Shot_Meta_Learning_images/iclr_2024_A_Hierarchical_Bayesian_Model_for_Few-Shot_Meta_Learning.pdf-0006-01.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/few-shot/iclr_2024_A_Hierarchical_Bayesian_Model_for_Few-Shot_Meta_Learning_images/iclr_2024_A_Hierarchical_Bayesian_Model_for_Few-Shot_Meta_Learning.pdf-0006-02.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/few-shot/iclr_2024_A_Hierarchical_Bayesian_Model_for_Few-Shot_Meta_Learning_images/iclr_2024_A_Hierarchical_Bayesian_Model_for_Few-Shot_Meta_Learning.pdf-0006-03.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/few-shot/iclr_2024_A_Hierarchical_Bayesian_Model_for_Few-Shot_Meta_Learning_images/iclr_2024_A_Hierarchical_Bayesian_Model_for_Few-Shot_Meta_Learning.pdf-0006-04.png)


**----- Start of picture text -----**<br>
(a) Weight dim-1 (b) Weight dim-2 (c) Intercept<br>**----- End of picture text -----**<br>


Figure 2: Toy experiments. Visualisation of the learned posterior means compared to the true values (blue-circled). (a) weight dim-1 ( _θ_ [0] vs. _w_ shared[0]), (b) weight dim-2 ( _θ_ [1] vs. _w_ shared[1]) and (c) intercept ( _β_ vs. _bj_ ( _∗_ )). In each plot, the X-axis shows the indices of the true intercepts sampled, that is, _j_ ( _∗_ ) _∈{_ 1 _,_ 2 _,_ 3 _}_ , for 10 test episodes. In the titles we also report the distances (errors) between the true values and the posterior means for the three methods, averaged over 10 episodes. 

the results. First, **Model II** ’s posterior means rarely change over test episodes, meaning the impact of test support data _D∗_ is limited. This behavior is expected since the model imposes too much regularisation with little flexibility, and the test prediction is dominated by the mean model obtained from training data _D_ 1: _N_ . **Model I** exhibits highly sensitive predictions over test episodes, which mainly originates from little regularisation – the posterior is too sensitive to the current episode’s support data, thus being vulnerable to overfitting especially when the support data size is small, typical in the few-shot learning. The model failed to capture useful shared information _w_ shared. Our **Model III** balances between these two extremes, imposing proper amount of regularisation and endowing adequate flexibility. Our posterior estimation best extracts the shared episode-agnostic information (the weight parameters fluctuate less over the test episodes), and captures the episode-specific features the most accurately (the estimated intercepts are aligned the best with the true values). 

## 5 THEORETICAL ANALYSIS: GENERALISATION ERROR BOUNDS 

We offer two theorems for the generalisation error of the proposed model. The first theorem relates the generalisation error to the ultimate ELBO loss (6) that we minimised in our algorithm, and we utilise the recent PAC-Bayes- _λ_ bound (Thiemann et al., 2017; Rivasplata et al., 2019). The second theorem is based on the recent regression analysis technique (Pati et al., 2018; Bai et al., 2020). Without loss of generality we assume _|Di|_ = _n_ for all episodes _i_ . We let ( _q[∗]_ ( _ϕ_ ) _, {qi[∗]_[(] _[θ][i]_[)] _[}][N] i_ =1[)][ be the] optimal solution of (6). We leave technical details and the proofs for both theorems in Appendix A. 

**Theorem 5.1** (PAC-Bayes- _λ_ bound) **.** _Let Ri_ ( _θ_ ) _be the generalisation error of model θ for the task i, more specifically, Ri_ ( _θ_ ) = E( _x,y_ ) _∼Ti_ [ _−_ log _p_ ( _y|x, θ_ )] _. As the number of training episodes N →∞, the following holds with probability at least_ 1 _−δ for arbitrary small δ >_ 0 _:_ 


![](raw/papers/2_LLM_NLP/LLM_paper_list/few-shot/iclr_2024_A_Hierarchical_Bayesian_Model_for_Few-Shot_Meta_Learning_images/iclr_2024_A_Hierarchical_Bayesian_Model_for_Few-Shot_Meta_Learning.pdf-0006-10.png)


**Theorem 5.2** (Bound derived from regression analysis) **.** _Let d_[2] _H_[(] _[P][θ] i[, P][ i]_[)] _[ be the expected squared] Hellinger distance between the true distribution P[i]_ ( _y|x_ ) _and model’s Pθi_ ( _y|x_ ) _for task i. As the number of training episodes N →∞, the following holds with high probability:_ 


![](raw/papers/2_LLM_NLP/LLM_paper_list/few-shot/iclr_2024_A_Hierarchical_Bayesian_Model_for_Few-Shot_Meta_Learning_images/iclr_2024_A_Hierarchical_Bayesian_Model_for_Few-Shot_Meta_Learning.pdf-0006-12.png)


_where λ[∗]_ = E _i∼T_ [ _λ[∗] i_[]] _[,][λ][∗] i_[= min] _[θ][∈]_[Θ] _[ ||]_[E] _[θ]_[[] _[y][|·]_[]] _[ −]_[E] _[i]_[[] _[y][|·]_[]] _[||] ∞_[2] _[is the lowest possible regression error] within_ Θ _, and rn, ϵn are decreasing sequences vanishing to_ 0 _as n increases._ 

## 6 RELATED WORK 

Due to limited space it is infeasible to review all general FSL and meta learning algorithms here. We refer the readers to (Hospedales et al., 2022; Wang et al., 2020a), the excellent comprehensive surveys on the latest techniques. We rather focus on discussing recent Bayesian approaches and their relation to ours. Although several Bayesian FSL approaches have been proposed before, most of them dealt with only a small fraction of the network weights (e.g., a readout head alone) as random variables (Garnelo et al., 2018; Kim et al., 2019; Requeima et al., 2019; Gordon et al., 2019; Patacchiola et al., 2020; Zhang et al., 2021). This considerably limits the benefits from uncertainty modeling of full network parameters. 

6 

Published as a conference paper at ICLR 2024 

Table 2: Classification accuracies with standard backbones on _**mini**_ **ImageNet** and _**tiered**_ **ImageNet** . 

|(a)_mini_ImageNet<br>Model<br>Backbone<br>1-Shot<br>5-Shot<br>AM3 (Xing et al., 2019)<br>ResNet12 65_._21_±_0_._49 75_._20_±_0_._36<br>RelationNet2 (Zhang et al., 2020) ResNet12 63_._92_±_0_._98 77_._15_±_0_._59<br>MetaOpt (Lee et al., 2019)<br>ResNet12 64_._09_±_0_._62 80_._00_±_0_._45<br>SimpleShot (Wang et al., 2019)<br>ResNet18 62_._85_±_0_._20 80_._02_±_0_._14<br>S2M2 (Mangla et al., 2020)<br>ResNet18 64_._06_±_0_._18 80_._58_±_0_._12<br>MetaQDA (Zhang et al., 2021)<br>ResNet18 65_._12_±_0_._66 80_._98_±_0_._75<br>**NIW-Meta(Ours)**<br>ResNet18 **65****_._49****_±_0****_._56** **81****_._71****_±_0****_._17**<br>SimpleShot<br>WRN28-10 63_._50_±_0_._20 80_._33_±_0_._14<br>S2M2<br>WRN28-10 64_._93_±_0_._18 83_._18_±_0_._22<br>MetaQDA<br>WRN28-10 67_._83_±_0_._64 84_._28_±_0_._69<br>**NIW-Meta (Ours)**<br>WRN28-10 **68****_._54****_±_0****_._26** **84****_._81****_±_0****_._28**|(b)_tiered_ImageNet<br>Model<br>Backbone<br>1-Shot<br>5-Shot<br>TapNet (Yoon et al., 2019) ResNet12 63_._08_±_0_._15 80_._26_±_0_._12<br>RelationNet2<br>ResNet12 68_._58_±_0_._63 80_._65_±_0_._91<br>MetaOpt<br>ResNet12 65_._81_±_0_._74 81_._75_±_0_._53<br>SimpleShot<br>ResNet18 69_._09_±_0_._22 84_._58_±_0_._16<br>MetaQDA<br>ResNet18 69_._97_±_0_._52 85_._51_±_0_._58<br>**NIW-Meta(Ours)**<br>ResNet18 **70****_._52****_±_0****_._19** **85****_._83****_±_0****_._17**<br>LEO (Rusu et al., 2019) WRN28-10 66_._33_±_0_._05 81_._44_±_0_._09<br>SimpleShot<br>WRN28-10 69_._75_±_0_._20 85_._31_±_0_._15<br>S2M2<br>WRN28-10 73_._71_±_0_._22 88_._59_±_0_._14<br>MetaQDA<br>WRN28-10 74_._33_±_0_._65 89_._56_±_0_._79<br>**NIW-Meta (Ours)**<br>WRN28-10 **74****_._59****_±_0****_._33** **89****_._76****_±_0****_._23**|
|---|---|



Bayesian approaches to MAML (Finn et al., 2018; Yoon et al., 2018; Ravi & Beatson, 2019; Nguyen et al., 2020) are popular probabilistic extensions of the gradient-based adaptation in MAML (Finn et al., 2017) with known theoretical support (Chen & Chen, 2022). However, they are weak in several aspects to be considered as principled Bayesian methods. For instance, Probabilistic MAML (PMAML) (Finn et al., 2018; Grant et al., 2018) has a similar hierarchical graphical model structure as ours, but their learning algorithm considerably deviates from the original variational inference objective. Unlike the original derivation of the KL term measuring the divergence between the posterior and prior on the task-specific variable _θi_ , namely E _q_ ( _ϕ_ )[KL( _qi_ ( _θi|ϕ_ ) _||p_ ( _θi|ϕ_ ))] as in (5), in PMAML they measure the divergence on the global variable _ϕ_ , aiming to align the two adapted models, one from the support data only _q_ ( _ϕ|Si_ ) and the other from both support and query _q_ ( _ϕ|Si, Qi_ ). VAMPIRE (Nguyen et al., 2020) incorporates uncertainty modeling to MAML by extending MAML’s point estimate to a distributional one that is learned by variational inference. However, it inherits all computational overheads from MAML, hindering scalability. The BMAML (Yoon et al., 2018) is not a hierarchical Bayesian model, but aims to replace MAML’s gradient-based _deterministic_ adaptation steps by the _stochastic_ counterpart using the samples (called particles) from _p_ ( _θi|Si_ ), thus adopting stochastic ensemble-based adaptation steps. If a single particle is used, it reduces exactly to MAML. Thus existing Bayesian approaches are not directly related to our hierarchical Bayesian perspective. A related but different line of research studied Bayesian neural processes (Volpp et al., 2023; 2021; Qi Wang, 2020; Garnelo et al., 2018) by treating the support set embedding as random variates. 

## 7 EVALUATION 

## 7.1 FEW-SHOT CLASSIFICATION 

**Standard benchmarks with ResNet backbones.** For standard benchmark comparison using the popular ResNet backbones, ResNet-18 (He et al., 2016) and WideResNet (Zagoruyko & Komodakis, 2016), we test our method on: _mini_ Imagenet and _tiered_ ImageNet (Table 2). We follow the standard protocols (details of experimental settings in Appendix D). Our NIW-Meta exhibits consistent improvement over the SOTAs for different settings in support set size and backbones. 

## **Large-scale ViT backbones.** 

We also test our method on large-scale (pretrained) ViT backbones DINO-small (Dino/s) and DINO-base (DINO/b) (Caron et al., 2021), similarly as the setup in (Hu et al., 2022). In Table 3 we report results on the three benchmarks: _mini_ Imagenet, CIFAR- 

Table 3: Classification accuracies with large-scale ViT backbones. 

|Model<br>Backbone<br>/ Pretrain|_mini_ImageNet<br>CIFAR-FS<br>_tiered_ImageNet|
|---|---|
||1-shot<br>5-shot<br>1-shot<br>5-shot<br>1-shot<br>5-shot|
|ProtoNet<br>DINO/s<br>MetaOpt<br>DINO/s<br>MetaQDA<br>DINO/s<br>**NIW-Meta**<br>DINO/s|93_._1_±_0_._12 98_._0_±_0_._14 81_._1_±_0_._29 92_._5_±_0_._13 89_._0_±_0_._11 95_._8_±_0_._09<br>92_._2_±_0_._22 97_._8_±_0_._16 70_._2_±_0_._22 84_._1_±_0_._27 87_._5_±_0_._25 94_._7_±_0_._20<br>92_._0_±_0_._31 97_._0_±_0_._18 77_._2_±_0_._34 90_._1_±_0_._18 87_._8_±_0_._27 95_._6_±_0_._16<br>**93****_._4****_±_0****_._17** **98****_._2****_±_0****_._15** **82****_._8****_±_0****_._26** **92****_._9****_±_0****_._11** **89****_._3****_±_0****_._16** **96****_._0****_±_0****_._14**|
|ProtoNet<br>DINO/b<br>MetaOpt<br>DINO/b<br>MetaQDA<br>DINO/b<br>**NIW-Meta**<br>DINO/b|95_._3_±_0_._13 98_._4_±_0_._12 84_._3_±_0_._19 92_._2_±_0_._13 91_._2_±_0_._15 96_._5_±_0_._10<br>94_._4_±_0_._19 98_._4_±_0_._16 72_._0_±_0_._29 86_._2_±_0_._18 89_._5_±_0_._27 95_._7_±_0_._15<br>94_._7_±_0_._21 **98****_._7****_±_0****_._14** 80_._9_±_0_._31 **93****_._8****_±_0****_._15** 89_._7_±_0_._21 96_._5_±_0_._07<br>**95****_._5****_±_0****_._15** **98****_._7****_±_0****_._12** **84****_._7****_±_0****_._13** 93_._2_±_0_._17 **91****_._4****_±_0****_._21** **96****_._7****_±_0****_._11**|



FS, and _tiered_ ImageNet. Our NIW-Meta adopts the same NCC head as ProtoNet after the ViT feature extractor. As claimed in (Hu et al., 2022), using the pretrained feature extractor and further finetuning it significantly boost the performance of few-shot learning algorithms including ours. Among the competing methods, our approach yields the best accuracy for most cases. In particular, compared to the shallow Bayesian MetaQDA (Zhang et al., 2021), treating all network weights as random variates in our model turns out to be more effective than the readout parameters alone. 

**Set-based adaptation backbones.** We also conduct experiments using the set-based adaptation architecture called FEAT introduced in (Ye et al., 2020). The network is tailored for 

Table 4: FEAT vs. our method. Classifcation accuracies. 

|Model|_mini_ImageNet<br>_tiered_ImageNet|
|---|---|
||1-shot<br>5-shot<br>1-shot<br>5-shot|
|FEAT<br>**NIW-Meta (Ours)**|66_._78<br>82_._05<br>70_._80_±_0_._23<br>84_._79_±_0_._16<br>**66****_._91****_±_0****_._10**<br>**82****_._28****_±_0****_._15**<br>**70****_._93****_±_0****_._27**<br>**85****_._20****_±_0****_._19**|



7 

Published as a conference paper at ICLR 2024 

few-shot adaptation, namely _y[Q]_ = _G_ ( _x[Q] , S_ ; _θ_ ) where the network _G_ takes the entire support set _S_ and query image _x[Q]_ as input. Note that our NIW-Meta can incorporate any network architecture, even the set-based one like FEAT. As shown in Table 4, the Bayesian treatment leads to further improvement over (Ye et al., 2020) with this set-based architecture. 

**Error calibration.** Bayesian models are Table 5: ECEs on _mini_ ImageNet. “ECE+TS” indicates known to be better calibrated than determinextra tuning of the temperature hyperparameter. istic counterparts. We measure the _expected_ Model Backbone ECE ECE+TS _calibration errors_ (ECE) (Guo et al., 2017) 1-shot 5-shot 1-shot 5-shot to judge how well the prediction accuracy Linear classifierSimpleShot Conv-4Conv-4 338 _.._ 5445 457 _.._ 4881 33 _.._ 8256 32 _.._ 3588 and the prediction confidence are aligned – MetaQDA-MAP Conv-4 8 _._ 03 5 _._ 27 2 _._ 75 0 _._ 89 we partition test instances into _ECE_ =[�] _[B] b_ =1 _NNb[|][acc]_[(] _[b]_[)] _[−][conf] B_ bins along[(] _[b]_[)] _[|]_[ where] **NIW-Meta** MetaQDA-FBSimpleShot **(Ours)** WRN-28-10Conv-4Conv-4 39 **2** 4 _**.** .._ **68** 3256 55 **1** 2 _**.** .._ **88** 9268 **1** 42 _**.** .._ **47** 3305 **0** 01 _**.** .._ **32** 4580 the model’s prediction confidence scores, S2M2+Linear WRN-28-10 33 _._ 23 36 _._ 84 4 _._ 93 2 _._ 31 and _conf_ ( _b_ ), _acc_ ( _b_ ) are the average conMetaQDA-MAP WRN-28-10 31 _._ 17 17 _._ 37 3 _._ 94 0 _._ 94 fidence and accuracy for the _b_ -th bin, reMetaQDA-FB WRN-28-10 30 _._ 68 15 _._ 86 2 _._ 71 0 _._ 74 spectively. The results on _mini_ ImageNet **NIW-Meta (Ours)** WRN-28-10 **10** _**.**_ **79 7** _**.**_ **11 2** _**.**_ **03 0** _**.**_ **65** are shown in Table 5. We used 20 bins and optionally performed the temperature search on validation sets, similarly as (Zhang et al., 2021). Again, Bayesian inference of whole network weights in our NIW-Meta leads to a far better calibrated model than the shallow Meta-QDA (Zhang et al., 2021). 

## 7.2 FEW-SHOT REGRESSION 

**Sine-Line dataset (Finn et al., 2018).** The 1 _D_ ( _x, y_ ) data pairs are generated by randomly selecting either linear or sine curves with different scales/slopes/frequencies/phases. For the episodic few-shot learning setup, we follow the standard Table 6: Sine-Line results. PMAML w/ protocol: each episode is comprised of _k_ = 5-shot support 5 inner steps incurred numerical errors. (regarded as a task).and 45 query samples randomly drawn from a random curveTo deal with real-valued targets, we RidgeNetModel Mean s0 _._ q8210uared error R-ECEN/A adopt the so-called **RidgeNet** , which has a parameter-free MAML (1-step) 0 _._ 8206 N/A readout head derived from the support data via (closedMAML (5-step) 0 _._ 8309 N/A form) estimation of the linear coefficient matrix using the PMAML (1-step) 0 _._ 9160 0 _._ 2666 **NIW-Meta (Ours) 0** _**.**_ **7822 0** _**.**_ **1728** ridge regression (the L2 regularisation coefficient _λ_ =0 _._ 1). 

It is analogous to the ProtoNet (Snell et al., 2017) in classification which has a parameter-free head derived from NCC on support data. A similar model was introduced in (Bertinetto et al., 2019) but mainly repurposed for classification. We find that RidgeNet leads to much more accurate prediction than the conventional trainable linear head. For instance, the test errors are: RidgeNet = 0 _._ 82 vs. MAML with linear head = 1 _._ 86. Furthermore, we adopt the ridge head in other models as well, such as MAML, PMAML (Finn et al., 2018), and our NIW-Meta. See Table 6 for the mean squared errors contrasting our NIW-Meta against competing meta learning methods. The table also contains the regression-ECE (R-ECE) calibration errors[2] . Clearly our model is calibrated the best. 

**Object pose estimation on ShapeNet datasets.** We consider the recent few-shot regression benchmarks (Gao et al., 2022; Yin et al., 2020) which introduced four datasets for object pose estimation: _Pascal-1D_ , _ShapeNet-1D_ , _ShapeNet-2D_ and _Distractor_ . In all datasets[3] , the main goal is to estimate the pose (positions in pixel and/or rotation angles) of the target object in an image. Each episode is specified by: i) selecting a target object randomly from a pool of objects with different object categories, and ii) rendering the same object in an image with several different random poses (position/rotation) to generate data instances. There are _k_ support samples (input images and target pose labels) and _kq_ query samples for each episode. For ShapeNet-1D, for instance, _k_ is randomly chosen from 3 to 15 while _kq_ =15. Except Pascal-1D, some object categories are dedicated solely for 

> 2The definition of the R-ECE is quite different from that of the classification ECE in Sec. 7.1. We follow the notion of _goodness of cumulative distribution matching_ used in (Tran et al., 2020; Cui et al., 2020). Specifically, denoting by _Q_[ˆ] _p_ ( _x_ ) the _p_ -th quantile of the predicted distribution _p_ ˆ( _y|x_ ), we measure the deviation ofthat by definition we can only measure R-ECE for models with _ptrue_ ( _y ≤ Q_[ˆ] _p_ ( _x_ ) _|x_ ) from _p_ by absolute difference. So it is 0 _probabilistic_ for the ideal case output _p_ ˆ _p_ (ˆ _y_ ( _y|x|x_ ) =). _ptrue_ ( _y|x_ ). Note 

3Pascal-1D and ShapeNet-1D are relatively easier datasets than the rest two as we have uniform noise-free backgrounds. To make the few-shot learning problem more challenging, ShapeNet-2D and Distractor datasets further introduce random (real-world) background images and/or so called the _distractors_ which are objects randomly drawn and rendered that have nothing to do with the target pose to estimate. 

8 

Published as a conference paper at ICLR 2024 

Table 7: Pose estimation results. Mean squared errors in rotation angle differences (Pascal-1D and ShapeNet-1D), quaternion differences _×_ 10 _[−]_[2] (ShapeNet-2D) and pixel errors (Distractor). The dataset-wise different augmentation schemes are shown in the parentheses. 

|Model<br>Pascal-1D<br>(TA)|ShapeNet-1D(TA+DA)<br>ShapeNet-2D(TA+DA+DR)<br>Distractor(DA)|
|---|---|
||Intra-category Cross-category Intra-category Cross-category Intra-category Cross-category|
|MAML<br>1_._02_±_0_._06<br>CNP (Garnelo et al., 2018)<br>1_._98_±_0_._22<br>CNP+BA (Volpp et al., 2021)<br>_−_<br>CNP+FCL (Gao et al., 2022)<br>_−_<br>ANP (Kim et al., 2019)<br>1_._36_±_0_._25<br>ANP+FCL(Gao et al.,2022)<br>_−_|17_._96<br>18_._79<br>_−_<br>_−_<br>_−_<br>_−_<br> 7_._66_±_0_._18<br>8_._66_±_0_._19<br>14_._20_±_0_._06<br>13_._56_±_0_._28<br>2_._45<br>3_._75<br>_−_<br>_−_<br>14_._16_±_0_._08<br>13_._56_±_0_._18<br>2_._44<br>3_._97<br>_−_<br>_−_<br>_−_<br>_−_<br>2_._00<br>3_._05<br> 5_._81_±_0_._23<br>6_._23_±_0_._12<br>14_._12_±_0_._14<br>13_._59_±_0_._10<br>2_._65<br>4_._08<br>_−_<br>_−_<br>14_._01_±_0_._09<br>13_._32_±_0_._18<br>_−_<br>_−_|
|NIW-Meta w/ C+R<br>**0****_._89** **_±_** **0****_._06**<br>NIW-Meta w/ CNP<br>0_._94_±_0_._15<br>NIW-Meta w/ ANP<br>0_._95_±_0_._09|5_._62_±_0_._38<br>6_._57_±_0_._39<br>21_._25_±_0_._76<br>20_._82_±_0_._43<br>8_._90_±_0_._26<br>17_._31_±_0_._38<br> 5_._74_±_0_._17<br>6_._91_±_0_._18<br>13_._86_±_0_._20<br>13_._04_±_0_._13<br>**1****_._80****_±_0****_._01**<br>**2****_._94****_±_0****_._14**<br> **5****_._47** **_±_** **0****_._12**<br>**6****_._06** **_±_** **0****_._18**<br>**13****_._74****_±_0****_._30**<br>**12****_._95****_±_0****_._48**<br>3_._10_±_0_._48<br>5_._20_±_0_._88|



meta testing and not revealed during training, thus yielding two different test scenarios: _intra-category_ and _cross-category_ , in which the test object categories are seen and unseen, respectively. 

In (Gao et al., 2022), they test different augmentation strategies in their baselines: conventional _data augmentation_ on input images (denoted by DA), _task augmentation_ (TA) (Rajendran et al., 2020) which adds random noise to the target labels to help reducing the memorisation issue (Yin et al., 2020), and _domain randomisation_ (DR) (Tobin et al., 2017) which randomly generates background images during training. Among several possible combinations reported in (Gao et al., 2022), we follow the strategies that perform the best. For the target error metrics (e.g., position Euclidean distances in pixels for Distractor, rotation angle differences for ShapeNet-1D), we follow the metrics used in (Gao et al., 2022). For instance, the quaternion metric may sound reasonable in ShapeNet-2D due to the non-uniform, non-symmetric structures that reside in the target space (3D rotation angles). 

The results are summarised in Table 7. In (Gao et al., 2022), they have shown that the set-based backbone networks, especially the Conditional Neural Process (CNP) (Garnelo et al., 2018) and Attentive Neural Process (ANP) (Kim et al., 2019) outperform the conventional architectures of the conv-net feature extractor with the linear head that are adapted by MAML (Finn et al., 2017) (except for the Pascal-1D case). Motivated by this, we adopt the same set-based CNP/ANP architectures within our NIW-Meta. In addition, we also test the ridge-head model with the conv-net feature extractor (denoted by **C** + **R** ). Two additional competing models contrasted here are: the Bayesian context aggregation in CNP (CNP+BA) (Volpp et al., 2021) and the use of the functional contrastive learning loss as extra regularisation (FCL) (Gao et al., 2022). 

For Pascal-1D and ShapeNet-1D, there is a dataset regime where MAML clearly outperforms (Pascal1D) and underperforms (ShapeNet-1D) the CNP/ANP architectures. Very promisingly, our NIW-Meta consistently performs the best for both datasets, regardless of the choice of the architectures: not just CNP/ANP but also conv-net feature extractor + ridge head (C+R). For ShapeNet-2D and Distractor where MAML is not reported due to the known computational issues and poor performance, our NIW-Meta still exhibits the best test performance with CNP/ANP architectures. Unfortunately, the conv-net + ridge head (C+R) did not work well, and our conjecture is that the presence of heavy noise and distractors in the input data requires more sophisticated modeling of interaction/relation among the input instances, as is mainly aimed (and successfully done) by CNP/ANP. 

**Computational complexity, running time and memory footprint.** We have analysed the computational complexity of our NIW-Meta compared to the simple feed-forward workflows (e.g., ProtoNet). Our method incurs only constant-factor overhead compared to the minimal-cost ProtoNet, as summarised in Table 8 in Appendix E. Also in Fig. 5 therein, we also report the memory footprints and running times of MAML and our NIW-Meta on real datasets, which show that NIW-Meta has far lower memory requirement than MAML. MAML suffers from heavy use of memory and thecomputational overhead of keeping track of a large computational graph for inner gradient descent steps. Our NIW-Meta has a much more efficient strategy of local episodic optimisation that is linked to global parameters, without storing the full optimisation trace. Please see all details in Appendix E. 

## 8 CONCLUSION 

We have proposed a new hierarchical Bayesian perspective to the episodic FSL problem. By having a higher-level task-agnostic random variate and episode-wise task-specific variables, we formulate a principled Bayesian inference view of the FSL problem with a large number of tasks (evidence). The effectiveness of our approach has been verified empirically in terms of both prediction accuracy and calibration, on a wide range of classification/regression tasks with complex backbones including ViT and set-based adaptation networks. 

9 

Published as a conference paper at ICLR 2024 

## REFERENCES 

- Jincheng Bai, Qifan Song, and Guang Cheng. Efficient Variational Inference for Sparse Deep Learning with Theoretical Guarantee. In _Advances in Neural Information Processing Systems_ , 2020. 

- Luca Bertinetto, Joao F Henriques, Philip HS Torr, and Andrea Vedaldi. Meta-learning with differentiable closed-form solvers. In _International Conference on Learning Representations_ , 2019. 

Christopher M. Bishop. _Pattern Recognition and Machine Learning_ . Springer, 2006. 

- S. Boucheron, G. Lugosi, and P. Massart. _Concentration Inequalities: A Nonasymptotic Theory of Independence_ . Oxford University Press, 2013. 

- Michael Braun and Jon McAuliffe. Variational inference for large-scale models of discrete choice. _arXiv preprint arXiv:0712.2526_ , 2008. 

- Mathilde Caron, Hugo Touvron, Ishan Misra, Hervé Jégou, Julien Mairal, Piotr Bojanowski, and Armand Joulin. Emerging properties in self-supervised vision transformers. In _Proceedings of the International Conference on Computer Vision (ICCV)_ , 2021. 

- Lisha Chen and Tianyi Chen. Is Bayesian Model-Agnostic Meta Learning Better than Model-Agnostic Meta Learning, Provably?, 2022. AI and Statistics (AISTATS). 

- Peng Cui, Wenbo Hu, , and Jun Zhu. Calibrated reliable regression using maximum mean discrepancy. In _Advances in Neural Information Processing Systems_ , 2020. 

- Alexey Dosovitskiy, Lucas Beyer, Alexander Kolesnikov, Dirk Weissenborn, Xiaohua Zhai, Thomas Unterthiner, Mostafa Dehghani, Matthias Minderer, Georg Heigold, Sylvain Gelly, Jakob Uszkoreit, and Neil Houlsby. An image is worth 16x16 words: Transformers for image recognition at scale. _ICLR_ , 2021. 

- Chelsea Finn, Pieter Abbeel, and Sergey Levine. Model-agnostic meta-learning for fast adaptation of deep networks. In _International Conference on Machine Learning_ , 2017. 

- Chelsea Finn, Kelvin Xu, and Sergey Levine. Probabilistic Model-Agnostic Meta-Learning. In _Advances in Neural Information Processing Systems_ , 2018. 

- Ning Gao, Hanna Ziesche, Ngo Anh Vien, Michael Volpp, and Gerhard Neumann. What matters for meta-learning vision regression tasks? In _Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)_ , pp. 14776–14786, June 2022. 

- Marta Garnelo, Dan Rosenbaum, Chris J. Maddison, Tiago Ramalho, David Saxton, Murray Shanahan, Yee Whye Teh, Danilo J. Rezende, and S. M. Ali Eslami. Conditional Neural Processes. In _International Conference on Machine Learning_ , 2018. 

- Andrew Gelman, John B. Carlin, Hal S. Stern, and Donald B. Rubin. _Bayesian Data Analysis_ . Texts in statistical science. Chapman & Hall / CRC, 2nd edition, 2003. 

- Jonathan Gordon, John Bronskill, Matthias Bauer, Sebastian Nowozin, and Richard Turner. Metalearning probabilistic inference for prediction. In _International Conference on Learning Representations_ , 2019. URL https://openreview.net/forum?id=HkxStoC5F7. 

- Erin Grant, Chelsea Finn, Sergey Levine, Trevor Darrell, and Tom Griffiths. Recasting gradient-based meta-learning as hierarchical bayes. In _ICLR_ , 2018. 

- Edward Grefenstette, Brandon Amos, Denis Yarats, Phu Mon Htut, Artem Molchanov, Franziska Meier, Douwe Kiela, Kyunghyun Cho, and Soumith Chintala. Generalized inner loop meta-learning. _arXiv preprint arXiv:1910.01727_ , 2019. 

- Chuan Guo, Geoff Pleiss, Yu Sun, and Kilian Q Weinberger. On calibration of modern neural networks. In _International Conference on Machine Learning_ , 2017. 

- K. He, X. Zhang, S. Ren, and J. Sun. Deep residual learning for image recognition. In _IEEE/CVF Conference on Computer Vision and Pattern Recognition_ , 2016. 

10 

Published as a conference paper at ICLR 2024 

Timothy Hospedales, Antreas Antoniou, Paul Micaelli, and Amos Storkey. Meta-learning in neural networks: A survey. _IEEE Transactions on Pattern Analysis and Machine Intelligence_ , 44: 5149–5169, 2022. 

- Shell Xu Hu, Da Li, Jan Stühmer, Minyoung Kim, and Timothy M. Hospedales. Pushing the limits of simple pipelines for few-shot learning: External data and fine-tuning make a difference. In _CVPR_ , 2022. 

- Pavel Izmailov, Dmitrii Podoprikhin, Timur Garipov, Dmitry Vetrov, and Andrew Gordon Wilson. Averaging weights leads to wider optima and better generalization. In _Uncertainty in Artificial Intelligence_ , 2018. 

- Hyunjik Kim, Andriy Mnih, Jonathan Schwarz, Marta Garnelo, Ali Eslami, Dan Rosenbaum, Oriol Vinyals, and Yee Whye Teh. Attentive Neural Processes. In _International Conference on Learning Representations_ , 2019. 

- Brenden M. Lake, Ruslan Salakhutdinov, and Joshua B. Tenenbaum. Human-level concept learning through probabilistic program induction. _Science_ , 2015. 

- John Langford and Rich Caruana. (Not) Bounding the True Error. In _Advances in Neural Information Processing Systems_ , 2001. 

- Kwonjoon Lee, Subhransu Maji, Avinash Ravichandran, and Stefano Soatto. Meta-learning with differentiable convex optimization. In _IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)_ , 2019. 

- David MacKay. _Information Theory, Inference, and Learning Algorithms_ . Cambridge University Press, 2003. 

- Wesley Maddox, Timur Garipov, Pavel Izmailov, Dmitry Vetrov, and Andrew Gordon Wilson. A Simple Baseline for Bayesian Uncertainty in Deep Learning. _arXiv preprint arXiv:1902.02476_ , 2019. 

- Puneet Mangla, Nupur Kumari, Abhishek Sinha, Mayank Singh, Balaji Krishnamurthy, and Vineeth N Balasubramanian. Charting the right manifold: Manifold mixup for few-shot learning. In _IEEE/CVF Winter Conference on Applications of Computer Vision (WACV)_ , 2020. 

Andreas Maurer. A Note on the PAC Bayesian Theorem. _arXiv preprint arXiv:0411099_ , 2004. 

David McAllester. Some pac-bayesian theorems. _Machine Learning_ , 37:355–363, 1999. 

- Kevin P. Murphy. _Probabilistic Machine Learning: An introduction_ . MIT Press, 2022. URL probml.ai. 

- Cuong Nguyen, Thanh-Toan Do, and Gustavo Carneiro. Uncertainty in model-agnostic meta-learning using variational inference. In _Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision_ , pp. 3090–3100, 2020. 

- Alex Nichol, Joshua Achiam, and John Schulman. On First-Order Meta-Learning Algorithms. In _arXiv preprint arXiv:1803.02999_ , 2018. 

- Adam Paszke, Sam Gross, Soumith Chintala, Gregory Chanan, Edward Yang, Zachary DeVito, Zeming Lin, Alban Desmaison, Luca Antiga, and Adam Lerer. Automatic differentiation in pytorch. 2017. 

- Massimiliano Patacchiola, Jack Turner, Elliot J. Crowley, and Amos Storkey. Bayesian meta-learning for the few-shot setting via deep kernels. In _Advances in Neural Information Processing Systems_ , 2020. 

- D. Pati, A. Bhattacharya, and Y. Yang. On the Statistical Optimality of Variational Bayes, 2018. AI and Statistics (AISTATS). 

- Herke van Hoof Qi Wang. Doubly Stochastic Variational Inference for Neural Processes with Hierarchical Latent Variables. In _International Conference on Machine Learning_ , 2020. 

11 

Published as a conference paper at ICLR 2024 

Janarthanan Rajendran, Alex Irpan, and Eric Jang. Meta-Learning Requires Meta-Augmentation. In _Advances in Neural Information Processing Systems_ , 2020. 

- Sachin Ravi and Alex Beatson. Amortized Bayesian meta-learning. In _International Conference on Learning Representations_ , 2019. 

- James Requeima, Jonathan Gordon, John Bronskill, Sebastian Nowozin, and Richard E. Turner. Fast and Flexible Multi-Task Classification Using Conditional Neural Adaptive Processes. In _Advances in Neural Information Processing Systems_ , 2019. 

- Omar Rivasplata, Vikram M Tankasali, and Csaba Szepesvari. PAC-Bayes with Backprop. _arXiv preprint arXiv:1908.07380_ , 2019. 

- Andrei A. Rusu, Dushyant Rao, Jakub Sygnowski, Oriol Vinyals, Razvan Pascanu, Simon Osindero, and Raia Hadsell. Meta-learning with latent embedding optimization. In _International Conference on Learning Representations_ , 2019. 

Matthias Seeger. PAC-Bayesian Generalization Error Bounds for Gaussian Process Classification. _Journal of Machine Learning Research_ , 3:233–269, 2002. 

Jake Snell, Kevin Swersky, and Richard S. Zemel. Prototypical networks for few-shot learning. _CoRR_ , abs/1703.05175, 2017. URL http://arxiv.org/abs/1703.05175. 

- Matthew Tancik, Ben Mildenhall, Terrance Wang, Divi Schmidt, Pratul P. Srinivasan, Jonathan T. Barron, and Ren Ng. Learned Initializations for Optimizing Coordinate-Based Neural Representations. In _IEEE/CVF Conference on Computer Vision and Pattern Recognition_ , 2021. 

- Niklas Thiemann, Christian Igel, Olivier Wintenberger, and Yevgeny Seldin. A strongly quasiconvex PAC-Bayesian bound. In _International Conference on Algorithmic Learning Theory_ , 2017. 

- Josh Tobin, Rachel Fong, Alex Ray, Jonas Schneider, Wojciech Zaremba, and Pieter Abbeel. Domain randomization for transferring deep neural networks from simulation to the real world. In _2017 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS)_ , pp. 23–30, 2017. doi: 10.1109/IROS.2017.8202133. 

- Kevin Tran, Willie Neiswanger, Junwoong Yoon, Qingyang Zhang, Eric Xing, and Zachary W Ulissi. Methods for comparing uncertainty quantifications for material property predictions. _Machine Learning: Science and Technology_ , 1(2):025006, 2020. 

- Michael Volpp, Fabian Flürenbrock, Lukas Grossberger, Christian Daniel, and Gerhard Neumann. Bayesian Context Aggregation for Neural Processes. In _International Conference on Learning Representations_ , 2021. 

- Michael Volpp, Philipp Dahlinger, Philipp Becker, Christian Daniel, and Gerhard Neumann. Accurate Bayesian Meta-Learning by Accurate Task Posterior Inference. In _International Conference on Learning Representations_ , 2023. 

- Yan Wang, Wei-Lun Chao, Kilian Q. Weinberger, and Laurens van der Maaten. Simpleshot: Revisiting nearestneighbor classification for few-shot learning. In _arXiv preprint arXiv:1911.04623_ , 2019. 

- Yaqing Wang, Quanming Yao, James T. Kwok, and Lionel M. Ni. Generalizing from a few examples: A survey on few-shot learning. _ACM Computing Surveys_ , 53(3):1–34, 2020a. 

Yaqing Wang, Quanming Yao, James T Kwok, and Lionel M Ni. Generalizing from a few examples: A survey on few-shot learning. _ACM Computing Surveys (CSUR)_ , 53(3):1–34, 2020b. 

- Max Welling and Yee Whye Teh. Bayesian Learning via Stochastic Gradient Langevin Dynamics. In _International Conference on Machine Learning_ , 2011. 

- Chen Xing, Negar Rostamzadeh, Boris Oreshkin, and Pedro O. Pinheiro. Adaptive cross-modal few-shot learning. In _Advances in Neural Information Processing Systems_ , 2019. 

- Han-Jia Ye, Hexiang Hu, De-Chuan Zhan, and Fei Sha. Few-shot learning via embedding adaptation with set-to-set functions. In _IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)_ , pp. 8808–8817, 2020. 

12 

Published as a conference paper at ICLR 2024 

Mingzhang Yin, George Tucker, Mingyuan Zhou, Sergey Levine, and Chelsea Finn. Meta-Learning without Memorization. In _International Conference on Learning Representations_ , 2020. 

- Jaesik Yoon, Taesup Kim, Ousmane Dia, Sungwoong Kim, Yoshua Bengio, and Sungjin Ahn. Bayesian Model-Agnostic Meta-Learning. In _Advances in Neural Information Processing Systems_ , 2018. 

- Sung Whan Yoon, Jun Seo, and Jaekyun Moon. TapNet: Neural network augmented with taskadaptive projection for few-shot learning. In _International conference on Machine Learning_ , 2019. 

- S. Zagoruyko and N. Komodakis. Wide residual networks. In _arXiv preprint arXiv:1605.07146_ , 2016. 

- Xueting Zhang, Yuting Qiang, Sung Flood, Yongxin Yang, and Timothy M. Hospedales. RelationNet2: Deep comparison columns for few-shot learning. In _International Joint Conference on Neural Networks (IJCNN)_ , 2020. 

- Xueting Zhang, Debin Meng, Henry Gouk, and Timothy Hospedales. Shallow Bayesian Meta Learning for Real-World Few-Shot Recognition. In _International Conference on Computer Vision_ , 2021. 

13 

Published as a conference paper at ICLR 2024 

## **Appendix** 

## **Table of Contents** 

- Proofs for Generalisation Error Bounds (Sec. A) 

   - Proof for PAC-Bayes- _λ_ Bound (Sec. A.1) 

   - Proof for Regression Analysis Bound (Sec. A.2) 

- Detailed Derivations (Sec. B) 

- Toy Experiment: Why Hierarchical Bayesian Model? (A Detailed Version) (Sec. C) 

- Implementation Details and Experimental Settings (Sec. D) 

- Computational Complexity, Running Time and Memory Footprint (Sec. E) 

- Training Stability and Impact of Number of Training Episodes (Sec. F) 

- Additional Discussions (Sec. G) 

   - Comparison to Bayesian Neural Processes (Sec. G.1) 

   - Justification of Model and Algorithm Choices (Sec. G.2) 

   - Limitations and Future Works (Sec. G.3) 

## A PROOFS FOR GENERALISATION ERROR BOUNDS 

We prove the two theorems Theorem 5.1 and Theorem 5.2 in the main paper that upper-bound the generalisation error of the model that is averaged over the learned posterior _q_ ( _ϕ, θ_ 1: _N_ ). Without loss of generality we assume _|Di|_ = _n_ for all episodes _i_ . We let ( _q[∗]_ ( _ϕ_ ) _, {qi[∗]_[(] _[θ][i]_[)] _[}][N] i_ =1[)][ be the optimal] solution of (6). In these theorems we often make the assumption of _N →∞_ , that the number of training episodes tends to infinity, mainly for mathematical convenience. This assumption may not be true for practical situations, however, the theorems can still be applicable with _approximate_ guarantees where we provide several justifications for this in Sec. F. 

## A.1 PROOF FOR PAC-BAYES- _λ_ BOUND 

First, Theorem 5.1, reiterated below as Theorem A.1, relates the generalisation error to the ultimate ELBO loss (6) that we minimised in our algorithm. 

**Theorem A.1** (PAC-Bayes- _λ_ bound) **.** _Let Ri_ ( _θ_ ) _be the generalisation error of model θ for the task i, more specifically, Ri_ ( _θ_ ) = E( _x,y_ ) _∼Ti_ [ _−_ log _p_ ( _y|x, θ_ )] _with the assumption of_ [0 _,_ 1] _-bounded errors. As the number of training episodes N →∞, the following holds with probability at least_ 1 _−δ for arbitrary small δ >_ 0 _:_ 


![](raw/papers/2_LLM_NLP/LLM_paper_list/few-shot/iclr_2024_A_Hierarchical_Bayesian_Model_for_Few-Shot_Meta_Learning_images/iclr_2024_A_Hierarchical_Bayesian_Model_for_Few-Shot_Meta_Learning.pdf-0014-20.png)


_where ϵ[∗] is the optimal value of (6)._ 

_Proof._ We utilise the recent PAC-Bayes- _λ_ bound (Thiemann et al., 2017; Rivasplata et al., 2019), a variant of the traditional PAC-Bayes bounds (McAllester, 1999; Langford & Caruana, 2001; Seeger, 2002; Maurer, 2004). It states that for any _λ ∈_ (0 _,_ 2), the following holds with probability at least 1 _−δ_ : 


![](raw/papers/2_LLM_NLP/LLM_paper_list/few-shot/iclr_2024_A_Hierarchical_Bayesian_Model_for_Few-Shot_Meta_Learning_images/iclr_2024_A_Hierarchical_Bayesian_Model_for_Few-Shot_Meta_Learning.pdf-0014-23.png)


where _β_ represents all model parameters (random variables), _R_ ( _β_ ) is the generalisation error/loss for a given model _β_ , and _R_[ˆ] _m_ ( _β_ ) is the empirical error/loss on the training data of size _m_ . It holds for any data- _independent_ (e.g., prior) distribution _p_ ( _β_ ) and any distribution (possibly data-dependent, e.g., posterior) _q_ ( _β_ ). 

14 

Published as a conference paper at ICLR 2024 

Now, with _N →∞_ , we rewrite (6) in an equivalent form as follows: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/few-shot/iclr_2024_A_Hierarchical_Bayesian_Model_for_Few-Shot_Meta_Learning_images/iclr_2024_A_Hierarchical_Bayesian_Model_for_Few-Shot_Meta_Learning.pdf-0015-02.png)


_N →∞_ Then we set _β_ := _{ϕ, θ_ 1: _N }_ , _q_ ( _β_ ) := _q_ ( _ϕ_ )[�] _i[q][i]_[(] _[θ][i]_[)][, and] _[ p]_[(] _[β]_[)][:=] _[p]_[(] _[ϕ]_[)][�] _i[p]_[(] _[θ][i][|][ϕ]_[)][.][We also define] the generalisation loss and the empirical loss as follows: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/few-shot/iclr_2024_A_Hierarchical_Bayesian_Model_for_Few-Shot_Meta_Learning_images/iclr_2024_A_Hierarchical_Bayesian_Model_for_Few-Shot_Meta_Learning.pdf-0015-04.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/few-shot/iclr_2024_A_Hierarchical_Bayesian_Model_for_Few-Shot_Meta_Learning_images/iclr_2024_A_Hierarchical_Bayesian_Model_for_Few-Shot_Meta_Learning.pdf-0015-05.png)


Taking _N →∞_ in (21) makes i) the LHS become E _i∼T_ E _qi_ ( _θi_ )[ _Ri_ ( _θi_ )], ii) the complexity term log(2 ~~_√_~~ _nNnN /δ_ ) in the RHS vanish, and iii) the RHS converge to _n_[2] _[Q]_[(] _[L]_[0] _[,][ {][L][i][}] i[∞]_ =1[)][.][That is,] 


![](raw/papers/2_LLM_NLP/LLM_paper_list/few-shot/iclr_2024_A_Hierarchical_Bayesian_Model_for_Few-Shot_Meta_Learning_images/iclr_2024_A_Hierarchical_Bayesian_Model_for_Few-Shot_Meta_Learning.pdf-0015-07.png)


Since (22) holds for _any q_ , we take the minimiser _q[∗]_ of (6), which completes the proof. 

## A.2 PROOF FOR REGRESSION ANALYSIS BOUND 

Theorem 5.2, reiterated below as Theorem A.2 in a more detailed form, is based on the recent regression analysis techniques (Pati et al., 2018; Bai et al., 2020). Before we prove the theorem, we formally state some core assumptions and notations. Let _P[i]_ ( _x, y_ ) be the _true_ data distribution for episode/task _i_ where _i_ = 1 _, . . . , N_ and _N →∞_ . We consider regression-based data modeling, assuming that the target _y_ is real vector-valued ( _y ∈_ R _[S][y]_ ). Also it is assumed that there exists a _true regression function f[i]_ : R _[S][x] →_ R _[S][y]_ for each _i_ , more formally _P[i]_ ( _y|x_ ) = _N_ ( _y_ ; _f[i]_ ( _x_ ) _, σϵ_[2] _[I]_[)][, where] _σϵ_[2][is constant Gaussian output noise variance.] 

For easier analysis we assume that the backbone network is an MLP with _L_ width- _M_ hidden layers, and all activation functions _σ_ ( _·_ ) are Lipschitz continuous with 1. We consider the bounded parameter space, _θ ∈_ Θ = _{θ ∈_ R _[G]_ : _||θ||∞ ≤ B}_ , where _G_ = dim( _θ_ ) and _B_ is the maximal norm bound. Then the prediction (regression) function _fθ_ : R _[S][x] →_ R _[S][y]_ is induced from _θ_ as: _Pθ_ ( _y|x_ ) = _N_ ( _y_ ; _fθ_ ( _x_ ) _, σϵ_[2] _[I]_[)][, where the true noise variance is assumed to be known.][The expressions] E _θ_ [ _·_ ] and E _[i]_ [ _·_ ] refer to the expectations with respect to model’s _Pθ_ and the true _P[i]_ , respectively. The generalisation error measure that we consider is the _expected squared Hellinger distance_ between the true _P[i]_ and the model _Pθ_ , more specifically, 


![](raw/papers/2_LLM_NLP/LLM_paper_list/few-shot/iclr_2024_A_Hierarchical_Bayesian_Model_for_Few-Shot_Meta_Learning_images/iclr_2024_A_Hierarchical_Bayesian_Model_for_Few-Shot_Meta_Learning.pdf-0015-12.png)


Now we state our theorem. 

**Theorem A.2** (Bound derived from regression analysis) **.** _Let d_[2] ( _Pθi, P[i]_ ) _be the expected squared Hellinger distance between the true distribution P[i]_ ( _y|x_ ) _and model’s Pθi_ ( _y|x_ ) _for task/episode i. Then the following holds with high probability:_ 


![](raw/papers/2_LLM_NLP/LLM_paper_list/few-shot/iclr_2024_A_Hierarchical_Bayesian_Model_for_Few-Shot_Meta_Learning_images/iclr_2024_A_Hierarchical_Bayesian_Model_for_Few-Shot_Meta_Learning.pdf-0015-15.png)


15 

Published as a conference paper at ICLR 2024 

_where C• >_ 0 _are some constant, λ[∗]_ = E _i∼T_ [ _λ[∗] i_[]] _[ with][ λ][∗] i_[= min] _[θ][∈]_[Θ][ max] _[x][ ||]_[E] _[θ]_[[] _[y][|][x]_[]] _[ −]_[E] _[i]_[[] _[y][|][x]_[]] _[||]_[2] _[is] the lowest possible regression error within the underlying network_ Θ _, rn_ = _[G] n_ ( _L_ + 1) log _M_ + � log � _Sx_ ~~�~~ _Gn_ �[�] _, and ϵn_ = _[√] rn_ log _[δ]_ ( _n_ ) _for δ >_ 1 _constant._ 

_Proof._ We utilise the Donsker-Varadhan’s (DV) theorem (Boucheron et al., 2013) to relate the variational ELBO objective function to the Hellinger distance. The DV theorem says that the following inequality holds for any distributions _p_ , _q_ and any (bounded) function _h_ ( _z_ ): 


![](raw/papers/2_LLM_NLP/LLM_paper_list/few-shot/iclr_2024_A_Hierarchical_Bayesian_Model_for_Few-Shot_Meta_Learning_images/iclr_2024_A_Hierarchical_Bayesian_Model_for_Few-Shot_Meta_Learning.pdf-0016-03.png)


In our case, we define: _p_ ( _z_ ) := _p_ ( _θi|ϕ_ ), _q_ ( _z_ ) := _qi_ ( _θi_ ), _h_ ( _z_ ) := log _ηi_ ( _θi_ ) with 


![](raw/papers/2_LLM_NLP/LLM_paper_list/few-shot/iclr_2024_A_Hierarchical_Bayesian_Model_for_Few-Shot_Meta_Learning_images/iclr_2024_A_Hierarchical_Bayesian_Model_for_Few-Shot_Meta_Learning.pdf-0016-05.png)


where _ρ_ ( _Pθi_ ( _Di_ ) _, P[i]_ ( _Di_ )) := log _PPθ[i] i_ (( _DDii_ ))[is the log-ratio.][Note that] _[ P]_[(] _[D][i]_[) =] _[ P]_[(] _[Y][i][|][X][i]_[)][.][Plugging] these into (25) leads to the following inequality which holds for any _ϕ_ : 


![](raw/papers/2_LLM_NLP/LLM_paper_list/few-shot/iclr_2024_A_Hierarchical_Bayesian_Model_for_Few-Shot_Meta_Learning_images/iclr_2024_A_Hierarchical_Bayesian_Model_for_Few-Shot_Meta_Learning.pdf-0016-07.png)


We take the expectation with respect to _q_ ( _ϕ_ ), which yields: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/few-shot/iclr_2024_A_Hierarchical_Bayesian_Model_for_Few-Shot_Meta_Learning_images/iclr_2024_A_Hierarchical_Bayesian_Model_for_Few-Shot_Meta_Learning.pdf-0016-09.png)


From the regression theorem (Pati et al., 2018) (Theorem 3.1 therein), it is known that E _s_ ( _θ_ )[ _η_ ( _θ_ )] _≤ e[Cnϵ] n_[2] for any distribution _s_ ( _θ_ ) with high probability. We apply this result to the last term of (28). Summing it over _i_ = 1 _, . . . , N_ leads to: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/few-shot/iclr_2024_A_Hierarchical_Bayesian_Model_for_Few-Shot_Meta_Learning_images/iclr_2024_A_Hierarchical_Bayesian_Model_for_Few-Shot_Meta_Learning.pdf-0016-11.png)


By dividing both sides by _N_ and sending _N →∞_ , we have: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/few-shot/iclr_2024_A_Hierarchical_Bayesian_Model_for_Few-Shot_Meta_Learning_images/iclr_2024_A_Hierarchical_Bayesian_Model_for_Few-Shot_Meta_Learning.pdf-0016-13.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/few-shot/iclr_2024_A_Hierarchical_Bayesian_Model_for_Few-Shot_Meta_Learning_images/iclr_2024_A_Hierarchical_Bayesian_Model_for_Few-Shot_Meta_Learning.pdf-0016-14.png)


As indicated, the right hand side is composed of _−_ ELBO( _q_ ) (the objective function of (6)), the constant log _P[i]_ ( _Di_ ), and the complexity term _Cnϵ_[2] _n_[[.]] 

_n_[[.]] 

The next step is to plug in the optimal _q[∗]_ to have a meaningful upper bound. To this end, we introduce/define _q_ ˜ _i_ ( _θi_ ) and _q_ ˜( _ϕ_ ) as follows: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/few-shot/iclr_2024_A_Hierarchical_Bayesian_Model_for_Few-Shot_Meta_Learning_images/iclr_2024_A_Hierarchical_Bayesian_Model_for_Few-Shot_Meta_Learning.pdf-0016-18.png)


16 

Published as a conference paper at ICLR 2024 

Since ( _{qi[∗]_[(] _[θ][i]_[)] _[}][N] i_ =1 _[, q][∗]_[(] _[ϕ]_[))][is][the][minimiser][of][the][negative][ELBO][(6),][we][clearly][have] _−_ ELBO( _q[∗]_ ) _≤−_ ELBO(˜ _q_ ). We plug _q[∗]_ into (30) and apply this ELBO inequality to have: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/few-shot/iclr_2024_A_Hierarchical_Bayesian_Model_for_Few-Shot_Meta_Learning_images/iclr_2024_A_Hierarchical_Bayesian_Model_for_Few-Shot_Meta_Learning.pdf-0017-02.png)


The second term of the right hand side of (34) is constant (independent of _n_ ) and denoted by _C_[˜] . For the first term of the right hand side, we use the following fact from the proof of Lemma 4.1 in (Bai et al., 2020), which says that with high probability, 


![](raw/papers/2_LLM_NLP/LLM_paper_list/few-shot/iclr_2024_A_Hierarchical_Bayesian_Model_for_Few-Shot_Meta_Learning_images/iclr_2024_A_Hierarchical_Bayesian_Model_for_Few-Shot_Meta_Learning.pdf-0017-04.png)


for some constant _C[′] >_ 0. Using this bound, (34) can be written as follows: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/few-shot/iclr_2024_A_Hierarchical_Bayesian_Model_for_Few-Shot_Meta_Learning_images/iclr_2024_A_Hierarchical_Bayesian_Model_for_Few-Shot_Meta_Learning.pdf-0017-06.png)


The proof completes by dividing both sides by _n_ . 

## B DETAILED DERIVATIONS 

## B.1 ELBO DERIVATION FOR (5) 

We derive the upper bound of the negative marginal log-likelihood for our Bayesian FSL model, that is, deriving (5) in the main paper. 


![](raw/papers/2_LLM_NLP/LLM_paper_list/few-shot/iclr_2024_A_Hierarchical_Bayesian_Model_for_Few-Shot_Meta_Learning_images/iclr_2024_A_Hierarchical_Bayesian_Model_for_Few-Shot_Meta_Learning.pdf-0017-11.png)


Since KL divergence is non-negative, _−L_ ( _L_ ) must be lower bound of the data log-likelihood log _p_ ( _D_ 1: _N_ ), rendering _L_ ( _L_ ) an upper bound of _−_ log _p_ ( _D_ 1: _N_ ). 


![](raw/papers/2_LLM_NLP/LLM_paper_list/few-shot/iclr_2024_A_Hierarchical_Bayesian_Model_for_Few-Shot_Meta_Learning_images/iclr_2024_A_Hierarchical_Bayesian_Model_for_Few-Shot_Meta_Learning.pdf-0017-13.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/few-shot/iclr_2024_A_Hierarchical_Bayesian_Model_for_Few-Shot_Meta_Learning_images/iclr_2024_A_Hierarchical_Bayesian_Model_for_Few-Shot_Meta_Learning.pdf-0017-14.png)


where _ψd_ ( _a_ ) =[�] _[d] j_ =1 _[ψ]_[(] _[a]_[+][(1] _[−][j]_[)] _[/]_[2)][is][the][multivariate][digamma][function,][and] _[ψ]_[(] _[·]_[)][is][the] digamma function. 

We begin with the definition of the KL divergence, 


![](raw/papers/2_LLM_NLP/LLM_paper_list/few-shot/iclr_2024_A_Hierarchical_Bayesian_Model_for_Few-Shot_Meta_Learning_images/iclr_2024_A_Hierarchical_Bayesian_Model_for_Few-Shot_Meta_Learning.pdf-0017-17.png)


where the first term is the negative entropy which admits a closed form due to Gaussian _qi_ ( _θi_ ) = _N_ ( _θi_ ; _mi, Vi_ ), 


![](raw/papers/2_LLM_NLP/LLM_paper_list/few-shot/iclr_2024_A_Hierarchical_Bayesian_Model_for_Few-Shot_Meta_Learning_images/iclr_2024_A_Hierarchical_Bayesian_Model_for_Few-Shot_Meta_Learning.pdf-0017-19.png)


17 

Published as a conference paper at ICLR 2024 

Next we expand the second term of (40) using _p_ ( _θi|ϕ_ ) = _N_ ( _θi_ ; _µ,_ Σ) as follows: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/few-shot/iclr_2024_A_Hierarchical_Bayesian_Model_for_Few-Shot_Meta_Learning_images/iclr_2024_A_Hierarchical_Bayesian_Model_for_Few-Shot_Meta_Learning.pdf-0018-02.png)


Using the following facts from (Bishop, 2006; Braun & McAuliffe, 2008): 


![](raw/papers/2_LLM_NLP/LLM_paper_list/few-shot/iclr_2024_A_Hierarchical_Bayesian_Model_for_Few-Shot_Meta_Learning_images/iclr_2024_A_Hierarchical_Bayesian_Model_for_Few-Shot_Meta_Learning.pdf-0018-04.png)


we can derive the two terms _T_ 1 and _T_ 2 as follows (Recall: _q_ ( _ϕ_ ) = _N_ ( _µ_ ; _m_ 0 _, l_ 0 _[−]_[1][Σ)] _[·IW]_[(Σ;] _[ V]_[0] _[, n]_[0][)][):] 


![](raw/papers/2_LLM_NLP/LLM_paper_list/few-shot/iclr_2024_A_Hierarchical_Bayesian_Model_for_Few-Shot_Meta_Learning_images/iclr_2024_A_Hierarchical_Bayesian_Model_for_Few-Shot_Meta_Learning.pdf-0018-06.png)


Combining all the above results yields the formula (39). 

## B.3 DERIVATION FOR (8) FROM (7) 

Using the result (39), we can easily show that the local episodic optimisation (7) in the main paper ((53) below) reduces to (8) ((54) below). 


![](raw/papers/2_LLM_NLP/LLM_paper_list/few-shot/iclr_2024_A_Hierarchical_Bayesian_Model_for_Few-Shot_Meta_Learning_images/iclr_2024_A_Hierarchical_Bayesian_Model_for_Few-Shot_Meta_Learning.pdf-0018-10.png)


Recall that the optimisation is with respect to _Li_ = ( _mi, Vi_ ) with _L_ 0 = _{m_ 0 _, V_ 0 _, l_ 0 _, n_ 0 _}_ fixed. Plugging (39) into (53) and removing the terms other than ( _mi, Vi_ ) leads to (54). 

## B.4 DERIVATION FOR (10) 

For the quadratic approximation of _li_ ( _θi_ ) = _−_ log _p_ ( _Di|θi_ ) _≈_[1] 2[(] _[θ][i]_ ~~_[−]_~~ _mi_ ) _[⊤] Ai_ ( _θi_ ~~_−_~~ _mi_ )+ const., here we show that the minimiser of (8) ((54) above) can be obtained by the closed-form formula (10) ((55) below). 


![](raw/papers/2_LLM_NLP/LLM_paper_list/few-shot/iclr_2024_A_Hierarchical_Bayesian_Model_for_Few-Shot_Meta_Learning_images/iclr_2024_A_Hierarchical_Bayesian_Model_for_Few-Shot_Meta_Learning.pdf-0018-14.png)


By replacing _li_ ( _θi_ ) by the quadratic approximation, the expected loss term in (8) or (54) can be written as follows: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/few-shot/iclr_2024_A_Hierarchical_Bayesian_Model_for_Few-Shot_Meta_Learning_images/iclr_2024_A_Hierarchical_Bayesian_Model_for_Few-Shot_Meta_Learning.pdf-0018-16.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/few-shot/iclr_2024_A_Hierarchical_Bayesian_Model_for_Few-Shot_Meta_Learning_images/iclr_2024_A_Hierarchical_Bayesian_Model_for_Few-Shot_Meta_Learning.pdf-0018-17.png)


18 

Published as a conference paper at ICLR 2024 

## **Algorithm 2** Meta-test prediction algorithm. 


![](raw/papers/2_LLM_NLP/LLM_paper_list/few-shot/iclr_2024_A_Hierarchical_Bayesian_Model_for_Few-Shot_Meta_Learning_images/iclr_2024_A_Hierarchical_Bayesian_Model_for_Few-Shot_Meta_Learning.pdf-0019-02.png)


After plugging this back to (54), we take the derivatives of the objective with respect to _mi_ and _Vi_ and set them to 0: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/few-shot/iclr_2024_A_Hierarchical_Bayesian_Model_for_Few-Shot_Meta_Learning_images/iclr_2024_A_Hierarchical_Bayesian_Model_for_Few-Shot_Meta_Learning.pdf-0019-04.png)


The solution becomes (10) or (55). 

B.5 DERIVATION FOR (11) 

It is quite straightforward that by plugging (10) or (55) and also (39) in (6), we have our final optimisation problem (11) in the main paper. It is reiterated below: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/few-shot/iclr_2024_A_Hierarchical_Bayesian_Model_for_Few-Shot_Meta_Learning_images/iclr_2024_A_Hierarchical_Bayesian_Model_for_Few-Shot_Meta_Learning.pdf-0019-08.png)


B.6 FORMULAS FOR TEST-TIME ELBO OPTIMISATION (13) 

We provide formulas for the test-time ELBO in (13) ((63) below). For the test-time variational density _v_ ( _θ_ ) = _N_ ( _θ_ ; _m, V_ ) to approximate _p_ ( _θ|D[∗] , ϕ[∗]_ ) for test support data _D[∗]_ and learned _ϕ[∗]_ = ( _µ[∗]_ = _m_ 0 _,_ Σ _[∗]_ = _V_ 0 _/_ ( _n_ 0 + _d_ +2)), we had 


![](raw/papers/2_LLM_NLP/LLM_paper_list/few-shot/iclr_2024_A_Hierarchical_Bayesian_Model_for_Few-Shot_Meta_Learning_images/iclr_2024_A_Hierarchical_Bayesian_Model_for_Few-Shot_Meta_Learning.pdf-0019-11.png)


Using the closed-form Gaussian KL divergence and the reparametrised sampling trick, we can express (63) as: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/few-shot/iclr_2024_A_Hierarchical_Bayesian_Model_for_Few-Shot_Meta_Learning_images/iclr_2024_A_Hierarchical_Bayesian_Model_for_Few-Shot_Meta_Learning.pdf-0019-13.png)


Also, our meta-test prediction algorithm is summarised as a pseudo code in Alg. 2. 

- C TOY EXPERIMENT: WHY HIERARCHICAL BAYESIAN MODEL? (A DETAILED VERSION) 

To demonstrate why our hierarchical Bayesian modelling is effective for few-shot meta learning problems, we devise a simple toy synthetic experiment as a proof of concept. 

19 

Published as a conference paper at ICLR 2024 


![](raw/papers/2_LLM_NLP/LLM_paper_list/few-shot/iclr_2024_A_Hierarchical_Bayesian_Model_for_Few-Shot_Meta_Learning_images/iclr_2024_A_Hierarchical_Bayesian_Model_for_Few-Shot_Meta_Learning.pdf-0020-01.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/few-shot/iclr_2024_A_Hierarchical_Bayesian_Model_for_Few-Shot_Meta_Learning_images/iclr_2024_A_Hierarchical_Bayesian_Model_for_Few-Shot_Meta_Learning.pdf-0020-02.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/few-shot/iclr_2024_A_Hierarchical_Bayesian_Model_for_Few-Shot_Meta_Learning_images/iclr_2024_A_Hierarchical_Bayesian_Model_for_Few-Shot_Meta_Learning.pdf-0020-03.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/few-shot/iclr_2024_A_Hierarchical_Bayesian_Model_for_Few-Shot_Meta_Learning_images/iclr_2024_A_Hierarchical_Bayesian_Model_for_Few-Shot_Meta_Learning.pdf-0020-04.png)


**----- Start of picture text -----**<br>
(a) Model I (b) Model II (c) Model III (Ours)<br>**----- End of picture text -----**<br>


Figure 3: (Toy experiment) Graphical models for the three competing Bayesian models. Here Θ = [ _θ, β_ ] is the the concatenation of the weight and intercept random variables. 

The problem that we consider is basically a multi-task (Bayesian) linear regression problem. First, we generate the multi-task/episodic data by the following process: The input-output data pairs ( _x ∈_ R[2] _, y ∈_ R) are generated from a linear model with a shared normal vector and episodespecific intercepts. More specifically, let _w_ shared _∈_ R[2] be the episode-agnostic shared weight vector, and _{b_ 1 _, b_ 2 _, b_ 3 _}_ ( _bj ∈_ R) be the three candidate intercepts among which each episode can take one randomly. The actual values of the true parameters are: _w_ shared = [ _−_ 5 _._ 4282 _,_ 4 _._ 9867] _, b_ 1 = 1 _._ 4149 _, b_ 2 = _−_ 7 _._ 5315 _, b_ 3 = _−_ 2 _._ 8930. The true data distribution _Ti_ for the episode _i_ (= 1 _,_ 2 _, . . . , N_ ) is defined by the following linear process: 

1. Sample the intercept ID for this episode, _j_ ( _i_ ) _∼{_ 1 _,_ 2 _,_ 3 _}_ uniformly at random. 

2. Repeat the following to collect ( _x, y_ ) pairs (so that ( _x, y_ ) _∼Ti_ ): 


![](raw/papers/2_LLM_NLP/LLM_paper_list/few-shot/iclr_2024_A_Hierarchical_Bayesian_Model_for_Few-Shot_Meta_Learning_images/iclr_2024_A_Hierarchical_Bayesian_Model_for_Few-Shot_Meta_Learning.pdf-0020-09.png)


In this way we ensure that the resulting episodes are not only related to one another through the shared weight vector _w_ shared, but they are differentiated by potentially different intercepts. We generate 50 episodes where _N_ = 40 episodes are used for training and the rest 10 episodes serve as test data. For each training episode _i_ , we have three ( _x, y_ ) samples as an episodic training set _Di_ (all available to a training algorithm, where we make no distinction between support and query sets). At test time, we take three samples as a (labeled) support set _D∗_ ( _∗_ denotes each of the 10 test episodes), and test performance is measured on about 50 unseen samples from the same distribution _T∗_ . 

**Three competing Bayesian methods.** We consider three Bayesian models which exhibit different levels/degrees of flexibility and regularisation. The first one is highly flexible by modeling each individual episode independently with its own parameters, thus with lack of regularisation. The second case is a conventional (non-hierarchical) Bayesian model where we consider a single parameter set shared across episodes, thus too much regularised with lack of flexibility. At last, our hierarchical Bayesian model imposes balanced flexibility and regularisation by introducing the higher-level variable _ϕ_ that captures the inter-episode shared information. 

1. **Model I** : This model has episode-wise parameters while they are all independent with minimal regularisation. More formally, 


![](raw/papers/2_LLM_NLP/LLM_paper_list/few-shot/iclr_2024_A_Hierarchical_Bayesian_Model_for_Few-Shot_Meta_Learning_images/iclr_2024_A_Hierarchical_Bayesian_Model_for_Few-Shot_Meta_Learning.pdf-0020-13.png)


where ( _θi, βi_ ) is the parameters for the episode _i_ . We place the prior _p_ ( _θi, βi_ ) = _N_ ( _µ,_ 10 _[−]_[4] _I_ ) with the model parameter _µ_ shared over episodes. The training amounts to learning the parameter _µ ∈_ R[3] via marginal likelihood maximisation (i.e., max _µ_ log _p_ ( _D_ 1 _, . . . , DN |µ_ )). At test time we do inference _p_ ( _θ∗, β∗|D∗, D_ 1: _N_ ) which boils down to _p_ ( _θ∗, β∗|D∗_ ) due to the cross-episode independence assumption. The graphical model diagram for the model is shown in Fig. 3(a). 

2. **Model II** : Unlike introducing episode-specific variables, this model has a single set of variables shared across all episodes. More specifically, 


![](raw/papers/2_LLM_NLP/LLM_paper_list/few-shot/iclr_2024_A_Hierarchical_Bayesian_Model_for_Few-Shot_Meta_Learning_images/iclr_2024_A_Hierarchical_Bayesian_Model_for_Few-Shot_Meta_Learning.pdf-0020-16.png)


where ( _θ, β_ ) is the episode-agnostic parameters, endowed with the prior _p_ ( _θ, β_ ) = _N_ ( _µ,_ 10 _[−]_[4] _I_ ). Due to this parameter sharing, this model is highly regularised but at the expense of significantly reduced flexibility. Once _µ_ is trained, the inference at test time is done by _p_ ( _θ, β|D∗, D_ 1: _N_ ) which is not simplified further and has to take into account all training and test data. The graphical model diagram in Fig. 3(b). 

20 

Published as a conference paper at ICLR 2024 


![](raw/papers/2_LLM_NLP/LLM_paper_list/few-shot/iclr_2024_A_Hierarchical_Bayesian_Model_for_Few-Shot_Meta_Learning_images/iclr_2024_A_Hierarchical_Bayesian_Model_for_Few-Shot_Meta_Learning.pdf-0021-01.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/few-shot/iclr_2024_A_Hierarchical_Bayesian_Model_for_Few-Shot_Meta_Learning_images/iclr_2024_A_Hierarchical_Bayesian_Model_for_Few-Shot_Meta_Learning.pdf-0021-02.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/few-shot/iclr_2024_A_Hierarchical_Bayesian_Model_for_Few-Shot_Meta_Learning_images/iclr_2024_A_Hierarchical_Bayesian_Model_for_Few-Shot_Meta_Learning.pdf-0021-03.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/few-shot/iclr_2024_A_Hierarchical_Bayesian_Model_for_Few-Shot_Meta_Learning_images/iclr_2024_A_Hierarchical_Bayesian_Model_for_Few-Shot_Meta_Learning.pdf-0021-04.png)


**----- Start of picture text -----**<br>
(a) Weight dim-1 (b) Weight dim-2 (c) Intercept<br>**----- End of picture text -----**<br>


Figure 4: Toy experiments. Visualisation of the learned posterior means compared to the true values (blue-circled). (a) weight dim-1 ( _θ_ [0] vs. _w_ shared[0]), (b) weight dim-2 ( _θ_ [1] vs. _w_ shared[1]) and (c) intercept ( _β_ vs. _bj_ ( _∗_ )). In each plot, the X-axis shows the indices of the true intercepts sampled, that is, _j_ ( _∗_ ) _∈{_ 1 _,_ 2 _,_ 3 _}_ , for 10 test episodes. In the titles we also report the distances (errors) between the true values and the posterior means for the three methods, averaged over 10 episodes. 

3. **Model III (Ours)** : This is our hierarchical model where each episode has its own parameters (like Model I) but there is a globally governing variable _ϕ_ to regularise the episode-wise parameters. That is, we have the same regression form as (66) with episode-wise ( _θi, βi_ ) parameters, but our prior distributions are defined hierarchically as: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/few-shot/iclr_2024_A_Hierarchical_Bayesian_Model_for_Few-Shot_Meta_Learning_images/iclr_2024_A_Hierarchical_Bayesian_Model_for_Few-Shot_Meta_Learning.pdf-0021-07.png)


where ( _m, V_ ) is the model parameters to be learned. At test time, we infer _p_ ( _θ∗, β∗|D∗, D_ 1: _N_ ), however, unlike the Model II case, each training data _Di_ and the test data _D∗_ do not have equal, symmetric contribution. Note that the training data affect the posterior indirectly only through the higher-level variable _ϕ_ as follows: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/few-shot/iclr_2024_A_Hierarchical_Bayesian_Model_for_Few-Shot_Meta_Learning_images/iclr_2024_A_Hierarchical_Bayesian_Model_for_Few-Shot_Meta_Learning.pdf-0021-09.png)


That is, while our model is as flexible as Model I due to the episode-wise parameters, their impacts on the test prediction are controlled/regularised in a very sensible manner. The graphical model diagram in Fig. 3(c). 

Note that all the posterior inferences of the above three models can be done in _closed forms_ due to the linear-Gaussian properties. The details of the posterior distributions and derivations as well as the model training can be found in Sec. C.1. 

**Results.** For the 10 test episodes, we obtain the posterior means of the weights and intercept parameters E[ _θ, β|D∗, D_ 1: _N_ ] for the three models via the closed-form solutions as detailed in Sec. C.1. The test support set size _|D∗|_ = 3. With these posterior means, we predict the outputs of about 50 unseen test inputs. The mean absolute errors (MAE) of the three models averaged over 10 test episodes are: **Model I** = **2** _._ **87** , **Model II** = **3** _._ **13** , and **Model III (ours)** = **1** _._ **28** , clearly showing the superiority of our model to other competing methods. In Fig. 4 we also visualise the posterior means to check how much they deviate from the true weights and intercept, namely the difference between the posterior mean of ( _θ, β_ ) and the true ( _w_ shared _, bj_ ( _∗_ )). First, we see that **Model II** ’s posterior means rarely change over the test episodes, in other words, the impact of test support data _D∗_ is diminished. This behavior is expected since the model imposes too much regularisation with little flexibility, and the test prediction is dominated by the mean model obtained from training data _D_ 1: _N_ . Secondly, **Model I** exhibits highly sensitive predictions over the test episodes, which mainly originates from little regularisation. In other words, the posterior is affected too sensitively by the current episode’s support data, thus being vulnerable to overfitting especially when the support data size is small, typical in the few-shot learning. The model failed to capture useful shared information, in this case _w_ shared, from diverse training episodes. On the other hand, our **Model III** takes the balance between the above two extremes, imposing proper amount of regularisation and endowing adequate flexibility. Our posterior estimation best extracts the shared episode-agnostic information (the weight parameters fluctuate less over the test episodes), and at the same time, captures the episode-specific features the most accurately (the estimated intercepts are aligned well with the true values). 

## C.1 DERIVATIONS FOR TRAINING AND POSTERIORS IN TOY EXPERIMENTS 

For the convenience in notation, we let Θ = [ _θ, β_ ] _∈_ R[3] be the concatenated random variables (subscripts can be applied accordingly). The training data _Di_ for each episode _i_ consists of the _inputs_ 

21 

Published as a conference paper at ICLR 2024 

with the constant 1 appended, denoted as _Xi ∈_ R _[n][×]_[3] , and the outputs _Yi ∈_ R _[n]_ . With this notation, the linear regression model can be succinctly written as _Yi_ = Θ _Xi_ . Similarly, the test-time support data _D∗_ is decomposed into ( _X∗, Y∗_ ). The noise standard deviation is denoted by _σ_ = 10 _[−]_[2] . 

## C.1.1 MODEL I 

Model I (the episode-wise independent model) can be formally written as: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/few-shot/iclr_2024_A_Hierarchical_Bayesian_Model_for_Few-Shot_Meta_Learning_images/iclr_2024_A_Hierarchical_Bayesian_Model_for_Few-Shot_Meta_Learning.pdf-0022-04.png)


The posterior _p_ (Θ1 _, . . . ,_ Θ _N |D_ 1 _, . . . , DN_ ) is fully factorised over _i_ , and we can deal with individual terms _p_ (Θ _i|Di_ ) where 


![](raw/papers/2_LLM_NLP/LLM_paper_list/few-shot/iclr_2024_A_Hierarchical_Bayesian_Model_for_Few-Shot_Meta_Learning_images/iclr_2024_A_Hierarchical_Bayesian_Model_for_Few-Shot_Meta_Learning.pdf-0022-06.png)


Due to the product-of-Gaussians form, we have the closed-form posterior: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/few-shot/iclr_2024_A_Hierarchical_Bayesian_Model_for_Few-Shot_Meta_Learning_images/iclr_2024_A_Hierarchical_Bayesian_Model_for_Few-Shot_Meta_Learning.pdf-0022-08.png)


The training amounts to maximising the data (log-)likelihood, max _µ_ log _p_ ( _D_ 1 _, . . . , DN |µ_ ) where the objective is fully decomposed over _i_ as follows: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/few-shot/iclr_2024_A_Hierarchical_Bayesian_Model_for_Few-Shot_Meta_Learning_images/iclr_2024_A_Hierarchical_Bayesian_Model_for_Few-Shot_Meta_Learning.pdf-0022-10.png)


Using the Gaussian posterior form (72), we can easily evaluate log _p_ ( _D_ 1 _, . . . , DN |µ_ ) and also optimise it with respect to _µ_ . At test time, the posterior of Θ _∗_ given all the training data and the test support data, _p_ (Θ _∗|D∗, D_ 1 _, . . . , DN_ ) equals _p_ (Θ _∗|D∗_ ) due to the independence assumption, and admits the same Gaussian form as (72) with the test support data ( _X∗, Y∗_ ) in the place of ( _Xi, Yi_ ). 

## C.1.2 MODEL II 

Model II (the shared model across episodes) can be formally written as: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/few-shot/iclr_2024_A_Hierarchical_Bayesian_Model_for_Few-Shot_Meta_Learning_images/iclr_2024_A_Hierarchical_Bayesian_Model_for_Few-Shot_Meta_Learning.pdf-0022-14.png)


The posterior _p_ (Θ _|D_ 1 _, . . . , DN_ ) can be derived as follows: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/few-shot/iclr_2024_A_Hierarchical_Bayesian_Model_for_Few-Shot_Meta_Learning_images/iclr_2024_A_Hierarchical_Bayesian_Model_for_Few-Shot_Meta_Learning.pdf-0022-16.png)


Again, due to the product-of-Gaussians form, we have the closed-form posterior: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/few-shot/iclr_2024_A_Hierarchical_Bayesian_Model_for_Few-Shot_Meta_Learning_images/iclr_2024_A_Hierarchical_Bayesian_Model_for_Few-Shot_Meta_Learning.pdf-0022-18.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/few-shot/iclr_2024_A_Hierarchical_Bayesian_Model_for_Few-Shot_Meta_Learning_images/iclr_2024_A_Hierarchical_Bayesian_Model_for_Few-Shot_Meta_Learning.pdf-0022-19.png)


Likewise, the training amounts to maximising the data (log-)likelihood, max _µ_ log _p_ ( _D_ 1 _, . . . , DN |µ_ ) where the objective becomes: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/few-shot/iclr_2024_A_Hierarchical_Bayesian_Model_for_Few-Shot_Meta_Learning_images/iclr_2024_A_Hierarchical_Bayesian_Model_for_Few-Shot_Meta_Learning.pdf-0022-21.png)


Again, using the Gaussian posterior form (78), we can easily evaluate log _p_ ( _D_ 1 _, . . . , DN |µ_ ) and also optimise it with respect to _µ_ . At test time, the posterior of Θ _∗_ given all the training data and the test support data, _p_ (Θ _∗|D∗, D_ 1 _, . . . , DN_ ) admits a Gaussian form, derived similarly as (78) with the test support data statistics _X∗[⊤][X][∗]_[and] _[ X] ∗[⊤][Y][∗]_[additionally added to the training statistics.] 

22 

Published as a conference paper at ICLR 2024 

## C.1.3 MODEL III (OURS) 

Our Model III (the hierarchical Bayesian model) can be formally written as: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/few-shot/iclr_2024_A_Hierarchical_Bayesian_Model_for_Few-Shot_Meta_Learning_images/iclr_2024_A_Hierarchical_Bayesian_Model_for_Few-Shot_Meta_Learning.pdf-0023-03.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/few-shot/iclr_2024_A_Hierarchical_Bayesian_Model_for_Few-Shot_Meta_Learning_images/iclr_2024_A_Hierarchical_Bayesian_Model_for_Few-Shot_Meta_Learning.pdf-0023-04.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/few-shot/iclr_2024_A_Hierarchical_Bayesian_Model_for_Few-Shot_Meta_Learning_images/iclr_2024_A_Hierarchical_Bayesian_Model_for_Few-Shot_Meta_Learning.pdf-0023-05.png)


where we use the property of the product of Gaussians for the derivation from (86) to (87). Now, in (87), due to the product-of-Gaussians form, we have the closed-form posterior: 

_p_ ( _ϕ|D_ 1 _, . . . , DN_ ) = _N_ ( _ϕ_ ; _A[−]_[1] _b, A[−]_[1] ) where 


![](raw/papers/2_LLM_NLP/LLM_paper_list/few-shot/iclr_2024_A_Hierarchical_Bayesian_Model_for_Few-Shot_Meta_Learning_images/iclr_2024_A_Hierarchical_Bayesian_Model_for_Few-Shot_Meta_Learning.pdf-0023-08.png)


The training amounts to maximising the data (log-)likelihood, max _m,V_ log _p_ ( _D_ 1 _, . . . , DN |m, V_ ) where the objective becomes: 

log _p_ ( _D_ 1 _, . . . , DN |m, V_ ) = log _p_ ( _ϕ, D_ 1 _, . . . , DN_ ) _−_ log _p_ ( _ϕ|D_ 1 _, . . . , DN_ ) (for any _ϕ_ ) (89) = log _p_ ( _ϕ_ ) + log _p_ ( _D_ 1 _, . . . , DN |ϕ_ ) _−_ log _p_ ( _ϕ|D_ 1 _, . . . , DN_ ) (for any _ϕ_ ) _,_ (90) and the _ϕ_ -conditioned data log-likelihood log _p_ ( _D_ 1 _, . . . , DN |ϕ_ ) can be derived as follows: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/few-shot/iclr_2024_A_Hierarchical_Bayesian_Model_for_Few-Shot_Meta_Learning_images/iclr_2024_A_Hierarchical_Bayesian_Model_for_Few-Shot_Meta_Learning.pdf-0023-11.png)


So we can easily evaluate log _p_ ( _D_ 1 _, . . . , DN |m, V_ ) and also optimise it with respect to ( _m, V_ ). At test time, the posterior of Θ _∗_ given all the training data and the test support data, _p_ (Θ _∗|D∗, D_ 1 _, . . . , DN_ ) can be derived as follows. We start with: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/few-shot/iclr_2024_A_Hierarchical_Bayesian_Model_for_Few-Shot_Meta_Learning_images/iclr_2024_A_Hierarchical_Bayesian_Model_for_Few-Shot_Meta_Learning.pdf-0023-13.png)


and the first term in the integration, _p_ (Θ _∗|ϕ, D∗_ ), since it is proportional to the product of two Gaussians _p_ (Θ _∗|ϕ, D∗_ ) _∝ p_ ( _D∗|_ Θ _∗_ ) _· p_ (Θ _∗|ϕ_ ) = _N_ ( _Y∗_ ; _X∗_ Θ _∗, σ_[2] _I_ ) _· N_ (Θ _∗_ ; _ϕ, σ_[2] _I_ ), it admits Gaussian (from the Gaussian properties), 


![](raw/papers/2_LLM_NLP/LLM_paper_list/few-shot/iclr_2024_A_Hierarchical_Bayesian_Model_for_Few-Shot_Meta_Learning_images/iclr_2024_A_Hierarchical_Bayesian_Model_for_Few-Shot_Meta_Learning.pdf-0023-15.png)


23 

Published as a conference paper at ICLR 2024 


![](raw/papers/2_LLM_NLP/LLM_paper_list/few-shot/iclr_2024_A_Hierarchical_Bayesian_Model_for_Few-Shot_Meta_Learning_images/iclr_2024_A_Hierarchical_Bayesian_Model_for_Few-Shot_Meta_Learning.pdf-0024-01.png)


The second term of (95) becomes a Gaussian following the derivation similar to (88) with the test support data _X∗_ and _Y∗_ included. Consequently we let _p_ ( _ϕ|D∗, D_ 1 _, . . . , DN_ ) = _N_ ( _ϕ_ ; _A[−] ∗_[1] _[b][∗][, A] ∗[−]_[1][)][.] At last, (95) is the marginalisation of the product of two Gaussians, which admits the following closed form: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/few-shot/iclr_2024_A_Hierarchical_Bayesian_Model_for_Few-Shot_Meta_Learning_images/iclr_2024_A_Hierarchical_Bayesian_Model_for_Few-Shot_Meta_Learning.pdf-0024-03.png)


## D IMPLEMENTATION DETAILS AND EXPERIMENTAL SETTINGS 

We implement our NIW-Meta using PyTorch (Paszke et al., 2017) and the Higher (Grefenstette et al., 2019)[4] library. The latter makes the implementation of the backpropagation through the functional network weights in PyTorch modules very easy. Real codes for the synthetic SineLine regression dataset and the large-scale ViT are also attached in the Supplement Material to help understanding of our algorithm. For all few-shot classification experiments, we use the ProtoNet-like parameter-free NCC head in our NIW-Meta. Some important implementation details on the SGLD iterations for quadratic approximation of the local episodic optimisation include: we have either 3 steps without burn-in (for large-scale backbones ViT) or 5 steps with 2 burn-in steps (for smaller backbones ConvNet, ResNet-18, and CNP). Before starting SGLD iterations, the network is initialised with the current model parameters _m_ 0. For reliable variance estimation of _Ai_ , a small regulariser is added to the diagonal entries of the variances. 

For the standard benchmarks with ConvNet/ResNet backbones, we follow the standard protocols of (Wang et al., 2019; Mangla et al., 2020; Zhang et al., 2021): With 64/16/20 and 391/97/160 train/validation/test class splits for _mini_ ImageNet and _tiered_ ImageNet datasets, respectively, the images are resized to 84 pixels. We initialise the _m_ 0 parameters from the pretrained models: checkpoints from (Wang et al., 2019) for Conv-4 and ResNet-18 and checkpoints from (Mangla et al., 2020) for WRN-28-10. With the stochastic gradient descent (SGD) optimizer, we set momentum 0.9, weight decay 0.0001, and initial learning rate 0.01 for _mini_ ImageNet and 0.001 for _tiered_ ImageNet. We have learning rate schedule by reducing the learning rate by the factor of 0.1 at epoch 70. 

For the large-scale ViT backbones, we utilise the code base from (Hu et al., 2022). We use the self-supervised pretrained checkpoints from (Caron et al., 2021) to initialise the _m_ 0 parameters. The CIFAR-FS dataset is formed by splitting the original CIFAR-100 into 64/16/20 train/validation/test classes. For training, we run 100 epochs, each epoch comprised of 2000 episodes. We follow the same warm-up plus cosine annealing learning rate scheduling as (Hu et al., 2022). For test evaluation, we have 600 episodes from the test splits. 

For the few-shot regression experiments with ShapeNet datasets, we basically follow all experimental settings and CNP/ANP network architectures from (Gao et al., 2022). For instance, in the ShapeNet1D dataset, we run our algorithm for 500 _K_ iterations with learning rate 10 _[−]_[4] where each batch iteration consists of 10 episodes. The CNP backbone, for instance, in the Distractor dataset case, has a ResNet image encoder and a linear target encoder, where the concatenated instance-wise embeddings then go through a three-layer fully connected network followed by max pooling. The decoder has a similar architecture and converts the support set embedding and a query image into a target label. For the conv-net plus ridge-regression head backbone (C+R) tested for our method, the conv-net feature extractors are formed by taking the encoder parts of the CNP architectures in (Gao et al., 2022) while discarding the pooling operations and decoders. Also the ridge-regression L2 regularisation coefficient is set to _λ_ = 1 _._ 0 for all datasets. 

## E COMPUTATIONAL COMPLEXITY, RUNNING TIME AND MEMORY FOOTPRINT 

Although we have introduced a principled Bayesian model/framework for FSL with solid theoretical support, the extra steps introduced in our training/test algorithms appear to be more complicated than simple feed-forward workflows (e.g., ProtoNet (Snell et al., 2017)). To this end, we have analysed the time complexity of the proposed algorithm contrasted with ProtoNet (Snell et al., 2017). For fair comparison, our approach adopts the same NCC head on top of the feature space as ProtoNet. 

> 4https://github.com/facebookresearch/higher 

24 

Published as a conference paper at ICLR 2024 

Table 8: (Per-episode) Time complexity of our NIW-Meta vs. ProtoNet. We denote by _FD_ and _BD_ the forward-pass and backpropagation times with data _D_ = _S_ upport or _Q_ uery. In our algorithm, _ML_ , _MV_ , and _MS_ indicate the numbers of SGLD iterations, test-time variational inference steps for (13) or (63,64), and the number of test-time model samples _θ_[(] _[s]_[)] , respectively. The costs required for reparametrised sampling in model space and regulariser computation in (11) or (62) are denoted by _O_ ( _d_ ) where _d_ = number of backbone parameters. 

||Trainingtime|Test time|
|---|---|---|
|NIW-Meta (Ours)|(_FS_+_FQ_+_BQ_)_·_(_ML_+1)<br>+_O_(_d_)|(_FS_+_BS_)_· MV_ +<br>(_FS_+_FQ_)_· MS_ +_O_(_d_)|
|ProtoNet|_FS_+_FQ_+_BQ_|_FS_+_FQ_|



Table 9: Effect of the finite number of episodes ( _N_ ) on the generalization error gap in terms of the sample complexity gap log(2 _√nN /δ_ ) _/_ ( _nN_ ) in (21). Here _δ_ =0 _._ 001. 

|_N_|(#episodes)|_n_ (#shots|_×_ #ways)|Sample complexityerrorgap (_↓_)|
|---|---|---|---|---|
||103|1_×_5|/ 5_×_5|2_._4_×_10_−_3 / 5_._0_×_10_−_4|
||104|1_×_5|/ 5_×_5|3_._0_×_10_−_4 / 5_._5_×_10_−_5|
||105|1_×_5|/ 5_×_5|2_._8_×_10_−_5 / 6_._0_×_10_−_6|



The computational complexity is summarised in Table 8. Despite seemingly increased complexity in the training/test algorithms, our method incurs only constant-factor overhead compared to the minimal-cost ProtoNet. 

As we claimed in the main paper, one of the main drawbacks of MAML (Finn et al., 2017) is the computational overhead to keep track of a large computational graph for inner gradient descent steps. Unlike MAML, our NIW-Meta has a much more efficient episodic optimisation strategy, i.e., our local episodic optimisation only computes the (constant) first/second-order moment statistics of the episodic loss function without storing the full optimisation trace. 

To verify this, we measure and compare the memory footprints and running times of MAML and NIWMeta on two real-world classification/regression datasets: _mini_ ImageNet 1-shot with the ResNet-18 backbone and ShapeNet-1D with the ConvNet backbone. The results in Fig. 5 show that NIW-Meta has far lower memory requirement than MAML (even smaller than 1-inner-step MAML) while MAML suffers from heavy use of memory space, nearly linearly increasing as the number of inner steps. The running times of our NIW-Meta are not prohibitively larger compared to MAML where the main computational bottleneck is the SGLD iterations for quadratic approximation of the local episodic optimisation. We tested two scenarios with the number of SGLD iterations 2 and 5, and we have nearly the same (or even better) training speed as the 1-inner-step MAML. 

## F TRAINING STABILITY AND IMPACT OF NUMBER OF TRAINING EPISODES 

In our theoretical analysis of the generalisation error (Sec. 5 in the main paper and our proofs in Sec. A), we regard the number of training episodes _N_ as infinity. In practice, _N_ is finite, but large enough ( _N ∼_ 100 _K_ in typical FSL), and we simply take it as **infinite** _N_ for mathematical convenience (e.g., to have the first KL term in (5) vanish; reduction to the task population mean from (29) to (30)). To see the effect of finite _N_ on the generalization performance, we exemplify several typical _N_ values and corresponding generalization (sample complexity) error gaps (21) in Table 9. We see that even for relatively small _N_ , error gaps are small/negligible. 

In addition, we investigate the impact of the number of training episodes _N_ on training stability. We illustrate it in Fig. 6, which shows that our method works stably well even for a small number of initial episodes, convergence being as fast as ProtoNet with far better generalization performance. 

25 

Published as a conference paper at ICLR 2024 


![](raw/papers/2_LLM_NLP/LLM_paper_list/few-shot/iclr_2024_A_Hierarchical_Bayesian_Model_for_Few-Shot_Meta_Learning_images/iclr_2024_A_Hierarchical_Bayesian_Model_for_Few-Shot_Meta_Learning.pdf-0026-01.png)


**----- Start of picture text -----**<br>
miniImageNet ShapeNet-1D<br>6000 NIW-Meta (Ours) 1600 NIW-Meta (Ours)<br>MAML MAML<br>5500<br>1400<br>5000 1200<br>4500 1000<br>800<br>4000<br>600<br>3500<br>400<br>1 2 3 4 5 1 2 3 4 5<br>Number of inner steps in MAML Number of inner steps in MAML<br>(a) GPU memory footprints<br>miniImageNet ShapeNet-1D<br>180 NIW-Meta (#SGLD=2) NIW-Meta (#SGLD=2)<br>160 NIW-Meta (#SGLD=5) 10 NIW-Meta (#SGLD=5)<br>MAML MAML<br>140<br>8<br>120<br>100 6<br>80<br>4<br>60<br>40<br>1 2 3 4 5 1 2 3 4 5<br>Number of inner steps in MAML Number of inner steps in MAML<br>(b) Per-episode training times<br>GPU memory footprint (MB) GPU memory footprint (MB)<br>Per-episode training time (msec) Per-episode training time (msec)<br>**----- End of picture text -----**<br>


Figure 5: Computational complexity of MAML (Finn et al., 2017) and our NIW-Meta. (a) GPU memory footprints (in MB) for a single batch. (b) Per-episode training times (in milliseconds). For our NIW-Meta models, the time for the number of burn-in steps (2 steps in this case) is also included. That is, NIW-Meta(#SGLD=2) runs 2 + 2 SGLD iterations, and NIW-Meta(#SGLD=5) runs 5 + 2, respectively, compared to MAML with 1 _∼_ 5 inner iterations. We use the ResNet-18 backbone for _mini_ ImageNet in 1-shot classification and the ConvNet backbone for ShapeNet-1D regression (10 episodes per batch). 


![](raw/papers/2_LLM_NLP/LLM_paper_list/few-shot/iclr_2024_A_Hierarchical_Bayesian_Model_for_Few-Shot_Meta_Learning_images/iclr_2024_A_Hierarchical_Bayesian_Model_for_Few-Shot_Meta_Learning.pdf-0026-03.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/few-shot/iclr_2024_A_Hierarchical_Bayesian_Model_for_Few-Shot_Meta_Learning_images/iclr_2024_A_Hierarchical_Bayesian_Model_for_Few-Shot_Meta_Learning.pdf-0026-04.png)


Figure 6: CIFAR 1-shot learning with ViT (DINO/s). (Left) Training losses vs. training episodes (2000 episodes per epoch). We plot the training loss of our NIW-Meta (i.e., (11)) in blue, and superimpose the training cross-entropy loss in red, where the latter is comparable to ProtoNet’s training CE loss in magenta. We see that our NIW-Meta training is pretty stable, where the convergence speed is as fast as ProtoNet. (Right) Performance on the validation set as training episodes increases. The validation losses of our NIW-Meta (blue) and ProtoNet (cyan) are comparable, while we also compare the validation accuracy of our NIW-Meta (red) against ProtoNet (magenta). 

26 

Published as a conference paper at ICLR 2024 

Table 10: Predictive log-marginal-likelihood scores (LMLHD) on the Sine-Line dataset. The higher the better uncertainty quantification and generalisation. 

|Context size|1|2|4|6|8|12|16|
|---|---|---|---|---|---|---|---|
|(Volpp et al., 2023)|-18.38|-15.98|-13.69|-12.74|-11.75|-11.14|-10.23|
|||||||||
|NIW-Meta (Ours)|-17.02|-14.74|-12.60|-8.14|-3.51|-1.55|-1.33|



## G ADDITIONAL DISCUSSIONS 

## G.1 COMPARISON TO BAYESIAN NEURAL PROCESSES 

An interesting question is what are the relative merits of a complete Bayesian treatment but with restricted Gaussian forms as in our model, versus a shallow Bayesian but GMM-like rich variational forms such as (Volpp et al., 2023). We do not know which is absolutely better than the other. Further theoretical and empirical study needs to be carried out in this regard. But we have some experimental evidence as demonstrated in our paper, the comparison with MetaQDA (Zhang et al., 2021) shallow Bayesian approach that only places prior distribution on the model head parts while freezing the feature extractor (Table 2, 3, 5). As shown, our complete Bayesian treatment outperformed it, in both test generalisation performance and uncertainty calibration. This is one supporting evidence of why a complete Bayesian treatment could be more promising than a shallow Bayesian treatment. 

For the experimental demonstration, we have done additional experiments on the Sine-Line dataset to report the predictive marginal test likelihood score, which is directly comparable to the shallow Bayesian embedding models, esp., the Bayesian Neural Process model of (Volpp et al., 2023). The predictive log-marginal-likelihood scores (LMLHD) are shown in Table 10. We try to match the experimental settings from (Volpp et al., 2023) so that the results are comparable. More specifically, we test on 256 tasks, each of which consists of 64 samples with a varying number of context samples (1, 2, 4, 6, 8, 12 and 16). The number of posterior samples used to compute/approximate the predictive marginal likelihood is 1024. As shown, our NIW-Meta has higher scores, especially for larger context size, implying that capturing uncertainty in full model parameters is important for generalisation capability. 

## G.2 JUSTIFICATION OF MODEL AND ALGORITHM CHOICES 

**Motivation for distributional estimate** _q_ ( _ϕ_ ) **.** If _ϕ_ were directly linked to the observed data _Di_ ’s in our graphical model Fig. 1, then _ϕ_ can tend to be determined deterministically, as the number of data _N_ becomes large. However, _ϕ_ is linked to _latent variables θi_ ’s, so the belief on _ϕ_ also needs to capture and accumulate the uncertainty in _θi_ ’s, which amounts to marginalising out the _θi_ variables. So, it may not not be appropriate to treat the posterior for _ϕ_ as a delta function (0 uncertainty), and it is better to follow the Bayesian inference principle, i.e., let the posterior be computed from the observed evidence. 

**Why distributional estimate** _q_ ( _ϕ_ ) **if we only use the mode of** _q_ ( _ϕ_ ) **at meta-test time.** Using the mode of _q_ ( _ϕ_ ) is only for practical convenience and simplicity. Although we used the mode, the distributional form _q_ ( _ϕ_ ) would take into account uncertainty in its optimisation, and thus lead to a different solution from the deterministic one. 

**Why not the same inference approach for meta testing as meta training.** For meta testing, we may attempt to solve the optimisation problem similar to (7) or (8) for meta training with _L_ 0 fixed, which also results in the same meta test solution as our derivation in Sec. 3.2. This can also be verified by inspecting the similarity between (13) and (7) or (8). However, this approach requires that _L_ 0 be fixed at the trained value, which we do not know for sure in the pure optimisation perspective. In Sec. 3.2, we aimed to derive the meta-test optimisation problem from the Bayesian perspective from the outset, which offers us a reasonable justification for why _L_ 0 can be fixed. 

**Quality of the quadratic approximation.** In (9), we have made the quadratic approximation for the negative likelihood, _−_ log _p_ ( _Di|θ_ ) _≈_[1] 2[(] _[θ][−] mi_ ) _[⊤] Ai_ ( _θ−mi_ ) + const. To see the quality of this approximation, we empirically evaluated the true value (the left hand side of the approximation) and the quadratic approximation (the right hand side) on the _mini_ ImageNet 5-shot dataset with 

27 

Published as a conference paper at ICLR 2024 

the ResNet-18 backbone. We take random 10 perturbations of _θ_ around the mode/mean _mi_ with perturbation radius 0 _._ 1. The relative error of the quadratic approximation is 0 _._ 0050 _±_ 0 _._ 0004. This shows that the negative log-likelihood is well approximated by our quadratic function in the vicinity of the mode/mean. 

## G.3 LIMITATIONS AND FUTURE WORKS 

**Limitations.** 1) Our NIW-Meta introduces some extra hyperparameters (e.g., the number of SGLD iterations, the number of burn-in steps). These are currently estimated empirically, but a more rigorous study on how to select them automatically needs to be addressed. 2) Although it is empirically verified that our quadratic episodic loss optimisation is effective, more theoretical analysis on the quality of this approximation as well as its impact on the final results, needs to be done. 

**Future works.** We have quite an extensive evaluation on popular few-shot classification and regression benchmarks. However, we would like to evaluate our approach on new emerging applications of few-shot learning such as efficient learning of the implicit neural representations such as NeRF, e.g., (Tancik et al., 2021). 

28 

