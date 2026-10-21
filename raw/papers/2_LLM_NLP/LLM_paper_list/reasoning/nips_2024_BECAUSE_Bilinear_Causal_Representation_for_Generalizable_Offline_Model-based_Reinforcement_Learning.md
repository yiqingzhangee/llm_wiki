# **BECAUSE: Bilinear Causal Representation for Generalizable Offline Model-based Reinforcement Learning** 

**Haohong Lin**[1] **, Wenhao Ding**[1] **, Jian Chen**[1] **, Laixi Shi**[2] **, Jiacheng Zhu**[3] **, Bo Li**[4] **, Ding Zhao**[1] 1CMU, 2Caltech, 3MIT, 4UChicago & UIUC `{haohongl, wenhaod, dingzhao}@andrew.cmu.edu` 

## **Abstract** 

Offline model-based reinforcement learning (MBRL) enhances data efficiency by utilizing pre-collected datasets to learn models and policies, especially in scenarios where exploration is costly or infeasible. Nevertheless, its performance often suffers from the objective mismatch between model and policy learning, resulting in inferior performance despite accurate model predictions. This paper first identifies the primary source of this mismatch comes from the underlying confounders present in offline data for MBRL. Subsequently, we introduce **B** ilin **E** ar **CAUS** al r **E** presentation (BECAUSE), an algorithm to capture causal representation for both states and actions to reduce the influence of the distribution shift, thus mitigating the objective mismatch problem. Comprehensive evaluations on 18 tasks that vary in data quality and environment context demonstrate the superior performance of BECAUSE over existing offline RL algorithms. We show the generalizability and robustness of BECAUSE under fewer samples or larger numbers of confounders. Additionally, we offer theoretical analysis of BECAUSE to prove its error bound and sample efficiency when integrating causal representation into offline MBRL. 

## **1 Introduction** 

Offline Reinforcement Learning (RL) has shown great promise in learning directly from historically collected datasets, especially in scenarios where active interaction is expensive or infeasible [1]. Specifically, offline model-based reinforcement learning (MBRL) [2, 3, 4], learning policies with an estimated world model, generally perform better than their model-free counterparts in long-horizon tasks such as self-driving vehicles [5], robotics [6], and healthcare [7]. However, offline RL suffers from distribution shift since the rollout data could sample from some unknown behavior policies that are sub-optimal or from slightly different environments compared to the deployment time [8]. 

Although identifying distribution shift issues, many Data collection with behavior policy of the current offline MBRL works fail to model the (Unknown) shift in environment dynamics, which is ubiquitous Data Buffer Model Policy test Environment and could cause catastrophic failure of trained policy {𝑠, 𝑎, 𝑠[!] , 𝑟} 𝑇[(] (𝑠[!] |𝑠, 𝑎) 𝜋(𝑎|𝑠) 𝑇(𝑠′|𝑠, 𝑎) at a slightly different deployment stage. Furthermore, **Objective Mismatch** since the learning objectives of the world models and low model loss ≠ success low model loss ≈ success policies are isolated from each other, a significant Success Our method challenge in offline MBRL is objective mismatch [9, Failure 10] problem (shown in Figure 1): models that achieve Model Loss Model Loss a lower training loss are not necessarily better for control performance. For example, in long-horizon Figure 1: The objective mismatch problem. planning tasks, the reward is sparse yet the prediction accuracy of the model may decay as the horizon 

Preprint. Under review. 

enlarges and compounding error accumulates. Previous works [9, 10] have attempted to reduce such objective mismatch by jointly learning the model and policy. However, the performance is suboptimal in the absence of the underlying cause of objective mismatch [8]. 

In this work, we identify that the objective mismatch between model estimation and policy learning comes from two sources of _distribution shift_ in offline MBRL: (1) shift between the online optimal policy and offline sub-optimal behavior policies, and (2) shift between the data collection environment and online testing environments. Unlike humans, who make decisions based on reasoning over taskrelevant factors, models in offline RL memorize correlations without learning the causality. The sub-optimal behavior policies introduce _spurious correlations_ [11] between actions and states, making the model memorize specific actions. When online testing environments differ from the data collection environment, the model could overfit spurious correlations in the state and fail to generalize to unseen states. Based on the analysis of the above mismatch, our work differs from previous work of causal model-based RL [12, 13] in that we model causality in both model and policy learning. We aim to avoid spurious correlation by discovering underlying structures between abstracted states and actions. 

To alleviate objective mismatch and generalize well, we introduce the BilinEar CAUSal rEpresentation (BECAUSE) that integrates the causal representation in both world model learning and planning of MBRL agents. Inspired by preliminary works that use bilinear MDPs to capture the structural representation in MBRL [14], we first approximate the causal representation to capture the low-rank structure in the world model, then use this learned representation to facilitate planning by quantifying the uncertainty of sampled transition pairs. Consequently, we factorize the spurious correlations and learn a unified representation for both the world model and planner. 

In summary, the contribution of this paper is threefold: 

- We formulate offline MBRL into the causal representation learning problem, highlighting the tight connection between structural causal models and low-rank structures in MDPs. To the best of our knowledge, this is the first work that systematically reveals the connection between causal representation learning and Bilinear MDPs. 

- We propose BECAUSE, an empirical causal representation framework, based on the above formulation. BECAUSE first learns a causal world model, then fosters the generalizability of offline RL agents by quantifying the uncertainty of the state transition, which facilitates conservative planning to mitigate the objective mismatch. 

- We provide extensive empirical studies and performance analysis in tasks of multiple domains to demonstrate the superiority of BECAUSE over existing baselines, which illustrates its potential to improve the generalizability and robustness of offline MBRL algorithms. 

## **2 Problem Formulation** 

To alleviate the objective mismatch problem and the degraded performance caused by the spurious correlation, we first provide our novel formulation of learning the underlying causal structures of Markov Decision Process (MDP) under the bilinear MDP setting, then introduce the causal discovery for MDP with confounders. 

## **2.1 Preliminary: MDP and Bilnear MDP** 

We denote an episodic finite-horizon MDP by _M_ = � _S, A, T , H, r_ �, which is composed of state space _S_ , action space _A_ , a set of transition functions _T_ , planning horizon _H_ and reward function _r_ associated with task preferences. Without loss of generality in many real-world practices, we assume that the reward function is bounded by _rh ∈_ [0 _,_ 1] _, ∀h ∈_ [ _H_ ]. Specifically, we are interested in a goal-conditioned reward setting, where _∀g ∈S_ , _r_ ( _s, a_ ; _g_ ) = 1 if and only if _s_ = _g_ . Given a policy _π_ and the state-action pair ( _s, a_ ) _∈S × A_ , we then define the state-action value _H_ function in the timestep _h_ as _Q[π] h_[(] _[s, a]_[) =][ E] _[π] i_ = _h[r][i]_[(] _[s][i][, a][i]_[)] _[|][s][h]_[=] _[ s, a][h]_[=] _[ a]_ , and the value func�� � 

_H_ tion _Vh[π]_[(] _[s]_[) =][ E] _[π] i_ = _h[r][i]_[(] _[s][i][, a][i]_[)] _[|][s][h]_[=] _[ s]_ . The expectation E _π_ here is integrated into randomness �� � throughout the trajectory, which is essentially induced by the random action of the policy _ai ∼ π_ ( _·|si_ ) and the time-homogeneous transition dynamics of the environment _si_ +1 _∼ T_ ( _·|si, ai_ ) _, ∀i ∈_ [ _h, H_ ]. In the offline dataset, the data rollouts can be seen as generated by some (mixed) behavior policy _πβ_ , resulting in a dataset _D_ with in total _n_ samples _{si, ai, s[′] i[, r][i][}]_[1] _[≤][i][≤][n]_[.] 

2 

**Definition 1** (Bilinear MDP [14]) **.** For each ( _s, a_ ) _∈S × A, s[′] ∈S_ , we have the corresponding feature vector _ϕ_ ( _·, ·_ ) : R _[|][S][|] ×_ R _[|][A][|] →_ R _[d] , µ_ ( _·_ ) : R _[|S|] →_ R _[d][′]_ . With some core matrix _M ∈_ R _[d][×][d][′]_ , we can represent the transition function kernel _T_ ( _·|·, ·_ ) as 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_BECAUSE_Bilinear_Causal_Representation_for_Generalizable_Offline_Model-based_Reinforcement_Learning_images/nips_2024_BECAUSE_Bilinear_Causal_Representation_for_Generalizable_Offline_Model-based_Reinforcement_Learning.pdf-0003-01.png)


where _ϕ_ ( _s, a_ ) and _µ_ ( _s[′]_ ) are embedding functions that map the original state and action to the latent space, _M_ is the core matrix that models the transition relationship between the previous timestep and next timestep in the latent space. Such a linear decomposition in the transition dynamics allows us to embed structures of the transition model without the loss of general function approximation capabilities to derive state and action representations. 

## **2.2 Action State Confounded MDP** 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_BECAUSE_Bilinear_Causal_Representation_for_Generalizable_Offline_Model-based_Reinforcement_Learning_images/nips_2024_BECAUSE_Bilinear_Causal_Representation_for_Generalizable_Offline_Model-based_Reinforcement_Learning.pdf-0003-04.png)


**----- Start of picture text -----**<br>
Causal Path State Nodes Confounders<br>Confounder Path Action Nodes<br> 𝑠  𝑠′  𝑢  𝑢!<br> 𝑠  𝑠′  𝑠  𝑠′<br> 𝑎  𝑢  𝑎  𝑢""  𝑎   𝑢!′!′′<br>(a) Confounded MDP (b) SC-MDP (c) ASC-MDP (Ours)<br>**----- End of picture text -----**<br>


We consider the existence of confounders in the MDP to represent the offline data collection process, and define action-state confounded MDP (ASC-MDP): 

**Definition 2** (ASC-MDP) **.** Besides the components in standard MDPs _M_ = � _S, A, T, H, r_ �, 𝑎 𝑢 𝑎 𝑢"" 𝑎 𝑢!′!′′ we introduce a set of unobserved confounders (a) Confounded MDP (b) SC-MDP (c) ASC-MDP (Ours) _u_ . In ASP-MDP, confounders are factorized as _u_ = _{uπ, uc}_ 1 _≤h≤H_ , where _uπ ∈U_ denotes Figure 2: Comparison of our ASC-MDP with two exthe confounders between _s_ and _a ∼ πβ_ ( _s_ ) inisting formulations. duced by behavior policies, and _uc ∈U_ denotes the confounders within the state-action pairs of the environment transition, that is, the inherent structure between ( _s, a_ ) and _s[′]_ . Here we assume a timeinvariant confounder distribution _u ∼ Pu_ ( _·_ ), _∀h ∈_ [ _H_ ], which is a common assumption [15, 16, 17] 

The resulting causal relationship of ASC-MDP is demonstrated in Figure 2. Originating from the original MDP, ASC-MDP is different from the Confounded MDP [18] and State-Confounded MDP (SC-MDP) [19] in that it models both the _spurious correlation_ between the current state _s_ and the current action _a_ , as well as those between the next state _s[′]_ and ( _s, a_ ). Yet, confounded MDP and SC-MDP only model part of the possible confounders between states and actions. The factorization of the confounder in ASC-MDP aligns with the source of spurious correlation in offline MBRL. 

## **3 Proposed Method: BECAUSE** 

We propose BECAUSE, our core methodology for modeling, learning, and applying our causal representations for generalizable offline MBRL. Section 3.1 models the basic format of causal representations and analyzes their properties. Section 3.2 gives a compact way to learn the causal representation _ϕ_ ( _s, a_ ) and _µ_ ( _s[′]_ ), as well as the core mask estimation _M_ . Section 3.3 utilizes these learned causal representations in both world model learning and MBRL planning from offline datasets. 

## **3.1 Causal Representation for ASC-MDP** 

In the presence of a hidden confounder _u_ , we model the confounder behind the transition dynamics as a linear confounded MDP [18]: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_BECAUSE_Bilinear_Causal_Representation_for_Generalizable_Offline_Model-based_Reinforcement_Learning_images/nips_2024_BECAUSE_Bilinear_Causal_Representation_for_Generalizable_Offline_Model-based_Reinforcement_Learning.pdf-0003-12.png)


where _u ∼ Pu_ ( _·_ ). Inspired by the Bilinear MDP in Definition 1, we decompose _ϕ_[�] ( _s, a, u_ ) into a confounder-aware core matrix _M_ ( _u_ ) and a feature mapping _ϕ_ ( _s, a_ ), which factorize the influence of the confounders. Given the factorization of confounder _u_ = _{uc, uπ}_ in Definition 2, we derive via _d-separation_ in the graphical model in Figure 2 that _s[′] ⊥⊥ uπ|{s, a, uc}_ . As a result, we only need to consider the confounder _uc_ from the environment when decomposing the transition model: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_BECAUSE_Bilinear_Causal_Representation_for_Generalizable_Offline_Model-based_Reinforcement_Learning_images/nips_2024_BECAUSE_Bilinear_Causal_Representation_for_Generalizable_Offline_Model-based_Reinforcement_Learning.pdf-0003-14.png)


0 _d×d M_ **Definition 3** (Construction of causal graph _G_ ) **.** In ASC-MDP, _G_ = . for all (sparse) �0 _[d][′][×][d]_ 0 _[d][′][×][d][′]_ � 

core matrix _M_ , the causal graph _G_ is bipartite, thus _∀ G, G ∈_ DAG. 

3 

Definition 3 reveals the connection between the core matrix _M_ and causal graph _G_ , as is formulated in the ASC-MDP. To reduce the influence of _u_ and estimate the unconfounded transition model _T_ ( _s[′] |s, a_ ), one way is to identify the causal structures induced by the confounder _uc_ for the transition dynamics [20]. Existing methods in differentiable causal discovery [21, 22, 23] transform causal discovery on some causal graph _G_ , into a maximum likelihood estimation (MLE) with regularization: 

� � _G_ = arg max log _p_ ( _D_ ; _ϕ, µ, G_ ) _− λ|G|_ = _⇒ M_ = arg max log _p_ ( _D_ ; _ϕ, µ, M_ ) _− λ|M |,_ (4) _G∈_ DAG _M_ 

Since in our case, _M_ is a sub-matrix of the causal graph _G_ . Given the Definition 3, _G_ automatically satisfies the formulation of ASC-MDP, thus discovering _G_ is essentially estimating the sparse submatrix _M_ **without** DAG constraints: _M_[�] = arg max _M_ log _p_ ( _D_ ; _ϕ, µ, M_ ) _− λ|M |_ . We elaborate Definition 3 and show the relationship between core matrix _M_ and causal graph _G_ in Appendix A.2. 

We also make the assumption that the sparse _G_ and _M_ remain invariant with different environment confounders in the offline training and online testing. 

**Assumption 1** (Invariant causal graph) **.** We denote the causal graph _G_ under confounder _u_ as _G_ ( _u_ ). The generalization problem that we aim to solve satisfies the invariance in the causal graph _G_ ( _uc_ ), where _G_ ( _uc_ ) = _G_ ( _u[′] c_[)] _[, M]_[(] _[u][c]_[) =] _[ M]_[(] _[u][′] c_[)][,] _[ u][c]_[and] _[ u][′] c_[are the confounders in training and testing.] _Remark_ 1 _._ The assumption 1 can also be interpreted as task independence in [24], invariant state representation [25], or invariant action effect in [26]. 

## **3.2 Learning Causal Representation from Offline Data** 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_BECAUSE_Bilinear_Causal_Representation_for_Generalizable_Offline_Model-based_Reinforcement_Learning_images/nips_2024_BECAUSE_Bilinear_Causal_Representation_for_Generalizable_Offline_Model-based_Reinforcement_Learning.pdf-0004-06.png)


**----- Start of picture text -----**<br>
Uncertainty start ···<br>···<br>Break correlation Planning …<br>··· goal<br>𝑠, 𝑎<br> 𝜙(⋅,⋅)<br>Offline 𝑢<br>Buffer 𝑀 [)]<br>𝑇 ["] (𝑠 [!] |𝑠, 𝑎) 𝐸!(s, a)<br>  𝑠′ World Model Uncertainty Quantifier<br>𝜇(⋅)<br>**----- End of picture text -----**<br>


We first learn the causal world model _T_ ( _s[′] |s, a_ ) in the presence of confounders _u_ in the offline datasets. As formulated in ASC-MDP 2, there are two sets of confounders: _uπ_ and _uc_ . To estimate an unconfounded transition model and remove the effect of confounder, we first remove the impact of _uc_ which comes from the dynamics shift by estimating a batch-wise transition matrix _M_ ( _uc_ ), then we apply a reweighting formula to deconfound _uπ_ induced by the behavior policies and mitigate the model objective mismatch. 

Figure 3: BECAUSE learns a causality-aware representation from the buffer and uses it in both the world model and uncertainty quantification to obtain a pessimistic planning policy. 

As discussed in Definition 3, we only need to simistic planning policy. optimize the part of the parameters of the causal graph _G_ , i.e. _M_ . Thus, we can remove the constraints in (4), then transform the original causal discovery problem into a regularized MLE problem as follows: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_BECAUSE_Bilinear_Causal_Representation_for_Generalizable_Offline_Model-based_Reinforcement_Learning_images/nips_2024_BECAUSE_Bilinear_Causal_Representation_for_Generalizable_Offline_Model-based_Reinforcement_Learning.pdf-0004-10.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_BECAUSE_Bilinear_Causal_Representation_for_Generalizable_Offline_Model-based_Reinforcement_Learning_images/nips_2024_BECAUSE_Bilinear_Causal_Representation_for_Generalizable_Offline_Model-based_Reinforcement_Learning.pdf-0004-11.png)


where _Kµ_ :=[�] _s[′] ∈S[µ]_[(] _[s][′]_[)] _[µ]_[(] _[s][′]_[)] _[T]_[is an invertible matrix.][The derivation of equation (5) is elaborated] in Appendix A.3. In practice, we use the _χ_[2] -test for discrete state and action space and the fast Conditional Independent Test (CIT) [27] for continuous variables to estimate each entry in the core matrix _M_ . We regularize the sparsity of _M_ by controlling the _p_ -value threshold in CIT and provide a more detailed implementation in Appendix C.1. Estimating the core mask provides a more accurate relationship between state and action representations, and we further refine the state action representation function _ϕ_ and _µ_ to help capture more accurate transition dynamics. We optimize them by solving the following problem, according to the transition model loss and spectral norm regularization [28] to satisfy the regularity constraints of the feature in Assumption 3: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_BECAUSE_Bilinear_Causal_Representation_for_Generalizable_Offline_Model-based_Reinforcement_Learning_images/nips_2024_BECAUSE_Bilinear_Causal_Representation_for_Generalizable_Offline_Model-based_Reinforcement_Learning.pdf-0004-13.png)


The world model learning process is illustrated in Figure 3. The estimation of individual _M_ ( _uc_ ) mitigates the spurious correlation brought by _uc_ . To further deal with the spurious correlation in 

4 

_uπ_ induced by the behavior policy _πβ_ ( _a|s, uπ_ ), we utilize the conditional independence property in the ASC-MDP shown in equation (3). The following equation shows that the true unconfounded transition _T_ can be rewritten in the reweighting formulas. This reweighting process in mask _M_ serves as the soft intervention approach [18, 29] to estimate the treatment effect in the transition function _T_ in an unconfounded way: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_BECAUSE_Bilinear_Causal_Representation_for_Generalizable_Offline_Model-based_Reinforcement_Learning_images/nips_2024_BECAUSE_Bilinear_Causal_Representation_for_Generalizable_Offline_Model-based_Reinforcement_Learning.pdf-0005-01.png)


The derivation of equation (7) is illustrated in Appendix A.4. equation (7) basically shows a re-weighting process given the empirical estimation of _M_ ( _uc_ ) in every batch of trajectories: _M_ ( _u_ ) =[E] _[pu]_[[] E _[M] pu_[(] _[u] πβ[c]_[)] ( _[π] a[β] |s,u_[(] _[a][|] π[s][,]_ ) _[u][π]_[)]] . Compared to the general reweighting strategies in previous MBRL literatures [18, 29] which reweights the entire value function, this re-weighting process is conducted only on the estimated matrix, while the representation _ϕ_ ( _s, a_ ) and _µ_ ( _s[′]_ ) are subsequently regularized by weighted estimation of _M_ . The pipeline of causal world model learning is described in the first part of the Algorithm 1. We discuss more details of the implementation and experiment in Appendix C. 

## **3.3 Causal Representation for Uncertainty Quantification** 

To avoid entering OOD states in the **Algorithm 1:** BECAUSE Training and Planning online deployment, we further design **Input:** Offline dataset _D_ , causal discovery frequency _k_ a pessimistic planner according to the uncertainty of the predicted trajecto- **Output:** Causal mask _M_ , feature function _ϕ,_[�] � _µ_ , policy � _π_ ries in the imagination rollout step to `// Causal world model learning` mitigate objective mismatch. _M_ 0 _←_ [1] _[d][′][×][d]_ We use the feature embedding from **for** _i ∈_ [ _K_ ] **do** bilinear causal representation to help Update _ϕ_[�] _n,_ � _µn_ by _L_ rep( _ϕ, µ_ ) in (6) quantify the uncertainty, denoted as **if** _i_ mod _k = 0_ **then** _Eθ_ ( _s, a_ ). As we have access to the Update _M_[�] _n_ with _L_ mask( _M_ ) in (5) offline dataset, we learn an EnergyWeighted average _M n_ with (7) based Model (EBM) [30, 31] based `// Uncertainty quantifier learning` on the abstracted state representation Fit _Eθ_ ( _s, a_ ) with _L_ EBM (8) _ϕ_ put of the energy function and core matrix _M_ . A higher out- _Eθ_ ( _·, ·_ ) in- `//` Initialize `Pessimistic` _V_[�] _H_ +1( _s_ ) = 0 `planning` _, ∀_ ( _s, a_ ) dicates a higher uncertainty in the cur- **while** _h < H_ **do** rent state as they are visited by the beEstimate the uncertainty with score _Eθ_ ( _s, a_ ) havior policies _πβ_ less frequently. In Compute� _Qh_ ( _s, a_ ) with (9) practice, the energy-based model usu- _Vh_ ( _s, a_ ) = max _a Q_ ( _s, a_ ) ally suffers from a high-dimensional data space [32]. To mitigate this over- _a ←_ arg max _a Q_ ( _s, a_ ) head of training a good uncertainty _s[′] , r ←_ env.step ( _a_ , _g_ ) quantifier, we first embed the state samples through the abstract representation _µ_ ( _s[′]_ ), and the state action pair via _ϕ_ ( _s, a_ ). 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_BECAUSE_Bilinear_Causal_Representation_for_Generalizable_Offline_Model-based_Reinforcement_Learning_images/nips_2024_BECAUSE_Bilinear_Causal_Representation_for_Generalizable_Offline_Model-based_Reinforcement_Learning.pdf-0005-05.png)


where _µ_ ( _**s**_ )[+] refers to the positive samples from the approximated transition dynamics _T_[�] ( _·|s, a_ ), and _µ_ ( _**s[−]**_ ) refers to the latent negative samples via the Langevin dynamics [30]. Additionally, we regularize the parameters of EBM to avoid overfitting issues. We attach more training details and results of EBMs in Appendix C.2 The learned energy function _Eθ_ ( _s, a_ ) is used to quantify the uncertainty based on the offline data. 

During the online planning stage, we use the learned EBM to adjust the reward estimation based on Model Predictive Control (MPC) [33]. At timestep _h_ , we basically subtract the original step return estimation _rh_ ( _s, a_ ) by its uncertainty _Eθ_ ( _s, a_ ): 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_BECAUSE_Bilinear_Causal_Representation_for_Generalizable_Offline_Model-based_Reinforcement_Learning_images/nips_2024_BECAUSE_Bilinear_Causal_Representation_for_Generalizable_Offline_Model-based_Reinforcement_Learning.pdf-0005-08.png)


5 

## **3.4 Theoretical Analysis of BECAUSE** 

Then we move on to develop the theoretical analysis for the proposed method BECAUSE. Based on two standard Assumption 2 and 3 on the feature’s existence and regularity, we achieve the finite-sample complexity guarantee — an upper bound of the suboptimality gap as follows, whose proof is postponed to Appendix B. 

**Theorem 1** (Performance guarantee) **.** _Consider any_ 0 _< δ <_ 1 _and any initial state_ � _s ∈S. Under the Assumption 2, 3 and that the transition model T is an SCM (defined in 4), for any accuracy level_ 0 _≤ ξ ≤_ 1 _, with probability at least_ 1 _− δ, the output policy π of BECAUSE (Algorithm 1) based on the historical dataset D with n_ =[�] ( _s,a_ ) _∈S×A[n]_[(] _[s, a]_[)] _[ samples generated from a behavior policy][ π][β] satisfies:_ 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_BECAUSE_Bilinear_Causal_Representation_for_Generalizable_Offline_Model-based_Reinforcement_Learning_images/nips_2024_BECAUSE_Bilinear_Causal_Representation_for_Generalizable_Offline_Model-based_Reinforcement_Learning.pdf-0006-03.png)


_where C_ 1 _, Cs are some universal constants, σ is SCM’s noise level (see Definition 4), and M ∈_ R _[d][×][d][′] is the optimal ground truth sparse transition matrix to be estimated._ 

The error bound shrinks as the offline sample size _n_ over all state-action pairs increase. It also grows proportionally to the planning horizon _H_ , SCM’s noise level _σ_ , and the _ℓ_ 0 norm of the ground true causal mask _M_ , which describes the intrinsic complexity of the world model. 

Consequently, with Proposition 1 in the Appendix, we can achieve _ξ_ -optimal policy ( _V_ 1 _[∗]_[(] _[s]_[�][)] _[−][V]_ 1 _[π]_[(] _[s]_[�][)] _[ ≤] ξ_ ) as long as the historical dataset satisfies the following conditions: _∀_ 0 _≤ ξ ≤_ 1, 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_BECAUSE_Bilinear_Causal_Representation_for_Generalizable_Offline_Model-based_Reinforcement_Learning_images/nips_2024_BECAUSE_Bilinear_Causal_Representation_for_Generalizable_Offline_Model-based_Reinforcement_Learning.pdf-0006-07.png)


## **4 Experiment Results** 

In this section, we conduct a comprehensive empirical evaluation of BECAUSE’s generalization performance in a diverse set of environments, covering different decision-making problems in the grid world, manipulation, and autonomous driving domains, shown in Appendix Figure 8. 

## **4.1 Experiment Setting** 

**Environment Design** We design 18 tasks in 3 representative RL environments. Agents need to acquire reasoning capabilities to receive higher rewards and achieve goals. 

- **Lift** : Object manipulation environment in RoboSuite [34]. We designed this environment for the agent to lift an object with a specific color configuration on the table to a desired height. In the OOD environment _Lift-O_ , there is an injected spurious correlation between the color of the cube and the position of the cube in the training phase. During the testing phase, the correlation between color and position is different from training. 

- **Unlock** : We designed this environment for the agent to collect a key to open doors in Minigrid [35]. In the OOD environment _Unlock-O_ , there will be a different number of goals (doors to be opened) in the testing environments from the training environments. 

- **Crash** : Safety is critical in autonomous driving, which is reflected by the collision avoidance capability. We consider a risky scenario where an AV collides with a jaywalker because its view is blocked by another car [36]. We design such a crash scenario based on highway-env [37], where the goal is to create crashes between a pedestrian and AVs. In the OOD environment _Crash-O_ , the distribution of reward (number of pedestrians) is different in online testing environments. 

For all three different environments, we set a specific subset of the state space as the goal _g ∈S_ , and the reward is defined as the goal-reaching reward _r_ ( _s, a, g_ ) = I( _r_ = _g_ ). When the episode ends in the goal state within the task horizon _H_ , the episode is considered a success. We then use the average _success rate_ as the general evaluation metrics for our BECAUSE and all baselines. 

In each environment, we collect three types of offline data: _random_ , _medium_ , and _expert_ based on the different levels of _uπ_ in the behavior policies. In _Unlock_ environments, we collect 200 episodes 

6 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_BECAUSE_Bilinear_Causal_Representation_for_Generalizable_Offline_Model-based_Reinforcement_Learning_images/nips_2024_BECAUSE_Bilinear_Causal_Representation_for_Generalizable_Offline_Model-based_Reinforcement_Learning.pdf-0007-00.png)


**----- Start of picture text -----**<br>
Lift-Random Lift-Medium Lift-Expert Unlock-Random Unlock-Random<br>Unlock-Random Unlock-Medium Unlock-Expert Unlock-Medium Unlock-Medium<br>Crash-Random Crash-Medium Crash-Expert Unlock-Expert Unlock-Expert<br>(a) (b) (c)<br>**----- End of picture text -----**<br>


Figure 4: Results of BECAUSE and baselines in different tasks. (a) Average success rate in distribution and out of distribution. (b) Average success rate w.r.t. ratio of offline samples. (c) Average success rate w.r.t. spurious level in the environments. We evaluate the mean and standard deviation of the best performance among 10 random seeds and report task-wise results in Appendix Table 3. 

from each level of behavior policies as the offline demonstration data, and the number of episodes is 1,000 in the environments _Lift_ and _Crash_ , which all have continuous state and action space. A more detailed view of the environment hyperparameters and behavior policy design is in Appendix C.5. 

**Baselines** We compare our proposed BECAUSE with several offline causal RL or MBRL baselines. **ICIL** [38] learns a dynamic-aware invariant causal representation learning to assist a generalizable policy learning from offline datasets. **CCIL** [39] conducts a soft intervention in our offline setting by jointly optimizing policy parameters and masks over the state. **MnM** [9] unifies the objective of jointly training the model and policy, which allocates larger weights in the state prediction loss in the high-reward region. **Delphic** [40] introduces delphic uncertainty to differentiate between uncertainties caused by hidden confounders and traditional epistemic and aleatoric uncertainties. **TD3+BC** [41] is an offline model-free RL approach that combines the Twin Delayed Deep Deterministic Policy Gradient (TD3) algorithm with Behavior Cloning (BC) to adopt both the actor-critic framework and supervised learning from expert demonstrations. **MOPO** [2] is an offline MBRL approach that uses flat latent space and count-based uncertainty quantification to maintain conservatism in online deployment. **GNN** [42] is a GNN-based baseline using a Relational Graph Convolutional Network to model the temporal dependency of state-action pairs in the dynamic model with message passing. **CDL** [24] uses causal discovery to learn a task-independent world model. **Denoised MDP** [12] and **IFactor** [13] conduct causal state abstraction based on their controllability and reward relevance. The last three methods are designed for online settings, so we only implement their model learning objectives. We attach more details of the baseline implementation in Appendix C.6. 

## **4.2 Experiment Results Analysis** 

We empirically answer the following research questions. 

- **RQ1** : How is the generalizability of BECAUSE in the online environments (which may be _unseen_ )? Specifically, how does BECAUSE perform under diverse qualities of demonstration data (different level of _uπ_ ), and different environment contexts (different _uc_ )? 

- **RQ2** : How does the design in BECAUSE contribute to the robustness of its final performance under different sample sizes or spurious levels? 

- **RQ3** : How does BECAUSE achieve the aforementioned generalizability by mitigating the objective mismatch problem in offline MBRL? 

7 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_BECAUSE_Bilinear_Causal_Representation_for_Generalizable_Offline_Model-based_Reinforcement_Learning_images/nips_2024_BECAUSE_Bilinear_Causal_Representation_for_Generalizable_Offline_Model-based_Reinforcement_Learning.pdf-0008-00.png)


**----- Start of picture text -----**<br>
Unlock-random Unlock-medium Unlock-expert<br>Negative Reward<br>Positive Reward<br>Example Policies<br>MOPO<br>BECAUSE<br>**----- End of picture text -----**<br>


Figure 5: Evaluation of the difference between the distribution of episodic model loss for success and failure trajectories. The higher difference indicates a reduction in model mismatch issues. An example of failure mode is trying to open the door without having the key. 

For **RQ1** , in Figure 4(a), we evaluate the success rate in the online environment against different baselines. The result shows that under different environments and different qualities of behavior policies _πβ_ (different _uπ_ ), BECAUSE consistently achieves the best performance in 8 out of 9 for both the in-distribution (I) and out-of-distribution (O) for all the demonstration data quality (different level of _uπ_ ). Where O here indicates the tasks under _unseen_ environment with confounder _u[′] c_[=] _[ u][c]_ different from offline training. Another finding is that model-based approaches generally perform better than model-free approaches at various levels of offline data, which shows the importance of world model learning for generalizable offline RL. We attach the detailed results in the Appendix Table 3, 4 and the causal masks discovered in each environment in Appendix Figure 9 for reference. 

For **RQ2** , we compare different aspects of BECAUSE’s robustness with MOPO without causal structures [2]. We compare their performance with different ratios of the entire offline dataset and illustrate the success rates in Figure 4(b). The result shows that, for any selected number of samples, BECAUSE consistently outperforms MOPO with a clear margin. We also evaluate BE- 

Figure 6: The ablation studies between BECAUSE and its variants. We report the overall Success rate (%) over 9 in-distribution (I) and 9 out-of-distribution (O) tasks, respectively. **Bold** is the best. 

||Variants|BECAUSE|Optimism|Linear|Full|
|---|---|---|---|---|---|
||Overall-I<br>Overall-O|**73.3**_±_**4.5**<br>**43.0**_±_**4.9**|64.4_±_6.4<br>32.4_±_3.3|57.9_±_6.1<br>33.2_±_5.2|39.3_±_6.3<br>25.2_±_3.9|



CAUSE performance at higher spurious levels in Figure 4(c). We add up to 8 _×_ of the original number of confounders in the environments to test the robustness of the agent’s performance. BECAUSE consistently outperforms MOPO and the margin enlarges as the spurious level grows higher. 

For **RQ3** , we aim to understand whether BECAUSE achieves higher performance by resolving the objective mismatch problem. We first collect two groups of trajectories: _**τpos**_ and _**τneg**_ , each with positive reward (success) and negative reward (failure) in _Unlock_ task with sparse goal-reaching reward. We want to have a model whose loss is informative for discriminating control results, that is, we wish _Lmodel_ ( _**τpos**_ ) _< Lmodel_ ( _**τneg**_ ). According to our visualization in Figure 5, in _Unlock-Expert_ and _Unlock-Medium_ , the ratio of _**τpos**_ is much higher in BECAUSE than MOPO among the trajectories with low model loss. In _Unlock-Random_ , the mismatch of the model and control objective is more significant, since the demonstration is poor in state coverage. MOPO cannot succeed even when the model loss is low, whereas our methods can. We perform a hypothesis test with _H_ 0 : _Lmodel_ ( _**τpos**_ ) _< Lmodel_ ( _**τneg**_ ). In BECAUSE, this desired property is more significant than MOPO attributed to the causal representation we learn, indicating a reduction of objective mismatch. We attach detailed discussions for the mismatch evaluation in Appendix C.3 and Table 2. 

## **4.3 Ablation Studies** 

We conducted ablation studies with three variants of BECAUSE and report the average success rate across nine in-distribution and nine out-of-distribution tasks in Table 6. The **Optimism** variant conducts optimistic planning instead of pessimistic planning in equation (9), which uses uniform sampling in the planner module. The **Linear** variant assumes a full connection to the causal matrix _M_ , then directly uses linear MDP to parameterize the dynamics model _T_ , which removes the causal discovery module in BECAUSE. The **Full** variant learns from the full batch of data to estimate the 

8 

causal mask without iterative update. We report the results of the task-wise ablation with confidence interval and significance in Appendix Table 5 and 6. 

## **5 Related Works** 

**Objective Mismatch in MBRL** The objective mismatch in MBRL [43, 44] refers to the fact that pure MLE estimation of the world model does not align well with the control objective. Previous works [29, 45] propose reweighting during model training to alleviate this mismatch, [46] proposes a goal-aware prediction by redistributing model error according to their task relevance. These works essentially reweight loss for the entire model training, while our work conducts reweighting just over the estimated causal mask more efficiently. More recently, [9, 10] proposed a joint training between the world model and policies. Although joint optimization improves performance, they do not address the generalizability of the learned model under the distribution shift setting. In the offline setting, Model-based RL [2, 3, 4] employs model ensemble, pessimistic policy optimization or value iteration [47, 48], and an energy-based model for planning [49] to quantify uncertainty and improve test performance. To the best of our knowledge, no previous work explored or modeled the impact of distribution shift on the objective mismatch problem in MBRL. 

**Causal Discovery with Confounder** Most of the existing causal discovery methods [50] can be categorized into constraint-based and score-based. Constraint-based methods [51] start from a complete graph and iteratively remove edges with statistical hypothesis testing [52, 53]. This type of method is highly data-efficient but not robust to noisy data. As a remedy, score-based methods [54, 55] use metrics such as the likelihood or BIC [56] as scores to manipulate edges in the causal graph. Recently, researchers have extended score-based methods with RL [57], order learning [58] or differentiable discovery [22, 59, 60]. To alleviate the non-identifiability under hidden confounders, active intervention methods have been explored [61], aiming to break spurious correlations in an online fashion. With extra assumptions on confounders, some recent works detect such correlations [62, 63, 64] so that models can effectively identify elusive confounders. 

**Causal Reinforcement Learning** Recently, many RL algorithms have incorporated causality to improve reasoning capability [65] and generalizability. For instance, [66] and [67] explicitly estimate causal structures with the interventional data obtained from the environment in an online setting. These structures can be used to constrain the output space [19] or to adjust the buffer priority [68]. Building dynamic models in model-based RL [24, 69, 70] based on causal graphs is widely studied. Most existing causal MBRL works focus on estimating the causal world model by predicting transition dynamics and rewards. Existing methods learn this causal world model via sparsity regularization [23, 71], conditional independence test [24, 69, 72], variational inference [73, 12], counterfactual data augmentation [74, 75], hierarchical skill abstraction [76, 77], uncertainty quantification [40], reward redistribution [24, 78], causal context modeling [79, 80] and structureaware state abstraction [12, 13, 81, 82] based on the controllability and task or reward relevance. However, the presence of confounders during data collection can skew the learned policy, making it susceptible to spurious correlations. Deconfounding solutions have been proposed either between actions and states [39, 83, 84] or among different dimensions of state variables [19, 85]. 

## **6 Conclusion** 

In this paper, we study how to mitigate the objective mismatch problem in MBRL, especially under the offline settings where distribution shift occurs. We first propose ASC-MDP and the bilinear causal representation associated with it. Based on the formulation, we proposed how to learn this causal abstraction by alternating between causal mask learning and feature learning in fitting the world dynamics. In the planning stage, we applied the learned causal representation to an uncertainty quantification module based on EBM, which improves the robustness under uncertainty in the online planning stage. We theoretically justify BECAUSE’s sub-optimality bound induced by the sparse matrix estimation problem and offline RL. Comprehensive experiments on 18 different tasks show that given a diverse level of demonstration as the offline dataset, BECAUSE has better generalizability than baselines in different online environments, and it robustly outperforms baselines under different spurious levels or sample sizes. We empirically show that BECAUSE mitigates the objective mismatch with causal awareness learned from offline data. One limitation of BECAUSE lies in its simplified assumption of time-homogeneous causal structure, which may not always hold in long-horizon or non-stationary settings. Besides, the current implementation is still based on vector observations. It will be interesting to scale up the causal reasoning framework into high-dimensional observations to discover concept factors in long-horizon visual RL settings. 

9 

## **References** 

- [1] Sergey Levine, Aviral Kumar, George Tucker, and Justin Fu. Offline reinforcement learning: Tutorial, review, and perspectives on open problems. _arXiv preprint arXiv:2005.01643_ , 2020. 

- [2] Tianhe Yu, Garrett Thomas, Lantao Yu, Stefano Ermon, James Y Zou, Sergey Levine, Chelsea Finn, and Tengyu Ma. Mopo: Model-based offline policy optimization. _Advances in Neural Information Processing Systems_ , 33:14129–14142, 2020. 

- [3] Rahul Kidambi, Aravind Rajeswaran, Praneeth Netrapalli, and Thorsten Joachims. Morel: Model-based offline reinforcement learning. In H. Larochelle, M. Ranzato, R. Hadsell, M.F. Balcan, and H. Lin, editors, _Advances in Neural Information Processing Systems_ , volume 33, pages 21810–21823. Curran Associates, Inc., 2020. 

- [4] Tianhe Yu, Aviral Kumar, Rafael Rafailov, Aravind Rajeswaran, Sergey Levine, and Chelsea Finn. Combo: Conservative offline model-based policy optimization. _Advances in neural information processing systems_ , 34:28954–28967, 2021. 

- [5] Zeyu Zhu and Huijing Zhao. A survey of deep rl and il for autonomous driving policy learning. _IEEE Transactions on Intelligent Transportation Systems_ , 23(9):14043–14065, 2021. 

- [6] Aviral Kumar, Anikait Singh, Frederik Ebert, Mitsuhiko Nakamoto, Yanlai Yang, Chelsea Finn, and Sergey Levine. Pre-training for robots: Offline rl enables learning new tasks from a handful of trials. _arXiv preprint arXiv:2210.05178_ , 2022. 

- [7] Shengpu Tang and Jenna Wiens. Model selection for offline reinforcement learning: Practical considerations for healthcare settings. In _Machine Learning for Healthcare Conference_ , pages 2–35. PMLR, 2021. 

- [8] Laixi Shi and Yuejie Chi. Distributionally robust model-based offline reinforcement learning with near-optimal sample complexity. _arXiv preprint arXiv:2208.05767_ , 2022. 

- [9] Benjamin Eysenbach, Alexander Khazatsky, Sergey Levine, and Russ R Salakhutdinov. Mismatched no more: Joint model-policy optimization for model-based rl. _Advances in Neural Information Processing Systems_ , 35:23230–23243, 2022. 

- [10] Shentao Yang, Shujian Zhang, Yihao Feng, and Mingyuan Zhou. A unified framework for alternating offline model training and policy learning. _Advances in Neural Information Processing Systems_ , 35:17216–17232, 2022. 

- [11] Jonas Peters, Dominik Janzing, and Bernhard Schölkopf. _Elements of causal inference: foundations and learning algorithms_ . The MIT Press, 2017. 

- [12] Tongzhou Wang, Simon S Du, Antonio Torralba, Phillip Isola, Amy Zhang, and Yuandong Tian. Denoised mdps: Learning world models better than the world itself. _arXiv preprint arXiv:2206.15477_ , 2022. 

- [13] Yu-Ren Liu, Biwei Huang, Zhengmao Zhu, Honglong Tian, Mingming Gong, Yang Yu, and Kun Zhang. Learning world models with identifiable factorization. _arXiv preprint arXiv:2306.06561_ , 2023. 

- [14] Lin Yang and Mengdi Wang. Reinforcement learning in feature space: Matrix bandit, kernels, and regret bound. In _International Conference on Machine Learning_ , pages 10746–10756. PMLR, 2020. 

- [15] Weitong Zhang, Jiafan He, Dongruo Zhou, Q Gu, and A Zhang. Provably efficient representation selection in low-rank markov decision processes: from online to offline rl. In _Uncertainty in Artificial Intelligence_ , pages 2488–2497. PMLR, 2023. 

- [16] Fan Feng and Sara Magliacane. Learning dynamic attribute-factored world models for efficient multi-object reinforcement learning. _arXiv preprint arXiv:2307.09205_ , 2023. 

- [17] Angela Zhou. Reward-relevance-filtered linear offline reinforcement learning. _arXiv preprint arXiv:2401.12934_ , 2024. 

- [18] Lingxiao Wang, Zhuoran Yang, and Zhaoran Wang. Provably efficient causal reinforcement learning with confounded observational data. _Advances in Neural Information Processing Systems_ , 34:21164–21175, 2021. 

- [19] Wenhao Ding, Laixi Shi, Yuejie Chi, and Ding Zhao. Seeing is not believing: Robust reinforcement learning against spurious correlation. _arXiv preprint arXiv:2307.07907_ , 2023. 

10 

- [20] Junzhe Zhang and Elias Bareinboim. Markov decision processes with unobserved confounders: A causal approach. Technical report, Technical report, Technical Report R-23, Purdue AI Lab, 2016. 

- [21] Karren Yang, Abigail Katcoff, and Caroline Uhler. Characterizing and learning equivalence classes of causal dags under interventions. In _International Conference on Machine Learning_ , pages 5541–5550. PMLR, 2018. 

- [22] Philippe Brouillard, Sébastien Lachapelle, Alexandre Lacoste, Simon Lacoste-Julien, and Alexandre Drouin. Differentiable causal discovery from interventional data. _Advances in Neural Information Processing Systems_ , 33:21865–21877, 2020. 

- [23] Zizhao Wang, Xuesu Xiao, Yuke Zhu, and Peter Stone. Task-independent causal state abstraction. _Workshop on Robot Learning: Self-Supervised and Lifelong Learning, NeurIPS_ , 2021. 

- [24] Zizhao Wang, Xuesu Xiao, Zifan Xu, Yuke Zhu, and Peter Stone. Causal dynamics learning for task-independent state abstraction. _arXiv preprint arXiv:2206.13452_ , 2022. 

- [25] Amy Zhang, Clare Lyle, Shagun Sodhani, Angelos Filos, Marta Kwiatkowska, Joelle Pineau, Yarin Gal, and Doina Precup. Invariant causal prediction for block mdps. In _International Conference on Machine Learning_ , pages 11214–11224. PMLR, 2020. 

- [26] Wenxuan Zhu, Chao Yu, and Qiang Zhang. Causal deep reinforcement learning using observational data. _arXiv preprint arXiv:2211.15355_ , 2022. 

- [27] Krzysztof Chalupka, Pietro Perona, and Frederick Eberhardt. Fast conditional independence test for vector variables with large sample sizes. _arXiv preprint arXiv:1804.02747_ , 2018. 

- [28] Yuichi Yoshida and Takeru Miyato. Spectral norm regularization for improving the generalizability of deep learning. _arXiv preprint arXiv:1705.10941_ , 2017. 

- [29] Nathan Lambert, Brandon Amos, Omry Yadan, and Roberto Calandra. Objective mismatch in model-based reinforcement learning. _arXiv preprint arXiv:2002.04523_ , 2020. 

- [30] Yilun Du and Igor Mordatch. Implicit generation and modeling with energy based models. _Advances in Neural Information Processing Systems_ , 32, 2019. 

- [31] Yezhen Wang, Bo Li, Tong Che, Kaiyang Zhou, Ziwei Liu, and Dongsheng Li. Energy-based open-world uncertainty modeling for confidence calibration. In _Proceedings of the IEEE/CVF International Conference on Computer Vision_ , pages 9302–9311, 2021. 

- [32] Bo Pang, Tian Han, Erik Nijkamp, Song-Chun Zhu, and Ying Nian Wu. Learning latent space energy-based prior model. _Advances in Neural Information Processing Systems_ , 33:21994– 22008, 2020. 

- [33] Eduardo F Camacho and Carlos Bordons Alba. _Model predictive control_ . Springer science & business media, 2013. 

- [34] Yuke Zhu, Josiah Wong, Ajay Mandlekar, Roberto Martín-Martín, Abhishek Joshi, Soroush Nasiriany, and Yifeng Zhu. robosuite: A modular simulation framework and benchmark for robot learning. In _arXiv preprint arXiv:2009.12293_ , 2020. 

- [35] Maxime Chevalier-Boisvert, Lucas Willems, and Suman Pal. Minimalistic gridworld environment for openai gym. `https://github.com/maximecb/gym-minigrid` , 2018. 

- [36] Zenna Tavares, James Koppel, Xin Zhang, Ria Das, and Armando Solar-Lezama. A language for counterfactual generative models. In _International Conference on Machine Learning_ , pages 10173–10182. PMLR, 2021. 

- [37] Edouard Leurent. An environment for autonomous driving decision-making. `https://github. com/eleurent/highway-env` , 2018. 

- [38] Ioana Bica, Daniel Jarrett, and Mihaela van der Schaar. Invariant causal imitation learning for generalizable policies. _Advances in Neural Information Processing Systems_ , 34:3952–3964, 2021. 

- [39] Pim De Haan, Dinesh Jayaraman, and Sergey Levine. Causal confusion in imitation learning. _Advances in Neural Information Processing Systems_ , 32, 2019. 

- [40] Alizée Pace, Hugo Yèche, Bernhard Schölkopf, Gunnar Ratsch, and Guy Tennenholtz. Delphic offline reinforcement learning under nonidentifiable hidden confounding. In _The Twelfth International Conference on Learning Representations_ , 2023. 

11 

- [41] Scott Fujimoto and Shixiang Shane Gu. A minimalist approach to offline reinforcement learning. _Advances in neural information processing systems_ , 34:20132–20145, 2021. 

- [42] Michael Schlichtkrull, Thomas N Kipf, Peter Bloem, Rianne van den Berg, Ivan Titov, and Max Welling. Modeling relational data with graph convolutional networks. In _European semantic web conference_ , pages 593–607. Springer, 2018. 

- [43] Amir-massoud Farahmand, Andre Barreto, and Daniel Nikovski. Value-aware loss function for model-based reinforcement learning. In _Artificial Intelligence and Statistics_ , pages 1486–1494. PMLR, 2017. 

- [44] Yuping Luo, Huazhe Xu, Yuanzhi Li, Yuandong Tian, Trevor Darrell, and Tengyu Ma. Algorithmic framework for model-based deep reinforcement learning with theoretical guarantees. _arXiv preprint arXiv:1807.03858_ , 2018. 

- [45] Pierluca D’Oro, Alberto Maria Metelli, Andrea Tirinzoni, Matteo Papini, and Marcello Restelli. Gradient-aware model-based policy search. In _Proceedings of the AAAI Conference on Artificial Intelligence_ , volume 34, pages 3801–3808, 2020. 

- [46] Suraj Nair, Silvio Savarese, and Chelsea Finn. Goal-aware prediction: Learning to model what matters. In _International Conference on Machine Learning_ , pages 7207–7219. PMLR, 2020. 

- [47] Chi Jin, Zhuoran Yang, Zhaoran Wang, and Michael I Jordan. Provably efficient reinforcement learning with linear function approximation. In _Conference on Learning Theory_ , pages 2137– 2143. PMLR, 2020. 

- [48] Masatoshi Uehara and Wen Sun. Pessimistic model-based offline reinforcement learning under partial coverage. _arXiv preprint arXiv:2107.06226_ , 2021. 

- [49] Yilun Du, Shuang Li, Joshua Tenenbaum, and Igor Mordatch. Learning iterative reasoning through energy minimization. In _International Conference on Machine Learning_ , pages 5570– 5582. PMLR, 2022. 

- [50] Clark Glymour, Kun Zhang, and Peter Spirtes. Review of causal discovery methods based on graphical models. _Frontiers in genetics_ , 10:524, 2019. 

- [51] Peter Spirtes, Clark N Glymour, Richard Scheines, and David Heckerman. _Causation, prediction, and search_ . MIT press, 2000. 

- [52] Karl Pearson. X. on the criterion that a given system of deviations from the probable in the case of a correlated system of variables is such that it can be reasonably supposed to have arisen from random sampling. _The London, Edinburgh, and Dublin Philosophical Magazine and Journal of Science_ , 50(302):157–175, 1900. 

- [53] Kun Zhang, Jonas Peters, Dominik Janzing, and Bernhard Schölkopf. Kernel-based conditional independence test and application in causal discovery. _arXiv preprint arXiv:1202.3775_ , 2012. 

- [54] David Maxwell Chickering. Optimal structure identification with greedy search. _Journal of machine learning research_ , 3(Nov):507–554, 2002. 

- [55] Alain Hauser and Peter Bühlmann. Characterization and greedy learning of interventional markov equivalence classes of directed acyclic graphs. _The Journal of Machine Learning Research_ , 13(1):2409–2464, 2012. 

- [56] Andrew A Neath and Joseph E Cavanaugh. The bayesian information criterion: background, derivation, and applications. _Wiley Interdisciplinary Reviews: Computational Statistics_ , 4(2):199–203, 2012. 

- [57] Shengyu Zhu, Ignavier Ng, and Zhitang Chen. Causal discovery with reinforcement learning. _arXiv preprint arXiv:1906.04477_ , 2019. 

- [58] Dezhi Yang, Guoxian Yu, Jun Wang, Zhengtian Wu, and Maozu Guo. Reinforcement causal structure learning on order graph. In _Proceedings of the AAAI Conference on Artificial Intelligence_ , volume 37, pages 10737–10744, 2023. 

- [59] Nan Rosemary Ke, Olexa Bilaniuk, Anirudh Goyal, Stefan Bauer, Hugo Larochelle, Bernhard Schölkopf, Michael C Mozer, Chris Pal, and Yoshua Bengio. Learning neural causal models from unknown interventions. _arXiv preprint arXiv:1910.01075_ , 2019. 

- [60] Yunzhu Li, Antonio Torralba, Anima Anandkumar, Dieter Fox, and Animesh Garg. Causal discovery in physical systems from videos. _Advances in Neural Information Processing Systems_ , 33:9180–9192, 2020. 

12 

- [61] Nino Scherrer, Olexa Bilaniuk, Yashas Annadani, Anirudh Goyal, Patrick Schwab, Bernhard Schölkopf, Michael C Mozer, Yoshua Bengio, Stefan Bauer, and Nan Rosemary Ke. Learning neural causal models with active interventions. _arXiv preprint arXiv:2109.02429_ , 2021. 

- [62] Christopher Clark, Mark Yatskar, and Luke Zettlemoyer. Don’t take the easy way out: Ensemblebased methods for avoiding known dataset biases. _arXiv preprint arXiv:1909.03683_ , 2019. 

- [63] Divyansh Kaushik, Eduard Hovy, and Zachary C Lipton. Learning the difference that makes a difference with counterfactually-augmented data. _arXiv preprint arXiv:1909.12434_ , 2019. 

- [64] Meike Nauta, Ricky Walsh, Adam Dubowski, and Christin Seifert. Uncovering and correcting shortcut learning in machine learning models for skin cancer diagnosis. _Diagnostics_ , 12(1):40, 2021. 

- [65] Prashan Madumal, Tim Miller, Liz Sonenberg, and Frank Vetere. Explainable reinforcement learning through a causal lens. In _Proceedings of the AAAI conference on artificial intelligence_ , volume 34, pages 2493–2500, 2020. 

- [66] Suraj Nair, Yuke Zhu, Silvio Savarese, and Li Fei-Fei. Causal induction from visual observations for goal directed tasks. _arXiv preprint arXiv:1910.01751_ , 2019. 

- [67] Sergei Volodin, Nevan Wichers, and Jeremy Nixon. Resolving spurious correlations in causal models of environments via interventions. _arXiv preprint arXiv:2002.05217_ , 2020. 

- [68] Maximilian Seitzer, Bernhard Schölkopf, and Georg Martius. Causal influence detection for improving efficiency in reinforcement learning. _Advances in Neural Information Processing Systems_ , 34, 2021. 

- [69] Zheng-Mao Zhu, Xiong-Hui Chen, Hong-Long Tian, Kun Zhang, and Yang Yu. Offline reinforcement learning with causal structured world models. _arXiv preprint arXiv:2206.01474_ , 2022. 

- [70] Mirco Mutti, Riccardo De Santi, Emanuele Rossi, Juan Felipe Calderon, Michael Bronstein, and Marcello Restelli. Provably efficient causal model-based reinforcement learning for systematic generalization. In _Proceedings of the AAAI Conference on Artificial Intelligence_ , volume 37, pages 9251–9259, 2023. 

- [71] Fan Feng, Biwei Huang, Kun Zhang, and Sara Magliacane. Factored adaptation for nonstationary reinforcement learning. _Advances in Neural Information Processing Systems_ , 35:31957–31971, 2022. 

- [72] Jiaheng Hu, Zizhao Wang, Peter Stone, and Roberto Martin-Martin. Elden: Exploration via local dependencies. _arXiv preprint arXiv:2310.08702_ , 2023. 

- [73] Wenhao Ding, Haohong Lin, Bo Li, and Ding Zhao. Generalizing goal-conditioned reinforcement learning with variational causal reasoning. _Advances in Neural Information Processing Systems_ , 35:26532–26548, 2022. 

- [74] Lars Buesing, Theophane Weber, Yori Zwols, Nicolas Heess, Sebastien Racaniere, Arthur Guez, and Jean-Baptiste Lespiau. Woulda, coulda, shoulda: Counterfactually-guided policy search. In _International Conference on Learning Representations_ , 2018. 

- [75] Silviu Pitis, Elliot Creager, Ajay Mandlekar, and Animesh Garg. Mocoda: Model-based counterfactual data augmentation. _arXiv preprint arXiv:2210.11287_ , 2022. 

- [76] Tabitha E Lee, Jialiang Alan Zhao, Amrita S Sawhney, Siddharth Girdhar, and Oliver Kroemer. Causal reasoning in simulation for structure and transfer learning of robot manipulation policies. In _2021 IEEE International Conference on Robotics and Automation (ICRA)_ , pages 4776–4782. IEEE, 2021. 

- [77] Tabitha Edith Lee, Shivam Vats, Siddharth Girdhar, and Oliver Kroemer. Scale: Causal learning and discovery of robot manipulation skills using simulation. In _Conference on Robot Learning_ , pages 2229–2256. PMLR, 2023. 

- [78] Yudi Zhang, Yali Du, Biwei Huang, Ziyan Wang, Jun Wang, Meng Fang, and Mykola Pechenizkiy. Interpretable reward redistribution in reinforcement learning: A causal approach. _Advances in Neural Information Processing Systems_ , 36, 2024. 

- [79] Biwei Huang, Fan Feng, Chaochao Lu, Sara Magliacane, and Kun Zhang. Adarl: What, where, and how to adapt in transfer reinforcement learning. In _International Conference on Learning Representations_ , 2021. 

13 

- [80] Peide Huang, Xilun Zhang, Ziang Cao, Shiqi Liu, Mengdi Xu, Wenhao Ding, Jonathan Francis, Bingqing Chen, and Ding Zhao. What went wrong? closing the sim-to-real gap via differentiable causal discovery. In _Conference on Robot Learning_ , pages 734–760. PMLR, 2023. 

- [81] Xiang Fu, Ge Yang, Pulkit Agrawal, and Tommi Jaakkola. Learning task informed abstractions. In _International Conference on Machine Learning_ , pages 3480–3491. PMLR, 2021. 

- [82] Biwei Huang, Chaochao Lu, Liu Leqi, José Miguel Hernández-Lobato, Clark Glymour, Bernhard Schölkopf, and Kun Zhang. Action-sufficient state representation learning for control with structural constraints. In _International Conference on Machine Learning_ , pages 9260–9279. PMLR, 2022. 

- [83] Zhihong Deng, Zuyue Fu, Lingxiao Wang, Zhuoran Yang, Chenjia Bai, Zhaoran Wang, and Jing Jiang. Score: Spurious correlation reduction for offline reinforcement learning. _arXiv preprint arXiv:2110.12468_ , 2021. 

- [84] Maxime Gasse, Damien Grasset, Guillaume Gaudron, and Pierre-Yves Oudeyer. Using confounded data in latent model-based reinforcement learning. _Transactions on Machine Learning Research_ , 2023. 

- [85] Junzhe Zhang, Daniel Kumor, and Elias Bareinboim. Causal imitation learning with unobserved confounders. _Advances in neural information processing systems_ , 33:12263–12274, 2020. 

- [86] Ying Jin, Zhuoran Yang, and Zhaoran Wang. Is pessimism provably efficient for offline rl? In _International Conference on Machine Learning_ , pages 5084–5096. PMLR, 2021. 

- [87] Thomas Blumensath and Mike E Davies. Iterative hard thresholding for compressed sensing. _Applied and computational harmonic analysis_ , 27(3):265–274, 2009. 

- [88] Amir Beck and Marc Teboulle. A fast iterative shrinkage-thresholding algorithm for linear inverse problems. _SIAM journal on imaging sciences_ , 2(1):183–202, 2009. 

- [89] Peter Auer, Thomas Jaksch, and Ronald Ortner. Near-optimal regret bounds for reinforcement learning. _Advances in neural information processing systems_ , 21, 2008. 

- [90] Ryan Tibshirani and Larry Wasserman. Sparsity, the lasso, and friends. _Lecture notes from “Statistical Machine Learning,” Carnegie Mellon University, Spring_ , 2017. 

- [91] Patrick E McKnight and Julius Najab. Mann-whitney u test. _The Corsini encyclopedia of psychology_ , pages 1–1, 2010. 

14 

## **A Auxiliary Details of BECAUSE Framework** 

## **A.1 Notation Summary** 

We illustrate all the notations used in the main paper and appendix in Table 1. 

Table 1: Notations used in this paper and their corresponding meanings. 

|Notation|Explanation|
|---|---|
|||
|_A_,_a_<br>_S_,_s_<br>_r_<br>_γ_<br>_T_(_·|·, ·_)<br>_h_,_H_<br>_π∗_<br>_πβ_<br>_V_(_·_)<br>_Q_(_·, ·_)<br>_D_<br>_n_(_s, a_)<br>B_h_|Action space, action<br>State space, state<br>Reward<br>Discount factor<br>Transition dynamics<br>Timestep_h_, Horizon_H_<br>Optimal policy in the online environments<br>Behavior policy generating the offine datasets<br>State value function<br>Action-state value function<br>Offine datasets<br>Number of samples for (s, a) pairs in offine datasets<br>Bellman operator|
|||
|_ϕ_(_·, ·_)<br>�_ϕ_(_·, ·_)<br>_µ_(_·_)<br>_d_,_d′_<br>_Kµ_<br>_Cϕ_,_Cµ, Cβ_<br>_Cs_<br>_κ_<br>_X_<br>_u, uc, uπ_<br>_M_,_M_(_u_)<br>_G_<br>�<br>_M_,<br>_M_(_u_)<br>_βM_, �_βM_<br>**PA**_G_(_·_)<br>_ϵ_<br>_λ_,_λϕ_,_λµ_<br>_σ_<br>_ξ_<br>_K_|Feature representation of state and action<br>Feature of state, action and confounder in equation (2)<br>Feature representation of next state<br>Dimensions of features for_ϕ_(_·, ·_)and_µ_(_·_)<br>Feature matrix expanded from_µ_in equation (5)<br>Feature regularity<br>Sparsity-related constant<br>Restrictive eigenvalue constant<br>Feature Kronecker product<br>Confounders<br>Binary transition matrix (under certain confounders)<br>Causal graph<br>Estimated causal matrix<br>Optimal / estimated parameters in causal matrix<br>Parental node in the causal graph_G_<br>Exogenous noise in SCM by Defnition 4<br>Spectrum regularizer weight in equation (6)<br>Standard deviation of exogenous noise in SCM<br>Accuracy level of the policy<br>Iterative update steps in BECAUSE|
|||
|_Eθ_<br>_δ_<br>Γ(_·, ·_)<br>_E_<br>_λ_EBM|Energy-based model<br>Level of ‘high probability’<br>Uncertainty quantifcation function<br>_δ_-uncertainty quantifer set<br>Regularizer weight for the_ℓ_2norm in EBM|



## **A.2 Derivation of Definition 3** 

0 _d×d M_ The node of this causal graph _G_ = contains two groups of entities: (1) The state �0 _[d][×][d][′]_ 0 _[d][×][d]_ � action abstraction _ϕ_ ( _s, a_ ), and (ii) the next state abstraction _µ_ ( _s[′]_ ). We denote _ϕ_ ( _·, ·_ )[(] _[i]_[)] as the _i[th]_ factor in the abstracted state action representations, and _µ_ ( _·_ )[(] _[j]_[)] for the _j[th]_ factor in the abstracted state representations. 

15 

The source node of all the nodes _G_ is _ϕ_ ( _s, a_ )[(] _[i]_[)] , which is the abstracted state-action representation, and the sink node of all the edges in _G_ is the to _µ_ ( _s_ ) _[′]_[(] _[i]_[)] . 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_BECAUSE_Bilinear_Causal_Representation_for_Generalizable_Offline_Model-based_Reinforcement_Learning_images/nips_2024_BECAUSE_Bilinear_Causal_Representation_for_Generalizable_Offline_Model-based_Reinforcement_Learning.pdf-0016-01.png)


Therefore, _G_ is a bipartite graph, since there will be no edges between _ϕ_ ( _s, a_ )[(] _[i]_[)] _, ϕ_ ( _s, a_ )[(] _[j]_[)] , or _µ_ ( _s_ )[(] _[i]_[)] _, µ_ ( _s_ )[(] _[j]_[)] . 

Consequently, we show that _G ∈_ DAG. 

## **A.3 Derivation of equation (5)** 

**Definition 4** (Structured Causal Model) **.** An SCM _θ_ := ( _S, E_ ) consists of a collection _S_ of _d_ functions [11], 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_BECAUSE_Bilinear_Causal_Representation_for_Generalizable_Offline_Model-based_Reinforcement_Learning_images/nips_2024_BECAUSE_Bilinear_Causal_Representation_for_Generalizable_Offline_Model-based_Reinforcement_Learning.pdf-0016-06.png)


where **PA** _[G] j[⊂{][s]_[1] _[, . . . , s][d][}\{][s][j][}]_[ are called parents of] _[ x][j]_[in the Directed Acyclic Graph (DAG)] _[ G]_[,] and _E_ = _{ϵi}[d] i_ =1[are jointly independent.][For instance,][in continuous state and action space,][we] parameterize the world model with joint Gaussian Distribution, i.e. _ϵ ∼N_ (0 _, σIdd′_ ). 

We then use bilinear MDP to approximate the original likelihood function in equation (4), i.e. 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_BECAUSE_Bilinear_Causal_Representation_for_Generalizable_Offline_Model-based_Reinforcement_Learning_images/nips_2024_BECAUSE_Bilinear_Causal_Representation_for_Generalizable_Offline_Model-based_Reinforcement_Learning.pdf-0016-09.png)


where _Kµ_ :=[�] _s[′] ∈S[µ]_[(] _[s][′]_[)] _[µ]_[(] _[s][′]_[)] _[T]_[is an invertible matrix.][Then we can apply an MLE in equation (5).] In our BECAUSE algorithm, the optimization of the causal world model is conducted by solving the regularized MLE problem in equation (13). The biggest difference between BECAUSE and the offline version of [14, 15] is that it aims to apply _ℓ_ 0 regression instead of ridge regression to estimate matrix _M_ : 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_BECAUSE_Bilinear_Causal_Representation_for_Generalizable_Offline_Model-based_Reinforcement_Learning_images/nips_2024_BECAUSE_Bilinear_Causal_Representation_for_Generalizable_Offline_Model-based_Reinforcement_Learning.pdf-0016-11.png)


## **A.4 Proof of equation (7)** 

The derivation depends on the following re-weighting formula in [18]: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_BECAUSE_Bilinear_Causal_Representation_for_Generalizable_Offline_Model-based_Reinforcement_Learning_images/nips_2024_BECAUSE_Bilinear_Causal_Representation_for_Generalizable_Offline_Model-based_Reinforcement_Learning.pdf-0016-14.png)


Then we apply equation (14) to the decomposition in equation (2) and equation (3), which yields 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_BECAUSE_Bilinear_Causal_Representation_for_Generalizable_Offline_Model-based_Reinforcement_Learning_images/nips_2024_BECAUSE_Bilinear_Causal_Representation_for_Generalizable_Offline_Model-based_Reinforcement_Learning.pdf-0016-16.png)


where the last equality holds by letting _M_ ( _u_ ) :=[E] _[pu]_ E[[] _[M] pu_[(] [ _[u] πβ[c]_[)] ( _[π] a[β] |s,u_[(] _[a][|] π[s][,]_ )] _[u][π]_[)]] . 

16 

## **B Proof of Theorem 1** 

In this section, we provide the proof of the sub-optimality upper bound in Theorem 1. We first show some useful definitions and lemmas in Section B.1. Armed with them, we provide the theoretical results tailored for the causal discovery setting in Section B.2. Furthermore, we give a detailed proof of the uncertainty set form in our causal discovery problems in Section B.3. 

## **B.1 Preliminary** 

In this subsection, we first define the _δ_ -uncertainty quantifier Γ, then we refer to the lemmas in the previous literature to construct a suboptimality bound based on the defined uncertainty quantifier Γ. 

First, we define the Bellman operator B _h_ , for some value function _V_ : _S �→_ R, the Bellman operator can be defined as: 

(B _hV_ )( _s, a_ ) = E[ _rh_ ( _sh, ah_ ) + _V_ ( _sh_ +1) _|sh_ = _s, ah_ = _a_ ] _._ (16) 

Similarly, we denote the approximate Bellman operator of the empirical MDP constructed from the offline dataset _D_ as B[�] _h_ for any _h ∈_ [ _H_ ]. 

**Definition 5** ( _δ_ -Uncertainty Quantifier) **.** We let _{_ Γ _h}[H] h_ =1[,][ Γ] _[h]_[:] _[S][× A][�→]_[R][ to be a] _[ δ]_[-uncertainty] quantifier with respect to data distribution _PD_ if the event: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_BECAUSE_Bilinear_Causal_Representation_for_Generalizable_Offline_Model-based_Reinforcement_Learning_images/nips_2024_BECAUSE_Bilinear_Causal_Representation_for_Generalizable_Offline_Model-based_Reinforcement_Learning.pdf-0017-08.png)


satisfies P _D_ ( _E_ ) _≥_ 1 _− δ_ . 

As we consider the offline model learning and planning, we define the model evaluation error at each step _h ∈_ [ _H_ ] as 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_BECAUSE_Bilinear_Causal_Representation_for_Generalizable_Offline_Model-based_Reinforcement_Learning_images/nips_2024_BECAUSE_Bilinear_Causal_Representation_for_Generalizable_Offline_Model-based_Reinforcement_Learning.pdf-0017-11.png)


where _ιh_ is the error induced by the approximate Bellman operator, especially the transition kernel based on _D_ . We then identify the source of sub-optimality in our offline MBRL setting by decomposing the sub-optimality error in Lemma 1. 

**Lemma 1** (Decomposition of Suboptimality [86]) **.** 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_BECAUSE_Bilinear_Causal_Representation_for_Generalizable_Offline_Model-based_Reinforcement_Learning_images/nips_2024_BECAUSE_Bilinear_Causal_Representation_for_Generalizable_Offline_Model-based_Reinforcement_Learning.pdf-0017-14.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_BECAUSE_Bilinear_Causal_Representation_for_Generalizable_Offline_Model-based_Reinforcement_Learning_images/nips_2024_BECAUSE_Bilinear_Causal_Representation_for_Generalizable_Offline_Model-based_Reinforcement_Learning.pdf-0017-15.png)


_where π is any learned policy, π[∗] is the optimal policy that maximizes the cumulative return as below:_ 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_BECAUSE_Bilinear_Causal_Representation_for_Generalizable_Offline_Model-based_Reinforcement_Learning_images/nips_2024_BECAUSE_Bilinear_Causal_Representation_for_Generalizable_Offline_Model-based_Reinforcement_Learning.pdf-0017-17.png)


Based on this decomposition, we will get the basic form of sub-optimality error bound for general offline RL settings in Lemma 2: 

**Lemma 2** (Suboptimality in standard MDP [86]) **.** _Suppose we have {_ Γ _h}[H] h_ =1 _[as][δ][-uncertainty] quantifier. Under E defined in equation (5), the suboptimality error bound by conservative planning satisfies:_ 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_BECAUSE_Bilinear_Causal_Representation_for_Generalizable_Offline_Model-based_Reinforcement_Learning_images/nips_2024_BECAUSE_Bilinear_Causal_Representation_for_Generalizable_Offline_Model-based_Reinforcement_Learning.pdf-0017-20.png)


The basic form of sub-optimality bound in Lemma 2 involves an uncertainty quantifier Γ _h_ , which in our case will be further replaced by an exact bound in our sparse matrix estimation problem of causal discovery algorithms. 

17 

## **B.2 Proof of Theorem 1** 

The main results hold under the following two assumptions: 

**Assumption 2** (Existence of a core matrix given the feature embedding) **.** For each ( _s, a_ ) _∈S × A_ , feature vectors _ϕ_ ( _s, a_ ) _∈_ R _[d] , µ_ ( _s_ ) _∈_ R _[d][′]_ are approximated as a priori. Given a specific confounder set _u_ , there exists an unknown matrix _M_ ( _u_ ) _[∗] ∈_ R _[d][′][×][d]_ such that, 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_BECAUSE_Bilinear_Causal_Representation_for_Generalizable_Offline_Model-based_Reinforcement_Learning_images/nips_2024_BECAUSE_Bilinear_Causal_Representation_for_Generalizable_Offline_Model-based_Reinforcement_Learning.pdf-0018-03.png)


**Assumption 3** (Feature regularity) **.** We assume feature regularity [14, 15] for the following components of the confounded bilinear MDP: 

- _∀u, ∥M_ ( _u_ ) _∥_[2] _F[≤][C][M][d]_[,] 

- _∀_ ( _s, a_ ) _∈S × A, ∥ϕ_ ( _s, a_ ) _∥_ 2[2] _[≤][C][ϕ][d]_[,] 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_BECAUSE_Bilinear_Causal_Representation_for_Generalizable_Offline_Model-based_Reinforcement_Learning_images/nips_2024_BECAUSE_Bilinear_Causal_Representation_for_Generalizable_Offline_Model-based_Reinforcement_Learning.pdf-0018-07.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_BECAUSE_Bilinear_Causal_Representation_for_Generalizable_Offline_Model-based_Reinforcement_Learning_images/nips_2024_BECAUSE_Bilinear_Causal_Representation_for_Generalizable_Offline_Model-based_Reinforcement_Learning.pdf-0018-08.png)


where _CM , Cϕ, Cµ, Cµ[′]_[are some universal constants.] 

Here, for any matrix _X_ , _∥X∥_ 2 _,∞_ := max _i_ �� _j[X] ij_[2][represents the operator][ 2] _[ �→∞]_[norm.] 

**Proof pipeline.** Armed with the above assumptions, we turn to the bilinear MDP setting, which this work focuses on. We shall develop the finite-sample analysis by specifying the main error term —- _δ_ -uncertainty quantifier Γ (see Lemma 2) for our time-homogeneous core matrix estimation problem in the following lemma. 

**Lemma 3** (Uncertainty bound for Bilinear Causal Representation) **.** _Under the Assumption 2, 3 and that T is an SCM (defined in 4), for the BECAUSE algorithm, for the ξ-optimal policy (V_ 1 _[∗]_[(] _[s]_[�][)] _[ −] V_ 1 _[π]_[(] _[s]_[�][)] _[ ≤][ξ][),][ ∀]_[0] _[ ≤][ξ][≤]_[1] _[, we have the][ δ][-uncertainty set as:]_ 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_BECAUSE_Bilinear_Causal_Representation_for_Generalizable_Offline_Model-based_Reinforcement_Learning_images/nips_2024_BECAUSE_Bilinear_Causal_Representation_for_Generalizable_Offline_Model-based_Reinforcement_Learning.pdf-0018-13.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_BECAUSE_Bilinear_Causal_Representation_for_Generalizable_Offline_Model-based_Reinforcement_Learning_images/nips_2024_BECAUSE_Bilinear_Causal_Representation_for_Generalizable_Offline_Model-based_Reinforcement_Learning.pdf-0018-14.png)


_where C_ 1 _is some universal constants._ 

Armed with the above lemma, we complete the proof of Theorem 1 by showing that 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_BECAUSE_Bilinear_Causal_Representation_for_Generalizable_Offline_Model-based_Reinforcement_Learning_images/nips_2024_BECAUSE_Bilinear_Causal_Representation_for_Generalizable_Offline_Model-based_Reinforcement_Learning.pdf-0018-17.png)


This concludes the proof of Theorem 1. 

## **B.3 Proof of Lemma 3** 

The key to proving Theorem 1 is to prove Lemma 3. The proof pipeline of Lemma 3 is illustrated below. In **Step 1** , we derive the estimation of the causal transition matrix _M_ in BECAUSE as a sparsity regression problem. In **Step 2** , we decompose the error terms within _δ_ -uncertainty set into two parts: (a) error due to the under-explored dataset, (b) error due to optimization error in the structured causal model. Then we bound both error terms in **Step 3** and **Step 4** , respectively. Finally, in **Step 5** , we sum up all the results and derive the form of _δ_ -uncertainty quantifier which will lead to our final results in Theorem 1. 

18 

**Step 1: deriving the output model of BECAUSE.** Recalling the original optimization problem in equation (13) to estimate the core matrix: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_BECAUSE_Bilinear_Causal_Representation_for_Generalizable_Offline_Model-based_Reinforcement_Learning_images/nips_2024_BECAUSE_Bilinear_Causal_Representation_for_Generalizable_Offline_Model-based_Reinforcement_Learning.pdf-0019-01.png)


This part of derivation aims to transform the above estimation problem into a linear regression problem, with the regression data pairs ( _X, T_ ) and some unknown parameters _β_ associated with mask _M_ to be estimated. Eventually, we’ll derive the representation of each part of _β[M] , X, T_ , and eventually reach the following form: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_BECAUSE_Bilinear_Causal_Representation_for_Generalizable_Offline_Model-based_Reinforcement_Learning_images/nips_2024_BECAUSE_Bilinear_Causal_Representation_for_Generalizable_Offline_Model-based_Reinforcement_Learning.pdf-0019-03.png)


We define each component of this target form of _ℓ_ 0 regression as follows: 

- **For unknown parameters** _β[M]_ : We first define _β[M] ∈_ [0 _,_ 1] _[dd][′]_ as a column dimensional vector consisting of all the entries in time-homogenous causal matrix _M_ , where _βi[M]_ denotes the _i_ -th entry of _β[M]_ . Besides, we define _βD[M]_[as the true core matrix given some offline dataset] _[ D]_[ and] corresponding data pairs _Tdata, Xdata_ that satisfies _Tdata_ = _βD[M][X][data]_[ +] _[ ϵ]_[.] 

- **For dataset** _D_ : Recall the transition pairs in the offline dataset _D_ = _{si, ai, s[′] i[}]_[1] _[≤][i][≤][n]_[.][Here,] _[ n]_ represents the sample size over certain state-action pairs in the rollout data by some behavior policy _πβ_ . For simplicity, we denote _n_ ≜ _n_ ( _s, a_ ) in the following derivation, which is mentioned in Section 2.1. 

- **For regression target** _Tπβ_ : Then, we introduce the following transition targets _Tπβ_ induced by the offline dataset _D_ sampled with behavior policy _πβ_ : 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_BECAUSE_Bilinear_Causal_Representation_for_Generalizable_Offline_Model-based_Reinforcement_Learning_images/nips_2024_BECAUSE_Bilinear_Causal_Representation_for_Generalizable_Offline_Model-based_Reinforcement_Learning.pdf-0019-08.png)


Under the _n_ finite samples in the offline dataset, we assume that _Tπβ ∼N_ (E[ _Tπβ_ ] _, σ_[2] _In_ ). The above definition specifies the regression target in the _ℓ_ 0 regression problem, and we denote _Tπβ_ = [ _Tπβ_ ( _s[′]_ 1 _[|][s]_[1] _[, a]_[1][)] _[,][ · · ·][, T][π] β_[(] _[s] n[′][|][s][n][, a][n]_[)]] _[T][∈]_[R] _[n]_[as the empirical transition probabilities of] certain transition pairs in the offline data _D_ = _{si, ai, s[′] i[}]_[1] _[≤][i][≤][n]_[.] 

- **For regression data** _X_ : Next, we need to specify the data _X_ in the regression problem. We denote the i-th row of _X_ as the i-th sample in the offline transition pairs _Xi ∈D_ , which is a _i_[)] 

- vector of Kronecker product between _ϕ_ ( _si, ai_ ) _∈_ R _[d]_ and normalized _[µ] C_[(] _[s] µ[′] ∈_ R _[d][′]_ (without loss of generality, we assume _Cϕ_ = 1 and only need to normalize _µ_ ( _s[′] i_[)][ by] _[ C][µ]_[):] 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_BECAUSE_Bilinear_Causal_Representation_for_Generalizable_Offline_Model-based_Reinforcement_Learning_images/nips_2024_BECAUSE_Bilinear_Causal_Representation_for_Generalizable_Offline_Model-based_Reinforcement_Learning.pdf-0019-11.png)


As a result, _Xi ∈_ R _[dd][′]_ , since there are in all _n_ samples in offline dataset, _X ∈_ R _[n][×][dd][′]_ is the dataset-dependent matrix with all _n_ rows of samples, and _d_ and _d[′]_ are the latent dimension of _ϕ_ and _µ_ , respectively. Based on the feature regularity criteria in Assumption 3, we have _∥Xi∥_ 2 _≤∥Xi∥_ 1 _≤_ 1 _, ∥X∥∞ ≤_ 1. 

The prior work [14] estimate the transition kernel of a bilinear MDP using the following ridge regression: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_BECAUSE_Bilinear_Causal_Representation_for_Generalizable_Offline_Model-based_Reinforcement_Learning_images/nips_2024_BECAUSE_Bilinear_Causal_Representation_for_Generalizable_Offline_Model-based_Reinforcement_Learning.pdf-0019-14.png)


19 

In this paper, in order to promote the sparsity of the matrix _M_ , we introduce the _ℓ_ 0 regularization term and arrive at the following optimization problem: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_BECAUSE_Bilinear_Causal_Representation_for_Generalizable_Offline_Model-based_Reinforcement_Learning_images/nips_2024_BECAUSE_Bilinear_Causal_Representation_for_Generalizable_Offline_Model-based_Reinforcement_Learning.pdf-0020-01.png)


where we denote the solution associated with the offline dataset _D_ as _β_[�] _D[M]_[.][Here, we use the empirical] version constructed by the finite samples in offline dataset _D_ . 

Given the goal-conditioned reward setting, for a single episode _s ∼ τ_ , _r_ ( _s, a_ ; _g_ ) = 1 if and only if _s_ = _g_ , otherwise _r_ ( _s, a_ ; _g_ ) = 0, as is specified in Section 2.1. Since we are essentially predicting the probabilities (normalized to a sum of 1) of whether the next state is the goal state, i.e.[�] _s∈S[V]_[�][ (] _[s]_[) = 1][.] Therefore, we have _∥V_[�] ( _·_ ) _∥_ 1 _≤_ 1. 

As is denoted by equation (23), for the specific offline dataset collected by some behavior policies _πβ_ , �( _si,ai,s[′] i_[)] _[∈D]_ **[ 1]**[(] _[s][i]_[=] _[s,a][i]_[=] _[a,s] i[′]_[=] _[s][′]_[)] we have the regression target _Tπβ_ ( _s[′] |s, a_ ) = �( _si,ai,s[′] i_[)] _[∈D]_ **[ 1]**[(] _[s][i]_[=] _[s,a][i]_[=] _[a]_[)] , and the corresponding features induced by the dataset _Xi_ = _ϕ_ ( _si, ai_ ) _⊗[µ] C_[(] _[s] µi[′]_[)][.][We have the following equations hold:] 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_BECAUSE_Bilinear_Causal_Representation_for_Generalizable_Offline_Model-based_Reinforcement_Learning_images/nips_2024_BECAUSE_Bilinear_Causal_Representation_for_Generalizable_Offline_Model-based_Reinforcement_Learning.pdf-0020-05.png)


where _βD_ is the true underlying transition mask given the offline dataset _D_ , _ϵ ∼N_ (0 _, σ · In_ ) is some exogenous noise of the transition model. 

Specifically, for the regression problem, we transform the original trajectory dataset _D_ as [ _X, Tπβ_ ], as the representation of transition pairs rolled out by the behavior policy _πβ_ . Similarly, we can define some ’well-explored dataset’ _D[∗]_ , which is an infinite dataset rollout by the behavior policy _πβ_ , _D[∗]_ = _{si, ai, ri}[∞] i_ =0[,][similar][to][the][definition][of][regression][target] _[T]_[in][equation][(23)][and] representation data _X_ in equation (24), we have [ _X,_ E[ _Tπβ_ ]] as the regression pairs with access to the true transition probability distribution for all the state action pairs ( _s, a_ ). Here _X ∈_ R _[n][×][dd][′]_ is the regression data defined in equation (22). Recalling the definition in equation (23), we can further build the regression target under well-explored dataset _D[∗]_ as: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_BECAUSE_Bilinear_Causal_Representation_for_Generalizable_Offline_Model-based_Reinforcement_Learning_images/nips_2024_BECAUSE_Bilinear_Causal_Representation_for_Generalizable_Offline_Model-based_Reinforcement_Learning.pdf-0020-08.png)


We denote E[ _Tπβ_ ] = �E[ _Tπβ_ ( _s[′]_ 1 _[|][s]_[1] _[, a]_[1][)]] _[,][ · · ·]_[ E][[] _[T][π] β_[(] _[s] n[′][|][s][n][, a][n]_[)]] � _T ∈_ R _n_ . In practice, with finite sample size _n_ , we have E[ _Tπβ_ ( _s[′] |s, a_ )] = _T_ ( _s[′] |s, a_ ) + _ϵ_ = _XβD[M][∗]_[+] _[ ϵ]_[.][In addition, we also introduce] a vector form **T** _∈_ R _[|S||A|×|S|]_ so that _∀_ ( _s, a_ ) _∈S × A_ , we have **T** ( _·|s, a_ ) = � _T_ ( _s[′]_ 1 _[|][s, a]_[)] _T_ ( _s[′]_ 2 _[|][s, a]_[)] _· · · T_ ( _s[′] |S|[|][s, a]_[)] � _T ∈_ R _|S|,_ where the state space is denoted as _S_ = _{s[′]_ 1 _[,][ · · ·][, s] |S|[}]_[are][all][possible][states.][Similar][to][the] Kronecker product we define for _X_ in equation (26), we define **X** in a matrix form for any state-action pair ( _s, a_ ): 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_BECAUSE_Bilinear_Causal_Representation_for_Generalizable_Offline_Model-based_Reinforcement_Learning_images/nips_2024_BECAUSE_Bilinear_Causal_Representation_for_Generalizable_Offline_Model-based_Reinforcement_Learning.pdf-0020-10.png)


In addition, we let **X** ( _s[′] |s, a_ ) _∈_ R[1] _[×][dd][′]_ denote the _s[′]_ -th row of **X** ( _·|s, a_ ) associated with the state _s[′] ∈S_ . Consequently, the estimated transition kernel can be expressed as follows: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_BECAUSE_Bilinear_Causal_Representation_for_Generalizable_Offline_Model-based_Reinforcement_Learning_images/nips_2024_BECAUSE_Bilinear_Causal_Representation_for_Generalizable_Offline_Model-based_Reinforcement_Learning.pdf-0020-12.png)


**Step 2: decomposing the term of interest.** To begin with, recalling the definition of Bellman operator B _h_ in equation (16) and applying Hölder’s inequality, the term of interest for any time step 

20 

1 _≤ h ≤ H_ and state-action pair ( _s, a_ ) _∈S × A_ can be controlled as 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_BECAUSE_Bilinear_Causal_Representation_for_Generalizable_Offline_Model-based_Reinforcement_Learning_images/nips_2024_BECAUSE_Bilinear_Causal_Representation_for_Generalizable_Offline_Model-based_Reinforcement_Learning.pdf-0021-01.png)


where the first inequality is held given our goal-conditioned reward formulation in section 2.1. To continue, we have 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_BECAUSE_Bilinear_Causal_Representation_for_Generalizable_Offline_Model-based_Reinforcement_Learning_images/nips_2024_BECAUSE_Bilinear_Causal_Representation_for_Generalizable_Offline_Model-based_Reinforcement_Learning.pdf-0021-03.png)


Here, we recall _β_[�] _D[M]_[represents the parameter vector in the estimated causal masks based on the offline] dataset _D_ sampled by _πβ_ . Similarly, we denote _β_[�] _D[M][∗]_[as the estimated causal mask outputted from] equation (26) based on the infinite dataset _D[∗]_ generated by the behavior policy _πβ_ . Then, we can further control equation (30) as 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_BECAUSE_Bilinear_Causal_Representation_for_Generalizable_Offline_Model-based_Reinforcement_Learning_images/nips_2024_BECAUSE_Bilinear_Causal_Representation_for_Generalizable_Offline_Model-based_Reinforcement_Learning.pdf-0021-05.png)


Here the last inequality comes from the fact that 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_BECAUSE_Bilinear_Causal_Representation_for_Generalizable_Offline_Model-based_Reinforcement_Learning_images/nips_2024_BECAUSE_Bilinear_Causal_Representation_for_Generalizable_Offline_Model-based_Reinforcement_Learning.pdf-0021-07.png)


based on the definition of _X_ in equation (24) and assumption 3. Here ( _a_ ) comes from the mismatch error between the demonstrated offline dataset and some optimal rollout datasets. And ( _b_ ) comes from the error of the _ℓ_ 0 optimization of causal masks given the existence of exogenous noise _σ_ defined by SCM in Definition 4. We will control them separately in the following. 

**Step 3: Controlling term (a).** We need to consider the optimization process in the original regression problem in equation (22) to fully understand the difference between _β_[�] _D[M]_[and] _[β]_[�] _D[M][∗]_[, where] the only difference is that the latter uses a perfect dataset with infinite samples. The optimization problem we target (cf. equation (26)) can be solved by the iterative hard thresholding algorithm (IHT) proposed by [87] IHT offers an iterative solution for the _ℓ_ 0 regression problem, armed with a hard thresholding operator as below: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_BECAUSE_Bilinear_Causal_Representation_for_Generalizable_Offline_Model-based_Reinforcement_Learning_images/nips_2024_BECAUSE_Bilinear_Causal_Representation_for_Generalizable_Offline_Model-based_Reinforcement_Learning.pdf-0021-10.png)


We denote _β_[�] _D[M]_[(] _[i]_[)][as][the][estimated][causal][mask][parameters][after] _[i]_[-th][iterations][with][dataset] _[D]_[.] Similarly, we denote _β_[�] _D[M][∗]_[(] _[i]_[)][ as the estimation after] _[ i]_[-th iterations with dataset] _[ D][∗]_[.][We initialize the] graph to be a full graph regardless of the datasets ( _D[∗]_ or _D_ ) used in the optimization process, leading to _β_[�] _D[M][∗]_[(0) =] _[β]_[�] _D[M]_[(0) =] **[ 1]** _[ ∈]_[R] _[dd][′]_[.] 

Recall that _X ∈_ R _[n][×][dd][′]_ , _Tπβ ∈_ R _[n] ,_ E[ _Tπβ_ ] _∈_ R _[n]_ and _β_[�] _D[M]_[is][ [0] _[,]_[ 1]] _[dd][′]_[based on the definition in the] original _ℓ_ 0 optimization problem in equation (22), equation (23) and equation (28). The update rules of using either the dataset _D_ or _D[∗]_ can be written as: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_BECAUSE_Bilinear_Causal_Representation_for_Generalizable_Offline_Model-based_Reinforcement_Learning_images/nips_2024_BECAUSE_Bilinear_Causal_Representation_for_Generalizable_Offline_Model-based_Reinforcement_Learning.pdf-0021-13.png)


21 

Note that the difference between the two parameters _β_[�] _D[M]_[and] _[β]_[�] _D[M][∗]_[essentially relies on the difference] between two pairs of transition kernel estimation datasets [ _X,_ E[ _Tπβ_ ]] and [ _X, Tπβ_ ]. It is easily verified that the hard-thresholding operator [ _gλ_ ( _·_ )] _i_ defined in equation (32) is _L_ = 1-Lipschitz. According to [88], we can set the learning rate _η ≤ L_[1][= 1][ here.][Using the Lipschitz property, at each] iterative update step _i_ , we can control the difference between the estimated parameters obtained by using _D_ or _D[∗]_ as 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_BECAUSE_Bilinear_Causal_Representation_for_Generalizable_Offline_Model-based_Reinforcement_Learning_images/nips_2024_BECAUSE_Bilinear_Causal_Representation_for_Generalizable_Offline_Model-based_Reinforcement_Learning.pdf-0022-01.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_BECAUSE_Bilinear_Causal_Representation_for_Generalizable_Offline_Model-based_Reinforcement_Learning_images/nips_2024_BECAUSE_Bilinear_Causal_Representation_for_Generalizable_Offline_Model-based_Reinforcement_Learning.pdf-0022-02.png)


(34) 

Here, _Idd′_ represent the identity matrix of size _dd[′] × dd[′]_ , the last inequality holds based on the fact that _∥X∥_ 2 _≤_ 1 defined in equation (24). Since _η ≤ L_[1][=][1][, as a result,] _[ ∥][I][dd][′][−][ηX][T][ X][∥]_[2][=] max(1 _− ηλ_ ( _X[T] X_ )) _≤_ 1. We perform IHT for sufficient _K >_ 0 iterations and output the last step estimation as our solutions for either the offline dataset _D_ (we use for practical optimization) or the perfect dataset _D[∗]_ . In practice, for any dataset such as _D_ and any accuracy level 0 _≤ ξ ≤_ 1, we have the output of IHT gradually converges to the optimal solution _β_[�] _D[M]_[of][the][problem][in] equation (26). Namely, after at most _K ≃_ log � _∥Mξ∥_ 0 � steps [87, Corollary 1], the output satisfies _βDM[−][β]_[�] _D[M]_[(] _[K]_[)] 4[.][Similarly,][based][on][the][perfect][dataset] _[D][∗]_[,][the][output][of][IHT][gradually] ���� ���2 _[≤][ξ]_ converges to _β_[�] _D[M][∗]_[at the same rate.][Consequently, the term of interest (a) can be bounded recursively] as: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_BECAUSE_Bilinear_Causal_Representation_for_Generalizable_Offline_Model-based_Reinforcement_Learning_images/nips_2024_BECAUSE_Bilinear_Causal_Representation_for_Generalizable_Offline_Model-based_Reinforcement_Learning.pdf-0022-05.png)


where the second inequality holds by recursively applying equation (34) to _K, K −_ 1 _, · · · ,_ 0 iterations, the last equality is due to the fact that _β_[�] _D[M]_[(0)][=] _[β]_[�] _D[M][∗]_[(0)][,][and][the last inequality holds][by setting] _K_ = log( _[∥][M] ξ[∥]_[2] ). 

Now the remaining of the proof will focus on controlling _∥Tπβ −_ E[ _Tπβ_ ] _∥∞_ . Recall that we have defined the regression target in equation (23) and equation (28): 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_BECAUSE_Bilinear_Causal_Representation_for_Generalizable_Offline_Model-based_Reinforcement_Learning_images/nips_2024_BECAUSE_Bilinear_Causal_Representation_for_Generalizable_Offline_Model-based_Reinforcement_Learning.pdf-0022-08.png)


**Proposition 1** (Well-explored dataset) **.** _With the offline dataset D of in total n samples with n_ ( _s, a_ ) _samples generated independently conditioned on any_ ( _s, a_ ) _. For any_ 0 _< δ <_ 1 _, with probability at_ 

22 

_least_ 1 _− δ, one has_ 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_BECAUSE_Bilinear_Causal_Representation_for_Generalizable_Offline_Model-based_Reinforcement_Learning_images/nips_2024_BECAUSE_Bilinear_Causal_Representation_for_Generalizable_Offline_Model-based_Reinforcement_Learning.pdf-0023-01.png)


_for some universal constant Cβ._ 

The above proposition can be directly proved by applying [89, Lemma 17] over all ( _s, a_ ) _∈S × A_ : 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_BECAUSE_Bilinear_Causal_Representation_for_Generalizable_Offline_Model-based_Reinforcement_Learning_images/nips_2024_BECAUSE_Bilinear_Causal_Representation_for_Generalizable_Offline_Model-based_Reinforcement_Learning.pdf-0023-04.png)


As a result, we can further extend the results in equation (35) to bound the term (a) as follows: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_BECAUSE_Bilinear_Causal_Representation_for_Generalizable_Offline_Model-based_Reinforcement_Learning_images/nips_2024_BECAUSE_Bilinear_Causal_Representation_for_Generalizable_Offline_Model-based_Reinforcement_Learning.pdf-0023-06.png)


where _C_ 1 = _CβCµ_ is some constant that is related to the feature regularity, _ξ_ is the level of error tolerance in the cumulative value returns, in our setting, 0 _≤ ξ ≤_ 1. 

**Step 4: Controlling term (b).** For (b) in equation (31), which is _∥β_[�] _D[M][∗][−][β][M][∥][∞]_[, we are interested] in what is the optimization error given finite well-explored offline dataset _D[∗]_ . Here the optimization error mainly originates from the Gaussian noise in the SCM formulation in Definition 4. Yet our causal discovery module, i.e. an _ℓ_ 0 estimator, will always encounter some estimation error induced by the exogenous noise. 

Firstly, we have the bounded relationship between _ℓ_ 2 and _ℓ∞_ norm: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_BECAUSE_Bilinear_Causal_Representation_for_Generalizable_Offline_Model-based_Reinforcement_Learning_images/nips_2024_BECAUSE_Bilinear_Causal_Representation_for_Generalizable_Offline_Model-based_Reinforcement_Learning.pdf-0023-10.png)


then we can analyze the error bound for _ℓ_ 0 regression in the sense of _ℓ_ 2 norm. 

The derivation below generally follows the _ℓ_ 0 regularized linear regression bound in [90]. Based on Assumption 2 and 3, there exists _M_ , we denote this optimal solution in the vector form as _β[M]_ . 

Besides the aforementioned optimization error in the iterative thresholding update process, according to the SCM in Definition 4 and Assumption 2 and Proposition 1, we denote a finite subset of observed transition probabilities of the well-explored data E[ _Tπβ_ ] with size _n_ : _Tobv ∈_ R _[n]_ . By definition above, we can then assume the finite-sample regression target _T_ is generated by causal features of transition pairs (denoted by _X_ = _Xπ∗_ = _Xπβ , X ∈_ R _[n][×][dd][′]_ ) and the ground-truth causal mask (represented by _β[M] ∈_ R _[dd][′]_ ) with the following equation: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_BECAUSE_Bilinear_Causal_Representation_for_Generalizable_Offline_Model-based_Reinforcement_Learning_images/nips_2024_BECAUSE_Bilinear_Causal_Representation_for_Generalizable_Offline_Model-based_Reinforcement_Learning.pdf-0023-14.png)


Here _ϵ ∼N_ (0 _, σIn_ ) is the independent exogenous noise defined in Definition 4. We’ll then use the above equation to bound term (b) in equation (31). 

Specifically, we solved this _ℓ_ 0 regression problem with its bounded form as follows: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_BECAUSE_Bilinear_Causal_Representation_for_Generalizable_Offline_Model-based_Reinforcement_Learning_images/nips_2024_BECAUSE_Bilinear_Causal_Representation_for_Generalizable_Offline_Model-based_Reinforcement_Learning.pdf-0023-17.png)


In BECAUSE, we select the sparsity level _s ≈∥β[M] ∥_ 0 = _∥M ∥_ 0 _≤ dd[′]_ . 

23 

For simplicity, we denote the approximate solution _β_[�] _D[M][∗]_[in equation (39) as] _[β]_[�] _[M]_[.][By the virtue of] optimality of the solution _β[M]_ in equation (39), we find that 

_∥Tobv − Xβ[M] ∥_ 2[2] _[≤∥][T][obv][−][X]_[ �] _[β][M][∥]_[2] 2 _[,]_ (40) 

then by expanding and shifting the terms, we can derive the basic inequality for the _ℓ_ 0 estimator above: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_BECAUSE_Bilinear_Causal_Representation_for_Generalizable_Offline_Model-based_Reinforcement_Learning_images/nips_2024_BECAUSE_Bilinear_Causal_Representation_for_Generalizable_Offline_Model-based_Reinforcement_Learning.pdf-0024-03.png)


Since both _β_[�] _[M]_ and _β[M]_ are _s_ -sparse, the vector _β_[�] _[M] − β[M]_ is at most 2 _s_ -sparse. We denote the set of all 2 _s_ -sparse _dd[′]_ -dimensional vector set as T _[dd][′]_ (2 _s_ ), we denote _v_ = I( _β_[�] _[M] − β[M]_ = 0) as an indicator vector, _vi_ = 0 if and only if _β_[�] _i[M] − βi[M]_[,] _[ ∀][i][ ∈]_[[] _[dd][′]_[]][.] 

By shifting the terms, we use the sub-Gaussian assumption and further use Hölder’s inequality to get the following results: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_BECAUSE_Bilinear_Causal_Representation_for_Generalizable_Offline_Model-based_Reinforcement_Learning_images/nips_2024_BECAUSE_Bilinear_Causal_Representation_for_Generalizable_Offline_Model-based_Reinforcement_Learning.pdf-0024-06.png)


where _ϵ ∼N_ (0 _, σ_[2] _In_ ) is the exogenous noise variable in the SCM in Definition 4. 

Then by simply applying restricted eigenvalue (RE) condition, for some _κ_ ( _S_ ) _>_ 0, we have 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_BECAUSE_Bilinear_Causal_Representation_for_Generalizable_Offline_Model-based_Reinforcement_Learning_images/nips_2024_BECAUSE_Bilinear_Causal_Representation_for_Generalizable_Offline_Model-based_Reinforcement_Learning.pdf-0024-09.png)


Therefore, we have: 

with probability at least 1 _− δ_ , which bounds term (b) in equation (31). 

**Step 5: Summing up the results** Summarizing both bounds for terms (a) and (b) in equation (31), we will get the following bounds with probability 1 _− δ_ : 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_BECAUSE_Bilinear_Causal_Representation_for_Generalizable_Offline_Model-based_Reinforcement_Learning_images/nips_2024_BECAUSE_Bilinear_Causal_Representation_for_Generalizable_Offline_Model-based_Reinforcement_Learning.pdf-0024-13.png)


Therefore, we complete the proof of Lemma 3 by showing that for all ( _s, a_ ) _∈A × A, h ∈_ [ _H_ ], 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_BECAUSE_Bilinear_Causal_Representation_for_Generalizable_Offline_Model-based_Reinforcement_Learning_images/nips_2024_BECAUSE_Bilinear_Causal_Representation_for_Generalizable_Offline_Model-based_Reinforcement_Learning.pdf-0024-15.png)


24 

The final bound of the term (b) includes a dependency of _O_ ( ~~_√_~~[1] _n_[)][ and logarithm of dimensionality] _[ dd][′]_ in the estimated transition matrix. It also incurs a dependency of error tolerance _ξ_ or SCM’s noise level _σ_ square root of sparsity level _[√] s_ = ~~�~~ _∥M ∥_ 0. So far, we prove the following bound in theorem 1: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_BECAUSE_Bilinear_Causal_Representation_for_Generalizable_Offline_Model-based_Reinforcement_Learning_images/nips_2024_BECAUSE_Bilinear_Causal_Representation_for_Generalizable_Offline_Model-based_Reinforcement_Learning.pdf-0025-01.png)


In order to achieve _ξ_ -optimal policy such that _V_ 1 _[∗]_[(] _[s]_[�][)] _[ −][V]_ 1 _[π]_[(] _[s]_[�][)][ ≲] _[ξ]_[, the RHS needs to satisfy:] 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_BECAUSE_Bilinear_Causal_Representation_for_Generalizable_Offline_Model-based_Reinforcement_Learning_images/nips_2024_BECAUSE_Bilinear_Causal_Representation_for_Generalizable_Offline_Model-based_Reinforcement_Learning.pdf-0025-03.png)


We first multiply ~~�~~ min( _s,a,h_ ) _∈S×A×_ [ _H_ ] E _π⋆_[�] _n_ ( _sh, ah_ ) _| s_ 1 = � _s_ �, and then take the square on both sides. We have: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_BECAUSE_Bilinear_Causal_Representation_for_Generalizable_Offline_Model-based_Reinforcement_Learning_images/nips_2024_BECAUSE_Bilinear_Causal_Representation_for_Generalizable_Offline_Model-based_Reinforcement_Learning.pdf-0025-05.png)


Given the definition, LHS in equation (47) satisfies: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_BECAUSE_Bilinear_Causal_Representation_for_Generalizable_Offline_Model-based_Reinforcement_Learning_images/nips_2024_BECAUSE_Bilinear_Causal_Representation_for_Generalizable_Offline_Model-based_Reinforcement_Learning.pdf-0025-07.png)


To ensure the satisfaction of _ξ_ -optimal policy, we thus would like the RHS of equation (47) satisfies: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_BECAUSE_Bilinear_Causal_Representation_for_Generalizable_Offline_Model-based_Reinforcement_Learning_images/nips_2024_BECAUSE_Bilinear_Causal_Representation_for_Generalizable_Offline_Model-based_Reinforcement_Learning.pdf-0025-09.png)


Consequently, we can shift the terms and get the sample complexity bound for the _ξ_ -optimal policy, _∀_ 0 _≤ ξ ≤_ 1: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_BECAUSE_Bilinear_Causal_Representation_for_Generalizable_Offline_Model-based_Reinforcement_Learning_images/nips_2024_BECAUSE_Bilinear_Causal_Representation_for_Generalizable_Offline_Model-based_Reinforcement_Learning.pdf-0025-11.png)


## **C Additional Experiments Details** 

In this section, we provide additional experiment results and algorithm implementation details. 

## **C.1 Implementation of Causal Discovery** 

We implement the causal discovery primarily based on equation (5). However, in practice, how to control the coefficient before the sparsity regularization terms is crucial to the final performance. In 

25 

practice, instead of controlling _λ_ , we use _p_ -value as a threshold to determine the following conditional independence: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_BECAUSE_Bilinear_Causal_Representation_for_Generalizable_Offline_Model-based_Reinforcement_Learning_images/nips_2024_BECAUSE_Bilinear_Causal_Representation_for_Generalizable_Offline_Model-based_Reinforcement_Learning.pdf-0026-01.png)


Here _ϕ_ ( _·, ·_ )[(] _[i]_[)] means that this element is the _i[th]_ factor in the abstracted state action representation, similar to _µ_ ( _·, ·_ )[(] _[j]_[)] , _ϕ_ ( _s, a_ ) _[−]_[(] _[i]_[)] means all the other factors in the representation except for the _i[th]_ factor. 

If the p-value based on the above conditional independence test is less than a threshold, we can remove the edge by setting _Mij_ = 0. Please refer to the Appendix Table 10 for the selection of threshold in each environment. 

## **C.2 Training Details of Energy-based Model** 

We train the EBM according to the margin loss in equation (8). In practice, we attach Tanh() to the output layer to clip the unnormalized score between -1 and +1. The energy networks take in both conditions and samples, then concatenate them together and sent it into MLP encoders. The detailed hyperparameters of EBM are listed in Table 9. 

In the vanilla EBM, people follow the Langevin dynamics to effectively sample the negative samples. Here, as we discover the causal mask and identify the causal representation in the model learning stage, we find that we can use these representations in both the energy networks and the sampling process to get some effective negative samples, which is similar to the practice of augmentation of causality-guided counterfactual data [75]. 

The interesting trick we employ here is the way to get our negative samples by mixing the latent factors from offline data. For example, for a positive sample array 

with conditions: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_BECAUSE_Bilinear_Causal_Representation_for_Generalizable_Offline_Model-based_Reinforcement_Learning_images/nips_2024_BECAUSE_Bilinear_Causal_Representation_for_Generalizable_Offline_Model-based_Reinforcement_Learning.pdf-0026-09.png)


Here, _si, ai, s[′] i_[denotes the timestep of the offline samples.] _[ϕ]_[(] _[·][,][ ·]_[)][(] _[i]_[)][means this element is the] _[ i][th]_ factor in the abstracted state action representation, similar to the _µ_ ( _·, ·_ )[(] _[i]_[)] 

as we already get the corresponding causal representation _µ_ ( _s[′]_ ) that is semantically meaningful, 

we can mix the columns to create useful counterfactual negative samples 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_BECAUSE_Bilinear_Causal_Representation_for_Generalizable_Offline_Model-based_Reinforcement_Learning_images/nips_2024_BECAUSE_Bilinear_Causal_Representation_for_Generalizable_Offline_Model-based_Reinforcement_Learning.pdf-0026-13.png)


These counterfactual negative samples seem to effectively use the causal representation and speed up the training, as we show in Figure 7. 

## **C.3 Additional Mismatch Analysis** 

To evaluate the significance of the objective mismatch effect at three different levels of offline datasets in _Unlock_ environments, we collect 5,000 episodes in each of the Unlock environments for each method. Then we evaluate the mismatch via the following two metrics. 

26 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_BECAUSE_Bilinear_Causal_Representation_for_Generalizable_Offline_Model-based_Reinforcement_Learning_images/nips_2024_BECAUSE_Bilinear_Causal_Representation_for_Generalizable_Offline_Model-based_Reinforcement_Learning.pdf-0027-00.png)


**----- Start of picture text -----**<br>
T=0                    T=100                  T=200                   T=300                  T=400<br>T=500                  T=600                   T=700                  T=800                  T=900<br>Next State<br>GT<br>Random<br>Causal EBM<br>Random<br>Causal EBM<br>**----- End of picture text -----**<br>


Figure 7: Comparison of the convergence speed in EBM training. Compared to random negative samples, our approach enjoys a higher rate of convergence empirically. 

- We conduct hypothesis testing via Mann-Whitney U Test [91], with Null hypothesis 

   - _H_ 0 : _Lmodel_ ( _**τpos**_ ) _< Lmodel_ ( _**τneg**_ ) _,_ 

- To understand the exact difference in model loss between two groups of samples, we compute their Wasserstein-1 distance in the episodic model loss between the trajectories with positive and negative rewards, i.e. _W_ 1( _**τpos** ∥_ _**τneg**_ ). 

We report the results of _p_ -value and the _W_ 1 distance of two groups of model loss samples in the following table. 

Table 2: The comparison results of the _p_ -value and the _W_ 1 distance ( _×_ 10 _[−]_[4] ) between MOPO and BECAUSE. **Bold** means the better. 

|CAUSE.**B**|**old**means the better.|**old**means the better.|||||
|---|---|---|---|---|---|---|
|Methods|Unlock-Expert||Unlock-Medium||Unlock-Random||
||_p_-value (_↓_)|_W_1 Dist (_↑_)|_p_-value (_↓_)|_W_1 Dist (_↑_)|_p_-value (_↓_)|_W_1 Dist (_↑_)|
||||||||
|MOPO<br>Ours|6_._5_×_10_−_5<br>**_≈_0**|0.7<br>**3.1**|1.0<br>**_≈_0**|1.1<br>3.0|1.0<br>**0.9**|N.A.<br>**_<_ 0****_._1**|



## **C.4 Additional Experiment Results** 

We report the results of the task-wise performance of all baselines in the main experiment and variants in the ablation studies in Table 3, 4, 5, and 6. 

27 

Table 3: Success rate (%) for 18 tasks in three different environments. We evaluate the mean and 95% confidence interval given by the _t_ -test of the best performance among 10 random seeds, as well as the p-value between the overall performance. **Bold** is the best. 

|**Env**|**ICIL**<br>**CCIL**<br>**TD3+BC**<br>**MOPO**<br>**GNN**<br>**CDL**<br>**Denoised**<br>**IFactor**<br>**MnM**<br>**Delphic**<br>**Ours**|
|---|---|
|||
|Lift-I-R<br>Lift-O-R|14.3_±_9.9<br>10.0_±_11.4<br>9.7_±_5.4<br>24.3_±_2.9<br>22.1_±_3.6<br>**33.8**_±_**5.0**<br>20.0_±_4.0<br>24.0_±_2.8<br>16.3_±_2.8<br>20.2_±_3.1<br>19.2_±_0.6<br>8.5_±_4.7<br>0.0_±_0.0<br>1.3_±_1.0<br>10.2_±_1.6<br>13.3_±_2.3<br>16.0_±_4.7<br>15.5_±_3.8<br>21.2_±_3.0<br>14.2_±_2.2<br>17.7_±_2.8<br>**21.4**_±_**3.9**|
|||
|Unlock-I-R<br>Unlock-O-R|3.4_±_1.1<br>2.2_±_0.8<br>4.39_±_0.9<br>21.5_±_1.9<br>11.7_±_2.1<br>6.6_±_0.5<br>6.9_±_0.7<br>8.1_±_1.1<br>8.6_±_1.3<br>14.4_±_1.1<br>**32.7**_±_**2.8**<br>11.6_±_4.0<br>13.5_±_3.7<br>13.3_±_3.0<br>16.6_±_1.3<br>12.1_±_1.5<br>7.6_±_1.0<br>7.0_±_0.8<br>8.0_±_1.1<br>9.0_±_1.0<br>12.2_±_1.1<br>**27.6**_±_**2.0**|
|||
|Crash-I-R<br>Crash-O-R|11.4_±_4.3<br>19.5_±_10.5<br>9.1_±_6.6<br>32.7_±_4.9<br>11.7_±_0.8<br>39.7_±_4.6<br>32.0_±_2.8<br>31.3_±_4.4<br>16.0_±_2.2<br>17.4_±_3.6<br>**59.4**_±_**6.1**<br>2.8_±_1.8<br>5.7_±_3.0<br>0.9_±_0.6<br>10.0_±_2.0<br>3.7_±_0.4<br>10.8_±_1.9<br>10.8_±_2.1<br>11.0_±_3.6<br>4.1_±_0.6<br>5.4_±_1.1<br>**19.7**_±_**1.4**|
|||
|Lift-I-M<br>Lift-O-M|54.0_±_13.3<br>44.0_±_20.8<br>26.6_±_14.8<br>39.8_±_5.3<br>27.5_±_5.3<br>32.9_±_5.7<br>35.3_±_4.8<br>41.0_±_5.7<br>28.7_±_1.8<br>24.0_±_2.3<br>**59.5**_±_**4.4**<br>**46.8**_±_**15.2**<br>20.0_±_21.9<br>13.0_±_1.1<br>31.9_±_2.8<br>25.8_±_1.8<br>26.0_±_4.2<br>27.0_±_3.2<br>30.2_±_2.7<br>24.3_±_2.3<br>18.3_±_2.6<br>32.3_±_4.9|
|||
|Unlock-I-M<br>Unlock-O-M|4.8_±_0.9<br>4.7_±_1.3<br>5.4_±_1.0<br>84.8_±_5.1<br>17.5_±_2.6<br>29.7_±_4.4<br>12.9_±_1.5<br>37.7_±_5.5<br>37.6_±_4.9<br>74.1_±_1.5<br>**98.0**_±_**4.9**<br>12.9_±_2.8<br>14.1_±_2.0<br>19.2_±_2.5<br>39.5_±_4.7<br>16.2_±_1.8<br>20.5_±_3.9<br>10.8_±_0.8<br>21.5_±_2.7<br>27.0_±_3.7<br>51.7_±_2.3<br>**68.8**_±_**1.5**|
|||
|Crash-I-M<br>Crash-O-M|35.5_±_9.9<br>24.5_±_12.2<br>16.3_±_9.0<br>47.7_±_7.3<br>11.5_±_1.1<br>63.5_±_4.0<br>63.8_±_4.0<br>45.5_±_5.0<br>18.4_±_2.9<br>58.2_±_2.2<br>**90.4**_±_**1.8**<br>11.7_±_3.5<br>7.9_±_4.4<br>5.6_±_3.1<br>17.3_±_2.3<br>3.8_±_0.4<br>20.0_±_2.3<br>20.3_±_1.6<br>16.0_±_2.0<br>6.3_±_1.9<br>**22.2**_±_**1.7**<br>20.3_±_1.9|
|||
|Lift-I-E<br>Lift-O-E|73.8_±_17.0<br>86.7_±_15.7<br>44.3_±_12.2<br>82.4_±_6.7<br>63.3_±_0.0<br>71.0_±_5.5<br>74.2_±_5.5<br>**98.0**_±_**3.7**<br>33.3_±_3.0<br>53.5_±_6.3<br>92.8_±_1.2<br>54.1_±_26.1<br>80.0_±_18.9<br>41.6_±_16.6<br>72.4_±_1.9<br>60.8_±_0.6<br>63.1_±_5.1<br>64.0_±_4.4<br>91.7_±_5.7<br>29.0_±_4.1<br>49.5_±_3.1<br>**93.7**_±_**5.9**|
|||
|Unlock-I-E<br>Unlock-O-E|11.6_±_3.2<br>13.7_±_3.1<br>15.6_±_2.2<br>88.8_±_4.6<br>15.3_±_1.6<br>73.2_±_2.8<br>50.3_±_3.0<br>59.3_±_2.6<br>60.7_±_2.2<br>83.2_±_1.2<br>**97.4**_±_**1.0**<br>22.4_±_5.0<br>35.4_±_6.0<br>41.6_±_6.2<br>39.9_±_4.4<br>13.8_±_1.4<br>41.3_±_4.1<br>35.7_±_2.3<br>29.5_±_3.5<br>38.7_±_2.0<br>54.4_±_1.9<br>**82.1**_±_**6.5**|
|||
|Crash-I-E<br>Crash-O-E|35.7_±_9.8<br>26.8_±_7.2<br>26.0_±_14.4<br>58.5_±_4.7<br>11.2_±_0.9<br>63.7_±_2.8<br>69.3_±_3.9<br>52.0_±_5.3<br>10.2_±_1.1<br>57.0_±_1.2<br>**95.3**_±_**1.3**<br>11.3_±_3.8<br>12.3_±_3.5<br>6.2_±_1.8<br>14.8_±_2.1<br>3.5_±_0.9<br>18.8_±_1.8<br>**20.7**_±_**2.2**<br>15.6_±_2.3<br>3.9_±_0.6<br>16.5_±_1.7<br>**20.7**_±_**3.2**|
|Overall-I<br>27.2<br>25.8<br>17.5<br>53.4<br>21.0<br>44.7<br>40.5<br>44.1<br>25.5<br>44.7<br>**73.3**<br>Overall-O<br>20.2<br>19.9<br>14.6<br>28.1<br>17.0<br>24.9<br>23.5<br>27.2<br>17.4<br>27.5<br>**43.0**||



Table 4: _p_ -values of different methods (each has 10 random trials) against BECAUSE in various environments. Under the significance level 0.05, we mark all the baseline results that are significantly lower than BECAUSE as green, and the rest as red. We can see that BECAUSE **significantly** outperforms 10 baselines in 18 tasks in 91.1% of the experiments (164 out of total 180 pairs of experiments). 

|periments).||
|---|---|
|Env|ICIL<br>TD3+BC<br>MOPO<br>GNN<br>CDL<br>Denoised<br>IFactor<br>CCIL<br>MnM<br>Delphic|
|||
|Lift-I-random<br>Lift-O-random<br>Unlock-I-random<br>Unlock-O-random<br>Crash-I-random<br>Crash-O-random<br>Lift-I-medium<br>Lift-O-medium<br>Unlock-I-medium<br>Unlock-O-medium<br>Crash-I-medium<br>Crash-O-medium<br>Lift-I-expert<br>Lift-O-expert<br>Unlock-I-expert<br>Unlock-O-expert<br>Crash-I-expert<br>Crash-O-expert|0.001<br>0.000<br>0.001<br>0.000<br>0.000<br>0.000<br>0.001<br>0.001<br>0.000<br>0.000<br>0.000<br>0.000<br>0.000<br>0.001<br>0.033<br>0.013<br>0.464<br>0.000<br>0.001<br>0.052<br>0.000<br>0.000<br>0.000<br>0.000<br>0.000<br>0.000<br>0.000<br>0.000<br>0.000<br>0.000<br>0.000<br>0.000<br>0.000<br>0.000<br>0.000<br>0.000<br>0.000<br>0.000<br>0.000<br>0.000<br>0.000<br>0.000<br>0.000<br>0.000<br>0.000<br>0.000<br>0.000<br>0.000<br>0.000<br>0.000<br>0.000<br>0.000<br>0.000<br>0.000<br>0.000<br>0.000<br>0.000<br>0.000<br>0.000<br>0.000<br>0.198<br>0.000<br>0.000<br>0.000<br>0.000<br>0.000<br>0.000<br>0.067<br>0.000<br>0.000<br>0.966<br>0.000<br>0.438<br>0.009<br>0.022<br>0.032<br>0.209<br>0.125<br>0.003<br>0.000<br>0.000<br>0.000<br>0.000<br>0.000<br>0.000<br>0.000<br>0.000<br>0.000<br>0.000<br>0.000<br>0.000<br>0.000<br>0.000<br>0.000<br>0.000<br>0.000<br>0.000<br>0.000<br>0.000<br>0.000<br>0.000<br>0.000<br>0.000<br>0.000<br>0.000<br>0.000<br>0.000<br>0.000<br>0.000<br>0.000<br>0.000<br>0.000<br>0.018<br>0.000<br>0.412<br>0.500<br>0.001<br>0.000<br>0.000<br>0.943<br>0.017<br>0.000<br>0.004<br>0.000<br>0.000<br>0.000<br>0.993<br>0.203<br>0.000<br>0.000<br>0.004<br>0.000<br>0.000<br>0.000<br>0.000<br>0.000<br>0.297<br>0.027<br>0.000<br>0.000<br>0.000<br>0.000<br>0.001<br>0.000<br>0.000<br>0.000<br>0.000<br>0.000<br>0.000<br>0.000<br>0.000<br>0.000<br>0.000<br>0.000<br>0.000<br>0.000<br>0.000<br>0.000<br>0.000<br>0.000<br>0.000<br>0.000<br>0.000<br>0.000<br>0.000<br>0.000<br>0.000<br>0.000<br>0.000<br>0.000<br>0.000<br>0.000<br>0.002<br>0.000<br>0.133<br>0.500<br>0.005<br>0.000<br>0.000<br>0.011|



28 

Table 5: Success rate (%) for 18 tasks in three different environments. We evaluate the mean and 95% confidence interval of the test performance among 10 random seeds. **Bold** means the best. 

|Env|**BECAUSE**<br>BECAUSE-Optimism<br>BECAUSE-Linear<br>BECAUSE-Full|
|---|---|
|||
|Lift-I-random<br>Lift-O-random|**33.8**_±_**5.0**<br>23.2_±_3.1<br>16.5_±_1.6<br>22.2_±_6.6<br>**21.4**_±_**3.9**<br>15.3_±_1.5<br>8.9_±_4.8<br>18.2_±_5.3|
|||
|Unlock-I-random<br>Unlock-O-random|**32.7**_±_**2.8**<br>31.2_±_2.4<br>20.7_±_1.7<br>10.7_±_0.8<br>**27.6**_±_**2.1**<br>26.3_±_1.7<br>24.0_±_2.7<br>9.3_±_0.6|
|||
|Crash-I-random<br>Crash-O-random|**59.4**_±_**6.2**<br>49.9_±_9.2<br>54.3_±_5.4<br>36.1_±_7.3<br>**19.7**_±_**1.4**<br>14.2_±_0.5<br>14.8_±_1.5<br>10.0_±_2.2|
|||
|Lift-I-medium<br>Lift-O-medium|**59.5**_±_**4.5**<br>46.8_±_2.1<br>24.4_±_5.3<br>36.4_±_6.7<br>**32.3**_±_**5.0**<br>24.5_±_2.5<br>16.4_±_2.3<br>28.9_±_4.3|
|||
|Unlock-I-medium<br>Unlock-O-medium|**98.0**_±_**4.9**<br>92.7_±_5.8<br>91.0_±_1.8<br>29.9_±_1.9<br>**68.8**_±_**1.5**<br>58.7_±_2.2<br>60.0_±_2.0<br>18.7_±_1.3|
|||
|Crash-I-medium<br>Crash-O-medium|**90.4**_±_**1.8**<br>82.8_±_9.9<br>66.7_±_7.4<br>60.8_±_1.8<br>**20.3**_±_**1.9**<br>17.5_±_0.0<br>15.9_±_2.8<br>24.6_±_0.9|
|||
|Lift-I-expert<br>Lift-O-expert|**92.8**_±_**1.2**<br>68.6_±_4.7<br>75.6_±_10.7<br>78.1_±_6.1<br>**93.7**_±_**6.0**<br>58.3_±_7.9<br>66.1_±_8.0<br>71.9_±_6.9|
|||
|Unlock-I-expert<br>Unlock-O-expert|**97.4**_±_**1.0**<br>93.1_±_1.9<br>94.0_±_2.0<br>29.3_±_1.3<br>**82.1**_±_**6.6**<br>64.6_±_3.1<br>65.8_±_3.2<br>20.2_±_1.7|
|Crash-I-expert<br>**95.3**_±_**1.4**<br>91.0_±_2.2<br>77.9_±_2.8<br>50.3_±_7.6<br>Crash-O-expert<br>20.7_±_3.2<br>12.5_±_0.0<br>**26.9**_±_**6.1**<br>25.0_±_1.8||
|||
|Overall-I<br>Overall-O|**73.3**<br>64.4<br>57.9<br>39.3<br>**43.0**<br>32.4<br>33.2<br>25.2|



Table 6: _p_ -values of different methods (each has 10 random trials) against BECAUSE in various environments. Under the significance level 0.05, we mark all the baseline results that are significantly lower than BECAUSE as green, and the rest as red. We can see that BECAUSE **significantly** outperforms 3 variants in 18 tasks 83.3% of the experiments (45 out of total 54 pairs of experiments). 

|Env|BECAUSE-Optimism<br>BECAUSE-Linear<br>BECAUSE-Full|
|---|---|
|||
|Lift-I-random<br>Lift-O-random<br>Unlock-I-random<br>Unlock-O-random<br>Crash-I-random<br>Crash-O-random<br>Lift-I-medium<br>Lift-O-medium<br>Unlock-I-medium<br>Unlock-O-medium<br>Crash-I-medium<br>Crash-O-medium<br>Lift-I-expert<br>Lift-O-expert<br>Unlock-I-expert<br>Unlock-O-expert<br>Crash-I-expert<br>Crash-O-expert|0.001<br>0.000<br>0.003<br>0.003<br>0.000<br>0.146<br>0.189<br>0.000<br>0.000<br>0.145<br>0.015<br>0.000<br>0.036<br>0.090<br>0.000<br>0.000<br>0.000<br>0.000<br>0.000<br>0.000<br>0.000<br>0.004<br>0.000<br>0.130<br>0.067<br>0.006<br>0.000<br>0.000<br>0.000<br>0.000<br>0.061<br>0.000<br>0.000<br>0.005<br>0.005<br>1.000<br>0.000<br>0.003<br>0.000<br>0.000<br>0.000<br>0.000<br>0.000<br>0.002<br>0.000<br>0.000<br>0.000<br>0.000<br>0.001<br>0.000<br>0.000<br>0.000<br>0.969<br>0.990|



29 

## **C.5 Additional Environment Description** 

We provide a more detailed description of the environments we use in the experiment, as shown in Table 7. 

**Lift** The Lift environment based on the robosuite [34] contains 33 dimensions of state space, including the end effector pose, joint pose, joint velocity, cube pose as well as its relative position, cube color, and a contact flag. It contains 4 dimensions of hybrid action space that uses Operation Space Control (OSC) to control the 3D position and the 1D gripper movement. The task is counted as a success when the assigned block is lifted from the table over 0.1m. The generalization setting in the Lift environment is to use an unseen combination of position and color during online testing. This environment can be abstracted into 15 dimensions of factorizable state space and 4 dimensions of factorizable action space. The causal graph of this environment is recorded in Figure 9(a). 

**Unlock** The Unlock environments based on the MiniGrid world [35] contain 110 dimensions of discrete state space, with 3 of 36-dimensional vector inputs representing the current position of the agent, key, and door in a 6x6 grid world. The rest 2 dimensions in the state space memorize the state of whether the agent has the key in hand. The action space is also discrete (with eight dimensions) to determine the movement (up/down/left/right) and the pick-key, open-door actions. An episode will be counted as a success when the agent holds the key and uses it to open the door in the right position. The generalization setting in the Unlock environment is to change the position of the door and increase the number of total goals in the environment. The agent will only successfully finish one episode by opening all the doors. The causal graph of this environment is recorded in Figure 9(b). 

**Crash** The Crash environments are based on the Highway environment [37] which contains 22 dimensions of continuous state space, with four vector inputs representing the current position, velocity, and orientation of the surrounding vehicles and ego vehicles. There are two additional dimensions of state memorizing the collision type between the ego vehicles and surrounding vehicles or pedestrians. The 8-dimensional action space is continuous to determine the acceleration in the _x − y_ directions of the ego and surrounding agents. The generalization of the Crash environment is to add different numbers of pedestrians that may cause the crash. An episode will only end when the ego vehicles have a near-miss with both of the pedestrians at the scene. We visualize the causal graph of this environment in Figure 9(c). 

All three environments are visualized in Figure 8. We list their basic configurations in Table 7. 

Table 7: Environment configurations used in experiments 

|Parameters|Environment|Environment|Environment|
|---|---|---|---|
||Lift|Unlock|Crash|
|||||
|Max step size<br>State dimension<br>Action dimension<br>Action type<br>Intrinsic state rank<br>Intrinsic action rank|30<br>33<br>4<br>Hybrid<br>15<br>4|15<br>110<br>8<br>Discrete<br>4<br>3|30<br>22<br>8<br>Hybrid<br>6<br>4|




![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_BECAUSE_Bilinear_Causal_Representation_for_Generalizable_Offline_Model-based_Reinforcement_Learning_images/nips_2024_BECAUSE_Bilinear_Causal_Representation_for_Generalizable_Offline_Model-based_Reinforcement_Learning.pdf-0030-08.png)


Figure 8: Three environments used in this paper. 

30 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_BECAUSE_Bilinear_Causal_Representation_for_Generalizable_Offline_Model-based_Reinforcement_Learning_images/nips_2024_BECAUSE_Bilinear_Causal_Representation_for_Generalizable_Offline_Model-based_Reinforcement_Learning.pdf-0031-00.png)


**----- Start of picture text -----**<br>
(a) Lift Environment (b) Unlock Environment (c) Crash Environment<br>**----- End of picture text -----**<br>


Figure 9: Underlying causal graph _G_ in all 3 environments with expert demonstration. 

## **C.6 Additional Baseline Information** 

We collect data on the above 3 different environments, thus forming 9 groups of offline datasets. 

Table 8: Bahavior policies used to collect offline data in different environments. 

|Environment|Behavior<br>#Episodes<br>Success Rate|Additional Description|
|---|---|---|
||||
|Lift|Random<br>1000<br>0.24<br>Medium<br>1000<br>0.60<br>Expert<br>1000<br>1.00|Random actions_after_a few steps of initialization.<br>Random actions_before_the goal-reaching expert.<br>Query expert policy for all time steps.|
||||
|Unlock|Random<br>200<br>0.21<br>Medium<br>200<br>0.46<br>Expert<br>200<br>0.87|Random navigation with high randomness<br>Targeted searching in goal directions<br>Shortest path planning via_A∗_|
||||
|Crash|Random<br>1000<br>0.14<br>Medium<br>1000<br>0.35<br>Expert<br>1000<br>0.66|Fixed ego, random pedestrians<br>Planned ego, random pedestrians<br>Planning in both ego and pedestrians|



After collecting the data using scripted policies in different environments, we train all agents as well as BECAUSE under 10 different random seeds. Then we report the best performance of each trial and compute the mean and standard deviation over 10 seeds for each task in the Appendix 3. 

We refer to the following codebase to implement all the baselines we use: 

- Invariant Causal Imitation Learning ( **ICIL** , [38]): https://github.com/ioanabica/InvariantCausal-Imitation-Learning, MIT License. 

- Causal Confusion Imitation Learning ( **CCIL** , [39]): reference link to the paper. 

- TD3 with Behavior Cloning ( **TD3+BC** , [41]): https://github.com/sfujim/TD3_BC, MIT License. 

- Model-based Offline Policy Otimization ( **MOPO** , [2]): https://github.com/junmingyang/mopo.git, MIT License. 

- Relational Graph Neural Network ( **GNN** , [42]): https://github.com/MichSchli/RelationPrediction.git, MIT License. 

- Causal Dynamics Learning ( **CDL** , [24]): https://github.com/wangzizhao/robosuite/tree/cdl, MIT License. 

- Denoised MDP ( **Denoised** , [12]): https://github.com/facebookresearch/denoised_mdp.git, CC BY-NC 4.0. 

- Mismatch No More ( **MnM** , [9]): reference link to the paper. 

- World model with identifiable factorization ( **IFactor** , [13]), reference link to the paper. 

- Delphic Offline RL ( **Delphic** , [40]): reference link to the paper. 

31 

The detailed hyperparameters we use in BECAUSE and other baselines are listed in Table 9 and Table 10: 

## **C.7 Experiment Support** 

Our code is available at the anonymous repo: https://anonymous.4open.science/r/BECAUSE-NeurIPS 

**Computing resources** The experiments are run on a server with 2 _×_ AMD EPYC 7542 32-Core Processor CPU, 2 _×_ NVIDIA RTX 3090 graphics and 2 _×_ NVIDIA RTX A6000 graphics, and 252 GB memory. For one single experiment, it takes BECAUSE and other baselines about 1 _._ 5 hours with 100 _,_ 000 iterations to train the world model and 1 _,_ 000 _,_ 000 steps to train the energy-based models. 

## **C.8 Broader Impact** 

This work incorporates causality into reinforcement learning methods, which helps humans understand the underlying mechanism of algorithms and check the source of failures. However, the learned causal world model may contain human-readable private information about the environment and the dataset. To mitigate this potential negative societal impact, the causal world model should only be accessible to trustworthy users. 

Table 9: Hyper-parameters of models used in experiments of BECAUSE and baselines (Part I) 

|Models|Parameters|Environment|Environment|Environment|
|---|---|---|---|---|
|||Lift|Unlock|Crash|
||||||
|BECAUSE|Learning rate<br>Size of data_D_<br>Epoch per iteration<br>Batch size<br>Planning horizon_H_<br>Planning population<br>Reward discount_γ_<br>Spectral norm regularizer_λϕ_<br>Spectral norm regularizer_λµ_<br>Causal discovery_pthres_<br>Encoder hiddens<br>EBM hidden<br>EBM negative buffer<br>EBM training steps<br>EBM regularizer_λ_EBM|0.0001<br>15000<br>20<br>256<br>15<br>1500<br>0.99<br>10_−_4<br>10_−_4<br>10_−_8<br>256<br>256<br>5000<br>1000<br>10_−_4|0.001<br>4000<br>5<br>256<br>10<br>100<br>0.99<br>10_−_4<br>10_−_4<br>10_−_4<br>64<br>64<br>1000<br>1000<br>10_−_4|0.0001<br>15000<br>10<br>256<br>20<br>1000<br>0.99<br>10_−_4<br>10_−_4<br>10_−_6<br>128<br>128<br>5000<br>1000<br>10_−_4|
||||||
|MOPO*|MLP hiddens<br>MLP layers<br>Ensemble number|256<br>2<br>5|64<br>2<br>5|128<br>2<br>5|
||||||
|CDL*|Initialized mask coef.<br>MLP hiddens<br>Sparsity regularizer|1.0<br>256<br>0.001|1.0<br>64<br>0.001|1.0<br>128<br>0.001|
|GNN*<br>GNN hiddens<br>GNN layers||256<br>64<br>128<br>3<br>1<br>3|||



> * Use the same planning parameters as BECAUSE. 

32 

Table 10: Hyper-parameters of models used in experiments of baselines (Continued) 

|Models|Parameters|Environment|Environment|Environment|
|---|---|---|---|---|
|||Lift|Unlock|Crash|
||||||
|ICIL|Learning rate of MINE<br>MINE hiddens<br>MLP hiddens<br>Learning rate of EBM<br>Size of buffer of EBM<br>EBM training steps<br>EBM hiddens<br>K of langevin rollout<br>_λV ar_ of langevin rollout|0.0001<br>256<br>256<br>0.01<br>1000<br>1000<br>256<br>60<br>0.01|0.0001<br>64<br>64<br>0.01<br>1000<br>1000<br>64<br>60<br>0.01|0.0001<br>128<br>128<br>0.01<br>1000<br>1000<br>128<br>60<br>0.01|
||||||
|TD3+BC|Learning rate of Critic<br>Critic hiddens<br>Learning rate of Actor<br>Actor hiddens<br>Target update rate<br>Policy noise<br>Balance coeffcient_α_|0.0003<br>256<br>0.0003<br>256<br>0.005<br>0.2<br>1.0|0.003<br>64<br>0.001<br>64<br>0.001<br>0.2<br>2.5|0.0003<br>128<br>0.0001<br>128<br>0.0001<br>0.2<br>2.5|
||||||
|Denoised MDP|x belief size<br>y belief size<br>z belief size<br>x state size<br>y state size<br>z state size<br>embedding size<br>Learning rate|256<br>256<br>0<br>33<br>33<br>0<br>256<br>0.0001|64<br>64<br>0<br>110<br>110<br>0<br>64<br>0.001|128<br>128<br>0<br>22<br>22<br>0<br>128<br>0.0001|
||||||
|IFactor|All hidden dim<br>Disentangled prior output size<br>Learning rate|256<br>19<br>0.0001|64<br>7<br>0.001|128<br>10<br>0.0001|
||||||
|MnM|All hidden dim<br>Discriminator learning rate<br>Discriminator clip norm|256<br>0.0001<br>0.25|64<br>0.001<br>0.25|128<br>0.0001<br>0.25|
||||||
|CCIL|Reg weight<br>Initial mask probability<br>Learning rate|0.0001<br>0.95<br>0.0001|0.001<br>0.95<br>0.001|0.0001<br>0.95<br>0.0001|
||||||
|Delphic|Ensemble model size<br>Uncertainty penalty weight<br>KL weight|5<br>0.0001<br>0.0001|5<br>0.0001<br>0.0001|5<br>0.00005<br>0.00005|



33 

