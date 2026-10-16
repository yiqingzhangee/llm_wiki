Published as a conference paper at ICLR 2023 

## CAUSAL REASONING IN THE PRESENCE OF LATENT CONFOUNDERS VIA NEURAL ADMG LEARNING 

**Matthew Ashman**[2] _[∗]_ **, Chao Ma**[1] _[∗]_ **, Agrin Hilmkil**[1] **, Joel Jennings**[1] **, Cheng Zhang**[1] 1Microsoft Research Cambridge 2University of Cambridge `mca39@cam.ac.uk` , `{chaoma,agrinhilmkil,joeljennings,chezha}@microsoft.com` 

## ABSTRACT 

Latent confounding has been a long-standing obstacle for causal reasoning from observational data. One popular approach is to model the data using acyclic directed mixed graphs (ADMGs), which describe ancestral relations between variables using directed and bidirected edges. However, existing methods using ADMGs are based on either linear functional assumptions or a discrete search that is complicated to use and lacks computational tractability for large datasets. In this work, we further extend the existing body of work and develop a novel gradientbased approach to learning an ADMG with non-linear functional relations from observational data. We first show that the presence of latent confounding is identifiable under the assumptions of bow-free ADMGs with non-linear additive noise models. With this insight, we propose a novel neural causal model based on autoregressive flows for ADMG learning. This not only enables us to determine complex causal structural relationships behind the data in the presence of latent confounding, but also estimate their functional relationships (hence treatment effects) simultaneously. We further validate our approach via experiments on both synthetic and real-world datasets, and demonstrate the competitive performance against relevant baselines. 

## 1 INTRODUCTION 

Learning causal relationships and estimating treatment effects from observational studies is a fundamental problem in causal machine learning, and has important applications in many areas of social and natural sciences (Pearl, 2010; Spirtes, 2010). They enable us to answer questions in causal nature; for example, _what is the effect on the expected lifespan of a patient if I increase the dose of X drug?_ However, many existing methods of causal discovery and inference overwhelmingly rely on the assumption that all necessary information is available. This assumption is often untenable in practice. Indeed, an important, yet often overlooked, form of causal relationships is that of latent confounding; that is, when two variables have an unobserved common cause (Verma & Pearl, 1990). If not properly accounted for, the presence of latent confounding can lead to incorrect evaluation of causal quantities of interest (Pearl, 2009). 

Traditional causal discovery methods that account for the presence of latent confoundings, such as the fast causal inference algorithm (FCI) (Spirtes et al., 2000) and its extensions (Colombo et al., 2012; Claassen et al., 2013; Chen et al., 2021), rely on uncovering an equivalence class of acyclic directed mixed graphs (ADMGs) that share the same conditional independencies. Without additional assumptions, however, these methods might return uninformative results as they cannot distinguish between members of the same Markov equivalence class (Bellot & van der Schaar, 2021). More recently, causal discovery methods based on structural causal models (SCMs) (Pearl, 1998) have been developed for latent confounding (Nowzohour et al., 2017; Wang & Drton, 2020; Maeda & Shimizu, 2020; 2021; Bhattacharya et al., 2021). By assuming that the causal effects follow specific functional forms, they have the advantage of being able to distinguish between members of the same Markov equivalence class (Glymour et al., 2019). Yet, existing approaches either rely on restrictive linear functional assumptions (Bhattacharya et al., 2021; Maeda & Shimizu, 2020; Bellot & van der Schaar, 2021), and/or discrete search over the discrete space of causal graphs (Maeda & 

> _∗_ Equal contribution. This work is done when Matthew Ashman was an intern at Microsoft Research. 

1 

Published as a conference paper at ICLR 2023 

Shimizu, 2021) that are computationally burdensome and unintuitive to use. As a result, modeling non-linear causal relationships between variables in the presence of latent confounders in a scalable way remains an outstanding task. 

In this work, we seek to utilize recent advances in differentiable causal discovery (Zheng et al., 2018; Bhattacharya et al., 2021) and neural causal models (Lachapelle et al., 2019; Morales-Alvarez et al., 2022; Geffner et al., 2022) to overcome these limitations. Our core contribution is to extend the framework of differentiable ADMG discovery for linear models (Bhattacharya et al., 2021) to non-linear cases using neural causal models. This enables us to build scalable and flexible methods capable of discovering non-linear, potentially confounded relationships between variables and perform subsequent causal inference. Specifically, our contributions include: 

1. **Sufficient conditions for ADMG identifiability with non-linear SCMs (Section 4).** We assume: i) the functional relationship follows non-linear additive noise SCM; ii) the effect of observed and latent variables do not modulate each other, and iii) all latent variables confound a pair of non-adjacent observed nodes. Under these assumptions, the underlying ground truth ADMG causal graph is identifiable. This serves as a foundation for designing ADMG identification algorithms for flexible, non-linear SCMs based on deep generative models. 

2. **A novel gradient-based framework for learning ADMGs from observational data (Section 5).** Based on our theoretical results, we further propose Neural ADMG Learning (NADMG), a neural autoregressive-flow-based model capable of learning complex non-linear causal relationships with latent confounding. N-ADMG utilizes variational inference to approximate posteriors over causal graphs and latent variables, whilst simultaneously learning the model parameters via gradient-based optimization. This is more efficient and accurate than discrete search methods, allowing us to replace task-specific search procedures with general purpose optimizers. 

3. **Empirical evaluation on synthetic and real-world datasets (Section 6).** We evaluate N- ADMG on a variety of synthetic and real-world datasets, comparing performance with a number of existing causal discovery and inference algorithms. We find that N-ADMG provides competitive or state-of-the-art results on a range of causal reasoning tasks. 

## 2 RELATED WORK 

**Causal discovery with latent confounding.** Constraint-based causal discovery methods in the presence of latent confounding have been well-studied (Spirtes et al., 2000; Zhang, 2008; Colombo et al., 2012; Claassen et al., 2013; Chen et al., 2021). Without further assumptions, these approaches can only identify a Markov equivalence class of causal structures (Spirtes et al., 2000). When certain assumptions are made on the data generating process in the form of SCMs (Pearl, 1998), additional constraints can help identify the true causal structure. In the most general case, additional nonparametric constraints have been identified (Verma & Pearl, 1990; Shpitser et al., 2014; Evans, 2016). Further refinement can be made through the assumption of stricter SCMs. For example, in the linear Gaussian additive noise model (ANM) case, Nowzohour et al. (2017) proposes a scorebased approach for finding an equivalent class of bow-free acyclic path diagrams. Both Maeda & Shimizu (2020) and Wang & Drton (2020) develop Independence tests based approach for linear non-Gaussian ANM case, with Maeda & Shimizu (2021) extending this to more general cases. 

**Differentiable characterization of causal discovery.** All aforementioned approaches employ a search over a discrete space of causal structures, which often requires task-specific search procedures, and imposes a computational burden for large-scale problems. More recently, (Zheng et al., 2018) proposed a differentiable constraint on directed acyclic graphs (DAG), and frames the graph structure learning problem as a differentiable constrained optimization task in the absence of latent confounders. This is further generalized to the latent confounding case (Bhattacharya et al., 2021) through differentiable algebraic constraints that characterize the space of ADMGs. Nonetheless, this work is limited in that it only considers linear Gaussian ANMs. 

2 

Published as a conference paper at ICLR 2023 

## 3 BACKGROUND 

## 3.1 STRUCTURAL CAUSAL MODELS IN THE ABSENCE OF LATENT CONFOUNDINGS 

Unlike constraint-based approaches such as the PC algorithm, SCMs capture the asymmetry between causal direction through functional assumptions on the data generating process, and have been central to many recent developments in causal discovery (Glymour et al., 2019). . Given a directed acyclic graph (DAG) _G_ on nodes _{_ 1 _, . . . , D}_ , SCMs describe the random variable **x** = ( _x_ 1 _, . . . , xD_ ) by _xi_ = _fi_ ( **x** pa( _i_ ; _G_ ) _, ϵi_ ), where _ϵi_ is an exogenous noise variable that is independent of all other variables in the model, pa( _i_ ; _G_ ) denotes the set of parents of node _i_ in _G_ , and _fi_ describes how _xi_ depends on its parents and the noise _ϵi_ . We focus on additive noise SCMs, commonly referred to as additive noise models (ANMs), which take the form 

_xi_ = _fi_ ( **x** pa( _i_ ; _G_ ) _, ϵi_ ) = _fi_ ( **x** pa( _i_ ; _G_ )) + _ϵi,_ or **x** = _fG_ ( **x** ) + _**ϵ**_ in vector form. (1) 

This induces a joint observation distribution _pθ_ ( **x** _[n] |G_ ), where _θ_ denotes the parameters for functions _{fi}_ . Under the additive noise model in Equation 1, the DAG _G_ is identifiable assuming causal minimality and no latent confounders (Peters et al., 2014). 

## 3.2 GRAPHICAL REPRESENTATION OF LATENT CONFOUNDERS USING ADMGS 

One of the most widely-used graphical representations of causal relationships involving latent confounding is the so-called acyclic directed mixed graph (ADMGs). ADMGs are an extension of DAGs, that contain both directed edges ( _→_ ) and bidirected edges ( _↔_ ) between variables. More concretely, the directed edge _xi → xj_ indicates that _xi_ is an ancestor of _xj_ , and the bidirected edge _xi ↔ xj_ indicates that _xi_ and _xj_ share a common, unobserved ancestor (Richardson & Spirtes, 2002; Tian & Pearl, 2002). An ADMG _G_ over a collection of _D_ variables **x** = ( _x_ 1 _, . . . , xD_ ) can be described using two binary adjacency matrices: _GD ∈_ R _[D][×][D]_ , for which an entry of 1 in position ( _i, j_ ) indicates the presence of the directed edge _xi → xj_ , and _GB ∈_ R _[D][×][D]_ , for which an entry of 1 in position ( _i, j_ ) indicates the presence of the bidirected edge _xi ↔ xj_ . Throughout this paper, we will use the graph notation _G_ to indicate the tuple _G_ = _{GD, GB}_ . When using ADMGs to represent latent confounding, causal discovery amounts to learning the matrices _GD_ and _GB_ . 

Similar to a DAG, an SCM can be specified to describe the causal relationships implied by an ADMG through the so-called _magnification_ process. As formulated in (Peña, 2016), whenever a bidirected edge _xi ↔ xj_ is present according to _GB_ in an ADMG, we will explicitly add a latent node _um_ to represent the latent parent (confounder) of _xi ↔ xj_ . Then, the SCM of _xi_ can be written as **x** by _xi_ = _fi_ ( **x** pa( _i_ ; _GD_ ) _,_ **u** pa( _i_ ; _GB_ )) + _ϵi_ , where **u** pa( _i_ ; _GB_ ) denotes the latent parents of _xi_ in the set of all latent nodes **u** = ( _u_ 1 _, . . . , uM_ ) added in the magnification process. In compact form, we can write: 

[ **x** _,_ **u** ] = _fG_ ( **x** _,_ **u** ) + _**ϵ** ._ (2) 

The ‘magnified SCM’ will serve as a practical device for learning ADMGs in this paper. Similar to DAGs, magnified SCMs induce an observational distribution on **x** , denoted by _pθ_ ( **x** _[n]_ ; _G_ ). Note that given an ADMG, the magnified SCM is not unique as latent variables may be shared. For example, _x_ 1 _↔ x_ 2, _x_ 2 _↔ x_ 3, _x_ 3 _↔ x_ 1 can be magnified as both _{x_ 1 _← u_ 1 _→ x_ 2 _, x_ 2 _← u_ 2 _→ x_ 3 _, x_ 3 _← u_ 3 _→ x_ 1 _}_ and _{u_ 1 _→ x_ 1 _, u_ 1 _→ x_ 2 _, u_ 1 _→ x_ 3 _}_ . Therefore, ADMG identifiability does not imply the structural identifiability of the magnified SCM. In this paper, we focus on ADMG identifiability. 

## 3.3 DEALING WITH GRAPH UNCERTAINTY 

Additive noise SCMs only guarantee DAG identifiability in the limit of infinite data. In the finite data regime, there is inherent uncertainty in the causal relationships. The Bayesian approach to causal discovery accounts for this uncertainty in a principled manner using probabilistic modeling (Heckerman et al., 2006), in which one’s belief in the true causal graph is updated using Bayes’ rule. This approach is grounded in causal decision theory, which states that rational decision-makers maximize the expected utility over a probability distribution representing their belief in causal graphs (Soto et al., 2019). We can model the causal graph _G_ jointly with observations **x**[1] _, . . . ,_ **x** _[N]_ as 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2023_Causal_Reasoning_in_the_Presence_of_Latent_Confounders_via_Neural_ADMG_Learning_images/iclr_2023_Causal_Reasoning_in_the_Presence_of_Latent_Confounders_via_Neural_ADMG_Learning.pdf-0003-13.png)


3 

Published as a conference paper at ICLR 2023 

where _θ_ denotes the parameters of the likelihood relating the observations to the underlying causal graph. In general, the posterior distribution _pθ_ ( _G|_ **x**[1] _, . . . ,_ **x** _[N]_ ) is intractable. One approach to circumvent this is variational inference (Jordan et al., 1999; Zhang et al., 2018), in which we seek an approximate posterior _qφ_ ( _G_ ) that minimises the KL-divergence KL � _qφ_ ( _G_ ) _||pθ_ ( _G|_ **x**[1] _, . . . ,_ **x** _[N]_ )�. This can be achieved through maximization of the evidence lower bound (ELBO), given by 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2023_Causal_Reasoning_in_the_Presence_of_Latent_Confounders_via_Neural_ADMG_Learning_images/iclr_2023_Causal_Reasoning_in_the_Presence_of_Latent_Confounders_via_Neural_ADMG_Learning.pdf-0004-02.png)


An additional convenience of this approach is that the ELBO serves as a lower bound to the marginal log-likelihood _pθ_ ( **x**[1] _, . . . ,_ **x** _[N]_ ), so it can also be maximized with respect to the model parameters _θ_ to find the parameters that approximately maximize the likelihood of the data (Geffner et al., 2022). 

## 4 ESTABLISHING ADMGS IDENTIFIABILITY UNDER NON-LINEAR SCMS 

To build flexible methods that are capable of discovering causal relationships under the presence of latent confounding, we first need to establish the identifiability of ADMGs under non-linear SCMs. The concept of structural identifiability of ADMGs is formalized in the following definition: 

**Definition 1** (ADMG structural identifiability) **.** _For a distribution pθ_ ( **x** ; _G_ ) _, the ADMG G_ = _{GD, GB} is said to be structurally identifiable from pθ_ ( **x** ; _G_ ) _if there exists no other distribution pθ′_ ( **x** ; _G[′]_ ) _such that G ̸_ = _G[′] and pθ_ ( **x** ; _G_ ) = _pθ′_ ( **x** ; _G[′]_ ) _._ 

Assuming that our model is correctly specified and _pθ_ 0( **x** ; _G_[0] ) denotes the true data generating distribution, then ADMG structural identifiability guarantees that if we find some _pθ_ ( **x** ; _G_ ) = _pθ_ 0( **x** ; _G_[0] ) (by e.g., maximum likelihood learning), we can recover _G_ = _G_[0] . In this section, we seek to establish sufficient conditions under which the ADMG identifiability is satisfied. Let **x** = ( _x_ 1 _, . . . , xD_ ) be a collection of observed random variables, and **u** = ( _u_ 1 _, . . . , uM_ ) be a collection of unobserved (latent) random variables. Our first assumption is that data generating process can be expressed as a specific non-linear additive noise SCM, in which the effect of the observed and latent variables do not modulate each other, see the functional form below: 

**Assumption 1.** _We assume that the data generating process takes the form_ 

[ **x** _,_ **u** ] _[⊤]_ = _fGD,_ **x** ( **x** ; _θ_ ) + _fGB ,_ **u** ( **u** ; _θ_ ) + _**ϵ**_ (5) 

_where each element of_ _**ϵ** is independent of all other variables in the model and θ denotes the parameters of the non-linear functions fGD,_ **x** _and fGB ,_ **u** _._ 

This assumption is one of the elements that separate us from previous work of (Bhattacharya et al., 2021; Maeda & Shimizu, 2020) (linearity is assumed) and (Maeda & Shimizu, 2021) (effects are assumed to be fully decoupled, _xi_ =[�] _m[f][im]_[(] _[x][m][∈]_ **[x]**[pa][(] _[i]_[;] _[G] D_[)][)][+][�] _k[g][ik]_[(] _[u][k][∈]_ **[u]**[pa][(] _[i]_[;] _[G] B_[)][)][+] _ϵi_ ). As discussed in Section 3.2, the discovery of an ADMG between observed variables amounts to the discovery of their ancestral relationships. Therefore, the mapping between ADMGs and their magnified SCMs is not one-to-one, which might cause issues when designing causal discovery methods using SCM-based approaches. To further simplify the underlying latent structure (without losing too much generality), our second assumption assumes that every latent variable is a parentless common cause of a pair of non-adjacent observed variables: 

**Assumption 2** (Latent variables confound pairs of non-adjacent observed variables) **.** _For each latent variable uk in the data generating process, there exists a non-adjacent pair xi and xj that is unique to uk, such that xi ← uk → xj._ 

Arguments for Assumption 2 are made by (Pearl & Verma, 1992), who show that this family of causal graphs is very flexible, and can produce the same conditional independencies amongst the observed variables in any given causal graph. Therefore, it has been argued that without loss of generality, we can assume latent variables to be exogenous, and have exactly two non-adjacent children.[1] This assumption also implies that we can specify a magnified SCM to the ADMG as shown in Section 3.2, which allows us to evaluate and optimize the induced likelihood later on. 

> 1Meanwhile, Evans (2016) show that this graph family cannot induce all possible observational distributions. In Appendix E we show that in the linear non-Gaussian ANM case there exist certain constraints on marginal distributions that cannot be satisfied under Assumption 2. Nonetheless, we could not derive such constraints for the non-linear case. 

4 

Published as a conference paper at ICLR 2023 

Given Assumptions 1 and 2, we now provide three lemmas which allow us to identify the ADMG matrices, _GD_ and _GB_ . These lemmas extend the previous results in Maeda & Shimizu (2021) under our new assumptions above. Note that all _gi_ and _gj_ functions in the lemmas denote functions satisfying the residual faithfulness condition (Maeda & Shimizu (2021)), as detailed in Appendix A. 

**Lemma 1** (Case 1) **.** _Given Assumptions 1 and 2, then_ [ _GB_ ] _i,j_ = 1 _and_ [ _GD_ ] _i,j_ = 0 _if and only if_ 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2023_Causal_Reasoning_in_the_Presence_of_Latent_Confounders_via_Neural_ADMG_Learning_images/iclr_2023_Causal_Reasoning_in_the_Presence_of_Latent_Confounders_via_Neural_ADMG_Learning.pdf-0005-03.png)


**Lemma 2** (Case 2) **.** _Given Assumptions 1 and 2, then_ [ _GB_ ] _i,j_ = 0 _and_ [ _GD_ ] _i,j_ = 0 _if and only if_ 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2023_Causal_Reasoning_in_the_Presence_of_Latent_Confounders_via_Neural_ADMG_Learning_images/iclr_2023_Causal_Reasoning_in_the_Presence_of_Latent_Confounders_via_Neural_ADMG_Learning.pdf-0005-05.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2023_Causal_Reasoning_in_the_Presence_of_Latent_Confounders_via_Neural_ADMG_Learning_images/iclr_2023_Causal_Reasoning_in_the_Presence_of_Latent_Confounders_via_Neural_ADMG_Learning.pdf-0005-06.png)


Since each case leads to mutually exclusive conditional independence/dependence constraints, for any _pθ_ ( **x** ; _G_ ) specified under the assumptions above, there cannot exist some _pθ′_ ( **x** ; _G[′]_ ) such that _G ̸_ = _G[′]_ and _pθ_ ( **x** ; _G_ ) = _pθ′_ ( **x** ; _G[′]_ ), thus structural identifiability is satisfied. This gives our ADMG identifiability result: 

**Proposition 1** (Identifiability of ADMGs under non-linear SCMs) **.** _Assume Assumptions 1 to 2 hold. Then, GD and GB are identifiable for the data generating process specified in Equation_ 5 _._ 

Whilst theoretically we could test for these conditions stated in Lemmas 1 to 3 directly, a more efficient approach is to use differentiable maximum likelihood learning. Assuming the model is correctly specified, then ADMG identifiability ensures the maximum likelihood estimate recovers the true graph in the limit of infinite data (Section 5.5). Hence, we can design the ADMG identification algorithms via maximum likelihood learning. 

## 5 NEURAL ADMG LEARNING 

Whilst we have outlined sufficient conditions under which the underlying ADMG causal structure can be identified, this does not directly provide a framework through which causal discovery and inference can be performed. In this section, we seek to formulate a practical framework for gradientbased ADMG identification. Three challenges that remain are: 1. How can we parameterize the magnified SCM models for ADMG to enable learning flexible causal relationships? 2. How can we optimise our model in the space of ADMG graphs as assumed in Section 4? 3. How do we learn the ADMG causal structure efficiently, whilst accounting for the missing data ( **u** ) and graph uncertainties in the finite data regime? In this section, we present Neural ADMG Learning (NADMG), a novel framework that addresses all three challenges. 

## 5.1 NEURAL AUTO-REGRESSIVE FLOW PARAMETERIZATION 

We assume that our model used to learn ADMGs from data is correctly specified. That is, it can be written in the same magnified SCM form as in Equation 5: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2023_Causal_Reasoning_in_the_Presence_of_Latent_Confounders_via_Neural_ADMG_Learning_images/iclr_2023_Causal_Reasoning_in_the_Presence_of_Latent_Confounders_via_Neural_ADMG_Learning.pdf-0005-14.png)


Following Khemakhem et al. (2020), we factorise the likelihood _pθ_ ( **x** _[n] ,_ **u** _[n] |G_ ) induced by Equation 10 in an autoregressive manner. We can rearrange Equation 10 as 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2023_Causal_Reasoning_in_the_Presence_of_Latent_Confounders_via_Neural_ADMG_Learning_images/iclr_2023_Causal_Reasoning_in_the_Presence_of_Latent_Confounders_via_Neural_ADMG_Learning.pdf-0005-16.png)


where **v** = ( **x** _,_ **u** ) _∈_ R _[D]_[+] _[M]_ , and _G_[˜] is the magnified adjacency matrix on **v** , defined as _G_[˜] _i,j ∈{_ 0 _,_ 1 _}_ if and only if _vi → vj_ . This allows us to express the likelihood as 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2023_Causal_Reasoning_in_the_Presence_of_Latent_Confounders_via_Neural_ADMG_Learning_images/iclr_2023_Causal_Reasoning_in_the_Presence_of_Latent_Confounders_via_Neural_ADMG_Learning.pdf-0005-18.png)


5 

Published as a conference paper at ICLR 2023 

Note that we have omitted the Jacobian-determinant term as it is equal to one since _G_[˜] is acyclic (Mooij et al., 2011). Following Geffner et al. (2022), we adopt an efficient, flexible parameterization for the functions _fi_ taking the form (consistent with Assumption 1) 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2023_Causal_Reasoning_in_the_Presence_of_Latent_Confounders_via_Neural_ADMG_Learning_images/iclr_2023_Causal_Reasoning_in_the_Presence_of_Latent_Confounders_via_Neural_ADMG_Learning.pdf-0006-02.png)


where _ξ_ 1 _,i_ , _ξ_ 2 _,i_ and _ℓi_ ( _i_ = 1 _, ..., D_ + _M_ ) are MLPs. A naïve implementation would require training 3( _D_ + _M_ ) neural networks. Instead, we construct these MLPs so that their weights are shared across nodes as _ξ_ 1 _,i_ ( _·_ ) = _ξ_ 1 _,i_ ( **e** _i, ·_ ), _ξ_ 2 _,i_ ( _·_ ) = _ξ_ 2 _,i_ ( **e** _i, ·_ ) and _ℓi_ ( _·_ ) = _ℓ_ ( **e** _i, ·_ ), with **e** _i ∈_ R _[D]_[+] _[M]_ a trainable embedding that identifies the output and input nodes respectively. 

## 5.2 ADMG LEARNING VIA MAXIMIZING EVIDENCE LOWER BOUND 

Our ADMG identifiability theory in Section 4 suggests that the ground truth ADMG graph can, in principle, be recovered via maximum likelihood learning of _pθ_ ( **x** _|G_ ). However, the aforementioned challenges remain: given a finite number of observations **x**[1] _, . . . ,_ **x** _[N]_ , how do we deal with the corresponding missing data **u**[1] _, . . . ,_ **u** _[N]_ while learning the ADMG? How do we account for graph uncertainties and ambiguities in the finite data regime? To address these issues, N-ADMG takes a Bayesian approach toward ADMG learning. Similar to Section 3.3, we may jointly model the distribution over the ADMG causal graph _G_ , the observations **x**[1] _, . . . ,_ **x** _[N]_ , and the corresponding latent variables **u**[1] _, . . . ,_ **u** _[N]_ , as 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2023_Causal_Reasoning_in_the_Presence_of_Latent_Confounders_via_Neural_ADMG_Learning_images/iclr_2023_Causal_Reasoning_in_the_Presence_of_Latent_Confounders_via_Neural_ADMG_Learning.pdf-0006-06.png)


where _pθ_ ( **x** _[n] ,_ **u** _[n] |G_ ) is the neural SCM model specified in Section 5.1, _θ_ denotes the corresponding model parameters, and _p_ ( _G_ ) is some prior distribution over the graph. Our goal is to learn both the model parameters _θ_ and an approximation to the posterior _qφ_ ( **u**[1] _, . . . ,_ **u** _[N] , G_ ) _≈ pθ_ ( **u**[1] _, . . . ,_ **u** _[N] , G|_ **x**[1] _, . . . ,_ **x** _[N]_ ). This can be achieved jointly using the variational inference framework (Zhang et al., 2018; Kingma & Welling, 2013), in which we maximize the evidence lower bound (ELBO) _L_ ELBO( _θ, φ_ ) _≤_[�] _n_[log] _[ p][θ]_[(] **[x]** _[n]_[)][ given by] 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2023_Causal_Reasoning_in_the_Presence_of_Latent_Confounders_via_Neural_ADMG_Learning_images/iclr_2023_Causal_Reasoning_in_the_Presence_of_Latent_Confounders_via_Neural_ADMG_Learning.pdf-0006-08.png)


In the following sections, we describe our choice of the graph prior _p_ ( _G_ ), and approximate posterior _qφ_ ( **u**[1] _, . . . ,_ **u** _[N] , G_ ) =[�] _n[q][φ]_[(] **[u]** _[n][|]_ **[x]** _[n][, G]_[)] _[q][φ]_[(] _[G]_[)][.][We][will][also][demonstrate][that][maximizing] _L_ ELBO( _θ, φ_ ) recovers the true ADMG causal graph in the limit of infinite data. See Section 5.5. 

## 5.3 CHOICE OF PRIOR OVER ADMG GRAPHS 

As discussed in Section 3.2, the ADMG _G_ can be parameterized by two binary adjacency matrices, _GD_ whose entries indicate the presence of a directed edge, and _GB_ whose edges indicate the presence of a bidirected edge. As discussed in Section 4, a necessary assumption for structural identifiability is that each latent variable is a parent-less confounder of a pair of non-adjacent observed variables. This further implies that the underlying ADMG must be bow-free (both a directed and a bidirected edge cannot exist between the same pair of observed variables). This constraint can be imposed by leveraging the bow-free constrain penalty introduced by Bhattacharya et al. (2021), 

_h_ ( _GD, GB_ ) = trace � _e[G][D]_[�] _− D_ + sum ( _GD ◦ GB_ ) (16) 

which is non-negative and zero only if ( _GD, GB_ ) is a bow-free ADMG. As suggested in Geffner et al. (2022), we implement the prior as 

_p_ ( _G_ ) _∝_ exp � _−λs_ 1 _∥GD∥F_[2] _[−][λ][s]_[2] _[∥][G][B][∥]_[2] _F[−][ρh]_[(] _[G][D][, G][B]_[)][2] _[ −][αh]_[(] _[G][D][, G][B]_[)] � (17) where the coefficients _α_ and _ρ_ are increased whilst maximizing _L_ ELBO( _θ, φ_ ), following an augmented Lagrangian scheme (Nemirovski, 1999). Prior knowledge about the sparseness of the graph is introduced by penalizing the norms _∥GD∥_[2] _F_[and] _[ ∥][G][B][∥]_[2] _F_[with scaling coefficients] _[ λ][s]_[1][ and] _[ λ][s]_[2][.] 

6 

Published as a conference paper at ICLR 2023 

## 5.4 CHOICE OF VARIATIONAL APPROXIMATION 

We seek to approximate the intractable true posterior _pθ_ ( **u**[1] _, . . . ,_ **u** _[N] , G|_ **x**[1] _, . . . ,_ **x** _[N]_ ) using the variational distribution _qφ_ ( **u**[1] _, . . . ,_ **u** _[N] , G_ ). We assume the following factorized approximate posterior 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2023_Causal_Reasoning_in_the_Presence_of_Latent_Confounders_via_Neural_ADMG_Learning_images/iclr_2023_Causal_Reasoning_in_the_Presence_of_Latent_Confounders_via_Neural_ADMG_Learning.pdf-0007-03.png)


For _qφ_ ( _G_ ), we use a product of Bernoulli distributions for each potential directed edge in _GD_ and bidirected edge in _GB_ . For _GD_ , edge existence and edge orientation are parameterized separately using the ENCO parameterization (Lippe et al., 2021). For _qφ_ ( **u** _[n] |_ **x** _[n]_ ), we apply amortized VI as in VAE literature (Kingma & Welling, 2013), where _qφ_ ( **u** _[n] |_ **x** _[n]_ ) is parameterized as a Gaussian distribution whose mean and variance is determined by passing the **x** _[n]_ through an inference MLP . 

## 5.5 MAXIMIZING _L_ ELBO( _θ, φ_ ) RECOVERS THE GROUND TRUTH ADMG 

In Section 4, we have proved the structural identifiability of ADMGs. In this section, we further show that under certain assumptions, maximizing _L_ ELBO( _θ, φ_ ) recovers the true ADMG graph (denoted by _G_[0] ) in the infinite data limit. This result is stated in the following proposition: 

**Proposition 2** (Maximizing _L_ ELBO( _θ, φ_ ) recovers the ground truth ADMG) **.** _Assume that:_ 

- _Assumptions 1 and 2 (hence the identifiability of ADMGs) holds for the model pθ_ ( **x** ; _G_ ) _._ 

- _The model is correctly specified (∃θ[∗] such that pθ∗_ ( **x** ; _G_[0] ) _recovers the data-generating process)._ 

- _Regularity condition: for all θ and G we have_ E _p_ ( **x** ; _G_ 0) [ _|_ log _pθ_ ( **x** ; _G_ ) _|_ ] _< ∞._ 

- _The variational family of qφ_ ( **u** _|_ **x** _, G_ ) _is flexible enough, i.e., it contains pθ_ ( **u** _|_ **x** _, G_ ) _._ 

_Then, the solution_ ( _θ[′] , qφ[′]_[(] _[G]_[))] _[ that maximizes][ L][ELBO]_[(] _[θ, φ]_[)] _[ satisfies][ q] φ[′]_[(] _[G]_[) =] _[ δ]_[(] _[G]_[ =] _[ G]_[0][)] _[.]_ 

The proof of Proposition 2 can be found in Appendix B, which justifies performing causal discovery by maximizing ELBO of the N-ADMG model. Once the model has been trained and the ADMG has been recovered, we can use the N-ADMG to perform causal inference as detailed in Appendix C. 

## 6 EXPERIMENTS 

We evaluate N-ADMG in performing both causal discovery and causal inference on a number of synthetic and real-world datasets. Note that we run our model _both with and without the bow-free constraint, identified as N-BF-ADMG (our full model) and N-ADMG (for ablation purpose)_ , respectively. We compare the performance of our model against five baselines: DECI (Geffner et al., 2022) (which we refer to as N-DAG for consistency), FCI (Spirtes et al., 2000), RCD (Maeda & Shimizu, 2020), CAM-UV (Maeda & Shimizu, 2021), and DCD (Bhattacharya et al., 2021). We evaluate the causal discovery performance using F1 scores for directed and and bidirected adjacency. The expected values of these metrics are reported using the learned graph posterior (which is deterministic for RCD, CAM-UV, and DCD). Causal inference is evaluated using the expected ATE as described in Appendix C. We evaluate the causal inference performance of the causal discovery benchmarks by fixing _q_ ( _G_ ) to either deterministic or uniform categorical distributions on the learned causal graphs, then learning a non-linear flow-based ANM by optimizing Equation 15 in an identical manner to N-ADMG. A full list of results and details of the experimental set-up are included in Appendix F. Our implementation will be available at `https://github.com/microsoft/causica` . 

## 6.1 SYNTHETIC FORK-COLLIDER DATASET 

We construct a synthetic fork-collider dataset consisting of five nodes (Figure 1a). The datagenerating process is a non-linear ANM with Gaussian noise. Variable pairs ( _x_ 2 _, x_ 3), and ( _x_ 3 _, x_ 4) are latent-confounded, whereas ( _x_ 4 _, x_ 5) share a observed confounder. We evaluate both causal discovery and inference performances. For discovery, we evaluate F-score measure on both _GD_ and _GB_ . For causal inference, we choose _x_ 4 as the treatment variable taking and _x_ 2, _x_ 3 and _x_ 5 as the 

7 

Published as a conference paper at ICLR 2023 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2023_Causal_Reasoning_in_the_Presence_of_Latent_Confounders_via_Neural_ADMG_Learning_images/iclr_2023_Causal_Reasoning_in_the_Presence_of_Latent_Confounders_via_Neural_ADMG_Learning.pdf-0008-01.png)


**----- Start of picture text -----**<br>
x 2 x 1 x 2 x 1 x 2 x 1<br>x 3 x 4 x 5 x 3 x 4 x 5 x 3 x 4 x 5<br>(a)  Ground truth. (b)  N-BF-ADMG. (c)  N-ADMG (no constraint).<br>x 2 x 1 x 2 x 1 x 2 x 1<br>x 3 x 4 x 5 x 3 x 4 x 5 x 3 x 4 x 5<br>(d)  RCD. (e)  DCD. (f)  CAM-UV.<br>**----- End of picture text -----**<br>


**Figure 1:** ADMG identification results on fork-collider dataset. 

response variables, and evaluate ATE RMSE to benchmark performance. It is therefore crucial that discovery methods do not misidentify latent confounded variables as having a direct cause between them, as this would result in biased ATE estimates. 

Figure 1b-1f visualizes the causal discovery results of different methods, and Table 1 summarizes both discovery and inference metrics. While no methods can fully recover the ground truth graph, our method N-BF-ADMG-G and CAM-UV achieved the best overall performance. N-BF-ADMG-G and N- ADMG-G on average are able to recover all bidirected edges from the data, while CAM-UV can only recover half of the latent variables. Without the bow-free constraint, N-ADMG-G discovers a di- 

**Table 1:** Causal discovery and inference results for the fork-collider dataset. The table shows the mean and stan- 

||**METHOD**<br>N-BF-ADMG-G<br>N-ADMG-G<br>N-DAG-G<br>FCI<br>RCD<br>CAM-UV|_GD_ FSCORE <br>0.64 (0.06)<br>0.49 (0.02)<br>0.50 (0.00)<br>0.00 (0.00)<br>0.00 (0.00)<br>**0.80 (0.00)**|_GB_ FSCORE <br>**0.93 (0.07)**<br>**0.99 (0.00)**<br>0.00 (0.00)<br>0.75 (0.00)<br>0.54 (0.00)<br>0.67 (0.00)|ATE RMSE<br>**0.022 (0.003)**<br>0.239 (0.067)<br>**0.046 (0.025)**<br>0.072 (0.015)<br>0.206 (0.029)<br>**0.017 (0.003)**|
|---|---|---|---|---|
||DCD|0.00 (0.00)|0.67 (0.00)|0.208 (0.064)|



rected edge from _x_ 4 to _x_ 3, which results in poor ATE RMSE performance. On the other hand, DAG-based method (N-DAG-G) is not able to deal with latent confounders, resulting in the poor f-scores in both _GD_ and _GB_ . Linear ANM-based methods (RCD and DCD) perform significantly worse than other methods, resulting in 0 f-scores for directed matrices and largest ATE errors. This demonstrates the necessity for introducing non-linear assumptions. 

## 6.2 RANDOM CONFOUNDED ER SYNTHETIC DATASET 

We generate synthetic datasets from ADMG extension of _Erd˝os-Rényi_ (ER) graph model (Lachapelle et al., 2019). We first sample random ADMGs from ER model, and simulate each variable using a randomly sampled nonlinear ANM. Latent confounders are then removed from the training set. See Appendix G.2 for details. We consider the number of nodes, directed edges and latent confounders triplets ( _d, e, m_ ) _∈{_ (4 _,_ 6 _,_ 2) _,_ (8 _,_ 20 _,_ 6) _,_ (12 _,_ 50 _,_ 10) _}_ . The resulting datasets are identified as **ER** ( _d, e, m_ ). Figure 2 compares the performance of N-ADMG with the baselines. All variants of N-ADMG outperform the baselines for most datasets, highlighting its effectiveness relative to other methods (even those that employ similar assumptions). Similar to the fork-collider dataset, we see that methods operating under the assumption of linearity perform poorly when the data-generating process is non-linear. It is worth noting that even when the exogenous noise of N- ADMG is misspecified, its still exceeds that of other methods in most cases. This robustness is a desirable property as in many settings the form of exogenous noise is unknown. 

## 6.3 INFANT HEALTH AND DEVELOPMENT PROGRAM (IHDP) DATASET 

For the real-world datasets, we evaluate treatment effect estimation performances on infant health and development program data (IHDP). This dataset contains measurements of both infants and their mother during real-life data collected in a randomized experiment. The main task is to estimate the effect of home visits by specialists on future cognitive test scores of infants, where the ground 

8 

Published as a conference paper at ICLR 2023 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2023_Causal_Reasoning_in_the_Presence_of_Latent_Confounders_via_Neural_ADMG_Learning_images/iclr_2023_Causal_Reasoning_in_the_Presence_of_Latent_Confounders_via_Neural_ADMG_Learning.pdf-0009-01.png)


**Figure 2:** Causal discovery results for synthetic ER datasets. For readability, the N-ADMG results are connected with lines. The figure shows mean results across five randomly generated datasets. 

truth outcomes are simulated as in (Hill, 2011). To make the task more challenging, additional confoundings are introduced by removing a subset (non-white mothers) of the treated population. More details can be found in Appendix G.3. We first perform causal discovery to learn the underlying ADMG of the dataset, and then perform causal inference. Since the true causal graph is unknown, we evaluate the causal inference performance of each method by estimating the ATE RMSE. 

Apart from the aforementioned baselines, here we introduce four more methods: PC-DWL (PC algorithm for discovery, DoWhy (Sharma et al., 2021) linear adjustment for inference); PC-DwNL (PC for discovery, DoWhy double machine learning for inference); N-DAG-G-DwL (N-DAG Gaussian for discovery, linear adjustment for inference); and N-DAG-S-DwL (N-DAG Spline for discovery, linear adjustment for inference). Results are summarized in Figures 3a to 3c. Generally, models with non-Gaussian exogenous assumptions tend to have lower ATE estimation errors; while models with linear assumptions (RCD and DCD) have the worst ATE RMSE. Interestingly, the DoWhy-based plug-in estimators tend to worsen the performances of SCM models. However, regardless of the assumptions made on exogenous noise, our method (N-BF-ADMG-G and N-BF-ADMG-S) consistently outperforms all other baselines with the same noise. It is evident that for causal inference in real-world datasets, the ability of N-BF-ADMG to handle latent confoundings and nonlinear causal relationships becomes very effective. 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2023_Causal_Reasoning_in_the_Presence_of_Latent_Confounders_via_Neural_ADMG_Learning_images/iclr_2023_Causal_Reasoning_in_the_Presence_of_Latent_Confounders_via_Neural_ADMG_Learning.pdf-0009-05.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2023_Causal_Reasoning_in_the_Presence_of_Latent_Confounders_via_Neural_ADMG_Learning_images/iclr_2023_Causal_Reasoning_in_the_Presence_of_Latent_Confounders_via_Neural_ADMG_Learning.pdf-0009-06.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2023_Causal_Reasoning_in_the_Presence_of_Latent_Confounders_via_Neural_ADMG_Learning_images/iclr_2023_Causal_Reasoning_in_the_Presence_of_Latent_Confounders_via_Neural_ADMG_Learning.pdf-0009-07.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2023_Causal_Reasoning_in_the_Presence_of_Latent_Confounders_via_Neural_ADMG_Learning_images/iclr_2023_Causal_Reasoning_in_the_Presence_of_Latent_Confounders_via_Neural_ADMG_Learning.pdf-0009-08.png)


**----- Start of picture text -----**<br>
(a)  Gaussian exogenous noise. (b)  Spline exogenous noise. (c)  DoWhy baselines.<br>**----- End of picture text -----**<br>


**Figure 3:** Causal inference results for the IHDP dataset. The figure shows mean _±_ standard error 

## 7 CONCLUSION AND FUTURE WORK 

In this work, we proposed Neural ADMG Learning (N-ADMG), a novel framework for gradientbased causal reasoning in the presence of latent confounding for nonlinear SCMs. We established identifiability theory for nonlinear ADMGs under latent confounding, and proposed a practical ADMG learning algorithm that is both flexible and efficient. In future work, we will further extend our framework on how to more general settings (e.g., the effect from observed and latent variables can modulate in certain forms; latent variables can confound adjacent variables; etc), and how to improve certain modelling choices such as variational approximation qualities over both causal graph and latent variables 

9 

Published as a conference paper at ICLR 2023 

## REPRODUCIBILITY STATEMENT 

A number of efforts have been made/will be made for the sake of reproducibility. First, we open source our package in our github page `github.com/microsoft/causica/tree/v0.0.0` . In addition, in this paper we included clear explanations of any assumptions and a complete proof of the claims in the Appendix, as well as model settings and hyperparameters. All datasets used in this paper are either publicly available data, or synthetic data whose generation process is described in detail. 

## REFERENCES 

- Alexis Bellot and Mihaela van der Schaar. Deconfounded score method: Scoring dags with dense unobserved confounding. _arXiv preprint arXiv:2103.15106_ , 2021. 

- Rohit Bhattacharya, Tushar Nagarajan, Daniel Malinsky, and Ilya Shpitser. Differentiable causal discovery under unmeasured confounding. In _International Conference on Artificial Intelligence and Statistics_ , pp. 2314–2322. PMLR, 2021. 

Ruichu Cai, Feng Xie, Clark Glymour, Zhifeng Hao, and Kun Zhang. Triad constraints for learning causal structure of latent variables. _Advances in neural information processing systems_ , 32, 2019. 

Wenyu Chen, Mathias Drton, and Ali Shojaie. Causal structural learning via local graphs. _arXiv preprint arXiv:2107.03597_ , 2021. 

- Tom Claassen, Joris M Mooij, and Tom Heskes. Learning sparse causal models is not NP-hard. In _Proceedings of the Twenty-Ninth Conference on Uncertainty in Artificial Intelligence_ , pp. 172– 181, 2013. 

- Diego Colombo, Marloes H Maathuis, Markus Kalisch, and Thomas S Richardson. Learning highdimensional directed acyclic graphs with latent and selection variables. _The Annals of Statistics_ , pp. 294–321, 2012. 

- Alexander D’Amour. On multi-cause approaches to causal inference with unobserved counfounding: Two cautionary failure cases and a promising alternative. In Kamalika Chaudhuri and Masashi Sugiyama (eds.), _Proceedings of the Twenty-Second International Conference on Artificial Intelligence and Statistics_ , volume 89 of _Proceedings of Machine Learning Research_ , pp. 3478–3486. PMLR, 16–18 Apr 2019. 

- Robin J Evans. Graphs for margins of Bayesian networks. _Scandinavian Journal of Statistics_ , 43 (3):625–648, 2016. 

- Tomas Geffner, Javier Antoran, Adam Foster, Wenbo Gong, Chao Ma, Emre Kiciman, Amit Sharma, Angus Lamb, Martin Kukla, Nick Pawlowski, et al. Deep end-to-end causal inference. _arXiv preprint arXiv:2202.02195_ , 2022. 

- Clark Glymour, Kun Zhang, and Peter Spirtes. Review of causal discovery methods based on graphical models. _Frontiers in genetics_ , 10:524, 2019. 

- David Heckerman, Christopher Meek, and Gregory Cooper. A bayesian approach to causal discovery. In _Innovations in Machine Learning_ , pp. 1–28. Springer, 2006. 

- Jennifer L Hill. Bayesian nonparametric modelling for causal inference. _Journal of Computational and Graphical Statistics_ , 20(1):217–240, 2011. 

Michael I Jordan, Zoubin Ghahramani, Tommi S Jaakkola, and Lawrence K Saul. An introduction to variational methods for graphical models. _Machine learning_ , 37(2):183–233, 1999. 

- Ilyes Khemakhem, Diederik Kingma, Ricardo Monti, and Aapo Hyvarinen. Variational autoencoders and nonlinear ICA: A unifying framework. In _International Conference on Artificial Intelligence and Statistics_ , pp. 2207–2217. PMLR, 2020. 

- Diederik P Kingma and Max Welling. Auto-encoding variational Bayes. _arXiv preprint arXiv:1312.6114_ , 2013. 

10 

Published as a conference paper at ICLR 2023 

- Sébastien Lachapelle, Philippe Brouillard, Tristan Deleu, and Simon Lacoste-Julien. Gradient-based neural DAG learning. _arXiv preprint arXiv:1906.02226_ , 2019. 

- Phillip Lippe, Taco Cohen, and Efstratios Gavves. Efficient neural causal discovery without acyclicity constraints. _arXiv preprint arXiv:2107.10483_ , 2021. 

- Christos Louizos, Uri Shalit, Joris M Mooij, David Sontag, Richard Zemel, and Max Welling. Causal effect inference with deep latent-variable models. _Advances in neural information processing systems_ , 30, 2017. 

- Takashi Nicholas Maeda and Shohei Shimizu. RCD: Repetitive causal discovery of linear nongaussian acyclic models with latent confounders. In _International Conference on Artificial Intelligence and Statistics_ , pp. 735–745. PMLR, 2020. 

- Takashi Nicholas Maeda and Shohei Shimizu. Causal additive models with unobserved variables. In _Uncertainty in Artificial Intelligence_ , pp. 97–106. PMLR, 2021. 

- Sara Mohammad-Taheri, Robert Ness, Jeremy Zucker, and Olga Vitek. Do-calculus enables causal reasoning with latent variable models. _arXiv preprint arXiv:2102.06626_ , 2021. 

- Joris M Mooij, Dominik Janzing, Tom Heskes, and Bernhard Schölkopf. On causal discovery with cyclic additive noise model. _Advances in neural information processing systems 24_ , 2011. 

- Pablo Morales-Alvarez, Wenbo Gong, Angus Lamb, Simon Woodhead, Simon Peyton Jones, Nick Pawlowski, Miltiadis Allamanis, and Cheng Zhang. Simultaneous missing value imputation and structure learning with groups. In _Advances in Neural Information Processing Systems_ , 2022. 

- A Nemirovski. Optimization II. Numerical methods for nonlinear continuous optimization. 1999. 

- Christopher Nowzohour, Marloes H Maathuis, Robin J Evans, and Peter Bühlmann. Distributional equivalence and structure learning for bow-free acyclic path diagrams. _Electronic Journal of Statistics_ , 11(2):5342–5374, 2017. 

- Judea Pearl. Graphs, causality, and structural equation models. _Sociological Methods & Research_ , 27(2):226–284, 1998. 

Judea Pearl. Causal inference in statistics: An overview. _Statistics surveys_ , 3:96–146, 2009. 

- Judea Pearl. An introduction to causal inference. _The International Journal of Biostatistics_ , 6(2): 1–62, 2010. 

Judea Pearl. On measurement bias in causal inference. _arXiv preprint arXiv:1203.3504_ , 2012. 

- Judea Pearl and Thomas S Verma. A statistical semantics for causation. _Statistics and Computing_ , 2(2):91–95, 1992. 

- Jose M Peña. Learning acyclic directed mixed graphs from observations and interventions. In _Conference on Probabilistic Graphical Models_ , pp. 392–402. PMLR, 2016. 

- Jonas Peters, Joris M Mooij, Dominik Janzing, and Bernhard Schölkopf. Causal discovery with continuous additive noise models. 2014. 

- Rajesh Ranganath and Adler Perotte. Multiple causal inference with latent confounding. _arXiv preprint arXiv:1805.08273_ , 2018. 

- Thomas Richardson and Peter Spirtes. Ancestral graph Markov models. _The Annals of Statistics_ , 30 (4):962–1030, 2002. 

- Amit Sharma, Vasilis Syrgkanis, Cheng Zhang, and Emre Kıcıman. Dowhy: Addressing challenges in expressing and validating causal assumptions. _arXiv preprint arXiv:2108.13518_ , 2021. 

- Ilya Shpitser, Robin J Evans, Thomas S Richardson, and James M Robins. Introduction to nested markov models. _Behaviormetrika_ , 41(1):3–39, 2014. 

11 

Published as a conference paper at ICLR 2023 

Mauricio Gonzalez Soto, David Danks, Hugo J Escalante Balderas, and L Enrique Sucar. Choosing with unknown causal information: Action-outcome probabilities for decision making can be grounded in causal models. _arXiv preprint arXiv:1907.11752_ , 2019. 

- Peter Spirtes. Introduction to causal inference. _Journal of Machine Learning Research_ , 11(5), 2010. 

- Peter Spirtes, Clark N Glymour, Richard Scheines, and David Heckerman. _Causation, prediction, and search_ . MIT press, 2000. 

- Jin Tian and Judea Pearl. A general identification condition for causal effects. In _Aaai/iaai_ , pp. 567–573, 2002. 

- Dustin Tran and David M Blei. Implicit causal models for genome-wide association studies. _arXiv preprint arXiv:1710.10742_ , 2017. 

- Thomas Verma and Judea Pearl. Equivalence and synthesis of causal models. In _Proceedings of the Sixth Annual Conference on Uncertainty in Artificial Intelligence_ , pp. 255–270, 1990. 

- Y Samuel Wang and Mathias Drton. Causal discovery with unobserved confounding and nonGaussian data. _arXiv preprint arXiv:2007.11131_ , 2020. 

- Yixin Wang and David M Blei. The blessings of multiple causes. _Journal of the American Statistical Association_ , 114(528):1574–1596, 2019. 

- Cheng Zhang, Judith Bütepage, Hedvig Kjellström, and Stephan Mandt. Advances in variational inference. _IEEE transactions on pattern analysis and machine intelligence_ , 41(8):2008–2026, 2018. 

- Jiji Zhang. On the completeness of orientation rules for causal discovery in the presence of latent confounders and selection bias. _Artificial Intelligence_ , 172(16-17):1873–1896, 2008. 

- Xun Zheng, Bryon Aragam, Pradeep K Ravikumar, and Eric P Xing. DAGs with no tears: Continuous optimization for structure learning. _Advances in Neural Information Processing Systems_ , 31, 2018. 

- Xun Zheng, Chen Dan, Bryon Aragam, Pradeep Ravikumar, and Eric Xing. Learning sparse nonparametric DAGs. In _International Conference on Artificial Intelligence and Statistics_ , pp. 3414– 3425. PMLR, 2020. 

12 

Published as a conference paper at ICLR 2023 

## APPENDIX 

## A PROOF OF LEMMAS 1 TO 3 

First we describe the residual faithfulness condition inherited from (Maeda & Shimizu, 2021) in our Lemmas 1 to 3: 

**Definition 2** (Residual faithfulness condition) **.** _We say that nonlinear functions gi, gj satisfies the residual faithfulness condition, if: for any two arbitrary subset of_ **x** _, denote by M and N , when both_ ( _xi−gi_ ( _M_ )) _and_ ( _xi−gj_ ( _M_ )) _have terms involving the same exogenous noise ϵk, then_ ( _xi−gi_ ( _M_ )) _and_ ( _xi − gj_ ( _M_ )) _are mutually dependent._ 

Next, we provide the proof for Lemmas 1 to 3 in Section 4. To prove those lemmas, we need the help of the following lemma, which extends Lemma A in (Maeda & Shimizu, 2021), except we don’t assume any form for _gi_ other than non-linearity. 

**Lemma 4.** _Let s_ ( _xi_ ) _denote an arbitrary function of xi. The residual of s_ ( _xi_ ) _regressed onto_ **x** _−i cannot be independent of ϵi:_ 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2023_Causal_Reasoning_in_the_Presence_of_Latent_Confounders_via_Neural_ADMG_Learning_images/iclr_2023_Causal_Reasoning_in_the_Presence_of_Latent_Confounders_via_Neural_ADMG_Learning.pdf-0013-07.png)


_Proof._ Assume that [ _s_ ( _xi_ ) _−gi_ ( **x** _−i_ ) _ϵi_ ] holds, then **x** _−i_ must contain at least one descendent of _xi_ as it must have dependence on the noise _ϵi_ to cancel effect of _ϵi_ in _s_ ( _xi_ ). We can express _gi_ ( **x** _−i_ ) as _ui_ ( _**ϵ**_ ). Since _gi_ operates on variables defined by non-linear transformations of the exogenous noise terms, we cannot express _ui_ as _ai_ ( _**ϵ** −i_ ) + _bi_ ( _ϵi_ ). **x** _−i_ contains a descendent of _xi_ , so _**ϵ** −i_ includes at least one noise term _ϵk_ that satisfies _xi ϵk_ (i.e. is not in _xi_ ). Thus, terms containing _ϵi_ cannot be fully removed from _s_ ( _xi_ ) _− gi_ ( **x** _−i_ ) and so [ _s_ ( _xi_ ) _− gi_ ( **x** _−i_ ) _ϵi_ ] does not hold. 

## A.1 PROOF OF LEMMA 1 

_Proof._ Define _gi_ and _gj_ as _gi_ ( **x** _−i_ ) = _fi,_ **x** ( **pax** ( _i_ ))+ _gi[′]_[(] **[x]** _[−][i]_[)][ and] _[ g][j]_[(] **[x]** _[−][j]_[) =] _[ f][j,]_ **[x]**[(] **[pa] x**[(] _[j]_[))+] _[g] i[′]_[(] **[x]** _[−][j]_[)] respectively. Then, Equation 6 becomes equivalent to 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2023_Causal_Reasoning_in_the_Presence_of_Latent_Confounders_via_Neural_ADMG_Learning_images/iclr_2023_Causal_Reasoning_in_the_Presence_of_Latent_Confounders_via_Neural_ADMG_Learning.pdf-0013-11.png)


Given Lemma 4 and following the same arguments as in Maeda & Shimizu (2021), this is equivalent to 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2023_Causal_Reasoning_in_the_Presence_of_Latent_Confounders_via_Neural_ADMG_Learning_images/iclr_2023_Causal_Reasoning_in_the_Presence_of_Latent_Confounders_via_Neural_ADMG_Learning.pdf-0013-13.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2023_Causal_Reasoning_in_the_Presence_of_Latent_Confounders_via_Neural_ADMG_Learning_images/iclr_2023_Causal_Reasoning_in_the_Presence_of_Latent_Confounders_via_Neural_ADMG_Learning.pdf-0013-14.png)


The first implies the existence of an unobserved mediator between _xj_ and _xi_ , the second implies the existence of an unobserved mediator between _xi_ and _xj_ , and the third implies the existence of an unobserved confounder. Given the assumption of latent variables being confounders and minimality, this indicates the presence of a latent confounder and no direct cause between _xi_ and _xj_ . 

## A.2 PROOF OF LEMMA 2 

_Proof._ When Equation 7 holds, Equation 6 does not. Thus, there is no unobserved confounder between _xi_ and _xj_ . Assume that _xj_ is a direct cause of _xi_ , and that Equation 7 is satisfied for _gi_ and _gj_ . _xi_ contains a nonlinear function of _ϵj_ that cannot be removed by _gi_ ( **x** _−_ ( _i,j_ )), thus ( _xi − gi_ ( **x** _−_ ( _i,j_ )))�� _ϵj_ . Similarly, ( _xj − gj_ ( **x** _−_ ( _i,j_ )))�� _ϵi_ . Thus, we have [( _xi − gi_ ( **x** _−_ ( _i,j_ ))��( _xj − gj_ ( **x** _−_ ( _i,j_ )))] which contradicts our initial assumption. The same arguments apply when _xi_ is a direct cause of _xj_ , implying that there can be no causal relationship between _xi_ and _xj_ . 

## A.3 PROOF OF LEMMA 3 

_Proof._ When Equation 9 holds, Equation 6 does not and so there is no latent confounder between _xi_ and _xj_ . When _Equation_ 8 holds, Equation 7 does not hold and so there is a direct causal relationship between _xi_ and _xj_ . 

13 

Published as a conference paper at ICLR 2023 

Assume that _xj_ is a direct cause of _xi_ . Define _g_ 1( **x** _−i_ ) = _fi,_ **x** ( **pax** ( _i_ )) and _g_ 2( **x** _−_ ( _i,j_ )) = _fj,_ **x** ( **pax** ( _j_ )), giving _xi − gi_ ( **x** _−i_ ) = _fi,_ **u** ( **pau** ( _i_ )) + _ϵi_ and _xj − gj_ ( **x** _−_ ( _i,j_ )) = _fj,_ **u** ( **pau** ( _j_ )) + _ϵj_ . When there is no latent confounder, _fi,_ **u** ( **pau** ( _i_ )) _fj,_ **u** ( **pau** ( _j_ )). Thus, ( _fi,_ **u** ( **pau** ( _i_ )) + _ϵi_ ) ( _fj,_ **u** ( **pau** ( _j_ )) + _ϵj_ ) hods and Equation 9 is satisfied. 

Now, assume that _xi_ is a direct cause of _xj_ . Using Lemma 4, we have 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2023_Causal_Reasoning_in_the_Presence_of_Latent_Confounders_via_Neural_ADMG_Learning_images/iclr_2023_Causal_Reasoning_in_the_Presence_of_Latent_Confounders_via_Neural_ADMG_Learning.pdf-0014-03.png)


Similarly, since _xj_ is a function of _xi_ we also have 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2023_Causal_Reasoning_in_the_Presence_of_Latent_Confounders_via_Neural_ADMG_Learning_images/iclr_2023_Causal_Reasoning_in_the_Presence_of_Latent_Confounders_via_Neural_ADMG_Learning.pdf-0014-05.png)


Collectively, this implies 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2023_Causal_Reasoning_in_the_Presence_of_Latent_Confounders_via_Neural_ADMG_Learning_images/iclr_2023_Causal_Reasoning_in_the_Presence_of_Latent_Confounders_via_Neural_ADMG_Learning.pdf-0014-07.png)


which contradicts Equation 9. Thus, if Equation 9 is satisfied then _xj_ is a direct cause of _xi_ . 

B PROOF OF PROPOSITION 2 

To prove Proposition 2, we need the following lemma: 

**Lemma 5.** _Assume a variational distribution qφ_ ( _G_ ) _over a space of graphs Gφ, where each graph G ∈Gφ has a non-zero associated weight wφ_ ( _G_ ) _. With the soft prior p_ ( _G_ ) _defined as Equation_ 17 _and bounded λ_ 1 _, λ_ 2 _, ρ, α, we have_ 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2023_Causal_Reasoning_in_the_Presence_of_Latent_Confounders_via_Neural_ADMG_Learning_images/iclr_2023_Causal_Reasoning_in_the_Presence_of_Latent_Confounders_via_Neural_ADMG_Learning.pdf-0014-12.png)


_Proof._ This directly follows from Lemma 1 of (Geffner et al., 2022). 

Now we can proceed to prove Proposition 2: 

_Proof._ For N-ADMG, in the infinite data limit _L_ ELBO becomes 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2023_Causal_Reasoning_in_the_Presence_of_Latent_Confounders_via_Neural_ADMG_Learning_images/iclr_2023_Causal_Reasoning_in_the_Presence_of_Latent_Confounders_via_Neural_ADMG_Learning.pdf-0014-16.png)


where the zeroing of the KL divergence follows from Lemma 5. Given fixed _θ_ , the optimal posterior _q[∗]_ ( **u** _n|_ **x** _n, G_ ) satisfies _q[∗]_ ( **u** _n|_ **x** _n, G_ ) = _pθ_ ( **u** _n|_ **x** _n, G_ ) due to the flexibility assumption. Thus, 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2023_Causal_Reasoning_in_the_Presence_of_Latent_Confounders_via_Neural_ADMG_Learning_images/iclr_2023_Causal_Reasoning_in_the_Presence_of_Latent_Confounders_via_Neural_ADMG_Learning.pdf-0014-18.png)


where _p_ ( **x** ; _G_[0] ) denotes the data generation distribution with the ground truth ADMG, _G_[0] . Let ( _θ[∗] , G[∗]_ ) = arg max � _p_ ( **x** ; _G_[0] ) log _pθ_ ( **x** _|G_ ) _d_ **x** be the MLE solution. Since[�] _G∈Gφ[w][φ]_[(] _[G]_[)][=][1][,] _wφ_ ( _G_ ) _>_ 0, we have 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2023_Causal_Reasoning_in_the_Presence_of_Latent_Confounders_via_Neural_ADMG_Learning_images/iclr_2023_Causal_Reasoning_in_the_Presence_of_Latent_Confounders_via_Neural_ADMG_Learning.pdf-0014-20.png)


14 

Published as a conference paper at ICLR 2023 

with the optimal value of[�] _G∈Gφ[w][φ]_[(] _[G]_[)][E] _[p]_[(] **[x]**[;] _[G]_[0][)][ [log] _[ p][θ]_[(] **[x]** _[|][G]_[)]][ is achieved when every graph] _[ G][∈] Gφ_ and associated parameter _θG_ satisfies 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2023_Causal_Reasoning_in_the_Presence_of_Latent_Confounders_via_Neural_ADMG_Learning_images/iclr_2023_Causal_Reasoning_in_the_Presence_of_Latent_Confounders_via_Neural_ADMG_Learning.pdf-0015-02.png)


Since the model is correctly specified, the MLE solution ( _θ[∗] , G[∗]_ ) satisfies 

E _p_ ( **x** ; _G_ 0) [log _pθ[∗]_ ( **x** _|G[∗]_ )] = E _p_ ( **x** ; _G_ 0) �log _p_ ( **x** ; _G_[0] )� 

Therefore, condition Equation 29 implies for every graph _G[′] ∈Gφ_ , _G[′]_ = _G_[0] under the regularity condition; or equivalently, _Gφ_ = _{G[′]_ = _G_[0] _}_ . This proves our statement that _qφ[′]_[(] _[G]_[)][=] _[δ]_[(] _[G]_[=] _[G][′]_[)][,] where _G[′]_ = _G_[0] . 

## C ESTIMATING TREATMENT EFFECTS 

For all experiments we consider, the causal quantity of interest we wish to estimate is the expected average treatment effect (ATE), E _qφ_ ( _G_ ) [ATE( **a** _,_ **b** _|G_ )], where the expectation is taken with respect to our learned posterior over causal graphs _qφ_ ( _G_ ): 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2023_Causal_Reasoning_in_the_Presence_of_Latent_Confounders_via_Neural_ADMG_Learning_images/iclr_2023_Causal_Reasoning_in_the_Presence_of_Latent_Confounders_via_Neural_ADMG_Learning.pdf-0015-08.png)


This requires samples from _p_ ( **x** _Y |_ do( **x** _T_ = **b** ) _, G_ ) = _p_ ( **x** _Y |_ **x** _T_ = **b** _, G_ do( **x** _T_ )), where _G_ do( **x** _T_ ) is the ‘mutilated’ graph obtained by removing incoming edges into **x** _T_ . We can achieve this by simulating the learnt SCM on _G_ do( **x** _T_ ) whilst keeping **x** _T_ = **b** fixed. Note that _qφ_ ( **u** _|_ **x** ) is not used to estimate the ATE; it suffices to sample **u** from the prior distribution, _p_ ( **u** ). In our setting, the inference MLP _qφ_ ( **u** _|_ **x** ) is only used as a means through which the likelihood of the data can be evaluated efficiently, and thus model parameters learned. This is in a similar spirit to VAEs (Kingma & Welling, 2013). 

## D RELATED WORK ON CAUSAL INFERENCE UNDER LATENT CONFOUNDING 

Attempting to perform causal inference in the presence of latent confounding can lead to biased estimates (Pearl, 2012). Whilst the observed data distribution may still be identifiable, estimating causal effects are not (Spirtes et al., 2000). A recent string of work has made progress in the case where the effects of multiple interventions are being estimated (Tran & Blei, 2017; Ranganath & Perotte, 2018; Wang & Blei, 2019; D’Amour, 2019). An alternative approach is to assume identifiability of the joint distribution over both latent and observed variables given just the observations. Louizos et al. (2017) point out that there are many cases in which this is possible (Khemakhem et al., 2020; Kingma & Welling, 2013). Nevertheless, all these methods assume the underlying causal graph is known. More recently, Mohammad-Taheri et al. (2021) argue that a DAG latent variable model trained on data can be used for down-stream causal inference tasks even if its parameters are nonidentifiable, as long as the query can be identified from the observed variables according to the do-calculus. 

## E ADDITIONAL DISCUSSIONS ON STRUCTURAL IDENTIFIABILITY OF LATENT VARIABLES 

In Section 3.2, we argued that the identifiability of ADMG does not imply the structural identifiability of the magnified SCM. In this section, we will present more discussions on certain identifiability of latent structures (of magnified SCMs). In general, these examples demonstrate that for linear non-Gaussian ANMs the structure of latent variables can be refined beyond the assumption of latent confounders acting between pairs of non-adjacent observed variables, whilst the same techniques cannot achieve the same for non-linear ANMs. 

15 

Published as a conference paper at ICLR 2023 

## E.1 DETERMINING CAUSAL STRUCTURE AMONGST LATENT VARIABLES 

Recently, Cai et al. (2019) demonstrated that it is possible to discover the structure amongst latent variables using their so-called Triad constraints. Their method is limited to the linear non-Gaussian ANM case. In this section, we demonstrate that an analogous constraint isn’t available for non-linear ANMs. 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2023_Causal_Reasoning_in_the_Presence_of_Latent_Confounders_via_Neural_ADMG_Learning_images/iclr_2023_Causal_Reasoning_in_the_Presence_of_Latent_Confounders_via_Neural_ADMG_Learning.pdf-0016-03.png)


**----- Start of picture text -----**<br>
f 12 f 21<br>u 1 u 2 u 1 u 2<br>f 1 i f 2 j f 2 k f 1 i f 2 j f 2 k<br>xi xj xk xi xj xk<br>(a) (b)<br>**----- End of picture text -----**<br>


**Figure 4:** The two possible latent variable structures considered by Cai et al. (2019). 

Consider the causal graphs shown in Figure 4a and Figure 4b. 

**Lemma 6** (Linear non-Gaussian identifiability) **.** _In the linear non-Gaussian ANM case, Equation_ 31 _is satisfied only for the causal graph shown in Figure 4b. In the non-linear ANM case, Equation_ 31 _is not satisfied for either Figure 4a or Figure 4b._ 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2023_Causal_Reasoning_in_the_Presence_of_Latent_Confounders_via_Neural_ADMG_Learning_images/iclr_2023_Causal_Reasoning_in_the_Presence_of_Latent_Confounders_via_Neural_ADMG_Learning.pdf-0016-07.png)


_Sketch of Proof._ For the causal graph in Figure 4a, Equation 31 is equivalent to 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2023_Causal_Reasoning_in_the_Presence_of_Latent_Confounders_via_Neural_ADMG_Learning_images/iclr_2023_Causal_Reasoning_in_the_Presence_of_Latent_Confounders_via_Neural_ADMG_Learning.pdf-0016-09.png)


On the right, we have some function of the latent variables _u_ 1 and _u_ 2, _f_ 2 _k_ ( _f_ 12( _u_ 1) _, u_ 2). On the left, we have some function of the same noise terms, _f_ 1 _i_ ( _u_ 1) _− g_ ( _f_ 2 _j_ ( _f_ 12( _u_ 1) _, ϵj_ ). To remove _u_ 1 from both sides, we require _g_ to be non-zero and so a term including _u_ 2 is still present. To remove _u_ 2, we again require _g_ to be non-zero and so a term including _u_ 1 is still present. Thus, Equation 31 is not 

For the causal graph in Figure 4b, Equation 31 is equivalent to 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2023_Causal_Reasoning_in_the_Presence_of_Latent_Confounders_via_Neural_ADMG_Learning_images/iclr_2023_Causal_Reasoning_in_the_Presence_of_Latent_Confounders_via_Neural_ADMG_Learning.pdf-0016-12.png)


In the linear case, we can construct a linear _g_ to remove _u_ 2 fom the left side so that Equation 31 holds (i.e. _g_ = _[f]_[1] _f[i]_ 2 _[f] j_[21][).][In the non-linear case,] _[ u]_[1][ and] _[ u]_[2][ are coupled in the leftmost term and cannot] be removed by a term involving _u_ 2 and _ϵj_ . Hence, Equation 31 is violated. 

## E.2 DETERMINING THE NUMBER OF LATENT CONFOUNDERS 

Here, we consider whether the number of latent confounders can be determined in the non-linear ANM case. Lemma 7 shows that in the linear non-Gaussian case, the number of latent confounders acting between a triplet of confounded observed variables can be determined (by verifying certain constraints on marginal distributions on observed variables), whilst the same approach cannot be used in the non-linear ANM case. 

Consider the two causal graphs shown in Figure 5a and Figure 5b. 

**Lemma 7** (Linear non-Gaussian identifiability) **.** _In the linear non-Gaussian ANM case, Equation_ 34 _is satisfied only for the causal graph shown in Figure 4b. In the non-linear ANM case, Equation_ 34 _is not satisfied for either Figure 4a or Figure 4b._ 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2023_Causal_Reasoning_in_the_Presence_of_Latent_Confounders_via_Neural_ADMG_Learning_images/iclr_2023_Causal_Reasoning_in_the_Presence_of_Latent_Confounders_via_Neural_ADMG_Learning.pdf-0016-18.png)


_Sketch of Proof._ For Figure 5b, Equation 34 is equivalent to 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2023_Causal_Reasoning_in_the_Presence_of_Latent_Confounders_via_Neural_ADMG_Learning_images/iclr_2023_Causal_Reasoning_in_the_Presence_of_Latent_Confounders_via_Neural_ADMG_Learning.pdf-0016-20.png)


16 

Published as a conference paper at ICLR 2023 

In the linear non-Gaussian case, it is straightforward to set _g_ = _f[f] j[i]_[to remove the common noise term] _u_ from the left term and make the two sides independent. In the non-linear case, when _fi_ = _fj_ the common noise term _u_ cannot be removed from the left as _g_ must be non-linear, and thus produces a term that involves both _u_ and _ϵj_ . For Figure 5a, Equation 34 is equivalent to 

_∃g fi_ ( _u_ 1 _, u_ 2) + _ϵi − g_ ( _fj_ ( _u_ 1 _, u_ 3) + _ϵj_ ) _fk_ ( _u_ 2 _, u_ 3) + _ϵk._ (36) _u_ 2 cannot be removed from the left, and so Equation 34 does not hold in either the linear nonGaussian or non-linear case. 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2023_Causal_Reasoning_in_the_Presence_of_Latent_Confounders_via_Neural_ADMG_Learning_images/iclr_2023_Causal_Reasoning_in_the_Presence_of_Latent_Confounders_via_Neural_ADMG_Learning.pdf-0017-03.png)


**----- Start of picture text -----**<br>
u 1 u 2 u 3 u<br>f 1 i f 2 i f 1 j f 3 j f 2 k f 3 k fi fj fk<br>xi xj xk xi xj xk<br>(a) (b)<br>**----- End of picture text -----**<br>


**Figure 5:** Two possible latent structures that confound each pair of variables _xi_ , _xj_ and _xk_ . 

## F OPTIMISATION DETAILS 

## F.1 OPTIMISATION DETAILS FOR N-ADMG 

As discussed in Section 5, we gradually increase the prior hyperparameters _ρ_ and _α_ throughout training. This is done using the augmented Lagrangian procedure for optimisation (). The optimisation process interleaves two steps: 1) optimise the objective for fixed values of _ρ_ and _α_ for a certain number of steps; and ii) update the values of _ρ_ and _α_ . Steps i) and ii) and ran until convergence, or the maximum number of optimisation steps is reached. We describe these two steps in more detail below. 

**Step i).** The objective is optimised for some fixed vales of _ρ_ and _α_ using Adam (). We use a learning rate of 10 _[−]_[3] for the model parameters and 5 _×_ 10 _[−]_[3] for the variational parameters. We optimise the objective for a maxmimum of 5000 steps or until convergence (we stop early if the loss does not improve for 1500 optimisation steps, moving to step ii)). During training, we reduce the learning rate by a factor of 10 if the training loss does not improve for 1000 steps a maximum of two times. If we reach the condition a third time, we assume optimisation has converged and move to step ii). We apply annealing to the KL-divergence between the approximate posterior and prior over the latent variables. The annealing contant is fixed for each step i), and increased linearly over the first optimisation loops. 

**Step ii).** We initialise _ρ_ = 1 and _α_ = 0. At the beginning of step i) we measure the DAG / bow-free penalty _P_ 1 = E _qφ_ ( _G_ ) [ _h_ ( _G_ )]. At the beginning of step ii), we measure this penalty again, _P_ 2 = E _qφ_ ( _G_ ) [ _h_ ( _G_ )]. If _P_ 2 _< P_ 1, we leave _ρ_ unchanged and update _α ← α_ + _ρP_ 2. Otherwise, if _P_ 2 _≥_ 0 _._ 65 _P_ 1, we leave _α_ unchanged and update _ρ ←_ 10 _ρ_ . We repeat the steps i) to ii) a maximum of 30 times or until convergence (measured as _α_ or _ρ_ reaching some max value which we set to 10[3] for both). 

F.2 ADDITIONAL HYPERPARAMETERS 

**Prior Hyperparameters.** We use the sparsity inducing prior hyperparameters _λs,_ 1 = _λs,_ 2 = 5. 

**ELBO approximation.** We construct an approximation to the ELBO in Equation 15 using a single sample from the approximate posteriors. For evaluating the gradients of the ELBO we use the Gumbel softmax method with a hard forward pass and soft backward pass with temperature of 0.25. 

**Neural network architectures.** The functions _ξ_ 1, _ξ_ 2 and _ℓ_ used in the likelihood and the inference network used to parameterise _qφ_ ( **u** _|_ **x** ) are all two hidden layer MLPs with 80 hidden units per hidden layer. 

17 

Published as a conference paper at ICLR 2023 

**Non-Gaussian noise model.** For the non-Gaussian noise model 

**ATE estimation.** For ATE estimation we compute expectations by drawing 1000 graphs from the learnt posterior, and for each graph we draw two samples of **x** _Y_ for a total of 2000 samples which we used to form a Monte Carlo estimate. 

## G DATASET DETAILS 

## G.1 SYNTHETIC FORK-COLLIDER DATASET 

We constructed a 2000 sample synthetic dataset with the causal structure shown in Figure 1 by sampling from the following SEM: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2023_Causal_Reasoning_in_the_Presence_of_Latent_Confounders_via_Neural_ADMG_Learning_images/iclr_2023_Causal_Reasoning_in_the_Presence_of_Latent_Confounders_via_Neural_ADMG_Learning.pdf-0018-06.png)


Variables _u_ 1 and _u_ 2 are latent confounders acting on variable pairs _x_ 2 and _x_ 3, and _x_ 3 and _x_ 4, respectively. 

## G.2 LATENT CONFOUNDED ER DATASET 

We generate synthetic datasets from ADMG extension of _Erd˝os-Rényi_ (ER) graph model (Lachapelle et al., 2019; Zheng et al., 2020). An **ER** ( _d, e, m_ ) dataset is generated according the following procedures: 

1. Generate a _d×d_ directed adjacency matrix _GD_ of a ADMG from _Erd˝os-Rényi_ (ER) graph model, whose expected directed edges equal to _e_ ; 

2. Simulate _d × d_ bidirected adjacency matrix _GB_ via random Bernoulli sampling, whose expected number equals to _m_ ; 

3. Simulate exogenous noises _ϵi_ from a zero mean Gaussian distribution with standard deviation of 0.1; 

4. Simulate latent variables **u** from a zero mean Gaussian distribution with standard deviation of 0.1; 

5. Simulate each observed variable as _xi_ = _fi_ ( **x** pa( _i_ ; _GD_ )) + _gi_ ( **u** pa( _i_ ; _GB_ )) + _ϵi_ , where _fi, gi_ are randomly sampled nonlinear functions of the form: _y_ = **w** _[T] e[−]_ **[x]**[2] . 

6. Remove **u** from the sampled dataset. 

## G.3 IHDP DATASET DETAILS 

This dataset contains measurements of both infants (birth weight, head circumference, etc.) and their mother (smoked cigarettes, drank alcohol, took drugs, etc) during real-life data collected in a randomised experiment. The main task is to estimate the effect of home visits by specialists on future cognitive test scores of infants. The outcomes of treatments are simulated artificially as in Hill (2011); hence the outcomes of both treatments (home visits or not) on each subject are known. Note that for each subject, our models are only exposed to only one of the treatments; the outcomes of the other potential/counterfactual outcomes are hidden from the mode, and are only used for the purpose of ATE evaluation. To make the task more challenging, additional confoundings are manually introduced by removing a subset (non-white mothers) of the treated children population. In this way we can construct the IHDP dataset of 747 individuals with 6 continuous covariates and 19 binary covariates. We use 10 replicates of different simulations based on setting B (log-linear response surfaces) 

18 

Published as a conference paper at ICLR 2023 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2023_Causal_Reasoning_in_the_Presence_of_Latent_Confounders_via_Neural_ADMG_Learning_images/iclr_2023_Causal_Reasoning_in_the_Presence_of_Latent_Confounders_via_Neural_ADMG_Learning.pdf-0019-01.png)


**Figure 6:** Run time results for N-BF-ADMG and CAM-UV on a 12 variable synthetic ER dataset. The figure shows mean results _±_ standard deviation across five randomly generated datasets. 

of Hill (2011), which can downloaded from `https://github.com/AMLab-Amsterdam/CEVAE` . We use a 70%/30% train-test split ratio. Before training our models, all continuous covariates are normalised. 

## H RUN TIME COMPARISON 

Here, we compare the run time of N-BF-ADMG and CAM-UV on a synthetically generated **ER** (12, 50, 10) dataset. The results are shown in Figure 6. N-BF-ADMG is trained using 30k epochs to ensure convergence, whose run time can be further reduced via early stopping etc. The results in Figure 6 highlight that methods based on continuous optimisation (N-BF-ADMG) offer a significant improvement in run time relative to methods based on conditional independency tests (CAM-UV) for large datasets. In combination with the results in the main paper, this shows the empirical validity of deep learning approach for identifying ADMGs that scales to larger dataset, without sacrificing accuracy. 

19 

