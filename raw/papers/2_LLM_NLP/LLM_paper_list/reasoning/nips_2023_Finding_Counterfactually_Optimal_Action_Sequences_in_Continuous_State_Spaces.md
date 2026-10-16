# Finding Counterfactually Optimal Action Sequences in Continuous State Spaces 

Stratis Tsirtsis and Manuel Gomez-Rodriguez 

Max Planck Institute for Software Systems _{_ stsirtsis, manuelgr _}_ @mpi-sws.org 

## **Abstract** 

Whenever a clinician reflects on the efficacy of a sequence of treatment decisions for a patient, they may try to identify critical time steps where, had they made different decisions, the patient’s health would have improved. While recent methods at the intersection of causal inference and reinforcement learning promise to aid human experts, as the clinician above, to _retrospectively_ analyze sequential decision making processes, they have focused on environments with finitely many discrete states. However, in many practical applications, the state of the environment is inherently continuous in nature. In this paper, we aim to fill this gap. We start by formally characterizing a sequence of discrete actions and continuous states using finite horizon Markov decision processes and a broad class of bijective structural causal models. Building upon this characterization, we formalize the problem of finding counterfactually optimal action sequences and show that, in general, we cannot expect to solve it in polynomial time. Then, we develop a search method based on the A _[∗]_ algorithm that, under a natural form of Lipschitz continuity of the environment’s dynamics, is guaranteed to return the optimal solution to the problem. Experiments on real clinical data show that our method is very efficient in practice, and it has the potential to offer interesting insights for sequential decision making tasks. 

## **1 Introduction** 

Had the chess player moved the king one round later, would they have avoided losing the game? Had the physician administered antibiotics one day earlier, would the patient have recovered? The process of mentally simulating alternative worlds where events of the past play out differently than they did in reality is known as counterfactual reasoning [1]. Thoughts of this type are a common by-product of human decisions and they are tightly connected to the way we attribute causality and responsibility to events and others’ actions [2]. The last decade has seen a rapid development of reinforcement learning agents, presenting (close to) human-level performance in a variety of sequential decision making tasks, such as gaming [3, 4], autonomous driving [5] and clinical decision support [6, 7]. In conjunction with the substantial progress made in the field of causal inference [8, 9], this has led to a growing interest in machine learning methods that employ elements of counterfactual reasoning to improve or to retrospectively analyze decisions in sequential settings [10–15]. 

In the context of reinforcement learning, sequential decision making is typically modeled using Markov Decision Processes (MDPs) [16]. Here, we consider MDPs with a finite horizon where each episode ( _i.e._ , each sequence of decisions) consists of a finite number of time steps. As an example, consider a clinician treating a patient in an intensive care unit (ICU). At each time step, the clinician observes the current state of the environment ( _e.g._ , the patient’s vital signs) and they choose among a set of potential actions ( _e.g._ , standardized dosages of a drug). Consequently, the chosen action causes the environment to transition (stochastically) into a new state, and the clinician earns a reward ( _e.g._ , satisfaction inversely proportional to the patient’s severity). The process repeats until the horizon is met and the goal of the clinician is to maximize the total reward. 

1 

In this work, our goal is to aid the retrospective analysis of individual episodes as the example above. For each episode, we aim to find an action sequence that differs slightly from the one taken in reality but, under the circumstances of that particular episode, would have led to a higher counterfactual reward. In our example above, assume that the patient’s condition does not improve after a certain period of time. A counterfactually optimal action sequence could highlight to the clinician a small set of time steps in the treatment process where, had they administered different drug dosages, the patient’s severity would have been lower. In turn, a manual inspection of those time steps could provide insights to the clinician about potential ways to improve their treatment policy. 

To infer how a particular episode would have evolved under a different action sequence than the one taken in reality, one needs to represent the stochastic state transitions of the environment using a structural causal model (SCM) [8, 17]. This has been a key aspect of a line of work at the intersection of counterfactual reasoning and reinforcement learning, which has focused on methods to either design better policies using offline data [10, 12] or to retrospectively analyze individual episodes [11, 13]. Therein, the work most closely related to ours is by Tsirtsis et al. [13], which introduces a method to compute counterfactually optimal action sequences in MDPs with discrete states and actions using a Gumbel-Max SCM to model the environment dynamics [11]. However, in many practical applications, such as in critical care, the state of the environment is inherently continuous in nature [18]. In our work, we aim to fill this gap by designing a method to compute counterfactually optimal action sequences in MDPs with continuous states and discrete actions. Refer to Appendix A for a discussion of further related work and to Pearl [8] for an overview of the broad field of causality. 

**Our contributions.** We start by formally characterizing sequential decision making processes with continuous states and discrete actions using finite horizon MDPs and a general class of _bijective_ SCMs [19]. Notably, this class of SCMs includes multiple models introduced in the causal discovery literature [20–25]. Building on this characterization, we make the following contributions: 

- (i) We formalize the problem of finding a counterfactually optimal action sequence for a particular episode in environments with continuous states under the constraint that it differs from the observed action sequence in at most _k_ actions. 

- (ii) We show that the above problem is NP-hard using a novel reduction from the classic partition problem [26]. This is in contrast with the computational complexity of the problem in environments with discrete states, which allows for polynomial time algorithms [13]. 

- (iii) We develop a search method based on the _A[∗]_ algorithm that, under a natural form of Lipschitz continuity of the environment’s dynamics, is guaranteed to return the optimal solution to the problem upon termination. 

Finally, we evaluate the performance and the qualitative insights of our method by performing a series of experiments using real patient data from critical care.[1] 

## **2 A Causal Model of Sequential Decision Making Processes** 

At each time step _t ∈_ [ _T −_ 1] := _{_ 0 _,_ 1 _, . . . , T −_ 1 _}_ , where _T_ is a time horizon, the decision making process is characterized by a _D_ -dimensional vector state _**s** t ∈S_ = R _[D]_ , an action _at ∈A_ , where _A_ is a finite set of _N_ actions, and a reward _R_ ( _**s** t, at_ ) _∈_ R associated with each pair of states and actions. Moreover, given an episode of the decision making process, _τ_ = _{_ ( _**s** t, at_ ) _}[T] t_ =0 _[ −]_[1][,][the][process’s][outcome] _[o]_[(] _[τ]_[) =][ �] _t[R]_[(] _**[s]**[t][, a][t]_[)][is][given] by the sum of the rewards. In the remainder, we will denote the elements of a vector _**s** t_ as _st,_ 1 _, . . . , st,D_ .[2] Further, we characterize the dynamics of the decision making process using the framework of structural causal models (SCMs). In general, an SCM is consisted of four parts: (i) a set of endogenous variables (ii) a set of exogenous (noise) variables (iii) a set of structural equations assigning values to the endogenous variables, and (iv) a set of prior distributions characterizing the exogenous variables [8]. In our setting, the endogenous variables of the SCM _C_ are the random variables representing the states _**S**_ 0 _, . . . ,_ _**S** T −_ 1 and the 

> 1Our code is accessible at https://github.com/Networks-Learning/counterfactual-continuous-mdp. 

> 2Table 1 in Appendix B summarizes the notation used throughout the paper. 

2 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2023_Finding_Counterfactually_Optimal_Action_Sequences_in_Continuous_State_Spaces_images/nips_2023_Finding_Counterfactually_Optimal_Action_Sequences_in_Continuous_State_Spaces.pdf-0003-00.png)


Figure 1: Directed acyclic graph _G_ of the SCM _C_ , modeling a sequential decision making process (figure adapted from [13]). Green boxes represent endogenous random variables and red boxes represent exogenous noise variables. All exogenous random variables are root nodes of the graph, and each one is independently sampled from its respective prior distribution. Each endogenous variable is an effect of its ancestors in the graph _G_ , and takes its value based on Eqs. 1 and 2. An intervention _do_ ( _At_ = _a_ ) breaks the dependence of the variable _At_ from its ancestors (highlighted by dotted lines) and sets its value to _a_ . After observing an event _**S** t_ +1 = _**s** t_ +1 _,_ _**S** t_ = _**s** t, At_ = _at_ , a counterfactual prediction corresponds to an intervention _do_ ( _At_ = _a_ ) in a modified SCM where _**U** t_ takes values _**u** t_ from a posterior distribution with support such that _**s** t_ +1 = _gS_ ( _**s** t, at,_ _**u** t_ ). 

actions _A_ 0 _, . . . , AT −_ 1. The action _At_ at time step _t_ is chosen based on the observed state _**S** t_ and is given by a structural (policy) equation 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2023_Finding_Counterfactually_Optimal_Action_Sequences_in_Continuous_State_Spaces_images/nips_2023_Finding_Counterfactually_Optimal_Action_Sequences_in_Continuous_State_Spaces.pdf-0003-03.png)


where _**Z** t ∈Z_ is a vector-valued noise variable, to allow some level of stochasticity in the choice of the action, and its prior distribution _P[C]_ ( _**Z** t_ ) is characterized by a density function _f_ _**Z**[C] t_[.][Similarly,][the][state] _**[S]**[t]_[+1][in][the] next time step is given by a structural (transition) equation 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2023_Finding_Counterfactually_Optimal_Action_Sequences_in_Continuous_State_Spaces_images/nips_2023_Finding_Counterfactually_Optimal_Action_Sequences_in_Continuous_State_Spaces.pdf-0003-05.png)


where _**U** t ∈U_ is a vector-valued noise variable with its prior distribution _P[C]_ ( _**U** t_ ) having a density function _f_ _**U**[C] t_[,][and][we][refer][to][the][function] _[g][S]_[as][the] _[transition][mechanism]_[.][Note][that,][in][Eq.][2,][the][noise][variables] _{_ _**U** t}[T] t_ =0 _[ −]_[1][are][mutually][independent][and,][keeping][the][sequence][of][actions][fixed,][they][are][the][only][source][of] stochasticity in the dynamics of the environment. In other words, a sampled sequence of noise values _{_ _**u** t}[T] t_ =0 _[ −]_[1] and a fixed sequence of actions _{at}[T] t_ =0 _[ −]_[1][result][into][a][single][(deterministic)][sequence][of][states] _[{]_ _**[s]**[t][}] t[T]_ =0 _[ −]_[1][.][This] implicitly assumes that the state transitions are stationary and there are no unobserved confounders. Figure 1 depicts the causal graph _G_ corresponding to the SCM _C_ defined above. 

The above representation of sequential decision making using an SCM _C_ is a more general reformulation of a Markov decision process, where a (stochastic) policy _π_ ( _a |_ _**s**_ ) is entailed by Eq. 1, and the transition distribution ( _i.e._ , the conditional distribution of _**S** t_ +1 _|_ _**S** t, At_ ) is entailed by Eq. 2. Specifically, the conditional density function of _**S** t_ +1 _|_ _**S** t, At_ is given by 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2023_Finding_Counterfactually_Optimal_Action_Sequences_in_Continuous_State_Spaces_images/nips_2023_Finding_Counterfactually_Optimal_Action_Sequences_in_Continuous_State_Spaces.pdf-0003-08.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2023_Finding_Counterfactually_Optimal_Action_Sequences_in_Continuous_State_Spaces_images/nips_2023_Finding_Counterfactually_Optimal_Action_Sequences_in_Continuous_State_Spaces.pdf-0003-09.png)


> where _do_ ( _At_ = _at_ ) denotes a (hard) intervention on the variable _At_ , whose value is set to _at_ .[3] Here, the first equality holds because _**S** t_ +1 and _At_ are d-separated by _**S** t_ in the sub-graph obtained from _G_ after removing all outgoing edges of _At_[4] and the second equality follows from Eq. 2. 

> Moreover, as argued elsewhere [11, 13], by using an SCM to represent sequential decision making, instead of a classic MDP, we can answer counterfactual questions. More specifically, assume that, at time step _t_ , we 

> 3In general, the _do_ operator also allows for soft interventions ( _i.e._ , setting a probability distribution for _At_ ). 

> 4This follows directly from the rules of _do_ -calculus. For further details, refer to Chapter 3 of Pearl [8]. 

3 

observed the state _**S** t_ = _**s** t_ , we took action _At_ = _at_ and the next state was _**S** t_ +1 = _**s** t_ +1. Retrospectively, we would like to know the probability that the state _**S** t_ +1 would have been _**s**[′]_ if, at time step _t_ , we had been in a state _**s**_ , and we had taken an action _a_ , (generally) different from _**s** t, at_ . Using the SCM _C_ , we can characterize this by a counterfactual transition density function 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2023_Finding_Counterfactually_Optimal_Action_Sequences_in_Continuous_State_Spaces_images/nips_2023_Finding_Counterfactually_Optimal_Action_Sequences_in_Continuous_State_Spaces.pdf-0004-01.png)


where _f_ _**U**[C |] t_ _**[ S]**[t]_[+1][=] _**[s]**[t]_[+1] _[,]_ _**[S]**[t]_[=] _**[s]**[t][,A][t]_[=] _[a][t]_ is the posterior distribution of the noise variable _**U** t_ with support such that _**s** t_ +1 = _gS_ ( _**s** t, at,_ _**u**_ ). 

In what follows, we will assume that the transition mechanism _gS_ is continuous with respect to its last argument and the SCM _C_ satisfies the following form of Lipschitz-continuity: 

**Definition 1.** _An SCM C is Lipschitz-continuous iff the transition mechanism gS and the reward R are Lipschitz-continuous with respect to their first argument,_ i.e. _, for each a ∈A,_ _**u** ∈U, there exists a Lipschitz constant Ka,_ _**u** ∈_ R+ _such that, for any_ _**s** ,_ _**s**[′] ∈S, ∥gS_ ( _**s** , a,_ _**u**_ ) _− gS_ ( _**s**[′] , a,_ _**u**_ ) _∥≤ Ka,_ _**u** ∥_ _**s** −_ _**s**[′] ∥, and, for each a ∈A, there exists a Lipschitz constant Ca ∈_ R+ _such that, for any_ _**s** ,_ _**s**[′] ∈S, |R_ ( _**s** , a_ ) _−R_ ( _**s**[′] , a_ ) _| ≤ Ca ∥_ _**s** −_ _**s**[′] ∥. In both cases, ∥·∥ denotes the Euclidean distance._ 

Note that, although they are not phrased in causal terms, similar Lipschitz continuity assumptions for the environment dynamics are common in prior work analyzing the theoretical guarantees of reinforcement learning algorithms [27–35]. Moreover, for practical applications ( _e.g._ , in healthcare), this is a relatively mild assumption to make. Consider two patients whose vitals _**s**_ and _**s**[′]_ are similar at a certain point in time, they receive the same treatment _a_ , and every unobserved factor _**u**_ that may affect their health is also the same. Intuitively, Definition 1 implies that their vitals will also evolve similarly in the immediate future, _i.e._ , the values _gS_ ( _**s** , a,_ _**u**_ ) and _gS_ ( _**s**[′] , a,_ _**u**_ ) will not differ dramatically. In this context, it is worth mentioning that, when the transition mechanism _gS_ is modeled by a neural network, it is possible to control its Lipschitz constant during training, and penalizing high values can be seen as a regularization method [36, 37]. 

Further, we will focus on bijective SCMs [19], a fairly broad class of SCMs, which subsumes multiple models studied in the causal discovery literature, such as additive noise models [20], post-nonlinear causal models [21], location-scale noise models [22] and more complex models with neural network components [23–25]. 

**Definition 2.** _An SCM C is bijective iff the transition mechanism gS is bijective with respect to its last argument,_ i.e. _, there is a well-defined inverse function gS[−]_[1] : _S × A × S →U such that, for every combination of_ _**s** t_ +1 _,_ _**s** t, at,_ _**u** t with_ _**s** t_ +1 = _gS_ ( _**s** t, at,_ _**u** t_ ) _, it holds that_ _**u** t_ = _gS[−]_[1][(] _**[s]**[t][, a][t][,]_ _**[ s]**[t]_[+1][)] _[.]_ 

Importantly, bijective SCMs allow for a more concise characterization of the counterfactual transition density given in Eq. 4. More specifically, after observing an event _**S** t_ +1 = _**s** t_ +1 _,_ _**S** t_ = _**s** t, At_ = _at_ , the value _**u** t_ of the noise variable _**U** t_ can only be such that _**u** t_ = _gS[−]_[1][(] _**[s]**[t][, a][t][,]_ _**[ s]**[t]_[+1][),] _[i.e.]_[,][the][posterior][distribution][of] _**[U]**[t]_[is][a] point mass and its density is given by 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2023_Finding_Counterfactually_Optimal_Action_Sequences_in_Continuous_State_Spaces_images/nips_2023_Finding_Counterfactually_Optimal_Action_Sequences_in_Continuous_State_Spaces.pdf-0004-09.png)


where 1[ _·_ ] denotes the indicator function. Then, for a given episode _τ_ of the decision making process, we have that the (non-stationary) counterfactual transition density is given by 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2023_Finding_Counterfactually_Optimal_Action_Sequences_in_Continuous_State_Spaces_images/nips_2023_Finding_Counterfactually_Optimal_Action_Sequences_in_Continuous_State_Spaces.pdf-0004-11.png)


Since this density is also a point mass, the resulting counterfactual dynamics are purely deterministic. That means, under a bijective SCM , the answer to the question “ _What would have been the state at time t_ + 1 _,_ 

4 

_had we been at state_ _**s** and taken action a at time t, given that, in reality, we were at_ _**s** t, we took at and the environment transitioned to_ _**s** t_ +1 _?_ ” is just given by _**s**[′]_ = _gS_ � _**s** , a, gS[−]_[1][(] _**[s]**[t][, a][t][,]_ _**[ s]**[t]_[+1][)] �. 

**On the counterfactual identifiability of bijective SCMs.** Very recently, Nasr-Esfahany and Kiciman [38] have shown that bijective SCMs are in general not counterfactually identifiable when the exogenous variable _**U** t_ is multi-dimensional. In other words, even with access to an infinite amount of triplets ( _**s** t, at,_ _**s** t_ +1) sampled from the true SCM _C_ , it is always possible to find an SCM _M ̸_ = _C_ with transition mechanism _hS_ and distributions _P[M]_ ( _**U** t_ ) that entails the same transition distributions as _C_ ( _i.e._ , it fits the observational data perfectly), but leads to different counterfactual predictions. Although our subsequent algorithmic results do not require the SCM _C_ to be counterfactually identifiable, the subclass of bijective SCMs we will use in our experiments in Section 5 is counterfactually identifiable. The defining attribute of this subclass, which we refer to as _element-wise bijective SCMs_ , is that the transition mechanism _gS_ can be decoupled into _D_ independent mechanisms _gS,i_ such that _St_ +1 _,i_ = _gS,i_ ( _**S** t, At, Ut,i_ ) for _i ∈{_ 1 _, . . . , D}_ . This implies _St_ +1 _,i ⊥⊥ Ut,j | Ut,i,_ _**S** t, At_ for _j_ = _i_ , however, _Ut,i_ , _Ut,j_ do not need to be independent. Informally, we have the following identifiability result (refer to Appendix C for a formal version of the theorem along with its proof, which follows a similar reasoning to proofs found in related work [12, 19]): 

**Theorem 3** (Informal) **.** _Let C and M be two element-wise bijective SCMs such that their entailed transition distributions for_ _**S** t_ +1 _given any value of_ _**S** t, At are always identical. Then, all their counterfactual predictions based on an observed transition_ ( _**s** t, at,_ _**s** t_ +1) _will also be identical._ 

**On the assumption of no unobserved confounding.** The assumption that there are no hidden confounders is a frequent assumption made by work at the intersection of counterfactual reasoning and reinforcement learning [10–13] and, more broadly, in the causal inference literature [39–43]. That said, there is growing interest in developing off-policy methods for partially observable MPDs (POMDPs) that are robust to certain types of confounding [44–46], and in learning dynamic treatment regimes in sequential settings with non-Markovian structure [47, 48]. Moreover, there is a line of work focusing on the identification of counterfactual quantities in non-sequential confounded environments [49–51]. In that context, we consider the computation of (approximately) optimal counterfactual action sequences under confounding as a very interesting direction for future work. 

## **3 Problem statement** 

Let _τ_ be an observed episode of a decision making process whose dynamics are characterized by a Lipschitzcontinuous bijective SCM. To characterize the counterfactual outcome that any alternative action sequence would have achieved under the circumstances of the particular episode, we build upon the formulation of Section 2, and we define a non-stationary counterfactual MDP _M_[+] = ( _S_[+] _, A, Fτ,t_[+] _[, R]_[+] _[, T]_[)][with][deterministic] transitions. Here, _S_[+] = _S ×_ [ _T −_ 1] is an enhanced state space such that each _**s**_[+] _∈S_[+] is a pair ( _**s** , l_ ) indicating that the counterfactual episode would have been at state _**s** ∈S_ with _l_ action changes already performed. Accordingly, _R_[+] is a reward function which takes the form _R_[+] (( _**s** , l_ ) _, a_ ) = _R_ ( _**s** , a_ ) for all ( _**s** , l_ ) _∈S_[+] , _a ∈A_ , _i.e._ , it does not change depending on the number of action changes already performed. Finally, the time-dependent transition function _Fτ,t_[+][:] _[ S]_[+] _[ × A →S]_[+][is][defined][as] 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2023_Finding_Counterfactually_Optimal_Action_Sequences_in_Continuous_State_Spaces_images/nips_2023_Finding_Counterfactually_Optimal_Action_Sequences_in_Continuous_State_Spaces.pdf-0005-06.png)


Intuitively, here we set the transition function according to the point mass of the counterfactual transition density given in Eq. 6, and we use the second coordinate to keep track of the changes that have been performed in comparison to the observed action sequence up to the time step _t_ . 

Now, given the initial state _**s**_ 0 of the episode _τ_ and any counterfactual action sequence _{a[′] t[}][T] t_ =0 _[ −]_[1][,][we] can compute the corresponding counterfactual episode _τ[′]_ = _{_ ( _**s**[′] t[, l][t]_[)] _[, a][′] t[}][T] t_ =0 _[ −]_[1][.][Its][sequence][of][states][is][given] recursively by 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2023_Finding_Counterfactually_Optimal_Action_Sequences_in_Continuous_State_Spaces_images/nips_2023_Finding_Counterfactually_Optimal_Action_Sequences_in_Continuous_State_Spaces.pdf-0005-09.png)


5 

and its counterfactual outcome is given by _o_[+] ( _τ[′]_ ) :=[�] _t[R]_[+][ ((] _**[s]** t[′][, l][t]_[)] _[ , a] t[′]_[) =][ �] _t[R]_[ (] _**[s]** t[′][, a] t[′]_[).] Then, similarly as in Tsirtsis et al. [13], our ultimate goal is to find the counterfactual action sequence _{a[′] t[}][T] t_ =0 _[ −]_[1][that,][starting][from][the][observed][initial][state] _**[s]**_[0][,][maximizes][the][counterfactual][outcome][subject][to] a constraint on the number of counterfactual actions that can differ from the observed ones, _i.e._ , 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2023_Finding_Counterfactually_Optimal_Action_Sequences_in_Continuous_State_Spaces_images/nips_2023_Finding_Counterfactually_Optimal_Action_Sequences_in_Continuous_State_Spaces.pdf-0006-01.png)


where _a_ 0 _, . . . , aT −_ 1 are the observed actions. Unfortunately, using a reduction from the classic partition problem [26], the following theorem shows that we cannot hope to find the optimal action sequence in polynomial time:[5] 

## **Theorem 4.** _The problem defined by Eq. 9. is NP-Hard._ 

The proof of the theorem relies on a reduction from the partition problem [26], which is known to be NP-complete, to our problem, defined in Eq. 9. At a high-level, we map any instance of the partition problem to an instance of our problem, taking special care to construct a reward function and an observed action sequence, such that the optimal counterfactual outcome _o_[+] ( _τ[∗]_ ) takes a specific value if and only if there exists a valid partition for the original instance. The hardness result of Theorem 4 motivates our subsequent focus on the design of a method that _always_ finds the optimal solution to our problem at the expense of a potentially higher runtime for some problem instances. 

## **4 Finding the optimal counterfactual action sequence via A* search** 

To deal with the increased computational complexity of the problem, we develop an optimal search method based on the classic _A[∗]_ algorithm [52], which we have found to be very efficient in practice. Our starting point is the observation that, the problem of Eq. 9 presents an optimal substructure, _i.e._ , its optimal solution can be constructed by combining optimal solutions to smaller sub-problems. For an observed episode _τ_ , let _Vτ_ ( _**s** , l, t_ ) be the maximum counterfactual reward that could have been achieved in a counterfactual episode where, at time _t_ , the process is at a (counterfactual) state _**s**_ , and there are so far _l_ actions that have been different in comparison with the observed action sequence. Formally, 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2023_Finding_Counterfactually_Optimal_Action_Sequences_in_Continuous_State_Spaces_images/nips_2023_Finding_Counterfactually_Optimal_Action_Sequences_in_Continuous_State_Spaces.pdf-0006-07.png)


Then, it is easy to see that the quantity _Vτ_ ( _**s** , l, t_ ) can be given by the recursive function 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2023_Finding_Counterfactually_Optimal_Action_Sequences_in_Continuous_State_Spaces_images/nips_2023_Finding_Counterfactually_Optimal_Action_Sequences_in_Continuous_State_Spaces.pdf-0006-09.png)


where ( _**s** a, la_ ) = _Fτ,t_[+][((] _**[s]**[, l]_[)] _[ , a]_[)][.][In the base case of] _[ l]_[=] _[ k]_[ (] _[i.e.]_[, all allowed action changes are already performed),] we have _Vτ_ ( _**s** , k, t_ ) = _R_ ( _**s** , at_ )+ _Vτ_ ( _**s** at, lat, t_ + 1) for all _**s** ∈S_ and _t < T −_ 1, and _Vτ_ ( _**s** , k, T −_ 1) = _R_ ( _**s** , aT −_ 1) for _t_ = _T −_ 1. Lastly, when _t_ = _T −_ 1 and _l < k_ , we have _Vτ_ ( _**s** , l, T −_ 1) = max _a∈A R_ ( _**s** , a_ ) for all _**s** ∈S_ . 

Given the optimal substructure of the problem, one may be tempted to employ a typical dynamic programming approach to compute the values _Vτ_ ( _**s** , l, t_ ) in a bottom-up fashion. However, the complexity of the problem lies in the fact that, the states _**s**_ are real-valued vectors whose exact values depend on the entire action sequence that led to them. Hence, to enumerate all the possible values that _**s**_ might take, one has to enumerate all possible action sequences in the search space, which is equivalent to solving our problem with a brute force search. In what follows, we present our proposed method to find optimal solutions using the _A[∗]_ algorithm, with the caveat that its runtime varies depending on the problem instance, and it can be equal to that of a brute force search in the worst case. 

> 5The supporting proofs of all Theorems, Lemmas and Propositions can be found in Appendix D. 

6 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2023_Finding_Counterfactually_Optimal_Action_Sequences_in_Continuous_State_Spaces_images/nips_2023_Finding_Counterfactually_Optimal_Action_Sequences_in_Continuous_State_Spaces.pdf-0007-00.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2023_Finding_Counterfactually_Optimal_Action_Sequences_in_Continuous_State_Spaces_images/nips_2023_Finding_Counterfactually_Optimal_Action_Sequences_in_Continuous_State_Spaces.pdf-0007-01.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2023_Finding_Counterfactually_Optimal_Action_Sequences_in_Continuous_State_Spaces_images/nips_2023_Finding_Counterfactually_Optimal_Action_Sequences_in_Continuous_State_Spaces.pdf-0007-02.png)


**----- Start of picture text -----**<br>
(a) Search graph (b) Heuristic function computation<br>**----- End of picture text -----**<br>


Figure 2: Main components of our search method based on the A* algorithm. Panel (a) shows the search graph for a problem instance with _|A|_ = 2. Here, each box represents a node _v_ = ( _**s** , l, t_ ) of the graph, and each edge represents a counterfactual transition. Next to each edge, we include the action _a ∈A_ causing the transition and the associated reward. Panel (b) shows the heuristic function computation, where the two axes represent a (continuous) state space _S_ = R[2] and the two levels on the z-axis correspond to differences in the (integer) values ( _l, t_ ) and ( _la, t_ + 1). Here, the blue squares correspond to the finite states in the anchor set _S†_ and ( _**s** a, la_ ) = _Fτ,t_[+][((] _**[s]**[, l]_[)] _[ , a]_[).] 

**Casting the problem as graph search.** We represent the solution space of our problem as a graph, where each node _v_ corresponds to a tuple ( _**s** , l, t_ ) with _**s** ∈S_ , _l ∈_ [ _k_ ] and _t ∈_ [ _T_ ]. Every node _v_ = ( _**s** , l, t_ ) with _l < k_ and _t < T −_ 1 has _|A|_ outgoing edges, each one associated with an action _a ∈A_ , carrying a reward _R_ ( _**s** , a_ ), and leading to a node _va_ = ( _**s** a, la, t_ + 1) such that ( _**s** a, la_ ) = _Fτ,t_[+][((] _**[s]**[, l]_[)] _[ , a]_[)][.][In][the][case][of] _[l]_[ =] _[ k]_[,][the][node] _[v]_ has exactly one edge corresponding to the observed action _at_ at time _t_ . Lastly, when _t_ = _T −_ 1, the outgoing edge(s) lead(s) to a common node _vT_ = ( _**s** ∅, k, T_ ) which we call the _goal node_ , and it has zero outgoing edges itself. Note that, the exact value of _**s** ∅_ is irrelevant, and we only include it for notational completeness. Let _**s**_ 0 be the initial state of the observed episode. Then, it is easy to notice that, starting from the root node _v_ 0 = ( _**s**_ 0 _,_ 0 _,_ 0), the first elements of each node _vi_ on a path _v_ 0 _, . . . , vi, . . . , vT_ form a sequence of counterfactual states, and the edges that connect those nodes are such that the corresponding counterfactual action sequence differs from the observed one in at most _k_ actions. That said, the counterfactual outcome _o_[+] ( _τ_ ) =[�] _[T] t_ =0 _[ −]_[1] _[R]_[(] _**[s]** t[′][, a][′] t_[)][is][expressed][as][the][sum][of][the][rewards][associated][with][each][edge][in][the][path,][and] the problem defined by Eq. 9 is equivalent to finding the path of maximum total reward that starts from _v_ 0 and ends in _vT_ . Figure 2a illustrates the search graph for a simple instance of our problem. Unfortunately, since the states _**s**_ are vectors of real values, even enumerating all the graph’s nodes requires time exponential in the number of actions _|A|_ , which makes classic algorithms that search over the entire graph non-practical. 

To address this challenge, we resort to the _A[∗]_ algorithm, which performs a more efficient search over the graph by preferentially exploring only parts of it where we have prior information that they are more likely to lead to paths of higher total reward. Concretely, the algorithm proceeds iteratively and maintains a queue of nodes to visit, initialized to contain only the root node _v_ 0. Then, at each step, it selects one node from the queue, and it retrieves all its children nodes in the graph which are subsequently added to the queue. It terminates when the node being visited is the goal node _vT_ . Refer to Algorithm 2 in Appendix E for a pseudocode implementation of the _A[∗]_ algorithm. 

The key element of the _A[∗]_ algorithm is the criterion based on which it selects which node from the queue to visit next. Let _vi_ = ( _**s** i, li, t_ ) be a candidate node in the queue and _rvi_ be the total reward of the path that the algorithm has followed so far to reach from _v_ 0 to _vi_ . Then, the _A[∗]_ algorithm visits next the node _vi_ that maximizes the sum _rvi_ + _V_[ˆ] _τ_ ( _**s** i, li, t_ ), where _V_[ˆ] _τ_ is a _heuristic function_ that aims to estimate the maximum reward that can be achieved via any path starting from _vi_ = ( _**s** i, li, t_ ) and ending in the goal node _vT_ , _i.e._ , it gives an estimate for the quantity _Vτ_ ( _**s** i, li, t_ ). Intuitively, the heuristic function can be thought of as an “eye into the future” of the graph search, that guides the algorithm towards nodes that are more likely to 

7 

**Algorithm 1:** It computes upper bounds _V_[ˆ] _τ_ ( _**s** , l, t_ ) for the values _Vτ_ ( _**s** , l, t_ ) 

**Input** : States _S_ , actions _A_ , observed action sequence _{at}[t] t_[=] =0 _[T][ −]_[1] , horizon _T_ , transition function _Fτ,t_[+][,][reward] function _R_ , constraint _k_ , anchor set _S†_ . **Initialize** : _V_[ˆ] ˆ _τ_ ( _**s** , l, T −_ 1) _←_ max _a∈A R_ ( _**s** , a_ ) _, s ∈S†, l_ = 0 _, . . . , k −_ 1 _. Vτ_ ( _**s** , k, T −_ 1) _← R_ ( _**s** , at_ ) _, s ∈S†._ **for** _t_ = _T −_ 2 _, . . . ,_ 0 **do for** _l_ = _k, . . . ,_ 0 **do** _available_ ~~_a_~~ _ctions ← at_ **if** _l_ = _k_ **else** _A_ **for** _**s** ∈S†_ **do** _bounds ←∅_ **for** _a ∈ available actions_ **do** _**s** a, la ← Fτ,t_[+][((] _**[s]**[, l]_[)] _[ , a]_[)][;] `/* Get the min bound for` _Vτ_ ( _**s** a, la, t_ + 1) _Va ←_ min _**s** †∈S† {V_[ˆ] _τ_ ( _**s** †, la, t_ + 1) + _Lt_ +1 _∥_ _**s** † −_ _**s** a∥}_ ; `based on` _S†_ `*/` _bounds ← bounds ∪{R_ ( _**s** , a_ ) + _Va}_ **end** ˆ _Vτ_ ( _**s** , l, t_ ) _←_ max( _bounds_ ) ; `/* Get the max bound over the actions */` **end end end return** _V_[ˆ] _τ_ ( _**s** , l, t_ ) for all _**s** ∈S†, l ∈_ [ _k_ ] _, t ∈_ [ _T −_ 1] 

lead to the optimal solution and the algorithm’s performance depends on the quality of the approximation of _Vτ_ ( _**s** i, li, t_ ) by _V_[ˆ] _τ_ ( _**s** i, li, t_ ). Next, we will look for a heuristic function that satisfies _consistency_[6] to guarantee that the _A[∗]_ algorithm as described above returns the optimal solution upon termination [52]. 

**Computing a consistent heuristic function.** We first propose an algorithm that computes the function’s values _V_[ˆ] _τ_ ( _**s** , l, t_ ) for a finite set of points such that _l ∈_ [ _k_ ], _t ∈_ [ _T −_ 1], _**s** ∈S† ⊂S_ , where _S†_ is a pre-defined _finite_ set of states—an _anchor set_ —whose construction we discuss later. Then, based on the Lipschitz-continuity of the SCM _C_ , we show that these computed values of _V_[ˆ] _τ_ are valid upper bounds of the corresponding values _Vτ_ ( _**s** , l, t_ ) and we expand the definition of the heuristic function _V_[ˆ] _τ_ over all _**s** ∈S_ by expressing it in terms of those upper bounds. Finally, we prove that the function resulting from the aforementioned procedure is consistent. 

To compute the upper bounds _V_[ˆ] _τ_ , we exploit the observation that the values _Vτ_ ( _**s** , l, t_ ) satisfy a form of Lipschitz-continuity, as stated in the following Lemma. 

**Lemma 5.** _Let_ _**u** t_ = _gS[−]_[1][(] _**[s]**[t][, a][t][,]_ _**[ s]**[t]_[+1][)] _[,][ K]_ _**[u]** t_[=][ max] _[a][∈A][ K][a,]_ _**[u]** t[,][ C]_[=][ max] _[a][∈A][ C][a][and the sequence][ L]_[0] _[, . . . , L][T][ −]_[1] _[∈]_ R+ _be such that LT −_ 1 = _C and Lt_ = _C_ + _Lt_ +1 _K_ _**u** t for t ∈_ [ _T −_ 2] _. Then, it holds that |Vτ_ ( _**s** , l, t_ ) _−Vτ_ ( _**s**[′] , l, t_ ) _| ≤ Lt ∥_ _**s** −_ _**s**[′] ∥, for all t ∈_ [ _T −_ 1] _, l ∈_ [ _k_ ] _and_ _**s** ,_ _**s**[′] ∈S._ 

Based on this observation, our algorithm proceeds in a bottom-up fashion and computes valid upper bounds of the values _Vτ_ ( _**s** , l, t_ ) for all _l ∈_ [ _k_ ], _t ∈_ [ _T −_ 1] and _**s**_ in the anchor set _S†_ . To get the intuition, assume that, for a given _t_ , the values _V_[ˆ] _τ_ ( _**s** , l, t_ + 1) are already computed for all _**s** ∈S†_ , _l ∈_ [ _k_ ], and they are indeed valid upper bounds of the corresponding _Vτ_ ( _**s** , l, t_ + 1). Then, let ( _**s** a, la_ ) = _Fτ,t_[+][((] _**[s]**[, l]_[)] _[ , a]_[)][for] some _**s** ∈S†_ and _l ∈_ [ _k_ ]. Since _**s** a_ itself may not belong to the finite anchor set _S†_ , the algorithm uses the values _V_ ˆ _τ_ ( _**s** , l, tV_[ˆ] _τ_ )(in _**s** †, l_ way _a, t_ + 1)that ofit isallalsoanchorsguaranteed _**s** † ∈S†_ toin becombinationa (maximallywithtight)their distanceupper boundto _**s** a_ ,ofand _Vτ_ (it _**s** , l, t_ sets).theFigurevalue2bof illustrates the above operation. Algorithm 1 summarizes the overall procedure, which is guaranteed to return upper bounds, as shown by the following proposition: 

> 6A heuristic function _V_ ˆ _τ_ is consistent iff, for nodes _v_ = ( _**s** , l, t_ ), _va_ = ( _**s** a, la, t_ + 1) connected with an edge associated with action _a_ , it satisfies _V_[ˆ] _τ_ ( _**s** , l, t_ ) _≥ R_ ( _**s** , a_ ) + _V_[ˆ] _τ_ ( _**s** a, la, t_ + 1) [53]. 

8 

**Proposition 6.** _For all_ _**s** ∈S†, l ∈_ [ _k_ ] _, t ∈_ [ _T −_ 1] _, it holds that V_[ˆ] _τ_ ( _**s** , l, t_ ) _≥ Vτ_ ( _**s** , l, t_ ) _, where V_[ˆ] _τ_ ( _**s** , l, t_ ) _are the values of the heuristic function computed by Algorithm 1._ 

Next, we use the values _V_[ˆ] _τ_ ( _**s** , l, t_ ) computed by Algorithm 1 to expand the definition of _V_[ˆ] _τ_ over the entire domain as follows. For some _**s** ∈S_ , _a ∈A_ , let ( _**s** a, la_ ) = _Fτ,t_[+][((] _**[s]**[, l]_[)] _[ , a]_[),][then,][we][have][that] 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2023_Finding_Counterfactually_Optimal_Action_Sequences_in_Continuous_State_Spaces_images/nips_2023_Finding_Counterfactually_Optimal_Action_Sequences_in_Continuous_State_Spaces.pdf-0009-02.png)


where _A[′]_ = _{at}_ for _l_ = _k_ and _A[′]_ = _A_ for _l < k_ . Finally, the following theorem shows that the resulting heuristic function _V_[ˆ] _τ_ is consistent: 

**Theorem 7.** _For any nodes v_ = ( _**s** , l, t_ ) _, va_ = ( _**s** a, la, t_ + 1) _with t < T −_ 1 _connected with an edge associated with action a, it holds that V_[ˆ] _τ_ ( _**s** , l, t_ ) _≥ R_ ( _**s** , a_ ) + _V_[ˆ] _τ_ ( _**s** a, la, t_ + 1) _. Moreover, for any node v_ = ( _**s** , l, T −_ 1) _and edge connecting it to the goal node vT_ = ( _**s** ∅, k, T_ ) _, it holds that V_[ˆ] _τ_ ( _**s** , l, T −_ 1) _≥ R_ ( _**s** , a_ ) + _V_[ˆ] _τ_ ( _**s** ∅, k, T_ ) _._ 

**Kick-starting the heuristic function computation with Monte Carlo anchor sets.** For any _**s** ̸∈S†_ , whenever we compute _V_[ˆ] _τ_ ( _**s** , l, t_ ) using Eq. 11, the resulting value is set based on the value _V_[ˆ] _τ_ ( _**s** †, la, t_ +1) of some anchor _**s** †_ , increased by a _penalty_ term _Lt_ +1 _∥_ _**s** † −_ _**s** a∥_ . Intuitively, this allows us to think of the heuristic function _V_[ˆ] _τ_ as an upper bound of the function _Vτ_ whose looseness depends on the magnitude of the penalty terms encountered during the execution of Algorithm 1 and each subsequent evaluation of Eq. 11. To speed up the _A[∗]_ algorithm, note that, ideally, one would want all penalty terms to be zero, _i.e._ , an anchor set that includes all the states _**s**_ of the nodes _v_ = ( _**s** , l, t_ ) that are going to appear in the search graph. However, as discussed in the beginning of Sec. 4, an enumeration of those states requires a runtime exponential in the number of actions. 

To address this issue, we introduce a Monte Carlo simulation technique that adds to the anchor set the observed states _{_ _**s**_ 0 _, . . . ,_ _**s** T −_ 1 _}_ and all unique states _{_ _**s**[′]_ 0 _[, . . . ,]_ _**[ s]**[′] T −_ 1 _[}]_[resulting][by] _[M]_[randomly][sampled] counterfactual action sequences _a[′]_ 0 _[, . . . , a][′] T −_ 1[.][Specifically,][for][each][action][sequence,][we][first][sample][a][number] _k[′]_ of actions to be changed and what those actions are going to be, both uniformly at random from _{_ 1 _, . . . , k}_ and _A[k][′]_ , respectively. Then, we sample from _{_ 0 _, . . . , T −_ 1 _}_ the _k[′]_ time steps where the changes take place, with each time step _t_ having a probability _Lt/_[�] _t[′][ L][t][′]_[to][be][selected.][This][biases][the][sampling][towards][earlier] time steps, where the penalty terms are larger due to the higher Lipschitz constants. As we will see in the next section, this approach works well in practice, and it allows us to control the runtime of the _A[∗]_ algorithm by appropriately adjusting the number of samples _M_ . We experiment with additional anchor set selection strategies in Appendix F. 

## **5 Experiments using clinical sepsis management data** 

**Experimental setup.** To evaluate our method, we use real patient data from MIMIC-III [54], a freely accessible critical care dataset commonly used in reinforcement learning for healthcare [6, 55–57]. We follow the preprocessing steps of Komorowski et al. [6] to identify a cohort of 20 _,_ 926 patients treated for sepsis [58]. Each patient record contains vital signs and administered treatment information in time steps of 4-hour intervals. As an additional preprocessing step, we discard patient records whose associated time horizon _T_ is shorter than 10, resulting in a final dataset of 15 _,_ 992 patients with horizons between 10 and 20. 

To form our state space _S_ = R _[D]_ , we use _D_ = 13 features. Four of these features are demographic or contextual and thus we always set their counterfactual values to the observed ones. The remaining _D_[˜] = 9 features are time-varying and include the SOFA score [59]—a standardized score of organ failure rate—along with eight vital signs that are required for its calculation. Since SOFA scores positively correlate with patient mortality [60], we assume that each _**s** ∈S_ gives a reward _R_ ( _**s**_ ) equal to the negation of its SOFA value. Here, it is easy to see that this reward function is just a projection of _**s**_ , therefore, it is Lipschitz continuous with constant _Ca_ = 1 for all _a ∈A_ . Following related work [6, 55, 57], we consider an action space _A_ that consists 

9 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2023_Finding_Counterfactually_Optimal_Action_Sequences_in_Continuous_State_Spaces_images/nips_2023_Finding_Counterfactually_Optimal_Action_Sequences_in_Continuous_State_Spaces.pdf-0010-00.png)


**----- Start of picture text -----**<br>
3 600<br>2 . 5 2 . 2<br>60<br>20<br>400<br>40 2 . 1 2<br>2 . 0<br>200<br>20 2 . 0 10<br>1<br>1 . 5 0 1 . 9 0<br>0.8 0.9 1.0 1.1 1.2 1.3 500 1000 1500 2000 2500 3000 0 1 2 3 4 5 6<br>Lipschitz constant, Lh # of Monte Carlo samples, M # of action changes, k<br>(a) Efficiency vs. Lh (b) Efficiency vs. M (c) Efficiency vs. k<br>(s) (s) (s)<br>runtime runtime runtime<br>EBF EBF EBF<br>average average average<br>A∗ A∗ A∗<br>**----- End of picture text -----**<br>


Figure 3: Computational efficiency of our method under different configurations, as measured by the effective branching factor (pink-left axis) and the runtime of the _A[∗]_ algorithm (green-right axis). In Panel (a), we set _M_ = 2000 and _k_ = 3. In Panel (b), we set _Lh_ = 1 _._ 0 and _k_ = 3. In Panel (c), we set _Lh_ = 1 _._ 0 and _M_ = 2000. In all panels, we set _Lϕ_ = 0 _._ 1 and error bars indicate 95% confidence intervals over 200 executions of the _A[∗]_ algorithm for 200 patients with horizon _T_ = 12. 

of 25 actions, which correspond to 5 _×_ 5 levels of administered vasopressors and intravenous fluids. Refer to Appendix G for additional details on the features and actions. 

To model the transition dynamics of the time-varying features, we consider an SCM _C_ whose transition _D_ mechanism takes a location-scale form _gS_ ( _**S** t, At,_ _**U** t_ ) = _h_ ( _**S** t, At_ ) + _ϕ_ ( _**S** t, At_ ) _⊙_ _**U** t_ , where _h, ϕ_ : _S × A →_ R[˜] , and _⊙_ denotes the element-wise multiplication [22, 24]. Notably, this model is element-wise bijective and hence it is counterfactually identifiable, as shown in Section 2. Moreover, we use neural networks to model the location and scale functions _h_ and _ϕ_ and enforce their Lipschitz constants to be _Lh_ and _Lϕ_ , respectively. This results in a Lipschitz continuous SCM _C_ with _Ka,_ _**u**_ = _Lh_ + _Lϕ_ max _i |ui|_ . Further, we assume that the noise variable _**U** t_ follows a multivariate Gaussian distribution with zero mean and allow its covariance matrix to be a (trainable) parameter. 

We jointly train the weights of the networks _h_ and _ϕ_ and the covariance matrix of the noise prior on the observed patient transitions using stochastic gradient descent with the negative log-likelihood of each transition as a loss. In our experiments, if not specified otherwise, we use an SCM with Lipschitz constants _Lh_ = 1 _._ 0, _Lϕ_ = 0 _._ 1 that achieves a log-likelihood only 6% lower to that of the best model trained without any Lipschitz constraint. Refer to Appendix G for additional details on the network architectures, the training procedure and the way we enforce Lipschitz continuity.[7] 

**Results.** We start by evaluating the computational efficiency of our method against (i) the Lipschitz constant of the location network _Lh_ , (ii) the number of Monte Carlo samples _M_ used to generate the anchor set _S†_ , and (iii) the number of actions _k_ that can differ from the observed ones. We measure efficiency using running time and the effective branching factor (EBF) [52]. The EBF is defined as a real number _b ≥_ 1 such that the number of nodes expanded by _A[∗]_ is equal to 1 + _b_ + _b_[2] + _· · ·_ + _b[T]_ , where _T_ is the horizon, and values close to 1 indicate that the heuristic function is the most efficient in guiding the search. Figure 3 summarizes the results, which show that our method maintains overall a fairly low running time that decreases with the number of Monte Carlo samples _M_ used for the generation of the anchor set and increases with the Lipschitz constant _Lh_ and the number of action changes _k_ . That may not come as a surprise since, as _Lh_ increases, the heuristic function becomes more loose, and as _k_ increases, the size of the search space increases exponentially. To put things in perspective, for a problem instance with _Lh_ = 1 _._ 0, _k_ = 3 and horizon _T_ = 12, the _A[∗]_ search led by our heuristic function is effectively equivalent to an exhaustive search over a full tree with 2 _._ 1[12] _≈_ 7 _,_ 355 leaves while the corresponding search space of our problem consists of more than 3 million action sequences—more than 3 million paths to reach from the root node to the goal node. 

Next, we investigate to what extent the counterfactual action sequences generated by our method would have led the patients in our dataset to better outcomes. For each patient, we measure their counterfactual 

> 7All experiments were performed using an internal cluster of machines equipped with 16 Intel(R) Xeon(R) 3.20GHz CPU cores, 512GBs of memory and 2 NVIDIA A40 48GB GPUs. 

10 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2023_Finding_Counterfactually_Optimal_Action_Sequences_in_Continuous_State_Spaces_images/nips_2023_Finding_Counterfactually_Optimal_Action_Sequences_in_Continuous_State_Spaces.pdf-0011-00.png)


**----- Start of picture text -----**<br>
10% 2000 100<br>50 !<br>5% 1000 0<br>20% 30%<br>0% 0<br>0 1 2 3 4 5 6 0% 10% 20% 30% 40%<br># of action changes, k Counterfactual improvement<br>(a) Average c/f improvement vs. k (b) Distribution of c/f improvement (c) Observed vs. c/f episode<br>patients<br>improvement of<br>c/f Number<br>Average<br>**----- End of picture text -----**<br>


Figure 4: Retrospective analysis of patients’ episodes. Panel (a) shows the average counterfactual improvement as a function of _k_ for a set of 200 patients with horizon _T_ = 12, where error bars indicate 95% confidence intervals. Panel (b) shows the distribution of counterfactual improvement across all patients for _k_ = 3, where the dashed vertical line indicates the median. Panel (c) shows the observed (solid) and counterfactual (dashed) SOFA score across time for a patient who presents a 19 _._ 9% counterfactual improvement when _k_ = 3. Upward (downward) arrows indicate action changes that suggest a higher (lower) dosage of vasopressors (V) and fluids (F). In all panels, we set _M_ = 2000 

improvement—the relative decrease in cumulative SOFA score between the counterfactual and the observed episode. Figures 4a and 4b summarize the results, which show that: (i) the average counterfactual improvement shows a diminishing increase as _k_ increases; (ii) the median counterfactual improvement is only 5%, indicating that, the treatment choices made by the clinicians for most of the patients were close to optimal, even with the benefit of hindsight; and (iii) there are 176 patients for whom our method suggests that a different sequence of actions would have led to an outcome that is at least 15% better. That said, we view patients at the tail of the distribution as “interesting cases” that should be deferred to domain experts for closer inspection, and we present one such example in Fig. 4c. In this example, our method suggests that, had the patient received an early higher dosage of intravenous fluids while some of the later administered fluids where replaced by vasopressors, their SOFA score would have been lower across time. Although we present this case as purely anecdotal, the counterfactual episode is plausible, since there are indications of decreased mortality when intravenous fluids are administered at the early stages of a septic shock [61]. 

## **6 Conclusions** 

In this paper, we have introduced the problem of finding counterfactually optimal action sequences in sequential decision making processes with continuous state dynamics. We showed that the problem is NP-hard and, to tackle it, we introduced a search method based on the _A[∗]_ algorithm that is guaranteed to find the optimal solution, with the caveat that its runtime can vary depending on the problem instance. Lastly, using real clinical data, we have found that our method is very efficient in practice, and it has the potential to offer interesting insights to domain experts by highlighting episodes and time-steps of interest for further inspection. 

Our work opens up many interesting avenues for future work. For example, it would be interesting to develop algorithms with approximation guarantees that run in polynomial time, at the expense of not achieving strict counterfactual optimality. Moreover, since the practicality of methods like ours relies on the assumption that the SCM describing the environment is accurate, it would be interesting to develop methods to learn SCMs that align with human domain knowledge. Finally, it would be interesting to validate our method using real datasets from other applications and carry out user studies in which the counterfactual action sequences found by our method are systematically evaluated by the human experts ( _e.g._ , clinicians) who took the observed actions. 

**Acknowledgements.** Tsirtsis and Gomez-Rodriguez acknowledge support from the European Research Council (ERC) under the European Union’s Horizon 2020 research and innovation programme (grant agreement No. 945719). 

11 

## **References** 

- [1] Ruth MJ Byrne. Counterfactual thought. _Annual review of psychology_ , 67:135–157, 2016. 

- [2] David A Lagnado, Tobias Gerstenberg, and Ro’i Zultan. Causal responsibility and counterfactuals. _Cognitive science_ , 37(6):1036–1073, 2013. 

- [3] David Silver, Aja Huang, Chris J Maddison, Arthur Guez, Laurent Sifre, George Van Den Driessche, Julian Schrittwieser, Ioannis Antonoglou, Veda Panneershelvam, Marc Lanctot, et al. Mastering the game of go with deep neural networks and tree search. _nature_ , 529(7587):484–489, 2016. 

- [4] �Lukasz Kaiser, Mohammad Babaeizadeh, Piotr Mi�los, B�la˙zej Osi´nski, Roy H Campbell, Konrad Czechowski, Dumitru Erhan, Chelsea Finn, Piotr Kozakowski, Sergey Levine, Afroz Mohiuddin, Ryan Sepassi, George Tucker, and Henryk Michalewski. Model based reinforcement learning for atari. In _International Conference on Learning Representations_ , 2020. 

- [5] B Ravi Kiran, Ibrahim Sobh, Victor Talpaert, Patrick Mannion, Ahmad A Al Sallab, Senthil Yogamani, and Patrick P´erez. Deep reinforcement learning for autonomous driving: A survey. _IEEE Transactions on Intelligent Transportation Systems_ , 23(6):4909–4926, 2021. 

- [6] Matthieu Komorowski, Leo A Celi, Omar Badawi, Anthony C Gordon, and A Aldo Faisal. The artificial intelligence clinician learns optimal treatment strategies for sepsis in intensive care. _Nature medicine_ , 24(11):1716–1720, 2018. 

- [7] Chao Yu, Jiming Liu, Shamim Nemati, and Guosheng Yin. Reinforcement learning in healthcare: A survey. _ACM Computing Surveys (CSUR)_ , 55(1):1–36, 2021. 

- [8] Judea Pearl. _Causality_ . Cambridge university press, 2009. 

- [9] Jonas Peters, Dominik Janzing, and Bernhard Sch¨olkopf. _Elements of causal inference: foundations and learning algorithms_ . The MIT Press, 2017. 

- [10] Lars Buesing, Theophane Weber, Yori Zwols, Nicolas Heess, Sebastien Racaniere, Arthur Guez, and Jean-Baptiste Lespiau. Woulda, coulda, shoulda: Counterfactually-guided policy search. In _International Conference on Learning Representations_ , 2019. 

- [11] Michael Oberst and David Sontag. Counterfactual off-policy evaluation with gumbel-max structural causal models. In _International Conference on Machine Learning_ , pages 4881–4890. PMLR, 2019. 

- [12] Chaochao Lu, Biwei Huang, Ke Wang, Jos´e Miguel Hern´andez-Lobato, Kun Zhang, and Bernhard Sch¨olkopf. Sample-efficient reinforcement learning via counterfactual-based data augmentation. _arXiv preprint arXiv:2012.09092_ , 2020. 

- [13] Stratis Tsirtsis, Abir De, and Manuel Gomez-Rodriguez. Counterfactual explanations in sequential decision making under uncertainty. _Advances in Neural Information Processing Systems_ , 34:30127–30139, 2021. 

- [14] Jonathan Richens, Rory Beard, and Daniel H Thompson. Counterfactual harm. In _Advances in Neural Information Processing Systems_ , volume 35, pages 36350–36365, 2022. 

- [15] Kimia Noorbakhsh and Manuel Gomez-Rodriguez. Counterfactual temporal point processes. _Advances in Neural Information Processing Systems_ , 35:24810–24823, 2022. 

- [16] Richard S Sutton and Andrew G Barto. _Reinforcement learning: An introduction_ . MIT press, 2018. 

- [17] Elias Bareinboim, Juan D Correa, Duligur Ibeling, and Thomas Icard. On pearl’s hierarchy and the foundations of causal inference. In _Probabilistic and causal inference: the works of judea pearl_ , pages 507–556. 2022. 

12 

- [18] Malcolm Elliott and Alysia Coventry. Critical care: the eight vital signs of patient monitoring. _British Journal of Nursing_ , 21(10):621–625, 2012. 

- [19] Arash Nasr-Esfahany, Mohammad Alizadeh, and Devavrat Shah. Counterfactual identifiability of bijective causal models. In _Proceedings of the 40th International Conference on Machine Learning_ , ICML’23. JMLR.org, 2023. 

- [20] Patrik Hoyer, Dominik Janzing, Joris M Mooij, Jonas Peters, and Bernhard Sch¨olkopf. Nonlinear causal discovery with additive noise models. _Advances in neural information processing systems_ , 21, 2008. 

- [21] Kun Zhang and Aapo Hyv¨arinen. On the identifiability of the post-nonlinear causal model. In _Proceedings of the Twenty-Fifth Conference on Uncertainty in Artificial Intelligence_ , UAI ’09, page 647–655, Arlington, Virginia, USA, 2009. AUAI Press. 

- [22] Alexander Immer, Christoph Schultheiss, Julia E Vogt, Bernhard Sch¨olkopf, Peter B¨uhlmann, and Alexander Marx. On the identifiability and estimation of causal location-scale noise models. In _International Conference on Machine Learning_ , pages 14316–14332. PMLR, 2023. 

- [23] Nick Pawlowski, Daniel Coelho de Castro, and Ben Glocker. Deep structural causal models for tractable counterfactual inference. _Advances in Neural Information Processing Systems_ , 33:857–869, 2020. 

- [24] Ilyes Khemakhem, Ricardo Monti, Robert Leech, and Aapo Hyvarinen. Causal autoregressive flows. In _International conference on artificial intelligence and statistics_ , pages 3520–3528. PMLR, 2021. 

- [25] Pedro Sanchez and Sotirios A. Tsaftaris. Diffusion causal models for counterfactual estimation. In _First Conference on Causal Learning and Reasoning_ , 2022. 

- [26] Richard M Karp. Reducibility among combinatorial problems. In _Complexity of computer computations_ , pages 85–103. Springer, 1972. 

- [27] Dimitri Bertsekas. Convergence of discretization procedures in dynamic programming. _IEEE Transactions on Automatic Control_ , 20(3):415–419, 1975. 

- [28] Karl Hinderer. Lipschitz continuity of value functions in markovian decision processes. _Mathematical Methods of Operations Research_ , 62:3–22, 2005. 

- [29] Emmanuel Rachelson and Michail G Lagoudakis. On the locality of action domination in sequential decision making. 2010. 

- [30] Jason Pazis and Ronald Parr. Pac optimal exploration in continuous space markov decision processes. In _Proceedings of the AAAI Conference on Artificial Intelligence_ , volume 27, pages 774–781, 2013. 

- [31] Matteo Pirotta, Marcello Restelli, and Luca Bascetta. Policy gradient in lipschitz markov decision processes. _Machine Learning_ , 100:255–283, 2015. 

- [32] Felix Berkenkamp, Matteo Turchetta, Angela Schoellig, and Andreas Krause. Safe model-based reinforcement learning with stability guarantees. _Advances in neural information processing systems_ , 30, 2017. 

- [33] Kavosh Asadi, Dipendra Misra, and Michael Littman. Lipschitz continuity in model-based reinforcement learning. In _International Conference on Machine Learning_ , pages 264–273. PMLR, 2018. 

- [34] Ahmed Touati, Adrien Ali Taiga, and Marc G Bellemare. Zooming for efficient model-free reinforcement learning in metric spaces. _arXiv preprint arXiv:2003.04069_ , 2020. 

- [35] Omer Gottesman, Kavosh Asadi, Cameron Allen, Sam Lobel, George Konidaris, and Michael Littman. Coarse-grained smoothness for rl in metric spaces. _arXiv preprint arXiv:2110.12276_ , 2021. 

13 

- [36] Cem Anil, James Lucas, and Roger Grosse. Sorting out lipschitz function approximation. In _International Conference on Machine Learning_ , pages 291–301. PMLR, 2019. 

- [37] Henry Gouk, Eibe Frank, Bernhard Pfahringer, and Michael J Cree. Regularisation of neural networks by enforcing lipschitz continuity. _Machine Learning_ , 110:393–416, 2021. 

- [38] Arash Nasr-Esfahany and Emre Kiciman. Counterfactual (non-)identifiability of learned structural causal models. _arXiv preprint arXiv:2301.09031_ , 2023. 

- [39] Scott Sussex, Caroline Uhler, and Andreas Krause. Near-optimal multi-perturbation experimental design for causal structure learning. _Advances in Neural Information Processing Systems_ , 34:777–788, 2021. 

- [40] Ioana Bica, Daniel Jarrett, and Mihaela van der Schaar. Invariant causal imitation learning for generalizable policies. _Advances in Neural Information Processing Systems_ , 34:3952–3964, 2021. 

- [41] Chris Cundy, Aditya Grover, and Stefano Ermon. Bcd nets: Scalable variational approaches for bayesian causal discovery. _Advances in Neural Information Processing Systems_ , 34:7095–7110, 2021. 

- [42] Virginia Aglietti, Neil Dhir, Javier Gonz´alez, and Theodoros Damoulas. Dynamic causal bayesian optimization. _Advances in Neural Information Processing Systems_ , 34:10549–10560, 2021. 

- [43] Xinshi Chen, Haoran Sun, Caleb Ellington, Eric Xing, and Le Song. Multi-task learning of order-consistent causal graphs. _Advances in Neural Information Processing Systems_ , 34:11083–11095, 2021. 

- [44] Guy Tennenholtz, Uri Shalit, and Shie Mannor. Off-policy evaluation in partially observable environments. In _Proceedings of the AAAI Conference on Artificial Intelligence_ , volume 34, pages 10276–10283, 2020. 

- [45] Hongseok Namkoong, Ramtin Keramati, Steve Yadlowsky, and Emma Brunskill. Off-policy policy evaluation for sequential decisions under unobserved confounding. _Advances in Neural Information Processing Systems_ , 33:18819–18831, 2020. 

- [46] Lingxiao Wang, Zhuoran Yang, and Zhaoran Wang. Provably efficient causal reinforcement learning with confounded observational data. _Advances in Neural Information Processing Systems_ , 34:21164–21175, 2021. 

- [47] Junzhe Zhang and Elias Bareinboim. Near-optimal reinforcement learning in dynamic treatment regimes. _Advances in Neural Information Processing Systems_ , 32, 2019. 

- [48] Junzhe Zhang and Elias Bareinboim. Designing optimal dynamic treatment regimes: A causal reinforcement learning approach. In _International Conference on Machine Learning_ , pages 11012–11022. PMLR, 2020. 

- [49] Ilya Shpitser and Eli Sherman. Identification of personalized effects associated with causal pathways. In _Uncertainty in artificial intelligence: proceedings of the... conference. Conference on Uncertainty in Artificial Intelligence_ , volume 2018. NIH Public Access, 2018. 

- [50] Juan Correa, Sanghack Lee, and Elias Bareinboim. Nested counterfactual identification from arbitrary surrogate experiments. _Advances in Neural Information Processing Systems_ , 34:6856–6867, 2021. 

- [51] Junzhe Zhang, Jin Tian, and Elias Bareinboim. Partial counterfactual identification from observational and experimental data. In _International Conference on Machine Learning_ , pages 26548–26558. PMLR, 2022. 

- [52] Stuart Russel, Peter Norvig, et al. _Artificial intelligence: a modern approach_ , volume 256. Pearson Education Limited London, 2013. 

- [53] Judea Pearl. _Heuristics: intelligent search strategies for computer problem solving_ . Addison-Wesley Longman Publishing Co., Inc., 1984. 

14 

- [54] Alistair EW Johnson, Tom J Pollard, Lu Shen, Li-wei H Lehman, Mengling Feng, Mohammad Ghassemi, Benjamin Moody, Peter Szolovits, Leo Anthony Celi, and Roger G Mark. Mimic-iii, a freely accessible critical care database. _Scientific data_ , 3(1):1–9, 2016. 

- [55] Aniruddh Raghu, Matthieu Komorowski, Imran Ahmed, Leo Celi, Peter Szolovits, and Marzyeh Ghassemi. Deep reinforcement learning for sepsis treatment. _arXiv preprint arXiv:1711.09602_ , 2017. 

- [56] Siqi Liu, Kay Choong See, Kee Yuan Ngiam, Leo Anthony Celi, Xingzhi Sun, and Mengling Feng. Reinforcement learning for clinical decision support in critical care: comprehensive review. _Journal of medical Internet research_ , 22(7):e18477, 2020. 

- [57] Taylor W Killian, Haoran Zhang, Jayakumar Subramanian, Mehdi Fatemi, and Marzyeh Ghassemi. An empirical study of representation learning for reinforcement learning in healthcare. _arXiv preprint arXiv:2011.11235_ , 2020. 

- [58] Mervyn Singer, Clifford S Deutschman, Christopher Warren Seymour, Manu Shankar-Hari, Djillali Annane, Michael Bauer, Rinaldo Bellomo, Gordon R Bernard, Jean-Daniel Chiche, Craig M Coopersmith, et al. The third international consensus definitions for sepsis and septic shock (sepsis-3). _Jama_ , 315(8):801– 810, 2016. 

- [59] Simon Lambden, Pierre Francois Laterre, Mitchell M Levy, and Bruno Francois. The sofa score—development, utility and challenges of accurate assessment in clinical trials. _Critical Care_ , 23(1):1–9, 2019. 

- [60] Flavio Lopes Ferreira, Daliana Peres Bota, Annette Bross, Christian M´elot, and Jean-Louis Vincent. Serial evaluation of the sofa score to predict outcome in critically ill patients. _Jama_ , 286(14):1754–1758, 2001. 

- [61] Jason Waechter, Anand Kumar, Stephen E Lapinsky, John Marshall, Peter Dodek, Yaseen Arabi, Joseph E Parrillo, R Phillip Dellinger, Allan Garland, Cooperative Antimicrobial Therapy of Septic Shock Database Research Group, et al. Interaction between fluids and vasoactive agents on mortality in septic shock: a multicenter, observational study. _Critical care medicine_ , 42(10):2158–2168, 2014. 

- [62] Lihong Li and Michael L Littman. Lazy approximation for solving continuous finite-horizon mdps. In _AAAI_ , volume 5, pages 1175–1180, 2005. 

- [63] Richard Dearden, Nicholas Meuleau, Richard Washington, and Zhengzhu Feng. Dynamic programming for structured continuous markov decision problems. In _Twentieth Conference on Uncertainty in Artificial Intelligene (UAI-04)_ , 2004. 

- [64] Prashan Madumal, Tim Miller, Liz Sonenberg, and Frank Vetere. Explainable reinforcement learning through a causal lens. In _Proceedings of the AAAI conference on artificial intelligence_ , volume 34, pages 2493–2500, 2020. 

- [65] Ioana Bica, Daniel Jarrett, Alihan H¨uy¨uk, and Mihaela van der Schaar. Learning” what-if” explanations for sequential decision-making. _arXiv preprint arXiv:2007.13531_ , 2020. 

- [66] Amir-Hossein Karimi, Julius Von K¨ugelgen, Bernhard Sch¨olkopf, and Isabel Valera. Algorithmic recourse under imperfect causal knowledge: a probabilistic approach. _Advances in neural information processing systems_ , 33:265–277, 2020. 

- [67] Stratis Tsirtsis and Manuel Gomez-Rodriguez. Decisions, counterfactual explanations and strategic behavior. _Advances in Neural Information Processing Systems_ , 33:16749–16760, 2020. 

15 

## **A Further related work** 

**Counterfactual reasoning and reinforcement learning.** As mentioned in Section 1, there is a closely related line of work [10–13] that focuses on the development of machine learning methods that employ elements of counterfactual reasoning to improve or to retrospectively analyze decisions in sequential settings. Buesing et al. [10] use SCMs to express the transition dynamics in Partially Observable MDPs (POMDPs), and they propose a method to efficiently compute a policy based on counterfactual realizations of logged episodes. Lu et al. [12] adopt a similar modeling framework, and they propose a counterfactual data augmentation approach to speed up standard Q-learning. Oberst and Sontag [11] introduce the Gumbel-Max SCM to express the dynamics of an arbitrary discrete POMPD, and they develop a method for counterfactual off-policy evaluation to identify episodes where a given alternative policy would have achieved a higher reward. However, none of these works aims to find an action sequence, close to the observed sequence of a particular episode, that is counterfactually optimal. 

**Planning in continuous-state MDPs.** Our work has additional connections to pieces of work that aim to approximate the optimal value function in an MDP with continuous states and a finite horizon [27, 62, 63]. Therein, the work most closely related to ours is the one by Bertsekas [27]. It shows that, under a Lipschitz continuity assumption on the transition dynamics, a value function computed via value iteration in an MDP with discretized states, converges to the optimal value function of the original (continous-state) MDP as the discretization becomes finer. Although some of the proof mechanics of this work are similar to ours, the contributions are orthogonal, as we do not employ any form of discretization, and we leverage the Lipschitz continuity assumption to compute optimal action sequences in continuous states using the _A[∗]_ algorithm. 

**Counterfactual reasoning and explainability.** Our work has ties to pieces of work that use forms of counterfactual reasoning as a tool towards learning explainable machine learning models. For example, Madumal et al. [64] propose to express the action selection process of a reinforcement learning agent as a causal graph, and they use it to generate explanations for the agent’s chosen actions. Bica et al. [65] introduce an inverse reinforcement learning approach to learn an interpretable reward function from expert demonstrated behavior that is expressed in terms of preferences over potential (counterfactual) outcomes. Moreover, our work is broadly related to the work on counterfactual explanations (in classification) that aims to find a minimally different set of features that would have led to a different outcome, in settings where single decisions are taken [66, 67]. 

16 

## **B Supporting notation table** 

Table 1: It summarizes the most important notation used in the main body of the paper. 

|**Symbol/Defnition**|**Description**|
|---|---|
|_T_|Time horizon|
|**_S_**_t ∈S, At ∈A_|State and action (random variables) at time _t_|
|**_s_**_t, at_|State and action (observed values) at time _t_|
|_R_:_S × A →_R|Reward function|
|_τ_ = _{_(**_s_**_t, at_)_}T −_1<br>_t_=0|Observed episode|
|_o_(_τ_)= �_T −_1<br>_t_=0 _R_(**_s_**_t, at_)|Outcome of episode _τ_|
|**_U_**_t ∈U_|Transition noise (random variable) at time _t_|
|**_u_**_t_|Transition noise (inferred value) at time _t_|
|_gS_ :_S × A × U →S_|Transition mechanism (def. in Eq. 2)|
|_Ka,_**_u_**|Lipschitz constant of _gS_ under _At_ =_a_ and **_U_**_t_ =**_u_**|
|_Ca_|Lipschitz constant of _R_ under _At_ =_a_|
|_S_+ =_S ×_ [_T −_1]|Enhanced state space|
|(**_s_**_, l_)_∈S_+|Enhanced state representing a c/f state **_s_** after _l_ action changes|
|_F_ +<br>_τ,t_ :_S_+ _× A →S_+|Time-dependent transition function (def. in Eq. 7)|
|_τ ′_ = _{_(**_s_**_′_<br>_t, a′_<br>_t_)_}T −_1<br>_t_=0|Counterfactual episode|
|_o_+(_τ ′_)= �_T −_1<br>_t_=0 _R_(**_s_**_′_<br>_t, a′_<br>_t_)|Counterfactual outcome of episode _τ ′_|
|_Vτ_(**_s_**_, l, t_)|Max. c/f reward achievable for _τ_ from enhanced state (**_s_**_, l_) at time _t_|
|(**_s_**_a, la_)=_F_ +<br>_τ,t_((**_s_**_, l_)_, a_)|Enhanced state resulting from (**_s_**_, l_) with action _a_|
|_v_ =(**_s_**_, l, t_)|Node in the search graph of the _A∗_algorithm|
|_v_0 =(**_s_**_,_0_,_0)|Root node|
|_vT_ =(**_s_**_∅, k, T_)|Goal node|
|_rv_|Reward accumulated along the path that _A∗_followed from _v_0 to _v_|
|_Lt_|Lipschitz constant of _Vτ_(**_s_**_, l, t_)|
|ˆ_Vτ_(**_s_**_, l, t_)|Heuristic function approximating _Vτ_(**_s_**_, l, t_)|
|_S†_|Anchor set|



17 

## **C Counterfactual identifiability of element-wise bijective SCMs** 

In this section, we show that _element-wise bijective_ SCMs, a subclass of bijective SCMs which we formally define next, are counterfactually identifiable. 

**Definition 8.** _An SCM C is element-wise bijective iff it is bijective and there exist functions gS,i_ : R _×A×_ R _→_ R _with i ∈{_ 1 _, . . . , D} such that, for every combination of_ _**s** t_ +1 _,_ _**s** t, at,_ _**u** t with_ _**s** t_ +1 = _gS_ ( _**s** t, at,_ _**u** t_ ) _, it holds that st_ +1 _,i_ = _gS,i_ ( _**s** t, at, ut,i_ ) _for i ∈{_ 1 _, . . . , D}._ 

Under our assumption that the transition mechanism _gS_ is continuous with respect to its third argument, it is easy to see that, for any element-wise bijective SCM, the functions _gS,i_ are always strictly monotonic functions of the respective _ut,i_ . Based on this observation, we have the following theorem of counterfactual identifiability: 

**Theorem 9.** _Let C and M be two element-wise bijective SCMs with transition mechanisms gS and hS, respectively, and, for any observed transition_ ( _**s** t, at,_ _**s** t_ +1) _, let_ _**u** t_ = _gS[−]_[1][(] _**[s]**[t][, a][t][,]_ _**[ s]**[t]_[+1][)] _[ and]_ _**[u]**_[˜] _[t]_[=] _[ h][−] S_[1][(] _**[s]**[t][, a][t][,]_ _**[ s]**[t]_[+1][)] _[.] Moreover, given any_ _**s** ∈S, a ∈A, let_ _**s**[′]_ = _gS_ ( _**s** , a,_ _**u** t_ ) _and_ _**s**[′′]_ = _hS_ ( _**s** , a,_ ˜ _**u** t_ ) _. If P[C]_ ( _**S** t_ +1 _|_ _**S** t_ = _**s** , At_ = _a_ ) _= P[M]_ ( _**S** t_ +1 _|_ _**S** t_ = _**s** , At_ = _a_ ) _for all_ _**s** ∈S, a ∈A, it must hold that_ _**s**[′]_ = _**s**[′′] ._ 

_Proof._ We prove the theorem by induction, starting by establishing the base case _s[′]_ 1[=] _[ s][′′]_ 1[.][Without][loss][of] generality, assume that both _gS,_ 1 and _hS,_ 1 are strictly increasing with respect to their third argument. Since the two SCMs entail the same transition distributions, we have that 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2023_Finding_Counterfactually_Optimal_Action_Sequences_in_Continuous_State_Spaces_images/nips_2023_Finding_Counterfactually_Optimal_Action_Sequences_in_Continuous_State_Spaces.pdf-0018-06.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2023_Finding_Counterfactually_Optimal_Action_Sequences_in_Continuous_State_Spaces_images/nips_2023_Finding_Counterfactually_Optimal_Action_Sequences_in_Continuous_State_Spaces.pdf-0018-07.png)


where ( _∗_ ) holds because both SCMs are element-wise bijective, and ( _∗∗_ ) holds because _gS,_ 1 and _hS,_ 1 are increasing with respect to their third argument. Similarly, we have that 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2023_Finding_Counterfactually_Optimal_Action_Sequences_in_Continuous_State_Spaces_images/nips_2023_Finding_Counterfactually_Optimal_Action_Sequences_in_Continuous_State_Spaces.pdf-0018-09.png)


where in ( _⋆_ ) _,_ ( _†_ ) we have used the monotonicity of _gS_ and _hS_ , and ( _⋆⋆_ ) follows from the previous result. The last equality implies that _s[′]_ 1[and] _[s][′′]_ 1[correspond][to][the][same][quantile][of][the][distribution][for] _[S][t]_[+1] _[,]_[1] _[|]_ _**[ S]**[t]_[=] _**[ s]**[, A][t]_[=] _[ a]_[.] Therefore, it is easy to see that _s[′]_ 1[=] _[ s][′′]_ 1[since][the][opposite][would][be][in][contradiction][to] _[g][S,]_[1][being][bijective.] Note that, we can reach that conclusion irrespective of the direction of monotonicity of _gS,_ 1 and _hS,_ 1, since any change in the direction of the inequalities happening at step ( _∗∗_ ) is reverted at steps ( _⋆_ ) and ( _†_ ). 

Now, starting from the inductive hypothesis that _s[′] i_[=] _[ s][′′] i_[for][all] _[i][ ∈{]_[1] _[, . . . , n][}]_[with] _[n < D]_[,][we][show][the] inductive step, _i.e._ , _s[′] n_ +1[=] _[ s][′′] n_ +1[.][Again,][without][loss][of][generality,][assume][that][both] _[g][S,n]_[+1][and] _[h][S,n]_[+1][are] strictly increasing with respect to their last argument. Note that, the two SCMs entail the same transition distributions, _i.e._ , the same joint distributions for _**S** t_ +1 _|_ _**S** t, At_ . Following from the law of total probability, they also entail the same conditional distributions for _St_ +1 _,n_ +1 _|_ _**S** t_ +1 _,≤n,_ _**S** t, At_ , where we use the notation _**x** ≤n_ to refer to a vector that contains the first _n_ elements of a _D_ -dimensional vector _**x**_ . Therefore, we have that 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2023_Finding_Counterfactually_Optimal_Action_Sequences_in_Continuous_State_Spaces_images/nips_2023_Finding_Counterfactually_Optimal_Action_Sequences_in_Continuous_State_Spaces.pdf-0018-12.png)


18 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2023_Finding_Counterfactually_Optimal_Action_Sequences_in_Continuous_State_Spaces_images/nips_2023_Finding_Counterfactually_Optimal_Action_Sequences_in_Continuous_State_Spaces.pdf-0019-00.png)


where for the first equality we have used the inductive hypothesis, ( _∗_ ) holds because both SCMs are element-wise bijective, and ( _∗∗_ ) holds because _gS,n_ +1 and _hS,n_ +1 are increasing with respect to their third argument. Similarly, we get that 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2023_Finding_Counterfactually_Optimal_Action_Sequences_in_Continuous_State_Spaces_images/nips_2023_Finding_Counterfactually_Optimal_Action_Sequences_in_Continuous_State_Spaces.pdf-0019-02.png)


where in ( _⋆_ ) _,_ ( _†_ ) we have used the invertibility and monotonicity of _gS_ and _hS_ , and ( _⋆⋆_ ) follows from the previous result. With the same argument as in the base case, the last equality implies that _s[′] n_ +1[=] _[s][′′] n_ +1[.] That concludes the proof. 

19 

## **D Proofs** 

## **D.1 Proof of Theorem 4** 

We prove the hardness of our problem as defined in Eq. 9 by performing a reduction from the partition problem [26], which is known to be NP-Complete. In the partition problem, we are given a multiset of _B_ positive integers _V_ = _{v_ 1 _, . . . , vB}_ and the goal is to decide whether there is a partition of _V_ into two subsets _V_ 1 _, V_ 2 with _V_ 1 _∩V_ 2 = _∅_ and _V_ 1 _∪V_ 2 = _V_ , such that their sums are equal, _i.e._ ,[�] _vi∈V_ 1 _[v][i]_[=][ �] _vj ∈V_ 2 _[v][j]_[.] 

Consider an instance of our problem where _S_ = _U_ = R[2] , _A_ contains 2 actions _a_ diff _, a_ null and the horizon is _T_ = _B_ + 1. Let _C_ be an element-wise bijective SCM with arbitrary prior distributions _P[C]_ ( _**U** t_ ) such that their support is on R[2] and a transition mechanism _gS_ such that 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2023_Finding_Counterfactually_Optimal_Action_Sequences_in_Continuous_State_Spaces_images/nips_2023_Finding_Counterfactually_Optimal_Action_Sequences_in_Continuous_State_Spaces.pdf-0020-04.png)


Moreover, assume that the reward function is given by 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2023_Finding_Counterfactually_Optimal_Action_Sequences_in_Continuous_State_Spaces_images/nips_2023_Finding_Counterfactually_Optimal_Action_Sequences_in_Continuous_State_Spaces.pdf-0020-06.png)


where _sum_ ( _V_ ) is the sum of all elements[�] _[B] i_ =1 _[v][i]_[.][Note that,][the SCM] _[ C]_[defined above is Lipschitz-continuous] as suggested by the following Lemma (refer to Appendix D.1.1 for a proof). 

**Lemma 10.** _The SCM C defined by Equations 12, 13 is Lipschitz-continuous according to Definition 1._ 

Now, assume that the counterfactual action sequence can differ in an arbitrary number of actions from the action sequence in the observed episode _τ_ , _i.e._ , _k_ = _T_ and, let the observed action sequence be such that _at_ = _a_ null for _t ∈{_ 0 _, . . . , T −_ 1 _}_ . Lastly, let the initial observed state be _**s**_ 0 = [0 _, v_ 1], the observed states _{_ _**s** t}[T] t_ =1 _[ −]_[2][be][such][that] _**[s]**[t]_[=] _ti_ =1 _[v][i][, v][t]_[+1] for _t ∈{_ 1 _, . . . , T −_ 2 _}_ and the last observed state be �� � _**s** T −_ 1 = [ _sum_ ( _V_ ) _,_ 0]. Then, it is easy to see that the noise variables _**U** t_ have posterior distributions with a point mass on the respective values 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2023_Finding_Counterfactually_Optimal_Action_Sequences_in_Continuous_State_Spaces_images/nips_2023_Finding_Counterfactually_Optimal_Action_Sequences_in_Continuous_State_Spaces.pdf-0020-10.png)


Note that, for all _t ∈{_ 1 _, . . . , T −_ 2 _}_ , we have 0 _≤ st,_ 1 _< sum_ ( _V_ ) and _st,_ 2 _≥_ 1, hence the immediate reward according to Eq. 13 is equal to 0. Consequently, the outcome of the observed episode _τ_ is _o_[+] ( _τ_ ) = _R_ ( _**s** T −_ 1 _, a_ null) = _−_ max(0 _,[sum]_ 2[(] _[V]_[)] ) _−_ max(0 _, −[sum]_ 2[(] _[V]_[)] ) = _−[sum]_ 2[(] _[V]_[)] . Next, we will characterize the counterfactual outcome _o_ ( _τ[′]_ ) of a counterfactual episode _τ[′]_ with a sequence of states _{_ _**s**[′] t[}][T] t_ =0 _[ −]_[1][resulting][from][an][alternative][sequence][of][actions] _[{][a][′] t[}] t[T]_ =0 _[ −]_[1][.][Let] _[D] t[′]_[,] _[N] t[ ′]_[denote][the][set][of] time steps until time _t_ , where the actions taken in a counterfactual episode _τ[′]_ are _a_ diff and _a_ null respectively. Formally, _Dt[′]_[=] _[{][t][′][∈{]_[0] _[, . . . , t][}]_[:] _[a][′] t[′]_[=] _[a]_[diff] _[}]_[,] _[N] t[ ′]_[=] _[{][t][′][∈{]_[0] _[, . . . , t][}]_[:] _[a][′] t[′]_[=] _[a]_[null] _[}]_[.][Then,][as][an][intermediate] result, we get the following Lemma (refer to Appendix D.1.2 for a proof). 

**Lemma 11.** _It holds that s[′] t,_ 1[=][ �] _t[′] ∈Nt[′] −_ 1 _[v][t][′]_[+1] _[for][all][t][ ∈{]_[1] _[, . . . T][−]_[1] _[}][.]_ 

Following from that, we get that 0 _≤ s[′] t,_ 1 _[≤][sum]_[(] _[V]_[)][for][all] _[t][ ∈{]_[1] _[, . . . , T][−]_[1] _[}]_[.][Moreover,][we][can][observe] that the transition mechanism given in Eq. 12 is such that _gS,_ 2( _**S** t, At, Ut,_ 2) = _Ut,_ 2 for all _t ∈{_ 0 _, . . . , T −_ 2 _}_ , independently of _**S** T_ and _At_ . Therefore, it holds that _s[′] t,_ 2[=] _[u][t][−]_[1] _[,]_[2] _[≥]_[1][for] _[t][∈{]_[1] _[, . . . , T][−]_[2] _[}]_[,][and] _s[′]_ 0 _,_ 2[=] _[ s]_[0] _[,]_[2][=] _[ v]_[1] _[≥]_[1.][As][a][direct][consequence,][it][is][easy][to][see][that] _[R]_[(] _**[s]**[′] t[, a][′] t_[) = 0][for][all] _[t][ ∈{]_[0] _[, . . . , T][−]_[2] _[}]_[,] and the counterfactual outcome is given by 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2023_Finding_Counterfactually_Optimal_Action_Sequences_in_Continuous_State_Spaces_images/nips_2023_Finding_Counterfactually_Optimal_Action_Sequences_in_Continuous_State_Spaces.pdf-0020-14.png)


20 

In addition to that, we have that _uT −_ 2 _,_ 2 = 0, hence 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2023_Finding_Counterfactually_Optimal_Action_Sequences_in_Continuous_State_Spaces_images/nips_2023_Finding_Counterfactually_Optimal_Action_Sequences_in_Continuous_State_Spaces.pdf-0021-01.png)


Now, we will show that, if we can find the action sequence _{a[∗] t[}] t[T]_ =0 _[ −]_[1][that][gives][the][optimal][counterfactual] outcome _o_[+] ( _τ[∗]_ ) for the aforementioned instance in polynomial time, then we can make a decision about the corresponding instance of the partition problem, also in polynomial time. To this end, let _{_ _**s**[∗] t[}][T] t_ =0 _[ −]_[1][be][the] sequence of states in the optimal counterfactual realization and, let _DT[∗] −_ 2[=] _[ {][t][ ∈{]_[0] _[, . . . , T][−]_[2] _[}]_[ :] _[ a][∗] t_[=] _[ a]_[diff] _[}]_[,] _NT[∗] −_ 2[=] _[ {][t][′][∈{]_[0] _[, . . . , T][−]_[2] _[}]_[ :] _[ a][∗] t[′]_[=] _[ a]_[null] _[}]_[.] From Eq. 14, we get that the optimal counterfactual outcome is _o_[+] ( _τ[∗]_ ) = _R_ ( _**s**[∗] T −_ 1 _[, a][∗] T −_ 1[),][and][it][is][easy][to] see that the reward function given in Eq. 13 is always less or equal than zero. If _o_ ( _τ[∗]_ ) = 0, it has to hold that 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2023_Finding_Counterfactually_Optimal_Action_Sequences_in_Continuous_State_Spaces_images/nips_2023_Finding_Counterfactually_Optimal_Action_Sequences_in_Continuous_State_Spaces.pdf-0021-03.png)


where ( _∗_ ) follows from Eq. 15. As a consequence, the subsets _V_ 1 = _{vi_ : _i −_ 1 _∈NT[∗] −_ 2 _[}]_[and] _[V]_[2][=] _[ {][v][i]_[:] _[ i][ −]_[1] _[ ∈] DT[∗] −_ 2 _[}]_[partition] _[V]_[and][their][sums][are][equal.] On the other hand, if _o_[+] ( _τ[∗]_ ) _<_ 0, as we will show, there is no partition of _V_ into two sets with equal sums. For the sake of contradiction, assume there are two sets _V_ 1 _, V_ 2 that partition _V_ , with _sum_ ( _V_ 1) = _sum_ ( _V_ 2) = _sum_ ( _V_ ) _/_ 2, and let _NT[′] −_ 2[=] _[{][t][∈{]_[0] _[, . . . , T][−]_[2] _[}]_[:] _[v][t]_[+1] _[∈V]_[1] _[}]_[and] _[D] T[′] −_ 2[=] _[{][t][∈{]_[0] _[, . . . , T][−]_[2] _[}]_[:] _[v][t]_[+1] _[∈V]_[2] _[}]_[.] Then, consider the counterfactual episode _τ[′]_ with an action sequence _{a[′] t[}] t[T]_ =0 _[ −]_[1][such][that][its][elements][take] values _a_ null and _a_ diff based on the sets _NT[′] −_ 2 _[,][ D] T[′] −_ 2[respectively,][with] _[a][′] T −_ 1[taking][an][arbitrary][value.][It][is] easy to see that 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2023_Finding_Counterfactually_Optimal_Action_Sequences_in_Continuous_State_Spaces_images/nips_2023_Finding_Counterfactually_Optimal_Action_Sequences_in_Continuous_State_Spaces.pdf-0021-05.png)


which is a contradiction. This step concludes the reduction and, therefore, the problem given in Eq. 9 cannot be solved in polynomial time, unless _P_ = _NP_ . 

## **D.1.1 Proof of Lemma 10** 

It is easy to see that, for all _**u** ∈U_ and for all _**s** ,_ _**s**[′] ∈S_ , the function _gS_ ( _**S** t, a_ null _,_ _**u**_ ) is such that _∥gS_ ( _**s** , a_ null _,_ _**u**_ ) _− gS_ ( _**s**[′] , a_ null _,_ _**u**_ ) _∥≤∥_ _**s** −_ _**s**[′] ∥_ , and therefore _Ka_ null _,_ _**u**_ = 1 satisfies Definition 1. For the 

21 

case of _At_ = _a_ diff, we have that 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2023_Finding_Counterfactually_Optimal_Action_Sequences_in_Continuous_State_Spaces_images/nips_2023_Finding_Counterfactually_Optimal_Action_Sequences_in_Continuous_State_Spaces.pdf-0022-01.png)


_√_ 2 _∥_ _**s** −_ _**s**[′] ∥_ = _√_ 2�( _s_ 1 _− s[′]_ 1[)][2][ + (] _[s]_[2] _[ −][s][′]_ 2[)][2] _[⇒]_[2] _[ ∥]_ _**[s]**[ −]_ _**[s]**[′][∥]_[2][= 2(] _[s]_[1] _[ −][s][′]_ 1[)][2][ + 2(] _[s]_[2] _[−][s][′]_ 2[)][2] _[.]_ 

By combining these, we get 

2 _∥_ _**s** −_ _**s**[′] ∥_[2] _−∥gS_ ( _**s** , a_ diff _,_ _**u**_ ) _− gS_ ( _**s**[′] , a_ diff _,_ _**u**_ ) _∥_[2] _≥_ ( _s_ 1 _− s[′]_ 1[)][2][ + (] _[s]_[2] _[−][s][′]_ 2[)][2] _[ −]_[2] _[|][s]_[1] _[−][s][′]_ 1 _[||][s]_[2] _[−][s][′]_ 2 _[| ⇒]_ 2 _∥_ _**s** −_ _**s**[′] ∥_[2] _−∥gS_ ( _**s** , a_ diff _,_ _**u**_ ) _− gS_ ( _**s**[′] , a_ diff _,_ _**u**_ ) _∥_[2] _≥_ ( _|s_ 1 _− s[′]_ 1 _[| −|][s]_[2] _[−][s][′]_ 2 _[|]_[)][2] _[≥]_[0] _[.]_ 

Hence, we can easily see that _Ka_ diff _,_ _**u**_ = _√_ 2 satisfies Definition 1. Next, we need to show that, for all _a ∈A_ there exists a _Ca ∈_ R+ such that, for all _**s** ,_ _**s**[′] ∈S_ , it holds _|R_ ( _**s** , a_ ) _− R_ ( _**s**[′] , a_ ) _| ≤ Ca ∥_ _**s** −_ _**s**[′] ∥_ . Note that, to show that a function of the form max(0 _, f_ ( _**s**_ )) with _f_ : R[2] _→_ R is Lipschitz continuous, it suffices to show that _f_ ( _**s**_ ) is Lipschitz continuous, since the function max(0 _, x_ ) with _x ∈_ R has a Lipschitz constant equal to 1. 

We start by showing that the function _f_ ( _**s**_ ) = _s_ 1 _− α − s_ 2 _· α_ is Lipschitz continuous, where _α_ = _sum_ ( _V_ ) _/_ 2 is a positive constant. For an arbitrary pair _**s** ,_ _**s**[′] ∈S_ , we have that 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2023_Finding_Counterfactually_Optimal_Action_Sequences_in_Continuous_State_Spaces_images/nips_2023_Finding_Counterfactually_Optimal_Action_Sequences_in_Continuous_State_Spaces.pdf-0022-07.png)


We also have that 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2023_Finding_Counterfactually_Optimal_Action_Sequences_in_Continuous_State_Spaces_images/nips_2023_Finding_Counterfactually_Optimal_Action_Sequences_in_Continuous_State_Spaces.pdf-0022-09.png)


By combining these, we get 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2023_Finding_Counterfactually_Optimal_Action_Sequences_in_Continuous_State_Spaces_images/nips_2023_Finding_Counterfactually_Optimal_Action_Sequences_in_Continuous_State_Spaces.pdf-0022-11.png)


Hence, we arrive to _|f_ ( _**s**_ ) _− f_ ( _**s**[′]_ ) _| ≤[√]_ 1 + _α ∥_ _**s** −_ _**s**[′] ∥_ , and the function _f_ is Lipschitz continuous. It is easy to see that the function _ϕ_ ( _**s**_ ) = _α − s_ 1 _− s_ 2 _· α_ is also Lipschitz continuous with the proof being almost identical. As a direct consequence, the reward function given in Equation 13 satisfies Definition 1 with _Ca_ null = _Ca_ diff = 2 ~~�~~ 1 + _[sum]_ 2[(] _[V]_[)] . This concludes the proof of the lemma. 

## **D.1.2 Proof of Lemma 11** 

We will prove the lemma by induction. For the base case of _t_ = 1, we distinguish between the cases _a[′]_ 0[=] _[ a]_[diff] and _a[′]_ 0[=] _[a]_[null][.][In][the][first][case,][we][have] _[s][′]_ 1 _,_ 1[=] _[u]_[0] _[,]_[1][+] _[ s]_[0] _[,]_[1] _[−][s]_[0] _[,]_[2][=] _[v]_[1][+ 0] _[ −][v]_[1][=][0][and] _[N]_ 0 _[ ′]_[=] _[∅]_[and,] therefore, the statement holds. In the second case, we have _s[′]_ 1 _,_ 1[=] _[ u]_[0] _[,]_[1][+] _[ s]_[0] _[,]_[1][=] _[ v]_[1][+ 0 =] _[ v]_[1][,] _[N]_ 0 _[ ′]_[=] _[ {]_[0] _[}]_[and] � _t[′] ∈N_ 0 _[′][v][t][′]_[+1][=] _[ v]_[1][.][Therefore,][the][statement][also][holds.] 

For the inductive step ( _t >_ 1), we assume that _s[′] t−_ 1 _,_ 1[=][�] _t[′] ∈Nt[′] −_ 2 _[v][t][′]_[+1][and][we][will][show][that] _[s][′] t,_ 1[=] � _t[′] ∈Nt[′] −_ 1 _[v][t][′]_[+1][.][Again,][we][distinguish][between][the][cases] _[a][′] t−_ 1[=] _[ a]_[diff][and] _[a][′] t−_ 1[=] _[ a]_[null][.][However,][note][that,] in both cases, _s[′] t−_ 1 _,_ 2[=] _[ u][t][−]_[2] _[,]_[2][+ 0 =] _[ v][t]_[.][Therefore,][in][the][case][of] _[a][′] t−_ 1[=] _[ a]_[diff][,][we][get] 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2023_Finding_Counterfactually_Optimal_Action_Sequences_in_Continuous_State_Spaces_images/nips_2023_Finding_Counterfactually_Optimal_Action_Sequences_in_Continuous_State_Spaces.pdf-0022-16.png)


22 

where the last equation holds because _a[′] t−_ 1[=] _[ a]_[diff][and,][therefore,] _[N] t[ ′] −_ 1[=] _[ N] t[ ′] −_ 2[.][In][the][case][of] _[a][′] t−_ 1[=] _[ a]_[null][,] we get 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2023_Finding_Counterfactually_Optimal_Action_Sequences_in_Continuous_State_Spaces_images/nips_2023_Finding_Counterfactually_Optimal_Action_Sequences_in_Continuous_State_Spaces.pdf-0023-01.png)


where the last equation holds because _a[′] t−_ 1[=] _[ a]_[null][and,][therefore,] _[N] t[ ′] −_ 1[=] _[ N] t[ ′] −_ 2 _[∪{][t][ −]_[1] _[}]_[.] 

## **D.2 Proof of Lemma 5** 

We will prove the proposition by induction, starting from the base case, where _t_ = _T −_ 1. First, Let _t_ = _T −_ 1 and _l_ = _k_ . It is easy to see that, if the process is at a state _**s** ∈S_ in the last time step with no action changes left, the best reward that can be achieved is _R_ ( _**s** , aT −_ 1), as already discussed after Eq. 10. Therefore, it holds that _|Vτ_ ( _**s** , k, T −_ 1) _− Vτ_ ( _**s**[′] , k, T −_ 1) _|_ = _|R_ ( _**s** , aT −_ 1) _− R_ ( _**s**[′] , aT −_ 1) _| ≤ CaT −_ 1 _∥_ _**s** −_ _**s**[′] ∥≤ C ∥_ _**s** −_ _**s**[′] ∥_ , where the last step holds because _C_ = max _a∈A Ca_ . Now, consider the case of _t_ = _T −_ 1 with _l_ taking an arbitrary value in _{_ 0 _, . . . , k −_ 1 _}_ . Let _**s** ,_ _**s**[′]_ be two states in _S_ and _a[∗]_ be the action that gives the maximum immediate reward at state _**s**_ , that is, _a[∗]_ = argmax _a∈A{R_ ( _**s** , a_ ) _}_ . Then, we get 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2023_Finding_Counterfactually_Optimal_Action_Sequences_in_Continuous_State_Spaces_images/nips_2023_Finding_Counterfactually_Optimal_Action_Sequences_in_Continuous_State_Spaces.pdf-0023-05.png)


where ( _∗_ ) follows from the fact that _R_ ( _**s**[′] , a[∗]_ ) _≤_ max _a∈A{R_ ( _**s**[′] , a_ ) _}_ . Therefore, for any _l ∈{_ 0 _, . . . , k}_ and _**s** ,_ _**s**[′] ∈S_ , it holds that _|Vτ_ ( _**s** , l, T −_ 1) _− Vτ_ ( _**s**[′] , l, T −_ 1) _| ≤ LT −_ 1 _∥_ _**s** −_ _**s**[′] ∥_ , where _LT −_ 1 = _C_ . 

Now, we will proceed to the induction step. Let _t < T −_ 1, _l < k_ and, as an inductive hypothesis, assume that _Lt_ +1 _∈_ R+ as defined in Lemma 5 is such that, for all _l ∈{_ 0 _, . . . , k}_ and _**s** ,_ _**s**[′] ∈S_ , it holds that _|Vτ_ ( _**s** , l, t_ + 1) _− Vτ_ ( _**s**[′] , l, t_ + 1) _| ≤ Lt_ +1 _∥_ _**s** −_ _**s**[′] ∥_ . Additionally, let ( _**s** a, la_ ) _,_ ( _**s**[′] a[, l][a]_[)][denote][the] enhanced states that follow from ( _**s** , l_ ) _,_ ( _**s**[′] , l_ ) after taking an action _a_ , _i.e._ , ( _**s** a, la_ ) = _Fτ,t_[+][((] _**[s]**[, l]_[)] _[ , a]_[)][and] ( _**s**[′] a[, l][a]_[) =] _[ F]_[ +] _τ,t_[((] _**[s]**[′][, l]_[)] _[ , a]_[)][.][Lastly,][let] _[a][∗]_[be][the][action][that][maximizes][the][future][total][reward][starting][from] state _**s**_ , _i.e._ , _a[∗]_ = argmax _a∈A{R_ ( _**s** , a_ ) + _Vτ_ ( _**s** a, la, t_ + 1) _}_ . Then, we have that 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2023_Finding_Counterfactually_Optimal_Action_Sequences_in_Continuous_State_Spaces_images/nips_2023_Finding_Counterfactually_Optimal_Action_Sequences_in_Continuous_State_Spaces.pdf-0023-08.png)


In the above, ( _∗_ ) holds due to _R_ ( _**s**[′] , a[∗]_ ) + _Vτ_ ( _**s**[′] a[∗][, l][a][∗][, t]_[ + 1)] _[≤]_[max] _[a][∈A][{][R]_[(] _**[s]**[′][, a]_[) +] _[ V][τ]_[(] _**[s]**[′] a[, l][a][, t]_[ + 1)] _[}]_[,][(] _[∗∗]_[)] follows from the inductive hypothesis, and ( _∗∗∗_ ) holds because _C_ = max _a∈A Ca_ and _K_ _**u** t_ = max _a∈A Ka,_ _**u** t_ . It is easy to see that, similar arguments hold for the simple case of _l_ = _k_ , therefore, we omit the details. This concludes the inductive step and the proof of Lemma 5. 

## **D.3 Proof of Proposition 6** 

We will prove the proposition by induction, starting from the base case, where _t_ = _T −_ 1. If _t_ = _T −_ 1, the algorithm initializes _V_[ˆ] _τ_ ( _**s** , l, T −_ 1) to max _a∈A R_ ( _**s** , a_ ) for all _**s** ∈S†_ , _l ∈{_ 0 _, . . . , k −_ 1 _}_ and _V_[ˆ] _τ_ ( _**s** , k, T −_ 1) to 

23 

_R_ ( _**s** , aT −_ 1). It is easy to see that those values are optimal, as already discussed after Eq. 10. Therefore, the base case _V_[ˆ] _τ_ ( _**s** , l, T −_ 1) _≥ Vτ_ ( _**s** , l, T −_ 1) follows trivially. 

ˆ Now, we will proceed to the induction step. Let _t < T −_ 1 and, as an inductive hypothesis,ˆ assume that _Vτ_ ( _**s** , l, t_ + 1) _≥ Vτ_ ( _**s** , l, t_ + 1) for all _**s** ∈S†_ , _l ∈{_ 0 _, . . . , k}_ . Our goal is to show that _Vτ_ ( _**s** , l, t_ ) _≥ Vτ_ ( _**s** , l, t_ ) for all _**s** ∈S†_ , _l ∈{_ 0 _, . . . , k}_ . First, let _l < k_ . For a given point _**s** ∈S†_ , Algorithm 1 finds the next state _**s** a_ that would have occurred by taking each action _a_ , _i.e._ , ( _**s** a, la_ ) = _Fτ,t_[+][((] _**[s]**[, l]_[)] _[ , a]_[)][,][and][it][computes][the] associated value _Va_ = min _**s** †∈S†{V_[ˆ] _τ_ ( _**s** †, la, t_ + 1) + _Lt_ +1 _∥_ _**s** † −_ _**s** a∥}_ . Then, it simply sets _V_[ˆ] _τ_ ( _**s** , l, t_ ) equal to max _a∈A {R_ ( _**s** , a_ ) + _Va}_ . We have that 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2023_Finding_Counterfactually_Optimal_Action_Sequences_in_Continuous_State_Spaces_images/nips_2023_Finding_Counterfactually_Optimal_Action_Sequences_in_Continuous_State_Spaces.pdf-0024-02.png)


where ( _∗_ ) follows from the inductive hypothesis, and ( _∗∗_ ) is a consequence of Lemma 5. Then, we get 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2023_Finding_Counterfactually_Optimal_Action_Sequences_in_Continuous_State_Spaces_images/nips_2023_Finding_Counterfactually_Optimal_Action_Sequences_in_Continuous_State_Spaces.pdf-0024-04.png)


Additionally, when _l_ = _k_ , we have _V_[ˆ] _τ_ ( _**s** , k, t_ ) = _R_ ( _**s** , at_ ) + min _**s** †∈S†{V_[ˆ] _τ_ ( _**s** †, k, t_ + 1) + _Lt_ +1 _∥_ _**s** † −_ _**s** at∥}_ and _Vτ_ ( _**s** , k, t_ ) = _R_ ( _**s** , at_ ) + _Vτ_ ( _**s** at, k, t_ + 1). Therefore, the proof for _V_[ˆ] _τ_ ( _**s** , k, t_ ) _≥ Vτ_ ( _**s** , k, t_ ) is almost identical. 

## **D.4 Proof of Theorem 7** 

We start from the case where _t_ = _T −_ 1. Let _v_ = ( _**s** , l, T −_ 1) and, consider an edge associated with action _a[∗]_ connecting _v_ to the goal node _vT_ = ( _**s** ∅, k, T_ ) that carries a reward _R_ ( _**s** , a[∗]_ ). Then, we have 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2023_Finding_Counterfactually_Optimal_Action_Sequences_in_Continuous_State_Spaces_images/nips_2023_Finding_Counterfactually_Optimal_Action_Sequences_in_Continuous_State_Spaces.pdf-0024-08.png)


and the base case holds. 

For the more general case, where _t < T −_ 1, we first establish the following intermediate result, whose proof is given in Appendix D.4.1. 

**Lemma 12.** _For every_ _**s** ,_ _**s**[′] ∈S, l ∈{_ 0 _, . . . , k}, t ∈{_ 0 _, . . . , T −_ 1 _}, it holds that |V_[ˆ] _τ_ ( _**s** , l, t_ ) _− V_[ˆ] _τ_ ( _**s**[′] , l, t_ ) _| ≤ Lt ∥_ _**s** −_ _**s**[′] ∥, where Lt is as defined in Lemma 5._ 

That said, consider an edge associated with an action _a[∗]_ connecting node _v_ = ( _**s** , l, t_ ) to node _va∗_ = ( _**s** a∗ , la∗ , t_ + 1). Then, we have 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2023_Finding_Counterfactually_Optimal_Action_Sequences_in_Continuous_State_Spaces_images/nips_2023_Finding_Counterfactually_Optimal_Action_Sequences_in_Continuous_State_Spaces.pdf-0024-13.png)


That concludes the proof and, therefore, the heuristic function _V_[ˆ] _τ_ is consistent. 

24 

## **D.4.1 Proof of Lemma 12** 

Without loss of generality, we will assume that _l < k_ , since the proof for the case of _l_ = _k_ is similar and more straightforward. We start from the case where _t_ = _T −_ 1 and, for two states _**s** ,_ _**s**[′] ∈S_ we have 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2023_Finding_Counterfactually_Optimal_Action_Sequences_in_Continuous_State_Spaces_images/nips_2023_Finding_Counterfactually_Optimal_Action_Sequences_in_Continuous_State_Spaces.pdf-0025-02.png)


where the last inequality follows from Lemma 5. 

Now, consider the case _t < T −_ 1, and let ( _**s** a, la_ ) denote the enhanced state that follows from ( _**s** , l_ ) after taking an action _a_ at time _t_ , _i.e._ , ( _**s** a, la_ ) = _Fτ,t_[+][((] _**[s]**[, l]_[)] _[ , a]_[).][Then,][we][have] 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2023_Finding_Counterfactually_Optimal_Action_Sequences_in_Continuous_State_Spaces_images/nips_2023_Finding_Counterfactually_Optimal_Action_Sequences_in_Continuous_State_Spaces.pdf-0025-05.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2023_Finding_Counterfactually_Optimal_Action_Sequences_in_Continuous_State_Spaces_images/nips_2023_Finding_Counterfactually_Optimal_Action_Sequences_in_Continuous_State_Spaces.pdf-0025-06.png)


Let _a[∗]_ be the action _a ∈A_ that maximizes the first part of the above subtraction, _i.e._ , 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2023_Finding_Counterfactually_Optimal_Action_Sequences_in_Continuous_State_Spaces_images/nips_2023_Finding_Counterfactually_Optimal_Action_Sequences_in_Continuous_State_Spaces.pdf-0025-08.png)


Then, Eq. 16 implies that 

Now, let _**s**_ ˜ be the _**s** † ∈S†_ that minimizes the second part of the above subtraction, _i.e._ , 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2023_Finding_Counterfactually_Optimal_Action_Sequences_in_Continuous_State_Spaces_images/nips_2023_Finding_Counterfactually_Optimal_Action_Sequences_in_Continuous_State_Spaces.pdf-0025-11.png)


As a consequence and in combination with Eq. 17, we get 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2023_Finding_Counterfactually_Optimal_Action_Sequences_in_Continuous_State_Spaces_images/nips_2023_Finding_Counterfactually_Optimal_Action_Sequences_in_Continuous_State_Spaces.pdf-0025-13.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2023_Finding_Counterfactually_Optimal_Action_Sequences_in_Continuous_State_Spaces_images/nips_2023_Finding_Counterfactually_Optimal_Action_Sequences_in_Continuous_State_Spaces.pdf-0025-14.png)


where in ( _∗_ ) we use the triangle inequality and the fact that the SCM _C_ is Lipschitz-continuous, and ( _∗∗_ ) follows from Lemma 5. 

25 

**Algorithm 2:** Graph search via _A[∗]_ **Input** : States _S_ , actions _A_ , observed action sequence _{at}[t] t_[=] =0 _[T][ −]_[1] , horizon _T_ , transition function _Fτ,t_[+][,] reward function _R_ , constraint _k_ , initial state _**s**_ 0, heuristic function _V_[ˆ] _τ_ . **Initialize** : node _v_ 0 _←{_ “tuple” : ( _**s**_ 0 _,_ 0 _,_ 0) _,_ “rwd” : 0 _,_ “par” : _Null,_ “act” : _Null}_ , stack _action_ ~~_s_~~ _equence ←_ [ ] queue _Q ←{root}_ set _explored ←∅_ **while** _True_ **do** _v ←_ argmax _v′∈Q{v[′] .rwd_ + _V_[ˆ] _τ_ ( _v[′] .tuple_ ) _}_ ; _Q ← Q \ v_ ; `/* Next node to visit */` **if** _v.tuple_ = ( _∗, ∗, T_ ) **then while** _v.par_ = _Null_ **do** _action_ ~~_s_~~ _equence.push_ ( _v.act_ ) ; `/* Retrieve final action sequence */` _v_ = _v.par_ **end return** _action_ ~~_s_~~ _equence_ **end** _explored ← explored ∪{v}_ ; `/* Set node` _v_ `as explored */` **if** _l_ = _k_ **then** _availabe_ ~~_a_~~ _ctions ←{at}_ **else** _availabe_ ~~_a_~~ _ctions ←A_ **end for** _a ∈ available_ ~~_a_~~ _ctions_ **do** ( _**s** , l, t_ ) _← v.tuple_ ( _**s** a, la_ ) _← Fτ,t_[+][((] _**[s]**[, l]_[)] _[ , a]_[)][;] `/* Identify` _v_ `’s children nodes */` _va ←{_ “tuple” : ( _**s** a, la, t_ + 1) _,_ “rwd” : _v.rwd_ + _R_ ( _**s** , a_ ) _,_ “par” : _v,_ “act” : _a}_ **if** _va ̸∈ Q and va ̸∈ explored_ **then** _Q ← Q ∪{va}_ ; `/* Add them to the queue if unexplored */` **end end end** 

## **E A* algorithm** 

Algorithm 2 summarizes the step-by-step process followed by the _A[∗]_ algorithm. Therein, we represent each node _v_ by an object with 4 attributes: (i) the “tuple” ( _**s** , l, t_ ) of the node, (ii) the total reward “rwd” of the path that has led the search from the root node _v_ 0 to the node _v_ , (iii) the parent node “par” from which the search arrived to _v_ , and (iv) the action “act” associated with the edge connecting the node _v_ with its parent. In addition to the queue of nodes to visit, the algorithm maintains a set of explored nodes, and it adds a new node to the queue only if it has not been previously explored. The algorithm terminates when the goal node is chosen to be visited, _i.e._ , the “tuple” attribute of _v_ has the format ( _∗, ∗, T_ ), where _∗_ denotes arbitrary values. Once the goal node _vT_ has been visited, the algorithm reconstructs the action sequence that led from the root node _v_ 0 to the goal node and returns it as output. 

26 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2023_Finding_Counterfactually_Optimal_Action_Sequences_in_Continuous_State_Spaces_images/nips_2023_Finding_Counterfactually_Optimal_Action_Sequences_in_Continuous_State_Spaces.pdf-0027-00.png)


**----- Start of picture text -----**<br>
2 . 1 2 . 1 3 . 10<br>2 . 0 2 . 0<br>3 . 05<br>1 . 9 1 . 9<br>3 . 00<br>10000 15000 20000 25000 30000 10000 15000 20000 25000 30000 10000 15000 20000 25000 30000<br>Size of anchor set, |S†| Size of anchor set, |S†| Size of anchor set, |S†|<br>(a) MC-Lipschitz (b) MC-Uniform (c) Facility-Location<br>EBF EBF EBF<br>**----- End of picture text -----**<br>


Figure 5: Computational efficiency of our method, measured by means of the Effective Branching Factor (EBF), under three different anchor set selection strategies against the size of the anchor set _S†_ . In all panels, we set _Lh_ = 1 _._ 0, _Lϕ_ = 0 _._ 1 and _k_ = 3. Error bars indicate 95% confidence intervals over 200 executions of the _A[∗]_ algorithm for 200 patients with horizon _T_ = 12. 

## **F Experimental evaluation of anchor set selection strategies** 

In this section, we benchmark the anchor set selection strategy presented in Section 4 against two alternative competitive strategies using the sepsis management dataset and the same experimental setup as in Section 5. More specifically, we consider the following anchor set selection strategies: 

- (i) _MC-Lipschitz_ : This is the strategy described in depth in Section 4, based on Monte Carlo simulations of counterfactual episodes under randomly sampled counterfactual action sequences. Notably, the time steps where each counterfactual action sequence differs from the observed one are sampled proportionally to the respective Lipschitz constant _Lt_ of the SCM’s transition mechanism. To ensure a fair comparison with other strategies, instead of controlling the number of sampled action sequences _M_ , we fix the desired size of the anchor set _S†_ , and we repeatedly sample counterfactual action sequences until the specified size is met. 

- (ii) _MC-Uniform_ : This strategy is a variant of the previous strategy where we sample the time steps where each counterfactual action sequence differs from the observed one uniformly at random, rather than biasing the sampling towards time steps with higher Lipschitz constants _Lt_ . 

- (iii) _Facility-Location_ : Under this strategy, the anchor set is the solution to a minimax facility location problem defined using the observed available data. Let _So_ be the union of all state vectors observed in all episodes _τ_ in a given dataset. Then, we choose an anchor set _S† ⊂So_ of fixed size _|S†|_ = _b_ , such that the maximum distance of any point in _So_ to its closest point in _S†_ is minimized. Here, the rationale is that counterfactual states resulting from counterfactual action sequences for one observed episode are likely to be close to the observed states of some other episode in the data. Formally, 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2023_Finding_Counterfactually_Optimal_Action_Sequences_in_Continuous_State_Spaces_images/nips_2023_Finding_Counterfactually_Optimal_Action_Sequences_in_Continuous_State_Spaces.pdf-0027-07.png)


Although the above problem is known to be NP-Complete, we find a solution using the farthest-point clustering algorithm, which is known to have an approximation factor equal to 2 and runs in polynomial time. The algorithm starts by adding one point from _So_ to _S†_ at random. Then, it proceeds iteratively and, at each iteration, it adds to _S†_ the point from _So_ that is the furthest from all points already in _S†_ , _i.e._ , _S†_ = _S† ∪_ _**s**_ , where _**s**_ = max _**s** ′∈So_ �min _**s** †∈S† ||_ _**s**[′] −_ _**s** †||_ �. The algorithm terminates after _b_ iterations. 

**Results.** We compare the computational efficiency of our method under each of the above anchor set selection strategies for various values of the size of the anchor set _|S†|_ . Figure 5 summarizes the results. We observe 

27 

Table 2: Levels of vasopressors and intravenous fluids corresponding to the 25 actions in _A_ 

|**Vasopressors (mcg/kg/min)**|**Intravenous fuids (mL/4 hours)**|
|---|---|
|0_._00|0|
|0_._04|30|
|0_._113|80|
|0_._225|279|
|0_._788|850|



that the Facility-Location selection strategy performs rather poorly compared to the other two strategies, achieving an effective branching factor (EBF) higher than 3. In contrast, the MC-Lipschitz and MC-Uniform strategies achieve an EBF close to 2, which decreases rapidly as the size of the anchor set increases. Among these two strategies, the MC-Lipschitz strategy, which we use in our experiments in Section 5, achieves the lowest EBF. 

## **G Additional details on the experimental setup** 

## **G.1 Features and actions in the sepsis management dataset** 

As mentioned in Section 5, our state space is _S_ = R _[D]_ , where _D_ = 13 is the number of features. We distinguish between three types of features: (i) demographic features, whose values remain constant across time, (ii) contextual features, for which we maintain their observed (and potentially varying) values throughout all counterfactual episodes and, (iii) time-varying features, whose counterfactual values are given by the SCM _C_ . The list of features is as follows: 

- Gender (demographic) 

- Re-admission (demographic) 

- Age (demographic) 

- Mechanical ventilation (contextual) 

- FiO2 (time-varying) 

- PaO2 (time-varying) 

- Platelet count (time-varying) 

- Bilirubin (time-varying) 

- Glasgow Coma Scale (time-varying) 

- Mean arterial blood pressure (time-varying) 

- Creatinine (time-varying) 

- Urine output (time-varying) 

- SOFA score (time-varying) 

To define our set of actions _A_ we follow related work [6, 55, 57], and we consider 25 actions corresponding to 5 _×_ 5 levels of administered vasopressors and intravenous fluids. Specifically, for both vasopressors and fluids, we find all non-zero values appearing in the data, and we divide them into 4 intervals based on the quartiles of the observed values. Then, we set the 5 levels to be the median values of the 4 intervals and 0. Table G.1 shows the resulting values of vasopressors and fluids. 

28 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2023_Finding_Counterfactually_Optimal_Action_Sequences_in_Continuous_State_Spaces_images/nips_2023_Finding_Counterfactually_Optimal_Action_Sequences_in_Continuous_State_Spaces.pdf-0029-00.png)


**----- Start of picture text -----**<br>
0.01<br>0.02<br>0.05 − 10%<br>0.1<br>0.2<br>0.5<br>1.0 − 20%<br>1.1<br>1.2<br>1.5<br>Lh<br>0.010.020.050.10.20.51.01.11.21.5<br>φ<br>L<br>**----- End of picture text -----**<br>


Figure 6: Goodness of fit of the Lipschitz-continuous SCM _C_ , measured by means of the percentage decrease in log likelihood of the data in comparison with an SCM trained without Lipschitz-continuity constraints. The x and y axes correspond to different enforced values for the Lipschitz constants _Lh_ , _Lϕ_ of the location and scale networks _h_ and _ϕ_ , respectively. Darker values indicate that the learned SCM achieves a significantly lower log likelihood than the unconstrained SCM. 

## **G.2 Additional details on the network architecture & training** 

We represent the location and scale functions _h_ and _ϕ_ of the SCM _C_ using neural networks with 1 hidden layer, 200 hidden units and _tanh_ activation functions. The mapping from a state _**s**_ and an action _a_ to the hidden vector _**z**_ takes the form _**z**_ = _tanh_ ( _Ws_ _**s**_ + _Wa_ _**a**_ ), where _**a**_ is a 2-D vector representation of the respective action. The mapping from the hidden vector _**z**_ to the network’s output is done via a fully connected layer with weights _Wz_ . To enforce a network to have a Lipschitz constant _L_ with respect to the state input, we apply spectral normalization to the weight matrices _Ws_ and _Wz_ , so that their spectral norms are _∥Ws∥_ 2 = _∥Wz∥_ 2 = 1. Additionally, we add 2 intermediate layers between the input and the hidden layer and between the hidden layer and the output layer, each one multiplying its respective input by a constant _√L_ . Since it is known that the _tanh_ activation function has a Lipschitz constant of 1, it is easy to see that, by function composition, the resulting network is guaranteed to be Lipschitz continuous with respect to its state input with constant _L_ . Note that, since the matrix _Wa_ is not normalized, the network’s Lipschitz constant with respect to the action input can be arbitrary. 

To train the SCM _C_ , for each sample, we compute the negative log-likelihood of the observed transition under the SCM’s current parameter values ( _i.e._ , network weight matrices & covariance matrix of the multivariate Gaussian prior), and we use that as a loss. Subsequently, we optimize those parameters using the Adam optimizer with a learning rate of 0 _._ 001, a batch size of 256, and we train the model for 100 epochs. 

We train the model under multiple values of the Lipschitz constants _Lh_ , _Lϕ_ of the location and scale networks, and we evaluate the log-likelihood of the data under each model using 5-fold cross-validation. Specifically, for each configuration of _Lh_ and _Lϕ_ , we randomly split the dataset into a training and a validation set (with a size ratio 4-to-1), we train the corresponding SCM using the training set, and we evaluate the log-likelihood of the validation set based on the trained SCM. This results in the log-likelihood always being measured on a different set of data points than the one used for training. For each configuration of _Lh_ and _Lϕ_ , we repeat the aforementioned procedure 5 times and we report the average log-likelihood achieved on the validation set. In addition, we train an unconstrained model without spectral normalization, which can have an arbitrary Lipschitz constant. 

Figure 6 shows the decrease in log-likelihood of the respective constrained model as a percentage of the log-likelihood achieved by the unconstrained model, under various values of the Lipschitz constants _Lh_ , _Lϕ_ . We observe that, the model’s performance is predominantly affected by the Lipschitz constant of the location 

29 

network _Lh_ , and its effect is more pronounced when _Lh_ takes values smaller than 1. Additionally, we can see that the scale network’s Lipschitz constant _Lϕ_ has a milder effect on performance, especially when _Lh_ is greater or equal than 1. Since we are interested in constraining the overall Lipschitz constant of the SCM _C_ , in our experiments in Section 5, we set _Lh_ = 1 and _Lϕ_ = 0 _._ 1, which achieves a log-likelihood only 6% lower to that of the best model trained without any Lipschitz constraint. 

30 

