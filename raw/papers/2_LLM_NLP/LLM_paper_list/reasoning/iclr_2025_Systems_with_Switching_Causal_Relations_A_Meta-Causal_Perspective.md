# SYSTEMS WITH SWITCHING CAUSAL RELATIONS: A META-CAUSAL PERSPECTIVE 

**Moritz Willig**[1] **, Tim Nelson Tobiasch**[1] **, Florian Peter Busch**[1] _[,]_[2] **, Jonas Seng**[1] **, Devendra Singh Dhami**[3] **, Kristian Kersting**[1] _[,]_[2] _[,]_[4] _[,]_[5] 

1 Department of Computer Science, Technical University of Darmstadt 

2 Hessian Center for AI (hessian.AI) 

> 3 Dept. of Mathematics and Computer Science, Eindhoven University of Technology 

4 Centre for Cognitive Science, Technical University of Darmstadt 

5 German Research Center for AI (DFKI) 

## ABSTRACT 

Most work on causality in machine learning assumes that causal relationships are driven by a constant underlying process. However, the flexibility of agents’ actions or tipping points in the environmental process can change the qualitative dynamics of the system. As a result, new causal relationships may emerge, while existing ones change or disappear, resulting in an altered causal graph. To analyze these qualitative changes on the causal graph, we propose the concept of _meta-causal states_ , which groups classical causal models into clusters based on equivalent qualitative behavior and consolidates specific mechanism parameterizations. We demonstrate how meta-causal states can be inferred from observed agent behavior, and discuss potential methods for disentangling these states from unlabeled data. Finally, we direct our analysis towards the application of a dynamical system, showing that meta-causal states can also emerge from inherent system dynamics, and thus constitute more than a context-dependent framework in which mechanisms emerge only as a result of external factors. 

## 1 INTRODUCTION 

Structural Causal Models (SCM) have become the de facto formalism in causality by representing causal relations a structural equation models (Pearl (2009); Spirtes et al. (2000)). While SCM are most useful for representing the intricate mechanistic details of systems, it can be challenging to derive the general qualitative behavior that emerges from the interplay of individual equations. Talking about the general _type_ of relation emitted by particular mechanisms generalizes above the narrow computational view and has the ability to inspect causal systems from a more general perspective. Causal graphs can be subject to change whenever novel mechanisms emerge or vanish within a system. Prominently, agents can ‘break’ the natural unfolding of systems dynamics by forecasting system behavior and preemptively intervening in the course of events. As inherent parts of the environment, agents commonly establish or suppress the emergence of causal connections (Zhang & Bareinboim, 2017; Lee & Bareinboim, 2018; Dasgupta et al., 2019). 

Consider the scenario shown in Fig. 1 (left), where an agent _A_ (with position _AX_ ) follows an agent _B_ (with position _BX_ ) according to its policy _A_ _**π**_ . We are interested in answering the question ‘What is the cause of the current position of agent A?’. In general, the observed system can be formalized as follows: _AX_ := _fA_ ( _BX , UA_ ) and _BX_ := _fB_ ( _UB_ ). From a classical causal perspective we observe that _B → A_ , since _AX_ is conditioned on _BX_ via the equation _fA_ . As a result, we infer that _BX_ is the cause of _A_ ’s position. This perspective, only attributes causes within the current state of the system. However, it disregards the causes that lead to the emergence of _fA_ . Only when we take a meta-causal stance and think about _how the equation fA came to be_ in the first place, we can give the answer that _A_ , through its policy _A_ _**π**_ , is the (meta-causal) cause for its own position.[1] 

> 1Although similar conclusions might be derived from classical causal perspectives, we refer the reader to our discussions in Sec. 4 for further considerations and limitations on these approaches. 

1 

Systems with Switching Causal Relations: A Meta-Causal Perspective 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_Systems_with_Switching_Causal_Relations_A_Meta-Causal_Perspective_images/iclr_2025_Systems_with_Switching_Causal_Relations_A_Meta-Causal_Perspective.pdf-0002-01.png)


**----- Start of picture text -----**<br>
Classical Attribution Meta-Causal Attribution<br>AX BX Aπ<br>A B<br>Agent A Establishes Distance Constraint Classical Causal Relation<br>Meta-Causal Mechanism AX BX<br>Root Cause Attribution<br>Observed Causal Direction<br>**----- End of picture text -----**<br>


Figure 1: **Meta-Causality Identifies the Policy as a Meta Root Cause.** Agent _A_ intends to maintain its distance from agent _B_ by conditioning its position _AX_ on the position _BX_ , which establishes a control mechanism, _AX_ := _f_ ( _BX_ ). In standard causal inference, we would infer _BX → AX_ and, therefore, _B_ to be the root cause. Taking a meta-causal perspective reveals however, that _A_ _**π**_ establishes the edge _BX → AX_ in the first place ( _A_ _**π** →_ ( _BX → AX_ )) such that _A_ _**π**_ is considered the root cause on the meta-level. _(Best Viewed in Color)_ 

Causal models do not exist in isolation, but emerge from the environmental dynamics of an underlying mediation process (Hofmann et al., 2020; Arnellos et al., 2006). Changes in causal relations due to intervention or environmental change are often assumed to be helpful conditions for identifying causal structures (Pearl, 2009; Peters et al., 2016; Zhang & Bareinboim, 2017; Dasgupta et al., 2019; Gerstenberg, 2024). These operations often work on the individual causal graphs, but never reason about the emerging meta-causal structure that governs the transitions between the different SCM. As a first formalization of metacausal models in this area, we describe how qualitative changes in the causal graphs can be summarized by metacausal models. 

**Contributions and Structure.** The contributions of this work are as follows: 

- To the best of our knowledge, we are the first to formally introduce typing mechanisms that generalize edges in causal graphs, and the first to provide a formalization of metacausal models (MCM) that are capable of capturing the switching type dynamics of causal mechanisms. 

- We present an approach to discover the number of meta-causal states in the bivariate case. 

- We demonstrate that meta-causal models can be more powerful than classical SCM when it comes to expressing qualitative differences within certain system dynamics. 

- We show how meta-causal analysis can lead to a different root cause attribution than classical causal inference, and is furthermore able to identify causes of mechanistic changes even when no actual change in effect can be observed. 

We proceed as follows: In Section 2 we describe the necessary basics of Pearlian causality, mediation processes, and typing assumptions. In Sec. 3 we introduce meta-causality by first formalizing metacausal frames, and the role of types as a generalization to a binary edge representations. Finally we define meta-causal models that are able to capture the qualitative dynamics in SCM behavior. In Sec. 4, we showcase several examples of meta-causal applications. Finally, we discuss connections to related work in Sec. 5 and conclude our findings in Sec. 6. 

## 2 BACKGROUND 

Providing a higher-level perspective on meta-causality touches on a number of existing works that leverage meta-causal ideas, even if not explicitly stated. We will highlight relations of these works in the ‘Related Work’ of section 5. Here, we continue to provide the necessary concepts on causality, mediation processes and typing, needed for the definitions in our paper: 

**Causal Models.** A common formalization of causality is provided via Structural Causal Models (SCM; (Spirtes et al., 2000; Pearl, 2009)). An SCM is defined as a tuple _M_ := ( **U** _,_ **V** _,_ **F** _, P_ **U** ), where **U** is the set of exogenous variables, **V** is the set of endogenous variables, **F** is the set of structural equations determining endogenous variables, and _P_ **U** is the distribution over exogenous variables **U** . Every endogenous variable _Vi ∈_ **V** is determined via a structural equation _vi_ := _fi_ (pa( _vi_ )) that takes in a set of parent values pa( _vi_ ), consisting of endogenous and exogenous variables, and 

2 

Systems with Switching Causal Relations: A Meta-Causal Perspective 

outputs the value of _vi_ . The set of all variables is denoted by **X** = **U** _∪_ **V** with values **x** _∈_ _**X**_ and _N_ = _|_ **X** _|_ . Every SCM _M_ entails a directed graph _G[M]_ that can be constructed by adding edges ( _i, j_ ) _∈_ **X** _×_ **X** for all variables _Xj ∈_ **X** and its parents _Xi ∈_ Pa( _Xj_ ). This can be expressed as an adjacency matrix _A ∈_ B _[N][×][N]_ where _Aij_ := 1 if ( _i, j_ ) _∈G_ and _Aij_ := 0 otherwise. 

**Mediation Processes.** Causal effects are embedded in an environment that governs the dynamics and mediates between different causes and effects. To reason about the existence of causal relations, we need to know and consider this process-embedding environment. We define a _mediation process E_ = ( _S, σ_ ), adapted from Markov Decision Processes (MDP; Bellman (1957)) for our setting. Here, _S_ is the state space of the environment and _σ_ : _S →S_ is the (possibly non-deterministic) transition function that takes the current state and outputs the next one. If we also have an initial state s0 _∈S_ , we call ( _E,_ s0) an _initialized mediation process_ . As we are concerned with the general mediation process, we omit the common notion of a reward function _r_ . Furthermore, we omit an explicit action space _A_ and agents’ policy _**π**_ and model actions directly as part of the transition function _σ_ . In accordance to considerations of (Cohen, 2022), this eases our treatise of environment processes and agent actions as now both are defined over the same domain. 

**Typing.** In the following section, we will make use of an identification function _I_ to determine the presence or absence of edges between any two variables. In particular, one could make use of a set of different identification functions to identify different _types_ of edges. Previous work on typing causality exists (Brouillard et al., 2022), but is primarily concerned with types of variables, rather than the typing of structural relations. Several other works in the field of cognitive science consider the perception of different types of mechanistic relations (e.g., ‘causing’, ‘enabling’, ‘preventing’, ‘despite’) based on the role that different objects play in physical scenes (Chockler & Halpern, 2004; Wolff, 2007; Sloman et al., 2009; Walsh & Sloman, 2011; Gerstenberg, 2022). Since all objects are usually governed by the same physical equations, this assignment of types serves to provide post hoc explanations of a scene, rather than to identify inherent properties of its computational aspects. Gerstenberg (2024) provides an ‘intuitive psychology’ example of agents that fits well with our scenario. 

## 3 META-CAUSALITY 

Meta-Causality is concerned with the general mechanisms that lead to the emergence of causal relations. Usually, the mediating process may be too fine-grained to yield interpretable models. Therefore, we consider a set of variables of interest **X** derived via _φ_ : _S →_ _**X**_ . Here, _φ_ could be defined as a summarization or abstraction operation over the state space (Rubenstein et al., 2017; Beckers & Halpern, 2019; Anand et al., 2022; Wahl et al., 2023; Keki´c et al., 2023; Willig et al., 2023). In order to identify the type of causal relations from a mediating process, we need to be able to decide on what constitutes such a type. 

**Definition 1** (Meta-Causal Frame) **.** _For a given mediation process E_ = ( _S, σ_ ) _a_ _**meta-causal frame** is a tuple F_ = ( _E,_ **X** _, τ, I_ ) _with:_ 

- _a type-encoder τ_ : _X i × X[S] j[→T][that determines the relationship between the value of a] causal variable X i and the change of another causal variable X[S] j_[=] _[{][ψ]_[:] _[S][→X][ j][}][ with] respect to a environmental state space S which we call a_ _**type** t ∈T ,_ 

- _an identification function I_ : _S ×_ **X** _×_ **X** _→T with I_ ( _s, Xi, Xj_ ) _�→ t_ := _τ_ ( _φ_ ( _s_ ) _i, φj ◦ σ_ ) _that takes a state of the environment and assigns a type for every pair of causal variables Xi, Xj ∈_ **X** _._ 

_Types_ generalize the role of edges in causal graphs. In most classical scenarios, the co-domain _T_ of the type encoder _τ_ is chosen to be Boolean, representing the _existence_ or _absence_ of edges. In other cases, different values _t ∈T_ can be understood as particular types of edges, like positive, negative, or the absence of influence. This will help us to distinguish meta-causal states that share the same graph adjacency. The only requirement for _T_ is that it must contain a special value _0_ , which indicates the total absence of an edge. _Meta-causal states_ now generalize the idea of binary adjacency matrices. 

3 

Systems with Switching Causal Relations: A Meta-Causal Perspective 

**Definition 2** (Meta-Causal State) **.** _In a meta-causal frame F_ = ( _E,_ **X** _, τ, I_ ) _, a_ _**meta-causal state** is a matrix T ∈T[N][×][N] . For a given environment state s ∈S, the_ _**actual meta-causal state** Ts has the entries Ts,ij_ := _I_ ( _s, Xi, Xj_ ) = _τ_ ( _φ_ ( _s_ ) _i, φj ◦ σ_ ) _._ 

A meta-causal state _T ∈T[N][×][N]_ represents a graph containing edges _eij_ of a particular type _t ∈T_ . In particular T _ij_ indicates the presence ( _Tij_ = _0_ ) or absence ( _Tij_ = _0_ ) of individual edges. Our goal for meta-causal models (MCM) is to capture the dynamics of the underlying model. In particular, we are interested in modeling how different meta-causal states transition into each other. This behavior allows us to model them in a finite-state machine (Moore et al., 1956): 

**Definition 3** (Meta-Causal Model) **.** _For a meta-causal frame F_ = ( _E,_ **X** _, σ, I_ ) _, a_ _**meta-causal model** is a finite-state machine defined as a tuple A_ = ( _T[N][×][N] , S, δ_ ) _, where the set of meta-causal states T[N][×][N] is the set of machine states, the set of environment states S is the input alphabet, and δ_ : _T[N][×][N] × S →T[N][×][N] is a transition function._ 

Usually, we have the objective to learn the transition function _δ_ for an unknown state transition function _σ_ . The state transition function _δ_ can be approximated as _δ_ ( _T, s_ ) := _I_ ( _σ[′]_ ( _T, s_ ) _, Xi, Xj_ ) = _τ_ ( _φ_ ( _s_ ) _i, φj ◦ σ[′]_ ( _T, s_ )). 

As standard causal relations emerge from the underlying mediation process, the meta-causal states emerge from different types of causal effects. The transition conditions of the finite-state machine are the configurations of the environment where the quality of some environmental dynamics represented by a type _t ∈T_ changes. 

**Inferring Meta-Causal States.** Even if the state transition function is known, it may be unclear from a single observation which exact meta-causal state led to the generation of a particular observation _S_ . This is especially the case when two different meta-causal states can fit similar environmental dynamics. Even in the presence of latent factors (e.g., an agent’s internal policy), the current dynamics of a system (e.g., induced by the agent’s current policy) can sometimes be inferred from a series of observed environment states. This requires knowledge of the meta-causal dynamics, and is subject to the condition that sequences of observed states uniquely characterize the meta-causal state. Since MCM are defined as finite state machines, the exact condition for identifying the metacausal state is that observed sequences are homing sequences (Moore et al., 1956). Note that the following example of a game of tag presented below exactly satisfies this condition, where the metacausal states produce disjunct sets of environment states (‘agent A faces agent B’, or ‘agent B faces agent A’), and thus can be inferred from either a single observation, when movement directions are observed, or two observations, when they need to be inferred from the change in position of two successive observations. 

**‘Game of Tag’ Example.** Consider an idealized game of tag between two agents, with a simple causal graph and two different meta-states. In general, we expect agent _B_ to make arbitrary moves that increase its distance to _A_ , while _A_ tries to catch up to _B_ , or vice versa. In essence, this is a cyclic causal relationship between the agents, where both states have the same binary adjacency matrix. Note, however, that the two states differ in the _type_ of relationship that goes from _A_ to _B_ (and _B_ to _A_ ). We can use an identification function that analyzes the current behavior of the agents to identify each edge. Since _A_ can tag _B_ , the behavior of the system changes when the directions of the typed arrows are reversed, so that the type of the edges is either ‘escaping’ or ‘chasing’. 

While the underlying policy of an agent may not be apparent from observation as an endogenous variable, it can be inferred by observing the agents’ actions over time. Knowing the rules of the game, one can assume that the encircling agent faces the other agent and thus moves towards it, while the fleeing agents show the opposite behavior: 

( _tA→B_ = chasing) _⇐⇒_ ( _A_[˙] _pos_ _**·**_ ( _Bpos − Apos_ ) _>_ 0) 

where _A_[˙] _pos_ is the velocity vector of agent _A_ (possibly computed from the position of two consecutive time steps); _Apos_ and _Bpos_ are the agent positions, and _**·**_ is the dot product. Once the edge types are known, the policy can be identified immediately. 

**Causal and Anti-Causal Considerations of Meta-Causal States.** Assigning meta-causal states to particular system observations can be understood as labeling the individual observations. However, it is generally unclear whether the meta-causal state has an observational or a controlling character 

4 

Systems with Switching Causal Relations: A Meta-Causal Perspective 

on the system under consideration. One could ask the question whether the system dynamics cause the meta-causal state, whether the meta-causal state causes the system dynamics, (or whether they are actually the same concept), similar to the well-known discussion “On Causal and Anticausal Learning” by Sch¨olkopf et al. (2012), but from a meta-causal perspective. 

At this point in time, we cannot give definitive conditions on how to answer this question, but we present two examples that support either one of two opposing views. First, in Sec. 4.3 we present a scenario of a dynamical system where the structural equations of both meta-causal states are the same, since the system dynamics are governed by its self-referential system dynamics. Intervening on the meta-causal state will therefore have no effect on the underlying structural equations, and thus can have no effect on the actual system dynamics. In such scenarios, the meta-causal is rather a descriptive label and cannot be considered an external conditioning factor. In the following, we discuss the opposite example, where a meta-causal state can be modeled as an external variable conditioning the structural equations. 

**Role of Contextual Independencies.** Changes in the causal graph are often times attributed to changes in the environment, as for example prominently leveraged in invariant causal prediction of Peters et al. (2016); Heinze-Deml et al. (2018). In general, the effects of a changing causal graph can be attributed to the change in some external factor acting on the system. Under certain circumstances, we can indeed represent meta-causal models via standard SCM conditioned on some external factor _Z ∈Z_ . Suppose there is a surjective function _ψ_ : _Z →T[N][×][N]_ that can map from the _Z_ to every meta causal state T. As a result, one can define a new set of structural equations _fi[′]_[:=] _[ ⃝][i][∈]_[1] _[..N][f] ij[ ′]_[and] 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_Systems_with_Switching_Causal_Relations_A_Meta-Causal_Perspective_images/iclr_2025_Systems_with_Switching_Causal_Relations_A_Meta-Causal_Perspective.pdf-0005-04.png)


where _fij[ψ]_[(] _[z]_[)] are the structural equations of the edge _eij_ that are active under the MCS _T_ = _ψ_ ( _z_ ) and (in a slight imprecision in the actual definition) ( _< ∗_ ) is the function that carries on the previous function of the concatenation and discards _Xj_ . While the ‘no-edge’ type _0_ could be handled like any other type, we have listed it for clarity such that individual variables **X** _i_ become _contextually independent_ whenever _ψ_ ( _Z_ ) _ij_ = _0_ . 

## 4 APPLICATIONS 

In this section, we discuss several applications of the meta-causal formalism. First, we revisit the motivational example and consider how meta-causal models can be used to attribute responsibility. Next, we identify the presence of multiple mechanisms for the bivariate case from sets of unlabeled data. Finally, we analyze the emergence of meta-causal states from a dynamical system, highlighting that meta-causal states can be more than simply conditioning the adjacency matrix of graphs on a latent variable. The code used in the experiments is available at https://anonymous.4open. science/r/metaCausal-CRL/. 

## 4.1 ATTRIBUTING RESPONSIBILITY 

Consider again the motivational example of Figure 1, where an agent _A_ with position _AX_ follows an agent _B_ with position _BX_ as dictated by its policy _A_ _**π**_ . Imagine a counterfactual scenario in which we replace the ‘following’ policy of agent _A_ with, e.g., a ‘standing still’ policy and find that the _BX → AX_ edge vanishes. As a result, we infer the meta-causal mechanism _A_ _**π** →_ ( _BX → AX_ ) for the system and thus _A_ _**π**_ as the root cause of for values of _AX_ . In conclusion, we find that, while _A_ is conditioned on _B_ on a low level, the meta-causal reason for the existence of the edge _B → A_ is caused by the _A_ _**π**_ . Both attributions, either tracing back causes through the structural equations _AX_ := _f_ ( _BX_ ) or our meta-causal approach, are valid conclusions in their own regard. 

Note that in this scenario, a classic counterfactual consideration, _A[A] X_ _**[π]**_[=][standing still] _− A[A] X_ _**[π]**_[=][following] , would also infer an effect of _A_ _**π**_ on _AX_ using a purely value-based perspective. Attributing effects via observed changes in variable values is a valid approach, but fails to explain preventative mechanisms. For example, consider the simple scenario where two locks prevent a door from opening. Classical counterfactual analysis would attribute zero effect to the opening of either lock. A 

5 

Systems with Switching Causal Relations: A Meta-Causal Perspective 

|_k∗_<br>_k′_|_d_= 0_._0<br>_d_= 0_._1<br>_d_= 0_._2<br>-<br>1<br>2<br>3<br>4<br>-<br>1<br>2<br>3<br>4<br>-<br>1<br>2<br>3<br>4|_d_= 0_._0<br>_d_= 0_._1<br>_d_= 0_._2<br>-<br>1<br>2<br>3<br>4<br>-<br>1<br>2<br>3<br>4<br>-<br>1<br>2<br>3<br>4|_d_= 0_._0<br>_d_= 0_._1<br>_d_= 0_._2<br>-<br>1<br>2<br>3<br>4<br>-<br>1<br>2<br>3<br>4<br>-<br>1<br>2<br>3<br>4|
|---|---|---|---|
|1|2<br>81<br>3<br>7<br>7|2<br>85<br>3<br>7<br>3|1<br>83<br>3<br>8<br>5|
|2|41<br>1<br>54<br>4<br>0|43<br>1<br>48<br>8<br>0|49<br>1<br>47<br>3<br>0|
|3|68<br>0<br>4<br>22<br>6|63<br>0<br>2<br>30<br>5|77<br>2<br>0<br>13<br>8|
|4|92<br>0<br>0<br>1<br>7|89<br>0<br>0<br>2<br>9|87<br>0<br>1<br>5<br>7|



Table 1: **Confusion Matrices for Identifying Meta-Causal Mechanisms.** The table shows identification results for predicting the number of mechanisms for the bivariate case for 100 randomly sampled meta-causal mechanisms. _d_ is the maximum sample deviation from the average mechanism probability. Rows indicate the true number of mechanisms, while columns indicate the algorithms’ predictions. ‘-’ indicates setups where the algorithm did not make a decision. In general, the algorithm is rather conservative in its predictions. In all cases where a decision is made, the number of correct predictions along the diagonals dominate. The first and second most frequent predictions are marked in green and orange, respectively. _(Best Viewed in Color)_ 

meta-causal perspective would reveal that the opening of the first lock already changes the metacausal state of the model by removing its causal mechanisms that conditions the state of the door. In a sense, our meta-causal causal perspective is similar to that of the actual causality framework (Halpern, 2016; Chockler & Halpern, 2004). However, actual causality operates at the ‘actual’, i.e. value-based, level and does not take the mechanistic meta-causal view into perspective. Meta-causal models already allow us to reason about causal effects after opening the first lock due to a change in the causal graph, without us having to consider the effects of opening the second lock at some point to reach that conclusion. 

## 4.2 DISCOVERING META-CAUSAL STATES IN THE BIVARIATE CASE 

Our goal in this experiment is to recover the number of meta-causal states _K ∈_ [1 _.._ 4] from data that exists between two variables _X, Y_ that are directly connected by a linear equation with added noise. We assume that each meta-causal state gives rise to a different linear equation _fk_ := _αkX_ + _βk_ + _N , k ∈_ N, where _αk, βk_ are the slope and intercept of the respective mechanism and _N_ is a zerocentered, symmetric, and quasiconvex noise distribution[2] . Without loss of generality, we assume Laplacian noise, for which an L1-regression can estimate the true parameters of the linear equations (Hoyer et al., 2008). The causal direction of the mechanism is randomly chosen between different meta-causal states. The exact sampling parameters and plots of the resulting distributions are given in Sec. C and Fig. 3 in the appendix, respectively. In general, this scenario corresponds to the setting described above of inferring the values of a latent variable _Z_ (with _K_ = _|Z|_ ) that indicate a particular meta-causal state of the system. Our goal is to recover the number and parameterizations of the causal mechanisms, and as a consequence to be able to reason about which points were generated by which mechanism. 

**Approach.** The problem we are trying to solve is twofold : first, we are initially unaware of the underlying meta-causal state _t_ that generated a particular data point ( _xi, yi_ ), which prevents us from estimating the parameterization ( _αk, βk_ ) of the mechanism. Conversely, our lack of knowledge about the mechanism parameterizations ( _αk, βk_ ) _k∈_ [1 _..K_ ] prevents us from assigning class probabilities to the individual data points. Since neither the state assignment nor the mechanism parameterizations are initially known, we perform an Expectation-Maximization (EM; Dempster et al. (1977)) procedure to iteratively estimate and assign the observed data points to the discovered MCS parameterizations. Due to the local convergence properties of the EM algorithm, we further embed it into a locally optimized local random sample consensus approach (LO-RANSAC; Fischler & Bolles (1981); Chum et al. (2003)). RANSAC approaches repeatedly sample initial parameter configurations to avoid local minima, and successively perform several steps of local optimization - here the EM algorithm - to regress the true parameters of the mechanism. 

> 2Implying unimodality and monotonic decreasing from zero allows us to distinguish the noise mean and intercept and to recover the parameters from a simple linear regression 

6 

Systems with Switching Causal Relations: A Meta-Causal Perspective 

||**1 Mechanism**<br>**2 Mechanisms**<br>**3 Mechanisms**<br>**4 Mechanisms**|
|---|---|
|_Max. Class Deviation = 0.0_||
|Theoretical<br>Empirical|1<br>23<br>363<br>8,179<br>2<br>8<br>24<br>173|
|_Max. Class Deviation = 0.1_||
|Theoretical<br>Empirical|1<br>26<br>429<br>10,659<br>2<br>8<br>25<br>177|
|_Max. Class Deviation = 0.2_||
|Theoretical<br>Empirical|1<br>30<br>526<br>14,859<br>2<br>8<br>26<br>177|



Table 2: **Estimated Number of Required Resamples for Obtaining a 95% Convergence Rate with the LO-RANSAC Algorithm per Number of Mechanisms and Maximum Class Deviation.** The empirical observed convergence rates of the EM algorithm drastically reduce the required number of samples of the theoretical derived bound. This reveals that our made assumptions where chosen to be quite conservative, and attests a good fit of the EM algorithm for regressing the mechanisms’ parameters. 

Assuming for the moment that the correct number of mechanisms _k[∗]_ has been chosen, we assume that the EM algorithm is able to regress the parameters of the mechanisms, _αk[∗][, β] k[∗]_[, whenever there] exists a pair of points for each of the mechanisms, where both points of the pair are samples generated by that particular mechanism. The chances of sampling such an initial configuration decrease rapidly with an increasing number of mechanisms (e.g. 0 _._ 036% probability for _k_ = 4 and equal class probabilities). Furthermore, we assume that the sampling probabilities of the individual mechanisms in the data can deviate from the mean by up to a certain factor _d_ . In our experiments, we consider setups with _d ∈{_ 0 _._ 0 _,_ 0 _._ 1 _,_ 0 _._ 2 _}_ . Given the number of classes and the maximum sample deviation of the mechanisms, one can compute an upper bound on the number of resamples required to have a 95% chance of drawing at least one valid initialization. The bound is maximized by assigning the first half of the classes the maximum deviation probability _P_ k-max = (1 + _d_ ) _/k_ and the other half the minimum deviation probability _P_ k-min = (1 _− d_ ) _/k_ . We provide the formulas for the upper bound estimation and in Sec. A in the Appendix and provide the calculated required resample counts in Table 2. In the worst case, for a scenario with _k_ = 4 mechanisms and _d_ = 0 _._ 2 maximum class probability deviation, nearly 14 _,_ 900 restarts of the EM algorithm are required, drastically increasing the potential runtime. 

In our experiments, we find that our assumptions about EM convergence are rather conservative. Our evaluations show that the EM algorithm is still likely to be able to regress the true parameters, given that some of the initial points are sampled from incorrect mechanisms. We measure the empirical convergence rate by measuring the convergence rate of the EM algorithm over 5,000 different setups (500 randomly generated setups with 10 parameter resamples each). We perform 5 EM steps for setups with _k_ = 1 and _k_ = 2 mechanisms, and increase to 10 EM iterations for 3 and 4 mechanisms. For each initialization, we count the EM algorithm as converged if the slope and intercept of the true and predicted values do not differ by more than an absolute value of 0 _._ 2. 

**Determining the Number of Mechanisms.** The above approach is able to regress the true parameterization of mechanisms when the real number of parameters _k[∗]_ is given. However, it is still unclear how to determine the correct _k[∗]_ . Computing the parameters for all _k ∈ K_ and comparing for the best goodness of fit is generally a bad indicator for choosing the right _k_ , since fitting more mechanisms usually captures additional noise and thus reduces the error. In our case, we take advantage of the fact that we assumed the noise to be Laplacian distributed. Thus, the residuals of the samples assigned to a particular mechanism can be tested against the Laplacian distribution. 

The EM algorithms return the estimated parameters _αk[′][, β] k[′]_[, and the mean standard deviation] _[ b][′] k_[of] the Laplacian[3] . This allows one to compute the class densities _k_ ( _xi, yi_ ; _αk[′][, β] k[′][, b][′] k_[)][for][a][pair][of] values ( _xi, yi_ ). Since the assignment of mechanisms to a data point may be ambiguous due to the overlap in the estimated PDFs, we normalize the density values of all mechanisms per data point 

> 3Since mechanisms can go in both directions, _X → Y_ and _Y → X_ , we repeat the regression for both directions and use an Anderson-Darling test (Anderson & Darling, 1952) on the residual to test which of the distributions more closely resembles a Laplacian distribution at each step. 

7 

Systems with Switching Causal Relations: A Meta-Causal Perspective 

and consider only those points for which the probability of the dominant mechanism is 0 _._ 4 higher than the second class: ( **x** _[j] ,_ **y** _[j]_ ) := (( _xi, yi_ ) _|_ #1 = _j_ ; _fi_[#2] _< Pi_[#1] _×_ (1 _−_ 0 _._ 4)), where # _n_ indicates the class with the n-th highest density value. Finally, the residuals **y** _[j] − fk[′][j]_[(] **[x]** _[j][,]_ **[ y]** _[j]_[;] _[ α] k[′][, β] k[′][, b][′] k_[)][ are] computed for all data points ( **x** _[j] ,_ **x** _[j]_ ) assigned to a particular predicted mechanism _fk[′][j]_[.][We choose] the parameterization that best fits the data for each _k ∈_ [1 _.._ 4] and, make use of the Anderson-Darling test (Anderson & Darling, 1952) to test the empirical distribution function of the residuals against the Laplacian distribution with an _α_ = 0 _._ 95 using significance values estimated by Chen (2002). If all residual distributions of all mechanisms for a given _k[′]_ pass the Anderson-Darling test, we choose that number as our predicted number of mechanisms. If the algorithm finds that none of the _k ∈_ [1 _.._ 4] setups pass the test, we refrain from making a decision. We provide the pseudo code for our method in Algorithm. 1 in the Appendix. 

**Evaluation and Results.** We evaluate our approach over all _k ∈_ [1 _.._ 4] by generating 100 different datasets for every particular number of mechanisms. For every data set we sample 500 data points from each mechanism ( _x[k] i[, y] i[k]_[) =] _[ f][k]_[(] _[α][k][x][k] i_[+] _[β][k]_[ +] _[l][i]_[)][ where] _[ l][i][∼L]_[(0] _[, b][k]_[)][, using the same sampling] method as before (c.f. Appendix C). Finally, we let the algorithm recover the number of mechanisms that generated the data. We compare theoretically computed and empirical in convergence results in Table 2. In practice, we observe that the convergence of the EM algorithm is more favorable than estimated, reducing from 23 to 8 required examples for the simple case of _k_ = 2 _, d_ = 0 _._ 0, and requiring up to 83-times fewer samples for the most challenging setup of _k_ = 4 _, d_ = 0 _._ 2, reducing from a theoretical of 14,859 to an empirical estimate of 177 samples. The actual convergence probabilities and the formula for deriving to sample counts are given in Table 3 and Sec. B of the Appendix. 

Table 1 shows the confusion matrices between the actual number of mechanism and the predicted number for different values of maximum class imbalances. In general, we find that our approach is rather conservative when in assigning a number of mechanisms. However, when only considering the cases where the number of mechanisms is assigned, the correct predictions along the main diagonal dominate with over 60% accuracy for _k_ = 4 and _d ∈{_ 0 _._ 0 _,_ 0 _._ 1 _}_ and rising above 80% for _k_ = 4 for. In the case of _d_ = 0 _._ 2, the results indicate higher confusion rates with 41 _._ 6% accuracy for the overall worst case of _K_ = 4. 

**Extension to Meta-Causal State Discovery on Graphs.** Our results indicate that identifying metacausal mechanisms even in the bivariate case comes with an increasing number of uncertainty when it comes to increasing numbers of mechanisms. Given that the number of mechanisms could be reliably inferred from data for all variables, the meta-causal state of a graph could be identified as all unique combinations of mechanisms that are jointly active at a certain point in time. To recover the full set of meta-causal states, one needs to be able to simultaneously estimate the triple of active parents for every mechanism, the mechanisms parameterizations, and the resulting metastate assignment of all data points for every meta-state of the system. All of the three components can vary between each meta-causal state (e.g. edges vanishing or possibly switching direction, thus altering the parent sets). Note that whenever any of the three components is known, the problem becomes rather trivial. However, without making any additional assumptions on the model or data and given the results on the already challenging task of identifying mechanisms in the bivariate case, the extension to a unsupervised full-fledged meta-causal state discovery is not obvious to us at the time of writing and we leave it to future work to come up with a feasible algorithm. 

## 4.3 A META-STATE ANALYSIS ON STRESS-INDUCED FATIGUE 

Stress is a major cause of fatigue and can lead to other long-term health problems (Maisel et al., 2021; Franklin et al., 2012; Dimsdale, 2008; Bremner, 2006). While short-term exposure may be helpful in enhancing individual performance, long-term stress is detrimental and resilience may decline over time (Wang & Saudino, 2011; Maisel et al., 2021). While actual and perceived stress levels (Cohen et al., 1983; Bremner, 2006; Schlotz et al., 2011; Giannakakis et al., 2019) may vary between individuals (Calkins, 1994; Haggerty, 1996), the overall effect remains the same. We want to model such a system as an example. For simplicity, we present an idealized system that is radically reduced to the only factors of external stress, modeling everyday environmental factors, and the selfinfluencing level of internal/perceived stress. 

8 

Systems with Switching Causal Relations: A Meta-Causal Perspective 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_Systems_with_Switching_Causal_Relations_A_Meta-Causal_Perspective_images/iclr_2025_Systems_with_Switching_Causal_Relations_A_Meta-Causal_Perspective.pdf-0009-01.png)


**----- Start of picture text -----**<br>
1 . 0<br>1.0<br>0 . 9 0.9<br>0 . 8 0.8 E S<br>0 . 7 0.7<br>0 . 6 0.6<br>0 . 5 0.5<br>00 .. 43 0.40.3 E S<br>0.2<br>0 . 2 0.1<br>0 . 1 0.0<br>0 . 1 0 . 2 0 . 3 0 . 4 0 . 5 0 . 6 0 . 7 0 . 8 0 . 9 1 . 0 0.0 0.2 0.4 0.6 0.8 1.0 E S<br>Stress Levelt-1 External Stressors<br>Levelt<br>Stress Level<br>Stress<br>**----- End of picture text -----**<br>


Figure 2: **Mechanistic Decomposition as Meta-Causal States** . **(left)** Effect of the stress level on itself (orange) plotted against the identity (blue). Once a threshold of 0 _._ 5 is reached, the function switches behavior from self-suppressing self-reinforcing. **(center)** Contribution of the stress level mechanism for varying external stressors. Red arrows indicate a self-reinforcing effect, while green arrows indicate suppression. The gray area indicates the system without external stressors. Although all states are governed by the same set of structural equations, our meta-causal analysis identifies mechanistic differences and assigns different behavioral states to the different starting conditions. **(right)** The standard SCM gets decomposed into different meta-causal states. While the graph adjacency remains the same, the identification function detect the different modes of the causal mechanism. _(Best viewed in color.)_ 

We use an identification function to distinguish between two different modes of operation of a causal mechanism. In particular, we are interested whether the inherent dynamics of the internal stress level are self-reinforcing or self-suppressing. For easier analysis of the system we decompose the dynamics of the internal stress variable into a ‘decayed stress’ _d_ and ‘resulting stress’ _s_ computation. The first term are the previous stress levels decayed over time with external factors added. The resulting stress is then the output of a Sigmoidal function (Fig. 2, left) that either reinforces or suppresses the value (Fig. 2,center). The structural equations are defined as follows and we assume that all values lie in the interval of [0 _,_ 1]: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_Systems_with_Switching_Causal_Relations_A_Meta-Causal_Perspective_images/iclr_2025_Systems_with_Switching_Causal_Relations_A_Meta-Causal_Perspective.pdf-0009-04.png)


where _s_ and _d_ are the resulting and decayed stress levels, and _s[′]_ is the previous stress level of _s_ . 

We now define the identification function to be _I_ := sign( _f_[¨] _s_ ), where _f_[¨] _s_ is the second order derivative of the Sigmoidal _fs_ , with either positive or negative effect on the stress level. The described system has two stable modes of dynamics. Note how the second-order inflection point at 0.5 of the Sigmoid acts as a transition point on the behavior of the mechanism. Stress values below 0.5 get suppressed, while values above 0.5 are amplified. Transitions between the two stable states can only be initiated via external stressors. Effectively this results in three possible meta-causal states which are governed via the following transition function: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_Systems_with_Switching_Causal_Relations_A_Meta-Causal_Perspective_images/iclr_2025_Systems_with_Switching_Causal_Relations_A_Meta-Causal_Perspective.pdf-0009-07.png)


While being rather simple in setup, this example serves as a good demonstration on how dynamical systems induce meta-causal states that exhibit qualitative different behavior although being governed by the same set of underlying structural equations. 

**Role of Latent Conditioning.** A key takeaway in this example is that the current meta-causal state persists due to the current stress levels and system dynamics and not due to a chance in the structural equations. As a consequence, the system can not solely be expressed via some external conditioning factors. In contrast to other examples where system dynamics where purely due to the meta-causal state, here the variable values themselves play a vital role in the overall system dynamics. As a result, the stressed state of a person would persist even when the initiating external stressors disappear. Intervening on the meta-causal state of the system is now ill-defined, as both positive and negative reinforcing effects are governed by the same equation. Thus, creating a disparity between the intervened meta-causal state and the systems’ identified functional behavior. 

9 

Systems with Switching Causal Relations: A Meta-Causal Perspective 

## 5 RELATED WORK 

**Causal Transportability and Reinforcement Learning.** Meta-causal models cover cases that reduce to conditionally dependent causal graphs due to changing environments (Peters et al., 2016; Heinze-Deml et al., 2018), but also extend beyond that for dynamical systems. In this sense, the work of Talon et al. (2024) takes a meta-causal view by transporting edges of different causal effects between environments. In general, the transportability of causal relations (Zhang & Bareinboim, 2017; Lee & Bareinboim, 2018; Correa et al., 2022) can be thought of as learning identification functions that have identified the general conditions of the underlying processes to transfer certain types of causal effects between environments Sonar et al. (2021); Yu et al. (2024); Brouillard et al. (2022). This has been studied to some extent under the name of meta-reinforcement learning, which attempts to predict causal relations from the observations of environments Sauter et al. (2023); Dasgupta et al. (2019). Generally, transferability has been considered in reinforcement learning, where the efficient use of data is omnipresent and causality provides guarantees regarding the transferability of the observed mechanisms under changing environments (Sæmundsson et al., 2018; Dasgupta et al., 2019; Zeng et al., 2023). 

**Large Language Models (LLMs).** Meta-causal representations are an important consideration for LLMs and other foundation models, since these models are typically limited to learning world dynamics from purely observational textual descriptions. In order to reason in a truly causal manner, LLMs need to learn meta-causal models (sometimes referred to as ‘world models’) that allow them to simulate the implied consequences of interventions (Lampinen et al., 2024; Li et al., 2021; 2022). To the best of our knowledge, Zeˇcevi´c et al. (2023) made the first attempt to define explicit meta-causal models that integrate with the Pearlian causal framework. However, their proposed models are defined purely as adjacency matrix memorization, such that causal reasoning in LLMs is equivalent to knowledge recall: ( _X → Y_ ) _⇐⇒_ ( _eXY ∈_ Mem). Thus, LLMs would not learn the underlying mediation process and would fail to predict the effects of novel interventions on the causal graph. 

**Actual Causality, Attribution and Cognition.** Our MCM framework can also be used to infer and attribute responsibility (as shown in Sec. 4.1) and therefore touches on the topics of fairness, actual causation (AC) and work on counterfactual reasoning in cognitive science. (Von K¨ugelgen et al., 2022; Karimi et al., 2021; Halpern, 2000; 2016; Chockler & Halpern, 2004; Gerstenberg et al., 2014; Gerstenberg, 2024). The similarities extend to how MCM encourage reasoning about the dependence between actual environmental contingencies and qualitative types of causal mechanisms. In this sense, MCM allow for the direct characterization of actual causes of system dynamics types. A rigorous formalization is left open for future work. While AC in combination with classical SCM only describes relationships between causal variable configurations and observable events, there are cases where we can take an MCM and derive an SCM that encodes the meta-causal types with instrumental variables, thus allowing a similar meta-causal analysis within the AC framework. 

## 6 CONCLUSION 

We formally introduced meta-causal models that are able to capture the dynamics of switching causal types of causal graphs and, in many cases, can better express the qualitative behavior of a causal system under consideration. Within MCM typing mechanisms generalize the notion of specific structural equations and abstract away unnecessary details. We presented a first, motivating example of how a classical causal analysis and a meta-causal perspective can attribute different root causes to the same situation. We extended these claims by considering that MCM are still able to capture and attribute changes in the mechanistic behavior of the system, even though no actual change is apparent. As an application, we presented a first approach to recover metacausal states in the bivariate case. Although our experimental results represent only a first preliminary approach, we find that meta-causal models are a powerful tool for modeling, reasoning, and inferring system dynamics. Finally, we have shown how MCM are able to express more information about a dynamical system than a simple conditioning variable in a standard SCM can do. 

**Limitations and Future Work.** While we have formally introduced meta-causal models, there remain several practical open directions to pursue. First, we have been able to provide examples that illustrate the differences between standard causal, and meta-causal attribution. In particular, the 

10 

Systems with Switching Causal Relations: A Meta-Causal Perspective 

joint application of standard and meta-causal explainability will allow for the joint consideration of actual and mechanistic in future attribution methods. Second, our approach to recover metacausal states from unlabeled data is open to extension. Discovery on the full causal graphs is a desirable goal that, however, needs further work to reliably control for the changing sets of parents under changing MCS. Finally, we have made a first attempt to present examples for and against the controlling or observational role of meta-causal models. While the observational perspective is mainly of interest for analytical applications, the application of meta-causal models to agent systems and reinforcement learning opens up a new field of application. 

## ACKNOWLEDGMENTS 

The TU Darmstadt authors acknowledge the support of the German Science Foundation (DFG) project “Causality, Argumentation, and Machine Learning” (CAML2, KE 1686/3-2) of the SPP 1999 “Robust Argumentation Machines” (RATIO). This work is supported by the Hessian Ministry of Higher Education, Research, Science and the Arts (HMWK; projects “The Third Wave of AI”). It was funded by the European Union. Views and opinions expressed are, however, those of the author(s) only and do not necessarily reflect those of the European Union or the European Health and Digital Executive Agency (HaDEA). Neither the European Union nor the granting authority can be held responsible for them. Grant Agreement no. 101120763 - TANGO. This work was supported from the National High-Performance Computing project for Computational Engineering Sciences (NHR4CES). 

The Eindhoven University of Technology authors received support from their Department of Mathematics and Computer Science and the Eindhoven Artificial Intelligence Systems Institute. 

## REFERENCES 

- Tara V Anand, Ad`ele H Ribeiro, Jin Tian, and Elias Bareinboim. Effect identification in cluster causal diagrams. _arXiv preprint arXiv:2202.12263_ , 2022. 

- Theodore W Anderson and Donald A Darling. Asymptotic theory of certain” goodness of fit” criteria based on stochastic processes. _The annals of mathematical statistics_ , pp. 193–212, 1952. 

- Argyris Arnellos, Thomas Spyrou, and John Darzentas. Dynamic interactions in artificial environments: causal and non-causal aspects for the emergence of meaning. _Systemics, Cybernetics and Informatics_ , 3:82–89, 2006. 

- Sander Beckers and Joseph Y Halpern. Abstracting causal models. In _Proceedings of the aaai conference on artificial intelligence_ , volume 33, pp. 2678–2685, 2019. 

- Richard Bellman. A markovian decision process. _Journal of mathematics and mechanics_ , pp. 679– 684, 1957. 

- J Douglas Bremner. Traumatic stress: effects on the brain. _Dialogues in clinical neuroscience_ , 8(4): 445–461, 2006. 

- Philippe Brouillard, Perouz Taslakian, Alexandre Lacoste, S´ebastien Lachapelle, and Alexandre Drouin. Typing assumptions improve identification in causal discovery. In _Conference on Causal Learning and Reasoning_ , pp. 162–177. PMLR, 2022. 

- SD Calkins. Origins and outcomes of individual differences in emotion regulation. _The Development of Emotion Regulation: Biological and Behavioral Considerations, Monographs of the Society for Research in Child Development_ , 59(2-3), 1994. 

- Colin Chen. Tests for the goodness-of-fit of the laplace distribution. _Communications in StatisticsSimulation and Computation_ , 31(1):159–174, 2002. 

- Hana Chockler and Joseph Y Halpern. Responsibility and blame: A structural-model approach. _Journal of Artificial Intelligence Research_ , 22:93–115, 2004. 

- Ondˇrej Chum, Jiˇr´ı Matas, and Josef Kittler. Locally optimized ransac. In _Pattern Recognition: 25th DAGM Symposium, Magdeburg, Germany, September 10-12, 2003. Proceedings 25_ , pp. 236–243. Springer, 2003. 

11 

Systems with Switching Causal Relations: A Meta-Causal Perspective 

- Sheldon Cohen, Tom Kamarck, and Robin Mermelstein. A global measure of perceived stress. _Journal of health and social behavior_ , pp. 385–396, 1983. 

Taco Cohen. Towards a grounded theory of causation for embodied ai. _arXiv preprint arXiv:2206.13973_ , 2022. 

- Juan D Correa, Sanghack Lee, and Elias Bareinboim. Counterfactual transportability: a formal approach. In _International Conference on Machine Learning_ , pp. 4370–4390. PMLR, 2022. 

- Ishita Dasgupta, Jane Wang, Silvia Chiappa, Jovana Mitrovic, Pedro Ortega, David Raposo, Edward Hughes, Peter Battaglia, Matthew Botvinick, and Zeb Kurth-Nelson. Causal reasoning from meta-reinforcement learning. _arXiv preprint arXiv:1901.08162_ , 2019. 

- Arthur P Dempster, Nan M Laird, and Donald B Rubin. Maximum likelihood from incomplete data via the em algorithm. _Journal of the royal statistical society: series B (methodological)_ , 39(1): 1–22, 1977. 

- Joel E Dimsdale. Psychological stress and cardiovascular disease. _Journal of the American College of Cardiology_ , 51(13):1237–1246, 2008. 

- Martin A Fischler and Robert C Bolles. Random sample consensus: a paradigm for model fitting with applications to image analysis and automated cartography. _Communications of the ACM_ , 24 (6):381–395, 1981. 

- Tamara B Franklin, Bechara J Saab, and Isabelle M Mansuy. Neural mechanisms of stress resilience and vulnerability. _Neuron_ , 75(5):747–761, 2012. 

- Tobias Gerstenberg. What would have happened? counterfactuals, hypotheticals and causal judgements. _Philosophical Transactions of the Royal Society B_ , 377(1866):20210339, 2022. 

- Tobias Gerstenberg. Counterfactual simulation in causal cognition. _Trends in Cognitive Sciences_ , 2024. 

- Tobias Gerstenberg, Noah Goodman, David Lagnado, and Josh Tenenbaum. From counterfactual simulation to causal judgment. In _Proceedings of the annual meeting of the cognitive science society_ , volume 36, 2014. 

- Giorgos Giannakakis, Dimitris Grigoriadis, Katerina Giannakaki, Olympia Simantiraki, Alexandros Roniotis, and Manolis Tsiknakis. Review on psychological stress detection using biosignals. _IEEE transactions on affective computing_ , 13(1):440–460, 2019. 

- Robert J Haggerty. _Stress, risk, and resilience in children and adolescents: Processes, mechanisms, and interventions_ . Cambridge University Press, 1996. 

- Joseph Y Halpern. Axiomatizing causal reasoning. _Journal of Artificial Intelligence Research_ , 12: 317–337, 2000. 

Joseph Y Halpern. _Actual causality_ . MiT Press, 2016. 

- Christina Heinze-Deml, Jonas Peters, and Nicolai Meinshausen. Invariant causal prediction for nonlinear models. _Journal of Causal Inference_ , 6(2):20170016, 2018. 

- Stefan G Hofmann, Joshua E Curtiss, and Steven C Hayes. Beyond linear mediation: Toward a dynamic network approach to study treatment processes. _Clinical Psychology Review_ , 76:101824, 2020. 

- Patrik Hoyer, Dominik Janzing, Joris M Mooij, Jonas Peters, and Bernhard Sch¨olkopf. Nonlinear causal discovery with additive noise models. _Advances in neural information processing systems_ , 21, 2008. 

- Amir-Hossein Karimi, Bernhard Sch¨olkopf, and Isabel Valera. Algorithmic recourse: from counterfactual explanations to interventions. In _Proceedings of the 2021 ACM conference on fairness, accountability, and transparency_ , pp. 353–362, 2021. 

12 

Systems with Switching Causal Relations: A Meta-Causal Perspective 

- Armin Keki´c, Bernhard Sch¨olkopf, and Michel Besserve. Targeted reduction of causal models. In _ICLR 2024 Workshop on AI4DifferentialEquations In Science_ , 2023. 

- Andrew Lampinen, Stephanie Chan, Ishita Dasgupta, Andrew Nam, and Jane Wang. Passive learning of active causal strategies in agents and language models. _Advances in Neural Information Processing Systems_ , 36, 2024. 

- Sanghack Lee and Elias Bareinboim. Structural causal bandits: Where to intervene? _Advances in neural information processing systems_ , 31, 2018. 

- Belinda Z Li, Maxwell Nye, and Jacob Andreas. Implicit representations of meaning in neural language models. _arXiv preprint arXiv:2106.00737_ , 2021. 

- Kenneth Li, Aspen K Hopkins, David Bau, Fernanda Vi´egas, Hanspeter Pfister, and Martin Wattenberg. Emergent world representations: Exploring a sequence model trained on a synthetic task. _arXiv preprint arXiv:2210.13382_ , 2022. 

- Peter Maisel, Erika Baum, and Norbert Donner-Banzhoff. Fatigue as the chief complaint: epidemiology, causes, diagnosis, and treatment. _Deutsches Arzteblatt[¨] International_ , 118(33-34): 566, 2021. 

- Edward F Moore et al. Gedanken-experiments on sequential machines. _Automata studies_ , 34:129– 153, 1956. 

Judea Pearl. _Causality_ . Cambridge university press, 2009. 

- Jonas Peters, Peter B¨uhlmann, and Nicolai Meinshausen. Causal inference by using invariant prediction: identification and confidence intervals. _Journal of the Royal Statistical Society Series B: Statistical Methodology_ , 78(5):947–1012, 2016. 

- Paul K Rubenstein, Sebastian Weichwald, Stephan Bongers, Joris M Mooij, Dominik Janzing, Moritz Grosse-Wentrup, and Bernhard Sch¨olkopf. Causal consistency of structural equation models. _arXiv preprint arXiv:1707.00819_ , 2017. 

- Steind´or Sæmundsson, Katja Hofmann, and Marc Peter Deisenroth. Meta reinforcement learning with latent variable gaussian processes. _arXiv preprint arXiv:1803.07551_ , 2018. 

- Andreas WM Sauter, Erman Acar, and Vincent Francois-Lavet. A meta-reinforcement learning algorithm for causal discovery. In _Conference on Causal Learning and Reasoning_ , pp. 602–619. PMLR, 2023. 

- Wolff Schlotz, Ilona S Yim, Peggy M Zoccola, Lars Jansen, and Peter Schulz. The perceived stress reactivity scale: measurement invariance, stability, and validity in three countries. _Psychological assessment_ , 23(1):80, 2011. 

- Bernhard Sch¨olkopf, Dominik Janzing, Jonas Peters, Eleni Sgouritsa, Kun Zhang, and Joris Mooij. On causal and anticausal learning. _arXiv preprint arXiv:1206.6471_ , 2012. 

- Steven A Sloman, Philip M Fernbach, and Scott Ewing. Causal models: The representational infrastructure for moral judgment. _Psychology of learning and motivation_ , 50:1–26, 2009. 

- Anoopkumar Sonar, Vincent Pacelli, and Anirudha Majumdar. Invariant policy optimization: Towards stronger generalization in reinforcement learning. In _Learning for Dynamics and Control_ , pp. 21–33. PMLR, 2021. 

- Peter Spirtes, Clark N Glymour, Richard Scheines, and David Heckerman. _Causation, prediction, and search_ . MIT press, 2000. 

- Davide Talon, Phillip Lippe, Stuart James, Alessio Del Bue, and Sara Magliacane. Towards the reusability and compositionality of causal representations. _arXiv preprint arXiv:2403.09830_ , 2024. 

- Julius Von K¨ugelgen, Amir-Hossein Karimi, Umang Bhatt, Isabel Valera, Adrian Weller, and Bernhard Sch¨olkopf. On the fairness of causal algorithmic recourse. In _Proceedings of the AAAI conference on artificial intelligence_ , volume 36, pp. 9584–9594, 2022. 

13 

Systems with Switching Causal Relations: A Meta-Causal Perspective 

Jonas Wahl, Urmi Ninad, and Jakob Runge. Foundations of causal discovery on groups of variables. _arXiv preprint arXiv:2306.07047_ , 2023. 

- Clare R Walsh and Steven A Sloman. The meaning of cause and prevent: The role of causal mechanism. _Mind & Language_ , 26(1):21–52, 2011. 

- Manjie Wang and Kimberly J Saudino. Emotion regulation and stress. _Journal of Adult Development_ , 18:95–103, 2011. 

- Moritz Willig, Matej Zeˇcevi´c, Devendra Singh Dhami, and Kristian Kersting. Do not marginalize mechanisms, rather consolidate! In _Proceedings of the 37th Conference on Neural Information Processing Systems (NeurIPS)_ , 2023. 

- Phillip Wolff. Representing causation. _Journal of experimental psychology: General_ , 136(1):82, 2007. 

- Zhongwei Yu, Jingqing Ruan, and Dengpeng Xing. Learning causal dynamics models in objectoriented environments. _arXiv preprint arXiv:2405.12615_ , 2024. 

- Matej Zeˇcevi´c, Moritz Willig, Devendra Singh Dhami, and Kristian Kersting. Causal parrots: Large language models may talk causality but are not causal. _Transactions on Machine Learning Research_ , 2023. 

Yan Zeng, Ruichu Cai, Fuchun Sun, Libo Huang, and Zhifeng Hao. A survey on causal reinforcement learning. _arXiv preprint arXiv:2302.05209_ , 2023. 

- Junzhe Zhang and Elias Bareinboim. Transfer learning in multi-armed bandit: a causal approach. In _Proceedings of the 16th Conference on Autonomous Agents and MultiAgent Systems_ , pp. 1778– 1780, 2017. 

14 

Systems with Switching Causal Relations: A Meta-Causal Perspective 

APPENDIX FOR “SYSTEMS WITH SWITCHING CAUSAL RELATIONS: A META-CAUSAL PERSPECTIVE” 

The appendix is structured as follows: 

In Sec. A we derive the formula for the theoretical upper bound of the 95% confidence interval. Sec B discusses the sample statistics and convergence results of the applied LO-RANSAC algorithm. Finally, Sec. C discusses the exact parameters of the bivariate meta-causal mechanism data sampling. 

## A PROBABILITIES FOR SAMPLE COMPUTATION AND UPPER BOUNDS 

Consider a dataset _D ∈_ **R** _[m][×]_[2] of _m_ samples over two variables where we want to separate _n_ different functions. We assume that the data distribution contains a uniform number of samples from each function, where each class could be under- or overrepresented by an offset of d. Specifically, we assume that each function is represented by (1 _±_ d) _n_[1] _[|D|]_[ samples, i.e., the probability of encountering] a particular class _X_ is[1] _[−] n_[d] _≤ P_ ( _X_ ) _≤_[1+d] _n_ and E[ _P_ ( _X_ )] = _n_[1][.][To identify all functions between] these two variables, we assume linearity and apply EM with RANSAC on random pairs of samples (see Section 4.2). By selecting _n_ pairs, there is a chance that one pair is chosen from each function (we will refer to such a set of pairs as a “correct” set of samples). In this section, we derive the probability of a correct pair being chosen at random, so that we can estimate how many times pairs need to be sampled to reliably encounter a correct set of samples. 

We denote _S_ as the event of “correctly” sampling all _n_ pairs from all _n_ different functions. If all classes have the same number of samples, the chance of randomly selecting a pair from a new class is _[n]_[1] _[n][−]_[1] _·_[1][1][1] _n[·] n_[for the first pair of samples,] _n n_[for the second,] _[ . . .]_[ , and] _n[·] n_[for the last; in short:] 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_Systems_with_Switching_Causal_Relations_A_Meta-Causal_Perspective_images/iclr_2025_Systems_with_Switching_Causal_Relations_A_Meta-Causal_Perspective.pdf-0015-07.png)


If the data distribution is not perfectly uniform, i.e., d = 0, we can also calculate a lower bound for the same probability. Consider two probabilities per sample: the **probability of selecting a new class** _Pn_ ( _S_[new] ) and the **probability of selecting a second sample of the same class** _Pn_ ( _S_[same] ) afterwards. Across all samples, these correspond to _n[n][n]_[!][and][ (] _n_[1][)] _[n]_[in][ E][[] _[P][n]_[(] _[S]_[)]][, respectively.] 

Let us first consider the **probability of selecting a new class** . When the first sample is taken, only one new class can be selected (probability of 1). If this sample was taken from the largest class first, the probability that subsequent samples will be taken from new classes decreases, since the space of “unsampled” classes is smaller. For this lower bound, we therefore assume that maximally large classes are sampled from as much and as early as possible. According to our assumptions, the largest classes each take up a fraction of[1+d] _n_ of the data. Therefore, the probability of selecting a new class for successive samples has the following probabilities _[n] n[,][n][−]_[(] _n_[1][+][d][)] _,[n][−]_[(][1] _n_[+][d][)][2] _, . . ._ . First, consider the case where _n_ is even. Here, after all the _[n]_ 2[largest][classes][have][been][selected,][only] the small classes remain. For the last, second to last, _. . ._ classes, this probability is represented by (1 _−n_ d) _,_[2][(][1] _n[−]_[d][)] _, . . ._ . Overall, we get the probability 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_Systems_with_Switching_Causal_Relations_A_Meta-Causal_Perspective_images/iclr_2025_Systems_with_Switching_Causal_Relations_A_Meta-Causal_Perspective.pdf-0015-10.png)


If _n_ is uneven, an average size class between the largest and smallest classes must be included: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_Systems_with_Switching_Causal_Relations_A_Meta-Causal_Perspective_images/iclr_2025_Systems_with_Switching_Causal_Relations_A_Meta-Causal_Perspective.pdf-0015-12.png)


The **probability of selecting a second sample of the same class** is easier to calculate. Instead of constant probabilities as in the expectation with _n_[1][,][we][now][have][two][different][probabilities][in][the] even case and three in the odd case. In the even case, we have _[n]_ 2[large batches and the same number] 

15 

Systems with Switching Causal Relations: A Meta-Causal Perspective 

of small batches, so the probability of choosing the right batch each time is 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_Systems_with_Switching_Causal_Relations_A_Meta-Causal_Perspective_images/iclr_2025_Systems_with_Switching_Causal_Relations_A_Meta-Causal_Perspective.pdf-0016-02.png)


In the uneven case, we also have to also consider the batch that has an average size 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_Systems_with_Switching_Causal_Relations_A_Meta-Causal_Perspective_images/iclr_2025_Systems_with_Switching_Causal_Relations_A_Meta-Causal_Perspective.pdf-0016-04.png)


Note that for both _Pn_ ( _S_ new) and _Pn_ ( _S_ same), the distribution of the data into the largest and smallest possible batches (according to our assumptions) results in the smallest possible probabilities; hence, the computed probability is a lower bound. If the batches were more evenly sized, the probability would be larger. We can also see that a deviation of up to 1 results in a probability of 0, since it is impossible to sample from a class that is not represented in the data. 

In total, the probability of selecting of a correct sample set is the product of the two probabilities above, i.e, 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_Systems_with_Switching_Causal_Relations_A_Meta-Causal_Perspective_images/iclr_2025_Systems_with_Switching_Causal_Relations_A_Meta-Causal_Perspective.pdf-0016-07.png)


_Pn_ ( _S_ ) is the (lower bound on the) probability of selecting a correct set of samples. We can calculate the number of trials _k_ needed to find such a set of samples with at least 95% probability. The opposite probability, of never finding it with less than 5% probability, is easier to calculate: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_Systems_with_Switching_Causal_Relations_A_Meta-Causal_Perspective_images/iclr_2025_Systems_with_Switching_Causal_Relations_A_Meta-Causal_Perspective.pdf-0016-09.png)


This allows us to determine how many attempts might be necessary. Note that while this would leave a 5% chance of not picking the right samples, there are various practical reasons why the actual probability of finding a working set of samples will be higher, e.g., if the number of samples from each class is not as uneven as assumed, or if some samples are distributed in such a way that even picking a sample from the “wrong” class might still lead to the identification of the correct mechanisms. 

For example, if _n_ = 2 and d = 0 _._ 2, we have an expected probability of E[ _P_ ( _S_ )] = 22![2] _[·]_[2][= 0] _[.]_[125][ and] a lower bound of _P_ 2( _S_ ) = _P_ ( _S_ even[new][)] _[P]_[(] _[S]_ even[same][)] _[ ≥]_[0] 2 _[.]_[8] _[·]_[2] 2 _[·]_[1] 2 _[.]_[2] _[·]_[0] 2 _[.]_[8][= 0] _[.]_[096][. Larger deviations decrease] the probability while a deviation of d = 0 results in the same probability as with E[ _P_ ( _S_ )]. For the lower bound, this results in _k_ = 30 samples for a confidence of 95% using the above calculation steps. All resulting sample counts can be found in Table 2. 

## B EM CONVERGENCE RESULTS 

The required number of resamples for a 95% success rate of the RANSAC algorithm is calculated by log(0 _._ 05) _/_ log(1 _− C_[1] ), where _S_ 1 are the convergence rates for the individual samples computed in Sec. A. 

Empirical convergence probabilities and resulting resampling counts for the LO-RANSAC algorithm are shown in Tables 2 and 3. Table 4 lists the goodness of fit for all converged samples. In general, we find that in cases where the approach is able to converge, it undercuts the required parameter convergence boundary of 0 _._ 2 by factors of 4 _._ 8 and 3 _._ 5 for the slope and intercept, respectively. 

## C MECHANISM SAMPLING 

For our experiments in Sec. 4.2 we uniformly sample the number of mechanisms to be in _K ∈ {_ 1 _.._ 4 _}_ . The slopes of the linear equations are uniformly sampled between _α ∈±_ [0 _._ 2 _.._ 5] and the 

16 

Systems with Switching Causal Relations: A Meta-Causal Perspective 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_Systems_with_Switching_Causal_Relations_A_Meta-Causal_Perspective_images/iclr_2025_Systems_with_Switching_Causal_Relations_A_Meta-Causal_Perspective.pdf-0017-01.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_Systems_with_Switching_Causal_Relations_A_Meta-Causal_Perspective_images/iclr_2025_Systems_with_Switching_Causal_Relations_A_Meta-Causal_Perspective.pdf-0017-02.png)


**----- Start of picture text -----**<br>
K=1<br>K=2<br>K=3<br>K=4<br>**----- End of picture text -----**<br>



![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_Systems_with_Switching_Causal_Relations_A_Meta-Causal_Perspective_images/iclr_2025_Systems_with_Switching_Causal_Relations_A_Meta-Causal_Perspective.pdf-0017-03.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_Systems_with_Switching_Causal_Relations_A_Meta-Causal_Perspective_images/iclr_2025_Systems_with_Switching_Causal_Relations_A_Meta-Causal_Perspective.pdf-0017-04.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_Systems_with_Switching_Causal_Relations_A_Meta-Causal_Perspective_images/iclr_2025_Systems_with_Switching_Causal_Relations_A_Meta-Causal_Perspective.pdf-0017-05.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_Systems_with_Switching_Causal_Relations_A_Meta-Causal_Perspective_images/iclr_2025_Systems_with_Switching_Causal_Relations_A_Meta-Causal_Perspective.pdf-0017-06.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_Systems_with_Switching_Causal_Relations_A_Meta-Causal_Perspective_images/iclr_2025_Systems_with_Switching_Causal_Relations_A_Meta-Causal_Perspective.pdf-0017-07.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_Systems_with_Switching_Causal_Relations_A_Meta-Causal_Perspective_images/iclr_2025_Systems_with_Switching_Causal_Relations_A_Meta-Causal_Perspective.pdf-0017-08.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_Systems_with_Switching_Causal_Relations_A_Meta-Causal_Perspective_images/iclr_2025_Systems_with_Switching_Causal_Relations_A_Meta-Causal_Perspective.pdf-0017-09.png)


Figure 3: **Sampled Mechanisms.** The figure shows a selection of different randomly sampled mechanism distributions, ranging from one to up to four simultaneously present mechanisms. The gray dotted lines represent the generating ground truth mechanisms. _(Best Viewed In Color)_ 

intercepts are in the range _β ∈_ [ _−_ 5 _,_ 5]. We add Laplacian noise _L_ ( _x|µ, b_ ) = 21 _b[∗]_[exp(] _[−][|][x][−] b[µ][|]_ ) with _µ_ = 0 and _b ∈_ [0 _._ 1 _,_ 4 _._ 0]. _X_ values are uniformly sampled in the range [ _−_ 5 _,_ 5] and _yi_ = _αxi_ + _β_ + _L_ ( _x|_ 0 _, b_ ). The average number of samples per class is set to 500. Throughout the experiments, we vary the class probabilities by a class deviation factor _D ∈{_ 0 _._ 0 _,_ 0 _._ 1 _,_ 0 _._ 2 _}_ . Specifically, we maximize the class deviation by assigning _K/_ 2 classes the maximum probability 1 _/K ∗_ (1 + _D_ ) and _K/_ 2 classes the minimum probability 1 _/K ∗_ (1 _−D_ ). If _K_ is odd, a class is assigned the average probability 1 _/K_ . We show a selection of the resulting sample distributions in Fig. 3. 

17 

Systems with Switching Causal Relations: A Meta-Causal Perspective 

|**Class Deviation**|**1 Mechanism**<br>**2 Mechanisms**<br>**3 Mechanisms**<br>**4 Mechanisms**|
|---|---|
|0.0<br>0.1<br>0.2|4219 (84_._38%)<br>1740 (34_._80%)<br>592 (11_._84%)<br>86 (1_._72%)<br>"<br>1702 (34_._04%)<br>577 (11_._54%)<br>84 (1_._68%)<br>"<br>1567 (31_._34%)<br>555 (11_._10%)<br>84 (1_._68%)|



Table 3: **Empirical estimated convergence percentage of the EM algorithm for different class imbalances and number of mechanisms.** The table shows the number of samples converged and the convergence rates (in parentheses) for a single random initialization and for estimating the parameterization of the underlying system for a given number of mechanisms. All results are reported per 5000 samples. 

|**Class Deviation**|**1 Mechanism**<br>**2 Mechanisms**<br>**3 Mechanisms**<br>**4 Mechanisms**|
|---|---|
|0.0<br>0.1<br>0.2|0_._0349_|_0_._0590<br>0_._0370_|_0_._0543<br>0_._0380_|_0_._0592<br>0_._0389_|_0_._0516<br>"<br>0_._0381_|_0_._0555<br>0_._0381_|_0_._0566<br>0_._0346_|_0_._0528<br>"<br>0_._0414_|_0_._0548<br>0_._0412_|_0_._0567<br>0_._0402_|_0_._0570|



Table 4: **Mean average error for the slope and intercept of the correctly predicted mechanism for different class imbalances and number of mechanisms.** Mechanisms are accepted if their parameters do not differ by more than 0 _._ 2 from the ground truth parameterization. The results show that converged samples are typically estimated with an error well below the threshold. 

18 

Systems with Switching Causal Relations: A Meta-Causal Perspective 

**Algorithm 1** Recovering Mechanisms for the Bivariate Case 

1: **procedure** RECOVERMECHANISMS( **x** _,_ **y** , maxClassDev _K_ max _,_ EMSteps) 2: **for all** _k ∈_ [1 _..K_ max] **do** 3: bestModelLogL _←−∞_ 4: N _←_ requiredSamples( _k,_ maxClassDev _,_ 0 _._ 95) _▷_ Compute # of samples (c.f. Sec. A) 5: **for all** _n ∈_ [1 _.._ N] **do** _▷_ RANSAC iteration. 6: **px** _,_ **py** _←_ sample( _xi, yi,_ 2 _∗ k_ ) _▷_ Initialize parameters with 2 _× k_ points. 7: **for all** _k[′] ∈_ [0 _..k_ ) **do** 8: _αk ←_ ( _py_ 2 _k_ +1 _− py_ 2 _k_ ) _/_ ( _px_ 2 _k_ +1 _− px_ 2 _k_ ) 9: _βk ← py_ 2 _k − αk ∗ x_ 2 _k_ 10: _bk ←_ 1 _._ 0 _▷_ Assume initial avg. deviation of the Laplacion to be 1. 11: _dk ←_ ‘XY’ _▷_ Assume _X → Y_ direction first. 12: **c** _k ← P_ Laplacian( **x** _,_ **y** ; _α, β, b, d_ ) _▷_ Initial class probabilities for all samples. 13: **end for** 14: **for all** _l ∈_ [1 _.._ EMSteps] **do** _▷_ EM Iteration. 15: _**α** ,_ _**β** ,_ **b** _,_ **d** _←_ regressLines( **x** _,_ **y** ; **c** ) _▷_ (Weighted) median regression. 16: **c** _← P_ Laplacian( **x** _,_ **y** ; _**α** ,_ _**β** ,_ **b** _,_ **d** ) 17: **end for** 18: modelLogL _←_[�] _i_[LogP] Laplacian[(] **[x]** _[,]_ **[ y]**[;] _**[ α]**[,]_ _**[ β]**[,]_ **[ b]** _[,]_ **[ d]**[)] _[i] ▷_ Obtain the joint log probs. 19: **if** modelLogL _>_ bestModelLogL **then** 20: bestModelLogL _←_ modelLogL 21: bestParameters _←_ ( _**α** ,_ _**β** ,_ **b** _,_ **d** ) 22: **end if** 23: **end for** 24: allLaplacian _←_ true _▷_ Check if residuals are Laplacian distributed. 25: **for all** _l ∈_ [1 _..k_ ] **do** 26: ( _**α** ,_ _**β** ,_ **b** _,_ **d** ) _←_ bestParameters 27: **x** _[l] ,_ **y** _[l] ,_ **c** _[l] ←_ selectClass( **x** _,_ **y** _,_ **c** _, l_ ) _▷_ Select points where argmax **c** = _l_ . 28: **x** _[l] ,_ **y** _[l] ←_ filter( **x** _[l] ,_ **y** _[l] ,_ **c** _[l] ,_ 0 _._ 4) _▷_ Keep points s.t. max[#2] **c** _[l] <_ 0 _._ 4(1 _−_ max[#1] ( **c** _[l]_ ). 29: _r_ = LinEq( **x** _[l] ,_ **y** _[l]_ ; _αl, βl, dl_ ) _−_ **y** _[l]_ 30: **if not** AndersonDarling( _r,_ 0 _._ 95) **then** 31: allLaplacian _←_ false 32: **break** 33: **end if** 34: **end for** 35: **if** allLaplacian **then** 36: **return** k 37: **end if** 38: **end for** 39: **return** 0 40: **end procedure** 

19 

