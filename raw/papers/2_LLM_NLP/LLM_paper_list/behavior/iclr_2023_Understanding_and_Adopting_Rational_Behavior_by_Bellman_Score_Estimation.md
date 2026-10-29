Published as a conference paper at ICLR 2023 

## - UNDERSTANDING AND ADOPTING RATIONAL BEHAV IOR BY BELLMAN SCORE ESTIMATION 

## **Kuno Kim** 

## **Stefano Ermon** 

Department of Computer Science Department of Computer Science Stanford University Stanford University Stanford, CA 94305 Stanford, CA 94305 khkim@cs.stanford.edu ermon@cs.stanford.edu 

## ABSTRACT 

We are interested in solving a class of problems that seek to understand and adopt rational behavior from demonstrations. We may broadly classify these problems into four categories of reward identification, counterfactual analysis, behavior imitation, and behavior transfer. In this work, we make a key observation that knowing how changes in the underlying rewards affect the optimal behavior allows one to solve a variety of aforementioned problems. To a local approximation, this quantity is precisely captured by what we term the _Bellman score_ , i.e the gradient of the log probabilities of the optimal policy with respect to the reward. We introduce the Bellman score operator which provably converges to the gradient of the infinite-horizon optimal _Q_ -values with respect to the reward which can then be used to directly estimate the score. Guided by our theory, we derive a practical _score-learning_ algorithm which can be used for score estimation in high-dimensional state-actions spaces. We show that score-learning can be used to reliably identify rewards, perform counterfactual predictions, achieve state-of-theart behavior imitation, and transfer policies across environments. 

## 1 INTRODUCTION 

A hallmark of intelligence is the ability to achieve goals with rational behavior. For sequential decision making problems, rational behavior is often formalized as a policy that is optimal with respect to a Markov Decision Process (MDP). In other words, intelligent agents are postulated to learn rational behavior for reaching goals by maximizing the reward of an underlying MDP (Doya, 2007; Neftci & Averbeck, 2019; Niv, 2009). The reward covers information about the goal while the remainder of the MDP characterizes the interplay between the agent’s decisions and the environment. 

In this work, we are interested in solving a class of problems that seek to understand and adopt rational (i.e optimal) behavior from demonstrations of sequential decisions. We may broadly classify them into categories of reward identification, counterfactual analysis, behavior imitation, and behavior transfer. (see Appendix D for detailed definitions) These four problems of understanding and adopting rationality arise across a wide spectrum of science fields. For example, in econometrics, the field of Dynamic Discrete Choice (DDC) seeks algorithms for fitting reward functions to human decision making behavior observed in labor (Keane & Wolpin, 1994) or financial markets (Arcidiacono & Miller, 2011). The identified utilities are leveraged to garner deeper insight into people’s decision strategies (Arcidiacono & Miller, 2020), make counterfactual predictions about how their choices will change in response to market interventions that alter the reward function (Keane & Wolpin, 1997), and train machine learning models that can make rational decisions like humans (Kalouptsidi et al., 2015). In animal psychology and neuroscience, practitioners model the decision strategies of animals by fitting reward models to their observed behavior. The fitted reward is analyzed and then used to train AI models that simulate animal movements (Yamaguchi et al., 2018; Schafer et al., 2022) in order to gain a better understanding of ecologically and evolutionarily significant phenomena such as habitat selection and migration (Hirakawa et al., 2018). In robot learning, Inverse Reinforcement Learning (IRL) is used to infer reward functions from teleoperated robots and the learned rewards are fed through an RL algorithm to teach a robot how to perform the same task without human controls (Fu et al., 2018; Finn et al., 2016b; Chan & van der Schaar, 2021). As rewards for a task are often invariant to perturbations in the environment, (e.g walking can be described by the same reward function which encourages forward velocity and stability regardless of whether the agent is walking 

1 

Published as a conference paper at ICLR 2023 

on ice or grass) the same task can be learned in various environmental conditions by optimizing the same inferred reward (Fu et al., 2018; Zhang et al., 2018). 

We make a key observation that knowing how changes in the underlying rewards affect the optimal behavior (policy) allows one to solve a variety of aforementioned problems. To a local approximation, this quantity is precisely captured by what we term the _Bellman score_ , i.e _gradient of log probabilities of the optimal policy with respect to the reward_ . Prior works which study related quantities to the Bellman score are scarce and relied on having full-knowledge of the environment dynamics or small discrete state-action spaces (Neu & Szepesvári, 2012; Vroman, 2014; Li et al., 2017). We introduce - the Bellman score operator that provably converges to the gradient of the infinite-horizon optimal _Q_ values with respect to the reward. This _Q_ -gradient can then be used to estimate the score. We further show that the _Q_ -gradient is equivalent to the conditional state-action visitation counts under the optimal policy. With these results, we derive the gradient of the Maximum Entropy IRL (Ziebart et al., 2008; Finn et al., 2016b; Fu et al., 2018) objective in the general setting with stochastic dynamics case and non-linear reward models. Guided by theory, we propose a powerful score-learning algorithm that can be used for model-free score estimation in continuous high-dimensional state-action spaces and an effective IRL algorithm, named Gradient Actor-Critic (GAC). Our experiments demonstrate that score-learning can be used to reliably identify rewards, make counterfactual predictions, imitate behaviors, and transfer policies across environments. 

## 2 PRELIMINARIES 

A Markov Decision Process (MDP) _M 2_ ⌦, where ⌦ is the set of all MDPs, is a tuple _M_ = ( _X , A, P, P_ 0 _, r✓, γ_ ) where _X_ is the discrete[1] state space, _A_ is the discrete action (decision) space, _P 2_ R _[|X⇥A|⇥|X|]_ is the transition probability matrix, _P_ 0 _2_ R _[|X|]_ is the initial state distribution, _r✓ 2_ R _[|X⇥A|]_ is the (stationary) parametric reward with parameters _✓ 2_ ⇥, and _γ 2_ [0 _,_ 1] is the discount factor. ⇥ is the parameter space, typically set to a finite-dimensional real vector space R[dim(] _[✓]_[)] . We will use _T ≥_ 0 to denote the time horizon for finite-horizon problems. A domain _d_ is an MDP without the reward _M \ r_ . Moving forward we will alternate between vector and function notation, e.g _P_ ( _x[0] |x, a_ ) _, r✓_ ( _x, a_ ) denote the value of vector _r✓_ at the dimension for the state-action ( _x, a_ ) and the value of matrix _P_ at the location for (( _x, a_ ) _, x[0]_ ). Furthermore, one-dimensional vectors are treated as row vectors, e.g R _[|X⇥A|]_ = R[1] _[⇥|X⇥A|]_ . A (stationary) policy is a vector _⇡ 2_ R _[|X⇥A|]_ that represents distributions over actions, i.e[P] _a2A[⇡]_[(] _[a][|][x]_[) = 1][ for all] _[ x]_[.][A non-stationary policy for] time horizon _T_ is a sequence of policies _⇡_ ˜ = ( _⇡t_ ) _[T] t_ =0 _[2]_[ (][R] _[|X⇥A|]_[)] _[T]_[ +1][where] _[ ⇡][t]_[is the policy used] when there are _t_ environment steps remaining[2] and _⇡_ ˜: _k_ = ( _⇡t_ ) _[k] t_ =0[for] _[ k][][T]_[denotes a subsequence.] When there is no confusion we will use _⇡_ to denote to both stationary and non-stationary policies. Next, _P⇡ 2_ R _[|X⇥A|⇥|X⇥A|]_ denotes the transition matrix of the Markov chain on state-action pairs induced by policy _⇡_ , i.e _P⇡_ ( _x[0] , a[0] |x, a_ ) = _P_ ( _x[0] |x, a_ ) _⇡_ ( _a[0] |x[0]_ ). Furthermore, _P⇡[n]_[denotes powers of] the transition matrix with _P⇡_[0][=] _[I]_[where] _[I][2]_[R] _[|X⇥A|⇥|X⇥A|]_[is][the][identity.][For][non-stationary] policies, we define _P⇡_ ˜ _[n]_[=] _[ P][⇡] T −_ 1 _[P][⇡] T −_ 2 _[...P][⇡] T −n_[for] _[ n][ ≥]_[1][ and] _[ P]_[ 0] _⇡_ ˜[=] _[ I]_[.] 

Let _x,a[2]_[ R] _[|X⇥A|]_[denote the indicator vector which has value][ 1][ at the dimension corresponding to] ( _x, a_ ) and 0 elsewhere. The conditional marginal distribution for both stationary and non-stationary policy _⇡_ after _n_ environment steps is _p⇡,n_ ( _·|x, a_ ) = _x,a[P] ⇡[ n][2]_[ R] _[|X⇥A|]_[.][The (unnormalized)] _[ t]_[-step][3] conditional occupancy measure of _⇡_ is the discounted sum of conditional marginals: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/behavior/iclr_2023_Understanding_and_Adopting_Rational_Behavior_by_Bellman_Score_Estimation_images/iclr_2023_Understanding_and_Adopting_Rational_Behavior_by_Bellman_Score_Estimation.pdf-0002-06.png)


Intuitively, _⇢⇡,t_ ( _x[0] , a[0] |x, a_ ) quantifies the visitation frequency for ( _x[0] , a[0]_ ) when an agent starts from ( _x, a_ ) and runs _⇡_ for _t_ environment steps, with more weight on earlier visits. The infinite horizon conditional occupancy exists for _γ <_ 1 and we will denote it by simply omitting the subscript _t_ , i.e _⇢⇡_ ( _·|x, a_ ) = lim _t!1 ⇢⇡,t_ ( _·|x, a_ ). The (unconditional) occupancy measure can be recovered by _⇢⇡,t_ ( _x[0] , a[0]_ ) =[P] _x,a[P]_[0][(] _[x]_[)] _[⇡][T]_[ (] _[a][|][x]_[)] _[⇢][⇡,t]_[(] _[x][0][, a][0][|][x, a]_[)][.][The] _[ t]_[-step] _[ Q]_[-values] _[ Q][⇡,t][2]_[R] _[|X⇥A|]_[for both] stationary and non-stationary policy _⇡_ are defined as _Q⇡,t_ ( _x, a_ ) = E _x0,a0⇠⇢⇡,t_ ( _·|x,a_ )[ _r✓_ ( _x[0] , a[0]_ )] = _⇢⇡,t_ ( _·|x, a_ ) _·r_ , i.e the conditional expectation of discounted reward sums when there are _t_ environment 

> 1We use discrete spaces for notational brevity later on, but our results can be extended to continuous spaces 

> 2Backwards indexing is standard in dynamic programming RL algorithms (Bertsekas & Tsitsiklis, 1995) 

> 3 _Step_ refers to the number of environment transitions and not the number of decisions made 

2 

Published as a conference paper at ICLR 2023 

steps remaining. The theory in this section will be derived in the Maximum Entropy RL (MaxEntRL) setting (Haarnoja et al., 2017; Yarats et al., 2021). Let us define the _t_ -step discounted conditional entropy of a policy as _H⇡,t_ ( _x, a_ ) = _−_[P] _[t] n_ =1 _[γ][n][p][⇡,n]_[(] _[·|][x, a]_[)] _[ ·]_[ log] _[ ⇡][T][ −][n]_[.][In][MaxEntRL,][the][soft] _Q_ -values are defined as _Q_[soft] _⇡,t_[(] _[x, a]_[)][=] _[Q][⇡,t]_[(] _[x, a]_[) +] _[ H][⇡,t]_[(] _[x, a]_[)][,][i.e the conditional expectation of] discounted reward and policy entropy sums. 

For reward _r✓_ , let _Q_[soft] _✓,t[2]_[ R] _[|X⇥A|]_[denote the] _[ t]_[-step optimal] _[ Q]_[-values defined iteratively by the soft] Bellman optimality operator _TH_ : R _[|X⇥A|] !_ R _[|X⇥A|]_ as 


![](raw/papers/2_LLM_NLP/LLM_paper_list/behavior/iclr_2023_Understanding_and_Adopting_Rational_Behavior_by_Bellman_Score_Estimation_images/iclr_2023_Understanding_and_Adopting_Rational_Behavior_by_Bellman_Score_Estimation.pdf-0003-03.png)


for all _t >_ 0 and _Q_[soft] _r,_ 0[=] _[r]_[.][Hence][the][optimal] _[Q]_[-values][for][a][decision][problem][with][horizon] _t_ can be obtained by sequentially computing the optimal _Q_ -values for 1 _, ..., t −_ 1 step decision problems using dynamic programming (Bellman, 1957), i.e value iteration. The operator _TH_ is a contraction in the max-norm (Bertsekas & Tsitsiklis, 1995) whose unique fixed point is the infinitehorizon optimal _Q_ -values _Q_[soft] _✓,1_[satisfying] _[ Q]_[soft] _✓,1_[=] _[ T][H][Q]_[soft] _✓,1_[.][For finite horizon problems, the (non-] stationary) optimal policy _⇡✓_[soft] = ( _⇡✓,t_[soft][)] _t[T]_ =0[is derived by] _[ ⇡] ✓,t_[soft][(] _[a][|][x]_[)][=][softmax(] _[Q]_[soft] _✓,t_[)(] _[x, a]_[)][=] _e[Q] ✓,t_[soft][(] _[x,a]_[)] _/_[P] _a[e][Q] ✓,t_[soft][(] _[x,a]_[)] and, similarly, the infinite-horizon (stationary) optimal policy is _⇡✓,_[soft] _1_[=] softmax( _Q_[soft] _✓,1_[)][.][We may now define the key quantity of interest, the] _[ Bellman score]_[.] 

**Definition 1** _The finite-horizon_ _**Bellman score** s_[soft] _✓ 2_ (R _[|X⇥A|⇥]_[dim(] _[✓]_[)] ) _[T]_[ +1] _is the gradient of the log probabilities of the (non-stationary) optimal policy with respect to the reward, s_[soft] _✓_ = ( _s_[soft] _✓,t_[)] _[T] t_ =0[= (] _[r][✓]_[log] _[ ⇡] ✓,t_[soft][)] _[T] t_ =0 _[.] Similarly the infinite-horizon Bellman score s_[soft] _✓,1[2]_[ R] _[|X⇥A|⇥]_[dim(] _[✓]_[)] _[is] s_[soft] _✓,1_[=] _[ r][✓]_[log] _[ ⇡] ✓,_[soft] _1[.]_ 

In words, the finite-horizon score _s_[soft] _✓,t_[is the Jacobian of the log optimal policy vector with respect] to the reward parameters, i.e _s_[soft] _✓,t_[(] _[x, a]_[)][=] _[r][✓]_[log] _[ ⇡] ✓,t_[soft][(] _[a][|][x]_[)][is][the][direction][in][which][the][reward] parameters _✓_ should perturbed in order to increase the log probability of action _a_ at state _x_ under the optimal policy for time _t_ . To a local linear approximation around _✓_ , the score conveys how the optimal behavior changes as a function of the underlying reward parameters. Next, we will prove useful properties of the score as well as derive a dynamic programming algorithm to compute it. 

## 3 SCORE ITERATION 

Since _⇡✓,t_[soft][(] _[a][|][x]_[)][=][softmax(] _[Q]_[soft] _✓,t_[)(] _[x, a]_[)][, the Bellman score can be written as a difference of the] gradients of optimal _Q_ -values with respect to the reward parameters: _s_[soft] _✓,t_[(] _[x, a]_[) =] _[ r][✓]_[log] _[ ⇡] ✓,t_[soft][(] _[a][|][x]_[) =] _[ r][✓][Q]_[soft] _✓,t_[(] _[x, a]_[)] _[ −]_[E] _a[0] ⇠⇡✓,t_[soft][(] _[·|][x]_[)][[] _[r][✓][Q] ✓,t_[soft][(] _[x, a][0]_[)]] _[ .]_ (3) 

We will refer to _r✓Q_[soft] _✓,t_[as the] _[ Q]_[-gradient or value gradient.][Eq.][3 shows that an unbiased sample] estimate of the score can be obtained from the _Q_ -gradient and the optimal policy. Thus, we will derive a dynamic programming algorithm termed _score iteration_ which efficiently computes the _Q_ -gradients. Similar to how value iteration proceeds by repeated application of the Bellman optimality operator (Eq. 2), score iteration will rely on the Bellman score operator which we define now. 

**Definition 2** _The_ _**Bellman score operator** G⇡,✓_ : R _[|X⇥A|⇥]_[dim(] _[✓]_[)] _!_ R _[|X⇥A|⇥]_[dim(] _[✓]_[)] _for a policy ⇡ and reward r✓ is defined on an input J 2_ R _[|X⇥A|⇥]_[dim(] _[✓]_[)] _as_ 


![](raw/papers/2_LLM_NLP/LLM_paper_list/behavior/iclr_2023_Understanding_and_Adopting_Rational_Behavior_by_Bellman_Score_Estimation_images/iclr_2023_Understanding_and_Adopting_Rational_Behavior_by_Bellman_Score_Estimation.pdf-0003-11.png)


To gain intuition about the score operator, consider the tabular reward _r✓_ ( _x, a_ ) = _✓_ ( _x, a_ ) where _✓ 2_ R _[|X⇥A|]_ . In this setting, the score operator simply computes the conditional occupancies for _⇡_ (Eq. 1) when applied to the identity matrix _I_ . For example, the values of _G⇡,✓I_ = _r✓r✓_ + _γP⇡I_ = _I_ + _γP⇡I_ at the row for state-action ( _x, a_ ) is _x,a_[+] _[ γP] ⇡_[(] _[·|][x, a]_[)][ which corresponds to the one-step conditional] occupancy _⇢⇡,_ 1( _·|x, a_ ). While the score operator has a pleasantly simple form, Theorem 1 will show that the _Q_ -gradients can be computed by its repeated application and, as a result, that the _Q_ -gradient is the conditional occupancy measure of the optimal policy for the underlying reward. 

3 

Published as a conference paper at ICLR 2023 

**Algorithm 1** Score Iteration: Comp ynamic programming 

_M_ = ( _X , A, P, P_ 0 _, r✓, γ_ ): Markov Decision Process _g_ : randomly initialized _Q_ -gradient vector **procedure** SCOREITERATION( _M_ ) For _M_ , learn the optimal policy _⇡✓,_[soft] _1_ **while** _g_ is not converged **do for** _x 2 X , a 2 A_ **do** Update _Q_ -gradient: _g_ ( _x, a_ ) _r✓r✓_ ( _x, a_ ) + _γ_[P] _x[0] ,a[0] 2X⇥A[P] ⇡✓,_[soft] _1_[(] _[x][0][, a][0][|][x, a]_[)] _[g]_[(] _[x][0][, a][0]_[)] **for** _x 2 X , a 2 A_ **do** Compute Bellman score: _s_ ( _x, a_ ) _g_ ( _x, a_ ) _−_[P] _a[0] 2A[⇡] ✓,_[soft] _1_[(] _[a][0][|][x]_[)] _[g]_[(] _[x, a][0]_[)] **return** _⇡✓,_[soft] _1[, g, s]_ 

**Theorem 1** _For all t_ = 1 _, ..., T and any matrix J 2_ R _[|X⇥A|⇥]_[dim(] _[✓]_[)] 


![](raw/papers/2_LLM_NLP/LLM_paper_list/behavior/iclr_2023_Understanding_and_Adopting_Rational_Behavior_by_Bellman_Score_Estimation_images/iclr_2023_Understanding_and_Adopting_Rational_Behavior_by_Bellman_Score_Estimation.pdf-0004-04.png)


_where, r✓Q_[soft] _✓,_ 0[=] _[ r][✓][r][✓][.][Furthermore, the][ Q][-gradient satisfies:]_ 


![](raw/papers/2_LLM_NLP/LLM_paper_list/behavior/iclr_2023_Understanding_and_Adopting_Rational_Behavior_by_Bellman_Score_Estimation_images/iclr_2023_Understanding_and_Adopting_Rational_Behavior_by_Bellman_Score_Estimation.pdf-0004-06.png)


Theorem 1 shows that both finite and infinite horizon _Q_ -gradients can be computed by repeatedly applying the score operator of the current reward _r✓_ and its optimal policy _⇡✓_[soft] _, ⇡✓,_[soft] _1_[.][Similar to] how the Bellman optimality operator (Eq. 2) converges any starting vector in R _[|X⇥A|]_ to the infinitehorizon optimal _Q_ -values (Bertsekas & Tsitsiklis, 1995), the Bellman score operator converges any starting matrix in R _[|X⇥A|⇥]_[dim(] _[✓]_[)] to the infinite-horizon _Q_ -gradient. 

Furthermore, we see that the Q-gradient is in fact the expected reward gradient under the conditional occupancy measure. To gain more intuition, once again consider the tabular reward _r✓_ ( _x, a_ ) = _✓_ ( _x, a_ ). Then, _r✓Q_[soft] _✓,t_[(] _[x, a]_[) =][ E] _[x][0][,a][0][⇠][⇢] ⇡✓,_ soft: _t[,t]_[(] _[·|][x,a]_[)][[] _x[0] ,a[0]_[] =] _[ ⇢] ⇡✓,_[soft] : _t[,t]_[.][We see that the] _[ Q]_[-gradient is in fact] equivalent to the conditional occupancy of the optimal policy, which is consistent with our previous analysis of the score operator. Thus, in order to increase the optimal Q-values at ( _x, a_ ), the rewards of state-actions ( _x[0] , a[0]_ ) should be increased proportional to their discounted visitation frequencies when starting from ( _x, a_ ) and following the optimal policy. As an alternative to the operator perspective, one may consider a Linear Programming (LP) interpretation. The optimal _Q_ -values are the solution to an LP optimization problem over the set of conditional occupancies _D_ that are feasible under the Markovian environment dynamics, i.e _Q_[soft] _✓,t_[(] _[x, a]_[)][=][max] _[⇢][2][D][ r][✓][·][ ⇢]_[.][Loosely speaking, invoking] the supremum rule (Boyd et al., 2003) allows one to show that the gradient of the LP solution, i.e _rr✓_ max _⇢2D r✓ · ⇢_ , is in fact the conditional occupancy that solves the LP. 

_The main practical contribution of Theorem 1 is that the operator convergence results enable designing a powerful dynamic programming method for score computation as shown in Algorithm 1._ (Finite-horizon version in Appendix D) The algorithm computes the optimal policy _⇡✓,_[soft] _1_[, then] proceeds by repeatedly applying the score operator _G⇡✓,_ soft _1[,✓]_[.][While highly effective in countable] state-action spaces, score iteration has limited applicability to continuous state-action spaces and requires full knowledge of the environment dynamics. Similar to how _Q_ -learning is a model-free instantiation of the value iteration algorithm, we will proceed to propose score-learning which is a model-free instantiation of the score iteration algorithm. 

## 4 SCALING UP SCORE ESTIMATION 

In order to scale score estimation to the model-free settings with (potentially) high-dimensional continuous state-action spaces, we start by parameterizing the policy, _Q_ -gradient network, and score network as neural networks _⇡φ_ : _X ⇥A !_ R, _g_ : _X ⇥A !_ R[dim(] _[✓]_[)] , and _s!_ : _X ⇥A !_ R[dim(] _[✓]_[)] . Just as _Q_ -learning (Silver et al., 2016) approximates an application of the Bellman optimality operator by learning a _Q_ -network that minimizes a boot-strapped regression loss, score-learning approximates 

4 

Published as a conference paper at ICLR 2023 

## **Algorithm 2** Score-learning: model-free instantiation of SCOREITERATION 

( _✓, φ,  , !_ ): Weights for reward, policy, _Q_ -gradient, and score network _B_ : Buffer of transitions (can be either online or offline data) _N, N_ : Total number of algorithm iterations, number of _Q_ -gradient update steps per policy update _⌘ , ⌘!_ : _Q_ -gradient learning rate, score-learning rate _↵g_ : Target _Q_ -gradient mixing rate RL: One policy update step in an RL algorithm, e.g one policy gradient step in SAC (Haarnoja et al., 2018). **procedure** SCORELEARNING( _✓, φ,  , !_ , _B_ , _N_ ) **for** _i 2 {_ 1 _, . . . , N }_ **do** _# Update policy ⇡φ_ **if** _φ_ is not None **then** Update policy to maximize reward _r✓_ : _φ_ RL( _✓, φ, B_ ) _# Update Q-gradient g_ **for** _j 2 {_ 1 _, . . . , N }_ **do** Sample batch: ( _x, a, x[0]_ ) _⇠B_ **if** is not None **then** Sample next action from current policy: _a[0] ⇠ ⇡φ_ ( _·|x[0]_ ) **else** Sample next action from buffer _a[0] ⇠B_ Update _Q_ -gradient (Eq. 4): _− ⌘ r_ ( _r✓r✓_ ( _x, a_ ) + _γg_ ¯ ( _x[0] , a[0]_ ) _− g_ ( _x, a_ ))[2] Update Target _Q_ -gradient ¯ _g_ with soft mixing rate _↵_ (Haarnoja et al., 2017) _# Update Bellman score s!_ Sample batch and contrastive action: ( _x, a_ ) _⇠B, a[0] ⇠ ⇡φ_ ( _·|x_ ) Update score network (Eq. 5): _! ! − ⌘sr!_ ( _g_ ( _x, a_ ) _− g_ ( _x, a[0]_ ) _− s!_ ( _x, a_ ))[2] **return** _φ,  , !_ 

an application of the score operator by learning a _Q_ -gradient network that minimizes the boot-strapped regression loss in Eq. 4. The score network then minimizes the regression loss in Eq. 5 which directly follows by replacing the _Q_ -gradient with our estimate _g_ in Eq. 3. 


![](raw/papers/2_LLM_NLP/LLM_paper_list/behavior/iclr_2023_Understanding_and_Adopting_Rational_Behavior_by_Bellman_Score_Estimation_images/iclr_2023_Understanding_and_Adopting_Rational_Behavior_by_Bellman_Score_Estimation.pdf-0005-04.png)


Note that _g_ ¯ is a target _Q_ -gradient network used for stabilizing optimization analogous to target _Q_ -networks that are used in _Q_ -learning (Haarnoja et al., 2017). Algorithm 2 shows the full execution flow of score-learning. Instead of sequentially learning each model until convergence, score-learning alternates between making small improvements to the policy (red), regressing to the _Q_ -gradient with the current estimate of the optimal policy (blue), and updating the score (yellow) with the current estimate of the _Q_ -gradient. This enables the algorithm to output approximate estimates of all three components when run for a few-steps, which will be useful for downstream algorithms that use approximate score-learning in the inner-loop. If the buffer _B_ already contains samples from the optimal policy, simply not passing _φ_ disables policy updates. Note that score-learning can operate with both online and offline data _B_ depending on the choice of the RL algorithm used to update the policy. We now show how the score-learning algorithm can be used for a variety of downstream tasks. 

## 4.1 MAXIMUM ENTROPY INVERSE REINFORCEMENT LEARNING WITH SCORE-LEARNING 

In this section we will show how score-learning can be used for maximum entropy inverse reinforcement learning (MaxEntIRL). Let _⌧_ = ( _x_ 0 _, a_ 0 _, ..., xT , aT_ ) denote a trajectory of state-actions, and _p✓_ ( _⌧_ ) = log _P_ 0( _x_ 0)[Q] _[T] t_ =0 _[⇡] ✓,t_[soft][(] _[a][t][|][x][t]_[)][ Q] _[T] t_ =0 _[ −]_[1] _[P]_[(] _[x][t]_[+1] _[|][x][t][, a][t]_[)][ denote the trajectory distribution of the] MaxEnt optimal policy _⇡✓_[soft] for reward _r✓_ and similarly let _p[⇤]_ ( _⌧_ ) be the trajectory distribution of the expert policy _⇡[⇤]_ = ( _⇡t[⇤]_[)] _[T] t_ =0[from which the demonstrations] _[ D]_[ are sampled.][The goal in MaxEntIRL] is to find reward parameters _✓_ that maximize the log-likelihood of the expert trajectories: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/behavior/iclr_2023_Understanding_and_Adopting_Rational_Behavior_by_Bellman_Score_Estimation_images/iclr_2023_Understanding_and_Adopting_Rational_Behavior_by_Bellman_Score_Estimation.pdf-0005-08.png)


The gradient of the objective in Eq. 6 is highly useful as it enables direct gradient ascent to solve the maximization. Previously, the MaxEntIRL gradient has been derived in either the limited setting of deterministic dynamics (Ziebart et al., 2008) or in the general stochastic dynamics setting for the special case of linear reward models (Ziebart et al., 2010). Here, we leverage Theorem 1 to derive the general MaxEntIRL gradient in the stochastic dynamics setting with non-linear rewards. 

5 

Published as a conference paper at ICLR 2023 

**Algorithm 3** Gradient Actor-Critic (GAC ): Inverse RL via _Q_ -gradient Estimation 

( _✓, φ,  L,  E_ ): Weights for reward, policy, learner _Q_ -gradient, and expert _Q_ -gradient network _D_ : Demonstrations of expert behavior _M_ : Total number GAC iterations _N, N✓_ : Number of score iteration steps per GAC iteration, Reward update interval _⌘r_ : Reward learning rate **procedure** GAC( _✓, φ, φ,_[¯] , _D_ ) **for** _i 2 {_ 1 _, . . . , M }_ **do** Take step in environment with _⇡φ_ , add transition ( _x, a, s[0]_ ) to buffer _B_ Update learner _Q_ -gradient and learner policy: _φ,  L_ SCORELEARNING( _✓, φ,  L, B, N_ ) Update expert _Q_ -gradient: _E_ SCORELEARNING( _✓, φ_ =None _,  E, D, N_ ) **if** _i_ mod _N✓_ == 0 **then** Sample expert batch of initial state-actionsSample learner batch of initial state-actions ( _x_ ˜ _x_ 00 _, a⇠B_ 0) _⇠D,_ ˜ _a_ 0 _⇠_ demo _⇡φ_ ( _·|x_ ˜0) Update reward with _Q_ -gradient: _✓ ✓_ + _⌘r_ ( _g_ ( _x_ 0 _, a_ 0) _− g_ (˜ _x_ 0 _,_ ˜ _a_ 0)) **return** _✓, φ_ 

**Theorem 2** _The gradient of the MaxEntIRL objective of Eq. 6 is_ 


![](raw/papers/2_LLM_NLP/LLM_paper_list/behavior/iclr_2023_Understanding_and_Adopting_Rational_Behavior_by_Bellman_Score_Estimation_images/iclr_2023_Understanding_and_Adopting_Rational_Behavior_by_Bellman_Score_Estimation.pdf-0006-04.png)


Theorem 2 shows that the reward should be updated to increase the _Q_ -values of the expert and decrease the optimal _Q_ -values for the learner (Eq. 7) which is equivalent to increasing the reward on state-actions sampled from the expert and decreasing the reward of state-actions sampled from the learner (Eq. 8). With deterministic dynamics, Eq. 8 simply reduces to the known contrastive divergence gradient (Finn et al., 2016b) for energy-based models. When _r✓_ is linear, Eq. 8 reduces to the expected feature gap derived in Ziebart et al. (2010). (see Appendix A for extended discussion) 

Many prior works in MaxEntIRL (Finn et al., 2016b; Fu et al., 2018) take Monte Carlo samples to estimate Eq. 8 and perform stochastic gradient updates on _✓_ . The key challenges with this approach are, first, the difficulty of reusing old samples ( _x, a_ ) _⇠ ⇡✓_[soft] old[from policies for older rewards] _[ ✓]_[old][and,] second, the high variance of the Monte Carlo estimator. _The main practical significance of Theorem 2 comes from Eq. 7 which reveals that we may instead learn Q-gradient networks with score-learning to approximate the expectations in Eq. 8_ . At the cost of introducing bias to the gradient estimates, this approach resolves the two aforementioned challenges; score-learning works with offline data, which enables old sample reuse, and the _Q_ -gradient network outputs the mean reward gradient, which reduces variance. We name our algorithm Gradient Actor-Critic (GAC ) (see Algorithm 3) which uses score-learning to estimate the expert _Q_ -gradient _r✓Q⇡⇤,T_ ( _x, a_ ) and learner _Q_ -gradient _r✓Q✓,T_ ( _x, a_ ) with parametric functions _g E_ and _g L_ . To reduce computational cost, GAC alternates between approximate _Q_ -gradient estimation, i.e running score-learning for _N_ steps, and using the rough gradients for reward updates. Our method is analogous to actor-critic methods (Haarnoja et al., 2017) that learn a _Q_ -network to estimate mean policy returns which are then used for policy gradient updates. These methods similarly introduce bias into the policy gradients in exchange for the ability to reuse old policy data and perform lower variance gradient updates. Just as how actor-critic typically outperforms Monte-Carlo policy gradient methods, our experiments will demonstrate a similar usefulness in trading off bias for lower variance. 

## 4.2 COUNTERFACTUAL PREDICTIONS WITH SCORE-LEARNING 

Here we describe how score-learning can be used for counterfactual predictions. The goal is to predict how the optimal policy changes when the original reward _r✓_ is perturbed to a counterfactual reward _r✓0_ . With access to an efficient environment simulator, one natural solution is to simply resolve the RL problem _r✓0_ . However, this is difficult in many domains where solving the RL problem is expensive, particularly when we wish to make many counterfacutal predictions for multiple reward alterations. Instead, we may use the score _s_[soft] _✓,1_[for the original reward] _[ r][✓]_[to estimate of the optimal policy at the] new reward _r✓0_ by simply using the first order taylor approximation of the log optimal policy: 

log _⇡✓_[soft] _[0] ,1_[(] _[x, a]_[)] _[ ⇡]_[log] _[ ⇡] ✓,_[soft] _1_[(] _[x, a]_[) +] _[ s]_[soft] _✓,1_[(] _[x, a]_[)] _[ ·]_[ (] _[✓][0][ −][✓]_[)] _[ .]_ (9) 

Eq. 9 shows that log _⇡✓_[soft] _[0] ,1_[(] _[a][|][x]_[)][ increases proportional to how well the direction of change in reward] ( _✓[0] − ✓_ ) aligns with the score _s✓_ ( _x, a_ ). We use Algorithm 2 to estimate _s✓_ with _g_ and apply Eq. 9. 

6 

Published as a conference paper at ICLR 2023 


![](raw/papers/2_LLM_NLP/LLM_paper_list/behavior/iclr_2023_Understanding_and_Adopting_Rational_Behavior_by_Bellman_Score_Estimation_images/iclr_2023_Understanding_and_Adopting_Rational_Behavior_by_Bellman_Score_Estimation.pdf-0007-01.png)


Figure 1: **Counterfactual Prediction Performance** for two scenarios: education subsidy (top row) and military enrollment incentives (bottom row). "no sub" corresponds to the initial reward from which the scores are estimated, "true sub" shows the probability of actions under the MaxEnt optimal policy obtained by resolving the RL model for perturbed rewards, and "pred sub" shows the scoreestimated optimal policy probabilities. 2000 _,_ 8000 denote the magnitude of the perturbation to the reward parameters which corresponds to the amount of subsidy or incentives. 

## 5 EXPERIMENTS 

## 5.1 REWARD IDENTIFICATION AND COUNTERFACTUAL PREDICTIONS 

**Environment** : We experiment with the Keane and Wolpin (KW) environment (Keane & Wolpin, 1994; 1997) which is a widely used benchmark in the DDC literature. The KW environment simulates a life-time of occupational decisions where at each time-step an agent selects among five actions: white collar work, blue collar work, military, school, and staying home. The state is a 25 dimensional vector consisting of various features about the agent such as the time spent in each occupation, and their current job. The dynamics captures how their current job decision alters the state. Agents are required to be forward looking and maximize the rewards summed over a life-time of occupational choices. The reward parameter _✓_ is a 63 dimensional vector where each parameter dimension corresponds to either a cost or benefit, e.g the cost of attending college or the benefit of transitioning from a white collar job to the military. The specific structure of the reward is omitted for brevity, but full details about the environment can be found in (Keane & Wolpin, 1994; 1997). 

**Results** : We solve the forward MaxEntRL problem in the KW environment and use the optimal policy to generate 100 trajectories of occupational choices for two different reward parameters: _✓edu_ and _✓mil_ . The first reward simulates a scenario with lower education costs and the latter captures a scenario with high benefits for military enrollment (Keane & Wolpin, 1994; 1997). 

Table 1: Mean squared error between true reward and IRL estimated reward evaluated on the demonstrations. 

||GCL|MSM|GAC|
|---|---|---|---|
|_✓edu_<br>12_._1_±_5_._3<br>**0**_._**14**_±_**0**_._**4**<br>**0**_._**16**_±_**0**_._**3**<br>_✓mil_<br>20_._5_±_6_._8<br>0_._14_±_0_._3<br>**0**_._**08**_±_**0**_._**2**||||



ground truth reward _r✓edu , r✓mil_ and the estimated rewards on _r✓_ ˆ _edu , r✓_ ˆ _mil_ evaluated on the demonstrations. As rewards are only identifiable up to additive constant shifts (Kim et al., 2021), we search this equivalence class for the parameters that minimize the mean squared error. Table 1 shows that GAC is able to identify the rewards to similar precision as the Method of Simulated Moments (MSM) baseline which is the state-of-the-art method for identification in the DDC literature. While GAC and GCL are optimizing the same objective, i.e Eq. 8, we posit that trading off bias for lower variance (as explained in section 4.1) accounts for GAC’s superior performance. 

Next we test the counterfactual prediction performance using the method in section 4.2. We simulate two counterfactual scenarios where the government provides a subsidy for college tuition and military participation. The scenarios are simulated by reducing the cost parameters in the reward for attending college and joining the military. Figure 1 shows how our score-estimated counterfactual predictions of policy changes compare to fully re-solving the RL problem for the perturbed rewards. We are able 

7 

Published as a conference paper at ICLR 2023 

Table 2: **State Control Performance** when provided with _one_ demonstration. 

||BC|DAC|AIRL|GCL|GAC (OURS)|
|---|---|---|---|---|---|
|BALL INCUPCATCH|39_._2_±_15_._2|291_._9_±_29_._1|124_._3_±_25_._0|502_._2_±_104_._2|**949**_._**9**_±_**42**_._**4**|
|CARTPOLESWING-UP|106_._0_±_50_._1|709_._0_±_40_._9|591_._4_±_134_._2|320_._1_±_105_._8|**970**_._**1**_±_**26**_._**1**|
|FINGERSPIN|159_._1_±_60_._1|203_._5_±_91_._9|105_._0_±_22_._8|304_._1_±_50_._6|**984**_._**0**_±_**18**_._**4**|
|FISHSWIM|205_._9_±_49_._2|150_._5_±_10_._8|10_._5_±_85_._1|203_._0_±_89_._4|**946**_._**8**_±_**22**_._**5**|
|HOPPERSTAND|10_._6_±_50_._2|809_._1_±_98_._4|592_._2_±_101_._2|**900**_._**1**_±_**150**_._**2**|**917**_._**1**_±_**15**_._**9**|
|WALKERSTAND|143_._5_±_22_._8|406_._5_±_89_._5|739_._6_±_63_._5|194_._4_±_39_._5|**990**_._**0**_±_**49**_._**0**|
|WALKERWALK|40_._1_±_39_._1|125_._9_±_78_._1|495_._1_±_77_._2|102_._4_±_30_._0|**943**_._**1**_±_**56**_._**7**|
|CHEETAHRUN|230_._1_±_99_._8|206_._10_±_75_._1|102_._4_±_29_._5|169_._1_±_20_._3|**807**_._**0**_±_**17**_._**5**|
|QUADRUPEDRUN|10_._45_±_91_._9|316_._3_±_33_._6|195_._4_±_91_._5|33_._5_±_98_._2|**961**_._**3**_±_**50**_._**1**|




![](raw/papers/2_LLM_NLP/LLM_paper_list/behavior/iclr_2023_Understanding_and_Adopting_Rational_Behavior_by_Bellman_Score_Estimation_images/iclr_2023_Understanding_and_Adopting_Rational_Behavior_by_Bellman_Score_Estimation.pdf-0008-03.png)


Figure 2: **Visual Control Performance** when a varying number of demonstrations. 

to accurately estimate the increase in time spent in school and the military. Particularly, when the reward perturbations are smaller, the score-based counterfactual predictions are more accurate. For significantly larger perturbations, we would expect our method’s accuracy to drop as the first order approximation is no longer valid. 

## 5.2 BEHAVIOR IMITATION 

**Environment and Baselines** : We experiment with high-dimensional robotics control environments from the DeepMind Control Suite (DMC) (Tassa et al., 2018), which is a widely used benchmark for RL and Imitation Learning (IL). For state control problems the agent has access to its proprioceptive states. For visual control the agent sees three consecutive observation images of size 84 _⇥_ 84 _⇥_ 3 and uses action-repeat following the standard DMC protocol for visual RL (Yarats et al., 2022). We compare GAC against Behavioral Cloning (BC) (Pomerleau, 1991), Adversarial IRL (AIRL) (Fu et al., 2018), Guided Cost Learning (GCL) (Finn et al., 2016b), and Discriminator Actor-Critic (DAC) (Kostrikov et al., 2019; Cohen et al., 2021). We report the mean performance and standard deviation across ten random seeds where performance of each seed is measured as the mean reward of twenty trajectories sampled from the final policy. (see Appendix B for experiment details) 

**Results** : For state control, we experiment in the extreme low-data regime where only one demonstration trajectory (consisting of 1000 state-actions) is provided. The experts for all tasks obtain rewards near 1000. Table 2 shows that GAC outperforms all baselines while achieving expert level performance. For visual control, Figure 2 shows that GAC outperforms baselines while achieving expert performance on many hard visual control tasks with just one demonstration. In particular, we found that the IRL baselines GCL and AIRL are incapable of performing well, which suggests that trading off bias for lower variance is beneficial in practice. While DAC achieves competitive performance on some tasks such as HOPPER STAND, it does not have the benefit of learning a reward. Some limitations of GAC include its longer compute time, potential optimization instabilities from the deadly triad (Van Hasselt et al., 2018), and double-sampling (Zhu & Ying, 2020). A more complete discussion of the limitations as well as hyper-parameter ablation studies are in Appendix C. 

8 

Published as a conference paper at ICLR 2023 

Table 3: **Reward Transfer** when provided with _ten_ demonstrations in the original environment 

|**ard Transfer**wh|en provided|with_ten_dem|onstrations|in the original|
|---|---|---|---|---|
||DIRECT|AIRL|GCL|GAC (OURS)|
|SLIPPERYWALKER|302_._5_±_29_._5|493_._2_±_49_._3|201_._9_±_89_._6|**789**_._**3**_±_**93**_._**3**|
|STIFFWALKER|205_._2_±_98_._3|455_._2_±_99_._3|301_._3_±_89_._4|**668**_._**9**_±_**60**_._**2**|
|SLIPPERYCHEETAH|366_._5_±_29_._3|591_._9_±_65_._7|93_._1_±_60_._1|**803**_._**9**_±_**98**_._**9**|
|STIFFCHEETAH|363_._2_±_39_._6|430_._3_±_68_._2|325_._0_±_44_._6|**650**_._**3**_±_**92**_._**3**|



## 5.3 BEHAVIOR TRANSFER 

Next we show that that learned rewards from GAC robustly transfer behaviors to perturbed environments (Fu et al., 2018). We create variations of the state-based DMC environments by altering the surface and joint frictions to simulate realistic scenarios for robot learning where we may want to learn a single reward that can teach robots of different joint mechanics to mobilize in different walking surfaces. For example, the SLIPPERYWALKER environment has about 80% less friction on the walking surface while STIFFCHEETAH has roughly 150% higher friction on the robot joints. Rewards are first learned on the non-perturbed environments using 10 demonstrations, then re-optimized via RL on the perturbed environments. As a sanity check, we also include the DIRECT baseline which simply uses policy learned in the non-perturbed environment. Table 3 shows that GAC is able to outperform the baselines. However, GAC is not able to attain perfect expert level performance, which we believe is due to the perturbed environments requiring slightly different state visitation strategies. 

## 6 RELATED WORKS 

**Inverse Reinforcement Learning and Dynamic Discrete Choice** : The field of IRL was pioneered by (Ng et al., 2000) and has since accumulated a rich history starting with early works that rely on linear reward models and hand-crafted features (Abbeel & Ng, 2004; Ramachandran & Amir, 2007; Choi & Kim, 2011; Neu & Szepesvári, 2012). Since the introduction of the Maximum Entropy IRL (Ziebart et al., 2008; 2010) framework, the field has gravitated towards learning more flexible rewards parameterized by deep neural networks (Wulfmeier et al., 2015) paired with adversarial training (Fu et al., 2018; Ho & Ermon, 2016; Finn et al., 2016b;a) while some works attempt scale more classical methods such as Bayesian IRL (Chan & van der Schaar, 2021; Mandyam et al., 2021). Most related to our work is gradient-based apprenticeship learning (Neu & Szepesvári, 2012) which showed that the gradient of the infinite horizon Q-function in the standard RL setting satisfies a linear fixed point equation. Bellman gradient iteration (Li et al., 2017; 2019) and maximum-likelihood IRL (Vroman, 2014) take a computational approach to estimating approximate _Q_ -gradients. Unlike these prior works, we generalize results to the MaxEntRL setting where we prove new properties of the Bellman score, and derive a dynamic programming algorithm that provably converges to the infinite-horizon _Q_ -gradient. Most significantly, we propose a model-free score estimation algorithm which scales to high-dimensional environments and apply it to various applications beyond IRL. The field of econometrics has a rich body of work on identifying Dynamic Discrete Choice (DDC) (Rust, 1994; Arcidiacono & Miller, 2011; 2020; Abbring & Daljord, 2020) models which is equivalent to the problem of MaxEntIRL (Ziebart et al., 2008). DDC literature more so focuses on counter factual predictions (Christensen & Connault, 2019) which has various application areas such as in labor markets (Keane & Wolpin, 1994), health care (Heckman & Navarro, 2007), and retail competition (Arcidiacono & Miller, 2011). 

**Imitation Learning** A related field of Imitation Learning (IL) seeks to learn policies from demonstrations (Pomerleau, 1991; Ho & Ermon, 2016; Zhang et al., 2020; Rajaraman et al., 2020; Xu et al., 2020) which is a viable alternative when the sole goal is behavior adoption. In recent years, IL has shown superior results than IRL for policy performance, particularly in the visual IL space. (Samak et al., 2021; Young et al., 2021) Techniques such as data augmentation and encoder sharing that have boosted performance in visual RL (Yarats et al., 2022) have been combined with adversarial IL methods to solve challenging control environments (Cohen et al., 2021). 

## 7 CONCLUSION 

We have studied the theoretical properties of the Bellman score which allowed us to derive a modelfree algorithm, score-learning, for score-estimation. We showed that score-learning has various applications in IRL, behavior transfer, reward identification, and counterfactual analysis. We are looking forward to seeing future works that apply score-learning to other problems such as reward design (Hadfield-Menell et al., 2017) and explainable RL (Puiutta & Veith, 2020). 

9 

Published as a conference paper at ICLR 2023 

## ACKNOWLEDGEMENTS 

This research was supported by NSF (#1651565), ARO (W911NF-21-1-0125), ONR (N00014-23-12159), CZ Biohub, and HAI. We would like to acknowledge Bertsekas & Tsitsiklis (1995) for their descriptions of the Value Iteration algorithm which sparked crucial intuitions for the theory in this work. 

## REFERENCES 

Pieter Abbeel and Andrew Y Ng. Apprenticeship learning via inverse reinforcement learning. In _Proceedings of the twenty-first international conference on Machine learning_ , pp. 1, 2004. 

Jaap H Abbring and Øystein Daljord. Identifying the discount factor in dynamic discrete choice models. _Quantitative Economics_ , 11(2):471–501, 2020. 

Peter Arcidiacono and Robert A Miller. Conditional choice probability estimation of dynamic discrete choice models with unobserved heterogeneity. _Econometrica_ , 79(6):1823–1867, 2011. 

Peter Arcidiacono and Robert A Miller. Identifying dynamic discrete choice models off short panels. _Journal of Econometrics_ , 215(2):473–485, 2020. 

Shaojie Bai, J Zico Kolter, and Vladlen Koltun. Deep equilibrium models. _Advances in Neural Information Processing Systems_ , 32, 2019. 

Richard Bellman. _Dynamic Programming_ . Dover Publications, 1957. ISBN 9780486428093. 

Dimitri P Bertsekas and John N Tsitsiklis. Neuro-dynamic programming: an overview. In _Proceedings of 1995 34th IEEE conference on decision and control_ , volume 1, pp. 560–564. IEEE, 1995. 

Stephen Boyd, Lin Xiao, and Almir Mutapcic. Subgradient methods. _lecture notes of EE392o, Stanford University, Autumn Quarter_ , 2004:2004–2005, 2003. 

- Alex James Chan and Mihaela van der Schaar. Scalable bayesian inverse reinforcement learning. In _International Conference on Learning Representations_ , 2021. URL https://openreview. net/forum?id=4qR3coiNaIv. 

Jaedeug Choi and Kee-Eung Kim. Map inference for bayesian inverse reinforcement learning. _Advances in Neural Information Processing Systems_ , 24, 2011. 

Timothy Christensen and Benjamin Connault. Counterfactual sensitivity and robustness. _arXiv preprint arXiv:1904.00989_ , 2019. 

- Samuel Cohen, Brandon Amos, Marc Peter Deisenroth, Mikael Henaff, Eugene Vinitsky, and Denis Yarats. Imitation learning from pixel observations for continuous control. In _Deep RL Workshop NeurIPS 2021_ , 2021. 

Kenji Doya. Reinforcement learning: Computational theory and biological mechanisms. _HFSP journal_ , 1(1):30, 2007. 

- Chelsea Finn, Paul Christiano, Pieter Abbeel, and Sergey Levine. A connection between generative adversarial networks, inverse reinforcement learning, and energy-based models. _arXiv preprint arXiv:1611.03852_ , 2016a. 

- Chelsea Finn, Sergey Levine, and Pieter Abbeel. Guided cost learning: Deep inverse optimal control via policy optimization. In _International Conference on Machine Learning_ , pp. 49–58, June 2016b. 

- Justin Fu, Katie Luo, and Sergey Levine. Learning robust rewards with adverserial inverse reinforcement learning. In _International Conference on Learning Representations_ , 2018. URL https://openreview.net/forum?id=rkHywl-A-. 

- Tuomas Haarnoja, Haoran Tang, Pieter Abbeel, and Sergey Levine. Reinforcement learning with deep energy-based policies. In _International conference on machine learning_ , pp. 1352–1361. PMLR, 2017. 

Tuomas Haarnoja, Aurick Zhou, Pieter Abbeel, and Sergey Levine. Soft actor-critic: Off-policy maximum entropy deep reinforcement learning with a stochastic actor. In _International conference on machine learning_ , pp. 1861–1870. PMLR, 2018. 

10 

Published as a conference paper at ICLR 2023 

Dylan Hadfield-Menell, Smitha Milli, Pieter Abbeel, Stuart J Russell, and Anca Dragan. Inverse reward design. _Advances in neural information processing systems_ , 30, 2017. 

- James J Heckman and Salvador Navarro. Dynamic discrete choice and dynamic treatment effects. _Journal of Econometrics_ , 136(2):341–396, 2007. 

- Tsubasa Hirakawa, Takayoshi Yamashita, Toru Tamaki, Hironobu Fujiyoshi, Yuta Umezu, Ichiro Takeuchi, Sakiko Matsumoto, and Ken Yoda. Can ai predict animal movements? filling gaps in animal trajectories using inverse reinforcement learning. _Ecosphere_ , 9(10):e02447, 2018. 

- Jonathan Ho and Stefano Ermon. Generative adversarial imitation learning. In _Advances in Neural Information Processing Systems_ , pp. 4565–4573, 2016. 

- Myrto Kalouptsidi, Paul T Scott, and Eduardo Souza-Rodrigues. dynamic discrete choice models. Technical report, National Bureau of Economic Research, 2015. 

- Michael P Keane and Kenneth I Wolpin. The solution and estimation of discrete choice dynamic programming models by simulation and interpolation: Monte carlo evidence. _The review of economics and statistics_ , pp. 648–672, 1994. 

- Michael P Keane and Kenneth I Wolpin. The career decisions of young men. _Journal of political Economy_ , 105(3):473–522, 1997. 

- Kuno Kim, Shivam Garg, Kirankumar Shiragur, and Stefano Ermon. reinforcement learning. In _International Conference on Machine Learning_ , pp. 5496–5505. PMLR, 2021. 

- Ilya Kostrikov, Kumar Krishna Agrawal, Debidatta Dwibedi, Sergey Levine, and Jonathan Tompson. Discriminator-actor-critic: Addressing sample inefficiency and reward bias in adversarial imitation learning. In _International Conference on Learning Representations_ , 2019. URL https:// openreview.net/forum?id=Hk4fpoA5Km. 

- Bai Li, Changyou Chen, Hao Liu, and Lawrence Carin. On connecting stochastic gradient mcmc and differential privacy. In _The 22nd International Conference on Artificial Intelligence and Statistics_ , pp. 557–566. PMLR, 2019. 

- Kun Li, Yanan Sui, and Joel W Burdick. Bellman gradient iteration for inverse reinforcement learning. _arXiv preprint arXiv:1707.07767_ , 2017. 

- Aishwarya Mandyam, Didong Li, Diana Cai, Andrew Jones, and Barbara Engelhardt. bayesian inverse reinforcement learning via conditional kernel density estimation. In _Fourth Symposium on Advances in Approximate Bayesian Inference_ , 2021. 

- Emre O Neftci and Bruno B Averbeck. _Nature Machine Intelligence_ , 1(3):133–143, 2019. 

- Gergely Neu and Csaba Szepesvári. Apprenticeship learning using inverse reinforcement learning and gradient methods. _arXiv preprint arXiv:1206.5264_ , 2012. 

- Andrew Y Ng, Stuart J Russell, et al. Algorithms for inverse reinforcement learning. In _Icml_ , pp. 663–670, 2000. 

- Yael Niv. Reinforcement learning in the brain. _Journal of Mathematical Psychology_ , 53(3):139–154, 2009. 

- Dean A Pomerleau. _Neural computation_ , 3(1):88–97, 1991. ISSN 0899-7667. 

- Erika Puiutta and Eric Veith. Explainable reinforcement learning: A survey. In _International cross-domain conference for machine learning and knowledge extraction_ , pp. 77–95. Springer, 2020. 

- Nived Rajaraman, Lin Yang, Jiantao Jiao, and Kannan Ramchandran. Toward the fundamental limits of imitation learning. _Advances in Neural Information Processing Systems_ , 33:2914–2924, 2020. 

- Deepak Ramachandran and Eyal Amir. Bayesian inverse reinforcement learning. In _IJCAI_ , volume 7, pp. 2586–2591, 2007. 

11 

Published as a conference paper at ICLR 2023 

- John Rust. Structural estimation of markov decision processes. _Handbook of econometrics_ , 4: 3081–3143, 1994. 

- Tanmay Vilas Samak, Chinmay Vilas Samak, and Sivanathan Kandhasamy. Robust behavioral cloning for autonomous vehicles using end-to-end imitation learning. _SAE International Journal of Connected and Automated Vehicles_ , 4(3):279–295, aug 2021. ISSN 2574-0741. doi: 10.4271/ 12-04-03-0023. URL https://doi.org/10.4271/12-04-03-0023. 

- Toryn LJ Schafer, Christopher K Wikle, and Mevin B Hooten. Bayesian inverse reinforcement learning for collective animal movement. _The Annals of Applied Statistics_ , 16(2):999–1013, 2022. 

- David Silver, Aja Huang, Chris J Maddison, Arthur Guez, Laurent Sifre, George Van Den Driessche, Julian Schrittwieser, Ioannis Antonoglou, Veda Panneershelvam, Marc Lanctot, et al. Mastering the game of go with deep neural networks and tree search. _nature_ , 529(7587):484, 2016. 

- Yuval Tassa, Yotam Doron, Alistair Muldal, Tom Erez, Yazhe Li, Diego de Las Casas, David Budden, Abbas Abdolmaleki, Josh Merel, Andrew Lefrancq, Timothy Lillicrap, and Martin Riedmiller. DeepMind control suite. _arXiv preprint arXiv:1801.00690_ , January 2018. 

- Hado Van Hasselt, Yotam Doron, Florian Strub, Matteo Hessel, Nicolas Sonnerat, and Joseph Modayil. Deep reinforcement learning and the deadly triad. _arXiv preprint arXiv:1812.02648_ , 2018. 

- Monica C Vroman. _Maximum likelihood inverse reinforcement learning_ . Rutgers The State University of New Jersey-New Brunswick, 2014. 

- Markus Wulfmeier, Peter Ondruska, and Ingmar Posner. Maximum entropy deep inverse reinforcement learning. _arXiv preprint arXiv:1507.04888_ , July 2015. 

- Tian Xu, Ziniu Li, and Yang Yu. Error bounds of imitating policies and environments. _Advances in Neural Information Processing Systems_ , 33:15737–15749, 2020. 

- Shoichiro Yamaguchi, Honda Naoki, Muneki Ikeda, Yuki Tsukada, Shunji Nakano, Ikue Mori, and Shin Ishii. Identification of animal behavioral strategies by inverse reinforcement learning. _PLoS computational biology_ , 14(5):e1006122, 2018. 

- Denis Yarats and Ilya Kostrikov. Soft actor-critic (sac) implementation in pytorch. https:// github.com/denisyarats/pytorch_sac, 2020. 

- Denis Yarats, Ilya Kostrikov, and Rob Fergus. Image augmentation is all you need: Regularizing deep reinforcement learning from pixels. In _International Conference on Learning Representations_ , 2021. URL https://openreview.net/forum?id=GY6-6sTvGaf. 

- Denis Yarats, Rob Fergus, Alessandro Lazaric, and Lerrel Pinto. Mastering visual continuous control: Improved data-augmented reinforcement learning. In _International Conference on Learning Representations_ , 2022. URL https://openreview.net/forum?id=_SJ-_yyes8. 

- Sarah Young, Dhiraj Gandhi, Shubham Tulsiani, Abhinav Gupta, Pieter Abbeel, and Lerrel Pinto. Visual imitation made easy. In _Conference on Robot Learning_ , pp. 1992–2005. PMLR, 2021. 

- Amy Zhang, Harsh Satija, and Joelle Pineau. Decoupling dynamics and reward for transfer learning, 2018. URL https://openreview.net/forum?id=H1aoddyvM. 

- Yufeng Zhang, Qi Cai, Zhuoran Yang, and Zhaoran Wang. Generative adversarial imitation learning with neural network parameterization: Global optimality and convergence rate. In _International Conference on Machine Learning_ , pp. 11044–11054. PMLR, 2020. 

- Yuhua Zhu and Lexing Ying. Borrowing from the future: An attempt to address double sampling. In _Mathematical and scientific machine learning_ , pp. 246–268. PMLR, 2020. 

- Brian D Ziebart, Andrew L Maas, J Andrew Bagnell, and Anind K Dey. Maximum entropy inverse reinforcement learning. In _AAAI_ , volume 8, pp. 1433–1438, 2008. 

- Brian D Ziebart, J Andrew Bagnell, and Anind K Dey. Modeling interaction via the principle of maximum causal entropy. In _ICML_ , 2010. 

12 

