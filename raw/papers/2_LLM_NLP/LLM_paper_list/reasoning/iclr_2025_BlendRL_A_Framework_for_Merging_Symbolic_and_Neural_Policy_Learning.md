# BLENDRL: A FRAMEWORK FOR MERGING SYMBOLIC AND NEURAL POLICY LEARNING 

**Hikaru Shindo**[1] **Quentin Delfosse**[1] **Devendra Singh Dhami**[2] **Kristian Kersting**[1] _[,]_[3] _[,]_[4] 

1Computer Science Department, TU Darmstadt, Germany 

2Mathematics & Computer Science Department, TU Eindhoven, Netherlands 

3Hessian Center for Artificial Intelligence (hessian.AI), Darmstadt, Germany 

4German Research Center for Artificial Intelligence (DFKI), Darmstadt, Germany 

_{_ hikaru.shindo, quentin.delfosse, kersting _}_ @tu-darmstadt.de d.s.dhami@tue.nl 

## ABSTRACT 

Humans can leverage both symbolic reasoning and intuitive reactions. In contrast, reinforcement learning policies are typically encoded in either opaque systems like neural networks or symbolic systems that rely on predefined symbols and rules. This disjointed approach severely limits the agents’ capabilities, as they often lack either the flexible lowlevel reaction characteristic of neural agents or the interpretable reasoning of symbolic agents. To overcome this challenge, we introduce _BlendRL_ , a neuro-symbolic RL framework that harmoniously integrates both paradigms within RL agents that use mixtures of both logic and neural policies. We empirically demonstrate that BlendRL agents outperform both neural and symbolic baselines in standard Atari environments, and showcase their robustness to environmental changes. Additionally, we analyze the interaction between neural and symbolic policies, illustrating how their hybrid use helps agents overcome each other’s limitations. 

## 1 INTRODUCTION 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_BlendRL_A_Framework_for_Merging_Symbolic_and_Neural_Policy_Learning_images/iclr_2025_BlendRL_A_Framework_for_Merging_Symbolic_and_Neural_Policy_Learning.pdf-0001-11.png)


**----- Start of picture text -----**<br>
Reach the joey Get oxygen and divers<br>Punch the enemy Shoot the enemy<br>High level reasoning<br>Low level reaction<br>**----- End of picture text -----**<br>


To solve complex problems, humans employ two fundamental modes of thinking: (1) instinctive responses for immediate reaction and motor control and (2) abstract reasoning, using distinct identifiable concepts. These two facets of human intelligence are commonly referred to as _System 1_ and _System 2_ (Kahneman, 2011). While our reasoning system requires symbols to build interpretable decision rules, instinctive reactions do not rely on such inductive Punch the enemy bias, but lack transparency. Both systems are required in challenging RL environments, such as _Kangaroo_ , where the agent’s goal is to reach its joey (at the top), captured by monkeys that need to be punched out of the way, or in _Seaquest_ , where the agent controls a submarine, and needs to collect swimming divers without running out of oxygen ( _cf._ Figure 1). Developing agents that can effectively Figure 1: **Most** use both information processing systems has proven to be a persis- **both** _**reasoning**_ **(top)** tent challenge (Lake et al., 2017; Mao et al., 2019; Kautz, 2022). **(bottom)** , to master _e.g._ The main difficulty not only lies in achieving high-level capabilities with both systems, but also in seamlessly integrating these systems to interact synergistically, in order to maximize performances without sacrificing transparency. 

Figure 1: **Most problems require both** _**reasoning**_ **(top) and** _**reacting**_ **(bottom)** , to master _e.g._ the Kangaroo (left) and Seaquest (right) Atari games. 

1 

Deep neural networks have demonstrated an ability to effectively learn policies across a wide range of tasks without relying on any prior knowledge about the task (Mnih et al., 2015; Schulman et al., 2017; Badia et al., 2020; Bhatt et al., 2024). However, these black box policies can leverage shortcuts that are imperceptible to human observers (Locatello et al., 2020; Liu & Borisyuk, 2024). For instance, in the simple Atari Pong game, deep agents tend to rely on the enemy’s position rather than the ball’s one (Delfosse et al., 2024d), demonstrating deep learning systems’ tendency to rely on shortcut learning opportunities, that then fail to generalize to slightly modified environments. 

To enhance reasoning capabilities, symbolic reasoning has been integrated into approaches, using _e.g._ as logic-based policies (Jiang & Luo, 2019; Kimura et al., 2021; Cao et al., 2022; Delfosse et al., 2023a) or program-based frameworks (Sun et al., 2020; Verma et al., 2018; Lyu et al., 2019; Cappart et al., 2021; Kohler et al., 2024). These methods offer transparency, revisability, better generalization, and the potential for curriculum learning. However, they often rely on the incorporation of specific human inductive biases, necessary for solving the tasks, thus requiring experts to provide essential concepts or potential logic rules. Moreover, actions involving low-level reactions with subtle movements are challenging, if not impossible, to encode within these frameworks. This limitation underscores the constraints of symbolic systems in terms of their learning capabilities. Consequently, an important question emerges: _How can we build intelligent agents that leverage the strengths of both neural and symbolic modeling?_ The current approach to combining these systems typically uses a top-down ( _i.e._ sequential) method: using deliberative systems ( _e.g._ planners) to provide slow, high-level reasoning to select reactive systems ( _e.g._ Deep RL), that offer quick, low-level responses (Kokel et al., 2021). However, this sequential approach is not always suitable, _e.g._ , a self-driving can recompute its plan on a low-density highway, but must react quickly in heavy traffic, without replanning. Agents that can select for either neural or symbolic modeling based on the context are necessary. 

We introduce _BlendRL_ , a framework that integrates neural and logic-based policy learning in parallel. BlendRL agents learn logic-based interpretable reasoning as well as low-level control, combined through a blending function that utilizes a hybrid state representation. They can utilize high-level reasoning ( _e.g._ for path finding), which benefits from a symbolic (or object-centric) state representation, and low-level reactions, for finetuned control skills ( _e.g._ shooting at enemies), using pixel-based state representations. Although its neural component is less interpretable, it assists the agent in adapting to situations where it lacks sufficient symbolic representations. BlendRL provides hybrid policies, by distinctly modeling these two types of information processing systems, and selects its actions using a mixture of deep neural networks and differentiable logic reasoners (Evans & Grefenstette, 2018; Shindo et al., 2023; 2024b). Additionally, we propose an Advantage Actor-Critic (A2C)-based learning algorithm for BlendRL agents, that incorporates Proximal Policy Optimization (PPO) and policy regularization, supplemented by analysis on the interactions between neural and symbolic components of trained agents. Overall, we make the following contributions: 

- **(i)** We propose BlendRL to _jointly and simultaneously_ train symbolic and neural policies. 

- **(ii)** For efficient learning on the proposed framework, we adapt the PPO Actor Critic algorithm on the hybrid state representation. Moreover, we proposed a regularization method to balance neural and symbolic policies, providing agents that are both transparent reasoners and accurate reactors. 

- **(iii)** We empirically show that BlendRL agents outperform neural and the state-of-the-art neuro-symbolic baselines on environments where agents need to perform both high-level reasoning and low-level reacting. Moreover, we demonstrate the robustness of the BlendRL agents to environmental changes. 

- **(iv)** We provide a deeper analysis of the interactions between neural and symbolic policies, revealing how hybrid representations and policies can help agents overcome each other’s limitations. 

We start off by providing the necessary background, then introduce our BlendRL method for policy reasoning and learning. We experimentally evaluate BlendRL on three complex Atari games, comparing its performance to purely neural and logic baselines. Following this, we discuss related work before concluding. Our code and resources are openly available.[1] 

> 1https://github.com/ml-research/blendrl 

2 

## 2 BACKGROUND 

Let us introduce the necessary background before formally introducing our BlendRL method. 

**Deep Reinforcement Learning** . In RL, the task is modelled as a Markov decision process, _M_ = _< S, A, P, R, γ>_ , where, at every timestep _t_ , an agent in a state _st ∈S_ , takes action _at ∈A_ , receives a reward _rt_ = _R_ ( _st, at_ ) and a transition to the next state _st_ +1, according to environment dynamics _P_ ( _st_ +1 _|st, at_ ). Deep agents attempt to learn a parametric policy, _πθ_ ( _at|st_ ), in order to maximize the return ( _i.e._[�] _t[γ][t][r][t]_[,] with _γ ∈_ [0 _,_ 1]). The desired input to output ( _i.e._ state to action) distribution is not directly accessible, as RL agents only observe returns. The value _Vπθ_ ( _st_ ) (resp. Q-value _Qπθ_ ( _st, at_ )) function provides the return of the state (resp. state/action pair) following the policy _πθ_ . We provide more details in App. A.1. **First-Order Logic (FOL).** In FOL, a _Language L_ is a tuple ( _P, D, F, V_ ), where _P_ is a set of predicates, _D_ a set of constants, _F_ a set of function symbols (functors), and _V_ a set of variables. A _term_ is either a constant ( _e.g._ `obj1` _,_ `agent` ), a variable ( _e.g._ `O1` ), or a term which consists of a function symbol. An _atom_ is a formula `p` ( `t1` _, . . . ,_ `tn` ), where `p` is a predicate symbol ( _e.g._ `closeby` ) and `t1` _, . . . ,_ `tn` are terms. A _ground atom_ or simply a _fact_ is an atom with no variables ( _e.g._ `closeby` ( `obj1` _,_ `obj2` )). A _literal_ is an atom ( _A_ ) or its negation ( _¬A_ ). A _clause_ is a finite disjunction ( _∨_ ) of literals. A _ground clause_ is a clause with no variables. A _definite clause_ is a clause with exactly one positive literal. If _A, B_ 1 _, . . . , Bn_ are atoms, then _A ∨¬B_ 1 _∨ . . . ∨¬Bn_ is a definite clause. We write definite clauses in the form of _A_ :- _B_ 1 _, . . . , Bn_ . _A_ is the rule _head_ , and the set _{B_ 1 _, . . . , Bn}_ is its _body_ . We interchangeably use definite clauses and _rules_ . 

**Differentiable Forward Reasoning** is a data-driven approach to reasoning in First-Order Logic (FOL) (Russell & Norvig, 2010). In forward reasoning, given a set of facts and rules, new facts are deduced by applying the rules to the facts. Differentiable forward reasoning is a differentiable implementation of forward reasoning, utilizing tensor-based differentiable operations (Evans & Grefenstette, 2018; Shindo et al., 2023) or graph-based approaches (Shindo et al., 2024b). This approach can be efficiently applied to reinforcement learning tasks by encoding actions in the form of rules, where the head defines an action and the body specifies its conditions. To learn the _importance_ or truth value of each rule, they can be associated with learnable rule weights. Consequently, hypotheses can be formulated in terms of rules and learned from data. 

## 3 BLENDRL 

BlendRL integrates both abstract reasoning and instinctive reactions by combining symbolic and neural policies. As illustrated in Figure 2, the neural policy processes sub-symbolic ( _i.e._ pixel-based) representations to compute a distribution over the actions, while the reasoning module employs differentiable reasoning on symbolic states. These action distributions are then blended to obtain the final action distribution. We begin by describing the inner workings of each policy type, and of the blending module. Next, we discuss how we leverage the common sense encapsulated in pretrained large language models (LLMs) to obtain symbolic concepts and their evaluation functions. Finally, we describe how we adapted the PPO actor-critic algorithm to perform end-to-end training of BlendRL modules. Let us first formally introduce the state representations. 

## 3.1 HYBRID STATE REPRESENTATIONS 

BlendRL agents employ _two_ distinct state representations: (i) _pixel-based representations_ and (ii) _objectcentric representations_ , that can be extracted by object discovery models (Redmon et al., 2016; Lin et al., 2020; Delfosse et al., 2023b; Zhao et al., 2023). Pixel-based representations usually consist of a stack of raw images, provided by the environment, fed to a deep convolutional network, as introduced by Mnih et al. (2015). Our considered symbolic (object-centric) representations consist of a list of objects, with attributes ( _e.g._ position, orientation, color, etc.), allowing for logical reasoning on structured representations (Zadaianchuk et al., 2021; Liu et al., 2021; Yoon et al., 2023; W¨ust et al., 2024; Stammer et al., 2024b). 

3 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_BlendRL_A_Framework_for_Merging_Symbolic_and_Neural_Policy_Learning_images/iclr_2025_BlendRL_A_Framework_for_Merging_Symbolic_and_Neural_Policy_Learning.pdf-0004-00.png)


**----- Start of picture text -----**<br>
Symbolic state Logic policy Logic distribution<br>obj   x      y      o      v  Reasoning module<br> 80    72    -1     0<br> 48    16     0    60<br> 68    44     1     0 0.2 ×<br> 46    73     1     0<br>...<br>+<br>Reacting module 0.8 ×<br>Blending module<br>Final action<br>distribution<br>*<br>Raw pixel state Neural policy Neural distribution : LLM Assisted Logic Rule Generation<br>**----- End of picture text -----**<br>


Figure 2: **Overview of the BlendRL framework** . BlendRL employs two policy types to compute action probabilities. A deep neural policy (bottom) handles low-level reactions, operating on pixel-based states, while differentiable forward reasoners (Shindo et al., 2023) (top) manage high-level reasoning on symbolic, object-centric states extracted from the pixel-based input. The blending module then merges the two action distributions by taking a weighted sum based on the current state. The logic policy and the blending module incorporate human inductive biases extracted using language models (LLMs) and the task context. All three components (the two policies and the blending module) can be trained jointly using gradients. 

Raw sub-symbolic states, _**x** ∈_ R _[F][ ×][W][ ×][H][×][C]_ , consist of the last _F_ observed frames, while symbolic states _**z** ∈_ R _[F][ ×][n][×][m]_ represents the corresponding symbolic (or object-centric) state, with _n_ denoting the number of objects and of _m_ extracted properties. As illustrated in Figure 2, on the _Seaquest_ Atari environment, a symbolic state consists of objects such as the agent’s submarine, the oxygen level, a diver, sharks, etc., with their _x_ and _y_ positional coordinates, as well as orientation and value. If a property is not relevant to an object ( _e.g._ the orientation for the oxygen bar), it is set to 0. 

## 3.2 THE BLENDED NEURO-SYMBOLIC POLICIES 

Using both state representations, BlendRL agents compute the action selection probabilities by aggregating the probabilities of its _neural_ and its _logic_ policies. 

**The neural policy** , _πθ[neural]_ : R _[F][ ×][W][ ×][H][×][C] →_ [0 _,_ 1] _[A]_ , consists of a neural network, parameterized by _θ_ , that computes action distributions based on pixel state representations ( _**x**_ ). Convolutional neural networks based policies (Mnih et al., 2015; Schulman et al., 2017; Hessel et al., 2018) are the most common deep policy types (Huang et al., 2022), but visual transformers can also be used (Chen et al., 2021; Parisotto et al., 2020). 

**The logic policy** , _πϕ[logic]_ : R _[n][×][m] →_ [0 _,_ 1] _[A]_ , is a differentiable forward reasoner (Shindo et al., 2023), parameterized by _ϕ_ , that reasons over object-centric states, such as the one depicted in Figure 2. Action rules are rules in first-order logic that encode conditions for an action to be selected (Reiter, 2001; Delfosse et al., 2023a). An action rule defines an action as its head atom ( _action atom_ ) and encodes its preconditions in its body atoms ( _state atoms_ ). We provide exemplary action rules for _Seaquest_ in Listing 1. 

These rules are transparent, _e.g._ [R1] can be inter[R1] 0.73: up(X):-is_empty(Oxygen). preted as ”Select UP if the oxygen is empty”. The [R2] 0.42: up(X):-above(Diver,Agent). body atom is ~~e~~ mpty is a _state predicate_ , whose [R3] 0.31: left(X):-left_of(Diver,Agent). truth level can be computed from an object-centric Listing 1: Exemplary action rules for _Seaquest_ . state. Each state predicate is associated with a (differentiable) function, known as a _valuation function_ , to compute its truth value, or confidence. 

4 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_BlendRL_A_Framework_for_Merging_Symbolic_and_Neural_Policy_Learning_images/iclr_2025_BlendRL_A_Framework_for_Merging_Symbolic_and_Neural_Policy_Learning.pdf-0005-00.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_BlendRL_A_Framework_for_Merging_Symbolic_and_Neural_Policy_Learning_images/iclr_2025_BlendRL_A_Framework_for_Merging_Symbolic_and_Neural_Policy_Learning.pdf-0005-01.png)


**----- Start of picture text -----**<br>
If an enemy or a coconut is close to the      :   logic(X)  :- nothing_around(Agent)<br>agent, use the neural policy, but if      :   neural(X) :- closeby(Agent, Enemy)<br>nothing is around the agent, use logic.     :   neural(X) :- closeby(Agent, Coconut)<br>x      y      o      v  x      y      o      v<br> 20     8     -1     0  20     8     -1       0<br>  0      0      0      0  24     8      0      0<br> 30    35     1     0  30    35     1       0<br> 46    73     1     0  46    73     1       0<br>**----- End of picture text -----**<br>


Figure 3: **LLMs allow for generating transparent blending function** , **Blending** exemplified on _Kangaroo_ . Top: An LLM generates a transparent blending function that evaluates if (i) there exists an enemy close to the player and if (ii) nothing is around the player. Bottom: Two states are evaluated: the body predicates ( _i.e._ nothing ~~a~~ round and closeby) values are multiplied with the rules’ associated learnable weight (displayed on their left), to obtain the neural/logic importance weighting. 

For example, is ~~e~~ mpty can be mapped to a function, such as _sigmoid_ (( _x − α_ ) _/γ_ ), which translates the actual oxygen value _x ∈_ [0 _,_ 100] (from the object-centric state) into a truth score ranging from [0 _,_ 1]. The second rule [R2] represents the same action selection, UP, but is motivated by the aim of collecting divers. The third rule [R3] selects another action (LEFT), if a diver is left of the player. After evaluating the valuation functions for each state predicate, we perform differentiable forward reasoning (Shindo et al., 2023) to deduce action atoms defined by the action rules based on the state atoms. Forward reasoning involves inferring all deducible knowledge ( _i.e._ the head atoms for actions) from an observed state. This process allows us to obtain confidence levels (as probabilities) for all actions defined in the symbolic policy. 

Contrary to NUDGE policies (Delfosse et al., 2023a), BlendRL generates action rules and their necessary elements (predicates and their valuation functions) using a Large Language Model (LLM), as described in Section 3.3. We also integrate memory-efficient message-passing forward reasoners (Shindo et al., 2024b) to overcome potential memory bottlenecks associated with conventional symbolic policies. NUDGE and other common logic policies consumes memory quadratically with respect to the number of relations and entities, significantly limiting their scalability. In contrast, BlendRL’s symbolic policy scales linearly, suitable to more complex environments scalable training parallelization. 

**The blending module** is a differentiable function parameterized by _λ_ . It can be encoded as an explicit logicbased function ( _Bλ_ : R _[F][ ×][n][×][m] →_ [0 _,_ 1]), as an implicit neural network-based ( _i.e._ pixel) state evaluator ( _Bλ_ : R _[F][ ×][W][ ×][H][×][C] →_ [0 _,_ 1]), or a combination of both. While a logic-based policy provides the advantage of transparency, it relies on inductive biases. If these are not available, a neural blender is necessary. 

Figure 3 describes the overall process of the blending module. It computes distributions over neural and logic policies given symbolic states, based on the blending rules generated by LLMs. The blending weighted rule set of an agent trained on _Kangaroo_ is depicted in the top right corner. It encodes the fact that the neural module should be selected when a monkey or a deadly thrown coconut is around (to allow for dodging the coconut or adjusting the position to optimally punch the monkey). When nothing is around the agent, it can safely rely on its logic policy (depicted just above it), which allows it to navigate the _Kangaroo_ environment. Formally, the blending module provides the weight _β ∈_ [0 _,_ 1] of the neural policy, _β_ = _Bλ_ ( _**z**_ ) or _β_ = _Bλ_ ( _**x**_ ), (and thus implicitly of the neural policy). We empirically show on _Kangaroo_ and _Seaquest_ BlendRL agents with symbolic blending modules outperform ones with neural modules ( _cf._ Table 6 in Appendix A.8). 

Finally, given a raw state _**x**_ , and its symbolic version, _**z**_ , the final action distribution is computed as: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_BlendRL_A_Framework_for_Merging_Symbolic_and_Neural_Policy_Learning_images/iclr_2025_BlendRL_A_Framework_for_Merging_Symbolic_and_Neural_Policy_Learning.pdf-0005-08.png)


where _s_ = ( _**x** ,_ _**z**_ ). Note that the blending module can also become a rigid selector, if replaced by 1 _β>_ 0 _._ 5. 

5 

_% Policy ruleset (Kangaroo)_ 

0.73 up(X):-on_ladder(Player,Ladder),same_floor(Player,Ladder). 

0.74 right(X):-left_of(Player,Ladder),same_floor(Player,Ladder). 0.72 left(X):-right_of(Player,Ladder),same_floor(Player,Ladder). ... 

_% Blending ruleset (Kangaroo)_ 

0.78 neural(X):-closeby(Player,Monkey). 

0.51 neural(X):-closeby(Player,Coconut). 0.11 logic(X):-nothing_around(Player). 

Figure 4: **BlendRL provides interpretable policies and blending modules as sets of weighted first-order logic rules.** A subset of the policy’s weighted action rules and the blending module’s rules from a BlendRL agent trained on _Kangaroo_ . All the logic rule sets for each environment are provided in Appendix A.4. 

## 3.3 LLM GENERATED LOGIC POLICIES 

BlendRL employs Language Models (LLMs) to generate symbolic programs for precise reasoning, following a chain of thought principle (Wei et al., 2022; Kojima et al., 2022): 

**(i)** it uses the task context and detectable objects description and implementation to create state predicates, 

**(ii)** it formulates action rules, with conjunctions of the generated predicates as body, 

**(iii)** it generates the predicates’ _valuations_ ( _i.e._ their python implemented functions). 

For steps **(ii)** and **(iii)** , we used the few-shot prompting approach, providing the LLM with an example logic ruleset obtained from NUDGE policies (Delfosse et al., 2023a). This circumvents the need for an expert to provide the set of logic rules, the used logic predicates, and their implementations, thus allowing users to effectively introduce inductive biases in natural language ( _cf._ Appendix A.3 and A.4 for additional details). A subset of the logic module of a BlendRL trained on _Kangaroo_ is provided in Figure 4. The associated weights of the LLM-generated rules have been adjusted to maximize performance. 

## 3.4 OPTIMIZATION 

We use the PPO actor-critic to train BlendRL agents, and also employ hybrid value functions. We compose the hybrid critics by computing values using both visual and object-centric states. 

**Mixed Value Function.** As the value function approximates the expected return, we did not use logic to encode it. However, BlendRL incorporates a hybrid value function, that uses both the subsymbolic ( _**x**_ ) and symbolic (or object-centric _**z**_ ) state representations. Given state _s_ = ( _**x** ,_ _**z**_ ), the value is defined as: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_BlendRL_A_Framework_for_Merging_Symbolic_and_Neural_Policy_Learning_images/iclr_2025_BlendRL_A_Framework_for_Merging_Symbolic_and_Neural_Policy_Learning.pdf-0006-16.png)


with _Vµ[CNN]_ a CNN, that share its convolutional layers with the neural actor (Schulman et al., 2017) and _Vω[OC]_ a small MLP on the object-centric state, and _β_ = _Bλ_ ( _**x** ,_ _**z**_ ), the blending weight. 

**Regularization.** To ensure the use of both neural and logic policies, we introduce a regularization term for BlendRL agents. To further enhance exploration, we penalize overly peaked blending distributions. This approach ensures that agents utilize both neural and logic policies without deactivating either entirely. 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_BlendRL_A_Framework_for_Merging_Symbolic_and_Neural_Policy_Learning_images/iclr_2025_BlendRL_A_Framework_for_Merging_Symbolic_and_Neural_Policy_Learning.pdf-0006-19.png)


This helps agents avoid suboptimal policies typically caused by neural policies due to the sparse rewards. 

6 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_BlendRL_A_Framework_for_Merging_Symbolic_and_Neural_Policy_Learning_images/iclr_2025_BlendRL_A_Framework_for_Merging_Symbolic_and_Neural_Policy_Learning.pdf-0007-00.png)


Figure 5: **BlendRL surpasses both purely neural and logic-based policies in environments requiring reasoning and reaction capabilities,** shown by its superior episodic returns after training (averaged over 3 seeded reruns). BlendRL outperforms the logic-based state-of-the-art agent ( _i.e._ NUDGE) across all tested environments. The purely neural PPO agents frequently get trapped in suboptimal policies, BlendRL leverages its integrated reasoning and reacting modules to tackle every aspect of the tasks. 

## 4 EXPERIMENTS 

We outline the benefits of BlendRL over purely neural or symbolic approaches, supported by additional investigations into BlendRL’s robustness to environmental changes. Furthermore, we examine the interactions between neural and symbolic components and demonstrate that BlendRL can generate faithful explanations. We specifically aim to answer the following research questions: 

**(Q1)** Can BlendeRL agents overcome both symbolic and neural agents’ shortcomings? 

**(Q2)** Can BlendRL can produce both neural and symbolic explanations for its action selection? 

**(Q3)** Are BlendRL agents robust to environmental changes? 

**(Q4)** How do the neural and symbolic modules interact to maximize BlendRL’s agents overall performances? 

Let us now provide empirical evidence for BlendRL’s ability to learn efficient and understandable policies, even without being provided with all the necessary priors to optimally solve the tasks. 

## 4.1 EXPERIMENTAL SETUP 

**Environments.** We evaluate BlendRL on Atari Learning Environments (Bellemare et al., 2013), the most popular benchmark for RL (particularly for relational reasoning tasks). For resource efficiency, we use the object centric extraction module of (Delfosse et al., 2024b). Specifically, in _Kangaroo_ , the agent needs to reach and climb ladders, leading to the captive joey, while punching incoming monkey antagonists, that try to stop it. In _Seaquest_ , the agent has to rescue divers, while shooting sharks and enemy submarines. It also needs to surface before the oxygen runs out. Finally, in _Donkey Kong_ the agent has to reach the princess at the top, while avoiding incoming barrels thrown by Donkey Kong. For additional details, _cf._ Appendix A.7. To further test BlendRL abilities to overcome the potential lack of concept necessary to solve task, we omitted some part of the game in our prompt for the LLM that generates the policy rules. Specifically, we kept out the facts that agents can punch the monkeys in _Kangaroo_ , can shoot the enemies in _Seaquest_ and can jump over the barrels in _DonkeyKong_ . To test robustness and isolate different abilities of the agents, we employ HackAtari (Delfosse et al., 2024a), that allow to customize the environments ( _e.g._ remove enemies). For these ablation studies, we provide details about the use modification at the relevant point of the manuscript. 

**Baselines.** We compare BlendRL to purely neural PPO agents. Both agent types incorporate the classic CNN used for Atari environments. Additionally, we evaluate NUDGE, that uses a pretrained neural PPO agent for searching viable policy rules (Delfosse et al., 2023a), SOTA among logic agents on Atari tasks. We train each agent types until all of them converge to a stable episodic return ( _i.e._ for 15K episodes for _Kangaroo_ and _DonkeyKong_ and 25K for _Seaquest_ ). For additional details, _cf._ Appendix A.6. 

7 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_BlendRL_A_Framework_for_Merging_Symbolic_and_Neural_Policy_Learning_images/iclr_2025_BlendRL_A_Framework_for_Merging_Symbolic_and_Neural_Policy_Learning.pdf-0008-00.png)


**----- Start of picture text -----**<br>
the agent<br>the next ladders<br>is above<br>are on the left of<br>the agent  the diver  the agent<br>is on the left of<br>the next ladder<br>**----- End of picture text -----**<br>



![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_BlendRL_A_Framework_for_Merging_Symbolic_and_Neural_Policy_Learning_images/iclr_2025_BlendRL_A_Framework_for_Merging_Symbolic_and_Neural_Policy_Learning.pdf-0008-01.png)


Figure 6: **Qualitative examples of explanations produced by BlendRL,** in the form of importance maps, computing with gradient-based attributions for both neural and logic policies. The logic-based module also allows BlendRL to generate textual interpretations, enhancing its transparency and interpretability. 

## 4.2 RESULTS AND ANALYSIS 

**Comparison to neural and neuro-symbolic agents (Q1).** Figure 5 illustrates the episodic returns of the BlendRL, NUDGE, and NeuralPPO agents across various Atari environments. BlendRL surpasses the logicbased state-of-the-art NUDGE baseline (Delfosse et al., 2023a) in all tested scenarios. In the _Kangaroo_ environment, which requires fewer intuitive actions due to the relatively small number of enemies, NUDGE shows fair performance, even if its are less able to punch the monkeys and avoid their thrown coconuts. However, in the other environments, populated with more incoming threats, where neural policy becomes critical for accurate controls, NUDGE lags behind. Additionally, the purely neural PPO agents often fall into suboptimal policies. For instance, in _Seaquest_ , surfacing without collecting any divers lead to negative reward. Neural PPO agents thus concentrate on shooting sharks to collect reward, but never refill their oxygen. In contrast, BlendRL effectively selects its logic module to collect divers and surface when needed and its neural module to efficiently align itself with the enemies and shoot them. Overall, BlendRL significantly outperforms both baselines across different environments, underscoring the efficacy of neuro-symbolic blending policies in efficiently harnessing both neural and symbolic approaches to policy learning. 

**BlendRL agents are interpretable and explainable (Q2).** BlendRL’s symbolic policies can easily be interpreted as they consist of a set of transparent symbolic weighted rules, as exemplified in Figure 4 for _Kangaroo_ . The interpretable blending module prioritizes neural agents in situations where finetuned control is needed ( _i.e._ for dodging an incoming deadly coconut or accurately placing oneself next to the monkey and punching it). Conversely, the logic module is in use when no immediate danger is present, _e.g._ as a path finding proxy. The logic rules for the other environments are provided in Appendix A.4. 

BlendRL also produces gradient-based explanations, as each component is differentiable. Logic-based explanations are computed using action gradients ( _i.e. ∂πϕ[logic]_ ( **z** ) _/∂_ **z** ) on the state atoms, that underline the most important object properties for the decision. Further, the integrated gradients method (Sundararajan et al., 2017) on the neural module provides importance maps. We can visualize these two explanations, merging them with the blending weight _β_ , as shown in Figure 6. The most important logic rules for the decision are also depicted. BlendRL provides interpretable rules for it reasoning part and importance maps. 

**BlendRL is robust to environmental changes (Q3).** Return Kangaroogarooaroo Seaquestquestuest DonkeyKongyKongKongg Deep agents usually learn “by heart” spurious correlaNeural PPO 0 _._ 0 _±_ 0 _._ 0 0 _._ 0 _±_ 0 _._ 0 0 _._ 0 _±_ 0 _._ 0 tions during training, and thus fail to generalize to unBlendRL **187** _±_ **120 113** _±_ **106 263** _±_ **124** seen environments, even on simple Atari games (FareTable 1: **Contrary to neural PPO, BlendRL is ro-** brother et al., 2018; Delfosse et al., 2024a). We used **bust to environmental changes.** Average returns HackAtari variations of the environment, to disable on the modified environments (over 10 seeds). 10 seeds). seeds). the enemies in _Kangaroo_ and _Seaquest_ and to disable the barrels in _DonkeyKong_ . We additionally used a modified _Kangaroo_ environment with relocated ladders. 

Return Kangaroogarooaroo Seaquestquestuest DonkeyKongyKongKongg Neural PPO 0 _._ 0 _±_ 0 _._ 0 0 _._ 0 _±_ 0 _._ 0 0 _._ 0 _±_ 0 _._ 0 BlendRL **187** _±_ **120 113** _±_ **106 263** _±_ **124** Table 1: **Contrary to neural PPO, BlendRL is robust to environmental changes.** Average returns on the modified environments (over 10 seeds). 10 seeds). seeds). 

8 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_BlendRL_A_Framework_for_Merging_Symbolic_and_Neural_Policy_Learning_images/iclr_2025_BlendRL_A_Framework_for_Merging_Symbolic_and_Neural_Policy_Learning.pdf-0009-00.png)


**----- Start of picture text -----**<br>
Kangaroo Seaquest Seaquest<br>1.0 1.0 1.0<br>neural<br>logic<br>0.5 0.5 0.5<br>0.0 0.0 0.0<br>episodic steps episodic steps training steps<br>selection ratio<br>action confidence 0<br>5M 10M 15M 20M<br>**----- End of picture text -----**<br>


Figure 7: **BlendRL uses both its neural and logic modules.** Through an episode, the highest action probability is depicted (left, top), with the selected module at this step (bottom). BlendRL makes use of both module types through an episode. It progressively modifies their selection ratio along training on Seaquest (right), as its progress allows it to uncover parts of the game with more enemies. 

As expected, our BlendRL agent can still complete the tasks (and thus collect reward) on these safer versions of the environments ( _cf._ Table 1). BlendRL agents indeed rely on the knowledge incorporated in their logic-based policies and blending modules, and only rely on their neural modules for control accurate skills ( _e.g._ aiming and shooting/punching), as we further show hereafter. 

**Neural and symbolic policy interactions in BlendRL (Q4).** We here investigate how the symbolic and neural policies interoperate. One concern of such an hybrid systems is its potential reliance on only one module. The system could indeed learn to _e.g._ only rely on its neural component ( _i.e. β ≈_ 1 _._ 0). Figure 7 (left and center) illustrates the outputs of BlendRL’s neural and logic policies for 1 _k_ steps ( _i.e._ 1-3 episodes). Specifically, the maximum value of the action distributions (max _a πϕ[logic]_ ( _a|st_ ) and max _a πθ[neural]_ ( _a|st_ )), at each time step _t_ is depicted (at the top), underlined by the importance weighting of each module (blue if logic is mainly selected ( _β ≈_ 0), red if neural is ( _β ≈_ 1)), indicating how much each module is used at each step on _Kangaroo_ and _Seaquest_ . Further, BlendRL’s agents can adapt the amount to which they select each component through training, as depicted on the right of this figure. As _Seaquest_ is a progressive environment, in which the agent first faces few enemies (thus mainly relying on its logic module to collect the divers), before progressively accessing states in which many more enemies are spawning ( _cf._ Figure 10), BlendRL agents first mainly relies on its logic module (blue), while progressively shifting this preference to its neural one (red) for accurately shooting enemies. These results demonstrate the efficacy of BlendRL’s policy reasoning and learning on neuro-symbolic hybrid representations, thereby enhancing overall performance. We provide a further analysis comparing neural and logic blending modules in Appendix A.8, highlighting that the logic-based blending module can utilize both policies effectively, resulting in better performance. 

Overall, our experimental evaluation has shown BlendRL’s agents ability to learn on several Atari environments that require both reasoning and reacting capabilities. We showed that they outperform the commonly used neural PPO baseline, as well as the state-of-the-art logic agents, NUDGE. We further demonstrated their ability to generalize to unseen scenarios, on slightly modified environments (compared to the training ones), and that they can efficiently alternate between the two module types to obtain policies that can both produce interpretations, quantifying the impact of each symbolic properties and of each pixel region. 

## 5 RELATED WORK 

Relational Reinforcement Learning (Relational RL) (Dzeroski et al., 2001; Kersting et al., 2004; Kersting & Driessens, 2008; Lang et al., 2012; Hazra & Raedt, 2023) has been developed to address RL tasks in relational domains by incorporating logical representations and probabilistic reasoning. BlendRL extends this approach by integrating differentiable logic programming with deep neural policies. The Neural Logic Reinforcement Learning (NLRL) framework (Jiang & Luo, 2019) was the first to incorporate Differentiable Inductive Logic Programming ( _∂_ ILP) (Evans & Grefenstette, 2018) into the RL domain. _∂_ ILP learns generalized logic rules from examples using gradient-based optimization. NUDGE (Delfosse et al., 2023a) extends this method by introducing neurally-guided symbolic abstraction, leveraging extensive work on 

9 

_∂_ ILP (Shindo et al., 2021b;a) to learn complex programs. INSIGHT (Luo et al., 2024) is another neurosymbolic framework that jointly learns structured states and symbolic policies, producing textual explanations. Unlike these approaches, which rely solely on symbolic policy representations, BlendRL integrates both neural and symbolic policies and trains them jointly. 

The integration of planners with RL has been explored to achieve deliberate thinking in policy learning. For example, RePReL (Kokel et al., 2021) decomposes multi-agent planning tasks using a planner and then employs reinforcement learning for each agent to solve subtasks. In these frameworks, planners for longterm (slow) reasoning are typically separate components. In contrast, BlendRL computes both symbolic and neural policies at the same level, allowing them to be learned jointly, thereby enhancing overall performance. Additionally, planners are often used in model-based RL to generate hypothetical experiences that improve value estimates (Sutton, 1991; Kaiser et al., 2019). In contrast, BlendRL incorporates the symbolic reasoning module directly into its policy, enabling joint learning with neural modules. 

Additionally, BlendRL is related to prior work using LLMs for program generation. For example, LLMs have been applied to generate probabilistic programs (Wong et al., 2023), Answer Set Programs (Ishay et al., 2023; Yang et al., 2023), differentiable logic programs (Shindo et al., 2024a), and programs for visual reasoning (Sur´ıs et al., 2023; Stani´c et al., 2024). Our symbolic policy representation is inspired by _situation calculus_ (Reiter, 2001), which is an established framework to describe states and actions in logic. 

## 6 CONCLUSION 

In this study, we introduced BlendRL, a pioneering framework that integrates symbolic and neural policies for reinforcement learning. BlendRL employs neural networks for reactive actions and differentiable logic reasoners for high-level reasoning, seamlessly merging them through a blending module that manages distributions over both policy types. We also developed a learning algorithm for BlendRL agents that hybridizes the state-value function on both pixel-based and object-centric states and includes a regularization approach to enhance the efficacy of both logic and neural policies. 

Our empirical evaluations demonstrate that BlendRL agents significantly outperform purely neural agents and state-of-the-art neuro-symbolic baselines in popular Atari environments. Furthermore, these agents exhibit robustness to environmental changes and are capable of generating clear, interpretable explanations across various types of reasoning, effectively addressing the limitations of purely neural policies. Our comprehensive analysis of the interactions between symbolic and neural policy representations highlights their synergistic potential to enhance overall performance. 

**Acknowledgements.** The authors thank Sriraam Natarajan for his valuable feedback on the manuscript. This research work was partly funded by the German Federal Ministry of Education and Research, the Hessian Ministry of Higher Education, Research, Science and the Arts (HMWK) within their joint support of the National Research Center for Applied Cybersecurity ATHENE, via the “SenPai: XReLeaS” project. It also benefited from the HMWK Cluster Project 3AI. We gratefully acknowledge support by the EU ICT48 Network of AI Research Excellence Center “TAILOR” (EU Horizon 2020, GA No 952215), and the Collaboration Lab “AI in Construction” (AICO). The Eindhoven University of Technology authors received support from their Department of Mathematics and Computer Science and the Eindhoven Artificial Intelligence Systems Institute. 

## REFERENCES 

Adri`a Puigdom`enech Badia, Bilal Piot, Steven Kapturowski, Pablo Sprechmann, Alex Vitvitskyi, Zhaohan Daniel Guo, and Charles Blundell. Agent57: Outperforming the atari human benchmark. In _Proceedings of the International Conference on Machine Learning (ICML)_ , 2020. 

10 

- Marc G. Bellemare, Yavar Naddaf, Joel Veness, and Michael Bowling. The arcade learning environment: An evaluation platform for general agents. _J. Artif. Intell. Res._ , 2013. 

- A. Bhatt, D. Palenicek, B. Belousov, M. Argus, A. Amiranashvili, T. Brox, and J. Peters. Crossq: Batch normalization in deep reinforcement learning for greater sample efficiency and simplicity. In _Proceedings of the International Conference on Learning Representations (ICLR)_ , 2024. 

- Yushi Cao, Zhiming Li, Tianpei Yang, Hao Zhang, Yan Zheng, Yi Li, Jianye Hao, and Yang Liu. GALOIS: boosting deep reinforcement learning via generalizable logic synthesis. In _Proceedings of the Conference on Advances in Neural Information Processing Systems (NeurIPS)_ , 2022. 

- Quentin Cappart, Thierry Moisan, Louis-Martin Rousseau, Isabeau Pr´emont-Schwarz, and Andre A Cire. Combining reinforcement learning and constraint programming for combinatorial optimization. In _Proceedings of the AAAI Conference on Artificial Intelligence (AAAI)_ , 2021. 

- Lili Chen, Kevin Lu, Aravind Rajeswaran, Kimin Lee, Aditya Grover, Misha Laskin, Pieter Abbeel, Aravind Srinivas, and Igor Mordatch. Decision transformer: Reinforcement learning via sequence modeling. _Proceedings of the Conference on Advances in Neural Information Processing Systems (NeurIPS)_ , 2021. 

- Quentin Delfosse, Hikaru Shindo, Devendra Dhami, and Kristian Kersting. Interpretable and explainable logical policies via neurally guided symbolic abstraction. In _Proceedings of the Conference on Advances in Neural Information Processing Systems (NeurIPS)_ , 2023a. 

- Quentin Delfosse, Wolfgang Stammer, Thomas Rothenbacher, Dwarak Vittal, and Kristian Kersting. Boosting object representation learning via motion and object continuity. In _Machine Learning and Knowledge Discovery in Databases: Research Track - European Conference (ECML PKDD)_ , 2023b. 

- Quentin Delfosse, Jannis Bl¨uml, Bjarne Gregori, and Kristian Kersting. Hackatari: Atari learning environments for robust and continual reinforcement learning. In _Workshop on Interpretable Policies in Reinforcement Learning @RLC-2024_ , 2024a. 

- Quentin Delfosse, Jannis Bl¨uml, Bjarne Gregori, Sebastian Sztwiertnia, and Kristian Kersting. Ocatari: Object-centric atari 2600 reinforcement learning environments. In _Proceedings of the First Conference on Reinforcement Learning (RLC)_ , 2024b. 

- Quentin Delfosse, Patrick Schramowski, Martin Mundt, Alejandro Molina, and Kristian Kersting. Adaptive rational activations to boost deep reinforcement learning. In _Proceedings of the International Conference on Learning Representations (ICLR)_ , 2024c. 

- Quentin Delfosse, Sebastian Sztwiertnia, Wolfgang Stammer, Mark Rothermel, and Kristian Kersting. Interpretable concept bottlenecks to align reinforcement learning agents. In _Proceedings of the Conference on Advances in Neural Information Processing Systems (NeurIPS)_ , 2024d. 

- Saso Dzeroski, Luc De Raedt, and Kurt Driessens. Relational reinforcement learning. _Machine Learning (MLJ)_ , 2001. 

- Richard Evans and Edward Grefenstette. Learning explanatory rules from noisy data. _Journal of Artificial Intelligence Research (JAIR)_ , 2018. 

- Jesse Farebrother, Marlos C Machado, and Michael Bowling. Generalization and regularization in dqn. _arXiv Preprint:1810.00123_ , 2018. 

- Rishi Hazra and Luc De Raedt. Deep explainable relational reinforcement learning: A neuro-symbolic approach. In _Machine Learning and Knowledge Discovery in Databases: Research Track - European Conference (ECML PKDD)_ , 2023. 

11 

- Lukas Helff, Wolfgang Stammer, Hikaru Shindo, Devendra Singh Dhami, and Kristian Kersting. V-lol: A diagnostic dataset for visual logical learning. _arXiv Preprint:2306.07743_ , 2023. 

- Matteo Hessel, Joseph Modayil, Hado Van Hasselt, Tom Schaul, Georg Ostrovski, Will Dabney, Dan Horgan, Bilal Piot, Mohammad Azar, and David Silver. Rainbow: Combining improvements in deep reinforcement learning. In _Proceedings of the AAAI Conference on Artificial Intelligence (AAAI)_ , 2018. 

- Shengyi Huang, Rousslan Fernand Julien Dossa, Chang Ye, Jeff Braga, Dipam Chakraborty, Kinal Mehta, and Jo˜ao G.M. Ara´ujo. Cleanrl: High-quality single-file implementations of deep reinforcement learning algorithms. _Journal of Machine Learning Research (JMLR)_ , 2022. 

- Adam Ishay, Zhun Yang, and Joohyung Lee. Leveraging large language models to generate answer set programs. In _Proceedings of the International Conference on Principles of Knowledge Representation and Reasoning (KR)_ , 2023. 

- Zhengyao Jiang and Shan Luo. Neural logic reinforcement learning. In _Proceedings of the International Conference on Machine Learning (ICML)_ , 2019. 

- Daniel Kahneman. _Thinking, fast and slow_ . Farrar, Straus and Giroux, New York, 2011. 

- Lukasz Kaiser, Mohammad Babaeizadeh, Piotr Milos, Blazej Osinski, Roy H Campbell, Konrad Czechowski, Dumitru Erhan, Chelsea Finn, Piotr Kozakowski, Sergey Levine, et al. Model-based reinforcement learning for atari. _arXiv Preprint:1903.00374_ , 2019. 

- Henry A. Kautz. The third ai summer: Aaai robert s. engelmore memorial lecture. _AI Magazine_ , 43(1): 105–125, 2022. 

- Kristian Kersting and Kurt Driessens. Non-parametric policy gradients: a unified treatment of propositional and relational domains. In _Proceedings of the International Conference on Machine Learning (ICML)_ , 2008. 

- Kristian Kersting, Martijn van Otterlo, and Luc DeRaedt. Bellman goes relational. In _Proceedings of the International Conference on Machine Learning (ICML)_ , 2004. 

- Daiki Kimura, Masaki Ono, Subhajit Chaudhury, Ryosuke Kohita, Akifumi Wachi, Don Joven Agravante, Michiaki Tatsubori, Asim Munawar, and Alexander Gray. Neuro-symbolic reinforcement learning with first-order logic. In _Proceedings of the Conference on Empirical Methods in Natural Language Processing (EMNLP)_ , 2021. 

- Diederik P. Kingma and Jimmy Ba. Adam: A method for stochastic optimization. In _Proceedings of the International Conference on Learning Representations (ICLR)_ , 2015. 

- Hector Kohler, Quentin Delfosse, Riad Akrour, Kristian Kersting, and Philippe Preux. Interpretable and editable programmatic tree policies for reinforcement learning. _arXiv Preprint:2405.14956_ , 2024. 

- Takeshi Kojima, Shixiang (Shane) Gu, Machel Reid, Yutaka Matsuo, and Yusuke Iwasawa. Large language models are zero-shot reasoners. In _Proceedings of the Conference on Advances in Neural Information Processing Systems (NeurIPS)_ , 2022. 

- Harsha Kokel, Arjun Manoharan, Sriraam Natarajan, Balaraman Ravindran, and Prasad Tadepalli. Reprel: Integrating relational planning and reinforcement learning for effective abstraction. In _Proceedings of the International Conference on Automated Planning and Scheduling (ICAPS)_ , 2021. 

- Vijay R. Konda and John N. Tsitsiklis. Actor-critic algorithms. In _Proceedings of the Conference on Advances in Neural Information Processing Systems (NeurIPS)_ , 1999. 

12 

- Brenden M. Lake, Tomer D. Ullman, Joshua B. Tenenbaum, and Samuel J. Gershman. Building machines that learn and think like people. _Behavioral and Brain Sciences_ , 2017. 

- Tobias Lang, Marc Toussaint, and Kristian Kersting. Exploration in relational domains for model-based reinforcement learning. _Journal of Machine Learning Research (JMLR)_ , 2012. 

- Zhixuan Lin, Yi-Fu Wu, Skand Vishwanath Peri, Weihao Sun, Gautam Singh, Fei Deng, Jindong Jiang, and Sungjin Ahn. SPACE: unsupervised object-oriented scene representation via spatial attention and decomposition. In _International Conference on Learning Representations_ , 2020. 

- Andrew Liu and Alla Borisyuk. A role of environmental complexity on representation learning in deep reinforcement learning agents. _arXiv Preprint:2407.03436_ , 2024. 

- Iou-Jen Liu, Zhongzheng Ren, Raymond A Yeh, and Alexander G Schwing. Semantic tracklets: An objectcentric representation for visual multi-agent reinforcement learning. In _2021 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS)_ , 2021. 

- Francesco Locatello, Dirk Weissenborn, Thomas Unterthiner, Aravindh Mahendran, Georg Heigold, Jakob Uszkoreit, Alexey Dosovitskiy, and Thomas Kipf. Object-centric learning with slot attention. In _Proceedings of the Conference on Advances in Neural Information Processing Systems (NeurIPS)_ , 2020. 

- Lirui Luo, Guoxi Zhang, Hongming Xu, Yaodong Yang, Cong Fang, and Qing Li. End-to-end neurosymbolic reinforcement learning with textual explanations. In _Proceedings of the International Conference on Machine Learning (ICML)_ , 2024. 

- Daoming Lyu, Fangkai Yang, Bo Liu, and Steven Gustafson. SDRL: interpretable and data-efficient deep reinforcement learning leveraging symbolic planning. In _Proceedings of the AAAI Conference on Artificial Intelligence (AAAI)_ , 2019. 

- Jiayuan Mao, Chuang Gan, Pushmeet Kohli, Joshua B. Tenenbaum, and Jiajun Wu. The neuro-symbolic concept learner: Interpreting scenes, words, and sentences from natural supervision. In _Proceedings of the International Conference on Learning Representations (ICLR)_ , 2019. 

- Volodymyr Mnih, Koray Kavukcuoglu, David Silver, Andrei A. Rusu, Joel Veness, Marc G. Bellemare, Alex Graves, Martin A. Riedmiller, Andreas Fidjeland, Georg Ostrovski, Stig Petersen, Charles Beattie, Amir Sadik, Ioannis Antonoglou, Helen King, Dharshan Kumaran, Daan Wierstra, Shane Legg, and Demis Hassabis. Human-level control through deep reinforcement learning. _Nature_ , 2015. 

- Volodymyr Mnih, Adri`a Puigdom`enech Badia, Mehdi Mirza, Alex Graves, Timothy P. Lillicrap, Tim Harley, David Silver, and Koray Kavukcuoglu. Asynchronous methods for deep reinforcement learning. In _Proceedings of the International Conference on Machine Learning (ICML)_ , 2016. 

- Emilio Parisotto, Francis Song, Jack Rae, Razvan Pascanu, Caglar Gulcehre, Siddhant Jayakumar, Max Jaderberg, Raphael Lopez Kaufman, Aidan Clark, Seb Noury, et al. Stabilizing transformers for reinforcement learning. In _Proceedings of the International Conference on Machine Learning (ICML)_ , 2020. 

- Joseph Redmon, Santosh Kumar Divvala, Ross B. Girshick, and Ali Farhadi. You only look once: Unified, real-time object detection. In _Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)_ , 2016. 

- Raymond Reiter. _Knowledge in Action: Logical Foundations for Specifying and Implementing Dynamical Systems_ . MIT Press, 2001. 

Stuart Russell and Peter Norvig. _Artificial intelligence - a modern approach_ . Prentice Hall, 2010. 

13 

- John Schulman, Filip Wolski, Prafulla Dhariwal, Alec Radford, and Oleg Klimov. Proximal policy optimization algorithms. _arXiv Preprint:1707.06347_ , 2017. 

- Jingyuan Sha, Hikaru Shindo, Kristian Kersting, and Devendra Singh Dhami. Neural-symbolic predicate invention: Learning relational concepts from visual scenes. In _Proceedings of the 17th International Workshop on Neural-Symbolic Learning and Reasoning (NeSy)_ , 2023. 

- Jingyuan Sha, Hikaru Shindo, Quentin Delfosse, Kristian Kersting, and Devendra Singh Dhami. EXPIL: explanatory predicate invention for learning in games. _arXiv Preprint:2406.06107_ , 2024. 

- Hikaru Shindo, Masaaki Nishino, and Akihiro Yamamoto. Using binary decision diagrams to enumerate inductive logic programming solutions. In _Up-and-Coming and Short Papers of the 28th International Conference on Inductive Logic Programming (ILP 2018)_ , 2018. 

- Hikaru Shindo, Devendra Singh Dhami, and Kristian Kersting. Neuro-symbolic forward reasoning. _arXiv Preprint:2110.09383_ , 2021a. 

- Hikaru Shindo, Masaaki Nishino, and Akihiro Yamamoto. Differentiable inductive logic programming for structured examples. In _Proceedings of the AAAI Conference on Artificial Intelligence (AAAI)_ , 2021b. 

- Hikaru Shindo, Viktor Pfanschilling, Devendra Singh Dhami, and Kristian Kersting. _α_ ilp: thinking visual scenes as differentiable logic programs. _Machine Learning (MLJ)_ , 2023. 

- Hikaru Shindo, Manuel Brack, Gopika Sudhakaran, Devendra Singh Dhami, Patrick Schramowski, and Kristian Kersting. Deisam: Segment anything with deictic prompting. In _Proceedings of the Conference on Advances in Neural Information Processing Systems (NeurIPS)_ , 2024a. 

- Hikaru Shindo, Viktor Pfanschilling, Devendra Singh Dhami, and Kristian Kersting. Learning differentiable logic programs for abstract visual reasoning. _Machine Learning (MLJ)_ , 2024b. 

- Wolfgang Stammer, Felix Friedrich, David Steinmann, Manuel Brack, Hikaru Shindo, and Kristian Kersting. Learning by self-explaining. _Transactions on Machine Learning Research (TMLR)_ , 2024a. 

- Wolfgang Stammer, Antonia W¨ust, David Steinmann, and Kristian Kersting. Neural concept binder. In _Proceedings of the Conference on Advances in Neural Information Processing Systems (NeurIPS)_ , 2024b. 

- Aleksandar Stani´c, Sergi Caelles, and Michael Tschannen. Towards truly zero-shot compositional visual reasoning with llms as programmers. _Transactions on Machine Learning Research (TMLR)_ , 2024. 

- Shao-Hua Sun, Te-Lin Wu, and Joseph J. Lim. Program guided agent. In _Proceedings of the International Conference on Learning Representations (ICLR)_ , 2020. 

- Mukund Sundararajan, Ankur Taly, and Qiqi Yan. Axiomatic attribution for deep networks. In _Proceedings of the International Conference on Machine Learning (ICML)_ , 2017. 

- D´ıdac Sur´ıs, Sachit Menon, and Carl Vondrick. Vipergpt: Visual inference via python execution for reasoning. In _Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)_ , 2023. 

- Richard S. Sutton. Dyna, an integrated architecture for learning, planning, and reacting. _SIGART Bull._ , 2 (4):160–163, 1991. 

- Abhinav Verma, Vijayaraghavan Murali, Rishabh Singh, Pushmeet Kohli, and Swarat Chaudhuri. Programmatically interpretable reinforcement learning. In _Proceedings of the International Conference on Machine Learning (ICML)_ , 2018. 

14 

- Jason Wei, Xuezhi Wang, Dale Schuurmans, Maarten Bosma, Brian Ichter, Fei Xia, Ed H Chi, Quoc V Le, and Denny Zhou. Chain-of-thought prompting elicits reasoning in large language models. In _Proceedings of the Conference on Advances in Neural Information Processing Systems (NeurIPS)_ , 2022. 

- Lionel Wong, Gabriel Grand, Alexander K. Lew, Noah D. Goodman, Vikash K. Mansinghka, Jacob Andreas, and Joshua B. Tenenbaum. From word models to world models: Translating from natural language to the probabilistic language of thought. _arXiv Preprint:2306.12672_ , 2023. 

- Antonia W¨ust, Wolfgang Stammer, Quentin Delfosse, Devendra Singh Dhami, and Kristian Kersting. Pix2code: Learning to compose neural visual concepts as programs. In _Proceedings of Conference on Uncertainty in Artificial Intelligence (UAI)_ , 2024. 

- Zhun Yang, Adam Ishay, and Joohyung Lee. Coupling large language models with logic programming for robust and general reasoning from text. In _Findings of the Association for Computational Linguistics (ACL)_ , 2023. 

- Zihan Ye, Hikaru Shindo, Devendra Singh Dhami, and Kristian Kersting. Neural meta-symbolic reasoning and learning. _arXiv Preprint:2211.11650_ , 2022. 

- Jaesik Yoon, Yi-Fu Wu, Heechul Bae, and Sungjin Ahn. An investigation into pre-training object-centric representations for reinforcement learning. In _Proceedings of the International Conference on Machine Learning (ICML)_ , 2023. 

- Andrii Zadaianchuk, Maximilian Seitzer, and Georg Martius. Self-supervised visual reinforcement learning with object-centric representations. In _Proceedings of the International Conference on Learning Representations (ICLR)_ , 2021. 

- Xu Zhao, Wenchao Ding, Yongqi An, Yinglong Du, Tao Yu, Min Li, Ming Tang, and Jinqiao Wang. Fast segment anything. _arXiv Preprint:2306.12156_ , 2023. 

15 

## A APPENDIX 

## A.1 DEITAILED BACKGROUND OF REINFORCEMENT LEARNING 

We provide more detailed background for reinforcement learning. Policy-based methods directly optimize _πθ_ using the noisy return signal, leading to potentially unstable learning. Value-based methods learn to approximate the value functions _V_[ˆ] _ϕ_ or _Q_[ˆ] _ϕ_ , and implicitly encode the policy, _e.g._ by selecting the actions with the highest Q-value with a high probability (Mnih et al., 2015). To reduce the variance of the estimated Q-value function, one can learn the advantage function _A_[ˆ] _ϕ_ ( _st, at_ ) = _Q_[ˆ] _ϕ_ ( _st, at_ ) _− V_[ˆ] _ϕ_ ( _st_ ). An estimate of the advantage function can be computed as _A_[ˆ] _ϕ_ ( _st, at_ ) =[�] _[k] i_ =0 _[−]_[1] _[γ][i][r][t]_[+] _[i]_[ +] _[γ][k]_[ ˆ] _[V][ϕ]_[(] _[s][t]_[+] _[k]_[)] _[−][V]_[ˆ] _[ϕ]_[(] _[s][t]_[)][ (Mnih et al.,] 2016). The Advantage Actor-critic (A2C) methods both encode the policy _πθ_ ( _i.e._ actor) and the advantage function _A_[ˆ] _ϕ_ ( _i.e._ critic), and use the critic to provide feedback to the actor, as in (Konda & Tsitsiklis, 1999).Eˆ[log _πθ_ To push( _a | s_ ) ˆ _A πϕθ_ ].to take actions that lead to higher returns,Proximal Policy Optimization (PPO) algorithms ensure minor policy updates that avoidgradient ascent can be applied to _L[P G]_ ( _θ_ ) = catastrophic drops (Schulman et al., 2017), and can be applied to actor-critic methods. To do so, the main objective constraints the policy ratio _r_ ( _θ_ ) = _ππθ_ old _θ_ ( _a_ ( _a|s|_ ) _s_ )[, following] _[ L][P R]_[(] _[θ]_[)][=][E][ˆ][[min(] _[r]_[(] _[θ]_[) ˆ] _[A][ϕ][,]_[ clip(] _[r]_[(] _[θ]_[)] _[,]_[ 1] _[ −]_ 

_ϵ,_ 1 + _ϵ_ ) _A_[ˆ] _ϕ_ )], where clip constrains the input within [1 _− ϵ,_ 1 + _ϵ_ ]. PPO actor-critic algorithm’s global objective is _L_ ( _θ, ϕ_ ) = E[ˆ] [ _L[P R]_ ( _θ_ ) _− c_ 1 _L[V F]_ ( _ϕ_ )], with _L[V F]_ ( _ϕ_ )=( _V_[ˆ] _ϕ_ ( _st_ ) _− V_ ( _st_ ))[2] being the value function loss. An entropy term can also be added to this objective to encourage exploration. 

## A.2 DETAILS OF DIFFERENTIABLE FORWARD REASONING 

We provide the details of differentiable forward reasoning. 

**Definition A.1** _A Forward Reasoning Graph is a bipartite directed graph_ ( _VG, V∧, EG→∧, E∧→G_ ) _, where VG is a set of nodes representing ground atoms (atom nodes), V∧ is set of nodes representing conjunctions (conjunction nodes), EG→∧ is set of edges from atom to conjunction nodes and E∧→G is a set of edges from conjunction to atom nodes._ 

BlendRL performs forward-chaining reasoning by passing messages on the reasoning graph. Essentially, forward reasoning consists of _two_ steps: (1) computing conjunctions of body atoms for each rule and (2) computing disjunctions for head atoms deduced by different rules. These two steps can be efficiently computed on bi-directional message-passing on the forward reasoning graph. We now describe each step in detail. 

**(Direction** _→_ **) From Atom to Conjunction.** First, messages are passed to the conjunction nodes from atom nodes. For conjunction node vi _∈V∧_ , the node features are updated: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_BlendRL_A_Framework_for_Merging_Symbolic_and_Neural_Policy_Learning_images/iclr_2025_BlendRL_A_Framework_for_Merging_Symbolic_and_Neural_Policy_Learning.pdf-0016-09.png)


where[�] is a soft implementation of _conjunction_ , and[�] is a soft implementation of _disjunction_ . Intuitively, probabilistic truth values for bodies of all ground rules are computed softly by Eq. 4. 

**(Direction** _←_ **) From Conjunction to Atom.** Following the first message passing, the atom nodes are then updated using the messages from conjunction nodes. For atom node vi _∈VG_ , the node features are updated: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_BlendRL_A_Framework_for_Merging_Symbolic_and_Neural_Policy_Learning_images/iclr_2025_BlendRL_A_Framework_for_Merging_Symbolic_and_Neural_Policy_Learning.pdf-0016-12.png)


16 

up(img) ∧ left(img) ∧ is_empty(img) ∧ above(diver,agent) left_of(diver,agent) 

Figure 8: Forward reasoning graph for rules in Listing 1. A reasoning graph consists of _atom nodes_ and _conjunction nodes_ , and is obtained by grounding rules _i.e._ , removing variables by, _e.g._ , `X` _←_ `obj1` , `Y` _←_ `obj2` . By performing bi-directional message passing on the reasoning graph using soft-logic operations, BlendRL computes logical consequences in a differentiable manner. Only relevant nodes are shown (Best viewed in color). 

where _wji_ is a weight of edge _ej→i_ . We assume that each rule _Ck ∈C_ has its weight _θk_ , and _wji_ = _θk_ if edge _ej→i_ on the reasoning graph is produced by rule _Ck_ . Intuitively, in Eq. 5, new atoms are deduced by gathering values from different ground rules and from the previous step. 

We used product for conjunction, and _log-sum-exp_ function for disjunction: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_BlendRL_A_Framework_for_Merging_Symbolic_and_Neural_Policy_Learning_images/iclr_2025_BlendRL_A_Framework_for_Merging_Symbolic_and_Neural_Policy_Learning.pdf-0017-04.png)


where _γ >_ 0 is a smooth parameter. Eq. 6 approximates the maximum value given input _x_ 1 _, . . . , xn_ . 

**Prediction.** The probabilistic logical entailment is computed by the bi-directional message-passing. Let **x**[(0)] _atoms[∈]_[[0] _[,]_[ 1]] _[|G|]_[be input node features, which map a fact to a scalar value,][ RG][ be the reasoning graph,] **[ w]** be the rule weights, _B_ be background knowledge, and _T ∈_ N be the infer step. For fact _Gi ∈G_ , BlendRL computes the probability as: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_BlendRL_A_Framework_for_Merging_Symbolic_and_Neural_Policy_Learning_images/iclr_2025_BlendRL_A_Framework_for_Merging_Symbolic_and_Neural_Policy_Learning.pdf-0017-07.png)


where **x**[(] _atoms[T]_[ )] _[∈]_[[0] _[,]_[ 1]] _[|G|]_[is][the][node][features][of][atom][nodes][after] _[T]_[-steps][of][the][bi-directional][message-] passing. 

By optimizing the cross-entropy loss, the differentiable forward reasoner can solve Inductive Logic Programming (ILP) problems with propositional encoding (Shindo et al., 2018), where the task is to find classification rules given positive and negative examples. It has been successfully applied to complex visual scenes (Helff et al., 2023), meta-level reasoning and learning (Ye et al., 2022), predicate invention (Sha et al., 2024; 2023), and self-explanatory learning (Stammer et al., 2024a). 

## A.3 BODY PREDICATES AND THEIR VALUATIONS 

We here provide examples of valuation functions for evaluating state predicates ( _e.g._ closeby, left ~~o~~ f, etc.) generated by LLMs (in Python). 

_# A subset of valuation functions for Kangaroo and DonkeyKong (generated by LLMs)_ **def** left_of(player: th.Tensor, obj: th.Tensor) -> th.Tensor: x = player[..., 1] obj_x = obj[..., 1] obj_prob = obj[:, 0] _# objectness_ **return** sigmoid(alpha < obj_x - x) * obj_prob * same_level_ladder(player, obj) **def** _close_by(player: th.Tensor, obj: th.Tensor) -> th.Tensor: player_x = player[:, 1] 

17 

player_y = player[:, 2] obj_x = obj[:, 1] obj_y = obj[:, 2] obj_prob = obj[:, 0] _# objectness_ x_dist = (player_x - obj_x). **pow** (2) y_dist = (player_y - obj_y). **pow** (2) dist = (x_dist + y_dist).sqrt() **return** sigmoid(dist) * obj_prob 

**def** on_ladder(player: th.Tensor, obj: th.Tensor) -> th.Tensor: player_x = player[..., 1] obj_x = obj[..., 1] **return** sigmoid( **abs** (player_x - obj_x) < gamma) ... 

_# A subset of valuation functions for Seaquest (generated by LLMs)_ **def** full_divers(objs: th.Tensor) -> th.Tensor: divers_vs = objs[:, -6:] num_collected_divers = th. **sum** (divers_vs[:,:,0], dim=1) diff = 6 - num_collected_divers **return** sigmoid(1 / diff) **def** not_full_divers(objs: th.Tensor) -> th.Tensor: **return** 1 - full_divers(objs) **def** above(player: th.Tensor, obj: th.Tensor) -> th.Tensor: player_y = player[..., 2] obj_y = obj[..., 2] obj_prob = obj[:, 0] **return** sigmoid( (player_y - obj_y) / gamma) * obj_prob ... 

## A.4 BLENDRL RULES 

We here provide the blending and action rules obtained by BlendRL on _Seaquest_ and _DonkeyKong_ . 

_% Blending rulset (Seaquest)_ 0.98 neural_agent(X):-close_by_enemy(P,E). 0.74 neural_agent(X):-close_by_missile(P,M). 0.02 logic_agent(X):-visible_diver(D). 0.02 logic_agent(X):-oxygen_low(B). 1.0 logic_agent(X):-full_divers(X). _% Policy rulset (Seaquest)_ 0.21 up_air(X):-oxygen_low(B). 0.22 up_rescue(X):-full_divers(X). 0.21 left_to_diver(X):-right_of_diver(P,D),visible_diver(D). 0.24 right_to_diver(X):-left_of_diver(P,D),visible_diver(D). 0.23 up_to_diver(X):-deeper_than_diver(P,D),visible_diver(D). 0.22 down_to_diver(X):-higher_than_diver(P,D),visible_diver(D). _% Blending ruleset (DonkeyKong)_ 0.92 neural_agent(X):-close_by_barrel(P,B). 0.28 logic_agent(X):-nothing_around(X). _% Policy ruleset (DonkeyKong)_ 0.88 up_ladder(X):-on_ladder(P,L),same_floor(P,L). 0.47 right_ladder(X):-left_of(P,L),same_floor(P,L). 0.18 left_ladder(X):-right_of(P,L),same_floor(P,L). 

18 

## A.5 EXPERIMENTAL DETAILS 

We here provide more details about our implementations. 

**Hardwares.** All experiments were performed on one NVIDIA A100-SXM4-40GB GPU with Xeon(R):8174 CPU@3.10GHz and 100 GB of RAM. 

**Value function update.** We update the value function and the policy as follows: In the following, we consider a (potentially pretrained) actor-critic neural agent, with _v_ _**ϕ**_ its differentiable state-value function parameterized by _ϕ_ (critic). Given a set of action rules _C_ , let _π_ ( _C,_ **W** ) be a differentiable logic policy. BlendRL learns the weights of the action rules in the following steps. For each non-terminal state _st_ of each episode, we store the actions sampled from the policy ( _at ∼ π_ ( _C,_ **W** )( _st_ )) and the next states _st_ +1. We update the value function and the policy as follows: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_BlendRL_A_Framework_for_Merging_Symbolic_and_Neural_Policy_Learning_images/iclr_2025_BlendRL_A_Framework_for_Merging_Symbolic_and_Neural_Policy_Learning.pdf-0019-04.png)


The logic policy _π_ ( _C,_ **W** ) thus learns to maximize the expected return. 

## A.6 TRAINING DETAILS 

We hereby provide further details about the training. Details regarding environments will be provided in the next section A.7. We used the Adam optimizer (Kingma & Ba, 2015) for all baselines. 

**BlendRL.** We adopted an implementation of the PPO algorithm from the CleanRL project (Huang et al., 2022). Hyperparameters are shown in Table 2. The object-centric critic is described in Table 3. We provide a pseudocode of BlendRL policy reasoning in Algorithm 1. 

||**Parameter**|**Parameter**|**Value**|**Explanation**|
|---|---|---|---|---|
|blend<br>~~e~~nt||~~c~~oef|0.01|Entropy coeffcient for blending regularization|
|||||(Eq. 3)|
|blender|~~l~~earning|~~r~~ate|0.00025|Learning rate for blending module|
||clip|~~c~~oef|0.1|Coeffcient for clipping gradients|
||ent|~~c~~oef|0.01|Entropy coeffcient for policy optimization|
|||_γ_|0.99|Discount factor for future rewards|
||learning|~~r~~ate|0.00025|Learning rate for neural modules|
|logic|~~l~~earning|~~r~~ate|0.00025|Learning rate for logic modules|
|max<br>~~g~~rad<br>~~n~~orm|||0.5|Maximum norm for gradient clipping|
||num|~~e~~nvs|512|Number of parallel environments|
||num<br>~~s~~teps||128|Number of steps per policy rollout|
|total<br>~~t~~imesteps|||20000000|Total number of trainingtimesteps|



Table 2: Hyperparameters for BlendRL training. 

|**Layer**|**Confguration**|
|---|---|
|Fully Connected Layer 1|Linear(_Nin_, 120)|
|Activation 1|ReLU()|
|Fully Connected Layer 2|Linear(120, 60)|
|Activation 2|ReLU()|
|FullyConnected Layer 3|Linear(60,_Nout_)|



Table 3: Object-centric critic networks for BlendRL. 

19 

## **Algorithm 1** BlendRL Policy Reasoning 

**Input:** _πθ[neural] , πϕ[logic] , Vµ[CNN] , Vω[OC]_ , blending function _B_ , state ( _**x** ,_ _**z**_ ) 1: _β_ = _B_ ( _**x** ,_ _**z**_ ) # Compute the blending weight _β_ 2: _action ∼ β · πθ[neural]_ ( _**x**_ ) + (1 _− β_ ) _· πϕ[logic]_ ( _**z**_ ) # Action is sampled from the mixed policy 3: _value_ = _β · Vµ[CNN]_ ( _**x**_ ) + (1 _− β_ ) _· Vω[OC]_ ( _**z**_ ) # Compute the state value using _β_ 4: **return** _action, value_ 

**NUDGE.** We used a public code[2] to perform experiments with the CleanRL training script. All hyperparameters are shared with the BlendRL agents, as described in Table 2. We used the same ruleset as BlendRL agents for NUDGE agents. The critic network on object-centric states is described in Table 3. 

**Neural PPO.** We used an implementation of the neural ppo algorithm[3] from the CleanRL project. The agent consists of an actor network and a critic network, sharing their weights except for the last layer. The base network shared by the actor and the critic is shown in Table. 4. A linear layer with non-shared weights follows for each actor and critic after on top of the base network. 

|**Layer**|**Confguration**|
|---|---|
|Convolutional Layer 1|Conv2d(4, 32, 8, stride=4)|
|Activation 1|ReLU()|
|Convolutional Layer 2|Conv2d(32, 64, 4, stride=2)|
|Activation 2|ReLU()|
|Convolutional Layer 3|Conv2d(64, 64, 3, stride=1)|
|Activation 3|ReLU()|
|Flatten Layer|Flatten()|
|Fully Connected Layer|Linear(64 * 7 * 7, 512)|
|Activation 4|ReLU()|



Table 4: Layer configuration of the neural ppo agent. 

## A.7 ENVIRONMENT DETAILS 

We hereby provide details of the environments we used in our experiments. We used HackAtari[4] , a framework that offers modifications of Atari environments to simply them or change them for the robustness test. In our experiments, the modifications we used are shown in Table 5. 

|**Environment**|**Option**|||**Explanation**|
|---|---|---|---|---|
|Kangaroo|disable|~~f~~alling|~~c~~oconut|Disable the falling coconut|
||change|~~l~~evel0||The frst stage is repeated|
||random|~~p~~osition||Randomize the starting position|
|Seaquest|No option||||
|DonkeyKong|change|~~l~~evel0||The frst stage is repeated|
||random|~~p~~osition||Randomize the starting position|



Table 5: Options and explanations for different environments 

> 2https://github.com/k4ntz/NUDGE 

> 3https://github.com/vwxyzjn/cleanrl/blob/master/cleanrl/ppo_atari.py 

> 4https://github.com/k4ntz/HackAtari 

20 

## A.8 ABLATION STUDY: NEURAL VS. LOGIC BLENDING MODULE 

We here provide an ablation study on the logic blending module, reruning BlendRL agents using a neural one on _Kangaroo_ and _Seaquest_ . As shown in Table 6, the agents that encompass logic-based blending modules outperform the neural ones. 

Figure 9 presents the entropies of the blending weights. An entropy value of 1.0 signifies that neural and logic policies are equally prioritized (with each receiving a weight of 0.5), while an entropy value of 0.0 indicates that only one policy is active, with the other being completely inactive. In both environments, the logic blending module consistently produced higher entropy values for the blending weights, indicating effective utilization of both neural and symbolic policies without overfitting to either one. 

|Episodic Return<br>Neural Blending<br>Logic Blending|Kangaroo<br>91_._6_±_43_._6<br>**186**_±_**12**|Seaquest<br>17_._8_±_0_._55<br>**39**_._**9**_±_**7**_._**12**|
|---|---|---|



Table 6: **Comparison: Neural v.s. Logic for policy blending.** Average returns over 10 different random seeds are shown of trained agents. The logic blending module outperforms neural one consistently. 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_BlendRL_A_Framework_for_Merging_Symbolic_and_Neural_Policy_Learning_images/iclr_2025_BlendRL_A_Framework_for_Merging_Symbolic_and_Neural_Policy_Learning.pdf-0021-05.png)


**----- Start of picture text -----**<br>
Entropies over Blending Weights<br>0.5 logic bl.<br>neural bl.<br>0.4<br>0.3<br>0.2<br>0.1<br>kangaroo seaquest<br>Environment<br>Entropy<br>**----- End of picture text -----**<br>


Figure 9: **Logic blending can keep both policies effective.** Entropies over blending weights are shown. 

## A.9 PROGRESSIVE ENVIRONMENT ILLUSTRATION 

As explained by Delfosse et al. (2024c), _Seaquest_ is a progressive environment in which the agent first needs to master easy tasks, before being provided with more complex ones in newly unlocked parts of the environment, reflected by the number of enemy to be shot here. 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_BlendRL_A_Framework_for_Merging_Symbolic_and_Neural_Policy_Learning_images/iclr_2025_BlendRL_A_Framework_for_Merging_Symbolic_and_Neural_Policy_Learning.pdf-0021-09.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_BlendRL_A_Framework_for_Merging_Symbolic_and_Neural_Policy_Learning_images/iclr_2025_BlendRL_A_Framework_for_Merging_Symbolic_and_Neural_Policy_Learning.pdf-0021-10.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_BlendRL_A_Framework_for_Merging_Symbolic_and_Neural_Policy_Learning_images/iclr_2025_BlendRL_A_Framework_for_Merging_Symbolic_and_Neural_Policy_Learning.pdf-0021-11.png)


Figure 10: **Seaquest is a progressive environment.** The more an agent collects points/reward (depicted at the top), the more enemies are spawning 

21 

