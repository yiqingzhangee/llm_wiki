# **Target Languages (vs. Inductive Biases) for Learning to Act and Plan** 

## **Hector Geffner** 

Universitat Pompeu Fabra, Barcelona, Spain Instituci´o Catalana de Recerca i Estudis Avanc¸ats (ICREA), Barcelona, Spain Link¨oping University, Link¨oping, Sweden hector.geffner@upf.edu 

## **Abstract** 

Recent breakthroughs in AI have shown the remarkable power of deep learning and deep reinforcement learning. These developments, however, have been tied to specific tasks, and progress in out-of-distribution generalization has been limited. While it is assumed that these limitations can be overcome by incorporating suitable inductive biases, the notion of inductive biases itself is often left vague and does not provide meaningful guidance. In the paper, I articulate a different learning approach where representations do not emerge from biases in a neural architecture but are learned over a given target language with a known semantics. The basic ideas are implicit in mainstream AI where representations have been encoded in languages ranging from fragments of first-order logic to probabilistic structural causal models. The challenge is to learn from data, the representations that have traditionally been crafted by hand. Generalization is then a result of the semantics of the language. The goals of this paper are to make these ideas explicit, to place them in a broader context where the design of the target language is crucial, and to illustrate them in the context of learning to act and plan. For this, after a general discussion, I consider learning representations of actions, general policies, and subgoals (“intrinsic rewards”). In these cases, learning is formulated as a combinatorial problem but nothing prevents the use of deep learning techniques instead. Indeed, learning representations over languages with a known semantics provides an account of _what_ is to be learned, while learning representations with neural nets provides a complementary account of _how_ representations can be learned. The challenge and the opportunity is to bring the two together. 

## **Introduction** 

A number of recent breakthroughs have shown the remarkable power of deep learning and deep reinforcement learning (LeCun, Bengio, and Hinton 2015; Bengio, Lecun, and Hinton 2021). These developments, however, have been tied to specific tasks like Chess, Go, or Atari games (Mnih et al. 2015; Silver et al. 2017a,b). Progress in out-of-distribution generalization or in the generation of modular components that can be assembled dynamically for different tasks, has been more limited (Lake et al. 2017; Darwiche 2018; Marcus 2018; Geffner 2018). 

Copyright © 2022, Association for the Advancement of Artificial Intelligence (www.aaai.org). All rights reserved. 

While it is assumed that these limitations can be overcome by adding suitable inductive biases in current neural network architectures (Garnelo and Shanahan 2019; Goyal and Bengio 2020), the notion of inductive biases itself is often left vague and does not always provide meaningful guidance. Traditionally, inductive biases refer to biases in the hypothesis space, and in the case of neural networks, to the structure of the parametric function captured by the architecture. More recently, the notion has been grounded on the invariant properties of such functions (Bronstein et al. 2021), but more often they are used to refer to intuitions that are not spelled out in formal detail and are not explicitly evaluated. 

In this paper, I aim to articulate a more abstract approach to representation learning where the learned representations are not those that emerge after training a neural network, but those that result over a given _target representation language with a well understood semantics._ The approach is implicit in mainstream, symbolic AI, from McCarthy’s observations about the representation of general abstractions (McCarthy 1960), to Pearl’s emphasis on the language required for reasoning about causality (Pearl and Mackenzie 2018). The challenge is to learn from data the representations that have traditionally been crafted by hand without having to appeal to background knowledge. 

The goals of the paper are to make the ideas behind the language-based approach to representation learning explicit, to place them in a broader context where the design of the target language is critical, and to illustrate them in the setting of learning to act and plan. For this, after a general discussion, I consider the problems of learning actions, general policies, and problem decompositions over suitable domainindependent languages. 

## **Languages, Semantics, Generality** 

_It is hard to find a needle in a haystack, but it helps to know what a needle looks like._ J. Pearl[1] 

In _Generality in AI_ , John McCarthy (1987), one of the founders of the field, quotes an early paper that says that “If one wants a machine to be able to discover an abstraction, it seems most likely that the machine must be able to represent this abstraction in some relatively simple way” (McCarthy 

1Pearl’s quotes are from his twitter feed unless otherwise noted; http://web.cs.ucla.edu/ _[∼]_ kaoru/jp-tweets. 

1960). From this and the need to use the learned abstractions in a _flexible way_ , McCarthy concludes that the representations have to be expressed in a logical language. 

Sixty years later, Yoshua Bengio, a leader in the field of deep learning interested in bridging the gap between deep learning and high-level reasoning, addresses similar issues but in slightly different terms:[2] “Systematic generalization is hypothesized to arise from an efficient factorization of knowledge into recomposable pieces corresponding to reusable factors ...This is related yet different in many ways from symbolic AI” (Goyal and Bengio 2020). 

Bengio’s point that the research agenda that he describes for capturing high-level reasoning is “related yet different” than McCarthy’s (symbolic AI) is certainly correct. The claim in this paper, however, is that there is much to be gained by making the two research agendas _complementary_ . In other words, symbolic AI has developed families of formal languages for “factorizing knowledge into reusable pieces” with the right semantics for supporting composition and generalization. The limitation of symbolic AI is not in the languages themselves but in their use by humans, which as Bengio says, does not scale. The challenge and the opportunity is to learn the representations over such languages (i.e., symbolic representations) directly from data. 

Pearl’s quote at the beginning of the section refers to representations for causal reasoning. I aim to illustrate that his point is more general and applies to all representations that must be learned and combined. The “needles” are the representations sought, and we know how they look like when they are representations over known languages. Interestingly, Bengio also finds language relevant for capturing the abstractions that are required for combinatorial generalization, but “language” for him, as for others, is natural language, not a formal language with a semantics. 

## **Example** 

_Toy problems is where you learn if you are on the right track. Non-toy problems is when you hide you don’t know which track you are on_ . J. Pearl 

A simple toy problem will help us to make the discussion of structural generalization concrete. Figure 1 shows the Minigrid environment; a benchmark introduced for learning to interpret and achieve goal instructions (ChevalierBoisvert et al. 2019). As shown in the figure, a goal may be: “pick up the grey box behind you, then go to grey key and open the door”. The agent is the red triangle and the limited field of view is displayed in light-grey. The general problem is to learn a _controller_ for the agent that accepts _goals_ and _observations_ , and outputs the _action_ to do in each step for reaching the goals. 

The Minigrid environment is similar to a classical planning problem (Geffner and Bonet 2013; Ghallab, Nau, and Traverso 2016) except that the action model and the goal language are not given. Both supervised and unsupervised approaches have been tried, and success has been partial (Chevalier-Boisvert et al. 2019; Chevalier-Boisvert, 

2From the abstract at https://ijcai-21.org/invited-talks. 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_Language_Models_Can_Articulate_Their_Implicit_Goals_images/iclr_2025_Language_Models_Can_Articulate_Their_Implicit_Goals.pdf-0002-09.png)


Figure 1: Minigrid environment and a possible goal: _pick up the grey box behind you, then go to the grey key and open door._ The agent is the red triangle; the light-grey area is his field of view. Problem is similar to a “classical planning problem” _except_ that the domain predicates and action schemas are not known. In addition, goals are to be achieved reactively from learned general policy, not by planning (Chevalier-Boisvert et al. 2019). 

Willems, and Pal 2018): millions of trials are required to achieve the given goals, and even then success rates are not 100%. For improving performance, intuitions leading to alternative architectures and loss functions are introduced (e.g., presence of objects or sparse interactions) which are then evaluated experimentally in relation to baselines. From a methodological point of view, this is not entirely satisfying, and two key questions are _what_ is it that we are trying to learn, and how this object can be characterized mathematically, independently of its computation. A step in this direction is to notice that we look for a mapping from _goals G_ into general policies _πG_ for achieving them in a broad range of situations involving any number of objects at any locations, and hence, different state spaces. 

Even this toy example is a hard problem given how little is known a priori. The surprise is not that supervised and unsupervised DL approaches struggle in the problem but that they manage to generate a meaningful behavior at all. The immediate goal for us is not to do better in the task but to identify the building blocks that are needed for approaching this and other problems in a meaningful way. Key questions from the perspective of language-based representation learning are what is a good, domain-independent language for expressing the _general policies πG_ , and how can representation in that language be learned? Related questions are what is a good, domain-independent language for expressing the _dynamics_ of Minigrid and how can representations over such a language be learned? The answers to these questions, to be sketched below, do not have to get in the way of developing neural architectures for solving these problems; the hope is that the answers can inform our understanding of the problems and their solutions, neural and otherwise. 

## **Causal Models** 

_My greatest challenge was to break away from probabilistic thinking and accept, first, that people are not_ 

_probability thinkers but cause-effect thinkers and, second, that causal thinking cannot be captured in the language of probability; it requires a formal language of its own._ J. Pearl[3] 

One formal language that is making it into mainstream ML is the language of (structural) causal models (SCMs). This owes to the work of Judea Pearl and others that has revolutionized our understanding of causality by articulating a simple formal language and a semantics to talk about causes and effects (Pearl 2009; Pearl and Mackenzie 2018). The language accommodates observations, interventions, and counterfactuals. A _structural causal model (SCM)_ can be understood as a deterministic Bayesian network (conditional probabilities are all one or zero) that defines not just one joint probability distribution over the variables but many. A SCM handles interventions (actions) of the form _do_ ( _X_ = _x_ ), by which a variable _X_ is set to a specific value _x_ , and the probability distribution that results from such actions is the distribution that is encoded by the “mutilated” Bayesian network where the parents of variable _X_ are replaced by the single parent _do_ ( _X_ = _x_ ) for which _P_ ( _X_ = _x|do_ ( _X_ = _x_ )) = 1. The answers to queries about combinations of observations, interventions, and counterfactuals are determined by a SCM once the priors on “exogenous” variables (those without parents) are given. The language of SCMs has been used, for example, to determine the conditions under which the answer to queries in one causal model “generalize” or “transport” to another causal model (Pearl and Bareinboim 2011). 

In principle, structural causal models can be learned from data, and this is a very active line of research, both in the cases where the variables in the model are given, and when they are not (Sch¨olkopf et al. 2021). This does not mean, however, that structural causal models can be bypassed when answering queries from data. In order to do that meaningfully, the design of the algorithms must take the semantics of SCMs into account (Pearl 2021). Doing this, model-free, while ignoring the language and semantics of SCMs runs the risk of reinventing the wheel with not much guidance, as experimental evaluations are no substitutes for a meaningful theory. This does not rule out the possibility of learning causal models using neural nets by stochastic gradient descent, but the architecture and loss functions must be aligned with the representations sought. 

## **Languages, Models, and Solvers** 

_Every science that has thriven, has thriven upon its own symbols._ De Morgan (1864), quoted by J. Pearl.[4] 

Bayesian networks and structural causal models are models that make predictions from knowledge expressed in terms of variables, graphs, and probabilities. In AI, other languages and models have been developed as well some of which are relevant to our focus on actions and planning. For 

> 3http://www.cambridgeblog.org/2012/07/qa-with-judea-pearlpart-one. 

4(Pearl and Mackenzie 2018). 

example, classical planning refers to planning with deterministic actions with known effects and preconditions, from a known initial state, given a compact encoding of the actions in terms of state variables. These encodings have a size that is polynomial in the number of variables but result in state models of exponential size. Compact languages for representing other state models like MDPs and POMDPs, have also been developed. 

The use of languages for encoding state models has been motivated by two reasons. First, state models need to be specified in a concise manner, as they would not fit in memory otherwise. Second, it is assumed that a compact specification reveals structure that can be exploited computationally. For example, a common technique for solving classical planning problems is using heuristic functions, yet these heuristics can be extracted from compact representations but not from flat models (McDermott 1999; Bonet and Geffner 2001). 

The benefits of languages supporting compact action representations, however, go well beyond the facilities that they provide for model specification and computation, as they also provide the ingredients needed for _generalization, transfer, and knowledge reuse._ Indeed, these languages have been designed for human to use with these goals in mind: when writing the description of a planning problem, we want the description to be reusable with minor modification in similar problems. The use of _first-order languages_ for referring to objects and relations has been essential for this purpose. 

Consider for example, a simplification of the Minigrid domain, that we refer to as Delivery, where there are _N_ objects in a grid _n × m_ , and the goal is to pick up the objects, one by one, and deliver them to a target cell. The actions available are to pick up and drop an object, and to move one cell at a time. Different instances of the Delivery domain are encoded in planning languages such as PDDL in two parts as _P_ = _⟨D, I⟩_ . One part, the domain _D_ , encodes what is common about all the Delivery instances in terms of three _action schemas_ : _pick_ , _drop_ , and _move_ . The other part, _I_ , details the objects in the instance, the ground atoms that are initially true, and those that must be made true in the goal (McDermott 2000; Haslum et al. 2019). For example, the action schema _pick_ can be defined as: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_Language_Models_Can_Articulate_Their_Implicit_Goals_images/iclr_2025_Language_Models_Can_Articulate_Their_Implicit_Goals.pdf-0003-12.png)


where _o_ , _x_ , and _y_ are the schema arguments, and preconditions and effects are atoms formed by predicate symbols and some of the schema arguments. The schema for _move_ can be expressed in turn as: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_Language_Models_Can_Articulate_Their_Implicit_Goals_images/iclr_2025_Language_Models_Can_Articulate_Their_Implicit_Goals.pdf-0003-14.png)


where _atr_ and _at_ are the predicates that encode the location of the agent and the packages, and _adjacent_ encodes 

the grid topology. These _pick_ and _move_ action schemas, along with the _drop_ scheme and the predicates involved in them, are precisely the “reusable pieces” over all Delivery instances, and hence if we want to learn a dynamic model from some instances that generalizes to other instances, we will be well advised to learn a representation of this type. 

There are indeed important similarities between planning languages and structural causal models: SCMs provide a _compact and invariant description_ of the effects of interventions on probability distributions, while planning languages provide a _compact and invariant description_ of the effects of interventions on states. Compact, first-order languages for defining probabilistic graphical models, MDPs, and POMDPs have also been developed (Raedt et al. 2016; van den Broeck et al. 2021; Younes et al. 2005; Sanner 2010), and if we want to learn _models that generalize_ , such languages would be good targets for learning as well. 

## **Languages vs. Inductive Biases** 

_It’s hard to understand why we should struggle to understand deep learning instead of learning deep understanding_ . J. Pearl 

There is a compelling reason for why learning approaches are either model-free and learn no models, or are modelbased but do not learn language-based representations (i.e., do not learn symbolic representations). The reason is that learning such representations appears to require humans in the loop, something that gets in the way of the automated learning pipeline. This impression, however, is wrong: while _languages_ like those of SCMs and planning have been developed to be used by humans, this does not mean that the _representations_ over such languages can only be provided by humans and cannot be learned from data. Certainly, there are obstacles to overcome for achieving this and a key one is the identification of the _(state) variables_ from unstructured data, but this is a technical problem that can be solved. I’ll show that there are indeed crisp solutions to this problem that exploit the natural inductive biases of language-based representations. 

Current DRL approaches can learn in principle policies that solve problems such as Delivery or Minigrid for any value of the parameters, but even then, it is not clear why this is so. For representations learned over languages designed to support modularity and reuse, the answers to these questions follow from their semantics. 

While the use of formal languages for learning representations is not common in deep learning, there is an increasing trend to reflect intuitions about the representations sought in the architectures. For example, RIM networks assume a dynamics determined by sparse object interactions (Goyal et al. 2020). Yet, informal talk of sparse interactions is no substitute for a language with a clear semantics that can represent the range of possible sparse interactions and lead to representations that can be understood in that way. 

As mentioned before, suitable target languages yield meaningful learning biases, as generalization is most often the result of learning compact descriptions. In Bayesian networks, compactness comes from sparse graphs, while in 

SCMs, compactness comes from the language and semantics of interventions. In planning, compact descriptions result from the language of action schemas and the predicates used in them. Compact descriptions are easier to learn and afford a powerful generalization, implying that representation size is a key bias in language-based representations learning. While there is no similar notion in deep learning, the “right” biases in neural nets would be the ones that deliver compact and reusable representations of this type. 

## **Related Research** 

Language-based representations, most often first order, are at the heart of the models and solvers studied in AI (Geffner 2014). The problem of learning such representations from data is active in some of these settings, like learning causal representations, mentioned above, and learning general action models, to be discussed below. Methods designed to learn symbolic representations from data provide a natural way for integrating learning and reasoning (Konidaris, Kaelbling, and Lozano-Perez 2018; Evans et al. 2021). Neurosymbolic methods make use of prior symbolic knowledge (Serafini and Garcez 2016; Yang, Ishay, and Lee 2020), possibly in terms of first-order probabilistic models (Raedt et al. 2016; Manhaeve et al. 2021). The use of logical languages has been central in recent characterizations of the expressive power of graph neural networks (Barcel´o et al. 2020; Grohe 2020). Domain specific task languages have been used in a number of settings (Lake et al. 2017; Silver et al. 2020; Tsividis et al. 2021), including the dynamics of MDPs (Diuk, Cohen, and Littman 2008). These languages are domain specific in the sense that they assume a particular vocabulary. Language-based representations of rewards have been explored too (Camacho et al. 2019; De Giacomo et al. 2019). Some deep learning approaches aim to approximate first-order formulas like conjunctions of atoms (Shanahan et al. 2020) or rules (Goyal et al. 2021), while others draw intuitions from them (Garnelo and Shanahan 2019; Goyal and Bengio 2020). Generally, symbolic methods like those considered in inductive logic programming (Muggleton and De Raedt 1994) assume and exploit background knowledge, while deep learning approches do not use and do not produce background knowledge; i.e., knowledge that can be reused. This is their advantage and also their limitation. 

## **Action models, policies, and decompositions** 

We consider next three concrete examples of domainindependent languages for acting and planning, and how representation over them can be learned. 

## **Languages for general action models** 

Classical planning problems _P_ = _⟨D, I⟩_ are described in terms of a planning domain _D_ involving action schemas and predicates, and instance information _I_ detailing the objects, the initial situation, and the goal. A planning instance _P_ defines a unique graph _G_ ( _P_ ) where the nodes _s_ stand for the states over _P_ and edges ( _s, s[′]_ ) express that there is a ground action _a_ in _P_ that maps the state _s_ into _s[′]_ . The states _s_ are the possible truth valuations over the ground atoms in _P_ . 

The general _model learning problem over this language_ can be expressed then as the following inverse problem (Bonet and Geffner 2020): 

Given plain graphs _G_ 1 _, . . . , Gk_ , find the simplest domain _D_ and instances _Pi_ = _⟨D, Ii⟩_ over _D_ such that the given graphs _Gi_ and _G_ ( _Pi_ ) are isomorphic, _i_ = 1 _, . . . , k_ . 

Variations of this problem have also been considered where the edges in the input graphs _Gi_ are labeled with action types (e.g., _pick_ , _drop_ , _move_ ), and edges may be missing or observations may be noisy (Rodriguez et al. 2021). The domains learned from some instances can then be used to predict the effects of actions in other, unseen instances. This learning formulation has been used to obtain the predicates and action schemas for domains such Blocks, Tower of Hanoi, and others. For example, the following domain description for Hanoi is learned from a single graph, produced by an instance with 3 disks (predicate and variable names are ours): 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_Language_Models_Can_Articulate_Their_Implicit_Goals_images/iclr_2025_Language_Models_Can_Articulate_Their_Implicit_Goals.pdf-0005-03.png)


Sta: _neq_ ( _d, to_ ), _neq_ ( _d, fr_ ), _neq_ ( _to, fr_ ), _¬larger_ ( _to, d_ ) Pre: _¬p_ ( _to, d_ ), _¬p_ ( _fr, fr_ ), _p_ ( _d, d_ ), _p_ ( _to, to_ ), _p_ ( _fr, d_ ) Eff: _¬p_ ( _to, to_ ), _¬p_ ( _fr, d_ ), _p_ ( _t, do_ ), _p_ ( _fr, fr_ ) . 

The domain learned can be shown to be correct for instances of any size (any number of disks and pegs), and use the predicate _p_ ( _x_ 1 _, x_ 2) for two different purposes: for capturing the relation _on_ ( _x_ 2 _, x_ 1) when _x_ 2 = _x_ 1, and for capturing clear( _x_ 1) otherwise. The three schema arguments _d_ , _fr_ , _to_ represent disks, and “Sta(tic)” refers to precondition atoms that are not affected by the actions. One can actually test that this domain description _D_ is correct experimentally, using (validation) graphs _G_ obtained from other instances, and checking if there is an _I_ such that _G_ and _G_ ( _P_ ) for _P_ = _⟨D, I⟩_ are isomorphic (a simpler version of the learning problem above). The learned representation also identifies the _state variables_ of the problem through the _p_ ( _d, d[′]_ ) atoms that encode the location of each of the disks _d_ . As discussed by Bonet and Geffner (2020), the use of a first-order target language with action schemas is critical for learning such state variables, as propositional representations cannot be reused in the same way and hence do not admit the same learning bias. 

The language of action models (action schemas and predicates) is suitable for learning in this setting, not just because it supports representations that generalize to other instances, but also because it defines a _heavily biased hypothesis space_ , with the space of possible domains being characterized by a small number of parameters with small integer values, like the number of action schemas and predicates and their arities. Provided with bounds on these values, the learning problem becomes a combinatorial optimization problem that can be solved in a number of ways, in many cases optimally. The optimization criterion used by Rodriguez et al. (2021), for example, minimizes the sum of actions and arities. Asai (2019), on the other hand, learn first-order action representations using deep learning, while Cresswell, McCluskey, and West (2013) learn first-order representations heuristi- 

cally assuming that action arguments in state transitions are observable. Many other works learn similar representations but assuming that the domain predicates are known. 

## **Languages for general policies** 

The target languages for learning in many settings can be taken off the shelf like the languages for representing actions and causal models discussed above. But for other tasks, new domain-independent languages with the right properties may have to be created. For example, in the Minigrid problem, DRL approaches are not after general dynamic models, but after general policies: policies that can deal with _any_ instance of the domain. What is then a good language for representing such policies that is not tied to this particular domain? This question has been considered in the area of _generalized planning_ (Srivastava, Immerman, and Zilberstein 2008; Hu and De Giacomo 2011), and the language below follows the one introduced by Bonet and Geffner (2018). 

A general policy _π_ for a class of domain instances _Q_ is a set of _policy rules_ of the form _C �→ E_ where _C_ contains boolean conditions of the form _p_ , _¬p_ , _n_ = 0, or _n >_ 0, and _E_ contains effects of the form _p_ , _¬p_ , _p_ ?, _n↓_ , _n↑_ , _n_ ?, where _p_ and _n_ stand for boolean and numerical _features_ . Features are functions over states. Boolean features _p_ can have value _true_ or _false_ , and numerical features _n_ can have any nonnegative integer value. Conditions in _C_ like _p_ and _n_ = 0 are true in a state when _p_ has value true, and _n_ has value 0 respectively, and effects in _E_ like _p_ ( _¬p_ ), _n↓_ ( _n↑_ ), and _p_ ? ( _n_ ?) indicate that _p_ must be made true (resp. false), that _n_ must decrease (resp. increase), and that _p_ (resp. _n_ ) can change in any way. Features not appearing in the effects of a rule must keep their values. The value of all the features Φ in a state _s_ is expressed as _f_ ( _s_ ), and _f_ without a state argument refers to an arbitrary feature valuation. 

A _pair of feature valuations_ ( _f, f[′]_ ) satisfies a policy rule _C �→ E_ if _f_ makes the conditions in _C_ true, and the change in feature values from _f_ to _f[′]_ is compatible with _E_ . A _state transition_ ( _s, s[′]_ ) in _P_ is compatible with a policy _π_ if ( _f_ ( _s_ ) _, f_ ( _s[′]_ )) satisfies a policy rule, and a _state trajectory s_ 0 _, . . . , sn_ is compatible with the policy in _P_ if _s_ 0 is the initial state of _P_ and each transition ( _si, si_ +1) is compatible with _π_ . Finally, the policy _π solves P_ if every maximal state trajectory compatible with _π_ reaches a goal state of _P_ , and it solves _Q_ if it solves every instance _P_ in _Q_ . 

For example, the policy _π_ over the features Φ = _{H, n}_ , captured by the following two rules 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_Language_Models_Can_Articulate_Their_Implicit_Goals_images/iclr_2025_Language_Models_Can_Articulate_Their_Implicit_Goals.pdf-0005-13.png)


where _H_ is true if holding a block and _n_ is the number of blocks above a block _x_ , solves the class _Q_ of Blocksworld problems where the goal is _clear_ ( _x_ ), regardless of the number or initial configuration of blocks. The first rule in the policy says to do any action that makes _H_ true and decreases the value of _n_ , provided that _H_ is false and _n_ is positive, while the second rule says to do any action that makes _H_ false and does not affect the value of _n_ , provided that _H_ is 

true and _n_ is positive. 

More interestingly, a general policy for the Delivery domain above can be defined using the features Φ = _{H, p, t, n}_ for “holding”, “distances to nearest package and target”, and “number of undelivered packages”, as: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_Language_Models_Can_Articulate_Their_Implicit_Goals_images/iclr_2025_Language_Models_Can_Articulate_Their_Implicit_Goals.pdf-0006-02.png)


The first rule says to do any action that decreases the distance _p_ to the nearest package when not holding a package and the distance is positive, whatever the effect on the distance _t_ to the target. The reading of the other rules is similar. These are policies written by hand though, and the question is how such policies can be learned? As before, this learning problem has been formulated and solved as a combinatorial optimization problem by creating a large but finite set _F_ of possible boolean and numerical features _from the domain predicates_ , using a standard grammar based on description logics, which is a fragment of 2-variable logics (Baader, Horrocks, and Sattler 2008). Provided with this set _F_ where each feature is given a cost (the number of grammar rules used to derive it), the learning task becomes: 

Given a domain _D_ , instances _P_ 1 _, . . . , Pk_ of _Q_ , and a finite pool of features _F_ , each with a cost, find the cheapest set of features Φ _⊂F_ and a policy _π_ over them such that _π_ solves the instances _P_ 1, ..., and _Pk_ . 

This is a combinatorial optimization problem that is cast and solved as a Weighted Max-SAT task (Franc`es, Bonet, and Geffner 2021). Once again, the language in which representations are sought provides a _strongly biased hypothesis space_ where policies that involve few simple features (in terms of the domain predicates) are preferred. As before, nothing precludes the use of deep learning to provide an alternative computational method, potentially more scalable and robust (Toyer et al. 2020; Garg, Bajpai, and Mausam 2020). A formal step in this direction is the computation of general optimal value functions using graph neural networks (St˚ahlberg, Bonet, and Geffner 2021), that exploits a correspondence between 2-variable logics and GNNs (Barcel´o et al. 2020; Grohe 2020). 

## **Language for decomposing goals into subgoals** 

The problem of _expressing_ and _using_ the common subgoal structure of a collection of problems _Q_ has been important in AI since the 1960s (Newell and Simon 1963; Erol, Hendler, and Nau 1994), while the problem of _learning_ such structure (in the form of intrinsic rewards) has become important in recent RL research as well (Zheng et al. 2020). We are interested in a similar problem but want to learn such structure over a suitable formal language. 

A _policy sketch_ or simply a _sketch_ is a set of sketch rules _C �→ E_ of the same form as policy rules. But while policy rules filter 1-step transitions; namely, when in a state 

_s_ , select a 1-step transition to any _s[′]_ such that the feature valuations ( _f_ ( _s_ ) _, f_ ( _s[′]_ )) satisfy a policy rule, sketch rules define _subproblems:_ when in a state _s_ , reach a state _s[′]_ , _not necessarily in one step_ , such that the feature valuations ( _f_ ( _s_ ) _, f_ ( _s[′]_ )) satisfy a sketch rule (Bonet and Geffner 2021). 

Sketches _decompose_ problems into _subproblems_ without prescribing how these subproblems should be solved (going from _s_ to _s[′]_ ). One is interested, however, in sketches that yield subproblems that can be solved efficiently, in low polynomial time (in the number of problem variables), and this is guaranteed when subproblems have a low, bounded width (Lipovetzky and Geffner 2012). In that case, the sketch has a bounded width, and all the problems in _Q_ can be solved in polynomial time. 

For example, a sketch _R_ 1 for Delivery that involves the single feature _n_ which tracks the number of packages not yet delivered, is given by the rule 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_Language_Models_Can_Articulate_Their_Implicit_Goals_images/iclr_2025_Language_Models_Can_Articulate_Their_Implicit_Goals.pdf-0006-12.png)


that expresses a decomposition where, in states _s_ where _n >_ 0, states _s[′]_ should be reached where the value of _n_ is lower than in _s_ . One can show that the resulting subproblems have a width bounded by 2. Likewise, a sketch _R_ 2 over the features _n_ and _H_ with the same meaning as above, can be given with two rules: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_Language_Models_Can_Articulate_Their_Implicit_Goals_images/iclr_2025_Language_Models_Can_Articulate_Their_Implicit_Goals.pdf-0006-14.png)


The rule on the left says that if not holding a package, get hold of one, while the other rule, that if holding a package, deliver it. The sketch _R_ 2 has width 1 meaning that all subproblems and hence all Delivery instances are rendered solvable in linear time. The uses of hand-crafted sketches and the learning of sketches have been addressed by Drexler, Seipp, and Geffner (2021a; 2021b). 

## **Summary** 

Deep learning and deep reinforcement learning are incredibly powerful techniques that struggle with structural generalization. While researchers assume that the right inductive bias in the architecture is all that is needed, no much guidance is offered to get there. In this paper, I’ve argued that learning representations over suitably designed formal languages with a semantics provides a research path that is crisp and meaningful, and illustrated the approach by focusing on the problems of learning general action models, policies, and subgoals (intrinsic rewards). This is all compatible with Bengio’s vision that systematic generalization arises from an “efficient factorization of knowledge into recomposable pieces”, but complements it by assuming that the pieces are expressed in a language. Learning languagebased representations from data, indeed, is not incompatible with the use of deep learning techniques. Moreover, the integration of language-based representations and deep learning, one describing what needs to be learned, and the other, delivering it at scale, has the potential to inform the design of deep learning methods that are more transparent and which can be assessed in ways that go beyond performance curves. 

## **Acknowledgments** 

I thank Judea Pearl, and not just for the quotations. I also thank Blai Bonet for comments and collaboration, the RLeap team, and the reviewers. The research is partially supported by an ERC Advanced Grant (No 885107), the EU Horizon 2020 project TAILOR (Grant No 952215), and the Knut and Alice Wallenberg (KAW) Foundation, under the WASP Program. 

## **References** 

Asai, M. 2019. Unsupervised Grounding of Plannable FirstOrder Logic Representation from Images. In _Proc. ICAPS_ . 

Baader, F.; Horrocks, I.; and Sattler, U. 2008. _Handbook of Knowledge Representation_ , chapter Description Logics. Elsevier. 

Barcel´o, P.; Kostylev, E. V.; Monet, M.; P´erez, J.; Reutter, J.; and Silva, J. P. 2020. The logical expressiveness of graph neural networks. In _ICLR_ . 

Bengio, Y.; Lecun, Y.; and Hinton, G. 2021. Deep learning for AI. _Communications of the ACM_ , 64(7): 58–65. 

Bonet, B.; and Geffner, H. 2001. Planning as Heuristic Search. _Artificial Intelligence_ , 129(1–2): 5–33. 

Bonet, B.; and Geffner, H. 2018. Features, Projections, and Representation Change for Generalized Planning. In _Proc. IJCAI_ , 4667–4673. 

Bonet, B.; and Geffner, H. 2020. Learning first-order symbolic representations for planning from the structure of the state space. In _Proc. ECAI_ . 

Bonet, B.; and Geffner, H. 2021. General Policies, Representations, and Planning Width. In _Proc. AAAI_ . 

Bronstein, M. M.; Bruna, J.; Cohen, T.; and Veliˇckovi´c, P. 2021. Geometric deep learning: Grids, groups, graphs, geodesics, and gauges. _arXiv preprint arXiv:2104.13478_ . 

Camacho, A.; Icarte, R. T.; Klassen, T. Q.; Valenzano, R. A.; and McIlraith, S. A. 2019. LTL and Beyond: Formal Languages for Reward Function Specification in Reinforcement Learning. In _IJCAI_ , 6065–6073. 

Chevalier-Boisvert, M.; Bahdanau, D.; Lahlou, S.; Willems, L.; Saharia, C.; Nguyen, T. H.; and Bengio, Y. 2019. BabyAI: A Platform to Study the Sample Efficiency of Grounded Language Learning. In _ICLR_ . 

Chevalier-Boisvert, M.; Willems, L.; and Pal, S. 2018. Minimalistic Gridworld Environment for OpenAI Gym. https: //github.com/maximecb/gym-minigrid. 

Cresswell, S. N.; McCluskey, T. L.; and West, M. M. 2013. Acquiring planning domain models using LOCM. _The Knowledge Engineering Review_ , 28(2): 195–213. 

Darwiche, A. 2018. Human-level intelligence or animal-like abilities? _Communications of the ACM_ , 61(10): 56–67. 

De Giacomo, G.; Iocchi, L.; Favorito, M.; and Patrizi, F. 2019. Foundations for restraining bolts: Reinforcement learning with LTLf/LDLf restraining specifications. In _ICAPS_ , volume 29, 128–136. 

Diuk, C.; Cohen, A.; and Littman, M. L. 2008. An objectoriented representation for efficient reinforcement learning. In _Proceedings of the 25th international conference on Machine learning_ , 240–247. 

Drexler, D.; Seipp, J.; and Geffner, H. 2021a. Expressing and Exploiting the Common Subgoal Structure of Classical Planning Domains Using Sketches. In _KR_ . ArXiv preprint arXiv:2105.04250. 

Drexler, D.; Seipp, J.; and Geffner, H. 2021b. Learning sketches for decomposing planning problems into subproblems of bounded width. Forthcoming. 

Erol, K.; Hendler, J.; and Nau, D. S. 1994. HTN planning: Complexity and expressivity. In _AAAI_ , volume 94, 1123– 1128. 

Evans, R.; Boˇsnjak, M.; Buesing, L.; Ellis, K.; Pfau, D.; Kohli, P.; and Sergot, M. 2021. Making sense of raw input. _Artificial Intelligence_ , 299. 

Franc`es, G.; Bonet, B.; and Geffner, H. 2021. Learning General Planning Policies from Small Examples Without Supervision. In _Proc. AAAI_ , 11801–11808. 

Garg, S.; Bajpai, A.; and Mausam. 2020. Generalized Neural Policies for Relational MDPs. In _Proc. ICML_ . 

Garnelo, M.; and Shanahan, M. 2019. Reconciling deep learning with symbolic artificial intelligence: representing objects and relations. _Current Opinion in Behavioral Sciences_ , 29: 17–23. 

Geffner, H. 2014. Artificial Intelligence: From programs to solvers. _AI Communications_ , 27(1): 45–51. 

Geffner, H. 2018. Model-free, model-based, and general intelligence. In _IJCAI_ , 10–17. 

Geffner, H.; and Bonet, B. 2013. _A Concise Introduction to Models and Methods for Automated Planning_ . Morgan & Claypool Publishers. 

Ghallab, M.; Nau, D.; and Traverso, P. 2016. _Automated planning and acting_ . Cambridge U.P. 

Goyal, A.; and Bengio, Y. 2020. Inductive biases for deep learning of higher-level cognition. _arXiv preprint arXiv:2011.15091_ . 

Goyal, A.; Didolkar, A.; Ke, N. R.; Blundell, C.; Beaudoin, P.; Heess, N.; Mozer, M.; and Bengio, Y. 2021. Neural Production Systems. _arXiv preprint arXiv:2103.01937_ . 

Goyal, A.; Lamb, A.; Hoffmann, J.; Sodhani, S.; Levine, S.; Bengio, Y.; and Sch¨olkopf, B. 2020. Recurrent Independent Mechanisms. In _ICLR_ . 

Grohe, M. 2020. The Logic of Graph Neural Networks. In _Proc. of the 35th ACM-IEEE Symp. on Logic in Computer Science_ . 

Haslum, P.; Lipovetzky, N.; Magazzeni, D.; and Muise, C. 2019. _An Introduction to the Planning Domain Definition Language_ . Morgan & Claypool. 

Hu, Y.; and De Giacomo, G. 2011. Generalized planning: Synthesizing plans that work for multiple environments. In _Proc. IJCAI_ , 918–923. 

Konidaris, G.; Kaelbling, L. P.; and Lozano-Perez, T. 2018. From skills to symbols: Learning symbolic representations for abstract high-level planning. _Journal of Artificial Intelligence Research_ , 61: 215–289. 

Lake, B.; Ullman, T.; Tenenbaum, J.; and Gershman, S. 2017. Building machines that learn and think like people. _Behavioral and Brain Sciences_ , 40. 

LeCun, Y.; Bengio, Y.; and Hinton, G. 2015. Deep learning. _Nature_ , 521(7553): 436. 

Lipovetzky, N.; and Geffner, H. 2012. Width and serialization of classical planning problems. In _Proc. ECAI_ , 540– 545. 

Manhaeve, R.; Dumanˇci´c, S.; Kimmig, A.; Demeester, T.; and De Raedt, L. 2021. Neural probabilistic logic programming in DeepProbLog. _Artificial Intelligence_ , 298: 103504. 

Marcus, G. 2018. Deep Learning: A Critical Appraisal. _arXiv preprint arXiv:1801.00631_ . 

McCarthy, J. 1960. Programs with common sense. In _Proc. Teddington Conf. on the Mechanization of Thought Processes_ . Reprinted in M. Minsky (Ed.), Semanfic Informafion Processing, 1968, MIT Press, Cambridge, Mass. 

McCarthy, J. 1987. Generality in artificial intelligence. _Communications of the ACM_ , 30(12): 1030–1035. 

McDermott, D. 1999. Using regression-match graphs to control search in planning. _Artificial Intelligence_ , 109(1-2): 111–159. 

McDermott, D. 2000. The 1998 AI Planning Systems Competition. _Artificial Intelligence Magazine_ , 21(2): 35–56. 

Mnih, V.; Kavukcuoglu, K.; Silver, D.; Rusu, A. A.; Veness, J.; Bellemare, M. G.; Graves, A.; Riedmiller, M.; Fidjeland, A. K.; Ostrovski, G.; et al. 2015. Human-level control through deep reinforcement learning. _Nature_ , 518(7540): 529. 

Muggleton, S.; and De Raedt, L. 1994. Inductive logic programming: Theory and methods. _The Journal of Logic Programming_ , 19: 629–679. 

Newell, A.; and Simon, H. 1963. GPS: a program that simulates human thought. In Feigenbaum, E.; and Feldman, J., eds., _Computers and Thought_ , 279–293. McGraw Hill. 

Pearl, J. 2009. _Causality: Models, Reasoning, and Inference (2nd Edition)_ . Cambridge University Press. 

Pearl, J. 2021. Radical empiricism and machine learning research. _Journal of Causal Inference_ , 9(1): 78–82. 

Pearl, J.; and Bareinboim, E. 2011. Transportability of causal and statistical relations: A formal approach. In _AAAI_ . 

Pearl, J.; and Mackenzie, D. 2018. _The book of why: the new science of cause and effect_ . Basic books. 

Raedt, L. D.; Kersting, K.; Natarajan, S.; and Poole, D. 2016. _Statistical relational artificial intelligence: Logic, probability, and computation_ . Morgan & Claypool Publishers. 

Rodriguez, I. D.; Bonet, B.; Romero, J.; and Geffner, H. 2021. Learning First-Order Representations for Planning from Black-Box States: New Results. In _KR_ . ArXiv preprint arXiv:2105.10830. 

Sanner, S. 2010. Relational Dynamic Influence Diagram Language (RDDL): Language Description. Sch¨olkopf, B.; Locatello, F.; Bauer, S.; Ke, N. R.; Kalchbrenner, N.; Goyal, A.; and Bengio, Y. 2021. Toward causal representation learning. _Proceedings of the IEEE_ , 109(5): 612–634. 

Serafini, L.; and Garcez, A. S. d. 2016. Learning and reasoning with logic tensor networks. In _Conference of the Italian Association for Artificial Intelligence_ , 334–348. Springer. 

Shanahan, M.; Nikiforou, K.; Creswell, A.; Kaplanis, C.; Barrett, D.; and Garnelo, M. 2020. An explicitly relational neural network architecture. In _International Conference on Machine Learning_ , 8593–8603. 

Silver, D.; Hubert, T.; Schrittwieser, J.; Antonoglou, I.; Lai, M.; Guez, A.; Lanctot, M.; Sifre, L.; Kumaran, D.; Graepel, T.; et al. 2017a. Mastering Chess and Shogi by Self-Play with a General Reinforcement Learning Algorithm. _arXiv preprint arXiv:1712.01815_ . 

Silver, D.; Schrittwieser, J.; Simonyan, K.; Antonoglou, I.; Huang, A.; Guez, A.; Hubert, T.; Baker, L.; Lai, M.; Bolton, A.; et al. 2017b. Mastering the game of go without human knowledge. _Nature_ , 550(7676): 354. 

Silver, T.; Allen, K. R.; Lew, A. K.; Kaelbling, L. P.; and Tenenbaum, J. 2020. Few-Shot Bayesian Imitation Learning with Logical Program Policies. In _Proc. AAAI_ , 10251– 10258. 

Srivastava, S.; Immerman, N.; and Zilberstein, S. 2008. Learning generalized plans using abstract counting. In _Proc. AAAI_ , 991–997. 

St˚ahlberg, S.; Bonet, B.; and Geffner, H. 2021. Learning General Optimal Policies with Graph Neural Networks: Expressive Power, Transparency, and Limits. Forthcoming. 

Toyer, S.; Thi´ebaux, S.; Trevizan, F.; and Xie, L. 2020. ASNets: Deep Learning for Generalised Planning. _Journal of Artificial Intelligence Research_ , 68: 1–68. 

Tsividis, P. A.; Loula, J.; Burga, J.; Foss, N.; Campero, A.; Pouncy, T.; Gershman, S. J.; and Tenenbaum, J. B. 2021. Human-level reinforcement learning through theorybased modeling, exploration, and planning. _arXiv preprint arXiv:2107.12544_ . 

van den Broeck, G.; Kersting, K.; Natarajan, S.; and Poole, D. 2021. _Introduction to Lifted Probabilistic Inference_ . MIT Press. 

Yang, Z.; Ishay, A.; and Lee, J. 2020. NeurASP: Embracing Neural Networks into Answer Set Programming. In _IJCAI_ , 1755–1762. 

Younes, H. L.; Littman, M. L.; Weissman, D.; and Asmuth, J. 2005. The first probabilistic track of the international planning competition. _Journal of Artificial Intelligence Research_ , 24: 851–887. 

Zheng, Z.; Oh, J.; Hessel, M.; Xu, Z.; Kroiss, M.; Van Hasselt, H.; Silver, D.; and Singh, S. 2020. What can learned intrinsic rewards capture? In _ICML_ , 11436–11446. 

