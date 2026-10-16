# **LogiCity: Advancing Neuro-Symbolic AI with Abstract Urban Simulation** 

**Bowen Li**[1] **Zhaoyu Li**[2] **Qiwei Du**[3] **Jinqi Luo**[4] **Wenshan Wang**[1] **Yaqi Xie**[1] **Simon Stepputtis**[1] **Chen Wang**[3] **Katia Sycara**[1] **Pradeep Ravikumar**[1] **Alexander Gray**[5] **Xujie Si**[2] _[,]_[6] **Sebastian Scherer**[1] _[ ∗]_ 

1Carnegie Mellon University 2University of Toronto 3University at Buffalo 4University of Pennsylvania 5Centaur AI Institute 6CIFAR AI Chair, Mila `{bowenli2, basti}@andrew.cmu.edu` 

## **Abstract** 

Recent years have witnessed the rapid development of Neuro-Symbolic (NeSy) AI systems, which integrate symbolic reasoning into deep neural networks. However, most of the existing benchmarks for NeSy AI fail to provide long-horizon reasoning tasks with complex multi-agent interactions. Furthermore, they are usually constrained by fixed and simplistic logical rules over limited entities, making them far from real-world complexities. To address these crucial gaps, we introduce LogiCity, the first simulator based on customizable first-order logic (FOL) for an urban-like environment with multiple dynamic agents. LogiCity models diverse urban elements using semantic and spatial _concepts_ , such as `IsAmbulance` ( `X` ) and `IsClose` ( `X` _,_ `Y` ). These concepts are used to define FOL rules that govern the behavior of various agents. Since the concepts and rules are _abstractions_ , they can be universally applied to cities with any agent compositions, facilitating the instantiation of diverse scenarios. Besides, a key feature of LogiCity is its support for user-configurable abstractions, enabling customizable simulation complexities for logical reasoning. To explore various aspects of NeSy AI, LogiCity introduces two tasks, one features long-horizon sequential decision-making, and the other focuses on one-step visual reasoning, varying in difficulty and agent behaviors. Our extensive evaluation reveals the advantage of NeSy frameworks in _abstract_ reasoning. Moreover, we highlight the significant challenges of handling more complex abstractions in long-horizon multi-agent scenarios or under high-dimensional, imbalanced data. With its flexible design, various features, and newly raised challenges, we believe LogiCity represents a pivotal step forward in advancing the next generation of NeSy AI. All the code and data are open-sourced at our website. 

## **1 Introduction** 

Unlike most existing deep neural networks [1, 2], humans are not making predictions and decisions in a relatively black-box way [3]. Instead, when we learn to drive a vehicle, play sports, or solve math problems, we naturally leverage and explore the underlying symbolic representations and structure [3–5]. Such capability enables us to swiftly and robustly reason over complex situations and to adapt to new scenarios. To emulate human-like learning and reasoning, the Neuro-Symbolic (NeSy) AI community [6] has introduced various hybrid systems [7–18], integrating symbolic reasoning into deep neural networks to achieve higher data efficiency, interpretability, and robustness[1] . 

Despite their rapid advancement, many NeSy AI systems are designed and tested only in very simplified and limited environments, such as visual sudoku [19], handwritten formula recogni- 

> 1This work mainly focuses on logical reasoning within the broad NeSy community. 

38th Conference on Neural Information Processing Systems (NeurIPS 2024) Track on Datasets and Benchmarks. 

tion [13], knowledge graphs [20], and reasoning games/simulations [7, 21–25] (see Table 1). A benefit of such environments is that they usually provide data with symbolic annotations, which the NeSy AI systems can easily integrate. However, they are still far from real-world complexity due to the lack of three key features: (1) Most simulators are governed by propositional rules tied to specific fixed entities [13, 19, 22] rather than _**abstractions**_ [7]. As a result, agents learned from them are hard to generalize compositionally. (2) In real life, we learn to reason gradually from simple to complex scenarios, requiring the rules within the environment to be _**flexible**_ . Either overly simplified [7, 19, 23] or overly complicated/unsuitable [26, 27] environments cannot promote the development of NeSy AI systems. (3) Few simulators offer realistic _**multi-agent**_ interactions, where the environment agents often need to actively adapt their behaviors in response to varying actions of the ego agent. Moreover, a comprehensive benchmark needs to provide both _**long-horizon**_ (e.g., _>_ 20 steps) [7] and _**visual reasoning**_ [19] scenarios to exercise different aspects of NeSy AI. 

To address these issues, we introduce LogiCity, the first customizable first-orderlogic (FOL)-based [28] simulator and benchmark motivated by complex urban dynamics. As illustrated in Figure 1, LogiCity allows users to freely customize spatial and semantic conceptual attributes (concepts), FOL rules, and agent sets as configurations. Since the concepts and rules are _abstractions_ , they can be universally applied to any agent compositions across different cities. For example, in Figure 1, concepts such as “ `IsClose` ( `X` , `Y` ), `IsAmbulance` ( `Y` )”, and rules like “ `Stop` ( `X` ):- `IsAmbulance` ( `Y` ), `IsClose` ( `X` , `Y` )” can be _grounded_ with specific and 

Table 1: Comparison of existing NeSy benchmarks and LogiCity. Our simulator is governed by diverse _abstractions_ , which can be flexibly customized. We also support long-horizon, multi-agent tasks and RGB rendering. “ _−_ ” denotes partially supported features. 

||Benchmarks<br>Features|Abstract|Flexible|Multi-<br>Agent|Long-<br>Horizon|RGB|
|---|---|---|---|---|---|---|
||Visual Sudoku [19]<br>Handwritten Formula [13]<br>Smokers & Friends [20]|✗<br>✗<br>✓|✗<br>✗<br>✗|✗<br>✗<br>✗|✗<br>✗<br>✗|✗<br>✗<br>✗|
||CLEVR [21]<br>BlocksWorld [7]<br>Atari Games [22]|✓<br>✓<br>✗|✓<br>✗<br>_−_|✗<br>✗<br>_−_|✗<br>✓<br>✓|✓<br>✗<br>✓|
||Minigrid & Miniworld [23]<br>BabyAI [24]<br>HighWay [25]|_−_<br>✓<br>✓|_−_<br>_−_<br>✗|✗<br>✗<br>✓|_−_<br>✓<br>✗|✓<br>✓<br>✓|
||**LogiCity (Ours)**|✓|✓|✓|✓|✓|



distinct train/test agent sets to govern their behaviors in the simulation. To render the environment into diverse RGB images, we leverage foundational generative models [1, 29–31]. Since our modular framework enables seamless configuration adjustments, researchers can explore compositional generalization by changing agent sets while keeping abstractions fixed, or study adaptation to new and more complex abstractions by altering rules and concepts. 

To exercise different aspects of NeSy AI, we use LogiCity to design tasks for both sequential decisionmaking (SDM) and visual reasoning. In the SDM task, algorithms need to navigate a lengthy path ( _>_ 40 steps) with minimal trajectory cost, considering rule violations and step-wise action costs. This involves planning amidst complex scenarios and interacting with multiple dynamic agents. For instance, decisions like speeding up may incur immediate costs but could lead to a higher return in achieving the goal sooner. Notably, our SDM task is also unique in that training and testing agent compositions are different, requiring an agent to learn the abstractions and generalize to new compositions. Contrarily, the visual reasoning task focuses on single-step reasoning but features highdimensional RGB inputs. Algorithms must perform sophisticated abstract reasoning to predict actions for all agents with high-level perceptual noise. Across both tasks, we vary reasoning complexity to evaluate the algorithms’ ability to adapt and learn new abstractions. 

While we show that existing NeSy approaches [7, 11] perform better in learning abstractions, both from scratch and continually, more complex scenarios from LogiCity still pose significant challenges for prior arts [7, 10, 11, 32–38]. On the one hand, LogiCity raises the abstract reasoning complexity with long-horizon multiple agents scenario, which have not been adequately addressed by current methods. Besides, it also highlights the difficulty of learning complex abstractions from highdimensional data even for one-step reasoning. On the other hand, LogiCity provides flexible groundtruth symbolic abstractions, allowing for the new methods to be gradually designed, developed, and validated. Therefore, we believe LogiCity represents a crucial step for advancing the next generation of NeSy AI capable of sophisticated abstract reasoning and learning. 

2 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_LogiCity_Advancing_Neuro-Symbolic_AI_with_Abstract_Urban_Simulation_images/nips_2024_LogiCity_Advancing_Neuro-Symbolic_AI_with_Abstract_Urban_Simulation.pdf-0003-00.png)


**----- Start of picture text -----**<br>
Concepts Configuration Diverse Rendering Ambulance Police House Store<br>% Spatial % Semantic A modern white  ambulance  car<br>IsInInter(X) IsCar(X) with red crosses, aerial view,<br>LeftOf(X,Y) IsPedes(X) Language pure background, single- Text-Driven<br>IsClose(X , Y) IsAmbulance(X) Model instance, for tile-based games Diffusion Model<br>... IsPolice(X) A black  police  cruiser with white<br>... doors<br>Rules … …<br>Stop(X):-IsAmbulance(Y),<br>IsClose(X,Y) Global Map Move & Update SMT Solver<br>Slow(X):-IsPolice(Y),<br>RightOf(X,Y)<br>Agents FOVs Groundings<br>𝐴𝐴𝐴 % !#$ Train:IsCarIsPolice:IsPedesIsOld:IsCarIsAmbulance % 𝐴𝐴𝐴𝐴!#$" Test:IsCar:IsPedesIsOld:IsPedesIsOld: Abstraction-based Simulation 𝐴" …𝐴! IsCar(IsPolice(IsPedes(IsInInter(...IsClose(LeftOf(... 𝐴𝐴! ):1 𝐴𝐴" , 𝐴!! ):0,  𝐴𝐴! ):1  𝐴!" ):1):1 " ):1<br>𝐴" :IsCar IsCarIsPolice Configuration Concepts, Rules Simulation Grid Rendering<br>... ... Agents Semantics Map<br>**----- End of picture text -----**<br>


Figure 1: LogiCity employs _abstract_ concepts and rules, allowing different agent sets to address compositional generalization. Its modular structure enables users to modify _abstractions_ flexibly. 

## **2 Related Works** 

## **2.1 Neuro-Symbolic AI** 

NeSy AI systems aim to integrate formal logical reasoning into deep neural networks. We distinguish these systems into two categories: _deductive_ methods and _inductive_ methods. 

**Deductive Methods** typically operate under the assumption that the underlying symbolic structure and the chain of deductive reasoning (rules) are either known [8, 9, 20, 39–41] or can be generated by an oracle [17, 18, 42, 43]. Some of these approaches constructed differentiable logical loss functions that constrain the training of deep neural networks [39, 40]. Others, such as DeepProbLog [8], have formulated differential reasoning engines, thus enabling end-to-end learning [44–46]. Recently, Large Language Models (LLMs) have been utilized to generate executable code [17, 42, 43] or planning abstractions [47], facilitating the modular integration of the grounding networks. Despite their success, deductive methods sidestep or necessitate the laborious manually engineered symbolic structure, which potentially limits their applicability in areas lacking formalized knowledge. 

**Inductive Methods** focus on generating the symbolic structure either through supervised learning [10, 11, 35, 48–50] or by interacting with the environment [51–53]. One line of research explicitly searches the rule space, such as _∂_ ILP [48], Difflog [54], and Popper [10]. However, as the rule space can be exponentially large for _abstractions_ , these methods often result in prolonged search times. To address this, some strategies incorporate neural networks to accelerate the search process [11, 49, 50]. Another avenue of inductive methods involves designing logic-inspired neural networks where rules are implicitly encoded within the learned weights [7, 19, 55, 56], such as SATNet [19] and Neural Logic Machines (NLM) [7]. While these methods show promise for scalability and generalization, their applications have been predominantly limited to overly simplistic test environments. 

## **2.2 Games and Simulations** 

Various gaming environments [22–24, 26] have been developed to advance AI agents. Atari games [22], for instance, provide diverse challenges ranging from spatial navigation in “Pac-Man” to real-time strategy in “Breakout”. More complex games include NetHack [26], StarCraft II [57], and MineCraft [27], where an agent is required to do strategic planning and resource management. LogiCity shares similarities with these games in that agent behavior is governed by rules. Especially, LogiCity can be viewed as a Rogue-like gaming environment [26], where maps and agent settings could be randomly generated in different runs. However, our simulator is uniquely tailored for the 

3 

NeSy AI community because: (1) LogiCity provides formal symbolic structure in FOL, enhancing the validation and design of NeSy frameworks. (2) Since FOL predicates and rules are abstractions, a user can arbitrarily customize the composition of the world, introducing adversarial scenarios. (3) LogiCity also supports customizable reasoning complexity through flexible configuration settings. Another key difference between LogiCity and most games [22, 26, 57] is that the behavior of non-player characters (NPCs) in LogiCity is governed by global logical constraints rather than human-engineered behavior trees [58–61]. This design enables NPCs to automatically commit to actions that ensure global rule satisfaction, without the need for manual scripting. Moreover, compared to these games, LogiCity is closer to real urban life, offering a more practical scenario. 

Addressing the need for realism, autonomous driving (AD) simulators [25, 62–67] deliver highquality rendering and accurate traffic simulations but often adhere to fixed rules for limited sets of _concepts_ . Among them, the SCENIC language [65–67] is the closest to LogiCity, which uses Linear Temporal Logic to specify AD scenarios. Unlike SCENIC, LogiCity uses _abstractions_ in FOL, which allows for the generation of a large number of cities with distinct agent compositions more easily. Besides, LogiCity goes beyond these AD simulators by introducing a broader range of _concepts_ and more complex rules, raising the challenge of sophisticated logical reasoning. 

## **3 LogiCity Simulator** 

The overall framework of LogiCity simulator is shown in Figure 1. In the configuration stage, a user is expected to provide _Concepts_ , Rules, and Agent set, which are fed into the _abstraction_ -based simulator to create a sequence of urban grid maps. These maps are rendered into diverse RGB images via generative models, including a LLM [1] and a text-driven diffusion model [29]. 

## **3.1 Configuration and Preliminaries** 

**Concepts** consist of _K_ background predicates _P_ = _{Pi_ ( _·_ ) _|i_ = 1 _, . . . , K}_ . In LogiCity, we can define both _semantic_ and _spatial_ predicates. For example, `IsAmbulance` ( _·_ ) is an unary semantic predicate and `IsClose` ( _·, ·_ ) is a binary spatial predicate. These predicates will influence the truth value of four action predicates { `Slow` ( _·_ ), `Normal` ( _·_ ), `Fast` ( _·_ ), `Stop` ( _·_ )} according to certain rules. 

**Rules** consist of _M_ FOL clauses, _C_ = _{Cm|m_ = 1 _, . . . , M }_ . Following ProLog syntax [28], an FOL clause _Cm_ can be written as: 

## `Stop` ( `X` ) : _−_ `IsClose` ( `X` _,_ `Y` ) _∧_ `IsAmbulance` ( `Y` ) _,_ 

where `Stop` ( `X` ) is the _head_ , and the rest after “: _−_ ” is the _body_ . `X` _,_ `Y` are variables, which will be _grounded_ into specific entities for rule inference. Note that the clause implicitly declares that the variables in the _head_ have a universal quantifier ( _∀_ ) and the other variables in the _body_ have an existential quantifier ( _∃_ ). We assume only action predicates appear in the _head_ , both action and background predicates could appear in the _body_ . 

The concepts _P_ and rules _C_ are _abstractions_ , which are not tied to specific entities. 

**Agents** serve as the _entities_ in the environment, which is used to _ground_ the _abstractions_ . We use _A_ = _{An|n_ = 1 _, . . . , N }_ to indicate all the _N_ agents in a city. Each agent will initially be annotated with the semantic _concepts_ defined in _P_ . For example, an ambulance car _A_ 1 is annotated as _A_ 1 = _{_ `IsCar` : `True` _,_ `IsAmbulance` : `True` _, . . . , p}_ , where _p ∈_ R denotes right-of-way priority. 

_P, C, A_ make up the configuration of LogiCity simulation. A user can flexibly change any of them seamlessly without modifying the simulation and rendering process. 

## **3.2 Simulation and Rendering** 

As the simulation initialization, a static urban map **M** s _∈{_ 0 _,_ 1 _}[W][ ×][H][×][B]_ is constructed, where _W, H_ denotes the width and height. _B_ indicates the number of static semantics in the city, _e.g._ , traffic streets, walking streets, intersections, _etc._ The agents then randomly sample collision-free start and goal locations on the map. These locations are fed into a search-based planner [68] to obtain the global paths that the agents will follow to navigate themselves. On top of the static map, each agent will create an additional dynamic layer, indicating their latest location and planned paths. We use **M** _[t] ∈{_ 0 _,_ 1 _}[W][ ×][H][×]_[(] _[B]_[+] _[N]_[)] to denote the full semantic map with all the _N_ agents at time step _t_ . 

4 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_LogiCity_Advancing_Neuro-Symbolic_AI_with_Abstract_Urban_Simulation_images/nips_2024_LogiCity_Advancing_Neuro-Symbolic_AI_with_Abstract_Urban_Simulation.pdf-0005-00.png)


**----- Start of picture text -----**<br>
𝑠 ["] Infos Grounding Scene Graph 𝐆 [$]<br>Ego 𝑠 ["] 𝐚!" Rewards𝑅(𝑠 ["] , 𝐚!" ) 𝐡𝑥𝑦!!! 𝐡𝑥𝑦""" 𝐡𝑥𝑦### Binary Unary IsOld(IsAtInter(... 𝐴! ): 0.7𝐴! ): 0.2<br>Simulation 𝑨𝟏 Slow(Normal(Fast(Stop( Actions𝐴𝐴𝐴!!! ):1):0):0 𝐴! 𝐚 ):0 !" Model 𝑤𝐝ℎ𝑝!!!! 𝑤𝐝ℎ𝑝"""" 𝑤𝐝ℎ𝑝#### Pair ... 𝐟𝒏 ... IsClose(RightOf(LeftOf(... 𝐴𝐴𝐴! , !! ,,  𝐴 𝐴 𝐴" ): "" ):): 0.10.20.1<br>Image 𝐈 bbox Reasoning<br>ROIAlign<br>FOV<br>Parsing Fun. Grounding Enc. 𝐅 ActionsPred. 𝐚𝐚𝐚"""!"#<br>...<br>(a) Safe Path Following Task (b) Visual Action Prediction Task<br>…<br>**----- End of picture text -----**<br>


Figure 2: Demonstration of the Safe Path Following (SPF) and Visual Action Prediction (VAP) tasks in LogiCity. SPF emphasizes sequential decision-making while VAP focuses on one-step sophisticated reasoning on RGB inputs. In the VAP task, we also display the baseline model structure. 

During the simulation, each agent _An_ only has a limited field-of-view (FOV) of the overall map **M** _[t]_ , which we denote **M** _[t] n_[. Additionally, FOV agents (self-included) in] **[ M]** _[t] n_[is obtained and denoted as] _[ A][t] n_[.] A group of _K_ pre-defined, binary functions _{Fi_ ( _·, ·_ ) _|i_ = 1 _, . . . , K}_ for the _K_ predicates are then employed to obtain the _grounding_ **g** _n[t]_[for the ego agent,] **[ g]** _n[t]_[= Cat] � _F_ 1( **M** _[t] n[,][ A][t] n_[)] _[, . . . ,][ F][K]_[(] **[M]** _[t] n[,][ A][t] n_[)] �. Here, Cat( _·_ ) denotes concatenation. Assuming we have a total of _Nn_ FOV agents, **g** _n[t]_[will be in the] _K ri_ shape of _{_ 0 _,_ 1 _}_ � _i_ =1 _[N] n_ , where _ri_ is the _arity_ for the _i_ -th predicate. Given **g** _n[t]_[and the rule clauses] _C_ , we leverage an SMT solver [69] to find the truth value of the four _grounded_ action predicates, **a** _[t] n_[=][SMT(] **[g]** _n[t][,][ C]_[)][.][Here,] **[ a]** _[t] n[∈{]_[0] _[,]_[ 1] _[}]_[4][denotes the truth value of the] _[ grounded]_[ four action predi-] cates for agent _An_ at time _t_ . An example of this procedure for _A_ 1 is provided in Figure 1. After all the agents take proper actions, we move their location, update the semantic map into **M** _[t]_[+1] , and repeatedly apply the same procedure. Whenever an agent reaches its goal, the end position becomes the new starting point, a new goal point is randomly sampled, and the navigation is re-started. 

To render the binary grid map **M** _[t]_ into an RGB image **I** _[t]_ with high visual variance, we leverage foundational generative models [1, 29–31]. We first feed the name of each _semantic concept_ , including different types of agents and urban elements, into GPT-4 [1] and asked it to generate diverse descriptions. These descriptions are fed into a diffusion-based generative model [29], which creates diverse icons. These icons will be randomly selected and composed into the grid map landscape to render highly diverse RGB image datasets. Detailed simulation procedure is shown in Appendix C. 

## **4 LogiCity Tasks** 

LogiCity introduced above can exercise different aspects of NeSy AI. For example, as shown in Figure 2, we design two different tasks. The Safe Path Following (SPF) task aims at evaluating sequential decision-making capability while the Visual Action Prediction (VAP) task focuses on reasoning with high-dimensional data. Both tasks assume no direct access to the rule clauses _C_ . 

## **4.1 Safe Path Following** 

SPF requires an algorithm to control an agent in LogiCity, following the global path to the goal while maximizing the trajectory return. The agent is expected to sequentially make a decision on the four actions based on its discrete, partial observations, which should minimize rule violation and action costs. In the following introduction, we assume the first agent, _i.e._ , _A_ 1 is the controlled agent. 

Specifically, the SPF task can be formulated into a Partially Observable Markov Decision Process (POMDP), which can be defined by the tuples ( _S,_ A _,_ Ω _, T, Z, R, γ_ ). The state at time _t_ is the global urban grid, together with all the agents and their conceptual attributes, _s[t]_ = _{_ **M** _[t] , A} ∈ S_ . The action space A is the 4-dimensional discrete action vector **a** _[t]_ 1[.][The observation at] _[ t]_[-th step is the] _grounding_ of the agent’s FOV, _o[t]_ = **g** 1 _[t][∈]_[Ω][, which can be obtained from the parsing functions] _[ F][i]_[.] State transition _T_ ( _s[t]_[+1] _|s[t] ,_ **a** _[t]_ 1[)][ is the simulation process introduced in Section][ 3.2][.][The observation] 

5 

function _Z_ ( _o[t]_[+1] _|s[t]_[+1] _,_ **a** _[t]_ 1[)][ is a deterministic cropping function.][The reward function] _[ R]_[(] _[s][t][,]_ **[ a]** _[t]_ 1[)][ is] defined as _R_ ( _s[t] ,_ **a** _[t]_ 1[) =][ �] _[M] m[w] m_[r] _[ψ]_[(] _[s][t][,]_ **[ a]** 1 _[t][, C][m]_[) +] _[ w]_[a] _[ϕ]_[(] **[a]** _[t]_ 1[) +] _[ w]_[overtime][(] _[t]_[)][, where] _[ w] m_[r][is the weight] of rule violation punishment for the _m−_ th clause _Cm_ . _ψ_ ( _·, ·, ·_ ) evaluates if clause _Cm_ is satisfied for agent _A_ 1 given _s[t]_ and **a** _[t]_ 1[.] _[ϕ]_[(] **[a]** _[t]_ 1[)][ indicates action cost at step] _[ t]_[ and] _[ w]_[a][ is a normalization factor.] _w_[overtime] ( _t_ ) gives constant punishment if _t_ is larger than the max horizon. Finally, _γ_ is a discount factor set to 0 _._ 99. An example of SPF is shown in Figure 2 (a), where _A_ 1 is the _Ambulance_ car in the purple box. The dashed box denotes its FOV, which will be _grounded_ by the parsing functions. A model needs to learn to sequentially output action decisions that maximize trajectory return. 

Compared to existing reasoning games [22–24], LogiCity’s SPF task presents two unique challenges: (1) Different agent configurations _A_ in training and testing cause distribution shifts in world transitions ( _T_ ). This requires the model to learn the _abstractions_ ( _P, C_ ) for compositional generalization. For example, training agents could include _ambulance_ plus _pedestrian_ and _police_ car plus _pedestrian_ . In testing, the algorithm may need to plan with _ambulance_ plus _police_ car. (2) LogiCity supports more realistic multi-agent interaction. For instance, if the controlled agent arrives at an intersection later than other agents, it must wait, resulting in a lower trajectory return; if it speeds up to arrive earlier, others yield, ending up with a higher score. This encourages learning both ego rules and world transitions with multiple agents (how to plan smartly by forecasting). 

## **4.2 Visual Action Prediction** 

Unlike SPF, which is long-horizon and assumes access to the _groundings_ , the VAP task is step-wise and requires reasoning on high-dimensional data [13, 19]. As shown in Figure 2 (b), inputs to VAP models include the rendered image **I** (We omit the time superscript _t_ here) and information for _N_ agents [ **h** 1 _, . . . ,_ **h** _N_ ] _∈_ R _[N][×]_[9] , where **h** _n_ = [ _xn, yn, wn, hn,_ **d** _n, p_ ] _[⊤]_ consists of location ( _xn, yn_ ), scale ( _wn, hn_ ), one-hot direction **d** _n ∈_ R[4] , and normalized priority _p_ . During training, the models learn to reason and output the action vectors ˆ **a** _n_ for all the _N_ agents with ground-truth supervision. During test, the models are expected to predict the actions for different sets of agents. 

This task is approached as a two-step graph reasoning problem [7, 37]. As illustrated in Figure 2 (b), a grounding module first predicts interpretable _grounded_ predicate truth values, which are then used by a reasoning module to deduce action predicates. To be more specific, a visual encoder [2, 70] first extracts global features **F** from **I** . Agent-centric regional features are derived from ROIAlign [71], which resizes the image-space bounding boxes to match the feature scale and then crops the global feature using bilinear interpolation. The resulting regional features for each agent, denoted as **f** _n_ , are fed into unary prediction heads to generate unary predicate _groundings_ . Meanwhile, binary prediction heads utilize paired agent information to predict binary predicates. Together, the _groundings_ form a scene graph **G**[ˆ] , which a graph reasoning engine [7, 37] uses to predict actions ˆ **a** _n_ . 

Similar to the SPF task, the VAP task also features different train/test agent compositions, necessitating the model’s ability to learn _abstractions_ . Additionally, unlike reasoning on structured, symbolic knowledge graphs [7, 11, 20], the diverse visual appearances in LogiCity introduce high-level perceptual noise, adding an extra challenge for reasoning algorithms. 

## **5 Experiments** 

## **5.1 Safe Path Following** 

We first construct a ground-truth rule-based agent as Oracle and a Random agent as the worst baseline, showing their results in Table 2. Two branches of methods are considered here, behavior cloning (BC) and reinforcement learning (RL), respectively. All the experiments in SPF are conducted on a single NVIDIA RTX 3090 Ti GPU with 32 AMD Ryzen 5950X 16-core processors. 

**Baselines.** In the BC branch, we provide oracle trajectories as demonstration and consider the inductive logical programming (ILP) algorithms [10], including symbolic ones [10, 35] and NeSy ones [7, 11]. We also construct a multi-layer perceptron (MLP) and graph neural networks (GNN) [37] as the pure neural baselines. In the RL track, we first build neural agents using various RL algorithms, including on-policy [32, 33], off-policy [7, 34] model-free approaches and model-based algorithms [36, 38]. Since most of the existing NeSy RL methods [51, 52] are carefully engineered for simpler environments, we find it hard to incorporate them into our LogiCity environment. To 

6 

Table 2: Empirical results of different methods in SPF task. TSR denotes trajectory success rate (most crucial) and DSR indicates decision success rate. _[†]_ means Popper timed out. _[‡]_ indicates conflict rules will be inducted for different actions. See our website and Appendix F for episode visualizations. 

|Supervision<br>Mode\Model<br>Metric|Supervision<br>Mode\Model<br>Metric|Easy<br>TSR<br>DSR<br>Score|Easy<br>TSR<br>DSR<br>Score|Medium<br>TSR<br>DSR<br>Score|Medium<br>TSR<br>DSR<br>Score|Hard<br>TSR<br>DSR<br>Score|Hard<br>TSR<br>DSR<br>Score|Expert<br>TSR<br>DSR<br>Score|
|---|---|---|---|---|---|---|---|---|
||||||||||
|N/A<br>Oracle<br>Random||1.00<br>1.00<br>8.51<br>0.07<br>0.00<br>0.00||1.00<br>1.00<br>8.45<br>0.06<br>0.00<br>0.00||1.00<br>1.00<br>9.63<br>0.04<br>0.01<br>0.00||1.00<br>1.00<br>4.33<br>0.05<br>0.06<br>0.00|
||||||||||
|Behavior<br>Cloning<br>Popper [10]<br>MaxSynth [35]<br>HRI [11]<br>NLM [7]<br>GNN [37]<br>MLP||**1.00**<br>1.00<br>8.51<br>**1.00**<br>1.00<br>8.51<br>0.37<br>0.78<br>4.40<br>0.75<br>1.00<br>7.29<br>0.26<br>0.39<br>2.58<br>0.61<br>0.63<br>4.80||N/A_†_<br>N/A_†_<br>N/A_†_<br>0.25<br>0.67<br>3.18<br>**0.48**<br>0.70<br>4.75<br>0.30<br>0.67<br>3.24<br>0.17<br>0.24<br>1.31<br>0.20<br>0.19<br>1.22||N/A_†_<br>N/A_†_<br>N/A_†_<br>0.15<br>0.60<br>2.96<br>0.08<br>0.15<br>0.59<br>**0.24**<br>0.27<br>2.00<br>0.19<br>0.39<br>2.19<br>0.12<br>0.13<br>0.81||N/A_†_<br>N/A_†_<br>N/A_†_<br>0.09<br>0.21<br>0.37<br>N/A_‡_<br>N/A_‡_<br>N/A_‡_<br>**0.22**<br>0.38<br>0.99<br>0.19<br>0.32<br>0.84<br>0.10<br>0.19<br>0.25|
||||||||||
|Reinforcement<br>Learning<br>NLM-DQN [7,34]<br>MB-shooting [36]<br>DreamerV2 [38]<br>DQN [34]<br>PPO [32]<br>A2C [33]||**0.53**<br>0.96<br>5.93<br>0.24<br>0.44<br>2.55<br>0.07<br>0.43<br>2.86<br>0.35<br>0.89<br>4.80<br>0.33<br>0.36<br>2.83<br>0.10<br>0.16<br>1.00||**0.47**<br>0.67<br>4.40<br>0.20<br>0.17<br>1.18<br>0.02<br>0.21<br>0.67<br>0.42<br>0.59<br>3.72<br>0.09<br>0.25<br>0.88<br>0.06<br>0.29<br>1.07||**0.29**<br>0.40<br>2.69<br>0.16<br>0.17<br>1.26<br>0.00<br>0.30<br>1.45<br>0.09<br>0.12<br>0.63<br>0.02<br>0.38<br>1.57<br>0.00<br>0.14<br>0.46||**0.15**<br>0.35<br>0.62<br>0.13<br>0.11<br>0.37<br>0.12<br>0.06<br>0.41<br>0.07<br>0.24<br>0.37<br>0.12<br>0.08<br>0.38<br>0.12<br>0.09<br>0.34|




![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_LogiCity_Advancing_Neuro-Symbolic_AI_with_Abstract_Urban_Simulation_images/nips_2024_LogiCity_Advancing_Neuro-Symbolic_AI_with_Abstract_Urban_Simulation.pdf-0007-02.png)


**----- Start of picture text -----**<br>
24%  Drop 34%  Drop<br>54%  Drop<br>**----- End of picture text -----**<br>



![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_LogiCity_Advancing_Neuro-Symbolic_AI_with_Abstract_Urban_Simulation_images/nips_2024_LogiCity_Advancing_Neuro-Symbolic_AI_with_Abstract_Urban_Simulation.pdf-0007-03.png)


**----- Start of picture text -----**<br>
30%  Data<br>60%  Data<br>**----- End of picture text -----**<br>


Figure 3: Results for different agent configurations in medium and hard modes of SPF task. We report the average of DSR and TSR here. 

Figure 4: Continual learning results for MLP and NLM [7]. The results achieved by training from scratch are reported in dashed lines. 

introduce NeSy AI in the RL track, we develop a new Q-learning agent based on NLM [7], which we denote as NLM-DQN [7, 34]. For more details, please see Appendix A. 

**Modes and Datasets.** As shown in Table 2, we provide four modes in the SPF task, namely easy, medium, hard, and expert. From easy to medium to hard mode, we progressively introduce more _concepts_ and more complex rules, constraining only `Stop` action. The expert mode constrains all four actions with the most complex rule sets. More details are included in Appendix B. 

**Metrics.** We consider three metrics in this task. Trajectory Success Rate (TSR) evaluates if an agent _T_ all can reach its goal within the allotted time. It is defined as TSR = � _i T_[all] Succ _i_ , where _T_[all] is the total number of episodes, and Succ _i_ = 1 if the _i_ -th episode is completed within twice the oracle steps without rule violations, and Succ _i_ = 0 otherwise. Decision Success Rate (DSR) assesses if an agent _T_ all adheres to all rules. It is defined as DSR = � _i T_[all] Dec _i_ , where Dec _i_ = 1 if the _i_ -th episode has at least one rule-constrained step and the agent does not violate any rules throughout, regardless of task completion, and Dec _i_ = 0 otherwise. The score metric is the averaged trajectory return over all episodes minus the return of a random agent. Among them, TSR is the most crucial. 

## **5.1.1 Empirical Evaluation** 

We present the empirical results in Table 2, showing LogiCity’s ability to vary reasoning complexity. In the BC track, symbolic methods [10, 35] perform well in the _easy_ mode but struggle with more 

7 

Table 3: Empirical results of different methods and settings in VAP task (Modular is more crucial). We report the recall rate for each action, averaged accuracy (aAcc.), and weighted accuracy (wAcc.). 

|||||
|---|---|---|---|
|Mode<br>Num. Actions<br>Supervision<br>Confg<br>Model||Easy<br>3042<br>3978<br>7220<br>-<br>-<br>Slow<br>Normal<br>Stop<br>aAcc.<br>wAcc.|Hard<br>4155<br>2882<br>715<br>6488<br>-<br>-<br>Slow<br>Normal<br>Fast<br>Stop<br>aAcc.<br>wAcc.|
|||||
||Modular<br>Fixed<br>GNN [37]<br>0.45<br>0.63<br>0.54<br>0.54<br>**0.53**<br>0.44<br>0.47<br>0.09<br>0.57<br>0.49<br>0.23<br>NLM [7]<br>0.31<br>0.57<br>0.75<br>0.61<br>0.49<br>0.39<br>0.54<br>0.11<br>0.48<br>0.45<br>**0.24**<br>Random<br>GNN [37]<br>0.52<br>0.63<br>0.43<br>0.51<br>0.54<br>0.26<br>0.51<br>0.19<br>0.63<br>0.48<br>0.28<br>NLM [7]<br>0.54<br>0.53<br>0.67<br>0.60<br>**0.56**<br>0.15<br>0.41<br>0.35<br>0.57<br>0.41<br>**0.36**|||
|||||
||E2E<br>Fixed<br>GNN [37]<br>NLM [7]<br>Random<br>GNN [37]<br>NLM [7]|0.76<br>0.69<br>0.98<br>0.85<br>**0.78**<br>0.78<br>0.47<br>1.00<br>0.83<br>0.71<br>0.88<br>0.64<br>1.00<br>0.87<br>**0.82**<br>0.90<br>0.53<br>1.00<br>0.85<br>0.79|0.46<br>0.62<br>0.27<br>0.99<br>0.72<br>0.40<br>0.33<br>0.69<br>0.37<br>1.00<br>0.71<br>**0.46**<br>0.14<br>0.66<br>0.52<br>1.00<br>0.65<br>**0.54**<br>0.25<br>0.67<br>0.45<br>1.00<br>0.69<br>0.50|




![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_LogiCity_Advancing_Neuro-Symbolic_AI_with_Abstract_Urban_Simulation_images/nips_2024_LogiCity_Advancing_Neuro-Symbolic_AI_with_Abstract_Urban_Simulation.pdf-0008-02.png)


Figure 5: Diverse renderings from LogiCity. Note that every city has distinct agent compositions. 

complex scenarios from the _medium_ mode. NeSy rule induction methods [7, 11] outperform pure neural MLP/GNN approaches. In the RL track, off-policy methods [7, 34, 36, 38] are more stable and effective than on-policy methods [32, 33] due to the high variance in training episodes affecting policy learning. Additionally, NeSy framework [7, 34] outperform pure neural agents [34, 36] by finding better representations from _abstract_ observations. To illustrate the compositional challenge in LogiCity, we compare results across different agent sets in Figure 3. Models trained on the training agent configuration show significant performance drops when transferred to test agents, but NeSy methods [7, 34] are less affected. We discuss more observation spaces in Appendix D. 

## **5.1.2 Continual Learning** 

Using LogiCity, we also examine how much data different models need to continually learn new _abstract_ rules. We initialize models with the converged weights from _easy_ mode and progressively provide data from _medium_ mode rules. The results from three random runs for MLP and NLM [7] are shown in Figure 4, alongside results from models trained from scratch. NLM reaches the best result with 30% of the target domain data, demonstrating superior continual learning capabilities. 

## **5.2 Visual Action Prediction** 

**Baselines.** As there exists very limited literature [72] studying FOL reasoning on RGB images, we self-construct two baselines using GNN [37] and NLM [7] as the reasoning engine, respectively. For fairness, we use the same visual encoder [2, 70] and hyperparameter configurations. We train and test all the models on a single NVIDIA H100 GPU. See Appendix A for more details. 

**Settings.** We explore four distinct training settings for the two methods. Regarding supervision signals, modular supervision offers ground truth for both scene graphs and final actions, training the two modules separately. This setting requires interpretable meanings of the scene graph elements, 

8 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_LogiCity_Advancing_Neuro-Symbolic_AI_with_Abstract_Urban_Simulation_images/nips_2024_LogiCity_Advancing_Neuro-Symbolic_AI_with_Abstract_Urban_Simulation.pdf-0009-00.png)


Figure 6: Continual learning results of GNN [37] and NLM [7] in the VAP task. The mean results from three random runs are displayed in solid lines and the variance is reported as the semi-transparent regions. We also show the results of the models trained from scratch using 100% data in dashed lines. 

which is crucial. We also explore end-to-end supervision (E2E), which provides guidance only on the final actions. For the training agent sets, we experiment with both fixed and random settings. 

**Modes and Datasets.** We present two modes for VAP task, namely _easy_ and _hard_ . In _easy_ mode, rules constrain only `Slow` and `Stop` actions with few _concepts_ . The _hard_ mode includes the _easy abstractions_ and additional constraints for all four actions, with a natural data imbalance making the `Fast` action rarer. We display some examples in Figure 5. More details are included in Appendix B. 

**Metrics.** We first report the action-wise recall rate (true positives divided by the number of samples). The average accuracy (aAcc.) is the correct prediction rate across all the test samples. To highlight the data imbalance issue, we also introduce weighted accuracy (wAcc.), defined as � `a`[Recall] `[a]` _[/][N]_ `[ a]` wAcc _._ = 1 , where Recall `[a]` is the recall rate for action `a` and _N_ `[a]` is the number of samples � `a` _N_ ~~`[a]`~~ for action `a` . This metric assigns larger weights to less frequent actions. 

## **5.2.1 Empirical Evaluation** 

The empirical results for the VAP task are shown in Table 3, highlighting LogiCity’s ability to adjust reasoning complexity. We observe that while GNN [37] slightly outperforms NLM [7] in the _easy_ mode, NLM excels in the _hard_ mode. We also find that random agent configurations improve the performance of both methods. The data imbalance poses an additional challenge, with the `Stop` action having 2 _× ∼_ 6 _×_ higher recall than the `Fast` action. Besides, the modular setting proves more challenging than the end-to-end (E2E) setting, as the modular system is more sensitive to perceptual noise. We further investigate this issue quantitatively in Appendix E. 

## **5.2.2 Continual Learning** 

Similar to the SPF task, we investigate how much data the methods need to continually learn abstract rules in the VAP task. The models pre-trained in _easy_ mode are used as the initial weights, which are continually trained with different sets of data from the _hard_ mode. The data are sampled for 3 times and the mean and variance of the results are reported in Figure 6, where we also report the results from the models trained with 100% data from scratch as dashed lines (“upper bound”). We observe that the two methods could struggle to reach their “upper bound” if fixed training agents are used. For the random agent setting, NLM [7] could progressively learn new rules and reach its “upper bound" with around 50% data while GNN fails even with 100% data. 

## **5.2.3 How Do LLMs and Human Perform in LogiCity?** 

Recent years have witnessed the increasing use of LLMs for decision making [73–76], concept understanding [77–79], and logical reasoning [80–83]. In this section, we investigate the capability of LLMs [1] and Human to solve the (subset of) VAP task in LogiCity through in-context demonstrations. Since we focus only on logical reasoning, true _groundings_ are provided in natural language documents without perceptual noise. Specifically, we first convert every scene (frame) into a paragraph of natural language description (see Figure B for examples). For each entity within the frame, given the scene descriptions, we ask LLMs to decide its next action from options (“A. Slow”, “B. Normal”, “C. Fast”, “D. Stop”). Since the entire test set of VAP is huge, we randomly selected a “Mini" test with 117 questions about the concept `IsTiro` and `IsReckless` . To construct demonstrations for 

9 

in-context learning, we randomly choose 5-shot samples from the training document used by human participants[2] and provide question-answer pairs. The performance of Human, `gpt-4o` (GPT-4o), `gpt-4o-mini` (GPT-4o mini), `gpt-4-turbo-2024-04-09` (GPT-4), and `gpt-3.5-turbo-1106` (GPT-3.5) on VAP hard mode test sets are reported in Table 4, where the random sampling results for options are also provided for reference. Based on experts’ evaluation, we also display the “hardness" of correctly answering each of the choice, where _[†]_ , _[††]_ , and _[†††]_ denote “easy", “medium", and “hard". 

We observe that the latest GPT-4o shows significantly better in-context learning capability than previous GPT-4 and GPT-3.5, surpassing them by over 20% in terms of overall accuracy. The results also demonstrate the importance of model scale for reasoning task, where GPT-4o mini falls far behind GPT-4o. However, it is still far from the inductive logical reasoning capability of Human, especially for harder reasoning choices like “Stop". Interestingly, the distribution of the decisions demonstrates that GPT-4 has a strong bias towards a conservative decision, which tends to 

Table 4: Action prediction accuracy of different LLMs in the VAP task hard mode. _[†]_ , _[††]_ , and _[†††]_ denote different logical reasoning hardness. 

||Method|Slow_††_|Normal_††_|Fast_†_|Stop_†††_|Overall|
|---|---|---|---|---|---|---|
||Human|**95.0**|**92.9**|48.0|**83.3**|**81.2**|
||GPT-4o|20.0|84.1|80.0|32.2|59.0|
||GPT-4|75.0|57.9|25.3|2.2|39.6|
||GPT-3.5<br>GPT-4o mini<br>Random|0.0<br>0.0<br>21.0|82.5<br>2.4<br>23.8|16.0<br>**86.7**<br>28.8|0.0<br>40.0<br>27.3|33.0<br>29.6<br>25.3|



predict “Slow” action. GPT-4o is better at reasoning in the context, yet they still tend to use common sense knowledge ( _e.g._ , Reckless cars always drive fast). In contrast, human participants tend to learn LogiCity’s rules through formal verification, where hypotheses are verified and refined based on training documents. Yet, due to the challenging nature of logical induction, human has made mistakes in learning rules of “Stop" (they concluded more general rules than GT), which affects the accuracy of “Fast". This suggests a promising future research direction that could involve coupling LLMs with a formal inductive logical reasoner [10, 35], creating a generation-verification loop. Another intriguing direction is using the LogiCity dataset to conduct Direct Preference Optimization (DPO) [84]. 

## **6 Discussions** 

**Conclusion.** This work presents LogiCity, a new simulator and benchmark for the NeSy AI community, featuring a dynamic urban environment with various _abstractions_ . LogiCity allows for flexible configuration on the _concepts_ and FOL rules, thus supporting the customization of logical reasoning complexity. Using the LogiCity simulator, we present sequential decision-making and visual reasoning tasks, both emphasizing _abstract_ reasoning. The former task is designed for a long-horizon, multi-agent interaction scenario while the latter focuses on reasoning with perceptual noise. With exhaustive experiments on various baselines, we show that NeSy frameworks [7, 11] can learn _abstractions_ better, and are thus more capable of the compositional generalization tests. Yet, LogiCity also demonstrates the challenge of learning _abstractions_ for all current methods, especially when the reasoning becomes more complex. Specifically, we highlight the crucial difficulty of longhorizon _abstract_ reasoning with multiple agents and that _abstract_ reasoning with high dimensional data remains hard. On the one hand, LogiCity poses a significant challenge for existing approaches with sophisticated reasoning scenarios. On the other hand, it allows for the gradual development of the next-generation NeSy AI by providing a flexible environment. 

**Limitations and Social Impact.** One limitation of our simulator is the need for users to pre-define rule sets that are conflict-free and do not cause deadlocks. Future work could involve distilling real-world data into configurations for LogiCity, streamlining this definition process. Currently, LogiCity does not support temporal logic [40]; incorporating temporal constraints on agent behaviors is intriguing. The simulation in LogiCity is deterministic, introducing randomness through fuzzy logic deduction engines [8, 9] could be interesting. For the autonomous driving community [25, 62], LogiCity introduces more various _concepts_ , requiring a model to plan with _abstractions_ , thus addressing a new aspect of real-life challenges. Enhancing the map of LogiCity and expanding the action space could make our simulator a valuable test bed for them. Additionally, since the ontologies and rules in LogiCity can be easily converted into Planning Definite Domain Language (PDDL), LogiCity has potential applications in _multi-agent_ task and motion planning [12, 85]. A potential negative social impact is that rules in LogiCity may not accurately reflect our real life, introducing sim-to-real gap. 

2Since human are able to learn from more samples without the context window limitation, they have read more training documents than LLMs for a more comprehensive understanding of LogiCity. 

10 

## **Acknowledgment** 

We acknowledge the support of the Air Force Research Laboratory (AFRL), DARPA, under agreement number FA8750-23-2-1015. This work used Bridges-2 at PSC through allocation cis220039p from the Advanced Cyberinfrastructure Coordination Ecosystem: Services & Support (ACCESS) program which is supported by NSF grants #2138259, #2138286, #2138307, #2137603, and #213296. This work was also supported, in part, by Individual Discovery Grants from the Natural Sciences and Engineering Research Council of Canada, and the Canada CIFAR AI Chair Program. We thank the Microsoft Accelerating Foundation Models Research (AFMR) program for providing generous support of Azure credits. We express sincere gratitude to all the human participants for their valuable time and intelligence devotion in the this research. The authors would also like to express sincere gratitude to Jiayuan Mao (MIT), Dr. Patrick Emami (NREL), and Dr. Peter Graf (NREL) for their valuable feedback and suggestions on the early draft of this work. 

11 

## **References** 

- [1] J. Achiam, S. Adler, S. Agarwal, L. Ahmad, I. Akkaya, F. L. Aleman, D. Almeida, J. Altenschmidt, S. Altman, S. Anadkat _et al._ , “Gpt-4 technical report,” _arXiv preprint arXiv:2303.08774_ , 2023. 

- [2] K. He, X. Zhang, S. Ren, and J. Sun, “Deep Residual Learning for Image Recognition,” in _Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)_ , 2016, pp. 770–778. 

- [3] J. B. Tenenbaum, C. Kemp, T. L. Griffiths, and N. D. Goodman, “How to Grow a Mind: Statistics, Structure, and Abstraction,” _Science_ , pp. 1279–1285, 2011. 

- [4] B. M. Lake, R. Salakhutdinov, and J. B. Tenenbaum, “Human-level Concept Learning through Probabilistic Program Induction,” _Science_ , pp. 1332–1338, 2015. 

- [5] E. S. Spelke and K. D. Kinzler, “Core Knowledge,” _Developmental Science_ , pp. 89–96, 2007. 

- [6] H. Kautz, “The third ai summer: Aaai robert s. engelmore memorial lecture,” _AI Magazine_ , pp. 105–125, 2022. 

- [7] H. Dong, J. Mao, T. Lin, C. Wang, L. Li, and D. Zhou, “Neural Logic Machines,” in _Proceedings of the International Conference on Learning Representations (ICLR)_ , 2019, pp. 1–10. 

- [8] D. Tran, M. D. Hoffman, R. A. Saurous, E. Brevdo, K. Murphy, and D. M. Blei, “Deep Probabilistic Programming,” in _Proceedings of the International Conference on Learning Representations (ICLR)_ , 2017, pp. 1–10. 

- [9] R. Riegel, A. Gray, F. Luus, N. Khan, N. Makondo, I. Y. Akhalwaya, H. Qian, R. Fagin, F. Barahona, U. Sharma, S. Ikbal, H. Karanam, S. Neelam, A. Likhyani, and S. Srivastava, “Logical Neural Networks,” _arXiv preprint arXiv:2006.13155_ , 2020. 

- [10] A. Cropper and R. Morel, “Learning Programs by Learning from Failures,” _Machine Learning_ , pp. 801–856, 2021. 

- [11] C. Glanois, Z. Jiang, X. Feng, P. Weng, M. Zimmer, D. Li, W. Liu, and J. Hao, “Neuro-Symbolic Hierarchical Rule Induction,” in _Proceedings of the International Conference on Machine Learning (ICML)_ , 2022, pp. 7583–7615. 

- [12] R. Chitnis, T. Silver, J. B. Tenenbaum, T. Lozano-Perez, and L. P. Kaelbling, “Learning Neuro-Symbolic Relational Transition Models for Bilevel Planning,” in _Proceedings of the IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS)_ , 2022, pp. 4166–4173. 

- [13] Q. Li, S. Huang, Y. Hong, Y. Chen, Y. N. Wu, and S.-C. Zhu, “Closed Loop Neural-Symbolic Learning Via Integrating Neural Perception, Grammar Parsing, and Symbolic Reasoning,” in _Proceedings of the International Conference on Machine Learning (ICML)_ , 2020, pp. 5884–5894. 

- [14] S. Bhagat, S. Stepputtis, J. Campbell, and K. Sycara, “Sample-Efficient Learning of Novel Visual Concepts,” in _Proceedings of the Conference on Lifelong Learning Agents (CoLLAs)_ , 2023, pp. 637–657. 

- [15] S. Li, S. Bhagat, J. Campbell, Y. Xie, W. Kim, K. Sycara, and S. Stepputtis, “ShapeGrasp: Zero-Shot Task-Oriented Grasping with Large Language Models through Geometric Decomposition,” 2024. 

- [16] C. Zhang, S. Stepputtis, J. Campbell, K. Sycara, and Y. Xie, “HiKER-SGG: Hierarchical Knowledge Enhanced Robust Scene Graph Generation,” _arXiv preprint arXiv:2403.12033_ , 2024. 

- [17] J. Hsu, J. Mao, and J. Wu, “NS3D: Neuro-Symbolic Grounding of 3D Objects and Relations,” in _Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)_ , 2023, pp. 2614–2623. 

- [18] J. Mao, C. Gan, P. Kohli, J. B. Tenenbaum, and J. Wu, “The Neuro-Symbolic Concept Learner: Interpreting Scenes, Words, and Sentences from Natural Supervision,” in _Proceedings of the International Conference on Learning Representations (ICLR)_ , 2019, pp. 1–10. 

- [19] P.-W. Wang, P. Donti, B. Wilder, and Z. Kolter, “SATNet: Bridging Deep Learning and Logical Reasoning using a Differentiable Satisfiability Solver,” in _Proceedings of the International Conference on Machine Learning (ICML)_ , 2019, pp. 6545–6554. 

- [20] S. Badreddine, A. d. Garcez, L. Serafini, and M. Spranger, “Logic Tensor Networks,” _Artificial Intelligence_ , p. 103649, 2022. 

12 

- [21] J. Johnson, B. Hariharan, L. Van Der Maaten, L. Fei-Fei, C. Lawrence Zitnick, and R. Girshick, “CLEVR: A Diagnostic Dataset for Compositional Language and Elementary Visual Reasoning,” in _Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)_ , 2017, pp. 2901–2910. 

- [22] M. C. Machado, M. G. Bellemare, E. Talvitie, J. Veness, M. Hausknecht, and M. Bowling, “Revisiting the Arcade Learning Environment: Evaluation Protocols and Open Problems for General Agents,” _Journal of Artificial Intelligence Research_ , pp. 523–562, 2018. 

- [23] M. Chevalier-Boisvert, B. Dai, M. Towers, R. Perez-Vicente, L. Willems, S. Lahlou, S. Pal, P. S. Castro, and J. Terry, “MiniGrid & MiniWorld: Modular & Customizable Reinforcement Learning Environments for Goal-Oriented Tasks,” in _Proceedings of the Advances in Neural Information Processing Systems (NeurIPS)_ , 2023, pp. 73 383–73 394. 

- [24] M. Chevalier-Boisvert, D. Bahdanau, S. Lahlou, L. Willems, C. Saharia, T. H. Nguyen, and Y. Bengio, “BabyAI: First Steps Towards Grounded Language Learning with a Human in the Loop,” in _Proceedings of the International Conference on Learning Representations (ICLR)_ , 2019, pp. 1–10. 

- [25] E. Leurent, “An Environment for Autonomous Driving Decision-Making,” https://github.com/eleurent/ highway-env, 2018. 

- [26] H. Küttler, N. Nardelli, A. Miller, R. Raileanu, M. Selvatici, E. Grefenstette, and T. Rocktäschel, “The NetHack Learning Environment,” in _Proceedings of the Advances in Neural Information Processing Systems (NeurIPS)_ , 2020, pp. 7671–7684. 

- [27] L. Fan, G. Wang, Y. Jiang, A. Mandlekar, Y. Yang, H. Zhu, A. Tang, D.-A. Huang, Y. Zhu, and A. Anandkumar, “MineDojo: Building Open-Ended Embodied Agents with Internet-Scale Knowledge,” in _Proceedings of the Advances in Neural Information Processing Systems (NeurIPS)_ , 2022, pp. 18 343–18 362. 

- [28] R. Kowalski, “Predicate Logic as a Programming Language,” in _Proceedings of the IFIP Congress_ , 1974, pp. 569–574. 

- [29] D. Podell, Z. English, K. Lacey, A. Blattmann, T. Dockhorn, J. Müller, J. Penna, and R. Rombach, “SDXL: Improving Latent Diffusion Models for High-Resolution Image Synthesis,” _arXiv preprint arXiv:2307.01952_ , 2023. 

- [30] R. Rombach, A. Blattmann, D. Lorenz, P. Esser, and B. Ommer, “High-Resolution Image Synthesis with Latent Diffusion Models,” in _Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)_ , 2021, pp. 10 684–10 695. 

- [31] C. Meng, Y. He, Y. Song, J. Song, J. Wu, J.-Y. Zhu, and S. Ermon, “SDEdit: Guided Image Synthesis and Editing with Stochastic Differential Equations,” in _Proceedings of the International Conference on Learning Representations (ICLR)_ , 2022. 

- [32] J. Schulman, F. Wolski, P. Dhariwal, A. Radford, and O. Klimov, “Proximal Policy Optimization Algorithms,” _arXiv preprint arXiv:1707.06347_ , 2017. 

- [33] V. Mnih, A. P. Badia, M. Mirza, A. Graves, T. Lillicrap, T. Harley, D. Silver, and K. Kavukcuoglu, “Asynchronous Methods for Deep Reinforcement Learning,” in _Proceedings of the International Conference on Machine Learning (ICML)_ , 2016, pp. 1928–1937. 

- [34] V. Mnih, K. Kavukcuoglu, D. Silver, A. Graves, I. Antonoglou, D. Wierstra, and M. Riedmiller, “Playing Atari with Deep Reinforcement Learning,” _arXiv preprint arXiv:1312.5602_ , 2013. 

- [35] C. Hocquette, A. Niskanen, M. Järvisalo, and A. Cropper, “Learning MDL Logic Programs from Noisy Data,” in _Proceedings of the AAAI Conference on Artificial Intelligence (AAAI)_ , 2024, pp. 10 553–10 561. 

- [36] A. Nagabandi, K. Konolige, S. Levine, and V. Kumar, “Deep Dynamics Models for Learning Dexterous Manipulation,” in _Proceedings of the Annual Conference on Robot Learning (CoRL)_ , 2020, pp. 1101–1112. 

- [37] K. Xu, W. Hu, J. Leskovec, and S. Jegelka, “How Powerful Are Graph Neural Networks?” in _Proceedings of the International Conference on Learning Representations (ICLR)_ , 2019, pp. 1–10. 

- [38] D. Hafner, T. P. Lillicrap, M. Norouzi, and J. Ba, “Mastering Atari with Discrete World Models,” in _Proceedings of the International Conference on Learning Representations (ICLR)_ , 2020, pp. 1–10. 

- [39] J. Xu, Z. Zhang, T. Friedman, Y. Liang, and G. Broeck, “A Semantic Loss Function for Deep Learning with Symbolic Knowledge,” in _Proceedings of the International Conference on Machine Learning (ICML)_ , 2018, pp. 5502–5511. 

13 

- [40] Y. Xie, Z. Xu, M. S. Kankanhalli, K. S. Meel, and H. Soh, “Embedding Symbolic Knowledge into Deep Networks,” in _Proceedings of the Advances in Neural Information Processing Systems (NeurIPS)_ , 2019, pp. 1–9. 

- [41] J. Huang, Z. Li, B. Chen, K. Samel, M. Naik, L. Song, and X. Si, “Scallop: From Probabilistic Deductive Databases to Scalable Differentiable Reasoning,” in _Proceedings of the Advances in Neural Information Processing Systems (NeurIPS)_ , 2021, pp. 25 134–25 145. 

- [42] R. Wang, J. Mao, J. Hsu, H. Zhao, J. Wu, and Y. Gao, “Programmatically Grounded, Compositionally Generalizable Robotic Manipulation,” in _Proceedings of the International Conference on Learning Representations (ICLR)_ , 2022, pp. 1–9. 

- [43] J. Hsu, J. Mao, J. Tenenbaum, and J. Wu, “What’s Left? Concept Grounding with Logic-Enhanced Foundation Models,” in _Proceedings of the Advances in Neural Information Processing Systems (NeurIPS)_ , 2023, pp. 38 798–38 814. 

- [44] Z. Yang, A. Ishay, and J. Lee, “NeurASP: Embracing Neural Networks into Answer Set Programming,” in _Proceedings of the International Joint Conference on Artificial Intelligence (IJCAI)_ , 2020, pp. 1755–1762. 

- [45] T. Winters, G. Marra, R. Manhaeve, and L. De Raedt, “DeepStochLog: Neural Stochastic Logic Programming,” in _Proceedings of the AAAI Conference on Artificial Intelligence (AAAI)_ , 2022, pp. 10 090–10 100. 

- [46] E. van Krieken, T. Thanapalasingam, J. Tomczak, F. Van Harmelen, and A. Ten Teije, “A-NESI: A Scalable Approximate Method for Probabilistic Neurosymbolic Inference,” _Proceedings of the Advances in Neural Information Processing Systems (NeurIPS)_ , pp. 24 586–24 609, 2023. 

- [47] W. Liu, G. Chen, J. Hsu, J. Mao, and J. Wu, “Learning Planning Abstractions from Language,” in _Proceedings of the International Conference on Learning Representations (ICLR)_ , 2022, pp. 1–9. 

- [48] R. Evans and E. Grefenstette, “Learning Explanatory Rules from Noisy Data,” _Journal of Artificial Intelligence Research_ , pp. 1–64, 2018. 

- [49] F. Yang, Z. Yang, and W. W. Cohen, “Differentiable Learning of Logical Rules for Knowledge Base Reasoning,” in _Proceedings of the Advances in Neural Information Processing Systems (NeurIPS)_ , 2017, pp. 1–10. 

- [50] Y. Yang and L. Song, “Learn to Explain Efficiently via Neural Logic Inductive Learning,” in _Proceedings of the International Conference on Learning Representations (ICLR)_ , 2019, pp. 1–10. 

- [51] P. Sen, B. W. de Carvalho, R. Riegel, and A. Gray, “Neuro-Symbolic Inductive Logic Programming with Logical Neural Networks,” in _Proceedings of the AAAI Conference on Artificial Intelligence (AAAI)_ , 2022, pp. 8212–8219. 

- [52] Q. Delfosse, H. Shindo, D. Dhami, and K. Kersting, “Interpretable and Explainable Logical Policies via Neurally Guided Symbolic Abstraction,” in _Proceedings of the Advances in Neural Information Processing Systems (NeurIPS)_ , 2023, pp. 50 838–50 858. 

- [53] D. Kimura, M. Ono, S. Chaudhury, R. Kohita, A. Wachi, D. J. Agravante, M. Tatsubori, A. Munawar, and A. Gray, “Neuro-Symbolic Reinforcement Learning with First-Order Logic,” in _Proceedings of the Conference on Empirical Methods in Natural Language Processing (EMNLP)_ , 2021, pp. 3505–3511. 

- [54] X. Si, M. Raghothaman, K. Heo, and M. Naik, “Synthesizing Datalog Programs using Numerical Relaxation,” in _Proceedings of the International Joint Conference on Artificial Intelligence (IJCAI)_ , 2019, pp. 6117–6124. 

- [55] Z. Li, J. Guo, Y. Jiang, and X. Si, “Learning Reliable Logical Rules with SATNet,” _Proceedings of the Advances in Neural Information Processing Systems (NeurIPS)_ , pp. 14 837–14 847, 2024. 

- [56] Z. Li, Y. Huang, Z. Li, Y. Yao, J. Xu, T. Chen, X. Ma, and J. Lu, “Neuro-symbolic Learning Yielding Logical Constraints,” _Proceedings of the Advances in Neural Information Processing Systems (NeurIPS)_ , pp. 21 635–21 657, 2024. 

- [57] O. Vinyals, T. Ewalds, S. Bartunov, P. Georgiev, A. S. Vezhnevets, M. Yeo, A. Makhzani, H. Küttler, J. Agapiou, J. Schrittwieser _et al._ , “StarCraft II: A New Challenge for Reinforcement Learning,” _arXiv preprint arXiv:1708.04782_ , 2017. 

- [58] M. Nicolau, D. Perez-Liebana, M. O’Neill, and A. Brabazon, “Evolutionary Behavior Tree Approaches for Navigating Platform Games,” _IEEE Transactions on Computational Intelligence and AI in Games_ , pp. 227–238, 2016. 

14 

- [59] Y. A. Sekhavat, “Behavior Trees for Computer Games,” _International Journal on Artificial Intelligence Tools_ , p. 1730001, 2017. 

- [60] M. Iovino, E. Scukins, J. Styrud, P. Ögren, and C. Smith, “A survey of Behavior Trees in Robotics and AI,” _Robotics and Autonomous Systems_ , p. 104096, 2022. 

- [61] M. Colledanchise and P. Ögren, _Behavior trees in robotics and AI: An introduction_ . 

   - CRC Press, 2018. 

- [62] A. Dosovitskiy, G. Ros, F. Codevilla, A. Lopez, and V. Koltun, “CARLA: An Open Urban Driving Simulator,” in _Proceedings of the Annual Conference on Robot Learning (CoRL)_ , 2017, pp. 1–16. 

- [63] D. Krajzewicz, “Traffic Simulation with SUMO–Simulation of Urban Mobility,” _Fundamentals of Traffic Simulation_ , pp. 269–293, 2010. 

- [64] S. Shah, D. Dey, C. Lovett, and A. Kapoor, “AirSim: High-Fidelity Visual and Physical Simulation for Autonomous Vehicles,” in _Field and Service Robotics: Results of the 11th International Conference_ , 2018, pp. 621–635. 

- [65] D. J. Fremont, T. Dreossi, S. Ghosh, X. Yue, A. L. Sangiovanni-Vincentelli, and S. A. Seshia, “Scenic: A Language for Scenario Specification and Scene Generation,” in _Proceedings of the ACM SIGPLAN Conference on Programming Language Design and Implementation (PLDI)_ , 2019, pp. 63–78. 

- [66] D. J. Fremont, E. Kim, T. Dreossi, S. Ghosh, X. Yue, A. L. Sangiovanni-Vincentelli, and S. A. Seshia, “Scenic: A Language for Scenario Specification and Data Generation,” _Machine Learning_ , pp. 3805–3849, 2023. 

- [67] E. Vin, S. Kashiwa, M. Rhea, D. J. Fremont, E. Kim, T. Dreossi, S. Ghosh, X. Yue, A. L. SangiovanniVincentelli, and S. A. Seshia, “3D Environment Modeling for Falsification and Beyond with Scenic 3.0,” in _Proceedings of the International Conference on Computer Aided Verification (CAV)_ , 2023, pp. 253–265. 

- [68] P. E. Hart, N. J. Nilsson, and B. Raphael, “A Formal Basis for the Heuristic Determination of Minimum Cost Paths,” _IEEE Transactions on Systems Science and Cybernetics_ , pp. 100–107, 1968. 

- [69] L. De Moura and N. Bjørner, “Z3: An Efficient SMT Solver,” in _Proceedings of the International Conference on Tools and Algorithms for the Construction and Analysis of Systems_ , 2008, pp. 337–340. 

- [70] T.-Y. Lin, P. Dollár, R. Girshick, K. He, B. Hariharan, and S. Belongie, “Feature Pyramid Networks for Object Detection,” in _Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)_ , 2017, pp. 2117–2125. 

- [71] S. Ren, K. He, R. Girshick, and J. Sun, “Faster R-CNN: Towards Real-Time Object Detection with Region Proposal Networks,” _IEEE Transactions on Pattern Analysis and Machine Intelligence_ , pp. 1137–1149, 2016. 

- [72] Y. Yang, J. C. Kerce, and F. Fekri, “LogicDef: An Interpretable Defense Framework Against Adversarial Examples via Inductive Scene Graph Reasoning,” in _Proceedings of the AAAI Conference on Artificial Intelligence (AAAI)_ , 2022, pp. 8840–8848. 

- [73] T. Sumers, S. Yao, K. Narasimhan, and T. Griffiths, “Cognitive architectures for language agents,” _Transactions on Machine Learning Research_ , 2024. 

- [74] S. Yao, D. Yu, J. Zhao, I. Shafran, T. L. Griffiths, Y. Cao, and K. Narasimhan, “Tree of Thoughts: Deliberate problem solving with large language models,” 2023. 

- [75] M. Li, S. Zhao, Q. Wang, K. Wang, Y. Zhou, S. Srivastava, C. Gokmen, T. Lee, L. E. Li, R. Zhang, W. Liu, P. Liang, L. Fei-Fei, J. Mao, and J. Wu, “Embodied agent interface: Benchmarking llms for embodied decision making,” 2024. 

- [76] Y. Zhai, H. Bai, Z. Lin, J. Pan, S. Tong, Y. Zhou, A. Suhr, S. Xie, Y. LeCun, Y. Ma _et al._ , “Fine-tuning large vision-language models as decision-making agents via reinforcement learning,” _arXiv preprint arXiv:2405.10292_ , 2024. 

- [77] G. Rajendran, S. Buchholz, B. Aragam, B. Schölkopf, and P. Ravikumar, “Learning interpretable concepts: Unifying causal representation learning and foundation models,” 2024. 

- [78] J. Luo, T. Ding, K. H. R. Chan, D. Thaker, A. Chattopadhyay, C. Callison-Burch, and R. Vidal, “Pace: Parsimonious concept engineering for large language models,” 2024. 

- [79] D. P. Jeong, Z. C. Lipton, and P. Ravikumar, “Llm-select: Feature selection with large language models,” 2024. 

15 

- [80] C. Hu, J. Fu, C. Du, S. Luo, J. J. Zhao, and H. Zhao, “Chatdb: Augmenting llms with databases as their symbolic memory,” 2023. 

- [81] S. Han, H. Schoelkopf, Y. Zhao, Z. Qi, M. Riddell, W. Zhou, J. Coady, D. Peng, Y. Qiao, L. Benson, L. Sun, A. Wardle-Solano, H. Szabo, E. Zubova, M. Burtell, J. Fan, Y. Liu, B. Wong, M. Sailor, A. Ni, L. Nan, J. Kasai, T. Yu, R. Zhang, A. R. Fabbri, W. Kryscinski, S. Yavuz, Y. Liu, X. V. Lin, S. Joty, Y. Zhou, C. Xiong, R. Ying, A. Cohan, and D. Radev, “Folio: Natural language reasoning with first-order logic,” 2022. 

- [82] L. Pan, A. Albalak, X. Wang, and W. Y. Wang, “Logic-lm: Empowering large language models with symbolic solvers for faithful logical reasoning,” 2023. 

- [83] H. Sun, W. Xu, W. Liu, J. Luan, B. Wang, S. Shang, J.-R. Wen, and R. Yan, “Determlr: Augmenting llm-based logical reasoning from indeterminacy to determinacy,” in _ACL_ , 2024. 

- [84] R. Rafailov, A. Sharma, E. Mitchell, C. D. Manning, S. Ermon, and C. Finn, “Direct Preference Optimization: Your Language Model is Secretly a Reward Model,” in _Proceedings of the Advances in Neural Information Processing Systems (NeurIPS)_ , 2023, pp. 1–10. 

- [85] T. Silver, A. Athalye, J. B. Tenenbaum, T. Lozano-Pérez, and L. P. Kaelbling, “Learning Neuro-Symbolic Skills for Bilevel Planning,” in _Proceedings of the Annual Conference on Robot Learning (CoRL)_ , 2022, pp. 1–8. 

- [86] D. P. Kingma and J. Ba, “Adam: A Method for Stochastic Optimization,” _arXiv preprint arXiv:1412.6980_ , 2014. 

- [87] J. Schulman, S. Levine, P. Abbeel, M. Jordan, and P. Moritz, “Trust Region Policy Optimization,” in _Proceedings of the International Conference on Machine Learning (ICML)_ , 2015, pp. 1889–1897. 

- [88] A. Raffin, A. Hill, A. Gleave, A. Kanervisto, M. Ernestus, and N. Dormann, “Stable-Baselines3: Reliable Reinforcement Learning Implementations,” _Journal of Machine Learning Research_ , pp. 1–8, 2021. 

- [89] J. Deng, W. Dong, R. Socher, L.-J. Li, K. Li, and L. Fei-Fei, “Imagenet: A Large-Scale Hierarchical Image Database,” in _Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)_ , 2009, pp. 248–255. 

- [90] I. Loshchilov and F. Hutter, “Decoupled Weight Decay Regularization,” in _Proceedings of the International Conference on Learning Representations (ICLR)_ , 2018, pp. 1–9. 

- [91] A. Bordes, N. Usunier, A. Garcia-Duran, J. Weston, and O. Yakhnenko, “Translating Embeddings for Modeling Multi-Relational Data,” in _Proceedings of the Advances in Neural Information Processing Systems (NeurIPS)_ , 2013, pp. 1–8. 

- [92] K. Toutanova and D. Chen, “Observed Versus Latent Features for Knowledge Base and Text Inference,” in _Proceedings of the 3rd workshop on Continuous Vector Space Models and Their Compositionality_ , 2015, pp. 57–66. 

- [93] A. Kirillov, E. Mintun, N. Ravi, H. Mao, C. Rolland, L. Gustafson, T. Xiao, S. Whitehead, A. C. Berg, W.-Y. Lo _et al._ , “Segment Anything,” in _Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)_ , 2023, pp. 4015–4026. 

- [94] H. Zhang, P. Zhang, X. Hu, Y.-C. Chen, L. Li, X. Dai, L. Wang, L. Yuan, J.-N. Hwang, and J. Gao, “Glipv2: Unifying Localization and Vision-Language Understanding,” in _Proceedings of the Advances in Neural Information Processing Systems (NeurIPS)_ , 2022, pp. 36 067–36 080. 

## **Checklist** 

1. For all authors... 

   - (a) Do the main claims made in the abstract and introduction accurately reflect the paper’s contributions and scope? [Yes] , the claims made in the abstract and introduction accurately reflect the paper’s contributions and scope. 

   - (b) Did you describe the limitations of your work? [Yes] , the limitations are discussed in Section 6. 

   - (c) Did you discuss any potential negative societal impacts of your work? [Yes] , the potential negative societal impacts are discussed in Section 6. 

16 

   - (d) Have you read the ethics review guidelines and ensured that your paper conforms to them? [Yes] , our paper conforms to the ethics review guidelines. 

2. If you are including theoretical results... 

   - (a) Did you state the full set of assumptions of all theoretical results? [N/A] , our paper does not include theoretical results. 

   - (b) Did you include complete proofs of all theoretical results? [N/A] , our paper does not include theoretical results. 

3. If you ran experiments (e.g. for benchmarks)... 

   - (a) Did you include the code, data, and instructions needed to reproduce the main experimental results (either in the supplemental material or as a URL)? [Yes] , the code, data, and instructions are fully open sourced in our website (link in abstract). 

   - (b) Did you specify all the training details (e.g., data splits, hyperparameters, how they were chosen)? [Yes] , the training details can be found at Appendix A and our code library. 

   - (c) Did you report error bars (e.g., with respect to the random seed after running experiments multiple times)? [Yes] , in continual learning experiments, we report mean and variance of three random runs. 

   - (d) Did you include the total amount of compute and the type of resources used (e.g., type of GPUs, internal cluster, or cloud provider)? [Yes] , hardware information is included in Section 5. 

4. If you are using existing assets (e.g., code, data, models) or curating/releasing new assets... 

   - (a) If your work uses existing assets, did you cite the creators? [N/A] , our work does not use existing assets. 

   - (b) Did you mention the license of the assets? [N/A] , our work does not use existing assets. 

   - (c) Did you include any new assets either in the supplemental material or as a URL? [Yes] , new assets are in our website. 

   - (d) Did you discuss whether and how consent was obtained from people whose data you’re using/curating? [N/A] , the data are purely synthetic. 

   - (e) Did you discuss whether the data you are using/curating contains personally identifiable information or offensive content? [N/A] , the data does not contain personally identifiable information or offensive content. 

5. If you used crowdsourcing or conducted research with human subjects... 

   - (a) Did you include the full text of instructions given to participants and screenshots, if applicable? [Yes] , We have included the full text of instructions given to participants and screenshots. 

   - (b) Did you describe any potential participant risks, with links to Institutional Review Board (IRB) approvals, if applicable? [Yes] , we have described the potential participant risks, with links to Institutional Review Board (IRB) approvals. 

   - (c) Did you include the estimated hourly wage paid to participants and the total amount spent on participant compensation? [Yes] , the human participants received an hourly wage of 15$. 

17 

## **A Detailed Baseline Configurations** 

To make our experiments reproducible, we provided detailed baseline introductions and configurations below. For more details, please refer to our code base. 

## **A.1 Safe Path Following** 

In the BC branch, we have considered ILP methods [10, 35], including both symbolic ones [10, 35] and NeSy ones [7, 11]. For them, we convert the demonstration trajectories (step-wise truth value of all the predicates) into _facts_ and conduct rule learning. Popper [10] is one of the most performant search-based rule induction algorithm, which uses failure samples to construct hyposithes spaces via answer set programming. It shows better scaling capability than previous template based methods [48]. Since Popper is a greedy approach, it usually costs too much time searching. Maxsynth [35] relax this greedy setting and aims at finding rules in noisy data via anytime solvers. In our experiments, we set 300 seconds (averaged training time for other methods) as the maximum search time for Popper and Maxsynth. For all the other parameters, official default settings are used for fairness. HRI [11] is a hierachical rule induction framework, which utilizes neural tensors to represent predicates and searches the explicit rules by finding paths between predicates. For different modes in LogiCity, we provided the number of background predicates as HRI initialization. All the other parameter settings are kept the same as the original implementation. When constructing the scene graph, we make sure the ratio of positive and negative samples is 1:1. For the other NLM [7] is an implicity rule induction method, which proposed a FOL-inspired network structure. The learnt rules are implicity stored in the network weights. For different modes in LogiCity, we provided the number of background predicates as NLM initialization. Across different modes, we used the same hyperparameters, _i.e._ , the output dimension of each layer is set to 8, the maximum depth is set to 4, and the breadth is 3. For the baselines above, we used their official optimizer during training. In addition, we constructed pure neural baselines, including an MLP and a GNN [37], both having two hidden layers with ReLU activations. In the easy and medium modes, the dimensions of the hidden layers are 128 _×_ 64 and 64 _×_ 64. In the hard and expert modes, the dimensions of the hidden layers are 128 _×_ 64 and 64 _×_ 128. These self-constructed baselines are trained with Adam optimizer [86]. For more details, please refer to our open-sourced code library. 

In the RL branch, we first build neural agents using different algorithms, which are learnt by interaction with the environment. A2C [33] is a synchronous, deterministic variant of Asynchronous Advantage Actor Critic (A3C) [33], which is an on-policy framework. It leverages multiple workers to replace the use of replay buffer. Proximal Policy Optimization (PPO) combines the idea in A2C and the trust region optimization in TRPO [87]. Different from these policy gradient-based methods, Deep Q network (DQN) [34] is an off-policy value-based approach, which has been one of the state-of-the-arts in Atari Games [22]. For these three baselines, we used a two-layer MLP as the feature extractor, which has the same structure as the MLP baseline in the BC branch. All the other configurations are borrowed from stable-baselines3 [88]. In addition to these model-free agents, we also considered model-based approaches [36, 38]. MB-shooting [36] uses the learnt world model to evaluate the randomly sampled future trajectories. In our experiments, we used an ensemble of 50 MLPs (with the same structure as above) as the dynamics model. The reward prediction is modeled as a regression problem while the state prediction is a classification problem. During inference, we sample a total of 100 random action sequences with a horizon of 10. DreamerV2 [38] is a more advanced model-based method, which introduced discrete distribution in the latent world representation. We find the official implementation for Atari games [38] is hard to work for LogiCity. Therefore, we have tried our best to carefully tune the parameters, which can be found in our code library. Additionally, we built a NeSy agent [7] based on DQN [34], named as NLM-DQN, which we show the detailed structure in Figure A. The observed _groundings_ is first reshaped into a list of predicates, which is fed into NLM to obtain the invented (8) new predicates. Since we are learning ego policy (for the first entity), the first axis of the feature is extracted as the truth value grounded to the ego entity. Then, similar to the vanilla DQN, we construct two MLPs to estimate the current Q value and the next Q value, which, together with the current reward, are used to update the model based on Bellman Equation. Despite its simple structure, NLM-DQN has been demonstrated as the most performant baseline in LogiCity SPF task RL branch, showcasing the power of NeSy in terms of complicated abstract reasoning. All the baselines in the RL branch are trained for a total of 200 _k_ steps in the training environment, the most performant checkpoints in validation environment is 

18 

|obs<br>NLM<br>Reshape<br>Q Net (MLP)<br>[None, 11x5, 6x5x5, None]<br>Axis1: 5x8<br>Q Net Target<br>(MLP)<br>Next Q Value<br>Curr. Q Value<br>Bellman Eq.<br>Reward<br>Action<br>Index0: 1x8<br>Argmax|Reward|Reward|
|---|---|---|
||||
||Bellman Eq.||



Figure A: Model structure of NLM-DQN [7, 34]. We display the feature dimension for the hard mode for reference. 

Table A: All the predicates in LogiCity. We also display which parts of them are involved in each mode of the two tasks. 

|Task<br>Predicates<br>Arity<br>Description|SPF<br>Easy Medium Hard Expert|VAP<br>Easy Hard|
|---|---|---|
||||
|`IsPedestrian(X)`<br>1<br>Checks if entity`X`is a pedestrian.<br>`IsCar(X)`<br>1<br>Checks if entity`X`is a car.<br>`IsAmbulance(X)`<br>1<br>Checks if entity`X`is an ambulance.<br>`IsBus(X)`<br>1<br>Checks if entity`X`is a bus.<br>`IsPolice(X)`<br>1<br>Checks if entity`X`is a police vehicle.<br>`IsTiro(X)`<br>1<br>Checks if entity`X`is a tiro.<br>`IsReckless(X)`<br>1<br>Checks if entity`X`is reckless.<br>`IsOld(X)`<br>1<br>Checks if entity`X`is old.<br>`IsYoung(X)`<br>1<br>Checks if entity`X`is young.<br>`IsAtInter(X)`<br>1<br>Checks if entity`X`is at the intersection.<br>`IsInInter(X)`<br>1<br>Checks if entity`X`is in the intersection.<br>`IsClose(X, Y)`<br>2<br>Checks if entity`X`is close to entity`Y`.<br>`HigherPri(X, Y)`<br>2<br>Checks if entity`X`has higher priority than entity`Y`.<br>`CollidingClose(X, Y)`<br>2<br>Checks if entity`X`is about to collide with entity`Y`.<br>`LeftOf(X, Y)`<br>2<br>Checks if entity`X`is left of entity`Y`.<br>`RightOf(X, Y)`<br>2<br>Checks if entity`X`is right of entity`Y`.<br>`NextTo(X, Y)`<br>2<br>Checks if entity`X`is next to entity`Y`.|✓<br>✓<br>✓<br>✓<br>✓<br>✓<br>✓<br>✓<br>✓<br>✓<br>✓<br>✓<br>✗<br>✓<br>✓<br>✓<br>✗<br>✗<br>✓<br>✓<br>✓<br>✓<br>✓<br>✓<br>✗<br>✗<br>✓<br>✓<br>✓<br>✓<br>✓<br>✓<br>✗<br>✗<br>✓<br>✓<br>✓<br>✓<br>✓<br>✓<br>✓<br>✓<br>✓<br>✓<br>✗<br>✗<br>✓<br>✓<br>✓<br>✓<br>✓<br>✓<br>✓<br>✓<br>✓<br>✓<br>✗<br>✗<br>✓<br>✓<br>✗<br>✓<br>✓<br>✓<br>✗<br>✓<br>✓<br>✓|✓<br>✓<br>✓<br>✓<br>✓<br>✓<br>✓<br>✓<br>✓<br>✓<br>✓<br>✓<br>✓<br>✓<br>✓<br>✓<br>✓<br>✓<br>✓<br>✓<br>✓<br>✓<br>✓<br>✓<br>✓<br>✓<br>✓<br>✓<br>✓<br>✓<br>✓<br>✓<br>✓<br>✓|



utilized for testing. Note that this is different from existing gaming environments [22, 27, 57], where train/val/test environments have very limited distribution shift. 

## **A.2 Visual Action Prediction** 

In the VAP task, we built two baseline models with similar structure, namely GNN [37] and NLM [7]. Across the two models, we used the same grounding framework. Specifically, ResNet50 [2] plus Feature Pyramid Network (FPN) [70] pre-trained on ImageNet [89] is leveraged as the feature encoder. After ROIAlign [71], the resulting regional features are in the shape of R[512] . The unary predicate heads are three-layer MLPs with BatchNorm1D, ReLU, and Dropout functions. Note that the unary predicates are all about the regional feature, requiring no additional information **h** . On the other hand, the binary predicates are all about the additional information. We first concatinate the information **h** for each pair of entities and used two-layer MLPs to predicate the truth values of binary predicates. For details about the structure of the MLPs, please see our code library. The truth values of unary and binary predicates form a scene graph for the reasoning networks [7, 37] to predict actions. For GNN [37], we used a hidden layer in the dimension of 128. For NLM [7], we employed official implementation, where each logic layer invents 8 new attributes, the maximum depth is set to 4 and the breadth is set to 3. In the end-to-end setting, both methods are trained using AdamW [90]. In the modular setting, the grounding module is trained using Adam [86] while the reasoning module is optimized using AdamW [90]. For all the experiments, we train the models for 30 epochs and test the best performing checkpoint in the validation set. Note that these settings are the same in the two modes of VAP task. 

## **B Detailed Task Configurations** 

The full list of predicates and rules and their descriptions are displayed in Table A and Table B, respectively. Across different modes in the two tasks, the involved predicates and rule clauses are the subsets of these full lists. We introduce the detailed configurations below. 

19 

Table B: All the rule clauses and their descriptions in the expert mode of the SPF tasks. The clauses in other modes are the subsets of this full list. 

|Rule|Description|
|---|---|
|`Stop(X):- Not(IsAmbulance(X)),`|If `X`is not an ambulance and not old, and `X`is at an intersection, and `Y`is in an|
|`Not(IsOld(X)), IsAtInter(X),`|intersection, then`X`should stop.|
|`IsInInter(Y).`||
|`Stop(X):- Not(IsAmbulance(X)),`|If `X`is not an ambulance and not old, and `X`is at an intersection, and `Y`is at an|
|`Not(IsOld(X)), IsAtInter(X),`|intersection, and`Y`has higher priority than`X`, then`X`should stop.|
|`IsAtInter(Y), HigherPri(Y, X).`||
|`Stop(X):- Not(IsAmbulance(X)),`|If `X`is not an ambulance and not old, and `X`is in an intersection, and `Y`is in an|
|`Not(IsOld(X)), IsInInter(X),`|intersection, and`Y`is an ambulance, then`X`should stop.|
|`IsInInter(Y), IsAmbulance(Y).`||
|`Stop(X):- Not(IsAmbulance(X)),`|If `X`is not an ambulance and not police, and `X`is a car, and `X`is not in or at an|
|`Not(IsPolice(X)), IsCar(X),`|intersection, and`Y`is left of and close to`X`, and`Y`is police, then`X`should stop.|
|`Not(IsInInter(X)),`||
|`Not(IsAtInter(X)),`||
|`LeftOf(Y, X), IsClose(Y, X),`||
|`IsPolice(Y).`||
|`Stop(X):- IsBus(X),`|If`X`is a bus, and`X`is not in or at an intersection, and`Y`is right of and next to`X`,|
|`Not(IsInInter(X)),`|and`Y`is a pedestrian, then`X`should stop.|
|`Not(IsAtInter(X)), RightOf(Y, X),`||
|`NextTo(Y, X), IsPedestrian(Y).`||
|`Stop(X):- IsAmbulance(X),`|If`X`is an ambulance, and`Y`is right of`X`, and`Y`is old, then`X`should stop.|
|`RightOf(Y, X), IsOld(Y).`||
|`Stop(X):- Not(IsAmbulance(X)),`|If `X`is not an ambulance and not old, and `X`is close to colliding with `Y`, then `X`|
|`Not(IsOld(X)), CollidingClose(X,`|should stop.|
|`Y).`||
|`Slow(X):- Not(Stop(X)), IsTiro(X),`|If`X`should not stop, and`X`is a tiro, and`Y`is a pedestrian, and`X`is close to`Y`, then|
|`IsPedestrian(Y), IsClose(X, Y).`|`X`should slow.|
|`Slow(X):- Not(Stop(X)), IsTiro(X),`|If `X` should not stop, and `X` is a tiro, and `X` is in an intersection, and `Y` is at an|
|`IsInInter(X), IsAtInter(Y).`|intersection, then`X`should slow.|
|`Slow(X):- Not(Stop(X)),`|If`X`should not stop, and`X`is police, and`Y`is young, and`Z`is young, and`Y`is next|
|`IsPolice(X),`|to`Z`, then`X`should slow.|
|`IsYoung(Y), IsYoung(Z),`||
|`NextTo(Y, Z).`||
|`Fast(X):- Not(Stop(X)),`|If `X` should not stop, and `X` should not slow, and `X` is reckless, and `Y` is at an|
|`Not(Slow(X)),`|intersection, then`X`should go fast.|
|`IsReckless(X), IsAtInter(Y).`||
|`Fast(X):- Not(Stop(X)),`|If`X`should not stop, and`X`should not slow, and`X`is a bus, then`X`should go fast.|
|`Not(Slow(X)),`||
|`IsBus(X).`||
|`Fast(X):- Not(Stop(X)),`|If`X`should not stop, and`X`should not slow, and`X`is police, and`Y`is reckless, then|
|`Not(Slow(X)),`|`X`should go fast.|
|`IsPolice(X), IsReckless(Y).`||



## **B.1 Safe Path Following** 

**Modes and Dataset:** _N_ ˜1 = 5. If the number of observed agents are fewer thanAcross all modes, we fix the (maximum) number of FOV agents into 5, 5, zero-padding (closed-world assumption) _i.e._ , is utilized, otherwise, we neglect the extra agents. The predicates involved in each mode are displayed˜ in Table A. Easy mode includes 7 unary and 2 binary predicates, resulting in an[�] _[K] i N_ 1 _[r][i]_ = 85 dimensional grounding vector. Rules involve only _spatial_ concepts and constrain the `Stop` ˜ action. Medium mode features 8 unary predicates and 4 binary predicates, creating a[�] _[K] i N_ 1 _[r][i]_ = 140 dimensional grounding vector. The medium rule sets is extended from the easy mode and incorporate bothpredicates and _spatial_ and 6 binary predicates, yielding a _semantic_ concepts, constraining the[�] _[K] i N_ ˜1 _[r][i]_ `Stop`[= 205] action.[ dimensional grounding vector.] Hard mode contains 11 unary[Rules] cover all _spatial_ and _semantic_ concepts and constrain the `Stop` action. The expert mode constrains all four actions with the most complex rule sets. We provide standard training/validation/test agent configurations and validation/test episodes for all the modes. The training agents cover all the necessary concepts in the rules, while validation and test agents are different and more complex, see our code library for the detailed agent configuration. For each mode, we collect 40 validation episodes and 100 test episodes using corresponding agent distribution, making sure the episodes cover 

20 

all the _concepts_ and actions. When training the BC branch algorithms, we collected 100 trajectories from the oracle as the demonstration. 

**Reward:** During test, the rule violation weight _w_[r] is set to _−_ 10 for easy, medium, and hard mode across all the _M_ clauses. For expert mode, we set this constant punishment to _−_ 5. In terms of step-wise action costs _ϕ_ ( **a** _[t]_ 1[)][, the easy, medium, and hard modes are configured as follows:] `[Slow]`[ :] _−_ 2 _,_ `Normal` : 0 _,_ `Fast` : _−_ 2 _,_ `Stop` : _−_ 5. In the expert mode, the costs are `Slow` : _−_ 2 _,_ `Normal` : _−_ 1 _,_ `Fast` : _−_ 2 _,_ `Stop` : _−_ 3. Note that the action costs will be normalized by the length of the global path. Overtime punishment is set to _−_ 3 for all the modes. During training, we find that different methods requires different reward functions to work effectively. Therefore, we first fix the action costs and have tried our best to tune the rule violation and overtime punishment for each method. For fairness, NLM-DQN and DQN used the same training reward. For more details about the reward, please see our code library. 

## **B.2 Visual Action Predication** 

**Modes:** As shown in Table A, the predicates in the two modes involve the full list. As for the clauses, easy mode only constrains `Stop` and `Slow` actions, setting `Normal` as the default action. Hard mode constrains all the three actions with `Normal` set as the default actions. Note that the rule clauses in hard model is a superset of that in the easy mode. 

**Datasets:** In the random agent setting, we randomly generated 100 and 20 cities with different agent compositions for training and validation, respectively. For each city, we run the simulation for 100 steps and only used the data after 10 steps. In the fixed agent setting, we first pre-define different training/validation/test agent compositions. Then, we randomly initialize the cities for 100 times. For each initialization, we run the simulation for 100 steps and only used the data after 10 steps. This process results in training sets with 8 _._ 9 _k_ images (with more than 89 _k_ entity samples). The models trained with different setting are tested in the same fixed agent setting test sets. See our code library and dataset for detailed agent compositions. 

## **C Full Procedure of LogiCity Simulation** 

We provide more details for the simulator here. 

**Static Urban Semantics:** There are a total of _B_ = 8 static semantics of the urban map, namely “Walking Street”, “Traffic Street”, “Crossing”, “House”, “Office”, “Garage”, “Store”, “Gas Station”. They are (currently) only used during initialization. Specifically, different types of agents will sample start and goal locations around different static semantics. Pedestrians will move from “House”, “Office”, and “Store” to “House”, “Office”, and “Store”, while Cars are navigating between “Garage”, “Gas Station”, and “Store” to “Garage”, “Gas Station”, and “Store”. Besides, the agents use different search algorithms based on these semantics to construct their global paths. Specifically, the pedestrians leverage A _[∗]_ search on the “movable region” of the map **M** s, which is defined as the union of Walking Streets and Crossings. In contrast, for cars, since they should move only on the right side of the road in real-world, we first construct the “one-way” road map of the Traffic Streets, which is a directed cyclic graph. Then, we connect the start and goal points to this road map and add them to the graph nodes. Finally, Dijkstra search is employed to construct the shortest path from start node to the goal node, which is the global path for a car. 

**Rendering Details:** As introduced above, there exist 8 static semantics. As shown in Table A, LogiCity also involves 9 semantic _concepts_ of the agents. Therefore, for each of the 17 semantics, we ask GPT4 [1] to generate 40 diverse descriptions. Then we leverage Stable-Diffusion XL model [29] to generate _∼_ 2000 diverse icons from these descriptions. Finally, we employed a human expert to select 50 _∼_ 200 icons for each semantic. For mode details, please see our code library. 

In addition, we also present the full procedure of the scene simulation by LogiCity in Algorithm 1. 

21 

## **D State Space Comparison** 

In the SPF task, we by default provide the predicate groundings as the observation of the agents, which is _abstract_ and could be _lossy_ [12]. Thus, we have also tried to provide exact states to the agents in this section. Specifically, we annotate each pixel of the FOV map with the agent semantics and convert the pixels into 2D semantic point clouds. Since these point clouds contain all the information needed for an optimal policy, it serves as the “Exact State" for the ego agent. The results comparison of using abstract (Abs.) and exact (Exa.) states is shown in Table C, where we find using “Exact State” could 

Table C: Comparison of results from different state space in the LogiCity SPF task. By default, the observation state is the _groundings_ of the predicates, which is abstract (Abs.) and lossy. We also tried to provide exact state (Exa.) as observation, which is the semantic point cloud in the ego agent FOV. 

|Mode|Easy|Hard|
|---|---|---|
||||
|Method<br>Obs<br>DQN<br>Abs.<br>Exa.<br>MB-Shooting Abs.<br>Exa.|TSR<br>DSR<br>Score<br>**0.35**<br>0.89<br>4.8<br>0.12<br>0.329<br>2.1<br>**0.24**<br>0.44<br>2.55<br>0.23<br>0.264<br>2.12|TSR DSR Score<br>**0.09**<br>0.12<br>0.63<br>0.01<br>0.56<br>2.69<br>**0.16**<br>0.17<br>1.26<br>0.02<br>0.24<br>1.32|



be much harder for the agents to learn the abstractions in _easy_ mode. In _hard_ mode, the agents can easily converge to overly careful policies and fail to complete the task in time. One possible future solution for “Exact State” is to combine bi-level planning [12] with reinforcement learning. 

## **E Quantitative Perceptual Noise** 

Compared with structured knowledge graphs [20, 91, 92], the VAP task of LogiCity introduces diverse RGB images, which require models to conduct abstract reasoning with high-level perceptual noise. We quantitatively display the perception accuracy of different _concepts_ from the NLM model [7] in Table D. Even with supervision, the averaged recall rate for the concepts is not satisfiable (Note that the errors will actually accumulate, which will be much more worse than the 55% averaged result). Compared with binary predicates, unary predicates need operation on the RGB image, which is thus harder. We also observe that the results are highly-imbalanced across concepts. For example, pedestrains and cars are easy to recognize, but a police/tiro car is extremely hard to be distinguished from normal ones. In terms of binary predicates, `CollidingClose` is the hardest to learn, since it needs to consider all the locations, sizes, and directions of the two entities, while the others only involves positions or priorities. One potential solution to the perceptual noise is borrowing off-the-shelf foundation models [93, 94] for the grounding task. 

## **F Visualizations** 

**SPF:** Visualizations of the SPF task episodes are displayed in Figure C. Compared with the training city shown on the left, test cities have different agent compositions. For example, training city only has 2 old man while test cities has 4 such entities, featuring compositional generalization challenge. Compared with pure neural networks, NeSy method (NLM-DQN) can better generalize to unseen compositions. For example, in Episode 92, Step 84, the ego agent sees two `pedestrians InIntersection` with an `Ambulance AtIntersection` , which is an unseen composition during training. DQN fails here, outputting `Normal` action while NLM-DQN succeeds with the correct `Stop` decision. SPF task also features realistic multi-agent interaction. As shown in Episode 93, Step 125, since the two algorithms made different decisions in previous steps, the city will be very different as the other agents are largely affected by the ego actions. 

**VAP:** Visualizations of the VAP task examples are shown in Figure D. Compared with GNN [37], the NeSy method NLM [7] can better understand the _abstractions_ of LogiCity. For example, `Reckless` cars drives `Normally` when it is `InIntersection` , while other cars should drive `Slow` . We find that GNN [37] shows limitation in understanding such _concept_ and rules, making wrong predictions. 

22 

## **Algorithm 1:** LogiCity Simulation 

**Input:** Concepts _P_ , Rules _C_ , Agents _A_ , Static urban map **M** s, Generative models for rendering Generate **M** s with dimensions ( _W, H, B_ ) and sample collision-free coordinates for agents Compute global paths for each agent using search-based planner 

**for** _each time step t_ **do** 

Update **M** _[t]_ with current agent locations and paths 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_LogiCity_Advancing_Neuro-Symbolic_AI_with_Abstract_Urban_Simulation_images/nips_2024_LogiCity_Advancing_Neuro-Symbolic_AI_with_Abstract_Urban_Simulation.pdf-0023-04.png)


**----- Start of picture text -----**<br>
for  each agent An do<br>Obtain  M [t] n [and] [ A][t] n [using cropping function] ▷ Local FOV observation<br>Compute  g n [t] [by applying] [ F][i] [to] [ M] [t] n [and] [ A][t] n ▷ Grounding predicates<br>Compute action predicates  a [t] n [using SMT solver with] [ g] n [t] [and] [ C] ▷ Rule inference<br>Move agent based on  a [t] n<br>Update  M [t] [+1] ▷ Update semantic map<br>if  agent reaches goal  then<br>Set new goal location and compute new path ▷ Re-sample goal and re-plan path<br>**----- End of picture text -----**<br>


Generate concept descriptions using GPT-4 and generate icons using diffusion model Compose icons into **M** _[t]_ to create RGB image **I** _[t] ▷_ `Rendering` 

**Output:** RGB images of urban grid maps **I** _[t]_ 

Table D: Quantitative results for concept recognition in the VAP task of LogiCity. We report the recall rate of NLM [7] model for each predicate (with threshold 0 _._ 5 on the predicted probability). The results are obtained from hard mode with random training agents. 

|Arity|Unary|
|---|---|
|Predicates<br>Num. Samples<br>Recall@0.5|`IsPed. IsCar IsAmbu. IsBus IsPolice IsTiro IsReckl. IsOld IsYoung IsAtInter IsInInter`<br>Avg.<br>10680<br>14240<br>1780<br>1780<br>3560<br>1780<br>3560<br>3560<br>5340<br>7490<br>3627<br>0.774<br>0.981<br>0.251<br>0.4<br>0.073<br>0.024<br>0.158<br>0.328<br>0.563<br>0.278<br>0.332<br>0.553|
||Arity<br>Binary<br>Predicates<br>`IsClose`<br>`HigherPri`<br>`CollidingClose`<br>`LeftOf`<br>`RightOf`<br>`NextTo`<br>Avg.<br>Num. Samples<br>23660<br>28902<br>500<br>33046<br>28064<br>15495<br>Recall@0.5<br>0.783<br>1<br>0.05<br>0.857<br>0.921<br>0.874<br>0.887|



23 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_LogiCity_Advancing_Neuro-Symbolic_AI_with_Abstract_Urban_Simulation_images/nips_2024_LogiCity_Advancing_Neuro-Symbolic_AI_with_Abstract_Urban_Simulation.pdf-0024-00.png)


**----- Start of picture text -----**<br>
In-Context Demonstrations and Example Questions from LogiCity<br>You are an expert in First-Order-Logic (FOL) Rule induction, the following question-answers are FOL reasoning<br>�→ examples. Here are 5 demonstrations:<br>Question: "In the scene you see a total of 12 entities, they are named as follows: Entity_0, Entity_1,<br>�→ Entity_2, Entity_3, Entity_4, Entity_5, Entity_6, Entity_7, Entity_8, Entity_9, Entity_10, Entity_11.<br>�→ There exist the following predicates as their attributes and relations: IsPedestrian (arity: 1), IsCar<br>�→ (arity: 1), IsAmbulance (arity: 1), IsBus (arity: 1), IsPolice (arity: 1), IsTiro (arity: 1), IsReckless<br>�→ (arity: 1), IsOld (arity: 1), IsYoung (arity: 1), IsAtInter (arity: 1), IsInInter (arity: 1), IsClose<br>�→ (arity: 2), HigherPri (arity: 2), CollidingClose (arity: 2), LeftOf (arity: 2), RightOf (arity: 2), NextTo<br>�→ (arity: 2), Sees (arity: 2). The truth value of these predicates grounded to the entities are as follows<br>�→ (Only the ones that are True are provided, assume the rest are False): IsPedestrian(Entity_1),<br>�→ IsPedestrian(Entity_2), IsPedestrian(Entity_3), IsPedestrian(Entity_4), IsPedestrian(Entity_5),<br>�→ IsCar(Entity_0), IsCar(Entity_6), IsCar(Entity_7), IsCar(Entity_8), IsCar(Entity_9), IsCar(Entity_10),<br>�→ IsCar(Entity_11), IsAmbulance(Entity_0), IsAmbulance(Entity_11), IsPolice(Entity_6), IsPolice(Entity_10),<br>�→ IsTiro(Entity_9), IsReckless(Entity_8), IsOld(Entity_3), IsOld(Entity_5), IsYoung(Entity_1),<br>�→ IsYoung(Entity_2), IsAtInter(Entity_5), IsAtInter(Entity_8), IsAtInter(Entity_11), IsInInter(Entity_0),<br>�→ IsInInter(Entity_6), IsInInter(Entity_10), IsClose(Entity_1, Entity_3), IsClose(Entity_3, Entity_1),<br>�→ IsClose(Entity_3, Entity_7), IsClose(Entity_4, Entity_10), IsClose(Entity_7, Entity_3), IsClose(Entity_7,<br>�→ Entity_10), IsClose(Entity_10, Entity_4), IsClose(Entity_10, Entity_7), IsClose(Entity_10, Entity_11),<br>�→ IsClose(Entity_11, Entity_10), ..., Sees(Entity_1, Entity_3), Sees(Entity_1, Entity_7), Sees(Entity_1,<br>�→ Entity_10), Sees(Entity_3, Entity_1), Sees(Entity_3, Entity_7), Sees(Entity_3, Entity_10), Sees(Entity_4,<br>�→ Entity_10), Sees(Entity_4, Entity_11), Sees(Entity_5, Entity_8), Sees(Entity_7, Entity_10), Sees(Entity_7,<br>�→ Entity_11), Sees(Entity_8, Entity_5), Sees(Entity_10, Entity_1), Sees(Entity_10, Entity_3), Sees(Entity_10,<br>�→ Entity_7), Sees(Entity_11, Entity_4), Sees(Entity_11, Entity_7), Sees(Entity_11, Entity_10). What is the<br>�→ next action of entity Entity_9?"<br>Option: (A) Slow (B) Normal (C) Fast (D) Stop<br>Answer: B<br>... (4 more demos not displayed)<br>Now try your best to first identify the FOL rules from the examples above and then answer the following<br>�→ question. Your answer should strictly end with the format of single letter: 'Answer: _.'<br>Question: "In the scene you see a total of 14 entities, they are named as follows: Entity_0, Entity_1,<br>�→ Entity_2, Entity_3, Entity_4, Entity_5, Entity_6, Entity_7, Entity_8, Entity_9, Entity_10, Entity_11,<br>�→ Entity_12, Entity_13. There exist the following predicates as their attributes and relations: IsPedestrian<br>�→ (arity: 1), IsCar (arity: 1), IsAmbulance (arity: 1), IsBus (arity: 1), IsPolice (arity: 1), IsTiro (arity:<br>�→ 1), IsReckless (arity: 1), IsOld (arity: 1), IsYoung (arity: 1), IsAtInter (arity: 1), IsInInter (arity:<br>�→ 1), IsClose (arity: 2), HigherPri (arity: 2), CollidingClose (arity: 2), LeftOf (arity: 2), RightOf (arity:<br>�→ 2), NextTo (arity: 2), Sees (arity: 2). The truth value of these predicates grounded to the entities are<br>�→ as follows (Only the ones that are True are provided, assume the rest are False): IsPedestrian(Entity_1),<br>�→ IsPedestrian(Entity_2), IsPedestrian(Entity_3), IsPedestrian(Entity_4), IsPedestrian(Entity_5),<br>�→ IsPedestrian(Entity_6), IsCar(Entity_0), IsCar(Entity_7), IsCar(Entity_8), IsCar(Entity_9),<br>�→ IsCar(Entity_10), IsCar(Entity_11), IsCar(Entity_12), IsCar(Entity_13), IsAmbulance(Entity_12),<br>�→ IsBus(Entity_10), IsPolice(Entity_9), IsPolice(Entity_11), IsTiro(Entity_8), IsReckless(Entity_0),<br>�→ IsReckless(Entity_7), IsOld(Entity_3), IsOld(Entity_5), IsYoung(Entity_1), IsYoung(Entity_2),<br>�→ IsYoung(Entity_4), IsAtInter(Entity_8), IsAtInter(Entity_13), IsInInter(Entity_6), IsInInter(Entity_11),<br>�→ IsClose(Entity_0, Entity_5), IsClose(Entity_1, Entity_3), IsClose(Entity_2, Entity_8), IsClose(Entity_3,<br>�→ Entity_1), IsClose(Entity_5, Entity_0), IsClose(Entity_5, Entity_6), IsClose(Entity_5, Entity_10),<br>�→ IsClose(Entity_6, Entity_5), IsClose(Entity_6, Entity_8), IsClose(Entity_6, Entity_13), IsClose(Entity_8,<br>�→ Entity_2), IsClose(Entity_8, Entity_6), IsClose(Entity_8, Entity_10), IsClose(Entity_10, Entity_5),<br>�→ IsClose(Entity_10, Entity_8), IsClose(Entity_10, Entity_13), IsClose(Entity_13, Entity_6),<br>�→ IsClose(Entity_13, Entity_10), HigherPri(Entity_0, Entity_8), HigherPri(Entity_0, Entity_10),<br>�→ HigherPri(Entity_0, Entity_12), HigherPri(Entity_0, Entity_13), HigherPri(Entity_2, Entity_0),<br>�→ HigherPri(Entity_2, Entity_8), HigherPri(Entity_2, Entity_10), HigherPri(Entity_2, Entity_13),<br>�→ HigherPri(Entity_5, Entity_0), HigherPri(Entity_5, Entity_8), HigherPri(Entity_5, Entity_10),<br>�→ HigherPri(Entity_5, Entity_12), HigherPri(Entity_5, Entity_13), HigherPri(Entity_6, Entity_0),<br>�→ HigherPri(Entity_6, Entity_8), HigherPri(Entity_6, Entity_10), HigherPri(Entity_6, Entity_12),<br>�→ HigherPri(Entity_6, Entity_13), HigherPri(Entity_7, Entity_9), HigherPri(Entity_8, Entity_10),<br>�→ HigherPri(Entity_8, Entity_12), HigherPri(Entity_8, Entity_13), HigherPri(Entity_10, Entity_12),<br>�→ HigherPri(Entity_10, Entity_13), HigherPri(Entity_12, Entity_13), CollidingClose(Entity_0, Entity_10),<br>�→ CollidingClose(Entity_7, Entity_9), CollidingClose(Entity_12, Entity_13), LeftOf(Entity_0, Entity_2),<br>�→ LeftOf(Entity_0, Entity_6), LeftOf(Entity_0, Entity_8), LeftOf(Entity_2, Entity_5), LeftOf(Entity_2,<br>�→ Entity_13), LeftOf(Entity_3, Entity_1), LeftOf(Entity_5, Entity_8), LeftOf(Entity_6, Entity_2),<br>�→ LeftOf(Entity_6, Entity_8), LeftOf(Entity_8, Entity_5), LeftOf(Entity_8, Entity_12), LeftOf(Entity_8,<br>�→ Entity_13), LeftOf(Entity_10, Entity_2), LeftOf(Entity_10, Entity_6), LeftOf(Entity_10, Entity_8),<br>�→ LeftOf(Entity_12, Entity_0), ..., Sees(Entity_0, Entity_5), Sees(Entity_0, Entity_6), Sees(Entity_0,<br>�→ Entity_8), Sees(Entity_0, Entity_10), Sees(Entity_0, Entity_12), Sees(Entity_0, Entity_13), Sees(Entity_1,<br>�→ Entity_3), Sees(Entity_3, Entity_1), Sees(Entity_5, Entity_4), Sees(Entity_6, Entity_0), Sees(Entity_6,<br>�→ Entity_10), Sees(Entity_6, Entity_12), Sees(Entity_6, Entity_13), Sees(Entity_7, Entity_9), Sees(Entity_8,<br>�→ Entity_0), Sees(Entity_8, Entity_2), Sees(Entity_8, Entity_6), Sees(Entity_8, Entity_10), Sees(Entity_8,<br>�→ Entity_13), Sees(Entity_10, Entity_5), Sees(Entity_10, Entity_6), Sees(Entity_10, Entity_8),<br>�→ Sees(Entity_10, Entity_12), Sees(Entity_10, Entity_13), Sees(Entity_12, Entity_0), Sees(Entity_12,<br>�→ Entity_6), Sees(Entity_12, Entity_10), Sees(Entity_12, Entity_13), Sees(Entity_13, Entity_0),<br>�→ Sees(Entity_13, Entity_2), Sees(Entity_13, Entity_5), Sees(Entity_13, Entity_6), Sees(Entity_13, Entity_8),<br>�→ Sees(Entity_13, Entity_10). What is the next action of entity Entity_11?"<br>Option: (A) Slow (B) Normal (C) Fast (D) Stop<br>**----- End of picture text -----**<br>


Figure B: In-context demos and questions for LLM testing. Note that the demos cover all the options and the test questions have different entity compositions. We only display one demo and part of the groundings due to the space limit. 

24 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_LogiCity_Advancing_Neuro-Symbolic_AI_with_Abstract_Urban_Simulation_images/nips_2024_LogiCity_Advancing_Neuro-Symbolic_AI_with_Abstract_Urban_Simulation.pdf-0025-00.png)


**----- Start of picture text -----**<br>
Success Epis. 92 Failed Success Epis. 93 Failed<br>Normal Normal Normal Normal Normal<br>#001 #001 #001 #001 #001<br>Stop Stop Stop Normal Stop<br>#047 #039 #039 #012 #012<br>Stop Stop Normal Stop Normal<br>Didn’t Stop<br>Died<br>#082 #084 #084 #040 #125<br>Normal Normal Normal Normal Normal<br>Didn’t Stop Didn’t Stop<br>Died Died<br>#087 #150 #084 #125 #175<br>Training Epis. Test Epis. Goal Start Global Path NLM-DQN DQN<br>**----- End of picture text -----**<br>


Figure C: Qualitative comparison between NLM-DQN [7, 34] and DQN [34] agents in the hard mode of SPF task. We display the training episode on the left, which has different agent sets from the test, featuring compositional generalization challenge. Compared with the pure neural network, NeSy method [7, 34] is better at abstract reasoning. In Episode 92, with unseen compositions of concepts, the DQN agent fails while NLM-DQN succeeds with the correct `Stop` decision. Note that in SPF, different ego decisions could significantly affect the city evolution (See Episode 93, Step 125). 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_LogiCity_Advancing_Neuro-Symbolic_AI_with_Abstract_Urban_Simulation_images/nips_2024_LogiCity_Advancing_Neuro-Symbolic_AI_with_Abstract_Urban_Simulation.pdf-0025-02.png)


**----- Start of picture text -----**<br>
Image Grounded Clause Predictions Image Grounded Clause Predictions<br>Normal(X):- GT: Normal Fast(X):- GT: Fast<br>IsReckless(X), Not(Stop(X))<br>IsInInter(X) NLM: Normal IsReckless(X), NLM: Fast<br>IsAtInter(Y),<br>GNN: Slow Sees(X,Y) GNN: Normal<br>GT: Stop GT: Slow<br>Stop(X):- Slow(X):-<br>IsAtInter(X), Not(IsReckless(X)),<br>IsInInter(Y), NLM: Stop Not(IsAmbulance(X)), NLM: Slow<br>Sees(X,Y) IsInInter(X)<br>GNN: Normal GNN: Normal<br>GT: Normal Slow(X):- GT: Slow<br>Normal(X):- IsPolice(X),<br>Not(Stop(IsReckless(X))X), NLM: Normal IsYoung(IsYoung(YZ),), NLM: Slow<br>IsAtInter(Not(Sees(XY,),Y)) GNN: Fast NextTo(Sees(X,YY,)Z) GNN: Normal<br>**----- End of picture text -----**<br>


Figure D: Qualitative comparison between NLM [7] and GNN [37] in the hard mode of VAP task. We display the _grounded_ clauses, where the involved entities are marked with boxes in corresponding colors. Correct predictions are shown in gree, while the wrong one is in red. 

25 

