IEEE ROBOTICS AND AUTOMATION LETTERS. PREPRINT VERSION. ACCEPTED JANUARY, 2023 

1 

## **Learning to Search in Task and Motion Planning with Streams** 

Mohamed Khodeir _[∗†]_ Ben Agro _[∗†]_ Florian Shkurti _[†]_ 

_**Abstract**_ **—Task and motion planning problems in robotics combine symbolic planning over discrete task variables with motion optimization over continuous state and action variables. Recent works such as PDDLStream [1] have focused on optimistic planning with an incrementally growing set of objects until a feasible trajectory is found. However, this set is exhaustively expanded in a breadth-first manner, regardless of the logical and geometric structure of the problem at hand, which makes longhorizon reasoning with large numbers of objects prohibitively time-consuming. To address this issue, we propose a geometrically informed symbolic planner that expands the set of objects and facts in a best-first manner, prioritized by a Graph Neural Network that is learned from prior search computations. We evaluate our approach on a diverse set of problems and demonstrate an improved ability to plan in difficult scenarios. We also apply our algorithm on a 7DOF robotic arm in block-stacking manipulation tasks.** 

## I. INTRODUCTION 

Task and motion planning (TAMP) involves searching over both symbolic actions that determine a high-level task sequence and low-level motions that result in feasible trajectories [2]–[4]. The symbolic task planner operates on high-level abstractions of the environment, including object definitions, operations that can be applied to them, and pre- and postconditions that these operations have to satisfy. The motion planner requires a non-abstracted description of the environment, and is responsible for informing the symbolic planner about the kinematic and dynamic feasibility of a proposed coarse task plan, possibly leading to backtracking. This interplay between high- and low-level planning has a significant effect on the total runtime required to discover a solution. 

PDDL [5] is a widely used representation for symbolic planners, but requires a pre-discretized state space and does not allow for continuous variables. PDDLStream [1] extends PDDL by introducing _streams_ , conditional generators that model black-box sampling procedures. Streams enable continuous quantities, such as poses and configurations to be created dynamically, allowing planning without the need for prediscretization. Existing PDDLStream algorithms use streams to iteratively add more of these quantities to the planning problem until a feasible trajectory can be found, leveraging existing PDDL planners as a subroutine. However, this process can induce very large planning problems, especially in cases that involve long-horizon reasoning with many objects. 

Manuscript received: August 23, 2022; Revised: November 26, 2022; Accepted: January 16, 2023. 

This paper was recommended for publication by Editor Hanna Kurniawati upon evaluation of the Associate Editor and Reviewers’ comments. _∗_ Authors contributed equally. 

> _†_ Robot Vision and Learning Laboratory, University of Toronto Robotics Institute. _{_ m.khodeir, ben.agro _}_ @mail.utoronto.ca, florian@cs.toronto.edu Digital Object Identifier (DOI): 10.1109/LRA.2023.3242201 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_Learning_Geometric_Reasoning_Networks_For_Robot_Task_And_Motion_Planning_images/iclr_2025_Learning_Geometric_Reasoning_Networks_For_Robot_Task_And_Motion_Planning.pdf-0001-12.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_Learning_Geometric_Reasoning_Networks_For_Robot_Task_And_Motion_Planning_images/iclr_2025_Learning_Geometric_Reasoning_Networks_For_Robot_Task_And_Motion_Planning.pdf-0001-13.png)


Fig. 1: Our planner solves the six-block stacking problem with a 7DOF robot arm[1] . Existing TAMP methods struggle to find a solution for more than four blocks (see Table II). By learning from easier intermediate planning problems, we show improved performance and run time for larger scenes. 

To address this shortcoming, we present a learning-based method to guide a best-first search in the space of streams. Our main contributions are (a) a Graph Neural Network (GNN) architecture that scores the relevance of streams, and (b) a queue-based algorithm that prioritizes their addition to the planning problem. The model is trained on prior planning experience from solving similar problem instances, drawn from a training distribution of environments. We compare our algorithm (INFORMED) across several simulated environments against the best performing PDDLStream algorithm, as well as a prior work which uses a learned GNN to predict object importance. Our experiments show that INFORMED can outperform baselines on previously unseen problems, finding feasible plans faster, including on problems of larger size, where the baseline fails to find solution. 

## II. RELATED WORK 

**Integrated task and motion planning.** There is a vast literature on the problem of integrating the geometric reasoning required by motion planning with the symbolic reasoning that is necessary for planning to achieve abstract goals; see [2] for a detailed taxonomy. Our work builds on the PDDLStream [1] formalism, which we introduce in more detail in the background section. Several algorithms for PDDLStream problems have since been proposed, including [6] which uses MonteCarlo Tree Search to efficiently explore the space of plans in search of a low cost solution. PDDLStream has also been used to facilitate belief-space planning in partially observed environments [7]. 

> 1 Code and video can be found on the project webpage: https://rvl.cs. toronto.edu/learning-based-tamp/ 

2 

IEEE ROBOTICS AND AUTOMATION LETTERS. PREPRINT VERSION. ACCEPTED JANUARY, 2023 

There is a long history of prior research, including [8], [9], which combine symbolic planners with complete geometric planner. The need for selecting the right hierarchical abstractions for symbolic planning and favoring feasibility and realtime results over optimality was emphasized in [3]. Logic Geometric Programming combined symbolic planning and trajectory optimization [4], [10], even for dynamic physical motions involving tool use, while [11] integrated sampling procedures with SAT solvers as symbolic planners. 

**Learning for TAMP** . Motivated by the success of learning in the context of robotics, recent work has sought to combine the ability of TAMP systems to plan for novel temporally extended goals with learning methods. There are several avenues of research under this umbrella: methods which learn continuous action samplers for capabilities that may be difficult to engineer (e.g. pouring) [13], [14], those which learn the symbolic representations with which to plan [15]–[17], those that integrate perception learning and scene understanding into TAMP [18], [19], and those which attempt to learn search guidance from experience [20]–[24]. In contrast to these works, we seek to learn a heuristic to guide the construction of an optimistic planning problem with a minimal set of facts. These represent contraints on the continuous parameters of a potential solution, and so our work can be seen as a generalization of [24], which similarly learns to select helpful constraints from a discrete set. We leverage an off-the-shelf domain-independent search sub-routine, thus our work can be synergistically combined with methods like [20]– [22] that learn a domain-specific heuristic. 

**Planning with many objects.** The motivation of our work shares similarities with [25] and [26], which attempt to speed up task planning by learning models which exclude irrelevant objects or actions (respectively) from the initial problem representation. Our work can be seen as an extension of these methods. 

## III. BACKGROUND 

A PDDLStream [1] problem ( _P, A, S, O, I, G_ ) is given by a set of predicates _P_ , actions _A_ , streams _S_ , initial objects _O_ , an initial state _I_ , and a goal state _G_ . Similarly to PDDL, a _predicate_ , _p_ is a boolean function. An instance of a predicate _p_ (¯ _x_ ) applied on a tuple of objects _x_ ¯ = _⟨x_ 1 _, . . . , xn⟩_ is called a _fact_ . PDDL _objects_ are references to entities in the planning problem. These can refer to anything from the identities of physical objects, to continuous quantities or even complex data structures such as those representing trajectories. A _state_ is a set of facts. _I_ is an example of a state. An action _a_ is a tuple of three components: parameters _X_[¯] , a set of preconditions pre( _a_ ), and a set of effects eff( _a_ ). An action instance _a_ (¯ _x_ ), whose parameters _X_[¯] have been assigned to particular objects _x_ ¯, is only applicable/available in a given state if it satisfies the set of facts in the action’s preconditions pre( _a_ (¯ _x_ )). The set of facts eff( _a_ (¯ _x_ )) specify how a state will change after _a_ (¯ _x_ ) is applied. A solution to the planning problem is a sequence of valid action instances _π_ = [ _a_ 1(¯ _x_ 1) _, . . . aN_ (¯ _xN_ )] that transform _I_ into _G_ . A helpful concept is the _preimage_ of a plan; the set of 

facts that must hold in order for the entire plan to be applicable in a given initial state. This is defined as follows: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_Learning_Geometric_Reasoning_Networks_For_Robot_Task_And_Motion_Planning_images/iclr_2025_Learning_Geometric_Reasoning_Networks_For_Robot_Task_And_Motion_Planning.pdf-0002-08.png)


**Example.** To describe the environment depicted in Fig. 2, we might define predicates such as on-block(?b1 ?b2) and at(?b ? _XW B_ ). We could then define an action that stacks one block onto another as stack(?q ?b ?lb ? _XW L_ ) with preconditions that include clear(?lb) and at(?lb ? _XW L_ ), requiring that the lower block is clear and at the given world pose _XW L_ . One effect of this action might be on-block(?b ?lb), but there may be other side-effects such as a new configuration of the robot arm. To describe a particular stacking problem in this environment, we simply need to specify objects _O_ = _{b_ 1 _, b_ 0 _, b_ 2 _, a, s_ 0 _, s_ 1 _, q_ 0 _, . . . }_ , initial state _I_ = _{_ on-block( _b_ 1 _b_ 0) _,_ at( _b_ 0 _XW b_ 0) _, . . . }_ , and goal state _G_ = _{_ on-surface( _b_ 0 _s_ 1) _,_ on-surface( _b_ 1 _s_ 1) _}_ . **Streams.** The set of streams, _S_ , distinguishes a PDDLStream problem from traditional PDDL. Streams are conditional generators which yield objects that satisfy specific constraints conditioned on their inputs. Formally, a stream, _s_ , consists of input parameters _s.input_ , a set of domain predicates _s.domain_ , output parameters _s.output_ = _o_ ¯, and a set of certified predicates _s.certified_ . _s.domain_ is the set of predicates that must evaluate to true for an input tuple _s.input_ = _x_ ¯ to be valid. This ensures the correct types of objects (e.g., configurations, poses etc.) are provided to the generators. _s.certified_ are predicates on _s.input_ and _s.output_ that assert facts that _⟨x,_ ¯ ¯ _o⟩_ always satisfy. 

**Example.** In our block stacking environment (Fig. 2), we can define a stream find-place(?b, ?s) _→_ ? _XW B_ which takes as input a block ?b and a surface ?s, and produces a world pose ? _XW B_ , at which ?b would be on ?s. The domain facts of this stream would restrict the types of its input objects (i.e. find-place.domain = _{_ block(?b), surface(?s) _}_ ), and the set of certified facts might contain the single fact block-support(?block ?surface ? _XW B_ ). Given such a stream, a planner could dynamically request placement poses for any object/surface pair. 

Streams can be applied recursively, to generate a potentially infinite set of objects and their associated facts, starting from those in _I_ and _O_ , respectively. A crucial aspect of PDDLStream problems is that there usually does not exist a plan whose _preimage_ is a subset of _I_ . Instead, solvers of PDDLStream problems must use the streams in _S_ to recursively expand _I_ to _Iopt_ such that there exists a plan _π_ whose _preimage_ is a subset of _Iopt_ . This is reminiscent of theorem proving problems that start from a set of axioms and use rules of inference to expand the set of known theorems. 

We use the term stream _instantiation_ to refer to the act of creating a new instance of a stream on a specific object tuple. We distinguish this from _evaluation_ or _sampling_ of a stream instance, which refers to the act of invoking the associated black-box sampler on the given inputs. When a stream instance is instantiated, its output objects (and associated certified facts) are said to be _optimistic_ - they are symbolic referents that may or may not have possible real values. In contrast, evaluation of the stream instance produces _grounded_ objects (i.e. with a 

KHODEIR ET AL.: LEARNING TO SEARCH IN TAMP WITH STREAMS 

3 

concrete value). A stream instance can only be evaluated if its inputs are themselves grounded. 

Existing PDDLStream algorithms expand the set of stream _results_ (objects and associated certified facts) in a breadthfirst manner according to their so-called ‘level’ . On a given iteration, all streams are instantiated up to a certain level _l_ , and a PDDL planner is invoked on the resulting set. If no plan is found, the next iteration starts with _l ← l_ + 1. The ‘level’ of a stream is defined recursively as: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_Learning_Geometric_Reasoning_Networks_For_Robot_Task_And_Motion_Planning_images/iclr_2025_Learning_Geometric_Reasoning_Networks_For_Robot_Task_And_Motion_Planning.pdf-0003-04.png)


Here, **count** refers to the number of times that a stream instance has been evaluated. A single stream instance may need to be evaluated many times to produce a value which results in a feasible trajectory. For example, the planner may consider many grasps for the same object before finding one that is collision-free. 

## IV. OUR APPROACH 

Our approach uses a learned model to guide the expansion of optimistic facts in a PDDLStream problem. We first present a PDDLStream algorithm which uses a priority queue to order the instantiation of streams. Second, we formulate the problem of assigning a priority to optimistic stream instantiations as a learning problem. Finally, we propose a model architecture which makes use of the structure of streams to recursively build up representations of optimistic objects, and assign priorities to them simultaneously. 

## _A. INFORMED Algorithm_ 

We propose the INFORMED algorithm, which uses a priority queue, _Q_ , to order the addition of optimistic results to the planning problem based on a _relevance score_ given by a black-box model, _Mθ_ . The algorithm is composed of four distinct phases: Expansion, Planning, Sampling, and Feedback. We describe each of these steps below, and provide the full pseudocode in 1. 

_Expansion_ . In each iteration of the algorithm, the stream result with the highest score is popped off of the queue and if the result is optimistic, its certified facts are added to _Iopt_ . A sub-routine EXPAND creates all the child stream instances which use these newly added objects. These are all scored and pushed onto the queue for later expansion (line 11 in Alg. 1). 

_Planning_ . In the planning phase, we invoke a PDDL planner on the current set of optimistic and grounded facts (line 17 in Alg. 1). If an optimistic plan _πopt_ is found, it is returned along with the _stream plan ψ_ , which is the sequence of stream instances that need to be evaluated to ground _πopt_ . This subroutine, which is a light wrapper around the underlying FastDownward planner, is the same as that presented in [1]. A key issue for our algorithm is how to decide when the set of included optimistic facts is sufficient for planning to succeed. This SHOULD-PLAN classifier may itself be a good candidate for learning from experience. However, in this work, we simply choose to plan every time _K_ = 100 optimistic and grounded facts have been added to the problem, so in 

our implementation SHOULD-PLAN returns true every _K_ iterations. 

_Sampling_ . If a plan has just been found or if we have found a plan in a previous iteration without yet grounding all of its parameters, then we have the option to devote time to sampling. This means running the underlying generators for each of the streams in the plan _ψ_ . This is facilitated by the PROCESS-STREAMS sub-routine (line 22), of which several variants are described in [1]. In this work, we employ the version used by the ADAPTIVE algorithm in [1]. 

_Feedback_ . If, in the course of sampling, the algorithm has managed to produce groundings for all the parameters in an optimistic plan, then our algorithm halts and the plan is returned (line 24). Otherwise, we would like to update our planning problem to account for information gleaned from sampling. By evaluating the streams that support an optimistic plan, PROCESS-STREAMS will produce grounded objects that we already suspect to be relevant. These new values are all immediately added to _Iground_ so that they can be used by the planner in the next expansion and planning phases (line 26). 

## **Algorithm 1** INFORMED( _A, S, I, G, Mθ_ , PROCESS-STREAMS): 

|**Alg**|**orithm 1** INFORMED(_A, S, I, G, Mθ_, PROCESS-STREAMS):|
|---|---|
|1|_Iground_ =_I_.copy(), _Iopt_ =_{}_, _Q_= []|
|2 <br>3<br>4<br>5|**for** _s_(¯_x_)_∈_INSTANTIATE(_S, I_) **do**<br>r = _s_(¯_x_).next<br>optimistic()<br>score = _Mθ_(_r, Iground ∪Iopt, G_)<br>_Q_.push(_⟨score, r⟩_)|
|6 <br>7<br>8|**while** True **do**<br>**if** _Q_.len()_>_0 **then**<br>_⟨score, r⟩_=_Q.pop_()|
|9|**if** _r_.is<br>optimistic() **then**|
|10<br>11|_Iopt_.add(_r.certified_)<br>**for** _s_(¯_x_)_∈_EXPAND(_r, Iground ∪Iopt, S_) **do**|
|12<br>13<br>14|r = _s_(¯_x_).next<br>optimistic()<br>score = _Mθ_(_r, Iground ∪Iopt, G_)<br>_Q_.push(_⟨score, r⟩_)|
|15|_πopt_ = None|
|16<br>17<br>18<br>19<br>20|**if** SHOULD-PLAN() **then**<br>_πopt, ψ_ =OPTIMISTIC-SOLVE(_A, Iground ∪Iopt, G_)<br>_t_=SHOULD-SAMPLE()<br>**if** _t ≤_0 and _Q_.len() = 0 **then**<br>**return** None|
|21<br>22<br>23|**if** _t >_0 **then**<br>_π, new,processed_=<br>PROCESS-STREAMS(_Iground ∪Iopt, ψ, πopt_;_t_)<br>**if** _π _=None **then**|
|24|**return** _π_|
|25<br>26<br>27|**for** _r ∈new_ **do**<br>_Iground.add_(_r.certified_)<br>score = _Mθ_(_r, Iground ∪Iopt, G_)|
|28|_Q_.push(_⟨score, r⟩_)|
|29<br>30|**for** _r ∈processed_ **do**<br>_Iopt_.remove(_r.certified_)|



**Algorithm 2** INSTANTIATE ( _S, I_ ): 

1 _{s_ (¯ _x_ ) _|∀s ∈S, ∀p ∈ s.domain, |x_ ¯ _|_ = _|s.input|, p_ (¯ _x_ ) _∈I}_ 

**Algorithm 3** EXPAND (r, _S, I_ ): 

1 _{sj_ (¯ _x_ ) _|∀sj_ (¯ _x_ ) _∈_ INSTANTIATE( _S, r.certified ∪I_ ) _, r.output ⊂ x_ ¯ _}_ 

## _B. Completeness of the INFORMED algorithm_ 

The algorithm described above inherits many of the same theoretical guarantees as those described in [1]. In particular, the proof of semi-completeness for optimistic algorithms (i.e. 

4 

IEEE ROBOTICS AND AUTOMATION LETTERS. PREPRINT VERSION. ACCEPTED JANUARY, 2023 

theorem 3) can be extended to the INFORMED algorithm under some assumptions about the relevance scores. We omit the proof from the paper for brevity, but include it in supplementary materials on the project webpage[1] . 

## _C. Relevance of Streams as Binary Classification_ 

The question of whether a stream instance is relevant to the planning problem can be boiled down to whether any of the objects that it produces are relevant. In this work we define an object to be relevant to a PDDLStream problem if: (1) it is part of the preimage of any optimal plan, or (2) it is used as input to a stream which produces a relevant object. We note, however, that applying this definition directly would be impractical. It is prohibitively expensive to compute optimal plans for problems of interest, so we apply this definition using satisficing plans found by existing PDDLStream algorithms. 

_Relevance of a PDDL object._ In a TAMP setting, the objects in the preimage of a given plan will correspond to real valued quantities, such as the 7DOF configuration of a robotic arm. However, these values are rarely ever unique; that a specific grasp configuration of an object was used in a plan should imply that a different grasp of the same object is also important to consider. Therefore, when deciding on the relevance of an object, we concern ourselves not with the actual numeric value of the object, but with the set of constraints that act on it. To determine whether an object is relevant, we simply need to check whether there exists a counterpart in a given preimage which has the same set of constraints (i.e. the same ancestor streams). 

_Training data._ This approach allows us to cast the relevance of stream instances as a supervised binary classification problem. Given a set of training problems _D_ = _{⟨Oi, I i, Gi⟩}[N] i_ =1 defined by objects _Oi_ , initial state _I i_ , and goal _Gi_ we use an existing PDDLStream solver to find a plan _πi_ , and record all the stream instances _sij_ produced along the way. For each _sij_ , we use PREIMAGE( _πi_ ) to assign a binary label _yij_ as described above. 

## _D. Modelling Relevance of Streams_ 

Given the _I_ , _G_ , the current set of optimistic stream results _Iopt_ and grounded stream results _Iground_ , we want to assign a _relevance score y ∈_ [0 _,_ 1] to a stream result _s_ (¯ _x_ ) _→ o_ ¯. 

_MLP representations of streams._ Our model architecture is motivated by the observation that a stream represents constraint(s) between its inputs _x_ ¯ and outputs _o_ ¯, and that for each domain there are only a handful of explicitly defined streams. We model each stream _s_ with a simple multi-layer perceptron (MLP) _Ms_ , which has fixed input and output sizes proportional to _|s._ input _|_ and _|s._ output _|_ . Given fixed dimensional ( _N_ = 64 in our experiments) dense embeddings _hxi_ of the input objects _x_ each¯, _Ms_ ofproducesthe outputan objectsembeddingin _o_ ¯ _h_ , _o_ as _j_ ofwelltheassamea scalardimensionscore to _y_ representing the relevance of those outputs to the planning problem. For an instance of stream _s_ with input objects _x_ ¯, the computation of _Ms_ is as follows (note _⊙_ denotes concatenation): 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_Learning_Geometric_Reasoning_Networks_For_Robot_Task_And_Motion_Planning_images/iclr_2025_Learning_Geometric_Reasoning_Networks_For_Robot_Task_And_Motion_Planning.pdf-0004-10.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_Learning_Geometric_Reasoning_Networks_For_Robot_Task_And_Motion_Planning_images/iclr_2025_Learning_Geometric_Reasoning_Networks_For_Robot_Task_And_Motion_Planning.pdf-0004-11.png)


Fig. 2: Top row: A block stacking task is solved and a set of labeled stream results is extracted from the plan. Bottom left: the Problem Graph representation of the problem. Bottom right: GNN-Stream is trained to predict relevance of optimistic stream results. 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_Learning_Geometric_Reasoning_Networks_For_Robot_Task_And_Motion_Planning_images/iclr_2025_Learning_Geometric_Reasoning_Networks_For_Robot_Task_And_Motion_Planning.pdf-0004-13.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_Learning_Geometric_Reasoning_Networks_For_Robot_Task_And_Motion_Planning_images/iclr_2025_Learning_Geometric_Reasoning_Networks_For_Robot_Task_And_Motion_Planning.pdf-0004-14.png)


As depicted in the bottom right of Fig. 2, these _Ms_ can be recursively combined in exactly the same way as streams, producing relevance scores and dense vector representations for each object their associated streams create. A similar architecture has been used for natural language processing [27] where it is referred to as a tree-recursive recurrent neural network. **Example.** Figure 2 depicts the flow of computation in our model for a simple example. In order to predict a score for the grasp pose _g_ 0 _[∗]_[of block] _[ b]_[2][, the GNN embedding of] _[ b]_[2][is] passed into the MLP for the grasp stream, _Mgrasp_ . The score is then computed as in equations 2, 3. Similarly, the vector representation of _g_ 0 _[∗]_[is][computed][as][in][equation][4][to][produce] _hg_ 0 _[∗]_[,][which][can][then][be][used][in][subsequent][streams.] 

_GNN representations of objects and facts._ To obtain the representations for the initial objects in the planning problem, we employ _Graph Neural Networks_ owing to their ability to represent objects based on their relations to one another. We provide details about the architecture and hyperparameters of the GNN in supplementary materials. The input to the GNN, which we call the _Problem Graph_ (PG), is a graph with nodes representing the PDDL objects and edges representing the facts that relate those objects to one another in both _I_ and _G_ (see bottom left of Fig. 2). This relational representation is similar to that used in [25], and allows our system to generalize to different numbers and identities of objects. The node feature vectors consist of the 3D position of the object if it exists, and padding otherwise. The edge feature vectors have a onehot encoding of their corresponding predicate, and a flag if the fact appears in _I_ (1) or _G_ (0). This graph representation is extensible and can include other geometric (or semantic) information about the scene. Since our experiments use only cubes of fixed sizes, we do not need to encode the geometry of each object in the scene. However, in principle, data such as point-clouds or depth-images can be used as node features to support planning in more diverse scenarios, but this is out of scope for this work. 

5 

KHODEIR ET AL.: LEARNING TO SEARCH IN TAMP WITH STREAMS 

Fig. 3: Typical example of the evaluation tasks in each environment. 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_Learning_Geometric_Reasoning_Networks_For_Robot_Task_And_Motion_Planning_images/iclr_2025_Learning_Geometric_Reasoning_Networks_For_Robot_Task_And_Motion_Planning.pdf-0005-03.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_Learning_Geometric_Reasoning_Networks_For_Robot_Task_And_Motion_Planning_images/iclr_2025_Learning_Geometric_Reasoning_Networks_For_Robot_Task_And_Motion_Planning.pdf-0005-04.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_Learning_Geometric_Reasoning_Networks_For_Robot_Task_And_Motion_Planning_images/iclr_2025_Learning_Geometric_Reasoning_Networks_For_Robot_Task_And_Motion_Planning.pdf-0005-05.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_Learning_Geometric_Reasoning_Networks_For_Robot_Task_And_Motion_Planning_images/iclr_2025_Learning_Geometric_Reasoning_Networks_For_Robot_Task_And_Motion_Planning.pdf-0005-06.png)


**----- Start of picture text -----**<br>
(a) Clutter (b) Non-monotonic (c) Sorting<br>(d) Stacking (e) Distractors (f) Kitchen<br>**----- End of picture text -----**<br>


The end-to-end model architecture _Mθ_ is depicted in the bottom row of Fig. 2. A given planning problem is used to produce a Problem Graph. A GNN produces an embedding of each object according to its relations in both the goal and the initial state. These representations can then be used as input to stream MLPs, which are combined recursively to obtain representations and scores for optimistic objects produced by their associated streams. This entire computation is differentiable, so _Mθ_ is trained end-to-end to minimize a weighted cross-entropy, giving a higher penalty to false negatives. _L_ ( _D, θ_ ) =[�] _[N] i_ =1 � _Mj_ =1 _i[L][CE]_[(] _[y][ij][,][ M][θ]_[(] _[I][i][,][ G][i][, s][ij]_[(¯] _[x][ij]_[)))] 

## V. EXPERIMENTAL RESULTS 

## _A. Environment Setup._ 

Evaluation of INFORMED was conducted across six problem types involving a 7DoF robot arm, loosely based on those outlined in [28]. All experiments are conducted using 2 cores of an Intel Broadwell processor with an 8GB memory limit. For each problem type, we randomly generate 100 instances which are used to train a separate model and 100 held-out test instances which we use for evaluation. We additionally report results for training a single ‘multi-task’ model in table III. _The training instances are drawn from distributions with few objects/goals so that the baseline is able to solve the majority of them within the timeout. We sample the test instances from a more challenging distribution to evaluate the model’s ability to generalize to problems with more objects and goal conjuncts than those seen during training._ Initial placement of objects and their goal states are randomly generated in each problem, so that even problems of the same type with the same number of objects may have different solutions. All methods use a 90 second planning timeout, so we report the proportion of problems solved within the timeout, as well as average planning times for solved instances. 

**Problem Types:** Blocks world problems are divided into five categories which share a domain definition, but present different challenges to the planner, as shown in Fig. 3. There 

are two types of blocks in these problems (not distinguished logically): blocks (shorter) and blockers (taller) (Figs. 3a and 3e). The robot arm is surrounded by four tables. 

Stacking: Blocks are arranged randomly in each scene, and the goal is to assemble them into specific towers. These problems involve long plans, low likelihood of obstructions. Training problems involved up to 4 objects and test problems up to 7. 

Sorting: Colorful are arranged arranged randomly, and the goal is to move them to the table with the corresponding color. Blockers may need to be moved if they obstruct a plan, but must be returned to their original tables. Sorting problems have high branching factor, long plans, high likelihood of obstructions. Training problems involved up to 14 objects and test problems up to 20. 

Clutter: Blocks need to be stacked or rearranged in the presence of twice as many blockers which may obstruct motions, grasps or placements. Initial positions are sampled using ordered Poisson-Disc Sampling so that blocks are placed close together. These problems have a high branching factor, short plans, high likelihood of obstructions. Training problems involved up to 15 objects, test problems up to 18. 

Non Monotonic: Blockers and blocks come in pairs (see Fig. 3b). Blocks have to be moved to a different table, whereas blockers must be returned to their exact starting positions. It is always infeasible to grasp the block without first moving its corresponding blocker, so the planner must temporarily undo subgoals (hence the name). These problems have low branching factor, long plans, guaranteed obstructions. Training problems involved up to 6 objects and test problems up to 8. 

Distractors: Blocks need to be stacked or rearranged in the presence of ”distractor” blockers which are placed on their own table. Unlike the blockers in other problem types, distractors do not appear in the goal, and do not need to be interacted with. This problem type tests the planner’s ability to ignore irrelevant objects. Test problems contain 2-3 blocks and up to 50 distractors. There are no training problems - instead, the model trained on the Clutter task is used here. 

Kitchen: In these problems, a robot arm must complete tasks in a kitchen involving different types of items. Items can be cleaned if they are in the sink, and clean items can be cooked if they are on a burner (Fig. 3f). The plans in this domain can be long, but the branching factor is lower than blocks world due to the absence of stacking actions. 

**Methods:** We compare our approach to ADAPTIVE, the best-performing PDDLStream algorithm presented in [1]. Our approach differs from ADAPTIVE in two key respects. First, while ADAPTIVE must expand all streams up to a given depth before planning, INFORMED will invoke the planner after a constant number of additional streams have been expanded. The second key difference is that INFORMED uses a given relevance model to order the expansion of streams. In order to isolate the effect of the model from that of planning frequency, we evaluate the following variants of our main method. Performance differences between these ablations and INFORMED can only be attributed to the learned model. _•_ INFORMED (Level): Stream expansion is ordered by level (as defined in equation 1). Ties are broken arbitrarily. This 

6 

IEEE ROBOTICS AND AUTOMATION LETTERS. PREPRINT VERSION. ACCEPTED JANUARY, 2023 

ablation does not use the training data. This ordering is the same as that used by ADAPTIVE, so only differs in the frequency with which the PDDL planner is invoked. 

_•_ INFORMED (Stats): Stream expansion is ordered based on the proportion of times the associated stream was marked relevant in the training data. This ablation assigns the same score to all instances of a given stream. 

_•_ INFORMED (PG): An ablation of our model architecture which makes use of the problem graph network, but not the recursively combined stream MLPS. The problem graph network assigns an importance score to each object in the initial scene. During planning, a candidate stream instance’s priority is the average importance score of its ancestors, divided by its depth. 

_•_ INFORMED (Ours): Our proposed approach, as described in section IV-D. 

Finally, we also compare to a prior work [25] (PLOI) which uses learned GNNs to predict the importance of objects in the planning problem. PLOI treats the PDDLStream solver as a black box, and simply excludes all objects below a threshold score from the call to the solver. In order to maintain completeness, an outer-loop iteratively reduces the importance threshold until a plan is found, or all objects pass the threshold. 

**Refined vs. unrefined mode:** ADAPTIVE and INFORMED admit two modes of operation with respect to stream expansions. The first, _refined_ expansion, operates exactly as described in [1] and section III - optimistic objects from different stream instances are assigned unique identifiers which are used in further stream expansions. In contrast, the second mode of operation, _unrefined_ , groups together all objects produced by a specific stream under a single shared identifier. This mode has the effect of bounding the combinatorial explosion from stream expansions, since it effectively limits the number of new objects. This comes at the expense of additional work, which must be done to disambiguate shared identifiers in the event that a plan is found which uses them. In our testing, we observed that this trade-off often results in faster planning times for ADAPTIVE. The relevance model we have described cannot score these unrefined streams, because of the inherent ambiguity in their meaning. In order to evaluate the effect of the model, we use both algorithms in _refined_ mode. However, for the sake of completeness, we also report ADAPTIVE’s performance in unrefined mode. 

## _B. Results and Discussion_ 

TABLE I: Planning performance on held-out _test_ problems. All runs are conducted with 90s timeouts. We report the number of solved instances (out of 100), and the average times over all **solved** problems of a given type. 

||**ADAPTIVE**<br>**(Unrefned)**|**ADAPTIVE**<br>**(Refned)**|**INFORMED**<br>**(Level)**|**INFORMED**<br>**(Stats)**|**INFORMED**<br>**(PG)**|**INFORMED**<br>**(Ours)**|**PLOI**<br>[25]|
|---|---|---|---|---|---|---|---|
|||||||||
|**Domain**|**Solved Time**|**Solved Time**|**Solved Time**|**Solved Time**|**Solved Time**|**Solved Time**|**Solved Time**|
|||||||||
|**Clutter**<br>**Non Mono**<br>**Sorting**<br>**Stacking**<br>**Distractors**<br>**Kitchen**|**54**<br>30.99<br>27<br>31.45<br>66<br>16.95<br>47<br>16.95<br>77<br>47.82<br>91<br>19.90|22<br>18.20<br>28<br>36.96<br>50<br>25.11<br>20<br>5.52<br>1<br>42.74<br>**95**<br>27.24|5<br>37.55<br>15<br>73.99<br>12<br>15.46<br>21<br>12.38<br>1<br>23.17<br>83<br>24.51|25<br>11.67<br>0<br>-<br>62<br>17.65<br>16<br>4.70<br>0<br>-<br>84<br>28.52|32<br>15.22<br>1<br>35.68<br>51<br>14.60<br>21<br>9.04<br>66<br>23.21<br>86<br>24.51|**54**<br>12.15<br>**58**<br>33.88<br>**77**<br>21.07<br>**54**<br>7.94<br>**100**<br>9.65<br>94<br>23.93|49<br>16.11<br>25<br>29.71<br>68<br>20.47<br>47<br>9.20<br>**100**<br>6.62<br>91<br>28.76|




![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_Learning_Geometric_Reasoning_Networks_For_Robot_Task_And_Motion_Planning_images/iclr_2025_Learning_Geometric_Reasoning_Networks_For_Robot_Task_And_Motion_Planning.pdf-0006-11.png)


**----- Start of picture text -----**<br>
Adaptive (Unrefined) PLOI Informed (Ours)<br>Clutter Non Monotonic Sorting<br>100<br>80<br>60<br>40<br>20<br>0<br>Stacking Distractors Kitchen<br>0 20 40 60 80 0 20 40 60 80 0 20 40 60 80<br>Run Time (s)<br>%<br>Solved<br>**----- End of picture text -----**<br>


Fig. 4: A comparison of ADAPTIVE, PLOI and INFORMED on percentage of test-distribution problems solved as a function of run time. Test-distribution includes problems with more objects and goals than seen in training. 

_Ablations._ Table I shows the success rates and planning times for the ablations described above. On all problem types, the learned model is able to significantly outperform ablations, showing that it is able to prioritize the inclusion of taskrelevant objects enabling the planner to solve more problems within the timeout. The Level ablation generally performs poorly compared to ADAPTIVE, despite using the same underlying ordering of stream expansions. This is primarily due to more frequent planner invocations using up more of the allotted time. We found that the Stats ablation is able to significantly outperform ADAPTIVE in Sorting problems, when using refined expansion. Upon closer inspection, we found that the training examples for Sorting contain no stack or unstack actions, which are indeed unnecessary in this task. By assigning a score of zero to objects which would be required for those actions, the Stats ablation is able to effectively reduce the branching factor of its search, and thus solve more problems compared to ADAPTIVE. The PG ablation performed significantly better than others on Distractors problems, where many initial objects were wholly irrelevant. However, it does not make use of the precise structure of relations that relate the stream’s outputs to its ancestor objects, and is thus unable to match the performance of our proposed approach in any of the problem types. This demonstrates the effectiveness of the recursive stream MLP’s at making use of the structure underlying stream instances. 

_Comparison to PLOI._ PLOI does quite well on distractors, but performs on par with ADAPTIVE on all other problem types. This is because apart from the Distractors problems, the majority of objects in the scene are needed to find a plan. Therefore, PLOI does not provide a significant advantage in planning time by excluding low-scored objects, and may incur overhead by having to invoke the PDDLStream planner several times until all the necessary objects have been included. In contrast, our approach is able to retain the benefit of PLOI when irrelevant objects are present (as evidenced by the 

KHODEIR ET AL.: LEARNING TO SEARCH IN TAMP WITH STREAMS 

7 

Distractors experiments) as well as provide speed-up in other problem types by making finer-grain predictions about the relevance of stream instances. 

_Generalization._ All of the problems we use for evaluaAdaptive (Unrefined) tion are from a held-out test Informed (Ours) set. In addition, we generate these problems from test dis80 tributions with more objects, 6040 and more goal conjuncts, so the 20 problem instances are unseen 0 at training time. In general, the 10 20 30 40 50 increased performance shown Num Distractors in Table I demonstrates genFig. 5: Average planning eralization to scenes with diftime as a function of the ferent numbers and placements number of distractors. of objects as well as different goals. The Distractors task is 

especially notable since this kind of problem is not part of any training set, and contained many more objects than any training problem. This demonstrates the ability of our architecture to generalize to problems of different size and composition. 

The Distractors experiment also demonstrates a key advantage of our approach when compared to ADAPTIVE. Figure 5 shows the planning time of both methods as a function of the number of distractor objects. Even though the problems are no more geometrically or logically challenging, the simple inclusion of additional objects increases the planning time rapidly. In contrast, our method is able to maintain a comparatively constant planning time. 

As shown in Table II, ADAPTIVE is unable to TABLE II: Percentage solve any stacking problems of Stacking problems with stacks of 5 or 6 blocks. solved across varying stack INFORMED’s performance heights. also deteriorates with stack height, however it is able to **Stack Height (Unrefined)ADAPTIVE INFORMED(Ours)** solve the majority of these **2** 100.00 100.00 problems despite the training **3** 100.00 100.00 set having only problems **45** 93.330.00 100.0072.22 with up to 4 blocks. We **6** 0.00 57.14 found this improvement to 

be primarily due to a reduction in time spent solving the induced planning problems compared to ADAPTIVE. Note that no problems with a stack higher than 4 were seen during training, therefore our approach shows out-of-distribution generalization to problems with stacks composed of more objects. 

_Comparison of Planning Time Breakdown._ We compared the breakdown of run time between INFORMED and ADAPTIVE in order to shed light on the differences in how the allotted time is spent. In figure 6 we report these average breakdowns for stacking and clutter problems. We observed that the time spent solving induced planning problems is drastically decreased by the use of the learned model. In particular, search time for ADAPTIVE increases quite sharply as a function of the stack height in stacking problems, and 

the model is able to effectively address this issue. We also observed that the time spent performing model inference remains relatively low, though this also increases as a function of the number of objects. 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_Learning_Geometric_Reasoning_Networks_For_Robot_Task_And_Motion_Planning_images/iclr_2025_Learning_Geometric_Reasoning_Networks_For_Robot_Task_And_Motion_Planning.pdf-0007-10.png)


**----- Start of picture text -----**<br>
Informed Adaptive Sampling Search Inference<br>40 60<br>30<br>40<br>20<br>20<br>10<br>0 0<br>2 3 4 2 4 6<br>Stack Height Num Blocks<br>(s)<br>Runtime<br>**----- End of picture text -----**<br>


Fig. 6: A breakdown of the run times of ADAPTIVE and INFORMED into the constituent phases. We average the time in each phase over all problems which both algorithms solve successfully. 

_Effect of Additional Training Data._ To explore to what extent more training data would lead to improved performance, we evaluated models trained on increasing subsets of a larger training set. We plot solution rates and run time over training set sizes up to 400 problems in figure 7. We found that planning performance improves sharply, but improvements plateau after around 100 training problems. 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_Learning_Geometric_Reasoning_Networks_For_Robot_Task_And_Motion_Planning_images/iclr_2025_Learning_Geometric_Reasoning_Networks_For_Robot_Task_And_Motion_Planning.pdf-0007-13.png)


**----- Start of picture text -----**<br>
Stacking Clutter<br>60<br>50 70<br>40<br>60<br>30<br>50<br>20<br>0 100 200 300 400 0 100 200 300 400<br>Training Problems Training Problems<br>Time<br>Solved<br>Solve<br>Percent<br>Mean<br>**----- End of picture text -----**<br>


Fig. 7: Effect of additional training data on planning performance in Stacking and Clutter tasks. 

_Multi-task Model._ Since all the blocks world tasks share a single domain definition (i.e. predicates, actions and streams), we additionally experimented with training a single model on problem instances from all the tasks. We then evaluate this model on held-out instances from each problem type. We compare the performance of this model to that of the domainspecific models we evaluated in section V. The performance of the combined model is usually on-par with the task-specific models, except in Sorting and Stacking, where the domainspecific model significantly outperforms. 

_Real-robot experiment._ Real-robot experiment. In addition to the experiments described above, We deployed the solutions from our planner on a Franka-Emika Panda robot as shown in Fig. 1. For these experiments, the relevance model is trained on stacking tasks in simulation, but we execute and evaluate 

8 

IEEE ROBOTICS AND AUTOMATION LETTERS. PREPRINT VERSION. ACCEPTED JANUARY, 2023 

TABLE III: Solve percentage on held-out test problems for single-task vs multi-task models across Blocks World problem types. 

||**Single-Task**|**Multi-Task**|
|---|---|---|
||||
|**Domain**|**Solved % Time (s)**|**Solved % Time (s)**|
||||
|**Clutter**<br>**Non Monotonic**<br>**Sorting**<br>**Stacking**<br>**Distractors**|54<br>12.15<br>58<br>33.88<br>**77**<br>21.07<br>**54**<br>7.94<br>**100**<br>9.65|**56**<br>14.23<br>**65**<br>35.69<br>68<br>16.71<br>45<br>13.17<br>99<br>17.90|



the trajectories it finds directly on the robot arm in an openloop fashion. 

## VI. CONCLUSION 

In this paper, we proposed a method for learning to search in Task and Motion Planning (TAMP) based on past experience computing successful task and motion plans in similar environments. A key component of our method is a learned, problemconditioned ranking model that outputs the relevance of tasklevel subgoals and logical facts, which offers betters guidance in terms of task planning. We proposed the INFORMED algorithm which uses this model to order its expansion of streams with the goal of accelerating planning times for unseen problems. Our experiments demonstrate the effectiveness of this approach in the context of TAMP on a variety of problem types, where it leads to a significant speedup over existing methods, especially in problems with high branching factor, and including on a real robot manipulation setting. 

## REFERENCES 

- [1] C. R. Garrett, T. Lozano-P´erez, and L. P. Kaelbling, “Pddlstream: Integrating symbolic planners and blackbox samplers via optimistic adaptive planning,” 2020. 

- [2] C. R. Garrett, R. Chitnis, R. Holladay, B. Kim, T. Silver, L. P. Kaelbling, and T. Lozano-P´erez, “Integrated task and motion planning,” 2020. 

- [3] L. P. Kaelbling and T. Lozano-P´erez, “Hierarchical task and motion planning in the now,” in _2011 IEEE International Conference on Robotics and Automation_ , 2011, pp. 1470–1477. 

- [4] M. Toussaint, K. Allen, K. Smith, and J. Tenenbaum, “Differentiable physics and stable modes for tool-use and manipulation planning,” in _Proceedings of Robotics: Science and Systems_ , Pittsburgh, Pennsylvania, June 2018. 

- [5] M. Ghallab, A. Howe, C. Knoblock, D. McDermott, A. Ram, M. Veloso, D. Weld, and D. Wilkins, “Pddl–the planning domain definition language, technical report,” _Yale University, New Haven, CT_ , 1998. 

- [6] T. Ren, G. Chalvatzaki, and J. Peters, “Extended tree search for robot task and motion planning,” 2021. [Online]. Available: https://arxiv.org/abs/2103.05456 

- [7] C. R. Garrett, C. Paxton, T. Lozano-P´erez, L. P. Kaelbling, and D. Fox, “Online replanning in belief space for partially observable task and motion problems,” in _2020 IEEE International Conference on Robotics and Automation (ICRA)_ , 2020, pp. 5678–5684. 

- [8] S. Cambon, R. Alami, and F. Gravot, “A hybrid approach to intricate motion, manipulation and task planning,” _The International Journal of Robotics Research_ , vol. 28, no. 1, pp. 104–126, 2009. 

- [9] E. Plaku and G. D. Hager, “Sampling-based motion and symbolic action planning with geometric and differential constraints,” in _2010 IEEE International Conference on Robotics and Automation_ . IEEE, 2010, pp. 5002–5008. 

   - [11] N. T. Dantam, Z. K. Kingston, S. Chaudhuri, and L. E. Kavraki, “An incremental constraint-based framework for task and motion planning,” _The International Journal of Robotics Research_ , vol. 37, no. 10, pp. 1134–1151, 2018. 

   - [12] K. Hauser and J.-C. Latombe, “Integrating task and prm motion planning: Dealing with many infeasible motion planning queries,” in _ICAPS09 Workshop on Bridging the Gap between Task and Motion Planning_ . Citeseer, 2009. 

   - [13] Z. Wang, C. R. Garrett, L. P. Kaelbling, and T. Lozano-P´erez, “Learning compositional models of robot skills for task and motion planning,” _CoRR_ , vol. abs/2006.06444, 2020. [Online]. Available: https://arxiv.org/abs/2006.06444 

   - [14] B. Kim, L. Kaelbling, and T. Lozano-P´erez, “Guiding search in continuous state-action spaces by learning an action sampler from off-target search experience,” in _Proceedings of the AAAI Conference on Artificial Intelligence_ , vol. 32, no. 1, 2018. 

   - [15] T. Silver, R. Chitnis, J. Tenenbaum, L. P. Kaelbling, and T. LozanoP´erez, “Learning symbolic operators for task and motion planning,” _arXiv preprint arXiv:2103.00589_ , 2021. 

   - [16] J. Loula, K. Allen, T. Silver, and J. Tenenbaum, “Learning constraintbased planning models from demonstrations,” in _2020 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS)_ . IEEE, 2020, pp. 5410–5416. 

   - [17] M. Diehl, C. Paxton, and K. Ramirez-Amaro, “Automated generation of robotic planning domains from observations,” in _2021 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS)_ . IEEE, 2021, pp. 6732–6738. 

   - [18] K. Kase, C. Paxton, H. Mazhar, T. Ogata, and D. Fox, “Transferable task execution from pixels through deep planning domain learning,” _IEEE International Conference on Robotics and Automation (ICRA)_ , pp. 10 459–10 465, 2020. 

   - [19] Y. Zhu, J. Tremblay, S. Birchfield, and Y. Zhu, “Hierarchical planning for long-horizon manipulation with geometric and symbolic scene graphs,” _ArXiv_ , vol. abs/2012.07277, 2020. 

   - [20] D. Driess, J.-S. Ha, and M. Toussaint, “Deep visual reasoning: Learning to predict action sequences for task and motion planning from an initial scene image,” 2020. 

   - [21] D. Driess, O. Oguz, J.-S. Ha, and M. Toussaint, “Deep visual heuristics: Learning feasibility of mixed-integer programs for manipulation planning,” in _2020 IEEE International Conference on Robotics and Automation (ICRA)_ . IEEE, 2020, pp. 9563–9569. 

   - [22] B. Kim and L. Shimanuki, “Learning value functions with relational state representations for guiding task-and-motion planning,” in _Proceedings of the Conference on Robot Learning_ , ser. Proceedings of Machine Learning Research, L. P. Kaelbling, D. Kragic, and K. Sugiura, Eds., vol. 100. PMLR, 30 Oct–01 Nov 2020, pp. 955–968. [Online]. Available: https://proceedings.mlr.press/v100/kim20a.html 

   - [23] L. Xu, T. Ren, G. Chalvatzaki, and J. Peters, “Accelerating integrated task and motion planning with neural feasibility checking,” 2022. [Online]. Available: https://arxiv.org/abs/2203.10568 

   - [24] B. Kim, Z. Wang, L. P. Kaelbling, and T. Lozano-P´erez, “Learning to guide task and motion planning using score-space representation,” _The International Journal of Robotics Research_ , vol. 38, no. 7, pp. 793–812, 2019. 

   - [25] T. Silver, R. Chitnis, A. Curtis, J. Tenenbaum, T. Lozano-Perez, and L. P. Kaelbling, “Planning with learned object importance in large problem instances using graph neural networks,” 2020. 

   - [26] D. Gnad, A. Torralba, M. Dominguez, C. Areces, and F. Bustos, “Learning how to ground a plan -partial grounding in classical planning,” 01 2019. 

   - [27] R. Socher, C. C. Lin, C. Manning, and A. Y. Ng, “Parsing natural scenes and natural language with recursive neural networks,” in _Proceedings of the 28th international conference on machine learning (ICML-11)_ , 2011, pp. 129–136. 

   - [28] F. Lagriffoul, N. T. Dantam, C. Garrett, A. Akbari, S. Srivastava, and L. E. Kavraki, “Platform-independent benchmarks for task and motion planning,” _IEEE Robotics and Automation Letters_ , vol. 3, no. 4, pp. 3765–3772, 2018. 

   - [29] P. W. Battaglia, J. B. Hamrick, V. Bapst, A. Sanchez-Gonzalez, V. Zambaldi, M. Malinowski, A. Tacchetti, D. Raposo, A. Santoro, R. Faulkner, C. Gulcehre, F. Song, A. Ballard, J. Gilmer, G. Dahl, A. Vaswani, K. Allen, C. Nash, V. Langston, C. Dyer, N. Heess, D. Wierstra, P. Kohli, M. Botvinick, O. Vinyals, Y. Li, and R. Pascanu, “Relational inductive biases, deep learning, and graph networks,” 2018. 

- [10] M. Toussaint, “Logic-geometric programming: An optimization-based approach to combined task and motion planning,” pp. 1930–1936, 2015. [Online]. Available: http://ijcai.org/Abstract/15/274 

## APPENDIX 

## _A. Completeness of the INFORMED algorithm_ 

The algorithm described above inherits many of the same theoretical guarantees as those described in [1]. In particular, the proof of semi-completeness for optimistic algorithms (i.e. theorem 3) can be extended to the INFORMED algorithm under some assumptions about the relevance scores. 

At a high level, the proof for theorem 3 argues that if[˜] _l_ is the minimum level at which a solution exists, a finite number of iterations will elapse before the algorithm considers all stream instances with level _l ≤_[˜] _l_ , at which point the solution must be found. 

The main point of departure between these optimistic algorithms and INFORMED is the use of predicted relevance scores, rather than level (equation 1) for ordering the instantiation of streams. Therefore, to extend the argument to INFORMED, we simply need to show that there is a finite number of stream instances with a score greater than _y_ for any _y >_ 0. 

**Theorem A.1.** _The INFORMED algorithm is semi-complete under the following assumptions: (1) that the score that Mθ assigns to a stream is strictly less than the scores assigned to its parents, and (2) that the relevance scores assigned to successive evaluations of the same stream instance are monotonically decreasing._ 

_Proof._ Let _π_ be a feasible solution to a given PDDLStream problem. Then PREIMAGE( _π_ ) is the set of stream-certified facts which must be added the initial conditions _I_ so that SEARCH may find the correct optimistic solution. Let _y[∗]_ be the minimum score assigned to any of the elements of PREIMAGE( _π_ ). 

We can show that under conditions (1) and (2), a stream with score _≥ y[∗]_ has a maximum possible level, as defined in equation 1. To see this, let _ϵchild >_ 0 be the minimum difference of scores between a stream and its child, and let _ϵeval >_ 0 be the minimum difference between successive evaluations of the same stream. We can bound the maximum level of a stream with score _y[∗]_ as max _l_ ( _y[∗]_ ) _≤_ 1 + _⌈_[1] _ϵ[−] eval[y][∗][⌉]_[+] max _l_ ( _y[∗]_ + _ϵchild_ ) _≤ e_[1] _child[−][y][∗]_[(1 +][1] _e[−] eval[y][∗]_[)][.] 

INFORMED will instantiate the stream with the highest score at every iteration. By the argument above, every stream with score _≥ y[∗]_ must have level _≤_ max _l_ ( _y[∗]_ ). By theorem 3, we know that the number of stream instances with level _≤ maxl_ ( _y[∗]_ ) is finite. Therefore, we conclude that a finite number of iterations will elapse before the algorithm considers all stream instances with score _≥ y[∗]_ , at which point a solution must be found. 

In practice we enforce condition (2) by decaying the relevance score by a factor _γ <_ 1 after every evaluation. Condition (1) is enforced by defining the score of a stream _s_ as: 

## _B. Model Architecture and Hyperparameters_ 

In section IV-C, we have outlined the representation of planning problems as a relational graph, augmented with features that encode the 3D position of each object. This Problem Graph is the input to a GNN, composed of 3 graph network (GN) blocks [29]. Each GN block is itself composed of a node model _ϕv_ , and an edge model _ϕe_ which share the same architecture in table IV. 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_Learning_Geometric_Reasoning_Networks_For_Robot_Task_And_Motion_Planning_images/iclr_2025_Learning_Geometric_Reasoning_Networks_For_Robot_Task_And_Motion_Planning.pdf-0009-12.png)


TABLE IV: Hyperparameters of node and edge encoders _ϕ_[[1] [ _v/e[/]_[2] _[/]_ ][3]] which comprise each of 3 GN blocks. 

While this GNN produces embeddings of both its nodes and its edges as output, we only use the node embeddings, which serve as the inputs to the stream MLPs. Each stream MLP has a different number of inputs _Is_ and outputs _Os_ (depending on the domain definitions), however each of the input and output objects will be represented by a fixed size embedding. We detail the architecture of each MLP in V. 

||_Mencoder_<br>_s_|_Mscorer_<br>_s_|_Mdecoder_<br>_s_|
|---|---|---|---|
|input size<br>64_· Is_<br>64<br>64<br>number of layers<br>1<br>2<br>2<br>hidden size<br>64<br>64<br>64<br>hidden activation<br>-<br>LeakyRelu<br>LeakyRelu<br>output size<br>64<br>1<br>64_· Os_||||



TABLE V: Architecture and hyperparameters for each stream MLP. Note that the size of the input to the encoder, and output of the decoder depend on the stream’s domain definitions. 

## _C. Problem Distribution Details_ 

TABLE VI: To supplement the descriptions of environments given in section V, we provide details of object numbers and placement distributions for the train/test sets. 

||**Clutter**<br>**Non Monotonic**<br>**Sorting**<br>**Stacking**|**Number of Objects**<br>**Train / Test**<br>Blocks: 2-4 / 2-6<br>Blockers: 4-8 / 4-12<br>Blocks: 1-3 / 2-6<br>Blockers: 1-3 / 2-6<br>Blocks: 2-7 / 2-10<br>Blockers: 2-7 / 2-10<br>Blocks: 2-4 / 2-7<br>Blockers: 0 / 0|**Placement Distribution**<br>Poisson Disc<br>Uniform<br>Adjacent<br>Poisson Disc<br>Uniform|
|---|---|---|---|
||**Distractors**<br>**Kitchen**|Blocks: - / 2-3<br>Blockers: - / 10-50<br>Cabbage: 1-3 / 1-4<br>Radish: 0-3 / 0-3<br>Glass: 0-2 / 0-2|Uniform(red/blue tables)<br>Uniform(green/purple tables)<br>Uniform|
|||Goals: 1-7 / 1-9||




![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_Learning_Geometric_Reasoning_Networks_For_Robot_Task_And_Motion_Planning_images/iclr_2025_Learning_Geometric_Reasoning_Networks_For_Robot_Task_And_Motion_Planning.pdf-0009-20.png)


During training, we approximate the min with a weighted average of the parent scores, where the weights are computed as a softmax over the negative scores. 

