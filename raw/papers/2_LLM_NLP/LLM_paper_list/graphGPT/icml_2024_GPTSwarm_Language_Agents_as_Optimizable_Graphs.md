## **GPTSwarm: Language Agents as Optimizable Graphs** 

**Mingchen Zhuge**[* 1] **Wenyi Wang**[* 1] **Louis Kirsch**[2] **Francesco Faccio**[1 2] **Dmitrii Khizbullin**[1] **J¨urgen Schmidhuber**[1 2] 


![](raw/papers/2_LLM_NLP/LLM_paper_list/graphGPT/icml_2024_GPTSwarm_Language_Agents_as_Optimizable_Graphs_images/icml_2024_GPTSwarm_Language_Agents_as_Optimizable_Graphs.pdf-0001-03.png)


https://gptswarm.org 

## **Abstract** 

Various human-designed prompt engineering techniques have been proposed to improve problem solvers based on Large Language Models (LLMs), yielding many disparate code bases. We unify these approaches by describing LLM-based agents as computational graphs. The nodes implement functions to process multimodal data or query LLMs, and the edges describe the information flow between operations. Graphs can be recursively combined into larger composite graphs representing hierarchies of inter-agent collaboration (where edges connect operations of different agents). Our novel automatic graph optimizers (1) refine node-level LLM prompts (node optimization) and (2) improve agent orchestration by changing graph connectivity (edge optimization). Experiments demonstrate that our framework can be used to efficiently develop, integrate, and automatically improve various LLM agents. The code can be found here. 

## **1. Introduction** 

Interest in LLM-powered autonomous problem solvers or agents and their varied applications is continually rising (Wang et al., 2023; Xi et al., 2023). However, much work remains to be done to effectively incorporate these agents into a cohesive society and improve their structure automatically. 

Early approaches zero-shot-prompted LLMs or prompted 

> *Equal contribution 1AI Initiative, King Abdullah University of Science and Technology (KAUST), Thuwal, Saudi Arabia 2The Swiss AI Lab IDSIA, USI, SUPSI, Lugano, Switzerland. Correspondence to: Mingchen Zhuge _<_ mingchen.zhuge@kaust.edu.sa _>_ , Dmitrii Khizbullin _<_ dmitrii.khizbullin@kaust.edu.sa _>_ . 

_Proceedings of the 41[st] International Conference on Machine Learning_ , Vienna, Austria. PMLR 235, 2024. Copyright 2024 by the author(s). 

them with few-shot examples (Kojima et al., 2022; Brown et al., 2020). Recent methods prompt LLMs in a structured way, such as chain of thought (COT) (Wei et al., 2022), ReAct (Yao et al., 2022), tree of thought (TOT) (Yao et al., 2023), Reflexion (Shinn et al., 2023), and Graph of Thought (GOT) (Besta et al., 2023), to improve textbased reasoning. Single agent applications such as AutoGPT (Torantulino et al., 2023), BabyAGI (Nakajima, 2023), LangChain (Chase, 2022), and Llama-index (Liu, 2022) utilize LLMs for various functionalities, including tool usage, function calling, and embodied actions. In multi-agent frameworks (Zeng et al., 2022; Zhuge et al., 2023) several LLMs take on different roles (Li et al., 2023; Park et al., 2023; Qian et al., 2023; Wu et al., 2023) to communicate in natural language and collectively solve a given task. This approach often outperforms single agents, exploiting the specialization (Hong et al., 2023) of various LLM agents. Unfortunately, it also leads to increasingly different and disparate code bases that require a lot of human engineering to define prompting schemes and the workflow of agents. 

In a “society of mind” (SOM) (Minsky, 1988; Zhuge et al., 2023), higher-level intelligence emerges from the combination of simpler and modular cognitive components. Inspired by SOMs, we describe language agent systems through graph representations. Language agents querying LLMs and utilizing external tools are modeled as computational graphs where each node is dedicated to a specific function, while the edges define a topology of how inputs are processed across nodes, mirroring the prompting schemes in prior studies. A swarm is defined as a composite graph, where each subgraph represents a collaborative agent. This creates a deeper hierarchy of intelligence. Agent graphs combine basic LLM operations (Kennedy, 2006; Nepusz & Vicsek, 2013), and swarm graphs contain subgraphs representing agents. Approaches such as COT (Wei et al., 2022), TOT (Yao et al., 2023), and Self-Consistency (Wang et al., 2022) can be represented by our graphs. 

Our graph representation lends itself to optimization via prompting and evolutionary or reinforcement-learning techniques, so that agents can improve their communication (or 

1 

## **GPTSwarm: Language Agents as Optimizable Graphs** 


![](raw/papers/2_LLM_NLP/LLM_paper_list/graphGPT/icml_2024_GPTSwarm_Language_Agents_as_Optimizable_Graphs_images/icml_2024_GPTSwarm_Language_Agents_as_Optimizable_Graphs.pdf-0002-01.png)


**----- Start of picture text -----**<br>
Main Features<br>Query<br>TOT<br>1 Operation = Node<br>Search<br>2 Agent = Graph of Nodes<br>3 Swarm = Composite Graph<br>Answer 1 2 3 B<br>A Collaboration and Communication = A C<br>Information Flow between Graphs<br>B Orchestration =  TOT COT<br>Edge Connections in the Composite Graph<br>C Optimization =<br>Optimization of Nodes or Edges<br>GPTSwarm:  Language Agents as Optimizable Graphs<br>**----- End of picture text -----**<br>


_Figure 1._ **GPTSwarm is a framework that represents agents as graphs.** In this framework, each node represents an operation (e.g., LLM inference or tool use). An agent is a graph composed of these nodes. An edge between two agent graphs characterizes a communication channel; each agent collaborates with others through different channels. When connected, multiple agents form a composite graph with a certain orchestration topology. This graph representation lends itself to optimization of nodes and edges via prompting and evolutionary or reinforcement learning techniques. 

orchestration) patterns. The graph connectivity (adjacency matrices) between agents can self-improve online as a task is being solved or its solution is transferred to another task. 

As a proof-of-concept, we demonstrate how suboptimal agent organization can be overcome and how existing prompting techniques, such as Tree of Thought and Reflexion, can be automatically recombined by optimizing edges in a composite graph. Apart from edge optimization, our framework allows each node in the graph to self-improve by adapting its prompts based on previous input and task feedback. 

## **Our contributions can be summarized as follows:** 

**(1)** We unify language agent systems by describing them as optimizable computational graphs. 

**(2)** We introduce an open-source framework that allows for constructing arbitrary agent systems by recombining fundamental operations. We describe these engineeringlevel contributions in Appendix A. 

**(3)** We develop optimization methods for nodes and edges, enabling automatic improvements of agent prompts and inter-agent orchestration. 

**(4)** We validate our framework on various benchmarks including MMLU, Mini CrossWords, HumanEval, and GAIA, with an emphasis on the benefits of automatic graph optimization. 

## **2. GPTSwarm** 

## **2.1. Language Agents as Graphs** 

Taking inspiration from the society of mind (SOM) (Minsky, 1988; Zhuge et al., 2023), we propose to organize intelligence within a modular and hierarchical framework. This framework consists of nodes, graphs, and composite graphs, with each component playing a specific role. A node represents a fundamental operation that includes, but is not limited to, LLM inference, tool use, function calls, and various embodied actions. An agent, conceptualized as a graph, consists of multiple nodes that form a coherent functional entity. A swarm, or composite graph, represents a complex system of agents where the collective capabilities of this system may exceed those of individual agents. Finally, the edges within an agent define its execution topology, while the edges between agents establish collaboration and communication among them. 

## **2.2. Graph Definition** 

**Single language agent as a graph.** We model a language agent as a directed computational graph _G_ , defined by a tuple ( _N, E, F, o_ ), where _N_ is a set of computational nodes, _E ⊂ N ×N_ is a set of directed edges, _F_ = _{fn}n∈N_ is a set of computational routines and _o ∈ N_ is an output node. The set of predecessors of node _n_ is denoted by pre( _n_ ). In this paper, we focus on directed acyclic graphs (DAGs). Given an input _x_ , a graph _G_ iteratively executes its nodes according 

2 

**GPTSwarm: Language Agents as Optimizable Graphs** 

to their topological order. Each node _n ∈ N_ receives as input _x_ and the output _zn_ from its predecessor nodes. In this work, inputs and outputs are strings in natural language, but may take on other data types more generally. Node _n_ applies the computational routine _fn_ ( _zn, x_ ) and sends the output to its successor nodes. The graph output, denoted ˆ _y_ = _G_ ( _x_ ), is the output _fo_ ( _zo, x_ ) from the output node _o_ . Note that in a DAG, some nodes will not have predecessors. For such nodes, the context _z_ will be empty. This graph execution procedure is summarized in Algorithm 1. 

**Algorithm 1** Graph Execution **Require:** Computational graph _G_ = ( _N, E, F, o_ ), input _x_ , empty context _z_ for each node without predecessors. **for** _n_ in TopologicalSort( _N_ ) **do** _zn ←{fv_ ( _zv, x_ ) : _v ∈_ pre( _n_ ) _}_ **end for Ensure:** _fo_ ( _zo, x_ ) 

In the context of language agents, for example, the input _x_ may correspond to a question in natural language. Each node processes the input _x_ and context information _z_ from its predecessor nodes by applying a computational routine _f_ . Examples of routines include LLM queries with input data from other agents, instructions to generate prompts for web searches that gather task-related information, or tool usage. Although our formalization specifies that the input _x_ is given to each node, in practice, many routines might be designed to ignore the input and operate solely in the context provided by the predecessor nodes. Finally, the output provided by the output node corresponds to the answer to the input question or, more generally, to the solution of the input task. 

**Swarm of language agents as a composite graph.** Given a set of _K_ language agents, each represented by a computational graph _{Gk_ = ( _Nk, Ek, Fk, ok_ ) _}[K] k_ =1[,][one][can] compose these agents to achieve high performance in specific tasks. Let _N[′]_ = _∪kNk_ represent the union of the nodes of the agents, _E[′]_ = _∪kEk_ be the union of the edges of the agents, _F[′]_ = _∪kFk_ be the union of the computational routines of the agents, and _o[′] ∈∪k{ok}_ be the output node for the composite graph. Consider a selection of edges _E ⊂∪i_ = _jNi×Nj_ that describe a set of connections between nodes from different agents. We define the composite graph representing the swarm of agents as _GE_ = ( _N[′] , EE , F[′] , o[′]_ ), where _EE_ = _E[′] ∪E_ is the union of the edges of the agents and the new edges connecting them. Composite graphs are restricted to DAGs. The composite graph _GE_ can be executed as described in Algorithm 1. In a swarm of language agents, the newly specified edges represent communication channels between agents. In the following sections, we explore how to optimize such a computational graph. 

## **2.3. Edge Optimization** 

Given a task _τ_ and its associated utility function _uτ_ that maps the candidate graphs to real numbers, we formulate an optimization problem about the choice of additional edges. The goal is to identify the edges that connect various language agents in a swarm, maximizing the utility. This process involves determining the most effective patterns of communication and information exchange among agents for the task at hand. We consider a set of potential edges _{ei}[d] i_ =1[=] _[E]_[,][which][leads][to][2] _[d]_[possible][edge] configurations, symbolized as _E ∈{_ 0 _,_ 1 _}[d]_ . We further restrict the search space to only consider composite graphs that are DAGs. Formally, optimization of the composite graph of language agents is achieved by solving the problem max _E uτ_ ( _GE_ ). 

## 2.3.1. PROBLEM REFORMULATION 

DAG optimization through pruning of nodes and edges was already present in the first work on “deep learning” with deep feedforward networks (Ivakhnenko et al., 1965; Ivakhnenko, 1968). Due to the combinatorial complexity induced by DAGs, recent studies have increasingly focused on the continuous optimization approach (Vowels et al., 2022). This is particularly relevant in scenarios where most node executions require one or more queries to LLMs for moderate-scale applications. Moreover, the utility function is typically non-differentiable due to the tokenization of LLMs, and this remains true even when a differentiable DAG sampling technique is employed. Therefore, we reformulate our edge optimization as a continuous optimization problem. Instead of optimizing in a discrete space, our approach is to optimize over a continuum of probabilistic distributions, each representing a distribution over the feasible DAGs. Formally, rather than solving the maximum utility function arg max _E uτ_ ( _GE_ ), we propose solving 


![](raw/papers/2_LLM_NLP/LLM_paper_list/graphGPT/icml_2024_GPTSwarm_Language_Agents_as_Optimizable_Graphs_images/icml_2024_GPTSwarm_Language_Agents_as_Optimizable_Graphs.pdf-0003-09.png)


where _Dθ_ is a parameterized distribution and Θ represents a feasible set of real-valued parameters. 

## 2.3.2. SOLUTION PARAMETERIZATION 

A straightforward way to define a parameterized probabilistic distribution over DAGs with fixed nodes _N_ and required edges _E_ is to assign a real-valued parameter _θi ∈_ R to each potential edge _ei_ . Let _θ_ = [ _θ_ 1; _θ_ 2; _. . ._ ; _θd_ ] _∈_ [0 _,_ 1] _[d]_ . The probability of _G[′]_ = _GE_ for _G[′] ∼ Dθ_ is 


![](raw/papers/2_LLM_NLP/LLM_paper_list/graphGPT/icml_2024_GPTSwarm_Language_Agents_as_Optimizable_Graphs_images/icml_2024_GPTSwarm_Language_Agents_as_Optimizable_Graphs.pdf-0003-13.png)


A sampling method that realizes this distribution is first to initialize a graph _G[′] ←_ ( _N, E_ ). Then, iteratively sample 

3 

**GPTSwarm: Language Agents as Optimizable Graphs** 

whether to include edge _ei_ in _G[′]_ for all _i_ ’s. If including _ei_ causes a cycle in current _G[′]_ , then the edge would not be included. Otherwise, add the edge to _G[′]_ with probability _θi_ . 

## 2.3.3. OPTIMIZATION ALGORITHM 

To optimize the objective function (Equation (1)), we apply the REINFORCE algorithm (Williams, 1992) by applying a gradient ascent variant (e.g., Adam (Kingma & Ba, 2014)) with an unbiased gradient estimation: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/graphGPT/icml_2024_GPTSwarm_Language_Agents_as_Optimizable_Graphs_images/icml_2024_GPTSwarm_Language_Agents_as_Optimizable_Graphs.pdf-0004-04.png)


where _G_ 1 _, G_ 2 _, . . . , GN ∼ Dθ_ are mutually independent and _u_ ˆ _τ_ ( _Gi_ ) is an independent unbiased estimate of _uτ_ ( _Gi_ ) for all _i_ and some _M ∈_ N. Algorithm 2 describes the optimization algorithm with vanilla gradient ascent. 

## **Algorithm 2** Edge Optimization with REINFORCE 

**Require:** A parameterized probabilistic distribution over computation graphs _Dθ_ , an unbiased utility estimator ˆ _uτ_ ( _·_ ), and a learning rate _α_ . Initialize _θ ∈_ R _[d]_ . **while** terminate condition not met **do** Sample _Gi ∼ Dθ_ for _i_ = 1 _,_ 2 _, . . . , M_ . Update _θ ← θ_ + _M[α]_ � _Mi_ =1 _[u]_[ˆ] _[τ]_[(] _[G][i]_[)] _[∇][θ]_[ log(] _[p][θ]_[(] _[G][i]_[))][.] **end while** 

## **2.4. Node Optimization** 

In our framework, each node implements a fundamental operation, such as querying an LLM, using a tool, calling an API, etc. In a language agent, most of these operations involve prompting an LLM once or several times. Optimizing the prompts of these nodes is crucial for improving the system’s overall performance. 

Unlike many other works on prompt optimization, which optimize a single global prompt (e.g., Yang et al., 2023; Pryzant et al., 2023; Deng et al., 2022), our node optimization problem naturally involves several operations where each of them consists of a node-level prompt. In our case, the optimization problem is more complex due to prompts affecting how other prompts operate on connected nodes. At the same time, our graph representation leads to a separation of concerns where each node has a specific purpose with its own associated prompt. Due to this separation of concerns, we hypothesize that, for every optimization step, it is sufficient to update each node-level prompt individually, assuming that all other prompts are fixed. 

Consider a parameterized computational graph _G[P]_ = ( _N, E, F[P] , o_ ), where _F[P]_ = _{fn[p][n][}]_[ are computational rou-] tines, each parameterized by a prompt _pn_ to be optimized for all _n ∈ N_ . To enable effective node optimization, we 

also require a natural language description of the intended function for each routine _fn[p][n] ∈ F_ denoted by _dn_ . For example, a suitable description for a node designed to write Python programs would be “a Python code generator”. Here, existing prompt optimization methods, such as OPRO (Yang et al., 2023), can be described as a function _I_ that iteratively maps a prompt, a function description, and a set of node input-output pairs (which may include annotations such as a quality measure for each pair) to an improved prompt. For example, _I_ could take a prompt such as “generate Python code”, a description “a Python code generator”, and an input-output pair “Input: evaluate two divided by one as an integer. Output: 2 / 1”, where the output yields 1 _._ 0 as the result of execution. A prompt optimization method would return an improved prompt “generate Python code and pay attention to data types”. 

Formally, our method begins by initializing an empty history set, denoted _hn_ , one for each node _n ∈ N_ . The process then proceeds iteratively: first, the graph _G[P]_ ( _x_ ) is executed using a randomly sampled input _x_ following Algorithm 1. Subsequently, for each node, a tuple consisting of the input to the node ( _zn, x_ ), where _zn_ is the context vector that includes the outputs of the predecessor nodes, and the node’s own output _fn[p][n]_[(] _[z][n][, x]_[)][,][is added to the node’s his-] tory _hn_ . The final step involves updating the node prompts. This is done by applying _I_ to the node’s updated history, its current prompt, and its function description, resulting in an improved prompt _I_ ( _hn, pn, dn_ ). This iterative process, described in Algorithm 3, continuously improves the operations of the nodes in the entire graph. 

## **Algorithm 3** Node Optimization 

**Require:** A parameterized graph _G[P]_ = ( _N, E, F[P] , o_ ), natural language function descriptions _D_ = _{dn}n∈N_ , and a distribution of inputs _DX_ . Initialize _pn_ for all _n ∈ N_ . Initialize _hn ←∅_ for all _n ∈ N_ . **while** terminate condition not met **do** Sample input _x ∼ DX_ . _y ← G[P]_ ( _x_ ) following Algorithm 1. _hn ← hn ∪{_ (( _zn, x_ ) _, fn[p][n]_[(] _[z][n][, x]_[))] _[}]_[ for all] _[ n][ ∈][N]_[.] _pn ← I_ ( _hn, pn, dn_ ), for all _n ∈ N_ . **end while** 

## **2.5. General Applicability** 

Frameworks such as AutoGPT (Torantulino et al., 2023) and LangChain (Chase, 2022) have set a standard for flexibility and reusability in various language-based tasks. Our framework, GPTSwarm, introduces a graph-based design of agents and swarms. This design further simplifies the reuse of modular components (nodes & agents) and the integration of such modules. For instance, GPTSwarm supports 41 types of file analysis, web search (e.g., Google Search), 

4 

## **GPTSwarm: Language Agents as Optimizable Graphs** 


![](raw/papers/2_LLM_NLP/LLM_paper_list/graphGPT/icml_2024_GPTSwarm_Language_Agents_as_Optimizable_Graphs_images/icml_2024_GPTSwarm_Language_Agents_as_Optimizable_Graphs.pdf-0005-01.png)


**----- Start of picture text -----**<br>
Evaluation Scores<br>Evaluation Mode<br>Full graph 1T1A<br>Randomly-connected<br>Optimized<br>Baseline<br>3T3A<br>5T5A<br>7T7A<br>0.0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8<br>Score<br>**----- End of picture text -----**<br>


_Figure 2._ **Score recovery through edge optimization.** “T” denotes truthful and “A” adversarial agents, e.g., a 3T3A swarm has 3 of each. Ablation studies include a “full graph” and random graphs sampled according to distribution _D_ **0** _._ **5** . The dashed line corresponds to the direct answer baseline. 

and index-based memory. By offering a wide range of modules, our framework makes it easier to implement various language agent systems. See Section 3.4 for further details. 

## **3. Experiments** 

## **3.1. MMLU** 

## 3.1.1. ADVERSARIAL SETTING 

**Motivation:** In our first experiments, we demonstrate that edge optimization effectively filters adversarial agents from a swarm, mirroring a scenario in multi-agent systems where some agents are detrimental rather than beneficial. Ideally, optimization would automatically eliminate harmful agents. We conducted this experiment using the 4-choice MMLU general knowledge question answering dataset, as detailed by Hendrycks et al. (2021b;a). Our setup involves initializing a swarm consisting of _k_ Input-Output (IO) agents and _k_ adversarial agents, following the terminology by Besta et al. (2023). The IO agents query an LLM and relay the LLM’s responses directly. In contrast, adversarial agents are deliberately programmed to manipulate the LLM to provide incorrect answers. The collective decision on the final answer is made through majority voting, bypassing any additional LLM query that could introduce corrective intelligence against adversarial influence. We benchmark the performance of a single IO agent as our baseline. An effective optimization is expected to elevate the swarm’s performance on the MMLU dataset to match this baseline level. 

**Analysis:** In Figure 2, we present the comparative performance scores of different swarm configurations: the baseline, the graph formed by sequentially including edges that do not create loops (denoted as the ‘full graph’), a randomly connected swarm sampled from the initial distribution _Dθ_ with _θ_ = 0 _._ 5, and the optimized swarm. These scores 

are derived from evaluating the initial 10% of the MMLU validation set. The edge optimization process applies REINFORCE (Alg. 2) for 200 iterations. Each iteration assesses four graph samples, each on a specific problem sourced from the MMLU dev set. In all experiments, we used GPT4-Turbo with the token sampling temperature 0 _._ 2. Figure 10 demonstrates how the optimized swarm score aligns asymptotically with that of the baseline. Table 4 compiles the key statistics and findings of these experiments. The findings indicate that our approach successfully safeguards a swarm against harmful adversaries. 

## 3.1.2. COLLABORATIVE SETTING 

In the adversarial setting above, we observe no score improvement over the single agent baseline. We attribute this to the fact that all IO agents are prompted identically. To elicit score improvement, we run a set of 7 different IO agents instructed to behave according to various roles. See Appendix E.1.3 for the list of roles. The optimized swarm improves over the baseline from Sec. 3.1.1 by 2.1% _±_ 1.1% averaged over 5 training seeds. 

## **3.2. Mini Crosswords** 

**Motivation:** This section investigates to what extent edge optimization can improve the performance of standard agents from the literature. We conduct our evaluation on the Mini Crosswords dataset[1] . A subset of 20 problems is used to optimize and evaluate our methods, in agreement with previous studies (Yao et al., 2023; Sel et al., 2023). The choice of Mini Crosswords for this analysis is strategic, as it highlights how the algorithmic structure of the solvers, such as the tree search employed by TOT, significantly influences their performance (Yao et al., 2023). Our hypothesis is that edge connections can meaningfully determine the algorithmic structure. Through edge optimization, we anticipate the automatic discovery and implementation of high-performance algorithms. 

**Analysis:** In our experiments, we explore the performance of swarms of three distinct agents. The first agent, which implements the TOT approach, iteratively branches over candidate solutions provided by an LLM, processing one word at each step. The second agent is based on the Reflexion method (Shinn et al., 2023). This agent first proposes a solution through a greedy approach and then creates an alternative solution informed by feedback from a critic, which is based on an LLM analysis of the initial solution. The third agent we examine is a Chain of Thought (COT) agent consisting of three nodes. Each node within the COT performs an internal brute-force search to select the optimal subset of candidates generated by the LLM for the current state, scored by the LLM. The agent or swarm then returns 

1https://www.goobix.com/crosswords 

5 

**GPTSwarm: Language Agents as Optimizable Graphs** 


![](raw/papers/2_LLM_NLP/LLM_paper_list/graphGPT/icml_2024_GPTSwarm_Language_Agents_as_Optimizable_Graphs_images/icml_2024_GPTSwarm_Language_Agents_as_Optimizable_Graphs.pdf-0006-01.png)


**----- Start of picture text -----**<br>
0 2 4<br>6 8 10<br>**----- End of picture text -----**<br>


_Figure 3._ **Visualizing the evolution of the probability distribution during optimization in adjacency-like matrices.** In this figure, we show the probability parameters (one corresponds to an edge) in an adjacency-like matrix for iterations 0, 2, 4, 6, 8, and 10 of optimizing the objective for the Mini Crosswords task. We observe that the parameters first change chaotically. However, after iteration 6, the parameters change almost monotonically. 


![](raw/papers/2_LLM_NLP/LLM_paper_list/graphGPT/icml_2024_GPTSwarm_Language_Agents_as_Optimizable_Graphs_images/icml_2024_GPTSwarm_Language_Agents_as_Optimizable_Graphs.pdf-0006-03.png)


**----- Start of picture text -----**<br>
Mini Crosswords Performance<br>Final Distribution with<br>GPT-4-Turbo<br>Final Distribution with<br>GPT-3.5-Turbo<br>0.125 Parameter-Valued<br>Distribution<br>Initial Distribution<br>Best-of-Three<br>ToT (0.675)<br>0.0 0.2 0.4 0.6 0.8<br>Objective Value<br>**----- End of picture text -----**<br>


_Figure 4._ **Edge optimization on the Mini Crosswords dataset improves over standard methods** The baseline methods are evaluated with GPT-3.5-Turbo. The optimized final distribution outperforms several baselines. When evaluating the already optimized edge distribution with GPT-4-Turbo, we achieve better results compared to the previous state-of-the-art method (Tree of Thought evaluated with GPT-4). 

## all the solutions generated by their output node. 

For the utility function, we choose the best of all the graphreturned solutions according to the number of words correctly filled (i.e., best state word accuracy) as done by Yao et al. (2023). During the evaluation, we average over 20 graph samples from the graph distribution, each evaluated on a unique question randomly sampled from the dataset. 

We optimize our composite graph of agents using the REINFORCE (Alg. 2), setting the initial edge probability to _θ_ = 10% and the learning rate to _α_ = 0 _._ 4. For each iteration, the gradient is estimated according to equation (2) by sampling _M_ = 20 graphs, each evaluated on a crossword problem. For cost-effectiveness, we optimize and evaluate graphs with the GPT-3.5-Turbo language model, where the temperature is set to zero. Figure 3 visualizes the evolution of probability parameters in the form of adjacency-like matrices over ten iterations. We observe that the parameters first change chaotically. However, after iteration 6, the parameters change almost monotonically. 

We follow Alg. 2 to optimize the objective in Equation 1, achieving an average accuracy of 0 _._ 575( _±_ 0 _._ 0275) after ten iterations (we report the average over 3 runs and the standard error). This surpasses the initial distribution’s score of 0 _._ 465( _±_ 0 _._ 0509). Furthermore, we evaluate the best-ofthree performance by aggregating the top results from each problem across the three agents, which yields an accuracy of 0 _._ 320( _±_ 0 _._ 0415). 

Note that denser graphs are likely to require more computational resources. To verify that the improvements of our method are not solely due to an increase in the number of edges and therefore a larger computational budget, we compare it with a distribution with all parameters set to _θ_ = 12 _._ 5%. This value reflects the average number of 

edges in the learned distribution, determined by sampling 1000 graphs from each run’s resulting distribution. The expected number of edges for both the learned distribution and the 0 _._ 125 parameter-valued distribution are approximately 32 _._ 76( _±_ 1 _._ 93) and 32 _._ 80( _±_ 0 _._ 11), respectively. Despite the similarity in the edge count, the 0 _._ 125 parameter-valued distribution achieves an accuracy of 0 _._ 510( _±_ 0 _._ 0552), allowing us to attribute the improvements to factors beyond the mere edge density. 

Furthermore, we evaluate one of our final optimized distributions (randomly selected) with the GPT-4-Turbo language model[2] (Figure 4). It achieves an accuracy of 0 _._ 800( _±_ 0 _._ 0616), significantly outperforming the previous state-of-the-art method, TOT evaluated with GPT-4, with an accuracy of 0 _._ 675 (Yao et al., 2023). For a fair comparison, we also evaluate the TOT implementation by Yao et al. (2023) with GPT-4-Turbo. It yields an accuracy of 0 _._ 668. 

## **3.3. HumanEval** 

**Motivation:** In the previous experiments on MMLU (math problems) and Mini Crosswords (open-ended puzzles) we have validated the utility of optimizing graph edges. In this section, we test the HumanEval dataset (Chen et al., 2021), which is known to be sensitive to prompt design. Previous research involved manually crafting prompts (Shinn et al., 2023; Hong et al., 2023) and achieved impressive performance. In contrast, here, we explore how node-based optimization can simplify this process. 

**Analysis:** In this section, we optimize the prompts of a 

> 2We are limiting our evaluation to a single graph distribution due to the high cost associated with API calls for this type of evaluation. 

6 

**GPTSwarm: Language Agents as Optimizable Graphs** 


![](raw/papers/2_LLM_NLP/LLM_paper_list/graphGPT/icml_2024_GPTSwarm_Language_Agents_as_Optimizable_Graphs_images/icml_2024_GPTSwarm_Language_Agents_as_Optimizable_Graphs.pdf-0007-01.png)


**----- Start of picture text -----**<br>
Optimization Curve on HumanEval Dataset<br>0.88<br>0.86<br>0.84<br>0.82<br>0.80<br>0.78<br>0.76<br>1 2 3 4 5 6 7 8<br>Number of Iterations<br>Accuracy<br>**----- End of picture text -----**<br>


_Figure 5._ **Optimization curve on HumanEval.** Accuracy as a function of the number of iterations. We observe significant improvements during the first five iterations. The results, including the mean and standard errors, are based on three repeated experiments. 

ReAct-style (Yao et al., 2022) agent. The agent first generates a Python program in response to a given question. If the generated program passes all test cases included in the problem statement, then the program is returned. Otherwise, the agent regenerates a program based on the execution feedback. We optimize the prompts by adding input-output pairs selected by assessing their effectiveness as demonstration examples, particularly in improving the node operation applied to the node’s ten most recent inputs. To evaluate a node operation, we determine if the generated program successfully solves the unit tests provided in the input problem statement. For more details on the node optimizer, please refer to Appendix E.3.1. 

Figure 5 shows the optimization results of the first eight iterations. After each iteration, the optimized solution is evaluated on the entire dataset. The mean and standard errors are presented for three repeated runs. We observe that the accuracy increases during the first five iterations. We also test our method in an online learning setting, continuously optimizing and evaluating without restarting. An improvement is observed from 0.76 (no optimization) to 0 _._ 88( _±_ 0 _._ 007) (mean and standard error). 

## **3.4. GAIA** 

**Motivation:** GAIA is a benchmark specifically designed for testing the generality of AI assistants focusing on realworld questions (Mialon et al., 2023). Abilities required to answer GAIA questions include reasoning, multi-modality processing, web browsing, and other tool use. Although conceptually straightforward for humans, these questions present significant challenges for current AI systems. 

Using this benchmark, we evaluate the general applicability of our framework. We construct swarms with multiple agents of the same type and employ self-consistency (a prompt-based majority vote) for the final decision (Wang 


![](raw/papers/2_LLM_NLP/LLM_paper_list/graphGPT/icml_2024_GPTSwarm_Language_Agents_as_Optimizable_Graphs_images/icml_2024_GPTSwarm_Language_Agents_as_Optimizable_Graphs.pdf-0007-08.png)


_Figure 6._ **Solving a wide range of tasks requires many different tools.** The GAIA benchmark (Mialon et al., 2023) tests for many of these capabilities by including questions that require several of these tools for successful completion. 

_Table 1._ **Performance on the GAIA Benchmark (Mialon et al., 2023).** Using our framework, we demonstrate significant improvements across several levels of difficulty. The ‘GPT-4 with plugins’ baseline is less significant since it involves the manual selection of the appropriate tools per question. We report the mean and standard deviation across 5 runs. 

|**Method**|**Level 1**<br>**Level 2**<br>**Level 3**<br>**Avg.**|**Level 1**<br>**Level 2**<br>**Level 3**<br>**Avg.**|
|---|---|---|
|GPT-3.5||7.55<br>4.65<br>0<br>4.85|
|GPT-4||15.09<br>2.33<br>0<br>6.06|
|GPT-4-Turbo||20.75<br>5.81<br>0<br>9.70|
|AutoGPT||13.21<br>0<br>3.85<br>4.85|
|GPTSwarm|**30.56**_±_3.25 **20.93**_±_1.27 **3.85**_±_2.43<br>**18.45**||
|_Improvement_|47.3%_↑_<br>260.2%_↑_<br>0.0%<br>90.2%_↑_||
|GPT4 with Plugins*|30.30<br>9.70<br>0<br>14.6||



et al., 2022). We also experimented with adding different types of agents to the swarm and using prompt-based best answer selection. The results indicate that prompt-based self-consistency yields the best performance. Note that these experiments are meant to demonstrate the generic capabilities of our modular framework and include neither edge-based nor node-level optimization, which is left for future work. 

**Analysis:** Table 1 shows the results of our swarm with seven TOT agents and the self-consistency strategy for the final decision. We compare the performance of the GPT-Series (Achiam et al., 2023) with plugins and AutoGPT (Torantulino et al., 2023) performance as reported by Mialon et al. (2023). Our methods significantly outperform these baselines. 

Table 2 presents a more comprehensive set of results. We experiment with varying numbers of agents and different node operations, such as different tool uses. Our observations indicate that the time requirement of a swarm grows approximately linearly with the number of agents. Despite the increased computational time, incorporating more agents 

7 

**GPTSwarm: Language Agents as Optimizable Graphs** 

_Table 2._ **Ablations on the GAIA benchmark (Level 1 validation set) (Mialon et al., 2023).** DA = DirectAnswer, GQ = GenerateQuery, WS = WebSearch, FA = FileAnalyzer, CA = CombinedAnswer. ‘�’ indicates the presence of a specific feature in the corresponding framework, ‘�’ its absence. Each type of experiment is run five times to record the mean, standard deviation, and best run (marked as Best). Self-Consistency describes prompt-based self-consistency (Wang et al., 2022); Choose “Best” refers to the LLM’s favorite answer among the different agents’ answers. All agents and swarms are implemented using our GPTSwarm framework. 

|**Agent or Swarm**<br>**DA**<br>**GQ**<br>**WS**<br>**FA**<br>**CA**<br>**Decision Strategy**|**Accuracy**<br>**Best**<br>**Duration(s)**|
|---|---|
|**(A)**Agent: IO<br>�<br>�<br>�<br>�<br>�<br>N/A|16.60_±_3.02<br>20.75%<br>_∼_13.37|
|**(B)**Agent: COTweb<br>�<br>�<br>�<br>�<br>�<br>N/A<br><br><br><br><br>|18.87_±_2.67<br>22.64%<br>_∼_60.90|
|**(C)**Agent: COTFA<br>�<br>�<br>�<br>�<br>�<br>N/A|25.28_±_3.50<br>30.18%<br>_∼_56.42|
|**(D)**Agent: TOT<br>�<br>�<br>�<br>�<br>�<br>N/A|25.66_±_3.50<br>30.18%<br>_∼_71.31|
|**(E)**Swarm(3_×_IO)<br>�<br>�<br>�<br>�<br>�<br>Choose “Best”<br>**(F)**Swarm(3_×_COT)<br>�<br>�<br>�<br>�<br>�<br>Choose “Best”<br>**(G)**Swarm(3_×_TOT)<br>�<br>�<br>�<br>�<br>�<br>Choose “Best”|15.85_±_0.92<br>18.87%<br>_∼_45.65<br>27.17_±_3.29<br>32.08%<br>_∼_152.89<br>30.18_±_4.30<br>35.85%<br>_∼_198.50|
|**(H)**Swarm(3_×_IO)<br>�<br>�<br>�<br>�<br>�<br>Self-Consistency|18.11_±_3.07<br>22.64%<br>_∼_45.70|
|**(I)**Swarm(3_×_COT)<br>�<br>�<br>�<br>�<br>�<br>Self-Consistency|27.17_±_4.06<br>32.08%<br>_∼_150.26|
|**(J)**Swarm(3_×_TOT)<br>�<br>�<br>�<br>�<br>�<br>Self-Consistency|28.30_±_3.38<br>32.08%<br>_∼_181.15|
|**(K)**Swarm(5_×_TOT)<br>�<br>�<br>�<br>�<br>�<br>Self-Consistency|29.06_±_2.56<br>32.08%<br>_∼_291.07|
|**(L)**Swarm(7_×_TOT)<br>�<br>�<br>�<br>�<br>�<br>Self-Consistency|30.56_±_3.25<br>35.85%<br>_∼_414.89|
|**(M)**Human<br>-<br>-<br>-<br>-<br>-<br>-|94%<br>-<br>_∼_422.26|



notably improves the overall performance of the system. We also found that a greater variety of node operations leads to better performance. As illustrated in Figure 6, web browsing is required for 43.9% of the tasks. Our current implementation accesses the Internet by only downloading materials directly from the URLs provided in the problem statement or querying a Google search[3] without further website navigation. Therefore, we believe that enhancing web capabilities would further increase performance significantly. 

Agents (Park et al., 2023), ChatDev (Qian et al., 2023), and AutoGen (Wu et al., 2023) focus on role-based communication, but struggle with hallucinations. MetaGPT (Hong et al., 2023) introduces standard operating procedures for better role definition and communication, making the collaboration between agents more effective. In contrast to these frameworks, we automatically optimize nodes and edges in a self-organizing society of agents. 

## **4.2. Language Agents with Graphs** 

## **4. Related Work** 

## **4.1. LLM-based Autonomous Agents** 

Current works on LLM-based autonomous agents or language agents vary in focus. Methods such as Chain of Thought (Wei et al., 2022), ReAct (Yao et al., 2022), Reflexion (Shinn et al., 2023), Tree of Thought (ToT) improve prompt strategies and structure to improve reasoning capabilities, while others such as exchange-of-thoughts (EOT) (Yin et al., 2023) focus on the benefits of various communication paradigms. Single LLM agent frameworks such as AutoGPT (Gravitas, 2023), LangChain (Chase, 2022), LlamaIndex (Liu, 2022), and XAgent (XAgent Team, 2023) showcase problem solving through various external functions and tools. In the space of LLM-based multiagent systems (Xie et al., 2023; Chen et al., 2023a;b), NLSOMs (Zhuge et al., 2023) employ various social structures for task-specific applications (inspired by SOMs (Minsky, 1988)), without exploring optimization over the social structure of agents. CAMEL (Li et al., 2023), Generalist 

Besta et al. (2023) introduced LLM-based problem-solving with graphs; however, the approach only encompasses LLM prompting schemes without modeling other fundamental capabilities of language agents, such as use of external tools. LangGraph (langchain ai, 2024), on the other hand, is a concurrent open-source framework that focuses on building multi-actor state LLM applications through possibly cyclical operations. However, its practical applicability has not yet been systematically studied. Unlike previous studies, our approach emphasizes the development of hierarchical intelligence, as discussed by Minsky (1988) and Kennedy (2006), through the construction of agent graphs and the composition of multiple graphs into swarms. Crucially, the graph representation facilitates automatic optimization on two levels. First, at the node level, since the majority of nodes in the graph involve prompting an LLM, prompt optimization methods can be employed. Second, at the edge level, we demonstrate the application of the REINFORCE algorithm (Williams, 1992) to optimize the potential connections between nodes. 

> 3We use SearchApi (https://www.searchapi.io/) in the experiments 

8 

**GPTSwarm: Language Agents as Optimizable Graphs** 

## **4.3. Optimizing LLM Inference and Self-Improvement** 

Much of deep learning research is concerned with tuning the learning algorithms, architectures, hyper-parameters, and other aspects of the learning pipeline (Schmidhuber, 2015; Yan et al., 2015). Meta-learning attempts to automate large parts of that process (Schmidhuber, 1987; Elsken et al., 2019; Kirsch & Schmidhuber, 2021). Similarly, recently, a lot of research and engineering has gone into the prompting and structuring of LLM inference to make better use of LLMs and build better agents. Due to the ability of LLMs to learn in context (Brown et al., 2020; Kirsch et al., 2022), one can view this process as configuring learning algorithms. The optimization of the inference structure and the prompts can then be viewed as meta-learning in LLMs. 

In the realm of prompt optimization, OPRO (Yang et al., 2023) generates better prompts through iterative LLM queries using prior solutions and their performance. PromptBreeder (Fernando et al., 2023) implements a mechanism that evolves and self-improves task-specific and metaprompts through mutation and LLM prompting. Related to these works, we self-improve future prompts by prompting LLMs. Similarly to our work, DSPy (Khattab et al., 2023) implements LLM pipelines as computational graphs with modular LLM queries as nodes, parameterized by prompts and neural network weights. It proposes a two-stage process to optimize the parameters of these nodes. Initially, it generates a set of candidate solutions for each node. Subsequently, it optimizes across the Cartesian product of these candidate solution sets, aiming to identify an effective combination of parameters for the entire graph. To address the combinatorial optimization challenge raised in DSPy, we propose an iterative optimization process. By virtue of decomposing a solution into nodes with expected functions, at each iteration, we improve each node individually, conditioned on the execution history of the graph with the current prompts of each node. 

Regarding the optimization of the inference structure, DyLAN (Liu et al., 2023) uses a fixed heuristic to improve the collaboration of LLM agents by selecting agents and determining the number of communication rounds. In line with previous ideas on self-referential learning (Schmidhuber, 1993; Irie et al., 2022; Kirsch & Schmidhuber, 2022), STOP (Zelikman et al., 2023) optimizes both the prompts and the inference structure together by introducing an initial improver program that is applied to itself to iteratively improve its performance. In our work, we optimize the inference structure by employing RL techniques applied to the potential edges of a given graph. 

## **5. Conclusion** 

This paper introduces GPTSwarm, an open-source framework that constructs language agents from graphs and agent societies from graph compositions. This approach allows for the easy implementation of existing methods from basic node operations and enables automatic optimization of the graph in the form of node-level improvement and edge-level REINFORCE optimization. Our experiments demonstrate the advantages of our language agent graphs and automatic optimization on several benchmarks. 

## **Impact Statement** 

This paper presents work whose goal is to advance the field of Machine Learning. The societal consequences of our work are multifaceted. On the one hand, it could lead to significant advancements in the efficiency and effectiveness of machine learning systems. On the other hand, the increased capability and automation of LLM agents might raise ethical and employment concerns. As AI systems become more autonomous and powerful, it is crucial to consider their impact on job displacement and the importance of implementing safeguards to prevent biased or unethical AI behaviors. Furthermore, the potential for misuse of advanced AI technologies requires rigorous oversight and the development of ethical guidelines to ensure that these technologies are used responsibly and for the benefit of society as a whole. 

## **Author Contributions** 

Mingchen initiated the project and conceived the initial idea, led the development of the codebase, conducted GAIA & HumanEval experiments, drafted the initial manuscript, and created most of the visualizations. Wenyi discussed the core ideas with Mingchen, contributed to the codebase, conducted Mini CrossWords & HumanEval experiments, and drafted the initial manuscript. Louis reviewed and polished the paper, extensively rewrote the introduction, and coordinated team meetings. Francesco reviewed and polished the paper, significantly revising the methods section. Furthermore, Louis and Francesco discussed and formalized various techniques for graph optimization. As the senior engineering lead, Dmitrii advised and made significant revisions to the codebase, conducted the MMLU experiments, and contributed to the visualizations. Juergen, as mentor and advisor, offered guidance and support throughout the project’s progression. 

## **Acknowledgements** 

This work was supported by the SDAIA-KAUST Center of Excellence in Data Science and Artificial Intelligence 

9 

**GPTSwarm: Language Agents as Optimizable Graphs** 

(SDAIA-KAUST AI). It was further supported by the European Research Council (ERC, Advanced Grant Number 742870) and the Swiss National Science Foundation (SNF, Grant Number 200021 192356). 

## **References** 

- Achiam, J., Adler, S., Agarwal, S., Ahmad, L., Akkaya, I., Aleman, F. L., Almeida, D., Altenschmidt, J., Altman, S., Anadkat, S., et al. Gpt-4 technical report. _arXiv preprint arXiv:2303.08774_ , 2023. 

- Auer, P., Cesa-Bianchi, N., and Fischer, P. Finite-time analysis of the multiarmed bandit problem. _Machine learning_ , 47:235–256, 2002. 

- Besta, M., Blach, N., Kubicek, A., Gerstenberger, R., Gianinazzi, L., Gajda, J., Lehmann, T., Podstawski, M., Niewiadomski, H., Nyczyk, P., et al. Graph of thoughts: Solving elaborate problems with large language models. _arXiv preprint arXiv:2308.09687_ , 2023. 

- Brown, T., Mann, B., Ryder, N., Subbiah, M., Kaplan, J. D., Dhariwal, P., Neelakantan, A., Shyam, P., Sastry, G., Askell, A., et al. Language models are few-shot learners. _Advances in neural information processing systems_ , 33: 1877–1901, 2020. 

- Chase, H. LangChain. https://github.com/ hwchase17/langchain, 2022. 

- Chen, G., Dong, S., Shu, Y., Zhang, G., Sesay, J., Karlsson, B. F., Fu, J., and Shi, Y. Autoagents: A framework for automatic agent generation. _arXiv preprint arXiv:2309.17288_ , 2023a. 

- Chen, M., Tworek, J., Jun, H., Yuan, Q., Pinto, H. P. d. O., Kaplan, J., Edwards, H., Burda, Y., Joseph, N., Brockman, G., et al. Evaluating large language models trained on code. _arXiv preprint arXiv:2107.03374_ , 2021. 

- Chen, W., Su, Y., Zuo, J., Yang, C., Yuan, C., Qian, C., Chan, C.-M., Qin, Y., Lu, Y., Xie, R., et al. Agentverse: Facilitating multi-agent collaboration and exploring emergent behaviors in agents. _arXiv preprint arXiv:2308.10848_ , 2023b. 

- Deng, M., Wang, J., Hsieh, C.-P., Wang, Y., Guo, H., Shu, T., Song, M., Xing, E. P., and Hu, Z. Rlprompt: Optimizing discrete text prompts with reinforcement learning. _arXiv preprint arXiv:2205.12548_ , 2022. 

- Du, Y., Li, S., Torralba, A., Tenenbaum, J. B., and Mordatch, I. Improving factuality and reasoning in language models through multiagent debate. _arXiv preprint arXiv:2305.14325_ , 2023. 

- Elsken, T., Metzen, J. H., and Hutter, F. Neural architecture search: A survey. _The Journal of Machine Learning Research_ , 20(1):1997–2017, 2019. 

- Fernando, C., Banarse, D., Michalewski, H., Osindero, S., and Rocktaschel, T.¨ Promptbreeder: Self-referential self-improvement via prompt evolution. _arXiv preprint arXiv:2309.16797_ , 2023. 

- Gravitas, S. Auto-gpt. _GitHub repository_ , 2023. 

- Hendrycks, D., Burns, C., Basart, S., Critch, A., Li, J., Song, D., and Steinhardt, J. Aligning ai with shared human values. _Proceedings of the International Conference on Learning Representations (ICLR)_ , 2021a. 

- Hendrycks, D., Burns, C., Basart, S., Zou, A., Mazeika, M., Song, D., and Steinhardt, J. Measuring massive multitask language understanding. _Proceedings of the International Conference on Learning Representations (ICLR)_ , 2021b. 

- Hong, S., Zheng, X., Chen, J., Cheng, Y., Zhang, C., Wang, Z., Yau, S. K. S., Lin, Z., Zhou, L., Ran, C., et al. Metagpt: Meta programming for multi-agent collaborative framework. _arXiv preprint arXiv:2308.00352_ , 2023. 

- Irie, K., Schlag, I., Csordas, R., and Schmidhuber, J. A mod-´ ern self-referential weight matrix that learns to modify itself. In _International Conference on Machine Learning_ , pp. 9660–9677. PMLR, 2022. 

- Ivakhnenko, A., Lapa, V., and ENGINEERING., P. U. L. I. S. O. E. _Cybernetic Predicting Devices_ . JPRS 37, 803. Joint Publications Research Service [available from the Clearinghouse for Federal Scientific and Technical Information], 1965. URL https://books.google. com.sa/books?id=l38DHQAACAAJ. 

- Ivakhnenko, A. G. The group method of data handling, a rival of the method of stochastic approximation. _Soviet Automatic Control_ , 13(3):43–55, 1968. 

- Kennedy, J. Swarm intelligence. In _Handbook of natureinspired and innovative computing: integrating classical models with emerging technologies_ , pp. 187–219. Springer, 2006. 

- Khattab, O., Singhvi, A., Maheshwari, P., Zhang, Z., Santhanam, K., Vardhamanan, S., Haq, S., Sharma, A., Joshi, T. T., Moazam, H., et al. Dspy: Compiling declarative language model calls into self-improving pipelines. _arXiv preprint arXiv:2310.03714_ , 2023. 

- Kingma, D. P. and Ba, J. Adam: A method for stochastic optimization. _arXiv preprint arXiv:1412.6980_ , 2014. 

- Kirsch, L. and Schmidhuber, J. Meta learning backpropagation and improving it. _Advances in Neural Information Processing Systems_ , 34:14122–14134, 2021. 

10 

**GPTSwarm: Language Agents as Optimizable Graphs** 

- Kirsch, L. and Schmidhuber, J. Eliminating meta optimization through self-referential meta learning. _arXiv preprint arXiv:2212.14392 and First Conference on Automated Machine Learning (Workshop)_ , 2022. 

- Kirsch, L., Harrison, J., Sohl-Dickstein, J., and Metz, L. General-purpose in-context learning by meta-learning transformers. _arXiv preprint arXiv:2212.04458_ , 2022. 

- Kojima, T., Gu, S. S., Reid, M., Matsuo, Y., and Iwasawa, Y. Large language models are zero-shot reasoners. _Advances in neural information processing systems_ , 35: 22199–22213, 2022. 

- langchain ai. LangGraph. https://github.com/ langchain-ai/langgraph, 2024. 

- Li, G., Hammoud, H. A. A. K., Itani, H., Khizbullin, D., and Ghanem, B. Camel: Communicative agents for” mind” exploration of large scale language model society. _arXiv preprint arXiv:2303.17760_ , 2023. 

- Liu, J. LlamaIndex, 11 2022. URL https://github. com/jerryjliu/llama_index. 

- Liu, Z., Zhang, Y., Li, P., Liu, Y., and Yang, D. Dynamic llm-agent network: An llm-agent collaboration framework with agent team optimization. _arXiv preprint arXiv:2310.02170_ , 2023. 

- Mialon, G., Fourrier, C., Swift, C., Wolf, T., LeCun, Y., and Scialom, T. Gaia: a benchmark for general ai assistants. _arXiv preprint arXiv:2311.12983_ , 2023. 

- Minsky, M. _Society of mind_ . Simon and Schuster, 1988. 

- Nakajima, Y. Babyagi. _Python. https://github. com/yoheinakajima/babyagi_ , 2023. 

- Nepusz, T. and Vicsek, T. Hierarchical self-organization of non-cooperating individuals. _Plos one_ , 8(12):e81449, 2013. 

- Park, J. S., O’Brien, J., Cai, C. J., Morris, M. R., Liang, P., and Bernstein, M. S. Generative agents: Interactive simulacra of human behavior. In _Proceedings of the 36th Annual ACM Symposium on User Interface Software and Technology_ , pp. 1–22, 2023. 

- Pryzant, R., Iter, D., Li, J., Lee, Y. T., Zhu, C., and Zeng, M. Automatic prompt optimization with” gradient descent” and beam search. _arXiv preprint arXiv:2305.03495_ , 2023. 

- Qian, C., Cong, X., Yang, C., Chen, W., Su, Y., Xu, J., Liu, Z., and Sun, M. Communicative agents for software development. _arXiv preprint arXiv:2307.07924_ , 2023. 

- Schmidhuber, J. _Evolutionary principles in self-referential learning, or on learning how to learn: the meta-meta... hook_ . PhD thesis, Technische Universitat M¨ unchen,¨ 1987. 

- Schmidhuber, J. A ‘self-referential’weight matrix. In _ICANN’93: Proceedings of the International Conference on Artificial Neural Networks Amsterdam, The Netherlands 13–16 September 1993 3_ , pp. 446–450. Springer, 1993. 

- Schmidhuber, J. Deep learning in neural networks: An overview. _Neural networks_ , 61:85–117, 2015. 

- Sel, B., Al-Tawaha, A., Khattar, V., Wang, L., Jia, R., and Jin, M. Algorithm of thoughts: Enhancing exploration of ideas in large language models. _arXiv preprint arXiv:2308.10379_ , 2023. 

- Shinn, N., Cassano, F., Labash, B., Gopinath, A., Narasimhan, K., and Yao, S. Reflexion: Language agents with verbal reinforcement learning. _arXiv preprint arXiv:2303.11366_ , 14, 2023. 

- Torantulino et al. Auto-gpt. https://github.com/ Significant-Gravitas/Auto-GPT, 2023. 

- Vowels, M. J., Camgoz, N. C., and Bowden, R. D’ya like dags? a survey on structure learning and causal discovery. _ACM Computing Surveys_ , 55(4):1–36, 2022. 

- Wang, L., Ma, C., Feng, X., Zhang, Z., Yang, H., Zhang, J., Chen, Z., Tang, J., Chen, X., Lin, Y., et al. A survey on large language model based autonomous agents. _arXiv preprint arXiv:2308.11432_ , 2023. 

- Wang, X., Wei, J., Schuurmans, D., Le, Q., Chi, E., Narang, S., Chowdhery, A., and Zhou, D. Self-consistency improves chain of thought reasoning in language models. _arXiv preprint arXiv:2203.11171_ , 2022. 

- Wei, J., Wang, X., Schuurmans, D., Bosma, M., Xia, F., Chi, E., Le, Q. V., Zhou, D., et al. Chain-of-thought prompting elicits reasoning in large language models. _Advances in Neural Information Processing Systems_ , 35: 24824–24837, 2022. 

- Williams, R. J. Simple statistical gradient-following algorithms for connectionist reinforcement learning. _Machine learning_ , 8:229–256, 1992. 

- Wu, Q., Bansal, G., Zhang, J., Wu, Y., Zhang, S., Zhu, E., Li, B., Jiang, L., Zhang, X., and Wang, C. Autogen: Enabling next-gen llm applications via multi-agent conversation framework. _arXiv preprint arXiv:2308.08155_ , 2023. 

- XAgent Team. Xagent: An autonomous agent for complex task solving, 2023. 

11 

**GPTSwarm: Language Agents as Optimizable Graphs** 

- Xi, Z., Chen, W., Guo, X., He, W., Ding, Y., Hong, B., Zhang, M., Wang, J., Jin, S., Zhou, E., et al. The rise and potential of large language model based agents: A survey. _arXiv preprint arXiv:2309.07864_ , 2023. 

- Xie, T., Zhou, F., Cheng, Z., Shi, P., Weng, L., Liu, Y., Hua, T. J., Zhao, J., Liu, Q., Liu, C., et al. Openagents: An open platform for language agents in the wild. _arXiv preprint arXiv:2310.10634_ , 2023. 

- Yan, L. C., Yoshua, B., and Geoffrey, H. Deep learning. _nature_ , 521(7553):436–444, 2015. 

- Yang, C., Wang, X., Lu, Y., Liu, H., Le, Q. V., Zhou, D., and Chen, X. Large language models as optimizers. _arXiv preprint arXiv:2309.03409_ , 2023. 

- Yao, S., Zhao, J., Yu, D., Du, N., Shafran, I., Narasimhan, K., and Cao, Y. React: Synergizing reasoning and acting in language models. _arXiv preprint arXiv:2210.03629_ , 2022. 

- Yao, S., Yu, D., Zhao, J., Shafran, I., Griffiths, T. L., Cao, Y., and Narasimhan, K. R. Tree of thoughts: Deliberate problem solving with large language models. In _Thirtyseventh Conference on Neural Information Processing Systems_ , 2023. URL https://openreview.net/ forum?id=5Xc1ecxO1h. 

- Yin, Z., Sun, Q., Chang, C., Guo, Q., Dai, J., Huang, X.J., and Qiu, X. Exchange-of-thought: Enhancing large language model capabilities through cross-model communication. In _Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing_ , pp. 15135–15153, 2023. 

- Zelikman, E., Lorch, E., Mackey, L., and Kalai, A. T. Selftaught optimizer (stop): Recursively self-improving code generation. _arXiv preprint arXiv:2310.02304_ , 2023. 

- Zeng, A., Attarian, M., Ichter, B., Choromanski, K., Wong, A., Welker, S., Tombari, F., Purohit, A., Ryoo, M., Sindhwani, V., et al. Socratic models: Composing zero-shot multimodal reasoning with language. _arXiv preprint arXiv:2204.00598_ , 2022. 

- Zhuge, M., Liu, H., Faccio, F., Ashley, D. R., Csordas,´ R., Gopalakrishnan, A., Hamdi, A., Hammoud, H. A. A. K., Herrmann, V., Irie, K., et al. Mindstorms in natural language-based societies of mind. _arXiv preprint arXiv:2305.17066_ , 2023. 

12 

**GPTSwarm: Language Agents as Optimizable Graphs** 

## **A. The GPTSwarm Framework** 

## **A.1. The Vision** 

Many recent language agents are described as compositions of components of different functionalities (Wang et al., 2023). A popular tweet states: “agent = LLM + memory + planning skills + tool use.”[4] Such additive formulations highlight individual components, but fail to address the essential aspect of component _integration_ . GPTswarm’s computational graph formulation, however, precisely focuses on _integration_ through edge optimization, to learn improved recommendations of agent orchestration and precise agent routing. This will become increasingly relevant as swarm size increases to millions or billions of agents. 

## **A.2. Class Diagram** 

The GPTSwarm framework is developed using Python and PyTorch. Its class diagram is illustrated in Figure 7. 

On the graph level, Node, Graph, and CompositeGraph are directly implemented as classes. Graph edges are implicitly stored as an adjacency list within each Node. Functionally, the framework distinguishes between Agents and Operations through various classes, such as DirectAnswer and WebSearch for operations, and IO and TOT for agents. To encapsulate the abstraction of an external LLM, we introduce an interface named after it. The primary implementation of this interface is a lightweight wrapper around the OpenAI API. To facilitate dataset integration for optimization and evaluation, we provide implementations for two interfaces: Dataset and PromptSet. The Dataset interface is designed to load benchmark datasets like GAIA and MMLU, while the PromptSet customizes node behavior for a specific Dataset. The Evaluator class manages the optimization processes for both edges and nodes. 

The framework is highly customizable, allowing users to add more LLM backends, Dataset and PromptSet combinations, Agents, and Nodes as needed. Additionally, the framework makes extensive use of asynchronous computations for task parallelism, leveraging Python’s async-await syntax. 

> 4https://twitter.com/lilianweng/status/1673535600690102273 

13 

**GPTSwarm: Language Agents as Optimizable Graphs** 


![](raw/papers/2_LLM_NLP/LLM_paper_list/graphGPT/icml_2024_GPTSwarm_Language_Agents_as_Optimizable_Graphs_images/icml_2024_GPTSwarm_Language_Agents_as_Optimizable_Graphs.pdf-0014-01.png)


_Figure 7._ The class diagram of the GPTSwarm framework. 

14 

**GPTSwarm: Language Agents as Optimizable Graphs** 

## **B. Swarm examples** 

To facilitate understanding of the concepts presented in this study, we are showing a simple example of a swarm consisting of 3 agents: Tree-of-Thought, Input-Output, and Decision Agents in Figure 8. 


![](raw/papers/2_LLM_NLP/LLM_paper_list/graphGPT/icml_2024_GPTSwarm_Language_Agents_as_Optimizable_Graphs_images/icml_2024_GPTSwarm_Language_Agents_as_Optimizable_Graphs.pdf-0015-03.png)


_Figure 8._ A simple example of a swarm consisting of one Tree-of-Thought, one Input-Output, and the Decision agent. 

15 

**GPTSwarm: Language Agents as Optimizable Graphs** 

## **C. More Visualizations** 


![](raw/papers/2_LLM_NLP/LLM_paper_list/graphGPT/icml_2024_GPTSwarm_Language_Agents_as_Optimizable_Graphs_images/icml_2024_GPTSwarm_Language_Agents_as_Optimizable_Graphs.pdf-0016-02.png)


**----- Start of picture text -----**<br>
Agent: GAIA_WebCOT<br>Swarm: MMLU_5T5A Swarm: MMLU_3COT<br>Swarm: CrossWords_COT_TOT_Reflexion_1<br>Agent: GAIA_FileCOT<br>Agent: HumanEval_ReAct<br>Agent: GAIA_IO<br>Swarm: CrossWords_COT_TOT_Reflexion_2<br>Agent: GAIA_TOT<br>Swarm: CrossWords_COT_TOT_Reflexion_3<br>Swarm: GAIA_3WebCOT<br>Swarm: GAIA_3FileCOT<br>Swarm: GAIA_3TOT Swarm: GAIA_5TOT Swarm: GAIA_7TOT<br>**----- End of picture text -----**<br>


_Figure 9._ Different agents or swarms implemented by GPTSwarm. 

16 

**GPTSwarm: Language Agents as Optimizable Graphs** 

## **D. Additional Experiments** 

## **D.1. Comparing Our Method with** _**Multiagent Debate**_ **and** _**DYLAN**_ **on MMLU** 

Our MMLU experimental setup can be viewed as a multi-agent optimization problem with an equal number of truthful and adversarial agents in the system. We compare our approach to two baselines from the literature that naturally fit this problem: Multiagent Debate (Du et al., 2023) and DyLAN (Liu et al., 2023). 

For the Multiagent Debate baseline, we directly use the original setting with three truthful agents and three adversarial agents. For DyLAN, we adopt the original implementation (Liu et al., 2023), adding system prompts ”You are a knowledgeable expert in question answering.” for truthful agents, and ”Pretend you are a non-expert in question answering. You can provide wrong answers to the questions.” for adversarial agents. DyLAN performs optimization by scoring each agent and pruning a certain number of agents with the lowest scores. Since DyLAN has not fixed the number of pruned agents, we report the average performance in multiple settings, from 1 to 5 agents pruned. 

The results in Table 3 indicate that the Multiagent Debate is less effective in this setting. DyLAN achieves a 0 _._ 0065 improvement in accuracy compared to our method. We suspect that this improvement is due to DyLAN’s more complex debate scheme and prompts. This design also contributes to DyLAN’s larger computational cost. 

_Table 3._ **Results of the Multiagent Debate, DyLAN, and our method on MMLU.** We report the performance and computational cost of these methods applied to an LLM-based multiagent system with adversaries. Computational cost for optimization and inference is presented separately except for the Multiagent Debate, where there is no explicit separation. DyLAN is reported as an average over five different choices for the number of pruned agents. 

|**Methods**|**Cost (USD)**<br>**# Prompt Tokens**<br>**# Completion Tokens**<br>**Time (h)**<br>**Accuracy**|
|---|---|
|Multiagent Debate|32.8<br>1,689,960<br>530,005<br>8.36<br>0.5751|
|DyLAN optimization|105.93<br>5,671,276<br>1,640,566<br>25.4<br>-|
|DyLAN inference|14.99<br>628,009<br>290,472<br>4.75<br>0.8366|
|GPTSwarm optimization|5.32<br>361,812<br>56,770<br>0.9<br>-|
|GPTSwarm inference|1.82<br>113,233<br>22,923<br>0.31<br>0.8301|



## **D.2. Applying Node Optimization after Edge Optimization** 

As described in Section 3.2, we first perform ten REINFORCE steps to optimize the edges of a three-agent swarm, resulting in a 0 _._ 575( _±_ 0 _._ 0275) accuracy. Based on this optimized edge distribution we further apply our node optimization algorithm (Algorithm 3). For each node _n_ , _pn_ corresponds to the initial prompt, _hn_ consists of the input-output pairs obtained by executing the graphs on the 20 Mini Crosswords problems, and d is ignored. For each node _n_ , the improver _I_ either chooses a demonstration example from _hn_ to include as a part of _n_ ’s prompt or keeps the prompt unchanged. This choice is implemented using the upper confidence bound algorithm UCB1 (Auer et al., 2002), over a hundred iterations. Preference is given to pairs (or the choice of no demonstration example) that help the node accurately fill in more words on its previous input. Applying this node optimization after edge optimization improves accuracy to 0 _._ 668( _±_ 0 _._ 0060). 

## **E. Experimental Details** 

In our experiments involving multiple agents, we incorporate an additional virtual agent, represented by a single node, to serve as a final decision aggregator. This node is designated as the output node for the composite graph, and its specific implementation varies between different experiments. Common implementations for this node include employing a majority vote and a self-consistency strategy for decision-making. Unless explicitly stated, communication between agents within a composite graph does not include this virtual agent. Additionally, in all our experiments, the potential edge set of a composite graph is defined as all possible node pairs, provided that the nodes in each pair originate from different agents. We exclude any edges that would connect the output node of a composite graph to other nodes. Moreover, we employ Adam (Kingma & Ba, 2014) optimizer with parameters _β_ 1 = 0 _._ 9 _, β_ 2 = 0 _._ 999 and a variable learning rate in place of the vanilla stochastic gradient accent method described in Alg. 2. Finally, we use the version gpt-4-1106-preview and gpt-3.5-turbo-1106 for LLMs. For the vision-language model utilized in the GAIA experiments, we employed the gpt-4-1106-vision-preview version of GPT-4-Turbo. 

17 

**GPTSwarm: Language Agents as Optimizable Graphs** 

## **E.1. MMLU** 

We provide adversarial robustness experiments in Table 4. The convergence of the train utility to the baseline for the 3T3A experiment is shown in Figure 10. The first 10% of the validation dataset comprises 153 questions. 

_Table 4._ **Stats for the adversarial experiments.** #Nodes means the number of nodes in the swarm excluding the final decision node. #Potential edges is the total number of edges that are optimized and potentially realized. The optimization time is measured as the wall clock time. #LLM inferences is the total number of LLM queries made during the optimization cycle when graph pruning is turned off. 

|**Swarm confguration**<br>**#Nodes**|**#Potential edges**<br>**Optimization time, mins**|
|---|---|
|1 Trustful Agent + 1 Adversarial Agent<br>2|4<br>9|
|3 Trustful Agents + 3 Adversarial Agents<br>6|36<br>23|
|5 Trustful Agents + 5 Adversarial Agents<br>10|100<br>58|
|7 Trustful Agents + 7 Adversarial Agents<br>14|196<br>95|




![](raw/papers/2_LLM_NLP/LLM_paper_list/graphGPT/icml_2024_GPTSwarm_Language_Agents_as_Optimizable_Graphs_images/icml_2024_GPTSwarm_Language_Agents_as_Optimizable_Graphs.pdf-0018-05.png)


**----- Start of picture text -----**<br>
0.75<br>0.70<br>0.65<br>0.60<br>0.55<br>Mean score<br>0.50 1-sigma range<br>Direct answer baseline (dev split)<br>0 25 50 75 100 125 150 175 200<br>Iteration<br>Train score<br>**----- End of picture text -----**<br>


_Figure 10._ **The training score during the optimization of the adversarial swarm (3T3A) on MMLU.** We apply smoothing with an unbiased exponential moving average and the smoothness factor of 0 _._ 97. 

## E.1.1. HYPER-PARAMETERS & PROMPTS 

We use the Adam optimizer with a learning rate of 0 _._ 1 to update the logit parameters associated with each potential edge. The prompts that have been used for the adversarial robustness experiments are collected in Table 5. 

_Table 5._ **Prompts for the Adversarial experiments on MMLU.** 

|**Prompt purpose**|**Prompt**|
|---|---|
|System prompt|You are a knowledgeable expert in question answering. I will ask you a question. I will also give<br>you 4 answers enumerated as A, B, C and D. Only one answer out of the offered 4 is correct.<br>You must choose the correct answer to the question. Your response must be one of the 4 letters:<br>A, B, C or D, corresponding to the correct answer. Only one letter (A, B, C or D) is allowed in<br>your answer.|
|Direct answer template|_{_question_}_|
|Adversarial answer tem-<br>plate|Answer a lie to the following question: _{_question_}_.|
|Question template|_{_open-ended question_}_. Option A:_{_option A_}_, Option B:_{_option B_}_, Option C:_{_option C_}_,<br>Option D: _{_option D_}_.|



18 

**GPTSwarm: Language Agents as Optimizable Graphs** 

## E.1.2. ADVERSARIAL SWARM OPTIMIZATION CASE STUDY 

An example of a swarm with 2 truthful and 2 adversarial examples is shown in Figure 11. Figure 11a shows all potential edges before optimization. Figure 11b shows only the edges that were connected after optimization was complete. Note that the disconnected agents and edges are pruned. 


![](raw/papers/2_LLM_NLP/LLM_paper_list/graphGPT/icml_2024_GPTSwarm_Language_Agents_as_Optimizable_Graphs_images/icml_2024_GPTSwarm_Language_Agents_as_Optimizable_Graphs.pdf-0019-03.png)


(a) An non-optimized swarm with 2 truthful agents and 2 adversarial agents. Dotted arrows depict potential edges. 


![](raw/papers/2_LLM_NLP/LLM_paper_list/graphGPT/icml_2024_GPTSwarm_Language_Agents_as_Optimizable_Graphs_images/icml_2024_GPTSwarm_Language_Agents_as_Optimizable_Graphs.pdf-0019-05.png)


(b) An optimized swarm with realized edges. 

_Figure 11._ A 2T2A swarm before (a) and after (b) optimization 

## E.1.3. SWARM ROLES 

The list of roles randomly assigned to the IO agents is displayed in Table 6. 

_Table 6._ Roles that can be assigned to IO agents. 

|Botanist|Data Scientist|Social Worker|Journalist|Pilot|
|---|---|---|---|---|
|Anthropologist|Fitness Coach|Politician|Artist|Marine Biologist|
|Ethicist|Entrepreneur|Linguist|Archaeologist|Nurse|
|Graphic Designer|Philanthropist|Meteorologist|Sommelier|Cybersecurity Expert|



19 

**GPTSwarm: Language Agents as Optimizable Graphs** 

## **E.2. Mini Crosswords** 

## E.2.1. AGENTS SETTING 

In our Mini Crosswords experiments, each node returns one or two solutions—either updated or unchanged—for each received solution. The solutions produced by a node are conditionally independent of each other, given the input solutions of the node. The output node of a composite graph forwards all received solutions without alteration. To ensure integration within the system, we mandate the existence of edges from any agent’s output node directly to the composite graph’s output node. 

Our TOT agent uses a tree-search strategy across a perfect binary tree with a depth of eight. Instead of constructing a graph of 2[9] _−_ 1 nodes to represent this tree, the search is carried out through a chain of eight branching nodes. Each branching node is designed to generate two solutions from every input solution that it processes, effectively embodying the TOT strategy. 

## E.2.2. HYPER-PARAMETERS & PROMPTS 

The candidate word generation prompt and the pruning prompt are adapted from the original TOT work (Yao et al., 2023) and detailed in Table 7. A clue is defined as a partial filling of the crossword, accompanied by its intended word description and specific position on the board. 

## **E.3. HumanEval** 

## E.3.1. THE NODE OPTIMIZATION METHOD 

For node optimization, we update each node after every four new problem executions. When addressing a new problem _q_ with graph _G_ , executing _G_ ( _q_ ) produces a program, denoted by _s_ , whose effectiveness is assessed against test examples associated with _q_ . The input-output pairs of the nodes generated during the evaluation of _G_ ( _q_ ) are classified as positive if _s_ passes the tests and as negative otherwise. We limit each node to include a maximum of four demonstration examples. Let _n_ be a node in the graph associated with a computational routine _fn[p]_[, which returns Python programs, parameterized] by demonstration examples _p_ . During an optimization step of _n_ , that is, an application of _I_ as described in Section 2.4, we assess whether to retain existing demonstration examples ( _p_[1] _n_[) or to augment them with positive examples from the] four most recent problems, subsequently randomly selecting up to four unique examples from this pool (denoted as _p_[2] _n_[).] More specifically, let _Z_ be the set of the last ten inputs of node _n_ received when solving the first-seen problems. We select _p[i] n_[to update the demonstration examples of] _[ n]_[, where] _[ i]_[ = arg max] _i∈{_ 1 _,_ 2 _}_ � _z∈Z_[1] _[z]_[(] _[f] n[ p] n[i]_[(] _[z, q] z_[))][,][ 1] _z_[determines whether a] program passes the unit tests stated in _z_ , and _qz_ is the original graph input associated with _z_ . 

The utility measure for Mini Crosswords experiments is defined as the best state word accuracy, as detailed in Section 3.2. To reduce the variance in gradient estimation with the REINFORCE algorithm, we adjust the utility by subtracting a constant of 0 _._ 4. For example, a perfectly completed solution results in a utility of 0 _._ 6, while an empty solution yields a utility of _−_ 0 _._ 4. 

## E.3.2. HYPER-PARAMETERS & PROMPTS 

Table 8 shows the prompts used in our experiments following the principle of ReAct (Yao et al., 2022). 

## **E.4. GAIA** 

## E.4.1. AGENT SETTING 

We design different agents and swarms. Representative agents and swarms are visualized in Figure 9. 

## E.4.2. HYPER-PARAMETERS & PROMPTS 

We use GPT-4-Turbo for the experiments and design different node operations to solve the GAIA tasks. Table 9 and Table 10 show the prompts used in our experiments. 

20 

**GPTSwarm: Language Agents as Optimizable Graphs** 

_Table 7._ **Prompts for the Mini Crosswords Experiments.** 

|**Prompt purpose**|**Prompt**|
|---|---|
|Candidate<br>words<br>generation prompt|Let’s play a 5 x 5 mini crossword, where each word should have exactly 5 letters.<br>_{_current board status_}_<br>Unflled:<br>_{_Unflled clues_}_<br>Filled:<br>_{_flled clues_}_<br>Changed:<br>_{_Changed clues_}_<br>Suggestions:<br>_{_suggestions generated by previous Refection nodes_}_<br>Given the current status, list all possible answers for unflled or changed words, and your confdence levels (cer-<br>tain/high/medium/low), using the format ”h1. apple (medium)”. Use ”certain” cautiously and only when you are<br>100% sure this is the correct word. You can list more then one possible answer for each word.|
|Pruning prompt|Evaluate if there exists a fve letter word of some meaning that ft some letter constraints (sure/maybe/impossible).<br>Incorrect; to injure: w<br>o<br>g<br>The letter constraint is: 5 letters, letter 1 is w, letter 3 is o, letter 5 is g.<br>Some possible words that mean ”Incorrect; to injure”:<br>wrong (w r o n g): 5 letters, letter 1 is w, letter 3 is o, letter 5 is g. ft!<br>sure<br>A person with an all-consuming enthusiasm, such as for computers or anime:<br>u<br>The letter constraint is: 5 letters, letter 5 is u.<br>Some possible words that mean ”A person with an all-consuming enthusiasm, such as for computers or anime”:<br>geek (g e e k): 4 letters, not 5<br>otaku (o t a k u): 5 letters, letter 5 is u<br>sure<br>Dewy; roscid: r<br>l<br>The letter constraint is: 5 letters, letter 1 is r, letter 5 is l.<br>Some possible words that mean ”Dewy; roscid”:<br>moist (m o i s t): 5 letters, letter 1 is m, not r<br>humid (h u m i d): 5 letters, letter 1 is h, not r<br>I cannot think of any words now. Only 2 letters are constrained, it is still likely<br>maybe<br>A woodland:<br>l<br>d e The letter constraint is: 5 letters, letter 2 is l, letter 4 is d, letter 5 is e.<br>Some possible words that mean ”A woodland”:<br>forest (f o r e s t): 6 letters, not 5 woods (w o o d s): 5 letters, letter 2 is o, not l<br>grove (g r o v e): 5 letters, letter 2 is r, not l I cannot think of any words now. 3 letters are constrained, and<br>l<br>d e<br>seems a common pattern<br>maybe<br>An inn:<br>d<br>w f<br>The letter constraint is: 5 letters, letter 2 is d, letter 4 is w, letter 5 is f.<br>Some possible words that mean ”An inn”: hotel (h o t e l): 5 letters, letter 2 is o, not d<br>lodge (l o d g e): 5 letters, letter 2 is o, not d<br>I cannot think of any words now. 3 letters are constrained, and it is extremely unlikely to have a word with pattern<br>d<br>w f to mean ”An inn”<br>impossible<br>Chance; a parasitic worm; a fsh: w r a k<br>The letter constraint is: 5 letters, letter 1 is w, letter 2 is r, letter 3 is a, letter 4 is k.<br>Some possible words that mean ”Chance; a parasitic worm; a fsh”:<br>fuke (f l u k e): 5 letters, letter 1 is f, not w<br>I cannot think of any words now. 4 letters are constrained, and it is extremely unlikely to have a word with pattern w r<br>a k<br>to mean ”Chance; a parasitic worm; a fsh”<br>impossible<br>_{_clue_}_|
|Suggestion Prompt|You are playing a 5 x 5 mini crossword, where each word should have exactly 5 letters. Given the current status:<br>_{_current board status_}_<br>The target words are classifed as Impossible Words, Correct Words, and Incorrect Words.<br>---<br>Impossible Words:<br>_{_impossible clues_}_<br>Correct Words:<br>_{_correct clues_}_<br>Incorrect Words:<br>_{_incorrect clues_}_<br>Respond at most fve sentences, one sentenceper line. Do not include thephrase ”next time” inyour response.|



21 

## **GPTSwarm: Language Agents as Optimizable Graphs** 

_Table 8._ **Prompts for the Node Optimization experiments on HumanEval.** 

|**Prompt purpose**|**Prompt**|
|---|---|
|System prompt|You are an AI that only responds with only Python code.|
|CodeWriting|You will be given a function signature and its docstring by the user. Write your full implemen-<br>tation (restate the function signature). Use a Python code block to write your response. For<br>example: “‘python print(‘Hello world!’) ”’<br>_{_Demonstrations_}_<br>_{_problem statement_}_|
|CodeWriting (ReAct)|You will be given a function signature and its docstring by the user. Write your full implemen-<br>tation (restate the function signature). Use a Python code block to write your response. For<br>example: “‘python print(‘Hello world!’) ”’<br>_{_Demonstrations_}_<br>Here is an unsuccessful attempt to solve the following question:<br>Question:<br>_{_problem statement_}_<br>Attempted Solution:<br>_{_previously generated program_}_<br>Feedback:<br>_{_internal unit test results_}_<br>Rewrite the code based on the feedback and the following question:<br>_{_problem statement_}_|



## **F. Resource Requirements** 

Table 11 presents the cost, token consumption, and time requirements of representative experiments. 

## **G. Limitation and Future Work** 

In this paper, we focus on optimizing communication between agents, laying the groundwork for more extensive graph optimization in the future. For example, while current methods optimize edge connections between agents, the internal node topology of each agent is also crucial. Dynamically changing the topology may enhance task planning. Additionally, scaling up the agent framework is essential. When the number of agents exceeds 100, maintaining communication efficiency and system robustness becomes a significant challenge. 

22 

**GPTSwarm: Language Agents as Optimizable Graphs** 

_Table 9._ **Prompts for the Task-Solving experiments on GAIA (1).** 

|**Prompt purpose**|**Prompt**|
|---|---|
|System prompt|You are a general AI assistant. I will ask you a question. Report your thoughts, and fnish your<br>answer with the following template: FINAL ANSWER: [YOUR FINAL ANSWER]. YOUR<br>FINAL ANSWER should be a number OR as few words as possible OR a comma separated<br>list of numbers and/or strings. If you are asked for a number, don’t use comma to write your<br>number neither use units such as $ or percent sign unless specifed otherwise. If you are asked<br>for a string, don’t use articles, neither abbreviations (e.g. for cities), and write the digits in plain<br>text unless specifed otherwise. If you are asked for a comma separated list, apply the above<br>rules depending of whether the element to be put in the list is a number or a string.|
|DirectAnswer|_{_question_}_|
|GenerateQuery|# Information Gathering for Question Resolution<br>Evaluate if additional information is needed to answer the question.<br>If a web search or fle analysis is necessary, outline specifc clues or details to be searched for.<br>## Target Question:<br>question<br>## Clues for Investigation:<br>Identify critical clues and concepts within the question that are essential for fnding the answer.|
|WebSearch|# Web Search Task<br>## Original Question:<br>—<br>_{_question_}_<br>—<br>## Targeted Search Objective:<br>—<br>query<br>—<br>## Simplifed Search Instructions:<br>Generate three specifc search queries directly related to the original question. Each query should<br>focus on key terms from the question. Format the output as a comma-separated list. For example,<br>if the question is ’Who will be the next US president?’, your queries could be: ’US presidential<br>candidates, current US president, next US president’. Remember to format the queries as ’query1,<br>query2, query3’.|
|DistillWebSearch|## Required Information for Summary:<br>—<br>_{_query_}_<br>—<br>## Analyzed Search Results:<br>—<br>_{_results_}_<br>—<br>## Instructions for Summarization:<br>1. Review the provided search results and identify the most relevant information related to the<br>question and query.<br>2. Extract and highlight the key fndings, facts, or data points from these results.<br>3. Organize the summarized information in a coherent and logical manner.<br>4. Ensure the summary is concise and directly addresses the query, avoiding extraneous details.<br>5. If the information from web search is useless, directly answer: ¨No useful information from<br>WebSearch¨.|
|FileAnalyse|# File Analysis Task<br>## Information Extraction Objective:<br>—<br>_{_query_}_<br>—<br>## File Under Analysis<br>—<br>_{_fle_}_<br>—<br>## Instructions:<br>1. Identify the key sections in the fle relevant to the query.<br>2. Extract and summarize the necessary information from these sections.<br>3. Ensure the response is focused and directly addresses the query.<br>Example: ’Identify the main theme in the text.’”<br>|



23 

**GPTSwarm: Language Agents as Optimizable Graphs** 

_Table 10._ **Prompts for the Task-Solving experiments on GAIA (2).** 

|**Prompt purpose**|**Prompt**|
|---|---|
|CombinedAnswer|Reference information for FileAnalysis:<br>—<br>_{_fle<br>~~a~~nalysis_}_<br>—<br>Reference information for Websearch:<br>—_{_web<br>~~s~~earch_}_—<br>Provide a specifc answer. For questions with known answers, ensure to provide accurate and<br>factual responses. Avoid vague responses or statements like ’unable to...’ that don’t contribute<br>to a defnitive answer. For example: if a question asks ’who will be the president of America’,<br>and the answer is currently unknown, you could suggest possibilities like ’Donald Trump’, or<br>’Biden’. However, if the answer is known, provide the correct information.”|
|FinalDecision<br>(Self-Consistency)|# Self-Consistency Evaluation Task<br>## Question for Review:<br>—<br>_{_question_}_<br>—<br>## Reviewable Answers:<br>—<br>_{_formatted<br>~~a~~nswers_}_<br>—<br>## Instructions for Selection:<br>1. Read each answer and assess how it addresses the question.<br>2. Compare the answers for their adherence to the given question’s criteria and logical coherence.<br>3. Identify the answer that best aligns with the question’s requirements and is the most logically<br>consistent.<br>4. Ignore the candidate answers if they do not give a direct answer, for example, using ’unable to<br>...’, ’as an AI ...’.<br>5. Copy the most suitable answer as it is, without modifcation, to maintain its original form.<br>6. Adhere to the constraints: _{_constraint_}_.<br>Note: If no answer fully meets the criteria, choose and copy the one that is closest to the<br>requirements.|
|FinalDecision<br>(Choose “Best”)|## Question:<br>—<br>_{_question_}_<br>—<br>## Candidate Answers for Evaluation:<br>—<br>_{_formatted<br>~~a~~nswers_}_<br>—<br>## Evaluation Instructions:<br>1. Examine the question closely to understand its requirements.<br>2. Read each candidate answer thoroughly and assess its relevance and accuracy about the<br>question.<br>3. Choose the answer that most accurately and completely addresses the question.<br>4. Ignore the candidate answers if they do not give a direct answer, for example, using ’unable to<br>...’, ’as an AI ...’.<br>”5. Copy the chosen answer exactly as it is presented, maintaining its original format.<br>6. Adhere to the constraints:<br>_{_constraint_}_.<br>Note: If none of the answers fully meet the question’s criteria, select the one closest to fulflling<br>them.|



24 

**GPTSwarm: Language Agents as Optimizable Graphs** 

_Table 11._ **Cost, token consumption, and time requirements.** The following experiments are performed with gpt-3.5-turbo-1106 if marked with GPT-3.5T, or gpt-4-1106-preview otherwise. 

|**Experiment**|**Cost (USD)**<br>**# Prompt Tokens**<br>**# Completion Tokens**<br>**Time (h)**|
|---|---|
|TOT - Mini Crosswords|65.61<br>1,515,826<br>2,013,511<br>8.5|
|GPTSwarm - Mini Crosswords Edge-Opt (GPT-3.5T)|77.42<br>50,394,028<br>13,511,265<br>2.82|
|GPTSwarm - Mini Crosswords Edge-Opt-Eval (GPT-3.5T)|9.89<br>6,448,660<br>1,718,613<br>0.73|
|GPTSwarm - Mini Crosswords Edge-Opt-Eval|377.54<br>13,137,160<br>8,205,522<br>5.56|
|GPTSwarm - Mini Crosswords Node-Opt (GPT-3.5T)|11.22<br>7,468,797<br>1,876,246<br>0.83|
|GPTSwarm - Mini Crosswords Node-Opt-Eval (GPT-3.5T)|28.18<br>22,791,158<br>2,693,575<br>0.91|
|GPTSwarm - HumanEval w/o Opt|1.61<br>59,646<br>33,951<br>0.68|
|GPTSwarm - HumanEval w/ Opt|28.46<br>2,298,140<br>182,594<br>1.49|
|GPTSwarm - GAIA (Level 1) - Agent(TOT)|2.21<br>123,801<br>32,599<br>1.05|
|LLM-Debate - MMLU (3A3T)|32.8<br>1,689,960<br>530,005<br>8.36|
|DyLAN - MMLU optimization (3A3T)|105.93<br>5,671,276<br>1,640,566<br>25.4|
|DyLAN - MMLU inference (3A3T)|14.99<br>628,009<br>290,472<br>4.75|
|GPTSwarm - MMLU optimization (3A3T)|5.32<br>361,812<br>56,770<br>0.9|
|GPTSwarm - MMLU inference (3A3T)|1.82<br>113,233<br>22,923<br>0.31|



25 

