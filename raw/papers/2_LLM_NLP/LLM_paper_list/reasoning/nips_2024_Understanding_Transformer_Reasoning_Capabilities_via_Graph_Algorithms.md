
![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_Understanding_Transformer_Reasoning_Capabilities_via_Graph_Algorithms_images/nips_2024_Understanding_Transformer_Reasoning_Capabilities_via_Graph_Algorithms.pdf-0001-01.png)


# Understanding Transformer Reasoning Capabilities via Graph Algorithms 

Clayton Sanford[1,2] , Bahare Fatemi[∗][1] , Ethan Hall[3] , Anton Tsitsulin[1] , Mehran Kazemi[4] , Jonathan Halcrow[1] , Bryan Perozzi[1] , and Vahab Mirrokni[1] 1Google Research, 2Columbia University,3Google, 4Google DeepMind 

## Abstract 

Which transformer scaling regimes are able to perfectly solve different classes of algorithmic problems? While tremendous empirical advances have been attained by transformer-based neural networks, a theoretical understanding of their algorithmic reasoning capabilities in realistic parameter regimes is lacking. We investigate this question in terms of the network’s depth, width, and number of extra tokens for algorithm execution. Our novel representational hierarchy separates 9 algorithmic reasoning problems into classes solvable by transformers in different realistic parameter scaling regimes. We prove that logarithmic depth is necessary and sufficient for tasks like graph connectivity, while single-layer transformers with small embedding dimensions can solve contextual retrieval tasks. We also support our theoretical analysis with ample empirical evidence using the GraphQA benchmark. These results show that transformers excel at many graph reasoning tasks, even outperforming specialized graph neural networks. 

## 1 Introduction 

The transformer neural network architecture, which was initially introduced for neural machine translation [5, 75], quickly became the standard neural network architecture across many fields, powering recent breakthroughs in language modeling [18, 64, 11, 74, 2, 73, 65], computer vision [19, 47], and natural sciences [34, 51]. Across fields, transformers have superseded other architectures, surpassing them in downstream performance while maintaining reasonable computational footprint. 

How can we analyze reasoning capabilities of neural networks? One approach is to study algorithms executable with their internal representations. Neural algorithmic reasoning [28, 87, 32, 35, 76] is a field of research dedicated to exploring such capabilities. Algorithmic execution is desirable because models use it to generalize out-of-distribution [90, 44] and scale to larger problem sizes [91]. 

In this work, we focus on classes of transformers solving algorithmic reasoning problems on graphs. Why graph problems in particular? Recent research by Besta et al. [9] suggests that graphs are an ideal abstraction of complex reasoning with dependencies, including chain-of-thought [80] and its generalizations [84, 16, 39, 38]. Furthermore, we investigate graph reasoning in order to evaluate transformer capabilities compared to specialized models that explicitly capture the structure of data. Graph neural networks (GNNs) [70, 42, 26, 6, 14] offer strong baselines for algorithmic reasoning [77, 12] and comprehensive theoretical frameworks for their capabilities and limitations [49, 82]. The complexity of graph reasoning tasks varies greatly; some are easily solved by non-graph architectures and others require sophistication beyond standard GNNs [88]. This makes graph tasks a compelling testbed for both theoretical and empirical evaluations. 

While transformer-based models are commonly optimized for downstream performance [36, 30, 62], theoretical investigations of transformer capabilities in realistic parameter regimes have been limited. 

> ∗Correspondence: clayton.h.sanford@gmail.com or baharef@google.com. 

Preprint. Under review. 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_Understanding_Transformer_Reasoning_Capabilities_via_Graph_Algorithms_images/nips_2024_Understanding_Transformer_Reasoning_Capabilities_via_Graph_Algorithms.pdf-0002-00.png)


**----- Start of picture text -----**<br>
Input X Transformer f Target f (X)<br>v1 ∅<br>v2 Embedding ∅<br>v3 ∅<br>Graph G<br>v4 Self-Attention ∅<br>v4 v5 ∅<br>v2 v6 MLP ∅<br>v3 v5 v7 ∅<br>v1 v8 Self-Attention ∅<br>v6 v8 v1, v2 ∅<br>v2, v3 MLP ∅<br>v7<br>v3, v4 ∅<br>Task: Are v2 and v4 connected? v6, v7 Self-Attention ∅<br>v6, v8 ... ∅<br>v7, v8 ∅<br>Task tokens<br>v2, v4 Yes<br>Vertex tokens<br>Edge tokens<br>**----- End of picture text -----**<br>


Figure 1: The graph encoding scheme employed in our theoretical and empirical analysis that presents a graph reasoning task (e.g. connectivity) as a tokenized input to a standard transformer model. 

We analyze the capabilities of standard transformer models to solve graph reasoning tasks by employing a graph tokenization scheme similar to [40]; see Figure 1 for a graph encoding example. We generalize the insights of [69]—which established that graph connectivity can be computed in a more depth-efficient manner with transformers than GNNs—to broader families of graph reasoning tasks. We also study the representational impacts of blank tokens, which can be thought of as a theoretically-tractable version of either chain-of-thought prompting [80], scratch space [59], or pause/filler tokens [27, 63]. 

We conduct an extensive study of graph reasoning problems with transformers from both theoretical and empirical perspectives. We summarize our principal contributions as follows: 

1. We introduce a novel representational hierarchy of graph reasoning tasks that formalizes reasoning capabilities of transformers in several realistic parameter scaling regimes. This includes two graph reasoning task classes—which we refer to as the parallelizable and search tasks and include well-known algorithmic tasks like connectivity and shortest path respectively. 

2. We prove that logarithmic-depth transformers are necessary and sufficient to solve parallelizable tasks in a highly parameter-efficient manner; similar constructions for search tasks employ much larger networks. We distinguish both of these classes from an easier family of problems— retrieval tasks, such as node count and edge existence—by showing that retrieval tasks can be efficiently computed by single-layer transformers, while parallelizable and search tasks cannot. 

3. We empirically validate our representational contrast by showing that transformers outperform GNNs on tasks that require the analysis of long-range dependencies, including parallelizable tasks like connectivity. Our experiments suggest that GNNs perform better on simpler tasks that require only local analysis of neighboring nodes in low sample-complexity regimes, benefiting from their well-aligned inductive bias. 

## 2 Related work 

Fundamental capabilities of transformers. Early representational research [86, 60, 79] established the universality of transformers by simulating Turing machines step-by-step, albeit in an unrealistic polynomial-depth scaling regime. These universality results were extended to bounded-depth transformers with chain-of-thought tokens [54, 50], but with a persistent gap between positive and negative results. A more fine-grained theoretical approach established the limitations of boundedsize transformers by relating them to threshold circuits, implying that L-complete tasks like graph connectivity are unsolvable by constant-depth transformers [53, 52, 55]. However, these circuit 

2 

complexity reductions are not bidirectional and their positive results pertaining to classes of regular languages [29] reveal little about when tasks like connectivity can be expressed. 

The closest analytical framework to ours characterizes transformers as distributed computing protocols by quantifying the informational bandwidth of their self-attention units. This line of work sharply separates transformers from other architectures and provides width and depth separations [68, 69]. The ability of transformers to simulate finite-state automata in logarithmic depth provides a further roadmap for how transformers efficiently leverage parallel computation [46]. 

Empirical analysis of transformer reasoning capabilities. Multiple empirical investigations explore the algorithmic capabilities of transformer-based models [89, 45, 44]. Graph reasoning problems have been used to evaluate capabilities of transformer-based LLMs [22, 78, 85, 72, 31] including works with graph-specific GNN adapters [13, 61]. In the empirical part of our study, we use the GraphQA dataset [22]—which was evaluated initially on LLM prompting appraoches and later on GNN-augmented prompts [60]—to validate our complexity hierarchy and compare transformers with GNNs and prompt-based LLMs on a selection of graph reasoning tasks. 

Transformers and graph neural networks. GNNs provide a useful benchmark for model expressivity on graph inputs. For instance, the inability of GNNs to efficiently solve “global structure” tasks like subgraph connectivity made evident by a connection to the CONGEST distributed computing model [49]. A further connection [57] to the Weisfeiler-Lehman (WL) isomorphism heuristic [81] captures the inability of feature-less GNNs to distinguish certain graph instances [82]. See Appendix D for a further discussion of the representational limitations of GNNs. 

Graph transformers [20, 66, 56] integrate the transformer architecture with graph-structured data. While GNNs can simulate the WL test [1], transformers can simulate GNNs and exceed their WL limitations [88]; they are strictly more expressive than 2-WL [40]. Despite our focus on standard transformers, our empirical results in Section 4 include comparisons to a wide range of GNNs. 

## 3 Hardness taxonomy of transformer graph reasoning tasks 

We provide our core result: a rigorous quantification of the hardness of graph reasoning tasks for transformer-based models. While graph reasoning tasks, like other algorithmic problems, can be categorized into well-known computational and circuit complexity classes (e.g. TC[0] , L, NL, NP, NP), the relationship between membership in these classes and the hardness of solving a task with a parameter-efficient neural network is not immediately obvious. Our hierarchy bridges the gap between these worst-case computational classes and the representational capabilities of boundedsize transformers of different parameter scaling regimes. These regimes include transformers whose depth L scales with the input sequence length N ; this contrasts with most theoretical results, which study on the constant-depth regime. 

Our positive and negative theoretical results employ the transformer model of [69], which is presented in Appendix A.1 and assumes that the embedding dimension m grows less rapidly than than the sequence length N and that the multi-layer perceptrons (MLPs) are arbitrary functions. The result is a model of a transformer as a bounded-capacity communication protocol. In this model, arbitrary functions of each individual embedding vector can be computed, but the interactions between these vectors are restricted by the low-rank of the attention matrix. The relevance of this model is motivated by the rapid scaling in the context lengths of modern transformers in recent years and the high ratio of MLP parameter count to the embedding dimension each operates on. 

This model also permits the inclusion of blank pause token inputs of [27], which provide additional computational power to the transformer model by extending the computational “tape” without introducing new information about the input. 

These results divide the tasks that are we empirically investigate in Section 4 into three difficulty families based on the hardness of solving the task with parallel computation. 

1. Retrieval tasks—including node count, edge count, edge existence, and node degree—are tasks that can intuitively be solved by a single lookup step or global aggregation. These are the easiest tasks in our framework, and we show in Section 3.3 that these retrieval tasks can be computed by a single-layer transformer with small embedding dimension. (In contrast, all other examined tasks cannot be solved in that regime.) 

3 

|Task class|Example tasks|Complexity|
|---|---|---|
|Retrieval (§3.3)|Node count|D1|
|L= 1|Edge count|D1|
|m=O(logN)|Edge existence|D1|
||Node degree|D1|
|Parallelizable (§3.1)|Connectivity|LD|
|L=O(logN)|Cycle check|LDP∩LDW|
|m=O(N ǫ)|Bipartiteness|LDP∩LDW|
|Search (§3.2)|Shortest path|LDW|
|L=O(logN)|Diameter|LDW|
|m=O(N 1/2+ǫ)|||




![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_Understanding_Transformer_Reasoning_Capabilities_via_Graph_Algorithms_images/nips_2024_Understanding_Transformer_Reasoning_Capabilities_via_Graph_Algorithms.pdf-0004-01.png)


**----- Start of picture text -----**<br>
Parallelizable<br>Retrieval (§3.3) Node count D1<br>L = 1 Edge count D1<br>LDP LD LDW m = O(log N ) Edge existence D1<br>Search Node degree D1<br>Parallelizable (§3.1) Connectivity LD<br>D1 L = O(log N ) Cycle check LDP∩ LDW<br>Retrieval m = O(N [ǫ] ) Bipartiteness LDP∩ LDW<br>Search (§3.2) Shortest path LDW<br>L = O(log N ) Diameter LDW<br>m = O(N [1][/][2+][ǫ] )<br>(a) The complexity hierarchy (b) Example tasks and their complexity.<br>**----- End of picture text -----**<br>


Figure 2: A summary of the theoretical hierarchy of Section 3 that visualizes which type of graph reasoning tasks can be solved in which transformer scaling regime (Depth1 (D1), LogDepth (LD), LogDepthWide (LDW) and LogDepthPause (LDP)). 

2. Parallelizable tasks—including connectivity, connected nodes, and cycle check from the experimental results and a host of other graph reasoning tasks, such as bipartiteness, planarity, and minimum spanning forest—are non-trivial tasks that can be solved efficiently in a parallel computation setting. Section 3.1 establishes that these tasks can be solved by bounded-size transformers with logarithmic depth. 

3. Search tasks—including shortest path and other tasks like diameter and directed reachability— comprise a harder family of tasks that are less easily solved by parallel algorithms. In Section 3.2, we prove that these tasks belong in an equivalence class and exhibit large-model scaling regimes where they can be computed. 

The representational hardness of these classes is quantified by several results that determine whether transformers that obey different parameter scaling rules can compute them. We define the following scaling regimes for the depth L, embedding dimension m, and number of “pause” tokens N[′] of a family of transformers as a function of the size of the input graph, N = O(|V | + |E|). 

- Depth1 (D1): Single-layer multi-headed transformers with small embedding dimension m = O(log N ) without any pause tokens. 

- LogDepth (LD): Transformers with depth L = O(log N ), embedding dimension m = O(N[ǫ] ) for any fixed ǫ > 0, and no pause tokens. 

- LogDepthPause (LDP): Transformers with the same depth and width constraints as LogDepth, with at most N[′] = poly(N ) blank “pause” tokens appended to the input sequence. 

- LogDepthWide (LDW): Transformers with depth L = O(log N ), embedding dimension m = O(N[1][/][2+][ǫ] ), and no pause tokens. 

Our positive and negative results that relate scaling regimes and graph reasoning tasks are displayed in Figure 2. We present high-level result summaries in the following sections with proofs in Appendix. 

The most technically significant results concern LogDepth models and are provided in Appendix B. These bounds are consequences of Theorem 1, an improved analysis of the relationship between transformers and the massively parallel computation (MPC) distributed computing model of [37][2] . This connection between MPC and transformers is a sharp improvement of a similar result by [69]. Theorem 1 (Simplified version of Theorem 8). For constant δ, ǫ > 0, any R-round MPC protocol with N machines with O(N[δ] ) bits of local memory each can be simulated by a transformer of depth L = O(R) and embedding dimension m = O(N[δ][+][ǫ] ). 

> 2MPC is a theoretical model of the MapReduce computational paradigm that distributed computation among a large number of machines with restricted local memory size and communication bandwidth. We formally introduce the MPC model in Appendix B.1. 

4 

Results pertaining to Depth1 are stated in detail and proved in Appendix C. In addition, we discuss the triangle counting task (and the more general clique counting task) in Section 3.4, where we show a distinct result for much smaller depth (L = O(log log N )) that includes pause tokens. 

## 3.1 Parallelizable tasks and LogDepth transformers 

We define the family of parallelizable tasks to consist of graph reasoning tasks that are L-complete and are equivalent to graph connectivity in O(1)-rounds, as proved by [58]. This family includes (but is not limited to): graph connectivity, minimum spanning forest, cycle detection, st-connectivity, number of connected components, graph bipartiteness, planarity testing, and one-cycle vs two-cycles testing. While graph connectivity and minimum spanning forest were shown to be computable by logarithmic depth transformers with small polynomial width (LogDepth) by previous work [69], this poses broader questions: Are all parallelizable graph reasoning tasks computable by a transformer of logarithmic depth and small embedding dimension? And do sub-logarithmic-depth transformers exist that solve any parallelizable tasks? 

We first show that all parallelizable tasks can be solved in two logarithmic-depth scaling settings. 

Theorem 2. For any parallelizable task, there exists transformers in LogDepthPause and LogDepthPause that solve the task. 

Theorem 2 is stated formally as Theorem 18 and proved in Appendix B.2.1. Both components of the theorem are a consequence of a novel relationship between the MPC model and transformers (Theorem 1) and the analysis of MPC protocols for graph reasoning tasks by [58]. The LogDepthPause result is the direct implication of an O(1)-round MPC equivalence (Theorem 9) between all parallelizable tasks and an MPC protocol that solves the connectivity task (Theorem 11). The LogDepthWide bound is a consequence of Theorem 15, which shows that all languages in NC[2] (including all languages in L and NL) can be evaluated by an MPC protocol with O(log N ) rounds and with local memory O(N[1][/][2+][ǫ] ) [58]. 

We further prove the conditional optimality of logarithmic depth. 

Theorem 3. Conditional on Conjecture 13, any transformer that solves some parallelizable task with width mH = O(N[ǫ] ) and pause tokens N[′] = poly(N ) must have depth L = Ω(log N ). 

This result (stated formally as Theorem 19 is a combination of the conditional depth lower bound on graph connectivity by [69] and the O(1)-round equivalence between all parallel tasks. 

## 3.2 Search tasks 

Search tasks are similarly defined to be those that are NL-complete and equivalent to shortest path in O(1)-rounds of MPC and include shortest path, strong connectivity, st-reachability, radius, diameter, and median. Like before, the O(1)-round MPC equivalence translates to an O(1)-depth equivalence in transformers. We give a similar positive result for LogDepthWide transformers; whether these tasks can be solved by LogDepthPause transformers is unknown. 

Theorem 4. For any search task, there exists a transformer in LogDepthWide that sovles the task. 

This theorem, which is restated in Appendix B.2.2 as Theorem 21, is also an immediate consequence of Theorem 15. 

While the minimum depth of a transformer with small embedding dimension that solves a search task is not identified, we prove that the minimum depth needed to solve some all search task is approximately equivalent in Theorem 22. 

## 3.3 Retrieval tasks and Depth1 transformers 

Graph tasks whose algorithms consist of a single look-up or aggregation step can be efficiently solved by single-layer transformers. This result assumes that the graph G = (V , E) is encoded as some input sequence X of length N = O(|V | + |E|) that expresses each edge and vertex a single token. This encoding scheme is detailed in Appendix C. 

Theorem 5. For any retrieval task (including node count, edge count, edge existence, and node degree) there exists a transformer in Depth1 that solves the task. 

5 

We formalize this statement in Theorem 36 and prove it their in Appendix C.1. These rely on proving the existence of a useful input MLP φ that precomputes embeddings with useful structure for all retrieval tasks. 

In contrast, we show that a collection of parallelizable and search tasks cannot be efficiently solved by transformers in Depth1. 

Theorem 6. Any single-layer transformer that solves the graph connectivity, shortest path, or cycle detection task has width satisfying mH = Ω([˜] N ). 

The proof of the formal counterpart of this statement (Theorem 38) appears in Appendix C.2 and is a consequence of a standard communication complexity argument. A more generalized result than Theorem 6 was proved by [55], which establishes that all problems outside of NC[1] —which include all L-complete and NL-complete languages, and hence, all parallelizable and search tasks—cannot be solved by constant-depth transformers with polynomial width because they cannot be computed by TC[0] (constant-depth threshold circuits). We nonetheless include this theorem to demonstrate a clean lower bound that applies to very simple input graph instances. 

## 3.4 Triangle counting 

We finally construct depth-efficient transformers for triangle counting due to the MPC algorithms of [10]. Unlike previous positive results, which applied uniformly across all graphs instances of bounded size, the complexity of the corresponding transformers for triangle counting is a function of the arboricity[3] of the input graph. When the arboricity grows sub-polynomially with N —as is the case for bounded-degree graphs—no pause tokens are necessary. Unlike the parallelizable and search classes of problems, strictly sub-logarithmic depth is attainable with pause tokens, even for worst-case graphs. 

Theorem 7. There exists a transformer that computes the number of triangles in any input graph of arboricity α and has embedding dimension m = O(N[ǫ] ), depth L = O(log log N ), and pause tokens 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_Understanding_Transformer_Reasoning_Capabilities_via_Graph_Algorithms_images/nips_2024_Understanding_Transformer_Reasoning_Capabilities_via_Graph_Algorithms.pdf-0006-07.png)


This result is a special case of Theorem 23, a more general result about clique counting that appears in Appendix B.2.3. 

Theoretical conclusions. These results provide a tight characterization of the the reasoning capabilities of transformers whose depth, width, and input padding conform to different scaling regimes. They strengthen the established connection between transformers and massively parallel computation (MPC) [69] and generalize the resulting representational bounds to broader categories of graph tasks. We conclude that the logarithmic-depth regime is apt for for considering tasks in L and NL, which had previous illuminated the limitations of transformers with constant depth and a limited number of chain-of-thought tokens [54]. While expressivity does not imply learnability, these theoretical benchmarks sharply characterize the fundamental limitations of transformers and coincide with experimental results conveyed in the subsequent section. 

## 4 Empirical graph reasoning capabilities 

We further illuminate the reasoning capabilities of transformers by conducting an empirical investigation of the abilities of a variety of neural architecture and training settings to learn graph algorithmic tasks. We use the GraphQA benchmark tasks [22] for our experiments. We evaluate standard autoregressive transformers—both small models (at most 60M parameters) trained from scratch and fine-tuned (FT) T5-11B model (with 11B parameters) [64]. For the fine-tuned models, we explore task-specific fine-tuning—and contrast those results with graph neural networks (GNNs) and prompting-based methods on pre-trained LLMs. 

These experimental results validate key tenets of our theoretical results and demonstrate the utility of transformers’ algorithmic reasoning capabilities. Our principal empirical conclusions are as follows: 

3The arboricity is the minimum number of spanning forests needed to cover the graph, which grows at most linearly with the degree of the graph. 

6 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_Understanding_Transformer_Reasoning_Capabilities_via_Graph_Algorithms_images/nips_2024_Understanding_Transformer_Reasoning_Capabilities_via_Graph_Algorithms.pdf-0007-00.png)


**----- Start of picture text -----**<br>
# of training samples 1<br>Model 1K 100K<br>0.95<br>GCN [42] 83.8 83.8<br>MPNN [26] 94.0 94.4 Train<br>GIN [82] 93.8 94.0 0.9 Test<br>60M transformer 92.9 98.0<br>10 [2] 10 [3] 10 [4] 10 [5]<br>11B transformer (FT) 98.4 —<br>Number of Training Samples (log scale)<br>(a) Connectivity classification accuracy for trained (b) Connectivity classification accuracy of 60M<br>transformers and GNNs. transformers by training set size.<br>Accuracy<br>**----- End of picture text -----**<br>


Figure 3: Accuracy of a variety of trained transformers and GNNs on the connectivity task. 

1. Transformers excel at global reasoning tasks. Transformers outperform GNNs on tasks that require efficiently aggregating information about distant nodes in a graph, such as connectivity and shortest path. 

2. GNNs uncover local graph structure with few samples. While transformers are capable of efficiently expressing all graph learning tasks under investigating, the structural limitations of GNNs provide them with favorable inductive biases for intrinsically local tasks, such as cycle check and node degree, and permit them to outperform transformers in a low-sample regime. 

3. Trained transformers outperform LLM prompting. Transformers trained to explicitly solve graph reasoning tasks consistently attain greater accuracy across tasks than a variety of prompting strategies applied to more recent larger LMs. 

A comprehensive evaluation of each GraphQA task on every training setting appears in Appendix E, in addition details about transformer training, the GraphQA benchmark, and alternative GNN and prompting approaches. 

## 4.1 Transformers excel at global reasoning tasks 

As indicated in Section 3, graph reasoning algorithms can be categorized based on the extent to which they entail aggregating “local” information about nodes and their immediate neighbors or modeling “global” connections between nodes separated by a long distances. This section investigates the following question about transformers and long-distance reasoning: 

When do transformers outperform GNNs on tasks that require global reasoning? 

We consider two tasks that require reasoning across long distances in a graph instance: evaluating connectivity and computing the shortest path between a pair of nodes. Neither of these tasks can be solved by only investigating the neighbors of the source and sink node, which therefore implies that some analysis of global graph structure is necessary. 

Figure 3a displays the accuracy of a variety of trained transformers and GNNs on the # of training samples connectivity task contrastsing the performance Model 1K 100K of all such models when trained on 1,000 and 100,000 graph connectivity instances. In GCN [42] 50.2 55.0 the most restricted sample complexity regime, MPNN [26] 66.8 72.6 trained GNNs are consistently more accurate GIN [82] 54.0 58.6 than the small transformer; however, increas60M transformer 57.4 97.1 ing the number of training samples yields a 11B transformer (FT) 92.8 — far more substantial improvement in the performance of the small transformer, which outTable 1: Transformers vs GNNs on shortest path: performs all GNNs trained on 100,000 samples. Fine-tuned large transformers outperform other Notably, the pre-trained transformer, fine-tuned transformers and GNNs, even the alternatives are on just 1000 training instances, nearly solves trained on much larger training sets. the connectivity task. This suggests significant enhancements due to the larger model size and the data-rich fine-tuning phase. Figure 3b plots the 

Table 1: Transformers vs GNNs on shortest path: Fine-tuned large transformers outperform other transformers and GNNs, even the alternatives are trained on much larger training sets. 

7 

training and test error of ten small transformers trained on connectivity datasets of increasing size and reveals a sharp and continual improvement in accuracy. The fine-tuned T5 transformer has similar accuracy to the most sample-rich small transformer and exceeds that of all GNNs. 

On the other hand, Table 1 demonstrates that the MPNN GNN models outperform small transformers when trained to compute the shortest path, even on larger training datasets. However, the fine-tuned T5 model has far higher accuracy than all alternatives, even when trained only 1000 samples. 

Theoretical interpretation: Because connectivity is the prototypical example of a task in the parallelizable class and can thus be efficiently implemented by LogDepth transformers with very small width (Theorem 2), the fact that small transformers succeed in solving nearly all connectivity instances is illuminating but not surprising. In contrast, message-passing GNNs are unable to solve connectivity in a similarly depth-efficient manner due to fundamental capacity limitations.[4] Shortest path belongs to the search class of graph reasoning tasks and is NL-complete. Theorem 4 implies that shortest path is computable by LogDepthWide transformers, which are likely to require very large embedding dimension to be learnable by finite samples. This task can only be computed in a depth- and parameter-efficient manner if a variety of search tasks, including all-pairs shortestpath and diameter, are as well (Theorem 22). The fact that only the pre-trained model has nearly optimal performance on shortest path reinforces the theoretical intuition that solving shortest path requires a very large number of model parameters. 

## 4.2 GNNs uncover local graph structure with few samples 

While small transformers outperform GNNs on graph reasoning algorithms that entail analysis of long-range graph structure, their empirical successes are not uniform. Here, we investigate the following question: 

When do GNNs outperform transformers on graph reasoning tasks? 

Figure 3b and Table 1 demonstrate that GNNs outperform small transformers in the low-sample regime, despite the sufficient expressivity of transformers. This gap in performance, which is reinforced for the node degree and cycle check tasks in Table 2, suggests that GNNs have a beneficial inductive bias for learning graph reasoning tasks that can be solved by attending exclusively to local heuristics. 

||Node|Degree|Cycle|Check|
|---|---|---|---|---|
|Model<br>GCN [42]<br>MPNN [26]<br>GIN [82]|1K<br>9.8<br>99.4<br>36.2|100K<br>9.4<br>99.8<br>37.8|1K<br>83.2<br>99.0<br>98.8|100K<br>83.2<br>100.0<br>83.2|
|60M transformer<br>11B transformer (FT)|31.6<br>68.8|91.7<br>—|97.1<br>98.0|98.0<br>—|



Table 2: Transformers vs GNNs on cycle check and node degree: GNNs are favorably biased for local structure. 

Just as the bounded kernel size of convolutional neural networks (CNNs) enables the sampleefficient learning of relevant textures and gradients for image analysis, message-passing GNNs are unable to send messages instantaneously across multiple edges, which simplifies a search of the space of “one-hop” graph algorithms and represents a positive inductive bias. In contrast, the ability of transformers to send information between any pair of input tokens—and the alternative inductive biases suggested by the input positional encoding—likely induces a steeper sample complexity to learn node degree. 

Theoretical interpretation: While model expressivity is necessary for learnability, it is not sufficient. The locality constraints of message-passing GNNs likely provides a favorable inductive bias for learning tasks like node degree with an exclusively on local structure that makes learning these tasks possible in a sample-efficient manner. While cycle check is more representationally difficult for GNNs than transformers in the worst case (see Appendix A.2), the random graphs sampled for the GraphQA benchmark have very small cycles (Figure 7) and do not resemble the large-diameter worst-case instance. 

> 4See Appendix A.2 for a discussion of prior work on theoretical relationships between GNNs and both the Weisfeiler-Leman graph isomorphism test [82] and the CONGEST distributed computing model and their implication that GNNs cannot solve connectivity in a parameter-efficient manner. 

8 

||||Retrieval tasks|Retrieval tasks||Parallelizable Tasks|Parallelizable Tasks|Search Tasks|Subgraph Counting|
|---|---|---|---|---|---|---|---|---|---|
||Method|Node count|Edge count Edge existence||Node degree|Connectivity Cycle check||Shortest path|Triangle counting|
||ZERO-SHOT[22]|21.7|12.4|44.5|14.0|84.9|76.0|11.5|1.5|
|Prompting|ZERO-COT[22]<br>FEW-SHOT[22]<br>COT[22]|14.6<br>25.3<br>27.6|9.4<br>12.0<br>12.8|33.5<br>36.8<br>42.8|10.4<br>17.4<br>29.2|73.5<br>79.4<br>45.2|32.3<br>37.4<br>58.0|33.6<br>22.7<br>38.6|12.7<br>3.0<br>8.1|
||COT-BAG[22]|26.9|12.5|37.3|28.0|45.2|52.1|40.4|8.1|
||60M transformer-1K|100.0|100.0|67.6|31.5|92.9|97.1|57.4|33.4|
|Ours|60M transformer-100K|100.0|100.0|96.1|91.7|98.0|98.0|97.2|40.5|
||11B transformer (FT)-1K|100.0|45.0|100.0|68.8|98.4|98.0|92.8|26.0|



Table 3: Comparison of GraphQA task accuracies of transformers explicitly trained for graph reasoning and LLMs with a variety of prompting strategies. 

## 4.3 Trained transformers outperform LLM prompting 

Large language models (LLMs) are regularly evaluated by their reasoning abilities, and it remains an open research question to determine what kinds of training data best teaches models to solve logical problems. We investigate the extent to which LLMs can already perform graph reasoning tasks without being trained explicitly to do so. 

Do transformers trained explicitly to solve graph reasoning tasks outperform prompt-tuning approaches on much larger LLMs? 

In Table 3, we contrast the capabilities of trained transformer models with several promptbased approaches to querying LLMs. Task-specific transformers—including the fine-tuned 11B transformers—consistently dominated the prompt-based approaches, despite the vast difference in parameter count and the almost certain presence of graph reasoning in the LLM’s corpus. 

Theoretical interpretation: While the representational capabilities of LLMs to solve reasoning tasks is much greater than that of small transformers, this performance gap suggests that their effective reasoning capacity is much weaker and that it may be improved by a richer training corpus that includes synthetic tasks. 

Finally, we observe that the near-perfect performanceof trained transformers on the node count, edge count, and edge existence is consistent with the representational easy of those tasks, as suggested by the existence of efficient Depth1 transformer implementations. 

## 5 Conclusion 

This paper provides a comprehensive evaluation of transformer models’ graph reasoning capabilities, shedding light on their effectiveness across diverse graph reasoning tasks. By introducing a novel representational hierarchy, the study distinguishes between retrieval, parallelizable, and search reasoning tasks and offers insights into the performance of transformers at varying levels of granularity. The empirical investigation reveals that transformers exhibit strong performance in graph-based reasoning problems, often matching or surpassing specialized graph models. Furthermore, the study highlights transformers’ exceptional ability to capture global graph patterns effectively, showcasing their capability in understanding long-range dependencies, a critical factor in solving tasks involving global graph structures. Overall, this work crystallizes precise representational trade-offs that reflect the fundamental reasoning capabilities of transformers and demonstrates that the tasks used to quantify those capabilities are indeed learnable in a sample-efficient and parameter-efficient manner. 

While the hierarchy introduced by this work effectively separates graph algorithmic tasks into distinct equivalence classes with significant implications for their computability by transformers, several questions remain for future research. We focused on graph reasoning tasks due to their relevance to the broader context of transformers, GNNs, and parallel algorithms. However, the complexity classes presented here could potentially be extended to a wider range of algorithmic problems. While our assumption of unbounded-size MLPs provides strong lower bounds, further research into whether parallelizable tasks can be represented by transformers with bounded-size MLP units would complement this existing work. Broader experimental results that empirically evaluate the scaling laws would more directly assess the relevance of representational theoretical results to learnability. 

9 

## References 

- [1] Anders Aamand, Justin Y. Chen, Piotr Indyk, Shyam Narayanan, Ronitt Rubinfeld, Nicholas Schiefer, Sandeep Silwal, and Tal Wagner. Exponentially improving the complexity of simulating the weisfeiler-lehman test with graph neural networks, 2022. Cited on page 3. 

- [2] Josh Achiam, Steven Adler, Sandhini Agarwal, Lama Ahmad, Ilge Akkaya, Florencia Leoni Aleman, Diogo Almeida, Janko Altenschmidt, Sam Altman, Shyamal Anadkat, et al. GPT-4 technical report. arXiv preprint arXiv:2303.08774, 2023. Cited on page 1. 

- [3] Alexandr Andoni, Zhao Song, Clifford Stein, Zhengyu Wang, and Peilin Zhong. Parallel graph connectivity in log diameter rounds. In FOCS, October 2018. Cited on pages 17 and 18. 

- [4] Rohan Anil, Andrew M Dai, Orhan Firat, Melvin Johnson, Dmitry Lepikhin, Alexandre Passos, Siamak Shakeri, Emanuel Taropa, Paige Bailey, Zhifeng Chen, et al. Palm 2 technical report. arXiv preprint arXiv:2305.10403, 2023. Cited on page 41. 

- [5] Dzmitry Bahdanau, Kyunghyun Cho, and Yoshua Bengio. Neural machine translation by jointly learning to align and translate. In ICLR, 2014. Cited on page 1. 

- [6] Peter W Battaglia, Jessica B Hamrick, Victor Bapst, Alvaro Sanchez-Gonzalez, Vinicius Zambaldi, Mateusz Malinowski, Andrea Tacchetti, David Raposo, Adam Santoro, Ryan Faulkner, et al. Relational inductive biases, deep learning, and graph networks. arXiv preprint arXiv:1806.01261, 2018. Cited on page 1. 

- [7] Paul Beame, Paraschos Koutris, and Dan Suciu. Communication steps for parallel query processing. JACM, 64(6), oct 2017. Cited on page 19. 

- [8] James Bergstra and Yoshua Bengio. Random search for hyper-parameter optimization. JMLR, 13(2), 2012. Cited on page 41. 

- [9] Maciej Besta, Nils Blach, Ales Kubicek, Robert Gerstenberger, Michal Podstawski, Lukas Gianinazzi, Joanna Gajda, Tomasz Lehmann, Hubert Niewiadomski, Piotr Nyczyk, and Torsten Hoefler. Graph of thoughts: Solving elaborate problems with large language models. AAAI, 38(16):17682–17690, March 2024. Cited on page 1. 

- [10] Amartya Shankha Biswas, Talya Eden, Quanquan C. Liu, Slobodan Mitrovi´c, and Ronitt Rubinfeld. Massively parallel algorithms for small subgraph counting, 2022. Cited on pages 6 and 19. 

- [11] Tom Brown, Benjamin Mann, Nick Ryder, Melanie Subbiah, Jared D Kaplan, Prafulla Dhariwal, Arvind Neelakantan, Pranav Shyam, Girish Sastry, Amanda Askell, et al. Language models are few-shot learners. NeurIPS, 2020. Cited on pages 1 and 41. 

- [12] Quentin Cappart, Didier Chételat, Elias B Khalil, Andrea Lodi, Christopher Morris, and Petar Veliˇckovi´c. Combinatorial optimization and reasoning with graph neural networks. JMLR, 24 (130):1–61, 2023. Cited on page 1. 

- [13] Ziwei Chai, Tianjie Zhang, Liang Wu, Kaiqiao Han, Xiaohai Hu, Xuanwen Huang, and Yang Yang. GraphLLM: Boosting graph reasoning ability of large language model, 2023. Cited on page 3. 

- [14] Ines Chami, Sami Abu-El-Haija, Bryan Perozzi, Christopher Ré, and Kevin Murphy. Machine learning on graphs: A model and comprehensive taxonomy. JMLR, 23(89):1–64, 2022. Cited on page 1. 

- [15] Sam Coy and Artur Czumaj. Deterministic massively parallel connectivity. In STOC, STOC ’22, June 2022. Cited on page 19. 

- [16] Antonia Creswell, Murray Shanahan, and Irina Higgins. Selection-inference: Exploiting large language models for interpretable logical reasoning. In ICLR, 2023. Cited on page 1. 

- [17] Jeffrey Dean and Sanjay Ghemawat. Mapreduce: Simplified data processing on large clusters. In OSDI’04: Sixth Symposium on Operating System Design and Implementation, pages 137– 150, San Francisco, CA, 2004. Cited on page 18. 

10 

- [18] Jacob Devlin, Ming-Wei Chang, Kenton Lee, and Kristina Toutanova. Bert: Pre-training of deep bidirectional transformers for language understanding. arXiv preprint arXiv:1810.04805, 2018. Cited on page 1. 

- [19] Alexey Dosovitskiy, Lucas Beyer, Alexander Kolesnikov, Dirk Weissenborn, Xiaohua Zhai, Thomas Unterthiner, Mostafa Dehghani, Matthias Minderer, Georg Heigold, Sylvain Gelly, et al. An image is worth 16x16 words: Transformers for image recognition at scale. In ICLR, 2021. Cited on page 1. 

- [20] Vijay Prakash Dwivedi and Xavier Bresson. A generalization of transformer networks to graphs. arXiv preprint arXiv:2012.09699, 2020. Cited on page 3. 

- [21] Paul Erd˝os and Alfred Rényi. On random graphs. Publicationes Mathematicae Debrecen, 6: 290–297, 1959. Cited on page 40. 

- [22] Bahare Fatemi, Jonathan Halcrow, and Bryan Perozzi. Talk like a graph: Encoding graphs for large language models. In ICLR, 2024. Cited on pages 3, 6, 9, 40, 41, and 42. 

- [23] Fabian Frei and Koichi Wada. Efficient circuit simulation in mapreduce. ArXiv, abs/1907.01624, 2019. Cited on page 19. 

- [24] Roy Frostig, Matthew James Johnson, and Chris Leary. Compiling machine learning programs via high-level tracing. Systems for Machine Learning, 4(9), 2018. Cited on page 41. 

- [25] Mohsen Ghaffari, Fabian Kuhn, and Jara Uitto. Conditional hardness results for massively parallel computation from distributed lower bounds. In FOCS, 2019. Cited on page 19. 

- [26] Justin Gilmer, Samuel S Schoenholz, Patrick F Riley, Oriol Vinyals, and George E Dahl. Neural message passing for quantum chemistry. In ICML, 2017. Cited on pages 1, 7, 8, 41, and 42. 

- [27] Sachin Goyal, Ziwei Ji, Ankit Singh Rawat, Aditya Krishna Menon, Sanjiv Kumar, and Vaishnavh Nagarajan. Think before you speak: Training language models with pause tokens. In ICLR, 2024. Cited on pages 2, 3, and 15. 

- [28] Alex Graves, Greg Wayne, and Ivo Danihelka. Neural turing machines. arXiv preprint arXiv:1410.5401, 2014. Cited on page 1. 

- [29] Yiding Hao, Dana Angluin, and Robert Frank. Formal language recognition by hard attention transformers: Perspectives from circuit complexity. TACL, 10:800–810, 2022. Cited on page 3. 

- [30] Jordan Hoffmann, Sebastian Borgeaud, Arthur Mensch, Elena Buchatskaya, Trevor Cai, Eliza Rutherford, Diego de Las Casas, Lisa Anne Hendricks, Johannes Welbl, Aidan Clark, et al. Training compute-optimal large language models. arXiv preprint arXiv:2203.15556, 2022. Cited on page 1. 

- [31] Bowen Jin, Gang Liu, Chi Han, Meng Jiang, Heng Ji, and Jiawei Han. Large language models on graphs: A comprehensive survey. arXiv preprint arXiv:2312.02783, 2023. Cited on page 3. 

- [32] Armand Joulin and Tomas Mikolov. Inferring algorithmic patterns with stack-augmented recurrent nets. NIPS, 2015. Cited on page 1. 

- [33] Norman P. Jouppi, Cliff Young, Nishant Patil, David Patterson, Gaurav Agrawal, Raminder Bajwa, Sarah Bates, Suresh Bhatia, Nan Boden, Al Borchers, Rick Boyle, Pierre-luc Cantin, Clifford Chao, Chris Clark, Jeremy Coriell, Mike Daley, Matt Dau, Jeffrey Dean, Ben Gelb, Tara Vazir Ghaemmaghami, Rajendra Gottipati, William Gulland, Robert Hagmann, C. Richard Ho, Doug Hogberg, John Hu, Robert Hundt, Dan Hurt, Julian Ibarz, Aaron Jaffey, Alek Jaworski, Alexander Kaplan, Harshit Khaitan, Daniel Killebrew, Andy Koch, Naveen Kumar, Steve Lacy, James Laudon, James Law, Diemthu Le, Chris Leary, Zhuyuan Liu, Kyle Lucke, Alan Lundin, Gordon MacKean, Adriana Maggiore, Maire Mahony, Kieran Miller, Rahul Nagarajan, Ravi Narayanaswami, Ray Ni, Kathy Nix, Thomas Norrie, Mark Omernick, Narayana Penukonda, Andy Phelps, Jonathan Ross, Matt Ross, Amir Salek, Emad Samadiani, Chris Severn, Gregory Sizikov, Matthew Snelham, Jed Souter, Dan Steinberg, Andy Swing, 

11 

Mercedes Tan, Gregory Thorson, Bo Tian, Horia Toma, Erick Tuttle, Vijay Vasudevan, Richard Walter, Walter Wang, Eric Wilcox, and Doe Hyun Yoon. In-datacenter performance analysis of a tensor processing unit. SIGARCH Comput. Archit. News, 2017. Cited on page 41. 

- [34] John Jumper, Richard Evans, Alexander Pritzel, Tim Green, Michael Figurnov, Olaf Ronneberger, Kathryn Tunyasuvunakool, Russ Bates, Augustin Žídek, Anna Potapenko, et al. Highly accurate protein structure prediction with alphafold. Nature, 596(7873):583–589,2021. Cited on page 1. 

- [35] Lukasz Kaiser and Ilya Sutskever. Neural GPUs learn algorithms. In ICLR, 2016. Cited on page 1. 

- [36] Jared Kaplan, Sam McCandlish, Tom Henighan, Tom B Brown, Benjamin Chess, Rewon Child, Scott Gray, Alec Radford, Jeffrey Wu, and Dario Amodei. Scaling laws for neural language models. arXiv preprint arXiv:2001.08361, 2020. Cited on page 1. 

- [37] Howard Karloff, Siddharth Suri, and Sergei Vassilvitskii. A Model of Computation for MapReduce, pages 938–948. Cited on pages 4, 17, and 18. 

- [38] Nora Kassner, Oyvind Tafjord, Ashish Sabharwal, Kyle Richardson, Hinrich Schuetze, and Peter Clark. Language models with rationality. In Houda Bouamor, Juan Pino, and Kalika Bali, editors, EMNLP, December 2023. Cited on page 1. 

- [39] Mehran Kazemi, Najoung Kim, Deepti Bhatia, Xin Xu, and Deepak Ramachandran. LAMBADA: Backward chaining for automated reasoning in natural language. In ACL, 2023. Cited on page 1. 

- [40] Jinwoo Kim, Tien Dat Nguyen, Seonwoo Min, Sungjun Cho, Moontae Lee, Honglak Lee, and Seunghoon Hong. Pure transformers are powerful graph learners, 2022. Cited on pages 2, 3, and 31. 

- [41] Diederik P Kingma and Jimmy Ba. Adam: A method for stochastic optimization. arXiv preprint arXiv:1412.6980, 2014. Cited on page 41. 

- [42] Thomas N Kipf and Max Welling. Semi-supervised classification with graph convolutional networks. arXiv preprint arXiv:1609.02907, 2016. Cited on pages 1, 7, 8, 41, and 42. 

- [43] Takeshi Kojima, Shixiang Shane Gu, Machel Reid, Yutaka Matsuo, and Yusuke Iwasawa. Large language models are zero-shot reasoners. NeurIPS, 35:22199–22213, 2022. Cited on page 41. 

- [44] Nayoung Lee, Kartik Sreenivasan, Jason D Lee, Kangwook Lee, and Dimitris Papailiopoulos. Teaching arithmetic to small transformers. In ICLR, 2024. Cited on pages 1 and 3. 

- [45] Yuxuan Li and James L. McClelland. Systematic generalization and emergent structures in transformers trained on structured tasks, 2022. Cited on page 3. 

- [46] Bingbin Liu, Jordan T. Ash, Surbhi Goel, Akshay Krishnamurthy, and Cyril Zhang. Transformers learn shortcuts to automata, 2022. Cited on page 3. 

- [47] Ze Liu, Yutong Lin, Yue Cao, Han Hu, Yixuan Wei, Zheng Zhang, Stephen Lin, and Baining Guo. Swin transformer: Hierarchical vision transformer using shifted windows. In CVPR, 2021. Cited on page 1. 

- [48] Ilya Loshchilov and Frank Hutter. Decoupled weight decay regularization. arXiv preprint arXiv:1711.05101, 2017. Cited on page 41. 

- [49] Andreas Loukas. What graph neural networks cannot learn: depth vs width. In ICLR, 2020. Cited on pages 1, 3, and 39. 

- [50] Eran Malach. Auto-regressive next-token predictors are universal learners, 2023. Cited on page 2. 

12 

- [51] Amil Merchant, Simon Batzner, Samuel S Schoenholz, Muratahan Aykol, Gowoon Cheon, and Ekin Dogus Cubuk. Scaling deep learning for materials discovery. Nature, 624(7990):80–85, 2023. Cited on page 1. 

- [52] William Merrill and Ashish Sabharwal. A logic for expressing log-precision transformers, 2022. Cited on page 2. 

- [53] William Merrill and Ashish Sabharwal. The parallelism tradeoff: Limitations of log-precision transformers. TACL, 11:531–545, 2023. Cited on pages 2 and 32. 

- [54] William Merrill and Ashish Sabharwal. The expressive power of transformers with chain of thought, 2023. Cited on pages 2 and 6. 

- [55] William Merrill, Ashish Sabharwal, and Noah A. Smith. Saturated transformers are constantdepth threshold circuits, 2021. Cited on pages 2 and 6. 

- [56] Erxue Min, Runfa Chen, Yatao Bian, Tingyang Xu, Kangfei Zhao, Wenbing Huang, Peilin Zhao, Junzhou Huang, Sophia Ananiadou, and Yu Rong. Transformer for graphs: An overview from architecture perspective, 2022. Cited on page 3. 

- [57] Christopher Morris, Yaron Lipman, Haggai Maron, Bastian Rieck, Nils M Kriege, Martin Grohe, Matthias Fey, and Karsten Borgwardt. Weisfeiler and Leman go machine learning: The story so far. JMLR, 24(1):15865–15923, 2023. Cited on pages 3 and 39. 

- [58] Danupon Nanongkai and Michele Scquizzato. Equivalence classes and conditional hardness in massively parallel computations. Distributed Computing, 35(2):165–183, January 2022. Cited on pages 5, 17, 18, 19, and 20. 

- [59] Maxwell Nye, Anders Johan Andreassen, Guy Gur-Ari, Henryk Michalewski, Jacob Austin, David Bieber, David Dohan, Aitor Lewkowycz, Maarten Bosma, David Luan, et al. Show your work: Scratchpads for intermediate computation with language models. arXiv preprint arXiv:2112.00114, 2021. Cited on page 2. 

- [60] Jorge Pérez, Pablo Barceló, and Javier Marinkovic. Attention is turing complete. J. Mach. Learn. Res., 22(1), jan 2021. ISSN 1532-4435. Cited on pages 2 and 3. 

- [61] Bryan Perozzi, Bahare Fatemi, Dustin Zelle, Anton Tsitsulin, Mehran Kazemi, Rami Al-Rfou, and Jonathan Halcrow. Let your graph do the talking: Encoding structured data for LLMs, 2024. Cited on pages 3, 41, and 42. 

- [62] Jackson Petty, Sjoerd van Steenkiste, Ishita Dasgupta, Fei Sha, Dan Garrette, and Tal Linzen. The impact of depth and width on transformer language model generalization. NAACL, 2024. Cited on page 1. 

- [63] Jacob Pfau, William Merrill, and Samuel R Bowman. Let’s think dot by dot: Hidden computation in transformer language models. arXiv preprint arXiv:2404.15758, 2024. Cited on pages 2 and 15. 

- [64] Colin Raffel, Noam Shazeer, Adam Roberts, Katherine Lee, Sharan Narang, Michael Matena, Yanqi Zhou, Wei Li, and Peter J Liu. Exploring the limits of transfer learning with a unified text-to-text transformer. JMLR, 21(1):5485–5551, 2020. Cited on pages 1, 6, and 41. 

- [65] Machel Reid, Nikolay Savinov, Denis Teplyashin, Dmitry Lepikhin, Timothy Lillicrap, Jeanbaptiste Alayrac, Radu Soricut, Angeliki Lazaridou, Orhan Firat, Julian Schrittwieser, et al. Gemini 1.5: Unlocking multimodal understanding across millions of tokens of context. arXiv preprint arXiv:2403.05530, 2024. Cited on page 1. 

- [66] Yu Rong, Yatao Bian, Tingyang Xu, Weiyang Xie, Ying Wei, Wenbing Huang, and Junzhou Huang. Self-supervised graph transformer on large-scale molecular data. NeurIPS, 33, 2020. Cited on page 3. 

- [67] Tim Roughgarden, Sergei Vassilvitskii, and Joshua R. Wang. Shuffles and circuits (on lower bounds for modern parallel computation). J. ACM, 65(6), nov 2018. Cited on page 19. 

13 

- [68] Clayton Sanford, Daniel Hsu, and Matus Telgarsky. Representational strengths and limitations of transformers, 2023. Cited on pages 3, 31, 32, 35, and 36. 

- [69] Clayton Sanford, Daniel Hsu, and Matus Telgarsky. Transformers, parallel computation, and logarithmic depth, 2024. Cited on pages 2, 3, 4, 5, 6, 15, 18, 20, 21, 22, 30, 31, 32, and 39. 

- [70] Franco Scarselli, Marco Gori, Ah Chung Tsoi, Markus Hagenbuchner, and Gabriele Monfardini. The graph neural network model. IEEE transactions on neural networks, 20(1):61–80, 2008. Cited on page 1. 

- [71] Noam Shazeer. GLU variants improve transformer. arXiv preprint arXiv:2002.05202, 2020. Cited on page 41. 

- [72] Kaya Stechly, Matthew Marquez, and Subbarao Kambhampati. GPT-4 doesn’t know it’s wrong: An analysis of iterative prompting for reasoning problems, 2023. Cited on page 3. 

- [73] Gemini Team, Rohan Anil, Sebastian Borgeaud, Yonghui Wu, Jean-Baptiste Alayrac, Jiahui Yu, Radu Soricut, Johan Schalkwyk, Andrew M Dai, Anja Hauth, et al. Gemini: a family of highly capable multimodal models. arXiv preprint arXiv:2312.11805, 2023. Cited on page 1. 

- [74] Hugo Touvron, Louis Martin, Kevin Stone, Peter Albert, Amjad Almahairi, Yasmine Babaei, Nikolay Bashlykov, Soumya Batra, Prajjwal Bhargava, Shruti Bhosale, et al. Llama 2: Open foundation and fine-tuned chat models. arXiv preprint arXiv:2307.09288, 2023. Cited on page 1. 

- [75] Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez, Lukasz Kaiser, and Illia Polosukhin. Attention is all you need. In NeurIPS, 2017. Cited on pages 1, 15, and 41. 

- [76] Petar Veliˇckovi´c and Charles Blundell. Neural algorithmic reasoning. Patterns, 2(7), 2021. Cited on page 1. 

- [77] Petar Veliˇckovi´c, Adrià Puigdomènech Badia, David Budden, Razvan Pascanu, Andrea Banino, Misha Dashevskiy, Raia Hadsell, and Charles Blundell. The CLRS algorithmic reasoning benchmark. In ICML, 2022. Cited on page 1. 

- [78] Heng Wang, Shangbin Feng, Tianxing He, Zhaoxuan Tan, Xiaochuang Han, and Yulia Tsvetkov. Can language models solve graph problems in natural language? In NeurIPS, 2023. Cited on pages 3 and 41. 

- [79] Colin Wei, Yining Chen, and Tengyu Ma. Statistically meaningful approximation: a case study on approximating turing machines with transformers, 2021. Cited on page 2. 

- [80] Jason Wei, Xuezhi Wang, Dale Schuurmans, Maarten Bosma, Fei Xia, Ed Chi, Quoc V Le, Denny Zhou, et al. Chain-of-thought prompting elicits reasoning in large language models. NeurIPS, 2022. Cited on pages 1, 2, 15, and 41. 

- [81] Boris Weisfeiler and Andrei Leman. The reduction of a graph to canonical form and the algebra which appears therein. nti, Series, 2(9):12–16, 1968. Cited on pages 3 and 39. 

- [82] Keyulu Xu, Weihua Hu, Jure Leskovec, and Stefanie Jegelka. How powerful are graph neural networks?, 2018. Cited on pages 1, 3, 7, 8, 39, 41, and 42. 

- [83] Andrew Chi-Chih Yao. Some complexity questions related to distributive computing(preliminary report). In STOC, page 209–213, New York, NY, USA, 1979. Cited on page 36. 

- [84] Shunyu Yao, Dian Yu, Jeffrey Zhao, Izhak Shafran, Tom Griffiths, Yuan Cao, and Karthik Narasimhan. Tree of thoughts: Deliberate problem solving with large language models. NeurIPS, 2024. Cited on page 1. 

- [85] Ruosong Ye, Caiqi Zhang, Runhui Wang, Shuyuan Xu, and Yongfeng Zhang. Language is all a graph needs, 2023. Cited on page 3. 

14 

- [86] Chulhee Yun, Srinadh Bhojanapalli, Ankit Singh Rawat, Sashank J. Reddi, and Sanjiv Kumar. Are transformers universal approximators of sequence-to-sequence functions?, 2019. Cited on page 2. 

- [87] Wojciech Zaremba and Ilya Sutskever. Learning to execute. In ICLR, 2015. Cited on page 1. 

- [88] Bohang Zhang, Shengjie Luo, Liwei Wang, and Di He. Rethinking the expressive power of GNNs via graph biconnectivity. In ICLR, 2023. Cited on pages 1 and 3. 

- [89] Yi Zhang, Arturs Backurs, Sébastien Bubeck, Ronen Eldan, Suriya Gunasekar, and Tal Wagner. Unveiling transformers with lego: a synthetic reasoning task, 2022. Cited on page 3. 

- [90] Hattie Zhou, Azade Nova, Hugo Larochelle, Aaron Courville, Behnam Neyshabur, and Hanie Sedghi. Teaching algorithmic reasoning via in-context learning. arXiv preprint arXiv:2211.09066, 2022. Cited on page 1. 

- [91] Hattie Zhou, Arwen Bradley, Etai Littwin, Noam Razin, Omid Saremi, Josh Susskind, Samy Bengio, and Preetum Nakkiran. What algorithms can transformers learn? a study in length generalization. In ICLR, 2024. Cited on page 1. 

## A Theoretical preliminaries 

For some vector v ∈ R[N] , let 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_Understanding_Transformer_Reasoning_Capabilities_via_Graph_Algorithms_images/nips_2024_Understanding_Transformer_Reasoning_Capabilities_via_Graph_Algorithms.pdf-0015-08.png)


Let ei ∈ R[m] denote the ith elementary vector,[⃗] 1 = (1, . . . , 1) ∈ R[m] and[⃗] 0 = (0, . . . , 0) ∈ R[m] . Let [n] = {1, . . . , n}. 

## A.1 Transformer models 

We use a similar theoretical definition of a multi-layer transformer model to the one employed by [69]. We use a bounded communication theoretical model of the standard bidirectional transformer of [75], which assumes that the principal limitation of the transformer’s representational power is the low rank of the self-attention matrix. Several theoretical assumptions are necessary to model transformers in this regime: 

- the interleaved element-wise multi-layer perceptron (MLP) units compute arbitrary functions; 

- the embedding dimension m and number of heads per layer H are much smaller than the sequence length N ; and 

- all model parameters, inputs, and intermediate products can be represented with O(log N )- bit numbers. 

These assumptions are justified in greater detail by [69]. 

Furthermore, we also model transformers as having N[′] additional “pause token” inputs, which are embeddings that contain no information relevant information about the input instance, but which provide a longer “work tape” that can be utilized for computational purposes [see e.g. 27, 63]. While structurally similar to chain-of-thought reasoning [80], these pause tokens are not determined by multiple auto-regressive passes over the transformer and do not encode useful information as input. Such pause tokens exist both in the theoretical results of Section 3 (as blank tokens at the end of the sequence) and in the empirical results of Section 4 (as placeholders between the graph representation and the task query for graphs that can be represented in fewer tokens than the maximum sequence length). 

Definition 1. Let Transformer[N] m[,] ,[N] H[ ′] ,L[denote the family of all bidirectional][ transformers][ with embed-] ding dimension m, number of heads H, and depth L, which operate on input sequences on length N with N[′] blank pause tokens appended to the end[5] . Any transformer f ∈ Transformer[N] m[,] ,[N] H[ ′] ,L[is a] 

> 5If N ′ = 0, we denote the family as TransformerNm,H,L[.] 

15 

function of the form f : R[N][×][d] → R[N] for some input dimension d, which parameterized by query, key, and value matrices 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_Understanding_Transformer_Reasoning_Capabilities_via_Graph_Algorithms_images/nips_2024_Understanding_Transformer_Reasoning_Capabilities_via_Graph_Algorithms.pdf-0016-01.png)


and arbitrary element-wise MLPs with positional embeddings φ[0] , . . . , φ[L] and ψ with 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_Understanding_Transformer_Reasoning_Capabilities_via_Graph_Algorithms_images/nips_2024_Understanding_Transformer_Reasoning_Capabilities_via_Graph_Algorithms.pdf-0016-03.png)


where for sequence Y of length N + N[′] , 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_Understanding_Transformer_Reasoning_Capabilities_via_Graph_Algorithms_images/nips_2024_Understanding_Transformer_Reasoning_Capabilities_via_Graph_Algorithms.pdf-0016-05.png)


To evaluate f (X) on some input sequence X = (x1, . . . , xN ) ∈ R[N][×][d] , we define intermediate embeddings Y[0] , . . . , Y[L] ∈ R[(][N][+][N][ ′][)][×][m] as follows: 

- The initial embedding is computed by applying the first MLP φ[0] to all elements of the input X, along with N[′] “blank” inputs: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_Understanding_Transformer_Reasoning_Capabilities_via_Graph_Algorithms_images/nips_2024_Understanding_Transformer_Reasoning_Capabilities_via_Graph_Algorithms.pdf-0016-08.png)


- The intermediate embeddings Y[ℓ] for ℓ ∈{1, . . . , L} are computed by applying a unit of multi-headed attention (parameterized by Qℓ,h, Kℓ,h, Vℓ,h for all h ∈ [H]) to the previous embedding Y[ℓ][−][1] , followed by an element-wise application of the MLP φ[ℓ] . That is, 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_Understanding_Transformer_Reasoning_Capabilities_via_Graph_Algorithms_images/nips_2024_Understanding_Transformer_Reasoning_Capabilities_via_Graph_Algorithms.pdf-0016-10.png)


- The output f (X) ∈ R[N] is computed to be the first N outputs of ψ(Y[L] ). That is, 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_Understanding_Transformer_Reasoning_Capabilities_via_Graph_Algorithms_images/nips_2024_Understanding_Transformer_Reasoning_Capabilities_via_Graph_Algorithms.pdf-0016-12.png)


While our theoretical results pertain to the bidirectional regime (where no masking is applied to the self-attention layer), all results in Appendix C and all lower bounds in Appendix B apply to autoregressive transformers as well. Our empirical results utilize causal masking. 

Furthermore, an L-layer bidirectional transformer that operates on sequences of length N with N[′] pause tokens can be simulated by and L-layer autoregressive transformer with O(L · (N + N[′] )) pause tokens. In the regime where N[′] = poly(N ) and L = O(log N ), any positive results for bidirectional models likewise apply to autoregressive models. 

## A.2 Graph reasoning tasks 

We exclusively consider graph reasoning tasks that apply to undirected and unweighted graphs G = (V , E) of bounded size, i.e. |V | + |E| = O(N ) for some size parameter N . We define the tasks used for experimentation as follows: 

- Node count: Given graph G, compute |V |. 

- Edge count: Given graph G, compute |E|. 

- Edge existence: Given graph G and vertices u, v ∈ V , return 1 {(u, v) ∈ E}. 

- Node degree: Given graph G and vertex u ∈ V , return deg(u) = |(u, v) ∈ E|. 

- Connectivity: Given graph G and vertices u, v ∈ V , return 1 if there exists a path of edges between u and v and 0 otherwise. 

- Cycle check: Given graph G, return 1 if there exists a cycle of any length ≥ 3 and 0 otherwise. 

- Shortest path: Given graph G and vertices u, v ∈ V , return the smallest number of edges that forms a path between u and v if one exists and −1 otherwise. 

16 

- Triangle counting: Given graph G, return the number of triangles in the graph: |{(u, v, w) : (u, v), (v, w), (u, w) ∈ E}|. 

These tasks are foundational to computer science and are solved by famous polynomial-time algorithms, such as Dijkstra’s algorithm for shortest path and Kruskal’s for spanning trees. Several of these tasks reap significant algorithmic benefits from parallel computing models, including graph connectivity, which can be evaluated in a logarithmic number of rounds by employing an “iterative merging” strategy [3]. 

The amenability of some of these tasks to parallel computing is the core principle underlying our theoretical results on transformer model capacity and the resulting contrasts with GNNs. For instance, the iterative merging algorithm for connectivity can be simulated by a logarithmic-depth transformer, but message-passing GNNs cannot do without a substantial width scaling (see Appendix D). Furthermore, this parallel computing algorithm is widely believed to be optimal, and a crystallization of this as Conjecture 13 is the bedrock of our transformer optimality conjectures. 

The multi-layer results of Appendix B are representation-agnostic; the bounds apply to any fixed encoding of vertices and edges as a transformer input sequence X ∈ R[N][×][d] . The input representation for Appendix C must satisfy the more specified node/edge encoding scheme, which represents each vertex and edge as a single embedding, followed by a query embedding (with optional blank embeddings as needed). This encoding scheme is reflected by Figure 1 and closely resembles the graph instance encoding used for the experimental results[6] . 

Throughout the paper, we partition these and other graph reasoning tasks into three representational categories: retrieval, parallelizable, and search tasks. The retrieval category contains only four of the tasks used for experiments: 

- Retrieval tasks include node count, edge count, edge existence, and node degree. 

On the other hand, the other categories reflect two equivalence classes introduce by [58]: 

- Parallelizable tasks are defined to be L-complete and equivalent to connectivity in O(1) rounds of MPC. These tasks include connectivity, cycle check, planarity testing, minimum cut, bipartiteness testing, minimum spanning forest, connected components, one-cycle versus two-cycles (see Conjecture 13), and # connected components. 

- Search tasks are defined to be NL-complete and equivalent to shortest path in O(1) rounds of MPC. These tasks include shortest path, strong connectivity (for directed graphs), allpairs shortest path, median, diameter, radius, directed cycle detection, and st-reachability. 

Triangle counting does not appear in any of these classes and is separately analyzed in Section 3.4. 

## B Multi-layer transformers and parallelizable/search graph reasoning tasks 

In this appendix, we formalize and prove all results in Sections 3.1 and 3.2 that pertain to the parallelizable and search categories of graph reasoning tasks. 

The primary technical tool used to establish these results is an improved relationship between the MPC model of distributed communication of [37] and our theoretical model of a transformers (Appendix A.1). This result is presented informally as Theorem 1, and formally as Theorem 8. Because graph reasoning tasks are well studied in the MPC model of computation [e.g. 3, 58], this theorem enables the transfer of those positive results to transformer architectures. Similarly, negative results that pertain to the MPC model, including those conditioned on the well-known one-cycle versus two-cycle conjecture (Conjecture 13), imply negative results for transformers. 

Theorem 8 (Formal version of Theorem 1; transformers simulate MPC). For constants 0 < δ < δ[′] < 1 and γ > 0, an R-round deterministic (γ, δ)-MPC protocol with n inputs can be simulated by a transformer f ∈ Transformer[N] m[,] ,[N] H[ ′] ,L[with][depth][L][=][O][(][R][)][,][single][heads][H][=][1][,] embedding dimension m = O(n[δ][′] ), context length N = n, and blank chain-of-thought tokens N[′] = max(0, O(n[1+][γ][−][δ] ) − n). 

> 6The empirical encoding scheme uses two tokens—not one—to encode each edge and has fixed size N . For all graphs whose encodings do not require all N tokens, the remaining tokens are blank placeholders that precede the query tokens. 

17 

We formally introduce the MPC distributed computing model in Appendix B.1, along with the onecycle versus two-cycle conjecture and a collection of positive results from [58]. In Appendix B.2, we formally present the task-specific graph reasoning results introduced in Sections 3.1 and 3.2 and prove them as implications of Theorem 8. We finally contextualize and prove Theorem 8 in Appendix B.3 by modifying the proof strategy of a similar result [69] and by showing that MPC protocols can be simulated by a weaker model of distributed computation. 

## B.1 MPC preliminaries 

The massively parallel computation (MPC) model of [37] formalizes distributed computing frameworks such as MapReduce [17] as theoretical models that are amenable to rigorous analysis. MPC pertains to a regime where an input that consists of n “words” is distributed across a very large number of machines q (e.g. q ≈ n[0.95] ), each of which contains a bounded local memory s (e.g. s ≈ n[0.1] ) where computation on individual machines is inexpensive but communication between machines is costly. We use the definition of MPC of [3], which quantifies the complexity of a protocol by the local memory s = O(n[δ] ), the global memory sq = O(n[1+][γ] ), and the number of communication rounds R. 

Definition 2. For global and local memory constants γ, δ > 0 and input size n, an R-round (γ, δ)MPC protocol specifies a distributed computing protocol over q = Θ(n[1+][γ][−][δ] ) machines, each of which contains a local memory of size s = O(n[δ] ). 

- An input to the protocol, which is encoded as a length-n sequence of p = Θ(log n)-bit words, is distributed across the local memories of the first ⌈n/s⌉ machines. 

- In each of the R rounds, each machine computes an arbitrary function its local memory at the time, which specifies at most s words to be transmitted to other machines. 

- Messages are simultaneously transmitted (subject to the constraint that each machine sends and receives at most s words of information), and each machine’s new local memory is set equal to the messages received. 

- After the final round, the output of the protocol is a concatenation of the local memories of the first ⌈n/s⌉ machines. 

We say that an MPC protocol computes some function f : Z[n] 2[p][→][Z][n] 2[p][if for any][ X][∈][Z][n] 2[p][encoded] as input to the protocol, the output of the protocol is f (X). 

## B.1.1 MPC and graph reasoning tasks 

In this section, we introduce previously known results about the abilities of Massively Parallel Computation protocols to solve graph reasoning tasks. The novel results presented in the subsequent section are the primarily the immediate implications of these results and Theorem 8. 

MPC round equivalence for parallelizable and search tasks These results largely reflect the contributions of [58], which establish a hierarchy of graph reasoning tasks that is reflected in our distinction between parallelizable and search tasks. They show that all parallelizable tasks are equivalent to connectivity in O(1) rounds of MPC and that all search tasks are likewise equivalent to shortest path. (See their Figure 1.) Concretely, they prove the following for all graphs G = (V , E) with |V | + |E| = O(n). 

Theorem 9 (Theorem 4 of [58]; round-equivalence of parallelizable tasks). If there exists an R-round (γ, δ)-MPC protocol for any parallelizable task—including graph connectivity, stconnectivity, cycle detection, formula evaluation, planarity testing, graph bipartiteness, connected components, minimum spanning forest, and minimum cut, one-cycle versus two-cycle testing—for some γ > 0 and δ ∈ (0, 1), then there exists an R[′] -round (γ[′] , δ)-MPC protocol for any other other parallelizable task where R[′] = R + O(1) and γ[′] = max(γ, 3). 

Theorem 10 (Theorem 5 of [58]; round-equivalence of search tasks). If there exists an R-round (γ, δ)-MPC protocol for any search task—including st-reachability (for directed graphs), strong connectivitity, directed cycle detection, shortest path, all-pairs shortest path, diameter, radius, and median—for some γ > 0 and δ ∈ (0, 1), then there exists an R[′] -round (γ[′] , δ)-MPC protocol for any other other search task where R[′] = R + O(1) and γ[′] = max(γ, 2). 

The equivalance of Theorem 9 has more immediate implications for the round complexity of parallelizable tasks than Theorem 10 does for search tasks because the round complexity of graph connec- 

18 

tivity is well-understood to be O(log n) and thought to be optimal. We first present a deterministic MPC positive result for graph connectivity. 

Theorem 11 (Theorem 6.2 of [15]; log-round connectivity MPC algorithm). For any γ > 0 and δ ∈ (0, 1), there exists a deterministic O(log n)-round (γ, δ)-MPC protocol that solves graph connectivity on any graph G = (V , E) of size |V | + |E| = O(n). 

Hence, all parallelizable tasks can be solved with a logarithmic number of MPC rounds with arbitrarily small polynomial local memory. 

Corollary 12 (Log-round parallelizable MPC algorithms). For any γ > 0 and δ ∈ (0, 1) and any parallelizable task, there exists a deterministic O(log n)-round (γ[′] , δ)-MPC protocol that solves the task on any graph G = (V , E) of size |V | + |E| = O(n) for γ[′] = max(γ, 3). 

The optimality of these logarithmic-round protocols is suggested by a conjecture about the hardness of distinguishing between two graphs with n vertices and n edges, one of whose edges are arranged in a single cycle of length n and the other in two disjoint cycles of length[n] 2[.][This is the well-known] “one-cycle versus two-cycle conjecture,” which is widely employed as a condition for distributed computing hardness [see e.g. 7, 67, 25]. 

Conjecture 13 (One-cycle versus two-cycle conjecture, see e.g. [25]). For any γ > 0 and δ ∈ (0, 1), any (γ, δ)-MPC protocol that distinguishes a single length-n cycle from two disjoint length-[n] 2[cycles] uses R = Ω(log n) rounds. 

Under this conjecture, Theorem 9 immediately implies the optimality of Corollary 12. 

Corollary 14 (Optimality of log-round parallelizable MPC algorithms). Conditional on Conjecture 13, for any γ > 0 and δ ∈ (0, 1), any (γ, δ)-MPC protocol that solves a parallelizable graph task on all graphs G = (V , E) of size |V | + |E| = O(n) uses R = Ω(log n) rounds. 

The round complexity of search tasks is less understood, and it is unknown if search tasks can be solved by O(log n)-round (γ, δ)-MPC protocols if δ ∈ (0,[1] 2[]][.] 

MPC protocols for problems with bounded circuit size More general MPC constructions are known for problems that solved by bounded-size boolean circuits, which include both parallelizable and search tasks. The well-known NC[i] classes of Boolean circuits that take N inputs and have poly(n) gates and depth O(log[i] n) have been shown to be computable by bounded-round MapReduce-like computational protocols [23] and by MPC protocols in particular [58]. 

Theorem 15 (Theorem 1 of [58]; log-round circuit MPC algorithms). For any problem in NC[i][+1] and any γ > 0 and δ ∈ ( 2[1][, 1)][,][there][exists][a][deterministic][O][(log][i][ n][)][-round][ (][γ][,][ δ][)][-MPC][protocol] that solves the problem. 

Since L and NL are known to belong to NC[2] , the following corollary is an immediate consequence. 

Corollary 16 (Log-round parallelizable and search MPC algorithms with high local memory). For any parallelizable or search graph reasoning task and any γ > 0 and δ ∈ ( 2[1][, 1)][,][there][exists][a] deterministic O(log n)-round (γ, δ)-MPC protocol that solves the task on all graphs G = (V , E) of size |V | + |E| = O(N ). 

Note that these results pertain exclusively to the “large local memory regime,” where each machine has memory at s = ω([√] n). Therefore, this does not guarantee the existence of a O(log n)-round MPC solution for any search task or for any parallelizable task with γ < 1. 

MPC protocol for triangle counting Finally, the triangle counting task can be solved in the MPC framework by utilizing a special case of a parallel algorithm for clique counting. These pertain to graphs with bounded arboricity α, a quantity that corresponds to the branching factor of a node that is bounded by the degree of the graph; these apply to arbitrary graphs by noting that α ≤|V |. Theorem 17 (Theorem 1.5 of [10]; loglog-round triangle counting MPC algorithm). For any k ≥ 3, δ ∈ (0, 1), and γ > 0, there exists an O(log log n)-round (γ, δ)-MPC protocol that computes the number of k-cliques in any graph G = (V , E) with |V | + |E| = O(n) and arboricity α = O(n[γ/][(][k][−][2)] ). 

19 

## B.2 Positive and negative graph reasoning results for multi-layer transformers 

We state formalized versions of the statements of Sections 3.1, 3.2 and 3.4 and prove them by invoking Theorem 8 jointly with the relationships between MPC and graph reasoning of Appendix B.1.1. 

## B.2.1 Parallelizable task results (Section 3.1) 

For the duration of this section, the class of parallelizable tasks includes all of those that are deemed equivalent to graph connectivity in O(1) rounds of MPC by [58], as stated in Theorem 9. 

We first present a statement that formalizes the existence of logarithmic-depth transformer constructions for solving parallelizable tasks. 

Theorem 18 (Formal version of Theorem 2; log-depth transformers compute parallelizable tasks). For any ǫ ∈ (0, 1) and any parallelizable task, there exists a transformer f ∈ Transformer[N] m[,] ,[N] H[ ′] ,L such that f (X) computes the task where X ∈ R[N] is some encoding of any graph G = (V , E) with |V | + |E| = O(N ) and f has depth L = O(log N ) and heads H = O(1) and embedding dimension m and pause tokens N[′] satisfying either 

- LogDepthPause: m = O(N[ǫ] ) and N[′] = O(N[4][−][ǫ][′] ), where ǫ[′] ∈ (0, ǫ); or 

- LogDepthWide: m = O(N[ǫ] ) and N[′] = 0, if ǫ > 2[1][.] 

Proof. The first bullet is an immediate implication of Corollary 12 and Theorem 8 with γ = 3, δ[′] = ǫ, and δ = ǫ[′] . The second bullet follows from Corollary 16 and Theorem 8 with γ = δ = 2[ǫ] and δ[′] = ǫ. 

We then establish that sub-logarithmic-depth solutions to any parallelizable task are impossible without having linear embedding dimension m or super-polynomial number of pause tokens N[′] , under the assumption that the one-cycle versus two-cycle conjecture holds. 

Theorem 19 (Formal version of Theorem 3; log-depth optimality for parallelizable tasks). Conditional on Conjecture 13, for any ǫ ∈ (0, 1) and γ > 0 and any parallelizable task, if there exists a transformer f ∈ Transformer[N] m[,] ,[N] H[ ′] ,L[that][solves the][task][and][has width][mH][=][O][(][N][ ǫ][)][and pause] tokens N + N[′] = O(N[1+][γ] ), then its depth satisfies L = Ω(log N ). 

Proof. The proof is a consequence of Corollary 14 and a result of [69] that proves the to simulate transformers with MPC protocols, an inversion of Theorem 8. We restate that result as follows. 

Theorem 20 (Theorem 3.4 of [69]; MPC simulates transformers). Fix any transformer f ∈ Transformer[N] m[,] ,[N] H[ ′] ,L[with][width][mH][=][O][(][N][ δ][)][for][some][δ][∈][(0, 1)][and][total][sequence][length] N + N[′] = O(N[1+][γ] ) for some γ ≥ 0. Then, for any δ[′] ∈ (δ, 1) and γ[′] = 1 + 2γ + δ[′] , there exists an O([L] δ[(1+][′] −δ[γ][)][)][-round][ (][γ][′][,][ δ][′][)][-MPC protocol that simulates][ f][.] 

Consider a transformer f ∈ Transformer[N] m[,] ,[N] L,[ ′] H[that solves the task with width][ mH][=][O][(][N][ ǫ][)][ and] total sequence length N + N[′] = O(N[1+][γ] for some constant ǫ ∈ (0, 1) and γ > 1. Then, there exists an Oǫ,γ(L)-round (1 + 2γ +[√] ǫ,[√] ǫ)-MPC protocol that solves the parallelizable task. If Conjecture 13 is true, then L = Ω(log N ). 

## B.2.2 Search task results (Section 3.2) 

We present the main result of Section 3.2 and show an equivalence between the minimum depth transformer needed to solve search tasks in a width-efficient manner. As before, the class of searchable tasks includes tasks that are equivalent to shortest path in O(1) MPC rounds (Theorem 10). 

Theorem 21 (Formal version of Theorem 4; LogDepthWide computes search tasks). For any ǫ ∈ ([1] 2[, 1)][and][any][search][task,][there][exists][a][transformer][f][∈][Transformer][N] m[,] ,[N] H[ ′] ,L[such][that] f (X) computes the task where X ∈ R[N] is some encoding of any graph G = (V , E) with |V | + |E| = O(N ), and f has depth L = O(log N ), heads H = O(1), embedding dimension m = O(N[ǫ] , and no pause tokens (N[′] = 0). 

20 

Proof. As before, the proof is an immediate consequence of Corollary 16 and Theorem 8. 

While both search and parallelizable tasks are computable logarithmic depth of considerable width, the minimum depth transformer of width O(N[ǫ] ) for small ǫ that computes search tasks is unknown. Despite this deficit, we can still establish the representation similarity of all search tasks by showing that their minimum depths vary by at most a constant additive factor. 

Theorem 22 (Depth-equivalence of search tasks). Suppose for some ǫ ∈ (0, 1) and γ > 0 there exists a transformer f ∈ Transformer[N] m[,] ,[N] H[ ′] ,L[with][embedding][dimension][m][=][O][(][N][ ǫ][)][and][total] sequence length N + N[′] = O(N[1+][γ] ) that computes the some search task on all graphs G = (V , E) of size |V | + |E| = O(N ). Then, for any other search task, there exists some transformer f[′] ∈ N[¯][′] Transformer[N] ¯[,][m][¯][ =][ O][(][m][)][, depth][L][¯][ =][ L][ +][ O][(1)][, and pause tokens] ¯ m,H,L[¯][with embedding dimension] N[′] = N[′] + O(N[3] ) that computes that search task. 

Proof. This statement is an immediate consequence of Theorem 10 and Theorem 8. 

## B.2.3 Clique counting task results (Section 3.4) 

We prove the existing of doubly-logarithmic-depth transformer that solves the triangle counting task giving a more general result that counts the number of k-cliques in any graph of bounded arboricity. Theorem 23 (Generalization of Theorem 7; loglog-depth computes clique counting). For any fixed ǫ ∈ (0, 1) and k ≥ 3, there exists a transformer f ∈ Transformer[N] m[,] ,[N] H[ ′] ,L[with embedding dimension] m = O(n[ǫ] ), heads H = O(1), depth L = O(log log n), and chain-of-thought tokens 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_Understanding_Transformer_Reasoning_Capabilities_via_Graph_Algorithms_images/nips_2024_Understanding_Transformer_Reasoning_Capabilities_via_Graph_Algorithms.pdf-0021-06.png)


that counts the number of k-cliques in all graphs G = (V , E) of arboricity α and size |V | + |E| = O(N ). 

Theorem 7 follows immediately from taking k = 3. 

Proof. This is the immediate implication of Theorem 8 and Theorem 23. 

## B.3 Proof of Theorem 8 

Our principal theoretical result establishes that any MPC protocol with sublinear local memory can be simulated by a transformer with sublinear embedding dimension. 

Theorem 8 (Formal version of Theorem 1; transformers simulate MPC). For constants 0 < δ < δ[′] < 1 and γ > 0, an R-round deterministic (γ, δ)-MPC protocol with n inputs can be simulated by a transformer f ∈ Transformer[N] m[,] ,[N] H[ ′] ,L[with][depth][L][=][O][(][R][)][,][single][heads][H][=][1][,] embedding dimension m = O(n[δ][′] ), context length N = n, and blank chain-of-thought tokens N[′] = max(0, O(n[1+][γ][−][δ] ) − n). 

This offers an improvement on Theorem 3.1 of [69], which only guarantees that MPC protocols with local memory s = O(n[1][/][4][−][ǫ] ) (or with δ < 4[1][) can be simulated by transformers with sublinear] embedding dimension. This distinction is significant because several positive results for the MPC protocol (e.g. the ability to solve all problems in L and NL in a logarithmic number of MPC rounds) require s = Ω(N[1][/][2] ) local memory, and hence could not be shown to be simulated by transformers of sublinear embedding dimension previously[7] . 

Including an allowance of N[′] blank pause tokens permits the simulation of MPC protocol with γ ≥ δ, i.e. where number of machines q grows super-linearly with n. If γ < δ, then the protocol can be simulated without pause tokens (i.e. N[′] = 0) for sufficiently large input sizes n. 

> 7For this theoretical model of the transformer to be non-vacuous, the transformer must have embedding dimension m = o(Nd). Without this, any function over X ∈ R[N][×][d] could be solved by a single-layer transformer that concatenates all inputs into a single vector in an attention unit and passes that as input to an MLP that solves the problem. 

21 

To prove Theorem 8, we define a restriction of the MPC computational model that disallows each machine from communicating with more than k other machines in each round of the protocol. 

Definition 3. For constants γ, δ, ρ > 0, an R-round (γ, δ, ρ)-MPC protocol on input sequences of length n is a (γ, δ)-MPC protocol (Definition 2) that obeys an additional constraint on the number of outgoing and incoming messages. Namely, for some capacity k = O(n[ρ] ), in each round, every machine can send its local memory to and receive information from at most distinct k machines. 

Theorem 8 is an immediate consequence of Proposition 24—which establishes that any (γ, δ)-MPC protocol can be simulated by a (γ, δ, ρ)-MPC protocol—and Corollary 25—which shows that any (γ, δ, ρ)-MPC protocol can be simulated by a transformer. 

Proposition 24 ((γ, δ, ρ)-MPC simulates (γ, δ)-MPC). For constants γ, δ > 0 and ρ ∈ (0, 2[δ][)][,] if f can be computed by an R-round (γ, δ)-MPC protocol, then there exists a O([R][(1+] ρ[2][γ][)][2] )-round (γ, δ, ρ)-MPC protocol that computes f as well. 

Corollary 25 (Transformers simulate (γ, δ, ρ)-MPC). For constants δ ∈ (0, 1) and γ, ρ > 0, an R-round deterministic (γ, δ, ρ)-MPC protocol with n inputs can be simulated by a transformer f ∈ Transformer[N] m[,] ,[N] H[ ′] ,L[with depth][ L][ =][ R][+1][, heads][ H][= 1][, embedding dimension][ m][ =][ O][(][n][δ][+4][ρ][ log][ n][)][,] context length N = n, and blank chain-of-thought tokens N[′] = max(0, O(n[1+][γ][−][δ] ) − n). 

Proof of Theorem 8. Let ρ := min([δ] 2[,][δ][′] 4[−][δ][)][ −][ǫ][ for some small constant][ ǫ][ (e.g.][ ǫ][ := min(][ δ] 4[,][δ][′] 8[−][δ][)][).] By Proposition 24, we can simulate the target R-round (γ, δ)-MPC protocol with an R[′] -round (γ, δ, ρ)-MPC protocol for 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_Understanding_Transformer_Reasoning_Capabilities_via_Graph_Algorithms_images/nips_2024_Understanding_Transformer_Reasoning_Capabilities_via_Graph_Algorithms.pdf-0022-06.png)


We apply Corollary 25 to conclude that this latter protocol can be simulated by a transformer of depth L = R[′] + 1 and embedding dimension 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_Understanding_Transformer_Reasoning_Capabilities_via_Graph_Algorithms_images/nips_2024_Understanding_Transformer_Reasoning_Capabilities_via_Graph_Algorithms.pdf-0022-08.png)


We prove Proposition 24 in Appendix B.3.1 by simulating a single round of a standard MPC with multiple rounds of restricted MPC, where messages are sent to intermediate “neighborhoods”— which contain collections of machines with similar destinations—of increasing fine granularity. We prove Corollary 25 in Appendix B.3.2 by a minor adaptation of the proof of Theorem 3.1 of [69]. 

## B.3.1 Proof of Proposition 24 

Fix an R-round (γ, δ)-MPC protocol π with input size n, q = Θ(n[1+][γ][−][δ] ) machines, and s = O(n[δ] ) local memory. To prove Proposition 24, it suffices to show that a single round of MPC communication can be simulated an O([(1+] ρ[2][γ][)][2] )-round (γ, δ, ρ)-MPC protocol. 

To formalize the communication procedure to simulate, we let Outbox = (Outbox1, . . . , Outboxq) denote an encoding of the outgoing “message packets” from each source machine that obeys π’s local memory constraints. Concretely, 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_Understanding_Transformer_Reasoning_Capabilities_via_Graph_Algorithms_images/nips_2024_Understanding_Transformer_Reasoning_Capabilities_via_Graph_Algorithms.pdf-0022-13.png)


Let Inbox = (Inbox1, . . . , Inboxq) denote the same collection of message packets, organized by their destination machines. 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_Understanding_Transformer_Reasoning_Capabilities_via_Graph_Algorithms_images/nips_2024_Understanding_Transformer_Reasoning_Capabilities_via_Graph_Algorithms.pdf-0022-15.png)


It suffices to prove Lemma 26. 

Lemma 26 ((γ, δ, ρ)-MPC simulates one communication round of (γ, δ)-MPC). If ρ < 2[δ][,][there] exists an O([(1+] ρ[2][γ][)][2] )-round (γ, δ, ρ)-MPC protocol that takes as input any Outbox and returns the corresponding Inbox. 

22 

Proof. We define r = O([1+] ρ[γ][)][intermediate][steps][and][prove][that][each][of][those][can][be][simulated.] The intuition is that each an intermediate step routes each packet (Msg, Src, Dst) to a machine that belongs to the same “neighborhood” as Dst. Each step maps each packet to a neighborhood of smaller radius than the step before, until all packets have been transmitted to their proper destination location. 

We define a tree of neighborhoods as follows. Fix some branching factor ℓ = Θ(n[ρ/][2] ) and number of intermediate steps r = � loglog q ℓ � = O([1+] ρ[γ][)][.][For][any step][t][=][0,][ . . .][ ,][ r][and neighborhood index] q j = 1, . . . , bt := � ℓ[r][−][t] �, we define 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_Understanding_Transformer_Reasoning_Capabilities_via_Graph_Algorithms_images/nips_2024_Understanding_Transformer_Reasoning_Capabilities_via_Graph_Algorithms.pdf-0023-02.png)


and observe that it satisfies the size constraint |Nbhd[t] j[|][≤][ℓ][r][−][t][.][We][let][Nbhd][t][(][Dst][)][denote][the] unique neighborhood Nbhd[t] j[satisfying][ Dst][∈][Nbhd][t] j[.][We say that][ Nbhd][t] j[<][Nbhd][t] j[′][if][ j][<][j][′][(and] equivalently, for all i ∈ Nbhd[t] j[and][ i][′][∈][Nbhd][t] j[′][,][ i < i][′][).][Let] 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_Understanding_Transformer_Reasoning_Capabilities_via_Graph_Algorithms_images/nips_2024_Understanding_Transformer_Reasoning_Capabilities_via_Graph_Algorithms.pdf-0023-04.png)


for some τ ≥ t. 

Note that the following are true of neighborhoods. 

- The initial “step 0” neighborhood contains all machines (i.e. Nbhd[0] 1[=][[][q][]][),][and the][final] “step r” neighborhoods contain a single machine (i.e. Nbhd[r] j[=][ {][j][}][ for all][ j][∈][[][q][]][).] 

- For any t < r and j ∈ [bt], Nbhd[t] j[is the disjoint union of all sets in][ Children][(][Nbhd][t] j[)][ and] |Children(Nbhd[t] j[)][| ≤][ℓ][.] 

- �Nbhd[t] 1[,][ . . .][ ,][ Nbhd][t] bt � comprise a disjoint union of [q]. 

- For any Dst ∈ [q], there exist unique sets Nbhd[0] (Dst), . . . , Nbhd[r] (Dst) that satisfy 

Nbhd[r] (Dst) ⊂· · · ⊂ Nbhd[0] (Dst) 

and Nbhd[t][−][1] (Dst) = Parent(Nbhd[t] (Dst)). 

We define a collection of MPC machine states as “intermediate outboxes” Outbox[t] = (Outbox[t] 1[,][ . . .][ ,][ Outbox][t] q[)][ for each step][ t][ to represent an assignment of message packets to machines] in the same tth-level neighborhood as the packet destination. That is, we require that each Outbox[t] i satisfy 

Outbox[t] i[=] �(Msg, Src, Dst) : i ∈ Nbhd[t] (Dst)� (1) 

We say that Outbox[t] are valid intermediate outboxes for Outbox if: 

1. Outbox[t] satisfies Equation (1); 

2. there is a one-to-one mapping between packets in Outbox and Outbox[t] ; and 

3. local memory constraints are maintained up to a factor of two[8] 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_Understanding_Transformer_Reasoning_Capabilities_via_Graph_Algorithms_images/nips_2024_Understanding_Transformer_Reasoning_Capabilities_via_Graph_Algorithms.pdf-0023-19.png)


> 8For the sake of simplicity, we assume that the encoding Msg also contains all relevant metadata about the packet Src and Dst. That is, we only track the message size |Msg|. 

23 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_Understanding_Transformer_Reasoning_Capabilities_via_Graph_Algorithms_images/nips_2024_Understanding_Transformer_Reasoning_Capabilities_via_Graph_Algorithms.pdf-0024-00.png)


**----- Start of picture text -----**<br>
Step 0 Step 1 Step 2 Step r<br>1 Nbhd ℓ Nbhds ℓ [2] Nbhds q Nbhds<br>q machines per Nbhd q/ℓ machines per Nbhd q/ℓ [2] machines per Nbhd 1 machine per Nbhd<br>Nbhd [0] 1 Nbhd [1] 1 Nbhd [2] 1 Nbhd [r] 1<br>Machine 1 Machine 1 Machine 1<br>Outbox [0] 1 Outbox [1] 1 Outbox [r] 1<br>Machine 2<br>Outbox [0] 2<br>Machine q/ℓ Nbhd [2] ℓ<br>Outbox [1]<br>q/ℓ<br>Nbhd [1] ℓ Nbhd [2] ℓ [2] −ℓ+1<br>Nbhd [r]<br>Nbhd [2] ℓ [2] q<br>Machine q Machine q Machine q<br>Outbox [0] Outbox [1] Outbox [r]<br>q q q<br>**----- End of picture text -----**<br>


Figure 4: The neighborhood routing structure. 

Note that Outbox satisfies the conditions for Outbox[0] , and the only sequence of embeddings that satisfies the conditions for Outbox[r] is Inbox. An inductive argument that applies Lemma 27 r times completes the proof. 

Lemma 27 ((γ, δ, ρ)-MPC simulates one intermediate step). For any t ∈{0, . . . , r − 1} and ρ <[δ] 2[,] there exists an O([1+] ρ[γ][)][-round][ (][γ][,][ δ][,][ ρ][)][-MPC protocol that takes as input any satisfactory][ Outbox][t] and returns some satisfactory Outbox[t][+1] . 

Proof. The protocol operates in two phases: 

1. The pre-computation phase, where relevant message size metadata is computed using O(r − t) rounds of communication. 

2. The message-passing phase, where all messages are propagated in r − t rounds of communication in order to convert Outbox[t] to Outbox[t][+1] . 

Since r = O([1+] ρ[γ][)][, the bound on total number of rounds is satisfied.] 

The algorithm maintains the invariant that all communication occurs within t-level neighborhoods Nbhd[t] j[without][any][mixing between][neighborhoods;][this][is][possible][because][of][the][assumed][valid-] ity of Outbox[t] , which implies that all messages whose packets appear in Nbhd[t] j[of][Outbox][t][have] ultimate destinations in Nbhd[t] j[.][Concretely,][if][(][Msg][,][ Src][,][ Dst][)][∈][Outbox][t] i[and][i][∈][Nbhd][t] j[,][then] Dst ∈ Nbhd[t] j[.] 

We first explain the routing strategy for the message-passing phase by describing an itinerary of machines that each message packet will be routed to and proving that this itinerary meets the con- 

24 

ditions needed to be executed by an (r − t)-round (γ, δ, ρ)-MPC protocol. We then describe how the metadata necessary for the itinerary computations can be obtained during the pre-computation phase. 

Message-passing phase. We first introduce some packet notation to track all relevant metadata about any message in the tth intermediate step. Fix some particular input Outbox[t] . 

- Let P = (Msg, Src, Dst, Src[t] ) denote a packet that contains a message Msg and metadata concerning its “global” source Src and destination Dst (i.e. (Msg, Src, Dst) ∈ OutboxSrc, InboxDst) and its “local” source Src[t] from within Outbox[t] (i.e. (Msg, Src, Dst) ∈ Outbox[t] Src[t][).] 

- We write P ∈ Outbox[[t]] 

   - [[t]] i[if][ Src][t][=][ i][ and][ P][ ∈][Nbhd][ if][ P][ ∈][Outbox][t] i[for some][ i][ ∈][Nbhd][.] 

- Let SrcNbhd[t][,][τ] (P) = Nbhd[τ] (Src[t] ) represent the neighborhood of size ℓ[r][−][τ] contains P (i.e. P ∈ SrcNbhd[t][,][τ] (P)) and DstNbhd[t] (P) = Nbhd[t][+1] (Dst) denote the neighborhood of size ℓ[r][−][t][−][1] that contains the ultimate destination Dst. Because P ∈ Nbhd[t] j[if and only if] Dst ∈ Nbhd[t] j[, it follows that][ DstNbhd][t][(][P][)][ ⊂][SrcNbhd][t][,][t][(][P][)][.] 

- Let |P| = |Msg| be the total number of bits needed to encode the packet. 

- We establish a lexicographical ordering that depends first on Src[t] and next on Dst. That is, we say that P[′] < P for some P[′] = (Msg[′] , Src[′] , Dst[′] , Src[t][′] ) if Src[t][′] < Src[t] , or if Src[t][′] = Src[t] and Dst[′] < Dst. 

We develop a packet scoring function z[t][,][τ] (P), which in turn induces an itinerary function b[t][,][τ] (P) ∈ [q] that assigns an intermediate machine to hold packet P after r − τ communication steps. These functions are carefully chosen in order to ensure that the local memories of each individual machine are bounded by O(s) and that the number of machine each sends messages to and receives messages from is bounded by O(ℓ[2] ). Critically, we ensure that b[t][,][τ] (P) ∈ SrcNbhd[t][,][τ] (P); that is, we initially rearrange packets within solely within the smallest neighborhoods of Outbox[t] and gradually propagate packets more widely. That way, each packet P obeys a trajectory of the following form over r − t MPC rounds: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_Understanding_Transformer_Reasoning_Capabilities_via_Graph_Algorithms_images/nips_2024_Understanding_Transformer_Reasoning_Capabilities_via_Graph_Algorithms.pdf-0025-09.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_Understanding_Transformer_Reasoning_Capabilities_via_Graph_Algorithms_images/nips_2024_Understanding_Transformer_Reasoning_Capabilities_via_Graph_Algorithms.pdf-0025-10.png)


To define these functions, we first define partial sums of message sizes in order to later determine an itinerary of machines that P will be routed to in between Src[t] and some destination machine in DstNbhd[t] (P). The first term, EqualDstSum[t][,][τ] (P), sums the size of all “lesser” packets that share a destination neighborhood DstNbhd[t] (P) and a τ th level input neighborhood SrcNbhd[t][,][τ] (P): 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_Understanding_Transformer_Reasoning_Capabilities_via_Graph_Algorithms_images/nips_2024_Understanding_Transformer_Reasoning_Capabilities_via_Graph_Algorithms.pdf-0025-12.png)


The second term, LessDstSum[t][,][τ] (P), sums the size of all packets that share an input neighborhood but have a “smaller” destination neighborhood than P: 

LessDstSum[t][,][τ] (P) = LessDstSum[t] (SrcNbhd[t][,][τ] (P), DstNbhd[t] (P)), 

where 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_Understanding_Transformer_Reasoning_Capabilities_via_Graph_Algorithms_images/nips_2024_Understanding_Transformer_Reasoning_Capabilities_via_Graph_Algorithms.pdf-0025-16.png)


We now define the packet scoring and itinerary functions for any τ ∈{t, . . . , r}: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_Understanding_Transformer_Reasoning_Capabilities_via_Graph_Algorithms_images/nips_2024_Understanding_Transformer_Reasoning_Capabilities_via_Graph_Algorithms.pdf-0025-18.png)


25 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_Understanding_Transformer_Reasoning_Capabilities_via_Graph_Algorithms_images/nips_2024_Understanding_Transformer_Reasoning_Capabilities_via_Graph_Algorithms.pdf-0026-00.png)


We prove a series of claims to establish that the packet scoring and itinerary functions are properly defined. 

Claim 28 (Itinerary range). For any τ ∈{t, . . . , r}, the itinerary function satisfies 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_Understanding_Transformer_Reasoning_Capabilities_via_Graph_Algorithms_images/nips_2024_Understanding_Transformer_Reasoning_Capabilities_via_Graph_Algorithms.pdf-0026-03.png)


As an immediate consequence, b[t][,][r] (P) = Src[t] for P = (Msg, Src, Dst, Src[t] ). 

Proof. We first bound the scoring function z[t][,][τ] . Note that 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_Understanding_Transformer_Reasoning_Capabilities_via_Graph_Algorithms_images/nips_2024_Understanding_Transformer_Reasoning_Capabilities_via_Graph_Algorithms.pdf-0026-06.png)


Therefore, for τ > t, 

z[t][,][τ] (P) ∈ �2s · min SrcNbhd[t][,][τ] (P), 2s · (min SrcNbhd[t][,][τ] (P) + |SrcNbhd[t][,][τ] (P)|) −|P|� , (2) 

and 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_Understanding_Transformer_Reasoning_Capabilities_via_Graph_Algorithms_images/nips_2024_Understanding_Transformer_Reasoning_Capabilities_via_Graph_Algorithms.pdf-0026-10.png)


which proves the first case of the claim. The second case follows by observing that 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_Understanding_Transformer_Reasoning_Capabilities_via_Graph_Algorithms_images/nips_2024_Understanding_Transformer_Reasoning_Capabilities_via_Graph_Algorithms.pdf-0026-12.png)


Were it not true, there would exist at least one machine i ∈ DstNbhd[t] (P) that must receive more an s quantity of messages at the end of the entire round of the protocol π (i.e.[�] Msg∈Inboxi[|][Msg][|][> s][),] which contradicts the MPC assumption. 

Claim 29 (Gaps between scores). If P1 = P2 and z[t][,][τ] (P1) ≤ z[t][,][τ] (P2), then 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_Understanding_Transformer_Reasoning_Capabilities_via_Graph_Algorithms_images/nips_2024_Understanding_Transformer_Reasoning_Capabilities_via_Graph_Algorithms.pdf-0026-15.png)


Proof. First, let τ > t. Consider the case where SrcNbhd[t][,][τ] (P1) = SrcNbhd[t][,][τ] (P2). By Claim 28 and our assumption that z[t][,][τ] (P1) ≤ z[t][,][τ] (P2), it must be the case that SrcNbhd[t][,][τ] (P1) < SrcNbhd[t][,][τ] (P2). Hence, we have the following by applying Equation (2). 

z[t][,][τ] (P2) − zt,τ (P1) 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_Understanding_Transformer_Reasoning_Capabilities_via_Graph_Algorithms_images/nips_2024_Understanding_Transformer_Reasoning_Capabilities_via_Graph_Algorithms.pdf-0026-18.png)


Otherwise, if SrcNbhd[t][,][τ] (P1) = SrcNbhd[t][,][τ] (P2), then 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_Understanding_Transformer_Reasoning_Capabilities_via_Graph_Algorithms_images/nips_2024_Understanding_Transformer_Reasoning_Capabilities_via_Graph_Algorithms.pdf-0026-20.png)


for some packet subsets S1, S2 ⊂ SrcNbhd[t][,][τ] (P1). By further inspecting the respective LessDstSum[t][,][τ] and EqualDstSum[t][,][τ] terms, we observe that S1 ⊂ S2 and P1 ∈ S2 \ S1. The claim immediately follows. 

The argument for the case τ = t is nearly identical. 

Claim 30 (Local memory bound). For any b ∈ N, 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_Understanding_Transformer_Reasoning_Capabilities_via_Graph_Algorithms_images/nips_2024_Understanding_Transformer_Reasoning_Capabilities_via_Graph_Algorithms.pdf-0026-24.png)


26 

Proof. The case τ = r is an immediate consequence of the inductive assumption that Outbox[t] satisfies the desired intermediate properties. 

For all other cases, let {P1, . . . , Pn} denote all packets with b[t][,][τ] (Pi) = b and let z[t][,][τ] (P1) ≤· · · ≤ z[t][,][τ] (Pn) without loss of generality. We use Claim 29, the assumption that all |Pi| ≤ s, and the boundedness of z[t][,][τ] (Pi) from Claim 28 to conclude the proof. 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_Understanding_Transformer_Reasoning_Capabilities_via_Graph_Algorithms_images/nips_2024_Understanding_Transformer_Reasoning_Capabilities_via_Graph_Algorithms.pdf-0027-02.png)


Claim 31 (Intra-class distance preservation). If P1 and P2 satisfy SrcNbhd[t][,][τ][+1] (P1) = SrcNbhd[t][,][τ][+1] (P2) and DstNbhd[t] (P1) = DstNbhd[t] (P2), then 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_Understanding_Transformer_Reasoning_Capabilities_via_Graph_Algorithms_images/nips_2024_Understanding_Transformer_Reasoning_Capabilities_via_Graph_Algorithms.pdf-0027-04.png)


Proof. Since SrcNbhd[t][,][τ][+1] (P1) = SrcNbhd[t][,][τ][+1] (P2), it follows that SrcNbhd[t][,][τ] (P1) = SrcNbhd[t][,][τ] (P2) and therefore, 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_Understanding_Transformer_Reasoning_Capabilities_via_Graph_Algorithms_images/nips_2024_Understanding_Transformer_Reasoning_Capabilities_via_Graph_Algorithms.pdf-0027-06.png)


Because P1, P2 ∈ SrcNbhd[t][,][τ][+1] (P1), the defined packet ordering implies that any P[′] ∈ [P1, P2) must satisfy P[′] ∈ SrcNbhd[t][,][τ][+1] (P1). Therefore, Equations (3) and (4) are equal and the claim holds. Claim 32 (Distinct recipients bound). For any b, 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_Understanding_Transformer_Reasoning_Capabilities_via_Graph_Algorithms_images/nips_2024_Understanding_Transformer_Reasoning_Capabilities_via_Graph_Algorithms.pdf-0027-08.png)


Proof. Within each root neighborhood Nbhd[τ] j[,][there][exist][at][most][ℓ][destination][neighborhoods in] DstNbhd[t] (P) for P ∈ Nbhd[τ] j[.] 

Fix some such DstNbhd and let P1, . . . , Pn denote all packets with b[t][,][τ][+1] (P) = b and DstNbhd[t] (Pi) = DstNbhd. Without loss of generality, assume that z[t][,][τ] (P1) ≤· · · ≤ z[t][,][τ] (Pn). Because all such packets belong to the same machine in step r − τ − 1 (i.e. b[t][,][τ][+1] (Pi) = b for all i ∈ [n]), they belong share the same source neighborhood of size ℓ[r][−][τ][−][1] (i.e. SrcNbhd[t][,][τ][+1] (Pi) = SrcNbhd[t][,][τ][+1] (P1)). By Claim 31 and the definition of b[t][,][τ] , 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_Understanding_Transformer_Reasoning_Capabilities_via_Graph_Algorithms_images/nips_2024_Understanding_Transformer_Reasoning_Capabilities_via_Graph_Algorithms.pdf-0027-11.png)


Therefore, there are at most three possible values of b[t][,][τ] (Pi). 

The claim follows by considering each of the ℓ destination neighborhoods separately: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_Understanding_Transformer_Reasoning_Capabilities_via_Graph_Algorithms_images/nips_2024_Understanding_Transformer_Reasoning_Capabilities_via_Graph_Algorithms.pdf-0027-14.png)


27 

Claim 33 (Distinct senders bound). For any b, 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_Understanding_Transformer_Reasoning_Capabilities_via_Graph_Algorithms_images/nips_2024_Understanding_Transformer_Reasoning_Capabilities_via_Graph_Algorithms.pdf-0028-01.png)


Proof. Within each Nbhd[τ] j[,][there][exist][at][most][ℓ][2][distinct][pairs][of][destination][neighborhoods] DstNbhd[t] (P) and source neighborhoods Nbhd[t][,][τ] (P) for P ∈ Nbhd[τ] j[.] 

As before, we fix some DstNbhd and SrcNbhd and let P1, . . . , Pn all satisfy b[t][,][τ] (Pi) = b, DstNbhd[t] (Pi) = DstNbhd, and SrcNbhd[t][,][τ][+1] = SrcNbhd. Using the same argument, we show that 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_Understanding_Transformer_Reasoning_Capabilities_via_Graph_Algorithms_images/nips_2024_Understanding_Transformer_Reasoning_Capabilities_via_Graph_Algorithms.pdf-0028-04.png)


We conclude by considering all such pairs. 

As a result of Claims 30, 32, and 33, we conclude that each packet P = (Msg, Src, Dst, Src[t] ) can be equipped with some itinerary 

Src[t] = b[t][,][r] (P), b[t][,][r][−][1] (P), . . . , b[t][,][t][+1] (P), b[t][,][t] (P) ∈ DstNbhd[t] (P) 

that properly translates an instances of Outbox[t] to Outbox[t][+1] and does so without ever requiring local memory more than 3s = O(N[δ] ) on any intermediate step or any machine to send or receive messages from more than 3ℓ[2] = O(N[ρ] ) other machines. This itinerary can be executed using an (r − t)-round (γ, δ, ρ)-MPC protocol. 

Pre-computation phase. It remains to show that each b[t][,][τ] can be computed for each packet. To do so, we prove that there exists an O(r−t)-round (γ, δ, ρ)-MPC protocol that ends with each machine i knowing EqualDstSum[t][,][τ] (P) and LessDstSum[t][,][τ] (P) for each P ∈ Outbox[t] i[and][ τ][∈{][t][,][ . . .][ ,][ r][}][. We] design an MPC protocol that uses the tree structure to propagate information about individual child neighborhoods to their parents and vice versa. We describe recursive relationships that elucidate how to compute the two salient quantities. 

First, we introduce a useful intermediate term. Let 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_Understanding_Transformer_Reasoning_Capabilities_via_Graph_Algorithms_images/nips_2024_Understanding_Transformer_Reasoning_Capabilities_via_Graph_Algorithms.pdf-0028-11.png)


denote the sum of all packet that are contained by “lesser” machines that share source and destination neighborhoods. Note that EqualDstSum[t][,][τ] (P) can be computed for any P ∈ Outbox[t] i[locally][by] machine i given prior knowledge of EqualDstSum[t][,][τ] (i, DstNbhd[t] (P)). Thus, the pre-computation phase need only compute the latter term. 

We also introduce a term that represents sum of the sizes of all packets that share source and destination neighborhoods: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_Understanding_Transformer_Reasoning_Capabilities_via_Graph_Algorithms_images/nips_2024_Understanding_Transformer_Reasoning_Capabilities_via_Graph_Algorithms.pdf-0028-14.png)


Now, we provide the recurrences for any τ < r (or for any SrcNbhd satisfying |SrcNbhd| > 1: 

EqualDstSum[t][,][τ] (i, DstNbhd) 

= EqualDstSum[t][,][τ][+1] (i, DstNbhd) + EqualDstSum[t][,][τ] (min Nbhd[τ][+1] (i), DstNbhd) 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_Understanding_Transformer_Reasoning_Capabilities_via_Graph_Algorithms_images/nips_2024_Understanding_Transformer_Reasoning_Capabilities_via_Graph_Algorithms.pdf-0028-18.png)


(6) 

28 

When τ = r, the terms EqualDstSum[t][,][τ] (i, DstNbhd), NbhdSum[t] (Nbhd[r] (i), DstNbhd), and LessDstSum[t] (Nbhd[r] (i), DstNbhd) can be computed locally within machine i. 

We follow a tree-like communication pattern to compute all relevant sums. Each machine i ∈ [q] computes 

�EqualDstSum[t][,][τ] (i, DstNbhd) : τ ≥ t, DstNbhd ∈ Children(Nbhd[t] (i))� 

and 

�LessDstSum[t] (Nbhd[τ] (i), DstNbhd) : τ ≥ t, DstNbhd ∈ Children(Nbhd[t] (i))� 

by completing r − t propagate-up rounds, r − t aggregation rounds, and r − t propagate-down rounds. 

- The propagate-up rounds compute the neighborhood-wide message-size summations NbhdSum[t] (Nbhd[τ] (i), DstNbhd) and LessDstSum[t] (Nbhd[τ] (i), DstNbhd) in each machine i satisfying i = min Nbhd[τ] (i) for each τ and DstNbhd. 

- The aggregation rounds accumulate NbhdSum terms of the same level into specific EqualDstSum terms. 

- The propagate-down rounds iteratively compute and share the EqualDstSum and LessDstSum terms with all relevant machines. 

Propagate-up rounds: Fix some neighborhood Nbhd[t] j[.] After r − τ propagate-up rounds, the goal is to compute the terms NbhdSum[t] (SrcNbhd, DstNbhd) and LessDstSum[t] (SrcNbhd, DstNbhd) for each relevant destination neighborhood DstNbhd ∈ Children(Nbhd[t] j[)][and][source][neighbor-] hood SrcNbhd ∈ Descendants[τ] (Nbhd[t] j[)][within][a][single][machine][in][each][source][neighborhood,] min SrcNbhd. We argue that this is possible inductively. Before performing any computation (that is, after “round 0”), each machine i individually computes NbhdSum[t] (Nbhd[r] (i), DstNbhd) and LessDstSum[t] (Nbhd[r] (i), DstNbhd) by aggregating the messages encoded in its own representation of Outbox[t] i[.] 

We assume that the procedure works as specified for r − τ rounds of communication. Fix some SrcNbhd[τ][−][1] ∈ Descendants[τ][−][1] (Nbhd[t] j[)][.][Then,][for][every][ SrcNbhd][τ][∈][ChildrenSrcNbhd][τ][−][1][,] the quantities 

- {NbhdSum[t] (SrcNbhd[τ] , DstNbhd) : DstNbhd ∈ Children(Nbhd[t] j[)][}] 

∪{LessDstSum[t] (SrcNbhd[τ] , DstNbhd) : DstNbhd ∈ Children(Nbhd[t] j[)][}] 

have already been computed and stored in min SrcNbhd[τ] . By the recurrence relations of Equations (5) and (6), NbhdSum[t] (SrcNbhd[τ][−][1] , DstNbhd) and LessDstSum[t] (SrcNbhd[τ][−][1] , DstNbhd) are functions of those quantities. Thus, it suffices to have each machine min SrcNbhd[τ] machine transmit its relevant terms to min SrcNbhd[τ][−][1] . A round of MPC communication that transmits such messages involves each machine sending most one message of size ℓ and receiving at most ℓ messages, each of size O(ℓ). 

Inductively, we ensure that all neighborhood sums are computed after r − t propagate-up rounds. Because each machine handles at most ℓ distinct messages having total size O(ℓ[2] ) per MPC round, this protocol does not violate the bounded message size and bounded distinct message constraints (so long as ℓ[2] ≪ s), which can be guaranteed for sufficiently large n, so long as ℓ = O(n[ρ/][2] ). 

Aggregation rounds: After the completion of the aggregation rounds, each machine i computes terms of the form 

�EqualDstSum[t][,][τ] (min Nbhd[τ][+1] (i), DstNbhd) : DstNbhd ∈ Children(Nbhd[t] j[)] � 

from relevant NbhdSum[t] terms if i = min Nbhd[τ][+1] (i). By the recurrence, it is sufficient for machine i to following ℓ[2] distinct terms 

�NbhdSum[t] (SrcNbhd, DstNbhd) : SrcNbhd ∈ Children(Nbhd[τ] (i)), DstNbhd ∈ Children(Nbhd[t] j[)] � . 

Since all machines i already knows such terms for SrcNbhd = Nbhd[τ][+1] (i), it can obtain the remaining NbhdSum[t] terms by simultaneously sharing information with its “cousin machines:” 

29 

min SrcNbhd, for all SrcNbhd ∈ Children(Nbhd[τ] (i)). This can be handled by a single round of communication where each “(τ + 1)th-level neighborhood representative” machine forwards its sums sums to up to ℓ other first representatives, for a total messaging cost of O(ℓ[2] ). 

We use r − t separate rounds to repeat this process for each τ ≥ t. 

Propagate-down rounds: It remains to compute each EqualDstSum[t][,][τ] (i, ·) and LessDstSum[t] (Nbhd[τ] (i), ·) term at each machine i. The relevant LessDstSum[t] terms have already been computed by each respective min Nbhd[τ] (i) machine and can be propagated to machine i by using r − t rounds of tree-propagation through intermediate min Nbhd[τ][ ′] (i) machines. The same is possible for EqualDstSum[t][,][τ] terms, although the individual terms need to be added in order to follow the recurrence. We propagate the terms in the same way as the LessDstSum[t] terms, but we take special care to carry out the extra additions. This can be accomplished simultaneously to the other propagate-down rounds. This protocol involves each first-child node sharing at most ℓ distinct messages, each of size at most O((r − t)ℓ). As before, every node sends and receives at most O((r − t)ℓ[2] ) ≪ s words. 

After these 3(r − t) rounds have elapsed, each machine i computes b[t][,][τ] (P) for each P ∈ Outbox[t] i[.] Using this itinerary, the machines routes tuples of the form 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_Understanding_Transformer_Reasoning_Capabilities_via_Graph_Algorithms_images/nips_2024_Understanding_Transformer_Reasoning_Capabilities_via_Graph_Algorithms.pdf-0030-04.png)


to the respective machine b[t][,][τ] (P) in round r − τ . Due to the arguments presented at the start of the section, this procedure terminates with each (Msg, Src, Dst) tuple being held by some machine i such that the resulting Outbox[t] i[+1] is valid, and the procedure involves at most O(r − t) = O([1+] ρ[γ][)] rounds of (γ, δ, ρ)-MPC computation. 

## B.3.2 Proof outline of Corollary 25 

Corollary 25 (Transformers simulate (γ, δ, ρ)-MPC). For constants δ ∈ (0, 1) and γ, ρ > 0, an R-round deterministic (γ, δ, ρ)-MPC protocol with n inputs can be simulated by a transformer f ∈ Transformer[N] m[,] ,[N] H[ ′] ,L[with depth][ L][ =][ R][+1][, heads][ H][= 1][, embedding dimension][ m][ =][ O][(][n][δ][+4][ρ][ log][ n][)][,] context length N = n, and blank chain-of-thought tokens N[′] = max(0, O(n[1+][γ][−][δ] ) − n). 

The proof of Corollary 25 involves an adaptation to the proof of Theorem 3.1 of [69]. To avoid restating the proof in its entirety, we provide a brief outline of the proof of Theorem 3.1 and explain which modification is necessary. 

Theorem 3.1 is a consequence of Lemmas B.4, B.5, and B.6 of [69], which establish that there exist single-layer transformers that simulate the initialization, a round of computation and communication, and the output formatting of any fixed (γ, δ)-MPC protocol. The input and output steps of (γ, δ), and (γ, δ, ρ)-MPC protocols are identical, only Lemma B.5 needs to be examined. 

To simulate a single round of an MPC protocol with a transformer, all local computations are simulated in the element-wise multi-layer perceptron (MLP) units, and all communication is handled in a single multi-headed self-attention layer (Lemma B.7) Since (γ, δ, ρ)-MPC protocols add no restrictions related to local computation, the former can be simulated in exactly the same manner with identical MLPs, and it remains to analyze the construction of Lemma B.7. We restate Lemma B.7 and provide an replacement lemma that suffices to prove Corollary 25. 

Lemma 34 (Lemma B.7 of [69]; multi-headed attention simulates MPC communication). For any R-round MPC protocol with local memory s and q machines and any round r ∈ [R − 1], there exists a single-layer transformer f ∈ Transformer[q] m[,0] ,H,1[with][H][=][O][(log log][ q][)][and][m][=][O][(][s][4][ log][ q][)][,] which, given as input a length-q encoding of each machine’s outgoing messages in round r, returns an encoding of each machine’s incoming messages in round r + 1. 

Lemma 35 (Single-headed attention simulates bounded-message MPC communication). For any R- round MPC protocol with local memory s, q machines, and a k-machine communication limit and any round r ∈ [R − 1], there exists a single-layer single-headed transformer f ∈ Transformer[q] m[,0] ,H,1 with H = 1 and m = O(k[4] s log q), which, given as input a length-q encoding of each machine’s outgoing messages in round r, returns an encoding of each machine’s incoming messages in round r + 1. 

30 

Lemma 35 is an immediate consequence of Lemma 3.2 of [69], their main technical result, which already applies to the regime with limits on the number of machines in communication. 

By replacing Lemma 34 with Lemma 35, applying the remainder of the proof of Theorem 3.1 of [69], and letting k = O(n[ρ] ), the proof of Corollary 25 is complete. 

## C Single-layer transformers and graph reasoning tasks 

This appendix presents the results of Section 3.3, which separates the collection of graph reasoning tasks into those retrieval tasks that can be efficiently solved by single-layer parameter-efficient transformers—including node count, edge count, edge existence, and node degree—and those parallelizable or search tasks that require deeper constructions—including connectivity, shortest path, cycle check, and triangle count. Taken together, these results establish that the single-layer transformers of the Depth1 regime are capable of solving simple aggregation-based tasks, but that their known limitations in capacity as communication protocols of [68] apply to non-trivial graph reasoning tasks. 

We specific a particular node/edge encoding of an input graph G = (V , E) and a graph reasoning task using a consistent encoding scheme that closely resembles the encoding used in our graph reasoning experiments and those of [40]. This encoding is distinguished by the fact that each node and vertex of the graph G is represented by exactly one token, rather than the pair of tokens utilized in our experiments. This choice ensures that any trivial pre-processing of graph inputs (e.g. using a positional embedding to associate each edge token pair) need not count towards the single-layer transformer model. 

Definition 4. The node/edge encoding of a graph G = (V , E) with V ⊆ [n] and |V | + |E| ≤ N − 1 and a graph reasoning task P is a sequence 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_Understanding_Transformer_Reasoning_Capabilities_via_Graph_Algorithms_images/nips_2024_Understanding_Transformer_Reasoning_Capabilities_via_Graph_Algorithms.pdf-0031-06.png)


where d = 5 and each 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_Understanding_Transformer_Reasoning_Capabilities_via_Graph_Algorithms_images/nips_2024_Understanding_Transformer_Reasoning_Capabilities_via_Graph_Algorithms.pdf-0031-08.png)


satisfies the following conditions: 

- For each v ∈ V , there exists exactly one i ∈ [N − 1] with xi = (v, 0, 1, 0, 0). 

- For each (u, v) ∈ E, there exists exactly i ∈ [N − 1] with xi = (u, v, 0, 1, 0). 

- The token xN encodes a particular instance of the task P , by encoding isTaskN = 1 with an optional edge or node encoding. That is, for tasks without arguments (such as triangle count), xN = (0, 0, 0, 0, 1). For tasks with a single node argument v ∈ [n] (such as node degree), xN = (v, 0, 1, 0, 1). For tasks with a pair of node arguments u, v ∈ [n] (such as shortest path and connectivity), xN = (u, v, 0, 1, 1). 

- All other tokens satisfy xi = (0, 0, 0, 0, 0). 

We say that a single-layer transformer f ∈ Transformer[N] m,H,1[solves][ task][ P][on graph][ G][ if the output] corresponding to the task description f (X(G, P ))N encodes the output of the task. Since f is a single-layer transformer, we can write this output as 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_Understanding_Transformer_Reasoning_Capabilities_via_Graph_Algorithms_images/nips_2024_Understanding_Transformer_Reasoning_Capabilities_via_Graph_Algorithms.pdf-0031-15.png)


for element-wise multi-layer perceptrons 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_Understanding_Transformer_Reasoning_Capabilities_via_Graph_Algorithms_images/nips_2024_Understanding_Transformer_Reasoning_Capabilities_via_Graph_Algorithms.pdf-0031-17.png)


broadcasted across each input (i.e. φ(X) = (φ(x1), . . . , φ(xN )) ∈ R[N][×][m] ) and weight matrices 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_Understanding_Transformer_Reasoning_Capabilities_via_Graph_Algorithms_images/nips_2024_Understanding_Transformer_Reasoning_Capabilities_via_Graph_Algorithms.pdf-0031-19.png)


31 

Throughout, we assume that all parameters in the transformer model and intermediate numerical quantities can be written using O(log N )-bit floating point numbers. This assumption can be satisfied for the positive results of Appendix C.1 and is necessary to obtain the negative results of Appendix C.2. 

We permit the element-wise MLPs φ and ψ to be arbitrary functions for the negative results, while restricting them to be MLPs that can be approximated using bounded-size multi-layer ReLU networks for the positive results. While we do not make these ReLU networks explicit, we restrict ourselves to simple operations that can be computed using linear transformations and the application of smooth univariate functions. 

Finally, we acknowledge that the negative results in are largely superseded by those of [53], which establishes that L-complete languages (including graph connectivity and cycle check) cannot be efficiently solved by constant-depth transformers, let alone single-layer transformers. We include these bounds anyway to mark a contrast with our positive results; draw further connections to the communication complexity lens on transformers; and exhibit simple task instances (G, P ) that require greater depth, including constant diameter and constant degree graphs. 

## C.1 Positive results for single-layer transformers 

Theorem 36 (Formal version of Theorem 5; Depth1 computes retrieval tasks). Fix any graph reasoning task among node count, edge count, edge existence, and node degree and any graph size N . Then, there exists a single-layer single-headed transformer f ∈ Transformer[N] m,1,1[with embedding] dimension m = O(log N ) that solves the task on all graphs G = (V , E) of size |V | + |E| ≤ N − 1 formatted as node/edge embedding sequences. 

Proof. We prove that a single head of self-attention with input and output MLPs can solve these retrieval and aggregation tasks in a parameter-efficient manner by first carefully designing a universal input MLP φ : R[d] → R[m] for some m = O(log N ) to produce embeddings that encode useful graph properties. Then, we define task-specific query, key, and value matrices Q, K, V :∈ R[m][×][m] and output MLPs ψ : R[m] → R that produce the correct answers. While we do not explicitly account for finite-precision computations, all of the operations utilized can be carefully implemented to respect O(log N )-bit floating-point numerical representations using the technical approaches of [68, 69]. 

Shared sinusoidal embedding MLP. We denote the embedding output of the input MLP as 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_Understanding_Transformer_Reasoning_Capabilities_via_Graph_Algorithms_images/nips_2024_Understanding_Transformer_Reasoning_Capabilities_via_Graph_Algorithms.pdf-0032-07.png)


for some φ[′] (xi) : R[d] → R[2][m][′] for some m[′] = O(log N ) and m = 2m[′] + 3. For some fixed a1, . . . , am′ ∈ [0, 1] to be determined, let 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_Understanding_Transformer_Reasoning_Capabilities_via_Graph_Algorithms_images/nips_2024_Understanding_Transformer_Reasoning_Capabilities_via_Graph_Algorithms.pdf-0032-09.png)


where η is a sinusoidal embedding MLP for nodes with 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_Understanding_Transformer_Reasoning_Capabilities_via_Graph_Algorithms_images/nips_2024_Understanding_Transformer_Reasoning_Capabilities_via_Graph_Algorithms.pdf-0032-11.png)


and ξ is an edge embedding satisfying 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_Understanding_Transformer_Reasoning_Capabilities_via_Graph_Algorithms_images/nips_2024_Understanding_Transformer_Reasoning_Capabilities_via_Graph_Algorithms.pdf-0032-13.png)


We first show that the node embeddings are approximately orthogonal. By employing standard trigonometric identities, we obtain 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_Understanding_Transformer_Reasoning_Capabilities_via_Graph_Algorithms_images/nips_2024_Understanding_Transformer_Reasoning_Capabilities_via_Graph_Algorithms.pdf-0032-15.png)


and note that ∥η(v)∥[2] 2[=][ m][′][. We use a standard concentration argument to prove that][ |][η][(][u][)][T][η][(][v][)][| ≪] m[′] if u ̸= v with high probability. 

32 

Claim 37 (Near-orthogonality of embeddings η). There exist coefficients a1, . . . , am′ ∈ [0, 1] that comprise the embedding η : [n] → R[m] of Equation (8) such that 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_Understanding_Transformer_Reasoning_Capabilities_via_Graph_Algorithms_images/nips_2024_Understanding_Transformer_Reasoning_Capabilities_via_Graph_Algorithms.pdf-0033-01.png)


Proof. We employ the probabilistic method. Consider m[′] iid random variables a1, . . . , am′ ∼ Unif([0, 1]) and let η represent the respective node embedding. Fix some arbitrary u, v ∈ [n] with u ̸= v and note that 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_Understanding_Transformer_Reasoning_Capabilities_via_Graph_Algorithms_images/nips_2024_Understanding_Transformer_Reasoning_Capabilities_via_Graph_Algorithms.pdf-0033-03.png)


For any j ∈ [m[′] ], the integrality of u − v implies that 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_Understanding_Transformer_Reasoning_Capabilities_via_Graph_Algorithms_images/nips_2024_Understanding_Transformer_Reasoning_Capabilities_via_Graph_Algorithms.pdf-0033-05.png)


Hoeffding’s inequality provides the following bound: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_Understanding_Transformer_Reasoning_Capabilities_via_Graph_Algorithms_images/nips_2024_Understanding_Transformer_Reasoning_Capabilities_via_Graph_Algorithms.pdf-0033-07.png)


By applying a union bound to all �n2� choices of u and v, we have the following: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_Understanding_Transformer_Reasoning_Capabilities_via_Graph_Algorithms_images/nips_2024_Understanding_Transformer_Reasoning_Capabilities_via_Graph_Algorithms.pdf-0033-09.png)


Hence, there exists a satisfactory set of coefficients a1, . . . , am′. 

For some η satisfying Claim 37, let χ = maxu,v∈[n],u=v |η(u)[T] η(v)| ≤ 2[√] m[′] log n. By taking m[′] = O(log n) be sufficiently large, we guarantee that χ ≤[m] c[′][for any constant][ c][ ≥][2][.] 

We then bound all relevant inner products between vertex and edge embeddings for sufficently large m[′] (and sufficiently small χ). We assume throughout that u, u[′] , v, v[′] are distinct and note that ξ(u, v) = ξ(v, u) (and omit symmetric inequalities). 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_Understanding_Transformer_Reasoning_Capabilities_via_Graph_Algorithms_images/nips_2024_Understanding_Transformer_Reasoning_Capabilities_via_Graph_Algorithms.pdf-0033-13.png)


33 

Therefore, we conclude the following bounds: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_Understanding_Transformer_Reasoning_Capabilities_via_Graph_Algorithms_images/nips_2024_Understanding_Transformer_Reasoning_Capabilities_via_Graph_Algorithms.pdf-0034-01.png)


We now apply the above bounds to prove that each task can be solved by a single-headed transformer with weights Q, K, V ∈ R[m][×][m] , output MLP ψ : R[m] → R, and shared input MLP φ discussed previously. 

Node count task. If P encodes the node count task, we specify weights matrices Q, K, V in order to ensure that Q[T] φ(xN ) = m[′] e1[9] ; K[T] φ(xi) = (isVertexi + isTaski) · e1; and V[T] φ(xi) = isTaski · e1[10] . Then, the following is true of exponentiated key/query inner products and scalings products of value vectors: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_Understanding_Transformer_Reasoning_Capabilities_via_Graph_Algorithms_images/nips_2024_Understanding_Transformer_Reasoning_Capabilities_via_Graph_Algorithms.pdf-0034-04.png)


The output of the softmax is then 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_Understanding_Transformer_Reasoning_Capabilities_via_Graph_Algorithms_images/nips_2024_Understanding_Transformer_Reasoning_Capabilities_via_Graph_Algorithms.pdf-0034-06.png)


Let 

yN := (softmax �φ(xN )[T] QK[T] φ(X)� φ(X)V )1 ∈ R denote the first coordinate of the softmax output. By taking m[′] ≥ log(4N ), we guarantee that 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_Understanding_Transformer_Reasoning_Capabilities_via_Graph_Algorithms_images/nips_2024_Understanding_Transformer_Reasoning_Capabilities_via_Graph_Algorithms.pdf-0034-09.png)


Edge count task. We similarly let Q[T] φ(xN ) = m[′] e1; K[T] φ(xi) = (isEdgei + isTaski) · e1; and V[T] φ(xi) = isTaski · e1. By an identical analysis of the attention matrix and ψ construction, we ensure that f (X)N = |E|. 

Edge existence task. Under the node/edge encoding, we assume that the edge existence task is encoded as xN = (x[1] N[,][ x] N[2][, 0, 1, 1)][for][some][x][1] N[,][ x][2] N[∈][V][and][should][return][f][(][X][)][N][=] 1 �(x[1] N[,][ x][2] N[)][ ∈][E] �. We choose our weight matrices to ensure that Q[T] φ(xN ) = φ[′] (xN ) = ξ(x[1] N[,][ x] N[2][)][;][K][T][φ][(][x][i][)][=][φ][′][(][x][i][)][,][and][V][T][φ][(][x][i][)][=][2(1][ −][isTask][i][)][e][1][.][By][applying][Claim][37][and] letting m[′] = O(log N ) to be sufficiently large, the following is true the query/key inner products: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_Understanding_Transformer_Reasoning_Capabilities_via_Graph_Algorithms_images/nips_2024_Understanding_Transformer_Reasoning_Capabilities_via_Graph_Algorithms.pdf-0034-12.png)


> 9Note that QTφ(xN ) is the only relevant query embedding to the output f (X)N ; see Equation (7). 

> 10While we do not specify the entries of Q, K, V , they are clearly linear transformations. 

34 

We can therefore analyze the softmax output yN to obtain the following bound: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_Understanding_Transformer_Reasoning_Capabilities_via_Graph_Algorithms_images/nips_2024_Understanding_Transformer_Reasoning_Capabilities_via_Graph_Algorithms.pdf-0035-01.png)


Hence, 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_Understanding_Transformer_Reasoning_Capabilities_via_Graph_Algorithms_images/nips_2024_Understanding_Transformer_Reasoning_Capabilities_via_Graph_Algorithms.pdf-0035-03.png)


Therefore, it suffices to design a threshold output MLP ψ that satisfies 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_Understanding_Transformer_Reasoning_Capabilities_via_Graph_Algorithms_images/nips_2024_Understanding_Transformer_Reasoning_Capabilities_via_Graph_Algorithms.pdf-0035-05.png)


in order to distinguish between the two possible output ranges. This can be easily constructed by taking a linear combination of two ReLU neurons. 

Node degree task. We assume that the task is encoded as xN = (x[1] N[, 0, 1, 0, 1)][ and should return] f (X)N = deg(x[1] N[)][:=] ���(x[1] N[,][ v][)][ ∈][E] ���. We use weight matrices with QTφ(xN ) = φ′(xN ) = η(xN ); K[T] φ(xi) = φ[′] (xi); and V[T] φ(xi) = isTaski. This ensure that the following is true about the query/key inner products for sufficiently large m[′] : 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_Understanding_Transformer_Reasoning_Capabilities_via_Graph_Algorithms_images/nips_2024_Understanding_Transformer_Reasoning_Capabilities_via_Graph_Algorithms.pdf-0035-08.png)


We similarly ensure that the following holds about the softmax output yN . 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_Understanding_Transformer_Reasoning_Capabilities_via_Graph_Algorithms_images/nips_2024_Understanding_Transformer_Reasoning_Capabilities_via_Graph_Algorithms.pdf-0035-10.png)


We then use the similar approach employed in the node count task to choose an output MLP that 1 approximately computes φ(z) = � z1[−][2] �. 

We thus conclude that there exist single-layer transformers that solve all of the described tasks. 

## C.2 Negative results for single-layer transformers 

Theorem 38 (Formal version of Theorem 6; Depth1 cannot compute search or retrieval tasks). Fix any graph reasoning task among graph connectivity, shortest path, and cycle detection. Any singlelayer transformer f ∈ Transformer[N] m,H,1[with][O][(log][ N][)][-bit][precision][that][solves][the][task][on][all] graphs G = (V , E) of size |V | + |E| ≤ N − 1 formatted as node/edge embedding sequences has width satisfying mH = Ω(N/ log N ). 

Our negative results generalizes and applies the approach of [68] to prove negative results for singlelayer transformers by communication complexity reductions. The bounds hinge on the following fundamental fact about the hardness of a two-player game where two agents jointly attempt to compute a set disjointness quantity that depends on both of their inputs with bounded communication. 

35 

Fact 1 (Disjointness communication complexity lower bound [83]). Suppose Alice and Bob are given inputs a, b ∈{0, 1}[r] and wish to jointly compute DISJ(a, b) = maxi aibi by alternately sending single-bit messages to one another. Any deterministic protocol that computes DISJ(a, b) requires at least r rounds of communication, or r bits of information. 

We first generalize Theorem 7 of [68] to show that no transformer can efficiently solve an embedded set disjointness problem without having a width that scales linearly in r. We prove the theorem and later apply to it prove negative results about relevant graph reasoning tasks. To demonstrate that the graph reasoning tasks do not require pathological input graphs G to be hard, we exhibit particular input graph instances with constant graph diameter or constant degree where the task cannot be efficiently solved. 

Lemma 39 (Generic Depth1 communication complexity negative result). For some sequence length, fix two disjoint subsets A, B ⊂ [N − 1], and consider a single-layer transformer f ∈ Transformer[N] m,H,1[with][ O][(log][ N][)][-bit precision that solves set disjointness, i.e.][ f][(][X][)][N][=][ D][ISJ][(][a][,][ b][)] for any input X where XA is a function of Alice’s input a ∈{0, 1}[r] , XB is a function of Bob’s input b ∈{0, 1}[r] , and X[N ]\(A∪B) is fixed regardless of a and b. Then, f has width satisfying mH = Ω(r/ log N ). 

Proof. Consider any transformer f that solves DISJ as specified in the theorem statement. We show that this implies the existence of a O(mH log N )-round communication protocol that solves DISJ(a, b) for any inputs a, b ∈{0, 1}[r] . An application of Fact 1 immediately proves the theorem statement. 

If such a transformer f exists, then the following is true for some φ, ψ, and Q, K, V : 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_Understanding_Transformer_Reasoning_Capabilities_via_Graph_Algorithms_images/nips_2024_Understanding_Transformer_Reasoning_Capabilities_via_Graph_Algorithms.pdf-0036-05.png)


for partial softmax and normalization terms[11] defined as follows for S ⊂ [N ] and h ∈ [H]: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_Understanding_Transformer_Reasoning_Capabilities_via_Graph_Algorithms_images/nips_2024_Understanding_Transformer_Reasoning_Capabilities_via_Graph_Algorithms.pdf-0036-07.png)


The definition of input instances X implies that Alice can compute Zh,A and Lh,A as a function of her input a; Bob can similarly compute Zh,B and Lh,B from b; and Zh,[N ]\(A∪B), Lh,[N ]\(A∪B), and all implementation details of f are known by all players. Therefore, Bob can compute DISJ(a, b) by an O(mH log N )-round protocol where Alice sends him (Zh,A, Lh,A) bit-by-bit. 

It remains to apply Lemma 39 to each graph reasoning task by defining graph instance encodings X that encode DISJ(a, b) in the solution to the task for some r = Θ(N ). 

Proof of Theorem 38. For each task, we provide a pair of “hard instances:” one with constant degree and one with constant diameter, in order to obey different notions of graph simplicity. 

> 11These terms are designed in order to facilitate computation with log(N )-bit floating-point numbers. Note thatbits of precision to accurately approximate thaneach Zh,S is a convex combination of value Vh[T] vectors[φ][(][x][i][)][. Similarly, each] Vh[T][φ][(][x][i][)][,][which][ L][means][h][,][S][is within an][that][Z][h][,][S][ O][requires][(log][ N][)][ additive][no][more] factor of maxi∈S φ(xN )[T] QhKh[T][φ][(][x][i][)][.][Finally, the output of the softmax term for head][ h][ is a convex][combi-] nation of Zh,A, Zh,B, and Zh,[N]\(A∪B), which makes accurate computation of each output possible without requiring greater precision. 

36 

Graph connectivity task. For both instances, on any disjointness input a, b ∈{0, 1}[r] , we define a graph G = (V , E) with |V | + |E| = O(r) whose edges encode the input, i.e. E = E(a, b). We define three fixed disjoint sets of potential edges E[¯] A, E[¯] B, E[¯] ∗ with |E[¯] A| + |E[¯] B| + |E[¯] ∗| = O(r) such that E = EA(a) ∪ EB(b) ∪ E[¯] ∗, where EA(a) ⊂ E[¯] A is a function of only Alice’s input a and EB(b) ⊂ E[¯] B is a function of Bob’s b. 

We define an enumeration of all vertices in V and potential edges in E[¯] A, E[¯] B, E[¯] ∗. We first fix their sizes |V |, |E[¯] A|, |E[¯] B|, |E[¯] ∗| and then index the vertices and edges in the order V , E[¯] A, E[¯] B, E[¯] ∗. That is, 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_Understanding_Transformer_Reasoning_Capabilities_via_Graph_Algorithms_images/nips_2024_Understanding_Transformer_Reasoning_Capabilities_via_Graph_Algorithms.pdf-0037-02.png)


This implies that the following transformer input X ∈ R[N][×][d] for N = |V | + |S∗| + |A| + |B| + 1 is a valid node/edge encoding such that inputs XA is a function of a, XB is a function of b, and X[N ]\(A∪B) is constant: 

- For i ∈ V , xi = (i, 0, 1, 0, 0) encodes a fixed vertex. 

- For i ∈ S∗, xi = (ui, vi, 0, 1, 0) encodes a fixed edge. 

- For i ∈ A, xi represents edge (ui, vi) if it exists: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_Understanding_Transformer_Reasoning_Capabilities_via_Graph_Algorithms_images/nips_2024_Understanding_Transformer_Reasoning_Capabilities_via_Graph_Algorithms.pdf-0037-07.png)


- For i ∈ B, xi represents edge (ui, vi) if it exists: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_Understanding_Transformer_Reasoning_Capabilities_via_Graph_Algorithms_images/nips_2024_Understanding_Transformer_Reasoning_Capabilities_via_Graph_Algorithms.pdf-0037-09.png)


- xN = (u, v, 0, 1, 1) encodes the task token for some fixed (u, v) ∈ V[2] . 

Constant-diameter instance: We define a graph G of diameter at most 8 that encodes an disjointness instance a, b ∈{0, 1}[r] in an instance of connectivity. 

- Let |V | = 3r + 2, and let nodes 1 and 3r + 2 denote the “source” and “sink” node respectively. That is, xN = (1, 3r + 2, 0, 1, 1). 

- Edges between the source node 1 and nodes {2, . . . , r + 1} and between the end node 3r + 2 and {2r + 2, . . . , 3r + 1} always are included. That is, 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_Understanding_Transformer_Reasoning_Capabilities_via_Graph_Algorithms_images/nips_2024_Understanding_Transformer_Reasoning_Capabilities_via_Graph_Algorithms.pdf-0037-14.png)


- Alice’s input is encoded as edges (i + 1, i + r + 1) for i ∈ [r]. That is, 

   - EA(a) = {(i + 1, i + r + 1) : ai = 1} ⊂ E[¯] A = {(i + 1, i + r + 1) : i ∈ [r]} . 

- Bob’s inputs are similarly encoded as (i + r + 1, i + 2r): 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_Understanding_Transformer_Reasoning_Capabilities_via_Graph_Algorithms_images/nips_2024_Understanding_Transformer_Reasoning_Capabilities_via_Graph_Algorithms.pdf-0037-18.png)


We visualize this construction of G in Figure 5. 

There exists a path between node 1 and node 3r + 2 if and only if there exists some i ∈ [r] such that (i + 1, i + r + 1), (i + r + 1, i + 2r + 1) ∈ E, which corresponds to ai = bi = 1. Thus, the graph G is connected if and only if DISJ(a, b) = 1. Any transformer f that computes 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_Understanding_Transformer_Reasoning_Capabilities_via_Graph_Algorithms_images/nips_2024_Understanding_Transformer_Reasoning_Capabilities_via_Graph_Algorithms.pdf-0037-21.png)


37 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_Understanding_Transformer_Reasoning_Capabilities_via_Graph_Algorithms_images/nips_2024_Understanding_Transformer_Reasoning_Capabilities_via_Graph_Algorithms.pdf-0038-00.png)


**----- Start of picture text -----**<br>
a1 b1<br>2 5 8<br>b2<br>1 3 6 9 11<br>a3<br>4 7 10<br>**----- End of picture text -----**<br>


Figure 5: The constant diameter graph construction for r = 3, A = (1, 0, 1) and B = (1, 1, 0). The source 1 and sink 11 are connected which is equivalent to DISJ(A, B) = 1 by construction. 

also solves disjointness. Since N = |V | + |S∗| + |A| + |B| + 1 = Θ(r), we conclude the proof of hardness of solving graph connectivity on this instance by applying Lemma 39. 

Constant-degree instance: We can define a degree-3 graph G that similarly encodes a disjointness instance a, b. To do so, we modify the previous construction by replacing the fixed edges E∗ with two binary trees, each of depth O(log r), between the source and end nodes as roots and Alice and Bob’s nodes incident to E[¯] A and E[¯] B respectively as leaves. The remainder of the construction— including the encoding of EA(a) and EB(b) and the connectivity analysis—is identical. Since a binary tree with r leaves has O(r) nodes and edges, the graph G can be similarly encoded as a length-N transformer input for N = O(r). See Figure 6 for a visualization. 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_Understanding_Transformer_Reasoning_Capabilities_via_Graph_Algorithms_images/nips_2024_Understanding_Transformer_Reasoning_Capabilities_via_Graph_Algorithms.pdf-0038-04.png)


**----- Start of picture text -----**<br>
b1<br>a2 b2<br>S X<br>a4<br>**----- End of picture text -----**<br>


Figure 6: The degree 3 graph construction for r = 4, A = (0, 1, 0, 1), B = (1, 1, 0, 0). The source node S and sink node X are connected, which is equivalent to DISJ(A, B) = 1 by construction. 

Shortest path task. We can modify the graph connectivity constructions G to create a decision variant of shortest path, rather than graph connectivity. Let D(G) be the length of a path between the source and sink node in each connectivity construction, and note that D(G) = 4 for the constantdiameter instance and D(G) = O(log r) for constant-degree. We design a new graph G[′] of size O(r) then appends a path of D(G) + 1 vertices to G. Then, the shortest path between the source and sink node is D(G) if they are connected and D(G) + 1 if not. 

If there exists a transformer f[′] solves shortest path on graphs of the form G[′] , then there exists a minor modification f that solves graph connectivity on graph G. 

Cycle check task. We again modify the graph connectivity constructions. We produce G[′] by adding a single edge to G between the source node 1 and the sink node 3r +2, which ensures that G[′] has a cycle if and only if G is connected between the source and sink nodes. Therefore, a transformer f[′] can only solve check check if there exists a transformer f that solves connectivity. 

## D Representational gaps between transformers and GNNs 

Our experiments in Section 4 highlight the differences in graph reasoning abilities between vanilla transformers with a naive input graph tokenization and graph neural networks (GNNs). These distinctions can be understood as the consequences of model capacity or inductive bias between trans- 

38 

formers and GNNs. This appendix contrasts the novel analysis of the capacities of vanilla transformers in Section 3 with previously established limitations on GNNs. We present two theoretical tools for deriving negative results on GNN capabilities: CONGEST distributed computing model and the Weisfeiler-Leman isomorphism test (WL-test). We discuss the results implied by these frameworks and contrast them with transformer capabilities and limitations. 

## D.1 Limitations of GNNs via CONGEST analogy 

The bidirectional relationship between transformers and the massively parallel computation (MPC) distributed computing model of [69] was partially inspired by a similar analogy between GNNs and the CONGEST model by [49]. 

The MPC and CONGEST distributed computing protocols differ in their models of how messages are passed between machines. While MPC protocols permit any machine to send a message to any other machine subject to capacity constraints, CONGEST protocols operate on a graph topology that restricts machines to send messages exclusively to and from neighbors. As a consequence, two nodes in a CONGEST graph that are separated by a path of length p cannot share information with one another without Ω(p) rounds of communication; no similar notion of “long-range communication” exists for MPC. 

All message-passing GNNs where each node initially knows only its own features can be simulated by a CONGEST protocol whose rounds scales linearly in the depth of the GNN [49]. This reduction implies a lower bound on solving several parallelizable and search tasks discussed in Appendix A.2. 

Theorem 40. Any graph neural network f with width (message size) m and depth L that computes any of the following tasks on n-node graphs has the following size lower bounds: 

- Subgraph connectivity, minimum spanning forest, minumum cut, shortest path: L[√] m = ˜Ω([√] n). 

- Cycle detection, diameter computation: Lm = Ω([˜] n). 

If L = O(log N ), then for all tasks, m = Ω([˜] n). 

In contrast, the quantitative bounds in Section 3 establish sharp trade-offs between transformers and GNNs for parallelizable tasks and suggest a possible equivalence for search tasks. 

All parallelizable tasks—including (subgraph) connectivity, minimum spanning forest, minimum cut, and cycle detection—can be solved by transformers of depth L = O(log n) and width m = O(n[ǫ] ) for any constant ǫ ∈ (0, 1) due to Theorem 2. In contrast, Theorem 40 requires that a similardepth GNNs have width m = Ω([˜] n), and a GNN of comparable width requires depth L = Ω([˜] n[1][−][ǫ] ). 

On the other hand, search tasks, such as shortest path and diamter, are only guaranteed to be solvable by transformers of depth O(log n) and width O(n[1+][ǫ] ) (for graphs with |E| = n[2] ) by Theorem 4. This statement compares to the GNN negative results of Theorem 40. 

## D.2 Limitations of GNNs via the Weisfeiler-Leman test 

A relationship between GNNs and the Weisfeiler-Leman heuristic graph isomorphism test [81] (WL test) further establishes representational limitations of message-passing GNNs. This connection and the rich literature surrounding it is presented in greater detail by [57]. 

The 1-WL test is a permutation-invariant test for predicting whether two candidate graphs are isomorphic that works by first labeling each node with the empty set ∅ and then repeatedly replacing each label with a multiset of its neighbors’ labels. A hierarchy of WL test variants exists where the k-WL test maintains a label for each k-tuple of vertices. The failure models of these heuristic solutions are well-understood; critically, the 1-WL test cannot distinguish between connected and disconnected graphs. 

The abilities of message-passing GNNs without unique node identifiers to determine whether two graphs are isomorphic are limited by the graphs distinguishable by the 1-WL test [82, 57]. As an immediate consequence, such GNNs cannot solve graph connectivity at all, unlike transformers, which can do so with a logarithmic-depth parameter-efficient representation. The relationship is bidirectional; message-passing GNNs (and transformers as well) admit an efficient approximation of the 1-WL test. 

39 

The effectiveness of these bounds is tempered by the assumption that no node includes identifying features, which is easily overcome by standard GNN models. The analogy is further limited by the fact that graph embeddings of transformers must have some (possibly arbitrary) node identifier as input in order to tokenize a graph using the node/edge encoding without losing the ability to associate each node with its incident edges. However, the juxtaposition of the 1-WL and CONGESTbased limitations on the abilities of GNNs to solve connectivity-based tasks suggests a fundamental gap in capabilities between models that is apparent in multiple theoretical lenses. 

## E Experimental Details 

## E.1 Datasets 

We evaluate our model on the diverse graph reasoning tasks presented in GraphQA [22]. We used the public code of the dataset available at https://github.com/google-research/google-research/tree/master/graphqa. The code to generate the datasets is licensed under the Apache License, Version 2.0. The tasks in the dataset range in difficulty and encompass the following categories: 

- Graph-level: node counting (counting the number of nodes in a graph), edge counting (counting the number of edges in a graph), cycle check (determining whether a graph contains a cycle), and triangle counting (counting the number of triangles in a graph). 

- Node-level: node degree (calculating the degree of a given node in a graph). 

- Edge-level: connectivity (finding if there is a path from one node to another), edge existence (whether a given edge exists in a graph, and shortest path (finding the length of the shortest path from one node to another). 

The graphs used in the experiments in this paper and the corresponding graph reasoning tasks are taken from [22]. There are 1, 000 graphs in the original train set, 500 graphs in the dev set, and 500 graphs in the test set. The graphs are generated randomly using Erd˝os-Rényi (ER) random graph model [21]. Graph size ranges from 5 to 20 nodes. 

Train set statistics. Average number of nodes: 11.90; average number of edges: 37.01; average node degree: 5.43. 

Test set statistics. Average number of nodes: 12.37; average number of edges: 39.79; average node degree: 5.70. 

While random instances of graph reasoning tasks provide a valuable assessment of the task complexity on realistic graphs, they do not nec800 essarily reflect the “worst case” graph inputs that convey negative results like Theorem 6 and 600 Theorem 3. For example, the reduction that establishes that cycle check is “as hard as” graph connectivity and the consequential logarithmic400 depth hardness results hinge on the consideration of graph instances with n nodes and polynomial cycle length. However, as witnessed by 200 Figure 7, the shortest cycles observed in 1000 instances of GraphQA cycle check is almost 0 always of length three, and only 3.2% of inno cycle 3 4 5 6 stances are larger. As a consequence, identifying the existence of a cycle on the GraphQA Figure 7: Histogram of minimum cycle lengths dataset is inherently local, which is reflected by for cycle check instances. a strong performance by heuristic-based GNN solutions (Table 2)—despite the fact that efficient GNNs for worst-case cycle check do not exist (Theorem 40). 

Figure 7: Histogram of minimum cycle lengths for cycle check instances. 

For our experiments on the effect of the scale of the number of training data points in the final results we obtain, we use the open-source code of GraphQA available to generate a larger training dataset of 100K examples. We follow the original instructions and parameters to create this larger training dataset. 

40 

## E.2 Implementation Details 

Model Hyperparameters. We fixed the number of iterations as 1,000,000 and train standard decoder-only transformers with L = 12 layers, m = 768 embedding dimension, H = 12 heads, learning rate 5 · 10[−][4] , and dropout 0.1. These models have an approximate parameter count of 60,000,000. 

We used random search [8] over the following set of hyperparameters to select a universal architecture for all tasks: The range provided for the learning rate and dropout rate are [10[−][4] , 10[−][1] ] and [0, 0.5]. The number of layers L and embedding dimension m is selected from L ∈ {4, 6, 8, 10, 12, 14, 16} and m ∈{192, 384, 576, 768, 960, 1152, 1344, 1536}. We employed the GLU [71] activation as a non-linearity. 

Model Selection. We implemented our model in JAX [24] and used AdamW [41, 48] as the optimizer. Optimal hyperparameters for each task and model were determined by training on the GraphQAT rain dataset and evaluating performance on the GraphQADev dataset. The results presented in the paper are based on the held-out GraphQAT est dataset. 

Hardware Acceleration. All experiments were conducted using Google’s TPUv3 and TPUv5e accelerators [33]. 

## E.3 Baseline Results 

To rigorously evaluate the performance of transformers on graph reasoning tasks, we compare them against three established categories of baselines: 

1. Prompting-based methods. These methods provide the LLM with a textual descriptions of the graph and question within the prompt. We consider the following variations and copy the results from the original papers: 

   - ZERO-SHOT. In this approach, the model is given a task description and immediately asked to produce the desired output. No additional examples or demonstrations are provided. 

   - FEW-SHOT. This approach provides the model with a few examples of the task and their desired outputs [11]. Unlike traditional training, these examples are included directly in the prompt, allowing the model to learn and adapt during the inference. 

   - COT. Chain-of-thought (CoT) prompting [80] provides examples each showing step-by-step reasoning, teaching the LLM to generate its own thought processes for tackling new tasks. 

   - ZERO-COT. Zero-shot CoT [43] builds upon Chain-of-Thought (CoT) prompting by eliminating the need for training examples. The LLM generates its own step-by-step reasoning process using a simple trigger phrase like “Let’s think step by step”. 

   - COT-BAG. BAG prompting [78] extends COT to improve the performance of LLMs on graphrelated tasks by appending “Let’s construct a graph with the nodes and edges first” to the prompt. 

2. Graph-based methods. These models are specifically designed to process graphs as input and are trained task-specific. They leverage the connections between nodes to learn patterns and make predictions, making them ideal for tasks where a graph is involved. We use GCN [42], MPNN [26], and GIN [82] from this category. GraphToken [61] is a GNN-based model that processes the graph and feed the output of the GNN as soft-tokens to an LLM. 

3. Transformer models (Ours). The last class of model are task-specific vanilla transformer models [75]. The 60M transformer-1K model is the one described above trained on 1, 000 training examples from the GraphQA training set. To investigate the impact of training data scale, we generated a larger dataset containing 100, 000 examples, ensuring the same distribution as the original training set by using the official GraphQA code and trained 60M transformer-100K on that. The 11B transformer (FT)-1K is a vanilla transformer model that is started with a pretrained checkpoint of T5 [64] and is fine-tuned on the 1K training dataset. We also include two fine-tuned PaLM 2 [4] transformers of size XXS and XS. Similar to prompting baselines, this model receives a textual description of the graph as input to leverage its textual reasoning capabilities. 

The results for ZERO-SHOT, ZERO-COT, FEW-SHOT, COT, and COT-BAG are taken from Fatemi et al. [22]. Results for SOFT-PROMPT and GraphToken are sourced from Perozzi et al. [61]. 

41 

||||Retrieval tasks|Retrieval tasks||Parallelizable Tasks|Parallelizable Tasks|Search Tasks|Subgraph Counting|
|---|---|---|---|---|---|---|---|---|---|
||Method|Node count|Edge count Edge existence||Node degree|Connectivity Cycle check||Shortest path|Triangle counting|
||ZERO-SHOT[22]|21.7|12.4|44.5|14.0|84.9|76.0|11.5|1.5|
|Prompting|ZERO-COT[22]<br>FEW-SHOT[22]<br>COT[22]|14.6<br>25.3<br>27.6|9.4<br>12.0<br>12.8|33.5<br>36.8<br>42.8|10.4<br>17.4<br>29.2|73.5<br>79.4<br>45.2|32.3<br>37.4<br>58.0|33.6<br>22.7<br>38.6|12.7<br>3.0<br>8.1|
||COT-BAG[22]|26.9|12.5|37.3|28.0|45.2|52.1|40.4|8.1|
|Graph-based|GCN [42]<br>MPNN [26]<br>GIN [82]<br>GraphToken [61]|6.4<br>19.4<br>71.2<br>99.6|1.2<br>16.2<br>4.4<br>42.6|47.0<br>69.2<br>71.2<br>73.8|9.8<br>99.4<br>36.2<br>96.2|83.8<br>94.0<br>93.8<br>93.2|83.2<br>99.0<br>98.8<br>95.6|50.2<br>66.8<br>54.0<br>63.8|4.0<br>30.6<br>30.4<br>34.8|
||60M transformer-1K|100.0|100.0|67.6|31.5|92.9|97.1|57.4|33.4|
||60M transformer-100K|100.0|100.0|96.1|91.7|98.0|98.0|97.2|40.5|
|Ours|XXS transformer (FT)-1K|100.0|70.6|73.0|31.0|93.6|98.0|60.4|29.0|
||XS transformer (FT)-1K|100.0|73.2|98.6|50.6|96.6|96.8|60.0|28.6|
||12B transformer (FT)-1K|100.0|45.0|100.0|68.8|98.4|98.0|92.8|26.0|



Table 4: Comparison of various methods in different categories on graph reasoning tasks of GraphQA. Here, we categorize the tasks using the taxonomy proposed in Section 3. 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_Understanding_Transformer_Reasoning_Capabilities_via_Graph_Algorithms_images/nips_2024_Understanding_Transformer_Reasoning_Capabilities_via_Graph_Algorithms.pdf-0042-02.png)


**----- Start of picture text -----**<br>
Train Test<br>Node count Edge count Edge existence Node degree<br>1 1 1 1<br>0.8 0.8 0.9 0.8<br>0.6 0.6<br>0.8 0.6<br>0.4 0.4<br>0.7<br>0.4<br>0.2 0.2<br>0.6<br>0.2<br>0 0<br>10 [2] 10 [3] 10 [4] 10 [5] 10 [2] 10 [3] 10 [4] 10 [5] 10 [2] 10 [3] 10 [4] 10 [5] 10 [2] 10 [3] 10 [4] 10 [5]<br>Connectivity Cycle check Shortest path Triangle counting<br>1 1 1 1<br>0.8<br>0.8<br>0.6 0.8<br>0.6<br>0.4<br>0.4<br>0.2 0.6<br>0.2<br>0<br>0.9<br>10 [2] 10 [3] 10 [4] 10 [5] 10 [2] 10 [3] 10 [4] 10 [5] 10 [2] 10 [3] 10 [4] 10 [5] 10 [2] 10 [3] 10 [4] 10 [5]<br>Training Samples Training Samples Training Samples Training Samples<br>Accuracy<br>Accuracy<br>**----- End of picture text -----**<br>


Figure 8: Comparison of train and test scaling on all tasks. 

We independently evaluated GCN, MPNN, and GIN models on these tasks. We used the original architectures proposed in their respective papers and performed hyperparameter tuning on the GraphQADev dataset. 

## E.4 Further Experimental results 

Table 4 presents a comprehensive comparison of the graph reasoning capabilities across various baseline models and our proposed transformer architectures. The results highlight several key findings, which we summarize below: 

Transformers Exhibit Strong Performance on Graph-based Reasoning Problems. While transformers are not explicitly designed for graph reasoning tasks like graph-based models, they demonstrate surprisingly strong performance in this domain. The results of this study indicate that transformers, despite their versatility as a general architecture, can often match or even surpass specialized graph models on a variety of graph reasoning benchmarks. 

42 

Transformers Excel at Retrieval Tasks. As proved in Theorem 36, retrieval tasks can be solved by transformers. The obtained results confirm that such tasks are relatively easy for transformers as they obtained the full accuracy on most of such tasks. One exception here is the node degree task that GNNs outperform transformers but still transformers perform relatively well. We discuss why GNNs outperform transformers well for this task. 

Larger Transformers Excel at Solving Search Tasks. As discussed in Theorem 4, transformers are effective for search tasks, albeit requiring a larger number of parameters compared to retrieval tasks. This is empirically evident in the comparison between Transformer-1K and Transformer1K (pretrained). It’s worth noting that the pretrained transformer used here has 11 billion parameters, a significant increase from the 1 million parameters in Transformer-1K. 

Transformers Excel at Capturing Global Patterns. An interesting observation here is the performance gap between Transformers and GNNs across tasks with varying emphasis on local versus global graph structure. Notably: 

1. Local Structure: The node degree task, which relies heavily on local node information, is best handled by MPNN, a GNN-based model. 

2. Global Structure: In contrast, tasks like connectivity, triangle counting and shortest path, which require understanding global graph patterns, are dominated by Transformer models. Notably, vanilla Transformers achieve a remarkable 45% relative improvement over even much larger LLMs augmented with GNN-generated soft prompts (GraphToken) on the shortest path task. This showcases the exceptional ability of Transformers to capture long-range dependencies, a critical factor in understanding global graph structure. 

## E.4.1 Sample complexity ablations 

To develop a more comprehensive understanding of graph reasoning tasks learnability by small transformers, we train a variety of transformers for 1,000,000 steps on each task on a range of sample sizes. In Figure 8, we demonstrate how the model performance improves as a function of sample complexity. By doing so, we witness the relative hardness of each task in terms of the marginal benefit of new samples. 

- Edge count and node count are “easy” retrieval tasks that can be solved perfectly with as few as 100 samples. 

- Edge existence attains near-perfect train and test classification accuracy as the number of training samples approaches 100,000. 

- Connectivity, shortest path, and node degree demonstrate a sharp improvement in evaluation error as a function of the sample size. These models perfectly fit the training set in most sample size regimes, but yield a closer correspondence between training and testing error when trained on 100,000 samples. 

- Cycle check and triangle count have persistent gaps between training and testing error and overfit even in the large sample setting. 

43 

This figure "cpc_adj.jpeg" is available in "jpeg"� format from: http://arxiv.org/ps/2405.18512v1 

This figure "cpc_int.jpeg" is available in "jpeg"� format from: http://arxiv.org/ps/2405.18512v1 

This figure "ic.jpeg" is available in "jpeg"� format from: 

http://arxiv.org/ps/2405.18512v1 

