Published as a conference paper at ICLR 2024 

# - LESS IS MORE: ONE SHOT SUBGRAPH REASONING - ON LARGE SCALE KNOWLEDGE GRAPHS 

**Zhanke Zhou**[1] **Yongqi Zhang**[2] **Jiangchao Yao**[3] **Quanming Yao**[4] **Bo Han**[1] _[ †]_ 

1TMLR Group, Hong Kong Baptist University 

2The Hong Kong University of Science and Technology (Guangzhou) 3CMIC, Shanghai Jiao Tong University 4Tsinghua University {cszkzhou, bhanml}@comp.hkbu.edu.hk yzhangee@connect.ust.hk sunarker@sjtu.edu.cn qyaoaa@tsinghua.edu.cn 

## ABSTRACT 

To deduce new facts on a knowledge graph (KG), a link predictor learns from the graph structure and collects local evidence to find the answer to a given query. However, existing methods suffer from a severe scalability problem due to the utilization of the whole KG for prediction, which hinders their promise on largescale KGs and cannot be directly addressed by vanilla sampling methods. In this work, we propose the _one-shot-subgraph_ link prediction to achieve efficient and adaptive prediction. The design principle is that, instead of directly acting on the whole KG, the prediction procedure is decoupled into two steps, _i.e._ , (i) extracting _only one_ subgraph according to the query and (ii) predicting on this single, querydependent subgraph. We reveal that the non-parametric and computation-efficient heuristics Personalized PageRank (PPR) can effectively identify the potential answers and supporting evidence. With efficient subgraph-based prediction, we further introduce the automated searching of the optimal configurations in both data and model spaces. Empirically, we achieve promoted efficiency and leading performances on five large-scale benchmarks. The code is publicly available at: https://github.com/tmlr-group/one-shot-subgraph. 

## 1 INTRODUCTION 

A knowledge graph (KG) is graph-structural data with relational facts (Battaglia et al., 2018; Ji et al., 2020; Chen et al., 2020), based on which, one can conduct link prediction to deduce new facts from existing ones. The typical problem is to find the answer entity for the specific query, _e.g._ , to find the answer _Los Angeles_ to the query _(LeBron, lives_in, ?)_ . With continuous advances in recent years, the link prediction on KG has been widely applied in recommendation systems (Cao et al., 2019; Wang et al., 2019), online question answering (Huang et al., 2019), and drug discovery (Yu et al., 2021). 

The prediction system learns from the local structure of a KG, where existing methods can be generally summarized as two categories: (1) _semantic_ models that implicitly capture the local evidence through learning the low-dimensional embeddings of entities and relations (Bordes et al., 2013; Dettmers et al., 2017; Zhang et al., 2019; Wang et al., 2017); and (2) _structural_ models that explicitly explore the KG’s structure based on relational paths or graphs with recurrent neural networks (RNNs) or graph neural networks (GNNs) (Das et al., 2017; Schlichtkrull et al., 2018; Sadeghian et al., 2019; Vashishth et al., 2019; Teru et al., 2020; Qu et al., 2021; Zhu et al., 2021; Zhang & Yao, 2022). 

Although achieving leading performances, these structural models suffer from a severe scalability problem as the entire KG has been potentially or progressively taken for prediction. This inefficient manner hinders their application and optimization on large-scale KGs, _e.g._ , OGB (Hu et al., 2020). Thus, it raises an open question: _Is all the information necessary for prediction on knowledge graphs?_ Intuitively, only partial knowledge stored in the human brain is relevant to a given question, which is extracted by recalling and then utilized in the careful thinking procedure. Similarly, generating candidates and then ranking promising ones are common practices in large-scale recommendation systems with millions even billions of users (Cheng et al., 2016; Covington et al., 2016). These facts motivate us to conduct efficient link prediction with an effective sampling mechanism for KGs. 

> _†_ Correspondence to Bo Han (bhanml@comp.hkbu.edu.hk). 

1 

Published as a conference paper at ICLR 2024 

In this work, we propose the novel _one-shot-subgraph_ link prediction on a knowledge graph. This idea paves a new way to alleviate the scalability problem of existing KG methods from a data-centric perspective: decoupling the prediction procedure into two steps with a corresponding sampler and a predictor. Thereby, the prediction of a specific query is conducted by (i) _fast_ sampling of one query-dependent subgraph with the sampler and (ii) _slow_ prediction on the subgraph with predictor. 

Nevertheless, it is non-trivial to achieve efficient and effective link prediction on large-scale KGs due to the two major challenges. (1) _Sampling speed and quality_ : The fast sampling of the one-shot sampler should be capable of covering the essential evidence and potential answers to support the query. (2) _Joint optimization_ : The sampler and predictor should be optimized jointly to avoid trivial solutions and to guarantee the expressiveness and adaptivity of the overall model to a specific KG. 

To solve these challenges technically, we first implement the one-shot-subgraph link prediction by the non-parametric and computation-efficient Personalized PageRank (PPR), which is capable of effectively identifying the potential answers without requiring learning. With the efficient subgraphbased prediction, we further propose to search the data-adaptive configurations in both data and model spaces. We show it unnecessary to utilize the whole KG in inference; meanwhile, only a relatively small proportion of information ( _e.g._ , 10% of entities) is sufficient. Our main contributions are: 

- We conceptually formalize the new manner of _one-shot-subgraph link prediction_ on KGs (Sec. 3) and technically instantiate it with efficient heuristic samplers and powerful KG predictors (Sec. 4.1). 

- We solve a non-trivial and bi-level optimization problem of searching the optimal configurations in both data and model spaces (Sec. 4.2) and theoretically analyze the extrapolation power (Sec. 4.3). 

- We conduct extensive experiments on five large-scale datasets and achieve an average of 94 _._ 4% improvement in efficiency of prediction and 6 _._ 9% promotion in effectiveness of prediction (Sec. 5). 

## 2 PRELIMINARIES 

**Notations.** A knowledge graph is denoted as _G_ = ( _V, R, E_ ), where the entity set _V_ , the relation set _R_ , and factual edge set _E_ = _{_ ( _x, r, v_ ) : _x, v ∈V, r ∈R}_ . Here, a fact is formed as a triplet and denoted as ( _x, r, v_ ). Besides, a sampled _subgraph_ of _G_ is denoted as _Gs_ =( _Vs, Rs, Es_ ), satisfying that _Vs ⊆V, Rs ⊆R, Es ⊆E_ . The atomic problem of link prediction is denoted as a query ( _u, q,_ ?), _i.e._ , given the query entity _u_ and query relation _q_ , to find the answer entity _v_ , making ( _u, q, v_ ) valid. 

**Semantic models** encode entities and relations to low-dimensional entity embeddings _**H** V ∈_ R _[|V|×][D][v]_ and relation embeddings _**H** R ∈_ R _[|R|×][D][r]_ , where _Dv,Dr_ are dimensions. A scoring function _f_ _**θ**_ , _e.g._ , TransE (Bordes et al., 2013) or QuatE (Zhang et al., 2019), is necessary here to quantify the plausibility of a query triplet ( _u, q, v_ ) with the learned embeddings ( _**h** u,_ _**h** q,_ _**h** v_ ) as _f_ _**θ**_ : (R _[D][v] ,_ R _[D][r] ,_ R _[D][v]_ ) _�→_ R. 

**Efficient semantic models** aim to reduce the size of entity embeddings. NodePiece (Galkin et al., _f_ _**θ**_ 2022) proposes an anchor-based approach that obtains fixed-size embeddings as _G �−→_ _**H**_ **[ˆ]** _V ∈_ R _[N][×][D][v]_ and inference as ( _**H**_ **[ˆ]** _, G_ ) _�−−−−−→f_ _**θ** ,_ ( _u,q_ ) _**Y**_[ˆ] , where _**Y**_[ˆ] are the scores of candidate answers, and _N ≪|V|_ . Designed to reduce the embedding size, NodePiece cannot reduce the graph size for structural models. 

**Structural models** are based on relational _paths_ or _graphs_ for link prediction. Wherein the _path_ -based models, _e.g._ , MINERVA (Das et al., 2017), DRUM (Sadeghian et al., 2019), and RNNLogic (Qu et al., 2021), aim to learn probabilistic and logical rules and well capture the sequential patterns in KGs. The _graph_ -based models such as R-GCN (Schlichtkrull et al., 2018) and CompGCN (Vashishth et al., 2019) propagate the low-level entity embeddings among the neighboring entities to obtain high-level embeddings. Recent methods NBFNet (Zhu et al., 2021) and RED-GNN (Zhang & Yao, 2022) progressively propagate from _u_ to its neighborhood in a breadth-first-searching (BFS) manner. 

**Sampling-based structural models** adopt graph sampling approaches to decrease the computation complexity, which can be categorized into two-fold as follows. First, _subgraph-wise_ methods such as GraIL (Teru et al., 2020) and CoMPILE (Mai et al., 2021) extract enclosing subgraphs between query entity _u_ and each candidate answer _v_ . Second, _layer-wise_ sampling methods extract a subgraph for message propagation in each layer of a model. Wherein designed for node-level tasks on homogeneous graphs, GraphSAGE (Hamilton et al., 2017) and FastGCN (Chen et al., 2018) randomly sample neighbors around the query entity. While the KG sampling methods, _e.g._ , DPMPN (Xu et al., 2019), AdaProp (Zhang et al., 2023c), and AStarNet (Zhu et al., 2023), extract a learnable subgraph in _ℓ_ -th layer by the GNN model in _ℓ_ -th layer, coupling the procedures of sampling and prediction. 

2 

Published as a conference paper at ICLR 2024 

## 3 _One-shot-subgraph_ LINK PREDICTION ON KNOWLEDGE GRAPHS 

To achieve efficient link prediction, we conceptually design the _one-shot-subgraph_ manner that avoids directly predicting with the entire KG. We formalize this new manner in the following Def. 1. **Definition 1** (One-shot-subgraph Link Prediction on Knowledge Graphs) **.** _Instead of directly predicting on the original graph G, the prediction procedure is decoupled to two-fold: (1) one-shot sampling of a query-dependent subgraph and (2) prediction on this subgraph. The prediction pipeline becomes_ 

_g_ _**ϕ** ,_ ( _u,q_ ) _f_ _**θ** G �−−−−−→Gs �−→_ _**Y**_[ˆ] _,_ (1) 

_where the sampler g_ _**ϕ** generates only one subgraph Gs (satisfies |Vs|≪|V|, |Es|≪|E|) conditioned on the given query_ ( _u, q,_ ?) _. Based on subgraph Gs, the predictor f_ _**θ** outputs the final predictions_ _**Y**_[ˆ] _._ 

**Comparison with existing manners of prediction.** In brief, semantic models follow the manner of encoding the entire _G_ to the embeddings _**H**_ =( _**H** V ,_ _**H** R_ ) and prediction (inference) without _G_ , _i.e._ , 

_f_ _**θ** ,_ ( _u,q_ ) _f_ _**θ H** �−−−−−→_ _**Y**_[ˆ] _,_ s.t. _G �−→_ _**H** ,_ 

which is parameter-expensive, especially when encountering a large-scale graph with a large entity set. On the other hand, structural models adopt the way of learning and prediction with _G_ , _i.e._ , 

_f_ _**θ** ,_ ( _u,q_ ) _G �−−−−−→_ _**Y**_[ˆ] _,_ 

that directly or progressively conduct prediction with the entire graph _G_ . Namely, all the entities and edges can be potentially taken in the prediction of one query, which is computation-expensive. By contrast, our proposed one-shot prediction manner (Def. 1) enjoys the advantages 1 & 2 as follows. **Advantage 1** (Low complexity of computation and parameter) **.** _The one-shot-subgraph model is (1) computation-efficient: the extracted subgraph is much smaller than the original graph, i.e., |Vs|≪|V| and |Es|≪|E|; and (2) parameter-efficient: it avoids learning the expensive entities’ embeddings._ **Advantage 2** (Flexible propagation scope) **.** _The scope here refers to the range of message propagation starting from the query entity u. Normally, an L-layer structural method will propagate to the full L-hop neighbors of u. By contrast, the adopted one-shot sampling enables the bound of propagation scope within the extracted subgraph, where the scope is decoupled from the model’s depth L._ 

**Comparison with existing sampling methods.** Although promising to the scalability issue, existing sampling methods for structural models are not efficient or effective enough for learning and prediction on large-scale KGs. To be specific, the _random_ layer-wise sampling methods cannot guarantee the coverage of answer entities, _i.e._ , 1 ( _v ∈Vu_ ). By contrast, the _learnable_ layer-wise sampling methods extract the query-dependent subgraph _Gs_[(] _[ℓ]_[)] in _ℓ_ -th layer via the GNN model _f_ _**θ**_[(] _[ℓ]_[)] in _ℓ_ -th layer as 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2024_Less_is_More_One-shot_Subgraph_Reasoning_on_Large-scale_Knowledge_Graphs_images/iclr_2024_Less_is_More_One-shot_Subgraph_Reasoning_on_Large-scale_Knowledge_Graphs.pdf-0003-11.png)


coupling the sampling and prediction procedures that (1) are bundled with specific architectures and (2) with extra computation cost in the layer-wise sampling operation. Besides, the subgraph-wise sampling methods extract the enclosing subgraphs between query entity _u_ and each candidate answer _v ∈V_ , and then independently reason on each of these subgraphs to obtain the final prediction _**Y**_[ˆ] as 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2024_Less_is_More_One-shot_Subgraph_Reasoning_on_Large-scale_Knowledge_Graphs_images/iclr_2024_Less_is_More_One-shot_Subgraph_Reasoning_on_Large-scale_Knowledge_Graphs.pdf-0003-13.png)


Note these approaches are extremely expensive on large-scale graphs, as each candidate ( _u, v_ ) corresponds to a subgraph to be scored. By contrast, one-shot sampling manner enjoys the advantage 3. **Advantage 3** (High efficiency in subgraph sampling) **.** _Our proposed prediction manner requires_ _**only one subgraph** for answering_ _**one query** , which is expected to cover all the potential answers and supporting facts. Notably, this query-specific subgraph is extracted in a_ _**one-shot** and_ _**decoupled** manner that does not involve the predictor, reducing the computation cost in subgraph sampling._ 

## 4 INSTANTIATING THE ONE-SHOT-SUBGRAPH LINK PREDICTION 

Note that it is non-trivial to achieve Def. 1, wherein (i) the implementation of sampler, (ii) the architecture of predictor, and (iii) the method to optimize these two modules need to be figured out. Here, the major challenge lies in the sampler, which is required to be efficient, query-dependent, and local-structure-preserving. In this section, we elaborate on the detailed implementation (Sec. 4.1), set up a bi-level problem for optimization (Sec. 4.2), and investigate the extrapolation power (Sec. 4.3). 

3 

Published as a conference paper at ICLR 2024 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2024_Less_is_More_One-shot_Subgraph_Reasoning_on_Large-scale_Knowledge_Graphs_images/iclr_2024_Less_is_More_One-shot_Subgraph_Reasoning_on_Large-scale_Knowledge_Graphs.pdf-0004-01.png)


**----- Start of picture text -----**<br>
Original KG 𝒢 Query-dependent subgraph 𝒢! Reasoning scores 𝒀 [%]<br>𝑞 v<br>u<br>One-shot Subgraph Reasoning Optimization Objective<br>(𝑢, 𝑞, ? )<br>𝒢 𝒢# $𝒀 maxℙ𝒢 * ℳ 𝑓𝜽 [∗] 𝒢) , 𝑣,<br>𝑔𝝓 𝑓𝜽 ",$,% ∈ℰ ["#$]<br>s. t. 𝒢) ∼ℙ𝒢, 𝜽 [∗] = arg min𝜽 * ℒ,-. 𝑓𝜽 𝒢) , 𝑣.<br>Fast sampling Slow reasoning ",$,% ∈ℰ [&'#()]<br>**----- End of picture text -----**<br>


Figure 1: The proposed framework of one-shot-subgraph link prediction. Specifically, (1) the sampler _g_ _**ϕ**_ extracts a subgraph _Gs_ from the whole graph _G_ with regard to the given query, and (2) the predictor _f_ _**θ**_ conducts deliberative prediction on the extracted subgraph _Gs_ and obtains the final predictions _**Y**_[ˆ] . 

## 4.1 REALIZATION: THREE-STEP PREDICTION WITH PERSONALIZED PAGERANK 

**Overview.** As illustrated in Fig. 1, the three key steps of our method are **(1)** generating the sampling distribution P _G_ by sampler _g_ _**ϕ**_ , **(2)** sampling a subgraph from the distribution as _Gs ∼_ P _G_ with top- _K_ entities and edges, and **(3)** predicting on the subgraph _Gs_ and acquiring the final prediction _**Y**_[ˆ] by predictor _f_ _**θ**_ . The three-step procedure is summarized in Algorithm 1 and elaborated on as follows. 

**Step-1. Generate sampling distribution.** Previous studies show that _v_ is generally near to _u_ (Zhu et al., 2021; Zhang & Yao, 2022), and the relational paths connecting _u_ and _v_ that support the query also lie close to _u_ (Das et al., 2017; Sadeghian et al., 2019). To efficiently capture the local evidence of _u_ , we choose the heuristic Personalized PageRank (PPR) (Page et al., 1999; Jeh & Widom, 2003) as the sampling indicator. Note that PPR is not only _efficient_ for its non-parametric nature but also _query-dependent_ and _local-structure-preserving_ for its single-source scoring that starts from _u_ . 

Specifically, PPR starts propagation from _u_ to evaluate the importance of each neighbor of _u_ and generates the PageRank scores as the sampling probability that encodes the local neighborhood of the query entity _u_ . Besides, it can also preserve the locality and connectivity of subgraphs by leveraging the information from a large neighborhood. Given a query entity _u_ , we obtain the probability _**p** ∈_ R _[|V|]_ 

Non-parametric indicator : _**p**_[(] _[k]_[+1)] _← α ·_ _**s**_ + (1 _− α_ ) _·_ _**D**[−]_[1] _**A** ·_ _**p**_[(] _[k]_[)] _,_ (2) by iteratively updating the scores up to _K_ =100 steps to approximate the converged scores efficiently. Here, the initial score _**p**_[(0)] = _**s**_ = 1 ( _u_ ) _∈{_ 0 _,_ 1 _}[|V|]_ indicates the query entity _u_ to be explored. The two-dimensional degree matrix _**D** ∈_ R _[|V|×|V|]_ and adjacency matrix _**A** ∈{_ 0 _,_ 1 _}[|V|×|V|]_ together work as the transition matrix, wherein _**A** ij_ =1 means an edge ( _i, r, j_ ) _∈E_ and _**D** ij_ = degree( _vi_ ) if _i_ = _j_ else _**D** ij_ =0. The damping coefficient _α_ (= 0 _._ 85 by default) controls the differentiation degree. 

**Step-2. Extract a subgraph.** Based on the PPR scores _**p**_ (Eqn. 2), the subgraph _Gs_ =( _Vs, Es, Rs_ ) (where _Rs_ = _R_ ) is extracted with the most important entities and edges. Denoting the sampling ratios of entities and edges as _rV[q][, r] E[q][∈]_[(0] _[,]_[ 1]][ that depend on the query relation] _[ q]_[, we sample] _[ |V][s][|]_[=] _[r] V[q][×|V|]_ entities and _|Es|_ = _rE[q][×|E|]_[ edges from the full graph] _[ G]_[.][With the][ TopK][(] _[D, P, K]_[)][ operation that picks] up top- _K_ elements from candidate _D w.r.t._ probability _P_ , the entities _Vs_ and edges _Es_ are given as Entity Sampling: _Vs ←_ TopK _V,_ _**p** , K_ = _rV[q][×|V|] ,_ � � (3) Edge Sampling: _Es ←_ TopK _E, {_ _**p** x ·_ _**p** o_ : _x, o ∈Vs,_ ( _x, r, o_ ) _∈E}, K_ = _rE[q][×|E|] ._ � � 

**Step-3. Reason on the subgraph.** From the model’s perspective, we build the configuration space of the predictor and further utilize the advantages of existing structural models introduced in Sec. 2. Three query-dependent message functions MESS( _·_ ) are considered, including DRUM, NBFNet, and RED-GNN, which are elaborated in Appendix. B. Note the effective message is propagated from _u_ to the sampled entities _o ∈Vs_ . Generally, the layer-wise updating of representations is formulated as 

4 

Published as a conference paper at ICLR 2024 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2024_Less_is_More_One-shot_Subgraph_Reasoning_on_Large-scale_Knowledge_Graphs_images/iclr_2024_Less_is_More_One-shot_Subgraph_Reasoning_on_Large-scale_Knowledge_Graphs.pdf-0005-01.png)


wheremanner. 1 (After a _o_ = _u_ ) _L_ is the indicator function that only labels the query entity-layer propagation, the predictor outputs the final score _u_ _**y**_ ˆ _o_ in a query-dependentof each entity _o ∈Vs_ ˆ based on their representations _**h**_[(] _o[L]_[)] as _**y** o_ = Readout( _**h**_[(] _o[L]_[)] _,_ _**h**_[(] _u[L]_[)][)] _[ ∈]_[R][.][The][loss][function] _[L] cls_ adopted in the training phase is the commonly-used binary cross-entropy loss on all the sampled entities. Namely, _Lcls_ = _−_[�] _o∈Vs[y][o]_[ log(ˆ] _**[y]**[o]_[)+(1] _[ −][y][o]_[) log(1] _[−]_ _**[y]**_[ˆ] _[o]_[)][, where] _[ y][o]_[ =1][ if] _[ o]_[=] _[v]_[ else] _[ y][o]_[ =0][.] 

**Algorithm 1** One-shot-subgraph Link Prediction on Knowledge Graphs 

**Require:** KG _G_ = ( _V, R, E_ ), degree matrix _**D** ∈_ R _[|V|×|V|]_ , adjacency matrix _**A** ∈{_ 0 _,_ 1 _}[|V|×|V|]_ , damping coefficient _α_ , maximun PPR iterations _K_ , query ( _u, q,_ ?), sampler _g_ _**ϕ**_ , predictor _f_ _**θ**_ . 1: # Step-1. Generate sampling distribution 2: initialize _**s** ←_ 1 ( _u_ ) _,_ _**p**_[(0)] _←_ 1 ( _u_ ). 3: **for** _k_ = 1 _. . . K_ **do** 4: _**p**_[(] _[k]_[+1)] _← α ·_ _**s**_ + (1 _− α_ ) _·_ _**D**[−]_[1] _**A** ·_ _**p**_[(] _[k]_[)] . 5: **end for** 6: # Step-2. Extract a subgraph _Gs_ 7: _Vs ←_ TopK( _V,_ _**p** , K_ = _rV[q][×|V|]_[)][.] 8: _Es ←_ TopK( _E, {_ _**p** u ·_ _**p** v_ : _u, v ∈Vs,_ ( _u, r, v_ ) _∈E}, K_ = _rE[q][×|E|]_[)][.] 9: # Step-3. Reason on the subgraph 10: initialize representations _**h**_[(0)] _o ←_ 1 ( _o_ = _u_ ). 11: **for** _ℓ_ = 1 _. . . L_ **do** 12: _**h**_[(] _o[ℓ]_[)] _[←]_[DROPOUT][(][ACT][(][AGG] _[{]_[MESS][(] _**[h]**_[(] _x[ℓ][−]_[1)] _,_ _**h**_[(] _r[ℓ][−]_[1)] _,_ _**h**_[(] _o[ℓ][−]_[1)] ):( _x, r, o_ ) _∈Es}_ )). 13: **end for** 14: **return** Prediction _**y**_ ˆ _o_ = Readout( _**h**_[(] _o[L]_[)] _,_ _**h**_[(] _u[L]_[)][)][ for each entity] _[ o][ ∈V] s_[.] 

- 4.2 OPTIMIZATION: EFFICIENT SEARCHING FOR DATA-ADAPTIVE CONFIGURATIONS 

**Search space.** Note that hyperparameters ( _rV[q][, r] E[q]_[)][and] _[L]_[play][important][roles][in][Algorithm][1][.] Analytically, a larger subgraph with larger _rV[q][, r] E[q]_[does not indicate a better performance, as more] irrelevant information is also covered. Besides, a deeper model with a larger _L_ can capture more complex patterns but is more likely to suffer from the over-smoothing problem (Oono & Suzuki, 2019). Overall, the ( _rV[q][, r] E[q]_[)][ are for sampler’s hyper-parameters] _**[ ϕ]**_[hyper][.][In addition to] _[ L]_[, predictor’s] hyper-parameters _**θ**_ hyper contain several intra-layer or inter-layer designs, as illustrated in Fig. 2(a). 

**Search problem.** Next, we propose the bi-level optimization problem to adaptively search for the optimal configuration ( _**ϕ**[∗]_ hyper _[,]_ _**[ θ]**_ hyper _[∗]_[)][ of design choices on a specific KG, namely,] 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2024_Less_is_More_One-shot_Subgraph_Reasoning_on_Large-scale_Knowledge_Graphs_images/iclr_2024_Less_is_More_One-shot_Subgraph_Reasoning_on_Large-scale_Knowledge_Graphs.pdf-0005-08.png)


where the performance measurement _M_ can be Mean Reciprocal Ranking (MRR) or Hits@k. Note the non-parametric sampler _g_ _**ϕ**_ only contains hyper-parameters _**ϕ**_ hyper. As for predictor _f_ _**θ**_ , its _**θ**_ = ( _**θ**_ hyper _,_ _**θ**_ learn) also includes learnable _**θ**_ learn that _**θ**_ learn _[∗]_[=arg min] _**[θ]**_ learn _[L][cls]_[(] _[f]_[(] _**[θ]**_ hyper _[,]_ _**[θ]**_ learn[)] _[, g]_ _**ϕ**_[ ¯] hyper _[,][ E]_[train][)][.] 

**Search algorithm.** Directly searching on both data and model spaces is expensive due to the large space size and data scale. Hence, we split the search into two sub-processes as Fig. 2(b), _i.e._ , 

- First, we freeze the sampler _g_ _**ϕ**_ ¯ (with constant _**ϕ**_ hyper) to search for the optimal predictor _f_ _**θ** ∗_ with (1) the hyper-parameters optimization for _**θ**_ hyper _[∗]_[and (2) the stochastic gradient descent for] _**[ θ]**_ learn _[∗]_[.] 

- Then, we freeze the predictor _f_ _**θ** ∗_ and search for the optimal sampler _g_ _**ϕ** ∗_ , simplifying to pure hyperparameters optimization for _**ϕ**[∗]_ hyper[in a zero-gradient manner with low computation complexity.] 

Specifically, we follow the sequential model-based Bayesian Optimization (BO) (Bergstra et al., 2013; Hutter et al., 2011) to obtain _**ϕ**[∗]_ hyper[and] _**[ θ]**_ hyper _[∗]_[.][Random forest (RF) (][Breiman][,][ 2001][) is chosen] as the surrogate model because it has a stronger power for approximating the complex and discrete 

5 

Published as a conference paper at ICLR 2024 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2024_Less_is_More_One-shot_Subgraph_Reasoning_on_Large-scale_Knowledge_Graphs_images/iclr_2024_Less_is_More_One-shot_Subgraph_Reasoning_on_Large-scale_Knowledge_Graphs.pdf-0006-01.png)


**----- Start of picture text -----**<br>
2$3%$)$1-"-(,1<br>41(-("5(6"-(,1 𝑯 [(#)] Input 𝒢! 𝝓-./)+ 𝜽()*+,, 𝜽-./)+<br>! !"# $%&'$()* Search<br>! +'$%,(-)%$ $%&'$()* !"#$%&'()$ Layer-1 prop. 7$))"8$ 9/1.-(,1 :7;**< ! ./012334 3563',4 0.78 Search 𝝓-./)+ 𝜽∗()*+,, 𝜽∗-./)+ with BO<br>0,11$.-(,1*+,%-./- 𝑯 [(%)] =.-(>"-$ 9/1.-(,1 :=0?< ! 9:'),(,;4<.'$=4<>%)? with BO 𝝓∗-./)+ 𝜽∗()*+,, 𝜽∗-./)+<br>… =88%$8"-(,1 9/1.-(,1 :=@@< ! 8%@4<8'%)4<A=B !.#$%&'$%/,0(%'+ ('")*&123<br>Layer-L prop. A%,3,/- 2"-$<br>! C!4 !DEF (𝑢, 𝑞, ? )<br>2$3%$)$1-"-(,1 A(B$1)(,1 *(6$ 𝒢 𝒢# $𝒀 𝑌<br>0,1."-$1"-(,1 𝑯 [(&)] ! #G4<HI4<GJ4<#IK 𝑔𝝓 𝑓𝜽<br>2$"C,/- 9/1.-(,1<br>! 0-, M+-:=N,! L()'%+ Readout Output 𝒀 [%] ℰ [1+*2,] = ℰ [34#] ∪ ℰ [56)+.]<br>!"#$%&'$('")*&$(+"*'$,-$𝜽/0123 !*#$%&'$'43'$(+51%$%)1*6<br>**----- End of picture text -----**<br>


Figure 2: Illustrations of the optimization procedure. Note the predictor in (a) is with hyper-parameters and learnable parameters in each layer’s propagation, and the _**H**_[(] _[L]_[)] indicates the representations in _L_ -th layer. By contrast, the sampler only contains hyper-parameters as it does not require learning. curvature (Grinsztajn et al., 2022), compared with other common surrogates, _e.g._ , Gaussian Process (GP) (Williams & Rasmussen, 1995) or Multi-layer Perceptron (MLP) (Gardner & Dorling, 1998). 

**Acceleration in searching.** We adopt a data split trick that balances observations and predictions. It saves time as the training does not necessarily traverse all the training facts. Specifically, partial training facts are sampled as training queries while the others are treated as observations, _i.e._ , we randomly separate the training facts into two parts as _E_[train] = _E_[obs] _∪E_[query] , where the overall prediction system _f_ _**θ** ◦ g_ _**ϕ**_ takes _E_[obs] as input and then predicts _E_[query] (see Fig. 2(c)). Here, the split ratio _r_[split] is to balance the sizes of these two parts as _r_[split] = _[|E]_[obs] _[|] /|E_[query] _|_ . Thus, the training becomes _**θ**[∗]_ = arg min _**θ**_ Σ( _u,q,v_ ) _∈E_ query _Lcls_ � _f_ _**θ**_ ( _Gs_ ) _, v_ � with the split query edges _E_[query] , where _Gs_ = _g_ _**ϕ**_ ( _E_[obs] _, u, q_ ) with the split observation edges _E_[obs] . More technical details can be found in the Appendix. B. 

## 4.3 THEORY: THE EXTRAPOLATION POWER OF ONE-SHOT-SUBGRAPH LINK PREDICTION 

Further, we investigate the extrapolation power of link prediction across graph scales, _i.e._ , training and inference on different scales of graphs. For example, training on small subgraphs _Gs_[train] and testing on large subgraphs _Gs_[test][, where the ratio of entities] _[|V] s_[test] _[|] /|Vs_[train] _| ≫_ 1. This scenario is practical for recommendation systems on social networks that can encounter much larger graphs in the testing phase. Intuitively, training on smaller _Gs_[train] can save time for its faster convergence on subgraphs (Shi et al., 2023), while predicting on larger _Gs_[test] might gain promotion for more support of facts in _Gs_[test][.] Nonetheless, the Theorem 1 below proves that link prediction can become unreliable as the test graph grows. That is, if we use a _small_ subgraph for training and a _large_ subgraph for testing, the predictor will struggle to give different predictions within and across sampling distributions by _g_ , even when these probabilities are arbitrarily different in the underlying graph model. Our empirical results in Fig. 4 support this theoretical finding. Hence, it is necessary to strike a balance of subgraphs’ scale. **Theorem 1.** _Let Gs[train] ∼_ P _G and Gs[test] ∼_ P _G be the training and testing graphs that are sampled from distributiondecision of fact_ ( P _u, q, vG. Consider any two test entities_ ) _with the predictor f_ _**θ** , i.e., u, v_ _**y**_ ˆ _v_ = _∈Vf_ _**θ**_ ( _s[test] Gs[test][, for which we can make a prediction]_[)] _[v]_[=] _[τ][.][Let][ G][test][ be large enough to] satisfy √|Vs[test][|] /[√]_ log(2 _|Vs[test][|][/p]_[)] _[≥]_[4] ~~_√_~~ 2 _/d_ min _, where d_ min _is the constant of graphon degree (Diaconis & Janson, 2007). Then, for an arbitrary threshold τ ∈_ [0 _,_ 1] _, the testing subgraph Gs[test] satisfies that_ 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2024_Less_is_More_One-shot_Subgraph_Reasoning_on_Large-scale_Knowledge_Graphs_images/iclr_2024_Less_is_More_One-shot_Subgraph_Reasoning_on_Large-scale_Knowledge_Graphs.pdf-0006-06.png)


_where the underlying generative function of graph signal g ∈ L[∞] is with the essential supreme norm as in (Maskey et al., 2022; Zhou et al., 2022). The p, C_ 1 _, C_ 2 _are constants and depend on M[train] where_ min( _supp_ ( _|Vs[test][|]_[))] _[ ≫][M][ train]_[ = max(] _[supp]_[(] _[|V] s[train] |_ )) _. It means any test graph can be much larger than the largest possible training graph, and supp indicates the support of a distribution. Then, if u and v are isomorphic in topology and with the same representations, we have a probability at least_ 1 _−_[�] _[L] ℓ_ =1[2(] _[|]_ _**[h]**_[(] _[ℓ]_[)] _[|]_[+1)] _[p][ with hidden size][ |]_ _**[h]**_[(] _[ℓ]_[)] _[|][ that the same predictions can be obtained] whether u, v are generated by the same or distinct g. The detailed proof can be found in Appendix. A._ 

6 

Published as a conference paper at ICLR 2024 

Table 1: Empirical results of WN18RR, NELL-995, YAGO3-10 datasets. Best performance is indicated by the **bold face** numbers, and the underline means the second best. “–” means unavailable results. “H@1” and “H@10” are short for Hit@1 and Hit@10 (in percentage), respectively. 

|type|models|WN18RR<br>MRR_↑_H@1_↑_H@10_↑_|NELL-995<br>MRR_↑_H@1_↑_H@10_↑_|YAGO3-10<br>MRR_↑_H@1_↑_H@10_↑_|
|---|---|---|---|---|
||||||
|Semantic Models|ConvE<br>QuatE<br>RotatE|0.427<br>39.2<br>49.8<br>0.480<br>44.0<br>55.1<br>0.477<br>42.8<br>57.1|0.511<br>44.6<br>61.9<br>0.533<br>46.6<br>64.3<br>0.508<br>44.8<br>60.8|0.520<br>45.0<br>66.0<br>0.379<br>30.1<br>53.4<br>0.495<br>40.2<br>67.0|
|Structural Models|||||
||MINERVA<br>DRUM<br>RNNLogic<br>CompGCN<br>DPMPN<br>NBFNet<br>RED-GNN|0.448<br>41.3<br>51.3<br>0.486<br>42.5<br>58.6<br>0.483<br>44.6<br>55.8<br>0.479<br>44.3<br>54.6<br>0.482<br>44.4<br>55.8<br>0.551<br>49.7<br>**66.6**<br>0.533<br>48.5<br>62.4|0.513<br>41.3<br>63.7<br>0.532<br>46.0<br>**66.2**<br>0.416<br>36.3<br>47.8<br>0.463<br>38.3<br>59.6<br>0.513<br>45.2<br>61.5<br>0.525<br>45.1<br>63.9<br>0.543<br>47.6<br>65.1|–<br>–<br>–<br>0.531<br>45.3<br>67.6<br>0.554<br>50.9<br>62.2<br>0.489<br>39.5<br>58.2<br>0.553<br>48.4<br>67.9<br>0.550<br>47.9<br>68.3<br>0.559<br>48.3<br>68.9|
||||||
||**one-shot-subgraph**|**0.567**<br>**51.4**<br>**66.6**|**0.547**<br>**48.5**<br>65.1|**0.606**<br>**54.0**<br>**72.1**|



Table 2: Empirical results of two OGB datasets (Hu et al., 2020) with regard to official leaderboards. 

|type<br>models|OGBL-BIOKG<br>Test MRR_↑_Valid MRR_↑_<br>#Params_↓_|OGBL-WIKIKG2<br>Test MRR_↑_Valid MRR_↑_<br>#Params_↓_|
|---|---|---|
||||
|Semantic Models<br>TripleRE<br>AutoSF<br>PairRE<br>ComplEx<br>DistMult<br>RotatE<br>TransE|0.8348<br>0.8360<br>469,630,002<br>0.8309<br>0.8317<br>93,824,000<br>0.8164<br>0.8172<br>187,750,000<br>0.8095<br>0.8105<br>187,648,000<br>0.8043<br>0.8055<br>187,648,000<br>0.7989<br>0.7997<br>187,597,000<br>0.7452<br>0.7456<br>187,648,000|0.5794<br>0.6045<br>500,763,337<br>0.5458<br>0.5510<br>500,227,800<br>0.5208<br>0.5423<br>500,334,800<br>0.4027<br>0.3759<br>1,250,569,500<br>0.3729<br>0.3506<br>1,250,569,500<br>0.4332<br>0.4353<br>1,250,435,750<br>0.4256<br>0.4272<br>1,250,569,500|
||||
|Structural Models **one-shot-subgraph**|**0.8430**<br>**0.8435**<br>**976,801**|**0.6755**<br>**0.7080**<br>**6,831,201**|



Table 3: Coverage Ratio of different heuristics. **Bold face** numbers indicate the best results in column. 

|heuristics|WN18RR<br>_rq_<br>_V_ =0_._1 _rq_<br>_V_ =0_._2 _rq_<br>_V_ =0_._5|NELL-995<br>_rq_<br>_V_ =0_._1 _rq_<br>_V_ =0_._2 _rq_<br>_V_ =0_._5|YAGO3-10<br>_rq_<br>_V_ =0_._1 _rq_<br>_V_ =0_._2 _rq_<br>_V_ =0_._5|
|---|---|---|---|
|||||
|Random Sampling (RAND)<br>PageRank (PR)<br>Random Walk (RW)<br>Breadth-frst-searching (BFS)<br>Personalized PageRank (PPR)|0.100<br>0.200<br>0.500<br>0.278<br>0.407<br>0.633<br>0.315<br>0.447<br>0.694<br>0.818<br>0.858<br>0.898<br>**0.876**<br>**0.896**<br>**0.929**|0.100<br>0.200<br>0.500<br>0.405<br>0.454<br>0.603<br>0.522<br>0.552<br>0.710<br>0.872<br>0.935<br>0.982<br>**0.965**<br>**0.977**<br>**0.987**|0.100<br>0.200<br>0.500<br>0.340<br>0.432<br>0.694<br>0.449<br>0.510<br>0.681<br>0.728<br>0.760<br>0.848<br>**0.943**<br>**0.957**<br>**0.973**|



## 5 EXPERIMENTS 

In this section, we empirically verify the effectiveness of the proposed framework. The major experiments are conducted with PyTorch (Paszke et al., 2017) and one NVIDIA RTX 3090 GPU. The OGB datasets are run with one NVIDIA A100 GPU. We use five benchmarks with more than ten thousand entities (see Tab. 11), including WN18RR (Dettmers et al., 2017), NELL-995 (Xiong et al., 2017), YAGO3-10 (Suchanek et al., 2007), OGBL-BIOKG, and OGBL-WIKIKG2 (Hu et al., 2020). 

**Metrics.** We adopt the filtered ranking-based metrics for evaluation, _i.e._ , mean reciprocal ranking (MRR) and Hit@ _k_ ( _i.e._ , both Hit@1 and Hit@10), following (Bordes et al., 2013; Teru et al., 2020; Wang et al., 2017; Zhu et al., 2021). For both metrics, a higher value indicates a better performance. 

**Main Results.** As results shown in Tab. 1 and Tab. 2, our one-shot-subgraph link prediction method achieves leading performances on all five large-scale benchmarks over all the baselines. Especially on the largest OGBL-WIKIKG2 dataset, a 16 _._ 6% improvement in Test MRR can be achieved. Note the results attribute to _a deep GNN (high expressiveness) and small subgraphs (essential information)_ extracted by sampling 10% of entities on average for answering specific queries. Which means, it is _unnecessary_ to utilize the whole KG in link prediction; meanwhile, only a small proportion of entities and facts are essential for answering specific queries that can be quickly identified by the PPR heuristics. In what follows, we conduct an in-depth analysis of the properties of the proposed method. 

**The Sampling Distribution.** We empirically evaluate to what extent the entities relevant to a specific query can be identified by heuristics, _e.g._ , BFS, RW, and PPR. We quantify their power of identifying potential answers via the metric of Coverage Ratio CR =[1] _/|E_[test] _|_[�] ( _u,q,v_ ) _∈E_[test][ I] _[{][v][∈V][s][}]_[,] _[ i.e.]_[, the ratio] of covered answer entities that remain in the set of sampled entities _Vs_ . As shown in Tab. 3 and Fig. 3, PPR gets a much higher CR and notably outperforms other heuristics in identifying potential answers. 

7 

Published as a conference paper at ICLR 2024 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2024_Less_is_More_One-shot_Subgraph_Reasoning_on_Large-scale_Knowledge_Graphs_images/iclr_2024_Less_is_More_One-shot_Subgraph_Reasoning_on_Large-scale_Knowledge_Graphs.pdf-0008-01.png)


**----- Start of picture text -----**<br>
WN18RR NELL-995 YAGO3-10<br>1.0 1.0 1.0<br>0.8 0.8 0.8<br>0.6 0.6 0.6<br>0.4 0.4 0.4<br>RAND BFS RAND BFS RAND BFS<br>0.2 PR PPR 0.2 PR PPR 0.2 PR PPR<br>RW RW RW<br>0.0 0.0 0.0<br>0.0 0.2 0.4 0.6 0.8 1.0 0.0 0.2 0.4 0.6 0.8 1.0 0.0 0.2 0.4 0.6 0.8 1.0<br>Ratio of sampled entities Ratio of sampled entities Ratio of sampled entities<br>Figure 3: Coverage Ratio (CR) of different heuristics (CR = [1] /|E [test] | [�] ( u,q,v ) ∈E [test][ I] [{][v] [∈V][s][}] [).]<br>WN18RR NELL-995 YAGO3-10<br>1.0 1.0 0.505 1.0 0.610<br>0.9 0.567 0.9 0.500 0.9<br>0.8 0.564 0.8 0.8 0.600<br>0.7 0.7 0.495 0.7<br>0.6 0.561 0.6 0.490 0.6 0.590<br>0.5 0.5 0.5<br>0.4 0.558 0.4 0.485 0.4 0.580<br>0.3 0.555 0.3 0.480 0.3<br>0.2 0.2 0.2 0.570<br>0.1 0.552 0.1 0.475 0.1<br>0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0<br>Ratio of sampled edges Ratio of sampled edges Ratio of sampled edges<br>Coverage Ratio Coverage Ratio Coverage Ratio<br>Ratio of sampled entities Ratio of sampled entities Ratio of sampled entities<br>**----- End of picture text -----**<br>


Figure 4: Heatmaps of validate MRR (the higher, the better) _w.r.t. rV[q]_[and] _[ r] E[q]_[on three benchmarks.] 

Table 4: Comparison of effectiveness with regard to subgraph sampling. 

|#layers (_L_)<br>_rq_<br>_V_<br>_rq_<br>_E_|WN18RR<br>MRR<br>H@1<br>H@10|NELL-995<br>MRR<br>H@1<br>H@10|YAGO3-10<br>MRR<br>H@1<br>H@10|
|---|---|---|---|
|||||
|8<br>1.0<br>1.0<br>8<br>0.1<br>1.0|Out of memory<br>**0.567**<br>**51.4**<br>**66.6**|Out of memory<br>**0.547**<br>**48.5**<br>**65.1**|Out of memory<br>**0.606**<br>**54.0**<br>**72.1**|
|||||
|6<br>1.0<br>1.0<br>6<br>0.1<br>1.0|0.543<br>49.2<br>64.3<br>0.566<br>51.2<br>66.5|0.519<br>45.3<br>62.7<br>0.540<br>48.0<br>63.8|0.538<br>46.9<br>66.0<br>0.599<br>53.1<br>71.8|
|||||
|4<br>1.0<br>1.0<br>4<br>0.1<br>1.0|0.513<br>46.6<br>59.8<br>0.523<br>47.7<br>60.5|0.518<br>45.4<br>61.5<br>0.538<br>47.4<br>63.4|0.542<br>47.6<br>66.1<br>0.589<br>52.2<br>70.4|



**Ablation Study. (1) Training with varying scales and layers:** We train the predictor from scratch with various scales of subgraphs and the number of layers. As can be seen from Tab. 4, involving all the entities with _rV[q]_[=1] _[.]_[0][ degenerates the prediction, as too many irrelevant entities are covered.] Besides, a deeper predictor with a larger _L_ consistently brings better results. These observations enlighten us to learn a deeper predictor with small subgraphs. **(2) Training with different heuristics:** We replace the PPR sampling module with four other common heuristics. However, as shown in Tab. 5, their final prediction performances are outperformed by PPR. **(3) Test-time extrapolation power across scales:** As in Theorem. 1, we evaluate the extrapolation power by generalizing to various scales of subgraphs that are different from the scale of training graphs, _e.g._ , the whole graph _rV[q]_[=] _[r] E[q]_[=1] _[.]_[0][.][As shown in Fig.][ 4][, the predictor also suffers when prediction with more irrelevant] entities, especially with larger _rV[q]_[,][while the generally good cases are a lower] _[r] V[q]_[to focus on the] relevant entities and a high _rE[q]_[to preserve the local structure (of head] _[ u]_[) within the sampled entities.] 

**Training and Inference Efficiency.** Next, we conduct an efficiency study to investigate the improvement of efficiency brought by the proposed one-shot-subgraph link prediction framework. The running time and GPU memory of an 8-layer GNN are summarized in Tab. 7. As can be seen, a notable advantage of decoupling is that it has less computing cost in both terms of less running time and also less memory cost. Particularly, on the YAGO3-10 dataset, the existing GNN-based methods will run out of memory with a deep architecture. However, with the subgraph sampling of lower ratios of _rV[q]_[and] _[ r] E[q]_[, the learning and prediction of GNNs become feasible that is with less memory] cost and achieving state-of-the-art performance. Hence, we show that our method is effective and also efficient that it supports the learning and prediction of deep GNNs on large-scale knowledge graphs. 

Besides, we provide a detailed efficiency comparison between our method (with 10% entities) and the original implementation (with 100% entities) on two SOTA methods, NBFNet and RED-GNN. Tab. 6 shows that the training time is significantly reduced when learning with our method. Notably, on the YAGO3-10 dataset, 94 _._ 3% and 94 _._ 5% of training time (for one epoch) can be saved for NBFNet and RED-GNN, respectively. Besides, our method boosts the performance as advantage 2, where the performance improvement can come from a deeper GNN and a smaller observation graph (detailed analysis in Appendix. D.3). Full evaluations and more discussions are elaborated in the Appendix. C. 

8 

Published as a conference paper at ICLR 2024 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2024_Less_is_More_One-shot_Subgraph_Reasoning_on_Large-scale_Knowledge_Graphs_images/iclr_2024_Less_is_More_One-shot_Subgraph_Reasoning_on_Large-scale_Knowledge_Graphs.pdf-0009-01.png)


**----- Start of picture text -----**<br>
0.06 1.5 0.00075 1.5<br>0.04 1.0 0.00050 1.0<br>0.02 0.5 0.00025 0.5<br>0.00 0 200 400 0 . 0 0 2 4 6 8 10 0.00000 0 20000 40000 60000 0.0 0 2 4 6 8 10<br>Degree of sampled entities Distance to the query entity Degree of sampled entities Distance to the query entity<br>Density Density Density Density<br>**----- End of picture text -----**<br>


Figure 5: Exemplar subgraphs sampled from WN18RR (left) and YAGO3-10 (right). The red and green nodes indicate the query entity and answer entity. The colors of the edges indicate relation types. The bottom distributions of degree and distance show the statistical properties of each subgraph. 

Table 5: Comparison of prediction performance with different sampling heuristics. 

|heuristics|WN18RR<br>MRR<br>H@1<br>H@10|YAGO3-10<br>MRR<br>H@1<br>H@10|
|---|---|---|
||||
|Random Sampling (RAND)<br>PageRank (PR)<br>Random Walk (RW)<br>Breadth-frst-searching (BFS)<br>**Personalized PageRank (PPR)**|0.03<br>43.4<br>3.5<br>0.124<br>11.5<br>14.2<br>0.507<br>45.8<br>59.8<br>0.543<br>49.6<br>63.0<br>**0.567**<br>**51.4**<br>**66.6**|0.057<br>5.1<br>6.5<br>0.315<br>28.9<br>35.9<br>0.538<br>46.3<br>67.2<br>0.562<br>49.4<br>69.0<br>**0.606**<br>**54.0**<br>**72.1**|



Table 6: Comparison of prediction performance with two recent GNN methods. 

|methods|WN18RR<br>MRR H@1 H@10<br>Time|YAGO3-10<br>MRR H@1 H@10<br>Time|
|---|---|---|
||||
|NBFNet (100% entities)<br>NBFNet**+ one-shot-subgraph (10% entities)**<br>RED-GNN (100% entities)<br>RED-GNN**+ one-shot-subgraph (10% entities)**|0.551<br>49.7<br>66.6<br>32.3 min<br>**0.554**<br>**50.5**<br>**66.3**<br>**2.6 min**<br>0.533<br>48.5<br>62.4<br>68.7 min<br>**0.567**<br>**51.4**<br>**66.6**<br>**4.5 min**|0.550<br>47.9<br>68.3<br>493.8 min<br>**0.565**<br>**49.6**<br>**69.2**<br>**28.2 min**<br>0.559<br>48.3<br>68.9<br>1382.9 min<br>**0.606**<br>**54.0**<br>**72.1**<br>**76.3 min**|



Table 7: Comparison of efficiency with an 8-layer predictor and different _rV[q][, r] E[q]_[.] 

|phase<br>_rq_<br>_V_|_rq_<br>_E_<br>WN18RR<br>Time<br>Memory|NELL-995<br>Time<br>Memory|YAGO3-10<br>Time<br>Memory|
|---|---|---|---|
|Training<br>1.0<br>0.5<br>0.2<br>0.2||||
||1.0<br>Out of memory<br>0.5<br>26.3m<br>20.3GB|Out of memory<br>1.6h<br>20.1GB|Out of memory<br>Out of memory|
|||||
||1.0<br>12.8m<br>20.2GB<br>0.2<br>6.7m<br>6.4GB|1.2h<br>18.5GB<br>0.6h<br>8.9GB|Out of memory<br>2.1h<br>23.1GB|
|||||
|0.1<br>0.1|1.0<br>7.2m<br>9.8GB<br>0.1<br>6.6m<br>5.1GB|0.8h<br>12.1GB<br>0.3h<br>5.3GB|1.3h<br>13.9GB<br>0.9h<br>10.2GB|
|Inference<br>1.0<br>0.5<br>0.2<br>0.2||||
||1.0<br>7.3m<br>6.7GB<br>0.5<br>6.0m<br>4.3GB|17.5m<br>12.8GB<br>8.3m<br>4.5GB|1.6h<br>15.0GB<br>1.1h<br>10.1GB|
|||||
||1.0<br>3.2m<br>5.8GB<br>0.2<br>2.8m<br>1.9GB|4.2m<br>12.1GB<br>3.6m<br>2.5GB|0.7h<br>14.7GB<br>0.6h<br>3.7GB|
|||||
|0.1<br>0.1|1.0<br>2.7m<br>2.7GB<br>0.1<br>2.3m<br>1.7GB|3.1m<br>9.4GB<br>2.9m<br>1.9GB|0.4h<br>9.7GB<br>0.4h<br>3.1GB|



**Case Study.** We visualize the sampled subgraph in Fig. 5 with the histograms of degree and distance distributions. As can be seen, the local structure of query entity _u_ is well preserved, while the true answers _v_ are also covered in the subgraphs. More cases and analyses can be found in Appendix. E. 

## 6 CONCLUSION 

In this paper, we propose the one-shot-subgraph link prediction to alleviate the scalability problem of structural methods and achieve efficient as well as adaptive learning on large-scale KGs. We discover that the non-parametric and computation-efficient heuristics PPR can effectively identify the potential answers and support to the prediction. We further introduce the automated searching for adaptive configurations in both data space and model space. Extensive experiments on five large-scale benchmarks verify the effectiveness and efficiency of our method. Importantly, we show it unnecessary to utilize the whole KG for answering specific queries; meanwhile, only a small proportion of information is essential and can be identified by the PPR heuristics without learning. 

9 

Published as a conference paper at ICLR 2024 

## ACKNOWLEDGMENTS 

ZKZ and BH were supported by the NSFC General Program No. 62376235, Guangdong Basic and Applied Basic Research Foundation Nos. 2022A1515011652 and 2024A1515012399, HKBU Faculty Niche Research Areas No. RC-FNRA-IG/22-23/SCI/04, and HKBU CSD Departmental Incentive Scheme. JCY was supported by 111 plan (No. BP0719010) and National Natural Science Foundation of China (No. 62306178). QMY was in part supported by NSFC (No. 92270106) and National Key Research and Development Program of China under Grant 2023YFB2903904. The authors thank Haobo Xu for his assistance in experiments. 

## ETHICS STATEMENT 

We would claim that this work does not raise any ethical concerns. Besides, this work does not involve any human subjects, practices to data set releases, potentially harmful insights, methodologies and applications, potential conflicts of interest and sponsorship, discrimination/bias/fairness concerns, privacy and security issues, legal compliance, and research integrity issues. 

## REPRODUCIBILITY STATEMENT 

The experimental setups for training and evaluation are described in detail in Sec. 5 and Appendix. C, and the experiments are all conducted using public datasets. The code is publicly available at: https://github.com/tmlr-group/one-shot-subgraph. 

## REFERENCES 

- P. Battaglia, J. Hamrick, V. Bapst, A. Sanchez-Gonzalez, V. Zambaldi, et al. Relational inductive biases, deep learning, and graph networks. arXiv preprint arXiv:1806.01261, 2018. 

- J. Bergstra, D. Yamins, D. Cox, et al. Hyperopt: A python library for optimizing the hyperparameters of machine learning algorithms. In Proceedings of the 12th Python in science conference, 2013. 

- A. Bordes, N. Usunier, A. Garcia-Duran, J. Weston, and O. Yakhnenko. Translating embeddings for modeling multi-relational data. In NeurIPS, 2013. 

- L. Breiman. Random forests. ML, 45(1):5–32, 2001. 

- Y. Cao, X. Wang, X. He, Z. Hu, and T. Chua. Unifying knowledge graph learning and recommendation: Towards a better understanding of user preferences. In TheWebConf, 2019. 

- J. Chen, T. Ma, and C. Xiao. Fastgcn: Fast learning with graph convolutional networks via importance sampling. In ICLR, 2018. 

- X. Chen, S. Jia, and Y. Xiang. A review: Knowledge reasoning over knowledge graph. Expert Systems with Applications, 141:112948, 2020. 

- Y. Chen, H. Yang, Y. Zhang, K. Ma, T. Liu, B. Han, and J. Cheng. Understanding and improving graph injection attack by promoting unnoticeability. In ICLR, 2022. 

- H. Cheng, L. Koc, J. Harmsen, T. Shaked, T. Chandra, H. Aradhye, G. Anderson, G. Corrado, W. Chai, M. Ispir, et al. Wide & deep learning for recommender systems. In Proceedings of the 1st workshop on deep learning for recommender systems, 2016. 

- P. Covington, J. Adams, and E. Sargin. Deep neural networks for youtube recommendations. In Proceedings of the 10th ACM conference on recommender systems, 2016. 

- R. Das, S. Dhuliawala, M. Zaheer, L. Vilnis, I. Durugkar, A. Krishnamurthy, A. Smola, and A. McCallum. Go for a walk and arrive at the answer: Reasoning over paths in knowledge bases using reinforcement learning. In ICLR, 2017. 

- T. Dettmers, P. Minervini, P. Stenetorp, and S. Riedel. Convolutional 2D knowledge graph embeddings. In AAAI, 2017. 

10 

Published as a conference paper at ICLR 2024 

- P. Diaconis and S. Janson. Graph limits and exchangeable random graphs. arXiv preprint arXiv:0712.2749, 2007. 

- M. Galkin, E. Denis, J. Wu, and W. Hamilton. Nodepiece: Compositional and parameter-efficient representations of large knowledge graphs. In ICLR, 2022. 

- M. Galkin, X. Yuan, H. Mostafa, J. Tang, and Z. Zhu. Towards foundation models for knowledge graph reasoning. arXiv preprint arXiv:2310.04562, 2023. 

- M. Gardner and S. Dorling. Artificial neural networks (the multilayer perceptron) – a review of applications in the atmospheric sciences. Atmospheric environment, 32(14-15):2627–2636, 1998. 

- L. Grinsztajn, E. Oyallon, and G. Varoquaux. Why do tree-based models still outperform deep learning on tabular data? arXiv preprint arXiv:2207.08815, 2022. 

- W. Hamilton, Z. Ying, and J. Leskovec. Inductive representation learning on large graphs. In NeurIPS, 2017. 

- W. Hu, M. Fey, M. Zitnik, Y. Dong, H. Ren, B. Liu, M. Catasta, and J. Leskovec. Open graph benchmark: Datasets for machine learning on graphs. In NeurIPS, 2020. 

- X. Huang, J. Zhang, D. Li, and P. Li. Knowledge graph embedding based question answering. In WSDM, 2019. 

- F. Hutter, H. Hoos, and K. Leyton-Brown. Sequential model-based optimization for general algorithm configuration. In ICLIO, 2011. 

- G. Jeh and J. Widom. Scaling personalized web search. In WWW, 2003. 

- S. Ji, S. Pan, E. Cambria, P. Marttinen, and P. S Yu. A survey on knowledge graphs: Representation, acquisition and applications. arxiv preprint arXiv:2002.00388, 2020. 

- J. Li, S. Zhang, J. Cao, and M. Tan. Learning defense transformations for counterattacking adversarial examples. Neural Networks, 164:177–185, 2023. 

- S. Mai, S. Zheng, Y. Yang, and H. Hu. Communicative message passing for inductive relation reasoning. In AAAI, 2021. 

- S. Maskey, R. Levie, Y. Lee, and G. Kutyniok. Generalization analysis of message passing neural networks on large random graphs. In NeurIPS, 2022. 

- S. Miao, M. Liu, and P. Li. Interpretable and generalizable graph learning via stochastic attention mechanism. In ICML, 2022. 

- H. Mohamed, D. Pilutti, S. James, A. Del Bue, M. Pelillo, and S. Vascon. Locality-aware subgraphs for inductive link prediction in knowledge graphs. Pattern Recognition Letters, 2023. 

- K. Oono and T. Suzuki. Graph neural networks exponentially lose expressive power for node classification. In ICLR, 2019. 

- L. Page, S. Brin, R. Motwani, and T. Winograd. The pagerank citation ranking: Bringing order to the web. Technical report, Stanford infolab, 1999. 

- A. Paszke, S. Gross, S. Chintala, G. Chanan, E. Yang, Z. DeVito, Z. Lin, A. Desmaison, L. Antiga, and A. Lerer. Automatic differentiation in PyTorch. In ICLR, 2017. 

- M. Qu, J. Chen, L. Xhonneux, Y. Bengio, and J. Tang. Rnnlogic: Learning logic rules for reasoning on knowledge graphs. In ICLR, 2021. 

- A. Sadeghian, M. Armandpour, P. Ding, and D. Z. Wang. Drum: End-to-end differentiable rule mining on knowledge graphs. In NeurIPS, 2019. 

- M. Schlichtkrull, T. N Kipf, P. Bloem, R. Van Den Berg, I. Titov, and M. Welling. Modeling relational data with graph convolutional networks. In ESWC, 2018. 

11 

Published as a conference paper at ICLR 2024 

- Z. Shi, X. Liang, and J. Wang. Lmc: Fast training of gnns via subgraph sampling with provable convergence. arXiv preprint arXiv:2302.00924, 2023. 

- F. Suchanek, G. Kasneci, and G. Weikum. Yago: A core of semantic knowledge. In TheWebConf, pp. 697–706, 2007. 

- Z. Sun, Z. Deng, J. Nie, and J. Tang. Rotate: Knowledge graph embedding by relational rotation in complex space. In ICLR, 2019. 

- Z. Tang, S. Shi, B. Li, and X. Chu. Gossipfl: A decentralized federated learning framework with sparsified and adaptive communication. IEEE Transactions on Parallel and Distributed Systems, 2022a. 

- Z. Tang, Y. Zhang, S. Shi, X. He, B. Han, and X. Chu. Virtual homogeneity learning: Defending against data heterogeneity in federated learning. In ICML, 2022b. 

- K. K Teru, E. Denis, and W. L Hamilton. Inductive relation prediction by subgraph reasoning. In ICML, 2020. 

- S. Vashishth, S. Sanyal, V. Nitin, and P. Talukdar. Composition-based multi-relational graph convolutional networks. In ICLR, 2019. 

- Q. Wang, Z. Mao, B. Wang, and L. Guo. Knowledge graph embedding: A survey of approaches and applications. TKDE, 2017. 

- X. Wang, X. He, Y. Cao, M. Liu, and T. Chua. Kgat: Knowledge graph attention network for recommendation. In SIGKDD, 2019. 

- C. Williams and C. Rasmussen. Gaussian processes for regression. In NIPS, 1995. 

- W. Xiong, T. Hoang, and W. Wang. Deeppath: A reinforcement learning method for knowledge graph reasoning. In EMNLP, 2017. 

- X. Xu, W. Feng, Y. Jiang, X. Xie, Z. Sun, and Z. Deng. Dynamically pruned message passing networks for large-scale knowledge graph reasoning. In ICLR, 2019. 

- M. Yasunaga, H. Ren, A. Bosselut, P. Liang, and J. Leskovec. Qa-gnn: Reasoning with language models and knowledge graphs for question answering. In NAACL, 2021. 

- Y. Yu, K. Huang, C. Zhang, L. M Glass, J. Sun, and C. Xiao. Sumgnn: multi-typed drug interaction prediction via efficient knowledge graph summarization. Bioinformatics, 37(18):2988–2995, 2021. 

- S. Zhang, Y. Tay, L. Yao, and Q. Liu. Quaternion knowledge graph embeddings. In NeurIPS, 2019. 

- S. Zhang, F. Liu, J. Yang, Y. Yang, C. Li, B. Han, and M. Tan. Detecting adversarial data by probing multiple perturbations using expected perturbation score. In ICML, 2023a. 

- Y. Zhang and Q. Yao. Knowledge graph reasoning with relational directed graph. In TheWebConf, 2022. 

- Y. Zhang, M. Gong, T. Liu, G. Niu, X. Tian, B. Han, B. Schölkopf, and K. Zhang. Adversarial robustness through the lens of causality. In ICLR, 2022a. 

- Y. Zhang, Z. Zhou, Q. Yao, and Y. Li. Efficient hyper-parameter search for knowledge graph embedding. In ACL, 2022b. 

- Y. Zhang, Q. Yao, L. Yue, X. Wu, Z. Zhang, Z. Lin, and Y. Zheng. Emerging drug interaction prediction enabled by a flow-based graph neural network with biomedical network. Nature Computational Science, 3(12):1023–1033, 2023b. 

- Y. Zhang, Z. Zhou, Q. Yao, X. Chu, and B. Han. Adaprop: Learning adaptive propagation for knowledge graph reasoning. In SIGKDD, 2023c. 

- Y. Zhou, G. Kutyniok, and B. Ribeiro. Ood link prediction generalization capabilities of messagepassing gnns in larger test graphs. In NeurIPS, 2022. 

12 

Published as a conference paper at ICLR 2024 

- Z. Zhou, J. Yao, J. Liu, X. Guo, Q. Yao, L. He, L. Wang, B. Zheng, and B. Han. Combating bilateral edge noise for robust link prediction. In NeurIPS, 2023a. 

- Z. Zhou, C. Zhou, X. Li, J. Yao, Q. Yao, and B. Han. On strengthening and defending graph reconstruction attack with markov chain approximation. In ICML, 2023b. 

- Z. Zhu, Z. Zhang, L. Xhonneux, and J. Tang. Neural bellman-ford networks: A general graph neural network framework for link prediction. In NeurIPS, 2021. 

- Z. Zhu, X. Yuan, M. Galkin, S. Xhonneux, M. Zhang, M. Gazeau, and J. Tang. A*net: A scalable path-based reasoning approach for knowledge graphs. In NeurIPS, 2023. 

13 

Published as a conference paper at ICLR 2024 

## **Appendix** 

## **Table of Contents** 

|**A **|**Theoretical Analysis**|**Theoretical Analysis**|||**15**|
|---|---|---|---|---|---|
||A.1|Notations<br>. . . .|. .|. . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .|15|
||A.2|Complexity Analysis||. . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .|15|
||A.3|Proof for Theorem 1 .||. . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .|16|
|**B**|**Implementation Details**||||**18**|
|**C **|**Full Evaluations**||||**19**|
||C.1|Setup. . . . . . .|. .|. . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .|19|
||C.2|Comparison With Other Effcient Methods<br>. . . . . . . . . . . . . . . . . . . .|||19|
||C.3|The Edge Split Scheme||. . . . . . . . . . . . . . . . . . . . . . . . . . . . . .|20|
||C.4|Robustness Against Data Perturbations . . . . . . . . . . . . . . . . . . . . . .|||21|
|**D **|**Further Discussion**||||**21**|
||D.1|Design Principles|. .|. . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .|21|
||D.2|Contributions<br>. .|. .|. . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .|22|
||D.3|Explanation for the|Improvement in Performance . . . . . . . . . . . . . . . . .||22|
||D.4|Extension<br>. . . .|. .|. . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .|23|
|**E**|**Case Study**||||**24**|



14 

Published as a conference paper at ICLR 2024 

## A THEORETICAL ANALYSIS 

## A.1 NOTATIONS 

We summarize the frequently used notations in Tab. 8. 

Table 8: The most frequently used notations in this paper. 

|notations|meanings|
|---|---|
|||
|_V, R, E_|the set of entities, relations, facts (edges) of the original KG|
|||
|_G_ = (_V, R, E_)|the original KG|
|||
|_Vs, Rs, Es_|the set of entities, relations, facts (edges) of the sampled KG|
|||
|_Gs_ = (_Vs, Rs, Es_)|the sampled KG generated by the sampler_g_**_ϕ_**|
|||
|(_x, r, o_)|a fact triplet in_E_ with subject entity_x_, relation_r_and object entity_o_|
|||
|(_u, q, v_)|a query triple with query entity_u_, query relation_q_and answer entity_v_|
|||
|_L_|the total number of propagation steps|
|||
|_ℓ_|the_ℓ_-th propagation step and_ℓ∈{_0_. . . L}_|
|||
|_K_|the maximum steps of updating the PPR scores|
|_k_<br>the_k_-th propagation step and_k ∈{_0_. . . K}_||
|||
|**_h_**(_ℓ_)<br>_o_|the representation of entity_o_at step_ℓ_|
|||
|_g_**_ϕ_**|the sampler of the one-shot-subgraph link prediction framework|
|||
|_f_**_θ_**|the predictor of the one-shot-subgraph link prediction framework|
|||
|_rq_<br>_V, rq_<br>_E_|the sampling ratios of entities and edges|



## A.2 COMPLEXITY ANALYSIS 

Next, we compare three different manners from perspectives of computation and parameter. 

In a nutshell, semantic models are computation-efficient but parameter-expensive, while structural models are parameter-efficient but computation-expensive. Here, we aim to make the best of the both worlds by designing a framework that is parameter-efficient and computation-efficient. The proposed one-shot-subgraph link prediction models, as will be elaborated in Sec. 3, is with the new prediction _g_ _**ϕ** ,_ ( _uq,rq_ ) _f_ _**θ**_ manner that _G �−−−−−−→Gs �−→_ _**Y**_[ˆ] . The key difference here is that, instead of the original _G_ , the predictor _f_ _**θ**_ is acting on the query-dependent subgraph _Gs_ extracted by the sampler _g_ _**ϕ**_ . 

By contrast, we show that the proposed subgraph models make the best of both worlds: (1) subgraph models are computation-efficient, as the extracted subgraph is much smaller than the original graph, _i.e._ , _|Vs|≪|V|_ and _|Es|≪|E|_ ; (2) inherits from Sadeghian et al. (2019); Zhu et al. (2021); Zhang & Yao (2022), subgraph models are also parameter-efficient: only requires the relations’ embeddings but not the expensive entities’ embeddings. By contrast, the semantic models need to learn entities’s embeddings, which are parameter-expensive and only applicable in transductive settings. 

A detailed comparison of parameter and computation complexity is summarized in Tab. 9. 

Table 9: The comparison of related works with parameter complexity and computation complexity. 

|Category|Parameter complexity<br>Computation complexity|
|---|---|
|||
|semantic models<br>structural models<br>subgraph models|_O_(_|V|·DV_)+_O_(_|R|·DR_)<br>_O_(_|R|·DV_)<br>_O_(_|R|·DR_)<br>_O_(_|E|·DR·L_)<br>_O_(_|R|·DR_)<br>_O_(_|E|·K_)+_O_(_|Es|·DR·L_)|



15 

Published as a conference paper at ICLR 2024 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2024_Less_is_More_One-shot_Subgraph_Reasoning_on_Large-scale_Knowledge_Graphs_images/iclr_2024_Less_is_More_One-shot_Subgraph_Reasoning_on_Large-scale_Knowledge_Graphs.pdf-0016-01.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2024_Less_is_More_One-shot_Subgraph_Reasoning_on_Large-scale_Knowledge_Graphs_images/iclr_2024_Less_is_More_One-shot_Subgraph_Reasoning_on_Large-scale_Knowledge_Graphs.pdf-0016-02.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2024_Less_is_More_One-shot_Subgraph_Reasoning_on_Large-scale_Knowledge_Graphs_images/iclr_2024_Less_is_More_One-shot_Subgraph_Reasoning_on_Large-scale_Knowledge_Graphs.pdf-0016-03.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2024_Less_is_More_One-shot_Subgraph_Reasoning_on_Large-scale_Knowledge_Graphs_images/iclr_2024_Less_is_More_One-shot_Subgraph_Reasoning_on_Large-scale_Knowledge_Graphs.pdf-0016-04.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2024_Less_is_More_One-shot_Subgraph_Reasoning_on_Large-scale_Knowledge_Graphs_images/iclr_2024_Less_is_More_One-shot_Subgraph_Reasoning_on_Large-scale_Knowledge_Graphs.pdf-0016-05.png)


**----- Start of picture text -----**<br>
(a) semantic model (b) structural model (c) one-shot-subgraph model (d) complexity comparison<br>**----- End of picture text -----**<br>


Figure 6: Illustrations of prediction manners. Semantic and structure models implicitly or explicitly take the whole graph _G_ for prediction. Our _one-shot-subgraph_ model _only requires one subgraph Gs for the prediction of one query_ that decreases computation complexity via decoupling _f_ _**θ**_ and _G_ . 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2024_Less_is_More_One-shot_Subgraph_Reasoning_on_Large-scale_Knowledge_Graphs_images/iclr_2024_Less_is_More_One-shot_Subgraph_Reasoning_on_Large-scale_Knowledge_Graphs.pdf-0016-07.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2024_Less_is_More_One-shot_Subgraph_Reasoning_on_Large-scale_Knowledge_Graphs_images/iclr_2024_Less_is_More_One-shot_Subgraph_Reasoning_on_Large-scale_Knowledge_Graphs.pdf-0016-08.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2024_Less_is_More_One-shot_Subgraph_Reasoning_on_Large-scale_Knowledge_Graphs_images/iclr_2024_Less_is_More_One-shot_Subgraph_Reasoning_on_Large-scale_Knowledge_Graphs.pdf-0016-09.png)


**----- Start of picture text -----**<br>
(a) layer-wise sampling (b) subgraph-wise sampling<br>**----- End of picture text -----**<br>


Figure 7: Illustrations of sampling methods. Detailed analysis is corresponding to Sec. 3. 

## A.3 PROOF FOR THEOREM 1 

_Proof._ Here, let _Gs_[train] _∼_ P _G_ and _Gs_[test] _∼_ P _G_ be the training and testing graphs that are sampled distribution P _G_ . Let _G_[test] be large enough to satisfy _√|Vs_[test] _[|] /[√]_ log(2 _|Vs_[test] _[|][/p]_[)] _[≥]_[4] ~~_√_~~ 2 _/d_ min, where _d_ min is the constant of graphon degree (Diaconis & Janson, 2007). 

In this part, we follow Maskey et al. (2022); Zhou et al. (2022) with the definitions of graph message-passing neural network Φ and continuous message passing neural network Ψ. Note that Φ is equivalent to the message passing function of the predictor _f_ _**w**_ that generates the entity representation _**h**_[(] _v[ℓ]_[)] in each layer _ℓ_ = 1 _,_ 2 _, · · · , L_ . Besides, Ψ is the continuous counterpart of Φ. It uses the continuous aggregation instead of the discrete one w.r.t. the discrete graph topology in Φ. 

With the two above functions, _∥_ Φ _∥∞, ∥_ Ψ _∥∞_ can be trained and determined by the original graph _G_[train] and hyperparameters _rV[q][, r] E[q]_[and] _[ L]_[.][Following][ Maskey et al.][ (][2022][);][ Zhou et al.][ (][2022][),][let] _p ∈_ (0 _,_[1] _/_[�] _ℓ[L]_ =1[2] _[|]_ _**[h]**_[(] _[ℓ]_[)] _[|]_[+1][)][ we with probability at least][ 1] _[−]_[�] _[L] ℓ_ =1[2(] _[|]_ _**[h]**_[(] _[ℓ]_[)] _[|]_[+1)] _[p]_[, the difference] _[ δ]_[ between] _∥_ Φ _∥∞_ and _∥_ Ψ _∥∞_ is bounded by 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2024_Less_is_More_One-shot_Subgraph_Reasoning_on_Large-scale_Knowledge_Graphs_images/iclr_2024_Less_is_More_One-shot_Subgraph_Reasoning_on_Large-scale_Knowledge_Graphs.pdf-0016-15.png)


Then, based on the proof for Lemma B.9 in (Maskey et al., 2022), we can derive _∥g∥∞≤ B_ 1[(] _[ℓ]_[)] + _B_ 2[(] _[ℓ]_[)] _[∥][g][∥][∞]_[, where] _[ B]_ 1[(] _[ℓ]_[)] _[, B]_ 2[(] _[ℓ]_[)] are independent of _g_ . Specifically, 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2024_Less_is_More_One-shot_Subgraph_Reasoning_on_Large-scale_Knowledge_Graphs_images/iclr_2024_Less_is_More_One-shot_Subgraph_Reasoning_on_Large-scale_Knowledge_Graphs.pdf-0016-17.png)


16 

Published as a conference paper at ICLR 2024 

Then, we define constants _C_ 1 _, C_ 2 as follows. 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2024_Less_is_More_One-shot_Subgraph_Reasoning_on_Large-scale_Knowledge_Graphs_images/iclr_2024_Less_is_More_One-shot_Subgraph_Reasoning_on_Large-scale_Knowledge_Graphs.pdf-0017-02.png)


This, we can derive the difference _δ_ as 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2024_Less_is_More_One-shot_Subgraph_Reasoning_on_Large-scale_Knowledge_Graphs_images/iclr_2024_Less_is_More_One-shot_Subgraph_Reasoning_on_Large-scale_Knowledge_Graphs.pdf-0017-04.png)


where _C_ 1 _, C_ 2 depends on _{L_[(] Φ _[ℓ]_[)][(] _[M]_[ train][)] _[}] ℓ[L]_ =1[and] _[ {][L]_[(] Ψ _[ℓ]_[)][(] _[M]_[ train][)] _[}] ℓ[L]_ =1[that] 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2024_Less_is_More_One-shot_Subgraph_Reasoning_on_Large-scale_Knowledge_Graphs_images/iclr_2024_Less_is_More_One-shot_Subgraph_Reasoning_on_Large-scale_Knowledge_Graphs.pdf-0017-06.png)


Then, consider any two test entities _u, v ∈Vs_[test][, for which we can make a prediction decision of fact] ( _u, q, v_ ), we have 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2024_Less_is_More_One-shot_Subgraph_Reasoning_on_Large-scale_Knowledge_Graphs_images/iclr_2024_Less_is_More_One-shot_Subgraph_Reasoning_on_Large-scale_Knowledge_Graphs.pdf-0017-08.png)


The first inequality holds by the triangle inequality. In contrast, the second inequality holds since Ψ _u_ =Ψ _v_ . Note that _u_ and _v_ are isomorphic in the topology of the test graph (Zhou et al., 2022) and thus with the same representations. 

Next, with probability at least 1 _−_[�] _[L] ℓ_ =1[2(] _[|]_ _**[h]**_[(] _[ℓ]_[)] _[|]_[+1)] _[p]_[, for arbitrary entity] _[ v][′][∈V] s_[test][, we have] 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2024_Less_is_More_One-shot_Subgraph_Reasoning_on_Large-scale_Knowledge_Graphs_images/iclr_2024_Less_is_More_One-shot_Subgraph_Reasoning_on_Large-scale_Knowledge_Graphs.pdf-0017-11.png)


Then, when the size of the test graph is satisfied and _**p** uv_ = _f_ _**w**_ ( _Gs_[test][)] _[uv]_[=][ READOUT][(] _**[h]** u_[(] _[L]_[)] _[,]_ _**[ h]**_[(] _v[L]_[)] ), we have 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2024_Less_is_More_One-shot_Subgraph_Reasoning_on_Large-scale_Knowledge_Graphs_images/iclr_2024_Less_is_More_One-shot_Subgraph_Reasoning_on_Large-scale_Knowledge_Graphs.pdf-0017-13.png)


Specifically, if _**p** uv ≥ τ_ , we have 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2024_Less_is_More_One-shot_Subgraph_Reasoning_on_Large-scale_Knowledge_Graphs_images/iclr_2024_Less_is_More_One-shot_Subgraph_Reasoning_on_Large-scale_Knowledge_Graphs.pdf-0017-15.png)


if _**p** uv ≤ τ_ , we have 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2024_Less_is_More_One-shot_Subgraph_Reasoning_on_Large-scale_Knowledge_Graphs_images/iclr_2024_Less_is_More_One-shot_Subgraph_Reasoning_on_Large-scale_Knowledge_Graphs.pdf-0017-17.png)


Thus, whether _u, v_ are generated by the same or distinct _g_ , where the underlying generative function of graph signal _g ∈ L[∞]_ is with the essential supreme norm as in Maskey et al. (2022); Zhou et al. (2022), we have a probability at least 1 _−_[�] _[L] ℓ_ =1[2(] _[|]_ _**[h]**_[(] _[ℓ]_[)] _[|]_[+1)] _[p]_[ that the same predictions are obtained.] 

17 

Published as a conference paper at ICLR 2024 

## B IMPLEMENTATION DETAILS 

**MESS** ( _·_ ) **and AGG** ( _·_ ) **for GNN-based methods.** The two functions of the GNN-based methods are summarized in Tab.10. The main differences between different methods are the combinators for entity and relation representations on the edges, the operators on the different representations, and the attention weights. Recent works (Vashishth et al., 2019; Zhu et al., 2021) have shown that the different combinators and operators only have a slight influence on the performance. Compared with GraIL and RED-GNN, even though the message functions and aggregation functions are similar, their empirical performances are quite different, with different propagation patterns. 

Table 10: Summary of the GNN functions for message propagation. The values in _{}_ represent different operation choices that are tuned as hyper-parameters for different datasets. 

|method|**_m_**(_ℓ_)<br>(_es,r,eo_) :=MESS(_·_)|**_h_**(_ℓ_)<br>_eo_ :=AGG(_·_)|_eo_)|
|---|---|---|---|
|R-GCN (Schlichtkrull et al.,2018)|**_W_** (_ℓ_)<br>_r_<br>**_h_**(_ℓ−_1)<br>_es_<br>, where**_W_**_r_ depends on the<br>relation_r_|**_W_** (_ℓ_)<br>_o_ **_h_**(_ℓ−_1)<br>_eo_<br>+�<br>_eo∈N_(_es_)<br>1<br>_co,r_ **_m_**(_ℓ_)<br>(_es,r,eo_)||
|CompGCN (Vashishth et al.,2019)|**_W_** (_ℓ_)<br>_λ_(_r_)_{−, ∗, ⋆}_(**_h_**(_ℓ−_1)<br>_es_<br>_,_**_h_**(_ℓ_)<br>_r_ ), where**_W_**(_ℓ_)<br>_λ_(_r_)<br>depends on the direction of_r_|�<br>_eo∈N_(_es_) **_m_**(_ℓ_)<br>(_es,r,eo_)||
|GraIL (Teru et al.,2020)|_α_(_ℓ_)<br>(_es,r,eo_)_|rq_(**_W_**(_ℓ_)<br>1<br>**_h_**(_ℓ−_1)<br>_es_<br>+**_W_** (_ℓ_)<br>2<br>**_h_**(_ℓ−_1)<br>_eo_<br>),<br>where_α_(_ℓ_)<br>(_es,r,eo_)_|rq_ is the attention weight.|**_W_** (_ℓ_)<br>_o_ **_h_**(_ℓ−_1)<br>_eo_<br>+�<br>_eo∈N_(_es_)<br>1<br>_co,r_ **_m_**(_ℓ_)<br>(_es,r,eo_)||
|NBFNet (Zhu et al.,2021)|**_W_** (_ℓ_)_{_+_, ∗, ◦}_(**_h_**(_ℓ−_1)<br>_es_<br>_,_**_w_**_q_(_r, rq_)), where<br>**_w_**_q_(_r, rq_)is a query-dependent weight vector|_{_Sum,Mean,Max,PNA_}eo∈N_(_es_)**_m_**(_ℓ_)<br>(_es,r,_||
|RED-GNN (Zhang & Yao,2022)|_α_(_ℓ_)<br>(_es,r,eo_)_|rq_(**_h_**(_ℓ−_1)<br>_es_<br>+**_h_**(_ℓ_)<br>_r_ ), where<br>_α_(_ℓ_)<br>(_es,r,eo_)_|rq_ is the attention weight|�<br>_eo∈N_(_es_) **_m_**(_ℓ_)<br>(_es,r,eo_)||



**Design space of the predictor.** From the model’s perspective, we build the configuration space of the predictor and further utilize the advantages of existing structural models introduced in Sec. 2. Three query-dependent message functions MESS( _·_ ) are considered here, including DRUM (Sadeghian et al., 2019) (denoted as MDRUM), NBFNet (Zhu et al., 2021) (MNBFNet), and RED-GNN (Zhang & Yao, 2022) (MREDGNN). The effective message is propagated from _uq_ to the entities in subgraph _Gs_ . 

Generally, the message propagation can be formulated as 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2024_Less_is_More_One-shot_Subgraph_Reasoning_on_Large-scale_Knowledge_Graphs_images/iclr_2024_Less_is_More_One-shot_Subgraph_Reasoning_on_Large-scale_Knowledge_Graphs.pdf-0018-07.png)


Note the effective message is propagated from _u_ to the entities in subgraph _Gs_ . The ranges for design dimensions of the configuration space are shown below, where the upper is intra-layer design while the lower is inter-layer design. 

|**DROPOUT**(_·_)|**ACT**(_·_)|**AGG**(_·_)|**MESS**(_·_)||**Dimension**|
|---|---|---|---|---|---|
|(0, 0.5)|Identity, Relu, Tanh|Max, Mean, Sum|MDRUM_,_MNBFNet_,_MREDGNN||16, 32, 64, 128|
|**No. layers (**_L_**) **|**Repre. initialization **|**Layer-wise shortcut **|**Repre. concatenation**||**READOUT**(_·_)|
|{4, 6, 8, 10}|Binary, Relational|True, False|True, False|Linear, Dot product||



**Head and tail prediction.** Note that predicting a missing head in KG can also be formulated as tail prediction by adding inverse relations. For example, predicting a missing head in KG (? _, q, v_ ) can also be formulated to tail prediction by adding inverse relations as ( _v, q_ inverse _,_ ?). This formulation involves augmenting the original KG with inverse relations, following the approach adopted in other KG methods (Zhu et al., 2021; Zhang & Yao, 2022; Zhang et al., 2023c; Zhu et al., 2023; Zhang et al., 2023b; Galkin et al., 2023). 

18 

Published as a conference paper at ICLR 2024 

## C FULL EVALUATIONS 

## C.1 SETUP 

**Datasets.** We use five benchmarks with more than ten-thousand entities, including WN18RR (Dettmers et al., 2017), NELL-995 (Xiong et al., 2017), YAGO3-10 (Suchanek et al., 2007), OGBL-BIOKG and OGBL-WIKIKG2 (Hu et al., 2020). The statistics are summarized in Tab. 11. 

Table 11: Statistics of the five KG datasets with more than ten-thousand entities. Fact triplets in _E_ are used to build the graph, and _E_[train] , _E_[val] , _E_[test] are edge sets of training, validation, and test set. 

|dataset|_|V|_<br>_|R|_|_|E|_<br>_|E_train_|_<br>_|E_val_|_<br>_|E_test_|_|
|---|---|---|
||||
|WN18RR<br>NELL-995<br>YAGO3-10<br>OGBL-BIOKG<br>OGBL-WIKIKG2|40.9k<br>11<br>74.5k<br>200<br>123.1k<br>37<br>93.7k<br>51<br>2500.6k<br>535|65.1k<br>59.0k<br>3.0k<br>3.1k<br>112.2k<br>108.9k<br>0.5k<br>2.8k<br>1089.0k<br>1079.0k<br>5.0k<br>5.0k<br>5088.4k<br>4762.7k<br>162.9k<br>162.9k<br>17137.1k<br>16109.2k<br>429.5k<br>598.5k|



**Baselines.** We compare our method with general link prediction methods, including (i) semantics models: ConvE (Dettmers et al., 2017), QuatE (Zhang et al., 2019), and RotatE (Sun et al., 2019); and (ii) structural models: MINERVA (Das et al., 2017), DRUM (Sadeghian et al., 2019), RNNLogic (Qu et al., 2021), CompGCN (Vashishth et al., 2019), DPMPN (Xu et al., 2019), NBFNet (Zhu et al., 2021), and RED-GNN (Zhang & Yao, 2022). The results of these baseline are taken from their papers or reproduced by their official codes. 

## C.2 COMPARISON WITH OTHER EFFICIENT METHODS 

To reduce entity vocabulary to be much smaller than full entities, NodePiece (Galkin et al., 2022) represents an anchor-based approach that facilitates the acquisition of a fixed-size entity vocabulary. Note that NodePiece was originally designed for semantic models, which diverges from the primary focus of our investigation centered around structural models. 

As shown below, our one-shot-subgraph link prediction method outperforms semantic models, NodePiece, and original RED-GNN in the MRR metric. Additionally, it’s worth noting that NodePiece can lead to substantial performance degradation when tasked with reducing unique embeddings for a sizable number of entities. The table below illustrates this phenomenon. In contrast, the RED-GNN, when augmented with our one-shot-subgraph method, showcases an ability to enhance performance even while learning and predicting with only 10% of entities. Furthermore, the parameter count for our structural models remains significantly lower than that of semantic models, even with the incorporation of NodePiece improvements. 

Table 12: Comparison with NodePiece. 

||WN18RR<br>MRR<br>H@10<br>#Params|YAGO3-10<br>MRR<br>H@10<br>#Params|
|---|---|---|
||||
|RotatE (100% entities)<br>RotatE + NodePiece (10% entities)<br>RED-GNN (100% entities)<br>RED-GNN**+ one-shot-subgraph (10% entities)**|0.476<br>57.1<br>41M<br>0.403<br>51.5<br>4.4M<br>0.533<br>62.4<br>0.02M<br>**0.567**<br>**66.6**<br>**0.03M**|0.495<br>67.0<br>123M<br>0.247<br>48.8<br>4.1M<br>0.559<br>68.9<br>0.06M<br>**0.606**<br>**72.1**<br>**0.09M**|



Besides, the other efficient link prediction method, DPMPN (Xu et al., 2019), contains two GNNs, one is a full-graph GNN that is similar to CompGCN, and the other one is pruned. DPMPN is a mixture of GNN methods, where the pruned GNN already knows the global information and adopts a layer-wise sampling manner. DPMPN requires several propagation steps to prune the message passing and sample the subgraph, while our PPR sampler can efficiently extract the subgraph without learning. In comparison, our method is simpler but much more effective that it observably outperforms DPMPN. Hence, considering the differences, one-shot-subgraph link prediction is still a novel subgraph sampling-based method for link prediction and achieves state-of-the-art prediction performance. 

As for other sampling-based methods, the extracted subgraph by Yasunaga et al. (2021) is equivalent to the Breadth-first-searching (BFS) that is compared in our work. As it comprises all entities on the k-hop neighbors in its subgraph, the number of sampled entities could be quite large. For example, 

19 

Published as a conference paper at ICLR 2024 

on the WN18RR dataset, the full 5-hop neighbors of the query entity take up 9 _._ 4% of entities and 13 _._ 5% of edges, while the full 8-hop neighbors can even take up 69 _._ 8% of entities and 89 _._ 7% of edges. As for the YAGO3-10 dataset, the full 5-hop neighbors take up 98 _._ 1% of entities and 95 _._ 6% of edges, almost equal to the entire KG. It could still be expensive for prediction and thus not suitable for prediction on large KGs like YAGO3-10. 

The sampling method of Mohamed et al. (2023) aims to extract an enclosing subgraph to answer one given query ( _u, q,_ ?), which is consistent with the previous work GraIL (Teru et al., 2020). Note the extracted subgraphs are different for different given triples. That is, for answering one query ( _u, q,_ ?), this method requires sampling subgraphs and predicting the score of _N_ potential links on each individual subgraph, where _N_ is the number of all entities. Hence, this method is extremely expensive and also not suitable for large KGs. By contrast, our one-shot-subgraph link prediction method only requires extracting one subgraph to answer one query rather than _N_ subgraphs. Meanwhile, the extracted subgraph is much smaller than the full k-hop neighbor. Thus, our method is more efficient and more suitable for learning and prediction on large KGs. 

## C.3 THE EDGE SPLIT SCHEME 

The edge split can be seen as a masking operation on KG, _i.e._ , removing the query edges from the observation edges (the inputs of a prediction model). It is necessary for KG learning; otherwise, the query edges for training can be found in the observation graph, which is not practical and not reasonable. We further clarify the edge split scheme in the following three folds. 

**A general perspective of KG incompleteness.** Note that the KG datasets are generally and naturally incomplete, and the link prediction tasks aim to predict the missing links among entities. An ideal link prediction model should be robust to the intrinsic incompleteness of a KG, and the local evidence to be utilized for the prediction of a specific query is also generally incomplete. In this view, the edge split can be seen as a data augmentation method. 

**Training details about edge split.** Edge split indeed influences the local connectivity of a KG, where a lower split ratio of fact:train can lead to a sparser observation graph for training. However, we afresh split the query and observation edges in each epoch, and thus, all the edges in the train set can be recursively used as the query edges for training. Besides, all these fact/train edges can be used in the test phase as factual observations. Hence, all the edges in the training set are recursively used in training and explicitly used in testing. 

**The influence of edge split on training.** We conduct a further experiment with different split ratios and constrain the experiments using the same amount of training time. As the data shown below, the split ratios greatly impact the training time for one epoch, and a higher fact:train ratio leads to short training time. Besides, various split ratios only slightly influence the converged result; however, they greatly influence the speed of convergence. Specifically, a higher split ratio brings a faster convergence speed, especially on the YAGO3-10. 

Table 13: Comparison of different split ratios. 

||WN18RR<br>MRR<br>H@1<br>H@10<br>Time|YAGO3-10<br>MRR<br>H@1<br>H@10<br>Time|
|---|---|---|
||||
|Fact:Train = 0.70<br>Fact:Train = 0.80<br>Fact:Train = 0.90<br>Fact:Train = 0.95<br>Fact:Train = 0.99|0.566<br>51.1<br>66.7<br>22.9min<br>0.561<br>50.8<br>66.0<br>16.2min<br>0.562<br>51.1<br>66.1<br>8.7min<br>0.567<br>51.4<br>66.6<br>4.5min<br>0.563<br>51.2<br>66.3<br>0.9min|0.552<br>46.9<br>69.9<br>55.6h<br>0.563<br>48.4<br>70.7<br>42.0h<br>0.586<br>51.4<br>71.8<br>23.2h<br>0.587<br>51.7<br>71.1<br>12.1h<br>0.598<br>52.9<br>71.9<br>2.5h|



Table 14: Comparison of effectiveness with regard to subgraph sampling. 

|_rq_<br>_V_<br>_rq_<br>_E_|WN18RR<br>MRR<br>H@1<br>H@10|NELL-995<br>MRR<br>H@1<br>H@10|YAGO3-10<br>MRR<br>H@1<br>H@10|
|---|---|---|---|
|||||
|1.0<br>1.0<br>0.5<br>0.5<br>0.2<br>0.2<br>0.1<br>0.1|0.549<br>50.2<br>63.5<br>0.555<br>50.6<br>64.4<br>0.563<br>51.0<br>66.0<br>0.567<br>51.4<br>66.4|0.507<br>43.8<br>61.9<br>0.537<br>47.3<br>64.2<br>0.541<br>47.9<br>63.9<br>0.539<br>48.0<br>63.0|0.598<br>53.3<br>71.2<br>0.599<br>53.4<br>71.4<br>0.603<br>53.8<br>71.8<br>0.599<br>53.6<br>70.8|



20 

Published as a conference paper at ICLR 2024 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2024_Less_is_More_One-shot_Subgraph_Reasoning_on_Large-scale_Knowledge_Graphs_images/iclr_2024_Less_is_More_One-shot_Subgraph_Reasoning_on_Large-scale_Knowledge_Graphs.pdf-0021-01.png)


Figure 8: Illustrations of structural models (a) and decoupled subgraph models (b). 

## C.4 ROBUSTNESS AGAINST DATA PERTURBATIONS 

**Randomly adding noise.** We generate a noisy dataset by randomly adding noisy edges that are not observed from the dataset. Specifically, the entity set and relation set are kept the same. The noise ratio is computed as _ϵ_ = _[|E]_[noise] _[|] /|E_[train] _|_ . The _E_[val] and _E_[test] are clean to guarantee the accurate evaluation. 

As the results shown in Tab. 15, such noise does not significantly influence the prediction performance. Even in the high-noise scenario ( _ϵ_ = 50%), the MRR result is nearly identical to that on the clean dataset. This result shows the preliminary robustness of adding noise, and a further investigation should be conducted with more complex noise patterns, e.g., by adopting adversarial techniques. 

Table 15: Model performance (MRR) with randomly added noise. 

||Table 15: Model performance (MRR) with randomly added noise.|
|---|---|
|Dataset|_ϵ_= 0%(clean)<br>_ϵ_= 10%<br>_ϵ_= 20%<br>_ϵ_= 30%<br>_ϵ_= 40%<br>_ϵ_= 50%|
|||
|WN18RR<br>NELL-995<br>YAGO3-10|0.567<br>0.567<br>0.566<br>0.566<br>0.564<br>0.564<br>0.547<br>0.547<br>0.547<br>0.547<br>0.546<br>0.547<br>0.606<br>0.605<br>0.603<br>0.601<br>0.602<br>0.601|



**Randomly deleting facts.** We create a more incomplete KG by randomly deleting the facts in the training set. The delete ratio is computed as _r_ = _[|E]_[delete] _[|] /|E_[train] _|_ . The _E_[val] and _E_[test] are kept the same as the original dataset. As shown in Tab. 16, this kind of input perturbation greatly degenerates the performance, as the link prediction could rely heavily on the correct edges in the original dataset. Here, a sparser graph with more deleted edges can not sufficiently support the prediction, and the message propagation from the query entity is also hindered. Hence, combating graph incompleteness, data heterogeneity (Tang et al., 2022b;a), noisy annotations (Zhou et al., 2023a), or adversarial attacks (Zhang et al., 2022a; Chen et al., 2022; Zhou et al., 2023b; Zhang et al., 2023a; Li et al., 2023), would be a valuable direction. 

Table 16: Model performance (MRR) with randomly added noise. 

||Table 16: Model performance (MRR) with randomly added noise.|
|---|---|
|Dataset|_r_ = 0%(full)<br>_r_ = 10%<br>_r_ = 20%<br>_r_ = 30%<br>_r_ = 40%<br>_r_ = 50%|
|WN18RR<br>0.567<br>0.507<br>0.451<br>0.394<br>0.338<br>0.278<br>NELL-995<br>0.547<br>0.515<br>0.483<br>0.446<br>0.412<br>0.365<br>YAGO3-10<br>0.606<br>0.553<br>0.489<br>0.434<br>0.385<br>0.329||



## D FURTHER DISCUSSION 

## D.1 DESIGN PRINCIPLES 

Note that it is non-trivial to achieve the one-shot-subgraph link prediction, where three-fold questions are required to be answered: **(i)** From the _data_ ’s perspective, what kind of sampler is suitable here? **(ii)** from the _model_ ’s perspective, how to build up the predictor’s architecture to be expressive on subgraphs? **(iii)** from the _optimization_ ’s perspective, how to optimize the sampler and predictor jointly and efficiently? 

21 

Published as a conference paper at ICLR 2024 

The major technical challenge lies in the design and implementation of an efficient and querydependent sampler. Given a query ( _u, q,_ ?), the sampler should not only preserve the local neighbors of _u_ that contain the potential answers and supporting facts, but also be able to distinguish different _q_ as the essential information could be different. Accordingly, the two design principles are as follows. 

**Principle-1: Local-structure-preserving.** Since the target entity _v_ is unknown, freely sampling from all the entities in original _G_ will inevitably discard the structural connection between _u_ and promising _v_ . This manner of existing node-wise or layer-wise sampling methods can deteriorate the message flow started from _u_ , and thus degenerate the whole prediction process. Here, the local structure of _u_ is expected to be preserved in _Gs_ , and each sampled entity should be reachable from _u_ . 

**Principle-2: Query-relation-aware.** The objective of learning relation-aware sampling here is to sample the promising targets _v_ according to the query relation _q_ . Note that, in common homogeneous graphs, the relation may be unnecessary, as all relations between nodes share the same semantics. However, the relation here does matter, as it presents the distinct characteristics of a KG. Thus, in our design, the sampler is expected to be query-relation-aware in order to adapt to KGs efficiently. 

## D.2 CONTRIBUTIONS 

Here, we further explain the novelty and contributions in the following three folds. 

**A valuable research problem.** Our investigation delves into the limitations of structural models, which confront a pronounced scalability challenge. These models rely on prediction over the entire Knowledge Graph, encompassing all entities and edges, with the additional burden of scoring all entities as potential answers. This approach proves to be highly inefficient, impeding their optimization when applied to large-scale KGs. As a consequence, we present an open question, pondering how to conduct prediction on knowledge graphs efficiently and effectively, seeking avenues to overcome this hindrance. 

**A conceptual framework with several practical instantiations.** In response to the prevalent scalability challenges faced by existing KG methods, we present a conceptual solution, _i.e._ , the one-shot-subgraph link prediction manner. This novel approach promises enhanced flexibility and efficacy in design. Specifically, instead of conducting direct prediction on the complete original KG, we advocate a two-step prediction process involving subgraph sampling and subgraph-based prediction. In this regard, our proposed prediction manner comprises a sampler and a predictor. Leveraging the efficiency gains derived from subgraph-based prediction, we introduce an optimization technique for subgraph-based searching, incorporating several well-crafted technical designs. These innovations aim to strike a balance between prediction efficiency and the complexity associated with identifying optimal configurations within both data and model spaces. 

**Several important discoveries from experiments.** Through comprehensive experimentation on three prevalent KGs, we demonstrate that our framework achieves state-of-the-art performance. Particularly noteworthy are the substantial advancements we achieve in both efficiency and effectiveness, a trend that is particularly evident in the case of the large-scale dataset YAGO3-10. Our quest for optimal configurations leads us to the intriguing revelation that utilizing the entire KG for prediction is unnecessary. Instead, simple heuristics can efficiently identify a small proportion of entities and facts essential for answering specific queries without the need for additional learning. These compelling findings hold significant meaning and are poised to pique the interest of the KG community. 

## D.3 EXPLANATION FOR THE IMPROVEMENT IN PERFORMANCE 

Here, we provide a two-fold explanation for better performance as follows. 

**Data perspective:** Extracting subgraphs can remove irrelevant information for link prediction. Conventional structural models explicitly take all the entities and edges into prediction, ignoring the correlation between the entities and the query relation. As delineated in Sec. 5 of this paper, our one-shot-subgraph link prediction approach effectively discerns and excludes irrelevant entities while retaining the proper answers. This strategic refinement, in turn, contributes to simplifying the learning problem, thereby amplifying prediction performance. Our findings demonstrate that a mere fraction ( _i.e._ , 10%) of entities suffices for answering specific queries, _i.e._ , a subset efficiently identified by the heuristic Personalized PageRank mechanism without the need for learnable sampling. 

22 

Published as a conference paper at ICLR 2024 

Another supporting material is that only relying on a subgraph for prediction can also boost the test-time performance (Miao et al., 2022). The proposed GAST method (Miao et al., 2022), aims to extract a subgraph _Gs_ as the interpretation of a GNN. It inherits the same spirit of information bottleneck in building its optimization objective, _i.e._ , min _−I_ ( _Gs_ ; _Y_ ) + _β · I_ ( _Gs_ ; _G_ ). The integrated subgraph sampler can explicitly remove the spurious correlation or noisy information in the entire graph _G_ , which is similar to our one-shot-subgraph link prediction framework. 

**Model perspective:** The higher learning efficiency on subgraphs can further boost hyper-parameter optimization (Zhang et al., 2022b). Generally, the task of hyperparameter optimization on extensive graph datasets poses substantial challenges due to the inherent inefficiencies in model training. However, the implementation of our subgraph-based method results in a substantial improvement in training efficiency. 

Consequently, we can rapidly obtain evaluation feedback for configurations sampled from the hyperparameter space. Sec. 4.2 introduces an optimization technique for subgraph-based searching that features meticulously crafted technical designs. The searched configuration usually leads to a deeper GNN ( _i.e._ , 8 layers, while previous studies are usually limited to 5 or 6 layers) that increases the expressiveness of prediction. These innovations aim to balance prediction efficiency and the complexity associated with identifying optimal configurations within data and model spaces, which also contributes to improved prediction performance. 

## D.4 EXTENSION 

Note that KG learning usually focuses more on link-level tasks, _e.g._ , link prediction. However, we firmly believe that one-shot-subgraph link prediction has the potential to be extended and adapted for various graph learning tasks. One general direction is to adapt the one-shot-subgraph link prediction framework to other kinds of graph learning tasks, _e.g._ , the node-level or graph-level tasks. 

For instance, with the PPR sampler, one can sample a single-source "local" subgraph for node classification or a multi-source "global" subgraph for graph classification, where the rationale of first sampling and then prediction remains applicable. Besides, enhancing the one-shot-subgraph link prediction with instance-wise adaptation is also a promising direction. That is, sampling a subgraph of suitable scale for each given query, which can potentially improve the upper limit of prediction. 

Furthermore, conducting link prediction with new relations or new entities is also a frontier topic. Improving the generalization or extrapolation power of GNN can be vital in practice. Considering the significant few-shot in-context learning of the large language model (LLM), an appropriate synergy between the latest LLM and current GNN will be a promising direction. Improving the efficiency and scalability of predicting with large graphs is also of great importance here. 

In addition, from a broader perspective of trustworthy machine learning, one should also consider the intrinsic interpretability and the robustness problem. These trustworthy properties can help users understand the model better and also keep it in a safe and controllable way. 

23 

Published as a conference paper at ICLR 2024 

## E CASE STUDY 

In this section, we show the sampled subgraph of different scales on three datasets as follows. 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2024_Less_is_More_One-shot_Subgraph_Reasoning_on_Large-scale_Knowledge_Graphs_images/iclr_2024_Less_is_More_One-shot_Subgraph_Reasoning_on_Large-scale_Knowledge_Graphs.pdf-0024-03.png)


**----- Start of picture text -----**<br>
0.10<br>0.75 0.06 1.5<br>0.05 0.50 0.04 1.0<br>0.25 0.02 0.5<br>0.00 0 100 200 300 0.00 0 2 4 6 8 10 0.00 0 200 400 0 . 0 0 2 4 6 8 10<br>Degree of sampled entities Distance to the query entity Degree of sampled entities Distance to the query entity<br>Figure 9: Subgraphs (0.1% and 1%) from WN18RR:  u =1 , q =12 , v = { 5305 } .<br>0.10 0.6 0.06 1.5<br>0.05 0.4 0.04 1.0<br>0.2 0.02 0.5<br>0.00 0 10 20 30 40 0.0 0 2 4 6 8 10 0.00 0 200 400 0 .0 0 2 4 6 8 10<br>Degree of sampled entities Distance to the query entity Degree of sampled entities Distance to the query entity<br>Figure 10: Subgraphs (0.1% and 1%) from WN18RR:  u =9 , q =20 , v = { 38116 } .<br>0.15 0.075<br>0.10 1.0 0.050 1.0<br>0.05 0.5 0.025 0.5<br>0.00 0 10 20 30 0.0 0 2 4 6 8 10 0.000 0 200 400 0 . 0 0 2 4 6 8 10<br>Degree of sampled entities Distance to the query entity Degree of sampled entities Distance to the query entity<br>Figure 11: Subgraphs (0.1% and 1%) from WN18RR:  u =29 , q =1 , v = { 11186 } .<br>Density Density Density Density<br>Density Density Density Density<br>Density Density Density Density<br>**----- End of picture text -----**<br>


24 

Published as a conference paper at ICLR 2024 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2024_Less_is_More_One-shot_Subgraph_Reasoning_on_Large-scale_Knowledge_Graphs_images/iclr_2024_Less_is_More_One-shot_Subgraph_Reasoning_on_Large-scale_Knowledge_Graphs.pdf-0025-01.png)


**----- Start of picture text -----**<br>
0.15 0.075<br>0.10 1.0 0.050 1.0<br>0.05 0.5 0.025 0.5<br>0.00 0 10 20 30 0.0 0 2 4 6 8 10 0.000 0 200 400 0 . 0 0 2 4 6 8 10<br>Degree of sampled entities Distance to the query entity Degree of sampled entities Distance to the query entity<br>Density Density Density Density<br>**----- End of picture text -----**<br>


Figure 12: Subgraphs (0.1% and 1%) from WN18RR: _u_ =29 _, q_ =12 _, v_ = _{_ 6226 _}_ . 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2024_Less_is_More_One-shot_Subgraph_Reasoning_on_Large-scale_Knowledge_Graphs_images/iclr_2024_Less_is_More_One-shot_Subgraph_Reasoning_on_Large-scale_Knowledge_Graphs.pdf-0025-03.png)


**----- Start of picture text -----**<br>
0.075 1.5 0.06 1.5<br>0.050 1.0 0.04 1.0<br>0.025 0.5 0.02 0.5<br>0.000 100 0 100 200 300 400 0.0 0 2 4 6 8 10 0.00 0 200 400 0 . 0 0 2 4 6 8 10<br>Degree of sampled entities Distance to the query entity Degree of sampled entities Distance to the query entity<br>Density Density Density Density<br>**----- End of picture text -----**<br>


Figure 13: Subgraphs (0.1% and 1%) from WN18RR: _u_ =44 _, q_ =12 _, v_ = _{_ 45 _}_ . 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2024_Less_is_More_One-shot_Subgraph_Reasoning_on_Large-scale_Knowledge_Graphs_images/iclr_2024_Less_is_More_One-shot_Subgraph_Reasoning_on_Large-scale_Knowledge_Graphs.pdf-0025-05.png)


**----- Start of picture text -----**<br>
0.10 1.0 0.06 1.5<br>0.05 0.5 0.04 1.0<br>0.02 0.5<br>0.00 0 50 100 0.0 0 2 4 6 8 10 0.00 0 200 400 0 . 0 0 2 4 6 8 10<br>Degree of sampled entities Distance to the query entity Degree of sampled entities Distance to the query entity<br>Figure 14: Subgraphs (0.1% and 1%) from WN18RR:  u =45 , q =1 , v = { 44 } .<br>Density Density Density Density<br>**----- End of picture text -----**<br>



![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2024_Less_is_More_One-shot_Subgraph_Reasoning_on_Large-scale_Knowledge_Graphs_images/iclr_2024_Less_is_More_One-shot_Subgraph_Reasoning_on_Large-scale_Knowledge_Graphs.pdf-0025-06.png)


**----- Start of picture text -----**<br>
0.03 1.0 0.075<br>1.5<br>0.02 0.5 0.050 1.0<br>0.01 0.025 0.5<br>0.00 0 200 400 600 0.0 0 2 4 6 8 10 0.000 0 200 400 0 . 0 0 2 4 6 8 10<br>Degree of sampled entities Distance to the query entity Degree of sampled entities Distance to the query entity<br>Figure 15: Subgraphs (0.1% and 1%) from WN18RR:  u =60 , q =2 , v = { 6577 } .<br>Density Density Density Density<br>**----- End of picture text -----**<br>


25 

Published as a conference paper at ICLR 2024 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2024_Less_is_More_One-shot_Subgraph_Reasoning_on_Large-scale_Knowledge_Graphs_images/iclr_2024_Less_is_More_One-shot_Subgraph_Reasoning_on_Large-scale_Knowledge_Graphs.pdf-0026-01.png)


**----- Start of picture text -----**<br>
0.075 0.75 0.075 3<br>0.050 0.50 0.050 2<br>0.025 0.25 0.025 1<br>0.000 0 200 400 600 0.00 0 2 4 6 8 10 0.000 0 200 400 0 0 2 4 6 8 10<br>Degree of sampled entities Distance to the query entity Degree of sampled entities Distance to the query entity<br>Density Density Density Density<br>**----- End of picture text -----**<br>


Figure 16: Subgraphs (0.1% and 1%) from WN18RR: _u_ =78 _, q_ =5 _, v_ = _{_ 172 _}_ . 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2024_Less_is_More_One-shot_Subgraph_Reasoning_on_Large-scale_Knowledge_Graphs_images/iclr_2024_Less_is_More_One-shot_Subgraph_Reasoning_on_Large-scale_Knowledge_Graphs.pdf-0026-03.png)


**----- Start of picture text -----**<br>
0.06 1.0 0.06 1.5<br>0.04 0.04 1.0<br>0.02 0.5 0.02 0.5<br>0.00 0 100 200 300 0.0 0 2 4 6 8 10 0.00 0 200 400 0 . 0 0 2 4 6 8 10<br>Degree of sampled entities Distance to the query entity Degree of sampled entities Distance to the query entity<br>Density Density Density Density<br>**----- End of picture text -----**<br>


Figure 17: Subgraphs (0.1% and 1%) from WN18RR: _u_ =88 _, q_ =12 _, v_ = _{_ 4621 _}_ . 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2024_Less_is_More_One-shot_Subgraph_Reasoning_on_Large-scale_Knowledge_Graphs_images/iclr_2024_Less_is_More_One-shot_Subgraph_Reasoning_on_Large-scale_Knowledge_Graphs.pdf-0026-05.png)


**----- Start of picture text -----**<br>
0.075 0.75 0.06 1.5<br>0.050 0.50 0.04 1.0<br>0.025 0.25 0.02 0.5<br>0.000 0 200 400 600 0.00 0 2 4 6 8 10 0.00 0 200 400 0 .0 0 2 4 6 8 10<br>Degree of sampled entities Distance to the query entity Degree of sampled entities Distance to the query entity<br>Figure 18: Subgraphs (0.1% and 1%) from WN18RR:  u =155 , q =19 , v = { 785 } .<br>0.075 0.75 0.06 1.0<br>0.050 0.50 0.04<br>0.025 0.25 0.02 0.5<br>0.000 0 200 400 600 0.00 0 2 4 6 8 10 0.00 0 200 400 0.0 0 2 4 6 8 10<br>Degree of sampled entities Distance to the query entity Degree of sampled entities Distance to the query entity<br>Figure 19: Subgraphs (0.1% and 1%) from WN18RR:  u =3297 , q =1 , v = { 2037 } .<br>Density Density Density Density<br>Density Density Density Density<br>**----- End of picture text -----**<br>


26 

Published as a conference paper at ICLR 2024 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2024_Less_is_More_One-shot_Subgraph_Reasoning_on_Large-scale_Knowledge_Graphs_images/iclr_2024_Less_is_More_One-shot_Subgraph_Reasoning_on_Large-scale_Knowledge_Graphs.pdf-0027-01.png)


**----- Start of picture text -----**<br>
0.010<br>0.0015 0.3 0.6<br>0.0010 0.2 0.005 0.4<br>0.0005 0.1 0.2<br>0.0000 0 1000 2000 3000 0.0 0 2 4 6 8 10 0.000 0 1000 2000 0.0 0 2 4 6 8 10<br>Degree of sampled entities Distance to the query entity Degree of sampled entities Distance to the query entity<br>Density Density Density Density<br>**----- End of picture text -----**<br>


Figure 20: Subgraphs (0.1% and 1%) from NELL-995: _u_ =4 _, q_ =238 _, v_ = _{_ 22677 _}_ . 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2024_Less_is_More_One-shot_Subgraph_Reasoning_on_Large-scale_Knowledge_Graphs_images/iclr_2024_Less_is_More_One-shot_Subgraph_Reasoning_on_Large-scale_Knowledge_Graphs.pdf-0027-03.png)


**----- Start of picture text -----**<br>
0.003 1.5 4<br>0.002 1.0 0.010<br>2<br>0.001 0.5 0.005<br>0.000 0 1000 2000 3000 0.0 0 2 4 6 8 10 0.000 0 1000 2000 0 0 2 4 6 8 10<br>Degree of sampled entities Distance to the query entity Degree of sampled entities Distance to the query entity<br>Density Density Density Density<br>**----- End of picture text -----**<br>


Figure 21: Subgraphs (0.1% and 1%) from NELL-995: _u_ =17 _, q_ =274 _, v_ = _{_ 57735 _,_ 61381 _,_ 63044 _}_ . 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2024_Less_is_More_One-shot_Subgraph_Reasoning_on_Large-scale_Knowledge_Graphs_images/iclr_2024_Less_is_More_One-shot_Subgraph_Reasoning_on_Large-scale_Knowledge_Graphs.pdf-0027-05.png)


**----- Start of picture text -----**<br>
1.5 4<br>0.0075 1.0 0.010<br>0.0050 2<br>0.0025 0.5 0.005<br>0.0000 0 1000 2000 3000 0.0 0 2 4 6 8 10 0.000 0 1000 2000 0 0 2 4 6 8 10<br>Degree of sampled entities Distance to the query entity Degree of sampled entities Distance to the query entity<br>Figure 22: Subgraphs (0.1% and 1%) from NELL-995: u =29 , q =260 , v = { 27725 ,  73985 } .<br>Density Density Density Density<br>**----- End of picture text -----**<br>



![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2024_Less_is_More_One-shot_Subgraph_Reasoning_on_Large-scale_Knowledge_Graphs_images/iclr_2024_Less_is_More_One-shot_Subgraph_Reasoning_on_Large-scale_Knowledge_Graphs.pdf-0027-06.png)


**----- Start of picture text -----**<br>
0.015 2<br>1.5<br>0.010 1.0 0.010 1<br>0.005 0.5 0.005<br>0.000 0 1000 2000 3000 0.0 0 2 4 6 8 10 0.000 0 1000 2000 0 0 2 4 6 8 10<br>Degree of sampled entities Distance to the query entity Degree of sampled entities Distance to the query entity<br>Figure 23: Subgraphs (0.1% and 1%) from NELL-995: u =44 , q =222 , v = { 11669 } .<br>Density Density Density Density<br>**----- End of picture text -----**<br>


27 

Published as a conference paper at ICLR 2024 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2024_Less_is_More_One-shot_Subgraph_Reasoning_on_Large-scale_Knowledge_Graphs_images/iclr_2024_Less_is_More_One-shot_Subgraph_Reasoning_on_Large-scale_Knowledge_Graphs.pdf-0028-01.png)


**----- Start of picture text -----**<br>
0.003 1.5 0.010 2<br>0.002 1.0<br>0.001 0.5 0.005 1<br>0.000 0 1000 2000 3000 0.0 0 2 4 6 8 10 0.000 0 1000 2000 0 0 2 4 6 8 10<br>Degree of sampled entities Distance to the query entity Degree of sampled entities Distance to the query entity<br>Density Density Density Density<br>**----- End of picture text -----**<br>


Figure 24: Subgraphs (0.1% and 1%) from NELL-995: _u_ =60 _, q_ =74 _, v_ = _{_ 164 _}_ . 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2024_Less_is_More_One-shot_Subgraph_Reasoning_on_Large-scale_Knowledge_Graphs_images/iclr_2024_Less_is_More_One-shot_Subgraph_Reasoning_on_Large-scale_Knowledge_Graphs.pdf-0028-03.png)


**----- Start of picture text -----**<br>
3<br>0.0075 1.0 0.010<br>0.0050 2<br>0.0025 0.5 0.005 1<br>0.0000 0 1000 2000 3000 0.0 0 2 4 6 8 10 0.000 0 1000 2000 0 0 2 4 6 8 10<br>Degree of sampled entities Distance to the query entity Degree of sampled entities Distance to the query entity<br>Density Density Density Density<br>**----- End of picture text -----**<br>


Figure 25: Subgraphs (0.1% and 1%) from NELL-995: _u_ =166 _, q_ =260 _, v_ = _{_ 6364 _}_ . 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2024_Less_is_More_One-shot_Subgraph_Reasoning_on_Large-scale_Knowledge_Graphs_images/iclr_2024_Less_is_More_One-shot_Subgraph_Reasoning_on_Large-scale_Knowledge_Graphs.pdf-0028-05.png)


**----- Start of picture text -----**<br>
1.5 6<br>0.0015 0.015<br>0.0010 1.0 0.010 4<br>0.0005 0.5 0.005 2<br>0.0000 0 1000 2000 3000 0.0 0 2 4 6 8 10 0.000 0 1000 2000 0 0 2 4 6 8 10<br>Degree of sampled entities Distance to the query entity Degree of sampled entities Distance to the query entity<br>Figure 26: Subgraphs (0.1% and 1%) from NELL-995: u =202 , q =101 , v = { 399 } .<br>0.010 2 0.010 2<br>0.005 1 0.005 1<br>0.000 0 1000 2000 3000 0 0 2 4 6 8 10 0.000 0 1000 2000 0 0 2 4 6 8 10<br>Degree of sampled entities Distance to the query entity Degree of sampled entities Distance to the query entity<br>Density Density Density Density<br>Density Density Density Density<br>**----- End of picture text -----**<br>


Figure 27: Subgraphs (0.1% and 1%) from NELL-995: _u_ =255 _, q_ =232 _, v_ = _{_ 1631 _,_ 9925 _,_ 11229 _}_ . 

28 

Published as a conference paper at ICLR 2024 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2024_Less_is_More_One-shot_Subgraph_Reasoning_on_Large-scale_Knowledge_Graphs_images/iclr_2024_Less_is_More_One-shot_Subgraph_Reasoning_on_Large-scale_Knowledge_Graphs.pdf-0029-01.png)


**----- Start of picture text -----**<br>
3<br>0.0075 1.0 0.010<br>0.0050 2<br>0.0025 0.5 0.005 1<br>0.0000 0 1000 2000 3000 0.0 0 2 4 6 8 10 0.000 0 1000 2000 0 0 2 4 6 8 10<br>Degree of sampled entities Distance to the query entity Degree of sampled entities Distance to the query entity<br>Density Density Density Density<br>**----- End of picture text -----**<br>


Figure 28: Subgraphs (0.1% and 1%) from NELL-995: _u_ =1371 _, q_ =260 _, v_ = _{_ 24193 _,_ 50385 _,_ 60718 _}_ . 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2024_Less_is_More_One-shot_Subgraph_Reasoning_on_Large-scale_Knowledge_Graphs_images/iclr_2024_Less_is_More_One-shot_Subgraph_Reasoning_on_Large-scale_Knowledge_Graphs.pdf-0029-03.png)


**----- Start of picture text -----**<br>
0.004 1.0 0.010 2<br>0.002 0.5 0.005 1<br>0.000 0 1000 2000 3000 0.0 0 2 4 6 8 10 0.000 0 1000 2000 0 0 2 4 6 8 10<br>Degree of sampled entities Distance to the query entity Degree of sampled entities Distance to the query entity<br>Density Density Density Density<br>**----- End of picture text -----**<br>


Figure 29: Subgraphs (0.1% and 1%) from NELL-995: _u_ =11200 _, q_ =38 _, v_ = _{_ 5737 _,_ 7292 _,_ 11199 _}_ . 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2024_Less_is_More_One-shot_Subgraph_Reasoning_on_Large-scale_Knowledge_Graphs_images/iclr_2024_Less_is_More_One-shot_Subgraph_Reasoning_on_Large-scale_Knowledge_Graphs.pdf-0029-05.png)


**----- Start of picture text -----**<br>
0.00075 1.5 0.00075<br>0.00050 1.0 0.00050 2<br>0.00025 0.5 0.00025 1<br>0.00000 0 20000 40000 60000 0.0 0 2 4 6 8 10 0.00000 0 20000 40000 60000 0 0 2 4 6 8 10<br>Degree of sampled entities Distance to the query entity Degree of sampled entities Distance to the query entity<br>Figure 30: Subgraphs (0.1% and 1%) from YAGO3-10: u =17 , q =39 , v = { 54968 } .<br>0.00075 0.00075 3<br>0.00050 1.0 0.00050 2<br>0.00025 0.5 0.00025 1<br>0.00000 0 20000 40000 60000 0.0 0 2 4 6 8 10 0.00000 0 20000 40000 60000 0 0 2 4 6 8 10<br>Degree of sampled entities Distance to the query entity Degree of sampled entities Distance to the query entity<br>Figure 31: Subgraphs (0.1% and 1%) from YAGO3-10: u =20 , q =2 , v = { 34580 } .<br>Density Density Density Density<br>Density Density Density Density<br>**----- End of picture text -----**<br>


29 

Published as a conference paper at ICLR 2024 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2024_Less_is_More_One-shot_Subgraph_Reasoning_on_Large-scale_Knowledge_Graphs_images/iclr_2024_Less_is_More_One-shot_Subgraph_Reasoning_on_Large-scale_Knowledge_Graphs.pdf-0030-01.png)


**----- Start of picture text -----**<br>
0.00075 1.5 6<br>0.00075<br>0.00050 1.0 0.00050 4<br>0.00025 0.5 0.00025 2<br>0.00000 0 20000 40000 60000 0.0 0 2 4 6 8 10 0.00000 0 20000 40000 60000 0 0 2 4 6 8 10<br>Degree of sampled entities Distance to the query entity Degree of sampled entities Distance to the query entity<br>Density Density Density Density<br>**----- End of picture text -----**<br>


Figure 32: Subgraphs (0.1% and 1%) from YAGO3-10: _u_ =25 _, q_ =37 _, v_ = _{_ 40490 _}_ . 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2024_Less_is_More_One-shot_Subgraph_Reasoning_on_Large-scale_Knowledge_Graphs_images/iclr_2024_Less_is_More_One-shot_Subgraph_Reasoning_on_Large-scale_Knowledge_Graphs.pdf-0030-03.png)


**----- Start of picture text -----**<br>
0.00075 1.5 0.00075 4<br>0.00050 1.0 0.00050<br>0.00025 0.5 0.00025 2<br>0.00000 0 20000 40000 60000 0.0 0 2 4 6 8 10 0.00000 0 20000 40000 60000 0 0 2 4 6 8 10<br>Degree of sampled entities Distance to the query entity Degree of sampled entities Distance to the query entity<br>Density Density Density Density<br>**----- End of picture text -----**<br>


Figure 33: Subgraphs (0.1% and 1%) from YAGO3-10: _u_ =27 _, q_ =39 _, v_ = _{_ 3801 _}_ . 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2024_Less_is_More_One-shot_Subgraph_Reasoning_on_Large-scale_Knowledge_Graphs_images/iclr_2024_Less_is_More_One-shot_Subgraph_Reasoning_on_Large-scale_Knowledge_Graphs.pdf-0030-05.png)


**----- Start of picture text -----**<br>
0.00075 1.5 0.00075<br>2<br>0.00050 1.0 0.00050<br>0.00025 0.5 0.00025 1<br>0.00000 0 20000 40000 60000 0.0 0 2 4 6 8 10 0.00000 0 20000 40000 60000 0 0 2 4 6 8 10<br>Degree of sampled entities Distance to the query entity Degree of sampled entities Distance to the query entity<br>Figure 34: Subgraphs (0.1% and 1%) from YAGO3-10: u =29 , q =38 , v = { 33723 ,  82573 } .<br>0.00075 1.5 0.00075<br>2<br>0.00050 1.0 0.00050<br>0.00025 0.5 0.00025 1<br>0.00000 0 20000 40000 60000 0.0 0 2 4 6 8 10 0.00000 0 20000 40000 60000 0 0 2 4 6 8 10<br>Degree of sampled entities Distance to the query entity Degree of sampled entities Distance to the query entity<br>Figure 35: Subgraphs (0.1% and 1%) from YAGO3-10: u =55 , q =38 , v = { 14834 ,  67740 } .<br>Density Density Density Density<br>Density Density Density Density<br>**----- End of picture text -----**<br>


30 

Published as a conference paper at ICLR 2024 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2024_Less_is_More_One-shot_Subgraph_Reasoning_on_Large-scale_Knowledge_Graphs_images/iclr_2024_Less_is_More_One-shot_Subgraph_Reasoning_on_Large-scale_Knowledge_Graphs.pdf-0031-01.png)


**----- Start of picture text -----**<br>
0.00075 1.5 0.00075<br>2<br>0.00050 1.0 0.00050<br>0.00025 0.5 0.00025 1<br>0.00000 0 20000 40000 60000 0.0 0 2 4 6 8 10 0.00000 0 20000 40000 60000 0 0 2 4 6 8 10<br>Degree of sampled entities Distance to the query entity Degree of sampled entities Distance to the query entity<br>Density Density Density Density<br>**----- End of picture text -----**<br>


Figure 36: Subgraphs (0.1% and 1%) from YAGO3-10: _u_ =102 _, q_ =12 _, v_ = _{_ 15823 _}_ . 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2024_Less_is_More_One-shot_Subgraph_Reasoning_on_Large-scale_Knowledge_Graphs_images/iclr_2024_Less_is_More_One-shot_Subgraph_Reasoning_on_Large-scale_Knowledge_Graphs.pdf-0031-03.png)


**----- Start of picture text -----**<br>
0.00075 0.00075 3<br>0.00050 1.0 0.00050 2<br>0.00025 0.5 0.00025 1<br>0.00000 0 20000 40000 60000 0.0 0 2 4 6 8 10 0.00000 0 20000 40000 60000 0 0 2 4 6 8 10<br>Degree of sampled entities Distance to the query entity Degree of sampled entities Distance to the query entity<br>Density Density Density Density<br>**----- End of picture text -----**<br>


Figure 37: Subgraphs (0.1% and 1%) from YAGO3-10: _u_ =108 _, q_ =39 _, v_ = _{_ 7271 _}_ . 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2024_Less_is_More_One-shot_Subgraph_Reasoning_on_Large-scale_Knowledge_Graphs_images/iclr_2024_Less_is_More_One-shot_Subgraph_Reasoning_on_Large-scale_Knowledge_Graphs.pdf-0031-05.png)


**----- Start of picture text -----**<br>
0.00075 1.5 0.00075 2<br>0.00050 1.0 0.00050<br>1<br>0.00025 0.5 0.00025<br>0.00000 0 20000 40000 60000 0.0 0 2 4 6 8 10 0.00000 0 20000 40000 60000 0 0 2 4 6 8 10<br>Degree of sampled entities Distance to the query entity Degree of sampled entities Distance to the query entity<br>Figure 38: Subgraphs (0.1% and 1%) from YAGO3-10: u =135 , q =38 , v = { 51096 } .<br>0.00075 0.00075<br>0.00050 1.0 0.00050 2<br>0.00025 0.5 0.00025 1<br>0.00000 0 20000 40000 60000 0.0 0 2 4 6 8 10 0.00000 0 20000 40000 60000 0 0 2 4 6 8 10<br>Degree of sampled entities Distance to the query entity Degree of sampled entities Distance to the query entity<br>Figure 39: Subgraphs (0.1% and 1%) from YAGO3-10: u =137 , q =39 , v = { 26722 } .<br>Density Density Density Density<br>Density Density Density Density<br>**----- End of picture text -----**<br>


31 

Published as a conference paper at ICLR 2024 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2024_Less_is_More_One-shot_Subgraph_Reasoning_on_Large-scale_Knowledge_Graphs_images/iclr_2024_Less_is_More_One-shot_Subgraph_Reasoning_on_Large-scale_Knowledge_Graphs.pdf-0032-01.png)


**----- Start of picture text -----**<br>
0.00075 0.00075 1.0<br>1.0<br>0.00050 0.00050<br>0.00025 0.5 0.00025 0.5<br>0.00000 0 20000 40000 60000 0.0 0 2 4 6 8 10 0.00000 0 20000 40000 60000 0.0 0 2 4 6 8 10<br>Degree of sampled entities Distance to the query entity Degree of sampled entities Distance to the query entity<br>Density Density Density Density<br>**----- End of picture text -----**<br>


Figure 40: Subgraphs (0.1% and 1%) from YAGO3-10: _u_ =446 _, q_ =38 _, v_ = _{_ 104297 _}_ . 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2024_Less_is_More_One-shot_Subgraph_Reasoning_on_Large-scale_Knowledge_Graphs_images/iclr_2024_Less_is_More_One-shot_Subgraph_Reasoning_on_Large-scale_Knowledge_Graphs.pdf-0032-03.png)


**----- Start of picture text -----**<br>
0.00075 0.00075<br>1.0 2<br>0.00050 0.00050<br>0.00025 0.5 0.00025 1<br>0.00000 0 20000 40000 60000 0.0 0 2 4 6 8 10 0.00000 0 20000 40000 60000 0 0 2 4 6 8 10<br>Degree of sampled entities Distance to the query entity Degree of sampled entities Distance to the query entity<br>Density Density Density Density<br>**----- End of picture text -----**<br>


Figure 41: Subgraphs (0.1% and 1%) from YAGO3-10: _u_ =1072 _, q_ =1 _, v_ = _{_ 23394 _}_ . 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2024_Less_is_More_One-shot_Subgraph_Reasoning_on_Large-scale_Knowledge_Graphs_images/iclr_2024_Less_is_More_One-shot_Subgraph_Reasoning_on_Large-scale_Knowledge_Graphs.pdf-0032-05.png)


**----- Start of picture text -----**<br>
0.00075 4 0.00075<br>2<br>0.00050 0.00050<br>0.00025 2 0.00025 1<br>0.00000 0 20000 40000 60000 0 0 2 4 6 8 10 0.00000 0 20000 40000 60000 0 0 2 4 6 8 10<br>Degree of sampled entities Distance to the query entity Degree of sampled entities Distance to the query entity<br>Density Density Density Density<br>**----- End of picture text -----**<br>


Figure 42: Subgraphs (0.1% and 1%) from YAGO3-10: _u_ =1255 _, q_ =38 _, v_ = _{_ 12418 _,_ 28138 _,_ 71366 _}_ . 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2024_Less_is_More_One-shot_Subgraph_Reasoning_on_Large-scale_Knowledge_Graphs_images/iclr_2024_Less_is_More_One-shot_Subgraph_Reasoning_on_Large-scale_Knowledge_Graphs.pdf-0032-07.png)


**----- Start of picture text -----**<br>
0.00075 1.5 0.00075<br>2<br>0.00050 1.0 0.00050<br>0.00025 0.5 0.00025 1<br>0.00000 0 20000 40000 60000 0.0 0 2 4 6 8 10 0.00000 0 20000 40000 60000 0 0 2 4 6 8 10<br>Degree of sampled entities Distance to the query entity Degree of sampled entities Distance to the query entity<br>Figure 43: Subgraphs (0.1% and 1%) from YAGO3-10: u =2252 , q =39 , v = { 9476 ,  77502 } .<br>Density Density Density Density<br>**----- End of picture text -----**<br>


32 

