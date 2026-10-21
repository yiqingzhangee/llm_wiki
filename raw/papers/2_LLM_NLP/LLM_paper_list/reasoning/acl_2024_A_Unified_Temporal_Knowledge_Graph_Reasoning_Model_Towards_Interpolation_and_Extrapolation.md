# **A Unified Temporal Knowledge Graph Reasoning Model Towards Interpolation and Extrapolation** 

**Kai Chen**[1] _[∗]_ **, Ye Wang**[1] _[∗]_ **, Yitong Li**[2] _[†]_ **, Aiping Li**[1] _[†]_ **, Han Yu**[1] **, Xin Song**[1] 

1National University of Defense Technology, Changsha, China 2Huawei Technologies Co., Ltd. 

{chenkai_,ye.wang,liaiping,yuhan17,songxin}@nudt.edu.cn liyitong3@huawei.com 

## **Abstract** 

Temporal knowledge graph (TKG) reasoning has two settings: interpolation reasoning and extrapolation reasoning. Both of them draw plenty of research interest and have great significance. Methods of the former deemphasize the temporal correlations among facts sequences, while methods of the latter require strict chronological order of knowledge and ignore inferring clues provided by missing facts of the past. These limit the practicability of TKG applications as almost all of the existing TKG reasoning methods are designed specifically to address either one setting. To this end, this paper proposes an original Temporal PAth-based Reasoning (TPAR) model for both the interpolation and extrapolation reasoning. TPAR performs a neural-driven symbolic reasoning fashion that is robust to ambiguous and noisy temporal data and with fine interpretability as well. Comprehensive experiments show that TPAR outperforms SOTA methods on the link prediction task for both the interpolation and the extrapolation settings. A novel pipeline experimental setting is designed to evaluate the performances of SOTA combinations and the proposed TPAR towards interpolation and extrapolation reasoning. More diverse experiments are conducted to show the robustness and interpretability of TPAR. 

## **1 Introduction** 

Knowledge graph (KG) is a semantic network that represents real-world facts in a structured way using entities and relations (Bollacker et al., 2008; Song et al., 2021; Zhao et al., 2021). Typically, a fact is represented by a triple ( _s, r, o_ ) in KG, consisting of a subject entity _s_ , an object entity _o_ , and a relation _r_ between _s_ and _o_ . 

In the real world, many facts are closely associated with a particular time interval. For example, the fact _“Barack Obama is the president of USA”_ 

> _∗_ Equal contributions. 

> _†_ Corresponding authors. 

is valid for the time period of _2009 January 20th - 2017 January 20th_ and the fact _“Donald Trump is the president of USA”_ is only valid for the following four years. To represent such time-sensitive facts, Temporal Knowledge Graphs (TKGs) have recently gained significant attention from both academic and industrial communities. Specifically, TKGs extend static KGs by incorporating the temporal information _t_ into fact triples, represented as a quadruple ( _s, r, o, t_ ), which allows for modelling the temporal dependencies and evolution of knowledge over time, being crucial for reasoning time-evolving facts in applications such as financial forecasting, social networks, and healthcare. 

TKG reasoning infers new knowledge with timesensitive facts in existing TKGs, which generally has two settings: the interpolation reasoning (Xu et al., 2020; Lacroix et al., 2020; Xu et al., 2021; Chen et al., 2022) and the extrapolation reasoning (Jin et al., 2020; Sun et al., 2021; Liu et al., 2022). Given a temporal knowledge graph with facts from time _t_ 0 to time _tT_ , the interpolation reasoning infers missing facts at any time in history ( _t_ 0 _≤ t ≤ tT_ ) and the extrapolation reasoning attempt to predict unknown facts that may occur in the future ( _t > tT_ ). 

Many approaches have been proposed to tackle the TKG reasoning problem, however, these two reasoning tasks are tackled in totally different and incompatible manners. On the one hand, interpolation methods de-emphasize the temporal correlations among fact sequences while training, thus it’s difficult to cope with the challenges of invisible timestamps and invisible entities in extrapolation due to their poor inductive reasoning ability (Sun et al., 2021). On the other hand, most stateof-the-art (SOTA) extrapolation solutions require a strict chronological order of data during training. As a result, they can only predict unknown future facts, but they could hardly infer missing historical facts which are crucial for completing the 

overall knowledge landscape and providing more clues for predicting accurate future events. These limit the practicability of TKG applications as almost all of the existing TKG reasoning methods are designed specifically to address either one setting. Experiments with a novel pipeline setting intuitively reveal that even with the SOTA methods from both settings, the composed methods show a frustrating decrease in reasoning performance. More in-depth analysis can be found in Section 5.3 and Appendix F. Therefore, the motivation of this work is to propose a unified method that can accommodate two types of reasoning settings, enabling temporal knowledge graph reasoning to be conducted simultaneously for both the interpolation and the extrapolation. 

To this end, we take inspiration from recent neural (Xu et al., 2021; Li et al., 2021) and symbolic (Sun et al., 2021; Liu et al., 2022) TKG reasoning approaches. Neural network approaches can perform effective reasoning but lack interpretation as they cannot provide explicit rules to explain the reasoning results, while symbolic reasoning approaches use logical symbols and rules to perform reasoning tasks but are not suitable for handling ambiguous and noisy data due to their strict matching and discrete logic operations used during rule searching (Zhang et al., 2021). In this paper, we propose a **T** emporal **PA** th based **R** easoning (TPAR) model with a neural-symbolic fashion applicable to both the interpolation and the extrapolation TKG Reasoning. Specifically, we utilize the BellmanFord Shortest Path Algorithm (Ford, 1956; Bellman, 1958; Baras and Theodorakopoulos, 2010) and introduce a recursive encoding method to score the destination entities of various temporal paths, and then our TPAR performs symbolic reasoning with the help of the obtained scores. It is noticeable that the neural-driven symbolic reasoning fashion we adopted is more robust to the uncertainty data compared to traditional pure symbolic reasoning methods, and comprehensible temporal paths with fine interpretability as well. We summarize our main contributions as follows: 

1. We propose an original unified Temporal PAthbased Reasoning (TPAR) model for both the interpolation and extrapolation reasoning settings. To the best of our knowledge, this is the first work to achieve the best of both worlds. 

2. We develop a novel neural-driven symbolic reasoning fashion on various temporal paths to 

enhance both the robustness and interpretability of temporal knowledge reasoning. 

3. Comprehensive experiments show that TPAR outperforms SOTA methods on the link prediction task for both the interpolation and the extrapolation settings with decent interpretability and robustness. An intriguing pipeline experiment is meticulously designed to demonstrate the strengths of TPAR in addressing the unified prediction through both settings. 

## **2 Related Work** 

## **2.1 Static KG Reasoning** 

Static KG reasoning methods can be summarized into three classes: the translation models (Bordes et al., 2013; Wang et al., 2014; Lin et al., 2015), the semantic matching models (Yang et al., 2015; Trouillon et al., 2016), and the embedding-based models (Dettmers et al., 2018; Shang et al., 2019). Recently, R-GCN (Schlichtkrull et al., 2018) and CompGCN (Vashishth et al., 2020) extended GCN to relation-aware GCN for KGs. 

## **2.2 TKG Reasoning** 

**Interpolation TKG Reasoning** Many interpolation TKG reasoning methods (Leblay and Chekol, 2018; Dasgupta et al., 2018; García-Durán et al., 2018; Xu et al., 2019; Lacroix et al., 2020; Jain et al., 2020; Xu et al., 2021) are extended from static KGs to TKGs. Besides, TeRo (Xu et al., 2020) represents each relation as dual complex embeddings and can handle the time intervals between relations. RotateQVS (Chen et al., 2022) represents temporal evolutions as rotations in quaternion vector space and can model various complex relational patterns. T-GAP (Jung et al., 2021) performs pathbased inference by propagating attention through the graph. 

**Extrapolation TKG Reasoning** To predict future facts, many extrapolation TKG reasoning methods (Jin et al., 2020; Li et al., 2021; Han et al., 2021b; Zhu et al., 2021a; Sun et al., 2021; Li et al., 2022b,a) have been studied. For better interpretability, TLogic (Liu et al., 2022) adopts temporary logical rules extracted via temporary random walks, while xERTE (Han et al., 2021a) utilizes an explainable subgraph and attention mechanism for predictions. Besides, RPC (Liang et al., 2023) employs relational graph convolutional networks and 

gated recurrent units to mine relational correlations and periodic patterns from temporal facts. 

## **2.3 Neural-Symbolic Reasoning** 

Neural-symbolic reasoning aims to combine the strengths of both approaches, namely the ability of symbolic reasoning to perform logical inference and the ability of neural networks to perform learning from data (Zhang et al., 2021). Generally, it can be divided into three categories: (1) Symbolicdriven neural fashion (Guo et al., 2016, 2018; Wang et al., 2019), which targets neural reasoning but leverages the logic rules to improve the embeddings. (2) Symbolic-driven probabilistic fashion (Richardson and Domingos, 2006; Qu and Tang, 2019; Raedt et al., 2007), which replaces the neural reasoning with a probabilistic framework, i.e., builds a probabilistic model to infer the answers. (3) Neural-driven symbolic fashion (Lao and Cohen, 2010; Lao et al., 2012; Neelakantan et al., 2015), which aims to infer rules by symbolic reasoning, and incorporates neural networks to deal with the uncertainty and ambiguity of data. For this work, we adopt the neural-driven symbolic fashion for our TKG reasoning. 

## **3 Preliminaries** 

## **3.1 Problem Definition of TKG Reasoning** 

A Temporal knowledge graph is a collection of millions of temporal facts, expressed as _G ⊂E × R × E × T_ . Here we use _E_ to denote the set of entities, _R_ to denote the set of relations, and _T_ to denote the set of time stamps. Each temporal fact in _G_ is formed as a quadruple ( _es, r, eo, t_ ), where a relation _r ∈R_ holds between a subject entity _es ∈ E_ and an object entity _eo ∈E_ at the time _t_ . And for each quadruple ( _es, r, eo, t_ )), an inverse quadruple ( _eo, r[−]_[1] _, es, t_ )) is also added to the dataset. 

Let _O_ = _{_ ( _es, r, eo, t_ ) _|t ∈_ [ _t_ 0 _, T_ ] _}_ represent the set of known facts that we can observe, and let [ _t_ 0 _, T_ ] denote the time interval we can access. For a query _**q**_ = ( _eq, rq,_ ? _, tq_ ), TKG Reasoning aims to infer the missing object entity given the other three elements. As we have two distinctive TKG Reasoning settings: the interpolation (inferring a query where _t_ 0 _≤ tq ≤ T_ ) and the extrapolation (predicting facts for _tq > T_ ), we use a unified concept of "temporal background" to represent all known temporal facts of existing TKG. 

## **3.2 Bellman-Fold Based Recursive Encoding** 

The Bellman-Ford Algorithm (Ford, 1956; Bellman, 1958; Baras and Theodorakopoulos, 2010) provides a recursive approach to search for the shortest path in a graph.[1] Since TKGs are far from complete, we do not directly copy it for TKG Reasoning, but utilize a recursive encoding method for representations. 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/acl_2024_A_Unified_Temporal_Knowledge_Graph_Reasoning_Model_Towards_Interpolation_and_Extrapolation_images/acl_2024_A_Unified_Temporal_Knowledge_Graph_Reasoning_Model_Towards_Interpolation_and_Extrapolation.pdf-0003-09.png)


Figure 1: An illustration of Bellman-Ford-based recursive encoding. 

As illustrated in Fig.1, the basic idea of our Bellman-Ford-based recursive encoding is: once the representations of the _ℓ −_ 1 iterations (the node _x_ ) starting from node _u_ are ready, we can obtain the representations of the _ℓ_ iterations (the node _v_ ) by combining them with the edge ( _x, v_ ): 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/acl_2024_A_Unified_Temporal_Knowledge_Graph_Reasoning_Model_Towards_Interpolation_and_Extrapolation_images/acl_2024_A_Unified_Temporal_Knowledge_Graph_Reasoning_Model_Towards_Interpolation_and_Extrapolation.pdf-0003-12.png)


where 1 ( _u_ = _v_ ) is the indicator function that outputs 1 if _u_ = _v_ and 0 otherwise, _⊗_ is the multiplication operator, _⊕_ is the summation operator, _E_ ( _v_ ) is the set of all edges connected to the node _v_ , and _w_ ( _x, v_ ) is the weight of the edge ( _x, v_ ). 

## **4 Methodology** 

In this section, we introduce a novel temporal pathbased reasoning model with a neural-driven symbolic fashion for both the interpolation and the extrapolation TKG Reasoning settings. 

## **4.1 Temporal Path in TKGs** 

**Definition 1.** _(Temporal link). In a TKG, each entity is viewed as a node in the graph, and a temporal link is viewed as an edge connecting two nodes which represent the subject es and the object entity eo in a certain quadruple_ ( _es, r, eo, t_ ) _. Due to the heterogeneity and time constraints, each link is bonded with a relation r and a particular time t. Thus, we can denote a temporal link as r[t]_[(] _[e][s][, e][o]_[)] _[.]_ 

1See Appendix A for more details. 

**Definition 2.** _(Temporal path). A temporal path P is a combination of several temporal links that are sequentially connected from subject entity to object entity. We denote a temporal path P as_ 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/acl_2024_A_Unified_Temporal_Knowledge_Graph_Reasoning_Model_Towards_Interpolation_and_Extrapolation_images/acl_2024_A_Unified_Temporal_Knowledge_Graph_Reasoning_Model_Towards_Interpolation_and_Extrapolation.pdf-0004-01.png)


_or_[�] _[ℓ] i_ =1 _rtii_[(] _[e][i][, e][i]_[+1][)] _[ for short, where][ e]_[1] _[,][ e]_[2] _[,][ ···][e][l]_[+1] _are sequentially connected to form a chain, and ℓ is the path length. We take e_ 1 _and el_ +1 _as the origin and the destination of the path respectively, and use P_ ( _i_ ) _to denote the temporal link r[t][i] i_[(] _[e][i][, e][i]_[+1][)] _[ of] the i-th step in P._[2] 

For the extrapolation setting, a non-increasing temporal path fashion has been applied in recent works (Han et al., 2021a; Liu et al., 2022), and requires _t_ 1 _≥ t_ 2 _≥· · ·tℓ ≥ tℓ_ +1. In this paper, we are not going to follow the non-increasing setup. For a query _**q**_ = ( _eq, rq,_ ? _, tq_ ), we require each _ti_ in a temporal path _P_ to satisfy _ti < tq_ such that we only use information from the past to predict the future, but have no restrictions to the chronological orders among _t_ 1, _t_ 2, _· · · tl_ , i.e. we can have _ti < ti_ +1, _ti_ = _ti_ +1 or _ti > ti_ +1. This relaxation provides more temporal paths deriving from the temporal background, enabling us to utilize more information for reasoning. Further discussions for this can be found in Appendix H. 

For the interpolation setting, we mainly focus on how to find the missing object entity from the given subject entity _eq_ and indeed we ignore any time constraints. For a temporal link _r[t][i] i_[(] _[e][i][, e][i]_[+1][)] in the temporal background, whether it is before or after _tq_ , we can include it in a temporal path as long as it has been known to us. 

## **4.2 Relative Time Encoding on Temporal Links** 

We adopt relative time encoding and focus on the impact of a certain temporal link in the temporal path on a query. For each temporal link _ti_[we][first][obtain][the][relative][time] _ri_[(] _[e][i][, e][i]_[+1][)][,] ∆ _ti_ = _ti − tq_ . Notice that a temporal link itself is query-independent, but the obtained relative time ∆ _ti_ is query-dependent. For the extrapolation setting, there is always an inequality ∆ _ti <_ 0 since _ti < tq_ . But for the interpolation setting, ∆ _ti <_ 0, 

> 2An illustration of temporal paths and temporal links is shared in Appendix B. 

∆ _ti_ = 0 and ∆ _ti >_ 0 are all possible since there is no restriction on the order of _ti_ and _tq_ . 

As mentioned by Zhu et al. (2021a), many facts have occurred repeatedly throughout history, such as economic crises that occur every seven to ten years (Korotayev and Tsirel, 2010). We pay close attention to the periodic property as well as the non-periodic property of time. We then extend the idea of periodic and non-periodic time vectors (Li et al., 2022a) to a relative time encoding fashion: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/acl_2024_A_Unified_Temporal_Knowledge_Graph_Reasoning_Model_Towards_Interpolation_and_Extrapolation_images/acl_2024_A_Unified_Temporal_Knowledge_Graph_Reasoning_Model_Towards_Interpolation_and_Extrapolation.pdf-0004-10.png)


where _ωp_ , _φp_ , _ωnp_ , _φnp_ are learnable parameters, sin() denotes the sine function, and the obtained _**h**[p] ti_[and] _**[ h]** t[np] i_[are the periodic and the non-periodic] relative time vectors, respectively. For the periodic vector _**h**[p] ti_[, the frequency] _[ ω][p]_[ and the phase shift] _[ φ][p]_ reflect the periodic property. For the non-periodic vector _**h**[np] ti_[,] _[ ω][np]_[ acts as a velocity vector (][Han et al.][,] 2020) indicating the temporal evolving dynamics. Finally, we combine both periodic and nonperiodic properties to design our relative time embedding as: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/acl_2024_A_Unified_Temporal_Knowledge_Graph_Reasoning_Model_Towards_Interpolation_and_Extrapolation_images/acl_2024_A_Unified_Temporal_Knowledge_Graph_Reasoning_Model_Towards_Interpolation_and_Extrapolation.pdf-0004-12.png)


where we use the same embedding dimension _d_ for _**h**[p] ti_[,] _**[ h]** t[np] i_[and] _**[ h]**[t] i_[.] 

## **4.3 Recursive Encoding on Temporal Paths** 

In Section 3.2, we have mentioned the idea of Bellman-Ford-based recursive encoding. Now, we start from the query and introduce our recursive encoding on temporal paths for TKG reasoning. 

**Basic Design** Our recursive encoding on temporal paths aims to encode the destination entities of various temporal paths. And the basic idea is: Once the representations of the destination entities of all the ( _ℓ −_ 1)-length paths are ready, we can combine them with the link _P_ ( _l_ ) to encode the destination entities of the _ℓ_ -length paths. 

For a query _**q**_ = ( _eq, rq,_ ? _, tq_ ), our recursive encoding is query-specific, and we start with the query entity _eq_ . Initially, we can only reach the original entity _eq_ , so we initialize _**h**_[0] _eq_[=] **[ 0]**[ and the] entity set _Ee_[0] _q_[=] _[{][e][q][}]_[.][In the] _[ ℓ]_[-th step, we collect] all the _ℓ_ -length temporal paths from temporal background based on _eq_ , and update _Ee[ℓ] q_[with the des-] tination entities. We utilize _D_ ( _Peq_ ) to denote the destination entity set of the temporal paths _Peq_ , and _Peq,_ ( _ℓ_ ) to denote the set of links of the _ℓ_ -th step in 

_Peq_ . Thus, we actually have _Ee[ℓ] q_[=] _[ E] e[ℓ] q[−]_[1] _∪D_ ( _Peq_ ) for _ℓ>_ 1. 

Then, the message is passed along temporal paths _Peq_ from _es_ to _eo_ , where _r[t]_[(] _[e][s][, e][o]_[)] _[ ∈P][e][q][,]_[(] _[l]_[)][.] And we can get the representations for all _eo ∈ D_ ( _Peq_ ) for _ℓ_ = 1 _,_ 2 _, · · · L_ by: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/acl_2024_A_Unified_Temporal_Knowledge_Graph_Reasoning_Model_Towards_Interpolation_and_Extrapolation_images/acl_2024_A_Unified_Temporal_Knowledge_Graph_Reasoning_Model_Towards_Interpolation_and_Extrapolation.pdf-0005-02.png)


where _δ_ denotes an activation function, _**W**[ℓ] ∈_ R _[d][×][d]_ is a learnable parameter, _ϕ_ () denotes a function to compute and process the messages, _**h**[ℓ] r_[is the] embedding for relation _r_ in the _ℓ_ -th step, and _**h** t_ is a result of the relative time encoding that we have introduced in Section 4.2. The proposed recursive encoding algorithm is shown in Algorithm 1.[3] 

**Algorithm 1** The Proposed Recursive Encoding Algorithm 

**Input:** A temporal knowledge graph _G_ , a query _**q**_ = ( _eq, rq,_ ? _, tq_ ), the maximum length _L_ **Output:** The query-specific entity representations 1: initialize _**h**_[0] _eq_[=] **[ 0]**[ and the entity set] _[ E] e_[0] _q_[=] _[ {][e] q[}]_[;] 2: **for** _ℓ_ = 1 _,_ 2 _, · · · L_ **do** 3: collect all the _ℓ_ -length temporal paths and update _Ee[ℓ] q_[with the destination entity set] _[ D]_[(] _[P] eq_[)][;] 4: message passing along temporal paths _Peq_ from _es ∈D_ ( _Peq ,_ ( _ℓ−_ 1)) to _eo ∈D_ ( _Peq ,_ ( _ℓ_ )): _**h**[ℓ] eo_[=] _[ δ]_ � _**W**[ℓ]_[�] _Peq,_ ( _l_ ) _[ϕ]_ � _**h**[ℓ] e[−] s_[1] _[,]_ _**[ h]** r[ℓ][,]_ _**[ h]** t_ �[�] , 5: **end for** 6: assign _**h**[L] ea_[=] **[ 0]**[ for all] _[ e] a[∈E][/] e[L] q_[;] 7: **return** _**h**[L] ea_[for all] _[ e] a[∈E]_[.] 

**Message Passing** We then give a detailed description of the message passing along temporal paths from _es_ to _eo_ . 

Given a quadruple ( _es, r, eo, t_ ) in temporal ground, we can transform it to a temporal link _t_[We take the message as a combination] _r_[(] _[e][s][, e][o]_[)][.] of the query-specific representation of _es_ learned in the ( _ℓ −_ 1)-th recursive encoding step, the representation of _r_ , and the representation of _t_ : 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/acl_2024_A_Unified_Temporal_Knowledge_Graph_Reasoning_Model_Towards_Interpolation_and_Extrapolation_images/acl_2024_A_Unified_Temporal_Knowledge_Graph_Reasoning_Model_Towards_Interpolation_and_Extrapolation.pdf-0005-08.png)


Note that we use the same embedding dimension _d_ for _**h**[ℓ] e[−] s_[1][,] _**[ h]** r[ℓ]_[,] _**[ h]**[ℓ] rq_[and] _**[ h]**[t]_[.] 

Then, we utilize an attention-based mechanism to weigh the message. Inspired by graph attention networks (Velickovic et al., 2018), we first take a concatenation operation on _**h**[ℓ] e[−] s_[1][,] _**[ h]** r[ℓ]_[,] _**[ h]**[ℓ] rq_[and] _**[ h]**[t]_[to] get 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/acl_2024_A_Unified_Temporal_Knowledge_Graph_Reasoning_Model_Towards_Interpolation_and_Extrapolation_images/acl_2024_A_Unified_Temporal_Knowledge_Graph_Reasoning_Model_Towards_Interpolation_and_Extrapolation.pdf-0005-11.png)


where _**U**_ ( _es,r,eo,t_ ) _|_ _**q** ∈_ R[4] _[d]_ is the concatenated result. Then, the attention weight for the message can be defined as 

_α_ ( _[ℓ] es,r,eo,t_ ) _|_ _**q**_[=] _[ σ]_ �( _**w** α[ℓ]_[)] _[T][ ReLU]_[(] _**[W]** α[ ℓ][·]_ _**[ U]**_ ( _es,r,eo,t_ ) _|_ _**q**_[)] � _,_ (7) where _**w** α[ℓ][∈]_[R] _[d][a]_[,] _**[W]**[ ℓ] α[∈]_[R] _[d][a][×]_[4] _[d]_[are][learnable] parameters for attention ( _da_ is the dimension for attention). Just like (Zhang and Yao, 2022), we use the Sigmoid function _σ_ rather than the Softmax function to ensure multiple temporal links can be selected in the same recursive step. 

**Aggregation** With messages and the weights obtained, we take an aggregation operation on messages along different temporal links in various temporal paths, by means of a weighted sum of messages. For the _ℓ_ -th recursive step, our encoded representation is specified as 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/acl_2024_A_Unified_Temporal_Knowledge_Graph_Reasoning_Model_Towards_Interpolation_and_Extrapolation_images/acl_2024_A_Unified_Temporal_Knowledge_Graph_Reasoning_Model_Towards_Interpolation_and_Extrapolation.pdf-0005-15.png)


So far, we have provided a detailed version of our recursive encoding function (also see a general version in Equation 4) for each recursive step _ℓ_ = 1 _,_ 2 _, · · · L_ . After _L_ recursive steps, the representation of each entity inside various _L_ -length temporal paths (starting from _eq_ ) can be learned step by step. We assign _**h**[L] ea_[(] _[e][q][, r][q][, t][q]_[)][=] **[0]**[for] all _ea ∈E/ e[L] q_[.][For][each][entity] _[e][a][∈E]_[,][we][finally] get its representation _**h**[L] ea_[after] _[ L]_[-length recursive] encoding process. 

## **4.4 Reasoning on Temporal Paths** 

Then, we can leverage the learned representations to measure the quality of temporal paths to reason. 

**Neural based Scoring** To accomplish neuralsymbolic reasoning, we take a neural-based scoring first and use a simple scoring function to compute the likelihood for each candidate answer entity _ea_ : 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/acl_2024_A_Unified_Temporal_Knowledge_Graph_Reasoning_Model_Towards_Interpolation_and_Extrapolation_images/acl_2024_A_Unified_Temporal_Knowledge_Graph_Reasoning_Model_Towards_Interpolation_and_Extrapolation.pdf-0005-20.png)


where _**w** ∈_ R _[d]_ is a parameter for scoring. 

And for training loss, we use a multi-class logloss function to train the neural networks for our TKG Reasoning, which has been proven effective (Lacroix et al., 2018; Zhang and Yao, 2022): 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/acl_2024_A_Unified_Temporal_Knowledge_Graph_Reasoning_Model_Towards_Interpolation_and_Extrapolation_images/acl_2024_A_Unified_Temporal_Knowledge_Graph_Reasoning_Model_Towards_Interpolation_and_Extrapolation.pdf-0005-23.png)


> 3The time complexity is _O_ ( _|E|d_ + _|V|d_ 2), where _|E|_ and where _T_ train is set of training queries, and[�] _∀e∈E |V|_ denote the number of entities and relations respectively. is for all the quadruples w.r.t. the same query _**q**_ . 

**Symbolic Reasoning** As for our symbolic reasoning, we target both the suitable temporal path and its destination entity. Using _mi_ to denote the _i_ -th intermediary entity, our reasoning can be expressed as 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/acl_2024_A_Unified_Temporal_Knowledge_Graph_Reasoning_Model_Towards_Interpolation_and_Extrapolation_images/acl_2024_A_Unified_Temporal_Knowledge_Graph_Reasoning_Model_Towards_Interpolation_and_Extrapolation.pdf-0006-01.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/acl_2024_A_Unified_Temporal_Knowledge_Graph_Reasoning_Model_Towards_Interpolation_and_Extrapolation_images/acl_2024_A_Unified_Temporal_Knowledge_Graph_Reasoning_Model_Towards_Interpolation_and_Extrapolation.pdf-0006-02.png)


where the left part of _→_ denotes a temporal path composed of _L_ sequentially connected temporal links, and the right part is the link we aim to infer. 

We use _**A**[ℓ] eq[∈{]_[0] _[,]_[ 1] _[}][N][ℓ][×][N][ℓ]_[to denote the adja-] cent matrix of the _ℓ_ -th step in the set of _L_ -length paths _Peq_ starting from _eq_ , i.e. _Peq,_ ( _ℓ_ ), where _Nℓ_ is the number of entities involved for _ℓ_ = 1 _,_ 2 _, · · · L_ . If _[t]_[1] _r_ 1[(] _[e][q][, m]_[1][)] _[ ∧] r[t]_[2] 2[(] _[m]_[1] _[, m]_[2][)] _[ ∧· · ·] r[t][L] L_[(] _[m][L][−]_[1] _[, e][a]_[)] _[ →] tq rq_[(] _[e][q][, e][a]_[)][ holds, ideally there exists a suitable path] _P[⋆]_ : 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/acl_2024_A_Unified_Temporal_Knowledge_Graph_Reasoning_Model_Towards_Interpolation_and_Extrapolation_images/acl_2024_A_Unified_Temporal_Knowledge_Graph_Reasoning_Model_Towards_Interpolation_and_Extrapolation.pdf-0006-05.png)


where _β[ℓ] ∈{_ 0 _,_ 1 _}[N][ℓ][×][N][ℓ]_ is the choice of links for step _ℓ_ . 

From Equation 6 and 7, we can know the learned _α[ℓ]_[and][ab-] ( _es,r,eo,t_ ) _|_ _**q**_[=] _[ MLP][ ℓ]_[(] _[e][s][, r, e][q][, r][q][,]_[ ∆] _[t]_[)][,] breviate it as _α[ℓ]_ . Our recursive encoding provides 

arg max **Θ** _Score_ ( _P_ ) = _α_[1] _**A**_[1] _eq[⊗][α]_[2] _**[A]** e_[2] _q[· · · ⊗][α][L]_ _**[A]** e[L] q[,]_ (13) 

where _Score_ ( _P_ ) is the score of the path _P_ obtained by Equation 9. Based on universal approximation theorem (Hornik, 1991), there exists a set of parameters Θ that can learn _α[ℓ] ≃ β[ℓ]_ , to make _P ≃P[⋆]_ . Note that multiple temporal paths may lead to the same destination. In such cases, the score can be regarded as a comprehensive measure of the various reasoning paths. Case-based analysis on this can be found in Section 5.4. 

To summarize, our temporal path-based reasoning first collects temporal paths from the temporal background, then scores the destination entities of various temporal paths by attention-based recursive encoding, and lastly takes the entity with the highest score as the reasoning answer. 

## **5 Experiments** 

## **5.1 Evaluation Protocol** 

Link prediction task that aims to infer incomplete time-wise fact with a missing entity (( _s, r,_ ? _, t_ ) or (? _, r, o, t_ )) is adopted to evaluate the proposed 

model for both interpolation and extrapolation reasoning over TKGs. During inference, we follow the same procedure of Xu et al. (2020) to generate candidates. For a test sample ( _s, r, o, t_ ), we first generate candidate quadruples set _C_ = _{_ ( _s, r, o, t_ ) : _o ∈E} ∪{_ ( _s, r, o, t_ ) : _s ∈E}_ by replacing _s_ or _o_ with all possible entities, and then rank all the quadruples by their scores (Equation 9). We use the time-wise filtered setting (Xu et al., 2019; Goel et al., 2020) to report the experimental results. 

We use ICEWS14, ICEWS05-15 (García-Durán et al., 2018), YAGO11k (Dasgupta et al., 2018) and WIKIDATA12k (Dasgupta et al., 2018) datasets for interpolation reasoning evaluation, and use ICEWS14 (García-Durán et al., 2018), ICEWS18 (Jin et al., 2020), YAGO (Mahdisoltani et al., 2015) and WIKI (Leblay and Chekol, 2018) for extrapolation.[4] The performance is reported on the standard evaluation metrics: the proportion of correct triples ranked in the top 1, 3 and 10 (Hits@1, Hits@3, and Hits@10), and Mean Reciprocal Rank (MRR). All of them are the higher the better. For all experiments, we report averaged results across 5 runs, and we omit the variance as it is generally low. 

## **5.2 Main Results** 

## **5.2.1 Interpolation Reasoning** 

Due to space limitations, we share the experimental details in Appendix D. Table 1 shows the results of the interpolation TKG Reasoning on link prediction over four experimented datasets where the proposed TPAR continuously outperforms all baselines across all metrics. We specifically compare our proposed TPAR with another GNN-based method, T-GAP (Jung et al., 2021), which samples a subgraph from the whole TKG for each node. In contrast, our TPAR recursively traverses the entire graph based on previously visited nodes to collect temporal paths. Our experimental results show that TPAR outperforms T-GAP across all evaluation metrics on the ICEWS14 and ICEWS05-15 datasets. This can be attributed to TPAR’s ability to capture the temporal information of the entire graph and generate longer paths, which leads to more accurate and comprehensive TKG reasoning. 

## **5.2.2 Extrapolation Reasoning** 

Experimental details for Extrapolation reasoning are shown in Appendix E. Table 2 presents the results of the extrapolation TKG reasoning on 

4 More details for datasets can be found in Appdendix C. 

|Interpolation|ICEWS14|YAGO11k|ICEWS05-15|WIKIDATA12k|
|---|---|---|---|---|
||Hits@1<br>Hits@3<br>Hits@10<br>MRR|Hits@1<br>Hits@3<br>Hits@10<br>MRR|Hits@1<br>Hits@3<br>Hits@10<br>MRR|Hits@1<br>Hits@3<br>Hits@10<br>MRR|
|TTransE<br>HyTE<br>TA-DistMult<br>ATiSE<br>TeRo<br>TComplEx<br>T-GAP<br>TELM<br>RotateQVS<br>TGoemE++|7.4<br>-<br>60.1<br>25.5<br>10.8<br>41.6<br>65.5<br>29.7<br>36.3<br>-<br>68.6<br>47.7<br>43.6<br>62.9<br>75.0<br>55.0<br>46.8<br>62.1<br>73.2<br>56.2<br>53<br>66<br>77<br>61<br>50.9<br>67.7<br>79.0<br>61.0<br>54.5<br>67.3<br>77.4<br>62.5<br>50.7<br>64.2<br>75.4<br>59.1<br>54.6<br>68.0<br>78.0<br>62.9|2.0<br>15.0<br>25.1<br>10.8<br>1.5<br>14.3<br>27.2<br>10.5<br>10.3<br>17.1<br>29.2<br>16.1<br>11.0<br>17.1<br>28.8<br>17.0<br>12.1<br>19.7<br>31.9<br>18.7<br>12.7<br>18.3<br>30.7<br>18.5<br>-<br>-<br>-<br>-<br>12.9<br>19.4<br>32.1<br>19.1<br>12.4<br>19.9<br>32.3<br>18.9<br>13.0<br>19.6<br>32.6<br>19.5|8.4<br>-<br>61.6<br>27.1<br>11.6<br>44.5<br>68.1<br>31.6<br>34.6<br>-<br>72.8<br>47.4<br>37.8<br>60.6<br>79.4<br>51.9<br>46.9<br>66.8<br>79.5<br>58.6<br>59<br>71<br>80<br>66<br>56.8<br>74.3<br>84.5<br>67.0<br>59.9<br>72.8<br>82.3<br>67.8<br>52.9<br>70.9<br>81.3<br>63.3<br>60.5<br>73.6<br>83.3<br>68.6|9.6<br>18.4<br>32.9<br>17.2<br>9.8<br>19.7<br>33.3<br>18.0<br>12.2<br>23.2<br>44.7<br>21.8<br>17.5<br>31.7<br>48.1<br>28.0<br>19.8<br>32.9<br>50.7<br>29.9<br>23.3<br>35.7<br>53.9<br>33.1<br>-<br>-<br>-<br>-<br>23.1<br>36.0<br>54.2<br>33.2<br>20.1<br>32.7<br>51.5<br>30.2<br>23.2<br>36.2<br>54.6<br>33.3|
|TPAR (Ours)|**57.03**<br>**69.74**<br>**80.41**<br>**65.07**|**17.35**<br>**25.14**<br>**37.42**<br>**24.12**|**62.17**<br>**75.29**<br>**85.86**<br>**69.33**|**25.05**<br>**38.68**<br>**54.79**<br>**34.89**|



Table 1: **Interpolation** TKG Reasoning results (in percentage) on link prediction over four experimented datasets. 

|Extrapolation|ICEWS14|YAGO|ICEWS18|WIKI|
|---|---|---|---|---|
||Hits@1<br>Hits@3<br>Hits@10<br>MRR|Hits@1<br>Hits@3<br>Hits@10<br>MRR|Hits@1<br>Hits@3<br>Hits@10<br>MRR|Hits@1<br>Hits@3<br>Hits@10<br>MRR|
|RE-NET<br>CyGNeT<br>TANGO<br>xERTE<br>RE-GCN<br>TITer<br>CEN<br>TLogic<br>TIRGN<br>RPC|30.11<br>44.02<br>58.21<br>39.86<br>27.43<br>42.63<br>57.90<br>37.65<br>-<br>-<br>-<br>-<br>32.70<br>45.67<br>57.30<br>40.79<br>31.63<br>47.20<br>61.65<br>42.00<br>32.76<br>46.46<br>58.44<br>41.73<br>32.08<br>47.46<br>61.31<br>42.20<br>33.56<br>48.27<br>61.23<br>43.04<br>33.83<br>48.95<br>63.84<br>44.04<br>34.87<br>49.80<br>65.08<br>44.55|58.59<br>71.48<br>86.84<br>66.93<br>58.97<br>76.80<br>86.98<br>68.98<br>60.04<br>65.19<br>68.79<br>63.34<br>80.09<br>88.02<br>89.78<br>84.19<br>78.83<br>84.27<br>88.58<br>82.30<br>80.09<br>89.96<br>90.27<br>87.47<br>-<br>-<br>-<br>-<br>-<br>-<br>-<br>-<br>84.34<br>91.37<br>92.92<br>87.95<br>85.10<br>92.57<br>94.04<br>88.87|19.73<br>32.55<br>48.46<br>29.78<br>17.21<br>30.97<br>46.85<br>27.12<br>18.68<br>30.86<br>44.94<br>27.56<br>21.03<br>33.51<br>46.48<br>29.31<br>22.39<br>36.79<br>52.68<br>32.62<br>22.05<br>33.46<br>44.83<br>29.98<br>21.70<br>35.44<br>50.59<br>31.50<br>20.54<br>33.95<br>48.53<br>29.82<br>23.19<br>37.99<br>54.22<br>33.66<br>24.34<br>38.74<br>55.89<br>34.91|50.01<br>61.23<br>73.57<br>58.32<br>47.89<br>66.44<br>78.70<br>58.78<br>51.52<br>53.84<br>55.46<br>53.04<br>69.05<br>78.03<br>79.73<br>73.60<br>74.50<br>81.59<br>84.70<br>78.53<br>71.70<br>75.41<br>76.96<br>73.91<br>75.05<br>81.90<br>84.90<br>78.93<br>-<br>-<br>-<br>-<br>77.77<br>85.12<br>87.08<br>81.65<br>76.28<br>85.43<br>88.71<br>81.18|
|TPAR (Ours)|**36.88**<br>**52.28**<br>**65.89**<br>**46.89**|**89.67**<br>**92.93**<br>**94.60**<br>**91.53**|**26.58**<br>**39.27**<br>**56.94**<br>**35.76**|**79.85**<br>**87.04**<br>**88.96**<br>**83.20**|



Table 2: **Extrapolation** TKG Reasoning results (in percentage) on link prediction over four experimented datasets. 

link prediction across four different datasets. Notably, we compare TPAR with two SOTA methods, namely, the neural network-based RPC (Liang et al., 2023) and the symbolic-based TLogic (Liu et al., 2022). Our results show that TPAR outperforms all baselines, underscoring the benefits of combining neural and symbolic-based approaches for TKG reasoning. 

|Interpolation|Extrapolation|Setting|Setting|Setting|
|---|---|---|---|---|
|||**A**(Ratio: 60%)|**B**(Ratio: 70%)|**C**(Ratio: 80%)|
|TELM|TLogic|40.88|41.55|42.09|
|�|TLogic|41.85|41.88|42.18|
|RotateQVS|TIRGN|40.86|42.35|42.78|
|�|TIRGN|42.53|43.68|43.69|
|TPAR|TPAR|44.56|46.07|46.66|
|�|TPAR|43.86|45.49|36.36|



Table 3: MRR performance of the pipeline setting on ICEWS 14. 

## **5.3 Analysis on Pipeline Setting** 

To test the hypothesis that completing missing knowledge about the past can enhance the accuracy of predicting future knowledge, we design a novel experimental pipeline. Building upon the facts within the known time range, we sample a ratio of facts to train interpolation, while the other unsampled facts have either the subject entity or the object entity masked, forming incomplete quadruples. Specifically, we have three settings: set the ratio to 60% for Setting A, 70% for Setting B, and 80% for Setting C.[5] The lower the ratio, the more incomplete facts need to be interpolated first. 

The pipeline involves first completing these incomplete quadruples through interpolation, and subsequently predicting future events. **Baselines** : Since there are currently no existing methods capable of both interpolation and extrapolation, we assemble SOTA interpolation and extrapolation meth- 

5See Appendix F for full pipeline setting details. 

ods[6] , and get two combinations: a) TELM and TLogic, b) RotateQVS and TIRGN. 

The experimental results in Table 3[7] show two key observations: a) For both of the SOTA combination baselines, employing the pipeline (interpolation followed by extrapolation) leads to a reduction in extrapolation performance compared to straightforward extrapolation. And this reduction in the effect is gradually alleviated as the ratio increases from 60% to 80%. We believe that this could be attributed to the fact that the newly acquired knowledge from interpolation may contain errors, which have the potential to propagate and amplify in subsequent extrapolation. As the amount of newly acquired knowledge decreases, the likelihood of introducing errors into the system also diminishes. 

> 6For fairness, we do not use TGoemE++ and RPC as they have not made their code publicly available. 7See Table 5 for the more detailed results. 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/acl_2024_A_Unified_Temporal_Knowledge_Graph_Reasoning_Model_Towards_Interpolation_and_Extrapolation_images/acl_2024_A_Unified_Temporal_Knowledge_Graph_Reasoning_Model_Towards_Interpolation_and_Extrapolation.pdf-0008-00.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/acl_2024_A_Unified_Temporal_Knowledge_Graph_Reasoning_Model_Towards_Interpolation_and_Extrapolation_images/acl_2024_A_Unified_Temporal_Knowledge_Graph_Reasoning_Model_Towards_Interpolation_and_Extrapolation.pdf-0008-01.png)


**----- Start of picture text -----**<br>
(a) Interpolation  for the maximum length  L  = 2.<br>**----- End of picture text -----**<br>



![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/acl_2024_A_Unified_Temporal_Knowledge_Graph_Reasoning_Model_Towards_Interpolation_and_Extrapolation_images/acl_2024_A_Unified_Temporal_Knowledge_Graph_Reasoning_Model_Towards_Interpolation_and_Extrapolation.pdf-0008-02.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/acl_2024_A_Unified_Temporal_Knowledge_Graph_Reasoning_Model_Towards_Interpolation_and_Extrapolation_images/acl_2024_A_Unified_Temporal_Knowledge_Graph_Reasoning_Model_Towards_Interpolation_and_Extrapolation.pdf-0008-03.png)


**----- Start of picture text -----**<br>
(b) Extrapolation  for the maximum length  L  = 3.<br>**----- End of picture text -----**<br>


Figure 2: Path interpretations of TKG reasoning on ICEWS14 test set. Dashed lines mean inverse relations. 

b) In all three settings, our TPAR with pipeline outperforms the straightforward extrapolation results, which demonstrates the superiority of our TPAR model to effectively integrate interpolation and extrapolation as a unified task for TKG reasoning. 

## **5.4 Case Study: Path Interpretations** 

Benefiting from the neural-symbolic fashion, we can interpret the reasoning results through temporal paths. Following local interpretation methods (Zeiler and Fergus, 2014; Zhu et al., 2021b), we approximate the local landscape of our TPAR with a linear model over the set of all paths, i.e., 1st-order Taylor polynomial. Thus, the importance of a path can be defined as its weight in the linear model and can be computed by the partial derivative of the prediction w.r.t. the path. Formally, top-k temporal path interpretations for _f_ ( _**q** , ea_ ) are defined as 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/acl_2024_A_Unified_Temporal_Knowledge_Graph_Reasoning_Model_Towards_Interpolation_and_Extrapolation_images/acl_2024_A_Unified_Temporal_Knowledge_Graph_Reasoning_Model_Towards_Interpolation_and_Extrapolation.pdf-0008-08.png)


where _Peqea_ denotes the set of all temporal paths starting from _eq_ to _ea_ . 

Fig. 2 illustrates the path interpretations of TKG reasoning for both two settings on the ICEWS14 test set. While users may have different insights 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/acl_2024_A_Unified_Temporal_Knowledge_Graph_Reasoning_Model_Towards_Interpolation_and_Extrapolation_images/acl_2024_A_Unified_Temporal_Knowledge_Graph_Reasoning_Model_Towards_Interpolation_and_Extrapolation.pdf-0008-11.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/acl_2024_A_Unified_Temporal_Knowledge_Graph_Reasoning_Model_Towards_Interpolation_and_Extrapolation_images/acl_2024_A_Unified_Temporal_Knowledge_Graph_Reasoning_Model_Towards_Interpolation_and_Extrapolation.pdf-0008-12.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/acl_2024_A_Unified_Temporal_Knowledge_Graph_Reasoning_Model_Towards_Interpolation_and_Extrapolation_images/acl_2024_A_Unified_Temporal_Knowledge_Graph_Reasoning_Model_Towards_Interpolation_and_Extrapolation.pdf-0008-13.png)


**----- Start of picture text -----**<br>
(a) Interpolation (b) Extrapolation<br>**----- End of picture text -----**<br>


Figure 3: Performance w.r.t. sparsity ratio on ICEWS14. 

towards the visualization, here are our understandings: 1) In Fig. 2(a), for the interpolation setting, when we need to reason about the fact that happened on _tq_ =2014-02-14, we can use the path formed by the links before or after _tq_ . 2) For the extrapolation of Fig. 2(b), we do not strictly impose a chronological order constraint on the links in the path, ensuring that the paths used in our reasoning can be as rich as possible. 3) Multiple different temporal paths may lead to the same reasoning destination, and the explanation is that a reasoning result can have multiple evidence chains. 

In addition, we conduct analyses on path lengths, and the contents can be found in Appendix G. 

## **5.5 Robustness Analysis** 

Fig.3 illustrates a performance comparison under varying degrees of data sparsity on the ICEWS14 dataset. We compare our TPAR model with RotateQVS (Chen et al., 2022) and TELM (Xu et al., 2021) in the interpolation setting, while TLogic (Liu et al., 2022) and TIRGN (Li et al., 2022a) are taken as extrapolation baselines. Our TPAR demonstrates superior performance compared to the other models across a wide range of data sparsity levels for both interpolation and extrapolation settings. This highlights its exceptional robustness and ability to handle sparse data effectively. 

Furthermore, we conduct an analysis on the chronological order in Appendix H, which also demonstrates robustness to some extent. 

## **6 Conclusion** 

We propose a temporal path-based reasoning (TPAR) model with a neural-symbolic fashion that can be applied to both interpolation and extrapolation TKG Reasoning settings in this paper. Introducing the Bellman-Ford Algorithm and a recursive encoding method to score the destination entities of various temporal paths, TPAR performs robustly on uncertain temporal knowledge with fine inter- 

pretability. Comprehensive experiments demonstrate the superiority of our proposed model for both settings. A well-designed pipeline experiment further verified that TPAR is capable of integrating two settings for a unified TKG reasoning. 

## **7 Limitations** 

We identify that there may be some possible limitations in this study. First, our reasoning results are dependent on the embeddings learned by the proposed (Bellman-Ford-based) recursive encoding algorithm, where the selection of hyperparameters may have a significant impact on the reasoning results. Further, we can explore additional neuralsymbolic reasoning approaches, expanding beyond the current neural-driven symbolic fashion. Second, the reasoning process relies on attention mechanisms, but it has been pointed out (Sui et al., 2022) that existing attention-based methods are prone to visit the noncausal features as the shortcut to predictions. In future work, we will try to focus on researching causal attention mechanisms for a more solid comprehension of the complicated correlations between temporal knowledge. 

## **Acknowledgements** 

This work is particularly supported by the National Key Research and Development Program of China (No. 2022YFB31040103), and the National Natural Science Foundation of China (No. 62302507). 

## **References** 

- John S. Baras and George Theodorakopoulos. 2010. _Path Problems in Networks_ . Synthesis Lectures on Communication Networks. Morgan & Claypool Publishers. 

- Richard Bellman. 1958. On a routing problem. _Quarterly of Applied Mathematics_ , 16:87–90. 

- Kurt D. Bollacker, Colin Evans, Praveen Paritosh, Tim Sturge, and Jamie Taylor. 2008. Freebase: a collaboratively created graph database for structuring human knowledge. In _Proceedings of the ACM SIGMOD International Conference on Management of Data, SIGMOD 2008_ , pages 1247–1250. 

- Antoine Bordes, Nicolas Usunier, Alberto GarciaDuran, Jason Weston, and Oksana Yakhnenko. 2013. Translating embeddings for modeling multirelational data. In _Advances in neural information processing systems_ , pages 2787–2795. 

- Kai Chen, Ye Wang, Yitong Li, and Aiping Li. 2022. Rotateqvs: Representing temporal information as 

rotations in quaternion vector space for temporal knowledge graph completion. In _Proceedings of the 60th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), ACL 2022, Dublin, Ireland, May 22-27, 2022_ , pages 5843– 5857. Association for Computational Linguistics. 

- Shib Sankar Dasgupta, Swayambhu Nath Ray, and Partha Talukdar. 2018. Hyte: Hyperplane-based temporally aware knowledge graph embedding. In _Proceedings of the 2018 conference on empirical methods in natural language processing_ , pages 2001– 2011. 

- Tim Dettmers, Pasquale Minervini, Pontus Stenetorp, and Sebastian Riedel. 2018. Convolutional 2d knowledge graph embeddings. In _Proceedings of the ThirtySecond AAAI Conference on Artificial Intelligence_ , pages 1811–1818. 

- Fredo Erxleben, Michael Günther, Markus Krötzsch, Julian Mendez, and Denny Vrandecic. 2014. Introducing wikidata to the linked data web. In _The Semantic Web - ISWC 2014 - 13th International Semantic Web Conference, Riva del Garda, Italy, October 19-23, 2014. Proceedings, Part I_ , volume 8796 of _Lecture Notes in Computer Science_ , pages 50–65. Springer. 

Lester Randolph Ford. 1956. Network flow theory. 

- Alberto García-Durán, Sebastijan Dumancic, and Mathias Niepert. 2018. Learning sequence encoders for temporal knowledge graph completion. In _Proceedings of the 2018 Conference on Empirical Methods in Natural Language Processing, Brussels, Belgium, October 31 - November 4, 2018_ , pages 4816–4821. Association for Computational Linguistics. 

- Rishab Goel, Seyed Mehran Kazemi, Marcus Brubaker, and Pascal Poupart. 2020. Diachronic embedding for temporal knowledge graph completion. _Proceedings of the AAAI Conference on Artificial Intelligence_ , 34(04):3988–3995. 

- Shu Guo, Quan Wang, Lihong Wang, Bin Wang, and Li Guo. 2016. Jointly embedding knowledge graphs and logical rules. In _Proceedings of the 2016 Conference on Empirical Methods in Natural Language Processing, EMNLP 2016, Austin, Texas, USA, November 1-4, 2016_ , pages 192–202. The Association for Computational Linguistics. 

- Shu Guo, Quan Wang, Lihong Wang, Bin Wang, and Li Guo. 2018. Knowledge graph embedding with iterative guidance from soft rules. In _Proceedings of the Thirty-Second AAAI Conference on Artificial Intelligence, (AAAI-18), the 30th innovative Applications of Artificial Intelligence (IAAI-18), and the 8th AAAI Symposium on Educational Advances in Artificial Intelligence (EAAI-18), New Orleans, Louisiana, USA, February 2-7, 2018_ , pages 4816–4823. AAAI Press. 

- Zhen Han, Peng Chen, Yunpu Ma, and Volker Tresp. 2020. Dyernie: Dynamic evolution of riemannian 

manifold embeddings for temporal knowledge graph completion. In _Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing, EMNLP 2020, Online, November 16-20, 2020_ , pages 7301–7316. Association for Computational Linguistics. 

- Zhen Han, Peng Chen, Yunpu Ma, and Volker Tresp. 2021a. Explainable subgraph reasoning for forecasting on temporal knowledge graphs. In _9th International Conference on Learning Representations, ICLR 2021, Virtual Event, Austria, May 3-7, 2021_ . OpenReview.net. 

- Zhen Han, Zifeng Ding, Yunpu Ma, Yujia Gu, and Volker Tresp. 2021b. Learning neural ordinary equations for forecasting future links on temporal knowledge graphs. In _Proceedings of the 2021 Conference on Empirical Methods in Natural Language Processing, EMNLP 2021, Virtual Event / Punta Cana, Dominican Republic, 7-11 November, 2021_ , pages 8352– 8364. Association for Computational Linguistics. 

- Kurt Hornik. 1991. Approximation capabilities of multilayer feedforward networks. _Neural Networks_ , 4(2):251–257. 

- Prachi Jain, Sushant Rathi, Mausam, and Soumen Chakrabarti. 2020. Temporal knowledge base completion: New algorithms and evaluation protocols. In _Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing, EMNLP 2020, Online, November 16-20, 2020_ , pages 3733– 3747. Association for Computational Linguistics. 

- Woojeong Jin, Meng Qu, Xisen Jin, and Xiang Ren. 2020. Recurrent event network: Autoregressive structure inference over temporal knowledge graphs. In _Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing (EMNLP)_ , pages 6669–6683. 

- Jaehun Jung, Jinhong Jung, and U Kang. 2021. Learning to walk across time for interpretable temporal knowledge graph completion. In _KDD ’21: The 27th ACM SIGKDD Conference on Knowledge Discovery and Data Mining, Virtual Event, Singapore, August 14-18, 2021_ , pages 786–795. ACM. 

- Andrey V. Korotayev and Sergey V. Tsirel. 2010. A spectral analysis of world gdp dynamics: Kondratieff waves, kuznets swings, juglar and kitchin cycles in global economic development, and the 2008c2009 economic crisis. _Structure and Dynamics: eJournal of Anthropological and Related Sciences_ , pages 1– 57. 

- Timothée Lacroix, Guillaume Obozinski, and Nicolas Usunier. 2020. Tensor decompositions for temporal knowledge base completion. In _8th International Conference on Learning Representations, ICLR 2020, Addis Ababa, Ethiopia, April 26-30, 2020_ . OpenReview.net. 

- Timothée Lacroix, Nicolas Usunier, and Guillaume Obozinski. 2018. Canonical tensor decomposition for knowledge base completion. In _Proceedings of the 35th International Conference on Machine Learning, ICML 2018, Stockholmsmässan, Stockholm, Sweden, July 10-15, 2018_ , volume 80 of _Proceedings of Machine Learning Research_ , pages 2869–2878. PMLR. 

- Ni Lao and William W. Cohen. 2010. Relational retrieval using a combination of path-constrained random walks. _Mach. Learn._ , 81(1):53–67. 

- Ni Lao, Amarnag Subramanya, Fernando C. N. Pereira, and William W. Cohen. 2012. Reading the web with learned syntactic-semantic inference rules. In _Proceedings of the 2012 Joint Conference on Empirical Methods in Natural Language Processing and Computational Natural Language Learning, EMNLPCoNLL 2012, July 12-14, 2012, Jeju Island, Korea_ , pages 1017–1026. ACL. 

- Jennifer Lautenschlager, Steve Shellman, and Michael Ward. 2015. Icews events and aggregations. _Harvard Dataverse_ , 3. 

- Julien Leblay and Melisachew Wudage Chekol. 2018. Deriving validity time in knowledge graph. In _Companion Proceedings of the The Web Conference 2018_ , pages 1771–1776. 

- Yujia Li, Shiliang Sun, and Jing Zhao. 2022a. Tirgn: Time-guided recurrent graph network with localglobal historical patterns for temporal knowledge graph reasoning. In _Proceedings of the Thirty-First International Joint Conference on Artificial Intelligence, IJCAI 2022, Vienna, Austria, 23-29 July 2022_ , pages 2152–2158. ijcai.org. 

- Zixuan Li, Saiping Guan, Xiaolong Jin, Weihua Peng, Yajuan Lyu, Yong Zhu, Long Bai, Wei Li, Jiafeng Guo, and Xueqi Cheng. 2022b. Complex evolutional pattern learning for temporal knowledge graph reasoning. In _Proceedings of the 60th Annual Meeting of the Association for Computational Linguistics (Volume 2: Short Papers), ACL 2022, Dublin, Ireland, May 22-27, 2022_ , pages 290–296. Association for Computational Linguistics. 

- Zixuan Li, Xiaolong Jin, Wei Li, Saiping Guan, Jiafeng Guo, Huawei Shen, Yuanzhuo Wang, and Xueqi Cheng. 2021. Temporal knowledge graph reasoning based on evolutional representation learning. In _SIGIR ’21: The 44th International ACM SIGIR Conference on Research and Development in Information Retrieval, Virtual Event, Canada, July 11-15, 2021_ , pages 408–417. ACM. 

- Ke Liang, Lingyuan Meng, Meng Liu, Yue Liu, Wenxuan Tu, Siwei Wang, Sihang Zhou, and Xinwang Liu. 2023. Learn from relational correlations and periodic events for temporal knowledge graph reasoning. In _Proceedings of the 46th International ACM SIGIR Conference on Research and Development in Information Retrieval, SIGIR 2023, Taipei, Taiwan, July 23-27, 2023_ , pages 1559–1568. ACM. 

- Yankai Lin, Zhiyuan Liu, Maosong Sun, Yang Liu, and Xuan Zhu. 2015. Learning entity and relation embeddings for knowledge graph completion. In _Proceedings of the Twenty-Ninth AAAI Conference on Artificial Intelligence, January 25-30, 2015, Austin, Texas, USA_ , pages 2181–2187. AAAI Press. 

- Yushan Liu, Yunpu Ma, Marcel Hildebrandt, Mitchell Joblin, and Volker Tresp. 2022. Tlogic: Temporal logical rules for explainable link forecasting on temporal knowledge graphs. In _Thirty-Sixth AAAI Conference on Artificial Intelligence, AAAI 2022, Thirty-Fourth Conference on Innovative Applications of Artificial Intelligence, IAAI 2022, The Twelveth Symposium on Educational Advances in Artificial Intelligence, EAAI 2022 Virtual Event, February 22 - March 1, 2022_ , pages 4120–4127. AAAI Press. 

- Farzaneh Mahdisoltani, Joanna Biega, and Fabian M. Suchanek. 2015. YAGO3: A knowledge base from multilingual wikipedias. In _CIDR 2015, Seventh Biennial Conference on Innovative Data Systems Research_ . 

- Arvind Neelakantan, Benjamin Roth, and Andrew McCallum. 2015. Compositional vector space models for knowledge base inference. In _2015 AAAI Spring Symposia, Stanford University, Palo Alto, California, USA, March 22-25, 2015_ . AAAI Press. 

- Meng Qu and Jian Tang. 2019. Probabilistic logic neural networks for reasoning. In _Advances in Neural Information Processing Systems 32: Annual Conference on Neural Information Processing Systems 2019, NeurIPS 2019, December 8-14, 2019, Vancouver, BC, Canada_ , pages 7710–7720. 

- Luc De Raedt, Angelika Kimmig, and Hannu Toivonen. 2007. Problog: A probabilistic prolog and its application in link discovery. In _IJCAI 2007, Proceedings of the 20th International Joint Conference on Artificial Intelligence, Hyderabad, India, January 6-12, 2007_ , pages 2462–2467. 

- Matthew Richardson and Pedro M. Domingos. 2006. Markov logic networks. _Mach. Learn._ , 62(1-2):107– 136. 

- Michael Sejr Schlichtkrull, Thomas N. Kipf, Peter Bloem, Rianne van den Berg, Ivan Titov, and Max Welling. 2018. Modeling relational data with graph convolutional networks. In _The Semantic Web - 15th International Conference, ESWC 2018_ , volume 10843, pages 593–607. 

- Chao Shang, Yun Tang, Jing Huang, Jinbo Bi, Xiaodong He, and Bowen Zhou. 2019. End-to-end structureaware convolutional networks for knowledge base completion. In _The Thirty-Third AAAI Conference on Artificial Intelligence_ , pages 3060–3067. 

- Yichen Song, Aiping Li, Hongkui Tu, Kai Chen, and Chenchen Li. 2021. A novel encoder-decoder knowledge graph completion model for robot brain. _Frontiers Neurorobotics_ , 15:674428. 

- Yongduo Sui, Xiang Wang, Jiancan Wu, Min Lin, Xiangnan He, and Tat-Seng Chua. 2022. Causal attention for interpretable and generalizable graph classification. In _KDD ’22: The 28th ACM SIGKDD Conference on Knowledge Discovery and Data Mining, Washington, DC, USA, August 14 - 18, 2022_ , pages 1696–1705. ACM. 

- Haohai Sun, Jialun Zhong, Yunpu Ma, Zhen Han, and Kun He. 2021. Timetraveler: Reinforcement learning for temporal knowledge graph forecasting. In _Proceedings of the 2021 Conference on Empirical Methods in Natural Language Processing, EMNLP 2021, Virtual Event / Punta Cana, Dominican Republic, 7- 11 November, 2021_ , pages 8306–8319. Association for Computational Linguistics. 

- Théo Trouillon, Johannes Welbl, Sebastian Riedel, Éric Gaussier, and Guillaume Bouchard. 2016. Complex embeddings for simple link prediction. In _Proceedings of the 33nd International Conference on Machine Learning, ICML 2016_ , volume 48 of _JMLR Workshop and Conference Proceedings_ , pages 2071– 2080. 

- Shikhar Vashishth, Soumya Sanyal, Vikram Nitin, and Partha P. Talukdar. 2020. Composition-based multirelational graph convolutional networks. In _8th International Conference on Learning Representations, ICLR 2020, Addis Ababa, Ethiopia, April 26-30, 2020_ . OpenReview.net. 

- Petar Velickovic, Guillem Cucurull, Arantxa Casanova, Adriana Romero, Pietro Liò, and Yoshua Bengio. 2018. Graph attention networks. In _6th International Conference on Learning Representations, ICLR 2018, Vancouver, BC, Canada, April 30 - May 3, 2018, Conference Track Proceedings_ . OpenReview.net. 

- Pengwei Wang, Dejing Dou, Fangzhao Wu, Nisansa de Silva, and Lianwen Jin. 2019. Logic rules powered knowledge graph embedding. _CoRR_ , abs/1903.03772. 

- Zhen Wang, Jianwen Zhang, Jianlin Feng, and Zheng Chen. 2014. Knowledge graph embedding by translating on hyperplanes. In _Proceedings of the TwentyEighth AAAI Conference on Artificial Intelligence, July 27 -31, 2014, Québec City, Québec, Canada_ , pages 1112–1119. AAAI Press. 

- Chengjin Xu, Yung-Yu Chen, Mojtaba Nayyeri, and Jens Lehmann. 2021. Temporal knowledge graph completion using a linear temporal regularizer and multivector embeddings. In _Proceedings of the 2021 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, NAACL-HLT 2021, Online, June 6-11, 2021_ , pages 2569–2578. Association for Computational Linguistics. 

- Chengjin Xu, Mojtaba Nayyeri, Fouad Alkhoury, Jens Lehmann, and Hamed Shariat Yazdi. 2019. Temporal knowledge graph embedding model based on additive time series decomposition. _CoRR_ , abs/1911.07893. 

- Chengjin Xu, Mojtaba Nayyeri, Fouad Alkhoury, Hamed Shariat Yazdi, and Jens Lehmann. 2020. TeRo: A time-aware knowledge graph embedding via temporal rotation. In _Proceedings of the 28th International Conference on Computational Linguistics_ , pages 1583–1593, Barcelona, Spain (Online). International Committee on Computational Linguistics. 

- Chengjin Xu, Mojtaba Nayyeri, Yung-Yu Chen, and Jens Lehmann. 2023. Geometric algebra based embeddings for static and temporal knowledge graph completion. _IEEE Trans. Knowl. Data Eng._ , 35(5):4838–4851. 

- Bishan Yang, Wen-tau Yih, Xiaodong He, Jianfeng Gao, and Li Deng. 2015. Embedding entities and relations for learning and inference in knowledge bases. In _3rd International Conference on Learning Representations, ICLR 2015_ . 

- Matthew D. Zeiler and Rob Fergus. 2014. Visualizing and understanding convolutional networks. In _Computer Vision - ECCV 2014 - 13th European Conference, Zurich, Switzerland, September 6-12, 2014, Proceedings, Part I_ , volume 8689 of _Lecture Notes in Computer Science_ , pages 818–833. Springer. 

- Jing Zhang, Bo Chen, Lingxi Zhang, Xirui Ke, and Haipeng Ding. 2021. Neural, symbolic and neuralsymbolic reasoning on knowledge graphs. _AI Open_ , 2:14–35. 

- Yongqi Zhang and Quanming Yao. 2022. Knowledge graph reasoning with relational digraph. In _WWW_ 

- _’22: The ACM Web Conference 2022, Virtual Event, Lyon, France, April 25 - 29, 2022_ , pages 912–924. ACM. 

- Xiaojuan Zhao, Yan Jia, Aiping Li, Rong Jiang, Kai Chen, and Ye Wang. 2021. Target relational attentionoriented knowledge graph reasoning. _Neurocomputing_ , 461:577–586. 

- Cunchao Zhu, Muhao Chen, Changjun Fan, Guangquan Cheng, and Yan Zhang. 2021a. Learning from history: Modeling temporal knowledge graphs with sequential copy-generation networks. In _Thirty-Fifth AAAI Conference on Artificial Intelligence, AAAI 2021, Thirty-Third Conference on Innovative Applications of Artificial Intelligence, IAAI 2021, The Eleventh Symposium on Educational Advances in Artificial Intelligence, EAAI 2021, Virtual Event, February 2-9, 2021_ , pages 4732–4740. AAAI Press. 

- Zhaocheng Zhu, Zuobai Zhang, Louis-Pascal A. C. Xhonneux, and Jian Tang. 2021b. Neural bellmanford networks: A general graph neural network framework for link prediction. In _Advances in Neural Information Processing Systems 34: Annual Conference on Neural Information Processing Systems 2021, NeurIPS 2021, December 6-14, 2021, virtual_ , pages 29476–29490. 

## **A Bellman-Ford Shortest Path Algorithm** 

**Algorithm 2** The Bellman-Ford Shortest Path Algorithm 

**Input:** A graph _G_ with _n_ nodes, the edge set _E_ , the source node _s_ 

- **Output:** The distance from node s to any node in the graph 

1: _d_[(0)] _←_ [+ _∞,_ + _∞, ...,_ + _∞_ ] 

- 2: _d_[(0)] [ _s_ ] _←_ 0 

- 3: **for** _i_ = 1 **to** _n −_ 1 **do** 

- 4: _d_[(] _[i]_[)] _← d_[(] _[i][−]_[1)] 

- 5: **for** each edge ( _u, v_ ) _∈ E_ **do** 6: _d_[(] _[i]_[)] [ _v_ ] _←_ min _d_[(] _[i]_[)] [ _v_ ] _, d_[(] _[i][−]_[1)] [ _u_ ] + _w_ ( _u, v_ ) � � 

- 7: **end for** 

- 8: **end for** 

- 9: **return** _d_[(] _[n][−]_[1)] 

The Bellman-Ford shortest path algorithm (Ford, 1956; Bellman, 1958; Baras and Theodorakopoulos, 2010), founded by Richard Bellman and Lester Ford, is an efficient algorithm for searching the shortest path from the starting point to each node in a graph. The details are shown in Algorithm 2. And the basic idea is: For a graph with _n_ nodes, the shortest path between two nodes in the graph can contain at most _n −_ 1 edges. Then, we can perform _n −_ 1 times of relaxation operations recursively on the graph to obtain the shortest path from the source node to all nodes in the graph. 

_t_ 1 _r_ 1[(] _[e]_[1] _[, e]_[2][)] _[ ∧] r[t]_[3] 3[(] _[e]_[2] _[, e]_[5][)] _[ ∧] r[t]_[6] 6[(] _[e]_[5] _[, e]_[6][)][ are two distinct] temporal paths from _e_ 1 to _e_ 6. 

## **C Benchmark Datasets** 

## **C.1 Interpolation datasets** 

To evaluate our proposed TKG Reasoning method, we perform link prediction task on four widely-used interpolation datasets, namely ICEWS14, ICEWS05-15 (García-Durán et al., 2018), YAGO11k (Dasgupta et al., 2018) and WIKIDATA12k (Dasgupta et al., 2018).[8] The left half of Table 4 summarizes the details of the four interpolation datasets that we use. 

ICEWS (Lautenschlager et al., 2015) is a repository containing political events with a specific timestamp. ICEWS14 and ICEWS05-15 (GarcíaDurán et al., 2018) are two subsets of ICEWS corresponding to facts from 1/1/2014 to 12/31/2014 and from 1/1/2005 to 12/31/2015. YAGO11k and WIKIDATA12k (Dasgupta et al., 2018) are subsets of YAGO3 (Mahdisoltani et al., 2015) and WIKI (Erxleben et al., 2014), where time annotations are represented as time intervals. We derive the two datasets from HyTE (Dasgupta et al., 2018) to obtain the same year-level granularity by dropping the month and date information. And the characteristic of a dataset used for the interpolation is disorderly arrangement, which means that it is random and not sorted according to time. 

## **C.2 Extrapolation datasets** 

## **B An Illustration of Temporal Paths and Temporal Links** 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/acl_2024_A_Unified_Temporal_Knowledge_Graph_Reasoning_Model_Towards_Interpolation_and_Extrapolation_images/acl_2024_A_Unified_Temporal_Knowledge_Graph_Reasoning_Model_Towards_Interpolation_and_Extrapolation.pdf-0013-19.png)


Figure 4: An illustration of temporal paths and temporal links. 

Fig. 4 illustrates some temporal paths and temporal links. For example, _rt_ 11[(] _[e]_[1] _[, e]_[2][)] is a temporal link pointing from _e_ 1 to _e_ 2, while _t_ 2 and _r_ 2[(] _[e]_[1] _[, e]_[3][)] _[ ∧] r[t]_[5] 5[(] _[e]_[3] _[, e]_[4][)] _[ ∧] r[t]_[7] 7[(] _[e]_[4] _[, e]_[6][)] 

The right half of Table 4 summarizes the details of the four extrapolation datasets that we use, namely ICEWS14 (García-Durán et al., 2018), ICEWS18 (Jin et al., 2020), YAGO (Mahdisoltani et al., 2015) and WIKI (Leblay and Chekol, 2018).[9] ICEWS14 (García-Durán et al., 2018) and ICEWS18 (Jin et al., 2020) are two subsets of ICEWS (Lautenschlager et al., 2015) corresponding to facts from 1/1/2014 to 12/31/2014 and from 1/1/2018 to 10/31/2018. We drop the month and date information of YAGO and WIKI here for ease of processing and to obtain the same year-level granularity as (Jin et al., 2020). Compared to the interpolation datasets (see the left half of Table 4), the facts in the four datasets for extrapolation are arranged in ascending order according to time. For the purpose 

8All of the four interpolation datasets can be downloaded from https://github.com/soledad921/ATISE. 

9All of the four extrapolation datasets can be downloaded from https://github.com/Lee-zix/RE-GCN. 

|Dataset|**Interpolation**|**Extrapolation**|
|---|---|---|
||ICEWS14<br>YAGO11k<br>ICEWS05-15<br>Wikidata12k|ICEWS14<br>ICEWS18<br>YAGO<br>WIKI|
|Entities<br>Relations<br>Train<br>Validation<br>Test<br>Time granularity|7,128<br>10,623<br>10488<br>12554<br>230<br>10<br>251<br>24<br>72,826<br>16,408<br>386,962<br>32,497<br>8,941<br>2,050<br>46,275<br>4,062<br>8,963<br>2,051<br>46,092<br>4,062<br>24 hours<br>1 year<br>24 hours<br>1 year|6,869<br>23,033<br>10,623<br>12,554<br>230<br>256<br>10<br>24<br>74,845<br>373,018<br>161,540<br>539,286<br>8,514<br>45,995<br>19,523<br>67,538<br>7,371<br>49,545<br>20,026<br>63,110<br>24 hours<br>24 hours<br>1 year<br>1 year|



Table 4: Statistics of our experimented datasets for both interpolation setting and extrapolation setting. 

of extrapolation, we follow Han et al. (2021a) and split each dataset into three subsets by time, ensuring (time of training set) < (time of validation set) < (time of test set). Thus, we can smoothly use past knowledge (facts in the training set) to predict the future. 

Some recent works have shown that the timeunwise filtered setting (Bordes et al., 2013) may probably get incorrect higher ranking scores (Li et al., 2021), and that the time-wise filtered setting (Xu et al., 2019; Goel et al., 2020) is more suitable for TKG Reasoning since it ensures the facts which do not appear at time _t_ are still considered as candidates for evaluating the given test quadruple (Xu et al., 2020). Therefore, we use the time-wise filtered setting (Xu et al., 2019; Goel et al., 2020) to report the experimental results. 

## **D Experimental Details for Interpolation Reasoning** 

## **D.1 Setup** 

In this setting, we split each dataset into three subsets: a training set _T_ train, a validation set _T_ valid and a test set _T_ test. 

For training, we further split the training set _T_ train into two parts: the fact set _F_ train including 3 _/_ 4 of the training quadruples used to extract temporal paths, and the query set _Q_ train including the other 1 _/_ 4 of the training quadruples used as queries. Thus, we can train our model by using temporal paths collected by _F_ train to answer queries in _Q_ train. 

For verification and testing, we accomplish reasoning by answering queries derived from the validation set and test set, where the whole training set _T_ train is prepared to extract and collect temporal paths. Since the entire training set is known to us and we do not use any data from validation or test set during the training process, we can ensure that there is no data leakage issue. 

## **D.2 Baselines** 

For interpolation TKG Reasoning baselines, we consider TTransE (Leblay and Chekol, 2018), HyTE (Dasgupta et al., 2018), TA-DistMult (García-Durán et al., 2018), ATiSE (Xu et al., 2019), and TeRo (Xu et al., 2020), TComplEx (Lacroix et al., 2020), T-GAP (Jung et al., 2021), TELM (Xu et al., 2021), RotateQVS (Chen et al., 2022) and TGeomE++ (Xu et al., 2023). 

## **D.3 Hyperparameter** 

To seek and find proper hyperparameters, we utilize a grid search empirically over the following ranges for all four datasets: embedding dimension _d_ in _{_ 32 _,_ 64 _,_ 128 _}_ , learning rate in [10 _[−]_[5] _,_ 10 _[−]_[3] ], dimension for attention _da_ in [3 _,_ 4 _,_ 5], dropout in [0 _._ 1 _,_ 0 _._ 2 _,_ 0 _._ 3], batch size in [5 _,_ 10 _,_ 20 _,_ 50], the maximum length of temporal paths _L_ in [3 _,_ 4 _,_ 5], activation function _δ_ in [identity, tanh, ReLU], and the optimizer we use is Adam. 

And we have found out the best hyperparameters combination as follows: for ICEWS14, we set learning rate as 0 _._ 0003, batch size as 10, and _δ_ as _identity_ ; for ICEWS05-15, we set learning rate as 0 _._ 00005, batch size as 5, and _δ_ as _identity_ ; for YAGO11k, we set learning rate as 0 _._ 00005, batch size as 20, and _δ_ as _ReLU_ ; for WIKIDATA12k, we set learning rate as 0 _._ 00002, batch size as 20, and _δ_ as _ReLU_ ; and for all of four datasets, we choose _L_ as 5, _da_ as 5, dropout as 0 _._ 2, and _d_ as 128. 

## **E Experimental Details for Extrapolation Reasoning** 

## **E.1 Setup** 

In this setting, first of all, we need to arrange each dataset in ascending order according to time. In addition to the same splitting operation as in Section D.1, we need to ensure: (time of training set) < (time of validation set) < (time of test set). Since some entities and relations in the validation or test set may not have appeared in the training set, our 

extrapolation setting can actually be seen as an inductive inference (Sun et al., 2021). 

We use the ground truths for our extrapolation TKG Reasoning, as is the case with many previous methods (Jin et al., 2020; Li et al., 2021; Sun et al., 2021). Specifically, for all of the training, validation and testing, we predict future events assuming ground truths of the preceding events are given at inference time (Han et al., 2021a). 

## **E.2 Baselines** 

For extrapolation TKG Reasoning baselines, we consider RE-NET (Jin et al., 2020), CyGNeT (Zhu et al., 2021a), TANGO (Han et al., 2021b), xERTE (Han et al., 2021a), RE-GCN (Li et al., 2021), TITer (Sun et al., 2021), CEN (Li et al., 2022b), TLogic (Liu et al., 2022), TIRGN (Li et al., 2022a) and RPC (Liang et al., 2023). 

## **E.3 Hyperparameter** 

A grid search has been taken empirically for the extrapolation TKG Reasoning on the same hyperparameter ranges as in Section D.3. And we have found out the best hyperparameters combination as follows: for ICEWS14, we set learning rate as 0 _._ 0003, batch size as 10, _d_ as 128, and _δ_ as _identity_ ; for ICEWS18, we set learning rate as 0 _._ 00005, batch size as 5, _d_ as 64, and _δ_ as _ReLU_ ; for YAGO, we set learning rate as 0 _._ 00003, batch size as 10, _d_ as 64, and _δ_ as _identity_ ; for WIKI, we set learning rate as 0 _._ 00003, batch size as 5, _d_ as 64, and _δ_ as _identity_ ; and for all of four datasets, we choose _L_ as 5, _da_ as 5, and dropout as 0 _._ 2. 

## **F Details of the Pipeline Setting** 

To conduct our experimental pipeline, we sample a certain ratio (60% for Setting A, 70% for Setting B, and 80% for Setting C) of facts to train interpolation, while the other unsampled facts serve to provide incomplete quadruples (with either the subject entity or object entity masked) to be completed. The completed quadruples are then treated as new knowledge. Finally, the newly acquired knowledge from interpolation could be combined with the interpolation training set, ordered chronologically, and used as the training set for downstream (extrapolation) tasks, with the test set of the standard extrapolation dataset used to evaluate experimental results. 

The process of the proposed pipeline can be seen in Table 6. 

|Ratio|Inter-|Extra-|MRR|Hits@1|Hits@3|Hits@10|
|---|---|---|---|---|---|---|
|60%<br>60%<br>60%<br>60%<br>60%<br>60%|-<br>TELM<br>-<br>RotateQVS<br>-<br>TPAR|TLogic<br>TLogic<br>TIRGN<br>TIRGN<br>TPAR<br>TPAR|41.85<br>40.88<br>42.53<br>40.86<br>43.86<br>44.56|32.51<br>31.66<br>32.04<br>30.24<br>33.81<br>34.68|47.05<br>45.86<br>48.10<br>45.78<br>49.04<br>49.68|59.58<br>58.69<br>62.16<br>61.63<br>62.83<br>63.32|
|70%<br>70%<br>70%<br>70%<br>70%<br>70%|-<br>TELM<br>-<br>RotateQVS<br>-<br>TPAR|TLogic<br>TLogic<br>TIRGN<br>TIRGN<br>TPAR<br>TPAR|41.88<br>44.55<br>43.68<br>42.35<br>45.49<br>46.07|32.51<br>32.29<br>33.23<br>31.87<br>35.69<br>35.92|46.91<br>46.55<br>48.99<br>47.78<br>50.33<br>51.06|59.97<br>59.53<br>63.59<br>62.25<br>64.30<br>65.02|
|80%<br>80%<br>80%<br>80%<br>80%<br>80%|-<br>TELM<br>-<br>RotateQVS<br>-<br>TPAR|TLogic<br>TLogic<br>TIRGN<br>TIRGN<br>TPAR<br>TPAR|42.18<br>42.09<br>43.69<br>42.78<br>46.36<br>46.66|32.81<br>32.69<br>32.97<br>32.15<br>36.42<br>36.68|47.27<br>47.31<br>49.10<br>48.06<br>51.34<br>51.82|60.18<br>59.78<br>64.28<br>63.14<br>65.45<br>65.54|



Table 5: A detailed version of Table 3. 

The detailed results of the pipeline setting can be found in Table 5, where a concise version has been shown in Table 3 in Section 5.3. 

## **G Analysis on Path Length** 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/acl_2024_A_Unified_Temporal_Knowledge_Graph_Reasoning_Model_Towards_Interpolation_and_Extrapolation_images/acl_2024_A_Unified_Temporal_Knowledge_Graph_Reasoning_Model_Towards_Interpolation_and_Extrapolation.pdf-0015-13.png)


Figure 5: The MRR performance with different maximum path lengths for the extrapolation reasoning on the ICEWS14 test set. 

Fig. 5 displays the MRR performance with different maximum path lengths for the extrapolation reasoning on ICEWS14 test set. We compare our TPAR with the symbolic method TLogic (Liu et al., 2022), where the maximum length _L_ means it contains all lengths not exceeding _L_ ( _L_ = 1 _,_ 2 _, · · ·_ 5). 

As the maximum length _L_ increases, a greater number of temporal links are covered and can provide more detailed information. However, this also makes it more challenging to learn and reason with this information. The results demonstrate that the performance of TLogic decreases for _L_ values of 4 or higher. In contrast, when L is too small, such as _L ≤_ 2, our TPAR approach struggles to perform 

**Inputs:** A ratio (denoted as _r_ ) of known facts sampled from _G_ , denoted as _rG_ . Step 1 **Queries:** The other unsampled facts (1 _− r_ ) _G_ with either the subject or the (Interpolation) object entity masked. **Outputs:** Newly acquired facts _Gnew_ by completing the queries. **Inputs:** The basic _rG_ , and the newly acquired _Gnew_ . Step 2 **Queries:** The standard extrapolation test set. (Extrapolation) **Outputs:** Performance evaluation. 

Table 6: The process of the proposed pipeline. 

well due to the limited amount of information encoded in these short temporal paths. An interesting discovery can be made: For symbolic TLogic, a balance needs to be struck between the richness of the information contained and the ability to effectively learn from it. However, our TPAR appears to alleviate this issue, as neural-symbolic reasoning is more tolerant of ambiguous and noisy data. 

|Length|Hits@1<br>Hits@3<br>Hits@10<br>MRR|
|---|---|
|1|10.03<br>10.45<br>10.58<br>10.32|
|2|10.39<br>11.06<br>11.45<br>10.84|
|3|15.13<br>22.03<br>32.16<br>21.03|
|4|15.59<br>22.56<br>33.15<br>21.62|
|5|17.35<br>25.14<br>37.42<br>24.12|



Table 7: Performance of our proposed TPAR with different maximum path lengths for the interpolation reasoning on the YAGO11k test set. 

We also take a similar study on the interpolation reasoning, and Table 7 illustrates the performance of our proposed TPAR with different maximum path lengths for the interpolation reasoning on the YAGO11k test set. As the length increases, all metrics of our TPAR have significantly improved. 

methods like TLogic (Liu et al., 2022) demand that all links in a path be ordered chronologically. For example, given a path _P_ = _r[t]_[1] 1[(] _[e]_[1] _[, e]_[2][)] _[∧] r[t]_[2] 2[(] _[e]_[2] _[, e]_[3][)] _[∧]_ 

_rt_ 33[(] _[e]_[3] _[, e]_[4][)][,][TLogic][requires] _[t][q][>][t]_[1] _[≥][t]_[2] _[≥][t]_[3][,] while we simply require _tq > ti_ for _i_ = 1 _,_ 2 _,_ 3. We conduct an analysis on the chronological order of links in a path, and the results are shown in Table 8. The results indicate that when chronological order is relaxed, the performance of TLogic decreases, while that of our TPAR increases. We believe that by relaxing chronological order in paths, we can gather more paths, thereby providing more information for inference. Nonetheless, this may introduce noise and uncertainty into the process, which ultimately reduces the effectiveness of symbolic methods such as TLogic. On the other hand, our TPAR employs a neural-symbolic approach, which not only mitigates the impact of these challenges but also ensures excellent inference performance, showcasing its strong robustness. 

## **H Analysis on Chronological Order** 

|Model|Model|**TLogic**|**TLogic**|**TPAR**|**TPAR**|
|---|---|---|---|---|---|
|Chronological Order||�|�|�|�|
|ICEWS14|Hits@1|32.60|33.56|36.88|36.02|
||Hits@3|47.06|48.27|52.28|51.49|
||Hits@10|60.06|61.23|65.89|64.30|
||MRR|42.02|43.04|46.89|45.77|
|ICEWS18|Hits@1|19.92|20.54|26.58|25.24|
||Hits@3|33.04|33.95|39.27|38.55|
||Hits@10|47.75|48.53|56.94|55.54|
||MRR|28.79|29.82|35.76|33.98|



Table 8: An analysis on chronological order for the extrapolation reasoning. 

As discussed in Section 4.1, for extrapolation reasoning, our TPAR does not require strict chronological order between links in a path. Alternatively, 

