# **AdaProp: Learning Adaptive Propagation for Graph Neural Network based Knowledge Graph Reasoning** 

Quanming Yao[†] qyaoaa@tsinghua.edu.cn Department of Electronic Engineering, Tsinghua University Beijing, China 

Zhanke Zhou[∗] cszkzhou@comp.hkbu.edu.hk Department of Computer Science, Hong Kong Baptist University Hong Kong, China 

Yongqi Zhang[∗] zhangyongqi@4paradigm.com 4Paradigm Inc. Beijing, China 

Xiaowen Chu 

Bo Han 

xwchu@ust.hk Data Science and Analytics Thrust, HKUST (Guangzhou) Guangzhou, China 

bhanml@comp.hkbu.edu.hk Department of Computer Science, Hong Kong Baptist University Hong Kong, China 

## **ABSTRACT** 

## **ACM Reference Format:** 

Yongqi Zhang, Zhanke Zhou, Quanming Yao, Xiaowen Chu, and Bo Han. 2023. AdaProp: Learning Adaptive Propagation for Graph Neural Network based Knowledge Graph Reasoning. In _Proceedings of the 29th ACM SIGKDD Conference on Knowledge Discovery and Data Mining (KDD ’23), August 6–10, 2023, Long Beach, CA, USA._ ACM, New York, NY, USA, 15 pages. https://doi.org/10.1145/3580305.3599404 

Due to the popularity of Graph Neural Networks (GNNs), various GNN-based methods have been designed to reason on knowledge graphs (KGs). An important design component of GNN-based KG reasoning methods is called the propagation path, which contains a set of involved entities in each propagation step. Existing methods use hand-designed propagation paths, ignoring the correlation between the entities and the query relation. In addition, the number of involved entities will explosively grow at larger propagation steps. In this work, we are motivated to learn an adaptive propagation path in order to filter out irrelevant entities while preserving promising targets. First, we design an incremental sampling mechanism where the nearby targets and layer-wise connections can be preserved with linear complexity. Second, we design a learning-based sampling distribution to identify the semantically related entities. Extensive experiments show that our method is powerful, efficient and semantic-aware. The code is available at https://github.com/LARS-research/AdaProp. 

## **1 INTRODUCTION** 

Knowledge graph (KG) is a semantic graph representing the relationships between real-world entities [16, 39]. Reasoning on KGs aims to deduce missing answers for given queries based on existing facts [1, 6, 20]. It has been widely applied in drug interaction prediction [49], personalized recommendation [4, 40], and question answering [12, 18]. An example of KG reasoning with query _(LeBron, lives_in, ?)_ and answer _Los Angeles_ is illustrated in Figure 1. Symbolically, the reasoning problem can be denoted as a _query_ ( _𝑒𝑞,𝑟𝑞,_ ?), with the query entity _𝑒𝑞_ and query relation _𝑟𝑞_ . The objective of KG reasoning is to find the target answer entity _𝑒𝑎_ for ( _𝑒𝑞,𝑟𝑞,_ ?) by the local evidence learned from the given KG. 

## **CCS CONCEPTS** 

Representative methods for KG reasoning can be classified into three categories: (i) triplet-based models [3, 10, 39, 52, 55] directly score each answer entity through the learned entity and relation embeddings; (ii) path-based methods [7, 9, 30, 31, 54] learn logic rules to generate the relational paths starting from _𝑒𝑞_ and explore which entity is more likely to be the target answer _𝑒𝑎_ ; (iii) GNNbased methods [32, 38, 53, 56] propagate entity representations among the local neigborhoods by graph neural networks [22]. Among the three categories, the GNN-based methods achieve the state-of-the-art performance [47, 53, 56]. Specifically, R-GCN [32] and CompGCN [38], propagate in the full neighborhoods over all the entities. Same as the general GCNs, they will easily suffer from over-smoothing problem [28]. NBFNet [56] and RED-GNN [53] propagates in the local neighborhoods around the query entity _𝑒𝑞_ . By selecting the range of propagation, they possess the stronger reasoning ability than R-GCN or CompGCN. However, their local neighborhoods mainly depend on the query entity, and the majority of entities will be involved at larger propagation steps. 

• **Computing methodologies** → **Knowledge representation and reasoning** . 

## **KEYWORDS** 

Knowledge graph, Graph embedding, Knowledge graph reasoning, Graph sampling, Graph neural network 

> ∗This work was partially performed when Z. Zhou was an intern in 4Paradigm. Y. Zhang and Z. Zhou have equal contribution, and correspondence is to Q. Yao. 

Permission to make digital or hard copies of all or part of this work for personal or classroom use is granted without fee provided that copies are not made or distributed for profit or commercial advantage and that copies bear this notice and the full citation on the first page. Copyrights for components of this work owned by others than the author(s) must be honored. Abstracting with credit is permitted. To copy otherwise, or republish, to post on servers or to redistribute to lists, requires prior specific permission and/or a fee. Request permissions from permissions@acm.org. _KDD ’23, August 6–10, 2023, Long Beach, CA, USA_ 

© 2023 Copyright held by the owner/author(s). Publication rights licensed to ACM. ACM ISBN 979-8-4007-0103-0/23/08...$15.00 https://doi.org/10.1145/3580305.3599404 

The existing GNN-based methods, which propagate among all the neighborhood over all the entities or around the query entity, 

KDD ’23, August 6–10, 2023, Long Beach, CA, USA 

Yongqi Zhang, Zhanke Zhou, Quanming Yao, Xiaowen Chu, and Bo Han 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/kdd_2024_AdaProp_Learning_Adaptive_Propagation_for_Graph_Neural_Network_based_Knowledge_Graph_Reasoning_images/kdd_2024_AdaProp_Learning_Adaptive_Propagation_for_Graph_Neural_Network_based_Knowledge_Graph_Reasoning.pdf-0002-02.png)


**----- Start of picture text -----**<br>
LeBron.JR Savan Nah<br>located_in<br>son_of<br>?<br>LeBron<br>lives_in<br>part_of<br>sponsor_of<br>spouse_of<br>A. Davis Britney<br>lives_in<br>part_of located_in<br>support<br>lives_in<br>teammate_of<br>spouse_of<br>lives_in<br>**----- End of picture text -----**<br>


**Figure 1: An example of KG reasoning. The query** _**(Lebron, lives_in,?)**_ **asks which city** _**Lebron**_ **lives in. Dashed lines are irrelevant facts to this query while solid lines are relevant.** 

ignore the semantic relevance between local neighborhoods and the query relation. Intuitively, the local information we need depends on both the query entity and query relation. For example, the most relevant local evidence to the query in Figure 1 includes the combination of _(LeBron, part_of, Lakers)_ & _(Lakers, located_in, L.A.)_ and _(LeBron, spouse_of, Savan Nah)_ & _(Savan Nah, lives_in, L.A.)_ . If the query changes to _(LeBron, father_of, ?)_ , the most relevant information will be changed to _(LeBron JR, son_of, LeBron)_ . In addition, the existing GNN-based methods will inevitably involve too many irrelevant entities and facts, increasing the learning difficulty and computing cost, especially on large-scale KGs. 

Motivated by above examples, we propose to adaptively sample semantically relevant entities during propagation. There are three challenges when designing the sampling algorithm: (i) the structure of KG is complex and the size is large; (ii) the KG is multi-relational where the edges and queries have different kinds of relations, representing different semantic meanings; (iii) there is no direct supervision and simple heuristic measurement to indicate the semantic relevance of entities to the given query. To address these challenges, we propose AdaProp, a GNN-based method with an adaptive propagation path. The key idea is to reduce the number of involved entities while preserving the relevant ones in the propagation path. This is achieved by an incremental sampling mechanism and a learning-based sampling distribution that adapts to different queries. In this way, our method learn an adaptive propagation path that preserves layer-wise connection, selects semantic related entities, with fewer reduction in the promising targets. The main contributions are summarized as follows: 

- We propose a connection-preserving sampling scheme, called incremental sampling, which only has linear complexity with regard to the propagation steps and can preserve the layer-wise connections between sampled entities. 

- We design a semantic-aware Gumble top- _𝑘_ distribution which adaptively selects local neighborhoods relevant to the query relation and is learned by a straight through estimator. 

- Experiments show that the proposed method achieves the stateof-the-art performance in both transductive and inductive KG reasoning settings. It can preserve more target entities than 

the other samplers. And the case study shows that the learned sampler is query-dependent and semantic-aware. 

Let G = (V _,_ R _,_ E _,_ Q) be an instance of KG, where V, R are the sets of entities and relations, E = {( _𝑒𝑠,𝑟,𝑒𝑜_ ) | _𝑒𝑠,𝑒𝑜_ ∈V _,𝑟_ ∈R} is the set of fact triplets, and Q = {( _𝑒𝑞,𝑟𝑞,𝑒𝑎_ ) | _𝑒𝑞,𝑒𝑎_ ∈V _,𝑟_ ∈R} is the set of query triplets with[1] query ( _𝑒𝑞,𝑟𝑞,_ ?) and target answer entity _𝑒𝑎_ . The reasoning task is to learn a function that predicts the answer _𝑒𝑎_ ∈V for each query ( _𝑒𝑞,𝑟𝑞,_ ?) based on the fact triplets E. 

## **2 RELATED WORKS** 

## **2.1 GNN for KG reasoning** 

Due to the success of graph neural networks (GNNs) [14, 22] in modeling the graph-structured data, recent works [36, 38, 53, 56] attempt to utilize the power of GNNs for KG reasoning. They generally follow the message propagation framework [14, 22] to propagate messages among entities. In the _𝑙_ -th propagation step, specifically, the messages are firstly computed on edges E _[ℓ]_ = {( _𝑒𝑠,𝑟,𝑒𝑜_ ) ∈E| _𝑒𝑠_ ∈V _[ℓ]_[−][1] ) _,𝑟_ ∈R _,𝑒𝑜_ ∈V _[ℓ]_ } with message function _𝒎[ℓ]_ ( _𝑒𝑠 ,𝑟,𝑒𝑜_ )[:][=][ MESS] � _𝒉𝑒[ℓ] 𝑠_[−][1] _[,][ 𝒉] 𝑟[ℓ]_ � based on entity representation _𝒉𝑒[ℓ] 𝑠_[−][1][and] relation representation _𝒉𝑟[ℓ]_ . And then the messages are propagated from entities of previous step ( _𝑒𝑠_ ∈V _[ℓ]_[−][1] ) to entities of current step ( _𝑒𝑜_ ∈V _[ℓ]_ ) via _𝒉𝑒[ℓ] 𝑜_ := _𝛿_ �AGG[�] _𝒎[ℓ]_ ( _𝑒𝑠 ,𝑟,𝑒𝑜_ ) _[,]_[ (] _[𝑒][𝑠][,𝑟,𝑒][𝑜]_[) ∈E] _[ℓ]_[��] , with an activation function _𝛿_ (·). The specified functions MESS(·) and AGG(·) for different methods are provided in Appendix A.1. After _𝐿_ steps’ propagation, the entity representations _𝒉[𝐿] 𝑒𝑜_ are obtained to measure the plausibility of each entity _𝑒𝑜_ ∈V _[𝐿]_ . 

For the GNN-based KG reasoning methods, the sets of entities participated in propagation are different. The existing GNN-based methods can be generally classified into three classes based on their design of propagation range: 

- _Full_ propagation methods, e.g., R-GCN [32] and CompGCN [38], propagate among all the entities, i.e., V _[ℓ]_ = V. They are limited to small propagation steps due to the large memory cost and the over-smoothing problem of GNNs [28] over full neighbors. 

- _Progressive_ propagation methods (RED-GNN [53] and NBFNet [56]), propagate from the query entity _𝑒𝑞_ and progressively to the _𝐿_ - hop neighborhood of _𝑒𝑞_ , i.e., V[0] = { _𝑒𝑞_ } and V _[ℓ]_ =[�] _𝑒_ ∈V _[ℓ]_[−][1][ N(] _[𝑒]_[)][,] where N( _𝑒_ ) contains the 1-hop neighbors of entity _𝑒_ . 

- _Constrained_ propagation methods, e.g., GraIL [36] and CoMPILE [25], propagate within a constrained range, i.e., V _[ℓ]_ = V _𝑒[𝐿] 𝑞,𝑒𝑎_[,] where V _𝑒[𝐿] 𝑞,𝑒𝑎_[⊂V][is][the][enclosing][subgraph][for] _[𝑒] 𝑞_[and] _[𝑒] 𝑎_[.] Since V _𝑒[𝐿] 𝑞,𝑒𝑎_[are][different][for][different][pairs][of][(] _[𝑒] 𝑞[,𝑒] 𝑎_[)][,][these] methods are extremely expensive, especially on large-scale KGs (see discussion in Appendix A.2). 

We provide graphical illustration of propagation scheme for the three categories in the middle of Figure 2. Among them, the progressive propagation methods achieved state-of-the-art reasoning performance. They are much more efficient than the constrained propagation methods, and the entities used for propagation depend on the query entity, filtering out many irrelevant entities. However, V _[𝐿]_ of progressive methods will also involve in a majority or even all of the entities when _𝐿_ gets large. 

> 1Note that predicting missing head in KG can also be formulated in this way by adding inverse relations (details in Appendix B.1). 

AdaProp: Learning Adaptive Propagation for Graph Neural Network based Knowledge Graph Reasoning 

KDD ’23, August 6–10, 2023, Long Beach, CA, USA 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/kdd_2024_AdaProp_Learning_Adaptive_Propagation_for_Graph_Neural_Network_based_Knowledge_Graph_Reasoning_images/kdd_2024_AdaProp_Learning_Adaptive_Propagation_for_Graph_Neural_Network_based_Knowledge_Graph_Reasoning.pdf-0003-02.png)


**----- Start of picture text -----**<br>
Example of KG reasoning Full propagation Progressive propagation Constrained propagation Adaptive propagation<br>query:(②,r,?) answer:⑦ (R-GCN / CompGCN) (NBFNet / RED-GNN) (GraIL / CoMPILE) (AdaProp)<br>1 4 9 1 4 9 1 4 9 1 4 7 9 1 4 7 9<br>2 5 7 2 5 7 2 5 7 2 5 2 5<br>3 3<br>3 6 8 3 6 8 3 6 8 6 8 6 8<br>/ / involved entities in " [!] , " ["] , " [#] uninvolved entities<br>**----- End of picture text -----**<br>


**Figure 2: Symbolic illustrations of the KG in Figure 1 and the different designs of propagation path. For simplicity, we use circles to represent entities and ignore the types of relation. Specifically, (1) full propagation propagates over the full set of entities** V **; (2) progressive propagation starts from the query entity** _𝑒𝑞_ **and gradually propagates to its** _ℓ_ **-hop neighbors in the** _ℓ_ **-th step; (3) constrained propagation propagates within a constrained range** V _𝑒[𝐿] 𝑞,𝑒𝑎_ **with entities less than 2 distance away from both** _𝑒𝑞_ **and** _𝑒𝑎_ **; (4) adaptive propagation of AdaProp starts from** _𝑒𝑞_ **and adaptively selects the semantic-relevant entities.** 

## **2.2 Sampling methods for GNN** 

The sampling methods can control and select entities during propagation. There are several methods introduced to sample nodes on homogeneous graphs with the objective to improve the scalability of GNNs. They can be classified into three categories: (i) node-wise sampling methods, such as GraphSAGE [15] and PASS [48], sample _𝐾_ entities from the neighbor set N ( _𝑒_ ) for each entity _𝑒_ in each propagation step; (ii) layer-wise sampling methods, like FastGCN [5], Adaptive-GCN [17] and LADIES [57], sample at most _𝐾_ entities in each propagation step; (iii) subgraph sampling methods, such as Cluster-GCN [8], GraphSAINT [51] and ShadowGNN [50], directly extract the local subgraph around the queried entity. All these methods are designed for homogeneous graphs with the purpose of solving scalability issues. 

There are also a few sampling methods proposed for KG reasoning. RS-GCN [13] proposes a node-wise sampling method to select neighbors for R-GCN/CompGCN. DPMPN [45] contains a full propagation GNN to learn the global feature, and prunes another GNN for reasoning in a layer-wise sampling manner. These sampling approaches reduce the memory cost for propagation on larger KGs, but their empirical performance does not gain much compared with the base models. 

## **3 THE PROPOSED METHOD 3.1 Problem formulation** 

We denote a GNN model for KG reasoning as _𝐹_ ( _𝒘,_ G[�] _[𝐿]_ ) with model parameters _𝒘_ . G[�] _[𝐿]_ here is called the _propagation path_ , containing the sets of involved entities in each propagation step, i.e., G[�] _[𝐿]_ = {V[0] _,_ V[1] _, ...,_ V _[𝐿]_ } where V _[ℓ] , ℓ_ = 1 _. . .𝐿_ is the set of involved entities in the _ℓ_ -th propagation step. The representations of entities are propagated from V[0] to V _[𝐿]_ in G[�] _[𝐿]_ for multiple steps. 

The objective is to design better propagation path G[�] _[𝐿]_ such that the model with optimized parameters can correctly predict target answer entity _𝑒𝑎_ for each query ( _𝑒𝑞,𝑟𝑞,_ ?). The progressive propagation methods have different propagation path according to the query entities, and achieved state-of-the-art performance. However, they ignore the dependency of query relations and will involve in too many entities at larger propagation steps. Motivated by the success of progressive propagation methods and sampling methods on homogeneous graph, we aim to use sampling technique to dynamically adapt the propagation paths. Formally, we define 

the _query-dependent propagation path_ for a query ( _𝑒𝑞,𝑟𝑞,_ ?) as 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/kdd_2024_AdaProp_Learning_Adaptive_Propagation_for_Graph_Neural_Network_based_Knowledge_Graph_Reasoning_images/kdd_2024_AdaProp_Learning_Adaptive_Propagation_for_Graph_Neural_Network_based_Knowledge_Graph_Reasoning.pdf-0003-11.png)


The propagation here starts from the query entity _𝑒𝑞_ , and the entities in V _𝑒[ℓ] 𝑞,𝑟𝑞_ is adaptively sampled as a subset of V _𝑒[ℓ] 𝑞_[−] _,𝑟_[1] _𝑞_[and its] neighbors with a sampling strategy _𝑆_ (·). 

The key problem here is how to design the sampling strategy _𝑆_ (·). For the KG reasoning task, we face two main challenges: 

- the target answer entity _𝑒𝑎_ is unknown given the query ( _𝑒𝑞,𝑟𝑞,_ ?), thus directly sampling neighboring entities may loss the connection between query entity and target answer entity; 

- the semantic dependency between entities in the propagation path and the query relation _𝑟𝑞_ is too complex to be captured by simple heuristics. 

The existing sampling methods are not very applicable since (i) they do not consider the preserving of unknown target entities; (ii) they are designed for homogeneous graph without modeling relation types; and (iii) there is no direct supervision on the dependency of entities in the propagation on the query. Therefore, we would argue that a specially designed sampling approach is needed for KG reasoning, which is expected to remove entities and also preserving the promising targets for corresponding query. 

In the following parts, we introduce the adaptive propagation (AdaProp) algorithm, which can adaptively select relevant entities into the propagation path based on the given query. To solve the first challenge, we design a connection-preserving sampling scheme, called incremental sampling, which can reduce the number of involved entities and preserve the layer-wise connections in Section 3.2. For the second challenge, we propose a relationdependent sampling distribution, which is jointly optimized with the model parameters, to select entities that are semantically relevant to the query relation in Section 3.3. The full algorithm is summarized in Section 3.4. 

## **3.2 The connection-preserving incremental sampling strategy** 

When designing the sampling strategy, we should consider what to be sampled, and how to sample effectively and efficiently. Since the target entity _𝑒𝑎_ is unknown given the query ( _𝑒𝑞,𝑟𝑞,_ ?), freely sampling from all the possible neighbors will loss the structural 

KDD ’23, August 6–10, 2023, Long Beach, CA, USA 

Yongqi Zhang, Zhanke Zhou, Quanming Yao, Xiaowen Chu, and Bo Han 

connection between query entity and promising targets. The existing sampling methods cannot well address this problem. For the node-wise sampling methods, the complexity grows exponentially w.r.t. propagation steps _𝐿_ . For the layer-wise sampling methods, the entities in previous step may have little connection with the sampled entities in the current step, failing to preserve the local structures. The subgraph sampling methods can better preserve local structure, but is harder to control which entities are more relevant to the query and deserve to be sampled. 

Observing that the majority of the target answer entities lie close to the query entity (see the distributions in Appendix C.1), we are motivated to preserve the entities already been selected in each propagation step. In other words, we use the constraint V _𝑒_[0] _𝑞,𝑟𝑞_ ⊆V _𝑒_[1] _𝑞,𝑟𝑞_ · · · ⊆V _𝑒[𝐿] 𝑞,𝑟𝑞_[to][preserve][the][previous][step][entities] and sample from the newly-visited ones in an incremental manner to avoid dropping the promising targets. On one hand, the nearby targets will be more likely to be preserved when the propagation steps get deeper. On the other hand, the layer-wise connection between entities in two consecutive layers can be preserved. The incremental sampler _𝑆_ (·) = SAMP(CAND(·)) contains two parts, i.e., candidate generation CAND(·) and candidate sampling SAMP(·). 

_3.2.1 Candidate generation._ Given the set of entities in the ( _ℓ_ −1)th step V _𝑒[ℓ] 𝑞_[−] _,𝑟_[1] _𝑞_[, we denote the candidate entities to sample in the] ~~_ℓ_~~ _ℓ_ -th step as V _𝑒𝑞,𝑟𝑞_[:][=][ CAND][(V] _𝑒[ℓ] 𝑞_[−] _,𝑟_[1] _𝑞_[)][. Basically, all the neighboring] entities in N(V _𝑒[ℓ] 𝑞_[−] _,𝑟_[1] _𝑞_[)][ =][�] _𝑒_ ∈V _𝑒𝑞,𝑟𝑞[ℓ]_[−][1][N(] _[𝑒]_[)][ can be regarded as candidates] for sampling. However, to guarantee that V _𝑒[ℓ] 𝑞_[−] _,𝑟_[1] _𝑞_[will be preserved] in the next sampling step, we regard the newly-visited entities as the candidates, i.e., 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/kdd_2024_AdaProp_Learning_Adaptive_Propagation_for_Graph_Neural_Network_based_Knowledge_Graph_Reasoning_images/kdd_2024_AdaProp_Learning_Adaptive_Propagation_for_Graph_Neural_Network_based_Knowledge_Graph_Reasoning.pdf-0004-05.png)


For example, given the entity _○_ 2 in Figure 2, we generate the 1-hop neighbors _○_ 1 , _○_ 3 , _○_ 4 , _○_ 5 , _○_ 6 as candidates and directly preserve _○_ 2 in the next propagation step. 

_3.2.2 Candidate sampling._ To control the number of involved ~~_ℓ_~~ entities, we sample _𝐾_ (≪| V _𝑒𝑞,𝑟𝑞_[|][) entities without replacement] from the candidate set V _𝑒_ ~~_ℓ_~~ _𝑞,𝑟𝑞_[.][The][sampled][entities][at] _[ℓ]_[-th][step] together with entities in the ( _ℓ_ −1)-th step constitute the involved entities at the _ℓ_ step, i.e., 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/kdd_2024_AdaProp_Learning_Adaptive_Propagation_for_Graph_Neural_Network_based_Knowledge_Graph_Reasoning_images/kdd_2024_AdaProp_Learning_Adaptive_Propagation_for_Graph_Neural_Network_based_Knowledge_Graph_Reasoning.pdf-0004-08.png)


Given the candidates of entity _○_ 2 in Figure 2, the entities _○_ 5 and _○_ 6 ~~_ℓ_~~ are sampled and activated from the five candidates in V _𝑒𝑞,𝑟𝑞_[. Along] with the previous layer entity _○_ 2 , the three entities are collected as V _𝑒[ℓ] 𝑞,𝑟𝑞_ for next step propagation. 

_3.2.3 Discussion._ There are three advantages of the incremental sampling strategy. First, it is more entity-efficient than the nodewise sampling methods, which have exponentially growing number of entities over propagation steps. As in Prop.1, the number of involved entities grows linearly w.r.t. the propagation steps _𝐿_ . Second, the layer-wise connection can be preserved based on Prop.2, which shows advantage over the layer-wise sampling methods. Third, as will be shown in Section 4.2.1, it has larger probability to preserve the target answer entities than the other sampling mechanisms with the same amount of involved entities. 

Proposition 1. _The number of involved entities in the propagation path (_ ��� _ℓ_ =0 _...𝐿_[V] _𝑒[ℓ] 𝑞,𝑟𝑞_ �� _) of incremental sampling is bounded by 𝑂_ ( _𝐿𝐾_ ) _._ 

Proposition 2. _For all the entities 𝑒_ ∈V _𝑒[ℓ] 𝑞_[−] _,𝑟_[1] _𝑞[with incremental] sampling, there exists at least one entity 𝑒_[′] ∈V _𝑒[ℓ] 𝑞,𝑟𝑞 and relation 𝑟_ ∈R _such that_ ( _𝑒,𝑟,𝑒_[′] ) ∈E _._ 

## **3.3 Learning semantic-aware distribution** 

The incremental sampling mechanism alleviates the problem of reducing target candidates. However, randomly sampling _𝐾_ entities from V _𝑒_ ~~_ℓ_~~ _𝑞,𝑟𝑞_[does not consider the semantic relevance of different] entities for different queries. The greedy-based search algorithms like beam search [42] are not applicable here as there is no direct heuristic to measure the semantic relevance. To solve this problem, we parameterize the sampler _𝑆_ (·) with parameter _𝜽[ℓ]_ for each propagation step _ℓ_ , i.e., V _𝑒[ℓ] 𝑞,𝑟𝑞_ = _𝑆_ (V _𝑒[ℓ] 𝑞_[−] _,𝑟_[1] _𝑞_[;] _[𝜽][ℓ]_[)][. In the following two] parts, we introduce the sampling distribution and how to optimize the sampling parameters. 

_3.3.1 Parameterized sampling distribution._ For the GNN-based KG reasoning methods, the entity representations _𝒉[𝐿] 𝑒𝑜_ in the final propagation step are used to measure the plausibility of entity _𝑒𝑜_ being the target answer entity. In other words, the entity representations are learned to indicate their relevance to the given query ( _𝑒𝑞,𝑟𝑞,_ ?). To share this knowledge, we introduce a linear mapping function _𝑔_[�] _𝒉𝑒[ℓ] 𝑜_ ; _𝜽[ℓ]_[�] = _𝜽[ℓ]_[⊤] _𝒉𝑒[ℓ] 𝑜_ with parameters _𝜽[ℓ]_ ∈ R _[𝑑]_ in each layer _ℓ_ = 1 _. . . 𝐿_ , and sample according to the probability distribution 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/kdd_2024_AdaProp_Learning_Adaptive_Propagation_for_Graph_Neural_Network_based_Knowledge_Graph_Reasoning_images/kdd_2024_AdaProp_Learning_Adaptive_Propagation_for_Graph_Neural_Network_based_Knowledge_Graph_Reasoning.pdf-0004-16.png)


with temperature value _𝜏 >_ 0. 

Sampling _𝐾_ entities without replacement from _𝑝[ℓ]_ ( _𝑒_ ) requires to sequentially sample and update the distribution for each sample, which can be expensive. We follow the Gumbel top-k trick [24, 43] to solve this issue. The Gumbel-trick firstly samples ~~�~~ �V _𝑒_ ~~_ℓ_~~ _𝑞,𝑟𝑞_ �� independent noises from the uniform distribution, i.e., _𝑈𝑒_ ∼ Uniform(0 _,_ 1) to form the Gumbel logits _𝐺𝑒_ = _𝑔_ ( _𝒉𝑒[ℓ]_ ; _𝜽[ℓ]_ ) − ~~_ℓ ℓ_~~ log(− log _𝑈𝑒_ ) for each _𝑒_ ∈ V _𝑒𝑞,𝑟𝑞_[.][The][top-] _[𝐾]_[entities][in] V _𝑒𝑞,𝑟𝑞_ are then collected based on their values of Gumbel logits _𝐺𝑒_ . As proved in [24, 43], this procedure is equivalent to sample without replacement from _𝑝[ℓ]_ ( _𝑒_ ). 

_3.3.2 Learning strategy._ We denote _𝜽_ = { _𝜽_[1] _, . . . , 𝜽[𝐿]_ } as the sampling parameters and the parameterized propagation path as G[�] _𝑒[𝐿] 𝑞,𝑟𝑞_ ( _𝜽_ ). Since our sampling distribution in (2) is strongly correlated with the GNN representations, jointly optimizing both the model parameters and sampler parameters can better share knowledge between the two parts. Specifically, we design the following objective 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/kdd_2024_AdaProp_Learning_Adaptive_Propagation_for_Graph_Neural_Network_based_Knowledge_Graph_Reasoning_images/kdd_2024_AdaProp_Learning_Adaptive_Propagation_for_Graph_Neural_Network_based_Knowledge_Graph_Reasoning.pdf-0004-20.png)


The loss function L on each instance is a binary cross entropy loss on all the entities _𝑒𝑜_ ∈V _𝑒[𝐿] 𝑞,𝑟𝑎_[, i.e.,] 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/kdd_2024_AdaProp_Learning_Adaptive_Propagation_for_Graph_Neural_Network_based_Knowledge_Graph_Reasoning_images/kdd_2024_AdaProp_Learning_Adaptive_Propagation_for_Graph_Neural_Network_based_Knowledge_Graph_Reasoning.pdf-0004-22.png)


KDD ’23, August 6–10, 2023, Long Beach, CA, USA 

AdaProp: Learning Adaptive Propagation for Graph Neural Network based Knowledge Graph Reasoning 

**Algorithm 1** AdaProp: learning adaptive propagation path. 

- **Require:** query ( _𝑒𝑞,𝑟𝑞,_ ?), V _𝑒_[0] _𝑞,𝑟𝑞_ = { _𝑒𝑞_ }, steps _𝐿_ , number of sampled entities _𝐾_ , and functions MESS(·) and AGG(·). 

- 1: **for** _ℓ_ = 1 _. . . 𝐿_ **do** 2: get the neighboring entities N(V _𝑒[ℓ] 𝑞_[−] _,𝑟_[1] _𝑞_[)][=][ ∪] _𝑒_ ∈V _𝑒𝑞,𝑟𝑞[ℓ]_[−][1][N(] _[𝑒]_[)][,] ~~_ℓ_~~ 

- the newly-visited entities V _𝑒𝑞,𝑟𝑞_[=][N(V] _𝑒[ℓ] 𝑞_[−] _,𝑟_[1] _𝑞_[)][\ V] _𝑒[ℓ] 𝑞_[−] _,𝑟_[1] _𝑞_[, and] edges E _[ℓ]_ = {( _𝑒𝑠,𝑟,𝑒𝑜_ )| _𝑒𝑠_ ∈V _𝑒[ℓ] 𝑞_[−] _,𝑟_[1] _𝑞[,𝑒][𝑜]_[∈N(V] _𝑒[ℓ] 𝑞_[−] _,𝑟_[1] _𝑞_[)}][;] 

- 3: obtain _𝒎[ℓ]_ ( _𝑒𝑠 ,𝑟,𝑒𝑜_ ) := MESS( _𝒉𝑒[ℓ] 𝑠_[−][1] _[,][ 𝒉] 𝑒[ℓ] 𝑜_[−][1] _[,][ 𝒉] 𝑟[ℓ][,][ 𝒉] 𝑟[ℓ] 𝑞_[)] for edges ( _𝑒𝑠,𝑟,𝑒𝑜_ ) ∈E _[ℓ]_ ; 

- 4: obtain _𝒉𝑒[ℓ] 𝑜_ := _𝛿_[�] AGG[�] _𝒎[ℓ]_ ( _𝑒𝑠 ,𝑟,𝑒𝑜_ ) _[,]_[ (] _[𝑒][𝑠][,𝑟,𝑒][𝑜]_[) ∈E] _[ℓ]_[��][for entities] _[ 𝑒][𝑜]_[∈] N(V _𝑒[ℓ] 𝑞_[−] _,𝑟_[1] _𝑞_[)][;] 

- 5: **logits computation:** obtain the Gumbel logits _𝐺𝑒𝑜_ = _𝑔_ ( _𝒉𝑒[ℓ] 𝑜_ ; _𝜽[ℓ]_ ) − ~~_ℓ_~~ 

- log(−log _𝑈𝑒𝑜_ ) with _𝑈𝑒𝑜_ ∼ Uniform(0 _,_ 1) for entities _𝑒𝑜_ ∈�V _𝑒𝑞,𝑟𝑞_[;] 

- 6: **candidate sampling:** obtain sampled entities V _𝑒[ℓ] 𝑞,𝑟𝑞_ = ~~_ℓ_~~ 

- {arg top _𝐾 𝐺𝑒𝑜 ,𝑒𝑜_ ∈ V _𝑒𝑞,𝑟𝑞_[}][;] 

- 7: **straight-through:** _𝒉𝑒[ℓ]_ :=[�] 1 − no_grad( _𝑝[ℓ]_ ( _𝑒_ )) + _𝑝[ℓ]_ ( _𝑒_ )[�] · _𝒉𝑒[ℓ]_ for entities _𝑒_ ∈ V[�] _𝑒[ℓ] 𝑞,𝑟𝑞_ ; 

- 8: **update propagation path:** update V _𝑒[ℓ] 𝑞,𝑟𝑞_ = V _𝑒[ℓ] 𝑞_[−] _,𝑟_[1] _𝑞_[∪] V[�] _𝑒[ℓ] 𝑞,𝑟𝑞_ ; 9: **end for** 

- 10: **return** _𝑓_[�] _𝒉[𝐿] 𝑒𝑜_ ; _𝒘_[⊤][�] for each _𝑒𝑜_ ∈V _𝑒[𝐿] 𝑞,𝑟𝑞_[.] 

where the likelihood score _𝜙𝑒𝑜_ = _𝑓_[�] _𝒉[𝐿] 𝑒𝑜_ ; _𝒘𝜙_[⊤] � ∈[0 _,_ 1] for entity _𝑒𝑜_ indicating the plausibility of _𝑒𝑜_ being the target answer entity, and the label _𝑦𝑒𝑜_ = 1 if _𝑒𝑜_ = _𝑒𝑎_ otherwise 0. 

The reparameterization trick is often used for Gumbel distributions [19]. However, it does not do explicit sampling, thus still requires high computing cost. Instead, we choose the straightthrough (ST) estimator [2, 19], which can approximately estimate gradient for discrete variables. The key idea of ST-estimator is to back-propagate through the sampling signal as if it was the identity function. Specifically, rather than directly use the entity representations _𝒉𝑒[ℓ]_ to compute messages, we use _𝒉𝑒[ℓ]_ :=[�] 1 − no_grad( _𝑝[ℓ]_ ( _𝑒_ )) + _𝑝[ℓ]_ ( _𝑒_ )[�] · _𝒉𝑒[ℓ]_ , where no_grad( _𝑝[ℓ]_ ( _𝑒_ )) means that the back-propagation signals will not go through this term. In this approach, the forward computation will not be influenced, while the backward signals can go through the probability multiplier _𝑝[ℓ]_ ( _𝑒_ ). An alternative choice to estimate the gradient on discrete sampling distribution is the REINFORCE technique [27, 41]. However, REINFORCE gradient is known to have high variance [35] and the variance will be accumulated across the propagation steps, thus not used here. The model parameters _𝒘_ and sampler parameters _𝜽_ are simultaneously updated by Adam optimizer [21]. 

Overall, the objective of learning semantic-aware distribution is to adaptively preserve the promising targets according to the query relation. As will be shown in Section 4.3. the selected information in the sampled propagation path are semantically related to the query relation by sharing knowledge with the entity representations. On the other hand, the learned distribution guarantees a higher coverage of target entities compared with the not learned version. 

## **3.4 The full algorithm** 

The full procedure of AdaProp is shown in Algorithm 1. Given the entities V _𝑒[ℓ] 𝑞_[−] _,𝑟_[1] _𝑞_[in the][ (] _[ℓ]_[−][1][)][-th step, we first obtain the neighboring] entities N(V _𝑒[ℓ] 𝑞_[−] _,𝑟_[1] _𝑞_[)][of][V] _𝑒[ℓ] 𝑞_[−] _,𝑟_[1] _𝑞_[, and generate the newly-visited ones] 

~~_ℓ_~~ V _𝑒𝑞,𝑟𝑞_[=][N(V] _[ℓ]_[−][1][)][\][V] _𝑒[ℓ] 𝑞_[−] _,𝑟_[1] _𝑞_[as][candidates][in][line][2.][Then,][query-] dependent messages are computed on edges E _[ℓ]_ in line 3, and propagated to entities in N(V _𝑒[ℓ] 𝑞_[−] _,𝑟_[1] _𝑞_[)][in][line][4.][The][top-] _[𝐾]_[entities] in V _𝑒_ ~~_ℓ_~~ _𝑞,𝑟𝑞_[are][sampled][in][line][6][according][to][the][Gumbel][logits] computed in line 5. As for the ST-estimator, we modify the hidden representations for entities in V[�] _𝑒[ℓ] 𝑞,𝑟𝑞_ in line 7. Finally, the set of sampled entities V[�] _𝑒[ℓ] 𝑞,𝑟𝑞_ are concatenated with V _𝑒[ℓ] 𝑞_[−] _,𝑟_[1] _𝑞_[to form the set] of _ℓ_ -th step entities V _𝑒[ℓ] 𝑞,𝑟𝑞_ in line 8. After propagating for _𝐿_ steps, the final step representations _𝒉[𝐿] 𝑒𝑜_ for all the involved entities are returned. The entity with the highest score evaluated by _𝑓_ ( _𝒉[𝐿] 𝑒𝑜_ ; _𝒘𝜙_ ) in V _𝑒[𝐿] 𝑞,𝑟𝑞_[is regarded as the predicted target.] 

The key difference compared with existing GNN-based KG reasoning methods is that the propagation path in Algorithm 1 is no longer given and fixed before the message propagation. Instead, it is adaptively adjusted based on the entity representations and is dynamically updated in each propagation step. In addition, Algorithm 1 also enjoys higher efficiency than the full propagation and progressive propagation methods, since only a small portion of entities are involved, while the extra computation cost for sampling is relatively low. Compared with constrained propagation methods, which needs to propagate in the answer-specific propagation paths for multiple times, the potential target answer entities are directly scored in the final propagation step in a single forward pass. 

## **4 EXPERIMENTS** 

In this section, we empirically verify the effectiveness and analyze the components of the proposed AdaProp on both transductive and inductive settings. All the experiments are implemented in Python with PyTorch [29] and run on a single NVIDIA RTX 3090 GPU with 24GB memory. 

## **4.1 Comparison with KG reasoning methods** 

We compare AdaProp with general KG reasoning methods in both transductive and inductive reasoning. In the transductive setting, the entities sets are the same for training and testing. While in the testing of inductive reasoning, the model are required to generalize to unseen entities w.r.t. the training set. Besides, we follow [3, 36, 39, 53, 56] to use the filtered ranking-based metrics for evaluation, i.e., mean reciprocal ranking (MRR) and Hit@ _𝑘_ (e.g., Hit@1 and Hit@10). The higher value indicates the better performance for both metrics. As for the hyper-parameters, we tune the number of propagation steps _𝐿_ from 5 to 8, the number of sampled entities _𝐾_ in {100 _,_ 200 _,_ 500 _,_ 1000 _,_ 2000}, and the temperature value _𝜏_ in {0 _._ 5 _,_ 1 _._ 0 _,_ 2 _._ 0}. The ranges of other hyperparameters are kept the same as RED-GNN [53]. 

## _4.1.1 Transductive setting._ 

Figure 1 shows an example of transductive KG reasoning, where a KG related to _Lakers_ is given and the task is to predict where _LeBron_ lives in based on the given facts. Formally speaking, we have one KG G = (V _,_ R _,_ E _,_ Q) where the query set Q is split into three disjoint sets Qtra/Qval/Qtst for training, validation and testing respectively. **Datasets.** We use six widely used KG completion datasets, including Family [23], UMLS [23], WN18RR [10], FB15k237 [37], NELL995 [44] and YAGO3-10 [33]. The statistics of these datasets are provided in Table 1. 

KDD ’23, August 6–10, 2023, Long Beach, CA, USA 

Yongqi Zhang, Zhanke Zhou, Quanming Yao, Xiaowen Chu, and Bo Han 

**Table 1: Statistics of the transductive KG datasets. Fact triplets in** E **are used to build the graph, and** Q **tra,** Q **val,** Q **tst are the query triplets used for reasoning.** 

|dataset|#entity #relation||E|<br>|Qtra|<br>|Qval| |Qtst||
|---|---|---|
|Family<br>UMLS<br>WN18RR<br>FB15k237<br>NELL-995<br>YAGO3-10|3.0k<br>12<br>135<br>46<br>40.9k<br>11<br>14.5k<br>237<br>74.5k<br>200<br>123.1k<br>37|23.4k<br>5.9k<br>2.0k<br>2,8k<br>5.3k<br>1.3k<br>569<br>633<br>65.1k<br>21.7k<br>3.0k<br>3.1k<br>204.1k<br>68.0k<br>17.5k<br>20.4k<br>112.2k<br>37.4k<br>543<br>2.8k<br>809.2k 269.7k<br>5.0k<br>5.0k|



**Baselines.** We compare the proposed AdaProp with (i) non-GNN methods ConvE [10], QuatE [52], RotatE [34], MINERVA [9], DRUM [31], RNNLogic [30] and RLogic [7]; and (ii) GNN-based full propagation method CompGCN [38] and progressive propagation methods NBFNet [56] and RED-GNN [53] here. R-GCN [32] is not compared here as it is much inferior to CompGCN [38]. As indicated by [53, 56], GraIL [36] and CoMPILE [25] are intractable to work on the transductive KGs with many entities, and thus not compared in this setting. Results of these baselines are taken from their papers or reproduced by their official codes. Missing results of RLogic are due to the lack of source code. 

**Results.** The comparison of AdaProp with the transductive reasoning methods is in Table 2. First, The GNN-based methods generally perform better than the non-GNN based ones by capturing both the structural and semantic information in KGs. Second, the progressive propagating methods, i.e., NBFNet and REDGNN, perform better than the full propagation method CompGCN, especially on larger KGs, like WN18RR, FB15k237, NELL-995 and YAGO3-10. In comparison, AdaProp achieves leading performance on WN18RR, NELL-995 and YAGO3-10 over all the baselines and slightly outperforms NBFNet on FB15k237. Besides, AdaProp also works well with competitive performances on smaller KGs Family and UMLS. To further investigate the superiority of propagation path learned by AdaProp, we analyze the properties in the following content. 

**Property of propagation path.** To quantitatively analyze the property of propagation path, we introduce the following metric, i.e., the ratio of target over entities (ToE), to indicates the property of G[�] _𝑒[𝐿] 𝑞,𝑟𝑞_ with propagation steps _𝐿_ : 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/kdd_2024_AdaProp_Learning_Adaptive_Propagation_for_Graph_Neural_Network_based_Knowledge_Graph_Reasoning_images/kdd_2024_AdaProp_Learning_Adaptive_Propagation_for_Graph_Neural_Network_based_Knowledge_Graph_Reasoning.pdf-0006-07.png)


where the nominator TC( _𝐿_ ) = | Q1tst | �( _𝑒𝑞,𝑟𝑞,𝑒𝑎_ )∈Qtst[I] � _𝑒𝑎_ ∈V _𝑒[𝐿] 𝑞,𝑟𝑞_ � measures the ratio of _target entity coverage_ by the propagation path, and the denominator EI( _𝐿_ ) = | Q1tst | �( _𝑒𝑞,𝑟𝑞,𝑒𝑎_ )∈Qtst ��� _ℓ_ =1 _...𝐿_[V] _𝑒[ℓ] 𝑞,𝑟𝑞_ �� measures the number of _entities involved_ in the propagation path. Qtst contains queries in the testing set, V _𝑒[ℓ] 𝑞,𝑟𝑞_ is the set of _ℓ_ -th step entities for query ( _𝑒𝑞,𝑟𝑞,_ ?), and I{ _𝑒𝑎_ ∈V _𝑒[𝐿] 𝑞,𝑟𝑞_ } = 1 if _𝑒𝑎_ is in V _𝑒[𝐿] 𝑞,𝑟𝑞_ otherwise 0. 

We compare the value ToE( _𝐿_ ) and the MRR performance of different GNN-based reasoning methods with different number of propagation steps _𝐿_ in Figure 3. WN18RR and NELL-995 are used here. As shown, CompGCN quickly decreases on WN18RR and does not gain on NELL-995 when _𝐿 >_ 2 since its ToE( _𝐿_ ) is fixed and very small. It will easily suffer from the problem 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/kdd_2024_AdaProp_Learning_Adaptive_Propagation_for_Graph_Neural_Network_based_Knowledge_Graph_Reasoning_images/kdd_2024_AdaProp_Learning_Adaptive_Propagation_for_Graph_Neural_Network_based_Knowledge_Graph_Reasoning.pdf-0006-10.png)


**----- Start of picture text -----**<br>
WN18RR NELL-995<br>2 2<br>CompGCN CompGCN<br>NBFNet/RED-GNN NBFNet/RED-GNN<br>AdaProp AdaProp<br>3 3<br>4 4<br>5 5<br>2 3 4 5 6 7 8 2 3 4 5 6 7 8<br>Propagation step Propagation step<br>(a) ToE( 𝐿 )<br>WN18RR NELL-995<br>0.6 0.6<br>0.5 0.5<br>0.4 CompGCN 0.4 CompGCN<br>RED-GNN RED-GNN<br>NBFNet NBFNet<br>AdaProp AdaProp<br>0.3 0.3<br>2 3 4 5 6 7 8 2 3 4 5 6 7 8<br>Propagation step Propagation step<br>(b) Testing performance<br>log-scale ToE(L) log-scale ToE(L)<br>MRR MRR<br>**----- End of picture text -----**<br>


**Figure 3: Comparison of GNN-based methods w.r.t.** _𝐿_ **.** 

of over-smoothing [28] and quickly run out-of-memory when _𝐿_ increases. By contrast, RED-GNN and NBFNet have increasing performance from _𝐿_ = 2 to _𝐿_ = 5 by capturing the more long-range information. But they get stuck in _𝐿 >_ 5 when too many entities are involved. Besides, RED-GNN and NBFNet, with the same design of propagation path, have similar performance curves here. As for AdaProp, the performance consistently gets improved with larger ToE( _𝐿_ ) in deeper propagation steps. The benefits of a deeper propagation path can be attributed to both a larger propagation step and a deeper GNN model (we provide more details in Appendix C.6). AdaProp, with larger ToE( _𝐿_ ) at larger _𝐿_ , alleviates the problem of over-smoothing compared with other methods. 

## _4.1.2 Inductive setting._ 

The inductive setting cares about reasoning on unseen entities. If the KG related to _Lakers_ in Figure 1 is used for training, then the testing can be reason on another KG related to _Warriors_ and a query _(Curry, lives_in,?)_ with unseen entity _Curry_ . Formally speaking, we have two KGs G1 = (V1 _,_ R _,_ E1 _,_ Q1) and G2 = (V2 _,_ R _,_ E2 _,_ Q2). G1 is used for training and validation by splitting Q1 into two disjoint sets Qtra and Qval, and the unseen KG G2 is used for testing with Qtst := Q2. The relation sets of the two KGs are the same, while the entity sets are disjoint, i.e., V1 ∩V2 = ∅. 

**Datasets.** Following [36, 53], we use the same subsets (4 versions each and 12 subsets in total) of WN18RR, FB15k237 and NELL-995, where each subset has a different split of training set and test set. We refer the readers to [36, 53] for the detailed splits and statistics. **Baselines.** All the reasoning methods which learn entity embeddings in training (ConvE [11], QuatE [52], RotatE [34], MINERVA [9], CompGCN [38]) cannot work in this setting. Hence, we compare AdaProp with non-GNN methods that learn rules without entity embeddings, i.e., RuleN [26], NeuralLP [46] and DRUM [31]. 

KDD ’23, August 6–10, 2023, Long Beach, CA, USA 

AdaProp: Learning Adaptive Propagation for Graph Neural Network based Knowledge Graph Reasoning 

**Table 2: Transductive setting. Best performance is indicated by the bold face numbers, and the underline means the second best. “–” means unavailable results. “H@1” and “H@10” are short for Hit@1 and Hit@10 (in percentage), respectively.** 

|type|models|Family<br>MRR H@1 H@10|UMLS<br>MRR H@1 H@10|WN18RR<br>MRR H@1 H@10<br>MR|FB15k237<br>R H@1 H@10|NELL-995<br>MRR H@1 H@1|YAGO3-10<br>0<br>MRR H@1 H@10|
|---|---|---|---|---|---|---|---|
|non-GNN|ConvE<br>QuatE<br>RotatE<br>MINERVA<br>DRUM<br>RNNLogic<br>RLogic|0.912<br>83.7<br>98.2<br>0.941<br>89.6<br>99.1<br>0.921<br>86.6<br>98.8<br>0.885<br>82.5<br>96.1<br>0.934<br>88.1<br>99.6<br>0.881<br>85.7<br>90.7<br>–<br>–<br>–|0.937<br>92.2<br>96.7<br>0.944<br>90.5<br>99.3<br>0.925<br>86.3<br>99.3<br>0.825<br>72.8<br>96.8<br>0.813<br>67.4<br>97.6<br>0.842<br>77.2<br>96.5<br>–<br>–<br>–|0.427<br>39.2<br>49.8<br>0.3<br>0.480<br>44.0<br>55.1<br>0.3<br>0.477<br>42.8<br>57.1<br>0.3<br>0.448<br>41.3<br>51.3<br>0.2<br>0.486<br>42.5<br>58.6<br>0.3<br>0.483<br>44.6<br>55.8<br>0.3<br>0.47<br>44.3<br>53.7<br>0.3|25<br>23.7<br>50.1<br>50<br>25.6<br>53.8<br>37<br>24.1<br>53.3<br>93<br>21.7<br>45.6<br>43<br>25.5<br>51.6<br>44<br>25.2<br>53.0<br>1<br>20.3<br>50.1|0.511<br>44.6<br>61.9<br>0.533<br>46.6<br>64.3<br>0.508<br>44.8<br>60.8<br>0.513<br>41.3<br>63.7<br>0.532<br>46.0<br>66.2<br>0.416<br>36.3<br>47.8<br>–<br>–<br>–|0.520<br>45.0<br>66.0<br><br>0.379<br>30.1<br>53.4<br><br>0.495<br>40.2<br>67.0<br><br>–<br>–<br>–<br><br>0.531<br>45.3<br>67.6<br><br>0.554<br>50.9<br>62.2<br>0.36<br>25.2<br>50.4|
|GNNs|CompGCN<br>NBFNet<br>RED-GNN|0.933<br>88.3<br>99.1<br>0.989<br>**98.8**<br>98.9<br>**0.992 98.8**<br>**99.7**|0.927<br>86.7<br>99.4<br>0.948<br>92.0<br>**99.5**<br>0.964<br>94.6<br>99.0|0.479<br>44.3<br>54.6<br>0.3<br>0.551<br>49.7<br>66.6<br>0.4<br>0.533<br>48.5<br>62.4<br>0.3|55<br>26.4<br>53.5<br>15<br>32.1<br>**59.9**<br>74<br>28.3<br>55.8|0.463<br>38.3<br>59.6<br>0.525<br>45.1<br>63.9<br>0.543<br>47.6<br>65.1|0.421<br>39.2<br>57.7<br><br>0.550<br>47.9<br>68.6<br><br>0.559<br>48.3<br>68.9|
||**AdaProp**|0.988<br>98.6<br>99.0|**0.969 95.6**<br>**99.5**|**0.562 49.9**<br>**67.1**<br>**0.4**|**17 33.1**<br>58.5|**0.554 49.3**<br>**65.**|**5**<br>**0.573 51.0**<br>**68.5**|



**Table 3: Inductive setting (evaluated with Hit@10). Full evaluation results of MRR and Hit@1 are in Appendix C.3.** 

|metric|methods|WN18RR<br>V1<br>V2<br>V3<br>V4|FB15k237<br>V1<br>V2<br>V3<br>V4|
|---|---|---|---|
|Hit@10 (%)|RuleN<br>Neural LP<br>DRUM|73.0<br>69.4<br>40.7<br>68.1<br>77.2<br>74.9<br>47.6<br>70.6<br>77.7<br>74.7<br>47.7<br>70.2|44.6<br>59.9<br>60.0<br>60.5<br>46.8<br>58.6<br>57.1<br>59.3<br>47.4<br>59.5<br>57.1<br>59.3|
||GraIL<br>CoMPILE<br>NBFNet<br>RED-GNN|76.0<br>77.6<br>40.9<br>68.7<br>74.7<br>74.3<br>40.6<br>67.0<br>82.7<br>79.9<br>56.3<br>70.2<br>79.9<br>78.0<br>52.4<br>72.1|42.9<br>42.4<br>42.4<br>38.9<br>43.9<br>45.7<br>44.9<br>35.8<br>51.7<br>63.9<br>58.8<br>55.9<br>48.3<br>62.9<br>60.3<br>62.1|
||**AdaProp**|**86.6**<br>**83.6**<br>**62.6**<br>**75.5**|**55.1**<br>**65.9**<br>**63.7**<br>**63.8**|



For GNN-based methods, we take GraIL [36], CoMPILE [25], REDGNN [53] and NBFNet [56] as baselines. RNNLogic [30] and RLogic [7] are not compared here since there lack source codes and reported results in this setting. We follow [53] to evaluate the ranking of target answer entity over all the negative entities, rather than 50 randomly sampled negative ones adopted in [36]. 

**Results.** As shown in Table 3, the constrained propagation methods GraIL and CoMPILE, even though have the inductive reasoning ability, are much worse than the progressive propagation methods RED-GNN and NBFNet. The patterns learned in the constrained range in GraIL and CoMPILE are not well generalized to the unseen KG. AdaProp consistently performs the best or second best in different datasets with all the split versions. This demonstrates that the learned adaptive propagation paths can be generalized to a new KG where entities are not seen during training, resulting in strong inductive reasoning ability. 

## _4.1.3 Running time._ 

An advantage of AdaProp is that it has less computing cost due to the reduced number of entities. In this part, we compare the running time of different methods. 

We show the learning curves of different GNN-based methods on transductive data WN18RR and inductive data FB15k237-v1 in Figure 4. First, the full propagation method CompGCN is the slowest method in transductive setting since it has the largest number of involved entities. Second, the constrained propagation methods, i.e., GraIL and CoMPILE, are slow in inductive setting, since their process of subgraph extraction is expensive. CoMPILE with the 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/kdd_2024_AdaProp_Learning_Adaptive_Propagation_for_Graph_Neural_Network_based_Knowledge_Graph_Reasoning_images/kdd_2024_AdaProp_Learning_Adaptive_Propagation_for_Graph_Neural_Network_based_Knowledge_Graph_Reasoning.pdf-0007-11.png)


**----- Start of picture text -----**<br>
WN18RR FB15k237-v1<br>0.6 60<br>GraIL<br>CoMPILE<br>RED-GNN<br>0.5 50 NBFNet<br>AdaProp<br>0.4 CompGCN 40<br>RED-GNN<br>NBFNet<br>AdaProp<br>0.3 30<br>0 1 2 3 4 0 100 200 300<br>Training time (hours) Training time (seconds)<br>Figure 4: Learning curves of different GNN-based methods.<br>MRR<br>Hit@10 (%)<br>**----- End of picture text -----**<br>


more complex message passing functions is even slower than GraIL. Finally, comparing AdaProp with the progressive propagation methods NBFNet and RED-GNN, the learning curve of AdaProp grows faster since it propagates with much fewer entities and the sampling cost is not high. Hence, the adaptive sampling technique is not only effective in both reasoning setting, but also efficient in propagating within fewer entities. 

## **4.2 Understanding the sampling mechanism** 

In this section, we conduct a thorough ablation study via evaluating the sampling mechanism from four perspectives, i.e., the comparison on alternative sampling strategies, the importance of learnable sampling distribution, the comparison on different learning strategies, and the influence of sampled entities _𝐾_ . The 

KDD ’23, August 6–10, 2023, Long Beach, CA, USA 

Yongqi Zhang, Zhanke Zhou, Quanming Yao, Xiaowen Chu, and Bo Han 

evaluation is based on a transductive dataset WN18RR with MRR metric and an inductive dataset FB15k237-v1 with Hit@10 metric. 

## _4.2.1 Sampling strategy._ 

In this part, we compare the sampling strategies (with design details in Appendix B.3) discussed in Section 2.2, i.e., node-wise sampling, layer-wise sampling and subgraph sampling, and the proposed incremental sampling strategy with metrics EI( _𝐿_ ), ToE( _𝐿_ ) in Eq.(4), and testing performance. For a fair comparison, we guarantee that the different variants have the same number of propagation steps _𝐿_ and the similar amount of entities involved, i.e., EI( _𝐿_ ). 

As shown in Table 4, the incremental sampling is better than the other sampling strategies in both the learned and not learned settings. The success is due to the larger ToE( _𝐿_ ) by preserving the previously sampled entities. The node-wise and layer-wise sampling perform much worse since the sampling there is uncontrolled. As the target entities lie close to the query entities, the subgraph sampling is relatively better than the node-wise and layer-wise sampling by propagating within the relevant local area around the query entity. 

**Table 4: Comparison of different sampling methods.** 

|learn<br>methods|WN18RR<br>EI(_𝐿_) ToE(_𝐿_) MRR|FB15k237-v1<br>EI(_𝐿_) ToE(_𝐿_) Hit@10|
|---|---|---|
|not<br>learned<br>Node-wise<br>Layer-wise<br>Subgraph<br>Incremental|4831<br>1_._38E−4 _._416<br>5035<br>1_._46E−4 _._428<br>5098<br>1_._57E−4 _._461<br>4954<br>1_._61E−4 _._472|585<br>1_._35E−3<br>38_._9<br>554<br>1_._45E−3<br>37_._2<br>578<br>1_._50E−3<br>40_._5<br>559<br>1_._52E−3<br>40_._1|
|learned<br>Node-wise<br>Layer-wise<br>Incremental|4913<br>1_._52E−4 _._529<br>4871<br>1_._64E−4 _._533<br>4749<br>1_._78E−4 _._562|561<br>1_._47E−3<br>50.4<br>556<br>1_._55E−3<br>52.4<br>564<br>1_._57E−3<br>55.1|



_4.2.2 Importance of learning._ 

To show the necessity of learning, we apply the same sampling distribution and learning strategy in Section 3.3 to node-wise sampling and layer-wise sampling. The subgraph sampling is hard to be learned with the proposed sampling distribution, thus not compared in this setting. By comparing the top part and bottom part in Table 4, we observe that learning is important for all the three sampling strategies. Since the entity representations can be used to measure the semantic relevance of different entities, a parameterized sampling distribution in proportion to _𝑔_ ( _𝒉𝑒[ℓ] 𝑜_ ; _𝜽[ℓ]_ ) helps to identify the promising targets, increasing ToE( _𝐿_ ) and as a result leading to better performance. Overall, the incremental sampling equipped with the learnable sampling distribution is the best choice among the sampling methods in Table 4. 

## _4.2.3 Learning strategies._ 

In this part, we compare the variants of learning strategies. Apart from straight-through estimator (ST) introduced in Section 3.3.2, another common gradient estimators for discrete variables is the REINFORCE [27, 41], where the gradients are back-propagated by the log-trick (named as REINFORCE). Alternatively, the optimization problem in Eq.(3) can also be formed as a bi-level problem (named as Bi-level), where the sampler parameters are optimized by loss on validation data and are alternatively updated with the model parameters. The random sampler without optimizer (named as Random) is included as a reference baseline. 

The evaluation results of the four learning strategies are provided in Figure 5. First, compared with the random sampler, all of the three learned distributions by Bi-level, REINFORCE and ST, are better, again verifying the importance of learning when sampling. Second, the learning curves of REINFORCE is quite unstable as this estimator has high variance. Third, the ST estimator in a single level optimization is better than the other variants with higher any-time performance. 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/kdd_2024_AdaProp_Learning_Adaptive_Propagation_for_Graph_Neural_Network_based_Knowledge_Graph_Reasoning_images/kdd_2024_AdaProp_Learning_Adaptive_Propagation_for_Graph_Neural_Network_based_Knowledge_Graph_Reasoning.pdf-0008-13.png)


**----- Start of picture text -----**<br>
WN18RR FB15k237-v1<br>0.6 60<br>50<br>0.4 40<br>Random Random<br>Bi-level 30 Bi-level<br>REINFORCE REINFORCE<br>ST ST<br>0.2 20<br>0 1 2 3 4 0 1 2<br>Training time (hours) Training time (minutes)<br>MRR<br>Hit@10 (%)<br>**----- End of picture text -----**<br>


**Figure 5: Comparison of learning strategies for the sampler.** 

## _4.2.4 Influence of 𝐾._ 

As discussed in Section 3.2, the number of sampled entities _𝐾_ bounds the number of involved entities in the propagation path. To figure out the influence of _𝐾_ , we show the performance of different _𝐾_ with respect to different propagation depth _𝐿_ in Figure 6. As shown, when _𝐾_ is very small, the results are worse than the no sampling baseline (RED-GNN) since the ratio of target entity coverage is also small. When increasing _𝐾_ , the performance tends to be better. However, the performance gains by sampling more entities would gradually become marginal or even worse, especially for larger _𝐿_ . It means that the more irrelevant information to the queries will be included with a larger _𝐾_ and _𝐿_ . Besides, a larger _𝐾_ will increase the training cost. Thus, _𝐾_ should be chosen with a moderate value. 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/kdd_2024_AdaProp_Learning_Adaptive_Propagation_for_Graph_Neural_Network_based_Knowledge_Graph_Reasoning_images/kdd_2024_AdaProp_Learning_Adaptive_Propagation_for_Graph_Neural_Network_based_Knowledge_Graph_Reasoning.pdf-0008-17.png)


**----- Start of picture text -----**<br>
WN18RR FB15k237-v1<br>52<br>0.56<br>50<br>0.54 48<br>500 entities 100 entities<br>1000 entities 46 200 entities<br>2000 entities 500 entities<br>no sampling no sampling<br>0.52 44<br>5 6 7 5 6 7<br>Propagation step Propagation step<br>MRR<br>Hit@10 (%)<br>**----- End of picture text -----**<br>


**Figure 6: Ablation study with** _𝐾_ **on different** _𝐿_ **. Each line represents the performance of a given** _𝐾_ **with different** _𝐿_ **.** 

## **4.3 Case study: adaptive propagation path** 

In Section 3.3, we claim that the sampling distribution is semanticaware. To demonstrate this point, we count the number of relation types in the sampled edges of the propagation path for different query relations. We plot the ratio of relation types as heatmap on two semantic friendly datasets, i.e. Family and FB15k237, in Figure 7. Specially, as there are 237 relations in FB15k237, we plot 

KDD ’23, August 6–10, 2023, Long Beach, CA, USA 

AdaProp: Learning Adaptive Propagation for Graph Neural Network based Knowledge Graph Reasoning 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/kdd_2024_AdaProp_Learning_Adaptive_Propagation_for_Graph_Neural_Network_based_Knowledge_Graph_Reasoning_images/kdd_2024_AdaProp_Learning_Adaptive_Propagation_for_Graph_Neural_Network_based_Knowledge_Graph_Reasoning.pdf-0009-02.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/kdd_2024_AdaProp_Learning_Adaptive_Propagation_for_Graph_Neural_Network_based_Knowledge_Graph_Reasoning_images/kdd_2024_AdaProp_Learning_Adaptive_Propagation_for_Graph_Neural_Network_based_Knowledge_Graph_Reasoning.pdf-0009-03.png)


**----- Start of picture text -----**<br>
(a) Progressive on Family (b) AdaProp on Family<br>(c) Progressive on FB15k237 (d) AdaProp on FB15k237<br>**----- End of picture text -----**<br>


**Figure 7: Heatmaps of relation type ratios in the propagation path. Rows are different query relations and columns are relation types in the selected edges. For** FB15k237 **, we only show 10 selected relations that are related to “** _**film**_ **” here.** 

10 relations related to “ _film_ ” in Figure 7(c) and Figure 7(d). We compare the propagation path of progressive propagation methods (NBFNet and RED-GNN have the same range) and AdaProp. 

First, as claimed in the introduction, the progressive propagation methods ignore the semantic relevance between local neighborhoods and the query relation. As a result, the different rows for different query relations in Figure 7(a) and 7(c) are the same. In comparison, the heatmaps in Figure 7(b) and Figure 7(d) vary for different query relations. Second, the more frequent relation types are semantic-aware. For the heatmap in Figure 7(b), we observe that when the query relation is _brother_ , columns with deeper color are all males, i.e., _brother_ , _nephew_ and _uncle_ . When the query relation is _niece_ , columns with deeper color are all females, i.e., _aunt_ and _niece_ . For the heatmap in Figure 7(d), the first four relations are about film award and the last six relations are about film categories. As a result, this figure shows two groups of these relations. Above evidence demonstrates that AdaProp can indeed learn semanticaware propagation path. 

In addition, we visualize the exemplar propagation paths of AdaProp and progressive propagation methods for two different queries _𝑞_ 1 = ( _𝑒𝑞,𝑟_ 1 _,_ ?) and _𝑞_ 2 = ( _𝑒𝑞,𝑟_ 2 _,_ ?) in Figure 8 (with more examples in in Figure 9 in Appendix), which have the same query entity _𝑒𝑞_ but different query relations ( _𝑟_ 1 ≠ _𝑟_ 2) on FB15k237-v1. As shown, there is a clear difference between the two propagation paths of _𝑞_ 1 and _𝑞_ 2 learned by AdaProp. But they are the same for the progressive propagation methods, where only the edge weights are different. Besides, the propagation paths for AdaProp have much fewer entities. The visualization cases here further demonstrate that AdaProp indeed can learn the query dependent propagation paths and is entity efficient. 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/kdd_2024_AdaProp_Learning_Adaptive_Propagation_for_Graph_Neural_Network_based_Knowledge_Graph_Reasoning_images/kdd_2024_AdaProp_Learning_Adaptive_Propagation_for_Graph_Neural_Network_based_Knowledge_Graph_Reasoning.pdf-0009-08.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/kdd_2024_AdaProp_Learning_Adaptive_Propagation_for_Graph_Neural_Network_based_Knowledge_Graph_Reasoning_images/kdd_2024_AdaProp_Learning_Adaptive_Propagation_for_Graph_Neural_Network_based_Knowledge_Graph_Reasoning.pdf-0009-09.png)


**----- Start of picture text -----**<br>
(a) AdaProp, 𝑞 1 (b) Progressive,  𝑞 1<br>(c) AdaProp,  𝑞 2 (d) Progressive,  𝑞 2<br>**----- End of picture text -----**<br>


**Figure 8: Exemplar propagation paths on** FB15k237-v1 **dataset. The big red and green nodes indicate the query entity and answer entity, respectively. The small nodes in red, yellow, orange, blue, purple are new entities involved in the** 1 ∼ 5 **propagation steps.** 

## **ACKNOWLEDGMENT** 

Y. Zhang is supported by National Key Research and Development Plan under Grant 2021YFE0205700. Q. Yao is supported by NSF of China (No. 92270106) and Independent scientific research fund (Foshan advanced manufacturing research institute, Tsinghua University). Z. Zhou and B. Han are supported by NSFC Young Scientists Fund No. 62006202, Guangdong Basic and Applied Basic Research Foundation No. 2022A1515011652, and HKBU CSD Departmental Incentive Grant. 

## **5 CONCLUSION** 

In this paper, we propose a new GNN-based KG reasoning method, called AdaProp. Different from existing GNN-based methods with hand-designed propagation path, AdaProp learns adaptive propagation path during message propagation. AdaProp contains two important components, i.e., an incremental sampling strategy where nearby targets and layer-wise connections can be preserved with linear complexity of involved entities, and a learning based sampling distribution that can identify semantically related entities during propagation and sampler is jointly optimized with the GNN models. Empirical results on several benchmarking datasets in both transductive and inductive KG reasoning settings prove the superiority of AdaProp by learning adaptive propagation path. The ablation studies show that incremental sampling is better than the other sampling strategies, and it is important as well as effective to learn the sampling distribution with straight-through gradient estimator. Case studies on the learned propagation path show that the selected entities are semantic-aware and query-dependent. 

KDD ’23, August 6–10, 2023, Long Beach, CA, USA 

Yongqi Zhang, Zhanke Zhou, Quanming Yao, Xiaowen Chu, and Bo Han 

## **REFERENCES** 

- [1] P. W Battaglia, J. B Hamrick, V. Bapst, A. Sanchez-Gonzalez, V. Zambaldi, et al. 2018. _Relational inductive biases, deep learning, and graph networks_ . Technical Report. arXiv:1806.01261. 

- [2] Y. Bengio, N. Léonard, and A. Courville. 2013. _Estimating or propagating gradients through stochastic neurons for conditional computation_ . Technical Report. arXiv:1308.3432. 

- [3] A. Bordes, N. Usunier, A. Garcia-Duran, J. Weston, and O. Yakhnenko. 2013. Translating embeddings for modeling multi-relational data. In _NeurIPS_ . 2787– 2795. 

- [4] Y. Cao, X. Wang, X. He, Z. Hu, and T. Chua. 2019. Unifying knowledge graph learning and recommendation: Towards a better understanding of user preferences. In _TheWebConf_ . 151–161. 

- [5] J. Chen, T. Ma, and C. Xiao. 2018. FastGCN: Fast Learning with Graph Convolutional Networks via Importance Sampling. In _ICLR_ . 

- [6] X. Chen, S. Jia, and Y. Xiang. 2020. A review: Knowledge reasoning over knowledge graph. _Expert Systems with Applications_ 141 (2020), 112948. 

- [7] K. Cheng, J. Liu, W. Wang, and Y. Sun. 2022. RLogic: Recursive Logical Rule Learning from Knowledge Graphs. In _KDD_ . 179–189. 

- [8] W. Chiang, X. Liu, S. Si, Y. Li, S. Bengio, and C. Hsieh. 2019. Cluster-gcn: An efficient algorithm for training deep and large graph convolutional networks. In _SIGKDD_ . 257–266. 

- [9] R. Das, S. Dhuliawala, M. Zaheer, L. Vilnis, I. Durugkar, A. Krishnamurthy, A. Smola, and A. McCallum. 2017. Go for a walk and arrive at the answer: Reasoning over paths in knowledge bases using reinforcement learning. In _ICLR_ . 

- [10] T. Dettmers, P. Minervini, P. Stenetorp, and S. Riedel. 2017. Convolutional 2D knowledge graph embeddings. In _AAAI_ . 

- [11] T. Dettmers, P. Minervini, P. Stenetorp, and S. Riedel. 2018. Convolutional 2d knowledge graph embeddings. In _AAAI_ . 

- [12] B. Dhingra, M. Zaheer, V. Balachandran, G. Neubig, R. Salakhutdinov, and W. W Cohen. 2019. Differentiable Reasoning over a Virtual Knowledge Base. In _ICLR_ . 

- [13] A. Feeney, R. Gupta, V. Thost, R. Angell, G. Chandu, Y. Adhikari, and T. Ma. 2021. _Relation matters in sampling: a scalable multi-relational graph neural network for drug-drug interaction prediction_ . Technical Report. arXiv:2105.13975. 

- [14] J. Gilmer, S. S Schoenholz, P. F Riley, O. Vinyals, and G. E Dahl. 2017. Neural Message Passing for Quantum Chemistry. In _ICML_ . 1263–1272. 

- [15] W. Hamilton, Z. Ying, and J. Leskovec. 2017. Inductive representation learning on large graphs. In _NeurIPS_ . 1024–1034. 

- [16] A. Hogan, E. Blomqvist, M. Cochez, C. d’Amato, G. de Melo, C. Gutierrez, S. Kirrane, J. Emilio L. Gayo, R. Navigli, S. Neumaier, et al. 2021. Knowledge graphs. _ACM Computing Surveys (CSUR)_ 54, 4 (2021), 1–37. 

- [17] W. Huang, T. Zhang, Y. Rong, and J. Huang. 2018. Adaptive Sampling Towards Fast Graph Representation Learning. In _NeurIPS_ , Vol. 31. 4558–4567. 

- [18] X. Huang, J. Zhang, D. Li, and P. Li. 2019. Knowledge graph embedding based question answering. In _WSDM_ . 105–113. 

- [19] E. Jang, S. Gu, and B. Poole. 2017. Categorical Reparameterization with GumbelSoftmax. In _ICLR_ . 

- [20] S. Ji, S. Pan, E. Cambria, P. Marttinen, and P. S Yu. 2020. _A survey on knowledge graphs: Representation, acquisition and applications_ . Technical Report. arXiv:2002.00388. 

- [21] D. P Kingma and J. Ba. 2014. _Adam: A method for stochastic optimization_ . Technical Report. arXiv:1412.6980. 

- [22] T. N Kipf and M. Welling. 2016. Semi-supervised classification with graph convolutional networks. In _ICLR_ . 

- [23] S. Kok and P. Domingos. 2007. Statistical predicate invention. In _ICML_ . 433–440. 

- [24] W. Kool, H. Van Hoof, and M. Welling. 2019. Stochastic beams and where to find them: The gumbel-top-k trick for sampling sequences without replacement. In _ICML_ . PMLR, 3499–3508. 

- [25] S. Mai, S. Zheng, Y. Yang, and H. Hu. 2021. Communicative Message Passing for Inductive Relation Reasoning.. In _AAAI_ . 4294–4302. 

- [26] C. Meilicke, M. Fink, Y. Wang, D. Ruffinelli, R. Gemulla, and H. Stuckenschmidt. 2018. Fine-grained evaluation of rule-and embedding-based systems for knowledge graph completion. In _ISWC_ . Springer, 3–20. 

- [27] A. Mnih and K. Gregor. 2014. Neural variational inference and learning in belief networks. In _ICML_ . PMLR, 1791–1799. 

- [28] K. Oono and T. Suzuki. 2019. Graph Neural Networks Exponentially Lose Expressive Power for Node Classification. In _ICLR_ . 

- [29] A. Paszke, S. Gross, S. Chintala, G. Chanan, E. Yang, Z. DeVito, Z. Lin, A. Desmaison, L. Antiga, and A. Lerer. 2017. Automatic differentiation in PyTorch. In _ICLR_ . 

- [30] M. Qu, J. Chen, L. Xhonneux, Y. Bengio, and J. Tang. 2021. RNNLogic: Learning Logic Rules for Reasoning on Knowledge Graphs. In _ICLR_ . 

- [31] A. Sadeghian, M. Armandpour, P. Ding, and D. Z. Wang. 2019. DRUM: End-To-End Differentiable Rule Mining On Knowledge Graphs. In _NeurIPS_ . 15347–15357. 

- [32] M. Schlichtkrull, T. N Kipf, P. Bloem, R. Van Den Berg, I. Titov, and M. Welling. 2018. Modeling relational data with graph convolutional networks. In _ESWC_ . Springer, 593–607. 

- [33] F. Suchanek, G. Kasneci, and G. Weikum. 2007. Yago: A core of semantic knowledge. In _The WebConf_ . 697–706. 

- [34] Z. Sun, Z. Deng, J. Nie, and J. Tang. 2019. Rotate: Knowledge graph embedding by relational rotation in complex space. In _ICLR_ . 

- [35] R. S Sutton and A. Barto. 2018. _Reinforcement learning: An introduction_ . MIT press. 

- [36] K. K Teru, E. Denis, and W. L Hamilton. 2020. _Inductive Relation Prediction by Subgraph Reasoning_ . Technical Report. arXiv:1911.06962. 

- [37] K. Toutanova and D. Chen. 2015. Observed versus latent features for knowledge base and text inference. In _PWCVSMC_ . 57–66. 

- [38] S. Vashishth, S. Sanyal, V. Nitin, and P. Talukdar. 2019. Composition-based multi-relational graph convolutional networks. 

- [39] Q. Wang, Z. Mao, B. Wang, and L. Guo. 2017. Knowledge graph embedding: A survey of approaches and applications. _TKDE_ 29, 12 (2017), 2724–2743. 

- [40] X. Wang, X. He, Y. Cao, M. Liu, and T. Chua. 2019. Kgat: Knowledge graph attention network for recommendation. In _SIGKDD_ . 950–958. 

- [41] R. J. Williams. 1992. Simple statistical gradient-following algorithms for connectionist reinforcement learning. _Machine Learning Journal_ 8, 3-4 (1992), 229–256. 

- [42] C. Wilt, J. Thayer, and W. Ruml. 2010. A comparison of greedy search algorithms. In _Proceedings of the International Symposium on Combinatorial Search_ , Vol. 1. 129–136. 

- [43] S. Xie and S. Ermon. 2019. Reparameterizable Subset Sampling via Continuous Relaxations. In _IJCAI_ . 

- [44] W. Xiong, T. Hoang, and W. Wang. 2017. DeepPath: A Reinforcement Learning Method for Knowledge Graph Reasoning. In _EMNLP_ . 564–573. 

- [45] X. Xu, W. Feng, Y. Jiang, X. Xie, Z. Sun, and Z. Deng. 2019. Dynamically Pruned Message Passing Networks for Large-Scale Knowledge Graph Reasoning. In _ICLR_ . 

- [46] F. Yang, Z. Yang, and W. W Cohen. 2017. Differentiable learning of logical rules for knowledge base reasoning. In _NeurIPS_ . 2319–2328. 

- [47] Z. Ye, Y. Jaya Kumar, G. Sing, F. Song, and J. Wang. 2022. A comprehensive survey of graph neural networks for knowledge graphs. _IEEE Access_ 10 (2022), 75729–75741. 

- [48] M. Yoon, T. Gervet, B. Shi, S. Niu, Q. He, and J. Yang. 2021. PerformanceAdaptive Sampling Strategy Towards Fast and Accurate Graph Neural Networks. In _SIGKDD_ . 2046–2056. 

- [49] Y. Yu, K. Huang, C. Zhang, L. M Glass, J. Sun, and C. Xiao. 2021. SumGNN: multityped drug interaction prediction via efficient knowledge graph summarization. _Bioinformatics_ 37, 18 (2021), 2988–2995. 

- [50] H. Zeng, M. Zhang, Y. Xia, A. Srivastava, A. Malevich, R. Kannan, V. Prasanna, L. Jin, and R. Chen. 2021. Decoupling the Depth and Scope of Graph Neural Networks. In _NeurIPS_ , Vol. 34. 

- [51] H. Zeng, H. Zhou, A. Srivastava, R. Kannan, and V. Prasanna. 2019. GraphSAINT: Graph Sampling Based Inductive Learning Method. In _ICLR_ . 

- [52] S. Zhang, Y. Tay, L. Yao, and Q. Liu. 2019. Quaternion knowledge graph embeddings. _NeurIPS_ 32 (2019). 

- [53] Y. Zhang and Q. Yao. 2022. Knowledge Graph Reasoning with Relational Directed Graph. In _TheWebConf_ . 

- [54] Y. Zhang, Q. Yao, and L. Chen. 2020. Interstellar: Searching Recurrent Architecture for Knowledge Graph Embedding. In _NeurIPS_ , Vol. 33. 

- [55] Y. Zhang, Q. Yao, W. Dai, and L. Chen. 2020. AutoSF: Searching scoring functions for knowledge graph embedding. In _ICDE_ . IEEE, 433–444. 

- [56] Z. Zhu, Z. Zhang, L. Xhonneux, and J. Tang. 2021. Neural Bellman-Ford Networks: A General Graph Neural Network Framework for Link Prediction. In _NeurIPS_ . 

- [57] D. Zou, Z. Hu, Y. Wang, S. Jiang, Y. Sun, and Q. Gu. 2019. Layer-Dependent Importance Sampling for Training Deep and Large Graph Convolutional Networks. _NeurIPS_ 32 (2019), 11249–11259. 

KDD ’23, August 6–10, 2023, Long Beach, CA, USA 

AdaProp: Learning Adaptive Propagation for Graph Neural Network based Knowledge Graph Reasoning 

**Notations.** In this paper, vectors are denoted by lowercase boldface, e.g., _𝒎, 𝒉, 𝒘_ . Matricies are denoted by uppercase boldface, e.g., _𝑾_ . Sets are denoted by script fonts, e.g,. E _,_ V. As introduced in Section 3, a knowledge graph is in the form of K = {V _,_ R _,_ E _,_ Q}. 

## **A DETAILS OF WORKS IN THE LITERATURE** 

## **A.1 Summary of MESS** (·) **and AGG** (·) 

The two functions of the GNN-based methods are summarized in Table 5. The main differences of these methods are the combinators for entity and relation representations on the edges, the operators on the different representations, and the attention weights. Recent works [38, 56] have shown that the different combinators and operators only have slight influence on the performance. Comparing GraIL and RED-GNN, even though the message functions and aggregation functions are similar, their empirical performances are quite different with different propagation patterns. Hence, the design of propagation path is the _key factor_ influencing the performance of different methods. 

## **A.2 Discussion on constrained propagation** 

In KG reasoning, the key objective is to find the target answer entity _𝑒𝑎_ given ( _𝑒𝑞,𝑟𝑞,_ ?). To fulfill this goal, the model is required to evaluate multiple target answer entities. Since all the entities in the KG can be potential answers, we need to evaluate |V| entities for a single query ( _𝑒𝑞,𝑟𝑞,_ ?). However, for the constrained propagation method GraIL [36] and CoMPILE [25], the propagation path depends on both the query entity _𝑒𝑞_ and the answer entity _𝑒𝑎_ , i.e., V _[ℓ]_ = V _𝑒[𝐿] 𝑞,𝑒𝑎_ . Hence, it requires |V| times of forward passes to evaluate all the entities for a single query, which is extremely expensive when the number of entities is large. For the settings of transductive reasoning, all datasets have tens of thousands of entities. Thus, running GraIL or CoMPILE on them is intractable. 

## **B DETAILS OF THE SAMPLING METHOD** 

## **B.1 Augmentation with inverse relations** 

As aforementioned, predicting missing head in KG can also be formulated in this way by adding inverse relations. This is achieved by augmenting the original KG by the reverse relations and the identity relation, as adopted in path-based methods [30, 31]. Taking Figure 1 as example, we add the reverse relation "part_of inv" to the original relation "part_of", which leads to a new triplet _(Lakers, part_of inv, LeBron)_ . We also add the identity relation as the selfloop, which leads to new triplets, such as _(LeBron, identity, LeBron)_ . 

## **B.2 Proofs of propositions** 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/kdd_2024_AdaProp_Learning_Adaptive_Propagation_for_Graph_Neural_Network_based_Knowledge_Graph_Reasoning_images/kdd_2024_AdaProp_Learning_Adaptive_Propagation_for_Graph_Neural_Network_based_Knowledge_Graph_Reasoning.pdf-0011-12.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/kdd_2024_AdaProp_Learning_Adaptive_Propagation_for_Graph_Neural_Network_based_Knowledge_Graph_Reasoning_images/kdd_2024_AdaProp_Learning_Adaptive_Propagation_for_Graph_Neural_Network_based_Knowledge_Graph_Reasoning.pdf-0011-13.png)


_B.2.2 Proposition 2._ Since V _𝑒[ℓ] 𝑞,𝑟𝑞_ := V _𝑒[ℓ] 𝑞_[−] _,𝑟_[1] _𝑞_[∪][SAMP][(] V _𝑒_ ~~_[ℓ]_~~ _𝑞,𝑟𝑞_[)][, we] have V _𝑒[ℓ] 𝑞_[−] _,𝑟_[1] _𝑞_[⊆V] _𝑒[ℓ] 𝑞,𝑟𝑞_[. With the introduction of identity relation] in Section B.1, for each _𝑒_ ∈V _𝑒[ℓ] 𝑞_[−] _,𝑟_[1] _𝑞_[, there exist at least one entity] _𝑒_ ∈V _𝑒[ℓ] 𝑞,𝑟𝑞_ such that ( _𝑒, identity,𝑒_ ) ∈E _[ℓ]_ . Thus, the connection can be preserved. 

## **B.3 Adapting sampling methods for KG** 

When modeling the homogeneous graphs, the sampling methods are designed to solve the scalability problem. Besides, they are mainly used in the node classification task, which is quite different from the link-level KG reasoning task here. Hence, we introduce the adaptations we made to use the sampling methods designed for homogeneous graphs in the KG reasoning scenarios as follows. 

- Node-wise sampling. The design in this type is based on GraphSAGE [15]. Given V _𝑒[ℓ] 𝑞_[−] _,𝑟_[1] _𝑞_[, we sample] _[ 𝐾]_[neighboring entities] from N ( _𝑒_ ) for entities _𝑒_ ∈V _𝑒[ℓ] 𝑞_[−] _,𝑟_[1] _𝑞_[.][Then][the][union][of][these] sampled entities forms V _𝑒[ℓ] 𝑞,𝑟𝑞_ for next step propagation. For the not-learned version, we randomly sample from N ( _𝑒_ ). For the learned version, we sample from N ( _𝑒_ ) based on the distribution _𝑝[ℓ]_ ( _𝑒_ ) ∼ _𝑔_ ( _𝒉𝑒[ℓ]_ ; _𝜽[ℓ]_ ). 

- Layer-wise sampling. The design in this type is based on LADIES [57]. Given V _𝑒[ℓ] 𝑞_[−] _,𝑟_[1] _𝑞_[,][we][firstly][collect][the][set][of][neighboring] entities N (V _[ℓ]_[−][1] ) = � _𝑒_ ∈V _[ℓ]_[−][1][Then,][we][sample] _[𝐾] 𝑒𝑞,𝑟𝑞_[N (] _[𝑒]_[)][.] 

- entities without replacement from N (V _𝑒[ℓ] 𝑞_[−] _,𝑟_[1] _𝑞_[)][ as][ V] _[ℓ]_[for next step] propagation. For the not-learned version, we sample according to the degree distribution for entities in N (V _𝑒[ℓ] 𝑞_[−] _,𝑟_[1] _𝑞_[)][. For the learned] version, we sample from N (V _𝑒[ℓ] 𝑞_[−] _,𝑟_[1] _𝑞_[)][based][on][the][distribution] _𝑝[ℓ]_ ( _𝑒_ ) ∼ _𝑔_ ( _𝒉𝑒[ℓ]_ ; _𝜽[ℓ]_ ). 

- Subgraph sampling. Given the query ( _𝑒𝑞,𝑟𝑞,_ ?), we generate multiple random walks starting from the query entity _𝑒𝑞_ and use the sampled entities to induce the subgraph V _𝑒𝑞_ . Then, the propagation starts from _𝑒𝑞_ like the progressive propagation, but it is constrained within the range of V _𝑒𝑞_ . Since learning to sample the subgraph is challenging, we only evaluate the subgraph sampling in the not-learned version. 

## **B.4 Implementation of ST estimator** 

The key idea of straight-through (ST) estimator is to back-propagate through the sampling signal as if it was the identity function. Given the sampled entities _𝑒𝑜_ ∈ V[�] _𝑒[ℓ] 𝑞,𝑟𝑞_ as well as their sampling probability _𝑝[ℓ]_ ( _𝑒_ ), the derivative of the sampling parameters can be obtained by multiplying _𝑝[ℓ]_ ( _𝑒_ ) with the hidden states _𝒉𝑒[ℓ]_ . Specifically, rather than directly use the _𝒉𝑒[ℓ]_ to compute messages, we use 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/kdd_2024_AdaProp_Learning_Adaptive_Propagation_for_Graph_Neural_Network_based_Knowledge_Graph_Reasoning_images/kdd_2024_AdaProp_Learning_Adaptive_Propagation_for_Graph_Neural_Network_based_Knowledge_Graph_Reasoning.pdf-0011-22.png)


where no_grad( _𝑝[ℓ]_ ( _𝑒_ )) means that the back-propagation signals will not go through this term. In this approach, the forward computation will not be influenced, while the backward signals can go through the multiplier _𝑝[ℓ]_ ( _𝑒_ ). Besides, the backward signals, even though not exactly equivalent to the true gradient, is in positive propagation to it. Hence, we can use such a backward signal to approximately estimate the gradient of L with regard to _𝜽_ . 

Namely, the number of involved entities is bounded by _𝑂_ ( _𝐿_ · _𝐾_ ). 

KDD ’23, August 6–10, 2023, Long Beach, CA, USA 

Yongqi Zhang, Zhanke Zhou, Quanming Yao, Xiaowen Chu, and Bo Han 

**Table 5: Summary of the GNN functions for message propagation. The values in** {} **represent different operation choices that are tuned as hyper-parameters for different datasets.** 

|method|_𝒎ℓ_<br>(_𝑒𝑠,𝑟,𝑒𝑜_) :=MESS(·)|_𝒉ℓ𝑒𝑜_:=AGG(·)|_𝑜_)|
|---|---|---|---|
|R-GCN [32]|_𝑾ℓ𝑟𝒉ℓ_−1<br>_𝑒𝑠_, where_𝑾𝑟_depends on the relation_𝑟_|_𝑾ℓ𝑜𝒉ℓ_−1<br>_𝑒𝑜_+�<br>_𝑒𝑜_∈N(_𝑒𝑠_)<br>1<br>_𝑐𝑜,𝑟𝒎ℓ_<br>(_𝑒𝑠,𝑟,𝑒𝑜_)||
|CompGCN [38]|_𝑾ℓ_<br>_𝜆_(_𝑟_){−_,_∗_,★_}(_𝒉ℓ_−1<br>_𝑒𝑠, 𝒉ℓ𝑟_), where_𝑾ℓ_<br>_𝜆_(_𝑟_) depends on the direction of_𝑟_|�<br>_𝑒𝑜_∈N(_𝑒𝑠_) _𝒎ℓ_<br>(_𝑒𝑠,𝑟,𝑒𝑜_)||
|GraIL [36]|_𝛼ℓ_<br>(_𝑒𝑠,𝑟,𝑒𝑜_) |_𝑟𝑞_(_𝑾ℓ_<br>1_𝒉ℓ_−1<br>_𝑒𝑠_<br>+_𝑾ℓ_<br>2_𝒉ℓ_−1<br>_𝑒𝑜_), where_𝛼ℓ_<br>(_𝑒𝑠,𝑟,𝑒𝑜_) |_𝑟𝑞_is the attention weight.|_𝑾ℓ𝑜𝒉ℓ_−1<br>_𝑒𝑜_+�<br>_𝑒𝑜_∈N(_𝑒𝑠_)<br>1<br>_𝑐𝑜,𝑟𝒎ℓ_<br>(_𝑒𝑠,𝑟,𝑒𝑜_)||
|NBFNet [56]|_𝑾ℓ_{+_,_∗_,_◦}(_𝒉ℓ_−1<br>_𝑒𝑠,𝒘𝑞_(_𝑟,𝑟𝑞_)), where_𝒘𝑞_(_𝑟,𝑟𝑞_)is a query-dependent weight<br>vector|{Sum,Mean,Max,PNA}_𝑒𝑜_∈N(_𝑒𝑠_)_𝒎ℓ_<br>(_𝑒𝑠,𝑟,𝑒_||
|RED-GNN [53]|_𝛼ℓ_<br>(_𝑒𝑠,𝑟,𝑒𝑜_) |_𝑟𝑞_(_𝒉ℓ_−1<br>_𝑒𝑠_<br>+_𝒉ℓ𝑟_), where_𝛼ℓ_<br>(_𝑒𝑠,𝑟,𝑒𝑜_) |_𝑟𝑞_is the attention weight|�<br>_𝑒𝑜_∈N(_𝑒𝑠_) _𝒎ℓ_<br>(_𝑒𝑠,𝑟,𝑒𝑜_)||
|AdaProp|_𝛼ℓ_<br>(_𝑒𝑠,𝑟,𝑒𝑜_) |_𝑟𝑞_{+_,_∗_,_◦}(_𝒉ℓ_−1<br>_𝑒𝑠, 𝒉ℓ𝑟_), where_𝛼ℓ_<br>(_𝑒𝑠,𝑟,𝑒𝑜_) |_𝑟𝑞_is as in [53].|{Sum,Mean,Max}_𝑒𝑜_∈N(_𝑒𝑠_)_𝒎ℓ_<br>(_𝑒𝑠,𝑟,𝑒𝑜_)||



**Table 7: Optimal hyper-parameter (HP) for each dataset. “BS” is short for batch size.** 

HP Family UMLS WN18RR FB15k237 NELL-995 YAGO3-10 

|HP<br>|Family|UMLS|WN18RR|WN18RR|FB15k237|NELL-995|NELL-995|YAGO3-10|
|---|---|---|---|---|---|---|---|---|
|_𝐿_<br>_𝐾_<br>_𝜏_<br>BS|8<br>100<br>1.0<br>20|5<br>100<br>1.0<br>10|8<br>1000<br>0.5<br>100||7<br>2000<br>2.0<br>50|6<br>2000<br>0.5<br>20||8<br>1000<br>1.0<br>5|
||||||||||
|HP|WN18RR<br>V1 V2 V3 V4|||FB15k237<br>V1 V2 V3 V4|||NELL-995<br>V1 V2 V3 V4||
|_𝐿_<br>_𝐾_<br>_𝜏_<br>BS|7<br>7<br>7<br>6<br>200 100 200 200<br>0.5 0.5 1.0 1.0<br>20<br>20<br>50<br>50|||8<br>6<br>7<br>7<br>200 200 200 500<br>0.5 2.0 1.0 1.0<br>50<br>20<br>20<br>50|||6<br>5<br>8<br>5<br>200 200 500 100<br>0.5 1.0 1.0 2.0<br>20<br>50<br>20<br>50||



## **C FURTHER MATERIALS OF EXPERIMENTS** 

## **C.1 Hop distributions of datasets** 

We summarize the distance distribution of test queries in Table 6 for the largest four KGs. Here, the distance for a query triple ( _𝑒𝑞,𝑟𝑞,𝑒𝑎_ ) is calculated as the length of the shortest path connecting _𝑒𝑞_ and _𝑒𝑎_ from the training triples. As shown, _most answer entities are close to query entities_ , i.e., within the 3-hop neighbors of query entities. 

**Table 6: Distance distribution (in %) of queries in** Q _𝑡𝑠𝑡_ **.** 

|distance|1|2|3|4|5|>5|
|---|---|---|---|---|---|---|
|WN18RR|34.9|9.3|21.5|7.5|8.9|17.9|
|FB15k237|0.0|73.4|25.8|0.2|0.1|0.5|
|NELL-995|40.9|17.2|36.5|2.5|1.3|1.6|
|YAGO3-10|56.0|12.9|30.1|0.5|0.1|0.4|



generally small. This means that _𝐾_ is not as large as possible. More irrelevant entities will influence the performance. (3) There is no much regularity in the choices of _𝜏_ , which depends on specific KGs. 

## **C.3 Hit@1 results of inductive reasoning** 

The full results of inductive reasoning, for MRR and Hit@1 metrics are shown in Table 8. AdaProp is also the best in most of the cases. 

## **C.4 Influence of temperature** _𝜏_ 

Besides, we evaluate the influence of different propagation depth _𝐿_ and number of entities _𝐾_ in Section 4. Here, we show the performances of different temperatures _𝜏_ in Table 9. As can be seen, the influence of different choices of _𝜏_ is not high on the different datasets. This means that the sampling signal is _stable_ and be adjusted by the learnable sampling weights. 

**Table 9: Model performance (MRR) of different** _𝜏_ **values.** 

|_𝜏_|WN18RR|FB15k237-v1|
|---|---|---|
|0.5<br>1.0<br>2.0|0.562<br>0.559<br>0.560|0.551<br>0.546<br>0.549|



## **C.5 Parameter size** 

The parameter size is shown in Table 10. As can be seen, AdaProp is _parameter-efficient_ , although it has larger propagation steps than other GNN-based models and has additional sampler parameters. Since AdaProp works without entity embeddings, the number of parameters is much smaller than the others. 

## **C.6 Importance of deeper propagation step** 

## **C.2 Hyper-parameters** 

We select the optimal hyper-parameters for evaluation based on the MRR metric or the Hit@10 metric on validation set for transductive setting or indcutive setting, respectively. We provide the hyperparameters of _𝐿_ , _𝐾_ , _𝜏_ and batch size in Table 7. We observe that (1) In most cases, we have a larger number of propagation steps with _𝐿_ ≥ 6. (2) In the transductive setting, the values of _𝐾_ are larger since the KGs are larger. But in the inductive setting, _𝐾_ ’s are 

In Section 4.1.1, we attribute the better performance of a deeper propagation path to the long-range information along with a deeper GNN model. Here, we compare the influence of the two factors via explicit decoupling. In Table 11, we show the performance of progressive propagation method RED-GNN by alternately increasing the number of GNN-layers or the depth of propagation path per row. The propagation path _𝐿_ with deeper GNN-layer _𝐿_ + 1 is achieved by repeating the last propagation entities, i.e., V _[𝐿]_[+][1] := V _[𝐿]_ . For example, a propagation path of 2-depth means 

KDD ’23, August 6–10, 2023, Long Beach, CA, USA 

AdaProp: Learning Adaptive Propagation for Graph Neural Network based Knowledge Graph Reasoning 

**Table 8: Inductive setting on MRR and Hit@1.** 

|metrics|methods|WN18RR<br>V1<br>V2<br>V3<br>V4|FB15k237<br>V1<br>V2<br>V3<br>V4|NELL-995<br>V1<br>V2<br>V3<br>V4|
|---|---|---|---|---|
|MRR|RuleN<br>Neural LP<br>DRUM|0.668<br>0.645<br>0.368<br>0.624<br>0.649<br>0.635<br>0.361<br>0.628<br>0.666<br>0.646<br>0.380<br>0.627|0.363<br>0.433<br>0.439<br>0.429<br>0.325<br>0.389<br>0.400<br>0.396<br>0.333<br>0.395<br>0.402<br>0.410|0.615<br>0.385<br>0.381<br>0.333<br>0.610<br>0.361<br>0.367<br>0.261<br>0.628<br>0.365<br>0.375<br>0.273|
||GraIL<br>CoMPILE<br>RED-GNN<br>NBFNet|0.627<br>0.625<br>0.323<br>0.553<br>0.577<br>0.578<br>0.308<br>0.548<br>0.701<br>0.690<br>0.427<br>0.651<br>0.684<br>0.652<br>0.425<br>0.604|0.279<br>0.276<br>0.251<br>0.227<br>0.287<br>0.276<br>0.262<br>0.213<br>**0.369**<br>0.469<br>0.445<br>0.442<br>0.307<br>0.369<br>0.331<br>0.305|0.481<br>0.297<br>0.322<br>0.262<br>0.330<br>0.248<br>0.319<br>0.229<br>0.637<br>0.419<br>**0.436**<br>0.363<br>0.584<br>0.410<br>0.425<br>0.287|
||**AdaProp**|**0.733**<br>**0.715**<br>**0.474**<br>**0.662**|0.310<br>**0.471**<br>**0.471**<br>**0.454**|**0.644**<br>**0.452**<br>0.435<br>**0.366**|
|Hit@1 (%)|RuleN<br>Neural LP<br>DRUM|63.5<br>61.1<br>34.7<br>59.2<br>59.2<br>57.5<br>30.4<br>58.3<br>61.3<br>59.5<br>33.0<br>58.6|**30.9**<br>34.7<br>34.5<br>33.8<br>24.3<br>28.6<br>30.9<br>28.9<br>24.7<br>28.4<br>30.8<br>30.9|**54.5**<br>30.4<br>30.3<br>24.8<br>50.0<br>24.9<br>26.7<br>13.7<br>50.0<br>27.1<br>26.2<br>16.3|
||GraIL<br>CoMPILE<br>RED-GNN<br>NBFNet|55.4<br>54.2<br>27.8<br>44.3<br>47.3<br>48.5<br>25.8<br>47.3<br>65.3<br>63.3<br>36.8<br>60.6<br>59.2<br>57.5<br>30.4<br>57.4|20.5<br>20.2<br>16.5<br>14.3<br>20.8<br>17.8<br>16.6<br>13.4<br>30.2<br>38.1<br>35.1<br>34.0<br>19.0<br>22.9<br>20.6<br>18.5|42.5<br>19.9<br>22.4<br>15.3<br>10.5<br>15.6<br>22.6<br>15.9<br>**52.5**<br>31.9<br>**34.5**<br>**25.9**<br>50.0<br>27.1<br>26.2<br>23.3|
||**AdaProp**|**66.8**<br>**64.2**<br>**39.6**<br>**61.1**|19.1<br>**37.2**<br>**37.7**<br>**35.3**|52.2<br>**34.4**<br>33.7<br>24.7|



**Table 11: Per-hop performance of RED-GNN with different GNN layers and propagation depth. “** ↑ **per row” is computed by minimizing the MRR or Hit@10 value in the current row with that in the previous row.** 

|WN18RR (evaluated by MRR)|WN18RR (evaluated by MRR)|WN18RR (evaluated by MRR)|WN18RR (evaluated by MRR)|
|---|---|---|---|
|depth layers|1-hop ↑per row|2-hop ↑per row|3-hop ↑per row|
|2<br>2|0.940<br>–|0.338<br>–|0.106<br>–|
|2<br>3<br>3<br>3|0.955<br>0.015<br>0.981<br>**0.026**|0.347<br>0.009<br>0.463<br>**0.116**|0.111<br>0.005<br>0.419<br>**0.313**|
|3<br>4<br>4<br>4|0.983<br>0.002<br>0.990<br>**0.007**|0.475<br>0.012<br>0.576<br>**0.101**|0.422<br>0.316<br>0.477<br>**0.371**|
|||||
|FB15k237-v1 (evaluated by Hit@10)||||
|depth layers|2-hop ↑per row|3-hop ↑per row|4-hop ↑per row|
|2<br>2|50.0<br>–|0.4<br>–|1.8<br>–|
|2<br>3<br>3<br>3|52.4<br>2.4<br>56.0<br>**6.0**|0.9<br>0.5<br>23.3<br>**22.9**|1.8<br>0.0<br>1.8<br>0.0|
|3<br>4<br>4<br>4|56.4<br>6.4<br>58.5<br>**8.5**|23.6<br>23.2<br>37.6<br>**37.2**|1.8<br>0.0<br>7.1<br>**5.3**|



**Table 10: Number of parameters of different models.** 

|methods|WN18RR<br>FB15k237|
|---|---|
|RotatE<br>CompGCN<br>RED-GNN<br>NBFNet<br>AdaProp|20,477k<br>29,319k<br>12,069k<br>9,447k<br>58k<br>118k<br>89k<br>3,103k<br>77k<br>195k|



neighborhoods, and the messages cannot be propagated further even with a 3-layer GNN. As indicated by the bold face numbers of row-wise performance improvement, the influence of larger propagation depth is more significant than deeper GNN-layers. Based on the above analysis, we conclude that _the influence of longrange information is more important than a deeper model._ 

## **C.7 Visualization of propagation paths** 

In Section 3.3, we claim that the sampling distribution is related to the query-dependent representations _𝒉𝑒[ℓ] 𝑜_ . We plot more examples of the propagation path of AdaProp (with _𝐿_ = 5 _, 𝐾_ = 10) and REDGNN (with _𝐿_ = 5) for two different queries _𝑞_ 1 = ( _𝑒𝑞,𝑟_ 1 _,_ ?) and _𝑞_ 2 = ( _𝑒𝑞,𝑟_ 2 _,_ ?) in Figure 9, which have the same query entity _𝑒𝑞_ but different query relations (i.e., _𝑟_ 1 ≠ _𝑟_ 2) on FB15k237-v1 dataset. As shown, there is a clear difference between the two propagation paths of _𝑞_ 1 and _𝑞_ 2 learned by AdaProp. But they are the same for the progressive propagation methods, where only the edge weights are different. The above visualization cases further demonstrate that _AdaProp indeed can learn the query dependent propagation paths._ 

We also plot the learned propagation paths G[�] _[𝐿]_ ( _𝜽_ ) with _𝐿_ = 2 _, 𝐾_ = 2 in Figure 10 with the Family dataset. For the query (2720 _,𝑠𝑜𝑛,_ ?), the propagation path goes to find who is the father or mother of 2720. As for (2720 _,𝑛𝑒𝑝ℎ𝑒𝑤,_ ?) with the same query entity, the propagation path tends to select the aunts and uncles. This observation shows that _the learned propagation path is semanticallyaware._ In addition, since AdaProp uses incremental sampling, all the entities in previous steps are preserved during propagation steps. Such a design can provide a further _explanation_ to GNNs due to its _tractable_ as well as _interpretable_ reasoning procedures. 

that the message propagation is constrained in at most 2-hop 

KDD ’23, August 6–10, 2023, Long Beach, CA, USA 

Yongqi Zhang, Zhanke Zhou, Quanming Yao, Xiaowen Chu, and Bo Han 

**Table 14: The values of TC** ( _𝐿_ ) **. Here,** _**Full**_ **indicates the full propagation methods, while** _**Progressive**_ **means the methods with progressive propagation.** 

|WN18RR|2<br>3<br>4<br>5<br>6<br>7<br>8|
|---|---|
|Full<br>Progressive<br>AdaProp|1.00<br>1.00<br>1.00<br>1.00<br>N.A.<br>N.A.<br>N.A.<br>0.44<br>0.66<br>0.73<br>0.82<br>0.86<br>0.86<br>N.A.<br>0.44<br>0.64<br>0.70<br>0.80<br>0.83<br>0.84<br>0.85|
|NELL-995<br>2<br>3<br>4<br>5<br>6<br>7<br>8||
|Full<br>Progressive<br>AdP|1.00<br>1.00<br>1.00<br>N.A.<br>N.A.<br>N.A.<br>N.A.<br>0.58<br>0.94<br>0.97<br>0.98<br>0.99<br>N.A.<br>N.A.<br>058<br>085<br>088<br>090<br>094<br>097<br>098|
|arop|.<br>.<br>.<br>.<br>.<br>.<br>.|



**Table 15: The values of EI** ( _𝐿_ ) **.** 

||**Table 15: The values of EI**(_𝐿_)**.**|
|---|---|
|WN18RR|2<br>3<br>4<br>5<br>6<br>7<br>8|
|Full<br>Progressive<br>AdaProp|43410 43410 43410 43410<br>N.A.<br>N.A.<br>N.A.<br>59<br>281<br>1159<br>3773<br>9703<br>18935<br>N.A.<br>59<br>268<br>848<br>1705<br>2641<br>3592<br>4544|
|NELL-995<br>2<br>3<br>4<br>5<br>6<br>7<br>8||
|Full<br>Progressive<br>AdaProp|74536 74536 74536<br>N.A.<br>N.A.<br>N.A.<br>N.A.<br>691<br>5171<br>19674 35298 43820<br>N.A.<br>N.A.<br>596<br>1966<br>3744<br>5569<br>7406<br>9242<br>11081|




![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/kdd_2024_AdaProp_Learning_Adaptive_Propagation_for_Graph_Neural_Network_based_Knowledge_Graph_Reasoning_images/kdd_2024_AdaProp_Learning_Adaptive_Propagation_for_Graph_Neural_Network_based_Knowledge_Graph_Reasoning.pdf-0014-06.png)


**----- Start of picture text -----**<br>
2446<br>2717 identity 2717<br>2720 identity 2720 identity 2720<br>2527 identity 2527<br>2729<br>(a) G [�] [𝐿] ( 𝜽 ) of query (2720, son, ?).<br>2723<br>603 identity 603<br>2720 identity 2720 identity 2720<br>2529 identity 2529<br>2724<br>(b) G [�] [𝐿] ( 𝜽 ) of query (2720, nephew, ?).<br>_father<br>nephew<br>father<br>uncle<br>_nephew<br>_niece<br>_mother<br>_aunt<br>mother<br>aunt<br>aunt<br>nephew<br>_mother<br>_aunt<br>_father<br>brother<br>_husband<br>husband<br>uncle<br>_nieceaunt<br>aunt<br>**----- End of picture text -----**<br>


**Figure 10: Visualization of the learned propagation path on** Family **dataset. Dashed lines mean reverse relations.** 

## **C.8 Distributions of degree and running time** 

Here, we show the degree distribution and running time distribution in the following Table 12 and Table 13. Comparing std with mean 

value, we observe that the running time of different queries does not change much. Based on Proposition 1, the number of sampled entities is linearly bounded. Thus, the running time is still under control even if hub nodes appear. 

**Table 12: Distribution of running time (in seconds, BS=1).** 

|dataset|max<br>min<br>average<br>std|
|---|---|
|WN18RR<br>FB15k237<br>NELL-995<br>YAGO3-10|0.13<br>0.02<br>0.03<br>0.01<br>0.22<br>0.03<br>0.13<br>0.0<br>0.16<br>0.02<br>0.07<br>0.0<br>0.44<br>0.11<br>0.32<br>0.0|



**Table 13: Distribution of degree.** 

|dataset|max<br>min<br>average<br>std|
|---|---|
|WN18RR<br>FB15k237<br>NELL-995<br>YAGO3-10|494<br>0<br>2.3<br>3.7<br>1518<br>0<br>21.3<br>33.5<br>1011<br>0<br>2.1<br>10.9<br>264<br>0<br>8.8<br>8.8|



## **C.9 Detailed values of TC** ( _𝐿_ ) **and EI** ( _𝐿_ ) 

Here, we provide the values of TC( _𝐿_ ) and EI( _𝐿_ ) in Eq.(4). As can be seen from the following Table 14 and Table 15. AdaProp has smaller values of TC compared with the baselines since it conducts sampling. However, the gap is not large ( _<_ 10%) in all the cases. 

## **C.10 Discussion on samples that get improved with larger steps** 

We add analysis of samples that get improved with larger steps. Specifically, we select the samples ( _𝑒𝑞,𝑟𝑞,𝑒𝑎_ ) that are not correctly predicted at step 4 but are correctly predicted at all the steps _𝐿 >_ 4 (denoted as Group 4). The criterion to judge whether a sample is correctly predicted is based on whether the ranking of _𝑒𝑎_ is ≤ 10. We do the same steps for steps 5 and 6 (denoted as Group 5 and 6). 

**Table 16: Distribution of samples in each group.** 

|N-sample|dist=1<br>dist=2<br>dist=3<br>dist=4<br>dist≥5|
|---|---|
|Group 4<br>743<br>Group 5<br>366<br>Group 6<br>118|1.6%<br>16.3%<br>42.7%<br>22.7%<br>16.7%<br>1.9%<br>11.5%<br>34.4%<br>31.1%<br>21.1%<br>1.7%<br>9.3%<br>24.6%<br>35.6%<br>28.8%|



In the above Table 16, we show the number of samples in each group, and the distribution of the shortest path distance on WN18RR. We observe that (1) the number of samples that will be improved at larger steps gets smaller with increased values of _𝐿_ (indicated by N-sample values); (2) comparing the shortest part distances, Group 6 will have farther away distance samples than Group 5, and the same for Group 5 v.s. Group 4. These results indicate that the samples that are improved in the larger steps are mainly those that have larger shortest path distances. 

KDD ’23, August 6–10, 2023, Long Beach, CA, USA 

AdaProp: Learning Adaptive Propagation for Graph Neural Network based Knowledge Graph Reasoning 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/kdd_2024_AdaProp_Learning_Adaptive_Propagation_for_Graph_Neural_Network_based_Knowledge_Graph_Reasoning_images/kdd_2024_AdaProp_Learning_Adaptive_Propagation_for_Graph_Neural_Network_based_Knowledge_Graph_Reasoning.pdf-0015-02.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/kdd_2024_AdaProp_Learning_Adaptive_Propagation_for_Graph_Neural_Network_based_Knowledge_Graph_Reasoning_images/kdd_2024_AdaProp_Learning_Adaptive_Propagation_for_Graph_Neural_Network_based_Knowledge_Graph_Reasoning.pdf-0015-03.png)


**----- Start of picture text -----**<br>
(a) AdaProp,  𝑞 1 (b) AdaProp,  𝑞 2 (c) Progressive,  𝑞 1 (d) Progressive,  𝑞 2<br>(e) AdaProp,  𝑞 1 (f) AdaProp,  𝑞 2 (g) Progressive,  𝑞 1 (h) Progressive,  𝑞 2<br>(i) AdaProp,  𝑞 1 (j) AdaProp,  𝑞 2 (k) Progressive,  𝑞 1 (l) Progressive,  𝑞 2<br>(m) AdaProp,  𝑞 1 (n) AdaProp,  𝑞 2 (o) Progressive,  𝑞 1 (p) Progressive,  𝑞 2<br>(q) AdaProp,  𝑞 1 (r) AdaProp,  𝑞 2 (s) Progressive,  𝑞 1 (t) Progressive,  𝑞 2<br>**----- End of picture text -----**<br>


**Figure 9: Exemplar propagation paths on** FB15k237-v1 **dataset. Specifically, the big red and green nodes indicate the query entity and answer entity, respectively. The small nodes in red, yellow, orange, blue, purple are entities involved in** 1 ∼ 5 **propagation steps. Besides, the colors and thickness of edges indicate the relation types and the learned edge weights, respectively.** 

