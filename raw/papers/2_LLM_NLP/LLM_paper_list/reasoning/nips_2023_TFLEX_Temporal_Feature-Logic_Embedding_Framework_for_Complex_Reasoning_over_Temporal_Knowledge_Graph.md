# **TFLEX: Temporal Feature-Logic Embedding Framework for Complex Reasoning over Temporal Knowledge Graph** 

**Xueyuan Lin**[1] **Haihong E**[1] *** Chengjin Xu**[2] *** Gengxian Zhou**[1] **Haoran Luo**[1] **Tianyi Hu**[1] **Fenglong Su**[3] **Ningyuan Li**[1] **Mingzhi Sun**[1] 

1 Beijing University of Posts and Telecommunications 

2 University of Bonn 

3 National University of Defense Technology 

linxy59@mail2.sysu.edu.cn, ehaihong@bupt.edu.cn, xuc@iai.uni-bonn.de z.gengxian@se18.qmul.ac.uk, {luohaoran, hutianyi}@bupt.edu.cn sufenglong18@nudt.edu.cn, {jason.ningyuan.li, sunmingzhi}@bupt.edu.cn 

## **Abstract** 

Multi-hop logical reasoning over knowledge graph plays a fundamental role in many artificial intelligence tasks. Recent complex query embedding methods for reasoning focus on static KGs, while temporal knowledge graphs have not been fully explored. Reasoning over TKGs has two challenges: 1. The query should answer entities or timestamps; 2. The operators should consider both set logic on entity set and temporal logic on timestamp set. To bridge this gap, we introduce the multi-hop logical reasoning problem on TKGs and then propose the first temporal complex query embedding named Temporal Feature-Logic Embedding framework (TFLEX) to answer the temporal complex queries. Specifically, we utilize fuzzy logic to compute the logic part of the Temporal Feature-Logic embedding, thus naturally modeling all first-order logic operations on the entity set. In addition, we further extend fuzzy logic on timestamp set to cope with three extra temporal operators ( **After** , **Before** and **Between** ). Experiments on numerous query patterns demonstrate the effectiveness of our method. 

## **1 Introduction** 

Multi-hop logical reasoning over knowledge graphs (KGs) is a fundamental issue in artificial intelligence. It aims to find the answer entities for a first-order logic (FOL) query which involves logical operators (existential quantification _∃_ , conjunction _∧_ , disjunction _∨_ and negation _¬_ ). Generally, the query is parsed into computation graph, according to which subgraph matching is executed on KG to find the answers. The computation graph is a directed acyclic graph (DAG) whose nodes represent entity sets, and edges represent logical operators acting on entity sets. However, results are inevitably incorrect as KGs are incomplete and noisy. Besides, the computation complexity will spiral for large-scale KGs or large queries. Therefore, people propose to embed query into low-dimensional space to solve the problem. 

Query embedding (QE) learns the embeddings of queries and entities, so that answer entities are close to its queries in the embedding space. It has attracted arising attention, as low-dimension embeddings can model implicit dependency and reduce computation. There follows a series of QE methods, including GQE[1], Query2box[2], BetaE[3], ConE[4], etc. However, existing works only focus 

> * Corresponding Authors 

37th Conference on Neural Information Processing Systems (NeurIPS 2023). 

## **Temporal Complex Query** 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2023_TFLEX_Temporal_Feature-Logic_Embedding_Framework_for_Complex_Reasoning_over_Temporal_Knowledge_Graph_images/nips_2023_TFLEX_Temporal_Feature-Logic_Embedding_Framework_for_Complex_Reasoning_over_Temporal_Knowledge_Graph.pdf-0002-01.png)


## **Computation Graph** 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2023_TFLEX_Temporal_Feature-Logic_Embedding_Framework_for_Complex_Reasoning_over_Temporal_Knowledge_Graph_images/nips_2023_TFLEX_Temporal_Feature-Logic_Embedding_Framework_for_Complex_Reasoning_over_Temporal_Knowledge_Graph.pdf-0002-03.png)


**----- Start of picture text -----**<br>
After And ?<br>TimeAnd AndNot<br>Before<br>François Hollande be elected as  President of France Xi Jinping make a visit<br>François Hollande step down from President of France Barack Obama make a visit<br>**----- End of picture text -----**<br>


Figure 1: A typical multi-hop temporal complex query and its computation graph: "During François Hollande was the president of France, which countries did Xi Jinping visit but Barack Obama did not visit?". In the computation graph, there are entity set (blue circle), timestamp set (green triangle), time set projection (green arrow), entity set projection (blue arrow) and logical operators (red rectangle). 

on queries over static KGs. These methods can neither handle an entity query involving temporal information and operators, nor answer the timestamp set for a temporal query. 

Temporal knowledge graph (TKG) augments triples in static knowledge graphs with temporal information, represented as _<_ **head, relation, tail, timestamp** _>_ named fact. For example, the fact _<_ **Angela Merkel, make a visit, China, 2010-07-16** _>_ specifies the time when the event happens. Generally speaking, TKGs are more close to real world than static KGs, because most knowledge needs to be updated with time, and static KGs cannot express this change. In recommendation systems, TKGs are used to model user behaviors, which includes liking, buying, reading, commenting and so on. In financial applications, TKGs involve stock holding behaviors, trading behaviors, and financial events. These applications require reasoning over TKGs to answer complex queries. However, recent researches in TKGs focus on temporal link prediction, which is simply single-hop. A complex logical query involving multiple facts for multi-hop reasoning is not fully explored yet. 

To fill the gap, we introduce a temporal multi-hop logical reasoning task over TKGs. The task aims to answer temporal complex queries, which have two main distinctions from existing queries over static KGs. Firstly, the answer sets for queries over TKGs are either entity sets or timestamp sets, while that for existing queries over static KGs can only be entity sets. Secondly, as temporal information is included in the query, temporal operators such as **After** , **Before** should be considered apart from FOL operators. To understand this new task, we provide the definition of temporal complex query in Section 3. In addition, an example of a temporal complex query is shown in Figure 1. This example query pertains to the financial event of China’s visit and may be of interest to financial analysts. 

According to the definition of the temporal complex query, we then generate datasets of temporal complex queries on three popular TKGs, and propose the first temporal complex query embedding framework, Temporal Feature-Logic Embedding framework (TFLEX) to answer these queries. In our framework, embeddings of objects (entity, query, timestamp) are divided into two parts, the entity part, and the timestamp part. The entity part handles the FOL operations over entities, while the timestamp part copes with the temporal operations over timestamps. Each part is further divided into feature and logic components. On the one hand, the computation of the logic components follows fuzzy logic, which enables our framework to handle all FOL operations. On the other hand, feature components are mingled and transformed under the guidance of logic components, thereby integrating logical information into the feature. Moreover, we extend fuzzy logic to support extra temporal operations ( **After** , **Before** and **Between** ) to handle temporal operations in the queries. 

The contributions of our work are summarized as follows: (1) For the first time, the definition of the task of multi-hop logical reasoning over TKGs is given. (2) We propose the first multi-hop logical reasoning framework on TKGs, namely Temporal Feature-Logic Embedding framework (TFLEX), 

2 

which supports all FOL operations and extra temporal operations ( **After** , **Before** and **Between** ). (3) We generate three new TKG datasets for the task of multi-hop logical reasoning. Experiments on three generated datasets demonstrate the efficacy of the proposed framework. The source code of our framework and the datasets are available online *. 

## **2 Related Work** 

**Complex Query Embedding** . It learns embeddings of queries and entities, and the answer entities are close to queries in the embedding space. Existing methods utilize a lot of objects to create the embeddings, such as (1) probability distribution [3, 5] (2) geometric object [1, 2, 4, 6, 7] (3) fuzzy logic [8–10] and (4) others [11–13]. However, existing embedding-based methods are considered on static KGs. They cannot utilize temporal information in the TKGs, and therefore cannot handle temporal queries on a temporal KG. Firstly, static query embeddings (QEs) are built over (s, r, o) triples instead of (s, r, o, t) quartets, thus ignoring the timestamps for temporal complex reasoning. The second reason is the order property of timestamps, which is on the contrary that entities are unordered, leading to that static QEs are unable to handle **Before** and **After** temporal logic. In addition, we also notice the semantic conflict in experiments (see section 5.2) when concatenating the geometric embedding (static QE) with the fuzzy embedding (that can handle temporal logic) to promote the static QE to temporal one. Therefore, it is challenging for static QEs to utilize temporal information in the TKGs. 

**Temporal Knowledge Graph Completion (TKGC)** . It aims at inferencing new facts in the TKGs. Existing TKGC methods could be categorized to (1) tensor decomposition [14–16], (2) timestampbased transformation [17–21], (3) dynamic embedding [22–25], (4) Markov process models [26, 27], (5) autoregressive models [28–30], (6) others [31–33] and so on. Among these works, most of them only confined to the one-hop link prediction task, also known as one-hop reasoning. Some works [25, 28–30, 32] can perform multi-hop reasoning via a path consisting of connected quartets. But none of them could answer logical queries that involve multiple logical operations (conjunction, negation and disjunction). In this paper, we focus on the temporal complex query answering task, which is more challenging than TKGC task. 

## **3 Definitions** 

**Temporal Knowledge Graph (TKG)** _G_ = _{V, R, T , F}_ consists of entity set _V_ , relation set _R_ , timestamp set _T_ and fact set _F_ = _{_ ( _s, r, o, t_ ) _} ⊆V × R × V × T_ containing subject-predicateobject-timestamp ( _s, r, o, t_ ) quartets. Without loss of generality, _G_ is a first-order logic knowledge base, where each quartet ( _s, r, o, t_ ) denotes an atomic formula _r_ ( _s, o, t_ ), with _r_ a predicate and _s, o, t_ its arguments. 

**Multi-hop Logical Reasoning over TKG** is the task to answer Temporal Complex Query _q_ when given a TKG _G_ . We focus on Existential Positive First-Order (EPFO) query [34] over TKG, namely **Temporal Complex Query** _q_ , which is categorized into entity query and timestamp query. Formally, the query _q_ consists of a target variable _A_ , a non-variable anchor entity set _Va ⊆V_ , a non-variable anchor timestamp set _Ta ⊆T_ , bound variables _V_ 1 _, · · · , Vk_ and _T_ 1 _, · · · , Tl_ , logical operations (existential quantification _∃_ , conjunction _∧_ , disjunction _∨_ , identity 1, negation _¬_ ), and extra temporal operations ( **After** , **Before** ). **After** ( _t_ 1 _, t_ 2) means _t_ 2 is after _t_ 1, while **Before** ( _t_ 1 _, t_ 2) means _t_ 2 is before _t_ 1. Inspired by [2, 3], the disjunctive normal form (DNF) of temporal query _q_ is defined as: 

_q_ [ _A_ ] = _A_ : _∃V_ 1 _, · · · , Vk, T_ 1 _, · · · , Tl_ : ( _e_[1] 1 _[∧· · · ∧][e]_[1] _n_ 1[)] _[ ∨· · · ∨]_[(] _[e]_ 1 _[m][∧· · · ∧][e][m] nm_[)] where _e_ = _f ◦ r_ ( _Vs, Vo_ or _A, T_ ) or _f ◦ r_ ( _Vs_ or _A, Vo, T_ ) or _g_ ( _Ti, Tj_ ) if _q_ is entity query, _e_ = _f ◦ r_ ( _Vs, Vo, T_ or _A_ ) or _g_ ( _Ti, Tj_ ) or _g_ ( _T, A_ ) or _g_ ( _A, T_ ) if _q_ is timestamp query with _Vs, Vo ∈ Va ∪{V_ 1 _, · · · , Vk}, T, Ti, Tj ∈ Ta ∪{T_ 1 _, · · · , Tl}, r ∈R, f ∈{_ 1 _, ¬}, g ∈{_ **After** _,_ **Before** _}_ 

In the equation, the DNF is a disjunction of _m_ conjunctions, where _e[j]_ 1 _[∧· · ·∧][e] n[j] j_[denotes a conjunction] between _nj_ logical atoms, and each _e[j] i_[denotes a logical atom.][We ignore indices in the definition] 

> * https://github.com/LinXueyuanStdio/TFLEX 

3 

of _e[j] i_[to keep the formula clean.][The goal of answering the query] _[ q]_[ is to find the set of entities (or] timestamps) � _q_ � that satisfy the query, such that _A ∈_ � _q_ � iff _q_ [ _A_ ] holds true, where � _q_ � is the answer set of the query _q_ . 

Following existing static query embedding works, we introduce **Computation Graph** , which is a directed acyclic graph (DAG) representing the structure of temporal complex query. Its nodes represent entity/timestamp sets _S ⊆ Va ∪ V ∪ Ta ∪ T_ , while directed edges represent logical or relational operations acting on these sets. A computation graph specifies how the reasoning of the query has proceeded on the TKG. Starting from anchor sets, we obtain the answer set after applying operations iteratively on non-answer sets according to the directed edges in the computation graph. The edge types on the computation graph are defined as follows: 

1. Relational Projection _P_ . Given an entity set _S_ 1 _⊆V_ , a timestamp set _S_ 2 _⊆T_ (or an entity set _S_ 2 _⊆V_ for entity projection) and a relation _r ∈R_ , projection operation maps _S_ 1 and _S_ 2 _′ ′_ 

to another set: _S[′]_ = _∪_ ( _v∈S_ 1 _,t∈S_ 2) _{v |_ ( _v, r, v , t_ ) _∈F}, P_ is entity projection � _∪_ ( _v∈S_ 1 _,v′∈S_ 2) _{t|_ ( _v, r, v[′] , t_ ) _∈F}, P_ is timestamp projection 

2. Intersection _I_ (Union _U_ , etc.). Given entity sets or timestamp sets _{S_ 1 _, · · · , Sn}_ , the intersection (union, etc.) operation computes logical intersection (union, etc.) of these sets _∩[n] i_ =1 _[S][i]_[(] _[∪][n] i_ =1 _[S][i]_[, etc.).] 

3. Complement/Negation _C_ . The complement set of a given set _S_ is _S_[¯] = 

   - _V − S, S ⊆V_ 

   - � _T − S, S ⊆T_ 

4. Extended temporal operators _f_ . Given a timestamp set _S_ , extended operators compute a certain set of timestamps _S[′]_ : _S[′]_ = _{t′|_ for some _t′ ∈T , t′ >_ max( _S_ ) _}, f_ is After � _{t[′] |_ for some _t[′] ∈T , t[′] <_ min( _S_ ) _}, f_ is Before 

In order to efficiently compute the answer set of a temporal complex query, we consider embedding the query set into a low-dimensional vector space, where the answer set is also represented by a continuous embedding vector. Formally, the **Temporal Query Embedding V** _q_ of a query _q_ is a continuous embedding vector, generated by executing operations according to the computation graph, starting from the temporal embeddings of anchor entity or timestamp sets. The **Temporal Query Answer** to the query _q_ is the entity _v_ (or timestamp _t_ ) whose embedding **v** (or **t** ) has the smallest distance _dist_ ( **v** _,_ **V** _q_ ) (or _dist_ ( **t** _,_ **V** _q_ )) to the embedding of query _q_ . 

## **4 Method** 

In this section, we replace the variables in the query formulation with temporal feature-logic embeddings, and perform logical operations via neural networks. We first introduce the temporal feature-logic embedding for entities, timestamps, and queries in Section 4.1. Afterwards, we introduce logical operators in Section 4.2 and how to train the model in Section 4.3. 

## **4.1 Temporal Feature-Logic Embeddings for Queries and Entities** 

In this section, we design temporal embeddings for queries, entities and timestamps. In general, the answers to queries may be entities or timestamps. Therefore, we consider a part of the embedding as an entity part to answer entities, while the other is the timestamp part to answer timestamps. Formally, the embedding of query set � _q_ � is **V** _q_ = ( _**q** f[e][,]_ _**[ q]** l[e][,]_ _**[ q]** f[t][,]_ _**[ q]** l[t]_[)][ where] _**[ q]** f[e][∈]_[R] _[d]_[ is entity feature,] _**[ q]** l[e][∈]_[[0] _[,]_[ 1]] _[d]_ is entity logic, _**q** f[t][∈]_[R] _[d]_[is time feature,] _**[ q]** l[t][∈]_[[0] _[,]_[ 1]] _[d]_[is time logic respectively,] _[ d]_[ is the embedding] dimension. The parameter _**q** l_ is the uncertainty _**q** l⃗s_ + (1 _−_ _**q** l_ ) _⃗n_ of the feature, according to fuzzy logic. An entity _v ∈V_ is a special query without uncertainty. We propose to represent an entity as the query with logic part **0** , which indicates that the entity’s uncertainty is 0. Formally, the embedding of entity _v_ is **v** = ( _**v** f[e][,]_ **[ 0]** _[,]_ **[ 0]** _[,]_ **[ 0]**[)][, where] _**[ v]** f[e][∈]_[R] _[d]_[is the entity feature part and] **[ 0]**[ is a] _[ d]_[-dimensional vector] with all elements being 0. Similarly, the embedding of timestamp _t_ is **t** = ( **0** _,_ **0** _,_ _**t**[t] f[,]_ **[ 0]**[)][ with entity] part and time logic being **0** . 

Attention that we introduce vector logic, which is a type of fuzzy logic over vector space, to cope with logical transformation in the vector space. Fuzzy logic is a generalization of Boolean logic, such that the truth value of a logical atom is a real number in [0 _,_ 1]. In comparison, as a generalization 

4 

of a real number, the truth value in vector logic is a vector [0 _,_ 1] _[d]_ in the semantic space. We denote the logical operations in vector logic as **AND** ( _∧_ ) _,_ **OR** ( _∨_ ) _,_ **NOT** ( _¬_ ), and so on, which receive one or multiple vectors and output one vector as answer. For more details about fuzzy logic, please refer to Appendix A.1. 

## **4.2 Logical Operators for Temporal Feature-Logic Embeddings** 

In this section, we introduce the designed neural logical operators, including projection, intersection, complement, and all other dyadic operators. 

**Projection Operator** _Pe_ **and** _Pt_ . The goal of operator _Pe_ is to map an entity set to another entity set under a given relation and a given timestamp, while operator _Pt_ outputting timestamp set given relation and two entity queries. We define a function _Pe_ : **V** _q,_ **r** _,_ **V** _t �→_ **V** _q[′]_[in][the][embedding] space to represent _Pe_ , and _Pt_ : **V** _q_ 1 _,_ **r** _,_ **V** _q_ 2 _�→_ **V** _q[′]_[for] _[P][t]_[respectively.][To][implement] _[P][e]_[and] _Pt_ , we first represent relations as translations on query embeddings and assign each relation with relational embedding **r** = ( _**r** f[e][,]_ _**[ r]** l[e][,]_ _**[ r]** f[t][,]_ _**[ r]** l[t]_[)][.][We follow the assumption of translation-based methods:] _qo ≈ qs_ + _r_ + _t_ . As a comparison, static KGE TransE [35] has _o ≈ s_ + _r_ , and temporal KGE TTransE [36] has _ot ≈ st_ + _r_ . The addition represents a semantic translation starting from the source entity set, following the relation and timestamp conditioning, ending at the target entity set. Therefore, we define _Pe_ and _Pt_ as: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2023_TFLEX_Temporal_Feature-Logic_Embedding_Framework_for_Complex_Reasoning_over_Temporal_Knowledge_Graph_images/nips_2023_TFLEX_Temporal_Feature-Logic_Embedding_Framework_for_Complex_Reasoning_over_Temporal_Knowledge_Graph.pdf-0005-04.png)


where **MLP** : R[4] _[d] →_ R[4] _[d]_ is a multi-layer perception network (MLP), + is element-wise addition and _g_ is an activate function to generate _**q** l[e][∈]_[[0] _[,]_[ 1]] _[d][,]_ _**[ q]** l[t][∈]_[[0] _[,]_[ 1]] _[d]_[.][We use MLP and activation function] _g_ ( _._ ) to make projection operator output a valid query embedding, which shows the closure property of the operator. _Pe_ and _Pt_ do not share parameters so the MLPs are different. We define _g_ as: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2023_TFLEX_Temporal_Feature-Logic_Embedding_Framework_for_Complex_Reasoning_over_Temporal_Knowledge_Graph_images/nips_2023_TFLEX_Temporal_Feature-Logic_Embedding_Framework_for_Complex_Reasoning_over_Temporal_Knowledge_Graph.pdf-0005-06.png)


where **x** [0 : _d_ ] is the slice containing element _xi_ of vector **x** with index 0 _≤ i < d_ , _σ_ ( _·_ ) is Sigmoid function and [ _·_ ; _·_ ] is the concatenation operator. 

**Dyadic Operators** . There are two types of dyadic operators for our framework. One for entity set and the other for timestamp set. Each type includes intersection (AND), union (OR), the symmetric difference (XOR), etc. With the help of fuzzy logic, our framework can model all dyadic operators directly. Below we take a unified way to build these operators. 

We start from intersection operators _Ie_ (on entity set) and _It_ (on timestamp set). The goal of intersection operator _Ie_ ( _It_ ) is to represent � _q_ � = _∩[n] i_ =1[�] _[q][i]_[�][based][on][their][entity][parts][(times-] tamp parts). Suppose that **V** _qi_ = ( _**q** i,f[e][,]_ _**[ q]** i,l[e][,]_ _**[ q]** i,f[t][,]_ _**[ q]** i,l[t]_[)][is][temporal][feature-logic][embedding][for] � _qi_ �. We notice that there exists **Alignment Rule** in the process of reasoning. When performing entity set intersection _Ie_ , we should also perform intersection on timestamp parts in order to align the entities into the same time set. The same also holds for timestamp set intersection _It_ and all other dyadic operators. Therefore, we firstly define the intersection operators as follows: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2023_TFLEX_Temporal_Feature-Logic_Embedding_Framework_for_Complex_Reasoning_over_Temporal_Knowledge_Graph_images/nips_2023_TFLEX_Temporal_Feature-Logic_Embedding_Framework_for_Complex_Reasoning_over_Temporal_Knowledge_Graph.pdf-0005-10.png)


where **AND** is the conjunction in fuzzy logic, _**α** i_ and _**β** i_ are attention weights. To notice the changes of logic, we compute _**α**[e,t] i_ and _**β** i[e,t]_ via the following attention mechanism: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2023_TFLEX_Temporal_Feature-Logic_Embedding_Framework_for_Complex_Reasoning_over_Temporal_Knowledge_Graph_images/nips_2023_TFLEX_Temporal_Feature-Logic_Embedding_Framework_for_Complex_Reasoning_over_Temporal_Knowledge_Graph.pdf-0005-12.png)


5 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2023_TFLEX_Temporal_Feature-Logic_Embedding_Framework_for_Complex_Reasoning_over_Temporal_Knowledge_Graph_images/nips_2023_TFLEX_Temporal_Feature-Logic_Embedding_Framework_for_Complex_Reasoning_over_Temporal_Knowledge_Graph.pdf-0006-00.png)


Figure 2: The computation of time part in temporal operators Before and After. 

wherenetwork will learn the hidden information from entity logic and leverage to entity feature, while the **MLP** _[e,t]_ 1 _,_ 2[:][ R][2] _[d][→]_[R] _[d]_[ are MLP networks,][ [] _[·]_[;] _[ ·]_[]][ is concatenation.][The first self-attention neural] second one gathers logical information from time logic to time feature. Note that the computation of entity logic, and time logic obeys the law of fuzzy logic, without any extra learnable parameters. In this way, all dyadic operators can be generated from fuzzy logic in our framework. Due to space limitation, we present the union, exclusive or, implication operators and so on in Appendix A.3. **Complement Operators:** _Ce_ **and** _Ct_ . The aim of _Ce_ is to identify the complement of query set � _q_ � such that � _¬q_ � = _V\_ � _q_ �, while _Ct_ aims to calculate the complement � _¬q_ � = _T \_ � _q_ � by the time parts. Suppose that **V** _q_ = ( _**q** f[e][,]_ _**[ q]** l[e][,]_ _**[ q]** f[t][,]_ _**[ q]** l[t]_[)][, we define the complement operator] _[ C][e]_[and] _[ C][t]_[ as:] 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2023_TFLEX_Temporal_Feature-Logic_Embedding_Framework_for_Complex_Reasoning_over_Temporal_Knowledge_Graph_images/nips_2023_TFLEX_Temporal_Feature-Logic_Embedding_Framework_for_Complex_Reasoning_over_Temporal_Knowledge_Graph.pdf-0006-03.png)


where _f_ not _[e]_[(] _**[q]**[f]_[)][=][tanh(] **[MLP]**[3][([] _**[q]** f[e]_[;] _**[ q]** l[e]_[]))] _[, f]_ not _[ t]_[(] _**[q]** f[t]_[)][=][tanh(] **[MLP]**[4][([] _**[q]** f[t]_[;] _**[ q]** l[t]_[]))][ are feature negation] functions, two **MLP** 3 _,_ 4 : R[2] _[d] →_ R _[d]_ are MLP networks, **NOT** is negation in fuzzy logic. **Temporal Operators: After** _At_ **, Before** _Bt_ **and Between** _Dt_ . The operator After _At_ : **V** _q �→_ **V** _q[′]_ (Before _Bt_ : **V** _q �→_ **V** _q[′]_[) aims to deduce the timestamps after(before) a given fuzzy time set][ �] _[q]_[�][.][Let] **V** _q_ = ( _**q** f[e][,]_ _**[ q]** l[e][,]_ _**[ q]** f[t][,]_ _**[ q]** l[t]_[)][, we define] _[ A][t]_[ and] _[ B][t]_[ as:] 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2023_TFLEX_Temporal_Feature-Logic_Embedding_Framework_for_Complex_Reasoning_over_Temporal_Knowledge_Graph_images/nips_2023_TFLEX_Temporal_Feature-Logic_Embedding_Framework_for_Complex_Reasoning_over_Temporal_Knowledge_Graph.pdf-0006-05.png)


The entity part does not change after computation because temporal operator only affects the time part (time feature _**q** f[t]_[and time logic] _**[ q]** l[t]_[).][The motivation of computation is illustrated in Figure][ 2][.][Since] _**q** l[t]_[is the uncertainty of time feature] _**[ q]** f[t]_[, the time part can be viewed as an interval][ [] _**[q]** f[t][−]_ _**[q]** l[t][,]_ _**[ q]** f[t]_[+] _**[ q]** l[t]_[]] whose center is _**q** f[t]_[and][half-length][is] _**[q]** l[t]_[.][The][interval][is][covered][by][[] _**[q]** f[t][−]_[1] _[,]_ _**[ q]** f[t]_[+][1]][because] the probability _**q** l[t][<]_[1][.][Then,][after][interval][[] _**[q]** f[t][−]_ _**[q]** l[t][,]_ _**[ q]** f[t]_[+] _**[ q]** l[t]_[]][is][the][interval][[] _**[q]** f[t]_[+] _**[ q]** l[t][,]_ _**[ q]** f[t]_[+ 1]] _l l_ whose center is _**q** f[t]_[+][1+] 2 _**[q]**[t]_ and half-length is[1] _[−]_ 2 _**[q]**[t]_ , which gives the time part of embedding _At_ ( **V** _q_ ). _l l_ Similarly, the time part of embedding _Bt_ ( **V** _q_ ) is _**q** f[t][−]_[1+] 2 _**[q]**[t]_ (time feature) and[1] _[−]_ 2 _**[q]**[t]_ (time logic), which are generated from [ _**q** f[t][−]_[1] _[,]_ _**[ q]** f[t][−]_ _**[q]** l[t]_[]][before][[] _**[q]** f[t][−]_ _**[q]** l[t][,]_ _**[ q]** f[t]_[+] _**[q]** l[t]_[]][.][The][operator][Between] _Dt_ : **V** _q_ 1 _,_ **V** _q_ 2 _�→_ **V** _q[′]_[inferences][the][time][set][after][�] _[q]_[1][�][and][before][�] _[q]_[2][�][.][Therefore,][we][define] Between _Dt_ as _Dt_ ( **V** _q_ 1 _,_ **V** _q_ 2) = _It_ ( _At_ ( **V** _q_ 1) _, Bt_ ( **V** _q_ 2)) to output the time between � _q_ 1� and � _q_ 2�. 

## **4.3 Learning Temporal Feature-Logic Embeddings** 

We expect the model to achieve high scores for the answers to the given query _q_ , and low scores for _v[′] ∈/_ � _q_ �. Therefore, we firstly define a distance function to measure the distance between a given answer embedding and a query embedding, and then we train the model with negative sampling loss. 

**Distance Function** Given an entity embedding **v** = ( _**v** f[e][,]_ **[ 0]** _[,]_ **[ 0]** _[,]_ **[ 0]**[)][,][a][timestamp][embedding] **[t]**[=] ( **0** _,_ **0** _,_ _**t**[t] f[,]_ **[ 0]**[)][ and a query embedding] **[V]** _[q]_[=][(] _**[q]** f[e][,]_ _**[ q]** l[e][,]_ _**[ q]** f[t][,]_ _**[ q]** l[t]_[)][,][we define the distance] _[d]_[ between the] answer and the query _q_ as the sum of the feature distance (between the feature parts) and the logic part (to expect uncertainty to be 0). If the query answers entities, the distance is _d[e]_ ( **v** ; **V** _q_ ) = _∥_ _**v** f[e][−]_ _**[q]** f[e][∥]_[1][ +] _**[ q]** l[e]_[.][Otherwise, the distance is] _[ d][t]_[(] **[t]**[;] **[ V]** _[q]_[)][=] _[∥]_ _**[t]**[t] f[−]_ _**[q]** f[t][∥]_[1][ +] _**[ q]** l[t]_[for queries answering] timestamp set, where _∥· ∥_ 1 is the _L_ 1 norm and + is element-wise addition. The distance function aims to optimize two losses. One is to push the answers to the neighbor of query in the embedding 

6 

space. It corresponds to the term L1 distance between answer and query. The other is to reduce the uncertainty of the query (the probability interpretation of the logic part), to make the answers more accurate. 

**Loss Function** Given a training set of queries, we optimize a negative sampling loss 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2023_TFLEX_Temporal_Feature-Logic_Embedding_Framework_for_Complex_Reasoning_over_Temporal_Knowledge_Graph_images/nips_2023_TFLEX_Temporal_Feature-Logic_Embedding_Framework_for_Complex_Reasoning_over_Temporal_Knowledge_Graph.pdf-0007-02.png)


where _γ >_ 0 is a fixed margin, _k_ is the number of negative entities, and _σ_ ( _·_ ) is the sigmoid function. When query _q_ is answering entities (timestamps), _v ∈_ � _q_ � is a positive entity (timestamp), _vi[′][∈][/]_[�] _[q]_[�][is] the _i_ -th negative entity (timestamp). 

## **5 Experiments** 

In this section, we evaluate the ability of TFLEX to reason over TKGs. We first introduce experimental settings in Section 5.1, and then present the experimental results in Section 5.2. 

## **5.1 Experimental Settings** 

**Datasets and Query Generation** Experiments are performed on three new datasets generated from standard benchmarks for TKGC: ICEWS14 [37], ICEWS05-15 [37], and GDELT-500 [38] (with statistics in Appendix B.1). We predefined 40 kinds of complex queries for each dataset. The definition of the 40 kinds of complex queries and the query generation process details are described in Appendix B.2. We consider 27 kinds of queries for training and all 40 kinds for evaluation and testing. Please refer to Appendix B.3 for summaries of the final datasets. 

To briefly summarize the results, we aggregate groups of queries that can be answered: entities (avg _e_ ), timestamps (avg _t_ ), entities with negation (avg _e,Ce_ ), timestamps with negation (avg _t,Ct_ ), entities with unseen union (avg _{Ue}_ ), timestamps with unseen union (avg _{Ut}_ ), and other hybrid unseen structures (avg _x_ ). These groups are inspired by the experiment settings of existing static QEs [1–4]. The detail that which query belongs to which group will be shown in Appendix B.7. Note that the training set only involves 4 groups of queries: avg _e_ , avg _t_ , avg _e,Ce_ , and avg _t,Ct_ . 

**Evaluation** Given a test query _q_ , for each non-trivial answer _v ∈_ � _q_ �test _−_ � _q_ �valid of the query _q_ , we rank it against non-answer entities _V −_ � _q_ �test (or non-answer timestamps _T −_ � _q_ �test if the query is answering timestamps). Then we calculate Mean Reciprocal Rank (MRR) based on the rank. The higher, the better. Please refer to Appendix B.5 for the definition of MRR. 

## **5.2 Main Results** 

For each group, we report the average MRR in Table 1. The raw MRR results on all query structures for each dataset in detail are presented in Appendix B.7. 

**How well can TFLEX answer temporal complex queries?** . We compare TFLEX with state-of-theart query embeddings Query2box [2], BetaE [3] and ConE [4] on answering entities. Existing query embeddings only handle FOL on entity set, but unable to cope with temporal logic over timestamp set. Therefore, the results of these three methods have to be obtained by ignoring the timestamps, so that the avg _t_ , avg _t,Ct_ , avg _{Ut}_ and avg _x_ of three methods are zeros. Comparing the results of these methods in Table 1, we can see that TFLEX outperforms all the baselines on all the metrics. These results demonstrate that TFLEX can perform reasoning over TKGs well, at least better than the existing query embeddings. 

**Ablation on entity part** . The variant **X(ConE)** replaces the entity part of TFLEX with ConE [4] to handle the logic over entity sets. In the entity part, ConE is geometric while TFLEX is fuzzy. The variant performs even worse than TFLEX and ConE. The dropping MRR indicates that the entity part plays an important role in the framework. Considering the performance of ConE, we think there is semantic conflict between the time part of TFLEX and the entity part of ConE. Simply combining static QEs with dynamic QEs is not a clever way to achieve the best performance. 

7 

Table 1: Average MRR results for different groups of temporal complex queries. **X** denotes the variant of TFLEX. **X(ConE)** replaces the entity part with ConE [4]. **FLEX** ablates the time part. **X-1F** merges entity feature and timestamp feature into one feature. **X-logic** removes the logic part. 

|**Dataset**|**Metrics**|**Query2box**|**BetaE**|**ConE**|**TFLEX**|**X(ConE)**|**FLEX**|**X-1F**|**X-logic**|
|---|---|---|---|---|---|---|---|---|---|
||avg_e_|25.06|37.19|41.94|56.79|40.93|43.67|56.89|56.64|
||avg_e,Ce_||36.69|44.88|50.82|42.15|44.41|49.78|51.17|
||avg_t_||||17.56|16.41||18.77|18.03|
|ICEWS14|avg_t,Ct_<br>avg_{Ue}_||19.95|26.47|36.37<br>35.74|35.24<br>25.46|29.25|37.73<br>34.48|36.39<br>34.68|
||avg_{Ut}_||||26.24|24.07||28.04|26.36|
||avg_x_||||28.03|26.65||29.31|28.61|
||**AVG**||||35.93|30.13||36.43|35.98|
||avg_e_|24.00|31.33|40.93|48.99|36.29|38.96|49.90|44.80|
||avg_e,Ce_||29.70|43.52|46.17|38.12|42.10|46.11|41.92|
||avg_t_||||4.39|4.41||4.43|3.29|
|ICEWS05-15|avg_t,Ct_<br>avg_{Ue}_||21.54|43.02|30.16<br>54.37|29.49<br>36.37|44.38|30.26<br>54.05|28.34<br>45.36|
||avg_{Ut}_||||28.69|26.40||27.70|23.39|
||avg_x_||||24.26|21.69||24.41|21.95|
||**AVG**||||33.72|27.54||33.98|29.86|
||avg_e_|9.67|14.75|18.44|19.60|17.83|19.07|17.92|17.36|
||avg_e,Ce_||11.15|12.67|13.52|12.34|13.35|12.13|12.11|
||avg_t_||||5.38|3.16||5.49|5.75|
|GDELT-500|avg_t,Ct_<br>avg_{Ue}_||6.20|6.96|6.31<br>7.58|3.93<br>7.41|7.44|6.50<br>6.92|6.86<br>6.91|
||avg_{Ut}_||||6.71|6.35||6.59|6.80|
||avg_x_||||6.17|6.17||6.47|6.64|
||**AVG**||||9.32|8.17||8.86|8.92|



**Ablation on time part** . The variant **FLEX** removes the time part of the temporal feature-logic embedding. Then, **FLEX** can only answer entities. The results show that **FLEX** slightly outperforms ConE. However, the performance of **FLEX** is worse than TFLEX with a large margin on all the datasets. Therefore, we conclude that the time part also plays an important role in the framework. 

**Ablation on feature part** . If we remove the entity and timestamp feature, the embeddings of entities and timestamps will crash to zeros. Instead, we consider another way to explore the impact of the feature part. Noticing that some TKGC approaches [36, 39] embed entities and timestamps into the same semantic space, we propose **X-1F** to merge the entity and timestamp features into one feature. Compared with TFLEX, **X-1F** achieves higher scores on ICEWS14 and ICEWS05-15, but lower on GDELT. The results imply that unifying the feature of entity and timestamp is potentially beneficial in some datasets. 

**Ablation on logic part** . The variant **X-logic** removes both entity logic and time logic. It achieves lower scores than TFLEX on all the datasets. This is because the logic part is responsible for reasoning over TKGs. Removing the logic part results in that neural logical operators completely rely on neural network to learn the logic, which is not enough to handle various temporal complex queries. 

**Out-of-data reasoning** . The results on avg _{Ue},_ avg _{Ut},_ avg _x_ support that the framework can reason over unseen entity logic, unseen time logic as well as their hybrid. The entity union and time union operators are not included in the training set, but the framework can still handle them well. 

**Sensitive analysis** . (1) The selection of hyperparameters (embedding dimension _d_ , the margin _γ_ ) has a substantial influence on the effectiveness of TFLEX. We train the model with embedding dimension _d ∈{_ 300 _,_ 400 _,_ 500 _,_ 600 _,_ 700 _,_ 800 _,_ 900 _,_ 1000 _}_ , the margin _γ ∈{_ 5 _,_ 10 _,_ 15 _,_ 20 _,_ 25 _,_ 30 _,_ 35 _,_ 40 _}_ . The best performance is achieved when _d_ = 800 and _γ_ = 15. Too small and too large _d, γ_ both lead to worse results, reported in Appendix B.6. (2) Besides, we also investigate the stability of TFLEX. We train and test for five times with different random seeds and report the error bars in Appendix B.6. The small standard variances demonstrate that the performance of TFLEX is stable. 

**Comparison between TFLEX and TKGC methods** . It’s natural to compare TFLEX with SOTA TKGC methods, since all of them can answer one-hop temporal queries. We present the comparison results in Table 2. We can see that TFLEX is competitive with translation-based methods 

8 

Table 2: TKGC Results (%) on ICEWS14, ICEWS05-15, and GDELT. The results from top to bottom are organized as static KGEs, timestamp-based transformation TKGEs, tensor decomposition, autoregressive models and ours. Best results are in bold. _†, ⋆_ indicate the results taken from [40, 17]. Other results are the best numbers reported in their respective paper. 

|Model|ICEWS14<br>MRR<br>Hit@1<br>Hit@3<br>Hit@10|ICEWS05-15<br>MRR<br>Hit@1<br>Hit@3<br>Hit@10|GDELT<br>MRR<br>Hit@1<br>Hit@3<br>Hit@10|
|---|---|---|---|
|TransE_⋆_[35]<br>DistMult_⋆_[41]<br>SimplE_⋆_[42]|28.0<br>9.4<br>-<br>63.7<br>43.9<br>32.3<br>-<br>67.2<br>45.8<br>34.1<br>51.6<br>68.7|29.4<br>9.0<br>-<br>66.3<br>45.6<br>33.7<br>-<br>69.1<br>47.8<br>35.9<br>53.9<br>70.8|11.3<br>0.0<br>15.8<br>31.2<br>19.6<br>11.7<br>20.8<br>34.8<br>20.6<br>12.4<br>22.0<br>36.6|
|ConT [43]<br>TTransE [36]<br>HyTE [44]<br>TA-DistMult [45]<br>DE-TransE [46]<br>DE-DistMult [46]<br>DE-SimplE [46]<br>ChronoR [17]|18.5<br>11.7<br>20.5<br>31.5<br>25.5<br>7.4<br>-<br>60.1<br>29.7<br>10.8<br>41.6<br>65.5<br>47.7<br>36.3<br>-<br>68.6<br>32.6<br>12.4<br>46.7<br>68.6<br>50.1<br>39.2<br>56.9<br>70.8<br>52.6<br>41.8<br>59.2<br>72.5<br>**62.5**<br>**54.7**<br>**66.9**<br>**77.3**|16.3<br>10.5<br>18.9<br>27.2<br>27.1<br>8.4<br>-<br>61.6<br>31.6<br>11.6<br>44.5<br>68.1<br>47.4<br>34.6<br>-<br>72.8<br>31.4<br>10.8<br>45.3<br>68.5<br>48.4<br>36.6<br>54.6<br>71.8<br>51.3<br>39.2<br>57.8<br>74.8<br>**67.5**<br>**59.6**<br>**72.3**<br>**82.0**|14.4<br>8.0<br>15.6<br>26.5<br>11.5<br>0.0<br>16.0<br>31.8<br>11.8<br>0.0<br>16.5<br>32.6<br>20.6<br>12.4<br>21.9<br>36.5<br>12.6<br>0.0<br>18.1<br>35.0<br>21.3<br>13.0<br>22.8<br>37.6<br>23.0<br>14.1<br>24.8<br>40.3<br>-<br>-<br>-<br>-|
|TuckERT [39]<br>TuckERTNT [39]|59.4<br>51.8<br>64.0<br>73.1<br>60.4<br>52.1<br>65.5<br>75.3|62.7<br>55.0<br>67.4<br>76.9<br>63.8<br>55.9<br>68.6<br>78.3|**41.1**<br>**31.0**<br>**45.3**<br>**61.4**<br>38.1<br>28.3<br>41.8<br>57.6|
|RGCRN_†_ [47,40]<br>CyGNet_†_ [48]<br>RE-NET_†_ [28]<br>RE-GCN_†_ [40]|33.3<br>24.0<br>36.5<br>51.5<br>34.6<br>25.3<br>38.8<br>53.1<br>35.7<br>25.9<br>40.1<br>54.8<br>37.7<br>27.1<br>42.5<br>58.8|35.9<br>26.2<br>40.0<br>54.6<br>35.4<br>25.4<br>40.2<br>54.4<br>36.8<br>26.2<br>41.8<br>57.6<br>38.2<br>27.4<br>43.0<br>59.9|18.6<br>11.5<br>19.8<br>32.4<br>18.0<br>11.1<br>19.1<br>31.5<br>19.6<br>12.0<br>20.5<br>33.8<br>19.1<br>11.9<br>20.4<br>33.1|
|TFLEX-1p<br>TFLEX|43.9<br>31.4<br>49.6<br>64.4<br>48.2<br>35.7<br>56.5<br>72.3|40.6<br>29.1<br>47.5<br>66.1<br>43.0<br>30.0<br>49.8<br>69.5|16.5<br>8.6<br>17.3<br>33.1<br>18.5<br>10.1<br>19.6<br>34.9|




![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2023_TFLEX_Temporal_Feature-Logic_Embedding_Framework_for_Complex_Reasoning_over_Temporal_Knowledge_Graph_images/nips_2023_TFLEX_Temporal_Feature-Logic_Embedding_Framework_for_Complex_Reasoning_over_Temporal_Knowledge_Graph.pdf-0009-02.png)


|Table 3: MRR of **Pe** on ICEWS14, ICEWS05-|Table 3: MRR of **Pe** on ICEWS14, ICEWS05-|
|---|---|
|15, and GDELT-500. The best results are|in bold.|
|Results of TTransE and HyTE are taken from Goel||
|et al. [46].||
|**Datasets**<br>TTransE*<br>HyTE*<br>TFLEX-1p|TFLEX|
|**ICEWS14**<br>25.5<br>29.7<br>42.9|**48.2**|
|**ICEWS05-15**<br>27.1<br>31.6<br>40.6|**43.0**|
|**GDELT-500**<br>11.5<br>11.8<br>16.5|**18.5**|



Figure 3: Score distributions of **Pt** , **bPt** and **aPt** . 

(ConT [43], TTransE [36], HyTE [44], etc.), but it doesn’t outperform the SOTA TKGC methods like ChronoR [17] and TuckERT [39]. However, the result doesn’t affect the novelty and contribution of this paper. Please note that the projection operator of TFLEX is as simple as TTransE, not further optimized for TKGC tasks only. Upgrading the projection operator to outperform SOTA TKGC methods remains a future work. 

**Necessity of training on temporal complex queries** . Our experiments demonstrate that training on complex queries is necessary to achieve the best performance. We compare with translation-based _Pe_ operators (TTransE [36], HyTE [44] and TFLEX’s variant **TFLEX-1p** ) using only one-hop **Pe** queries for training. We choose TTransE and HyTE because our projection operator is also translation-based ( _Pe_ ( **V** _q,_ **r** _,_ **V** _t_ ) _∝_ **V** _q_ + **r** + **V** _t_ ). From the result table 3, we observe that TFLEX achieves the best performance when comparing with these translation-based baselines on all datasets. Besides, compared with **TFLEX-1p** , TFLEX achieves 7.9% relative improvement on average on MRR, which demonstrates that training on complex queries could improve the one-hop query-answering ability. 

**Effectiveness of neural temporal operators** . We found that the temporal operators _At_ and _Bt_ change the semantic of predicted timestamp embedding logically. We randomly select an event ( _s, r, o_ ) from ICEWS14 and consider three temporal queries **Pt** , **bPt** and **aPt** . Then, **Pt** (= _Pt_ ( _s, r, o_ )) predicts the date _t_ when this event happened. And **bPt** (= _Bt_ ( _Pt_ ( _s, r, o_ ))) should predict the date before _t_ , while **aPt** (= _At_ ( _Pt_ ( _s, r, o_ ))) is supposed to predict the time after _t_ . The output of three queries are time embeddings. Because the time embedding is fuzzy, we score it to all possible Timestamps, and visualize the normalized similarity score distribution over all days in a year, from 0 to 365 in ICEWS14. The higher the score, the more possible the predicted date is the day. We plot the three score distributions in Figure 3. For each distribution, we highlight the periods of the highest scores 

9 

with a colored background. These colored blocks represent the most likely happening time interval of the event. We observe that the order of colored blocks (corresponding to the predictions of **bPt** , **Pt** , and **aPt** ) matches the logical meanings of these operators (Before the event, On the event, After the event). The visualization shows that neural temporal operators perform the time transformation correctly. More examples are attached in Appendix D. 

**Explaining answers with temporal feature-logic framework** . We take query **Pe2** for example. Given temporal query **Pe2** : _q_ [ _V_ ?] = _V_ ? _, ∃Va, r_ 1( _e_ 1 _, Va, t_ 1) _∧ r_ 2( _Va, V_ ? _, t_ 2), let’s try an example which can be written as: On 2014-04-04, who consulted the man who was appealed to or requested by the Head of Government (Latvia) on 2014-08-01? In this example, _e_ 1 is "Head of Government (Latvia)", _r_ 1 is "Make an appeal or request", _t_ 1 is "2014-08-01", _r_ 2 is "consult _[−]_[1] ", and _t_ 2 is "201404-04". Then we use TFLEX to execute the query and get answers. We classify the answers into easy, hard and wrong ones. The easy answer is the correct answer that appears in the training set, while the hard answer is the correct answer that exists in the testing set instead of training set. 

We present five most likely answers for interpretation in Figure 4. From the table we observe that TFLEX ranks easy answers "François Hollande", "Taavi Rõivas" and hard answer "Andris Berzins" higher than wrong answers "Angela Merkel" and "Head of Government (Latvia)". This shows that TFLEX successfully finds the hard answer by performing complex reasoning, and distinguishes the right answers from the wrong ones. 

We provide 36 examples in Appendix E, including the visualization and intermediate explanation of answers for each query structure. 

|||||**Query Sentence**<br>On 2014-04-04, who consulted the man who was appealed to or<br>requested by the Head of Government (Latvia) on 2014-08-01?|
|---|---|---|---|---|
|<br>**P**||**e2**|||
||||**?**|**Temporal Query**<br>_q_[_V_?] =_V_?_, ∃Va, r_1(_e_1_, Va, t_1)_∧r_2(_Va, V_?_, t_2)|
|||||**Rank**<br>**Query Answers**<br>**Correctness**<br>**Answer Type**|
|||Head of Government (Latvia)<br>Make an appeal or request<br>2014-08-01<br>2014-04-04|||
|||||1<br>François Hollande<br>✔<br>Easy<br>2<br>Taavi Rõivas<br>✔<br>Easy<br>3<br>Jyrki Katainen<br>✔<br>Hard<br>4<br>Angela Merkel<br>✘<br>-<br>5<br>Head of Government (Latvia)<br>✘<br>-|
||||||



Figure 4: Intermediate variable assignments and ranks for example Pe2 query. Correctness indicates whether the answer belongs to the ground-truth set of answers. 

## **6 Conclusion** 

In this paper, we firstly define a temporal multi-hop logical reasoning task on temporal knowledge graphs. Then we generate three new TKG datasets and propose the Temporal Feature-Logic embedding framework, TFLEX, to handle temporal complex queries in datasets. Fuzzy logic is used to guide all FOL transformations on the logic part of embeddings. We also further extended fuzzy logic to implement extra temporal operators ( **Before** , **After** and **Between** ). To the best of our knowledge, TFLEX is the first framework to support multi-hop complex reasoning on temporal knowledge graphs. Experiments on benchmark datasets demonstrate the efficacy of the proposed framework in handling different operations in complex queries. 

## **Acknowledgments and Disclosure of Funding** 

This work is supported by the National Science Foundation of China (Grant No. 62176026) and Beijing Natural Science Foundation (M22009); Engineering Research Center of Information Networks, Ministry of Education. We would like to express our sincere gratitude to our corresponding authors, Prof. Haihong E and Dr. Chengjin Xu, for their guidance, expertise, and unwavering support throughout the project. Furthermore, we would like to thank Fenglong Su, Gengxian Zhou, Tianyi Hu, Ningyuan Li, Mingzhi Sun, and Haoran Luo for their individual contributions and collaboration in various aspects of this project. Their expertise and dedication have significantly enriched the final outcome. Additionally, we extend our thanks to the anonymous reviewers for their time and efforts in reviewing our work. Their constructive feedback and comments have been instrumental in improving the overall clarity and rigor of this research. 

10 

## **Broader Impact** 

Multi-hop reasoning makes the information stored in TKGs more valuable. With the help of multi-hop reasoning, we can digest more hidden and implicit information in TKGs. It will broaden and deepen KG applications in various fields, such as question answering, recommend systems, and information retrieval. It may also bring about risks exposing unexpected personal information on public data. 

Finance temporal knowledge graph is a good example to illustrate the broader impact of multi-hop reasoning. In the financial field, the information stored in TKGs is very sensitive. The one-hop reasoning can complete the hidden financial transaction, while the multi-hop reasoning can help to detect the fraud. The After operator could also be used to predict the future financial transaction. People may take the advantages of logical reasoning to digest financial factor to obtain excess returns. 

Military temporal knowledge graph is another example. With the help of multi-hop logical reasoning, we may predict the future military strategy of a country. Besides, with the fuzzy completion of the hidden military information, we can also detect the hidden militarily moves, which may save thousands of soldiers’ lives. 

The last example is social temporal knowledge graph. The behaviors of people are left and stored in TKGs. With the help of logical reasoning, we may predict a short future of a person. For example, the query may answer the evolution of user profiles: how long may a person transfer to another role, from student to worker, becoming a parent, or being a grandparent. Tracking the user’s identity change can provide super benefits for merchants’ advertising. Detecting the role of a person is also helpful to provide more personalized services. 

However, we should agree that the multi-hop reasoning is still at an extremely early stage, though it may bring about risks. Therefore, we should pay more attention to the security of TKGs and the privacy of users at the mean time when we explore the technology of multi-hop reasoning over TKGs. 

## **Limitation & Future Work** 

In this section, we list 4 limitations and talk about the possible future work. 

**Limitation 1: The temporal operators are not enough.** . We define three extra temporal operators ( **Before** , **After** and **Between** ) in query generation. However, there exists more temporal operators in the real world. For example, Allen [49] defined 13 types of temporal relations represented by two intervals, including before/after, during/contains, overlaps/overlapped-by, meets/met-by, starts/startedby, finishes/finished-by and equal. In the future we would like to promote these temporal operators to TKGs. 

**Limitation 2: The temporal embedding could improve.** . In this paper, the embedding of the timestamp is randomly initialized and finally learned by the model via logical advantages. Such embedding ignores the order of different timestamps. The order property is learned by the After and Before operators, which may be not enough. We recall that in the field of NLP, positional embeddings also have order features, which may be used for the construction of timestamp embeddings. 

**Limitation 3: The query generation is time-consuming.** . There are 40 predefined query structures in our query generation module. Each query structure has 10k+ queries for training. With the scale of TKGs increasing, the number of queries will also increase, even 40x faster. Therefore, we need to find a more efficient way to generate queries for large scale TKGs. 

**Limitation 4: The MRR and Hits@k are weak.** . The MRR and Hits@k evaluation metric may not inflect the performance of multi-hop reasoning. We observe that some queries have lots of answers. When the number of answers is larger than k, the MRR and Hits@k will be low, even if all answers are correctly ranked at top. Because the right answers that ranked after k are labeled as false. The Hits@k decreases with the increase number of right answers, which is not reasonable. This disadvantage prevents us from comparing the performance across different datasets. In this paper, GDELT is much denser, and the count of right answers is larger than the other two datasets. Therefore, the MRR of GDELT is lower than the other two datasets. It can also be verified that on the average MRR, group avg _t_ is lower than group avg _e_ . The reason is that the number of right answers of group avg _t_ is larger than that of group avg _e_ , which can be seen from the statistic of average answer count of queries in Table 10. In the future, there should be a more reasonable evaluation metric for multi-hop reasoning. 

11 

## **References** 

- [1] Will Hamilton, Payal Bajaj, Marinka Zitnik, Dan Jurafsky, and Jure Leskovec. Embedding logical queries on knowledge graphs. _Advances in Neural Information Processing Systems_ , 31:2026–2037, 2018. 

- [2] H Ren, W Hu, and J Leskovec. Query2box: Reasoning over knowledge graphs in vector space using box embeddings. In _International Conference on Learning Representations (ICLR)_ , 2020. 

- [3] Hongyu Ren and Jure Leskovec. Beta embeddings for multi-hop logical reasoning in knowledge graphs. _Neural Information Processing Systems (NeurIPS)_ , 2020. 

- [4] Zhanqiu Zhang, Jie Wang, Jiajun Chen, Shuiwang Ji, and Feng Wu. Cone: Cone embeddings for multi-hop reasoning over knowledge graphs. _Advances in Neural Information Processing Systems_ , 34, 2021. 

- [5] Nurendra Choudhary, Nikhil Rao, Sumeet Katariya, Karthik Subbian, and Chandan Reddy. Probabilistic entity representation model for reasoning over knowledge graphs. _Advances in Neural Information Processing Systems_ , 34, 2021. 

- [6] Nurendra Choudhary, Nikhil Rao, Sumeet Katariya, Karthik Subbian, and Chandan K Reddy. Selfsupervised hyperboloid representations from logical queries over knowledge graphs. In _Proceedings of the Web Conference 2021_ , pages 1373–1384, 2021. 

- [7] Lihui Liu, Boxin Du, Heng Ji, ChengXiang Zhai, and Hanghang Tong. Neural-answering logical queries on knowledge graphs. In _Proceedings of the 27th ACM SIGKDD Conference on Knowledge Discovery & Data Mining_ , pages 1087–1097, 2021. 

- [8] Xueyuan Lin, Haihong E, Gengxian Zhou, Tianyi Hu, Li Ningyuan, Mingzhi Sun, and Haoran Luo. Flex: Feature-logic embedding framework for complex knowledge graph reasoning. _ArXiv_ , abs/2205.11039, 2023. URL https://api.semanticscholar.org/CorpusID:248986747. 

- [9] Erik Arakelyan, Daniel Daza, Pasquale Minervini, and Michael Cochez. Complex query answering with neural link predictors. _arXiv preprint arXiv:2011.03459_ , 2020. 

- [10] Francois P. S. Luus, Prithviraj Sen, Pavan Kapanipathi, Ryan Riegel, Ndivhuwo Makondo, Thabang Lebese, and Alexander G. Gray. Logic embeddings for complex query answering. _ArXiv_ , abs/2103.00418, 2021. 

- [11] Haitian Sun, Andrew Arnold, Tania Bedrax Weiss, Fernando Pereira, and William W Cohen. Faithful embeddings for knowledge base queries. _Advances in Neural Information Processing Systems_ , 33, 2020. 

- [12] Dinesh Garg, Shajith Ikbal, Santosh K Srivastava, Harit Vishwakarma, Hima Karanam, and L Venkata Subramaniam. Quantum embedding of knowledge for reasoning. _Advances in Neural Information Processing Systems_ , 32:5594–5604, 2019. 

- [13] Santosh Kumar Srivastava, Dinesh Khandelwal, Dhiraj Madan, Dinesh Garg, Hima Karanam, and L Venkata Subramaniam. Inductive quantum embedding. _Advances in Neural Information Processing Systems_ , 33, 2020. 

- [14] Lifan Lin and Kun She. Tensor decomposition-based temporal knowledge graph embedding. _2020 IEEE 32nd International Conference on Tools with Artificial Intelligence (ICTAI)_ , pages 969–975, 2020. URL https://api.semanticscholar.org/CorpusID:229701707. 

- [15] Timothée Lacroix, Guillaume Obozinski, and Nicolas Usunier. Tensor decompositions for temporal knowledge base completion. _ArXiv_ , abs/2004.04926, 2020. URL https://api.semanticscholar. org/CorpusID:214390104. 

- [16] Pengpeng Shao, Guohua Yang, Dawei Zhang, Jianhua Tao, Feihu Che, and Tong Liu. Tucker decompositionbased temporal knowledge graph completion. _Knowl. Based Syst._ , 238:107841, 2020. URL https: //api.semanticscholar.org/CorpusID:226965423. 

- [17] Ali Reza Sadeghian, Mohammadreza Armandpour, Anthony Colas, and Daisy Zhe Wang. Chronor: Rotation based temporal knowledge graph embedding. In _AAAI Conference on Artificial Intelligence_ , 2021. URL https://api.semanticscholar.org/CorpusID:232269660. 

- [18] Julien Leblay and Melisachew Wudage Chekol. Deriving validity time in knowledge graph. _Companion Proceedings of the The Web Conference 2018_ , 2018. URL https://api.semanticscholar.org/ CorpusID:13846713. 

12 

- [19] Wessel Radstok and Melisachew Wudage Chekol. Leveraging static models for link prediction in temporal knowledge graphs. _2021 IEEE 33rd International Conference on Tools with Artificial Intelligence (ICTAI)_ , pages 1034–1041, 2021. URL https://api.semanticscholar.org/CorpusID: 235670110. 

- [20] Chengjin Xu, M. Nayyeri, Fouad Alkhoury, Hamed Shariat Yazdi, and Jens Lehmann. Tero: A timeaware knowledge graph embedding via temporal rotation. In _International Conference on Computational Linguistics_ , 2020. URL https://api.semanticscholar.org/CorpusID:222124934. 

- [21] Shib Sankar Dasgupta, Swayambhu Nath Ray, and Partha Talukdar. Hyte: Hyperplane-based temporally aware knowledge graph embedding. In _Proceedings of the 2018 conference on empirical methods in natural language processing_ , pages 2001–2011, 2018. 

- [22] Chengjin Xu, M. Nayyeri, Fouad Alkhoury, Hamed Shariat Yazdi, and Jens Lehmann. Temporal knowledge graph completion based on time series gaussian embedding. In _International Workshop on the Semantic Web_ , 2019. URL https://api.semanticscholar.org/CorpusID:218900866. 

- [23] Zhen Han, Peng Chen, Yunpu Ma, and Volker Tresp. Dyernie: Dynamic evolution of riemannian manifold embeddings for temporal knowledge graph completion. _ArXiv_ , abs/2011.03984, 2020. URL https://api.semanticscholar.org/CorpusID:226262324. 

- [24] Rakshit S. Trivedi, Hanjun Dai, Yichen Wang, and Le Song. Know-evolve: Deep temporal reasoning for dynamic knowledge graphs. In _International Conference on Machine Learning_ , 2017. URL https: //api.semanticscholar.org/CorpusID:8040343. 

- [25] Jiapeng Wu, Mengyao Cao, Jackie Chi Kit Cheung, and William L. Hamilton. Temp: Temporal message passing for temporal knowledge graph completion. In _Conference on Empirical Methods in Natural Language Processing_ , 2020. URL https://api.semanticscholar.org/CorpusID: 222177069. 

- [26] Youri Xu, Haihong E, Meina Song, Wenyu Song, Xiaodong Lv, Wang Haotian, and Yang Jinrui. Rtfe: A recursive temporal fact embedding framework for temporal knowledge graph completion. In _North American Chapter of the Association for Computational Linguistics_ , 2020. URL https: //api.semanticscholar.org/CorpusID:222067007. 

- [27] Siyuan Liao, Shangsong Liang, Zaiqiao Meng, and Qiang Zhang. Learning dynamic embeddings for temporal knowledge graphs. _Proceedings of the 14th ACM International Conference on Web Search and Data Mining_ , 2021. URL https://api.semanticscholar.org/CorpusID:232126110. 

- [28] Woojeong Jin, Meng Qu, Xisen Jin, and Xiang Ren. Recurrent event network: Autoregressive structure inference over temporal knowledge graphs. In _Conference on Empirical Methods in Natural Language Processing_ , 2019. URL https://api.semanticscholar.org/CorpusID:222205878. 

- [29] Zixuan Li, Xiaolong Jin, Wei Li, Saiping Guan, Jiafeng Guo, Huawei Shen, Yuanzhuo Wang, and Xueqi Cheng. Temporal knowledge graph reasoning based on evolutional representation learning. _Proceedings of the 44th International ACM SIGIR Conference on Research and Development in Information Retrieval_ , 2021. URL https://api.semanticscholar.org/CorpusID:233324265. 

- [30] Zhen Han, Zifeng Ding, Yunpu Ma, Yujia Gu, and Volker Tresp. Learning neural ordinary equations for forecasting future links on temporal knowledge graphs. In _Conference on Empirical Methods in Natural Language Processing_ , 2021. URL https://api.semanticscholar.org/CorpusID: 237304520. 

- [31] Zhen Han, Peng Chen, Yunpu Ma, and Volker Tresp. Explainable subgraph reasoning for forecasting on temporal knowledge graphs. In _International Conference on Learning Representations_ , 2021. URL https://api.semanticscholar.org/CorpusID:235614395. 

- [32] Jaehun Jung, Jinhong Jung, and U Kang. Learning to walk across time for interpretable temporal knowledge graph completion. _Proceedings of the 27th ACM SIGKDD Conference on Knowledge Discovery & Data Mining_ , 2021. URL https://api.semanticscholar.org/CorpusID:236979995. 

- [33] Ruijie Wang, Zheng Li, Dachun Sun, Shengzhong Liu, Jinning Li, Bing Yin, and Tarek Abdelzaher. Learning to sample and aggregate: Few-shot reasoning over temporal knowledge graphs. _Advances in Neural Information Processing Systems_ , 35:16863–16876, 2022. 

- [34] Nilesh Dalvi and Dan Suciu. Efficient query evaluation on probabilistic databases. _The VLDB Journal_ , 16: 523–544, 2007. 

13 

- [35] Antoine Bordes, Nicolas Usunier, Alberto García-Durán, Jason Weston, and Oksana Yakhnenko. Translating embeddings for modeling multi-relational data. In _NIPS 2013._ , 2013. 

- [36] Tingsong Jiang, Tianyu Liu, Tao Ge, Lei Sha, Baobao Chang, Sujian Li, and Zhifang Sui. Towards timeaware knowledge graph completion. In _Proceedings of COLING 2016, the 26th International Conference on Computational Linguistics: Technical Papers_ , pages 1715–1724, Osaka, Japan, December 2016. The COLING 2016 Organizing Committee. URL https://aclanthology.org/C16-1161. 

- [37] Elizabeth Boschee, Jennifer Lautenschlager, Sean O’Brien, Steve Shellman, James Starz, and Michael Ward. ICEWS Coded Event Data, 2015. URL https://doi.org/10.7910/DVN/28075. 

- [38] Kalev Leetaru and Philip A. Schrodt. Gdelt: Global data on events, location, and tone. _ISA Annual Convention_ , 2013. 

- [39] Pengpeng Shao, Guohua Yang, Dawei Zhang, Jianhua Tao, Feihu Che, and Tong Liu. Tucker decompositionbased temporal knowledge graph completion. _Knowl. Based Syst._ , 238:107841, 2020. 

- [40] Zixuan Li, Xiaolong Jin, Wei Li, Saiping Guan, Jiafeng Guo, Huawei Shen, Yuanzhuo Wang, and Xueqi Cheng. Temporal knowledge graph reasoning based on evolutional representation learning. _Proceedings of the 44th International ACM SIGIR Conference on Research and Development in Information Retrieval_ , 2021. URL https://api.semanticscholar.org/CorpusID:233324265. 

- [41] Bishan Yang, Scott Wen-tau Yih, Xiaodong He, Jianfeng Gao, and Li Deng. Embedding entities and relations for learning and inference in knowledge bases. In _Proceedings of the International Conference on Learning Representations (ICLR) 2015_ , May 2015. 

- [42] Seyed Mehran Kazemi and David L. Poole. Simple embedding for link prediction in knowledge graphs. In _Neural Information Processing Systems_ , 2018. URL https://api.semanticscholar.org/ CorpusID:3674966. 

- [43] Yunpu Ma, Volker Tresp, and Erik A. Daxberger. Embedding models for episodic knowledge graphs. _J. Web Semant._ , 59, 2018. URL https://api.semanticscholar.org/CorpusID:54444869. 

- [44] Shib Sankar Dasgupta, Swayambhu Nath Ray, and Partha Talukdar. HyTE: Hyperplane-based temporally aware knowledge graph embedding. In _Proceedings of the 2018 Conference on Empirical Methods in Natural Language Processing_ , pages 2001–2011, Brussels, Belgium, October-November 2018. Association for Computational Linguistics. doi: 10.18653/v1/D18-1225. URL https://aclanthology.org/ D18-1225. 

- [45] Alberto García-Durán, Sebastijan Dumancic, and Mathias Niepert. Learning sequence encoders for temporal knowledge graph completion. In _Conference on Empirical Methods in Natural Language Processing_ , 2018. URL https://api.semanticscholar.org/CorpusID:52183483. 

- [46] Rishab Goel, Seyed Mehran Kazemi, Marcus Brubaker, and Pascal Poupart. Diachronic embedding for temporal knowledge graph completion. In _Proceedings of the AAAI Conference on Artificial Intelligence_ , volume 34, pages 3988–3995, 2020. 

- [47] Youngjoo Seo, Michaël Defferrard, Pierre Vandergheynst, and Xavier Bresson. Structured sequence modeling with graph convolutional recurrent networks. In _International Conference on Neural Information Processing_ , 2016. URL https://api.semanticscholar.org/CorpusID:2687749. 

- [48] Cunchao Zhu, Muhao Chen, Changjun Fan, Guangquan Cheng, and Yan Zhan. Learning from history: Modeling temporal knowledge graphs with sequential copy-generation networks. In _AAAI Conference on Artificial Intelligence_ , 2020. URL https://api.semanticscholar.org/CorpusID:229180723. 

- [49] James F. Allen. Maintaining knowledge about temporal intervals. _Commun. ACM_ , 26(11):832–843, nov 1983. ISSN 0001-0782. doi: 10.1145/182.358434. URL https://doi.org/10.1145/182. 358434. 

- [50] Diederick P Kingma and Jimmy Ba. Adam: A method for stochastic optimization. In _Proceedings of the International Conference on Learning Representations (ICLR)_ , 2015. 

14 

## **Appendix** 

## **A More Details about Neural Operators** 

## **A.1 Fuzzy Logic** 

**Fuzzy logic** is a generalization of Boolean logic, which is based on the idea that the truth of a statement can be expressed as a number between 0 and 1. It is a mathematical framework for reasoning with imprecise information. In fuzzy logic, there are also logical operators, such as conjunction, disjunction, negation, implication, equivalence, and exclusive or. Most popular fuzzy logic operators are defined as follows ( _a, b ∈_ [0 _,_ 1]): 

- (1) **conjunction** : _a ∧ b_ = min( _a, b_ ). **disjunction** : _a ∨ b_ = max( _a, b_ ). 

- (2) **negation** : _¬a_ = 1 _− a_ . **exclusive or** : _a ⊕ b_ = min( _a,_ 1 _− b_ ) + min(1 _− a, b_ ). 

- (3) **implication** : _a → b_ = max(1 _− a, b_ ). **equivalence** : _a ↔ b_ = min( _a, b_ ) + max(1 _− a,_ 1 _− b_ ). 

However, the fuzzy logic operators are defined over real space, not suitable for reasoning over vector space. We expect that the fuzzy logic operators can be executed by matrix computation. Because the embedding space of knowledge graph is a vector space. In order to cope with tensor computation in neural networks, we introduce the fuzzy logic operators over vector space in the following, which is called vector logic. 

## **A.2 Vector Logic** 

Vector logic is an elementary logical model based on matrix algebra. In vector logic, true values are mapped to the vector, and logical operators are executed by matrix computation. 

**Truth Value Vector Space** _V_ 2. A two-valued vector logic uses two _d_ -dimensional ( _d ≥_ 2) column vectors _⃗s_ and _⃗n_ to represent true and false in the classic binary logic. The two vectors _⃗s_ and _⃗n_ are real-valued, normally orthogonal to each other, and normalized vectors, _i.e._ , _∥⃗s∥_ = 1 _, ∥⃗n∥_ = 1. Truth value vector space are generated by _V_ 2 = _{⃗s,⃗n}_ , and operations on vectors in truth value space is based on scalar product. 

**Operators** . The basic logical operators are associated with their own matrices by vectors in truthvalue vector space. Two common types of operators are monadic and dyadic. 

(1) **Monadic Operators** are functions: _V_ 2 _→ V_ 2. Two examples are Identity _I_ = _⃗s⃗s[T]_ + _⃗n⃗n[T]_ and Negation _N_ = _⃗n⃗s[T]_ + _⃗s⃗n[T]_ such that _I⃗s_ = _⃗s, I⃗n_ = _⃗n, N⃗n_ = _⃗s, N⃗s_ = _⃗n_ . Note that the truth tables of _I_ and _N_ fulfill the real logical rules of identity and negation in classic boolean logic. 

(2) **Dyadic operators** are functions: _V_ 2 _⊗ V_ 2 _→ V_ 2, where _⊗_ denotes Kronecker product. Dyadic operators include conjunction _C_ , disjunction _D_ , implication IMPL, equivalence _E_ , exclusive or XOR, etc. For example, the conjunction between two logical propositions ( _p ∧ q_ ) is performed by _C_ ( _⃗u ⊗ ⃗v_ ), where _C_ = _⃗s_ ( _⃗s ⊗ ⃗s_ ) _[T]_ + _⃗n_ ( _⃗s ⊗ ⃗n_ ) _[T]_ + _⃗n_ ( _⃗n ⊗ ⃗s_ ) _[T]_ + _⃗n_ ( _⃗n ⊗ ⃗n_ ) _[T]_ . It can be verified that _C_ ( _⃗s ⊗ ⃗s_ ) = _⃗s, C_ ( _⃗s ⊗ ⃗n_ ) = _C_ ( _⃗n ⊗ ⃗s_ ) = _C_ ( _⃗n ⊗ ⃗n_ ) = _⃗n_ . Dyadic operators which correspond to logical operations in classic binary logic are defined by its formulation to perform logical operations on truth value vectors. Their associated matrices has _d_[2] rows and _d_ columns. 

**Many-valued Two-dimensional Logic** . Many-valued logic is introduced to include uncertainties in the logic vector. Weighting _⃗s_ and _⃗n_ by probabilities, uncertainties are introduced: _f[⃗]_ = _ϵ⃗s_ + _δ⃗n_ , where _ϵ, δ ∈_ [0 _,_ 1] _, ϵ_ + _δ_ = 1. Besides, operations on vectors can be simplified to computation on the scalar of these vectors. For example, given two vectors _⃗u_ = _α⃗s_ + _β⃗n,⃗v_ = _α[′] ⃗s_ + _β[′] ⃗n_ , we have: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2023_TFLEX_Temporal_Feature-Logic_Embedding_Framework_for_Complex_Reasoning_over_Temporal_Knowledge_Graph_images/nips_2023_TFLEX_Temporal_Feature-Logic_Embedding_Framework_for_Complex_Reasoning_over_Temporal_Knowledge_Graph.pdf-0015-15.png)


15 

## **A.3 Neural Dyadic Operators** 

In this section, we show that all dyadic operators are generated from fuzzy logic in our framework. Below we take union operator for example. We define entity union operator _Ue_ and time union operator _Ut_ according to **Alignment Rule** 4.2 as follows: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2023_TFLEX_Temporal_Feature-Logic_Embedding_Framework_for_Complex_Reasoning_over_Temporal_Knowledge_Graph_images/nips_2023_TFLEX_Temporal_Feature-Logic_Embedding_Framework_for_Complex_Reasoning_over_Temporal_Knowledge_Graph.pdf-0016-02.png)


where **AND** , **OR** are the conjunction, disjunction operators in fuzzy logic respectively, _αi_ and _βi_ are attention weights, as designed in intersection operators. In the time part of entity union operator _Ue_ and the entity part of time union operator _Ut_ , we follows **Alignment Rule** 4.2 to perform intersection. Besides, please be aware that each operator owns its MLPs and parameters. These operators do not share parameters with each other. 

For _n_ queries to intersection/union, the **AND** and **OR** can be written as: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2023_TFLEX_Temporal_Feature-Logic_Embedding_Framework_for_Complex_Reasoning_over_Temporal_Knowledge_Graph_images/nips_2023_TFLEX_Temporal_Feature-Logic_Embedding_Framework_for_Complex_Reasoning_over_Temporal_Knowledge_Graph.pdf-0016-05.png)


The equations come from the definition of AND and OR in vector logic. The cost of **OR** operator’s implementation is not as expensive as it seems to be. In fact, it’s _O_ ( _n ∗ d_ ) complexity, where _d_ is the embedding dimension. We provide the proof in next section. 

Other dyadic operators can be generated in the same way. Since listing all operators is boring, we provide the last example. Then, the entity implication operator _Le_ and time implication operator _Lt_ are defined as follows: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2023_TFLEX_Temporal_Feature-Logic_Embedding_Framework_for_Complex_Reasoning_over_Temporal_Knowledge_Graph_images/nips_2023_TFLEX_Temporal_Feature-Logic_Embedding_Framework_for_Complex_Reasoning_over_Temporal_Knowledge_Graph.pdf-0016-08.png)


## **A.4 Theoretical Analysis** 

To understand why the feature-logic framework works, we have the following proposition, which shows that the designed intersection and union operators obey commutative law of real logical operations. The proofs are presented in next section in Appendix A.5: 

**Proposition 1.** _**Commutativity** : Given Temporal Feature-Logic embedding Vqa , Vqb, we have Ie_ ( _Vqa , Vqb_ ) = _Ie_ ( _Vqb , Vqa_ ) _and Ue_ ( _Vqa , Vqb_ ) = _Ue_ ( _Vqb , Vqa_ ) _, It_ ( _Vqa , Vqb_ ) = _It_ ( _Vqb , Vqa_ ) _and Ut_ ( _Vqa , Vqb_ ) = _Ut_ ( _Vqb , Vqa_ ) _._ 

## **A.5 Proofs** 

In this part, we prove that TFLEX has the property of commutativity. Besides, we also show that the computation complexity of **OR** operation is linear to the number of queries. 

16 

_Proof._ (Proposition 1) **Commutativity** : 

For the intersection operations, as the calculations of _Ie_ and _It_ are identical, here, we only prove that _Ie_ complies with commutative law. The entity feature part and the time feature part of the result are computed as a weighted summation of each query’s corresponding parts. Since addition is commutative and the attention weights do not concern the order of calculations, both feature parts’ calculations are commutative. 

Then, we discuss the logic parts. The logic parts only include the AND in fuzzy logic which, essentially, is just the multiplication of each element by the definition provided above. Because multiplication is surely commutative, the calculation of either entity logic part or time logic part is commutative. Thus the intersection operation _Ie_ ( _It_ ) is commutative. 

As for _Ue_ and _Ut_ , their feature parts have the same form of weighted summation as the intersection operations do. Thus, the feature parts of both _Ue_ and _Ut_ comply to commutative law. Also, the _time logic part_ of _Ue_ and the _entity logic part_ of _Ut_ solely concern AND operator which has been proved commutative before. The OR operator, by definition, gives the aggregation of different accumulative parts each of which is commutative itself. Also, the multiplications in each of the summations are commutative. Hence, OR operation is invariant to the order of calculations, which finally gives the calculations of _entity logic part_ of _Ue_ and the _time logic part_ of _Ut_ are commutative. Then we can naturally affirm that _Ue_ and _Ut_ are commutative as well. 

_Proof._ Additionally, we prove that the cost of OR operator’s implementation is not as expensive as it seems to be. The process of computation can be described as follows: 

|**Algorithm 1:**OR|
|---|
|**Input**: queries_{q_1_...qn}_|
|**Output**: the union of queries|
|1: Let_u_= 0.|
|2: For_q_ In_{q_1_...qn}_|
|3:<br>_u_=_u_+_q −u ∗q_|
|4: EndFor|
|5: **return**_u_|



As we implement OR step by step on a series of _n_ queries, the loop goes n-1 times in total. Assuming the embedding dimension of a query is _d_ , we can have the cost of OR is O( _n*d_ ). 

## **B More Details about Experiments** 

In this section, we show more details about our experiments. Firstly, we introduce the origin datasets in Section B.1. Then, we describe the details on how we define the query structure and how to sample queries in Section B.2. With the generated queries, we construct three datasets and report their statistics in Section B.3. We also show the details of our model implementation in Section B.4 and evaluation protocol in Section B.5. Besides, we show more experimental data for sensitive analysis in Section B.6. Finally, we present the detail metrics of main results in Section B.7. 

## **B.1 Details of Origin Datasets** 

To construct the reasoning dataset, we use three datasets of temporal KGs that have become standard benchmarks for TKGC: ICEWS14 [37], ICEWS05-15 [37], and GDELT-500 [38]. The two subsets of ICEWS are generated by Boschee et al. [37]: ICEWS14 corresponding to the facts in 2014 and ICEWS05-15 corresponding to the facts between 2005 and 2015. GDELT-500 generated by Leetaru and Schrodt [38] corresponds to the facts from April 1, 2015, to March 31, 2016. The statistics of the dataset are shown in Table 8. 

17 

Table 4: Definition of query structures in static query embeddings [2–4]. Operators are defined on entity sets only, including Pe, And, Or, and Not. ’e’, ’r’, ’n’, ’u’ denote entity, relation, negation and union, respectively. The braket ’()’ denotes a relational projection or intersection operation. 

|**Query Name**|**Query Structure Defnition**|
|---|---|
|**1p**|(’e’, (’r’,))|
|**2p**|(’e’, (’r’, ’r’))|
|**3p**|(’e’, (’r’, ’r’, ’r’))|
|**2i**|((’e’, (’r’,)), (’e’, (’r’,)))|
|**3i**|((’e’, (’r’,)), (’e’, (’r’,)), (’e’, (’r’,)))|
|**ip**|(((’e’, (’r’,)), (’e’, (’r’,))), (’r’,))|
|**pi**|((’e’, (’r’, ’r’)), (’e’, (’r’,)))|
|**2in**|((’e’, (’r’,)), (’e’, (’r’, ’n’)))|
|**3in**|((’e’, (’r’,)), (’e’, (’r’,)), (’e’, (’r’, ’n’)))|
|**inp**|(((’e’, (’r’,)), (’e’, (’r’, ’n’))), (’r’,))|
|**pin**|((’e’, (’r’, ’r’)), (’e’, (’r’, ’n’)))|
|**pni**|((’e’, (’r’, ’r’, ’n’)), (’e’, (’r’,)))|
|**2u-DNF**|((’e’, (’r’,)), (’e’, (’r’,)), (’u’,))|
|**up-DNF**|(((’e’, (’r’,)), (’e’, (’r’,)), (’u’,)), (’r’,))|
|**2u-DM**|(((’e’, (’r’, ’n’)), (’e’, (’r’, ’n’))), (’n’,))|
|**up-DM**|(((’e’, (’r’, ’n’)), (’e’, (’r’, ’n’))), (’n’, ’r’))|



Table 5: Basic functions. _Qe_ is entity set and _Qt_ is timestamp set. _E_ is the set of all entities, _T_ is the set of all timestamps and _F_ is the set of triples. 

|**Name**|**Symbol**|**Input**|**Output**|
|---|---|---|---|
|And|And|_Qe_1,_Qe_2|_Qe_1_∩Qe_2|
|Or|Or|_Qe_1,_Qe_2|_Qe_1_∪Qe_2|
|Not|Not|_Qe_|_E −Qe_|
|EntityProjection|Pe|_Qe_, r,_Qt_|_{o|s ∈Qe, t ∈Qt,_(_s, r, o, t_)_∈F}_|
|TimeProjection|Pt|_Qe_1, r,_Qe_2|_{t|s ∈Qe_1_, o ∈Qe_2_,_(_s, r, o, t_)_∈F}_|
|TimeAnd|TimeAnd|_Qt_1,_Qt_2|_Qt_1_∩Qt_2|
|TimeOr|TimeOr|_Qt_1,_Qt_2|_Qt_1_∪Qt_2|
|TimeNot|TimeNot|_Qt_|_T −Qt_|
|before|before|_Qt_|_{t|t <_min(_Qt_)_}_|
|after|after|_Qt_|_{t|t >_max(_Qt_)_}_|



## **B.2 Details of Query Generation** 

**The Design of Query Structure Schema** . Previous query embedding methods share the same query structure schema (Table 4) to perform complex logical reasoning over static knowledge graph. When it comes to dynamic knowledge graph (TKG), the static schema is not sufficient to capture the temporal reasoning. In this paper, we propose a novel query structure schema for reasoning. The schema is composed of basic functions and query structures. The basic functions are defined in Table 5. The query structures are shown in Table 6 with visualization in Figure 5 and Figure 6. The query structures are based on the basic functions: **And** , **Or** , **Not** , **EntityProjection** , **TimeProjection** , **TimeAnd** , **TimeOr** , **TimeNot** , **Before** , and **After** . Of course, users can define more basic functions or query structures to extend the schema to multi-modal KGs, hyper-relation KGs and so on. It is a flexible schema overall. 

In order to keep a similar experiment settings with previous static query embeddings, the query structures are further aggregated into groups, shown in Table 7. The groups avg _e,_ avg _t,_ avg _e,Ce ,_ avg _t,Ct_ are for training and testing. Besides, extra groups avg _{Ue},_ avg _{Ut},_ avg _x_ are for evaluation and testing only. 

In implementation, we define the basic functions and query structures as functions in python, which are named **Complex Query Function (CQF)** . There are lots of advantages of using python functions to define query structures. Firstly, the functions are easy to understand and debug. Secondly, the functions are easy to be reused in the dataset-sampling process and model-training process. Thirdly, 

18 

Table 6: Definition of query structures in temporal query embeddings. Operators on entity sets include Pe, And, Or, and Not. Operators on timestamp sets include Pt, TimeAnd, TimeOr, TimeNot, after, and before. The prefix "a" and "b" represent "after" and "before" respectively. The prefix "s" or "o" mean that the sub-query is a subject query or an object query respectively. 

|**Type**|**Query Name**<br>**Query Structure Defnition**|
|---|---|
|entity multi-hop<br>entity not<br>entity union|**Pe**<br>Pe(s, r, t)<br>**Pe2**<br>Pe(Pe(s1, r1, t1), r2, t2)<br>**Pe3**<br>Pe(Pe(Pe(s1, r1, t1), r2, t2), r3, t3)<br>**e2i**<br>And(Pe(s1, r1, t1), Pe(s2, r2, t2))<br>**e3i**<br>And(Pe(s1, r1, t1), Pe(s2, r2, t2), Pe(e3, r3, t3))<br>**e2i_N**<br>And(Pe(s1, r1, t1), Not(Pe(s2, r2, t2)))<br>**e3i_N**<br>And(Pe(s1, r1, t1), Pe(s2, r2, t2), Not(Pe(s3, r3, t3)))<br>**Pe_e2i_N**<br>Pe(And(Pe(s1, r1, t1), Not(Pe(s2, r2, t2))), r3, t3)<br>**e2i_PeN**<br>And(Pe(Pe(s1, r1, t1), r2, t2), Not(Pe(s2, r3, t3)))<br>**e2i_NPe**<br>And(Not(Pe(Pe(s1, r1, t1), r2, t2)), Pe(s2, r3, t3))<br>**e2u**<br>Or(Pe(s1, r1, t1), Pe(s2, r2, t2))<br>**Pe_e2u**<br>Pe(Or(Pe(s1, r1, t1), Pe(s2, r2, t2)), r3, t3)|
|time multi-hop<br>time not<br>time union|**Pt**<br>Pt(s, r, o)<br>**aPt**<br>after(Pt(s, r, o))<br>**bPt**<br>before(Pt(s, r, o))<br>**Pe_Pt**<br>Pe(s1, r1, Pt(s2, r2, o1))<br>**Pt_sPe_Pt**<br>Pt(Pe(s1, r1, Pt(s2, r2, o1)), r3, o2)<br>**Pt_oPe_Pt**<br>Pt(s1, r1, Pe(s2, r2, Pt(s3, r3, o1)))<br>**t2i**<br>TimeAnd(Pt(s1, r1, o1), Pt(s2, r2, o2))<br>**t3i**<br>TimeAnd(Pt(s1, r1, o1), Pt(s2, r2, o2), Pt(s3, r3, o3))<br>**t2i_N**<br>TimeAnd(Pt(s1, r1, o1), TimeNot(Pt(s2, r2, o2)))<br>**t3i_N**<br>TimeAnd(Pt(s1, r1, o1), Pt(s2, r2, o2), TimeNot(Pt(s3, r3, o3)))<br>**Pe_t2i_N**<br>Pe(s1, r1, TimeAnd(Pt(Pe(s2, r2, t1), r3, o1), TimeNot(Pt(s3, r4, o2))))<br>**t2i_NPt**<br>TimeAnd(TimeNot(Pt(Pe(s1, r1, t1), r2, o1)), Pt(s2, r3, o2))<br>**t2i_PtN**<br>TimeAnd(Pt(Pe(s1, r1, t1), r2, o1), TimeNot(Pt(s2, r3, o2)))<br>**t2u**<br>TimeOr(Pt(s1, r1, o1), Pt(s2, r2, o2))<br>**Pe_t2u**<br>Pe(s1, r1, TimeOr(Pt(s2, r2, o1), Pt(s3, r3, o2)))|
|hybrid multi-hop|**between**<br>TimeAnd(after(Pt(s1, r1, o1)), before(Pt(s2, r2, o2)))<br>**e2i_Pe**<br>And(Pe(Pe(s1, r1, t1), r2, t2), Pe(s2, r3, t3))<br>**Pe_e2i**<br>Pe(e2i(s1, r1, t1, s2, r2, t2), r3, t3)<br>**t2i_Pe**<br>TimeAnd(Pt(Pe(s1, r1, t1), r2, o1), Pt(s2, r3, o2))<br>**Pe_t2i**<br>Pe(s1, r1, t2i(s2, r2, o1, s3, r3, o2))<br>**Pe_aPt**<br>Pe(s1, r1, after(Pt(s2, r2, o1)))<br>**Pe_bPt**<br>Pe(s1, r1, before(Pt(s2, r2, o1)))<br>**Pe_at2i**<br>Pe(s1, r1, after(t2i(s2, r2, o1, s3, r3, o2)))<br>**Pe_bt2i**<br>Pe(s1, r1, before(t2i(s2, r2, o1, s3, r3, o2)))<br>**Pt_sPe**<br>Pt(Pe(s1, r1, t1), r2, o1)<br>**Pt_oPe**<br>Pt(s1, r1, Pe(s2, r2, t1))<br>**Pt_se2i**<br>Pt(e2i(s1, r1, t1, s2, r2, t2), r3, o1)<br>**Pt_oe2i**<br>Pt(s1, r1, e2i(s2, r2, t1, s3, r3, t2))|



19 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2023_TFLEX_Temporal_Feature-Logic_Embedding_Framework_for_Complex_Reasoning_over_Temporal_Knowledge_Graph_images/nips_2023_TFLEX_Temporal_Feature-Logic_Embedding_Framework_for_Complex_Reasoning_over_Temporal_Knowledge_Graph.pdf-0020-00.png)


**----- Start of picture text -----**<br>
Pe ? e2i_NPe Not<br>And ?<br>Pt<br>?<br>e2i_PeN<br>And ?<br>Pe2 ?<br>Not<br>Pe3 ?<br>Pe_e2i_Pe_NPe<br>And ?<br>AndNot<br>Pe_Pt ?<br>e2i_N<br>e2i And ?<br>And ? AndNot<br>e3i_N<br>e3i<br>And ?<br>And ?<br>AndNot<br>e2u<br>e2i_Pe Or ?<br>And ?<br>Pe_e2u<br>Pe_e2i AndOr ?<br>AndAnd ?<br>Pt_lPe<br>?<br>Pe_t2i<br>?<br>TimeAnd Pt_rPe<br>?<br>**----- End of picture text -----**<br>


Figure 5: Visualization of temporal query structures answering entity sets. These structures are basic functions that can be used to construct more complex temporal query structures. 

20 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2023_TFLEX_Temporal_Feature-Logic_Embedding_Framework_for_Complex_Reasoning_over_Temporal_Knowledge_Graph_images/nips_2023_TFLEX_Temporal_Feature-Logic_Embedding_Framework_for_Complex_Reasoning_over_Temporal_Knowledge_Graph.pdf-0021-00.png)


**----- Start of picture text -----**<br>
t2i t2i_N<br>TimeAnd ? TimeAnd ?<br>TimeNot<br>t3i t3i_N<br>TimeAnd ? TimeAnd ?<br>TimeNot<br>t2i_Pe t2u<br>TimeAnd ? TimeOr ?<br>Pt_le2i<br>Pe_t2u<br>?<br>And<br>? TimeOr<br>Pt_re2i<br>Pe_aPt ?<br>AndAnd After<br>?<br>Pe_bPt ?<br>t2i_NPt<br>Before<br>Pe_at2i<br>TimeNot<br>TimeAnd ?<br>?<br>TimeAndTimeAnd After<br>t2i_PtN<br>Pe_bt2i<br>TimeAnd ? ?<br>TimeNot TimeAndTimeAnd Before<br>Pe_t2i_PtPe_NPt<br>between<br>? After<br>TimeAnd TimeAnd ?<br>TimeNot Before<br>**----- End of picture text -----**<br>


Figure 6: Visualization of temporal query structures answering timestamp sets. These structures are basic functions that can be used to construct more complex temporal query structures. 

21 

Table 7: The query structures are aggregated into groups. The groups are inspired by the experiment settings in static query embeddings [1–4]. The groups avg _e,_ avg _t,_ avg _e,Ce ,_ avg _t,Ct_ are for training and testing. Besides, extra groups avg _{Ue},_ avg _{Ut},_ avg _x_ are for evaluation and testing only. 

||1p|2p|3p|2i|3i|
|---|---|---|---|---|---|
|avg_e_|Pe|Pe2|Pe3|e2i|e3i|
|avg_t_|Pt, aPt, bPt|Pe_Pt|Pt_sPe_Pt, Pt_oPe_Pt|t2i|t3i|
||2in|3in|inp|pin|pni|
|avg_e,Ce_|e2i_N|e3i_N|Pe_e2i_N|e2i_PeN|e2i_NPe|
|avg_t,Ct_|t2i_N|t3i_N|Pe_t2i_N|t2i_PtN|t2i_NPt|
||2u|up||||
|avg_{Ue}_|e2u|Pe_e2u||||
|avg_{Ut}_|t2u|Pe_t2u||||
||pi|ip||||
|avg_x_|e2i_Pe|Pe_e2i||Pe_aPt|Pe_bPt|
||t2i_Pe|Pe_t2i||Pe_at2i|Pe_bt2i|
|||||Pt_sPe|Pt_oPe|
||||between|Pt_se2i|Pt_oe2i|



the functions are easy to be extended to support more complex query structures and reimplement the existing query embedding methods. 

As is introduced in Section 4, we use the computation graph to represent the reasoning process. To get the computation graph of a CQF, we use the python interpreter to parse the function to Abstract Syntax Tree (AST), which is a more friendly readable subset of computation graph. In this way, executing the computation graph is equivalent to executing the CQF in python interpreter. Since we have the god privileged access to the AST, we can modify the interpreter to support various dynamic reasoning processes. Below we show how to unify (1) the reasoning of ground-truth, (2) the sampling of anchors, and (3) the reasoning of embedding-based methods by modifying the python interpreter. 

## **Ground-Truth Reasoning Interpreter** . 

In the reasoning of ground-truth, we need to get all real answers of a query under a given TKG. To achieve this, the first we need to do is to implement the basic functions. We wrap the python built-in set to QuerySet class to store entities and timestamps. The basic functions are implemented as lambda function in python, with input and output as QuerySet. Then, the basic functions are registered as symbols to the python interpreter. When executing the CQF, the python interpreter will call the corresponding basic functions to get the final answer. The answers are finally generated from subgraph matching, which depends on the ground truth in TKG. We show the pseudocode of the ground-truth reasoning interpreter in Figure 7. 

## **Anchors Sampling Interpreter** . 

The anchor, which may be entity, relation or timestamp, is the input argument of the CQF. The aim of the anchors sampling interpreter is to get the possible anchors and answers of a query under a given TKG. For the sampled anchors, we expect the answer set not empty. Since the answers could be obtained from the ground-truth reasoning interpreter, we only focus on the sampling of anchors in the anchors sampling interpreter. 

Given the standard split of edges into training ( _F_ train), validation ( _F_ valid) and test ( _F_ test) sets, we append inverse relations and double the number of edges in the graph. Then we create three graph: _G_ train = _{V, R, T , F_ train _}_ , _G_ valid = _{V, R, T , F_ train + _F_ valid _}_ , _G_ test = _{V, R, T , F_ train + _F_ valid + _F_ test _}_ . Given a query _q_ , let � _q_ �train, � _q_ �valid, and � _q_ �test denote a set of answers (entities or timestamps) obtained by subgraph matching of _q_ on _G_ train, _G_ valid and _G_ test. For each query _q_ , the reasoning process starts from anchor nodes and computes the final answer set via subgraph matching. 

For the basic function **Pe** and **Pt** , we simply use all the triples and extract the entities and timestamps as the anchors. These anchors are supposed to have no empty answers under **Pe** and **Pt** . However, when it comes to some hard queries, such as **e2i** and **e3i** , the random sampled anchors may have empty answers. Because the TKG is sparse and incomplete, the intersection of two query sets has a high probability to be empty. The empty answers are meaningless for model-training and damage to 

22 

```
#1.predefinedquerystructuresandtheInterpreter
ComplexQueryFunctions={
"Pe2":"defPe2(e1,r1,t1,r2,t2):returnPe(Pe(e1,r1,t1),r2,t2)",#2p
"Pe3":"defPe3(e1,r1,t1,r2,t2,r3,t3):returnPe(Pe(Pe(e1,r1,t1),r2,t2),r3,t3)",#3p
"aPt":"defaPt(s,r,o):returnTimeAfter(Pt(s,r,o))",#aforafter
"bPt":"defbPt(s,r,o):returnTimeBefore(Pt(s,r,o))",#bforbefore
...
}
classInterpreter:
def__init__(self,symbols:Dict[str,Any]):
self.symbols=symbols
fork,CQFinComplexQueryFunctions.items():
self.eval(CQF)
defeval(self,line:str)->Any:returnast.parse(line).eval(self)
#2.GroundTruthReasoningInterpreter
QuerySet=set
EntitySet=set
TimeSet=set
classGroundTruthReasoningInterpreter(Interpreter):
def__init__(self,TKG):
Pe=defaultdict(lambda:defaultdict(lambda:defaultdict(set)))
Pt=defaultdict(lambda:defaultdict(lambda:defaultdict(set)))
fors,r,o,tinTKG.F:#Fisthesetofallfacts
Pe[s][r][t].add(o)
Pt[s][r][o].add(t)
symbols={
"Pe":lambda(qs,r,qt):EntitySet(reduce(|,[Pe[s][r][t]fors,tinproduct(qs,qt)]))
"Pt":lambda(qs,r,qo):TimeSet(reduce(|,[Pt[s][r][o]fors,oinproduct(qs,qo)]))
"And":lambda(q1,q2):EntitySet(q1&q2)
"Or":lambda(q1,q2):EntitySet(q1|q2)
"Not":lambda(qe):EntitySet(TKG.V-qe)#Visthesetofallentities
"TimeAnd":lambda(q1,q2):TimeSet(q1&q2)
"TimeOr":lambda(q1,q2):TimeSet(q1|q2)
"TimeNot":lambda(qt):TimeSet(TKG.T-qt)#Tisthesetofalltimestamps
"TimeBefore":lambda(qt):TimeSet([tfortinTKG.Tift<min(qt)])
"TimeAfter":lambda(qt):TimeSet([tfortinTKG.Tift>max(qt)])
}
super().__init__(symbols)
```

```
#TKG.F={(AngelaMerkel,makeavisit,China,2010-07-16)}
p=GroundTruthReasoningInterpreter(TKG)
answer=p.eval("Pe({'AngelaMerkel'},'makeavisit',{'2010-07-16'})")
assertanswer==EntitySet('China')
```

Figure 7: Python-style pseudocode of ground-truth reasoning interpreter. 

the performance of data-sampling process. To solve this problem, we use the following two strategies to accelerate the sampling. 

(1) **Inverse Sampling** : We randomly sample an entity as objective, denoted _eo_ . Then we sample subjective entity _es_ 1 _, es_ 2, relation _r_ 1 _, r_ 2 and timestamp _t_ 1 _, t_ 2, according to the fact ( _es_ 1 _, r_ 1 _, eo, t_ 1) and ( _es_ 2 _, r_ 2 _, eo, t_ 2). The sampled anchors are ( _es_ 1 _, r_ 1 _, t_ 1; _es_ 2 _, r_ 2 _, t_ 2). These anchors under **e2i** have the answer _eo_ at least, which asserts that the answer set is not empty. Such sampling method that samples the answer first and then samples the anchors is called **Inverse Sampling** . 

(2) **Bi-directional Sampling** : For long multi-hop queries, such as **Pe2** , the complexity of random sampling is _O_ ( _N_[2] _[L]_ ), where _N_ is the entity count and _L_ is the length of the query. **Pe2** has _L_ = 2. It contains an anchor-sampling complexity of _O_ ( _N[L]_ ) and a ground-truth reasoning (to get answers) complexity of _O_ ( _N[L]_ ). The origin sampling is to sample one subjective entity, get the objective as answers as next subjective anchors, again get the next objective answers. The final answers recursively depend on the initial subjective entity, leading to low performance. To accelerate, we sample an entity at the middle of AST, denoted _em_ . Then we sample in two directions. One is forward, _em_ as the subjective entity, to sample relation _r_ 2 and timestamp _t_ 2. The other is backward, _em_ as the objective entity, to sample subjective anchor _e_ 1, relation _r_ 1 and timestamp _t_ 1. The sampled anchors are ( _e_ 1 _, r_ 1 _, t_ 1; _r_ 2 _, t_ 2). The final answer set is asserted not empty, because it at least contains all the answers of Pe( _em, r_ 2 _, t_ 2). Be aware that the two directions are independent, which can be sampled in parallel. Such sampling method that samples the answer in two directions is called **Bi-directional Sampling** , which can reduce the computation complexity to _O_ ( _N[L]_[+] _[L]_ 2 ), composed of a sampling _L_ complexity of _O_ ( _N_ 2 ) and a ground-truth reasoning complexity of _O_ ( _N[L]_ ). 

23 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2023_TFLEX_Temporal_Feature-Logic_Embedding_Framework_for_Complex_Reasoning_over_Temporal_Knowledge_Graph_images/nips_2023_TFLEX_Temporal_Feature-Logic_Embedding_Framework_for_Complex_Reasoning_over_Temporal_Knowledge_Graph.pdf-0024-00.png)


Figure 8: The screenshot of a real running interpreter. 

## **Embedding Reasoning Interpreter** . 

On the contrary of the ground-truth reasoning interpreter which has ’ **set** ’ as the input and output of the CQFs, in the embedding reasoning interpreter, the input and output of CQFs are ’ **embedding vectors** ’. The embedding-based method for reasoning over TKG is called **Temporal Query Embedding** . The computation complexity of the embedding reasoning interpreter is _O_ ( _L_ + _N_ ), where _L_ is the length of the query and _N_ is the entity count. The complexity consists of a query-embedding complexity of _O_ ( _L_ ) and a scoring-to-answer complexity of _O_ ( _N_ ). The embedding reasoning is much faster than the ground-truth reasoning, which has a computation complexity of _O_ ( _N[L]_ ). 

To implement the embedding reasoning interpreter, we just need to replace the basic functions with neural networks. The symbols dict to pass to the interpreter is 

_{_ "Pe": _Pe,_ "Pt": _Pt,_ "And": _Ie,_ "Or": _Ue,_ "Not": _Ne,_ (8) "TimeAnd": _It,_ "TimeOr": _Ut,_ "TimeNot": _Nt,_ "TimeAfter": _At,_ "TimeBefore": _Bt }_ 

The embedding vectors are learned as is introduced in Section 4. Unlike the ground-truth reasoning, the embedding reasoning is fuzzy. The output of the embedding reasoning interpreter is an embedding vector, which is a fuzzy set representation. To get the final answer set, we need to use the distance function (in Section 4.3) to score to candidate answers. In this way, the embedding vector is converted to the final answer set. The final answers are given in the ranking list, where each answer is followed by its distance to the query. Usually the top-k answers are accepted as the final answers. 

Note that the interpreter is flexible, and easy to implement and extend. In order to reproduce static query embeddings [2–4] over dynamic knowledge graphs, the only thing to do is to implement the symbols dict and the distance function. 

We hope that the design of interpreter is helpful for the future research of temporal query embedding. 

## **Screenshot** . 

Additionally, we show the screenshot of a real running interpreter in Figure 8. In the screenshot, we randomly select an example of **e2i** query. Then, we use the embedding reasoning interpreter to get the final answer set step by step. The final answer set is shown in the ranking list, where each answer is followed by its similarity score to the query. Finally, the bot correctly predicts the hard answer which only exists in the test set! 

## **B.3 Details of Generated Datasets** 

Finally, we generate three datasets for the task of temporal complex reasoning using the process in Section B.2. The statistics of the generated datasets are listed below. Table 9 show the count of query 

24 

Table 8: Statistics on ICEWS14, ICEWS05-15, and GDELT-500. 

|**Dataset**|**Entities**|**Relations**|**Timestamps**|_|_**Training**_|_|_|_**Validation**_|_|_|_**Test**_|_|**Total Edges**|
|---|---|---|---|---|---|---|---|
|ICEWS14|7,128|230|365|72,826|8,941|8,963|90,730|
|ICEWS05-15|10,488|251|4,017|368,962|46,275|46,092|479,329|
|GDELT-500|500|20|366|2,735,685|341,961|341,961|3,419,607|



Table 9: Queries count for each dataset. 

|**Query Name**|**ICEWS14**<br>**Train**<br>**Validate**<br>**Test**|**ICEWS05-15**<br>**Train**<br>**Validate**<br>**Test**|**GDELT-500**<br>**Train**<br>**Validate**<br>**Test**|
|---|---|---|---|
|**Pe**<br>**Pe2**<br>**Pe3**<br>**e2i**<br>**e3i**|66783<br>8837<br>8848<br>72826<br>3482<br>4037<br>72826<br>3492<br>4083<br>72826<br>3305<br>3655<br>72826<br>2966<br>3023|344042<br>45829<br>45644<br>368962<br>10000<br>10000<br>368962<br>10000<br>10000<br>368962<br>10000<br>10000<br>368962<br>10000<br>10000|1115102<br>273842<br>273432<br>2215309<br>10000<br>10000<br>2215309<br>10000<br>10000<br>2215309<br>10000<br>10000<br>2215309<br>10000<br>10000|
|**Pt**<br>**aPt**<br>**bPt**<br>**Pe_Pt**<br>**Pt_sPe_Pt**<br>**Pt_oPe_Pt**<br>**t2i**<br>**t3i**|42690<br>7331<br>7419<br>13234<br>4411<br>4411<br>13234<br>4411<br>4411<br>7282<br>3385<br>3638<br>13234<br>5541<br>6293<br>13234<br>5480<br>6242<br>72826<br>5112<br>6631<br>72826<br>3094<br>3296|142771<br>28795<br>28752<br>68262<br>10000<br>10000<br>68262<br>10000<br>10000<br>36896<br>10000<br>10000<br>68262<br>10000<br>10000<br>68262<br>10000<br>10000<br>368962<br>10000<br>10000<br>368962<br>10000<br>10000|687326<br>199780<br>199419<br>221530<br>10000<br>10000<br>221530<br>10000<br>10000<br>221530<br>10000<br>10000<br>221530<br>10000<br>10000<br>221530<br>10000<br>10000<br>2215309<br>10000<br>10000<br>2215309<br>10000<br>10000|
|**e2i_N**<br>**e3i_N**<br>**Pe_e2i_N**<br>**e2i_PeN**<br>**e2i_NPe**|7282<br>2949<br>2975<br>7282<br>2913<br>2914<br>7282<br>2968<br>3012<br>7282<br>2971<br>3031<br>7282<br>3061<br>3192|36896<br>10000<br>10000<br>36896<br>10000<br>10000<br>36896<br>10000<br>10000<br>36896<br>10000<br>10000<br>36896<br>10000<br>10000|221530<br>10000<br>10000<br>221530<br>10000<br>10000<br>221530<br>10000<br>10000<br>221530<br>10000<br>10000<br>221530<br>10000<br>10000|
|**t2i_N**<br>**t3i_N**<br>**Pe_t2i_N**<br>**t2i_PtN**<br>**t2i_NPt**|7282<br>3135<br>3328<br>7282<br>2924<br>2944<br>7282<br>3031<br>3127<br>7282<br>3300<br>3609<br>7282<br>4873<br>5464|36896<br>10000<br>10000<br>36896<br>10000<br>10000<br>36896<br>10000<br>10000<br>36896<br>10000<br>10000<br>36896<br>10000<br>10000|221530<br>10000<br>10000<br>221530<br>10000<br>10000<br>221530<br>10000<br>10000<br>221530<br>10000<br>10000<br>221530<br>10000<br>10000|
|**e2u**<br>**Pe_e2u**|-<br>2913<br>2913<br>-<br>2913<br>2913|-<br>10000<br>10000<br>-<br>10000<br>10000|-<br>10000<br>10000<br>-<br>10000<br>10000|
|**t2u**<br>**Pe_t2u**|-<br>2913<br>2913<br>-<br>2913<br>2913|-<br>10000<br>10000<br>-<br>10000<br>10000|-<br>10000<br>10000<br>-<br>10000<br>10000|
|**between**<br>**e2i_Pe**<br>**Pe_e2i**<br>**t2i_Pe**<br>**Pe_t2i**<br>**Pe_aPt**<br>**Pe_at2i**<br>**Pt_sPe**<br>**Pt_se2i**<br>**Pe_bPt**<br>**Pe_bt2i**<br>**Pt_oPe**<br>**Pt_oe2i**|7282<br>2913<br>2913<br>-<br>2913<br>2913<br>-<br>2913<br>2913<br>-<br>2913<br>2913<br>-<br>2913<br>2913<br>7282<br>4134<br>4733<br>7282<br>4607<br>5338<br>7282<br>4976<br>5608<br>7282<br>3226<br>3466<br>7282<br>3970<br>4565<br>7282<br>4583<br>5386<br>7282<br>3321<br>3621<br>7282<br>3236<br>3485|36896<br>10000<br>10000<br>-<br>10000<br>10000<br>-<br>10000<br>10000<br>-<br>10000<br>10000<br>-<br>10000<br>10000<br>68262<br>10000<br>10000<br>36896<br>10000<br>10000<br>36896<br>10000<br>10000<br>36896<br>10000<br>10000<br>36896<br>10000<br>10000<br>36896<br>10000<br>10000<br>36896<br>10000<br>10000<br>36896<br>10000<br>10000|221530<br>10000<br>10000<br>-<br>10000<br>10000<br>-<br>10000<br>10000<br>-<br>10000<br>10000<br>-<br>10000<br>10000<br>221530<br>10000<br>10000<br>221530<br>10000<br>10000<br>221530<br>10000<br>10000<br>221530<br>10000<br>10000<br>221530<br>10000<br>10000<br>221530<br>10000<br>10000<br>221530<br>10000<br>10000<br>221530<br>10000<br>10000|



structures in the split of training, validation and testing. The average answers count of each query structure are reported in Table 10. The number of queries for each dataset is shown in Table 11. 

## **B.4 Implementation Details** 

We use the Embedding Reasoning Interpreter as is introduced in Appendix B.2. We implement our model with PyTorch and use Adam [50] as a gradient optimizer. For each experiment, we use one GTX1080 graphic card. The hyperparameters for each dataset are shown in Table 12. Note that we do not perform hyperparameter tuning for each dataset. Instead, we use the same hyperparameters 

25 

Table 10: Average answers count for each dataset. All numbers are rounded to two decimal places. 

|**Query Name**|**ICEWS14**<br>**Train**<br>**Validate**<br>**Test**|**ICEWS05-15**<br>**Train**<br>**Validate**<br>**Test**|**GDELT-500**<br>**Train**<br>**Validate**<br>**Test**|
|---|---|---|---|
|**Pe**<br>**Pe2**<br>**Pe3**<br>**e2i**<br>**e3i**|1.09<br>1.01<br>1.01<br>1.03<br>2.19<br>2.23<br>1.04<br>2.25<br>2.29<br>1.02<br>2.76<br>2.84<br>1.00<br>1.57<br>1.59|1.07<br>1.01<br>1.01<br>1.02<br>2.15<br>2.19<br>1.02<br>2.18<br>2.21<br>1.01<br>2.36<br>2.52<br>1.00<br>1.26<br>1.26|2.07<br>1.21<br>1.21<br>2.61<br>6.51<br>6.13<br>5.11<br>10.86<br>10.70<br>1.05<br>2.30<br>2.32<br>1.00<br>1.20<br>1.35|
|**Pt**<br>**aPt**<br>**bPt**<br>**Pe_Pt**<br>**Pt_sPe_Pt**<br>**Pt_oPe_Pt**<br>**t2i**<br>**t3i**|1.71<br>1.22<br>1.21<br>177.99<br>176.09<br>175.89<br>181.20<br>179.88<br>179.26<br>1.58<br>7.90<br>8.62<br>1.79<br>7.26<br>7.47<br>1.75<br>7.27<br>7.48<br>1.19<br>6.29<br>6.38<br>1.01<br>2.88<br>3.14|2.58<br>1.61<br>1.60<br>2022.16<br>2003.85<br>1998.71<br>1929.98<br>1923.75<br>1919.83<br>2.84<br>18.11<br>20.63<br>2.49<br>13.51<br>10.86<br>2.55<br>13.01<br>14.34<br>3.07<br>29.45<br>25.61<br>1.08<br>10.03<br>10.22|3.36<br>1.66<br>1.66<br>156.48<br>155.38<br>153.41<br>160.38<br>159.29<br>157.42<br>26.56<br>42.54<br>41.33<br>4.92<br>14.13<br>12.80<br>4.62<br>14.47<br>12.90<br>1.97<br>8.98<br>7.76<br>1.06<br>3.79<br>3.52|
|**e2i_N**<br>**e3i_N**<br>**Pe_e2i_N**<br>**e2i_PeN**<br>**e2i_NPe**|1.02<br>2.10<br>2.14<br>1.00<br>1.00<br>1.00<br>1.04<br>2.21<br>2.25<br>1.04<br>2.22<br>2.26<br>1.18<br>3.03<br>3.11|1.01<br>2.05<br>2.08<br>1.00<br>1.00<br>1.00<br>1.02<br>2.16<br>2.19<br>1.02<br>2.17<br>2.21<br>1.12<br>2.87<br>2.99|2.04<br>4.66<br>4.58<br>1.02<br>1.19<br>1.37<br>3.67<br>8.54<br>8.12<br>3.67<br>8.66<br>8.36<br>4.00<br>8.15<br>7.81|
|**t2i_N**<br>**t3i_N**<br>**Pe_t2i_N**<br>**t2i_PtN**<br>**t2i_NPt**|1.15<br>3.31<br>3.44<br>1.00<br>1.02<br>1.03<br>1.08<br>2.59<br>2.70<br>1.41<br>5.22<br>5.47<br>8.14<br>25.96<br>26.23|1.21<br>4.06<br>4.20<br>1.01<br>1.02<br>1.02<br>1.08<br>2.47<br>2.62<br>1.70<br>8.10<br>8.11<br>66.99<br>154.01<br>147.34|2.91<br>8.78<br>7.56<br>1.15<br>3.19<br>3.20<br>4.10<br>12.02<br>11.37<br>4.56<br>12.56<br>11.32<br>17.58<br>35.60<br>32.22|
|**e2u**<br>**Pe_e2u**|-<br>3.12<br>3.17<br>-<br>2.38<br>2.44|-<br>2.38<br>2.40<br>-<br>1.24<br>1.25|-<br>5.04<br>5.41<br>-<br>9.39<br>10.78|
|**t2u**<br>**Pe_t2u**|-<br>4.35<br>4.53<br>-<br>2.72<br>2.83|-<br>5.57<br>5.92<br>-<br>1.24<br>1.28|-<br>9.70<br>10.51<br>-<br>9.90<br>11.27|
|**between**<br>**e2i_Pe**<br>**Pe_e2i**<br>**t2i_Pe**<br>**Pe_t2i**<br>**Pe_aPt**<br>**Pe_at2i**<br>**Pt_sPe**<br>**Pt_se2i**<br>**Pe_bPt**<br>**Pe_bt2i**<br>**Pt_oPe**<br>**Pt_oe2i**|122.61<br>120.94<br>120.27<br>-<br>1.00<br>1.00<br>-<br>2.18<br>2.24<br>-<br>1.03<br>1.03<br>-<br>1.14<br>1.16<br>4.67<br>16.73<br>16.50<br>7.26<br>22.63<br>21.98<br>8.65<br>28.86<br>29.22<br>1.31<br>5.72<br>6.19<br>4.53<br>17.07<br>16.80<br>7.27<br>21.92<br>21.23<br>1.41<br>5.23<br>5.46<br>1.32<br>6.51<br>7.00|1407.87<br>1410.39<br>1404.76<br>-<br>1.00<br>1.00<br>-<br>1.32<br>1.33<br>-<br>1.01<br>1.02<br>-<br>1.07<br>1.08<br>18.68<br>43.80<br>46.23<br>30.40<br>60.03<br>53.18<br>71.51<br>162.36<br>155.46<br>1.37<br>9.00<br>9.30<br>18.70<br>45.81<br>48.23<br>30.31<br>61.59<br>64.98<br>1.68<br>8.36<br>8.21<br>1.44<br>10.49<br>10.89|214.16<br>210.99<br>207.85<br>-<br>1.07<br>1.10<br>-<br>5.08<br>5.49<br>-<br>1.34<br>1.44<br>-<br>2.01<br>2.20<br>49.31<br>66.21<br>68.88<br>88.77<br>101.60<br>101.88<br>27.55<br>45.83<br>43.73<br>2.76<br>8.72<br>7.66<br>67.67<br>84.79<br>83.00<br>88.80<br>100.64<br>100.67<br>3.84<br>11.31<br>10.06<br>2.55<br>8.17<br>7.27|



Table 11: Number of queries. Pe represents query answering (s,r,?,t). Pt represents query answering (s,r,o,?). QoE represents the query of entities (except Pe). QoT represents query of timestamps (except Pt). n1p represents a query that is not Pe or Pt. 

|**Dataset**|**Training**<br>**Pe**<br>**Pt**<br>**QoE**<br>**QoT**|**Validation**<br>**Pe**<br>**Pt**<br>**n1p**|**Test**<br>**Pe**<br>**Pt**<br>**n1p**|
|---|---|---|---|
|||||
|ICEWS14<br>ICEWS05-15<br>GDELT-500|273,710<br>27,371<br>59,078<br>8,000<br>149,689<br>14,968<br>20,094<br>5,000<br>107,982<br>10,798<br>16,910<br>4,000|66,990<br>66,990<br>10,000<br>66,990<br>22,804<br>10,000<br>66,990<br>17,021<br>10,000|66,990<br>66,990<br>10,000<br>66,990<br>66,990<br>10,000<br>66,990<br>66,990<br>10,000|



26 

Table 12: Hyperparameters on each dataset. _d_ is the embedding dimension, _b_ is the batch size, _n_ is the negative sampling size, _γ_ is the parameter in the loss function, _m_ is the maximum training step, and _l_ is the learning rate. 

|grate.|||||||||
|---|---|---|---|---|---|---|---|---|
|Dataset|_d_|_b_|_n_|_γ_|_m_|||_l_|
|ICEWS14|800|512|128|15|300k|1|_×_|10_−_4|
|ICEWS05-15|800|512|128|30|300k|1|_×_|10_−_4|
|GDELT-500|800|512|128|30|300k|1|_×_|10_−_4|




![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2023_TFLEX_Temporal_Feature-Logic_Embedding_Framework_for_Complex_Reasoning_over_Temporal_Knowledge_Graph_images/nips_2023_TFLEX_Temporal_Feature-Logic_Embedding_Framework_for_Complex_Reasoning_over_Temporal_Knowledge_Graph.pdf-0027-02.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2023_TFLEX_Temporal_Feature-Logic_Embedding_Framework_for_Complex_Reasoning_over_Temporal_Knowledge_Graph_images/nips_2023_TFLEX_Temporal_Feature-Logic_Embedding_Framework_for_Complex_Reasoning_over_Temporal_Knowledge_Graph.pdf-0027-03.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2023_TFLEX_Temporal_Feature-Logic_Embedding_Framework_for_Complex_Reasoning_over_Temporal_Knowledge_Graph_images/nips_2023_TFLEX_Temporal_Feature-Logic_Embedding_Framework_for_Complex_Reasoning_over_Temporal_Knowledge_Graph.pdf-0027-04.png)


**----- Start of picture text -----**<br>
(a) The impact of embedding dimension on MRR. (b) The impact of the margin  γ on MRR.<br>**----- End of picture text -----**<br>


Figure 9: The impact of embedding dimensionality _d_ and the margin _γ_ on MRR. 

found in sensitive analysis on ICEWS14 (Section B.6) for all datasets. Besides, the source code is available at Gitbub[†] . We cite these projects[‡ §] , and thank them for their great contributions! 

## **B.5 Evaluation** 

Given the standard split of edges into training ( _F_ train), validation ( _F_ valid) and test ( _F_ test) sets, we append inverse relations and double the number of edges in the graph. Then we create three graph: _G_ train = _{V, R, T , F_ train _}_ , _G_ valid = _{V, R, T , F_ train + _F_ valid _}_ , _G_ test = _{V, R, T , F_ train + _F_ valid + _F_ test _}_ . Given a query _q_ , let � _q_ �train, � _q_ �valid, and � _q_ �test denote a set of answers (entities or timestamps) obtained by subgraph matching of _q_ on _G_ train, _G_ valid and _G_ test. For a test query _q_ and its non-trivial answer set _v ∈_ � _q_ � _test −_ � _q_ � _valid_ , we denote the rank of each answer _vi_ as _rank_ ( _vi_ ). The mean reciprocal rank (MRR) is MRR( _q_ ) = _||v_ 1 _||_ � _vi∈v rank_ 1( _vi_ )[and Hits at K (Hits@K) is][ Hits@K][(] _[q]_[) =] _||v_ 1 _||_ � _vi∈v[f]_[(] _[rank]_[(] _[v][i]_[))][, where] _[ f]_[(] _[n]_[) =] �01 _,, n > Kn ≤ K_[.] 

## **B.6 Experiment Data for Sensitive Analysis** 

We conduct experiments on ICEWS14 to analyze the impact of the embedding dimensionality _d_ and the margin _γ_ on the performance of TFLEX. 

**Impacts of Embedding Dimensionality** Our experiments indicate that the selection of embedding dimension has a substantial influence on the effectiveness of TFLEX. We train TFLEX with embedding dimension _d ∈{_ 300 _,_ 400 _,_ 500 _,_ 600 _,_ 700 _,_ 800 _,_ 900 _,_ 1000 _}_ and plot results based on the validation set, as shown in Figure 9a. With the increase of _d_ , the model performance (indicated by MRR) increases rapidly and reaches its top at _d_ = 800. Therefore, we assign 800 as the best setting. 

> †TFLEX: https://github.com/LinXueyuanStdio/TFLEX 

> ‡ GQE, Query2box, BetaE: https://github.com/snap-stanford/KGReasoning 

> §ConE: https://github.com/MIRALab-USTC/QE-ConE 

27 

Table 13: The mean values and standard variances of TFLEX’s MRR results on ICEWS14. 

|**Pe**|**Pe2**<br>**Pe3**<br>**e2i**<br>**e3i**|
|---|---|
|48.21<br>_±_0.37|37.27<br>33.53<br>69.24<br>95.70<br>_±_0.12<br>_±_0.42<br>_±_0.46<br>_±_0.19|
|**Pt**|**aPt**<br>**bPt**<br>**Pe_Pt**<br>**Pt_sPe_Pt**<br>**Pt_oPe_Pt**<br>**t2i**<br>**t3i**|
|20.87<br>_±_0.43|3.00<br>2.96<br>14.27<br>9.57<br>9.46<br>27.49<br>52.84<br>_±_0.34<br>_±_0.50<br>_±_0.57<br>_±_0.44<br>_±_0.66<br>_±_0.51<br>_±_0.63|
|**e2i_N**|**e3i_N**<br>**Pe_e2i_N**<br>**e2i_PeN**<br>**e2i_NPe**|
|45.55<br>_±_0.12|99.56<br>34.74<br>35.63<br>38.61<br>_±_0.50<br>_±_0.44<br>_±_0.44<br>_±_0.20|
|t2i_N|t3i_N<br>Pe_t2i_N<br>t2i_PtN<br>t2i_NPt|
|25.38<br>_±_0.60|98.91<br>34.05<br>11.42<br>12.07<br>_±_0.13<br>_±_0.49<br>_±_0.57<br>_±_0.53|
|**e2u**|**Pe_e2u**|
|29.20<br>_±_0.12|42.28<br>_±_0.43|
|**t2u**|**Pe_t2u**|
|30.73<br>_±_0.54|21.74<br>_±_0.35|
|**e2i_Pe**<br>98.86<br>_±_0.34<br>**t2i_Pe**<br>96.62<br>_±_0.13|**Pe_e2i**<br>**Pe_aPt**<br>**Pe_bPt**<br>**Pe_at2i**<br>**Pe_bt2i**<br>36.77<br>8.66<br>9.74<br>7.90<br>7.78<br>_±_0.42<br>_±_0.61<br>_±_0.33<br>_±_0.39<br>_±_0.39<br>**Pe_t2i**<br>**Pt_sPe**<br>**Pt_oPe**<br>**Pt_se2i**<br>**Pt_oe2i**<br>**between**<br>64.50<br>4.32<br>10.58<br>8.20<br>7.95<br>2.57<br>_±_0.12<br>_±_0.46<br>_±_0.20<br>_±_0.34<br>_±_0.34<br>_±_0.14|



**Impacts of Parameter** _γ_ We train TFLEX with parameter _γ ∈{_ 5 _,_ 10 _,_ 15 _,_ 20 _,_ 25 _,_ 30 _,_ 35 _,_ 40 _}_ and plot MRR results in Figure 9b. Too small and too large _γ_ both get bad results, while _γ_ = 15 in the middle is the best. Therefore, we choose _γ_ = 15. 

**Error Bars of Main Results** In order to evaluate how stable the performance of TFLEX is, we run five times with random seeds _{_ 1 _,_ 10 _,_ 100 _,_ 1000 _,_ 10000 _}_ and report the error bars of these results. Table 13 shows the error bar of TFLEX’s MRR results on ICEWS14. Overall, the standard variances are small, which demonstrates that the performance of TFLEX is stable. 

## **B.7 Detail Metrics of Main Results** 

We report the MRR results on all query structures for each dataset in Table 14 (ICEWS14), Table 15 (ICEWS05-15) and Table 16 (GDELT-500). Query2box [2] can only answer queries without negation. BetaE [3] and ConE [4] can answer queries with negation, except temporal logic. ConE(temporal) and X(ConE) are the variants of ConE, aiming at exploring the right way to promote the static query embeddings to temporal ones in Section C. FLEX, X(ConE), X-1F, X-logic are the variants of TFLEX, introduced in the main body of the paper. 

## **C Exploration of Static Query Embeddings Toward Dynamic** 

In this section, we present more research questions that aim to explore the right way to promote the static query embeddings to temporal ones. Since ConE [4] is a strong baseline for complex logical reasoning over static knowledge graphs, we conduct exploration experiments by introducing variants of ConE. We present the overall results for each dataset in Table 17, detailed results in Table 14 (ICEWS14), Table 15 (ICEWS05-15) and Table 16 (GDELT-500). Overall, we are interested in the following research questions: 

28 

Table 14: MRR results on ICEWS14. The group avg _x_ wraps to two rows. **AVG** denotes average performance under all query types. 

|**Model**<br>avg_e_|**Pe**<br>**Pe2**<br>**Pe3**<br>**e2i**<br>**e3i**|
|---|---|
|Query2box<br>25.06<br>BetaE<br>37.19<br>ConE<br>41.94<br>ConE(temporal)<br>42.23<br>FLEX<br>43.67<br>TFLEX<br>56.79<br>X(ConE)<br>40.93<br>X-1F<br>56.89<br>X-logic<br>56.64|25.81<br>17.29<br>12.97<br>16.00<br>53.25<br>39.52<br>23.77<br>17.50<br>23.77<br>81.38<br>42.55<br>30.90<br>24.78<br>26.52<br>84.93<br>41.21<br>31.83<br>24.99<br>28.98<br>84.17<br>45.25<br>33.07<br>27.22<br>27.62<br>85.18<br>48.21<br>37.27<br>33.53<br>69.24<br>95.70<br>40.58<br>28.84<br>24.31<br>30.93<br>79.98<br>48.61<br>37.51<br>32.61<br>71.25<br>94.47<br>48.00<br>37.42<br>31.23<br>73.11<br>93.45|
|avg_t_|**Pt**<br>**aPt**<br>**bPt**<br>**Pe_Pt**<br>**Pt_sPe_Pt**<br>**Pt_oPe_Pt**<br>**t2i**<br>**t3i**|
|TFLEX<br>17.56<br>X(ConE)<br>16.41<br>X-1F<br>18.77<br>X-logic<br>18.03|20.87<br>3.00<br>2.96<br>14.27<br>9.57<br>9.46<br>27.49<br>52.84<br>19.45<br>3.06<br>3.01<br>13.96<br>8.79<br>8.67<br>25.01<br>49.36<br>21.94<br>3.04<br>2.99<br>16.69<br>10.73<br>10.44<br>29.66<br>54.71<br>21.33<br>3.10<br>3.05<br>15.76<br>9.85<br>9.94<br>28.57<br>52.63|
|avg_e,Ce_|**e2i_N**<br>**e3i_N**<br>**Pe_e2i_N**<br>**e2i_PeN**<br>**e2i_NPe**|
|BetaE<br>36.69<br>ConE<br>44.88<br>ConE(temporal)<br>44.94<br>FLEX<br>44.41<br>TFLEX<br>50.82<br>X(ConE)<br>42.15<br>X-1F<br>49.78<br>X-logic<br>51.17|30.44<br>87.93<br>19.94<br>22.80<br>22.36<br>40.98<br>99.12<br>29.22<br>29.84<br>25.22<br>41.45<br>98.99<br>29.48<br>30.71<br>24.09<br>38.30<br>99.22<br>31.18<br>30.49<br>22.84<br>45.55<br>99.56<br>34.74<br>35.63<br>38.61<br>39.10<br>96.25<br>26.92<br>27.59<br>20.89<br>42.31<br>99.31<br>34.89<br>34.77<br>37.61<br>44.65<br>99.25<br>35.30<br>36.31<br>40.33|
|avg_t,Ct_|t2i_N<br>t3i_N<br>Pe_t2i_N<br>t2i_PtN<br>t2i_NPt|
|TFLEX<br>36.37<br>X(ConE)<br>35.24<br>X-1F<br>37.73<br>X-logic<br>36.39|25.38<br>98.91<br>34.05<br>11.42<br>12.07<br>26.90<br>98.82<br>30.30<br>9.44<br>10.75<br>26.40<br>98.82<br>37.85<br>12.38<br>13.23<br>25.68<br>98.69<br>33.30<br>11.34<br>12.93|
|avg_{Ue}_|**e2u**<br>**Pe_e2u**|
|BetaE<br>19.95<br>ConE<br>26.47<br>ConE(temporal)<br>27.63<br>FLEX<br>29.25<br>TFLEX<br>35.74<br>X(ConE)<br>25.46<br>X-1F<br>34.48<br>X-logic<br>34.68|18.61<br>21.28<br>21.84<br>31.11<br>23.01<br>32.26<br>24.05<br>34.46<br>29.20<br>42.28<br>20.26<br>30.67<br>30.04<br>38.93<br>29.44<br>39.92|
|avg_{Ut}_|**t2u**<br>**Pe_t2u**|
|TFLEX<br>26.24<br>X(ConE)<br>24.07<br>X-1F<br>28.04<br>X-logic<br>26.36|30.73<br>21.74<br>27.63<br>20.51<br>33.91<br>22.16<br>31.21<br>21.52|
|avg_x_<br>TFLEX<br>28.03<br>X(ConE)<br>26.65<br>X-1F<br>29.31<br>X-logic<br>28.61<br>**AVG**<br>TFLEX<br>35.93<br>X(ConE)<br>30.13<br>X-1F<br>36.43<br>X-logic<br>35.98|**e2i_Pe**<br>**Pe_e2i**<br>**Pe_aPt**<br>**Pe_bPt**<br>**Pe_at2i**<br>**Pe_bt2i**<br>98.86<br>36.77<br>8.66<br>9.74<br>7.90<br>7.78<br>87.31<br>28.72<br>11.01<br>10.40<br>11.21<br>11.27<br>98.87<br>36.00<br>10.04<br>10.21<br>8.30<br>8.06<br>97.64<br>36.96<br>10.09<br>10.27<br>8.76<br>8.93<br>**t2i_Pe**<br>**Pe_t2i**<br>**Pt_sPe**<br>**Pt_oPe**<br>**Pt_se2i**<br>**Pt_oe2i**<br>**between**<br>96.62<br>64.50<br>4.32<br>10.58<br>8.20<br>7.95<br>2.57<br>95.40<br>63.04<br>3.77<br>8.82<br>6.44<br>5.76<br>3.32<br>97.24<br>70.32<br>4.88<br>10.73<br>12.24<br>11.55<br>2.55<br>96.35<br>64.31<br>4.97<br>10.35<br>10.59<br>10.35<br>2.34|



29 

Table 15: MRR results on ICEWS05-15. The group avg _x_ wraps to two rows. **AVG** denotes average performance under all query types. 

|**Model**<br>avg_e_|**Pe**<br>**Pe2**<br>**Pe3**<br>**e2i**<br>**e3i**|
|---|---|
|Query2box<br>24.00<br>BetaE<br>31.33<br>ConE<br>40.93<br>ConE(temporal)<br>40.74<br>FLEX<br>38.96<br>TFLEX<br>48.99<br>X(ConE)<br>36.29<br>X-1F<br>49.90<br>X-logic<br>44.80|25.94<br>16.62<br>14.22<br>17.68<br>45.52<br>35.78<br>21.47<br>18.18<br>18.10<br>63.11<br>42.67<br>29.39<br>24.79<br>26.15<br>81.64<br>42.64<br>29.30<br>24.76<br>25.35<br>81.65<br>41.60<br>29.80<br>24.58<br>24.37<br>74.46<br>43.04<br>36.28<br>33.89<br>41.17<br>90.60<br>39.90<br>26.62<br>22.85<br>22.52<br>69.56<br>43.07<br>36.87<br>34.96<br>40.85<br>93.76<br>40.57<br>31.47<br>30.67<br>35.80<br>85.47|
|avg_t_|**Pt**<br>**aPt**<br>**bPt**<br>**Pe_Pt**<br>**Pt_sPe_Pt**<br>**Pt_oPe_Pt**<br>**t2i**<br>**t3i**|
|TFLEX<br>4.39<br>X(ConE)<br>4.41<br>X-1F<br>4.43<br>X-logic<br>3.29|10.62<br>0.38<br>0.38<br>7.29<br>2.63<br>1.83<br>3.85<br>8.15<br>10.98<br>0.38<br>0.38<br>6.75<br>2.16<br>1.63<br>4.22<br>8.81<br>10.71<br>0.38<br>0.38<br>7.40<br>2.73<br>1.86<br>3.80<br>8.16<br>8.57<br>0.38<br>0.39<br>6.37<br>1.69<br>1.41<br>2.49<br>5.06|
|avg_e,Ce_|**e2i_N**<br>**e3i_N**<br>**Pe_e2i_N**<br>**e2i_PeN**<br>**e2i_NPe**|
|BetaE<br>29.70<br>ConE<br>43.52<br>ConE(temporal)<br>43.34<br>FLEX<br>42.10<br>TFLEX<br>46.17<br>X(ConE)<br>38.12<br>X-1F<br>46.11<br>X-logic<br>41.92|21.68<br>71.08<br>20.31<br>21.11<br>14.30<br>43.04<br>96.08<br>28.41<br>28.75<br>21.30<br>42.85<br>96.08<br>28.33<br>28.62<br>20.83<br>36.22<br>97.47<br>27.93<br>27.23<br>21.64<br>41.34<br>96.69<br>34.29<br>34.63<br>23.88<br>33.70<br>88.83<br>24.80<br>25.60<br>17.64<br>38.06<br>96.82<br>35.76<br>36.20<br>23.73<br>36.57<br>91.16<br>30.37<br>31.42<br>20.07|
|avg_t,Ct_|**t2i_N**<br>**t3i_N**<br>**Pe_t2i_N**<br>**t2i_PtN**<br>**t2i_NPt**|
|TFLEX<br>30.16<br>X(ConE)<br>29.49<br>X-1F<br>30.26<br>X-logic<br>28.34|16.09<br>98.40<br>31.39<br>3.35<br>1.58<br>16.55<br>98.46<br>28.23<br>2.53<br>1.69<br>16.05<br>98.37<br>32.07<br>3.27<br>1.56<br>12.94<br>96.03<br>29.47<br>2.18<br>1.07|
|avg_{Ue}_|**e2u**<br>**Pe_e2u**|
|BetaE<br>21.54<br>ConE<br>43.02<br>ConE(temporal)<br>43.14<br>FLEX<br>44.38<br>TFLEX<br>54.37<br>X(ConE)<br>36.37<br>X-1F<br>54.05<br>X-logic<br>45.36|20.95<br>22.13<br>37.21<br>48.83<br>37.05<br>49.24<br>35.72<br>53.04<br>52.99<br>55.75<br>29.89<br>42.86<br>52.47<br>55.64<br>43.84<br>46.88|
|avg_{Ut}_|**t2u**<br>**Pe_t2u**|
|TFLEX<br>28.69<br>X(ConE)<br>26.40<br>X-1F<br>27.70<br>X-logic<br>23.39|44.99<br>12.39<br>40.35<br>12.46<br>43.38<br>12.02<br>37.07<br>9.71|
|avg_x_<br>TFLEX<br>24.26<br>X(ConE)<br>21.69<br>X-1F<br>24.41<br>X-logic<br>21.95<br>**AVG**<br>TFLEX<br>33.72<br>X(ConE)<br>27.54<br>X-1F<br>33.98<br>X-logic<br>29.86|**e2i_Pe**<br>**Pe_e2i**<br>**Pe_aPt**<br>**Pe_bPt**<br>**Pe_at2i**<br>**Pe_bt2i**<br>94.23<br>57.16<br>5.07<br>4.56<br>4.38<br>3.93<br>81.36<br>38.20<br>5.63<br>5.03<br>5.80<br>5.36<br>94.26<br>55.97<br>5.14<br>4.79<br>4.37<br>3.95<br>87.48<br>50.11<br>4.28<br>3.95<br>3.96<br>3.55<br>**t2i_Pe**<br>**Pe_t2i**<br>**Pt_sPe**<br>**Pt_oPe**<br>**Pt_se2i**<br>**Pt_oe2i**<br>**between**<br>92.25<br>48.35<br>0.46<br>2.82<br>0.98<br>1.02<br>0.22<br>91.92<br>43.83<br>0.58<br>2.18<br>0.95<br>1.02<br>0.14<br>92.21<br>50.77<br>0.46<br>2.89<br>1.13<br>1.23<br>0.17<br>86.17<br>40.78<br>0.62<br>1.98<br>1.10<br>1.21<br>0.16|



30 

Table 16: MRR results on GDELT-500. The group avg _x_ wraps to two rows. **AVG** denotes average performance under all query types. 

|**Model**<br>avg_e_|**Pe**<br>**Pe2**<br>**Pe3**<br>**e2i**<br>**e3i**|
|---|---|
|Query2box<br>9.67<br>BetaE<br>14.75<br>ConE<br>18.44<br>ConE(temporal)<br>18.51<br>FLEX<br>19.07<br>TFLEX<br>19.60<br>X(ConE)<br>17.83<br>X-1F<br>17.92<br>X-logic<br>17.36|10.70<br>4.73<br>3.41<br>12.28<br>17.22<br>14.02<br>5.76<br>4.71<br>18.61<br>30.67<br>16.65<br>6.18<br>4.70<br>23.11<br>41.55<br>17.61<br>5.62<br>4.57<br>24.52<br>40.22<br>17.72<br>5.78<br>4.67<br>24.30<br>42.85<br>18.50<br>5.76<br>4.68<br>25.94<br>43.14<br>17.05<br>5.76<br>4.66<br>24.02<br>37.66<br>16.18<br>6.35<br>4.84<br>22.41<br>39.80<br>15.80<br>6.22<br>4.85<br>22.45<br>37.47|
|avg_t_|**Pt**<br>**aPt**<br>**bPt**<br>**Pe_Pt**<br>**Pt_sPe_Pt**<br>**Pt_oPe_Pt**<br>**t2i**<br>**t3i**|
|TFLEX<br>5.38<br>X(ConE)<br>3.16<br>X-1F<br>5.49<br>X-logic<br>5.75|6.49<br>3.25<br>3.27<br>6.16<br>3.06<br>2.88<br>6.78<br>11.17<br>2.15<br>3.39<br>3.33<br>6.96<br>2.29<br>2.31<br>2.34<br>2.49<br>6.62<br>3.30<br>3.30<br>5.84<br>3.07<br>3.05<br>7.08<br>11.66<br>7.15<br>3.34<br>3.31<br>5.95<br>3.14<br>3.06<br>7.52<br>12.54|
|avg_e,Ce_|**e2i_N**<br>**e3i_N**<br>**Pe_e2i_N**<br>**e2i_PeN**<br>**e2i_NPe**|
|BetaE<br>11.15<br>ConE<br>12.67<br>ConE(temporal)<br>12.67<br>FLEX<br>13.35<br>TFLEX<br>13.52<br>X(ConE)<br>12.34<br>X-1F<br>12.13<br>X-logic<br>12.11|11.22<br>24.17<br>5.53<br>5.46<br>9.39<br>12.69<br>29.09<br>5.47<br>5.42<br>10.66<br>13.42<br>27.98<br>5.31<br>5.00<br>11.63<br>13.57<br>30.25<br>5.47<br>5.47<br>11.99<br>14.46<br>30.09<br>5.51<br>5.19<br>12.36<br>13.42<br>26.28<br>5.54<br>5.35<br>11.14<br>11.77<br>27.63<br>5.70<br>5.40<br>10.14<br>12.32<br>27.00<br>5.70<br>5.55<br>10.01|
|avg_t,Ct_|**t2i_N**<br>**t3i_N**<br>**Pe_t2i_N**<br>**t2i_PtN**<br>**t2i_NPt**|
|TFLEX<br>6.31<br>X(ConE)<br>3.93<br>X-1F<br>6.50<br>X-logic<br>6.86|5.92<br>9.01<br>9.30<br>2.76<br>4.56<br>2.24<br>2.43<br>10.00<br>2.26<br>2.74<br>5.90<br>9.70<br>9.15<br>2.85<br>4.90<br>6.51<br>10.94<br>8.78<br>2.94<br>5.14|
|avg_{Ue}_|**e2u**<br>**Pe_e2u**|
|BetaE<br>6.20<br>ConE<br>6.96<br>ConE(temporal)<br>7.30<br>FLEX<br>7.44<br>TFLEX<br>7.58<br>X(ConE)<br>7.41<br>X-1F<br>6.92<br>X-logic<br>6.91|4.64<br>7.75<br>4.68<br>9.25<br>4.50<br>10.09<br>4.47<br>10.42<br>4.62<br>10.55<br>4.64<br>10.19<br>4.80<br>9.04<br>4.74<br>9.09|
|avg_{Ut}_|**t2u**<br>**Pe_t2u**|
|TFLEX<br>6.71<br>X(ConE)<br>6.35<br>X-1F<br>6.59<br>X-logic<br>6.80|9.03<br>4.38<br>10.39<br>2.31<br>8.60<br>4.58<br>8.85<br>4.76|
|avg_x_<br>TFLEX<br>6.17<br>X(ConE)<br>6.17<br>X-1F<br>6.47<br>X-logic<br>6.64<br>**AVG**<br>TFLEX<br>9.32<br>X(ConE)<br>8.17<br>X-1F<br>8.86<br>X-logic<br>8.92|**e2i_Pe**<br>**Pe_e2i**<br>**Pe_aPt**<br>**Pe_bPt**<br>**Pe_at2i**<br>**Pe_bt2i**<br>17.53<br>6.38<br>4.73<br>4.58<br>4.47<br>4.58<br>17.53<br>6.44<br>6.45<br>6.18<br>6.02<br>5.97<br>18.13<br>6.82<br>5.48<br>5.35<br>5.19<br>5.31<br>17.45<br>6.79<br>5.56<br>5.44<br>5.35<br>5.37<br>**t2i_Pe**<br>**Pe_t2i**<br>**Pt_sPe**<br>**Pt_oPe**<br>**Pt_se2i**<br>**Pt_oe2i**<br>**between**<br>8.36<br>15.83<br>3.11<br>2.81<br>2.93<br>3.07<br>1.85<br>2.45<br>17.97<br>2.70<br>2.24<br>2.20<br>2.29<br>1.76<br>8.80<br>15.13<br>2.92<br>2.72<br>3.11<br>3.29<br>1.87<br>10.28<br>15.28<br>3.27<br>3.11<br>3.23<br>3.36<br>1.79|



31 

**RQ1** : How about incorporating temporal information into the static query embedding, without considering time logic? 

**RQ2** : Further, how about enhancing the static query embedding with temporal logic? 

To address research question ( **RQ1** ), we propose a variant of ConE called **ConE(temporal)** that incorporates temporal information into the static query embedding. The only difference between ConE and ConE(temporal) is the projection operator used. In ConE(temporal), the projection operator incorporates temporal information through a technique called TTransE, which uses the temporal embedding of timestamps. In terms of formulation, the projection operator for ConE(temporal) is given by _Pe_ ( **V** _q,_ **r** _,_ **V** _t_ ) = _g_ ( **MLP** ( **V** _q_ + **r** + **V** _t_ )), while for ConE it is given by _Pe_ ( **V** _q,_ **r** _,_ **V** _t_ ) = _g_ ( **MLP** ( **V** _q_ + **r** )). Note that **V** _q,_ **r** _,_ **V** _t_ are cone embeddings. 

From Table 17, we observe that there is slightly improvement in terms of MRR when comparing ConE(temporal) with ConE. It shows that the temporal information can help to improve the performance of static query embedding on answering entities over TKGs. However, the improvement is not significant, which indicates that simply aware of the temporal information is not enough to promote the static query embedding to efficient temporal one. 

Next, we wonder if it could help when we utilize more queries involving time logic, which provides more temporal information. Therefore, we address research question ( **RQ2** ) by proposing a variant of ConE and TFLEX, called **X(ConE)** , that enhances the static query embedding with temporal logic. **X(ConE)** replaces the entity part of temporal feature-logic embedding with ConE, as is introduced in Section 5.2. In the other view, **X(ConE)** is based on ConE, using cone embedding concatenated with the time part of temporal feature-logic embedding, which can handle temporal logic. 

From Table 17, to our surprise, **X(ConE)** performs worse than ConE and TFLEX. The concatenation of cone embedding and feature-logic embedding does not help to improve the overall performance of temporal complex reasoning. This is because cone embedding is geometric, while temporal feature-logic embedding is fuzzy. We think there is a mismatch or semantic conflict between the two embeddings, which leads to the poor performance of **X(ConE)** . 

Be aware that we do not deny the importance of temporal logic. In fact, temporal logic is very important for temporal complex reasoning (See **Ablation on time part** in Section 5.2). However, we think that it is not a good idea to promote the static query embedding to temporal one by simply concatenating the geometric embedding with the fuzzy embedding. The right way to promote remains an open question. 

Table 17: Average MRR results for TFLEX and the variants of ConE. **AVG** _e_ denotes average of avg _e_ , avg _e,Ce_ and avg _Ue_ . **AVG** denotes average of all groups. 

|**Dataset**|**Model**|avg_e_|avg_e,Ce_|avg_Ue_|avg_t_|avg_t,Ct_|avg_Ut_|avg_x_|**AVG**_e_|**AVG**|
|---|---|---|---|---|---|---|---|---|---|---|
||ConE|41.94|44.88|26.47|||||37.76||
|ICEWS14|ConE(temporal)<br>X(ConE)|42.23<br>40.93|44.94<br>42.15|27.63<br>25.46|16.41|35.24|24.07|26.65|38.27<br>36.18|30.13|
||TFLEX|56.79|50.82|35.74|17.56|36.37|26.24|28.03|47.78|35.93|
||ConE|40.93|43.52|43.02|||||42.49||
|ICEWS05-15|ConE(temporal)<br>X(ConE)|40.74<br>36.29|43.34<br>38.12|43.14<br>36.37|4.41|29.49|26.40|21.69|42.41<br>36.93|27.54|
||TFLEX|48.99|46.17|54.37|4.39|30.16|28.69|24.26|49.84|33.72|
||ConE|18.44|12.67|6.96|||||12.69||
|GDELT-500|ConE(temporal)<br>X(ConE)|18.51<br>17.83|12.67<br>12.34|7.30<br>7.41|3.16|3.93|6.35|6.17|12.83<br>12.53|8.17|
||TFLEX|19.60|13.52|7.58|5.38|6.31|6.71|6.17|13.57|9.32|



## **D Visualization of Semantic Changes by Neural Temporal Operators** 

In this section, we present more visualization of the semantic changes of temporal operators in TFLEX. The examples are randomly selected in the test set of ICEWS14. The visualization is shown in Figure 10, Figure 11 and Figure 12. We attach the anchors and the ground-truth answers in the corresponding caption of figures. From the figures, we observe that the model correctly ranks the hard answers at top when answering query **Pt** . It can also be seen from the figure that the semantic changes of temporal operators **aPt** and **bPt** are intuitive and reasonable. 

32 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2023_TFLEX_Temporal_Feature-Logic_Embedding_Framework_for_Complex_Reasoning_over_Temporal_Knowledge_Graph_images/nips_2023_TFLEX_Temporal_Feature-Logic_Embedding_Framework_for_Complex_Reasoning_over_Temporal_Knowledge_Graph.pdf-0033-00.png)


Figure 10: Score distributions of **Pt** , **bPt** and **aPt** , where _s_ is **Citizen (India)** , _r_ is **Criticize or denounce** , _o_ is **Sadhu (India)** , ground-truth answer of **Pt** is **2014-03-27** . 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2023_TFLEX_Temporal_Feature-Logic_Embedding_Framework_for_Complex_Reasoning_over_Temporal_Knowledge_Graph_images/nips_2023_TFLEX_Temporal_Feature-Logic_Embedding_Framework_for_Complex_Reasoning_over_Temporal_Knowledge_Graph.pdf-0033-02.png)


Figure 11: Score distributions of **Pt** , **bPt** and **aPt** , where _s_ is **Citizen (India)** , _r_ is **Use unconventional violence** , _o_ is **Chief Secretary Chandra** , ground-truth answer of **Pt** is **2014-05-11** . 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2023_TFLEX_Temporal_Feature-Logic_Embedding_Framework_for_Complex_Reasoning_over_Temporal_Knowledge_Graph_images/nips_2023_TFLEX_Temporal_Feature-Logic_Embedding_Framework_for_Complex_Reasoning_over_Temporal_Knowledge_Graph.pdf-0033-04.png)


Figure 12: Score distributions of **Pt** , **bPt** and **aPt** , where _s_ is **Mohammad Javad Zarif** , _r_ is **Appeal for economic cooperation** , _o_ is **United Arab Emirates** , ground-truth answer of **Pt** is **2014-04-14** . 

33 

## **E Explaining Answers with Temporal Feature-Logic Framework** 

In this section, we present the explanation of answers given by TFLEX. Below we take the query " **e2i** " as an example. The temporal query " **e2i** " is defined as follows: _q_ [ _V_ ?] = _V_ ? _, ∃Va, Vb, r_ 1( _e_ 1 _, Va, t_ 1) _∧ r_ 2( _e_ 2 _, Vb, t_ 2). Let’s consider the specific query: "Who was consulted by Mohammad Javad Zarif on 2014-04-07 and consulted Mohammad Javad Zarif on 2014-04-07?" In this example, _e_ 1 = "Mohammad Javad Zarif", _r_ 1 = "consulted by", _t_ 1 = "2014-04-07", _r_ 2 = "consulted", and _t_ 2 = "201404-07". We use TFLEX to execute the query and obtain the answers. The answers are categorized into easy, hard, and wrong ones. Table 18 presents the five most likely answers and their rankings for interpretation. From the table, we observe that TFLEX ranks the answers "Mohammad Javad Zarif", "Sebastian Kurz", "China" and "Catherine Ashton" high, while ranking the wrong answer "Iran" low. TFLEX even infers a hard answer "Catherine Ashton" with higher score than two easy answers "Sebastian Kurz" and "China". This demonstrates that TFLEX successfully identifies the correct answers using complex reasoning and distinguishes them from the wrong ones. 

Navigation: Pt(13), Pe(14), Pe2(15), Pe3(16), Pe_Pt(17), e2i(18), e3i(19), e2i_Pe(20), Pe_e2i(21), Pe_t2i(22), t2i(23), t3i(24), t2i_Pe(25), Pt_se2i(26), Pt_oe2i(27), t2i_NPt(28), t2i_PtN(29), Pe_t2i_PtPe_NPt(30), e2i_NPe(31), e2i_PeN(32), Pe_e2i_Pe_NPe(33), e2i_N(34), e3i_N(35), e2u(36), Pe_e2u(37), Pt_sPe(38), Pt_oPe(39), t2i_N(40), t3i_N(41), t2u(42), Pe_t2u(43), Pe_aPt(44), Pe_bPt(45), Pe_at2i(46), Pe_bt2i(47), between(48). 

|**?**<br>**Pt**<br>North Atlantic Treaty Organization<br>Consult<br>Taavi Rõivas|**Query Sentence**<br>When did North Atlantic Treaty Organization consult Taavi Rõi-<br>vas?|
|---|---|
||**Query DNF**<br>_q_[_V_?] =_V_?_, r_1(_e_1_, e_2_, V_?)|
||**Rank**<br>**Query Answers**<br>**Correctness**<br>**Answer Type**|
||1<br>2014-11-20<br>✔<br>Hard<br>2<br>2014-12-03<br>✘<br>-<br>3<br>2014-06-16<br>✘<br>-<br>4<br>2014-06-25<br>✘<br>-<br>5<br>2014-10-13<br>✘<br>-|



Figure 13: Intermediate variable assignments and ranks for example Pt query. Correctness indicates whether the answer belongs to the ground-truth set of answers. 

|**Pe**<br>Crit|**?**<br>Philippines<br>icize or denounce<br>2014-04-01|**Query Sentence**<br>Philippines denounced or criticized who on 2014-04-01?|
|---|---|---|
|||**Query DNF**<br>_q_[_V_?] =_V_?_, r_1(_e_1_, V_?_, t_1)|
|||**Rank**<br>**Query Answers**<br>**Correctness**<br>**Answer Type**|
|||1<br>China<br>✔<br>Hard<br>2<br>Japan<br>✘<br>-<br>3<br>Malaysia<br>✘<br>-<br>4<br>Philippines<br>✘<br>-<br>5<br>Iran<br>✘<br>-|



Figure 14: Intermediate variable assignments and ranks for example Pe query. Correctness indicates whether the answer belongs to the ground-truth set of answers. 

|||||**Query Sentence**<br>On 2014-04-04, who consulted the man who was appealed to or<br>requested by the Head of Government (Latvia) on 2014-08-01?|
|---|---|---|---|---|
|<br>**P**||**e2**|||
||||**?**|**Temporal Query**<br>_q_[_V_?] =_V_?_, ∃Va, r_1(_e_1_, Va, t_1)_∧r_2(_Va, V_?_, t_2)|
|||||**Rank**<br>**Query Answers**<br>**Correctness**<br>**Answer Type**|
|||Head of Government (Latvia)<br>Make an appeal or request<br>2014-08-01<br>2014-04-04|||
|||||1<br>François Hollande<br>✔<br>Easy<br>2<br>Taavi Rõivas<br>✔<br>Easy<br>3<br>Jyrki Katainen<br>✔<br>Hard<br>4<br>Angela Merkel<br>✘<br>-<br>5<br>Head of Government (Latvia)<br>✘<br>-|
||||||



Figure 15: Intermediate variable assignments and ranks for example Pe2 query. Correctness indicates whether the answer belongs to the ground-truth set of answers. 

34 

||**Query Sentence**<br>Who was criticized or denounced on 2014-08-26 by the person<br>who made a statement on 2014-02-05 at the place where the<br>presidential movement made an appeal or request on 2014-04-<br>24?|
|---|---|
|**Pe3**<br>Presidential Movement<br>Make an appeal or request<br>2014-04-24<br>**?**<br>2014-02-05<br>Criticize or denounce<br>2014-08-26||
||**Temporal Query**<br>_q_[_V_?]<br>=<br>_V_?_, ∃Va, Vb, r_1(_e_1_, Va, t_1)<br>_∧_<br>_r_2(_Va, Vb, t_2)<br>_∧_<br>_r_3(_Vb, V_?_, t_3)|
||**Rank**<br>**Query Answers**<br>**Correctness**<br>**Answer Type**|
||<br> <br>1<br>Government (France)<br>✔<br>Easy<br>2<br>France<br>✘<br>-<br>3<br>François Hollande<br>✔<br>Hard<br>4<br>Angela Merkel<br>✘<br>-<br>5<br>Citizen (International)<br>✘<br>-|
|||



Figure 16: Intermediate variable assignments and ranks for example Pe3 query. Correctness indicates whether the answer belongs to the ground-truth set of answers. 

||**Query Sentence**<br>Who arrested, detained, or charged the citizen (Thailand) with<br>legal action after Police (Greece) arrested, detained, or charged<br>the illegal immigrant (Greece) with legal action?|
|---|---|
|**?**<br>**Pe_Pt**<br>Citizen (Thailand)<br>Police (Greece)<br>Illegal Immigrant (Greece)<br>Arrest, detain, or charge with legal action||
||**Temporal Query**<br>_q_[_V_?] =_V_?_, ∃Ta, r_1(_e_1_, V_?_, Ta_)_∧r_2(_e_2_, e_3_, Ta_)|
||**Rank**<br>**Query Answers**<br>**Correctness**<br>**Answer Type**|
||1<br>Military (Thailand)<br>✔<br>Easy<br>2<br>Municipal Court (Thailand)<br>✔<br>Easy<br>3<br>Thailand<br>✔<br>Hard<br>4<br>National Council for Peace and Order of Thailand<br>✘<br>-<br>5<br>Police (Cambodia)<br>✘<br>-|
|||



Figure 17: Intermediate variable assignments and ranks for example Pe_Pt query. Correctness indicates whether the answer belongs to the ground-truth set of answers. 

|Consult<br>**And**<br>**?**<br>**e2i**<br>Mohammad Javad Zarif<br>2014-04-07<br>2014-04-07<br>Mohammad Javad Zarif|Consult<br>**And**<br>**?**<br>**e2i**<br>Mohammad Javad Zarif<br>2014-04-07<br>2014-04-07<br>Mohammad Javad Zarif||
|---|---|---|
|||**Query Sentence**<br>Who was consulted by Mohammad Javad Zarif on 2014-04-07<br>|
||**And**<br>**?**|and consulted Mohammad Javad Zarif on 2014-04-07?|
|||**Temporal Query**<br>_q_[_V_?] =_V_?_, r_1(_e_1_, V_?_, t_1)_∧r_2(_e_2_, V_?_, t_2))|
|||**Rank**<br>**Query Answers**<br>**Correctness**<br>**Answer Type**|
|||1<br>Mohammad Javad Zarif<br>✔<br>Easy<br>2<br>Catherine Ashton<br>✔<br>Hard<br>3<br>Sebastian Kurz<br>✔<br>Easy<br>4<br>China<br>✔<br>Easy<br>5<br>Iran<br>✘<br>-|
||||



Figure 18: Intermediate variable assignments and ranks for example e2i query. Correctness indicates whether the answer belongs to the ground-truth set of answers. 

|**And**<br>**?**<br>**e3i**<br>Joseph Robinette Biden<br>2014-10-05<br>Joseph Robinette Biden<br>Discuss by telephone<br>2014-10-05<br>Saud bin Faisal bin Abdul-Aziz<br>Make a visit<br>2014-01-20||
|---|---|
||**Query Sentence**<br>Where was Joseph Robinette Biden discussed by telephone on<br>2014-10-05 and visited by Saud bin Faisal bin Abdul-Aziz on<br>2014-01-20?|
||**Temporal Query**<br>_q_[_V_?] =_V_?_, r_1(_e_1_, V_?_, t_1)_∧r_2(_e_2_, V_?_, t_2)_∧r_3(_e_3_, V_?_, t_3)|
||**Rank**<br>**Query Answers**<br>**Correctness**<br>**Answer Type**|
||1<br>UAE Armed Forces<br>✔<br>Easy<br>2<br>United Arab Emirates<br>✔<br>Hard<br>3<br>Middle East<br>✘<br>-<br>4<br>Morocco<br>✘<br>-<br>5<br>Abdel Fattah Al-Sisi<br>✘<br>-|
|||



Figure 19: Intermediate variable assignments and ranks for example e3i query. Correctness indicates whether the answer belongs to the ground-truth set of answers. 

35 

|**e2i_Pe**<br>**And**<br>**?**<br>Obaid Humaid Al Tayer<br>Make statement<br>2014-06-11<br>Make a visit<br>2014-06-04<br>Naval (Yemen)<br>Occupy territory<br>2014-03-01|**e2i_Pe**<br>**And**<br>**?**<br>Obaid Humaid Al Tayer<br>Make statement<br>2014-06-11<br>Make a visit<br>2014-06-04<br>Naval (Yemen)<br>Occupy territory<br>2014-03-01|**Query Sentence**<br>Where was occupied territory on 2014-03-01 and was visited on<br>2014-06-04 by the person who Obaid Humaid Al Tayer made a<br>|
|---|---|---|
||**And**<br>**?**|statement on 2014-06-11?<br>**Temporal Query**<br>_q_[_V_?] =_V_?_, ∃Va, r_1(_e_1_, Va, t_1)_∧r_2(_Va, V_?_, t_2)_∧r_3(_e_2_, V_?_, t_3)|
|||**Rank**<br>**Query Answers**<br>**Correctness**<br>**Answer Type**<br>1<br>Iran<br>✔<br>Easy<br>2<br>China<br>✘<br>-<br>3<br>Iraq<br>✘<br>-<br>4<br>France<br>✘<br>-<br>5<br>Afghanistan<br>✘<br>-|



Figure 20: Intermediate variable assignments and ranks for example e2i_Pe query. Correctness indicates whether the answer belongs to the ground-truth set of answers. 

|**And**<br>**?**<br>**Pe_e2i**<br>Labor Union (Croatia)<br>Make an appeal or request<br>2014-02-03<br>Government (Croatia)<br>2014-03-25<br>Consult<br>2014-03-25|**And**<br>**?**<br>**Pe_e2i**<br>Labor Union (Croatia)<br>Make an appeal or request<br>2014-02-03<br>Government (Croatia)<br>2014-03-25<br>Consult<br>2014-03-25|**Query Sentence**<br>Where was consulted on 2014-03-25 by the place where Labor<br>Union (Croatia) made an appeal or request on 2014-02-03 and|
|---|---|---|
||**And**<br>**?**|<br>was consulted by Government (Croatia) on 2014-03-25?<br>**Temporal Query**<br>_q_[_V_?] =_V_?_, ∃Va, r_1(_e_1_, Va, t_1)_∧r_2(_e_2_, Va, t_2)_∧r_3(_Va, V_?_, T_3)|
|||**Rank**<br>**Query Answers**<br>**Correctness**<br>**Answer Type**<br>1<br>Labor Union (Croatia)<br>✔<br>Hard<br>2<br>Business (Croatia)<br>✘<br>-<br>3<br>Government (Croatia)<br>✔<br>Easy<br>4<br>Ministry (Croatia)<br>✔<br>Easy<br>5<br>Branko Grcic<br>✘<br>-|



Figure 21: Intermediate variable assignments and ranks for example Pe_e2i query. Correctness indicates whether the answer belongs to the ground-truth set of answers. 

|**?**<br>**TimeAnd**<br>Government (South Africa)<br>Gazprom<br>China<br>Mzukisi Fatyela<br>Make statement<br>Police (South Africa)<br>**Pe_t2i**|**Query Sentence**<br>Who signed a formal agreement with Government (South Africa)<br>at the time when Gazprom signed a formal agreement with China<br>and Mzukisi Fatyela made a statement to Police (South Africa)?<br>**Temporal Query**<br>_q_[_V_?] =_V_?_, ∃Ta, r_1(_e_1_, V_?_, Ta_)_∧r_2(_e_2_, e_3_, Ta_)_∧r_3(_e_5_, e_5_, Ta_)<br>**Rank**<br>**Query Answers**<br>**Correctness**<br>**Answer Type**<br>1<br>South Africa<br>✔<br>Easy<br>2<br>Tom Motsoahae Thabane<br>✘<br>-<br>3<br>China<br>✘<br>-<br>4<br>Bank (China)<br>✘<br>-<br>5<br>Henry Rotich<br>✘<br>-|
|---|---|



Figure 22: Intermediate variable assignments and ranks for example Pe_t2i query. Correctness indicates whether the answer belongs to the ground-truth set of answers. 

|**t2**|**t2**|**i**|**i**|**Query Sentence**<br>At what time did Barack Obama negotiate with and consult Xi<br>Jinping?<br>**Temporal Query**<br>_q_[_T_?] =_T_?_, r_1(_e_1_, e_2_, T_?)_∧r_2(_e_3_, e_4_, T_?))<br>**Rank**<br>**Query Answers**<br>**Correctness**<br>**Answer Type**<br>1<br>2014-11-15<br>✔<br>Easy<br>2<br>2014-03-24<br>✔<br>Easy<br>3<br>2014-11-12<br>✔<br>Easy<br>4<br>2014-11-11<br>✔<br>Easy<br>d 5<br>2014-11-14<br>✔<br>Easy|
|---|---|---|---|---|
||||||
||||||
|||Xi Jinping<br>Barack Obama<br>Xi Jinping<br>Barack Obama|||



Figure 23: Intermediate variable assignments and ranks for example t2i query. Correctness indicates whether the answer belongs to the ground-truth set of answers. 

36 

|**TimeAnd**<br>**?**<br>**t3i**<br>Japan<br>Engage in diplomatic cooperation<br>Barack Obama<br>South Korea<br>Make statement<br>Mohammad Javad Zarif<br>France<br>François Hollande|**Query Sentence**<br>When did Japan engage in diplomatic cooperation with South<br>Korea, Mohammad Javad Zarif make a statement to Iran, and<br>François Hollande make a statement to France?<br>**Temoral Quer**<br>[_T_] =_T r_(_e e T_)_∧r_(_e e T_)_∧r_(_e e T_)|
|---|---|
||**p y**<br>_q_?  ?_,_11_,_2_,_? 23_,_4_,_? 15_,_6_,_?<br>**Rank**<br>**Query Answers**<br>**Correctness**<br>**Answer Type**|
||1<br>2014-09-18<br>✔<br>Hard<br>2<br>2014-09-20<br>✔<br>Easy<br>3<br>2014-08-22<br>✔<br>Easy<br>4<br>2014-07-01<br>✔<br>Hard<br>5<br>2014-08-16<br>✘<br>-|



Figure 24: Intermediate variable assignments and ranks for example t3i query. Correctness indicates whether the answer belongs to the ground-truth set of answers. 

|**TimeAnd**<br>**?**<br>**t2i_Pe**<br>Finance / Economy / Commerce / Trade Ministry (Oman)<br>Share intelligence or information<br>2014-01-06<br>Citizen (Oman)<br>Activist (Somalia)<br>Armed Gang (Somalia)|**Query Sentence**<br>When did Activist (Somalia) fght with small arms and light<br>weapons with Armed Gang (Somalia) while Citizen (Oman) en-<br>gage in a symbolic act with the person with whom fnance /<br>Economy / Commerce / Trade Ministry (Oman) shared intelli-<br>gence or information on 2014-01-06?<br>**Temoral Quer**<br>[_T_] =_T ∃V r_(_e V t_)_∧r_(_V e T_)_∧r_(_e e T_)|
|---|---|
||**p y**<br>_q_?  ?_, a, _11_,a, _1 2_a, _2_,_? 33_,_4_,_?<br>**Rank**<br>**Query Answers**<br>**Correctness**<br>**Answer Type**|
||1<br>2014-01-30<br>✔<br>Easy<br>2<br>2014-02-05<br>✘<br>-<br>3<br>2014-01-29<br>✘<br>-<br>4<br>2014-02-07<br>✘<br>-<br>5<br>2014-02-04<br>✘<br>-|



Figure 25: Intermediate variable assignments and ranks for example t2i_Pe query. Correctness indicates whether the answer belongs to the ground-truth set of answers. 

|**Pt_le2i**<br>**And**<br>**?**<br>Iran<br>2014-03-09<br>Iran<br>Host a visit<br>2014-03-09<br>Engage in negotiation<br>Iran|**Pt_le2i**<br>**And**<br>**?**<br>Iran<br>2014-03-09<br>Iran<br>Host a visit<br>2014-03-09<br>Engage in negotiation<br>Iran|**Query Sentence**<br>When did Iran negotiate with the country that visited Iran on|
|---|---|---|
||**And**<br>**?**|2014-03-09?|
|||**Temporal Query**<br>_q_[_T_?] =_T_?_, ∃Va, r_1(_e_1_, Va, t_1)_∧r_2(_e_2_, Va, t_2)_∧r_3(_Va, e_3_, T_?)|
||||
|||**Rank**<br>**Query Answers**<br>**Correctness**<br>**Answer Type**|
|||1<br>2014-03-10<br>✔<br>Easy<br>2<br>2014-03-08<br>✔<br>Easy<br>3<br>2014-06-16<br>✔<br>Easy<br>4<br>2014-03-13<br>✔<br>Easy<br>5<br>2014-03-09<br>✔<br>Easy|



Figure 26: Intermediate variable assignments and ranks for example Pt_se2i query. Correctness indicates whether the answer belongs to the ground-truth set of answers. 

|**And**<br>**?**<br>**And**<br>**Pt_re2i**<br>Armenia<br>Cooperate militarily<br>Armenia<br>Cooperate militarily<br>2014-12-08<br>Armenia<br>Cooperate militarily<br>2014-07-02|**And**<br>**?**<br>**And**<br>**Pt_re2i**<br>Armenia<br>Cooperate militarily<br>Armenia<br>Cooperate militarily<br>2014-12-08<br>Armenia<br>Cooperate militarily<br>2014-07-02|**Query Sentence**<br>When did Armenia cooperate militarily with the country that<br>|
|---|---|---|
||**And**<br>**?**<br>**And**|cooperates militarily with Armenia on 2014-12-08 and 2014-07-<br>02?|
|||**Temporal Query**<br>_q_[_V_?] =_V_?_, ∃Va, r_1(_e_1_, Va, T_?)_∧r_2(_e_2_, Va, t_1)_∧r_3(_e_3_, Va, t_2)|
|||**Rank**<br>**Query Answers**<br>**Correctness**<br>**Answer Type**|
|||1<br>2014-12-08<br>✔<br>Easy<br>2<br>2014-07-02<br>✔<br>Easy<br>3<br>2014-05-06<br>✔<br>Easy<br>4<br>2014-10-26<br>✔<br>Easy<br>5<br>2014-07-17<br>✘<br>-|



Figure 27: Intermediate variable assignments and ranks for example Pt_oe2i query. Correctness indicates whether the answer belongs to the ground-truth set of answers. 

37 

||**Query Sentence**<br>When did Barack Obama express intent to meet or negotiate with<br>Japan, and on that day, Catherine Ashton wasn’t consulted by the<br>person Edgars Rinkevics expressed intent to meet or negotiate<br>with on 2014-04-22?<br>**Temporal Query**<br>_q_[_T_?] =_T_?_, ∃Va, r_1(_e_1_, Va, t_1)_∧¬r_2(_Va, e_2_, T_?)_∧r_3(_e_3_, e_4_, T_?)<br>**Rank**<br>**Query Answers**<br>**Correctness**<br>**Answer Type**<br>1<br>2014-04-17<br>✔<br>Easy<br>2<br>2014-04-18<br>✔<br>Easy<br>3<br>2014-02-13<br>✔<br>Easy<br>4<br>2014-04-05<br>✔<br>Easy<br>5<br>2014-02-11<br>✘<br>-|
|---|---|
|**TimeAnd**<br>**?**<br>**t2i_NPt**<br>**TimeNot**<br>Edgars Rinkevics<br>Express intent to meet or negotiate<br>2014-04-22<br>Consult<br>Catherine Ashton<br>Japan<br>Barack Obama||
|||



Figure 28: Intermediate variable assignments and ranks for example t2i_NPt query. Correctness indicates whether the answer belongs to the ground-truth set of answers. 

||**Query Sentence**<br>When was the citizen (Greece) made an appeal or request by the<br>country that made an appeal or request to the Party President<br>(Greece), and on that day, Angela Merkel did not express intent<br>to meet or negotiate with Evo Morales?<br>**Temporal Query**<br>_q_[_T_?] =_T_?_, ∃Va, r_1(_e_1_, Va, t_1)_∧r_2(_Va, e_2_, T_?)_∧¬r_3(_e_3_, e_4_, T_?)<br>**Rank**<br>**Query Answers**<br>**Correctness**<br>**Answer Type**<br>1<br>2014-05-16<br>✔<br>Easy<br>2<br>2014-07-24<br>✔<br>Easy<br>3<br>2014-05-25<br>✔<br>Easy<br>4<br>2014-04-29<br>✔<br>Easy<br>5<br>2014-07-29<br>✔<br>Easy|
|---|---|
|**TimeAnd**<br>**?**<br>**t2i_PtN**<br>**TimeNot**<br>Party President (Greece)<br>2014-11-03<br>Make an appeal or request<br>Citizen (Greece)<br>Evo Morales<br>Angela Merkel||
|||



Figure 29: Intermediate variable assignments and ranks for example t2i_PtN query. Correctness indicates whether the answer belongs to the ground-truth set of answers. 

|||**Query Sentence**<br>Who discussed by telephone with Sergey Viktorovich Lavrov on<br>the day when Hun Sen was threatened by the person who was<br>arrested detained or chared with leal action b the Buddhist|
|---|---|---|
|**Pe_t2i_PtPe_NPt**|||
||**TimeAnd**<br>**?**|, ,  g  g  y<br>(Cambodia) on 2014-04-30, and on that day, Education (Colom-<br>bia) made a statement to Eric Garner?|
||**TimeNot**<br>Sergey Viktorovich Lavrov<br>Buddhist (Cambodia)<br>Arrest, detain, or charge with legal action<br>2014-04-30<br>Threaten<br>Hun Sen<br>Eric Garner<br>Education (Colombia)|**Temporal Query**<br>_q_[_V_?]<br>=<br>_V_?_, ∃Va, Ta, r_1(_e_1_, V_?_, Ta_) _∧r_2(_e_2_, Va, t_1) _∧_<br>_r_3(_Va, e_3_, Ta_)_∧¬r_4(_e_4_, e_5_, Ta_)|
|||<br>**Rank**<br>**Query Answers**<br>**Correctness**<br>**Answer Type**|
|||1<br>John Kerry<br>✔<br>Hard<br>2<br>Cabinet / Council of Ministers / Advisors (United States)<br>✔<br>Easy<br>3<br>Catherine Ashton<br>✘<br>-<br>4<br>Sergey Viktorovich Lavrov<br>✘<br>-<br>5<br>Joseph Robinette Biden<br>✘<br>-|
||||



Figure 30: Intermediate variable assignments and ranks for example Pe_t2i_N query. Correctness indicates whether the answer belongs to the ground-truth set of answers. 

||**Query Sentence**<br>Who visited Iran on 2014-04-29 and did not made a statement at<br>the place where Carlos Saul Menem made a statement on 2014-<br>09-20?<br>**Temporal Query**<br>_q_[_V_?]<br>=<br>_V_?_, ∃Va, Vb, r_1(_e_1_, Va, t_1) _∧¬r_2(_Va, Vb, t_2) _∧_<br>_r_3(_e_2_, V_?_, t_3)<br>**Rank**<br>**Query Answers**<br>**Correctness**<br>**Answer Type**<br>1<br>Santos Edelmar Lopez<br>✔<br>Easy<br>2<br>Simon Gass<br>✔<br>Easy<br>3<br>Sebastian Kurz<br>✔<br>Easy<br>4<br>Catherine Ashton<br>✘<br>-<br>5<br>Iran<br>✘<br>-|
|---|---|
|**And**<br>**?**<br>**e2i_NPe**<br>**Not**<br>Carlos Saul Menem<br>Make statement<br>2014-09-20<br>2014-09-20<br>Iran<br>Host a visit<br>2014-04-29||
|||



Figure 31: Intermediate variable assignments and ranks for example e2i_NPe query. Correctness indicates whether the answer belongs to the ground-truth set of answers. 

38 

||**Query Sentence**<br>Who was commented pessimistically on 2014-03-13 by the<br>person who expressed accord to Energy Department/Ministry<br>(Turkey) on 2014-03-27 and not visited by Mainland Affairs<br>Council on 2014-02-11?|
|---|---|
|**And**<br>**?**<br>**e2i_PeN**<br>**Not**<br>Energy Department/Ministry (Turkey)<br>2014-03-27<br>Make pessimistic comment<br>2014-03-13<br>Mainland Affairs Council<br>Make a visit<br>2014-02-11||
||**Temporal Query**<br>_q_[_V_?]<br>=<br>_V_?_, ∃Va, Vb, r_1(_e_1_, Va, t_1) _∧r_2(_Va, V_?_, Tb_)_∧_<br>_r_3(_e_2_, V_?_, t_3)|
||**Rank**<br>**Query Answers**<br>**Correctness**<br>**Answer Type**|
||1<br>Member of the Judiciary (Turkey)<br>✔<br>Easy<br>2<br>Lawyer/Attorney (Turkey)<br>✔<br>Hard<br>3<br>Justice and Development Party<br>✘<br>-<br>4<br>Other Authorities / Offcials (Turkey)<br>✘<br>-<br>5<br>Party Member (Turkey)<br>✘<br>-|
|||



Figure 32: Intermediate variable assignments and ranks for example e2i_PeN query. Correctness indicates whether the answer belongs to the ground-truth set of answers. 

|||**Query Sentence**<br>On 2014-10-21, who negotiated with the person who consulted<br>with the Treasury/Finance Ministry (Eritrea) on 2014-09-17 but<br>did not arrest, detain, or charge Drug Gang (Beltrán -Leyva Car-<br>|
|---|---|---|
|**And**<br>**?**<br>**And**<br>**And**<br>**Not**<br>**Pe_e2i_Pe_NPe**<br>Treasury/Finance Ministry (Eritrea)<br>2014-09-17<br>Drug Gang (Beltrán -Leyva Cartel)<br>2014-10-16<br>2014-10-21|||
||**And**<br>**?**<br>**And**<br>**And**<br>**Not**|tel) with legal action on 2014-10-16?<br> <br><br>    <br>|
|||**Temporal Query**<br>_q_[_V_?]<br>=<br>_V_?_, ∃Va, Vb, r_1(_e_1_, Va, t_1)_∧_<br>_r_2(_e_2_, Va, t_2) _∧_<br>_V V t_|
|||_r_3(_a, _?_,_3)|
|||**Rank**<br>**Query Answers**<br>**Correctness**<br>**Answer Type**|
|||1<br>South Korea<br>✔<br>Easy<br>2<br>Japan<br>✘<br>-<br>3<br>China<br>✔<br>Hard<br>4<br>North Korea<br>✘<br>-<br>5<br>Thailand<br>✘<br>-|
||||



Figure 33: Intermediate variable assignments and ranks for example Pe_e2i_N query. Correctness indicates whether the answer belongs to the ground-truth set of answers. 

|**And**<br>**?**<br>**e2i_N**<br>**And**<br>**Not**<br>Botswana<br>2014-03-11<br>Citizen (Brunei)<br>Make empathetic comment<br>2014-03-26|**And**<br>**?**<br>**e2i_N**<br>**And**<br>**Not**<br>Botswana<br>2014-03-11<br>Citizen (Brunei)<br>Make empathetic comment<br>2014-03-26|**Query Sentence**<br>Who rallied opposition against Botswana on 2014-03-11 but was<br>|
|---|---|---|
||**And**<br>**?**<br>**And**<br>**Not**|not empathetically commented upon by Citizen (Brunei) on 2014-<br>03-26?<br>|
|||**Temporal Query**<br>_q_[_V_?] =_V_?_, r_1(_e_1_, V_?_, t_1)_∧¬r_2(_e_2_, V_?_, t_2)|
|||**Rank**<br>**Query Answers**<br>**Correctness**<br>**Answer Type**|
|||1<br>Industry (Botswana)<br>✔<br>Easy<br>2<br>Citizen (Botswana)<br>✔<br>Hard<br>3<br>Richard Sezibera<br>✘<br>-<br>4<br>Labor and Employment Ministry (Botswana)<br>✘<br>-<br>5<br>Paul Kagame<br>✘<br>-|



Figure 34: Intermediate variable assignments and ranks for example e2i_N query. Correctness indicates whether the answer belongs to the ground-truth set of answers. 

|Envoy (Kazakhstan)<br>Praise or endorse<br>2014-05-19<br>RIA Novosti<br>Sign formal agreement<br>2014-12-12<br>Governor (South Korea)<br>2014-05-09<br>**And**<br>**?**<br>**e3i_N**<br>**And**<br>**Not**|**Query Sentence**<br>Who was praised or endorsed by Envoy (Kazakhstan) on 2014-<br>05-19 and signed a formal agreement with RIA Novosti on 2014-<br>12-12 but not signed a formal agreement with Governor (South<br>Korea) on 2014-05-09?|
|---|---|
||<br>**Temporal Query**<br>_q_[_V_?] =_V_?_, r_1(_e_1_, V_?_, t_1)_∧r_2(_e_2_, V_?_, t_2)_∧ r_3(_e_3_, V_?_, t_3)|
||<br>        <br>**Rank**<br>**Query Answers**<br>**Correctness**<br>**Answer Type**|
||1<br>Iran<br>✔<br>Easy<br>2<br>Iraq<br>✘<br>-<br>3<br>China<br>✘<br>-<br>4<br>Afghanistan<br>✘<br>-<br>5<br>North Atlantic Treaty Organization<br>✘<br>-|



Figure 35: Intermediate variable assignments and ranks for example e3i_N query. Correctness indicates whether the answer belongs to the ground-truth set of answers. 

39 

|**Or**<br>**?**<br>**e2u**<br>Foreign Affairs (Uruguay)<br>2014-11-24<br>Julian Paul Assange<br>2014-06-19|**Or**<br>**?**<br>**e2u**<br>Foreign Affairs (Uruguay)<br>2014-11-24<br>Julian Paul Assange<br>2014-06-19||
|---|---|---|
|||**Query Sentence**<br>WhoconsultedForeignAffairs(Uruguay)on2014-11-24or|
||**Or**<br>**?**|<br>       <br>granted asylum for Julian Paul Assange on 2014-06-19?|
|||**Temporal Query**<br>_q_[_V_?] =_V_?_, r_1(_e_1_, V_?_, t_1)_∨r_2(_e_2_, V_?_, t_2)|
|||**Rk**<br>**A**<br>**C**<br>**A T**|
|||**an**<br>**Query nswers**<br>**orrectness**<br>**nswer ype**|
|||1<br>Foreign Affairs (Ecuador)<br>✔<br>Easy<br>2<br>Mexico<br>✔<br>Easy<br>3<br>Government (Mexico)<br>✔<br>Hard<br>4<br>Guatemala<br>✘<br>-<br>5<br>Other Authorities / Offcials (Mexico)<br>✘<br>-|
||||



Figure 36: Intermediate variable assignments and ranks for example e2u query. Correctness indicates whether the answer belongs to the ground-truth set of answers. 

|**And**<br>**?**<br>**Pe_e2u**<br>**Or**<br>Lower House of Parliament (France)<br>Praise or endorse<br>2014-11-18<br>Newspaper (Afghanistan)<br>Make an appeal or request<br>2014-05-05<br>2014-05-24|**Query Sentence**<br>On 2014-05-24, who occupied the land praised or endorsed by the<br>Lower House of Parliament (France) on 2014-11-18 or requested<br>or appealed by a Newspaper (Afghanistan) on 2014-05-05?|
|---|---|
||**Temporal Query**<br>_q_[_V_?] =_V_?_, ∃Va,_(_r_1(_e_1_, Va, t_1)_∨r_2(_e_2_, Va, t_2))_∧r_2(_Va, V_?_, t_3)|
||**Rank**<br>**Query Answers**<br>**Correctness**<br>**Answer Type**|
||1<br>Armed Gang (Afghanistan)<br>✔<br>Easy<br>2<br>Armed Band (Afghanistan)<br>✔<br>-<br>3<br>Military Personnel - Special (Afghanistan)<br>✔<br>Hard<br>4<br>Militant (Taliban)<br>✘<br>-<br>5<br>Afghanistan<br>✘<br>-|



Figure 37: Intermediate variable assignments and ranks for example Pe_e2u query. Correctness indicates whether the answer belongs to the ground-truth set of answers. 

|**Pt_lPe**<br>Middle E<br>2014-09-<br>Make a v<br>Iraq|**Pt_lPe**<br>Middle E<br>2014-09-<br>Make a v<br>Iraq|**Pt_lPe**<br>Middle E<br>2014-09-<br>Make a v<br>Iraq|||
|---|---|---|---|---|
|||||**Query Sentence**<br>When was Iraq visited by the person who visited The Middle East<br>on 2014-09-15?|
||||**?**||
|||||**Temporal Query**<br>_q_[_T_?] =_T_?_, ∃Va, r_1(_e_1_, Va, t_1)_∧r_2(_Va, e_2_, T_?)|
|||Middle E<br>2014-09-<br>Make a v<br>Iraq|ast<br>15<br>isit|**Rank**<br>**Query Answers**<br>**Correctness**<br>**Answer Type**|
|||||1<br>2014-09-13<br>✔<br>Easy<br>2<br>2014-06-25<br>✔<br>Easy<br>3<br>2014-06-26<br>✔<br>Easy<br>4<br>2014-06-27<br>✔<br>Easy<br>5<br>2014-06-24<br>✔<br>Easy|
||||||



Figure 38: Intermediate variable assignments and ranks for example Pt_sPe query. Correctness indicates whether the answer belongs to the ground-truth set of answers. 

|**Pt_rPe**<br>**?**<br>France<br>Meet at a 'third' location<br>Peter Humphrey<br>2014-07-31|**Pt_rPe**<br>**?**<br>France<br>Meet at a 'third' location<br>Peter Humphrey<br>2014-07-31|**Pt_rPe**<br>**?**<br>France<br>Meet at a 'third' location<br>Peter Humphrey<br>2014-07-31|||
|---|---|---|---|---|
|||||**Query Sentence**<br>At what time did France meet the person who made a statement|
|||**?**||to Peter Humphrey at a ’third’ location?|
|||||**Temporal Query**<br>_q_[_T_?] =_T_?_, ∃Va, r_1(_e_1_, Va, T_?)_∧r_2(_e_2_, Va, t_1))|
|||France<br>Meet at a 'third' location<br>Peter Humphrey<br>2014-07-31||<br>|
|||||**Rank**<br>**Query Answers**<br>**Correctness**<br>**Answer Type**|
|||||1<br>2014-11-21<br>✔<br>Hard<br>2<br>2014-11-25<br>✔<br>Easy<br>3<br>2014-11-20<br>✔<br>Easy<br>4<br>2014-09-23<br>✔<br>Easy<br>5<br>2014-09-21<br>✔<br>Easy|
||||||



Figure 39: Intermediate variable assignments and ranks for example Pt_oPe query. Correctness indicates whether the answer belongs to the ground-truth set of answers. 

40 

||**Query Sentence**<br>When did Abu Bakr Abdullah al-Qirbi discuss with Mohammad<br>Javad Zarif by telephone, and on that day Senate (Slovakia) did<br>not reject Justice Department/Ministery (Slovakia)?|
|---|---|
|**TimeAnd**<br>**?**<br>**TimeNot**<br>**t2i_N**<br>Abu Bakr Abdullah al-Qirbi<br>Discuss by telephone<br>Mohammad Javad Zarif<br>Senate (Slovakia)<br>Reject<br>Department/Ministery (Slovakia)||
||**Temporal Query**<br>_q_[_T_?] =_T_?_, r_1(_e_1_, e_2_, T_?)_∧¬r_2(_e_3_, e_4_, T_?)|
||**Rank**<br>**Query Answers**<br>**Correctness**<br>**Answer Type**|
||1<br>2014-01-19<br>✔<br>Easy<br>2<br>2014-01-18<br>✔<br>Hard<br>3<br>2014-11-02<br>✘<br>-<br>4<br>2014-02-02<br>✘<br>-<br>5<br>2014-12-13<br>✘<br>-|



Figure 40: Intermediate variable assignments and ranks for example t2i_N query. Correctness indicates whether the answer belongs to the ground-truth set of answers. 

|||**Query Sentence**<br>When did the Police (Egypt) use tactics of violent repression|
|---|---|---|
|**TimeAnd**<br>**?**<br>**TimeNot**<br>**t3i_N**<br>Police (Egypt)<br>Use tactics of violent repression<br>Protester (Egypt)<br>Opposition Activist (Ukraine)<br>Police (Ukraine)<br>Information / Communication / Transparency Ministry<br>Consult<br>Warren Truss<br>(Chin|**TimeAnd**<br>**?**<br>**TimeNot**|<br><br>against protesters (Egypt) and the Police (Ukraine) use tactics<br>of violent repression against protesters (Egypt), and on that day<br>Information / Communication / Transparency Ministry (China)<br>did not consult Warren Truss?|
|||<br>|
|||**Temporal Query**<br>_q_[_T_?] =_T_?_, r_1(_e_1_, e_2_, T_?)_∧r_2(_e_3_, e_4_, T_?)_∧¬r_3(_e_5_, e_6_, T_?)|
|||**Rank**<br>**Query Answers**<br>**Correctness**<br>**Answer Type**|
|||a)<br>1<br>2014-01-21<br>✔<br>Easy<br>2<br>2014-01-13<br>✘<br>-<br>3<br>2014-01-22<br>✘<br>-<br>4<br>2014-02-20<br>✘<br>-<br>5<br>2014-01-14<br>✘<br>Hard|
||||



Figure 41: Intermediate variable assignments and ranks for example t3i_N query. Correctness indicates whether the answer belongs to the ground-truth set of answers. 

|**TimeOr**<br>**?**<br>**t2u**<br>Nonofo Molefhi<br>Citizen (Botswana)<br>Zheng Zeguang<br>Make statement<br>China||
|---|---|
||**Query Sentence**<br>When did citizens (Botswana) obstruct passage and block Nonofo<br>Molefhi or Zheng Zeguang make statements about China?|
||**Temporal Query**<br>_q_[_T_?] =_T_?_, r_1(_e_1_, e_2_, T_?)_∨r_2(_e_3_, e_4_, T_?)|
||**Rank**<br>**Query Answers**<br>**Correctness**<br>**Answer Type**|
||1<br>2014-04-09<br>✔<br>Easy<br>2<br>2014-05-21<br>✔<br>Easy<br>3<br>2014-05-20<br>✔<br>Hard<br>4<br>2014-05-22<br>✘<br>-<br>5<br>2014-05-22<br>✘<br>-|
|||



Figure 42: Intermediate variable assignments and ranks for example t2u query. Correctness indicates whether the answer belongs to the ground-truth set of answers. 

||**Query Sentence**<br>Who was appealed or requested by Citizen (Nigeria) on the day<br>when the University of Cape Town consulted News Editor (Nige-<br>ria) or State Security Court (United Arab Emirates) arrested,<br>detained, or charged Muslim (United Arab Emirates) with legal<br>action?|
|---|---|
|**?**<br>**TimeOr**<br>**Pe_t2u**<br>Citizen (Nigeria)<br>Make an appeal or request<br>University of Cape Town<br>Consult<br>News Editor (Nigeria)<br>Muslim (United Arab Emirates)<br>State Security Court (United Arab Emirates)||
||**Temporal Query**<br>_q_[_V_?] =_V_?_, ∃Ta, r_1(_e_1_, V_?_, Ta_)_∧_(_r_2(_e_2_, e_3_, Ta_)_∨r_3(_e_4_, e_5_, Ta_))|
||**Rank**<br>**Query Answers**<br>**Correctness**<br>**Answer Type**|
||1<br>Government (Nigeria)<br>✔<br>Easy<br>2<br>Assemblies (Nigeria)<br>✔<br>Hard<br>3<br>Other Authorities / Offcials (Nigeria)<br>✘<br>-<br>4<br>Head of Government (Nigeria)<br>✘<br>-<br>5<br>Citizen (Nigeria)<br>✘<br>-|
|||



Figure 43: Intermediate variable assignments and ranks for example Pe_t2u query. Correctness indicates whether the answer belongs to the ground-truth set of answers. 

41 

|**Pe_aPt**<br>**After**<br>**?**<br>Citizen (India)<br>Stephane Le Foll<br>Representatives (France)|**Pe_aPt**<br>**After**<br>**?**<br>Citizen (India)<br>Stephane Le Foll<br>Representatives (France)||
|---|---|---|
|||**Query Sentence**<br>Who made a statement for citizens (India) before Representatives<br>|
||**After**<br>**?**|(France) consulted Stephane Le Foll?|
|||<br>**Temporal Query**<br>_q_[_V_?]<br>=<br>_V_?_, ∃Ta, Tb, r_1(_e_1_, V_?_, Tb_) _∧After_(_Ta, Tb_) _∧_<br>_r_(_e e T_)|
|||22_,_3_,a_|
|||**Rank**<br>**Query Answers**<br>**Correctness**<br>**Answer Type**<br><br> <br><br>|
|||1<br>Court Judge (India)<br>✔<br>Easy<br>2<br>Governor (India)<br>✔<br>Easy<br>3<br>Manohar Lal Khattar<br>✔<br>Easy<br>4<br>Vasundhara Raje<br>✘<br>Easy<br>5<br>Head of Government (India)<br>✘<br>-|
||||



Figure 44: Intermediate variable assignments and ranks for example Pe_aPt query. Correctness indicates whether the answer belongs to the ground-truth set of answers. 

||**Query Sentence**<br>Who expressed intent to meet or negotiate with the Head of Gov-<br>ernment (Bangladesh) after South Korea Criticized or denounced<br>Sadako Ogata?|
|---|---|
|**Pe_bPt**<br>**Before**<br>**?**<br>South Korea<br>Criticize or denounce<br>Sadako Ogata<br>Head of Government(Bangladesh)||
||**Temporal Query**<br>_q_[_V_?]<br>=<br>_V_?_, ∃Ta, Tb, r_1(_e_1_, V_?_, Tb_) _∧Before_(_Ta, Tb_) _∧_<br>_r_2(_e_2_, e_3_, Ta_)|
||**Rank**<br>**Query Answers**<br>**Correctness**<br>**Answer Type**|
||1<br>Japan<br>✔<br>Easy<br>2<br>North Korea<br>✔<br>Easy<br>3<br>China<br>✔<br>Easy<br>4<br>South Korea<br>✘<br>-<br>5<br>Kim Jong-Un<br>✘<br>-|
|||



Figure 45: Intermediate variable assignments and ranks for example Pe_bPt query. Correctness indicates whether the answer belongs to the ground-truth set of answers. 

|||**Query Sentence**<br>Who was criticized or denounced by North Korea on the day<br>before Admiral (Russia) made a statement for Chuck Hagel and|
|---|---|---|
|**TimeAnd**<br>**?**<br>**After**<br>**TimeAnd**<br>**Pe_at2i**<br>North Korea<br>Criticize or denounce<br>Admiral (Russia)<br>Make statement<br>Chuck Hagel<br>Ministry (Madagascar)<br>Make statement<br>Government (Madagascar)|||
||**TimeAnd**<br>**?**<br>**After**<br>**TimeAnd**<br>**Pe_at2i**|Ministry (Madagascar) made a statement for Government (Mada-<br>gascar)?|
|||**Temporal Query**<br>_q_[_V_?]<br>=<br>_V_?_, ∃Ta, Tb, r_1(_e_1_, V_?_, Tb_)) _∧After_(_Ta, Tb_) _∧_|
|||<br><br>     <br>_r_2(_e_2_, e_3_, Ta_)_∧r_3(_e_4_, e_5_, Ta_)|
|||**Rank**<br>**Query Answers**<br>**Correctness**<br>**Answer Type**|
|||1<br>South Korea<br>✔<br>Easy<br>2<br>Japan<br>✔<br>Easy<br>3<br>North Korea<br>✔<br>-<br>4<br>Head of Government (South Korea)<br>✔<br>Easy<br>5<br>John Kerry<br>✔<br>Easy|
||||



Figure 46: Intermediate variable assignments and ranks for example Pe_at2i query. Correctness indicates whether the answer belongs to the ground-truth set of answers. 

|||**Query Sentence**<br>Who was declined commented by Citizen (India) after the day<br>when the employee (Mozambique) consulted the Presidential<br>Cdidt Mbi d E Ml d  ttt t|
|---|---|---|
|**TimeAnd**<br>**?**<br>**Before**<br>**TimeAnd**<br>**Pe_bt2i**<br>Citizen (India)<br>Decline comment<br>Presidential Candidate (Mozambique)<br>Employee (Mozambique)<br>Evo Morales<br>Make statement<br>Military (Bolivia)|||
||**TimeAnd**<br>**?**<br>**Before**<br>**TimeAnd**<br>**Pe_bt2i**|anae (ozamque), an vo oraes mae a saemen o<br>the Military (Bolivia)?<br><br>|
|||**Temporal Query**<br>_q_[_V_?]<br>=<br>_V_?_, ∃Ta, Tb, r_1(_e_1_, V_?_, Tb_) _∧Before_(_Ta, Tb_) _∧_|
|||_r_2(_e_2_, e_3_, Ta_)_∧r_3(_e_4_, e_5_, Ta_)|
|||**Rank**<br>**Query Answers**<br>**Correctness**<br>**Answer Type**|
|||1<br>Media (India)<br>✔<br>Easy<br>2<br>Member of Parliament (India)<br>✘<br>-<br>3<br>Court Judge (India)<br>✔<br>Easy<br>4<br>Head of Government (India)<br>✘<br>-<br>5<br>Governor (India)<br>✘<br>-|
||||



Figure 47: Intermediate variable assignments and ranks for example Pe_bt2i query. Correctness indicates whether the answer belongs to the ground-truth set of answers. 

42 

|||**Query Sentence**<br>What was the day between the day that Tsai Ing-wen accused the<br>Party Member (Taiwan) and the foreign affairs (United States)<br>appealed for diplomatic cooperation (such as policy support) from<br>the other Authorities/offcials (Romania)?<br>**Temporal Query**<br>_q_[_T_?]<br>=<br>_T_?_, ∃Ta, Tb, r_1(_e_1_, e_2_, Ta_)<br>_∧_<br>_r_2(_e_3_, e_4_, Tb_)<br>_∧_<br>_After_(_Ta, T_?)_∧Before_(_Tb, T_?)<br><br> <br><br>|
|---|---|---|
|**TimeAnd**<br>**?**<br>**After**<br>**Before**<br>**between**<br>Other Authorities / Officials (Romania)<br>Foreign Affairs (United States)<br>Tsai Ing-wen<br>Accuse<br>Party Member (Taiwan)|||
||Other Authorities / Officials (Romania)<br>Foreign Affairs (United States)|**Rank**<br>**Query Answers**<br>**Correctness**<br>**Answer Type**|
||||
|||1<br>2014-07-19<br>✔<br>Easy<br>2<br>2014-07-29<br>✔<br>-<br>3<br>2014-06-10<br>✔<br>Easy<br>4<br>2014-08-21<br>✔<br>-<br>d 5<br>2014-06-25<br>✔<br>Easy|
||||



Figure 48: Intermediate variable assignments and ranks for example between query. Correctness indicates whether the answer belongs to the ground-truth set of answers. 

43 

