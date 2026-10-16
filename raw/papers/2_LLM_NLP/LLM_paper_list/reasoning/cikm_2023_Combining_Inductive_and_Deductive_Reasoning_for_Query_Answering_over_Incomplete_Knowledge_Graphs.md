# **Combining Inductive and Deductive Reasoning for Query Answering over Incomplete Knowledge Graphs** 

Medina Andresel[∗] 

Csaba Domokos Bosch Center for Artificial Intelligence Renningen, Germany csaba.domokos@de.bosch.com 

Trung-Kien Tran 

Bosch Center for Artificial Intelligence Renningen, Germany trungkien.tran@de.bosch.com 

AIT Austrian Institute of Technology Vienna, Austria medina.andresel@ait.ac.at 

Pasquale Minervini 

Daria Stepanova Bosch Center for Artificial Intelligence Renningen, Germany daria.stepanova@de.bosch.com 

University of Edinburgh Edinburgh, United Kingdom p.minervini@ed.ac.uk 

## **ABSTRACT** 

_𝑞_ ( _𝑋_ ) ← _degreeFrom_ ( _𝑋, mit_ ) ∧ _worksFor_ ( _𝑋, amazon_ ). Answering such queries is very challenging when KGs are incomplete, which is often the case due to their (semi-) automatic construction, and obtaining complete answers typically requires further domain knowledge, i.e., the application of deductive reasoning. For instance, _mary_ is a missing but desired answer of _𝑞_ that can be obtained by combining the fact _managerAt_ ( _mary, amazon_ ), predicted using machine learning models, and the axiom stating that _managerAt_ implies _worksFor_ in the ontology O of Figure 1. Therefore, such a task requires both inductive and deductive reasoning. 

Current methods for embedding-based query answering over incomplete Knowledge Graphs (KGs) only focus on inductive reasoning, i.e., predicting answers by learning patterns from the data, and lack the complementary ability to do deductive reasoning, which requires the application of domain knowledge to infer further information. To address this shortcoming, we investigate the problem of incorporating ontologies into embedding-based query answering models by defining the task of embedding-based ontology-mediated query answering. We propose various integration strategies into prominent representatives of embedding models that involve (1) different ontology-driven data augmentation techniques and (2) adaptation of the loss function to enforce the ontology axioms. We design novel benchmarks for the considered task based on the LUBM and the NELL KGs and evaluate our methods on them. The achieved improvements in the setting that requires both inductive and deductive reasoning are from 20% to 55% in HITS@3. 

Recently, _Knowledge Graph Embedding_ (KGE) techniques [33] that can be used to predict missing facts have been proposed to answer logical queries over incomplete KGs [20, 30, 36, 37, 39]. At the same time, in the Knowledge Representation and Reasoning area answering queries over incomplete data has also received a lot of attention and one of the most successful approaches for this task is to exploit ontologies when querying KGs, referred to as _Ontology-Mediated Query Answering_ [OMQA, 38]. 

While promising, existing embedding-based methods do not take ontologies, which formalize domain knowledge, into account. Since large portions of expert knowledge can be conveniently encoded using ontologies, the benefits of coupling ontology reasoning and embedding methods for KG completion are evident and have been acknowledged in several works, e.g. [18, 28]. However, to the best of our knowledge, coupling inductive and deductive reasoning to answer queries over incomplete KGs has not been considered yet. 

## **CCS CONCEPTS** 

• **Computing methodologies** → _Description logics_ ; _Machine learning approaches_ ; • **Computer systems organization** → _Neural networks_ . 

## **KEYWORDS** 

Knowledge Graphs, Ontologies, Embeddings, Query Answering, Neuro-Symbolic AI 

Answering queries over the KG augmented with triples resulting from the naive process of interchangeably using embedding methods and ontology reasoning, comes with a big scalability challenge [27] and commonly known error accumulation issues. In practice, we need to restrict ourselves to computing merely small subsets of likely fact predictions required for answering a given query; thus more sophisticated proposals are needed. Hence, we investigate three open questions: (1) How to adapt existing OMQA techniques to the setting of KGEs? (2) How do different data augmentation strategies impact the accuracy of existing embedding models for the OMQA task? (3) Does enforcing ontology axioms in the embedding space via loss function help to improve inductive and deductive reasoning performance? 

## **1 INTRODUCTION** 

Knowledge Graphs (KGs) have recently received much attention due to their relevance in various applications, such as natural question answering or web search. Prominent KGs include NELL [8], YAGO [31], and Wikidata [12]. A KG describes facts about entities by interconnecting them via relations, e.g., _hasAlumnus_ ( _mit, bob_ ) in Figure 1 states that Bob is an MIT alumnus. 

A crucial task in leveraging information from knowledge graphs is that of answering logical queries such as _Who works for Amazon and has a degree from MIT?_ , which can be formally written as 

∗The work has been done during the PhD sabbatical of Medina Andresel at Bosch Center for Artificial Intelligence 

Medina Andresel, Trung-Kien Tran, Csaba Domokos, Pasquale Minervini, & Daria Stepanova 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/cikm_2023_Combining_Inductive_and_Deductive_Reasoning_for_Query_Answering_over_Incomplete_Knowledge_Graphs_images/cikm_2023_Combining_Inductive_and_Deductive_Reasoning_for_Query_Answering_over_Incomplete_Knowledge_Graphs.pdf-0002-01.png)


**Figure 1: An exemplary ontology** O **and a KG** G **. Solid edges illustrate existing facts in** G **, and dashed ones indicate missing facts that could be predicted using KG embeddings.** 

We answer (1)-(3) by the following contributions: 

- We formally define the novel task of _Embedding-Based OntologyMediated Query Answering_ (E-OMQA), analyze and systematically compare various extensions of embedding-based query answering models to incorporate ontologies. 

- We propose ontology-driven strategies for sampling queries to train embedding models for query answering, as well as a loss function modification to enforce the ontology axioms within the embedding space, and demonstrate the effectiveness of these proposals on widely-used representatives of query-based and atom-based models. 

- As no previous benchmarks exist for E-OMQA, we design new ones using LUBM and NELL, i.e., well-known benchmarks for OMQA and embedding models, respectively. 

- Extensive evaluation shows that enforcing the ontology via the loss function, in general, improves the deductive power regardless of how the training data is sampled, while ontology-driven sampling strategy has a further significant positive impact on performance. We obtain overall improvements, ranging from 20% to 55% in HITS@3, in the settings that require both inductive and deductive reasoning. 

## **2 PRELIMINARIES** 

_Knowledge Graphs and Ontologies._ We assume a signature consisting of countable pairwise disjoint sets E _,_ C, and R of entities (constants), concepts (types), and roles (binary relations), respectively. A knowledge graph G ( _a.k.a._ ABox) is a set of triples, such as ( _mit, type, University_ ) and ( _bob, worksFor, mit_ ), where _mit, bob_ ∈ E, _worksFor, type_ ∈ R, and _University_ ∈ C. These triples can also be represented as _University_ ( _mit_ )[1] and _worksFor_ ( _bob, mit_ ). An ontology O ( _a.k.a._ TBox) is a set of axioms in Description Logics [4] over the signature Σ = ⟨E _,_ C _,_ R⟩. We focus on ontologies in _DL_ - _Lite_ R DL fragment [7] that have the following syntax: 

_𝐴_ ⊑ _𝐴_[′] _𝐴_ ⊑∃ _𝑝_ ∃ _𝑝_ ⊑ _𝐴_ ∃ _𝑝_[−] ⊑ _𝐴 𝑝_ ⊑ _𝑠 𝑝_[−] ⊑ _𝑠,_ where _𝐴, 𝐴_[′] ∈ C are concepts and _𝑝,𝑠_ ∈ R are roles and _𝑝_[−] denotes the inverse relation of _𝑝_ . The KG and its ontology from Figure 1 are in _DL_ - _Lite_ R . The _DL_ - _Lite_ R syntax and semantics are summarized in Table 1. Given a KG G and an ontology O, an interpretation I _is a model of_ G _w.r.t_ O if I satisfies each fact in G and each axiom in O. 

> 1Unary facts can also be modeled using the binary _type_ relation. 

**Table 1: DL syntax and semantics defined using FO interpretations** (Δ[I] _,_ ·[I] ) **with a non-empty domain** Δ[I] **and an interpretation function** ·[I] **.** _𝐶_ **and** _𝐷_ **denote concepts in** _DL_ **-** _Lite_ R **.** 

|**DL Syntax**|**Semantics**|
|---|---|
|_𝑒_|_𝑒_I ∈ΔI|
|_𝐴_(resp._𝑝_)|_𝐴_I ⊆ΔI (resp._𝑝_I ⊆ΔI×ΔI)|
|∃_𝑝_|(∃_𝑝_)I = {_𝑑_∈ΔI | ∃_𝑑_′_,_(_𝑑,𝑑_′) ∈_𝑝_I }|
|_𝑝_−|(_𝑝_−)I = {(_𝑑_′_,𝑑_) | (_𝑑,𝑑_′) ∈_𝑝_I }.|
|_𝐶_⊑_𝐷_(resp._𝑝_⊑_𝑠_)|_𝐶_I ⊆_𝐷_I (resp._𝑝_I ⊆_𝑠_I)|
|_𝐴_(_𝑐_) (resp._𝑝_(_𝑐,𝑐_′))|_𝑐_∈_𝐴_I (resp. (_𝑐,𝑐_′) ∈_𝑝_I)|



For _DL_ - _Lite_ R , a _canonical model_ exists that can be homomorphically mapped into any other model, obtained from the _deductive closure_ O[∞] (G), which extends G with triples derived from existing triples in G by exhaustively applying axioms in O [7]. 

_Ontology-Mediated Query Answering._ A _query atom_ is an expression of the form _𝑝_ ( _𝑇_ 1 _,𝑇_ 2), where _𝑝_ ∈ R, and each _𝑇𝑖_ ∈ V ∪ E is called a _term_ , with V disjoint with E _,_ C _,_ and R being a set of variables. A _monadic conjunctive query_ (CQ) _𝑞_ ( _𝑋_ ) is a First-Order (FO) formula of the form _𝑞_ ( _𝑋_ ) ←∃ _𝑌.𝑝_[�] 1 ( _𝑇_[�] 1) ∧· · · ∧ _𝑝𝑛_ ( _𝑇_[�] _𝑛_ ) where each _𝑝𝑖_ ( _𝑇_[�] _𝑖_ ) is a query atom, and _vars_ ( _𝑞_ ) = { _𝑋_ } ∪ _𝑌_[�] denotes the set of variables appearing in _𝑞_ , with _𝑋_ ∉ _𝑌_[�] being the _answer variable_ . In this work, we focus on _monadic Existential Positive FO_ (EPFO) queries, i.e., unions of monadic CQs [10]. For a query _𝑞_ ( _𝑋_ ) and a KG G, a constant _𝑎_ is an answer of _𝑞_ if a mapping _𝜋_ : _var_ ( _𝑞_ ) ↦→ E exists, s.t. _𝑞𝜋_ ∈G and _𝜋_ ( _𝑋_ ) = _𝑎_ ; _q_ [G] are the answers of _𝑞_ on G. 

_Ontology-Mediated Query Answering (OMQA)_ concerns answering queries by accounting for both the KG and the accompanying ontology. Since the model constructed from O[∞] (G) can be homomorphically mapped to every other model, the deductive closure can be used to evaluate queries [7]. 

_Definition 2.1._ Given a KG G and an ontology O, an entity _𝑎_ from G is a _certain answer_ of _𝑞_ ( _𝑋_ ) over (G _,_ O) if _𝑎_ is an answer to _q_ ( _X_ ) over O[∞] (G). We use _𝑞_ [G _,_ O] to denote the set of _certain answers of 𝑞 over_ (G _,_ O). 

Let _𝑞_ and _𝑞_[′] be two monadic queries over (G _,_ O), then _𝑞_ is _contained_ in _𝑞_[′] w.r.t. O if _𝑞_ [G _,_ O] ⊆ _𝑞_[′] [G _,_ O]; we call _𝑞_ a _specialization_ of _𝑞_[′] (written as _𝑞_[′] �s _𝑞_ ), and _𝑞_ ′ a _generalization_ of _𝑞_ (written as _𝑞_ �g _𝑞_ ′). Query generalizations and specializations can be obtained by exploiting ontology axioms; such process (and result) is referred to as _query rewriting_ . 

_Example 2.2._ Consider the KG G in Figure 1 and the query _q_ ( _X_ ) ← type( _𝑋,_ Professor) ∧ degreeFrom( _𝑋,_ mit). Since mat ∈ _𝑞_ [G], it is a certain answer. Moreover, according to O,AProfessor is a sub-type of Professor and degreeFrom is inverse of hasAlumnus, thus bob is also a certain answer. For _𝑞_[′] ( _𝑋_ )←type( _𝑋,_ AProfessor)∧ degreeFrom( _𝑋,_ mit) it holds that _𝑞_ �s _𝑞_ ′ as mat ∉ _𝑞_ ′ [G _,_ O]. 

_Embedding-Based Approximate Query Answering._ Since, in reality, KGs might be missing facts, existing query answering techniques designed for complete data might not compute all answers. In such settings, one assumes that the given KG G is a subset of a complete 

Combining Inductive and Deductive Reasoning for Query Answering over Incomplete Knowledge Graphs 

but unobservable KG G _[𝑖]_ , and one aims at estimating the likely answers to _𝑞_ over G _[𝑖]_ . E.g., G _[𝑖]_ for the graph G given in Figure 1 includes the links denoted by dashed edges. In practice, to evaluate the accuracy of a considered method, G _[𝑖]_ is typically fixed at the beginning, and G is created by removing facts from G _[𝑖]_ . 

The set of all answers to a given query _𝑞_ comprises those that can be obtained by directly querying the given KG G only and those that require predicting missing KG facts. Thus, one typically distinguishes easy and hard answers as follows: 

_Definition 2.3._ Given a KG G, a subgraph of a complete but unobservable KG G _[𝑖]_ and a query _𝑞_ ( _𝑋_ ), _𝑎_ is an _easy answer_ to _𝑞_ if _𝑎_ ∈ _𝑞_ [G], and it is a _hard answer_ to _𝑞_ if _𝑎_ ∈ _𝑞_ [G _[𝑖]_ ]\ _𝑞_ [G]. 

Recently, embedding-based methods have been proposed for _approximate answering of existential positive FO queries_ over incomplete KGs [9, 25, 30, 36, 37]. Broadly, such methods can be divided into two categories: _query-based_ [9, 25, 30, 36, 37] and _atombased_ [2]. Generally, any neural QA model relies on an embedding function which maps entities and relations into a _𝑑_ -dimensional embedding space. It then computes a score of each entity _𝑐_ for being an answer to a given query _𝑞_ over G _[𝑖]_ via a scoring function _𝜙𝑞_ (c) : R _[𝑑]_ ↦→[0 _,_ 1], where c denotes the embedding vector of _𝑐_ .[2] Using these scoring functions, the final _embedding QA function_ E G takes as input a query and returns its approximate answers over the knowledge graph G _[𝑖]_ , i.e., answers that have the scoring above some predefined threshold. We say that E G is _reliable w.r.t._ G _[𝑖]_ whenever for each query _𝑞_ , _𝑐_ is an approximate answer to _𝑞_ iff _𝑐_ is an answer to _𝑞_ over G _[𝑖]_ . Clearly, the challenge of identifying hard answers is still valid also for embedding QA models. 

## **3 EMBEDDING-BASED OMQA** 

Existing methods for embedding-based query answering compute approximate answers to queries over an unobservable KG G _[𝑖]_ by performing inductive reasoning. However, they are not capable of simultaneously applying deductive reasoning, and thus cannot account for ontologies with which KGs are often accompanied. 

To address this shortcoming, we propose ways to combine inductive and deductive reasoning for approximate query answering over incomplete KGs. For that, we first formalize the task of _Embeddingbased Ontology-Mediated Query Answering (E-OMQA)_ in which both types of reasoning are exploited. The goal of this task is to approximate certain answers to OMQs over G _[𝑖]_ . 

_Definition 3.1 (E-OMQA)._ Let G be a KG, which is a subgraph of a complete but not observable KG G _[𝑖]_ , let O be an ontology and _𝑞_ a query. _Embedding-based ontology-mediated query answering_ is concerned with constructing an embedding function E G _,_ O that is reliable w.r.t. O[∞] (G _[𝑖]_ ). 

Note that, _𝑞_ [G _[𝑖] ,_ O] subsumes both _𝑞_ [G _[𝑖]_ ], the answers requiring inductive reasoning, and _𝑞_ [G _,_ O], the answers computed via deductive reasoning only. Analogously as for embedding-based query answering, for E-OMQA, we distinguish between easy certain answers and hard certain answers as follows. 

_Definition 3.2._ Given a KG G, a subgraph of a complete but unobservable KG G _[𝑖]_ , an ontology O and a query _𝑞_ ( _𝑋_ ), _𝑎_ is an _easy_ 2Bold small letters denote vector representations. 

_certain answer_ to _𝑞_ if _𝑎_ ∈ _𝑞_ [G _,_ O], and it is a _hard certain answer_ to _𝑞_ if _𝑎_ ∈ _𝑞_ [G _[𝑖] ,_ O]\ _𝑞_ [G _,_ O]. 

Next, we discuss several embedding-based methods for ontologymediated query answering under incompleteness. 

_Query Rewriting over Pre-trained Models._ In the traditional OMQA setting, each query _𝑞_ can be evaluated by first rewriting _𝑞_ into a set of FO-queries _𝑄_ O and then evaluating each query in _𝑄_ O over G alone. For E-OMQA, this amounts to constructing an embedding QA function E G for G alone and using it to compute the answers to all queries in _𝑄_ O. For _DL_ - _Lite_ R , such FO-rewriting is obtained by extensively applying ontology axioms in a specializing fashion, which results in the so-called _perfect reformulation_ [7]. 

_Ontology-Aware Models._ An alternative to query rewriting is to develop an embedding query answering function that accounts for axioms in O. To the best of our knowledge, there are no KGE models that directly address the problem of E-OMQA. Thus, we suggest the following: (1) Train existing embedding models for QA on the data derived from O[∞] (G) instead of G; (2) Develop an _ontology-aware_ embedding model that will be trained on G but will have special terms in the training objective structurally enforcing O. (3) Combine (1) and (2), i.e., train ontology-aware embedding models on the data derived from O[∞] (G). 

While the proposed approaches can be realized on top of any embedding model for logical query answering, in this work, we verify their effectiveness on a query-based model _Query2Box_ [36] and an atom-based model _CQD_ [2]. In Section 3.1, we present several effective ontology-driven training methods for realizing (1). As for (2), building on _Query2Box_ , in Section 3.2 we develop its ontologyaware version. Moreover, we build an ontology-aware extension of _CQD_ , on top of the neural link predictor using _adversarial sets regularization_ (ASR) [32] to enforce the ontology axioms. We chose this approach, since it is general and allows us to incorporate rules into any off-the-shelf neural link predictor. In our experiments, we use ComplEx-N3 [29] as it requires minimal modification to CQD and outperforms other neural link predictors (see [29]). Finally, we evaluate the effectiveness of our ontology-driven strategies from Section 3.1 on the extended models described in Section 3.2 and verify the feasibility of the classical query-rewriting approach in the knowledge graph embedding setting. 

## **3.1 Ontology-Driven Data Sampling** 

Let Q G be the set of all possible queries that can be formed using the signature Σ. During the training process, existing embedding models are trained on a set of sampled queries of certain shapes and their answers over the KG G. 

_3.1.1 Random Query Sampling._ The existing sampling procedure from the literature [19, 36] arbitrarily chooses entities and relations in the graph to construct queries of various shapes. Query2Box is trained on complex queries involving multiple atoms, while CQD is trained only on atomic queries, as it relies on a neural link predictor. For verifying how well the model generalizes, the test set includes queries whose shapes have not been encountered during training. 

Naturally, this procedure is not guaranteed to capture ontological knowledge that comes with the knowledge graph, and using it over O[∞] (G) could generate a bias towards concepts and roles that 

Medina Andresel, Trung-Kien Tran, Csaba Domokos, Pasquale Minervini, & Daria Stepanova 

**Table 2: Rules to specialize and generalize an atom** _𝛽_ **from** _𝑞_ ( _𝑋_ ) ← _𝛼_ ∧ _𝛽_ **, where** _𝐴, 𝐵_ ∈ C **,** _𝑝,𝑟,𝑠_ ∈ R **and** _𝑇,𝑇_ 1 _,𝑇_ 2 ∈ _vars_ ( _𝑞_ ) ∪ E **. The** s g **operators** � **and** � **are used for constructing specializations and generalizations respectively of a given query.** 

s g **(R1)** If _𝐴_ ⊑ _𝐵_ ∈O then: _𝛼_ ∧ type( _𝑇, 𝐵_ ) � _𝛼_ ∧ type( _𝑇,𝐴_ ) _𝛼_ ∧ type( _𝑇,𝐴_ ) � _𝛼_ ∧ type( _𝑇, 𝐵_ ) s g **(R2)** If ∃ _𝑝_ ⊑ _𝐴_ ∈O then: _𝛼_ ∧ type( _𝑇_ 1 _,𝐴_ ) � _𝛼_ ∧ _𝑝_ ( _𝑇_ 1 _,𝑇_ 2 ) _𝛼_ ∧ _𝑝_ ( _𝑇_ 1 _,𝑇_ 2 ) � _𝛼_ ∧ type( _𝑇_ 1 _,𝐴_ ) s g **(R3)** If _𝐴_ ⊑∃ _𝑝_ ∈O then: _𝛼_ ∧ _𝑝_ ( _𝑇_ 1 _,𝑇_ 2 ) � _𝛼_ ∧ type( _𝑇,𝐴_ ) _𝛼_ ∧ type( _𝑇_ 1 _,𝐴_ ) � _𝛼_ ∧ _𝑝_ ( _𝑇_ 1 _,𝑇_ 2 ) s g **(R4)** If ∃ _𝑝_[−] ⊑ _𝐴_ ∈O then: _𝛼_ ∧ _𝐴_ ( _𝑇_ ) � _𝛼_ ∧ _𝑝_ ( _𝑇_ 2 _,𝑇_ 1 ) _𝛼_ ∧ _𝑝_ ( _𝑇_ 2 _,𝑇_ 1 ) � _𝛼_ ∧ type( _𝑇_ 1 _,𝐴_ ) s g **(R5)** If _𝐴_ ⊑∃ _𝑝_ ∈O then: _𝛼_ ∧ _𝑝_ ( _𝑇_ 1 _,𝑇_ 2 ) � _𝛼_ ∧ type( _𝑇,𝐴_ ) _𝛼_ ∧ type( _𝑇_ 1 _,𝐴_ ) � _𝛼_ ∧ _𝑝_ ( _𝑇_ 1 _,𝑇_ 2 ) s g **(R6)** If _𝑝_ ⊑ _𝑠_ ∈O then: _𝛼_ ∧ _𝑠_ ( _𝑇_ 1 _,𝑇_ 2 ) � _𝛼_ ∧ _𝑝_ ( _𝑇_ 1 _,𝑇_ 2 ) _𝛼_ ∧ _𝑝_ ( _𝑇_ 1 _,𝑇_ 2 ) � _𝛼_ ∧ _𝑠_ ( _𝑇_ 1 _,𝑇_ 2 ) s g **(R7)** If _𝑠_[−] ⊑ _𝑝_ ∈O then: _𝛼_ ∧ _𝑝_ ( _𝑇_ 1 _,𝑇_ 2 ) � _𝛼_ ∧ _𝑠_ ( _𝑇_ 2 _,𝑇_ 1 ) _𝛼_ ∧ _𝑠_ ( _𝑇_ 1 _,𝑇_ 2 ) � _𝛼_ ∧ _𝑝_ ( _𝑇_ 2 _,𝑇_ 1 ) s **(R8)** If _𝜃_ : _vars_ ( _𝑞_ )↦→ _vars_ ( _𝑞_ )∪E then: _𝛼_ ∧ _𝑝_ ( _𝑇_ 1 _,𝑇_ 2 ) ∧ _𝑝_ ( _𝑇_ 1[′] _[,𝑇]_ 2[′][)][∈] _[𝑞]_ � _𝛼𝜃_ ∧ _𝑝_ ( _𝑇_ 1 _,𝑇_ 2 ) _𝜃_ g _𝑠.𝑡. 𝜃_ ( _𝑇𝑖_ ) = _𝜃_ ( _𝑇𝑖_[′][)] _𝛼_ ∧ _𝑝_ ( _𝑇_ 1 _,𝑇_ 2 ) s.t. _𝑇_ 1 or _𝑇_ 2 ∈ E � _𝛼_ ∧ _𝑝_ ( _𝑍,𝑇_ 2 ) or _𝛼_ ∧ _𝑝_ ( _𝑇_ 1 _,𝑍_ ) 

are very general. Moreover, using all possible queries from Q G with their certain answers might be infeasible in practice. In the following, we discuss various options for guiding the sampling of queries to train ontology-aware knowledge graph embedding models for query answering. 

_3.1.2 Incorporating Query Rewritings and Certain Answers._ The first natural attempt to incorporate ontologies is to consider certain answers, which for _DL_ - _Lite_ R can be done efficiently. An example of this training case is to randomly sample query _𝑞_ ( _𝑌_ ) ← ∃ _𝑋._ hasAlumnus(mit _,𝑋_ ) ∧ worksFor( _𝑋,𝑌_ ) and, given (G, O) in Fig. 1, use it along with all its certain answers: mit _,_ yale during training. To incorporate the ontology, we can randomly sample queries over the KG, using the standard procedure, and then add their generalizations and specializations obtained using the rules in Tab. 2. To rewrite a query we select an atom and apply an ontology axiom. For example, the first rule **(R1)** applies a concept inclusion axiom, while **(R6)** applies a role inclusion. 

The specializations of a query _𝑞_ (i.e. _Spec_ ( _q_ )), incorporate more specific information regarding the answers of _𝑞_ , while the generalizations of _𝑞_ (i.e. _Gen_ ( _q_ )) incorporate additional related entities. 

_Example 3.3._ Take the queries _𝑞_ 1 ( _𝑋_ )←∃ _𝑌._ type( _𝑋,_ University) and _𝑞_ 2 ( _𝑋_ )←∃ _𝑍._ teachesAt( _𝑍,𝑋_ ). Using **R2** in Table 2 and (5) in s Figure 1 we get _𝑞_ 1� _𝑞_ 2, i.e., _𝑞_ 2 is a specialization of _𝑞_ 1. 

In general, there are exponentially many rewritings, thus we fix a rewriting depth, up to which the respective training queries are generated, via a dedicated parameter. 

_3.1.3 Strategic Ontology-Based Sampling._ While adding generalizations and specializations of randomly selected queries should partially reflect the background knowledge, many relevant axioms can be overlooked if they are not explicitly captured in the data. To overcome this, we consider the set of target query shapes as directed acyclic graphs (DAGs) of the form ( _𝑁, 𝐸_ ), where _𝑁_ is a set of nodes and _𝐸_ ⊆ _𝑁_ × _𝑁_ is a set of directed edges. The set of training queries is then obtained by applying a labeling function that assigns symbols in Σ to nodes and edges based on the ontology. 

_Definition 3.4 (Query Shape)._ A _query shape 𝑆_ is a tuple ( _𝑁, 𝐸,𝑛_ ) such that ( _𝑁, 𝐸_ ) is a DAG and _𝑛_ ∈ _𝑁_ is the _distinguished node_ of _𝑆_ 

(i.e., the node for the answer variable). For a given set of relations and constants in Σ, _a labeling function 𝑓_ : _𝑁_ ∪ _𝐸_ ↦→ Σ ∪ V maps each node to either a variable or an entity and each edge to a relation symbol in the KG signature Σ. 

We rely on the ontology when labeling query shapes to create semantically meaningful queries. Let ⊑[∗] be the reflexive and transitive closure of ⊑. Then, for a given relation _𝑝_ we have: 

- _inv_ ( _𝑝_ ) = { _𝑝_[′] | _𝑝_ ⊑ _𝑝_[′−] ∈O}, _dom_ ( _𝑝_ ) = { _𝐴_ | ∃ _𝑝_[′] ⊑ _𝐴_[′] ∈ O s.t. _𝑝_ ⊑[∗] _𝑝_[′] _,𝐴_ ⊑[∗] _𝐴_[′] or _𝐴_[′] ⊑[∗] _𝐴_ }, 

- _range_ ( _𝑝_ )={ _𝐴_ | ∃ _p_[′−] ⊑ _A_[′] ∈O s.t. _p_ ⊑[∗] _p_[′] _, A_ ⊑[∗] _A_[′] or _A_[′] ⊑[∗] _A_ }, 

- _follows_ ( _𝑝_ ) = { _𝑝_[′] | _range_ ( _𝑝_ ) ∩ _dom_ ( _𝑝_[′] ) ≠ ∅}, 

- _inter𝑟_ ( _𝑝_ ) = { _𝑝_[′] | _range_ ( _𝑝_ ) ∩ _range_ ( _𝑝_[′] ) ≠ ∅ or _𝑝_ 1 ∈ _inv_ ( _𝑝_ ) _, 𝑝_ 2 ∈ _inv_ ( _𝑝_[′] ) and _dom_ ( _𝑝_ 1 ) ∩ _dom_ ( _𝑝_ 2 ) ≠ ∅}, 

- _inter𝑑_ ( _𝑝_ ) = { _𝑝_[′] | _dom_ ( _𝑝_ ) ∩ _dom_ ( _𝑝_[′] ) ≠ ∅ or _𝑝_ 1 ∈ _inv_ ( _𝑝_ ) _, 𝑝_ 2 ∈ _inv_ ( _𝑝_[′] ) and _range_ ( _𝑝_ 1 ) ∩ _range_ ( _𝑝_ 2 ) ≠ ∅}. 

Intuitively, for a given relation _𝑝_ , the set _inv_ ( _p_ ) contains all inverse relations of _𝑝_ , _dom_ ( _𝑝_ ) contains all domain types for _𝑝_ , _range_ ( _𝑝_ ) all range types for _𝑝_ , _follows_ ( _𝑝_ ) stores all relations _𝑝_[′] which can follow _𝑝_ , and _inter𝑟_ ( _𝑝_ ), _inter𝑑_ ( _𝑝_ ) contain resp. all relations _𝑝_[′] which can intersect with _𝑝_ on range and domain positions. Then, for each shape we label nodes and edges to create queries that are valid w.r.t. O as shown in Figure 2. Note that this query sampling process uses only the ontology, i.e., it is data independent. 

## **3.2 Ontology-Aware Query2Box** 

Query2Box [36] is a prominent _query-based_ embedding models, in which entities and queries are embedded as _points_ and _boxes_ , resp., in a _𝑑_ -dimensional vector space. A _𝑑-dimensional embedding_ is a function _𝜑_ that maps _𝑐_ ∈ E ∪ C to c ∈ R _[𝑑]_ and a query _𝑞_ to q=(cen _𝑞,_ off _𝑞_ )∈R _[𝑑]_ × R _[𝑑]_ ≥0[, which is used to define a] _[ query box]_[as] 

box _𝑞_ = {v ∈ R _[𝑑]_ | cen _𝑞_ − off _𝑞_ ⪯ v ⪯ cen _𝑞_ + off _𝑞_ } _,_ 

where ⪯ is the element-wise inequality, cen _𝑞_ is the center of the box, and off _𝑞_ is the positive offset of the box, modeling its size. The score for an entity _𝑐_ being an answer to _𝑞_ is computed based on the distance from c to box _𝑞_ . The Query2Box model relies on the following geometric operators. 

_3.2.1 Projection._ Let _𝑆_ ⊆ E ∪ C be a set of entities, and _𝑟_ ∈ R a relation. Intuitively, the _projection_ operator performs graph traversal, 

Combining Inductive and Deductive Reasoning for Query Answering over Incomplete Knowledge Graphs 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/cikm_2023_Combining_Inductive_and_Deductive_Reasoning_for_Query_Answering_over_Incomplete_Knowledge_Graphs_images/cikm_2023_Combining_Inductive_and_Deductive_Reasoning_for_Query_Answering_over_Incomplete_Knowledge_Graphs.pdf-0005-01.png)


**Figure 2: Ontology-driven rules to label query shapes;** _𝑟_[−] **denotes any of** _inv_ ( _𝑟_ ) **.** 

e.g. given an entity _𝑒_ , the projection operator for the relation _𝑟_ provides the box corresponding to the set { _𝑒_[′] ∈ E ∪ C | _𝑟_ ( _𝑒,𝑒_[′] ) ∈G}. Given the embedding r = (cen _𝑟 ,_ off _𝑟_ ) ∈ R _[𝑑]_ × R _[𝑑]_ ≥0[for the relation] _𝑟_ , we model the projection of a box v = (cen _𝑣,_ off _𝑣_ ) by applying element-wise summation v + r = (cen _𝑣_ + cen _𝑟 ,_ off _𝑣_ + off _𝑟_ ). This relational translation [5] operation corresponds to the translation and enlargement of the box v. 

_3.2.2 Intersection._ Given a set of entity sets { _𝑆_ 1 _, . . . ,𝑆𝑛_ }, each of which is represented by a box in Query2Box, the _intersection_ operator computes their intersection. The intersection w = (cen _𝑤,_ off _𝑤_ ) of a set of boxes {(cen _𝑣_ 1 _,_ off _𝑣_ 1 ) _, . . . ,_ (cen _𝑣𝑛 ,_ off _𝑣𝑛_ )} for { _𝑆_ 1 _, . . . ,𝑆𝑛_ } is modeled by applying the following operations: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/cikm_2023_Combining_Inductive_and_Deductive_Reasoning_for_Query_Answering_over_Incomplete_Knowledge_Graphs_images/cikm_2023_Combining_Inductive_and_Deductive_Reasoning_for_Query_Answering_over_Incomplete_Knowledge_Graphs.pdf-0005-05.png)


where ⊙ and min denote the element-wise multiplication and minimum, respectively. NN: R _[𝑑]_ → R _[𝑑]_ is a 2-layer feed-forward neural network having the same dimensionality for the hidden layers as for the input layer. Φ and _𝜎_ stand for the softmax and sigmoid functions, resp., applied in a dimension-wise manner. Ψ is a permutation invariant function composed of a 2-layer feed-forward network followed by element-wise mean operation and a linear transformation. The center cen _𝑤_ is calculated as the weighted mean of the box centers cen _𝑣_ 1 _, . . . ,_ cen _𝑣𝑛_ . This geometric intersection provides a smaller box that lies inside a given set of boxes – for more details, we refer the reader to [36]. 

The goal of the Query2Box model is to learn the embedding of queries, such that the _distance_ between the box corresponding to the query and its answers is minimized, while the _distance_ to this box from other randomly sampled non-answers is maximized. 

In what follows, we present our proposal for integrating ontological axioms into the Query2Box model. Similarly to [36], we define the distance between q ∈ R _[𝑑]_ × R _[𝑑]_ ≥0[and][ v][ ∈][R] _[𝑑]_[as] _[ 𝑑]_[(][q] _[,]_[ v][)][=] ∥cen _𝑞_ − v∥1, namely the _𝐿_ 1 distance from the entity v to the center of the box. Using the sigmoid function we transform the distance into the (0 _,_ 1) interval, that is, _𝑝_ (v | q) = _𝜎_[�] −( _𝑑_ (q _,_ v) − _𝛾_ )[�] , where _𝛾 >_ 0 is a margin, which denotes the probability of _𝑣_ ∈ _𝑞_ [G _[𝑖] ,_ O]. Note that for every ontological axiom its both left- and righthand side can be turned into queries. When embedding those queries as boxes, axioms can be naturally enforced if in the vector space the inclusion of the boxes corresponding to the respective queries is ensured. For a query _𝑞_ , let _𝐺𝑒𝑛_ ( _𝑞_ ) = { _𝑞_ 1 _. . .𝑞𝑛_ } be the set of all generalizations of _𝑞_ based on O. Given a train query _𝑞_ and _𝑣_ ∈ _𝑞_ [G _,_ O], we aim at maximizing[�] _[𝑛] 𝑖_ =1 _[𝑝]_[(][v][ |][ q] _[𝑖]_[)] _[𝛽][𝑖]_[, where] _[ 𝛽][𝑖]_[≥][0] is a weighting parameter for all _𝑖_ = 1 _, . . . ,𝑛_ . This is achieved by minimizing the negative log-likelihood: − log �� _𝑛𝑖_ =1 _[𝑝]_[(][v][ |][ q] _[𝑖]_[)] _[𝛽][𝑖]_ � = 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/cikm_2023_Combining_Inductive_and_Deductive_Reasoning_for_Query_Answering_over_Incomplete_Knowledge_Graphs_images/cikm_2023_Combining_Inductive_and_Deductive_Reasoning_for_Query_Answering_over_Incomplete_Knowledge_Graphs.pdf-0005-09.png)


**Figure 3: Illustration of our extension of Query2Box. The KG nodes and relations are embedded as points and projection operators, resp. The axiom** teachesAt ⊑ worksFor **is captured by the inclusion of the respective boxes for queries.** 

−[�] _[𝑛] 𝑖_ =1 _[𝛽][𝑖]_[log][ �] _[𝑝]_[(][v][ |][ q] _[𝑖]_[)][�] _[.]_[ By exploiting that] _[ 𝜎]_[(] _[𝑥]_[)][=][1][ −] _[𝜎]_[(−] _[𝑥]_[)][, for] any v[′] _𝑗_[∉] _[𝑞]_[[G] _[,]_[ O]][, we have] _[ 𝑝]_[(][v][′][ |][ q][)][=][ 1][−] _[𝑝]_[(][v][ |][ q] _[𝑖]_[)][=] _[ 𝜎]_[(] _[𝑑]_[(][q] _[,]_[ v][)−] _[𝛾]_[)] _[.]_ Our goal is to enforce that if _𝑞_[′] ∈ _Gen_ ( _𝑞_ ) then the box of _𝑞_[′] contains the box of _𝑞_ . In order for that to hold, we need to ensure that, if _𝑎_ is an answer to _𝑞_ then the distance not only between _𝑎_ and _𝑞_ should be minimized, but also between _𝑎_ and all generalizations of _𝑞_ . The following training objective reflects our goal: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/cikm_2023_Combining_Inductive_and_Deductive_Reasoning_for_Query_Answering_over_Incomplete_Knowledge_Graphs_images/cikm_2023_Combining_Inductive_and_Deductive_Reasoning_for_Query_Answering_over_Incomplete_Knowledge_Graphs.pdf-0005-12.png)


where v[′][is][a][random][entity][for][all] _[𝑗]_[=][1] _[, . . . ,𝑘]_[ob-] _𝑗_[∉] _[𝑞]_[[G] _[,]_[ O]] tained via negative sampling. In our experiments, we use _𝛽𝑖_ = | _𝐺𝑒𝑛_ ( _𝑞_ )|[−][1] =[1] / _𝑛_ . 

_Example 3.5._ In Figure 3, the entities and relations are embedded into the vector space as points and projection operators, resp. The embedding of _𝑞_ ( _𝑌_ ) ←∃ _𝑋._ hasAlumnus(mit _,𝑋_ ) ∧ worksFor( _𝑋,𝑌_ ) is represented by the larger gray box, obtained by applying the projection hasAlumnus to the embedding of entity mit followed by the projection on worksFor. To enforce teachesAt ⊑ worksFor we ensure that the box of _𝑞_[′] ( _𝑌_ ) ←∃ _𝑋._ hasAlumnus(mit _,𝑋_ ) ∧ teachesAt( _𝑋,𝑌_ ), is contained in the box corresponding to _𝑞_ . 

Conceptually, our training data sampling techniques and the loss function modifications are flexible in terms of the DL, in which the ontology is encoded. The only restriction is the existence of efficient query rewriting algorithms. 

## **3.3 Ontology-aware CQD** 

A prominent atom-based query-answering method is CQD [2], which relies on neural link predictors for answering atomic subqueries, and then aggregates the resulting scores via t-norms. 

We now describe how we inject the ontology axioms into the neural link predictor employed by CQD [2]. For that we rely on the FO translation of the DL axioms. Following [32], for each rule the goal is to identify the entity embeddings which maximize an _inconsistency loss_ , i.e., the entities for which the scoring of the head is much lower compared to the scoring of the body. For example, given the rule Γ: ∀ _𝑋,𝑌_ teachesAt( _𝑋,𝑌_ ) → type( _𝑌,_ University), the goal is to map 

Medina Andresel, Trung-Kien Tran, Csaba Domokos, Pasquale Minervini, & Daria Stepanova 

the variables to _𝑑_ -dimensional embeddings, i.e. _𝜙_ : _var_ (Γ) ↦→ R _[𝑑]_ , s.t. [ _score_ teachesAt ( _𝜙_ ( _𝑋_ ) _,𝜙_ ( _𝑌_ )) − _score_ type ( _𝜙_ ( _𝑌_ ) _,_ University)]+ is maximal, where [ _𝑥_ ]+ = _max_ ([ _𝑥_ ] _,_ 0) with [ _𝑥_ ] being the integral part of x, and _score𝑟_ being the scoring function for the relation _𝑟_ determining whether there is an _𝑟_ -edge between any two given entities. Mapping _𝜙_ determines a so-called _adversarial input set_ , which is used as an adaptive regulariser for the neural link predictor. The inconsistency loss is then incorporated into the final loss function of the ontology-aware model which tries to minimize the maximal inconsistency loss while learning to predict the target graph over the given sets of correct triples. In experiments we rely on the existing implementation of the adversarial sets regularisation method in ComplEx-N3, which is the default neural link predictor for CQD. 

## **4 EVALUATION** 

We evaluate our training strategies on popular models: Query2Box ( _Q2B_ ) and _CQD_ , as well as our ontology-aware adaptations _O2B_ and _CQD[ASR]_ . Specifically, we test their ability to perform inductive reasoning, deductive reasoning, and their combination. 

## **4.1 Benchmarks for E-OMQA** 

Since the task of embedding-based ontology mediated query answering has not been considered in the literature before, no benchmarks for it existed prior to our work. Thus, we have created two novel benchmarks based on LUBM [17] and NELL [8] KGs, available online[3] . LUBM has a rich ontology including domain and range axioms as well as concept and role inclusions, while the NELL KG is accompanied with a more simple ontology containing only (inverse) role inclusions. Following common practice, each input KG is completed w.r.t. inverse edges. In Table 3 we present the number of ontology axioms of various types as well as the number of (materialized) triples, entities and relations in these datasets. 

_4.1.1 Query and Answers Sampling._ We use the same type of queries (corresponding to directed acyclic graphs with entities as the source nodes, also known as _anchors_ ) as in prior work [36] (see Figure 4). We assume that each input KG is complete (i.e. G _[𝑖]_ ) and then partition it into G _𝑣𝑎𝑙𝑖𝑑_ for validation and G _𝑡𝑟𝑎𝑖𝑛_ for training by discarding 10% of edges at each step; this yields G _𝑡𝑟𝑎𝑖𝑛_ ⊊ G _𝑣𝑎𝑙𝑖𝑑_ ⊊ G. We then create several training sets of queries according to our ontology-aware data sampling strategies from Section 3.1: 

• _plain_ : the training queries are randomly sampled from G _train_ , and we take their plain answers, i.e. over G _train_ . 

• _gen_ : queries in _plain_ augmented with their ontology-based generalizations[4] ; answers are certain, i.e., over O[∞] (G _train_ ). 

• _spec_ : queries from _gen_ augmented with specializations; 

• _onto_ : queries from Section 3.1, with randomly chosen percentage of valid entities as anchors; all answers are certain. 

Specializations and generalizations non-compliant with the shapes from Figure 4 are discarded. Note that for NELL, the _plain_ data is exactly the one from [37]. We observe that the number of 1p queries obtained for _gen_ and _spe_ settings are identical. This is probably because the set of 1p queries in _plain_ covers all edges in the training KG. For the LUBM dataset, we have created the training and testing 

3https://github.com/medinaandresel/eomqa 

> 4This is similar to random sampling over O∞ (G _train_ ) but unlike the deductive closure, our procedure is guaranteed to terminate. We used the rewriting depth of up to 10. 

sets from scratch, and the 1p queries in _plain_ do not contain the entire training KG. The _onto_ set of queries leverages the proposed ontology-driven technique, given that the ontology covers all relations and concepts in the KG and describes how they interact, i.e., the ontology axioms support all the constructed queries, and we chose 50 % of valid entities as anchors. As there are too many queries to chose from, due to the large number of relations, we had to select a smaller number of valid entities as anchors, namely 20-30%. This explains the smaller number of 1p queries. Moreover, the NELL ontology does not contain interesting axioms that can be leveraged by ontology-driven query sampling technique, thus to obtain _onto_ we had to rely on the patterns from the data alone. 

We generate three different test sets for verifying the ability of the query answering model to perform inductive reasoning, deductive reasoning and their combination. More formally, 

- **Inductive case (I)** . Is the model able to predict missing answers to queries over the complete, but not observable KG G _[𝑖]_ ? (accounts for the standard test case) 

- **Deductive case (D)** . Is the model able to predict answers that can be inferred from the known triples in G _train_ using ontology? 

- **Inductive + Deductive case (I+D)** . Is the model able to predict missing answers inferred from the complete but not observable KG G _[𝑖]_ using axioms in O? 

For test case **I** , respectively **I+D** , test queries are randomly sampled over G, respectively O[∞] (G), while for **D** they are randomly sampled over O[∞] (G _train_ ) s.t. they cannot be trivially answered over G _train_ , and unseen during training. In each test case the validation queries are generated similarly but over G _𝑣𝑎𝑙𝑖𝑑_ . 

The size of each training/testing set, and the number of queries per shape for each of the considered cases are presented in Table 4. For each test and validation query, we measure the accuracy based on _hard (certain) answers_ , i.e., those that cannot be trivially answered over G _train_ (or G _valid_ for test queries) and require prediction of missing edges and/or application of ontology axioms (see Definition 2.3 and 3.2). 

## **4.2 Models and Evaluation Procedure** 

We consider _Q2B_ , _O2B_ , _CQD_ and _CQD[ASR]_ trained in each described setting: i.e., _𝑀𝑥_ , where _𝑀_ ∈{ _Q2B_ , _O2B_ , _CQD_ , _CQD[ASR]_ } and _𝑥_ ∈{plain, gen, spec, onto}; _Q2Bplain_ and _CQDplain_ are taken as baselines. _Q2B_ and _O2B_ are trained on five query shapes that require projection and intersection, while _CQD_ and _CQD[ASR]_ are trained on atomic queries. We have configured both _Q2B_ and _O2B_ systems as follows: The size of the embedding dimension was set to 400, and the models were trained for 15 × 10[4] steps using Adam optimizer with an initial learning rate of 10[−][4] and the batch size of 512. The rest of the parameters were set in the same way as in [36]. 

For CQD, we used the code from [2] with ComplEx-N3 [29] employed as the base model. The embedding size was set to 1000, and the regularisation weight was selected based on the validation set by searching in {10[−][3] _,_ 5 × 10[−][3] _, . . . ,_ 10[−][1] }. For LUBM, the regularization weight was set to 0 _._ 1 in the _gen_ , _spe_ , and _onto_ settings, and to 0 _._ 01 in the _plain_ setting. For NELL, the regularization weight was set to 0 _._ 005 in the _plain_ setting, to 0 _._ 001 in the _gen_ and _spe_ settings, and to 0 _._ 05 in the _onto_ setting. For _CQD[ASR]_ we have additionally 

Combining Inductive and Deductive Reasoning for Query Answering over Incomplete Knowledge Graphs 

**Table 3: The total number of axioms** |O| **and of each type, the size of the input KG** |G| **, the number of entities** |E| **, the number of relations** |R| **, and the number of materialized triples** |O[∞] (G)| **.** 

|**Dataset**|**Ontology** O<br>**KG** G<br>|O|<br>_𝐴_⊑_𝐴_′<br>_𝑝_⊑_𝑠_<br>_𝑝_−⊑_𝑠_<br>∃_𝑝_⊑_𝐴_<br>∃_𝑝_−⊑_𝐴_<br>| G|<br>|E|<br>|R|<br>|O∞(G) ||**Ontology** O<br>**KG** G<br>|O|<br>_𝐴_⊑_𝐴_′<br>_𝑝_⊑_𝑠_<br>_𝑝_−⊑_𝑠_<br>∃_𝑝_⊑_𝐴_<br>∃_𝑝_−⊑_𝐴_<br>| G|<br>|E|<br>|R|<br>|O∞(G) ||
|---|---|---|
|LUBM<br>NELL|68<br>13<br>5<br>28<br>11<br>11<br>307<br>–<br>92<br>215<br>–<br>–|284K<br>55684<br>28<br>565K<br>285K<br>63361<br>400<br>497K|



**Table 4: Number of test and train queries of each shape in each of the settings.** 

|**Dataset**|**Train/Test**|**Query Shape**<br>1p<br>2p<br>3p<br>2i<br>3i<br>ip<br>pi<br>2u<br>up|
|---|---|---|
|LUBM|_Plain_<br>_Gen_<br>_Spe_<br>_Onto_|110000<br>110000<br>110000<br>110000<br>110000<br>–<br>–<br>–<br>–<br>117124<br>136731<br>150653<br>181234<br>208710<br>–<br>–<br>–<br>–<br>117780<br>154851<br>173678<br>271532<br>230085<br>–<br>–<br>–<br>–<br>116893<br>166159<br>333406<br>212718<br>491707<br>–<br>–<br>–<br>–|
||**I**<br>**D**<br>**I+D**|8000<br>8000<br>8000<br>8000<br>8000<br>8000<br>8000<br>8000<br>8000<br>1241<br>4701<br>6472<br>3829<br>4746<br>7393<br>7557<br>4986<br>7122<br>8000<br>8000<br>8000<br>8000<br>8000<br>8000<br>8000<br>7986<br>8000|
|NELL|_Plain_<br>_Gen_<br>_Spe_<br>_Onto_|107982<br>107982<br>107982<br>107982<br>107982<br>–<br>–<br>–<br>–<br>174310<br>408842<br>864268<br>398412<br>930787<br>–<br>–<br>–<br>–<br>174310<br>419664<br>906609<br>401954<br>936537<br>–<br>–<br>–<br>–<br>114614<br>542923<br>864268<br>629144<br>930787<br>–<br>–<br>–<br>–|
||**I**<br>**D**<br>**I+D**|15688<br>3910<br>3918<br>3828<br>3786<br>3932<br>3895<br>3940<br>3966<br>346<br>4461<br>4294<br>4842<br>5996<br>7295<br>5862<br>5646<br>6894<br>8000<br>8000<br>8000<br>8000<br>8000<br>8000<br>8000<br>7990<br>8000|
|1p|up<br>u<br>u<br>u<br>2u<br>u<br>3i<br>2p<br>3p<br>2i<br>ip<br>pi||



**Figure 4: Query shapes considered in our experiments, where blue nodes correspond to anchor entities and red ones to answer variables; p stands for projection, i for intersection and u for union. The first five shapes are used in training.** 

used a regularisation weight of the following values: 10[−][2] , 10[−][3] and 10[−][4] . The batch size of the adversarial examples was set to 32. 

We evaluated the models periodically and report the test results of the models with the best performance on the validation dataset. The performance of each trained model is measured using standard metric HITS@ _𝐾_ for _𝐾_ =3 (HITS@3), indicating the frequency that the correct answer is ranked among the top-3 results. 

## **4.3 Evaluation Results** 

_4.3.1 Inductive+Deductive case._ First, we present the detailed results for the most challenging setting ( **I+D** ) for LUBM and NELL in Table 5 and Figure 5. 

Based on the average accuracy of the models across all query shapes reported in Table 5, the improvements of the proposed ontology-aware adaptations of _Q2B_ and _CQD_ are evident. For LUBM _O2B_ trained using the _onto_ strategy improves the _Q2B_ baseline by almost 50%, while in case of _CQD_ , 54% enhancement is achieved. For NELL similar behaviour is observed with the improvement of almost 20% for _Q2B_ , and 25% for _CQD_ . Next, we discuss the impact of each of the proposed techniques for the E-OMQA task. 

The first observation is that incorporating the ontology in the training data is crucial as both _Q2B_ and _CQD_ trained in settings _gen_ 

and _onto_ yield significant improvements over the baselines. Additional incorporation of specializations (setting _spec_ ) does not seem to have a major impact though (see Figure 5). On LUBM, for all models, the advantage of the ontology-driven query sampling (i.e., _onto_ setting) is significant compared to _gen_ setting. Remarkably, for LUBM _CQDonto_ , resp. _CQD[ASR] onto_[trained on less data than] _[ CQD] gen_[,] resp. _CQD[ASR] gen_[results in higher accuracy. This shows that random] query sampling is not adequate for E-OMQA. The ontology for NELL is not expressive enough, thus, when generating training queries in the _onto_ setting (see Table 3 for statistics) we proceeded in a bottom-up fashion as follows: We randomly labeled query shapes which produce answers, and constructed their generalizations as before; thus the settings _gen_ and _onto_ are similar, but _onto_ has significantly less atomic queries, which explains why _CQDgen_ outperforms _CQDonto_ on NELL. 

On average ontology-aware models (i.e., _O2B_ and _CQD[ASR]_ ) significantly outperform their baselines (i.e., _Q2B_ and _CQD_ , resp.) for the majority of training data sampling strategies. This trend is more prominent for atom-based models on the complex LUBM ontology, and for query-based ones on NELL, which is less expressive. 

In Table 6 we present the detailed results for the query rewriting over pre-trained embeddings. In order to evaluate this procedure, for 

Medina Andresel, Trung-Kien Tran, Csaba Domokos, Pasquale Minervini, & Daria Stepanova 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/cikm_2023_Combining_Inductive_and_Deductive_Reasoning_for_Query_Answering_over_Incomplete_Knowledge_Graphs_images/cikm_2023_Combining_Inductive_and_Deductive_Reasoning_for_Query_Answering_over_Incomplete_Knowledge_Graphs.pdf-0008-01.png)


**----- Start of picture text -----**<br>
·10 [6] ·10 [6]<br>0 . 8 3<br>0 . 6<br>1<br>0 . 7<br>2<br>0 . 4 0 . 6<br>0 . 5<br>1<br>0 . 5<br>0 . 2<br>0 0<br>plain gen spe onto plain gen spe onto<br>Avg. HITS@3  Q2B Train  Q2B Avg. HITS@3  Q2B Train  Q2B<br>Avg. HITS@3  O2B Train  O2B Avg. HITS@3  O2B Train  O2B<br>Avg. HITS@3  CQD Train  CQD Avg. HITS@3  CQD Train  CQD<br>Avg. HITS@3  CQD [ASR] Train  CQD [ASR] Avg. HITS@3  CQD [ASR] Train  CQD [ASR]<br>LUBM NELL<br>HITS@3 HITS@3<br>#Training Queries #Training Queries<br>**----- End of picture text -----**<br>


**Figure 5: Performance of** _Q2B_ **,** _O2B_ **,** _CQD_ **, and** _CQD[ASR]_ **on I+D and size of the training set for each setting** _plain_ **,** _gen_ **,** _spe_ **,** _onto_ **. The number of training queries is scaled by multiplying with** 10[6] **.** 

**Table 5: HITS@3 scores in the inductive and deductive setting (I+D) for each query shape (the higher the better)** 

|**Dataset**|**LUBM**<br>**NELL**|**LUBM**<br>**NELL**|**LUBM**<br>**NELL**|**LUBM**<br>**NELL**|**LUBM**<br>**NELL**|**LUBM**<br>**NELL**|**LUBM**<br>**NELL**|**LUBM**<br>**NELL**|**LUBM**<br>**NELL**|**LUBM**<br>**NELL**|
|---|---|---|---|---|---|---|---|---|---|---|
|**Model**|**Avg.**|**1p**<br>**2p**<br>**3p**|**2i**<br>**3i**|**ip**<br>**pi**|**2u**<br>**up**|**Avg.**|**1p**<br>**2p**<br>**3p**|**2i**<br>**3i**|**ip**<br>**pi**|**2u**<br>**up**|
|_Q2Bplain_<br>_O2Bplain_<br>_CQDplain_<br>_CQDASR_<br>_plain_|0.218|0.173 0.101 0.107|0.433 0.546|0.167 0.200|0.133 0.100|0.458|0.516 0.343 0.286<br>**0.79**<br>0.409 **0.359**<br>0.664 0.383 0.304<br>0.716 **0.518** 0.337|0.747<br>0.81<br>**0.904 0.936**<br>0.853 0.903<br>0.807 0.831|0.404 0.447<br>0.479 **0.521**<br>0.471 0.512<br>**0.547** 0.513|0.325 0.241<br>**0.666** 0.303<br>0.599 0.306<br>0.614 **0.445**|
||0.245|0.235 0.109 0.095|0.488 0.584|0.176 0.218|0.2<br>0.103|**0.596**|||||
||0.179|0.109 0.058 0.104|0.384 0.502|0.130 0.187|0.092 0.046|0.555|||||
||**0.56**|**0.682 0.589 0.393**|**0.659 0.664**|**0.547 0.488**|**0.509 0.509**|0.592|||||
|_Q2Bgen_<br>_O2Bgen_<br>_CQDgen_<br>_CQDASR_<br>_gen_|0.458|0.592 0.267 0.129|**0.789 0.870**|0.360 0.282|0.552 0.279|0.642|0.858 0.485 0.397<br>0.859 0.494 0.420<br>**0.903 0.775 0.473**<br>0.9<br>0.771 0.467|0.928<br>0.95<br>0.928 0.953<br>**0.957 0.969**<br>0.919 0.924|0.538 0.539<br>0.552 0.559<br>**0.821 0.757**<br>0.793 0.723|0.768 0.312<br>0.77<br>0.329<br>**0.886 0.743**<br>0.846 0.741|
||0.447|0.577 0.257 0.114|0.777 0.859|0.359<br>0.27|0.546 0.264|0.652|||||
||0.408|0.539 0.214 0.098|0.710 0.791|0.304 0.302|0.513 0.208|**0.809**|||||
||**0.628**|**0.733 0.640 0.413**|0.717 0.720|**0.598 0.599**|**0.653 0.582**|0.787|||||
|_Q2Bonto_<br>_O2Bonto_<br>_CQDonto_<br>_CQDASR_<br>_onto_|0.687|0.762 0.617 0.447|0.868 0.915|0.693 0.555|0.732 0.600|0.636|0.858 0.472 0.398<br>0.862 0.498 0.423<br>0.667 0.368 0.293<br>**0.9**<br>**0.741 0.456**|0.927 0.948<br>**0.933 0.953**<br>0.848 0.904<br>0.922 0.923|0.529 0.524<br>0.557 0.555<br>0.453 0.506<br>**0.77**<br>**0.701**|0.747 0.317<br>0.773 0.340<br>0.595 0.275<br>**0.851 0.672**|
||0.707|**0.771** 0.629 0.476|**0.878 0.927**|0.694 0.619|**0.752 0.618**|0.655|||||
||**0.723**|0.752 **0.681 0.481**|0.870 0.924|**0.735 0.728**|0.738 0.604|0.545|||||
||0.664|0.753 **0.681** 0.421|0.744 0.755|0.643 0.666|0.704 0.615|**0.77**|||||



each hard answer _𝑎_ we take the best (i.e., minimum) ranking among all rankings generated by all queries in the rewriting of each test query. In other words, we take the minimal distance between the embedding of _𝑎_ and all rewritings of _𝑞_ . Note that, for measuring the performance we use the pre-trained models _Q2Bplain_ , and _CQDplain_ obtained after 450K training steps. Due to the reliance on particular query shapes of the respective models, the complete rewriting for each query is not guaranteed. In Table 6, we present the results for this method compared to the _plain_ setting. Minor improvements of only at most 10% are observed. 

_4.3.2 Inductive case._ Next, we present the average HITS@3 metric for the inductive **I** test case (see Figure 6). For **I** ontology-injection methods do not yield any improvement, which is expected, since ontologies cannot handle missing edges and facts in a KG that are not inferred from the data using ontological reasoning. 

_4.3.3 Deductive case._ For the test case **D** , when the ontology is simple (e.g., NELL), CQD and Q2B are able to more or less learn 

to apply the ontology rules when they are explicitly injected in the training set. Moreover, the results on NELL in the _plain_ setting show that rule enforcement is also competitive for deductive reasoning. For query-based models the best performance is achieved by combining ontology-driven sampling with rule enforcement, while for atom-based models, the inclusion of generalizations seems to be already sufficient. 

For expressive ontologies, such as LUBM, the ontology-driven query sampling is crucial for optimal performance. For query-based models the best results are achieved when the ontology-driven query sampling is combined with rule enforcement, while for atombased models rule enforcement does not seem to be necessary. 

## **5 RELATED WORK** 

The task of answering queries that involve multiple atoms using embedding techniques has recently received a lot of attention (see [35] for overview). The existing proposals can be divided into _querybased_ (e.g., [9, 25, 30, 36, 37, 39, 44, 45]) and _atom-based_ (e.g., [2, 3]). 

Combining Inductive and Deductive Reasoning for Query Answering over Incomplete Knowledge Graphs 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/cikm_2023_Combining_Inductive_and_Deductive_Reasoning_for_Query_Answering_over_Incomplete_Knowledge_Graphs_images/cikm_2023_Combining_Inductive_and_Deductive_Reasoning_for_Query_Answering_over_Incomplete_Knowledge_Graphs.pdf-0009-01.png)


**----- Start of picture text -----**<br>
0 . 34 1<br>0 . 7<br>0 . 8 0 . 32 0 . 9<br>0 . 3<br>0 . 65 0 . 6 0 . 8<br>0 . 28<br>0 . 6 0 . 4 0 . 26 0 . 7<br>0 . 24 0 . 6<br>0 . 55 0 . 2 0 . 22 0 . 5<br>plain gen spe onto plain gen spe onto plain gen spe onto plain gen spe onto<br>Q2B CQD O2B CQD [ASR] Q2B CQD O2B CQD [ASR] Q2B CQD O2B CQD [ASR] Q2B CQD O2B CQD [ASR]<br>LUBM: test case I LUBM: test case D NELL: test case I NELL: test case D<br>Hits@3 Hits@3 Hits@3 Hits@3<br>**----- End of picture text -----**<br>


**Figure 6: Comparison of** _Q2B_ **,** _O2B_ **,** _CQD_ **and** _CQD[ASR]_ **in each training setting for test cases (I) and (D)** 

**Table 6: Avg. HITS@3 for QA of shapes 1p, 2p, 3p, 2i, 3i using rewriting on top of pre-trained** _plain_ **model vs the** _plain_ **model.** 

|**Models**|**Test Case D**<br>LUBM<br>NELL|**Test Case I+D**<br>LUBM<br>NELL|
|---|---|---|
|_Q2Bplain_<br>_Q2Brew_<br>_plain_|0.189<br>0.617<br>0.248<br>0.683|0.193<br>0.539<br>0.261<br>0.639|
|**Gain**|+0.059<br>+0.066|+0.068<br>+ 0.1|
|_CQDplain_<br>_CQDrew_<br>_plain_|0.225<br>0.656<br>0.228<br>0.743|0.231<br>0.621<br>0.249<br>0.708|
|**Gain**|+0.003<br>+0.087|+0.018<br>+ 0.087|



## **6 CONCLUSION** 

We have presented methods for Embedding-based Ontology Mediated Query Answering (E-OMQA) that operate in the embedding space to enable simultaneous inductive and deductive reasoning over the incomplete data. Experiments show that embedding-based methods for query answering applied naively or combined with query rewriting techniques are not effective. At the same time, our ontology-aware extensions of the popular models for embeddingbased QA and the proposed ontology-driven training strategies yield promising results on the novel benchmarks that we introduce for the considered task. 

For future work we plan to study the effectiveness of our methods for embedding-based ontology mediated query answering for other more complex query forms [37, 40, 42], e.g., queries with negation, as well as evaluate the proposed approach for the cases when the ontology is more expressive. 

## **ACKNOWLEDGMENTS** 

The works [14] and [6] study the relation between the problem of conjunctive QA in the embedding space and over probabilistic databases. Our work is different from the above proposals in that along with the data we also rely on ontologies. 

Integration of ontologies into KG embeddings has been recently actively investigated, for instance, in [1, 13, 16, 21, 23, 24, 26, 32, 41] (see also [43]), but these works typically focus on the task of link prediction rather than query answering. Recently, a type-aware model (called TEMP) for query answering over incomplete KGs has been proposed [22]. While TEMP allows for the exploitation of the type information, to the best of our knowledge it cannot handle more complex ontological axioms, which are the focus of our work. 

The capability of embeddings to model hierarchical data has been explored in several works, e.g., [18, 34]. Another relevant direction is concerned with the construction of models for EL ontologies in the embedding space [28]. While the above works are related, they do not touch upon the problem of OMQA, studied in this work. 

The problem of ontology-mediated query answering has been considered in the area of knowledge representation and reasoning (see e.g. [38] for an overview), but available methods, e.g.[11, 15], only focus on logic-based deductive reasoning, but do not aim at predicting missing links in knowledge graphs using machine learning approaches. 

Pasquale was partially funded by the European Union’s Horizon 2020 research and innovation programme under grant agreement no. 875160, ELIAI (The Edinburgh Laboratory for Integrated Artificial Intelligence) EPSRC (grant no. EP/W002876/1), an industry grant from Cisco, and a donation from Accenture LLP; and is grateful to NVIDIA GPU donations. This work was partially funded by the European project SMARTEDGE (grant number 101092908). 

## **REFERENCES** 

- [1] Ralph Abboud, İsmail İlkan Ceylan, Thomas Lukasiewicz, and Tommaso Salvatori. 2020. BoxE: A Box Embedding Model for KB Completion. In _NeurIPS_ . 

- [2] Erik Arakelyan, Daniel Daza, Pasquale Minervini, and Michael Cochez. 2021. Complex Query Answering with Neural Link Predictors. In _ICLR_ . 

- [3] Erik Arakelyan, Pasquale Minervini, and Isabelle Augenstein. 2023. Adapting Neural Link Predictors for Complex Query Answering. _CoRR_ abs/2301.12313 (2023). 

- [4] Franz Baader, Ian Horrocks, and Ulrike Sattler. 2009. Description Logics. In _Handbook on Ontologies_ . 21–43. 

- [5] Antoine Bordes, Nicolas Usunier, Alberto García-Durán, Jason Weston, and Oksana Yakhnenko. 2013. Translating Embeddings for Modeling Multi-relational Data. In _NIPS_ . 2787–2795. 

- [6] Stefan Borgwardt, İsmail İlkan Ceylan, and Thomas Lukasiewicz. 2019. OntologyMediated QA over Log-Linear Probabilistic Data. In _AAAI_ . 2711–2718. 

- [7] Diego Calvanese, Giuseppe De Giacomo, _et al_ Rosati. 2007. Tractable Reasoning and Efficient Query Answering in Description Logics: The _DL-Lite_ Family. _J. Aut. R._ 39, 3 (2007), 385–429. 

- [8] Andrew Carlson, Justin Betteridge, Bryan Kisiel, Burr Settles, Estevam R. Hruschka Jr., and Tom M. Mitchell. 2010. Toward an Architecture for Never-Ending Language Learning. In _AAAI_ . 

Medina Andresel, Trung-Kien Tran, Csaba Domokos, Pasquale Minervini, & Daria Stepanova 

- [9] Nurendra Choudhary, Nikhil Rao, and Sumeet Katariya. 2021. Self-Supervised Hyperboloid Represent. from Logical Queries over KGs. In _WWW_ . 1373–1384. 

- [10] Nilesh N. Dalvi and Dan Suciu. 2007. Efficient query evaluation on probabilistic databases. _VLDB J._ 16, 4 (2007), 523–544. 

- [11] Thomas Eiter, Magdalena Ortiz, Mantas Simkus, Trung-Kien Tran, and Guohui Xiao. 2012. Query Rewriting for Horn-SHIQ Plus Rules. In _Proceedings of the Twenty-Sixth AAAI Conference on Artificial Intelligence, July 22-26, 2012, Toronto, Ontario, Canada_ , Jörg Hoffmann and Bart Selman (Eds.). AAAI Press. 

- [12] Fredo Erxleben, Michael Günther, Markus Krötzsch, Julian Mendez, and Denny Vrandecic. 2014. Introducing Wikidata to the Linked Data Web. In _ISWC_ . 

   - [42] Hang Yin, Zihao Wang, and Yangqiu Song. 2023. On Existential First Order Queries Inference on Knowledge Graphs. _CoRR_ abs/2304.07063 (2023). 

   - [43] Wen Zhang, Jiaoyan Chen, Juan Li, Zezhong Xu, Jeff Z. Pan, and Huajun Chen. 2022. Knowledge Graph Reasoning with Logics and Embeddings: Survey and Perspective. _CoRR_ abs/2202.07412 (2022). 

   - [44] Zhanqiu Zhang, Jie Wang, Jiajun Chen, Shuiwang Ji, and Feng Wu. 2021. ConE: Cone Embeddings for Multi-Hop Reasoning over Knowledge Graphs. In _NeurIPS 2021_ . 19172–19183. 

   - [45] Zhaocheng Zhu, Mikhail Galkin, Zuobai Zhang, and Jian Tang. 2022. NeuralSymbolic Models for Logical Queries on KGs. In _ICML_ , Vol. 162. 27454–27478. 

- [13] Bahare Fatemi, Siamak Ravanbakhsh, and David Poole. 2019. Improved KG Embedding Using Background Taxonomic Info. In _IAAI_ . 3526–3533. 

- [14] Tal Friedman and Guy Van den Broeck. 2020. Symbolic Querying of Vector Spaces. In _UAI_ . 1268–1277. 

- [15] Birte Glimm, Ian Horrocks, Carsten Lutz, and Ulrike Sattler. 2011. Conjunctive Query Answering for the Description Logic SHIQ. _CoRR_ abs/1111.0049 (2011). 

- [16] Shu Guo, Quan Wang, Lihong Wang, Bin Wang, and Li Guo. 2016. Jointly Embedding Knowledge Graphs and Logical Rules. In _EMNLP_ . 192–202. 

- [17] Yuanbo Guo, Zhengxiang Pan, and Jeff Heflin. 2005. LUBM: A benchmark for OWL knowledge base systems. _J. Web Semant._ 3, 2-3 (2005), 158–182. 

- [18] Víctor Gutiérrez-Basulto and Steven Schockaert. 2018. From Knowledge Graph Embedding to Ontology Embedding?. In _KR_ . 379–388. 

- [19] William L. Hamilton, Payal Bajaj, and Marinka Zitnik _et al_ . 2018. Embedding Logical Queries on Knowledge Graphs. In _NeurIPS_ . 2030–2041. 

- [20] William L. Hamilton, Payal Bajaj, Marinka Zitnik, Dan Jurafsky, and Jure Leskovec. 2018. Embedding Logical Queries on Knowledge Graphs. In _NeurIPS 2018_ . 2030–2041. 

- [21] Junheng Hao, Muhao Chen, and _et al_ Wenchao Yu. 2019. Universal Representation Learning of KBs by Jointly Embedding Instances and Ontological Concepts. In _SIGKDD_ . 1709–1719. 

- [22] Zhiwei Hu, Víctor Gutiérrez-Basulto, Zhiliang Xiang, Xiaoli Li, Ru Li, and Jeff Z. Pan. 2022. Type-aware Embeddings for Multi-Hop Reasoning over Knowledge Graphs. In _IJCAI 2022_ . 3078–3084. 

- [23] Nitisha Jain, Trung-Kien Tran, Mohamed H. Gad-Elrab, and Daria Stepanova. 2021. Improving Knowledge Graph Embeddings with Ontological Reasoning. In _ISWC 2021_ . 

- [24] Seyed Mehran Kazemi and David Poole. 2018. SimplE Embedding for Link Prediction in Knowledge Graphs. In _Neurips_ . 4289–4300. 

- [25] Bhushan Kotnis, Carolin Lawrence, and Mathias Niepert. 2021. Answering Complex Queries in KGs with Bidirectional Sequence Encoders. In _AAAI 2021_ . 4968–4977. 

- [26] Denis Krompaß, Stephan Baier, and Volker Tresp. 2015. Type-Constrained Representation Learning in Knowledge Graphs. In _ISWC (1)_ , Vol. 9366. 640–655. 

- [27] Denis Krompaß, Maximilian Nickel, and Volker Tresp. 2014. Querying Factorized Probabilistic Triple Databases. In _ISWC_ . 

- [28] Maxat Kulmanov, Wang Liu-Wei, Yuan Yan, and Robert Hoehndorf. 2019. EL Embeddings: Geometric construction of models for the Description Logic EL ++. _CoRR_ abs/1902.10499 (2019). 

- [29] Timothée Lacroix, Nicolas Usunier, and Guillaume Obozinski. 2018. Canonical Tensor Decomposition for KB Completion. In _ICML_ , Vol. 80. 2869–2878. 

- [30] Lihui Liu, Boxin Du, Heng Ji, ChengXiang Zhai, and Hanghang Tong. 2021. Neural-Answering Logical Queries on Knowledge Graphs. In _KDD_ . 1087–1097. 

- [31] Farzaneh Mahdisoltani, Joanna Biega, and Fabian M. Suchanek. 2015. YAGO3: A Knowledge Base from Multilingual Wikipedias. In _CIDR_ . 

- [32] Pasquale Minervini, Thomas Demeester, Tim Rocktäschel, and Sebastian Riedel. 2017. Adversarial Sets for Regularising Neural Link Predictors. In _UAI_ . 

- [33] Maximilian Nickel, Kevin Murphy, Volker Tresp, and Evgeniy Gabrilovich. 2016. A Review of Relational Machine Learning for Knowledge Graphs. _Proc. IEEE_ 104, 1 (2016), 11–33. 

- [34] Dhruvesh Patel, Shib Sankar Dasgupta, Michael Boratko, Xiang Li, Luke Vilnis, and Andrew McCallum. 2020. Representing Joint Hierarchies with Box Embeddings. In _AKBC_ . 

- [35] Hongyu Ren, Mikhail Galkin, and Michael Cochez _et al_ . 2023. Neural Graph Reasoning: Complex Logical Query Answering Meets Graph Databases. _CoRR_ abs/2303.14617 (2023). 

- [36] Hongyu Ren, Weihua Hu, and Jure Leskovec. 2020. Query2box: Reasoning over Knowledge Graphs in Vector Space Using Box Embeddings. In _ICLR_ . 

- [37] Hongyu Ren and Jure Leskovec. 2020. Beta Embeddings for Multi-Hop Logical Reasoning in Knowledge Graphs. In _NeurIPS_ . 

- [38] Thomas Schneider and Mantas Simkus. 2020. Ontologies and Data Management: A Brief Survey. _KI_ 34, 3 (2020), 329–353. 

- [39] Haitian Sun, Andrew O. Arnold, and Tania Bedrax-Weiss _et al_ . 2020. Faithful Embeddings for Knowledge Base Queries. In _NeurIPS_ . 

- [40] Zihao Wang, Hang Yin, and Yangqiu Song. 2021. Benchmarking the Combinatorial Generalizability of Complex QA on KGs. In _NeurIPS Datasets and Benchmarks_ . 

- [41] Bo Xiong, Nico Potyka, Trung-Kien Tran, Mojtaba Nayyeri, and Steffen Staab. 2022. Faithful Embeddings for _E_ ℒ[++] KBs. In _ISWC_ , Vol. 13489. 22–38. 

