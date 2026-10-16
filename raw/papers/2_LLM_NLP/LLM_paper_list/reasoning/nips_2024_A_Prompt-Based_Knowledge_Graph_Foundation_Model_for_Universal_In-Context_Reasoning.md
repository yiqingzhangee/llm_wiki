# **A Prompt-Based Knowledge Graph Foundation Model for Universal In-Context Reasoning** 

**Yuanning Cui** _[†]_ **, Zequn Sun** _[†]_ **, Wei Hu** _[†‡∗]_ 

> _†_ State Key Laboratory for Novel Software Technology, Nanjing University, Nanjing, China 

> _‡_ National Institute of Healthcare Data Science, Nanjing University, Nanjing, China `yncui.nju@gmail.com, {sunzq, whu}@nju.edu.cn` 

## **Abstract** 

Extensive knowledge graphs (KGs) have been constructed to facilitate knowledgedriven tasks across various scenarios. However, existing work usually develops separate reasoning models for different KGs, lacking the ability to generalize and transfer knowledge across diverse KGs and reasoning settings. In this paper, we propose a prompt-based KG foundation model via in-context learning, namely **KG-ICL** , to achieve a universal reasoning ability. Specifically, we introduce a prompt graph centered with a query-related example fact as context to understand the query relation. To encode prompt graphs with the generalization ability to unseen entities and relations in queries, we first propose a unified tokenizer that maps entities and relations in prompt graphs to predefined tokens. Then, we propose two message passing neural networks to perform prompt encoding and KG reasoning, respectively. We conduct evaluation on 43 different KGs in both transductive and inductive settings. Results indicate that the proposed KG-ICL outperforms baselines on most datasets, showcasing its outstanding generalization and universal reasoning capabilities. The source code is accessible on GitHub: `https://github.com/nju-websoft/KG-ICL` . 

## **1 Introduction** 

Reasoning on knowledge graphs (KGs) involves inferring new relational facts from existing ones. Early related work primarily focuses on reasoning over a static KG in the transductive setting, but lacks the generalization ability to handle new entities or relations in the KG. Recent research [1–4] considers the relational patterns between seen and unseen entities, enabling inductive reasoning. However, these methods still lack the transferability to reason over unseen KGs due to the unshared and unlinked entity and relation vocabularies between the pre-trained KG and unseen KGs. 

The primary challenge in generalizing to new entities, relations, and even different KGs lies in how to represent such unseen data. Some methods [1–4] aggregate query-conditioned relational structures to represent entities. They can conduct inductive reasoning over unseen entities using these relative entity representations without the need of pre-trained entity embeddings. However, these methods cannot reason over unseen relations. To resolve this issue, some recent methods [5, 6] develop relative relation representations. They model relation interactions using a query-conditioned relation graph, where each node represents a relation and an edge indicates that the linked two relations share a subject or object entity in the KG. They conduct message passing on the query-conditioned relation graph to represent relations. 

However, the relation graph only describes the connectivity of relations in the KG, with less attention to the local context of the entity and relation in a query. As a result, these methods usually fail to 

> _∗_ Corresponding author 

38th Conference on Neural Information Processing Systems (NeurIPS 2024). 

generate discriminative relation representations. For example, to infer the query relation `parentOf` , the most relevant relation is `coupleOf` . While in the KG, since every student has parents and most teachers are parents, the relation graph would also contain edges “ `parentOf` _⇒_ `teach` ” and “ `teach` _⇒_ `parentOf` ”. The relation `teach` appears as noise in representing `parentOf` , which may mislead the model, resulting in prediction failures. This inspires us to capture the local contexts and highlight the important relations relevant to queries, rather than relying on a global relation graph. 

In this paper, we propose a novel KG reasoning foundation model with in-context learning, namely KG-ICL. In-context learning is a method that allows pre-trained models to learn tasks based on only a few examples without updating model parameters. The extraordinary success of in-context learning in language modeling [7] hinges on three crucial fundamentals: prompt design, unified tokenization, as well as contextual understanding and utilization. 

The art of prompt design lies in highlighting task-critical information. We construct a prompt graph to model query-related contexts, which starts with an example fact about the query relation, i.e., ( `subject` , `query relation` , `object` ). We consider two types of contexts as prompts. The first is entity context, which includes the neighboring entities of the example subject and object. The second is relation context, which considers relational paths between the subject and object entities. Thus, the node set of our prompt graph includes the neighbors of the example subject and object, as well as the entities within the paths connecting the subject and object in the KG. We utilize the induced subgraph of these entities as a prompt graph. 

Then, we design a unified tokenizer that is applicable to various prompt graphs. The key challenge is that the entities and relations usually vary across different KGs [8, 9], and this issue extends to prompt graphs as well. Conventional KG reasoning models [10–14] merely learn an individual embedding for each entity or relation, resulting in the inability to reason over unseen KGs. We extend the entity labeling method of GraIL [1] to relations, proposing a unified tokenizer for various prompt graphs. Given a query relation and its prompt graph, we first group the involved entities based on the lengths of their shortest path to the example subject and object entities. Similarly, we categorize relations into two classes depending on whether they represent query relations. Finally, the entities or relations in the same group will be mapped to the same token. As a result, prompt graphs from different KGs are described in “the same language”. 

Given the above prompt graph and unified tokenizer, we propose two message passing neural networks as the prompt encoder and KG reasoner, respectively. The input of the prompt encoder is the prompt graph and the learnable token representations. At each layer of prompt encoding, we introduce an entity-centric and a relation-centric aggregation. Notably, in relation-centric aggregation, we treat relations as special nodes and update their representations by aggregating messages from facts containing them. After prompt encoding, we read the relation representations from the prompt graphs to support KG encoding. At the beginning of KG encoding, we initialize the relation representations in the KG as the prompt relation representations. As for entities, we initialize the subject entity as the query relation representation, and other entities are initialized as zero vectors. After performing message passing over the KG, we score all entities based on the output entity representations. 

We conduct extensive experiments on 43 datasets to validate the effectiveness of our model. The experimental results indicate that our model not only possesses universal reasoning capabilities across diverse KGs but also outperforms supervised and pre-training models. Moreover, we observe that the proposed model exhibits robustness and high efficiency in utilizing examples. 

In summary, our main contributions are listed below: 

- Our key contribution is an in-context KG reasoning foundation model. It prompts the pre-trained model to engage in relational reasoning over diverse KGs. 

- We propose a prompt graph as context to support in-context learning. It consists of an example fact about the query relation and its relevant subgraphs and paths. We also employ a unified tokenizer to map entities and relations in prompt graphs to predefined tokens. 

- Given a prompt graph with token representations, we propose two message passing networks for prompt graph encoding and KG reasoning. The foundation model can be further finetuned on specific KGs to obtain improved performance. 

- We conduct extensive experiments on 43 KGs in both transductive and inductive settings to demonstrate the universal reasoning capability of our model. 

2 

## **2 Related Work** 

**KG reasoning.** KG reasoning primarily involves three settings: transductive, inductive, and fullyinductive. Early studies [10–14] focus mainly on the transductive setting, assuming that KGs are static. Real-world KGs are dynamic, inspiring the development of inductive models [1–4, 15–23] that allows for emerging entities. In the fully-inductive setting [5, 24–26], both unseen entities and relations can emerge in the query facts. This setting remains limited to the same KG. In contrast, our in-context learning and KG foundation model seek to break down the barriers imposed by these settings and achieve universal reasoning capabilities. 

**Prompt and in-context learning in graph pre-training.** Our work is also related to graph prompt learning and graph in-context learning. Inspired by the success of pre-training models in NLP [27] and computer vision [28], some graph pre-training models [29–33] have been proposed. These models follow the paradigm of “pre-train and finetune”, where a model is initially pre-trained and then finetuned for the target task. The work [34] further develops a KG pre-training model. Consequently, recent work [8, 35–45] has shifted focus to the “pre-train, prompt, and finetune” paradigm. The relation graph of the KG pre-training model [6] can also be seen as a special prompt. This paradigm leverages task prompts to enhance the knowledge transfer and generalization abilities of pre-trained models. Inspired by the recent success of large language models like GPT [7], recent work uses in-context learning to avoid finetuning. It imparts general capabilities to pre-trained models with just a few examples. PRODIGY [46] introduces an in-context learning-based model to handle various classification tasks on graphs. While it can perform relation classification, it is not suitable for KG reasoning with a massive number of candidate entities. 

We discuss more related work in Appendix D. 

## **3 Problem Definition** 

**KG Reasoning.** We define a KG as _K_ = ( _E, R, T_ ), where _E_ , _R_ , and _T_ denote the sets of entities, relations, and facts, respectively. A fact ( _s, r, o_ ) _∈T_ consists of a subject entity _s ∈E_ , a relation _r ∈R_ , and an object entity _o ∈E_ . Given a KG and a query fact in the form of ( _s, q,_ ?), the reasoning task is to predict the missing entity from _E_ . We refer to the relation _q_ as a query relation. 

In practice, we follow the convention [10] to introduce inverse relations. For each relation _r ∈R_ , we add its inverse relation _r[−]_ into the relation set and add the reverse fact ( _o, r[−] , s_ ) into the fact set. 

**In-Context KG Reasoning.** In in-context reasoning, a model is pre-trained using a set of source KGs, denoted by _{K_ 1 _, . . . , Kn}_ . After pre-training, the model conducts reasoning on emerging KGs based on only a few related examples without updating model parameters. Each pre-training or reasoning query is prompted with some relevant examples as context. 

The prompt is crucial for in-context learning. For each query relation _q_ , we first randomly sample some of its facts, e.g., _c_ = ( _u, q, v_ ) _∈T_ . Next, we extract a subgraph _Pc_ = ( _E_ pmt _, R_ pmt _, T_ pmt) from the KG for each example fact to construct a prompt graph. In the following, we provide a broad definition of prompt graphs, allowing for a broad design space: 

**Prompt Graph.** Given an example fact _c_ = ( _u, q, v_ ) in a KG _K_ = ( _E, R, T_ ), where _c ∈T_ , we define its prompt graph _Pc_ = ( _E_ pmt _⊆E, R_ pmt _⊆R, T_ pmt _⊆T_ ) as a subgraph of _K_ , and _c ∈T_ pmt. 

To encode prompt graphs, we extend the KG-independent entity labeling [1] to relations and propose a unified tokenizer, which maps entities and relations from different KGs to unified tokens: 

**Unified Tokenizer.** The unified tokenizer is a many-to-one mapping function. It maps entities and relations of different prompt graphs to the predefined tokens. Specifically, it maps each entity based on the length of its shortest paths to the subject and object entities of the example fact, i.e., tokenize( _e_ ) _←_ [dist( _u, e_ ) _,_ dist( _v, e_ )], where dist( _·_ ) is the length of the shortest path between two entities. It maps each relation to the tokens by whether it is the same as the query relation. That is, tokenize( _r_ ) _←_ [same( _r, q_ )], where same( _r, q_ ) = 1 if _r_ is the same as _q_ , otherwise same( _r, q_ ) = 0. 

In Section 4.2, we assign a learnable representation for each token. 

3 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_A_Prompt-Based_Knowledge_Graph_Foundation_Model_for_Universal_In-Context_Reasoning_images/nips_2024_A_Prompt-Based_Knowledge_Graph_Foundation_Model_for_Universal_In-Context_Reasoning.pdf-0004-00.png)


**----- Start of picture text -----**<br>
Knowledge Graph<br>Relation Scores<br>embeddings<br>(felix_pie, player Query: message passing for prompt encoding Readout message passing for KG reasoning embeddingsFinal entity<br>in league, ?) Concatenate Add & Norm<br>AGG<br>Add & Norm Add & Norm<br>extractionsubgraph tokenizerunified  AGG AGG Add & Norm Message<br>player in league [1] NBA [1, 0] L × Query-awareMessage Query-awareMessage × L N × embeddingsRelation Linear Query-awareNext-hop attention  × N<br>attention  attention  neighbors<br>Lakers [1, 1] input graph input features input features input graph<br>[1] [1, 0] [0][1] [ s ] ( s ,  q , ?)<br>[0, 1] [others]<br>[1, 1] [1, 0] Entity embeddings<br>BryantKobe  [1, 1]teammate [0] GasolPau  [0, 1] [1, 1] [0] [0, 1] [1, 1][1, 2][2, 1]Token embeddings embeddingsRelation<br>(A) Prompt graph generation (B) Prompt graph encoding (C) Knowledge graph reasoning<br>[1] Entity encoding Relation encoding Relation encoding Entity encoding<br>[1]<br>player in league<br>**----- End of picture text -----**<br>


Figure 1: Overview of the in-context KG reasoning foundation model. (A) Given the query and KG, we extract prompt graphs as context for the query relation “player in league”. The entities and relations in the prompt graphs are mapped to the unified tokens. (B) We employ a message passing neural network to encode the prompt graph and readout the relation representations as the prompts. (C) Then we use the prompts to initialize the representations of entities and relations in the KG. After KG encoding, we score the candidate entities according to their embeddings in the last layer. 

## **4 In-context Reasoning over KGs** 

The overview of the proposed model is shown in Figure 1. Given a KG and a query, we first generate prompt graphs for the query relation. Then, we use an encoding module to encode the prompt graphs and readout prompts. Finally, we incorporate the prompts into the KG reasoning process. 

## **4.1 Prompt Graph Generation** 

The prompt graph defined in Section 3 allows for a broad design space. In this section, we introduce a specific method for generating prompt graphs. We primarily address two challenges: (i) How to make the prompt graph general for diverse KGs? (ii) How to provide valuable prompts to enhance reasoning? We propose a prompt graph generation pipeline to address these challenges. It involves two steps: example sampling and prompt graph extraction. 

**Example sampling.** For a query relation _q_ , we first randomly sample _M_ example facts as follows: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_A_Prompt-Based_Knowledge_Graph_Foundation_Model_for_Universal_In-Context_Reasoning_images/nips_2024_A_Prompt-Based_Knowledge_Graph_Foundation_Model_for_Universal_In-Context_Reasoning.pdf-0004-07.png)


where _Nq_ = _{_ ( _u, r, v_ ) _| r_ = _q ∧_ ( _u, r, v_ ) _∈T }_ and _ci_ = ( _u, q, v_ ) is a _q_ -specific example fact. 

**Prompt graph extraction.** The key point of the prompt graph design is highlighting information crucial for query relation-specific reasoning. The example fact consists of a subject entity, an object entity, and the query relation between them. To depict the example subject and object entities, we draw inspiration from the research on prompt-based graph model [35, 46] to use neighboring nodes centered around the central node to construct prompt graphs. To abstract the semantics of query relation, we include the paths between example subject and entities, considering the success of logical rules in KG reasoning [47–50]. The body of the rules involves paths between the subject and object entities. Therefore, given an example fact _c_ = ( _u, q, v_ ) _∈Sq_ and a KG _K_ = _{E, R, T }_ , we include the neighboring entities of _u_ and _v_ and the _k_ -hop paths between _u_ and _v_ in the prompt graph: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_A_Prompt-Based_Knowledge_Graph_Foundation_Model_for_Universal_In-Context_Reasoning_images/nips_2024_A_Prompt-Based_Knowledge_Graph_Foundation_Model_for_Universal_In-Context_Reasoning.pdf-0004-10.png)


where _k_ is a hyperparameter denoting the maximum value of dist( _x, u_ ) + dist( _x, v_ ). As we have added reverse facts, _E_ pmt includes all 1-hop neighbors. Next, we extract the facts and relations among them, i.e., _T_ pmt = �( _s, r, o_ ) _| s ∈E_ pmt _∧ o ∈E_ pmt _∧_ ( _s, r, o_ ) _∈T_ � and _R_ pmt = � _r | ∃_ ( _s, r, o_ ) _∈T_ pmt�. 

4 

## **4.2 Prompt Encoding** 

In this section, we design a message passing neural network for prompt encoding. It comprises three sub-modules: token representation, message passing, and readout. We begin by initializing the token representations of entities and relations in the given prompt graph. Subsequently, a multi-layer message passing neural network is employed to encode the prompt graph. Finally, we introduce a readout sub-module to obtain the prompt representation. 

**Token representations.** We assign each token a learnable vector representation. Specifically, according to Equation (2), the tokens for entities satisfy _i_ + _j ≤ k_ , 0 _≤ i ≤ k −_ 1 and 0 _≤ j ≤ k −_ 1. Therefore, we set a representation matrix **T** _∈_ R[(][(] _[k]_[+][1][)(] 2 _[k]_[+][2][)] _−_ 2( _k−_ 1)) _×d_ for entity tokens, where ( _k_ +1)(2 _k_ +2) _−_ 2( _k −_ 1) denotes the total number of entity tokens. As for relations, the representation of token [ _z_ ] is initialized as **q**[token] _· z_ , where **q**[token] _∈_ R[1] _[×][d]_ is a learnable representation. We denote the input representation matrix of entities and relations for the prompt graph as **H**[(0)] E and **H**[(0)] R[,] respectively. 

**Message passing for prompt graph.** Then, we employ an _L_ -layers message passing neural network, which incorporates two types of aggregation: an entity-centric aggregation and a relation-centric aggregation. In each layer, we first update the entity representations as follows: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_A_Prompt-Based_Knowledge_Graph_Foundation_Model_for_Universal_In-Context_Reasoning_images/nips_2024_A_Prompt-Based_Knowledge_Graph_Foundation_Model_for_Universal_In-Context_Reasoning.pdf-0005-04.png)


where _Ne ⊆T_ pmt is the set of facts containing the entity _e_ , and _q_ is the query relation of this prompt graph. Then we update the relation representations using the updated entity representations and the relation representations from the previous layer: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_A_Prompt-Based_Knowledge_Graph_Foundation_Model_for_Universal_In-Context_Reasoning_images/nips_2024_A_Prompt-Based_Knowledge_Graph_Foundation_Model_for_Universal_In-Context_Reasoning.pdf-0005-06.png)


where _Nr ⊆T_ pmt is the set of facts containing the relation _r_ . Under this message passing framework, we present two specific aggregation and message functions in Appendix A.1. 

**Readout.** After _L_ -layers message passing on the prompt graph _P_ , we obtain the prompt as follows: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_A_Prompt-Based_Knowledge_Graph_Foundation_Model_for_Universal_In-Context_Reasoning_images/nips_2024_A_Prompt-Based_Knowledge_Graph_Foundation_Model_for_Universal_In-Context_Reasoning.pdf-0005-09.png)


where **W** Readout _∈_ R _[d][×][Ld]_ is a learnable weight matrix. Note that the relations in different prompt graphs may vary. **H** ˆ _P ∈_ R _[|R|×][d]_ , ensuring that the shapes of every representation matrix are the same.We fill in the relations not present in the prompt graph with zero vectors to obtainFinally, we use mean-pooling to aggregate the information from multiple prompt graphs as follows: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_A_Prompt-Based_Knowledge_Graph_Foundation_Model_for_Universal_In-Context_Reasoning_images/nips_2024_A_Prompt-Based_Knowledge_Graph_Foundation_Model_for_Universal_In-Context_Reasoning.pdf-0005-11.png)


where **H** pmt _∈_ R _[|R|×][d]_ is the prompt relation representation matrix, _Sq_ is the set of example facts of the query relation _q_ , and _Pc_ is the prompt graph corresponding to the example fact _c_ . In practice, we parallel encode these prompt graphs to ensure efficiency. 

## **4.3 In-Context KG Encoding and Reasoning** 

Based on the prompt encoding, we conduct reasoning on KGs. To achieve a KG-independent encoding, we draw inspiration from the conditional message passing neural network [3, 4, 20–22] to present a novel KG reasoning module. It separately encodes entities based on the query, rather than mapping them to specific embeddings, offering us an opportunity for knowledge transfer across diverse KGs. It comprises three sub-modules: initialization, KG encoding, and reasoning. 

**Initialization.** The input relation representations in the KG are initialized as the prompt relation embeddings, i.e., **V** R[(0)] = **H** pmt. As for entity representations, given a query fact ( _s, q, x_ ), the representation of _s_ is initialized as the representation of the query relation, i.e., **s** = **q** . Other entities are represented by zero vectors. We denote the input representation matrix of entities in KG as **V** E[(0)][.] 

5 

**Message passing for KG.** Here we employ an _N_ -layers message passing neural network to aggregate multi-hop information. At each layer, we first update relation representations as follows: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_A_Prompt-Based_Knowledge_Graph_Foundation_Model_for_Universal_In-Context_Reasoning_images/nips_2024_A_Prompt-Based_Knowledge_Graph_Foundation_Model_for_Universal_In-Context_Reasoning.pdf-0006-01.png)


where LN denotes the layer normalization operation, and **W** R[(] _[l]_[)] _[∈]_[R] _[d][×][d]_[is a learnable weight matrix.] 

Then, we update entity representations based on the updated relation representations. Some studies [3, 20] have shown that the distance-based inductive bias is crucial for KG reasoning. Inspired by this, we introduce a hop-by-hop message passing neural network to update entity representations, starting from the subject entity and expanding one-hop neighbors at each layer: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_A_Prompt-Based_Knowledge_Graph_Foundation_Model_for_Universal_In-Context_Reasoning_images/nips_2024_A_Prompt-Based_Knowledge_Graph_Foundation_Model_for_Universal_In-Context_Reasoning.pdf-0006-04.png)


where _q_ is the query relation, _L_[(] _[l]_[)] is the set of entities in _l_ -hop neighbors of _s_ , and _L_[(0)] = _{s}_ , _L_[(] _[l]_[+1)] = _L_[(] _[l]_[)] _∪_ � _e | ∃_ ( _x, y, e_ ) _∈T ∧ x ∈L_[(] _[l]_[)][�] . Under this message passing framework, we present a specific message passing neural network for KG encoding in Appendix A.2. 

**Reasoning.** Finally, we read the representations of candidate entities and assign scores to them, i.e., _f_ ( _s, q, e_ ) = **W** score **e**[(] _s,q[N]_[)][,where] **[ e]**[(] _s,q[N]_[)] _[∈]_ **[V]** E[(] _[N]_[)] is the output representation of the entity _e_ , and **W** score _∈_ R[1] _[×][d]_ is a weight matrix. Note that the message passing neural network we employ for KG encoding is conditioned on specific queries of the form ( _s, q,_ ?), meaning the output representations **V** E[(] _[N]_[)] have taken into account the conditional messages related to both _s_ and _q_ . In addition, it encodes only the _N_ -hop neighbor entities of the subject entity. We assign a score of 0 to other entities. 

## **4.4 Pre-training Objective** 

Given a set of source KGs _C_ = _{K_ 0 _, . . . , Kn}_ , where _Ki_ = ( _Ei, Ri, Ti_ ), we pre-train the model using the multi-class log-loss [51]: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_A_Prompt-Based_Knowledge_Graph_Foundation_Model_for_Universal_In-Context_Reasoning_images/nips_2024_A_Prompt-Based_Knowledge_Graph_Foundation_Model_for_Universal_In-Context_Reasoning.pdf-0006-09.png)


where _f_ is the score function mentioned above. Minimizing Equation (9) enlarges scores of positive facts while reducing scores of all negatives that replace the correct object entity with another entity from the KG. We describe our reasoning process in Algorithm 1 of Appendix B. 

## **5 Experiments** 

## **5.1 Settings** 

**Datasets and implementations.** We conduct experiments on 43 datasets of various schemata and sizes to evaluate our model. The datasets fall into three groups: (i) 14 inductive datasets, including 12 datasets in GraIL [1] and 2 datasets in ILPC 2022 [52], (ii) 13 fully-inductive datasets in [5], and (iii) 16 transductive datasets, including FB15k-237 [53], WN18RR [12], NELL-995, [54], YAGO3-10 [55], 3 datasets in CoDEx [56], 5 datasets in [57], AristoV4 [58], DBpedia100k [59], ConceptNet100k [60], and Hetionet [61]. The statistics of datasets are in Appendix F. We pre-train our model on three datasets, i.e., FB V1 [1] with 180 relations, NELL V1 [1] with 14 relations, and CoDEx-s [56] with 42 relations, to capture various relational structures in KGs and prompt graphs. The implementation details are in Appendix C. We assess the impact of pre-training KGs in Appendix E.1. 

**Baselines.** We compare KG-ICL with two categories of baseline models: (i) Supervised state-of-theart (abbr. supervised SOTA), which refers to the models achieving the best MRR result on specific target datasets. We list the supervised SOTA model on each dataset in Appendix F. (ii) Pre-training model. ULTRA [6] is a KG pre-training model, consisting of pre-training and finetuning versions. To investigate the ability of finetuning on target datasets to yield improvement of the proposed model, we also introduce two versions of our model: “KG-ICL pre-train” and “KG-ICL finetune”. After pre-training, the finetuning model undergoes finetuning for 5 epochs on the target dataset using the same configuration as the pre-training. The main focus of this work is on in-context learning without 

6 

the need for finetuning. Therefore, we report the results of both versions of our model in Section 5.2 and Appendix G, and use the pre-training version in further analyses. 

**Evaluation protocol.** For each sample ( _s, r, o_ ) in the test set, we generate two query facts ( _s, r,_ ?) and ( _o, r[−] ,_ ?), where _r[−]_ is the inverse relation of _r_ . As mentioned in Section 3, we add inverse relations and facts before conducting reasoning. The pre-training model considers all entities in the entity set as candidates, scoring and ranking them for each query fact. Following the convention [62–64], we employ two standard evaluation metrics: mean reciprocal rank (MRR) and Hits@10 (abbr. H@10). Higher scores of both metrics indicate superior performance. We follow the widely-used filtered setting [10], i.e., wherein all known true entities are removed from the candidate set, except for the target entity. Due to the abundance of datasets, we categorize them and report the scores in terms of the groups, such as the inductive dataset group. This involves calculating scores for each dataset individually and computing the average of all scores within each group. 

Table 1: KG reasoning results in various settings. 

|Models|Inductive<br>(14 KGs)<br>MRR<br>H@10|Fully-inductive<br>(13 KGs)<br>MRR<br>H@10|Transductive<br>(16 KGs)<br>MRR<br>H@10|Average<br>(43 KGs)<br>MRR<br>H@10|
|---|---|---|---|---|
|Supervised SOTA<br>ULTRA pre-train<br>ULTRA fnetune|0.466<br>0.607<br>0.513<br>0.664<br>0.528<br>0.684|0.210<br>0.347<br>0.352<br>0.536<br>0.350<br>0.542|0.365<br>0.511<br>0.329<br>0.479<br>0.384<br>0.548|0.351<br>0.493<br>0.396<br>0.557<br>0.421<br>0.590|
|KG-ICL pre-train<br>KG-ICL fnetune|0.554<br>0.707<br>**0.582**<br>**0.727**|0.439<br>0.635<br>**0.449**<br>**0.647**|0.346<br>0.493<br>**0.397**<br>**0.554**|0.442<br>0.606<br>**0.473**<br>**0.638**|



## **5.2 Main Results** 

We divide the datasets into three groups according to their reasoning settings, i.e., inductive, fullyinductive and transductive, and report the average results for each group as well as the overall average results in Table 1. ULTRA employs three different source KGs distinct from ours. For ease of presentation, we incorporate the source KGs into their respective groups rather than listing them separately. We can observe that our “KG-ICL pre-train” outperforms both versions of ULTRA on inductive and fully-inductive datasets, with further enhancements achieved by our “KG-ICL finetune”, resulting in the best performance across all groups. We report detailed results of each dataset and more analyses in Figure 2 and Appendix G. 

**Inductive datasets.** The inductive setting aims to complete facts involving unseen entities. In each inductive dataset, at least two graphs are included: one for training and the other for evaluation. The evaluation graph incorporates new entities not seen in the training graph. The MRR results are depicted in Figure 2(a). We observe that our “KG-ICL pre-train” outperforms supervised SOTA models on 10 datasets and surpasses the “ULTRA pre-train” model on 11 datasets. The “KG-ICL finetune” achieves further improvements in scores compared to the pre-trained version, achieving the best results on 13 datasets, and yielding the best average results. 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_A_Prompt-Based_Knowledge_Graph_Foundation_Model_for_Universal_In-Context_Reasoning_images/nips_2024_A_Prompt-Based_Knowledge_Graph_Foundation_Model_for_Universal_In-Context_Reasoning.pdf-0007-07.png)


**----- Start of picture text -----**<br>
0.8<br>(a) inductive<br>0.4<br>0<br>0.6 (b) fully-inductive<br>0.4<br>0.2<br>0<br>0.6<br>(c) transductive<br>0.4<br>0.2<br>0<br>Supervised SOTA ULTRA pre-train ULTRA finetune KG-ICL pre-train KG-ICL finetune<br>MRR<br>MRR<br>MRR<br>**----- End of picture text -----**<br>


Figure 2: MRR results on various KGs. 

7 

**Fully-inductive datasets.** In fully-inductive datasets, the evaluation graphs not only include new entities unseen during training but also introduce new relations. The MRR results are shown in Figure 2(b). This setting poses a significant challenge with the introduction of unseen relations, leading to relatively lower scores for supervised SOTA models. However, both versions of KG-ICL, aided by prompt graphs, demonstrate the ability to extract valuable information and adaptively perform reasoning for unseen query relations. Consequently, they consistently outperform supervised SOTA models across all 13 datasets and exhibit a notable improvement over the previous pre-training model, ULTRA, on all datasets. 

**Transductive datasets.** In the transductive setting, where entities and relations in the test set are encountered during training, the MRR results are presented in Figure 2(c). It is evident that, in comparison to the first two settings, the advantage of the proposed model over the supervised SOTA model is somewhat attenuated. The reason is that the supervised signals on transductive datasets directly target entities and relations in the test set, allowing supervised models to effectively learn representations and achieve high performance. Nonetheless, “KG-ICL pre-train” maintains its superiority over the supervised SOTA models on 7 datasets. “KG-ICL finetune” achieves the best average MRR score. We report detailed results of each dataset and more analyses in Appendix G. 

## **5.3 Further Analyses** 

In this section, we conduct experiments to devise the impact of each module. In the appendix, we include more experimental analyses about the pre-training datasets (Appendix E.1), complexity analyses of preprocessing (Appendix E.2), and the variant incorporating other message passing layer (Appendix E.3). 

Table 2: Ablation study results in various settings. 

|Models|Inductive<br>(14 KGs)<br>MRR<br>H@10|Fully-inductive<br>(13 KGs)<br>MRR<br>H@10|Transductive<br>(16 KGs)<br>MRR<br>H@10<br>|Average<br>(43 KGs)<br>MRR<br>H@10|
|---|---|---|---|---|
|Intact model<br>w/o prompt graph<br>w/o unifed tokenizer<br>w/ GraIL’s labeling|0.554<br>0.707<br>0.219<br>0.420<br>0.511<br>0.660<br>0.531<br>0.704|0.439<br>0.635<br>0.105<br>0.228<br>0.419<br>0.617<br>0.434<br>0.634|0.346<br>0.493<br><br>0.076<br>0.143<br><br>0.296<br>0.453<br><br>0.343<br>0.492<br>|0.442<br>0.606<br>0.132<br>0.259<br>0.403<br>0.570<br>0.431<br>0.604|



**Ablation study.** We hereby conduct an ablation study to evaluate the impact of each module. Specifically, we construct three variants by removing certain modules: “w/o prompt graph”, “w/o unified tokenizer”, and “w/ GraIL’s labeling”. “w/o prompt graph” removes the prompt graph generation and encoding module. Its prompt representations are initialized with the Xavier normal initialization. “w/o unified tokenizer” eliminates the unified tokenizer and initializes the input representations of entities and relations in prompt graphs with the Xavier normal initialization. “w/ GraIL’s labeling” replaces our token representation with GraIL’s one-hot labeling [1]. The results are presented in Table 2. We observe a significant performance decline in the “w/o prompt graph” variant compared to the intact model, highlighting the necessity of the prompt graph as a knowledge transfer bridge. The “w/o unified tokenizer” variant also exhibits a performance drop, indicating the importance of the unified tokenizer for in-context learning. The “w/ GraIL’s labeling” can also achieve promising results, although it still falls behind our intact model, which shows the generalization ability of our model and the effectiveness of the token representation. 

**Example efficiency.** The efficiency of utilizing examples is crucial for in-context learning. To determine the optimal number of example prompt graphs needed to support in-context reasoning, we conduct experiments under the settings of 1-shot, 3-shot, 5-shot, 10-shot, and 20-shot. The results are illustrated in Figure 3. Overall, the results remain consistent with a slight fluctuation across the range from 1-shot to 20-shot, which shows that the proposed model is robust to the changes in the num- 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_A_Prompt-Based_Knowledge_Graph_Foundation_Model_for_Universal_In-Context_Reasoning_images/nips_2024_A_Prompt-Based_Knowledge_Graph_Foundation_Model_for_Universal_In-Context_Reasoning.pdf-0008-08.png)


**----- Start of picture text -----**<br>
0.6<br>0.4<br>inductive fully-inductive transductive<br>0.2<br>1 3 5 10 20<br>Number of examples<br>MRR<br>**----- End of picture text -----**<br>


Figure 3: MRR with different numbers of examples. 

8 

ber of prompt graphs. The reason for the slight performance fluctuation is that more examples may also introduce more noise. Besides, multiple examples tend to share popular reasoning patterns, so only one or three prompt graphs can still suffice. For overall performance, we choose _M_ = 5 in the main experiment. These results suggest that KG-ICL can unleash universal reasoning capabilities with only a few examples, showcasing high efficiency in example utilization. 

**Prompt graph variants.** The core of a prompt graph lies in highlighting essential information for reasoning. In this paper, we propose a prompt graph generation process that combines paths and neighbors of the subject and object entities. To further explore the critical components for reasoning, we introduce several variants, with the proposed model referred to as “neighbor & 3-hop path”. We present four variants by altering the entity sampling method: the “neighbor” variant, considering only neighbors of the subject and object entities, and the “ _x_ -hop path” variant, considering _x_ -hop paths between the entity and object entities, where _x ∈{_ 1 _,_ 2 _,_ 3 _}_ . The results in Table 3 demonstrate the impact of the prompt graph on reasoning. We observe that both paths and neighbors of the subject and object entities are crucial for reasoning. The optimal performance is achieved when combining both components. The variants considering only paths within one or two hops exhibit poor performance, indicating insufficient support for effective reasoning. 

Table 3: MRR results on diverse prompt graphs. 

|Models|Inductive<br>(14 KGs)<br>MRR<br>H@10|Fully-inductive<br>(13 KGs)<br>MRR<br>H@10|Transductive<br>(16 KGs)<br>MRR<br>H@10|Average<br>(43 KGs)<br>MRR<br>H@10|
|---|---|---|---|---|
|Neighbor & 3-hop path<br>Neighbor<br>1-hop path<br>2-hop path<br>3-hop path|0.554<br>0.707<br>0.552<br>0.702<br>0.208<br>0.449<br>0.256<br>0.419<br>0.544<br>0.697|0.439<br>0.635<br>0.429<br>0.628<br>0.145<br>0.314<br>0.137<br>0.285<br>0.409<br>0.601|0.346<br>0.493<br>0.311<br>0.459<br>0.112<br>0.216<br>0.125<br>0.235<br>0.294<br>0.464|0.442<br>0.606<br>0.425<br>0.590<br>0.153<br>0.322<br>0.171<br>0.310<br>0.410<br>0.582|



**Robustness to low-resource relations.** We conduct experiments to assess the robustness of the proposed model to low-resource relations with limited training samples. Specifically, we choose the supervised model RED-GNN [3] as a baseline and conduct experiments on 12 widely used inductive datasets [1] and 3 transductive datasets (FB15k-237, WN18RR, and NELL-995). We organize relations within each dataset group into six subgroups based on the number of training samples. Subsequently, we compute the MRR score for each relation and calculate the average score within each subgroup. The results, as illustrated in Figure 4, reveal a gradual decline in the performance of RED-GNN, as the number of training samples decreases. In contrast, our model exhibits robustness across a spectrum of relations. The results suggest that our model maintains effective performance even under resource constraints. This can be attributed to our model of avoiding the representation of each relation independently with specific embeddings. We employ a universal prompt graph and a unified tokenizer for the relation representation, fostering cross-relation knowledge transfer and achieving superior robustness. 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_A_Prompt-Based_Knowledge_Graph_Foundation_Model_for_Universal_In-Context_Reasoning_images/nips_2024_A_Prompt-Based_Knowledge_Graph_Foundation_Model_for_Universal_In-Context_Reasoning.pdf-0009-05.png)


**----- Start of picture text -----**<br>
>10000 0.1% >10000 1.6%<br>[1000, 10000) 1.7% [1000, 10000) 21.0%<br>[100, 1000) 16.8% [100, 1000) 74.3%<br>[10, 100) 47.4% [10, 100) 3.1%<br>[5, 10) 14.2% [5, 10) 0.0%<br>[1, 5) 19.9% [1, 5) 0.0%<br>0 0.2 0.4 0.6 0.8 0 0.2 0.4 0.6<br>MRR score MRR score<br>RED-GNN Ours RED-GNN Ours<br>(a) inductive (b) transductive<br>Number of training samples Number of training samples<br>**----- End of picture text -----**<br>


Figure 4: Average MRR results of relation subgroups. Relations in the inductive and transductive dataset groups are divided into 6 subgroups based on the number of training samples, and the results represent the average scores for the relations within their respective subgroups. The percentage on the right side of each data bar indicates the proportion of relations in that subgroup to the total number of relations in their respective groups. 

9 

**Case study.** We conduct a case study to investigate the reasons behind the proposed model’s generalizability across different KGs. Specifically, we select two similar and easily interpretable query relations, “teamSport” and “film/language” from NELL-995 and FB15k-237, respectively. We extract several relation paths from their prompt graphs, forming two similar subgraphs. Subsequently, we execute the model and save the prompt representations for both query relations. Finally, we compute the cosine similarities between relations in the two prompt graphs and visualize the heatmap in Figure 5. We observe that the values along the diagonal of the heatmap are notably high, indicating that different relations with similar roles in the reasoning of the two query relations have correspondingly similar model encodings. This suggests that the prompt representations effectively capture the roles of various relations in reasoning, thereby improving transferability across different KGs. 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_A_Prompt-Based_Knowledge_Graph_Foundation_Model_for_Universal_In-Context_Reasoning_images/nips_2024_A_Prompt-Based_Knowledge_Graph_Foundation_Model_for_Universal_In-Context_Reasoning.pdf-0010-01.png)


**----- Start of picture text -----**<br>
KG:  NELL-995<br>query relation:  teamSport<br>B. teamInLeague_inv<br>subject  object<br>entity entity<br>KG:  FB15k-237<br>query relation:  film/language<br>B* film/country_inv<br>subject  object<br>entity entity<br>**----- End of picture text -----**<br>


Figure 5: Case study on prompt graphs. The left side shows some relation paths extracted from two prompt graphs of NELL-995 and FB15k-237. The right side depicts a heatmap where cosine similarities between relations in two prompt graphs are pairwise computed. 

## **6 Conclusions** 

This paper introduces a KG foundation model with in-context learning to improve the effectiveness and transferability of KG reasoning. Specifically, we introduce a prompt graph and a unified tokenizer as the bridge to knowledge transfer between different KGs. Following that, we propose a prompt graph generation module, a prompt encoding module, and a KG reasoning module to achieve incontext learning. We evaluate the in-context reasoning ability on 43 different KGs in both transductive and inductive settings. Extensive experimental results validate our model’s universal reasoning ability across diverse KGs. In future work, we plan to explore the application of in-context reasoning in more challenging scenarios, such as personal KGs that are dynamic and diverse. This is motivated by the demonstrated robustness of our KG-ICL in Section 5.3. Additionally, investigating how to extend in-context reasoning to more knowledge-driven applications, e.g., recommender systems and question answering, is another promising avenue for future research. 

## **Acknowledgments** 

We thank the anonymous reviewers for their valuable comments. This work was funded by National Natural Science Foundation of China (Nos. 62272219 and 62406136), Postdoctoral Fellowship Program of CPSF (No. GZC20240685), and CCF-Tencent Rhino-Bird Open Research Fund. 

10 

## **References** 

- [1] Komal Teru, Etienne Denis, and Will Hamilton. Inductive relation prediction by subgraph reasoning. In _ICML_ , pages 9448–9457, Virtual, 2020. PMLR. 

- [2] Sijie Mai, Shuangjia Zheng, Yuedong Yang, and Haifeng Hu. Communicative message passing for inductive relation reasoning. In _AAAI_ , pages 4294–4302, Virtual, 2021. AAAI Press. 

- [3] Yongqi Zhang and Quanming Yao. Knowledge graph reasoning with relational digraph. In _WWW_ , pages 912–924, Lyon, France, 2022. ACM. 

- [4] Zhaocheng Zhu, Zuobai Zhang, Louis-Pascal A. C. Xhonneux, and Jian Tang. Neural bellmanford networks: A general graph neural network framework for link prediction. In _NeurIPS_ , pages 29476–29490, Virtual, 2021. PMLR. 

- [5] Jaejun Lee, Chanyoung Chung, and Joyce Jiyoung Whang. Ingram: Inductive knowledge graph embedding via relation graphs. In _ICML_ , volume 202, pages 18796–18809, Honolulu, HI, USA, 2023. PMLR. 

- [6] Mikhail Galkin, Xinyu Yuan, Hesham Mostafa, Jian Tang, and Zhaocheng Zhu. Towards foundation models for knowledge graph reasoning. In _ICLR_ , Vienna, Austria, 2024. OpenReview.net. 

- [7] Tom B. Brown, Benjamin Mann, Nick Ryder, Melanie Subbiah, Jared Kaplan, Prafulla Dhariwal, Arvind Neelakantan, Pranav Shyam, Girish Sastry, Amanda Askell, Sandhini Agarwal, Ariel Herbert-Voss, Gretchen Krueger, Tom Henighan, Rewon Child, Aditya Ramesh, Daniel M. Ziegler, Jeffrey Wu, Clemens Winter, Christopher Hesse, Mark Chen, Eric Sigler, Mateusz Litwin, Scott Gray, Benjamin Chess, Jack Clark, Christopher Berner, Sam McCandlish, Alec Radford, Ilya Sutskever, and Dario Amodei. Language models are few-shot learners. In _NeurIPS_ , pages 1877–1901, Virtual, 2020. PMLR. 

- [8] Xiangguo Sun, Jiawen Zhang, Xixi Wu, Hong Cheng, Yun Xiong, and Jia Li. Graph prompt learning: A comprehensive survey and beyond. _CoRR_ , abs/2311.16534, 2023. 

- [9] Haitao Mao, Zhikai Chen, Wenzhuo Tang, Jianan Zhao, Yao Ma, Tong Zhao, Neil Shah, Mikhail Galkin, and Jiliang Tang. Graph foundation models. _CoRR_ , abs/2402.02216, 2024. 

- [10] Antoine Bordes, Nicolas Usunier, Alberto García-Durán, Jason Weston, and Oksana Yakhnenko. Translating embeddings for modeling multi-relational data. In _NIPS_ , pages 2787–2795, Lake Tahoe, NV, USA, 2013. Curran Associates, Inc. 

- [11] Bishan Yang, Wen-tau Yih, Xiaodong He, Jianfeng Gao, and Li Deng. Embedding entities and relations for learning and inference in knowledge bases. In _ICLR_ , San Diego, CA, USA, 2015. OpenReview.net. 

- [12] Tim Dettmers, Pasquale Minervini, Pontus Stenetorp, and Sebastian Riedel. Convolutional 2D knowledge graph embeddings. In _AAAI_ , pages 1811–1818, New Orleans, LA, USA, 2018. AAAI Press. 

- [13] Shikhar Vashishth, Soumya Sanyal, Vikram Nitin, and Partha P. Talukdar. Compositionbased multi-relational graph convolutional networks. In _ICLR_ , Addis Ababa, Ethiopia, 2020. OpenReview.net. 

- [14] Zhiqing Sun, Zhi-Hong Deng, Jian-Yun Nie, and Jian Tang. Rotate: Knowledge graph embedding by relational rotation in complex space. In _ICLR_ , pages 1–18, New Orleans, LA, USA, 2019. OpenReview.net. 

- [15] Takuo Hamaguchi, Hidekazu Oiwa, Masashi Shimbo, and Yuji Matsumoto. Knowledge transfer for out-of-knowledge-base entities: A graph neural network approach. In _IJCAI_ , pages 1802– 1808, Melbourne, Australia, 2017. ijcai.org. 

- [16] Peifeng Wang, Jialong Han, Chenliang Li, and Rong Pan. Logic attention based neighborhood aggregation for inductive knowledge graph embedding. In _AAAI_ , pages 7152–7159, Honolulu, Hawaii, USA, 2019. AAAI Press. 

11 

- [17] Jiajun Chen, Huarui He, Feng Wu, and Jie Wang. Topology-aware correlations between relations for inductive link prediction in knowledge graphs. In _AAAI_ , pages 6271–6278, Virtual, 2021. AAAI Press. 

- [18] Muhao Chen, Yingtao Tian, Mohan Yang, and Carlo Zaniolo. Multilingual knowledge graph embeddings for cross-lingual knowledge alignment. In _IJCAI_ , pages 1511–1517, Melbourne, Australia, 2017. ijcai.org. 

- [19] Mikhail Galkin, Etienne G. Denis, Jiapeng Wu, and William L. Hamilton. Nodepiece: Compositional and parameter-efficient representations of large knowledge graphs. In _ICLR_ , Virtual, 2022. OpenReview.net. 

- [20] Yongqi Zhang, Zhanke Zhou, Quanming Yao, Xiaowen Chu, and Bo Han. Adaprop: Learning adaptive propagation for graph neural network based knowledge graph reasoning. In _KDD_ , pages 3446–3457, Long Beach, CA, USA, 2023. ACM. 

- [21] Zhaocheng Zhu, Xinyu Yuan, Mikhail Galkin, Sophie Xhonneux, Ming Zhang, Maxime Gazeau, and Jian Tang. A*net: A scalable path-based reasoning approach for knowledge graphs. _CoRR_ , abs/2206.04798, 2022. 

- [22] Xingyue Huang, Miguel Romero, Ismail[˙][˙] Ilkan Ceylan, and Pablo Barceló. A theory of link prediction via relational weisfeiler-leman on knowledge graphs. In _NeurIPS_ , pages 1–13. PMLR, 2023. 

- [23] Mingyang Chen, Wen Zhang, Yuxia Geng, Zezhong Xu, Jeff Z. Pan, and Huajun Chen. Generalizing to unseen elements: A survey on knowledge extrapolation for knowledge graphs. In _IJCAI_ , pages 6574–6582, Macao, China, 2023. ijcai.org. 

- [24] Yuxia Geng, Jiaoyan Chen, Jeff Z. Pan, Mingyang Chen, Song Jiang, Wen Zhang, and Huajun Chen. Relational message passing for fully inductive knowledge graph completion. In _ICDE_ , pages 1221–1233, Anaheim, CA, USA, 2023. IEEE. 

- [25] Jianfei Gao, Yangze Zhou, and Bruno Ribeiro. Double permutation equivariance for knowledge graph completion. _CoRR_ , abs/2302.01313:12, 2023. 

- [26] Jincheng Zhou, Beatrice Bevilacqua, and Bruno Ribeiro. An OOD multi-task perspective for link prediction with new relation types and nodes. _CoRR_ , abs/2307.06046, 2023. 

- [27] Jacob Devlin, Ming-Wei Chang, Kenton Lee, and Kristina Toutanova. Bert: Pre-training of deep bidirectional transformers for language understanding. In _NAACL_ , pages 4171–4186, Stroudsburg, PA, USA, 2019. ACL. 

- [28] Alexey Dosovitskiy, Lucas Beyer, Alexander Kolesnikov, Dirk Weissenborn, Xiaohua Zhai, Thomas Unterthiner, Mostafa Dehghani, Matthias Minderer, Georg Heigold, Sylvain Gelly, Jakob Uszkoreit, and Neil Houlsby. An image is worth 16x16 words: Transformers for image recognition at scale. In _ICLR_ , Virtual, 2021. OpenReview.net. 

- [29] Zhenyu Hou, Xiao Liu, Yukuo Cen, Yuxiao Dong, Hongxia Yang, Chunjie Wang, and Jie Tang. GraphMAE: Self-supervised masked graph autoencoders. _CoRR_ , abs/2205.10803:1–11, 2022. 

- [30] Weihua Hu, Bowen Liu, Joseph Gomes, Marinka Zitnik, Percy Liang, Vijay S. Pande, and Jure Leskovec. Strategies for pre-training graph neural networks. In _ICLR_ , Addis Ababa, Ethiopia, 2020. OpenReview.net. 

- [31] Yu Rong, Yatao Bian, Tingyang Xu, Weiyang Xie, Ying Wei, Wenbing Huang, and Junzhou Huang. Self-supervised graph transformer on large-scale molecular data. In _NeurIPS_ , Virtual, 2020. PMLR. 

- [32] Fan-Yun Sun, Jordan Hoffmann, Vikas Verma, and Jian Tang. Infograph: Unsupervised and semi-supervised graph-level representation learning via mutual information maximization. In _ICLR_ , Addis Ababa, Ethiopia, 2020. OpenReview.net. 

- [33] Petar Velickovic, William Fedus, William L. Hamilton, Pietro Liò, Yoshua Bengio, and R. Devon Hjelm. Deep graph infomax. In _ICLR_ , Addis Ababa, Ethiopia, 2019. OpenReview.net. 

12 

- [34] Zequn Sun, Jiacheng Huang, Jinghao Lin, Xiaozhou Xu, Qijin Chen, and Wei Hu. Joint pretraining and local re-training: Transferable representation learning on multi-source knowledge graphs. In _KDD_ , pages 2132–2144, Long Beach, CA, USA, 2023. ACM. 

- [35] Mingchen Sun, Kaixiong Zhou, Xin He, Ying Wang, and Xin Wang. Gppt: Graph pre-training and prompt tuning to generalize graph neural networks. In _KDD_ , pages 1717–1727, Washington, DC, USA, 2022. ACM. 

- [36] Taoran Fang, Yunchao Zhang, Yang Yang, and Chunping Wang. Prompt tuning for graph neural networks. _CoRR_ , abs/2209.15240, 2022. 

- [37] Xiangguo Sun, Hong Cheng, Jia Li, Bo Liu, and Jihong Guan. All in one: Multi-task prompting for graph neural networks. In _KDD_ , pages 2120–2131, Long Beach, CA, USA, 2023. ACM. 

- [38] Zemin Liu, Xingtong Yu, Yuan Fang, and Xinming Zhang. Graphprompt: Unifying pre-training and downstream tasks for graph neural networks. In _WWW_ , pages 417–428, Austin, TX, USA, 2023. ACM. 

- [39] Chenghua Gong, Xiang Li, Jianxiang Yu, Cheng Yao, Jiaqi Tan, Chengcheng Yu, and Dawei Yin. Prompt tuning for multi-view graph contrastive learning. _CoRR_ , abs/2310.10362, 2023. 

- [40] Yun Zhu, Jianhao Guo, and Siliang Tang. Sgl-pt: A strong graph learner with graph prompt tuning. _CoRR_ , abs/2302.12449, 2023. 

- [41] Reza Shirkavand and Heng Huang. Deep prompt tuning for graph transformers. _CoRR_ , abs/2309.10131, 2023. 

- [42] Mouxiang Chen, Zemin Liu, Chenghao Liu, Jundong Li, Qiheng Mao, and Jianling Sun. Ultradp: Ultra-unifying graph pre-training with multi-task graph dual prompt. _CoRR_ , abs/2310.14845, 2023. 

- [43] Yihong Ma, Ning Yan, Jiayu Li, Masood S. Mortazavi, and Nitesh V. Chawla. Hetgpt: Harnessing the power of prompt tuning in pre-trained heterogeneous graph neural networks. _CoRR_ , abs/2310.15318, 2023. 

- [44] Qingqing Ge, Zeyuan Zhao, Yiding Liu, Anfeng Cheng, Xiang Li, Shuaiqiang Wang, and Dawei Yin. Enhancing graph neural networks with structure-based prompt. _CoRR_ , abs/2310.17394, 2023. 

- [45] Michaël Defferrard, Xavier Bresson, and Pierre Vandergheynst. Convolutional neural networks on graphs with fast localized spectral filtering. In _NeurIPS_ , pages 3837–3845, Barcelona, Spain, 2016. PMLR. 

- [46] Qian Huang, Hongyu Ren, Peng Chen, Gregor Krzmanc, Daniel Zeng, Percy Liang, and Jure Leskovec. Prodigy: Enabling in-context learning over graphs. _CoRR_ , abs/2305.12600, 2023. 

- [47] Luis Antonio Galárraga, Christina Teflioudi, Katja Hose, and Fabian M. Suchanek. Amie: Association rule mining under incomplete evidence in ontological knowledge bases. In _WWW_ , pages 413–422, Rio de Janeiro, Brazil, 2013. ACM. 

- [48] Fan Yang, Zhilin Yang, and William W. Cohen. Differentiable learning of logical rules for knowledge base reasoning. In _NeurIPS_ , pages 2319–2328, Long Beach, CA, USA, 2017. PMLR. 

- [49] Christian Meilicke, Melisachew Wudage Chekol, Daniel Ruffinelli, and Heiner Stuckenschmidt. Anytime bottom-up rule learning for knowledge graph completion. In _IJCAI_ , pages 3137–3143, Macao, China, 2019. ijcai.org. 

- [50] Ali Sadeghian, Mohammadreza Armandpour, Patrick Ding, and Daisy Zhe Wang. Drum: End-to-end differentiable rule mining on knowledge graphs. In _NeurIPS_ , pages 15321–15331, Vancouver, Canada, 2019. PMLR. 

- [51] Timothée Lacroix, Nicolas Usunier, and Guillaume Obozinski. Canonical tensor decomposition for knowledge base completion. In _ICML_ , volume 80, pages 2869–2878. PMLR, 2018. 

13 

- [52] Mikhail Galkin, Max Berrendorf, and Charles Tapley Hoyt. An open challenge for inductive link prediction on knowledge graphs. _CoRR_ , abs/2203.01520, 2022. 

- [53] Kristina Toutanova and Danqi Chen. Observed versus latent features for knowledge base and text inference. In _CVSC_ , pages 57–66, Beijing, China, 2015. ACL. 

- [54] Wenhan Xiong, Thien Hoang, and William Yang Wang. Deeppath: A reinforcement learning method for knowledge graph reasoning. In _EMNLP_ , pages 564–573, Copenhagen, Denmark, 2017. ACL. 

- [55] Farzaneh Mahdisoltani, Joanna Biega, and Fabian M. Suchanek. Yago3: A knowledge base from multilingual wikipedias. In _CIDR_ , Virtual, 2015. www.cidrdb.org. 

- [56] Tara Safavi and Danai Koutra. Codex: A comprehensive knowledge graph completion benchmark. In _EMNLP_ , pages 8328–8350, Virtual, 2020. ACL. 

- [57] Xin Lv, Xu Han, Lei Hou, Juanzi Li, Zhiyuan Liu, Wei Zhang, Yichi Zhang, Hao Kong, and Suhui Wu. Dynamic anticipation and completion for multi-hop reasoning over sparse knowledge graph. In _EMNLP_ , pages 5694–5703, Virtual, 2020. ACL. 

- [58] Yihong Chen, Pasquale Minervini, Sebastian Riedel, and Pontus Stenetorp. Relation prediction as an auxiliary training objective for improving multi-relational graph representations. In _AKBC_ , Virtual, 2021. Online. 

- [59] Boyang Ding, Quan Wang, Bin Wang, and Li Guo. Improving knowledge graph embedding using simple constraints. In _ACL_ , pages 110–121, Melbourne, Australia, 2018. ACL. 

- [60] Chaitanya Malaviya, Chandra Bhagavatula, Antoine Bosselut, and Yejin Choi. Commonsense knowledge base completion with structural and semantic context. In _AAAI_ , pages 2925–2933, New York City, NY, USA, 2020. AAAI Press. 

- [61] Daniel Scott Himmelstein, Antoine Lizee, Christine Hessler, Leo Brueggeman, Sabrina L Chen, Dexter Hadley, Ari Green, Pouya Khankhanian, and Sergio E Baranzini. Systematic integration of biomedical knowledge prioritizes drugs for repurposing. _eLife_ , 6:e26726, 2017. 

- [62] Quan Wang, Zhendong Mao, Bin Wang, and Li Guo. Knowledge graph embedding: A survey of approaches and applications. _IEEE Trans. Knowl. Data Eng._ , 29(12):2724–2743, 2017. 

- [63] Shaoxiong Ji, Shirui Pan, Erik Cambria, Pekka Marttinen, and S Yu Philip. A survey on knowledge graphs: Representation, acquisition, and applications. _IEEE Trans. Neural Netw. Learn. Syst._ , 33(2):494–514, 2021. 

- [64] Andrea Rossi, Denilson Barbosa, Donatella Firmani, Antonio Matinata, and Paolo Merialdo. Knowledge graph embedding for link prediction: A comparative analysis. _ACM Trans. Knowl. Discov. Data_ , 15(2):1–49, 2021. 

- [65] Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun. Deep residual learning for image recognition. In _CVPR_ , pages 770–778, Las Vegas, NV, USA, 2016. IEEE. 

- [66] Lei Jimmy Ba, Jamie Ryan Kiros, and Geoffrey E. Hinton. Layer normalization. _CoRR_ , abs/1607.06450, 2016. 

- [67] Zequn Sun, Qingheng Zhang, Wei Hu, Chengming Wang, Muhao Chen, Farahnaz Akrami, and Chengkai Li. A benchmarking study of embedding-based entity alignment for knowledge graphs. _Proc. VLDB Endow._ , 13(11):2326–2340, 2020. 

- [68] Zequn Sun, Wei Hu, Qingheng Zhang, and Yuzhong Qu. Bootstrapping entity alignment with knowledge graph embedding. In _IJCAI_ , pages 4396–4402, Stockholm, Sweden, 2018. ijcai.org. 

- [69] Zequn Sun, Chengming Wang, Wei Hu, Muhao Chen, Jian Dai, Wei Zhang, and Yuzhong Qu. Knowledge graph alignment network with gated multi-hop neighborhood aggregation. In _AAAI_ , pages 222–229, New York City, NY, USA, 2020. AAAI Press. 

14 

- [70] Yuting Wu, Xiao Liu, Yansong Feng, Zheng Wang, Rui Yan, and Dongyan Zhao. Relation-aware entity alignment for heterogeneous knowledge graphs. In _IJCAI_ , pages 5278–5284, Macao, China, 2019. ijcai.org. 

- [71] Zhichun Wang, Qingsong Lv, Xiaohan Lan, and Yu Zhang. Cross-lingual knowledge graph alignment via graph convolutional networks. In _EMNLP_ , pages 349–357, Brussels, Belgium, 2018. ACL. 

- [72] Xin Mao, Wenting Wang, Yuanbin Wu, and Man Lan. Boosting the speed of entity alignment 10×: Dual attention matching network with normalized hard sample mining. In _WWW_ , pages 821–832, Ljubljana, Slovenia, 2021. ACM. 

- [73] Xuelu Chen, Muhao Chen, Changjun Fan, Ankith Uppunda, Yizhou Sun, and Carlo Zaniolo. Multilingual knowledge graph completion via ensemble knowledge transfer. In _Findings of EMNLP_ , pages 3227–3238, Virtual, 2020. Findings of ACL. 

- [74] Harkanwar Singh, Soumen Chakrabarti, Prachi Jain, Sharod Roy Choudhury, and Mausam. Multilingual knowledge graph completion with joint relation and entity alignment. In _AKBC_ , Virtual, 2021. Online. 

- [75] Zijie Huang, Zheng Li, Haoming Jiang, Tianyu Cao, Hanqing Lu, Bing Yin, Karthik Subbian, Yizhou Sun, and Wei Wang. Multilingual knowledge graph completion with self-supervised adaptive graph alignment. In _ACL_ , pages 474–485, Dublin, Ireland, 2022. ACL. 

- [76] Soumen Chakrabarti, Harkanwar Singh, Shubham Lohiya, Prachi Jain, and Mausam. Joint completion and alignment of multilingual knowledge graphs. In _EMNLP_ , pages 11922–11938, Abu Dhabi, United Arab Emirates, 2022. ACL. 

- [77] Rongchuan Tang, Yang Zhao, Chengqing Zong, and Yu Zhou. Multilingual knowledge graph completion with language-sensitive multi-graph attention. In _ACL_ , pages 10508–10519, Toronto, Canada, 2023. ACL. 

- [78] Wen Zhang, Yushan Zhu, Mingyang Chen, Yuxia Geng, Yufeng Huang, Yajing Xu, Wenting Song, and Huajun Chen. Structure pretraining and prompt tuning for knowledge graph transfer. In _WWW_ , pages 2581–2590, Austin, TX, USA, 2023. ACM. 

- [79] Shuwen Liu, Bernardo Cuenca Grau, Ian Horrocks, and Egor V. Kostylev. INDIGO: GNN-based inductive knowledge graph completion using pair-wise encoding. In _NeurIPS_ , pages 2034–2045, Virtual, 2021. Curran Associates, Inc. 

- [80] Yihong Chen, Pasquale Minervini, Sebastian Riedel, and Pontus Stenetorp. Relation prediction as an auxiliary training objective for improving multi-relational graph representations. In _AKBC_ , Virtual, 2021. Online. 

- [81] Tao He, Ming Liu, Yixin Cao, Zekun Wang, Zihao Zheng, Zheng Chu, and Bing Qin. Exploring & exploiting high-order graph structure for sparse knowledge graph completion. _CoRR_ , abs/2306.17034, 2023. 

- [82] Xin Lv, Xu Han, Lei Hou, Juanzi Li, Zhiyuan Liu, Wei Zhang, Yichi Zhang, Hao Kong, and Suhui Wu. Dynamic anticipation and completion for multi-hop reasoning over sparse knowledge graph. In _EMNLP_ , pages 5694–5703, Virtual, 2020. ACL. 

- [83] Jia Guo and Stanley Kok. Bique: Biquaternionic embeddings of knowledge graphs. In _EMNLP_ , pages 8338–8351, Punta Cana, Dominican Republic, 2021. ACL. 

- [84] Boyang Ding, Quan Wang, Bin Wang, and Li Guo. Improving knowledge graph embedding using simple constraints. In _ACL_ , pages 110–121, Melbourne, Australia, 2018. ACL. 

15 

## **A Message Passing Architectures** 

## **A.1 Message Passing Neural Network for Prompt Encoding** 

Based on the framework mentioned in Section 4.2, we present two types of aggregation: entity-centric and relation-centric aggregations. In each layer, we first update the entity representations and then update the relation representations. Specifically, given a central entity _e_ and the query relation _q_ , we update the representation of _e_ using following entity-centric aggregation function: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_A_Prompt-Based_Knowledge_Graph_Foundation_Model_for_Universal_In-Context_Reasoning_images/nips_2024_A_Prompt-Based_Knowledge_Graph_Foundation_Model_for_Universal_In-Context_Reasoning.pdf-0016-03.png)


where _Ne ⊆T_ pmt is the set of fact containing _e_ . **W** E-msg[(] _[l]_[)] _[∈]_[R] _[d][×]_[3] _[d]_[and] **[W]** E-attn[(] _[l]_[)] _[∈]_[R][1] _[×]_[2] _[d]_[are] two learnable parameter matrices. **s**[(] _[l]_[)] _,_ **r**[(] _[l]_[)] _,_ **q**[(] _[l]_[)] are the representations of _s, r, q_ in the _l_ -th layer, separately. ( _·||·_ ) denotes the concatenate operation. _σ_ ( _·_ ) denotes the Sigmoid activation function. 

We also adopt a query-aware attention mechanism for the relation-centric aggregation: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_A_Prompt-Based_Knowledge_Graph_Foundation_Model_for_Universal_In-Context_Reasoning_images/nips_2024_A_Prompt-Based_Knowledge_Graph_Foundation_Model_for_Universal_In-Context_Reasoning.pdf-0016-06.png)


where _Nr ⊆T_ pmt is the set of fact containing _r_ . **W** R-msg[(] _[l]_[)] _[∈]_[R] _[d][×]_[3] _[d]_[and] **[ W]** R-attn[(] _[l]_[)] _[∈]_[R][1] _[×]_[2] _[d]_[are two] learnable parameter matrices, and **o**[(] _[l]_[+1)] _∈_ **H**[(] E _[l]_[+1)] is the representation of _o_ . We also incorporate residual connection [65] and layer normalization [66] to enhance learning. 

## **A.2 Message Passing Neural Network for KG Encoding** 

Based on the framework mentioned in Section 4.3, we present a message passing neural network for KG encoding. Specifically, given a central entity _e_ and the query relation _q_ , we update the representation of _e_ using following aggregation function: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_A_Prompt-Based_Knowledge_Graph_Foundation_Model_for_Universal_In-Context_Reasoning_images/nips_2024_A_Prompt-Based_Knowledge_Graph_Foundation_Model_for_Universal_In-Context_Reasoning.pdf-0016-10.png)


where _Ne ⊆T_ is the set of fact containing _e_ , **W** msg[(] _[l]_[)] _[,]_ **[ W]** attn _[l][∈]_[R] _[d][×][d]_[ and] **[ W]** s[(] _[l]_[)] _[,]_ **[ W]** r[(] _[l]_[)] _[,]_ **[ W]** q[(] _[l]_[)] _∈_ R[1] _[×][d]_ are learnable parameter matrices, **s**[(] _[l]_[)] _,_ **r**[(] _[l]_[+1)] _,_ **q**[(] _[l]_[+1)] are the representations of _s, r, q_ , separately. 

## **B In-Context Reasoning Pipeline** 

We integrate the modules in Section 4 together and present a pipeline of in-context reasoning in Algorithm 1. Given an input query fact and its corresponding KG, we perform reasoning by scoring all candidate entities. In Lines 1–2, we first generate a few prompt graphs as context of the query relation and map entities and relations within them to predefined tokens. In Lines 3–9, we encode the prompt graphs to obtain prompt representations. In practice, we parallel encode these prompt graphs to ensure efficiency. In Line 10, we initialize the representations of KG entities and relations based on the prompts. In Lines 11–13, a multi-layer message passing neural network is employed for KG encoding. In Line 14, we assign a score for each candidate entity based on the output entity representations. For inference, these scores are used for entity ranking and metric calculation. For pre-training, these scores are utilized in Equation (9) to obtain the loss and update model parameters. 

16 

**Algorithm 1:** In-context reasoning 

**Input:** A query ( _s, q,_ ?) and the KG _K_ = ( _E, R, T_ ) it within. **Output:** Scores of all candidate entities in _E_ . `/* Stage 1: Prompt graph generation */` **1** Generate _M_ prompt graphs _{Pc_ 1 _, . . . , PcM }_ for the query relation _q_ ; 

**2** Map the entities and relations in prompt graphs to predefined tokens using unified tokenizer; `/* Stage 2: Prompt encoding */` **3 for** _Pci ← 1_ **to** _M_ **do** 

**4** Initialize entity and relation representations **H**[(0)] E and **H**[(0)] R using token representations; **5 for** _l ← 1_ **to** _L_ **do 6** Update entity representations **H**[(] E _[l]_[)][using Equation (3);] **7** Update relation representations **H**[(] R _[l]_[)][using Equation (4);] **8** Readout the relation representations **H** _Pci_ using Equation (5); **9** Obtain prompt representation matrix **H** pmt using Equation (6); `/* Stage 3: KG encoding and reasoning */` **10** Initialize entity and relation representations _V_ E[(0)] and _V_ R[(0)] based on **H** pmt; **11 for** _l ← 1_ **to** _N_ **do** 

**12** Update relation representations **V** R[(] _[l]_[)][using Equation (7);] **13** Update entity representations **V** E[(] _[l]_[)][using Equation (8);] **14** Score entities in _E_ based on entity representations **V** _E_[(] _[N]_[)] using reasoning module in Section 4.3; 

## **C Implementation Details** 

Under the framework in Section 4, we implement an in-context reasoning model KG-ICL, which employs a 5-shot 3-hop prompt graph as context, along with 3 stacked layers for prompt graph encoding, and 6 stacked layers for KG encoding and reasoning, i.e., _M_ = 5, _k_ = 3, _L_ = 3 and _N_ = 6. The dimension _d_ of the hidden layers is set to 32. Following the standard in-context learning process [46], we first pre-train a model on source datasets and then freeze the model parameters for evaluation. We pre-train our model on three source datasets, i.e., FB V1 [1] with 180 relations, NELL V1 [1] with only 14 relations, and CoDEx-small [56] with 42 relations. We use Adam optimizer and set the learning rate to 0.001 and the patience of early stopping to 5. The pre-training process is conducted on a workstation with two Intel Xeon Gold CPUs, four NVIDIA RTX A6000 GPUs, and Ubuntu 18.04 LTS. The pre-training model maintains a modest size with only 89k parameters, and the pre-training process converges in less than six hours. 

## **D Related Work** 

## **D.1 Knowledge Graph Reasoning** 

_Diverse KG reasoning settings._ KG reasoning primarily involves three settings: transductive, inductive, and fully-inductive. Early studies [10–14] focus mainly on the transductive setting, assuming that KGs are static. They learn an embedding for each specific entity, making it challenging to handle the addition of new entities. Real-world KGs are dynamic, inspiring the development of inductive models [1–4, 15–21, 23] that allows for unseen entities. These models base their reasoning on relation patterns rather than entity embeddings. In the fully-inductive setting [5, 24–26], unseen entities and relations can both emerge in the query facts. While this setting is closer to pre-training, it remains limited to the same KG. The distinction among these settings arises from the fact that text data can be naturally split into unified tokens, while the entity and relation sets across KGs are not shared. In this paper, we propose a prompt graph and a unified tokenizer to support in-context learning, breaking down the barriers imposed by these settings and achieving universal reasoning capabilities. 

_Entity alignment and pre-training._ Extensive research efforts have been concentrated on establishing a unified entity vocabulary to support pre-training through the recognition of identical entities in different KGs, a task commonly known as entity alignment [18, 67–72]. Based on these aligned 

17 

entities, some KG pre-training models [34, 73–77] have been proposed to map the entities in diverse KGs into a unified semantic space. Nevertheless, this paradigm heavily depends on expensive pre-labeled alignment, which is not always sufficient in the real world. More critically, it relies on similar schemata [34], lacking robust generality across KGs that span diverse domains. ULTRA [6] introduces a foundation model following the paradigm of “pre-training then finetuning”, which is an alignment-free reasoning model. Stand on the shoulders of previous work, we aim to avoid dataset-specific finetuning and propose a model that can achieve universal reasoning capabilities with just a few examples as contextual prompts. In Section 5, we conduct comparative experiments with ULTRA, and the results demonstrate the effectiveness of our in-context learning. Additionally, KGTransformer [78] introduces a prompt-based KG pre-training model to support a variety of downstream tasks. Their objective is not KG reasoning but rather the transfer of knowledge from KGs to enhance downstream tasks such as image classification. 

## **D.2 Prompt-based In-Context Learning** 

Our work is also related to prompt learning and in-context learning. Here, our primary focus is on KG reasoning, which shares similar challenges with graph learning. Drawing inspiration from the success of early pre-training models in NLP [27] and computer vision [28], some graph pretraining models [29–33] have been proposed. These models follow the paradigm of “pre-train and finetune”, where a model is initially pre-trained and then finetuned for the target task. However, this paradigm has limitations in terms of generalization and may sometimes lead to negative knowledge transfer. Consequently, recent researches [8, 35–45] have shifted focus towards the “pre-train, prompt, and finetune” paradigm. This paradigm leverages task prompts to enhance the knowledge transfer and generalization capabilities of pre-trained models, achieving significant progress. KG reasoning involves making inferences based on multi-relational data. Therefore, these pre-trained models are not easily applicable to KG reasoning tasks. Inspired by the success of recent black-box large language models like GPT [7], the in-context learning paradigm aims to avoid finetuning on the target dataset. Instead, it imparts general capabilities to pre-trained models with just a few examples. PRODIGY [46] introduces an in-context learning-based graph pre-training model to handle various classification tasks. While it can perform relation classification tasks, it is not suitable for KG reasoning with a massive number of candidate entities. 

## **E Further Analyses** 

## **E.1 Impact of Pre-training Mixture** 

The effectiveness of in-context learning is inherently tied to the quality and diversity of the source datasets used for pre-training. Here, we analyze the impact of the bias of source KGs by introducing six different combinations of source KGs. The results are reported in Table 4. We observe that (i) more source KGs help reduce the influence of biases in individual datasets, and (ii) these three source KGs are of good quality, as even using just one for pre-training yields decent performance. Besides, we find that our pre-training does not require a large scale of KG facts. The variety of relational structures is more important for our pre-training. Thus, in practice, we can choose several KGs with different schemata or from different domains for pre-training. 

Table 4: Performance w.r.t. different pre-training KGs. 

|Source Datasets<br>FB V1<br>NELL V1<br>CoDEx-small|Average Results<br>MRR<br>H@10|
|---|---|
|✓<br>✓<br>✓<br>✓<br>✓<br>✓<br>✓<br>✓<br>✓|0.424<br>0.586<br>0.392<br>0.565<br>0.389<br>0.561<br>0.425<br>0.592<br>0.436<br>0.606<br>0.423<br>0.595|
|✓<br>✓<br>✓|0.442<br>0.606|



We also conducted experiments using the same pre-training mixture as ULTRA [6], specifically WN18RR, FB15k-237, and CoDEx-medium, to pre-train KG-ICL. The results are shown in Table 5. 

18 

We observe that KG-ICL outperforms ULTRA in this setting. Pre-training with these datasets causes a slight decrease in the inductive performance and a slight improvement in the transductive results, but neither change is significant. We use three smaller datasets to pre-train KG-ICL, as smaller datasets do not significantly affect model performance and can expedite the pre-training process. 

Table 5: Performance w.r.t the same pre-training mixture as ULTRA [6]. 

|Models|Inductive<br>(14 KGs)<br>MRR<br>H@10|Fully-inductive<br>(13 KGs)<br>MRR<br>H@10|Transductive<br>(16 KGs)<br>MRR<br>H@10|Average<br>(43 KGs)<br>MRR<br>H@10|
|---|---|---|---|---|
|ULTRA (FB15k-237 WN18RR CoDEx-medium)<br>KG-ICL (FB15k-237 WN18RR CoDEx-medium)<br>KG-ICL (FB V1 NELL V1 CoDEx-small)|0.513<br>0.664<br>0.547<br>0.700<br>0.554<br>0.707|0.352<br>0.536<br>0.431<br>0.629<br>0.439<br>0.635|0.329<br>0.479<br>0.357<br>0.506<br>0.346<br>0.493|0.396<br>0.557<br>0.441<br>0.606<br>0.442<br>0.606|



Following [6], we conduct experiments on growing pre-training mixtures, sequentially adding pretraining datasets in the same order as in [6], i.e., FB15k-237, WN18RR, CoDEx-medium, NELL-995, YAGO3-10, ConceptNet100K, DBpedia100K, and AristoV4. The results are shown in Figure 6. We observe that the performance improves with the number of pre-training datasets. Unlike ULTRA, KG-ICL even performs well with pre-training on a single KG. This improvement is due to two key factors: first, we generate a diverse set of prompt graphs for different relations within the same KG, which increases sample diversity. Second, our targeted prompt engineering reduces learning complexity and facilitates better generalization. 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_A_Prompt-Based_Knowledge_Graph_Foundation_Model_for_Universal_In-Context_Reasoning_images/nips_2024_A_Prompt-Based_Knowledge_Graph_Foundation_Model_for_Universal_In-Context_Reasoning.pdf-0019-04.png)


**----- Start of picture text -----**<br>
inductive fully-inductive transductive<br>0.60 0.75<br>0.50 0.65<br>0.40 0.55<br>0.30 0.45<br>1 2 3 4 5 6 7 8 1 2 3 4 5 6 7 8<br>Number of pre-training datasets Number of pre-training datasets<br>MRR H@10<br>**----- End of picture text -----**<br>


Figure 6: MRR and H@10 results with increasing number of pre-training datasets. 

## **E.2 Complexity Analyses of Prompt Graph Preprocessing** 

The proposed model relies on generating and processing prompt graphs. Here, we analyze the computational complexity of the pre-processing efficiency. The generation processing of a prompt graph includes two steps: (i) Subgraph extraction. We take the intersection of the _k_ -hop neighbors of the subject entity _u_ and the object entity _v_ to obtain the set of nodes ( _O_ ( _|T |_ + _|E|_ )), where _T_ and _E_ are the set of facts and entities in the KG, respectively. Then, we retrieve the facts between these entities ( _O_ ( _|T |_ )). (ii) Labeling. We perform twice single-source shortest path length calculations (for _u_ and _v_ ) on the prompt subgraph ( _O_ ( _|T_ pmt _|_ + _|E_ pmt _|_ ) _×_ log _|E_ pmt _|_ ) to get token labels. In summary, the overall computational complexity is _O_ ( _|T |_ + _|E|_ + ( _|T_ pmt _|_ + _|E_ pmt _|_ ) _×_ log _|E_ pmt _|_ ). Note that the size of the prompt graph is usually much smaller than the entire KG. In the _M_ -shot (e.g., 5-shot) in-context learning setting, we only need to extract _M × |R|_ prompt graphs, where _R_ denotes the set of relations in KG. We report the preprocessing times under the 5-shot setting in Table 6. We can observe that preprocessing all 43 datasets (including some large KGs) requires 1597 seconds, averaging 37.1 seconds per dataset. Among them, AristoV4 [58], with the most relations (with 1605 relations, 44949 entities, and 242567 facts), has the longest preprocessing time, at 995 seconds, averaging 0.62 seconds per relation. Hetionet [61], with the most facts (with 24 relations, 45158 entities, and 2025177 facts), has a preprocessing time of 120 seconds, averaging 5 seconds per relation. YAGO3-10 [55], with the most entities (with 34 relations, 123182 entities, and 1079040 facts), has a preprocessing time of 50 seconds, averaging 1.47 seconds per relation. In summary, the preprocessing method is scalable because we extract only a small number of examples for each relation. Evaluations on large-scale datasets containing millions of facts also confirm its scalability. 

19 

Table 6: The total and average preprocessing time. 

|Datasets|Time (s)|
|---|---|
||Total<br>Average|
|Inductive (14 KGs)<br>Fully-inductive (13 KGs)<br>Transductive (16 KGs)<br>All (43 KGs)|42.0<br>3.0<br>96.0<br>7.4<br>1459.0<br>91.2<br>1597.0<br>37.1|



## **E.3 Incorporating Other Message Passing Layer** 

The proposed model can also be incorporated with other message passing neural networks that can aggregate messages conditioned with specific queries. Here, we implement a variant, KG-ICL (NBFNet), by incorporating the message passing of NBFNet [4], which is used by ULTRA [6]. Note that NBFNet only outputs entity representations but not updates or outputs relation representations, which is also one reason we did not adopt NBFNet initially. ULTRA treats relations as nodes to obtain relation representations. Therefore, we incorporate Equation (4) into NBFNet (default configuration) to support relation encoding. The results are shown in Table 7. We can observe that KG-ICL (NBFNet) also achieved promising results, slightly below KG-ICL. This demonstrates the potential of combining KG-ICL with more passing message neural networks. Moreover, the structure of this variant is similar to ULTRA, but the input is prompt graphs rather than relation graphs, which indicates the superiority of our prompt graph to that of ULTRA’s relation graph in relation modeling. 

Table 7: The results of the variant incorporating with NBFNet. 

|Models|Inductive<br>(14 KGs)<br>MRR<br>H@10|Fully-inductive<br>(13 KGs)<br>MRR<br>H@10|Transductive<br>(16 KGs)<br>MRR<br>H@10|Average<br>(43 KGs)<br>MRR<br>H@10|
|---|---|---|---|---|
|KG-ICL<br>KG-ICL (NBFNet)<br>ULTRA|0.554<br>0.707<br>0.545<br>0.703<br>0.513<br>0.664|0.439<br>0.635<br>0.423<br>0.622<br>0.352<br>0.536|0.346<br>0.493<br>0.298<br>0.438<br>0.329<br>0.479|0.442<br>0.606<br>0.416<br>0.580<br>0.396<br>0.557|



## **F Dataset Statistics** 

We conduct extensive evaluations on 43 datasets. We categorize the datasets into three types: inductive datasets, fully-inductive datasets, and transductive datasets. The statistical data of these datasets and their state-of-the-art models are reported in Tables 8, 9, and 10, respectively. 

## **G Detailed Results** 

To validate the effectiveness of our in-context reasoning model, we compare KG-ICL with the supervised SOTA models and ULTRA’s pre-training and finetuning versions on 43 datasets. The detailed results for each dataset are presented in Table 11. We observe that (i) KG-ICL outperforms the competitors on most datasets, demonstrating the universal reasoning capability of our in-context model. (ii) “KG-ICL pre-train” outperforms “ULTRA pre-train” on 11 inductive datasets, all 13 fully-inductive datasets, and 9 transductive datasets, which demonstrates the superiority of our incontext KG reasoning foundation model. In addition, “KG-ICL finetune” also outperforms “ULTRA finetune” on most datasets. (iii) InGram [5] transfers knowledge to new query relations through a relation graph, while we employ the prompt graph as a bridge for knowledge transfer. Our KG-ICL outperforms InGram on all 13 fully-inductive datasets, indicating that our prompt graphs can better highlight important clues for specific query relations than relation graphs. (iv) On transductive datasets, KG-ICL’s performance improvement compared to supervised baseline models is less than that in the previous two settings. There are two reasons for this: first, the supervised signals on transductive datasets directly target entities and relations in the test set, allowing supervised models to effectively learn representations and achieve high performance. Second, most existing KG reasoning models are developed based on several transductive datasets such as FB15k-237 [53], WN18RR [12], YAGO3-10 [55], and NELL-995 [54]. Models specifically designed for these datasets also contribute 

20 

Table 8: Statistics of inductive datasets. 

|Datasets<br>#Rel.|Datasets<br>#Rel.|Training graphs<br>#Ent.<br>#Facts|Validation graphs<br>#Ent.<br>#Facts<br>#Valid.|Test graphs<br>SOTA<br>#Ent.<br>#Facts<br>#Test|Test graphs<br>SOTA<br>#Ent.<br>#Facts<br>#Test|
|---|---|---|---|---|---|
|FB V1 [1]<br>FB V2 [1]<br>FB V3 [1]<br>FB V4 [1]<br>WN V1 [1]<br>WN V2 [1]<br>WN V3 [1]<br>WN V4 [1]<br>NELL V1 [1]<br>NELL V2 [1]<br>NELL V3 [1]<br>NELL V4 [1]<br>ILPC-small [52]<br>ILPC-large [52]|180<br>215<br>215<br>219<br>9<br>10<br>11<br>9<br>14<br>88<br>142<br>76<br>48<br>65|1594<br>4245<br>3668<br>9799<br>3668<br>17986<br>4707<br>27203<br>2746<br>5410<br>6054<br>15606<br>12078<br>25901<br>3861<br>7940<br>3103<br>4687<br>2564<br>8219<br>4647<br>16393<br>2092<br>7546<br>10230<br>78616<br>46626<br>202446|1594<br>4245<br>489<br>3668<br>9799<br>1166<br>3668<br>17986<br>2194<br>4707<br>27203<br>3352<br>2746<br>5410<br>630<br>6054<br>15606<br>1838<br>12078<br>25901<br>3097<br>3861<br>7940<br>934<br>3103<br>4687<br>414<br>2564<br>8219<br>922<br>4647<br>16393<br>1851<br>2092<br>7546<br>876<br>6553<br>29060<br>2908<br>29246<br>77044<br>10179|1093<br>1993<br>411<br>2501<br>4406<br>947<br>2501<br>7406<br>1731<br>3051<br>11714<br>2840<br>922<br>1618<br>373<br>2757<br>4011<br>852<br>5084<br>6327<br>1143<br>7084<br>12334<br>2823<br>225<br>833<br>201<br>2086<br>4586<br>935<br>3566<br>8048<br>1620<br>2795<br>7073<br>1447<br>6653<br>29060<br>2902<br>29246<br>77044<br>10184|A*Net [21]<br>NBFNet [4]<br>NBFNet [4]<br>A*Net [21]<br>NBFNet [4]<br>NBFNet [4]<br>NBFNet [4]<br>A*Net [21]<br>RED-GNN [3]<br>RED-GNN [3]<br>RED-GNN [3]<br>RED-GNN [3]<br>NodePiece [19]<br>NodePiece [19]|



Table 9: Statistics of fully-inductive datasets. 

|Datasets|Training graphs<br>#Ent.<br>#Rel.<br>#Facts|Validation graphs<br>#Ent.<br>#Rel.<br>#Facts<br>#Valid.|Test graphs<br>SOTA<br>#Ent.<br>#Rel.<br>#Facts<br>#Test|Test graphs<br>SOTA<br>#Ent.<br>#Rel.<br>#Facts<br>#Test|
|---|---|---|---|---|
|FB-25 [5]<br>FB-50 [5]<br>FB-75 [5]<br>FB-100 [5]<br>WK-25 [5]<br>WK-50 [5]<br>WK-75 [5]<br>WK-100 [5]<br>NL-0 [5]<br>NL-25 [5]<br>NL-50 [5]<br>NL-75 [5]<br>NL-100 [5]|5190<br>163<br>91571<br>5190<br>153<br>85375<br>4659<br>134<br>62809<br>4659<br>134<br>62809<br>12659<br>47<br>41873<br>12022<br>72<br>82481<br>6853<br>52<br>28741<br>9784<br>67<br>49875<br>1814<br>134<br>7796<br>4396<br>106<br>17578<br>4396<br>106<br>17578<br>2607<br>96<br>11058<br>1258<br>55<br>7832|4097<br>216<br>17147<br>5716<br>4445<br>205<br>11636<br>3879<br>2792<br>186<br>9316<br>3106<br>2624<br>77<br>6987<br>2329<br>3228<br>74<br>3391<br>1130<br>9328<br>93<br>9672<br>3224<br>2722<br>65<br>3430<br>1143<br>12136<br>37<br>13487<br>4496<br>2026<br>112<br>2287<br>763<br>2146<br>120<br>2230<br>743<br>2335<br>119<br>2576<br>859<br>1578<br>116<br>1818<br>606<br>1709<br>53<br>2378<br>793|4097<br>216<br>17147<br>5716<br>4445<br>205<br>11636<br>3879<br>2792<br>186<br>9316<br>3106<br>2624<br>77<br>6987<br>2329<br>3228<br>74<br>3391<br>1131<br>9328<br>93<br>9672<br>3225<br>2722<br>65<br>3430<br>1144<br>12136<br>37<br>13487<br>4496<br>2026<br>112<br>2287<br>763<br>2146<br>120<br>2230<br>744<br>2335<br>119<br>2576<br>859<br>1578<br>116<br>1818<br>607<br>1709<br>53<br>2378<br>793|InGram [5]<br>InGram [5]<br>InGram [5]<br>InGram [5]<br>InGram [5]<br>InGram [5]<br>InGram [5]<br>InGram [5]<br>InGram [5]<br>InGram [5]<br>InGram [5]<br>InGram [5]<br>InGram [5]|



to high performance. Nevertheless, our KG-ICL still achieves results superior to supervised baseline models on 11 transductive datasets. 

A few extra inductive datasets [15, 79] and fully-inductive data-sets [26] are often evaluated under the 1 vs. 50 setting, where the target entity is selected from 50 randomly sampled candidates. In [6], it evaluates on them under the full candidate setting, which reduces the uncertainty caused by random samples and provides a more stable evaluation. Therefore, we also conduct comparative experiments with ULTRA on these datasets under the full candidate setting. The results are reported in Table 12. We observe that KG-ICL outperforms ULTRA on most datasets. 

Table 10: Statistics of transductive datasets. 

|Datasets|#Ent.<br>#Rel.<br>#Training<br>#Valid.<br>#Test|SOTA|
|---|---|---|
|CoDEx-small [56]<br>WDsinger [57]<br>FB15k-237-10 [57]<br>FB15k-237-20 [57]<br>FB15k-237-50 [57]<br>FB15k-237 [53]<br>CoDEx-medium [56]<br>NELL23k [57]<br>WN18RR [12]<br>AristoV4 [58]<br>Hetionet [61]<br>NELL-995 [54]<br>CoDEx-large [56]<br>ConceptNet100k [60]<br>DBpedia100k [59]<br>YAGO3-10 [55]|2034<br>42<br>32888<br>1827<br>1828<br>10282<br>35<br>16442<br>1641<br>1640<br>11512<br>237<br>27211<br>15624<br>18150<br>13166<br>237<br>54423<br>16963<br>19776<br>14149<br>237<br>136057<br>17449<br>20324<br>14541<br>237<br>272115<br>17535<br>20466<br>17050<br>69<br>185584<br>10390<br>10391<br>22925<br>200<br>25445<br>4961<br>4952<br>40943<br>11<br>86835<br>3034<br>3134<br>44949<br>1605<br>242567<br>20000<br>20000<br>45158<br>24<br>2025177<br>112510<br>112510<br>74536<br>200<br>149678<br>543<br>2818<br>77951<br>69<br>551193<br>30622<br>30622<br>78334<br>34<br>100000<br>1200<br>1200<br>99604<br>470<br>597572<br>50000<br>50000<br>123182<br>37<br>1079040<br>5000<br>5000|ComplEx RP [80]<br>LR-GCN [81]<br>DacKGR [82]<br>DacKGR [82]<br>DacKGR [82]<br>NBFNet [4]<br>ComplEx RP [80]<br>LR-GCN [81]<br>NBFNet [4]<br>ComplEx RP [80]<br>RotatE [14]<br>RED-GNN [3]<br>ComplEx RP [80]<br>BiQUE [83]<br>ComplEx-NNE+AER [84]<br>NBFNet [4]|



21 

Table 11: Detailed results on 43 datasets. 

|Datasets|Supervised SOTA<br>MRR<br>H@10|ULTRA pre-train<br>MRR<br>H@10|KG-ICL pre-train<br>MRR<br>H@10|ULTRA fnetune<br>MRR<br>H@10|KG-ICL fnetune<br>MRR<br>H@10|
|---|---|---|---|---|---|
|FB V1<br>FB V2<br>FB V3<br>FB V4<br>ILPC-large<br>ILPC-small<br>NELL V1<br>NELL V2<br>NELL V3<br>NELL V4<br>WN V1<br>WN V2<br>WN V3<br>WN V4|0.457<br>0.589<br>0.510<br>0.672<br>0.476<br>0.637<br>0.466<br>0.645<br>0.070<br>0.146<br>0.130<br>0.251<br>0.637<br>0.866<br>0.419<br>0.601<br>0.436<br>0.594<br>0.363<br>0.556<br>0.741<br>0.826<br>0.704<br>**0.798**<br>0.452<br>0.568<br>0.661<br>0.743|0.498<br>0.656<br>0.512<br>0.700<br>0.491<br>0.654<br>0.486<br>0.677<br>0.290<br>0.424<br>0.302<br>0.443<br>0.785<br>0.913<br>0.526<br>0.707<br>0.515<br>0.702<br>0.479<br>0.712<br>0.648<br>0.768<br>0.663<br>0.765<br>0.376<br>0.476<br>0.611<br>0.705|0.520<br>0.678<br>0.565<br>0.749<br>0.535<br>0.695<br>0.513<br>0.699<br>0.288<br>0.412<br>0.288<br>0.446<br>0.693<br>0.915<br>**0.644**<br>**0.835**<br>0.613<br>0.792<br>0.590<br>0.791<br>0.733<br>**0.838**<br>0.696<br>0.783<br>0.425<br>0.548<br>0.652<br>0.722|0.509<br>0.670<br>0.524<br>0.710<br>0.504<br>0.663<br>0.496<br>0.684<br>**0.308**<br>**0.431**<br>0.303<br>0.453<br>0.757<br>0.878<br>0.575<br>0.761<br>0.563<br>0.755<br>0.469<br>0.733<br>0.685<br>0.793<br>0.679<br>0.779<br>0.411<br>0.546<br>0.614<br>0.720|**0.531**<br>**0.700**<br>**0.568**<br>**0.768**<br>**0.537**<br>**0.704**<br>**0.525**<br>**0.706**<br>0.295<br>0.411<br>**0.316**<br>**0.473**<br>**0.841**<br>**0.995**<br>0.641<br>0.835<br>**0.631**<br>**0.799**<br>**0.594**<br>**0.802**<br>**0.762**<br>0.827<br>**0.721**<br>0.787<br>**0.503**<br>**0.626**<br>**0.683**<br>**0.749**|
|FB-25<br>FB-50<br>FB-75<br>FB-100<br>NL-0<br>NL-25<br>NL-50<br>NL-75<br>NL-100<br>WK-25<br>WK-50<br>WK-75<br>WK-100|0.223<br>0.371<br>0.189<br>0.325<br>0.117<br>0.218<br>0.133<br>0.271<br>0.309<br>0.506<br>0.261<br>0.464<br>0.281<br>0.453<br>0.334<br>0.501<br>0.269<br>0.431<br>0.107<br>0.169<br>0.247<br>0.362<br>0.068<br>0.135<br>0.186<br>0.309|0.388<br>0.640<br>0.338<br>0.543<br>0.403<br>0.604<br>0.449<br>0.642<br>0.342<br>0.523<br>0.395<br>0.569<br>0.407<br>0.570<br>0.368<br>0.547<br>0.471<br>0.651<br>0.316<br>0.532<br>0.166<br>0.324<br>0.365<br>0.537<br>0.164<br>0.286|0.396<br>0.656<br>0.341<br>0.559<br>0.438<br>0.633<br>0.487<br>0.694<br>**0.557**<br>**0.777**<br>**0.550**<br>**0.736**<br>**0.534**<br>0.704<br>**0.452**<br>0.673<br>0.556<br>0.762<br>0.423<br>0.621<br>0.273<br>0.430<br>0.437<br>0.602<br>0.262<br>0.409|0.383<br>0.635<br>0.334<br>0.538<br>0.400<br>0.598<br>0.444<br>0.643<br>0.329<br>0.551<br>0.407<br>0.596<br>0.418<br>0.595<br>0.374<br>0.570<br>0.458<br>0.684<br>0.321<br>0.535<br>0.140<br>0.280<br>0.380<br>0.530<br>0.168<br>0.286|**0.434**<br>**0.694**<br>**0.384**<br>**0.598**<br>**0.458**<br>**0.664**<br>**0.499**<br>**0.703**<br>0.555<br>0.765<br>0.540<br>0.730<br>0.528<br>**0.708**<br>0.446<br>**0.681**<br>**0.557**<br>**0.766**<br>**0.425**<br>**0.628**<br>**0.277**<br>**0.432**<br>**0.466**<br>**0.626**<br>**0.270**<br>**0.415**|
|AristoV4<br>CoDEx-small<br>CoDEx-medium<br>CoDEx-large<br>ConceptNet100K<br>DBpedia100K<br>FB15k-237<br>FB15k-237-10<br>FB15k-237-20<br>FB15k-237-50<br>Hetionet<br>NELL-995<br>NELL23K<br>WD-singer<br>WN18RR<br>YAGO3-10|0.311<br>0.447<br>0.473<br>0.663<br>0.352<br>0.490<br>0.345<br>0.473<br>0.320<br>0.553<br>0.306<br>0.418<br>**0.415**<br>**0.599**<br>0.219<br>0.337<br>0.247<br>0.391<br>0.293<br>0.458<br>0.257<br>0.403<br>**0.543**<br>0.651<br>0.253<br>0.419<br>0.393<br>0.500<br>**0.551**<br>**0.666**<br>**0.563**<br>0.708|0.182<br>0.282<br>0.472<br>0.667<br>0.372<br>0.525<br>0.338<br>0.469<br>0.082<br>0.162<br>0.398<br>0.576<br>0.368<br>0.564<br>0.248<br>0.398<br>0.272<br>0.436<br>0.324<br>0.526<br>0.257<br>0.379<br>0.406<br>0.543<br>0.239<br>0.408<br>0.382<br>0.498<br>0.480<br>0.614<br>0.451<br>0.615|0.203<br>0.306<br>0.465<br>0.654<br>0.330<br>0.474<br>0.261<br>0.376<br>0.249<br>0.416<br>0.390<br>0.541<br>0.359<br>0.541<br>**0.274**<br>**0.433**<br>**0.285**<br>0.454<br>**0.329**<br>0.520<br>0.260<br>0.371<br>0.532<br>0.653<br>0.317<br>0.532<br>0.470<br>0.582<br>0.455<br>0.527<br>0.352<br>0.503|**0.343**<br>**0.496**<br>**0.490**<br>**0.686**<br>0.372<br>0.525<br>0.343<br>0.478<br>0.310<br>0.529<br>0.436<br>0.603<br>0.368<br>0.564<br>0.254<br>0.411<br>0.274<br>0.445<br>0.325<br>**0.528**<br>**0.399**<br>**0.538**<br>0.509<br>0.660<br>0.268<br>0.450<br>0.417<br>0.526<br>0.480<br>0.614<br>0.557<br>**0.710**|0.313<br>0.480<br>0.479<br>0.662<br>**0.402**<br>**0.565**<br>**0.388**<br>**0.508**<br>**0.371**<br>**0.584**<br>**0.455**<br>**0.604**<br>0.376<br>0.538<br>0.260<br>0.416<br>0.284<br>**0.456**<br>0.324<br>0.499<br>0.269<br>0.402<br>0.534<br>**0.672**<br>**0.329**<br>**0.552**<br>**0.493**<br>**0.599**<br>0.536<br>0.637<br>0.545<br>0.688|
|Average|0.351<br>0.493|0.396<br>0.557|0.442<br>0.606|0.421<br>0.590|**0.473**<br>**0.638**|



Table 12: Results on more datasets under the full candidate setting. 

|Datasets|ULTRA pre-train<br>MRR<br>H@10|KG-ICL pre-train<br>MRR<br>H@10|ULTRA fnetune<br>MRR<br>H@10|KG-ICL fnetune<br>MRR<br>H@10|
|---|---|---|---|---|
|HM 1k<br>HM 3k<br>HM 5k<br>IndigoBM<br>MT1 tax<br>MT1 health<br>MT2 org<br>MT2 sci<br>MT3 art<br>MT3 infra<br>MT4 sci<br>MT4 health<br>Metafam<br>FBNELL|0.059<br>0.092<br>0.037<br>0.077<br>0.034<br>0.071<br>**0.440**<br>**0.648**<br>0.224<br>0.305<br>0.298<br>0.374<br>0.095<br>0.159<br>0.258<br>0.354<br>0.259<br>0.402<br>0.619<br>0.755<br>0.274<br>0.449<br>0.624<br>0.737<br>0.238<br>0.644<br>0.485<br>0.652|0.059<br>0.107<br>0.049<br>0.099<br>0.043<br>0.091<br>0.351<br>0.558<br>0.280<br>0.451<br>0.378<br>0.464<br>0.093<br>0.156<br>**0.326**<br>**0.476**<br>0.258<br>0.406<br>0.633<br>0.777<br>0.296<br>0.470<br>0.648<br>0.767<br>0.500<br>0.886<br>0.509<br>0.692|0.042<br>0.100<br>0.030<br>0.090<br>0.025<br>0.068<br>0.432<br>0.639<br>0.330<br>0.459<br>0.380<br>0.467<br>**0.104**<br>0.170<br>0.311<br>0.451<br>**0.306**<br>**0.473**<br>0.657<br>0.807<br>0.303<br>**0.478**<br>0.704<br>**0.785**<br>0.997<br>**1.000**<br>0.481<br>0.661|**0.089**<br>**0.144**<br>**0.081**<br>**0.129**<br>**0.070**<br>**0.108**<br>**0.440**<br>0.641<br>**0.411**<br>**0.521**<br>**0.387**<br>**0.479**<br>0.100<br>**0.171**<br>0.303<br>0.396<br>**0.306**<br>0.460<br>**0.676**<br>**0.808**<br>**0.307**<br>0.473<br>**0.710**<br>0.776<br>**1.000**<br>**1.000**<br>**0.516**<br>**0.699**|



22 

## **H Limitations** 

The evaluations on 43 datasets demonstrate the proposed in-context KG foundation model’s performance and generalization across transductive and inductive settings. Nonetheless, there are several limitations and open questions. Some KGs have special facts in addition to the mainstream triple facts involving subject, object entities, and their relations. These include facts with time stamps and facts containing multi-relational aspects. The foundation model for these special KGs also deserves attention. Scalability is an open challenge faced by existing KG reasoning models. Our proposed model addresses this by extracting a few prompt graphs with a small scale to represent relations, which has been demonstrated as a scalable approach in Appendix E.2. Our evaluations on large-scale datasets containing millions of facts also confirm its scalability. In future work, we plan to further enhance the scalability by incorporating strategies such as pruning and parallelization. 

## **I Broader Impacts** 

Our work seeks to build a KG foundation model with effective, efficient, and transferable reasoning capabilities over unseen entities, relations, and even previously unseen KGs, all without requiring retraining from scratch. We believe that the proposed model has the potential to be applied in broad knowledge-driven applications, such as question-answering and recommender systems. Its ability to adapt to changes in the graph and generalize to unseen data will be beneficial in addressing issues such as cold start. Nevertheless, excessive reliance on knowledge from pre-training data and a few examples may lead to societal biases and unfairness. We have discussed the quality and potential impacts of the pre-training data in Appendix E.1. In practical applications, we also should carefully design example selection strategies to avoid potential societal biases and unfairness. 

23 

