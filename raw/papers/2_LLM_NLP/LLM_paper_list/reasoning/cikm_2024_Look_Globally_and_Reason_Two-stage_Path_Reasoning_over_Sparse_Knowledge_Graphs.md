# **Look Globally and Reason: Two-stage Path Reasoning over Sparse Knowledge Graphs** 

## Xiaolong Jin 

Saiping Guan 

Jiyao Wei 

Key Laboratory of Network Data Science and Technology, Institute of Computing Technology, Chinese Academy of Sciences Beijing, China University of Chinese Academy of Sciences Beijing, China guansaiping@ict.ac.cn 

Key Laboratory of Network Data Science and Technology, Institute of Computing Technology, Chinese Academy of Sciences Beijing, China University of Chinese Academy of Sciences Beijing, China weijiyao20z@ict.ac.cn 

Key Laboratory of Network Data Science and Technology, Institute of Computing Technology, Chinese Academy of Sciences Beijing, China University of Chinese Academy of Sciences Beijing, China jinxiaolong@ict.ac.cn 

Xueqi Cheng 

## Jiafeng Guo 

Key Laboratory of Network Data Science and Technology, Institute of Computing Technology, Chinese Academy of Sciences 

Key Laboratory of Network Data Science and Technology, Institute of Computing Technology, Chinese Academy of Sciences 

Beijing, China University of Chinese Academy of Sciences Beijing, China cxq@ict.ac.cn 

Beijing, China University of Chinese Academy of Sciences Beijing, China guojiafeng@ict.ac.cn 

## **Abstract** 

## **CCS Concepts** 

Sparse Knowledge Graphs (KGs), frequently encountered in realworld applications, contain fewer facts in the form of (head entity, relation, tail entity) compared to more populated KGs. The sparse KG completion task, which reasons answers for given queries in the form of (head entity, relation, ?) for sparse KGs, is particularly challenging due to the necessity of reasoning missing facts based on limited facts. Path-based models, known for excellent explainability, are often employed for this task. However, existing path-based models typically rely on external models to fill in missing facts and subsequently perform path reasoning. This approach introduces unexplainable factors or necessitates meticulous rule design. In light of this, this paper proposes an alternative approach by looking inward instead of seeking external assistance. We introduce a two-stage path reasoning model called LoGRe (Look Globally and Reason) over sparse KGs. LoGRe constructs a relation-path reasoning schema by globally analyzing the training data to alleviate the sparseness problem. Based on this schema, LoGRe then aggregates paths to reason out answers. Experimental results on five benchmark sparse KG datasets demonstrate the effectiveness of the proposed LoGRe model. 

• **Computing methodologies** → **Knowledge representation and reasoning** . 

## **Keywords** 

Sparse Knowledge Graph, Sparse Knowledge Graph Completion, Path Reasoning, Reasoning Schema 

## **ACM Reference Format:** 

Saiping Guan, Jiyao Wei, Xiaolong Jin, Jiafeng Guo, and Xueqi Cheng. 2024. Look Globally and Reason: Two-stage Path Reasoning over Sparse Knowledge Graphs. In _Proceedings of the 33rd ACM International Conference on Information and Knowledge Management (CIKM ’24), October 21–25, 2024, Boise, ID, USA._ ACM, New York, NY, USA, 11 pages. https: //doi.org/10.1145/3627673.3679845 

## **1 Introduction** 

Sparse Knowledge Graphs (KGs) are a special type of KGs, containing fewer facts in the form of (head entity, relation, tail entity) compared to more populated KGs. They are frequently encountered in real-world applications, such as location recommendation, fund recommendation, and fraud detection [8, 21, 27, 31]. The sparseness of sparse KGs impedes the performance improvement of such applications. Hence, sparse KG completion, formalized as reasoning answers for given queries in the form of (head entity, relation, ?) for sparse KGs, is a crucial and urgent task. 

This work is licensed under a Creative Commons Attribution International 4.0 License. 

The sparse KG completion task is challenging, as it requires reasoning missing facts based on limited available facts. Thus, the performance of existing models for KG completion drops significantly when applying them to sparse KGs [14, 20, 23, 25]. To address 

_CIKM ’24, October 21–25, 2024, Boise, ID, USA_ © 2024 Copyright held by the owner/author(s). ACM ISBN 979-8-4007-0436-9/24/10 https://doi.org/10.1145/3627673.3679845 

CIKM ’24, October 21–25, 2024, Boise, ID, USA 

Saiping Guan, Jiyao Wei, Xiaolong Jin, Jiafeng Guo, & Xueqi Cheng 

this, researchers have specially extended these models. They can be classified into embedding-based, rule-based, and path-based models. Before devising a score function to determine the likelihood of a candidate fact or candidate tail entity, embedding-based models carefully design the embedding network to incorporate additional information, such as neighboring facts [23, 25], external textual information [9, 13], and external KGs [11], which lacks explainability. Rule-based models utilize embedding-based models (and transfer learning) to derive rules from sparse KGs, which are then employed to reason out answers [18, 24, 38]. Path-based models typically use embedding-based or rule-based external models to fill in missing facts and then perform path reasoning to obtain answers [14, 32, 35]. However, these models either rely on unexplainable embeddingbased reasoning facts or require carefully designed rules. 

With these considerations in mind, this paper follows the line of path-based models for sparse KG completion but emphasizes internal exploration rather than relying on external assistance. The primary challenge lies in how to effectively address the sparseness problem. It is difficult to reason missing facts from an individual standpoint. However, from a global perspective, it is possible to identify paths leading to answers. For instance, in a sparse KG, person _𝐴_ may possess relations _𝑟_ 1 and _𝑟_ 2 while missing others; person _𝐵_ may only have relations _𝑟_ 1 and _𝑟_ 3, etc. We can group the relations of the same entity type together and obtain relatively complete relations associated with the corresponding entities. We can then extract reasoning paths for each relation and assign a score to each path. We can also identify relations shared across various entity type groups and combine them to calculate the overall scores of their paths. This process yields a pool of relation-path reasoning elements called relation-path reasoning schema in this paper, with elements categorized by entity types (i.e., type-specific relationpath reasoning schema) and a shared group (i.e., cross-type relationpath reasoning schema). Based on this schema, we aggregate the endpoints of the related paths to reason out answers for given queries in the form of (head entity, relation, ?). The underlying rationale is that individual indeterministic paths cannot be relied upon in isolation, but if they all converge towards the same answer, a sensible result can be inferred. 

Based on this simple idea, we propose a two-stage path reasoning model for sparse KG completion. It Looks Globally to construct a relation-path reasoning schema and then performs path Reasoning, thus called LoGRe. Generally, our main contributions are as follows: 

- For the first time in the line of path-based sparse KG completion models, we propose to look inward rather than relying on external assistance. 

- We introduce a two-stage path reasoning model, LoGRe, for sparse KG completion, which constructs a relation-path reasoning schema from a global perspective and then aggregates paths to reason out answers. 

- Experimental results on five benchmark datasets and further analyses validate the superiority of LoGRe. 

## **2 Problem Statement** 

Definition 1. _**KG** is a graph of entities 𝐸 and their relations 𝑅 with facts represented as_ ( _ℎ,𝑟,𝑡_ ) _, where_ { _ℎ,𝑡_ } ∈ _𝐸 and 𝑟_ ∈ _𝑅, indicating the head entity ℎ has the relation 𝑟 with the tail entity 𝑡._ 

Definition 2. _**Sparse KG** is a special type of KG, where entities are less connected, containing fewer facts compared to a regular KG._ 

Sparse KG is a relative concept compared to KG. As introduced in DacKGR [14], a KG is considered dense or normal if the average degree of its entities exceeds a certain threshold; otherwise, it is considered sparse. There is no common consensus on the precise value of this threshold. 

Definition 3. _**Sparse KG completion** is the task of reasoning missing facts based on limited facts available in a sparse KG. For example, given a query_ ( _ℎ,𝑟,_ ?) _, it is to determine answers_[1] _, also called correct tail entities in this paper._ 

## **3 Related Works** 

This paper handles sparse KG completion. Its literature can be divided into embedding-based, rule-based, and path-based models. 

## **3.1 Embedding-based Models** 

Embedding-based KG completion models learn embeddings of entities and relations, and devise a score function to determine the likelihood of a candidate fact or candidate tail entity through vector operation. For example, TuckER [2] utilize multiplication operations. More complexly, ConvE [6] reshapes and concatenates the embeddings of the head entity and relation to form a matrix, where a 2D convolution operation is applied to predict the tail entity. InteractE [28] further enhances the interaction between the head entity and relation through feature permutation, feature reshaping, and circular convolution. NBFNet [39] adopts a graph neural network to predict the conditional likelihood of the tail entity. 

To address the sparseness problem of sparse KGs, researchers incorporate additional information into their models. RelaGraph [23] extends neighborhood information during entity encoding and incorporates neighborhood relations to mine deeper graph structure information. KRACL [25] introduces a knowledge relational attention network to integrate neighboring facts and proposes a knowledge contrastive loss that considers more negative samples. Liu et al. [13] fine-tuned pretrained embeddings of textual information based on the sparse KG to compensate for the lack of structure information. He et al. [9] trained existing structure-based and textbased models, and fused knowledge acquired by them. Jia et al. [11] even linked to external KGs to introduce additional entity information and employed graph convolutional neural networks to aggregate the information. 

## **3.2 Rule-based Models** 

Rule-based KG completion models deduce logical rules based on the graph structure of KG and use these rules to reason out answers for given queries. For example, NTP [22] introduces neural networks for end-to-end differentiable proving of queries to KG, where rules are induced using gradient descent. RlvlR [17] proposes a target relation-oriented sampling method and a matrix operationsbased rule evaluation mechanism to mine massive rules efficiently. RuLES [10] iteratively extends rules induced from a KG by relying on the feedback from a precomputed embedding model over the KG 

> 1The query (? _,𝑟,𝑡_ ) can be transformed to ( _ℎ,𝑟,_ ?) by introducing inverse relations. We follow the line of path-based models focusing on ( _ℎ,𝑟,_ ?) [14, 32]. 

CIKM ’24, October 21–25, 2024, Boise, ID, USA 

Look Globally and Reason: Two-stage Path Reasoning over Sparse Knowledge Graphs 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/cikm_2024_Look_Globally_and_Reason_Two-stage_Path_Reasoning_over_Sparse_Knowledge_Graphs_images/cikm_2024_Look_Globally_and_Reason_Two-stage_Path_Reasoning_over_Sparse_Knowledge_Graphs.pdf-0003-02.png)


**----- Start of picture text -----**<br>
Stage1: Construction of relation-path reasoning schema Stage2: Path reasoning<br>Query: ℎ, 𝑟, ? , reasoning paths:  𝑝1, 𝑝2, ⋯ .<br>Cross-type 𝑟2: [𝑝21, 𝑦21, 𝑝22, 𝑥21, 𝑞22, 𝑦22,⋯ ]<br>Reasoning 𝑟3: [𝑞31, 𝑝31,𝑝32, ⋯ ] 𝑝2<br>schema ⋯ 𝑡1<br>𝑝4<br>ℎ<br>Bottom up<br>𝑝5<br>𝑝3 𝑡2<br>Person Organization<br>𝑟1: [𝑝11,𝑝12, ⋯] 𝑟2:[𝑝21, 𝑞22, ⋯]<br>𝑟2:[𝑝21, 𝑝22,⋯] 𝑟3:[𝑞31, 𝑝32, ⋯] 𝑝1<br>Type-specificReasoning 𝑟3:[𝑝31 ⋯ , 𝑝32,⋯] 𝑟4:[𝑝41⋯, 𝑝42, ⋯] 𝑡3 ⋯<br>schema Location Work of art The scores of candidate tail entities:<br>𝑟 𝑟 2 5 :[𝑥 :[𝑝 21 51 , 𝑞 , 𝑝 22 52 , ⋯] ,⋯] 𝑟 𝑟 2 7 :[𝑦 :[𝑝 21 71 , 𝑦 , 𝑝 22 72 , ⋯] ,⋯] 𝑠 𝑡1 = 𝑠 𝑝2 + 𝑠 𝑝4<br>𝑟6:[𝑝61, 𝑝62,⋯] ⋯ 𝑠 𝑡2 = 𝑠 𝑝3 + 𝑠 𝑝5<br>⋯ ⋯ 𝑠 𝑡3 = 𝑠 𝑝1<br>⋯<br>, 𝑞, 𝑥, 𝑦: paths, 𝑟: relation, and paths are ordered reversely by scores. 𝑞, 𝑥, 𝑦: paths, 𝑟: relation, and paths are ordered reversely by scores., 𝑥, 𝑦: paths, 𝑟: relation, and paths are ordered reversely by scores. 𝑥, 𝑦: paths, 𝑟: relation, and paths are ordered reversely by scores., 𝑦: paths, 𝑟: relation, and paths are ordered reversely by scores. 𝑦: paths, 𝑟: relation, and paths are ordered reversely by scores.: paths, 𝑟: relation, and paths are ordered reversely by scores. 𝑟: relation, and paths are ordered reversely by scores.: relation, and paths are ordered reversely by scores.<br>**----- End of picture text -----**<br>


𝑝, 𝑞, 𝑥, 𝑦: paths, 𝑟: relation, and paths are ordered reversely by scores. 𝑞, 𝑥, 𝑦: paths, 𝑟: relation, and paths are ordered reversely by scores., 𝑥, 𝑦: paths, 𝑟: relation, and paths are ordered reversely by scores. 𝑥, 𝑦: paths, 𝑟: relation, and paths are ordered reversely by scores., 𝑦: paths, 𝑟: relation, and paths are ordered reversely by scores. 𝑦: paths, 𝑟: relation, and paths are ordered reversely by scores.: paths, 𝑟: relation, and paths are ordered reversely by scores. 𝑟: relation, and paths are ordered reversely by scores.: relation, and paths are ordered reversely by scores. 

**Figure 1: The framework of the proposed LoGRe model.** 

and external information, including text corpora. AnyBURL [16] introduces Reinforcement Learning (RL) to guide the rule mining process. Based on the rules learned from AnyBURL [16], Ott et al. [19] further proposed a supervised approach to learn a reweighted confidence value for each rule given a specific aggregation function. 

To mitigate the impact of missing facts in sparse KGs on rule mining, researchers seek assistance from other models. Zhao et al. [38] proposed an iterative framework of rule learning and embedding, where rules are inferred and scored based on the initial embeddings of entities and relations. New facts are then inferred from rules and embedded into the model. Omran et al. [18] employed transfer learning to address the sparseness issue. They introduced embedding similarities of entities, relations, and rules to automatically select the most relevant source KGs and rules for transfer. Sun et al. [24] further constructed mappings between the source KG and the target KG to transfer rules from the source KG to the target KG. 

## **3.3 Path-based Models** 

Path-based KG completion models perform path exploration on KG, starting from the given head entity, to reach answers with the guidance of the given relation. For example, Das et al. [4, 5] applied paths from similar entities of the given head entity to reason out tail entities. RuleGuider [12] leverages high-quality rules to provide reward supervision for RL agents. In contrast, CPL [7] trains two collaborative RL agents jointly: a fact extractor and a multi-hop reasoner. The extractor generates facts from corpora to enrich KG, while the reasoner offers feedback to the extractor and guides it towards promoting facts helpful for the reasoning. 

To address the challenges in selecting correct paths due to the lack of facts in sparse KGs, researchers usually use external models to fill in missing facts before path reasoning. DacKGR [14] dynamically adds edges based on embedding-based models as additional 

actions during the path search in the RL framework. SparKGR [32] employs rule guidance to complete missing paths, augmenting the action space for the RL agent. Moreover, it incorporates global information from KG through iterative optimization of rule induction and fact reasoning to guide the exploration of the RL agent. Whereas, SQUIRE [1], SelfHier [37], and DT4KGR [33] learn to generate paths and answers, and apply AnyBURL [16] to find evidential paths for model training. Hi-KnowE [35] conducts hierarchical RL, with high-level policy to reason relations and low-level policy to obtain tail entities. The relation action space is enlarged under rule guidance from AnyBURL [16] and the top results from ConvE [6] are used to refine the entity action space. 

## **3.4 Summarization and Comparison** 

In general, there have been numerous studies for KG completion, with fewer and more recent studies specializing in sparse KGs. Among them, path-based models are anticipated to offer superior explainability compared to embedding-based models, while obviating the need for meticulously designed rules compared to rule-based models. However, existing path-based sparse KG completion models necessitate external assistance from other models to tackle the sparseness problem. They introduce unexplainable embeddings or rely on high-quality rules, deviating from the original intention of path-based models. Thus, this paper proposes an alternative approach by internally addressing the sparseness problem instead of seeking external assistance. 

## **4 The Proposed LoGRe Model** 

We propose a two-stage model called LoGRe to look inward without reliance on external models. As depicted in Figure 1, it looks globally to construct a relation-path reasoning schema from the training data and then aggregates paths to reason. 

CIKM ’24, October 21–25, 2024, Boise, ID, USA 

Saiping Guan, Jiyao Wei, Xiaolong Jin, Jiafeng Guo, & Xueqi Cheng 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/cikm_2024_Look_Globally_and_Reason_Two-stage_Path_Reasoning_over_Sparse_Knowledge_Graphs_images/cikm_2024_Look_Globally_and_Reason_Two-stage_Path_Reasoning_over_Sparse_Knowledge_Graphs.pdf-0004-02.png)


**----- Start of picture text -----**<br>
Type-specific reasoning schema Type-specific and cross-type reasoning schemas<br>{Person: { {cross-type: {<br>𝑐𝑜𝑢𝑛𝑡𝑟𝑦: [ 𝑐𝑜𝑢𝑛𝑡𝑟𝑦: [<br>[𝑒𝑑𝑢𝑐𝑎𝑡𝑒𝑑𝑎𝑡, 𝑐𝑜𝑢𝑛𝑡𝑟𝑦], [𝑙𝑜𝑐𝑎𝑡𝑒𝑑𝑖𝑛, 𝑐𝑜𝑢𝑛𝑡𝑟𝑦],<br>Leonardo da Vinci [𝑒𝑑𝑢𝑐𝑎𝑡𝑒𝑑𝑎𝑡, 𝑙𝑜𝑐𝑎𝑡𝑒𝑑𝑖𝑛, 𝑐𝑜𝑢𝑛𝑡𝑟𝑦]], [𝑙𝑜𝑐𝑎𝑡𝑒𝑑𝑖𝑛 [−1] , 𝑐𝑜𝑢𝑛𝑡𝑟𝑦],<br>𝑒𝑑𝑢𝑐𝑎𝑡𝑒𝑑𝑎𝑡 𝑐𝑜𝑢𝑛𝑡𝑟𝑦 𝑒𝑑𝑢𝑐𝑎𝑡𝑒𝑑𝑎𝑡: [ [𝑒𝑑𝑢𝑐𝑎𝑡𝑒𝑑𝑎𝑡, 𝑐𝑜𝑢𝑛𝑡𝑟𝑦],<br>[𝑐𝑜𝑢𝑛𝑡𝑟𝑦, 𝑐𝑜𝑢𝑛𝑡𝑟𝑦 [−1] ], [𝑒𝑑𝑢𝑐𝑎𝑡𝑒𝑑𝑎𝑡 [−1] , 𝑐𝑜𝑢𝑛𝑡𝑟𝑦],<br>[𝑐𝑜𝑢𝑛𝑡𝑟𝑦, 𝑐𝑜𝑢𝑛𝑡𝑟𝑦 [−1] , 𝑙𝑜𝑐𝑎𝑡𝑒𝑑𝑖𝑛 [−1] ]]}, [𝑒𝑑𝑢𝑐𝑎𝑡𝑒𝑑𝑎𝑡, 𝑙𝑜𝑐𝑎𝑡𝑒𝑑𝑖𝑛, 𝑐𝑜𝑢𝑛𝑡𝑟𝑦]]},<br>𝑐𝑜𝑢𝑛𝑡𝑟𝑦 Organization: { Person: {<br>𝑐𝑜𝑢𝑛𝑡𝑟𝑦: [ 𝑒𝑑𝑢𝑐𝑎𝑡𝑒𝑑𝑎𝑡: [<br>Italy<br>[𝑙𝑜𝑐𝑎𝑡𝑒𝑑𝑖𝑛, 𝑐𝑜𝑢𝑛𝑡𝑟𝑦], [𝑐𝑜𝑢𝑛𝑡𝑟𝑦, 𝑐𝑜𝑢𝑛𝑡𝑟𝑦 [−1] ],<br>University of Florence [𝑒𝑑𝑢𝑐𝑎𝑡𝑒𝑑𝑎𝑡 [−1] , 𝑐𝑜𝑢𝑛𝑡𝑟𝑦]], [𝑐𝑜𝑢𝑛𝑡𝑟𝑦, 𝑐𝑜𝑢𝑛𝑡𝑟𝑦 [−1] , 𝑙𝑜𝑐𝑎𝑡𝑒𝑑𝑖𝑛 [−1] ]]},<br>𝑐𝑜𝑢𝑛𝑡𝑟𝑦 𝑙𝑜𝑐𝑎𝑡𝑒𝑑𝑖𝑛: [ Organization: {<br>𝑙𝑜𝑐𝑎𝑡𝑒𝑑𝑖𝑛 [𝑐𝑜𝑢𝑛𝑡𝑟𝑦, 𝑐𝑜𝑢𝑛𝑡𝑟𝑦 [−1] ], 𝑙𝑜𝑐𝑎𝑡𝑒𝑑𝑖𝑛: [<br>[𝑒𝑑𝑢𝑐𝑎𝑡𝑒𝑑𝑎𝑡 [−1] , 𝑐𝑜𝑢𝑛𝑡𝑟𝑦, 𝑐𝑜𝑢𝑛𝑡𝑟𝑦 [−1] ]]}, [𝑐𝑜𝑢𝑛𝑡𝑟𝑦, 𝑐𝑜𝑢𝑛𝑡𝑟𝑦 [−1] ],<br>Geo-political entity: { [𝑒𝑑𝑢𝑐𝑎𝑡𝑒𝑑𝑎𝑡 [−1] , 𝑐𝑜𝑢𝑛𝑡𝑟𝑦, 𝑐𝑜𝑢𝑛𝑡𝑟𝑦 [−1] ]]}<br>Florence 𝑐𝑜𝑢𝑛𝑡𝑟𝑦 : [ }<br>[𝑙𝑜𝑐𝑎𝑡𝑒𝑑𝑖𝑛 [−1] , 𝑐𝑜𝑢𝑛𝑡𝑟𝑦]]}}<br>**----- End of picture text -----**<br>


**Figure 2: A small example of the construction process for the relation-path reasoning schema. Dark red, yellow, and blue represent the entities of type person, organization, and geo-political entity, respectively.** 

## **4.1 Stage1: Construction of Relation-path Reasoning Schema** 

This stage globally analyzes the training data to obtain type-specific relation-path reasoning schema (referred to as type-specific reasoning schema), and finally relation-path reasoning schema, which encompasses type-specific and cross-type ones. For clarity, we present a small example of this stage in Figure 2. Now, Let us elaborate on this stage. 

_4.1.1 Type-specific Reasoning Schema._ **The Construction Process.** Type-specific relation-path reasoning schema is constructed as follows: 

- For each entity _𝑒_ ∈ _𝐸_ , we randomly collect a maximum number of _𝑁𝑝𝑎𝑡ℎ_ paths[2] from the training data to construct the entity-path dictionary _𝐷_ . To avoid the time-consuming collection of excessively long paths that may not contribute to the reasoning, the hop of each path is restricted to a maximum of _𝑁_ . _ℎ𝑜𝑝_ 

- Entities are divided into groups based on their types, such as person, organization, and location. 

- For each group of index _𝑖_ , we gather the relation set _𝑅𝑖_ possessed by its entities _𝐸𝑖_ . 

- For each relation _𝑟 𝑗_ ∈ _𝑅𝑖_ , we retrieve the specific paths from the entity-path dictionary _𝐷_ by the entities in _𝐸𝑖_ that have _𝑟 𝑗_ . These paths should lead to the correct tail entities in the training data. 

Thus, the relations and their corresponding paths constitute an initial type-specific relation-path reasoning schema. As illustrated in the left part of Figure 1 and Figure 2, its relation-path reasoning elements grouped by entity types, are in the form of: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/cikm_2024_Look_Globally_and_Reason_Two-stage_Path_Reasoning_over_Sparse_Knowledge_Graphs_images/cikm_2024_Look_Globally_and_Reason_Two-stage_Path_Reasoning_over_Sparse_Knowledge_Graphs.pdf-0004-12.png)


where _𝑝 𝑗𝑘_ ( _𝑘_ = 1 _,_ 2 _,_ · · · ) is the _𝑘_ -th path of relation _𝑟 𝑗_ in the group, which satisfies the two conditions mentioned above: At least one 

> 2If the number of paths for an entity is less than _𝑁𝑝𝑎𝑡ℎ_ , the number of collected paths will also be less than _𝑁𝑝𝑎𝑡ℎ_ . 

entity in the group has _𝑟 𝑗_ ; path _𝑝 𝑗𝑘_ is among the collected paths of these entities in _𝐷_ , starting from them and reaching the correct tail entities of _𝑟 𝑗_ . Path _𝑝 𝑗𝑘_ is a list of relations of the corresponding exploration in the sparse KG, i.e., 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/cikm_2024_Look_Globally_and_Reason_Two-stage_Path_Reasoning_over_Sparse_Knowledge_Graphs_images/cikm_2024_Look_Globally_and_Reason_Two-stage_Path_Reasoning_over_Sparse_Knowledge_Graphs.pdf-0004-16.png)


where _𝑟 𝑗𝑘𝑙_ ( _𝑙_ = 1 _,_ 2 _,_ · · · ) is the _𝑙_ -th relation in _𝑝 𝑗𝑘_ . 

**Path Score Calculation.** To facilitate path reasoning, we calculate a score for each path. For sparse KGs, where facts and certain paths are missing, we think path precision is crucial. Thus, the score _𝑠_ ( _𝑝 𝑗𝑘_ ) of path _𝑝 𝑗𝑘_ is defined as its precision: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/cikm_2024_Look_Globally_and_Reason_Two-stage_Path_Reasoning_over_Sparse_Knowledge_Graphs_images/cikm_2024_Look_Globally_and_Reason_Two-stage_Path_Reasoning_over_Sparse_Knowledge_Graphs.pdf-0004-19.png)


where M _𝑗𝑘_ is the number of occurrences of _𝑝 𝑗𝑘_ for the entities in the group and N _𝑗𝑘_ is the number of times that _𝑝 𝑗𝑘_ reaches the correct tail entities of _𝑟 𝑗_ . 

_4.1.2 Cross-type Reasoning Schema._ **The Motivation and Construction Process.** Some relations may appear in multiple entity type groups. For instance, the entity types of person and organization have common attributes (also treated as relations, as they are handled in the same way), such as _𝑡𝑒𝑙𝑒𝑝ℎ𝑜𝑛𝑒𝑛𝑢𝑚𝑏𝑒𝑟_ and _𝑎𝑑𝑑𝑟𝑒𝑠𝑠_ . Similarly, relations like _𝑐𝑜𝑢𝑛𝑡𝑟𝑦_ are possessed by the entity types of person, organization, and geo-political entity, as exemplified in Figure 2. To handle this, we adopt an approach inspired by the bottom-up construction process of KG schema, where common relations of entity types are moved to their parent type. Analogically, we move the common relations of certain entity types to the cross-type group, which is considered as their parent group. As depicted in the left part of Figure 1, relation _𝑟_ 2, appearing in all the four groups, is moved to the top cross-type group. Likewise, relation _𝑟_ 3, shared by person and organization, is also relocated. Illustratively, as demonstrated in Figure 2, _𝑐𝑜𝑢𝑛𝑡𝑟𝑦_ possessed by person, organization, and geo-political entity is shifted to the top cross-type group. The corresponding paths are combined: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/cikm_2024_Look_Globally_and_Reason_Two-stage_Path_Reasoning_over_Sparse_Knowledge_Graphs_images/cikm_2024_Look_Globally_and_Reason_Two-stage_Path_Reasoning_over_Sparse_Knowledge_Graphs.pdf-0004-22.png)


CIKM ’24, October 21–25, 2024, Boise, ID, USA 

Look Globally and Reason: Two-stage Path Reasoning over Sparse Knowledge Graphs 

where _𝑃_ ( _𝑟_ ) is the path set of the cross-type relation _𝑟_ , ∪ is the union operation, _𝑛_ is the number of the entity type groups having _𝑟_ , and _𝑃_ ( _𝑟_ ) _𝑖_ is the path set of _𝑟_ in its _𝑖_ -th entity type group. 

**Path Score Calculation for the Cross-type Relations.** The path scores of the cross-type relations are aggregated from the entity types where they appear. For a path _𝑝_ of the cross-type relation _𝑟_ , its score is: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/cikm_2024_Look_Globally_and_Reason_Two-stage_Path_Reasoning_over_Sparse_Knowledge_Graphs_images/cikm_2024_Look_Globally_and_Reason_Two-stage_Path_Reasoning_over_Sparse_Knowledge_Graphs.pdf-0005-04.png)


where _𝑛_[′] is the number of the entity type groups in which _𝑟_ and _𝑝_ co-appear in their relation-path reasoning elements, M _𝑖_ is the number of occurrences of _𝑝_ for the _𝑖_ -th group, and N _𝑖_ is the number of times that _𝑝_ reaches the correct tail entities of _𝑟_ . 

_4.1.3 Relation-path Reasoning Schema._ After shifting common relations to the cross-type group and aggregating their paths and scores, we obtain the final relation-path reasoning schema consisting of type-specific and cross-type ones. The paths of each relation are sorted in reverse order by their scores. Before sorting, these scores are adjusted based on the following consideration: Although for sparse KG, missing facts often make the model need to try more hops when short paths are unavailable, short paths are preferred. Actually, rules are typically short in length. Short paths are more likely to be rules. Therefore, we introduce a hop decay to the score of each path _𝑝_ : 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/cikm_2024_Look_Globally_and_Reason_Two-stage_Path_Reasoning_over_Sparse_Knowledge_Graphs_images/cikm_2024_Look_Globally_and_Reason_Two-stage_Path_Reasoning_over_Sparse_Knowledge_Graphs.pdf-0005-07.png)


where _𝑙𝑒𝑛_ ( _𝑝_ ) is the length of _𝑝_ , i.e., the number of hops of _𝑝_ , and _𝑓_ ( _𝑙𝑒𝑛_ ( _𝑝_ )) is defined as: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/cikm_2024_Look_Globally_and_Reason_Two-stage_Path_Reasoning_over_Sparse_Knowledge_Graphs_images/cikm_2024_Look_Globally_and_Reason_Two-stage_Path_Reasoning_over_Sparse_Knowledge_Graphs.pdf-0005-09.png)


where _𝑑_ is the hop decay factor. The score is multiplied by _𝑑_ for each increase in the number of hops. 

Generally, the construction of the relation-path reasoning schema is a bottom-up approach. It thoroughly explores the training data, thereby alleviating the sparseness problem of sparse KGs. Besides, it enables us to obtain more precise path scores by considering paths from a global perspective. 

## **4.2 Stage2: Path Reasoning** 

Based on the relation-path reasoning schema, we perform path reasoning and get the scores of candidate tail entities. 

_4.2.1 The Reasoning Process._ In particular, we perform path reasoning as follows: 

- For each given query ( _ℎ,𝑟,_ ?), we retrieve the corresponding reasoning paths [ _𝑝_ 1 _, 𝑝_ 2 _,_ · · · ] by the entity type of _ℎ_ and _𝑟_ . 

- For the top _𝑁𝑡𝑜𝑝_ paths, we start from _ℎ_ and follow the paths to arrive at candidate tail entities. If a relation in the path does not appear, the process is halted midway. Here, _𝑁𝑡𝑜𝑝_ is the number of explored top paths. We try top paths instead of all paths to relieve the impact from spurious paths. 

- Sort all the candidate tail entities reversely by scores, which are introduced in the following. 

_4.2.2 The Scores of Candidate Tail Entities._ We obtain the score of each candidate tail entity by summing the scores of the paths reaching it. The rationale is that individual indeterministic paths alone cannot be reliable. However, when multiple paths converge towards the same answer, it is possible to infer a meaningful result. Specifically, for a candidate tail entity _𝑡𝑖_ , its score _𝑠_ ( _𝑡𝑖_ ) is: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/cikm_2024_Look_Globally_and_Reason_Two-stage_Path_Reasoning_over_Sparse_Knowledge_Graphs_images/cikm_2024_Look_Globally_and_Reason_Two-stage_Path_Reasoning_over_Sparse_Knowledge_Graphs.pdf-0005-19.png)


where _𝑄_ is the set of paths of relation _𝑟_ that connect _ℎ_ to _𝑡𝑖_ . 

_4.2.3 One Update to the Scores._ The scores of candidate tail entities are further adjusted with the following answer similarity: The answer should be similar to other tail entities with the same _𝑟_ and path. This is inspired by the phenomenon of analogy. It suggests that entities sharing the same _𝑟_ and path may also exhibit similarities in their tail entities. For example, these tail entities may also belong to the same group. 

Thus, the score _𝑠_ ( _𝑡𝑖_ ) is multiplied by the similarity of the candidate and other similar tail entities: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/cikm_2024_Look_Globally_and_Reason_Two-stage_Path_Reasoning_over_Sparse_Knowledge_Graphs_images/cikm_2024_Look_Globally_and_Reason_Two-stage_Path_Reasoning_over_Sparse_Knowledge_Graphs.pdf-0005-23.png)


where _𝑇_ is the set of other similar tail entities sharing the same _𝑟_ and path, and _𝑐𝑜𝑠_  𝑠𝑖𝑚_ ( _𝑡𝑖,𝑡_[′] ) is the cosine similarity between the vectors of _𝑡𝑖_ and _𝑡_[′] . The maximum similarity is used to mitigate the impact of spurious paths, which arrive at the answer coincidentally. To keep the explainability of LoGRe, we represent each entity as a | _𝑅_ |-hot vector, where | _𝑅_ | is the size of the relation set _𝑅_ . An entry in the vector is set to 1 if the entity has that relation with some entities; otherwise, it is set to 0. 

Then, the candidates are sorted by their scores and candidates with higher scores are more likely to be the correct tail entities. 

## **4.3 The Explainability of LoGRe** 

The proposed LoGRe model is inherently explainable in its design. Moreover, for each given query, it outputs not only the candidate tail entities but also the reasoning paths with scores for each candidate tail entity. As illustrated in the right part of Figure 1, for candidate tail entity _𝑡_ 1, the reasoning paths _𝑝_ 2 and _𝑝_ 4 with scores reveal the step-by-step processes from the head entity _ℎ_ to _𝑡_ 1 via the relations. 

## **5 Experiments** 

We evaluate LoGRe on five benchmark sparse KG datasets and conduct comprehensive analyses to examine its performance. 

## **5.1 Datasets** 

The adopted sparse KG datasets include FB15K-237-10%, FB15K237-20%, FB15K-237-50%, NELL23K, and WD-singer [14]. The first three datasets randomly retain 10%, 20%, and 50% facts from FB15K237 [26], respectively. WD-singer is a dataset of singer domain extracted from Wikidata [30]. NELL23K is a randomly sampled dataset from NELL [3]. The detailed statistics of these datasets are listed in Table 1, where | _𝐸_ |, | _𝑇𝑟𝑎𝑖𝑛_ |, | _𝑉𝑎𝑙𝑖𝑑_ |, and | _𝑇𝑒𝑠𝑡_ | are the sizes of the entity set, training set, validation set, and test set, respectively; Degree is the average degree of the entities in the corresponding training set. 

CIKM ’24, October 21–25, 2024, Boise, ID, USA 

Saiping Guan, Jiyao Wei, Xiaolong Jin, Jiafeng Guo, & Xueqi Cheng 

**Table 1: The statistics of the five experimental datasets.** 

|Dataset||_𝐸_|||_𝑅_|||_𝑇𝑟𝑎𝑖𝑛_|||_𝑉𝑎𝑙𝑖𝑑_|||_𝑇𝑒𝑠𝑡_||Degree|
|---|---|---|---|---|---|---|
|FB15K-237-10%<br>FB15K-237-20%<br>FB15K-237-50%<br>NELL23K<br>WD-singer|11,512<br>13,166<br>14,149<br>22,925<br>10,282|237<br>237<br>237<br>200<br>131|27,211<br>54,423<br>136,057<br>25,445<br>15,906|15,624<br>16,963<br>17,449<br>4,961<br>2,084|18,150<br>19,776<br>20,324<br>4,952<br>2,134|4.727<br>8.267<br>19.232<br>2.220<br>3.094|



## **5.2 Experimental Settings** 

_5.2.1 Baselines and Evaluation Metrics._ Baselines comprise three types of representative or state-of-the-art models. The first type is path-based models, of the same type with LoGRe, including DacKGR [14], SparKGR [32], DT4KGR [33], and Hi-KnowE [35]. The second type is rule-based models[3] , including NTP [22], RlvlR [17], and AnyBURL [16]. We also refer to embedding-based models, including TuckER [2], ConvE [6], InteractE [28], NBFNet [39], KRACL [25], and Ref. [11]. Lastly, we compare to ChatGPT[4] , the prominent large language model. 

For every fact ( _ℎ,𝑟,𝑡_ ) in the test set, we convert it to a query ( _ℎ,𝑟,_ ?) and apply models to get the ranking list of candidate tail entities, following Refs. [14, 32]. The standard Mean Reciprocal Rank (MRR) and Hits@K are adopted as evaluation metrics for model comparison. MRR is the mean reciprocal rank of all the correct tail entities among the candidates and Hits@K is the proportion of correct tail entities ranking within the top K positions. Higher values of MRR and Hits@K indicate better performance. 

_5.2.2 Implementation Details._ **Entity Types.** LoGRe requires entity types[5] to construct the relation-path reasoning schema and perform path reasoning. For the FB15K-237 series, we retrieve entity types from FB15K-237 [34]. For NELL23K, entity types are obtained from entity names, as they are in the form of “concept_type_entity”. For WD-singer, we obtain entity types from Wikidata [30] via the tail entities of relation “ _𝑖𝑛𝑠𝑡𝑎𝑛𝑐𝑒𝑜𝑓_ ”. When there are multiple types, we select the most frequent one[6] as the type to mitigate the impact of noisy data. If no type is available for an entity, it is assigned to the type “others”. 

**Hyper-parameters.** The hyper-parameters of LoGRe are selected from the following ranges in terms of MRR: The maximum number of collected paths _𝑁𝑝𝑎𝑡ℎ_ ∈{1000 _,_ 5000 _,_ 10000 _,_ 15000 _,_ 20000}, the maximum number of path hops _𝑁ℎ𝑜𝑝_ ∈{3 _,_ 4 _,_ 5 _,_ 6}, the number of explored top paths _𝑁𝑡𝑜𝑝_ ∈{100 _,_ 500 _,_ 1000}, and the hop decay factor _𝑑_ ∈{0 _._ 95 _,_ 0 _._ 9 _,_ 0 _._ 8 _,_ 0 _._ 7 _,_ · · · _,_ 0 _._ 1}. The applied settings are: _𝑁𝑝𝑎𝑡ℎ_ = 20000, _𝑁ℎ𝑜𝑝_ = 6, _𝑁𝑡𝑜𝑝_ = 1000, _𝑑_ = 0 _._ 95 for FB15K-237-10%, _𝑁𝑝𝑎𝑡ℎ_ = 5000, _𝑁ℎ𝑜𝑝_ = 5, _𝑁𝑡𝑜𝑝_ = 500, _𝑑_ = 0 _._ 6 for FB15K-237-20%, _𝑁𝑝𝑎𝑡ℎ_ = 1000, _𝑁ℎ𝑜𝑝_ = 4, _𝑁𝑡𝑜𝑝_ = 100, _𝑑_ = 0 _._ 8 for FB15K-237-50%, _𝑁𝑝𝑎𝑡ℎ_ = 10000, _𝑁ℎ𝑜𝑝_ = 6, _𝑁𝑡𝑜𝑝_ = 100, _𝑑_ = 0 _._ 5 for NELL23K, and _𝑁𝑝𝑎𝑡ℎ_ = 20000, _𝑁ℎ𝑜𝑝_ = 6, _𝑁𝑡𝑜𝑝_ = 100, _𝑑_ = 0 _._ 2 for WD-singer. 

3The latest rule-based model Ref. [24] for sparse KG completion is not applied as a baseline, as it necessitates an additional source KG for rule transfer. 4https://openai.com/blog/chatgpt/ 

5If all entity types are unavailable to obtain, entities can be grouped using various clustering algorithms, such as hierarchical clustering, where entities sharing numerous relations are clustered into a group. 6For the FB15K-237 series, “/common/topic” is removed before selection, as almost all entities belong to this type. 

**Baseline Results.** Baseline results reported in [32] are used, while the results of Ref. [11], DT4KGR [33], and Hi-KnowE [35][7] are obtained from the original papers. We further tune the hyperparameters of KRACL [25] on the five experimental datasets to get the best results. For ChatGPT, we follow the state-of-the-art LambdaKG [36] to construct the prompt, which comprises the task description with candidates, demonstrations, and test information: 

- The task description is: “Given head entity and relation, predict the tail entity from the following candidates or others: _𝐶_ ”. Here, _𝐶_ is the top 100 most relevant entities from the training set, obtained by the retrieval model BM25 [15]. 

- Similarly, the demonstrations utilize the top 5 most similar training data as examples. 

- The test information is “What is the _𝑟_ of _ℎ_ ?”, given the query ( _ℎ,𝑟,_ ?). 

During this process, the names of each entity and relation are required. We get the entity names and relation names of the FB15K237 series from LambdaKG [36]. Those of NELL23K are obtained by removing the “concept” prefix and segmenting the remaining characters, while those of WD-singer are retrieved from Wikidata [30]. For a fair comparison, ChatGPT solely utilizes these fundamental entity and relation names, without any supplementary information. 

## **5.3 Results of Sparse KG Completion** 

The experimental results are presented in Tables 2 and 3. For the experiments on ChatGPT, we focus on FB15K-237-50%, which has the largest test set among the FB15K-237 series, NELL23K, and WD-singer. As we can only get Hits@1 by evaluating the accuracy of the responses from ChatGPT, we adopt only Hits@1 following LambdaKG [36]. For the other baselines, we utilize all the five datasets and follow Refs. [11, 14, 32] to apply MRR and Hits@{3, 10} as metrics. From the results, we have the following observations: 

**Compared with rule-based and path-based models,** LoGRe outperforms them (except for Hi-KnowE [35]), particularly on the challenging dataset NELL23K, where the state-of-the-art baselines gain the worst performance among the five datasets. LoGRe significantly improves all the metrics on NELL23K, 0.056 on MRR (27.6% relative improvement), 5.70% on Hits@3 (25.7% relative improvement), and 7.80% on Hits@10 (23.0% relative improvement). LoGRe is better or comparable than much more intricate Hi-KnowE [35], which relies on AnyBURL [16], ConvE [6], and Transformer [29]. This validates the effectiveness of LoGRe looking inward globally. Additionally, state-of-the-art path-based models perform better than state-of-the-art rule-based ones. The poor performance of rule-based models may be due to the large number of truncated paths in sparse KG, which prevents these models from effective rule mining. Whereas, path-based models have more flexibility in exploring KG rather than relying on strict rules. This enables them to consider various possibilities for reaching the correct tail entities. LoGRe further aggregates and evaluates the paths and candidate tail entities from a global perspective. Thus, LoGRe is more capable of helping the head entities select more high-quality paths to arrive at the correct tail entities than path-based baselines. 

> 7DT4KGR and Hi-KnowE are available online on 19 January 2024 and 11 April 2024, respectively, lacking source codes, and with certain implementation details undisclosed. Thus, some of their results are missing in Table 2. 

CIKM ’24, October 21–25, 2024, Boise, ID, USA 

Look Globally and Reason: Two-stage Path Reasoning over Sparse Knowledge Graphs 

**Table 2: Experimental results in terms of MRR and Hits@{3, 10} (%). The best scores of rule-based (the second block) and path-based models (the third block) are in bold, and the best scores of embedding-based models (the first block) are underlined.** 

|Model|FB15K-237-10%|FB15K-237-20%|FB15K-237-50%|NELL23K|WD-singer|
|---|---|---|---|---|---|
||MRR Hits@3 Hits@10|MRR Hits@3 Hits@10|MRR Hits@3 Hits@10|MRR Hits@3 Hits@10|MRR Hits@3 Hits@10|
|TuckER<br>ConvE<br>InteractE<br>NBFNet<br>KRACL<br>Ref. [11]|0.252<br>27.2<br>40.4<br>0.245<br>26.2<br>39.1<br>0.254<br>27.3<br>40.2<br>0.241<br>26.3<br>38.8<br>0.164<br>17.0<br>21.2<br>0.268<br>27.9<br>38.7|0.268<br>28.9<br>42.8<br>0.261<br>28.3<br>41.8<br>0.265<br>28.8<br>42.8<br>0.260<br>27.8<br>41.7<br>0.170<br>16.9<br>19.8<br>0.267<br>28.2<br>42.1|0.314<br>34.2<br>50.1<br>0.313<br>34.2<br>50.1<br>0.308<br>33.6<br>49.7<br>0.316<br>34.1<br>50.3<br>0.222<br>26.8<br>44.4<br>0.309<br>35.0<br>51.9|0.276<br>30.2<br>46.7<br>0.276<br>30.1<br>46.4<br>0.265<br>28.6<br>46.7<br>0.274<br>28.9<br>46.9<br>0.158<br>15.8<br>27.6<br>0.261<br>29.6<br>47.2|0.421<br>47.1<br>57.1<br>0.448<br>47.8<br>56.9<br>0.435<br>47.6<br>57.4<br>0.453<br>49.3<br>58.9<br>0.142<br>13.4<br>20.7<br>-<br>-<br>-|
|NTP<br>RlvlR<br>AnyBURL|0.083<br>11.4<br>16.9<br>0.107<br>12.2<br>20.6<br>0.149<br>15.5<br>26.7|0.173<br>16.1<br>21.7<br>0.132<br>15.2<br>27.1<br>0.164<br>16.7<br>29.3|0.222<br>23.1<br>30.7<br>0.199<br>20.8<br>32.4<br>0.198<br>21.3<br>35.1|0.132<br>14.9<br>24.1<br>0.152<br>17.3<br>25.0<br>0.176<br>18.5<br>25.2|0.292<br>31.1<br>44.2<br>0.374<br>32.0<br>47.6<br>0.392<br>34.1<br>48.6|
|DacKGR<br>SparKGR<br>DT4KGR<br>Hi-KnowE|0.218<br>23.9<br>33.7<br>**0.228**<br>24.5<br>35.0<br>-<br>-<br>-<br>0.224<br>**25.5**<br>34.1|0.242<br>27.2<br>38.9<br>0.252<br>27.7<br>39.1<br>0.254<br>-<br>40.1<br>0.247<br>27.7<br>38.1|0.293<br>32.0<br>45.7<br>0.292<br>32.0<br>46.2<br>**0.297**<br>-<br>46.2<br>-<br>-<br>-|0.197<br>20.0<br>31.6<br>0.203<br>22.2<br>33.9<br>-<br>-<br>-<br>-<br>-<br>-|0.377<br>42.1<br>48.5<br>0.393<br>43.7<br>50.7<br>-<br>-<br>-<br>-<br>-|
|LoGRe|**0.228**<br>24.5<br>**36.2**|**0.261**<br>**28.0**<br>**41.3**|**0.297**<br>**32.7**<br>**46.4**|**0.259**<br>**27.9**<br>**41.7**|**0.459**<br>**48.9**<br>**54.5**|



**Table 3: Comparison with ChatGPT on Hits@1 (%).** 

|Model|FB15K-237-50%|NELL23K|WD-singer|
|---|---|---|---|
|ChatGPT<br>LoGRe|14.0<br>**21.2**|10.8<br>**18.4**|32.7<br>**40.6**|



**LoGRe is comparable but worse than embedding-based models.** However, embedding-based models serve as a reference only, as they are different categories of models as opposed to our path-based model. Unlike path-based models, embedding-based models usually require tuning many more parameters carefully and are not explainable. The same phenomenon that embedding-based models outperform rule-based and path-based models is revealed in previous path-based studies [14, 32]. The reason is that embeddingbased models implicitly encoding the facts of KG into vectors, are free from rule mining and path traversing. Thus, they are less affected by path truncation in sparse KG. Combining the superior performance of embedding-based models and the explainability of path-based models is our future work. 

**Compared with ChatGPT,** LoGRe performs better by more than 7.0% on all the three datasets. We attribute this to the difficulty of sparse KG completion. As displayed in Table 4, we often need to explore multiple hops to get the correct tail entities on sparse KG. However, ChatGPT, targeting natural language understanding, cannot well handle structure and path reasoning. Thus, it is challenging for ChatGPT to get accurate results via limited context understanding. In contrast, LoGRe excels in multi-hop exploration. 

**Table 4: Average hops to the correct tail entities on test sets. FB15K-237 is denoted as FB.** 

|Dataset|FB-10%|FB-20%|FB-50%|NELL23K|WD-singer|
|---|---|---|---|---|---|
|Average hops|5.372|4.603|3.493|4.436|4.093|



## **5.4 Ablation Study** 

LoGRe has two stages, with the construction of the type-specific reasoning schema in stage 1 and stage 2 (path reasoning) being essential. Thus, we remove the construction of the cross-type reasoning schema in stage 1 to evaluate its effectiveness. This ablation is denoted as LoGRe(-cross_type). Besides, LoGRe prefers short ones via hop decay and considers the answer similarity. To study their contributions to the performance, we remove one of them each time, denoted as LoGRe(-prefer_shortpath) and LoGRe(-answer_similarity), respectively. The experimental results are reported in Table 5. 

Based on the results, we observe that removing any of them reduces the effectiveness of LoGRe, which confirms the necessity of all the three designs. Particularly, removing the construction of the cross-type reasoning schema significantly decreases the performance. It underscores the importance of shifting common relations to the cross-type group and aggregating their paths and scores globally. Additionally, not giving preference to short paths on WDsinger leads to a more obvious drop in performance compared to the other datasets. It is reasonable, as WD-singer is specific to the singer domain, while the other datasets contain general facts. Specific WD-singer may have simpler relation patterns among entities, and when there are high-quality short paths, long paths are less important. This observation aligns with the small optimal hop decay factor, i,e, 0.2, as introduced in Section 5.2.2. In contrast, on general-domain datasets like the FB15K-237 series, long paths are crucial for reasoning out the answer across entities of diverse types. 

## **5.5 Case Study** 

To illustrate the explainability of LoGRe, we conduct case study on WD-singer, following DacKGR [14]. Two representative examples, ( _𝐿𝑜𝑢𝑖𝑠𝑒𝐾𝑖𝑟𝑘𝑏𝑦𝐿𝑢𝑛𝑛,𝑐𝑖𝑡𝑖𝑧𝑒𝑛𝑠ℎ𝑖𝑝,_ ?) and ( _𝐶𝑎𝑟𝑙𝑜𝑆𝑐𝑎𝑡𝑡𝑜𝑙𝑎, 𝑝𝑎𝑟𝑡𝑖𝑐𝑖𝑝𝑎𝑛𝑡𝑖𝑛,_ ?) are selected from the test set. The first one is about person’s basic information, also selected in DacKGR [14] and the second one relates to person’s activities. We find that the corresponding correct tail entities _𝑈𝑛𝑖𝑡𝑒𝑑𝐾𝑖𝑛𝑔𝑑𝑜𝑚_ and _𝐿𝑎𝐶𝑒𝑛𝑒𝑟𝑒𝑛𝑡𝑜𝑙𝑎𝑜𝑠𝑖𝑎𝐿𝑎_ 

CIKM ’24, October 21–25, 2024, Boise, ID, USA 

Saiping Guan, Jiyao Wei, Xiaolong Jin, Jiafeng Guo, & Xueqi Cheng 

**Table 5: Ablation study results regarding MRR and Hits@{3, 10} (%).** 

|Model|FB15K-237-10%|FB15K-237-20%|FB15K-237-50%|NELL23K|WD-singer|
|---|---|---|---|---|---|
||MRR Hits@3 Hits@10|MRR Hits@3 Hits@10|MRR Hits@3 Hits@10|MRR Hits@3 Hits@10|MRR Hits@3 Hits@10|
|LoGRe(-cross_type)<br>LoGRe(-prefer_shortpath)<br>LoGRe(-answer_similarity)<br>LoGRe|0.221<br>23.7<br>35.1<br>**0.228**<br>24.4<br>36.1<br>0.227<br>24.3<br>35.9<br>**0.228**<br>**24.5**<br>**36.2**|0.257<br>27.6<br>40.9<br>**0.261**<br>**28.0**<br>41.2<br>0.260<br>27.8<br>41.1<br>**0.261**<br>**28.0**<br>**41.3**|0.294<br>32.4<br>46.3<br>0.295<br>**32.7**<br>46.3<br>0.294<br>32.3<br>46.0<br>**0.297**<br>**32.7**<br>**46.4**|0.219<br>23.2<br>36.1<br>0.254<br>27.7<br>41.3<br>0.255<br>27.3<br>41.5<br>**0.259**<br>**27.9**<br>**41.7**|0.448<br>48.1<br>53.7<br>0.415<br>46.0<br>53.4<br>0.455<br>**48.9**<br>**54.5**<br>**0.459**<br>**48.9**<br>**54.5**|



_𝑣𝑖𝑟𝑡𝑢𝑖𝑛𝑡𝑟𝑖𝑜𝑛𝑓𝑜_ ` are ranked first by LoGRe among the candidates. The used reasoning paths are presented in Table 6. For space limitation, we present the top 10 high-score paths. 

**Table 6: The top 10 high-score paths of LoGRe on two representative cases. Order**[′] **is the path order without hop decay and** _𝑟_[−][1] **is the inverse relation of** _𝑟_ **. Case 2 only has 3 paths.** 

|Query: (_𝐿𝑜𝑢𝑖𝑠𝑒𝐾𝑖𝑟𝑘𝑏𝑦𝐿𝑢𝑛𝑛,𝑐𝑖𝑡𝑖𝑧𝑒𝑛𝑠ℎ𝑖𝑝,_?)|Query: (_𝐿𝑜𝑢𝑖𝑠𝑒𝐾𝑖𝑟𝑘𝑏𝑦𝐿𝑢𝑛𝑛,𝑐𝑖𝑡𝑖𝑧𝑒𝑛𝑠ℎ𝑖𝑝,_?)|Query: (_𝐿𝑜𝑢𝑖𝑠𝑒𝐾𝑖𝑟𝑘𝑏𝑦𝐿𝑢𝑛𝑛,𝑐𝑖𝑡𝑖𝑧𝑒𝑛𝑠ℎ𝑖𝑝,_?)|
|---|---|---|
|Order|Path|Order′|
|1<br>2<br>3<br>4|(_𝑠𝑡𝑢𝑑𝑒𝑛𝑡_−1_,𝑠𝑡𝑢𝑑𝑒𝑛𝑡,𝑐𝑖𝑡𝑖𝑧𝑒𝑛𝑠ℎ𝑖𝑝_)<br>(_𝑠𝑡𝑢𝑑𝑒𝑛𝑡𝑜𝑓,𝑠𝑡𝑢𝑑𝑒𝑛𝑡𝑜𝑓_−1_,𝑐𝑖𝑡𝑖𝑧𝑒𝑛𝑠ℎ𝑖𝑝_)<br>(_𝑠𝑡𝑢𝑑𝑒𝑛𝑡𝑜𝑓,𝑠𝑡𝑢𝑑𝑒𝑛𝑡,𝑐𝑖𝑡𝑖𝑧𝑒𝑛𝑠ℎ𝑖𝑝_)<br>(_𝑠𝑡𝑢𝑑𝑒𝑛𝑡_−1_,𝑠𝑡𝑢𝑑𝑒𝑛𝑡𝑜𝑓_−1_,𝑐𝑖𝑡𝑖𝑧𝑒𝑛𝑠ℎ𝑖𝑝_)|1<br>3<br>4<br>5|
|5<br>6<br>7<br>8|(_𝑠𝑡𝑢𝑑𝑒𝑛𝑡_−1_,𝑠𝑡𝑢𝑑𝑒𝑛𝑡,𝑐𝑖𝑡𝑖𝑧𝑒𝑛𝑠ℎ𝑖𝑝,𝑐𝑖𝑡𝑖𝑧𝑒𝑛𝑠ℎ𝑖𝑝_−1_,_<br>_𝑐𝑖𝑡𝑖𝑧𝑒𝑛𝑠ℎ𝑖𝑝_)<br>(_𝑠𝑡𝑢𝑑𝑒𝑛𝑡_−1_,𝑠𝑡𝑢𝑑𝑒𝑛𝑡𝑜𝑓_−1_,𝑐𝑖𝑡𝑖𝑧𝑒𝑛𝑠ℎ𝑖𝑝,𝑐𝑖𝑡𝑖𝑧𝑒𝑛𝑠ℎ𝑖𝑝_−1_,_<br>_𝑐𝑖𝑡𝑖𝑧𝑒𝑛𝑠ℎ𝑖𝑝_)<br>(_𝑠𝑡𝑢𝑑𝑒𝑛𝑡𝑜𝑓,𝑠𝑡𝑢𝑑𝑒𝑛𝑡𝑜𝑓_−1_,𝑐𝑖𝑡𝑖𝑧𝑒𝑛𝑠ℎ𝑖𝑝,𝑐𝑖𝑡𝑖𝑧𝑒𝑛𝑠ℎ𝑖𝑝_−1_,_<br>_𝑐𝑖𝑡𝑖𝑧𝑒𝑛𝑠ℎ𝑖𝑝_)<br>(_𝑠𝑡𝑢𝑑𝑒𝑛𝑡𝑜𝑓,𝑠𝑡𝑢𝑑𝑒𝑛𝑡,𝑐𝑖𝑡𝑖𝑧𝑒𝑛𝑠ℎ𝑖𝑝,𝑐𝑖𝑡𝑖𝑧𝑒𝑛𝑠ℎ𝑖𝑝_−1_,_<br>_𝑐𝑖𝑡𝑖𝑧𝑒𝑛𝑠ℎ𝑖𝑝_)|2<br>6<br>7<br>8|
|9<br>(_𝑠𝑡𝑢𝑑𝑒𝑛𝑡𝑜𝑓,𝑠𝑡𝑢𝑑𝑒𝑛𝑡𝑜𝑓_−1_,𝑠𝑡𝑢𝑑𝑒𝑛𝑡𝑜𝑓,𝑠𝑡𝑢𝑑𝑒𝑛𝑡𝑜𝑓_−1_,_<br>_𝑐𝑖𝑡𝑖𝑧𝑒𝑛𝑠ℎ𝑖𝑝_)<br>9<br>10<br>(_𝑠𝑡𝑢𝑑𝑒𝑛𝑡𝑜𝑓,𝑠𝑡𝑢𝑑𝑒𝑛𝑡,𝑠𝑡𝑢𝑑𝑒𝑛𝑡𝑜𝑓,𝑠𝑡𝑢𝑑𝑒𝑛𝑡𝑜𝑓_−1_,_<br>_𝑐𝑖𝑡𝑖𝑧𝑒𝑛𝑠ℎ𝑖𝑝_)<br>10|||
|Query: (_𝐶𝑎𝑟𝑙𝑜𝑆𝑐𝑎𝑡𝑡𝑜𝑙𝑎, 𝑝𝑎𝑟𝑡𝑖𝑐𝑖𝑝𝑎𝑛𝑡𝑖𝑛,_?)|||
|1<br>2<br>3|(_𝑐𝑎𝑠𝑡𝑚𝑒𝑚𝑏𝑒𝑟_−1)<br>(_𝑐𝑎𝑠𝑡𝑚𝑒𝑚𝑏𝑒𝑟_−1_,𝑐𝑎𝑠𝑡𝑚𝑒𝑚𝑏𝑒𝑟, 𝑝𝑎𝑟𝑡𝑖𝑐𝑖𝑝𝑎𝑛𝑡𝑖𝑛_)<br>(_𝑐𝑎𝑠𝑡𝑚𝑒𝑚𝑏𝑒𝑟_−1_,𝑐𝑎𝑠𝑡𝑚𝑒𝑚𝑏𝑒𝑟,𝑐𝑎𝑠𝑡𝑚𝑒𝑚𝑏𝑒𝑟_−1_,_<br>_𝑐𝑎𝑠𝑡𝑚𝑒𝑚𝑏𝑒𝑟, 𝑝𝑎𝑟𝑡𝑖𝑐𝑖𝑝𝑎𝑛𝑡𝑖𝑛_)|3<br>1<br>2|



From the upper part of Table 6, it can be observed that the 5-hop path ( _𝑠𝑡𝑢𝑑𝑒𝑛𝑡_[−][1] _,𝑠𝑡𝑢𝑑𝑒𝑛𝑡, 𝑐𝑖𝑡𝑖𝑧𝑒𝑛𝑠ℎ𝑖𝑝,𝑐𝑖𝑡𝑖𝑧𝑒𝑛𝑠ℎ𝑖𝑝_[−][1] _, 𝑐𝑖𝑡𝑖𝑧𝑒𝑛𝑠ℎ𝑖𝑝_ ) ranks second when hop decay is disabled; otherwise, its ranking drops to more appropriate fifth place. This also proves the necessity of our preferring short paths. Furthermore, as _𝑠𝑡𝑢𝑑𝑒𝑛𝑡_[−][1] is equal to _𝑠𝑡𝑢𝑑𝑒𝑛𝑡𝑜𝑓_ and _𝑠𝑡𝑢𝑑𝑒𝑛𝑡𝑜𝑓_[−][1] is equal to _𝑠𝑡𝑢𝑑𝑒𝑛𝑡_ , the first four paths tell that the _𝑐𝑖𝑡𝑖𝑧𝑒𝑛𝑠ℎ𝑖𝑝_ of a person may be the same as that of his/her classmate; the middle four paths suggest that it may be the same as the _𝑐𝑖𝑡𝑖𝑧𝑒𝑛𝑠ℎ𝑖𝑝_ of someone who shares the _𝑐𝑖𝑡𝑖𝑧𝑒𝑛𝑠ℎ𝑖𝑝_ with his/her classmate; the remaining two paths imply that it may be the same as the _𝑐𝑖𝑡𝑖𝑧𝑒𝑛𝑠ℎ𝑖𝑝_ of his/her classmate’s classmate. Interestingly, LoGRe finds weak signals like the above ones and gathers them to determine the answer. In contrast, existing path-based studies typically use one path to get the answer, 

extremely difficult and indeterministic. Actually, the above indeterministic paths alone are not worthy of attention, while if many indeterministic paths point to the same answer, it makes sense. 

From the bottom part of Table 6, we have the same observation. LoGRe uses all available signals, including the deterministic signal that _𝑝𝑎𝑟𝑡𝑖𝑐𝑖𝑝𝑎𝑛𝑡𝑖𝑛_ is equal to _𝑐𝑎𝑠𝑡𝑚𝑒𝑚𝑏𝑒𝑟_[−][1] and the indeterministic signals that a person may be _𝑝𝑎𝑟𝑡𝑖𝑐𝑖𝑝𝑎𝑛𝑡𝑖𝑛_ something in which his/her co-participant participates and his/her co-participant’s coparticipant participates, to obtain a relatively deterministic answer. 

## **5.6 Hyper-parameter Analysis** 

The hyper-parameters of LoGRe are the maximum number of collected paths, maximum number of path hops, number of explored top paths, and hop decay factor. We thoroughly analyze these hyperparameters to evaluate their impacts on the performance. Without loss of generality, we also conduct the analysis on WD-singer. The results are exhibited in Figure 3. 

It can be observed from the first and third subfigures of Figure 3 that the maximum number of collected paths and number of explored top paths impact the performance of WD-singer slightly. Notably, 20000 and 100, respectively, yield marginally superior results on MRR. Concerning the second subfigure, the performance improves on all the metrics as the maximum number of path hops increases. When the maximum number of path hops is 6, LoGRe achieves the best performance. As illustrated in the fourth subfigure, a decrease in the hop decay factor leads to performance improvement. The optimal value is 0.2, indicating a substantial penalization of long paths. Overall, the performance variations of different parameter values are limited, demonstrating the robustness of LoGRe. 

## **5.7 Complexity Analyses** 

Model complexity comprises space and time aspects. We compare LoGRe with the best baselines[8] for these two dimensions. In terms of space complexity, LoGRe only necessitates storage for the relationpath reasoning schema of space complexity _𝑂_ (| _𝐸_ | _𝑁𝑝𝑎𝑡ℎ𝑁ℎ𝑜𝑝_ ). As presented in Table 7, it generally occupies more space compared to excellent embedding-based KG completion models TuckER [2], ConvE [6], InteractE [28], and NBFNet [39]. However, it stands out as the most space-efficient option when compared to the state-ofthe-art sparse KG completion models SparKGR [32], DT4KGR [33], and Hi-KnowE [35], which rely on external assistance. These models typically require embeddings for all entities and relations, parameters for neural networks, and storage space for rules. 

> 8Ref. [11] is not included in the comparison, as it introduces external knowledge graphs and an entity linking pipeline, rendering it significantly more complex. 

CIKM ’24, October 21–25, 2024, Boise, ID, USA 

Look Globally and Reason: Two-stage Path Reasoning over Sparse Knowledge Graphs 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/cikm_2024_Look_Globally_and_Reason_Two-stage_Path_Reasoning_over_Sparse_Knowledge_Graphs_images/cikm_2024_Look_Globally_and_Reason_Two-stage_Path_Reasoning_over_Sparse_Knowledge_Graphs.pdf-0009-02.png)


**----- Start of picture text -----**<br>
55.0<br>54.0 54.0 54.0 MRR<br>52.0 52.5 Hits@3<br>52.0 52.0 Hits@10<br>50.0 50.0<br>50.0 48.0 50.0 47.5<br>48.0 46.0 48.0 45.0<br>46.0 44.0<br>46.0 42.5<br>42.0<br>1000 5000 10000 15000 20000 3 4 5 6 100 500 1000 0.95 0.9 0.8 0.7 0.6 0.5 0.4 0.3 0.2 0.1<br>Max number of collected paths Max number of path hops Number of explored top paths Hop decay factor<br>Performance<br>**----- End of picture text -----**<br>


**Figure 3: Hyper-parameter analysis of LoGRe on WD-singer.** 

**Table 7: Complexity comparison with the best baselines. Here,** _𝛼_ **is the embedding dimension,** _𝛽_ **is the number of parameters in the neural networks,** _𝑁𝑒𝑝𝑜𝑐ℎ_ **is the number of training epochs, and** S = _𝑁𝑡𝑖𝑚𝑒 𝑁ℎ𝑜𝑝_ **and** T = _𝑁𝑡𝑖𝑚𝑒 𝑁ℎ𝑜𝑝_ + (| _𝑉𝑎𝑙𝑖𝑑_ | + | _𝑇𝑒𝑠𝑡_ |) _𝑁𝑟𝑢𝑙𝑒 𝑁ℎ𝑜𝑝_ **are the space and time complexities of the rule mining algorithm AnyBURL [16], respectively.** _𝑁𝑡𝑖𝑚𝑒_ **is the number of times for the path exploration and** _𝑁𝑟𝑢𝑙𝑒_ **is the number of matched rules for each query.** 

|Model|Space complexity|Time complexity|
|---|---|---|
|TuckER<br>ConvE<br>InteractE<br>NBFNet<br>SparKGR<br>DT4KGR<br>Hi-KnowE<br>LoGRe|_𝑂_(|_𝐸_|_𝛼_+ |_𝑅_|_𝛼_+_𝛼_3)<br>_𝑂_(|_𝐸_|_𝛼_+ |_𝑅_|_𝛼_+_𝛽_)<br>_𝑂_(|_𝐸_|_𝛼_+ |_𝑅_|_𝛼_+_𝛽_)<br>_𝑂_(|_𝑅_|_𝛼_2 +_𝛽_)<br>_𝑂_(S + |_𝐸_|_𝛼_+ |_𝑅_|_𝛼_+_𝛽_)<br>_𝑂_(S + |_𝐸_|_𝛼_+ |_𝑅_|_𝛼_+_𝛽_)<br>_𝑂_(S + |_𝐸_|_𝛼_+ |_𝑅_|_𝛼_+_𝛽_)<br>_𝑂_(|_𝐸_|_𝑁𝑝𝑎𝑡ℎ𝑁ℎ𝑜𝑝_)|_𝑂_((_𝑁𝑒𝑝𝑜𝑐ℎ_|_𝑇𝑟𝑎𝑖𝑛_| + |_𝑉𝑎𝑙𝑖𝑑_| + |_𝑇𝑒𝑠𝑡_|)_𝛼_3)<br>_𝑂_((_𝑁𝑒𝑝𝑜𝑐ℎ_|_𝑇𝑟𝑎𝑖𝑛_| + |_𝑉𝑎𝑙𝑖𝑑_| + |_𝑇𝑒𝑠𝑡_|)_𝛼_2)<br>_𝑂_((_𝑁𝑒𝑝𝑜𝑐ℎ_|_𝑇𝑟𝑎𝑖𝑛_| + |_𝑉𝑎𝑙𝑖𝑑_| + |_𝑇𝑒𝑠𝑡_|)_𝛼_2)<br>_𝑂_((_𝑁𝑒𝑝𝑜𝑐ℎ_|_𝑇𝑟𝑎𝑖𝑛_| + |_𝑉𝑎𝑙𝑖𝑑_| + |_𝑇𝑒𝑠𝑡_|)_𝛼_2)<br>_𝑂_(T + (_𝑁𝑒𝑝𝑜𝑐ℎ_|_𝑇𝑟𝑎𝑖𝑛_| + |_𝑉𝑎𝑙𝑖𝑑_| + |_𝑇𝑒𝑠𝑡_|)_𝛼_2)<br>_𝑂_(T + (_𝑁𝑒𝑝𝑜𝑐ℎ_|_𝑇𝑟𝑎𝑖𝑛_| + |_𝑉𝑎𝑙𝑖𝑑_| + |_𝑇𝑒𝑠𝑡_|)_𝛼_2)<br>_𝑂_(T + (_𝑁𝑒𝑝𝑜𝑐ℎ_|_𝑇𝑟𝑎𝑖𝑛_| + |_𝑉𝑎𝑙𝑖𝑑_| + |_𝑇𝑒𝑠𝑡_|)_𝛼_2)<br>_𝑂_(|_𝐸_|_𝑁𝑝𝑎𝑡ℎ𝑁ℎ𝑜𝑝_+ (|_𝑉𝑎𝑙𝑖𝑑_| + |_𝑇𝑒𝑠𝑡_|)_𝑁𝑡𝑜𝑝𝑁ℎ𝑜𝑝_)|



Regarding time complexity, LoGRe does not require training. It mainly involves constructing the relation-path reasoning schema, which exhibits a time complexity of _𝑂_ (| _𝐸_ | _𝑁𝑝𝑎𝑡ℎ𝑁ℎ𝑜𝑝_ ), and conducting path reasoning on the validation or test set of time complexity _𝑂_ (| _𝑉𝑎𝑙𝑖𝑑_ | _𝑁𝑡𝑜𝑝 𝑁ℎ𝑜𝑝_ ) or _𝑂_ (| _𝑇𝑒𝑠𝑡_ | _𝑁𝑡𝑜𝑝 𝑁ℎ𝑜𝑝_ ). The time complexity associated with the preference for short paths and answer similarity is considerably smaller and can be disregarded. Thus, the overall time complexity is _𝑂_ (| _𝐸_ | _𝑁𝑝𝑎𝑡ℎ𝑁ℎ𝑜𝑝_ + (| _𝑉𝑎𝑙𝑖𝑑_ | + | _𝑇𝑒𝑠𝑡_ |) _𝑁𝑡𝑜𝑝 𝑁ℎ𝑜𝑝_ ). In contrast, the best baselines necessitate training over numerous epochs. As illustrated in Table 7, their complexity can be summarized as at least _𝑂_ (( _𝑁𝑒𝑝𝑜𝑐ℎ_ | _𝑇𝑟𝑎𝑖𝑛_ | + (| _𝑉𝑎𝑙𝑖𝑑_ | + | _𝑇𝑒𝑠𝑡_ |) _𝑔_ ( _𝛼_ )) (some models need additional time for rule mining). Here, _𝑔_ ( _𝛼_ ) is the complexity of the embedding-based networks. Given that typically _𝑁𝑒𝑝𝑜𝑐ℎ_ | _𝑇𝑟𝑎𝑖𝑛_ | _> 𝑁𝑝𝑎𝑡ℎ𝑁ℎ𝑜𝑝_ , _𝑔_ ( _𝛼_ ) _>_ | _𝐸_ |, and _𝑔_ ( _𝛼_ ) _> 𝑁𝑡𝑜𝑝 𝑁ℎ𝑜𝑝_ , LoGRe is generally more time-efficient than existing KG completion models, especially in comparison to the three sparse KG completion models SparKGR [32], DT4KGR [33], and Hi-KnowE [35], which require additional time for rule mining. Notably, we do not compare the actual run times of these methods, as run times significantly influenced by the configuration of the realistic computing environment. Instead, we compare their time complexities, which are more objective and independent of the environment. 

## **6 Conclusion and Future Work** 

In this paper, we looked inward rather than seeking external help for the first time in the line of path-based sparse KG completion 

models. We proposed a two-stage model LoGRe, which constructs a relation-path reasoning schema by globally analyzing the training data to alleviate the sparseness problem and then aggregates paths to reason out answers. Experiments on five benchmark sparse KG datasets demonstrate that LoGRe outperforms rule-based and pathbased baselines. It is also comparable to the referred embeddingbased models and significantly superior to ChatGPT. Further comprehensive analyses substantiate the effectiveness and efficiency of LoGRe. 

This paper looks back to the path-based research line and proposes an explainable model, in contrast to the prevailing trend of embedding-based models and large language models. We hope to inspire research endeavors that prioritize explainability, lacking in the current research landscape. In the future, we intend to enhance LoGRe by introducing external models and knowledge, such as harnessing the natural language understanding capabilities of large language models as mentioned in Section 5.3, while striving to preserve a high degree of explainability. 

## **Acknowledgments** 

The work is supported by the SMP-IDATA Open Youth Fund, the Lenovo-CAS Joint Lab Youth Scientist Project, the project under Grant No. JCKY2022130C039, the Strategic Priority Research Program of the CAS under Grant No. XDB0680102, the National Natural Science Foundation of China under Grant No. 62002341, and the JCJQ Project of China. 

CIKM ’24, October 21–25, 2024, Boise, ID, USA 

Saiping Guan, Jiyao Wei, Xiaolong Jin, Jiafeng Guo, & Xueqi Cheng 

## **References** 

- [1] Yushi Bai, Xin Lv, Juanzi Li, Lei Hou, Yincen Qu, Zelin Dai, and Feiyu Xiong. 2022. SQUIRE: A Sequence-to-sequence Framework for Multi-hop Knowledge Graph Reasoning. In _Proceedings of the 2022 Conference on Empirical Methods in Natural Language Processing_ . 1649–1662. https://aclanthology.org/2022.emnlp-main.107 

- [2] Ivana Balažević, Carl Allen, and Timothy M Hospedales. 2019. TuckER: Tensor factorization for knowledge graph completion. In _Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing and the 9th International Joint Conference on Natural Language Processing_ . 5185–5194. https: //aclanthology.org/D19-1522 

- [3] Andrew Carlson, Justin Betteridge, Bryan Kisiel, Burr Settles, Estevam Hruschka, and Tom Mitchell. 2010. Toward an Architecture for Never-Ending Language Learning. In _Proceedings of the 24th AAAI Conference on Artificial Intelligence_ . 1306–1313. https://ojs.aaai.org/index.php/AAAI/article/view/7519 

- [4] Rajarshi Das, Ameya Godbole, Shehzaad Dhuliawala, Manzil Zaheer, and Andrew McCallum. 2020. A Simple Approach to Case-Based Reasoning in Knowledge Bases. In _Proceedings of the 2nd Conference on Automated Knowledge Base Construction_ . https://openreview.net/forum?id=AEY9tRqlU7 

- [5] Rajarshi Das, Ameya Godbole, Nicholas Monath, Manzil Zaheer, and Andrew McCallum. 2020. Probabilistic Case-based Reasoning for Open-World Knowledge Graph Completion. In _Findings of the Association for Computational Linguistics: EMNLP 2020_ . 4752–4765. https://aclanthology.org/2020.findings-emnlp.427 

- [6] Tim Dettmers, Pasquale Minervini, Pontus Stenetorp, and Sebastian Riedel. 2018. Convolutional 2D knowledge graph embeddings. In _Proceedings of the 32nd AAAI Conference on Artificial Intelligence_ . 1811–1818. https://ojs.aaai.org/index.php/ AAAI/article/view/11573 

- [7] Cong Fu, Tong Chen, Meng Qu, Woojeong Jin, and Xiang Ren. 2019. Collaborative Policy Learning for Open Knowledge Graph Reasoning. In _Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing and the 9th International Joint Conference on Natural Language Processing_ . 2672–2681. https://aclanthology.org/D19-1269 

- [8] Qing Guo, Zhu Sun, Jie Zhang, and Yin-Leng Theng. 2020. An attentional recurrent neural network for personalized next location recommendation. In _Proceedings of the 34th AAAI Conference on artificial intelligence_ . 83–90. https: //doi.org/10.1609/aaai.v34i01.5337 

- [9] Tao He, Ming Liu, Yixin Cao, Meng Qu, Zihao Zheng, and Bing Qin. 2024. VEM2L: An easy but effective framework for fusing text and structure knowledge on sparse knowledge graph completion. _Data Mining and Knowledge Discovery_ 38, 2 (2024), 343–371. https://doi.org/10.1007/s10618-023-01001-y 

- [10] Vinh Thinh Ho, Daria Stepanova, Mohamed H. Gad-Elrab, Evgeny Kharlamov, and Gerhard Weikum. 2018. Rule Learning from Knowledge Graphs Guided by Embedding Models. In _Proceedings of the 17th International Semantic Web Conference_ . 72–90. https://doi.org/10.1007/978-3-030-00671-6_5 

- [11] Hai-Tao Jia, Bo-Yang Zhang, Chao Huang, Wen-Han Li, Wen-Bo Xu, Yu-Feng Bi, and Li Ren. 2023. Application of graph neural network and feature information enhancement in relation inference of sparse knowledge graph. _Journal of Electronic Science and Technology_ 21, 2 (2023), 100194. https://www.sciencedirect. com/science/article/pii/S1674862X23000125 

- [12] Deren Lei, Gangrong Jiang, Xiaotao Gu, Kexuan Sun, Yuning Mao, and Xiang Ren. 2020. Learning Collaborative Agents with Rule Guidance for Knowledge Graph Reasoning. In _Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing_ . 8541–8547. https://aclanthology.org/2020.emnlp-main.688 

- [13] Xinglan Liu, Hussain Hussain, Houssam Razouk, and Roman Kern. 2022. Effective Use of BERT in Graph Embeddings for Sparse Knowledge Graph Completion. In _Proceedings of the 37th ACM/SIGAPP Symposium on Applied Computing_ . 799–802. https://doi.org/10.1145/3477314.3507031 

- [14] Xin Lv, Xu Han, Lei Hou, Juanzi Li, Zhiyuan Liu, Wei Zhang, Yichi Zhang, Hao Kong, and Suhui Wu. 2020. Dynamic Anticipation and Completion for Multi-Hop Reasoning over Sparse Knowledge Graph. In _Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing_ . 5694–5703. https://aclanthology.org/2020.emnlp-main.459 

- [15] Christopher D. Manning, Prabhakar Raghavan, and Hinrich Schütze. 2008. _Introduction to Information Retrieval_ . Cambridge University Press. 

- [16] Christian Meilicke, Melisachew Wudage Chekol, Manuel Fink, and Heiner Stuckenschmidt. 2020. Reinforced Anytime Bottom Up Rule Learning for Knowledge Graph Completion. arXiv:2004.04412 https://arxiv.org/abs/2004.04412 

- [17] Pouya Ghiasnezhad Omran, Kewen Wang, and Zhe Wang. 2018. Scalable Rule Learning via Learning Representation. In _Proceedings of the 27th International Joint Conference on Artificial Intelligence_ . 2149–2155. https://www.ijcai.org/ Proceedings/2018/0297.pdf 

- [18] Pouya Ghiasnezhad Omran, Zhe Wang, and Kewen Wang. 2019. Knowledge Graph Rule Mining via Transfer Learning. In _Proceedings of the 23rd PacificAsia Conference on Knowledge Discovery and Data Mining_ . 489–500. https: //doi.org/10.1007/978-3-030-16142-2_38 

- [19] Simon Ott, Patrick Betz, Daria Stepanova, Mohamed H. Gad-Elrab, Christian Meilicke, and Heiner Stuckenschmidt. 2023. Rule-based Knowledge Graph Completion with Canonical Models. In _Proceedings of the 32nd ACM International Conference on Information and Knowledge Management_ . 1971–1981. 

https://doi.org/10.1145/3583780.3615042 

- [20] Jay Pujara, Eriq Augustine, and Lise Getoor. 2017. Sparsity and Noise: Where Knowledge Graph Embeddings Fall Short. In _Proceedings of the 2017 Conference on Empirical Methods in Natural Language Processing_ . 1751–1756. https: //aclanthology.org/D17-1184 

- [21] Tieyun Qian, Bei Liu, Quoc Viet Hung Nguyen, and Hongzhi Yin. 2019. Spatiotemporal Representation Learning for Translation-Based POI Recommendation. _ACM Transactions on Information Systems_ 37, 2 (2019), 1–24. https: //doi.org/10.1145/3295499 

- [22] Tim Rocktäschel and Sebastian Riedel. 2017. End-to-end Differentiable Proving. In _Proceedings of the 31st International Conference on Neural Information Processing Systems_ , Vol. 30. https://proceedings.neurips.cc/paper_files/paper/2017/file/ b2ab001909a8a6f04b51920306046ce5-Paper.pdf 

- [23] Bin Shi, Hao Wang, Yueyan Li, and Sanhong Deng. 2023. RelaGraph: Improving Embedding on Small-Scale Sparse Knowledge Graphs by Neighborhood Relations. _Information Processing and Management_ 60, 5 (2023). https://doi.org/10.1016/j. ipm.2023.103447 

- [24] Yongjiao Sun, Jiancheng Guo, Boyang Li, and Nur Al Hasan Haldar. 2023. Effective Rule Mining of Sparse Data Based on Transfer Learning. _World Wide Web_ 26, 1 (2023), 461–480. https://doi.org/10.1007/s11280-022-01042-1 

- [25] Zhaoxuan Tan, Zilong Chen, Shangbin Feng, Qingyue Zhang, Qinghua Zheng, Jundong Li, and Minnan Luo. 2023. KRACL: Contrastive Learning with Graph Context Modeling for Sparse Knowledge Graph Completion. In _Proceedings of the ACM Web Conference 2023_ . 2548–2559. https://doi.org/10.1145/3543507.3583412 

- [26] Kristina Toutanova, Danqi Chen, Patrick Pantel, Hoifung Poon, Pallavi Choudhury, and Michael Gamon. 2015. Representing Text for Joint Embedding of Text and Knowledge Bases. In _Proceedings of the 2015 Conference on Empirical Methods in Natural Language Processing_ . 1499–1509. https://aclanthology.org/D15-1174 

- [27] Ke Tu, Wei Qu, Zhengwei Wu, Zhiqiang Zhang, Zhongyi Liu, Yiming Zhao, Le Wu, Jun Zhou, and Guannan Zhang. 2023. Disentangled Interest importance aware Knowledge Graph Neural Network for Fund Recommendation. In _Proceedings of the 32nd ACM International Conference on Information and Knowledge Management_ . 2482–2491. https://doi.org/10.1145/3583780.3614846 

- [28] Shikhar Vashishth, Soumya Sanyal, Vikram Nitin, Nilesh Agrawal, and Partha Talukdar. 2020. InteractE: Improving Convolution-Based Knowledge Graph Embeddings by Increasing Feature Interactions. In _Proceedings of the 34th AAAI Conference on Artificial Intelligence_ . 3009–3016. https://ojs.aaai.org/index.php/ AAAI/article/view/5694 

- [29] Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez, Łukasz Kaiser, and Illia Polosukhin. 2017. Attention is all you need. In _Proceedings of the 31st Conference on Neural Information Processing Systems_ . 1–11. https://proceedings.neurips.cc/paper_files/paper/2017/file/ 3f5ee243547dee91fbd053c1c4a845aa-Paper.pdf 

- [30] Denny Vrandečić and Markus Krötzsch. 2014. Wikidata: A free collaborative knowledge base. _Commun. ACM_ 57, 10 (2014), 78–85. https://dl.acm.org/doi/ fullHtml/10.1145/2629489 

- [31] Cheng Wang, Hangyu Zhu, Ruixin Hu, Rui Li, and Changjun Jiang. 2023. LongArms: Fraud Prediction in Online Lending Services Using Sparse Knowledge Graph. _IEEE Transactions on Big Data_ 9, 2 (2023), 758–772. https://doi.org/10. 1109/TBDATA.2022.3172060 

- [32] Yi Xia, Mingjing Lan, Junyong Luo, Xiaohui Chen, and Gang Zhou. 2022. Iterative rule-guided reasoning over sparse knowledge graphs with deep reinforcement learning. _Information Processing and Management_ 59, 5 (2022), 103040. https: //www.sciencedirect.com/science/article/pii/S0306457322001492 

- [33] Yi Xia, Junyong Luo, Gang Zhou, Mingjing Lan, Xiaohui Chen, and Jing Chen. 2024. DT4KGR: Decision transformer for fast and effective multi-hop reasoning over knowledge graphs. _Information Processing and Management_ 61, 3 (2024), 103648. https://doi.org/10.1016/j.ipm.2024.103648 

- [34] Ruobing Xie, Zhiyuan Liu, and Maosong Sun. 2016. Representation learning of knowledge graphs with hierarchical types. In _Proceedings of the 25th International Joint Conference on Artificial Intelligence_ . 2965–2971. https://www.ijcai.org/ Proceedings/16/Papers/421.pdf 

- [35] Shaorong Xie, Ruishen Liu, Xinzhi Wang, Xiangfeng Luo, Vijayan Sugumaran, and Hang Yu. 2024. Hierarchical Knowledge-Enhancement Framework for multihop knowledge graph reasoning. _Neurocomputing_ (2024), 127673. https://doi. org/10.1016/j.neucom.2024.127673 

- [36] Xin Xie, Zhoubo Li, Xiaohan Wang, Zekun Xi, and Ningyu Zhang. 2023. LambdaKG: A Library for Pre-trained Language Model-Based Knowledge Graph Embeddings. In _Proceedings of the 3rd Conference of the Asia-Pacific Chapter of the Association for Computational Linguistics: System Demonstrations_ . https://aclanthology.org/2023.ijcnlp-demo.4 

- [37] Mayi Xu, Ke Sun, Yongqi Li, and Tieyun Qian. 2023. Cold-Start Multi-hop Reasoning by Hierarchical Guidance and Self-verification. In _Proceedings of the 2023 Joint European Conference on Machine Learning and Knowledge Discovery in Databases_ . 577–592. https://doi.org/10.1007/978-3-031-43415-0_34 

- [38] Feng Zhao, Haoran Sun, Langjunqing Jin, and Hai Jin. 2020. Structure-augmented knowledge graph embedding for sparse data with rule learning. _Computer Communications_ 159 (2020), 271–278. https://www.sciencedirect.com/science/ 

Look Globally and Reason: Two-stage Path Reasoning over Sparse Knowledge Graphs 

article/pii/S0140366419319310 

- [39] Zhaocheng Zhu, Zuobai Zhang, Louis-Pascal Xhonneux, and Jian Tang. 2021. Neural Bellman-Ford Networks: A General Graph Neural Network Framework 

CIKM ’24, October 21–25, 2024, Boise, ID, USA 

for Link Prediction. In _Proceedings of the 35th Conference on Neural Information Processing Systems_ , Vol. 34. 29476–29490. https://proceedings.neurips.cc/paper_ files/paper/2021/file/f6a673f09493afcd8b129a0bcf1cd5bc-Paper.pdf 

