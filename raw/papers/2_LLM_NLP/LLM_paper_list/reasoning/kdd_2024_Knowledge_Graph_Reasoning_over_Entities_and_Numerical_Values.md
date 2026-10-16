## **Knowledge Graph Reasoning over Entities and Numerical Values** 

Chen Luo 

Zheng Li 

Jiaxin Bai[∗] Department of CSE, HKUST Hong Kong SAR, China jbai@connect.ust.hk 

Amazon.com Inc Palo Alto, USA cheluo@amazon.com 

Amazon.com Inc Palo Alto, USA amzzhe@amazon.com 

Yangqiu Song[†] Department of CSE, HKUST Hong Kong SAR, China yqsong@cse.ust.hk 

Qingyu Yin Amazon.com Inc Palo Alto, USA qingyy@amazon.com 

Bing Yin Amazon.com Inc Palo Alto, USA alexbyin@amazon.com 

## **CCS CONCEPTS** 

## **ABSTRACT** 

A complex logic query in a knowledge graph refers to a query expressed in logic form that conveys a complex meaning, such as _where did the Canadian Turing award winner graduate from?_ Knowledge graph reasoning-based applications, such as dialogue systems and interactive search engines, rely on the ability to answer complex logic queries as a fundamental task. In most knowledge graphs, edges are typically used to either describe the relationships between entities or their associated attribute values. An attribute value can be in categorical or numerical format, such as dates, years, sizes, etc. However, existing complex query answering (CQA) methods simply treat numerical values in the same way as they treat entities. This can lead to difficulties in answering certain queries, such as _which Australian Pulitzer award winner is born before 1927_ , and _which drug is a pain reliever and has fewer side effects than Paracetamol._ In this work, inspired by the recent advances in numerical encoding and knowledge graph reasoning, we propose numerical complex query answering. In this task, we introduce new numerical variables and operations to describe queries involving numerical attribute values. To address the difference between entities and numerical values, we also propose the framework of Number Reasoning Network (NRN) for alternatively encoding entities and numerical values into separate encoding structures. During the numerical encoding process, NRN employs a parameterized density function to encode the distribution of numerical values. During the entity encoding process, NRN uses established query encoding methods for the original CQA problem. Experimental results show that NRN consistently improves various query encoding methods on three different knowledge graphs and achieves state-of-the-art results. 

• **Information systems** → **Data mining** ; • **Computing methodologies** → **Semantic networks** ; **Logic programming and answer set programming** . 

## **KEYWORDS** 

knowledge graph, complex query answering, numerical attribute 

## **ACM Reference Format:** 

Jiaxin Bai, Chen Luo, Zheng Li, Qingyu Yin, Bing Yin, and Yangqiu Song. 2023. Knowledge Graph Reasoning over Entities and Numerical Values. In _Proceedings of the 29th ACM SIGKDD Conference on Knowledge Discovery and Data Mining (KDD ’23), August 6–10, 2023, Long Beach, CA, USA._ ACM, New York, NY, USA, 12 pages. https://doi.org/10.1145/3580305.3599399 

## **1 INTRODUCTION** 

Reasoning over knowledge graphs (KG) is the process of deriving new knowledge or drawing new conclusions from the existing ones in the KG [9]. Complex query answering (CQA) is a recently developed knowledge graph reasoning task, which aims to answer complex knowledge graph queries [15, 24, 25]. As shown in Figure 1, a complex knowledge graph query, or in short, complex query, targets finding the entities from the KG such that the logic expression can be satisfied [15, 24]. The logic expression of a complex query usually contains multiple terms connected by logic connectives, thus it is able to carry a complicated meaning. For example, in Figure 1, the logic form of _𝑞_ 1 includes terms like _Win(TuringAward,V)_ and operations like conjunctions ∧ and disjunctions ∨, and it carries the meaning of _Find where the Canadian Turing award laureates graduated from_ . The complex query answering (CQA) task is considered a KG reasoning task because it needs to deal with the incompleteness problem of the KGs. The KGs, like Freebase [8] and YAGO [26], are typically sparse and incomplete. They often include missing relations and attribute values, although these missing edges can be inferred from other edges in the KG. Consequently, subgraph matching algorithms cannot directly be used to find the answers that need to be inferred from missing edges and attributes. 

> ∗ Work done during an internship at Amazon. 

> † Visiting academic scholar at Amazon. 

Permission to make digital or hard copies of all or part of this work for personal or classroom use is granted without fee provided that copies are not made or distributed for profit or commercial advantage and that copies bear this notice and the full citation on the first page. Copyrights for components of this work owned by others than the author(s) must be honored. Abstracting with credit is permitted. To copy otherwise, or republish, to post on servers or to redistribute to lists, requires prior specific permission and/or a fee. Request permissions from permissions@acm.org. _KDD ’23, August 6–10, 2023, Long Beach, CA, USA_ 

Query encoding methods [4, 14, 15, 20, 24, 25, 27, 39] are proposed to address this incompleteness challenge in CQA. In query encoding methods, queries and entities are simultaneously encoded into the same embedding space. Then the answers are retrieved according to the similarity between the query embedding and entity embedding. In detail, there are two steps in encoding a complex query. First, the query is parsed into a computational graph as a 

© 2023 Copyright held by the owner/author(s). Publication rights licensed to ACM. ACM ISBN 979-8-4007-0103-0/23/08...$15.00 https://doi.org/10.1145/3580305.3599399 

KDD ’23, August 6–10, 2023, Long Beach, CA, USA 

Jiaxin Bai, Chen Luo, Zheng Li, Qingyu Yin, Bing Yin, and Yangqiu Song 

directed acyclic graph (DAG). Then, the query encoding is iteratively computed along the computational graph, by using neural parameterized logic operations and relational projections. Different query encoding methods encode their queries into different structures, and they have different parameterizations for their logic and projection operations as well. For example, GQE [15] uses vectors, Query2Box [24] uses hyper-rectangles, and Query2Particles [4] uses multiple vectors in the embedding space. 

On the other hand, numerical values are also important components of various knowledge graphs like Freebase [8], DBpedia [7], and YAGO [26]. Though existing query encoding methods achieve good performance on complex query answering, they typically focus only on the relations between entities while being unable to reasonably deal with attribute values. There are in general two reasons. First, the query encoding methods are unable to encode the semantic information of numerical values. For _𝑞_ 2 in Figure 1, to find its corresponding answers, we need to first encode the semantics of _the year of 1927_ such that we can compute the distributions of _the years before 1927_ . However, existing query encoding methods can only learn the graph features of _1927_ , and ignore its numerical meaning. Second, the existing query encoding method is unable to describe the distributional changes of values when there are computations and comparisons between numerical values. In Figure 1, the complex queries _𝑞_ 2, _𝑞_ 3, and _𝑞_ 4 involve numerical comparisons _𝐺𝑟𝑒𝑎𝑡𝑒𝑟𝑇ℎ𝑎𝑛_ or _𝑆𝑚𝑎𝑙𝑙𝑒𝑟𝑇ℎ𝑎𝑛_ . Meanwhile, _𝑞_ 4 includes the numerical computation of _𝑇𝑖𝑚𝑒𝑠𝐵𝑦𝑇𝑤𝑜_ . As the existing methods only have relational projections between entities, they are insufficient to deal with the relations involving numerical values. 

To formally define and evaluate the problem of complex query answering with numerical values, we propose the task of numerical complex query answering (Numerical CQA). In Numerical CQA, we extend the existing problem scope to further include the numerical variables. We also introduce new logic terms expressing numerical relationships. In Figure 1, the logic terms expressing numerical attributes are underlined. Three example numerical queries are given in Figure 1, and their definitions will be given in §2. Meanwhile, we also create the corresponding benchmark dataset to evaluate the performance of Numerical CQA. To do this, we use three public knowledge graphs: Freebase, YAGO, and DBpedia. 

To address the challenges of reasonably encoding both entities and numerical values in a query, we propose the Number Reasoning Network (NRN) as a solution. In NRN, there are two encoding phases, which are respectively used for encoding entities and numerical values, following the computation graph. The descriptions of NRN will be given in §3. Experiment results show that the NRN can constantly improve the performance on Numerical CQA when combined with different types of query encoding methods.[1] The major contributions of this paper are three-fold: 

- We propose a new problem of numerical complex query answering (Numerical CQA) aiming at answering the complex queries that need reasoning over numerical values. 

- We create a new benchmark dataset based on three widely used knowledge graphs: Freebase, Dbpedia, and YAGO, for evaluating the performance of Numerical CQA. 

**Table 1: The table for frequent notations with definitions.** 

|Notations|Defnitions|
|---|---|
|G<br>V<br>R<br>A<br>N<br>_𝑉𝑎_,_𝑋𝑎_<br>_𝑉𝑖_,_𝑋𝑖_<br>_𝑟_(_𝑉,𝑉_′)<br>_𝑎_(_𝑉,𝑋_)<br>_𝑓_(_𝑋,𝑋_′)|The knowledge graph<br>The set of knowledge graph vertices in KG<br>The set of relations between entities in KG<br>The set of attribute types in KG<br>The set of numerical values in KG<br>The anchor variables that are known in a KG query<br>The existentially quantifed variables in a KG query<br>Whether the relation_𝑟_holds between_𝑉_and_𝑉_′<br>Whether_𝑉_has an attribute_𝑎_with value_𝑋_<br>Whether values_𝑋_and_𝑋_′ satisfy numerical relation_𝑓_|



- We propose Number Reasoning Network (NRN) for CQA. It iteratively encodes a numerical complex query into two different structures for entities and values respectively and achieves state-of-the-art Numerical CQA. 

## **2 DEFINITION OF NUMERICAL CQA** 

To precisely describe the problem, we mathematically define numerical complex query answering in this section. A multi-relational knowledge graph with numerical attributes is defined as G = (V _,_ R _,_ A _,_ N). V is the set of the nodes or vertices, R is the set of relation types, A is the set of numerical attribute types, and N is the set of numerical attribute values. Each vertex _𝑣_ ∈V represents an entity. Each of the relations _𝑟_ ∈R is a binary function defined as _𝑟_ : V×V →{0 _,_ 1}. For any _𝑟_ ∈R, and _𝑢, 𝑣_ ∈V, there is a relation _𝑟_ between _𝑢_ and _𝑣_ if and only if _𝑟_ ( _𝑢, 𝑣_ ) = 1. Meanwhile, each attribute _𝑎_ ∈A is also a binary function describing numerical attribute values, and it is defined on V × N →{0 _,_ 1}. The entity _𝑣_ ∈V has an attribute _𝑎_ ∈A of value _𝑥_ ∈N , if and only if _𝑎_ ( _𝑣,𝑥_ ) = 1. In some queries, we also need to express the numerical relations, like comparison and computation between two numerical values, so we use another set of binary functions _𝑓_ ( _𝑥_ 1 _,𝑥_ 2) : N × N →{0 _,_ 1} to describe whether two numerical values satisfy certain constraints, such as _𝑥_ 1 _< 𝑥_ 2, _𝑥_ 1 _> 𝑥_ 2 + 20, or _𝑥_ 1 = 2 _𝑥_ 2 + 3. The relations are defined as binary functions so that they can be directly used to express boolean values in complex queries, and we will introduce the definition of the complex queries in the following paragraph. 

A complex numerical query is defined in existential positive first-order logic form, which is a type of logic expression including existential quantifiers ∃, logic conjunctions ∧, and disjunctions ∨. In the logic query, there is a set of anchor entities _𝑉𝑎_ and values _𝑋𝑎_ . These anchor entities and values are the given entities and values in a query, like _𝐵𝑒𝑖𝑗𝑖𝑛𝑔_ in _𝑞_ 3 and _𝐶𝑎𝑙𝑖𝑓𝑜𝑟𝑛𝑖𝑎_ in _𝑞_ 4 in Figure 1. Meanwhile, there are existential quantified entity variables _𝑉_ 1, ..., _𝑉𝑘_ , and numerical variables _𝑋_ 1, ..., _𝑋𝑙_ in a complex query. There is a unique target variable, _𝑉_ ? in each logic query. The query intends to find the values of _𝑉_ ?, such that there simultaneously exist _𝑉_ 1 _, ...,𝑉𝑘_ ∈ V and _𝑋_ 1 _, ...,𝑋𝑙_ ∈N satisfying the logic expression. As every logic query can be converted into a disjunctive normal form, the complex numerical queries can be expressed in the following formula: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/kdd_2024_Knowledge_Graph_Reasoning_over_Entities_and_Numerical_Values_images/kdd_2024_Knowledge_Graph_Reasoning_over_Entities_and_Numerical_Values.pdf-0002-14.png)


Here, _𝑐𝑖_ are conjunctive expressions of several atomic logic expressions _𝑒𝑖,𝑗_ , where each _𝑒𝑖,𝑗_ is any of the following expressions: 

1Experiment code available: https://github.com/HKUST-KnowComp/NRN 

~~KDD ’23, August 6–10, 2023, Long Beach, CA, USA~~ 

~~Knowledge Graph Reasoning over Entities and Numerical Values~~ 

|**Query Category**<br>**Complex Queries**<br>**Interpretations**<br>CQA<br>𝑞�= 𝑉?. ∃𝑉: 𝑊𝑖𝑛<br>𝑉, 𝑇𝑢𝑟𝑖𝑛𝑔𝐴𝑤𝑎𝑟𝑑<br>∧𝐼𝑠𝐶𝑖𝑡𝑖𝑧𝑒𝑛𝑂𝑓<br>𝑉, 𝐶𝑎𝑛𝑎𝑑𝑎∧𝐺𝑟𝑎𝑑𝑢𝑎𝑡𝑒𝑑𝐹𝑟𝑜𝑚(𝑉, 𝑉?)<br>Find where the Canadian Turing award<br>laureates graduated from.<br>Numerical CQA<br>𝑞�= 𝑉?. ∃𝑋�, 𝑋�:  𝑊𝑖𝑛<br>𝑉?, 𝑃𝑢𝑙𝑖𝑡𝑧𝑒𝑟𝐴𝑤𝑎𝑟𝑑<br>∧𝐺𝑟𝑒𝑎𝑡𝑒𝑟𝑇ℎ𝑎𝑛<br>1927, 𝑋�∧𝐵𝑜𝑟𝑛𝐼𝑛<br> (𝑉?, 𝑋�)<br>Find the Pulitzer award winners thatis<br>born before the year of 1927.<br>Numerical CQA<br>𝑞�= 𝑉?. ∃𝑋�, 𝑋�: 𝑇𝑟𝑒𝑎𝑡<br>𝑉?, 𝑃𝑎𝑖𝑛∧𝑁𝑢𝑚𝑆𝑖𝑑𝑒𝐸𝑓𝑓𝑒𝑐𝑡<br>𝑉?, 𝑋�<br>∧𝑆𝑚𝑎𝑙𝑙𝑒𝑟𝑇ℎ𝑎𝑛<br>𝑋�, 𝑋�∧𝑁𝑢𝑚𝑆𝑖𝑑𝑒𝐸𝑓𝑓𝑒𝑐𝑡<br>Paracetamol, 𝑋�<br>Find the drug that can relieve pain and<br>have fewer side effects than Paracetamol.<br>Numerical CQA<br>𝑞�= 𝑉?. ∃𝑋�, 𝑋�, 𝑋�:  𝐿𝑜𝑐𝑎𝑡𝑒𝑑𝐼𝑛<br>𝑉?, 𝑈𝑛𝑖𝑡𝑒𝑑𝑆𝑡𝑎𝑡𝑒𝑠∧𝐻𝑎𝑠𝑃𝑜𝑝𝑢𝑙𝑎𝑡𝑖𝑜𝑛<br>𝑉?, 𝑋�<br>∧𝑆𝑚𝑎𝑙𝑙𝑒𝑟𝑇ℎ𝑎𝑛<br>𝑋�, 𝑋�∧𝑇𝑖𝑚𝑒𝑠𝐵𝑦𝑇𝑤𝑜<br>𝑋�, 𝑋�∧𝐻𝑎𝑠𝑃𝑜𝑝𝑢𝑙𝑎𝑡𝑖𝑜𝑛<br>𝐶𝑎𝑙𝑖𝑓𝑜𝑟𝑛𝑖𝑎, 𝑋�<br>Find the states in US that have atwice<br>smaller population than California?|**Query Category**<br>**Complex Queries**<br>**Interpretations**<br>CQA<br>𝑞�= 𝑉?. ∃𝑉: 𝑊𝑖𝑛<br>𝑉, 𝑇𝑢𝑟𝑖𝑛𝑔𝐴𝑤𝑎𝑟𝑑<br>∧𝐼𝑠𝐶𝑖𝑡𝑖𝑧𝑒𝑛𝑂𝑓<br>𝑉, 𝐶𝑎𝑛𝑎𝑑𝑎∧𝐺𝑟𝑎𝑑𝑢𝑎𝑡𝑒𝑑𝐹𝑟𝑜𝑚(𝑉, 𝑉?)<br>Find where the Canadian Turing award<br>laureates graduated from.<br>Numerical CQA<br>𝑞�= 𝑉?. ∃𝑋�, 𝑋�:  𝑊𝑖𝑛<br>𝑉?, 𝑃𝑢𝑙𝑖𝑡𝑧𝑒𝑟𝐴𝑤𝑎𝑟𝑑<br>∧𝐺𝑟𝑒𝑎𝑡𝑒𝑟𝑇ℎ𝑎𝑛<br>1927, 𝑋�∧𝐵𝑜𝑟𝑛𝐼𝑛<br> (𝑉?, 𝑋�)<br>Find the Pulitzer award winners thatis<br>born before the year of 1927.<br>Numerical CQA<br>𝑞�= 𝑉?. ∃𝑋�, 𝑋�: 𝑇𝑟𝑒𝑎𝑡<br>𝑉?, 𝑃𝑎𝑖𝑛∧𝑁𝑢𝑚𝑆𝑖𝑑𝑒𝐸𝑓𝑓𝑒𝑐𝑡<br>𝑉?, 𝑋�<br>∧𝑆𝑚𝑎𝑙𝑙𝑒𝑟𝑇ℎ𝑎𝑛<br>𝑋�, 𝑋�∧𝑁𝑢𝑚𝑆𝑖𝑑𝑒𝐸𝑓𝑓𝑒𝑐𝑡<br>Paracetamol, 𝑋�<br>Find the drug that can relieve pain and<br>have fewer side effects than Paracetamol.<br>Numerical CQA<br>𝑞�= 𝑉?. ∃𝑋�, 𝑋�, 𝑋�:  𝐿𝑜𝑐𝑎𝑡𝑒𝑑𝐼𝑛<br>𝑉?, 𝑈𝑛𝑖𝑡𝑒𝑑𝑆𝑡𝑎𝑡𝑒𝑠∧𝐻𝑎𝑠𝑃𝑜𝑝𝑢𝑙𝑎𝑡𝑖𝑜𝑛<br>𝑉?, 𝑋�<br>∧𝑆𝑚𝑎𝑙𝑙𝑒𝑟𝑇ℎ𝑎𝑛<br>𝑋�, 𝑋�∧𝑇𝑖𝑚𝑒𝑠𝐵𝑦𝑇𝑤𝑜<br>𝑋�, 𝑋�∧𝐻𝑎𝑠𝑃𝑜𝑝𝑢𝑙𝑎𝑡𝑖𝑜𝑛<br>𝐶𝑎𝑙𝑖𝑓𝑜𝑟𝑛𝑖𝑎, 𝑋�<br>Find the states in US that have atwice<br>smaller population than California?|**Query Category**<br>**Complex Queries**<br>**Interpretations**<br>CQA<br>𝑞�= 𝑉?. ∃𝑉: 𝑊𝑖𝑛<br>𝑉, 𝑇𝑢𝑟𝑖𝑛𝑔𝐴𝑤𝑎𝑟𝑑<br>∧𝐼𝑠𝐶𝑖𝑡𝑖𝑧𝑒𝑛𝑂𝑓<br>𝑉, 𝐶𝑎𝑛𝑎𝑑𝑎∧𝐺𝑟𝑎𝑑𝑢𝑎𝑡𝑒𝑑𝐹𝑟𝑜𝑚(𝑉, 𝑉?)<br>Find where the Canadian Turing award<br>laureates graduated from.<br>Numerical CQA<br>𝑞�= 𝑉?. ∃𝑋�, 𝑋�:  𝑊𝑖𝑛<br>𝑉?, 𝑃𝑢𝑙𝑖𝑡𝑧𝑒𝑟𝐴𝑤𝑎𝑟𝑑<br>∧𝐺𝑟𝑒𝑎𝑡𝑒𝑟𝑇ℎ𝑎𝑛<br>1927, 𝑋�∧𝐵𝑜𝑟𝑛𝐼𝑛<br> (𝑉?, 𝑋�)<br>Find the Pulitzer award winners thatis<br>born before the year of 1927.<br>Numerical CQA<br>𝑞�= 𝑉?. ∃𝑋�, 𝑋�: 𝑇𝑟𝑒𝑎𝑡<br>𝑉?, 𝑃𝑎𝑖𝑛∧𝑁𝑢𝑚𝑆𝑖𝑑𝑒𝐸𝑓𝑓𝑒𝑐𝑡<br>𝑉?, 𝑋�<br>∧𝑆𝑚𝑎𝑙𝑙𝑒𝑟𝑇ℎ𝑎𝑛<br>𝑋�, 𝑋�∧𝑁𝑢𝑚𝑆𝑖𝑑𝑒𝐸𝑓𝑓𝑒𝑐𝑡<br>Paracetamol, 𝑋�<br>Find the drug that can relieve pain and<br>have fewer side effects than Paracetamol.<br>Numerical CQA<br>𝑞�= 𝑉?. ∃𝑋�, 𝑋�, 𝑋�:  𝐿𝑜𝑐𝑎𝑡𝑒𝑑𝐼𝑛<br>𝑉?, 𝑈𝑛𝑖𝑡𝑒𝑑𝑆𝑡𝑎𝑡𝑒𝑠∧𝐻𝑎𝑠𝑃𝑜𝑝𝑢𝑙𝑎𝑡𝑖𝑜𝑛<br>𝑉?, 𝑋�<br>∧𝑆𝑚𝑎𝑙𝑙𝑒𝑟𝑇ℎ𝑎𝑛<br>𝑋�, 𝑋�∧𝑇𝑖𝑚𝑒𝑠𝐵𝑦𝑇𝑤𝑜<br>𝑋�, 𝑋�∧𝐻𝑎𝑠𝑃𝑜𝑝𝑢𝑙𝑎𝑡𝑖𝑜𝑛<br>𝐶𝑎𝑙𝑖𝑓𝑜𝑟𝑛𝑖𝑎, 𝑋�<br>Find the states in US that have atwice<br>smaller population than California?|
|---|---|---|
|**Query Category**<br>**Complex Queries**<br>**Interpretations**|||
|CQA<br>𝑞�= 𝑉?. ∃𝑉: 𝑊𝑖𝑛<br>𝑉, 𝑇𝑢𝑟𝑖𝑛𝑔𝐴𝑤𝑎𝑟𝑑<br>∧𝐼𝑠𝐶𝑖𝑡𝑖𝑧𝑒𝑛𝑂𝑓<br>𝑉, 𝐶𝑎𝑛𝑎𝑑𝑎∧𝐺𝑟𝑎𝑑𝑢𝑎𝑡𝑒𝑑𝐹𝑟𝑜𝑚(𝑉, 𝑉?)<br>Find where the Canadian Turing award<br>laureates graduated from.|||
|Numerical CQA<br>𝑞�= 𝑉?. ∃𝑋�, 𝑋�:  𝑊𝑖𝑛<br>𝑉?, 𝑃𝑢𝑙𝑖𝑡𝑧𝑒𝑟𝐴𝑤𝑎𝑟𝑑<br>∧𝐺𝑟𝑒𝑎𝑡𝑒𝑟𝑇ℎ𝑎𝑛<br>1927, 𝑋�∧𝐵𝑜𝑟𝑛𝐼𝑛<br> (𝑉?, 𝑋�)<br>Find the Pulitzer award winners thatis<br>born before the year of 1927.|||
|Numerical CQA<br>𝑞�= 𝑉?. ∃𝑋�, 𝑋�: 𝑇𝑟𝑒𝑎𝑡<br>𝑉?, 𝑃𝑎𝑖𝑛∧𝑁𝑢𝑚𝑆𝑖𝑑𝑒𝐸𝑓𝑓𝑒𝑐𝑡<br>𝑉?, 𝑋�<br>∧𝑆𝑚𝑎𝑙𝑙𝑒𝑟𝑇ℎ𝑎𝑛<br>𝑋�, 𝑋�∧𝑁𝑢𝑚𝑆𝑖𝑑𝑒𝐸𝑓𝑓𝑒𝑐𝑡<br>Paracetamol, 𝑋�<br>Find the drug that can relieve pain and<br>have fewer side effects than Paracetamol.|||
|Numerical CQA<br>𝑞�= 𝑉?. ∃𝑋�, 𝑋�, 𝑋�:  𝐿𝑜𝑐𝑎𝑡𝑒𝑑𝐼𝑛<br>𝑉?, 𝑈𝑛𝑖𝑡𝑒𝑑𝑆𝑡𝑎𝑡𝑒𝑠∧𝐻𝑎𝑠𝑃𝑜𝑝𝑢𝑙𝑎𝑡𝑖𝑜𝑛<br>𝑉?, 𝑋�<br>∧𝑆𝑚𝑎𝑙𝑙𝑒𝑟𝑇ℎ𝑎𝑛<br>𝑋�, 𝑋�∧𝑇𝑖𝑚𝑒𝑠𝐵𝑦𝑇𝑤𝑜<br>𝑋�, 𝑋�∧𝐻𝑎𝑠𝑃𝑜𝑝𝑢𝑙𝑎𝑡𝑖𝑜𝑛<br>𝐶𝑎𝑙𝑖𝑓𝑜𝑟𝑛𝑖𝑎, 𝑋�<br>Find the states in US that have atwice<br>smaller population than California?|||
||||



**Figure 1: Examples of entity relations and attribute complex queries. Existing complex query answering research only focuses on entities and their relations. The queries involving numerical values are not defined and studied in the previous literature. Here we underline the logic terms that involve at least one numerical variable in this figure.** 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/kdd_2024_Knowledge_Graph_Reasoning_over_Entities_and_Numerical_Values_images/kdd_2024_Knowledge_Graph_Reasoning_over_Entities_and_Numerical_Values.pdf-0003-04.png)


**----- Start of picture text -----**<br>
(A) Find the cities that  (B) q = 𝑉? . ∃𝑉�, 𝑋�, 𝑋�: HasLatitude 𝑉?, 𝑋�<br>have a higher  ∧𝐺𝑟𝑒𝑎𝑡𝑒𝑟𝑇ℎ𝑎𝑛 𝑋�, 𝑋�<br>latitudes than ∧𝐻𝑎𝑠𝐿𝑎𝑡𝑖𝑡𝑢𝑑𝑒 𝑉�, 𝑋�<br>Japanese cities. ∧ 𝐿𝑜𝑐𝑎𝑡𝑒𝑑𝐼𝑛 𝑉�, 𝐽𝑎𝑝𝑎𝑛<br>(C) 𝐻𝑎𝑠𝐶𝑖𝑡𝑦 𝐻𝑎𝑠𝐿𝑎𝑡𝑖𝑡𝑢𝑑𝑒 𝐺𝑟𝑒𝑎𝑡𝑒𝑟𝑇ℎ𝑎𝑛 𝐻𝑎𝑠𝐿𝑎𝑡𝑖𝑡𝑖𝑑𝑒 [-1]<br>𝐽𝑎𝑝𝑎𝑛 𝑇𝑎𝑟𝑔𝑒𝑡<br>(D)<br>𝑇𝑜𝑘𝑦𝑜<br>𝐾𝑦𝑜𝑡𝑜<br>𝐽𝑎𝑝𝑎𝑛<br>𝑂𝑠𝑎𝑘𝑎<br>Relational  Attribute  Numerical  Reverse<br>Projection Projection Projection Attribute Projection<br>**----- End of picture text -----**<br>


**Figure 2: An example query demonstrating the Number Reasoning Network (NRN) for complex query with numerical values. (A) The query interpretation; (B) The logic knowledge graph query involving attribute value reasoning; (C) The computational graph parsed from the logic query; (D) The illustrations of four types of projections. The attribute projection, the numerical projection, and the reverse attribute projections are newly introduced for reasoning over values.** 

encoding phase, in which we use parameterized distribution to encode the intermediate numerical representation. Here is a sample encoding process of NRN. For example, we want to answer the logic query in Figure 2 (B), expressing the meaning of _Find cities that have a higher latitude than Japanese cities._ The query will first be parsed into the computational graph as shown in (C). To start with, we are given the entity embedding of the anchor entity _Japan_ , and then a _relational projection_ is used to compute the representation of the _cities in Japan_ . In these steps, the encoding process is in the entity phase. Then, we will compute the numerical values indicating the _latitude values of cities in Japan_ by using _attribute projection_ . Then we use another encoding structure to encode the set of numerical values in the numerical encoding phase. Specifically, we use parameterized probabilistic distributions, to estimate the intermediate numerical values. When we obtain the intermediate representation of _latitudes of cities in Japan_ , we will estimate the distribution of the latitude values that are larger than previous results by using a _numerical projection_ of _GreaterThan_ . Finally, a _reverse attribute projection_ is computed to find the cities that have such latitude values from the previous step, and the encoding process switches back to the entity phase. In the following part of this section, we are going to first give the full definition of the NRN computational graph, and then introduce the query encoding structure for both entity and numerical variables. Finally, we discuss how to parameterize the operations in the computational graph of NRN. 

_𝑒𝑖,𝑗_ = _𝑟_ ( _𝑉,𝑉_[′] ), _𝑒𝑖,𝑗_ = _𝑎_ ( _𝑉,𝑋_ ), or _𝑒𝑖,𝑗_ = _𝑓_ ( _𝑋,𝑋_[′] ), where the _𝑟_ , _𝑎_ , and _𝑓_ are the binary expressing entity relation, attribute, and numerical relation respectively. They are defined in the previous subsection. _𝑉,𝑉_[′] _,𝑋,𝑋_[′] are either anchor entities or values, or existentially quantified variables in { _𝑉_ 1 _,𝑉_ 2 _, ...,𝑉𝑘_ } or { _𝑋_ 1 _,𝑋_ 2 _, ...,𝑋𝑙_ }. 

## **3 NUMBER REASONING NETWORK FOR CQA** 

To deal with the complex queries that involve both entities and numerical values, in this section, we propose the number reasoning network (NRN) method. Different from the original query encoding methods, in NRN, there are two phases in the encoding process to encode the entities and numerical values separately. One of the phases is the entity encoding phase, where we use the existing query encoding structure, like Box Embedding [24], to encode intermediate entity set representation. The other phase is the numerical 

## **3.1 Computational Graph for NRN** 

In the query encoding process for NRN, each intermediate state in the encoding process corresponds to either a set of entities or a set of attribute values. For the computational graph of entity-only complex queries, only relational projections and logic operations appear. Their definitions are as follows: 

- _Relational Projection_ : Given a set of entities _𝑆_ and a relation between entities _𝑟_ ∈R, the relational projection will produce all the entities that hold a relation r with any of the entities in _𝑆_ . Thus the relational projection can be described as _𝑃𝑟_ ( _𝑆_ ) = { _𝑣_ ∈V| _𝑣_[′] ∈ _𝑆,𝑟_ ( _𝑣_[′] _, 𝑣_ ) = 1}; 

- _Intersection_ : Given a set of entities or values _𝑆_ 1 _,𝑆_ 2 _, ...,𝑆𝑛_ , this logic operation of intersection computes the intersection of those sets as ∩ _[𝑛] 𝑖_ =1 _[𝑆][𝑖]_[;] 

KDD ’23, August 6–10, 2023, Long Beach, CA, USA 

Jiaxin Bai, Chen Luo, Zheng Li, Qingyu Yin, Bing Yin, and Yangqiu Song 

- _Union_ : Given a set of entities or values _𝑆_ 1 _,𝑆_ 2 _, ...,𝑆𝑛_ , the operation computes the union of ∪ _[𝑛] 𝑖_ =1 _[𝑆][𝑖]_[.] 

As shown in Figure 2 (D), to compute the computational graph involving numerical attributes, we need to define three types of new projections. An _attribute projection_ is to compute a set of attribute values of a given set of entities. A _reverse attribute projection_ is to find a set of entities that have the attribute values in a given set of values. A _numerical projection_ is used to find a set of values that holds certain constraints with another set of values, for example, _GreatThan_ , _EqualTo_ , etc. Their formal definitions are as follows: 

- _Attribute Projection_ : Given a set of entities S and a numerical attribute type _𝑎_ , the attribute projection gives all the values that can be achieved under the attribute type a by any of the entities in _𝑆_ . Then the attribute projection can be described by _𝑃𝑎_ ( _𝑆_ ) = { _𝑥_ ∈N| _𝑣_ ∈ _𝑆,𝑎_ ( _𝑣,𝑥_ ) = 1}; 

- _Reverse Attribute Projection_ : Given a set of values _𝑆_ ⊂N and a numerical attribute type _𝑎_ , the reverse attribute projection gives all the entities whose attribute a has a value in S. As a result, the reverse attribute projection can be described by the following formula _𝑃𝑎_[−][1][(] _[𝑆]_[)][=][{] _[𝑣]_[∈V|] _[𝑥]_[∈] _[𝑆,𝑎]_[(] _[𝑣,𝑥]_[)][=][ 1][}][;] 

- _Numerical Projection_ : A numerical projection is used to compute a set of values from a set of existing values. This computation depends on the numerical relation function _𝑓_ , and it can be described by _𝑃𝑓_ ( _𝑆_ ) = { _𝑥_ ∈N| _𝑥_[′] ∈ _𝑆, 𝑓_ ( _𝑥_[′] _,𝑥_ ) = 1}; 

## **3.2 Representations and Operations in NRN** 

When a complex query is iteratively encoded through the computation graph, as shown in Figure 2 (C), it is either in the entity encoding phase or in the numerical encoding phase. Correspondingly, the intermediate state either represents a set of entities in the knowledge graph or a set of attribute values. 

In previous work on query encoding, various methods were proposed to encode entity variables, but they cannot reasonably encode numerical attributes. In this paper, we will focus on how to compute and represent numerical values and how to use them to answer complex queries. For simplicity, suppose in the step _𝑖_ and the encoding process is entity phase, then the query embedding is denoted as: _𝑞𝑖_ ∈ _𝑅[𝑑]_ . Otherwise, if the encoding process is in the numerical phase, queries are represented in a more sophisticated way. Previous research shows that directly using the numerical value itself as input to a reasoning model is not effective. As a result, we employ static numerical encoding methods to project the numerical values from the real space _𝑅_ to the _𝑑_ -dimensional real space _𝑅[𝑑]_ , and this numerical encoding is denoted as _𝜓_ (·). Meanwhile, we further assume the intermediate numerical answers are drawn from a multi-variant probabilistic distribution in the _𝑑_ -dimensional space. 

If the query encoding process is in the numerical encoding phase, the query encoding is represented by a family of probability _𝑝𝜃𝑖_ ( _𝑥_ ) density functions on _𝑅𝑑_ , where the parameters are denoted as _𝜃𝑖_ ∈ _𝑅[𝑘]_ . Meanwhile, when we have queries like _Find latitudes of the Canadian cities_ , we are not only asking for some numerical quantity, but also the types or the units of measure of the quantity, like _Degrees_ , _Meters_ , _Dates_ , or _Years_ . Suppose the distribution of type _𝑡_ in the numerical space is described by another distribution _𝜙𝑡_ . 

_3.2.1 Relational Projection and Logic Operations on Entities._ In the number reasoning network, we adopt established query encoding methods as the backbones to conduct relational projections and logic operations for entities. In the implementation, we used three different encoding structures to encode the entity phase [4, 15, 24]. Despite their differences, their query embeddings can be uniformly flattened to vectors. Then, the query encoding in the entity phase is expressed as a _𝑑_ -dimensional vector: _𝑞𝑖_ ∈ _𝑅[𝑑] ._ 

_3.2.2 Attribute Projection._ In the attribute projection, suppose we have query embedding of the entity variable at the _𝑖_ -th step of _𝑞𝑖_ , and we want to obtain its representation on the attribute values on the attribute type _𝑎_ , we can use a neural structure to parameterize the distribution parameters for the _𝑖_ + 1-th step, 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/kdd_2024_Knowledge_Graph_Reasoning_over_Entities_and_Numerical_Values_images/kdd_2024_Knowledge_Graph_Reasoning_over_Entities_and_Numerical_Values.pdf-0004-13.png)


where the _𝐹𝑝_ is a neural structure whose weights will be optimized in the learning process. A typical parameterization of _𝐹𝑝_ is gated transition, which is originally inspired by the recurrent neural network unit [4]. Here are the equations for our parameterization: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/kdd_2024_Knowledge_Graph_Reasoning_over_Entities_and_Numerical_Values_images/kdd_2024_Knowledge_Graph_Reasoning_over_Entities_and_Numerical_Values.pdf-0004-15.png)


Here, _𝑒𝑎_ ∈ _𝑅[𝑘]_ is the embedding used to encode the attribute type _𝑎_ . Meanwhile, _𝜎_ and _𝜙_ are the sigmoid and hyperbolic tangent functions, and ⊙ is the Hadamard product. Also, _𝑊𝑝[𝑝]_[∈] _[𝑅][𝑑]_[·] _[𝑘]_[and] _𝑊𝑧[𝑝][,𝑊] 𝑟[𝑝][,𝑊] ℎ[𝑝][,𝑈] 𝑧[𝑝][,𝑈] 𝑟[𝑝][,𝑈] ℎ[𝑝]_[∈] _[𝑅][𝑘]_[·] _[𝑘]_[are parameter matrices.] 

_3.2.3 Reverse Attribute Projection._ Suppose we want to find entities that have a numerical attribute value from a given set of values. Then we compute the query embedding for the resulting entity set by using the reverse attribute projection operation: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/kdd_2024_Knowledge_Graph_Reasoning_over_Entities_and_Numerical_Values_images/kdd_2024_Knowledge_Graph_Reasoning_over_Entities_and_Numerical_Values.pdf-0004-18.png)


where the _𝐹𝑟_ is a neural structure whose weights will also be optimized in the learning process. Equation (3)’s gated transition function can serve for reverse attribute projection. While the weights are substituted with a different set of parameters _𝑊𝑝[𝑟]_ , _𝑊𝑧[𝑟]_ , _𝑊𝑟[𝑟]_ , _𝑊[𝑟] ℎ_[∈] _[𝑅][𝑘]_[·] _[𝑑]_[and] _[ 𝑈][𝑟𝑧]_[,] _[ 𝑈][𝑟𝑟]_[,] _[ 𝑈] ℎ[𝑟]_[∈] _[𝑅][𝑑]_[·] _[𝑑]_[.] 

_3.2.4 Numerical Projection._ The numerical projections are directly applied to the parameters of distributions. As the numerical constraint of _𝑓_ could be an arbitrary binary function from _𝑅_[2] to {0 _,_ 1}, for now, we use the function _𝐹𝑓_ to express the projection operation: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/kdd_2024_Knowledge_Graph_Reasoning_over_Entities_and_Numerical_Values_images/kdd_2024_Knowledge_Graph_Reasoning_over_Entities_and_Numerical_Values.pdf-0004-21.png)


In a typical parameterization, we can use an embedding _𝑒𝑓_ to encode the characteristics of the numerical function _𝑓_ . The transition of distribution caused the function _𝑓_ to be learned in the training process. For simplicity, we can still use gated transition in Equation (3) while replacing the embedding _𝑒𝑟_ with _𝑒𝑓_ , and replacing their weights with _𝑊𝑗[𝑓][,𝑈] 𝑗[𝑓]_[∈] _[𝑅][𝑘]_[·] _[𝑘]_[, where] _[𝑗]_[∈{] _[𝑧,𝑟,ℎ]_[}][ respectively.] 

KDD ’23, August 6–10, 2023, Long Beach, CA, USA 

Knowledge Graph Reasoning over Entities and Numerical Values 

**Table 2: The statistics of the three knowledge graphs used to construct the benchmark.** 

|Graphs|Data Split|#Nodes|#Rel.|# Attr.|#Rel. Edges|#Attr. Edges|#Num. Edges|#Edges|
|---|---|---|---|---|---|---|---|---|
||Training|25,106|1,345|15|947,540|20,248|27,020|1,015,056|
|FB15k|Validation|26,108|1,345|15|1,065,982|22,779|27,376|1,138,916|
||Testing|27,144|1,345|15|1,184,426|25,311|27,389|1,262,437|
||Training|31,980|279|30|145,262|33,131|25,495|237,019|
|DB15k|Validation|34,191|279|30|161,978|37,269|25,596|262,112|
||Testing|36,358|279|30|178,394|41,411|25,680|286,896|
||Training|32,112|32|7|196,616|21,732|26,616|266,696|
|YAGO15k|Validation|33,078|32|7|221,194|22,748|26,627|293,317|
||Testing|33,610|32|7|245,772|23,520|26,631|319,443|



_3.2.5 Intersection and Union on Attribute Values._ Suppose we have n attribute variables _𝜃_[1] _𝑖[,𝜃] 𝑖_[2] _[, ...,𝜃] 𝑖[𝑛]_[as inputs for intersection opera-] tions. Their intersection and union are expressed as: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/kdd_2024_Knowledge_Graph_Reasoning_over_Entities_and_Numerical_Values_images/kdd_2024_Knowledge_Graph_Reasoning_over_Entities_and_Numerical_Values.pdf-0005-05.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/kdd_2024_Knowledge_Graph_Reasoning_over_Entities_and_Numerical_Values_images/kdd_2024_Knowledge_Graph_Reasoning_over_Entities_and_Numerical_Values.pdf-0005-06.png)


For functions _𝐹𝑖_ and _𝐹𝑢_ , we use DeepSet [38] functions to parameterize them respectively. 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/kdd_2024_Knowledge_Graph_Reasoning_over_Entities_and_Numerical_Values_images/kdd_2024_Knowledge_Graph_Reasoning_over_Entities_and_Numerical_Values.pdf-0005-08.png)


The intersection and union operations are thus invariant to input variable permutations. In our implementation, we use a selfattention network followed by a feed-forward network to implement such permutation invariant DeepSet function. Suppose _𝜃𝑖_ = [ _𝜃𝑖_[1] _[,𝜃] 𝑖_[2] _[, ...,𝜃] 𝑖[𝑛]_[]][, then their formulas are as follows:] 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/kdd_2024_Knowledge_Graph_Reasoning_over_Entities_and_Numerical_Values_images/kdd_2024_Knowledge_Graph_Reasoning_over_Entities_and_Numerical_Values.pdf-0005-10.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/kdd_2024_Knowledge_Graph_Reasoning_over_Entities_and_Numerical_Values_images/kdd_2024_Knowledge_Graph_Reasoning_over_Entities_and_Numerical_Values.pdf-0005-11.png)


The _𝑊𝑞,𝑊𝑘,𝑊𝑣_ ∈ _𝑅[𝑘]_[×] _[𝑘]_ are parameters used for modeling the input Query, Key, and Value for the attention module Attn. The Attn represents the scaled dot-product attention, 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/kdd_2024_Knowledge_Graph_Reasoning_over_Entities_and_Numerical_Values_images/kdd_2024_Knowledge_Graph_Reasoning_over_Entities_and_Numerical_Values.pdf-0005-13.png)


Here, the _𝑄_ , _𝐾_ , and _𝑉_ represent the input Query, Key, and Value for this attention layer. The MLP here denotes a multi-layer perceptron layer with ReLU activation. 

_3.2.6 Learning Attribute Reasoning._ As there are two types of queries with different types of target variables, we define two loss functions respectively. Suppose there are total _𝐼_ steps in the computational graphs, the target variable is an entity variable, and its query embedding is _𝑞𝐼_ , then we compute the normalized probability of the entity _𝑣_ being the correct answer to the query _𝑞_ by using the softmax function on all similarity scores, 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/kdd_2024_Knowledge_Graph_Reasoning_over_Entities_and_Numerical_Values_images/kdd_2024_Knowledge_Graph_Reasoning_over_Entities_and_Numerical_Values.pdf-0005-16.png)


Then the objective function of entities is expressed as 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/kdd_2024_Knowledge_Graph_Reasoning_over_Entities_and_Numerical_Values_images/kdd_2024_Knowledge_Graph_Reasoning_over_Entities_and_Numerical_Values.pdf-0005-18.png)


Here, each ( _𝑞_[(] _[𝑖]_[)] _, 𝑣_[(] _[𝑖]_[)] ) is one of the positive query-answer parties, and there are _𝑁_ such pairs. 

Meanwhile, for the numerical variables, we are going to use the maximize a posteriori probability (MAP) estimation to derive an objective function for type-aware attribute reasoning. Suppose the parameters of the attribute distribution of the final step are _𝜃𝐼_ , the positive answer is _𝑣_ , and the answer has a value type _𝑡_ . The _𝜃𝐼_ is computed from _𝜃𝑖_ and _𝑞𝑖_ where _𝑖_ ∈{1 _,_ 2 _,_ 3 _, ..., 𝐼_ − 1} by using a series of projections like _𝐹𝑝_ , _𝐹𝑟_ , _𝐹𝑓_ , and logic operations _𝐹𝑖_ and _𝐹𝑢_ . Then our goal is to optimize _𝜃𝐼_ given _𝑣_ and _𝑡_ . Then suppose _𝑓_ and _𝑔_ are the conditional distribution of _𝑣_ given _𝜃𝐼_ and _𝑡_ , and _𝜃𝐼_ given _𝑡_ , and meanwhile _𝑣_ is conditional independent of _𝑡_ given _𝜃𝐼_ . Then according to MAP estimation, 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/kdd_2024_Knowledge_Graph_Reasoning_over_Entities_and_Numerical_Values_images/kdd_2024_Knowledge_Graph_Reasoning_over_Entities_and_Numerical_Values.pdf-0005-21.png)


Suppose ( _𝑞_[(] _[𝑗]_[)] _, 𝑣_[(] _[𝑗]_[)] ) is a pair of positive query-answer pairs, and we have totally M such pairs. Suppose the value type of _𝑣_[(] _[𝑖]_[)] is _𝑡_[(] _[𝑖]_[)] . Using the previous defined notions, then _𝑓_ ( _𝑣_[(] _[𝑗]_[)] | _𝜃𝐼_[(] _[𝑗]_[)] ) can be written as _𝑝𝜃𝐼_ ( _𝑗_ ) ( _𝑣_[(] _[𝑗]_[)] ), and _𝑔_ ( _𝜃𝐼_ | _𝑡_ )) can be written as _𝜙𝑡_ ( _𝑗_ ) ( _𝜃𝐼_[(] _[𝑗]_[)] ). Then, the loss of explicit queries can be described by 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/kdd_2024_Knowledge_Graph_Reasoning_over_Entities_and_Numerical_Values_images/kdd_2024_Knowledge_Graph_Reasoning_over_Entities_and_Numerical_Values.pdf-0005-23.png)


where the _𝜓_ (·) is numerical encoding defined on the domain of _𝑅_ with the range of _𝑅[𝑑]_ . In the training process, we alternatively optimize these two loss functions to train all parameters in different neural structures together. 

## **4 BENCHMARK CONSTRUCTION** 

In this section, we construct three benchmark datasets for evaluating the performance of numerical complex query answering. 

KDD ’23, August 6–10, 2023, Long Beach, CA, USA 

Jiaxin Bai, Chen Luo, Zheng Li, Qingyu Yin, Bing Yin, and Yangqiu Song 

**Table 3: The statistics of the queries sampled from the FB15k, DB15k, and YAGO15k knowledge graphs.** 

|Graphs|Data Split|1p|2p|2i|3i|pi|ip|2u|up|All|
|---|---|---|---|---|---|---|---|---|---|---|
||Training|304,633|138,192|226,729|288,874|260,057|233,834|284,301|284,931|2,021,551|
|FB15k|Validation|8,271|15,860|23,359|28,836|25,081|22,930|29,187|29,210|182,734|
||Testing|7,969|15,431|23,346|28,865|24,810|22,232|29,212|29,274|181,139|
||Training|124,851|99,698|140,427|190,413|171,353|163,687|190,364|194,244|1,275,037|
|DB15k|Validation|3,529|10,388|9,792|13,817|14,594|16,651|19,512|19,792|108,075|
||Testing|3,387|10,047|9,914|14,603|14,642|15,897|19,504|19,773|107,767|
||Training|84,014|76,238|136,282|183,850|162,712|145,994|183,963|183,459|1,156,512|
|YAGO15k|Validation|2,833|7,986|10,757|16,884|13,485|13,899|18,444|19,105|103,393|
||Testing|2,713|7,949|10,935|17,171|13,481|13,526|18,433|18,997|103,205|




![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/kdd_2024_Knowledge_Graph_Reasoning_over_Entities_and_Numerical_Values_images/kdd_2024_Knowledge_Graph_Reasoning_over_Entities_and_Numerical_Values.pdf-0006-04.png)


**----- Start of picture text -----**<br>
u<br>1p 2p 2u up<br>2i ip 3i pi<br>nv rap nv np nv rap e rp nv np nv rap<br>e e<br>e nv<br>e e e rp e rp<br>rp ap<br>nv rap e rp e ap nv rap e ap nv np<br>e e e nv<br>e rp e rp e rp e ap<br>… …<br>**----- End of picture text -----**<br>


**Figure 3: Eight general query types. Each general query type corresponds to multiple specific query types.** 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/kdd_2024_Knowledge_Graph_Reasoning_over_Entities_and_Numerical_Values_images/kdd_2024_Knowledge_Graph_Reasoning_over_Entities_and_Numerical_Values.pdf-0006-06.png)


**----- Start of picture text -----**<br>
FB15k DB15k<br>5000 8000 12000 [YAGO15k]<br>4000 10000<br>6000<br>8000<br>3000<br>4000 6000<br>2000<br>4000<br>2000<br>1000 2000<br>0 0 0<br>Degrees Integer Double Integer Float Degrees Date<br>Year Date<br>**----- End of picture text -----**<br>


**Figure 4: The distribution of the types of numerical values on three different knowledge graphs.** 

into various types of specific types of queries. As shown in Figure 3, _e_ stands for an entity, _nv_ stands for numerical values, and _ap_ , _rap_ , _rp_ , and _np_ are the abbreviations of _attribute projection_ , _reverse attribute projection_ , _relational projection_ , and _numerical projections_ respectively. The general query types are shown in Table 4. 

## **4.1 Knowledge Graphs** 

We use FB15k, DB15k, and YAGO15k [17] as the knowledge graphs to create our benchmarks. These KGs are publicly available, and they include both triples describing entity relations and numerical values. For each knowledge graph, we first randomly divide the edges into training, validation, and testing edges with a ratio of 8:1:1 respectively. Then, the training graph G _𝑡𝑟𝑎𝑖𝑛_ , validation graph G _𝑣𝑎𝑙_ , and testing graph G _𝑡𝑒𝑠𝑡_ are aggregated from the training edges, training+validation edges, and training+validation+testing edges respectively. The detailed statistics of the knowledge graphs are shown in Table 2. The columns #Rel. and #Attr. denote the number of entity relation types and the number of attribute types respectively. #Rel. Edges, #Attr. Edges, and #Num. Edges represent the total number of edges describing entity relations, attribute values, and numerical relations between values. 

## **4.2 Query Types** 

Following previous work [15], we used the following eight types of query as general query types. For each general query type, each edge represents a projection, and each node either represents either a set of entities or numerical values. Based on these query types and knowledge graph structures, we further derive various specific sub-types. For example, the query type _pi_ can be instantiated 

## **4.3 Sampling of Complex Attribute Queries** 

In this section, we show how to sample numerical complex queries. Before sampling, we added dummy numerical edges to the graphs to evaluate the performance on numerical relations. For simplicity, we consider seven types of numerical relations: _EqualTo_ , _SmallerThan_ , _GreaterThan_ , _TwiceEqualTo_ , _ThreeTimesEqualTo_ , _TwiceGreaterThan_ , and _ThreeTimesGreaterThan_ . These numerical relations are randomly added to the attribute value pairs with the same value type and satisfy the corresponding constraints. For each type of numerical relation, we set an upper limit of four thousand edges on each graph. The total number of numerical edges added is shown in Table 2. There are two general steps in sampling the complex attribute queries. In the first step, we are given eight types of general queries with abbreviations of _1p_ , _2p_ , _2i_ , _3i_ , _ip_ , _ip_ , _2u_ , and _up_ . Their definitions are given in Table 3. Given the general query type and a knowledge graph _𝐺_ , a sampling algorithm is used to generate a complex attribute query that has at least one answer. The details of the algorithm are given in the Appendix. In the second step, given a complex attribute query from the first step, we conduct a graph search and find its corresponding answers on the KG. 

For each knowledge graph, we have constructed their corresponding Training, Validation, and Testing graphs. Then, the first 

KDD ’23, August 6–10, 2023, Long Beach, CA, USA 

Knowledge Graph Reasoning over Entities and Numerical Values 

**Table 4: The abbreviations of the query types and structures.** 

|Abbr.<br>General Query Type<br>Specifc Query Type Example|Abbr.<br>General Query Type<br>Specifc Query Type Example|Abbr.<br>General Query Type<br>Specifc Query Type Example|
|---|---|---|
|1p<br>2p<br>2i<br>3i<br>ip<br>pi<br>2u<br>up|(p,(e))<br>(p,(p,(e)))<br>(i,(p,(e)),(p,(e)))<br>(i,(p,(e)),(p,(e)),(p,(e)))<br>(p,(i,(p,(e)),(p,(e))))<br>(i,(p,(e)),(p,(p,(e))))<br>(u,(p,(e)),(p,(e)))<br>(p,(u,(p,(e)),(p,(e))))|(rap,(nv))<br>(rap, (np,(nv)))<br>(i, (np,(nv)), (ap, (e)))<br>(i, (np,(nv)), (ap, (e)), (ap, (e)))<br>(rp, (i, (np,(nv)), (ap, (e))))<br>(i, (np,(nv)), (ap, (rp, (e))))<br>(u, (np,(nv)), (ap, (e)))<br>(rp, (u, (np,(nv)), (ap, (e))))|



step of the previously described query sampling process is conducted on these three data splits independently. After this, we obtained the training queries, validation queries, and test queries respectively. Then, For the training queries, we conduct a graph search on the training graph to find corresponding training answers. For the validation queries, we search their answers on both the training graph and the validation graph, and we only use the queries that have a different number of training and validation answers. Finally, for the testing queries, the graph searches are conducted on training, validation, and testing respectively. Similarly, we only use the testing queries that have a different number of answers on validation and test graphs. After these two steps are done, the statistics of the resulting training, validation, and testing queries of three knowledge graphs are shown in Table 3. 

## **5 EXPERIMENT** 

In this section, we implement the proposed number reasoning network (NRN) and compare it to existing query encoding models. 

## **5.1 Baseline Methods** 

NRN uses previous query encoding methods to encode entities during the entity encoding phase. It can be combined with a wide range of entity encoding structures that can be transformed into a fixed length of real-valued vectors. We use the following methods: 

- GQE [15]: The graph query encoding model encodes a complex logic query into a single vector; 

- Q2B [24]: The Query2Box model encodes a complex query into hyper-rectangles in the embedding space; 

- Q2P [4]: The Query2Particles model encodes a complex logic query into multiple vectors. 

## **5.2 Numerical Encoding** 

As explained in §3.2, we use numerical encoding methods to project the attribute values from _𝑅_ to _𝑅[𝐷]_ to obtain number embedding for NRN. We use the following two deterministic numerical encodings. 

- DICE [28]: DICE is a deterministic independent-of-corpus word embedding of numbers. DICE managed to use a linear mapping from the numerical values _𝑣_ to an angle _𝛼_ ∈[0 _, 𝜋_ ], and then conduct a polar-to-Cartesian transformation in the 

- _𝐷_ -dimensional space according to the following formula: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/kdd_2024_Knowledge_Graph_Reasoning_over_Entities_and_Numerical_Values_images/kdd_2024_Knowledge_Graph_Reasoning_over_Entities_and_Numerical_Values.pdf-0007-16.png)


**Table 5: The main experiment results of numerical complex query answering. NRN constantly outperforms the baseline methods on Hit@1, Hit@3, Hit@10, and MRR.** 

|Dataset|Entity|Value|H@1<br>H@3<br>H@10<br>MRR|
|---|---|---|---|
|FB15k|GQE|Baseline<br>NRN + DICE<br>NRN + Sinusoidal|13.56<br>23.47<br>34.80<br>20.80<br>14.16<br>24.69<br>36.18<br>21.72<br>**14.33**<br>**24.81**<br>**36.25**<br>**21.86**|
||Q2B|Baseline<br>NRN + DICE<br>NRN + Sinusoidal|14.49<br>25.96<br>38.20<br>22.56<br>15.55<br>27.31<br>39.24<br>23.72<br>**15.78**<br>**27.65**<br>**39.54**<br>**23.96**|
||Q2P|Baseline<br>NRN + DICE<br>NRN + Sinusoidal|15.54<br>25.50<br>36.88<br>22.84<br>16.98<br>27.02<br>38.36<br>24.27<br>**17.72**<br>**27.96**<br>**39.23**<br>**25.09**|
|DB15k|GQE|Baseline<br>NRN + DICE<br>NRN + Sinusoidal|6.86<br>12.98<br>21.28<br>11.73<br>7.26<br>13.36<br>21.81<br>12.15<br>7.30<br>13.63<br>**22.32**<br>**12.34**|
||Q2B|Baseline<br>NRN + DICE<br>NRN + Sinusoidal|7.47<br>14.17<br>22.81<br>12.66<br>**8.07**<br>**15.21**<br>**24.22**<br>**13.53**<br>8.05<br>15.04<br>24.05<br>13.45|
||Q2P|Baseline<br>NRN + DICE<br>NRN + Sinusoidal|8.41<br>14.57<br>22.84<br>13.31<br>8.81<br>15.10<br>23.27<br>13.73<br>**9.13**<br>**15.69**<br>**24.22**<br>**14.25**|
|YAGO15k|GQE|Baseline<br>NRN + DICE<br>NRN + Sinusoidal|10.56<br>18.12<br>27.65<br>16.35<br>11.68<br>19.49<br>29.05<br>17.59<br>**11.78**<br>**19.74**<br>**29.12**<br>**17.73**|
||Q2B|Baseline<br>NRN + DICE<br>NRN + Sinusoidal|13.47<br>22.67<br>32.56<br>20.02<br>13.94<br>23.74<br>33.56<br>20.77<br>**14.41**<br>**23.96**<br>**33.78**<br>**21.13**|
||Q2P|Baseline<br>NRN + DICE<br>NRN + Sinusoidal|6.70<br>11.98<br>20.11<br>11.29<br>9.80<br>**16.98**<br>**26.74**<br>**15.53**<br>**9.90**<br>16.83<br>26.43<br>15.49|



- Sinusoidal [30]: The Sinusoidal encoding of value is first proposed in encoding token position in the transformer model. In this paper, we found that this function is also effective in encoding numerical values. Its formula is expressed by: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/kdd_2024_Knowledge_Graph_Reasoning_over_Entities_and_Numerical_Values_images/kdd_2024_Knowledge_Graph_Reasoning_over_Entities_and_Numerical_Values.pdf-0007-20.png)


These two functions are both deterministic and preserve geometric properties from the original value space to the embedding space. 

## **5.3 Evaluation Metrics** 

For a testing query _𝑞_ , the training, validation, and testing answers are denoted as [ _𝑞_ ] _𝑡𝑟𝑎𝑖𝑛_ , [ _𝑞_ ] _𝑣𝑎𝑙_ , and [ _𝑞_ ] _𝑡𝑒𝑠𝑡_ respectively. In the experiment, we evaluate the generalization capability of the attribute reasoning models by computing the rankings of the answers that cannot be directly searched from an observed knowledge graph. Suppose [ _𝑞_ ] _𝑣𝑎𝑙_ /[ _𝑞_ ] _𝑡𝑟𝑎𝑖𝑛_ represents the answer set of the query _𝑞_ that can be searched from the validation graph but cannot be searched from the training graph. Similarly, [ _𝑞_ ] _𝑡𝑒𝑠𝑡_ /[ _𝑞_ ] _𝑣𝑎𝑙_ is the set of answers that can be found on the test graph but cannot be found in the validation graph. Then, the evaluation metrics of the test query _𝑞_ can be expressed in the following equation: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/kdd_2024_Knowledge_Graph_Reasoning_over_Entities_and_Numerical_Values_images/kdd_2024_Knowledge_Graph_Reasoning_over_Entities_and_Numerical_Values.pdf-0007-24.png)


When the evaluation metric is Hit@K, the _𝑚_ ( _𝑟_ ) in Equation (17) is defined as _𝑚_ ( _𝑟_ ) = **1** [ _𝑟_ ≤ _𝐾_ ]. In other words, _𝑚_ ( _𝑟_ ) = 1 if _𝑟_ ≤ _𝐾_ , 

KDD ’23, August 6–10, 2023, Long Beach, CA, USA 

Jiaxin Bai, Chen Luo, Zheng Li, Qingyu Yin, Bing Yin, and Yangqiu Song 

**Table 6: The mean reciprocal ranking (MRR) results for different types of numerical complex queries.** 

|Dataset|Entity Encoding|Values Encoding|1p<br>2p<br>2i<br>3i<br>pi<br>ip<br>2u<br>up|
|---|---|---|---|
|FB15k|GQE|Baseline<br>NRN + DICE<br>NRN + Sinusoidal|30.63<br>7.54<br>32.70<br>39.61<br>21.57<br>8.87<br>7.18<br>4.17<br>**31.75**<br>**8.16**<br>33.96<br>40.32<br>22.70<br>9.80<br>**8.45**<br>**4.66**<br>31.69<br>8.15<br>**33.99**<br>**41.19**<br>**23.10**<br>**10.10**<br>7.57<br>4.51|
||Q2B|Baseline<br>NRN + DICE<br>NRN + Sinusoidal|35.16<br>7.12<br>35.39<br>41.31<br>22.99<br>9.17<br>12.76<br>4.02<br>36.96<br>7.66<br>36.50<br>42.30<br>24.17<br>**10.41**<br>14.53<br>**4.79**<br>**37.01**<br>**7.74**<br>**36.88**<br>**42.72**<br>**24.25**<br>10.09<br>**15.61**<br>4.54|
||Q2P|Baseline<br>NRN + DICE<br>+ Sinusoidal|39.29<br>12.45<br>30.40<br>35.29<br>20.09<br>11.05<br>21.25<br>7.80<br>41.45<br>12.65<br>32.91<br>37.61<br>22.02<br>12.85<br>21.74<br>**8.03**<br>**42.75**<br>**12.87**<br>**33.71**<br>**38.75**<br>**23.14**<br>**13.23**<br>**23.16**<br>7.89|
|DB15k|GQE|Baseline<br>NRN + DICE<br>NRN + Sinusoidal|9.83<br>2.41<br>18.83<br>34.48<br>11.21<br>2.11<br>1.86<br>1.94<br>**10.46**<br>**2.58**<br>**20.17**<br>34.80<br>11.88<br>2.48<br>1.94<br>2.03<br>10.29<br>2.53<br>20.14<br>**35.46**<br>**12.50**<br>**2.52**<br>**2.08**<br>**2.14**|
||Q2B|Baseline<br>NRN + DICE<br>NRN + Sinusoidal|10.18<br>2.53<br>20.81<br>36.29<br>12.57<br>2.72<br>2.06<br>1.67<br>**11.01**<br>2.66<br>22.38<br>**37.47**<br>**14.17**<br>**3.09**<br>**2.62**<br>**2.20**<br>10.96<br>**2.71**<br>**22.60**<br>37.44<br>13.81<br>3.05<br>2.41<br>2.13|
||Q2P|Baseline<br>NRN + DICE<br>NRN + Sinusoidal|14.44<br>**3.96**<br>20.67<br>33.85<br>13.64<br>**3.48**<br>4.66<br>**2.96**<br>14.58<br>3.81<br>22.15<br>35.36<br>13.90<br>2.83<br>**4.94**<br>2.67<br>**14.71**<br>3.81<br>**23.75**<br>**36.66**<br>**14.47**<br>2.96<br>4.63<br>2.81|
|YAGO15k|GQE|Baseline<br>NRN + DICE<br>NRN + Sinusoidal|14.28<br>3.18<br>32.83<br>37.83<br>15.91<br>4.89<br>3.62<br>1.82<br>**15.21**<br>4.20<br>35.08<br>**40.32**<br>17.01<br>**5.71**<br>3.95<br>1.95<br>14.79<br>**4.23**<br>**35.68**<br>39.64<br>**18.29**<br>5.65<br>4.23<br>**1.96**|
||Q2B|Baseline<br>NRN + DICE<br>NRN + Sinusoidal|18.84<br>3.91<br>38.62<br>44.67<br>18.72<br>7.31<br>6.90<br>2.47<br>20.83<br>4.53<br>38.87<br>**45.19**<br>**20.61**<br>7.69<br>8.72<br>2.73<br>**21.40**<br>**4.59**<br>**39.72**<br>45.16<br>19.62<br>**7.90**<br>**9.05**<br>**2.82**|
||Q2P|Baseline<br>NRN + DICE<br>NRN + Sinusoidal|19.84<br>4.64<br>17.42<br>19.34<br>11.14<br>4.33<br>9.92<br>3.07<br>22.68<br>5.36<br>25.45<br>**29.70**<br>14.07<br>**5.67**<br>11.73<br>3.47<br>**22.97**<br>**5.70**<br>**25.70**<br>29.04<br>**14.38**<br>5.40<br>**11.87**<br>**3.51**|



otherwise _𝑚_ ( _𝑟_ ) = 0. Meanwhile, if the evaluation metric is mean reciprocal ranking (MRR), then the _𝑚_ ( _𝑟_ ) is defined as _𝑚_ ( _𝑟_ ) = _𝑟_[1][.] Similarly, if _𝑞_ is a validation query, its generalization capability is evaluated on [ _𝑞_ ] _𝑣𝑎𝑙_ /[ _𝑞_ ] _𝑡𝑟𝑎𝑖𝑛_ instead of [ _𝑞_ ] _𝑡𝑒𝑠𝑡_ /[ _𝑞_ ] _𝑣𝑎𝑙_ . In our experiment, we train all the models by using the training queries, and we select the hyper-parameters by using the validation queries. The evaluation is then finally conducted on the testing queries. 

## **5.4 Experiment Results** 

As discussed in §5.1, we use three different query encoding methods GQE [15], Q2B [24], and Q2P [4] as the backbone for entities in our framework. Meanwhile, to evaluate the performance of NRN, we also use the original version of the GQE, Q2B, and Q2P methods as baselines. In these baselines, all attribute values in the knowledge graphs are treated in the same way as other entities in the knowledge graph. In our proposed NRN framework, we use two ways of incorporating attribute value information in the reasoning model. First, we use the DICE [28] and Sinusoidal [30] function to encode attribute values as the numerical encoding function _𝜓_ ( _𝑣_ ) in Equation (14). Meanwhile, for simplicity, we choose to use the multi-variant Gaussian distributions as the distribution family used for the distribution of numerical queries _𝑝𝜃_ ( _𝑥_ ) and the distribution of the parameters given the value type _𝜙𝑡_ ( _𝜃_ ) in Equation (14). Experiment results are reported in Table 5 and Table 6. 

Table 5 reports the averaged results in Hit@1, Hit@3, Hit@10, and MRR. The experiment results show that the number reasoning 

networks constantly outperform their baseline reasoning methods on three different knowledge graphs under four different evaluation metrics. The difference between the baseline query encoding method and our number reasoning network is that we use parameterized distributions to encode the numerical attribute values instead of using a unified query encoding structure. As a result, from the empirical results, we can conclude that parameterized distributions are effective encoding structures for numerical attributes. On the other hand, the Sinusoidal function performs on par or even better than DICE on different KGs under various metrics. Thus Sinusoidal is a simple yet effective method to encode the magnitude of knowledge graph attribute values. Moreover, the mean reciprocal rank (MRR) scores for different types of queries are shown in Table 6. Empirical results show that NRN performs better than the baselines on almost every type of query on the three datasets. 

## **5.5 Computing Time Analysis** 

Further analysis on computing time is also conducted. In this experiment, we measure the average training and inference time on FB15k, DB15k, and YAGO15K with the Q2P. The experiment results are shown in Table 7. We run our experiments on the RTX2080ti graphics card with a batch size of 1024. Then we compute the average training and inference time per query using the batch processing time divided by the batch size. The empirical results show that the average inference time of NRN is 0.11 milliseconds longer than the baseline, and the training time is 0.16 milliseconds longer. Although 

KDD ’23, August 6–10, 2023, Long Beach, CA, USA 

Knowledge Graph Reasoning over Entities and Numerical Values 

**Table 7: The average training time and inference time per query in milliseconds (ms) of complex numerical queries. Although the training and inference are slower than the baseline, NRN still achieves high training and inference speed.** 

|Time (ms)|FB15k<br>Training<br>Inference|DB15k<br>Training<br>Inference|YAGO15k<br>Training<br>Inference|Average<br>Training<br>Inference|
|---|---|---|---|---|
|Baseline<br>NRN|0.090<br>0.042<br>0.253<br>0.149|0.078<br>0.031<br>0.229<br>0.134|0.088<br>0.040<br>0.255<br>0.152|0.085<br>0.037<br>0.245<br>0.145|



the computing time is longer than the baseline model, the NRN is still fast and effective. 

## **6 RELATED WORK** 

The query encoding methods are closely related to this work. Some methods use different geometric to encode queries. The GQE [15] model encodes a KG query as a vector in embedding space. Query2Box [24] encodes a KG query as a hyper-rectangle in an embedding space. It can be used for answering existential positive first-order (EPFO) complex queries. EmQL [27] proposes to use count sketches to improve the faithfulness of the query embeddings by using countmin sketches. ConE [39] improves the Q2B by using cone embeddings. The cone embeddings are able to express negation and encode arbitrary first-order logic queries. CylE [22] proposes to use cylinders to encode complex queries. Q2P [4] encodes a logic query into multiple vectors in the embeddings space. Meanwhile, HypE [11] encodes the complex queries as hyperbolic query embedding. Wang et al. [32] propose Wasserstein-Fisher-Rao Embedding for CQA. Meanwhile, some methods try to encode logic queries into specific probabilistic structures. Beta Embedding [25] proposes to use multiple Beta distributions to encode logic knowledge graph queries expressed in First-order logic queries. Yang et al. [37] propose to use Gamma distributions to encode complex queries. Then, PERM [10] embeddings encode the complex queries by using a mixture of Gaussians. Recently, Line Embedding [16] is proposed to relax the distributional assumptions by using a logic space transformation layer to convert probabilistic embeddings to LinE space embedding. Meanwhile, there is some work attempting to use neural structures to encode complex queries. The transformer structure with specially designed graph hierarchy encoding is proposed by BiQE [18] for complex queries. Meanwhile, Bai et al. [5] propose to linearize the computational graph and then use sequence encoders to encode queries. Newlook [20] proposes to use different types of neural networks to iteratively encode complex queries. Simultaneously, [1] managed to use MLP and MLP-Mixers [29] to achieve improved performance. ENeSy [36] uses an entangled method of the neural and symbolic method to conduct query encoding. QE-GNN [40] uses the message passing over KG to conduct query encoding. The Mask-and-Reasoning [21] proposes to use a knowledge graph transformer, which is a type of graph neural network, to conduct pre-training on the knowledge graph. The pre-trained KG transformer is then fine-tuned to answer complex queries. 

Meanwhile, query decomposition [2] is another line of research for answering complex KG queries. In the query decomposition method, a complex query is first decomposed into atomic queries. Then the probabilities of atomic queries are computed by a neural link predictor. Recently, Wang et al. [33] propose to use one-hop message passing on query graphs to conduct complex query answering. In the inference process, either continuous optimization 

or beam search is used for finding the answers. More recently, SMORE [23] is proposed as a framework to train and evaluate large scaled knowledge graph completion and complex query answering models. The compositional generalization capability of complex query answering is bench-evaluated by the EFO-1 dataset [34]. On the other hand, ROMA [35] is proposed to answer complex logic queries on multi-view knowledge graphs. Though existing methods can effectively conduct reasoning on entities and their relations on a knowledge graph, they cannot be directly used to reasonably answer numerical complex queries. Recently, Bai et al. [3] propose to use memory-enhanced query encoding for complex query answering on eventuality knowledge graphs. 

Prior research has explored the treatment of numerical attributes in knowledge graphs. KBLRN [13] proposes to learn the representation of knowledge bases by jointly using latent, relational, and numerical features. KR-EAR [19] proposes a knowledge representation model to jointly learn the entity, relations, and attributes in knowledge graphs. Numerical value propagation [17] is proposed to predict the value of entities’ numerical attributes. Multi-relational attribute propagation [6] is then proposed to improve attribute completion by using message passing on a knowledge graph. However, these methods are not able to answer complex logic queries. Duan et al. [12] propose to use the KG embedding methods to train the representations of numerical values over synthetic numerical relations, and this method can is an alternative to the DICE and Sinusoidal for encoding real numbers for NRN. Meanwhile, NeuralNum-LP [31] proposes to learn multi-hop rules involving numerical attributes from a knowledge graph. However, it still cannot deal with complex logic queries. 

## **7 CONCLUSION** 

In this work, we proposed the new task of numerical complex attribute query answering (Numerical CQA). Meanwhile, we constructed a benchmark dataset based on three public KGs: FB15k, YAGO15k, and DBpedia15k. Finally, we proposed a new framework that can conduct numerical attribute value answering. Experiments show that our number encoding network (NRN) with two-phase query encoding can significantly outperform the previous query encoding methods and achieve state-of-the-art Numerical CQA. 

## **ACKNOWLEDGMENTS** 

The authors of this paper are supported by the NSFC Fund (U20B2053) from the NSFC of China, the RIF (R6020-19 and R6021-20), and the GRF (16211520 and 16205322) from RGC of Hong Kong, the MHKJFS (MHP/001/19) from ITC of Hong Kong and the National Key R&D Program of China (2019YFE0198200) with special thanks to HKMAAC and CUSBLT. We also thank the UGC Research Matching Grants (RMGS20EG01-D, RMGS20CR11, RMGS20CR12, RMGS20EG19, RMGS20EG21, RMGS23CR05, RMGS23EG08). 

KDD ’23, August 6–10, 2023, Long Beach, CA, USA 

Jiaxin Bai, Chen Luo, Zheng Li, Qingyu Yin, Bing Yin, and Yangqiu Song 

## **REFERENCES** 

- [1] Alfonso Amayuelas, Shuai Zhang, Susie Xi Rao, and Ce Zhang. 2022. Neural Methods for Logical Reasoning over Knowledge Graphs. In _The Tenth International Conference on Learning Representations, ICLR 2022, Virtual Event, April 25-29, 2022_ . OpenReview.net. https://openreview.net/forum?id=tgcAoUVHRIB 

- [2] Erik Arakelyan, Daniel Daza, Pasquale Minervini, and Michael Cochez. 2021. Complex Query Answering with Neural Link Predictors. In _9th International Conference on Learning Representations, ICLR 2021, Virtual Event, Austria, May 3-7, 2021_ . OpenReview.net. https://openreview.net/forum?id=Mos9F9kDwkz 

- [3] Jiaxin Bai, Xin Liu, Weiqi Wang, Chen Luo, and Yangqiu Song. 2023. Complex Query Answering on Eventuality Knowledge Graph with Implicit Logical Constraints. arXiv:2305.19068 [cs.CL] 

- [4] Jiaxin Bai, Zihao Wang, Hongming Zhang, and Yangqiu Song. 2022. Query2Particles: Knowledge Graph Reasoning with Particle Embeddings. In _Findings of the Association for Computational Linguistics: NAACL 2022_ . Association for Computational Linguistics, Seattle, United States, 2703–2714. https: //doi.org/10.18653/v1/2022.findings-naacl.207 

- [5] Jiaxin Bai, Tianshi Zheng, and Yangqiu Song. 2023. Sequential Query Encoding For Complex Query Answering on Knowledge Graphs. _CoRR_ abs/2302.13114 (2023). https://doi.org/10.48550/arXiv.2302.13114 arXiv:2302.13114 

- [6] Eda Bayram, Alberto García-Durán, and Robert West. 2021. Node attribute completion in knowledge graphs with multi-relational propagation. In _ICASSP 2021-2021 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)_ . IEEE, 3590–3594. 

- [7] Christian Bizer, Jens Lehmann, Georgi Kobilarov, Sören Auer, Christian Becker, Richard Cyganiak, and Sebastian Hellmann. 2009. Dbpedia-a crystallization point for the web of data. _Journal of web semantics_ 7, 3 (2009), 154–165. 

- [8] Kurt D. Bollacker, Colin Evans, Praveen K. Paritosh, Tim Sturge, and Jamie Taylor. 2008. Freebase: a collaboratively created graph database for structuring human knowledge. In _Proceedings of the ACM SIGMOD International Conference on Management of Data, SIGMOD 2008, Vancouver, BC, Canada, June 10-12, 2008_ , Jason Tsong-Li Wang (Ed.). ACM, 1247–1250. https://doi.org/10.1145/1376616. 1376746 

- [9] Xiaojun Chen, Shengbin Jia, and Yang Xiang. 2020. A review: Knowledge reasoning over knowledge graph. _Expert Syst. Appl._ 141 (2020). https://doi.org/10.1016/ j.eswa.2019.112948 

- [10] Nurendra Choudhary, Nikhil Rao, Sumeet Katariya, Karthik Subbian, and Chandan K. Reddy. 2021. Probabilistic Entity Representation Model for Reasoning over Knowledge Graphs. In _Advances in Neural Information Processing Systems 34: Annual Conference on Neural Information Processing Systems 2021, NeurIPS 2021, December 6-14, 2021, virtual_ , Marc’Aurelio Ranzato, Alina Beygelzimer, Yann N. Dauphin, Percy Liang, and Jennifer Wortman Vaughan (Eds.). 23440–23451. https://proceedings.neurips.cc/paper/2021/hash/ c4d2ce3f3ebb5393a77c33c0cd95dc93-Abstract.html 

- [11] Nurendra Choudhary, Nikhil Rao, Sumeet Katariya, Karthik Subbian, and Chandan K. Reddy. 2021. Self-Supervised Hyperboloid Representations from Logical Queries over Knowledge Graphs. In _WWW ’21: The Web Conference 2021, Virtual Event / Ljubljana, Slovenia, April 19-23, 2021_ , Jure Leskovec, Marko Grobelnik, Marc Najork, Jie Tang, and Leila Zia (Eds.). ACM / IW3C2, 1373–1384. https://doi.org/10.1145/3442381.3449974 

- [12] Hanyu Duan, Yi Yang, and Kar Yan Tam. 2021. Learning Numeracy: A Simple Yet Effective Number Embedding Approach Using Knowledge Graph. In _Findings of the Association for Computational Linguistics: EMNLP 2021, Virtual Event / Punta Cana, Dominican Republic, 16-20 November, 2021_ , Marie-Francine Moens, Xuanjing Huang, Lucia Specia, and Scott Wen-tau Yih (Eds.). Association for Computational Linguistics, 2597–2602. https://doi.org/10.18653/v1/2021.findingsemnlp.221 

- [13] Alberto García-Durán and Mathias Niepert. 2018. KBlrn: End-to-End Learning of Knowledge Base Representations with Latent, Relational, and Numerical Features. In _Proceedings of the Thirty-Fourth Conference on Uncertainty in Artificial Intelligence, UAI 2018, Monterey, California, USA, August 6-10, 2018_ , Amir Globerson and Ricardo Silva (Eds.). AUAI Press, 372–381. http://auai.org/uai2018/proceedings/ papers/149.pdf 

- [14] Jia Guo and Stanley Kok. 2021. BiQUE: Biquaternionic Embeddings of Knowledge Graphs. In _Proceedings of the 2021 Conference on Empirical Methods in Natural Language Processing, EMNLP 2021, Virtual Event / Punta Cana, Dominican Republic, 7-11 November, 2021_ , Marie-Francine Moens, Xuanjing Huang, Lucia Specia, and Scott Wen-tau Yih (Eds.). Association for Computational Linguistics, 8338–8351. https://doi.org/10.18653/v1/2021.emnlp-main.657 

- [15] William L. Hamilton, Payal Bajaj, Marinka Zitnik, Dan Jurafsky, and Jure Leskovec. 2018. Embedding Logical Queries on Knowledge Graphs. In _Advances in Neural Information Processing Systems 31: Annual Conference on Neural Information Processing Systems 2018, NeurIPS 2018, December 3-8, 2018, Montréal, Canada_ , Samy Bengio, Hanna M. Wallach, Hugo Larochelle, Kristen Grauman, Nicolò Cesa-Bianchi, and Roman Garnett (Eds.). 2030–2041. https://proceedings. neurips.cc/paper/2018/hash/ef50c335cca9f340bde656363ebd02fd-Abstract.html 

- [16] Zijian Huang, Meng-Fen Chiang, and Wang-Chien Lee. 2022. LinE: Logical Query Reasoning over Hierarchical Knowledge Graphs. In _KDD ’22: The 28th_ 

_ACM SIGKDD Conference on Knowledge Discovery and Data Mining, Washington, DC, USA, August 14 - 18, 2022_ , Aidong Zhang and Huzefa Rangwala (Eds.). ACM, 615–625. https://doi.org/10.1145/3534678.3539338 

- [17] Bhushan Kotnis and Alberto García-Durán. 2019. Learning Numerical Attributes in Knowledge Bases. In _1st Conference on Automated Knowledge Base Construction, AKBC 2019, Amherst, MA, USA, May 20-22, 2019_ . https://doi.org/10.24432/C5Z59Q 

- [18] Bhushan Kotnis, Carolin Lawrence, and Mathias Niepert. 2021. Answering Complex Queries in Knowledge Graphs with Bidirectional Sequence Encoders. In _Thirty-Fifth AAAI Conference on Artificial Intelligence, AAAI 2021, Thirty-Third Conference on Innovative Applications of Artificial Intelligence, IAAI 2021, The Eleventh Symposium on Educational Advances in Artificial Intelligence, EAAI 2021, Virtual Event, February 2-9, 2021_ . AAAI Press, 4968–4977. https://ojs.aaai.org/ index.php/AAAI/article/view/16630 

- [19] Yankai Lin, Zhiyuan Liu, and Maosong Sun. 2016. Knowledge Representation Learning with Entities, Attributes and Relations. In _Proceedings of the Twenty-Fifth International Joint Conference on Artificial Intelligence, IJCAI 2016, New York, NY, USA, 9-15 July 2016_ , Subbarao Kambhampati (Ed.). IJCAI/AAAI Press, 2866–2872. http://www.ijcai.org/Abstract/16/407 

- [20] Lihui Liu, Boxin Du, Heng Ji, ChengXiang Zhai, and Hanghang Tong. 2021. Neural-Answering Logical Queries on Knowledge Graphs. In _KDD ’21: The 27th ACM SIGKDD Conference on Knowledge Discovery and Data Mining, Virtual Event, Singapore, August 14-18, 2021_ , Feida Zhu, Beng Chin Ooi, and Chunyan Miao (Eds.). ACM, 1087–1097. https://doi.org/10.1145/3447548.3467375 

- [21] Xiao Liu, Shiyu Zhao, Kai Su, Yukuo Cen, Jiezhong Qiu, Mengdi Zhang, Wei Wu, Yuxiao Dong, and Jie Tang. 2022. Mask and Reason: Pre-Training Knowledge Graph Transformers for Complex Logical Queries. In _KDD ’22: The 28th ACM SIGKDD Conference on Knowledge Discovery and Data Mining, Washington, DC, USA, August 14 - 18, 2022_ , Aidong Zhang and Huzefa Rangwala (Eds.). ACM, 1120–1130. https://doi.org/10.1145/3534678.3539472 

- [22] Chau Nguyen, Tim French, Wei Liu, and Michael Stewart. 2023. CylE: Cylinder Embeddings for Multi-hop Reasoning over Knowledge Graphs. In _Proceedings of the 17th Conference of the European Chapter of the Association for Computational Linguistics, EACL 2023, Dubrovnik, Croatia, May 2-6, 2023_ , Andreas Vlachos and Isabelle Augenstein (Eds.). Association for Computational Linguistics, 1728–1743. https://aclanthology.org/2023.eacl-main.127 

- [23] Hongyu Ren, Hanjun Dai, Bo Dai, Xinyun Chen, Denny Zhou, Jure Leskovec, and Dale Schuurmans. 2022. SMORE: Knowledge Graph Completion and Multi-hop Reasoning in Massive Knowledge Graphs. In _KDD ’22: The 28th ACM SIGKDD Conference on Knowledge Discovery and Data Mining, Washington, DC, USA, August 14 - 18, 2022_ , Aidong Zhang and Huzefa Rangwala (Eds.). ACM, 1472– 1482. https://doi.org/10.1145/3534678.3539405 

- [24] Hongyu Ren, Weihua Hu, and Jure Leskovec. 2020. Query2box: Reasoning over Knowledge Graphs in Vector Space Using Box Embeddings. In _8th International Conference on Learning Representations, ICLR 2020, Addis Ababa, Ethiopia, April 26-30, 2020_ . OpenReview.net. https://openreview.net/forum?id=BJgr4kSFDS 

- [25] Hongyu Ren and Jure Leskovec. 2020. Beta Embeddings for Multi-Hop Logical Reasoning in Knowledge Graphs. In _Advances in Neural Information Processing Systems 33: Annual Conference on Neural Information Processing Systems 2020, NeurIPS 2020, December 6-12, 2020, virtual_ , Hugo Larochelle, Marc’Aurelio Ranzato, Raia Hadsell, Maria-Florina Balcan, and Hsuan-Tien Lin (Eds.). https://proceedings.neurips.cc/paper/2020/hash/ e43739bba7cdb577e9e3e4e42447f5a5-Abstract.html 

- [26] Fabian M. Suchanek, Gjergji Kasneci, and Gerhard Weikum. 2007. Yago: a core of semantic knowledge. In _Proceedings of the 16th International Conference on World Wide Web, WWW 2007, Banff, Alberta, Canada, May 8-12, 2007_ , Carey L. Williamson, Mary Ellen Zurko, Peter F. Patel-Schneider, and Prashant J. Shenoy (Eds.). ACM, 697–706. https://doi.org/10.1145/1242572.1242667 

- [27] Haitian Sun, Andrew O. Arnold, Tania Bedrax-Weiss, Fernando Pereira, and William W. Cohen. 2020. Faithful Embeddings for Knowledge Base Queries. In _Advances in Neural Information Processing Systems 33: Annual Conference on Neural Information Processing Systems 2020, NeurIPS 2020, December 6-12, 2020, virtual_ , Hugo Larochelle, Marc’Aurelio Ranzato, Raia Hadsell, Maria-Florina Balcan, and Hsuan-Tien Lin (Eds.). https://proceedings.neurips.cc/paper/2020/ hash/fe74074593f21197b7b7be3c08678616-Abstract.html 

- [28] Dhanasekar Sundararaman, Shijing Si, Vivek Subramanian, Guoyin Wang, Devamanyu Hazarika, and Lawrence Carin. 2020. Methods for Numeracy-Preserving Word Embeddings. In _Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing (EMNLP)_ . Association for Computational Linguistics, Online, 4742–4753. https://doi.org/10.18653/v1/2020.emnlp-main.384 

- [29] Ilya O. Tolstikhin, Neil Houlsby, Alexander Kolesnikov, Lucas Beyer, Xiaohua Zhai, Thomas Unterthiner, Jessica Yung, Andreas Steiner, Daniel Keysers, Jakob Uszkoreit, Mario Lucic, and Alexey Dosovitskiy. 2021. MLP-Mixer: An all-MLP Architecture for Vision. In _Advances in Neural Information Processing Systems 34: Annual Conference on Neural Information Processing Systems 2021, NeurIPS 2021, December 6-14, 2021, virtual_ , Marc’Aurelio Ranzato, Alina Beygelzimer, Yann N. Dauphin, Percy Liang, and Jennifer Wortman Vaughan (Eds.). 24261–24272. https://proceedings.neurips.cc/paper/2021/hash/ cba0a4ee5ccd02fda0fe3f9a3e7b89fe-Abstract.html 

KDD ’23, August 6–10, 2023, Long Beach, CA, USA 

Knowledge Graph Reasoning over Entities and Numerical Values 

- [30] Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Lukasz Kaiser, and Illia Polosukhin. 2017. Attention is All you Need. In _Advances in Neural Information Processing Systems 30: Annual Conference on Neural Information Processing Systems 2017, December 4- 9, 2017, Long Beach, CA, USA_ , Isabelle Guyon, Ulrike von Luxburg, Samy Bengio, Hanna M. Wallach, Rob Fergus, S. V. N. Vishwanathan, and Roman Garnett (Eds.). 5998–6008. https://proceedings.neurips.cc/paper/2017/hash/ 3f5ee243547dee91fbd053c1c4a845aa-Abstract.html 

- [31] Po-Wei Wang, Daria Stepanova, Csaba Domokos, and J. Zico Kolter. 2020. Differentiable learning of numerical rules in knowledge graphs. In _8th International Conference on Learning Representations, ICLR 2020, Addis Ababa, Ethiopia, April 26-30, 2020_ . OpenReview.net. https://openreview.net/forum?id=rJleKgrKwS 

- [32] Zihao Wang, Weizhi Fei, Hang Yin, Yangqiu Song, Ginny Y. Wong, and Simon See. 2023. Wasserstein-Fisher-Rao Embedding: Logical Query Embeddings with Local Comparison and Global Transport. _CoRR_ abs/2305.04034 (2023). https: //doi.org/10.48550/arXiv.2305.04034 arXiv:2305.04034 

- [33] Zihao Wang, Yangqiu Song, Ginny Wong, and Simon See. 2023. Logical Message Passing Networks with One-hop Inference on Atomic Formulas. In _The Eleventh International Conference on Learning Representations_ . 

- [34] Zihao Wang, Hang Yin, and Yangqiu Song. 2021. Benchmarking the Combinatorial Generalizability of Complex Query Answering on Knowledge Graphs. In _NeurIPS Datasets and Benchmarks Track_ . https://openreview.net/forum?id=pX4x8f6Km5T 

- [35] Zhaohan Xi, Ren Pang, Changjiang Li, Tianyu Du, Shouling Ji, Fenglong Ma, and Ting Wang. 2022. Reasoning over Multi-view Knowledge Graphs. _CoRR_ abs/2209.13702 (2022). https://doi.org/10.48550/arXiv.2209.13702 arXiv:2209.13702 

- [36] Zezhong Xu, Wen Zhang, Peng Ye, Hui Chen, and Huajun Chen. 2022. Neural-Symbolic Entangled Framework for Complex Query Answering. In _NeurIPS_ . http://papers.nips.cc/paper_files/paper/2022/hash/ 

   - 0bcfb525c8f8f07ae10a93d0b2a40e00-Abstract-Conference.html 

- [37] Dong Yang, Peijun Qing, Yang Li, Haonan Lu, and Xiaodong Lin. 2022. GammaE: Gamma Embeddings for Logical Queries on Knowledge Graphs. In _Proceedings of the 2022 Conference on Empirical Methods in Natural Language Processing, EMNLP 2022, Abu Dhabi, United Arab Emirates, December 7-11, 2022_ , Yoav Goldberg, Zornitsa Kozareva, and Yue Zhang (Eds.). Association for Computational Linguistics, 745–760. https://aclanthology.org/2022.emnlp-main.47 

- [38] Manzil Zaheer, Satwik Kottur, Siamak Ravanbakhsh, Barnabás Póczos, Ruslan Salakhutdinov, and Alexander J. Smola. 2017. Deep Sets. In _Advances in Neural Information Processing Systems 30: Annual Conference on Neural Information Processing Systems 2017, December 4-9, 2017, Long Beach, CA, USA_ , Isabelle Guyon, Ulrike von Luxburg, Samy Bengio, Hanna M. Wallach, Rob Fergus, S. V. N. Vishwanathan, and Roman Garnett (Eds.). 3391–3401. https://proceedings.neurips. cc/paper/2017/hash/f22e4747da1aa27e363d86d40ff442fe-Abstract.html 

- [39] Zhanqiu Zhang, Jie Wang, Jiajun Chen, Shuiwang Ji, and Feng Wu. 2021. ConE: Cone Embeddings for Multi-Hop Reasoning over Knowledge Graphs. In _Advances in Neural Information Processing Systems 34: Annual Conference on Neural Information Processing Systems 2021, NeurIPS 2021, December 6-14, 2021, virtual_ , Marc’Aurelio Ranzato, Alina Beygelzimer, Yann N. Dauphin, Percy Liang, and Jennifer Wortman Vaughan (Eds.). 19172–19183. https://proceedings.neurips.cc/ paper/2021/hash/a0160709701140704575d499c997b6ca-Abstract.html 

- [40] Zhaocheng Zhu, Mikhail Galkin, Zuobai Zhang, and Jian Tang. 2022. NeuralSymbolic Models for Logical Queries on Knowledge Graphs. In _International Conference on Machine Learning, ICML 2022, 17-23 July 2022, Baltimore, Maryland, USA (Proceedings of Machine Learning Research, Vol. 162)_ , Kamalika Chaudhuri, Stefanie Jegelka, Le Song, Csaba Szepesvári, Gang Niu, and Sivan Sabato (Eds.). PMLR, 27454–27478. https://proceedings.mlr.press/v162/zhu22c.html 

KDD ’23, August 6–10, 2023, Long Beach, CA, USA 

Jiaxin Bai, Chen Luo, Zheng Li, Qingyu Yin, Bing Yin, and Yangqiu Song 

**Algorithm 1** Ground General Type **Require:** _𝐺_ is a knowledge graph. **function** GroundGeneralType( _𝑇, 𝑣_ ) _𝑇_ is an arbitrary node of the computation graph. _𝑣_ is an arbitrary knowledge graph vertex **if** _𝑇.𝑜𝑝𝑒𝑟𝑎𝑡𝑖𝑜𝑛_ = _𝑝_ **then** _𝑢_ ← Sample({ _𝑢_ |( _𝑢, 𝑣_ )is an edge in _𝐺_ }) _𝑅𝑒𝑙𝑇𝑦𝑝𝑒_ ← type of ( _𝑢, 𝑣_ ) in _𝐺_ **if** _𝑣.𝑡𝑦𝑝𝑒_ = _𝐸𝑛𝑡𝑖𝑡𝑦_ & _𝑢.𝑡𝑦𝑝𝑒_ = _𝐸𝑛𝑡𝑖𝑡𝑦_ **then** _𝑃𝑟𝑜𝑗𝑒𝑐𝑡𝑖𝑜𝑛𝑇𝑦𝑝𝑒_ ← _𝑟𝑝_ **else if** _𝑣.𝑡𝑦𝑝𝑒_ = _𝐸𝑛𝑡𝑖𝑡𝑦_ & _𝑢.𝑡𝑦𝑝𝑒_ = _𝑉𝑎𝑙𝑢𝑒𝑠_ **then** _𝑃𝑟𝑜𝑗𝑒𝑐𝑡𝑖𝑜𝑛𝑇𝑦𝑝𝑒_ ← _𝑟𝑎𝑝_ **else if** _𝑣.𝑡𝑦𝑝𝑒_ = _𝑉𝑎𝑙𝑢𝑒𝑠_ & _𝑢.𝑡𝑦𝑝𝑒_ = _𝐸𝑛𝑡𝑖𝑡𝑦_ **then** _𝑃𝑟𝑜𝑗𝑒𝑐𝑡𝑖𝑜𝑛𝑇𝑦𝑝𝑒_ ← _𝑎𝑝_ **else** _𝑃𝑟𝑜𝑗𝑒𝑐𝑡𝑖𝑜𝑛𝑇𝑦𝑝𝑒_ ← _𝑛𝑝_ **end if** _𝑆𝑢𝑏𝑄𝑢𝑒𝑟𝑦_ ← GroundGeneralType( _𝑇.𝑐ℎ𝑖𝑙𝑑,𝑢_ ) **return** ( _𝑃𝑟𝑜𝑗𝑒𝑐𝑡𝑖𝑜𝑛𝑇𝑦𝑝𝑒, 𝑅𝑒𝑙𝑇𝑦𝑝𝑒,𝑆𝑢𝑏𝑄𝑢𝑒𝑟𝑦_ ) **else if** _𝑇.𝑜𝑝𝑒𝑟𝑎𝑡𝑖𝑜𝑛_ = _𝑖_ **then** _𝐼𝑛𝑡𝑒𝑟𝑠𝑒𝑐𝑡𝑖𝑜𝑛𝑅𝑒𝑠𝑢𝑙𝑡_ = ( _𝑖_ ) **for** _𝑐ℎ𝑖𝑙𝑑_ ∈ _𝑇.𝐶ℎ𝑖𝑙𝑑𝑟𝑒𝑛_ **do** _𝑆𝑢𝑏𝑄𝑢𝑒𝑟𝑦_ ← GroundGeneralType( _𝑇.𝑐ℎ𝑖𝑙𝑑, 𝑣_ ) _𝐼𝑛𝑡𝑒𝑟𝑠𝑒𝑐𝑡𝑖𝑜𝑛𝑅𝑒𝑠𝑢𝑙𝑡._ pushback( _𝑐ℎ𝑖𝑙𝑑, 𝑣_ ) **end for return** _𝐼𝑛𝑡𝑒𝑟𝑠𝑒𝑐𝑡𝑖𝑜𝑛𝑅𝑒𝑠𝑢𝑙𝑡_ **else if** _𝑇.𝑜𝑝𝑒𝑟𝑎𝑡𝑖𝑜𝑛_ = _𝑢_ **then** _𝑈𝑛𝑖𝑜𝑛𝑅𝑒𝑠𝑢𝑙𝑡_ = ( _𝑢_ ) **for** _𝑐ℎ𝑖𝑙𝑑_ ∈ _𝑇.𝐶ℎ𝑖𝑙𝑑𝑟𝑒𝑛_ **do if** _𝑈𝑛𝑖𝑜𝑛𝑅𝑒𝑠𝑢𝑙𝑡.𝑙𝑒𝑛𝑔𝑡ℎ >_ 2 **then** _𝑣_ ← Sample( _𝐺_ ) **end if** _𝑆𝑢𝑏𝑄𝑢𝑒𝑟𝑦_ ← GroundGeneralType( _𝑇.𝑐ℎ𝑖𝑙𝑑, 𝑣_ ) _𝑈𝑛𝑖𝑜𝑛𝑅𝑒𝑠𝑢𝑙𝑡._ pushback( _𝑐ℎ𝑖𝑙𝑑, 𝑣_ ) **end for return** _𝑈𝑛𝑖𝑜𝑛𝑅𝑒𝑠𝑢𝑙𝑡_ **else if** _𝑇.𝑜𝑝𝑒𝑟𝑎𝑡𝑖𝑜𝑛_ = _𝑒_ **then if** _𝑇.𝑡𝑦𝑝𝑒_ = _𝐸𝑛𝑡𝑖𝑡𝑦_ **then return** ( _𝑒,𝑇.𝑣𝑎𝑙𝑢𝑒_ ) **else return** ( _𝑛𝑣,𝑇.𝑣𝑎𝑙𝑢𝑒_ ) **end if end if end function** 

## **A SAMPLING ALGORITHM** 

In this section, we introduce the algorithm used for sampling the numerical complex queries from a given knowledge graph. The detailed algorithm is described in Algorithm 1. For a given knowledge _𝐺_ and a general query type _𝑡_ , we start with a random node _𝑣_ to reversely find a query that has answer _𝑣_ with the corresponding structure _𝑡_ . Basically, this process is conducted in a recursion process. In this recursion, we first look at the last operation in this query. If the operation is projection, we randomly select one of its predecessors _𝑢_ that holds the corresponding relation to _𝑣_ as the answer of its sub-query. Then we call the recursion on node _𝑢_ and the sub-query type of _𝑡_ again. Similarly, for intersection and union, we will apply recursion on their sub-queries on the same node _𝑣_ . Differently for union, only one of the sub-queries is keeping _𝑢_ , while the others will use a randomly selected node, due to the nature of the union operation. The recursion will stop when there are no more operations for the current node. The specific query types will be determined according to the relation types during the process of sampling projection operation. 

