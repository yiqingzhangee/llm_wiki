Published as a conference paper at ICLR 2024 

## UNDERSTANDING EXPRESSIVITY OF GNN IN RULE LEARNING 

**Haiquan Qiu**[1] **, Yongqi Zhang**[2] **, Yong Li**[1] **, Quanming Yao**[1] _[∗]_ 

1Department of Electronic Engineering, Tsinghua University 

2The Hong Kong University of Science and Technology (Guangzhou) qyaoaa@tsinghua.edu.cn 

## ABSTRACT 

Rule learning is critical to improving knowledge graph (KG) reasoning due to their ability to provide logical and interpretable explanations. Recently, Graph Neural Networks (GNNs) with tail entity scoring achieve the state-of-the-art performance on KG reasoning. However, the theoretical understandings for these GNNs are either lacking or focusing on single-relational graphs, leaving what the kind of rules these GNNs can learn an open problem. We propose to fill the above gap in this paper. Specifically, GNNs with tail entity scoring are unified into a common framework. Then, we analyze their expressivity by formally describing the rule structures they can learn and theoretically demonstrating their superiority. These results further inspire us to propose a novel labeling strategy to learn more rules in KG reasoning. Experimental results are consistent with our theoretical findings and verify the effectiveness of our proposed method. The code is publicly available at https: //github.com/LARS-research/Rule-learning-expressivity. 

## 1 INTRODUCTION 

A knowledge graph (KG) (Battaglia et al., 2018; Ji et al., 2021) is a type of graph where edges represent multiple types of relationships between entities. These relationships can be of different types, such as friend, spouse, coworker, or parent-child, and each type of relationship is represented by a separate edge. By encapsulating the interactions among entities, KGs provide a way for machines to understand and process complex information. KG reasoning refers to the task of deducing new facts from the existing facts in KG. This task is important because it helps in many real-world applications, such as recommendation systems (Cao et al., 2019) and drug discovery (Mohamed et al., 2019). 

With the success of graph neural networks (GNNs) in modeling graph-structured data, GNNs have been developed for KG reasoning in recent years. Classical methods such as R-GCN (Schlichtkrull et al., 2018) and CompGCN (Vashishth et al., 2020) are proposed for KG reasoning by aggregating the representations of two end entities of a triplet. And they are known to fail to distinguish the structural role of different neighbors. GraIL (Teru et al., 2020) and RED-GNN (Zhang & Yao, 2022) tackle this problem by encoding the subgraph around the target triplet. GraIL predicts a new triplet using the subgraph representations, while RED-GNN employs dynamic programming for efficient subgraph encoding. Motivated by the effectiveness of heuristic metrics over paths between a link, NBFNet (Zhu et al., 2021) proposes a neural network based on Bellman-Ford algorithm for KG reasoning. AdaProp (Zhang et al., 2023) and A _[⋆]_ Net (Zhu et al., 2022) enhance the scalability of RED-GNN and NBFNet respectively by selecting crucial nodes and edges iteratively. Among these methods, NBFNet, RED-GNN and their variants score a triplet with its tail entity representation and achieve state-of-the-art (SOTA) performance on KG reasoning. However, these methods are motivated by different heuristics, e.g., Bellman-Ford algorithm and enclosing subgraph encoding, which make the understanding of their effectiveness for KG reasoning difficult. 

In this paper, inspired by the importance of rule learning in KG reasoning, we propose to study expressivity of SOTA GNNs for KG reasoning by analyzing the kind of rules they can learn. First, we unify SOTA GNNs for KG reasoning into a common framework called QL-GNN, based on the 

> *Quanming Yao is the corresponding author. 

1 

Published as a conference paper at ICLR 2024 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2024_Understanding_Expressivity_of_GNN_in_Rule_Learning_images/iclr_2024_Understanding_Expressivity_of_GNN_in_Rule_Learning.pdf-0002-01.png)


**----- Start of picture text -----**<br>
Target Relations father EL-GNN<br>F<br>Rule Structure Description<br>friend teacher A spouse B parent C R [′] (h , x ) in CML[ G,  h ,  c1 ,  c2 , · · · ,  c k ]<br>father<br>A C Exemplar Methods<br>sisterhood EL-NBFNet, EL-RED-GNN<br>spouse parent brother D<br>QL-GNN<br>B E Rule Structure Description<br>Target Relations aunt R (h , x ) in CML[ G, h ]<br>sisterhood aunt<br>Exemplar Methods<br>D sisterhood parent brother NBFNet, RED-GNN, A*Net, AdaProp<br>D B C E<br>Knowledge Graph Rule structure Expressivity<br>**----- End of picture text -----**<br>


Figure 1: The existence of a triplet in KG is determined by the corresponding rule structure. We investigates the kind of rule structures can be learned by SOTA GNNs for KG reasoning (i.e., QLGNN), and proposes EL-GNN, which can learn more rule structures compared to QL-GNN. 

observation that they score a triplet with its tail entity representation and essentially extract rule structures from subgraphs with same pattern. Then, we analyze the logical expressivity of QL-GNN to study its ability of learning rule structures. The analysis helps us reveal the underneath theoretical reasons that contribute to the empirical success of QL-GNN, elucidating their effectiveness over classical methods. Specifically, our analysis is based on the formal description of rule structures in graph, which differs from previous analysis that relies on graph isomorphism testing (Xu et al., 2019; Zhang et al., 2021) and focuses on the expressivity of distinguishing various rules. The new analysis tool allows us to understand the rules learned by QL-GNN and reveals the maximum expressivity that QL-GNN can generalize through training. Based on the new theory, we also uncover the deficiencies of QL-GNN in learning rule structures and we propose EL-GNN based on labeling trick as an improvement upon QL-GNN to improve its learning ability. In summary, our paper has the following contributions: 

- Our work unifies state-of-the-art GNNs for KG reasoning into a common framework named QL-GNN, and analyzes their logical expressivity to study their ability of learning rule structures, explaining their superior performance over classical methods. 

- The logical expressivity of QL-GNN demonstrates its capability in learning a particular class of rule structures. Consequently, based on further theoretical analysis, we introduce EL-GNN, a novel GNN designed to learn rule structures that are beyond the learning capacity of QL-GNN. 

- Synthetic datasets are generated to evaluate the expressivity of various GNNs, whose experimental results are consistent with our theory. Also, results of the proposed labeling method show improved performance on real datasets. 

- 2 A COMMON FRAMEWORK FOR THE STATE-OF-THE-ART METHODS 

To study the state-of-the-art GNNs for KG reasoning, we find that they (e.g., RED-GNN and NBFNet) essentially learn rule structures from GNN’s tail entity representation which encodes subgraphs with the same pattern, i.e., a subgraph with the query entity as the source node and the tail entity as the sink node. Based on this observation, we are motivated to derive a common framework for these SOTA methods and analyze their ability of learning rule structures with the derived framework. 

Given a query ( _h, R,_ ?), the labeling trick of query entity _h_ ensures the SOTA methods to extract rules from a graph with the same pattern because it makes the query entity distinguishable among all entities in graph. Therefore, we unify NBFNet, RED-GNN and their variants to a common framework called Query Labeling (QL) GNN (see correspondence in Appdendix B). For a query ( _h, R,_ ?), QL-GNN first applies labeling trick by assigning special initial representation **e**[(0)] _h_ to entity _h_ , which make the query entity distinguishable from other entities. Base on these initial features, QL-GNN aggregates 

2 

Published as a conference paper at ICLR 2024 

entity representations with a _L_ -layer message passing neural network (MPNN) for each candidate _t ∈V_ . MPNN’s last layer representation of entity _t_ in QL-GNN is denoted as **e**[(] _t[L]_[)] [ _h_ ] indicating its dependency on query entity _h_ . Finally, QL-GNN scores new facts ( _h, R, t_ ) with tail entity representation **e**[(] _t[L]_[)] [ _h_ ]. For example, NBFNet uses the score function _s_ ( _h, R, t_ ) = FFN( **e**[(] _t[L]_[)] [ _h_ ]) for new triplet ( _h, R, t_ ) where FFN( _·_ ) denotes a feed-forward neural network. 

Even RED-GNN, NBFNet and their variant may take the different MPNNs to calculate **e**[(] _t[L]_[)] [ _h_ ], without loss of generality, their MPNNs can take the following form in QL-GNN (omit [ _h_ ] for simplicity): 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2024_Understanding_Expressivity_of_GNN_in_Rule_Learning_images/iclr_2024_Understanding_Expressivity_of_GNN_in_Rule_Learning.pdf-0003-03.png)


where _δ_ and _ϕ_ are combination and aggregation functions respectively, _ψ_ is the message function encoding the relation _R_ and entity _u_ neighboring to _v_ , _{{· · · }}_ is a multiset, and _NR_ ( _v_ ) is the neighboring entity set _{u|_ ( _u, R, v_ ) _∈E}_ . 

## 3 EXPRESSIVITY OF QL-GNN 

In this section, we explore the logical expressivity of QL-GNN to analyze the types of rule structures QL-GNN can learn. First, we provide the logic to describe rules in KGs. Then, we analyze logical expressivity of QL-GNN using Theorem 3.2 and Corollary 3.3, formally demonstrating the kind of rule structures it can learn. Finally, we compare QL-GNN with classical methods and highlight its superior expressivity in KG reasoning. 

## 3.1 EXPRESSIVITY ANALYSIS WITH LOGIC OF RULE STRUCTURES 

From previous works of rule mining on KG (Yang et al., 2017; Sadeghian et al., 2019), rule structures are usually described as a formula in first-order logic. We also follow this way to formally describe the rule structures in KG. Therefore, we have the following correspondence between the elements in rule structures and logic: 

- Variable: variables denoted with lowercase italic letters _x, y, z_ represent entities in a KG; 

- Unary predicate: unary predicate _Pi_ ( _x_ ) is corresponding to the entity property _Pi_ in a KG, e.g., red( _x_ ) denotes the color of an entity _x_ is red; 

- Binary predicate: binary predicate _Rj_ ( _x, y_ ) is corresponding to the relation _Rj_ in a KG, e.g., father( _x, y_ ) denotes _x_ is the father of _y_ ; 

- Constant: constant denoted with lowercase letters h _,_ c with serif typestyle is the unique identifier of some entity in a KG. 

Except from the above elements, the quantifier _∃_ expresses the existence of entities satisfying a condition, _∀_ expresses universal quantification, and _∃[≥][N]_ represents the existence of at least _N_ entities satisfying a condition. The logical connective _∧_ denotes conjunction, _∨_ denotes disjunction, and _⊤_ and _⊥_ represent true and false, respectively. Using these symbols, rule structures can be represented by describing their elements directly. For example, _C_ 3( _x, y_ ) := _∃z_ 1 _z_ 2 _, R_ 1( _x, z_ 1) _∧ R_ 2( _z_ 1 _, z_ 2) _∧ R_ 3( _z_ 2 _, y_ ) in Figure 2 describes a chain-like structure between _x_ and _y_ with three relations _R_ 1 _, R_ 2 _, R_ 3. Rule structures can be represented using the rule formula _R_ ( _x, y_ ), and the existence of a rule structure for the triplet ( _h, R, t_ ) is equivalent to the satisfaction of the rule formula _R_ ( _x, y_ ) at the entity pair ( _h, t_ ). In this paper, **logical expressivity** of GNN is a measurement of the ability of GNN to learn logical formulas and is defined as the set of logical formulas that GNN can learn. Therefore, since rule structures can be described by logical formulas, the logical expressivity of QL-GNN can determine their ability to learn rule structures in KG reasoning. 

## 3.2 WHAT KIND OF RULE STRUCTURES CAN QL-GNN LEARN? 

In this section, we analyze the logical expressivity of QL-GNN regarding what kind of rule structure it can learn. Given a query ( _h, R,_ ?), we first have the following proposition about the rule formula describing a rule structure. 

3 

Published as a conference paper at ICLR 2024 

**Proposition 3.1.** _The rule structure for query_ ( _h, R,_ ?) _can be described with rule formula R_ ( _x, y_ ) _or rule formula R_ (h _, x_ )[1] _where_ h _is the logical constant assigned to query entity h._ 

QL-GNN applies labeling trick to the query entity _h_ , which can be equivalently seen as assigning constant h to query entity _h_[2] .With Proposition 3.1 (proven in Appendix A), the logical expressivity of QL-GNN can be analyzed by the types of rule formula _R_ (h _, x_ ) it can learn. In this case, the rule structure of triplet ( _h, R, t_ ) exists if and only if the logical formula _R_ (h _, x_ ) is satisfied at entity _t_ . 

## 3.2.1 EXPRESSIVITY OF QL-GNN 

Before presenting the logical expressivity of QL-GNN, we start by explaining how QL-GNN learns the rule formula _R_ (h _, x_ ). Following the definition in Barceló et al. (2020), we treat _R_ (h _, x_ ) as a binary classifier. When given a candidate tail entity _t_ , if the triplet ( _h, R, t_ ) exists in a KG, the binary classifier _R_ (h _, x_ ) should output true; otherwise, it should output false. If QL-GNN can learn the rule formula _R_ (h _, x_ ), it implies that QL-GNN can estimate binary classifier _R_ (h _, x_ ). Consequently, if the rule formula _R_ (h _, x_ ) is satisfied at entity _t_ , the representation **e**[(] _t[L]_[)] [ _h_ ] is mapped to a high probability value, indicating the existence of triplet ( _h, R, t_ ) in KG. Conversely, when the rule formula is not satisfied at _t_ , **e**[(] _t[L]_[)] [ _h_ ] is mapped to a low probability value, indicating the absence of the triplet. 

The rule structures that QL-GNN can learn are described by a family of logic called graded modal logic (CML) (De Rijke, 2000; Otto, 2019). CML is defined by recursion with the base elements _⊤, ⊥_ , all unary predicates _Pi_ ( _x_ ), and the recursion rule: if _φ_ ( _x_ ) _, φ_ 1( _x_ ) _, φ_ 2( _x_ ) are formulas in CML, _¬φ_ ( _x_ ) _, φ_ 1( _x_ ) _∧ φ_ 2( _x_ ) _, ∃[≥][N] y_ ( _R_ ( _y, x_ ) _∧ φ_ ( _y_ )) are also formulas in CML. Since QL-GNN introduces a constant h to the query entity _h_ , we use the notation CML[ _G,_ h] to denote the CML recursively built from base elements in _G_ and constant h (equivalent to constant predicate _Ph_ ( _x_ )). Then, the following theorem and corollary show the expressivity of QL-GNN for KG reasoning. 

**Theorem 3.2** (Logical expressivity of QL-GNN) **.** _For KG reasoning, given a query_ ( _h, R,_ ?) _, a rule formula R_ (h _, x_ ) _is learned by QL-GNN if and only if R_ (h _, x_ ) _is a formula in CML_ [ _G,_ h] _._ 

**Corollary 3.3.** _The rule structures learned by QL-GNN can be constructed with the recursion:_ 

- _**Base case:** all unary predicates Pi_ ( _x_ ) _can be learned by QL-GNN; the constant predicate Ph_ ( _x_ ) _can be learned by QL-GNN;_ 

- _**Recursion rule:** if the rule structures R_ 1(h _, x_ ) _, R_ 2(h _, x_ ) _, R_ (h _, y_ ) _are learned by QL-GNN, R_ 1(h _, x_ ) _∧ R_ 2(h _, y_ ) _, ∃[≥][N] y_ ( _Ri_ ( _y, x_ ) _∧ R_ (h _, y_ )) _are learned by QL-GNN._ 

Theorem 3.2 (proved in Appendix C) provides the logical expressivity of QL-GNN with rule formula _R_ (h _, x_ ) in CML[ _G,_ h], which shows that querying labeling transforms _R_ ( _x, y_ ) to _R_ (h _, x_ ) and enable QL-GNN to learn the corresponding rule structure. To gain a concrete understanding of the rule structures learned by QL-GNN, Corollary 3.3 provides the recursive definition for these rule structures. Note that Theorem 3.2 cannot be directly applied to analyze the expressivity of QL-GNN when learning more than one rule structures. The ability of learning more than one rule structures relates to the capacity of QL-GNN, which we take as a future direction. Theorem 3.2 also reveals the maximum expressivity that QL-GNN can generalize through training, and its proof also provides some insights about the design QL-GNN with better generalization (more discussions are provided in Appendix F.1). Besides, our results in this section can be reduced to single relational-graph by restricting the relation type to a single relation type, and we give these results as corollaries in Appendix E. 

## 3.2.2 EXAMPLES 

We analyze several rule structures and their corresponding rule formulas in Figure 2 as illustrative examples, demonstrating the application of our theory in analyzing the rule structures that QL-GNN can learn. The real examples of these rule structures are shown in Figure 1. In Appdendix A, we have detailed analysis of rule structures discussed in the paper and present some rules from real datasets. 

Chain-like rules, e.g., _C_ 3( _x, y_ ) in Figure 2, are basic rule structures investigated in many previous works (Sadeghian et al., 2019; Teru et al., 2020; Zhu et al., 2021). QL-GNN assigns constant h to 

> 1The rule formula _R_ (h _, x_ ) is equivalent to _∃zR_ ( _z, x_ ) _∧ Ph_ ( _z_ ) where _Ph_ ( _x_ ) denotes the assignment of constant h to _x_ and is called constant predicate in our paper. 

> 2The initial representation of an entity should be unique among all entities to be regarded as constant in logic. The initial representation assigned to query entity are indeed unique in NBFNet, RED-GNN and their variants. 

4 

Published as a conference paper at ICLR 2024 

query entity _h_ , thus triplets with relation _C_ 3 can be predicted by learning the rule formula _C_ 3(h _, x_ ). _C_ 3(h _, x_ ) are formulas in CML[ _G,_ h] and can be recursively defined with rules in Corollary 3.3 (proven in Corollary A.2). Therefore, our theory gives a general proof of QL-GNN’s ability to learn chain-like structures. 

The second type of rule structure _I_ 1(h _, x_ ) in Figure 2 is composed of a chain-like structure from query entity to tail entity along with additional entity _z_ 2 connected to the chain. _I_ 1(h _, x_ ) are formulas in CML[ _G,_ h] and can be defined with recursive rules in Corollary 3.3 (proven in Corollary A.3), which indicates that _I_ 1(h _, x_ ) can be learned by QL-GNN. These structures are important in KG reasoning because the entity connected to the 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2024_Understanding_Expressivity_of_GNN_in_Rule_Learning_images/iclr_2024_Understanding_Expressivity_of_GNN_in_Rule_Learning.pdf-0005-03.png)


**----- Start of picture text -----**<br>
C 3( x, y ) := ∃z 1 z 2 , R 1( x, z 1)  ∧ R 2( z 1 , z 2)<br>𝑅!   𝑧! 𝑅"   𝑧" 𝑅# C 3(h , x ) := ∃∧z 1 Rz 23 , R ( z 21 , y (h) , z 1)  ∧ R 2( z 1 , z 2)<br>∧ R 3( z 2 , x )<br>  𝑧" query entity<br>𝑅# tail entity I 1(h , x ) := ∃z 1 , z 2 , R 1(h , z 1)  ∧ R 3( z 2 , z 1)<br>𝑅!   𝑧! 𝑅" ∧ R 2( z 1 , x )<br>Rule structure Rule formula<br>**----- End of picture text -----**<br>


Figure 2: Example of rule structures and their corresponding rule formulas QL-GNN can learn. 

chain can bring extra information about property of the entity it connected to (see examples of rule in Appendix A). 

## 3.3 COMPARISON WITH CLASSICAL METHODS 

Classical methods such as R-GCN and CompGCN perform KG reasoning by first applying MPNN (1) to compute the entity representations **e**[(] _v[L]_[)] _, v ∈V_ and then scoring the triplet ( _h, R, t_ ) by _s_ ( _h, R, t_ ) = Agg( **e**[(] _h[L]_[)] _[,]_ **[ e]** _t_[(] _[L]_[)] ) with aggregation function Agg( _·, ·_ ). For simplicity, we take CompGCN as an example to analyze the expressivity of the classical methods on learning rule structures. 

Since CompGCN scores a triplet using its query and tail entity representations without applying labeling trick, the rule structures learned by CompGCN should be in the form of _R_ ( _x, y_ ). In CompGCN, the query and tail entities’ representations encode different subgraphs. However, the joint subgraph they represent may not necessarily be connected. This suggests that the rule structures learned by CompGCN are non-structural, indicating there is no path between its query and tail entities except for relation _R_ . This observation is proven with the following theorem. 

**Theorem 3.4** (Logical expressivity of CompGCN) **.** _For KG reasoning, CompGCN can learn the rule formula R_ ( _x, y_ ) = _fR_ ( _{φ_ ( _x_ ) _}, {φ[′]_ ( _y_ ) _}_ ) _where fR is a formula involving sub-formulas from {φ_ ( _x_ ) _} and {φ[′]_ ( _y_ ) _} which are the sets of formulas in CML_ [ _G_ ] _._ 

_Remark._ Theorem 3.4 indicates that representations of two end entities encoding two formulas respectively, and these two formulas are independent. Thus, the rule structures learned by CompGCN should be two disconnected subgraphs surrounding the query and tail entities respectively. 

Similar to Theorem 3.2, CompGCN learns rule formula _R_ ( _x, y_ ) by treating it as a binary classifier. In a KG, the binary classifier _R_ ( _x, y_ ) should output true if the triplet ( _h, R, t_ ) exists; otherwise, it should output false. If CompGCN can learn the rule formula _R_ ( _x, y_ ), it implies that it can estimate the binary classifier _R_ ( _x, y_ ). Consequently, if the rule formula _R_ ( _x, y_ ) is (not) satisfied at entity pair ( _h, t_ ), the score _s_ ( _h, R, t_ ) is a high (low) value, indicating the existence (absence) of triplet ( _h, R, t_ ). 

Theorem 3.4 (proven in Appendix C) shows that CompGCN can only learn rule formula _R_ ( _x, y_ ) for non-structural rules. One important type of relation in this category is the similarity between two entities (experiments in Appendix D.2), like same_color( _x, y_ ) indicating entities with the same color. However, structural rules are more commonly observed in KG reasoning (Lavrac & Dzeroski, 1994; Sadeghian et al., 2019; Srinivasan & Ribeiro, 2020). Since Theorem 3.4 indicates CompGCN fails to learn connected rule structures that are not independent, the structural rules in Figure 2 cannot be learned by CompGCN. Such a comparison shows why QL-GNN is more efficient than classical methods, e.g., R-GCN and CompGCN, in real applications. Compared with previous work on single-relational graphs, Zhang et al. (2021) shows CompGCN cannot distinguish many non-isomorphic links, while our paper derives expressivity of CompGCN for learning rule structures. 

5 

Published as a conference paper at ICLR 2024 

## 4 ENTITY LABELING GNN BASED ON RULE FORMULA TRANSFORMATION 

QL-GNN is proven to be able to learn the class of rule structures defined in Corollary 3.3. For rule structures outside this class, we try to learn them with a novel labeling trick based on QL-GNN. Our general idea is to transform the rule structures outside this class into the rule structures in this class by adding constants to the graph. The following proposition and corollary show how to add constants to a rule structure so that it can be described by formulas in CML and how to apply labeling trick to make it learnable for QL-GNN. 

**Proposition 4.1.** _Let R_ (h _, x_ ) _describe a single-connected rule structure_ G _in G. If we assign constants_ c1 _,_ c2 _, · · · ,_ c _k to all k entities with out-degree larger than one in_ G _, the rule structure_ G _can be described with a new rule formula R[′]_ (h _, x_ ) _in CML_ [ _G,_ h _,_ c1 _,_ c2 _, · · · ,_ c _k_ ] _._ 

**Corollary 4.2.** _Applying labeling trick with unique initial representations to entities assigned with constants_ c1 _,_ c2 _, · · · ,_ c _k in Proposition 4.1, the rule structure_ G _can be learned by QL-GNN._ 

For instance, in Figure 3, the rule structure _U_ cannot be distinguished from the rule structure _T_ by recursive definition in Corollary 3.3, thus cannot be learned by QL-GNN. In this example, Proposition 4.1 suggests assigning constant c to the entity colored with gray in Figure 3, then a new rule formula 

_U[′]_ (h _, x_ ) := _R_ 1(h _,_ c) _∧_ � _∃z_ 2 _, z_ 3 _, R_ 2(c _, z_ 2) _∧ R_ 4( _z_ 2 _, x_ ) _∧ R_ 3(c _, z_ 3) _∧ R_ 5( _z_ 3 _, x_ )� 

in CML[ _G,_ h _,_ c] (Corollary A.5) can describe the rule structure of _U_ . Therefore, the rule structure of _U_ can be learned with _U[′]_ (h _, x_ ) by QL-GNN with constant c and cannot be learned by classical methods and vanilla QL-GNN. 

## **Algorithm 1** Entity Labeling 

**Require:** query ( _h, R,_ ?), knowledge graph _G_ , degree threshold _d_ . 

1: compute the out-degree _dv_ of each entity _v_ in _G_ ; 

2: **for** entity _v_ in _G_ **do** 3: **if** _dv > d_ **then** 4: assign a unique representation **e**[(0)] _v_ to entity _v_ ; 5: **end if** 6: **end for** 7: assign initial representation **e**[(0)] _h_ to the query entity _h_ ; 

8: **Return:** initial representation of all entities. 

> Based on Corollary 4.2, we needapply labeling trick to entities 𝑅! 𝑧! 𝑅" 𝑧" 𝑅$ _T_ (h _, x_ ) := _∃z_ 1 _z_ 2 _, R_ 1(h _, z_ 1) _∧ R_ 2( _z_ 1 _, z_ 2) _∧ R_ 4( _z_ 2 _, x_ ) otherinstructuresQL-GNNthanoutsidetheto querylearnthe scopetheentitiesruleof 𝑅! 𝑧! 𝑅# 𝑧" 𝑅% _∧∃z_ 1 _z_ 2 _, R_ 1(h _, z_ 1) _∧ R_ 3( _z_ 1 _, z_ 2) _∧ R_ 5( _z_ 2 _, x_ ) Corollary 3.3.is called Entity-LabelingThe new method(EL) query entitytail entity 𝑅" 𝑧" 𝑅$ GNN shown in Algorithm 1 and 𝑅! 𝑧! _U_ (h _, x_ ) := _∃_[1] _z_ 1 _, R_ 1(h _, z_ 1) _∧_ � _∃z_ 2 _, z_ 3 _, R_ 2( _z_ 1 _, z_ 2) is different from QL-GNN in as𝑅# 𝑧# 𝑅% _∧ R_ 4( _z_ 2 _, x_ ) _∧ R_ 3( _z_ 1 _, z_ 3) _∧ R_ 5( _z_ 3 _, x_ )� signing constants to all the enti- **Rule structure Rule formula** ties with out-degree larger than _d_ . We choose the degree threshold Figure 3: Two rule structures cannot be distinguished by QL- _d_ as a hyperparameter because a GNN. small _d_ (such as 1) will introduce too many constants to KG, which impedes the generalization of GNN (Abboud et al., 2021) (see an explanation from logical perspective in Appendix F.2). In fact, a smaller _d_ makes GNN learn the rule formulas with many constants and results bad generalization, while a larger _d_ may not be able to transform indistinguishable rules into formulas in CML. As a result, the degree threshold _d_ should be tuned to balance the expressivity and generalization of GNN. Same as the constant h in QL-GNN, we add a unique initial representation **e**[(0)] _v_ for entities _v_ whose out-degree _dv > d_ in steps 3-5. For the query entity _h_ , we assign it with a unique initial representation **e**[(0)] _h_ in step 7. In Algorithm 1, it can be seen that the additional time of EL-GNN comes from traversing all entities in the graph. The additional time complexity is linear with respect to the number 

6 

Published as a conference paper at ICLR 2024 

of entities, which is negligible compared to QL-GNN. For convenience, GNN initialized with EL algorithm is denoted as EL-GNN (e.g., EL-NBFNet) in our paper. 

**Discussion** In Figure 1, we visually compare the expressivity of QL-GNN and EL-GNN. Classical methods, e.g., R-GCN and CompGCN, are not compared here because they can solely learn nonstructural rules which are not commonly-seen in real applications. QL-GNN, e.g., NBFNet and RED-GNN, excels at learning rule structures described by formula _R_ (h _, x_ ) in CML[ _G,_ h]. The proposed EL-GNN, encompassing QL-GNN as a special case, can learn rule structures described by formula _R_ (h _, x_ ) in CML[ _G,_ h _,_ c1 _, · · · ,_ c _k_ ] which has a larger description scope than CML[ _G,_ h]. 

## 5 RELATED WORKS 

## 5.1 EXPRESSIVITY OF GRAPH NEURAL NETWORK (GNN) 

GNN (Kipf & Welling, 2016; Gilmer et al., 2017) has shown good performance on a wide range of tasks involving graph-structured data, thus many existing works try to analyze the expressivity of GNNs. Most of these works analyze the expressivity of GNNs from the perspective of graph isomorphism testing. A well-known result (Xu et al., 2019) shows that the expressivity of vanilla GNN is limited to WL test and the result is extended to KG by Barcelo et al. (2022). To improve the expressivity of GNNs, most of the existing works either design GNNs motivated by high-order WL test (Morris et al., 2019; 2020; Barcelo et al., 2022) or apply special initial representations (Abboud et al., 2021; You et al., 2021; Sato et al., 2021; Zhang et al., 2021). Except for using graph isomorphism testing, Barceló et al. (2020) analyze the logical expressivity of GNNs and identify that the logical rules from graded modal logic can be learned by vanilla GNN. However, their analysis is limited to node classification on the single-relational graph. Except from the expressivity of vanilla GNN, Tena Cucala et al. (2022) propose monotonic GNN whose prediction can be explained by symbolical rules in Datalog and the expressivity of monotonic GNN is further analyzed in Cucala et al. (2023). 

Regarding the expressivity of GNNs for link prediction, Srinivasan & Ribeiro (2020) demonstrate that GNNs’ structural node representations alone are insufficient for accurate link prediction. To overcome this limitation, they introduce a method that incorporates Monte Carlo samples of node embeddings obtained from network embedding techniques instead of relying solely on GNNs. However, Zhang et al. (2021) discovered that by leveraging the labeling trick in GNNs, it is indeed possible to learn structural link representations for effective link prediction. This finding provides reassurance regarding the viability of GNNs for this task. Nonetheless, their analysis is confined to single-relational graphs, and their conclusions are limited to the fact that the labeling trick enables distinct representations for some non-isomorphic links, which other approaches cannot achieve. In this paper, we delve into the analysis of GNNs’ logical expressivity to study their ability of learning rule structures. By doing so, we aim to gain a comprehensive understanding of the rule structures that SOTA GNNs can learn in graphs. Our analysis encompasses both single-relational graph and KGs, thus broadening the applicability of our findings. 

A concurrent work by Huang et al. (2023) analyzes the expressivity of GNNs for NBFNet (a kind of QL-GNN in our paper) with conditional MPNN while our work unifies state-ot-the-art GNNs into QL-GNN and analyzes the expressivity from a different perspective focusing on the understanding of relationship between labeling trick and constants in logic. 

## 5.2 KNOWLEDGE GRAPH REASONING 

KG reasoning is the task to predict new facts based on the known facts in a KG _G_ = ( _V, E, R_ ) where _V, E, R_ are sets of entities, edges and relation types in the graph respectively. The facts (or edges, links) are typically expressed as triplets in the form of ( _h, R, t_ ), where the head entity _h_ and tail entity _t_ are related with the relation type _R_ . KG reasoning can be modeled as the process of predicting the tail entity _t_ of a query in the form ( _h, R,_ ?) where _h_ is called the query entity in our paper. The head prediction (? _, R, t_ ) can be transformed into tail prediction ( _t, R[−]_[1] _,_ ?) with inverse relation _R[−]_[1] . Thus, we focus on tail prediction in this paper. 

Embedding-based methods like TransE (Bordes et al., 2013), ComplEx (Trouillon et al., 2016), RotatE (Sun et al., 2019), and QuatE (Zhang et al., 2019) have been developed for KG reasoning. 

7 

Published as a conference paper at ICLR 2024 

They learn embeddings for entities and relations, and predict facts by aggregating their representations. To capture local evidence within graphs, Neural LP (Yang et al., 2017) and DRUM (Sadeghian et al., 2019) learn logical rules based on predefined chain-like structures. However, apart from chain-like rules, these methods failed to learn more complex structures in KG (Hamilton et al., 2018; Ren et al., 2019). GNNs have also been used for KG reasoning, such as R-GCN (Schlichtkrull et al., 2018) and CompGCN (Vashishth et al., 2020), which aggregate entity and relation representations to calculate scores for new facts. However, these methods struggle to differentiate between the structural roles of different neighbors (Srinivasan & Ribeiro, 2020; Zhang et al., 2021). GraIL (Teru et al., 2020) addresses this by extracting enclosing subgraphs to predict new facts, while RED-GNN (Zhang & Yao, 2022) employs dynamic programming for efficient subgraph extraction and predicts new facts based on the tail entity representation. To extract relevant structures from graph, AdaProp (Zhang et al., 2023) improves RED-GNN by employing adaptive propagation to filter out irrelevant entities and retain promising targets. Motivated by the effectiveness of heuristic path-based metrics for link prediction, NBFNet (Zhu et al., 2021) proposes a neural network aligned with Bellman-Ford algorithm for KG reasoning. Zhu et al. (2022) propose A _[⋆]_ Net to learn a priority function to select important nodes and edges at each iteration. Specifically, AdaProp and A _[⋆]_ Net are variants of RED-GNN and NBFNet, respectively, designed to enhance their scalability. Among these methods, RED-GNN, NBFNet, AdaProp, and A _[⋆]_ Net achieve state-of-the-art performance on KG reasoning. 

## 6 EXPERIMENT 

In this section, we validate our theoretical findings from Section 3 and showcase the efficacy of our proposed EL-GNN (Section 4) on synthetic and real datasets through experiments. All experiments were implemented in Python using PyTorch and executed on A100 GPUs with 80GB memory. 

## 6.1 EXPERIMENTS ON SYNTHETIC DATASETS 

We generate six KGs based on rule structures in Figure 2, 3, 6 to validate our theory on expressivity and verify the improved performance of EL-GNN. These rule structures are either analyzed in the previous sections, or representative for evaluating GNN’s ability for learning rule structures. We evaluate R-GCN, CompGCN, RED-GNN, NBFNet, EL-RED-GNN, and EL-NBFNet (using RED-GNN/NBFNet as backbone with Algorithm 1). Our evaluation metric is prediction _Accuracy_ which measures how well a rule structure is learned. We report testing accuracy of classical methods, QL-GNN, and EL-GNN on six synthetic graphs. Hyperparameters for all methods are automatically tuned with Ray (Liaw et al., 2018) based on the validation accuracy. 

Table 1: Accuracy on synthetic data. 

|Method|Method|_C_3|_C_4|_I_1|_I_2|_T_|_U_|
|---|---|---|---|---|---|---|---|
|||||||||
|Classical|R-GCN<br>CompGCN|0.016<br>0.016|0.031<br>0.021|0.044<br>0.053|0.024<br>0.039|0.067<br>0.067|0.014<br>0.027|
|||||||||
|QL-GNN|RED-GNN<br>NBFNet|1.0<br>1.0|1.0<br>1.0|1.0<br>1.0|1.0<br>1.0|1.0<br>1.0|0.405<br>0.541|
|||||||||
|EL-GNN|EL-RED-GNN<br>EL-NBFNet|1.0<br>1.0|1.0<br>1.0|1.0<br>1.0|1.0<br>1.0|1.0<br>1.0|0.797<br>0.838|



**Dataset generation** Given a target relation, there are three steps to generate a dataset: (1) rule structure generation: generate specific rule structures according to their definition; (2) noisy triplets generation: generate noisy triplets to avoid GNN from learning naive rule structures; (3) missing triplets completion: generate missing triplets based on the target rule structure because the noisy triplets generation step could add triplets satisfying the target rule structure. We use triplets generated from rule structure and noisy triplets generation steps as known triplets in graph. Triplets with the target relation are separated into training, validation, and testing sets. Our experimental setting differs slightly from previous works as all GNNs in the experiments only perform message passing on the known triplets in the graph. This setup is reasonable and allows for evaluating the performance of GNNs in learning rule structures because the presence of a triplet can be determined based on the known triplets in the graph, following the rule structure generation process. 

8 

Published as a conference paper at ICLR 2024 

**Results** Table 1 presents the testing accuracy of classical GNN methods, QL-GNN, and EL-GNN on six synthetic datasets (denoted as _C_ 3 _, C_ 4 _, I_ 1 _, I_ 2 _, T,_ and _U_ ) generated from their corresponding rule structures. The experimental results support our theory. CompGCN performs poorly on all six datasets, as it fails to learn the underlying rule structures discussed in examples of Section 3 (refer to Section D.2 for experiments of CompGCN). QL-GNN achieves perfect predictions (100% accuracy) for triplets with relations _Cl, Ii,_ and _T_ , successfully learning the corresponding rule formulas from CML[ _G,_ h]. EL-GNN demonstrates improved expressivity, as evidenced by its performance on dataset _U_ , aligning with the analysis in Section 4. Furthermore, EL-GNN effectively learns rule formulas _C_ (h _, x_ ) and _I_ (h _, x_ ), validating its expressivity. 

Furthermore, we demonstrate the impact of the degree threshold _d_ on EL-GNN with dataset _U_ . The testing accuracy in Figure 4 reveals that an excessively small or large out-degree _d_ hinders the performance of ELGNN. Therefore, it is important to empirically fine-tune the hyperparameter _d_ . To test the robustness of QLGNN and EL-GNN in learning rules with incomplete structures, we randomly remove triplets in the training set to evaluate the accuracy of learning rule structures. The results can be found in Appendix D.4. 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2024_Understanding_Expressivity_of_GNN_in_Rule_Learning_images/iclr_2024_Understanding_Expressivity_of_GNN_in_Rule_Learning.pdf-0009-03.png)


**----- Start of picture text -----**<br>
0.85<br>0.80<br>0.75<br>0.70<br>0.65<br>0.60<br>0.55<br>0.50<br>0 10 20 30 40 50<br>Out-degree d<br>Accuracy<br>**----- End of picture text -----**<br>


Figure 4: Accuracy versus out-degree _d_ of EL-GNN on the dataset with relation _U_ . 

## 6.2 EXPERIMENTS ON REAL DATASETS 

In this section, we follow the standard setup as Zhu et al. (2021) to test EL-GNN’s effectiveness on four real datasets: Family (Kok & Domingos, 2007), Kinship (Hinton et al., 1986), UMLS (Kok & Domingos, 2007), WN18RR (Dettmers et al., 2017), and FB15k-237 (Toutanova & Chen, 2015). For a fair comparison, we evaluate EL-NBFNet and EL-RED-GNN (applying EL to NBFNet and RED-GNN) using the same hyperparameters as NBFNet and RED-GNN and handcrafted _d_ . We compare it with embedding-based methods (RotatE, QuatE), rule-based methods (Neural LP, DRUM), and GNN-based methods (CompGCN, NBFNet, RED-GNN). To evaluate performance, we provide testing accuracy and standard deviation obtained from three repetitions for thorough evaluation. 

In Table 2, we present our experimental findings. The results first show that NBFNet and RED-GNN (QL-GNN) outperform CompGCN. Furthermore, the proposed EL algorithm improves the accuracy of RED-GNN and NBFNet on real datasets. However, the degree of improvement varies across datasets due to the number and variations of rule types, and the quality of missing triplets in training sets. More experimental results, e.g., time cost and more performance metrics, are in Appendix D.5. 

Table 2: Accuracy and standard deviation on real datasets. The best (and comparable best) results are in “ **bold** ”, the second (and comparable second) best are underlined. 

|Method Class|Methods|Family|Kinship|UMLS|WN18RR|FB15k-237|
|---|---|---|---|---|---|---|
||||||||
|Embedding-<br>based|RotatE<br>QuatE|0.865±0.004<br>0.897±0.001|0.704±0.002<br>0.311±0.003|0.860±0.003<br>0.907±0.002|0.427±0.003<br>0.441±0.002|0.240±0.001<br>0.255±0.004|
||||||||
|Rule-based|Neural LP<br>DRUM|0.872±0.002<br>0.880±0.003|0.481±0.006<br>0.459±0.005|0.630±0.001<br>0.676±0.004|0.369±0.003<br>0.424±0.002|0.190±0.002<br>0.252±0.003|
|GNN-based|||||||
||CompGCN<br>RED-GNN<br>NBFNet|0.883±0.001<br>**0.988±0.002**<br>0.977±0.001|0.751±0.003<br>0.820±0.003<br>0.819±0.002|0.867±0.002<br>0.946±0.001<br>0.946±0.002|0.443±0.001<br>0.502±0.001<br>0.496±0.002|0.265±0.001<br>0.284±0.002<br>0.320±0.001|
||||||||
||EL-RED-GNN<br>EL-NBFNet|**0.990±0.002**<br>**0.985±0.001**|**0.839±0.001**<br>**0.842±0.003**|**0.952±0.003**<br>**0.953±0.002**|**0.504±0.001**<br>0.501±0.003|0.322±0.002<br>**0.332±0.001**|



7 CONCLUSION 

In this paper, we analyze the expressivity of the state-of-the-art GNNs for learning rules in KG reasoning, explaining their superior performance over classical methods. Our analysis sheds light on the rule structures that GNNs can learn. Additionally, our theory motivates an effective labeling method to improve GNN’s expressivity. Moving forward, we will extend our analysis to GNNs with general labeling trick and try to extract explainable rule structures from trained GNN. Limitations and impacts are discussed in Appendix G. 

9 

Published as a conference paper at ICLR 2024 

## ACKNOWLEDGMENTS 

Q. Yao was in part supported by National Key Research and Development Program of China under Grant 2023YFB2903904 and NSFC (No. 92270106). 

## REFERENCES 

Ralph Abboud, Ismail Ilkan Ceylan, Martin Grohe, and Thomas Lukasiewicz. The surprising power of graph neural networks with random node initialization. In _International Joint Conference on Artificial Intelligence_ , 2021. 

- Erik Arakelyan, Daniel Daza, Pasquale Minervini, and Michael Cochez. Complex query answering with neural link predictors. In _International Conference on Learning Representations_ , 2021. 

- Pablo Barceló, Egor V Kostylev, Mikael Monet, Jorge Pérez, Juan Reutter, and Juan-Pablo Silva. The logical expressiveness of graph neural networks. In _International Conference on Learning Representations_ , 2020. 

- Pablo Barcelo, Mikhail Galkin, Christopher Morris, and Miguel Romero Orth. Weisfeiler and leman go relational. In _The First Learning on Graphs Conference_ , 2022. URL https://openreview. net/forum?id=wY_IYhh6pqj. 

- Peter W Battaglia, Jessica B Hamrick, Victor Bapst, Alvaro Sanchez-Gonzalez, Vinicius Zambaldi, Mateusz Malinowski, Andrea Tacchetti, David Raposo, Adam Santoro, Ryan Faulkner, et al. Relational inductive biases, deep learning, and graph networks. _arXiv preprint arXiv:1806.01261_ , 2018. 

- Antoine Bordes, Nicolas Usunier, Alberto Garcia-Duran, Jason Weston, and Oksana Yakhnenko. Translating embeddings for modeling multi-relational data. _Advances in Neural Information Processing Systems_ , 2013. 

- Yixin Cao, Xiang Wang, Xiangnan He, Zikun Hu, and Tat-Seng Chua. Unifying knowledge graph learning and recommendation: Towards a better understanding of user preferences. In _International World Wide Web Conference_ , 2019. 

- David Tena Cucala, Bernardo Cuenca Grau, Boris Motik, and Egor V Kostylev. On the correspondence between monotonic max-sum gnns and datalog. _arXiv preprint arXiv:2305.18015_ , 2023. 

Maarten De Rijke. A note on graded modal logic. _Studia Logica_ , 64(2):271–283, 2000. 

Tim Dettmers, Pasquale Minervini, Pontus Stenetorp, and Sebastian Riedel. Convolutional 2D knowledge graph embeddings. In _AAAI conference on Artificial Intelligence_ , 2017. 

- Justin Gilmer, Samuel S Schoenholz, Patrick F Riley, Oriol Vinyals, and George E Dahl. Neural message passing for quantum chemistry. In _International Conference on Machine Learning_ , 2017. 

- Will Hamilton, Payal Bajaj, Marinka Zitnik, Dan Jurafsky, and Jure Leskovec. Embedding logical queries on knowledge graphs. _Advances in neural information processing systems_ , 31, 2018. 

- Geoffrey E Hinton et al. Learning distributed representations of concepts. In _Annual Conference of the Cognitive Science Society_ , 1986. 

- Xingyue Huang, Miguel Romero Orth, Ismail[˙][˙] Ilkan Ceylan, and Pablo Barceló. A theory of link prediction via relational weisfeiler-leman. _arXiv preprint arXiv:2302.02209_ , 2023. 

- Shaoxiong Ji, Shirui Pan, Erik Cambria, Pekka Marttinen, and S Yu Philip. A survey on knowledge graphs: Representation, acquisition, and applications. _IEEE transactions on neural networks and learning systems_ , 33(2):494–514, 2021. 

- Thomas N Kipf and Max Welling. Semi-supervised classification with graph convolutional networks. In _International Conference on Learning Representations_ , 2016. 

10 

Published as a conference paper at ICLR 2024 

- Stanley Kok and Pedro Domingos. Statistical predicate invention. In _International Conference on Machine Learning_ , 2007. 

- Nada Lavrac and Saso Dzeroski. Inductive logic programming. In _WLP_ , pp. 146–160. Springer, 1994. 

- Richard Liaw, Eric Liang, Robert Nishihara, Philipp Moritz, Joseph E Gonzalez, and Ion Stoica. Tune: A research platform for distributed model selection and training. _arXiv preprint arXiv:1807.05118_ , 2018. 

- Sameh K. Mohamed, Vít Novácek, and Aayah Nounu. Discovering protein drug targets using knowledge graph embeddings. _Bioinformatics_ , 2019. 

- Christopher Morris, Martin Ritzert, Matthias Fey, William L Hamilton, Jan Eric Lenssen, Gaurav Rattan, and Martin Grohe. Weisfeiler and leman go neural: Higher-order graph neural networks. In _AAAI conference on Artificial Intelligence_ , 2019. 

- Christopher Morris, Gaurav Rattan, and Petra Mutzel. Weisfeiler and leman go sparse: Towards scalable higher-order graph embeddings. _Advances in Neural Information Processing Systems_ , 2020. 

- Martin Otto. Graded modal logic and counting bisimulation. _arXiv preprint arXiv:1910.00039_ , 2019. 

- Hongyu Ren, Weihua Hu, and Jure Leskovec. Query2box: Reasoning over knowledge graphs in vector space using box embeddings. In _International Conference on Learning Representations_ , 2019. 

- Ali Sadeghian, Mohammadreza Armandpour, Patrick Ding, and Daisy Zhe Wang. Drum: End-to-end differentiable rule mining on knowledge graphs. _Advances in Neural Information Processing Systems_ , 2019. 

- Ryoma Sato, Makoto Yamada, and Hisashi Kashima. Random features strengthen graph neural networks. In _SIAM International Conference on Data Mining_ , 2021. 

- Michael Schlichtkrull, Thomas N Kipf, Peter Bloem, Rianne van den Berg, Ivan Titov, and Max Welling. Modeling relational data with graph convolutional networks. In _European Semantic Web Conference_ , 2018. 

- Balasubramaniam Srinivasan and Bruno Ribeiro. On the equivalence between positional node embeddings and structural graph representations. _ICLR_ , 2020. 

- Zhiqing Sun, Zhi-Hong Deng, Jian-Yun Nie, and Jian Tang. Rotate: Knowledge graph embedding by relational rotation in complex space. In _International Conference on Learning Representations_ , 2019. 

- DJ Tena Cucala, B Cuenca Grau, Egor V Kostylev, and Boris Motik. Explainable gnn-based models over knowledge graphs. 2022. 

- Komal Teru, Etienne Denis, and Will Hamilton. Inductive relation prediction by subgraph reasoning. In _International Conference on Machine Learning_ , 2020. 

- Kristina Toutanova and Danqi Chen. Observed versus latent features for knowledge base and text inference. In Alexandre Allauzen, Edward Grefenstette, Karl Moritz Hermann, Hugo Larochelle, and Scott Wen-tau Yih (eds.), _Proceedings of the 3rd Workshop on Continuous Vector Space Models and their Compositionality_ , pp. 57–66, Beijing, China, July 2015. Association for Computational Linguistics. doi: 10.18653/v1/W15-4007. URL https://aclanthology.org/W15-4007. 

- Théo Trouillon, Johannes Welbl, Sebastian Riedel, Éric Gaussier, and Guillaume Bouchard. Complex embeddings for simple link prediction. In _International Conference on Machine Learning_ , 2016. 

- Shikhar Vashishth, Soumya Sanyal, Vikram Nitin, and Partha Talukdar. Composition-based multirelational graph convolutional networks. In _International Conference on Learning Representations_ , 2020. 

11 

Published as a conference paper at ICLR 2024 

Keyulu Xu, Weihua Hu, Jure Leskovec, and Stefanie Jegelka. How powerful are graph neural networks? In _International Conference on Learning Representations_ , 2019. 

- Fan Yang, Zhilin Yang, and William W Cohen. Differentiable learning of logical rules for knowledge base reasoning. _Advances in Neural Information Processing Systems_ , 2017. 

- Jiaxuan You, Jonathan M Gomes-Selman, Rex Ying, and Jure Leskovec. Identity-aware graph neural networks. In _AAAI Conference on Artificial Intelligence_ , 2021. 

- Muhan Zhang, Pan Li, Yinglong Xia, Kai Wang, and Long Jin. Labeling trick: A theory of using graph neural networks for multi-node representation learning. _Advances in Neural Information Processing Systems_ , 2021. 

- Shuai Zhang, Yi Tay, Lina Yao, and Qi Liu. Quaternion knowledge graph embeddings. _Advances in Neural Information Processing Systems_ , 32, 2019. 

Yongqi Zhang and Quanming Yao. Knowledge graph reasoning with relational digraph. In _International World Wide Web Conference_ , 2022. 

- Yongqi Zhang, Zhanke Zhou, Quanming Yao, Xiaowen Chu, and Bo Han. Adaprop: Learning adaptive propagation for graph neural network based knowledge graph reasoning. In _Proceedings of the 29th ACM SIGKDD Conference on Knowledge Discovery and Data Mining_ , pp. 3446–3457, 2023. 

- Zhaocheng Zhu, Zuobai Zhang, Louis-Pascal Xhonneux, and Jian Tang. Neural bellman-ford networks: A general graph neural network framework for link prediction. _Advances in Neural Information Processing Systems_ , 2021. 

Zhaocheng Zhu, Xinyu Yuan, Mikhail Galkin, Sophie Xhonneux, Ming Zhang, Maxime Gazeau, and Jian Tang. A*net: A scalable path-based reasoning approach for knowledge graphs. _arXiv preprint arXiv:2206.04798_ , 2022. 

12 

Published as a conference paper at ICLR 2024 

## A RULE ANALYSIS 

We first give a simple proof for Proposition 3.1. 

_proof of Proposition 3.1._ Since _R_ (h _, x_ ) is equivalent to _∃zR_ ( _z, x_ ) _∧ Ph_ ( _z_ ), where _Ph_ ( _z_ ) is the constant predicate only satisfied at entity _h_ . Because _R_ ( _z, x_ ) can describe the rule structure of ( _h, R,_ ?), _∃zR_ ( _z, x_ ) _∧ Ph_ ( _z_ ) can describe the rule structure of ( _h, R,_ ?) as well. 

We use the notation _G, v |_ = _Pi_ ( _G, v_ ⊭ _Pi_ ) to represent that the unary predicate _Pi_ ( _x_ ) is (not) satisfied at entity _v_ . 

**Definition A.1** (Definition of graded modal logic) **.** A formula in graded modal logic of KG _G_ is recursively defined as follows: 

1. If _φ_ ( _x_ ) = _⊤_ , _G, v |_ = _φ_ if _v_ is an entity in KG; 

2. If _φ_ ( _x_ ) = _Pc_ ( _x_ ), _G, v |_ = _φ_ if and only if _v_ has the property _Pc_ or can be uniquely identified by constant c; 

3. If _φ_ ( _x_ ) = _φ_ 1( _x_ ) _∧ φ_ 2( _x_ ), _G, v |_ = _φ_ if and only if _G, v |_ = _φ_ 1 and _G, v |_ = _φ_ 2; 

4. If _φ_ ( _x_ ) = _¬ϕ_ ( _x_ ), _G, v |_ = _φ_ if and only if _G, v_ ⊭ _ϕ_ ; 

5. If _φ_ ( _x_ ) = _∃[≥][N] y, Rj_ ( _y, x_ ) _∧ ϕ_ ( _y_ ), _G, v |_ = _φ_ if and only if the set of entities _{u|u ∈ NRj_ ( _v_ ) and _G, u |_ = _ϕ}_ has cardinality at least _N_ . 

**Corollary A.2.** _C_ 3(h _, x_ ) _are formulas in CML_ [ _G,_ h] _._ 

_Proof. C_ 3(h _, x_ ) is a formula in CML[ _G,_ h] as it can be recursively defined as follows 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2024_Understanding_Expressivity_of_GNN_in_Rule_Learning_images/iclr_2024_Understanding_Expressivity_of_GNN_in_Rule_Learning.pdf-0013-13.png)


**Corollary A.3.** _I_ 1(h _, x_ ) _is a formula in CML_ [ _G,_ h] _._ 

_Proof. I_ 1(h _, x_ ) is a formula in CML[ _G,_ h] as it can be recursively defined as follows 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2024_Understanding_Expressivity_of_GNN_in_Rule_Learning_images/iclr_2024_Understanding_Expressivity_of_GNN_in_Rule_Learning.pdf-0013-16.png)


**Corollary A.4.** _T_ (h _, x_ ) _is a formula in CML_ [ _G,_ h] _._ 

_Proof._ By Corollary A.2, _C_ 3 _[′]_[(][h] _[, x]_[) :=] _[ ∃][z]_[1] _[z]_[2] _[, R]_[1][(][h] _[, z]_[1][)] _[ ∧][R]_[2][(] _[z]_[1] _[, z]_[2][)] _[ ∧][R]_[4][(] _[z]_[2] _[, x]_[)][ and] _[ C]_ 3 _[⋆]_[(][h] _[, x]_[) :=] _∃z_ 1 _z_ 2 _, R_ 1(h _, z_ 1) _∧ R_ 3( _z_ 1 _, z_ 2) _∧ R_ 5( _z_ 2 _, x_ ) are formulas in CML[ _G,_ h]. Thus _T_ (h _, x_ ) = _C_ 3 _[′]_[(][h] _[, x]_[)] _[ ∧] C_ 3 _[⋆]_[(][h] _[, x]_[)][ is a formula in CML][[] _[G,]_[ h][]][.] 

**Corollary A.5.** _U[′]_ (h _, x_ ) _is a formula in CML_ [ _G,_ h _,_ c] _._ 

13 

Published as a conference paper at ICLR 2024 

_Proof. U[′]_ (h _, x_ ) is a formula in CML[ _G,_ h _,_ c] as it can be recursively defined as follows 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2024_Understanding_Expressivity_of_GNN_in_Rule_Learning_images/iclr_2024_Understanding_Expressivity_of_GNN_in_Rule_Learning.pdf-0014-02.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2024_Understanding_Expressivity_of_GNN_in_Rule_Learning_images/iclr_2024_Understanding_Expressivity_of_GNN_in_Rule_Learning.pdf-0014-03.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2024_Understanding_Expressivity_of_GNN_in_Rule_Learning_images/iclr_2024_Understanding_Expressivity_of_GNN_in_Rule_Learning.pdf-0014-04.png)


where the constant c ensures that there is only one entity satisfied for unary predicate _φ_ 3( _x_ ). 

**Example of rules** We can find some relations in reality corresponding to rules in Figure 2. Here are two examples of _C_ 3 and _I_ 1: 

- Relation nationality ( _C_ 3): Einstein _→_ born_in Ulm _→_ hometown_of Born _→nationality_ Germany; 

- Relation father ( _I_ 1): A _→_ spouse B _→_ parent C and D _→_ sisterhood B. 

**Rule structures in real datasets** To show that the expressivity is meaningful in our paper, we select three rule structures from Family and FB15k-237 in Figure 5 to show the existence of rule structures in real datasets. With the definition of CML, the rule structure in Figure 5(a) is not a formula in CML and rule structures in Figure 5(b) and 5(c) are formulas in CML. The real rules shows that rules defined by CML is common in real-world datasets and the rules beyond CML also exist, which highlights the importance of our work. 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2024_Understanding_Expressivity_of_GNN_in_Rule_Learning_images/iclr_2024_Understanding_Expressivity_of_GNN_in_Rule_Learning.pdf-0014-10.png)


**----- Start of picture text -----**<br>
brother Scottish Walter<br>Americans Hill<br>1482 nephew 1432 father 1249 _uncle 1480 HardyOliver USA Aliens PredatorAlien vs. genre actionfilm<br>Las Vegas Alien 4<br>Target Relations son Target Relations nationality Target Relations genre<br>(a) (b) (c)<br>_ethnicity distribution<br>marriage<br>location<br>nominated for<br>distribution<br>_nominated for prequel<br>story by<br>**----- End of picture text -----**<br>


Figure 5: Some rule structures in real datasets. The rule structure (a) is from Family dataset and is not a rule formula in CML[ _G,_ h], which cannot not be learned by QL-GNN. The rule structures (b) and (c) are from FB15k-237 dataset and are rule formulas in CML[ _G,_ h], which can be learned by QL-GNN. 

**Summary** Here we give Table 3 to illustrate the correspondence between GNNs for KG reasoning, rule structures, and theories presented in our paper. 

Table 3: Whether GNNs investigated in our paper can learn the rule formulas in Figure 2 and 3 and the exemplar methods of these GNNs. ✓(✗) mean the corresponding GNN can(not) lean the rule formula. 

|formula.||||
|---|---|---|---|
|Rule formula|_C_3(h_, x_)<br>_I_1(h_, x_)<br>_T_(h_, x_)<br>_U_(h_, x_)|Theoretical result|Exemplar Methods|
|||||
|Classical<br>QL-GNN<br>EL-GNN|✗<br>✗<br>✗<br>✗<br>✓<br>✓<br>✓<br>✗<br>✓<br>✓<br>✓<br>✓|Theorem 3.4<br>Theorem 3.2<br>Proposition 4.1|R-GCN, CompGCN<br>NBFNet, RED-GNN<br>EL-NBFNet/RED-GNN|



B RELATION BETWEEN QL-GNN AND NBFNET/RED-GNN 

In this part, we show that NBFNet and RED-GNN are special cases of QL-GNN in Table 4 and 5 respectively. 

14 

Published as a conference paper at ICLR 2024 

Table 4: NBFNet is a special case of QL-GNN. 

|||NBFNet|NBFNet|
|---|---|---|---|
|||||
|Query representation<br>Non-query representation<br>MPNN<br>Triplet score||Relation embedding<br>0<br>AGGREGATE<br>��<br>MESSAGE<br>�<br>**_h_**(_t−_1)<br>_x_<br>_,_**_w_**_q_(_x, r, v_)<br>����(_x, r, v_)_∈E_(_v_)<br>�<br>_∪_<br>�<br>**_h_**(0)<br>_v_<br>��<br>Feed-forward network||
||Table 5: RED-GNN is a special case ofQL-GNN.<br>RED-GNN<br>Query representation<br>0<br>Non-query representation<br>NULL<br>MPNN<br>_δ_<br>��<br>_{es,r}_:(_es,r,e_)_∈Eℓeq φ_<br>�<br>**_h_**_ℓ−_1<br>_eq,es,_**_h_**_ℓ_<br>_r_<br>��<br>Triplet score<br>Linear transformation|||
||Query representation<br>Non-query representation<br>MPNN<br>Triplet score||0<br>NULL<br>_δ_<br>��<br>_{es,r}_:(_es,r,e_)_∈Eℓeq φ_<br>�<br>**_h_**_ℓ−_1<br>_eq,es,_**_h_**_ℓ_<br>_r_<br>��<br>Linear transformation|



C PROOF 

We use the notation _G,_ ( _h, t_ ) _|_ = _Rj_ ( _G,_ ( _h, t_ ) ⊭ _Rj_ ) to denote _Rj_ ( _x, y_ ) is (not) satisfied at _h, t_ . 

- C.1 BASE THEOREM: WHAT KIND OF LOGICAL FORMULAS CAN MPNN BACKBONE FOR KG LEARN? 

In this section, we analyze the expressivity of MPNN backbone (1) for learning logical formulas in KG. This section is the extension of Barceló et al. (2020) to KG. 

In a KG _G_ = ( _V, E, R_ ), MPNN with _L_ layers is a type of neural network that applies graph _G_ and initial entity representation **e**[(0)] _v_ to learn the representations **e**[(] _v[L]_[)] _, v ∈V_ . MPNN employs messagepassing mechanisms (Gilmer et al., 2017) to propagate information between entities in graph. The _k_ -th layer of MPNN updates the entity representation via the following message-passing formula 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2024_Understanding_Expressivity_of_GNN_in_Rule_Learning_images/iclr_2024_Understanding_Expressivity_of_GNN_in_Rule_Learning.pdf-0015-08.png)


where _δ_ and _ϕ_ are combination and aggregation functions respectively, _ψ_ is the message function encoding the relation _R_ and entity _u_ neighboring to _v_ , _{{· · · }}_ is a multiset, and _NR_ ( _v_ ) is the neighboring entity set _{u|_ ( _u, R, v_ ) _∈E}_ . 

To understand how MPNN can learn logical formulas, we regard logical formula _φ_ ( _x_ ) as a binary classifier indicating whether _φ_ ( _x_ ) is satisfied at entity _x_ . Then, we commence with the following definition. 

**Definition C.1.** A MPNN captures a logical formula _φ_ ( _x_ ) if and only if given any graph _G_ , the MPNN representation can be mapped to a binary value, where True indicates that _φ_ ( _x_ ) satisfies on entity _x_ , while False does not satisfy. 

According to the above definition, MPNN can learn logical formula in KG by encoding whether these logical formulas is satisfied in the representation of the corresponding entity. For example, if MPNN can learn a logical formula _φ_ ( _x_ ), it implies that **e**[(] _v[L]_[)] can be mapped to a binary value True/False by a function indicating whether _φ_ ( _x_ ) is satisfied at entity _v_ . Previous work (Barceló et al., 2020) has proven that vanilla GNN for single-relational graph can learn the logical formulas from graded modal logic (De Rijke, 2000; Otto, 2019) (a.k.a., Counting extension of Modal Logic, CML). In this section, we will present a similar theory of MPNN for KG. 

The insight of MPNN’s ability to learn formulas in CML lies in the alignment between certain CML formulas and the message-passing mechanism, which also holds for KG. Specifically, _∃[≥][N] y_ ( _Rj_ ( _y, x_ ) _∧ φ_ ( _y_ )) is the formula aligned with MPNN’s message-passing mechanism and allows to check the property of neighbor _y_ of entity variable _x_ . We use notation CML[ _G_ ] to denote 

15 

Published as a conference paper at ICLR 2024 

CML of a graph _G_ . Then, we give the following theorem to find out the kind of logical formula MPNN (1) can learn in KG. 

**Theorem C.2.** _In a KG G, a logical formula φ_ ( _x_ ) _is learned by MPNN (1) from its representations if and only if φ_ ( _x_ ) _is a formula in CML_ [ _G_ ] _._ 

Our theorem can be viewed as an extension of Theorem 4.2 in Barceló et al. (2020) to KG and is the elementary tool for analyzing the expressivity of GNNs for KG reasoning. The proof of Theorem C.2 is in Appendix C and employs novel techniques that specifically account for relation types. Our theorem shows that CML of KG is the tightest subclass of logic that MPNN can learn. Similarly, our theorem is about the ability to implicitly learn logical formulas by MPNN rather than explicitly extracting them. 

## C.2 PROOF OF THEOREM C.2 

The backward direction of Theorem C.2 is proven by constructing a MPNN that can learn any formula _φ_ ( _x_ ) in CML. The forward direction relies on the results from recent theoretical results in Otto (2019). Our theorem can be seen as an extension of Theorem 4.2 in Barceló et al. (2020) to KG. 

We first prove the backward direction of Theorem C.2. 

**Lemma C.3.** _Each formula φ_ ( _x_ ) _in CML can be learned by MPNN (1) from its entity representations._ 

_Proof._ Let _φ_ ( _x_ ) be a formula in CML. We decompose _φ_ into a series of sub-formulas sub[ _φ_ ] = ( _φ_ 1 _, φ_ 2 _, · · · , φL_ ) where _φk_ is a sub-formula of _φℓ_ if _k ≤ ℓ_ and _φ_ = _φL_ . Assume the MPNN representation **e**[(] _v[i]_[)] _∈_ R _[L] , v ∈V, i_ = 1 _· · · L_ . In this proof, the theoretical analysis will based on the following simple choice of (1) 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2024_Understanding_Expressivity_of_GNN_in_Rule_Learning_images/iclr_2024_Understanding_Expressivity_of_GNN_in_Rule_Learning.pdf-0016-09.png)


with _σ_ = min(max(0 _, x_ ) _,_ 1), **A** _Rj ,_ **C** _∈_ R _[L][×][L]_ and **b** _∈_ R _[L]_ . The entries of the _ℓ_ -th columns of **A** _Rj ,_ **C** , and **b** depend on the sub-formulas of _φ_ as follows: 

- **Case 0.** if _φℓ_ ( _x_ ) = _Pℓ_ ( _x_ ) where _Pℓ_ is a unary predicate, **C** _ℓℓ_ = 1; 

- **Case 1.** if _φℓ_ ( _x_ ) = _φj_ ( _x_ ) _∧ φk_ ( _x_ ), **C** _jℓ_ = **C** _kℓ_ = 1 and **b** _ℓ_ = _−_ 1; 

- **Case 2.** if _φℓ_ = _¬φk_ ( _x_ ), **C** _kℓ_ = _−_ 1 and **b** _ℓ_ = 1; 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2024_Understanding_Expressivity_of_GNN_in_Rule_Learning_images/iclr_2024_Understanding_Expressivity_of_GNN_in_Rule_Learning.pdf-0016-14.png)


with all the other values set to 0. 

Before the proof, for every entity _v ∈V_ , the initial representation **e**[(0)] _v_ = ( _t_ 1 _, t_ 2 _, · · · , tn_ ) has _tℓ_ = 1 if the sub-formula _φℓ_ = _Pℓ_ ( _x_ ) is satisfied at _v_ , and _tℓ_ = 0 otherwise. 

Let _G_ = ( _V, E, R_ ) be a KG. We next prove that for every _φℓ ∈_ sub[ _φ_ ] and every entity _v ∈V_ it holds that 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2024_Understanding_Expressivity_of_GNN_in_Rule_Learning_images/iclr_2024_Understanding_Expressivity_of_GNN_in_Rule_Learning.pdf-0016-18.png)


for every _ℓ ≤ i ≤ L_ . 

Now, we prove this by induction of the number of formulas in _φ_ . 

**Base case:** One sub-formula in _φ_ . In this case, the formula is an atomic predicate _φ_ = _φℓ_ ( _x_ ) = _Pℓ_ ( _x_ ). Because **C** _ℓℓ_ = 1 and ( **e**[(0)] _v_[)] _ℓ_[=][1] _[,]_[ (] **[e]**[(0)] _v_[)] _i_[=][0] _[, i]_[=] _[ℓ]_[,][we][have][(] **[e]**[(1)] _v_[)] _ℓ_[=][1][if] _[G, v][|]_[=] _[φ] ℓ_[and] ( **e**[(1)] _v_[)] _ℓ_[= 0][ otherwise.][For] _[ i][ ≥]_[1][,] **[ e]**[(] _v[i]_[)] satisfies the same property. 

**Induction Hypothesis:** _k_ sub-formulas in _φ_ with _k < ℓ_ . Assume **e**[(] _v[i]_[)][1][ if] _[ G, v][|]_[=] _[φ][k]_[and] � � _k_[=] **e**[(] _v[i]_[)] _[≤][i][ ≤][L]_[.] � � _k_[= 0][ otherwise for] _[ k]_ 

16 

Published as a conference paper at ICLR 2024 

**Proof:** _ℓ_ sub-formulas in _φ_ . Let _i ≥ ℓ_ . Case 1-3 should be considered. 

Case 1. Let _φℓ_ ( _x_ ) = _φj_ ( _x_ ) _∧ φk_ ( _x_ ). Then **C** _jℓ_ = **C** _kℓ_ = 1 and **b** _ℓ_ = _−_ 1. Then we have 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2024_Understanding_Expressivity_of_GNN_in_Rule_Learning_images/iclr_2024_Understanding_Expressivity_of_GNN_in_Rule_Learning.pdf-0017-03.png)


By the induction hypothesis, ( **e**[(] _v[i][−]_[1)] ) _j_ = 1 if only if _G, v |_ = _φj_ and ( **e**[(] _v[i][−]_[1)] ) _j_ = 0 otherwise. Similarly, ( **e**[(] _v[i][−]_[1)] ) _k_ = 1 if and only if _G, v |_ = _φk_ and ( **e**[(] _v[i][−]_[1)] ) _k_ = 0 otherwise. Then we have ( **e**[(] _v[i]_[)][)] _ℓ_[= 1][ if and only if][ (] **[e]** _v_[(] _[i][−]_[1)] ) _j_ +( **e**[(] _v[i][−]_[1)] ) _k −_ 1 _≥_ 1, which means ( **e**[(] _v[i][−]_[1)] ) _j_ = 1 and ( **e**[(] _v[i][−]_[1)] ) _k_ = 1. Then ( **e**[(] _v[i]_[)][)] _ℓ_[=][1][if][and][only][if] _[G, v][|]_[=] _[φ] j_[and] _[G, v][|]_[=] _[φ] k_[,][i.e.,] _[G, v][|]_[=] _[φ] ℓ_[,][and][(] **[e]**[(] _v[i]_[)][)] _ℓ_[=][0] otherwise. 

Case 2. Let _φℓ_ ( _x_ ) = _¬φk_ ( _x_ ). Because of **C** _kℓ_ = _−_ 1 and **b** _ℓ_ = 1, we have 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2024_Understanding_Expressivity_of_GNN_in_Rule_Learning_images/iclr_2024_Understanding_Expressivity_of_GNN_in_Rule_Learning.pdf-0017-06.png)


By the induction hypothesis, ( **e**[(] _v[i][−]_[1)] ) _k_ = 1 if and only if _G, v |_ = _φk_ and ( **e**[(] _v[i][−]_[1)] ) _k_ = 0 otherwise. Then we have ( **e**[(] _v[i]_[)][)] _ℓ_[= 1][ if and only if] _[ −]_[(] **[e]** _v_[(] _[i][−]_[1)] ) _k_ + 1 _≥_ 1, which means ( **e**[(] _v[i][−]_[1)] ) _k_ = 0. Because ( **e**[(] _v[i][−]_[1)] ) _k_ = 0 if and only if _G, v_ ⊭ _φk_ , we have ( **e**[(] _v[i]_[)][)] _ℓ_[= 1][ if and only if] _[ G, v]_[⊭] _[φ] k_[, i.e.,] _[ G, v][|]_[=] _[ φ] ℓ_[,] and ( **e**[(] _v[i]_[)][)] _ℓ_[= 0][ otherwise.] 

Case 3. Let _φℓ_ ( _x_ ) = _∃[≥][N] y_ ( _Rj_ ( _y, x_ ) _∧ φk_ ( _y_ )). Because of � **A** _Rj_ � _kℓ_[=][1][ and] **[ b]** _[ℓ]_[=] _[−][N]_[+ 1][, we] have 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2024_Understanding_Expressivity_of_GNN_in_Rule_Learning_images/iclr_2024_Understanding_Expressivity_of_GNN_in_Rule_Learning.pdf-0017-09.png)


By the induction hypothesis, ( **e**[(] _u[i][−]_[1)] ) _k_ = 1 if and only if _G, u |_ = _φk_ and ( **e**[(] _u[i][−]_[1)] ) _k_ = 0 otherwise. Let _m_ = _|{u|u ∈NRj_ ( _v_ ) and _G, u |_ = _φk}|_ . Then we have ( **e**[(] _v[i]_[)][)] _ℓ_[=][1][if][and][only][if] � _u∈NRj_ ( _v_ )[(] **[e]**[(] _u[i][−]_[1)] ) _k − N_ + 1 _≥_ 1, which means _m ≥ N_ . Because _G, u |_ = _φk_ , _u_ is connected to _v_ with relation _Rj_ , and _m ≥ N_ , we have ( **e**[(] _v[i]_[)][)] _ℓ_[=][1][ if and only if] _[ G, v][|]_[=] _[φ] ℓ_[and][ (] **[e]** _v_[(] _[i]_[)][)] _ℓ_[=][0] otherwise. 

To learn a logical formula _φ_ ( _x_ ), we only apply a linear classifier to **e**[(] _v[L]_[)] _, v ∈V_ to extract the component of **e**[(] _v[L]_[)] corresponding to _φ_ . If _G, v |_ = _φ_ , the value of the corresponding extracted component is 1. 

Next, we prove the forward direction of Theorem C.2. 

**Theorem C.4.** _A formula φ_ ( _x_ ) _is learned by MPNN (1) if it can be expressed as a formula in CML._ 

To prove Theorem C.4, we introduce Definition C.5, Lemma C.6, Theorem C.7, and Lemma C.8. **Definition C.5** (Unraveling tree) **.** Let _G_ be a KG, _v_ be entity in _G_ , and _L ∈_ N. The unravelling of _v_ in _G_ at depth _L_ , denoted by Unr _[L] G_[(] _[v]_[)][, is a tree composed of] 

- a node ( _v, R_ 1 _, u_ 1 _, · · · , Ri, ui_ ) for each path ( _v, R_ 1 _, u_ 1 _, · · · , Ri, ui_ ) in _G_ with _i ≤ L_ , 

- an edge _Ri_ between ( _v, R_ 1 _, u_ 1 _, · · · , Ri−_ 1 _, ui−_ 1) and ( _v, R_ 1 _, u_ 1 _, · · · , Ri, ui_ ) when ( _ui, Ri, ui−_ 1) is a triplet in _G_ (assume _u_ 0 is _v_ ), and 

- each node ( _v, R_ 1 _, u_ 1 _, · · · , Ri, ui_ ) has the same properties as _ui_ in _G_ . 

**Lemma C.6.** _Let G and G[′] be two KGs, v and v[′] be two entities in G and G[′] respectively. Then for every L ∈_ N _, the RWL test (Barcelo et al., 2022) assigns the same color/hash to v and v[′] at round L if and only if there is an isomorphism between Unr[L] G_[(] _[v]_[)] _[ and Unr][L] G[′]_[(] _[v][′]_[)] _[ sending][ v][ to][ v][′][.]_ 

17 

Published as a conference paper at ICLR 2024 

_Proof._ **Base Case:** When _L_ = 1, the result is obvious. 

**Induction Hypothesis:** Relational WL (RWL) test assigns the same color to _v_ and _v[′]_ at round _L −_ 1 if and only if there is an isomorphism between Unr _[L] G[−]_[1] ( _v_ ) and Unr _[L] G[−][′]_[1][(] _[v][′]_[)][ sending] _[ v]_[ to] _[ v][′]_[.] 

**Proof:** In the _L_ -th round, 

- Prove “same color _⇒_ isomorphism”. 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2024_Understanding_Expressivity_of_GNN_in_Rule_Learning_images/iclr_2024_Understanding_Expressivity_of_GNN_in_Rule_Learning.pdf-0018-05.png)


Because _c[L]_ ( _v_ ) = _c[L]_ ( _v[′]_ ), we have _c[L][−]_[1] ( _v_ ) = _c[L][−]_[1] ( _v[′]_ ), and there exists an entity pair ( _u, u[′]_ ) _, u ∈ NRi_ ( _v_ ) _, u[′] ∈NRi_ ( _v[′]_ ) that 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2024_Understanding_Expressivity_of_GNN_in_Rule_Learning_images/iclr_2024_Understanding_Expressivity_of_GNN_in_Rule_Learning.pdf-0018-07.png)


Then we have _c[L][−]_[1] ( _u_ ) = _c[L][−]_[1] ( _u[′]_ ). According to induction hypothesis, we have Unr _[L] G[−]_[1] ( _u_ ) = _[∼]_ Unr _[L] G[−][′]_[1][(] _[u][′]_[)][.][Also, because the edge connecting entity pair][ (] _[v, u]_[)][ and][ (] _[v][′][, u][′]_[)][ is] _[ R][i]_[, so there is an] isomorphism between Unr _[L] G_[(] _[v]_[)][ and Unr] _G[L][′]_[(] _[v][′]_[)][ sending] _[ v]_[ to] _[ v][′]_[.] 

- Prove “isomorphism _⇒_ same color”. 

Because there exists an isomorphism _π_ between Unr _[L] G_[(] _[v]_[)][ and][ Unr] _[L] G[′]_[(] _[v][′]_[)][ sending] _[ v]_[ to] _[ v][′]_[, assume] _[ π]_[ is] an bijective between the neighbors of _v_ and _v[′]_ , e.g, _u ∈NRi_ ( _v_ ) _, u[′] ∈NRi_ ( _v[′]_ ) and _u[′] i_[=] _[ π]_[(] _[u][i]_[)][, the] relation between entity pair ( _u, v_ ) and ( _u[′] , v[′]_ ) is _Ri_ . 

Next we prove _c[L][−]_[1] ( _u_ ) = _c[L][−]_[1] ( _u[′]_ ). Because Unr _[L] G_[(] _[v]_[)][and][Unr] _[L] G_[(] _[v][′]_[)][are][isomorphism,][and] _[π]_ maps _u ∈NRi_ ( _v_ ) to _u[′] ∈NRi_ ( _v[′]_ ), for the left tree with _L −_ 1 depth, i.e., Unr _[L] G[−]_[1] ( _u_ ) and Unr _[L] G[−][′]_[1][(] _[u][′]_[)][,] _[ π]_[ can be the isomorphism mapping between][ Unr] _[L] G[−]_[1] ( _u_ ) and Unr _[L] G[−][′]_[1][(] _[u][′]_[)][.][According to] induction hypothesis, we have _c[L][−]_[1] ( _u_ ) = _c[L][−]_[1] ( _u[′]_ ). Because Unr _[L] G_[(] _[v]_[)] _[ ∼]_[=][ Unr] _[L] G[′]_[(] _[v][′]_[)][, we also have] Unr _[L] G[−]_[1] ( _u_ ) = _[∼]_ Unr _[L] G[−][′]_[1][(] _[u][′]_[)][ which means] _[ c][L][−]_[1][(] _[u]_[)][=] _[c][L][−]_[1][(] _[u][′]_[)][.][After running RWL test, we have] _c[L]_ ( _v_ ) = _c[L]_ ( _v[′]_ ). 

**Theorem C.7.** _Let φ_ ( _x_ ) _be a unary formula in the formal description of graph G in Section 3.1. If φ_ ( _x_ ) _is not equivalent to a formula in CML, there exist two KGs G and G[′] and two entities v in G and v[′] in G[′] such that Unr[L] G_[(] _[v]_[)] _[ ∼]_[=] _[ Unr][L] G[′]_[(] _[v][′]_[)] _[ for every][ L][ ∈]_[N] _[ and such that][ G, v][|]_[=] _[ φ][ but][ G][′][, v][′]_[⊭] _[φ][.]_ 

> _Proof._ The theorem follows directly from Theorem 2.2 in Otto (2019). Because _G, v ∼_ # _G[′] , v[′]_ and Unr _[L] G_[(] _[v]_[)] _[ ∼]_[=][ Unr] _[L] G[′]_[(] _[v][′]_[)][ are equivalent with the definition of counting bisimulation (i.e., notation] _∼_ #). 

**Lemma C.8.** _If a formula φ_ ( _x_ ) _is not equivalent to any formula in CML, there is no MPNN (1) that can learn φ_ ( _x_ ) _._ 

_Proof._ Assume for a contradiction that there exists a MPNN that can learn _φ_ ( _x_ ). Since _φ_ ( _x_ ) is not equivalent to any formula in CML, with Theorem C.7, there exists two KGs _G_ and _G[′]_ and two entities _v_ in _G_ and _v[′]_ in _G[′]_ such that Unr _[L] G_[(] _[v]_[)] _[∼]_[=][Unr] _[L] G[′]_[(] _[v][′]_[)][ for every] _[ L][∈]_[N][ and such that] _[ G, v][|]_[=] _[φ]_[ and] _G[′] , v[′]_ ⊭ _φ_ . By Lemma C.6, because Unr _[L] G_[(] _[v]_[)] _[ ∼]_[=][ Unr] _[L] G[′]_[(] _[v][′]_[)][ for every] _[ L][ ∈]_[N][, we have] **[ e]** _v_[(] _[L]_[)] = **e**[(] _v[L][′]_[)][.] But this contradicts the assumption that MPNN is supposed to learn _φ_ ( _x_ ). 

_Proof of Theorem C.4._ Theorem can be obtained directly from Lemma C.8. 

_Proof of Theorem C.2._ Theorem can be obtained directly by combining Lemma C.3 and Theorem C.4. 

The following two remarks intuitively explain why MPNN can learn formulas in CML. 

18 

Published as a conference paper at ICLR 2024 

_Remark_ C.9 _._ Theorem C.2 applies to both CML[ _G_ ] and CML[ _G,_ c1 _,_ c2 _, · · · ,_ c _k_ ]. The atomic unary predicate _Pi_ ( _x_ ) in CML of graph _G_ is learned by the initial representations **e**[(0)] _v[, v][∈V]_[,][which] can be achieved by assigning special vectors to **e**[(0)] _v[, v][∈V]_[.][In particular,][the constant predicate] _Pc_ ( _x_ ) in CML[ _G,_ c] is learned by assigning a unique vector (e.g., one-hot vector for different entities) as the initial representation of the entity with unique identifier c. The other sub-formulas _¬φ_ ( _x_ ) _, φ_ 1( _x_ ) _∧ φ_ 2( _x_ ) in Definition A.1 can be learned by continuous logical operations (Arakelyan et al., 2021) which are independent of message-passing mechanisms. 

_Remark_ C.10 _._ Assume the ( _i −_ 1)-th layer representations **e**[(] _v[i][−]_[1)] _, v ∈V_ can learn the formula _φ_ ( _x_ ) in CML, the _i_ -th layer representations **e**[(] _v[i]_[)] _[, v][∈V]_[of MPNN can learn] _[ ∃][≥][N][y, R] j_[(] _[y, x]_[)] _[ ∧][φ]_[(] _[y]_[)][ with] specific aggregation function in (1) because **e**[(] _v[i]_[)] _[, v][∈V]_[can aggregate the logical formulas in the] one-hop neighbor representation **e**[(] _v[i][−]_[1)] _, v ∈V_ (i.e., _φ_ ( _x_ )) with message-passing mechanisms. 

The following remark clarifies the scope of Theorem C.2 and 3.2. 

_Remark_ C.11 _._ The positive results for our theorem (e.g., a MPNN variant can learn a logical formula) hold for MPNNs powerful than the MPNN we construct in (2), while our negative results (e.g., a MPNN variant cannot learn a logical formula) hold for any general MPNNs (1). Hence, the backward direction remains valid irrespective of the aggregate and combine operators under consideration. This limitation is inherent to the MPNN architecture represented by (1) and not specific to the chosen representation update functions. On the other hand, the forward direction holds for MPNNs that are more powerful than (2). 

## C.3 PROOF OF THEOREM 3.2 

**Definition C.12.** QL-GNN learns a rule formula _R_ (h _, x_ ) if and only if given any graph _G_ , the QL-GNN’s score of a new triplet ( _h, R, t_ ) can be mapped to a binary value, where True indicates that _R_ (h _, x_ ) satisfies on entity _t_ , while False does not satisfy. 

_Proof._ We set the KG as _G_ and restrict the unary formulas in CML[ _G,_ h] to the form of _R_ (h _, x_ ). This theorem is directly obtained by Theorem C.2 because constant _h_ can be equivalently transformed to constant predicate _Ph_ ( _x_ ). 

_Proof of Corollary 3.3._ **Base case:** Since the unary predicate can be encoded into the initial representation of the entity according to Section C.1. Then the base case is obvious. 

**Recursion rule:** Since the rule structures _R_ (h _, x_ ) _, R_ 1(h _, x_ ) _, R_ 2(h _, x_ ) are unary predicate and can be learned by QL-GNN, they are formulas in CML[ _G,_ h]. According to recursive definition of CML, _R_ 1(h _, x_ ) _∧ R_ 2(h _, y_ ), _∃[≥][N] y_ ( _Ri_ ( _y, x_ ) _∧ R_ (h _, y_ )) are also formulas in CML[ _G,_ h], therefore can be learned by QL-GNN. 

## C.4 PROOF OF THEOREM 3.4 

**Definition C.13.** CompGCN learns a rule formula _R_ ( _x, y_ ) if and only if given any graph _G_ , the QL-GNN’s score of a new triplet ( _h, R, t_ ) can be mapped to a binary value, where True indicates that _R_ ( _x, y_ ) satisfies on entity pair ( _h, t_ ), while False does not satisfy. 

_Proof._ According to Theorem C.2, the MPNN representation **e**[(] _v[L]_[)] can represent the formulas in CML[ _G_ ]. Assume _φ_ 1( _x_ ) and _φ_ 2( _y_ ) can be represented by the MPNN representation **e**[(] _v[L]_[)] _, v ∈V_ and there exists two functions _g_ 1 and _g_ 2 that can extract the logical formulas from **e**[(] _v[L]_[)] , i.e., _gi_ ( **e**[(] _v[L]_[)] ) = 1 if _G, v |_ = _φi_ and _gi_ ( **e**[(] _v[L]_[)] ) = 0 if _G, v_ ⊭ _φi_ for _i_ = 1 _,_ 2. We show how the following two logical operators can be learned by _s_ ( _h, R, t_ ) for candidate triplet ( _h, R, t_ ): 

- Conjunction: _φ_ 1( _x_ ) _∧ φ_ 2( _y_ ). The conjunction of _φ_ 1( _x_ ) _, φ_ 2( _y_ ) can be learned with function _s_ ( _h, R, t_ ) = _g_ 1( **e**[(] _h[L]_[)][)] _[ ·][ g]_[2][(] **[e]** _t_[(] _[L]_[)] ). 

19 

Published as a conference paper at ICLR 2024 

- Negation: _¬φ_ 1( _x_ ). The negation of _φ_ 1( _x_ ) can be learned with function _s_ ( _h, R, t_ ) = 1 _− g_ 1( **e**[(] _h[L]_[)][)][.] 

The disjunction _∨_ can be obtained by _¬_ ( _¬φ_ 1( _x_ ) _∧¬φ_ 2( _y_ )). More complex formula involving sub-formulas from _{φ_ ( _x_ ) _}_ and _{φ[′]_ ( _y_ ) _}_ can be learned by combining the score functions above. 

## C.5 PROOF OF PROPOSITION 4.1 

**Lemma C.14.** _Assume φ_ ( _x_ ) _describes a single-connected rule structure_ G _in a KG. If assign constant to entities with out-degree large than 1 in the KG, the structure_ G _can be described with formula φ[′]_ ( _x_ ) _in CML of KG with assigned constants._ 

_Proof._ According to Theorem C.7, assume _φ[′]_ ( _x_ ) with assigned constants is not equivalent to a formula in CML, there should exist two rule structures G _,_ G _[′]_ in KG _G, G[′]_ , and entity _v_ in G and entity _v[′]_ in G _[′]_ such that Unr _[L]_ G[(] _[v]_[)] _[ ∼]_[=][ Unr] _[L]_ G _[′]_[(] _[v][′]_[)][ for every] _[ L][ ∈]_[N][ and such that][ G] _[, v][|]_[=] _[ φ][′]_[but][ G] _[′][, v][′]_[⊭] _[φ][′]_[.] 

Since each entity in G (G _[′]_ ) with out-degree larger than 1 is assigned with a constant, the rule structure G (G _[′]_ ) can be uniquely recovered from its unravelling tree Unr _[L]_ G[(] _[v]_[)][ (][Unr] _[L]_ G _[′]_[(] _[v]_[)][) for sufficient large] _L_ . Therefore, if Unr _[L]_ G[(] _[v]_[)] _[ ∼]_[=][ Unr] _[L]_ G _[′]_[(] _[v][′]_[)][ for every] _[ L][ ∈]_[N][, the corresponding rule structures][ G][ and][ G] _[′]_ should be isomorphism too, which means G _, v |_ = _φ[′]_ and G _[′] , v[′] |_ = _φ[′]_ . Thus, _φ[′]_ ( _x_ ) must be a formula in CML. 

_Proof of Proposition 4.1._ The theorem holds by restricting the unary formula to the form of _R_ (h _, x_ ) on Lemma C.14. 

_Proof of Corollary 4.2._ By converting new constants c1 _,_ c2 _, · · · ,_ c _k_ to constant predicates _Pc_ 1( _x_ ) _, Pc_ 2( _x_ ) _, · · · , Pck_ ( _x_ ), the corollary holds by using Theorem 3.2. 

## D EXPERIMENTS 

## D.1 MORE RULE STRUCTURES IN SYNTHETIC DATASETS 

In Section 6.1, we also include the following rule structures in the synthetic datasets, i.e., _C_ 4 and _I_ 2 in Figure 6, for experiments. _C_ 4 and _I_ 2 are both formulas from CML[ _G,_ h]. The proof of _C_ 4 is similar to the proof of _C_ 3 in Corollary A.2. The proof of _I_ 2 is similar to that of _I_ 1 and is in Corollary D.1. 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2024_Understanding_Expressivity_of_GNN_in_Rule_Learning_images/iclr_2024_Understanding_Expressivity_of_GNN_in_Rule_Learning.pdf-0020-12.png)


**----- Start of picture text -----**<br>
𝑅!   𝑧! 𝑅"   𝑧" 𝑅#   𝑧# 𝑅$ C 4(h , x ) :=  ∃z 1 z 2 z 3 ,R 1(h , z 1)  ∧ R 2( z 1 , z 2)<br>∧ R 3( z 2 , z 3)  ∧ R 4( z 3 , x )<br>query entity   𝑧#   𝑧#<br>tail entity 𝑅$ 𝑅$ I 2(h , x ) := ∃z 1 z 2 , R 1(h , z 1)  ∧ R 2( z 1 , z 2)<br>𝑅!   𝑧! 𝑅"   𝑧" 𝑅# ∧ ( ∃ [≥][N] z 3 , R 4( z 3 , z 2))  ∧ R 3( z 2 , x )<br>Rule structure Rule formula<br>**----- End of picture text -----**<br>


Figure 6: In the synthetic experiments, we also compare the performance of various GNNs on the synthetic datasets generated from _C_ 4 and _I_ 2. 

**Corollary D.1.** _I_ 2(h _, x_ ) _is a formula in CML_ [ _G,_ h] _._ 

20 

Published as a conference paper at ICLR 2024 

_Proof. I_ 2(h _, x_ ) is a formula in CML[ _G,_ h] as it can be recursively defined as follows 

_φ_ 1( _x_ ) = _Ph_ ( _x_ ) _, φ_ 2( _x_ ) = _∃y, R_ 1( _y, x_ ) _∧ φ_ 1( _y_ ) _, φ_ 3( _x_ ) = _∃y, R_ 2( _y, x_ ) _∧ φ_ 2( _y_ ) _, φs_ ( _x_ ) = _∃[≥]_[2] _y, R_ 4( _y, x_ ) _∧⊤, φ_ 4( _x_ ) = _φs_ ( _x_ ) _∧ φ_ 3( _x_ ) _, I_ 2(h _, x_ ) = _∃y, R_ 3( _y, x_ ) _∧ φ_ 4( _y_ ) _._ 

## D.2 EXPERIMENTS FOR COMPGCN 

The classical framework of KG reasoning is inadequate for assessing the expressivity of CompGCN because the query ( _h, R,_ ?) assumes that certain logical formula _φ_ ( _x_ ) are satisfied at the head entity _h_ by default. In order to validate the expressivity of CompGCN, it is necessary to predict all missing triplets directly based on entity representations without relying on the query ( _h, R,_ ?). To accomplish this, we create a new dataset called _S_ that adheres to the rule formula _S_ ( _x, y_ ) = _φ[⋆]_ ( _x_ ) _∧ φ[⋆]_ ( _y_ ), where the logical formula is defined as: 

## _φ[⋆]_ ( _x_ ) = _∃yR_ 1( _x, y_ ) _∧_ ( _∃xR_ 2( _y, x_ ) _∧_ ( _∃yR_ 3( _x, y_ ))) _._ 

Here, _φ[⋆]_ ( _x_ ) is represented with parameter reusing (reusing _x_ and _y_ ) and is a formula in CML. Therefore, the formula _S_ ( _x, y_ ) takes the form of _R_ ( _x, y_ ) = _fR_ ( _{φ_ ( _x_ ) _}, {φ[′]_ ( _y_ ) _}_ ) and can be learned by CompGCN, as indicated by Theorem 3.4. To validate our theorem, we generate a synthetic dataset _S_ using the same steps outlined in Section 6.1, following the rule _S_ ( _x, y_ ). We then train CompGCN on dataset _S_ . The experimental results demonstrate that CompGCN effectively learns the rule formula _S_ ( _x, y_ ) with 100% accuracy. Comparing it with QL-GNN is unnecessary since the latter is specifically designed for KG reasoning setting involving the query ( _h, R,_ ?). 

D.3 STATISTICS OF SYNTHETIC DATASETS 

Table 6: Statistics of the synthetic datasets. 

|Datasets|_C_3<br>_C_4<br>_I_1<br>_I_2<br>_T_<br>_U_<br>_S_|
|---|---|
|||
|known triplets|1514<br>2013<br>843<br>1546<br>2242<br>2840<br>320|
|||
|training|1358<br>2265<br>304<br>674<br>83<br>396<br>583|
|||
|validation|86<br>143<br>20<br>43<br>6<br>26<br>37|
|||
|testing|254<br>424<br>57<br>126<br>15<br>183<br>109|



## D.4 RESULTS ON SYNTHETIC WITH MISSING TRIPLETS 

We randomly remove 5%, 10%, and 20% edges from synthetic datasets to test the robustness of QL-GNN and EL-GNN for rule structures learning. The results of QL-GNN and EL-GNN are shown in Table 7 and 8 respectively. The results show that the completeness of rule structure correlates strongly with the performance of QL-GNN and EL-GNN. 

Table 7: The accuracy of QL-GNN on synthetic datasets with missing triplets. 

|le 7: The accuracyof|QL-GNN on synthetic datasets with missingtrip|
|---|---|
|Triplet missing ratio|_C_3<br>_C_4<br>_I_1<br>_I_2<br>_T_<br>_U_|
|||
|5%<br>10%<br>20%|0.899<br>0.866<br>0.760<br>0.783<br>0.556<br>0.329<br>0.837<br>0.718<br>0.667<br>0.685<br>0.133<br>0.279<br>0.523<br>0.465<br>0.532<br>0.468<br>0.111<br>0.162|



21 

Published as a conference paper at ICLR 2024 

Table 8: The accuracy of EL-GNN on synthetic datasets with missing triplets. 

|le 8: The accuracyof|EL-GNN on synthetic datasets with missingtrip|
|---|---|
|Triplet missing ratio|_C_3<br>_C_4<br>_I_1<br>_I_2<br>_T_<br>_U_|
|||
|5%<br>10%<br>20%|0.878<br>0.807<br>0.842<br>0.857<br>0.244<br>0.5<br>0.766<br>0.674<br>0.725<br>0.661<br>0.222<br>0.347<br>0.499<br>0.405<br>0.637<br>0.458<br>0.111<br>0.257|



D.5 MORE EXPERIMENTAL DETAILS ON REAL DATASETS 

**MRR and Hit@10** Here we supplement MRR and Hit@10 of NBFNet and EL-NBFNet on real datasets in Table 9. The improvement of EL-NBFNet on MRR and Hit@10 is not as significant as that on Accuracy because the EL-NBFNet is designed for exactly learning rule formulas and only Accuracy can be guaranteed to be improved. 

Table 9: MRR and Hit@10 of NBFNet and EL-NBFNet on real datasets. 

||Family<br>Kinship<br>UMLS<br>WN18RR<br>FB15k-237|
|---|---|
|||
||MRR<br>Hit@10<br>MRR<br>Hit@10<br>MRR<br>Hit@10<br>MRR<br>Hit@10<br>MRR<br>Hit@10|
|NBFNet<br>0.983<br>0.993<br>0.900<br>0.997<br>0.970<br>0.997<br>0.548<br>0.657<br>0.415<br>0.599<br>EL-NBFNet<br>0.990<br>0.991<br>0.905<br>0.996<br>0.975<br>0.993<br>0.562<br>0.669<br>0.424<br>0.607||



**Different hyperparameters of** _d_ We have observed that a larger or smaller _d_ does not necessarily lead to better performance in Figure 4. For real datasets, we also observed similar phenomenon in Table 10. For real datasets, we uses _d_ = 5 _,_ 30 _,_ 100 _,_ 100 _,_ 300 for Family, Kinship, UMLS, WN18RR, and FB15k-237, respectively. 

Table 10: The accuracy of EL-NBFNet on UMLS with different _d_ . 

|_d_= 0|_d_= 50|_d_= 100|_d_= 150|NBFNet|
|---|---|---|---|---|
|0.948|0.958|0.963|0.961|0.951|



**Time cost of EL-NBFNet** In Table 11, we show the time cost of EL-NBFNet and NBFNet on real datasets. The time cost is measured by seconds of testing phase. The results show that EL-NBFNet is slightly slower than NBFNet. The reason is that EL-NBFNet needs to traverse all entities on KG to assign constants to entities with out-degree larger than degree threshold _d_ . 

Table 11: Time cost (seconds of testing) of EL-NBFNet on real datasets. 

|Methods|Family|Kinship|UMLS|WN18RR|FB15k-237|
|---|---|---|---|---|---|
|EL-NBFNet|270.3|14.0|6.7|35.6|20.1|
|NBFNet|269.6|13.5|6.4|34.3|19.8|



## E THEORY OF GNNS FOR SINGLE-RELATIONAL LINK PREDICTION 

Our theory of KG reasoning can be easily extended to the single-relational link prediction. The following two corollaries are the extensions of Theorem 3.2 and Theorem 3.4 to the single-relational link prediction, respectively. 

**Corollary E.1** (Theorem 3.2 on single-relational link prediction) **.** _For single-relational link prediction, given a query_ ( _h, R,_ ?) _, a rule formula R_ (h _, x_ ) _is learned by QL-GNN if and only if R_ (h _, x_ ) _is a formula in CML_ [ _G,_ h] _._ 

22 

Published as a conference paper at ICLR 2024 

**Corollary E.2** (Theorem 3.4 on single-relational link prediction) **.** _For single-relational link prediction, CompGCN can learn the rule formula R_ ( _x, y_ ) = _fR_ ( _{φ_ ( _x_ ) _}, {φ[′]_ ( _y_ ) _}_ ) _where fR is a logical formula involving sub-formulas from {φ_ ( _x_ ) _} and {φ[′]_ ( _y_ ) _} which are the sets of formulas in CML_ [ _G_ ] _that can be learned by GNN (1)._ 

Corollary E.1 and E.2 can be directly proven by restricting the logic of KG to single-relational graph, which means there is only one binary predicate in logic of graph. 

## F UNDERSTANDING GENERALIZATION BASED ON EXPRESSIVITY 

## F.1 UNDERSTANDING EXPRESSIVITY VS. GENERALIZATION 

In this section, we provide some insights on the relation between expressivity and generalization. Expressivity in deep learning pertains to a model’s capacity to accurately represent information, whereas the ability of a model to achieve this level of expressivity depends on its generalization. Considering generalization requires not only contemplating the model design but also assessing whether the training algorithm can enable the model to achieve its expressivity. The experiments in this paper can also show this relation about expressivity and generalization from two perspective: (1) The experimental results of QL-GNN shows that its expressivity can be achieved with classical deep learning training strategies; (2) In the development of deep learning, a consensus is that more expressivity often leads to better generalization. The experimental results of EL-GNN verify this consensus. 

In addition, our theory can provide some insights on model design with better generalization. Based on the constructive proof of Lemma C.3, if QL-GNN can learn a rule formula _R_ (h _, x_ ) with _L_ recursive definition, QL-GNN can learn _R_ (h _, x_ ) with layers and hidden dimensions no less than _L_ . Assuming learning _r_ relations with QL-GNN and numbers of recursive definition for these relations are _L_ 1 _, L_ 2 _, · · · , Lr_ respectively, QL-GNN can learn these relations with layers no more than _maxiLi_ and hidden dimensions no more than[�] _Li_ . Since these bounds are nearly worst-case scenarios, both the dimensions and layers can be further optimized. Also, in the constructive proof of Lemma C.3, the aggregation function is summation, and it is difficult for mean and max/min aggregation function to capture sub-formula _∃[≥][N] y_ ( _Ri_ ( _y, x_ ) _∧ R_ (h _, y_ )). From the perspective of rule learning, QL-GNN extracts structural information at each layer. Therefore, to learn rule structures, QL-GNN needs an activation function with compression capability for information extraction from inputs. Empirically, QL-GNN with identify activation function fails to learn with rules in synthetic dataset. 

Moreover, because our theory cannot help understand generalization related to network training, the dependence to hyperparameters of network training, e.g., the number of training examples, graph size, number of entities, cannot be revealed from our theory. 

## F.2 WHY ASSIGNING LOTS OF CONSTANTS HURTS GENERALIZATION? 

We take the relation _C_ 3 as an example to show why assigning lots of constants hurts generalization from logical perspective. We add two different constants c1 and c2 to the rule formula _C_ 3( _h, x_ ), which results two different rule formulas _C_ 3 _[′]_[(][h] _[, y]_[)][=] _[∃][z]_[1] _[R]_[1][(][h] _[, z]_[1][)] _[ ∧][R]_[2][(] _[z]_[1] _[,]_[ c][1][)] _[ ∧][R]_[3][(][c][1] _[, x]_[)][and] _C_ 3 _[⋆]_[(][h] _[, y]_[)][=] _[∃][z]_[1] _[R]_[1][(][h] _[, z]_[1][)] _[ ∧][R]_[2][(] _[z]_[1] _[,]_[ c][2][)] _[ ∧][R]_[3][(][c][2] _[, x]_[)][.][Predicting][new][triplets][for][relation] _[C]_[3][can] now be achieved by learning the rule formulas _C_ 3(h _, x_ ) _, C_ 3 _[′]_[(][h] _[, x]_[)][, or] _[ C]_ 3 _[⋆]_[(][h] _[, x]_[)][.][Among these rule] formulas, _C_ 3(h _, x_ ) is the rule with the best generalization, while _C_ 3 _[′]_[(][h] _[, x]_[)][ and] _[ C]_ 3 _[⋆]_[(][h] _[, x]_[)][ require the] rule structure to pass through the entities with identifiers of constants c1 and c2, respectively. Thus, when adding constants, maintaining performance requires the network to learn both rule formulas _C_ 3 _[′]_[(][h] _[, x]_[)] _[, C]_ 3 _[⋆]_[(][h] _[, x]_[)][simultaneously][which][may][potentially][require][a][network][with][larger][capacity.] Even EL-GNN is unnecessary to learn _C_ 3 _[′]_[(][h] _[, x]_[)] _[, C]_ 3 _[⋆]_[(][h] _[, x]_[)][since] _[C]_[3][(][h] _[, x]_[)][is][learnable,][EL-GNN] cannot avoid learning rules with more than one constant in it when the rules are out of CML. 

## G LIMITATIONS AND IMPACTS 

Our work offers a fresh perspective on understanding GNN’s expressivity in KG reasoning. Unlike most existing studies that focus on distinguishing ability, we analyze GNN’s expressivity based solely on its ability to learn rule structures. Our work has the potential to inspire further studies. For 

23 

Published as a conference paper at ICLR 2024 

instance, our theory analyzes GNN’s ability to learn a single relation, but in practice, GNNs are often applied to learn multiple relations. Therefore, determining the number of relations that GNNs can effectively learn for KG reasoning remains an interesting problem that can help determine the size of GNNs. Furthermore, while our experiments are conducted on synthetic datasets without missing triplets, real datasets are incomplete (e.g., missing triplets in testing sets). Thus, understanding the expressivity of GNNs for KG reasoning on incomplete datasets remains an important challenge. 

24 

