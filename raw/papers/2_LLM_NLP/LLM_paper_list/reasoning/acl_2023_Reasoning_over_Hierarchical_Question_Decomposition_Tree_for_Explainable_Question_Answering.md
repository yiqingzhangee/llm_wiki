# **Reasoning over Hierarchical Question Decomposition Tree for Explainable Question Answering** 

# **Jiajie Zhang**[1] _[∗]_ **, Shulin Cao**[2] _[∗]_ **, Tingjian Zhang**[2] **, Xin Lv**[2] **, Jiaxin Shi**[3] **, Qi Tian**[3] **, Juanzi Li**[2] **, Lei Hou**[2] 

1Institute for Interdisciplinary Information Sciences, Tsinghua University, Beijing, China 

2Department of Computer Science and Technology, Tsinghua University, Beijing, China 

3Cloud BU, Huawei Technologies 

{jiajie-z19, caosl19}@mails.tsinghua.edu.cn 

{houlei, lijuanzi}@tsinghua.edu.cn 

## **Abstract** 

Explainable question answering (XQA) aims to answer a given question and provide an explanation why the answer is selected. Existing XQA methods focus on reasoning on a single knowledge source, e.g., structured knowledge bases, unstructured corpora, etc. However, integrating information from heterogeneous knowledge sources is essential to answer complex questions. In this paper, we propose to leverage question decomposing for heterogeneous knowledge integration, by breaking down a complex question into simpler ones, and selecting the appropriate knowledge source for each sub-question. To facilitate reasoning, we propose a novel two-stage XQA framework, _Reasoning over Hierarchical Question Decomposition Tree_ (RoHT). First, we build the _Hierarchical Question Decomposition Tree_ (HQDT) to understand the semantics of a complex question; then, we conduct probabilistic reasoning over HQDT from root to leaves recursively, to aggregate heterogeneous knowledge at different tree levels and search for a best solution considering the decomposing and answering probabilities. The experiments on complex QA datasets KQA Pro and Musique show that our framework outperforms SOTA methods significantly, demonstrating the effectiveness of leveraging question decomposing for knowledge integration and our RoHT framework. 

## **1 Introduction** 

Explainable question answering (XQA) is the task of (i) answering a question and (ii) providing an explanation that enables the user to understand why the answer is selected (Neches et al., 1985; Schuff et al., 2020). It provides a qualified way to test the reasoning ability and interpretability of intelligent systems, and plays an important role in artificial intelligence (Lu et al., 2022). 

Recent work in XQA can be grouped into two directions: 1) neuro-symbolic methods (Berant et al., 

> _∗_ Indicates equal contribution. 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/acl_2023_Reasoning_over_Hierarchical_Question_Decomposition_Tree_for_Explainable_Question_Answering_images/acl_2023_Reasoning_over_Hierarchical_Question_Decomposition_Tree_for_Explainable_Question_Answering.pdf-0001-14.png)


**----- Start of picture text -----**<br>
𝒒 [𝟎] : Which is higher, the highest mountain in North<br>America or the highest mountain in Africa?<br>𝒒 [𝟏] : How high is the highest  𝒒 [𝟐] : How high is the highest  𝒒 [𝟑] :[SelectBetween]<br>mountain in North America? mountain in Africa? [greater] #1 #2<br>𝒒 [𝟒] : Which mountain  𝒒 [𝟓] : How high is #4? 𝒒 [𝟔] : Which mountain  𝒒 [𝟕] : How high is #6?<br>is the highest in  is the highest in<br>North America? Africa?<br>**----- End of picture text -----**<br>


Figure 1: An example of Hierarchical Question Decomposition Tree (HQDT). _q[i]_ represents the index of node in its BFS ordering enumeration. 

2013; Liang et al., 2017; Cao et al., 2022b) translate natural language questions into formal representations ( _e.g._ , SPARQL (Sun et al., 2020), KoPL (Cao et al., 2022a), lambda-DCS (Liang, 2013), _etc._ ), whose execution on structured knowledge bases (KBs) gives the answer. Here, the formal representation acts as an explanation of the final answer. 2) Decompose-based models generate natural language intermediate steps that lead to the final answer (e.g., question decomposing which decomposes a complex question into sub-questions (Min et al., 2019; Perez et al., 2020; Deng et al., 2022), chain-of-thought prompting (Wei et al., 2022; Dua et al., 2022; Khot et al., 2022), etc.). Here, the intermediate steps shows the rationale of reasoning. 

Although achieving significant results, both directions have key limitations. For neuro-symbolic methods, the formal representation can only be executed on KBs. However, even the largest KBs are incomplete, thus limits the recall of model. For decompose-based methods, they employ free-text corpora as the knowledge source, and the diversity of natural language makes XQA difficult. In fact, integrating knowledge from heterogeneous sources is of great importance to QA (Wolfson et al., 2020), especially for answering complex questions. Several attempts have been made for knowledge integration (e.g., KBs, text corpora) (Sun et al., 2018, 2019; Shi et al., 2021). Although promising, these graph-based methods suffer from lacking explain- 

ability or are constrained to limited reasoning capability. 

Intuitively, leveraging question decomposing to integrate heterogeneous knowledge sources is a promising direction, since we can flexibly select the appropriate knowledge source for each subquestion. The challenges lie in: 1) How to determine the granularity of question decomposing, since certain complex questions can be directly answered with a knowledge source, and further decomposition increases the possibility of error. For example, in Figure 1, _q_[1] can be answered with the Wikipedia corpus without further decomposition. 2) How to find the optimal solution among various possible ones, since question decomposing and answering are both uncertain. For example, _q_[0] can also be decomposed as “Which mountains are in North America or Afirica”, “ _What’s the height of #1_ ”, “ _[SelectAmong] [largest] #2_ ”. 

To this end, we propose a novel two-stage XQA framework _Reasoning over Hierarchical Question Decomspotion Tree_ , dubbed RoHT. First, we propose to understand the complex question by building its _hierarchical question decomposition tree_ ( **HQDT** ). In this tree, the root node is the original complex question, and each non-root node is a subquestion of its parent. The leaf nodes are atomic questions that cannot be further decomposed. Compared with existing representations that directly decompose a question into the atomic ones, e.g., QDMR (Wolfson et al., 2020), our tree structure provides the flexibility to determine solving a question whether by directly answering or further decomposing. Second, we propose **probabilistic reasoning over HQDT** , to fuse the knowledge from KB and text at different levels of the tree, and take into consideration the probability score of both tree generation and answering. The reasoning process is recursive, from the root to leaves, and constitues three steps: 1) a scheduler determines the appropriate knowledge sources for a particular question (from KB, text, or solving its children sequentially); 2) the corresponding executors output the answers with probabilities; 3) an aggregator aggregates the candidate answers from all the knowledge sources and outputs the best ones. 

In evaluation, we instantiate our RoHT framework on two complex QA datasets: KQA Pro (Cao et al., 2022a), where we remove half of the triples in its KB and supplement it with Wikipedia corpus, and Musique (Trivedi et al., 2022), where we take 

Wikidata (Vrandecic and Krötzsch, 2014) as additional KB besides the given text paragraphs. Experimental results show that, RoHT improves the performance significantly under the KB+Text setting, by 29.7% and 45.8% EM score on KQA Pro and Musique compared with existing SOTA model. In addition, compared with the decompose-based methods, RoHT improves the SOTA by 11.3% F1 score on Musique. 

**Our contributions** include: 1) proposing to leverage question decomposing to integrate heterogeneous knowledge sources for the first time; 2) designing a novel two-stage XQA famework RoHT by first building HQDT and then reasoning over HQDT; 3) demonstrating the effectiveness of our RoHT framework through extensive experiments and careful ablation studies on two benchmark datasets. 

## **2 Related Work** 

## **2.1 QA over Text and KB** 

Over time, the QA task has evolved into two main streams: 1) QA over unstructured data (e.g., freetext corpora like Wikipedia); 2) QA over structured data (e.g., large structured KBs like DBpedia (Lehmann et al., 2015), Wikidata (Vrandecic and Krötzsch, 2014)). As structured and unstructured data are intuitively complementary information sources (Oguz et al., 2022), several attempts have been made to combines the best of both worlds. 

An early approach IBM Watson (Ferrucci, 2012) combines multiple expert systems and re-ranks them to produce the answer. (Xu et al., 2016) maps relational phrases to KB and text simultaneously, and use an integer linear program model to provide a globally optimal solution. Universal schema based method (Das et al., 2017) reasons over both KBs and text by aligning them in a common embedded space. GraftNet (Sun et al., 2018) and its successor PullNet (Sun et al., 2019) incorporate free text into graph nodes to make texts amenable to KBQA methods. TransferNet (Shi et al., 2021) proposes the relation graph to model the label-form relation from KBs and text-form relation from corpora uniformly. 

Although achieving promising results, these methods lack interpretability or are constrained to limited question type, _i.e._ , TransferNet shows interpretability with transparent step transfering, however, it can only answer multi-hop questions, 

and cannot deal with questions that require attribute comparison or value verification. In contrast, our proposed framework shows great interpretability with HQDT and cover more question types. 

## **2.2 Question Decomposing** 

For datasets, KQA Pro (Cao et al., 2022a) proposes to decompose a complex question into a multi-step program KoPL, which can be executed on KBs. BREAK (Wolfson et al., 2020) proposes to decompose questions into QDMR, which constitutes the ordered list of steps, expressed through natural language. Musique (Trivedi et al., 2022) is a QA dataset constructed by composing single-hop questions obtained from existing datasets, and thus naturally provides question decompositions. 

For models, several attempts have been made for learning to decompose with weak-supervision, such as span prediction based method (Min et al., 2019), unsupervised sequence transduction method ONUS (Perez et al., 2020), AMR-based method QDAMR (Deng et al., 2022). Another line of work is to employ large language models with in-context learning, such as Least-to-most Prompting (Zhou et al., 2022), decomposed prompting (Khot et al., 2022), successive prompting (Dua et al., 2022). 

Compared with existing works, we are the first to design a hierarchical question decomposition tree for integrating information from multiple knowledge sources. 

## **3 Definition of HQDT** 

Formally, given a complex question, its HQDT is a tree _T_ . Each node _q[i] ∈ T_ represents a question. For root node, it represents the given complex question, and for non-root nodes, it represents a sub-question of its parent node. The leaf nodes are simple ("atomic") questions that cannot be decomposed. Note that HQDT is a 3-ary ordered tree. As shown in Figure 1, we enumerate the nodes of _T_ with BFS ordering, and _q_[0] is the root question. 

A question _q[i]_ = � _w_ 1 _, · · · , wj, · · · , w|qi|_ � can be categorized into one of the three types according to the token vocabulary: 1) natural language question (e.g., _q_[4] : “ _Which mountain is the highest in North America?_ ”), here, _wj ∈V_ , and _V_ is the word vocabulary; 2) bridge question (e.g., _q_[5] : “ _How high is_ #4 _?_ ”), here, _wj ∈V ∪R_ , and _R_ is the reference token vocabulary. In this question, “#4” refers to the answer of _q_[4] , which is the sibling question of _q_[5] ; 3) symbolic operation ques- 

tion (e.g., _q_[3] : “ _[SelectBetween][greater]_ #1 #2”), here, _wj ∈V ∪R ∪O_ , and _O_ is the vocabulary of pre-defined symbolic operations, which are designed for supporting various reasoning capacity (e.g., attribute comparison and set operation) and are shown in appendix A in details. Note that all the bridge questions and symbolic operation questions are atomic questions and can only appear in leaf nodes. 

For every non-leaf question _q[i]_ , we define two ordered lists: 

- _q[i] .children_ = � _q[st][i] , · · · , q[ed][i]_[�] , which are children of _q[i]_ , successively indexed from _sti_ to _edi_ . For example, for question _q_[1] in Figure 1, _q_[1] _.children_ is � _q_[4] _, q_[5][�] . 

- _q[i] .atoms_ = � _a[i]_ 1 _[,][ · · ·][, a][i] ni_ �, which is a list of atomic questions deduced from the _ni_ leaf nodes of the sub-tree rooted by _q[i]_ , by rearranging the reference tokens. For example, for _q_[0] in Figure 1, its leaf nodes is � _q_[4] _, q_[5] _, q_[6] _, q_[7] _, q_[3][�] , and correspondingly, _q_[0] _.atoms_ is � _q_[4] _,_ ˜ _q_[5] _, q_[6] _,_ ˜ _q_[7] _,_ ˜ _q_[3][�] , with _q_ ˜[5] as “ _How high is #1?_ ”, ˜ _q_[7] as “ _How high is #3_ ”, and _q_ ˜[3] as “ _[SelectBetween][greater] #2 #4_ ”. The detailed deduction algorithm is in appendix B due to space limit. We also call _q[i] .atoms_ the _atomic representation_ of _q[i]_ . 

Specially, among _q[i] .children_ , _q[st][i] , . . . , q[ed][i][−]_[1] are all natural language questions, and _q[ed][i]_ is either a bridge question or a symbolic operation question. Answering _q[i]_ is semantically equivalent to answering sub-questions in _q[i] .children_ or in _q[i] .atoms_ sequentially. The last question in _q[i] .children_ or _q[i] .atoms_ returns the answer of _q[i]_ . 

## **4 Methodology** 

Our framework RoHT is composed of two stages: 1) Building HQDT. We understand the hierarchical compositional structure of a complex question _q_[0] by generating its HQDT _T_ with probability, where each question _q[i] ∈ T_ has a score _p[i] g_[that represents] the certainty of its generation. 

2) Probabilistic Reasoning over HQDT. We conduct recursive probabilistic reasoning over the HQDT from root to leaves to solve _q_[0] . For each question _q[i]_ , we will utilize KBs, text and its child questions together to get a list _R[i]_ , which contains answers of _q[i]_ with probabilistic scores. Finally the answer with the highest score in _R_[0] will be picked out as the final answer of _q_[0] . 

The details are introduced as follows. 

## **4.1 Building HQDT** 

To build the HQDT for a complex question, we first generate its atomic representation, which corresponds the leaf nodes of HQDT, then generate every non-leaf nodes based on this atomic representation. We compute certainty score of each node based on the likelihood of each step of generation. 

**Building Leaf Nodes** Given a complex question _q_[0] , we first use a BART (Lewis et al., 2020)-based question decomposer _Mθ_ to generate its atomic representation and output the likelihood of generation: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/acl_2023_Reasoning_over_Hierarchical_Question_Decomposition_Tree_for_Explainable_Question_Answering_images/acl_2023_Reasoning_over_Hierarchical_Question_Decomposition_Tree_for_Explainable_Question_Answering.pdf-0004-04.png)


Here, _L_[0] = _a_[0] 1 _[⟨][sep][⟩][a]_[0] 2 _[⟨][sep][⟩][. . .][ ⟨][sep][⟩][a] n_[0] 0[is the] serialization of _q_[0] _.atoms_ , where _⟨sep⟩_ is a separating token. _ld_ = Pr( _L_[0] _|q_[0] ; _θ_ ) is the likelihood of generation. Since _q_[0] is the root of _T_ , each atomic question in _q_[0] _.atoms_ corresponds to a leaf node in _T_ (with the deterministic algorithm in Appendix C), and the certainty score of each leaf node in _T_ is _ld_ . 

**Building Non-leaf Nodes** Based on _q_[0] _.atoms_ , we can generate all the non-leaf questions in HQDT. The root question is just _q_[0] and thus has certainty score _p_[0] _g_[=][1][.][For][every][other][non-leaf] question _q[i]_ , its atomic representation _q[i] .atoms_ = _⟨a[i]_ 1 _[, . . . , a] n[i] i[⟩]_[can be translated from a specific sub-] set of _q_[0] _.atoms_ by rearranging the reference tokens. The subset can be determined by considering the reference relations of a bridge or symbolic operation question _a_[0] _j[∈][q]_[0] _[.atoms]_[,][which][corre-] sponds to the leaf node _q[ed][i]_ , with other questions in _q_[0] _.atoms_ . We show the details in Appendix C. For example, _q_[2] _.atoms_ in Figure 1 is (“ _Which mountain is the highest in Africa?_ ”, “ _How high is_ #1 _?_ ”), and it can be obtained from ( _a_[0] 3 _[, a]_ 4[0][)][ in] _[ q]_[0] _[.atoms]_[.] Then we can use a BART-based question generator _Mϕ_ to generate _q[i]_ from _q[i] .atoms_ : 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/acl_2023_Reasoning_over_Hierarchical_Question_Decomposition_Tree_for_Explainable_Question_Answering_images/acl_2023_Reasoning_over_Hierarchical_Question_Decomposition_Tree_for_Explainable_Question_Answering.pdf-0004-07.png)


where _L[i]_ = _a[i]_ 1 _[⟨][sep][⟩][a][i]_ 2 _[⟨][sep][⟩][. . .][ ⟨][sep][⟩][a] n[i] i_[is the] serialized _q[i] .atoms_ , and _lg[i]_[=][Pr(] _[q][i][|][L][i]_[;] _[ ϕ]_[)][ is the] likelihood of _q[i]_ given _L[i]_ . The certainty score of _q[i]_ is computed as: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/acl_2023_Reasoning_over_Hierarchical_Question_Decomposition_Tree_for_Explainable_Question_Answering_images/acl_2023_Reasoning_over_Hierarchical_Question_Decomposition_Tree_for_Explainable_Question_Answering.pdf-0004-09.png)


**Learning of Question Decomposer and Generator** The question decomposer _Mθ_ can be trained with paired ( _q_[0] , _q_[0] _.atoms_ ) data, where the atomic representation can be from either given annotation or unsupervised construction. The question generator _Mϕ_ can also be trained with the same data by exchanging the input and output. The details are shown in Section 5.2. 

**4.2 Probabilistic Reasoning over HQDT** _f_ ( _q[i] , p[i] g[, G, C]_[)] _[ →][R][i]_[:] _[ {]_[(] _[ans][i] j[, p] j[i]_[)] _[}][,]_ (4) where _ans[i] j_[is an answer of] _[ q][i]_[, and score] _[ p][i] j_[repre-] sents the certainty of _ans[i] j_[.] 

As shown in Figure 3, the implementation of _f_ contains tree steps: 1) a scheduler determines the suitable knowledge sources for a particular question, i.e., whether the question can be answered from KB, text, or by solving its child questions sequentially; 2) according to the suitable sources output by the scheduler, executors aim to get the answers with probabilities via executing on KB (KB executor) or retrieving from text (text executor), or answering the child questions (call _f_ recursively); 3) an aggregator aggregates candidate answers from all the knowledge sources and outputs the top- _k_ answers according to their probabilities. In the following, we will introduce their details when answering _q[i]_ . 

**Scheduler** We formalize the scheduler as: 

_suit_ kb _, suit_ text _, suit_ child = Scheduler( _q[i] , G, C_ ) _,_ 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/acl_2023_Reasoning_over_Hierarchical_Question_Decomposition_Tree_for_Explainable_Question_Answering_images/acl_2023_Reasoning_over_Hierarchical_Question_Decomposition_Tree_for_Explainable_Question_Answering.pdf-0004-15.png)


Where _suit_ kb, _suit_ text and _suit_ child are 0/1 variables, respectively representing whether the answers of _q[i]_ are suitable to get from the KB _G_ , the corpus _C_ , or by solving _q[i] .children_ sequentially. 

Specifically, to check whether _G_ is suitable, the scheduler employs a semantic parser (Cao et al., 2022a) _M_ sp to parse _q[i]_ into a program _K_ with probability _p_ parse: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/acl_2023_Reasoning_over_Hierarchical_Question_Decomposition_Tree_for_Explainable_Question_Answering_images/acl_2023_Reasoning_over_Hierarchical_Question_Decomposition_Tree_for_Explainable_Question_Answering.pdf-0004-18.png)


Then it classifies the type of _q[i]_ according to the function skeleton of _K_ . For example, the function skeleton of _K_ in Figure 2 is “ _Find-RelateFilterConcept-SelectAmong_ ”. If the precision of _G_ on the questions that have the same function skeleton with _K_ is larger than a predefined threshold _γ_ 1, the scheduler will set _suit_ kb to be 1. 

1The precision of KB is calculated with questions in training set 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/acl_2023_Reasoning_over_Hierarchical_Question_Decomposition_Tree_for_Explainable_Question_Answering_images/acl_2023_Reasoning_over_Hierarchical_Question_Decomposition_Tree_for_Explainable_Question_Answering.pdf-0005-00.png)


**----- Start of picture text -----**<br>
𝒔𝒖𝒊𝒕𝐤𝐛 :  No<br>𝑲: [Find] <arg> North America <func>  KB Executor<br>[Relate] <arg> mountain range <arg> backward <func> [FilterConcept] <arg>  𝑹𝒊𝒌𝒃 :  []<br>mountain range <func> [SelectAmong]  Return<br>largest<arg> elevation above sea level <arg>  empty set<br>𝒒 [𝒊] : How high is the  𝒑𝒑𝒂𝒓𝒔𝒆: 0.96<br>highest mountain in<br>North America?𝒑𝒈 [𝒊] :  0.98 Scheduler 𝒔𝒖𝒊𝒕𝑪 in North America, with a summit elevation of  𝒆 [𝒊] : [ 𝐭𝐞𝐱𝐭 “Denali is the highest mountain peak  :  Yes Text ExecutorReturn  𝑹𝒊𝒕𝒆𝒙𝒕 :  [ (“ 20310 feet ”, 0.96) ] Aggregator 𝑹(“ [𝒊] 6190m :  [ (“ 20310 feet ”, 0.95) ] ”, 0.96),<br>Knowledge sources: 20,310 feet above sea level” , … ] empty set<br>knowledge base 𝐺,<br>text corpus 𝐶. 𝒒 [𝒔𝒕][𝒊] Scheduler … Aggregator 𝑹 [𝒔𝒕][𝒊]<br>𝒔𝒖𝒊𝒕𝐜𝐡𝐢𝐥𝐝 :  Yes 𝒒 [𝒔𝒕][𝒊][6𝟏] Scheduler … Aggregator 𝑹 [𝒔𝒕][𝒊][6𝟏]<br>… …<br>Replace reference tokens<br>𝒒 [𝒆𝒅][𝒊] Scheduler … Aggregator 𝑹(“ [𝒆𝒅] 20310 feet [𝒊] :  [ (“ 6190m ”, 0.92) ]”, 0.95),<br>else<br>else<br>if suitable<br>if suitable<br>**----- End of picture text -----**<br>


Figure 2: Illustration of the recursive reasoning function _f_ . For a question _q[i]_ , _f_ uses the scheduler to determine suitable knowledge sources and calls executors to retrieve answers from them. _f_ also recursively calls itself to get answers from the children of _q[i]_ . Finally the answers from different sources are fused by the aggregator. 

To check whether the corpus _C_ is suitable, the scheduler tries to find a set of evidence paragraphs for _q[i]_ . If _C_ is too large, the scheduler will first use BM25 (Robertson and Zaragoza, 2009) to recall dozens of most relevant paragraphs. For each paragraphs, we train a RoBERTa (Liu et al., 2019)based selector _M_ sl to classify whether it is an evidence paragraph for _q[i]_ . Suppose the set of selected evidence paragraphs, _Ce_ is not empty, the scheduler will set _suit_ text as 1. 

To make best use of knowledge from all levels, the scheduler simply set _suit_ child to be 1 if _q[i]_ is a non-leaf question otherwise 0. 

**Executors** For the KB executor, it takes the program _K_ in Equation 6 on KB _G_ to get the answers, and takes the parsing score _p_ parse in Equation 6 to calculate the probability score for each answer: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/acl_2023_Reasoning_over_Hierarchical_Question_Decomposition_Tree_for_Explainable_Question_Answering_images/acl_2023_Reasoning_over_Hierarchical_Question_Decomposition_Tree_for_Explainable_Question_Answering.pdf-0005-05.png)


For the text executor, it takes the selected paragraph set _Ce_ as described above, and employs a Transformer-based reading comprehension model _M_ rc to extract answers from _Ce_ : 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/acl_2023_Reasoning_over_Hierarchical_Question_Decomposition_Tree_for_Explainable_Question_Answering_images/acl_2023_Reasoning_over_Hierarchical_Question_Decomposition_Tree_for_Explainable_Question_Answering.pdf-0005-07.png)


where _p[i]_ ex _,j_[is the extraction probability of] _[ ans][i]_ text _,j_ given by _M_ rc. 

For solving _q[i]_ by answering its children, _f_ will recursively call itself to solve _q[st][i] , . . . , q[ed][i]_ in or- 

der: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/acl_2023_Reasoning_over_Hierarchical_Question_Decomposition_Tree_for_Explainable_Question_Answering_images/acl_2023_Reasoning_over_Hierarchical_Question_Decomposition_Tree_for_Explainable_Question_Answering.pdf-0005-11.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/acl_2023_Reasoning_over_Hierarchical_Question_Decomposition_Tree_for_Explainable_Question_Answering_images/acl_2023_Reasoning_over_Hierarchical_Question_Decomposition_Tree_for_Explainable_Question_Answering.pdf-0005-12.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/acl_2023_Reasoning_over_Hierarchical_Question_Decomposition_Tree_for_Explainable_Question_Answering_images/acl_2023_Reasoning_over_Hierarchical_Question_Decomposition_Tree_for_Explainable_Question_Answering.pdf-0005-13.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/acl_2023_Reasoning_over_Hierarchical_Question_Decomposition_Tree_for_Explainable_Question_Answering_images/acl_2023_Reasoning_over_Hierarchical_Question_Decomposition_Tree_for_Explainable_Question_Answering.pdf-0005-14.png)


Here, _f_ ref is a variant of _f_ to solve bridge and symbolic questions, which refer to the answers of their sibling questions. Suppose _q[ed][i]_ refers to the answers of its siblings _q[r]_[1] _, . . . , q[r][hi]_ in order. If _q[ed][i]_ is a bridge question, _fref_ will 1) convert _q[ed][i]_ into several possible natural language question _q_ nl[1] _[, . . . , q]_ nl _[K]_ by replacing the reference tokens with every combination (( _x[k]_ 1 _[, v]_ 1 _[k]_[)] _[, . . . ,]_[ (] _[x][k] hi[, v] h[k] i_[))] _[∈][R][r]_[1] _[× · · · ×] R[r][hi]_ , 2) call _f_ to solve each _q_ nl _[k]_[and 3) fuse the an-] swers from each _R_ nl _[k]_[and select the top-] _[k]_[ answers] with the highest scores: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/acl_2023_Reasoning_over_Hierarchical_Question_Decomposition_Tree_for_Explainable_Question_Answering_images/acl_2023_Reasoning_over_Hierarchical_Question_Decomposition_Tree_for_Explainable_Question_Answering.pdf-0005-16.png)


Note that the score of answer _ans[k]_ nl _,j_[is computed] by averaging _p[k]_ nl _,j_[and] _[ v]_ 1 _[k][, . . . , v] h[k] i_[, instead of mul-] tiplying them, to avoid exponential shrink during recursion. If _q[ed][i]_ is a symbolic operation question with operation _op_ and arguments, _f_ ref will execute simple program to apply the operation _op_ over _R[r]_[1] _, . . . , R[r][hi]_ to get _R[ed][i]_ . The score of each answer _ans[ed] j[i]_ is computed as the average of _p[ed] g[i]_ and the scores of answers in _R[r]_[1] _, . . . , R[r][hi]_ used by the program to get _ans[ed] j[i]_[.] 

**Aggregator** The aggregator fuses _R_ kb _[i]_[,] _[ R]_ text _[i]_[and] **5.2 Implementations** _R_ child _[i]_[by selecting the][ top][-] _[k]_[ answers with the high-] est scores from them. If several answers have the **KQA Pro** same surface form, only the one with the highest score will be preserved. 

**KQA Pro** For the experiments of KQA Pro, a key challenge is that there are no annotations for atomic representation, which are required for training the question decomposer and generator in RoHT. Because the KoPL program of a complex question follows context free grammar, every atomic question will correspond to a specific span of the program. Therefore we first split the KoPL program into subprograms according to the grammar, then use each sub-program to generate the atomic question by applying BART model fintuned with the (KoPL, question) pairs from the original dataset. For the answers for each atomic question, we execute the corresponding sub-programs on the KB to get corresponding answers. Using these constructed atomic representations, we train two BART-base models as the question decomposer and generator, respectively. 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/acl_2023_Reasoning_over_Hierarchical_Question_Decomposition_Tree_for_Explainable_Question_Answering_images/acl_2023_Reasoning_over_Hierarchical_Question_Decomposition_Tree_for_Explainable_Question_Answering.pdf-0006-02.png)


## **5 Experiments** 

## **5.1 Datasets** 

Currently, there are few high-quality complex QA datasets based on both KBs and text. Previous methods (Sun et al., 2018, 2019; Shi et al., 2021) evaluated their models on MetaQA (Zhang et al., 2018) by pairing its KB with the text corpus of WikiMovies (Miller et al., 2016). However, the questions in MetaQA are too simple since there are only 9 relations in its KB. Therefore, we conduct our experiments on two more challenging complex QA datasets: KQA Pro and Musique, and their details are as follows. 

For the scheduler, we directly use the semantic parser trained by (Cao et al., 2022a) on KQAPro, and set the precision threshold _γ_ to be 0.7. We train a RoBERTa-large as the evidence selector via weak supervised method: for each question in the training set and constructed atomic representations, we first use BM25 to recall 10 related paragraphs from wikipedia, then take the paragraphs that contain the answer as positive samples and take other recalled paragraphs as negative samples. For the text executor, we also train a BART-large reading comprehension model on these positive samples. 

**KQA Pro** (Cao et al., 2022a) is a large scale complex QA dataset, including 120k diverse natural language questions up to 5 hops over KB. Its KB is a subset of Wikidata (Vrandecic and Krötzsch, 2014), and consists of 16k entities, 363 predicates, 794 concepts and 890k triple facts. For each question, KQA Pro also provides the corresponding KoPL program. To simulate the realistic case where KB is incomplete, following (Sun et al., 2019; Shi et al., 2021), we randomly discard 50% triples in the KB and take Wikipedia as supplementary text corpus. 

**Musique** Since Musique provides golden atomic representation for every complex question in the training set, we directly use them to train BARTbase models as question decomposer and generator. For the scheduler, we adapt semantic parser trained by (Cao et al., 2022a) on Wikidata. The KB precision threshold _γ_ is set to be 0.4, which is determined by the top-10 types of questions with the highest precision. We train the RoBERTa selector model on complex and atomic questions in the training set together, taking annotated evidence paragraphs as positive samples and distractor paragraphs as negative samples. For the text executor, we pre-train a Longformer-large (Beltagy et al., 2020) reading comprehension model on SQUAD (Rajpurkar et al., 2016), then finetune it on complex questions and atomic questions of Musique. 

**Musique** (Trivedi et al., 2022) is a multi-hop QA dataset over text, including 25k 2-4 hop questions. We evaluate our framework under Musique-Ans setting where all the questions are answerable. Its questions are carefully constructed from several single-hop QA datasets via manually composition and paraphrase, and are hard to cheat via reasoning shortcut. For each complex question, Musique gives 20 paragraphs (including annotated evidence paragraphs and distractor paragraphs) as the corpus. Specially, for each question in the training set, Musique also provides a golden atomic representation, together with the answer and the evidence paragraph of each atomic question. In addition to the given paragraphs, we choose Wikidata as the KB to acquire additional knowledge. 

|Model|**Overall**|Multihop|Qualifer|Comparison|Logical|Count|Verify|Zero-shot|
|---|---|---|---|---|---|---|---|---|
|_50% KB_|||||||||
|KVMemNN|17.72|17.63|18.53|1.39|15.48|28.38|59.30|0.06|
|RGCN|34.77|33.71|28.44|31.46|35.39|**39.76**|64.27|0.06|
|BART KoPL|38.04|33.10|29.40|51.81|29.92|33.69|60.12|29.03|
|RoHTKB|38.94|34.16|31.54|50.91|31.61|33.69|60.4|30.52|
|_50%KB + Text_|||||||||
|TransferNet|16.80|15.94|17.93|45.35|14.84|10.47|0.00|8.43|
|RoHTmix|**46.45**|**41.76**|**41.73**|**52.21**|**41.95**|31.26|**65.45**|**38.76**|



Table 1: EM results on the dev set of KQA Pro. RoHT outperforms all the baselines by a large margin and achieves the best performance on most types of questions. 

## **5.3 Baselines** 

## **5.4 Main Results** 

## **5.4.1 Results on KQA Pro** 

we compare RoHT with several representative methods for complex QA, including memory-based methods, graph-based methods, and XQA methods. **KVMemNN** (Miller et al., 2016) stores encoded knowledge in key-value memory and iteratively reads the memory to update the query vector to conduct multi-hop reasoning. 

**RGCN** (Schlichtkrull et al., 2018) is a variant of graph convolutional network and utilizes the graph structure of KB to tackle complex questions. **BART KoPL** (Cao et al., 2022a) is a BART-based semantic parser which can convert complex question into KoPL program. It achieves over 90% accuracy on KQA Pro on the complete KB. **SA** (Trivedi et al., 2022) is a two-stage model that first uses a RoBERTa-large selector to rank and select the _K_ most relevant paragraphs with the question and then uses a Longformer-large answerer to predict answer based on selected paragraphs. **EX(SA)** (Trivedi et al., 2022) is the state-of-the-art model on Musique. It first explicitly decomposes the complex question into atomic representation and then calling SA model repeatedly to answer each atomic question in order. 

**TransferNet** (Shi et al., 2021) iteratively transfer entity scores via activated path on the relation graph that consists of both text-form relations and KB-form relations. It is existing state-of-the-art model that utilizes both KBs and text as knowledge soruces, and nearly solves MetaQA. We reimplement it on both KQA Pro and Musique, and the details are shown in Appendix D. 

**RoHT** : RoHT[KB] , RoHT[text] and RoHT[mix] denote the RoHT models that only use KB, only use text and use both KB and text, respectively. 

The experimental results for KQA Pro are shown in Table 1. When using only the incomplete KB, RoHT[KB] model respectively improves EM by 21.22, 4.17 and 0.90 compared to KVMemNN, RGCN and BART KoPL, showing the benefit of integrating the answers of sub-questions of different levels. After adding Wikipedia as supplementary text corpus, RoHT[mix] yields substantial improvement compared with RoHT[KB] (7.51 on EM), demonstrating the effectiveness of utilizing knowledge from KB and text together. RoHT[mix] also outperforms TransferNet, which is end-to-endly trained with a mixed relation graph, by a large margin (29.65 on EM). This is because unlike graphbased methods, RoHT explicitly shows the compositional structure of a complex question in natural language form via HQDT generation, and thus can retrieve answers from the KB and text with more advanced and flexible sub-modules (e.g., semantic parser and reading comprehension model). Moreover, our designed atomic operations in the HQDT also enable RoHT to solve a wide variety of complex questions: we can see that RoHT[mix] achieves the best results on 6 types of questions among 7 types, showing comprehensive reasoning capacity. 

## **5.4.2 Results on Musique** 

Table 2 presents the results on the dev set of Musique dataset. As expected, our RoHT models show significant improvement over all the baselines. With only given paragraphs, RoHT[text] improves EM/F1 by 13.8/14.3 and 11.6/11.9 compared with SA and EX(SA), respectively; With both text and KB, the performance of RoHT[mix] is also remarkably better than TransferNet (62.3 v.s. 10.9 on F1). Comparing RoHT[text] and RoHT[mix] , 

|Model|EM|F1|
|---|---|---|
|_Text_|||
|SA|39.3|47.3|
|EX(SA)|41.5|49.7|
|RoHTtext|53.1|61.6|
|_Text+KB_|||
|TransferNet|8.6|10.9|
|RoHTmix|**54.4**|**62.3**|



Table 2: EM and F1 results on the dev set of Musique. Compared with state-of-the-art methods, RoHT achieves significant improvement. 

|**Model**|**KQA Pro**|**Musique**|
|---|---|---|
|RoHTmix<br>w/o scheduler<br>RoATmix|46.5<br>40.7<br>32.3|54.4<br>47.0<br>47.6|



Table 3: EM performance of RoHT[mix] with and without scheduler, and EM performance of RoAT[mix] . 

we can also see some benefits of supplementing the text information with KB information, though the improvement is smaller than supplementing the KB with text on KQA Pro because KBs have lower coverage than text and the semantic parser is not specially finetuned for questions of Musique. We submit the predictions of RoHT[mix] on the test set and achieve 63.6 F1 score, which significantly outperforms the best public result 52.3. 

## **5.5 Further Analysis** 

## **5.5.1 Effect of Scheduler** 

To show the effect of the scheduler module, we remove it from the RoHT[mix] model, i.e, default that the KB and recalled/given text paragraphs are suitable for all questions in the HQDT, and evaluate the performance again on the dev set of KQA Pro and Musique. The results are shown in Table 3. We can see that after discarding the scheduler, the EM performance on KQA Pro and Musique drops by 5.8 and 7.4, respectively. Therefore, it is important to use the scheduler to select suitable knowledge sources for each question. 

## **5.5.2 Effect of Hierarchical Decomposition** 

Many existing methods generate non-hierarchical decomposition of complex questions, similar to the atomic representation, to assist reasoning (Min et al., 2019; Wolfson et al., 2020; Deng et al., 2022). To demonstrate the superiority of hierarchical decomposition, we compare our RoHT[mix] model with 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/acl_2023_Reasoning_over_Hierarchical_Question_Decomposition_Tree_for_Explainable_Question_Answering_images/acl_2023_Reasoning_over_Hierarchical_Question_Decomposition_Tree_for_Explainable_Question_Answering.pdf-0008-10.png)


**----- Start of picture text -----**<br>
Question:  Why did Roncalli leave the city where the painter of Venus with a Mirror died?<br>RoHT RoAT<br>𝒒 [𝟎] : Why did Roncalli leave the city where the painter of Venus with a Mirror died? Suitable sources:  𝒂𝟎𝟑: Why did Roncalli leave #2?    text<br>Suitable sources:  text, children Text ans:  [ (“ the death of Pope Pius XII ”,<br>Text ans:  [ (“ for the conclave in Rome ”, 0.91) ] 0.81)]<br>Child ans:  [ (“ for the conclave in Rome ”, 0.93),  Final ans:  [ (“ the death of Pope Pius XII ”,<br>(“ the death of Pope Pius XII ”, 0.81) ] 0.81) ]<br>Final Ans:  [ (“ for the conclave in Rome ”, 0.93),<br>(“ the death of Pope Pius XII ”, 0.81) ]<br>𝒂𝟎𝟐: Where did #1 die?<br>𝒒 [𝟏] : Where did the creator of The  𝒒 [𝟐] : Why did Roncalli leave #1? Suitable sources:  KB, text<br>Venus with a Mirror die? Suitable sources:  text KB ans:  []<br>Suitable sources:  children  KB, text,  Text ans:  XII ”, 0.81)[ , (“(“ the death of Pope Pius for the conclave in  Text ans: Final ans:  [ [ (“(“ WashingtonWashington ”, 0.95) ”, 0.95) ]]<br>KB ans:  [] Rome ”, 0.93) ]<br>Text ans:  [ (“ Venice ”, 0.88) ] Final ans:  [ (“ for the conclave in<br>Child ans:Final Ans:   [ [ (“(“ WashingtonWashington ”, 0.95) ”, 0.95), ] RomePius XII ”, 0.93)”, 0.81) , (“ the death of Pope  ] 𝒂𝟎𝟏: The Venus with a Mirror was made by whom?<br>(“ Venice ”, 0.88) ] Suitable sources:  KB, text<br>KB ans:   [ (“ Titian ”, 0.93) ]<br>𝒒 [𝟑] : The Venus with a Mirror  𝒒 [𝟒] : Where did #3 die? Text ans: Final ans:  [ [ (“(“ TitianTitian ”, 0.97)”, 0.97)  ]]<br>was made by whom? Suitable sources:  KB, text<br>Suitable sources:  KB, text KB ans:  []<br>KB ans:   [ (“ Titian ”, 0.93) ] Text ans:  [ (“ Washington ”, 0.95) ]<br>Text ans:  [ (“ Titian ”, 0.97) ] Final ans:  [ (“ Washington ”, 0.95) ] replace reference<br>Final ans:  [ (“ Titian ”, 0.97) ] return children answerstokens<br>**----- End of picture text -----**<br>


Figure 3: A case from Musique. We mark the correct answers in green and the wrong answers in red. 

RoAT[mix] model, which uses the same scheduler, executors, and aggregator as RoHT[mix] , but solves the complex question by directly answering the atomic questions in its atomic representation in order. As shown in Table 3, RoHT[mix] outperforms RoAT[mix] by a large margin on both KQA Pro and Musique. This is because the hierarchical structure of HQDT enables RoHT model to fuse the knowledge from KBs and text at different question levels, and to discard wrong answers via comparing the problisitic scores of answers. 

To further understand the reason, we show a case from Musique in Figure 3. We can see that both RoHT[mix] and RoAT[mix] fail to answer the question “ _Where did (Titian) die?_ ” ( _q_[4] in the left, _a_[0] 2[in] the right). However, RoHT[mix] directly extracts the correct answer of _q_[1] from text and finally gets the correct answer of _q_[0] with the highest score, while RoHT[mix] fails to solve _a_[0] 3[because it must rely on] the wrong answer from _a_[0] 2[.] 

## **6 Conclusion** 

In this paper, we propose RoHT, an understandingreasoning XQA framework that uses both a KB and a text corpus to derive answers of complex questions. RoHT first builds the HQDT for a complex question to understand its hierarchical compositional structure, then conducts recursive probabilistic reasoning over the HQDT to solve the question, integrating answers from the KB, text, and sub-questions. Experiments show that RoHT significantly outperforms previous methods. We also demonstrate the superiority of HQDT compared with non-hierarchical decomposition. 

## **7 Limitation** 

Currently, RoHT framework is restricted to incorporating KBs and text. However, since RoHT retrieves answers from each knowledge source in a separate way, it could in principle utilize knowledge from more heterogeneous sources such as tables, and we will study this in future work. In addition, a device with large storage space and memory is needed for the storage and usage of Wikipeida and Wikidata. 

## **8 Ethics Statement** 

The data used in this paper are drawn from publicly published datasets, encyclopedias and knowledge bases. Most of them do not involve sensitive data. 

## **References** 

- Iz Beltagy, Matthew E. Peters, and Arman Cohan. 2020. Longformer: The long-document transformer. _CoRR_ , abs/2004.05150. 

- Jonathan Berant, Andrew Chou, Roy Frostig, and Percy Liang. 2013. Semantic parsing on freebase from question-answer pairs. In _Proceedings of the 2013 Conference on Empirical Methods in Natural Language Processing, EMNLP 2013, 18-21 October 2013, Grand Hyatt Seattle, Seattle, Washington, USA, A meeting of SIGDAT, a Special Interest Group of the ACL_ , pages 1533–1544. ACL. 

- Shulin Cao, Jiaxin Shi, Liangming Pan, Lunyiu Nie, Yutong Xiang, Lei Hou, Juanzi Li, Bin He, and Hanwang Zhang. 2022a. KQA pro: A dataset with explicit compositional programs for complex question answering over knowledge base. In _Proceedings of the 60th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), ACL 2022, Dublin, Ireland, May 22-27, 2022_ , pages 6101– 6119. Association for Computational Linguistics. 

- Shulin Cao, Jiaxin Shi, Zijun Yao, Xin Lv, Jifan Yu, Lei Hou, Juanzi Li, Zhiyuan Liu, and Jinghui Xiao. 2022b. Program transfer for answering complex questions over knowledge bases. In _Proceedings of the 60th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), ACL 2022, Dublin, Ireland, May 22-27, 2022_ , pages 8128–8140. Association for Computational Linguistics. 

- Rajarshi Das, Manzil Zaheer, Siva Reddy, and Andrew McCallum. 2017. Question answering on knowledge bases and text using universal schema and memory networks. In _Proceedings of the 55th Annual Meeting of the Association for Computational Linguistics, ACL 2017, Vancouver, Canada, July 30 - August 4, Volume 2: Short Papers_ , pages 358–365. Association for Computational Linguistics. 

- Zhenyun Deng, Yonghua Zhu, Yang Chen, Michael Witbrock, and Patricia Riddle. 2022. Interpretable amrbased question decomposition for multi-hop question answering. pages 4093–4099. 

- Dheeru Dua, Shivanshu Gupta, Sameer Singh, and Matt Gardner. 2022. Successive prompting for decomposing complex questions. _CoRR_ , abs/2212.04092. 

- David A. Ferrucci. 2012. Introduction to "this is watson". _IBM J. Res. Dev._ , 56(3):1. 

- Matthew Honnibal, Ines Montani, Sofie Van Landeghem, and Adriane Boyd. 2020. spacy: Industrialstrength natural language processing in python. 

- Tushar Khot, Harsh Trivedi, Matthew Finlayson, Yao Fu, Kyle Richardson, Peter Clark, and Ashish Sabharwal. 2022. Decomposed prompting: A modular approach for solving complex tasks. _CoRR_ , abs/2210.02406. 

- Jens Lehmann, Robert Isele, Max Jakob, Anja Jentzsch, Dimitris Kontokostas, Pablo N. Mendes, Sebastian Hellmann, Mohamed Morsey, Patrick van Kleef, Sören Auer, and Christian Bizer. 2015. Dbpedia - A large-scale, multilingual knowledge base extracted from wikipedia. _Semantic Web_ , 6(2):167–195. 

- Mike Lewis, Yinhan Liu, Naman Goyal, Marjan Ghazvininejad, Abdelrahman Mohamed, Omer Levy, Veselin Stoyanov, and Luke Zettlemoyer. 2020. BART: denoising sequence-to-sequence pre-training for natural language generation, translation, and comprehension. pages 7871–7880. 

- Chen Liang, Jonathan Berant, Quoc V. Le, Kenneth D. Forbus, and Ni Lao. 2017. Neural symbolic machines: Learning semantic parsers on freebase with weak supervision. In _Proceedings of the 55th Annual Meeting of the Association for Computational Linguistics, ACL 2017, Vancouver, Canada, July 30 - August 4, Volume 1: Long Papers_ , pages 23–33. Association for Computational Linguistics. 

- Percy Liang. 2013. Lambda dependency-based compositional semantics. _CoRR_ , abs/1309.4408. 

- Yinhan Liu, Myle Ott, Naman Goyal, Jingfei Du, Mandar Joshi, Danqi Chen, Omer Levy, Mike Lewis, Luke Zettlemoyer, and Veselin Stoyanov. 2019. Roberta: A robustly optimized BERT pretraining approach. _CoRR_ , abs/1907.11692. 

- Pan Lu, Swaroop Mishra, Tony Xia, Liang Qiu, KaiWei Chang, Song-Chun Zhu, Oyvind Tafjord, Peter Clark, and Ashwin Kalyan. 2022. Learn to explain: Multimodal reasoning via thought chains for science question answering. volume abs/2209.09513. 

- Alexander H. Miller, Adam Fisch, Jesse Dodge, AmirHossein Karimi, Antoine Bordes, and Jason Weston. 2016. Key-value memory networks for directly reading documents. In _Proceedings of the 2016 Conference on Empirical Methods in Natural Language Processing, EMNLP 2016, Austin, Texas, USA, November 1-4, 2016_ , pages 1400–1409. The Association for Computational Linguistics. 

- Sewon Min, Victor Zhong, Luke Zettlemoyer, and Hannaneh Hajishirzi. 2019. Multi-hop reading comprehension through question decomposition and rescoring. pages 6097–6109. 

- Robert Neches, William R. Swartout, and Johanna D. Moore. 1985. Explainable (and maintainable) expert systems. In _Proceedings of the 9th International Joint Conference on Artificial Intelligence. Los Angeles, CA, USA, August 1985_ , pages 382–389. Morgan Kaufmann. 

- Barlas Oguz, Xilun Chen, Vladimir Karpukhin, Stan Peshterliev, Dmytro Okhonko, Michael Sejr Schlichtkrull, Sonal Gupta, Yashar Mehdad, and Scott Yih. 2022. Unik-qa: Unified representations of structured and unstructured knowledge for opendomain question answering. In _Findings of the Association for Computational Linguistics: NAACL 2022, Seattle, WA, United States, July 10-15, 2022_ , pages 1535–1546. Association for Computational Linguistics. 

- Ethan Perez, Patrick S. H. Lewis, Wen-tau Yih, Kyunghyun Cho, and Douwe Kiela. 2020. Unsupervised question decomposition for question answering. pages 8864–8880. 

- Pranav Rajpurkar, Jian Zhang, Konstantin Lopyrev, and Percy Liang. 2016. Squad: 100, 000+ questions for machine comprehension of text. pages 2383–2392. 

- Stephen E. Robertson and Hugo Zaragoza. 2009. The probabilistic relevance framework: BM25 and beyond. _Found. Trends Inf. Retr._ , 3(4):333–389. 

- Michael Sejr Schlichtkrull, Thomas N. Kipf, Peter Bloem, Rianne van den Berg, Ivan Titov, and Max Welling. 2018. Modeling relational data with graph convolutional networks. In _The Semantic Web - 15th International Conference, ESWC 2018, Heraklion, Crete, Greece, June 3-7, 2018, Proceedings_ , volume 10843 of _Lecture Notes in Computer Science_ , pages 593–607. Springer. 

- Hendrik Schuff, Heike Adel, and Ngoc Thang Vu. 2020. F1 is not enough! models and evaluation towards user-centered explainable question answering. In _Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing, EMNLP 2020, Online, November 16-20, 2020_ , pages 7076– 7095. Association for Computational Linguistics. 

- Jiaxin Shi, Shulin Cao, Lei Hou, Juanzi Li, and Hanwang Zhang. 2021. Transfernet: An effective and transparent framework for multi-hop question answering over relation graph. In _Proceedings of the 2021 Conference on Empirical Methods in Natural Language Processing, EMNLP 2021, Virtual Event / Punta Cana, Dominican Republic, 7-11 November, 2021_ , pages 4149–4158. Association for Computational Linguistics. 

- Haitian Sun, Tania Bedrax-Weiss, and William W. Cohen. 2019. Pullnet: Open domain question answering 

   - with iterative retrieval on knowledge bases and text. In _Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing and the 9th International Joint Conference on Natural Language Processing, EMNLP-IJCNLP 2019, Hong Kong, China, November 3-7, 2019_ , pages 2380–2390. Association for Computational Linguistics. 

- Haitian Sun, Bhuwan Dhingra, Manzil Zaheer, Kathryn Mazaitis, Ruslan Salakhutdinov, and William W. Cohen. 2018. Open domain question answering using early fusion of knowledge bases and text. pages 4231–4242. 

- Yawei Sun, Lingling Zhang, Gong Cheng, and Yuzhong Qu. 2020. SPARQA: skeleton-based semantic parsing for complex questions over knowledge bases. In _The Thirty-Fourth AAAI Conference on Artificial Intelligence, AAAI 2020, The Thirty-Second Innovative Applications of Artificial Intelligence Conference, IAAI 2020, The Tenth AAAI Symposium on Educational Advances in Artificial Intelligence, EAAI 2020, New York, NY, USA, February 7-12, 2020_ , pages 8952–8959. AAAI Press. 

- Harsh Trivedi, Niranjan Balasubramanian, Tushar Khot, and Ashish Sabharwal. 2022. Musique: Multihop questions via single-hop question composition. _Trans. Assoc. Comput. Linguistics_ , 10:539–554. 

- Denny Vrandecic and Markus Krötzsch. 2014. Wikidata: a free collaborative knowledgebase. _Commun. ACM_ , 57(10):78–85. 

- Jason Wei, Xuezhi Wang, Dale Schuurmans, Maarten Bosma, Ed H. Chi, Quoc Le, and Denny Zhou. 2022. Chain of thought prompting elicits reasoning in large language models. _CoRR_ , abs/2201.11903. 

- Tomer Wolfson, Mor Geva, Ankit Gupta, Yoav Goldberg, Matt Gardner, Daniel Deutch, and Jonathan Berant. 2020. Break it down: A question understanding benchmark. _Trans. Assoc. Comput. Linguistics_ , 8:183–198. 

- Kun Xu, Yansong Feng, Songfang Huang, and Dongyan Zhao. 2016. Hybrid question answering over knowledge base and free text. In _COLING 2016, 26th International Conference on Computational Linguistics, Proceedings of the Conference: Technical Papers, December 11-16, 2016, Osaka, Japan_ , pages 2397– 2407. ACL. 

- Yuyu Zhang, Hanjun Dai, Zornitsa Kozareva, Alexander J. Smola, and Le Song. 2018. Variational reasoning for question answering with knowledge graph. In _Proceedings of the Thirty-Second AAAI Conference on Artificial Intelligence, (AAAI-18), the 30th innovative Applications of Artificial Intelligence (IAAI18), and the 8th AAAI Symposium on Educational Advances in Artificial Intelligence (EAAI-18), New Orleans, Louisiana, USA, February 2-7, 2018_ , pages 6069–6076. AAAI Press. 

Denny Zhou, Nathanael Schärli, Le Hou, Jason Wei, Nathan Scales, Xuezhi Wang, Dale Schuurmans, Olivier Bousquet, Quoc Le, and Ed H. Chi. 2022. Least-to-most prompting enables complex reasoning in large language models. _CoRR_ , abs/2205.10625. 

## **A Atomic Operations** 

We design 6 atomic operations: _Verify_ , _SelectBetween_ , _SelectAmong_ , _Count_ , _Intersection_ , _Union_ , to support various reasoning capacity. We show their input, output, and examples in Table 4. 

## **B Get Atomic Representation from Leaf Nodes** 

Algorithm 1 describes that how to get the atomic representation of a question _q[i] ∈ T_ from the leaf nodes in the sub-tree rooted by _q[i]_ . 

**Algorithm 1** Get Atomic Representation from Leaf Nodes 

**Input:** An HQDT _T_ and a index _i_ . **Output:** _q[i] .atoms_ 

1: **function** DFS( _j_ , _atoms_ , _ids_ , _n_ ) 2: **if** _q[j]_ is a leaf question **then** 

3: _n ← n_ + 1 4: _ids_ [ _j_ ] _← n_ 5: _a ← q[j]_ 6: **for** _k_ in **GetRefTokens** ( _q[j]_ ) **do** 7: **if** _q[k]_ is a leaf question **then** 8: _a ←_ **ModifyRefTokens** ( _a, k, ids_ [ _k_ ]) 9: **else** 10: _a ←_ **ModifyRefTokens** ( _a, k, ids_ [ _edk_ ]) 11: _atoms._ append( _a_ ) 12: **return** 13: **for** _k ← stj, . . . , edj_ **do** 14: **Dfs** ( _k_ ) 15: 16: _q[i] .atoms ←_ [] 17: _ids ←_ empty dict 

18: **Dfs** ( _i, q[i] .atoms, ids,_ 0) 

## **C Pseudocode for Building HQDT** 

Algorithm 2 shows the pseudocode for generating the HQDT of a complex question with probability. 

## **D Reimplementation of TransferNet** 

To reimplemente TransferNet, we build the mixed relation graphs that consist of both label-form relations (i.e., KB triples) and text-form relations for KQA Pro and Musique, respectively, and train the models with the open source code. We show the details of graph building as follows. 

**KQA Pro** We follow the method used by the original paper on MetaQA to build the relation graph of KQA Pro. As mentioned in Section 5.2, we use half of its KB triples as the label form. We constructe the text form by extracting sentences from Wikipedia. Following the original paper, we use exact match of surface forms for entity recognition 

## **Algorithm 2** Generation of HQDT 

**Input:** a complex question _q_[0] , a question decomposer _Mθ_ , a question generator _Mϕ_ . 

**Output:** a list _T_ representing the HQDT, where element ( _q[i] , p[i] g[, fa] i_[)][ in] _[ T]_[denote a sub-question] _[ q][i]_[, certainty score of] _q[i]_ and the father of _q[i]_ , respectively. 

- 1: **function** REARRANGEREFTOKENS( _ar_ ) 

2: _atoms ←_ [] 3: _ids ←_ empty dict 4: _h ←_ 0 5: **for** ( _i, ai_ ) in _ar_ **do** 6: _h ← h_ + 1 7: _ids_ [ _i_ ] _← h_ 8: **for** _k_ in **GetRefTokens** ( _ai_ ) **do** 9: _ai ←_ **ModifyRefTokens** ( _ai, k, ids_ [ _k_ ]) 10: _atoms._ append( _ai_ ) 

11: **return** _atoms_ 12: 

13: ([ _a_[0] 1 _[, . . . , a]_[0] _n_ 0[]] _[, l] d_[)] _[ ←][M] θ_[(] _[q]_[0][)] 14: _n ← n_ 0 15: _T ←_ [] 16: **for** _i ←_ 1 _,_ 2 _, . . . , n_ 0 **do** 17: ( _q[i] , p[i] g_[)] _[ ←]_[(] _[a] i_[0] _[, l][d]_[)] 18: _ar[i] ←_ [( _i, ai_[0][)]] 19: **if** _a_[0] _i_[contains referring tokens] **[ then]** 20: _r_ 1 _, . . . , rh ←_ **GetRefTokens** ( _a_[0] _i_[)] 21: _n ← n_ + 1 22: _ar[n] ←_ [] 23: **for** _j ← r_ 1 _, . . . , rh, i_ **do** 24: **if** _faj_ has been identified **then** 25: _q[i] ←_ **ModifyRefTokens** ( _q[i] , j, faj_ ) 26: _j ← faj_ 27: _faj ← n_ 28: _T_ .append(( _q[j] , p[j] g[, fa] j_[))] 29: _ar[n]_ .extend( _ar[j]_ ) 30: _q[n] .atoms ←_ **RearrangeRefTokens** ( _ar[n]_ ) 31: _L[n]_ = **Serialize** ( _q[n] .atoms_ ) 32: ( _q[n] , lg[n]_[)] _[ ←][M] ϕ_[(] _[L][n]_[)] 33: _p[n] g[←][l] d[·][ l] g[n]_ 34: _T_ .append(( _q_[0] _,_ 1 _,_ 0)) _▷_ directly use _q_[0] as root 35: _T ←_ **ReIndexByBFS** ( _T_ ) 36: **return** _T_ 

and linking. For every entity in the KB, we recall all the paragraphs in Wikipedia titled by it, then take the entity as subject and other relevant entities appeared in these paragraphs as objects. The sentences that contain the objects are selected as the relation texts. The recall of answer is 51%, i.e, for 51% questions, there exist a complete path from the topic entity to the answer in the relation graph, and this is a upper bound for the performance of TransferNet. 

**Musique** For each question in Musique, we utilize the 20 given paragraphs to build individual relation graph. Specifically, we first identify entities mentioned in these paragraphs via Spacy (Honnibal et al., 2020) and exact match of surface forms with Wikidata entities. Then we take the co-occuring 

|**Operation**|**Argument**|**Input**_→_**Output**|**Example**|
|---|---|---|---|
|Verify|Value, > / < / = / !=|(Value)_→_(Bool)|[Verify] [2005] [<] #3<br>(1998)_→_(yes)|
|SelectBetween|greater / smaller|(Value, Ent) (Value, Ent)_→_(ENT)|[SelectBetween] [smaller] #3 #4<br>(6670 km,Nile River) (6440 km,Amazon River)_→_(Amazon River)|
|SelectAmong|largest / smallest|[(Value, Ent)]_→_(Ent)|[SelectAmong] [largest] #1<br>[(8848m,Everest) (8611m,K2) (8516m,Makalu)]_→_(Everest)|
|Count|/|[(Ent)]_→_(Value)|[Count] #2<br>[(BronnyJames) (Bryce James) (Zhuri James)]_→_(3)|
|Intersection|/|[(Ent)] [(Ent)]_→_[(Ent)]|[Intersection] #1 #2<br>[(apple) (orange) (peach)] [(orange)]_→_[(orange)]|
|Union|/|[(Ent)] [(Ent)]_→_[(Ent)]|[Union] #1 #2<br>[(apple) (orange)] [(orange) (peach)]_→_[(apple) (orange) (peach)]|



Table 4: Atomic operations proposed for QDKT, along with the corresponding examples. Ent, Value, Pred, and Bool denote entity, attribute, predicate, and boolean variable, respectively. () means tuple and [] means list. 

sentences of two entities as the text-form, and take the triples in Wikidata whose subject or object is one of these entities as the label-form. The recall of answer is 72%. 

