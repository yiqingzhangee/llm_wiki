# **RAGRAPH: A General Retrieval-Augmented Graph Learning Framework** 

**Xinke Jiang** _[♠∗]_ **, Rihong Qiu** _[♠∗]_ **, Yongxin Xu** _[♠∗]_ **, Wentao Zhang** _[♢]_ **, Yichen Zhu** _[♢]_ **, Ruizhe Zhang** _[♠]_ **Yuchen Fang** _[♡]_ **, Xu Chu**[♦⋆] **, Junfeng Zhao** _[♠♣†]_ , **Yasha Wang** _[♠]_[⋆] _[†]_ 


![](raw/papers/2_LLM_NLP/LLM_paper_list/rag/nips_2024_RAGraph_A_General_Retrieval-Augmented_Graph_Learning_Framework_images/nips_2024_RAGraph_A_General_Retrieval-Augmented_Graph_Learning_Framework.pdf-0001-03.png)


```
{xinkejiang,rihongqiu,xuyx,ruizhezhang}@stu.pku.edu.cn
{wentaozh2001,yichenzhu2014,fyclmiss}@gmail.com
```


![](raw/papers/2_LLM_NLP/LLM_paper_list/rag/nips_2024_RAGraph_A_General_Retrieval-Augmented_Graph_Learning_Framework_images/nips_2024_RAGraph_A_General_Retrieval-Augmented_Graph_Learning_Framework.pdf-0001-05.png)


```
{chu_xu,zhaojf,wangyasha}@pku.edu.cn
```

> _♠_ Key Laboratory of High Confidence Software Technologies (Peking University), School of Computer Science, Peking University, China 

> _♢_ No Affiliation, _♡_ University of Electronic Science and Technology of China 

♦Center on Frontiers of Computing Studies, Peking University, Beijing, China 

> _♣_ Big Data Technology Research Center, Nanhu Laboratory, Jiaxing, China 

⋆Peking University Information Technology Institute, Tianjin Binhai, China 


![](raw/papers/2_LLM_NLP/LLM_paper_list/rag/nips_2024_RAGraph_A_General_Retrieval-Augmented_Graph_Learning_Framework_images/nips_2024_RAGraph_A_General_Retrieval-Augmented_Graph_Learning_Framework.pdf-0001-12.png)


```
https://github.com/Artessay/RAGraph/
```

## **Abstract** 

Graph Neural Networks (GNNs) have become essential in interpreting relational data across various domains, yet, they often struggle to generalize to unseen graph data that differs markedly from training instances. In this paper, we introduce a novel framework called General **R** etrieval- **A** ugmented **Graph** Learning ( **RAGRAPH** ), which brings external graph data into the general graph foundation model to improve model generalization on unseen scenarios. On the top of our framework is a toy graph vector library that we established, which captures key attributes, such as features and task-specific label information. During inference, the **RAGRAPH** adeptly retrieves similar toy graphs based on key similarities in downstream tasks, integrating the retrieved data to enrich the learning context via the message-passing prompting mechanism. Our extensive experimental evaluations demonstrate that **RAGRAPH** significantly outperforms state-of-the-art graph learning methods in multiple tasks such as node classification, link prediction, and graph classification across both dynamic and static datasets. Furthermore, extensive testing confirms that **RAGRAPH** consistently maintains high performance without the need for task-specific fine-tuning, highlighting its adaptability, robustness, and broad applicability. 

## **1 Introduction** 

**Graph Neural Networks (GNNs)** [5, 50, 98, 65, 127] have recently burgeoned a surge of interest in both academic and industry communities due to their robust capability to model complex, real-world data in diverse domains, including societal [74, 57, 81], biochemical [17, 112, 108], and traffic-related [56, 24, 46, 22, 21] fields and etc [55, 38, 70, 15, 26, 25]. Utilizing a message-passing mechanism [50, 30], GNNs have transcended traditional node embedding approaches [29, 80, 96], enablingthe captureofintricaterelationships withindatathrough sophisticatedarchitecturesandadvanced graph representation learning techniques [50, 52, 56, 18, 98]. However, the challenge of generalizing GNNs across different modalities, domains [64, 63], and tasks remains largely unexplored [58, 114]. This is in stark contrast to the significant successes of large models such as GPTs [75, 76] in NLP and Sora [66] in CV, presenting a crucial frontier for further research and realms for graph data generalizing. 

> _∗_ Indicates equal contribution. 

> _†_ Yasha Wang and Junfeng Zhao are corresponding authors. 

38th Conference on Neural Information Processing Systems (NeurIPS 2024). 

In graph learning tasks, providing the necessary context is crucial for graph generalization [130, 53, 39, 135], _i.e.,_ retrieve similar shopping context as illustrated in Figure 1 (c). Therefore, our insight is to enhance the model’s generalization ability and prediction accuracy by retrieving necessary contexts during graph learning through retrieval. **Retrieval-Augmented Generation (RAG)** represents a prominent methodology, significantly augmenting language model functionalities through the integration of a dynamic retrieval mechanism during the generation process [136, 78] ( _e.g._ , a person asks what animal it is, and we use some visual [139] or text retrieval [2] methods to retrieve more descriptive features or even the wanted category). RAG enriches not only accurate and reliable content but also reduces factual errors, addressing challenges such as incorrect answers, hallucinations, and limited interpretability in knowledge-intensive tasks [42, 2, 1], obviating the need for updating model parameters and could be generalized even in unseen scenarios. 

However, how to enable retrieval-augmented generation for graph learning, _i.e._ , retrieving the user’s historical purchasing behavior to enhance recommendation ability [31, 114, 36] and identifying fraud crimes by searching for similar fraudulent relationship behaviors [86, 65], still remains unexplored and faces the following challenges _**C1& C2**_ . 

_**C1.**_ The first challenge is how to leverage the retrieved context _i.e.,_ features ( _X_ ) and labels ( _Y_ ) into the GNNs model under dynamic changing scenarios. Previous studies, such as PRODIGY [39], have adopted the concept of in-context learning (ICL) by constructing consistent and static task graphs for each specific task or dataset. These task graphs determine labels through the calculation of similarities using hidden vectors, employing a few-shot learning approach. However, PRODIGY’s reliance on a fixed set of examples as rules may not sufficiently address and generalize the variety of scenarios encountered in real-world settings, which is particularly problematic in dynamically changing environments, as the system focuses primarily on teaching the direct mapping paradigm from inputs to outputs ( _X → Y_ ), rather than truly integrate the input ( _X_ ) and output ( _Y_ ) data into the analysis. In contrast to RAG, PRODIGY struggles to incorporate external information ( _X_ and _Y_ ) related to data nodes, which is crucial for enriching the learning process in graph-based systems. 

_**C2.**_ Moreover, it is challenging to develop a tune-free prompt mechanism to support retrieved knowledge and be applicable to seamlessly switch unseen scenarios and multi-tasks. Numerous initiatives have been undertaken in the realm of graph pre-training [34, 117, 35, 82, 99, 37, 7, 89, 126], however, the challenge persists in designing a plug-and-play RAG module that can seamlessly interface with already pre-trained models. Insights derived from prior investigations into the graph prompt [9, 27, 91, 67, 114, 20, 95], the knowledge obtained by RAG can be facilitated and injected into prompt via a plug-and-play manner. 

For endeavoring to address these two challenges previously mentioned, we put forward the General **R** etrieval- **A** ugmented **Graph** Learning Framework ( **RAGRAPH** ). Drawing inspiration from the success of RAG on LLMs [136] and the ICL on GNNs [39] (we detail the difference between RAG and ICL in Appendix E), we constructed a toy graphs vector library by chunking from resource graphs, where the library key stores key information, including environmental, historical, structural, and semantic details, while 


![](raw/papers/2_LLM_NLP/LLM_paper_list/rag/nips_2024_RAGraph_A_General_Retrieval-Augmented_Graph_Learning_Framework_images/nips_2024_RAGraph_A_General_Retrieval-Augmented_Graph_Learning_Framework.pdf-0002-05.png)


**----- Start of picture text -----**<br>
Query Query Query<br>Which mammal catches mice, enjoys eating fish and has a tail? ? inpaint the picture.Please help me  What will this user buy next? ... ... ... ?<br>RAG RAG RAG<br>Cat,  a mammal with  Similar photos:  ��−2 ��−1 ��<br>a tail, primarily feeds<br>on mice. ？<br>Answer Answer Answer<br>It is a cat. Maybe a<br>T-shirt.<br>(a) NLP                        (b) CV                        (c) GNNs<br>**----- End of picture text -----**<br>


Figure 1: **(a)** RAG in NLP utilizes retrieval to enhance model responses, based on a query to retrieve related features ( _e.g., a tail, primarily feeds on mice_ ) and answers ( _e.g., Cat_ ). **(b)** In CV, RAG employs similar photo retrieval to enhance model comprehension, assisting in downstream tasks such as inpainting or image question answering. **(c)** For GNNs, RAG could leverage retrieval of similar historical subgraphs or scenarios to aid in graph-based tasks ( _e.g.,_ recommendations or fraud detection). 

node features and label information (task-specific output vector) are stored as values. For downstream tasks, the key value of the query node would be leveraged to retrieve toy graphs by the key similarities, 

2 

and the stored features (X) and labels (Y) would be aggregated structurally to provide essential knowledge to the query node, instead of the mapping paradigm, to address challenge _**C1**_ . In prompt mechanism design, we start by transferring features and task-specific output from the toy graphs to their master nodes (the central node of the toy graph) via message-passing. Subsequently, features from the master nodes and the query node’s neighbors are aggregated to the query node, along with the task-specific output from master nodes. This process could be parameter-free, indicating that our model can be applied across different tasks and datasets without the need to fine-tune for downstream tasks, effectively addressing challenge _**C2**_ . 

In summary, our contributions are listed as follows: 

- To the best of our knowledge, our proposed framework, RAGRAPH, is the first to integrate RAG with pre-trained GNNs. By constructing a key-value vector library for toy graphs, RAGRAPH facilitates explicit plug-and-play access to pre-trained GNNs, achieving commendable performance even without fine-tuning, demonstrating its superiority on cross-task and cross-dataset capabilities. 

- Our RAGRAPH employs a classic message-passing mechanism and introduces a well-designed prompt mechanism to integrate knowledge. This approach effectively incorporates the retrieved knowledge _X_ and _Y_ from toy graphs, into the pre-trained GNNs model, enhancing the accuracy and relevance of the model’s outputs. 

- We have extensively tested RAGRAPH on both static and dynamic graphs across multiple levels of graph tasks (node, edge, and graph). The results validate the effectiveness of our model, showing significant improvements over state-of-the-art baselines in both fine-tuned and tuning-free scenarios, particularly in cross-dataset validations. 

## **2 Related Work** 

## **2.1 Retrieval-Augmented Generation on Large Language Models** 

RAG integrates an external knowledge retrieval component and through prompt engineering into pretrained language models to enhance factual consistency, thus improving the reliability and interpretability of LLM responses [132, 136, 51, 23, 45, 119, 59, 111, 128]. Traditional RAG approaches utilize retriever models to source relevant documents from extensive knowledge corpora [107, 83, 71, 49], which are then processed further by reader models—primarily LLMs [77, 85]. Furthermore, several studies focus on fine-tuning reader LLMs by applying prompt-tuning with retrieved knowledge or using RAG API calls [69, 42, 2, 116, 102, 129, 62]. While RAG has seen considerable success in the NLP field, it has also been applied to tasks involving joint visual and text retrieval [139, 61, 60, 8, 125, 10], code retrieval [68, 134], audio retrieval [6, 32] and video retrieval [3, 101]. Although there have been applications of RAG on structured data such as KG-RAG for knowledge graphs [48, 45, 87, 93, 94, 40], these primarily leverage the text information of knowledge graph nodes to enhance language or graph models. In contrast, there are no significant studies utilizing RAG on structured graphs without text information to enhance pre-trained GNNs. Our work aims to extend this successful approach similarly to graph data, to enhance the capabilities of pre-trained GNNs, and can be adapted to various tasks and across different graphs without additional fine-tuning by integrating a plug-and-play RAG module. 

## **2.2 Graph Prompt Learning** 

Inspired by the application of pre-training models [75, 76] and prompt learning [103, 134, 43] in NLP, recently, learning on the graph has been divided into pre-training models on large-scale graph data [34, 117, 35, 82, 39, 131, 105, 90, 120, 122, 123, 121, 124], with or without labels, followed by fine-tuning model parameters via prompts for diverse downstream tasks [67, 114, 39, 138, 90, 95]. The adoption of prompting mechanisms in graph learning represents a promising avenue to overcome the constraints of traditional graph representation methods, striking a balance between flexibility and expressiveness [92]. For instance, VNT [95] utilizes virtual nodes as prompts to refine the application of pre-trained graph models. GraphPrompt [67] introduces a task-specific readout mechanism to tailor models for various tasks, while GraphPro [114] implements spatial- and temporal-based gating mechanisms suited for dynamic recommendation systems. Furthermore, PRODIGY [39] constructs task graphs (prompts) and data graphs to enhance the model’s ICL capabilities. Leveraging the successes in graph prompt learning, we aim to inject retrieved knowledge via prompt into pre-trained GNNs to support downstream tasks. 

3 

## **3 Preliminaries** 

In RAGRAPH, we focus on RAG on multi-level graph tasks. For consistency, we define the graphs as dynamic graphs, considering static graphs as the special cases within this framework. The subsequent definition provides a detailed description of toy graphs, including the definitions of keys and values utilized in RAGRAPH. Additionally, inspired by GraphPrompt [67], we have unified node-level, edgelevel, and graph-level tasks into a cohesive framework, and employ query graphs to tackle downstream tasks with precision. 

_**Definition 1. (Dynamic Graph)**_ Let _G_ = _{Gt}[T] t_ =1[denote a dynamic graph comprising a sequence of] graph snapshots, each represented as a static graph _Gt_ =( _Vt,Et,Xt,At,Yt_ ). _V_ =[�] _[T] t_ =1 _[V][t]_[ =] _[{][v]_[1] _[,...,v][n][}]_ defines the combined set of nodes across all snapshots and _E_ =[�] _[T] t_ =1 _[E][t][⊆V × V]_[is the edge set,] where _Vt_ and _Et_ represent the nodes and edges of the _t_ -th snapshot, respectively. Feature matrix _Xt_ = _{xv | v ∈V} ∈_ R _[n][×][d]_ contains the feature vectors for the nodes in the _t_ -th snapshot, where _d_ is the feature dimension. _At_ denotes the edge weight matrix at time _t_ , where edge weight _At_ [ _i,j_ ] _∈_ (0 _,_ 1] if _vi,vj ∈ Vt_ and ( _vi,vj_ ) _∈ Et_ , and 0 otherwise. Furthermore, _Yt_ represents the task-related labels associated with nodes, edges, or the graph at time _t_ . Note that a graph is static if _T_ = 1 and for consistency in terminology, we unify static graphs as a particular instance of dynamic graphs. 

_**Definition 2. (Toy Graph Vector Base)**_ Let _G[R]_ = _{G[R] t[}][T] t_ =1[denote a dynamic resource graph.] We chunk _G[R]_ into snapshots and take each node in _G[R]_ as the master node _vm_ of the corresponding toy graph, and then store _vm_ with its neighbors within _k_ hops as subgraphs. Data augmentation techniques [56, 133] such as node dropout, edge dropout, and random noise addition are employed on subgraphs to enhance the robustness and variability when generating each toy graph _G[T]_ (c.f. Section 4.1 for details). Each toy graph _G[T] ⊆G[R]_ is associated with a specific timestamp _τ_ and master node _vm ∈V_ , with each toy graph’s scale being considerably smaller in scale compared to their corresponding _G[R]_ . ❶ Toy graphs can be retrieved using **keys** that include the timestamp _τ_ , the hidden embedding of the master node _h[τ] m[∈]_[R] _[f]_[1][ (] _[e.g.,]_[ embedded by pre-trained GNNs in RAG][RAPH][), the environmental key] ( _e.g.,_ the neighbors set _N_ ( _vm[τ]_[)=] _[{][v] i[τ][|][A][τ]_[[] _[m,i]_[]] _[>]_[0] _[,v] i[τ][∈][G][T][ }]_[) and the structure-based position-aware] code _s[τ] m_[(cf. Appendix][ C.2][ for details).][ ❷][By retrieving based on key similarity (c.f. Section][ 4.2][ for] details), we can obtain the required **values** of _G[T]_ , _i.e._ task-specific output vector _{o[τ] i[∈]_[R] _[f]_[2] _[|][v][i][ ∈][G][T][ }]_ and hidden embeddings _{h[τ] i[∈]_[R] _[f]_[1] _[|][v][i][ ∈][G][T][ }]_[ of the master node and its neighbors, where] _[ f]_[1][ and] _[ f]_[2] represent the dimensions. Finally, we denote the **key-value** vector base for the toy graph as _G[T]_ . 

_**Definition 3. (A Unified Graph Task Definition)**_ Given a dynamic graph _G_ , it can be divided into training and testing subsets, _i.e. G_ = _G_ train _∪G_ test based on either snapshot or node set partitioning. The label _yi_ of a node _vi_ , edge ( _vi,vj_ ) or subgraph _Gi_ can be observed only if they belong to _G_ train. The objective of label prediction is to predict test labels _Y_ test _∈G_ test. Following GraphPrompt [67], we unify the three types of graph learning tasks (node-level, edge-level, and graph-level) into a single framework via similarity comparison `sim` ( _·,·_ ) of the task-specific output vector (abbreviated as _O_ , where each entry is _o_ ) with the ground-truth ( _i.e.,_ the one-hot vector or the prototype embedding under few-shot setting). It’s noted that _o_ can be either low-dimensional (with the dimension equal to the number of predicted classes) under normal settings [50, 127], or high-dimensional under few-shot settings [67] or in link prediction tasks [114, 31]. In our experiment, ❶ for node-level and graph-level tasks, the downstream tasks are given in few-shot settings following [67]: For node / graph classification on a node / graph set, let _C_ be the set of classes with _yi ∈C_ denoting the class label of node / graph. For each node / graph class, the class prototypical output vector is calculated by the mean value of the _κ_ similarity with the class prototype as:-shot set _D_ : _o_ ˜ _c_ = _κ_[1] �( _i,yi_ ) _∈D,yi_ = _c[o] y[i]_[. The class] _i_ =argmax _[ y][i] c_[ of the node or graph is determined by calculating] _∈C_ `sim` ( _oi,o_ ˜ _c_ ). ❷ For edge-level tasks, to predict a link between nodes _vi_ and _vq_ , if _∃vj,_ ( _vi,vj_ ) _∈E_ train _∈G_ train and `sim` ( _oi,oq_ ) _≥_ `sim` ( _oi,oj_ )+ _ϵ_ , we regard ( _vi,vq_ ) as linked. Following PRODIGY [39] and GraphPrompt [67], we also apply a query graph _G[Q]_ that includes the center node and its neighbors within _k_ hops. Specifically, for graph-level task, we apply a full-link virtual node as the center node inside the query graph _G[Q]_ . 

4 

## **4 RAGRAPH Framework** 

In this section, we introduce RAGRAPH, a general and novel retrieval-augmented graph learning framework that can operate on arbitrary graphs with or without additional fine-tuning, as illustrated in Figure 2. Initially, in Section 4.1, we elucidate the methodology for constructing the Resource Toy Graphs. Subsequently, in Section 4.2 we detail the Toy Graphs Retrieval Process. Finally, the Training and Inference processes are elaborated in Section 4.3, which utilize retrieved toy graphs from two propagation views—intra and inter-propagation—and handle two types of information: hidden embeddings and task-specific output vectors in two techniques (noisy trainable approach or parameter-free approach). The main notations of RAGRAPH are summarized in Table 3, Appendix A. For enhanced clarity, the Toy Graph Construction is outlined in Algorithm 1 (cf. Appendix C.5) and the Training and Inference with Toy Graphs Retrieval are detailed in Algorithm 2 (cf. Appendix C.5). Moreover, in Appendix C.4, we theoretically prove the effectiveness of applying RAG on GNNs from the perspective of mutual information gain. 


![](raw/papers/2_LLM_NLP/LLM_paper_list/rag/nips_2024_RAGraph_A_General_Retrieval-Augmented_Graph_Learning_Framework_images/nips_2024_RAGraph_A_General_Retrieval-Augmented_Graph_Learning_Framework.pdf-0005-02.png)


**----- Start of picture text -----**<br>
Toy Graphs Embedding Pipeline<br>Toy Graphs Pretrained Graph Model Resource Toy Graphs  (Key-Values) Base<br>Resource Graph<br>...<br>Chunk �� Feed<br>... Store Values �� �� �� ��<br>�� Hidden<br>Embedding (Values)<br>Environment   / History   /<br>Position-aware  /  Hidden Embedding  (Keys) Store Keys<br>Toy Graphs Retrieval Process<br>Learnable<br>Hidden Embedding<br>Frozen<br>Query Graph<br>Feed �� ... �� Propogate<br>�� Position-aware Code SearchJoint  Retrieve Toy Graphs HiddenEmbe.<br>Fetch Keys  History Information OutputVec.<br>Environment  �� ... �� Node<br>Connect<br>�� Center Node<br>Toy Graph Intra Propogate Training and Inference �� Master Node<br>Graph-Level<br>�� �� HiddenEmbe.Propogation Graph Prototype          OutputVec. 1 0 Encoder<br>�� ... / Node Prototype  Node-Level 1 Decoder<br>+ OutputVec. 0 0    0 + Weighted<br>�� Edge-Level dd<br>Output Other Node  (Learnable)<br>Vec. OutputVec. Edge Weight<br>Query-Toy-Graph Inter Propogate<br>��<br>��<br>/<br>**----- End of picture text -----**<br>


Figure 2: The overall framework of RAGRAPH. ❶ Given resource graph _G[R]_ , we chunk it and augment toy graphs _{G[T] }_ , and feed them into pre-trained GNNs to generate hidden embeddings via the encoder and task-specific output vectors via decoder, which are stored as values. Keys such as environment, history, position-aware, and hidden embeddings are stored to form the key-value database of toy graphs _G[T]_ . ❷ For a given query graph _G[Q]_ , the keys are fetched to retrieve the _topK_ toy graphs _G[T]_ topK[from] the database. ❸ Leveraging _G[T]_ topK[, intra- and inter-propagation are performed to propagate hidden] embeddings and task-specific output vectors to pass retrieved knowledge to center node _vc_ . Through a weighted fusion, the aggregated output is used to perform graph-, node- and edge-level tasks. 

## **4.1 Toy Graphs Embedding Pipeline** 

In graph-based learning, nodes with higher connectivity—typically with higher degrees—often hold more significance, meaning their information is more extensively learned during graph-pre-training processes. Conversely, less important nodes—those in the long tail—often have their features over- 

5 

looked. This issue is particularly pronounced in LLMs performing RAG, where the predominance of common knowledge overshadows the long-tail knowledge that RAG is meant to leverage. To tackle this, we construct toy graphs using an inverse importance sampling strategy, thereby countering this bias by preferentially sampling and augmenting toy graphs that accentuate the long-tail knowledge. 

**Inverse Importance Sampling Strategy.** To achieve this, we calculate each node’s importance _I_ ( _v_ ) for node _v ∈ G[R] τ_[by combining PageRank][ PR][(] _[v]_[)][ and Degree Centrality][ DC][(] _[v]_[)][ using the formula] _I_ ( _v_ )= _α_ PR( _v_ )+(1 _−α_ )DC( _v_ ), where _α ∈_ (0 _,_ 1) is the balance weight. We reverse the node importance 1 _I[′]_ ( _vi_ ) with _I[′]_ ( _v_ )= _I_ ( _v_ )+ _ϵ[,ϵ][→]_[0][, normalize it to obtain node] _[ v][i]_[ ’s sampling probabilities] _[ p][i]_[ =] � _nj_ =1 _[I][′]_[(] _[v][j]_[)][, and] perform weighted sampling function WEIGHTEDSAMPLING( _G[R] τ_[,] _[ p][i]_[) to prioritize nodes with higher] sampling probability (lower importance) according to _pi_ . When sampling, for each master node _vm_ , we generate its _k_ -hop neighbors, termed an ego net _G[e] τ_[(] _[v][m]_[)][. Given the constrained size of the resource] graph, we adopt data augmentation techniques commonly used in contrastive learning [56, 118, 117] to enhance the representativeness and diversity of the resultant toy graphs. 

**Toy Graphs Augmentation Strategy.** For augmentation, we first calculate the average reversed importance _I_[¯] _[′]_ ( _G[e] τ_[(] _[v][m]_[))][ of the nodes within an ego graph as] _[I]_[¯] _[′]_[(] _[G][e] τ_[(] _[v][m]_[))=] _|G[e] τ_ (1 _vm_ ) _|_ � _v∈G[e] τ_ ( _vm_ ) _[I][′]_[(] _[v]_[)][,] which then determines the number of augmentations _n_ aug( _G[e] τ_[(] _[v][m]_[)) =] _[ ⌊][K][ ·]_[ ¯] _[I][′]_[(] _[G][e] τ_[(] _[v][m]_[))] _[⌋]_[, where] _[ K]_ is a scaling constant that adjusts the intensity of the augmentation. For node _vi,vj ∈ G[e] τ_[(] _[v][m]_[)][, the] augmentation techniques DATAAUGMENTATION( _G[e] τ_[(] _[v][m]_[)][,] _[ n]_[aug][) employed include:] 

   - _[[e]] τ_[(] _[v][m]_[)][ has a probability of being dropped:] _[ p]_[(] _[v][i]_[being dropped][)=1] _[−][p][i]_[.] 

- ❶ **Node Dropout:** _vi ∈ G[[e]] τ_ 

- ❷ **Addition of Gaussian Noise:** we add gaussian noise to node features as augmentation _X[′]_ ( _vi_ )= _X_ ( _vi_ )+ _N_ (0 _,σ_[2] ). 

- ❸ **Node Interpolation:** a new node feature _X[′]_ ( _vnew_ ) is created by linearly combining the features of two existing nodes _vi_ and _vj_ , calculated as _X[′]_ ( _vnew_ ) = _λX_ ( _vi_ ) + (1 _− λ_ ) _X_ ( _vj_ ) _,vi,vj ∈ G[T]_ . And the edge weight between the new node _vnew_ and node _vi_ is updated to _λA_ [ _i,j_ ] and node _vj_ is (1 _−λ_ ) _A_ [ _i,j_ ] accordingly [109]. 

- ❹ **Edge Rewriting:** we alter connections based on the average of the involved nodes’ sampling probabilities, expressed as _p_ (( _vi,vj_ ) being rewired)= _[p][i]_[+] 2 _[p][j]_ . 

      - 2 . 

**Key-Value Pairs Construction.** After completing the sampling and augmentation procedures, the generated toy graphs are transformed into key-value pairs for storage [110]. Specifically, we collect each master node’s _vm_ historical information (timestamps _τ_ ), environmental information (neighbors _N_ ( _vm[τ]_[)][), structural encodings] _[ s][τ] m_[(as described in the Appendix][ C.2][), and the hidden embeddings] _[ h][τ] m_ (obtained by processing the toy graph through the frozen pre-trained GNNs) and store them as keys at the master node _vm_ of the toy graph. Additionally, we store task-specific output vectors _{o[τ] i[|][v][i][ ∈][G][T][ }]_ and hidden embeddings _{h[τ] i[|][v][i][ ∈][G][T][ }]_[ as values at each node of the toy graph.][For storage of these] key-value pairs, we utilize the FAISS vector library [14] to facilitate accelerated retrieval and storage. 

## **4.2 Toy Graphs Retrieval Process** 

After constructing the key-value toy graphs vector database, we proceed with the retrieval process for sub-tasks according to the four sub-similarities between the key values of the master node _vm_ in the toy graph and the center node _vc_ in the query graph, as detailed in Appendix C.3. The final similarity score is a weighted combination of these factors, and the _topK_ toy graphs are selected as the retrieval results: 

_S_ ( _vc,vm_ )= **w** _×_ [ _S_ time( _vc,vm_ ) _,S_ structure( _vc,vm_ ) _,S_ environment( _vc,vm_ ) _,S_ semantic( _vc,vm_ )][T] _,_ (1) 

where **w** = [ _w_ 1 _,w_ 2 _,w_ 3 _,w_ 4] are the hyper-parameterized weights attributed to the time, structure, environment, and semantic similarities, respectively. Using this composite similarity, we rank and retrieve the _topK_ toy graphs: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/rag/nips_2024_RAGraph_A_General_Retrieval-Augmented_Graph_Learning_Framework_images/nips_2024_RAGraph_A_General_Retrieval-Augmented_Graph_Learning_Framework.pdf-0006-14.png)


where _G[T]_ TopK[represents the subset of toy graphs that best match the query based on the combined] criteria. This process ensures that we retrieve the most relevant toy graphs based on a comprehensive similarity measure, incorporating historical, structural, and environmental information. 

6 

## **4.3 Training and Inference** 

In Section 4.3.1, we detail the Knowledge Injection Propagation process, which includes two distinct propagation manners. Next, in Section 4.3.2, we present our approach for combining the retrieved hidden embeddings with the task-specific output vectors. Additionally, to enhance the robustness of RAGRAPH, a noise-based prompt tuning strategy is introduced in Section 4.3.3. 

## **4.3.1 Knowledge Injection Propagation** 

After retrieving the _topK_ toy graphs _G[T]_ TopK[, knowledge, specifically the task-specific output vectors] _[ O]_ and hidden embeddings _H_ , is propagated from these toy graphs to the master nodes (Toy Graph Intra Propagation) and then to the center node _vc_ (Query-Toy Graph Inter Propagation). This propagation utilizes message-passing mechanisms via GNNs (cf. Appendix C.1). Each master node _vm_ in the toy graphs is connected to the center node _vc_ of the query graph based on the similarity scores _S_ ( _vc,vm_ ) computed in Eq.(1) and the connection weights dictate the influence of each toy graph, ensuring that graphs with higher similarity have a more substantial impact. This process can be implemented using either a parameter-free or a learnable approach. Moreover, it is worth noting that for learnable methods, the parameters of GNN are different. 

❶ **Toy Graph Intra Propagation** Within each toy graph, information **z** is propagated from neighbors to the master node using pre-trained GNNs. The task-specific output vectors _o_ and hidden embeddings _h_ from the neighbors are aggregated and transmitted to the master node. For each node _vi_ in a toy graph _G[T]_ , the GNN aggregates information from its neighbors _N_ ( _vi_ ) to update the master node _vm_ : 

**z** _m_ = GNN( _{_ **z** _i | vi ∈N_ ( _vm_ ) _}_ ) _,_ (3) 

where **z** _i_ and **z** _m_ represent the hidden embeddings _hi,hm_ or task-specific output vectors _oi,om_ of the neighbor nodes and master node, respectively. For parameter-free situations, we can prepare **z** _m_ in advance when constructing the toy graph to improve inference efficiency. 

❷ **Query-Toy Graph Inter Propagation** Next, information from the toy graphs is aggregated to the query graph. Specifically, during propagation, information **z** from the neighbors and master node of the toy graph is propagated to the center node using the same pre-trained GNNs. For a center node _vc_ in the query graph _G[Q]_ , the GNN aggregates hidden embeddings _H_ from its neighbors _N_ ( _vc_ ) and the master node _vm_ from the toy graph: 

_hc_ = GNN( _{hi | vi ∈N_ ( _vc_ ) _∪{vm}}_ ) _._ (4) 

When propagating the task-specific output vector _O_ , only the master node’s information is passed to the center node: 

_oc_ = GNN( _{oi | vi ∈{vm}}_ ) _._ (5) 

For scenarios where the propagation mechanism is learnable, attention mechanisms can be adapted on the edges. In parameter-free scenarios—where there are no learnable weights—the attention on the edges is determined based on the edge weights from the previous resource graph. 

## **4.3.2 Knowledge Fusion Layer** 

Finally, at the data fusion layer, the aggregated hidden embeddings _H_ of the center node _vc_ are processed through the pre-trained GNN’s decoder DECODER( _·_ ) to obtain an output vector _O_ . This output vector is then combined with the aggregated task-specific output vector in a weighted manner to produce the final output for downstream tasks as illustrated in Definition 3. The combined output is formulated as follows: 

ˆ _oc_ = _γoc_ +(1 _−γ_ )DECODER( _hc_ ) _,_ (6) 

where _γ_ is a reweighting hyper-parameter. The resulting vector ˆ _oc_ is then utilized to perform node-, graph-, or edge-level tasks via a similarity function. 

For the same task, the decoder can be directly used to generate outputs. For different tasks, the decoder can be masked, allowing the model to utilize pre-computed embeddings without additional training. Furthermore, the decoder can be fine-tuned to better meet the specific requirements of each task, providing both flexibility and optimized performance. This approach ensures that the model effectively integrates and leverages information from both the toy graphs and the query graph, enhancing its effectiveness in various downstream tasks through the use of the aggregated task-specific output vector. 

7 

## **4.3.3 Noise-based Graph Prompting Tuning** 

When prompt tuning, RAGRAPH employs the same prompt loss function _L_ prompt as the backbone model (e.g., GraphPro, GraphPrompt). However, to mitigate the challenge of noise retrieval—a common issue in traditional RAG where highly related but irrelevant data is often retrieved—we enhance the training process by incorporating noise data to bolster model robustness, motivated by [55]. Specifically, we implement two types of noise integration strategies: 

- ❶ **Inner-Toy-Graph Noise:** This strategy involves artificially introducing irrelevant nodes ( _vj ∈/ G[e] τ_[(] _[v][m]_[)][) into the toy graph during its construction, complementing other augmentation techniques.] 

- ❷ **Toy-Graph Noise:** Throughout the training phase, we not only retrieve the _topK_ toy graphs that are most relevant but also deliberately include the _bottomK_ toy graphs to incorporate noise knowledge. 

The integration of these noise elements is intended to enhance the model’s ability to distinguish relevant information from irrelevant information, significantly improving its robustness and overall performance in downstream tasks by noise training. However, during the inference stage, we do not incorporate the noise. 

## **5 Experiments** 

In this section, we conduct a series of experiments to evaluate the performance of RAGRAPH against state-of-the-art baselines on three dynamic and five static datasets on three-level graph tasks. Further details and experiment results are provided in Appendix D. 

## **5.1 Experimental Setup** 

**Datasets.** We use four static datasets _PROTEINS, COX2, ENZYMES_ and _BZR_ for graph classification and node classification, as well as three dynamic datasets _TAOBAO, KOUBEI_ and _AMAZON_ for link prediction. More details about these datasets can be found in Table 4 in Appendix D.1. 

**Methods and Baselines.** We consider three versions of our proposed framework RAGRAPH: 1) RAGRAPH/NF, which indicates we utilize the plug-and-plag RAGRAPH without fine-tuning on the train set; 2) RAGRAPH/FT, which employs prompt tuning on the train set with RAG; and 3) RAGRAPH/NFT, which applies noise prompt tuning on the train set with RAG. For the baseline of the dynamic graph, we choose LightGCN [31], SGL [104], MixGCF [41], SimGCL [118], GraphPro [114] and GraphPro+PRODIGY [39]. For the static graph, we choose GCN [50], GraphSAGE [30],GAT [98], GIN [106], GraphPrompt [67], GraphPrompt+PRODIGY [39] as baselines. In addition, we denote ’/NF’ and ’/FT’ respectively to represent without fine-tuning and fine-tuning. A detailed description of baselines can be referred to in Appendix D.3. 

**Settings and Evaluation.** We establish a training-resource split with the remainder of the data reserved as unseen during fine-tuning. For static graphs, the split is based on node partitioning with the ratio of 50%:30% [67], while for dynamic graphs, it is based on partitioning snapshots with the history snapshots as resource graph [39]. For fair comparisons, for methods employing PRODIGY and RAGRAPH, ❶ we fine-tune models using the training set while retrieving the resource graph to prevent information leakage and over-fitting; ❷ when testing, we retrieve the combined training and resource graphs. For other methods, fine-tuning was directly performed on the combined train and resource set for fairness. For the evaluation of static graphs, we refer GraphPrompt, utilizing pre-trained GNNs for both node- and graph-level tasks within a _k_ -shot classification framework. For dynamic graphs, we follow GraphPro to employ pre-trained GNNs on a substantial dataset fraction, with fine-tuning and testing conducted on later snapshots. Moreover, we pre-train GraphPro and GraphPrompt unsupervised on other datasets within the similar domain following [67, 39] to avoid information leakage. For classification tasks, we utilize the accuracy as evaluation matric; For link prediction tasks, we use standard metrics Recall@k and nDCG@k at _k_ =20, in line with existing methodologies [31, 104, 118]. The metrics used in the experiment are detailed in Appendix D.2 and the implementation details of RAGRAPH and baselines are in Appendix D.4. 

8 

Table 1: Accuracy evaluation on node and graph classification. All tabular results (%) are in mean _±_ standard deviation across five seeds run, with best **bolded** and runner-up underlined. 

|Methods|Node Classifcation<br>PROTEINS ENZYMES <br>(5-shot)<br>(5-shot)|Node Classifcation<br>PROTEINS ENZYMES <br>(5-shot)<br>(5-shot)|Graph Classifcation|Graph Classifcation|BZR<br>(5-shot)|
|---|---|---|---|---|---|
||||PROTEINS<br>COX2<br>(5-shot)<br>(5-shot)|ENZYMES<br>(5-shot)||
|GCN<br>GraphSAGE<br>GAT<br>GIN<br>**GraphPrompt**+<br>Vanilla/NF<br>Vanilla/FT<br>PRODIGY/NF<br>PRODIGY/FT<br>RAGRAPH/NF<br>RAGRAPH/FT<br>RAGRAPH/NFT|46.63_±_03_._04 <br>48.87_±_02_._64 <br>48.13_±_07_._90 <br>49.61_±_01_._58|52.80_±_12_._89 <br> 48.75_±_01_._59 <br> 47.75_±_01_._23 <br> 48.82_±_01_._58|54.80_±_06_._64 67.87_±_03_._39 <br>52.99_±_10_._57 67.02_±_05_._42 <br>55.82_±_07_._31 64.89_±_03_._23 <br>56.17_±_08_._58 62.77_±_02_._85|22.67_±_05_._20 <br> 21.17_±_05_._49 <br> 20.67_±_03_._27 <br> 19.00_±_03_._74|58.76_±_05_._08<br> 58.27_±_04_._79<br> 57.04_±_06_._70<br> 56.54_±_04_._20<br> 68.77_±_03_._44<br> 69.01_±_02_._21<br> 67.18_±_08_._93<br> 68.08_±_06_._68<br> **77.53**_±_05_._26<br> 76.79<br>_±_05_._02<br> 74.81_±_04_._25|
|||||||
||44.88_±_13_._17 <br>48.99_±_01_._88 <br>47.32_±_08_._12 <br>53.26_±_06_._42|48.81_±_01_._88 <br> 51.99_±_01_._36 <br> 43.80_±_14_._03 <br> 57.98_±_12_._37|56.68_±_03_._63 53.04_±_04_._13 <br>57.04_±_03_._88 64.04_±_08_._20 <br>53.48_±_06_._72 53.97_±_10_._34 <br>57.14_±_10_._34 65.31_±_04_._28|36.50_±_03_._31 <br> 40.00_±_04_._36 <br> 22.12_±_13_._84 <br> 25.94_±_05_._12||
||56.12_±_04_._11 <br>58.74<br>_±_00_._87 <br> **59.83**_±_00_._40|75.92<br>_±_01_._72 <br> 75.74_±_01_._92 <br> **76.23**_±_01_._63|58.48_±_03_._93 55.32_±_04_._15 <br>**62.33**_±_02_._52 **76.60**_±_02_._30 <br>59.08<br>_±_03_._50 71.70<br>_±_04_._29|38.17_±_03_._39 <br> 47.71<br>_±_06_._88 <br> **49.17**_±_04_._64||



## **5.2 Retrieval-Augmented Graph Results** 

As discussed, we conduct experiments and report the results of the three graph tasks for static graph and dynamic graph, as illustrated in Table 1 and Table 2. From the reported accuracy, we can find the following observations: 

**Outperforming SOTA Methods.** First, our proposed RAGRAPH outperforms almost all the baselines across the three graph tasks, demonstrating the effectiveness of RAGRAPH in transferring knowledge from the pre-training to downstream tasks compared to traditional GNNs _i.e.,_ GCN and GraphSAGE. It achieves the highest average accuracy across almost all tasks on ENZYMES, with an improvement of at least 5.19% in the static graph, and up to 11.78‰ on the dynamic graph over the best baseline PRODIGY/FT. We argue that by virtue of the integration of hidden embedding and task-specific output vector, RAGRAPH is able to comprehend more knowledge than simply learns the paradigm from _X → Y_ . Second, compared with the models of PRODIGY/NF and RAGRAPH/NF, the introduction of noise training in noise prompt tuning also improves the robustness of the model, avoiding the influence of a large amount of noise on the information aggregation inside the query graph. 

**Strong Retrieval-Augmented Performance on Unseen Datasets.** We observe that PRODIGY/NF and RAGRAPH/NF are better to Vanilla/NF, indicating that the retrieval knowledge truly works when testing on unseen datasets. Moreover, the difference between PRODIGY/NF and PRODIGY/FT is much greater than that of RAGRAPH, which also indicates that a simple learning paradigm for ICL is not enough and that RAGRAPH can achieve acceptable results even on unseen downstream datasets without the need for sophisticated fine-tuning. 


![](raw/papers/2_LLM_NLP/LLM_paper_list/rag/nips_2024_RAGraph_A_General_Retrieval-Augmented_Graph_Learning_Framework_images/nips_2024_RAGraph_A_General_Retrieval-Augmented_Graph_Learning_Framework.pdf-0009-06.png)


**----- Start of picture text -----**<br>
80 80<br>75 75<br>70 70<br>65 65<br>60 60<br>55 55<br>50 50<br>45 45<br>40 PROTEINS ENZYMES 40 PROTEINS ENZYMES<br>0 1 2 3 4 5 0 1 5 10 15 20<br>k topK<br>Accuracy (%) Accuracy (%)<br>**----- End of picture text -----**<br>


Figure 3: Hyper-parameter study with hops _k_ ( **Left** ) from 1 to 5 and _topk_ from 1 to 20 ( **Right** ) on node classification with PROTEINS, and ENZYMES datasets with the setting in Table 1. 

9 

Table 2: Performance evaluation (‰) on link prediction. 

|Method||TAOBAO<br>Recall<br>nDCG|TAOBAO<br>Recall<br>nDCG||KOUBEI<br>Recall<br>nDCG|AMAZON<br>Recall<br>nDCG|
|---|---|---|---|---|---|---|
|||Recall|||||
|LightGCN<br>SGL<br>MixGCF<br>SimGCL<br>**GraphPro**+<br>Vanilla/NF<br>Vanilla/FT<br>PRODIGY/NF<br>PRODIGY/FT<br>RAGRAPH/NF<br>RAGRAPH/FT<br>RAGRAPH/NFT|22.47_±_02_._53 <br>22.15_±_02_._20 <br>22.84_±_02_._15 <br>22.18_±_02_._22||21.89_±_02_._80 <br> 22.12_±_03_._09 <br> 23.05_±_03_._87 <br> 23.15_±_02_._75|30.21_±_06_._45 22.24_±_05_._83 15.07_±_06_._48 06.53_±_02_._66<br> 32.61_±_04_._27 22.36_±_04_._82 15.78_±_07_._12 07.90_±_02_._49<br> 32.06_±_04_._20 22.49_±_06_._91 15.24_±_08_._98 07.40_±_03_._44<br> 33.07_±_05_._28 23.08_±_05_._55 16.10_±_07_._91 07.58_±_03_._51<br> 21.31_±_04_._59 15.31_±_03_._11 12.56_±_07_._45 06.31_±_03_._92<br> 33.96<br>_±_04_._13 24.66<br>_±_02_._78 18.14_±_07_._55 08.73_±_03_._74<br> 21.66_±_03_._21 14.82_±_03_._92 11.88_±_02_._61 05.84_±_01_._84<br> 33.46_±_04_._70 23.28_±_03_._40 16.72_±_04_._28 08.09_±_02_._66<br> 22.86_±_03_._44 16.68_±_02_._48 13.78_±_05_._54 06.52_±_02_._69<br> **34.27**_±_03_._93 **24.82**_±_02_._69 **18.69**_±_07_._45 **09.09**_±_03_._89<br> 33.27_±_04_._37 24.09_±_03_._14 18.32<br>_±_06_._91 09.01<br>_±_03_._67|||
||||||||
||20.10_±_01_._50 <br>23.99_±_02_._11 <br>21.67_±_01_._42 <br>23.74_±_01_._22||20.12_±_01_._30 <br> 23.26_±_01_._42 <br> 23.15_±_03_._20 <br> 23.65_±_02_._31||||
||20.31_±_01_._60 <br>24.78<br>_±_01_._93 <br> **24.89**_±_02_._01||20.45_±_01_._44 <br> 24.35<br>_±_01_._34 <br> **24.51**_±_01_._44||||



## **5.3 Hyper-parameter Study** 

In this section, we examine the impact of various hyper-parameters on RAGRAPH. We specifically analyze the effects of varying the number of hops _k_ in toy graphs from the list [1 _,_ 2 _,_ 3 _,_ 4 _,_ 5] and the number of linked toy graphs _topK_ from the list [1 _,_ 5 _,_ 10 _,_ 15 _,_ 30 _,_ 50] to verify the sensitive: 

Figure 3 (Left) illustrates relationships between accuracy and the toy graph hop _k_ . We observe that as _k_ increases, the volume of retrieved knowledge grows exponentially. However, an excessive accumulation of knowledge not only fails to enhance accuracy but also introduces increased irrelevant noise that burdens the GNNs. Notably, accuracy shows a trend of initial improvement followed by a decline as _k_ is increased. This pattern suggests that at lower _k_ values, the retrieved information tends to consist of isolated, less useful knowledge. In contrast, at higher _k_ values, the GNNs struggle to process extensive reasoning chains, leading to the utilization of complex and abundant information that is less effective than even the baseline model’s performance. Figure 3 (Right) shows effects on accuracy with different numbers of toy graphs _topK_ . As with the previous figure, increasing _topK_ demonstrates that an excessive amount of knowledge can hinder the GNNs’ comprehension capabilities. Conversely, smaller _topK_ results in insufficient knowledge to enhance performance on downstream tasks. 

## **6 Conclusion** 

We introduced RAGRAPH, a novel and general framework that enhances Graph Neural Networks (GNNs) by integrating Retrieval-Augmented Generation (RAG) techniques. This plug-and-play approach improves GNNs’ ability to generalize to unseen data by retrieving relevant information. Experimental results show that RAGRAPH outperforms state-of-the-art methods in various graph learning tasks, demonstrating its adaptability and robustness. While RAGRAPH is currently limited to retrieving subgraphs, future research could explore using more graph-structured data such as nodes, edges, and trees to further enhance its capabilities. In general, our work provides valuable insights and serves as a reference for future Large Graph Models. 

## **Acknowledgments** 

This work is supported by the National Natural Science Foundation of China (No.U23A20468). 

10 

## **References** 

- [1] A. Asai, S. Min, Z. Zhong, and D. Chen. Retrieval-based language models and applications. In _Proceedings of the 61st Annual Meeting of the Association for Computational Linguistics (Volume 6: Tutorial Abstracts)_ , pages 41–46, 2023. 

- [2] A. Asai, Z. Wu, Y. Wang, A. Sil, and H. Hajishirzi. Self-rag: Learning to retrieve, generate, and critique through self-reflection. In _ICLR_ , 2024. 

- [3] S.-V. Bogolin, I. Croitoru, H. Jin, Y. Liu, and S. Albanie. Cross modal retrieval with querybank normalisation. In _CVPR_ , 2022. 

- [4] K. M. Borgwardt, C. S. Ong, S. Schönauer, S. Vishwanathan, A. J. Smola, and H.-P. Kriegel. Protein function prediction via graph kernels. _Bioinformatics_ , 21(suppl_1):i47–i56, 2005. 

- [5] H. Cai, V. W. Zheng, and K. C.-C. Chang. A comprehensive survey of graph embedding: Problems, techniques, and applications. _IEEE TKDE_ , 2018. 

- [6] D. M. Chan, S. Ghosh, A. Rastrow, and B. Hoffmeister. Using external off-policy speech-to-text mappings in contextual end-to-end automated speech recognition, 2023. 

- [7] M. Chen, W. Zhang, W. Zhang, Q. Chen, and H. Chen. Meta relational learning for few-shot link prediction in knowledge graphs. _arXiv preprint arXiv:1909.01515_ , 2019. 

- [8] W. Chen, H. Hu, X. Chen, P. Verga, and W. W. Cohen. Murag: Multimodal retrieval-augmented generator for open question answering over images and text. In _EMNLP_ , 2022. 

- [9] X. Chen, S. Zhang, Y. Xiong, X. Wu, J. Zhang, X. Sun, Y. Zhang, Y. Zhao, and Y. Kang. Prompt learning on temporal interaction graphs. _arXiv:2402.06326_ , 2024. 

- [10] Z. Cheng, J. Zhang, X. Xu, G. Trajcevski, T. Zhong, and F. Zhou. Retrieval-augmented hypergraph for multimodal social media popularity prediction. In _Proceedings of the 30th ACM SIGKDD Conference on Knowledge Discovery and Data Mining_ , KDD ’24, page 445–455, New York, NY, USA, 2024. Association for Computing Machinery. 

- [11] X. Chu, Y. Jin, X. Wang, S. Zhang, Y. Wang, W. Zhu, and H. Mei. Wasserstein barycenter matching for graph size generalization of message passing neural networks. In _Proceedings of the 40th International Conference on Machine Learning_ , ICML’23. JMLR.org, 2023. 

- [12] F. Cuconasu, G. Trappolini, F. Siciliano, S. Filice, C. Campagnano, Y. Maarek, N. Tonellotto, and F. Silvestri. The power of noise: Redefining retrieval for rag systems. In _Proceedings of the 47th International ACM SIGIR Conference on Research and Development in Information Retrieval_ , volume 17 of _SIGIR 2024_ , page 719–729. ACM, July 2024. 

- [13] H. Cui, Z. Lu, P. Li, and C. Yang. On positional and structural node features for graph neural networks on non-attributed graphs, 2021. 

- [14] M. Douze, A. Guzhva, C. Deng, J. Johnson, G. Szilvasy, P.-E. Mazaré, M. Lomeli, L. Hosseini, and H. Jégou. The faiss library, 2024. 

- [15] Y. Duan, G. Zhang, S. Wang, X. Peng, Z. Wang, J. Mao, H. Wu, X. Jiang, and K. Wang. Catgnn: Enhancing credit card fraud detection via causal temporal graph neural networks. _ArXiv_ , abs/2402.14708, 2024. 

- [16] Y. Duan, J. Zhao, pengcheng, J. Mao, H. Wu, J. Xu, S. Wang, C. Ma, K. Wang, K. Wang, and X. Li. Causal deciphering and inpainting in spatio-temporal dynamics via diffusion model. 2024. 

- [17] D. K. Duvenaud, D. Maclaurin, J. Iparraguirre, R. Bombarell, T. Hirzel, A. Aspuru-Guzik, and R. P. Adams. Convolutional networks on graphs for learning molecular fingerprints. In _NeurIPS_ , 2015. 

- [18] P. L. Eli Chien, Jianhao Peng and O. Milenkovic. Adaptive universal generalized pagerank graph neural network. In _ICLR_ , 2021. 

11 

- [19] F. Fang, Y. Bai, S. Ni, M. Yang, X. Chen, and R. Xu. Enhancing noise robustness of retrievalaugmented language models with adaptive adversarial training, 2024. 

- [20] T. Fang, Y. Zhang, Y. Yang, C. Wang, and L. Chen. Universal prompt tuning for graph neural networks. In _NeurIPS_ , 2024. 

- [21] Y. Fang, Y. Liang, B. Hui, Z. Shao, L. Deng, X. Liu, X. Jiang, and K. Zheng. Efficient large-scale traffic forecasting with transformers: A spatial data management perspective. In _Inproceedings of SIGKDD_ , 2025. 

- [22] Y. Fang, Y. Qin, H. Luo, F. Zhao, B. Xu, L. Zeng, and C. Wang. When spatio-temporal meet wavelets: Disentangled traffic forecasting via efficient spectral graph attention networks. In _ICDE_ , 2023. 

- [23] L. Gao, X. Ma, J. Lin, and J. Callan. Precise zero-shot dense retrieval without relevance labels, 2022. 

- [24] X. Gao, H. Chen, and J. Haworth. A spatiotemporal analysis of the impact of lockdown and coronavirus on london’s bicycle hire scheme: from response to recovery to a new normal. _GIS_ , 2023. 

- [25] X. Gao, J. Haworth, D. Zhuang, H. Chen, and X. Jiang. Uncertainty quantification in the roadlevel traffic risk prediction by spatial-temporal zero-inflated negative binomial graph neural network (stzinb-gnn). _GIScience 2023_ , 2023. 

- [26] X. Gao, X. Jiang, J. Haworth, D. Zhuang, S. Wang, H. Chen, and S. Law. Uncertainty-aware probabilistic graph neural networks for road-level traffic crash prediction. _Accident Analysis & Prevention_ , 208:107801, 2024. 

- [27] Z. Gao, X. Sun, Z. Liu, Y. Li, H. Cheng, and J. Li. Protein multimer structure prediction via PPIguided prompt learning. In _The Twelfth International Conference on Learning Representations (ICLR)_ , 2024. 

- [28] M. Gjoka, M. Kurant, C. T. Butts, and A. Markopoulou. Walking in facebook: A case study of unbiased sampling of osns. In _2010 Proceedings IEEE INFOCOM_ , pages 1–9, 2010. 

- [29] A. Grover and J. Leskovec. node2vec: Scalable feature learning for networks. In _SIGKDD_ , 2016. 

- [30] W. Hamilton, Z. Ying, and J. Leskovec. Inductive representation learning on large graphs. In _NeurIPS_ , 2017. 

- [31] X. He, K. Deng, X. Wang, Y. Li, Y. Zhang, and M. Wang. Lightgcn: Simplifying and powering graph convolution network for recommendation. In _SIGIR_ , 2020. 

- [32] Z. He, W. Hao, W.-T. Lu, C. Chen, K. Lerman, and X. Song. Alcap: Alignment-augmented music captioner. In _ICASSP_ , 2023. 

- [33] E. J. Hu, Y. Shen, P. Wallis, Z. Allen-Zhu, Y. Li, S. Wang, L. Wang, and W. Chen. Lora: Low-rank adaptation of large language models, 2021. 

- [34] W. Hu, B. Liu, J. Gomes, M. Zitnik, P. Liang, V. Pande, and J. Leskovec. Strategies for pre-training graph neural networks. _arXiv preprint arXiv:1905.12265_ , 2019. 

- [35] Z. Hu, Y. Dong, K. Wang, K.-W. Chang, and Y. Sun. Gpt-gnn: Generative pre-training of graph neural networks. In _Proceedings of the 26th ACM SIGKDD International Conference on Knowledge Discovery & Data Mining_ , pages 1857–1867, 2020. 

- [36] J. Huang, G. Cai, J. Zhu, Z. Dong, R. Tang, W. Zhang, and Y. Yu. Recall-augmented ranking: Enhancing click-through rate prediction accuracy with cross-stage data. In _WWW_ , 2024. 

- [37] K. Huang and M. Zitnik. Graph meta learning via local subgraphs, 2020. 

- [38] M. Huang, Y. Liu, X. Ao, K. Li, J. Chi, J. Feng, H. Yang, and Q. He. Auc-oriented graph neural network for fraud detection. In _WWW_ , 2022. 

12 

- [39] Q. Huang, H. Ren, P. Chen, G. Kržmanc, D. Zeng, P. Liang, and J. Leskovec. Prodigy: Enabling in-context learning over graphs, 2023. 

- [40] Q. Huang, H. Ren, and J. Leskovec. Few-shot relational reasoning via connection subgraph pretraining. In _NeurIPS_ , 2022. 

- [41] T. Huang, Y. Dong, M. Ding, Z. Yang, W. Feng, X. Wang, and J. Tang. Mixgcf: An improved training method for graph neural network-based recommender systems. In _Proceedings of the 27th ACM SIGKDD Conference on Knowledge Discovery & Data Mining_ , pages 665–674, 2021. 

- [42] G. Izacard, P. Lewis, M. Lomeli, L. Hosseini, F. Petroni, T. Schick, J. Dwivedi-Yu, A. Joulin, S. Riedel, and E. Grave. Atlas: Few-shot learning with retrieval augmented language models, 2022. 

- [43] X. Jiang, Y. Fang, R. Qiu, H. Zhang, Y. Xu, H. Chen, W. Zhang, R. Zhang, Y. Fang, X. Chu, J. Zhao, and Y. Wang. Tc-rag:turing-complete rag’s case study on medical llm systems. _ArXiv_ , abs/2408.09199, 2024. 

- [44] X. Jiang, Z. Qin, J. Xu, and X. Ao. Incomplete graph learning via attribute-structure decoupled variational auto-encoder. In _Proceedings of the 17th ACM International Conference on Web Search and Data Mining_ , WSDM ’24, page 304–312, New York, NY, USA, 2024. Association for Computing Machinery. 

- [45] X. Jiang, R. Zhang, Y. Xu, R. Qiu, Y. Fang, Z. Wang, J. Tang, H. Ding, X. Chu, J. Zhao, and Y. Wang. Hykge: A hypothesis knowledge graph enhanced framework for accurate and reliable medical llms responses, 2024. 

- [46] X. Jiang, D. Zhuang, X. Zhang, H. Chen, J. Luo, and X. Gao. Uncertainty quantification via spatial-temporal tweedie model for zero-inflated and long-tail travel demand prediction. In _CIKM_ , 2023. 

- [47] R. Y. Jiaxuan You and J. Leskovec. Position-aware graph neural networks. In _ICML_ , 2019. 

- [48] B. Jin, Y. Zhang, Q. Zhu, and J. Han. Heterformer: Transformer-based deep node representation learning on heterogeneous text-rich networks. In _SIGKDD_ , pages 1020–1031, 2023. 

- [49] J. Kim, S. M. Jaehyun Nam, J. Park, S.-W. Lee, M. Seo, J.-W. Ha, and J. Shin. Sure: Summarizing retrievals using answer candidates for open-domain qa of llms. In _ICLR_ , 2024. 

- [50] T. N. Kipf and M. Welling. Semi-supervised classification with graph convolutional networks. In _ICLR_ , 2016. 

- [51] P. Lewis, E. Perez, A. Piktus, F. Petroni, V. Karpukhin, N. Goyal, H. Küttler, M. Lewis, W. tau Yih, T. Rocktäschel, S. Riedel, and D. Kiela. Retrieval-augmented generation for knowledgeintensive nlp tasks, 2021. 

- [52] G. Li, M. Müller, A. Thabet, and B. Ghanem. Deepgcns: Can gcns go as deep as cnns? In _ICCV_ , 2019. 

- [53] H. Li, X. Wang, Z. Zhang, and W. Zhu. Out-of-distribution generalization on graphs: A survey, 2022. 

- [54] K. Li, Y. Chen, Y. Liu, J. Wang, Q. He, M. Cheng, and X. Ao. Boosting the adversarial robustness of graph neural networks: An ood perspective. In _International Conference on Learning Representations_ , 2024. 

- [55] K. Li, Y. Liu, X. Ao, J. Chi, J. Feng, H. Yang, and Q. He. Reliable representations make a stronger defender: Unsupervised structure refinement for robust gnn. In _SIGKDD_ , 2022. 

- [56] R. Li, T. Zhong, X. Jiang, G. Trajcevski, J. Wu, and F. Zhou. Mining spatio-temporal relations via self-paced graph contrastive learning. In _SIGKDD_ , 2022. 

- [57] S. Li, R. Xie, Y. Zhu, X. Ao, F. Zhuang, and Q. He. User-centric conversational recommendation with multi-aspect user modeling. In _SIGIR_ , 2022. 

13 

- [58] X. Li, D. Lian, Z. Lu, J. Bai, Z. Chen, and X. Wang. Graphadapter: Tuning vision-language models with dual knowledge graph. In _NeurIPS_ , volume 36, 2024. 

- [59] X. Li, R. Zhao, Y. K. Chia, B. Ding, S. Joty, S. Poria, and L. Bing. Chain-of-knowledge: Grounding large language models via dynamic knowledge adapting over heterogeneous sources, 2023. 

- [60] W. Lin and B. Byrne. Retrieval augmented visual question answering with outside knowledge. In _EMNLP_ , 2022. 

- [61] W. Lin, J. Mei, J. Chen, and B. Byrne. Preflmr: Scaling up fine-grained late-interaction multi-modal retrievers, 2024. 

- [62] X. V. Lin, X. Chen, M. Chen, W. Shi, M. Lomeli, R. James, P. Rodriguez, J. Kahn, G. Szilvasy, M. Lewis, L. Zettlemoyer, and S. Yih. Ra-dit: Retrieval-augmented dual instruction tuning. In _ICLR_ , 2024. 

- [63] Y. Liu, X. Ao, F. Feng, and Q. He. Ud-gnn: Uncertainty-aware debiased training on semihomophilous graphs. In _Proceedings of the 28th ACM SIGKDD Conference on Knowledge Discovery and Data Mining_ , pages 1131–1140, 2022. 

- [64] Y. Liu, X. Ao, F. Feng, Y. Ma, K. Li, T.-S. Chua, and Q. He. Flood: A flexible invariant learning framework for out-of-distribution generalization on graphs. In _Proceedings of the 29th ACM SIGKDD Conference on Knowledge Discovery and Data Mining_ , pages 1548–1558, 2023. 

- [65] Y. Liu, X. Ao, Z. Qin, J. Chi, J. Feng, H. Yang, and Q. He. Pick and choose: a gnn-based imbalanced learning approach for fraud detection. In _Proceedings of the Web Conference 2021_ , pages 3168–3177, 2021. 

- [66] Y. Liu, K. Zhang, Y. Li, Z. Yan, C. Gao, R. Chen, Z. Yuan, Y. Huang, H. Sun, J. Gao, L. He, and L. Sun. Sora: A review on background, technology, limitations, and opportunities of large vision models, 2024. 

- [67] Z. Liu, X. Yu, Y. Fang, and X. Zhang. Graphprompt: Unifying pre-training and downstream tasks for graph neural networks. In _WWW_ , 2023. 

- [68] S. Lu, N. Duan, H. Han, D. Guo, S. won Hwang, and A. Svyatkovskiy. Reacc: A retrievalaugmented code completion framework. In _ACL_ , 2022. 

- [69] H. Luo, Y.-S. Chuang, Y. Gong, T. Zhang, Y. Kim, X. Wu, D. Fox, H. Meng, and J. Glass. Sail: Search-augmented instruction learning, 2023. 

- [70] J. Luo, W. Zhang, Y. Fang, X. Gao, D. Zhuang, H. Chen, and X. Jiang. Timeseries suppliers allocation risk optimization via deep black litterman model. _ArXiv_ , abs/2401.17350, 2024. 

- [71] K. Ma, H. Cheng, Y. Zhang, X. Liu, E. Nyberg, and J. Gao. Chain-of-skills: A configurable model for open-domain question answering, 2023. 

- [72] H. Mao, X. Chen, Q. Fu, L. Du, S. Han, and D. Zhang. Neuron campaign for initialization guided by information bottleneck theory. In _CIKM_ , 2021. 

- [73] S. Maskey, R. Levie, Y. Lee, and G. Kutyniok. Generalization analysis of message passing neural networks on large random graphs. In S. Koyejo, S. Mohamed, A. Agarwal, D. Belgrave, K. Cho, and A. Oh, editors, _Advances in Neural Information Processing Systems_ , volume 35, pages 4805–4817. Curran Associates, Inc., 2022. 

- [74] S. Matsugu, Y. Fujiwara, and H. Shiokawa. Uncovering the largest community in social networks at scale. In _IJCAI_ , 2023. 

- [75] OpenAI. Introducing chatgpt. `https://openai.com/blog/chatgpt` , 2022. 

- [76] OpenAI. Gpt-4 technical report. _ArXiv_ , abs/2303.08774, 2023. 

- [77] S. Pan, L. Luo, Y. Wang, C. Chen, J. Wang, and X. Wu. Unifying large language models and knowledge graphs: A roadmap. _IEEE Transactions on Knowledge and Data Engineering_ , 2024. 

14 

- [78] S. Parashar, Z. Lin, T. Liu, X. Dong, Y. Li, D. Ramanan, J. Caverlee, and S. Kong. The neglected tails of vision-language models. In _CVPR_ , 2024. 

- [79] A.Paszke, S.Gross, F.Massa, A.Lerer, J.Bradbury, G.Chanan, T.Killeen, Z.Lin, N.Gimelshein, L. Antiga, et al. Pytorch: An imperative style, high-performance deep learning library. In _NeurIPS_ , 2019. 

- [80] B. Perozzi, R. Al-Rfou, and S. Skiena. Deepwalk: Online learning of social representations. In _SIGKDD_ , 2014. 

- [81] Y. Qin, Y. Fang, H. Luo, F. Zhao, and C. Wang. Next point-of-interest recommendation with auto-correlation enhanced multi-modal transformer network. In _SIGIR_ , 2022. 

- [82] J. Qiu, Q. Chen, Y. Dong, J. Zhang, H. Yang, M. Ding, K. Wang, and J. Tang. Gcc: Graph contrastive coding for graph neural network pre-training. In _Proceedings of the 26th ACM SIGKDD international conference on knowledge discovery & data mining_ , pages 1150–1160, 2020. 

- [83] Y. Qu, Y. Ding, J. Liu, K. Liu, R. Ren, W. X. Zhao, D. Dong, H. Wu, and H. Wang. Rocketqa: An optimized training approach to dense passage retrieval for open-domain question answering, 2021. 

- [84] R. A. Rossi and N. K. Ahmed. The network data repository with interactive graph analytics and visualization. In _AAAI Conference on Artificial Intelligence_ , pages 4292–4293, 2015. 

- [85] P. Sarthi, S. Abdullah, A. Tuli, S. Khanna, A. Goldie, and C. D. Manning. Raptor: Recursive abstractive processing for tree-organized retrieval. In _ICLR_ , 2024. 

- [86] S. Sharma and R. Sharma. Identifying possible rumor spreaders on twitter: A weak supervised learning approach. In _2021 International Joint Conference on Neural Networks (IJCNN)_ , pages 1–8, 2021. 

- [87] K. Soman, P. W. Rose, J. H. Morris, R. E. Akbas, B. Smith, B. Peetoom, C. Villouta-Reyes, G. Cerono, Y. Shi, A. Rizk-Jackson, S. Israni, C. A. Nelson, S. Huang, and S. E. Baranzini. Biomedical knowledge graph-enhanced prompt generation for large language models, 2023. 

- [88] X. Su and T. M. Khoshgoftaar. A survey of collaborative filtering techniques. _Adv. in Artif. Intell._ , 2009, Jan. 2009. 

- [89] J. Sun, Y. Zhou, and C. Zong. One-shot relation learning for knowledge graphs via neighborhood aggregation and paths encoding. _Transactions on Asian and Low-Resource Language Information Processing_ , 21(3):1–19, 2021. 

- [90] M. Sun, K. Zhou, X. He, Y. Wang, and X. Wang. Gppt: Graph pre-training and prompt tuning to generalize graph neural networks. In _SIGKDD_ , 2022. 

- [91] X. Sun, H. Cheng, J. Li, B. Liu, and J. Guan. All in one: Multi-task prompting for graph neural networks. In _Proceedings of the 26th ACM SIGKDD international conference on knowledge discovery & data mining (KDD’23)_ , page 2120–2131, 2023. 

- [92] X.Sun, J.Zhang, X.Wu, H.Cheng, Y.Xiong, andJ.Li. Graphpromptlearning: Acomprehensive survey and beyond. _arXiv:2311.16534_ , 2023. 

- [93] Y. Tan, H. Lv, X. Huang, J. Zhang, S. Wang, and C. Yang. Musegraph: Graph-oriented instruction tuning of large language models for generic graph mining, 2024. 

- [94] Y. Tan, Z. Zhou, H. Lv, W. Liu, and C. Yang. Walklm: A uniform language model fine-tuning framework for attributed graph embedding. In A. Oh, T. Naumann, A. Globerson, K. Saenko, M. Hardt, and S. Levine, editors, _NeurIPS_ , volume 36, pages 13308–13325. Curran Associates, Inc., 2023. 

- [95] Z. Tan, R. Guo, K. Ding, and H. Liu. Virtual node tuning for few-shot node classification. In _SIGKDD_ , 2023. 

15 

- [96] J. Tang, M. Qu, M. Wang, M. Zhang, J. Yan, and Q. Mei. Line: Large-scale information network embedding. In _WWW_ , 2015. 

- [97] B. Teji, S. Roy, D. S. Dhami, D. Bhandari, and P. H. Guzzi. Graph embedding techniques for predicting missing links in biological networks: An empirical evaluation. _IEEE Transactions on Emerging Topics in Computing_ , 12(1):190–201, 2024. 

- [98] P. Veliˇckovi´c, G. Cucurull, A. Casanova, A. Romero, P. Liò, and Y. Bengio. Graph attention networks. In _ICLR_ , 2018. 

- [99] S. Wang, K. Ding, C. Zhang, C. Chen, and J. Li. Task-adaptive few-shot node classification. _Proceedings of the 28th ACM SIGKDD Conference on Knowledge Discovery and Data Mining_ , 2022. 

- [100] S. Wang, Y. Dong, X. Huang, C. Chen, and J. Li. FAITH: Few-shot graph classification with hierarchical task graphs. In _International Joint Conference on Artificial Intelligence_ , 2022. 

- [101] X. Wang, L. Zhu, and Y. Yang. T2vlad: Global-local sequence alignment for text-video retrieval. In _CVPR_ , 2021. 

- [102] Y. Wang, R. Ren, J. Li, W. X. Zhao, J. Liu, and J. Wen. Rear: A relevance aware retrieval augmented framework for open-domain question answering, 2024. 

- [103] J. Wei, X. Wang, D. Schuurmans, M. Bosma, B. Ichter, F. Xia, E. Chi, Q. Le, and D. Zhou. Chain-of-thought prompting elicits reasoning in large language models, 2023. 

- [104] J. Wu, X. Wang, F. Feng, X. He, L. Chen, J. Lian, and X. Xie. Self-supervised graph learning for recommendation. In _Proceedings of the 44th international ACM SIGIR conference on research and development in information retrieval_ , pages 726–735, 2021. 

- [105] L. Xia, B. Kao, and C. Huang. Opengraph: Towards open graph foundation models, 2024. 

- [106] K. Xu, W. Hu, J. Leskovec, and S. Jegelka. How powerful are graph neural networks? In _ICLR_ , 2019. 

- [107] P. Xu, W. Ping, X. Wu, L. McAfee, C. Zhu, Z. Liu, S. Subramanian, E. Bakhturina, M. Shoeybi, and B. Catanzaro. Retrieval meets long context large language models. In _ICLR_ , 2024. 

- [108] Y. Xu, X. Chu, K. Yang, Z. Wang, P. Zou, H. Ding, J. Zhao, Y. Wang, and B. Xie. Seqcare: Sequential training with external medical knowledge graph for diagnosis prediction in healthcare data. In _Proceedings of the ACM Web Conference 2023_ , pages 2819–2830, 2023. 

- [109] Y. Xu, X. Jiang, X. Chu, Y. Xiao, C. Zhang, H. Ding, J. Zhao, Y. Wang, and B. Xie. Protomix: Augmenting health status representation learning via prototype-based mixup. In _Knowledge Discovery and Data Mining_ , 2024. 

- [110] Y. Xu, K. Yang, C. Zhang, P. Zou, Z. Wang, H. Ding, J. Zhao, Y. Wang, and B. Xie. Vecocare: Visit sequences-clinical notes joint learning for diagnosis prediction in healthcare data. In _IJCAI_ , volume 23, pages 4921–4929, 2023. 

- [111] Y. Xu*, R. Zhang*, X. Jiang*, Y. Feng, Y. Xiao, X. Ma, R. Zhu, X. Chu, J. Zhao, and Y. Wang. Parenting: Optimizing knowledge selection of retrieval-augmented language models with parameter decoupling and tailored tuning. _arXiv preprint arXiv:2410.10360_ , 2024. 

- [112] K. Yang, Y. Xu, P. Zou, H. Ding, J. Zhao, Y. Wang, and B. Xie. Kerprint: local-global knowledge graph enhanced diagnosis prediction for retrospective and prospective interpretations. In _Proceedings of the AAAI Conference on Artificial Intelligence_ , volume 37, pages 5357–5365, 2023. 

- [113] S. Yang, X. Jiang, H. Zhao, W. Zeng, H. Liu, and Y. Jia. Faima: Feature-aware in-context learning for multi-domain aspect-based sentiment analysis. In _COLING_ , 2024. 

- [114] Y. Yang, L. Xia, D. Luo, K. Lin, and C. Huang. Graphpro: Graph pre-training and prompt learning for recommendation. In _WWW_ , 2024. 

16 

- [115] J. Yoo, N. Ahn, and K.-A. Sohn. Rethinking data augmentation for image super-resolution: A comprehensive analysis and a new strategy. In _2020 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)_ , pages 8372–8381, 2020. 

- [116] O. Yoran, T. Wolfson, O. Ram, and J. Berant. Making retrieval-augmented language models robust to irrelevant context. In _ICLR_ , 2024. 

- [117] Y. You, T. Chen, Y. Sui, T. Chen, Z. Wang, and Y. Shen. Graph contrastive learning with augmentations. _Advances in neural information processing systems_ , 33:5812–5823, 2020. 

- [118] J. Yu, H. Yin, X. Xia, T. Chen, L. Cui, and Q. V. H. Nguyen. Are graph augmentations necessary? simple graph contrastive learning for recommendation. In _Proceedings of the 45th international ACM SIGIR conference on research and development in information retrieval_ , pages 1294–1303, 2022. 

- [119] W. Yu, H. Zhang, X. Pan, K. Ma, H. Wang, and D. Yu. Chain-of-note: Enhancing robustness in retrieval-augmented language models, 2023. 

- [120] X. Yu, Y. Fang, Z. Liu, Y. Wu, Z. Wen, J. Bo, X. Zhang, and S. C. Hoi. Few-shot learning on graphs: from meta-learning to pre-training and prompting. _arXiv preprint arXiv:2402.01440_ , 2024. 

- [121] X. Yu, Z. Liu, Y. Fang, Z. Liu, S. Chen, and X. Zhang. Generalized graph prompt: Toward a unification of pre-training and downstream tasks on graphs. _IEEE Transactions on Knowledge and Data Engineering_ , 2024. 

- [122] X. Yu, Z. Liu, Y. Fang, and X. Zhang. Dygprompt: Learning feature and time prompts on dynamic graphs. _arXiv preprint arXiv:2405.13937_ , 2024. 

- [123] X. Yu, J. Zhang, Y. Fang, and R. Jiang. Non-homophilic graph pre-training and prompt learning. _arXiv preprint arXiv:2408.12594_ , 2024. 

- [124] X. Yu, C. Zhou, Y. Fang, and X. Zhang. Multigprompt for multi-task pre-training and prompting on graphs. In _WWW_ , 2024. 

- [125] Z. Yuan, Q. Jin, C. Tan, Z. Zhao, H. Yuan, F. Huang, and S. Huang. Ramm: Retrieval-augmented biomedical visual question answering with multi-modal pre-training. In _CVPR_ , 2023. 

- [126] C. Zhang, H. Yao, C. Huang, M. Jiang, Z. Li, and N. V. Chawla. Few-shot knowledge graph completion. In _AAAI_ , 2020. 

- [127] R. Zhang, X. Jiang, Y. Fang, J. Luo, Y. Xu, Y. Zhu, X. Chu, J. Zhao, and Y. Wang. Infinitehorizon graph filters: Leveraging power series to enhance sparse information aggregation. _ArXiv_ , abs/2401.09943, 2024. 

- [128] R. Zhang, Y. Xu, Y. Xiao, R. Zhu, X. Jiang, X. Chu, J. Zhao, and Y. Wang. Kapo: Knowledgeaware preference optimization for controllable knowledge selection in retrieval-augmented language models. _arXiv preprint arXiv:2408.03297_ , 2024. 

- [129] Y. Zhang, Z. Chen, Y. Fang, L. Cheng, Y. Lu, F. Li, W. Zhang, and H. Chen. Knowledgeable preference alignment for llms in domain-specific question answering, 2023. 

- [130] Z. Zhang, H. Li, Z. Zhang, Y. Qin, X. Wang, and W. Zhu. Graph meets llms: Towards large graph models, 2023. 

- [131] H. Zhao, A. Chen, X. Sun, H. Cheng, and J. Li. All in one and one for all: A simple yet effective method towards cross-domain graph pretraining, 2024. 

- [132] R. Zhao, H. Chen, W. Wang, F. Jiao, X. L. Do, C. Qin, B. Ding, X. Guo, M. Li, X. Li, and S. Joty. Retrieving multimodal information for augmented generation: A survey, 2023. 

- [133] J. Zhou, C. Xie, Z. Wen, X. Zhao, and Q. Xuan. Data augmentation on graphs: A technical survey, 2023. 

17 

- [134] S. Zhou, U. Alon, F. F. Xu, Z. Wang, Z. Jiang, and G. Neubig. Docprompting: Generating code by retrieving the docs. In _ICLR_ , 2023. 

- [135] X. Zhou, R. Lumbantoruan, Y. Ren, L. Chen, X. Yang, and J. Shao. Dynamic bi-layer graph learning for context-aware sequential recommendation. _ACM Trans. Recomm. Syst._ , 2(2), apr 2024. 

- [136] F. Zhu, W. Lei, C. Wang, J. Zheng, S. Poria, and T.-S. Chua. Retrieving and reading: A comprehensive survey on open-domain question answering, 2021. 

- [137] K. Zhu, X. Feng, X. Du, Y. Gu, W. Yu, H. Wang, Q. Chen, Z. Chu, J. Chen, and B. Qin. An information bottleneck perspective for effective noise filtering on retrieval-augmented generation. In L.-W. Ku, A. Martins, and V. Srikumar, editors, _Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)_ , pages 1044–1069, Bangkok, Thailand, Aug. 2024. Association for Computational Linguistics. 

- [138] Y. Zhu, J. Guo, and S. Tang. Sgl-pt: A strong graph learner with graph prompt tuning, 2023. 

- [139] Y. Zhu, Z. Ou, X. Mou, and J. Tang. Retrieval-augmented embodied agents. In _CVPR_ , 2024. 

18 

## **A Notations** 

The notations in this paper are summarized in Table 3. 

Table 3: Notations Tables in RAGRAPH 

|Notation|Defnition|
|---|---|
|_G_/_V_/_E_<br>_GR_<br>_GQ_<br>_GT_<br>_GT_<br>TopK<br>_GT_<br>_Ge_<br>_τ_(_vm_)<br>_Xt_/_At_/_Yt_<br>_C_<br>_N_(_v_)<br>_H_/_O_<br>_D_<br>_t/τ_<br>_n_<br>_d_<br>_vi_<br>_vm/vc_<br>_f_1_/f_2<br>_ϵ_<br>_k_<br>_I_(_v_)<br>PR(_·_)_/_DC(_·_)<br>_pi_<br>_K_<br>_n_aug(_GT_ )<br>_S_(_vc,vm_)<br>_l_<br>_α_<br>_λ_<br>_wi_<br>_γ_|The dynamic graph / node / edge set with_Gt_/_Vt_/_Et_as its entry<br>The resource graph with_GR_<br>_t_ as its entry<br>The query graph with_vc_as its center node<br>The toy graph database<br>The topK retrieved toy graphs<br>The toy graph<br>_k_-hop ego net for node_vm_<br>The feature / edge weight / label matrix at time_t_<br>The set of label classes<br>The neighbors of node_v_<br>The hidden embeddings / task-specifc output vector with_hi_/_oi_as its_vi_vector<br>The_κ_-shot labeled set<br>Timestamp<br>Number of nodes<br>The dimension of node feature<br>The_i_-th node<br>The master node of toy graph / The center node of toy graph<br>The dimension of hidden embedding / task-specifc output vector<br>The minimum value<br>_k_hop<br>The node importance for node_v∈GR_<br>The PageRank / Degree Centrality value<br>The sampling probability of node_vi_<br>The scaling constant<br>The number of augmentations of toy graph_GT_<br>The weighted similarity between query node_vc_and master node_vm_<br>The layer of a GNN<br>The balance weight with_α∈_(0_,_1)<br>The weight of mixup<br>The weights of the time, structure, semantic, and environment similarities<br>The reweight hyper-parameter|



## **B More Motivation Details** 

## **B.1 Why Toy Graph Augmentation is needed** 

The reasons for toy graph augmentation: 

- Expanding toy graph base, enriching the scale of the knowledge repository [115]. 

- Simulating Real-World Scenarios: Real-world graphs often encounter challenges such as missing nodes [44], noisy attributes [54], and unexplored connections [97]. We introduce node dropout, noise injection, and edge removal to simulate these scenarios accurately. 

- Addressing Graph Domain Shift: To mitigate domain shift between the graph knowledge base and testing graphs, our augmentations employ Mixup techniques such as Node Interpolation and Edge Rewiring. These techniques interpolate between training samples to generate synthetic samples, effectively smoothing decision boundaries in embedding and reducing the model’s sensitivity to minor variations in input data, thereby stabilizing predictions on domain shift testing samples [109]. 

19 

## **B.2 Why Noise-based Graph Prompt Tuning is needed** 

To address inherent challenges in toy graph quality, we introduce Noise-based Graph Prompting Tuning (c.f. Section 4.3.3). This method involves fine-tuning the model with artificially introduced noisy toy graphs (Inner-Toy-Graph Noise & Toy-Graph Noise), inspired by noise-tuning techniques in NLP [19, 12, 116]. Our approach enhances the model’s robustness against real-world retrieval noise, as evidenced by superior performance compared to traditional tuning methods (in Main Text Tables 1 and 2). This approach reduces the stringent requirement for an exceptionally high-quality graph vector base, thereby ensuring robust performance across various tasks within our RAGRAPH, and significantly mitigating data quality impacts. 

## **B.3 Difficulty to construct and maintain high-quality and diverse graph vector base** 

In RAGRAPH, the toy graph base largely leverages significant prior research datasets in pre-trained GNNs [67, 105, 39, 124], which are trained on meticulously curated graph datasets and cover diverse domains, suchasbiology, chemistry, medicinerecommendationtasks, etc. Forexample, thePROTEINS dataset [4], derived from cryo-electron microscopy and X-ray crystallography, and the ENZYMES dataset [100], based on EC enzyme classification, are meticulously annotated by medical experts. 

## **B.4 Why Inverse Importance Sampling Strategy is needed** 

The adoption of the Inverse Importance Sampling strategy is crucial. In RAGRAPH, subgraphs are sampled as toy graphs, where nodes with higher degrees (non-long-tail knowledge, extensively learned and embedded into GNN parameters) are more frequently included in subgraphs due to their extensive connections with neighbors, resulting in higher frequency in toy graph base [28]. Conversely, nodes with low degrees (long-tail knowledge), are more important but ignored. To mitigate this issue, we propose this by prioritizing nodes with lower degrees to capture long-tail knowledge when sampling. 

## **B.5 Why Four Similarities are needed** 

In practical applications, the four similarities all contribute to performance improvement and we state the significance as follows: 

- Time information is crucial to predict future states or trends [114] via node history, i.e. in social networks, analyzing historical user interaction aids in predicting future behaviors. 

- Structure pertains to how nodes are interconnected and overall graph topology, vital for capturing similar graph structure patterns [13, 44, 113]. In transportation networks, factories are always located on the outer ring of the city, sharing similar structural connectivity, aiding in the discovery of spatiotemporal patterns [56, 16]. 

- Sharing similar neighborhoods is essential for evaluating node similarity and correlation. In recommendations, shared purchase histories between users and products indicate potential interests, akin to collaborative filtering [88]. 

- Semantic information measures similarity based on features [39]. In knowledge graphs, identifying relevant subgraphs to query nodes enhances retrieval accuracy based on semantic similarity. 

## **B.6 Why Knowledge Fusion is needed** 

Fusion and decoder here represent one of the core contributions of RAGRAPH: 

- Overall Task Perspective: For the same tasks, the decoder can be directly employed to obtain outputs. For different tasks, the decoder can be masked and utilize pre-computed embeddings without training or be tuned to better adapt. This underscores our primary contribution, where the decoder functions as a versatile "plug-and-play" and "tune-free" component. 

- Integral Fusion Strategy: Fusion Strategy facilitates concurrent information propagation from toy graphs X (hidden embeddings) and Y (task-specific output vector) to query graph, aligning with our secondary contribution. 

20 

## **B.7 How RAGRAPH works on par with RAG in NLP and CV** 

In NLP, RAG enhances the generation of LLM by retrieving relevant information via prompts. Similarly, in RAGRAPH, we enhance downstream graph learning by integrating information from retrieved toy graphs. Using these toy graphs with shared patterns assists the model inference. In our framework, the "generation" involves the retrieval-enhanced Graph Prompt: Toy Graph Intra Propagate & Query-ToyGraph Inter Propagate to propagate retrieved knowledge (X and Y) into the query graph. To illustrate, we analyze this from both experiment and theory. 

1. **Experiment 1:** We perform a case study to illustrate how "generation" works by displaying specific instances of node vectors in Appendix D.6. 

2. **Experiment 2:** In traditional GNN tasks, GCN, GAT, and GIN typically expand their receptive fields through stacked message-passing layers or neighborhood subgraph sampling for inference. Patterns learned in these contexts are often localized within the constrained receptive field. In contrast, in RAGRAPH, we observe that subgraphs sharing similar patterns often exhibit properties more aligned with downstream tasks. These subgraphs provide richer information for inference compared to simply enlarging receptive fields. As shown in Main Text Tables 1 and 2, Figure 3, RAGRAPH’s strategy of incorporating toy graphs significantly outperforms baselines. 

3. **Theory 1:** Furthermore, we provide a theoretical justification of retrieval augmentation in GNNs (see Appendix B.4). From an information-theoretic perspective, introducing RAG knowledge into GNNs enhances the mutual information between input features _X_ and output labels _Y_ , such that: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/rag/nips_2024_RAGraph_A_General_Retrieval-Augmented_Graph_Learning_Framework_images/nips_2024_RAGraph_A_General_Retrieval-Augmented_Graph_Learning_Framework.pdf-0021-05.png)


thereby improving the performance of downstream tasks. This is aligned with the information theory of RAG in NLP [73]. 

4. **Theory 2:** Recent studies [11, 137] also suggest the generalization error diminishes with an increase in the node number of the graph in Theorem 1.1 [137]: the generalization error between the expected loss _R_ exp(Θ) = E( _x,y_ ) _∼µG_ [ _L_ (Θ( _x_ ) _,y_ )] and empirical loss _R_ emp(Θ) = _m_[1] � _mi_ =1 _[L]_[(Θ(] _[x][i]_[)] _[,y][i]_[)][ are] super bounded: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/rag/nips_2024_RAGraph_A_General_Retrieval-Augmented_Graph_Learning_Framework_images/nips_2024_RAGraph_A_General_Retrieval-Augmented_Graph_Learning_Framework.pdf-0021-08.png)


where _C_ represents the model complexity (e.g., parameters), _m_ denotes the training set size, 1 and _q_ ( _n_ ) = E _n∼νG_ [ _n[−] D_ +1 ] depends on the average graph size (node number) with _ν_ as the graph size distribution and _D_ is the metric-measure space dimension. In RAGRAPH, retrieving similar toy graphs significantly increases the number of graph nodes (via Query-Toy-Graph Inter Propagate, linking toy graph nodes to query graph), significantly augmenting _n_ while reducing _q_ ( _n_ ). Consequently, the upper bound of generalization error decreases, promoting smoother graph learning convergence and enhancing pattern learning. 

## **C Further Methods Details** 

## **C.1 Revisiting Graph Neural Networks** 

The goal of a GNN is to learn node embeddings based on an iterative aggregation of messages from the local network neighborhood. We use embedding matrix _{_ **z**[(] _v[L]_[)] _}v∈V_ to denote the embedding for all the nodes after applying an _L_ -layer GNN. The _l_ -th layer of a GNN, _{_ **z**[(] _v[L]_[)] _}_ = GNN[(] _[l]_[)] ( _{_ **z**[(] _v[l][−]_[1)] _}_ ), can be written as: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/rag/nips_2024_RAGraph_A_General_Retrieval-Augmented_Graph_Learning_Framework_images/nips_2024_RAGraph_A_General_Retrieval-Augmented_Graph_Learning_Framework.pdf-0021-13.png)


where **z**[(] _v[l]_[)][is the embedding for] _[ v][ ∈][V]_[after passing through] _[ l]_[ layers,] **[ z]**[(0)] _v_ = _xv_ or _hv_ or _ov_ , **m**[(] _u[l] →_[)] _v_[is] the message embedding, and _N_ ( _v_ ) is the set of direct neighbors of _v_ . Different GNNs can have various definitions of message-passing functions MSG[(] _[l]_[)] ( _·_ ) and aggregation functions AGG[(] _[l]_[)] ( _·_ ) and these two functions could be parameter-free. 

21 

## **C.2 Key Construction of Position-aware Code** 

Given a randomly selected node anchor set _VS ⊂V_ , we calculate the minimal distances, _a.k.a._ hops between the two node sets. Suppose _vu ∈V,vw ∈VS_ , the distance similarity between node _vu_ and _vw_ can be depicted as _dis_ ( _vu,vw_ ). By normalizing the similarity to [0, 1], distance-to-centroid _d_ 2 _c_ ( _vu,vw_ ): 


![](raw/papers/2_LLM_NLP/LLM_paper_list/rag/nips_2024_RAGraph_A_General_Retrieval-Augmented_Graph_Learning_Framework_images/nips_2024_RAGraph_A_General_Retrieval-Augmented_Graph_Learning_Framework.pdf-0022-02.png)


here hyperparameter _disq_ is the maximum hops, the distance beyond this boundary is considered invalid. The structure feature of node _vu_ is _d_ 2 _c_ ( _vu,VS_ ). By collecting all distances with anchor-set _VS_ , the structure _S ∈ R[n][×|V][S][|]_ is written as follows: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/rag/nips_2024_RAGraph_A_General_Retrieval-Augmented_Graph_Learning_Framework_images/nips_2024_RAGraph_A_General_Retrieval-Augmented_Graph_Learning_Framework.pdf-0022-04.png)


where [ _·_ ] means the concatenation operation. The distance-to-centroid feature converts the nonEuclidean structure to the Euclidean structure. _d_ 2 _c_ dramatically reduces the size of the matrix and meanwhile contains more structure information instead of identifier information, the size of the anchor set is log2 _n_ follows P-GNNs[47, 44]. 

## **C.3 Similarity Functions** 

For the history key, we adopt an exponential decay function to measure the time similarity values. We smooth the impact of time differences and provide a controlled decay coefficient _η>_ 0. The time similarity, _S_ time, between the same node _vc_ and _vm_ with different timestamp _t_ ( _vm_ ) _,t_ ( _vc_ ), is defined as: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/rag/nips_2024_RAGraph_A_General_Retrieval-Augmented_Graph_Learning_Framework_images/nips_2024_RAGraph_A_General_Retrieval-Augmented_Graph_Learning_Framework.pdf-0022-08.png)


where _η_ is a positive parameter that controls the rate of exponential decay. 

For the environment key, we match the environment of node _v_ using Jaccard similarity to compare the sets of neighbors _N_ ( _vc_ ) in the query graph and _N_ ( _vm_ ) in the toy graph: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/rag/nips_2024_RAGraph_A_General_Retrieval-Augmented_Graph_Learning_Framework_images/nips_2024_RAGraph_A_General_Retrieval-Augmented_Graph_Learning_Framework.pdf-0022-11.png)


For the hidden embedding key, we input the query graph into pre-trained GNNs to obtain the hidden embedding for the query node, with the similarity defined as: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/rag/nips_2024_RAGraph_A_General_Retrieval-Augmented_Graph_Learning_Framework_images/nips_2024_RAGraph_A_General_Retrieval-Augmented_Graph_Learning_Framework.pdf-0022-13.png)


For the position-aware code, we denote _sc,sm_ as the position-aware code of node _vc,vm_ , and utilize cosine similarity as before, defined as _S_ structure( _vc,vm_ )= `cosine` ( _sc,sm_ ). 

## **C.4 Proof of the Effectiveness of RAG** 

In this section, we will theoretically prove that introducing RAG knowledge can significantly improve the predictive performance of the model. 

Assume _X_ represents the input features, _Y_ represents the target output labels, and _RAG_ represents external knowledge related to the input features (or even the output labels). We analyze from the mutual information view, where _I_ ( _X_ ; _Y_ ) quantifies the dependency between _X_ and _Y_ , which reflects the performance of the model, the larger the value, the better the performance of the model [56, 72]. By introducing RAG knowledge _RAG_ into GNNs, we can effectively increase the mutual information between the input features _X_ and the output labels _Y_ as _I_ ( _X,RAG_ ; _Y_ ) _≥ I_ ( _X_ ; _Y_ ), thereby improve 

22 

the model’s downstream task performance. The derivation is as follows: _I_ ( _X,RAG_ ; **Y** ) _−I_ ( _X_ ; **Y** ) 


![](raw/papers/2_LLM_NLP/LLM_paper_list/rag/nips_2024_RAGraph_A_General_Retrieval-Augmented_Graph_Learning_Framework_images/nips_2024_RAGraph_A_General_Retrieval-Augmented_Graph_Learning_Framework.pdf-0023-01.png)


where[�] _X,RAG,_ **Y**[=][�] _X_ � _RAG_ � **Y**[. Moreover,] _[ I]_[(] _[RAG]_[;] **[Y]** _[|][X]_[)][ measures that how much additional] knowledge _RAG_ provides to assist in predicting _Y_ based on _X_ , this term will approach zero when the _RAG_ is noise to the prediction task. In summary, the integration of RAG knowledge can enhance the mutual information between _X_ and _Y_ , thereby improving the performance and accuracy of downstream tasks. 

## **C.5 Algorithms** 

In this section, we will provide a detailed description of the algorithms of Toy Graph Construction (Algorithm 1) and Training and Inference with Toy Graphs Retrieval (Algorithm 2). 

**Algorithm 1.** We outline the process for constructing toy graphs in Algorithm. 1. In line 1, the toy graph database _G[T]_ is initialized. Lines 2-15 describe the steps to construct toy graphs by iterating through each snapshot of the dynamic resource graph _G[R]_ . 

In more detail, lines 3-5 details calculate the importance and reverse importance of each node within the snapshot. Following this, lines 6-7 involve normalizing the sampling probabilities according to the reverse importance values. The selection of a master node and the generation of its _k_ -hop ego network are carried out in lines 8-9. Subsequently, line 10 involves augmenting the toy graph through specific data augmentation techniques. 

Lines 11-16 detail the generation of key-value pairs for each toy graph. This includes saving the timestamp as the history key, the neighbors of the master node as the environment key, the structural encoding as the structure key, the hidden embedding as the semantic key, and the task-specific output vectors as the value. Each toy graph is then stored in the database _G[T]_ . 

Ultimately, in line 18, the algorithm returns the toy graph database _G[T]_ . 

**Algorithm 2.** We introduce the algorithm for training and inference with toy graph retrieval in Algorithm. 2. Initially, in line 1, we define the required inputs, including the testing graph _G_ test, the toy graph database _G[T]_ , and other relevant parameters. The final output is the aggregated result ˜ _oc_ . 

The RETRIEVETOYGRAPHS function, detailed in lines 3-11, initializes an empty similarity list and iterates over each toy graph in the database. Lines 5-6 compute various similarity metrics, and the overall similarity is determined in line 7. This similarity score is then added to the list. After sorting by similarity, the _topK_ toy graphs are retrieved and returned in line 11. 

Within the PROPAGATION function (lines 12-17), each retrieved toy graph undergoes intra-propagation in line 14. The intra-propagation step follows in line 16, ultimately returning the propagated results **z** _c_ . 

The KNOWLEDGEFUSION function, found in lines 18-21, combines the outputs from previous steps. The final combined outputs ˜ _oc_ are generated by the decoder and returned in line 21. 

23 

**Algorithm 1** Toy Graph Construction 

**Require:** DynamicResourceGraph _G[R]_ , nodeimportancebalanceweight _α_ , toygraphscalingconstant _K_ , maximum hop _k_ **Ensure:** Toy graph embedding key-value database _G[T]_ 

1: Initialize toy graph database _G[T] ←∅_ 2: **for** each snapshot _G[R] τ[∈G][R]_ **[ do]** _▷_ **Construct Toy Graphs** 3: **for** each node _v ∈ G[R] τ_ **[do]** 4: Calculate importance _I_ ( _v_ ) _← α_ PR( _v_ )+(1 _−α_ )DC( _v_ ) 5: Reverse node importance _I[′]_ ( _v_ ) _← I_ ( _v_ 1)+ _ϵ_ 6: **end for** 7: **for** each node _v ∈ G[R] τ_ **[do]** _I[′]_ ( _vi_ ) 8: Normalize sampling probabilities _pi ← n_ � _j_ =1 _[I][′]_[(] _[v][j]_[)] 9: **end for** 10: Sample master node _vm ←_ WEIGHTEDSAMPLING( _G[R] τ_[,] _[ p][i]_[) based on probability] _[ p][i]_ 11: Generate _k_ -hop ego net _G[e] τ_[(] _[v][m]_[)][ for node] _[ v][m]_ 12: Augment toy graph _{G[T] }←_ DATAAUGMENTATION( _G[e] τ_[(] _[v][m]_[)][,] _[ n]_[aug][)][ with] _[ n]_[aug][(] _[G][e] τ_[(] _[v][m]_[))=] _⌊K ·I_[¯] _[′]_ ( _G[e] τ_[(] _[v][m]_[))] _[⌋]_ 13: **for** each _G[T] ∈{G[T] }_ **do** _▷_ **Generate keys-values pair** 14: Save timestamp _τ_ as history key 15: Save neighbors _N_ ( _vm[τ]_[)][ of master node] _[ v][m]_[as environment key] 16: Save structural encoding _s[τ] m_[of node] _[ v][m]_[via Eq. (][9][) as structure key] 17: Save hidden embedding _h[τ] m_[by feeding] _[ G][T]_[into pre-trained GNNs as semantic key] 18: Save the hidden embedding _{h[τ] i[|][v][i][ ∈][G][T][ }]_[ as value] 19: Save task-specific output vectors _{o[τ] i[|][v][i][ ∈][G][T][ }]_[ by feeding] _[ {][h][τ] i[|][v][i][ ∈][G][T][ }]_[ into decoder as] value 20: Store toy graph _G[T]_ into database _G[T]_ 21: **end for** 22: **end for** 23: **return** Toy graph database _G[T]_ 

The main algorithm begins in line 22. If fine-tuning is enabled and the prompt loss has not converged, lines 23-34 detail the process of toy graph retrieval and propagation for each query graph. This includes the optional addition of noise in lines 26-29. The hidden embedding and task-specific output vectors are propagated in lines 30-31, and the aggregated outputs are fused in line 32. The prompt loss is computed, and fine-tuned parameters are updated in lines 33-34. 

If fine-tuning is not required, lines 35-39 describe a similar process of toy graph retrieval and propagation, without the fine-tuning steps. The aggregated outputs are computed directly. 

## **D Further Experiment Details** 

## **D.1 Datasets Statics** 

We employ eight benchmark datasets for evaluation including four public static classification datasets for node- and graph-level tasks. 

(1) PROTEINS [4] is a collection of protein graphs, including the amino acid sequence, conformation, structure, and features such as active sites of the proteins. Each node represents a secondary structure, while each edge illustrates the neighboring relationship either within the amino acid sequence or in 3D space. The nodes are divided into three categories, and the graphs are classified into two distinct classes. This dataset is used for node and graph classification tasks, containing 1,113 graphs with an average of 39.06 nodes and 72.82 edges per graph, with a density of 4.8e-2. 

(2) COX2 [84] is a dataset of molecular structures, including 467 cyclooxygenase-2 inhibitors. Each node represents an atom, and each edge signifies a chemical bond between atoms, such as single, double, triple, or aromatic bonds. All the molecules belong to two categories. This dataset is used for 

24 

**Algorithm 2** Training and Inference with Toy Graphs Retrieval 

_·_ **Require:** Testing graph _G_ test, toy graph database _G[T]_ , pre-trained GNN model GNNΘ0( ), number of _TopK_ toy graphs to retrieve, similarity weights _w_ 1 _,w_ 2 _,w_ 3 _,w_ 4, fine-tuning flag _fine_  tune_ , noise prompt-tuning flag _add_  noise_ **Ensure:** Aggregated output ˜ _oc_ 1: **function** RETRIEVETOYGRAPHS( _G[Q]_ , _G[T]_ , _TopK_ ) 2: Initialize similarity list _{S}←∅_ 3: **for** each toy graph _G[T] ∈G[T]_ **do** 4: Calculate time similarity _S_ time, environment similarity _S_ environment, structure similarity _S_ structure and semantic similarity _S_ semantic 5: Compute similarity _S ← w_ 1 _·S_ time + _w_ 2 _·S_ structure + _w_ 3 _·S_ environment + _w_ 4 _·S_ semantic 6: Add ( _G[T] ,S_ ) to _{S}_ 7: **end for** 8: Sort _{S}_ by similarity in descending order 9: Retrieve _topK_ toy graphs _G[T]_ TopK _[←{][G][T][∈{][S][}]_[ with] _[ topK]_[ similarities] _[}]_ 10: **return** Retrieved toy graphs _G[T]_ TopK 11: **end function** 12: **function** PROPAGATION( _G[Q]_ , _G[T]_ TopK[)] 13: **for** each toy graph _G[T] ∈ G[T]_ TopK **[do]** 14: Perform Intra Propagation **z** _m ←_ INTRAPROPAGATION( _G[T]_ ) 15: **end for** 16: **z** _c ←_ INTERPROPAGATION( _G[Q]_ , _G[T]_ TopK[)] 17: **return z** _c_ 18: **end function** 19: **function** KNOWLEDGEFUSION( _hc,oc_ ) 20: Combined output ˆ _oc ← γoc_ +(1 _−γ_ )DECODER( _hc_ ) 21: **return** Combined outputs ˜ _oc_ 22: **end function** 23: **if** _fine_  tune_ & _L_ prompt not converged **then** 24: **for** each query graph _G[Q] ∈G_ test **do** _▷_ Toy Graph Retrieval and Propagation 25: _G[T]_ TopK _[←]_[R][ETRIEVE][T][OY][G][RAPHS][(] _[G][Q]_[,] _[ G][T]_[ ,] _[ TopK]_[)] 26: **if** _add_  noise_ **then** 27: **for** each toy graph _G[T] ∈ G[T]_ TopK **[do]** 28: Introduce noise _G[T] ←_ ADDNOISE( _G[T]_ ) _▷_ Inner-Toy-Graph Noise 29: **end for** 30: Add _bottomK_ toy graphs to _G[T]_ TopK _▷_ Toy-Graph Noise 31: **end if** 32: _hc ←_ PROPAGATION( _G[Q]_ , _G[T]_ TopK[)] _▷_ Propogate hidden embedding 33: _oc ←_ PROPAGATION( _G[Q]_ , _G[T]_ TopK[)] _▷_ Propogate task-specific output vector 34: Aggregated outputs ˆ _oc ←_ KNOWLEDGEFUSION( _hc,oc_ ) 35: Compute loss _L_ prompt via ˜ _oc_ and ˆ _oc ▷_ Based on task-specific loss function 36: Update fine-tuned parameters Θ by minimizing _L_ prompt 37: **end for** 38: **else** 39: **for** each query graph _G[Q] ∈G_ test **do** _▷_ Toy Graph Retrieval and Propagation 40: _G[T]_ TopK _[←]_[R][ETRIEVE][T][OY][G][RAPHS][(] _[G][Q]_[,] _[ G][T]_[ ,] _[ TopK]_[)] 41: _hc ←_ PROPAGATION( _G[Q]_ , _G[T]_ TopK[)] _▷_ Propogate hidden embedding 42: _oc ←_ PROPAGATION( _G[Q]_ , _G[T]_ TopK[)] _▷_ Propogate task-specific output vector 43: Aggregated outputs ˆ _oc ←_ KNOWLEDGEFUSION( _hc,oc_ ) 44: **end for** 45: **end if** 46: **return** Aggregated outputs ˜ _oc_ 

25 

Table 4: Statistics of the experimental datasets and summary of datasets. 

|Table 4: Statistics of the experimental datasets and summaryof datasets.|Table 4: Statistics of the experimental datasets and summaryof datasets.|
|---|---|
|Statistics<br>TAOBAO<br>KOUBEI<br>AMAZON<br>PROTEINS<br>COX2<br>ENZYMES<br>BZR||
|# Nodes per Graph<br># Edges per Graph<br># Density<br># Graphs<br># Graph Classes<br># Node Features<br># Node Classes|204,168<br>221,366<br>238,735<br>39.06<br>41.22<br>32.63<br>35.75<br>8,795,404<br>3,986,609<br>876,237<br>72.82<br>43.45<br>62.14<br>38.36<br>8.6e-4<br>3.3e-4<br>6.2e-5<br>4.8e-2<br>2.6e-2<br>5.9e-2<br>3.0e-2<br>1<br>1<br>1<br>1,113<br>467<br>600<br>405<br>/<br>/<br>/<br>2<br>2<br>6<br>2<br>1<br>1<br>1<br>1<br>3<br>18<br>3<br>/<br>/<br>/<br>3<br>/<br>3<br>/|
|Snapshot Granularity<br>Task<br>Type<br>Dataset Partition|daily<br>weekly<br>weekly<br>/<br>/<br>/<br>/<br>Edge<br>Edge<br>Edge<br>Node, Graph<br>Graph<br>Node, Graph<br>Graph<br>Dynamic<br>Dynamic<br>Dynamic<br>Static<br>Static<br>Static<br>Static<br>Snapshot<br>Snapshot<br>Snapshot<br>Node, Graph<br>Graph<br>Node, Graph<br>Graph|



graph classification tasks, with each graph having an average of 41.22 nodes and 43.45 edges and a density of 2.6e-2. 

(3) ENZYMES [100] is a dataset of 600 enzymes collected from the BRENDA enzyme database. These enzymes are labeled into 6 categories according to their top-level EC enzyme classification. This dataset is used for node and graph classification tasks, with each graph having an average of 32.63 nodes and 62.14 edges and a density of 5.9e-2. 

(4) BZR [84] is a collection of 405 ligands for the benzodiazepine receptor. Each ligand is represented by a graph, and all ligands are categorized into two groups. This dataset is used for graph classification tasks, with each graph having an average of 35.75 nodes and 38.36 edges and a density of 3.0e-2. 

Additionally, we leverage three publicly available datasets encompassing a wide array of real-world scenarios in dynamic recommendation (link prediction): 

(5) The TAOBAO dataset captures implicit feedback data from Taobao.com, a prominent Chinese e-commerce platform, collected over a span of 10 days. This dataset is used for edge classification tasks, containing 204,168 nodes and 8,795,404 edges, with a density of 8.6e-4. 

(6) The KOUBEI dataset records 9 weeks of user interactions with nearby stores on Koubei, a platform integrated within Alipay. This dataset is used for edge classification tasks, containing 221,366 nodes and 3,986,609 edges, with a density of 3.3e-4. 

(7) The AMAZON dataset comprises a collection of product reviews sourced from Amazon, spanning a duration of 13 weeks. This dataset is used for edge classification tasks, containing 238,735 nodes and 876,237 edges, with a density of 6.2e-5. 

These datasets’ detailed statistics are summarized in Table 4. The "Task" column provides information about the type of downstream task conducted on each dataset: "Node" denotes node classification tasks, "Graph" signifies graph classification tasks, and "Edge" indicates tasks related to link prediction. The "Type" column indicates the type of graph dataset: "Dynamic" for dynamic dataset _t ≥_ 1, and "Static" for static dataset _t_ =1. For dynamic datasets, the "Snapshot Granularity" denotes the time granularity for each dataset. In our experimental setup, for dataset partition, dynamic graphs are partitioned according to snapshots, while static graphs are partitioned either by node or by the entire graph. 

## **D.2 Evaluation Matrices** 

**Node and Graph classification evaluation.** For the node classification, we use the prediction accuracy to measure the model. 

**Link prediction evaluation.** For the link prediction, we evaluate the recall and ranking quality of the effects of recommendation following previous studies [118, 31]. We use Recall@k and NDCG@k as metrics. Note that this task should be a binary task. We denote the _topk_ largest value as _relij,j ∈_ [1 _,k_ ] for node _vi_ . 

26 

Recall@k measures the ratio of true positive links contained in the top _k_ predicted links for each node: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/rag/nips_2024_RAGraph_A_General_Retrieval-Augmented_Graph_Learning_Framework_images/nips_2024_RAGraph_A_General_Retrieval-Augmented_Graph_Learning_Framework.pdf-0027-01.png)


where _relij_ =1 if the _j_ -th predicted link for node _vi_ exists, otherwise 0. I( _·_ ) is the indicator function, and if _A_ [ _i_ :] _>_ 0 then I( _A_ [ _i_ :] _>_ 0)=1. 

NDCG@k (Normalized Discounted Cumulative Gain) is computed by normalizing DCG@k (Discounted Cumulative Gain) which accounts for the position of correctly predicted links. DCG@k is defined as: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/rag/nips_2024_RAGraph_A_General_Retrieval-Augmented_Graph_Learning_Framework_images/nips_2024_RAGraph_A_General_Retrieval-Augmented_Graph_Learning_Framework.pdf-0027-04.png)


## **D.3 Baseline Details** 

In this section, we present the details of baselines. 

Table 5: Baseline Code URLs of Github Repository 

|**Baseline**|**Type**|**Code Repo URL**|
|---|---|---|
|GCN|Static|`https://github.com/tkipf/gcn`|
|GraphSAGE|Static|`https://github.com/williamleif/GraphSAGE`|
|GAT|Static|`https://github.com/PetarV-/GAT`|
|GIN|Static|`https://github.com/weihua916/powerful-gnns`|
|LightGCN|Dynamic|`https://github.com/kuandeng/LightGCN`|
|SGL|Dynamic|`https://github.com/wujcan/SGL-Torch`|
|MixGCF|Dynamic|`https://github.com/Wu-Xi/SimGCL-MixGCF`|
|SimGCL|Dynamic|`https://github.com/Wu-Xi/SimGCL-MixGCF`|
|GraphPro|Dynamic|`https://github.com/HKUDS/GraphPro`|
|GraphPrompt|Dynamic, Static|`https://github.com/Starlien95/GraphPrompt`|
|PRODIGY|Dynamic,Static|`https://github.com/snap-stanford/prodigy`|



- **GCN** [50]: GCN is an end-to-end learning framework for graph-structured data. It utilizes neighborhood aggregation to integrate structural information, which is particularly effective in node classification and graph classification tasks. 

- **GraphSAGE** [30]: GraphSAGE, is a general and inductive framework that leverages node feature information (e.g., text attributes) to efficiently generate node embeddings for previously unseen data. 

- **GAT** [98]: GAT is a spatial domain method, which aggregates information through the attentionlearned edge weights. 

- **GIN** [106]: GIN utilizes a multi-layer perceptron to sum the results of GNN and learns a parameter to control residual connection. 

- **LightGCN** [31]: LightGCN learns user and item embeddings by linearly propagating them on the user-item interaction graph, and uses the weighted sum of the embeddings learned at all layers as the final embedding. 

- **SGL** [104]: SGL is to supplement the classical supervised task of recommendation with an auxiliary self-supervised task, which reinforces node representation learning via self-discrimination. 

- **MixGCF** [41]: MixGCF generates the synthetic negative by aggregating embeddings from different layers of raw negatives’ neighborhoods to perform collaborative filtering. 

27 

- **SimGCL** [118]: SimGCL applies unsupervised contrastive learning to enhance representation learning, making it suitable for link prediction tasks. It is applied to dynamic graphs to test its adaptability and performance. 

- **GraphPro** [114]: GraphPro extends GraphPrompt by introducing spatial and temporal prompts tailored for dynamic graph learning, enhancing the ability to capture both structural and temporal relationships within graph data. 

- **GraphPrompt** [67]: GraphPrompt integrates pre-training and downstream tasks using a unified template approach and employs task-specific prompts to enhance sub-task learning, applicable to both dynamic and static graph contexts. 

- **PRODIGY** [39]: PRODIGY focuses on facilitating downstream tasks through in-context examples and learning from the _X → Y_ paradigm. It is implemented to enhance learning in both dynamic and static graphs by leveraging contextual learning strategies. 

## **D.4 Implementation Details** 

Implementations are done using the PyTorch 2.3.0 framework [79] in Python 3.11, on an Ubuntu server equipped with 1 V100 GPU and an Intel(R) Xeon(R) CPU. 

In node and graph classification tasks: For baseline GCN [50], we employ a 2-layer architecture and set the hidden dimension as 256. For GraphSAGE [30], we utilize the mean aggregator and employ a 2-layer architecture. The hidden dimension is also set to 256. For GAT [98], we employ a 2-layer architecture and set the hidden dimension as 256. Besides, we apply 8 attention heads in the first GAT layer. Similarly, for GIN [106], we also employ a 2-layer architecture and set the hidden dimension as 256. For GraphPrompt, we follow [67] to employ a 2-layer GCN as the backbone and set the hidden dimensions as 256. 

In the link prediction task: For LightGCN, SGL, MixGCF, SimGCL and GraphPro, we employ a 3-layer GNN architecture and set the hidden dimension as 64 with Low-Rank Adaptation (LoRA) [33] rank equals to 16. For GraphPro, the backbone graph encoder is SimGCL. 

Moreover, for all three tasks, the hyper-parameters of baselines are based on the recommended values provided in the paper. In PRODIGY and RAGRAPH, _k_ is set to 2, _topK_ is set to 5, _γ_ is set to 0.8 for PROTEINS and 0.5 for ENZYMES in node level, _γ_ is set to 0.5 for PROTEINS, 0.6 for COX2, 0.8 for ENZYMES and 0.5 for BZR in graph level, _α_ = _λ_ =0 _._ 5 _,K_ =3 _,w_ 1 = _w_ 2 = _w_ 3 =0 _._ 05 _,w_ 4 =0 _._ 85. 

## **D.5 Resource Graph Scalability Study** 


![](raw/papers/2_LLM_NLP/LLM_paper_list/rag/nips_2024_RAGraph_A_General_Retrieval-Augmented_Graph_Learning_Framework_images/nips_2024_RAGraph_A_General_Retrieval-Augmented_Graph_Learning_Framework.pdf-0028-10.png)


**----- Start of picture text -----**<br>
60<br>70 55<br>60 50<br>45<br>50<br>40<br>40<br>35<br>30 GIN +PRODIGY 30 GIN +PRODIGY<br>GraphPrompt +RAGraph GraphPrompt +RAGraph<br>10% 20% 30% 40% 50% 60% 70% 80% 10% 20% 30% 40% 50% 60% 70% 80%<br>Train & Resource Graph Size (%) Train & Resource Graph Size (%)<br>(a) Node Classification on ENZYMES dataset (b) Node Classification on PROTEINS dataset<br>Accuracy (%) Accuracy (%)<br>**----- End of picture text -----**<br>


Figure 4: Performance comparisons of RAGRAPH and several baselines with different proportions of training and resource data. 

We assess the impact of varying amounts of training and resource data on model performance. As illustrated in Figure 4, we vary the proportion of train and resource graph size from 10% to 80%, with increments of 10%, and conduct experiments on node classification tasks using the ENZYMES and PROTEINS datasets, respectively. For comparative analysis, we select GIN, GraphPrompt, and PRODIGY as baseline models. To ensure fairness in our experiments, we maintain a consistent ratio of train to resource data at 3:5 during fine-tuning, utilizing the sum of these as a retrieval database. 

28 

As shown in Figure 4, there is a clear trend where the accuracy of the model improves as the proportion of the dataset increases. However, the rate of accuracy improvement starts to plateau once a certain dataset proportion is reached ( _i.e._ , 30% in PROTEINS for PRODIGY and RAGRAPH, 40% in PROTEINS for GraphPrompt). Among the evaluated models, GIN and GraphPrompt show the slowest convergence rates, whereas PRODIGY converges at a moderate pace, and RAGRAPH converges the fastest. This rapid convergence in PRODIGY and RAGRAPH is attributed to its ability to engage in effective knowledge retrieval, significantly enhancing the model’s comprehension abilities. Remarkably, both PRODIGY and RAGRAPH can achieve commendable results in downstream tasks even with a small proportion of the dataset. Compared to PRODIGY, RAGRAPH exhibits superior performance because while PRODIGY primarily learns a mapping from _X_ to _Y_ , RAGRAPH not only learns this mapping but also integrates additional knowledge into GNNs more effectively. This integration becomes increasingly beneficial as the dataset proportion grows, allowing RAGRAPH to outperform other models, particularly at higher data volumes where it can better leverage its knowledge integration capabilities. 

## **D.6 Qualitative Analyses of Toy Graphs Retrieving** 

In this section, we conduct qualitative analyses of the toy graphs retrieving experiment. For the sake of understanding, we conduct experiments under normal settings where the dimensionality of the task-specific output vector is equal to the number of classes. 

On the ENZYMES dataset, for a 3-class node classification task, regarding node "13984", which belongs to class 3, if we only use the GraphPrompt Backbone, the resulting one-hot encoding is [0 _._ 28 _,_ 0 _._ 34 _,_ 0 _._ 38]. 

However, since the node is of class 3, we expect the one-hot encoding to be as close as possible to [0 _,_ 0 _,_ 1]. In RAGRAPH retrieval, taking the top 3 retrieved graphs as examples, the connection weights for these 3 toy graphs to query graphs are 0.5, 0.7, and 0.1, respectively, and their corresponding label one-hot encodings are [0 _,_ 0 _,_ 1], [0 _,_ 0 _,_ 1], and [0 _,_ 1 _,_ 0]. Therefore, the result obtained by propagating the task-specific output vector through toy graphs is: [0 _,_ 0 _._ 1 _,_ 1 _._ 2], and after normalization, the result is [0 _,_ 0 _._ 08 _,_ 0 _._ 92]. 

Meanwhile, the vector obtained by propagating toy graphs hidden embedding and via decoder is [0 _._ 37 _,_ 0 _._ 32 _,_ 0 _._ 66]. The retrieval of toy graphs notably enhances performance at both the task-specific output vector and hidden embedding levels. The final vector is obtained through a weighted sum with _γ_ = 0 _._ 5 in Eq(6) is [0 _._ 185 _,_ 0 _._ 20 _,_ 0 _._ 79], after normalization the result is [0 _._ 157 _,_ 0 _._ 170 _,_ 0 _._ 673], which greatly enhances the model’s discriminative ability compared to GraphPrompt [0 _._ 28 _,_ 0 _._ 34 _,_ 0 _._ 38]. 


![](raw/papers/2_LLM_NLP/LLM_paper_list/rag/nips_2024_RAGraph_A_General_Retrieval-Augmented_Graph_Learning_Framework_images/nips_2024_RAGraph_A_General_Retrieval-Augmented_Graph_Learning_Framework.pdf-0029-06.png)


**----- Start of picture text -----**<br>
Output Vector via Decode the Hidden Embedding<br>[0.37, 0.32, 0.66] 0.1 13994 [0, 0.1, 1.2]<br>[0.1, 0.6, 0.3]<br>0.5<br>13984 [0, 1, 0] [0.37, 0.32, 0.66]<br>[0.3, 0.2, 0.5]<br>13997<br>[0.2, 0.1, 0.7] Task-specific Output Vector<br>[0.185, 0.20, 0.79]<br>[0, 0.1, 1.2]<br>0.7<br>[0, 0, 1] Normalization<br>[0.157, 0.17, 0.673]<br>13983<br>[0.3, 0.3, 0.4] [0, 0, 1]<br>Task-specific Output Vector   Center Node of the Query Graph<br>Master Node of the Toy Graph<br>Output Vector via Decode the Hidden Embedding<br>**----- End of picture text -----**<br>


Figure 5: Qualitative analyses of toy graphs retrieving – how “generation” works. 

29 


![](raw/papers/2_LLM_NLP/LLM_paper_list/rag/nips_2024_RAGraph_A_General_Retrieval-Augmented_Graph_Learning_Framework_images/nips_2024_RAGraph_A_General_Retrieval-Augmented_Graph_Learning_Framework.pdf-0030-00.png)


**----- Start of picture text -----**<br>
 PRODIGY : Learn  X       Y  by  ICL<br>Static Example Graphs Query Node Embedding Toy Master Embedding<br>Query Graph Query Graph Toy Graph<br>�� ��<br>�� ... �� ��<br>�� ��  similarity<br>Assign the same label  Y to query graph<br> RAGraph : Learn  X & Y  from RAG<br>Dynamic  Toy Graphs Toy Graph Intra Propogate<br>Query Graph �� ��<br>�� ��<br>�� ...<br>�� ...<br>�� �� ��<br>Query-Toy-Graph Inter Propogate<br>Inject the retrieved X and Y into query node<br>��<br>**----- End of picture text -----**<br>


Figure 6: Difference Illustration between PRODIGY and RAGRAPH. 

## **E Difference between ICL (PRODIGY) and RAG (RAGRAPH)** 

In this section, we explore the distinctions between PRODIGY and RAGRAPH from several critical perspectives, as illustrated in Figure 6: 

- **PRODIGY:** This approach utilizes fixed examples as rules, which may not be optimal for dynamic and evolving scenarios. PRODIGY primarily focuses on learning direct mappings from _X_ to _Y_ through in-context learning. However, it encounters challenges in integrating external information that is more pertinent to the query node. This is particularly problematic when the distribution of each node belonging to the same label class varies, and simply learning the mapping based on the prototype node will somehow be misleading. 

- **RAGRAPH:** In contrast, RAGRAPH is designed to handle non-static, streaming knowledge, making it well-suited to dynamic graph structures and evolving tasks. It actively retrieves relevant knowledge on-demand, effectively incorporating information about both _X_ and _Y_ from external sources into GNNs. Moreover, RAGRAPH can operate without the need for model fine-tuning, providing substantial flexibility. This adaptability enables RAGRAPH to excel in tasks that require continuous adaptation to changing conditions and the integration of external, relevant information. 

In summary, we argue that a qualified Retrieval-Augmented Generation (RAG) system for Graph Learning should fulfill several essential criteria to effectively support complex reasoning tasks: 1) It should retrieve ample feature and task-related label information, analogous to how attributes are gathered in the NLP domain to stimulate the reasoning capabilities of LLMs; 2) The system should adapt to new tasks or unseen datasets without requiring fine-tuning of model parameters; 3) Knowledge within the system must be dynamically updated and stored, ensuring current and relevant data utilization. 

## **F Broader Impacts** 

Our work builds on the widespread application of Retrieval-Augmented Generation (RAG) in large language models (LLMs) and aims to extend its success to graph data, thereby constructing graph foundation models. This approach allows models to transfer rapidly without requiring learnable parameters, avoiding potential performance degradation from fine-tuning pre-trained models. As a result, RAG is particularly effective in domains with scarce and long-tail data, such as network anomaly detection, rare disease diagnosis/treatment, supply chain disruption, and new user recommendations. 

30 

Additionally, our model establishes an excellent paradigm by incorporating retrieved features and label information into the learning process, significantly enhancing the model’s understanding capabilities. Our work provides valuable insights and serves as a reference for future Large Graph Models. 

## **G Data Ethics Statement** 

To evaluate the efficacy of this work, we conducted experiments that only use publicly available datasets, namely, PROTEINS, COX2, ENZYMES, BZR[3] , TAOBAO, KOUBEI and AMAZON in accordance to their usage terms and conditions if any. We further declare that no personally identifiable information was used, and no human or animal subject was involved in this research. 

> 3 `https://chrsmrrs.github.io/datasets/` 

31 

