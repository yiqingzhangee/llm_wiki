## **UniGAD: Unifying Multi-level Graph Anomaly Detection** 

**Anonymous Author(s)** Affiliation Address `email` 

## **Abstract** 

1 Graph Anomaly Detection (GAD) aims to identify uncommon, deviated, or sus2 picious objects within graph-structured data. Existing methods generally focus 3 on a single graph object type (node, edge, graph, etc.) and often overlook the 4 inherent connections among different object types of graph anomalies. For in5 stance, a money laundering transaction might involve an abnormal account and 6 the broader community it interacts with. To address this, we present UniGAD, 7 the first unified framework for detecting anomalies at node, edge, and graph lev8 els jointly. Specifically, we develop the Maximum Rayleigh Quotient Subgraph 9 Sampler (MRQSampler) that unifies multi-level formats by transferring objects 10 at each level into graph-level tasks on subgraphs. We theoretically prove that 11 MRQSampler maximizes the accumulated spectral energy of subgraphs (i.e., the 12 Rayleigh quotient) to preserve the most significant anomaly information. To 13 further unify multi-level training, we introduce a novel GraphStitch Network to 14 integrate information across different levels, adjust the amount of sharing required 15 at each level, and harmonize conflicting training goals. Comprehensive experi16 ments show that UniGAD outperforms both existing GAD methods specialized 17 for a single task and graph prompt-based approaches for multiple tasks, while 18 also providing robust zero-shot task transferability. All codes can be found at 19 `https://anonymous.4open.science/r/UniGAD-A087/.` 

## 20 **1 Introduction** 

- 21 Graph Anomaly Detection (GAD) involves identifying a minority of uncommon graph objects that 22 significantly deviate from the majority within graph-structured data [16, 2]. These anomalies can 23 manifest as abnormal nodes, unusual relationships, irregular substructures within the graph, or 24 entire graphs that deviate significantly from others. GAD has many practical applications in various 25 contexts, including the identification of bots and fake news on social media [3, 1, 4], detection of 26 sensor faults and internet invasions in IoT networks [7, 13], and prevention of fraudsters and money 27 laundering activities in transaction networks [18, 52]. The mainstream GAD models originate from 28 the Graph Neural Networks (GNNs), which have recently gained popularity for mining graph data 29 [49, 23, 54, 15]. To address the specific challenges of graph anomalies such as label imbalance 30 [32, 29], relation camouflage [11, 36], and feature heterophily [48, 14], numerous adaptations of 31 standard GNNs have been proposed [8, 63, 12, 38, 37, 9, 50, 39, 59]. 

- 32 However, existing GAD approaches typically focus on a single type of graph object, such as node-level 33 or graph-level anomaly detection, often overlooking the inherent correlations between different types 34 of objects in graph-structured data. For example, a money laundering transaction might involve both 35 an abnormal account and the broader community it interacts with, while the specific cancer of a cell 36 is determined by particular proteins or protein complex within the cell. Although some unsupervised 37 methods fuse information from nodes, edges, and subgraphs through reconstruction [26, 9, 45] 

Submitted to 38th Conference on Neural Information Processing Systems (NeurIPS 2024). Do not distribute. 


![](raw/papers/4_Other_ML/Graph_paper_list/2024-NIPS-林依清-UniGAD_images/2024-NIPS-林依清-UniGAD.pdf-0002-00.png)


**----- Start of picture text -----**<br>
Pre-train Shared GNN SubgraphSampling RepresentationGraph Multi-levelLayer1 Multi-levelLayer2 Prediction<br>Stitch Stitch<br>node Layer1 Layer2 node AnomaliesNode<br>Node<br>Input EncoderGNN  MQRSampler Pooling edge   𝛼  𝛼!" …   𝛼  𝛼!" edge OutputAnomaliesEdge<br>Edge   𝛼#   𝛼#<br>graph graph AnomaliesGraph<br>… Graph<br>(I) Unify multi-level formats. (II) Unify multi-level training.<br>**----- End of picture text -----**<br>


Figure 1: The overall framework of UniGAD. 

- 38 or contrastive pre-training [55, 12, 34], they are still limited to single-level label supervision or 39 prediction. There is a need for a unified approach that considers these correlations information across 40 different levels and performs multi-level anomaly detection. 

- 41 To design a unified model for addressing multi-level GAD, we identify two key challenges: 42 **1.** _**How to unify multi-level formats?**_ Addressing node-level, edge-level, and graph-level tasks 43 uniformly is challenging due to their inherent differences. Some recent works provide insights into 44 unifying these tasks through the use of large language models (LLMs) or prompt tuning. While some 45 methods leverage the generalization capability of LLMs [30, 51, 28] on text-attributed graphs, such 46 semantic information is often unavailable in anomaly detection scenarios due to privacy concerns. 47 On the other hand, graph prompt learning methods [46, 35, 58] design induced _k_ -hop graphs to 48 transform node or edge levels into graph-level tasks. Nevertheless, their sampling strategies are not 49 specifically tailored to anomaly data, resulting in inappropriate node selections that ‘erase’ critical 50 anomaly information. This oversight can severely impact the effectiveness of anomaly detection. 51 **2.** _**How to unify multi-level training?**_ Training a single model for multi-level tasks involves various 52 influencing factors, such as transferring information between different levels and achieving a balanced 53 training of these level tasks. There is limited research on multi-task learning in the graph learning 54 domain. Efforts like ParetoGNN [22] employ multiple self-supervised learning objectives (e.g., 55 similarity, mutual information) as independent tasks, but these are insufficient for managing multi56 level supervision. A comprehensive approach is needed to effectively integrate and balance the 57 training of different level tasks in multi-level GAD. 

- 58 In this paper, we propose UniGAD, a unified GAD model that leverages the transferability of 59 information across node-level, edge-level, and graph-level tasks. To address the first challenge, we 60 develop a novel subgraph sampler, **MRQSampler** , that maximizes accumulated spectral energy 61 (i.e., the Rayleigh quotient) in the sampled subgraph with theoretical guarantee, ensuring that the 62 sampled subgraphs contain the most critical anomaly information from nodes and edges. For the 63 second challenge, we introduce the **GraphStitch** Network, which unifies multi-level training by 64 integrating separate but identical networks for nodes, edges, and graphs into a unified multi-level 65 model. This is achieved using a novel GraphStitch Unit that facilitates information sharing across 66 different levels while maintaining the effectiveness of individual tasks. We perform comprehensive 67 experiments on 13 GAD datasets and compare 17 state-of-the-art methods covering both node-level 68 and graph-level GAD techniques, as well as prompt-based general multi-task graph learning methods. 69 Results show that UniGAD achieves superior performance and offers robust zero-shot transferability 70 across different tasks. 

- 70 

## 71 

## **2 Related Work and Preliminaries** 

- 72 **Graph Anomaly Detection.** Leveraging deep learning techniques in GAD has led to significant 73 advancements and a wide range of applications [3, 13, 1, 4, 18], thoroughly reviewed in a comprehen74 sive survey [41]. Node-level anomaly detection, the most prevalent scenario in GAD, has witnessed 75 numerous adaptations and improvements in graph neural networks (GNNs) aimed at enhancing 76 performance from either a spatial [32, 36, 24] or spectral [27, 48, 14] perspective. Despite these 

- 77 advancements, recent benchmarks such as BOND [31] for unsupervised settings and GADBench [47] 

2 

- 78 for supervised settings reveal that no single model excels across all datasets, highlighting the need for 79 model selection tailored to specific datasets and task characteristics. For graph-level anomaly detec80 tion, various methodologies have been proposed, including transformation learning [62], knowledge 81 distillation [40], and evolutionary mapping [42]. SIGNET [33] employs information bottleneck to 82 generate informative subgraphs for explaining graph-level anomalies, while Rayleigh Quotient GNN 83 [10] explores the spectral properties of anomalous graphs. Although both node-level and graph-level 84 anomaly detection are rapidly evolving fields, to the best of our knowledge, there is no existing model 85 that supports the joint detection of both node-level and graph-level anomalies. 

- 86 **Multi-task Learning on Graphs.** Multi-task learning involves training a model to handle multiple 87 tasks simultaneously, utilizing shared representations and relationships within the graph to enhance 88 performance across all tasks. Recently, techniques such as graph prompt-based approaches and large 89 language model (LLM)-based approaches have shown promise in this area. Prompt frameworks 90 like GraphPrompt [35], All-in-One [46], PRODIGY [21], MultiGPrompt [58], and SGL-PT [64] 91 are designed to address a wide array of graph tasks. These approaches transform tasks at other 92 levels into graph-level tasks by leveraging induced graphs. The All-in-One framework enhances 93 connectivity by adding links between the prompt graph and the original graph, whereas GraphPrompt 94 inserts the prompt token into graph nodes through element-wise multiplication. On the other hand, 95 LLM-based frameworks [30, 51, 28] utilize the power of LLMs to learn from different levels, but 96 they require graphs with text attributes, which are often not applicable in most anomaly detection 97 scenarios. Additionally, some multi-task GNN efforts [22] focus on multiple self-supervised specific 98 objectives (such as similarity and mutual information) as independent tasks, which are not suitable 99 for unifying GAD with multi-level label supervision and prediction. 

- 100 **Notation.** Let _G_ = _{V, E,_ _**X** }_ denote a connected undirected graph, where _V_ = _{v_ 1 _, v_ 2 _, ..., vN }_ is 101 the set of _N_ nodes, _E_ = _{eij}_ is the set of edges, and _**X** ∈_ R _[n][×][F]_ is node features. Let _**A**_ be the 102 corresponding adjacency matrix, _**D**_ be the degree matrix with _**D** ii_ =[�] _j_ _**[A]**[ij]_[.][Laplacian matrix] _**[ L]**_[ is] 103 then defined as _**D** −_ _**A**_ (regular) or as _**I** −_ _**D**[−]_ 2[1] _**AD**[−]_ 2[1] (normalized), where _**I**_ is an identity matrix. 104 The Laplacian matrix is a symmetric matrix and can be eigen-decomposed as _**L**_ = _**U**_ **Λ** _**U**[T]_ , where 105 the diagonal matrix **Λ** consists of real eigenvalues (graph spectrum). Besides, we define the subgraph 106 as _Gi_ centered on node _vi_ and our sampled subgraph for node _vi_ as _Si_ . 

- 107 **Problem Formulation.** The multi-level graph anomaly detection problem introduces a more universal 108 challenge compared to traditional single-level approaches, described as follows: 109 **Definition 2.1** (Multi-level GAD) **.** _Given a training set T r_ ( _N , E, G_ ) _containing nodes, edges, and_ 110 _graphs with arbitrary labels at any of these levels, the goal is to train a unified model to predict_ 111 _anomalies in a test set T e_ ( _N , E, G_ ) _, which also contains arbitrary labels at any of these levels._ 

- 111 

- 112 Note that our approach does not require the presence of labels at all three levels simultaneously. It is 113 feasible to have labels at one or more levels. Our proposed model aims to leverage the transferability 114 of information across different levels to enhance its predictive capability. 

- 115 

## **3 Methodology** 

- 116 This section details the proposed model UniGAD for multi-level GAD, comprising a GNN encoder, 117 MRQSampler, and GraphStitch Network, as shown in Fig. 1. Firstly, a shared pre-trained unsuper118 vised GNN encoder is utilized to learn a more generalized node representation. To unify multi-level 119 formats, the MRQSampler employs spectral sampling to extract subgraphs that contain the highest 120 amount of anomalous information from nodes and edges, thus converting tasks at all three levels 121 into graph-level tasks (Sec. 3.1). To unify multi-level training, the GraphStitch Network integrates 122 information from different levels, adjusts the amount of sharing required at each level, and harmonizes 123 conflicting training goals. (Sec. 3.2). 

## 124 **3.1 Spectral Subgraph Sampler for Unifying Multi-level Formats** 

- 125 In this subsection, we present the Maximum Rayleigh Quotient Subgraph Sampler (MRQSampler), 126 the core module of our unified framework. By sampling subgraphs of nodes or edges, we transform 127 node-level and edge-level tasks into graph-level tasks. Our sampler optimizes these subgraphs to 128 maximize the Rayleigh quotient, ensuring that the sampled subgraphs retain a higher concentration 129 of anomaly information. 

3 


![](raw/papers/4_Other_ML/Graph_paper_list/2024-NIPS-林依清-UniGAD_images/2024-NIPS-林依清-UniGAD.pdf-0004-00.png)


**----- Start of picture text -----**<br>
Message<br>Passing Sampling<br>Graph Rooted Subtree Sampled Subtree<br>**----- End of picture text -----**<br>


Figure 2: Message passing in GNNs and rooted subtree sampling. 

## 130 **3.1.1 Analysis of the Subgraph Sampling** 

- 131 **What is a suitable subgraph for GAD?** Existing methods on selecting subgraphs for target nodes or 132 edges often use straightforward approaches like _r_ -ego or _k_ -hop subgraphs [46]. However, the size of 133 the subgraph is critical for classification outcomes. If the subgraph is too large, it includes too many 134 irrelevant nodes, while if it is too small, it may not align effectively with graph-level tasks. 

- 135 To measure anomaly information in a subgraph, recent studies [48, 10] have identified a ‘right136 shift’ phenomenon in the spectral energy distribution, moving from low to higher frequencies. This 137 accumulated spectral energy can be quantified by the Rayleigh quotient [19]: 


![](raw/papers/4_Other_ML/Graph_paper_list/2024-NIPS-林依清-UniGAD_images/2024-NIPS-林依清-UniGAD.pdf-0004-05.png)


- 138 The following lemma [48] illustrates the relationship between the Rayleigh quotient _RQ_ ( _**x** ,_ _**L**_ ) and 139 anomaly information: 

- 140 **Lemma 1** (Tang, 2022) **.** _Rayleigh quotient RQ_ ( _**x** ,_ _**L**_ ) _, i.e. the accumulated spectral energy of the_ 141 _graph signal, is monotonically increasing with the anomaly degree._ 

- 142 Thus, for any node _vi_ , our sampling objective is to identify the induced subgraph with the highest 143 Rayleigh quotient containing the most anomaly information. 

- 144 **Where to Sample Subgraph From?** To preserve the properties of target nodes, it is essential to 145 sample subgraphs centered around these nodes, capturing key surrounding nodes. The most intuitive 146 methods are _r_ -ego graphs or _k_ -hop graphs. However, considering the message-passing mechanisms 147 of most GNNs [23, 53, 15], a classical work [54] provides valuable insight: 

- 148 **Lemma 2** (Xu, 2018) **.** _A GNN recursively updates each node’s feature vector through its rooted_ 149 _subtree structures to capture the network structure and features of surrounding nodes._ 

- 150 As shown in Fig. 2, the message-passing process of GNNs suggests that a rooted subtree centered on 151 the target node is more consistent with the GNN’s architecture. Therefore, we sample subgraphs from 152 these rooted subtree structures. The remaining question is: _**How to implement subgraph sampling**_ 153 _**based on the above?**_ To address this, we introduce a novel MRQSampler in the next subsection. 

- 153 

## 154 **3.1.2 Maximum Rayleigh Quotient Subgraph Sampler (MRQSampler)** 

- 155 Building on the motivation in Section 3.1.1, our approach involves sampling subgraphs for each 156 node starting from the rooted subtree with the node as its root. The target node is always included. 157 We then select the subtree with the maximum Rayleigh quotient from all possible subtrees as the 158 representative subgraph for that node to ensure it contain the maximum anomaly information. We 159 formulate this as the following optimization problem: 


![](raw/papers/4_Other_ML/Graph_paper_list/2024-NIPS-林依清-UniGAD_images/2024-NIPS-林依清-UniGAD.pdf-0004-15.png)


> 160 where _G_ represents _k_ -depth rooted subtree from _v_ , and _S_ is a possible subgraph from _G_ . The first 161 constraint ensures the target node is included, and the second constraint ensures message passability. 

4 


![](raw/papers/4_Other_ML/Graph_paper_list/2024-NIPS-林依清-UniGAD_images/2024-NIPS-林依清-UniGAD.pdf-0005-00.png)


**----- Start of picture text -----**<br>
Decouple into  Decouple into<br>4 unconnected subtrees 9 independent nodes Definition of Sub-problem：<br>Down into Ø Compute the optimal nodeset with<br>Simpler Sub-problems Δ!"# in rooted tree.<br>Ø (Sub) Compute the optimal nodeset<br>with ∆!"#%&' in each child's rooted tree.<br>Rooted Subtree Each Child’s Subtree  Recursively solve the sub-problem<br>from the bottom up<br>? ?<br>?<br>?<br>? ? ? ? ? ? ? ?<br>**----- End of picture text -----**<br>


Figure 3: MRQSampler: (i) Derive the condition (Theorem 2) satisfied with the optimal subtree. (ii) Decompose the problem into simpler sub-problems by recursing through the tree depth to solve the optimal subtree with the dynamic programming (DP) algorithm. 

- 162 Generally, similar selecting subgraphs in this manner is considered an NP-Hard problem [56]. 163 However, leveraging the properties of trees, we propose an algorithm to solve **the optimal solution** . 

- 164 We first determine the conditions that increase a subgraph’s Rayleigh quotient when adding a node, 165 presented in the following theorem: 

- 166 **Theorem 1.** _For a graph G, let one of its subgraphs be S, and let its Rayleigh quotient be RQ_ ( _S_ ) _. If_ 167 _a new node vnew ∈G −S is added to S, the Rayleigh quotient RQ_ ( _S_ ) _will increase if and only if:_ 


![](raw/papers/4_Other_ML/Graph_paper_list/2024-NIPS-林依清-UniGAD_images/2024-NIPS-林依清-UniGAD.pdf-0005-05.png)


- 168 The proof of Theorem 1 can be found in Appendix A.1. We can extend this theorem from a single 169 new node _vnew_ to a new node set _Vnew_ , leading to the following corollary: 

- 170 **Corollary 1.** _For a graph G, let one of its subgraphs be S, and let its Rayleigh quotient be RQ_ ( _S_ ) _._ 171 _If a new nodeset Vnew ⊂G −S is added to S, the Rayleigh quotient RQ_ ( _S_ ) _will increase if and_ 172 _only if:_ 


![](raw/papers/4_Other_ML/Graph_paper_list/2024-NIPS-林依清-UniGAD_images/2024-NIPS-林依清-UniGAD.pdf-0005-08.png)


173 The proof details are also in Appendix A.2. While the above analysis can indeed increase the 174 Rayleigh quotient of the sampled subgraph, the sampling order may cause the results to fall into a 175 local optimum, which may not guarantee a globally optimal solution. To identify the nodes that must 176 be sampled in the optimal subgraph, we present the following theorem: 

- 177 **Theorem 2.** _For a graph G, let one of its subgraph be S, the S[∗] be its final optimal subgraph, and_ 178 _S ⊂S[∗] . For a new_ _**connected** nodeset V_[˜] _new ∩ S_ = _∅, it is contained in S[∗] when it satisfies:_ 


![](raw/papers/4_Other_ML/Graph_paper_list/2024-NIPS-林依清-UniGAD_images/2024-NIPS-林依清-UniGAD.pdf-0005-11.png)


- 179 We refer readers to Appendix A.3 for the rigorous proof. Through the above analysis, we derive 180 the conditions of the nodeset contained in the optimal subtree (Theorem 2). When _V_[˜] _new_ satisfies 181 Eq. (5), it always increases the Rayleigh quotient based on the current subgraph, ensuring that _Vnew_ 182 is contained in the optimal solution. Thus, we decouple the problem of finding the subgraph with 183 the maximum Rayleigh quotient into a process of finding the maximum ∆ _max_ ( _Vnew_ ) each time, 184 until adding any node/node set fails to increase the _RQ_ ( _S_ ). Following this, we design a dynamic 185 programming (DP) algorithm to ensure the optimal subset satisfies these conditions. 

- 186 **MRQSampler Algorithm.** We introduce the Maximum Rayleigh Quotient Subgraph Sampler 187 (MRQSampler), which uses dynamic programming (DP) to find the optimal solution. We break down 188 the computation for the central node into sub-problems, initially addressing each of its neighboring 189 nodes. In Fig. 3, we consider a 2-depth subtree and summarize our algorithm as follows: 

- 190 • For a rooted subtree with the target node (edge) as the root, its children are unconnected to 191 each other. Based on Theorem 2, we iteratively select the child (within its own rooted subtree) 192 that maximizes ∆ _max_ ( _V_[˜] _new_ ) until the conditions of Theorem 2 are no longer satisfied, i.e., the 193 Rayleigh quotient of the sampled subgraph no longer increases. 

5 


![](raw/papers/4_Other_ML/Graph_paper_list/2024-NIPS-林依清-UniGAD_images/2024-NIPS-林依清-UniGAD.pdf-0006-00.png)


**----- Start of picture text -----**<br>
Multi-level Multi-level<br>Layer1 Layer2<br>…<br>node node<br>[𝛼"",𝛼"!, 𝛼"#] [𝛼"",𝛼"!, 𝛼"#]<br>edge [𝛼!",𝛼!!, 𝛼!#] edge [𝛼!",𝛼!!, 𝛼!#] …<br>[𝛼#",𝛼#!, 𝛼##] [𝛼#",𝛼#!, 𝛼##]<br>graph GraphStitch Unit graph GraphStitch Unit …<br>Separate but Identical  Separate but Identical<br>Architecture Architecture<br>**----- End of picture text -----**<br>


Figure 4: GraphStitch network structure in UniGAD. Node level is highlighted. 

- 194 • For each child’s rooted subtree, it is treated as a sub-problem similar to the previous one, applies 195 the same paradigm, and calculates each layer in the tree recursively from the bottom up. 

- 196 For efficiency, this approach can obtain the subgraph with the maximum Rayleigh quotient of the 197 target node/edge’s rooted subtree while reducing the algorithmic complexity to _O_ ( _N_ log _N_ ). It can be 198 further accelerated in parallel since the computation for different nodes is independent. Additionally, 199 the sampling process only needs to be computed once in training and inference processes, minimally 200 impacting model efficiency. For the detailed pseudocode of the algorithm, see Appendix B. Note that 201 we use mean pooling for entire graphs, but for subgraphs, we use weighted pooling to highlight central 202 nodes/edges, with an exponential decay based on the number of hops to the central nodes/edges. This 203 method transforms node-level and edge-level tasks into graph-level tasks, ensuring that the most 204 anomaly information is retained in the sampled subgraphs. 

## 205 **3.2 GraphStitch Network for Unifying Multi-level Training** 

206 After obtaining graph representations, training them together through a fully connected layer can 207 negatively impact individual levels due to the inherent differences across different-level anomalies. 208 This can result in mediocre performance at all levels. A key challenge, therefore, is to facilitate 209 information transfer between multi-levels without compromising single-level effectiveness. Inspired 210 by work in the computer vision field [43], we introduce the novel GraphStitch Network to jointly 211 consider multi-level representations. 

212 Specifically, we train separate but identical networks for each level and use the GraphStitch unit 213 to combine these networks into a multi-level network, managing the degree of sharing required at 214 different levels. This approach aims to maintain single-level effectiveness while enhancing multi-level 215 information transfer. The network structure is illustrated in Fig. 4. 

216 To elaborate, we denote **e** _N_ , **e** _E_ , and **e** _G_ as the embeddings for nodes, edges, and graphs, respec217 tively. The node embedding **e** _N_ = ( **e** _nn,_ **e** _ne,_ **e** _ng_ ) _[⊤]_ consists of outputs from three separate but 218 identically structured networks specialized for nodes, edges, and graphs. Similarly, the edge and 219 graph embeddings are represented as **e** _E_ = ( **e** _en,_ **e** _ee,_ **e** _eg_ ) _[⊤]_ and **e** _G_ = ( **e** _gn,_ **e** _ge,_ **e** _gg_ ) _[⊤]_ . 

219 **e** _E_ 220 We define a GraphStitch operation as follows: 


![](raw/papers/4_Other_ML/Graph_paper_list/2024-NIPS-林依清-UniGAD_images/2024-NIPS-林依清-UniGAD.pdf-0006-09.png)


221 The sharing of representations is achieved by learning a linear combination of the outputs from the 222 three networks. This linear combination is parameterized using learnable _α_ . In particular, when 223 training data lacks a certain level, the influence of that level on other levels is defined as zero during 224 training but still retains the influence of other levels on this level. In this way, it allows the labels for 225 training and testing to be arbitrary. Besides, if all the cross terms ( _αne_ , _αng_ , _αen_ , _αeg_ , _αgn_ , _αge_ ) are 226 equal to 0 means that training the three networks jointly is equivalent to training them independently. 227 Finally, the embeddings for nodes, edges, and graphs are fed into three independent multi-layer 228 perceptrons (MLPs) to compute the abnormal probabilities _p[N] i_[,] _[ p][E] i_[, and] _[ p][G] i_[, respectively.] 

- 229 In addition to the GraphStitch structure, UniGAD optimizes the loss functions for multi-level tasks. 230 Specifically, the gradients of each level task’s loss may conflict in direction or magnitude, potentially 

6 

231 causing negative effects and resulting in worse performance compared to learning single-level tasks 232 individually. Therefore, UniGAD uses a multi-level weighted cross-entropy loss for training: 


![](raw/papers/4_Other_ML/Graph_paper_list/2024-NIPS-林依清-UniGAD_images/2024-NIPS-林依清-UniGAD.pdf-0007-01.png)


233 where _γ_ is the ratio of anomaly labels ( _yi_ = 1) to normal labels ( _yi_ = 0), and _β[{N][ ,][E][,][G}]_ are adaptive 234 weights for different tasks. We adopt a ‘Gradient Surgery’ approach [57] to adjust the _β[{N][ ,][E][,][G}]_ , 235 altering the gradients by projecting each onto the normal plane of the others. This prevents the 236 interfering components of the gradients from affecting the network and minimizes interference 237 among different-level GAD tasks. In this way, UniGAD ensures that each level remains relatively 238 independent while facilitating cross-passing of relevant information between multi-level tasks. 

## 239 

## **4 Experiments** 

240 In this section, we conduct experiments to evaluate our UniGAD with node-level, edge-level, and 241 graph-level tasks by answering the following questions: **Q1:** How effective is UniGAD in unifying 242 in multi-level anomaly detection? **Q2:** Can UniGAD transfer information across different levels in 243 zero-shot learning? **Q3:** What are the contributions of the modular design in the UniGAD model? 

## 244 **4.1 Experimental Setup** 

245 **Datasets.** We consider a total of 13 datasets, including both single-graph datasets and multi-graph 246 datasets. Six single-graph datasets are used to evaluate the performance of unifying node-level and 247 edge-level tasks: Reddit, Weibo, Amazon, Yelp, Tolokers, and Questions from the work [47], which 248 contain node-level anomaly labels. For edge anomaly labels, we generated them according to a 249 specific anomaly probability following the formula _P_ anom[(] _[i,j]_[)][=] _[ avg]_[(] _[P][ i]_ anom _[, P]_ anom _[ j]_[)][.][Seven multi-graph] 250 datasets are used to validate the performance of unifying node-level and graph-level tasks, including 251 BM-MN, BM-MS, BM-MT, MUTAG, MNIST0, MNIST1, and T-Group. The first six datasets are 252 from [33] and a large-scale social group dataset T-Group from [25], containing both node anomaly 253 labels and graph anomaly labels. Statistical data for these datasets can be found in Appendix C. 

254 **Baselines.** To comprehensively compare with traditional single-level tasks, we consider nine repre255 sentative node-level methods: GCN [23], GIN [54], GraphSAGE [15], SGC [53], GAT [49], BernNet 256 [17], PNA [6], AMNet [5], and BWGNN [48]. Given the limited work on edge anomalies, we adapt 257 a concatenated strategy [61] from link prediction, resulting in nine corresponding edge-level methods: 258 GCNE, GINE, GSAGEE, SGCE, GATE, BernE, PNAE, AME, and BWE. For graph-level anomaly 259 detection, we consider six state-of-the-art methods: OCGIN [62], OCGTL [44], GLocalKD [40], 260 iGAD [60], GmapAD [42], and RQGNN [10]. Additionally, to compare multi-task models, we 261 include two recent multi-task graph prompt methods: GraphPrompt [35] and All-in-One [46]. While 262 these methods were not originally proposed for joint multi-task training, we adapt their ideas and 263 develop multi-task versions, GraphPrompt-U and All-in-One-U, for our comparison. 

264 **Implementations.** We evaluate two metrics: AUROC (Sec. 4) and Macro F1-score (Appendix. E), 265 similar to previous work [48]. For each result, we conduct 5 runs and report the mean results. In 266 Uni-GAD, we choose two backbone GNN encoders: GCN [23] and BWGNN [48]. We use a shared 267 graph pre-training method, GraphMAE [20], to obtain a more generalized node representation. To 268 avoid data leakage, for single-graph datasets, edges between the training set and the test set are not 269 considered; for multi-graph datasets, the training set and the test set consist of different graphs. More 270 details on the implementation can be found in the Appendix C. 

## 271 **4.2 Multi-Level Performance Comparison (RQ1)** 

272 To compare the performance of multi-level anomaly detection, we conduct experiments under two 273 settings. For the single-graph datasets, we compare the performance of unified training on node-level 274 and edge-level data. For the multi-graph datasets, we compare the performance of unified training on 275 node-level and graph-level data. 

276 **Node-level and edge-level jointly.** We first evaluate the performance of unified training on node-level 277 and edge-level data. We compare UniGAD against three groups of GNN models mentioned above: 

7 

Table 1: Comparison of unified performance (AUROC) at both node and edge levels with different single-level methods, multi-task methods, and our proposed method. 

||**Dataset**<br>**Task-level**|**Reddit**<br>**Node**<br>**Edge**|**Weibo**<br>**Node**<br>**Edge**|**Amazon**<br>**Node**<br>**Edge**|**Yelp**<br>**Node**<br>**Edge**|**Tolokers**<br>**Node**<br>**Edge**|**Questions**<br>**Node**<br>**Edge**|
|---|---|---|---|---|---|---|---|
|Node-Level|GCN<br>GIN<br>GraphSAGE<br>SGC<br>GAT<br>BernNet<br>PNA<br>AMNet<br>BWGNN|62_._60<br>/<br>65_._59<br>/<br>62_._25<br>/<br>52_._12<br>/<br>65_._87<br>/<br>66_._68<br>/<br>65_._28<br>/<br>68_._31<br>/<br>64_._65<br>/|97_._97<br>/<br>95_._64<br>/<br>94_._45<br>/<br>97_._71<br>/<br>94_._40<br>/<br>93_._93<br>/<br>97_._43<br>/<br>94_._17<br>/<br>97_._42<br>/|82_._37<br>/<br>92_._17<br>/<br>84_._53<br>/<br>80_._24<br>/<br>96_._24<br>/<br>96_._62<br>/<br>81_._41<br>/<br>97_._31<br>/<br>97_._80<br>/|57_._62<br>/<br>74_._46<br>/<br>82_._12<br>/<br>53_._03<br>/<br>77_._40<br>/<br>81_._48<br>/<br>71_._81<br>/<br>81_._42<br>/<br>83_._11<br>/|75_._21<br>/<br>75_._15<br>/<br>79_._74<br>/<br>69_._51<br>/<br>78_._90<br>/<br>76_._68<br>/<br>75_._82<br>/<br>76_._67<br>/<br>80_._51<br>/|70_._15<br>/<br>69_._13<br>/<br>72_._47<br>/<br>70_._59<br>/<br>71_._38<br>/<br>70_._28<br>/<br>71_._78<br>/<br>68_._63<br>/<br>70_._25<br>/|
|Edge-level|GCNE<br>GINE<br>GSAGEE<br>SGCE<br>GATE<br>BernE<br>PNAE<br>AME<br>BWE|/<br>63_._10<br>/<br>67_._36<br>/<br>**67**_._**52**<br>/<br>53_._36<br>/<br>67_._07<br>/<br>65_._57<br>/<br>64_._15<br>/<br>66_._73<br>/<br>67_._39|/<br>99_._03<br>/<br>98_._09<br>/<br>98_._67<br>/<br>98_._55<br>/<br>97_._92<br>/<br>97_._87<br>/<br>99_._10<br>/<br>97_._08<br>/<br>98_._93|/<br>78_._63<br>/<br>79_._74<br>/<br>78_._92<br>/<br>76_._41<br>/<br>90_._20<br>/<br>89_._60<br>/<br>75_._71<br>/<br>89_._36<br>/<br>91_._61|/<br>57_._80<br>/<br>67_._58<br>/<br>73_._30<br>/<br>52_._02<br>/<br>72_._96<br>/<br>73_._93<br>/<br>67_._98<br>/<br>73_._69<br>/<br>75_._63|/<br>73_._59<br>/<br>69_._27<br>/<br>**76**_._**98**<br>/<br>70_._59<br>/<br>71_._92<br>/<br>73_._39<br>/<br>75_._09<br>/<br>71_._99<br>/<br>75_._66|/<br>79_._05<br>/<br>80_._75<br>/<br>**87**_._**51**<br>/<br>74_._24<br>/<br>81_._64<br>/<br>84_._78<br>/<br>84_._05<br>/<br>84_._93<br>/<br>85_._00|
|Multi-task|GraphPrompt-U<br>All-in-One-U|50_._03<br>49_._78<br>51_._35<br>54_._10|55_._29<br>50_._71<br>48_._61<br>52_._63|OOT<br>OOT<br>OOT<br>OOT|OOT<br>OOT<br>OOT<br>OOT|51_._24<br>49_._66<br>50_._41<br>49_._29|55_._16<br>50_._01<br>51_._49<br>64_._24|
|UniGAD<br>(Ours)|UniGAD - GCN<br>UniGAD - BWG|**71**_._**65**<br>65_._46<br>68_._73<br>61_._33|**99**_._**32**<br>**99**_._**12**<br>99_._24<br>99_._00|82_._92<br>80_._04<br>**97**_._**84**<br>**92**_._**18**|63_._22<br>61_._74<br>77_._26<br>72_._89<br>**86**_._**23**<br>**79**_._**05**<br>**80**_._**62**<br>74_._85||**73**_._**92**<br>74_._72<br>70_._97<br>73_._45|



Table 2: Comparison of unified performance (AUROC) at both node and graph levels with different single-level methods, multi-task methods, and our proposed method. 

||**Dataset**<br>**Task-level**|**BM-MN**<br>**Node**<br>**Graph**|**BM-MS**<br>**Node**<br>**Graph**|**BM-MT**<br>**Node**<br>**Graph**|**MUTAG**<br>**Node**<br>**Graph**|**MNIST0**<br>**Node**<br>**Graph**|**MNIST1**<br>**Node**<br>**Graph**|**T-Group**<br>**Node**<br>**Graph**|
|---|---|---|---|---|---|---|---|---|
|Node-level|GCN<br>GIN<br>GraphSAGE<br>SGC<br>GAT<br>BernNet<br>PNA<br>BWGNN|86_._31<br>/<br>56_._73<br>/<br>50_._00<br>/<br>50_._27<br>/<br>58_._47<br>/<br>60_._06<br>/<br>72_._96<br>/<br>93_._05<br>/|90_._17<br>/<br>50_._41<br>/<br>50_._00<br>/<br>50_._87<br>/<br>62_._52<br>/<br>65_._58<br>/<br>55_._19<br>/<br>87_._22<br>/|92_._30<br>/<br>54_._90<br>/<br>49_._95<br>/<br>49_._44<br>/<br>65_._72<br>/<br>59_._18<br>/<br>75_._61<br>/<br>88_._97<br>/|99_._38<br>/<br>99_._39<br>/<br>99_._26<br>/<br>89_._19<br>/<br>99_._42<br>/<br>98_._97<br>/<br>98_._76<br>/<br>99_._50<br>/|94_._10<br>/<br>93_._55<br>/<br>**99**_._**99**<br>/<br>86_._97<br>/<br>99_._90<br>/<br>**99**_._**99**<br>/<br>99_._80<br>/<br>**99**_._**99**<br>/|93_._84<br>/<br>93_._49<br>/<br>**99**_._**99**<br>/<br>86_._97<br>/<br>**99**_._**99**<br>/<br>**99**_._**99**<br>/<br>99_._87<br>/<br>**99**_._**99**<br>/|91_._81<br>/<br>61_._51<br>/<br>64_._15<br>/<br>82_._55<br>/<br>78_._17<br>/<br>93_._85<br>/<br>55_._66<br>/<br>94_._81<br>/|
|Graph-level|OCGIN<br>OCGTL<br>GLocalKD<br>iGAD<br>GmapAD<br>RQGNN|/<br>98_._46<br>/<br>98_._48<br>/<br>92_._36<br>/<br>91_._68<br>/<br>50_._00<br>/<br>**98**_._**79**|/<br>81_._97<br>/<br>83_._17<br>/<br>77_._25<br>/<br>96_._68<br>/<br>50_._00<br>/<br>97_._98|/<br>58_._05<br>/<br>59_._99<br>/<br>53_._23<br>/<br>99_._14<br>/<br>50_._00<br>/<br>**99**_._**83**|/<br>89_._50<br>/<br>92_._19<br>/<br>72_._77<br>/<br>96_._28<br>/<br>75_._48<br>/<br>96_._41|/<br>57_._24<br>/<br>59_._35<br>/<br>66_._69<br>/<br>OOM<br>/<br>OOM<br>/<br>96_._62|/<br>86_._15<br>/<br>93_._45<br>/<br>57_._42<br>/<br>OOM<br>/<br>OOM<br>/<br>95_._57|/<br>64.53<br>/<br>46.77<br>/<br>78.53<br>/<br>OOM<br>/<br>OOM<br>/<br>73.90|
|Multi-task|GraphPrompt-U<br>All-in-One-U|51_._59<br>46_._85<br>67_._87<br>3_._21|50_._54<br>48_._67<br>54_._70<br>19_._42|51_._42<br>49_._38<br>69_._70<br>45_._89|97_._08<br>68_._23<br>50_._63<br>48_._98|OOT<br>OOT<br>OOT<br>OOT|OOT<br>OOT<br>OOT<br>OOT|OOT<br>OOT<br>OOT<br>OOT|
|UniGAD<br>(Ours)|UniGAD - GCN<br>UniGAD - BWG|**99**_._**75**<br>94_._29<br>92_._60<br>68_._74|**99**_._**60**<br>**99**_._**67**<br>93_._30<br>68_._55|**97**_._**62**<br>68_._07<br>90_._76<br>56_._01|99_._50<br>96_._33<br>**99**_._**54**<br>**96**_._**73**|97_._93<br>98_._99<br>**99**_._**99**<br>**99**_._**61**|98_._11<br>99_._59<br>**99**_._**99**<br>**99**_._**98**|95_._57<br>88_._73<br>**96**_._**19**<br>**88**_._**78**|



278 node-level models, edge-level models, and multi-task graph learning methods. Table 1 reports the 279 AUROC of each model on six datasets, with the best result on each dataset highlighted in boldface. 280 Overall, we find that UniGAD achieves state-of-the-art performance in nearly all scenarios. UniGAD 281 outperforms single-level specialized models, indicating that unified training with UniGAD leverages 282 information from other levels to enhance the performance of individual tasks. Multi-task approaches 283 (GraphPrompt-U and All-in-One-U) tend to negatively impact multi-task performance, potentially 284 because they are unable to effectively handle different types of anomaly label supervision. 

285 **Node-level and graph-level jointly.** We then evaluate the unified training of node-level and graph286 level tasks under similar settings. Table 2 shows the results, and UniGAD achieves state-of-the-art 287 performance in nearly all scenarios. Our observations are as follows. First, there is a multi-level 288 synergy in UniGAD, where strong performance in one task benefits the performance of other tasks. 289 For example, in MNIST-0 and MNIST-1, compared to other graph-level GAD methods, UniGAD 290 significantly boosts graph-level performance by leveraging strong node-level results. Second, Uni291 GAD performs better on large graphs, likely because graph structure plays a more significant role in 292 smaller datasets. However, the backbones of UniGAD (GCN, BWGNN) are primarily node-level 293 models, which may not effectively encode graph-level structural information. This limitation’s impact 294 diminishes in large-scale graph datasets. Besides, methods like GraphPrompt-U and All-in-One-U 295 often run out of time (OOT) with large datasets because they redundantly learn the same node 296 representations across different subgraphs, making processing impractically slow, especially for large 297 graph-level datasets like T-Group. UniGAD addresses this issue by using a shared GNN encoder 298 across tasks, avoiding redundant learning and enhancing efficiency. 

8 

Table 3: Zero-shot transferability (AUROC) at node and edge levels. 

|**Methods**|**Reddit**<br>**N**_→_**E**<br>**E**_→_**N**|**Weibo**<br>**N**_→_**E**<br>**E**_→_**N**|**Amazon**<br>**N**_→_**E**<br>**E**_→_**N**|**Yelp**<br>**N**_→_**E**<br>**E**_→_**N**|**Tolokers**<br>**N**_→_**E**<br>**E**_→_**N**|**Questions**<br>**N**_→_**E**<br>**E**_→_**N**|
|---|---|---|---|---|---|---|
|GraphPrompt-U<br>All-in-One-U|54_._06<br>47_._43<br>49_._23<br>49_._93|57_._03<br>42_._85<br>52_._22<br>54_._30|OOT<br>OOT<br>OOT<br>OOT|OOT<br>OOT<br>OOT<br>OOT|48_._56<br>51_._08<br>48_._34<br>50_._22|54_._26<br>51_._97<br>49_._83<br>51_._97|
|UniGAD - GCN<br>UniGAD - BWG|**59**_._**67**<br>**59**_._**46**<br>51_._91<br>57_._63|**98**_._**67**<br>91_._05<br>93_._80<br>**93**_._**35**|76_._20<br>82_._38<br>**82**_._**64**<br>**96**_._**41**|58_._28<br>60_._92<br>**75**_._**56**<br>**84**_._**08**|71_._45<br>73_._35<br>**74**_._**04**<br>**78**_._**49**|69_._54<br>**65**_._**37**<br>**71**_._**02**<br>62_._72|



Table 4: Zero-shot transferability (AUROC) at node and graph levels. 

|**Methods**|**BM-MN**<br>**N**_→_**G**<br>**G**_→_**N**|**BM-MS**<br>**N**_→_**G**<br>**G**_→_**N**|**BM-MT**<br>**N**_→_**G**<br>**G**_→_**N**|**MUTAG**<br>**N**_→_**G**<br>**G**_→_**N**|**MNIST0**<br>**N**_→_**G**<br>**G**_→_**N**|**T-Group**<br>**N**_→_**G**<br>**G**_→_**N**|
|---|---|---|---|---|---|---|
|GraphPrompt-U<br>All-in-One-U|50_._60<br>51_._57<br>**94**_._**39**<br>65_._69|51_._97<br>46_._95<br>52_._63<br>40_._88|46_._62<br>48_._06<br>44_._86<br>34_._27|59_._62<br>64_._26<br>61_._63<br>36_._13|OOT<br>OOT<br>OOT<br>OOT|OOT<br>OOT<br>OOT<br>OOT|
|UniGAD - GCN<br>UniGAD - BWG|80_._11<br>**87**_._**63**<br>67_._83<br>57_._56|**81**_._**49**<br>**90**_._**83**<br>65_._33<br>51_._34|**70**_._**13**<br>**79**_._**26**<br>55_._78<br>53_._41|65_._10<br>**88**_._**53**<br>**78**_._**05**<br>87_._03|**85**_._**24**<br>**70**_._**57**<br>74_._23<br>63_._70|**86**_._**86**<br>**75**_._**89**<br>86_._81<br>64_._81|



## 299 **4.3 The Transferability in Zero-Shot Learning (RQ2)** 

300 To assess the transfer capability of UniGAD, we explore zero-shot learning scenarios where labels 301 for a given level have never been exposed during training, as shown in Tables 3 and 4. In these 302 experiments, UniGAD is trained solely with labels from alternative levels. The notation **N** _→_ **E** 303 indicates using node labels to infer edge labels, with analogous notations for other label transfers. 304 Our findings indicate that in zero-shot scenarios, UniGAD outperforms existing multi-task prompt 305 learning methods. Moreover, the classification performance of UniGAD under zero-shot transfer 306 learning even surpasses those of leading baselines in supervised settings on Yelp and BM-MS. It 307 highlights the superior transfer capability of UniGAD across various GAD tasks. 

## 308 **4.4 Ablation Study (RQ3)** 

309 To investigate the contribution of each 310 module in UniGAD, we present the ab311 lation study results in Table 5. For the 312 sampler module, we compare the re313 sults without subgraph sampling (w/o 314 GS.), using a simple sampler with all 2- 315 hop neighbors (w 2hop.), and using ran316 dom sampling (w RS.). For the Graph317 Stitch module, we replace it with a uni318 fied MLP (w/o ST.). The results in319 dicate that both the subgraph sampler 

Table 5: Performance of UniGAD and its variants. 

||BM-MS|BM-MS|Reddit|Reddit|
|---|---|---|---|---|
||||||
|Metrics<br>Task-level|AUROC<br>node<br>graph|Macro F1<br>node<br>graph|AUROC<br>node<br>edge|Macro F1<br>node<br>edge|
||||||
|w/o GS.<br>w 2hop.<br>w RS.|97_._13<br>98_._99<br>97_._49<br>99_._94<br>93_._85<br>84_._92|80_._35<br>95_._79<br>67_._29<br>84_._20<br>85_._88<br>72_._21|68_._69<br>66_._06<br>67_._53<br>63_._62<br>65_._32<br>61_._85|53_._83<br>52_._78<br>51_._77<br>50_._69<br>52_._32<br>51_._03|
||||||
|w/o ST.|99_._94<br>95_._51|99_._47<br>84_._91|67_._74<br>65_._92|54_._35<br>52_._52|
||||||
|UniGAD|99_._60<br>99_._67|99_._57<br>95_._86|71_._65<br>65_._46|56_._70<br>53_._80|



320 (SG.) and the GraphStitch (ST.) modules enhance the overall performance of UniGAD. Additionally, 321 inappropriate subgraph sampling may perform worse than no subgraph sampling, likely due to the 322 loss of anomalous information during the sampling process. 

## 323 

## **5 Conclusion** 

324 This paper presents UniGAD, a unified graph anomaly detection framework that jointly addresses 325 anomalies at the node, edge, and graph levels. The model integrates two novel components: the 326 MRQSampler and the GraphStitch network. MRQSampler maximizes spectral energy to ensure 327 subgraphs capture critical anomaly information, addressing the challenge of unifying different graph 328 object formats. The GraphStitch Network unifies multi-level training by using identical networks for 329 nodes, edges, and graphs, facilitated by the GraphStitch Unit for effective information sharing. Our 330 thorough evaluations across 13 GAD datasets and comparisons with 17 graph learning methods show 331 that UniGAD not only surpasses existing models in various tasks but also exhibits strong zero-shot 332 transferability capabilities. A limitation of our paper is that the GNN encoder primarily focuses on 333 node-level embeddings, which may result in lost information about the graph structure. We leave the 334 exploration of multi-level tasks pre-training in the future works. 

9 

## 335 **References** 

- 336 [1] Esma Aïmeur, Sabrine Amri, and Gilles Brassard. Fake news, disinformation and misinforma337 tion in social media: a review. _Social Network Analysis and Mining_ , 13(1):30, 2023. 

- 338 [2] Leman Akoglu, Hanghang Tong, and Danai Koutra. Graph based anomaly detection and 339 description: a survey. _Data mining and knowledge discovery_ , 29(3):626–688, 2015. 

- 340 [3] Ketan Anand, Jay Kumar, and Kunal Anand. Anomaly detection in online social network: A 341 survey. In _2017 International Conference on Inventive Communication and Computational_ 342 _Technologies (ICICCT)_ , pages 456–459. IEEE, 2017. 

- 343 [4] Alessandro Bondielli and Francesco Marcelloni. A survey on fake news and rumour detection 344 techniques. _Information Sciences_ , 497:38–55, 2019. 

- 345 [5] Ziwei Chai, Siqi You, Yang Yang, Shiliang Pu, Jiarong Xu, Haoyang Cai, and Weihao Jiang. 346 Can abnormality be detected by graph neural networks? In _IJCAI_ , 2022. 

- 347 [6] Gabriele Corso, Luca Cavalleri, Dominique Beaini, Pietro Liò, and Petar Veliˇckovi´c. Principal 348 neighbourhood aggregation for graph nets. _Advances in Neural Information Processing Systems_ , 349 33:13260–13271, 2020. 

- 350 [7] Kelton AP Da Costa, João P Papa, Celso O Lisboa, Roberto Munoz, and Victor Hugo C 351 de Albuquerque. Internet of things: A survey on machine learning-based intrusion detection 352 approaches. _Computer Networks_ , 151:147–157, 2019. 

- 353 [8] Ailin Deng and Bryan Hooi. Graph neural network-based anomaly detection in multivariate 354 time series. In _Proceedings of the AAAI conference on artificial intelligence_ , pages 4027–4035, 355 2021. 

- 356 [9] Kaize Ding, Jundong Li, Rohit Bhanushali, and Huan Liu. Deep anomaly detection on attributed 357 networks. In _Proceedings of the 2019 SIAM International Conference on Data Mining_ . SIAM, 358 2019. 

- 359 [10] Xiangyu Dong, Xingyi Zhang, and Sibo Wang. Rayleigh quotient graph neural networks for 360 graph-level anomaly detection. _arXiv preprint arXiv:2310.02861_ , 2023. 

- 361 [11] Yingtong Dou, Zhiwei Liu, Li Sun, Yutong Deng, Hao Peng, and Philip S Yu. Enhancing graph 362 neural network-based fraud detectors against camouflaged fraudsters. In _CIKM_ , pages 315–324, 363 2020. 

- 364 [12] Jingcan Duan, Siwei Wang, Pei Zhang, En Zhu, Jingtao Hu, Hu Jin, Yue Liu, and Zhibin Dong. 365 Graph anomaly detection via multi-scale contrastive learning networks with augmented view. 366 _arXiv preprint arXiv:2212.00535_ , 2022. 

- 367 [13] Anuroop Gaddam, Tim Wilkin, Maia Angelova, and Jyotheesh Gaddam. Detecting sensor 368 faults, anomalies and outliers in the internet of things: A survey on the challenges and solutions. 369 _Electronics_ , 9(3):511, 2020. 

- 370 [14] Yuan Gao, Xiang Wang, Xiangnan He, Zhenguang Liu, Huamin Feng, and Yongdong Zhang. 371 Addressing heterophily in graph anomaly detection: A perspective of graph spectrum. In 372 _Proceedings of the ACM Web Conference_ , 2023. 

- 373 [15] William L Hamilton, Rex Ying, and Jure Leskovec. Inductive representation learning on large 374 graphs. In _NeurIPS_ , 2017. 

- 375 [16] Jiawei Han, Micheline Kamber, and Jian Pei. _Data Mining: Concepts and Techniques, 3rd_ 376 _edition_ . Morgan Kaufmann, 2011. 

- 377 [17] Mingguo He, Zhewei Wei, Hongteng Xu, et al. Bernnet: Learning arbitrary graph spectral 378 filters via bernstein approximation. _Advances in Neural Information Processing Systems_ , 379 34:14239–14251, 2021. 

- 380 [18] Waleed Hilal, S Andrew Gadsden, and John Yawney. Financial fraud: a review of anomaly 381 detection techniques and recent advances. _Expert systems With applications_ , 193:116429, 2022. 

- 382 [19] Roger A Horn and Charles R Johnson. _Matrix analysis_ . Cambridge university press, 2012. 

- 383 [20] Zhenyu Hou, Xiao Liu, Yukuo Cen, Yuxiao Dong, Hongxia Yang, Chunjie Wang, and Jie Tang. 384 Graphmae: Self-supervised masked graph autoencoders. In _Proceedings of the 28th ACM_ 385 _SIGKDD Conference on Knowledge Discovery and Data Mining_ , pages 594–604, 2022. 

10 

- 386 [21] Qian Huang, Hongyu Ren, Peng Chen, Gregor Kržmanc, Daniel Zeng, Percy S Liang, and Jure 387 Leskovec. Prodigy: Enabling in-context learning over graphs. _Advances in Neural Information_ 388 _Processing Systems_ , 36, 2024. 389 [22] Mingxuan Ju, Tong Zhao, Qianlong Wen, Wenhao Yu, Neil Shah, Yanfang Ye, and Chuxu 390 Zhang. Multi-task self-supervised graph neural networks enable stronger task generalization. 391 _arXiv preprint arXiv:2210.02016_ , 2022. 

- 392 [23] Thomas N. Kipf and Max Welling. Semi-supervised classification with graph convolutional 393 networks. In _ICLR_ , 2017. 

- 394 [24] Ao Li, Zhou Qin, Runshi Liu, Yiqun Yang, and Dong Li. Spam review detection with graph con395 volutional networks. In _Proceedings of the 28th ACM International Conference on Information_ 396 _and Knowledge Management_ , pages 2703–2711, 2019. 

- 397 [25] Jia Li, Yu Rong, Hong Cheng, Helen Meng, Wenbing Huang, and Junzhou Huang. Semi398 supervised graph classification: A hierarchical graph perspective. In _The World Wide Web_ 399 _Conference_ , pages 972–982, 2019. 

- 400 [26] Jundong Li, Harsh Dani, Xia Hu, and Huan Liu. Radar: Residual analysis for anomaly detection 401 in attributed networks. In _IJCAI_ , pages 2152–2158, 2017. 

- 402 [27] Yuening Li, Xiao Huang, Jundong Li, Mengnan Du, and Na Zou. Specae: Spectral autoencoder 403 for anomaly detection in attributed networks. In _CIKM_ , pages 2233–2236, 2019. 

- 404 [28] Yuhan Li, Zhixun Li, Peisong Wang, Jia Li, Xiangguo Sun, Hong Cheng, and Jeffrey Xu Yu. A 405 survey of graph meets large language model: Progress and future directions. _arXiv preprint_ 406 _arXiv:2311.12399_ , 2023. 

- 407 [29] Fanzhen Liu, Xiaoxiao Ma, Jia Wu, Jian Yang, Shan Xue, Amin Beheshti, Chuan Zhou, Hao 408 Peng, Quan Z Sheng, and Charu C Aggarwal. Dagad: Data augmentation for graph anomaly 409 detection. _arXiv preprint arXiv:2210.09766_ , 2022. 

- 410 [30] Hao Liu, Jiarui Feng, Lecheng Kong, Ningyue Liang, Dacheng Tao, Yixin Chen, and Muhan 411 Zhang. One for all: Towards training one graph model for all classification tasks. _arXiv preprint_ 412 _arXiv:2310.00149_ , 2023. 

- 413 [31] Kay Liu, Yingtong Dou, Yue Zhao, Xueying Ding, Xiyang Hu, Ruitong Zhang, Kaize Ding, 414 Canyu Chen, Hao Peng, Kai Shu, Lichao Sun, Jundong Li, George H Chen, Zhihao Jia, and 415 Philip S Yu. Bond: Benchmarking unsupervised outlier node detection on static attributed 416 graphs. In _Advances in Neural Information Processing Systems_ , volume 35, 2022. 

- 417 [32] Yang Liu, Xiang Ao, Zidi Qin, Jianfeng Chi, Jinghua Feng, Hao Yang, and Qing He. Pick and 418 choose: A gnn-based imbalanced learning approach for fraud detection. In _Proceedings of the_ 419 _Web Conference 2021_ , 2021. 

- 420 [33] Yixin Liu, Kaize Ding, Qinghua Lu, Fuyi Li, Leo Yu Zhang, and Shirui Pan. Towards self421 interpretable graph-level anomaly detection. _Advances in Neural Information Processing_ 422 _Systems_ , 36, 2024. 

- 423 [34] Yixin Liu, Zhao Li, Shirui Pan, Chen Gong, Chuan Zhou, and George Karypis. Anomaly 424 detection on attributed networks via contrastive self-supervised learning. _IEEE transactions on_ 425 _neural networks and learning systems_ , 2021. 

- 426 [35] Zemin Liu, Xingtong Yu, Yuan Fang, and Xinming Zhang. Graphprompt: Unifying Pre-Training 427 and Downstream Tasks for Graph Neural Networks. In _The Web Conference_ , pages 417–428, 428 2023. 

- 429 [36] Zhiwei Liu, Yingtong Dou, Philip S Yu, Yutong Deng, and Hao Peng. Alleviating the inconsis430 tency problem of applying graph neural network to fraud detection. In _SIGIR_ , pages 1569–1572, 431 2020. 

- 432 [37] Zhiyuan Liu, Chunjie Cao, and Jingzhang Sun. Mul-gad: a semi-supervised graph anomaly 433 detection framework via aggregating multi-view information. _arXiv preprint arXiv:2212.05478_ , 434 2022. 

- 435 [38] Zhiyuan Liu, Chunjie Cao, Fangjian Tao, and Jingzhang Sun. Revisiting graph contrastive 436 learning for anomaly detection. _arXiv preprint arXiv:2305.02496_ , 2023. 

- 437 [39] Ziqi Liu, Chaochao Chen, Longfei Li, Jun Zhou, Xiaolong Li, Le Song, and Yuan Qi. Geniepath: 438 Graph neural networks with adaptive receptive paths. In _Proceedings of the AAAI Conference_ 439 _on Artificial Intelligence_ , pages 4424–4431, 2019. 

11 

- 440 [40] Rongrong Ma, Guansong Pang, Ling Chen, and Anton van den Hengel. Deep graph-level 441 anomaly detection by glocal knowledge distillation. In _Proceedings of the fifteenth ACM_ 442 _international conference on web search and data mining_ , pages 704–714, 2022. 

- 443 [41] Xiaoxiao Ma, Jia Wu, Shan Xue, Jian Yang, Chuan Zhou, Quan Z Sheng, Hui Xiong, and 444 Leman Akoglu. A comprehensive survey on graph anomaly detection with deep learning. _IEEE_ 445 _Transactions on Knowledge and Data Engineering_ , 2021. 

- 446 [42] Xiaoxiao Ma, Jia Wu, Jian Yang, and Quan Z Sheng. Towards graph-level anomaly detection 447 via deep evolutionary mapping. In _Proceedings of the 29th ACM SIGKDD Conference on_ 448 _Knowledge Discovery and Data Mining_ , pages 1631–1642, 2023. 

- 449 [43] Ishan Misra, Abhinav Shrivastava, Abhinav Gupta, and Martial Hebert. Cross-stitch networks 450 for multi-task learning. In _Proceedings of the IEEE conference on computer vision and pattern_ 451 _recognition_ , pages 3994–4003, 2016. 

- 452 [44] Chen Qiu, Marius Kloft, Stephan Mandt, and Maja Rudolph. Raising the bar in graph-level 453 anomaly detection. _arXiv preprint arXiv:2205.13845_ , 2022. 

- 454 [45] Amit Roy, Juan Shu, Jia Li, Carl Yang, Olivier Elshocht, Jeroen Smeets, and Pan Li. Gad-nr: 455 Graph anomaly detection via neighborhood reconstruction. In _Proceedings of the 17th ACM_ 456 _International Conference on Web Search and Data Mining_ , pages 576–585, 2024. 

- 457 [46] Xiangguo Sun, Hong Cheng, Jia Li, Bo Liu, and Jihong Guan. All in one: Multi-task prompting 458 for graph neural networks. In _Proceedings of the 29th ACM SIGKDD Conference on Knowledge_ 459 _Discovery and Data Mining_ , pages 2120–2131, 2023. 

- 460 [47] Jianheng Tang, Fengrui Hua, Ziqi Gao, Peilin Zhao, and Jia Li. Gadbench: Revisiting and 461 benchmarking supervised graph anomaly detection. _Advances in Neural Information Processing_ 462 _Systems_ , 36, 2024. 

- 463 [48] Jianheng Tang, Jiajin Li, Ziqi Gao, and Jia Li. Rethinking graph neural networks for anomaly 464 detection. In _International Conference on Machine Learning_ , 2022. 

- 465 [49] Petar Veliˇckovi´c, Guillem Cucurull, Arantxa Casanova, Adriana Romero, Pietro Lio, and Yoshua 466 Bengio. Graph attention networks. In _ICLR_ , 2017. 

- 467 [50] Daixin Wang, Jianbin Lin, Peng Cui, Quanhui Jia, Zhen Wang, Yanming Fang, Quan Yu, Jun 468 Zhou, Shuang Yang, and Yuan Qi. A semi-supervised graph attentive network for financial 469 fraud detection. In _ICDM_ , pages 598–607. IEEE, 2019. 

- 470 [51] Jianing Wang, Junda Wu, Yupeng Hou, Yao Liu, Ming Gao, and Julian McAuley. Instructgraph: 471 Boosting large language models via graph-centric instruction tuning and preference alignment. 472 _arXiv preprint arXiv:2402.08785_ , 2024. 

- 473 [52] Mark Weber, Giacomo Domeniconi, Jie Chen, Daniel Karl I Weidele, Claudio Bellei, Tom 474 Robinson, and Charles E Leiserson. Anti-money laundering in bitcoin: Experimenting with 475 graph convolutional networks for financial forensics. _arXiv preprint arXiv:1908.02591_ , 2019. 

- 476 [53] Felix Wu, Amauri Souza, Tianyi Zhang, Christopher Fifty, Tao Yu, and Kilian Weinberger. 477 Simplifying graph convolutional networks. In _ICML_ , pages 6861–6871, 2019. 

- 478 [54] Keyulu Xu, Weihua Hu, Jure Leskovec, and Stefanie Jegelka. How powerful are graph neural 479 networks? _ICLR_ , 2019. 

- 480 [55] Zhiming Xu, Xiao Huang, Yue Zhao, Yushun Dong, and Jundong Li. Contrastive attributed 481 network anomaly detection with data augmentation. In _Proceedings of the PAKDD_ , 2022. 

- 482 [56] Kuo Yang, Zhengyang Zhou, Xu Wang, Pengkun Wang, Limin Li, and Yang Wang. Raye-sub: 483 Countering subgraph degradation via perfect reconstruction. 

- 484 [57] Tianhe Yu, Saurabh Kumar, Abhishek Gupta, Sergey Levine, Karol Hausman, and Chelsea Finn. 485 Gradient surgery for multi-task learning. _Advances in Neural Information Processing Systems_ , 486 33:5824–5836, 2020. 

- 487 [58] Xingtong Yu, Chang Zhou, Yuan Fang, and Xinming Zhang. Multigprompt for multi-task 488 pre-training and prompting on graphs. _arXiv preprint arXiv:2312.03731_ , 2023. 

- 489 [59] Ge Zhang, Jia Wu, Jian Yang, Amin Beheshti, Shan Xue, Chuan Zhou, and Quan Z Sheng. 490 Fraudre: Fraud detection dual-resistant to graph inconsistency and imbalance. In _2021 IEEE_ 491 _International Conference on Data Mining (ICDM)_ , pages 867–876. IEEE, 2021. 

12 

- 492 [60] Ge Zhang, Zhenyu Yang, Jia Wu, Jian Yang, Shan Xue, Hao Peng, Jianlin Su, Chuan Zhou, 493 Quan Z Sheng, Leman Akoglu, et al. Dual-discriminative graph neural network for imbalanced 494 graph-level anomaly detection. _Advances in Neural Information Processing Systems_ , 35:24144– 495 24157, 2022. 

- 496 [61] Muhan Zhang. Graph neural networks: link prediction. _Graph Neural Networks: Foundations,_ 497 _Frontiers, and Applications_ , pages 195–223, 2022. 

- 498 [62] Lingxiao Zhao and Leman Akoglu. On using classification datasets to evaluate graph outlier 499 detection: Peculiar observations and new insights. _Big Data_ , 11(3):151–180, 2023. 

- 500 [63] Li Zheng, Zhenpeng Li, Jian Li, Zhao Li, and Jun Gao. Addgraph: Anomaly detection in 501 dynamic graph using attention-based temporal gcn. In _IJCAI_ , pages 4419–4425, 2019. 

- 502 [64] Yun Zhu, Jianhao Guo, and Siliang Tang. Sgl-pt: A strong graph learner with graph prompt 503 tuning. _arXiv preprint arXiv:2302.12449_ , 2023. 

13 

## 504 **A Proofs** 

505 **A.1 The proof of Theorem 1** 

506 _Proof._ For a new node _vnew_ , let _S[′]_ be the subgraph after the addition of _vnew_ . Based on the fact 507 that the definition of Rayleigh quotient _RQ_ ( _S_ ) = �( _p,q_ �) _∈Ep∈SS_[(] _[x][x][p]_[2] _p[−][x][q]_[)][2] , it need to satisfy the following 508 condition in order to increase the Rayleigh quotient _RQ_ ( _S[′]_ ) _> RQ_ ( _S_ ): 


![](raw/papers/4_Other_ML/Graph_paper_list/2024-NIPS-林依清-UniGAD_images/2024-NIPS-林依清-UniGAD.pdf-0014-03.png)


509 where[�] _vr∈S_[(] _[x][new][ −][x][r]_[)][2][represents the sum of the feature difference between the new node] _[ v][new]_ 510 and the connecting edge of the node _v_ in the subgraph _S_ . It is worth noting that these edges are 511 present in the original graph _G_ . Since both the numerator and denominator are positive numbers, the 512 Eq. (A.1) can be transformed into: 


![](raw/papers/4_Other_ML/Graph_paper_list/2024-NIPS-林依清-UniGAD_images/2024-NIPS-林依清-UniGAD.pdf-0014-05.png)


513 which can be obviously simplified to: 

514 We move the term with the new node to the same side of the equation and rearrange the Eq. (A.3), 515 and we obtain: 


![](raw/papers/4_Other_ML/Graph_paper_list/2024-NIPS-林依清-UniGAD_images/2024-NIPS-林依清-UniGAD.pdf-0014-08.png)


518 **A.2 The proof of Corollary 1** 


![](raw/papers/4_Other_ML/Graph_paper_list/2024-NIPS-林依清-UniGAD_images/2024-NIPS-林依清-UniGAD.pdf-0014-10.png)



![](raw/papers/4_Other_ML/Graph_paper_list/2024-NIPS-林依清-UniGAD_images/2024-NIPS-林依清-UniGAD.pdf-0014-11.png)


526 which is the same as Corollary 1 in Section 3.1.2. 

527 

14 

## 528 **A.3 The proof of Theorem 2** 

529 _Proof._ We define the nodeset _Vnew[∗]_[has the highest][ ∆] _[max]_[(] _[V][new]_[)][ and][ ∆] _[max]_[(] _[V][new]_[)] _[>][RQ]_[(] _[S]_[)][.][To] 530 prove that the _Vnew[∗]_[is contained in the optimal subgraph] _[ S][∗]_[,][we give the proof by contradiction.] 531 Assume that the negation of the statement is true, so there does not exist _Vnew[∗]_[in] _[ S][∗]_[.][We will discuss] 532 the issues based on two scenarios. 

533 In the first scenario, we assume that the current subgraph _S_ is already the optimal solution. Accord534 ing to Corollary 1, we find that adding _Vnew[∗]_[can][increase] _[RQ]_[(] _[S]_[)][since][it][satisfies][the][condition] 535 ∆ _max_ ( _Vnew_ ) _> RQ_ ( _S_ ). Therefore, it is obvious that the current set _S_ is not the optimal solution. 536 In the other scenario, we assume that there is another nodeset _Vnew[′]_[(] _[ V] new[′][∩S][∗]_[=] _[ ∅]_[), which together] 537 with the current subgraph _S_ + _Vnew[′]_[forms the optimal solution.][According to the corollary 1, we] 538 have 


![](raw/papers/4_Other_ML/Graph_paper_list/2024-NIPS-林依清-UniGAD_images/2024-NIPS-林依清-UniGAD.pdf-0015-03.png)


539 and it satisfies: 

540 To continue with the proof, we present a useful inequality first. 

541 **Lemma 3** (Dan’s Favorite Inequality) **.** _Let a_ 1 _, ..., an and b_ 1 _, ..., bn be positive numbers. Then_ 


![](raw/papers/4_Other_ML/Graph_paper_list/2024-NIPS-林依清-UniGAD_images/2024-NIPS-林依清-UniGAD.pdf-0015-07.png)


542 _Proof._ Here we give a classical proof, we have 


![](raw/papers/4_Other_ML/Graph_paper_list/2024-NIPS-林依清-UniGAD_images/2024-NIPS-林依清-UniGAD.pdf-0015-09.png)


543 So, 

544 One can similarly prove 

545 

546 Combining Lemma 3 and Eq. (A.9), we obtain the following inequality. 


![](raw/papers/4_Other_ML/Graph_paper_list/2024-NIPS-林依清-UniGAD_images/2024-NIPS-林依清-UniGAD.pdf-0015-14.png)


547 Analyzing the above equation reveals that the right side of the formula is _RQ_ ( _Vnew[′]_[+] _[ S]_[)][.][That is, for] 548 any _Vnew[′]_[, adding] _[ V] new[∗]_[still makes] _[ RQ]_[(] _[V] new[′]_[+] _[ S]_[)][ increasing according to the Corollary 1, which] 549 contradicts the assumption that _RQ_ ( _Vnew[′]_[+] _[ S]_[)][ is the optimal solution.] 


![](raw/papers/4_Other_ML/Graph_paper_list/2024-NIPS-林依清-UniGAD_images/2024-NIPS-林依清-UniGAD.pdf-0015-16.png)


550 However, identifying the maximum ∆ _max_ ( _Vnew_ ) from the _Vnew ⊆G −S_ is still a NP-hard problem. 551 We consider relaxing any nodesets _Vnew[′]_[to any] **[connected]**[nodesets] _[V] new[c]_[.][Any nodesets can be] 552 decomposed into several disconnected smaller nodesets, that is, _Vnew[′]_[=] _[ V] new[c]_[1] _[∪V] new[c]_[2] _[∪][. . .]_[.][Since] 553 there are no edges connecting these nodesets, the following decomposition formula can be derived. 

15 


![](raw/papers/4_Other_ML/Graph_paper_list/2024-NIPS-林依清-UniGAD_images/2024-NIPS-林依清-UniGAD.pdf-0016-00.png)


(A.16) 

554 Considering the condition that maximizes the Rayleigh quotient of any connected _Vnew[ci]_[,] 


![](raw/papers/4_Other_ML/Graph_paper_list/2024-NIPS-林依清-UniGAD_images/2024-NIPS-林依清-UniGAD.pdf-0016-03.png)


555 According to Lemma 3, Eq. (A.14) is still satisfied. Therefore, we derive that _Vnew[∗]_[is contained in] 556 the optimal solution. 

557 

## 558 **B The Pseudocode of MRQSampler Algorithm** 

559 We give the pseudocode of MRQSampler in Algorithm 1, which illustrates the algorithm for finding 560 the subgraph of a tree containing the root node that maximizes the Rayleigh quotient. Specifically, 561 for node _i_ , we focus on the k-hop spanning tree _T_ with _i_ as the root node, where the nodes in the 562 _l_ th level are all the nodes reachable in _l_ hops from _i_ and the edges between them are the same as in 563 the original graph. Note that by this definition, a node may appear in multiple levels or as a child 564 of different nodes, which we view as it being cloned multiple times. Therefore, we do not need to 565 explicitly construct or store _T_ , but simply follow breadth-first searching the k-hop subgraph without 566 memoization of visited nodes. 

567 For ease of computation, we store the optimal Rayleigh Quotient _S_ by its numerator _ai_ and denomi568 nator _bi_ (line 6-12). Next, we recursively calculate the result for each subtree rooted by every child of 569 the current root _i_ (line 13). To simplify complexity in the following steps, we store these results in a 570 sorted container (e.g. binary search tree) _Q_ at line 15. Next, we keep retrieving the subgraph with 571 the highest Rayleigh quotient value from Q and compute whether the Rayleigh quotient increases 572 after adding it to the current solution (line 19-22). According to Theorem 1, we obtain the optimal 573 node set _N_ for the current subtree with root _i_ when the candidate cannot make _S_ larger. However, 574 sets from subtrees that are not optimal for _i_ may still be selected at higher levels. Therefore, we also 575 need to keep track of those inferior subtrees and re-consider them when other subtrees that connect 576 to them are merged into the solution (line 27). If the optimal subtree _i_ is selected in the future at a 577 higher level, the influence of _x_ itself needs to be added to the connected inferior subtrees, which is 578 handled in line 30-35. Note that subtrees in _I_ are only considered as candidates when the optimal 579 subtree with root _i_ is selected at a higher layer (the “activation” in line 27). 

580 Assuming that the k-hop spanning tree _T_ has _K_ nodes, the time complexity of Algorithm _O_ ( _KlogK_ ), 581 since we will at worst examine each edge and sort them. Notice that the computation is irrelevant 582 between different nodes, it can be further accelerated by simultaneously processing multiple nodes. 583 In practice, we observe that the optimal choice of k-hop is typically <= 2, and thus the recursive 584 computation can be unrolled thus further improving the efficiency. 

## 585 

## **C Implementation Details** 

586 **Node-level Baselines. GCN** (Graph Convolutional Network [23]) leverages convolution operations 587 to propagate information from nodes to their neighbors. **SGC** (Simple Graph Convolution [53]) further 588 simplifies GCN by removing non-linearities and collapsing weight matrices between consecutive 589 layers to improve efficiency. **GIN** (Graph Isomorphism Network [54]) captures graph structures by 590 generating identical embeddings for structurally identical graphs, ensuring invariance to node label 591 permutations. **GraphSAGE** (Graph Sample and AggregatE [15]) generates node embeddings through 592 sampling and aggregating features from local neighborhoods, supporting inductive learning. **GAT** 593 (Graph Attention Networks [49]) incorporates an attention mechanism to assign varying importance 594 levels to different nodes during neighborhood aggregation, focusing on the most informative parts. 595 **PNA** (Principle Neighbor Aggregation [6]) combines multiple aggregators with degree-scalers for 

16 

**Algorithm 1** MRQSampler 

**Globals:** _x_ : node features; _T_ [ _i_ ] : child nodes of node _i_ in the tree; **Inputs:** _i_ := root of the subtree in current recursion; _p_ : parent node of _i_ ; **Returns:** _ai, bi_ : numerator/denominator of the maximum score; _N_ : optimal nodes of the subtree; _I_ : inferior candidates of the subtree. 

1: **function** `GetSampleRestult` ( _i, p_ ) 2: _N ←{} ▷_ Selected nodes 3: _Q ←_ `SortedSet()` _▷_ Candidates queue 4: _I ←{} ▷_ Inferior candidates 5: _U ←{} ▷_ Un-selected subtrees 6: **if** _p ̸_ = None **then** 7: _ai ←_ ( _x_ [ _i_ ] _− x_ [ _ip_ ])[2] _▷_ Initialize the numerator 8: **else** 9: _ai ←_ 0 _▷_ The root has no parents 10: **end if** 11: _bi ← x_ [ _i_ ][2] _▷_ Initialize the denominator 12: _S ← ai/bi ▷_ Initialize the best score 13: **for** _c_ in _T_ [ _i_ ] **do** 14: _Rc ←_ `GetSampleRestult` (c, i) _▷_ Result from the subtree 15: _Q_ . `insert([` _Rc, c_ `])` 16: _U_ . `insert(` _c_ `)` 17: **end for** 18: **while** _Q_ . `size()` _̸_ = 0 **do** 19: _Rc, c ← Q_ . `pop_largest()` 20: **if** _S >_ ( _ai_ + _Rc.ai_ ) _/_ ( _bi_ + _Rc.bi_ ) **then** _▷_ Optimization criterion 21: Break 22: **end if** 23: _U_ . `remove_if_exist(` _c_ `)` 24: _ai ← ai_ + _Rc.ai ▷_ Update the result 25: _bi ← bi_ + _Rc.bi_ 26: _S ← ai/bi_ 27: _Q_ . `insert(` _Rc.I_ `)` _▷_ Activate the inferior candidates 28: _N_ . `insert(` _Rc.N_ `)` 29: **end while** 30: **for** _c_ in _U_ **do** 31: _Rc ← Q_ . `find_and_remove(` _c_ `)` 32: _Rc.ai ← Rc.ai_ + ( _x_ [ _c_ ] _− x_ [ _i_ ])[2] 33: _Rc.bi ← Rc.bi_ + ( _x_ [ _c_ ])[2] 34: _Q_ . `insert(` _Rc_ `)` _▷_ Update the RQ value of inferior candidates 35: **end for** 36: _I ← Q_ 37: **return** _{ai, bi, N, I}_ 38: **end function** 

596 effective neighborhood aggregation. **AMNet** (Adaptive Multi-frequency Graph Neural Network 597 [5]) captures both low and high-frequency signals by stacking multiple BernNets [17], adaptively 598 combining signals of different frequencies. **BWGNN** (Beta Wavelet Graph Neural Network [48]) 599 employs the Beta kernel to tackle higher frequency anomalies with flexible band-pass filters. 

600 **Graph-level Baselines. OCGIN** [62] is a one-class graph-level anomaly detector based on a 601 graph isomorphism network that addresses performance fluctuations in general graph classification 602 methods. **OCGTL** [44] combines deep one-class classification with graph transformation learning. 603 **GlocalKD** learns rich global and local normal pattern information by joint distillation of graph 604 and node representations. **iGAD** [60] employs an attribute-aware GNN and a substructure-aware 605 deep random walk kernel to achieve dual-discriminative capability for anomalous attributes and 606 substructures. **GmapAD** [42] proposes an explainable graph mapping approach that projects graphs 

17 

607 into a latent space for effective anomaly detection. **RQGNN** [10] identifies differences in the spectral 608 energy distributions between anomalous and normal graphs. 

- 609 **Multi-task Baselines. GraphPrompt** [35] learns different task-specific prompt vectors for each 610 task, which are added to the graph-level representations by element-wise multiplication. **All-in-One** 611 [46] treats an extra subgraph as a prompt and merges it with the original graph by cross links. 

612 **Datasets.** In Table 6, we present detailed statistics of the datasets used in our experiments, including 613 the percentage of training data, the number of graphs, edges, nodes, feature dimensions, and the 614 proportions of abnormal nodes, edges, and graphs (Nodes _ab_ , Edges _ab_ , and Graphs _ab_ ). 

Table 6: Detailed statistics of the datasets used in our experiments. 

|**Dataset **|**Train% **|**# Graphs**|**# Edges**|**# Nodes**|**# Dims **|**Nodes**_ab_|**Edges**_ab_|**Graphs**_ab_|
|---|---|---|---|---|---|---|---|---|
|Reddit|40%|1|168,016|10,984|64|3.33%|2.72%|/|
|Weibo|40%|1|416,368|8,405|400|10.33%|5.71%|/|
|Amazon|70%|1|8,847,096|11,944|25|6.87%|2.49%|/|
|Yelp|70%|1|7,739,912|45,954|32|14.53%|13.89%|/|
|Tolokers|50%|1|530,758|11,758|10|21.82%|33.44%|/|
|Questions|50%|1|202,461|48,921|301|2.98%|7.50%|/|
|BM_MN|40%|700|40,032|12,911|1|48.91%|/|14.29%|
|BM_MS|40%|700|30,238|9,829|1|31.99%|/|14.29%|
|BM_MT|40%|700|32,042|10,147|1|34.49%|/|14.29%|
|MUTAG|40%|2,951|179,732|88,926|14|4.81%|/|34.40%|
|MNIST0|10%|70,000|41,334,380|4,939,668|5|35.46%|/|9.86%|
|MNIST1|10%|70,000|41,334,380|4,939,668|5|35.46%|/|11.25%|
|T-Group|40%|37,402|93,367,082|11,015,616|10|0.64%|/|4.26%|



615 **Hardware Specifications.** All our experiments were carried out on a Linux server equipped with 616 dual AMD EPYC 7763 64-core CPU processor, 256GB RAM, and an NVIDIA RTX 4090 GPU with 617 24GB memory. We mark the results as OOT (Out of Time) if the model training exceeds 1 day, and 618 as OOM (Out of Memory) if the model cannot fit into the 24GB GPU for a specific dataset. 

619 **Metrics.** In line with previous studies [48], we utilize two widely used metrics to evaluate the 620 performance of all methods: **F1-macro** and **AUROC** . **F1-macro** is the unweighted mean of the 621 F1-scores for the two classes, disregarding the imbalance ratio between normal and anomaly labels. 622 **AUROC** represents the area under the Receiver Operating Characteristic Curve. 

Table 7: Hyperparameters search space for UniGAD. 

|**Hyperparameter**|**Distribution**|
|---|---|
|learning rate|Range(5_−_4_,_10_−_2)|
|activation|[ReLU, LeakyReLU, Tanh]|
|hidden dimension|[16,32,64]|
|MRQSampler tree depth|[1,2]|
|GraphStitch Network layer|2|
|epochs|[100, 200, 300, 400, 500]|



- 623 **Hyperparameter Tuning.** As described in Section 3.2, we first use an unsupervised model based 624 on GraphMAE [20] to learn the general representation of the input features. The hyperparameters for 625 this step are set to the default values from the official GraphMAE implementation, with 50 training 626 epochs. Table 7 lists all the hyperparameters used in our model along with their corresponding search 627 spaces. During training, we conduct a **grid search** to identify the model that achieves the highest 628 AUROC score on the validation set. Finally, we evaluate the selected model on the test sets and report 629 the performance metrics. 

18 

## 630 **D Limitations and Impacts** 

631 Since the GNN encoder we employ mainly focuses on node-level features, the learned representations 632 may not be perfectly suited for edge and graph level tasks. Therefore, we leave the exploration of how 633 to integrate multiple tasks in the pre-training phase to future work. As a generalized graph anomaly 634 detection model, our work will be helpful in detecting classical graph anomaly applications, such as 635 financial fraud, cybercrime, etc. On the other hand, an error in the recognition result may put normal 636 groups or behaviors into anomalies, causing disturbance for the normal users in the graph network. 

## 637 **E Additional Experimental Results** 

638 We also provide the results of all experiments under the F1-macro evaluation metric. Similar to the 639 arrangement in the main text, we show the results of multi-level performance comparison under 640 F1-macro metrics in Table 8 and Table 9. The results of Zero-Shot Comparison under F1-macro 641 metrics are in Table 10 and Table 11. 

Table 8: Comparison of unified performance (AUROC) at both node and edge levels with different single-level methods, multi-task methods, and our proposed method. 

||**Dataset**<br>**Task-level**|**Reddit**<br>**Node**<br>**Edge**|**Weibo**<br>**Node**<br>**Edge**|**Amazon**<br>**Node**<br>**Edge**|**Yelp**<br>**Node**<br>**Edge**|**Tolokers**<br>**Node**<br>**Edge**|**Questions**<br>**Node**<br>**Edge**|
|---|---|---|---|---|---|---|---|
|Node-level|GCN<br>GIN<br>GraphSAGE<br>SGC<br>GAT<br>BernNet<br>PNA<br>AMNet<br>BWGNN|38_._81<br>/<br>27_._22<br>/<br>28_._30<br>/<br>12_._42<br>/<br>31_._28<br>/<br>46_._27<br>/<br>14_._08<br>/<br>45_._14<br>/<br>42_._25<br>/|92_._90<br>/<br>92_._01<br>/<br>85_._05<br>/<br>90_._49<br>/<br>92_._46<br>/<br>89_._59<br>/<br>88_._93<br>/<br>89_._04<br>/<br>86_._94<br>/|62_._45<br>/<br>72_._21<br>/<br>64_._05<br>/<br>57_._43<br>/<br>87_._96<br>/<br>89_._16<br>/<br>61_._84<br>/<br>89_._67<br>/<br>90_._49<br>/|42_._84<br>/<br>58_._78<br>/<br>64_._20<br>/<br>47_._26<br>/<br>57_._56<br>/<br>61_._42<br>/<br>52_._52<br>/<br>58_._75<br>/<br>64_._89<br>/|58_._64<br>/<br>59_._40<br>/<br>63_._71<br>/<br>54_._48<br>/<br>62_._68<br>/<br>58_._73<br>/<br>58_._49<br>/<br>58_._47<br>/<br>63_._43<br>/|48_._74<br>/<br>49_._95<br>/<br>51_._07<br>/<br>48_._59<br>/<br>47_._28<br>/<br>47_._59<br>/<br>46_._38<br>/<br>49_._90<br>/<br>52_._09<br>/|
|Edge-level|GCNE<br>GINE<br>SAGEE<br>SGCE<br>GATE<br>BernE<br>PNAE<br>AME<br>BWE|/<br>42_._97<br>/<br>36_._89<br>/<br>11_._94<br>/<br>41_._08<br>/<br>40_._65<br>/<br>39_._85<br>/<br>23_._03<br>/<br>41_._11<br>/<br>45_._36|/<br>92_._32<br>/<br>91_._54<br>/<br>89_._59<br>/<br>86_._70<br>/<br>90_._53<br>/<br>92_._15<br>/<br>92_._30<br>/<br>87_._04<br>/<br>91_._72|/<br>47_._99<br>/<br>47_._82<br>/<br>57_._79<br>/<br>55_._97<br>/<br>64_._69<br>/<br>69_._12<br>/<br>49_._27<br>/<br>66_._27<br>/<br>67_._56|/<br>42_._32<br>/<br>53_._14<br>/<br>57_._87<br>/<br>45_._13<br>/<br>49_._99<br>/<br>59_._76<br>/<br>52_._94<br>/<br>57_._09<br>/<br>59_._30|/<br>63_._15<br>/<br>60_._54<br>/<br>65_._95<br>/<br>56_._50<br>/<br>61_._43<br>/<br>61_._53<br>/<br>64_._98<br>/<br>61_._42<br>/<br>65_._09|/<br>58_._40<br>/<br>56_._96<br>/<br>**74**_._**15**<br>/<br>52_._07<br>/<br>62_._80<br>/<br>67_._32<br>/<br>65_._39<br>/<br>66_._74<br>/<br>66_._28|
|Multi-task|GraphPrompt-U<br>All-in-One-U|31.23<br>38.54<br>49.12<br>2.41|50.64<br>46.53<br>51.23<br>48.65|OOT<br>OOT<br>OOT<br>OOT|OOT<br>OOT<br>OOT<br>OOT|48.26<br>48.34<br>50.17<br>47.90|39.43<br>44.61<br>48.81<br>33.29|
|UniGAD<br>(Ours)|UniGAD - GCN<br>UniGAD - BWG|**56**_._**70**<br>**53**_._**80**<br>54_._08<br>51_._44|**95**_._**75**<br>**94**_._**29**<br>95_._35<br>94_._22|69_._39<br>59_._12<br>**91**_._**33**<br>**73**_._**59**|58_._23<br>56_._76<br>65_._20<br>64_._55<br>**74**_._**57**<br>**66**_._**66**<br>**68**_._**15**<br>**66**_._**20**||58_._06<br>57_._77<br>**59**_._**45**<br>57_._54|



Table 9: Comparison of unified performance (AUROC) at both node and graph levels with different single-level methods, multi-task methods, and our proposed method. 

||**Dataset**<br>**Task-level**|**BM-MN**<br>**Node**<br>**Graph**|**BM-MS**<br>**Node**<br>**Graph**|**BM-MT**<br>**Node**<br>**Graph**|**MUTAG**<br>**Node**<br>**Graph**|**MNIST0**<br>**Node**<br>**Graph**|**MNIST1**<br>**Node**<br>**Graph**|**T-Group**<br>**Node**<br>**Graph**|
|---|---|---|---|---|---|---|---|---|
|Node-level|GCN<br>GIN<br>GraphSAGE<br>SGC<br>GAT<br>BernNet<br>PNA<br>BWGNN|68_._25<br>/<br>32_._96<br>/<br>32_._96<br>/<br>32_._96<br>/<br>32_._96<br>/<br>35_._04<br>/<br>32_._96<br>/<br>82_._48<br>/|77_._77<br>/<br>24_._25<br>/<br>24_._25<br>/<br>24_._33<br>/<br>24_._25<br>/<br>51_._71<br>/<br>24_._25<br>/<br>75_._22<br>/|69_._72<br>/<br>25_._69<br>/<br>25_._69<br>/<br>25_._72<br>/<br>25_._69<br>/<br>30_._26<br>/<br>25_._69<br>/<br>76_._19<br>/|90_._41<br>/<br>92_._33<br>/<br>88_._87<br>/<br>54_._95<br>/<br>92_._07<br>/<br>86_._76<br>/<br>87_._49<br>/<br>92_._75<br>/|92_._03<br>/<br>88_._88<br>/<br>**99**_._**99**<br>/<br>49_._04<br>/<br>99_._94<br>/<br>**99**_._**99**<br>/<br>98_._83<br>/<br>**99**_._**99**<br>/|91_._95<br>/<br>88_._04<br>/<br>**99**_._**99**<br>/<br>82_._70<br>/<br>99_._96<br>/<br>**99**_._**99**<br>/<br>99_._27<br>/<br>**99**_._**99**<br>/|49_._50<br>/<br>49_._24<br>/<br>50_._77<br>/<br>49_._04<br>/<br>50_._29<br>/<br>52_._84<br>/<br>49_._88<br>/<br>51_._81<br>/|
|Graph-level|OCGIN<br>OCGTL<br>GLocalKD<br>iGAD<br>GmapAD<br>RQGNN|/<br>46_._15<br>/<br>46_._15<br>/<br>12_._50<br>/<br>68_._29<br>/<br>46_._15<br>/<br>95_._46|/<br>46_._15<br>/<br>46_._15<br>/<br>12_._50<br>/<br>81_._59<br>/<br>46_._15<br>/<br>93_._02|/<br>46_._11<br>/<br>46_._15<br>/<br>12_._50<br>/<br>89_._89<br>/<br>46_._15<br>/<br>97_._56|/<br>39_._78<br>/<br>39_._62<br>/<br>25_._59<br>/<br>89_._78<br>/<br>75_._48<br>/<br>89_._39|/<br>47_._79<br>/<br>47_._41<br>/<br>8_._98<br>/<br>OOM<br>/<br>OOM<br>/<br>93_._42|/<br>49_._74<br>/<br>47_._02<br>/<br>10_._11<br>/<br>OOM<br>/<br>OOM<br>/<br>96_._99|/<br>48_._91<br>/<br>48_._91<br>/<br>4_._09<br>/<br>OOM<br>/<br>OOM<br>/<br>48_._91|
|Multi-task|GraphPrompt-U<br>All-in-One-U|36_._95<br>45_._55<br>34_._98<br>12_._86|37_._46<br>12_._86<br>21_._24<br>20_._38|47_._92<br>46_._01<br>41_._36<br>12_._86|83_._08<br>45_._70<br>38_._71<br>25_._71|OOT<br>OOT<br>OOT<br>OOT|OOT<br>OOT<br>OOT<br>OOT|OOT<br>OOT<br>OOT<br>OOT|
|UniGAD<br>(Ours)|UniGAD - GCN<br>UniGAD - BWG|**99**_._**20**<br>**83**_._**62**<br>87_._91<br>55_._89|**99**_._**57**<br>**95**_._**86**<br>83_._79<br>61_._67|**91**_._**23**<br>**62**_._**37**<br>82_._53<br>51_._36|**93**_._**33**<br>**90**_._**00**<br>93_._07<br>89_._19|92_._17<br>93_._38<br>**99**_._**99**<br>**95**_._**54**|92_._49<br>97_._23<br>**99**_._**99**<br>**97**_._**60**|64_._98<br>74_._57<br>**68**_._**69**<br>**78**_._**09**|



19 

Table 10: Zero-shot transferability (F1-macro) at node and edge levels. 

|**Methods**|**Reddit**<br>**N**_→_**E**<br>**E**_→_**N**|**Weibo**<br>**N**_→_**E**<br>**E**_→_**N**|**Amazon**<br>**N**_→_**E**<br>**E**_→_**N**|**Yelp**<br>**N**_→_**E**<br>**E**_→_**N**|**Tolokers**<br>**N**_→_**E**<br>**E**_→_**N**|**Questions**<br>**N**_→_**E**<br>**E**_→_**N**|
|---|---|---|---|---|---|---|
|GraphPrompt-U<br>All-in-One-U|29_._87<br>41_._00<br>2_._38<br>49_._17|50_._48<br>47_._06<br>36_._82<br>51_._58|OOT<br>OOT<br>OOT<br>OOT|OOT<br>OOT<br>OOT<br>OOT|47_._77<br>43_._72<br>33_._39<br>49_._83|47_._63<br>39_._00<br>33_._08<br>49_._54|
|UniGAD - GCN<br>UniGAD - BWG|**50**_._**61**<br>**50**_._**58**<br>49_._79<br>50_._04|**94**_._**58**<br>71_._76<br>91_._10<br>**94**_._**22**|56_._70<br>61_._88<br>**68**_._**47**<br>**85**_._**07**|51_._29<br>53_._36<br>**63**_._**69**<br>**71**_._**10**|61_._14<br>57_._14<br>**65**_._**46**<br>**65**_._**57**|52_._01<br>52_._26<br>**55**_._**81**<br>**53**_._**44**|



Table 11: Zero-shot transferability (F1-macro) at node and graph levels. 

|**Methods**|**BM-MN**<br>**N**_→_**G**<br>**G**_→_**N**|**BM-MS**<br>**N**_→_**G**<br>**G**_→_**N**|**BM-MT**<br>**N**_→_**G**<br>**G**_→_**N**|**MUTAG**<br>**N**_→_**G**<br>**G**_→_**N**|**MNIST0**<br>**N**_→_**G**<br>**G**_→_**N**|**T-Group**<br>**N**_→_**G**<br>**G**_→_**N**|
|---|---|---|---|---|---|---|
|GraphPrompt-U<br>All-in-One-U|12_._86<br>34_._98<br>12_._86<br>34_._98|20_._59<br>42_._78<br>46_._01<br>41_._25|46_._01<br>43_._85<br>12_._86<br>22_._74|42_._15<br>27_._73<br>39_._53<br>48_._80|OOT<br>OOT<br>OOT<br>OOT|OOT<br>OOT<br>OOT<br>OOT|
|UniGAD - GCN<br>UniGAD - BWG|55_._49<br>**69**_._**14**<br>**57**_._**47**<br>53_._47|**57**_._**84**<br>**74**_._**63**<br>46_._15<br>52_._27|**52**_._**78**<br>**54**_._**87**<br>46_._22<br>39_._54|39_._89<br>**77**_._**47**<br>**45**_._**58**<br>66_._32|**65**_._**78**<br>**63**_._**75**<br>44_._56<br>60_._84|**66**_._**47**<br>**49**_._**22**<br>63_._92<br>48_._24|



20 

## 642 **NeurIPS Paper Checklist** 

643 

## 1. **Claims** 

644 Question: Do the main claims made in the abstract and introduction accurately reflect the 645 paper’s contributions and scope? 

646 Answer: [Yes] 

647 Justification: We confirm that the main claims made in the abstract and introduction accu648 rately reflect the paper’s contributions and scope. 649 Guidelines: 

- 650 • The answer NA means that the abstract and introduction do not include the claims 651 made in the paper. 

- 652 • The abstract and/or introduction should clearly state the claims made, including the 653 contributions made in the paper and important assumptions and limitations. A No or 654 NA answer to this question will not be perceived well by the reviewers. 

- 655 • The claims made should match theoretical and experimental results, and reflect how 656 much the results can be expected to generalize to other settings. 657 • It is fine to include aspirational goals as motivation as long as it is clear that these goals 658 are not attained by the paper. 

659 

## 2. **Limitations** 

660 Question: Does the paper discuss the limitations of the work performed by the authors? 661 Answer: [Yes] 

662 Justification: We discuss the limitations of our work in Appendix D. 663 Guidelines: 

- 664 • The answer NA means that the paper has no limitation while the answer No means that 665 the paper has limitations, but those are not discussed in the paper. 

- 666 • The authors are encouraged to create a separate "Limitations" section in their paper. 

- 667 • The paper should point out any strong assumptions and how robust the results are to 668 violations of these assumptions (e.g., independence assumptions, noiseless settings, 669 model well-specification, asymptotic approximations only holding locally). The authors 670 should reflect on how these assumptions might be violated in practice and what the 671 implications would be. 

- 672 • The authors should reflect on the scope of the claims made, e.g., if the approach was 673 only tested on a few datasets or with a few runs. In general, empirical results often 674 depend on implicit assumptions, which should be articulated. 

- 675 • The authors should reflect on the factors that influence the performance of the approach. 676 For example, a facial recognition algorithm may perform poorly when image resolution 677 is low or images are taken in low lighting. Or a speech-to-text system might not be 678 used reliably to provide closed captions for online lectures because it fails to handle 679 technical jargon. 

- 680 • The authors should discuss the computational efficiency of the proposed algorithms 681 and how they scale with dataset size. 

- 682 • If applicable, the authors should discuss possible limitations of their approach to 683 address problems of privacy and fairness. 

- 684 • While the authors might fear that complete honesty about limitations might be used by 685 reviewers as grounds for rejection, a worse outcome might be that reviewers discover 686 limitations that aren’t acknowledged in the paper. The authors should use their best 687 judgment and recognize that individual actions in favor of transparency play an impor688 tant role in developing norms that preserve the integrity of the community. Reviewers 689 will be specifically instructed to not penalize honesty concerning limitations. 

690 3. **Theory Assumptions and Proofs** 

691 Question: For each theoretical result, does the paper provide the full set of assumptions and 692 a complete (and correct) proof? 

21 

- 693 Answer: [Yes] 694 Justification: We provide a complete proof of our theoretical results (Theorem 1, Corollary 695 1, Theorem 2) in Appendix A.1. 696 Guidelines: 697 • The answer NA means that the paper does not include theoretical results. 698 • All the theorems, formulas, and proofs in the paper should be numbered and cross699 referenced. 700 • All assumptions should be clearly stated or referenced in the statement of any theorems. 701 • The proofs can either appear in the main paper or the supplemental material, but if 702 they appear in the supplemental material, the authors are encouraged to provide a short 703 proof sketch to provide intuition. 704 • Inversely, any informal proof provided in the core of the paper should be complemented 705 by formal proofs provided in appendix or supplemental material. 706 • Theorems and Lemmas that the proof relies upon should be properly referenced. 707 4. **Experimental Result Reproducibility** 708 Question: Does the paper fully disclose all the information needed to reproduce the main ex709 perimental results of the paper to the extent that it affects the main claims and/or conclusions 710 of the paper (regardless of whether the code and data are provided or not)? 711 Answer: [Yes] 712 Justification: We provide a link to the code in the abstract and include detailed implementa713 tion information in Appendix C to enhance reproducibility. 714 Guidelines: 715 • The answer NA means that the paper does not include experiments. 716 • If the paper includes experiments, a No answer to this question will not be perceived 717 well by the reviewers: Making the paper reproducible is important, regardless of 718 whether the code and data are provided or not. 719 • If the contribution is a dataset and/or model, the authors should describe the steps taken 720 to make their results reproducible or verifiable. 721 • Depending on the contribution, reproducibility can be accomplished in various ways. 722 For example, if the contribution is a novel architecture, describing the architecture fully 723 might suffice, or if the contribution is a specific model and empirical evaluation, it may 724 be necessary to either make it possible for others to replicate the model with the same 725 dataset, or provide access to the model. In general. releasing code and data is often 726 one good way to accomplish this, but reproducibility can also be provided via detailed 727 instructions for how to replicate the results, access to a hosted model (e.g., in the case 728 of a large language model), releasing of a model checkpoint, or other means that are 729 appropriate to the research performed. 730 • While NeurIPS does not require releasing code, the conference does require all submis731 sions to provide some reasonable avenue for reproducibility, which may depend on the 732 nature of the contribution. For example 733 (a) If the contribution is primarily a new algorithm, the paper should make it clear how 734 to reproduce that algorithm. 735 (b) If the contribution is primarily a new model architecture, the paper should describe 736 the architecture clearly and fully. 737 (c) If the contribution is a new model (e.g., a large language model), then there should 738 either be a way to access this model for reproducing the results or a way to reproduce 739 the model (e.g., with an open-source dataset or instructions for how to construct 740 the dataset). 741 (d) We recognize that reproducibility may be tricky in some cases, in which case 742 authors are welcome to describe the particular way they provide for reproducibility. 743 In the case of closed-source models, it may be that access to the model is limited in 744 some way (e.g., to registered users), but it should be possible for other researchers 745 to have some path to reproducing or verifying the results. 

746 5. **Open access to data and code** 

22 

747 Question: Does the paper provide open access to the data and code, with sufficient instruc748 tions to faithfully reproduce the main experimental results, as described in supplemental 749 material? 

749 

Answer: [Yes] 

750 

751 Justification: We make the code of our model open-sourced at `https://anonymous.` 752 `4open.science/r/UniGAD-A087/.` . As stated in Section 4, we used only publicly avail753 able datasets. 

753 

Guidelines: 

754 

- The answer NA means that paper does not include experiments requiring code. 

755 

- 756 • Please see the NeurIPS code and data submission guidelines ( `https://nips.cc/` 757 `public/guides/CodeSubmissionPolicy` ) for more details. 

757 

- 758 • While we encourage the release of code and data, we understand that this might not be 759 possible, so “No” is an acceptable answer. Papers cannot be rejected simply for not 760 including code, unless this is central to the contribution (e.g., for a new open-source 761 benchmark). 

- 762 • The instructions should contain the exact command and environment needed to run to 763 reproduce the results. See the NeurIPS code and data submission guidelines ( `https:` 764 `//nips.cc/public/guides/CodeSubmissionPolicy` ) for more details. 

- 765 • The authors should provide instructions on data access and preparation, including how 766 to access the raw data, preprocessed data, intermediate data, and generated data, etc. 

- 767 • The authors should provide scripts to reproduce all experimental results for the new 768 proposed method and baselines. If only a subset of experiments are reproducible, they 769 should state which ones are omitted from the script and why. 

- 770 • At submission time, to preserve anonymity, the authors should release anonymized 771 versions (if applicable). 

- 772 • Providing as much information as possible in supplemental material (appended to the 773 paper) is recommended, but including URLs to data and code is permitted. 

- 774 

## 6. **Experimental Setting/Details** 

- 775 Question: Does the paper specify all the training and test details (e.g., data splits, hyper776 parameters, how they were chosen, type of optimizer, etc.) necessary to understand the 777 results? 

- 778 Answer: [Yes] 

779 Justification: Please refer to Appendix C for the experiment implementation details. 780 Guidelines: 

- 781 

   - The answer NA means that the paper does not include experiments. 

- 782 • The experimental setting should be presented in the core of the paper to a level of detail 783 that is necessary to appreciate the results and make sense of them. 

- 784 • The full details can be provided either with the code, in appendix, or as supplemental 785 material. 

- 785 

- 786 7. **Experiment Statistical Significance** 

- 787 Question: Does the paper report error bars suitably and correctly defined or other appropriate 788 information about the statistical significance of the experiments? 

- 789 

Answer: [Yes] 

790 Justification: All of our experimental results come from the mean of 5 randomized trials. 791 Guidelines: 

- 792 

   - The answer NA means that the paper does not include experiments. 

- 793 • The authors should answer "Yes" if the results are accompanied by error bars, confi794 dence intervals, or statistical significance tests, at least for the experiments that support 795 the main claims of the paper. 

- 796 • The factors of variability that the error bars are capturing should be clearly stated (for 797 example, train/test split, initialization, random drawing of some parameter, or overall 798 run with given experimental conditions). 

23 

- 799 • The method for calculating the error bars should be explained (closed form formula, 800 call to a library function, bootstrap, etc.) 801 • The assumptions made should be given (e.g., Normally distributed errors). 802 • It should be clear whether the error bar is the standard deviation or the standard error 803 of the mean. 804 • It is OK to report 1-sigma error bars, but one should state it. The authors should 805 preferably report a 2-sigma error bar than state that they have a 96% CI, if the hypothesis 806 of Normality of errors is not verified. 807 • For asymmetric distributions, the authors should be careful not to show in tables or 808 figures symmetric error bars that would yield results that are out of range (e.g. negative 809 error rates). 810 • If error bars are reported in tables or plots, The authors should explain in the text how 811 they were calculated and reference the corresponding figures or tables in the text. 

812 

## 8. **Experiments Compute Resources** 

813 Question: For each experiment, does the paper provide sufficient information on the com814 puter resources (type of compute workers, memory, time of execution) needed to reproduce 815 the experiments? 

816 Answer: [Yes] 

817 Justification: Please refer to Appendix C for the computation resources. 818 Guidelines: 

- 819 • The answer NA means that the paper does not include experiments. 

- 820 • The paper should indicate the type of compute workers CPU or GPU, internal cluster, 821 or cloud provider, including relevant memory and storage. 822 • The paper should provide the amount of compute required for each of the individual 823 experimental runs as well as estimate the total compute. 824 • The paper should disclose whether the full research project required more compute 825 than the experiments reported in the paper (e.g., preliminary or failed experiments that 826 didn’t make it into the paper). 

## 9. **Code Of Ethics** 

||||
|---|---|---|
|827|9.|**Code Of Ethics**|
|828||Question: Does the research conducted in the paper conform, in every respect, with the|
|829||NeurIPS Code of Ethics`https://neurips.cc/public/EthicsGuidelines`?|
|830||Answer: [Yes]|



831 Justification: We confirmed the research conducted in the paper conform with the NeurIPS 832 Code of Ethics. 

833 Guidelines: 

- 834 • The answer NA means that the authors have not reviewed the NeurIPS Code of Ethics. 

- 835 • If the authors answer No, they should explain the special circumstances that require a 836 deviation from the Code of Ethics. 837 • The authors should make sure to preserve anonymity (e.g., if there is a special consid838 eration due to laws or regulations in their jurisdiction). 

839 10. **Broader Impacts** 

840 Question: Does the paper discuss both potential positive societal impacts and negative 841 societal impacts of the work performed? 

842 Answer: [Yes] 

843 Justification: Please refer to Appendix D for the relative discussion. 

844 Guidelines: 

- 845 • The answer NA means that there is no societal impact of the work performed. 

- 846 • If the authors answer NA or No, they should explain why their work has no societal 847 impact or why the paper does not address societal impact. 

24 

- 848 • Examples of negative societal impacts include potential malicious or unintended uses 849 (e.g., disinformation, generating fake profiles, surveillance), fairness considerations 850 (e.g., deployment of technologies that could make decisions that unfairly impact specific 851 groups), privacy considerations, and security considerations. 852 • The conference expects that many papers will be foundational research and not tied 853 to particular applications, let alone deployments. However, if there is a direct path to 854 any negative applications, the authors should point it out. For example, it is legitimate 855 to point out that an improvement in the quality of generative models could be used to 856 generate deepfakes for disinformation. On the other hand, it is not needed to point out 857 that a generic algorithm for optimizing neural networks could enable people to train 858 models that generate Deepfakes faster. 859 • The authors should consider possible harms that could arise when the technology is 860 being used as intended and functioning correctly, harms that could arise when the 861 technology is being used as intended but gives incorrect results, and harms following 862 from (intentional or unintentional) misuse of the technology. 

- 863 • If there are negative societal impacts, the authors could also discuss possible mitigation 864 strategies (e.g., gated release of models, providing defenses in addition to attacks, 865 mechanisms for monitoring misuse, mechanisms to monitor how a system learns from 866 feedback over time, improving the efficiency and accessibility of ML). 

- 867 11. **Safeguards** 

868 Question: Does the paper describe safeguards that have been put in place for responsible 869 release of data or models that have a high risk for misuse (e.g., pretrained language models, 870 image generators, or scraped datasets)? 

- 871 Answer: [NA] 

- 872 Justification: The paper poses no such risks. 873 Guidelines: 

- 874 

   - The answer NA means that the paper poses no such risks. 

- 875 • Released models that have a high risk for misuse or dual-use should be released with 876 necessary safeguards to allow for controlled use of the model, for example by requiring 877 that users adhere to usage guidelines or restrictions to access the model or implementing 878 safety filters. 

- 879 • Datasets that have been scraped from the Internet could pose safety risks. The authors 880 should describe how they avoided releasing unsafe images. 

- 881 • We recognize that providing effective safeguards is challenging, and many papers do 882 not require this, but we encourage authors to take this into account and make a best 883 faith effort. 

- 884 

## 12. **Licenses for existing assets** 

885 Question: Are the creators or original owners of assets (e.g., code, data, models), used in 886 the paper, properly credited and are the license and terms of use explicitly mentioned and 887 properly respected? 

- 888 Answer: [Yes] 

- 889 Justification: We properly credit all referenced baseline works in Appendix C. 890 Guidelines: 

- 891 • The answer NA means that the paper does not use existing assets. 

- 892 • The authors should cite the original paper that produced the code package or dataset. 

- 893 • The authors should state which version of the asset is used and, if possible, include a 894 URL. 

- 895 • The name of the license (e.g., CC-BY 4.0) should be included for each asset. 

- 896 • For scraped data from a particular source (e.g., website), the copyright and terms of 897 service of that source should be provided. 

- 898 • If assets are released, the license, copyright information, and terms of use in the 899 package should be provided. For popular datasets, `paperswithcode.com/datasets` 900 has curated licenses for some datasets. Their licensing guide can help determine the 901 license of a dataset. 

25 

- 902 • For existing datasets that are re-packaged, both the original license and the license of 903 the derived asset (if it has changed) should be provided. 904 • If this information is not available online, the authors are encouraged to reach out to 905 the asset’s creators. 

## 13. **New Assets** 

906 907 Question: Are new assets introduced in the paper well documented and is the documentation 908 provided alongside the assets? 909 Answer: [Yes] 910 Justification: We provide the code via the anonymous link in the abstract, which will be 911 open-sourced under the MIT license. Comprehensive documentation is included with the 912 assets to ensure ease of use and understanding. 

Guidelines: 

913 

- 914 • The answer NA means that the paper does not release new assets. 

- 915 • Researchers should communicate the details of the dataset/code/model as part of their 916 submissions via structured templates. This includes details about training, license, 917 limitations, etc. 

|914<br>915<br>916<br>917||• The answer NA means that the paper does not release new assets.<br>• Researchers should communicate the details of the dataset/code/model as part of their<br>submissions via structured templates. This includes details about training, license,<br>limitations, etc.|
|---|---|---|
|918||• The paper should discuss whether and how consent was obtained from people whose|
|919||asset is used.|
|920||• At submission time, remember to anonymize your assets (if applicable). You can either|
|921||create an anonymized URL or include an anonymized zip fle.|
|922|14.|**Crowdsourcing and Research with Human Subjects**|
|923||Question: For crowdsourcing experiments and research with human subjects, does the paper|
|924||include the full text of instructions given to participants and screenshots, if applicable, as|
|925||well as details about compensation (if any)?|



926 Answer: [NA] 

927 Justification: The paper does not involve crowdsourcing nor research with human subjects. 928 Guidelines: 

- 929 • The answer NA means that the paper does not involve crowdsourcing nor research with 930 human subjects. 

- 931 • Including this information in the supplemental material is fine, but if the main contribu932 tion of the paper involves human subjects, then as much detail as possible should be 933 included in the main paper. 

- 934 • According to the NeurIPS Code of Ethics, workers involved in data collection, curation, 935 or other labor should be paid at least the minimum wage in the country of the data 936 collector. 

- 937 15. **Institutional Review Board (IRB) Approvals or Equivalent for Research with Human** 

- 938 **Subjects** 939 Question: Does the paper describe potential risks incurred by study participants, whether 940 such risks were disclosed to the subjects, and whether Institutional Review Board (IRB) 941 approvals (or an equivalent approval/review based on the requirements of your country or 942 institution) were obtained? 

943 Answer: [NA] 

- 944 Justification: The paper does not involve crowdsourcing nor research with human subjects. 945 Guidelines: 

- 946 • The answer NA means that the paper does not involve crowdsourcing nor research with 947 human subjects. 

- 948 • Depending on the country in which research is conducted, IRB approval (or equivalent) 949 may be required for any human subjects research. If you obtained IRB approval, you 950 should clearly state this in the paper. 

- 951 • We recognize that the procedures for this may vary significantly between institutions 952 and locations, and we expect authors to adhere to the NeurIPS Code of Ethics and the 953 guidelines for their institution. 

26 

- 954 • For initial submissions, do not include any information that would break anonymity (if 955 applicable), such as the institution conducting the review. 

27 

