**KNOWFORMER: Revisiting Transformers for Knowledge Graph Reasoning** 

## **Junnan Liu**[1 2 *] **Qianren Mao**[1] _[ †]_ **Weifeng Jiang**[3] **Jianxin Li**[1 2] _[ †]_ 

## **Abstract** 

Knowledge graph reasoning plays a vital role in various applications and has garnered considerable attention. Recently, path-based methods have achieved impressive performance. However, they may face limitations stemming from constraints in message-passing neural networks, such as missing paths and information over-squashing. In this paper, we revisit the application of transformers for knowledge graph reasoning to address the constraints faced by path-based methods and propose a novel method KNOWFORMER. KNOWFORMER utilizes a transformer architecture to perform reasoning on knowledge graphs from the messagepassing perspective, rather than reasoning by textual information like previous pretrained language model based methods. Specifically, we define the attention computation based on the query prototype of knowledge graph reasoning, facilitating convenient construction and efficient optimization. To incorporate structural information into the self-attention mechanism, we introduce structureaware modules to calculate query, key, and value respectively. Additionally, we present an efficient attention computation method for better scalability. Experimental results demonstrate the superior performance of KNOWFORMER compared to prominent baseline methods on both transductive and inductive benchmarks. 

## **1. Introduction** 

Knowledge graphs (KGs) are structured knowledge bases that store known facts in the form of triplets (Hogan et al., 2022; Ji et al., 2022). Each triplet consists of a head entity, a relation, and a tail entity. However, real-world KGs often 

> *Work done during the internship at Zhongguancun Laboratory. _[†]_ Corresponding authors.[1] Zhongguancun Laboratory, Beijing, P.R.China.[2] SCSE, Beihang University, Beijing, P.R.China. 

> 3SCSE, Nanyang Technological University, Singapore. Correspondence to: Junnan Liu _<_ to.liujn@outlook.com _>_ . 

_Proceedings of the 41[st] International Conference on Machine Learning_ , Vienna, Austria. PMLR 235, 2024. Copyright 2024 by the author(s). 

suffer from high incompleteness, making it challenging to retrieve the desired facts (Wang et al., 2017; Ji et al., 2022). Knowledge graph reasoning, which involves inferring new facts based on existing ones, is a fundamental and indispensable task in KGs with a wide range of applications (Qiu et al., 2019; Cao et al., 2019; Huang et al., 2019; Abujabal et al., 2018; Cheng et al., 2020; Zeng et al., 2022). 

A substantial line of previous research has been dedicated to developing effective and robust methods for knowledge graph reasoning. Knowledge graph embedding (KGE) methods focus on embedding entities and relations into a lowdimensional space (Bordes et al., 2013; Sun et al., 2019; Yang et al., 2015; Trouillon et al., 2017; Li et al., 2022). Despite their impressive performance, these embeddingbased methods struggle to generalize to inductive scenarios since they cannot leverage the local structures of knowledge graphs (Teru et al., 2020). As a result, researchers have turned to path-based methods to enhance reasoning performance and improve generalization. These methods learn pairwise representations from subgraphs (Teru et al., 2020; Mai et al., 2021; Wang et al., 2022a; Chamberlain et al., 2023) or relational paths (Zhu et al., 2021; Zhang & Yao, 2022; Zhang et al., 2023; Zhu et al., 2023) between entities which can be used to predict unseen facts. However, path-based methods can be limited by the missing paths (Franceschi et al., 2019) and over-squashing information (Alon & Yahav, 2021) as shown in Figure 1. 

Recent works have leveraged the transformers for knowledge graph reasoning, drawing inspiration from the success of transformer-based models in various domains (Vaswani et al., 2017; Devlin et al., 2019; Dosovitskiy et al., 2021; Rives et al., 2021). Typically, these studies encode components of knowledge graphs using their text descriptions through pretrained language models (PLMs) and utilize either a discriminative paradigm (Lv et al., 2022; Wang et al., 2022b) or a generative paradigm (Xie et al., 2022; Saxena et al., 2022) to predict new facts. However, there are two main issues: (1) textual descriptions often contain ambiguity and require domain-specific knowledge for accurate encoding, _e.g._ , pretrained language models like BERT (Devlin et al., 2019), which are trained with commonsense knowledge, may not transfer well to domain-specific KGs; and (2) capturing structural information, which can be crucial for knowledge graph reasoning owing to the _low-rank_ 

1 

**KNOWFORMER: Revisiting Transformers for Knowledge Graph Reasoning** 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/icml_2024_KnowFormer_Revisiting_Transformers_for_Knowledge_Graph_Reasoning_images/icml_2024_KnowFormer_Revisiting_Transformers_for_Knowledge_Graph_Reasoning.pdf-0002-01.png)


**----- Start of picture text -----**<br>
tail entity query realtion<br>query realtion<br>Missing Path tail entity<br>head entity head entity<br>❌<br>Over-<br>Squashing<br>**----- End of picture text -----**<br>


_Figure 1._ Path-based methods could be limited by the missing paths (Franceschi et al., 2019) and over-squashing information (Alon & Yahav, 2021). 

_assumption_ (Koren et al., 2009; Lacroix et al., 2018) is challenging, and some approaches address this by utilizing path and context encoding or pretraining on large-scale KGs as alternatives (Chen et al., 2021; Li et al., 2023). 

In this paper, we aim to further investigate the potential benefits of applying the _transformer architecture_ (Vaswani et al., 2017) to the task of knowledge graph reasoning. Contrary to previous approaches that utilize pretrained language models for encoding text descriptions, we leverage the attention mechanism within the transformer architecture to capture interactions between any pair of entities. Our principal contribution is the introduction of an expressive and scalable attention mechanism tailored specifically to knowledge graph reasoning. The resulting model, named **KNOWFORMER** , provides informative representations for knowledge graph reasoning in both transductive and inductive scenarios. It can address the limitations of vanilla path-based methods, such as the path missing and information over-squashing. Specifically: 

- We redefine the self-attention mechanism, originally introduced by Vaswani et al. (2017), as a weighted aggregation of pairwise information for specific queries based on the plausibility of entity pairs as query prototypes. This redefinition enables us to perform attention computation on multi-relational KGs and reduce the modeling complexity. 

- For constructing the attention mechanism, we introduce two modules that generate informative representations for query, key, and value, respectively. These modules are designed based on relational message passing neural networks, allowing us to consider structural information during attention calculation. 

- To improve scalability, we adopt an instance-based similarity measure (Cui & Chen, 2022) to reduce the significant number of degrees of freedom and introduce an approximation method that maintains complexity linearly proportional to entities. Moreover, we provide theoretical guarantees for its stability and expressivity. 

- We also provide empirical evidence of the effectiveness of our proposed KNOWFORMER on knowledge graph 

reasoning benchmarks in both transductive and inductive settings, surpassing the performance of salient baselines. 

## **2. Related Work** 

We classify the related work on knowledge graph reasoning into three main paradigms: embedding-based methods, pathbased methods, and transformer models. 

**Embedding-Based Methods.** Embedding-based methods aim to learn distributed representations for entities and relations by preserving the triplets in the knowledge graph. Notable early methods include TransE (Bordes et al., 2013), DistMult (Yang et al., 2015) and ComplEX (Trouillon et al., 2017). Subsequent work has focused on improving the score function or embedding space of these methods to enhance the modeling of semantic patterns (Sun et al., 2019; Tang et al., 2020; Zhang et al., 2020; Li et al., 2022). 

In addition, some researchers have explored the application of neural networks in an encoder-decoder paradigm to obtain adaptive and robust embeddings. Representative methods include convolutional neural networks (Dettmers et al., 2018; Nguyen et al., 2018) and graph neural networks (Schlichtkrull et al., 2018; Vashishth et al., 2020; You et al., 2021). Although embedding methods have demonstrated promising performance in knowledge graph reasoning and scalability on large KGs, these embeddings are hard to generalize to unseen entities and relations, thus limiting their applicability in the inductive setting. 

**Path-Based Methods.** Path-based methods in knowledge graph reasoning have their origins in traditional heuristic similarity approaches, which include measuring the weighted count of paths (Katz, 1953), random walk probability (Page, 1998), and shortest path length (Liben-Nowell & Kleinberg, 2007). In recent years, there have been proposals to employ neural networks to encode paths between entities, such as recurrent neural networks (Neelakantan et al., 2015), and aggregate these representations for reasoning. Another research direction focuses on learning probabilistic logical rules over KGs (Yang et al., 2017; Sadeghian et al., 2019) 

2 

**KNOWFORMER: Revisiting Transformers for Knowledge Graph Reasoning** 

and utilizing these rules to assign weights to paths. More recent works have achieved state-of-the-art performance by incorporating well-designed message passing neural networks (Wang et al., 2021; Zhu et al., 2021; Zhang & Yao, 2022). These methods enhance expressivity through the use of _target distinguishable_ initial function and _relational_ message and aggregation functions. PathCon (Wang et al., 2021), for instance, introduces a relational message-passing framework for the KG completion task and achieves impressive results. However, prominent path-based methods still face limitations, including the recognized shortcomings of message-passing neural networks such as incompleteness (Franceschi et al., 2019) and over-squashing (Alon & Yahav, 2021). 

**Transformers for Knowledge Graph Reasoning.** Currently, most research utilizes pretrained models based on transformer architecture to encode textual descriptions and leverage its knowledge and the semantic understanding ability for reasoning. For instance, KG-BERT (Yao et al., 2019) and PKGC (Lv et al., 2022) employ a pretrained language model encoder to represent KG triples with text descriptions, enabling the inference of new facts through a classification layer using a special token representation. In contrast, some studies explore a sequence-to-sequence approach for generating reasoning results (Xie et al., 2022; Saxena et al., 2022). This paradigm has received increasing attention with the emergence of large language models. To bridge the gap between unstructured textual descriptions and structured KGs, researchers have explored incorporating contextual information by sampling and encoding neighborhood triplets or paths (Xie et al., 2022; Chen et al., 2021; Li et al., 2023). However, these methods face limitations due to their heavy reliance on text description, including the lack of textual data, domain knowledge requirements, and the neglect of structural information in KGs. In this paper, we revisit how to perform knowledge graph reasoning leveraging the transformer architecture from the perspective of knowledge graph structure. 

## **3. Preliminary** 

In this section, we introduce the background knowledge of knowledge graphs and knowledge graph reasoning. Due to the page limitations, more preliminaries about transformers can be found in the appendix. 

**Knowledge Graph.** Typically, a knowledge graph _G_ = _{V, E, R}_ is a collection of triplets _E_ = _{_ ( _hi, ri, ti_ ) _| hi, ti ∈V, ri ∈R}_ consist a set of entities _V_ and a set of relations _R_ . Each triplet is a relational edge from head entity _hi_ to tail entity _ti_ with the relation _ri_ . For ease of notation, we can also represent a fact as _r_ ( _u, v_ ) _∈E_ where _u, v ∈V_ and _r ∈R_ . Additionally, we 

define the neighborhood set of an entity _u ∈V_ relative to a relation _r ∈R_ as _Nr_ ( _u_ ) = _{v | r_ ( _u, v_ ) _∈E}_ . 

**Knowledge Graph Reasoning.** Given a knowledge graph _G_ = _{V, E, R}_ , the goal of knowledge graph reasoning is to leverage existing facts to infer the missing elements of the query fact ( _h, rq, t_ ), where _rq_ is a query relation. Based on the type of missing elements, there are three sub-tasks: head reasoning to infer (? _, rq, t_ ), tail reasoning to infer ( _h, rq,_ ?), and relation reasoning to infer ( _h,_ ? _, t_ ). This paper mainly focuses on the tail reasoning task, as the other tasks can be transformed into the same form. 

## **4. Proposed Method** 

In this section, we introduce the construction of the KNOWFORMER. Specifically, we focus on how to create the proposed attention mechanism and integrate it into a transformer model. 

## **4.1. Attention Computation of KNOWFORMER** 

We adopt a modified definition of self-attention, similar to the formulation used in Tsai et al. (2019) and Chen et al. (2022). Assuming that _**X** ∈_ R _[n][×][d]_ is input features, we have: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/icml_2024_KnowFormer_Revisiting_Transformers_for_Knowledge_Graph_Reasoning_images/icml_2024_KnowFormer_Revisiting_Transformers_for_Knowledge_Graph_Reasoning.pdf-0003-12.png)


_· ·_ here, _fq_ ( ) represents the query function, _fv_ ( ) represents the value function, and _κ_ ( _·, ·_ ) : R _[d] ×_ R _[d] →_ R[+] is a positivedefinite kernel that measures the pairwise similarity. 

In a knowledge graph, different relation types _r ∈R_ determine various modes of interaction between entity pairs. Consequently, a straightforward approach is to incorporate relation-specific attention to model these diverse interactions: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/icml_2024_KnowFormer_Revisiting_Transformers_for_Knowledge_Graph_Reasoning_images/icml_2024_KnowFormer_Revisiting_Transformers_for_Knowledge_Graph_Reasoning.pdf-0003-15.png)


where _f[r]_ ( _._ ) means a relation-specific function. However, this approach will result in significant computational overhead ( _O_ ( _|R| · |V|_[2] )) and suboptimal optimization due to a high degree of freedom. Inspired by instance-based learning (Stanfill & Waltz, 1986; Cui & Chen, 2022), we consider modeling the pairwise interactions based on the plausibility of entity pairs as query prototypes. 

**Definition 4.1** (Query Prototype) **.** Given a relation _r ∈R_ , entity _u ∈V_ and _v ∈V_ are the prototypes for relation _r_ if ( _u, r,_ ?) _∧_ ( _v, r,_ ?) _̸_ = _∅_ . 

For instance, Barack Obama and Joseph Biden are the prototypes for relation Nationality, since 

3 

**KNOWFORMER: Revisiting Transformers for Knowledge Graph Reasoning** 

we have (Barack Obama _,_ Nationality _,_ ?) _∧_ (Joseph Biden _,_ Nationality _,_ ?) = _{_ USA _}_ . Based on Definition 4.1, given query ( _h, rq,_ ?), we propose to calculate attention scores between entity pairs referring to their plausibility as prototypes. Our motivation is that the greater the plausibility of two entities as prototypes, the more _similarity_ they exhibit under the current query relation, leading to higher attention scores. In this way, for each query in the dataset, we do not need to consider the interaction of entities for all relations, reducing the complexity to _O_ ( _|V|_[2] ). Our idea considers the plausibility of two entities _u_ and _v_ being prototypes to the query relation _rq_ and enables information propagation between them based on the plausibility. To achieve this, we propose to utilize the function _fq_ ( _·_ ) to obtain informative representations of entities to distinguish query prototypes, the function _κ_ ( _·, ·_ ) to serve as the score function for measuring plausibility, _·_ and lastly, _fv_ ( ) to generate propagated information. Next, we will provide a detailed explanation of constructing the attention mechanism by defining the functions _fq_ ( _·_ ), _fv_ ( _·_ ), and _κ_ ( _·, ·_ ). 

_·_ vide informative representations, **Query Function.** Query functiondenoted as _fq_ ( ) is designed to pro- _**z**_ � _u_ , for each entity _u_ to distinguish query prototypes effectively. A key insight is that considering the context of each entity on the knowledge graph ( _i.e._ , its _k_ -hop neighbor facts) is essential in determining query prototypes. For example, entities such as Barack Obama and Joseph Biden can serve as prototypes for the relation Nationality due to their common neighbor relation President of. We introduce Q-RMPNN, a relational message-passing neural network (Gilmer et al., 2017; Xu et al., 2019; Wang et al., 2021) designed to incorporate neighbor facts into the entity representations. Q-RMPNN consists of two stages: (1) each entity sends relational messages to its neighbors; (2) each entity aggregates the received relational messages and updates its representation. Drawing inspiration from knowledge graph embeddings, we generate a relational message _**m** v|u,r_ for each fact _r_ ( _u, v_ ) by maximizing its continuous truth value (Wang et al., 2023): 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/icml_2024_KnowFormer_Revisiting_Transformers_for_Knowledge_Graph_Reasoning_images/icml_2024_KnowFormer_Revisiting_Transformers_for_Knowledge_Graph_Reasoning.pdf-0004-03.png)


where _D_ indicates the range domain for the entity embeddings and _**z** u_ is the representation of entity _u_ within _fq_ ( _·_ ), _**r**_ **ˆ** is the representation of relation _r_ conditioned on query relation _rq_ , _ϕ_ ( _·, ·, ·_ ) is a score function and _g_ ( _·, ·_ ) is an estimation function. In this paper, we adopt the DistMult method (Yang et al., 2015) as the foundation for our implementation similar to Zhu et al. (2021), to say: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/icml_2024_KnowFormer_Revisiting_Transformers_for_Knowledge_Graph_Reasoning_images/icml_2024_KnowFormer_Revisiting_Transformers_for_Knowledge_Graph_Reasoning.pdf-0004-05.png)


where _**R**_ is the input features of relation set _R_ , and _**W** r,_ _**b** r_ is the relation-specific parameters. At the _l_ -th layer of _fq_ ( _·_ ), we fuse _**z**_[(] _u[l]_[)][for each] _[ u][ ∈V]_[by the summation of aggregated] information from the ( _l −_ 1)-th layer: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/icml_2024_KnowFormer_Revisiting_Transformers_for_Knowledge_Graph_Reasoning_images/icml_2024_KnowFormer_Revisiting_Transformers_for_Knowledge_Graph_Reasoning.pdf-0004-07.png)


where _**x** u ∈_ R _[d]_ represents the input features, _**ϵ** ∼ N_ ( **0** _,_ _**I**_ ) denotes Gaussian noise to distinguish different source entities, [ _·, ·_ ] indicates concatenate function, _**α**_[(] _[l]_[)] captures layerspecific parameters to retain the original information, and **Φ**[(] _[l]_[)] ( _·_ ) represents a layer-specific update function parameterized by an MLP network. After _L_[�] layers of updates, � � we obtain the final representation _**z** u_ given by _**z** u_ = _**z**_[(] _u[L]_[�][)][.] Through Q-RMPNN, _**z**_ � _u_ is optimized to capture _L_[�] -hop neighbor facts, resulting in a structure-aware and knowledgeoriented representation. 

**Value Function.** We leverage the value function to generate pairwise representations _**z**_ **ˆ** _u_ conditioned on the query relation _rq_ for the query ( _h, rq,_ ?). The structural information between node pairs is crucial in knowledge graph reasoning, which can be viewed as a link prediction task (Zhang et al., 2021). However, it is challenging for the attention mechanism to capture the structural information of the input graph explicitly. To address this, we design V-RMPNN to encode pairwise structural information into the value. Specifically, V-RMPNN is implemented as follows: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/icml_2024_KnowFormer_Revisiting_Transformers_for_Knowledge_Graph_Reasoning_images/icml_2024_KnowFormer_Revisiting_Transformers_for_Knowledge_Graph_Reasoning.pdf-0004-10.png)


where _**x** u_ , _**m**_[(] _u[l] |[−] v,r_[1)][,] _**[ β]**_[(] _[l]_[)][,][ [] _[·][,][ ·]_[]][ and] **[ Ψ]**[(] _[l]_[)][(] _[·]_[)][ are defined simi-] larly to Q-RMPNN. The term I _u_ = _h ⊙_ **1** represents head entity labeling features to enhance node representation, which is essential for the link prediction task (Zhang et al., 2021; You et al., 2021). After _L_[�] layers of updates, the final repre- **ˆ ˆ** sentation _**z** u_ is obtained as _**z** u_ = _**z**_[(] _[L]_[�][)][[.]] 

_u|h,rq_[[.]] 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/icml_2024_KnowFormer_Revisiting_Transformers_for_Knowledge_Graph_Reasoning_images/icml_2024_KnowFormer_Revisiting_Transformers_for_Knowledge_Graph_Reasoning.pdf-0004-13.png)


where _**W**_ 1 and _**W**_ 2 are the linear projection matrixes and the bias is omitted for convenience. However, the _dot-thenexponentiate operation_ will lead to quadratic complexity 

4 

**KNOWFORMER: Revisiting Transformers for Knowledge Graph Reasoning** 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/icml_2024_KnowFormer_Revisiting_Transformers_for_Knowledge_Graph_Reasoning_images/icml_2024_KnowFormer_Revisiting_Transformers_for_Knowledge_Graph_Reasoning.pdf-0005-01.png)


**----- Start of picture text -----**<br>
Matmul Add<br>Matmul<br>Q-RMPNN Q-RMPNN<br>Q-RMPNNQ-RMPNN Q-RMPNNV-RMPNN<br>Query Function Value Function<br>**----- End of picture text -----**<br>


_Figure 2._ Overview of the proposed attention mechanism, which takes entity features _**X**_ and _**R**_ as input and outputs _**Z**_[�] for all _u ∈V_ . 

since we must calculate the inner product of the query matrix with the key matrix before the exponentiate function. In practice, we approximate the exponential function exp( _·_ ) using the _first-order Taylor expansion_ around zero (Li et al., 2020; Wu et al., 2023a; Dass et al., 2023), resulting in: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/icml_2024_KnowFormer_Revisiting_Transformers_for_Knowledge_Graph_Reasoning_images/icml_2024_KnowFormer_Revisiting_Transformers_for_Knowledge_Graph_Reasoning.pdf-0005-04.png)


To ensure non-negativity of the attention scores, we further normalize the inputs by their Frobenius norm, leading to the kernel function implementation as follows: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/icml_2024_KnowFormer_Revisiting_Transformers_for_Knowledge_Graph_Reasoning_images/icml_2024_KnowFormer_Revisiting_Transformers_for_Knowledge_Graph_Reasoning.pdf-0005-06.png)


Based on Equation (8) and Equation (9), we can initially compute the inner product of the key matrix with the value matrix, followed by calculating the inner product of the query matrix with the resultant matrix as shown in Figure 2, thereby reducing the complexity to linear. 

**Theorem 4.2** (Approximation Error for Exponential Kernel) **.** _For each u, v ∈V, the approximation error_ ∆= � � _|κ_ ( _**z** u,_ � _**z** v_ ) _− κexp_ ( _**z** u,_ � _**z** v_ ) _| will be bounded by O_ ( _e[γ] /_ 2) _, where γ ∈_ (0 _,_ 1) _._ 

We provide the proof in Appendix B. Theorem 4.2 demonstrates that the approximation error remains bounded regardless of the size of KGs, which implies Equation (8) is a stable estimation. 

**Summary.** Now, we present the detailed computational process of the proposed attention mechanism, as illustrated in Figure 2. Assume _**X** ∈_ R _[|V|×][d]_ is the input feature matrix 

of entities, _**R** ∈_ R _[|R|×][d]_ is the input features matrix of relations, _fq_ ( _·_ ) and _fv_ ( _·_ ) are the query and value function, and _κ_ ( _·, ·_ ) is the kernel function. We firstly obtain _**Z**_[�] = [ _fq_ ( _**x** u_ )] _u∈V_ and _**Z**_[�] = [ _fv_ ( _**x** u_ )] _u∈V_ by iteratively applying Equation (5) and Equation (6). Then the output feature matrix _**Z**_ can be calculated as follows: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/icml_2024_KnowFormer_Revisiting_Transformers_for_Knowledge_Graph_Reasoning_images/icml_2024_KnowFormer_Revisiting_Transformers_for_Knowledge_Graph_Reasoning.pdf-0005-12.png)


where we omit the bias and diag( _·_ ) means a transformation to diagonal matrix. We further show the overall procedure in Algorithm 1. We can observe that the time complexity of Equation (10) is linearly related to _|V|_ and _|E|_ , which is scalable to massive KGs. We will discuss the details of complexity in Section 4.3. 

## **4.2. Overall Architecture of KNOWFORMER** 

Based on the attention definition provided in Section 4.1, we complete the construction of our proposed model KNOWFORMER. The remaining components of KNOWFORMER adhere to the standard transformer architecture described in Vaswani et al. (2017). Each layer of KNOWFORMER comprises an attention function and a feedforward network. In addition, a skip-connection is applied after the attention function, and normalization layers are inserted before and after the feedforward network. Finally, we stack _L_ layers to construct KNOWFORMER. 

At the start of training, we initialize the entity feature matrix _**X**_ and relation feature matrix _**R**_ . Specifically, _**X**_ is set to an all-zero vector, while _**R**_ is randomly initialized and is learnable. We then iteratively calculate the entity representations as follows: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/icml_2024_KnowFormer_Revisiting_Transformers_for_Knowledge_Graph_Reasoning_images/icml_2024_KnowFormer_Revisiting_Transformers_for_Knowledge_Graph_Reasoning.pdf-0005-17.png)


Lastly, we obtain the final representation of entities _**X**_[(] _[L]_[)] . We update model parameters by optimizing a negative sampling loss (Mikolov et al., 2013; Sun et al., 2019) using Adam optimizer (Kingma & Ba, 2015). The loss function is defined for each training fact ( _h, r, t_ ) as: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/icml_2024_KnowFormer_Revisiting_Transformers_for_Knowledge_Graph_Reasoning_images/icml_2024_KnowFormer_Revisiting_Transformers_for_Knowledge_Graph_Reasoning.pdf-0005-19.png)


5 

**KNOWFORMER: Revisiting Transformers for Knowledge Graph Reasoning** 

**Algorithm 1** Attention Computation 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/icml_2024_KnowFormer_Revisiting_Transformers_for_Knowledge_Graph_Reasoning_images/icml_2024_KnowFormer_Revisiting_Transformers_for_Knowledge_Graph_Reasoning.pdf-0006-02.png)


20: **end for** 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/icml_2024_KnowFormer_Revisiting_Transformers_for_Knowledge_Graph_Reasoning_images/icml_2024_KnowFormer_Revisiting_Transformers_for_Knowledge_Graph_Reasoning.pdf-0006-04.png)


where _σ_ ( _·|h, r_ ) represents the score of a candidate fact, computed by a multilayer perceptron (MLP) and a sigmoid function using the output feature _**X**_[(] _[L]_[)] and _t[′] ∈V \ {t}_ indicates the negative samples. 

## **4.3. Discussion** 

In this section, we provide a thorough discussion of the KNOWFORMER, focusing on the analysis of its time complexity and expressivity. 

**Time Complexity.** Time complexity of our proposed model primarily depends on the attention function. We can 

break down the time complexity into the query function, the value function, and the kernel function. The query function is called _L_ times, with each call taking _O_ ( _L_[�] ( _|E|d_ + _|V|d_[2] )). Similarly, the value function is called _L_ times, with a single call taking _O_ ( _L_[�] ( _|E|d_ + _|V|d_[2] )). Finally, the kernel function is called _L_ times, with each call taking _O_ ( _|V|d_[2] ). Therefore, the total complexity amounts to _O_ ( _L_ ( _L_[�] + _L_[�] )( _|E|d_ + _|V|d_[2] )). The above conclusion shows that the time complexity of our proposed method is linearly related to the number of facts and entities in the knowledge graph, exhibiting better scalability. 

**Expressivity Analysis.** Huang et al. (2023) introduced a variant of the Weisfeiler-Leman Test (Weisfeiler & Leman, 1968; Xu et al., 2019; Barcelo et al.´ , 2022) called the _Relational Asymmetric Local 2-WL Test_ (RA-WL2) to evaluate the expressive power of message-passing networks in the knowledge graph reasoning task. We formally demonstrate the expressivity of KNOWFORMER based on RA-WL2. This is stated in the following theorem: 

**Theorem 4.3** (Expressivity of KNOWFORMER) **.** _Assuming the estimated graph by the attention layer of_ KNOWFORMER _is E_[˜] _, the attention layer of_ KNOWFORMER _can achieve at least the same level of expressive ability as RA-WL_ 2 _on the extended graph E ∪ E_[˜] _._ 

We provide proof in Appendix D. The estimated graph _E_[˜] can be viewed as a collection of facts of special relations determining the equivalence between entity pairs for a specific query. KNOWFORMER exhibits stronger expressive power due to its ability to propagate information on an extended graph, distinguishing it from vanilla path-based methods. 

## **5. Experimental Results** 

In this section, we conduct empirical studies motivated by the following aspects: 

- **Transductive Performance.** As a general knowledge graph reasoning task, we aim to demonstrate the performance of KNOWFORMER in transductive knowledge graph reasoning tasks. 

- **Inductive Performance.** Our proposed model supports reasoning in inductive settings. How does KNOWFORMER perform in inductive knowledge graph reasoning tasks? 

- **Ablation Study.** How important are the various components within our proposed framework? For instance, how does model performance change if we omit the query function? 

- **Further Experiments.** We conduct additional experiments to further showcase the effectiveness of KNOWFORMER. For example, does KNOWFORMER perform 

6 

**KNOWFORMER: Revisiting Transformers for Knowledge Graph Reasoning** 

|**Method**|**FB15k-237**||**WN18RR**|**NELL-995**|**NELL-995**|**YAGO3-10**<br>MRR<br>H@1<br>H@10|
|---|---|---|---|---|---|---|
||MRR<br>H@1<br>H@10||MRR<br>H@1<br>H@10|MRR|H@1<br>H@10||
|TransE(Bordes et al.,2013)<br>DistMult(Yang et al.,2015)<br>RotatE(Sun et al.,2019)<br>HousE (Li et al.,2022)<br>DRUM(Sadeghian et al.,2019)<br>CompGCN(Vashishth et al.,2020)<br>RNNLogic(Qu et al.,2021)<br>NBFNet(Zhu et al.,2021)<br>RED-GNN(Zhang & Yao,2022)<br>A*Net(Zhu et al.,2023)<br>AdaProp(Zhang et al.,2023)<br>ULTRA (Galkin et al.,2024)<br>HittER(Chen et al.,2021)<br>SimKGC(Wang et al.,2022b)<br>KGT5 (Saxena et al.,2022)|0.330<br>23.2<br>52.6<br>0.358<br>26.4<br>55.0<br>0.337<br>24.1<br>53.3<br>0.361<br>26.6<br>55.1<br>0.343<br>25.5<br>51.6<br>0.355<br>26.4<br>53.5<br>0.344<br>25.2<br>53.0<br>0.415<br>32.1<br>59.9<br>0.374<br>28.3<br>55.8<br>0.411<br>32.1<br>58.6<br>0.417<br>33.1<br>58.5<br>0.368<br>-<br>56.4<br>0.373<br>27.9<br>55.8<br>0.336<br>24.9<br>51.1<br>0.276<br>21.0<br>41.4||0.222<br>1.4<br>52.8<br>0.455<br>41.0<br>54.4<br>0.477<br>42.8<br>57.1<br>0.511<br>46.5<br>60.2<br>0.486<br>42.5<br>58.6<br>0.479<br>44.3<br>54.6<br>0.483<br>44.6<br>55.8<br>0.551<br>49.7<br>66.6<br>0.533<br>48.5<br>62.4<br>0.549<br>49.5<br>65.9<br>0.562<br>49.9<br>67.1<br>0.480<br>-<br>61.4<br>0.503<br>46.2<br>58.4<br>**0.666**<br>**58.5**<br>**80.0**<br>0.508<br>48.7<br>54.4|0.507<br>0.510<br>0.508<br>0.519<br>0.532<br>0.463<br>0.516<br>0.525<br>0.543<br>0.521<br>0.554<br>0.509<br>-<br>0.501<br>-|42.4<br>64.8<br>43.8<br>63.6<br>44.8<br>60.8<br>45.8<br>61.8<br>46.0<br>66.2<br>38.3<br>59.6<br>46.3<br>57.8<br>45.1<br>63.9<br>47.6<br>65.1<br>44.7<br>63.1<br>49.3<br>65.5<br>-<br>66.0<br>-<br>-<br>42.6<br>65.3<br>-<br>-|0.510<br>41.3<br>68.1<br>0.566<br>49.1<br>70.4<br>0.495<br>40.2<br>67.0<br>0.571<br>49.1<br>71.4<br>0.531<br>45.3<br>67.6<br>0.421<br>39.2<br>57.7<br>0.554<br>50.9<br>62.2<br>0.563<br>48.0<br>70.8<br>0.556<br>48.3<br>68.9<br>0.556<br>47.0<br>70.7<br>0.573<br>51.0<br>68.5<br>0.557<br>-<br>71.0<br>-<br>-<br>-<br>0.211<br>14.1<br>35.1<br>0.426<br>36.8<br>52.8|
||||||||
||||||||
|**KNOWFORMER**|**0.430**<br>**34.3**<br>**60.8**||0.579<br>52.8<br>68.7|**0.566**|**50.2**<br>**67.5**|**0.615**<br>**54.7**<br>**73.4**|



_Table 1._ Transductive knowledge graph reasoning performance for 4 different datasets. The best results are **boldfaced** and the second-best results are underlined. Our proposed model, KNOWFORMER, achieves SOTA performance in most cases marked by **tan** . 

better for longer paths? 

Our code is available at https://github.com/ jnanliu/KnowFormer. 

## **5.1. Transductive Performance** 

**Datasets.** We conduct experiments on four widely utilized transductive knowledge graph reasoning datasets: FB15k-237 (Toutanova & Chen, 2015), WN18RR (Dettmers et al., 2018), NELL-995 (Xiong et al., 2017) and YAGO310 (Mahdisoltani et al., 2015). For consistency, we utilize the same data splits as in prior studies (Zhu et al., 2021; Zhang et al., 2023). 

**Baselines.** We compare KNOWFORMER to several prominent baselines, categorized into three classes: 

- _Embedding-based methods_ , including TransE (Bordes et al., 2013), DistMult (Yang et al., 2015), RotatE (Sun et al., 2019), and HousE (Li et al., 2022). These methods learn embeddings for entities and relations and perform reasoning based on distance or similarity. 

- _Path-based methods_ , including DRUM (Sadeghian et al., 2019), CompGCN (Vashishth et al., 2020), RNNLogic (Qu et al., 2021), NBFNet (Zhu et al., 2021), RED-GNN (Zhang & Yao, 2022), A*Net (Zhu et al., 2023), AdaProp (Zhang et al., 2023), and ULTRA (Galkin et al., 2024). These methods conduct reasoning by utilizing the path information connecting entities. 

- _Text-based methods_ , including HittER (Chen et al., 2021), SimKGC (Wang et al., 2022b), and KGT5 (Saxena et al., 2022). These methods utilize textual infor- 

mation for reasoning in knowledge graphs. 

**Results and Analysis.** We evaluate the performance of reasoning using standard metrics (Wang et al., 2017; Ji et al., 2022), namely MRR ( _↑_ ), Hit@1 ( _↑_ ), and Hit@10 ( _↑_ ). The results on various datasets are presented in Table 1. Across all metrics, KNOWFORMER demonstrates a substantial performance advantage over the best baseline method in FB15k-237, NELL-995, and YAGO3-10. Particularly in the large-scale YAGO3-10 dataset, KNOWFORMER exhibits a significant performance advantage over the best baseline methods. This highlights the high effectiveness of KNOWFORMER in transductive knowledge graph reasoning. In the WN18RR dataset, KNOWFORMER achieves the second-best performance among all baselines. Notably, SimKGC (Wang et al., 2022b) performs exceptionally well in this dataset, which can be attributed to its ability to acquire knowledge from extensive pretrained data, allowing it to capture the semantic relationships in WN18RR as a subset of a comprehensive English lexical database. Conversely, methods like SimKGC that rely on pretrained language models tend to underperform in datasets that require domain-specific knowledge. In contrast, our proposed text-free method relies solely on the structure of the knowledge graphs, resulting in enhanced robustness and scalability. 

## **5.2. Inductive Performance** 

Inductive reasoning has become a prominent research topic (Hamilton et al., 2017; Teru et al., 2020) due to the ubiquitous occurrence of emergent entities in real-world applications (Zhang & Chen, 2020). In this part, we will show the performance of KNOWFORMER on the inductive knowl- 

7 

**KNOWFORMER: Revisiting Transformers for Knowledge Graph Reasoning** 

|**Method**||**v1**|**v1**||**v2**|**v2**||**v3**|**v4**<br>MRR<br>H@1<br>H@10|
|---|---|---|---|---|---|---|---|---|---|
|||MRR<br>H@1<br>H@10|||MRR<br>H@1<br>H@10||MRR|H@1<br>H@10||
||||||**FB15k-237**|||||
|DRUM(Sadeghian et al.,2019)<br>NBFNet(Zhu et al.,2021)<br>RED-GNN(Zhang & Yao,2022)<br>A*Net(Zhu et al.,2023)<br>AdaProp(Zhang et al.,2023)<br>INGRAM (Lee et al.,2023)||0.333<br>24.7<br>0.442<br>33.5<br>0.369<br>30.2<br>0.457<br>**38.1**<br>0.310<br>19.1<br>0.293<br>16.7|47.4<br>57.4<br>48.3<br>58.9<br>55.1<br>49.3||0.395<br>28.4<br>59.5<br>0.514<br>42.1<br>68.5<br>0.469<br>38.1<br>62.9<br>0.510<br>41.9<br>67.2<br>0.471<br>37.2<br>65.9<br>0.274<br>16.3<br>48.2||0.402<br>0.476<br>0.445<br>0.476<br>0.471<br>0.233|30.8<br>57.1<br>38.4<br>63.7<br>35.1<br>50.3<br>38.9<br>62.9<br>37.7<br>63.7<br>14.0<br>40.8|0.410<br>30.9<br>59.3<br>0.453<br>36.0<br>62.7<br>0.442<br>34.0<br>62.1<br>0.466<br>36.5<br>64.5<br>0.454<br>35.3<br>63.8<br>0.214<br>11.4<br>39.7|
|**KNOWFORMER**||**0.466**<br>37.8|**60.6**||**0.532**<br>**43.3**<br>**70.3**||**0.494**|**40.0**<br>**65.9**|**0.480**<br>**38.3**<br>**65.3**|
||||||**WN18RR**|||||
|DRUM(Sadeghian et al.,2019)<br>NBFNet(Zhu et al.,2021)<br>RED-GNN(Zhang & Yao,2022)<br>A*Net(Zhu et al.,2023)<br>AdaProp(Zhang et al.,2023)<br>INGRAM(Lee et al.,2023)<br>SimKGC (Wanget al.,2022b)||0.666<br>61.3<br>0.741<br>69.5<br>0.701<br>65.3<br>0.727<br>68.2<br>0.733<br>66.8<br>0.277<br>13.0<br>0.315<br>19.2|77.7<br>**82.6**<br>79.9<br>81.0<br>80.6<br>60.6<br>56.7||0.646<br>0.704<br>0.690<br>0.704<br>**0.715**<br>0.236<br>0.378|59.5<br>74.7<br>65.1<br>79.8<br>63.3<br>78.0<br>64.9<br>80.3<br>64.2<br>**82.6**<br>11.2<br>48.0<br>23.9<br>65.0|0.380<br>0.452<br>0.427<br>0.441<br>**0.474**<br>0.230<br>0.303|33.0<br>47.7<br>39.2<br>56.8<br>36.8<br>52.4<br>38.6<br>54.4<br>39.6<br>**58.8**<br>11.6<br>46.6<br>18.6<br>54.3|0.627<br>58.6<br>70.2<br>0.641<br>60.8<br>69.4<br>0.651<br>60.6<br>72.1<br>0.661<br>**61.6**<br>74.3<br>**0.662**<br>61.1<br>**75.5**<br>0.118<br>4.1<br>25.9<br>0.308<br>17.5<br>57.7|
|**KNOWFORMER**||**0.752**<br>**71.5**|81.9||0.709|**65.6**<br>81.7|0.467|**40.6**<br>57.1|0.646<br>60.9<br>72.7|
||||||**NELL-995**|||||
|NBFNet(Zhu et al.,2021)<br>RED-GNN(Zhang & Yao,2022)<br>AdaProp(Zhang et al.,2023)<br>INGRAM (Lee et al.,2023)||0.584<br>50.0<br>0.637<br>52.2<br>0.644<br>52.2<br>0.697<br>57.5|79.5<br>86.6<br>88.6<br>86.5||0.410<br>27.1<br>63.5<br>0.419<br>31.9<br>60.1<br>0.452<br>34.4<br>65.2<br>0.358<br>25.3<br>59.6||0.425<br>0.436<br>0.435<br>0.308|26.2<br>60.6<br>34.5<br>59.4<br>33.7<br>61.8<br>19.9<br>50.9|0.287<br>25.3<br>59.1<br>0.363<br>25.9<br>**60.7**<br>0.366<br>24.7<br>**60.7**<br>0.221<br>12.4<br>44.0|
|**KNOWFORMER**||**0.827**<br>**77.0**|**93.0**||**0.465**<br>**35.7**<br>**65.7**||**0.478**|**37.8**<br>**65.7**|**0.378**<br>**26.7**<br>59.8|



_Table 2._ Inductive knowledge graph reasoning performance for 3 different datasets and 12 different versions. For each version, the best results are **boldfaced** and the second-best results are underlined. Our proposed model, KNOWFORMER is marked by **tan** . 

edge graph reasoning task. Note that in this paper, we focus on the _semi-inductive_ task, which has an invariant relation set. However, we believe that our method can be adapted to handle full-inductive tasks with slight modifications. We leave this aspect as future work to address. 

**Datasets.** In line with Teru et al. (2020), we employ the same data divisions of FB15k-237 (Toutanova & Chen, 2015), WN18RR (Dettmers et al., 2018), and NELL995 (Xiong et al., 2017). Each of these divisions comprises 4 versions, resulting in a total of 12 subsets. It is worth noting that each subset has a unique separation between the training and test sets. Specifically, the training and test sets within each subset have unique sets of entities while sharing the same set of relations. 

**Baselines.** We compare KNOWFORMER with 7 baseline methods for inductive knowledge graph reasoning, including DRUM (Sadeghian et al., 2019), NBFNet (Zhu et al., 2021), RED-GNN (Zhang & Yao, 2022), A*Net (Zhu et al., 2023), AdaProp (Zhang et al., 2023), SimKGC (Wang et al., 2022b), and INGRAM (Lee et al., 2023). Note that ULTRA (Galkin et al., 2024) is not included as a baseline model, since we consider ULTRA to be a distinctive method based on the pre-training and fine-tuning paradigm, which can 

benefit from a large-scale amount of training data. However, our method still achieves comparable results to ULTRA on some datasets. Please refer to the experimental results in Appendix F.1 for more details. 

**Results and Analysis.** We also evaluate the performance by standard metrics (Wang et al., 2017; Ji et al., 2022), including MRR ( _↑_ ), Hit@1 ( _↑_ ), and Hit@10 ( _↑_ ). Table 2 showcases the inductive performance of KNOWFORMER and baselines. We have observed that KNOWFORMER consistently achieves the highest performance on the majority of metrics across all versions of inductive datasets. Furthermore, our method also produces competitive results for the remaining metrics. Compared to transductive settings, KNOWFORMER demonstrates a relatively small performance gap. This can be attributed to the limited size of inductive datasets. Path-based methods can attain superior performance by utilizing shorter paths and avoiding excessive compression of information. However, KNOWFORMER maintains a performance advantage, indicating its effectiveness. Another noteworthy observation is the inadequate performance of SimKGC (Wang et al., 2022b) in inductive reasoning. This suggests a potential necessity for text-based and pretrained language model methods to have access to larger amounts of training data to converge 

8 

**KNOWFORMER: Revisiting Transformers for Knowledge Graph Reasoning** 

|**Method**|**MRR**|**H@1**|**H@10**|
|---|---|---|---|
|FB15k-237||||
|**KNOWFORMER**|**0.430**|**34.3**|**60.8**|
|_w/o_attention<br>_w/o_query function<br>_w/o_value function|0.417<br>0.422<br>0.367|32.8<br>33.0<br>30.7|58.8<br>59.2<br>48.7|
|RF-based kernel function|0.419|32.9|57.6|
|UMLS||||
|**KNOWFORMER**|0.971|**95.8**|**99.8**|
|full-exponential kernel function|**0.974**|95.2|99.5|



_Table 3._ Ablation study of KNOWFORMER and its variants on transductive FB15k-237 dataset and UMLS dataset. We mark the vanilla KNOWFORMER by **tan** . The best performance is **boldfaced** . 

and attain improved results. 

## **5.3. Ablation Study** 

In this part, we aim to evaluate the effectiveness of the proposed attention mechanism in KNOWFORMER. We conduct comparisons against several variations, including: (1) removing attention, which results in KNOWFORMER degenerating into vanilla path-based methods such as NBFNet (Zhu et al., 2021); (2) excluding the query function; (3) excluding the value function; (4) substituting the kernel function with an approximation method based on random features (Sinha & Duchi, 2016; Liu et al., 2022; Wu et al., 2022); (5) substituting the kernel function with full-exponential kernel function. 

The results of the ablation study on FB15k-237 are presented in Table 3. We can initially observe a decline in model performance when attention is removed, underscoring the importance of the proposed attention mechanism. The performance decrease caused by omitting the query function is relatively limited, suggesting that the input of the attention layer retains some structural information after the update of previous layers. Furthermore, omitting the value function leads to a significant decrease in model performance, as the explicit integration of structural information is crucial for reasoning tasks in knowledge graphs. In contrast, the pure transformer model faces difficulties in achieving this, highlighting the necessity of structural-aware modules. Lastly, a comparison with RF-based methods demonstrates the effectiveness of our kernel function. 

To compare the performance between our proposed approximation kernel and the original full-exponential kernel, we conduct an additional ablation study on a small dataset UMLS (Kok & Domingos, 2007) in Table 6 due to the computation overhead. The minimal variation in the performance of both variants indicates the effectiveness of the approximate kernel. The full exponential kernel has a slight performance advantage, yet the quadratic computational complexity it brings is unacceptable in knowledge graph 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/icml_2024_KnowFormer_Revisiting_Transformers_for_Knowledge_Graph_Reasoning_images/icml_2024_KnowFormer_Revisiting_Transformers_for_Knowledge_Graph_Reasoning.pdf-0009-08.png)


**----- Start of picture text -----**<br>
Our NBFNet AdaProp Our NBFNet AdaProp<br>0 . 58 0 . 6<br>0 . 57 0 . 5<br>0 . 56<br>0 . 4<br>0 . 55<br>0 . 3<br>0 . 54<br>0 . 53 0 . 2<br>0 . 52<br>0 . 1<br>0 . 51<br>3 4 5 6 7 8 0 . 0 0.05 0.25 0.50<br>Path Length Drop Ratio<br>MRR MRR<br>**----- End of picture text -----**<br>


_Figure 3._ Experimental results on WN18RR. In the left chart, we evaluate the performance of all methods under different lengths of reasoning paths. In the right chart, we randomly drop some facts from the test data and report the performance of all methods. 

reasoning. 

## **5.4. Further Analysis** 

In this section, we present additional experiments to demonstrate the superiority of KNOWFORMER over vanilla pathbased methods (Zhu et al., 2021; Zhang & Yao, 2022). Specifically, we concentrate on two folds: (1) comparing methods in scenarios with gradually growing reasoning paths, and (2) comparing methods in scenarios with missing reasoning paths, where we randomly drop paths between pairs of entities in the test data. The experimental results on the WN18RR dataset are illustrated in Figure 3. 

**Performance w.r.t. Longer Paths.** The individual performance of KNOWFORMER remained unaffected even as the reasoning path extended, validating its capacity to mitigate information over-squashing resulting from prolonged entity interactions. This finding is consistent with the results in Alon & Yahav (2021). 

**Performance w.r.t. Missing Paths.** As the proportion of missing paths increases, KNOWFORMER exhibits stronger robustness compared to the baseline methods, which show a noticeable performance decline. This demonstrates that the presence of all-pair interactions in KNOWFORMER enables it to better handle missing paths. 

## **6. Conclusion** 

This paper proposes a novel transformer-based method KNOWFORMER for knowledge graph reasoning. KNOWFORMER consists of an expressive and scalable attention mechanism. Specifically, we introduce message-passing neural network based query function and value function to obtain informative key and value representations. Additionally, we present an efficient attention computation method to enhance the scalability of KNOWFORMER. Experimental results show that KNOWFORMER outperforms salient baselines on both transductive and inductive benchmarks. 

9 

**KNOWFORMER: Revisiting Transformers for Knowledge Graph Reasoning** 

## **Acknowledgements** 

We appreciate all anonymous reviewers for their careful review and valuable comments. The authors of this paper are supported by the National Natural Science Foundation of China through grant No.62225202. 

## **Impact Statement** 

This paper introduces KNOWFORMER, a novel method for knowledge graph reasoning. Knowledge graph reasoning is widely applied in various fields, including personalized recommendations, drug discovery, and enterprise-level knowledge management decisions. Each of these applications has the potential to generate significant societal impacts. Personalized recommendations, for instance, can enhance user experience, but also raise concerns about privacy breaches. For example, in a social platform, users can provide personalized friend recommendations based on their personal information, but this information is also easy to cause privacy disclosure. Similarly, drug discovery contributes to advancements in the field of biomedicine, but it also introduces risks associated with medical accidents. We encourage researchers to further investigate and address concerns related to privacy risks and errors that may arise in knowledge graph reasoning. 

## **References** 

- Abujabal, A., Roy, R. S., Yahya, M., and Weikum, G. Neverending learning for open-domain question answering over knowledge bases. In _WWW_ , pp. 1053–1062. ACM, 2018. 1 

- Alon, U. and Yahav, E. On the bottleneck of graph neural networks and its practical implications. In _ICLR_ . OpenReview.net, 2021. 1, 2, 3, 9 

- Ba, L. J., Kiros, J. R., and Hinton, G. E. Layer normalization. _CoRR_ , abs/1607.06450, 2016. 15 

- Barcelo,´ P., Galkin, M., Morris, C., and Orth, M. A. R. Weisfeiler and leman go relational. In _LoG_ , volume 198 of _Proceedings of Machine Learning Research_ , pp. 46. PMLR, 2022. 6, 16 

- Bordes, A., Usunier, N., Garc´ıa-Duran, A., Weston, J., and´ Yakhnenko, O. Translating embeddings for modeling multi-relational data. In _NeurIPS_ , pp. 2787–2795, 2013. 1, 2, 7, 19 

- Cao, Y., Wang, X., He, X., Hu, Z., and Chua, T. Unifying knowledge graph learning and recommendation: Towards a better understanding of user preferences. In _WWW_ , pp. 151–161. ACM, 2019. 1 

- Chamberlain, B. P., Shirobokov, S., Rossi, E., Frasca, F., Markovich, T., Hammerla, N. Y., Bronstein, M. M., and Hansmire, M. Graph neural networks for link prediction with subgraph sketching. In _ICLR_ . OpenReview.net, 2023. 1 

- Chen, D., O’Bray, L., and Borgwardt, K. M. Structureaware transformer for graph representation learning. In _ICML_ , volume 162 of _Proceedings of Machine Learning Research_ , pp. 3469–3489. PMLR, 2022. 3, 15 

- Chen, S., Liu, X., Gao, J., Jiao, J., Zhang, R., and Ji, Y. Hitter: Hierarchical transformers for knowledge graph embeddings. In _EMNLP_ , pp. 10395–10407. Association for Computational Linguistics, 2021. 2, 3, 7 

- Cheng, D., Yang, F., Wang, X., Zhang, Y., and Zhang, L. Knowledge graph-based event embedding framework for financial quantitative investments. In _SIGIR_ , pp. 2221– 2230. ACM, 2020. 1 

- Cui, W. and Chen, X. Instance-based learning for knowledge base completion. In _NeurIPS_ , pp. 30744–30755, 2022. 2, 3 

- Dass, J., Wu, S., Shi, H., Li, C., Ye, Z., Wang, Z., and Lin, Y. Vitality: Unifying low-rank and sparse approximation for vision transformer acceleration with a linear taylor attention. In _HPCA_ , pp. 415–428. IEEE, 2023. 5 

- Davidson, T. R., Falorsi, L., Cao, N. D., Kipf, T., and Tomczak, J. M. Hyperspherical variational auto-encoders. In _UAI_ , pp. 856–865. AUAI Press, 2018. 21 

- Dettmers, T., Minervini, P., Stenetorp, P., and Riedel, S. Convolutional 2d knowledge graph embeddings. In _AAAI_ , pp. 1811–1818. AAAI Press, 2018. 2, 7, 8 

- Devlin, J., Chang, M., Lee, K., and Toutanova, K. BERT: pre-training of deep bidirectional transformers for language understanding. In _NAACL-HLT_ , pp. 4171–4186. Association for Computational Linguistics, 2019. 1 

- Dosovitskiy, A., Beyer, L., Kolesnikov, A., Weissenborn, D., Zhai, X., Unterthiner, T., Dehghani, M., Minderer, M., Heigold, G., Gelly, S., Uszkoreit, J., and Houlsby, N. An image is worth 16x16 words: Transformers for image recognition at scale. In _ICLR_ . OpenReview.net, 2021. 1 

- Dwivedi, V. P. and Bresson, X. A generalization of transformer networks to graphs. _CoRR_ , abs/2012.09699, 2020. 15 

- Franceschi, L., Niepert, M., Pontil, M., and He, X. Learning discrete structures for graph neural networks. In _ICML_ , volume 97 of _Proceedings of Machine Learning Research_ , pp. 1972–1982. PMLR, 2019. 1, 2, 3 

10 

**KNOWFORMER: Revisiting Transformers for Knowledge Graph Reasoning** 

- Galkin, M., Yuan, X., Mostafa, H., Tang, J., and Zhu, Z. Towards foundation models for knowledge graph reasoning. In _ICLR_ . OpenReview.net, 2024. 7, 8, 21 

- Gilmer, J., Schoenholz, S. S., Riley, P. F., Vinyals, O., and Dahl, G. E. Neural message passing for quantum chemistry. In _ICML_ , volume 70 of _Proceedings of Machine Learning Research_ , pp. 1263–1272. PMLR, 2017. 4 

- Glorot, X., Bordes, A., and Bengio, Y. Deep sparse rectifier neural networks. In _AISTATS_ , volume 15 of _JMLR Proceedings_ , pp. 315–323. JMLR.org, 2011. 20 

- Hamilton, W. L., Ying, Z., and Leskovec, J. Inductive representation learning on large graphs. In _NeurIPS_ , pp. 1024–1034, 2017. 7 

- Hogan, A., Blomqvist, E., Cochez, M., d’Amato, C., de Melo, G., Gutierrez, C., Kirrane, S., Gayo, J. E. L., Navigli, R., Neumaier, S., Ngomo, A. N., Polleres, A., Rashid, S. M., Rula, A., Schmelzeisen, L., Sequeda, J. F., Staab, S., and Zimmermann, A. Knowledge graphs. _ACM Comput. Surv._ , 54(4):71:1–71:37, 2022. 1 

- Hornik, K. Approximation capabilities of multilayer feedforward networks. _Neural Networks_ , 4(2):251–257, 1991. 18 

- Hornik, K., Stinchcombe, M. B., and White, H. Multilayer feedforward networks are universal approximators. _Neural Networks_ , 2(5):359–366, 1989. 18 

- Huang, X., Zhang, J., Li, D., and Li, P. Knowledge graph embedding based question answering. In _WSDM_ , pp. 105–113. ACM, 2019. 1 

- Huang, X., Orth, M. R., Ceylan, I. I., and Barcelo, P. A theory of link prediction via relational weisfeiler-leman on knowledge graphs. In _NeurIPS_ , 2023. 6, 17 

- Jeh, G. and Widom, J. Simrank: a measure of structuralcontext similarity. In _KDD_ , pp. 538–543. ACM, 2002. 21 

- Ji, S., Pan, S., Cambria, E., Marttinen, P., and Yu, P. S. A survey on knowledge graphs: Representation, acquisition, and applications. _IEEE Trans. Neural Networks Learn. Syst._ , 33(2):494–514, 2022. 1, 7, 8 

- Katz, L. A new status index derived from sociometric analysis. _Psychometrika_ , 18(1):39–43, 1953. 2, 21 

- Kim, J., Nguyen, D., Min, S., Cho, S., Lee, M., Lee, H., and Hong, S. Pure transformers are powerful graph learners. In _NeurIPS_ , 2022. 15 

- Kingma, D. P. and Ba, J. Adam: A method for stochastic optimization. In _ICLR (Poster)_ , 2015. 5 

- Kipf, T. N. and Welling, M. Variational graph auto-encoders. _CoRR_ , abs/1611.07308, 2016. 21 

- Kok, S. and Domingos, P. M. Statistical predicate invention. In _ICML_ , volume 227 of _ACM International Conference Proceeding Series_ , pp. 433–440. ACM, 2007. 9 

- Koren, Y., Bell, R. M., and Volinsky, C. Matrix factorization techniques for recommender systems. _Computer_ , 42(8): 30–37, 2009. 2 

- Lacroix, T., Usunier, N., and Obozinski, G. Canonical tensor decomposition for knowledge base completion. In _ICML_ , volume 80 of _Proceedings of Machine Learning Research_ , pp. 2869–2878. PMLR, 2018. 2 

- Lee, J., Chung, C., and Whang, J. J. Ingram: Inductive knowledge graph embedding via relation graphs. In _ICML_ , volume 202 of _Proceedings of Machine Learning Research_ , pp. 18796–18809. PMLR, 2023. 8 

- Li, J., Wang, Q., and Mao, Z. Inductive relation prediction from relational paths and context with hierarchical transformers. In _ICASSP_ , pp. 1–5. IEEE, 2023. 2, 3 

- Li, R., Su, J., Duan, C., and Zheng, S. Linear attention mechanism: An efficient attention for semantic segmentation. _CoRR_ , abs/2007.14902, 2020. 5 

- Li, R., Zhao, J., Li, C., He, D., Wang, Y., Liu, Y., Sun, H., Wang, S., Deng, W., Shen, Y., Xie, X., and Zhang, Q. House: Knowledge graph embedding with householder parameterization. In _ICML_ , volume 162 of _Proceedings of Machine Learning Research_ , pp. 13209–13224. PMLR, 2022. 1, 2, 7 

- Liben-Nowell, D. and Kleinberg, J. M. The link-prediction problem for social networks. _J. Assoc. Inf. Sci. Technol._ , 58(7):1019–1031, 2007. 2 

- Liu, F., Huang, X., Chen, Y., and Suykens, J. A. K. Random features for kernel approximation: A survey on algorithms, theory, and beyond. _IEEE Trans. Pattern Anal. Mach. Intell._ , 44(10):7128–7148, 2022. 9 

- Lv, X., Lin, Y., Cao, Y., Hou, L., Li, J., Liu, Z., Li, P., and Zhou, J. Do pre-trained models benefit knowledge graph completion? A reliable evaluation and a reasonable approach. In _ACL (Findings)_ , pp. 3570–3581. Association for Computational Linguistics, 2022. 1, 3 

- Ma, L., Lin, C., Lim, D., Romero-Soriano, A., Dokania, P. K., Coates, M., Torr, P. H. S., and Lim, S. Graph inductive biases in transformers without message passing. In _ICML_ , volume 202 of _Proceedings of Machine Learning Research_ , pp. 23321–23337. PMLR, 2023. 15 

11 

**KNOWFORMER: Revisiting Transformers for Knowledge Graph Reasoning** 

- Mahdisoltani, F., Biega, J., and Suchanek, F. M. YAGO3: A knowledge base from multilingual wikipedias. In _CIDR_ . www.cidrdb.org, 2015. 7 

- Mai, S., Zheng, S., Yang, Y., and Hu, H. Communicative message passing for inductive relation reasoning. In _AAAI_ , pp. 4294–4302. AAAI Press, 2021. 1 

- Mikolov, T., Sutskever, I., Chen, K., Corrado, G. S., and Dean, J. Distributed representations of words and phrases and their compositionality. In _NeurIPS_ , pp. 3111–3119, 2013. 5 

- Min, E., Chen, R., Bian, Y., Xu, T., Zhao, K., Huang, W., Zhao, P., Huang, J., Ananiadou, S., and Rong, Y. Transformer for graphs: An overview from architecture perspective. _CoRR_ , abs/2202.08455, 2022. 15 

- Neelakantan, A., Roth, B., and McCallum, A. Compositional vector space models for knowledge base completion. In _ACL_ , pp. 156–166. The Association for Computer Linguistics, 2015. 2 

- Nguyen, D. Q., Nguyen, T. D., Nguyen, D. Q., and Phung, D. Q. A novel embedding model for knowledge base completion based on convolutional neural network. In _NAACL-HLT_ , pp. 327–333. Association for Computational Linguistics, 2018. 2 

- Page, L. The pagerank citation ranking: Bringing order to the web. technical report. _Stanford Digital Library Technologies Project_ , 1998. 2, 21 

- Paszke, A., Gross, S., Massa, F., Lerer, A., Bradbury, J., Chanan, G., Killeen, T., Lin, Z., Gimelshein, N., Antiga, L., Desmaison, A., Kopf,¨ A., Yang, E. Z., DeVito, Z., Raison, M., Tejani, A., Chilamkurthy, S., Steiner, B., Fang, L., Bai, J., and Chintala, S. Pytorch: An imperative style, high-performance deep learning library. In _NeurIPS_ , pp. 8024–8035, 2019. 20 

- Perozzi, B., Al-Rfou, R., and Skiena, S. Deepwalk: online learning of social representations. In _KDD_ , pp. 701–710. ACM, 2014. 21 

- Qiu, D., Zhang, Y., Feng, X., Liao, X., Jiang, W., Lyu, Y., Liu, K., and Zhao, J. Machine reading comprehension using structural knowledge graph-aware network. In _EMNLP/IJCNLP_ , pp. 5895–5900. Association for Computational Linguistics, 2019. 1 

- Qu, M., Chen, J., Xhonneux, L. A. C., Bengio, Y., and Tang, J. Rnnlogic: Learning logic rules for reasoning on knowledge graphs. In _ICLR_ . OpenReview.net, 2021. 7 

- Rampasek, L., Galkin, M., Dwivedi, V. P., Luu, A. T., Wolf,´ G., and Beaini, D. Recipe for a general, powerful, scalable graph transformer. In _NeurIPS_ , 2022. 15 

- Rives, A., Meier, J., Sercu, T., Goyal, S., Lin, Z., Liu, J., Guo, D., Ott, M., Zitnick, C. L., Ma, J., and Fergus, R. Biological structure and function emerge from scaling unsupervised learning to 250 million protein sequences. _Proc. Natl. Acad. Sci. USA_ , 118(15):e2016239118, 2021. 1 

- Sadeghian, A., Armandpour, M., Ding, P., and Wang, D. Z. DRUM: end-to-end differentiable rule mining on knowledge graphs. In _NeurIPS_ , pp. 15321–15331, 2019. 2, 7, 8 

- Saxena, A., Kochsiek, A., and Gemulla, R. Sequence-tosequence knowledge graph completion and question answering. In _ACL_ , pp. 2814–2828. Association for Computational Linguistics, 2022. 1, 3, 7 

- Schlichtkrull, M. S., Kipf, T. N., Bloem, P., van den Berg, R., Titov, I., and Welling, M. Modeling relational data with graph convolutional networks. In _ESWC_ , volume 10843 of _Lecture Notes in Computer Science_ , pp. 593– 607. Springer, 2018. 2 

- Sen, P., Namata, G., Bilgic, M., Getoor, L., Gallagher, B., and Eliassi-Rad, T. Collective classification in network data. _AI Mag._ , 29(3):93–106, 2008. 20 

- Sinha, A. and Duchi, J. C. Learning kernels with random features. In _NeurIPS_ , pp. 1298–1306, 2016. 9 

- Stanfill, C. and Waltz, D. L. Toward memory-based reasoning. _Commun. ACM_ , 29(12):1213–1228, 1986. 3 

- Sun, Z., Deng, Z., Nie, J., and Tang, J. Rotate: Knowledge graph embedding by relational rotation in complex space. In _ICLR (Poster)_ . OpenReview.net, 2019. 1, 2, 5, 7 

- Tang, J., Qu, M., Wang, M., Zhang, M., Yan, J., and Mei, Q. LINE: large-scale information network embedding. In _WWW_ , pp. 1067–1077. ACM, 2015. 21 

- Tang, Y., Huang, J., Wang, G., He, X., and Zhou, B. Orthogonal relation transforms with graph context modeling for knowledge graph embedding. In _ACL_ , pp. 2713–2722. Association for Computational Linguistics, 2020. 2 

- Teru, K. K., Denis, E. G., and Hamilton, W. L. Inductive relation prediction by subgraph reasoning. In _ICML_ , volume 119 of _Proceedings of Machine Learning Research_ , pp. 9448–9457. PMLR, 2020. 1, 7, 8 

- Toutanova, K. and Chen, D. Observed versus latent features for knowledge base and text inference. In _CVSC_ , pp. 57– 66. Association for Computational Linguistics, 2015. 7, 8 

- Trouillon, T., Dance, C. R., Gaussier, E., Welbl, J., Riedel,[´] S., and Bouchard, G. Knowledge graph completion via 

12 

**KNOWFORMER: Revisiting Transformers for Knowledge Graph Reasoning** 

complex tensor factorization. _J. Mach. Learn. Res._ , 18: 130:1–130:38, 2017. 1, 2 

- Tsai, Y. H., Bai, S., Yamada, M., Morency, L., and Salakhutdinov, R. Transformer dissection: An unified understanding for transformer’s attention via the lens of kernel. In _EMNLP/IJCNLP_ , pp. 4343–4352. Association for Computational Linguistics, 2019. 3 

- Vashishth, S., Sanyal, S., Nitin, V., and Talukdar, P. P. Composition-based multi-relational graph convolutional networks. In _ICLR_ . OpenReview.net, 2020. 2, 7 

- Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A. N., Kaiser, L., and Polosukhin, I. Attention is all you need. In _NeurIPS_ , pp. 5998–6008, 2017. 1, 2, 5, 15 

- Wang, C., Zhou, X., Pan, S., Dong, L., Song, Z., and Sha, Y. Exploring relational semantics for inductive knowledge graph completion. In _AAAI_ , pp. 4184–4192. AAAI Press, 2022a. 1 

- Wang, H., Ren, H., and Leskovec, J. Relational message passing for knowledge graph completion. In _KDD_ , pp. 1697–1707. ACM, 2021. 3, 4 

- Wang, L., Zhao, W., Wei, Z., and Liu, J. Simkgc: Simple contrastive knowledge graph completion with pre-trained language models. In _ACL_ , pp. 4281–4294. Association for Computational Linguistics, 2022b. 1, 7, 8 

- Wang, Q., Mao, Z., Wang, B., and Guo, L. Knowledge graph embedding: A survey of approaches and applications. _IEEE Trans. Knowl. Data Eng._ , 29(12):2724–2743, 2017. 1, 7, 8 

- Wang, Z., Song, Y., Wong, G. Y., and See, S. Logical message passing networks with one-hop inference on atomic formulas. In _ICLR_ . OpenReview.net, 2023. 4 

- Weisfeiler, B. and Leman, A. The reduction of a graph to canonical form and the algebra which appears therein. _nti, Series_ , 2(9):12–16, 1968. 6, 16 

- Wu, Q., Zhao, W., Li, Z., Wipf, D. P., and Yan, J. Nodeformer: A scalable graph structure learning transformer for node classification. In _NeurIPS_ , 2022. 9, 15 

- Wu, Q., Yang, C., Zhao, W., He, Y., Wipf, D., and Yan, J. Difformer: Scalable (graph) transformers induced by energy constrained diffusion. In _ICLR_ . OpenReview.net, 2023a. 5, 15 

- Wu, Q., Zhao, W., Yang, C., Zhang, H., Nie, F., Jiang, H., Bian, Y., and Yan, J. Simplifying and empowering transformers for large-graph representations. In _NeurIPS_ , 2023b. 15 

- Wu, Z., Jain, P., Wright, M. A., Mirhoseini, A., Gonzalez, J. E., and Stoica, I. Representing long-range context for graph neural networks with global attention. In _NeurIPS_ , pp. 13266–13279, 2021. 15 

- Xie, X., Zhang, N., Li, Z., Deng, S., Chen, H., Xiong, F., Chen, M., and Chen, H. From discrimination to generation: Knowledge graph completion with generative transformer. In _WWW (Companion Volume)_ , pp. 162– 165. ACM, 2022. 1, 3 

- Xiong, W., Hoang, T., and Wang, W. Y. Deeppath: A reinforcement learning method for knowledge graph reasoning. In _EMNLP_ , pp. 564–573. Association for Computational Linguistics, 2017. 7, 8 

- Xu, K., Hu, W., Leskovec, J., and Jegelka, S. How powerful are graph neural networks? In _ICLR_ . OpenReview.net, 2019. 4, 6, 16, 18 

- Yan, Z., Ma, T., Gao, L., Tang, Z., and Chen, C. Link prediction with persistent homology: An interactive view. In _ICML_ , volume 139 of _Proceedings of Machine Learning Research_ , pp. 11659–11669. PMLR, 2021. 21 

- Yang, B., Yih, W., He, X., Gao, J., and Deng, L. Embedding entities and relations for learning and inference in knowledge bases. In _ICLR (Poster)_ , 2015. 1, 2, 4, 7 

- Yang, F., Yang, Z., and Cohen, W. W. Differentiable learning of logical rules for knowledge base reasoning. In _NeurIPS_ , pp. 2319–2328, 2017. 2 

- Yao, L., Mao, C., and Luo, Y. KG-BERT: BERT for knowledge graph completion. _CoRR_ , abs/1909.03193, 2019. 3 

- You, J., Selman, J. M. G., Ying, R., and Leskovec, J. Identity-aware graph neural networks. In _AAAI_ , pp. 10737–10745. AAAI Press, 2021. 2, 4 

- Zeng, X., Tu, X., Liu, Y., Fu, X., and Su, Y. Toward better drug discovery with knowledge graph. _Curr. Opin. Struct. Biol._ , 72:114–126, 2022. 1 

- Zhang, M. and Chen, Y. Link prediction based on graph neural networks. In _NeurIPS_ , pp. 5171–5181, 2018. 21 

- Zhang, M. and Chen, Y. Inductive matrix completion based on graph neural networks. In _ICLR_ . OpenReview.net, 2020. 7 

- Zhang, M., Li, P., Xia, Y., Wang, K., and Jin, L. Labeling trick: A theory of using graph neural networks for multinode representation learning. In _NeurIPS_ , pp. 9061–9073, 2021. 4 

13 

**KNOWFORMER: Revisiting Transformers for Knowledge Graph Reasoning** 

- Zhang, Y. and Yao, Q. Knowledge graph reasoning with relational digraph. In _WWW_ , pp. 912–924. ACM, 2022. 1, 3, 7, 8, 9 

- Zhang, Y., Zhou, Z., Yao, Q., Chu, X., and Han, B. Adaprop: Learning adaptive propagation for graph neural network based knowledge graph reasoning. In _KDD_ , pp. 3446– 3457. ACM, 2023. 1, 7, 8 

- Zhang, Z., Cai, J., Zhang, Y., and Wang, J. Learning hierarchy-aware knowledge graph embeddings for link prediction. In _AAAI_ , pp. 3065–3072. AAAI Press, 2020. 2 

- Zhu, Z., Zhang, Z., Xhonneux, L. A. C., and Tang, J. Neural bellman-ford networks: A general graph neural network framework for link prediction. In _NeurIPS_ , pp. 29476– 29490, 2021. 1, 3, 4, 7, 8, 9, 21 

- Zhu, Z., Yuan, X., Galkin, M., Xhonneux, S., Zhang, M., Gazeau, M., and Tang, J. A* net: A scalable path-based reasoning approach for knowledge graphs. In _NeurIPS_ , 2023. 1, 7, 8 

14 

**KNOWFORMER: Revisiting Transformers for Knowledge Graph Reasoning** 

## **A. More Preliminaries & Backgrounds** 

**Transformer Architecture.** The Transformer architecture is originally introduced in Vaswani et al. (2017). A vanilla transformer block consists of two main modules: a self-attention module followed by a feed-forward neural network. In the self-attention module, the input feature matrix denoted as _**X** ∈_ R _[n][×][d]_ are projected to query _**Q**_ , key _**K**_ and value _**V**_ using linear projection matrices _**W** q ∈_ R _[d][′][×][n]_ , _**W** k ∈_ R _[d][′][×][n]_ and _**W** v ∈_ R _[d][′][×][n]_ . This is done by _**Q**_ = _**XW** query_ , _**K**_ = _**XW** key_ and _**V**_ = _**XW** value_ , where the bias is omited. The self-attention computation is then given by: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/icml_2024_KnowFormer_Revisiting_Transformers_for_Knowledge_Graph_Reasoning_images/icml_2024_KnowFormer_Revisiting_Transformers_for_Knowledge_Graph_Reasoning.pdf-0015-03.png)


In practice, it is common to utilize _multi-head_ attention, where multiple instances of Equation (13) are concatenated. This approach has demonstrated effectiveness in capturing multi-view interactions. Subsequently, the output of the self-attention is combined with layer normalization (Ba et al., 2016) and a feedforward network (FFN) to form a transformer block. 

**Graph Transformers.** Recently, transformer models (Vaswani et al., 2017) have gained popularity in graph learning due to their ability to capture complex relationships beyond those captured by regular graph neural networks (GNNs) differently. Injecting structural bias into the original attention mechanism is a key problem in graph transformers. Early work by Dwivedi & Bresson (2020) used Laplacian eigenvectors as positional encodings, and since then, several extensions and other models have been proposed (Min et al., 2022; Kim et al., 2022; Ma et al., 2023). Wu et al. (2021) propose a hybrid architecture that uses a stack of message-passing GNN layers followed by regular transformer layers. The most relevant work to this paper is SAT (Chen et al., 2022), which reformulates the self-attention mechanism as a smooth kernel and incorporates structural information by extracting a subgraph representation rooted at each node before attention computation. Computation during the process follows the equations displayed below: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/icml_2024_KnowFormer_Revisiting_Transformers_for_Knowledge_Graph_Reasoning_images/icml_2024_KnowFormer_Revisiting_Transformers_for_Knowledge_Graph_Reasoning.pdf-0015-06.png)


where _f_ ( _**x**_ ) = _**xW** value_ + _bvalue_ , and the non-symmetric kernel _κ_ is defined as: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/icml_2024_KnowFormer_Revisiting_Transformers_for_Knowledge_Graph_Reasoning_images/icml_2024_KnowFormer_Revisiting_Transformers_for_Knowledge_Graph_Reasoning.pdf-0015-08.png)


Moreover, other works focus on the development of scalable models, such as NodeFormer (Wu et al., 2022), GraphGPS (Ramp´asek et al., 2022), DIFFormer (Wu et al., 2023a), and SGFormer (Wu et al., 2023b). 

## **B. Proof of Theorem 4.2** 

_Proof._ Recall that we have the following equations: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/icml_2024_KnowFormer_Revisiting_Transformers_for_Knowledge_Graph_Reasoning_images/icml_2024_KnowFormer_Revisiting_Transformers_for_Knowledge_Graph_Reasoning.pdf-0015-12.png)


where we omit the bias. For convenience, we define _**u**_ = _∥_ _**z**_ � _**z**_ � _uu_ _**WW**_ 1 1 _∥F_[and] _**[ v]**_[=] _∥_ _**z**_ � _**z**_ � _vv_ _**WW**_ 2 2 _∥F_[, yielding:] 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/icml_2024_KnowFormer_Revisiting_Transformers_for_Knowledge_Graph_Reasoning_images/icml_2024_KnowFormer_Revisiting_Transformers_for_Knowledge_Graph_Reasoning.pdf-0015-14.png)


� where _⟨_ _**u** ,_ _**v** ⟩∈_ [ _−_ 1 _,_ 1]. According to the Taylor formula, the expression for _κ_ exp( _**z** u,_ � _**z** v_ ) is as follows: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/icml_2024_KnowFormer_Revisiting_Transformers_for_Knowledge_Graph_Reasoning_images/icml_2024_KnowFormer_Revisiting_Transformers_for_Knowledge_Graph_Reasoning.pdf-0015-16.png)


15 

**KNOWFORMER: Revisiting Transformers for Knowledge Graph Reasoning** 

where _ξ ∈_ (0 _, ⟨_ _**u** ,_ _**v** ⟩_ ). In other words, we have: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/icml_2024_KnowFormer_Revisiting_Transformers_for_Knowledge_Graph_Reasoning_images/icml_2024_KnowFormer_Revisiting_Transformers_for_Knowledge_Graph_Reasoning.pdf-0016-02.png)


Further, by taking _ξ_ = _γ · ⟨_ _**u** ,_ _**v** ⟩_ , where _γ ∈_ (0 _,_ 1), we can rewrite the Lagrange remainder term in Equation (18) as: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/icml_2024_KnowFormer_Revisiting_Transformers_for_Knowledge_Graph_Reasoning_images/icml_2024_KnowFormer_Revisiting_Transformers_for_Knowledge_Graph_Reasoning.pdf-0016-04.png)


Then we have: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/icml_2024_KnowFormer_Revisiting_Transformers_for_Knowledge_Graph_Reasoning_images/icml_2024_KnowFormer_Revisiting_Transformers_for_Knowledge_Graph_Reasoning.pdf-0016-06.png)


## **C. Details of Relational Asymmetric Local 2-WL Test** 

Firstly, we present the general form of the Weisfeiler-Lehman (WL) Test (Weisfeiler & Leman, 1968). Let _G_ = _{V, E, c}_ denotes a graph where _V_ represents the set of nodes, _E_ represents the set of edges, and _c_ represents the node coloring. The node coloring, also known as the feature mapping, is denoted as _c_ : _V →_ R _[d]_ . The purpose of the Weisfeiler-Lehman (WL) test is to detect graph isomorphism. 

**Definition C.1** (Graph Isomorphism) **.** An _isomorphism_ between a graph _G_ = _{V, E, c}_ and _G[′]_ = _{V[′] , E[′] , c[′] }_ is a _bijection f_ : _V →V[′]_ that satisfies _c_ ( _v_ ) = _c[′]_ ( _f_ ( _v_ )) for all _v ∈V_ , and ( _u, v_ ) _∈E_ if and only if ( _f_ ( _u_ ) _, f_ ( _v_ )) _∈E[′]_ for all _u, v ∈V_ . 

**Definition C.2** (WL test) **.** Consider a graph _G_ = _{V, E, c}_ . The WL test is defined as follows: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/icml_2024_KnowFormer_Revisiting_Transformers_for_Knowledge_Graph_Reasoning_images/icml_2024_KnowFormer_Revisiting_Transformers_for_Knowledge_Graph_Reasoning.pdf-0016-11.png)


where _{{·}}_ denotes a multiset, _N_ ( _u_ ) means the neighborhood of node _u_ , and _τ_ maps the pair above injectively to a unique color that has not been used in previous iterations. Repeat the above steps _T_ times until convergence. 

In general, the WL test can be used as a necessary but not sufficient condition for detecting graph isomorphism. Additionally, there exists the WL _k_ ( _k >_ 1) test (Xu et al., 2019), which operates on _k_ -tuples of nodes _**u** ∈V[k]_ and provides enhanced expressive power. 

Now let us delve into the realm of knowledge graphs. Consider a knowledge graph denoted by _G_ = _{V, E, R, c}_ , where _V_ represents the set of entities, _R_ represents the set of relations, _E_ represents the set of facts, and _c_ represents the entity coloring scheme. Similarly, we provide the definition for knowledge graph isomorphism. 

**Definition C.3** (Knowledge Graph Isomorphism) **.** An _isomorphism_ between knowledge graphs _G_ = _{V, E, R, c}_ and _G[′]_ = _{V[′] , E[′] , R[′] , c[′] }_ is a _bijection f_ : _V →V[′]_ that satisfies _c_ ( _v_ ) = _c[′]_ ( _f_ ( _v_ )) for all _v ∈V_ , and _r_ ( _u, v_ ) _∈E_ if and only if _r_ ( _f_ ( _u_ ) _, f_ ( _v_ )) _∈E[′]_ for all _r ∈R_ and _u, v ∈V_ . 

The R-WL test is a relational variant of the WL test, proposed by Barcel´o et al. (2022). It is defined as follows. 

**Definition C.4** (R-WL test) **.** Let _G_ = _{V, E, R, c}_ be a knowledge graph. The R-WL test is defined as: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/icml_2024_KnowFormer_Revisiting_Transformers_for_Knowledge_Graph_Reasoning_images/icml_2024_KnowFormer_Revisiting_Transformers_for_Knowledge_Graph_Reasoning.pdf-0016-18.png)


where _{{·}}_ denotes a multiset, _Nr_ ( _u_ ) refers to the neighborhood of node _u_ corresponding to relation _r_ , and _τ_ injectively maps the aforementioned pair to a unique color not yet used in prior iterations. 

However, the aforementioned tests are not suitable for quantifying the expressive power of methods employed in link prediction, as they require a measurement of a _binary variant_ . 

16 

**KNOWFORMER: Revisiting Transformers for Knowledge Graph Reasoning** 

**Definition C.5** ( _binary variant_ on knowledge graphs) **.** A _binary variant_ on knowledge graphs is represented as a function _ξ_ that associates each knowledge graph _G_ = _{V, E, R, c}_ with a function _ξ_ ( _G_ ) defined on _V_[2] . This function satisfies the condition that for all knowledge graphs _G_ and _G[′]_ , all isomorphisms _f_ from _G_ to _G[′]_ , and all 2-tuples of entities _**u** ∈V_[2] , it holds that _ξ_ ( _G_ )( _**u**_ ) = _ξ_ ( _G[′]_ )( _f_ ( _**u**_ )). 

To analyze the expressive power of knowledge graph reasoning methods, Huang et al. (2023) introduces the _relational asymmetric local 2-WL_ test denoted by RA-WL2. Specifically, RA-WL2 is defined on a knowledge graph _G_ = _{V, E, R, c, η}_ with a _pairwise coloring η_ : _V × V →D_ that satisfies _target node distinguishability_ , meaning _η_ ( _u, u_ ) = _η_ ( _u, v_ ) for all _u ̸_ = _v ∈V_ . Then RA-WL2 is defined as: 

**Definition C.6** (RA-WL2 test) **.** Let _G_ = _{V, E, R, c}_ be a knowledge graph. The RA-WL2 test is defined as: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/icml_2024_KnowFormer_Revisiting_Transformers_for_Knowledge_Graph_Reasoning_images/icml_2024_KnowFormer_Revisiting_Transformers_for_Knowledge_Graph_Reasoning.pdf-0017-04.png)


where _{{·}}_ denotes a multiset, _Nr_ ( _u_ ) means the neighborhood of node _u_ corresponding to relation _r_ , and _τ_ maps the pair above injectively to a unique color that has not been used in previous iterations. 

RA-WL2 test allows us to characterize the power of methods in terms of their ability to distinguish pairs of entities on knowledge graphs. 

## **D. Proof of Theorem 4.3** 

_Proof._ To prove Theorem 4.3, we begin by establishing the equivalence between the query function _fq_ ( _·_ ) and RA-WL2 on original knowledge graph _E_ . The proof is mainly based on the results in Huang et al. (2023). As stated in Huang et al. (2023), a conditional message-passing neural network for inferring the query ( _eh, rq,_ ?) is defined as follows: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/icml_2024_KnowFormer_Revisiting_Transformers_for_Knowledge_Graph_Reasoning_images/icml_2024_KnowFormer_Revisiting_Transformers_for_Knowledge_Graph_Reasoning.pdf-0017-09.png)


where _**h**_[(] _u[t] |_[)] _eh,rq_[represents a pairwise representation corresponding to] _**[ h]**_[(] _r[t] q_[)][(] _[e][h][, u]_[) :] _[ V][× V][→]_[R] _[d]_[.][Here,][ I][NIT][,][ U][PD][,][ A][GG][,] READ, and MSG _r_ are differentiable functions responsible for _initialization_ , _update_ , _aggregation_ , _global readout_ , and _relation-specific message_ computations, respectively. Based on Theorem 5.1 in Huang et al. (2023), it is suggested that a message-passing neural network with an architecture equivalent to Equation (25), featuring a _target node distinguishable_ INIT, exhibits the same expressivity as RA-WL2. 

Now we establish the equivalence of our proposed query function and Equation (25). 

**Lemma D.1.** _The query function fq_ ( _·_ ) _shares the same architecture as the conditional message-passing neural network defined in Equation_ (25) _._ 

_Proof._ From Equation (5), we derive the following: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/icml_2024_KnowFormer_Revisiting_Transformers_for_Knowledge_Graph_Reasoning_images/icml_2024_KnowFormer_Revisiting_Transformers_for_Knowledge_Graph_Reasoning.pdf-0017-14.png)


where we have omitted the READ function in our query function. 

17 

**KNOWFORMER: Revisiting Transformers for Knowledge Graph Reasoning** 

We will now demonstrate that the INIT function in our query function ensures _target node distinguishability_ , which is defined as INIT( _u|u, rq_ ) _̸_ = INIT( _v|u, rq_ ) if _u ̸_ = _v_ for each _v ∈V \ {u}_ . 

**Lemma D.2.** _For each v ∈V \ {u}, there exists_ lim _d→∞ P_ ( _**z**_[(0)] _u_ = _**z**_[(0)] _v_[) = 0] _[, where]_ _**[ z]**_[(0)] _u[,]_ _**[ z]**_[(0)] _v ∈_ R _[d] ._ 

_Proof._ For simplicity, we consider only the random component of _**z**_[(0)] _∈_ R _[d]_ , assuming _**z**_[(0)] [ _i_ ] _∼ N_ (0 _,_ 1). Let _**z**_[(0)] _u_ = [ _zu,_ 0 _, zu,_ 1 _, . . . , zu,d_ ] and _**z**_[(0)] _v_ = [ _zv,_ 0 _, zv,_ 1 _, . . . , zv,d_ ]. We can calculate: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/icml_2024_KnowFormer_Revisiting_Transformers_for_Knowledge_Graph_Reasoning_images/icml_2024_KnowFormer_Revisiting_Transformers_for_Knowledge_Graph_Reasoning.pdf-0018-04.png)


From Lemma D.2, it follows that our INIT function ensures _target node distinguishability_ when employing a sufficiently large dimension _d_ for random features. Based on Lemma D.1 and Lemma D.2, we can derive the following lemma: 

**Lemma D.3.** _The proposed query function fq_ ( _·_ ) _can be as expressive as RA-WL_ 2 _on graph E._ 

Since the initialization in Equation (6) also satisfies the _target node distinguishability_ . By employing a similar approach, we can also derive the following lemma. 

**Lemma D.4.** _The proposed value function fv_ ( _·_ ) _can be as expressive as RA-WL_ 2 _on graph E._ 

To further showcase the expressivity of our proposed attention function on the estimated graph _E_[˜] , we aim to establish the injectiveness of our attention function. The update process of KNOWFORMER’s attention can be reformulated as: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/icml_2024_KnowFormer_Revisiting_Transformers_for_Knowledge_Graph_Reasoning_images/icml_2024_KnowFormer_Revisiting_Transformers_for_Knowledge_Graph_Reasoning.pdf-0018-10.png)


where _**z** u_ means the output of the value function, _{_ _**z**_ **ˆ** _v_ : _v ∈V}_ indicates the weighted values _αuv_ _**z** v_ where _αuv_ is the attention score and _ξ_ is a constant factor which is equal to _|V|_ here. 

We denote the space of **ˆ** _**z**_ as _Z_ . Recall that the query function _fq_ ( _·_ ) shares the same expressive capacity as RA-WL2, enabling it to distinguish the diverse neighbor structures associated with each _u ∈V_ . Consequently, for distinct _u_ values, we can obtain diverse multisets _{_ _**z**_ **ˆ** _v_ : _v ∈V} ⊂Z_ which forms the estimated graph _E_[˜] . 

Further, we have the following theorem: 

**Lemma D.5.** _Assuming Z is countable, the update process of_ KNOWFORMER _’s attention is injective to each pair_ ~~(~~ _**z** v, {_ _**z**_ **ˆ** _v_ : _v ∈V}_ ) _._ 

_Proof._ To prove Lemma D.5, we express the update process as follows: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/icml_2024_KnowFormer_Revisiting_Transformers_for_Knowledge_Graph_Reasoning_images/icml_2024_KnowFormer_Revisiting_Transformers_for_Knowledge_Graph_Reasoning.pdf-0018-16.png)


It has been shown in Lemma 5 and Corollary 6 in Xu et al. (2019) that there exists a function _f_ : _Z →_ R _[d]_ for which Equation (29) is unique for ~~(~~ _**z** v, {_ _**z**_ **ˆ** _v_ : _v ∈V}_ ). The FFN following the attention can be employed to model _g ◦ f_ based on the universal approximation theorem (Hornik et al., 1989; Hornik, 1991). Consequently, Lemma D.5 can be concluded. 

18 

**KNOWFORMER: Revisiting Transformers for Knowledge Graph Reasoning** 

Lemma D.5 provides evidence for the injectiveness of our proposed attention function on the estimated graph _E_[˜] . From the definition of RA-WL2, we can conclude that the attention layer of KNOWFORMER is capable of attaining the same level of expressiveness as RA-WL2, thereby validating Theorem 4.3. 

## **E. Experimental Details** 

## **E.1. Dataset Statistics** 

We conduct experiments on four transductive knowledge graph reasoning datasets, and the statistics of these datasets are summarized in Table 4. Additionally, we perform experiments on three inductive knowledge graph reasoning datasets, each of which contains four different splits. The statistics of the inductive datasets are summarized in Table 5. 

|**Dataset**<br>**#Relation**<br>**#Entity**|**#Triplet**|
|---|---|
||#Train<br>#Valid<br>#Test|
|FB15k-237<br>237<br>14,541<br>WN18RR<br>11<br>40,943<br>NELL-995<br>200<br>74,536<br>YAGO3-10<br>37<br>123,182|272,115<br>17,535<br>20,466<br>86,835<br>3,034<br>3,134<br>149,678<br>543<br>2,818<br>1,079,040<br>5,000<br>5,000|



_Table 4._ Dataset Statistics for transductive knowledge graph reasoning datasets. 

|**Dataset**<br>**#Relation**|**Train**<br>**Validation**<br>**Test**|
|---|---|
||#Entity<br>#Query<br>#Fact<br>#Entity<br>#Query<br>#Fact<br>#Entity<br>#Query<br>#Fact|
|FB15k-237<br>v1<br>180<br>v2<br>200<br>v3<br>215<br>v4<br>219|1,594<br>4,245<br>4,245<br>1,594<br>489<br>4,245<br>1,093<br>205<br>1,993<br>2,608<br>9,739<br>9,739<br>2,608<br>1,166<br>9,739<br>1,660<br>478<br>4,145<br>3,668<br>17,986<br>17,986<br>3,668<br>2,194<br>17,986<br>2,501<br>865<br>7,406<br>4,707<br>27,203<br>27,203<br>4,707<br>3,352<br>27,203<br>3,051<br>1,424<br>11,714|
|WN18RR<br>v1<br>9<br>v2<br>10<br>v3<br>11<br>v4<br>9|2,746<br>5,410<br>5,410<br>2,746<br>630<br>5,410<br>922<br>188<br>1,618<br>6,954<br>15,262<br>15,262<br>6,954<br>1,838<br>15,262<br>2,757<br>441<br>4,011<br>12,078<br>25,901<br>25,901<br>12,078<br>3,097<br>25,901<br>5,084<br>605<br>6,327<br>3,861<br>7,940<br>7,940<br>3,861<br>934<br>7,940<br>7,084<br>1,429<br>12,334|
|NELL-995<br>v1<br>14<br>v2<br>86<br>v3<br>142<br>v4<br>76|3,103<br>4,687<br>4,687<br>3,103<br>414<br>4,687<br>225<br>100<br>833<br>2,564<br>15,262<br>8,219<br>8,219<br>922<br>8,219<br>2,086<br>476<br>4,586<br>4,647<br>16,393<br>16,393<br>4,647<br>1,851<br>16,393<br>3,566<br>809<br>8,048<br>2,092<br>7,546<br>7,546<br>2,092<br>876<br>7,546<br>2,795<br>7,073<br>731|



_Table 5._ Dataset Statistics for inductive knowledge graph reasoning datasets. In each split, one needs to infer #Query triplets based #Fact triplets. 

## **E.2. Evaluation** 

For each dataset, we evaluate the performance of ranking each answer entity against all negative entities given a query ( _h, r,_ ?). The evaluation metrics we use are the mean reciprocal rank (MRR) (Bordes et al., 2013) and Hits at _n_ (H@ _n_ ) (Bordes et al., 2013), which are computed as follows: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/icml_2024_KnowFormer_Revisiting_Transformers_for_Knowledge_Graph_Reasoning_images/icml_2024_KnowFormer_Revisiting_Transformers_for_Knowledge_Graph_Reasoning.pdf-0019-11.png)


where _A_ indicates the answer entity set, I[ _·_ ] indicates the indicative function, and _ranki_ indicates the rank of each answer entity. 

19 

**KNOWFORMER: Revisiting Transformers for Knowledge Graph Reasoning** 

## **E.3. Implement Details** 

Our model is comprised of three modules: the input layer, the attention layer, and the output layer. We present the details for each of them as follows. 

- The input layer consists of the initialization of entity representations and relation representations. We initialize entity representations as all-zero vectors _**X**_ = [0] _[|V|×][d]_ and relation representations as randomly initialized learnable vectors _**R** ∈_ R _[|R|×][d]_ . 

- The attention layer is composed of the attention function which is followed by a skip-connection, an FFN, and two normalization layers before and after the FFN. The attention function consists of three sub-modules as described in Section 4.1. Specifically, each attention function feeds entity representations _**X**_[(] _[l][−]_[1)] and relation representation _**R**_ and outputs the updated entity representation _**X**_[(] _[l]_[)] . The MLPs in the attention function are implemented as threelayer MLPs with ReLU (Glorot et al., 2011) activation. Additionally, We implement FFN as a two-layer MLP with ReLU (Glorot et al., 2011) activation. 

- The output layer is a feed-forward layer for prediction, which maps the entity representations into the predicted scores. The scores will be used for ranking answer entities. 

For each experiment, we employ a fixed random seed and run it multiple times to report the average performance. 

## **E.4. Hyperparameters Setup** 

For each dataset, we performed hyperparameter tuning on the validation set. We considered different values for the learning rate ( _lr_ ) from the set _{_ 1 _e −_ 4 _,_ 5 _e −_ 4 _,_ 1 _e −_ 3 _,_ 5 _e −_ 3 _}_ , weight decay ( _wd_ ) from the set _{_ 0 _,_ 1 _e −_ 6 _,_ 1 _e −_ 5 _,_ 1 _e −_ 4 _}_ , hidden dimension ( _d_ ) from the set _{_ 16 _,_ 32 _,_ 64 _}_ , number of negative samples ( _|_ [ _t[′]_ ] _|_ ) from the set _{_ 2[6] _,_ 2[8] _,_ 2[10] _,_ 2[12] _,_ 2[14] _,_ 2[16] _}_ , number of layers for the query function ( _L_[�] ) from the set _{_ 1 _,_ 2 _,_ 3 _}_ , number of layers for the value function ( _L_[�] ) from the set _{_ 1 _,_ 2 _,_ 3 _}_ , and number of layers for KNOWFORMER ( _L_ ) from the set _{_ 1 _,_ 2 _,_ 3 _}_ . 

## **E.5. Hardcore Configurations** 

We conduct all experiments with: 

- Operating System: Ubuntu 22.04.3 LTS. 

- CPU: Intel (R) Xeon (R) Platinum 8358 CPU @ 2.60GHz with 1TB DDR4 of Memory and Intel Xeon Gold 6148 CPU @ 2.40GHz with 384GB DDR4 of Memory. 

- GPU: NVIDIA Tesla A100 SMX4 with 40GB of Memory and NVIDIA Tesla V100 SXM2 with 32GB of Memory. 

- Software: CUDA 12.1, Python 3.9.14, PyTorch (Paszke et al., 2019) 2.1.0. 

## **F. More Experimental Results** 

## **F.1. KNOWFORMER v.s. ULTRA** 

We present the experimental results of KNOWFORMER and ULTRA in Table 6. It can be observed that KNOWFORMER achieves comparable performance to ULTRA. Notably, ULTRA demonstrates state-of-the-art (SOTA) performance in certain datasets due to its utilization of extensive pretraining data. As mentioned previously, our method can be easily adapted to address full-inductive tasks with minor modifications, making it compatible with a pre-training and fine-tuning paradigm. We intend to explore this avenue in future work. 

## **F.2. Performance of Homogeneous Graph Link Prediction** 

Although KNOWFORMER is primarily designed for knowledge graph reasoning, it can also be applied to general link prediction tasks. In this section, we present experimental results for the homogeneous graph link task using datasets including Cora, Citeseer, and PubMed (Sen et al., 2008). We follow the same settings (baselines and other experimental 

20 

**KNOWFORMER: Revisiting Transformers for Knowledge Graph Reasoning** 

|**Method**|**v1**|**v2**|**v3**|**v4**<br>MRR<br>H@1<br>H@10|
|---|---|---|---|---|
||MRR<br>H@1<br>H@10|MRR<br>H@1<br>H@10|MRR<br>H@1<br>H@10||
|||**WN18RR**|||
|ULTRA (Galkin et al.,2024)|0.685<br>-<br>79.3|0.679<br>-<br>77.9|0.411<br>-<br>54.6|0.614<br>-<br>72.0|
|**KNOWFORMER**|**0.752**<br>**71.5**<br>**81.9**|**0.709**<br>**65.6**<br>**81.7**|**0.467**<br>**40.6**<br>**57.1**|**0.646**<br>**60.9**<br>**72.7**|
|||**NELL-995**|||
|ULTRA (Galkin et al.,2024)|0.757<br>-<br>87.8|**0.575**<br>-<br>**76.1**|**0.563**<br>-<br>**75.5**|**0.469**<br>-<br>**73.3**|
|**KNOWFORMER**|**0.827**<br>**77.0**<br>**93.0**|0.465<br>35.7<br>65.7|0.478<br>37.8<br>65.7|0.378<br>26.7<br>59.8|



_Table 6._ Inductive knowledge graph reasoning performance comparison between KNOWFORMER and ULTRA. The best results are **boldfaced** . Our proposed model, KNOWFORMER is marked by **tan** . 

setup) as in Zhu et al. (2021). Table 7 summarizes the results of KNOWFORMER and baselines on three datasets. We can observe that KNOWFORMER surpasses the baselines on Cora and PubMed and achieves comparable performance on Citeseer, which demonstrates the effectiveness of KNOWFORMER on general graph link prediction tasks. 

|**Method**|**Cora**|**Citeseer**|**PubMed**<br>AUROC<br>AP|
|---|---|---|---|
||AUROC<br>AP|AUROC<br>AP||
|Katz Index (Katz,1953)<br>Personalized PageRank (Page,1998)<br>SimRank (Jeh & Widom,2002)<br>DeepWalk (Perozzi et al.,2014)<br>LINE (Tang et al.,2015)<br>VGAE (Kipf & Welling,2016)<br>S-VGAE (Davidson et al.,2018)<br>SEAL (Zhang & Chen,2018)<br>TLC-GNN (Yan et al.,2021)<br>NBFNet (Zhu et al.,2021)|0.834<br>0.889<br>0.845<br>0.899<br>0.838<br>0.888<br>0.831<br>0.850<br>0.844<br>0.879<br>0.914<br>0.926<br>0.941<br>0.941<br>0.933<br>0.942<br>0.934<br>0.931<br>0.956<br>0.962|0.768<br>0.810<br>0.762<br>0.814<br>0.755<br>0.805<br>0.805<br>0.836<br>0.838<br>0.868<br>0.908<br>0.920<br>**0.947**<br>**0.952**<br>0.905<br>0.924<br>0.909<br>0.916<br>0.923<br>0.936|0.757<br>0.856<br>0.763<br>0.860<br>0.743<br>0.829<br>0.844<br>0.841<br>0.891<br>0.914<br>0.944<br>0.947<br>0.960<br>0.960<br>0.978<br>0.979<br>0.970<br>0.968<br>0.983<br>0.982|
|**KNOWFORMER**|**0.961**<br>**0.965**|0.941<br>0.950|**0.987**<br>**0.988**|



_Table 7._ Results of homogeneous graph link prediction task. The best results are **boldfaced** . Our proposed model, KNOWFORMER is marked by **tan** . Most baseline results are taken from original papers. 

## **F.3. Impact of the Number of Layers** 

The number of layers, denoted as _L_ , _L_[�] , and _L_[�] , is a crucial hyperparameter in KNOWFORMER. To illustrate the influence of the number of layers, we present the experimental results on the transductive WN18RR dataset in Table 8. We can observe that both the number of attention layers and the number of value function layers play a crucial role in determining the reasoning performance of KNOWFORMER. One notable finding is that increasing the number of attention layers can yield significant performance improvements, even in scenarios where the value function layers are limited. This finding further reinforces the effectiveness of the attention layers proposed in this paper. Moreover, increasing the number of query function layers enhances the inference performance, particularly for shallow models ( _e.g._ , limited number of attention layers and value function layers). This observation suggests that the query function is adept at effectively capturing structural information. 

## **F.4. Case Study on KNOWFORMER’s Interpretation** 

To gain a deeper understanding of the attention function of KNOWFORMER, we present visualizations of the attention matrix in Figure 4. These examples demonstrate that KNOWFORMER is capable of effectively capturing the inherent patterns among various entities, which contributes to the enhancement of its performance. For instance, when examining the test fact (Egg _,_ nutrition ~~f~~ act _,_ ?), we observe that KNOWFORMER successfully distinguishes the nutrients of eggs, such as Lipid, Choline, and Riboflvain, aligning with human cognition. 

21 

**KNOWFORMER: Revisiting Transformers for Knowledge Graph Reasoning** 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/icml_2024_KnowFormer_Revisiting_Transformers_for_Knowledge_Graph_Reasoning_images/icml_2024_KnowFormer_Revisiting_Transformers_for_Knowledge_Graph_Reasoning.pdf-0022-01.png)


**----- Start of picture text -----**<br>
(a) (Decca R ecords , artist , ?)<br>(b) (Egg , nutrition f act , ?)<br>(c) (Jewish p eople , ethnicity , ?)<br>(d) (Piano , music g roup , ?)<br>(e) (Bachelor’s d egree , educational d egree , ?)<br>Lipid Choline Riboflavin PotassiumAlpha-TocopherolIsoleucine Valine ThiamineBeta-carotene Leucine<br>CarlSaganLalitaPawar BenBurttSheldonLeonardHowardAshmanKatieCouricLewisBlackAnthonyNewleyLaurenBacallAndyHarries<br>Coldplay Tree Yes MurphysParamore Wilco BandExodus theDiscoPigface<br>Porcupine Brothers at<br>Dropkick Panic!<br>Allman<br>JaneLynch GeneKellyStephenFryBrianMay AlGore RobertGatesDavidCranePhilipYordanChrisPennAlexGibney<br>NatKingRayCharles Camel EricBurdonBenBurtt DallasHowardDarylHannahJamieCullumGarryMarshallSantoDomingo<br>Bryce<br>Norman<br>Jessye<br>acid<br>Pantothenic<br>Feldman<br>Corey<br>Crimson<br>King<br>Moore<br>Julianne<br>**----- End of picture text -----**<br>


_Figure 4._ Visualization of KNOWFORMER attention on FB15k-237 test set. We select the attention matrix corresponding to the answer entity for each test fact and visualize the top-10 entities, excluding the answer entity itself. 

|**#Attention Layers=1**|**#Attention Layers=2**<br>MRR<br>H@1<br>H@10|**#Attention Layers=3**<br>MRR<br>H@1<br>H@10<br>0.543<br>50.1<br>62.5<br>0.572<br>52.0<br>67.6<br>0.578<br>52.4<br>67.8<br>0.572<br>52.1<br>66.8<br>0.576<br>52.5<br>67.5<br>0.578<br>52.2<br>68.1<br>0.568<br>51.7<br>66.8<br>0.575<br>52.0<br>67.9<br>0.577<br>51.9<br>67.3|
|---|---|---|
|MRR<br>H@1<br>H@10|||
|**#Query Layers=1**|||
|0.374<br>36.4<br>39.4<br>0.442<br>41.3<br>49.5|||
|0.425<br>40.5<br>46.1<br>0.554<br>50.7<br>65.0|||
|0.538<br>49.7<br>62.0<br>0.573<br>52.0<br>67.0|||
|**#Query Layers=2**|||
|0.376<br>36.4<br>39.6<br>0.540<br>50.0<br>61.9|||
|0.425<br>40.5<br>46.0<br>0.551<br>50.7<br>62.1|||
|0.539<br>49.8<br>61.9<br>0.572<br>51.5<br>67.2|||
|**#Query Layers=3**|||
|0.379<br>37.0<br>39.7<br>0.554<br>50.7<br>64.6|||
|0.428<br>40.9<br>46.2<br>0.569<br>51.8<br>67.1|||
|0.537<br>49.6<br>61.9<br>0.572<br>52.3<br>67.0|||



_Table 8._ Results of different numbers of layers on the WN18RR dataset, with each metric represented by a different color. The darkness or lightness of the color corresponds to the performance of the metric. 

22 

