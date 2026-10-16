# **Advancing Abductive Reasoning in Knowledge Graphs through Complex Logical Hypothesis Generation** 

**Jiaxin Bai**[1] _[∗]_ **, Yicheng Wang**[1] _[∗]_ **, Tianshi Zheng**[1] **, Yue Guo**[1] **, Xin Liu**[2] **, and Yangqiu Song**[1] 1Hong Kong University of Science and Technology, Clear Water Bay, Hong Kong SAR 

2Amazon.com Inc, Palo Alto, USA 

{jbai, ywangmy, tzhengad, ygouar}@connect.ust.hk, xliucr@amazon.com, yqsong@cse.ust.hk 

## **Abstract** 

Abductive reasoning is the process of making educated guesses to provide explanations for observations. Although many applications require the use of knowledge for explanations, the utilization of abductive reasoning in conjunction with structured knowledge, such as a knowledge graph, remains largely unexplored. To fill this gap, this paper introduces the task of complex logical hypothesis generation, as an initial step towards abductive logical reasoning with KG. In this task, we aim to generate a complex logical hypothesis so that it can explain a set of observations. We find that the supervised trained generative model can generate logical hypotheses that are structurally closer to the reference hypothesis. However, when generalized to unseen observations, this training objective does not guarantee better hypothesis generation. To address this, we introduce the Reinforcement Learning from Knowledge Graph (RLF-KG) method, which minimizes differences between observations and conclusions drawn from generated hypotheses according to the KG. Experiments show that, with RLF-KG’s assistance, the generated hypotheses provide better explanations, and achieve stateof-the-art results on three widely used KGs.[1] 

## **1 Introduction** 

Abductive reasoning plays a vital role in generating explanatory hypotheses for observed phenomena across various research domains (Haig, 2012). It is a powerful tool with wide-ranging applications. For example, in cognitive neuroscience, reverse inference (Calzavarini and Cevolani, 2022), which is a form of abductive reasoning, is crucial for inferring the underlying cognitive processes based on observed brain activation patterns. Similarly, in clinical diagnostics, abductive reasoning is recognized as a key approach for studying cause-and-effect 

> *Equal Contribution 

1https://github.com/HKUST-KnowComp/AbductiveKGR 

relationships (Martini, 2023). Moreover, abductive reasoning is fundamental to the process of hypothesis generation in humans, animals, and computational machines (Magnani, 2023). Its significance extends beyond these specific applications and encompasses diverse fields of study. In this paper, we are focused on abductive reasoning with structured knowledge, specifically, a knowledge graph. 

A typical knowledge graph (KG) stores information about entities, like people, places, items, and their relations in graph structures. Meanwhile, KG reasoning is the process that leverages knowledge graphs to infer or derive new information (Zhang et al., 2021a, 2022; Ji et al., 2022). In recent years, various logical reasoning tasks are proposed over knowledge graph, for example, answering complex queries expressed in logical structure (Hamilton et al., 2018; Ren and Leskovec, 2020), or conducting logical rule mining (Galárraga et al., 2015; Ho et al., 2018; Meilicke et al., 2019). 

However, the abductive perspective of KG reasoning is crucial yet unexplored. Consider the first example in Figure 1, where observation _O_ 1 depicts five celebrities followed by a user on a social media platform. The social network service provider is interested in using structured knowledge to explain the users’ observed behavior. By leveraging a knowledge graph like Freebase (Bollacker et al., 2008), which contains basic information about these individuals, a complex logical hypothesis _H_ 1 can be derived suggesting that they are all actors and screenwriters born in Los Angeles. In the second example from Figure 1, related to a user’s interactions on an e-commerce platform ( _O_ 2), the structured hypothesis _H_ 2 can explain the user’s interest in _Apple_ products released in 2010 excluding _phones_ . The third example, dealing with medical diagnostics, presents three diseases ( _O_ 3). The corresponding hypothesis _H_ 3 indicates they are diseases _V_ ? with symptom _V_ 1, and _V_ 1 can be relieved by _Panadol_ . From a general perspective, 

||||||
|---|---|---|---|---|
||**Observations (O)**|**Hypotheses (H)**|**Interpretations**||
||𝑂1 = {Grant Heslov, Jason Segel,|𝐻1 = 𝑉? ∶𝑂𝑐𝑐𝑢𝑝𝑎𝑡𝑖𝑜𝑛(𝑉?, 𝐴𝑐𝑡𝑜𝑟) ∧|The actors and screenwriters||
||Robert Towne, Ronald Bass,|𝑂𝑐𝑐𝑢𝑝𝑎𝑡𝑖𝑜𝑛<br>𝑉?, 𝑆𝑐𝑟𝑒𝑒𝑒𝑊𝑟𝑖𝑡𝑒𝑟∧|born in Los Angeles||
||Rashida Jones}|𝐵𝑜𝑟𝑛𝐼𝑛<br>𝑉?, 𝐿𝑜𝑠𝐴𝑛𝑔𝑒𝑙𝑒𝑠|||
||𝑂2 = {Ipad 1st Gen, Ipod touch 4th|𝐻2 = 𝑉? ∶𝐵𝑟𝑎𝑛𝑑(𝑉?, 𝐴𝑝𝑝𝑙𝑒) ∧|The Apple products released in||
||Gen, Apple TV 1st Gen}|𝑅𝑒𝑙𝑒𝑎𝑠𝑒𝑌𝑒𝑎𝑟(𝑉?, 2010) ∧¬𝑇𝑦𝑝𝑒<br>𝑉?, 𝑃ℎ𝑜𝑛𝑒|2010 that are notphones||
||𝑂3 = {Covid-19, Seasonal Flu,|𝐻3 = 𝑉?, ∃𝑉1: 𝐻𝑎𝑣𝑒𝑆𝑦𝑚𝑝𝑡𝑜𝑚(𝑉?, 𝑉1)|The disease whose symptoms||
||Dysmenorrhea}|∧𝑅𝑒𝑙𝑖𝑒𝑣𝑒𝑑𝐵𝑦<br>𝑉1, 𝑃𝑎𝑛𝑎𝑑𝑜𝑙|can be relieved byPanadol||



Figure 1: This figure shows some examples of observations and inferred logical hypotheses, expressed with discrepancies in interpretations. 

these problems illustrate how abductive reasoning with knowledge graphs seeks hypotheses that best explain given observations (Josephson and Josephson, 1996; Thagard and Shelley, 1997). 

A straightforward approach to tackle this reasoning task is to employ a search-based method to explore potential hypotheses based on the given observation. However, this approach faces two significant challenges. The first challenge arises from the incompleteness of knowledge graphs (KGs), as searching-based methods heavily rely on the availability of complete information. In practice, missing edges in KGs can negatively impact the performance of search-based methods (Ren and Leskovec, 2020). The second challenge stems from the complexity of logically structured hypotheses. The search space for search-based methods becomes exponentially large when dealing with combinatorial numbers of candidate hypotheses. Consequently, the search-based method struggles to effectively and efficiently handle observations that require complex hypotheses for explanation. 

To overcome these challenges, we propose a solution that leverages generative models within a supervised learning framework to generate logical hypotheses for given observations. Our approach involves sampling hypothesis-observation pairs from observed knowledge graphs (Ren et al., 2020; Bai et al., 2023d) and training a transformerbased generative model (Vaswani et al., 2017) using the teacher-forcing method. However, a potential limitation of supervised training is that it primarily captures structural similarities, without necessarily guaranteeing improved explanations when applied to unseen observations. To address this, we introduce a technique called Reinforcement Learning from the Knowledge Graph (RLF-KG). It utilizes proximal policy optimization (PPO) (Schulman et al., 2017) to minimize the discrepancy between the observed evidence and the conclusion derived from the generated hypothesis. By incorporating 

reinforcement learning techniques, our approach aims to directly improve the explanatory capability of the generated hypotheses and ensure their effectiveness when generalized to unseen observations. 

We evaluate the proposed methods for effectiveness and efficiency on three knowledge graphs: FB15k-237 (Toutanova and Chen, 2015), WN18RR (Toutanova and Chen, 2015), and DBpedia50 (Shi and Weninger, 2018). The results consistently demonstrate the superiority of our approach over supervised generation baselines and search-based methods, as measured by two evaluation metrics across all three datasets. Our contributions can be summarized as follows: 

- We introduce the task of complex logical hypothesis generation, which aims to identify logical hypotheses that best explain a given set of observations. This task can be seen as a form of abductive reasoning with KGs. 

- To address the challenges posed by the incompleteness of knowledge graphs and the complexity of logical hypotheses, we propose a generation-based method. This approach effectively handles these difficulties and enhances the quality of generated hypotheses. 

- Additionally, we developed the Reinforcement Learning from the Knowledge Graph (RLF-KG) technique. By incorporating feedback from the knowledge graph, RLF-KG further improves the hypothesis generation model. It minimizes the discrepancies between the observations and the conclusions of the generated hypotheses, leading to more accurate and reliable results. 

## **2 Problem Formulation** 

In this task, a knowledge graph is denoted as _G_ = ( _V, R_ ), where _V_ is the set of vertices and _R_ is the set of relation types. Each _relation type_ 

_r ∈R_ is a function : _V × V →{_ `true` _,_ `false` _}_ , with _r_ ( _u, v_ ) = `true` indicating the existence of a directed ( _u, r, v_ ) from vertex _u_ to vertex _v_ of type _r_ in the graph , and `false` otherwise. 

We adopt the open-world assumption of knowledge graphs (Drummond and Shearer, 2006), treating missing edges as unknown rather than false. The reasoning model can only access the observed KG _G_ , while the true KG _G_[¯] is hidden and encompasses the observed graph _G_ . 

Abductive reasoning is a type of logical reasoning that involves making educated guesses to infer the most likely reasons for the observations (Josephson and Josephson, 1996; Thagard and Shelley, 1997). For further details on the distinctions between abductive, deductive, and inductive reasoning, refer to Appendix A. In this work, we focus on a specific abductive reasoning type in the context of knowledge graphs. We first introduce some concepts in this context. 

An _observation_ is a set _O_ of entities in _V_ . A logical _hypothesis H_ on a graph _G_ = ( _V, R_ ) is defined as a predicate of a variable vertex _V_ ? in firstorder logical form, including existential quantifiers, AND ( _∧_ ), OR ( _∨_ ), and NOT ( _¬_ ). The hypothesis can always be written in disjunctive normal form, 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/acl_2024_Advancing_Abductive_Reasoning_in_Knowledge_Graphs_through_Complex_Logical_Hypothesis_Generation_images/acl_2024_Advancing_Abductive_Reasoning_in_Knowledge_Graphs_through_Complex_Logical_Hypothesis_Generation.pdf-0003-04.png)


where each _rij_ can take the forms: _rij_ = _r_ ( _v, V_ ), _rij_ = _¬r_ ( _v, V_ ), _rij_ = _r_ ( _V, V[′]_ ), _rij_ = _¬r_ ( _V, V[′]_ ), where _v_ represents a fixed vertex, the _V, V[′]_ represent variable vertices in _{V_ ? _, V_ 1 _, V_ 2 _, . . . , Vk}_ , and _r_ is a relation type. 

The subscript _|G_ denotes that the hypothesis is formulated based on the given graph _G_ . This means that all entities and relations in the hypothesis must exist in _G_ , and the domain for variable vertices is the entity set of _G_ . For example, please refer to Appendix B. The same hypothesis _H_ can be applied to a different knowledge graph, _G[′]_ , provided that _G[′]_ includes the entities and edges present in _H_ . When the context is clear or the hypothesis pertains to a general statement applicable to multiple knowledge graphs (e.g., observed and hidden graphs), the symbol _H_ is used without the subscript. 

The _conclusion_ of the hypothesis _H_ on a graph _G_ , denoted by [[ _H_ ]] _G_ , is the set of entities for which _H_ holds true on _G_ : 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/acl_2024_Advancing_Abductive_Reasoning_in_Knowledge_Graphs_through_Complex_Logical_Hypothesis_Generation_images/acl_2024_Advancing_Abductive_Reasoning_in_Knowledge_Graphs_through_Complex_Logical_Hypothesis_Generation.pdf-0003-08.png)


Suppose _O_ = _{v_ 1 _, v_ 2 _, ..., vk}_ represents an observation, _G_ is the observed graph, and _G_[¯] is the hidden graph. Then _abductive reasoning in knowledge graphs_ aims to find the hypothesis _H_ on _G_ whose conclusion on the hidden graph _G_[¯] , [[ _H_ ]] _G_ ¯, is most similar to _O_ . Formally, the similarity is quantified using the Jaccard index, defined as: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/acl_2024_Advancing_Abductive_Reasoning_in_Knowledge_Graphs_through_Complex_Logical_Hypothesis_Generation_images/acl_2024_Advancing_Abductive_Reasoning_in_Knowledge_Graphs_through_Complex_Logical_Hypothesis_Generation.pdf-0003-10.png)


In other words, the goal is to find a hypothesis _H_ that maximizes `Jaccard` ([[ _H_ ]] _G_ ¯ _, O_ ). 

## **3 Hypothesis Generation with RLF-KG** 

Our methodology, referred to as reinforcement learning from the knowledge graph (RLF-KG), is depicted in Fig. 4 and involves the following steps: (1) Randomly sample observation-hypothesis pairs from the knowledge graph. (2) Train a generative model to generate hypotheses from observations using the pairs. (3) Enhance the generative model using RLF-KG, leveraging reinforcement learning to minimize discrepancies between observations and generated hypotheses. 

## **3.1 Sampling Observation-Hypothesis Pairs** 

In the first step, we randomly sample hypotheses from the observed training knowledge graph. This process starts by randomly selecting a hypothesis, followed by conducting a graph search on the training graph to derive its conclusion, which is then treated as the observation corresponding to the hypothesis. Details of the hypothesis sampling algorithm are provided in Appendix D. 

Then, we convert both hypotheses and observations into sequences suitable for the generative models. For observations, we standardize the order of the elements, ensuring that permutations of the same observation set yield identical outputs. Each entity in an observation is represented as a unique token, such as [Apple] and [Phone], as shown in Figure 2, and is associated with an embedding. 

Since each hypothesis can be represented as a directed acyclic graph, for hypotheses, we use a representation inspired by action-based parsing, similar to approaches seen in other logical reasoning studies (Hamilton et al., 2018; Ren and Leskovec, 2020; Ren et al., 2020). This involves utilizing a depth-first search algorithm, described in Appendix D, to generate a sequence of _actions_ that represents the content and structure of the graph. 

|1<br>2<br>7<br>3<br>5<br>8<br>4<br>6<br>9<br>[Apple]<br>[2010]<br>[Phone]<br>[Type]<br>[Release]<br>[Brand]<br>[I]<br>[I]<br>[N]<br>[I]<br>[I]<br>Hypothesis:𝐻= 𝑉?: 𝐵𝑟𝑎𝑛𝑑<br>𝑉?, 𝐴𝑝𝑝𝑙𝑒∧<br>𝑅𝑒𝑙𝑒𝑎𝑠𝑒(𝑉?, 2010) ∧¬𝑇𝑦𝑝𝑒(𝑉?, 𝑃ℎ𝑜𝑛𝑒)<br>_⇐⇒_|**Nodes**|**Actions**|**Stack**|
|---|---|---|---|
|||||
||1<br>2<br>3<br>4<br>5<br>6<br>7<br>8<br>9|[I]<br>[I]<br>[Brand]<br>[Apple]<br>[Release]<br>[2010]<br>[N]<br>[Type]<br>[Phone]|1<br>1,2<br>1,2,3<br>1,2,3,4_→_1,2<br>1,2,5<br>1,2,5,6_→_1<br>1,7<br>1,7,8<br>1,7,8,9_→_empty|
||Tokens: [I][I][Brand][Apple]<br>[Release][2010][N][Type][Phone]|||



Figure 2: The figure demonstrates the tokenization process for hypotheses. We uniformly consider logical operations, relations, and entities as individual tokens, establishing a correspondence between the hypotheses and a sequence of tokens. For a more detailed explanation, please refer to the Appendix D. 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/acl_2024_Advancing_Abductive_Reasoning_in_Knowledge_Graphs_through_Complex_Logical_Hypothesis_Generation_images/acl_2024_Advancing_Abductive_Reasoning_in_Knowledge_Graphs_through_Complex_Logical_Hypothesis_Generation.pdf-0004-02.png)


**----- Start of picture text -----**<br>
Step 1: Step 2:<br>Sample observation-hypothesis pairs. Train hypothesis generation model by using teacher forcing.<br>KG: Generated Hypotheses<br>Hypothesis Generation Model<br>Hypotheses Observations Observations<br>**----- End of picture text -----**<br>


Figure 3: The first two steps of training a hypothesis generation model: In Step 1, we randomly sample logical hypotheses with diverse patterns and perform graph searches on the training graphs to obtain observations. These observations are then tokenized. In Step 2, a conditional generation model is trained to generate hypotheses based on given tokenized observations. 

Logical operations such as intersection, union, and negation are denoted by special tokens [I], [U], and [N] respectively, following prior work (Bai et al., 2023d). Relations and entities are similarly treated as unique tokens, for example, [Brand] and [Apple]. 

Furthermore, Algorithm 3 facilitates the reconstruction of a graph from an action sequence, serving as the detokenization process for hypotheses. 

## **3.2 Supervised Training of Hypothesis Generation Model** 

In the second step, we train a generative model using the sampled pairs. Let **o** = [ _o_ 1 _, o_ 2 _, ..., om_ ] represent the token sequences for observations, and **h** = [ _h_ 1 _, h_ 2 _, ..., hn_ ] for hypotheses. The loss for the generative model _ρ_ on this example is based on the standard sequence modeling loss: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/acl_2024_Advancing_Abductive_Reasoning_in_Knowledge_Graphs_through_Complex_Logical_Hypothesis_Generation_images/acl_2024_Advancing_Abductive_Reasoning_in_Knowledge_Graphs_through_Complex_Logical_Hypothesis_Generation.pdf-0004-08.png)


We utilize a standard transformer to implement the conditional generative model, employing two distinct approaches. The first approach follows the encoder-decoder architecture as described by Vaswani et al. (2017), where observation tokens are fed into the transformer encoder, and the shifted hypothesis tokens are input into the transformer decoder. The second approach involves concatenating the observation and hypothesis tokens and using a decoder-only transformer to generate hypotheses. Following the setup of these architectures, we train the model using supervised training techniques. 

## **3.3 Reinforcement Learning from Knowledge Graph Feedback (RLF-KG)** 

During the supervised training process, the model learns to generate hypotheses that have similar structures to reference hypotheses. However, higher structural similarity towards reference answers does not necessarily guarantee the ability to generate logical explanations, especially when encountering unseen observations during training. 

To address this limitation, in the third step, 

Step 3: 

Optimize hypothesis generation model with Reinforcement Learning From Knowledge Graph feedback (RLF-KG). 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/acl_2024_Advancing_Abductive_Reasoning_in_Knowledge_Graphs_through_Complex_Logical_Hypothesis_Generation_images/acl_2024_Advancing_Abductive_Reasoning_in_Knowledge_Graphs_through_Complex_Logical_Hypothesis_Generation.pdf-0005-02.png)


**----- Start of picture text -----**<br>
Policy Gradient Optimization<br>Model Log-probabilities  KL-Div<br>Generated Hypothesis Reference Model Log-probabilities  PPO Training<br>KG Hypothesis Conclusion Jaccard<br>Observation<br>**----- End of picture text -----**<br>


Figure 4: In Step 3, we employ RLF-KG to encourage the model to generate hypotheses that align more closely with the given observations from the knowledge graph. RLF-KG helps improve the consistency between the generated hypotheses and the observed evidence in the knowledge graph. 

we employ reinforcement learning (Ziegler et al., 2020) in conjunction with knowledge graph feedback (RLF-KG) to enhance the trained conditional generation model _ρ_ . Let _G_ train represents the observed training graph, **h** the hypothesis token sequence, **o** the observation token sequence, and _H, O_ the corresponding hypothesis and observation, respectively. We select the reward to be the Jaccard similarity between _O_ and and the conclusion [[ _H_ ]] _G_ train, which is a reliable and information leakage-free approximation for the objective of the abductive reasoning task in Equation 4. Formally, the reward function is defined as 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/acl_2024_Advancing_Abductive_Reasoning_in_Knowledge_Graphs_through_Complex_Logical_Hypothesis_Generation_images/acl_2024_Advancing_Abductive_Reasoning_in_Knowledge_Graphs_through_Complex_Logical_Hypothesis_Generation.pdf-0005-05.png)


Following Ziegler et al. (2020), we treat the trained model _ρ_ obtained from supervised training as the reference model, and initialize the model _π_ to be optimized with _ρ_ . Then, we modify the reward function by incorporating a KL divergence penalty. This modification aims to discourage the model _π_ from generating hypotheses that deviate excessively from the reference model. 

To train the model _π_ , we employ the proximal policy optimization (PPO) algorithm (Schulman et al., 2017). The objective is to maximize the expected modified reward, given as follows: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/acl_2024_Advancing_Abductive_Reasoning_in_Knowledge_Graphs_through_Complex_Logical_Hypothesis_Generation_images/acl_2024_Advancing_Abductive_Reasoning_in_Knowledge_Graphs_through_Complex_Logical_Hypothesis_Generation.pdf-0005-08.png)


where _D_ the is training observation distribution and _π_ ( _·|_ **o** ) is the conditional distribution of **h** on **o** modeled by the model _π_ . By utilizing PPO and the modified reward function, we can effectively guide the model _π_ towards generating hypotheses that strike a balance between the similarity to the reference model and logical coherence. 

## **4 Experiment** 

We utilize three distinct knowledge graphs, namely FB15k-237 (Toutanova and Chen, 2015), DBpedia50 (Shi and Weninger, 2018), and WN18RR (Toutanova and Chen, 2015), for our experiments. Table 1 provides an overview of the number of training, evaluation, and testing edges, as well as the total number of nodes in each knowledge graph. To ensure consistency, we randomly partition the edges of these knowledge graphs into three sets - training, validation, and testing - using an 8:1:1 ratio. Consequently, we construct the training graph _G_ train, validation graph _G_ valid, and testing graph _G_ test by including the corresponding edges: training edges only, training + validation edges, and training + validation + testing edges, respectively. 

Following the methodology outlined in Section 3.2, we proceed to sample pairs of observations and hypotheses. To ensure the quality and diversity of the samples, we impose certain constraints during the sampling process. Firstly, we restrict the size of the observation sets to a maximum of thirty-two elements. This limitation is enforced ensuring that the observations remain manageable. Additionally, specific constraints are applied to the validation and testing hypotheses. Each validation hypothesis must incorporate additional entities in the conclusion compared to the training graph, while each testing hypothesis must have additional entities in the conclusion compared to the validation graph. This progressive increase in entity complexity ensures a challenging evaluation setting. The statistics of queries sampled for datasets are detailed in Appendix C. 

In line with previous work on KG reasoning (Ren and Leskovec, 2020; Ren et al., 2020), we utilize thirteen pre-defined logical patterns to sam- 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/acl_2024_Advancing_Abductive_Reasoning_in_Knowledge_Graphs_through_Complex_Logical_Hypothesis_Generation_images/acl_2024_Advancing_Abductive_Reasoning_in_Knowledge_Graphs_through_Complex_Logical_Hypothesis_Generation.pdf-0006-00.png)


**----- Start of picture text -----**<br>
u<br>1p 2p 2i 3i ip pi 2u<br>n<br>u<br>n n n n<br>2in 3in inp pni pin up<br>**----- End of picture text -----**<br>


Figure 5: Our task involves considering thirteen distinct types of logical hypotheses. Each hypothesis type corresponds to a specific type of query graph, which is utilized during the sampling process. By associating each hypothesis type with a corresponding query graph, we ensure that a diverse range of hypotheses is sampled. 

|**Dataset**|**Relations**<br>**Entities**|**Training**<br>**Validation**<br>**Testing**|**All Edges**|
|---|---|---|---|
|||||
|FB15k-237<br>WN18RR<br>DBpedia50|237<br>14,505<br>11<br>40,559<br>351<br>24,624|496,126<br>62,016<br>62,016<br>148,132<br>18,516<br>18,516<br>55,074<br>6,884<br>6,884|620,158<br>185,164<br>68,842|



Table 1: This figure provides basic information about the three knowledge graphs utilized in our experiments. The graphs are divided into standard sets of training, validation, and testing edges to facilitate the evaluation process. 

ple the hypotheses. Eight of these patterns, known as existential positive first-order (EPFO) hypotheses (1p/2p/2u/3i/ip/up/2i/pi), do not involve negations. The remaining five patterns are negation hypotheses (2in/3in/inp/pni/pin), which incorporate negations. It is important to note that the generated hypotheses may or may not match the type of the reference hypothesis. The structures of the hypotheses are visually presented in Figure 5, and the corresponding numbers of samples drawn for each hypothesis type can be found in Table 5. 

## **4.1 Evaluation Metric** 

**Jaccard Index** The quality of the generated hypothesis is primarily measured using the Jaccard index same as that defined for abductive reasoning in Section 2, but we treat the constructed test graph _G_ test as the hidden graph. It is important to note that the test graph contains ten percent of edges that were not observed during the training or validation stages. Formally, given an observation _O_ and a generated hypothesis _H_ , we employ a graph search algorithm to determine the conclusion of _H_ on _G_ test, denoted as [[ _H_ ]] _G_ test. Then the Jaccard metric for evaluation is defined as 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/acl_2024_Advancing_Abductive_Reasoning_in_Knowledge_Graphs_through_Complex_Logical_Hypothesis_Generation_images/acl_2024_Advancing_Abductive_Reasoning_in_Knowledge_Graphs_through_Complex_Logical_Hypothesis_Generation.pdf-0006-07.png)


quantifing the similarity between the conclusion [[ _H_ ]] _G_ test and the observation _O_ . 

**Smatch Score** Smatch (Cai and Knight, 2013) is originally designed for comparing semantic graphs but has been recognized as a suitable metric for 

evaluating complex logical queries, which can be treated as a specialized form of semantic graphs (Bai et al., 2023d). In this task, a hypothesis can be represented as a graph, e.g. Fig. 2, and we can transform it to be compatible with the semantic graph format. The detailed computation of the Smatch score on hypothesis graphs is described in detail in Appendix F. Intuitively, the Smatch score between the generated hypothesis _H_ and the reference hypothesis _H_ ref, denoted as _S_ ( _H, H_ ref), quantifies the structural resemblance between the graphs corresponding to _H_ and _H_ ref i.e., how much the nodes, edges, and the labels on them look the same between the two graphs. 

## **4.2 Experiment Details** 

In this experiment, we use two transformer structures as the backbones of the generation model. For the encoder-decoder transformer structure, we use three encoder layers and three decoder layers. Each layer has eight attention heads with a hidden size of 512. Note that the positional encoding for the input observation sequence is disabled, as we believe that the order of the entities in the observation set does not matter. For the decoder-only structure, we use six layers, and the other hyperparameters are the same. In the supervised training process, we use AdamW optimizer and grid search to find hyper-parameters. For the encoderdecoder structure, the learning rate is 0 _._ 0001 with the resulting batch size of 768, 640, and 256 for FB15k-237, WN18RR, and DBpedia, respectively. For the decoder-only structure, the learning rate is 

|**Dataset**||**Model**|**1p**<br>**2p**<br>**2i**<br>**3i**<br>**ip**<br>**pi**<br>**2u**<br>**up**|**2in**<br>**3in**<br>**pni**<br>**pin**<br>**inp**|**Ave.**|
|---|---|---|---|---|---|
|FB15k-237||||||
|||Encoder-Decoder<br>+ RLF-KG|0.626<br>0.617<br>0.551<br>0.513<br>0.576<br>0.493<br>0.818<br>0.613<br>**0.855**<br>**0.711**<br>**0.661**<br>**0.595**<br>**0.715**<br>**0.608**<br>**0.776**<br>**0.698**|0.532<br>0.451<br>0.499<br>0.529<br>0.533<br>**0.670**<br>**0.530**<br>**0.617**<br>**0.590**<br>**0.637**|0.565<br>**0.666**|
|||||||
|||Decoder-Only<br>+ RLF-KG|0.666<br>0.643<br>0.593<br>0.554<br>0.612<br>0.533<br>0.807<br>0.638<br>**0.789**<br>**0.681**<br>**0.656**<br>**0.605**<br>**0.683**<br>**0.600**<br>**0.817**<br>**0.672**|0.588<br>0.503<br>0.549<br>0.559<br>0.564<br>**0.672**<br>**0.560**<br>**0.627**<br>**0.596**<br>**0.626**|0.601<br>**0.660**|
|WN18RR||||||
|||Encoder-Decoder<br>+ RLF-KG|0.793<br>0.734<br>0.692<br>0.692<br>0.797<br>0.627<br>0.763<br>0.690<br>**0.850**<br>**0.778**<br>**0.765**<br>**0.763**<br>**0.854**<br>**0.685**<br>**0.767**<br>**0.719**|0.707<br>0.694<br>0.704<br>0.653<br>0.664<br>**0.743**<br>**0.732**<br>**0.738**<br>**0.682**<br>**0.710**|0.708<br>**0.753**|
|||||||
|||Decoder-Only<br>+ RLF-KG|0.760<br>0.734<br>0.680<br>0.684<br>0.770<br>0.614<br>0.725<br>0.650<br>**0.821**<br>**0.760**<br>**0.694**<br>**0.693**<br>**0.827**<br>**0.656**<br>**0.770**<br>**0.680**|0.683<br>0.672<br>0.688<br>0.660<br>0.677<br>**0.717**<br>**0.704**<br>**0.720**<br>**0.676**<br>**0.721**|0.692<br>**0.726**|
|DBpedia50||||||
|||Encoder-Decoder<br>+ RLF-KG|0.706<br>0.657<br>0.551<br>0.570<br>0.720<br>0.583<br>0.632<br>0.636<br>**0.842**<br>**0.768**<br>**0.636**<br>**0.639**<br>**0.860**<br>**0.667**<br>**0.714**<br>**0.758**|0.602<br>0.572<br>0.668<br>0.625<br>0.636<br>**0.699**<br>**0.661**<br>**0.775**<br>**0.716**<br>**0.769**|0.627<br>**0.731**|
|||||||
|||Decoder-Only|0.739<br>0.692<br>0.426<br>0.434<br>0.771<br>0.527<br>**0.654**<br>0.688|0.602<br>0.563<br>0.663<br>**0.640**<br>0.701|0.623|
|||+ RLF-KG|**0.777**<br>**0.701**<br>**0.470**<br>**0.475**<br>**0.821**<br>**0.534**<br>0.646<br>**0.702**|**0.626**<br>**0.575**<br>**0.696**<br>0.626<br>**0.713**|**0.643**|
|||Table 2: The detailed Jaccard performance of various methods.||||




![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/acl_2024_Advancing_Abductive_Reasoning_in_Knowledge_Graphs_through_Complex_Logical_Hypothesis_Generation_images/acl_2024_Advancing_Abductive_Reasoning_in_Knowledge_Graphs_through_Complex_Logical_Hypothesis_Generation.pdf-0007-01.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/acl_2024_Advancing_Abductive_Reasoning_in_Knowledge_Graphs_through_Complex_Logical_Hypothesis_Generation_images/acl_2024_Advancing_Abductive_Reasoning_in_Knowledge_Graphs_through_Complex_Logical_Hypothesis_Generation.pdf-0007-02.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/acl_2024_Advancing_Abductive_Reasoning_in_Knowledge_Graphs_through_Complex_Logical_Hypothesis_Generation_images/acl_2024_Advancing_Abductive_Reasoning_in_Knowledge_Graphs_through_Complex_Logical_Hypothesis_Generation.pdf-0007-03.png)


Figure 6: The curve of the reward values of RLF-KG training over three different datasets. 

0 _._ 00001 with batch-size of 256, 160, and 160 for FB15k-237, WN18RR, and DBpedia respectively, and linear warming up of 100 steps. In the reinforcement learning process, we use the dynamic adjustment of the penalty coefficient _β_ (Ouyang et al., 2022). More detailed hyperparameters are shown in Appendix I. All the experiments can be conducted on a single GPU with 24GB memory. 

## **4.3 Experiment Results and Discussions** 

We validate RLF-KG effectiveness by comparing the Jaccard metric of the model before and after this process. Table 2 displays performance across thirteen hypothesis types on FB15k-237, WN18RR, and DBpedia50. It illustrates Jaccard indices between observations and conclusions of generated hypotheses from the test graph. The encoder-decoder and the decoder-only transformers are assessed under fully supervised training on each dataset. Additionally, performance is reported when models collaborate with reinforcement learning from knowledge graph feedback (RLF-KG). 

## **4.3.1 Performance Gain after RLF-KG** 

We notice RLF-KG consistently enhances hypothesis generation across three datasets, improving both encoder-decoder and decoder-only models. This can be explained by RLF-KG’s ability to incorporate knowledge graph information into the generation model, diverging from simply generating hypotheses akin to reference hypotheses. 

Additionally, after the RLF-KG training, the encoder-decoder model surpasses the decoder-only structured transformer model. This is due to the task’s nature, where generating a sequence of tokens from an observation set does not necessitate the order of the observation set. Figure 6 supplements the previous statement by illustrating the increasing reward throughout the PPO process. We also refer readers to Appendix K for qualitative examples demonstrating the improvement in the generated hypotheses for the same observation. 

## **4.3.2 Adding Structural Reward to PPO** 

We explore the potential benefits of incorporating structural similarity into the reward function used in PPO training. While RLF-KG originally relies on the Jaccard index, we consider adding the Smatch score, a measure of structural differences between generated and sampled hypotheses. We conducted additional experiments to include _S_ ( _H, H_ ref) as an extra term in the reward function. 

The results, presented in Table 3, indicate that by incorporating the structural reward, the model can indeed generate hypotheses that are closer to the reference hypotheses yc in structural sense. However, the Jaccard scores reveal that with the inclusion of structural information, the overall performance is comparable to or slightly worse than the original reward function. This is because adding Smatch score tends to cause the generation model to fit or potentially overfit the training data accord- 

||**FB15k-237**<br>Jaccard<br>Smatch|**WN18RR**<br>Jaccard<br>Smatch|**DBpedia50**<br>Jaccard<br>Smatch|
|---|---|---|---|
|||||
|Encoder-Decoder<br>+ RLF-KG (Jaccard)<br>+ RLF-KG (Jaccard + Smatch)|0.565<br>**0.602**<br>**0.666**<br>0.530<br>0.660<br>0.568|0.708<br>**0.558**<br>0.753<br>0.540<br>**0.757**<br>0.545|0.627<br>0.486<br>**0.731**<br>**0.541**<br>0.696<br>0.532|
|||||
|Decoder-Only<br>+ RLF-KG (Jaccard)<br>+ RLF-KG (Jaccard + Smatch)|0.601<br>**0.614**<br>**0.660**<br>0.598<br>0.656<br>0.612|0.692<br>**0.564**<br>**0.726**<br>0.518<br>0.713<br>0.540|0.623<br>**0.510**<br>0.643<br>0.492<br>**0.645**<br>0.504|



Table 3: The Jaccard and Smatch performance of different reward functions. 

|**Method**|**FB15k-237**<br>Runtime<br>Jaccard<br>Smatch|**WN18RR**<br>Runtime<br>Jaccard<br>Smatch|**DBpedia50**<br>Runtime<br>Jaccard<br>Smatch|
|---|---|---|---|
|||||
|Brute-force Search<br>Generation + RLF-KG|16345 mins<br>0.635<br>0.305<br>**264 mins**<br>**0.666**<br>**0.530**|4084 mins<br>0.742<br>0.322<br>**32 mins**<br>**0.753**<br>**0.540**|1132 mins<br>0.702<br>0.322<br>**5 mins**<br>**0.731**<br>**0.541**|



Table 4: Performance on testing data and runtime for inference for various methods on testing data. 

ing to the above graph-level similarity, limiting its ability to generalize to unseen test graphs. The Jaccard reward (Eq. 7), however, captures a measure closer to the goal of our task . Detailed Smatch scores by query types can be found in Appendix G. 

on the test graph using the training graph for 1p queries is found to be quite accurate, which enhances the average score. The detailed Jaccard scores are presented in Appendix H. However, the brute-force search struggles with more complex types of queries. 

## **4.3.3 Comparison Between Search Methods** 

We chose the relatively simple brute-force search (Algorithm 4) as our search-based baseline due to the inherent high complexity of search algorithms for this task. For each observation, the algorithm explores all potential 1p hypotheses within the training graph and selects the one with the highest Jaccard similarity on the training graph. Despite its simplicity and complexity linear to the number of edges, the algorithm requires significantly more time than our method, not to mention other more complex heuristics. 

Following this choice, we compare inference time and performance between the generationbased and the brute-force search algorithm. Table 4 highlights the unsuitability of the brute-force search for scaling up due to its high complexity. In contrast, generation-based methods demonstrate substantially faster performance. 

Moreover, the generation methods not only outperform the search-based method in Jaccard performance but also show a significant improvement in Smatch performance. The relatively high Jaccard score of the brute-force search is attributed to its inherent access to an approximation of Jaccard during the search process. Due to the nature of the graph splits, the approximation of Jaccard 

## **5 Related Work** 

The problem of abductive knowledge graph reasoning shares connections with various other knowledge graph reasoning tasks, including knowledge graph completion, complex logical query answering, and rule mining. 

**Rule Mining** Rule mining focuses on inductive logical reasoning, namely discovering logical rules over the knowledge graph. Various methods are proposed in this line of work (Galárraga et al., 2015; Ho et al., 2018; Meilicke et al., 2019; Cheng et al., 2022, 2023). While the direct application of rule mining to solve this task is theoretically applicable, rule mining algorithms such as those described by Galárraga et al. (2015) also rely on searches to construct the horn clauses, which face scalability issues. 

**Complex logical query** Complex logical query answering is a task of answering logically structured queries on KG. Query embedding primary focus is the enhancement of embedding structures for encoding sets of answers (Hamilton et al., 2018; Sun et al., 2020; Liu et al., 2021; Bai et al., 2023c,a,b; Hu et al., 2023; Yin et al., 2023; Wang et al., 2023a,b; Hu et al., 2024; Liu et al., 2024). 

For instance, Ren and Leskovec (2020) and Zhang et al. (2021b) introduce the utilization of geometric structures such as rectangles and cones within hyperspace to represent entities. Neural MLP (Mixer) (Amayuelas et al., 2022) use MLP and MLP-Mixer as the operators. (Bai et al., 2022) suggests employing multiple vectors to encode queries, thereby addressing the diversity of answer entities. FuzzQE (Chen et al., 2022) uses fuzzy logic to represent logical operators. Probabilistic distributions can also serve as a means of query encoding (Choudhary et al., 2021a,b), with examples including Beta Embedding (Ren and Leskovec, 2020) and Gamma Embedding (Yang et al., 2022). 

## **6 Conclusion** 

In summary, this paper has introduced the task of abductive logical knowledge graph reasoning. Meanwhile, this paper has proposed a generationbased method to address knowledge graph incompleteness and reasoning efficiency by generating logical hypotheses. Furthermore, this paper demonstrates the effectiveness of our proposed reinforcement learning from knowledge graphs (RLF-KG) to enhance our hypothesis generation model by leveraging feedback from knowledge graphs. 

RMGS20CR11, RMGS20CR12, RMGS20EG19, RMGS20EG21, RMGS23CR05, RMGS23EG08). 

## **References** 

- Alfonso Amayuelas, Shuai Zhang, Susie Xi Rao, and Ce Zhang. 2022. Neural methods for logical reasoning over knowledge graphs. In _The Tenth International Conference on Learning Representations, ICLR 2022, Virtual Event, April 25-29, 2022_ . OpenReview.net. 

- Jiaxin Bai, Xin Liu, Weiqi Wang, Chen Luo, and Yangqiu Song. 2023a. Complex query answering on eventuality knowledge graph with implicit logical constraints. In _Advances in Neural Information Processing Systems 36: Annual Conference on Neural Information Processing Systems 2023, NeurIPS 2023, New Orleans, LA, USA, December 10 - 16, 2023_ . 

- Jiaxin Bai, Chen Luo, Zheng Li, Qingyu Yin, and Yangqiu Song. 2023b. Understanding inter-session intentions via complex logical reasoning. _CoRR_ , abs/2312.13866. 

- Jiaxin Bai, Chen Luo, Zheng Li, Qingyu Yin, Bing Yin, and Yangqiu Song. 2023c. Knowledge graph reasoning over entities and numerical values. In _Proceedings of the 29th ACM SIGKDD Conference on Knowledge Discovery and Data Mining, KDD 2023, Long Beach, CA, USA, August 6-10, 2023_ , pages 57–68. ACM. 

## **Limitations** 

Our proposed methods and techniques in the paper are evaluated on a specific set of knowledge graphs, namely FB15k-237, WN18RR, and DBpedia50. It is unclear how well these approaches would perform on other KGs with different characteristics or domains. Meanwhile, knowledge graphs can be massive and continuously evolving, our method is not yet able to address the dynamic nature of knowledge evolutions, like conducting knowledge editing automatically. It is important to note that these limitations should not undermine the significance of the work but rather serve as areas for future research and improvement. 

## **Acknowledgements** 

We thank the anonymous reviewers and the area chair for their constructive comments. The authors of this paper were supported by the NSFC Fund (U20B2053) from the NSFC of China, the RIF (R6020-19 and R6021-20), and the GRF (16211520 and 16205322) from RGC of Hong Kong. We also thank the support from the UGC Research Matching Grants (RMGS20EG01-D, 

- Jiaxin Bai, Zihao Wang, Hongming Zhang, and Yangqiu Song. 2022. Query2Particles: Knowledge graph reasoning with particle embeddings. In _Findings of the Association for Computational Linguistics: NAACL 2022_ , pages 2703–2714, Seattle, United States. Association for Computational Linguistics. 

- Jiaxin Bai, Tianshi Zheng, and Yangqiu Song. 2023d. Sequential query encoding for complex query answering on knowledge graphs. _Transactions on Machine Learning Research_ . 

- Kurt D. Bollacker, Colin Evans, Praveen K. Paritosh, Tim Sturge, and Jamie Taylor. 2008. Freebase: a collaboratively created graph database for structuring human knowledge. In _Proceedings of the ACM SIGMOD International Conference on Management of Data, SIGMOD 2008, Vancouver, BC, Canada, June 10-12, 2008_ , pages 1247–1250. ACM. 

- Shu Cai and Kevin Knight. 2013. Smatch: an evaluation metric for semantic feature structures. In _Proceedings of the 51st Annual Meeting of the Association for Computational Linguistics, ACL 2013, 4-9 August 2013, Sofia, Bulgaria, Volume 2: Short Papers_ , pages 748–752. The Association for Computer Linguistics. 

- Fabrizio Calzavarini and Gustavo Cevolani. 2022. Abductive reasoning in cognitive neuroscience: Weak and strong reverse inference. _Synthese_ , 200(2):1–26. 

- Xuelu Chen, Ziniu Hu, and Yizhou Sun. 2022. Fuzzy logic based logical query answering on knowledge graphs. In _Thirty-Sixth AAAI Conference on Artificial Intelligence, AAAI 2022, Thirty-Fourth Conference on Innovative Applications of Artificial Intelligence, IAAI 2022, The Twelveth Symposium on Educational Advances in Artificial Intelligence, EAAI 2022 Virtual Event, February 22 - March 1, 2022_ , pages 3939– 3948. AAAI Press. 

- Kewei Cheng, Nesreen K. Ahmed, and Yizhou Sun. 2023. Neural compositional rule learning for knowledge graph reasoning. In _The Eleventh International Conference on Learning Representations, ICLR 2023, Kigali, Rwanda, May 1-5, 2023_ . OpenReview.net. 

- Kewei Cheng, Jiahao Liu, Wei Wang, and Yizhou Sun. 2022. Rlogic: Recursive logical rule learning from knowledge graphs. In _KDD ’22: The 28th ACM SIGKDD Conference on Knowledge Discovery and Data Mining, Washington, DC, USA, August 14 - 18, 2022_ , pages 179–189. ACM. 

- Nurendra Choudhary, Nikhil Rao, Sumeet Katariya, Karthik Subbian, and Chandan K. Reddy. 2021a. Probabilistic entity representation model for reasoning over knowledge graphs. In _Advances in Neural Information Processing Systems 34: Annual Conference on Neural Information Processing Systems 2021, NeurIPS 2021, December 6-14, 2021, virtual_ , pages 23440–23451. 

- Nurendra Choudhary, Nikhil Rao, Sumeet Katariya, Karthik Subbian, and Chandan K. Reddy. 2021b. Self-supervised hyperboloid representations from logical queries over knowledge graphs. In _WWW_ 

- _’21: The Web Conference 2021, Virtual Event / Ljubljana, Slovenia, April 19-23, 2021_ , pages 1373–1384. ACM / IW3C2. 

- Igor Douven. 2021. Abduction. In Edward N. Zalta, editor, _The Stanford Encyclopedia of Philosophy_ , Summer 2021 edition. Metaphysics Research Lab, Stanford University. 

- Nick Drummond and Rob Shearer. 2006. The open world assumption. In _eSI Workshop: The Closed World of Databases meets the Open World of the Semantic Web_ , volume 15, page 1. 

- Luis Galárraga, Christina Teflioudi, Katja Hose, and Fabian M. Suchanek. 2015. Fast rule mining in ontological knowledge bases with AMIE+. _VLDB J._ , 24(6):707–730. 

- Brian D. Haig. 2012. _Abductive Learning_ , pages 10–12. Springer US, Boston, MA. 

- William L. Hamilton, Payal Bajaj, Marinka Zitnik, Dan Jurafsky, and Jure Leskovec. 2018. Embedding logical queries on knowledge graphs. In _Advances in Neural Information Processing Systems 31: Annual Conference on Neural Information Processing Systems 2018, NeurIPS 2018, December 3-8, 2018, Montréal, Canada_ , pages 2030–2041. 

- Vinh Thinh Ho, Daria Stepanova, Mohamed H. GadElrab, Evgeny Kharlamov, and Gerhard Weikum. 2018. Rule learning from knowledge graphs guided by embedding models. In _The Semantic Web - ISWC 2018 - 17th International Semantic Web Conference, Monterey, CA, USA, October 8-12, 2018, Proceedings, Part I_ , volume 11136 of _Lecture Notes in Computer Science_ , pages 72–90. Springer. 

- Qi Hu, Weifeng Jiang, Haoran Li, Zihao Wang, Jiaxin Bai, Qianren Mao, Yangqiu Song, Lixin Fan, and Jianxin Li. 2024. Fedcqa: Answering complex queries on multi-source knowledge graphs via federated learning. _CoRR_ , abs/2402.14609. 

- Qi Hu, Haoran Li, Jiaxin Bai, and Yangqiu Song. 2023. Privacy-preserving neural graph databases. _CoRR_ , abs/2312.15591. 

- Shaoxiong Ji, Shirui Pan, Erik Cambria, Pekka Marttinen, and Philip S. Yu. 2022. A survey on knowledge graphs: Representation, acquisition, and applications. _IEEE Trans. Neural Networks Learn. Syst._ , 33(2):494–514. 

- John R Josephson and Susan G Josephson. 1996. _Abductive inference: Computation, philosophy, technology_ . Cambridge University Press. 

- Lihui Liu, Boxin Du, Heng Ji, ChengXiang Zhai, and Hanghang Tong. 2021. Neural-answering logical queries on knowledge graphs. In _KDD ’21: The 27th ACM SIGKDD Conference on Knowledge Discovery and Data Mining, Virtual Event, Singapore, August 14-18, 2021_ , pages 1087–1097. ACM. 

- Lihui Liu, Zihao Wang, Jiaxin Bai, Yangqiu Song, and Hanghang Tong. 2024. New frontiers of knowledge graph reasoning: Recent advances and future trends. In _Companion Proceedings of the ACM on Web Conference 2024, WWW 2024, Singapore, Singapore, May 13-17, 2024_ , pages 1294–1297. ACM. 

- Lorenzo Magnani. 2023. _Handbook of Abductive Cognition_ . Springer International Publishing. 

- Carlo Martini. 2023. _Abductive Reasoning in Clinical Diagnostics_ , pages 467–479. Springer International Publishing, Cham. 

- Christian Meilicke, Melisachew Wudage Chekol, Daniel Ruffinelli, and Heiner Stuckenschmidt. 2019. Anytime bottom-up rule learning for knowledge graph completion. In _Proceedings of the Twenty-Eighth International Joint Conference on Artificial Intelligence, IJCAI 2019, Macao, China, August 10-16, 2019_ , pages 3137–3143. ijcai.org. 

- Long Ouyang, Jeffrey Wu, Xu Jiang, Diogo Almeida, Carroll L. Wainwright, Pamela Mishkin, Chong Zhang, Sandhini Agarwal, Katarina Slama, Alex Ray, John Schulman, Jacob Hilton, Fraser Kelton, Luke Miller, Maddie Simens, Amanda Askell, Peter Welinder, Paul F. Christiano, Jan Leike, and Ryan Lowe. 2022. Training language models to follow instructions with human feedback. In _NeurIPS_ . 

- Hongyu Ren, Weihua Hu, and Jure Leskovec. 2020. Query2box: Reasoning over knowledge graphs in vector space using box embeddings. In _8th International Conference on Learning Representations, ICLR 2020, Addis Ababa, Ethiopia, April 26-30, 2020_ . OpenReview.net. 

- Hongyu Ren and Jure Leskovec. 2020. Beta embeddings for multi-hop logical reasoning in knowledge graphs. In _Advances in Neural Information Processing Systems 33: Annual Conference on Neural Information Processing Systems 2020, NeurIPS 2020, December 6-12, 2020, virtual_ . 

- John Schulman, Filip Wolski, Prafulla Dhariwal, Alec Radford, and Oleg Klimov. 2017. Proximal Policy Optimization Algorithms. ArXiv:1707.06347 [cs]. 

- Baoxu Shi and Tim Weninger. 2018. Open-world knowledge graph completion. In _Proceedings of the ThirtySecond AAAI Conference on Artificial Intelligence, (AAAI-18), the 30th innovative Applications of Artificial Intelligence (IAAI-18), and the 8th AAAI Symposium on Educational Advances in Artificial Intelligence (EAAI-18), New Orleans, Louisiana, USA, February 2-7, 2018_ , pages 1957–1964. AAAI Press. 

- Haitian Sun, Andrew O. Arnold, Tania Bedrax-Weiss, Fernando Pereira, and William W. Cohen. 2020. Faithful embeddings for knowledge base queries. In _Advances in Neural Information Processing Systems 33: Annual Conference on Neural Information Processing Systems 2020, NeurIPS 2020, December 6- 12, 2020, virtual_ . 

- Paul Thagard and Cameron Shelley. 1997. Abductive reasoning: Logic, visual thinking, and coherence. In _Logic and Scientific Methods: Volume One of the Tenth International Congress of Logic, Methodology and Philosophy of Science, Florence, August 1995_ , pages 413–427. Springer. 

- Kristina Toutanova and Danqi Chen. 2015. Observed versus latent features for knowledge base and text inference. In _Proceedings of the 3rd Workshop on Continuous Vector Space Models and their Compositionality, CVSC 2015, Beijing, China, July 26-31, 2015_ , pages 57–66. Association for Computational Linguistics. 

   - Zihao Wang, Yangqiu Song, Ginny Y. Wong, and Simon See. 2023b. Logical message passing networks with one-hop inference on atomic formulas. In _The Eleventh International Conference on Learning Representations, ICLR 2023, Kigali, Rwanda, May 1-5, 2023_ . OpenReview.net. 

   - Dong Yang, Peijun Qing, Yang Li, Haonan Lu, and Xiaodong Lin. 2022. Gammae: Gamma embeddings for logical queries on knowledge graphs. In _Proceedings of the 2022 Conference on Empirical Methods in Natural Language Processing, EMNLP 2022, Abu Dhabi, United Arab Emirates, December 7-11, 2022_ , pages 745–760. Association for Computational Linguistics. 

   - Hang Yin, Zihao Wang, Weizhi Fei, and Yangqiu Song. 2023. Efok-cqa: Towards knowledge graph complex query answering beyond set operation. _CoRR_ , abs/2307.13701. 

   - Jing Zhang, Bo Chen, Lingxi Zhang, Xirui Ke, and Haipeng Ding. 2021a. Neural, symbolic and neuralsymbolic reasoning on knowledge graphs. _AI Open_ , 2:14–35. 

   - Wen Zhang, Jiaoyan Chen, Juan Li, Zezhong Xu, Jeff Z. Pan, and Huajun Chen. 2022. Knowledge graph reasoning with logics and embeddings: Survey and perspective. _CoRR_ , abs/2202.07412. 

   - Zhanqiu Zhang, Jie Wang, Jiajun Chen, Shuiwang Ji, and Feng Wu. 2021b. Cone: Cone embeddings for multi-hop reasoning over knowledge graphs. In _Advances in Neural Information Processing Systems 34: Annual Conference on Neural Information Processing Systems 2021, NeurIPS 2021, December 6-14, 2021, virtual_ , pages 19172–19183. 

   - Daniel M. Ziegler, Nisan Stiennon, Jeffrey Wu, Tom B. Brown, Alec Radford, Dario Amodei, Paul Christiano, and Geoffrey Irving. 2020. FineTuning Language Models from Human Preferences. ArXiv:1909.08593 [cs, stat]. 

- Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Lukasz Kaiser, and Illia Polosukhin. 2017. Attention is all you need. In _Advances in Neural Information Processing Systems 30: Annual Conference on Neural Information Processing Systems 2017, December 4-9, 2017, Long Beach, CA, USA_ , pages 5998–6008. 

- Zihao Wang, Weizhi Fei, Hang Yin, Yangqiu Song, Ginny Y. Wong, and Simon See. 2023a. Wassersteinfisher-rao embedding: Logical query embeddings with local comparison and global transport. In _Findings of the Association for Computational Linguistics: ACL 2023, Toronto, Canada, July 9-14, 2023_ , pages 13679–13696. Association for Computational Linguistics. 

## **A Clarification on Abductive, Deductive, and Inductive Reasoning** 

Here we use simple syllogisms to explain the connections and differences between abductive reasoning and the other two types of reasoning, namely, deductive and inductive reasoning. In deductive reasoning, the inferred conclusion is necessarily true if the premises are true. Suppose we have a major premise _P_ 1: _All men are mortal_ , a minor premise _P_ 2: _Socrates is a man_ , then we can conclude _C_ that _Socrates is mortal_ . This can also be _P_ 1 _∧ P_ 2 

expressed as the inference _C_ . On the other hand, abductive reasoning aims to explain an observation and is non-necessary, i.e., the inferred hypothesis is not guaranteed to be true. We also start with a premise _P_ : _All cats like catching mice_ , and then we have some observation _O_ : _Katty like catching mice_ . The abduction gives a simple yet most probable hypothesis _H_ : _Katty is a cat_ , as an explanation. This can also be written as the _P ∧ O_ 

inference _H_ . Different than deductive reasoning, the observation _O_ should be entailed by the premise _P_ and the hypotheses _H_ , which can be expressed by the implication _P ∧ H_ = _⇒ O_ . The other type of non-necessary reasoning is inductive reasoning, where, in contrast to the appeal to explanatory considerations in abductive reasoning, there is an appeal to observed frequencies (Douven, 2021). For instance, premises _P_ 1: _Most Math students learn linear algebra in their first year_ and _P_ 2: Alice is a Math student infer _H_ : _Alice learned P_ 1 _∧ P_ 2 

_linear algebra in her first year_ , i.e., _H_ . Note that the inference rules in the last two examples are not strictly logical implications. 

It is worth mentioning that there might be different definitions or interpretations of these forms of reasoning. 

## **B Example of Observation-Hypothesis Pair** 

For example, observation _O_ can be a set of name entities like _{GrantHeslov_ , _JasonSegel_ , _RobertTowne_ , _RonaldBass, RashidaJones}_ . Given this observation, an abductive reasoner is required to give the logical hypothesis that best explains it. For the above example, the expected hypothesis _H_ in natural language is that they are _actors_ and _screenwriters_ , and they are also born in _Los Angeles_ . Mathematically, the hypothesis _H_ can 

be represented by a logical expression of the facts of the KG: _H_ ( _V_ ) = _Occupation_ ( _V, Actor_ ) _∧ Occupation_ ( _V, ScreenWriter_ ) _∧ BornIn_ ( _V, LosAngeles_ ). Although the logical expression here only contains logical conjunction AND ( _∧_ ), we consider more general first-order logical form as defined in Section 2. 

## **C Statistics of Queries Sampled for Datasets** 

Table 5 presents the numbers of queries sampled for each dataset in each stage. 

## **D Algorithm for Sampling Observation-Hypothesis Pairs** 

Algorithm 1 is designed for sampling complex hypotheses from a given knowledge graph. Given a knowledge graph _G_ and a hypothesis type _T_ , the algorithm starts with a random node _v_ and proceeds to recursively construct a hypothesis that has _v_ as one of its conclusions and adheres the type _T_ . During the recursion process, the algorithm examines the last operation in the current hypothesis. If the operation is a _projection_ , the algorithm randomly selects one of _v_ ’s in-edge ( _u, r, v_ ). Then, the algorithm calls the recursion on node _u_ and the sub-hypothesis type of _T_ again. If the operation is an _intersection_ , it applies recursion on the subhypotheses and the same node _v_ . If the operation is a _union_ , it applies recursion on one sub-hypothesis with node _v_ and on other sub-hypotheses with an arbitrary node, as union only requires one of the sub-hypotheses to have _v_ as an answer node. The recursion stops when the current node contains an entity. 

## **E Algorithms for Conversion between Queries and Actions** 

We here present the details of tokenizing the hypothesis graph (Algorithm 2), and formulating a graph according to the tokens, namely the process of de-tokenization (Algorithm 3). Inspired by the action-based semantic parsing algorithms, we view tokens as actions. It is worth noting that we employ the symbols _G, V, E_ for the hypothesis graph to differentiate it from the knowledge graph. 

## **F Details of using Smatch to evaluate structural differneces of queries** 

Smatch (Cai and Knight, 2013) is an evaluation metric for Abstract Meaning Representation 

|**Dataset**|**Training Samples**<br>Each Type<br>Total|**Validation Samples**<br>Each Type<br>Total|**Testing Samples**<br>Each Type<br>Total|
|---|---|---|---|
|||||
|FB15k-237<br>WN18RR<br>DBpedia50|496,126<br>6,449,638<br>148,132<br>1,925,716<br>55,028<br>715,364|62,015<br>806,195<br>18,516<br>240,708<br>6,878<br>89,414|62,015<br>806,195<br>18,516<br>240,708<br>6,878<br>89,414|



Table 5: The detailed information about the queries used for training, validation, and testing. 

**Algorithm 2** HypothesisToActions 

**Algorithm 1** Sampling Hypothesis According to Type 

**Input** Knowledge graph _G_ , hypothesis type _T_ **Output** Hypothesis sample 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/acl_2024_Advancing_Abductive_Reasoning_in_Knowledge_Graphs_through_Complex_Logical_Hypothesis_Generation_images/acl_2024_Advancing_Abductive_Reasoning_in_Knowledge_Graphs_through_Complex_Logical_Hypothesis_Generation.pdf-0013-05.png)


**if** _T.operation_ = _p_ **then** ( _h, r_ ) _←_ SAMPLE( _{_ ( _h, r_ ) _|_ ( _h, r, t_ ) _∈ E_ ( _G_ ) _}_ ) _T_ ˆ _←_ the only subtype in _T.SubTypes H ←_ GROUNDTYPE( _G, T, h_[ˆ] ) **return** ( _p, r, H_ ) **else if** _T.operation_ = _i_ **then** _SubHypotheses ←∅_ **for** _T_[ˆ] _∈ T.SubTypes_ **do** _H ←_ GROUNDTYPE( _G, T, t_[ˆ] ) 

_SubHypotheses._ PUSHBACK( _H_ ) **end for return** ( _i, SubHypotheses_ ) **else if** _T.operation_ = _u_ **then** _SubHypotheses ←∅_ **for** _T_[ˆ] _∈ T.SubTypes_ **do if** _T_[ˆ] is the first subtype **then** _H ←_ GROUNDTYPE( _G, T, t_[ˆ] ) 

**else** ˆ _t ←_ SAMPLE( _V_ ( _G_ )) _H ←_ GROUNDTYPE( _G, T,_[ˆ] _t_[ˆ] ) **end if** _SubHypotheses._ PUSHBACK( _H_ ) **end for return** ( _u, SubHypotheses_ ) **else if** _T.operation_ = _e_ **then return** ( _e, t_ ) **end if end function** _v ←_ SAMPLE( _V_ ( _G_ )) **return** GROUNDTYPE( _G, T, v_ ) 

**Input** Hypothesis plan graph _G_ **Output** Action sequence _A_ **function** DFS( _G, t, A_ ) **if** _t_ is an anchor node **then** _action ←_ entity associated with _t_ **else** _action ←_ operator associated with the first in-edge of _t_ **end if** _A._ PUSHBACK( _action_ ) **for all** in-edges to _t_ in _G_ ( _h, r, t_ ) **do** DFS( _G, h, A_ ) **end for end function** _root ←_ the root of _G A ←_ DFS( _G, root, ∅_ ) **return** _A_ 

(AMR) graphs. An AMR graph is a directed acyclic graph with two types of nodes: variable nodes and concept nodes, and three types of edges: 

- Instance edges, which connect a variable node to a concept node and are labeled literally “instance”. Every variable node must have exactly one instance edge, and vice versa. 

- Attribute edges, which connect a variable node to a concept node and are labeled with attribute names. 

- Relation edges, which connect a variable node to another variable node and are labeled with relation names. 

Given a predicted AMR graph _G_ p and the gold AMR graph _G_ g, the Smatch score of _G_ p with respect to _G_ g is denoted by `Smatch` ( _G_ p _, G_ g). `Smatch` ( _G_ p _, G_ g) is obtained by finding an approximately optimal mapping between the variable nodes of the two graphs and then matching the edges of the graphs. 

Our hypothesis graph is similar to the AMR graph, in: 

- The nodes are both categorized as fixed nodes and variable nodes 

- The edges can be categorized into two types: edges from a variable node to a fixed node and edges from a variable node to another variable node. And edges are labeled with names. 

However, they are different in that the AMR graph requires every variable node to have instance edges, while the hypothesis graph does not. 

The workaround for leveraging the Smatch score to measure the similarity between hypothesis graphs is creating an instance edge from every entity to some virtual node. Formally, given a hypothesis _H_ with hypothesis graph _G_ ( _H_ ), we create a new hypothesis graph _GAA_ ( _H_ ) to accommodate the AMR settings as follows: First, we initialize _GAA_ ( _H_ ) = _G_ ( _H_ ).. Then, create a new relation type _instance_ and add a virtual node _v[[′]]_ into _GA_ ( _H_ ). Finally, for every variable node _v ∈ G_ ( _H_ ), we add a relation _instance_ ( _v, v[[′]]_ ) into into _GAA_ ( _H_ ).. Then, given a predicted hypothesis _Hpp_ and a gold hypothesis _Hgg_ , the Smatch score is defined as 

## **Algorithm 3** ActionsToHypothesis 

**Input** Action sequence _A_ esis graphs is creating an instance edge from every **Output** Hypothesis plan graph _G_ entity to some virtual node. _S ←_ an empty stack pothesis _H_ with hypothesis graph _G_ ( _H_ Create an map _deg_ . _deg_ [ _i_ ] = _deg_ [ _u_ ] = 2 and a new hypothesis graph _GAA_ ( _H_ = 1 otherwise. the AMR settings as follows: First, we _V ←∅, E ←∅ GAA_ ( _H_ ) = _G_ ( _H_ ).. Then, create a new relation type **for** _a ∈ A_ **do** _instance_ and add a virtual node _v[[′]]_ into Create a new node _h_ , _V ← V ∪{h}_ Finally, for every variable node _v ∈ G_ ( _H_ **if** _S_ = _∅_ **then** add a relation _instance_ ( _v, v[[′]]_ ) into into _GAA_ ( _H_ ).. ( _t, operator, d_ ) _← S._ TOP given a predicted hypothesis _Hpp E ← E ∪{_ ( _h, operator, t_ ) _}_ esis _Hgg_ , the Smatch score is defined as **end if if** _a_ represents an anchor **then** _S_ ( _Hp, Hg_ ) = `Smatch` ( _GA_ ( _Hp_ ) _, GA_ ( _Hg_ )) Mark _h_ as an anchor with entity _a_ **while** _S_ = _∅_ **do** Through this conversion, a variable ( _t, operator, d_ ) _← S._ POP of _H_ g is mapped to a variable entity _vpp d ← d −_ 1 if and only if _instance_ ( _vg, vg, v, v[[′]]_ ) is matched **if** _d >_ 0 **then** _instance_ ( _vp, vp, v, v[[′]]_ ).. This modification does fect the overall algorithm for finding the optimal _S._ PUSHBACK(( _t, operator, d_ )) mapping between variable nodes and hence gives Break a valid and consistent similarity score. **end if** this adds an extra point for matching between in- **end while** stance edges, no matter how the variable nodes are **else** mapped. _S._ PUSHBACK(( _h, a, deg_ [ _a_ ])) **end if G Detailed Smatch Scores by Query end for Types** _G ←_ ( _V, E_ ) **return** _G_ Tables 6 and 7 show the detailed Smatch 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/acl_2024_Advancing_Abductive_Reasoning_in_Knowledge_Graphs_through_Complex_Logical_Hypothesis_Generation_images/acl_2024_Advancing_Abductive_Reasoning_in_Knowledge_Graphs_through_Complex_Logical_Hypothesis_Generation.pdf-0014-06.png)


Through this conversion, a variable entity _vg_ of _H_ g is mapped to a variable entity _vpp_ of _H_ p if and only if _instance_ ( _vg, vg, v, v[[′]]_ ) is matched with _instance_ ( _vp, vp, v, v[[′]]_ ).. This modification does not affect the overall algorithm for finding the optimal mapping between variable nodes and hence gives a valid and consistent similarity score. However, this adds an extra point for matching between instance edges, no matter how the variable nodes are mapped. 

Tables 6 and 7 show the detailed Smatch performance of various methods. 

## **H Detailed Jaccard Performance of the Brute-force Search** 

Table 8 shows the detailed Jaccard performance of the brute-force search. 

## **I Hyperparameters of the RL Process** 

The PPO hyperparameters are shown in Table 9. We warm-uped the learning rate from 0 _._ 1 of the peak to the peak value within the first 10% of total 

|**Dataset**||**Model**|**1p**<br>**2p**<br>**2i**<br>**3i**<br>**ip**<br>**pi**<br>**2u**<br>**up**|**2in**<br>**3in**<br>**pni**<br>**pin**<br>**inp**|**Ave.**|
|---|---|---|---|---|---|
|FB15k-237||Enc.-Dec.<br>0.342<br>0.506<br>0.595<br>0.602<br>0.570<br>0.598<br>0.850<br>0.571<br>0.652<br>0.641<br>0.650<br>0.655<br>0.599<br>0.602<br>RLF-KG (J)<br>0.721<br>0.643<br>0.562<br>0.480<br>0.364<br>0.475<br>0.769<br>0.431<br>0.543<br>0.499<br>0.513<br>0.518<br>0.370<br>0.530<br>RLF-KG (J+S)<br>0.591<br>0.583<br>0.577<br>0.531<br>0.447<br>0.520<br>0.820<br>0.505<br>0.602<br>0.563<br>0.571<br>0.595<br>0.484<br>0.568||||
|||||||
|||Dec.-Only<br>RLF-KG (J)<br>RLF-KG (J+S)|0.287<br>0.481<br>0.615<br>0.623<br>0.599<br>0.626<br>0.847<br>0.574<br>0.344<br>0.445<br>0.675<br>0.585<br>0.537<br>0.638<br>0.853<br>0.512<br>0.303<br>0.380<br>0.692<br>0.607<br>0.565<br>0.671<br>0.857<br>0.506|0.680<br>0.656<br>0.675<br>0.677<br>0.636<br>0.696<br>0.575<br>0.647<br>0.688<br>0.574<br>0.727<br>0.600<br>0.676<br>0.734<br>0.634|0.614<br>0.598<br>0.612|
|WN18RR||||||
|||Enc.-Dec.<br>RLF-KG (J)<br>RLF-KG (J+S)|0.375<br>0.452<br>0.591<br>0.555<br>0.437<br>0.585<br>0.835<br>0.685<br>0.455<br>0.468<br>0.563<br>0.562<br>0.361<br>0.530<br>0.810<br>0.646<br>0.443<br>0.457<br>0.565<br>0.572<br>0.366<br>0.545<br>0.814<br>0.661|0.586<br>0.516<br>0.561<br>0.549<br>0.530<br>0.560<br>0.530<br>0.536<br>0.539<br>0.465<br>0.541<br>0.553<br>0.532<br>0.546<br>0.491|0.558<br>0.540<br>0.545|
|||||||
|||Dec.-Only<br>RLF-KG (J)<br>RLF-KG (J+S)|0.320<br>0.443<br>0.582<br>0.551<br>0.486<br>0.597<br>0.809<br>0.696<br>0.400<br>0.438<br>0.566<br>0.491<br>0.403<br>0.519<br>0.839<br>0.667<br>0.375<br>0.447<br>0.584<br>0.499<br>0.432<br>0.545<br>0.825<br>0.679|0.594<br>0.526<br>0.575<br>0.574<br>0.577<br>0.547<br>0.450<br>0.497<br>0.466<br>0.450<br>0.584<br>0.477<br>0.543<br>0.522<br>0.507|0.564<br>0.518<br>0.540|
|DBpedia50||||||
|||Enc.-Dec.<br>RLF-KG (J)<br>RLF-KG (J+S)|0.345<br>0.396<br>0.570<br>0.548<br>0.344<br>0.576<br>0.712<br>0.544<br>0.461<br>0.424<br>0.634<br>0.584<br>0.361<br>0.575<br>0.809<br>0.579<br>0.419<br>0.410<br>0.638<br>0.555<br>0.373<br>0.586<br>0.785<br>0.579|0.474<br>0.422<br>0.477<br>0.488<br>0.428<br>0.584<br>0.497<br>0.544<br>0.533<br>0.454<br>0.560<br>0.459<br>0.536<br>0.542<br>0.474|0.486<br>0.541<br>0.532|
|||||||
|||Dec.-Only<br>RLF-KG (J)<br>RLF-KG (J+S)|0.378<br>0.408<br>0.559<br>0.526<br>0.397<br>0.568<br>0.812<br>0.626<br>0.405<br>0.411<br>0.558<br>0.496<br>0.376<br>0.507<br>0.825<br>0.621<br>0.398<br>0.415<br>0.567<br>0.497<br>0.383<br>0.533<br>0.827<br>0.630|0.480<br>0.414<br>0.489<br>0.494<br>0.474<br>0.477<br>0.397<br>0.468<br>0.444<br>0.406<br>0.510<br>0.420<br>0.484<br>0.457<br>0.430|0.510<br>0.492<br>0.504|



Table 6: The detailed Smatch performance of various methods. 

|**Dataset**|**1p**<br>**2p**<br>**2i**<br>**3i**<br>**ip**<br>**pi**<br>**2u**<br>**up**|**2in**<br>**3in**<br>**pni**<br>**pin**<br>**inp**|**Ave.**|
|---|---|---|---|
|||||
|FB15k-237<br>WN18RR<br>DBpedia|0.945 0.340 0.365 0.218 0.184 0.267 0.419 0.185<br>0.957 0.336 0.420 0.274 0.182 0.275 0.427 0.183<br>0.991 0.336 0.399 0.259 0.182 0.245 0.441 0.183|0.301 0.182 0.245 0.155 0.157<br>0.323 0.224 0.270 0.155 0.156<br>0.332 0.226 0.290 0.154 0.155|0.305<br>0.322<br>0.322|



Table 7: The detailed Smatch performance of the searching method. 

|Dataset|1p|2p|2i|3i|ip|pi|2u|up|2in|3in|pni|pin|inp|Ave.|
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
||||||||||||||||
|FB15k-237<br>WN18RR<br>DBpedia|0.980<br>0.997<br>0.997|0.563<br>0.622<br>0.705|0.639<br>0.784<br>0.517|0.563<br>0.776<br>0.517|0.732<br>0.955<br>0.982|0.633<br>0.666<br>0.461|0.744<br>0.753<br>0.783|0.585<br>0.605<br>0.754|0.659<br>0.783<br>0.722|0.479<br>0.757<br>0.658|0.607<br>0.762<br>0.782|0.464<br>0.540<br>0.544|0.603<br>0.630<br>0.700|0.635<br>0.741<br>0.702|



Table 8: The detailed Jaccard performance of the brute-force search. 

iterations, followed by a decay to 0 _._ 1 of the peak using a cosine schedule. 

## **J Algorithms for One-Hop Searching** 

We now introduce the Algorithm 4 used for searching the best one-hop hypothesis with the tail among all entities in the observation set to explain the observations. 

## **K Case Studies** 

Explore Table 10, 11 and 12 for concrete examples generated by various abductive reasoning methods, namely search, generative model with supervised training, and generative model with supervised training incorporating RLF-KG. 

## **Algorithm 4** One-Hop-Search 

**Input** Observation set _O_ 

**Output** Hypothesis _bestHypothesis_ 

_candidates ←{_ ( _h, r, t_ ) _∈R_ train _|t ∈ O} bestJaccard ←_ 0 

_bestHypothesis ←_ Null **for** ( _h, r, t_ ) _∈ candidates_ **do** 

- _H ←_ the one-hop hypothesis consisting of 

- edge ( _h, r, t_ ) 

   - _nowJaccard ←_ `Jaccard` ([[ _H_ ]] _G_ train _, A_ ) 

   - **if** _nowJaccard > bestJaccard_ **then** 

   - _bestJaccard ← nowJaccard bestHypothesis ← H_ 

   - **end if** 

**end for** 

**return** _bestHypothesis_ 

|**Hyperparam.**|**Enc.-Dec.**<br>FB15k-237<br>WN18RR<br>DBpedia50|**Dec.-Only**<br>FB15k-237<br>WN18RR<br>DBpedia50|
|---|---|---|
||||
|Learning rate<br>Batch size<br>Minibatch size<br>Horizon|2.4e-5<br>3.1e-5<br>1.8e-5<br>16384<br>16384<br>4096<br>512<br>512<br>64<br>4096<br>4096<br>4096|0.8e-5<br>0.8e-5<br>0.6e-5<br>3072<br>2048<br>2048<br>96<br>128<br>128<br>2048<br>2048<br>2048|



Table 9: PPO Hyperparamters. 

|||Interpretation|Companies operating in industries that intersect with Yahoo! but not with IBM.|
|---|---|---|---|
|||||
|||Hypothesis|The<br>observations<br>are<br>the<br>_V_?<br>such<br>that<br>_∃V_1_, inIndustry_(_V_1_, V_?) _∧_<br>_¬industryOf_(_IBM, V_1)_∧industryOf_(_Y ahoo_!_, V_1)|
|||||
|**Sample**||Observation|EMI,<br>CBS_Corporation,<br>Columbia,<br>GMA_Network,<br>Viacom,<br>Victor_Entertainment,<br>Yahoo!,<br>Sony_Music_Entertainment_(Japan)_Inc.,<br>Bandai,<br>Toho_Co.,_Ltd.,<br>Rank_Organisation,<br>The_New_York_Times_Company,<br>Gannett_Company,<br>Star_Cinema,<br>NBCUniversal,<br>TV5,<br>Pony_Canyon,<br>Avex_Trax,<br>The_Graham_Holdings_Company,<br>The_Walt_Disney_Company,<br>Televisa,<br>Metro-Goldwyn-Mayer,<br>Google,<br>Time_Warner,<br>Microsoft_Corporation,<br>Dell,<br>Munhwa_Broadcasting_Corporation,<br>News_Corporation|
|**Searching**||||
|||Interpretation|Which companies operate in media industry?|
|||||
|||Hypothesis|The observations are the_V_? such that_inIndustry_(_Media, V_?)|
|||||
|||Conclusion|Absent:<br>- Google,<br>- Microsoft_Corporation,<br>- Dell|
|||||
|||Jaccard|0.893|
|||||
|||Smatch|0.154|
|**Enc.-Dec.**||||
|||Interpretation|Companies operating in industries that intersect with<br>Yahoo! but not with Microsoft Corporation.|
|||||
|||Hypothesis|The<br>observations<br>are<br>the<br>_V_?<br>such<br>that<br>_∃V_1_, inIndustry_(_V_1_, V_?) _∧_<br>_¬industryOf_(_Microsoft_Corporation, V_1)_∧industryOf_(_Y ahoo_!_, V_1)|
|||Conclusion<br>Absent: Microsoft_Corporation||
|||Jaccard<br>0.964||
|||Smatch<br>0.909||
|**+ RLF-KG**||||
|||Interpretation|Companies operating in industries that intersect with<br>Yahoo! but not with Oracle Corporation.|
|||||
|||Hypothesis|The<br>observations<br>are<br>the<br>_V_?<br>such<br>that<br>_∃V_1_, inIndustry_(_V_1_, V_?) _∧_<br>_¬industryOf_(_Oracle_Corporation, V_1)_∧industryOf_(_Y ahoo_!_, V_1)|
|||Concl.<br>Same||
|||||
|||Jaccard|1.000|
|||||
|||Smatch|0.909|



Table 10: Example FB15k-237 Case study 1. 

|**Sample**|**Sample**|Interpretation|Locations that adjoin second-level divisions of the United<br>States of America that adjoin Washtenaw County.|
|---|---|---|---|
|||||
|||Hypothesis|The<br>observations<br>are<br>the<br>_V_?<br>such<br>that<br>_∃V_1_, adjoins_(_V_1_, V_?)<br>_∧_<br>_adjoins_(_Washtenaw_County, V_1)_∧secondLevelDivisions_(_USA, V_1)|
|||||
|||Observation|Jackson_County,<br>Macomb_County,<br>Wayne_County,<br>Ingham_County<br>Washtenaw_County,|
|**Searching**||||
|||Interpretation|Locations that adjoin Oakland County.|
|||||
|||Hypothesis|The observations are the_V_? such that_adjoins_(_Oakland_County, V_?)|
|||||
|||Conclusion|Absent:<br>- Jackson_County<br>- Ingham_County|
|||Jaccard<br>0.600||
|||Smatch<br>0.182||
|**Enc.-Dec.**||||
|||Interpretation|Second-level divisions of the United States of America<br>that adjoin locations that adjoin Oakland County.|
|||||
|||Hypothesis|The<br>observations<br>are<br>the<br>_V_?<br>such<br>that<br>_∃V_1_, secondLevelDivisions_(_USA, V_?)<br>_∧_<br>_adjoins_(_V_1_, V_?)<br>_∧_<br>+_adjoins_(_Oakland_County, V_1)|
|||||
|||Conclusion|Extra: Oakland_County<br>Absent: Wayne_County|
|||Jaccard<br>0.667||
|||Smatch<br>0.778||
|**+ RLF-KG**||||
|||Interpretation|Second-level divisions of the United States of America<br>that adjoin locations contained within Michigan.|
|||||
|||Hypothesis|The<br>observations<br>are<br>the<br>_V_?<br>such<br>that<br>_∃V_1_, secondLevelDivisions_(_USA, V_?)<br>_∧_<br>_adjoins_(_V_1_, V_?)<br>_∧_<br>_containedIn_(_Michigan, V_1)|
|||||
|||Conclusion|Extra:<br>- Oakland_County<br>- Genesee_County|
|||||
|||Jaccard|0.714|
|||Smatch<br>0.778||



Table 11: FB15k-237 Case study 2. 

|**Ground Truth**|**Ground Truth**|Interpretation|Works, except for “Here ’Tis,” that have subsequent works in the jazz genre.|
|---|---|---|---|
|||||
|||Hypothesis|The observations are the _V_? such that _∃V_1_, subsequentWork_(_V_1_, V_?) _∧_<br>_¬previousWork_(_Here_′Tis, V_1)_∧genre_(_Jazz, V_1)|
|||||
|||Observation|Deep,_Deep_Trouble,<br>Lee_Morgan_Sextet,<br>Good_Dog,_Happy_Man,<br>Paris_Nights\/New_York_Mornings,<br>I_Don’t_Want_to_Be_Your_Friend,<br>Take_the_Box<br>Interior_Music,|
|**Searching**||||
|||Interpretation|Works subsequent to “Closer” (Corinne Bailey Rae song).|
|||||
|||Hypothesis|The<br>observations<br>are<br>the<br>_V_?<br>such<br>that<br>_subsequentWork_(_Closer__(_Corinne_Bailey_Rae_song_)_, V_?)|
|||Conclusion<br>Only Paris_Nights\/New_York_Mornings||
|||||
|||Jaccard|0.143|
|||||
|||Smatch|0.154|
|**Enc.-Dec.**||||
|||Interpretation|Works, except for “Lee Morgan Sextet,” that have subsequent works<br>in the jazz genre.|
|||||
|||Hypothesis|The observations are the _V_? such that _∃V_1_, subsequentWork_(_V_1_, V_?) _∧_<br>_¬previousWork_(_Lee_Morgan_Sextet, V_1)_∧genre_(_Jazz, V_1)|
|||||
|||Conclusion|Extra: Here_’Tis<br>Absent: Lee_Morgan_Sextet|
|||Jaccard<br>0.750||
|||||
|||Smatch|0.909|
|**+ RLF-KG**||||
|||Interpretation|Works that have subsequent works in the jazz genre.|
|||||
|||Hypothesis|The observations are the _V_? such that _∃V_1_, subsequentWork_(_V_1_, V_?) _∧_<br>_genre_(_Jazz, V_1)|
|||||
|||Conclusion|Extra: Here_’Tis|
|||Jaccard<br>0.875||
|||||
|||Smatch|0.400|



Table 12: DBpedia50 Case study. 

