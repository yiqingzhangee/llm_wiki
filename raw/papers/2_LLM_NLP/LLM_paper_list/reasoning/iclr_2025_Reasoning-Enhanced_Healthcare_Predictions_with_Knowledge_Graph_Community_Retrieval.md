Under review as a conference paper 

# REASONING-ENHANCED HEALTHCARE PREDICTIONS WITH KNOWLEDGE GRAPH COMMUNITY RETRIEVAL 

**Pengcheng Jiang** _[∗]_ **Cao Xiao** _[†]_ **Minhao Jiang** _[†]_ **Parminder Bhatia** _[†]_ **Taha Kass-Hout** _[†]_ **Jimeng Sun** _[∗]_ **Jiawei Han** _[∗]_ 

> _∗_ UIUC _†_ GE HealthCare 

## ABSTRACT 

Large language models (LLMs) have demonstrated significant potential in clinical decision support. Yet LLMs still suffer from hallucinations and lack finegrained contextual medical knowledge, limiting their high-stake healthcare applications such as clinical diagnosis. Traditional retrieval-augmented generation (RAG) methods attempt to address these limitations but frequently retrieve sparse or irrelevant information, undermining prediction accuracy. We introduce KARE, a novel framework that integrates knowledge graph (KG) community-level retrieval with LLM reasoning to enhance healthcare predictions. KARE constructs a comprehensive multi-source KG by integrating biomedical databases, clinical literature, and LLM-generated insights, and organizes it using hierarchical graph community detection and summarization for precise and contextually relevant information retrieval. Our key innovations include: (1) a dense medical knowledge structuring approach enabling accurate retrieval of relevant information; (2) a dynamic knowledge retrieval mechanism that enriches patient contexts with focused, multi-faceted medical insights; and (3) a reasoning-enhanced prediction framework that leverages these enriched contexts to produce both accurate and interpretable clinical predictions. Extensive experiments demonstrate that KARE outperforms leading models by up to 10.8-15.0% on MIMIC-III and 12.6-12.7% on MIMIC-IV for mortality and readmission predictions. In addition to its impressive prediction accuracy, our framework leverages the reasoning capabilities of LLMs, enhancing the trustworthiness of clinical predictions. 

## 1 INTRODUCTION 

Large language models (LLMs) (Touvron et al., 2023a;b; OpenAI et al., 2024; Team et al., 2024) has revolutionized natural language processing, offering unprecedented capabilities in understanding and generating human-like text. In the healthcare domain, LLMs hold the potential to transform clinical decision-making by providing insights derived from vast amounts of medical data (Wornow et al., 2023; Yang et al., 2022). There has been many recent explorations on applying ML-based methods in healthcare domain (Choi et al., 2016; 2017; Shickel et al., 2018; Choi et al., 2018; Ma et al., 2020a; Gao et al., 2020; Zhang et al., 2021; Wu et al., 2023; Jiang et al., 2024a; Zhu et al., 2024a; Xu et al., 2024). However, deploying LLMs in clinical settings presents significant challenges, mainly because LLMs may produce hallucinations or incorrect information due to a lack of specialized medical knowledge. Traditional retrieval-augmented generation (RAG) techniques (Lewis et al., 2021), which aim to mitigate hallucinations by retrieving external knowledge, often fall short in healthcare applications. They tend to retrieve information that, while semantically similar in latent space, fails to provide meaningful clinical insights, leading to suboptimal outcomes for precise healthcare predictions (Shi et al., 2024; Magesh et al., 2024; Li et al., 2024). For instance, when dealing with the diagnosis of heart failure, a traditional retrieval and generation (RAG) model might retrieve data on several conditions that are semantically similar, such as “acute coronary syndrome” or “ischemic heart disease” due to their close proximity in latent space. However, these conditions, while related, do not capture the specific nuances of heart failure, such as the impact of left ventricular ejection fraction or specific biomarkers like NT-proBNP levels. 

Knowledge graphs (KGs) offer a promising solution by providing structured representations of medical knowledge, capturing complex relationships between clinical entities (Liu et al., 2019; Yasunaga 

1 

Under review as a conference paper 

et al., 2022; Zhang et al., 2022). Integrating KGs with LLMs can enhance the models’ reasoning capabilities and provide domain-specific knowledge essential for accurate healthcare predictions (Soman et al., 2024). However, previous studies have often lacked interpretability and failed to fully leverage the reasoning strengths of LLMs (Jiang et al., 2024a; Xu et al., 2024; Zhu et al., 2024a). 

Graph community retrieval has been a proven technique in various domains, such as social network analysis (Fortunato, 2010; Jin et al., 2021) and recommendation systems (Salha et al., 2019) for efficiently extracting relevant and contextual information from large-scale graphs. Recent work like GraphRAG (Edge et al., 2024) has demonstrated the superior performance of graph community retrieval compared to na¨ıve RAG in the query-focused summarization task. However, the application of graph retrieval for LLM-based healthcare prediction remains largely unexplored. 

In this paper, we introduce KARE ( **K** nowledge **A** ware **R** easoning- **E** nhanced HealthCare Prediction), a new framework that combines KG community-level retrieval (e.g., retrieving relevant subgraphs) with LLM reasoning to improve healthcare prediction. 

Our technical contributions can be summarized as follows: 

1. **Multi-Source Medical Knowledge Structuring and Indexing:** We develop a novel method to construct and index multi-source medical concept KGs by integrating concept-specific knowledge derived from relationships among different concepts in patients’ electronic health records (EHRs). We employ hierarchical graph community detection and summarization techniques to organize the KG into semantically meaningful communities, facilitating precise, fine-grained, and contextually relevant information retrieval. 

2. **Context Augmentation with Dynamic Knowledge Retrieval from KG:** We propose a context augmentation technique that can dynamically enrich patient data with knowledge from relevant KG communities tailored to the patient context. By retrieving pre-summarized communities, we enrich the input to the LLMs with focused, multi-faceted medical insights, addressing the limitations of traditional RAG methods. 

3. **Reasoning-Enhanced Clinical Prediction Framework:** We leverage the augmented patient context to enable LLMs to generate step-by-step reasoning chains, enhancing both interpretability and prediction accuracy in clinical tasks. 

To evaluate the KARE framework, we conducted experiments on in-hospital mortality and hospital readmission prediction tasks using the MIMIC-III and MIMIC-IV datasets(Johnson et al., 2016; 2020). KARE significantly outperforms the best baseline models. Specifically, KARE achieves improvements over best baselines up to 10.8%, 15.0%, 12.6%, and 12.7% on the MIMIC-III mortality, MIMIC-III readmission, MIMIC-IV mortality, and MIMIC-IV readmission prediction tasks, respectively. By attaining higher prediction accuracy and leveraging reasoning capabilities, KARE enhances the trustworthiness of clinical decision support systems. The reasoning process incorporates valuable evidence from relevant medical knowledge, facilitating more informed and explainable predictions that are needed in clinical decision making. 

## 2 RELATED WORKS 

**Clinical Predictive Models.** Electronic health record (EHR) data have become invaluable in the medical field, supporting predictive tasks aimed at improving patient care and clinical outcomes (Cai et al., 2016; Ashfaq et al., 2019; Bhoi et al., 2021). The development of deep learning models (Hochreiter & Schmidhuber, 1997; Chung et al., 2014; Vaswani et al., 2017) has enabled researchers to capture complex patterns within structured EHR data. Models such as RETAIN (Choi et al., 2016), GRAM (Choi et al., 2017), and others (Nguyen et al., 2016; Choi et al., 2018; Ma et al., 2020a;b; Gao et al., 2020; Zhang et al., 2021; Yang et al., 2023b) have shown promise in various predictive tasks. However, traditional predictive models are often inflexible, requiring specific labeled training data and struggling to generalize beyond their original scope. This limitation is particularly problematic in the dynamic healthcare environment. To address this, there is growing interest in using Large Language Models (LLMs) for clinical predictive tasks. LLMs offer greater adaptability and potential to interpret diverse medical information, including unstructured text and knowledge graphs enabling more robust and versatile clinical decision support systems. 

2 

Under review as a conference paper 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_Reasoning-Enhanced_Healthcare_Predictions_with_Knowledge_Graph_Community_Retrieval_images/iclr_2025_Reasoning-Enhanced_Healthcare_Predictions_with_Knowledge_Graph_Community_Retrieval.pdf-0003-01.png)


**----- Start of picture text -----**<br>
Step 1. Medical Concept Knowledge Graph Construction and Indexing Step 3. Reasoning-Enhanced Precise Healthcare Prediction<br>Knowledge Source 1.2 Semantic Clustering<br>(a) (b) (c) Ent./Rel. Set clustering New  3.1 Training Sample Generation Training Sample<br>Emb. Ent./Rel. Set Patient Context ReasoningChains INPUT<br>Biomedical KG Biomedical Corpus LLMs Run 1<br>1.3 Community Indexing<br>1.1 KG Extraction ⋮ ⋮ OUTPUT<br>… Ground Truth Run  𝑘<br>0/1<br>0/1<br>Community Summarization<br>Medical Concept Knowledge Graph …<br>3.2 Multitask-Based Fine-Tuning & Prediction<br>Step 2. Patient Context Construction and Augmentation<br>Reasoning  Local  Reasoning<br>Instruction LLM Chain<br>(Similar Cases) ⊕ Patient Context<br>Base Context Augmented Context<br>Patient’s EHR Patient KG Top Community Summaries Selection Metrics (1) Node Hits 0/1<br>(2) Coherence Label Prediction  Label<br>(3) Recency Instruction<br>(4) Theme-Relev.<br>**----- End of picture text -----**<br>


Figure 1: **A conceptual illustration of our KARE framework. Step 1** constructs a comprehensive medical concept knowledge graph by integrating information from multiple sources, organizing it into a hierarchical community structure. This structure allows for the generation of community summaries that facilitate precise knowledge retrieval. **Step 2** dynamically augments the patient’s EHR context with relevant summaries from the knowledge graph, offering the LLM focused and relevant medical insights. **Step 3** generates training samples by employing an expert LLM to create reasoning chains based on the augmented patient context and ground truth labels. It then fine-tunes a local LLM using a multitask learning approach to produce interpretable reasoning chains and accurate predictions. By combining knowledge retrieval with LLM-driven reasoning, KARE significantly enhances the accuracy and reliability of clinical predictions. 

**LLMs for Healthcare Predictions.** LLMs have revolutionized healthcare applications due to their advanced language understanding and generation capabilities (Xu et al., 2023; Kim et al., 2024; Bedi et al., 2024; Denecke et al., 2024). Recent works like GraphCare (Jiang et al., 2024a), RAM-EHR (Xu et al., 2024), and EMERGE (Zhu et al., 2024a) have attempted to inject knowledge from LLMs into patient representations, but they still lack interpretability and do not fully exploit the reasoning capabilities of LLMs. On the other hand, when directly applied to domain-specific tasks like EHR prediction, LLMs can produce significant errors and hallucinations due to the lack of integration of specialized domain knowledge (Zhu et al., 2024b; Xu et al., 2024; Cui et al., 2024; Shi et al., 2024). Therefore, our work integrates KG community indexing and dynamic graph retrieval, compared to traditional RAG (Lewis et al., 2021; Niu et al., 2024) and KGs, to construct and query finegrained, highly relevant knowledge, enhancing patient context. Furthermore, existing LLM-based methods often fail to fully harness the inherent reasoning capabilities of LLMs. Recent efforts (Cui et al., 2024; Shi et al., 2024) have explored agentic frameworks with foundational models for EHR prediction but rely on simplistic prompting that does not deeply engage with underlying EHR data patterns, resulting in suboptimal performance. Our approach distinguishes itself by fine-tuning a specialized, smaller LLM that incorporates reasoning abilities distilled from larger models. 

## 3 KARE: **K** NOWLEDGE **A** WARE **R** EASONING- **E** NHANCED FRAMEWORK 

Our KARE framework aims to improve healthcare predictions by combining relevant medical knowledge along with reasoning capabilities with LLMs. The following steps achieve this: (1) medical concept knowledge graph construction and indexing, (2) patient context construction and augmentation, and (3) reasoning-enhanced precise healthcare prediction. We provide a conceptual illustration of our KARE framework in Figure 1 and elaborate on the three steps in this section. 

## 3.1 STEP 1: MEDICAL CONCEPT KNOWLEDGE GRAPH CONSTRUCTION AND INDEXING 

_**Objective of Step 1**_ is to create a medical knowledge base that is specifically tailored to electronic health record (EHR) data. Unlike most existing medical knowledge graphs, which are static and not 

3 

Under review as a conference paper 

connected to the EHR data, KARE dynamically generates a high-quality knowledge base that can be used for retrieving and predicting information in later stages. 

## 3.1.1 MEDICAL CONCEPT-SPECIFIC KNOWLEDGE GRAPH EXTRACTION 

For each medical concept _ci_ in the EHR coding system, we extract a concept-specific knowledge graph _Gci_ = ( _Vci, Eci_ ) globally tailored to the EHR datasets from three sources: 

(a) **Biomedical KG** (e.g., UMLS (Bodenreider, 2004)): For each medical concept _ci_ in EHR data, we extract a subgraph _G_[KG] _ci_[by][first][iterating][through][the][patient][EHR][dataset][to][collect][the][top] _[X]_ co-existing concepts appeared in each patient’s data, forming the set of related concepts _Rci_ . We then find the shortest path _pij_ in the KG for each pair ( _ci, cj_ ) in _Rci_ , with a specified maximum path length. _G_[KG] _ci_ = ( _Vc_[KG] _i[, E] c_[KG] _i_[)][is][constructed][by][combining][all][these][shortest][paths,][where] _[V] c_[KG] _i_ and _Ec_[KG] _i_[are the union of nodes and edges in all] _[ p][ij]_[, respectively.][See more details in Appendix][ B.1] . 

(b) **Biomedical Corpus** (e.g., PubMed (Canese & Weis, 2013)): We iterate through the EHR dataset and, for each visit of the patients, and collect all the involved medical concepts. We then retrieve the top _n_ documents from the corpus based on these medical concepts. For each retrieved document, we perform entity extraction and relation extraction to extract KG triples. The extracted triples are then added to the KG of the medical concepts mentioned in the document. By doing so, _G_[BC] _ci_[is built] for each concept _ci_ . We showcase more details in Appendix B.2. 

(c) **LLMs** : We iterate through the EHR dataset and prompt the LLM to identify the relationships among the concepts that are helpful to the clinical predictions, where we allow the LLM to add intermediate relationships within two concepts. The process is detailed in Appendix B.3. 

The final concept-specific KG _Gci_ is the union of the subgraphs from each source: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_Reasoning-Enhanced_Healthcare_Predictions_with_Knowledge_Graph_Community_Retrieval_images/iclr_2025_Reasoning-Enhanced_Healthcare_Predictions_with_Knowledge_Graph_Community_Retrieval.pdf-0004-08.png)


Finally, we integrate all concept-specific KGs for the medical concepts in our EHR coding system. The resulting knowledge graph _G[′]_ = ( _V[′] , R[′] , E[′]_ ) is defined as _G[′]_ =[�] _ci∈_ **C** _[G][c] i_[where] **[ C]**[ is the set] of all medical concepts in the specified EHR coding system. 

_Note_ : Different from the KG construction method introduced by GraphCare (Jiang et al., 2024a), which retrieves sparse and random relationships from LLMs and biomedical KGs, our approach utilizes the EHR dataset to anchor the relevant relationships and interactions among the medical concepts present in patient data. This targeted focus allows us to construct a more relevant and context-tailored KG for clinical predictions. 

## 3.1.2 SEMANTIC CLUSTERING 

Semantic clustering in our KG addresses the challenge of differently named entities and relations from various sources that may refer to the same concept. We employ agglomerative clustering (M¨ullner, 2011) with an automatically determined optimal threshold. First, we generate text embeddings **e** _i_ = TextEmbed( _vi_ ) for each entity _vi ∈ V[′]_ and **e** _j_ = TextEmbed( _rj_ ) for each relation _rj ∈ R[′]_ using an LLM. To determine the optimal clustering thresholds _θe_ and _θr_ for entities and relations, we refer to the silhouette score (Shahapure & Nicholas, 2020; Jiang et al., 2024b), which considers both intra-cluster similarity and inter-cluster dissimilarity. We sample a subset of entities and relations, perform agglomerative clustering with varying distance thresholds, and select those yield the highest scores. We then cluster all entities and relations using their respective optimal thresholds. Each cluster is represented by the element closest to the cluster center, determined by the average embedding of all elements within the cluster. We create mappings _ϕe_ : _V[′] → V_ and _ϕr_ : _R[′] → R_ between the original entities/relations and their cluster representatives. Each triple ( _h[′] , r[′] , t[′]_ ) in the original KG is mapped to its corresponding cluster representatives ( _h, r, t_ ) = ( _ϕe_ ( _h[′]_ ) _, ϕr_ ( _r[′]_ ) _, ϕe_ ( _t[′]_ )), resulting in a refined knowledge graph _G_ = ( _V, R, E_ ). 

## 3.1.3 HIERARCHICAL KG COMMUNITY DETECTION AND INDEXING 

We organize the refined knowledge graph (KG) into a hierarchical structure of communities using the Leiden algorithm (Traag et al., 2019). This is done at multiple levels of granularity, from coarse to fine. Different from GraphRAG (Edge et al., 2024), we run the algorithm _multiple times_ with 

4 

Under review as a conference paper 

different randomness parameters to explore diverse community structures, and generate _multifaceted summaries_ for each community, providing a more comprehensive understanding of the knowledge contained withtin the KG. 

To keep computation manageable, we limit the maximum number of triples per community ( _Zc_ ) and the maximum number of triples per initial summary ( _Zs_ ). 

For each community, we generate two types of summaries using an LLM (prompts used: Figure 9): 

- _General summary_ : A concise summarization of the medical concepts and relationships in the community without focusing on any specific theme. 

- _Theme-specific summary_ : A summary that highlights how the knowledge in the community relates to a specific theme (e.g., mortality prediction), if relevant. Figure 11 shows an example. 

The summarization process depends on the community size: 

- For small communities (size _≤ Zs_ ), we directly summarize all triples. 

- For large communities ( _Zs <_ size _≤ Zc_ ), we shuffle and split the triples into subsets, summarize each subset, and then iteratively aggregate the summaries until we get a single comprehensive summary (prompt shown in Figure 10). 

- For extremely large communities (size _> Zc_ ), we do not generate summaries due to the limit of the LLM context window. 

As we move up the hierarchy from fine to coarse levels, triples from small communities get merged into larger ones, which are then summarized using the same process. 

The result is a hierarchical structure of communities at different granularities, each with themespecific summaries. Running the Leiden algorithm multiple times with different randomness parameter gives us diverse communities, allowing entities to contribute to multiple summaries. This rich, multi-level representation of the KG is the foundation for later steps. 

## 3.2 STEP 2: PATIENT CONTEXT CONSTRUCTION AND AUGMENTATION 

_**Objective of Step 2:**_ This step constructs patient’s EHR context with the highly relevant and finegrained medical knowledge attached. 

**Base Context Construction.** For a patient _p_ , we construct a base context _Bp_ with their EHR data with a standardized template. This context focuses on (1) task description, (2) the patient’s conditions, procedures, and medications across different visits, and (3) similar patients to the target patient. For (3), two most similar patients are retrieved from the reference set (i.e., training data) based on the EHR similarity where one has the same label as patient _p_ and the other has a different label (Cui et al., 2024). Figure 8 shows an example of the base context and the template used. 

**Context Augmentation.** To enrich the patient’s base context with relevant information from the knowledge graph, we first construct a patient-specific knowledge graph _Gp_ by aggregating the concept-specific graphs _Gci_ (defined in Eq. 1) for all medical concepts _ci_ in the patient’s EHR, using the mappings _ϕe_ and _ϕr_ from §3.1.2: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_Reasoning-Enhanced_Healthcare_Predictions_with_Knowledge_Graph_Community_Retrieval_images/iclr_2025_Reasoning-Enhanced_Healthcare_Predictions_with_Knowledge_Graph_Community_Retrieval.pdf-0005-16.png)


From _Gp_ , we derive two sets of nodes: _Vp_[direct] , representing medical concepts that directly appear in the patient’s EHR, and _Vp_[indirect] , containing the remaining nodes in _Gp_ . 

We then introduce a combined relevance score for each community _Ck_ to select the most relevant summaries for context augmentation: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_Reasoning-Enhanced_Healthcare_Predictions_with_Knowledge_Graph_Community_Retrieval_images/iclr_2025_Reasoning-Enhanced_Healthcare_Predictions_with_Knowledge_Graph_Community_Retrieval.pdf-0005-19.png)


In Eq. 3, _H_ ( _Ck, Vp_[direct] ) and _H_ ( _Ck, Vp_[indirect] ) calculate the normalized counts of direct and indirect node hits by comparing the nodes in community _Ck_ with the corresponding sets of direct and indirect 

5 

Under review as a conference paper 

nodes. The parameter _α ∈_ [0 _,_ 1) weights the importance of indirect hits relative to direct hits. The decay function Decay( _Ck, Vp_[direct] ) reduces the contribution of previously hit nodes in community _Ck_ by a factor _β[H]_[(] _[v]_[)] , where _β ∈_ (0 _,_ 1] is a decay constant and _H_ ( _v_ ) is the hit count of node _v_ in previous selections, considering only the direct nodes _Vp_[direct] . Additional factors are defined as: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_Reasoning-Enhanced_Healthcare_Predictions_with_Knowledge_Graph_Community_Retrieval_images/iclr_2025_Reasoning-Enhanced_Healthcare_Predictions_with_Knowledge_Graph_Community_Retrieval.pdf-0006-02.png)


Here, _e_ ( _·_ ) denotes a text embedding function, cos( _·, ·_ ) is the cosine similarity between embeddings, visit( _v_ ) returns the visit index of node _v_ , and _λ_ 1 _, λ_ 2 _, λ_ 3 _∈_ [0 _,_ 1] control the weights of the metrics. The set _Tτ_ contains representative terms for the theme _τ_ (e.g., _{end-stage, life-threatening, ...}_ for mortality prediction), same as those used for _attention initialization_ in (Jiang et al., 2024a). 

The proposed metrics in Eq. 3 serve different purposes: node hits _H_ ensure specificity to the patient’s conditions, decay factor promotes diversity, coherence aligns the selected summaries with the patient’s overall context, recency prioritizes more recent information, and theme relevance maintains task-oriented selection. In addition, we propose a Dynamic Graph Retrieval and Augmentation (DGRA) method to iteratively select the most relevant summaries to augment the patient’s context. At each iteration, it performs as: 

**Algorithm 1** Dynamic Graph Retrieval and Augmentation (1) Compute the relevance scores forallall candidate communities _Ckk ∈C_ **Input:** Settext _B_ of _p_ communities, desired number of summaries _C_ , patient graph _G Np_ , base conusing Eq. 3. 3.. **Output:** Augmented patient context _Ap_ (2) Identify the community _C_ best with with Initialize _Sp ←∅_ the highest relevance score and add Initialize hit counts _H_ ( _v_ ) _←_ 0 for each node _v ∈ Vp_[direct] its summary _SCC_ best to the set of se- **while** _|Sp| < N_ **do** lected summaries _Spp_ . Compute Relevance( _Ck_ ) for all _Ck ∈C_ using Eq. 3 (3) Increment the hit count _H_ ( _v_ ) for Select _C_ best _←_ arg max _Ck∈C_ Relevance( _Ck_ ) eachpact the decay in future relevance cal-nodepact the decay in future relevance cal-node _v_ in _VCC_ best,, which will imAddFor each _SC_ best _v_ to _∈ SVpC_ :best _Sp∩←Vp_[direct] _Sp ∪_ , _HSC_ ( _v_ best) _← H_ ( _v_ ) + 1 culations. Remove _C_ best from _C_ : _C ←C \ C_ best **end** (4)communitiesconsidered in subsequent iterations.Remove _CC_ best, ensuringfrom theit candidateis not reAugment patient context: **return** _Ap Ap_ = _Bp ⊕ Sp_ 

(1) Compute the relevance scores forallall candidate communities _Ckk ∈C_ using Eq. 3. 3.. 

(2) Identify the community _C_ best with with the highest relevance score and add its summary _SCC_ best to the set of selected summaries _Spp_ . 

(3) Increment the hit count _H_ ( _v_ ) for node _v_ in _VCC_ best,, which will im- eachpact the decay in future relevance cal-nodepact the decay in future relevance cal-node culations. 

The process continues until _N_ summaries have been selected. The final augmented patient context _Ap_ is obtained by concatenating the base context _Bp_ with the selected summaries. 

By dynamically updating the node hits and recalculating relevance scores at each iteration, we prioritize communities that contribute new and valuable information. This ensures that the augmented context includes the most relevant and diverse information from the KG, tailored to the patient’s specific conditions and the prediction task. 

## 3.3 STEP 3: REASONING-ENHANCED PRECISE HEALTHCARE PREDICTION 

_**Objective of Step 3:**_ This step trains an LLM capable of predicting healthcare outcome while generating a reasoning process, using the augmented patient context constructed in the previous step. 

## 3.3.1 TRAINING SAMPLE GENERATION 

Drawing from the structured rationale distillation strategy introduced in TriSum (Jiang et al., 2024b), we employ an LLM to generate reasoning chains in a unified format for each patient _p_ and task _τ_ . This process involves entering (1) the task description _Dτ_ (e.g., Figure 12), (2) the augmented patient context _Ap_ , and (3) the corresponding ground truth label _yp,τ[∗]_[into][the][LLM.][The][specific][prompt] utilized for the reasoning chain (training sample) generation is showcased in Figure 13 in Appendix. 

6 

Under review as a conference paper 

Table 1: Statistics of pre-processed EHR datasets. ”#”: ”the number of”, ”/ patient”: ”per patient”. 

||**MIMIC-III-Mort.**<br>Train<br>Valid<br>Test|**MIMIC-III-Read.**<br>Train<br>Valid<br>Test|**MIMIC-IV-Mort.**<br>Train<br>Valid<br>Test|**MIMIC-IV-Read.**<br>Train<br>Valid<br>Test|
|---|---|---|---|---|
|# Patients (Samples)<br># Visits / Patient<br># Conditions / Patient<br># Procedures / Patient<br># Medications / Patient|7730<br>991<br>996<br>1.56<br>1.60<br>1.61<br>23.27<br>23.92<br>25.89<br>6.22<br>6.56<br>7.17<br>54.79<br>55.77<br>63.73|7730<br>991<br>996<br>1.56<br>1.60<br>1.61<br>23.27<br>23.92<br>25.89<br>6.22<br>6.56<br>7.17<br>54.79<br>55.77<br>63.73|8018<br>996<br>986<br>1.26<br>1.30<br>1.21<br>14.34<br>15.30<br>13.59<br>2.96<br>3.08<br>2.84<br>30.66<br>32.86<br>28.40|8029<br>958<br>1013<br>1.26<br>1.28<br>1.25<br>13.62<br>14.21<br>13.21<br>2.89<br>2.96<br>2.81<br>28.74<br>30.61<br>27.59|



The LLM generates _K_ reasoning chains _ρp,τ,k_ along with confidence levels. We select the reasoning chain with the highest confidence, ensuring that only the most reliable explanations are used. The final set of training data for each patient-task pair is then _{_ ( _Dτ , Ap, ρ_[best] _p,τ[, y] p,τ[∗]_[)] _[}]_[,][where] _[ ρ]_[best] _p,τ_[is the] reasoning chain with the highest confidence level. 

## 3.3.2 MULTITASK-BASED FINE-TUNING AND PREDICTION 

We fine-tune a relatively small local LLM (e.g., a 7B-parameter model) to perform both reasoning chain generation and label prediction for each patient _p_ and healthcare prediction task _τ_ (such as mortality or readmission prediction). The model is trained using inputs that consist of the task description _Dτ_ and the augmented patient context _Ap_ , with an prepended instruction indicating whether to generate a reasoning chain or predict the label. These inputs and outputs are formatted according to the templates shown in Figure 14 in the Appendix. 

During fine-tuning, when instructed to generate a reasoning chain (with the prefix [Reasoning]), the model aligns its output with the reasoning chain _ρ_[best] _p,τ_[obtained][from][the][previous][step.][When] instructed to predict the label (with the prefix [Label Prediction]), it aligns its output with the ground truth label _yp,τ[∗]_[.][We minimize the cross entropy loss across both tasks, encouraging the] development of shared representations that enhance performance in both reasoning and prediction. 

In the prediction phase, given a new patient _p_ new and task _τ_ , we provide _Ap_ new , _τ_ , and the appropriate instruction to the fine-tuned model. Based on the instruction, the model can either generate the reasoning chain _ρp_ new _,τ_ or predict the label _yp_ new _,τ_ . This flexible approach allows us to obtain detailed reasoning when necessary or perform efficient label prediction, leveraging the multitask training to effectively handle both tasks during inference. 

## 4 EXPERIMENTS 

## 4.1 EXPERIMENTAL SETTING 

**Tasks.** In this work, we focus on the following EHR-based prediction tasks: 

- _Mortality Prediction._ This task estimates mortality outcome for next visit, defined as _f_ : ( _x_ 1 _, x_ 2 _, . . . , xt−_ 1) _→ y_ [ _xt_ ], where _y_ [ _xt_ ] _∈{_ 0 _,_ 1 _}_ is patient’s survival status during visit _xt_ . 

- _Readmission Prediction._ This task predicts if patient will be readmitted within _σ_ days, defined as _f_ : ( _x_ 1 _, x_ 2 _, . . . , xt−_ 1) _→ y_ [ _φ_ ( _xt_ ) _− φ_ ( _xt−_ 1)], where _y ∈{_ 0 _,_ 1 _}_ , _φ_ ( _xt_ ) is timestamp of visit _xt_ , and _y_ [ _φ_ ( _xt_ ) _− φ_ ( _xt−_ 1)] = 1 if _φ_ ( _xt_ ) _− φ_ ( _xt−_ 1) _≤ σ_ , else 0. _σ_ is set to 15 in this study. 

**Datasets.** We utilize the publicly available MIMIC-III (Johnson et al., 2016) (v1.4) and MIMIC-IV (Johnson et al., 2020) (v2.0) EHR datasets, and use PyHealth (Yang et al., 2023a) for preprocessing. MIMIC-III is processed (full set) using the same approach as GraphCare (Jiang et al., 2024a). For MIMIC-IV mortality prediction, we retain 2,152 patients with a label of 1 (mortality), excluding 54 patients with more than 10 visits. We then randomly (seed=42) sample unique patients with a label of 0, each having no more than 10 visits, until reaching a sample size of 10,000. For MIMIC-IV readmission prediction, we randomly (seed=42) select 5,000 unique patients with a label of 1 (will be readmitted) and 5,000 with a label of 0. Both datasets are split into training, validation, and test sets in a 0.8/0.1/0.1 ratio by patient, ensuring that all samples from the same patient are confined to a single subset, preventing data leakage. We use Clinical Classifications Software (CCS) for condition/procedure mappings and the Anatomical Therapeutic Chemical classification system at the third level (ATC3) for medication mapping, with the resulting statistics presented in Table 1. 

**Evaluation Metrics.** We employ the following four key metrics: (1) **Accuracy** , which measures overall correct predictions across both outcomes; (2) **Macro-F1** , which provides a balanced mea- 

7 

Under review as a conference paper 

Table 2: Comparative analysis of mortality and readmission predictions using MIMIC-III and MIMIC-IV datasets. ”pos” indicates the proportion of positive samples (label = 1) in the test set. Metrics like Macro F1 and Sensitivity are emphasized by an asterisk ( _[∗]_ ) due to their importance for handling imbalanced datasets. Results are averaged by multiple runs: 30 for ML-based, 10 for LM+ML-based, and 3 for LLM-based methods with different random seeds. We **highlight** the highest value for each metric. 

|**Type**|**Models**|**MIMIC**<br>**Mortality Prediction**(pos= 5_._42%)<br>**Accuracy**<br>**Macro F1**_∗_<br>**Sensitivity**_∗_<br>**Specifcity**|**-III**<br>**Readmission Prediction**(pos= 54_._82%)<br>**Accuracy**<br>**Macro F1**<br>**Sensitivity**<br>**Specifcity**|
|---|---|---|---|
|ML|GRU (Chung et al.,2014)<br>Transformer (Vaswani et al.,2017)<br>RETAIN (Choi et al.,2016)<br>GRAM (Choi et al.,2017)<br>Deepr (Nguyen et al.,2016)<br>TCN (Bai et al.,2018)<br>ConCare (Ma et al.,2020b)<br>AdaCare (Ma et al.,2020a)<br>GRASP (Zhang et al.,2021)<br>StageNet (Gao et al.,2020)<br>KerPrint (Yang et al.,2023b)|92.7<br>50.7<br>3.7<br>97.8<br>92.7<br>51.9<br>5.6<br>97.6<br>92.4<br>50.6<br>3.7<br>97.6<br>92.4<br>50.2<br>5.2<br>95.2<br>91.9<br>51.0<br>3.7<br>98.2<br>91.6<br>53.2<br>9.3<br>96.4<br>94.6<br>48.6<br>0.0<br>**100.0**<br>90.6<br>54.1<br>9.1<br>97.6<br>93.7<br>49.9<br>1.9<br>98.9<br>90.5<br>50.5<br>5.6<br>95.4<br>92.4<br>52.2<br>9.8<br>94.7|62.2<br>61.5<br>68.9<br>54.0<br>58.8<br>58.2<br>65.0<br>51.3<br>59.1<br>56.9<br>74.9<br>40.0<br>61.8<br>60.4<br>74.9<br>46.4<br>62.6<br>62.1<br>66.7<br>57.6<br>63.4<br>62.7<br>70.7<br>54.7<br>59.2<br>59.0<br>61.5<br>56.4<br>61.6<br>60.5<br>70.8<br>50.3<br>61.3<br>59.5<br>74.9<br>44.8<br>60.5<br>60.0<br>65.1<br>54.9<br>63.5<br>62.1<br>68.0<br>56.1|
|LM+ML|GraphCare (Jiang et al.,2024a)<br>RAM-EHR (Xu et al.,2024)<br>EMERGE (Zhu et al.,2024a)|94.9<br>58.3<br>17.2<br>97.1<br>94.4<br>59.6<br>14.8<br>98.9<br>94.1<br>57.7<br>13.2<br>98.4|65.4<br>64.1<br>70.3<br>57.8<br>64.8<br>63.5<br>74.7<br>52.4<br>63.7<br>62.0<br>68.0<br>55.9|
|LLM|Zero-shot (LLM: Claude 3.5 Sonnet)<br>w/ EHR context only<br>w/ Classic RAG[a]<br>w/ KARE-augmented context[b]|89.5<br>50.4<br>6.4<br>94.4<br>89.9<br>51.2<br>10.2<br>92.8<br>92.3<br>54.6<br>14.2<br>94.6|54.3<br>35.4<br>98.9<br>0.2<br>53.2<br>34.6<br>91.2<br>1.4<br>56.3<br>43.8<br>93.9<br>10.6<br>52.7<br>42.2<br>87.0<br>11.1<br>53.6<br>44.7<br>84.0<br>15.7<br>55.2<br>46.1<br>78.2<br>20.1<br>57.1<br>49.3<br>75.5<br>27.2<br>**73.9**<br>**73.7**<br>**76.7**<br>**70.7**|
||Few-Shot (LLM: Claude 3.5 Sonnet)<br>w/ exemplar only (N=2)[c]<br>w/ exemplar only (N=4)<br>w/ EHR-CoAgent[d] (Cui et al.,2024)<br>w/ KARE-augmented context|88.7<br>49.5<br>5.6<br>93.4<br>88.0<br>49.2<br>5.6<br>92.7<br>87.4<br>51.7<br>13.0<br>91.8<br>91.5<br>53.5<br>13.7<br>94.0||
||**KARE**(ours)|**95.3**<br>**64.6**<br>**24.7**<br>98.3||
|||||
|**Type**|**Models**|**MIMI**<br>**Mortality Prediction**(pos= 19_._16%)<br>**Accuracy**<br>**Macro F1**_∗_<br>**Sensitivity**_∗_<br>**Specifcity**|**C-IV**<br>**Readmission Prediction**(pos= 46_._50%)<br>**Accuracy**<br>**Macro F1**<br>**Sensitivity**<br>**Specifcity**|
|ML|GRU (Chung et al.,2014)<br>Transformer (Vaswani et al.,2017)<br>RETAIN (Choi et al.,2016)<br>GRAM (Choi et al.,2017)<br>Deepr (Nguyen et al.,2016)<br>TCN (Bai et al.,2018)<br>ConCare (Ma et al.,2020b)<br>AdaCare (Ma et al.,2020a)<br>GRASP (Zhang et al.,2021)<br>StageNet (Gao et al.,2020)<br>KerPrint (Yang et al.,2023b)|88.7<br>76.4<br>42.9<br>99.6<br>83.7<br>71.3<br>47.1<br>92.3<br>84.8<br>73.8<br>52.4<br>92.4<br>86.4<br>74.4<br>50.6<br>93.9<br>89.2<br>78.9<br>50.8<br>98.2<br>89.9<br>79.2<br>47.6<br>**99.9**<br>89.8<br>78.9<br>47.1<br>**99.9**<br>88.7<br>78.2<br>50.3<br>97.8<br>89.9<br>79.1<br>47.6<br>99.8<br>88.1<br>77.8<br>51.9<br>96.7<br>88.7<br>79.8<br>53.1<br>98.0|62.4<br>62.2<br>68.3<br>56.2<br>61.3<br>61.3<br>63.0<br>59.5<br>62.8<br>62.6<br>68.7<br>56.6<br>62.5<br>62.5<br>67.4<br>57.8<br>59.2<br>59.2<br>57.0<br>61.5<br>63.6<br>63.5<br>72.2<br>56.1<br>59.8<br>59.8<br>63.5<br>56.6<br>62.9<br>62.9<br>58.4<br>**67.7**<br>59.7<br>59.6<br>53.1<br>66.7<br>62.8<br>62.7<br>62.6<br>62.9<br>63.5<br>63.3<br>67.0<br>60.1|
|LM+ML|GraphCare (Jiang et al.,2024a)<br>RAM-EHR (Xu et al.,2024)<br>EMERGE (Zhu et al.,2024a)|91.5<br>80.3<br>57.8<br>96.6<br>90.5<br>78.4<br>52.6<br>97.0<br>90.7<br>78.3<br>53.4<br>96.6|65.7<br>65.5<br>66.2<br>65.0<br>65.5<br>65.5<br>64.0<br>67.0<br>63.3<br>63.2<br>61.5<br>64.9|
|LLM|Zero-shot (LLM: Claude 3.5 Sonnet)<br>w/ EHR context only<br>w/ Classic RAG[a]<br>w/ KARE-augmented context[b]|80.5<br>47.0<br>2.7<br>98.7<br>81.0<br>49.9<br>8.1<br>94.6<br>83.2<br>54.3<br>12.7<br>96.3|49.4<br>45.7<br>81.8<br>21.5<br>49.0<br>44.2<br>83.2<br>18.8<br>52.3<br>49.7<br>80.6<br>27.7<br>49.3<br>44.7<br>84.0<br>19.1<br>49.0<br>44.1<br>84.3<br>18.2<br>51.2<br>46.3<br>78.4<br>24.0<br>54.1<br>51.9<br>75.2<br>34.1<br>**73.9**<br>**73.8**<br>**85.6**<br>63.7|
||Few-Shot (LLM: Claude 3.5 Sonnet)<br>w/ exemplar only (N=2)[c]<br>w/ exemplar only (N=4)<br>w/ EHR-CoAgent[d] (Cui et al.,2024)<br>w/ KARE-augmented context|80.8<br>46.7<br>2.1<br>99.5<br>81.6<br>49.9<br>5.3<br>99.8<br>81.0<br>55.5<br>13.8<br>97.0<br>84.5<br>57.4<br>15.8<br>97.6||
||**KARE**(ours)|**94.1**<br>**90.4**<br>**73.2**<br>99.8||



[a] We retrieve up to ten documents from 30 M PubMed abstracts that are most similar to the base context. Dense retrieval is applied with Nomic (Nussbaum et al., 2024) (dim=768). [b] Context is augmented as described in Section 3.2 using our constructed KG communities. Up to ten community summaries are appended as supplementary information. [c] N=2 means that we retrieve two most similar patients (one from positive and the other from negative samples), similar to the strategy introduced in (Cui et al., 2024). [d] Since the code for EHR-CoAgent was not open-sourced at the time of our paper submission, we implemented it ourselves, as detailed in Appendix C. 

sure, crucial for the imbalanced datasets; (3) **Sensitivity** , which quantifies the model’s ability to identify patients at risk of mortality or readmission; and (4) **Specificity** , which assesses accuracy in identifying patients unlikely to experience these outcomes, helping avoid unnecessary measures. 

**Baselines.** We compare to three categories of baselines: (1) ML-based methods: GRU (Chung et al., 2014), Transformer (Vaswani et al., 2017), RETAIN (Choi et al., 2016), GRAM (Choi et al., 2017), Deepr (Nguyen et al., 2016), TCN (Bai et al., 2018), StageNet (Gao et al., 2020), ConCare (Ma et al., 2020b), AdaCare (Ma et al., 2020a), GRASP (Zhang et al., 2021), and KerPrint (Yang et al., 2023b); (2) LM + ML-based methods: GraphCare (Jiang et al., 2024a), RAM-EHR (Xu et al., 2024), and EMERGE (Zhu et al., 2024a); and (3) LLM-based methods: zero-shot and few-shot prompting with the advanced LLM Claude 3.5 Sonnet (Anthropic, 2024), and a few-shot-based method EHRCoAgent (Cui et al., 2024). We showcase the details of baseline implementation in Appendix C. 

**Implementation Details.** We utilize Scikit-learn (Pedregosa et al., 2018) for agglomerative clustering and Graspy (Chung et al., 2019) for the hierarchical Leiden algorithm. Semantic similarity calculations are performed using the Nomic embedding (Nussbaum et al., 2024) for PubMed abstracts 

8 

Under review as a conference paper 

Table 3: Ablation study of fine-tuning components. Results are averaged by 3 runs with different seeds. 

|**Similar**<br>**Retrieved**<br>**Reasoning**<br>**Patients**<br>**Knowledge**|**MIMIC-III-Mortality**<br>**Accuracy**<br>**Macro F1**<br>**Sensitivity**<br>**Specifcity**|**MIMIC-III-Readmission**|
|---|---|---|
|||**Accuracy**<br>**Macro F1**<br>**Sensitivity**<br>**Specifcity**|
||90.4<br>53.0<br>11.4<br>94.3|57.6<br>57.6<br>50.5<br>66.3|
||93.1<br>58.4<br>15.8<br>97.5|65.5<br>64.7<br>62.3<br>67.7|
||95.3<br>64.6<br>24.7<br>98.3|72.8<br>72.6<br>74.7<br>70.6|
||93.6<br>61.3<br>18.4<br>98.6|73.9<br>73.7<br>76.7<br>70.7|
||||
|**Similar**<br>**Retrieved**<br>**Reasoning**<br>**Patients**<br>**Knowledge**|**MIMIC-IV-Mortality**<br>**Accuracy**<br>**Macro F1**<br>**Sensitivity**<br>**Specifcity**<br>92.2<br>83.1<br>65.0<br>96.2|**MIMIC-IV-Readmission**|
|||**Accuracy**<br>**Macro F1**<br>**Sensitivity**<br>**Specifcity**|
|||56.1<br>46.7<br>23.1<br>76.2|
||93.3<br>85.4<br>67.3<br>97.5|64.7<br>62.1<br>69.3<br>55.9|
||93.8<br>89.6<br>74.5<br>98.8|72.2<br>71.9<br>81.1<br>64.0|
||94.1<br>90.4<br>73.2<br>99.9|73.9<br>73.8<br>85.6<br>63.7|



and the text-embedding-3-large model from Azure OpenAI for all other purposes. The optimal thresholds for semantic clustering are determined to be _θe_ = _θr_ = 0 _._ 14. We generate community summaries using _Zs_ = 20 and _Zc_ = 150, and employ hyperparameters _α_ = 0 _._ 1, _β_ = 0 _._ 7, _λ_ 1 = 0 _._ 2, _λ_ 2 = 0 _._ 2, and _λ_ 3 = 0 _._ 3 for patient context augmentation. Claude 3.5 Sonnet, accessed via the Amazon Bedrock platform[1] , is used as our expert LLM for generating reasoning chain training samples. Our fine-tuning framework is implemented using the TRL (von Werra et al., 2020), Transformers (Wolf et al., 2020), and FlashAttention-2 (Dao, 2024), with Mistral-7B-Instruct-v0.3 (Jiang et al., 2023) as our local LLM. We provide step-by-step implementation details in Appendix D.[2] 

## 4.2 EXPERIMENTAL RESULTS 

**Main Results.** Table 2 presents the main results and highlights several key observations: (1) KARE consistently outperforms all other methods across every dataset and task; (2) the na¨ıve RAG model sometimes fails to enhance zero-shot performance, while our method effectively augments context, leading to improved zero-shot predictions; (3) our context augmentation method is comparable to the state-of-the-art EHR-CoAgent in few-shot scenarios; and (4) our approach identifies more unique patterns, particularly excelling in correctly predicting true positives for mortality prediction, which other supervised models struggle to capture. We place some case studies in Appendix E. 

_Note_ : In mortality prediction using MIMIC-III/IV, sensitivity is crucial because positive cases are significantly fewer than negative ones, increasing the risk of overfitting. Accurately predicting positive cases is essential. Our model’s specificity is not always the highest, as efforts to enhance model’s overall capability can sometimes lead to misclassification of negative cases as positive. This is a well-known trade-off between sensitivity and specificity (Zweig & Campbell, 1993; Powers, 2020). Conversely, for readmission prediction, where datasets are balanced, the model is expected to perform equally well on both positive and negative samples. 

## **Ablation Study of Fine-tuning Components.** 

We perform an ablation study to assess the individual contributions of each component in boosting the performance of our fine-tuned model, as illustrated in Table 3 and Figure 2. The results in Table 3 show that all components (similar patients, retrieved knowledge, and reasoning) contribute positively to performance in most cases. However, in highly imbalanced datasets like MIMIC-III Mortality, including similar patients can degrade performance. This is likely because the retrieved patients for positive cases (label = 1) tend to be less similar when positive samples are scarce. Additionally, the absence of these components makes the fine-tuned model more prone to label bias, as seen in MIMIC-III Mortality and MIMIC-IV Readmission. 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_Reasoning-Enhanced_Healthcare_Predictions_with_Knowledge_Graph_Community_Retrieval_images/iclr_2025_Reasoning-Enhanced_Healthcare_Predictions_with_Knowledge_Graph_Community_Retrieval.pdf-0009-09.png)


**----- Start of picture text -----**<br>
0.28<br>Base w/o reasoning<br>0.26 Base w/ reasoning<br>Aug. w/o reasoning<br>0.24 Aug. w/ reasoning<br>0.22<br>0.20<br>0.18<br>0.16<br>0.00 0.25 0.50 0.75 1.00 1.25 1.50 1.75 2.00<br>Training Epoch<br>Validation Loss<br>**----- End of picture text -----**<br>


Figure 2: Validation loss of the label prediction task during the fine-tuning with different settings. Loss is computed every 1/4 epoch. Task: mortality prediction on MIMIC-IV. “Base” and “Aug.” denote base context and augmented context, respectively. 

> 1The use of Amazon Bedrock is authorized by MIMIC: https://physionet.org/news/post/gpt-responsible-use 

> 2Our code is available at: https://github.com/pat-jj/KARE 

9 

Under review as a conference paper 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_Reasoning-Enhanced_Healthcare_Predictions_with_Knowledge_Graph_Community_Retrieval_images/iclr_2025_Reasoning-Enhanced_Healthcare_Predictions_with_Knowledge_Graph_Community_Retrieval.pdf-0010-01.png)


**----- Start of picture text -----**<br>
Study of Context Augmentation Metrics Study of Knowledge Source<br>74<br>Full Implementation<br>72<br>70<br>68<br>66<br>w/o N.H. w/o Coh. w/o Rec. w/o T.R. w/o DGRA w/o G [KG] w/o G [BC] w/o G [LLM]<br>Macro F1<br>**----- End of picture text -----**<br>


Figure 3: Ablation study of ( _left_ ) the metrics we proposed for patient context augmentation, and ( _right_ ) the KG used as the knowledge source. N.H., Coh., Rec., and T.R. denote node hits, coherence, recency, and theme relevance, respectively. Tested task: MIMIC-IV-Readmission. 

Figure 2 further illustrates that fine-tuning with base context leads to early overfitting compared to fine-tuning with augmented context. Moreover, adding reasoning as a multitask objective accelerates convergence for models using base context, whereas it slows convergence when applied to models with augmented context. This suggests that learning reasoning over more information-rich contexts is more challenging, but ultimately results in a lower final loss once mastered. 

**Effect of Context Augmentation Components.** The LHS of Figure 3 compares the contribution of each metric proposed for community summary selection in patient context augmentation. The study shows that node hits is the most critical factor, followed by the DGRA algorithm, theme relevance, coherence, and recency, with each playing a distinct role in enhancing the final performance. 

**Effect of the Knowledge Source.** The RHS of Figure 3 shows how removing individual knowledge sources affects the model’s performance on the MIMIC-IV readmission task. While all KGs improve predictions, removing _G[KG]_ causes the smallest performance drop, whereas removing _G[LLM]_ leads to the largest decline. This highlights the importance of the LLM-extracted KG, likely due to its contextually relevant, clinically specific relationships. The UMLS-derived KG contributes less, likely because code mapping introduces sparsity by generalizing fine-grained concepts into more abstract categories (e.g., mapping “acute myocardial infarction” to “cardiovascular diseases”). This generalization limits the exploration of detailed relationships within the large KG. Future work will explore methods for retrieving knowledge with more fine-grained concepts from biomedical KGs. 

**Benefit of Multitask Learning.** We compare our multitask learning approach, which treats reasoning chain generation and outcome prediction as separate tasks, with a ”Two-In-One” method that only outputs the concatenated reasoning chain and groundtruth label. As shown in Table 4, mul- 

Table 4: Comparison of two strategies for fine-tuning LLM with reasoning chain and label. 

|Strategy|**MIMIC-IV-Mortality**<br>Macro F1<br>Sensitivity|**MIMIC-IV-Readmission**<br>Accuracy<br>Macro F1|
|---|---|---|
|Multitask<br>“Two-In-One”|90.4<br>73.2<br>86.5<br>68.0|73.9<br>73.8<br>67.2<br>65.4|



titask learning significantly outperforms the ”Two-In-One” approach for both mortality and readmission prediction on MIMIC-IV. This demonstrates that decoupling tasks allows better capture of each component’s nuances, yielding more robust patient representations. This framework enables the LLM to specialize in generating quality reasoning chains while making accurate predictions, resulting in a more effective and interpretable model. 

## 5 CONCLUSION 

We propose KARE, a novel framework that combines community-based knowledge graph retrieval with large language model reasoning to enhance healthcare predictions. KARE constructs a comprehensive knowledge graph, employs hierarchical community detection, and dynamically augments patient context with fine-grained, contextually relevant information. By fine-tuning a specialized smaller LLM, KARE generates interpretable reasoning chains for accurate predictions. Experiments on MIMIC-III and MIMIC-IV datasets demonstrate KARE’s superiority over state-of-the-art methods for mortality and readmission prediction tasks. Future work will focus on scaling KARE to more challenging healthcare tasks and exploring its applicability to other scientific domains, where integrating knowledge graphs and powerful language models can potentially drive groundbreaking scientific progress. We discuss ethics, broader impacts, and limitations in Appendix A. 

10 

Under review as a conference paper 

## REFERENCES 

Anthropic. Introducing claude 3.5 sonnet. https://www.anthropic.com/news/ claude-3-5-sonnet, 2024. Accessed: 2024-10-01. 

- Awais Ashfaq, Anita Sant’Anna, Markus Lingman, and Sławomir Nowaczyk. Readmission prediction using deep learning on electronic health records. _Journal of Biomedical Informatics_ , 97: 103256, 2019. ISSN 1532-0464. doi: https://doi.org/10.1016/j.jbi.2019.103256. URL https: //www.sciencedirect.com/science/article/pii/S1532046419301753. 

- Shaojie Bai, J Zico Kolter, and Vladlen Koltun. An empirical evaluation of generic convolutional and recurrent networks for sequence modeling. _arXiv preprint arXiv:1803.01271_ , 2018. 

- Suhana Bedi, Yutong Liu, Lucy Orr-Ewing, Dev Dash, Sanmi Koyejo, Alison Callahan, Jason A. Fries, Michael Wornow, Akshay Swaminathan, Lisa Soleymani Lehmann, Hyo Jung Hong, Mehr Kashyap, Akash R. Chaurasia, Nirav R. Shah, Karandeep Singh, Troy Tazbaz, Arnold Milstein, Michael A. Pfeffer, and Nigam H. Shah. A systematic review of testing and evaluation of healthcare applications of large language models (llms). _medRxiv_ , 2024. doi: 10.1101/2024.04.15.24305869. URL https://www.medrxiv.org/content/early/ 2024/04/16/2024.04.15.24305869. 

- Suman Bhoi, Mong Li Lee, Wynne Hsu, Hao Sen Andrew Fang, and Ngiap Chuan Tan. Personalizing medication recommendation with a graph-based approach. 40(3), nov 2021. ISSN 1046-8188. doi: 10.1145/3488668. URL https://doi.org/10.1145/3488668. 

- Olivier Bodenreider. The unified medical language system (umls): integrating biomedical terminology. _Nucleic acids research_ , 32(suppl ~~1~~ ):D267–D270, 2004. 

- X Cai, O Perez-Concha, E Coiera, F Martin-Sanchez, R Day, D Roffe, and B Gallego. Realtime prediction of mortality, readmission, and length of stay using electronic health record data. _Journal of the American Medical Informatics Association_ , 23(3):553–561, May 2016. doi: 10. 1093/jamia/ocv110. Epub 2015 Sep 15. 

- Kathi Canese and Sarah Weis. Pubmed: the bibliographic database. _The NCBI handbook_ , 2(1), 2013. 

- Edward Choi, Mohammad Taha Bahadori, Jimeng Sun, Joshua Kulas, Andy Schuetz, and Walter Stewart. Retain: An interpretable predictive model for healthcare using reverse time attention mechanism. _Advances in neural information processing systems_ , 29, 2016. 

- Edward Choi, Mohammad Taha Bahadori, Le Song, Walter F Stewart, and Jimeng Sun. Gram: graph-based attention model for healthcare representation learning. In _Proceedings of the 23rd ACM SIGKDD international conference on knowledge discovery and data mining_ , pp. 787–795, 2017. 

- Edward Choi, Cao Xiao, Walter F. Stewart, and Jimeng Sun. Mime: Multilevel medical embedding of electronic health records for predictive healthcare, 2018. URL https://arxiv.org/ abs/1810.09593. 

- Jaewon Chung, Benjamin D Pedigo, Eric W Bridgeford, Bijan K Varjavand, Hayden S Helm, and Joshua T Vogelstein. Graspy: Graph statistics in python. _Journal of Machine Learning Research_ , 20(158):1–7, 2019. 

- Junyoung Chung, Caglar Gulcehre, KyungHyun Cho, and Yoshua Bengio. Empirical evaluation of gated recurrent neural networks on sequence modeling. _arXiv preprint arXiv:1412.3555_ , 2014. 

- Hejie Cui, Zhuocheng Shen, Jieyu Zhang, Hui Shao, Lianhui Qin, Joyce C Ho, and Carl Yang. Llms-based few-shot disease predictions using ehr: A novel approach combining predictive agent reasoning and critical agent instruction. _arXiv preprint arXiv:2403.15464_ , 2024. 

- Tri Dao. FlashAttention-2: Faster attention with better parallelism and work partitioning. In _International Conference on Learning Representations (ICLR)_ , 2024. 

11 

Under review as a conference paper 

- K. Denecke, R. May, LLMHealthGroup, and O. Rivera Romero. Potential of large language models in health care: Delphi study. _Journal of Medical Internet Research_ , 26:e52399, May 2024. doi: 10.2196/52399. 

- Darren Edge, Ha Trinh, Newman Cheng, Joshua Bradley, Alex Chao, Apurva Mody, Steven Truitt, and Jonathan Larson. From local to global: A graph rag approach to query-focused summarization. _arXiv preprint arXiv:2404.16130_ , 2024. 

- Santo Fortunato. Community detection in graphs. _Physics reports_ , 486(3-5):75–174, 2010. 

- Junyi Gao, Cao Xiao, Yasha Wang, Wen Tang, Lucas M Glass, and Jimeng Sun. Stagenet: Stageaware neural networks for health risk prediction. In _Proceedings of The Web Conference 2020_ , pp. 530–540, 2020. 

- Sepp Hochreiter and J¨urgen Schmidhuber. Long short-term memory. _Neural Comput._ , 9(8): 1735–1780, nov 1997. ISSN 0899-7667. doi: 10.1162/neco.1997.9.8.1735. URL https: //doi.org/10.1162/neco.1997.9.8.1735. 

- Albert Q Jiang, Alexandre Sablayrolles, Arthur Mensch, Chris Bamford, Devendra Singh Chaplot, Diego de las Casas, Florian Bressand, Gianna Lengyel, Guillaume Lample, Lucile Saulnier, et al. Mistral 7b. _arXiv preprint arXiv:2310.06825_ , 2023. 

- Pengcheng Jiang, Cao Xiao, Adam Richard Cross, and Jimeng Sun. Graphcare: Enhancing healthcare predictions with personalized knowledge graphs. In _The Twelfth International Conference on Learning Representations_ , 2024a. 

- Pengcheng Jiang, Cao Xiao, Zifeng Wang, Parminder Bhatia, Jimeng Sun, and Jiawei Han. TriSum: Learning summarization ability from large language models with structured rationale. In Kevin Duh, Helena Gomez, and Steven Bethard (eds.), _Proceedings of the 2024 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies (Volume 1: Long Papers)_ , pp. 2805–2819, Mexico City, Mexico, June 2024b. Association for Computational Linguistics. doi: 10.18653/v1/2024.naacl-long.154. URL https: //aclanthology.org/2024.naacl-long.154. 

- Di Jin, Zhizhi Yu, Pengfei Jiao, Shirui Pan, Dongxiao He, Jia Wu, S Yu Philip, and Weixiong Zhang. A survey of community detection approaches: From statistical modeling to deep learning. _IEEE Transactions on Knowledge and Data Engineering_ , 35(2):1149–1170, 2021. 

- Alistair Johnson, Lucas Bulgarelli, Tom Pollard, Steven Horng, Leo Anthony Celi, and Roger Mark. Mimic-iv. _PhysioNet. Available online at: https://physionet. org/content/mimiciv/1.0/(accessed August 23, 2021)_ , 2020. 

- Alistair EW Johnson, Tom J Pollard, Lu Shen, Li-wei H Lehman, Mengling Feng, Mohammad Ghassemi, Benjamin Moody, Peter Szolovits, Leo Anthony Celi, and Roger G Mark. Mimic-iii, a freely accessible critical care database. _Scientific data_ , 3(1):1–9, 2016. 

- Yubin Kim, Xuhai Xu, Daniel McDuff, Cynthia Breazeal, and Hae Won Park. Health-llm: Large language models for health prediction via wearable sensor data, 2024. URL https://arxiv. org/abs/2401.06866. 

- Patrick Lewis, Ethan Perez, Aleksandra Piktus, Fabio Petroni, Vladimir Karpukhin, Naman Goyal, Heinrich K¨uttler, Mike Lewis, Wen tau Yih, Tim Rockt¨aschel, Sebastian Riedel, and Douwe Kiela. Retrieval-augmented generation for knowledge-intensive nlp tasks, 2021. URL https: //arxiv.org/abs/2005.11401. 

- Jiarui Li, Ye Yuan, and Zehua Zhang. Enhancing llm factual accuracy with rag to counter hallucinations: A case study on domain-specific queries in private knowledge-bases, 2024. URL https://arxiv.org/abs/2403.10446. 

- Weijie Liu, Peng Zhou, Zhe Zhao, Zhiruo Wang, Qi Ju, Haotang Deng, and Ping Wang. K-bert: Enabling language representation with knowledge graph, 2019. URL https://arxiv.org/ abs/1909.07606. 

12 

Under review as a conference paper 

- Liantao Ma, Junyi Gao, Yasha Wang, Chaohe Zhang, Jiangtao Wang, Wenjie Ruan, Wen Tang, Xin Gao, and Xinyu Ma. Adacare: Explainable clinical health status representation learning via scaleadaptive feature extraction and recalibration. In _Proceedings of the AAAI Conference on Artificial Intelligence_ , volume 34, pp. 825–832, 2020a. 

- Liantao Ma, Chaohe Zhang, Yasha Wang, Wenjie Ruan, Jiangtao Wang, Wen Tang, Xinyu Ma, Xin Gao, and Junyi Gao. Concare: Personalized clinical feature embedding via capturing the healthcare context. In _Proceedings of the AAAI Conference on Artificial Intelligence_ , volume 34, pp. 833–840, 2020b. 

- Varun Magesh, Faiz Surani, Matthew Dahl, Mirac Suzgun, Christopher D. Manning, and Daniel E. Ho. Hallucination-free? assessing the reliability of leading ai legal research tools, 2024. URL https://arxiv.org/abs/2405.20362. 

- Daniel M¨ullner. Modern hierarchical, agglomerative clustering algorithms. _arXiv preprint arXiv:1109.2378_ , 2011. 

- Phuoc Nguyen, Truyen Tran, Nilmini Wickramasinghe, and Svetha Venkatesh. Deepr: a convolutional net for medical records. _IEEE journal of biomedical and health informatics_ , 21(1):22–30, 2016. 

- Cheng Niu, Yuanhao Wu, Juno Zhu, Siliang Xu, Kashun Shum, Randy Zhong, Juntong Song, and Tong Zhang. Ragtruth: A hallucination corpus for developing trustworthy retrieval-augmented language models, 2024. URL https://arxiv.org/abs/2401.00396. 

- Zach Nussbaum, John X Morris, Brandon Duderstadt, and Andriy Mulyar. Nomic embed: Training a reproducible long context text embedder. _arXiv preprint arXiv:2402.01613_ , 2024. 

- OpenAI, Josh Achiam, Steven Adler, Sandhini Agarwal, Lama Ahmad, Ilge Akkaya, Florencia Leoni Aleman, Diogo Almeida, Janko Altenschmidt, Sam Altman, Shyamal Anadkat, Red Avila, Igor Babuschkin, Suchir Balaji, Valerie Balcom, Paul Baltescu, Haiming Bao, Mohammad Bavarian, Jeff Belgum, Irwan Bello, Jake Berdine, Gabriel Bernadett-Shapiro, Christopher Berner, Lenny Bogdonoff, Oleg Boiko, Madelaine Boyd, Anna-Luisa Brakman, Greg Brockman, Tim Brooks, Miles Brundage, Kevin Button, Trevor Cai, Rosie Campbell, Andrew Cann, Brittany Carey, Chelsea Carlson, Rory Carmichael, Brooke Chan, Che Chang, Fotis Chantzis, Derek Chen, Sully Chen, Ruby Chen, Jason Chen, Mark Chen, Ben Chess, Chester Cho, Casey Chu, Hyung Won Chung, Dave Cummings, Jeremiah Currier, Yunxing Dai, Cory Decareaux, Thomas Degry, Noah Deutsch, Damien Deville, Arka Dhar, David Dohan, Steve Dowling, Sheila Dunning, Adrien Ecoffet, Atty Eleti, Tyna Eloundou, David Farhi, Liam Fedus, Niko Felix, Sim´on Posada Fishman, Juston Forte, Isabella Fulford, Leo Gao, Elie Georges, Christian Gibson, Vik Goel, Tarun Gogineni, Gabriel Goh, Rapha Gontijo-Lopes, Jonathan Gordon, Morgan Grafstein, Scott Gray, Ryan Greene, Joshua Gross, Shixiang Shane Gu, Yufei Guo, Chris Hallacy, Jesse Han, Jeff Harris, Yuchen He, Mike Heaton, Johannes Heidecke, Chris Hesse, Alan Hickey, Wade Hickey, Peter Hoeschele, Brandon Houghton, Kenny Hsu, Shengli Hu, Xin Hu, Joost Huizinga, Shantanu Jain, Shawn Jain, Joanne Jang, Angela Jiang, Roger Jiang, Haozhun Jin, Denny Jin, Shino Jomoto, Billie Jonn, Heewoo Jun, Tomer Kaftan, Łukasz Kaiser, Ali Kamali, Ingmar Kanitscheider, Nitish Shirish Keskar, Tabarak Khan, Logan Kilpatrick, Jong Wook Kim, Christina Kim, Yongjik Kim, Jan Hendrik Kirchner, Jamie Kiros, Matt Knight, Daniel Kokotajlo, Łukasz Kondraciuk, Andrew Kondrich, Aris Konstantinidis, Kyle Kosic, Gretchen Krueger, Vishal Kuo, Michael Lampe, Ikai Lan, Teddy Lee, Jan Leike, Jade Leung, Daniel Levy, Chak Ming Li, Rachel Lim, Molly Lin, Stephanie Lin, Mateusz Litwin, Theresa Lopez, Ryan Lowe, Patricia Lue, Anna Makanju, Kim Malfacini, Sam Manning, Todor Markov, Yaniv Markovski, Bianca Martin, Katie Mayer, Andrew Mayne, Bob McGrew, Scott Mayer McKinney, Christine McLeavey, Paul McMillan, Jake McNeil, David Medina, Aalok Mehta, Jacob Menick, Luke Metz, Andrey Mishchenko, Pamela Mishkin, Vinnie Monaco, Evan Morikawa, Daniel Mossing, Tong Mu, Mira Murati, Oleg Murk, David M´ely, Ashvin Nair, Reiichiro Nakano, Rajeev Nayak, Arvind Neelakantan, Richard Ngo, Hyeonwoo Noh, Long Ouyang, Cullen O’Keefe, Jakub Pachocki, Alex Paino, Joe Palermo, Ashley Pantuliano, Giambattista Parascandolo, Joel Parish, Emy Parparita, Alex Passos, Mikhail Pavlov, Andrew Peng, Adam Perelman, Filipe de Avila Belbute Peres, Michael Petrov, Henrique Ponde de Oliveira Pinto, Michael, Pokorny, Michelle Pokrass, Vitchyr H. Pong, Tolly Powell, Alethea Power, Boris Power, Elizabeth Proehl, 

13 

Under review as a conference paper 

Raul Puri, Alec Radford, Jack Rae, Aditya Ramesh, Cameron Raymond, Francis Real, Kendra Rimbach, Carl Ross, Bob Rotsted, Henri Roussez, Nick Ryder, Mario Saltarelli, Ted Sanders, Shibani Santurkar, Girish Sastry, Heather Schmidt, David Schnurr, John Schulman, Daniel Selsam, Kyla Sheppard, Toki Sherbakov, Jessica Shieh, Sarah Shoker, Pranav Shyam, Szymon Sidor, Eric Sigler, Maddie Simens, Jordan Sitkin, Katarina Slama, Ian Sohl, Benjamin Sokolowsky, Yang Song, Natalie Staudacher, Felipe Petroski Such, Natalie Summers, Ilya Sutskever, Jie Tang, Nikolas Tezak, Madeleine B. Thompson, Phil Tillet, Amin Tootoonchian, Elizabeth Tseng, Preston Tuggle, Nick Turley, Jerry Tworek, Juan Felipe Cer´on Uribe, Andrea Vallone, Arun Vijayvergiya, Chelsea Voss, Carroll Wainwright, Justin Jay Wang, Alvin Wang, Ben Wang, Jonathan Ward, Jason Wei, CJ Weinmann, Akila Welihinda, Peter Welinder, Jiayi Weng, Lilian Weng, Matt Wiethoff, Dave Willner, Clemens Winter, Samuel Wolrich, Hannah Wong, Lauren Workman, Sherwin Wu, Jeff Wu, Michael Wu, Kai Xiao, Tao Xu, Sarah Yoo, Kevin Yu, Qiming Yuan, Wojciech Zaremba, Rowan Zellers, Chong Zhang, Marvin Zhang, Shengjia Zhao, Tianhao Zheng, Juntang Zhuang, William Zhuk, and Barret Zoph. Gpt-4 technical report, 2024. URL https://arxiv.org/abs/2303.08774. 

- Fabian Pedregosa, Ga¨el Varoquaux, Alexandre Gramfort, Vincent Michel, Bertrand Thirion, Olivier Grisel, Mathieu Blondel, Andreas M¨uller, Joel Nothman, Gilles Louppe, Peter Prettenhofer, Ron Weiss, Vincent Dubourg, Jake Vanderplas, Alexandre Passos, David Cournapeau, Matthieu Brucher, Matthieu Perrot, and Edouard[´] Duchesnay. Scikit-learn: Machine learning in python, 2018. URL https://arxiv.org/abs/1201.0490. 

- David MW Powers. Evaluation: from precision, recall and f-measure to roc, informedness, markedness and correlation. _arXiv preprint arXiv:2010.16061_ , 2020. 

- Jeff Rasley, Samyam Rajbhandari, Olatunji Ruwase, and Yuxiong He. Deepspeed: System optimizations enable training deep learning models with over 100 billion parameters. In _Proceedings of the 26th ACM SIGKDD International Conference on Knowledge Discovery & Data Mining_ , KDD ’20, pp. 3505–3506, New York, NY, USA, 2020. Association for Computing Machinery. ISBN 9781450379984. doi: 10.1145/3394486.3406703. URL https://doi.org/10. 1145/3394486.3406703. 

- Guillaume Salha, Stratis Limnios, Romain Hennequin, Viet-Anh Tran, and Michalis Vazirgiannis. Gravity-inspired graph autoencoders for directed link prediction. In _Proceedings of the 28th ACM international conference on information and knowledge management_ , pp. 589–598, 2019. 

- Ketan Rajshekhar Shahapure and Charles Nicholas. Cluster quality analysis using silhouette score. In _2020 IEEE 7th international conference on data science and advanced analytics (DSAA)_ , pp. 747–748. IEEE, 2020. 

- Wenqi Shi, Ran Xu, Yuchen Zhuang, Yue Yu, Jieyu Zhang, Hang Wu, Yuanda Zhu, Joyce Ho, Carl Yang, and May D. Wang. Ehragent: Code empowers large language models for few-shot complex tabular reasoning on electronic health records, 2024. URL https://arxiv.org/ abs/2401.07128. 

- Benjamin Shickel, Patrick James Tighe, Azra Bihorac, and Parisa Rashidi. Deep ehr: A survey of recent advances in deep learning techniques for electronic health record (ehr) analysis. _IEEE Journal of Biomedical and Health Informatics_ , 22(5):1589–1604, September 2018. ISSN 21682208. doi: 10.1109/jbhi.2017.2767063. URL http://dx.doi.org/10.1109/JBHI. 2017.2767063. 

- Karthik Soman, Peter W Rose, John H Morris, Rabia E Akbas, Brett Smith, Braian Peetoom, Catalina Villouta-Reyes, Gabriel Cerono, Yongmei Shi, Angela Rizk-Jackson, Sharat Israni, Charlotte A Nelson, Sui Huang, and Sergio E Baranzini. Biomedical knowledge graph-optimized prompt generation for large language models, 2024. URL https://arxiv.org/abs/ 2311.17330. 

- Gemma Team, Thomas Mesnard, Cassidy Hardin, Robert Dadashi, Surya Bhupatiraju, Shreya Pathak, Laurent Sifre, Morgane Rivi`ere, Mihir Sanjay Kale, Juliette Love, Pouya Tafti, L´eonard Hussenot, Pier Giuseppe Sessa, Aakanksha Chowdhery, Adam Roberts, Aditya Barua, Alex Botev, Alex Castro-Ros, Ambrose Slone, Am´elie H´eliou, Andrea Tacchetti, Anna Bulanova, Antonia Paterson, Beth Tsai, Bobak Shahriari, Charline Le Lan, Christopher A. Choquette-Choo, 

14 

Under review as a conference paper 

Cl´ement Crepy, Daniel Cer, Daphne Ippolito, David Reid, Elena Buchatskaya, Eric Ni, Eric Noland, Geng Yan, George Tucker, George-Christian Muraru, Grigory Rozhdestvenskiy, Henryk Michalewski, Ian Tenney, Ivan Grishchenko, Jacob Austin, James Keeling, Jane Labanowski, Jean-Baptiste Lespiau, Jeff Stanway, Jenny Brennan, Jeremy Chen, Johan Ferret, Justin Chiu, Justin Mao-Jones, Katherine Lee, Kathy Yu, Katie Millican, Lars Lowe Sjoesund, Lisa Lee, Lucas Dixon, Machel Reid, Maciej Mikuła, Mateo Wirth, Michael Sharman, Nikolai Chinaev, Nithum Thain, Olivier Bachem, Oscar Chang, Oscar Wahltinez, Paige Bailey, Paul Michel, Petko Yotov, Rahma Chaabouni, Ramona Comanescu, Reena Jana, Rohan Anil, Ross McIlroy, Ruibo Liu, Ryan Mullins, Samuel L Smith, Sebastian Borgeaud, Sertan Girgin, Sholto Douglas, Shree Pandya, Siamak Shakeri, Soham De, Ted Klimenko, Tom Hennigan, Vlad Feinberg, Wojciech Stokowiec, Yu hui Chen, Zafarali Ahmed, Zhitao Gong, Tris Warkentin, Ludovic Peran, Minh Giang, Cl´ement Farabet, Oriol Vinyals, Jeff Dean, Koray Kavukcuoglu, Demis Hassabis, Zoubin Ghahramani, Douglas Eck, Joelle Barral, Fernando Pereira, Eli Collins, Armand Joulin, Noah Fiedel, Evan Senter, Alek Andreev, and Kathleen Kenealy. Gemma: Open models based on gemini research and technology, 2024. URL https://arxiv.org/abs/2403.08295. 

- Brandon Theodorou, Cao Xiao, and Jimeng Sun. Synthesize high-dimensional longitudinal electronic health records via hierarchical autoregressive language model. _Nature communications_ , 14 (1):5305, 2023. 

- Hugo Touvron, Thibaut Lavril, Gautier Izacard, Xavier Martinet, Marie-Anne Lachaux, Timoth´ee Lacroix, Baptiste Rozi`ere, Naman Goyal, Eric Hambro, Faisal Azhar, Aurelien Rodriguez, Armand Joulin, Edouard Grave, and Guillaume Lample. Llama: Open and efficient foundation language models, 2023a. URL https://arxiv.org/abs/2302.13971. 

- Hugo Touvron, Louis Martin, Kevin Stone, Peter Albert, Amjad Almahairi, Yasmine Babaei, Nikolay Bashlykov, Soumya Batra, Prajjwal Bhargava, Shruti Bhosale, Dan Bikel, Lukas Blecher, Cristian Canton Ferrer, Moya Chen, Guillem Cucurull, David Esiobu, Jude Fernandes, Jeremy Fu, Wenyin Fu, Brian Fuller, Cynthia Gao, Vedanuj Goswami, Naman Goyal, Anthony Hartshorn, Saghar Hosseini, Rui Hou, Hakan Inan, Marcin Kardas, Viktor Kerkez, Madian Khabsa, Isabel Kloumann, Artem Korenev, Punit Singh Koura, Marie-Anne Lachaux, Thibaut Lavril, Jenya Lee, Diana Liskovich, Yinghai Lu, Yuning Mao, Xavier Martinet, Todor Mihaylov, Pushkar Mishra, Igor Molybog, Yixin Nie, Andrew Poulton, Jeremy Reizenstein, Rashi Rungta, Kalyan Saladi, Alan Schelten, Ruan Silva, Eric Michael Smith, Ranjan Subramanian, Xiaoqing Ellen Tan, Binh Tang, Ross Taylor, Adina Williams, Jian Xiang Kuan, Puxin Xu, Zheng Yan, Iliyan Zarov, Yuchen Zhang, Angela Fan, Melanie Kambadur, Sharan Narang, Aurelien Rodriguez, Robert Stojnic, Sergey Edunov, and Thomas Scialom. Llama 2: Open foundation and fine-tuned chat models, 2023b. URL https://arxiv.org/abs/2307.09288. 

- Vincent A Traag, Ludo Waltman, and Nees Jan Van Eck. From louvain to leiden: guaranteeing well-connected communities. _Scientific reports_ , 9(1):1–12, 2019. 

- Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez, Łukasz Kaiser, and Illia Polosukhin. Attention is all you need. _Advances in neural information processing systems_ , 30, 2017. 

- Leandro von Werra, Younes Belkada, Lewis Tunstall, Edward Beeching, Tristan Thrush, Nathan Lambert, Shengyi Huang, Kashif Rasul, and Quentin Gallou´edec. Trl: Transformer reinforcement learning. https://github.com/huggingface/trl, 2020. 

- Thomas Wolf, Lysandre Debut, Victor Sanh, Julien Chaumond, Clement Delangue, Anthony Moi, Pierric Cistac, Tim Rault, R´emi Louf, Morgan Funtowicz, Joe Davison, Sam Shleifer, Patrick von Platen, Clara Ma, Yacine Jernite, Julien Plu, Canwen Xu, Teven Le Scao, Sylvain Gugger, Mariama Drame, Quentin Lhoest, and Alexander M. Rush. Transformers: State-of-the-art natural language processing. In _Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing: System Demonstrations_ , pp. 38–45, Online, October 2020. Association for Computational Linguistics. URL https://www.aclweb.org/anthology/ 2020.emnlp-demos.6. 

- M. Wornow, Y. Xu, R. Thapa, et al. The shaky foundations of large language models and foundation models for electronic health records. _NPJ Digital Medicine_ , 6:135, 2023. doi: 10.1038/ s41746-023-00879-8. URL https://doi.org/10.1038/s41746-023-00879-8. 

15 

Under review as a conference paper 

- Zhenbang Wu, Cao Xiao, and Jimeng Sun. Medlink: De-identified patient health record linkage. In _Proceedings of the 29th ACM SIGKDD Conference on Knowledge Discovery and Data Mining_ , pp. 2672–2682, 2023. 

- Ran Xu, Wenqi Shi, Yue Yu, Yuchen Zhuang, Bowen Jin, May D Wang, Joyce C Ho, and Carl Yang. Ram-ehr: Retrieval augmentation meets clinical predictions on electronic health records. In _Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics_ , 2024. 

- Yongxin Xu, Xu Chu, Kai Yang, Zhiyuan Wang, Peinie Zou, Hongxin Ding, Junfeng Zhao, Yasha Wang, and Bing Xie. Seqcare: Sequential training with external medical knowledge graph for diagnosis prediction in healthcare data. In _Proceedings of the ACM Web Conference 2023_ , WWW ’23, pp. 2819–2830, New York, NY, USA, 2023. Association for Computing Machinery. ISBN 9781450394161. doi: 10.1145/3543507.3583543. URL https: //doi.org/10.1145/3543507.3583543. 

- Chaoqi Yang, Zhenbang Wu, Patrick Jiang, Zhen Lin, Junyi Gao, Benjamin Danek, and Jimeng Sun. PyHealth: A deep learning toolkit for healthcare predictive modeling. In _Proceedings of the 27th ACM SIGKDD International Conference on Knowledge Discovery and Data Mining (KDD) 2023_ , 2023a. URL https://github.com/sunlabuiuc/PyHealth. 

- Kai Yang, Yongxin Xu, Peinie Zou, Hongxin Ding, Junfeng Zhao, Yasha Wang, and Bing Xie. Kerprint: local-global knowledge graph enhanced diagnosis prediction for retrospective and prospective interpretations. In _Proceedings of the AAAI Conference on Artificial Intelligence_ , volume 37, pp. 5357–5365, 2023b. 

- X. Yang, A. Chen, N. PourNejatian, H. C. Shin, K. E. Smith, C. Parisien, C. Compas, C. Martin, A. B. Costa, M. G. Flores, Y. Zhang, T. Magoc, C. A. Harle, G. Lipori, D. A. Mitchell, W. R. Hogan, E. A. Shenkman, J. Bian, and Y. Wu. A large language model for electronic health records. _NPJ Digital Medicine_ , 5(1):194, December 2022. doi: 10.1038/s41746-022-00742-2. 

- Michihiro Yasunaga, Hongyu Ren, Antoine Bosselut, Percy Liang, and Jure Leskovec. Qa-gnn: Reasoning with language models and knowledge graphs for question answering, 2022. URL https://arxiv.org/abs/2104.06378. 

- Chaohe Zhang, Xin Gao, Liantao Ma, Yasha Wang, Jiangtao Wang, and Wen Tang. Grasp: generic framework for health status representation learning based on incorporating knowledge from similar patients. In _Proceedings of the AAAI conference on artificial intelligence_ , volume 35, pp. 715–723, 2021. 

- Xikun Zhang, Antoine Bosselut, Michihiro Yasunaga, Hongyu Ren, Percy Liang, Christopher D. Manning, and Jure Leskovec. Greaselm: Graph reasoning enhanced language models for question answering, 2022. URL https://arxiv.org/abs/2201.08860. 

- Yinghao Zhu, Changyu Ren, Zixiang Wang, Xiaochen Zheng, Shiyun Xie, Junlan Feng, Xi Zhu, Zhoujun Li, Liantao Ma, and Chengwei Pan. Emerge: Integrating rag for improved multimodal ehr predictive modeling. _arXiv preprint arXiv:2406.00036_ , 2024a. 

- Yinghao Zhu, Zixiang Wang, Junyi Gao, Yuning Tong, Jingkun An, Weibin Liao, Ewen M Harrison, Liantao Ma, and Chengwei Pan. Prompting large language models for zero-shot clinical prediction with structured longitudinal electronic health record data. _arXiv preprint arXiv:2402.01713_ , 2024b. 

- Mark H Zweig and Gregory Campbell. Receiver-operating characteristic (roc) plots: a fundamental evaluation tool in clinical medicine. _Clinical chemistry_ , 39(4):561–577, 1993. 

16 

Under review as a conference paper 

|**Contents of Appendix**|**Contents of Appendix**||
|---|---|---|
|**A **|**Ethics, Broader Impacts, and Limitations**|**18**|
|**B**|**Details of Knowledge Graph Construction**|**18**|
||B.1<br>KG Extraction from Large Biomedical KG. . . . . . . . . . . . . . . . . . . . . .|18|
||B.2<br>KG Extraction from Large Biomedical Corpus . . . . . . . . . . . . . . . . . . . .|19|
||B.3<br>KG Extraction from Large Language Models<br>. . . . . . . . . . . . . . . . . . . .|21|
|**C **|**Baseline Implementations**|**23**|
|**D **|**Implementation Details of KARE**|**26**|
||D.1<br>Step 1: Medical Concept Knowledge Graph Construction . . . . . . . . . . . . . .|26|
||D.2<br>Step 2: Patient Context Construction and Augmentation . . . . . . . . . . . . . . .|26|
||D.3<br>Step 3: Reasoning-Enhanced Precise Healthcare Prediction . . . . . . . . . . . . .|26|
|**E**|**Case Study**|**28**|
||E.1<br>Reasoning-Enhanced Prediction by Our Fine-Tuned Model . . . . . . . . . . . . .|28|
||E.2<br>Zero-Shot & Few-Shot Comparison<br>. . . . . . . . . . . . . . . . . . . . . . . . .|29|
|**F**|**Templates, Prompts, and Examples**|**34**|
|**G **|**Notations**|**39**|



17 

Under review as a conference paper 

## A ETHICS, BROADER IMPACTS, AND LIMITATIONS 

**Ethics:** Our work involves analysis of electronic health record (EHR) data, which contains sensitive personal medical information. To ensure the ethical handling of this data, we conducted all interactions between the language models and the EHR data through Amazon Bedrock34, which provides rigorous compliance standards and privacy protection measures. This allowed us to fully leverage the capabilities of the LLMs while maintaining strict confidentiality of the patient data. 

**Broader Impacts:** Our framework, KARE, demonstrates the potential for integrating knowledge graphs and language model reasoning to enhance clinical decision support systems. By providing more accurate and interpretable predictions for critical outcomes like mortality and readmission, KARE could assist healthcare providers in identifying high-risk patients who may require additional interventions or closer monitoring. This could ultimately lead to improved patient outcomes and more efficient allocation of healthcare resources. However, it is important to recognize that our models are intended to augment, rather than replace, the judgment of healthcare professionals. The predictions should be considered as additional data points to inform clinical decision making, not as definitive diagnoses or treatment recommendations. 

**Limitations:** While KARE achieves promising results, there are several limitations to consider. First, our evaluation is based on the MIMIC-III and MIMIC-IV datasets, which represent a specific patient population from a single hospital system in the United States. The generalizability of our findings to other patient populations or healthcare settings may be limited. Second, our knowledge graphs are constructed from a subset of biomedical databases, literature, and language model outputs, and may not capture the full breadth of medical knowledge. Expanding the knowledge sources and improving the knowledge extraction and integration processes could further enhance the performance of our models. Third, our framework relies on the outputs of large language models, which are known to have biases and can generate hallucinations. While we have taken steps to mitigate these issues, such as collaborating with medical experts to validate the extracted knowledge, there remains a risk of the models producing incorrect or biased predictions in some cases. Fourth, our work is based on English biomedical literature and EHR data, and we did not evaluate the performance of KARE on data in other languages. The applicability and effectiveness of our approach for non-English clinical settings requires further investigation. Ongoing research is needed to develop more robust and reliable language models for clinical applications across diverse languages and populations. 

In conclusion, KARE represents an important step towards leveraging knowledge graphs and language model reasoning for improved clinical predictions. However, further research is needed to address the limitations and ensure the safe and responsible deployment of such models in real-world healthcare settings. We encourage future work to focus on enhancing the generalizability, interpretability, and robustness of these approaches, as well as engaging with healthcare stakeholders to develop guidelines for the ethical and effective use of AI in clinical decision support. 

## B DETAILS OF KNOWLEDGE GRAPH CONSTRUCTION 

## B.1 KG EXTRACTION FROM LARGE BIOMEDICAL KG 

To construct concept-specific knowledge graphs from the Unified Medical Language System (UMLS), we follow a multi-step process: 

1. **Extracting Co-existing Concepts from EHR Data** : We iterate through the patient EHR dataset and collect the top _X_ ( _X_ = 20 in our implementation) most frequently co-existing concepts for each unique medical concept based on their co-occurrence in patient records. 

2. **Mapping Concepts to UMLS CUIs** : We map the medical concepts from the EHR data to their corresponding Concept Unique Identifiers (CUIs) in UMLS. This involves mapping condition and procedure concepts to CCS codes, then to ICD-9 codes, and finally to UMLS CUIs. Drug concepts are directly mapped to ATC codes and then to UMLS CUIs. 

> 3https://docs.aws.amazon.com/bedrock/latest/userguide/what-is-bedrock. html 

> 4https://physionet.org/news/post/gpt-responsible-use 

18 

Under review as a conference paper 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_Reasoning-Enhanced_Healthcare_Predictions_with_Knowledge_Graph_Community_Retrieval_images/iclr_2025_Reasoning-Enhanced_Healthcare_Predictions_with_Knowledge_Graph_Community_Retrieval.pdf-0019-01.png)


**----- Start of picture text -----**<br>
EHR Concepts in  Top N Coexisting Concepts Pairwise Shortest Path Finding  Concept-Specific<br>Each Visit with MAX Length = 2 KGs<br>concept_1: {<br>⋮ [concept_1] concept_4,<br>[concept_2] concept_5, G [KG] [concept 1 ]<br>[concept_3] …,<br>[concept_4] }<br>⋮ [concept_5]<br>⋮ ⋮ ⋮ ⋮<br>**----- End of picture text -----**<br>


Figure 4: Our pipeline to construct concept-specific KG _G[KG]_ with bio KG (UMLS) and EHR. 

3. **Extracting Subgraphs from UMLS** : For each medical concept _ci_ and its top _X_ co-existing concepts _Rci_ , we extract a concept-specific subgraph _G_[KG] _ci_ from UMLS using a bidirectional shortest path finding algorithm. The algorithm parameters are set as follows: 

   - max ~~l~~ ength=7: The maximum length of the shortest paths considered between concepts. 

   - max ~~p~~ aths=40: The maximum number of shortest paths to retrieve for each pair of concepts. 

   - max ~~n~~ odes=12000: The maximum number of nodes to explore during the bidirectional search. 

The bidirectional shortest path finding algorithm is implemented as follows: 

**Algorithm 2** Bidirectional Shortest Path Finding 

**Input:** Graph _G_ , start node _s_ , end node _t_ , max length _l_ , max paths _p_ , max nodes _n_ **Output:** Set of shortest paths _P_ Initialize forward queue _Qf ←{_ [ _s_ ] _}_ and backward queue _Qb ←{_ [ _t_ ] _}_ Initialize forward visited dict _Vf ←{s_ : [ _s_ ] _}_ and backward visited dict _Vb ←{t_ : [ _t_ ] _}_ Initialize paths _P ←∅_ and nodes explored _N ←_ 0 **while** _Qf and Qb and |P | < p and N < n_ **do** Expand forward path _πf ← Qf ._ popleft() and last node _u ← πf_ [ _−_ 1] **if** _u ∈ Vb_ **then** _P ← P ∪{πf_ + _Vb_ [ _u_ ][:: _−_ 1][1 :] _}_ **end** Expand backward path _πb ← Qb._ popleft() and last node _u ← πb_ [ _−_ 1] **if** _u ∈ Vf_ **then** _P ← P ∪{Vf_ [ _u_ ] + _πb_ [:: _−_ 1][1 :] _}_ **end if** _len_ ( _πf_ ) _<_ 2 _l or len_ ( _πb_ ) _<_ 2 _l_ **then** Expand neighbors and update _Qf_ , _Qb_ , _Vf_ , _Vb_ **end end return** _P_ 

The extracted shortest paths for each pair ( _ci, cj_ ) are combined to form the concept-specific subgraph _G_[KG] _ci_[.][The][nodes] _[V] c_[KG] _i_ and edges _Ec_[KG] _i_[of][the][subgraph][are][the][union][of][all][nodes][and] edges in the extracted paths. 

By following this process, we construct a set of concept-specific knowledge graphs _{G_[KG] _ci[}]_[that] capture the relevant relationships and contextual information for each medical concept _ci_ based on the UMLS knowledge graph and real-world co-occurrence patterns in patient EHR data. 

The number of the resulting KG triples from UMLS is 29,434. 

B.2 KG EXTRACTION FROM LARGE BIOMEDICAL CORPUS 

To construct concept-specific knowledge graphs, we process the EHR dataset and the PubMed Abstracts corpus using the following pipeline (Figure 5): 

19 

Under review as a conference paper 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_Reasoning-Enhanced_Healthcare_Predictions_with_Knowledge_Graph_Community_Retrieval_images/iclr_2025_Reasoning-Enhanced_Healthcare_Predictions_with_Knowledge_Graph_Community_Retrieval.pdf-0020-01.png)


**----- Start of picture text -----**<br>
EHR Concepts in  Top N Documents Concept-SpecificKGs<br>Each Visit<br>Doc 1 : “… [concept_1] …  G [BC] [concept 1 ]<br>[concept_1] [concept_2] … [concept_4] …”<br>⋮ [concept_2] G [BC] [concept 2 ]<br>[[concept_3concept_4]] Doc 2 [concept_5: “…… [] ……”concept_2] …  Entity/Relation Extraction G [BC] [concept 3 ]<br>⋮ [concept_5⋮ ] (query with the set of concepts) Doc N : “… [concept_3⋮ ] …  ⋮ G [BC] [concept 4 ]<br>[concept 4] …” G [BC] [concept 5 ]<br>**----- End of picture text -----**<br>


Figure 5: Our pipeline to construct concept-specific KG _G[BC]_ from biomedical corpus with EHR. 

1. **Concept Set Extraction from EHR Data** : We iterate through the EHR dataset and, for each patient visit, we collect all the involved medical concepts (conditions, procedures, and drugs) into a concept set. This results in a list of concept sets, where each set represents the concepts associated with a specific patient visit. 

2. **Filtering Similar Concept Sets** : To reduce redundancy and computational overhead, we filter out highly similar concept sets based on their concept multi-hot vector representation. We use the CountVectorizer from scikit-learn (Pedregosa et al., 2018) to create a vocabulary of unique concepts and transform each concept set into a multi-hot vector. We then compare the symmetric difference between pairs of concept sets and filter out sets that differ by fewer than a specified similarity threshold (5 in our case). This step helps to eliminate nearly duplicate concept sets while retaining a diverse range of concepts for knowledge graph construction. This process results in 26,134 concept sets in our experiment. 

3. **Dense Retrieval of Relevant PubMed Abstracts** : For each filtered concept set, we retrieve the top _n_ ( _n_ = 3 in our case) most relevant documents from PubMed Abstracts using dense retrieval. Considering the expensive time consumption to retrieve abstracts from the full corpus (with 30 million abstracts), we randomly (seed=42) select 1/10 abstracts as the subset corpus to proceed. We employ the nomic-ai/nomic-embed-text-v1.5 model to embed both the concepts and the PubMed abstracts into a dense vector space (dim=768). We then compute the cosine similarity between the concept set embedding and the abstract embeddings to identify the most relevant documents. The dense retrieval process is optimized for efficiency by processing the embeddings in chunks and utilizing GPU acceleration when available. 

4. **Triple Extraction from PubMed Abstracts** : For each retrieved PubMed abstract, we perform entity extraction and relation extraction to extract knowledge graph triples. We use a large language model (Claude 3.5 Sonnet in our case), to identify relationships between the concepts mentioned in the abstract. The LLM is provided with a prompt that includes the abstract text, the list of relevant concepts, and an example of the desired triple format. The prompt instructs the LLM to extract at most 10 informative and logically sound triples for each abstract, focusing on relationships closely related to the provided concepts. The extracted triples follow the format [ENTITY1, RELATIONSHIP, ENTITY2], where the entities are replaced with the exact concept terms when applicable. The prompt we used for the triple extraction is: 

Given a medical text and a list of important concepts, extract relevant relationships between the concepts from the text (if present). For each triple, if an entity matches one of the given concepts, replace the entity with the exact concept term. 

Focus on generating high-quality triples closely related to the provided concepts. Aim to extract at most 10 triples for each text. Each triple should follow this format: [ENTITY1, RELATIONSHIP, ENTITY2]. Ensure the triples are informative and logically sound. 

Example: 

Text: 

Asthma is a chronic respiratory condition characterized by inflammation and narrowing of the airways, leading to breathing difficulties. Common symptoms include wheezing, coughing, shortness of breath, and chest tightness. Triggers can vary but often include allergens, air pollution, exercise, and respiratory infections. Management typically involves a combination of long-term control medications, such as inhaled corticosteroids, and 

20 

Under review as a conference paper 

quick-relief medications like short-acting beta-agonists. Recent research has focused on personalized treatment approaches, including biologics for severe asthma and the role of the microbiome in asthma development and progression. Proper inhaler technique and adherence to medication regimens are crucial for effective management. Asthma action plans, developed in partnership with healthcare providers, help patients manage symptoms and exacerbations. 

Concepts: 

[asthma, inflammation, airways, wheezing, coughing, inhaled corticosteroids, short-acting beta-agonists, allergens, respiratory infections] 

Extracted triples: 

[[asthma, is a, chronic respiratory condition], [asthma, characterized by, inflammation of airways], [inflammation, causes, narrowing of airways], [narrowing of airways, leads to, breathing difficulties], [wheezing, is a symptom of, asthma], [coughing, is a symptom of, asthma], [allergens, can trigger, asthma], [respiratory infections, can trigger, asthma], [inhaled corticosteroids, used for, long-term control of asthma], [short-acting beta-agonists, provide, quick relief in asthma]] 

Text: _{_ text _}_ Concepts: _{_ concepts _}_ Extracted triples: 

5. **Knowledge Graph Construction** : The extracted triples from each abstract are added to the knowledge graph of the medical concepts mentioned in the document. This process builds a concept-specific knowledge graph _G_[BC] _ci_[for][each][concept] _[c][i]_[,][incorporating][relevant][information] from the PubMed corpus. The resulting knowledge graphs capture the relationships and contextual information associated with each medical concept. 

By following this pipeline, we construct a comprehensive set of concept-specific knowledge graphs that integrate information from both EHR data and the PubMed corpus. These knowledge graphs serve as a valuable resource for EHR-based downstream tasks, such as patient representation learning and predictive modeling. 

The number of resulting KG triples from the PubMed Abstracts is 259,938. 

## B.3 KG EXTRACTION FROM LARGE LANGUAGE MODELS 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_Reasoning-Enhanced_Healthcare_Predictions_with_Knowledge_Graph_Community_Retrieval_images/iclr_2025_Reasoning-Enhanced_Healthcare_Predictions_with_Knowledge_Graph_Community_Retrieval.pdf-0021-11.png)


**----- Start of picture text -----**<br>
Instruction<br>“Please identify the relationships among these medical<br>concepts that can be potentially helpful to clinical predictions  Concept-Specific<br>EHR Concepts in  (e.g., mortality prediction, readmission prediction). You can introduce intermediate relationships with other  Identified Relationships KGs<br>Each Visit entities based on your knowledge. There’s no need to keep all the relationships connected. [[concept_2, …, concept_5], G [LLM] concept 1<br>⋮ [[[concept_1concept_2concept_3]]] Output format:[[ENTITY1, RELATIONSHIP_1, ENTITY2], [ENTITY2, RELATIONSHIP_2, ENTITY3], ...] [[…, …, …],[…, …, …],concept_5, …, …], GG [LLM] concept [LLM] concept 23<br>[[concept_4concept_5]] [… ,…, [concept_4concept_4, …, concept_3], ] G [LLM] concept 4<br>⋮ ⋮ ] G [LLM] [concept 5 ]<br>**----- End of picture text -----**<br>


Figure 6: Our pipeline to construct concept-specific KG with LLM and EHR. 

To extract concept-specific knowledge graphs from large language models (LLMs), we follow a process similar to the initial steps of the “KG extraction from corpus” pipeline (Appendix B.2): 

21 

Under review as a conference paper 

1. **Concept Set Extraction from EHR Data** : We iterate through the EHR dataset and, for each patient visit, we collect all the involved medical concepts (conditions, procedures, and drugs) into a concept set. This results in a list of concept sets, where each set represents the concepts associated with a specific patient visit. 

2. **Filtering Similar Concept Sets** : To reduce redundancy and computational overhead, we filter out highly similar concept sets based on their concept multi-hot vector representation, as described in the previous subsection. This step helps to eliminate nearly duplicate concept sets while retaining a diverse range of concepts for knowledge graph construction. 

3. **Prompting LLMs for Relationship Extraction** : For each filtered concept set, we prompt a large language model (Claude 3.5 Sonnet in our case) to identify relationships among the medical concepts that can be potentially helpful for clinical predictions, such as mortality prediction and readmission prediction. The LLM is encouraged to introduce intermediate relationships with other entities based on its knowledge, and there is no requirement to keep all the relationships connected. The prompt instructs the LLM to use the original names of the provided concepts in the output, which should follow the format [[ENTITY1, RELATIONSHIP ~~1~~ , ENTITY2], [ENTITY2, RELATIONSHIP ~~2~~ , ENTITY3], ...]. The prompt we used for the relationship extraction is: 

Please identify the relationships among these medical concepts that can be potentially helpful to clinical predictions (e.g., mortality prediction, readmission prediction) as many as possible. 

You can introduce intermediate relationships with other entities based on your knowledge. 

Consider how these concepts would interact with others to be useful for clinical predictions. There’s no need to keep all the relationships connected. 

For the concepts provided in the list, you MUST use the their original name without any changes. Please output only the list of triples without any other information. 

Output format: [[ENTITY1, RELATIONSHIP ~~1~~ , ENTITY2], [ENTITY2, RELATIONSHIP ~~2~~ , ENTITY3], ...] 

Medical Concepts: concepts Output: 

4. **Knowledge Graph Construction** : The extracted triples for each concept set are used to construct a concept-specific knowledge graph _G_[LLM] _ci_ for each concept _ci_ . For each concept _ci_ in the concept set, we store the connected 3-hop subgraph sourced from _ci_ to its corresponding concept-specific knowledge graph _G_[LLM] _ci_ . This step ensures that only the triples directly or indirectly connected to the concept _ci_ are included in its concept-specific knowledge graph. This process is iteratively performed for all the concepts in the concept set. The resulting knowledge graphs capture the relationships and contextual information associated with each medical concept based on the knowledge embedded in the large language model. 

By leveraging the knowledge embedded in large language models, this process allows us to construct concept-specific knowledge graphs that incorporate a broad range of information beyond what is explicitly stated in the EHR data or biomedical literature. These knowledge graphs can provide valuable insights and support various downstream tasks in the clinical domain. 

The number of resulting KG triples from the LLM is 315,492. 

22 

Under review as a conference paper 

## C BASELINE IMPLEMENTATIONS 

**ML-based Models.** To ensure a fair performance comparison, we implemented all machine learning (ML)-based Electronic Health Record (EHR) models using PyHealth (Yang et al., 2023a). Since GRAM (Choi et al., 2017) and KerPrint (Yang et al., 2023b) were not yet integrated into PyHealth, we separately implemented these models within the pyhealth.models module. Our implementations were based on the original codebases for GRAM[5] and KerPrint[6] . For consistency across all ML-based models, we set the embedding size to 256. We trained the models using a learning rate of 1e-4 and employed an early stopping mechanism based on validation loss to prevent overfitting. 

**LM+ML Models.** We implement GraphCare (Jiang et al., 2024a) using their official codebase[7] with their default setting for each component. We use text-embedding-3-large (an improved version of text-embedding-ada-002 used in the original implementation) as the embedding model for the embedding initialization, and use their proposed BAT as the base GNN model. We implement RAM-EHR (Xu et al., 2024) using their codebase[8] with the settings mentioned in the implementation details in their paper. We implement EMERGE (Zhu et al., 2024a) fully following the implementation details provided in their paper (with the LLMs Clinical-LongFormer, BGE-M3, Qwen 1.5-7B Chat, and DeepSeek-V2 Chat used for different purposes in the pipeline). 

**LLM-based Methods.** For zero-shot and few-shot prompting-based EHR prediction with the LLM, we utilize the template presented in Table 7 which includes _task description_ , _task-specific instruction_ , _patient base context_ , _supplementary information_ (based on retrieval), and _Patient References_ (similar patients). Unlike the structured format used for reasoning-chain generation, the reasoning here is presented in a free-style manner, which, as our study indicates, results in better performance. 

We implement the EHR-CoAgent (Cui et al., 2024) approach as described in their paper[9] , which combines the strengths of predictive agent reasoning and critical agent instruction to create a accurate few-shot prediction system for our tasks. The implementation consists of two main components: a predictor agent and a critic agent. 

The predictor agent is responsible for generating predictions and providing explanatory reasoning based on the input EHR data. Given a patient’s medical history, the predictor agent analyzes the relevant information and generates the most likely prediction along with a step-by-step explanation of its reasoning process. The prompt used for the predictor agent is as follows: 

Given the following task description, patient EHR context, task instructions, and similar patients, please make a prediction with reasoning. 

# Task # [Task Definition] + [Regulator] # Patient EHR Context # [Patient’s Context (Base)] # Task Instructions (Guidelines) # [Refined Guidelines if iteration _>_ 1 else Initial Guidelines] # Similar Patients # [Top-K similar patients’ contexts] 

Give the prediction and reasoning in the following format: # Reasoning # 

- 5https://github.com/mp2893/gram 

- 6https://github.com/xyxpku/KerPrint 

- 7https://github.com/pat-jj/GraphCare 

- 8https://github.com/ritaranx/RAM-EHR 

9 _Note_ : While EHR-CoAgent was not originally designed for mortality or readmission prediction tasks, we have made minor custom modifications to adapt it to our specific use cases. 

23 

Under review as a conference paper 

[Your reasoning here] 

# Prediction # [Your prediction here (1/0)] 

Output: 

where _Task Definition_ is a brief definition of the task (e.g., Mortality Prediction Task: Objective: Predict the mortality outcome for a patient’s subsequent hospital visit based solely on conditions, procedures, and medications. Labels: 1 = mortality, 0 = survival). We introduce a _Regulator_ to import prior knowledge of the dataset to avoid the LLM to over-focus on improving true positive or true negative. For example, we set “ _**Must to Notice:** Only the patients with extremely very high risk of mortality should be predicted as 1._ ” as the regulator for the mortality prediction task. The existence of the regulator significantly affect the final result for imbalanced datasets, as shown in Table 5. This is because that the instruction-updating approach used by EHR-CoAgent tends to excessively penalize false positives to produce instructions that boost true positives, especially in cases of imbalanced data like mortality prediction in the two datasets. However, this can compromise true negatives and accuracy in such scenarios. 

Table 5: Significant performance difference between EHR-CoAgent w/ regulator and w/o regulator. 

||**MIMIC-III Mortality**<br>**Accuracy**<br>**Macro F1**<br>**Sensitivity**<br>**Specifcity**|**MIMIC-IV Mortality**<br>**Accuracy**<br>**Macro F1**<br>**Sensitivity**<br>**Specifcity**|
|---|---|---|
|EHR-CoAgent w/ Regulator<br>EHR CoAgent w/o Regulator|87.4<br>51.7<br>13.0<br>91.7<br>53.6<br>39.7<br>51.9<br>53.6|81.0<br>55.5<br>13.8<br>97.0<br>72.4<br>61.8<br>51.3<br>77.4|



_Refined Guidelines_ are the instructions refined by the critic agent, while _Initial Guidelines_ are the seed instructions, which is as same as the task descriptions we used for other methods. 

The critic agent, on the other hand, plays a different role in the EHR-CoAgent framework. It observes the predictor agent’s outputs alongside the ground truth labels and identifies error patterns and discrepancies in the predictor’s reasoning process. Based on this analysis, the critic agent refine the instructions to improve the reasoning process of the predictor. The prompt used for the critic agent is as follows: 

You are an assistant who is good at self-reflection, gaining experience, and summarizing criteria. By reflecting on failure predictions that are given below, your task is to reflect on these incorrect predictions, compare them against the ground truth, and formulate criteria and guidelines to enhance the accuracy of future predictions. 

The original instructions are provided under ”# Task Instructions (Guidelines) #”. Your task is to refine the instructions based on the discrepancies between the predictions and the ground truth. 

## # Input Data # 

[input data batch with prompts, predictions, and ground-truth labels] 

## # Instructions # 

1. Please always remember that the predictions above are all incorrect. You should always use the ground truth as the final basis to discover many unreasonable aspects in the predictions and then summarize them into experience and criteria. 

2. Identify why the wrong predictions deviated from the ground truth by examining discrepancies in the medical history analysis. 

3. Determine key and potential influencing factors, reasoning methods, and relevant feature combinations that could better align predictions with the ground truth. 

4. The instructions should be listed in distinct rows, each representing a criteria or guideline. 

5. The instructions should be generalizable to multiple samples, rather than specific to individual samples. 

6. Conduct detailed analysis and write criteria based on the input samples, rather than writing some criteria without foundation. 

24 

Under review as a conference paper 

7. Please note that the criteria you wrote should not include the word ”ground truth”. 

Your output should be the new set of guidelines under ”# Task Instructions (Guidelines) #” that can be used to improve the predictor’s reasoning process. 

Output: 

which is mostly the same as the one in their paper (Cui et al., 2024). 

Our implementation of EHR-CoAgent follows an iterative refinement process, where the predictor agent generates predictions, the critic agent analyzes incorrect predictions and refines the instruction, which is consolidated and integrated into the predictor’s prompts for the next round. The incorrect predictions are divided into batches, and the critic agent refines the instructions for each batch. The instructions from all batches is then consolidated using the LLM to identify the most important and recurring insights across the entire refined instruction list. This consolidated new guidelines are integrated into the predictor’s prompts for the next round, allowing the system to effectively improve its prediction performance. We iterate the process 5 times. 

The consolidated instructions are generated using the following prompt: 

Given the following set of guidelines, please consolidate the insights into a concise and coherent set of guidelines for refining the predictor’s reasoning process. 

# Set of Guidelines # 

[A Batch of Guidelines] 

## # Instructions # 

1. Analyze the provided guidelines and identify common themes, patterns, and key insights. 2. Synthesize the insights into a consolidated set of guidelines that capture the most important and recurring aspects. 

3. Ensure that the consolidated guidelines are clear, concise, and actionable to refine the predictor’s reasoning process. 

4. Create a numbered list of the consolidated guidelines in the same format as the original guidelines. 

## Output: 

The consolidated instructions are then recursively consolidated until the final list size is smaller than 10. This is done to ensure that the consolidated instructions can be effectively integrated into the predictor’s prompts for the next round, considering the limited context window size of the LLM. 

By incorporating the consolidated instructions into the predictor’s prompts, the EHR-CoAgent approach enables an iterative refinement process to improve the accuracy predictions. 

25 

Under review as a conference paper 

## D IMPLEMENTATION DETAILS OF KARE 

## D.1 STEP 1: MEDICAL CONCEPT KNOWLEDGE GRAPH CONSTRUCTION 

## **Step 1.1: Medical Concept-Specific Knowledge Graph Extraction** 

We use UMLS, PubMed Abstracts, and Claude 3.5 Sonnet as the sources for knowledge graph extraction from Biomedical KG, Biomedical Corpus, and Large Language Model, respectively. The extraction details are showcased in Appendix B.1, B.2, and B.3, respectively. 

For UMLS, we utilize the “Full Release” version under “2024AA Full UMLS Release Files”[10] . For dense retrieval from PubMed abstracts, we utilize the local embedding model Nomic (dimension = 768) (Nussbaum et al., 2024). We use Amazon Bedrock[11] to access the Claude model. 

The resulting KG triples from UMLS, PubMed Abstracts, and Sonnet are 29,434, 259,938, and 315,492, respectively. 

## **Step 1.2: Semantic Clustering** 

For semantic clustering of entities and relations in the KG we build above: we (1) first use the textembedding-3-large model (dimension = 1024) from Azure OpenAI to retrieve the text embeddings of entities and relations; and (2) use Scikit-learn (Pedregosa et al., 2018) to perform agglomerative clustering based on those embeddings. The optimal cosine distance thresholds _θe_ and _θr_ are both found to be 0.14, resulting in 513,867 triples in total after clustering. 

## **Step 1.3: Hierarchical KG Community Detection and Indexing** 

We employ Graspy (Chung et al., 2019) to implement the hierarchical Leiden algorithm, setting the maximum size for each top-level community (max ~~c~~ luster ~~s~~ ize) to 5. 

To enhance community diversity, the algorithm is run 25 times with different randomness at each iteration, resulting in unique 59,832 communities (with different combinations of triples) where there are 40,934 communities with the size smaller than 20 ( _Zs_ ), and 57,247 communities with the size smller than 150 ( _Zc_ ). 

Using Claude 3.5 Sonnet as the LLM, we generate 147,264 community summaries (including both general and theme-specific summaries) with the prompts shown in Figure 9 and 10. 

## D.2 STEP 2: PATIENT CONTEXT CONSTRUCTION AND AUGMENTATION 

We use the template as shown in Figure 8 to construct patient’s base context based on their EHR. 

To retrieve the relevant medical knowledge for context augmentation, we set _α_ = 0 _._ 1, _β_ = 0 _._ 7, _λ_ 1 = 0 _._ 2, _λ_ 2 = 0 _._ 2, and _λ_ 3 = 0 _._ 3 as the hyperparameters, each tuned in the range of [0 _,_ 1]. 

D.3 STEP 3: REASONING-ENHANCED PRECISE HEALTHCARE PREDICTION 

## **Step 3.1: Training Sample Generation** 

To generate reasoning chain training samples, we leverage Claude 3.5 Sonnet as our expert LLM. Ensuring EHR data protection and ethical use is paramount; therefore, all LLM interactions are conducted via the Amazon Bedrock platform[12] , a cloud infrastructure that allows us to fully harness LLM capabilities while maintaining strict privacy measures. The maximum output length for Sonnet is set as 4,096 tokens. We use the prompt in Figure 13 for the reasoning chain generation here. 

## **Step 3.2 Multitask-Based Fine-Tuning and Prediction** 

Our fine-tuning framework is implemented using the TRL (von Werra et al., 2020), Transformers (Wolf et al., 2020), and FlashAttention-2 (Dao, 2024) Python libraries. We use Mistral-7B-Instruct- 

> 10https://www.nlm.nih.gov/research/umls/licensedcontent/ umlsknowledgesources.html 

> 11https://docs.aws.amazon.com/bedrock/ 

> 12The use of Amazon Bedrock is authorized by MIMIC: https://physionet.org/news/post/gpt-responsible-use 

26 

Under review as a conference paper 

v0.3 (Jiang et al., 2023) as our local LLM, full-parameter fine-tuned using DeepSpeed (Rasley et al., 2020) with the following configurations: 

|**Parameter**|**Value**|
|---|---|
|model<br>~~n~~ame<br>~~o~~r<br>~~p~~ath<br>torch<br>~~d~~type<br>use<br>~~f~~ash<br>~~a~~ttention<br>~~2~~<br>preprocessing<br>~~n~~um<br>~~w~~orkers<br>bf16<br>gradient<br>~~a~~ccumulation<br>~~s~~teps<br>gradient<br>~~c~~heckpointing<br>learning<br>~~r~~ate<br>max<br>~~s~~eq<br>~~l~~ength<br>num<br>~~t~~rain<br>~~e~~pochs<br>per<br>~~d~~evice<br>~~t~~rain<br>~~b~~atch<br>~~s~~ize<br>lr<br>~~s~~cheduler<br>~~t~~ype<br>warmup<br>~~r~~atio|mistralai/Mistral-7B-Instruct-v0.3<br>bfoat16<br>true<br>12<br>true<br>4<br>true<br>5.0e-06<br>6000<br>3<br>1<br>cosine<br>0.1|



Table 6: LLM Fine-tuning Configuration Parameters 

_Hardware Information_ : The experiments were conducted on a system with an AMD EPYC 7513 32-Core Processor and 1.0 TB of RAM. The setup includes eight NVIDIA A100 80GB PCIe GPUs, each with 81920 MiB of memory, providing a total of 640 GB GPU memory. The system’s root partition has 32 GB of storage. 

Training of each model runs on eight NVIDIA A100 GPUs and typically completes within five hours. After the training process, we select the best performing model checkpoint based on validation loss to perform the prediction. 

27 

Under review as a conference paper 

## E CASE STUDY 

- E.1 REASONING-ENHANCED PREDICTION BY OUR FINE-TUNED MODEL 

## **INPUT (Reasoning Task)** 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_Reasoning-Enhanced_Healthcare_Predictions_with_Knowledge_Graph_Community_Retrieval_images/iclr_2025_Reasoning-Enhanced_Healthcare_Predictions_with_Knowledge_Graph_Community_Retrieval.pdf-0028-04.png)


**----- Start of picture text -----**<br>
[Reasoning]  Given the following task description,<br>patient EHR context, similar patients, and retrieved<br>medical knowledge, Please provide a step-by-step<br>reasoning process that leads to the prediction<br>outcome based on the patient's context and<br>relevant medical knowledge. Procedures: ========================================<br>After the reasoning process, provide the prediction  1. repair of current obstetric laceration # Retrieved Medical Knowledge #<br>label (0/1). 2. other procedures to assist delivery<br>3. artificial rupture of membranes to assist delivery [<br>======================================== 1. “Polyhydramnios, a condition of excess amniotic<br># Task # Medications: fluid, can increase the risk of preterm labor and<br>1. thyroid preparations premature rupture of membranes. This may lead to<br>Readmission Prediction Task: 2. agents for treatment of hemorrhoids and anal  complications after delivery that require close<br>Objective: Predict if the patient will be readmitted  fissures for topical use monitoring and potential readmission.”<br>to the hospital within 15 days of discharge. 3. antipruritics, incl. antihistamines, anesthetics, etc. 2. “Thyroid disorders during pregnancy can impact<br>Labels: 1 = readmission within 15 days, 0 = no  4. antacids both maternal and fetal health. If not properly<br>readmission within 15 days 5. calcium supplements managed, thyroid conditions may result in adverse<br>6. other analgesics and antipyretics in atc outcomes and increase the likelihood of<br>Note: Analyze the information comprehensively to  7. drugs for constipation postpartum complications necessitating<br>determine the likelihood of readmission. The goal is  8. throat preparations readmission.”<br>to accurately distinguish between patients who are  9. antiinflammatory and antirheumatic products,  3. ”The combination of obstetric perineal trauma<br>likely to be readmitted and those who are not. non-steroids repair and the use of topical hemorrhoid treatments<br>======================================== 10. cough suppressants, excl. combinations with expectorants suggests uncomplicated healing, reducing the likelihood of readmission.”<br># Patient EHR Context # 11. opioid analgesics 4. "The use of antiinflammatory and analgesic<br>12. expectorants, excl. combinations with cough  medications postpartum is typical for managing<br>Patient ID: xxxxxxx suppressants pain and discomfort. However, careful monitoring is<br>necessary to ensure proper pain control and avoid<br>Visit 0: Label: adverse effects that could prolong recovery or<br>Conditions: 0 require readmission."<br>1. polyhydramnios and other problems of amniotic  5. "Calcium supplements are often prescribed<br>cavity Patient ID: zzzzzzz during pregnancy and postpartum to support<br>2. other complications of pregnancy maternal bone health. Adequate calcium intake<br>3. thyroid disorders Visit 0: and monitoring can help prevent complications<br>4. ob-related trauma to perineum and vulva Conditions: and reduce the risk of readmission related to<br>5. other pregnancy and delivery including normal 1. polyhydramnios and other problems of amniotic  calcium deficiency or imbalance."<br>cavity 6. " The use of bacterial and viral vaccines<br>Procedures: 2. ob-related trauma to perineum and vulva combined with thyroid preparations indicates<br>1. repair of current obstetric laceration 3. other complications of pregnancy proactive management of the patient's thyroid<br>2. other procedures to assist delivery 4. thyroid disorders condition and immunization status, supporting a<br>5. other complications of birth; puerperium affecting  lower risk of readmission.<br>Medications: management of mother 7. "Hemorrhoids and anal fissures are common<br>1. throat preparations 6. other pregnancy and delivery including normal during pregnancy and after vaginal delivery. The<br>2. antiinflammatory and antirheumatic products,  use of topical agents for treatment suggests active<br>non-steroids Procedures: management of these conditions, which may help<br>3. other analgesics and antipyretics in atc 1. repair of current obstetric laceration prevent potential complications and reduce the risk<br>4. agents for treatment of hemorrhoids and anal  2. other procedures to assist delivery of readmission."<br>fissures for topical use 8. "Individuals taking thyroid preparations are likely<br>5. antipruritics, incl. antihistamines, anesthetics, etc. Medications: managing a pre-existing or pregnancy-related<br>6. drugs for constipation 1. thyroid preparations thyroid condition. Proper thyroid hormone<br>7. bacterial and viral vaccines, combined 2. cough suppressants, excl. combinations with  regulation is crucial in the postpartum period.<br>8. antacids expectorants Imbalances in thyroid hormones may lead to<br>9. calcium supplements 3. drugs for constipation complications requiring hospital readmission if not<br>10. thyroid preparations 4. bacterial and viral vaccines, combined closely monitored and adjusted”<br>======================================== 5. opioid analgesics6. other analgesics and antipyretics in atc 9. "Constipation is a frequent issue postpartum, and the use of drugs for constipation indicates active<br># Similar Patients # 7. agents for treatment of hemorrhoids and anal  management. Adequate bowel function and<br>fissures for topical use prevention of severe constipation can help avoid<br>Patient ID: yyyyyyy 8. antipruritics, incl. antihistamines, anesthetics, etc. complications and reduce the likelihood of<br>9. antacids readmission."<br>Visit 0: 10. calcium supplements 10. “Extensive lacerations or complications from<br>Conditions: 11. throat preparations assisted delivery may require additional care and<br>1. polyhydramnios and other problems of amniotic  12. antiinflammatory and antirheumatic products,  monitoring, potentially leading to readmission.”<br>cavity non-steroids ]<br>2. ob-related trauma to perineum and vulva 13. urologicals<br>3. other complications of pregnancy<br>4. thyroid disorders Label:<br>5. other pregnancy and delivery including normal 1<br>**----- End of picture text -----**<br>


## **OUTPUT** 

**# Reasoning#** 1. ****Patient Overview**** : The patient has a history of polyhydramnios, other complications of pregnancy, thyroid disorders, and ob-related trauma to the perineum and vulva. The patient underwent repair of current obstetric laceration and other procedures to assist delivery. Postpartum medications include throat preparations, antiinflammatory and antirheumatic products, analgesics, antipyretics, agents for hemorrhoids and anal fissures, antipruritics, drugs for constipation, bacterial and viral vaccines, antacids, calcium supplements, and thyroid preparations. 

2. ****Relevant Retrieved Medical Knowledge**** : - Polyhydramnios can increase the risk of preterm labor and premature rupture of membranes, potentially leading to complications after delivery that require close monitoring and readmission. - Thyroid disorders during pregnancy, if not properly managed, may result in adverse outcomes and increase the likelihood of postpartum complications necessitating readmission. - Extensive lacerations or complications from assisted delivery may require additional care and monitoring, potentially leading to readmission. - Proper thyroid hormone regulation is crucial postpartum, and imbalances may lead to complications requiring readmission if not closely monitored and adjusted. 3. ****Comparison with Similar Patients**** : 

The patient shares several similarities with the two similar patients, including polyhydramnios, ob-related trauma to perineum and vulva, other complications of pregnancy, thyroid disorders, repair of current obstetric laceration, other procedures to assist delivery, and similar postpartum medications. However, the patient has an additional condition of "other complications of pregnancy" compared to the second similar patient. The first similar patient, who was readmitted within 15 days, also had this condition, along with "other complications of birth; puerperium affecting management of mother," which the current patient does not have. 

4. ****Reasoning Towards Prediction**** : 

The patient's history of polyhydramnios, thyroid disorders, and other complications of pregnancy, combined with the need for repair of obstetric laceration and assisted delivery procedures, suggests a higher risk of postpartum complications that may require readmission. The patient's similarity to the first patient, who was readmitted within 15 days and had the additional condition of "other complications of pregnancy," further supports this increased risk. While the patient is receiving appropriate postpartum medications and preventive measures, the presence of multiple risk factors and the need for close monitoring of thyroid function and potential complications from assisted delivery indicate a substantial likelihood of readmission within 15 days. 

5. ****Conclusion**** : Based on the patient's history of polyhydramnios, thyroid disorders, other complications of pregnancy, and the need for repair of obstetric laceration and assisted delivery procedures, along with the similarity to a patient who was readmitted within 15 days, there is a high likelihood that this patient will be readmitted to the hospital within 15 days of discharge. **# Prediction #** 1 

Figure 7: Case Study of the Fine-tuned KARE Model. 

28 

Under review as a conference paper 

## E.2 ZERO-SHOT & FEW-SHOT COMPARISON 

Table 7: Case Study of zero-shot and few-shot EHR prediction with LLM (Sonnet-3.5). Ground Truth: The patient will die in the next visit (Prediction = **1** ). For the ethic concern, the patients involved are all from synthetic MIMIC-III by HALO (Theodorou et al., 2023). 

|**Case**|**Description**|
|---|---|
|Input Prompt with EHR|Given the following task description and patient context, please make a prediction with reasoning based on the|
||patient’s context.|
||**Task**: Mortality Prediction Task|
||**Objective**: Predict the mortality outcome for a patient’s subsequent hospital visit based solely on conditions,|
||procedures, and medications.|
||**Labels**: 1 = mortality, 0 = survival|
||**Key Considerations**:|
||Conditions:|
||• Severity of diagnosed conditions (e.g., advanced cancer, severe heart failure, sepsis)|
||• Presence of multiple comorbidities|
||• Acute vs. chronic nature of conditions|
||Procedures:|
||• Invasiveness and complexity of recent procedures|
||• Emergency vs. elective procedures|
||• Frequency of life-sustaining procedures (e.g., dialysis, mechanical ventilation)|
||Medications:|
||• Use of high-risk medications (e.g., chemotherapy drugs, immunosuppressants)|
||• Multiple medication use indicating complex health issues|
||• Presence of medications typically used in end-of-life care|
||Note: Focus on combinations of conditions, procedures, and medications that indicate critical illness or a high|
||risk of mortality. Consider how these factors interact and potentially exacerbate each other. Only the patients with|
||extremely very high risk of mortality should be predicted as 1.|



**Patient Context** : Patient ID: 29488 Visit 0: Conditions: - Deficiency and other anemia - Essential hypertension - Complication of device; implant or graft - Congestive heart failure; nonhypertensive - Cancer of prostate - Anxiety disorders - Thyroid disorders - Disorders of lipid metabolism - Conduction disorders - Mycoses - Other diseases of kidney and ureters - Cancer of esophagus Procedures: - Diagnostic cardiac catheterization; coronary arteriography - Other or procedures on vessels other than head and neck - Colorectal resection Drugs: - Other drugs for obstructive airway diseases, inhalants in ATC - Lipid modifying agents, plain - Antithrombotic agents - Angiotensin II receptor blockers (ARBs), plain Visit 1: Conditions: - Congestive heart failure; nonhypertensive - Cardiac dysrhythmias - Shock - Alcohol-related disorders - Diabetes mellitus with complications - E codes: adverse effects of medical care - Disorders of teeth and jaw Procedures: - Other vascular catheterization; not heart - Blood transfusion Drugs: - Other drugs for obstructive airway diseases, inhalants in ATC - Other analgesics and antipyretics in ATC - Drugs for peptic ulcer and gastro-oesophageal reflux disease (GORD) - Beta blocking agents - Potassium supplements 

Continued on next page 

29 

Under review as a conference paper 

|||**Table 7 – continued frompreviouspage**|
|---|---|---|
|**Case**|**Description**||
||**Supplementary Information**: [PLACEHOLDER]||
||**Patient References**: [PLACEHOLDER]||
||**Format**:||
||# Reasoning|#|
||[Your reasoning here]||
||# Prediction #||
||[Your prediction here (1/0)]||
|Case (1): Zero-Shot|As provided, Sonnet 3.5 gave the following incorrect prediction:||
|Prompting with EHR Base|**Model Output**:||
|Context Only|# Reasoning|#|
||While this patient’s condition is serious and deteriorating, there are several factors to consider:||
||**Severity of conditions:**||
||•|The patient has multiple severe conditions, including congestive heart failure, cancer (prostate and|
|||esophagus), and cardiac issues.|
||•|New complications have arisen in visit 1, such as shock and cardiac dysrhythmias, which are|
|||concerning.|
||•|However, some of the most immediately life-threatening conditions (e.g., sepsis, advanced stage|
|||cancer, multi-organ failure) are not explicitly mentioned.|
||**Procedures:**||
||•|The procedures in visit 1 (blood transfusion and vascular catheterization) suggest a decline in health,|
|||but they are also interventions aimed at stabilizing the patient.|
||•|There’s no mention of extreme life-sustaining procedures like mechanical ventilation or dialysis.|
||**Medications:**||
||•|The medications listed are primarily for managing chronic conditions rather than end-of-life care.|
||•|There’s no mention of high-risk medications like chemotherapy drugs or strong opioids for palliative|
|||care.|
||**Disease progression:**||
||•|While there’s a clear decline between visits 0 and 1, the conditions in visit 1, while serious, are|
|||potentially manageable with appropriate medical intervention.|
||**Absence of certain high-mortality indicators:**||
||•|There’s no mention of conditions like sepsis, acute respiratory failure, or advanced stage metastatic|
|||cancer, which would signifcantly increase mortality risk.|
||Given the task description’s note to only predict mortality (1) for patients with ”extremely very high risk of||
||mortality”, this patient’s condition, while serious, does not meet that extremely high threshold. The presence of||
||active interventions (blood transfusion, medications for managing conditions) suggests ongoing treatment rather||
||than purely palliative care.||
||# Prediction #||
||**0**(incorrect)||
|Case (2): Zero-Shot|**Supplementary Information**:||
|Prompting with Classic<br>RAG-Augmented Context<br>(Knowledge Source:<br>PubMed Abstract)|•|Rank 1, Similarity: 0.7204270362854004 PMID: 314160 Title: Resection of posterior ventricular<br>aneurysm, replacement of mitral valve, and coronary bypass. Authors: C M Chavez, G Rodriquez, J<br>H Conn Abstract: A 58-year-old man had sudden and progressive heart failure after a severe<br>myocardial infarction. Aggressive medical treatment consisting of diuretics, vasopressors, and|
|||digitalis failed to improve his condition signifcantly. Cardiac catheterization disclosed a critical|
|||stenosis in the left anterior descending branch of the left coronary artery, a large posterior left|
|||ventricul aneurysm, and severe mitral insuffciency. Intermittent third degree heart block developed|
|||after admission. Surgical correction resulted in a dramatic recovery, and three years after operation|
|||he is fully recovered and asymptomatic. Publication Year: 1979|
||•|Rank 2, Similarity: 0.7123403549194336 PMID: 804881 Title: [Acute coronary thrombosis in a 28|
|||year-old woman]. Authors: G Mott´e, M Vogel, G Coatantiec, P Mariette Abstract: A 28-year-old|
|||woman, with no previous cardiovascular history, was hospitalized for myocardial infarction|
|||complicated by bifascicular block followed by complete atrio-ventricular block with a regressive|
|||course. A coronary arteriography performed on the 10th day demonstrated a thrombosis of the|
|||anterior interventricular artery, the rest of the coronary network being normal. The infuence of a|
|||dyslipidaemia and the taking of oral contraceptives was discussed as an aetiology. Publication Year:|
|||1975|



Continued on next page 

30 

Under review as a conference paper 

**Table 7 – continued from previous page** 

|**Case**|**Description**||
|---|---|---|
||•|Rank 3, Similarity: 0.7102002501487732 PMID: 42254 Title: [Non arrhythmogenic sudden death as|
|||complication of coronary heart disease]. Authors: H Zilcher, D Glogar Abstract: In a cohort of 417|
|||patients admitted consecutively to the Coronary Care Unit for acute myocardial ischemia (unstable|
|||angina pectoris in 121, acute myocardial infarction in 296 patients) 21 cases of non arrhythmogenic|
|||sudden death occurred within 24 hours after admission. 16 of these patients suffered from acute|
|||myocardial infarction and 5 from unstable angina pectoris. Cause of death was cardiac rupture in 12|
|||and pump failure in 4 patients with acute myocardial infarction, whereas all patients with unstable|
|||angina pectoris died from pump failure. Patients with cardiac rupture within 24 hours after|
|||admission, had signifcantly higher systolic and diastolic blood pressure in comparison with the|
|||other groups and with patients dying from cardiac rupture on the third day, or later. All patients|
|||dying from pump failure with unstable angina pectoris and one of the patients dying from pump|
|||failure with acute myocardial infarction had beta blocker therapy. ... Publication Year: 1979|
||•|(...)|
||•|Rank 9, Similarity: 0.6921100616455078 PMID: 443259 Title: Chest pain, shock, arrhythmias and|
|||death in a young woman. Authors: Abstract: 9 fgures form the core of this article describing and|
|||discussing a case of sudden death, 2 hours after a 30-year old woman presented at a hospital|
|||emergency with chest pains. She had taken no medications other than oral contraceptives (OCs) for|
|||10 years. The patient was admitted to the coronary care unit where fndings included a palpable|
|||blood pressure of 94 mm of Hg, a heart rate of 128/min, and a respiratory rate of 28/minute.|
|||Cyanosis was noted, jugular veins were distended, and there were rales over the lung bases|
|||bilaterally; cardiac sounds were soft and a third heart sound was audible. Arterial oxygen tension|
|||was 15 mm of Hg, and carbon dioxide tension was 42 mm of Hg; pH was 7.2. Ventricular|
|||tachycardia developed and ventricular fbrillation ensued. The patient was intubated and well|
|||oxygenated, external cardiac compression was performed, sodium bicarbonate, epinephrine, and|
|||calcium were administered, and electrical defbrillation was performed. After several attempts, the|
|||latter resulted in a slow idioventricular rhythm on the electrocardiogram, but neither the blood|
|||pressure nor pulse was detectable. ... Publication Year: 1979|
||•|Rank 10, Similarity: 0.6920691132545471 PMID: 938220 Title: Left ventricular function during|
|||aortic surgery. Authors: R M Carroll, R B Laravuso, J F Schauble Abstract: Fourteen patients|
|||undergoing surgery for aneurysm or occlusive disease of the abdominal aorta were studied. Thirteen|
|||patients had a history of hypertension or myocardial infarction; two patients had chronic obstructive|
|||pulmonary disease. Tachycardia, hypertension, and elevated pulmonary artery occluded. (PAo)|
|||pressure occurred in response to laryngoscopy and intubation in two patients; elevation of PAo|
|||pressure in response to aortic cross-clamping occurred in two patients. In three of these four|
|||patients, electrocardiographic evidence of myocardial ischemia appeared. These events are|
|||important in a consideration of the occurence of myocardial infarction in patients undergoing|
|||abdominal aortic surgery. Satisfactory treatment of myocardial ischemia has been accomplished|
|||with the use of propranolol hydrochloride and sodium nitroprusside. Publication Year: 1976|
||**Model Output**:||
||# Reasoning|#|
||Based on the patient’s context and the given task description, I will analyze the key factors to predict mortality||
||risk for the subsequent hospital visit:||
||**Conditions**:||
||•|The patient has multiple serious conditions, including congestive heart failure (nonhypertensive),|
|||cardiac dysrhythmias, and shock. These are severe cardiovascular issues that signifcantly increase|
|||mortality risk.|
||•|The presence of diabetes mellitus with complications adds to the overall complexity of the patient’s|
|||health status.|
||•|Alcohol-related disorders may complicate treatment and recovery.|
||**Procedures**:||
||•|The patient underwent blood transfusion, which might indicate severe blood loss or anemia,|
|||potentially linked to the shock condition.|
||•|Other vascular catheterization (not heart) was performed, possibly related to managing circulatory|
|||issues.|
||•|The supplementary materials discuss these procedures but focus on their use in orthopedic surgeries.|
||**Medications**:||
||•|The patient is on multiple medications, including beta-blocking agents and drugs for obstructive|
|||airway diseases, indicating complex health issues.|
||•|Potassium supplements might be linked to managing electrolyte imbalances, possibly due to heart|
|||failure or other medications.|
||**Progression**|**from previous visit**:|
||•|The patient’s condition appears to have worsened since the previous visit, with new serious|
|||conditions like shock and cardiac dysrhythmias appearing.|
||**Comorbidities**:||
||•|The combination of cardiovascular issues, diabetes complications, and alcohol-related disorders|
|||creates a complex clinical picture with multiple interacting risk factors.|



Continued on next page 

31 

Under review as a conference paper 

**Table 7 – continued from previous page** 

|||**Table 7 – continued frompreviouspage**|
|---|---|---|
|**Case**|**Description**||
||However, there are some factors that suggest the patient may not be at the highest risk of mortality:||
||- The patient is receiving active treatment, including medications for their conditions.||
||- There’s no mention of advanced cancer or other terminal illnesses.||
||- The procedures mentioned are not highly invasive or indicative of end-stage disease.||
||- There’s no indication of mechanical ventilation or other life-sustaining procedures.||
||# Prediction|#|
||**0**(incorrect)||
|Case (3): Zero-Shot|**Supplementary Information**:||
|Prompting with<br>KARE-Augmented Context<br>(Knowledge Source: KG|•|Patients with congestive heart failure combined with cardiac dysrhythmias and shock have a<br>signifcantly increased risk of mortality due to impaired cardiac output and hemodynamic instability.|
|Communities)|•|The presence of esophageal cancer, especially when co-occurring with prostate cancer, indicates a|
|||higher burden of malignancy, which can exacerbate overall health decline.|
||•|Complications of devices, implants, or grafts can lead to severe infections or organ dysfunction,|
|||which is life-threatening, especially in patients with weakened cardiac function.|
||•|The combination of anemia and thyroid disorders can exacerbate cardiovascular conditions, leading|
|||to poorer outcomes in heart failure patients.|
||•|Alcohol-related disorders can negatively impact medication adherence and exacerbate conditions|
|||like hypertension and diabetes, increasing the risk of death.|
||•|The use of blood transfusions and vascular catheterization procedures indicates severe clinical|
|||interventions often linked to critical conditions and a higher risk of death.|
||•|Polypharmacy, including beta-blockers, potassium supplements, and medications for obstructive|
|||airway diseases, refects complex medical management, which can be associated with increased risk|
|||of adverse events.|
||•|Patients with diabetes mellitus with complications face a greater risk of serious outcomes due to|
|||potential organ damage and delayed healing processes.|
||**Model Output**:||
||# Reasoning|#|
||Based on the patient’s context and the supplementary information:||
||**Severity of Conditions**:||
||•|The patient has congestive heart failure, cardiac dysrhythmias, and shock, which together|
|||signifcantly impair cardiac function and increase mortality risk.|
||•|The presence of two cancers (esophageal and prostate) indicates a high burden of malignancy,|
|||worsening the patient’s overall prognosis.|
||•|Complications of devices, implants, or grafts may lead to severe infections or organ dysfunction.|
||**Interactions Between Conditions**:||
||•|Anemia and thyroid disorders can worsen heart failure outcomes.|
||•|Alcohol-related disorders may impair medication adherence and liver function, complicating|
|||management of other conditions.|
||•|Diabetes mellitus with complications further increases mortality risk due to potential organ damage.|
||**Procedures**:||
||•|The need for blood transfusions and vascular catheterization suggests critical interventions due to|
|||severe anemia or hemodynamic instability.|
||•|These procedures are associated with higher mortality rates in critically ill patients.|
||**Medications**:||
||•|Polypharmacy with beta-blockers, potassium supplements, and other medications indicates complex|
|||management and high risk of adverse events.|
||•|Medications for obstructive airway diseases suggest respiratory complications.|
||The provided information highlights that this combination of severe conditions, multiple comorbidities, and||
||invasive procedures is associated with a very high risk of mortality.||
||Given the critical combination of severe cardiac conditions, multiple cancers, complex comorbidities, and the||
||necessity for|invasive procedures, the patient is at an extremely high risk of mortality and is likely to die in the|
||next visit.||
||# Prediction|#|
||**1**(correct)||



Continued on next page 

32 

Under review as a conference paper 

**Table 7 – continued from previous page** 

|||**Table 7 – continued frompreviouspage**|
|---|---|---|
|**Case**|**Description**||
|Case (4): Few-Shot|**Supplementary Information**: (Same as in Case 3)||
|Prompting with Positive and|**Patient References**:||
|Negative Patient References<br>and KARE-Augmented|•|**Example 1 (Mortality = 1)**:|
|Context||**– Patient ID**: 10269|
|||**– Visit Information**:|
|||* **Conditions**:|
|||· Congestive heart failure; nonhypertensive|
|||· Sepsis|
|||· Acute respiratory failure|
|||· Shock|
|||* **Procedures**:|
|||· Mechanical ventilation|
|||· Blood transfusion|
|||· Vascular catheterization|
|||* **Medications**:|
|||· High-dose vasopressors|
|||· Broad-spectrum antibiotics|
||•|**Example 2 (Mortality = 0)**:|
|||**– Patient ID**: 11445|
|||**– Visit Information**:|
|||* **Conditions**:|
|||· Essential hypertension|
|||· Type 2 diabetes mellitus without complications|
|||· Hyperlipidemia|
|||* **Procedures**:|
|||· Routine blood tests|
|||· Electrocardiogram|
|||* **Medications**:|
|||· Metformin|
|||· ACE inhibitors|
||**Model Output**:||
||# Reasoning|#|
||Comparing the new patient to the provided references:||
||**Similarity to Example 1:**||
||•|Both patients have congestive heart failure and shock.|
||•|Both require invasive procedures like blood transfusions and vascular catheterization.|
||•|The patient 10269 has acute respiratory failure and sepsis, while the new patient has multiple|
|||cancers and cardiac dysrhythmias.|
||•|Both have complex comorbidities and are undergoing critical interventions, indicating a high risk of|
|||mortality.|
||**Difference from Example 2:**||
||•|The negative reference has manageable chronic conditions without severe complications.|
||•|No invasive procedures or critical conditions are present.|
||•|The new patient has signifcantly more severe conditions and interventions.|
||**Patient’s Conditions:**||
||•|The combination of congestive heart failure, cardiac dysrhythmias, shock, and multiple cancers|
|||places the patient at a very high risk of mortality.|
||•|Disorders linked to alcohol use may reduce compliance with prescribed medications and worsen|
|||chronic illnesses such as high blood pressure and diabetes, elevating mortality risk.|
||•|Diabetes mellitus with complications and alcohol-related disorders further exacerbate health risks.|
||**Procedures**|**and Medications:**|
||•|Invasive procedures and polypharmacy indicate critical illness and complex management.|
||•|Similar to the positive reference, these factors contribute to a higher mortality risk.|
||The provided knowledge and patient references highlight that patients with this combination of severe conditions||
||and interventions are likely to die. Given the strong similarities to the positive reference and the critical nature of||
||the patient’s|health status, it is highly likely that the patient will die in the next visit.|
||# Prediction|#|
||**1**(correct)||
|Analysis|-**Case (2)**demonstrates that when using traditional RAG methods with sparse retrieval from PubMed abstracts,||
||the supplementary information contains several keywords matching the patient’s conditions but fails to provide||
||relevant information regarding the patient’s high mortality risk.||
|||Continued on next page|



33 

Under review as a conference paper 

**Table 7 – continued from previous page** 

||**Table 7 – continued frompreviouspage**|
|---|---|
|**Case**|**Description**|
||-**Case (3)**shows that KARE’s dense and relevant knowledge retrieval provides comprehensive insights into the|
||patient’s condition, interactions between comorbidities, and the implications of procedures and medications,|
||leading to a correct prediction.|
||-**Case (4)**illustrates that including patient references with KARE-augmented context further aids the model by|
||providing concrete examples to compare against, enhancing reasoning and ensuring a correct prediction.|
|Conclusion|This case study emphasizes the importance of KARE’s approach in providing dense, relevant knowledge that|
||signifcantly improves the model’s ability to make accurate predictions. By integrating comprehensive knowledge|
||graphs and considering complex interactions between medical concepts, KARE enhances reasoning capabilities|
||beyond what is achievable with base context or traditional RAG methods.|



## F TEMPLATES, PROMPTS, AND EXAMPLES 

|`========================================`|`========================================`|`========================================`|
|---|---|---|
|**`# Task #`**||**`# Similar Patients #`**|
|`Readmission Prediction Task:`||`Patient ID: yyyyyyyy`|
|`Objective: Predict if the patient`|`will be`||
|`readmitted to the hospital within`|`15 days of`|`Visit 0:`|
|`discharge.`||`Conditions:`|
|`Labels: 1 = readmission within 15`|`days, 0 = no`|`1. pleurisy; pneumothorax; pulmonary collapse`|
|`readmission within 15 days`||`2. cardiac dysrhythmias`|
|||`3. congestive heart failure; nonhypertensive`|
|`Note: Analyze the information comprehensively to`||`4. other liver diseases`|
|`determine the likelihood of readmission. The goal`||⋮|
|`is to accurately distinguish between patients who`<br>`are likely to be readmitted and those who are not`||`Procedures:`<br>`1. incision of pleura; thoracentesis; chest`|
|||`drainage`|
|||`2. other or heart procedures`|
|`========================================`<br>**`# Patient EHR Context #`**||`3. other diagnostic cardiovascular procedures`|
|`Patient ID: xxxxxxxx`||`Medications:`<br>`1. antithrombotic agents`|
|`Visit 0:`<br>`Conditions:`<br>`1. pleurisy; pneumothorax; pulmonary collapse`<br>`2. pneumonia`<br>`3. congestive heart failure; nonhypertensive`||`2. other analgesics and antipyretics in atc`<br>`3. high-ceiling diuretics`<br>`4. antacids`<br>`5. drugs for constipation`<br>⋮|
|`4. nutritional deficiencies`||`Label:`|
|`5. other circulatory disease`||`1`|
|`6. cardiac dysrhythmias`|||
|`7. screening and history of mental health and`||`-----------------------------------------`|
|`substance abuse codes`||`Patient ID: zzzzzzzz`|
|`8. heart valve disorders`|||
|⋮<br>`Procedures:`<br>`1. incision of pleura; thoracentesis; chest`<br>`drainage`||`Visit 0:`<br>`Conditions:`<br>`1. pleurisy; pneumothorax; pulmonary collapse`<br>`2. congestive heart failure; nonhypertensive`<br>`3. complications of surgical procedures or`|
|`Medications:`<br>`1. high-ceiling diuretics`<br>`2. adrenergics, inhalants`<br>`3. potassium supplements`<br>`4. other drugs for obstructive airway diseases,`||`medical care`<br>`4. screening and history of mental health and`<br>`substance abuse codes`<br>`5. cardiac dysrhythmias`<br>⋮|
|`inhalants in atc`||`Procedures:`|
|`5. beta blocking agents`||`1. other diagnostic procedures of respiratory`|
|`6. viral vaccines`||`tract and mediastinum`|
|`7. other analgesics and antipyretics in atc`||`2. other non-or therapeutic procedures on`|
|`8. drugs for constipation`||`respiratory system`|
|`9. antithrombotic agents`||`3. incision of pleura; thoracentesis; chest`|
|`10. opioid analgesics`||`drainage`|
|`11. antiemetics and antinauseants`|||
|`12. other mineral supplements in`|`atc`|`Medications:`|
|`13. i.v. solution additives`||`1. other agents acting on the renin-`|
|`14. other diagnostic agents in atc`||`angiotensin system in atc`|
|`15. antiarrhythmics, class i and`|`iii`|`2. antiadrenergic agents, peripherally acting`|
|⋮||⋮|
||(continue)|`Label:`<br>`0`|



Figure 8: An example of the patient base context. 

34 

Under review as a conference paper 

**General Summarization Theme-Specific Summarization** You are a knowledgeable medical assistant tasked with You are a knowledgeable medical assistant tasked with generating a comprehensive summary of the medical generating a theme-specific summary of the medical concepts and relationships provided below. concepts and relationships provided below. Given a list of medical triples in the format (entity1, Given a list of medical triples in the format (entity1, relationship, entity2), please create a single, coherent relationship, entity2) and a specific theme, please create a summary that captures the key medical knowledge summary that focuses on how the knowledge represented represented by these concepts and their relationships. The by these triples is relevant to the given theme. The summary summary should be written in the third person and include should highlight the key concepts, relationships, and all the entity names for full context. implications that are most pertinent to the theme. Focus on how these medical concepts are interconnected The summary should be written in the third person and and their relevance to medical understanding or patient include all the relevant entity names for context. Please care. If there are any contradictions in the triples, please provide only the summary without any starting words or resolve them in the summary. phrases, and without mentioning the individual triples. Please provide only the summary without any starting words Examples: or phrases, and without mentioning the individual triples. Mortality Prediction: The summary should be an integrated representation of the Triples: medical knowledge contained in the triples. - (Diabetes, is a risk factor for, Cardiovascular Disease) - (Hypertension, is associated with, Diabetes) Example: - (Obesity, contributes to, Diabetes) Triples: - (Diabetes, is a risk factor for, Cardiovascular Disease) Theme: Mortality prediction - (Hypertension, is associated with, Diabetes) **{Theme Description}** - (Obesity, contributes to, Diabetes) Summary: Summary: Diabetes, hypertension, and obesity are significant risk Diabetes, hypertension, and obesity are closely factors that can increase the likelihood of mortality. interconnected medical conditions. Diabetes is a significant Diabetes is directly associated with an increased risk of risk factor for developing cardiovascular disease. cardiovascular disease, which is a leading cause of death. Hypertension, or high blood pressure, is often associated Hypertension and obesity, which often co-occur with with diabetes. Obesity contributes to the development of diabetes, further compound these risks. Patients with these diabetes, as excess body weight can lead to insulin conditions require close monitoring and aggressive resistance. Managing these conditions together is crucial for management to mitigate the risk of mortality. reducing the risk of serious complications and improving overall patient care. Triples: Triples: **{Formatted Triples} {Formatted Triples}** Theme: **{Theme}** Summary: Summary: 

Figure 9: The prompts for ( _left_ ) general and ( _right_ ) theme-specific KG community summarization. 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_Reasoning-Enhanced_Healthcare_Predictions_with_Knowledge_Graph_Community_Retrieval_images/iclr_2025_Reasoning-Enhanced_Healthcare_Predictions_with_Knowledge_Graph_Community_Retrieval.pdf-0035-03.png)


**----- Start of picture text -----**<br>
Summary Combination for Large Community<br>You are a knowledgeable medical assistant tasked with generating a comprehensive summary of the medical concepts<br>and relationships based on the individual summaries provided below.<br>Given the following summaries of a community, please create a single, coherent summary that captures the key medical<br>knowledge represented by this community. The combined summary should include all the important entities and<br>relationships mentioned in the individual summaries.<br>Focus on the main concepts, their relationships, and their relevance to medical understanding or patient care. The summary<br>should be concise yet informative, providing a comprehensive overview of the community's medical knowledge.<br>Please ensure that the combined summary is written in the third person and includes all the relevant entity names for full<br>context.<br>Please provide only the summary without any starting words or phrases.<br>Summaries:<br>{summaries}<br>Combined Summary:<br>**----- End of picture text -----**<br>


Figure 10: The prompt for summary combination for large communities. 

35 

Under review as a conference paper 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_Reasoning-Enhanced_Healthcare_Predictions_with_Knowledge_Graph_Community_Retrieval_images/iclr_2025_Reasoning-Enhanced_Healthcare_Predictions_with_Knowledge_Graph_Community_Retrieval.pdf-0036-01.png)


**----- Start of picture text -----**<br>
KG Community  Theme-Specific Summarization<br>Hypertension Diabetes Theme: Mortality Theme: Readmission<br>“Diabetes and Hypertension are major risk  “Chronic Kidney Disease (CKD), commonly caused by<br>factors for Chronic Kidney Disease (CKD). As  Diabetes and Hypertension, can lead to Anemia,<br>Chronic Kidney  CKD progresses, it can lead to complications  characterized by a decreased number of red blood cells.<br>Disease such as Anemia, which can further increase  Anemia may cause symptoms such as fatigue and<br>morbidity and mortality. Early detection and  shortness of breath, increasing the risk of hospital<br>management of Diabetes and Hypertension are  readmissions. To prevent readmissions in CKD patients,<br>crucial in preventing the development and  it is essential to manage the underlying causes, monitor<br>SupplementationIron  Anemia progression of CKD, ultimately reducing the risk of premature death in these patient populations.” hemoglobin levels, treat Anemia with Iron Supplementation, and ensure treatment adherence.”<br>treat<br>cause cause<br>leads to<br>**----- End of picture text -----**<br>


Figure 11: An example of theme-specific summarization of knowledge graph community. 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_Reasoning-Enhanced_Healthcare_Predictions_with_Knowledge_Graph_Community_Retrieval_images/iclr_2025_Reasoning-Enhanced_Healthcare_Predictions_with_Knowledge_Graph_Community_Retrieval.pdf-0036-03.png)


**----- Start of picture text -----**<br>
Task Description – Mortality Prediction<br>Mortality Prediction Task:<br>Objective: Predict the mortality outcome for a patient's subsequent hospital visit based solely on<br>conditions, procedures, and medications.<br>Labels: 1 = mortality, 0 = survival<br>Key Considerations:<br>1. Conditions:<br>- Severity of diagnosed conditions (e.g., advanced cancer, severe heart failure, sepsis)<br>- Presence of multiple comorbidities<br>- Acute vs. chronic nature of conditions “Key Considerations” section is<br>2. Procedures: only for zero/few-shot prompting &<br>- Invasiveness and complexity of recent procedures  training sample generation<br>- Emergency vs. elective procedures<br>- Frequency of life-sustaining procedures (e.g., dialysis, mechanical ventilation)<br>3. Medications:<br>- Use of high-risk medications (e.g., chemotherapy drugs, immunosuppressants)<br>- Multiple medication use indicating complex health issues<br>- Presence of medications typically used in end-of-life care<br>Note: Focus on combinations of conditions, procedures, and medications that indicate critical illness or a<br>high risk of mortality. Consider how these factors interact and potentially exacerbate each other. Only the<br>patients with extremely very high risk of mortality should be predicted as 1.<br>Task Description – Readmission Prediction<br>Readmission Prediction Task:<br>Objective: Predict if the patient will be readmitted to the hospital within 15 days of discharge based solely<br>on conditions, procedures, and medications.<br>Labels: 1 = readmission within 15 days, 0 = no readmission within 15 days<br>Key Considerations:<br>1. Conditions: “Key Considerations” section is<br>- Chronic diseases with high risk of exacerbation only for zero/few-shot prompting &<br>- Conditions requiring close monitoring or frequent adjustments training sample generation<br>- Recent acute conditions with potential for complications<br>2. Procedures:<br>- Recent major surgeries or interventions with high complication rates<br>- Procedures that require extensive follow-up care<br>- Incomplete or partially successful procedures<br>3. Medications:<br>- New medication regimens that may require adjustment<br>- Medications with narrow therapeutic windows or high risk of side effects<br>- Complex medication schedules that may lead to adherence issues<br>Note: Analyze the information comprehensively to determine the likelihood of readmission. The goal is to<br>accurately distinguish between patients who are likely to be readmitted and those who are not.<br>**----- End of picture text -----**<br>


Figure 12: Task descriptions of the mortality prediction and the readmission prediction tasks used in the paper. _Note_ : For the fine-tuning process, we do not include the ”Key Considerations” section in the input template. 

36 

Under review as a conference paper 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_Reasoning-Enhanced_Healthcare_Predictions_with_Knowledge_Graph_Community_Retrieval_images/iclr_2025_Reasoning-Enhanced_Healthcare_Predictions_with_Knowledge_Graph_Community_Retrieval.pdf-0037-01.png)


**----- Start of picture text -----**<br>
Given the following task description, patient EHR context, similar patients, retrieved medical knowledge,<br>and ground truth label, provide a step-by-step reasoning process that leads to the correct prediction:<br>========================================<br># Task #<br>{task_description}<br>========================================<br># Patient EHR Context #<br>{context}<br>========================================<br># Similar Patients #<br>{similar_patients}<br>========================================<br># Retrieved Medical Knowledge #<br>{medical_knowledge}<br>========================================<br># Ground Truth #<br>{ground_truth}<br>========================================<br>Please provide a step-by-step reasoning process that leads to the correct prediction based on the<br>patient's context, similar patients, and the retrieved relevant medical knowledge.<br>The reasoning chain should follow this structured format:<br>**----- End of picture text -----**<br>



![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_Reasoning-Enhanced_Healthcare_Predictions_with_Knowledge_Graph_Community_Retrieval_images/iclr_2025_Reasoning-Enhanced_Healthcare_Predictions_with_Knowledge_Graph_Community_Retrieval.pdf-0037-02.png)


**----- Start of picture text -----**<br>
1.  Patient Overview : Check the key information in the patient's context, with the Key Considerations from<br>the task description in mind.<br>2.  Relevant Retrieved Medical Knowledge : Highlight the retrieved medical knowledge pertinent to the<br>patient's condition.<br>3.  Comparison with Similar Patients : Analyze the similarities and differences between the patient and similar<br>patients, explaining how these factors influence the prediction.<br>4.  Reasoning Towards Prediction : Integrate the above information to logically reason towards the<br>predicted outcome.<br>5.  Conclusion : Summarize the reasoning and state the prediction without mentioning the ground truth.<br>The reasoning should be comprehensive, medically sound, and clearly explain how the patient's<br>information leads to the predicted outcome.<br>Important Notes:<br>- Do not mention the ground truth label in the reasoning process.<br>- Use the relevant knowledge as needed.<br>- Analyze the similarities and differences between the patient and similar patients to justify the prediction.<br>After generating the reasoning chain, please review it and indicate your confidence in the reasoning<br>chain at the end.<br>Options of confidence: [Very Confident, Confident, Neutral, Not Confident, Very Not Confident.]<br>Output Format:<br># Reasoning Chain #<br>1. Patient Overview:<br>[YOUR OUTPUT]<br>2. Relevant Retrieved Medical Knowledge:<br>[YOUR OUTPUT]<br>3. Comparison with Similar Patients:<br>[YOUR OUTPUT]<br>4. Reasoning Towards Prediction:<br>[YOUR OUTPUT]<br>5. Conclusion:<br>[YOUR OUTPUT]<br># Confidence #<br>[CONFIDENCE ("Very Confident", "Confident", "Neutral", "Not Confident", "Very Not Confident")]<br>**----- End of picture text -----**<br>


Figure 13: Prompt used for reasoning chain generation for training sample. 

37 

Under review as a conference paper 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_Reasoning-Enhanced_Healthcare_Predictions_with_Knowledge_Graph_Community_Retrieval_images/iclr_2025_Reasoning-Enhanced_Healthcare_Predictions_with_Knowledge_Graph_Community_Retrieval.pdf-0038-01.png)


**----- Start of picture text -----**<br>
Fine-Tuning Input (Reasoning) Fine-Tuning Input (Label Prediction)<br>[Reasoning]  Given the following task description, patient EHR  [Label Prediction]  Given the following task description, patient<br>context, similar patients, and retrieved medical knowledge,  EHR context, similar patients, and retrieved medical knowledge,<br>please provide a step-by-step reasoning process that leads to  please directly predict the label (0/1)<br>the prediction outcome based on the patient's context and<br>relevant medical knowledge. After the reasoning process,  ========================================<br>provide the prediction label (0/1). # Task #<br>{task_description}<br>======================================== ========================================<br># Task # # Patient EHR Context #<br>{task_description} {context}<br>======================================== ========================================<br># Patient EHR Context # # Similar Patients #<br>{context} {similar_patients}<br>======================================== ========================================<br># Similar Patients # # Retrieved Medical Knowledge #<br>{similar_patients} {medical_knowledge}<br>======================================== ========================================<br># Retrieved Medical Knowledge #<br>{medical_knowledge}<br>========================================<br>Fine-Tuning Output (Reasoning) Fine-Tuning Output (Label Prediction)<br># Reasoning # {Label (0/1)}<br>{reasoning}<br># Prediction #<br>{Label (0/1)}<br>**----- End of picture text -----**<br>


Figure 14: Template used for the input and output of fine-tuning. (To improve the reproducibility of KARE, we will publicize the processed data for fine-tuning the local LLM through PhysioNet) 

38 

Under review as a conference paper 

## G NOTATIONS 

Table 8: Notations used in our paper. 

||Table 8: Notations used in ourpaper.|
|---|---|
|**Notation**|**Description**|
|_G′, V ′, R′, E′_|Knowledge graph before semantic clustering and its components|
|_G, V, R, E_|Refned knowledge graph after semantic clustering and its components|
|_Gci, Vci, Eci_|Concept-specifc knowledge graph for concept_ci_ and its components|
|_G_KG<br>_ci_<br>_G_BC<br>_ci_<br>_G_LLM<br>_ci_|Subgraph of_Gci_ from a biomedical knowledge graph<br>Subgraph of_Gci_ extracted from a biomedical corpus<br>Subgraph of_Gci_ extracted using a large language model|
|_Gp, VGp_|Patient-specifc knowledge graph for patient_p_and its entities|
|_Cm_<br>_l , Cl_<br>_k,m_|Set of communities and the_k_-th community at level_l_in run_m_|
|_C, Ck, C_best|Set of all communities, a community, and the best community|
|_VCk, SCk_|Entities and summary of community_Ck_|
|_Bp, Ap_|Base and augmented context for patient_p_|
|_Sp_|Selected community summaries for patient_p_|
|**e**_i,_**e**_j, e_(_·_)|Text embedding of entity_i_/ relation_j_ and embedding function|
|_H_(_v_)|Hit count of node_v_in previous selections|
|_τ, Tτ_|Healthcare prediction task (theme) and its representative terms|
|_y∗_<br>_p,τ, yp,τ_<br>_ρp,τ,k, ρ_best<br>_p,τ_|Ground truth and predicted labels for patient_p_and task_τ_<br>The_k_-th and best reasoning chains for patient_p_and task_τ_|
|_ci,_**C**|A medical concept and the set of all concepts|
|_Rci_|Top_X_ co-existing concepts for concept_ci_|
|_ϕe, ϕr_|Mappings from original entities and relations to cluster representatives|
|_pij_|Shortest path between concepts_ci_ and_cj_|
|_L, M_|Number of levels and runs in community detection|
|_Zc, Zs_|Maximum triples per community, triples per summary|
|_θe, θr_|Clustering thresholds for entities and relations|
|_α, β, λ_1_, λ_2_, λ_3|Hyperparameters for context augmentation|



39 

