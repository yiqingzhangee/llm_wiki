# KNOWLEDGE GRAPH BASED AGENT FOR COMPLEX, KNOWLEDGE-INTENSIVE QA IN MEDICINE 

**Xiaorui Su** _[∗]_ **Yibo Wang** _[†]_ **Shanghua Gao** _[∗]_ **Xiaolong Liu** _[†]_ **Valentina Giunchiglia** _[∗][,][‡]_ **Djork-Arn´e Clevert** _[§]_ **Marinka Zitnik** _[∗]_ 

> _∗_ Harvard University _†_ University of Illinois Chicago _‡_ Imperial College London _§_ Pfizer 

## ABSTRACT 

Biomedical knowledge is uniquely complex and structured, requiring distinct reasoning strategies compared to other scientific disciplines like physics or chemistry. Biomedical scientists do not rely on a single approach to reasoning; instead, they use various strategies, including rule-based, prototype-based, and casebased reasoning. This diversity calls for flexible approaches that accommodate multiple reasoning strategies while leveraging in-domain knowledge. We introduce KGAREVION, a knowledge graph (KG) based agent designed to address the complexity of knowledge-intensive medical queries. Upon receiving a query, KGAREVION generates relevant triplets by using the knowledge base of the LLM. These triplets are then verified against a grounded KG to filter out erroneous information and ensure that only accurate, relevant data contribute to the final answer. Unlike RAG-based models, this multi-step process ensures robustness in reasoning while adapting to different models of medical reasoning. Evaluations on four gold-standard medical QA datasets show that KGAREVION improves accuracy by over 5.2%, outperforming 15 models in handling complex medical questions. To test its capabilities, we curated three new medical QA datasets with varying levels of semantic complexity, where KGAREVION achieved a 10.4% improvement in accuracy. 

## 1 INTRODUCTION 

Medical reasoning involves making diagnostic and therapeutic decisions while also understanding the pathology of diseases (Patel et al., 2005). Unlike many other scientific domains, medical reasoning often relies on vertical reasoning, using analogy more heavily (Patel et al., 2005). For instance, in biomedical research, an organism such as Drosophila is used as an exemplar to model a disease mechanism, which is then applied by analogy to other organisms, including humans. In clinical practice, the patient serves as an exemplar, with generalizations drawn from many overlapping disease models and similar patient populations (Charles et al., 1997; Menche et al., 2015). In contrast, fields like physics and chemistry tend to be horizontally organized, where general principles are applied to specific cases (Blois, 1988). This distinction highlights the unique challenges that medical reasoning poses for question-answering (QA) models. 

While large language models (LLMs) (OpenAI, 2024; Dubey et al., 2024; Gao et al., 2024) have demonstrated strong general capabilities, their responses to medical questions often suffer from incorrect retrieval, missing key information, and misalignment with current scientific and medical knowledge. Additionally, they can struggle to provide contextually relevant answers that account for specific local contexts, such as patient demographics or geography, as well as specific areas of biology (Harris, 2023). A major issue lies in these models’ inability to systematically integrate different types of evidence. Specifically, they have difficulty combining scientific factual ( _structured, codified_ ) knowledge derived from formal, rigorous research with tacit ( _noncodified_ ) knowledge—expertise and lessons learned—which is crucial for contextualizing and interpreting scientific evidence in relation to the specific modifying factors of a given medical question (Harris, 2023). 

LLM-powered QA models often lack such _multi-source_ and _grounded knowledge_ necessary for medical reasoning, which requires understanding the nuanced and specialized nature of medical concepts. Additionally, LLMs trained on general knowledge may struggle to solve medical prob- 

1 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_Knowledge_Graph_Based_Agent_For_Complex,_Knowledge-Intensive_QA_in_Medicine_images/iclr_2025_Knowledge_Graph_Based_Agent_For_Complex,_Knowledge-Intensive_QA_in_Medicine.pdf-0002-00.png)


**----- Start of picture text -----**<br>
a b Basic Intermediate Expert<br>0.6<br>Basic Which pharmacological agents  Is there an interaction between the<br>are associated with genes or  Which gene/protein is not  Heat Shock Protein 70 family that<br>0.5 ExpertIntermidate proteins that function in relation to the rough endoplasmic reticulum?A) Zinc cation  V expressed in both the renal tissues and the medial segment of the globus pallidus?A) MYOT  V acts as a molecular chaperone and the gene or protein implicated in Retinitis Pigmentosa 59 due to DHDDS mutation?V<br>B) Cupric chloride  B) MYF5  A) HSPA4<br>C) Cupric oxide  C) MYOG  B) HSPA8<br>0.4 D) Copper D) MYOD1 V C)D) HSPA1B HSPA1A<br>KG D KG A KG A<br>0.3<br>agentKG  D agentKG  C agentKG  B<br>0.2<br>LLM D LLM B LLM C<br>LLaMA3-8B LLaMA3.1-8B GPT-4-Turbo<br>Accuracy<br>**----- End of picture text -----**<br>


Figure 1: **a)** Performance of LLMs on three new datasets introduced in this paper with questions of varying difficulty. **b)** Sample questions from new MedDDx-Basic, MedDDx-Intermediate, and MedDDx-Expert datasets. 

lems that demand _specialized in-domain knowledge_ . This shortcoming arises from their inability to discern subtle, granular differences that are critical in medical contexts. As a result, LLMs face challenges in complex medical reasoning because such reasoning requires both: 1) simultaneous consideration of dependencies across multiple medical concepts within an input question, and 2) precise, local in-domain knowledge of semantically similar concepts that can carry different medical meanings, as we demonstrate in Fig. 1. 

The prevailing strategy to address these challenges is the use of information retrieval techniques, such as retrieval-augmented generation (RAG), which follows a Retrieve-then-Answer paradigm (Xiong et al., 2024b; Shi et al., 2024). Although these methods can provide multi-source knowledge from external databases (Fan et al., 2024), the accuracy of the generated answers depends heavily on the quality of the retrieved information, making them vulnerable to potential errors (Karpukhin et al., 2020). Data repositories and knowledge bases these models draw from contain incomplete or incorrect information, leading to inaccurate retrieval Adlakha et al. (2024); Thakur et al. (2023). Further, many RAG-based methods lack post-retrieval verification mechanisms to validate that retrieved information is factually correct and does not miss key information (Zhao et al., 2023). Knowledge graphs (KGs) of medical concepts have been widely adopted as a grounded knowledge base to provide precise and specialized in-domain knowledge for medical QA models (Qi et al., 2024; Murali et al., 2023; Chandak et al., 2023; Jiang et al.). While KGs can enhance the performance of these models, they are often incomplete. Consequently, approaches that retrieve medical concepts from a KG based solely on the presence of direct associations (edges) between concepts are insufficient. For instance, concepts representing two proteins with distinct biological roles may not be directly connected in the KG, even though these proteins share similar biological representations (Menche et al., 2015). To advance LLM-powered models for knowledge-intensive medical QA, it is essential to develop models that can (1) consider complex associations between several medical concepts at the same time, (2) systematically integrate multi-source knowledge, and (3) effectively verify and ground the retrieved information to ensure contextual relevance and accuracy. 

**Present work.** We introduce KGAREVION, a KG-based LLM agent for complex medical QA that leverages non-codified knowledge of LLMs and structured, codified knowledge of medical concepts within KGs. KGAREVION operates through four key actions, Fig. 2. First, KGAREVION prompts the LLM to generate relevant triplets based on the input question. To ensure the accuracy of the generated triples and to fully use structured KG, KGAREVION fine-tunes the LLM on a KG completion task by incorporating pre-trained structural embeddings of triplets as prefix tokens. The fine-tuned model is then used to evaluate the correctness of generated triplets. Following this, KGAREVION executes a ‘Revise’ action to correct any erroneous triplets, ultimately identifying the correct answer based on the verified triplets. Given the complexity of medical reasoning, KGAREVION agent adaptively selects the most appropriate reasoning approach for each input question, enabling a more nuanced and contextualized QA. This flexibility allows KGAREVION to tackle both _multi-choice_ and _open-ended_ QA. Our key contributions of KGAREVION are: _⃝_ 1 We develop KGAREVION, a versatile KG-agent that dynamically adjusts reasoning strategy, leading to 6.75% improvement when compared to 15 models on seven datasets, including three new challenging datasets; _⃝_ 2 Results on multiple KGs show that grounding through generated triplets can improve KGAREVION ’s capabilities; _⃝_ 3 Results on both multiple-choice and open-ended setups show that KGAREVION can effectively handle complex, knowledge-intensive medical QA. 

2 

## 2 RELATED WORK 

**LLM-based reasoning.** General-purpose LLMs (GPT (OpenAI, 2024), LLaMA family (Dubey et al., 2024; Touvron et al., 2023), Mistral (Jiang et al., 2023)), and LLMs fine-tuned on biomedical data (BioMedLM (Venigalla et al., 2022), Codex (Li´evin et al., 2024), MedAlpaca (Han et al., 2023), Med-PaLM (Singhal et al., 2023), PMC-LLaMA (Wu et al., 2024a)) are used for medical reasoning by leveraging their vast embedded knowledge. Other models utilize the open-ended reasoning capabilities of LLMs to break down queries into sub-tasks, arriving at the final answer step by step, such as Chain-of-Though (CoT) (Wei et al., 2024), CODEX COT (Gramopadhye et al., 2024). However, these methods often struggle with knowledge-intensive medical queries requiring multi-sources an specific knowledge. 

**RAG-based models.** Self-RAG (Asai et al., 2024) is a pioneering framework that enhances LLM performance through retrieval and self-reflection. LLM-AMT (Wang et al., 2023b) improves medical question answering by integrating authoritative medical textbooks into large language models with specialized knowledge retrieval and self-refinement techniques. Adaptive-RAG (Jeong et al., 2024) introduces a dynamic RAG framework that adapts retrieval strategies based on question complexity. However, its accuracy is constrained by the quality of retrieved knowledge (Zhang et al., 2024). 

**KG-based models.** Before the rise of LLMs, several models, such as QAGNN (Yasunaga et al., 2021), JointLK (Sun et al., 2022), and Dragon (Yasunaga et al., 2022), were developed to tackle medical queries solely using KGs in an end-to-end manner. However, these methods cannot be easily applied to questions involving unseen nodes or incomplete knowledge within the graphs. In addition, KGs, with their structured and reliable information, have driven research toward RAG models based on graph data, motivating models like GraphRAG (Edge et al., 2024), KG-RAG (Soman et al., 2023), and MedGraphRAG (Wu et al., 2024b). To improve retrieval accuracy, KG-Rank (Yang et al., 2024) is introduced to rank retrieved triplets and filter out irrelevant knowledge. Additionally, GenGround (Shi et al., 2024) uses a Generate-then-Ground pipeline that grounds answers by prompting LLMs to validate retrieved knowledge. However, all these approaches rely heavily on semantic dependencies, overlooking the rich structural information within KGs. 

## 3 APPROACH 

Given is a set of medical questions _Q_ , each question comprising the question stem _q_ , and a set of candidate answers _C_ . For example, the sample question in Fig. 2b has a stem _q_ = “ _Is there an interaction between the Heat Shock Protein 70 family that acts as a molecular chaperone and the gene or protein implicated in Retinitis Pigmentosa 59 due to DHDDS mutation?_ ” along with a set of semantically related candidate answers _C_ = _{_ HSPA4 _,_ HSPA8 _,_ HSPA1B _,_ HSPA1A _}_ . The goal is to identify the correct answer _a ∈ C_ using an LLM (denoted as _P_ ) and a KG (denoted as _G_ ). Here, a KG is given as a set of triplets _G_ = _{_ ( _h, r, t_ ) _}_ , where each triplet consists of a head entity, a relationship, and a tail entity. Full notation is listed in Table D.2. Note that in addition to this multi-choice setting, we consider open-ended reasoning as well (see Results). 

To address this problem, we propose developing an LLM-powered agent framework (Wu et al., 2023; Li et al., 2023) that leverages various actions (Schick et al., 2023; Shen et al., 2023; Nakano et al., 2021) to collaboratively perform complex tasks (Tang et al., 2023; Bran et al., 2023; Boiko et al., 2023). Fig. 2 shows an overview of KGAREVION, which comprises four key actions, including Generate (§3.1), Review (§3.2), Revise (§3.3), and Answer (§3.3) actions. The Generate action is responsible for generating triplets related to the input question. The Review action then assesses the correctness of each generated triplet, while the Revise action corrects any triplet identified as being incorrect. Finally, the Answer action outputs the final answer based on the triplets identified as correct by the Review action. 

## 3.1 GENERATE ACTION 

The Generate action aims to gather comprehensive structured knowledge from input questions. Specifically, this action first identifies all medical concepts involved in the input question stem _q_ and then generates a set of triplets _T_ related to the question based on the extracted medical concepts. 

3 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_Knowledge_Graph_Based_Agent_For_Complex,_Knowledge-Intensive_QA_in_Medicine_images/iclr_2025_Knowledge_Graph_Based_Agent_For_Complex,_Knowledge-Intensive_QA_in_Medicine.pdf-0004-00.png)


**----- Start of picture text -----**<br>
a Is there an interaction between the Heat Shock Protein 70 family that acts as a molecular  b Review (HSPA8, interacts, DHDDS)<br>chaperone and the gene or protein implicated in<br>Retinitis Pigmentosa 59 due to DHDDS mutation? Aligned embedding<br>A: HSPA4  B: HSPA8  C: HSPA1B  D: HSPA1A LLM<br>Knowledge graph Description dictionary FFN<br>HSPA8<br>KGARevion<br>Description : HSPA1B interacts with DHDDS,  Norm<br>indicating a general interaction that may involve<br>Generate binding, modulation, or other forms of molecular<br>communication.<br>(HSPA8, interacts, DHDDS)<br>(Heat shock proteins, play a role, retinitis pigmentosa) Pre-trained structural  Description tokens Matmul<br>(HSPA1A, interactions, DHDDS)  ... embedding<br>Review Alignment between structural and concept embeddings Softmax<br>(HSPA8, interacts, DHDDS) Matmul<br>(Heat shock proteins, play a role,  Instruction:  Given a triple from a knowledge graph.<br>retinitis pigmentosa)(HSPA1A, interactions, DHDDS) ... Each triple consists of a head entity, a relation, and a tail entity.  the triple and response True or False.Please determine the correctness of  Projection Token embedding<br>Revise LLM<br>(HSPA1B, interactions, DHDDS) ... embeddingAligned  Instruction tokens ...<br>Pre-trained  Description tokens<br>LLM embedding<br>Answer<br>(HSPA8, interacts, DHDDS)(Heat shock proteins, play a role,  Loss Frozen Trainable<br>retinitis pigmentosa) ...<br>No triplet<br>...<br>...<br>**----- End of picture text -----**<br>


Figure 2: **a)** The overview of KGAREVION. **b** ) The architecture of fine-tuning stage in the Review action, where embeddings get from KGs are structural embeddings, while concept embeddings from LLMs. 

Depending on the content of each answer candidate _a ∈ C_ , the input questions can be broadly categorized into two types: choice-aware and non-choice-aware. The answer candidates in the choice-aware group have specific contents, whereas the ones in the non-choice-aware group only contain yes-or-no options. These different types of questions require distinct reasoning processes: choice-aware questions involve analyzing the content of each answer candidate, while non-choiceaware questions only require focusing on the question stem. 

To handle this, the Generate action is designed to prompt the LLM (Ouyang et al., 2022; Wang et al., 2023a) to follow different procedures for generating relevant triplets according to the type of input question. 

- For choice-aware questions, the Generate action generates triplets based on the contents of each answer candidate and extracted medical concepts in question stem _q_ ; 

- For non-choice-aware questions, the Generation action directly generates triplets based on medical concepts presented in question stem _q_ . 

The rationale behind this design is that LLMs have inherent biases in their knowledge, often generating more detailed information on familiar topics compared to less familiar ones when all answer candidates are presented simultaneously (Dai et al., 2024). Additionally, this approach helps reduce the impact of the order in which the answer candidates are presented. The process of the Generate action can be formulated as: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_Knowledge_Graph_Based_Agent_For_Complex,_Knowledge-Intensive_QA_in_Medicine_images/iclr_2025_Knowledge_Graph_Based_Agent_For_Complex,_Knowledge-Intensive_QA_in_Medicine.pdf-0004-07.png)


where the LLM is prompted to extracts triplets from the medical concepts involved in input question stem _q_ . 

## 3.2 REVIEW ACTION 

To enable LLMs to accurately judge the correctness of generated triplets, beyond relying solely on semantic dependencies inferred by LLMs (Shinn et al., 2023), the Review action also leverages the relationships among various medical concepts contained in KGs. This is achieved by fine-tuning the LLM on a KG completion task, explicitly integrating entity structural embeddings learned from KGs into the LLM. Once fine-tuned, the Review action utilizes the model to assess the correctness of generated triplets, as shown in Fig. 2. 

4 

To enable the LLM to capture the structural information embedded in KGs, we employ TransE (Bordes et al., 2013) to learn structural embeddings for both entities and relations in _G_ . These obtained embeddings are fixed when fine-tuning the LLM on knowledge completion tasks. 

**Fine-tuning stage.** Given a triplet ( _h, r, t_ ) _∈ G_ and its pre-trained embedding **e** _h ∈_ R _[d]_ , **e** _r ∈_ R _[d]_ , and **e** _t ∈_ R _[d]_ , where _d_ is the dimension of the embedding, the aim of this stage is to learn a function _· f_ ( ) that makes the LLM capable of determining if a given triplet is True or False by considering its structural embeddings in the KG, as follows: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_Knowledge_Graph_Based_Agent_For_Complex,_Knowledge-Intensive_QA_in_Medicine_images/iclr_2025_Knowledge_Graph_Based_Agent_For_Complex,_Knowledge-Intensive_QA_in_Medicine.pdf-0005-02.png)


where _b_ is a bool value (True or False) and _P_ ( _·_ ) denotes the LLM. 

However, it is complex for LLMs to directly understand structural embeddings given that the embeddings in LLMs are obtained based on their vocabularies (Radford et al., 2019). To make sure that LLMs can understand structural embeddings, we generate a description for each triplet based on the relationship between the two entities. Then, we tokenize the generated descriptions (Touvron et al., 2023) and obtain the corresponding token embeddings. The token embeddings are aligned with the pre-trained structural embeddings of the triplets to produce new embeddings for each input triplet. 

In detail, given the input triplet ( _h, r, t_ ), we denote its description as _D_ ( _r_ ), where _D_ represents the description dictionary with the key as relation and the value as a pre-defined description template, as shown in Appendix 8. Assuming the token embedding of _D_ ( _r_ ) obtained from LLM _P_ ( _·_ ) is **X** _∈_ R _[|][l][|×][d][p]_ , where _|l|_ denotes the number of max tokens and _dp_ denotes the dimension of the embeddings in _P_ ( _·_ ), we first adopt a linear layer ( _g_ ( _·_ ) : R _[d] →_ R _[d][p]_ ) to map the dimension of pre-trained embeddings of entities and relations to the same dimension of that of token embeddings. Then, we concatenate them to create the triplet embedding matrix **V** = [ _g_ ( **e** _h_ ); _g_ ( **e** _r_ ); _g_ ( **e** ) _t_ )] _∈_ R[3] _[×][d][p]_ . Afterwards, we adopt an attention block (Vaswani, 2017), followed by a two-layer feedforward neural network (denoted as FFN in Fig. 2b) to obtain the aligned triplet embedding matrix **Z** _∈_ R[3] _[×][d][p]_ , as follows: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_Knowledge_Graph_Based_Agent_For_Complex,_Knowledge-Intensive_QA_in_Medicine_images/iclr_2025_Knowledge_Graph_Based_Agent_For_Complex,_Knowledge-Intensive_QA_in_Medicine.pdf-0005-06.png)


where _σ_ ( _·_ ) denotes the Softmax function, _φ_ ( _·_ ) denotes the layer normalization function, **W** 1 _∈_ R _[d][p][×][d][h]_ and **W2** _∈_ R _[d][h][×][d][p]_ denotes the trainable parameters in the two-layer FFN, and _dh_ denotes the dimension of hidden layer in the FFN. 

The obtained aligned triplet embedding matrix is input into the LLM in the form of three prefix tokens together with an instruction _s_ to fine-tune the LLM to execute the knowledge graph completion task. Then, the Eq. 2 could be rewritten by: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_Knowledge_Graph_Based_Agent_For_Complex,_Knowledge-Intensive_QA_in_Medicine_images/iclr_2025_Knowledge_Graph_Based_Agent_For_Complex,_Knowledge-Intensive_QA_in_Medicine.pdf-0005-09.png)


During the fine-tuning stage, the pre-trained entity and relation embeddings are frozen, and LoRA (Hu et al., 2022) is adopted to fine-tune the LLM. The trainable parameters are optimized using the next token prediction loss (Radford, 2018). 

**Inference stage.** After fine-tuning, the model is used to evaluate the triplets in _T_ derived from the Generate action (3.1). In detail, we first adopt UMLS code (Bodenreider, 2004) to map entities in each triplet ( _h, r, t_ ) _∈ T_ to KGs to get the corresponding pre-trained entity and relation embeddings, denoted as **e** _h,_ **e** _r,_ **e** _t_ . Then, the triplets are input into the fine-tuned LLM to determine if they are correct or not by Eq. 5. 

However, not all entities in generated triplet ( _h, r, t_ ) _∈ T_ can be mapped to those in KGs. To address this, the Review action applies a soft constraint rule to distinguish whether the generated triplet is factually incorrect or the result of incomplete knowledge in KGs, as follows: 

- Factually Wrong: if we can map _h_ and _t_ to entities in KGs and _b_ = False, then the triplet ( _h, r, t_ ) is factually wrong and is removed from _T_ . 

- Incomplete Knowledge: if we cannot map either _h_ or _t_ to entities in KGs, then the triplet ( _h, r, t_ ) is considered incomplete knowledge and is kept. 

In this way, the triplet in _T_ can be grouped into two categories, i.e., the True triplet set _V_ and False triplet set _F_ , where _T_ = _V ∪ F_ and _V ∩ F_ = _∅_ . 

5 

## 3.3 REVISE AND ANSWER ACTIONS 

If _F_ has triplets, KGAREVION calls the Revise action to adjust the triplets in _F_ to include more knowledge that helps with the answering of the input question. The head and tail entities of the revised triplets are then reviewed by the Review action to make sure that they are correct and related to the input question. If the Review action outputs “True”, then the revised triplets are added to the set of True triplets _V_ . Otherwise, KGAREVION continues to call the Revise action until the max round _k_ ( _k ≥_ 1) is achieved. 

After obtaining the set of True triplets from the Review or Revise actions, KGAREVION finally calls the Answer action to prompt the LLM to select the most suitable answer _y_ from the set of answer candidates _C_ of input question _q_ based on the triplets in _V_ , where _y_ = _P_ ( _q, V, C_ ) and _y ∈ C_ . 

## 4 RESULTS 

**Datasets.** We first start with four multi-choice medical QA benchmarks (Xiong et al., 2024a) (Table 1). In addition, we introduce a new benchmark for multi-choice complex medical QA focused on differential diagnosis (DDx), named MedDDx. We begin by collecting questions and corresponding answers from STaRK-Prime (Wu et al., 2024c). For each question, we then select the top three entities with the highest semantic similarity to serve as additional answer candidates. MedDDx comprises a total of 1,769 multi-choice QA samples. Based on the standard deviation of semantic similarity between answer candidates and the correct answer, we categorize the dataset into three difficulty levels: MedDDx-Basic, MedDDx-Intermediate, and MedDDxExpert (The samples in each dataset are shown in Fig. 2b, and details are available in Appendix 5). 

**Baselines.** We consider 8 LLM-based reasoning Datasets Size QA OR models, 4 RAG-based models, and 3 KG-based MMLU-Med 1,089 A/B/C/D ✔ models. The LLM-based reasoning models inMedQA-USPubMedQA* 1,273500 Yes/No/MaybeA/B/C/D ✔✔ clude LLaMA (2-7B/13B, 3-8B, 3.1-8B) (Touvron BioASQ-Y/N 618 Yes/No ✔ et al., 2023; Dubey et al., 2024), Mistral (Jiang MedDDx-Basic 483 A/B/C/D ✔ et al., 2023), MedAlpaca (7B) (Han et al., 2023), MedDDx-Intermediate 1,041 A/B/C/D ✔ PMC-LLaMA (7B) (Wu et al., 2024a), LLaMA3MedDDx-Expert 245 A/B/C/D ✔ OpenBioLLM-8B (Ankit Pal, 2024), and MEDITRON (Chen et al., 2023). The RAG-based modTable 1: QA benchmarks and three new Medels include Self-RAG (Asai et al., 2024), MedRAG DDx datasets. ‘OR’ indicates whether the (Xiong et al., 2024b), KG-RAG (Soman et al., open-ended reasoning evaluation is done. 2023), and KG-Rank (Yang et al., 2024). The KG-based models include QAGNN (Yasunaga et al., 2021), JointLK (Sun et al., 2022), and Dragon (Yasunaga et al., 2022). 

**Evaluation setup.** We consider two evaluation settings. **Multi-choice reasoning:** This setting evaluates the model’s performances on all collected multi-choice QA datasets. The model is tasked to select the correct answer to a user input question from a set of candidate answers. **Open-ended reasoning:** All candidate answers are masked, meaning that the model has to generate a response to the input question independently without being presented with a set of candidate answers. The model produces an answer solely on its own generated response. Additionally, we design two new evaluation scenarios for each setting to test model abilities in solving complex medical questions by considering the number of medical concepts and the semantic similarity among answer candidates. **Query complexity scenario (QSS):** This is a hard evaluation scenario to test how the model performs with the increase of the number of medical concepts present in a question since the question often becomes more intricate, requiring more nuanced inferences between concepts to achieve the correct answer, with the increase of medical concepts. **Semantic complexity scenario (CSS):** This is a harder evaluation scenario that tests the model’s ability to identify the correct answer among semantically similar and closely medically related candidate answers. 

## 4.1 BENCHMARKING KGAREVION UNDER MULTI-CHOICE REASONING SETTING 

Table 2 shows the accuracy and variance of KGAREVION and all baselines on all datasets. Evaluation on four gold standard medical QA datasets shows that KGAREVION improves the average accuracy by over 4.8%, outperforming all baselines in handling medical queries. 

6 

|**Multi-choice Reasoning**|Established Medical QA Benchmarks|MedDDx|
|---|---|---|
|Method|MMLU-Med<br>MedQA-US<br>PubMedQA*<br>BioASQ-Y/N|Basic<br>Intermidate<br>Expert|
|Metrics|Acc. (std)<br>Acc. (std)<br>Acc. (std)<br>Acc. (std)|Acc. (std)<br>Acc. (std)<br>Acc. (std)|
|LLaMA2-7B|0.376(.006)<br>0.281(.004)<br>0.448(.010)<br>0.568(.006)|0.215(.030)<br>0.198(.004)<br>0.192(.012)|
|LLaMA2-7B (CoT)|0.318(.005)<br>0.251(.002)<br>0.465(.011)<br>0.547(.011)|0.289(.010)<br>0.265(.006)<br>0.229(.023)|
|Mistral-7B|0.634(.004)<br>0.477(.007)<br>0.400(.002)<br>0.644(.001)|0.412(.003)<br>0.356(.003)<br>0.375(.007)|
|Mistral-7B (CoT)|0.634(.003)<br>0.474(.002)<br>0.372(.005)<br>0.651(.002)|0.404(.010)<br>0.368(.023)<br>**0.379**(.027)|
|MedAlpaca-7B|0.600(.004)<br>0.401(.001)<br>0.333(.015)<br>0.493(.034)|0.399(.012)<br>0.325(.004)<br>0.311(.009)|
|MedAlpaca-7B (CoT)|0.603(.004)<br>0.399(.003)<br>0.315(.015)<br>0.485(.025)|0.395(.007)<br>0.321(.011)<br>0.312(.010)|
|PMC-LLaMA-7B|0.207(.011)<br>0.247(.004)<br>0.179(.007)<br>0.346(.017)|0.087(.015)<br>0.086(.002)<br>0.079(.006)|
|PMC-LLaMA-7B (CoT)|0.204(.008)<br>0.208(.002)<br>0.125(.014)<br>0.208(.002)|0.088(.002)<br>0.077(.004)<br>0.063(.005)|
|LLaMA3-8B|0.634(.005)<br>**0.566**(.004)<br>**0.586**(.008)<br>0.654(.005)|0.428(.015)<br>0.319(.002)<br>0.306(.009)|
|LLaMA3-8B (CoT)|**0.651**(.003)<br>0.552(.003)<br>0.574(.002)<br>**0.681**(.002)|**0.434**(.010)<br>**0.368**(.004)<br>0.313(.003)|
|Llama3-OpenBioLLM-8B|0.636(.005)<br>0.383(.003)<br>0.350(.026)<br>0.623(.005)|0.238(.004)<br>0.235(.011)<br>0.229(.020)|
|Llama3-OpenBioLLM-8B(CoT)|0.571 (.003)<br>0.295 (.002)<br>0.283 (.001)<br>0.646 (.004)|0.370 (.013)<br>0.330 (.001)<br>0.327 (.011)|
|LLaMA3.1-8B|0.677(.007)<br>**0.563**(.006)<br>0.596(.009)<br>0.687(.006)|0.434(.018)<br>0.368(.002)<br>0.306(.021)|
|LLaMA3.1-8B (CoT)|**0.681**(.005)<br>0.549(.003)<br>**0.600**(.005)<br>0.706(.002)|**0.439**(.017)<br>0.393(.005)<br>0.322(.014)|
|LLaMA2-13B|0.442(.002)<br>0.253(.004)<br>0.252(.004)<br>0.455(.002)|0.286(.003)<br>0.338(.006)<br>0.317(.006)|
|LLaMA2-13B(CoT)|0.415 (.002)<br>0.354 (.005)<br>0.232 (.006)<br>0.422 (.003)|0.309 (.005)<br>0.263 (.013)<br>0.243 (.016)|
|QAGNN|0.317(.003)<br>0.450(.005)<br>0.439(.033)<br>0.644(.002)|0.295(.003)<br>0.265(.002)<br>0.253(.003)|
|JointLK|0.288(.005)<br>0.472(.003)<br>0.468(.007)<br>0.640(.007)|0.247(.004)<br>0.250(.003)<br>0.244(.004)|
|Dragon|0.319 (.003)<br>0.475 (.002)<br>0.472 (.005)<br>0.646 (.003)|0.286 (.003)<br>0.247 (.005)<br>0.240 (.004)|
|Self-RAG (7B)|0.322(.019)<br>0.380(.028)<br>0.534(.028)<br>0.594(.012)|0.238(.007)<br>0.199(.037)<br>0.224(.045)|
|Self-RAG (13B)|0.502(.004)<br>0.408(.020)<br>0.331(.158)<br>0.646(.050)|0.249(.010)<br>0.290(.018)<br>0.266(.031)|
|KG-Rank (13B)|0.452(.005)<br>0.362(.011)<br>0.305(.019)<br>0.503(.015)|0.253(.021)<br>0.256(.013)<br>0.234(.010)|
|KG-RAG (8B)|0.516(.005)<br>0.343(.001)<br>0.429(.017)<br>0.662(.005)|0.434(.021)<br>**0.413**(.007)<br>**0.391**(.004)|
|MedRAG(70B)|0.579 (.015)<br>0.487 (.014)<br>0.574 (.022)<br>**0.719** (.018)|0.365 (.008)<br>0.348 (.011)<br>0.327 (.003)|
|KGAREVION(LLaMA3, w/o Review)|0.621(.002)<br>0.528(.003)<br>0.556(.002)<br>0.713(.004)|0.310(.006)<br>0.334(.004)<br>0.313(.008)|
|KGAREVION(LLaMA3, w/o Revise)|0.657(.004)<br>0.594(.006)<br>0.562(.002)<br>0.723(.005)|0.386(.008)<br>0.372(.004)<br>0.327(.003)|
|KGAREVION(LLaMA3,_k_ = 1)|**0.703**(.004)<br>0.610(.010)<br>0.562(.002)<br>**0.744**(.003)|**0.473**(.008)<br>0.404(.006)<br>0.395(.003)|
|KGAREVION(LLaMA3,_k_ = 2)|0.696(.006)<br>0.616(.008)<br>0.566(.002)<br>0.723(.010)|0.457(.006)<br>0.414(.006)<br>0.395(.005)|
|KGAREVION(LLaMA3,_k_ = 3)<br>|0.678(.006)<br>**0.628**(.002)<br>**0.590**(.005)<br>0.737(.007)<br><br><br><br>|0.469(.008)<br>**0.451**(.004)<br>**0.411**(.005)<br><br><br>|
|**Improvement over best baseline**|**+5.2%**<br>**+6.2%**<br>**+0.4%**<br>**+6.3%**|**+3.9%**<br>**+8.3%**<br>**+3.2%**|
|KGAREVION(LLaMA3.1, w/o Review)|0.695(.006)<br>0.546(.002)<br>0.560(.004)<br>0.736(.003)|0.298(.015)<br>0.299(.003)<br>0.327(.006)|
|KGAREVION(LLaMA3.1, w/o Revise)|0.716(.006)<br>0.573(.005)<br>0.568(.011)<br>0.749(.003)|0.392(.012)<br>0.337(.006)<br>0.352(.005)|
|KGAREVION(LLaMA3.1,_k_ = 1)|**0.734**(.004)<br>0.618(.002)<br>0.619(.004)<br>**0.763**(.001)|**0.483**(.013)<br>**0.457**(.010)<br>0.409(.005)|
|KGAREVION(LLaMA3.1,_k_ = 2)|0.720(.003)<br>0.616(.005)<br>0.656(.006)<br>0.745(.005)|0.396(.003)<br>0.454(.008)<br>0.342(.004)|
|KGAREVION(LLaMA3.1,_k_ = 3)<br>|0.716(.004)<br>**0.620**(.003)<br>**0.638**(.004)<br>0.749(.003)<br><br><br><br>|0.469(.012)<br>0.411(.010)<br>**0.447**(.005)<br><br><br>|
|**Improvement over best baseline**|**+5.3%**<br>**+5.7%**<br>**+3.8%**<br>**+4.4%**|**+4.4%**<br>**+4.4%**<br>**+5.6%**|



Table 2: The accuracy of KGAREVION and all baselines on four gold standard and three newly created datasets under multi-choice reasoning settings. The value highlighted in Blue indicates the best result among LLM-based reasoning models with a size smaller than 8B, including LLaMA38B, while Red marks the top value among LLaMA3.1-8B and other types of baselines. std means the standard deviation under three runs. 

**Results under QSS.** Fig. 3a illustrates the trends of KGAREVION alongside the top-performing baselines as the number of medical concepts increases. It can first be observed that KGAREVION outperforms baselines of the same size, regardless of the number of medical concepts involved. Moreover, KGAREVION maintains stable performance as the number of medical concepts increases and even improves when processing questions with n = 6, compared to those with n = 5. In contrast, baseline models struggle with complex questions containing 5 or 6 medical concepts. Such an observation indicates that KGAREVION advances in handling complex medical questions involving multiple medical concepts. 

**Results under CSS.** As seen in Table 2, evaluations on three difficult levels in MedDDx indicate that KGAREVION exhibits a strong ability in handling differential diagnosis questions that request professional and accurate knowledge. In addition, the obtained results also show that KGAREVION excels in identifying the correct answer among semantically similar answer candidates since it improves the accuracy on MedDDx-Expert by 3.2% and 5.6% with LLaMA3-8B and LLaMA3.1-8B as the backbone LLM, respectively. 

## 4.2 BENCHMARKING KGAREVION UNDER OPEN-ENDED REASONING SETTING 

We transform multiple-choice questions into descriptive, open-ended ones to better simulate realworld medical scenarios, where such inquiries are more common (see details in Appendix A.3). This adjustment requires our model to generate responses without predefined choices, encouraging holistic reasoning and the integration of diverse knowledge sources. By removing answer choices, we can more effectively assess the reasoning ability of KGAREVION in complex medical situations, resulting in a more realistic evaluation of its capabilities. Table 3 shows the accuracy and variance 

7 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_Knowledge_Graph_Based_Agent_For_Complex,_Knowledge-Intensive_QA_in_Medicine_images/iclr_2025_Knowledge_Graph_Based_Agent_For_Complex,_Knowledge-Intensive_QA_in_Medicine.pdf-0008-00.png)


**----- Start of picture text -----**<br>
a Multi-choice reasoning setting b Open reasoning setting<br>0.8 0.8 0.8 0.8<br>0.7 0.7<br>0.6 0.6<br>0.6 0.6<br>0.4 LLaMA3-8B 0.4 LLaMA3.1-8B LLaMA3-8B LLaMA3.1-8B<br>KGARevion (LLaMA3-8B) KGARevion (LLaMA3.1-8B) KGARevion (LLaMA3-8B) KGARevion (LLaMA3.1-8B)<br>0.5 0.5<br>1 2 3 4 5 6 1 2 3 4 5 6 1 2 3 4 5 6 1 2 3 4 5 6<br>the number of medical concepts the number of medical concepts<br>Accuracy Accuracy<br>**----- End of picture text -----**<br>


Figure 3: The accuracy of KGAREVION and pure LLMs with the medical concepts increase under **a** ) multi-choice reasoning setting and **b** ) open-ended reasoning setting. 

|**Open-ended Reasoning**|**Open-ended inquiries without pre-defned choices**|MedDDx-No-Opt<br>(**Open-ended**)|
|---|---|---|
|Method|MMLU-Med<br>MedQA-US<br>PubMedQA*<br>BioASQ-Y/N|Basic<br>Intermediate<br>Expert|
|Metrics|Acc. ∆Acc.<br>Acc. ∆Acc.<br>Acc. ∆Acc.<br>Acc. ∆Acc.|Acc. ∆Acc.<br>Acc. ∆Acc.<br>Acc. ∆Acc.|
|LLaMA2-7B|0.328(-0.048)<br>0.302(+0.021)<br>0.546(+0.098)<br>0.625(+0.057)|0.286(+0.071)<br>0.305(+0.107)<br>0.302(+0.110)|
|LLaMA2-7B (CoT)|0.362(+0.044)<br>0.243(-0.008)<br>0.418(-0.047)<br>0.642(+0.095)|0.265(-0.024)<br>0.270(+0.005)<br>0.280(+0.051)|
|Mistral-7B|0.591(-0.043)<br>0.412(-0.065)<br>0.344(-0.056)<br>0.629(-0.015)|0.249(-0.163)<br>0.228(-0.128)<br>0.273(-0.102)|
|Mistral-7B (CoT)|0.583(-0.051)<br>0.398(-0.076)<br>0.212(-0.160)<br>0.657(+0.006)|0.245(-0.159)<br>0.232(-0.136)<br>0.286(-0.093)|
|PMC-LLaMA-7B|0.073(-0.134)<br>0.082(-0.165)<br>0.090(-0.089)<br>0.139(-0.201)|0.249(+0.162)<br>0.191(+0.105)<br>0.232(+0.153)|
|PMC-LLaMA-7B (CoT)|0.080(-0.124)<br>0.079(-0.129)<br>0.092(-0.033)<br>0.125(-0.083)|0.220(+0.132)<br>0.245(+0.168)<br>0.228(+0.165)|
|LLaMA3-8B|0.595(-0.039)<br>**0.458**(-0.108)<br>0.532(-0.054)<br>0.672(+0.018)|**0.343**(-0.085)<br>**0.314**(-0.005)<br>0.317(+0.011)|
|LLaMA3-8B (CoT)|**0.608**(-0.043)<br>0.449(-0.103)<br>**0.562**(-0.012)<br>**0.714**(+0.033)|0.289(-0.145)<br>0.304(-0.064)<br>**0.327**(+0.014)|
|Llama3-OpenBioLLM-8B|0.324(-0.312)<br>0.157(-0.226)<br>0.157(-0.193)<br>0.324(-0.299)|0.016(-0.222)<br>0.006(-0.229)<br>0.008(-0.221)|
|Llama3-OpenBioLLM-8B(CoT)|0.398 (-0.173)<br>0.146 (-0.149)<br>0.100 (-0.183)<br>0.142 (-0.539)|0.098 (-0.272)<br>0.084 (-0.246)<br>0.108 (-0.219)|
|LLaMA3.1-8B|0.607(-0.070)<br>0.551(-0.012)<br>0.514(-0.082)<br>0.694(+0.007)|**0.322**(-0.112)<br>0.289(-0.079)<br>**0.335**(+0.029)|
|LLaMA3.1-8B (CoT)|**0.697**(+0.016)<br>**0.563**(+0.014)<br>**0.572**(-0.028)<br>**0.706**(-0.000)|0.306(-0.133)<br>**0.294**(-0.099)<br>0.315(-0.007)|
|LLaMA2-13B|0.348(-0.094)<br>0.283(+0.030)<br>0.218(-0.034)<br>0.421(-0.034)|0.190(-0.096)<br>0.123(-0.215)<br>0.153(-0.164)|
|LLaMA2-13B(CoT)|0.311 (-0.104)<br>0.267 (-0.087)<br>0.266 (+0.034)<br>0.471 (+0.049)|0.269 (-0.040)<br>0.268 (+0.005)<br>0.282 (+0.039)|
|Self-RAG (7B)|0.256(-0.066)<br>0.235(-0.145)<br>0.316(-0.218)<br>0.379(-0.215)|0.167(-0.071)<br>0.246(+0.047)<br>0.213(-0.011)|
|Self-RAG (13B)|0.309(-0.193)<br>0.297(-0.111)<br>0.438(+0.107)<br>0.539(-0.107)|0.212(-0.037)<br>0.232(-0.058)<br>0.226(–0.040)|
|KG-Rank (13B)|0.151(-0.301)<br>0.189(-0.173)<br>0.203(-0.102)<br>0.188(-0.315)|0.127(-0.126)<br>0.133(-0.123)<br>0.133(-0.101)|
|KG-RAG(8B)|0.310 (-0.206)<br>0.290 (-0.053)<br>0.316 (-0.113)<br>0.359 (-0.303)|0.216 (-0.218)<br>0.220 (-0.193)<br>0.213 (-0.178)|
|KGAREVION(LLaMA3, w/o Review)|0.645(+0.024)<br>0.609(+0.081)<br>0.552(-0.004)<br>0.701(-0.012)|0.400(+0.090)<br>0.360(+0.026)<br>0.356(+0.043)|
|KGAREVION(LLaMA3, w/o Revise)|0.668(+0.011)<br>0.626(+0.032)<br>0.572(+0.010)<br>0.716(-0.007)|0.426(+0.040)<br>0.403(+0.031)<br>0.412(+0.085)|
|KGAREVION(LLaMA3,_k_ = 1)|0.687(-0.016)<br>0.628(+0.018)<br>**0.578**(+0.016)<br>0.730(-0.014)|0.465(-0.008)<br>0.430(+0.026)<br>0.428(+0.033)|
|KGAREVION(LLaMA3,_k_ = 2)|0.682(-0.014)<br>**0.638**(+0.022)<br>0.566(-0.000)<br>**0.736**(+0.013)|**0.527**(+0.070)<br>**0.463**(+0.049)<br>**0.489**(+0.094)|
|KGAREVION(LLaMA3,_k_ = 3)<br>|**0.696**(+0.018)<br>0.632(+0.004)<br>0.572(-0.018)<br>0.733(-0.004)<br><br><br><br>|0.489(+0.020)<br>0.411(-0.040)<br>0.429(+0.018)<br><br><br>|
|**Improvement over best baseline**|**+8.8%**<br>**+18.0%**<br>**+1.6%**<br>**+2.2%**|**+18.4%**<br>**+14.9%**<br>**+16.2%**|
|KGAREVION(LLaMA3.1, w/o Review)|0.659(-0.036)<br>0.526(-0.020)<br>0.556(-0.004)<br>0.726(-0.010)|0.457(+0.159)<br>0.435(+0.136)<br>0.439(+0.112)|
|KGAREVION(LLaMA3.1, w/o Revise)|0.695(-0.021)<br>0.626(+0.053)<br>0.556(-0.012)<br>0.736(-0.013)|**0.489**(+0.097)<br>0.436(+0.099)<br>0.451(+0.099)|
|KGAREVION(LLaMA3.1,_k_ = 1)|**0.720**(-0.014)<br>**0.644**(+0.026)<br>0.560(-0.059)<br>**0.757**(-0.006)|0.469(-0.014)<br>0.454(-0.003)<br>0.437(+0.028)|
|KGAREVION(LLaMA3.1,_k_ = 2)|0.704(-0.016)<br>0.636(+0.020)<br>**0.572**(-0.084)<br>0.734(-0.011)|0.469(+0.073)<br>0.446(-0.008)<br>0.432(+0.090)|
|KGAREVION(LLaMA3.1,_k_ = 3)<br>|0.712(-0.004)<br>0.639(+0.019)<br>0.562(-0.076)<br>0.748(-0.001)<br><br><br><br>|0.449(-0.020)<br>**0.470**(+0.059)<br>**0.451**(+0.004)<br><br><br>|
|**Improvement over best baseline**|**+2.4%**<br>**+8.1%**<br>**+0.0%**<br>**+4.9%**|**+16.7%**<br>**+17.6%**<br>**+11.6%**|



Table 3: The accuracy of KGAREVION and baselines on four gold standard and three newly created datasets under open-ended reasoning settings. The value highlighted in Blue indicates the best result among LLM-based reasoning models with a size smaller than 8B, including LLaMA3-8B, while Red marks the top value among LLaMA3.1-8B and other types of baselines. ∆Acc. denotes the difference in performance between the open-ended and multiple-choice reasoning settings. 

across all datasets. The variance here denotes the difference in accuracy compared with that in the multi-choice reasoning setting. 

**Results under QSS.** Fig. 3b shows the accuracy obtained by pure LLM and KGAREVION with the increase of medical concepts under open-ended reasoning setting. Compared to pure LLMs in the open-ended reasoning setting, KGAREVION shows a significant improvement in handling complex medical reasoning tasks involving more than 4 medical concepts and a comparable performance in questions with less than 3 medical concepts. 

**Results under CSS.** To evaluate the model’s ability to solve complex medical QA with differential diagnosis, we compare KGAREVION and all baselines on newly created datasets, as shown in Table 3. KGAREVION still achieves the best performance in the open-ended reasoning setting. In addition, compared with the results in multi-choice reasoning setting, KGAREVION performs better. On the one hand, such a result demonstrates the strong ability of KGAREVION in the open-ended reasoning setting. On the other hand, it also indicates that these semantically candidates may affect the reasoning process to a certain extent. 

8 

## 4.3 ABLATION ANALYSES 

**Effect of the ‘Review’ action.** As shown in Table 2, 3, and Fig. 4 (KGAREVION (w/o Review) vs. KGAREVION (w/o Revise))), the Review action plays an important role in answering medical questions under two settings, which improves the average accuracy across all datasets by 3.3% and 3%, respectively. Fig. 4 shows that the Review action has a more pronounced effect on the MedDDx dataset than the four gold-standard datasets under two settings, suggesting that its integration enhances the model’s ability to tackle complex medical questions. Furthermore, the Review action leads to greater accuracy improvements in the four gold-standard datasets under the open-ended reasoning setting compared to the multiple-choice reasoning setting. This highlights the significance of verifying generated answers, particularly in an open-ended reasoning setup. 

**Number of refinement rounds in the ‘Revise’ action.** The Revise action is designed to enhance accuracy by correcting erroneous triplets until they are verified as true by the Review action. Tables 2 and 3, and Fig. 4 demonstrate its positive impact on KGAREVION across both settings. Specifically, Figure 4 indicates that the Review action significantly improves performance on MedDDx dataset, yielding average enhancements of 9% and 4% in accuracy for both settings compared to questions in the goldstandard datasets. Additionally, we investigate 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_Knowledge_Graph_Based_Agent_For_Complex,_Knowledge-Intensive_QA_in_Medicine_images/iclr_2025_Knowledge_Graph_Based_Agent_For_Complex,_Knowledge-Intensive_QA_in_Medicine.pdf-0009-03.png)


**----- Start of picture text -----**<br>
a Multi-choice reasoning setting b Open reasoning setting<br>w/o Review w/o Review<br>1.0 w/o Revise 1.0 w/o Revise<br>KGARevion KGARevion<br>0.8 0.62 0.64 0.68 0.8 0.62 0.65 0.67<br>0.6 0.6<br>0.45 0.41 0.44 0.48<br>0.4 0.31 0.36 0.4<br>Gold standard datasets MedDDx Gold standard datasets MedDDx<br>Accuracy  Accuracy<br>**----- End of picture text -----**<br>


Figure 4: The results of ablation studies across all datasets under two settings. 

the impact of the number of revision rounds across all datasets in both settings, as shown in Table 2 and 3. The results indicate that KGAREVION can achieve optimal performance with _k_ = 1 on most of datasets in the multi-choice reasoning setting. However, it benefits from additional iterations when addressing complex questions, such as those in the MedDDx-Expert dataset. In the openended reasoning setting, KGAREVION typically requires more iterations to arrive at the correct answer. 

## 4.4 VERSATILITY OF KGAREVION 

**KGAREVION can be used with different LLMs.** KGAREVION is a versatile agent that can be implemented with a variety of LLMs. We implement KGAREVION using three distinct models: LLaMA3-8B, LLaMA3.1-8B, and GPT-4o. The averaged results across all datasets, as shown in Fig. 5a, demonstrate the effectiveness of KGAREVION’s architecture, consistently improving the performance of the backbone LLMs by 6%, 7%, and 2%, respectively. 

**medicalKGAR** VION grounds the generated triplets using KGs. **EVIONKGs.** The **can** Review **be used** action **with** in KGAR **different** E- **a** 1.00.8 0.50+6%Pure LLMsKGARevion 0.56 0.52+7%0.59 0.63+2%0.65 **b** 1.00.8 0.69 Gold standard datasetsMedDDx 0.67 To evaluate the impact of different KGs (see de0.6 0.6 0.45 tails in Appendix B.5), we implement KGARE0.4 0.4 0.41 VION with two comprehensive KGs and assess its performance across all datasets, as shown in LLaMA3-8B LLaMA3.1-8B GPT-4-Turbo PrimeKG OGB-biokg Fig. 5b. The results show that KGAREVION is not sensitive to the choice of knowledge bases, Figure 5: **a)** Performance of KGAREVION with highlighting its robustness and generalizability, different backbone LLMs across all datasets. despite PrimeKG (Chandak et al., 2023) being **b)** Performance of KGAREVION with different much larger than OGB-biokg (Hu et al., 2020). KGs used in the fine-tuning stage of the Review This robustness arises because KGs are used action. only in the Review action to verify generated triplets rather than as a source for retrieving knowledge. This also explains why KGAREVION outperforms KG-based RAG models, which heavily rely on the chosen KGs, whereas KGAREVION uses comprehensive KGs simply to ensure that the generated triplets are aligned with medical knowledge. 

9 

## 4.5 SENSITIVITY ANALYSES 

Recent studies have revealed that LLMs can be surprisingly sensitive to both how candidate answers are ordered and indexed in multi-choice setups (Zheng et al., 2023; Pezeshkpour & Hruschka, 2023). These studies found that LLMs are not robust multiple-choice selectors and exhibit order sensitivity, favoring answers at the first position (Li et al., 2024). To investigate this issue, we examine how the order and indexing of answers affect model performance. We evaluate KGAREVION using LLaMA3-8B and LLaMA3.1-8B as the backbone and compare its performance with their LLMonly counterparts across all datasets (details are provided in Appendix B.4.1). Fig. 6 illustrates the changes in accuracy when the order or labels of the candidate answers are altered. 

**Ordering of candidate answers in multi-choice setups.** Fig. 6a shows that pure LLMs are sensitive to answer order, with an average accuracy shift of 8.4% for LLaMA3-8B and 16.0% for LLaMA3.1-8B. In contrast, KGAREVION demonstrates significantly greater robustness to answer order. This robustness is primarily due to KGAREVION’s ability to fairly evaluate each answer using the Generate action, effectively mitigating the impact of answer order on model performance. 

**Indexing of candidate answers in multi-choice setups.** The accuracy of pure LLMs shows a substantial shift when relabeling answers from ABCD to EFGH, as illustrated in Fig. 6b. Specifically, the average accuracy shift is 8.1% for LLaMA38B and 12.9% for LLaMA3.1-8B. In contrast, our agent KGAREVION significantly improves the stability of these LLMs, reducing the accuracy loss to 2.59% and 3.86%, respectively. This finding further highlights the robustness of KGAREVION. 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_Knowledge_Graph_Based_Agent_For_Complex,_Knowledge-Intensive_QA_in_Medicine_images/iclr_2025_Knowledge_Graph_Based_Agent_For_Complex,_Knowledge-Intensive_QA_in_Medicine.pdf-0010-04.png)


**----- Start of picture text -----**<br>
a 20 Changing Order b 20 Relabeling Index<br>Pure LLMs Pure LLMs<br>KGARevion 16.00 KGARevion<br>15 15 12.90<br>10 8.40 10 8.10<br>5 5 3.86<br>2.46 2.59<br>1.50<br>0 0<br>LLaMA3-8B LLaMA3.1-8B LLaMA3-8B LLaMA3.1-8B<br>Accuracy | (%) Accuracy | (%)<br>| Δ  | Δ<br>**----- End of picture text -----**<br>


Figure 6: _|_ ∆Accuracy _|_ of LLMs and KGAREVION when changing order or relabeling index. 

## 4.6 CASE STUDIES 

Fig. 7 illustrates the reasoning process of KGAREVION in both settings, using the same input question. In both cases, KGAREVION arrives at the correct answer, but the reasoning processes differ. In the open reasoning setting, KGAREVION requires more iterations to revise triplets and guide the reasoning compared to the multiple-choice setting. Additionally, the verified correct triplets provide a reasoning path that helps explain the final answer, such as ’ _partial deletion of the long arm of chromosome 19 → associated with → 19q13.11 deletion syndrome and partial chromosome 19 deletions → associated with → disease_ ’. 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_Knowledge_Graph_Based_Agent_For_Complex,_Knowledge-Intensive_QA_in_Medicine_images/iclr_2025_Knowledge_Graph_Based_Agent_For_Complex,_Knowledge-Intensive_QA_in_Medicine.pdf-0010-08.png)


**----- Start of picture text -----**<br>
Question: syndrome and partial chromosome 19 deletions in terms of shared symptoms or genetic links?Which diseases resemble or stem from 19q13.11 deletion  Question: syndrome and partial chromosome 19 deletions in terms of shared symptoms or genetic links?Which diseases resemble or stem from 19q13.11 deletion<br>A: partial deletion of chromosome 19<br>B: partial deletion of the short arm of chromosome 19C: partial deletion of the long arm of chromosome 19 KGARevion<br>D: partial deletion of the long arm of chromosome 18<br>Generate<br>Generated Triplets are: ['19q13.11 deletion syndrome', 'linked to', 'partial chromosome 19 deletions'], ['partial chromosome 19 deletions', 'associated with', 'diseases'],['19q13.11 deletion syndrome', 'associated with', 'diseases']Generate KGARevion    Generated Triplets are: ●●●●●●●● ['19q13.11 deletion syndrome', 'linked to', 'partial deletion of chromosome 19']['partial chromosome 19 deletions', 'associated with', 'physical disabilities']['19q13.11 deletion syndrome', 'associated with', 'partial deletion of chromosome 19']['partial deletion of chromosome 19', 'associated with', 'diseases'], ['19q13.11 deletion syndrome', 'shared symptoms with', 'diseases']['partial deletion of chromosome 19', 'shared symptoms with', 'diseases']['19q13.11 deletion syndrome', 'genetic link with', 'partial deletion of chromosome 19'], ['partial deletion of chromosome 19', 'genetic link with', 'diseases'],<br>Review ......<br>●● ['19q13.11 deletion syndrome', 'linked to', 'partial chromosome 19 deletions'] Output: True<br>●● ['partial chromosome 19 deletions', 'associated with', 'diseases']  Output: False Review Revise<br>●● ['19q13.11 deletion syndrome', 'associated with', 'diseases'] Output: True<br>Revised Triplet is:  ['partial deletion of the long arm of chromosome 19', 'associated with', '19q13.11 deletion syndrome and partial chromosome 19 deletions']●● ['partial deletion of the long arm of chromosome 19', 'associated with', '19q13.11 deletion syndrome and partial chromosome 19 deletions'] Output: True ReviewRevise ●●●●●●● ['19q13.11 deletion syndrome', 'linked to', 'partial deletion of chromosome 19'], ['partial deletion of the long arm of chromosome 19', 'linked to', 'diseases'], ['19q13.11 deletion syndrome', 'associated with', 'partial deletion of the long arm of chromosome 19'], ['partial deletion of the long arm of chromosome 19', 'associated with', 'diseases'], ['19q13.11 deletion syndrome', 'shared symptoms with', 'diseases'], ['19q13.11 deletion syndrome', 'genetic link with', 'partial deletion of chromosome 19'], ['partial deletion of the long arm of chromosome 19', 'genetic link with', 'diseases'], ......<br>Answer Answer<br>C: partial deletion of the long arm of chromosome 19 Answer: partial deletion of the long arm of chromosome 19<br>**----- End of picture text -----**<br>


## 5 CONCLUSION 

Figure 7: The reasoning process of KGAREVION under multi-choice setting (Left) and open reasoning setting (Right). 

Medical reasoning presents unique challenges that require integrating multi-source, grounded, 

and specialized domain knowledge. In this work, we introduced KGAREVION, a KG-based LLM agent that addresses these challenges by combining the non-codified knowledge of LLMs with the structured, codified knowledge of medical concepts stored in KGs. Through its adaptive reasoning and mechanisms for generating, verifying, and revising knowledge, KGAREVION can handle complex medical QA. Experiments across multiple-choice and open-ended tasks, using a variety of datasets—including challenging new benchmarks—demonstrate KGAREVION’s ability to systematically improve accuracy. By grounding LLM-generated knowledge in KGs, KGAREVION ensures contextual relevance and reliability, making it a valuable tool for knowledge-intensive medical QA. 

10 

## 6 ACKNOWLEDGEMENT 

We gratefully acknowledge the support of NIH R01-HD108794, NSF CAREER 2339524, US DoD FA8702-15-D-0001, ARPA-H BDF Toolbox Program, awards from Pfizer Research, Harvard Data Science Initiative, Amazon Faculty Research, Google Research Scholar Program, AstraZeneca Research, Roche Alliance with Distinguished Scientists, Sanofi iDEA-iTECH Award, Chan Zuckerberg Initiative, John and Virginia Kaneb Fellowship award at Harvard Medical School, Biswas Computational Biology Initiative in partnership with the Milken Institute, Kempner Institute for the Study of Natural and Artificial Intelligence, and Harvard Medical School Dean’s Innovation Awards. V.G. is supported by the UK Medical Research Council, MR/W00710X/1. The content is solely the responsibility of the authors. 

## REFERENCES 

- Vaibhav Adlakha, Parishad BehnamGhader, Xing Han Lu, Nicholas Meade, and Siva Reddy. Evaluating correctness and faithfulness of instruction-following models for question answering. _Transactions of the Association for Computational Linguistics_ , 12:775–793, 2024. 

- Malaikannan Sankarasubbu Ankit Pal. Openbiollms: Advancing open-source large language models for healthcare and life sciences. https://huggingface.co/aaditya/ OpenBioLLM-Llama3-70B, 2024. 

- Akari Asai, Zeqiu Wu, Yizhong Wang, Avirup Sil, and Hannaneh Hajishirzi. Self-RAG: Learning to retrieve, generate, and critique through self-reflection. In _The Twelfth International Conference on Learning Representations_ , 2024. URL https://openreview.net/forum?id= hSyW5go0v8. 

- Marsden S Blois. Medicine and the nature of vertical reasoning. _The New England journal of medicine_ , 318(13):847–851, 1988. 

- Olivier Bodenreider. The unified medical language system (umls): integrating biomedical terminology. _Nucleic acids research_ , 32(suppl ~~1~~ ):D267–D270, 2004. 

- Daniil A Boiko, Robert MacKnight, Ben Kline, and Gabe Gomes. Autonomous chemical research with large language models. _Nature_ , 624(7992):570–578, 2023. 

- Antoine Bordes, Nicolas Usunier, Alberto Garcia-Duran, Jason Weston, and Oksana Yakhnenko. Translating embeddings for modeling multi-relational data. _Advances in neural information processing systems_ , 26, 2013. 

- Andres M Bran, Sam Cox, Oliver Schilter, Carlo Baldassari, Andrew White, and Philippe Schwaller. Augmenting large language models with chemistry tools. In _NeurIPS 2023 AI for Science Workshop_ , 2023. URL https://openreview.net/forum?id=wdGIL6lx3l. 

- Payal Chandak, Kexin Huang, and Marinka Zitnik. Building a knowledge graph to enable precision medicine. _Scientific Data_ , 10(1):67, 2023. 

- Cathy Charles, Amiram Gafni, and Tim Whelan. Shared decision-making in the medical encounter: what does it mean?(or it takes at least two to tango). _Social science & medicine_ , 44(5):681–692, 1997. 

- Zeming Chen, Alejandro Hern´andez Cano, Angelika Romanou, Antoine Bonnet, Kyle Matoba, Francesco Salvi, Matteo Pagliardini, Simin Fan, Andreas K¨opf, Amirkeivan Mohtashami, et al. Meditron-70b: Scaling medical pretraining for large language models. _arXiv preprint arXiv:2311.16079_ , 2023. 

- Sunhao Dai, Chen Xu, Shicheng Xu, Liang Pang, Zhenhua Dong, and Jun Xu. Bias and unfairness in information retrieval systems: New challenges in the llm era. In _Proceedings of the 30th ACM SIGKDD Conference on Knowledge Discovery and Data Mining_ , pp. 6437–6447, 2024. 

- Abhimanyu Dubey, Abhinav Jauhri, Abhinav Pandey, Abhishek Kadian, Ahmad Al-Dahle, Aiesha Letman, Akhil Mathur, Alan Schelten, Amy Yang, and Angela Fan. The llama 3 herd of models, 2024. URL https://arxiv.org/abs/2407.21783. 

11 

- Darren Edge, Ha Trinh, Newman Cheng, Joshua Bradley, Alex Chao, Apurva Mody, Steven Truitt, and Jonathan Larson. From local to global: A graph rag approach to query-focused summarization. _arXiv preprint arXiv:2404.16130_ , 2024. 

- Wenqi Fan, Yujuan Ding, Liangbo Ning, Shijie Wang, Hengyun Li, Dawei Yin, Tat-Seng Chua, and Qing Li. A survey on rag meeting llms: Towards retrieval-augmented large language models. In _Proceedings of the 30th ACM SIGKDD Conference on Knowledge Discovery and Data Mining_ , pp. 6491–6501, 2024. 

- Shanghua Gao, Ada Fang, Yepeng Huang, Valentina Giunchiglia, Ayush Noori, Jonathan Richard Schwarz, Yasha Ektefaie, Jovana Kondic, and Marinka Zitnik. Empowering biomedical discovery with ai agents. _arXiv preprint arXiv:2404.02831_ , 2024. 

- Ojas Gramopadhye, Saeel Sandeep Nachane, Prateek Chanda, Ganesh Ramakrishnan, Kshitij Sharad Jadhav, Yatin Nandwani, Dinesh Raghu, and Sachindra Joshi. Few shot chain-ofthought driven reasoning to prompt llms for open ended medical question answering. _arXiv preprint arXiv:2403.04890_ , 2024. 

- Tianyu Han, Lisa C Adams, Jens-Michalis Papaioannou, Paul Grundmann, Tom Oberhauser, Alexander L¨oser, Daniel Truhn, and Keno K Bressem. Medalpaca–an open-source collection of medical conversational ai models and training data. _arXiv preprint arXiv:2304.08247_ , 2023. 

- Emily Harris. Large language models answer medical questions accurately, but can’t match clinicians’ knowledge. _JAMA_ , 2023. 

- Edward J Hu, yelong shen, Phillip Wallis, Zeyuan Allen-Zhu, Yuanzhi Li, Shean Wang, Lu Wang, and Weizhu Chen. LoRA: Low-rank adaptation of large language models. In _International Conference on Learning Representations_ , 2022. URL https://openreview.net/forum? id=nZeVKeeFYf9. 

- Weihua Hu, Matthias Fey, Marinka Zitnik, Yuxiao Dong, Hongyu Ren, Bowen Liu, Michele Catasta, and Jure Leskovec. Open graph benchmark: Datasets for machine learning on graphs. _Advances in neural information processing systems_ , 33:22118–22133, 2020. 

- Soyeong Jeong, Jinheon Baek, Sukmin Cho, Sung Ju Hwang, and Jong C Park. Adaptive-rag: Learning to adapt retrieval-augmented large language models through question complexity. _arXiv preprint arXiv:2403.14403_ , 2024. 

- Albert Q. Jiang, Alexandre Sablayrolles, Arthur Mensch, Chris Bamford, Devendra Singh Chaplot, Diego de las Casas, Florian Bressand, Gianna Lengyel, Guillaume Lample, Lucile Saulnier, L´elio Renard Lavaud, Marie-Anne Lachaux, Pierre Stock, Teven Le Scao, Thibaut Lavril, Thomas Wang, Timoth´ee Lacroix, and William El Sayed. Mistral 7b, 2023. URL https: //arxiv.org/abs/2310.06825. 

- Pengcheng Jiang, Cao Xiao, Adam Richard Cross, and Jimeng Sun. Graphcare: Enhancing healthcare predictions with personalized knowledge graphs. In _The Twelfth International Conference on Learning Representations_ . 

- Vladimir Karpukhin, Barlas Oguz, Sewon Min, Patrick Lewis, Ledell Wu, Sergey Edunov, Danqi Chen, and Wen-tau Yih. Dense passage retrieval for open-domain question answering. In _Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing (EMNLP)_ , pp. 6769–6781, 2020. 

- Guohao Li, Hasan Abed Al Kader Hammoud, Hani Itani, Dmitrii Khizbullin, and Bernard Ghanem. Camel: Communicative agents for” mind” exploration of large language model society. In _Thirtyseventh Conference on Neural Information Processing Systems_ , 2023. 

- Wangyue Li, Liangzhi Li, Tong Xiang, Xiao Liu, Wei Deng, and Noa Garcia. Can multiple-choice questions really be useful in detecting the abilities of llms? _arXiv preprint arXiv:2403.17752_ , 2024. 

- Valentin Li´evin, Christoffer Egeberg Hother, Andreas Geert Motzfeldt, and Ole Winther. Can large language models reason about medical questions? _Patterns_ , 5(3), 2024. 

12 

- J¨org Menche, Amitabh Sharma, Maksim Kitsak, Susan Dina Ghiassian, Marc Vidal, Joseph Loscalzo, and Albert-L´aszl´o Barab´asi. Uncovering disease-disease relationships through the incomplete interactome. _Science_ , 347(6224):1257601, 2015. 

- Lino Murali, G Gopakumar, Daleesha M Viswanathan, and Prema Nedungadi. Towards electronic health record-based medical knowledge graph construction, completion, and applications: A literature study. _Journal of biomedical informatics_ , 143:104403, 2023. 

- Reiichiro Nakano, Jacob Hilton, Suchir Balaji, Jeff Wu, Long Ouyang, Christina Kim, Christopher Hesse, Shantanu Jain, Vineet Kosaraju, William Saunders, et al. Webgpt: Browser-assisted question-answering with human feedback. _arXiv preprint arXiv:2112.09332_ , 2021. 

OpenAI. Gpt-4 technical report, 2024. URL https://arxiv.org/abs/2303.08774. 

- Long Ouyang, Jeffrey Wu, Xu Jiang, Diogo Almeida, Carroll Wainwright, Pamela Mishkin, Chong Zhang, Sandhini Agarwal, Katarina Slama, Alex Gray, John Schulman, Jacob Hilton, Fraser Kelton, Luke Miller, Maddie Simens, Amanda Askell, Peter Welinder, Paul Christiano, Jan Leike, and Ryan Lowe. Training language models to follow instructions with human feedback. In Alice H. Oh, Alekh Agarwal, Danielle Belgrave, and Kyunghyun Cho (eds.), _Advances in Neural Information Processing Systems_ , 2022. URL https://openreview.net/forum?id= TG8KACxEON. 

- Vimla L Patel, Jos´e F Arocha, and Jiajie Zhang. Thinking and reasoning in medicine. _The Cambridge handbook of thinking and reasoning_ , 14:727–750, 2005. 

- Pouya Pezeshkpour and Estevam Hruschka. Large language models sensitivity to the order of options in multiple-choice questions. _arXiv preprint arXiv:2308.11483_ , 2023. 

- Kunxun Qi, Jianfeng Du, and Hai Wan. Learning from both structural and textual knowledge for inductive knowledge graph completion. _Advances in Neural Information Processing Systems_ , 36, 2024. 

- Alec Radford. Improving language understanding by generative pre-training. 2018. 

- Alec Radford, Jeffrey Wu, Rewon Child, David Luan, Dario Amodei, Ilya Sutskever, et al. Language models are unsupervised multitask learners. _OpenAI blog_ , 1(8):9, 2019. 

- Timo Schick, Jane Dwivedi-Yu, Roberto Dessi, Roberta Raileanu, Maria Lomeli, Eric Hambro, Luke Zettlemoyer, Nicola Cancedda, and Thomas Scialom. Toolformer: Language models can teach themselves to use tools. In _Thirty-seventh Conference on Neural Information Processing Systems_ , 2023. URL https://openreview.net/forum?id=Yacmpz84TH. 

- Yongliang Shen, Kaitao Song, Xu Tan, Dongsheng Li, Weiming Lu, and Yueting Zhuang. HuggingGPT: Solving AI tasks with chatGPT and its friends in hugging face. In _Thirty-seventh Conference on Neural Information Processing Systems_ , 2023. URL https://openreview. net/forum?id=yHdTscY6Ci. 

- Zhengliang Shi, Shuo Zhang, Weiwei Sun, Shen Gao, Pengjie Ren, Zhumin Chen, and Zhaochun Ren. Generate-then-ground in retrieval-augmented generation for multi-hop question answering. In Lun-Wei Ku, Andre Martins, and Vivek Srikumar (eds.), _Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)_ , pp. 7339–7353, Bangkok, Thailand, August 2024. Association for Computational Linguistics. URL https://aclanthology.org/2024.acl-long.397. 

- Noah Shinn, Federico Cassano, Ashwin Gopinath, Karthik R Narasimhan, and Shunyu Yao. Reflexion: Language agents with verbal reinforcement learning. In _Thirty-seventh Conference on Neural Information Processing Systems_ , 2023. 

- Karan Singhal, Shekoofeh Azizi, Tao Tu, S Sara Mahdavi, Jason Wei, Hyung Won Chung, Nathan Scales, Ajay Tanwani, Heather Cole-Lewis, Stephen Pfohl, et al. Large language models encode clinical knowledge. _Nature_ , 620(7972):172–180, 2023. 

13 

- Karthik Soman, Peter W Rose, John H Morris, Rabia E Akbas, Brett Smith, Braian Peetoom, Catalina Villouta-Reyes, Gabriel Cerono, Yongmei Shi, Angela Rizk-Jackson, et al. Biomedical knowledge graph-enhanced prompt generation for large language models. _arXiv preprint arXiv:2311.17330_ , 2023. 

- Yueqing Sun, Qi Shi, Le Qi, and Yu Zhang. Jointlk: Joint reasoning with language models and knowledge graphs for commonsense question answering. In Marine Carpuat, Marie-Catherine de Marneffe, and Iv´an Vladimir Meza Ru´ız (eds.), _Proceedings of the 2022 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, NAACL 2022, Seattle, WA, United States, July 10-15, 2022_ , pp. 5049–5060. Association for Computational Linguistics, 2022. doi: 10.18653/V1/2022.NAACL-MAIN.372. URL https://doi.org/10.18653/v1/2022.naacl-main.372. 

- Xiangru Tang, Anni Zou, Zhuosheng Zhang, Yilun Zhao, Xingyao Zhang, Arman Cohan, and Mark Gerstein. Medagents: Large language models as collaborators for zero-shot medical reasoning. _arXiv preprint arXiv:2311.10537_ , 2023. 

- Nandan Thakur, Luiz Bonifacio, Xinyu Zhang, Odunayo Ogundepo, Ehsan Kamalloo, David Alfonso-Hermelo, Xiaoguang Li, Qun Liu, Boxing Chen, Mehdi Rezagholizadeh, and Jimmy Lin. Nomiracl: Knowing when you don’t know for robust multilingual retrieval-augmented generation. _CoRR_ , abs/2312.11361, 2023. doi: 10.48550/ARXIV.2312.11361. URL https: //doi.org/10.48550/arXiv.2312.11361. 

- Hugo Touvron, Louis Martin, Kevin Stone, Peter Albert, Amjad Almahairi, Yasmine Babaei, and Nikolay Bashlykov. Llama 2: Open foundation and fine-tuned chat models, 2023. URL https: //arxiv.org/abs/2307.09288. 

- A Vaswani. Attention is all you need. _Advances in Neural Information Processing Systems_ , 2017. 

- A Venigalla, Jonathan Frankle, and M Carbin. Biomedlm: a domain-specific large language model for biomedical text. _MosaicML. Accessed: Dec_ , 23(3):2, 2022. 

- Guanzhi Wang, Yuqi Xie, Yunfan Jiang, Ajay Mandlekar, Chaowei Xiao, Yuke Zhu, Linxi Fan, and Anima Anandkumar. Voyager: An open-ended embodied agent with large language models. In _Intrinsically-Motivated and Open-Ended Learning Workshop @NeurIPS2023_ , 2023a. URL https://openreview.net/forum?id=nfx5IutEed. 

- Yubo Wang, Xueguang Ma, and Wenhu Chen. Augmenting black-box llms with medical textbooks for clinical question answering. _arXiv preprint arXiv:2309.02233_ , 2023b. 

- Jason Wei, Xuezhi Wang, Dale Schuurmans, Maarten Bosma, Brian Ichter, Fei Xia, Ed H. Chi, Quoc V. Le, and Denny Zhou. Chain-of-thought prompting elicits reasoning in large language models. In _Proceedings of the 36th International Conference on Neural Information Processing Systems_ , NIPS ’22, Red Hook, NY, USA, 2024. Curran Associates Inc. ISBN 9781713871088. 

- Chaoyi Wu, Weixiong Lin, Xiaoman Zhang, Ya Zhang, Weidi Xie, and Yanfeng Wang. Pmc-llama: toward building open-source language models for medicine. _Journal of the American Medical Informatics Association_ , pp. ocae045, 2024a. 

- Junde Wu, Jiayuan Zhu, and Yunli Qi. Medical graph rag: Towards safe medical large language model via graph retrieval-augmented generation. _arXiv preprint arXiv:2408.04187_ , 2024b. 

- Qingyun Wu, Gagan Bansal, Jieyu Zhang, Yiran Wu, Shaokun Zhang, Erkang Zhu, Beibin Li, Li Jiang, Xiaoyun Zhang, and Chi Wang. Autogen: Enabling next-gen llm applications via multiagent conversation framework. _arXiv preprint arXiv:2308.08155_ , 2023. 

- Shirley Wu, Shiyu Zhao, Michihiro Yasunaga, Kexin Huang, Kaidi Cao, Qian Huang, Vassilis N. Ioannidis, Karthik Subbian, James Zou, and Jure Leskovec. Stark: Benchmarking llm retrieval on textual and relational knowledge bases. 2024c. 

- Guangzhi Xiong, Qiao Jin, Zhiyong Lu, and Aidong Zhang. Benchmarking retrieval-augmented generation for medicine. In Lun-Wei Ku, Andre Martins, and Vivek Srikumar (eds.), _Findings of the Association for Computational Linguistics ACL 2024_ , pp. 6233–6251, Bangkok, Thailand and virtual meeting, August 2024a. Association for Computational Linguistics. URL https: //aclanthology.org/2024.findings-acl.372. 

14 

Guangzhi Xiong, Qiao Jin, Zhiyong Lu, and Aidong Zhang. Benchmarking retrieval-augmented generation for medicine, 2024b. URL https://arxiv.org/abs/2402.13178. 

- Rui Yang, Haoran Liu, Edison Marrese-Taylor, Qingcheng Zeng, Yuhe Ke, Wanxin Li, Lechao Cheng, Qingyu Chen, James Caverlee, Yutaka Matsuo, and Irene Li. KG-rank: Enhancing large language models for medical QA with knowledge graphs and ranking techniques. In Dina Demner-Fushman, Sophia Ananiadou, Makoto Miwa, Kirk Roberts, and Junichi Tsujii (eds.), _Proceedings of the 23rd Workshop on Biomedical Natural Language Processing_ , pp. 155–166, Bangkok, Thailand, August 2024. Association for Computational Linguistics. URL https://aclanthology.org/2024.bionlp-1.13. 

- Michihiro Yasunaga, Hongyu Ren, Antoine Bosselut, Percy Liang, and Jure Leskovec. QA-GNN: Reasoning with language models and knowledge graphs for question answering. In Kristina Toutanova, Anna Rumshisky, Luke Zettlemoyer, Dilek Hakkani-Tur, Iz Beltagy, Steven Bethard, Ryan Cotterell, Tanmoy Chakraborty, and Yichao Zhou (eds.), _Proceedings of the 2021 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies_ , pp. 535–546, Online, June 2021. Association for Computational Linguistics. doi: 10.18653/v1/2021.naacl-main.45. URL https://aclanthology.org/2021. naacl-main.45. 

- Michihiro Yasunaga, Antoine Bosselut, Hongyu Ren, Xikun Zhang, Christopher D. Manning, Percy Liang, and Jure Leskovec. Deep bidirectional language-knowledge graph pretraining. In _Neural Information Processing Systems (NeurIPS)_ , 2022. 

- Jiahao Zhang, Haiyang Zhang, Dongmei Zhang, Liu Yong, and Shen Huang. End-to-end beam retrieval for multi-hop question answering. In Kevin Duh, Helena Gomez, and Steven Bethard (eds.), _Proceedings of the 2024 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies (Volume 1: Long Papers)_ , pp. 1718–1731, Mexico City, Mexico, June 2024. Association for Computational Linguistics. doi: 10.18653/v1/2024.naacl-long.96. URL https://aclanthology.org/2024. naacl-long.96. 

- Ruochen Zhao, Xingxuan Li, Shafiq Joty, Chengwei Qin, and Lidong Bing. Verify-and-edit: A knowledge-enhanced chain-of-thought framework. In Anna Rogers, Jordan Boyd-Graber, and Naoaki Okazaki (eds.), _Proceedings of the 61st Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)_ , pp. 5823–5840, Toronto, Canada, July 2023. Association for Computational Linguistics. doi: 10.18653/v1/2023.acl-long.320. URL https://aclanthology.org/2023.acl-long.320. 

- Chujie Zheng, Hao Zhou, Fandong Meng, Jie Zhou, and Minlie Huang. Large language models are not robust multiple choice selectors. In _The Twelfth International Conference on Learning Representations_ , 2023. 

15 

## APPENDIX 

|**Datasets**||**17**|
|---|---|---|
|Gold Standard Multi-choice Medical QA Dataset|. . . . . . . . . . . . . . . . . . . . .|17|
|Newly Created Medical QA Dataset . . . . . .|. . . . . . . . . . . . . . . . . . . . . .|17|
|Conversion of Multi-Choice Type Questions To Descriptive Type . . . . . . . . . . . . .||18|
|**Implementation Details**||**18**|
|Experiment Environments . . . . . . . . . . . .|. . . . . . . . . . . . . . . . . . . . . .|18|
|Fine-tuning Details . . . . . . . . . . . . . . .|. . . . . . . . . . . . . . . . . . . . . .|19|
|Implementation Details of Baseline Models<br>. .|. . . . . . . . . . . . . . . . . . . . . .|19|
|Implementation Details of Sensitivity Analysis|. . . . . . . . . . . . . . . . . . . . . .|20|
|Effect of Answer Order/Index<br>. . . . . .|. . . . . . . . . . . . . . . . . . . . . .|20|
|Implementation Details of Adaptability Analysis|. . . . . . . . . . . . . . . . . . . . . .|20|
|**Prompts**||**21**|
|Prompt Template for Evaluating Baseline Models Under Multi-choice Reasoning Setting||21|
|Prompt Template for Evaluating Baseline Models Under Open Reasoning Setting . . . .||21|
|Prompt Template in KGAREVION . . . . . . .|. . . . . . . . . . . . . . . . . . . . . .|22|
|**Tables**||**24**|
|Description Template . . . . . . . . . . . . . .|. . . . . . . . . . . . . . . . . . . . . .|24|
|Notations<br>. . . . . . . . . . . . . . . . . . . .|. . . . . . . . . . . . . . . . . . . . . .|25|



16 

## A DATASETS 

## A.1 GOLD STANDARD MULTI-CHOICE MEDICAL QA DATASET 

In this work, we use four well-known multi-choice medical QA datasets to evaluate the model performance, including two medical examination QA datasets (MMLU-Med, MedQA-US) and two biomedical research QA datasets (PubMedQA*, BioASQ-Y/N). These datasets are derived from (Xiong et al., 2024a). The samples in these datasets are shown in Table 4. 

|Dataset Name|Sample|
|---|---|
||Which of the following best describes the structure that collects urine in the|
|MMLU-Med|body?|
||A: Bladder B: Kidney C: Ureter D: Urethra|
||A microbiologist is studying the emergence of a virulent strain of the virus. After|
|MedQA-US|a detailed study of the virus and its life cycle, he proposes a theory: Initially,|
||a host cell is co-infected with 2 viruses from the same virus family. Within|
||the host cell, concomitant production of various genome segments from both|
||viruses occurs. Ultimately, the different genome segments from the viruses are|
||packaged into a unique and novel virus particle. The newly formed virus particle|
||is both stable and viable and is a new strain from the virus family that caused the|
||outbreak of infection. Which of the following viruses is capable of undergoing|
||the above-mentioned process?|
||A: Epstein-Barr virus B: Human immunodefciency virus C: Rotavirus D: Vac-|
||cinia virus|
||Is anorectal endosonography valuable in dyschesia?|
|PubMedQA*||
||A: yes B: no C: maybe|
||Can losartan reduce brain atrophy in Alzheimer’s disease?|
|BioASQ-Y/N||
||A:yes B: no|



Table 4: Examples of four widely used medical QA datasets 

## A.2 NEWLY CREATED MEDICAL QA DATASET – MEDDDX 

MedDDx is a newly constructed dataset designed to test model performance on semantically complex answers. The motivation behind creating this dataset is twofold: 

- While large language models (LLMs) can perform QA tasks, they often rely heavily on semantic dependencies, making it difficult for them to identify the correct answer among semantically similar answer candidates; 

- In real-world medical scenarios, researchers often focus on identifying subtle differences between similar molecules, particularly in treatment or diagnostic settings. For instance, proteins may share similar names but have significantly different structures and functions, making it crucial to distinguish these differences to be able to provide accurate answers (see Table 5 for an example). 

Because of these reasons, we construct MedDDx, a multi-choice medical QA dataset that focuses on answering semantically complex multi-choice QA. These questions are sourced from STaRK-Prime (Wu et al., 2024c), which provides both the questions and their corresponding answers. We extract questions with a single correct answer from the STaRK-Prime testing set and transform them into the multi-choice format. To generate three strong alternative answer candidates, we use semantic similarity to increase the difficulty, selecting the top three entities that have the highest semantic similarity as the correct answer. The semantic embeddings used for this process are derived from text-embedding-ada-002 model from OpenAI. 

17 

|Dataset Name|Sample|
|---|---|
||Can you recommend medications effective against peptic ulcer disease that also|
|Basic|suppress Helicobacter pylori in the stomach?|
||A: Rebamipide B: Ecabet C: Bendazac D: Nepafenac|
||Can you recommend medications that treat both eosinophilic pneumonia and a|
|Intermediate|parasitic worm infection?|
||A: Thiabendazole B: Albendazole C: Diethylcarbamazine D: Triclabendazole|
||Which genes or proteins are expressed exclusively in the pericardium and not in|
|Expert|either the dorsal or ventral regions of the thalamus?|
||A: ADH1A B: ADH1C C: ADH4 D: ADH1B|



Table 5: Examples of four widely used medical QA datasets. 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_Knowledge_Graph_Based_Agent_For_Complex,_Knowledge-Intensive_QA_in_Medicine_images/iclr_2025_Knowledge_Graph_Based_Agent_For_Complex,_Knowledge-Intensive_QA_in_Medicine.pdf-0018-02.png)


Figure 8: The distribution of the standard deviation of semantic similarities between answer candidates and the correct answer. A lower value indicates greater similarity among the answers. 

The semantic similarity is calculated using cosine similarity. We also compute the standard deviation of semantic similarity between the correct answer and the other three candidates. The density distribution of these values is shown in Fig. 8. Based on this distribution, we divide the queries into three complexity groups using quantile analysis: MedDDx-Expert (0-0.02), MedDDx-Intermediate (0.02-0.04), and MedDDx-Basic ( _>_ 0.04). 

## A.3 CONVERSION OF MULTI-CHOICE TYPE QUESTIONS TO DESCRIPTIVE TYPE 

The conversion of multi-choice questions to descriptive ones is aimed to evaluate real-world medical scenarios where open-ended inquiries are prevalent. To achieve this, we modify the question by adding more descriptive terms, as shown in Table 6. 

## B IMPLEMENTATION DETAILS 

## B.1 EXPERIMENT ENVIRONMENTS 

**Hardware.** All experiments are conducted on a machine equipped with 4 NVIDIA H100. We use 1 NVIDIA H100 to implement baselines with small LLMs. In the fine-tuning stage, we use 4 NVIDIA H100 to fine-tune the review module. 

18 

|Multi-Choice type|Open-ended type|||
|---|---|---|---|
|Which of the following best describes<br>the|What best describes|the structure that col-||
|structure that collects urine in the body?|lects urine in the body?|||
|A: Bladder B: Kidney C: Ureter D: Urethra||||
|A microbiologist is studying the emergence|A microbiologist is studying the emergence|||
|of a virulent strain of the virus.<br>Af-|of a virulent strain of the virus. After a de-|||
|ter a detailed study of the virus and its|tailed study of the virus and its life cycle, he|||
|life cycle,<br>he proposes a theory:<br>Ini-|proposes a theory: Initially, a host cell is co-|||
|tially, a host cell is co-infected with 2|infected with 2 viruses from the same virus|||
|viruses from the same virus family.<br>......|family. ......<br>Which virus||is capable of un-|
|Which of the following viruses<br>is capable|dergoing the above-mentioned process?|||
|of undergoing the above-mentioned process?||||
|A: Epstein-Barr virus B: Human immunode-||||
|fciencyvirus C: Rotavirus D: Vaccinia virus||||



Table 6: Examples of conversation of multi-choice type question to descriptive type. 

**Software.** We implement KGAREVION using Python 3.9.19, PyTorch 2.3.1, Transformers 4.43.1, and Tokenizers 0.19.1. All LLMs adopted in this study are downloaded from Hugging Face, except for OpenAI models. 

## B.2 FINE-TUNING DETAILS 

During the fine-tuning stage, we first split PrimeKG Chandak et al. (2023) into two parts: a training set and a testing set, in a ratio of 8:2. We use LoRA to fine-tune the LLMs on a single machine equipped with 4 NVIDIA H100 GPUs for knowledge graph completion tasks in the Review action. 

For hyperparameter tuning, we use grid search to identify the optimal parameter combinations by evaluating the fine-tuned model’s performance on the knowledge graph completion task using the testing set. Specifically, we focus on the parameter _r_ in LoRA training and the batch size during the fine-tuning stage. The values explored for _r_ are 16, 32, 64, 128, while the tested batch sizes _bz_ are 128, 256, 512, 1024. The best parameters identified are _r_ = 32, _bz_ = 256. 

## B.3 IMPLEMENTATION DETAILS OF BASELINE MODELS 

All the results reported in this paper are averaged over multiple runs with different random seeds. For consistency and reproducibility, we select three commonly used seeds: 42, 777, and 1234. 

## B.3.1 IMPLEMENTATION DETAILS OF BASELINE MODELS UNDER MULTI-CHOICE SETTING 

**LLM-based Reasoning Models.** We evaluate these LLMs across all datasets in two distinct settings: the traditional inference setting and the CoT inference setting, using the Transformers package (v 4.31.1) with default parameters. The prompts used for evaluation are provided in C.1. 

**KG-based Models.** Since the baseline models are trained in an end-to-end way, they cannot be applied directly. To ensure fairness, we train these three models on a collective dataset called MedMCQA, which contains 4182 question-answering samples, and then evaluate them on all datasets used in this study. In addition, we utilize PrimeKG as the knowledge base for all three methods to construct subgraphs for each sample in both the training and evaluation stages. For knowledge graph processing, we follow the same procedure as JointLK, converting each entity in the KG to its corresponding UMLS code and retrieving entities that match those in the question to construct the subgraph for each question. In the case of Dragon, since it requires a pre-training stage, we first complete its pre-training on the MedMCQA dataset and then directly infer the model on all datasets adopted in this study. We set the epoch number for all three models as 20. All other parameters were derived from the original publications. 

19 

**RAG-based Models.** We implement the Self-RAG with Llama2-7B/13B with VLLM (v 0.4.3). We set the temperature in _SamplingParams_ as the default value as their original LLMs. MedRAG is implemented using the original code repository. We adopt Llama2-70B as its backbone model, ’ _textbooks_ ’ as corpus, and MedCPT as retriever. KG-RAG is implemented with the same KG as in KGAREVION, which is PrimeKG and is utilized only during the inference stage. For KG-Rank, we report the results using its original backbone model, which is Llama2-13B and adopt the MedCPT as the ranking method due to limitations with the Cohere API. 

## B.3.2 IMPLEMENTATION DETAILS OF BASELINE MODELS UNDER OPEN-ENDED REASONING SETTING 

To test baseline models under the open-ended reasoning setting, we add a new module for both LLM-based reasoning models and RAG-based models. In terms of KG-based models, they could not be deployed under the open-ended reasoning setting due to their model architectures, as shown in C.2. 

For each model in the other two groups, we first get a response to the input descriptive query and then adopt the same LLM as their original design to match the response to the content of the correct answer without considering the input question. 

## B.4 IMPLEMENTATION DETAILS OF SENSITIVITY ANALYSIS 

## B.4.1 EFFECT OF ANSWER ORDER/INDEX 

To assess the sensitivity of each model to the answers’ order, we swap the positions of two options. For datasets with four options, the order is changed from ABCD to BCAD. For three-option datasets, the order is adjusted from ABC to CAB, and for two-option datasets, it is reversed from AB to BA. It is important to note that we alter the order along with the corresponding content, not just the answer labels. To test the model sensitivity to the answers’ index, we relabel the answer indices from ABCD to EFGH. The question obtained after changing the order or the index is presented in Table 7. 

Sample Which of the following best describes the structure that collects urine in the body? Original A: Bladder B: Kidney C: Ureter D: Urethra Which of the following best describes the structure that collects urine in the body? Changing Order B: Kidney C: Ureter A: Bladder D: Urethra Which of the following best describes the structure that collects urine in the body? Relabeling Index E: Bladder F: Kidney G: Ureter H: Urethra 

Table 7: Examples of four widely used medical QA datasets. 

## B.5 IMPLEMENTATION DETAILS OF ADAPTABILITY ANALYSIS 

To show the flexibility of KGAREVION, we further conduct adaptability analysis by implementing the model with different backbones and KGs. Overall, in this work, we test the KGAREVION with two KGs (i.e., PrimeKG and OGB-biokg). Their details are as follows: 

PrimeKG (Chandak et al., 2023) is a precision medicine-oriented knowledge graph that provides a holistic view of diseases. PrimeKG integrates 20 high-quality resources to describe 17,080 diseases with 4,050,249 relationships representing ten major biological scales, including disease-associated protein perturbations, biological processes and pathways, anatomical and phenotypic scale, and the entire range of approved and experimental drugs with their therapeutic action, considerably expanding previous efforts in disease-rooted knowledge graphs. 

The OGB-biokg (Hu et al., 2020) dataset contains 5 types of entities: diseases (10,687 nodes), proteins (17,499), drugs (10,533 nodes), side effects (9,969 nodes), and protein functions (45,085 

20 

nodes). There are 51 types of directed relations connecting two types of entities, including 38 kinds of drug-drug interactions, 8 kinds of protein-protein interaction, as well as drug-protein, drug-side effect, and function-function relations. 

## C PROMPTS 

- C.1 PROMPT TEMPLATE FOR EVALUATING BASELINE MODELS UNDER MULTI-CHOICE REASONING SETTING 

## Prompts for Evaluating LLMs 

The following is a multiple-choice medical question. Please select and provide the correct answer from options ‘A’, ‘B’, ‘C’ or ‘D’. 

Question: _{_ question _}_ 

Answer: 

## Prompts for Evaluating LLMs with CoT 

The following is a multiple-choice medical question. Let’s think step by step. Please select and provide the correct answer from options ‘A’, ‘B’, ‘C’ or ‘D’. 

Question: _{_ question _}_ 

Answer: 

- C.2 PROMPT TEMPLATE FOR EVALUATING BASELINE MODELS UNDER OPEN REASONING SETTING 

## Prompts for Evaluating LLMs 

The following are medical questions. Please generate a response for input question. Question: _{_ question _}_ 

Answer: 

## Prompts for Evaluating LLMs with CoT 

The following are medical questions. Let’s think step by step. Please generate a response for input question. 

Question: _{_ question _}_ 

Answer: 

## Prompts for Evaluating LLMs 

Given a context, please select the most match answer from options by using ’A’, ’B’, ’C’, and ’D’. 

Context: _{_ context _}_ 

Options: _{_ options _}_ Answer: 

21 

## C.3 PROMPT TEMPLATE IN KGAREVION 

The Generate action is implemented with two prompts. One is responsible for identifying medical concepts involved in question stem, the other is for generating triplets. 

## Prompts for Generate Action 

### Instruction: 

Given the following multiple-choice question, extract all relevant medical entities contained within the question stem. Identify and extract all medical entities, such as diseases, proteins, genes, drugs, phenotypes, anatomical regions, treatments, or other relevant medical entities. Ensure that the extracted entities are specific and medically relevant. If no medical entities are found in a particular part, return an empty list for that section. Only return the extracted entities in JSON format with the key ”medical ~~t~~ erminologies” and the value is a list of extracted entities. ### Input: 

Question: _{_ question _}_ 

### Response: 

## Prompts for Generate Action 

### Instruction: 

Given the following question stem, medical terminologies, and options, generate a set of related undirected triplets. Each triplet should consist of a head entity, a relation, and a tail entity. The relations should describe meaningful interactions or associations between the entities, particularly in a medical or biomedical context. Use the query stem and the medical entities contained each option to extract triplets that are relevant to the query and can answer the query correctly. Each triplet should be in the format: (Head Entity, Relationship, Tail Entity). Since the triplets are undirected, the order of Head Entity and Tail Entity does not imply any directional relationship between them. The relationship should be one of the following: [’protein protein’, ’carrier’, ’enzyme’, ’target’, ’transporter’, ’contraindication’, ’indication’, ’off-label use’, ’synergistic interaction’, ’associated with’, ’parentchild’, ’phenotype absent’, ’phenotype present’, ’side effect’, ’interacts with’, ’linked to’, ’expression present’, ’expression absent’]. Ensure that each entity in the triplet is specific and concise, such as diseases, proteins, conditions, symptoms, drugs, treatments, anatomical parts, or other relevant medical entities. 

Generate 1-3 triplets for each option, focusing on the ones most relevant to answering the query. 

Only return the generated triplets in a structured JSON format with the key as ”Triplets” and the value as a list of triplets. The format should be: ”Triplets”: [(Head Entity, Relationship, Tail Entity), (Head Entity, Relationship, Tail Entity)] 

### Input: 

Question: _{_ query ~~s~~ tem _}_ 

Medical ~~T~~ erminologies: _{_ medical terminologies _}_ Options: _{_ option _}_ ### Response: 

22 

## Prompts for Review Action 

Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request. 

## ### Instruction: 

Given a triple from a knowledge graph. Each triple consists of a head entity, a relation, and a tail entity. Taking (PHYHIP, protein ~~p~~ rotein, KIF15) as an example, it means that protein PHYHIP has an interaction with protein KIF15. Please determine the correctness of the triple and response True or False. Please directly output ’True’ or ’False’. 

### Input: 

_{_ triplet _}_ ### Response: 

## Prompts for Revise Action 

## ### Instruction: 

Given the following triplet consisting of a head entity, relation, and tail entity, please review and revise the triplet to ensure it is correct and helpful for answering given question. The revision should focus on correcting the head entity, relation, or tail entity as needed to make the triplet accurate and relevant. The triplet should follow the format (head entity, relation, tail entity). Ensure that the revised triplet is factually accurate and contextually appropriate. The relation should clearly define the relationship between the head entity and the tail entity. If no changes are necessary, return the original triplet. 

Only return the revised triplet in JSON format with the key ’Revised ~~T~~ riplets’ and the value as the corrected triplet. The format should be: ”Revised ~~T~~ riplets”: [(Head Entity, Relationship, Tail Entity)] 

### Input: 

Triplets: _{_ triplets _}_ 

Questions: _{_ question _}_ 

### Response: 

23 

## D TABLES 

## D.1 DESCRIPTION TEMPLATE 

|Relation|Description|
|---|---|
|protein<br>~~p~~rotein|Protein_{_A_}_interacts with protein_{_B_}_, indicating that the two proteins directly|
||or indirectly associate with each other to perform a biological function.|
|carrier|_{A}_ acts as a carrier for _{B}_, facilitating its transport or delivery to specifc|
||locations within the body or within a cell|
|enzyme|_{A}_functions as an enzyme that catalyzes a reaction involving_{B}_, converting|
||it into a different molecule or modifying its structure|
|target|_{A}_serves as a target for_{B}_, meaning that_{B}_binds to or interacts with_{A}_|
||to exert its biological effect.|
|transporter|_{A}_is a transporter that facilitates the movement of_{B}_across cellular mem-|
||branes or within different compartments of the body.|
|contraindication|The interaction between_{A}_and_{B}_is contraindicated, meaning that the pres-|
||ence of one molecule may have adverse effects or reduce the effcacy of the other|
|indication|_{A}_is indicated for the treatment or management of a condition associated with|
||_{B}_, suggesting that_{A}_has a therapeutic role related to_{B}_|
|off-label use|_{A}_is used off-label in relation to_{B}_, meaning it is utilized in a manner not|
||specifcally approved but based on clinical judgment.|
|synergistic interaction|_{A}_ and _{B}_ interact synergistically, where their combined effect is greater|
||than the sum of their individual effects|
|associated with|_{A}_is associated with_{B}_, indicating a relationship or correlation between the|
||two, often in the context of disease or biological processes|
|parent-child|_{A}_is related to_{B}_in a parent-child relationship, where_{A}_gives rise to or|
||infuences the formation of_{B}_|
|phenotype absent|The interaction between_{A}_and_{B}_results in the absence of a specifc phe-|
||notype, indicating that the normal trait is not expressed|
|phenotype present|The interaction between_{A}_and_{B}_results in the presence of a specifc phe-|
||notype, indicating that a particular trait is expressed|
|side effect|The interaction between_{A}_and_{B}_can cause a side effect, where the pres-|
||ence of one molecule leads to unintended and possibly adverse effects|
|interacts with|_{A}_interacts with_{B}_, indicating a general interaction that may involve bind-|
||ing, modulation, or other forms of molecular communication.|
|linked to|_{A}_ is linked to _{B}_, suggesting a connection or association between the two|
||molecules, often in a biological or pathological context.|
|expression present|_{A}_is expressed in the presence of_{B}_, indicating that the existence or activity|
||of_{B}_leads to or correlates with the expression of_{A}_|
|expression absent|_{A}_ is not expressed in the presence of _{B}_, indicating that the existence or|
||activityof _{B}_suppresses or does not correlate with the expression of _{A}_|



Table 8: The description templates 

24 

## D.2 NOTATIONS 

|**Variable**|**Description**|
|---|---|
|Q|A set of medical queries|
|q|One question stem in Q|
|C|A set of answer candidates for one question stem|
|a|Correct answer in C for the question q|
|P|A large language model|
|G|Knowledge graph|
|h|A head entity in one triplet|
|r|A relationship in one triplet|
|t|A tail entity in one triplet|
|T|A set of triplets generated in the Generation action|
|_ai_|One answer candidate in C|
|M|A set of medical concepts in q|
|**e**_h_|Pre-trained embeddings of h|
|**e**_r_|Pre-trained embeddings of r|
|**e**_t_|Pre-trained embeddings of t|
|d|Dimension of pre-trained embeddings|
|f|The fne-tuned model used in Review action|
|b|Bool value that determining the correctness of triplet|
|D|The description dictionary for relationship r|
|_|l|_|The max length of description tokens|
|_dp_|The dimension of token embeddings in LLM|
|**X**|Token embedding matrix with the shape of_|l| × dp_ obtained from P|
|g|A linear layer to map the dimension of pre-trained embeddings_d_to that of token embeddings_dp_|
|**V**|The triplet embedding matrix|
|**Z**|Aligned triplet embedding matrix|
|_σ_(_·_)|TheSoftmaxfunction|
|_φ_(_·_)|The layer normalization function|
|**W**1 and**W**2|Trainable parameters in the two layer forward neural network|
|_dh_|The dimension of hidden layers in the two layer forward neural network|
|s|An instruction to the LLM|
|V|True triplet set determined by Review action|
|F|False triplet set determined by Review action|
|k|Max round of iterative review and revise actions|
|y|Predicted answer from the Answer action|



Table 9: Additional notation. 

25 

