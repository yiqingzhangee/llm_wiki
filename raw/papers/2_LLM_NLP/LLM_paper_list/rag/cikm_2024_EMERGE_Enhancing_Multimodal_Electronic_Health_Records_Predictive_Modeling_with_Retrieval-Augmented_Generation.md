# **EMERGE: Enhancing Multimodal Electronic Health Records Predictive Modeling with Retrieval-Augmented Generation** 

Yinghao Zhu[∗] Beihang University Peking University Beijing, China yhzhu99@gmail.com 

Changyu Ren[∗] Beihang University Beijing, China cyren@buaa.edu.cn 

Shiyun Xie Beihang University Beijing, China xieshiyun@buaa.edu.cn 

Junlan Feng China Mobile Research Institute Beijing, China fengjunlan@chinamobile.com 

Liantao Ma Peking University Beijing, China malt@pku.edu.cn 

## **Abstract** 

The integration of multimodal Electronic Health Records (EHR) data has significantly advanced clinical predictive capabilities. Existing models, which utilize clinical notes and multivariate time-series EHR data, often fall short of incorporating the necessary medical context for accurate clinical tasks, while previous approaches with knowledge graphs (KGs) primarily focus on structured knowledge extraction. In response, we propose EMERGE, a Retrieval-Augmented Generation (RAG) driven framework to enhance multimodal EHR predictive modeling. We extract entities from both time-series data and clinical notes by prompting Large Language Models (LLMs) and align them with professional PrimeKG, ensuring consistency. In addition to triplet relationships, we incorporate entities’ definitions and descriptions for richer semantics. The extracted knowledge is then used to generate task-relevant summaries of patients’ health statuses. Finally, we fuse the summary with other modalities using an adaptive multimodal fusion network with cross-attention. Extensive experiments on the MIMIC-III and MIMIC-IV datasets’ in-hospital mortality and 30-day readmission tasks demonstrate the superior performance of the EMERGE framework over baseline models. Comprehensive ablation studies and analysis highlight the efficacy of each designed module and robustness to data sparsity. 

∗Equal contribution. 

†Corresponding author. 

Permission to make digital or hard copies of all or part of this work for personal or classroom use is granted without fee provided that copies are not made or distributed for profit or commercial advantage and that copies bear this notice and the full citation on the first page. Copyrights for components of this work owned by others than the author(s) must be honored. Abstracting with credit is permitted. To copy otherwise, or republish, to post on servers or to redistribute to lists, requires prior specific permission and/or a fee. Request permissions from permissions@acm.org. _CIKM ’24, October 21–25, 2024, Boise, ID, USA._ 

© 2024 Copyright held by the owner/author(s). Publication rights licensed to ACM. ACM ISBN 979-8-4007-0436-9/24/10 https://doi.org/10.1145/3627673.3679582 

Xiaochen Zheng ETH Zürich Zürich, Switzerland xzheng@ethz.ch 

Zixiang Wang Peking University Beijing, China wangzx@stu.pku.edu.cn 

Xi Zhu Zhoujun Li China Mobile Research Beihang University Institute Beijing, China Beijing, China lizj@buaa.edu.cn zhuqian@chinamobile.com 

Chengwei Pan[†] Beihang University Zhongguancun Laboratory Beijing, China pancw@buaa.edu.cn 

EMERGE contributes to refining the utilization of multimodal EHR data in healthcare, bridging the gap with nuanced medical contexts essential for informed clinical predictions. We have publicly released the code at https://github.com/yhzhu99/EMERGE. 

## **CCS Concepts** 

• **Applied computing** → **Health informatics** ; • **Information systems** → **Data mining** . 

## **Keywords** 

electronic health record; multimodal learning; large language model; retrieval-augmented generation 

## **ACM Reference Format:** 

Yinghao Zhu, Changyu Ren, Zixiang Wang, Xiaochen Zheng, Shiyun Xie, Junlan Feng, Xi Zhu, Zhoujun Li, Liantao Ma, and Chengwei Pan. 2024. EMERGE: Enhancing Multimodal Electronic Health Records Predictive Modeling with Retrieval-Augmented Generation. In _Proceedings of the 33rd ACM International Conference on Information and Knowledge Management (CIKM ’24), October 21–25, 2024, Boise, ID, USA._ ACM, New York, NY, USA, 11 pages. https://doi.org/10.1145/3627673.3679582 

## **1 Introduction** 

The advent of Electronic Health Records (EHR) marks a pivotal advancement in the way patient data is gathered and analyzed, contributing to a more effective and informed healthcare delivery system for clinical prediction [15, 23, 27]. This advancement is largely attributed to the utilization of multimodal EHR data, which primarily includes clinical notes and multivariate time-series data from patient records [39, 49, 50]. Such data types are integral to healthcare prediction tasks, mirroring the holistic approach practitioners adopt by leveraging various patient data points to inform their clinical decisions and treatment strategies, rather than depending on a single data source [43]. Deep learning-based methods 

CIKM ’24, October 21–25, 2024, Boise, ID, USA. 

Yinghao Zhu et al. 

have become the mainstream approach, processing multimodal data to learn a mapping from heterogeneous inputs to output labels [8, 26, 49]. However, in contrast to healthcare professionals, who have a deep understanding of medical contexts through extensive experience and knowledge, neural networks trained from scratch lack these insights into medical concepts [30]. Without deliberate integration of external knowledge, these networks often lack the ability or sensitivity to recognize crucial disease entities or laboratory test results within the EHR, essential for accurate prediction tasks [53]. In response, some recent studies have begun incorporating knowledge graphs to infuse additional medical insights into their analyses [14, 46]. These graphs offer a supplementary layer of clinically relevant concepts, thereby enhancing the model’s ability to provide contextually meaningful representations and interpretable evidence [45]. Despite these advancements, significant limitations remain in fully linking external knowledge with multiple EHR modalities, underscoring the imperative need for continuous research to integrate multi-source insights and improve the multimodal EHR data predictive modeling. 

Previous methods integrating external medical knowledge into EHR data analysis tend to extract knowledge from data modalities such as ICD disease codes, patient conditions, procedures, and drugs, neglecting the use of clinical notes and time-series data, which are more common and practical [35] ( **Limitation 1** ). Additionally, these methods primarily extract hierarchical and structured knowledge from clinical-context knowledge graphs. However, these medical concepts—entity names and their relationships into a graph have limited direct contribution to predictive tasks ( **Limitation 2** ). With Large Language Models (LLMs) like GPT-4 [1] demonstrating strong capabilities in diverse clinical tasks [36, 40, 53] and serving as large medical knowledge graphs (KGs) [37]. By prompting the LLM, GraphCare [17] constructs a GPT-KG using structured condition, procedure, and drug record data, represented as triples (entity 1, relationship, entity 2). It further employs graph neural networks for downstream tasks. However, this approach encounters the hallucination issue [51], where LLMs may generate incorrect or fabricated information. To mitigate this, GraphCare collaborates with medical professionals to scrutinize and remove potentially harmful content, a process that is both complex and labor-intensive, requiring significant expertise to validate and refine the generated triples. Moreover, directly generating the KG via LLMs introduces a domain gap since this task is likely untrained for the LLMs, leading to potentially lower accuracy compared to professional knowledge graphs built through established methodologies ( **Limitation 3** ). 

To overcome these limitations, we propose utilizing LLMs in a Retrieval-augmented Generation (RAG) approach [21]. The RAG framework integrates structured time-series EHR data, unstructured clinical notes, and an established KG (PrimeKG [6]) with LLM’s semantic reasoning capabilities [38]. The LLMs are prompted to generate comprehensive summaries of patients’ health statuses, and these summaries are then fused for downstream tasks. Despite its apparent simplicity, applying this method to clinical tasks presents several technical challenges: 

challenging. Moreover, unlike structured codes where it can directly compare the code-related entities’ embedding with KG’s entity, the entities extracted by LLM have hallucination issues. Accurately matching extracted entities with those in an external knowledge graph while eliminating the potential for hallucinations posed by LLMs is crucial for maintaining the integrity and reliability of the clinical prediction tasks [16]. 

**Challenge 2: How to encode and incorporate long-text retrieved knowledge with task-relevant characteristics?** The extracted textual knowledge likely contains too many tokens [42] for conventional language model inputs (e.g., BERT supports only 512 tokens [11]). However, with the development of long-context LLMs [52], it is feasible to leverage LLMs to distill this knowledge further. Additionally, simply integrating the retrieved knowledge may not be task-specific, creating a gap between the knowledge and downstream tasks [3–5]. Therefore, a task-relevant prompting strategy [31] is necessary during the LLM distillation process. 

To these ends, We propose EMERGE framework to address the above limitations and challenges with the following approaches, which are our three-fold contributions: 

- (1) We design a RAG-driven multimodal EHR enhancement framework for clinical notes and time-series EHR data ( **Response to Limitation 1** ). EMERGE leverages the capabilities of LLMs and professionally labeled large medical knowledge graphs. We retrieve medical entities by prompting the LLM for clinical notes and using z-score-based filtering for time-series data, then match them in KG with post-validation and alignment to mitigate hallucination ( **Response to Limitation 3** ). In addition to triples of entities, we also include more knowledge by extending the entities’ definition and description. ( **Response to Limitation 2** ). 

- (2) Methodologically, we first compare LLM-generated entities with original clinical notes to ensure the entities appear in the raw text. We then compute their embeddings and cosine similarities among extracted entities and KG entities, aligning the entities through threshold-based filtering. This ensures that the overall entity extraction and matching process adheres to clinical standards with consistency guarantees ( **Response to Challenge 1** ). We prompt the long-context LLM to summarize the extracted knowledge into a distilled reflection of the patient’s health status, instructing the generated content is task-relevant. To integrate the extracted knowledge and consider heterogeneity, we design an adaptive multimodal fusion network with a cross-attention mechanism that attentively fuses each modality’s representation ( **Response to Challenge 2** ). 

- (3) Experimentally, our extensive experiments on the MIMIC-III and MIMIC-IV datasets, focusing on in-hospital mortality and 30-day readmission tasks, demonstrate EMERGE’s superior performance and the effectiveness of each designed module. Additionally, to meet practical clinical needs, we evaluate the model’s robustness with fewer training samples, showing EMERGE’s remarkable resilience against data sparsity. 

**Challenge 1: How to extract entities from multimodal EHR data and match these entities with external KG consistently?** Extracting entities from the diverse and complex formats of EHR data (including clinical notes and multivariate time-series data) is 

CIKM ’24, October 21–25, 2024, Boise, ID, USA. 

EMERGE: Enhancing Multimodal EHR Predictive Modeling with RAG 

## **2 Related Work 2.1 Multimodal EHR Learning** 

Advances in medical technology enable analysis of various medical modalities, including clinical notes, time-series lab data, demographics, conditions, procedures, drugs, and imaging. MedGTX [32] introduces a pre-trained model for joint multi-modal representation learning, interpreting structured data as a graph and using a graph-text multi-modal framework. M3Care [49] addresses missing modalities by imputing task-related information in the latent space with auxiliary data from similar patients, employing a modalityadaptive similarity metric to handle missing data. Zhang et al. [50] explore irregular time intervals in time-series EHR data and clinical notes via a time attention mechanism. Xu et al. [43] propose a joint learning approach from visit sequences and clinical notes, using Gromov-Wasserstein Distance for contrastive learning and dual-channel retrieval to enhance patient similarity analysis. Lee et al. [20] introduce a unified framework for learning across all EHR modalities with modality-aware attention mechanisms, avoiding separate imputation modules. 

Despite their effectiveness, these methods often overlook clinical background information, where external medical knowledge could enhance EHR data insights. The absence of semantic medical knowledge also complicates the training pipeline, especially with limited data. 

raw data matrix or via a tokenization process. Specifically, the multivariate time-series data, denoted as _𝒙𝑇𝑆_ ∈ R _[𝑇]_[×] _[𝐹]_ , encapsulate information across _𝑇_ visits and _𝐹_ numeric or categorical features. Clinical notes, denoted as _𝒙𝑁𝑜𝑡𝑒_ , contain recorded notes documenting the health status of each patient. Additionally, external knowledge graphs (KGs) are incorporated to enhance the personalized representation of each patient. 

## **3.2 Predictive Objective Formulation** 

The prediction objective is conceptualized as a binary classification task, which involves predicting in-hospital mortality and 30-day readmission. By leveraging the comprehensive patient information derived from EHR data and KGs, the model aims to predict specific clinical outcomes. The prediction task is formulated as: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/rag/cikm_2024_EMERGE_Enhancing_Multimodal_Electronic_Health_Records_Predictive_Modeling_with_Retrieval-Augmented_Generation_images/cikm_2024_EMERGE_Enhancing_Multimodal_Electronic_Health_Records_Predictive_Modeling_with_Retrieval-Augmented_Generation.pdf-0003-08.png)


where _𝑦_ ˆ represents the targeted prediction outcome. 

For the in-hospital mortality prediction task, our objective is to determine the discharge status based on data from the initial 48-hour window of an ICU stay, where a status of 0 indicates the patient is alive and 1 indicates the patient is deceased. In the same vein, the 30-day readmission task aims to predict whether a patient will be readmitted within 30 days after discharge, with 0 indicating no readmission and 1 indicating readmission. 

## **4 Methodology** 

## **2.2 Incorporating External Knowledge for EHR** 

To integrate clinical knowledge with EHR data, several studies leverage medical knowledge graphs (KGs) to enhance EHR representation learning and predictive performance. GRAM [8] uses hierarchical medical ontologies via a graph attention network to refine medical representations. KAME [26] embeds ontology information throughout the prediction process, enriching contextual understanding. MedPath [46] employs graph neural networks to integrate high-order connections from KGs into input representations. MedRetriever [47] enhances health risk prediction and interpretability by combining EHR embeddings with features from disease-specific documents. Collaborative graph learning models like CGL [25] explore patient-disease interactions and domain knowledge, while KerPrint [45] addresses knowledge decay across multiple visits. Recent advancements in Large Language Models (LLMs) as comprehensive knowledge bases [37] offer new possibilities, as seen in GraphCare [17], which creates a KG from structured EHR data for GNN learning, despite challenges like hallucination. 

These studies primarily focus on structured medical data, often neglecting the rich semantic information in unstructured EHR data. This limitation underscores the need for methods that comprehensively utilize both structured and unstructured data. 

## **3 Problem Formulation** 

## **3.1 EHR Datasets Formulation** 

The electronic health records (EHR) dataset comprises both structured and unstructured data, represented as multivariate time-series data and clinical notes, respectively. To facilitate analysis, these two modalities are initially processed separately, either from the 

Figure 1 shows the overall framework architecture of EMERGE. 

## **4.1 Multimodal EHR Embedding Extraction** 

We delve into the techniques used for embedding extraction from multimodal EHR, emphasizing the transformation from raw, humanreadable inputs, denoted as _𝒙_ , to deep semantic embeddings _𝒉_ for comprehensive analysis. 

When dealing with time-series data, we employ the Gated Recurrent Unit (GRU) network as the encoder. GRU is a highly efficient variant of recurrent neural networks, capable of capturing the time dependencies in sequence data and encoding this temporally linked information. We extract the representation of time-series data as follows: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/rag/cikm_2024_EMERGE_Enhancing_Multimodal_Electronic_Health_Records_Predictive_Modeling_with_Retrieval-Augmented_Generation_images/cikm_2024_EMERGE_Enhancing_Multimodal_Electronic_Health_Records_Predictive_Modeling_with_Retrieval-Augmented_Generation.pdf-0003-22.png)


where _𝑥𝑇𝑆_ is the time-series data and _ℎ𝑇𝑆_ denotes the output of the time-series encoder. 

As for text records, we utilize a medical domain language model to obtain text embeddings, represented as TextEncoder: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/rag/cikm_2024_EMERGE_Enhancing_Multimodal_Electronic_Health_Records_Predictive_Modeling_with_Retrieval-Augmented_Generation_images/cikm_2024_EMERGE_Enhancing_Multimodal_Electronic_Health_Records_Predictive_Modeling_with_Retrieval-Augmented_Generation.pdf-0003-25.png)


where _𝒙𝑁𝑜𝑡𝑒_ are the textual clinical notes and _ℎ𝑁𝑜𝑡𝑒_ denotes the note representation. 

## **4.2 RAG-Driven Enhancement Pipeline** 

_4.2.1 Extract Entities from Multimodal EHR Data._ To exploit the expert information encapsulated within the knowledge graph, it is necessary to extract disease entities from both time-series data and clinical notes, and subsequently align them with the information present in the graph. The set of disease entities in the time-series data is denoted as _𝑬𝑇𝑆_ , while those in the clinical notes data are 

CIKM ’24, October 21–25, 2024, Boise, ID, USA. 

Yinghao Zhu et al. 


![](raw/papers/2_LLM_NLP/LLM_paper_list/rag/cikm_2024_EMERGE_Enhancing_Multimodal_Electronic_Health_Records_Predictive_Modeling_with_Retrieval-Augmented_Generation_images/cikm_2024_EMERGE_Enhancing_Multimodal_Electronic_Health_Records_Predictive_Modeling_with_Retrieval-Augmented_Generation.pdf-0004-02.png)


**----- Start of picture text -----**<br>
Visit 1 Visit i Visit T ℎ𝑇𝑆<br>Feature 1  ↑ Feature 1  ↑ Feature 1  ↑<br>Feature 2 ↓ Feature 2  ↑ Feature 2 ↓ GRU<br>𝑥𝑇𝑆 Feature 3 ↑ …… Feature 3 ↑ …… Feature 3  ↑<br>Extract Abnormal Features (by z-score)<br>Retrieval Generation<br>Extracted Entities fromTime Series &   -  Role & Instruction Prompt ℎ𝑅𝐴𝐺<br>Clinical Notes Integrate - Extracted Entities Multimodal<br>- Retrieved Nodes & Triplets LM Z<br>Entity Embedding Similarity Fusion<br>𝐾𝐺 LLM<br>Threshold Filtered<br>Match Entities in<br>Knowledge Graph Relevant InfoEntities &  Summary of health status Response<br>Extract Disease Names (by LLM) ℎ𝑁𝑜𝑡𝑒<br>The patient presents with                          accompanied by                           … disease 1 disease 2 LM<br>𝑥𝑁𝑜𝑡𝑒<br>**----- End of picture text -----**<br>


**Figure 1:** _**Overall architecture of our proposed**_ **EMERGE** _**framework.**_ **The modules enclosed within the dashed box illustrate the RAG-driven enhancement pipeline. “LM” denotes Language Model (basically BERT-based model), while “LLM” in this paper normally refers to the GPT-based Large Language Model.** 

denoted as _𝑬𝑁𝑜𝑡𝑒_ . Naturally, we design two separate processes tailored to each modality. 

_Retrieval process for time-series data._ Time-series data is a structured format encompassing feature names and resultant values post-clinical examination. Each feature name reflects specific aspects of an individual’s physical condition, highlighting the deviations from the reference range. As shown in Figure 2, the specified record showcases low blood pressure and high blood urea nitrogen, significantly surpassing the normal range. This implies the potential risk of hypotension and uremia for the patient. Indeed, such feature names occur in disease definitions and descriptions, typically indicating serious health threats. 

For each patient, there are usually more than one entity (or abnormal feature), and some may be missing values. Consequently, our focus is primarily on non-empty values. For each feature _𝒙𝑇𝑆𝑖_ , we can identify outliers through the z-score method [9], which measures anomalies by calculating the deviation of data points from the mean, using standard deviation as a unit as below: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/rag/cikm_2024_EMERGE_Enhancing_Multimodal_Electronic_Health_Records_Predictive_Modeling_with_Retrieval-Augmented_Generation_images/cikm_2024_EMERGE_Enhancing_Multimodal_Electronic_Health_Records_Predictive_Modeling_with_Retrieval-Augmented_Generation.pdf-0004-07.png)


where _𝑠𝑖_ represents the z-score of the _𝑖_ -th feature of a patient. Features over a specified threshold _𝜖_ (such as 3- _𝜎_ deviation) are identified as abnormal, indicating potential health issues. 

_Retrieval process for clinical notes._ Unlike structured data, clinical notes are presented in a textual format, which makes it challenging to comprehend and extract valuable information. However, LLMs have exhibited exceptional performance on natural language understanding tasks, including named entity recognition (NER). Therefore, we utilize an LLM to identify potential disease names 


![](raw/papers/2_LLM_NLP/LLM_paper_list/rag/cikm_2024_EMERGE_Enhancing_Multimodal_Electronic_Health_Records_Predictive_Modeling_with_Retrieval-Augmented_Generation_images/cikm_2024_EMERGE_Enhancing_Multimodal_Electronic_Health_Records_Predictive_Modeling_with_Retrieval-Augmented_Generation.pdf-0004-10.png)


**----- Start of picture text -----**<br>
Feature name Feature value Abnormal<br>Heart rate 90 bpm √<br>Time Series<br>Blood pressure 40 mmHg ↓<br>Blood urea nitrogen 50 mg/dL ↑<br>Potential<br>Abnormal blood pressure  too low blood urea nitrogen  too high<br>Features<br>Hypotension Disorder<br>[definition]  A condition where the blood<br>pressure in the arteries is lower than normal.<br>[description]  This can lead to inadequate blood<br>flow to the organs, causing symptoms such as<br>Disease  dizziness, fainting, fatigue, and shock...<br>Definitions &<br>Descriptions Uremia<br>[definition]  A condition characterized by the<br>buildup of waste products and excess…<br>[description]  It is a common complication of<br>advanced kidney disease, where the kidneys are<br>not able to filter and remove toxic substances…<br>hypotension  uremia<br>disorder kidney<br>Disease failure<br>Relationships cardiovascular  uremic<br>disease neuropathy<br>Phenazopyridine<br>**----- End of picture text -----**<br>


**Figure 2:** _**Process of information retrieval for time-series data.**_ 

that the patient may have, as shown in Figure 4. Moreover, we implement specified rules for effective post-processing. 

- (1) **Entities Extraction:** We include an example and provide clear instructions in the prompt (Figure 3), instructing the LLM to concentrate on disease entities that the patient may suffer from. 

CIKM ’24, October 21–25, 2024, Boise, ID, USA. 

EMERGE: Enhancing Multimodal EHR Predictive Modeling with RAG 


![](raw/papers/2_LLM_NLP/LLM_paper_list/rag/cikm_2024_EMERGE_Enhancing_Multimodal_Electronic_Health_Records_Predictive_Modeling_with_Retrieval-Augmented_Generation_images/cikm_2024_EMERGE_Enhancing_Multimodal_Electronic_Health_Records_Predictive_Modeling_with_Retrieval-Augmented_Generation.pdf-0005-02.png)


**Figure 3:** _**Prompt template for extracting entities.**_ 

Sometimes, there may be no entities yielded in a single invocation, so we utilize multiple rounds to incrementally expand the current extracted entity set as shown below: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/rag/cikm_2024_EMERGE_Enhancing_Multimodal_Electronic_Health_Records_Predictive_Modeling_with_Retrieval-Augmented_Generation_images/cikm_2024_EMERGE_Enhancing_Multimodal_Electronic_Health_Records_Predictive_Modeling_with_Retrieval-Augmented_Generation.pdf-0005-05.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/rag/cikm_2024_EMERGE_Enhancing_Multimodal_Electronic_Health_Records_Predictive_Modeling_with_Retrieval-Augmented_Generation_images/cikm_2024_EMERGE_Enhancing_Multimodal_Electronic_Health_Records_Predictive_Modeling_with_Retrieval-Augmented_Generation.pdf-0005-06.png)


where _𝑷𝑒𝑥𝑡𝑟𝑎𝑐𝑡_ represents the prompt template. _𝑬𝑁𝑜𝑡𝑒[𝑖]_[repre-] sents the entity set obtained in the _𝑖_ -th round and _𝑬𝑁𝑜𝑡𝑒_ represents the aggregate set. 

- (2) **Entities Refinement:** Considering the hallucination issue associated with LLMs, we design a post-processing process to address it. This process consists of three primary steps: first, we discard entities that do not appear in the original text; second, we leverage an LLM to filter entities not in the disease type; and finally, we delete duplicated entities to prevent semantic redundancy. 


![](raw/papers/2_LLM_NLP/LLM_paper_list/rag/cikm_2024_EMERGE_Enhancing_Multimodal_Electronic_Health_Records_Predictive_Modeling_with_Retrieval-Augmented_Generation_images/cikm_2024_EMERGE_Enhancing_Multimodal_Electronic_Health_Records_Predictive_Modeling_with_Retrieval-Augmented_Generation.pdf-0005-09.png)


where _𝑬𝑖𝑙𝑙𝑒𝑔𝑎𝑙_ denotes the illegal entity set, which we then remove from _𝑬𝑁𝑜𝑡𝑒_ . 

To ensure the quantity and quality of the extracted entities, we execute steps 1 and 2 iteratively until achieving convergence. 

_4.2.2 Retrieve Information from External KG._ To ensure an accurate match between the extracted entities and nodes within the knowledge graph, we adapt a semantic-based dense vector retrieval approach. Initially, we utilize a sentence embedding model denoted as TextEncoder to encode all KG nodes, denoted as _𝑁𝑜𝑑𝑒𝑠_ . Subsequently, for each entity in _𝑬𝑇𝑆_ or _𝑬𝑁𝑜𝑡𝑒_ , we deploy the same embedding model to encode them. This process ensures that all embeddings are aligned within the same vector space, as shown below: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/rag/cikm_2024_EMERGE_Enhancing_Multimodal_Electronic_Health_Records_Predictive_Modeling_with_Retrieval-Augmented_Generation_images/cikm_2024_EMERGE_Enhancing_Multimodal_Electronic_Health_Records_Predictive_Modeling_with_Retrieval-Augmented_Generation.pdf-0005-13.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/rag/cikm_2024_EMERGE_Enhancing_Multimodal_Electronic_Health_Records_Predictive_Modeling_with_Retrieval-Augmented_Generation_images/cikm_2024_EMERGE_Enhancing_Multimodal_Electronic_Health_Records_Predictive_Modeling_with_Retrieval-Augmented_Generation.pdf-0005-14.png)


where _𝑛_ and _𝑒_ symbolize disease entities from _𝑁𝑜𝑑𝑒𝑠_ and the extracted entity set, respectively. _ℎ𝑛_ and _ℎ𝑒_ denote their corresponding embeddings. 

When matching relative nodes, we take the current entity _𝑒_ (including abnormal features and potential disease names) as the 


![](raw/papers/2_LLM_NLP/LLM_paper_list/rag/cikm_2024_EMERGE_Enhancing_Multimodal_Electronic_Health_Records_Predictive_Modeling_with_Retrieval-Augmented_Generation_images/cikm_2024_EMERGE_Enhancing_Multimodal_Electronic_Health_Records_Predictive_Modeling_with_Retrieval-Augmented_Generation.pdf-0005-17.png)


**----- Start of picture text -----**<br>
4:25 pm chest clip… impression : right seven<br>Clinical through nine posterolateral rib deformities ,<br>Notes new from prior radiograph dated .  Cardiome-<br>galy  with mild  pulmonary edema  superimposed<br>on background  emphysema  …<br>Potential<br>pulmonary edema emphysema cardiomegaly<br>Diseases (unretrieved)<br>Pulmonary Edema<br>[definition]  a condition in which fluid accumu-<br>lates in the air sacs of the lungs.<br>[description]  It can be caused by a variety of<br>Disease factors, including heart failure, high altitude...<br>Definitions &<br>Descriptions Pulmonary Emphysema<br>[definition]  A chronic lung disease that is part<br>of a group of conditions known as COPD.<br>[description]  The primary cause of emphysema<br>is long-term exposure to cigarette smoke…<br>Glimepiride pulmonary<br>emphysema compensatory<br>pulmonary  emphysema<br>Disease edema<br>Relationships<br>Trimethaphan<br>acute respiratory  TGFBR1 Endotoxins<br>failure<br>**----- End of picture text -----**<br>


**Figure 4:** _**Process of information retrieval for textual clinical notes.**_ **The grey block in potential diseases means no corresponding node found in external KG.** 

query. Then, we compute the similarities between _𝑒_ and each node in the KG. The metric used for these calculations is cosine similarity: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/rag/cikm_2024_EMERGE_Enhancing_Multimodal_Electronic_Health_Records_Predictive_Modeling_with_Retrieval-Augmented_Generation_images/cikm_2024_EMERGE_Enhancing_Multimodal_Electronic_Health_Records_Predictive_Modeling_with_Retrieval-Augmented_Generation.pdf-0005-20.png)


where _𝒉𝑒_ and _𝒉𝑛_ are embeddings of the entity _𝑒_ and the node. 

We establish a threshold to gauge the requisite similarity between two embeddings. We focus on nodes that surpass this threshold, ensuring that only the most relevant matches are considered: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/rag/cikm_2024_EMERGE_Enhancing_Multimodal_Electronic_Health_Records_Predictive_Modeling_with_Retrieval-Augmented_Generation_images/cikm_2024_EMERGE_Enhancing_Multimodal_Electronic_Health_Records_Predictive_Modeling_with_Retrieval-Augmented_Generation.pdf-0005-23.png)


where _𝑛_ ˆ = arg max _𝑛_ ∈Nodes _𝜃𝑒[𝑛]_ , _𝜂_ is the threshold for similarity, and _𝑓_ ( _𝑒,_ Nodes) denotes the set of nodes that we exclusively accept as matches for the entity _𝑒_ . 

Subsequently, we can obtain the definitions and descriptions within the disease entities, each represented as a node of the graph. Furthermore, relationships between diseases, encapsulated within triples, act as the edges of the graph. These pieces of information elaborate on the severity of the diseases, the harm they pose to the human body, and their interconnections from various perspectives. They further clarify the entity information from the original notes, thereby enhancing the LLM’s understanding of the patient’s health condition. 

_4.2.3 Summarize and Encode KG Knowledge._ Drawing from the entities extracted from time-series data and clinical notes, along with supplementary information about them, we have compiled extensive details about the patient’s medical condition. However, 

CIKM ’24, October 21–25, 2024, Boise, ID, USA. 

Yinghao Zhu et al. 

this content contains too many tokens for conventional language model inputs (such as BERT). As a countermeasure, we utilize retrieval-augmented generation to condense the aforementioned details, thereby attaining a concise representation of the patient’s health status. 

The prompt template, as illustrated in Figure 5, begins by defining a role and instructions to guide the generation by the LLM. Subsequently, we enumerate all abnormal features derived from the time-series data, and disease names extracted from clinical notes, which reflect the patient’s health threats. To enhance comprehension, we integrate retrieved disease definitions and descriptions, along with the relationships sampled from the KG to form a comprehensive supplementary resource. Based on this augmented information, the LLM compiles a summary of the patient’s health status. 


![](raw/papers/2_LLM_NLP/LLM_paper_list/rag/cikm_2024_EMERGE_Enhancing_Multimodal_Electronic_Health_Records_Predictive_Modeling_with_Retrieval-Augmented_Generation_images/cikm_2024_EMERGE_Enhancing_Multimodal_Electronic_Health_Records_Predictive_Modeling_with_Retrieval-Augmented_Generation.pdf-0006-04.png)


**----- Start of picture text -----**<br>
𝑍 Cross Attention<br>Feed Forward BatchNorm<br>Concat<br>Feed Forward<br>Cross Attention Cross Attention<br>Batch Norm<br>ℎ𝑇𝑒𝑥𝑡 ℎ𝑇𝑆<br>Multi-Head<br>Text Fusion Attention<br>𝑞 𝑘 𝑣<br>ℎ𝑁𝑜𝑡𝑒 ℎ𝑅𝐴𝐺<br>**----- End of picture text -----**<br>


**Figure 6:** _**Fusion module.**_ **It combines multimodal embeddings with attention mechanism into a fused representation.** 

Finally, we employ a language model, denoted as TextEncoder, to encode the retrieved knowledge from the external KG as below: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/rag/cikm_2024_EMERGE_Enhancing_Multimodal_Electronic_Health_Records_Predictive_Modeling_with_Retrieval-Augmented_Generation_images/cikm_2024_EMERGE_Enhancing_Multimodal_Electronic_Health_Records_Predictive_Modeling_with_Retrieval-Augmented_Generation.pdf-0006-07.png)


where _𝒉𝑅𝐴𝐺_ symbolizes the sentence embedding of the summary, which we will combine with _𝒉𝑇𝑆_ and _𝒉𝑁𝑜𝑡𝑒_ to obtain a comprehensive representation of the patient’s health status. 


![](raw/papers/2_LLM_NLP/LLM_paper_list/rag/cikm_2024_EMERGE_Enhancing_Multimodal_Electronic_Health_Records_Predictive_Modeling_with_Retrieval-Augmented_Generation_images/cikm_2024_EMERGE_Enhancing_Multimodal_Electronic_Health_Records_Predictive_Modeling_with_Retrieval-Augmented_Generation.pdf-0006-09.png)


**Figure 5:** _**Prompt template for summary generation.**_ 

to a unified dimension: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/rag/cikm_2024_EMERGE_Enhancing_Multimodal_Electronic_Health_Records_Predictive_Modeling_with_Retrieval-Augmented_Generation_images/cikm_2024_EMERGE_Enhancing_Multimodal_Electronic_Health_Records_Predictive_Modeling_with_Retrieval-Augmented_Generation.pdf-0006-12.png)


To better integrate information from different modalities, we propose an attention-based fusion network primarily consisting of cross-attention layers. First, the _𝑄𝑢𝑒𝑟𝑦_ vector is computed from the hidden representation of the other modality, while the _𝐾𝑒𝑦_ and _𝑉𝑎𝑙𝑢𝑒_ vectors are computed from the hidden representations of the current modality: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/rag/cikm_2024_EMERGE_Enhancing_Multimodal_Electronic_Health_Records_Predictive_Modeling_with_Retrieval-Augmented_Generation_images/cikm_2024_EMERGE_Enhancing_Multimodal_Electronic_Health_Records_Predictive_Modeling_with_Retrieval-Augmented_Generation.pdf-0006-14.png)


where _𝑄_ , _𝐾_ , _𝑉_ are the _𝑄𝑢𝑒𝑟𝑦_ , _𝐾𝑒𝑦_ , _𝑉𝑎𝑙𝑢𝑒_ vectors respectively, and _𝑾𝑞_ , _𝑾𝑘_ , _𝑾𝑣_ are the corresponding projection matrices. Following this, we compute the attention outputs as follows: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/rag/cikm_2024_EMERGE_Enhancing_Multimodal_Electronic_Health_Records_Predictive_Modeling_with_Retrieval-Augmented_Generation_images/cikm_2024_EMERGE_Enhancing_Multimodal_Electronic_Health_Records_Predictive_Modeling_with_Retrieval-Augmented_Generation.pdf-0006-16.png)


In addition, we apply residual connections and BatchNorm to every multi-head attention layer and feedforward network. 

As a result, the outputs of the two cross-attention modules have carried information from both modalities. We further concatenate them and use an MLP layer to obtain the fused information: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/rag/cikm_2024_EMERGE_Enhancing_Multimodal_Electronic_Health_Records_Predictive_Modeling_with_Retrieval-Augmented_Generation_images/cikm_2024_EMERGE_Enhancing_Multimodal_Electronic_Health_Records_Predictive_Modeling_with_Retrieval-Augmented_Generation.pdf-0006-19.png)


Finally, the fused representation _𝒛_ is expected to predict downstream tasks. We pass _𝒛_ through a two-layer MLP structure, with an additional dropout layer between two fully connected layers, to obtain the final prediction results _𝑦_ ˆ: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/rag/cikm_2024_EMERGE_Enhancing_Multimodal_Electronic_Health_Records_Predictive_Modeling_with_Retrieval-Augmented_Generation_images/cikm_2024_EMERGE_Enhancing_Multimodal_Electronic_Health_Records_Predictive_Modeling_with_Retrieval-Augmented_Generation.pdf-0006-21.png)


## **4.3 Multimodal Fusion Network** 

Currently, there are three learned hidden representations, denoted respectively as _𝒉𝑇𝑆_ , _𝒉𝑁𝑜𝑡𝑒_ , and _𝒉𝑅𝐴𝐺_ . We first concatenate the hidden representations extracted from entities with those from the text, and then utilize a fusion network to combine and map them 

The BCE Loss is selected as the loss function for the binary mortality outcome and readmission prediction task: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/rag/cikm_2024_EMERGE_Enhancing_Multimodal_Electronic_Health_Records_Predictive_Modeling_with_Retrieval-Augmented_Generation_images/cikm_2024_EMERGE_Enhancing_Multimodal_Electronic_Health_Records_Predictive_Modeling_with_Retrieval-Augmented_Generation.pdf-0006-25.png)


CIKM ’24, October 21–25, 2024, Boise, ID, USA. 

EMERGE: Enhancing Multimodal EHR Predictive Modeling with RAG 

where _𝑁_ is the number of patients within one batch, _𝑦_ ˆ ∈[0 _,_ 1] is the predicted probability, and _𝑦_ is the ground truth. 

By converting these three different types of data into compatible embeddings, our model lays a solid groundwork for the multimodal analysis of EHR. This strategy of embedding extraction sets the stage for further analysis tasks under the RAG framework, allowing us to accurately and comprehensively understand and analyze the complex information in EHR. 

## **5 Experimental Setups** 

## **5.1 Experimented Datasets and Utilized KG** 

Sourced from the EHRs of the Beth Israel Deaconess Medical Center, MIMIC-III and MIMIC-IV dataset is extensive and widely used in healthcare research. We adhere to the established EHR benchmark pipeline [15, 54] for preprocessing time-series data. 17 lab test features (include categorical features) and 2 demographic features (age and gender) are extracted. To minimize missing data, we consolidate every consecutive 12-hour segment into a single record for each patient, focusing on the first 48 records. And we follow Clinical-LongFormer[22]’s approach to extract and preprocess clinical notes, which includes minimal but essential steps: removing all de-identification placeholders to protect Protected Health Information (PHI), replacing non-alphanumeric characters and punctuation marks, converting all letters to lowercase for consistency, and stripping extra white spaces. 

We excluded all patients without any notes or time-series data. We randomly split the dataset into training, validation, and test set with 7:1:2 percentage. The statistics of datasets is in Table 1. 

DeepSeek-v2 [10]’s strict content censor policy, which flagged “Content Exists Risk.” We replaced these with “None”. 

## **5.2 Evaluation Metrics** 

We adopt the following evaluation metrics, which are widely used in binary classification tasks: 

- **AUROC** : This metric is our primary consideration in binary classification tasks due to its widespread use in clinical settings and its effectiveness in handling imbalanced datasets [29]. 

- **AUPRC** : The AUPRC is particularly useful for evaluating performance in datasets with a significant imbalance between classes [19]. 

- **min(+P, Se)** : This composite metric represents the minimum value between precision (+P) and sensitivity (Se), providing a balanced measure of model performance [28]. 

All these three metrics are the higher the better. 

## **5.3 Baseline Models** 

_5.3.1 EHR Prediction Models._ We include multimodal EHR baseline models (MPIM [50], UMM [20], MedGTX [32], VecoCare[43], M3Care [49]) and approaches that incorporating external knowledge from KG (GRAM [8], KAME [26], CGL [25], KerPrint [45], MedPath [46], MedRetriever [47]), and LLM facilitated model GraphCare [17] as our baselines. 

_5.3.2 Multimodal Fusion Methods._ To examine the effectiveness of our fusion network, we consider fusion methods: Add [41], Concat [12, 18], Tensor Fusion (TF) [48], and MAG [34, 44]. 

## **5.4 Implementation Details** 

**Table 1:** _**Statistics of datasets after preprocessing.**_ **The number and proportion for labels are the percentage of the label with value** 1 **.** _𝑂𝑢𝑡._ **denotes Mortality Outcome,** _𝑅𝑒._ **denotes Readmission.** 

|**Dataset**|**Split**|**Samples**|**Label**_𝑂𝑢𝑡._|**Label**_𝑅𝑒._|
|---|---|---|---|---|
||Train|10776 (70.00%)|1389 (12.89%)|1787 (16.58%)|
|MIMIC-III|Val|1539 (10.00%)|193 (12.54%)|258 (16.76%)|
||Test|3080 (20.00%)|361 (11.72%)|489 (15.88%)|
||Train|13531 (70.00%)|1608 (11.88%)|2099 (15.51%)|
|MIMIC-IV|Val|1933 (10.00%)|244 (12.62%)|297 (15.36%)|
||Test|3867 (20.00%)|448 (11.59%)|599 (15.49%)|



The external knowledge base we utilized is PrimeKG [6], which integrates 20 high-quality resources to describe 17,080 diseases with 4,050,249 relationships representing ten major biological scales, including disease-associated entities. Futhermore, PrimeKG extracts textual features of disease nodes containing information about disease prevalence, symptoms, etiology, risk factors, epidemiology, clinical descriptions, management and treatment, complications, prevention, and when to seek medical attention, which are highly relevant to the clinical prediction tasks. 

The median number of retrieved entities is 14 for MIMIC-III and 7 for MIMIC-IV, with an average effective extracted entity rate of 67.25% and 66.88%, respectively, from a total of 468,948 and 319,893 extracted entities for the two datasets. 

When prompting the LLM to generate the summary, 4 patients in the MIMIC-III dataset were not successfully generated due to 

_5.4.1 Hardware and Software Configuration._ All runs are trained on a single Nvidia RTX 3090 GPU with CUDA 12.4. The server’s system memory (RAM) size is 128GB. We implement the model in Python 3.11, PyTorch 2.2.2 [33], PyTorch Lightning 2.2.4 [13], and pyehr [54]. 

_5.4.2 Model Training and Hyperparameters._ AdamW [24] is employed with a batch size of 256 patients. All models are trained for 100 epochs with an early stopping strategy based on AUPRC after 10 epochs without improvement. The learning rate 0 _._ 01 _,_ 0 _._ 001 _,_ 0 _._ 0001 and hidden dimensions 32 _,_ 64 _,_ 128 _,_ 256 are tuned using a grid search strategy on the validation set. The searched hyperparameter for EMERGE is: 128 hidden dimensions, 0.001 learning rate. The dropout rate is set to 0 _._ 25. Performance is reported in the form of mean±std by applying bootstrapping on all test set samples 10 times for the MIMIC-III and MIMIC-IV datasets, following practices in Ma et al. [27]. The threshold _𝜖_ for identifying anomalies in time-series data is set as 2 (z-score value=2). The threshold _𝜂_ for matching entities in KG is set as 0.6 for MIMIC-III and 0.7 for MIMIC-IV. 

_5.4.3 Utilized (Large) Language Models._ EMERGE utilizes both Language Models (LMs) and Large Language Models (LLMs) in the pipeline. For LMs, we use the frozen-parameter pretrained ClinicalLongFormer [22]’s [CLS] token [11] for extracting textual embeddings and BGE-M3 [7] as the text embedding model to compute entity embeddings. For LLMs, we deploy an offline Qwen-7B [2] to extract entities from clinical notes and call the DeepSeek-V2 Chat [10] API to generate summaries. 

CIKM ’24, October 21–25, 2024, Boise, ID, USA. 

Yinghao Zhu et al. 

**Table 2:** _**In-hospital mortality and 30-day readmission prediction results on the MIMIC-III and MIMIC-IV datasets.**_ **Bold indicates the best performance. All metrics are multiplied by 100 for readability purposes.** 

|**Methods**|**MIMIC-III Mortality**<br>AUROC (↑)<br>AUPRC (↑)<br>min(+P, Se) (↑)|**MIMIC-III Readmission**<br>AUROC (↑)<br>AUPRC (↑)<br>min(+P, Se) (↑)|**MIMIC-IV Mortality**<br>AUROC (↑)<br>AUPRC (↑)<br>min(+P, Se) (↑)|**MIMIC-IV Readmission**<br>AUROC (↑)<br>AUPRC (↑)<br>min(+P, Se) (↑)|
|---|---|---|---|---|
|MPIM<br>UMM<br>MedGTX<br>VecoCare<br>M3Care<br>GRAM<br>KAME<br>CGL<br>KerPrint<br>MedPath<br>MedRetriever<br>GraphCare|85.24±1.12<br>50.52±2.56<br>50.59±2.33<br>84.01±1.10<br>49.76±2.21<br>49.41±2.45<br>85.97±1.04<br>49.36±3.05<br>48.20±2.27<br>83.43±1.49<br>47.28±2.68<br>47.92±2.22<br>83.33±1.24<br>47.86±2.33<br>49.96±1.99<br>84.70±1.34<br>49.21±4.45<br>49.64±2.85<br>84.59±1.11<br>49.48±3.37<br>49.51±2.33<br>84.20±1.16<br>47.64±3.47<br>47.67±2.61<br>85.29±1.21<br>51.23±3.48<br>50.88±2.24<br>85.61±1.34<br>48.90±3.24<br>48.86±3.00<br>85.62±1.47<br>49.99±3.06<br>49.03±2.54<br>85.85±0.95<br>50.16±2.20<br>49.15±2.57|78.65±1.51<br>48.26±2.84<br>46.94±1.97<br>77.46±1.36<br>47.81±2.55<br>47.27±1.91<br>78.60±1.17<br>46.44±2.69<br>45.99±2.60<br>76.93±1.82<br>46.18±2.76<br>47.22±2.63<br>76.80±1.55<br>46.29±2.62<br>45.38±2.32<br>77.84±1.49<br>47.97±3.68<br>46.95±2.12<br>78.04±1.34<br>48.23±3.21<br>47.41±2.50<br>77.47±1.33<br>46.68±3.33<br>47.73±2.25<br>78.81±1.68<br>47.92±2.45<br>47.32±2.52<br>77.92±0.85<br>45.66±2.61<br>45.72±2.24<br>77.77±0.90<br>46.81±2.36<br>46.89±2.08<br>78.70±1.19<br>47.19±2.33<br>46.82±2.04|89.45±0.59<br>60.10±1.67<br>57.62±1.41<br>87.82±0.73<br>53.84±2.35<br>55.40±1.98<br>88.77±0.73<br>58.33±2.31<br>58.25±1.59<br>88.01±0.68<br>55.37±2.20<br>55.35±1.72<br>88.14±0.78<br>54.06±2.04<br>54.30±1.73<br>87.75±0.65<br>54.01±2.93<br>54.62±2.63<br>87.76±0.67<br>55.74±2.37<br>54.79±1.44<br>88.42±0.94<br>56.64±2.21<br>54.80±1.62<br>88.28±0.60<br>57.90±1.80<br>55.12±1.46<br>88.85±1.00<br>56.82±2.60<br>57.96±2.63<br>89.01±0.42<br>57.75±1.60<br>58.16±1.32<br>89.13±0.57<br>60.85±2.01<br>59.16±1.85|79.13±0.78<br>47.67±1.95<br>49.52±1.99<br>78.75±0.63<br>48.63±1.45<br>49.58±1.29<br>78.82±1.32<br>47.48±1.88<br>49.54±1.76<br>79.17±1.20<br>51.58±1.93<br>51.42±1.48<br>79.87±1.31<br>51.03±1.95<br>51.10±1.36<br>79.53±1.01<br>50.13±2.53<br>50.80±1.67<br>78.91±1.01<br>47.62±1.66<br>49.63±1.28<br>78.95±0.90<br>47.74±1.66<br>49.16±1.24<br>79.84±1.03<br>53.55±1.61<br>52.34±1.64<br>78.88±0.83<br>47.58±2.23<br>49.75±2.39<br>79.15±0.90<br>48.26±1.08<br>49.49±1.18<br>79.18±1.15<br>48.55±1.86<br>49.64±1.58|
|EMERGE|**86.25±1.50**<br>**52.08±2.87**<br>**51.42±2.40**|**79.06±1.05**<br>**48.59±2.52**<br>**47.86±2.58**|**89.50±0.57**<br>**63.11±2.12**<br>**59.95±1.49**|**80.61±1.09**<br>**57.28±2.01**<br>**54.50±1.71**|



## **6 Experimental Results and Analysis 6.1 Experimental Results** 

The performance of our EMERGE framework on in-hospital mortality and 30-day readmission prediction tasks on the MIMIC-III and MIMIC-IV datasets is summarized in Table 2. EMERGE consistently outperforms the baseline models, indicating its superior practical applicability in real-world clinical settings. 

## **6.2 Ablation Studies** 

_6.2.1 Comparing Different Modality Fusion Strategies._ To understand the contribution of each modality and the modality fusion approaches, we compare their performance, as illustrated in Table 3. The results reveal that: 1) Utilizing multiple modalities is better than using a single modality. 2) The RAG pipeline-generated summary exhibits stronger representation capability (by comparing the settings “Note only” vs. “RAG only”, and “TS+Note” vs. “TS+RAG”). This showcases the effectiveness of task-relevant generated summaries in facilitating prediction modeling. 3) EMERGE’s cross-attention-based adaptive multimodal fusion network outperforms other modality fusion strategies. 

_6.2.2 Comparing Different Time-series Encoders._ From Figure 7, we compare the performance of four different time-series encoders: GRU, LSTM, Transformer, and RNN, in encoding EHR data. The evaluation focuses exclusively on time-series data inputs, excluding any text inputs, to determine which model is most effective in handling such data. The GRU model consistently performs well, therefore we have selected GRU as the backbone encoder for timeseries data in EMERGE. 

_6.2.3 Comparing Different Text Fusion Approaches._ From Figure 8, similar as modality fusion, we conduct the comparison for multiple text fusion approaches: note only (“OnlyNote”), summary only (“OnlyRAG”), add, concat, adaptive concat, and MAG. The evaluation focuses exclusively on text inputs with no time-series data. The concat strategy performs the best on the MIMIC-III model and shows decent performance on MIMIC-IV. Considering its simplicity, we choose concat as the text fusion method. 

_6.2.4 Comparing Internal Design of Fusion Module._ To explore in detail the role of the cross-attention mechanism for multimodal fusion in Figure 6, we provide experiments on alternative internal 


![](raw/papers/2_LLM_NLP/LLM_paper_list/rag/cikm_2024_EMERGE_Enhancing_Multimodal_Electronic_Health_Records_Predictive_Modeling_with_Retrieval-Augmented_Generation_images/cikm_2024_EMERGE_Enhancing_Multimodal_Electronic_Health_Records_Predictive_Modeling_with_Retrieval-Augmented_Generation.pdf-0008-11.png)


**----- Start of picture text -----**<br>
90 90<br>85 85<br>80 80<br>75 75<br>70 70<br>65 Task 65 Task<br>Mortality Mortality<br>Readmission Readmission<br>60 GRU LSTM Transformer RNN 60 GRU LSTM Transformer RNN<br>Model Model<br>(a) MIMIC-III (b) MIMIC-IV<br>AUROC Score AUROC Score<br>**----- End of picture text -----**<br>


**Figure 7:** _**AUROC performance of four time-series encoders in in-hospital mortality prediction and 30-day readmission prediction tasks on two datasets.**_ 


![](raw/papers/2_LLM_NLP/LLM_paper_list/rag/cikm_2024_EMERGE_Enhancing_Multimodal_Electronic_Health_Records_Predictive_Modeling_with_Retrieval-Augmented_Generation_images/cikm_2024_EMERGE_Enhancing_Multimodal_Electronic_Health_Records_Predictive_Modeling_with_Retrieval-Augmented_Generation.pdf-0008-13.png)


**----- Start of picture text -----**<br>
75 80<br>70 75<br>70<br>65<br>65<br>60<br>60<br>55 Task 55 Task<br>Mortality Mortality<br>Readmission Readmission<br>50 OnlyNote OnlyRAG Add Concat Adaptive MAG 50 OnlyNote OnlyRAG Add Concat Adaptive MAG<br>Text Fusion Methods Text Fusion Methods<br>(a) MIMIC-III (b) MIMIC-IV<br>AUROC Score AUROC Score<br>**----- End of picture text -----**<br>


**Figure 8:** _**AUROC performance of different text fusion methods in in-hospital mortality prediction and 30-day readmission prediction tasks on two datasets.**_ 

components in Figure 9: “Ours” represents the version in Figure 6, “TSQuery” can be regarded as the left branch with the time-series embedding serving as the query, “TextQuery” as the right branch, “SelfAttention” replaces the cross-attention and retains the concat and projection layer, and “Concat” does not include any attention module. The superior performance of our final employed fusion approach demonstrates the effectiveness of the cross-modality fusion approach in a bi-directional way. 

CIKM ’24, October 21–25, 2024, Boise, ID, USA. 

EMERGE: Enhancing Multimodal EHR Predictive Modeling with RAG 

**Table 3:** _**Ablation study results of 1) comparing each modality with RAG enhancement, and 2) comparing different multimodal fusion networks.**_ **Bold and Underlined indicates the best and 2nd best performance. All metrics are multiplied by 100.** 

|**Meth**|**Meth**|**ods**|A|URO|<br>C (↑)|**MIMIC-III Mortality**<br>AUPRC (↑)<br>min(+P, Se) (↑)|**MIMIC-III Mortality**<br>AUPRC (↑)<br>min(+P, Se) (↑)|**MIMIC-III Mortality**<br>AUPRC (↑)<br>min(+P, Se) (↑)|**MIMIC-III Mortality**<br>AUPRC (↑)<br>min(+P, Se) (↑)|AU|RO|**MIM**<br>C (↑)|**MIM**<br>C (↑)|**IC-III Readmission**<br>AUPRC (↑)<br>min(+P, Se) (↑)|**IC-III Readmission**<br>AUPRC (↑)<br>min(+P, Se) (↑)|**MI**<br>AUROC (↑)|**MI**<br>AUROC (↑)|**MI**<br>AUROC (↑)|**MIC-I**<br>AUPR|**V Mortality**<br>C (↑)<br>min(+P, Se) (↑)|**V Mortality**<br>C (↑)<br>min(+P, Se) (↑)|**V Mortality**<br>C (↑)<br>min(+P, Se) (↑)|**MIM**<br>AUROC (↑)|**MIM**<br>AUROC (↑)|**MIM**<br>AUROC (↑)|**IC-IV Readmission**<br>AUPRC (↑)<br>min(+P, Se) (↑)|**IC-IV Readmission**<br>AUPRC (↑)<br>min(+P, Se) (↑)|**IC-IV Readmission**<br>AUPRC (↑)<br>min(+P, Se) (↑)|**IC-IV Readmission**<br>AUPRC (↑)<br>min(+P, Se) (↑)|
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
|TS only<br>Note onl<br>RAG onl<br>TS+Note<br>TS+RAG<br>Note+RA||y<br>y<br>G|8<br>6<br>6<br>8<br>8<br>7|4.57±<br>6.50±<br>9.21±<br>5.72±<br>6.21±|1.50<br>1.40<br>1.54<br>1.34<br>1.29|46.53±3.14<br>19.62±0.68<br>22.46±2.68<br>49.02±2.76<br>51.15±3.24<br>27.07±1.66|48<br>23<br>27<br>48<br>50|.89±2<br>.22±1<br>.04±2<br>.28±2<br>.62±2|.92<br>.23<br>.62<br>.36<br>.78<br>.72|77<br>64<br>64<br>78<br>78<br>68|.17±<br>.76±<br>.65±<br>.36±|1.3<br>1.0<br>1.0<br>1.0|6<br>0<br>5<br>6<br>0<br>0|43.87±2.72<br>24.64±0.76<br>24.12±1.78<br>46.95±2.49<br>46.94±2.54<br>28.87±1.47|46.21±2.83<br>27.07±0.51<br>27.65±1.63<br>45.79±2.17<br>47.11±2.46<br>31.96±1.62|87.96±0.65<br>69.47±1.03<br>71.84±1.27<br>88.55±0.58<br>89.49±0.58<br>74.96±1.12|||55.62<br>27.70<br>27.68<br>60.01<br>62.49<br>32.28|±2.00<br>±1.26<br>±2.76<br>±1.84<br>±2.19<br>±2.97|55.02±2.01<br>30.90±1.30<br>30.62±2.91<br>57.95±1.47<br>58.75±2.20<br>35.43±2.54||79.03±1.17<br>66.40±0.97<br>67.37±1.29<br>79.93±0.94<br>80.55±1.12<br>70.72±1.23|||51.79±1.93<br>29.52±1.31<br>28.26±2.37<br>54.29±1.67<br>55.64±2.07<br>32.42±2.26||51.02±1.66<br>32.39±1.61<br>31.83±2.16<br>52.84±1.45<br>52.38±1.77<br>35.33±2.70||
||||||||||||.24±<br>.80±|0.9<br>0.8||||||||||||||||||
|||||2.32±|1.14||28|.66±1||||||||||||||||||||||
|TS+T<br>TS+T<br>TS+T<br>TS+T|ext: <br>ext: <br>ext: <br>ext:|Concat<br>TF<br>MAG<br>Ours|8<br>8<br>8<br>**86**|5.66±<br>5.55±<br>6.09±<br>**.25**|1.44<br>1.42<br>1.47<br>**±1.50**|49.41±2.89<br>50.30±2.92<br>49.14±2.51<br>**52.08±2.87**|48<br>50<br>49<br>**51**|.18±3<br>.11±3<br>.12±2<br>**.42±**|.09<br>.24<br>.92<br>**2.40**|78<br>77<br>77<br>**79**|.04±<br>.83±<br>.69±<br>**.06±**|1.0<br>1.1<br>0.8<br>**1.0**|0<br>5<br>9<br>**5**|46.72±2.36<br>46.73±2.50<br>44.86±2.04<br>**48.59±2.52**|46.18±2.21<br>46.70±2.59<br>45.76±1.67<br>**47.86±2.58**|89.33±0.57<br>89.08±0.57<br>**89.56±0.62**<br>89.50±0.57|||62.42<br>59.47<br>62.64|±2.10<br>±2.28<br>±2.04<br>**±2.12**|59.75±1.23<br>59.53±1.53<br>**60.16±1.52**<br>59.95±1.49||80.58±0.96<br>80.34±0.96<br>**80.66±1.08**<br>80.61±1.09|||55.40±1.84<br>53.01±1.87<br>56.62±1.96<br>**57.28±2.01**||52.77±1.47<br>51.81±1.35<br>53.97±1.71<br>**54.50±1.71**||
||||||||||||||||||||**63.11**|||||||||||
|65<br>70<br>75<br>80<br>85<br>AUROC Score||||||||||||||ention<br>Concat<br>ds<br>Task<br>Mortality<br>Readmission|1<br>20<br>40<br>60<br>80<br>Training Set Percentage<br>40<br>50<br>60<br>70<br>80<br>90<br>AUROC Score<br>~~Model~~<br>EMERGE<br>MPIM<br>VecoCare<br>M3Care<br>~~KerPrint~~<br>MedRetriever<br>GraphCare<br>(a) Mortality<br>1<br>20<br>40<br>60<br>80<br>Training Set Percentage<br>40<br>50<br>60<br>70<br>80<br>AUROC Score<br>Model<br>~~EMERGE~~<br>MPIM<br>VecoCare<br>~~M3Care~~<br>KerPrint<br>MedRetriever<br>GraphCare<br>(b) Readmission<br>**Figure 11:** **_AUROC performance across 5 Training Set Per_**|90||||||||||||||
|||||||||||||||||1<br>2<br>40<br>50<br>60<br>70<br>80<br>|||||||1<br>20<br> <br>40<br>50<br>60<br>70<br>80|||||||
||||||||||||||||||||||~~Model~~|||||||Model<br>~~EMERGE~~||
||||||||||||||||||||||EMERGE<br>MPIM<br>VecoCare|||||||||
|||||||||||||||||||||||||||||MPIM<br>VecoCare||
||||||||||||||||||||||M3Care<br>~~KerPrint~~<br>MedRetriev<br>GraphCare|er||||||~~M3Care~~<br>KerPrint<br>MedRetriev<br>GraphCare|er|
|||||||||||||||||||0<br>T|4<br>raining Se|0<br>6<br>t Percentage||||||40<br>6<br>Training Set Percentage||||
|||||||Task<br>Mortality<br>Readmission||||||||Task<br>Mortality<br>Readmission||||||||||||||||
|||Ours<br>TSQu<br>Int|ery<br>ernal|TextQ<br>Desi|uery<br>Self<br>gn Met|||Ou|rs<br>TSQ<br>In|uery<br>ternal|TextQ<br>Desi|uery<br><br>gn M|SelfAtt<br>etho|||||||||||||||||
|||(a)|M|IMI|C-III||||(b|) M|IM|IC-|IV|||||||||||||||||



**Figure 11:** _**AUROC performance across 5 Training Set Percentage in in-hospital mortality prediction (left) and 30-day readmission prediction (right) task on MIMIC-IV.**_ 

**Figure 9:** _**AUROC performance of different internal designs of our proposed fusion module in in-hospital mortality prediction and 30-day readmission prediction tasks on two datasets.**_ 

## **7 Conclusions** 

In this work, we propose EMERGE, an RAG-driven multimodal EHR data representation learning framework that incorporates timeseries EHR, clinical notes data, and an external knowledge graph for healthcare prediction. The EMERGE framework comprehensively leverages LLMs’ semantic reasoning ability, long-context encoding capacity, and the medical context of the knowledge graph. The EMERGE framework achieves superior performance on two realworld datasets’ in-hospital mortality and 30-day readmission tasks against the latest baseline models. Extensive experiments showcase EMERGE’s effectiveness and robustness to data sparsity. EMERGE marks a step towards more effective utilization of multimodal EHR data in healthcare, offering a potent solution to enhance clinical representations with external knowledge and LLMs. 

## **6.3 Further Analysis** 

_6.3.1 Sensitivity to Hidden Dimensions and Learning Rates._ To assess the sensitivity of our EMERGE framework to different hidden dimensions and learning rates, we conducted experiments on the MIMIC-III and MIMIC-IV datasets (Figure 10). The results indicate that a hidden dimension of 128 and a learning rate of 1e-3 yield the best performance. The minimal variations across different settings demonstrate EMERGE’s low sensitivity to these hyperparameters. 


![](raw/papers/2_LLM_NLP/LLM_paper_list/rag/cikm_2024_EMERGE_Enhancing_Multimodal_Electronic_Health_Records_Predictive_Modeling_with_Retrieval-Augmented_Generation_images/cikm_2024_EMERGE_Enhancing_Multimodal_Electronic_Health_Records_Predictive_Modeling_with_Retrieval-Augmented_Generation.pdf-0009-10.png)


**----- Start of picture text -----**<br>
90 90 Tas k<br>88 8580 MortalityReadmission<br>86 Task 75<br>84 MortalityReadmission 7065<br>82 60<br>80 5550<br>16 32 64 128 256 512 1e-1 1e-2 1e-3 1e-4 1e-5 1e-6<br>Hidden Dimensions Learning Rates<br>(a) Hidden Dimensions (b) Learning Rates<br>AUROC Score AUROC Score<br>**----- End of picture text -----**<br>


## **Ethical Statement** 

This study, involving the analysis of de-identified Electronic Health Records (EHR) from the MIMIC-III and MIMIC-IV datasets, upholds high ethical standards. It should be noted that in our use of the online API of the LLM to generate patient summaries, the content in the prompts is derived from publicly accessible knowledge graphs and only includes feature names from the MIMIC dataset. Therefore, privacy concerns are limited. Overall, our methodology aims to minimize harm and ensure unbiased, equitable findings, reflecting the complex nature of medical data. We rigorously adhere to these ethical values throughout our research. 

**Figure 10:** _**AUROC performance to various hidden dimensions (left) and learning rates (right) in in-hospital mortality and 30-day readmission prediction tasks on MIMIC-IV.**_ 

_6.3.2 Robustness to Data Sparsity._ To evaluate the robustness of our EMERGE framework against data sparsity, we conduct experiments using 1%, 20%, 40%, 60%, and 80% of the training set. As depicted in Figure 11, EMERGE shows remarkable resilience, especially with only 1% (less than 150) of the training samples. This robustness is crucial in clinical settings where data collection is often challenging, making EMERGE valuable for clinical practice. 

## **Acknowledgments** 

This work was supported by the Defense Industrial Technology Development Program under Grant JCKY2021601B104. 

CIKM ’24, October 21–25, 2024, Boise, ID, USA. 

Yinghao Zhu et al. 

## **References** 

- [1] Josh Achiam, Steven Adler, Sandhini Agarwal, Lama Ahmad, Ilge Akkaya, Florencia Leoni Aleman, Diogo Almeida, Janko Altenschmidt, Sam Altman, Shyamal Anadkat, et al. 2023. GPT-4 Technical Report. _arXiv preprint arXiv:2303.08774_ (2023). 

- [2] Jinze Bai, Shuai Bai, Yunfei Chu, Zeyu Cui, Kai Dang, Xiaodong Deng, Yang Fan, Wenbin Ge, Yu Han, Fei Huang, et al. 2023. Qwen technical report. _arXiv preprint arXiv:2309.16609_ (2023). 

- [3] Jiaqi Bai, Hongcheng Guo, Jiaheng Liu, Jian Yang, Xinnian Liang, Zhao Yan, and Zhoujun Li. 2023. GripRank: Bridging the Gap between Retrieval and Generation via the Generative Knowledge Improved Passage Ranking. In _Proceedings of the 32nd ACM International Conference on Information and Knowledge Management_ (, Birmingham, United Kingdom,) _(CIKM ’23)_ . Association for Computing Machinery, New York, NY, USA, 36–46. https://doi.org/10.1145/3583780.3614901 

- [4] Jiaqi Bai, Zhao Yan, Shun Zhang, Jian Yang, Hongcheng Guo, and Zhoujun Li. 2024. Infusing internalized knowledge of language models into hybrid prompts for knowledgeable dialogue generation. _Knowledge-Based Systems_ (2024), 111874. 

- [5] Jiaqi Bai, Ze Yang, Jian Yang, Hongcheng Guo, and Zhoujun Li. 2023. Kinet: Incorporating relevant facts into knowledge-grounded dialog generation. _IEEE/ACM Transactions on Audio, Speech, and Language Processing_ 31 (2023), 1213–1222. 

- [6] Payal Chandak, Kexin Huang, and Marinka Zitnik. 2023. Building a knowledge graph to enable precision medicine. _Scientific Data_ 10, 1 (2023), 67. 

- [7] Jianlv Chen and Shitao Xiao. 2024. BGE M3-Embedding: Multi-Lingual, MultiFunctionality, Multi-Granularity Text Embeddings Through Self-Knowledge Distillation. https://synthical.com/article/9ffce599-0640-457c-bd1c-502cab06e8af. arXiv:2402.03216 [cs.AI] 

- [8] Edward Choi, Mohammad Taha Bahadori, Le Song, Walter F Stewart, and Jimeng Sun. 2017. GRAM: graph-based attention model for healthcare representation learning. In _Proceedings of the 23rd ACM SIGKDD international conference on knowledge discovery and data mining_ . 787–795. 

- [9] Alexander E Curtis, Tanya A Smith, Bulat A Ziganshin, and John A Elefteriades. 2016. The mystery of the Z-score. _Aorta_ 4, 04 (2016), 124–130. 

- [10] DeepSeek-AI. 2024. DeepSeek-V2: A Strong, Economical, and Efficient Mixtureof-Experts Language Model. arXiv:2405.04434 [cs.CL] 

- [11] Jacob Devlin, Ming-Wei Chang, Kenton Lee, and Kristina Toutanova. 2018. Bert: Pre-training of deep bidirectional transformers for language understanding. _arXiv preprint arXiv:1810.04805_ (2018). 

- [12] Iman Deznabi, Mohit Iyyer, and Madalina Fiterau. 2021. Predicting in-hospital mortality by combining clinical notes with time-series data. In _Findings of the Association for Computational Linguistics: ACL-IJCNLP 2021_ . 4026–4031. 

- [13] William A Falcon. 2019. Pytorch lightning. _GitHub_ 3 (2019). 

- [14] Junyi Gao, Chaoqi Yang, Joerg Heintz, Scott Barrows, Elise Albers, Mary Stapel, Sara Warfield, Adam Cross, and Jimeng Sun. 2022. MedML: fusing medical knowledge and machine learning models for early pediatric COVID-19 hospitalization and severity prediction. _Iscience_ 25, 9 (2022). 

- [15] Junyi Gao, Yinghao Zhu, Wenqing Wang, Guiying Dong, Wen Tang, Hao Wang, Yasha Wang, Ewen M Harrison, and Liantao Ma. 2024. A Comprehensive Benchmark for COVID-19 Predictive Modeling Using Electronic Health Records in Intensive Care. _Patterns_ (2024). https://doi.org/10.1016/j.patter.2024.100951 

- [16] Fergus Imrie, Paulius Rauba, and Mihaela van der Schaar. 2023. Redefining Digital Health Interfaces with Large Language Models. _arXiv preprint arXiv:2310.03560_ (2023). 

- [17] Pengcheng Jiang, Cao Xiao, Adam Richard Cross, and Jimeng Sun. 2024. GraphCare: Enhancing Healthcare Predictions with Personalized Knowledge Graphs. In _The Twelfth International Conference on Learning Representations_ . 

- [18] Swaraj Khadanga, Karan Aggarwal, Shafiq Joty, and Jaideep Srivastava. 2019. Using clinical notes with time series data for ICU management. _arXiv preprint arXiv:1909.09702_ (2019). 

- [19] Misuk Kim and Kyu-Baek Hwang. 2022. An empirical evaluation of sampling methods for the classification of imbalanced data. _PLoS One_ 17, 7 (2022), e0271260. 

- [20] Kwanhyung Lee, Soojeong Lee, Sangchul Hahn, Heejung Hyun, Edward Choi, Byungeun Ahn, and Joohyung Lee. 2023. Learning Missing Modal Electronic Health Records with Unified Multi-modal Data Embedding and Modality-Aware Attention. _arXiv preprint arXiv:2305.02504_ (2023). 

- [21] Patrick Lewis, Ethan Perez, Aleksandra Piktus, Fabio Petroni, Vladimir Karpukhin, Naman Goyal, Heinrich Küttler, Mike Lewis, Wen-tau Yih, Tim Rocktäschel, et al. 2020. Retrieval-augmented generation for knowledge-intensive nlp tasks. _Advances in Neural Information Processing Systems_ 33 (2020), 9459–9474. 

- [22] Yikuan Li, Ramsey M Wehbe, Faraz S Ahmad, Hanyin Wang, and Yuan Luo. 2023. A comparative study of pretrained language models for long clinical text. _Journal of the American Medical Informatics Association_ 30, 2 (2023), 340–347. 

- [23] Weibin Liao, Yinghao Zhu, Zixiang Wang, Xu Chu, Yasha Wang, and Liantao Ma. 2024. Learnable Prompt as Pseudo-Imputation: Reassessing the Necessity of Traditional EHR Data Imputation in Downstream Clinical Prediction. _arXiv preprint arXiv:2401.16796_ (2024). 

- [24] Ilya Loshchilov and Frank Hutter. 2017. Decoupled weight decay regularization. _arXiv preprint arXiv:1711.05101_ (2017). 

- [25] Chang Lu, Chandan K Reddy, Prithwish Chakraborty, Samantha Kleinberg, and Yue Ning. 2021. Collaborative Graph Learning with Auxiliary Text for Temporal Event Prediction in Healthcare. In _Proceedings of the Thirtieth International Joint Conference on Artificial Intelligence, IJCAI-21_ , Zhi-Hua Zhou (Ed.). International Joint Conferences on Artificial Intelligence Organization, 3529–3535. https: //doi.org/10.24963/ijcai.2021/486 Main Track. 

- [26] Fenglong Ma, Quanzeng You, Houping Xiao, Radha Chitta, Jing Zhou, and Jing Gao. 2018. Kame: Knowledge-based attention model for diagnosis prediction in healthcare. In _Proceedings of the 27th ACM International Conference on Information and Knowledge Management_ . 743–752. 

- [27] Liantao Ma, Chaohe Zhang, Junyi Gao, Xianfeng Jiao, Zhihao Yu, Yinghao Zhu, Tianlong Wang, Xinyu Ma, Yasha Wang, Wen Tang, Xinju Zhao, Wenjie Ruan, and Tao Wang. 2023. Mortality prediction with adaptive feature importance recalibration for peritoneal dialysis patients. _Patterns_ 4, 12 (2023). https://doi. org/10.1016/j.patter.2023.100892 

- [28] Xinyu Ma, Yasha Wang, Xu Chu, Liantao Ma, Wen Tang, Junfeng Zhao, Ye Yuan, and Guoren Wang. 2022. Patient Health Representation Learning via Correlational Sparse Prior of Medical Features. _IEEE Transactions on Knowledge and Data Engineering_ (2022). 

- [29] Matthew B. A. McDermott, Lasse Hyldig Hansen, Haoran Zhang, Giovanni Angelotti, and Jack Gallifant. 2024. A Closer Look at AUROC and AUPRC under Class Imbalance. _arXiv preprint arXiv:2401.06091_ (2024). 

- [30] Riccardo Miotto, Fei Wang, Shuang Wang, Xiaoqian Jiang, and Joel T Dudley. 2018. Deep learning for healthcare: review, opportunities and challenges. _Briefings in bioinformatics_ 19, 6 (2018), 1236–1246. 

- [31] OpenAI. 2023. Prompt engineering. https://platform.openai.com/docs/guides/ prompt-engineering. Accessed: 2024-05-20. 

- [32] Sungjin Park, Seongsu Bae, Jiho Kim, Tackeun Kim, and Edward Choi. 2022. Graph-text multi-modal pre-training for medical representation learning. In _Conference on Health, Inference, and Learning_ . PMLR, 261–281. 

- [33] Adam Paszke, Sam Gross, Francisco Massa, Adam Lerer, James Bradbury, Gregory Chanan, Trevor Killeen, Zeming Lin, Natalia Gimelshein, Luca Antiga, et al. 2019. Pytorch: An imperative style, high-performance deep learning library. _Advances in neural information processing systems_ 32 (2019). 

- [34] Wasifur Rahman, Md Kamrul Hasan, Sangwu Lee, Amir Zadeh, Chengfeng Mao, Louis-Philippe Morency, and Ehsan Hoque. 2020. Integrating multimodal information in large pretrained transformers. In _Proceedings of the conference. Association for Computational Linguistics. Meeting_ , Vol. 2020. NIH Public Access, 2359. 

- [35] Alvin Rajkomar, Eyal Oren, Kai Chen, Andrew M Dai, Nissan Hajaj, Michaela Hardt, Peter J Liu, Xiaobing Liu, Jake Marcus, Mimi Sun, et al. 2018. Scalable and accurate deep learning with electronic health records. _NPJ digital medicine_ 1, 1 (2018), 18. 

- [36] Shuhua Shi, Shaohan Huang, Minghui Song, Zhoujun Li, Zihan Zhang, Haizhen Huang, Furu Wei, Weiwei Deng, Feng Sun, and Qi Zhang. 2024. ResLoRA: Identity Residual Mapping in Low-Rank Adaption. _arXiv preprint arXiv:2402.18039_ (2024). 

- [37] Kai Sun, Yifan Ethan Xu, Hanwen Zha, Yue Liu, and Xin Luna Dong. 2023. Headto-tail: How knowledgeable are large language models (llm)? AKA will llms replace knowledge graphs? _arXiv preprint arXiv:2308.10168_ (2023). 

- [38] Boshi Wang, Xiang Yue, and Huan Sun. 2023. Can ChatGPT defend its belief in truth? evaluating LLM reasoning via debate. In _Findings of the Association for Computational Linguistics: EMNLP 2023_ . 11865–11881. 

- [39] Jiaqi Wang, Junyu Luo, Muchao Ye, Xiaochen Wang, Yuan Zhong, Aofei Chang, Guanjie Huang, Ziyi Yin, Cao Xiao, Jimeng Sun, and Fenglong Ma. 2024. Recent Advances in Predictive Modeling with Electronic Health Records. arXiv:2402.01077 [cs.LG] 

- [40] Michael Wornow, Yizhe Xu, Rahul Thapa, Birju Patel, Ethan Steinberg, Scott Fleming, Michael A Pfeffer, Jason Fries, and Nigam H Shah. 2023. The shaky foundations of large language models and foundation models for electronic health records. _npj Digital Medicine_ 6, 1 (2023), 135. 

- [41] Aming Wu and Yahong Han. 2018. Multi-modal Circulant Fusion for Video-toLanguage and Backward.. In _IJCAI_ , Vol. 3. 8. 

- [42] Guangxuan Xiao, Yuandong Tian, Beidi Chen, Song Han, and Mike Lewis. 2023. Efficient Streaming Language Models with Attention Sinks. arXiv:2309.17453 [cs.CL] 

- [43] Yongxin Xu, Kai Yang, Chaohe Zhang, Peinie Zou, Zhiyuan Wang, Hongxin Ding, Junfeng Zhao, Yasha Wang, and Bing Xie. 2023. VecoCare: Visit SequencesClinical Notes Joint Learning for Diagnosis Prediction in Healthcare Data. In _Proceedings of the Thirty-Second International Joint Conference on Artificial Intelligence, IJCAI-23_ , Edith Elkind (Ed.). International Joint Conferences on Artificial Intelligence Organization, 4921–4929. https://doi.org/10.24963/ijcai.2023/547 Main Track. 

- [44] Bo Yang and Lijun Wu. 2021. How to leverage multimodal EHR data for better medical predictions? _arXiv preprint arXiv:2110.15763_ (2021). 

- [45] Kai Yang, Yongxin Xu, Peinie Zou, Hongxin Ding, Junfeng Zhao, Yasha Wang, and Bing Xie. 2023. KerPrint: local-global knowledge graph enhanced diagnosis prediction for retrospective and prospective interpretations. In _Proceedings of the AAAI Conference on Artificial Intelligence_ , Vol. 37. 5357–5365. 

- [46] Muchao Ye, Suhan Cui, Yaqing Wang, Junyu Luo, Cao Xiao, and Fenglong Ma. 2021. Medpath: Augmenting health risk prediction via medical knowledge paths. 

CIKM ’24, October 21–25, 2024, Boise, ID, USA. 

EMERGE: Enhancing Multimodal EHR Predictive Modeling with RAG 

In _Proceedings of the Web Conference 2021_ . 1397–1409. 

- [47] Muchao Ye, Suhan Cui, Yaqing Wang, Junyu Luo, Cao Xiao, and Fenglong Ma. 2021. Medretriever: Target-driven interpretable health risk prediction via retrieving unstructured medical text. In _Proceedings of the 30th ACM International Conference on Information & Knowledge Management_ . 2414–2423. 

- [48] Amir Zadeh, Minghai Chen, Soujanya Poria, Erik Cambria, and Louis-Philippe Morency. 2017. Tensor fusion network for multimodal sentiment analysis. _arXiv preprint arXiv:1707.07250_ (2017). 

- [49] Chaohe Zhang, Xu Chu, Liantao Ma, Yinghao Zhu, Yasha Wang, Jiangtao Wang, and Junfeng Zhao. 2022. M3Care: Learning with Missing Modalities in Multimodal Healthcare Data. In _Proceedings of the 28th ACM SIGKDD Conference on Knowledge Discovery and Data Mining_ (Washington DC, USA) _(KDD ’22)_ . Association for Computing Machinery, New York, NY, USA, 2418–2428. https://doi.org/10.1145/ 3534678.3539388 

- [50] Xinlu Zhang, Shiyang Li, Zhiyu Chen, Xifeng Yan, and Linda Ruth Petzold. 2023. Improving medical predictions by irregular multimodal electronic health records 

   - modeling. In _International Conference on Machine Learning_ . PMLR, 41300–41313. 

- [51] Yue Zhang, Yafu Li, Leyang Cui, Deng Cai, Lemao Liu, Tingchen Fu, Xinting Huang, Enbo Zhao, Yu Zhang, Yulong Chen, et al. 2023. Siren’s song in the ai ocean: A survey on hallucination in large language models. _arXiv preprint arXiv:2309.01219_ (2023). 

- [52] Dawei Zhu, Nan Yang, Liang Wang, Yifan Song, Wenhao Wu, Furu Wei, and Sujian Li. 2024. Pose: Efficient context window extension of llms via positional skip-wise training. In _The Twelfth International Conference on Learning Representations_ . 

- [53] Yinghao Zhu, Junyi Gao, Zixiang Wang, Weibin Liao, Xiaochen Zheng, Lifang Liang, Yasha Wang, Chengwei Pan, Ewen M Harrison, and Liantao Ma. 2024. Is larger always better? Evaluating and prompting large language models for non-generative medical tasks. _arXiv preprint arXiv:2407.18525_ (2024). 

- [54] Yinghao Zhu, Wenqing Wang, Junyi Gao, and Liantao Ma. 2024. PyEHR: A Predictive Modeling Toolkit for Electronic Health Records. https://github.com/ yhzhu99/pyehr. 

