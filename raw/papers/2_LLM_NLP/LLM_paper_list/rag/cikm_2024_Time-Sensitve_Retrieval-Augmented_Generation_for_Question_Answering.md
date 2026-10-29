
![](raw/papers/2_LLM_NLP/LLM_paper_list/rag/cikm_2024_Time-Sensitve_Retrieval-Augmented_Generation_for_Question_Answering_images/cikm_2024_Time-Sensitve_Retrieval-Augmented_Generation_for_Question_Answering.pdf-0001-00.png)


# **Time-sensitive Retrieval-Augmented Generation for Question Answering** 

Feifan Wu[∗†] 220224380@seu.edu.cn Southeast University Nanjing, China 

Wentao He[∗] 220225816@seu.edu.cn Southeast University Nanjing, China 

Lingyuan Liu[∗] 220224295@seu.edu.cn Southeast University Nanjing, China 

Haofen Wang[‡] Meng Wang[‡] carter.whfcarter@gmail.com mengwangtj@tongji.edu.cn College of Design and Innovation, Tongji University Shanghai, China 

Ziqi Liu Zhiqiang Zhang ziqiliu@antgroup.com lingyao.zzq@antfin.com Ant Group Hangzhou, China 

## **Abstract** 

## **ACM Reference Format:** 

Feifan Wu, Lingyuan Liu, Wentao He, Ziqi Liu, Zhiqiang Zhang, Haofen Wang, and Meng Wang. 2024. Time-sensitive Retrieval-Augmented Generation for Question Answering. In _Proceedings of the 33rd ACM International Conference on Information and Knowledge Management (CIKM ’24), October 21–25, 2024, Boise, ID, USA._ ACM, New York, NY, USA, 10 pages. https://doi.org/10.1145/3627673.3679800 

Retrieval-augmented generation (RAG) enhances large language models (LLMs) by accessing external data sources, offering a promising way to improve accuracy and reliability. Despite its potential, conventional retrievers encounter bias and flaws with timesensitive queries. In this paper, a benchmark query dataset is constructed to retrieve documents containing time-evolving facts, and the results show that current embedding-based similarity-matching methods struggle to handle queries with explicit temporal constraints. Therefore, we propose a novel approach that integrates supervised contrastive learning with tailored negative sample pairs for temporal constraints to train the retriever of an RAG system, along with query-side fine-tuning and routing techniques. Experimental results show that our approach significantly enhances the retriever performance of time-sensitive queries while ensuring the effectiveness of general queries. We will make the code and dataset publicly available at https://github.com/suzhou-22/TS-Retriever. 

## **1 Introduction** 

Retrieval-augmented generation (RAG) models have emerged as a promising approach to enhance the accuracy and reliability of large language models (LLMs) by leveraging external data sources. However, most existing text retrievers used in RAG models exhibit bias and limitations when processing time-sensitive queries. For instance, considering the query “Who was the winner of the Nobel Prize in Physics 1943?” in Figure 1, the top 5 retrieval results by the advanced text-embedding-ada-002 model of OpenAI are indeed relevant to Nobel Prize winners in physics but all of them are incorrect because of mismatched years. 

## **CCS Concepts** 

• **Information systems** → **Document filtering** ; **Question answering** . 

Similar queries with time specifiers (e.g., “in 2019”) are common across existing question answering (QA) datasets [17, 29]. Such queries require retrievers of RAG models to understand not only the semantics of the query but also accurately capture temporal constraints. Current embedding-based similarity-matching retrievers struggle with this task, often prioritizing non-temporal semantic relevance over temporal constraints, leading to the retrieval of inaccurate or irrelevant information. In this paper, we first construct a time-sensitive query benchmark dataset to confirm the limitations of existing retrievers and then propose a novel method to achieve better performance. 

## **Keywords** 

retrieval-augmented generation, large language model, retriever, supervised contrastive learning 

∗Both authors contributed equally to this research. †Also with XAI Lab, Tongji University. ‡Corresponding Authors. 

Permission to make digital or hard copies of all or part of this work for personal or classroom use is granted without fee provided that copies are not made or distributed for profit or commercial advantage and that copies bear this notice and the full citation on the first page. Copyrights for components of this work owned by others than the author(s) must be honored. Abstracting with credit is permitted. To copy otherwise, or republish, to post on servers or to redistribute to lists, requires prior specific permission and/or a fee. Request permissions from permissions@acm.org. _CIKM ’24, October 21–25, 2024, Boise, ID, USA_ 

Specifically, we construct a benchmark dataset comprising 3,224 queries, each containing a time specifier to add temporal constraints. Based on this dataset, we conduct comprehensive tests on five popular sparse and dense retrieval models, namely BM25 [27], Contriever [13], TAS-B [12], text-embedding-ada-002 of OpenAI, and jina-embeddings-v2-base-en of JinaAI [11]. The experimental results demonstrate that the top-performing retriever achieves a 

© 2024 Copyright held by the owner/author(s). Publication rights licensed to ACM. ACM ISBN 979-8-4007-0436-9/24/10 https://doi.org/10.1145/3627673.3679800 

2544 

CIKM ’24, October 21–25, 2024, Boise, ID, USA 

Feifan Wu et al. 


![](raw/papers/2_LLM_NLP/LLM_paper_list/rag/cikm_2024_Time-Sensitve_Retrieval-Augmented_Generation_for_Question_Answering_images/cikm_2024_Time-Sensitve_Retrieval-Augmented_Generation_for_Question_Answering.pdf-0002-02.png)


**Figure 1: Example of retriever failure when handling a timesensitive query. The correct and incorrect answers are highlighted in green and red .** 

nDCG@10[1] metric of only 41.40%, which is considerably lower than the 48.45% attained on the MTEB benchmark [11, 25]. Therefore, we propose constructing negative sample pairs during the contrastive learning process by altering the time specifiers in the questions to create a mismatch with the annotated time range of the facts in the answer-containing passages. This approach ensures that the key distinction between positive and negative pairs lies in the temporal information conveyed by the time specifiers. Consequently, the model learns to consider documents relevant to the temporal context, rather than merely those that are semantically related. 

Furthermore, we find the potential drawback of fine-tuning the retriever exclusively on time-sensitive data, which may compromise its effectiveness on broader tasks. Hence, we introduce an effective fine-tuning strategy that selectively targets the query encoder. This approach preserves the improvements for time-sensitive queries without necessitating frequent updates to the document indices. By dynamically routing queries to the most suitable encoder during inference, our method ensures robust performance across varied queries. This paper demonstrates that our approach not only addresses the specific challenge of time-sensitive queries but also maintains retrieval efficiency and accuracy for general tasks. 

The key contributions of this paper are as follows: 

- We construct a novel benchmark dataset to evaluate retriever performance on time-sensitive queries, consisting of 3,224 queries. 

- We provide a comprehensive analysis of the limitations of current retrievers in handling time-sensitive queries and identify the causes. 

- We propose a method that significantly enhances the ability of the retriever to process time-sensitive queries and validate the efficacy through extensive experimentation. 

## **2 Problem Analysis** 

To evaluate different retrievers handling time-sensitive queries, we perform an experimental study focused on constructing and applying a benchmark dataset. 

1nDCG@10 is a widely used evaluation metric that measures the quality of a ranked list of retrieved results by considering both the relevance of the retrieved documents and their positions in the ranked list, with higher weight given to more relevant documents at higher ranks, normalized to a range of 0 to 1 for easier interpretation and comparison across different queries or systems. 

## **2.1 Construction of the Dataset** 

Existing research lacks a dedicated open-domain QA dataset for time-sensitive queries. While previous works like TimeQA [4] have constructed datasets for time-sensitive QA. The questions in these datasets are often written based on documents from Wikipedia[2] , leading to the high semantic similarity between queries and documents. We believe these datasets are insufficient for accurately assessing the performance of retrievers in handling time-sensitive queries, as successful retrieval might be possible based solely on semantic similarity. 

To more accurately measure the performance of retrievers on time-sensitive queries, it is necessary to construct a new opendomain QA dataset that includes documents that are semantically similar but differ in temporal aspects. This will enable a better evaluation and enhancement of the accuracy and efficiency of retrievers in processing temporal information. 

_Dataset collection._ To collect data for our benchmark dataset, we first gather structured data containing temporal information from platforms such as Kaggle[3] . For example, we use the Nobel Prize dataset, which contains detailed factual information for each entry, such as the winner’s name, award year, motivation for the award, place and time of birth and death, proportion of prize money, and affiliations at the time of the award. For instance, the entry for Albert Einstein details his 1921 Nobel Prize in Physics, his birth on March 14, 1879, in Ulm, Germany, his death on April 18, 1955, in Princeton, New Jersey, USA, his award motivation, and his affiliation with the Kaiser Wilhelm Institute for Physics in Berlin. We use manually defined templates to convert the structured data of each entry into textual data. This template-based technique ensures that these texts have similar semantic information and text structure. Each generated text from an entry is considered an independent document, forming the knowledge base to be retrieved. 

_Query construction._ To construct queries, we identify the types of time-evolving facts present in our dataset, including award year, and birth and death dates of laureates. We develop various query templates based on these facts, such as “Who was the winner of the Nobel Peace Prize [time specifier]?” and “Who was born [time specifier] and won the Nobel Prize in Physics?” For the generation of time specifiers, we select two typical temporal relationships: “in” and “between”. We further differentiate time specifiers by granularity, distinguishing between year and decade. This approach yields a range of time specifiers, such as “in 1990”, “in the early 1990s”, and “between 1990 and 1993”, which we integrate into our query templates. To establish the relevance of a document to a given query, we utilize regular expressions to extract time-evolving facts from each document. We then compare these extracted facts against the award category and time specifier in the query. A document is considered relevant if it matches the award category of the query and is aligned with the specified time specifiers. The determination of such relevance is achieved through a rule-based matching process, which ensures precision in its execution. Comprehensive statistics of the benchmark dataset are shown in Table 1. 

> 2https://www.wikipedia.org 

> 3https://www.kaggle.com 

2545 

CIKM ’24, October 21–25, 2024, Boise, ID, USA 

Time-sensitive Retrieval-Augmented Generation for Question Answering 

**Table 1: The statistics our dataset,** _**#Avg. D / Q**_ **indicates the average relevant documents per query,** _**#Avg. Words**_ **indicates the average number of words within the document.** 

|**Query**|**Doc**|**#Avg. D / Q**|**#Avg. Words**|
|---|---|---|---|
|3224|989|5.3|48.7|



## **2.2 Experiment Setting and Metrics** 

Our experiment procedure employs the naive RAG paradigm [10, 24] for sparse and dense retrievers. We begin by encoding the entire knowledge base using different embedding models. Then, each time-sensitive query is encoded using the same embedding model. Using cosine similarity scores, we select the top _𝑘_ relevant documents as our retrieved set. For sparse retrievers, we opt for BM25, a commonly used bag-of-words retriever. However, our primary focus is on the effectiveness of dense retrievers, as we believe that retrievers based on term-frequency mechanisms are inherently challenged in addressing time-sensitive issues. To evaluate the performance of different dense embedding models in the RAG framework for time-sensitive query retrieval, we selected four distinctive embedding models: text-embedding-ada-002 of OpenAI, jina-embeddings-v2-base-en of JinaAI, TAS-B, and Contriver. 

Our assessment of retriever performance utilizes Recall@ _𝑘_ and Precision@ _𝑘_ . Precision@ _𝑘_ measures the ratio of relevant documents in the top _𝑘_ results. Higher Precision signifies better identification of relevant documents, reducing the impact of irrelevant information. Conversely, Recall@ _𝑘_ assesses the ability of a model to retrieve all relevant documents, whereas higher Recall indicates thorough coverage of necessary information for accurate RAG system responses. Increasing _𝑘_ typically decreases Precision but increases Recall, presenting a balance between larger document retrieval and computational costs, with the risk of introducing irrelevant, noisy data. Additionally, we use nDCG@10 and mAP@10 for a nuanced comparison of models. Liu et al. [23] note the significant impact of the positioning of relevant information within the input context of LLMs on their performance, with the models demonstrating a proclivity for leveraging information presented at the very beginning. Unlike Precision and Recall, nDCG and mAP consider the order of retrieved results, with nDCG valuing highly ranked relevant results more. mAP calculates average Precision across queries, assessing Precision at each ranking level. Our dataset averages 5.3 relevant documents per query, with up to 10, nDCG@10 and mAP@10 offering detailed performance insights across models. 

## **2.3 Results and Analysis** 

We first summarize the performance of different retrievers on our dataset in Figure 2, results in detail are shown in Table 2 and Table 3. 

_Are the current retrievers capable of efficiently handling timesensitive queries?_ Our findings suggest otherwise. Current retrievers typically measure relevance to select _𝑘_ documents, representing their relevance through the distance between queries and documents, using metrics such as cosine similarity or dot product similarity. As expected, the performance of BM25 is lower than that 


![](raw/papers/2_LLM_NLP/LLM_paper_list/rag/cikm_2024_Time-Sensitve_Retrieval-Augmented_Generation_for_Question_Answering_images/cikm_2024_Time-Sensitve_Retrieval-Augmented_Generation_for_Question_Answering.pdf-0003-10.png)


**----- Start of picture text -----**<br>
BM25 Contriever JinaAI OpenAI TAS-B<br>100%<br>80%<br>60%<br>40%<br>20%<br>0%<br>1 3 5 10 20 50 100<br>The Number of Retrieved Documents<br>(a) Recall for different retrievers on our dataset.<br>BM25 Contriever JinaAI OpenAI TAS-B<br>100%<br>80%<br>60%<br>40%<br>20%<br>0%<br>1 3 5 10 20 50 100<br>The Number of Retrieved Documents<br>(b) Precision for different retrievers on our dataset.<br>Recall<br>Precision<br>**----- End of picture text -----**<br>


**Figure 2: Recall and Precision for retrievers on our dataset based on the number of retrieved documents (** _𝑘_ **).** 

of all dense retrievers. We attribute this to the fact that the frequency gap of term occurrences in different documents within the corpus is not significant when the topics in the corpus are consistent. This makes it difficult for BM25 to discern the finer-grained differences between documents. In contrast, dense retrieval based on semantic similarity is theorized to possess the capacity to handle time-sensitive queries. Consider the example mentioned in Figure 1, for the query, “Who was the winner of the Nobel Prize in Physics 1943?” and its relevant document “Otto Stern. The Nobel Prize in Physics 1943. ... TLDR (dozens of words)”, the time specifier is explicitly present, with the year 1943 appearing in both the query and the relevant document. In this case, no temporal reasoning is required, as the temporal constraints are explicitly presented in the query. However, the retrieval results are unexpectedly inaccurate, with the retriever incorrectly retrieving documents related to other Nobel Prize in Physics winners from different years. Experimental results displayed in Figure 2 demonstrate that current state-of-the-art neural network-based embedding models also struggle with time-sensitive queries. When _𝑘_ =1, even though a query may correspond to multiple relevant documents, the Precision of the best-performing retrieval model is only 38%, indicating that in most cases, the document with the highest similarity retrieved by the retriever is not the relevant document to the query. At this point, the document retrieved by the retriever is of no help and can even be a hindrance to the generator. At _𝑘_ =10, the best-performing jina-embeddings-v2-base-en model achieves only a 50% Recall. 

2546 

CIKM ’24, October 21–25, 2024, Boise, ID, USA 

Feifan Wu et al. 


![](raw/papers/2_LLM_NLP/LLM_paper_list/rag/cikm_2024_Time-Sensitve_Retrieval-Augmented_Generation_for_Question_Answering_images/cikm_2024_Time-Sensitve_Retrieval-Augmented_Generation_for_Question_Answering.pdf-0004-02.png)


**----- Start of picture text -----**<br>
JinaAI Contriever OpenAI TAS-B<br>70% 69.5%<br>60.8%<br>60%<br>50% 49.7%<br>40% 37.0% 39.3%<br>30% 28.2%<br>21.0%<br>20% 18.8%<br>10% 9.8% 7.1% 10.3% 9.4%<br>0%<br>explicit hybrid implicit<br>Query Types<br>Recall<br>**----- End of picture text -----**<br>


**Figure 3: Recall@10 for retrievers on different query types.** 

At _𝑘_ =100, considering that the total number of documents in the corpus is only 989 and the average number of relevant documents per query is less than 10, none of the models achieves a 100% Recall. 

_At what degree does the type of a time-sensitive query impact on the retrieval effect?_ Time-sensitive queries in our experiment can be classified into three types: explicit, implicit, and hybrid. An explicit time-sensitive query asks for events that occurred in a specific year, such as “Who was the winner of the Nobel Peace Prize in 1933?” In contrast, implicit and hybrid time-sensitive queries inquire about events over a period. An implicit time-sensitive query does not specify the starting and ending years, for example, “Who was born in the mid-1970s and won the Nobel Prize in Physics?”. A hybrid time-sensitive query specifies the starting and ending years, such as “Who won the Nobel Prize in Physics between 1968 and 1970?” 

We test multiple retrievers on different query settings to assess their retrieval effectiveness. As shown in the Figure 3, we observe a consistent trend: all dense retrievers performed well on explicit queries, slightly worse on hybrid queries, and poorly on most implicit queries. We hypothesize that this is related to the inference abilities of retrievers. In the explicit query setting, retrievers only need to search for events in a clearly specified year. However, in the implicit query setting, retrievers must first infer the time span from the query and then find the relevant passages, posing a significant challenge. In the hybrid query setting, although the starting and ending years are provided, the intermediate years are not explicitly stated, leading to a slight decrease in accuracy. 

_Why do current embedding models perform poorly on time-sensitive queries?_ Our hypothesis is that the sub-optimal performance of current models may stem from their tendency to over-emphasize certain information in the query while overlooking the time specifiers. We refer to this phenomenon as attention bias. This attention bias causes embedding models to retrieve documents that, although semantically similar, do not genuinely address the query and are, in fact, irrelevant. In our constructed queries, we incorporated both temporal information and award category information, specifically 

various Nobel Prize categories. When both the query and documents share identical Nobel Prize category information, the key to discerning the relevance of documents to the query lies in the constraint of time specifiers. 

To illustrate the impact of information beyond time specifiers, we quantify the frequency of erroneous retrievals of documents containing matching Nobel Prize category information in response to queries. The results, shown in Figure 4, depict the frequency of erroneous retrievals by four different dense retrieval models for queries about the Nobel Prize in Physics, focusing on the first 10 retrieved documents. The statistics reveal that all four models exhibit biases. When the specified Nobel Prize category information in the query aligns with the category information in the document, certain documents are more likely to be retrieved, even if the time range specified in the query does not match the corresponding time information in the document. 

Additionally, we observe that models with a flatter descent curve, indicating smaller variance, are more proficient in retrieving timesensitive queries. This finding supports our hypothesis that embedding models prioritize additional semantic information, which introduces biases derived from these semantics. If a model is more frequently exposed to information about certain Nobel laureates during the training phase, it could develop a preferential bias toward these laureates. This bias may result in not only accurate retrievals but also erroneous retrievals of documents related to these laureates in unrelated queries. 

In general, the lack of inference ability in retrievers and attention bias contribute to poor performance on time-sensitive queries. Users often provide vague hints about the time period they are interested in. Even if queries include a specific time span, it is unrealistic and sometimes impossible for users to list all the timestamps within that span. As a result, the retriever may overlook some potential timestamps that fit the requirements. For example, the year 1969 is a potential candidate but is not explicitly mentioned in the query “between 1968 and 1970.” In a more challenging case, the term “1990s”refers to the years between 1990 and 1999, but recognizing this connection requires strong reasoning ability from the retriever. Additionally, the attention bias can cause the retriever to overlook the temporal constraints and focus on other factors. 

## **3 Methodology** 

In this section, we propose a supervised contrastive learning approach to enhance the temporal sensitivity of dense retrievers for QA, incorporating query-side fine-tuning for scalability and a query routing mechanism to optimize performance. 

## **3.1 Method Overview** 

The details of our proposed framework are shown in Figure 5. Positive sample pairs consist of questions with time specifiers aligned to relevant passages within the annotated time ranges of timeevolving facts. Negative pairs are generated by modifying the time specifiers in questions to create mismatches with the passage time ranges. A contrastive loss function brings positive pairs closer together and pushes negative pairs farther apart in the embedding space during training. To balance performance across query types, we employ query-side training, which means fine-tuning only the 

2547 

CIKM ’24, October 21–25, 2024, Boise, ID, USA 

Time-sensitive Retrieval-Augmented Generation for Question Answering 


![](raw/papers/2_LLM_NLP/LLM_paper_list/rag/cikm_2024_Time-Sensitve_Retrieval-Augmented_Generation_for_Question_Answering_images/cikm_2024_Time-Sensitve_Retrieval-Augmented_Generation_for_Question_Answering.pdf-0005-02.png)


**----- Start of picture text -----**<br>
Contriever JinaAI OpenAI TAS-B<br>500<br>400<br>300<br>200<br>100<br>0<br>10 20 30 40 50<br># different documents.<br>Frequency<br>**----- End of picture text -----**<br>


**Figure 4: Frequency of incorrect retrieval of different documents. # different documents refer to the documents sharing identical Nobel Prize in Physics information with query.** 

query encoder while keeping the document encoder fixed. This approach enables the reuse of pre-computed document indices. Additionally, a query routing mechanism classifies incoming queries as time-sensitive or non-time-sensitive at inference, routing them to the appropriate fine-tuned or original query encoder. 

Our methodology integrates contrastive learning tailored for temporal information, query-side fine-tuning for scalability, and query routing to optimize performance across different query categories. Detailed descriptions follow in subsequent subsections. 

## **3.2 Supervised Fine-tuning** 

In this section, we outline the methodology to enhance the retrieval efficacy of dense retrievers for temporally sensitive queries using contrastive learning. We begin with a brief review of contrastive learning and the rationale for employing supervised contrastive learning. Next, we discuss our specific approach to constructing positive and negative sample pairs with temporal constraints. Finally, we describe the contrastive loss function used during training. 

_Contrastive learning._ Contrastive learning is a pivotal technique in machine learning, particularly within unsupervised learning paradigms. It fundamentally aims to cultivate representations by contrasting positive pairs against negative pairs [15]. In a selfsupervised setting, since there are no available labels, positive sample pairs are usually formed by data augmentation of the anchor, while negative sample pairs are formed by pairing the anchor with randomly selected samples from a small batch. Supervised contrastive learning allows us to effectively utilize label information [20]. Specifically, it enables us to directly use the information of time specifiers in the text to explicitly define positive and negative sample pairs. Compared to self-supervised methods, this approach allows for more effective optimization of the model’s embedding space. Expressions with similar temporal meanings are brought closer together in the embedding space, while expressions that have different temporal meanings are pushed farther apart. 

_Constructing positive sample pairs._ To create positive samples, we utilize an existing dataset tailored for time-sensitive QA. This dataset identifies time-evolving facts from Wikidata [30], where these facts are manually annotated to demarcate their temporal range within Wikidata passages. Using various templates, different QA pairs are synthesized based on these annotated facts, resulting in approximately 20,000 question-answer pairs addressing 5,500 distinct time-evolving facts. These queries are inherently timesensitive, embedding specific time specifiers within them, and the corresponding answers vary depending on these temporal markers. For each query, the dataset provides not only the precise answer but also the relevant passage from the source text that can address the query. Consequently, we treat a question and its corresponding answer-containing passage as a sample pair ( _𝑞,𝑑_ ). It is noteworthy that not all passages in these sample pairs contain time specifiers; some may only encompass the answer, with the relevant temporal markers embedded in the surrounding context of the passage. Given our focus on the informational content carried by the time specifiers in both questions and passages, we further refine our collected sample pairs. We stipulate that the passage in a sample pair must contain time specifiers that align with the time range annotated for the corresponding “time-evolving fact” that generated the question. For instance, for a query like, “The international governing body called World Athletics had which title officially from 2005 to Apr 2017?”, where the annotated time range of the time-evolving fact is “2001 - 2019”, we require the passage to exhibit temporal markers congruent with “2001 - 2019”. Sample pairs failing to meet this criterion are discarded. After refinement, our dataset encompasses a total of 8,600 valid positive sample pairs. 

_Constructing negative sample pairs._ An equally crucial component of contrastive learning is the formulation of negative sample pairs. The objective is to condition the model to prioritize documents relevant to the temporal context, rather than those that are merely semantically related. For a given positive sample pair ( _𝑞,𝑑_ ), we avoid the typical approach of randomly sampling negative samples from other instances or selecting other paragraphs from the same passage [5, 9]. While these methods can distinguish between positive and negative pairs based on semantic information, they do not sufficiently emphasize the importance of temporal features. 

As illustrated in Figure 6, we adopt the following strategy for constructing negative sample pairs: For a given positive sample pair ( _𝑞_[+] _,𝑑_ ), we retain paragraph _𝑑_ and modify the time specifiers in question _𝑞_ to create a mismatch with the time range of the annotated time-evolving fact, resulting in a negative sample pair ( _𝑞_[−] _,𝑑_ ). The rationale behind adjusting question _𝑞_ rather than paragraph d is based on the fact that question _𝑞_ is generated from a single annotated time-evolving fact using a template, whereas paragraph _𝑑_ may encompass several time-evolving facts. It is challenging to pinpoint the specific time specifiers in paragraph _𝑑_ that correlate with the time-evolving fact in question _𝑞_ . Furthermore, altering time specifiers in paragraph _𝑑_ might lead to semantic inconsistencies with other time-evolving facts present in the same paragraph. Given that certain paragraphs _𝑑_ might serve as relevant documents for multiple similar-type questions _𝑞_ (differing only in time specifiers), we ensure that the modified time specifiers in _𝑞_[−] do not align with the time ranges of any time-evolving facts associated 

2548 

CIKM ’24, October 21–25, 2024, Boise, ID, USA 

Feifan Wu et al. 


![](raw/papers/2_LLM_NLP/LLM_paper_list/rag/cikm_2024_Time-Sensitve_Retrieval-Augmented_Generation_for_Question_Answering_images/cikm_2024_Time-Sensitve_Retrieval-Augmented_Generation_for_Question_Answering.pdf-0006-02.png)


**Figure 5: The overview of our framework. During training, we construct a time-constrained negative sample to perform contrastive learning. During inference, a query router is initially employed to determine whether the fine-tuned query encoder or the query encoder before fine-tuned should be utilized based on whether the query is time-sensitive.** 


![](raw/papers/2_LLM_NLP/LLM_paper_list/rag/cikm_2024_Time-Sensitve_Retrieval-Augmented_Generation_for_Question_Answering_images/cikm_2024_Time-Sensitve_Retrieval-Augmented_Generation_for_Question_Answering.pdf-0006-04.png)


**Figure 6: Schematic diagram of constructing the negative sample pair.** 

with these questions. By generating negative samples through the modification of time specifiers in question _𝑞_ , we ensure a rapid and rule-based generation of a large pool of negative samples while keeping the other semantic information consistent in both positive and negative pairs. The key differentiator in distinguishing positive and negative pairs is solely the information encapsulated in the adjusted time specifiers in the question. 

_Training Loss._ Our training process utilized a contrastive InfoNCE loss function. Specifically, given an anchor document _𝑑_ , a relevant query _𝑞_[+] , and a set of irrelevant queries { _𝑞𝑖_[−][}] _[𝑖]_[=][1] _[..𝐾]_[, the] contrastive loss is defined as: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/rag/cikm_2024_Time-Sensitve_Retrieval-Augmented_Generation_for_Question_Answering_images/cikm_2024_Time-Sensitve_Retrieval-Augmented_Generation_for_Question_Answering.pdf-0006-08.png)


where _𝑓_ (·) represents the feature extraction function, and _𝜏_ is the temperature parameter in the contrastive loss, used to scale the similarity measure between samples. 

## **3.3 Training and Inference Strategy** 

In our study, we observe that while training on our dataset significantly enhanced the performance of the retriever on time-sensitive queries, it conversely led to a decline in performance on non-timesensitive queries compared to pre-tuned models. To address this issue, we employ query-side training and query-routing strategies during the training and inference phases. This approach not only improves the retriever’s efficacy in handling temporally sensitive queries but also maintains its performance on non-temporal queries. By effectively balancing the model’s capabilities across different types of queries, we ensure overall retrieval quality. 

_Query-side training._ The Bi-encoder architecture features two independent encoding processes that separately encode two groups of input texts, generating two sets of independent representations. A key characteristic of the Bi-encoder architecture is the independence of its encoding processes, meaning the output of one encoder does not influence the output of the other encoder. Since this task involves only the text modality, existing work typically uses the same encoder to independently encode both documents and queries. In addition to the approach of simultaneously fine-tuning a single encoder for both documents and queries as shown in Figure 7 (a), we also adopt an alternative query-side fine-tuning strategy as shown in Figure 7 (b): In this strategy, only the encoder used for queries is fine-tuned, while the encoder for documents remains fixed. We 

2549 

CIKM ’24, October 21–25, 2024, Boise, ID, USA 

Time-sensitive Retrieval-Augmented Generation for Question Answering 


![](raw/papers/2_LLM_NLP/LLM_paper_list/rag/cikm_2024_Time-Sensitve_Retrieval-Augmented_Generation_for_Question_Answering_images/cikm_2024_Time-Sensitve_Retrieval-Augmented_Generation_for_Question_Answering.pdf-0007-02.png)


**Figure 7: Diagram of our training strategy for Bi-encoder architecture retriever.** 

believe the second strategy offers greater scalability. In real-world applications, the document corpus can be vast. By keeping the original document encoder unchanged, we can utilize previously pre-computed and stored document indices, eliminating the need to recalculate them when training new query encoders. 

_Query-routing._ In the RAG system, query routing selects the optimal path. Various implementations of routing exist, including LLM Function Calling Routers, Classification Routers, and Semantic Routers. We adopt a lightweight Classification Router approach, which routes queries by classifying them into time-sensitive and non-time-sensitive categories. Specifically, we used training data from TimeQA and Natural Questions [21], with embeddings encoded by the Contriever model as inputs to a two-layer binary classification neural network. We trained the network on 8,000 samples for three epochs, achieving 96% test accuracy. 

## **4 Evaluation** 

In this section, we evaluate the retrievers fine-tuned through contrastive learning, specifically designed for time-sensitive queries, which we refer to as TsContriever. 

## **4.1 Setting** 

_Implementation._ We utilize the Contriever model as our foundational pre-trained model. During the training phase, we finetuned the pre-trained model on our training dataset over a total of 5 epochs. We fine-tuned two versions of the model: one with query-side fine-tuning and the other without. For our training, we employ the open-source fine-tuning code provided by Izacard et al. [13]. We adhere to the default settings specified in the open-source code, including the same learning rate, batch size, and other hyperparameters as defined by the original authors. During the inference phase, we integrate a new model equipped with a query router. Depending on its characteristics, the query is directed either to the Contriever model or to the query-side fine-tuned model. 

_Datasets._ We utilize the benchmark dataset constructed in Section 2.1 and the same evaluation metrics for assessment. We also utilize the Natural Questions dataset from the BEIR benchmark along with the nDCG@10 and mAP@10 metrics to assess the performance of our model on general queries. 

## **4.2 Results and Analysis** 

Table 3 presents the performance of TsContriever on the nDCG@10 and mAP@10 metrics, in comparison with other dense retriever models introduced in Section 2.2. The results demonstrate the exceptional performance of TsContriever on these key performance indicators, significantly surpassing the Contriever, TAS-B, jinaembeddings-v2-base-en, and text-embedding-ada-002 models. 

Notably, TsContriever achieves a score of 0.60 on nDCG@10, substantially exceeding the performance of other models. This indicates its higher relevance scoring in the top 10 retrieved documents. Similarly, on the mAP@10 metric, TsContriever leads with a score of 0.47, further proving its superiority in providing high-quality, relevant documents. 

Table 2 offers a more comprehensive analysis of performance, including multiple levels of Recall and Precision. From these results, it is observed that TsContriever achieves the highest scores in Recall and Precision across all levels. Particularly, at Recall@1, TsContriever reaches a score of 19.92, almost double that of other models, underscoring its efficiency in identifying the most relevant documents. The Precision results display a similar trend, with TsContriever outperforming other models at all evaluation levels. 

These results highlight the robust performance of TsContriever in time-sensitive retrieval tasks. By precisely tuning the contrastive learning process, TsContriever demonstrates exceptional capabilities in handling time-sensitive queries. These findings attest to the potential of contrastive learning methods in designing retrieval systems tailored to specific needs. 

In this discourse, we elucidate the efficacy of TsContriever, a model refined through the nuanced process of fine-tuning the foundational Contriever framework. This enhancement, underscored by the method of contrastive learning, has been specifically orchestrated to augment the model’s proficiency in managing timesensitive retrieval tasks. The development of TsContriever is rooted in the recognition of inherent biases in the training data, which previously hampered the original performance of Contriever in time-sensitive contexts. This fine-tuning exercise on specialized datasets has markedly amplified the retrieval capabilities of models for time-sensitive queries. 

However, it is imperative to acknowledge that this method, while effective, is not devoid of limitations. The performance metrics, though improved, still indicate considerable scope for enhancement. Furthermore, the specificity of fine-tuning on particular datasets might inadvertently reduce the model’s efficacy in other, non-targeted tasks. 

To illustrate the disparity in performance between our approach and other models on non-target tasks, we conducted experiments on the Natural Questions dataset. As depicted in Table 3, experimental results indicate a decrease in performance for our finetuned model on non-target tasks. To explore the impact of different fine-tuning configurations on target and non-target tasks, we attempted various methods for handling document and query encoders. TsContrieveronly-q in the experimental results denotes freezing the document encoder and solely fine-tuning the query encoder. Under this configuration, the model exhibits a decrease in performance on temporal questions compared to previous iterations, yet it still outperforms other models and shows improvement on the 

2550 

CIKM ’24, October 21–25, 2024, Boise, ID, USA 

Feifan Wu et al. 

**Table 2: Model Recall and Precision results of our datasets. #JinaAI refers to the jina-embeddings-v2-base-en model. #OpenAI refers to the text-embedding-ada-002 model. TsContrieveronly-q refers to freezing the document encoder and solely fine-tuning the query encoder.** 

|**Model**|**Recall**<br>R@1<br>R@5<br>R@10<br>R@20<br>R@50<br>R@100|**Precision**<br>P@1<br>P@5<br>P@10<br>P@20<br>P@50<br>P@100|
|---|---|---|
|Contriever<br>TAS-B<br>#JinaAI<br>#OpenAI|4.72<br>16.89<br>28.14<br>43.78<br>68.40<br>85.17<br>2.62<br>10.17<br>18.40<br>31.02<br>55.18<br>75.56<br>12.77<br>35.58<br>50.10<br>65.97<br>84.15<br>94.46<br>9.50<br>29.56<br>40.93<br>55.33<br>77.53<br>92.84|17.69<br>13.14<br>11.12<br>8.90<br>5.50<br>3.43<br>9.49<br>7.27<br>6.56<br>5.56<br>4.03<br>2.85<br>38.47<br>24.68<br>18.19<br>12.38<br>6.57<br>3.77<br>29.93<br>20.38<br>14.75<br>10.31<br>6.01<br>3.68|
|TsContriever<br>TsContrieveronly-q<br>+ query-router|**19.92**<br>**52.70**<br>**69.13**<br>**81.52**<br>**91.53**<br>94.57<br>11.62<br>38.96<br>56.38<br>74.44<br>89.88<br>94.82<br>10.94<br>36.71<br>52.95<br>71.01<br>88.94<br>**95.11**|**58.63**<br>**36.49**<br>**25.34**<br>**15.72**<br>**7.30**<br>3.81<br>41.27<br>29.15<br>21.60<br>14.54<br>7.18<br>3.81<br>39.49<br>27.67<br>20.40<br>14.02<br>7.15<br>**3.83**|



**Table 3: Model nDCG@10 and mAP@10 Results of our datasets and NQ. #JinaAI refers to the jina-embeddings-v2base-en model. #OpenAI refers to the text-embedding-ada002 model. TsContrieveronly-q refers to freezing the document encoder and solely fine-tuning the query encoder.** 

|**Model**|**Our Dataset**<br>**NaturalQuestions**|
|---|---|
||nDCG@10<br>mAP@10<br>nDCG@10<br>mAP@10|
|Contriever<br>TAS-B<br>#JinaAI<br>#OpenAI|21.51<br>13.93<br>25.37<br>**19.68**<br>13.10<br>8.13<br>**46.30**<br>_<br>41.40<br>29.86<br>_<br>_<br>33.51<br>23.42<br>_<br>_|
|TsContriever<br>TsContrieveronly-q<br>+ query router|**60.09**<br>**47.30**<br>10.71<br>7.29<br>45.91<br>32.88<br>19.24<br>14.29<br>43.21<br>30.87<br>25.35<br>19.67|



**Table 4: Examples of time-sensitive queries where TsContriever successfully retrieves relevant documents in the Top-1 result, while other retrievers fail within the Top-10.** 

||**Query**|**Top-1 Retrieve Result**|
|---|---|---|
||Who was born in<br>the early 1840s and<br>won the Nobel<br>Prize in Medicine?|Robert Koch. The Nobel Prize in<br>Medicine 1905. Born: 11, Dec, 1843,<br>Clausthal (now Clausthal-Zellerfeld) ...|
||Who was the winner|Gary Becker. The Nobel Prize in|
||of the Nobel Prize|Economics 1992. Born: 2, Dec,|
||in Economics 1992?|1930, Pottsville PA, USA ...|
||Who won the Nobel|International Committee of the|
||Peace Prize between|Red Cross. The Nobel Peace|
||1914 and 1917?|Prize 1917. Founded: 1863 ...|



Natural Questions dataset compared to TsContriever. To further balance the performance on target and non-target tasks, we introduce an additional query router model based on TsContrieveronly-q. This model first categorizes queries into temporal and non-temporal questions, subsequently employing the fine-tuned query encoder and the query encoder before fine-tuning for each respective category. This approach further enhances the effectiveness of our method on the Natural Questions dataset, nearly aligning its performance with that of the original Contriever model before fine-tuning. 

## **4.3 Case Study** 

The Table 4 intuitively shows cases where our retriever successfully returns relevant documents in the top-1 result for time-sensitive queries, while other retrievers fail to recall any relevant documents within the top-10. For queries like “Who was born in the early 1840s and won the Nobel Prize in Medicine?”, “Who was the winner of the Nobel Prize in Economics 1992?”, and “Who won the Nobel Peace Prize between 1914 and 1917?”, our method demonstrates its superior ability to handle temporal constraints by ranking truly relevant documents at the top, unlike other retrievers. This highlights the effectiveness of our approach for time-sensitive retrieval. 

## **5 Related Work** 

## **5.1 Retriever in QA** 

In open-domain QA, retrievers are employed to extract relevant documents pertaining to a given query to provide candidate contexts [1, 2]. Although LLMs are capable of performing open-domain QA tasks without any provided context, they rely on a static training corpus. Once training is completed, LLMs cannot access data beyond their training cutoff date. This limitation can result in inaccurate responses, outdated answers, or hallucinations when queried about topics that fall outside their training data [35]. RAG enhances the capabilities of LLMs by retrieving relevant information from external knowledge bases. This approach significantly reduces the risk of generating erroneous information [10, 22]. 

Retrieval is achieved by calculating the similarity between question embeddings and document embeddings, often using similarities such as cosine and dot product. The semantic representation capability of the embedding model plays a crucial role in this process. Depending on the type of embedding model, retrievers can be categorized into sparse and dense retrievers. 

For sparse retrievers, documents and queries are typically represented as sparse vectors, where each element of the vector corresponds to a term in the vocabulary. The most well-known method for assessing the significance of a specific term within a document 

This section briefly summarizes related work from two perspectives: retrievers in the QA domain and time-sensitive QA. 

2551 

CIKM ’24, October 21–25, 2024, Boise, ID, USA 

Time-sensitive Retrieval-Augmented Generation for Question Answering 

is TF-IDF, which assigns weights based on the inverse document frequency and term frequency. BM25, an evolution of TF-IDF, is still widely used today. 

For dense retrievers, documents and queries are encoded into vector space primarily to utilize pre-trained language models [7]. For the QA task, DPR [18] proposes a dense paragraph retriever based on a bi-encoder architecture. It uses positive examples from the QA dataset and hard negative examples from BM25 for discriminative training. Further advancements, such as the implementation of ANCE [32] for model-intrinsic hard negative mining during the optimization process, have enhanced training efficiencies. Besides using supervised learning, Contriever employs contrastive learning in an unsupervised setting using the Wikipedia corpus and CCNet data for pre-training, ensuring broad knowledge coverage. The Bi-encoder architecture interacts by independently encoding documents and questions into vectors, which limits the possibility of more fine-grained interactions at the token level. To address this issue, Nogueira and Cho [26] introduce the cross-encoder architecture, which jointly encodes questions and documents. Since the cross-encoder architecture cannot pre-compute and store document index, its retrieval efficiency is lower than that of the bi-encoder architecture, making it commonly used as a re-ranking method. 

Knowledge distillation can be used to train retrievers. TAS-B uses dual-supervision knowledge distillation by combining pairwise and in-batch teacher signals to enhance learning efficiency and effectiveness. BGE-M3 [3] proposes a novel self-knowledge distillation method, where relevance scores from different retrieval features can be integrated as teacher signals to enhance encoder training quality. 

To align the retriever with the generator, there are efforts to use results from LLMs as the supervision signal for fine-tuning, known as language model-supervised retrieval. PROMPTAGATOR [6] utilizes LLMs as few-sample query generators to create taskspecific retrievers, addressing the challenges in supervised finetuning, particularly in data-scarce domains. Another approach, Embedder [34], leverages LLMs to generate reward signals across multiple downstream tasks, with retrievers fine-tuned using both dataset labels and rewards from LLMs. This dual-signal method allows for more effective fine-tuning, making embedding models adaptable to various downstream applications. REPLUG [28] utilizes the retriever and LLMs to compute the probability distribution of retrieved documents, then supervises training through the computation of Kullback-Leibler (KL) divergence. 

Existing methods propose improvements to retrievers through techniques such as contrastive learning, knowledge distillation, and supervision by LLMs. However, according to our literature review, there has not yet been work specifically analyzing and enhancing the performance of retrievers on time-sensitive queries. 

## **5.2 Time-Senstive QA** 

Research on time-sensitive QA has made certain leaps, with the emergence of significant datasets and typical methods. The TempQuestions dataset [16] comprises 1,271 questions with time signals sourced from Freebase, including explicit dates, implicit event references, and ordinal numbers. The TimeQA dataset contains 20,000 

question-answer pairs across 70 relationships, covering 5,500 evolving facts, and features two subsets: a “simple” set with explicit time ranges and a “hard” set that requires implicit time reasoning. Besides, the SituatedQA dataset [33] emphasizes time reasoning using Wikipedia context and includes 4,000 open-domain questions. Recently, Wei et al. [31] focus on time ranges, temporal ordering, and counterfactual queries, introducing the MenatQA dataset. 

Existing methods for addressing time-sensitive QA via RAG primarily improve in two ways: updating the knowledge base and considering temporal context information. Kasai et al. [19] propose a model using a simple RAG with Wikipedia snapshots to address real-time issues as information evolves over time. Similarly, Atlas [14] utilizes static Wikipedia snapshots but swaps the knowledge base based on temporal context. Zhang and Choi [33] attempt to provide temporal context by modifying queries to tackle time-sensitive open-domain QA issues, yet it does not significantly enhance the capability to handle time-sensitive problems. This study also found that although updating the retrieval corpus or fine-tuning the language model with new data can refresh some world knowledge, this approach is inefficient in maintaining the timeliness of model responses. Gade and Jetcheva [8] consider documents relevant both semantically and temporally to the given query, rather than relying solely on semantic similarity. However, this method only calculates the score between the query and documents based on temporal context, ignoring the temporal information within the queries and documents themselves. In summary, although existing RAG models have made attempts and progress in dealing with time-sensitive issues, challenges still exist in processing real-time data updates, integrating temporal information, and accurately retrieving timerelevant documents. 

## **6 Conclusion and Limitation** 

This study investigates the limitations of widely used dense retrievers in RAG systems regarding their proficiency in accurately handling time-sensitive queries. First, we construct an open-domain QA dataset to evaluate the capabilities of different models on timesensitive queries. Our experiments demonstrate that a contributing factor to the sub-optimal performance of current dense retrievers in processing time-sensitive queries is the inherent neglect of time specifiers in text by the embedding models, due to biases in the training datasets. Finally, we introduce an enhancement technique utilizing contrastive learning, which notably improves the efficacy of the retrieval model in handling time-sensitive retrieval tasks. 

While our study presents several improvements, there are still limitations to address. One limitation is that it does not accommodate all types of constraints, such as spatial constraints. Another limitation is that our method has not been tested on more complex cases that involve specific events to indicate time periods. 

## **Acknowledgments** 

This work is supported by the National Key Research and Development Program of China (2022YFF0712400), the National Natural Science Foundation of China (U23B2057, 62176185, 62276063), the Natural Science Foundation of Jiangsu Province (BK20221457), and the Ant Group. 

2552 

CIKM ’24, October 21–25, 2024, Boise, ID, USA 

Feifan Wu et al. 

## **References** 

- [1] Danqi Chen, Adam Fisch, Jason Weston, and Antoine Bordes. 2017. Reading Wikipedia to Answer Open-Domain Questions. In _Proceedings of the 55th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)_ , Regina Barzilay and Min-Yen Kan (Eds.). Association for Computational Linguistics, Vancouver, Canada, 1870–1879. https://doi.org/10.18653/v1/P17-1171 

- [2] Danqi Chen and Wen-tau Yih. 2020. Open-domain question answering. In _Proceedings of the 58th annual meeting of the association for computational linguistics: tutorial abstracts_ . 34–37. 

- [3] Jianlv Chen, Shitao Xiao, Peitian Zhang, Kun Luo, Defu Lian, and Zheng Liu. 2024. BGE M3-Embedding: Multi-Lingual, Multi-Functionality, Multi-Granularity Text Embeddings Through Self-Knowledge Distillation. _ArXiv_ abs/2402.03216 (2024). https://api.semanticscholar.org/CorpusID:267413218 

- [4] Wenhu Chen, Xinyi Wang, and William Yang Wang. 2021. A Dataset for Answering Time-Sensitive Questions. In _Thirty-fifth Conference on Neural Information Processing Systems Datasets and Benchmarks Track (Round 2)_ . 

- [5] Yung-Sung Chuang, Rumen Dangovski, Hongyin Luo, Yang Zhang, Shiyu Chang, Marin Soljacic, Shang-Wen Li, Scott Yih, Yoon Kim, and James Glass. 2022. DiffCSE: Difference-based Contrastive Learning for Sentence Embeddings. In _Proceedings of the 2022 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies_ , Marine Carpuat, Marie-Catherine de Marneffe, and Ivan Vladimir Meza Ruiz (Eds.). Association for Computational Linguistics, Seattle, United States, 4207–4218. https://doi.org/10.18653/v1/2022.naacl-main.311 

- [6] Zhuyun Dai, Vincent Y Zhao, Ji Ma, Yi Luan, Jianmo Ni, Jing Lu, Anton Bakalov, Kelvin Guu, Keith Hall, and Ming-Wei Chang. 2022. Promptagator: Few-shot Dense Retrieval From 8 Examples. In _The Eleventh International Conference on Learning Representations_ . 

- [7] Jacob Devlin, Ming-Wei Chang, Kenton Lee, and Kristina Toutanova. 2019. BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding. In _Proceedings of the 2019 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, Volume 1 (Long and Short Papers)_ , Jill Burstein, Christy Doran, and Thamar Solorio (Eds.). Association for Computational Linguistics, Minneapolis, Minnesota, 4171–4186. https://doi. org/10.18653/v1/N19-1423 

- [8] Anoushka Gade and Jorjeta Jetcheva. 2024. It’s About Time: Incorporating Temporality in Retrieval Augmented Language Models. arXiv:2401.13222 [cs.IR] 

- [9] Tianyu Gao, Xingcheng Yao, and Danqi Chen. 2021. SimCSE: Simple Contrastive Learning of Sentence Embeddings. In _Proceedings of the 2021 Conference on Empirical Methods in Natural Language Processing_ . 6894–6910. 

- [10] Yunfan Gao, Yun Xiong, Xinyu Gao, Kangxiang Jia, Jinliu Pan, Yuxi Bi, Yi Dai, Jiawei Sun, Qianyu Guo, Meng Wang, and Haofen Wang. 2024. Retrieval-Augmented Generation for Large Language Models: A Survey. arXiv:2312.10997 [cs.CL] 

- [11] Michael Günther, Jackmin Ong, Isabelle Mohr, Alaeddine Abdessalem, Tanguy Abel, Mohammad Kalim Akram, Susana Guzman, Georgios Mastrapas, Saba Sturua, Bo Wang, et al. 2023. Jina Embeddings 2: 8192-Token General-Purpose Text Embeddings for Long Documents. _arXiv preprint arXiv:2310.19923_ (2023). 

- [12] Sebastian Hofstätter, Sheng-Chieh Lin, Jheng-Hong Yang, Jimmy Lin, and Allan Hanbury. 2021. Efficiently Teaching an Effective Dense Retriever with Balanced Topic Aware Sampling. In _Proceedings of the 44th International ACM SIGIR Conference on Research and Development in Information Retrieval_ (, Virtual Event, Canada,) _(SIGIR ’21)_ . Association for Computing Machinery, New York, NY, USA, 113–122. https://doi.org/10.1145/3404835.3462891 

- [13] Gautier Izacard, Mathilde Caron, Lucas Hosseini, Sebastian Riedel, Piotr Bojanowski, Armand Joulin, and Edouard Grave. 2022. Unsupervised Dense Information Retrieval with Contrastive Learning. arXiv:2112.09118 [cs.IR] 

- [14] Gautier Izacard, Patrick Lewis, Maria Lomeli, Lucas Hosseini, Fabio Petroni, Timo Schick, Jane Dwivedi-Yu, Armand Joulin, Sebastian Riedel, and Edouard Grave. 2024. Atlas: few-shot learning with retrieval augmented language models. _J. Mach. Learn. Res._ 24, 1, Article 251 (mar 2024), 43 pages. 

- [15] Ashish Jaiswal, Ashwin Ramesh Babu, Mohammad Zaki Zadeh, Debapriya Banerjee, and Fillia Makedon. 2020. A survey on contrastive self-supervised learning. _Technologies_ 9, 1 (2020), 2. 

- [16] Zhen Jia, Abdalghani Abujabal, Rishiraj Saha Roy, Jannik Strötgen, and Gerhard Weikum. 2018. Tempquestions: A benchmark for temporal question answering. _Companion Proceedings of the The Web Conference 2018_ (2018), 1057–1062. 

- [17] Mandar Joshi, Eunsol Choi, Daniel Weld, and Luke Zettlemoyer. 2017. TriviaQA: A Large Scale Distantly Supervised Challenge Dataset for Reading Comprehension. 

   - In _Proceedings of the 55th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)_ , Regina Barzilay and Min-Yen Kan (Eds.). Association for Computational Linguistics, Vancouver, Canada, 1601–1611. https: //doi.org/10.18653/v1/P17-1147 

- [18] Vladimir Karpukhin, Barlas Oguz, Sewon Min, Patrick Lewis, Ledell Wu, Sergey Edunov, Danqi Chen, and Wen-tau Yih. 2020. Dense Passage Retrieval for OpenDomain Question Answering. In _Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing (EMNLP)_ , Bonnie Webber, Trevor Cohn, Yulan He, and Yang Liu (Eds.). Association for Computational Linguistics, Online, 6769–6781. https://doi.org/10.18653/v1/2020.emnlp-main.550 

- [19] Jungo Kasai, Keisuke Sakaguchi, Ronan Le Bras, Akari Asai, Dragomir Radev, Noah A Smith, Yejin Choi, Kentaro Inui, et al. 2023. RealTime QA: What’s the Answer Right Now?. In _Thirty-seventh Conference on Neural Information Processing Systems Datasets and Benchmarks Track_ . 

- [20] Prannay Khosla, Piotr Teterwak, Chen Wang, Aaron Sarna, Yonglong Tian, Phillip Isola, Aaron Maschinot, Ce Liu, and Dilip Krishnan. 2020. Supervised contrastive learning. _Advances in Neural Information Processing Systems_ 33 (2020), 18661– 18673. 

- [21] Tom Kwiatkowski, Jennimaria Palomaki, Olivia Redfield, Michael Collins, Ankur Parikh, Chris Alberti, Danielle Epstein, Illia Polosukhin, Matthew Kelcey, Jacob Devlin, Kenton Lee, Kristina N. Toutanova, Llion Jones, Ming-Wei Chang, Andrew Dai, Jakob Uszkoreit, Quoc Le, and Slav Petrov. 2019. Natural Questions: a Benchmark for Question Answering Research. _Transactions of the Association of Computational Linguistics_ (2019). 

- [22] Patrick Lewis, Ethan Perez, Aleksandra Piktus, Fabio Petroni, Vladimir Karpukhin, Naman Goyal, Heinrich Küttler, Mike Lewis, Wen-tau Yih, Tim Rocktäschel, et al. 2020. Retrieval-augmented generation for knowledge-intensive nlp tasks. _Advances in Neural Information Processing Systems_ 33 (2020), 9459–9474. 

- [23] Nelson F Liu, Kevin Lin, John Hewitt, Ashwin Paranjape, Michele Bevilacqua, Fabio Petroni, and Percy Liang. 2024. Lost in the Middle: How Language Models Use Long Contexts. _Transactions of the Association for Computational Linguistics_ 12 (2024). 

- [24] Xinbei Ma, Yeyun Gong, Pengcheng He, Hai Zhao, and Nan Duan. 2023. Query Rewriting in Retrieval-Augmented Large Language Models. In _Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing_ . 5303–5315. 

- [25] Niklas Muennighoff, Nouamane Tazi, Loic Magne, and Nils Reimers. 2023. MTEB: Massive Text Embedding Benchmark. In _Proceedings of the 17th Conference of the European Chapter of the Association for Computational Linguistics_ . 2014–2037. 

- [26] Rodrigo Nogueira and Kyunghyun Cho. 2020. Passage Re-ranking with BERT. arXiv:1901.04085 [cs.IR] 

- [27] Stephen Robertson, Hugo Zaragoza, et al. 2009. The probabilistic relevance framework: BM25 and beyond. _Foundations and Trends® in Information Retrieval_ 3, 4 (2009), 333–389. 

- [28] Weijia Shi, Sewon Min, Michihiro Yasunaga, Minjoon Seo, Rich James, Mike Lewis, Luke Zettlemoyer, and Wen tau Yih. 2023. REPLUG: Retrieval-Augmented Black-Box Language Models. arXiv:2301.12652 [cs.CL] 

- [29] Alon Talmor and Jonathan Berant. 2018. The Web as a Knowledge-Base for Answering Complex Questions. In _Proceedings of the 2018 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, Volume 1 (Long Papers)_ , Marilyn Walker, Heng Ji, and Amanda Stent (Eds.). Association for Computational Linguistics, New Orleans, Louisiana, 641–651. https://doi.org/10.18653/v1/N18-1059 

- [30] Theo Van Veen. 2019. Wikidata. _Information Technology and Libraries_ 38, 2 (2019), 72–81. 

- [31] Yifan Wei, Yisong Su, Huanhuan Ma, Xiaoyan Yu, Fangyu Lei, Yuanzhe Zhang, Jun Zhao, and Kang Liu. 2023. MenatQA: A New Dataset for Testing the Temporal Comprehension and Reasoning Abilities of Large Language Models. In _Findings of the Association for Computational Linguistics: EMNLP 2023_ . 1434–1447. 

- [32] Lee Xiong, Chenyan Xiong, Ye Li, Kwok-Fung Tang, Jialin Liu, Paul N. Bennett, Junaid Ahmed, and Arnold Overwijk. 2020. Approximate Nearest Neighbor Negative Contrastive Learning for Dense Text Retrieval. (2020). 

- [33] Michael Zhang and Eunsol Choi. 2021. SituatedQA: Incorporating ExtraLinguistic Contexts into QA. In _Proceedings of the 2021 Conference on Empirical Methods in Natural Language Processing_ . 7371–7387. 

- [34] Peitian Zhang, Shitao Xiao, Zheng Liu, Zhicheng Dou, and Jian-Yun Nie. 2023. Retrieve Anything To Augment Large Language Models. arXiv:2310.07554 [cs.IR] 

- [35] Yue Zhang, Yafu Li, Leyang Cui, Deng Cai, Lemao Liu, Tingchen Fu, Xinting Huang, Enbo Zhao, Yu Zhang, Yulong Chen, Longyue Wang, Anh Tuan Luu, Wei Bi, Freda Shi, and Shuming Shi. 2023. Siren’s Song in the AI Ocean: A Survey on Hallucination in Large Language Models. arXiv:2309.01219 [cs.CL] 

2553 

