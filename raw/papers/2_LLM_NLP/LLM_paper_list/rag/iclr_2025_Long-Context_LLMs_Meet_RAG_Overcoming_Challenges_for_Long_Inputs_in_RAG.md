
![](raw/papers/2_LLM_NLP/LLM_paper_list/rag/iclr_2025_Long-Context_LLMs_Meet_RAG_Overcoming_Challenges_for_Long_Inputs_in_RAG_images/iclr_2025_Long-Context_LLMs_Meet_RAG_Overcoming_Challenges_for_Long_Inputs_in_RAG.pdf-0001-01.png)


## **Long-Context LLMs Meet RAG: Overcoming Challenges for Long Inputs in RAG** 

**Bowen Jin**[1 2 *] **, Jinsung Yoon**[1] **, Jiawei Han**[2] **and Sercan Ö. Arık**[1] 

1Google Cloud AI Research, 2University of Illinois at Urbana-Champaign 

**Retrieval-augmented generation (RAG) empowers large language models (LLMs) to utilize external knowledge sources. The increasing capacity of LLMs to process longer input sequences opens up avenues for providing more retrieved information, to potentially enhance the quality of generated outputs. It is plausible to assume that a larger retrieval set would contain more relevant information (higher recall), that might result in improved performance. However, our empirical findings demonstrate that for many long-context LLMs, the quality of generated output initially improves first, but then subsequently declines as the number of retrieved passages increases. This paper investigates this phenomenon, identifying the detrimental impact of retrieved "hard negatives" as a key contributor. To mitigate this and enhance the robustness of long-context LLM-based RAG, we propose both training-free and trainingbased approaches. We first showcase the effectiveness of retrieval reordering as a simple yet powerful training-free optimization. Furthermore, we explore training-based methods, specifically RAG-specific implicit LLM fine-tuning and RAG-oriented fine-tuning with intermediate reasoning, demonstrating their capacity for substantial performance gains. Finally, we conduct a systematic analysis of design choices for these training-based methods, including data distribution, retriever selection, and training context length.** 

## **1. Introduction** 

Retrieval-augmented generation (RAG) (Gao et al., 2023) empowers large language models (LLMs) to utilize external information sources by selecting the most relevant pieces from a large corpus (Zhao et al., 2023), thereby enhancing their effectiveness, customizability and efficiency in complex problem-solving. RAG can also mitigate issues such as factual inaccuracies (Augenstein et al., 2023) and hallucinations (Huang et al., 2023), which LLMs often exhibit when confronted with knowledgeintensive tasks. RAG systems typically employ a retriever to identify relevant information from a corpus, which is then presented in the context of an LLM as the generator. 

Recent advances in computational resources and methodological innovations have enabled the development of LLMs that support increasingly longer context (Dubey et al., 2024; Reid et al., 2024). This has even opened up new avenues for directly inputting entire corpora or knowledge bases into the LLMs. Yet, it would still not be feasible for large corpora ( _e.g._ , Wikipedia) and can incur higher computational costs. Despite extensive research on RAG (Lee et al., 2024; Li et al., 2024; Xu et al., 2023), the interplay with long-context LLMs, particularly how to optimally design RAG systems using them effectively, remains under-explored. Existing works (Asai et al., 2024; Lin et al., 2024; Yoran et al., 2024) propose tuning LLMs for RAG, but predominantly focus on a limited number of retrieved passages (fewer than 10). Intuitively, longer context would allow for the inclusion of more retrieved passages, leading to higher recall and potentially improved performance. However, our findings reveal that this does not always hold true and highlight the need for a careful re-evaluation of standard RAG designs when utilizing long-context LLMs. We demonstrate that achieving optimal performance in such systems and to fully utilize the opportunities provided by the LLMs require a holistic rethinking and effective novel approaches to the unique challenges. 

_Corresponding author(s): bowenj4@illinois.edu_ 

* This work was done while Bowen was a student researcher at Google Cloud AI Research. 

Long-Context LLMs Meet RAG: Overcoming Challenges for Long Inputs in RAG 

This paper presents comprehensive analyses on long-context LLMs in RAG systems. Contrary to the suggestions of previous work (Li et al., 2024; Xu et al., 2023), our research reveals that increasing the number of retrieved passages does not consistently improve performance with long-context LLMs (Section 3.1). Instead, we observe that the generative modeling performance initially increases and then declines – simply providing more retrieved passages does not guarantee better outcomes. Using stronger retrievers is also not a mitigation mechanism – indeed the performance degradation can even be more severe with them. For deeper understanding of the phenomenon, we conduct further investigations, which reveal that increasing the number of retrieved passages can introduce irrelevant information (“noise”) that can mislead the LLM generation (Section 3.2). We also examine the impact of “hard negatives” of different retrievers on the LLMs, and show that there are scenarios where the ‘hard negatives’ from stronger retrievers might confuse the LLM generation even more than those from weaker retrievers (Section 3.3). 

To address the challenges identified in our analyses, we propose three methods, encompassing both training-free and training-based approaches, to enhance the performance of long-context LLMs in RAG applications: (1) Retrieval reordering: recognizing the "lost-in-the-middle" phenomenon observed for long-context LLMs (Liu et al., 2024), we propose reordering retrieved documents based on their retrieval scores. By prioritizing documents with higher scores at the beginning and end of the input sequences, we guide the LLMs’ attention towards more relevant information and mitigate the impact of hard negatives. (2) Implicit robustness fine-tuning: given the ability to handle noisy retrieved context is not explicitly acquired during standard LLM training, we propose tuning the LLMs with the data comprising queries and retrieved documents, including those with potential noise. This encourages the LLMs to implicitly learn robustness to hard negatives. (3) Explicit relevance fine-tuning: while the previous method implicitly enhances robustness, it does not explicitly teach the LLMs to identify relevant documents. Therefore, we propose augmenting the LLM tuning with an intermediate reasoning step, where the LLMs are trained to analyze the retrieved documents and explicitly identify relevant information before generating the final output. This approach aims to improve the LLMs’ ability to discern relevant information from noise within the retrieved context. 

Overall, the main contributions can be summarized as follows: 

- Systematic analysis of long-context RAG: we systematically analyze the use of long-context LLMs in RAG systems, specifically examining the impact of retrieved "hard negatives" on performance. 

- Novel methods for robust RAG: we propose three methods to improve the robustness of longcontext LLMs in RAG: (1) a training-free method based on retrieval reordering, (2) implicit tuning for robustness to hard negatives and (3) explicit tuning with intermediate reasoning for relevance identification. Overall, our proposed approaches show significant accuracy and robustness improvements on long-context RAG performance. 

- Comprehensive study of RAG-specific LLM tuning: we conduct a thorough investigation into various factors influencing the effectiveness of RAG-specific tuning, including data distribution, the employed retriever, and training context length. 

## **2. Related Work** 

Large language models (LLMs) can be prone to hallucinations especially at knowledge-intensive tasks (Augenstein et al., 2023; Huang et al., 2023; Zhao et al., 2023). Retrieval-augmented generation (RAG) addresses this by incorporating external knowledge sources to provide accurate and relevant information (Gao et al., 2023). Traditional RAG systems comprise a retriever to identify relevant information and a generator to synthesize the answer (Zhao et al., 2024; Zhu et al., 2021). While previous research focused on improving either the retriever (Izacard et al., 2021; Karpukhin et al., 

2 

Long-Context LLMs Meet RAG: Overcoming Challenges for Long Inputs in RAG 

2020; Wang et al., 2022) or the generator (Agarwal et al., 2024; Dong et al., 2022; Liu et al., 2024) in isolation, we take a holistic approach. Conducting comprehensive analyses of the entire RAG system, we focus on the challenges and opportunities presented by using long-context LLMs as generators. We propose novel solutions to better employ them in long-context RAG. 

Increased computational resources and advancements in efficient training methods have pushed LLMs supporting longer inputs (Wang et al., 2024; Zhou et al., 2024). While long-context LLMs (Reid et al., 2024) demonstrated impressive performance on benchmarks like "needle-in-the-haystack" (Kamradt, 2023) and RULER (Hsieh et al., 2024a), these benchmarks often rely on random negative examples and do not accurately reflect the challenges posed by the "hard negatives" encountered in real-world RAG scenarios (Cuconasu et al., 2024). Furthermore, existing studies on long-context LLMs in multi-document settings (Liu et al., 2024; Shi et al., 2023) often assume a single "golden" document and random negatives, which differs from the RAG context where multiple relevant passages and hard negatives may exist (Cuconasu et al., 2024; Hsieh et al., 2024b). Although some research has explored the relationship between RAG and long-context LLMs (Lee et al., 2024; Li et al., 2024; Xu et al., 2023), these works take different perspectives. They mainly focus on studying the (1) trade-offs between RAG and long-context LLMs (Xu et al., 2023), (2) routers to manage RAG and long-context LLMs (Li et al., 2024), (3) and the potential for LLMs to replace retrieval entirely (Lee et al., 2024), while leaving long-context LLMs as generators in RAG under-explored. We delve deeper into the potential benefits of long-context LLMs for RAG and investigate how to optimize these LLMs specifically for this application. 

Previous research has explored adapting LLMs for RAG using instruction tuning (Zhang et al., 2023). RetRobust (Yoran et al., 2024) fine-tunes LLMs with 1 retrieved relevant passage or random negative passage to make it robust to irrelevant passage. RA-DIT (Lin et al., 2024) conducts dual instruction tuning to make the LLM more effectively leverage retrieved information and retriever provide results more aligned with LLM preference. Self-RAG (Asai et al., 2024) introduces a framework to train a LM that dynamically retrieves passages, generates content, and evaluates the retrieved passages for improved performance. RAFT (Zhang et al., 2024) trains the LLMs to improve their ability to answer questions in “open-book” in-domain settings. More recently, RankRAG (Yu et al., 2024) tunes a LLM for the dual purpose of context ranking and answer generation in RAG. InstructRAG (Wei et al., 2024) finetunes the LLM to generate self-synthesized rationales rather than directly answering the question. However, these existing efforts primarily focus on tuning with a limited number of retrieved passages (typically fewer than 10) and do not fully leverage the potential of long-context LLMs. This work aims to address this gap by specifically investigating how to optimize long-context LLMs for large-scale RAG, where the number of retrieved passages can be significantly higher. 

## **3. Challenges of Long context LLMs in RAG** 

We present a systematic investigation into the challenges of utilizing long-context LLMs in RAG. Each subsection focuses on a specific research question, outlining corresponding experiments and analyzing the results on the key challenges. These insights inform the development of targeted solutions for improving RAG performance with long-context LLMs, which are presented in subsequent sections. 

## **3.1. The Effect of retrieved context size on RAG performance** 

This subsection investigates the relationship between the number of retrieved passages and the performance of long-context LLMs in RAG systems. 

**Research question.** Long-context LLMs offer the potential to incorporate more retrieved passages 

3 

Long-Context LLMs Meet RAG: Overcoming Challenges for Long Inputs in RAG 

into RAG systems. This raises a crucial question: _Does a larger volume of retrieved context consistently translate to better performance when using long-context LLMs in RAG?_ 

**Experimental setting.** We evaluate the performance of RAG systems on the Natural Questions (NQ) (Kwiatkowski et al., 2019) dataset using two different retrievers (BM25 (Robertson et al., 2009) and e5 (Wang et al., 2022), where e5 exhibits higher performance on NQ (Recall@40 is 0.90 with e5 and 0.73 with BM25)) and four long-context LLMs (Gemma-7B-Chat (Team et al., 2024a), Gemma-2-9BChat (Team et al., 2024b), Mistral-Nemo-12B-Instruct (Jiang et al., 2023) and Gemini-1.5-pro (Reid et al., 2024)). We systematically vary the number of passages retrieved by each retriever. 


![](raw/papers/2_LLM_NLP/LLM_paper_list/rag/iclr_2025_Long-Context_LLMs_Meet_RAG_Overcoming_Challenges_for_Long_Inputs_in_RAG_images/iclr_2025_Long-Context_LLMs_Meet_RAG_Overcoming_Challenges_for_Long_Inputs_in_RAG.pdf-0004-03.png)


**----- Start of picture text -----**<br>
0.60 0.60<br>0.55 0.55<br>0.50 0.50<br>0.45 0.45<br>0.40 0.40<br>0.35 0.35<br>0.30 Gemma-7B-Chat 0.30 Gemma-7B-Chat<br>Gemma-2-9B-Chat Gemma-2-9B-Chat<br>0.25 Mistral-Nemo-12B-Instruct 0.25 Mistral-Nemo-12B-Instruct<br>Gemini-1.5-Pro Gemini-1.5-Pro<br>0.20 0.20<br>10 [0] 10 [1] 10 [2] 10 [3] 10 [0] 10 [1] 10 [2] 10 [3]<br># Retrieved Passages # Retrieved Passages<br>(a) RAG performance with e5 retriever (b) RAG performance with BM25 retriever<br>RAG Accuracy RAG Accuracy<br>**----- End of picture text -----**<br>


Figure 1 | Impact of retrieved context size on RAG performance with 4 different LLMs on NQ. Increasing the number of retrieved passages initially improves performance but then leads to a decline. This degradation is more pronounced using a retriever (e5) that exhibits higher recall@k on NQ compared to BM25 (Recall@40 is 0.90 with e5 and 0.73 with BM25). The maximum number of retrieved passages varies across LLMs due to differences in their maximum token limits. 

**Observations.** Figure 1 presents the following key observations: 1) Strong Retriever (e5): Across all LLMs, increasing the number of retrieved passages initially improves performance, but then leads to a sharp decline or plateau. 2) Weak Retriever (BM25): Performance generally exhibits a continuous increase or a slight decrease as the number of retrieved passages increases. While these observations may appear counter-intuitive - given that one might expect monotonic improvements due to higher recall ( _i.e._ , a greater chance of retrieving relevant information) - the inclusion of additional documents can reduce precision, with irrelevant or misleading passages detracting LLMs from overall performance. Comparison of different retrievers and the results on other datasets are shown in Appendix A and B.1. 

**Insights.** The effectiveness of increasing retrieved context size in RAG depends on the strength of the retriever. With a strong retriever, performance exhibits an “inverted-U pattern”, while a weak retriever shows more consistent, albeit potentially limited, improvement. This suggests that factors beyond simply the amount of retrieved information are at play. 

## **3.2. The interplay of retrieval quality and LLM capabilities** 

This subsection delves into the factors hindering the performance of long-context LLMs in RAG, aiming to discern whether limitations arise from retrieval quality or the LLM’s ability to process the retrieved information. 

**Research question.** _Do the observed performance bottlenecks originate from limitations in the retriever’s_ 

4 

Long-Context LLMs Meet RAG: Overcoming Challenges for Long Inputs in RAG 

_ability to identify relevant information, or from the long-context LLM’s capacity to effectively utilize the retrieved context?_ 

**Experimental setting.** We analyze the relationship between RAG performance and retrieval quality, specifically recall and precision, using the Gemma-2-9B-Chat LLM with both e5 and BM25 retrievers (Figure 2). Recall@k measures the presence of relevant passages within the top-k retrieved passages, while precision@k quantifies the proportion of relevant passages among them. 


![](raw/papers/2_LLM_NLP/LLM_paper_list/rag/iclr_2025_Long-Context_LLMs_Meet_RAG_Overcoming_Challenges_for_Long_Inputs_in_RAG_images/iclr_2025_Long-Context_LLMs_Meet_RAG_Overcoming_Challenges_for_Long_Inputs_in_RAG.pdf-0005-03.png)


**----- Start of picture text -----**<br>
1.0 1.0<br>0.54 RAG accuracy 0.425 RAG accuracy Recall Precision<br>0.53 0.8 0.8<br>0.400<br>0.52<br>0.375<br>0.6 0.6<br>0.51<br>0.350<br>0.50<br>0.4 0.325 0.4<br>0.49<br>0.300<br>0.48 0.2 0.2<br>Recall 0.275<br>0.47 Precision<br>0.0 0.0<br>0 5 10 15 20 25 30 35 40 0 5 10 15 20 25 30 35 40<br># Retrieved Passages # Retrieved Passages<br>(a) Retrieval with e5 retriever (b) Retrieval with BM25 retriever<br>RAG accuracy RAG accuracy<br>Recall / Precision Recall / Precision<br>**----- End of picture text -----**<br>


Figure 2 | Analyzing the relationship between RAG performance and retrieval quality (recall/precision) using Gemma-2-9B-Chat with e5 and BM25 retrievers. (1) Accuracy vs. Recall: RAG accuracy consistently falls below retrieval recall for both retrievers, indicating that the presence of relevant information does not guarantee correct answers. This highlights the detrimental impact of irrelevant passages on LLM performance. (2) Precision and hard negatives: Despite higher precision with e5, the performance degradation with increasing retrieval size is more pronounced compared to BM25. This demonstrates that precision alone is an insufficient metric for assessing the impact of "hard negatives," as the nature of irrelevant information significantly influences LLM performance. 

**Observations.** Increasing the number of retrieved passages consistently leads to higher recall but lower precision, irrespective of the retriever used. Crucially, the overall accuracy of the RAG system falls below the recall across all retrieval sizes. This indicates that even when relevant information is present in the retrieved context, the LLM may fail to generate the correct answer. This demonstrates that the irrelevant retrieved passages can sometimes mislead the LLM. Furthermore, despite exhibiting higher precision, the e5 retriever leads to a more pronounced performance degradation as the number of retrieved passages increases compared to BM25. 

**Insights.** These observations yield two key insights: (1) _Influence of irrelevant passages_ : The discrepancy between retrieval recall and RAG accuracy underscores the detrimental effect of irrelevant retrieved passages ("hard negatives") on the LLMs’ performance. Even when relevant information is available, the presence of hard negatives can mislead the LLMs and hinder their ability to generate accurate answers. (2) _Limitations of precision as a metric_ : The contrasting performance trends observed with e5 and BM25, despite the former’s higher precision, reveal that precision alone is an inadequate measure of retrieval quality in this context, when the end-to-end performance is considered. The specific characteristics of the irrelevant passages, rather than just their quantity, significantly impact the LLMs’ performance. Retrievers might significantly differ in their way of priorization of them, and that might not be fully captured in metrics like precision. In this scenario, it is observed that “hard negatives” retrieved by a stronger retriever (e5) might even more detrimental to the LLM than those 

5 

Long-Context LLMs Meet RAG: Overcoming Challenges for Long Inputs in RAG 

retrieved by a weaker one (BM25). 

## **3.3. The importance of hard negatives for long-context LLM evaluation** 

This subsection investigates the impact of "hard negatives" on the performance of long-context LLMs in RAG, highlighting the need for more robust evaluation methodologies. 

**Research question.** In long-context RAG scenarios, where a vast knowledge source necessitates retrieving numerous passages, the likelihood of including relevant information ( _i.e._ obtaining high recall) increases. However, this also elevates the risk of introducing hard negatives. This raises two critical questions: (1) _How robust are current long-context LLMs to these hard negatives?_ and (2) _Does the impact of hard negatives vary with the retriever used?_ 

**Experimental setting.** This study investigates the effect of hard negative passages on long-context LLM performance in a controlled setting. We tasked three LLMs (Gemma2-7B-Chat, Mistral-Nemo12B-Instruct, and Gemini-1.5-Pro) with answering queries based on a context comprising a single golden passage and a varying number of hard negative passages retrieved using different methods (e5, Contriever, BM25, and random sampling). This synthetic experiment, detailed in Figure 3, isolates the impact of hard negatives by holding the golden passage constant and intentionally excluding scenarios with multiple golden passages, which are common in real-world RAG systems. See Appendix C for a complete illustration of the experimental setup. 


![](raw/papers/2_LLM_NLP/LLM_paper_list/rag/iclr_2025_Long-Context_LLMs_Meet_RAG_Overcoming_Challenges_for_Long_Inputs_in_RAG_images/iclr_2025_Long-Context_LLMs_Meet_RAG_Overcoming_Challenges_for_Long_Inputs_in_RAG.pdf-0006-06.png)


**----- Start of picture text -----**<br>
e5<br>0.5 BM25 0.75 random 0.75 random 0.80 random<br>contriever BM25 BM25 BM25<br>0.4 0.70 contriever 0.70 contriever 0.75 contriever<br>0.65 e5 e5 e5<br>0.3 0.65 0.70<br>0.60<br>0.60 0.65<br>0.2 0.55<br>0.50 0.55 0.60<br>0.1<br>0.45 0.50 0.55<br>0.30.40.50.60.70.80.9 5 10 15 20 25 30 35 40 5 10 15 20 25 30 35 40 5 10 15 20 25 30 35 40<br>Recall # Passages # Passages # Passages<br>(a) Retrievers (b) Gemma2-9B-Chat (c) Mistral-12B-Instruct (d) Gemini-1.5-Pro<br>Precision RAG Accuracy RAG Accuracy RAG Accuracy<br>**----- End of picture text -----**<br>


Figure 3 | Evaluating the impact of hard negatives on long-context LLMs. (a) The retriever performance on NQ dataset: e5 > contriever > BM25. (b)(c)(d) For each query, a single golden passage (containing the correct answer) is combined with varying numbers of hard negative passages retrieved by different methods: e5, Contriever, BM25, and random sampling. The LLMs are then tasked with answering the query based on this context. This setup allows us to assess the robustness of LLMs to hard negatives and the influence of retriever characteristics on their overall impact. 

**Observations.** (1) Sensitivity to hard negatives: Across all LLMs, increasing the number of hard negative passages generally leads to a decline in RAG answer accuracy. (2) Retriever strength and hard negative difficulty: The strength of the retriever directly correlates with the difficulty of the retrieved hard negatives. LLMs struggle more with hard negatives from stronger retrievers ( _e.g._ , e5) compared to those from weaker retrievers ( _e.g._ , BM25) or random sampling. (3) Distinguishing random and hard negatives: While Gemini-1.5-Pro demonstrates robustness to random negatives, it remains susceptible to the influence of hard negatives. More results on other datasets and qualitative studies can be found in Appendix B.2 and D. 

**Insights.** Existing benchmarks for evaluating long-context LLMs, such as "needle-in-the-haystack" (Kamradt, 2023) and RULER (Hsieh et al., 2024a), predominantly utilize random negatives. Our findings demonstrate that such benchmarks may not adequately capture the challenges posed by 

6 

Long-Context LLMs Meet RAG: Overcoming Challenges for Long Inputs in RAG 

hard negatives, which are prevalent in real-world RAG applications. Their takeaways would have limitations. The need for new evaluation methodologies that incorporate hard negatives (specific to the employed retrievers) is highlighted, to provide a more comprehensive and realistic assessment of long-context LLM performance in RAG. 

## **4. Simple and effective training-free RAG improvement** 

Building upon the analyses in Section 3 on the detrimental impact of hard negatives on long-context LLMs in RAG, we focus on the training-free solution, _retrieval reordering_ . This method leverages the inherent "lost-in-the-middle" phenomenon observed in LLMs to mitigate the negative effects of hard negatives. As highlighted by Liu et al. (2024), LLMs exhibit a tendency to prioritize information presented at the beginning and end of an input sequence, while paying less attention to the middle. 

Exploiting this "lost-in-the-middle" behavior, we consider a simple and effective strategy: reordering the retrieved passages based on their relevance scores calculated by the retriever. Given a query _𝑞_ and a set of retrieved passages _𝑑_ 1, _𝑑_ 2, ..., _𝑑𝑘_ with decreasing relevance scores, the standard input sequence construction for an LLM with instruction _𝐼_ would be: [ _𝐼, 𝑑_ 1 _, 𝑑_ 2 _, ..., 𝑑𝑘_ −1 _, 𝑑𝑘, 𝑞_ ]. Retrieval reordering modifies this to prioritize passages with higher scores at the beginning and end: [ _𝐼, 𝑑_ 1 _, 𝑑_ 3 _, ..., 𝑑_ 4 _, 𝑑_ 2 _, 𝑞_ ] where the position of passage _𝑑𝑖_ is determined by 


![](raw/papers/2_LLM_NLP/LLM_paper_list/rag/iclr_2025_Long-Context_LLMs_Meet_RAG_Overcoming_Challenges_for_Long_Inputs_in_RAG_images/iclr_2025_Long-Context_LLMs_Meet_RAG_Overcoming_Challenges_for_Long_Inputs_in_RAG.pdf-0007-05.png)


This reordering strategy aims to guide the LLM’s attention towards the most relevant passages, thereby reducing the influence of hard negatives positioned in the middle of the sequence. The pseudo-code for retrieval reordering can be found in Appendix E. 


![](raw/papers/2_LLM_NLP/LLM_paper_list/rag/iclr_2025_Long-Context_LLMs_Meet_RAG_Overcoming_Challenges_for_Long_Inputs_in_RAG_images/iclr_2025_Long-Context_LLMs_Meet_RAG_Overcoming_Challenges_for_Long_Inputs_in_RAG.pdf-0007-07.png)


**----- Start of picture text -----**<br>
0.545 0.44 0.56 0.48<br>0.540 0.54<br>0.535 0.42 0.52 0.46<br>0.40 0.50 0.44<br>0.530 0.48<br>0.525 0.38 0.46 0.42<br>original order original order 0.44 original order 0.40 original order<br>0.520 reordering 0.36 reordering 0.42 reordering 0.38 reordering<br>5 10 15 20 25 30 35 40 5 10 15 20 25 30 35 40 0 25 50 75 100125150175200 0 25 50 75 100125150175200<br># Retrieved Passages # Retrieved Passages # Retrieved Passages # Retrieved Passages<br>(a) NQ: Gemma2+e5 (b) NQ: Gemma2+BM25 (c) NQ: Mistral+e5 (d) NQ: Mistral+BM25<br>0.550<br>0.54 0.39 0.525 original order 0.42<br>0.53 0.38 0.500 reordering 0.40<br>0.37 0.475 0.38<br>0.52 0.450<br>0.36 0.425 0.36<br>0.51<br>0.50 original order reordering 0.350.34 original order reordering 0.4000.375 0.340.32 original order reordering<br>0.350<br>5 10 15 20 25 30 35 40 5 10 15 20 25 30 35 40 0 25 50 75 100125150175200 0 25 50 75 100125150175200<br># Retrieved Passages # Retrieved Passages # Retrieved Passages # Retrieved Passages<br>(e) PQA: Gemma2+e5 (f) PQA: Gemma2+BM25 (g) PQA: Mistral+e5 (h) PQA: Mistral+BM25<br>RAG Accuracy RAG Accuracy RAG Accuracy RAG Accuracy<br>RAG Accuracy RAG Accuracy RAG Accuracy RAG Accuracy<br>**----- End of picture text -----**<br>


Figure 4 | Evaluating the effectiveness of retrieval reordering in various RAG configurations. Results demonstrate that reordering retrieved passages consistently enhances performance, particularly when the number of retrieved passages is large. (Retrievers: e5, BM25; LLMs: Gemma2-9b-Chat, Mistral-Nemo-12B-Instruct; Datasets: NQ, PopQA) 

**Retrieval reordering significantly improves RAG performance, particularly with larger numbers of retrieved passages.** To assess the effectiveness of retrieval reordering, we conduct experiments 

7 

Long-Context LLMs Meet RAG: Overcoming Challenges for Long Inputs in RAG 

with two retrievers (e5 and BM25), two long-context LLMs (Gemma-2-9B-Chat and Mistral-Nemo12B-Instruct), and two datasets (NQ and PopQA). As illustrated in Figure 4, retrieval reordering yields negligible improvements with smaller retrieval sets, but significantly and consistently outperforms the original ordering when the number of retrieved passages is large. This behavior is attributed to the interplay of two factors that become increasingly significant with larger retrieval sets: (1) the amplified "lost-in-the-middle" phenomenon, where LLMs prioritize information at the beginning and end of the input sequence, and (2) the increased prevalence of hard negatives, which can hinder accurate answer generation. By strategically placing passages, retrieval reordering mitigates these issues, highlighting the potential of _position engineering_ as a complementary technique to prompt engineering for optimizing long-context LLMs in RAG. 

## **5. Improving Robustness for RAG via Data-Augmented Fine-Tuning** 

## **5.1. Implicitly improving LLM robustness through fine-tuning** 

While the _retrieval reordering_ strategy presented in Section 4 mitigates the detrimental impact of hard negatives, it does not inherently enhance the LLM’s ability to handle such irrelevant information within the context. To address this, we conduct a systematic investigation into RAG-specific tuning as a means of improving long-context LLMs for RAG applications. 

Our tuning paradigm involves training LLM to generate the correct answer ( _𝑎_ ) given a comprehensive input comprising an instruction ( _𝐼_ ), a query ( _𝑞_ ), and a set of retrieved passages ( _𝑑_ 1 _, 𝑑_ 2 _, ..., 𝑑𝑘_ ): 


![](raw/papers/2_LLM_NLP/LLM_paper_list/rag/iclr_2025_Long-Context_LLMs_Meet_RAG_Overcoming_Challenges_for_Long_Inputs_in_RAG_images/iclr_2025_Long-Context_LLMs_Meet_RAG_Overcoming_Challenges_for_Long_Inputs_in_RAG.pdf-0008-06.png)


This approach aims to implicitly enhance the LLM’s robustness to hard negatives by exposing it to a diverse range of retrieved contexts during fine-tuning, thus enabling it to learn to effectively identify and utilize relevant information even in the presence of noise. 

To assess the generalization capabilities of RAG-tuned LLMs, we fine-tune Gemma-2-9B-Base, Mistral-Nemo-12B-Base and Gemini-1.0-Pro using a diverse dataset comprising NQ, WoW, Fever, and MMLU. We then evaluate on a range of unseen datasets, including TriviaQA, PopQA, HotpotQA, 2wikimultihopqa, Webquestions, Bamboogle, ASQA, T-REx, and zsRE. We compare the performance of the RAG-tuned model (RAG FT) with two types of baselines: (1) Chat model with retrieval augmentation: the Gemma-2-9B-Chat/Mistral-Nemo-12B-Instruct/Gemini-1.0-Pro w. RAG; (2) Direct SFT: the ones fine-tuned with standard supervised fine-tuning (SFT) on question-answer pairs without retrieved context (Direct FT w/o RAG). Further details regarding the datasets and experimental setup can be found in Appendix F and G. 

Figure 5 shows the three key observations: (1) Consistent improvement over baselines: RAG FT consistently outperforms the chat model w. RAG and the Direct FT model across all evaluated datasets. (2) Robustness to hard negatives: the curve of RAG FT is generally flatter than that of the chat model, which demonstrates that our finetuned LLM is more robust to the hard negatives as the number of retrieved passages increases. (3) Superiority over direct fine-tuning: In most cases, RAG FT demonstrates superior performance compared to Direct FT. This indicates that RAG FT not only enables the LLM to "memorize" knowledge during training but also equips it with the ability to effectively "extract" relevant information from retrieved context during inference. These findings highlight the effectiveness of RAG-specific tuning in enhancing the generalization capabilities of LLMs for knowledge-intensive tasks. Separate results on those three LLMs are shown in Appendix J, K and L. Qualitative studies can be found in Appendix I. 

8 

Long-Context LLMs Meet RAG: Overcoming Challenges for Long Inputs in RAG 


![](raw/papers/2_LLM_NLP/LLM_paper_list/rag/iclr_2025_Long-Context_LLMs_Meet_RAG_Overcoming_Challenges_for_Long_Inputs_in_RAG_images/iclr_2025_Long-Context_LLMs_Meet_RAG_Overcoming_Challenges_for_Long_Inputs_in_RAG.pdf-0009-01.png)


**----- Start of picture text -----**<br>
0.775 0.60 0.50<br>0.750 0.55 0.45<br>0.725 0.50<br>0.40<br>0.700 0.45<br>0.675 0.40 0.35<br>0.650 0.35 0.30<br>0.625 0.30<br>0.25<br>0.600 0.25<br>0 5 10 15 20 25 30 35 40 0 5 10 15 20 25 30 35 40 0 5 10 15 20 25 30 35 40<br># Retrieved Passages # Retrieved Passages # Retrieved Passages<br>(a) TriviaQA (b) PopQA (c) HotpotQA<br>0.425 0.40 0.7<br>0.400 0.35<br>0.6<br>0.375 0.30<br>0.350 0.25 0.5<br>0.325<br>0.20<br>0.300 0.4<br>0.15<br>0.275<br>0.250 0.10 0.3<br>0.225 0.05<br>0 5 10 15 20 25 30 35 40 0 5 10 15 20 25 30 35 40 0 5 10 15 20 25 30 35 40<br># Retrieved Passages # Retrieved Passages # Retrieved Passages<br>(d) 2wikimultihopqa (e) Bamboogle (f) ASQA<br>0.70<br>0.7<br>0.65 Gemma-2-9B-Chat<br>0.60 0.6 Gemma-2 RAG FT<br>0.55 0.5 Gemma-2 Direct FT<br>0.50 0.4 Mistral-Nemo-12B-Chat<br>Mistral-Nemo RAG FT<br>0.45<br>0.3 Mistral-Nemo Direct FT<br>0.40<br>Gemini-1.0-Pro<br>0.35 0.2 Gemini RAG FT<br>0.30 0.1 Gemini Direct FT<br>0 5 10 15 20 25 30 35 40 0 5 10 15 20 25 30 35 40<br># Retrieved Passages # Retrieved Passages<br>(g) T-REx (h) zsRE (i) Legend<br>RAG Accuracy RAG Accuracy RAG Accuracy<br>RAG Accuracy RAG Accuracy RAG Accuracy<br>RAG Accuracy RAG Accuracy<br>**----- End of picture text -----**<br>


Figure 5 | Generalization ability of LLMs fine-tuned with RAG-specific data (RAG FT). RAG FT consistently outperforms the chat LLM w. RAG and the model fine-tuned directly on question-answer pairs (Direct FT). This demonstrates the effectiveness of RAG FT in enabling the LLM to effectively extract knowledge from retrieved context on unseen tasks. Note that Direct FT is evaluated without retrieval to align with its training paradigm and all others are evaluated with retrieval augmentation. (LLMs: Gemma-2-9B-Base, Mistral-Nemo-12B-Base, Gemini-1.0-Pro) 

## **5.2. Enhancing relevance identification through reasoning augmentation** 

While the fine-tuning approach described in Section 5.1 implicitly enhances the LLM’s robustness to hard negatives, it does not explicitly train the model to differentiate between relevant and irrelevant passages within the retrieved context. To address this, we investigate the effectiveness of incorporating an intermediate reasoning step into the fine-tuning process. 

This modified paradigm involves training the LLM to generate both a reasoning paragraph ( _𝑟_ ) that explicitly identifies the relevant passages for the given query ( _𝑞_ ) and the final answer ( _𝑎_ ): 


![](raw/papers/2_LLM_NLP/LLM_paper_list/rag/iclr_2025_Long-Context_LLMs_Meet_RAG_Overcoming_Challenges_for_Long_Inputs_in_RAG_images/iclr_2025_Long-Context_LLMs_Meet_RAG_Overcoming_Challenges_for_Long_Inputs_in_RAG.pdf-0009-06.png)


During training, the LLMs are provided with labeled reasoning paragraphs to guide its learning process. During inference, the LLMs are instructed to first generate the reasoning paragraph and then 

9 

Long-Context LLMs Meet RAG: Overcoming Challenges for Long Inputs in RAG 


![](raw/papers/2_LLM_NLP/LLM_paper_list/rag/iclr_2025_Long-Context_LLMs_Meet_RAG_Overcoming_Challenges_for_Long_Inputs_in_RAG_images/iclr_2025_Long-Context_LLMs_Meet_RAG_Overcoming_Challenges_for_Long_Inputs_in_RAG.pdf-0010-01.png)


**----- Start of picture text -----**<br>
0.55<br>0.80<br>0.6 0.50<br>0.75<br>0.45<br>0.5<br>0.70 0.40<br>0.65 0.4 0.35<br>0.60 0.3 0.30<br>0.55 0.25<br>0.2<br>0.20<br>0.50<br>0 5 10 15 20 25 30 35 40 0 5 10 15 20 25 30 35 40 0 5 10 15 20 25 30 35 40<br># Retrieved Passages # Retrieved Passages # Retrieved Passages<br>(a) TriviaQA (b) PopQA (c) HotpotQA<br>0.60 0.8<br>0.55<br>0.7 Gemma-2-9B-Chat<br>0.50 Gemma-2 RAG FT<br>0.6<br>0.45 Gemma-2 RAG FT w. Int<br>0.5 Gemma-2 Direct FT<br>0.40<br>Gemini-1.0-Pro<br>0.35 0.4 Gemini RAG FT<br>0.30 0.3 Gemini RAG FT w. Int<br>0.25 Gemini Direct FT<br>0 5 10 15 20 25 30 35 40 0 5 10 15 20 25 30 35 40<br># Retrieved Passages # Retrieved Passages<br>(d) 2wikimultihopqa (e) ASQA (f) Legend<br>RAG Accuracy RAG Accuracy RAG Accuracy<br>RAG Accuracy RAG Accuracy<br>**----- End of picture text -----**<br>


Figure 6 | Evaluating the impact of intermediate reasoning on the performance of RAG-tuned LLMs. Results demonstrate that fine-tuning with an intermediate reasoning step (RAG FT w. Int) leads to further improvements compared to implicit RAG fine-tuning (RAG FT) and direct fine-tuning (Direct FT). Direct FT is evaluated without retrieval to align with its training and all others are evaluated with retrieval augmentation. Due to the computational complexity of inference with reasoning augmentation, results are shown for 1000 randomly-sampled queries from each dataset. (LLMs: Gemma-2-9B-Base and Gemini-1.0-Pro, more results in Appendix J, K and L) 

utilize this analysis to produce the answer. This approach aims to explicitly enhance the LLMs’ ability to discern relevant information from noise within the retrieved context, thereby improving its overall performance in RAG. 

We utilize the same training data mixture as in Section 5.1 and augment it with reasoning labels generated by Gemini-1.5-Pro for each question-passage pair. These labels provide explicit guidance on identifying relevant passages. Further details of the experimental setup and the generation of reasoning labels can be found in Appendix H. 

Figure 6 demonstrates the effectiveness of this approach. The LLM fine-tuned with explicit intermediate reasoning consistently outperforms training with implicit RAG data. This improvement can be attributed to two key factors: (1) Explicit relevance training: Providing intermediate reasoning labels during training explicitly teaches the LLM to differentiate between relevant and irrelevant passages, enhancing its ability to discern crucial information from noise. (2) Structured reasoning for enhanced understanding: Generating a reasoning paragraph before answering introduces a structured approach to processing the retrieved context. This step, akin to chain-of-thought reasoning (Wei et al., 2022), helps decouple the complex information and facilitates a more focused analysis, ultimately leading to improved performance. These highlight the value of incorporating explicit reasoning mechanisms in RAG tuning to enhance the LLM’s ability to effectively utilize retrieved context. More results on Gemma-2-9B models, Mistral-Nemo-12B models and Gemini-1.0-Pro models are shown in Appendix J, K and L. Qualitative studies can be found in Appendix I. 

10 

Long-Context LLMs Meet RAG: Overcoming Challenges for Long Inputs in RAG 


![](raw/papers/2_LLM_NLP/LLM_paper_list/rag/iclr_2025_Long-Context_LLMs_Meet_RAG_Overcoming_Challenges_for_Long_Inputs_in_RAG_images/iclr_2025_Long-Context_LLMs_Meet_RAG_Overcoming_Challenges_for_Long_Inputs_in_RAG.pdf-0011-01.png)


**----- Start of picture text -----**<br>
0.50 0.8 0.61<br>No FT<br>0.45 0.7 FT w. BM25 0.60<br>0.40 FT w. e5 0.59<br>FT w. mix<br>0.35 Mixed data 0.6 25% max<br>NQ only 0.58 50% max<br>0.30<br>Wow only 0.5 0.57 100% max<br>0.25 Fever only 0-100% max<br>0.20 Mmlu only 0.4 0.56 50-100% max<br>5 10 15 20 25 30 35 40 BM25 contriever bge e5 Avg 5 10 15 20 25 30 35 40<br># Retrieved Passages Inference retriever # Retrieved Passages<br>RAG Accuracy RAG Accuracy RAG Accuracy<br>**----- End of picture text -----**<br>


(a) Analysis of training data distribu(b) Influence of retriever variations on (c) Investigation of the optimal numtion. (Test: HotpotQA) fine-tuning effectiveness. (NQ) ber of passages for training. 

Figure 7 | (a) Impact of training data distribution: A diverse mix of training data sources enhances the generalization ability of the LLM. (b) Influence of the retriever choice: Fine-tuning with data retrieved from multiple retrievers improves generalization to unseen retrievers during inference. (c) Effect of training context length: Fine-tuning with the maximum context length yields optimal performance across varying numbers of retrieved passages during inference. (LLM: Gemma-2-9B-Base) 

## **6. Data-Centric Perspectives on Fine-tuning LLMs for RAG** 

**Impact of training data distribution on generalization.** We first examine how the distribution of training data affects the generalization of the fine-tuned LLM. We train LLMs on five different data distributions, each with 50k samples: (1) a mixed dataset comprising NQ, WoW, Fever, and MMLU (12.5k samples from each); (2) NQ only; (3) WoW only; (4) Fever only; and (5) MMLU only. 

Figure 7(a) demonstrates that a mixed distribution of training data leads to superior generalization performance on unseen RAG tasks compared to training on a single data source. This highlights the importance of data diversity in enhancing the adaptability of LLMs to new RAG scenarios. 

**Influence of retrievers on generalization.** In real-world RAG deployments, LLMs might be paired with different retrievers depending on specific external knowledge corpus and retrievers’ capabilities. To investigate the impact of different retrievers on fine-tuning, we explore three adaptation scenarios on NQ: fine-tuning with (1) passages retrieved by BM25 (FT w. BM25); (2) passages retrieved by e5 (FT w. e5); and (3) mixture of passages retrieved by both BM25 and e5 (FT w. mix). We evaluate the performance of these fine-tuned LLMs using both retrievers seen during training (BM25 and e5) and unseen retrievers (Contriever (Izacard et al., 2021) and BGE (Chen et al., 2024)). 

Figure 7(b) presents the results, revealing two key findings: (1) Superiority of mixed retriever training: Fine-tuning with the data corresponding to a mix of retrievers consistently yields the best performance across both seen and unseen retrievers during inference. This suggests that training on a diverse set of retrieved passages enhances the LLMs’ ability to adapt to different retrieval strategies and knowledge sources. (2) Retriever similarity and generalization: The generalization ability of an LLM fine-tuned with a specific retriever is influenced by the similarity between the training retriever and the inference retriever. For instance, an LLM trained with BM25 generalizes better to Contriever, while an LLM trained with e5 generalizes better to BGE. This observation suggests that "hard negatives" exhibit different characteristics depending on the employed retriever, and training with a specific retriever implicitly equips the LLM to better handle similar types of hard negatives. See Appendix A for a detailed analysis of retriever similarity. 

**Optimizing training for variable retrieval sizes.** In real-world RAG systems, the number of retrieved passages can vary depending on the specific knowledge source and user requirements. Therefore, it is essential to determine the optimal training strategy for LLMs to ensure robust performance across 

11 

Long-Context LLMs Meet RAG: Overcoming Challenges for Long Inputs in RAG 

different retrieval sizes during inference. We investigate this aspect with the Gemma-2-9B-Base model, which has a maximum input sequence length of 8192 tokens (corresponding to approximately 40 passages). We evaluate five different training configurations: (1) Fixed 10 retrieved passages (25% max). (2) Fixed 20 retrieved passages (50% max). (3) Fixed 40 retrieved passages (maximum input capacity) (100% max). (4) Dynamic 0-40 retrieved passages (0-100% max). (5) Dynamic 20-40 retrieved passages (50-100% max). 

Figure 7(c) presents the results on NQ, demonstrating that fine-tuning with the maximum number of retrieved passages (100% max) consistently yields the best performance across various retrieval sizes during inference. This suggests that training with the full context capacity enhances the LLM’s ability to effectively handle varying amounts of retrieved information, leading to improved generalization and robustness. More analyses of RAG-specific tuning can be found in in Appendix M and N. 

## **7. Conclusions** 

This paper investigates the impact of increasing the number of retrieved passages on the performance of long-context LLMs in retrieval-augmented generation (RAG) systems. Contrary to expectations, we observe that performance initially improve but then degrade as more passages are included. This phenomenon is attributed to the detrimental influence of retrieved "hard negatives". To mitigate this issue, we propose and evaluate three solutions: training-free retrieval reordering, RAG-specific implicit LLM fine-tuning, and RAG-oriented LLM fine-tuning with intermediate reasoning. A systematic analysis of the training-based methods explores the effects of data distribution, retriever for training, and training context length. Interesting future directions include exploring (automated) position optimization with more advanced retrieval ordering methods, and fine-tuning the LLMs for RAG with more fine-grained and multi-step reasoning chains. 

## **References** 

- R. Agarwal, A. Singh, L. M. Zhang, B. Bohnet, S. Chan, A. Anand, Z. Abbas, A. Nova, J. D. Co-Reyes, E. Chu, et al. Many-shot in-context learning. _arXiv preprint arXiv:2404.11018_ , 2024. 

- A. Asai, Z. Wu, Y. Wang, A. Sil, and H. Hajishirzi. Self-rag: Learning to retrieve, generate, and critique through self-reflection. In _The Twelfth International Conference on Learning Representations_ , 2024. 

- I. Augenstein, T. Baldwin, M. Cha, T. Chakraborty, G. L. Ciampaglia, D. Corney, R. DiResta, E. Ferrara, S. Hale, A. Halevy, et al. Factuality challenges in the era of large language models. _arXiv preprint arXiv:2310.05189_ , 2023. 

- J. Chen, S. Xiao, P. Zhang, K. Luo, D. Lian, and Z. Liu. Bge m3-embedding: Multi-lingual, multifunctionality, multi-granularity text embeddings through self-knowledge distillation. _arXiv preprint arXiv:2402.03216_ , 2024. 

- F. Cuconasu, G. Trappolini, F. Siciliano, S. Filice, C. Campagnano, Y. Maarek, N. Tonellotto, and F. Silvestri. The power of noise: Redefining retrieval for rag systems. In _Proceedings of the 47th International ACM SIGIR Conference on Research and Development in Information Retrieval_ , pages 719–729, 2024. 

- Q. Dong, L. Li, D. Dai, C. Zheng, Z. Wu, B. Chang, X. Sun, J. Xu, and Z. Sui. A survey on in-context learning. _arXiv preprint arXiv:2301.00234_ , 2022. 

- A. Dubey, A. Jauhri, A. Pandey, A. Kadian, A. Al-Dahle, A. Letman, A. Mathur, A. Schelten, A. Yang, A. Fan, et al. The llama 3 herd of models. _arXiv preprint arXiv:2407.21783_ , 2024. 

12 

Long-Context LLMs Meet RAG: Overcoming Challenges for Long Inputs in RAG 

- Y. Gao, Y. Xiong, X. Gao, K. Jia, J. Pan, Y. Bi, Y. Dai, J. Sun, and H. Wang. Retrieval-augmented generation for large language models: A survey. _arXiv preprint arXiv:2312.10997_ , 2023. 

- C.-P. Hsieh, S. Sun, S. Kriman, S. Acharya, D. Rekesh, F. Jia, and B. Ginsburg. Ruler: What’s the real context size of your long-context language models? _arXiv preprint arXiv:2404.06654_ , 2024a. 

- C.-Y. Hsieh, Y.-S. Chuang, C.-L. Li, Z. Wang, L. T. Le, A. Kumar, J. Glass, A. Ratner, C.-Y. Lee, R. Krishna, et al. Found in the middle: Calibrating positional attention bias improves long context utilization. _arXiv preprint arXiv:2406.16008_ , 2024b. 

- L. Huang, W. Yu, W. Ma, W. Zhong, Z. Feng, H. Wang, Q. Chen, W. Peng, X. Feng, B. Qin, et al. A survey on hallucination in large language models: Principles, taxonomy, challenges, and open questions. _arXiv preprint arXiv:2311.05232_ , 2023. 

- G. Izacard, M. Caron, L. Hosseini, S. Riedel, P. Bojanowski, A. Joulin, and E. Grave. Unsupervised dense information retrieval with contrastive learning. _arXiv preprint arXiv:2112.09118_ , 2021. 

- A. Q. Jiang, A. Sablayrolles, A. Mensch, C. Bamford, D. S. Chaplot, D. d. l. Casas, F. Bressand, G. Lengyel, G. Lample, L. Saulnier, et al. Mistral 7b. _arXiv preprint arXiv:2310.06825_ , 2023. 

- G. Kamradt. Needle in a haystack - pressure testing llms, 2023. URL `https://github.com/ gkamradt/LLMTestNeedleInAHaystack/tree/main` . Accessed: 2024-09-10. 

- V. Karpukhin, B. Oğuz, S. Min, P. Lewis, L. Wu, S. Edunov, D. Chen, and W.-t. Yih. Dense passage retrieval for open-domain question answering. _arXiv preprint arXiv:2004.04906_ , 2020. 

- T. Kwiatkowski, J. Palomaki, O. Redfield, M. Collins, A. Parikh, C. Alberti, D. Epstein, I. Polosukhin, J. Devlin, K. Lee, et al. Natural questions: a benchmark for question answering research. _Transactions of the Association for Computational Linguistics_ , 7:453–466, 2019. 

- J. Lee, A. Chen, Z. Dai, D. Dua, D. S. Sachan, M. Boratko, Y. Luan, S. M. Arnold, V. Perot, S. Dalmia, et al. Can long-context language models subsume retrieval, rag, sql, and more? _arXiv preprint arXiv:2406.13121_ , 2024. 

- Z. Li, C. Li, M. Zhang, Q. Mei, and M. Bendersky. Retrieval augmented generation or long-context llms? a comprehensive study and hybrid approach. _arXiv preprint arXiv:2407.16833_ , 2024. 

- X. V. Lin, X. Chen, M. Chen, W. Shi, M. Lomeli, R. James, P. Rodriguez, J. Kahn, G. Szilvasy, M. Lewis, et al. Ra-dit: Retrieval-augmented dual instruction tuning. In _The Twelfth International Conference on Learning Representations_ , 2024. 

- N. F. Liu, K. Lin, J. Hewitt, A. Paranjape, M. Bevilacqua, F. Petroni, and P. Liang. Lost in the middle: How language models use long contexts. _Transactions of the Association for Computational Linguistics_ , 12:157–173, 2024. 

- M. Reid, N. Savinov, D. Teplyashin, D. Lepikhin, T. Lillicrap, J.-b. Alayrac, R. Soricut, A. Lazaridou, O. Firat, J. Schrittwieser, et al. Gemini 1.5: Unlocking multimodal understanding across millions of tokens of context. _arXiv preprint arXiv:2403.05530_ , 2024. 

- S. Robertson, H. Zaragoza, et al. The probabilistic relevance framework: Bm25 and beyond. _Foundations and Trends® in Information Retrieval_ , 3(4):333–389, 2009. 

- F. Shi, X. Chen, K. Misra, N. Scales, D. Dohan, E. H. Chi, N. Schärli, and D. Zhou. Large language models can be easily distracted by irrelevant context. In _International Conference on Machine Learning_ , pages 31210–31227. PMLR, 2023. 

13 

Long-Context LLMs Meet RAG: Overcoming Challenges for Long Inputs in RAG 

- G. Team, T. Mesnard, C. Hardin, R. Dadashi, S. Bhupatiraju, S. Pathak, L. Sifre, M. Rivière, M. S. Kale, J. Love, et al. Gemma: Open models based on gemini research and technology. _arXiv preprint arXiv:2403.08295_ , 2024a. 

- G. Team, M. Riviere, S. Pathak, P. G. Sessa, C. Hardin, S. Bhupatiraju, L. Hussenot, T. Mesnard, B. Shahriari, A. Ramé, et al. Gemma 2: Improving open language models at a practical size. _arXiv preprint arXiv:2408.00118_ , 2024b. 

- L. Wang, N. Yang, X. Huang, B. Jiao, L. Yang, D. Jiang, R. Majumder, and F. Wei. Text embeddings by weakly-supervised contrastive pre-training. _arXiv preprint arXiv:2212.03533_ , 2022. 

- X. Wang, M. Salmani, P. Omidi, X. Ren, M. Rezagholizadeh, and A. Eshaghi. Beyond the limits: A survey of techniques to extend the context length in large language models. _arXiv preprint arXiv:2402.02244_ , 2024. 

- J. Wei, X. Wang, D. Schuurmans, M. Bosma, F. Xia, E. Chi, Q. V. Le, D. Zhou, et al. Chain-of-thought prompting elicits reasoning in large language models. _Advances in neural information processing systems_ , 35:24824–24837, 2022. 

- Z. Wei, W.-L. Chen, and Y. Meng. Instructrag: Instructing retrieval-augmented generation with explicit denoising. _arXiv preprint arXiv:2406.13629_ , 2024. 

- P. Xu, W. Ping, X. Wu, L. McAfee, C. Zhu, Z. Liu, S. Subramanian, E. Bakhturina, M. Shoeybi, and B. Catanzaro. Retrieval meets long context large language models. _arXiv preprint arXiv:2310.03025_ , 2023. 

- O. Yoran, T. Wolfson, O. Ram, and J. Berant. Making retrieval-augmented language models robust to irrelevant context. In _The Twelfth International Conference on Learning Representations_ , 2024. 

- Y. Yu, W. Ping, Z. Liu, B. Wang, J. You, C. Zhang, M. Shoeybi, and B. Catanzaro. Rankrag: Unifying context ranking with retrieval-augmented generation in llms. _arXiv preprint arXiv:2407.02485_ , 2024. 

- S. Zhang, L. Dong, X. Li, S. Zhang, X. Sun, S. Wang, J. Li, R. Hu, T. Zhang, F. Wu, et al. Instruction tuning for large language models: A survey. _arXiv preprint arXiv:2308.10792_ , 2023. 

- T. Zhang, S. G. Patil, N. Jain, S. Shen, M. Zaharia, I. Stoica, and J. E. Gonzalez. Raft: Adapting language model to domain specific rag. _arXiv preprint arXiv:2403.10131_ , 2024. 

- W. X. Zhao, K. Zhou, J. Li, T. Tang, X. Wang, Y. Hou, Y. Min, B. Zhang, J. Zhang, Z. Dong, et al. A survey of large language models. _arXiv preprint arXiv:2303.18223_ , 2023. 

- W. X. Zhao, J. Liu, R. Ren, and J.-R. Wen. Dense text retrieval based on pretrained language models: A survey. _ACM Transactions on Information Systems_ , 42(4):1–60, 2024. 

- Z. Zhou, X. Ning, K. Hong, T. Fu, J. Xu, S. Li, Y. Lou, L. Wang, Z. Yuan, X. Li, et al. A survey on efficient inference for large language models. _arXiv preprint arXiv:2404.14294_ , 2024. 

- F. Zhu, W. Lei, C. Wang, J. Zheng, S. Poria, and T.-S. Chua. Retrieving and reading: A comprehensive survey on open-domain question answering. _arXiv preprint arXiv:2101.00774_ , 2021. 

14 

Long-Context LLMs Meet RAG: Overcoming Challenges for Long Inputs in RAG 

## **Appendix** 

## **A. Retriever performance and similarity** 

We analyze the performance and similarity of four retrievers (BM25, contriever, e5 and bge) on the NQ dataset shown in Figure 8. Each data point corresponds to a retrieval (recall, precision) pair for a specific number of retrieved passages. The overall retrieval performances on NQ are observed as e5 > bge > contriever > bm25, with contriever having a similar performance with BM25 and bge having a similar performance with e5 (as their curves are closer). 


![](raw/papers/2_LLM_NLP/LLM_paper_list/rag/iclr_2025_Long-Context_LLMs_Meet_RAG_Overcoming_Challenges_for_Long_Inputs_in_RAG_images/iclr_2025_Long-Context_LLMs_Meet_RAG_Overcoming_Challenges_for_Long_Inputs_in_RAG.pdf-0015-04.png)


**----- Start of picture text -----**<br>
BM25<br>contriever<br>0.5 e5<br>bge<br>0.4<br>0.3<br>0.2<br>0.1<br>0.3 0.4 0.5 0.6 0.7 0.8 0.9<br>Recall<br>Precision<br>**----- End of picture text -----**<br>


Figure 8 | Retriever performance on NQ. (1) Retrieval performance: e5 > bge > contriever > BM25; (2) Contriever is more similar to BM25, while bge is more similar to e5 (since their curves are closer respectively). 

## **B. Long context LLMs in RAG analysis on other datasets** 

In addition to the analysis presented on the NQ dataset in Section 3, we conduct further studies on the PopQA dataset to underscore the generality of our findings. 

## **B.1. The Effect of retrieved context size on RAG performance** 

**Observations.** Figure 9 presents the following key observations similar to that in Section 3.1: 1) Strong Retriever (e5): Across all LLMs, increasing the number of retrieved passages initially enhances performance, but subsequently results in either a sharp decline or a plateau. 2) Weak Retriever (BM25): Performance generally shows a continuous improvement or a slighter decrease as the number of retrieved passages increases. While these observations may appear counter-intuitive - given that one might expect monotonic improvements due to higher recall ( _i.e._ , a greater chance of retrieving relevant information) - the inclusion of additional documents can reduce precision, with irrelevant or misleading passages detracting LLMs from overall performance. 

## **B.2. The importance of hard negatives for long-context LLM evaluation** 

**Observations.** Figure 10 shows the following observations similar to that in Section 3.3: (1) Sensitivity to hard negatives: Across all LLMs, increasing the number of hard negative passages generally results 

15 

Long-Context LLMs Meet RAG: Overcoming Challenges for Long Inputs in RAG 


![](raw/papers/2_LLM_NLP/LLM_paper_list/rag/iclr_2025_Long-Context_LLMs_Meet_RAG_Overcoming_Challenges_for_Long_Inputs_in_RAG_images/iclr_2025_Long-Context_LLMs_Meet_RAG_Overcoming_Challenges_for_Long_Inputs_in_RAG.pdf-0016-01.png)


**----- Start of picture text -----**<br>
0.7 0.7<br>Gemma-7B-Chat<br>Gemma-2-9B-Chat<br>0.6 0.6 Mistral-Nemo-12B-Instruct<br>Gemini-1.5-Pro<br>0.5 0.5<br>0.4 0.4<br>0.3 Gemma-7B-Chat 0.3<br>Gemma-2-9B-Chat<br>Mistral-Nemo-12B-Instruct<br>0.2 0.2<br>Gemini-1.5-Pro<br>10 [0] 10 [1] 10 [2] 10 [3] 10 [0] 10 [1] 10 [2] 10 [3]<br># Retrieved Passages # Retrieved Passages<br>(a) RAG performance with e5 retriever (b) RAG performance with BM25 retriever<br>RAG Accuracy RAG Accuracy<br>**----- End of picture text -----**<br>


Figure 9 | Impact of retrieved context size on RAG performance (on PopQA) with 4 different LLMs. Increasing the number of retrieved passages initially improves performance but then leads to a decline. This degradation is more pronounced using a retriever (e5) that exhibits higher recall@k on PopQA compared to BM25 (Recall@40 is 0.85 with e5 and 0.57 with BM25). The maximum number of retrieved passages varies across LLMs due to differences in their maximum token limits. 


![](raw/papers/2_LLM_NLP/LLM_paper_list/rag/iclr_2025_Long-Context_LLMs_Meet_RAG_Overcoming_Challenges_for_Long_Inputs_in_RAG_images/iclr_2025_Long-Context_LLMs_Meet_RAG_Overcoming_Challenges_for_Long_Inputs_in_RAG.pdf-0016-03.png)


**----- Start of picture text -----**<br>
0.35<br>0.775 0.800<br>random 0.82<br>0.30 0.750 BM25 0.775 0.80<br>e5 0.725 contriever 0.750 random 0.78 random<br>0.25 BM25 0.700 e5 0.725 BM25 BM25<br>0.76<br>contriever 0.675 0.700 contriever contriever<br>0.20 0.650 0.675 e5 0.74 e5<br>0.72<br>0.625 0.650<br>0.15 0.70<br>0.600 0.625<br>0.4 0.5 0.6 0.7 0.8 5 10 15 20 25 30 35 40 5 10 15 20 25 30 35 40 5 10 15 20 25 30 35 40<br>Recall # Retrieved Passages # Retrieved Passages # Retrieved Passages<br>(a) Retrievers (b) Gemma2-9B-Chat (c) Mistral-12B-Instruct (d) Gemini-1.5-Pro<br>Precision RAG Accuracy RAG Accuracy RAG Accuracy<br>**----- End of picture text -----**<br>


Figure 10 | Evaluating the impact of hard negatives on long-context LLMs. (a) The retriever performance on PopQA dataset: e5 > contriever > BM25. (b)(c)(d) For each query, a single golden passage (containing the correct answer) is combined with varying numbers of hard negative passages retrieved by different methods (e5, Contriever, BM25, and random sampling). The LLMs are then tasked with answering the query based on this context. This setup allows us to assess the robustness of LLMs to hard negatives and the influence of retriever strength on their impact. 

in a decline in RAG answer accuracy. (2) Retriever strength and hard negative difficulty: The strength of the retriever is directly correlated with the difficulty of the retrieved hard negatives. LLMs struggle more with hard negatives generated by stronger retrievers ( _e.g._ , e5) compared to those produced by weaker retrievers ( _e.g._ , BM25) or through random sampling. (3) Distinguishing random and hard negatives: While all the LLMs demonstrates robustness to random negatives, it remains susceptible to the influence of hard negatives. 

16 

Long-Context LLMs Meet RAG: Overcoming Challenges for Long Inputs in RAG 

## **C. Illustration of Section 3.3: Hard negative study** 

**Algorithm 1** Data Construction for Hard Negative Study 

**Require:** Query _𝑞_ , instruction _𝐼_ , golden passage _𝑑_ gold, golden answer _𝑎_ , retrieved passages _𝐷_ = [ _𝑑_ 1 _, 𝑑_ 2 _, . . . , 𝑑𝑁_ ] with decreasing retriever relevance scores, desired number of passages _𝐾_ ( _𝐾_ ≪ _𝑁_ ). **Ensure:** Input sequence _𝑆_ . 

- 1: Initialize list _𝑆_ ←[ _𝑑_ gold] 

- 2: **for** each passage _𝑑𝑖_ in _𝐷_ **do** 3: **if** _𝑑𝑖_ ≠ _𝑑_ gold **and** _𝑎_ not in _𝑑𝑖_ **then** 4: Append _𝑑𝑖_ to _𝑆_ 5: **end if** 6: **if** | _𝑆_ | = _𝐾_ **then** 7: **break** 8: **end if** 

- 9: **end for** 

- 10: Randomly shuffle _𝑆_ . 

- 11: Construct input sequence [ _𝐼, 𝑆_ [1] _, 𝑆_ [2] _, . . . , 𝑆_ [ _𝐾_ ] _, 𝑞_ ]. 

- 12: **return** The input sequence [ _𝐼, 𝑆_ [1] _, 𝑆_ [2] _, . . . , 𝑆_ [ _𝐾_ ] _, 𝑞_ ]. 

17 

Long-Context LLMs Meet RAG: Overcoming Challenges for Long Inputs in RAG 

## **D. Hard negatives case study** 

In this section, we provide a case study to compare the hard negatives returned by different retrievers. We classify the negative passages into two types: (1) _Related but Irrelevant_ : passages related to some entities mentioned in the question but not containing the ground truth answer; (2) _Not Related_ : passages not related to the question at all. Note that _Related but Irrelevant_ passages are harder and more misleading to the LLMs compared with _Not Related_ passages. We show the top-5 negatives from each retriever for a random sampled question as below. From the case study, we can find that the negatives retrieved by e5 contain more _Related but Irrelevant_ passages compared with those retrieved by bm25, while those retrieved by bm25 still have more _Related but Irrelevant_ passages than random sampling. This qualitatively demonstrates that the hardness of the negatives from different retrievers as e5 > bm25 > random. 

|**Question**|The south west wind blows across Nigeria between?|
|---|---|
|**Ground**|Till September|
|**Truth**||
|**Retrieved Hard Negative Passages**(high retrieval score but lacking ground truth answer)||
|**w. e5**|**Doc 1**(Title: "Geography of Nigeria") ... south atlantic ocean, locally known as the south|
||western wind, or by its main name, The Tropical Maritime (MT) airmass. These two major|
||wind systems in Nigeria are known as the trade winds. The tropical maritime airmass (MT) is|
||responsible for Nigeria’s rainy season. This wind (the tropical maritime airmass) invades the|
||country from February in the southern part of Nigeria while it takes longer for the wind to|
||fully cover the whole of the country, reaching the northern part of Nigeria in June. Its|
||invasion is as a result of the northward retreat, ... _[Related but Irrelevant]_|
||**Doc 2**(Title: "Onikwu") ... The dry season is accompanied by a dust laden airmass from the|
||Sahara Desert, locally known as Harmattan, or by its main name, The Tropical Continental|
||(CT) airmass, while the rainy season is heavily infuenced by an airmass originating from the|
||south Atlantic Ocean, locally known as the south west wind, or by its main name, The|
||Tropical Maritime (MT) airmass. These two major wind systems in Nigeria are known as the|
||trade winds. The region Onikwu/Ndoni is food prone communities, this is because the|
||inland part of Rivers state consists of tropical rainforest ... _[Related but Irrelevant]_|
||**Doc 3**(Title: "Geography of Nigeria") ... northern end is south of the 15 degrees line at about|
||14 degrees. Nigeria’s location in the wetter part of the easterly waves south of the 15 degree|
||line creates wetter climatic conditions for Nigeria especially during the monsoons. The|
||Tropical Continental Airmass (CT) locally known as the harmattan, is a wind originating from|
||North Africa which crosses the Sahara Desert into west Africa to Nigeria. This airmass|
||dominates Nigeria’s climate during the dry season from December to March. The Tropical|
||continental airmass is dusty and creates a haze within the atmosphere of west Africa and|
||Nigeria when it predominates. ... _[Related but Irrelevant]_|
||**Doc 4**(Title: "Nigeria") ... Niger, Chad, Cameroon, and has a coastline of at least s. Nigeria|
||lies between latitudes 4 and 14N, and longitudes 2 and 15E. The highest point in Nigeria is|
||Chappal Waddi at . The main rivers are the Niger and the Benue, which converge and empty|
||into the Niger Delta. This is one of the world’s largest river deltas, and the location of a large|
||area of Central African mangroves. Nigeria has a varied landscape. The far south is defned|
||by its tropical rainforest climate, where annual rainfall is a year. In the southeast stands the ...|
||_[Related but Irrelevant]_|



18 

Long-Context LLMs Meet RAG: Overcoming Challenges for Long Inputs in RAG 

||**Doc 5**(Title: "Geography of Nigeria") ... Nigeria, has a temperature range of to , and an|
|---|---|
||annual rainfall of about with a single rainfall maxima in September. The single Dry season|
||experienced in this climate, the tropical savanna climate in central Nigeria beginning from|
||December to march, is hot and dry with the Harmattan wind, a continental tropical (CT)|
||airmass laden with dust from the Sahara Desert prevailing throughout this period. With the|
||Intertropical Convergence Zone (ITCZ) swinging northward over West Africa from the|
||Southern Hemisphere in April, heavy showers coming from pre-monsoonal convective clouds|
||mainly in the form of ... _[Related but Irrelevant]_|
|**w. bm25**|**Doc 1**(Title: "Oron people") ... Civil War. Oron is found in the food plain of South Eastern|
||Nigeria, with the land mainly intersected by numerous streams and tributaries fowing into|
||Cross River. The entire coastline stretches from Uya Oron to Udung Uko. Oron is in the|
||tropical region and has a uniformly high temperature all the year round. The two main|
||seasons are the dry which spans between October and April and wet season which starts|
||around May and ends in September. There are also two prevailing winds – the South-West|
||onshore winds which brings heavy rains and the ... _[Not Related]_|
||**Doc 2**(Title: "South Equatorial Current") ... is driven directly by the trade winds which blow|
||from east to west. In the Indian Ocean, the westward-fowing South Equatorial Current is|
||well-developed only south of the equator. Directly on the equator, the winds reverse twice a|
||year due to the monsoons, and so the surface current can be either eastward or westward.|
||South Equatorial Current Ocean current in the Pacifc, Atlantic, and Indian Ocean that fows|
||east-to-west between the equator and about 20 degrees south. In the Pacifc and Atlantic|
||Oceans, it extends across the equator to about 5 degrees north. Within the southern|
||hemisphere, the South Equatorial ... _[Related but Irrelevant]_|
||**Doc 3**(Title: "Wind direction") ... Wind direction Wind direction is reported by the direction|
||from which it originates. For example, a ""northerly"" wind blows from the north to the south.|
||Wind direction is usually reported in cardinal directions or in azimuth degrees. Wind|
||direction is measured in degrees clockwise from due north. Consequently, a wind blowing|
||from the north has a wind direction of 0; a wind blowing from the east has a wind direction|
||of 90; a wind blowing from the south has a wind direction of 180; and a wind blowing from|
||the west has a wind direction of 270 ... _[Related but Irrelevant]_|
||**Doc 4**(Title: "Gulf Stream") ... this current interacts with the northeastern coast of South|
||America, the current forks into two branches. One passes into the Caribbean Sea, while a|
||second, the Antilles Current, fows north and east of the West Indies. These two branches|
||rejoin north of the Straits of Florida. The trade winds blow westward in the tropics, and the|
||westerlies blow eastward at mid-latitudes. This wind pattern applies a stress to the|
||subtropical ocean surface with negative curl across the north Atlantic Ocean. The resulting|
||Sverdrup transport is equatorward. Because of conservation of potential vorticity caused by|
||the northward-moving winds on the subtropical ... _[Not Related]_|
||**Doc 5**(Title: "Climate of the United Kingdom") ... climate that western parts of the UK|
||experience. The high latitude and proximity to a large ocean to the west means that the|
||United Kingdom experiences strong winds. The prevailing wind is from the south-west, but it|
||may blow from any direction for sustained periods of time. Winds are strongest near westerly|
||facing coasts and exposed headlands. Gales — which are defned as winds with speeds of —|
||are strongly associated with the passage of deep depressions across the country. The Hebrides|
||experience on average 35 days of gale a year (a day where there are gale-force winds) while|
||inland ... _[Not Related]_|
|**w.**|**Doc 1**(Title: "Queen of Peace, Bray") ... of Bugisi in Tanzania for a number of years. The|
|**random**|parish has a very close relationship with St Cronan’s B.N.S., Scoil Chualann and Gaelscoil Uí|
|**sampling**|Chéadaigh. The parish provides the sacraments of Communion and Confrmation to the|
||children in the schools. They also help to raise funds for the twin parish of Bugisi. Queen of|
||Peace, Bray The Queen of Peace is a Catholic church situated at the junction of the Putland|
||Road and the Vevay Road in Bray, Co. Wicklow, Ireland. The present church was built in 1946|
||by TJ Macken of St Patrick’s Street, Dún Laoghaire, ... _[Not Related]_|



19 

Long-Context LLMs Meet RAG: Overcoming Challenges for Long Inputs in RAG 

**Doc 2** (Title: "Cordova Congressional Internship Program") ... Puerto Rico’s Constitutional Convention from 1951 to 1952. By 2012, over 670 students from colleges and universities in Puerto Rico had enjoyed internships under the program, and the Spring 2009 class included a record 24 members. A private sector committee, recently headed by Univision Puerto Rico president Larry Sands, provides private funds to supplement the 350,000 annual grant provided by the Puerto Rico Legislative Assembly. Under the auspices of TWC, seventeen states have since established similar legislative-funded Congressional internship programs. The Center established in 2008 the McClintock Award to the State Legislator of the Year ... _[Not Related]_ 

**Doc 3** (Title: "V bomber") ... Puerto Rico’s Constitutional Convention from 1951 to 1952. By 2012, over 670 students from colleges and universities in Puerto Rico had enjoyed internships under the program, and the Spring 2009 class included a record 24 members. A private sector committee, recently headed by Univision Puerto Rico president Larry Sands, provides private funds to supplement the 350,000 annual grant provided by the Puerto Rico Legislative Assembly. Under the auspices of TWC, seventeen states have since established similar legislative-funded Congressional internship programs. The Center established in 2008 the McClintock Award to the ... _[Not Related]_ 

**Doc 4** (Title: "Defence Materials and Stores Research and Development Establishment") ... materials for the Indian Armed Forces. DMSRDE has developed Nuclear Shielding Pad, Boot Anti Mine, Blast Protection Suit, Bullet Proof Jackets, etc.. ""The Defence Material and Stores Research Development Establishment in Kanpur has developed a new NBC suit that would be proved effective against any kind of dangerous weapons or chemicals and protect soldiers from any sort of attack,"" DMSRDE Director Arvind Kumar Saxena was quoted by media-persons. 40,000 pieces of NBC suits costing about Rs 30,000 had been requested by Indian army. ""the further progress on the other two suits are going on."" further ... _[Not Related]_ 

**Doc 5** (Title: "Chess title") ... retain the title of Candidate Master, if it was earned according to criteria above). This is in contrast to international titles awarded by FIDE, which are awarded for life. In European countries the term of ""expert"" is not used. Instead, players of that level are called ""Candidate Masters"", although the FIDE Candidate Master title generally requires a higher rating (2200 FIDE). It is possible (and common), however, for players in the United States to have a rating that places them in the ’expert’ category while still retaining the title of ’Life Master’ or ’National Master’ ... _[Not Related]_ 

20 

Long-Context LLMs Meet RAG: Overcoming Challenges for Long Inputs in RAG 

## **E. Retrieval reordering** 

**Algorithm 2** Retrieval Reordering Algorithm 

**Require:** Query _𝑞_ , instruction _𝐼_ , retrieved passages _𝐷_ = [ _𝑑_ 1 _, 𝑑_ 2 _, ..., 𝑑𝑘_ ] with decreasing retriever relevance scores. 

**Ensure:** Reordered sequence _𝑆_ . 

- 1: Initialize an empty list _𝑆_ of length _𝑘_ . 

- 2: **for** _𝑖_ = 1 to _𝑘_ **do** 

3: **if** mod( _𝑖_ , 2) = 1 **then** 4: _Order_ ( _𝑑𝑖_ ) ← _[𝑖]_[+][ 1] { _𝑖_ is odd} 2 5: **else** 6: _Order_ ( _𝑑𝑖_ ) ← _𝑘_ + 1 − _[𝑖]_ 2[{] _[𝑖]_[is even}] 7: **end if** 8: Place _𝑑𝑖_ at position _Order_ ( _𝑑𝑖_ ) in _𝑆_ . 

- 9: **end for** 

- 10: Construct input sequence [ _𝐼, 𝑆_ [1] _, 𝑆_ [2] _, ..., 𝑆_ [ _𝑘_ ] _, 𝑞_ ]. 

- 11: **return** The reordered sequence [ _𝐼, 𝑆_ [1] _, 𝑆_ [2] _, ..., 𝑆_ [ _𝑘_ ] _, 𝑞_ ]. 

## **F. Datasets** 

In this section, we discuss the datasets for RAG-specific LLM training and evaluation. 

## **F.1. Training datasets** 

|Dataset<br>the number of instances|Dataset<br>the number of instances|
|---|---|
|||
|Natural Question<br>Wizard of Wikipedia<br>FEVER<br>MMLU|12,500<br>12,500<br>12,500<br>12,500|



Table 1 | Training data statistics. 

We select a series of fine-tuning data designed to enhance the model’s robustness to hard negatives in the retrieval context and improve its contextual awareness in generating predictions. The training data are from four sources with different answer types: Natural Question (short-form), Wizard of Wikipedia (long-form), FEVER (true/false), and MMLU (close-set). The statistics of the training data mix can be found in Table 1. 

## **F.2. Testing datasets** 

To comprehensively evaluate our methods, we select testing datasets across different tasks including: (1) Question-answering: TriviaQA, PopQA, WebQuestions; (2) Multi-hop tasks: HotpotQA, 2WikiMultiHopQA, Bamboogle; (3) Long-form tasks: ASQA; (4) Slot filling: T-REx, Zero-shot RE. The statistics of all the datasets can be found in Table 2. 

21 

Long-Context LLMs Meet RAG: Overcoming Challenges for Long Inputs in RAG 

|Dataset<br>Task<br>the number of instances|Dataset<br>Task<br>the number of instances|Dataset<br>Task<br>the number of instances|
|---|---|---|
|TriviaQA<br>PopQA<br>WebQuestions<br>HotpotQA<br>2WikiMultiHopQA<br>Bamboogle<br>ASQA<br>T-REx<br>Zero-shot RE|QA<br>QA<br>QA<br>Multi-Hop QA<br>Multi-Hop QA<br>Multi-Hop QA<br>Long-form QA<br>Slot flling<br>Slot flling|11,313<br>14,267<br>2,032<br>7,405<br>12,576<br>125<br>948<br>5,000<br>3,724|



Table 2 | Testing data statistics. 

## **F.3. Retrieval corpus** 

Following Karpukhin et al. (2020), we use the text chunks from 2018 Wikipedia dump as the retrieval corpus. The articles are split by section, where long sections are further split into text chunks of equal sizes and contain less than 100 words, leading to a total of 21M text chunks. 

22 

Long-Context LLMs Meet RAG: Overcoming Challenges for Long Inputs in RAG 

## **G. Implicit RAG Fine-Tuning Experimental Setting** 

## **G.1. Training settings** 

**Hyperparameters.** We use the top-40 retrieved text chunks for a given example to generate the fine-tuning samples and use e5 as the retriever for the main results. We fine-tune both Gemma-2-9BBase and Mistral-Nemo-12B-Base using 8 H100 GPUs. For both models, we use the chat template corresponding to Gemma-2-9B-Chat and Mistral-Nemo-12B-Instruct respectively when tuning the models. We use the axolotl[1] codebase for their tuning. For Gemini-1.0-Pro tuning, we use the Google Cloud Tuning API[2] with the default settings. The hyperparameters can be found in Table 3. 

||Model||peak lr|lr scheduler<br>warm up<br># epoch<br>batch size|Flash Att|
|---|---|---|---|---|---|
||Gemma-2-9B-Base||1e-6|cosine<br>5%<br>4<br>64|False|
||Mistral-Nemo-12B-Base||1e-6|cosine<br>5%<br>4<br>64|True|
||Gemini-1.0-Pro||default|default<br>default<br>1<br>default|default|
|Table|3 | Implicit RAG|fnetuning hyperparameters.||||
|**Training RAG instruction templates.**||||The RAG instruction templates for diferent training datasets||
|can be found in Table||4.||||
|Task||||Instruction Templates||
|NQ|||Answer the question based on the given document.|||
||||Only|give me the answer and do not output any other words.||
|||||The following are given documents.||
|||||{reference}||
|Wizard of Wikipedia|||Provide a response to the conversation based on the given document.|||
|||||The following are given documents.||
|||||{reference}||
|FEVER||||Verify a fact based on the given documents||
|||||The following are given documents.||
|||||{reference}||
|MMLU||Given a question, choose the answer from the options based on the given documents.||||
|||||The following are given documents.||
|||||{reference}||



Table 4 | Training instruction templates for implicit RAG tuning. 

> 1 `https://github.com/axolotl-ai-cloud/axolotl` 

> 2 `https://cloud.google.com/vertex-ai/generative-ai/docs/model-reference/tuning` 

23 

Long-Context LLMs Meet RAG: Overcoming Challenges for Long Inputs in RAG 

**Training RAG answer templates.** The RAG answer templates for different training datasets can be found in Table 5. 

|Task|Answer Templates|
|---|---|
|NQ|Question: {question}|
||Answer:|
|Wizard of Wikipedia|Conversation: {question}|
||Response:|
|FEVER|Return SUPPORTS if it is correct and return REFUTES if it is not correct.|
||Fact: {question}|
||Response:|
|MMLU|Question: {question}|
||Options: {choices}|
||Answer:|



Table 5 | Training answer templates for implicit RAG tuning. 

## **G.2. Evaluation Settings** 

**Hyperparameters.** For all the compared LLMs, we conduct top-p sampling (p = 1) and the maximum number of generated token is set to be 32. For Gemma-2 series models, we use the huggingface inference pipeline[3] . For Gemini series models, we use Google Cloud Inference API[4] . While for other series of LLMs, we utilize vLLM[5] codebase for efficient generation. 

**Evaluation RAG instruction templates.** The RAG instruction templates for different testing datasets can be found in Table 6. 

|Task|Instruction Templates|
|---|---|
|QA|Answer the question based on the given document.|
||Only give me the answer and do not output any other words.|
||The following are given documents.|
||{reference}|
|Multi-hop|Answer the question based on the given document.|
||Only give me the answer and do not output any other words.|
||The following are given documents.|
||{reference}|
|Long-form|Answer the question based on the given document.|
||Please give in-depth explanation and avoid only returning the answer.|
||The following are given documents.|
||{reference}|
|Slot flling|Given a question, choose the answer from the options based on the given documents.|
||The following are given documents.|
||{reference}|



Table 6 | Testing instruction templates for implicit RAG tuning. 

**Evaluation RAG answer templates.** The RAG answer templates for different testing datasets are all: "Question: {question}. Answer:" 

> 3 `https://huggingface.co/docs/transformers/` 

> 4 `https://cloud.google.com/vertex-ai/generative-ai/docs/model-reference/inference` 

> 5 `https://github.com/vllm-project/vllm` 

24 

Long-Context LLMs Meet RAG: Overcoming Challenges for Long Inputs in RAG 

## **H. RAG Finetuning with Intermediate Reasoning Experimental Setting** 

## **H.1. Training settings** 

**Hyperparameters.** The hyperparameter setting is the same to that in Appendix G.1. 

**Training RAG instruction templates.** The RAG instruction templates with intermediate reasoning for different training datasets can be found in Table 7. 

|Task|Instruction Templates|
|---|---|
|NQ|Answer the question based on the given documents.|
||Please frst provide an analysis with clear reasoning details of which documents are relevant to answer the question.|
||Then output a concise answer to the question based on the analysis.|
||The following are given documents.|
||{reference}|
|Wizard of Wikipedia|Provide a response to the conversation based on the given documents.|
||Please frst provide an analysis with clear reasoning details of which documents are relevant to provide the response.|
||Then output a concise response to the question based on the analysis.|
||The following are given documents.|
||{reference}|
|FEVER|Verify a fact based on the given documents.|
||Please frst provide an analysis with clear reasoning details of which documents are relevant to verify the fact.|
||Then output a concise answer (SUPPORTS or REFUTES) based on the analysis.|
||The following are given documents.|
||{reference}|
|MMLU|Given a question, choose the answer from the options based on the given documents.|
||Please frst provide an analysis with clear reasoning details of which documents are relevant to answer the question|
||Then output a concise answer to the question based on the analysis.|
||The following are given documents.|
||{reference}|



Table 7 | Training instruction templates for RAG tuning with intermediate reasoning. 

**Training RAG Answer templates.** The RAG answer templates for different training datasets can be found in Table 8. 

|Task|Answer Templates|
|---|---|
|NQ|Question: {question}|
||Answer:|
|Wizard of Wikipedia|Conversation: {question}|
||Response:|
|FEVER|Return SUPPORTS if it is correct and return REFUTES if it is not correct.|
||Fact: {question}|
||Response:|
|MMLU|Question: {question}|
||Options: {choices}|
||Answer:|



Table 8 | Training answer templates for RAG tuning with intermediate reasoning. 

25 

Long-Context LLMs Meet RAG: Overcoming Challenges for Long Inputs in RAG 

**Instructions to generate intermediate reasoning from Gemini-1.5-pro.** The prompt that guides Gemini-1.5-pro for intermediate reasoning generation can be found in Table 9. 

|Task|Prompts|
|---|---|
|NQ|Read the following documents relevant to the given question: {question}|
||{reference}|
||Please identify documents that are useful to answer the given question: {question},|
||and explain how the contents lead to the answer: {answers}.|
||If none of the documents is aligned with the answer,|
||in that case, you have to explain the answer only based on your own knowledge,|
||without referring to the provided information.|
||Note that the question may be compositional and require intermediate analysis to deduce the fnal answer.|
||Make sure your response is grounded and provides clear reasoning details followed by a concise conclusion.|
|Wizard of Wikipedia|Read the following documents relevant to the given conversation: {question}|
||{reference}|
||Please identify documents that are useful to provide a response to a conversation: {question}|
||and explain how the contents lead to the response: {answers}.|
||If none of the documents is aligned with the response,|
||in that case, you have to explain the response only based on your own knowledge,|
||without referring to the provided information.|
||Make sure your response is grounded and provides clear reasoning details followed by a concise conclusion.|
|FEVER|Read the following documents relevant to the given question: {question}|
||{reference}|
||Please identify documents that are useful to verify a fact: {question}|
||(Return SUPPORTS if it is correct and return REFUTES if it is not correct.),|
||and explain how the contents lead to the answer: {answers}.|
||If none of the documents is aligned with the answer,|
||in that case, you have to explain the answer only based on your own knowledge|
||without referring to the provided information.|
||Make sure your response is grounded and provides clear reasoning details followed by a concise conclusion.|
|MMLU|Read the following documents relevant to the given question: {question}|
||{reference}|
||Please identify documents that are useful to answer the given question: {question} with options: {choices},|
||and explain how the contents lead to the answer: {answers}.|
||If none of the documents is aligned with the answer,|
||in that case, you have to explain the answer only based on your own knowledge,|
||without referring to the provided information.|
||Make sure your response is grounded and provides clear reasoning details followed by a concise conclusion.|



Table 9 | Prompts to guide Gemini-1.5-pro for intermediate reasoning generation. 

26 

Long-Context LLMs Meet RAG: Overcoming Challenges for Long Inputs in RAG 

## **H.2. Evaluation settings** 

**Hyperparameters.** For all the compared LLMs, we conduct top-p sampling (p = 1) and the maximum number of generated token is set to be 256. For Gemma-2 series models, we use the huggingface inference pipeline. While for other series of LLMs, we utilize vLLM codebase for efficient generation. 

**Evaluation RAG instruction templates.** The RAG instruction templates for different testing datasets can be found in Table 10. 

|Task|Instruction Templates|
|---|---|
|QA|Answer the question based on the given documents.|
||Please frst provide an analysis with clear reasoning details of which documents are relevant to answer the question|
||Then output a concise answer to the question based on the analysis.|
||The following are given documents.|
||{reference}|
|Multi-hop|Answer the question based on the given documents.|
||Please frst provide an analysis with clear reasoning details of which documents are relevant to answer the question|
||Then output a concise answer to the question based on the analysis.|
||The following are given documents.|
||{reference}|
|Long-form|Answer the question based on the given document.|
||Please frst provide an analysis with clear reasoning details of which documents are relevant to answer the question|
||Then provide an in-depth long-form answer for the question (avoid only returning the answer) based on the analysis.|
||The following are given documents.|
||{reference}|
|Slot flling|Provided an answer to the given slot flling question based on the given document.|
||In the question, the words before and after [SEP] correspond to the head entity and relation respectively.|
||You are asked to output the tail entity corresponded to the given head entity and relation.|
||Please frst provide an analysis with clear reasoning details of which documents are relevant to answer the question.|
||Then output a concise answer to the question based on the analysis.|
||The following are given documents.|
||{reference}|



Table 10 | Testing instruction templates for RAG tuning with intermediate reasoning. 

**Evaluation RAG answer templates.** The RAG answer templates for different testing datasets are all: "Question: {question}. Answer:" 

27 

Long-Context LLMs Meet RAG: Overcoming Challenges for Long Inputs in RAG 

## **I. Data-Augmented RAG Case Studies** 

|**Question**|Which flm features the Dawes Tomes Mousley Grubbs Fidelity Fiduciary Bank?|
|---|---|
|**Ground**|Mary Poppins|
|**Truth**||
|**Retrieved**|**Doc 1**(Title: "Fidelity Fiduciary Bank") Fidelity Fiduciary Bank ""Fidelity Fiduciary|
|**Passages**|Bank"" is a song from Walt Disney’s flm ""Mary Poppins"", and it is composed by|
||Richard M. Sherman and Robert B. Sherman. The song sung by the stodgy old bankers|
||at the ""Dawes, Tomes, Mousely, Grubbs Fidelity Fiduciary Bank"", led by the ""Elder|
||Mr. Dawes"" (Nackvid Keyed), to George Banks’s two children, Jane and Michael, in an|
||attempt to get Michael Banks to invest his tuppence in the bank. As the song continues|
||the pressure is on George, a junior clerk at the bank, to sway Michael. When Michael|
||fnally, and ever so slightly, opens ...|
||**Doc 2**(Title: "Fidelity Fiduciary Bank") convince Michael to invest his money goes like|
||this: has as its origins an essay by C. C. Turner titled ’Money London’ in the book|
||edited by G. R. Sims called ""Living London"" (London: 1903): Fidelity Fiduciary Bank|
||""Fidelity Fiduciary Bank"" is a song from Walt Disney’s flm ""Mary Poppins"", and it is|
||composed by Richard M. Sherman and Robert B. Sherman. The song sung by the|
||stodgy old bankers at the ""Dawes, Tomes, Mousely, Grubbs Fidelity Fiduciary Bank"",|
||led by the ""Elder Mr. Dawes"" (Nackvid Keyed), to George Banks’s two children, Jane|
||and Michael, in an attempt to get ...|
||...|
||**Doc 40**(Title: "The Firm (soundtrack)") The Firm (soundtrack) The Firm is an album|
||by American pianist Dave Grusin released in 1993, recorded for the GRP label. This|
||album is a soundtrack to the Motion Picture ""The Firm"" directed by Sidney Pollack. It|
||reached No. 131 on ""Billboard""’s Contemporary Jazz chart. This soundtrack was a|
||1993 Academy Award nominee for Best Original Score. All tracks composed by Dave|
||Grusin; except where indicated In 2015 La-La Land Records issued a remastered and|
||expanded edition, featuring the flm score on disc one and the 1993 soundtrack album|
||and bonus tracks on disc two. Italicised tracks on disc one ...|



||reached No. 131 on ""Billboard""’s Contemporary Jazz chart. This soundtrack was a<br>1993 Academy Award nominee for Best Original Score. All tracks composed by Dave<br>Grusin; except where indicated In 2015 La-La Land Records issued a remastered and<br>expanded edition, featuring the flm score on disc one and the 1993 soundtrack album<br>and bonus tracks on disc two. Italicised tracks on disc one ...|
|---|---|
||**Model Predictions**|
|**Gemma-**|Fidelity Fiduciary Bank|
|**2-9B-**||
|**Chat**||
|**Implicit**|_Mary Poppins_(1964 flm)|
|**RAG FT**||
|**RAG FT**|Documents 1 and 2 are useful to answer the question.|
|**with Rea-**||
|**soning**|Both documents explicitly state that the song "Fidelity Fiduciary Bank" from the|
||musical flm *Mary Poppins* features the bank "Dawes, Tomes, Mousely, Grubbs|
||Fidelity Fiduciary Bank".|
||Therefore, the flm that features the mentioned bank is_Mary Poppins_.|



28 

Long-Context LLMs Meet RAG: Overcoming Challenges for Long Inputs in RAG 

|**Question**|Which English chemist discovered the most elements?|
|---|---|
|**Ground**|Humphry Davy|
|**Truth**||
|**Retrieved**|**Doc 1**(Title: Chemistry) J.J. Thomson of Cambridge University discovered the electron|
|**Passages**|and soon after the French scientist Becquerel as well as the couple Pierre and Marie|
||Curie investigated the phenomenon of radioactivity. In a series of pioneering scattering|
||experiments Ernest Rutherford at the University of Manchester discovered the internal|
||structure of the atom and the existence of the proton, classifed and explained the|
||diferent types of radioactivity and successfully transmuted the frst element by|
||bombarding nitrogen with alpha particles. His work on atomic structure was improved|
||on by his students, the Danish physicist Niels Bohr and Henry Moseley. The electronic|
||theory of chemical ...|
||**Doc 2**(Title: Chemistry) weights. The development of the electrochemical theory of|
||chemical combinations occurred in the early 19th century as the result of the work of|
||two scientists in particular, J.J. Berzelius and Humphry Davy, made possible by the|
||prior invention of the voltaic pile by Alessandro Volta. Davy discovered nine new|
||elements including the alkali metals by extracting them from their oxides with electric|
||current. British William Prout frst proposed ordering all the elements by their atomic|
||weight as all atoms had a weight that was an exact multiple of the atomic weight of|
||hydrogen. J.A.R. Newlands devised an early table of ...|
||...|
||**Doc 40**(Title: "History of chemistry") many had already applied the concept of|
||atomism to chemistry. A major example was the ion theory of Svante Arrhenius which|
||anticipated ideas about atomic substructure that did not fully develop until the 20th|
||century. Michael Faraday was another early worker, whose major contribution to|
||chemistry was electrochemistry, in which (among other things) a certain quantity of|
||electricity during electrolysis or electrodeposition of metals was shown to be|
||associated with certain quantities of chemical elements, and fxed quantities of the|
||elements therefore with each other, in specifc ratios. These fndings, like those of|
||Dalton’s combining ratios, were early clues ...|
||**Model Predictions**|
|**Gemma-**|William Hyde Wollaston|
|**2-9B-**||
|**Chat**||
|**Implicit**|William Ramsay|
|**RAG FT**||
|**RAG FT**|The best document to answer this question is **Doc 2(Chemistry)**.|
|**with Rea-**|This document states that **Humphry Davy** "discovered nine new elements|
|**soning**|including the alkali metals by extracting them from their oxides with electric current."|
||Therefore, the answer is_Humphry Davy_.|
||While other documents mention chemists and their discoveries, none of them|
||contradict or provide a higher number of discovered elements than Davy’s nine. This|
||makes Doc 2 the most useful in identifying the English chemist who discovered the|
||most elements.|
||It’s important to note that this answer is based on Doc 2 alone. If other documents|
||mentioned an English chemist who discovered more than nine elements, they would|
||supersede the information in Doc 2. However, without that information, we can|
||confdently say that based on these documents, Humphry Davy discovered the most|
||elements among the English chemists mentioned.|
||**Conclusion:** Based on the given documents,_Humphry Davy_is the English chemist|
||who discovered the most elements.|



29 

Long-Context LLMs Meet RAG: Overcoming Challenges for Long Inputs in RAG 

## **J. Data-Augmented RAG Finetuning on Gemma-2-9B** 

In Figure 5, we illustrate the performance of implicit RAG finetuning on eight datasets with three different base models due to the space limitation. The whole results with Gemma-2-9B models can be found in Figure 11. 


![](raw/papers/2_LLM_NLP/LLM_paper_list/rag/iclr_2025_Long-Context_LLMs_Meet_RAG_Overcoming_Challenges_for_Long_Inputs_in_RAG_images/iclr_2025_Long-Context_LLMs_Meet_RAG_Overcoming_Challenges_for_Long_Inputs_in_RAG.pdf-0030-03.png)


**----- Start of picture text -----**<br>
0.78 0.60 0.50<br>0.76<br>0.55 0.45<br>0.74<br>0.50<br>0.72 Gemma-2-9B-Chat Gemma-2-9B-Chat 0.40<br>RAG FT 0.45 RAG FT<br>0.70<br>Direct FT 0.40 Direct FT 0.35<br>0.68 0.35 Gemma-2-9B-Chat<br>0.66 0.30 RAG FT<br>0.30<br>0.64 Direct FT<br>0.25 0.25<br>0 5 10 15 20 25 30 35 40 0 5 10 15 20 25 30 35 40 0 5 10 15 20 25 30 35 40<br># Retrieved Passages # Retrieved Passages # Retrieved Passages<br>(a) TriviaQA (b) PopQA (c) HotpotQA<br>0.44 0.52 0.40<br>0.42 0.50<br>0.35<br>0.40<br>0.48<br>0.38 Gemma-2-9B-Chat 0.30 Gemma-2-9B-Chat<br>0.46<br>0.36 RAG FT RAG FT<br>0.34 0.44 Direct FT 0.25 Direct FT<br>Gemma-2-9B-Chat<br>0.32 0.42<br>RAG FT 0.20<br>0.30 Direct FT 0.40<br>0.28 0.15<br>0 5 10 15 20 25 30 35 40 0 5 10 15 20 25 30 35 40 0 5 10 15 20 25 30 35 40<br># Retrieved Passages # Retrieved Passages # Retrieved Passages<br>(d) 2wikimultihopqa (e) Webquestions (f) Bamboogle<br>0.70 0.7<br>0.65<br>0.65 0.6<br>0.60<br>0.60 0.5 Gemma-2-9B-Chat<br>0.55<br>0.55 0.4 RAG FT<br>0.50 Direct FT<br>0.50 Gemma-2-9B-Chat Gemma-2-9B-Chat 0.3<br>0.45<br>RAG FT RAG FT<br>0.45 Direct FT 0.40 Direct FT 0.2<br>0.40<br>0 5 10 15 20 25 30 35 40 0 5 10 15 20 25 30 35 40 0 5 10 15 20 25 30 35 40<br># Retrieved Passages # Retrieved Passages # Retrieved Passages<br>(g) ASQA (h) T-REx (i) zsRE<br>RAG Accuracy RAG Accuracy RAG Accuracy<br>RAG Accuracy RAG Accuracy RAG Accuracy<br>RAG Accuracy RAG Accuracy RAG Accuracy<br>**----- End of picture text -----**<br>


Figure 11 | Generalization ability of LLMs fine-tuned with RAG-specific data (RAG FT). RAG FT consistently outperforms the chat LLM w. RAG and the model fine-tuned directly on question-answer pairs (Direct FT). This demonstrates the effectiveness of RAG FT in enabling the LLM to effectively extract knowledge from retrieved context on unseen tasks. Note that Direct FT is evaluated without retrieval to align with its training paradigm and all others are evaluated with retrieval augmentation. (LLM: Gemma-2-9B-Base) 

In Figure 6, we show the power of RAG finetuning with intermediate reasoning on five datasets because of the space limitation. The whole results on all the nine datasets with Gemma-2-9B models can be found in Figure 12. Note that due to the computational complexity of inference with reasoning augmentation, results are shown for 1000 randomly-sampled queries for each dataset. 

30 

Long-Context LLMs Meet RAG: Overcoming Challenges for Long Inputs in RAG 


![](raw/papers/2_LLM_NLP/LLM_paper_list/rag/iclr_2025_Long-Context_LLMs_Meet_RAG_Overcoming_Challenges_for_Long_Inputs_in_RAG_images/iclr_2025_Long-Context_LLMs_Meet_RAG_Overcoming_Challenges_for_Long_Inputs_in_RAG.pdf-0031-01.png)


**----- Start of picture text -----**<br>
0.55<br>0.80<br>0.6 0.50<br>0.75<br>0.45<br>0.5<br>0.70 0.40<br>0.65 Gemma-2-9B-Chat 0.4 Gemma-2-9B-Chat 0.35 Gemma-2-9B-Chat<br>0.60 RAG FT 0.3 RAG FT 0.30 RAG FT<br>0.55 RAG FT w. Int RAG FT w. Int 0.25 RAG FT w. Int<br>Direct FT 0.2 Direct FT Direct FT<br>0.20<br>0.50<br>0 5 10 15 20 25 30 35 40 0 5 10 15 20 25 30 35 40 0 5 10 15 20 25 30 35 40<br># Retrieved Passages # Retrieved Passages # Retrieved Passages<br>(a) TriviaQA (b) PopQA (c) HotpotQA<br>0.60 Gemma-2-9B-Chat 0.60 0.40<br>0.55 RAG FT 0.55 0.35<br>0.50 RAG FT w. Int 0.50<br>Direct FT 0.30<br>0.45 0.45<br>0.25<br>0.40 0.40 Gemma-2-9B-Chat Gemma-2-9B-Chat<br>0.20<br>0.35 0.35 RAG FT RAG FT<br>0.30 0.30 RAG FT w. Int 0.15 RAG FT w. Int<br>Direct FT Direct FT<br>0.25 0.25 0.10<br>0 5 10 15 20 25 30 35 40 0 5 10 15 20 25 30 35 40 0 5 10 15 20 25 30 35 40<br># Retrieved Passages # Retrieved Passages # Retrieved Passages<br>(d) 2wikimultihopqa (e) Webquestions (f) Bamboogle<br>0.8<br>0.7<br>0.7<br>0.7 0.6<br>0.6 0.6 0.5 Gemma-2-9B-Chat<br>RAG FT<br>0.5 0.4<br>0.5 Gemma-2-9B-Chat Gemma-2-9B-Chat RAG FT w. Int<br>0.3 Direct FT<br>RAG FT 0.4 RAG FT<br>0.4 0.2<br>RAG FT w. Int RAG FT w. Int<br>0.3 Direct FT 0.3 Direct FT 0.1<br>0 5 10 15 20 25 30 35 40 0 5 10 15 20 25 30 35 40 0 5 10 15 20 25 30 35 40<br># Retrieved Passages # Retrieved Passages # Retrieved Passages<br>(g) ASQA (h) T-REx (i) zsRE<br>RAG Accuracy RAG Accuracy RAG Accuracy<br>RAG Accuracy RAG Accuracy RAG Accuracy<br>RAG Accuracy RAG Accuracy RAG Accuracy<br>**----- End of picture text -----**<br>


Figure 12 | Evaluating the impact of intermediate reasoning on the performance of RAG-tuned LLMs. Results demonstrate that fine-tuning with an intermediate reasoning step (RAG FT w. Int) leads to further improvements compared to implicit RAG fine-tuning (RAG FT) and direct fine-tuning (Direct FT). Direct FT is evaluated without retrieval to align with its training paradigm and all others are evaluated with retrieval augmentation. Due to the computational complexity of inference with reasoning augmentation, results are shown for 1000 randomly-sampled queries from each dataset. (LLM: Gemma-2-9B-Base) 

31 

Long-Context LLMs Meet RAG: Overcoming Challenges for Long Inputs in RAG 

## **K. Data-Augmented RAG Finetuning on Mistral-Nemo-12B** 

In addition to the comprehensive data-augmented RAG fine-tuning results with three different base LLMs reported in Section 5, we also would like to show the results specifically with the Mistral-Nemo12B models in Figure 13. 


![](raw/papers/2_LLM_NLP/LLM_paper_list/rag/iclr_2025_Long-Context_LLMs_Meet_RAG_Overcoming_Challenges_for_Long_Inputs_in_RAG_images/iclr_2025_Long-Context_LLMs_Meet_RAG_Overcoming_Challenges_for_Long_Inputs_in_RAG.pdf-0032-03.png)


**----- Start of picture text -----**<br>
0.80 0.60<br>0.50<br>0.55<br>0.75 0.45<br>0.50<br>0.70 0.45 0.40<br>0.65 0.40 0.35<br>Mistral-Nemo-12B-Chat Mistral-Nemo-12B-Chat Mistral-Nemo-12B-Chat<br>0.35 0.30<br>0.60 RAG FT RAG FT RAG FT<br>RAG FT w. Int 0.30 RAG FT w. Int 0.25 RAG FT w. Int<br>0.55 Direct FT 0.25 Direct FT 0.20 Direct FT<br>0 5 10 15 20 25 30 35 40 0 5 10 15 20 25 30 35 40 0 5 10 15 20 25 30 35 40<br># Retrieved Passages # Retrieved Passages # Retrieved Passages<br>(a) TriviaQA (b) PopQA (c) HotpotQA<br>0.55 Mistral-Nemo-12B-Chat 0.60 0.40<br>0.50 RAG FT 0.55 0.35<br>0.45 RAG FT w. Int 0.50 0.30<br>Direct FT<br>0.40 0.45 0.25<br>0.35 Mistral-Nemo-12B-Chat 0.20 Mistral-Nemo-12B-Chat<br>0.40<br>0.30 0.35 RAG FT RAG FT w. Int 0.15 RAG FTRAG FT w. Int<br>0.10<br>0.25 Direct FT Direct FT<br>0.30 0.05<br>0 5 10 15 20 25 30 35 40 0 5 10 15 20 25 30 35 40 0 5 10 15 20 25 30 35 40<br># Retrieved Passages # Retrieved Passages # Retrieved Passages<br>(d) 2wikimultihopqa (e) Webquestions (f) Bamboogle<br>0.70<br>0.7<br>0.7 0.65<br>0.60 0.6<br>Mistral-Nemo-12B-Chat<br>0.6 0.55 0.5<br>RAG FT<br>0.50<br>0.4 RAG FT w. Int<br>0.5 Mistral-Nemo-12B-Chat RAG FT 0.450.40 Mistral-Nemo-12B-Chat RAG FT 0.3 Direct FT<br>0.4 RAG FT w. Int 0.35 RAG FT w. Int 0.2<br>Direct FT Direct FT<br>0.30 0.1<br>0 5 10 15 20 25 30 35 40 0 5 10 15 20 25 30 35 40 0 5 10 15 20 25 30 35 40<br># Retrieved Passages # Retrieved Passages # Retrieved Passages<br>(g) ASQA (h) T-REx (i) zsRE<br>RAG Accuracy RAG Accuracy RAG Accuracy<br>RAG Accuracy RAG Accuracy RAG Accuracy<br>RAG Accuracy RAG Accuracy RAG Accuracy<br>**----- End of picture text -----**<br>


Figure 13 | Evaluating RAG-specific tuning with Mistral-Nemo-12B models. Results demonstrate that fine-tuning with an intermediate reasoning step (RAG FT w. Int) leads to further improvements compared to implicit RAG fine-tuning (RAG FT), while implicit RAG fine-tuning outperforms LLMs without RAG-specific tuning (Mistral-Nemo-12B-Chat) and direct fine-tuning (Direct FT). Direct FT is evaluated without retrieval to align with its training paradigm and all others are evaluated with retrieval augmentation. (LLM: Mistral-Nemo-12B-Base) 

32 

Long-Context LLMs Meet RAG: Overcoming Challenges for Long Inputs in RAG 

## **L. Data-Augmented RAG Finetuning on Gemini-1.0-Pro** 

In addition to the comprehensive data-augmented RAG fine-tuning results with three different base LLMs reported in Section 5, we also would like to show the results specifically with the Gemini-1.0-Pro models in Figure 14. Due to the Gemini-1.0-Pro API call credit limitation, we random sample 1000 queries for each dataset. 


![](raw/papers/2_LLM_NLP/LLM_paper_list/rag/iclr_2025_Long-Context_LLMs_Meet_RAG_Overcoming_Challenges_for_Long_Inputs_in_RAG_images/iclr_2025_Long-Context_LLMs_Meet_RAG_Overcoming_Challenges_for_Long_Inputs_in_RAG.pdf-0033-03.png)


**----- Start of picture text -----**<br>
0.65<br>0.80 0.60 0.50<br>0.78 0.55 0.45<br>0.50<br>0.76 0.40<br>Gemini-1.0-Pro 0.45 Gemini-1.0-Pro Gemini-1.0-Pro<br>0.74 RAG FT RAG FT 0.35 RAG FT<br>0.40<br>RAG FT w. Int RAG FT w. Int RAG FT w. Int<br>0.72 Direct FT 0.35 Direct FT 0.30 Direct FT<br>0 5 10 15 20 25 30 35 40 0 5 10 15 20 25 30 35 40 0 5 10 15 20 25 30 35 40<br># Retrieved Passages # Retrieved Passages # Retrieved Passages<br>(a) TriviaQA (b) PopQA (c) HotpotQA<br>0.375<br>0.575 Gemini-1.0-Pro<br>0.50 0.550 RAG FT 0.350<br>0.45 Gemini-1.0-Pro 0.525 RAG FT w. Int 0.325<br>RAG FT 0.500 Direct FT 0.300<br>0.40 RAG FT w. Int 0.475 0.275<br>Gemini-1.0-Pro<br>0.35 Direct FT 0.450 0.250 RAG FT<br>0.425 0.225 RAG FT w. Int<br>0.30 0.400 0.200 Direct FT<br>0.375<br>0 5 10 15 20 25 30 35 40 0 5 10 15 20 25 30 35 40 0 5 10 15 20 25 30 35 40<br># Retrieved Passages # Retrieved Passages # Retrieved Passages<br>(d) 2wikimultihopqa (e) Webquestions (f) Bamboogle<br>0.8 0.75<br>0.7<br>0.7 0.70<br>0.6<br>0.6 Gemini-1.0-Pro 0.65 Gemini-1.0-Pro<br>RAG FT 0.60 0.5 RAG FT<br>0.5 RAG FT w. Int RAG FT w. Int<br>0.55 Gemini-1.0-Pro 0.4<br>Direct FT Direct FT<br>0.4 0.50 RAG FT<br>0.3<br>RAG FT w. Int<br>0.3 0.45<br>Direct FT 0.2<br>0.40<br>0 5 10 15 20 25 30 35 40 0 5 10 15 20 25 30 35 40 0 5 10 15 20 25 30 35 40<br># Retrieved Passages # Retrieved Passages # Retrieved Passages<br>(g) ASQA (h) T-REx (i) zsRE<br>RAG Accuracy RAG Accuracy RAG Accuracy<br>RAG Accuracy RAG Accuracy RAG Accuracy<br>RAG Accuracy RAG Accuracy RAG Accuracy<br>**----- End of picture text -----**<br>


Figure 14 | Evaluating RAG-specific tuning with Gemini-1.0-Pro models. Results demonstrate that finetuning with an intermediate reasoning step (RAG FT w. Int) leads to further improvements compared to implicit RAG fine-tuning (RAG FT), while implicit RAG fine-tuning outperforms LLMs without RAGspecific tuning (Gemini-1.0-Pro) and direct fine-tuning (Direct FT). Direct FT is evaluated without retrieval to align with its training paradigm and all others are evaluated with retrieval augmentation. Due to the Gemini-1.0-Pro API call credit limitation, results are shown for 1000 randomly-sampled queries from each dataset. (LLM: Gemini-1.0-Pro) 

33 

Long-Context LLMs Meet RAG: Overcoming Challenges for Long Inputs in RAG 

## **M. Training data scaling and RAG performance.** 

|Number of Retrieval Passages|5k|20k|50k|200k|
|---|---|---|---|---|
|10|0.5942|0.5925|0.6058|0.6277|
|20|0.5909|0.5925|0.6078|0.6294|
|30|0.5787|0.5792|0.6072|0.6150|
|40|0.5582|0.5582|0.5859|0.5983|
|Avg.|0.5805|0.5806|0.6017|0.6176|



Table 11 | Impact of RAG-specific training data scale on LLM performance in RAG. 

To investigate the influence of the size of the training data on the effectiveness of RAG-specific tuning, we fine-tune the Gemma-2-9B-Base model using varying amounts (5k to 200k samples) of mixed training data from NQ, WoW, Fever, and MMLU. Table 11 presents the evaluation results on the NQ dataset, demonstrating a clear positive correlation between the scale of training data and the performance of the resulting LLM in RAG. Increasing the amount of training data consistently leads to improved accuracy, highlighting the benefits of leveraging larger datasets for fine-tuning LLMs in RAG applications. 

## **N. RAG-specific tuning data inside SFT mixtures** 

|Dataset|base|SFT only|SFT + RAG-FT|
|---|---|---|---|
|MT-Bench|2.3125|5.8969|5.6031|
|NQ|0.2105|0.5687|0.6033|
|TriviaQA|0.4940|0.7155|0.7481|



Table 12 | Combining RAG-specific data with general SFT data for enhanced LLM performance in RAG. 

Having established the effectiveness of RAG-specific fine-tuning for improving LLM performance in RAG tasks, we now investigate whether combining RAG-specific data with general SFT data can further enhance performance while preserving the LLM’s general capabilities ( _e.g._ , reasoning and long-form generation), as a way to assess the potential of the proposed tuning methods to be useful for construction of foundation models. We train the Gemma-2-9B model using two different strategies: (1) SFT data only: The LLM is trained solely on general SFT data (Ultrachat 200k). (2) SFT data + RAG-specific data: The LLM is trained on a combination of Ultrachat 200k and 50k RAG-specific data (the same data used in Figure 5). We evaluate the resulting models on MT-Bench to assess their general language capabilities and on NQ and TriviaQA to measure their RAG performance. 

Table 12 presents the results, demonstrating that incorporating RAG-specific data into the SFT process can significantly improve the LLM’s performance on RAG tasks while maintaining its performance on general language tasks. This finding suggests that combining task-specific and general-purpose data during fine-tuning can be a viable strategy for enhancing LLMs in specialized applications without compromising their overall capabilities. 

34 

