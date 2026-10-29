Under review 

# MEASURING AND ENHANCING TRUSTWORTHINESS OF LLMS IN RAG THROUGH GROUNDED ATTRIBUTIONS AND LEARNING TO REFUSE 

**Maojia Song**[1] _[∗]_ **, Shang Hong Sim**[1] **, Rishabh Bhardwaj** _[∗]_[1] **, Hai Leong Chieu**[2] , **Navonil Majumder**[1] , **Soujanya Poria**[1] 

1 Singapore University of Technology and Design, 

2 DSO National Laboratories, Singapore 

_{_ maojia ~~s~~ ong, shanghong ~~s~~ im, rishabh ~~b~~ hardwaj _}_ @mymail.sutd.edu.sg chaileon@dso.org.sg 

_{_ navonil majumder, sporia _}_ @sutd.edu.sg 

## ABSTRACT 

LLMs are an integral component of retrieval-augmented generation (RAG) systems. While many studies focus on evaluating the overall quality of end-to-end RAG systems, there is a gap in understanding the appropriateness of LLMs for the RAG task. To address this, we introduce TRUST-SCORE, a holistic metric that evaluates the trustworthiness of LLMs within the RAG framework. Our results show that various prompting methods, such as in-context learning, fail to effectively adapt LLMs to the RAG task as measured by TRUST-SCORE. Consequently, we propose TRUST-ALIGN, a method to align LLMs for improved TRUST-SCORE performance. The LLaMA-3 family, aligned using our method, significantly outperforms open-source LLMs of similar sizes on ASQA ( _↑_ 13.4), QAMPARI ( _↑_ 28.9), and ELI5 ( _↑_ 13.7). TRUST-ALIGN also significantly enhances models’ ability to correctly refuse and provide quality citations. We also demonstrate the effectiveness of TRUST-ALIGN across different open-weight models, including the LLaMA series (1b to 8b), Qwen-2.5 series (0.5b to 7b), and Phi3.5 (3.8b). We release our code at https://github.com/declare-lab/ trust-align. 

## 1 INTRODUCTION 

Large language models (LLMs) are increasingly used for information retrieval, but a persistent issue is their tendency to produce hallucinations—responses that appear convincing but are factually incorrect (Ji et al., 2023). This undermines their effectiveness as reliable sources of accurate information. One common approach to mitigate this is to integrate external knowledge through a Retrieval-Augmented Generation (RAG) setup. 

External knowledge, when attached to Language Models, has been observed to increase the likelihood of generating correct tokens, as evidenced by reduced perplexity in Khandelwal et al. (2019), and subsequently impact downstream task performance, such as machine translation (Zheng et al., 2021) and classification (Bhardwaj et al., 2023). Similarly, LLMs have been found to generate more desired responses when connected to external knowledge (documents) with the help of a retriever (Shuster et al., 2021; B´echard & Ayala, 2024), which get enhanced by asking model to attribute documents they used to generate the answer (Gao et al., 2023b; Hsu et al., 2024). 

In this paper, we study the ability of LLMs to ground their knowledge on the provided documents, rather than relying on the encoded knowledge acquired during the training process, which we refer to as _parametric_ knowledge. An LLM response is considered grounded if it correctly answers the 

> _∗_ These authors contributed equally. 

1 

Under review 

question using only the information in the documents, and the generated response can be inferred from the inline attributions (citations) to the documents. 

An important aspect of groundedness we study is the _refusal_ capabilities of LLMs i.e. if LLMs refuse to generate claims owing to the documents attached lack sufficient information related to the question. Besides this, we study several other important behaviors of LLMs including the tendency to answer a question, a fraction of claims that are grounded in the documents, and if the cited documents support the generated statements (set of claims). 

To comprehensively understand LLMs’ groundedness, we propose a new metric **TRUST-SCORE** . It assesses an LLM across multiple dimensions: 1) The ability to discern which questions can be answered or refused based on the provided documents (Grounded Refusals); 2) Claim recall scores for the answerable questions; 3) The extent to which generated claims are supported by the corresponding citations; and 4) The relevance of the citations to the statements. Unlike existing metrics that primarily assess the overall performance of RAG systems (Gao et al., 2023b)—where a weak retriever can significantly decrease the scores—TRUST-SCORE is designed to specifically measure the LLM’s performance within a RAG setup, isolating it from the influence of retrieval quality. 

Our investigation in Section 6.2 shows that many state-of-the-art systems, including GPT-4 and Claude-3.5-Sonnet, heavily rely on their parametric knowledge to answer questions (OpenAI, 2023; Anthropic, 2024). This reliance limits their suitability for RAG tasks, where models should base responses solely on the provided documents, resulting in a low TRUST-SCORE. Additionally, prompting approaches intended to enhance model groundability have proven ineffective, as models become overly sensitive to the prompt, leading to exaggerated refusals or excessive responsiveness shown in Appendix F.4. To enhance the groundedness of LLMs, i.e., achieve a higher TRUST-SCORE, we propose an alignment method, **TRUST-ALIGN** . This approach first constructs an alignment dataset consisting of 19K questions, documents, positive (preferred) responses, and negative (unpreferred) responses. The dataset covers a range of LLM errors—Inaccurate Answer, Over-Responsiveness, Excessive Refusal, Over-Citation, and Improper Citation. We regard these errors as LLM hallucinations within an RAG framework. 

Evaluations on the benchmark datasets ASQA, QAMPARI, and ELI show that models trained with TRUST-ALIGN outperform the competitive baselines on TRUST-SCORE, with respective improvements of 13.44%, 28.89%, and 13.66%. TRUST-ALIGN significantly enhances the ability of models to correctly refuse or provide grounded answers, with correct refusals observed to increase by 16.93%, 22.15%, and 4.11% compared to the baselines. Moreover, citation groundedness scores were observed to increase by 26.75%, 31.86%, and 31.33%. Due to the gamification of the metric, where parametric knowledge can artificially inflate the scores, we notice mixed results on exact match recall. Specifically, we observe a notable increase in recall scores for QAMPARI (32.64%) and ELI5 (5.56%), but a decrease of 3.32% for ASQA. 

Our key contributions to this work are as follows: 

- We study LLM groundedness problem, where model model responses should be derived from retrieved documents (external memory) rather than the parametric knowledge (knowledge stored in model parameters). 

- To measure LLM’s groundedness under RAG, we introduce **TRUST-SCORE** , a holistic metric for quantifying LLM’s grounding errors. 

- We propose **TRUST-ALIGN** , an alignment approach designed to improve the trustworthiness of LLMs in RAG (Figure 2). It first creates an alignment dataset of 19K samples with paired positive and negative responses, followed by aligning the model using direct preference optimization (DPO) (Rafailov et al., 2024b). 

**Comparison with existing approaches.** Current evaluations of RAG focus on the overall system performance (Gao et al., 2023b; Xu et al., 2024), conflating the effects of retriever quality and LLM performance in the metric scores (Fan et al., 2024). This highlights the need for new ways to measure LLM effectiveness in RAG systems without the influence of the retriever. The work by Thakur et al. (2024) is closest to ours, as it analyzes the refusal capabilities of LLMs in a RAG context but lacks holistic evaluation, as it does not account for both response and citation groundedness. On the other hand, Ye et al. (2024); Hsu et al. (2024); Huang et al. (2024b) propose frameworks to improve LLM response groundedness but overlook refusal behaviors in their metrics. Ignoring refusal be- 

2 

Under review 

haviors, retriever influence, citation and answer groundedness weakens the ability of current metrics to effectively measure LLM performance in RAG. TRUST-SCORE comprehensively evaluates LLM performance, including refusal, citation, and answer groundedness, while TRUST-ALIGN creates a corresponding alignment dataset, making the metric and approach more unique and holistic for LLM evaluations and alignment in RAG. A more detailed comparison can be found in Appendix C. 

## 2 PROBLEM DESCRIPTION 

## 2.1 TASK SETUP 

Given a question _q_ and a set of retrieved documents _D_ as input, the LLM is instructed to generate a response _S_ which consists of a set of citation-grounded statements _{s_ 1 _, . . . , sn}_ ; each statement _si_ follows a set of inline citations _Ci_ = _{ci,_ 1 _, ci,_ 2 _, . . .}_ referring to the documents in _D_ . If _D_ is not sufficient to answer _q_ , the gold response would be a refusal statement[1] , such as, _“I apologize, but I couldn’t find an answer to your question in the search results”_ . Otherwise, the response would follow the pattern: “statement1 [1][2] statement2 [3]” where [1][2] and [3] denote the enumeration of documents that leads to statement1 and statement2, respectively. 


![](raw/papers/2_LLM_NLP/LLM_paper_list/rag/iclr_2025_Measuring_and_Enhancing_Trustworthiness_of_LLMs_in_RAG_through_Grounded_Attributions_and_Learning_to_Refuse_images/iclr_2025_Measuring_and_Enhancing_Trustworthiness_of_LLMs_in_RAG_through_Grounded_Attributions_and_Learning_to_Refuse.pdf-0003-05.png)


**----- Start of picture text -----**<br>
TRUSTSCORE<br>Response Truthfulness Attribution Groundedness<br>Grounded Refusal (F1RG) Exact Match (EM [F1] AC) Citations Grounded (F1CG)<br>Ar  : Set of answered questions Ag  : Set of answerable questions | A | : Number of elements in the set : Exact match recall for question  qi si , { ci ,1,  ci ,2, …}: Statement and corresponding citations<br>si , { ci ,1,  ci ,2, …}: Statement and corresponding citations : Statement-wise citation recall : Citation precision<br>**----- End of picture text -----**<br>


Figure 1: TRUST-SCORE calculation shown as a computational graph. 

## 2.2 ON ANSWERABILITY OF A QUESTION 

To label if a response should be a refusal or consist of claims, we define the notion of _**answerability**_ . A question _q_ is considered answerable if _D_ contains sufficient information to answer _q_ . Formally, we label a question as answerable if a subset of the retrieved documents entails at least one of the gold claims; otherwise, _q_ is unanswerable and thus should result in a ground truth _**refusal**_ . A refusal response contains no claims or citations but provides a generic message conveying the LLM’s inability to respond to _q_ . 

## 2.3 HALLUCINATION IN LLM IN RAG 

We define an LLM’s response as grounded when it correctly answers a question using only the information in the documents, and the response can be inferred from the inline citations to those documents. When a response is not grounded, it is considered a case of hallucination. We define _**hallucination**_ as an erroneous LLM response, categorized into five types: (1) _Inaccurate Answer_ – The generated statements _S_ fail to cover the claims in the gold response, (2) _Over-Responsiveness_ – The model answers a question that should result in a refusal, (3) _Excessive Refusal_ – The model refuses to answer a question that is answerable, (4) _Overcitation_ – The model generates redundant citations, and (5) _Improper Citation_ – The citations provided do not support the statement. Next, we 

1While there are many applications where retrieved documents are used to assist LLMs in providing better responses, and LLMs are expected to use their parametric knowledge, in this paper, we study the problem of complete groundedness—i.e., all claims should be documents derivable, typically making this an IR task. 

3 

Under review 

introduce a comprehensive metric to concretely measure hallucinations in LLMs, i.e., to assess an LLM’s groundedness or trustworthiness[2] . 

## 3 METRICS FOR LLM-IN-RAG 

Given a question _q_ and the corresponding ground truth response _AG_ = _{ag_ 1 _, . . . , agn}_ consisting of gold claims, we define the claims obtainable from the provided documents as _AD_ = _{ad_ 1 _, . . . , adn}_ and the claims generated in the response as _AR_ = _{ar_ 1 _, . . . , arn}_ . We aim to measure two aspects of an LLM in RAG: 1) the Correctness of the generated claims (Response Truthfulness); and 2) the Correctness of citations generated (Attribution Groundedness). 

**Insufficiency of the existing metrics.** Gao et al. (2023b) measure Response Truthfulness by first computing the per-sample exact match recall (EMr) score for gold claims _AG_ , disregarding how many of these claims are obtainable from _D_ . This is followed by averaging the recall scores across samples to obtain a single score for the dataset. This method introduces inconsistencies: models that rely on parametric knowledge ( _Mp_ ) may generate gold claims not found in _D_ , leading to an artificially inflated recall value. In contrast, an ideal LLM ( _Mi_ ) would rely solely on _D_ to generate responses (a desired trait) and would be constrained by an upper recall limit of _[|][A][G] |A[∩] G[A] |[D][|]_ , which varies depending on the question. This approach presents two key problems: (1) _Recall Consolidation_ : Since the measurement range depends on the claims present in _D_ , it is infeasible to provide a consistent, consolidated EMr score across the dataset, (2) _Recall Gamification_ : _Mp_ may have a higher upper limit on EMr (up to 1) because they can generate gold claims not present in _D_ (an undesirable trait), unlike _Mi_ that depend entirely on _D_ . 

**Answer Calibration.** To address the challenges of recall consolidation and gamification in existing evaluation metrics, we propose new metrics that measure sample-wise recall score based on the fraction of gold claims that can be obtained from _D_ . Specifically, this involves computing _|AG∩AD|_ , which measures the exact match (EM) recall after calibrating the gold claims. This approach sets a maximum recall limit of 1 for all models. For dataset-wide scoring, we consolidate per-sample EM recall scores using two methods: 1) **EM** _[α]_ **AC**[:][The][average][recall][score][across][samples] _[answered]_[by] the LLM, i.e., samples where _AR_ = _∅_ ; 2) **EM** _[β]_ **AC**[:][The average recall score across samples that are] _answerable_ , i.e., samples where _AG ∩ AD_ = _∅_[3] . These metrics, illustrated in Fig. 1, are then combined into a single score, **EM[F1] AC**[,][which serves as a comprehensive measure of how well the LLM] grounds its claims on the document _D_ . This combined metric not only facilitates the consolidation of recall but also addresses issues related to recall gamification. 

**Scoring Refusals.** An important capability of an LLM in RAG is its ability to identify when a response is unanswerable based on the provided documents _D_ . To measure this, we introduce a metric called Grounded Refusals. This metric evaluates the model’s refusal performance by calculating dataset-wide precision and recall for both ground-truth answerable cases and refusals. These values are then combined into their respective F1 scores, **F1ref** for refusals and **F1ans** for answerable cases. The final score, **F1RG** , is the average of these two F1 scores, as shown in Figure 1. 

**Measuring Attribution Groundedness.** While Response Truthfulness metrics like **EM[F1] AC**[and] **F1CG** evaluate the quality of generated claims, it is equally important to measure how well these statements are supported by relevant citations—what we call Attribution Groundedness. To this end, we adopt two sub-metrics from (Gao et al., 2023b): Citation Recall ( **CR** ) and Citation Precision ( **CP** ). To compute **CR** , we first determine if a generated statement _si_ is supported by its cited documents using an NLI model[4] , thus obtaining sample-wise recall scores **CR** _[s][i]_ . Then we take the mean across all samples to obtain the final **CR** score (Figure 1). To compute **CP** , we first score each citation _ci,j_ of a statement _si_ , followed by computing the average across citations in a response _S_ (sample-wise score). The dataset-wide citation score is computed by averaging the citation scores 

> 2In this paper, we use LLM groundedness and trustworthiness interchangeably in the context of RAG. 

> 3Notably, both **EM** _α_ **AC**[and] **[EM]** _[β]_ **AC**[sum over][samples that are][both answered and][answerable,][differing][pri-] marily in their normalization values. 

> 4An NLI model checks if the cited document entails the statement. 

4 

Under review 

across all the samples. To provide a single metric for Attribution Groundedness, we calculate the harmonic mean of **CP** and **CR** , resulting in the final score, **F1CG** . 

Thus, we define a new metric **TRUST-SCORE** =[1] 3[(] **[F1][RG]**[ +] **[ EM] AC[F1]**[+] **[ F1][CG]**[)] _[.]_ 

We relegate a detailed computation on the metric to Appendix D and Figure 1. 

**Responsiveness.** To measure the answering tendency of an LLM, we define **Responsiveness** . It is the fraction of answered questions, denoted by the Answered Ratio (AR %), which is calculated # answered as AR % =[A model is expected to show a high AR% for answerable questions and a] # total questions[.] low AR% for unanswerable ones, with the scores expected to align with the dataset distribution. 


![](raw/papers/2_LLM_NLP/LLM_paper_list/rag/iclr_2025_Measuring_and_Enhancing_Trustworthiness_of_LLMs_in_RAG_through_Grounded_Attributions_and_Learning_to_Refuse_images/iclr_2025_Measuring_and_Enhancing_Trustworthiness_of_LLMs_in_RAG_through_Grounded_Attributions_and_Learning_to_Refuse.pdf-0005-05.png)


**----- Start of picture text -----**<br>
1 Seed Prompt Curation 4 Positive Answer Generation<br>ASQA Retrieve Answerable Questions<br>Question<br>QAMPARIELI5 Wikipedia,Sphere documentsTop-100 QuestionsSeed Set10k Ove Johansson(Gold Claim 1) DocumentsOracle synthesizerGPT-4 ... Matt Prater at 64 yards[Gold Claim 2]Johansson in a 1976... [Gold Claim 1], ... Ove. CitationMapper Positive Answer (r  )Johansson in a 1976... Matt Prater at 64yards ...[1][3][2][4], ... Ove. +<br>Questions Text Clustering Knowledge Scoring Filter top 3/4kquestions Documents (Gold Claim 2)Matt Prater Seed Set10k Aug. Set70k<br>Unanswerable Questions Qns r      + Qns r      +<br>Positive Answer: “I apologize, but I couldn't find an answer to your question in the search results.” Docs Docs<br>Question Top-100Docs RecombinationDocument 2 Augmented Prompt Curation Question Combinationof 5 docs Question Combinationof 5 docs ... DocumentsQuestionsAug. Set70k Positive AnsOracle DocsQuestionsSeed Set Supervised Finetuning  LLaMA-2-7b5 Negative Answer Generation 70k Responses HallucinationOnly 40k have CalculateScore top-50%Filter to 20kFilter Answer (r  )Negative-<br>3 Answerability Labelling Set of 5 DocsQuestionsAug. Set SFT Inference DocsQnsAug. Set20k r     r      +-<br>Gold claim 1 Doc 1 Doc 4<br>Document Gold claim 3Gold claim 2 Entailmentpattern:[1,1,0] [1,1,0][1,1,0] Doc 2Doc 3 [1,0,0][0,0,0] Doc 5 Union [1,1,0] Questions Positive Answer 6 Alignment Direct PreferenceOptimization DPO-<br>TRUE NLI [1,0,0] Answerable Set of  5 Documents  Negative Answer LLaMA<br>ParametersTrainable ParametersFrozen Seed Set AugmentedSet Document<br>20k Aug. Set<br>**----- End of picture text -----**<br>


Figure 2: Overview of the TRUST-ALIGN. **Left** : The curation of both seed and augmented prompts (QD pairs) and an example of the answerability labeling process during the retrieval stage. **Right** : The response paired data generation process. First, we obtain positive answers and then select hard negative answers. Finally, we align our model via DPO. 

## 4 THE TRUST-ALIGN DATASET 

To align LLMs towards trustworthiness, we propose a new approach, **TRUST-ALIGN** . The approach constructs an LLM trustworthiness alignment dataset, where each sample in the dataset consists of a question _q_ , a set of retrieved documents _D_ , and a pair of positive (preferred) and negative (unpreferred) responses ( _r_[+] , _r[−]_ ). The positive response corresponds to an answer that encompasses expected gold claims for _q_ and corresponding citations referring to the documents. If _D_ is not sufficient to answer _q_ , _r_[+] is assigned a refusal response, while _r[−]_ is its non-refusal counterpart. We build the dataset in multiple steps: 1) Obtain a set of high-quality and diverse questions, 2) Obtain documents for each question, 3) Augmenting ( _q, D_ ) pairs that cover diverse hallucination types, 4) Construct positive responses entailing gold claims, and 5) Construct negative (unpreferred) responses by prompting a fine-tuned model and observing its hallucinations. We relegate fine-grained details about the dataset to Figure 2 and Appendix E. 

**Collecting Quality Questions.** The dataset construction begins by collecting a set of high-quality and diverse questions from ASQA, QAMPARI, and ELI5, referred to as **seed samples** . We first divide the questions into _k_ clusters and use Mixtral-8x7B to assign each a quality score from 1 to 7, based on how difficult they are to answer without additional information. Clusters with scores of 4 or higher are selected. Next, we sample questions from the clusters of each dataset to construct approximately 10K questions in the seed set. 

**Collecting** _**D**_ **’s.** Next, we collect relevant documents for each question in the seed set by querying Wikipedia and Common Crawl, retrieving the top 100 documents. We filter out seed questions where 

5 

Under review 

relevant documents are not retrieved. We then identify 5 documents that perform as well as the full 100 in terms of EM recall, referring to these as _**oracle**_ documents for question _q_ .[5] Gold claims for each _q_ are sourced from the respective datasets. 

**Augmenting** _**(q,D)**_ **set.** Using the questions and oracle documents, we create diverse samples (i.e., varying combinations of relevant and irrelevant documents) to trigger multiple hallucinations from LLMs (Section 2.3). The document order is shuffled to avoid citation bias. To construct unanswerable questions, we select documents similar to those entailing gold claims but still irrelevant to _q_ . This process results in approximately 70K question-document pairs. 

**Obtaining r**[+] **and r** _[−]_ **.** To generate preferred responses, by prompting GPT-4, we stitch together the gold claims and citations[6] . For unanswerable questions, we assign a ground truth refusal response. To obtain quality negative (unpreferred) responses, we fine-tune LLaMA-2-7b on the source datasets, creating _Msft_ . Testing _Msft_ on the 70K dataset identified 40K responses with hallucinations. Table 1 shows hallucination severity ( _ei_ ) and frequency ( _wi_ ). To obtain good negative samples, we first rank each of the 40K responses according to their severity score _eq_ . We then select the top 50%[7] of the corresponding samples for both answerable and unanswerable responses. We perform DPO using this set of 19k samples to obtain the final aligned model. 

|**Hallucination type**|**Frequency (**_wi_**)**|**Frequency (**_wi_**)**|**Severity (**_ei_**)**|
|---|---|---|---|
|Unwarranted Refusal|8,786|0.50|_I_(_Ag_=_∅,Ar_=_∅_)|
|Over Responsiveness|13,067|0.50|_I_(_Ag_=_∅,Ar_=_∅_)|
|Overcitation|12,656|0.34|1 - CP|
|Improper Citation|9,592|0.26|1 - CR|
|Inaccurate Claims|14,783|0.40|1 - EMF1<br>AC|



Table 1: Fraction of each hallucination amongst all the observed hallucinations in _Msft_ (40,985), with possible overlap. _wi_ shows the severity computation of each hallucination. _I_ condition = 1 if condition is True otherwise it is 0. See Fig. 5 for the detailed breakdown of the last three errors. 

## 5 EXPERIMENTAL SETUP 

**Evaluation datasets.** We evaluate on the test-set of attributable factoid and long-form questionanswering tasks from ASQA (Stelmakh et al., 2023), QAMPARI (Amouyal et al., 2023), and ELI5 (Fan et al., 2019). Additionally, we include ExpertQA (Malaviya et al., 2024) for OOD evaluations. For each question, we append the top 5 retrieved documents. For ELI5 and ExpertQA, the ground truth answers are decomposed into three claims. The dataset statistics are detailed in Appendix H.2. 

**Models Studied.** To comprehensively measure performance of open-source models, we perform TRUST-SCORE computations on vanilla and TRUST-ALIGNed version of a range of open-weight models such as LLaMA series (LLaMA-2-7b, LLaMA-2-13b, LLaMA-2-13b, etc.), Qwen series (Qwen-2.5-0.5b, Qwen-2.5-7b, etc.) and Phi3.5-mini. 

**Baselines.** Models[8] trained with TRUST-ALIGN are compared against the following baselines: 

- ICL (Gao et al., 2023b): Prepends two demonstrations to each query, consisting of an example query, top-5 retrieved documents, and an inline cited answer 

- PostCite (Gao et al., 2023b): Generates an uncited answer in a closed-book setting, then retrieves most similar documents from top-5 documents using GTR for citations. 

- PostAttr (Gao et al., 2023b): Similar to POSTCITE, produces an uncited response in a closed-book setting, but uses the TRUE-NLI model to find the best matching citation among top-5 documents. 

> 5Clustering and document retrieval details are in Appendix E. 

> 6Prompt template can be found at Table 21. 

> 7See Appendix F.8 for more details on this hyperparameter. 

> 8All models used are instruct tuned or chat versions. 

6 

Under review 

- Self-RAG (Asai et al., 2024): Trains the LLM to retrieve relevant documents on demand using reflection tokens, enhancing generation quality. The 7b and 13b models were reproduced using the default settings for evaluation. 

- FRONT (Huang et al., 2024b): Uses a fine-grained attribution framework to improve grounding and citation quality. We reproduce its 7b model for comparison. 

## 6 RESULTS AND ANALYSIS 

|**Model**<br>**Type**|**ASQA**|_(610 answerable, 3_|_38 unanswerable)_|**QAMPA**|**RI**_(295 answerable_|_, 705 unanswerable)_|**ELI5**|_(207 answerable, 7_|_93 unanswerable)_|
|---|---|---|---|---|---|---|---|---|---|
||**Resp.**|**Trustw**|**orthiness**|**Resp.**|**Trustw**|**orthiness**|**Resp.**|**Trust**|**worthiness**|
||**AR (%)**|**Truthfullness**|**Att-Grd.**<br>**TRUST**<br>**F1CG**|**AR (%)**|**Truthfullness**|**Att-Grd.**<br>**TRUST**<br>**F1CG**|**AR (%)**|**Truthfullness**|**Att-Grd.**<br>**TRUST**<br>**F1CG**|
|||**EMF1**<br>**AC**<br>**F1RG**|||**EMF1**<br>**AC**<br>**F1RG**|||**EMF1**<br>**AC**<br>**F1RG**||
|LLaMA-2<br>-7b<br>ICL<br>PostCite<br>PostAttr<br>Self-RAG<br>FRONT<br>SFT<br>DPO|0.00<br>10.44<br>10.44<br>100.00<br>100.00|0.00<br>26.28<br>0.07<br>35.23<br>0.07<br>35.23<br>45.19<br>39.15<br>60.47<br>39.15|0.00<br>8.76<br>0.00<br>11.77<br>0.00<br>11.77<br>63.49<br>49.28<br>68.86<br>56.16|0.00<br>34.40<br>34.40<br>96.00<br>100.00|0.00<br>41.35<br>0.00<br>57.34<br>0.00<br>57.34<br>6.81<br>28.23<br>17.27<br>22.78|0.00<br>13.78<br>9.50<br>22.28<br>3.78<br>20.37<br>19.95<br>18.33<br>24.26<br>21.44|0.50<br>0.90<br>0.90<br>73.50<br>100.00|0.00<br>46.71<br>1.86<br>44.98<br>1.86<br>44.98<br>14.94<br>40.20<br>21.66<br>17.15|0.00<br>15.57<br>5.04<br>17.29<br>0.00<br>15.61<br>13.80<br>22.98<br>52.72<br>30.51|
||80.17<br>65.30|53.21<br>63.43<br>52.48<br>66.12|79.61<br>65.42<br>83.94<br>67.51|31.60<br>32.30|33.76<br>71.13<br>32.03<br>71.67|46.37<br>50.42<br>49.42<br>51.04|29.50<br>21.60|21.58<br>63.30<br>22.54<br>63.27|39.59<br>41.49<br>47.35<br>44.39|
|LLaMA-2<br>-13b<br>ICL<br>PostCite<br>PostAttr<br>Self-RAG|17.41<br>90.51<br>90.51<br>100.00|21.52<br>41.40<br>2.21<br>49.91<br>2.21<br>49.91<br>48.52<br>39.15|13.83<br>25.58<br>1.53<br>17.88<br>0.17<br>17.43<br>69.79<br>52.49|26.50<br>100.00<br>100.00<br>72.70|0.44<br>59.57<br>0.00<br>22.78<br>0.00<br>22.78|0.00<br>20.00<br>8.05<br>10.28<br>2.95<br>8.58|46.40<br>76.60<br>76.60<br>22.10|19.97<br>54.81<br>2.27<br>38.05<br>2.27<br>38.05|4.73<br>26.50<br>0.72<br>13.68<br>0.09<br>13.47|
||||||2.71<br>48.58|26.91<br>**26.07**||12.77<br>58.68|24.54<br>**32.00**|
|LLaMA-3.2<br>-1b<br>ICL<br>PostCite<br>PostAttr<br>FRONT<br>SFT<br>DPO|60.23<br>43.57<br>45.78<br>79.11|35.95<br>50.94<br>0.59<br>50.22<br>0.48<br>48.42<br>48.22<br>54.48|9.96<br>32.28<br>0.24<br>17.02<br>0.00<br>16.30<br>48.29<br>50.33|19.20<br>41.20<br>34.00<br>98.60|6.32<br>52.64<br>0.32<br>49.79<br>0.63<br>48.43<br>7.57<br>24.54|0.38<br>19.78<br>1.61<br>17.24<br>0.21<br>16.42<br>15.32<br>15.81|88.40<br>18.40<br>18.40<br>97.20|12.87<br>27.10<br>2.04<br>50.88<br>2.04<br>50.88<br>16.11<br>20.76|5.23<br>15.07<br>1.02<br>17.98<br>0.07<br>17.66<br>30.19<br>22.35|
||63.82<br>41.67|45.61<br>63.91<br>38.64<br>58.61|73.10<br>60.87<br>79.35<br>58.87|26.00<br>20.00|27.98<br>68.20<br>27.22<br>67.92|37.96<br>44.71<br>49.42<br>48.19|20.50<br>9.60|14.56<br>63.93<br>13.20<br>59.35|37.28<br>38.59<br>48.21<br>40.25|
|LLaMA-3.2<br>-3b<br>ICL<br>PostCite<br>PostAttr<br>FRONT<br>SFT<br>DPO|1.27<br>47.26<br>47.15<br>95.25|2.04<br>27.98<br>31.03<br>56.59<br>29.76<br>56.71|53.95<br>27.99<br>22.99<br>36.87<br>4.69<br>30.39|34.10<br>39.60<br>42.00<br>92.70|16.06<br>59.65<br>6.34<br>55.22<br>5.10<br>53.74<br>12.99<br>32.89|12.87<br>29.53<br>6.83<br>22.80<br>0.27<br>19.70<br>19.19<br>21.69|21.90<br>92.80<br>92.80<br>86.90|18.55<br>55.56<br>18.12<br>25.14<br>18.48<br>25.14<br>19.95<br>32.21|30.70<br>34.94<br>4.44<br>15.90<br>0.53<br>14.72<br>41.97<br>31.38|
|||63.19<br>49.45|57.46<br>**56.70**|||||||
||68.04<br>77.85|49.23<br>65.47|75.63<br>63.44|27.60<br>48.20|28.09<br>70.22<br>29.13<br>70.85|38.03<br>45.45<br>45.65<br>48.54|14.70<br>17.50|15.92<br>62.59|53.33<br>43.95|
|||59.82<br>66.38|84.21<br>**70.14**|||||18.33<br>62.79|55.87<br>**45.66**|
|LLaMA-3<br>-8b<br>ICL<br>PostCite<br>PostAttr<br>FRONT<br>SFT<br>DPO|1.48<br>77.53<br>77.53<br>99.05|3.01<br>28.58<br>32.98<br>53.31<br>32.98<br>53.31<br>62.25<br>41.62|86.50<br>39.36<br>28.01<br>38.10<br>5.95<br>30.75<br>66.14<br>56.67|3.90<br>87.00<br>87.00<br>100.00|5.92<br>48.60<br>6.10<br>34.52<br>6.10<br>34.52<br>13.53<br>22.78|20.24<br>24.92<br>8.42<br>16.35<br>1.64<br>14.09<br>20.42<br>18.91|0.00<br>62.00<br>62.00<br>99.50|0.00<br>44.23<br>20.80<br>45.88<br>20.80<br>45.88<br>18.99<br>17.85|0.00<br>14.74<br>8.06<br>24.91<br>1.25<br>22.64<br>44.69<br>27.18|
||68.99<br>56.43|52.35<br>66.06<br>53.94<br>65.49|80.95<br>66.45<br>88.26<br>69.23|24.20<br>22.40|33.85<br>71.11|48.01<br>50.99|23.60<br>15.50|22.57<br>65.06<br>20.81<br>63.57|46.85<br>44.83<br>50.24<br>44.87|
||||||35.35<br>70.73|58.77<br>**54.95**||||
|∆||_↓_3.37<br>_↑_16.93|_↑_26.75<br>_↑_13.44||_↑_32.64<br>_↑_22.15|_↑_31.86<br>_↑_28.89||_↑_5.56<br>_↑_4.11|_↑_31.33<br>_↑_13.66|



Table 2: LLaMA family evaluated on the ASQA, QAMPARI, and ELI5 datasets. Best TRUSTSCORE values within the baseline and TRUST-ALIGN sets are marked in **bold** (corresponding rows highlighted ), with the overall best TRUST-SCORE values underlined. ∆ values indicate the difference between the best baseline and best TRUST-ALIGN rows. **AR%** := Answered Ratio in %; **EM[F1] AC**[:= Exact Match F1 (Calibrated);] **[ F1][RG]**[ := Grounded Refusals F1;] **[ F1][CG]**[ := Citation Grounded] F1; **TRUST** := TRUST-SCORE; **Resp.** := Responsiveness; **Att-Grd.** := Attribution Groundedness. 

**TRUST-ALIGN boosts trustworthiness over baseline methods.** Our best TRUST-ALIGNed models significantly outperform the best-performing baselines by 13.44% (ASQA), 28.89% (QAMPARI), and 13.66% (ELI5) on TRUST-SCORE (see ∆ row and highlighted rows in Table 2). This improvement suggests that our models are more capable of generating responses grounded in the documents. 

**TRUST-ALIGN improves models’ refusal capability.** We also observe that the best TRUSTALIGNed models significantly outperform the best-performing baselines on F1RG scores by 16.93% (ASQA), 22.15% (QAMPARI), and 4.11% (ELI5). This indicates that TRUST-ALIGN significantly enhances the ability of models to correctly refuse or provide answers. 

**TRUST-ALIGN enhances models’ citation quality.** The best TRUST-ALIGNed models significantly outperform the best-performing baselines on F1CG by 26.75% (ASQA), 31.86% (QAMPARI), and 31.33% (ELI5). This demonstrates that aligning with TRUST-ALIGN improves the model’s ability to provide citations that fully support claims. 

7 

Under review 

**TRUST-ALIGN has mixed effects on EM[F1] AC[.]** We observe a notable increase in EM[F1] AC[for QAM-] PARI (32.64%) and ELI5 (5.56%) but a decrease of 3.37% for ASQA. The mixed performance in ASQA can be explained by the composition of EM[F1] AC[,][which][is][derived][from][EM] _[α]_ AC[and][EM] _[β]_ AC (Eq. (9)). As shown in Appendix I, our models generally achieve higher EM _[α]_ AC[compared to base-] lines (54.63% for Llama-3.2-3b-DPO vs. 52.94% for FRONT-3.2-3b) despite having a lower AR% (77.85% for Llama-3.2-3b-DPO vs. 95.25% for FRONT-3.2-3b). This suggests that our models have a higher expected value for EM _[q]_ AC (per-sample EM recall), as the denominator depends on the number of answered questions. This trend is observed across models and datasets as shown in Appendix I. 

However, in ASQA, our models underperform in EM[F1] AC[due to the overwhelmingly adverse impact] of EM _[β]_ AC[.][The][recall][of][answerable][questions (R][ans][) is][lower for our][model compared to][baselines] (89.02% for Llama-3.2-3b-DPO vs. 98.69% for FRONT-3.2-3b), which rarely refuse questions. As a result, fewer terms are summed in the numerator of EM _[β]_ AC[,][while][the][denominator][remains] constant (the number of answerable questions). This leads to a lower overall EM[F1] AC[score. To further] analyze the baseline models’ performance, we investigated how much of their answering ability relies on parametric knowledge versus document-based information, as discussed in Section 6.2 and Appendix F.3. 

**Models aligned with DPO generally outperform those trained with SFT.** Taking LLaMA-3.23b as an example, DPO models outperform SFT models in TRUST-SCORE by 6.70% (ASQA), 3.96% (QAMPARI), and 0.04% (ELI5). Across models shown in Table 2, DPO also attains significantly better F1CG compared to SFT (eg 84.21% vs. 75.63% in LLaMA-3.2-3b ASQA), indicating that training with DPO produces a model that has better citation quality. Despite DPO’s mixed performance on EM[F1] AC[and][F1][RG][relative][to][SFT,][DPO][models][achieve][better][overall][T][RUST][-S][CORE] scores. 

|**Model**<br>**Type**|**ASQA**|_(610 answerable, 3_|_38 unanswerable)_<br>**orthiness**<br>**Att-Grd.**<br>**TRUST**<br>**F1CG**|**QAMPA**|**RI**_(295 answerable_|_, 705 unanswerable)_<br>**orthiness**<br>**Att-Grd.**<br>**TRUST**<br>**F1CG**|**ELI5**|_(207 answerable, 7_|_93 unanswerable)_|
|---|---|---|---|---|---|---|---|---|---|
||**Resp.**<br>**AR (%)**|**Trustw**||**Resp.**<br>**AR (%)**|**Trustw**||**Resp.**<br>**AR (%)**|**Trustw**|**orthiness**|
|||**Truthfullness**<br>**EMF1**<br>**AC**<br>**F1RG**|||**Truthfullness**<br>**EMF1**<br>**AC**<br>**F1RG**|||**Truthfullness**<br>**EMF1**<br>**AC**<br>**F1RG**|**Att-Grd.**<br>**TRUST**<br>**F1CG**|
|Qwen-2.5<br>-0.5b<br>ICL<br>PostCite<br>PostAttr<br>SFT<br>DPO|29.85<br>46.10<br>46.10|20.96<br>47.19<br>8.55<br>50.84<br>8.55<br>50.84|0.35<br>22.83<br>8.23<br>22.54<br>2.23<br>20.54|11.40<br>17.00<br>17.00|2.45<br>50.67<br>0.67<br>52.51<br>0.67<br>52.51|0.00<br>17.71<br>5.72<br>19.63<br>0.90<br>18.03|82.30<br>89.80<br>89.80|13.73<br>33.14<br>9.87<br>27.10<br>9.87<br>27.10|0.37<br>15.75<br>4.10<br>13.69<br>0.68<br>12.55|
||83.44<br>71.84|38.71<br>58.03<br>50.59<br>61.28|57.47<br>51.40<br>52.40<br>54.76|18.50<br>17.90|16.02<br>61.35<br>15.76<br>61.84|27.82<br>35.06<br>29.73<br>35.78|35.50<br>21.70|10.50<br>57.19<br>13.68<br>60.79|19.57<br>29.09<br>22.72<br>32.40|
|Qwen-2.5<br>-1.5b<br>ICL<br>PostCite<br>PostAttr<br>SFT<br>DPO|98.52<br>71.73<br>71.73|50.55<br>41.74<br>16.36<br>52.46<br>16.36<br>52.46|6.69<br>32.99<br>15.40<br>28.07<br>4.45<br>24.42|85.00<br>11.20<br>11.20|15.60<br>41.27<br>3.44<br>51.11<br>3.44<br>51.11|8.61<br>21.83<br>13.95<br>22.83<br>1.07<br>18.54|99.40<br>91.50<br>91.50|20.56<br>17.78<br>15.63<br>26.71<br>15.63<br>26.71|4.99<br>14.44<br>5.17<br>15.84<br>0.62<br>14.32|
||78.27<br>72.57|44.23<br>58.75<br>52.68<br>62.38|71.08<br>58.02<br>66.81<br>60.62|25.50<br>20.00|23.89<br>69.66<br>23.80<br>68.46|37.68<br>43.74<br>50.98<br>47.75|41.30<br>33.60|14.14<br>55.35<br>19.03<br>57.91|27.69<br>32.39<br>31.63<br>36.19|
|Qwen-2.5<br>-3b<br>ICL<br>PostCite<br>PostAttr<br>SFT<br>DPO|27.43<br>8.76<br>8.76|37.72<br>51.36<br>9.58<br>35.30<br>9.58<br>35.30|51.72<br>46.93<br>10.94<br>18.61<br>36.29<br>27.06|22.30<br>0.10<br>0.10|23.17<br>63.27<br>0.00<br>41.31<br>0.00<br>41.31|41.20<br>42.55<br>0.00<br>13.77<br>25.00<br>22.10|68.80<br>49.70<br>49.70|29.12<br>46.31<br><br>|34.34<br>**36.59**<br><br>|
|||||||||21.73<br>48.49<br>21.73<br>48.49|7.56<br>25.93<br>1.31<br>23.84|
||75.21<br>49.47|47.26<br>60.61<br>55.19<br>63.76|73.09<br>60.32<br>78.64<br>65.86|27.20<br>48.10|28.80<br>68.12<br>35.69<br>70.31|37.34<br>44.75<br>45.64<br>50.55|34.50<br>13.50|14.85<br>61.47<br>22.52<br>64.38|35.87<br>37.40<br>42.01<br>42.97|
|Qwen-2.5<br>-7b<br>ICL<br>PostCite<br>PostAttr<br>SFT<br>DPO|92.09<br>91.46<br>91.46|58.94<br>54.34|75.46<br>**62.91**|56.30<br>26.70<br>26.70|28.92<br>63.67|39.28<br>**43.96**|82.70<br>95.60<br>95.60|28.27<br>37.13<br>21.82<br>22.23<br>21.82<br>22.23|44.13<br>36.51<br>7.03<br>17.03<br>0.96<br>15.00|
|||27.52<br>45.93<br>27.52<br>45.93|4.19<br>25.88<br>17.92<br>30.46||8.59<br>60.16<br>8.59<br>60.16|1.05<br>23.27<br>13.55<br>27.43||||
||65.30<br>59.49|50.73<br>64.50|82.07<br>65.77|31.70<br>32.10|33.58<br>70.10<br>30.11<br>70.68|49.08<br>50.92<br>53.48<br>51.42|25.50<br>21.00|20.78<br>64.25<br>24.30<br>63.79|46.89<br>43.97<br>47.02<br>45.04|
|||55.04<br>66.22|83.57<br>**68.28**|||||||
|Phi3.5<br>-mini<br>ICL<br>PostCite<br>PostAttr<br>SFT<br>DPO|63.19<br>23.10<br>23.10|50.24<br>51.95<br>14.98<br>41.38<br>14.98<br>41.38|42.64<br>48.28<br>9.40<br>21.92<br>1.24<br>19.20|70.20<br>76.90<br>76.90|11.91<br>43.90<br>3.57<br>42.36<br>3.57<br>42.36|12.26<br>22.69<br>4.49<br>16.81<br>0.46<br>15.46|81.50<br>84.50<br>84.50|27.59<br>37.17<br>20.50<br>30.81<br>21.26<br>30.81|30.14<br>31.63<br>4.67<br>18.66<br>0.68<br>17.58|
||66.46<br>66.56|51.92<br>64.34<br>52.23<br>64.20|82.77<br>66.34<br>85.36<br>67.26|29.10<br>30.10|35.04<br>73.93|49.38<br>52.78|24.50<br>24.90|22.50<br>65.70|46.79<br>45.00|
||||||36.42<br>73.95|53.40<br>**54.59**||23.39<br>67.62|47.42<br>**46.14**|
|∆||_↓_3.90<br>_↑_11.88|_↑_8.11<br>_↑_5.37||_↑_7.50<br>_↑_10.28|_↑_14.12<br>_↑_10.63||_↓_5.73<br>_↑_21.31|_↑_13.08<br>_↑_9.55|



Table 3: Qwen2.5 and Phi3.5 families evaluated on the three datasets. Best TRUST-SCORE values within the baseline and TRUST-ALIGN sets are marked in **bold** (corresponding rows highlighted ), with the overall best TRUST-SCORE values underlined. ∆ values indicate the difference between the best baseline and best TRUST-ALIGN rows. 

**TRUST-ALIGN generalizes across model families and sizes.** As seen in Table 3, our experiments across various sizes and architectures demonstrate that the TRUST-ALIGN approach consis- 

8 

Under review 

tently improves the models’ TRUST-SCORE. In small models, we observe a drastic improvement in TRUST-SCORE compared to ICL baselines after applying TRUST-ALIGN (Qwen-2.5-0.5b ASQA improves from 22.83% to 54.76%). A similar trend is seen in bigger models (Qwen-2.5-7b ASQA improves from 62.91% to 68.28%), demonstrating the scalability of TRUST-ALIGN across model sizes. We note that TRUST-ALIGN yields the largest improvements on small models. Applying TRUST-ALIGN on Phi3.5 yields significant TRUST-SCORE improvement compared to ICL baseline18.98% (ASQA), 31.90% (QAMPARI), and 14.51% (ELI5). This demonstrates that our findings are robust across different architectures. 

## 6.1 ANALYSIS 

## 6.1.1 DATA SYNTHESIS TECHNIQUES 

|||**ASQA**||**QAMPARI**||**ELI5**<br>**Trustworthiness**<br>**Truthfullness**<br>**Att-Grd.**<br>**TRUST**<br>**EMF1**<br>**AC**<br>**F1RG**<br>**F1CG**|
|---|---|---|---|---|---|---|
||**Resp.**|**Trustworthiness**|**Resp.**|**Trustworthiness**|**Resp.**||
||**AR (%)**|**Truthfullness**<br>**Att-Grd.**<br>**TRUST**<br>**EMF1**<br>**AC**<br>**F1RG**<br>**F1CG**|**AR (%)**|**Truthfullness**<br>**Att-Grd.**<br>**TRUST**<br>**EMF1**<br>**AC**<br>**F1RG**<br>**F1CG**|**AR (%)**||
|**DPO-LLaMA-2-7b**|65.30|52.48<br>66.12<br>83.94<br>**67.51**|31.10|32.09<br>71.83<br>51.33<br>**51.75**|21.60|22.54<br>63.27<br>48.43<br>**44.75**|
|TRUST-ALIGNw/o. augmented instructions<br>TRUST-ALIGNw/o. answer HT<br>TRUST-ALIGNw/o. citation HT<br>TRUST-ALIGNw/o. refusal HT<br>GPT-4 as critic|79.43<br>77.74<br>77.32<br>79.11<br>70.36|53.54<br>63.33<br>81.15<br>66.01<br>53.29<br>63.7<br>81.2<br>66.06<br>52.55<br>63.88<br>81.51<br>65.98<br>53.55<br>63.33<br>81.85<br>66.24<br>54.91<br>65.29<br>78.47<br>66.22|32.20<br>33.40<br>33.10<br>31.10<br>25.90|33.14<br>70.82<br>45.94<br>49.97<br>33.56<br>71.36<br>46.17<br>50.36<br>34.13<br>71.40<br>46.91<br>50.81<br>34.40<br>71.35<br>48.12<br>51.29<br>30.77<br>70.29<br>48.87<br>49.98|29.50<br>27.60<br>26.70<br>28.30<br>23.50|23.98<br>63.30<br>40.28<br>42.52<br>23.47<br>63.56<br>38.28<br>41.77<br>22.65<br>64.33<br>42.81<br>43.26<br>22.93<br>64.05<br>41.18<br>42.72<br>17.27<br>62.24<br>42.38<br>40.63|



Table 4: Ablations of data synthesis techniques for LLaMA-2-7b on three evaluation datasets using refusal prompting; **AR%** := Answered Ratio in %; **EM[F1] AC**[:= Exact Match F1 (Calibrated);] **[ F1][RG]**[:= Grounded Refusals] F1; **F1CG** := Citation Grounded F1; **TRUST** := TRUST score; **HT** := hallucination types; **Resp.** := Responsiveness; **Att-Grd.** := Attribution Groundedness. The original error types in Section 2.3 were summarized into three main classes: answer-related (Inaccurate Answer), citation-related (Overcitation, Improper Citation), refusal-related (Over Responsiveness, Excessive Refusal). 

Table 4 demonstrates the effectiveness of our data construction approach. Adding augmented prompts targeting five hallucination types improves TRUST-SCORE by 1.50% (ASQA), 1.78% (QAMPARI), and 2.23% (ELI5), highlighting the value of our synthetic data in improving trustworthiness and reducing hallucinations. Similarly, dropping data for specific hallucination types causes TRUST-SCORE to consistently drop, suggesting the importance of each subtype. In particular, removing refusal-related hallucinations significantly affects F1RG, with decreases of 2.79% for ASQA, 0.46% for QAMPARI, and 2.03% for ELI5, showing that refusal-related data is critical for improving a model’s ability to decide when to answer, thereby enhancing trustworthiness. 

To validate our approach, we compared it against a strong baseline: the GPT-4 critic pipeline (Huang et al., 2024a; Li et al., 2024a; Huang et al., 2024b), an automated data collection method that uses advanced prompts to iteratively identify and correct errors (details in Appendix G). Our data pipeline outperformed GPT-4, particularly on ELI5 (with a 4.12% improvement), further demonstrating the effectiveness of our method. 

## 6.1.2 IMPORTANCE OF REFUSAL SAMPLES IN TRUST-ALIGN 

Table 5 underscores the importance of including refusal samples during fine-tuning in ASQA. Training with refusal samples in TRUST-ALIGN achieves the highest TRUST-SCORE score of 69.23%. Removing all unanswerable questions from the training set creates a set without refusals or refusalrelated hallucination types. Without refusal samples, TRUST-SCORE scores drop significantly—by 10.2% for LLaMA-3-8b and 11.41% for LLaMA-2-7b. This decline is particularly pronounced in F1RG (down 26.34% for LLaMA-3-8b and 26.97% for LLaMA-2-7b) and F1CG (down 6.87% for LLaMA-3-8b and 6.57% for LLaMA-2-7b). 

We also observe that EM[F1] AC[is higher for LLaMA-3-8b in the answerable-only set compared to the] set with refusals. As discussed in main results, this is because EM _[β]_ AC[favors over-responsive models,] which artificially inflates EM[F1] AC[. When refusal samples are excluded, responsiveness (AR%) reaches] 100%, meaning the models answer all questions, even without supporting documents. This suggests that the models rely more on ungrounded parametric knowledge, as discussed in Section 6.2. 

9 

Under review 

|||**Model**|**Responsiveness (AR%)**|**EMF1**<br>**AC**|**F1RG**|**F1CG**|**TRUST**|
|---|---|---|---|---|---|---|---|
|Only|Answerable|DPO-LLaMA-2-7b<br>DPO-LLaMA-3-8b|100<br>100|51.79<br>**56.54**|39.15<br>39.15|77.37<br>81.39|56.10<br>59.03|
|With|Refusal|DPO-LLaMA-2-7b<br>DPO-LLaMA-3-8b|65.30<br>56.43|52.48<br>53.94|**66.12**<br>65.49|83.94<br>**88.26**|67.51<br>**69.23**|



Table 5: Effect of adding refusal samples on the ASQA. 

|**Model**|**Type**|**AR (%)**|**EMF1**<br>**AC**|**F1RG**|**F1CG**|**TRUST**|
|---|---|---|---|---|---|---|
||ICL|78.24|21.42|38.71|0.44|20.19|
|Qwen-2.5<br>-0.5b|PostCite<br>PostAttr<br>Front|51.41<br>51.41<br>99.86|13.32<br>13.32<br>18.27|48.08<br>48.08<br>24.05|5.6<br>1.49<br>34.62|22.33<br>20.96<br>25.65|
||DPO|32.96|18.16|63.31|35.07|**38.85**|
||ICL|98.34|30.67|26.09|6.89|21.22|
|Qwen-2.5<br>-1.5b|PostCite<br>PostAttr<br>Front|62.19<br>62.19<br>99.59|22.22<br>22.22<br>29.15|48.66<br>48.66<br>24.6|16.92<br>13.15<br>50.22|29.27<br>28.01<br>34.66|
||DPO|30.2|25.06|68.38|51.44|**48.29**|
||ICL|68.88|35.14|49.65|42.67|42.49|
|Qwen-2.5<br>-3b|PostCite<br>PostAttr<br>Front|0.05<br>0.05<br>95.48|0<br>0<br>25.67|40.66<br>40.66<br>29.86|0<br>0<br>44.48|13.55<br>13.55<br>33.34|
||DPO|17.15|20.97|65.79|60.25|**49.0**|
||ICL|84.56|36.33|42.28|56.09|44.9|
|Qwen-2.5<br>-7b|PostCite<br>PostAttr<br>Front|42.14<br>42.14<br>65.51|25.58<br>25.58<br>32.41|54.9<br>54.9<br>55.56|13.77<br>12.46<br>67.35|31.42<br>30.98<br>51.77|
||DPO|24.99|25.57|69.16|62.7|**52.48**|
||ICL|85.15|37.49|40.22|36.14|37.95|
|Phi3.5<br>-mini|PostCite<br>PostAttr<br>Front|52.01<br>52.01<br>97.37|27.96<br>27.96<br>28.19|53.64<br>53.64<br>27.5|7.39<br>5.7<br>65.82|29.66<br>29.1<br>40.5|
||DPO|26.05|27.69|69.56|61.6|**52.95**|
||ICL|0.51|0|41.01|9.52|16.84|
|LLaMA-2<br>-7b|PostCite<br>PostAttr<br>Front|5.62<br>5.62<br>100|4.85<br>4.85<br>9.33|44.27<br>44.27<br>23.92|5.23<br>2.26<br>74.75|18.12<br>17.13<br>36.00|
||DPO|20.01|25.03|67.91|62.46|**51.8**|
||ICL|90|21.55|32.83|9.04|21.14|
|LLaMA-3.2<br>-1b|PostCite<br>PostAttr<br>Front|30.84<br>48.41<br>95.62|5.48<br>8.24<br>20.83|49.1<br>47.72<br>29.26|2.67<br>1.5<br>37.45|19.08<br>19.15<br>29.18|
||DPO|15.44|20.32|64.87|62.1|**49.1**|
||ICL|58.74|33.5|51.21|38.37|41.03|
|LLaMA-3.2<br>-3b|PostCite<br>PostAttr<br>Front|82.85<br>82.85<br>83.36|25.68<br>25.45<br>27.24|38.11<br>38.58<br>43.34|5.29<br>3.4<br>50.91|23.03<br>22.48<br>40.5|
||DPO|7.24|11.72|56.93|78.35|**49.0**|
||ICL|0.65|2.82|42.5|69.46|38.26|
|LLaMA-3<br>-8b|PostCite<br>PostAttr<br>Front|15.68<br>15.68<br>99.26|14.06<br>14.06<br>30.34|50.08<br>50.08<br>24.92|7.09<br>6.29<br>56.7|23.74<br>23.47<br>37.32|
||DPO|16.41|27.36|67.07|70.11|**54.85**|
|GPT-3.5|ICL|59.47|36.65|56.39|63.93|52.32|
|GPT-4|ICL|72.20|41.32|52.91|69.83|**54.69**|
|Claude-3.5|ICL|73.95|11.68|51.91|10.7|24.76|



Table 6: Generalization test results on ExpertQA using refusal prompting. 

10 

Under review 

## 6.1.3 OUT-OF-DOMAIN ANALYSIS 

Following Huang et al. (2024a), we use ExpertQA (Malaviya et al., 2024) to assess our model’s generalizability. As shown in Table 6, the open-source ICL models perform significantly worse w.r.t. TRUST-SCORE as compared to the proprietary models, with a 9.79% gap between ICL-LLaMA-3-8b and ICL-GPT-4. TRUST-ALIGN not only closes this gap but establishes a lead: the tuned LLaMA3-8b model achieves the highest TRUST score of 54.85, surpassing 54.69 of GPT-4. 

In LLaMA-3-8B, using TRUST-ALIGN results in a 16.59% improvement in grounded refusal judgment (F1RG) compared to ICL-LLaMA-3-8B and significantly outperforms GPT-3.5 and Claude 3.5 in both grounded citation generation (F1CG) and refusal judgment (F1RG). Although GPT-3.5 and GPT-4 achieve higher EM[F1] AC scores, indicating better document understanding and answer extraction, they rely heavily on parametric knowledge (Section 6.2 and Appendix F.3). This leads to higher responsiveness, which can result in less precise and trustworthy responses, as reflected in their TRUST-SCORE scores. In contrast, our model’s superior performance in F1CG and F1RG demonstrates its strength in refusal and grounding, making it more reliable. The trends remain the same for the other language model backbones. 

## 6.2 STUDYING PARAMETRIC KNOWLEDGE ACCESS 

For an LLM used for RAG task, it is important to study the tendency of LLM towards grounding its knowledge on the provided documents. To partially quantify this, we compute EM score for questions that are unanswerable by the provided documents; thus a fraction of cases where _AG ∩AD_ = _∅_ but _AG_ = _∅_ (more details on the metric in Appendix F.2). In Table 9, our analysis reveals that responsive models (high AR%) tend to rely on parametric knowledge more frequently (high Pscore). Notably, closed-source models like GPT-4 exhibit higher parametric knowledge usage compared to open-source and TRUST-ALIGN models. However, Pscore only partially captures the models’ utilization of parametric knowledge. For instance, it does not account for cases where the document contains the answer and the model still relied on the parametric knowledge to generate the correct answer (also present in the document). This phenomenon is evident in Table 11, where on ASQA, GPT-4 achieves a significantly higher **EM[F1] AC**[than our models, yet its attribution groundedness score] **F1CG** is five points lower. 

## 7 CONCLUSION 

In this study, we introduced a new holistic metric to evaluate the suitability of LLMs for RAG applications, where they are expected to ground their responses in the provided documents. We proposed TRUST-SCORE, which comprehensively measures the quality of answers, citations, and refusal performance of an LLM. Additionally, we presented TRUST-ALIGN, a method that uses a constructed dataset to align models for improved TRUST-SCORE performance. By applying Direct Preference Optimization (DPO) techniques, we trained LLaMA-2-7b and LLaMA-3-8b on this dataset, significantly reducing hallucinations in an RAG environment. Our approach, TRUST-ALIGN, demonstrates performance comparable to major closed-source models like GPT-4. 

## REFERENCES 

- Samuel Joseph Amouyal, Tomer Wolfson, Ohad Rubin, Ori Yoran, Jonathan Herzig, and Jonathan Berant. Qampari: An open-domain question answering benchmark for questions with many answers from multiple paragraphs, 2023. URL https://arxiv.org/abs/2205.12665. 

- Anthropic. Introducing claude 3.5 sonnet. _Anthropic News_ , 2024. URL https://www. anthropic.com/news/claude-3-5-sonnet. 

- Akari Asai, Zeqiu Wu, Yizhong Wang, Avirup Sil, and Hannaneh Hajishirzi. Self-RAG: Learning to retrieve, generate, and critique through self-reflection. In _The Twelfth International Conference on Learning Representations_ , 2024. URL https://openreview.net/forum?id= hSyW5go0v8. 

- Patrice B´echard and Orlando Marquez Ayala. Reducing hallucination in structured outputs via retrieval-augmented generation. _arXiv preprint arXiv:2404.08189_ , 2024. 

11 

Under review 

- Rishabh Bhardwaj, Yingting Li, Navonil Majumder, Bo Cheng, and Soujanya Poria. knn-cm: A nonparametric inference-phase adaptation of parametric text classifiers. In _Findings of the Association for Computational Linguistics: EMNLP 2023_ , pp. 13546–13557, 2023. 

- Tom B. Brown, Benjamin Mann, Nick Ryder, Melanie Subbiah, Jared Kaplan, Prafulla Dhariwal, Arvind Neelakantan, Pranav Shyam, Girish Sastry, Amanda Askell, Sandhini Agarwal, Ariel Herbert-Voss, Gretchen Krueger, Tom Henighan, Rewon Child, Aditya Ramesh, Daniel M. Ziegler, Jeffrey Wu, Clemens Winter, Christopher Hesse, Mark Chen, Eric Sigler, Mateusz Litwin, Scott Gray, Benjamin Chess, Jack Clark, Christopher Berner, Sam McCandlish, Alec Radford, Ilya Sutskever, and Dario Amodei. Language models are few-shot learners, 2020. URL https://arxiv.org/abs/2005.14165. 

- Jan Buchmann, Xiao Liu, and Iryna Gurevych. Attribute or abstain: Large language models as long document assistants, 2024. URL https://arxiv.org/abs/2407.07799. 

- Abhimanyu Dubey, Abhinav Jauhri, Abhinav Pandey, Abhishek Kadian, Ahmad Al-Dahle, Aiesha Letman, Akhil Mathur, Alan Schelten, Amy Yang, Angela Fan, Anirudh Goyal, Anthony Hartshorn, Aobo Yang, Archi Mitra, Archie Sravankumar, Artem Korenev, Arthur Hinsvark, Arun Rao, Aston Zhang, Aurelien Rodriguez, Austen Gregerson, Ava Spataru, Baptiste Roziere, Bethany Biron, Binh Tang, Bobbie Chern, Charlotte Caucheteux, Chaya Nayak, Chloe Bi, Chris Marra, et al. The llama 3 herd of models, 2024. URL https://arxiv.org/abs/2407. 21783. 

- Angela Fan, Yacine Jernite, Ethan Perez, David Grangier, Jason Weston, and Michael Auli. Eli5: Long form question answering, 2019. URL https://arxiv.org/abs/1907.09190. 

- Wenqi Fan, Yujuan Ding, Liangbo Ning, Shijie Wang, Hengyun Li, Dawei Yin, Tat-Seng Chua, and Qing Li. A survey on rag meeting llms: Towards retrieval-augmented large language models. In _Proceedings of the 30th ACM SIGKDD Conference on Knowledge Discovery and Data Mining_ , pp. 6491–6501, 2024. 

- Luyu Gao, Zhuyun Dai, Panupong Pasupat, Anthony Chen, Arun Tejasvi Chaganty, Yicheng Fan, Vincent Zhao, Ni Lao, Hongrae Lee, Da-Cheng Juan, and Kelvin Guu. RARR: Researching and revising what language models say, using language models. In Anna Rogers, Jordan BoydGraber, and Naoaki Okazaki (eds.), _Proceedings of the 61st Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)_ , pp. 16477–16508, Toronto, Canada, July 2023a. Association for Computational Linguistics. doi: 10.18653/v1/2023.acl-long.910. URL https://aclanthology.org/2023.acl-long.910. 

- Tianyu Gao, Howard Yen, Jiatong Yu, and Danqi Chen. Enabling large language models to generate text with citations, 2023b. 

- Yunfan Gao, Yun Xiong, Xinyu Gao, Kangxiang Jia, Jinliu Pan, Yuxi Bi, Yi Dai, Jiawei Sun, and Haofen Wang. Retrieval-augmented generation for large language models: A survey. _arXiv preprint arXiv:2312.10997_ , 2023c. 

- Or Honovich, Roee Aharoni, Jonathan Herzig, Hagai Taitelbaum, Doron Kukliansy, Vered Cohen, Thomas Scialom, Idan Szpektor, Avinatan Hassidim, and Yossi Matias. TRUE: Re-evaluating factual consistency evaluation. In Song Feng, Hui Wan, Caixia Yuan, and Han Yu (eds.), _Proceedings of the Second DialDoc Workshop on Document-grounded Dialogue and Conversational Question Answering_ , pp. 161–175, Dublin, Ireland, May 2022. Association for Computational Linguistics. doi: 10.18653/v1/2022.dialdoc-1.19. URL https://aclanthology.org/ 2022.dialdoc-1.19. 

- I-Hung Hsu, Zifeng Wang, Long T. Le, Lesly Miculicich, Nanyun Peng, Chen-Yu Lee, and Tomas Pfister. Calm: Contrasting large and small language models to verify grounded generation, 2024. URL https://arxiv.org/abs/2406.05365. 

- Chengyu Huang, Zeqiu Wu, Yushi Hu, and Wenya Wang. Training language models to generate text with citations via fine-grained rewards, 2024a. 

12 

Under review 

- Lei Huang, Xiaocheng Feng, Weitao Ma, Yuxuan Gu, Weihong Zhong, Xiachong Feng, Weijiang Yu, Weihua Peng, Duyu Tang, Dandan Tu, and Bing Qin. Learning fine-grained grounded citations for attributed large language models. In Lun-Wei Ku, Andre Martins, and Vivek Srikumar (eds.), _Findings of the Association for Computational Linguistics ACL 2024_ , pp. 14095–14113, Bangkok, Thailand and virtual meeting, August 2024b. Association for Computational Linguistics. URL https://aclanthology.org/2024.findings-acl.838. 

- Bin Ji, Huijun Liu, Mingzhe Du, and See-Kiong Ng. Chain-of-thought improves text generation with citations in large language models. _Proceedings of the AAAI Conference on Artificial Intelligence_ , 38(16):18345–18353, Mar. 2024. doi: 10.1609/aaai.v38i16.29794. URL https://ojs.aaai.org/index.php/AAAI/article/view/29794. 

- Ziwei Ji, Nayeon Lee, Rita Frieske, Tiezheng Yu, Dan Su, Yan Xu, Etsuko Ishii, Ye Jin Bang, Andrea Madotto, and Pascale Fung. Survey of hallucination in natural language generation. _ACM Computing Surveys_ , 55(12):1–38, March 2023. ISSN 1557-7341. doi: 10.1145/3571730. URL http://dx.doi.org/10.1145/3571730. 

- Zhengbao Jiang, Frank F Xu, Luyu Gao, Zhiqing Sun, Qian Liu, Jane Dwivedi-Yu, Yiming Yang, Jamie Callan, and Graham Neubig. Active retrieval augmented generation. _arXiv preprint arXiv:2305.06983_ , 2023. 

- Vladimir Karpukhin, Barlas Oguz, Sewon Min, Patrick Lewis, Ledell Wu, Sergey Edunov, Danqi Chen, and Wen-tau Yih. Dense passage retrieval for open-domain question answering. In Bonnie Webber, Trevor Cohn, Yulan He, and Yang Liu (eds.), _Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing (EMNLP)_ , pp. 6769–6781, Online, November 2020. Association for Computational Linguistics. doi: 10.18653/v1/2020.emnlp-main.550. URL https://aclanthology.org/2020.emnlp-main.550. 

- Muhammad Khalifa, David Wadden, Emma Strubell, Honglak Lee, Lu Wang, Iz Beltagy, and Hao Peng. Source-aware training enables knowledge attribution in language models, 2024. 

- Urvashi Khandelwal, Omer Levy, Dan Jurafsky, Luke Zettlemoyer, and Mike Lewis. Generalization through memorization: Nearest neighbor language models. _arXiv preprint arXiv:1911.00172_ , 2019. 

- Patrick Lewis, Ethan Perez, Aleksandra Piktus, Fabio Petroni, Vladimir Karpukhin, Naman Goyal, Heinrich K¨uttler, Mike Lewis, Wen tau Yih, Tim Rockt¨aschel, Sebastian Riedel, and Douwe Kiela. Retrieval-augmented generation for knowledge-intensive nlp tasks, 2021. 

- Dongfang Li, Zetian Sun, Xinshuo Hu, Zhenyu Liu, Ziyang Chen, Baotian Hu, Aiguo Wu, and Min Zhang. A survey of large language models attribution. _arXiv preprint arXiv:2311.03731_ , 2023. 

- Dongfang Li, Zetian Sun, Baotian Hu, Zhenyu Liu, Xinshuo Hu, Xuebo Liu, and Min Zhang. Improving attributed text generation of large language models via preference learning. In Lun-Wei Ku, Andre Martins, and Vivek Srikumar (eds.), _Findings of the Association for Computational Linguistics ACL 2024_ , pp. 5079–5101, Bangkok, Thailand and virtual meeting, August 2024a. Association for Computational Linguistics. URL https://aclanthology.org/2024. findings-acl.301. 

- Weitao Li, Junkai Li, Weizhi Ma, and Yang Liu. Citation-enhanced generation for llm-based chatbots, 2024b. 

- Nelson Liu, Tianyi Zhang, and Percy Liang. Evaluating verifiability in generative search engines. In Houda Bouamor, Juan Pino, and Kalika Bali (eds.), _Findings of the Association for Computational Linguistics: EMNLP 2023_ , pp. 7001–7025, Singapore, December 2023. Association for Computational Linguistics. doi: 10.18653/v1/2023.findings-emnlp.467. URL https://aclanthology.org/2023.findings-emnlp.467. 

- Chaitanya Malaviya, Subin Lee, Sihao Chen, Elizabeth Sieber, Mark Yatskar, and Dan Roth. Expertqa: Expert-curated questions and attributed answers, 2024. URL https://arxiv.org/ abs/2309.07852. 

13 

Under review 

Yu Meng, Mengzhou Xia, and Danqi Chen. SimPO: Simple preference optimization with a reference-free reward. _arXiv preprint arXiv:2405.14734_ , 2024. 

- Jacob Menick, Maja Trebacz, Vladimir Mikulik, John Aslanides, Francis Song, Martin Chadwick, Mia Glaese, Susannah Young, Lucy Campbell-Gillingham, Geoffrey Irving, et al. Teaching language models to support answers with verified quotes. _arXiv preprint arXiv:2203.11147_ , 2022. 

- Reiichiro Nakano, Jacob Hilton, Suchir Balaji, Jeff Wu, Long Ouyang, Christina Kim, Christopher Hesse, Shantanu Jain, Vineet Kosaraju, William Saunders, et al. Webgpt: Browser-assisted question-answering with human feedback. _arXiv preprint arXiv:2112.09332_ , 2021. 

- Jianmo Ni, Chen Qu, Jing Lu, Zhuyun Dai, Gustavo Hernandez Abrego, Ji Ma, Vincent Zhao, Yi Luan, Keith Hall, Ming-Wei Chang, and Yinfei Yang. Large dual encoders are generalizable retrievers. In Yoav Goldberg, Zornitsa Kozareva, and Yue Zhang (eds.), _Proceedings of the 2022 Conference on Empirical Methods in Natural Language Processing_ , pp. 9844–9855, Abu Dhabi, United Arab Emirates, December 2022. Association for Computational Linguistics. doi: 10.18653/v1/2022.emnlp-main.669. URL https://aclanthology.org/2022. emnlp-main.669. 

- OpenAI. Chatgpt, 2023. URL https://openai.com/index/chatgpt/. Accessed: 202409-01. 

- OpenAI, Josh Achiam, Steven Adler, Sandhini Agarwal, Lama Ahmad, Ilge Akkaya, Florencia Leoni Aleman, Diogo Almeida, Janko Altenschmidt, Sam Altman, Shyamal Anadkat, Red Avila, Igor Babuschkin, Suchir Balaji, Valerie Balcom, Paul Baltescu, Haiming Bao, Mohammad Bavarian, Jeff Belgum, Irwan Bello, Jake Berdine, Gabriel Bernadett-Shapiro, Christopher Berner, Lenny Bogdonoff, Oleg Boiko, Madelaine Boyd, Anna-Luisa Brakman, Greg Brockman, Tim Brooks, Miles Brundage, Kevin Button, Trevor Cai, Rosie Campbell, et al. Gpt-4 technical report, 2024. URL https://arxiv.org/abs/2303.08774. 

- Long Ouyang, Jeff Wu, Xu Jiang, Diogo Almeida, Carroll L. Wainwright, Pamela Mishkin, Chong Zhang, Sandhini Agarwal, Katarina Slama, Alex Ray, John Schulman, Jacob Hilton, Fraser Kelton, Luke Miller, Maddie Simens, Amanda Askell, Peter Welinder, Paul Christiano, Jan Leike, and Ryan Lowe. Training language models to follow instructions with human feedback, 2022. URL https://arxiv.org/abs/2203.02155. 

- Aleksandra Piktus, Fabio Petroni, Vladimir Karpukhin, Dmytro Okhonko, Samuel Broscheit, Gautier Izacard, Patrick Lewis, Barlas O˘guz, Edouard Grave, Wen-tau Yih, et al. The web is your oyster-knowledge-intensive nlp against a very large web corpus. _arXiv preprint arXiv:2112.09924_ , 2021. 

- Ori Press, Andreas Hochlehnert, Ameya Prabhu, Vishaal Udandarao, Ofir Press, and Matthias Bethge. Citeme: Can language models accurately cite scientific claims?, 2024. URL https: //arxiv.org/abs/2407.12861. 

- Rafael Rafailov, Archit Sharma, Eric Mitchell, Stefano Ermon, Christopher D. Manning, and Chelsea Finn. Direct preference optimization: Your language model is secretly a reward model, 2024a. URL https://arxiv.org/abs/2305.18290. 

- Rafael Rafailov, Archit Sharma, Eric Mitchell, Christopher D Manning, Stefano Ermon, and Chelsea Finn. Direct preference optimization: Your language model is secretly a reward model. _Advances in Neural Information Processing Systems_ , 36, 2024b. 

- Colin Raffel, Noam Shazeer, Adam Roberts, Katherine Lee, Sharan Narang, Michael Matena, Yanqi Zhou, Wei Li, and Peter J Liu. Exploring the limits of transfer learning with a unified text-to-text transformer. _Journal of machine learning research_ , 21(140):1–67, 2020. 

- Hannah Rashkin, Vitaly Nikolaev, Matthew Lamm, Lora Aroyo, Michael Collins, Dipanjan Das, Slav Petrov, Gaurav Singh Tomar, Iulia Turc, and David Reitter. Measuring attribution in natural language generation models, 2022. URL https://arxiv.org/abs/2112.12870. 

- John Schulman, Filip Wolski, Prafulla Dhariwal, Alec Radford, and Oleg Klimov. Proximal policy optimization algorithms, 2017. URL https://arxiv.org/abs/1707.06347. 

14 

Under review 

- Freda Shi, Xinyun Chen, Kanishka Misra, Nathan Scales, David Dohan, Ed H. Chi, Nathanael Sch¨arli, and Denny Zhou. Large language models can be easily distracted by irrelevant context. In Andreas Krause, Emma Brunskill, Kyunghyun Cho, Barbara Engelhardt, Sivan Sabato, and Jonathan Scarlett (eds.), _Proceedings of the 40th International Conference on Machine Learning_ , volume 202 of _Proceedings of Machine Learning Research_ , pp. 31210–31227. PMLR, 23–29 Jul 2023. URL https://proceedings.mlr.press/v202/shi23a.html. 

- Kurt Shuster, Spencer Poff, Moya Chen, Douwe Kiela, and Jason Weston. Retrieval augmentation reduces hallucination in conversation. _arXiv preprint arXiv:2104.07567_ , 2021. 

- Aviv Slobodkin, Eran Hirsch, Arie Cattan, Tal Schuster, and Ido Dagan. Attribute first, then generate: Locally-attributable grounded text generation, 2024. 

- Yixiao Song, Yekyung Kim, and Mohit Iyyer. Veriscore: Evaluating the factuality of verifiable claims in long-form text generation, 2024. URL https://arxiv.org/abs/2406.19276. 

- Ivan Stelmakh, Yi Luan, Bhuwan Dhingra, and Ming-Wei Chang. Asqa: Factoid questions meet long-form answers, 2023. URL https://arxiv.org/abs/2204.06092. 

- Nandan Thakur, Luiz Bonifacio, Xinyu Zhang, Odunayo Ogundepo, Ehsan Kamalloo, David Alfonso-Hermelo, Xiaoguang Li, Qun Liu, Boxing Chen, Mehdi Rezagholizadeh, and Jimmy Lin. Nomiracl: Knowing when you don’t know for robust multilingual retrieval-augmented generation, 2024. URL https://arxiv.org/abs/2312.11361. 

- Hugo Touvron, Louis Martin, Kevin Stone, Peter Albert, Amjad Almahairi, Yasmine Babaei, Nikolay Bashlykov, Soumya Batra, Prajjwal Bhargava, Shruti Bhosale, Dan Bikel, Lukas Blecher, Cristian Canton Ferrer, Moya Chen, Guillem Cucurull, David Esiobu, Jude Fernandes, Jeremy Fu, Wenyin Fu, Brian Fuller, Cynthia Gao, Vedanuj Goswami, Naman Goyal, Anthony Hartshorn, Saghar Hosseini, Rui Hou, Hakan Inan, Marcin Kardas, Viktor Kerkez, Madian Khabsa, Isabel Kloumann, Artem Korenev, Punit Singh Koura, Marie-Anne Lachaux, Thibaut Lavril, Jenya Lee, Diana Liskovich, Yinghai Lu, Yuning Mao, Xavier Martinet, Todor Mihaylov, Pushkar Mishra, Igor Molybog, Yixin Nie, Andrew Poulton, Jeremy Reizenstein, Rashi Rungta, Kalyan Saladi, Alan Schelten, Ruan Silva, Eric Michael Smith, Ranjan Subramanian, Xiaoqing Ellen Tan, Binh Tang, Ross Taylor, Adina Williams, Jian Xiang Kuan, Puxin Xu, Zheng Yan, Iliyan Zarov, Yuchen Zhang, Angela Fan, Melanie Kambadur, Sharan Narang, Aurelien Rodriguez, Robert Stojnic, Sergey Edunov, and Thomas Scialom. Llama 2: Open foundation and fine-tuned chat models, 2023. URL https://arxiv.org/abs/2307.09288. 

- Sirui Xia, Xintao Wang, Jiaqing Liang, Yifei Zhang, Weikang Zhou, Jiaji Deng, Fei Yu, and Yanghua Xiao. Ground every sentence: Improving retrieval-augmented llms with interleaved referenceclaim generation, 2024. URL https://arxiv.org/abs/2407.01796. 

- Fangyuan Xu, Weijia Shi, and Eunsol Choi. Recomp: Improving retrieval-augmented lms with compression and selective augmentation, 2023. URL https://arxiv.org/abs/2310. 04408. 

- Yilong Xu, Jinhua Gao, Xiaoming Yu, Baolong Bi, Huawei Shen, and Xueqi Cheng. Aliice: Evaluating positional fine-grained citation generation, 2024. URL https://arxiv.org/abs/ 2406.13375. 

- Xi Ye, Ruoxi Sun, Sercan O. Arik, and Tomas Pfister.[¨] Effective large language model adaptation for improved grounding and citation generation, 2024. 

- Ori Yoran, Tomer Wolfson, Ori Ram, and Jonathan Berant. Making retrieval-augmented language models robust to irrelevant context, 2024. URL https://arxiv.org/abs/2310.01558. 

- Jingyu Zhang, Marc Marone, Tianjian Li, Benjamin Van Durme, and Daniel Khashabi. Verifiable by design: Aligning language models to quote from pre-training data, 2024a. 

- Weijia Zhang, Mohammad Aliannejadi, Yifei Yuan, Jiahuan Pei, Jia-Hong Huang, and Evangelos Kanoulas. Towards fine-grained citation evaluation in generated text: A comparative analysis of faithfulness metrics, 2024b. URL https://arxiv.org/abs/2406.15264. 

- Xin Zheng, Zhirui Zhang, Junliang Guo, Shujian Huang, Boxing Chen, Weihua Luo, and Jiajun Chen. Adaptive nearest neighbor machine translation. _arXiv preprint arXiv:2105.13022_ , 2021. 

15 

Under review 

## A NUANCES OF ANSWERABILITY. 

Determining answerability can be challenging. To determine answerability, we use a system that evaluates the entailment of gold claims against provided documents, referred to as the Natural Language Inference (NLI) system. An NLI system can range from a simple exact match (EM) identifier to an LLM or even a human evaluator, with answerability determined based on _q, D_ and biases of the NLI[9] . These biases can be useful in specific RAG applications, such as solving mathematical problems where the documents provide a formula and the question assigns values to variables. The choice of NLI depends on whether the RAG system requires the LLM to have mathematical understanding. **Ideally, to prevent improper evaluations, the NLI model used to construct the gold claims should also be used to evaluate the LLM responses.** 

In this paper, our focus is on evaluating the generic comprehension capabilities of LLMs without specialized knowledge. Thus, we use two NLI mechanisms: 1) identifying whether an exact match of claims is present in the gold claims, and 2) using a Machine Learning (ML) model to determine if the documents can entail the gold claims. The ML-based NLI model is used for multiple purposes, such as alignment dataset construction (data/training) and evaluating generated responses (metric/testing). For this, we adopt the NLI model from Rashkin et al. (2022). _ϕ_ ( _cij, si_ ) = 1 if _cij_ (premise) entails _si_ (hypothesis); otherwise, 0. To determine answerability, we employ the TRUEbased method (Honovich et al., 2022) to assess whether a gold claim can be entailed by a given document. 

**The knowledge grounding problem.** Typically, LLMs are designed to perform questionanswering tasks, where response generation heavily relies on the parametric (internal) knowledge acquired during their pre-training, tuning, and alignment phases (OpenAI, 2023; Anthropic, 2024). Thus, most of their knowledge is grounded in parametric memory. This makes them inherently less suitable for RAG applications, where the knowledge generated by the LLM is expected to be grounded in input documents. RAG is analogous to a reading comprehension task, where the answers must come from the provided passage (documents in RAG) rather than the prior knowledge of the person taking the test. Thus, any reliance on parametric knowledge can result in statements that are not fully grounded in the documents, including providing answers to unanswerable questions. _Our investigation shows that state-of-the-art models, such as GPT-4 and Claude-3.5-Sonnet, overtly rely on parametric knowledge even when used in a RAG setting._[10] 

## B ANSWERABILITY: A CASE STUDY 

Prior works (Liu et al., 2023; Gao et al., 2023b; Ye et al., 2024; Huang et al., 2024a; Li et al., 2024a) have employed substring matching to indicate entailment. While this syntactic approach is fast, it often proves inadequate in complex, long contexts. A case study is presented in Table 7. To address the limitations of this superficial entailment, we adopt a TRUE-based method (Honovich et al., 2022), which combines the strengths of both syntactic and semantic approaches. Specifically, we enhance the process by using the TRUE model, a T5-11B model (Raffel et al., 2020) fine-tuned for the NLI task, to verify, from a semantic perspective, whether a substring match corresponds to meaningful entailment within document passages. The input to the TRUE model is the concatenation of a premise and a hypothesis, and the output is an entailment score between 0 and 1, indicating the degree to which the premise entails the hypothesis. We treat the corresponding documents as the premise, and to minimize ambiguity, the associated question is concatenated with each gold answer as the hypothesis. In cases where the TRUE model does not yield a positive entailment score despite a substring match, we rely on the TRUE judgment as the final label. However, if the substring match fails, we bypass TRUE calculation, thus reducing the computational cost of relying solely on TRUE for semantic entailment. 

> 9For EM, the bias is that a _q_ is answerable if an exact match for claims is present in _D_ . 10We show a detailed analysis in Appendices F.2 and F.3. 

16 

Under review 

|**Question**|How manystateparks are there in Virginia?|
|---|---|
|**Gold Answer**|38|
|**Retrieved document**|Virginia has 30 National Park Service units, such as Great Falls Park and|
||the Appalachian Trail, and one national park, the Shenandoah National|
||Park. With over 500 miles of trails, including **38** miles of the iconic|
||Appalachian Trail, it’s a paradise for hikers, nature lovers, and those|
||seekingserene mountain landscapes.|
|**Substring match**|Substringis matched and as such thequestion is answerable.|
|**TRUE Judgement**|Not entailed as such thequestion is unanswerablegiven the document.|



Table 7: Case study showcasing the limitations of substring matching and necessity of TRUE judgement. 

## C RELATED WORKS 

## C.1 ATTRIBUTABLE RETRIEVAL AUGMENTED GENERATION 

Retrieval Augmented Generation (RAG) has been widely studied for reducing the knowledge gap and providing more referenced information to enhance answer generation (Karpukhin et al., 2020; Lewis et al., 2021; Gao et al., 2023c). However, LLMs are prone to being misled by irrelevant information, leading to hallucinations and less factual outputs (Shi et al., 2023; Yoran et al., 2024; Xu et al., 2023). This challenge has spurred research into attributable RAG, which aims to verify model outputs by identifying supporting sources. Rashkin et al. (2022) first introduced the concept of Attributable to Identified Sources (AIS) to evaluate attribution abilities. Subsequently, Gao et al. (2023b) adapted this approach to verify generated content with citations, improving the reliability of RAG systems. Simultaneously, Press et al. (2024) and Song et al. (2024) explored related aspects: citation attribution for paper identification and the verifiability of long-form generated text, respectively. Further fine-grained evaluations have been examined, such as assessing the degree of support (Zhang et al., 2024b) and the granularity of claims (Xu et al., 2024). Recent studies (Buchmann et al., 2024; Hsu et al., 2024) have also investigated attribution ability by disentangling the confounding effects of retrievers and LLMs. Unlike existing work, we prioritize trustworthiness in LLMs, ensuring that generated responses are derived solely from the provided documents and refrain from producing unverifiable content. 

## C.2 ENHANCE GROUNDED TEXT GENERATION IN ATTRIBUTED LARGE LANGUAGE MODELS 

To enhance grounded text generation, various attributed LLMs have been proposed, falling into two main paradigms: training-free and training-based. For training-free methods: 1) In-context learning (Gao et al., 2023b) is used to generate in-line citations with few-shot demonstrations. 2) Post-hoc attribution (Gao et al., 2023a; Li et al., 2024b) first generates an initial response and then retrieves evidence as attribution. 3) Ji et al. (2024) demonstrate that using chain-of-thought reasoning improves the quality of text generated with citations. For training-based methods: 1) Asai et al. (2024); Slobodkin et al. (2024); Xia et al. (2024); Ye et al. (2024) apply supervised fine-tuning (SFT) to LLMs, training them to identify useful information from documents and guide cited text generation with them. 2) Beyond simple SFT, recent studies model the task as preference learning, employing Reinforcement Learning with Human Feedback (RLHF) (Ouyang et al., 2022) and Direct Preference Optimization (DPO) (Rafailov et al., 2024a). Huang et al. (2024a) proposed a method to improve attribution generation using fine-grained rewards and Proximal Policy Optimization (PPO) (Schulman et al., 2017), while Li et al. (2024a); Huang et al. (2024b) introduced the modified DPO framework to enhance fine-grained attribution abilities. 3) While many approaches rely on external documents provided by the user or retrieved during generation, Khalifa et al. (2024); Zhang et al. (2024a) focus on tuning LLMs to cite sources from pre-training data using learned parametric knowledge. 

## D METRICS 

In this section, we elaborate on how we compute metrics that are components of TRUST-SCORE. 

17 

Under review 

## D.1 RESPONSE TRUTHFULNESS 

Truthfulness captures the model’s ability to answer or refuse a question correctly by computing the grounded refusal (F1RG) and the factual accuracy by computing the answer-calibrated exact match score (EMAC). 

**Grounded Refusal [F1RG]:** A macro-averaged F1 score that measures the LLM’s ability in correctly refusing to answer a question (F1ref) and correctly providing an answer when required (F1ans). 

- **F1ref** : This metric evaluates a model’s ability to correctly refuse unanswerable questions. We calculate it based on how accurately the model identifies and refuses these questions. Let _Ag_ and _¬Ag_ represent the sets of ground truth answerable and unanswerable questions, respectively, and _Ar_ and _¬Ar_ denote the sets of questions where the model provided an answer and refused to answer, respectively. F1ref is computed from precision Pref and recall Rref: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/rag/iclr_2025_Measuring_and_Enhancing_Trustworthiness_of_LLMs_in_RAG_through_Grounded_Attributions_and_Learning_to_Refuse_images/iclr_2025_Measuring_and_Enhancing_Trustworthiness_of_LLMs_in_RAG_through_Grounded_Attributions_and_Learning_to_Refuse.pdf-0018-05.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/rag/iclr_2025_Measuring_and_Enhancing_Trustworthiness_of_LLMs_in_RAG_through_Grounded_Attributions_and_Learning_to_Refuse_images/iclr_2025_Measuring_and_Enhancing_Trustworthiness_of_LLMs_in_RAG_through_Grounded_Attributions_and_Learning_to_Refuse.pdf-0018-06.png)


where Pref measures the proportion of correctly refused unanswerable questions among all refused questions, and Rref measures the proportion of correctly refused unanswerable questions out of all unanswerable questions. Here, _|·|_ denote the cardinality of the set, thus Pref, Rref, and F1ref are scalar values. 

- **F1ans** : This metric evaluates a model’s ability to correctly answer answerable questions. It is computed based on the precision Pans and recall Rans for non-refusal responses to answerable questions: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/rag/iclr_2025_Measuring_and_Enhancing_Trustworthiness_of_LLMs_in_RAG_through_Grounded_Attributions_and_Learning_to_Refuse_images/iclr_2025_Measuring_and_Enhancing_Trustworthiness_of_LLMs_in_RAG_through_Grounded_Attributions_and_Learning_to_Refuse.pdf-0018-09.png)


**F1RG (Grounded Refusals)** provides an overall assessment of the model’s refusal capabilities by computing the macro-average of F1ref and F1ans: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/rag/iclr_2025_Measuring_and_Enhancing_Trustworthiness_of_LLMs_in_RAG_through_Grounded_Attributions_and_Learning_to_Refuse_images/iclr_2025_Measuring_and_Enhancing_Trustworthiness_of_LLMs_in_RAG_through_Grounded_Attributions_and_Learning_to_Refuse.pdf-0018-11.png)


F1ref evaluates the model’s ability to correctly refuse unanswerable questions, while F1ans assesses its ability to correctly answer answerable ones. By penalizing both incorrect refusals and incorrect non-refusals, F1RG offers a balanced evaluation of the model’s over-responsiveness and underresponsiveness 

**Exact Match (Answer Calibrated) [EM[F1] AC]:** Given a question _q_ and the corresponding gold claims _AG_ = _{ag_ 1 _, . . . , agn}_ , we define the claims obtainable from the provided documents as _AD_ = _{ad_ 1 _, . . . , adn}_ and the claims generated in the response _r_ as _AR_ = _{ar_ 1 _, . . . , arn}_ . EM _[q]_ AC disregards the claims that cannot be inferred from _D_ (answer calibration), and the exact match recall scores is computed on the remaining claims, i.e., _AG ∩ AD_ : 


![](raw/papers/2_LLM_NLP/LLM_paper_list/rag/iclr_2025_Measuring_and_Enhancing_Trustworthiness_of_LLMs_in_RAG_through_Grounded_Attributions_and_Learning_to_Refuse_images/iclr_2025_Measuring_and_Enhancing_Trustworthiness_of_LLMs_in_RAG_through_Grounded_Attributions_and_Learning_to_Refuse.pdf-0018-14.png)


18 

Under review 

For the whole dataset with multiple questions _{q_ 1 _. . . qk}_ , one can compute the average: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/rag/iclr_2025_Measuring_and_Enhancing_Trustworthiness_of_LLMs_in_RAG_through_Grounded_Attributions_and_Learning_to_Refuse_images/iclr_2025_Measuring_and_Enhancing_Trustworthiness_of_LLMs_in_RAG_through_Grounded_Attributions_and_Learning_to_Refuse.pdf-0019-02.png)


Where _Ag_ denote the set of questions that are answerable using the provided documents, fully or partially; _Ar_ denote the set of questions that are answered by the model (non-refusal). There are two variants of EMAC we study— first **EM** _[α]_ **AC**[with][denominator] _[k]_[=] _[|][A][r][|]_[(number][of][answered] questions). Second variant **EM** _[β]_ **AC**[with][denominator] _[k]_[=] _[|][A][g][|]_[(number][of][answerable][questions).] Here _| · |_ denotes the cardinality of the set. We denote the aggregated score by 


![](raw/papers/2_LLM_NLP/LLM_paper_list/rag/iclr_2025_Measuring_and_Enhancing_Trustworthiness_of_LLMs_in_RAG_through_Grounded_Attributions_and_Learning_to_Refuse_images/iclr_2025_Measuring_and_Enhancing_Trustworthiness_of_LLMs_in_RAG_through_Grounded_Attributions_and_Learning_to_Refuse.pdf-0019-04.png)


The primary reason for adjusting the conventional Exact Match (EM) metric to account for the presence of answers in retrieved documents is to avoid rewarding models for generating correct answers without locating them in the provided documents. This approach discourages models from relying solely on their pre-trained knowledge to answer questions, instead encouraging them to find and ground their answers within the provided documents. 

## D.2 ATTRIBUTION GROUNDEDNESS 

Attribution or citation groundedness measures the relevance of generated citations to their corresponding statements, both individually and collectively. A citation _ci,j_ is deemed ” _relevant_ ” when the statement it cites can be inferred from the cited document. The collective importance of citations is assessed using a statement-wise recall metric, while the individual importance of each citation is evaluated using a precision metric. Given that a generated response _r_ consists of multiple statements _S_ and their corresponding citations _C_ , we first compute statement-wise citation recall and per-citation precision. These scores are then averaged to obtain sample-wise scores, which are finally averaged to produce dataset-wide scores. 

**Citation Grounded F1 [F1CG]:** For a given statement _si_ , statement-wise citation recall is computed by: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/rag/iclr_2025_Measuring_and_Enhancing_Trustworthiness_of_LLMs_in_RAG_through_Grounded_Attributions_and_Learning_to_Refuse_images/iclr_2025_Measuring_and_Enhancing_Trustworthiness_of_LLMs_in_RAG_through_Grounded_Attributions_and_Learning_to_Refuse.pdf-0019-09.png)


where _ϕ_ ( _{ci,_ 1 _, . . . , ci,j}, si_ ) _→{_ 0 _,_ 1 _}_ is a function that determines whether the concatenation of all cited documents fully supports the statement _si_ . Next, we compute precision for a generated citation _ci,j_ for statement _si_ as: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/rag/iclr_2025_Measuring_and_Enhancing_Trustworthiness_of_LLMs_in_RAG_through_Grounded_Attributions_and_Learning_to_Refuse_images/iclr_2025_Measuring_and_Enhancing_Trustworthiness_of_LLMs_in_RAG_through_Grounded_Attributions_and_Learning_to_Refuse.pdf-0019-11.png)


Thus, citation precision is 0 if and only if the cited document _ci,j_ does not entail the statement _si_ , while all other citations collectively entail _si_ without _ci,j_ . 

As an aggregate measure, we report **F1CG** , which computes the F1 score using cumulative precision and recall over the answered questions only (non-refusals): 


![](raw/papers/2_LLM_NLP/LLM_paper_list/rag/iclr_2025_Measuring_and_Enhancing_Trustworthiness_of_LLMs_in_RAG_through_Grounded_Attributions_and_Learning_to_Refuse_images/iclr_2025_Measuring_and_Enhancing_Trustworthiness_of_LLMs_in_RAG_through_Grounded_Attributions_and_Learning_to_Refuse.pdf-0019-14.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/rag/iclr_2025_Measuring_and_Enhancing_Trustworthiness_of_LLMs_in_RAG_through_Grounded_Attributions_and_Learning_to_Refuse_images/iclr_2025_Measuring_and_Enhancing_Trustworthiness_of_LLMs_in_RAG_through_Grounded_Attributions_and_Learning_to_Refuse.pdf-0019-15.png)


Where _Ar_ denotes the number of samples answered by the model, _S_ denotes the set of statements in a generated response, and _A[s] r_[denotes][the][set][of][responses][(including][only][statements,][ignoring] citations) in the dataset. Similarly, _C_ denotes the set of citations in a generated response, and _A[c] r_ denotes the set of responses (including only citations, ignoring statements) in the dataset. 

19 

Under review 

**TRUST-SCORE:** Finally, we combine the metrics to produce a single trustworthiness score, which allows us to rank models based on their trustworthiness. This score is calculated as the average of each component metric. 


![](raw/papers/2_LLM_NLP/LLM_paper_list/rag/iclr_2025_Measuring_and_Enhancing_Trustworthiness_of_LLMs_in_RAG_through_Grounded_Attributions_and_Learning_to_Refuse_images/iclr_2025_Measuring_and_Enhancing_Trustworthiness_of_LLMs_in_RAG_through_Grounded_Attributions_and_Learning_to_Refuse.pdf-0020-02.png)


## E THE TRUST-ALIGN DATASET 

To align LLMs towards trustworthiness, we propose a new approach, **TRUST-ALIGN** . The approach constructs an LLM trustworthiness alignment dataset, where each sample in the dataset consists of a question _q_ , a set of retrieved documents _D_ , and a pair of positive (preferred) and negative (unpreferred) responses ( _r_[+] , _r[−]_ ). The positive response corresponds to an answer that encompasses expected gold claims for _q_ and corresponding citations referring to the documents. If _D_ is not sufficient to answer _q_ , _r_[+] is assigned a refusal response, while _r[−]_ is its non-refusal counterpart. We build the dataset in multiple steps: 1) Obtain a set of high quality and diverse questions, 2) Obtain documents for each question, 3) Augmenting ( _q, D_ ) pairs that cover diverse hallucination types, 4) Construct positive responses entailing gold claims, and 5) Construct negative (unpreferred) responses by prompting a fine-tuned model and observing its hallucinations. 

## E.1 COLLECTING QUALITY QUESTIONS. 

The dataset construction begins by collecting a set of high-quality (challenging) and diverse questions from source datasets i.e. ASQA, QAMPARI, and ELI5—referred to as **seed samples** . To collect such samples, we first divide the questions in a dataset into _k_ clusters using a Hugginface pipeline[11] . After identifying the diverse clusters, we use Mixtral-8x7B with the prompt described in Table 22 to assign each a quality score ranging from 1 to 7. The quality of a cluster is determined by how difficult it is to answer the questions without requiring additional information i.e. a higher score corresponds to a high difficulty. We then select clusters with a quality score of 4 or higher and sample the desired number of questions from these top clusters. Suppose we have three clusters, _C_ 1 _, C_ 2 _, C_ 3, with respective sizes _N_ 1 _, N_ 2 _, N_ 3, where _Nc_ = _N_ 1 + _N_ 2 + _N_ 3. To sample _Ns_ questions from the clusters, we sample _Ns × N[C][i] c_[questions from cluster] _[ C][i]_[.][If this number exceeds the] available questions in the cluster, we randomly sample the remaining questions from the filtered-out clusters (those with a quality score below 4). This process ensures that the seed set prioritizes both high quality and diversity. For this paper, we set _Ns_ to 3K, 3K, and 4K for ASQA, QAMPARI, and ELI5, respectively, resulting in approximately 10K questions in the seed set. 

## E.2 COLLECTING _D_ ’S. 

For each seed question _q_ that is obtained from ASQA and QAMPARI, we used gtr-t5-xxl (Ni et al., 2022) to retrieve the top 100 relevant documents _D_ from the 2018-12-20 Wikipedia snapshot. For the ELI5 dataset, we employed BM25 in conjunction with Sphere (Piktus et al., 2021), a filtered version of Common Crawl, as it better encompasses the wide range of topics present in ELI5. We filter seed question for which the retriever fails to retrieve relevant documents. 

We utilize TRUE-NLI to derive the entailment pattern for each document. This pattern represents the set of gold claims that the document supports. The TRUE model takes as input a concatenation of a premise and a hypothesis, producing an entailment score (0 or 1) that indicates whether the premise entails the hypothesis. In our approach, the documents serve as the premise, while the hypothesis is formed by combining the relevant question with each corresponding gold claim to reduce ambiguity. We take the union of the entailment patterns across documents to assess the answerability of each question—if the pattern contains at least one supporting claim, the question is considered answerable. 

Following Gao et al. (2023b), we identify 5 documents that are equally effective for the model as the 100 documents in terms of achieving the EM recall value); we refer to such documents as _**oracle**_ documents for question _q_ . Notably, to compute EM, gold claims are obtained from respective source datasets. 

> 11https://github.com/huggingface/text-clustering/ 

20 

Under review 

## E.3 AUGMENTING _**(q,D)**_ SET. 

Now that we have the questions and the most relevant (oracle) documents, our goal is to create samples of diverse types (i.e., different proportions of relevant documents for the same question) that can trigger multiple hallucinations from LLMs (Section 2.3). As illustrated in Fig. 3, for answerable questions, we first utilize the identified entailment patterns to generate all possible combinations of documents, then select _k_ combinations that cover diverse patterns. To create samples with unanswerable questions, we select documents that are similar to gold-claim-entailing documents but do not entail any gold claims. To minimize the risk of introducing bias in citation indices, we shuffle the order of documents in each sample. As a result, we generate approximately 70K question-document pairs. 

After obtaining ( _q, D_ ) pairs for the alignment dataset, we obtain positive and negative responses ( _r_[+] _, r[−]_ ) for each pair—an essential component of the dataset signaling the model’s preferred and unpreferred responses. To achieve this, we introduce a response generation pipeline. 

## E.4 OBTAINING **r**[+] . 

We develop an automated data labeling pipeline that synthesizes natural responses from gold claims and maps each statement to the corresponding documents for embedded in-line citations. The gold claims are obtained from the source datasets (ASQA, QAMPARI, ELI5) and calibrated to the provided documents, i.e., filtering out claims that cannot be derived from _D_ . We first split the questions into answerable and unanswerable samples based on whether the provided documents entail the gold claims. For an answerable sample, consisting of a question _q_ , a set of documents _D_ , and a list of (calibrated) gold claims, we prompt GPT-4 to generate a natural response by stitching together the gold claims using a template (Table 21). Please refer to the subsection below for more details on how the prompt is structured for each dataset. The prompt template asks GPT-4 to label each gold claim used with its index from the provided list (e.g., ”[Gold Claim X]”), allowing for later matching of claims to documents. For unanswerable questions, a refusal response is assigned. To generate citations corresponding to each statement generated, we map the ”[Gold Claim X]” labels to the appropriate documents. First, we extract all such labels from a sentence (which may contain multiple claims and labels). Then, we greedily identify the smallest combination of documents that covers these claims, minimizing over-citation. Details of this process is illustrated in Fig. 4. 

**Details on prompt structure for each dataset** For ASQA, we include the question _q_ , a list of (calibrated) gold claims, and their corresponding supporting documents _D_ as additional context. For ELI5, we follow Gao et al. (2023b) by decomposing each labeled response into three claims, which serve as a set of ground truth answers. Since the claim labels already provide sufficient context, we only fit the question and calibrated claims into the template. For QAMPARI, since its response format aligns with its labeled ground truth format (a list of entities), no additional action is required. 


![](raw/papers/2_LLM_NLP/LLM_paper_list/rag/iclr_2025_Measuring_and_Enhancing_Trustworthiness_of_LLMs_in_RAG_through_Grounded_Attributions_and_Learning_to_Refuse_images/iclr_2025_Measuring_and_Enhancing_Trustworthiness_of_LLMs_in_RAG_through_Grounded_Attributions_and_Learning_to_Refuse.pdf-0021-07.png)


**----- Start of picture text -----**<br>
Document Recombination<br>Valid DocumentsDoc 1 Doc 4 Get all entailmentpatterns [1,0,0] Doc 1 [1,0,0] Doc 1Doc 2 [1,0,0] Doc 2 covering all entailmentSelect k combinations<br>[1,0,0] [1,0,1] [1,0,1] [1,0,0] patterns<br>Doc 2 Doc 3 [1,0,0] [0,0,1] Doc 3<br>[1,0,0] [0,0,1]<br>[0,0,1] Doc 1 Doc 2 Doc 4<br>[1,0,0] Doc 3 [1,0,0] Doc 3 [1,0,1]<br>Invalid Documents [0,0,1] [0,0,1]<br>Doc 5 Doc 7<br>[0,0,0] [0,0,0]<br>Doc 6 [1,0,0] Doc 1 Doc 81 Doc 66 Doc 53 Doc 44<br>[0,0,0] ...<br>Pool of 96 invaliddocuments For each group, fill remainingplaces with invalid docs [0,0,1] Doc 3 Doc 11 Doc 6 Doc 97 Doc 77<br>most similarFind docs most similar to the existingvalid docs  [1,0,0] Doc 1 [0,0,1] Doc 3 Doc 86 Doc 32 Doc 27<br>Pool of 50 invalidto question [1,0,0] Doc 2 Doc 5 Doc 91 Doc 70 Doc 75<br>documents<br>**----- End of picture text -----**<br>


Figure 3: Document recombination process in augmented prompt curation. 

21 

Under review 


![](raw/papers/2_LLM_NLP/LLM_paper_list/rag/iclr_2025_Measuring_and_Enhancing_Trustworthiness_of_LLMs_in_RAG_through_Grounded_Attributions_and_Learning_to_Refuse_images/iclr_2025_Measuring_and_Enhancing_Trustworthiness_of_LLMs_in_RAG_through_Grounded_Attributions_and_Learning_to_Refuse.pdf-0022-01.png)


**----- Start of picture text -----**<br>
Claim-document mapping<br>[1,0,1] Doc 1 [1,0,0] Doc 4 Union Get Mapper<br>[1,0,1] Doc 2 [0,0,0] Doc 5 [1,0,1] {'1':0,'2':2}<br>Doc 3<br>[1,0,0]<br>[Gold Claim 1] Ove Johansson Calibration<br>[1,0,1] [Gold Claim 1] Ove Johansson ...  Matt Prater  at 64 yards<br>[Gold Claim 2] Ronaldo Cristiano [Gold Claim 2] . ...  Ove Johansson<br>[Gold Claim 3] Matt Prater Transformedclaim idx [Gold Claim 2] Matt Prater synthesizerGPT-4 in a 1976...  [Gold Claim 1] Sentencetokenizer.<br>Matt Prater yards  [1]  at 64 [1,0,1] Doc 1 Check doc combinations cancover minimally in greedyfashion ans_idx = {2} {'1':0,'2':2} claim idxoriginalMap to [Gold Claim 2][2] claim labelExtract Matt Prater[Gold Claim 2] yards at 64<br>**----- End of picture text -----**<br>


Figure 4: Claim-document-mapping process. 

## E.5 OBTAINING **r** _[−]_ . 

To create high-quality preference data, we aim to obtain quality negative (unpreferred) responses. We first fine-tune LLaMA-2-7b on the training set of the source datasets[12] , creating _Msft_ . We then test _Msft_ on the above-obtained dataset with approximately 70K questions and identify that 40K responses exhibit hallucinations. Table 8 shows the severity computation ( _ei_ ) and the frequency of each hallucination type ( _wi_ ). Thus, we can compute hallucination severity for each sample as 

|**Hallucination type**|**Frequency (**_wi_**)**|**Frequency (**_wi_**)**|**Severity (**_ei_**)**|
|---|---|---|---|
|Unwarranted Refusal|8,786|0.50|_I_(_Ag_=_∅,Ar_=_∅_)|
|Over Responsiveness|13,067|0.50|_I_(_Ag_=_∅,Ar_=_∅_)|
|Overcitation|12,656|0.34|1 - CP|
|Improper Citation|9,592|0.26|1 - CR|
|Inaccurate Claims|14,783|0.40|1 - EMF1<br>AC|



Table 8: Fraction of each hallucination amongst all the observed hallucinations in _Msft_ (40,985), with possible overlap. _wi_ shows the severity computation of each hallucination. _I_ condition = 1 if condition is True otherwise it is 0. See Fig. 5 for the detailed breakdown of the last three errors. 

To obtain good negative samples, we first rank each of the 40K responses according to their severity score _eq_ . We then select the top 50% of the corresponding samples for both answerable and unanswerable responses. **Thus, we demonstrate the alignment data construction phase of TRUSTALIGN, i.e., obtaining 19K samples with all the desired attributes** ( **q** _,_ **D** _,_ **r**[+] _,_ **r** _[−]_ ) **.** We perform DPO using this set of 19k samples to obtain the final aligned model. 


![](raw/papers/2_LLM_NLP/LLM_paper_list/rag/iclr_2025_Measuring_and_Enhancing_Trustworthiness_of_LLMs_in_RAG_through_Grounded_Attributions_and_Learning_to_Refuse_images/iclr_2025_Measuring_and_Enhancing_Trustworthiness_of_LLMs_in_RAG_through_Grounded_Attributions_and_Learning_to_Refuse.pdf-0022-08.png)


**----- Start of picture text -----**<br>
Overlap of Inaccurate Answer, Improper Citation, Overcitation Errors in Augmented Set<br> (# of hallucinations in these three categories = 19132, # Over Responsiveness hallucinations = 13067,<br> # Excessive Refusal hallucinations = 8786, # of samples with no error: 28518, Total # of samples = 69503)<br>Inaccurate Answer Improper Citation<br>65<br>6384 27<br>6636<br>1698 2864<br>1458<br>Overcitation<br>**----- End of picture text -----**<br>


Figure 5: Statistics of hallucinations from the output of LLaMA-2-7b SFT model prompted using 70K ( _q, D_ ) samples obtained in Step-2 of TRUST-ALIGN. 

> 12 + Seed questions, corresponding oracle documents, and the gold answers ( _r_ ). 

22 

Under review 

## F ADDITIONAL ANALYSIS 

## F.1 REVISED METRICS ARE LESS BIASED 


![](raw/papers/2_LLM_NLP/LLM_paper_list/rag/iclr_2025_Measuring_and_Enhancing_Trustworthiness_of_LLMs_in_RAG_through_Grounded_Attributions_and_Learning_to_Refuse_images/iclr_2025_Measuring_and_Enhancing_Trustworthiness_of_LLMs_in_RAG_through_Grounded_Attributions_and_Learning_to_Refuse.pdf-0023-03.png)


**----- Start of picture text -----**<br>
DPO<br>Baseline<br>60<br>50<br>40<br>30<br>20<br>10<br>0<br>L-2-7b L-2-7b L-3-8b L-3-8b L-3.2-1b L-3.2-1b L-3.2-3b L-3.2-3b P-3.5- P-3.5- Q-2.5- Q-2.5- Q-2.5- Q-2.5- Q-2.5-3b Q-2.5-3b Q-2.5-7b Q-2.5-7b<br>EM reg EM calib EM reg EM calib EM reg EM calib EM reg EM calib mini EM mini EM 0.5b EM 0.5b EM 1.5b EM 1.5b EM EM reg EM calib EM reg EM calib<br>reg calib reg calib reg calib<br>Models<br>EM Score<br>**----- End of picture text -----**<br>


Figure 6: Comparison of EM regular and EM[F1] AC[across models on ASQA.] 

Fig. 6 shows the performance of our models on EM regular and EM[F1] AC[.][As seen across the models,] EM regular tends to unduly penalize our models for refusals, resulting in baselines performing disproportionately better than our models. By measuring EM on both the answerable and answered set to arrive at EM[F1] AC[,][we][arrive][at][an][EM][metric][that][is][more][fair][in][the][presence][of][refusals.][By] correcting for the bias toward answering at all costs, we are able to reveal more balanced perspective on model performance as demonstrated by a reduction in the performance gap (eg.. in LLaMA-2-7b, LLaMA-3-8b, LLaMA-3.2-1b, LLaMA-3.2-3b) or even revealing our model’s stronger performance as compared to baseline (e.g. Phi-3.5 mini, Qwen-2.5-1.5b, Qwen2.5-3b). The ability to grade the model more fairly underscores the need for our calibrated metrics. 

## F.2 UTILIZATION OF PARAMETRIC KNOWLEDGE 

For an LLM used for RAG task, it is important to study the tendency of LLM towards grownding its knowledge on the provided documents. To partially quantify this, we compute EM uncalibrated score for questions that are unanswerable by the provided documents; thus _AG ∩ AD_ = _∅_ but _AG_ = _∅_ , 


![](raw/papers/2_LLM_NLP/LLM_paper_list/rag/iclr_2025_Measuring_and_Enhancing_Trustworthiness_of_LLMs_in_RAG_through_Grounded_Attributions_and_Learning_to_Refuse_images/iclr_2025_Measuring_and_Enhancing_Trustworthiness_of_LLMs_in_RAG_through_Grounded_Attributions_and_Learning_to_Refuse.pdf-0023-08.png)


Where, _AG_ , _AD_ , and _AR_ are claims in the ground truth answer, claims present in the documents, and the claims generated in the response, respectively. _Nr_ is the number of answered questions. 

In Table 9, our analysis reveals that responsive models tend to rely on parametric knowledge more frequently. Notably, closed-source models like GPT-4 exhibit higher parametric knowledge usage compared to our models. However, this metric only partially captures the models’ utilization of parametric knowledge. For instance, cases where models correctly generate gold claims without proper grounding may also indicate reliance on parametric knowledge. This phenomenon is evident in Table 11, where on ASQA, GPT-4 achieves a significantly higher **EM[F1] AC**[than our models, yet its] attribution groundedness score **F1CG** is five points lower. 

23 

Under review 

|**Model**|**ASQA**<br>**AR (%)**<br>**Pscore**|**QAMPRARI**<br>**AR (%)**<br>**Pscore**|**ELI5**<br>**AR (%)**<br>**Pscore**|
|---|---|---|---|
|ICL-LLaMA-2 7B<br>ICL-LLaMA-3 8B<br>ICL-GPT-3.5<br>ICL-GPT-4<br>ICL-Claude-3.5|0.00<br>0.00<br>1.48<br>1.79<br>71.20<br>9.74<br>86.81<br>12.71<br>84.60<br>12.99|0.00<br>0.00<br>3.90<br>16.92<br>65.30<br>11.45<br>73.40<br>13.05<br>69.80<br>12.55|0.50<br>0.00<br>0.00<br>0.00<br>49.00<br>7.89<br>61.50<br>9.05<br>59.00<br>1.76|
|DPO-LLaMA-2-7B<br>DPO-LLaMA-3-8B|65.30<br>8.15<br>56.42<br>8.65|31.10<br>8.45<br>23.10<br>8.97|21.60<br>5.56<br>15.50<br>7.26|



Table 9: Detection of parametric knowledge usage under refusal prompting. 

## F.3 THE SOURCE OF LLM HALLUCINATIONS 

Model errors can be categorized into two primary sources: 

1. **Parametric knowledge-based hallucination:** Errors arising from the model’s internal knowledge representation. 

2. **Information extraction failures:** Inability to accurately extract relevant information from provided documents. 

To quantify these error types, we employ the following methodology: 

- For the non-refused questions with errors, calculate the proportion of the incorrect answers that are: 

   - Present in the provided documents 

   - Absent from the provided documents 

For answers absent from the documents, we can attribute the error to parametric knowledge-based hallucination. For answers present in the documents, the specific source of the error remains indeterminate as it can be attributed to both. 

The substring matching (Gao et al., 2023b) is used here for searching for the existence of incorrect answers in the documents. As the model’s response only on QAMPARI can be decomposed into atomic facts, we chose to perform this analysis on it. Specifically, for every answered question, we calculate the proportion of incorrect answers present in or absent from the documents using the equations below: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/rag/iclr_2025_Measuring_and_Enhancing_Trustworthiness_of_LLMs_in_RAG_through_Grounded_Attributions_and_Learning_to_Refuse_images/iclr_2025_Measuring_and_Enhancing_Trustworthiness_of_LLMs_in_RAG_through_Grounded_Attributions_and_Learning_to_Refuse.pdf-0024-13.png)


Where _Ae_ denotes the set of answerable questions that answered by the model with one or more incorrect answers; _AD_ , _A[e] R_[are facts present in the documents and erroneous facts generated in the] response, respectively. 

The findings are presented in Table 10. Our analysis reveals that, with the exception of LLaMA-2 7B which provides no responses, all other ICL-based models exhibit a higher tendency to produce erroneous answers based on their parametric knowledge compared to our models. Notably, Claude-3.5 demonstrates a more frequent reliance on its parametric knowledge, which elucidates its significantly lower TRUST-SCORE score in Table 11. 

In summary, our investigation indicates that baseline models, including GPT-4 and GPT-3.5, are more susceptible to hallucinations stemming from their parametric knowledge. 

## F.4 TRUST-ALIGN ENHANCES TRUSTWORTHINESS MORE ROBUSTLY THAN PROMPTING. 

Aligning with TRUST-ALIGN leads to more significant improvements in TRUST-SCORE compared to using prompting alone. While adding a refusal prompt has inconsistent effects on TRUST-SCORE 

24 

Under review 

|**Model**|**QAMPARI**<br>**Presence (%)**<br>**Absence (%)**|
|---|---|
|ICL-LLaMA-2 7B<br>ICL-LLaMA-3 8B<br>ICL-GPT-3.5<br>ICL-GPT-4<br>ICL-Claude-3.5|0.00<br>0.00<br>84.41<br>15.59<br>85.04<br>14.96<br>89.3<br>10.7<br>72.18<br>27.82|
|DPO-LLaMA-2-7B<br>DPO-LLaMA-3-8B|93.26<br>6.74<br>95.63<br>4.37|



Table 10: The proportions of erroneous answers present in or absent from the documents. 

and its subcomponents, it tends to be more beneficial in more capable models, such as LLaMA-213b and LLaMA-3-8b. 

Relying solely on prompting to teach refusal is ineffective, as models’ responsiveness becomes overly sensitive to the prompt. Under the default prompt, models rarely refuse (AR% close to 100), while adding a refusal prompt in ICL drastically reduces AR%, often to near zero, indicating indiscriminate refusal. This lack of nuanced refusal ability is also seen in post hoc methods. At both extremes, TRUST-SCORE scores suffer due to errors in correctly refusing questions and lower citation groundedness scores. In contrast, TRUST-ALIGN enables models to identify and correctly answer appropriate questions, resulting in AR% closer to the maximum answerable percentage and improvements in F1RG. 

It’s important to note that responsiveness should not be the primary metric for comparing RAG systems when the retrieved documents are the same. The TRUST score rewards accurate answers, appropriate refusals, and correct citations while penalizing failures. Systems with low responsiveness will score poorly on TRUST, regardless of their overall response rate. 

In PostCite, PostAttr, and Self-RAG, adding a refusal prompt results in minimal changes in TRUSTSCORE (e.g., ASQA Self-RAG with LLaMA-2-13b: 51.69% vs. 52.49%). Subcomponent analysis shows little difference in F1RG (42.74% vs. 39.15%), indicating that the refusal prompt does not effectively help models distinguish between answerable and unanswerable questions. These findings highlight the instability of relying on prompting to enhance trustworthiness and underscore the robustness of our system in achieving this goal. 

## F.5 COMPARISON WITH CLOSED-SOURCE MODELS 

We continue our comparison of trustworthiness against competitive closed-source models utilizing in-context learning techniques. As shown in Table 11, our aligned models outperform GPT-3.5 (69 _._ 23 vs. 67 _._ 64) and Claude-3.5 (69 _._ 23 vs. 64 _._ 36) on the ASQA dataset, and substantially outperform GPT-3.5 (55 _._ 31 vs. 38 _._ 95), GPT-4 (55 _._ 31 vs. 40 _._ 35), and Claude-3.5 (55 _._ 31 vs. 39 _._ 78) on QAMPARI. However, the responsiveness of current closed-source models remains much higher than that of our models: even with a refusal prompt, ICL-GPT-4 still answers a significant fraction of questions (86.81% on ASQA, 73.40% on QAMPARI). As discussed in Section 6, this tendency allows GPT-4 to achieve higher **EM[F1] AC**[scores][on][ASQA,][but][it][negatively][impacts][its][attribution] groundedness: its **F1CG** scores on both datasets are lower than those of our models. Similarly, GPT-4’s **F1RG** scores on both datasets are also lower. On QAMPARI, the **EM[F1] AC**[scores][of][all] closed-source models are lower than those of our models. 

Moreover, there still remains a gap between our models and the closed-source models on the ELI5 dataset. Our models’ TRUST-SCORE is 2.45 points lower than that of the advanced ICL-GPT-4, and specifically, the **EM[F1] AC**[and] **[F1][CG]**[scores][are][lower.][For][higher] **[EM][F1] AC**[,][as][discussed][in][Section][6,] it is due to a higher number of its answered answerable questions with comparable EM _[α]_ AC[.][As] for higher **F1CG** , We hypothesize that this gap could be attributed to the information density of the extracted claims utilized in constructing the alignment data (Section 4). Specifically, the three claims derived from the decomposition process may either be redundant or inadequate to fully encapsulate the information inherent in the original labelled response. In some cases, the decomposed claims may even fail to align with the original facts. First, insufficient information can lead the model 

25 

Under review 

to learn to extract fewer facts from the document, thereby reducing the answerability by covering fewer correct answers after training. Second, redundant information can impair grounded citation learning, as it repeats the same information across different claims, making the model less capable of performing precise citations from the corresponding documents. This issue is illustrated in the case study presented in Table 12. 

This experiment reveals that proprietary models demonstrate greater responsiveness compared to our models. While GPT-4 achieves superior **EM[F1] AC**[scores, it underperforms in terms of] **[ F1][CG]**[ and] **F1RG** , suggesting limitations in its ability to ground responses and refuse unanswerable questions. Overall, GPT-3.5 and GPT-4 outperform our models in utilizing retrieved documents for longform question answering, primarily due to the limited capacity of our base model. 

|**Prompt**||**ASQA**|||**QAMPAR**|**I**||**ELI5**|**stworthiness**<br>**Attr. Grdness**<br>**TRUST**<br>**F1CG**|
|---|---|---|---|---|---|---|---|---|---|
||**Responsiveness**|**Tr**|**ustworthiness**|**Responsiveness**|**Tru**|**stworthiness**|**Responsiveness**|**Tru**||
||**AR (%)**|**Truthfullness**|**Attr. Grdness**<br>**TRUST**<br>**F1CG**|**AR (%)**|**Truthfullness**|**Attr. Grdness**<br>**TRUST**<br>**F1CG**|**AR (%)**|**Truthfullness**||
|||**EMF1**<br>**AC**<br>**F1RG**|||**EMF1**<br>**AC**<br>**F1RG**|||**EMF1**<br>**AC**<br>**F1RG**||
|ICL-GPT-3.5<br>R<br>ICL-GPT-4<br>R<br>ICL-Claude-3.5<br>R<br>ICL-GPT-3.5<br>D<br>ICL-GPT-4<br>D<br>ICL-Claude-3.5<br>D<br>DPO-LLaMA-2-7b<br>R<br>DPO-LLaMA-3-8b<br>R|||||**Closed-source**<br><br>|**Models**<br><br>||||
||71.20<br>86.81<br>84.60<br>94.41<br>92.72<br>82.49|52.91<br>66.07<br>**62.96**<br>61.85<br>59.97<br>64.77<br>55.03<br>52.48<br>62.37<br>54.17<br>54.20<br>**66.49**|83.94<br>67.64<br>84.35<br>**69.72**<br>68.35<br>64.36<br>78.04<br>61.85<br>79.70<br>65.41<br>58.88<br>59.86|65.30<br>73.40<br>69.80<br>94.50<br>87.70<br>69.90|26.57<br>58.49<br>30.13<br>55.46<br>28.40<br>58.10<br>20.30<br>29.54<br>26.19<br>40.03<br>0.00<br>57.40|31.80<br>38.95<br>35.45<br>40.35<br>32.83<br>39.78<br>21.22<br>23.69<br>30.02<br>32.08<br>0.00<br>19.13|49.00<br>61.50<br>59.00<br>93.50<br>82.80<br>56.60|32.38<br>58.27<br>**33.05**<br>53.11<br>11.34<br>54.00<br>23.88<br>24.68<br>29.09<br>37.02<br>11.56<br>56.03|57.29<br>49.31<br>**61.84**<br>**49.33**<br>12.43<br>25.92<br>46.28<br>31.61<br>48.33<br>38.15<br>11.22<br>26.27|
||||||**Our Mode**<br><br>|**ls**<br><br>||||
||65.30<br>56.43|52.48<br>66.12<br>53.94<br>65.49|83.94<br>67.51<br>**88.26**<br>69.23|31.10<br>23.10|32.09<br>**71.83**<br>**35.94**<br>71.11|51.33<br>51.75<br>**58.87**<br>**55.31**|21.60<br>15.50|22.54<br>63.27<br>22.81<br>**64.00**|48.43<br>44.75<br>53.84<br>46.88|



Table 11: Our models vs closed source: **AR%** := Answered Ratio in %; **EM[F1] AC**[:= Exact Match F1] (Calibrated); **F1RG** := Grounded refusals F1; **F1CG** := Citation Grounded F1; **TRUST** := TRUST score. **R** := Refusal prompt is used. **D** := Default prompt is used. 

26 

Under review 

Insufficient case 

Question: Why do burns blister and why do burn wounds remain warm long after the injury occurred? Label: Burn blisters occur when the second layer of the skin is damaged, they occur to protect the underlying skin layers from more damage and infection. You could see it as the bodys/skins natural bandage, so never pop them. The skin remain warm because of the increased blood in the area to repair and replace the damaged skin. 

Decomposed claims: 1. Burn blisters occur when the second layer of skin is damaged. 2. Burn wounds remain warm due to increased blood flow to the area to repair and replace damaged skin. 

Missing points: 1. Protection and Infection: The first claim does not mention that the blisters protect the underlying skin from more damage and infection, which is a significant part of the explanation in the answer. 2. Never Pop Them: The answer advises against popping blisters, which is a preventive measure not mentioned in the claims. Redundant case 

Question: How do fitness trackers know that you actually sleeping but not just laying there resting, being awake? Label: Your heart beats slows down when you sleep, they will use a mixture of heart rate and how long you haven’t moved to determine how you’ve slept 

Decomposed claims: 1. The combined factors of heart rate and inactivity determine sleep assessment. 2. Fitness trackers consider the duration of inactivity to assess sleep. 3. A slowed heart rate is an indicator of sleep that fitness trackers monitor. 

Redundant point: The first claim has already summarised the core statement, and the last two claims just expand it and give more details 

Table 12: A case study of the failure of decomposition. 

## F.6 ADAPTABILITY WITH DIFFERENT ALIGNMENT TECHNIQUES 

To demonstrate the robustness of our synthesized alignment data across different training methods, Table 13 also includes the performance of SFT and SIMPO (Meng et al., 2024) methods. Compared to the SFT baseline, which only utilizes the positive data points in the alignment pairs to fine-tune the base model, preference optimization methods, such as DPO and SIMPO, consistently show performance improvements, highlighting the versatility of our data pipeline. Unlike the SFT approach, DPO and SIMPO demonstrate improved TRUST scores, albeit with a reduction in responsiveness. This decrease in responsiveness is actually a favorable outcome, as it indicates that the models are less likely to attempt to answer questions for which they lack sufficient information. 

|**Alignment**|**Model**|**Responsiveness (AR%)**|**EMF1**<br>**AC**|**F1RG**|**F1CG**|**TRUST**|
|---|---|---|---|---|---|---|
|DPO|LLaMA-2-7b<br>LLaMA-3-8b|65.30<br>56.43|52.48<br>**53.94**|66.12<br>65.49|83.94<br>**88.26**|67.51<br>**69.23**|
|SIMPO|LLaMA-2-7b<br>LLaMA-3-8b|72.47<br>57.38|53.19<br>49.84|**66.44**<br>64.13|82.21<br>86.86|67.28<br>66.94|



Table 13: Results using different alignment methods on the ASQA dataset. 

27 

Under review 

## F.7 EVALUATION DATA CREATION WITHOUT USING TRUE 

The determination of question answerability in our dataset is based on a combination of substring matching and TRUE criteria, as detailed in Section 2. Additionally, we developed an alternative version of the evaluation data that relies solely on substring matching, disregarding the TRUE criterion. This relaxation of answerability constraints results in an increased number of answerable questions. The findings from this analysis are presented in Table 14. It is worth noting that the overall trends observed in this analysis align with those reported in Table 2, which employs the combined approach of substring matching followed by TRUE verification. 

|**Prompt**|**ASQA**_(779 answerable, 1_|**ASQA**_(779 answerable, 1_|_69 unanswerable)_<br>**stworthiness**<br>**Attr. Grdness**<br>**TRUST**<br>**F1CG**|**QAMPARI**_(586 answerable_|**QAMPARI**_(586 answerable_|_, 414 unanswerable)_<br>**stworthiness**<br>**Attr. Grdness**<br>**TRUST**<br>**F1CG**|
|---|---|---|---|---|---|---|
||**Responsiveness**<br>**AR (%)**|**Tru**||**Responsiveness**<br>**AR (%)**|**Tru**||
|||**Truthfullness**<br>**EMF1**<br>**AC**<br>**F1RG**|||**Truthfullness**<br>**EMF1**<br>**AC**<br>**F1RG**||
|ICL<br>R<br>PostCite<br>R<br>PostAttr<br>R<br>Self-RAG<br>R<br>ICL<br>D<br>PostCite<br>D<br>PostAttr<br>D<br>Self-RAG<br>D<br>ICL<br>R<br>PostCite<br>R<br>PostAttr<br>R<br>Self-RAG<br>R<br>ICL<br>D<br>PostCite<br>D<br>PostAttr<br>D<br>Self-RAG<br>D<br>ICL<br>R<br>PostCite<br>R<br>PostAttr<br>R<br>ICL<br>D<br>PostCite<br>D<br>PostAttr<br>D<br>DPO-LLaMA-2-7b<br>R<br>DPO-LLaMA-3-8b<br>R|||**LLaM**<br>|**A-2-7b**|||
||0.00<br>10.44<br>10.44<br>100.00<br>94.30<br>88.71<br>87.24<br>98.00|0.00<br>15.13<br>0.13<br>24.91<br>0.13<br>24.91<br>44.40<br>45.11<br>51.13<br>54.01<br>2.64<br>54.63<br>2.71<br>55.63<br>47.22<br>46.27|0.00<br>5.04<br>0.00<br>8.35<br>0.00<br>8.35<br>63.49<br>51.00<br>44.86<br>50.00<br>0.98<br>19.42<br>0.43<br>19.59<br>56.59<br>50.03|0.00<br>34.40<br>34.40<br>96.00<br>93.60<br>56.30<br>51.10<br>96.20|0.00<br>29.28<br>0.00<br>52.57<br>0.00<br>52.57<br>9.64<br>44.15<br>13.31<br>43.37<br>0.00<br>52.85<br>0.00<br>52.45<br>12.13<br>40.83|0.00<br>9.76<br>9.50<br>20.69<br>3.78<br>18.78<br>19.95<br>24.58<br>3.88<br>20.19<br>7.73<br>20.19<br>4.70<br>19.05<br>15.44<br>22.80|
||||**LLaMA**<br>|**-2-13b**<br>|||
||17.41<br>90.51<br>90.51<br>100.00<br>97.57<br>89.77<br>89.24<br>97.68|19.29<br>31.22<br>2.04<br>56.40<br>2.04<br>56.40<br>48.10<br>45.11<br>51.18<br>50.16<br>0.07<br>54.96<br>0.07<br>55.01<br>49.10<br>48.47|14.14<br>21.55<br>1.53<br>19.99<br>0.17<br>19.54<br>69.79<br>54.33<br>9.40<br>36.91<br>0.00<br>18.34<br>0.00<br>18.36<br>63.39<br>53.65|26.50<br>100.00<br>100.00<br>72.70<br>97.80<br>63.00<br>58.50<br>96.30|0.63<br>53.67<br>0.00<br>36.95<br>0.00<br>36.95<br>4.90<br>60.20<br>0.05<br>41.05<br>0.00<br>53.22<br>0.00<br>52.31<br>6.04<br>41.17|0.00<br>18.10<br>8.05<br>15.00<br>2.95<br>13.30<br>26.91<br>30.67<br>0.00<br>13.70<br>7.14<br>20.12<br>4.56<br>18.96<br>21.06<br>22.76|
||||**LLaM**<br>|**A-3-8b**|||
||1.48<br>77.53<br>77.53<br>89.66<br>97.26<br>97.47|2.12<br>17.09<br>34.32<br>54.76<br>34.32<br>54.76<br>58.83<br>**64.47**<br>37.48<br>49.41<br>37.44<br>48.95|**89.14**<br>36.12<br>28.01<br>39.03<br>5.95<br>31.68<br>62.12<br>61.81<br>17.89<br>34.93<br>3.18<br>29.86|3.90<br>87.00<br>87.00<br>70.80<br>92.00<br>93.00|4.77<br>35.42<br>9.90<br>47.98<br>9.90<br>47.98<br>7.48<br>61.03<br>3.35<br>45.43<br>3.32<br>46.03|20.24<br>20.14<br>8.42<br>22.10<br>1.64<br>19.84<br>4.81<br>24.44<br>11.14<br>19.97<br>5.65<br>18.33|
||||**Our M**<br>|**odels**|||
||65.30<br>56.43|47.85<br>61.60<br>48.18<br>57.60|84.95<br>64.80<br>88.84<br>**64.87**|32.30<br>22.40|**27.80**<br>**63.60**<br>26.57<br>56.84|49.42<br>46.94<br>**58.77**<br>**47.39**|



Table 14: Results on ASQA, QAMPARI, and ELI5 evaluation datasets where the data are created without using TRUE; **AR%** := Answered Ratio in %; **EM[F1] AC**[:= Exact Match F1 (Calibrated);] **[ F1][RG]** := Grounded refusals F1; **F1CG** := Citation Grounded F1; **TRUST** := TRUST-SCORE. **R** := Refusal prompt is used. **D** := Default prompt is used. 

28 

Under review 

## F.8 EFFECT OF DATA SIZE ON DPO PERFORMANCE 


![](raw/papers/2_LLM_NLP/LLM_paper_list/rag/iclr_2025_Measuring_and_Enhancing_Trustworthiness_of_LLMs_in_RAG_through_Grounded_Attributions_and_Learning_to_Refuse_images/iclr_2025_Measuring_and_Enhancing_Trustworthiness_of_LLMs_in_RAG_through_Grounded_Attributions_and_Learning_to_Refuse.pdf-0029-02.png)


**----- Start of picture text -----**<br>
75<br>ASQA<br>QAMPARI<br>70 68.77 68.96 69.23 ELI5<br>65<br>60<br>57.73<br>55 54.39 54.95<br>50<br>47.55<br>45 45.65 44.87<br>40<br>10.0 25.0 50.0<br>Proportion of Augmented Samples(%)<br>TRUST SCORE<br>**----- End of picture text -----**<br>


Figure 7: Influence of the proportion of augmented training samples on TRUST-SCORE in LLaMA3-8b. 

During the bulk of our experiments, we chose to utilize the top 50% of augmented samples to form the training dataset for DPO alignment due to time constraints. We then investigated whether varying the quantity and difficulty of the samples would influence the final model performance. Fig. 7 shows how the amount of training samples affect the final performance of TRUST-ALIGN model. Notably, selecting the top 25% of augmented samples achieved the highest performance on QAMPARI (57.73) and ELI5 (47.55). The absence of a clear trend suggesting that ”more data is better” can be attributed to the nature of the data itself. Although document recombination resulted in a substantial increase in sample size, the generated samples exhibit significant information overlap, as the top 100 retrieved documents contain highly similar content. Consequently, the positive and negative responses produced through our response construction pipeline are often quite similar across samples. When training with the top 50% of augmented samples, the model may be experiencing overfitting, which could explain the observed decline in performance. Therefore, it is likely that even better performance could be attained by carefully tuning the amount of augmented data used. This finding underscores a limitation of our pipeline, revealing that the diversity of document content plays a crucial role in determining the quality of the augmented samples. 

## G GPT-4 BASED DATA PIPELINE 

For the GPT-4 data pipeline, we employ GPT-4 to simulate a critic that performs two key tasks in succession. First, it identifies and revises mistakes or supplements missing information in the given response based on correct answers. Second, it validates the attribution of statement-level citations and corrects them accordingly. The detailed instruction is provided in Table 24. 

**Coverage critiques.** To ensure that the correct answers are accurately reflected in the given response, we prompt GPT-4 with the corresponding question, correct answers, and reference facts (documents that support the provided correct answers) as context. GPT-4 is then asked to locate specific mistakes or identify any missing correct answers in the given response. After identifying coverage-related issues, GPT-4 is instructed to minimally revise the original response to correct these issues based on the detected problems. This minimal revision approach is intended to generate more precise data for alignment learning. 

**Citation critiques.** Based on the revised content, we further tokenize it into individual statements to enable a more fine-grained citation check in later stages. We format all documents in the instruction as holistic facts and instruct GPT-4 to determine the attribution of each statement relative to these facts. We define three levels of attribution: SUPPORT, OPPOSE, and IRRELEVANT. We then compare GPT-4’s attribution results to the original attributions in the response, modifying the original attributions wherever they do not align with GPT-4’s critiques. Finally, we concatenate all citation-revised statements to form the final revised response. 

29 

Under review 

## H EXPERIMENTAL SETUP 

## H.1 IMPLEMENTATION DETAILS 

For all experiments involving our tuned models and baselines, we provided the top 5 retrieved documents as context and used decoding temperatures of 0.1 and 0.5, respectively, with other settings consistent with those in (Gao et al., 2023b). We benchmarked against three open-source models—LLaMA-2-7b, LLaMA-2-13b (Touvron et al., 2023), and LLaMA-3-8b (Dubey et al., 2024)—and three closed-source models—GPT-4 (OpenAI et al., 2024), GPT-3.5 (Brown et al., 2020)[13] , and Claude-3.5-Sonnet[14] . LLaMA-2-7b and LLaMA-3-8b were used as base models for fine-tuning. For supervised fine-tuning (SFT), we trained the models for 2 epochs with a learning rate of 2e-5. For direct preference optimization (DPO) alignment, we trained the models for 2 epochs with a beta value of 0.5. All experiments were conducted on NVIDIA A40 40G GPUs. 

## H.2 DATASET DETAILS 

Following Liu et al. (2023); Gao et al. (2023b), to form _D_ , we divide large text documents into 100-word passages and limit the number of citations _Ci_ for each claim to a maximum of three. If the response is empty, it is excluded from evaluation. We provide statistics of our evaluation in Table 15. 

||ASQA|QAMPARI|ELI5|ExpertQA|
|---|---|---|---|---|
|Total # of Samples|948|1000|1000|2169|
|# Answerable Samples|610|295|207|682|
|# Unanswerable Samples|338|705|793|1487|



Table 15: Statistics of Evaluation Dataset 

**ASQA (Stelmakh et al., 2023)** This long-form factoid dataset features ambiguous queries from AmbigQA (Li et al., 2023), requiring multiple short answers to address different aspects. It includes comprehensive long-form answers that combine these short responses. 

**QAMPARI (Amouyal et al., 2023)** This factoid QA dataset is derived from Wikipedia, with answers consisting of lists of entities gathered from various passages. 

**ELI5 (Fan et al., 2019)** This dataset is a long-form QA collection based on the Reddit forum “Explain Like I’m Five” (ELI5). Most ELI5 questions require the model to utilize knowledge from multiple passages to formulate a complete answer. The ELI5 dataset is frequently used in related research due to its challenging nature (Nakano et al., 2021; Menick et al., 2022; Jiang et al., 2023). 

**ExpertQA (Malaviya et al., 2024)** This dataset spans various topics and requires domain-specific knowledge to solve long-form questions. To further verify the generalizability of our approach, we test our best model and some of the baselines on this unseen dataset. 

## H.3 BASELINES 

H.3.1 IN-CONTEXT LEARNING (ICL) 

Following Gao et al. (2023b), we prepend with two demonstrations, each consisting of a query, top-5 retrieved passages, and an answer with inline citations. 

## H.3.2 POST-HOC SEARCH (POSTCITE) 

Following Gao et al. (2023b), we first prompt the model under a closed book setting i.e. without any retrieved passages, to obtain an uncited answer. Then, GTR is used to find the best matching citation among the top-5 retrieved passages for every statement. 

> 13We utilize the latest version on the AzureOpenAI Service: https://learn.microsoft.com/ en-us/azure/ai-services/openai/concepts/models 

> 14https://www.anthropic.com/news/claude-3-5-sonnet 

30 

Under review 

## H.3.3 POST-HOC ATTRIBUTE (POSTATTR) 

Similar to PostCite, we first obtain model response under a closed book setting. Then we use the TRUE NLI model to find the best matching citation among top-5 retrieved passages. 

## H.3.4 SELF-RAG 

Self-RAG trains the LLM to retrieve documents on demand using special reflection tokens and enhances generation quality through self-reflection. We compare our results against the 7b and 13b models released, using the default settings as described in (Asai et al., 2024). 

## H.3.5 FRONT 

FRONT (Huang et al., 2024b) utilizes a fine-grained attribution training framework that first grounds specific supporting quotes, and then generates responses with citations based on those quotes. It tunes the LLM with automatically collected data based on ChatGPT and quality filtering. We reproduce its 7b model for the evaluation. 

## H.4 REFUSAL DETECTION 

We employed two methods to measure refusals robustly. In a refusal prompt, models were explicitly instructed to respond only with the phrase: _”I apologize, but I couldn’t find an answer to your question in the search results.”_ without providing any further explanation. As the models generally complied with this pattern, we were able to apply fuzzy matching[15] to detect the phrase above indicating refusal. For models responding to a default prompt, refusals did not adhere to a fixed pattern, making detection more challenging. Two human annotators verified that fuzzy matching yielded poor performance Table 16. Hence, GPT-4o was employed as an evaluator to classify whether an answer should be considered a refusal. The specific prompt used is provided in Table 25. 

Question: Who killed the man in thelma and louise? 

Answer: "In the movie[¨] Thelma and Louise,the[¨] man who is killed is **not explicitly identified as being killed by any specific character** . The movie ends with a shootout and a crash, and it is unclear who survives or who is responsible for the deaths. **The search results you provided do not provide any clear answers to this question.** 

Fuzzy match judgement: Not refusal Human judgement: Refusal GPT-4o judgement: Refusal 

Table 16: A case study illustrating the necessity of GPT-4o matching for detecting refusals in models prompted with default prompt. 

## I DETAILED RESULTS 

Table 17, Table 18, Table 19 and Table 20 show the full results of our experiments. 

## J PROMPT TEMPLATES 

Table 21, Table 22, Table 23, Table 24 and Table 25 show the prompts used in our experiments. 

> 15Fuzz Partial Ratio was used to mitigate the impact of string length. 

31 

Under review 

||Prompt|AR%|**EMreg**|**EM**_α_<br>**AC**|**EM**_β_<br>**AC**|**EMF1AC**|Rref|Pref|F1ref|Rans|Pans|F1ans|**F1RG**|CR|CP|F1CG|TRUST-SCORE|
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
||||||||||**Qwen-2.5-0.5b**|||||||||
|ICL|R|29.85|7.26|33.06|15.34|20.96|74.56|37.89|50.25|32.30|69.61|44.12|47.19|0.35|0.35|0.35|22.83|
|PostCite|R|46.10|4.72|10.24|7.34|8.55|57.40|37.96|45.70|48.03|67.05|55.97|50.84|8.23|8.23|8.23|22.54|
|PostAttr|R|46.10|4.72|10.24|7.34|8.55|57.40|37.96|45.70|48.03|67.05|55.97|50.84|2.23|2.23|2.23|20.54|
|FRONT|R|100.00|27.19|35.19|54.69|42.83|0.00|0.00|0.00|100.00|64.35|78.31|39.15|47.55|44.30|45.87|42.62|
|SFT|R|83.44|20.12|34.28|44.45|38.71|27.81|59.87|37.98|89.67|69.15|78.09|58.03|61.93|53.61|57.47|51.40|
|DPO|R|71.84|25.98|47.95|53.53|50.59|42.31|53.56|47.27|79.67|71.37|75.29|61.28|52.98|51.84|52.40|54.76|
|DPO-half|R|77.00|22.03|39.36|47.10|42.88|37.28|57.80|45.32|84.92|70.96|77.31|61.32|59.25|57.56|58.39|54.20|
||||||||||**Qwen-2.5-1.5b**|||||||||
|ICL|R|98.52|32.70|41.78|63.98|50.55|2.66|64.29|5.11|99.18|64.78|78.37|41.74|5.47|8.62|6.69|32.99|
|PostCite|R|71.73|9.80|15.51|17.30|16.36|31.66|39.93|35.31|73.61|66.03|69.61|52.46|15.40|15.40|15.40|28.07|
|PostAttr|R|71.73|9.80|15.51|17.30|16.36|31.66|39.93|35.31|73.61|66.03|69.61|52.46|4.45|4.45|4.45|24.42|
|FRONT|R|99.26|38.16|47.58|73.40|57.74|2.07|100.00|4.06|100.00|64.82|78.66|41.36|53.48|58.11|55.70|51.60|
|SFT|R|78.27|23.09|40.30|49.02|44.24|33.14|54.37|41.18|84.59|69.54|76.33|58.75|75.67|67.01|71.08|58.02|
|DPO|R|72.57|27.57|49.70|56.05|52.69|42.90|55.77|48.49|81.15|71.95|76.27|62.38|68.06|65.61|66.81|60.63|
|DPO-half|R|76.48|25.08|45.08|53.57|48.96|39.05|59.19|47.06|85.08|71.59|77.75|62.41|76.30|74.72|75.50|62.29|
||||||||||**Qwen-2.5-3b**|||||||||
|ICL|R|27.43|33.55|63.11|26.90|37.72|84.02|41.28|55.36|33.77|79.23|47.36|51.36|42.49|66.08|51.72|46.93|
|PostCite|R|8.76|13.58|39.96|5.44|9.58|93.79|36.65|52.70|10.16|74.70|17.89|35.30|10.94|10.94|10.94|18.61|
|PostAttr|R|8.76|13.58|39.96|5.44|9.58|93.79|36.65|52.70|10.16|74.70|17.89|35.30|36.29|36.29|36.29|27.06|
|FRONT|R|97.47|34.56|45.78|69.35|55.15|5.03|70.83|9.39|98.85|65.26|78.62|44.01|64.80|60.78|62.72|53.96|
|SFT|R|75.21|24.28|43.85|51.25|47.26|38.17|54.89|45.03|82.62|70.69|76.19|60.61|75.99|70.40|73.09|60.32|
|DPO|R|49.47|23.72|63.48|48.81|55.19|71.01|50.10|58.75|60.82|79.10|68.77|63.76|78.77|78.52|78.64|65.86|
|DPO-half|R|61.92|22.07|50.36|48.46|49.39|55.92|52.35|54.08|71.80|74.62|73.18|63.63|81.63|79.19|80.39|64.47|
||||||||||**Qwen-2.5-7b**|||||||||
|ICL|R|92.09|35.79|50.06|71.64|58.94|17.46|78.67|28.57|97.38|68.04|80.11|54.34|77.62|73.42|75.46|62.91|
|PostCite|R|91.46|20.91|23.44|33.31|27.52|9.76|40.74|15.75|92.13|64.82|76.10|45.93|4.19|4.19|4.19|25.88|
|PostAttr|R|91.46|20.91|23.44|33.31|27.52|9.76|40.74|15.75|92.13|64.82|76.10|45.93|17.92|17.92|17.92|30.46|
|FRONT|R|86.39|41.77|56.34|75.64|64.58|27.51|72.09|39.83|94.10|70.09|80.34|60.08|61.45|55.41|58.27|60.98|
|SFT|R|65.30|24.25|50.36|51.10|50.73|53.25|54.71|53.97|75.57|74.47|75.02|64.50|85.79|78.67|82.07|65.77|
|DPO|R|59.49|25.15|57.28|52.96|55.04|62.13|54.69|58.17|71.48|77.30|74.28|66.22|84.33|82.83|83.57|68.28|
|DPO-half|R|58.44|23.56|55.14|50.08|52.49|63.61|54.57|58.74|70.66|77.80|74.05|66.40|87.60|85.17|86.37|68.42|
||||||||||**Phi3.5-mini**|||||||||
|ICL|R|63.19|38.58|50.70|49.78|50.24|39.35|38.11|38.72|64.59|65.78|65.18|51.95|40.49|45.03|42.64|48.28|
|PostCite|R|23.10|21.76|28.36|10.18|14.98|76.92|35.67|48.73|23.11|64.38|34.02|41.38|9.40|9.40|9.40|21.92|
|PostAttr|R|23.10|21.76|28.36|10.18|14.98|76.92|35.67|48.73|23.11|64.38|34.02|41.38|1.24|1.24|1.24|19.20|
|FRONT|R|99.79|41.59|52.06|80.74|63.30|0.59|100.00|1.18|100.00|64.48|78.41|39.79|74.60|68.89|71.63|58.24|
|SFT|R|66.46|24.75|51.10|52.77|51.92|51.78|55.03|53.35|76.56|74.13|75.32|64.34|87.38|78.62|82.77|66.34|
|DPO|R|66.56|24.81|51.36|53.13|52.23|51.48|54.89|53.13|76.56|74.01|75.26|64.20|87.88|82.99|85.36|67.26|
|DPO-half|R|66.14|25.46|53.51|55.00|54.25|54.44|57.32|55.84|77.54|75.44|76.48|66.16|87.13|81.63|84.29|68.23|
||||||||||**LLaMA-2-7b**|||||||||
|ICL|R|0.00|12.78|0.00|0.00|0.00|100.00|35.65|52.57|0.00|0.00|0.00|26.28|0.00|0.00|0.00|8.76|
|PostCite|R|10.44|8.49|0.25|0.04|0.07|90.53|36.04|51.56|10.98|67.68|18.90|35.23|0.00|0.00|0.00|11.77|
|PostAttr|R|10.44|8.49|0.25|0.04|0.07|90.53|36.04|51.56|10.98|67.68|18.90|35.23|0.00|0.00|0.00|11.77|
|Self-RAG|R|100.00|28.87|37.13|57.71|45.19|0.00|0.00|0.00|100.00|64.35|78.31|39.15|59.27|68.35|63.49|49.28|
|FRONT|R|100.00|40.72|49.69|77.22|60.47|0.00|0.00|0.00|100.00|64.35|78.31|39.15|68.45|69.27|68.86|56.16|
|ICL|D|94.30|32.29|42.06|62.79|50.38|11.54|72.22|19.90|97.54|66.55|79.12|49.51|44.21|43.14|43.67|47.85|
|PostCite|D|88.71|1.91|1.98|2.73|2.30|16.27|51.40|24.72|91.48|66.35|76.91|50.82|0.98|0.98|0.98|18.03|
|PostAttr|D|87.24|1.91|2.01|2.73|2.32|18.05|50.41|26.58|90.16|66.51|76.55|51.56|0.43|0.43|0.43|18.10|
|Self-RAG|D|98.00|30.11|38.63|59.41|46.82|2.37|42.11|4.48|98.20|64.48|77.84|41.16|50.69|64.05|56.59|48.19|
|SFT|R|80.17|29.21|47.96|59.76|53.21|36.69|65.96|47.15|89.51|71.84|79.71|63.43|83.36|76.18|79.61|65.42|
|DPO|R|65.30|25.04|52.10|52.87|52.48|55.33|56.84|56.07|76.72|75.61|76.16|66.12|85.35|82.57|83.94|67.51|
|DPO-half|R|71.31|27.17|51.32|56.87|53.95|48.82|60.66|54.10|82.46|74.41|78.23|66.16|84.37|82.07|83.20|67.77|
||||||||||**LLaMA-2-13b**|||||||||
|ICL|R|17.41|9.17|50.54|13.67|21.52|86.39|37.29|52.10|19.51|72.12|30.71|41.40|10.94|18.81|13.83|25.58|
|PostCite|R|90.51|1.88|1.89|2.66|2.21|14.20|53.33|22.43|93.11|66.20|77.38|49.91|1.53|1.53|1.53|17.88|
|PostAttr|R|90.51|1.88|1.89|2.66|2.21|14.20|53.33|22.43|93.11|66.20|77.38|49.91|0.17|0.17|0.17|17.43|
|Self-RAG|R|100.00|30.82|39.87|61.96|48.52|0.00|0.00|0.00|100.00|64.35|78.31|39.15|66.42|73.52|69.79|52.49|
|ICL|D|97.57|33.31|40.57|62.35|49.16|5.03|73.91|9.42|99.02|65.30|78.70|44.06|7.22|13.25|9.35|34.19|
|PostCite|D|89.77|0.06|0.03|0.04|0.04|15.09|52.58|23.45|92.46|66.27|77.21|50.33|0.00|0.00|0.00|16.79|
|PostAttr|D|89.24|0.06|0.03|0.04|0.04|16.57|54.90|25.45|92.46|66.67|77.47|51.46|0.00|0.00|0.00|17.17|
|Self-RAG|D|97.68|31.36|40.53|61.73|48.93|3.85|59.09|7.22|98.52|64.90|78.26|42.74|58.31|69.44|63.39|51.69|
||||||||||**LLaMA-3.2-1b**|||||||||
|ICL|R|60.23|33.07|37.17|34.80|35.95|41.12|36.87|38.88|60.98|65.15|63.00|50.94|7.92|13.42|9.96|32.28|
|PostCite|R|43.57|1.04|0.73|0.49|0.59|59.76|37.76|46.28|45.41|67.07|54.15|50.22|0.24|0.24|0.24|17.02|
|PostAttr|R|45.78|0.76|0.58|0.41|0.48|54.44|35.80|43.19|45.90|64.52|53.64|48.42|0.00|0.00|0.00|16.30|
|FRONT|R|79.11|27.23|43.72|53.76|48.22|27.51|46.97|34.70|82.79|67.33|74.26|54.48|47.15|49.48|48.29|50.33|
|SFT|R|63.82|21.19|45.80|45.42|45.61|54.14|53.35|53.74|73.77|74.38|74.07|63.91|77.55|69.13|73.10|60.87|
|DPO|R|41.67|15.06|49.16|31.83|38.64|73.96|45.21|56.12|50.33|77.72|61.09|58.61|80.93|77.84|79.35|58.87|
|DPO-half|R|51.69|17.93|49.80|40.01|44.37|67.46|49.78|57.29|62.30|77.55|69.09|63.19|80.41|77.68|79.02|62.19|
||||||||||**LLaMA-3.2-3b**|||||||||
|ICL|R|1.27|0.63|52.78|1.04|2.04|99.41|35.90|52.75|1.64|83.33|3.22|27.98|53.47|54.44|53.95|27.99|
|PostCite|R|47.26|16.56|36.64|26.91|31.03|63.91|43.20|51.55|53.44|72.77|61.63|56.59|22.99|22.99|22.99|36.87|
|PostAttr|R|47.15|15.76|35.18|25.78|29.76|64.20|43.31|51.73|53.44|72.93|61.68|56.71|4.69|4.69|4.69|30.39|
|FRONT|R|95.25|40.68|52.94|78.37|63.19|10.95|82.22|19.32|98.69|66.67|79.58|49.45|60.04|55.09|57.46|56.70|
|SFT|R|68.04|23.75|47.89|50.64|49.23|51.48|57.43|54.29|78.85|74.57|76.65|65.47|80.23|71.52|75.63|63.44|
|DPO|R|77.85|31.81|54.63|66.09|59.82|42.31|68.10|52.19|89.02|73.58|80.56|66.38|85.00|83.43|84.21|70.14|
|DPO-half|R|62.45|23.28|50.74|49.24|49.98|56.80|53.93|55.33|73.11|75.34|74.21|64.77|83.95|77.08|80.37|65.04|
||||||||||**LLaMA-3-8b**|||||||||
|ICL|R|1.48|0.69|67.14|1.54|3.01|99.70|36.08|52.99|2.13|92.86|4.17|28.58|92.86|80.95|86.50|39.36|
|PostCite|R|77.53|22.15|30.17|36.36|32.98|27.51|43.66|33.76|80.33|66.67|72.86|53.31|28.01|28.01|28.01|38.10|
|PostAttr|R|77.53|22.15|30.17|36.36|32.98|27.51|43.66|33.76|80.33|66.67|72.86|53.31|5.95|5.95|5.95|30.75|
|ICL|D|89.66|36.41|49.83|70.17|58.28|20.41|70.41|31.65|95.25|68.35|79.59|55.62|61.40|61.77|61.59|58.50|
|PostCite|D|97.26|27.65|28.91|43.69|34.80|4.73|61.54|8.79|98.36|65.08|78.33|43.56|17.89|17.89|17.89|32.08|
|PostAttr|D|97.47|27.65|28.84|43.69|34.75|4.14|58.33|7.73|98.36|64.94|78.23|42.98|3.18|3.18|3.18|26.97|
|FRONT|R|99.05|41.78|51.34|79.03|62.25|2.37|88.89|4.61|99.84|64.86|78.63|41.62|65.89|66.40|66.14|56.67|
|SFT|R|68.99|25.22|50.59|54.24|52.35|51.18|58.84|54.75|80.16|74.77|77.37|66.06|86.09|76.38|80.95|66.45|
|DPO|R|56.43|23.53|57.72|50.63|53.94|64.79|53.03|58.32|68.20|77.76|72.66|65.49|88.93|87.60|88.26|69.23|
|DPO-half|R|56.75|25.06|60.15|53.05|56.38|64.20|52.93|58.02|68.36|77.51|72.65|65.33|85.64|84.71|85.17|68.96|
|||||||||**Closed-source Models**||||||||||
|GPT-3.5|R|71.20|27.30|50.36|55.72|52.91|48.82|60.44|54.01|82.30|74.37|78.13|66.07|84.66|83.24|83.94|67.64|
|GPT-4|R|86.81|37.93|54.81|73.95|62.96|28.99|78.40|42.33|95.57|70.84|81.37|61.85|85.82|82.93|84.35|69.72|
|Claude-3.5|R|84.60|36.29|52.79|69.41|59.97|34.02|78.77|47.52|94.92|72.19|82.01|64.77|67.29|69.43|68.35|64.36|
|GPT-3.5|D|94.41|34.67|46.27|67.88|55.03|14.20|90.57|24.55|99.18|67.60|80.40|52.48|78.13|77.95|78.04|61.85|
|GPT-4|D|92.72|41.13|52.58|76.65|62.37|16.86|82.61|28.01|98.03|68.03|80.32|54.17|79.48|79.92|79.70|65.41|
|Claude-3.5|D|82.49|32.68|47.64|62.86|54.20|37.87|77.11|50.79|93.77|73.15|82.18|66.49|57.41|60.44|58.88|59.86|



Table 17: Detailed ASQA results. Best TRUST-SCORE values within the open-source baseline, closed-source baseline, and TRUST-ALIGN sets are marked in **bold** (corresponding rows highlighted ), with the overall best TRUST-SCORE values underlined. 

32 

Under review 

||Prompt|AR%|**EM**_α_<br>**AC**|**EM**_β_<br>**AC**|**EMF1AC**|Rref|Pref|F1ref|Rans|Pans|F1ans|**F1RG**|CR|CP|F1CG|TRUST-SCORE|
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
|||||||||**Qwen-2.5-0.5b**|||||||||
|ICL|R|11.40|4.39|1.70|2.45|90.07|71.67|79.82|14.92|38.60|21.52|50.67|0.00|0.00|0.00|17.71|
|PostCite|R|17.00|0.92|0.53|0.67|84.82|72.05|77.92|21.36|37.06|27.10|52.51|5.72|5.72|5.72|19.63|
|PostAttr|R|17.00|0.92|0.53|0.67|84.82|72.05|77.92|21.36|37.06|27.10|52.51|0.90|0.90|0.90|18.03|
|FRONT|R|99.30|7.47|25.14|11.52|0.57|57.14|1.12|98.98|29.41|45.34|23.23|15.08|16.82|15.90|16.88|
|SFT|R|18.50|20.78|13.03|16.02|87.80|75.95|81.45|33.56|53.51|41.25|61.35|27.35|28.31|27.82|35.06|
|DPO|R|17.90|20.86|12.66|15.76|88.65|76.13|81.91|33.56|55.31|41.77|61.84|29.70|29.75|29.73|35.78|
|DPO-half|R|17.40|23.95|14.13|17.77|88.94|75.91|81.91|32.54|55.17|40.94|61.42|31.98|32.36|32.17|37.12|
|||||||||**Qwen-2.5-1.5b**|||||||||
|ICL|R|85.00|10.51|30.28|15.61|19.86|93.33|32.75|96.61|33.53|49.78|41.27|8.47|8.76|8.61|21.83|
|PostCite|R|11.20|6.24|2.37|3.44|90.50|71.85|80.10|15.25|40.18|22.11|51.11|13.95|13.95|13.95|22.83|
|PostAttr|R|11.20|6.24|2.37|3.44|90.50|71.85|80.10|15.25|40.18|22.11|51.11|1.07|1.07|1.07|18.54|
|FRONT|R|98.80|10.42|34.88|16.05|1.56|91.67|3.07|99.66|29.76|45.83|24.45|10.34|13.20|11.60|17.37|
|SFT|R|25.50|25.77|22.27|23.89|85.67|81.07|83.31|52.20|60.39|56.00|69.66|37.41|37.96|37.68|43.74|
|DPO|R|20.00|29.45|19.97|23.80|90.07|79.38|84.39|44.07|65.00|52.53|68.46|50.79|51.17|50.98|47.75|
|DPO-half|R|18.40|36.89|23.01|28.34|91.49|79.04|84.81|42.03|67.39|51.77|68.29|51.65|52.25|51.95|49.53|
||||||||||**Qwen-2.5-3b**||||||||
|ICL|R|22.30|26.91|20.34|23.17|85.11|77.22|80.97|40.00|52.91|45.56|63.27|40.21|42.25|41.20|42.55|
|PostCite|R|0.10|0.00|0.00|0.00|99.86|70.47|82.63|0.00|0.00|0.00|41.31|0.00|0.00|0.00|13.77|
|PostAttr|R|0.10|0.00|0.00|0.00|99.86|70.47|82.63|0.00|0.00|0.00|41.31|25.00|25.00|25.00|22.10|
|FRONT|R|79.10|14.20|38.07|20.69|28.65|96.65|44.20|97.63|36.41|53.04|48.62|25.21|26.16|25.67|31.66|
|SFT|R|27.20|30.02|27.68|28.80|83.26|80.63|81.93|52.20|56.62|54.32|68.12|37.12|37.56|37.34|44.75|
|DPO|R|48.10|28.79|46.94|35.69|66.81|90.75|76.96|83.73|51.35|63.66|70.31|45.51|45.77|45.64|50.55|
|DPO-half|R|17.50|40.53|24.04|30.18|92.34|78.91|85.10|41.02|69.14|51.49|68.29|47.67|47.74|47.71|48.73|
||||||||||**Qwen-2.5-7b**||||||||
|ICL|R|56.30|22.04|42.05|28.92|55.74|89.93|68.83|85.08|44.58|58.51|63.67|38.82|39.76|39.28|**43.96**|
|PostCite|R|26.70|9.04|8.18|8.59|79.15|76.13|77.61|40.68|44.94|42.70|60.16|1.05|1.05|1.05|23.27|
|PostAttr|R|26.70|9.04|8.18|8.59|79.15|76.13|77.61|40.68|44.94|42.70|60.16|13.55|13.55|13.55|27.43|
|FRONT|R|84.70|11.47|32.94|17.02|21.13|97.39|34.73|98.64|34.36|50.96|42.85|24.48|24.48|24.48|28.12|
|SFT|R|31.70|32.41|34.83|33.58|80.43|83.02|81.70|60.68|56.47|58.50|70.10|48.93|49.23|49.08|50.92|
|DPO|R|32.10|28.89|31.44|30.11|80.43|83.51|81.94|62.03|57.01|59.42|70.68|53.48|53.48|53.48|51.42|
|DPO-half|R|29.00|32.02|31.48|31.75|83.83|83.24|83.53|59.66|60.69|60.17|71.85|55.71|56.15|55.93|53.18|
||||||||||**Phi3.5-mini**||||||||
|ICL|R|70.20|8.46|20.14|11.92|31.35|74.16|44.07|73.90|31.05|43.73|43.90|10.66|14.43|12.26|22.69|
|PostCite|R|76.90|2.47|6.45|3.57|25.67|78.35|38.68|83.05|31.86|46.05|42.36|4.49|4.49|4.49|16.81|
|PostAttr|R|76.90|2.47|6.45|3.57|25.67|78.35|38.68|83.05|31.86|46.05|42.36|0.46|0.46|0.46|15.46|
|FRONT|R|100.00|7.75|26.27|11.97|0.00|0.00|0.00|100.00|29.50|45.56|22.78|19.15|24.49|21.50|18.75|
|SFT|R|29.10|35.28|34.80|35.04|84.96|84.49|84.72|62.71|63.57|63.14|73.93|49.13|49.63|49.38|52.78|
|DPO|R|30.10|36.06|36.79|36.42|84.11|84.84|84.47|64.07|62.79|63.42|73.95|53.26|53.55|53.40|54.59|
|DPO-half|R|30.10|35.32|36.04|35.68|84.82|85.55|85.19|65.76|64.45|65.10|75.14|52.21|52.82|52.52|54.45|
|||||||||**LLaMA-2-7b**|||||||||
|ICL|R|0.00|0.00|0.00|0.00|100.00|70.50|82.70|0.00|0.00|0.00|41.35|0.00|0.00|0.00|13.78|
|PostCite|R|34.40|0.00|0.00|0.00|70.21|75.46|72.74|45.42|38.95|41.94|57.34|9.50|9.50|9.50|22.28|
|PostAttr|R|34.40|0.00|0.00|0.00|70.21|75.46|72.74|45.42|38.95|41.94|57.34|3.78|3.78|3.78|20.37|
|Self-RAG|R|96.00|4.45|14.49|6.81|5.25|92.50|9.93|98.98|30.42|46.53|28.23|17.92|22.50|19.95|18.33|
|FRONT|R|100.00|11.18|37.89|17.27|0.00|0.00|0.00|100.00|29.50|45.56|22.78|24.20|24.32|24.26|21.44|
|ICL|D|93.60|5.49|17.51|8.36|8.23|90.63|15.08|97.97|30.88|46.95|31.02|3.83|3.93|3.88|14.42|
|PostCite|D|56.30|0.00|0.00|0.00|45.67|73.68|56.39|61.02|31.97|41.96|49.18|7.73|7.73|7.73|18.97|
|PostAttr|D|51.10|0.00|0.00|0.00|50.21|72.39|59.30|54.24|31.31|39.70|49.50|4.70|4.70|4.70|18.07|
|Self-RAG|D|96.20|5.04|16.45|7.72|4.40|81.58|8.34|97.63|29.94|45.82|27.08|13.25|18.50|15.44|16.75|
|SFT|R|31.60|32.63|34.96|33.76|81.13|83.63|82.36|62.03|57.91|59.90|71.13|46.25|46.49|46.37|50.42|
|DPO|R|32.30|30.64|33.55|32.03|80.85|84.19|82.49|63.73|58.20|60.84|71.67|49.34|49.50|49.42|51.04|
|DPO-half|R|31.50|32.50|34.70|33.56|81.13|83.50|82.30|61.69|57.78|59.67|70.99|52.01|52.19|52.10|52.22|
|||||||||**LLaMA-2-13b**|||||||||
|ICL|R|26.50|0.47|0.42|0.44|79.01|75.78|77.36|39.66|44.15|41.79|59.57|0.00|0.00|0.00|20.00|
|PostCite|R|100.00|0.00|0.00|0.00|0.00|0.00|0.00|100.00|29.50|45.56|22.78|8.05|8.05|8.05|10.28|
|PostAttr|R|100.00|0.00|0.00|0.00|0.00|0.00|0.00|100.00|29.50|45.56|22.78|2.95|2.95|2.95|8.58|
|Self-RAG|R|72.70|1.90|4.69|2.71|32.91|84.98|47.44|86.10|34.94|49.71|48.58|25.73|28.20|26.91|**26.07**|
|ICL|D|97.80|0.00|0.00|0.00|3.12|100.00|6.05|100.00|30.16|46.35|26.20|0.00|0.00|0.00|8.73|
|PostCite|D|63.00|0.00|0.00|0.00|39.01|74.32|51.16|67.80|31.75|43.24|47.20|7.14|7.14|7.14|18.11|
|PostAttr|D|58.50|0.00|0.00|0.00|43.69|74.22|55.00|63.73|32.14|42.73|48.86|4.56|4.56|4.56|17.81|
|Self-RAG|D|96.30|2.39|7.80|3.66|4.40|83.78|8.36|97.97|30.01|45.95|27.15|19.46|22.95|21.06|17.29|
|||||||||**LLaMA-3.2-1b**|||||||||
|ICL|R|19.20|8.01|5.22|6.32|82.55|72.03|76.93|23.39|35.94|28.34|52.64|0.26|0.71|0.38|19.78|
|PostCite|R|41.20|0.27|0.38|0.32|59.15|70.92|64.50|42.03|30.10|35.08|49.79|1.61|1.61|1.61|17.24|
|PostAttr|R|34.00|0.59|0.67|0.63|65.11|69.55|67.25|31.86|27.65|29.61|48.43|0.21|0.21|0.21|16.42|
|FRONT|R|98.60|4.92|16.44|7.57|1.70|85.71|3.34|99.32|29.72|45.75|24.54|14.77|15.91|15.32|15.81|
|SFT|R|26.00|29.87|26.32|27.98|84.40|80.41|82.35|50.85|57.69|54.05|68.20|37.92|38.01|37.96|44.71|
|DPO|R|20.00|33.69|22.84|27.22|89.79|79.12|84.12|43.39|64.00|51.72|67.92|49.42|49.42|49.42|48.19|
|DPO-half|R|21.90|34.83|25.85|29.68|89.22|80.54|84.66|48.47|65.30|55.64|70.15|49.31|49.31|49.31|49.71|
|||||||||**LLaMA-3.2-3b**|||||||||
|ICL|R|34.10|14.98|17.31|16.06|71.91|76.93|74.34|48.47|41.94|44.97|59.65|12.19|13.63|12.87|29.53|
|PostCite|R|39.60|5.53|7.43|6.34|64.11|74.83|69.06|48.47|36.11|41.39|55.22|6.83|6.83|6.83|22.80|
|PostAttr|R|42.00|4.34|6.17|5.10|60.99|74.14|66.93|49.15|34.52|40.56|53.74|0.27|0.27|0.27|19.70|
|FRONT|R|92.70|8.56|26.90|12.99|9.93|95.89|17.99|98.98|31.50|47.79|32.89|18.47|19.96|19.19|21.69|
|SFT|R|27.60|29.06|27.19|28.09|84.11|81.91|83.00|55.59|59.42|57.44|70.22|37.83|38.24|38.03|45.45|
|DPO|R|48.20|23.48|38.36|29.13|67.09|91.31|77.35|84.75|51.87|64.35|70.85|45.43|45.88|45.65|48.54|
|DPO-half|R|18.30|39.01|24.20|29.87|91.49|78.95|84.76|41.69|67.21|51.46|68.11|49.79|50.15|49.97|49.32|
|||||||||**LLaMA-3-8b**|||||||||
|ICL|R|3.90|25.36|3.35|5.92|97.87|71.80|82.83|8.14|61.54|14.37|48.60|17.22|24.53|20.24|24.92|
|PostCite|R|87.00|4.08|12.05|6.10|14.04|76.15|23.71|89.49|30.34|45.32|34.52|8.42|8.42|8.42|16.35|
|PostAttr|R|87.00|4.08|12.05|6.10|14.04|76.15|23.71|89.49|30.34|45.32|34.52|1.64|1.64|1.64|14.09|
|ICL|D|70.80|4.12|9.89|5.82|35.60|85.96|50.35|86.10|35.88|50.65|50.50|4.45|5.23|4.81|20.38|
|PostCite|D|92.00|1.60|5.24|2.45|8.37|73.75|15.03|92.88|29.78|45.10|30.07|11.14|11.14|11.14|14.55|
|PostAttr|D|93.00|1.58|5.24|2.43|7.80|78.57|14.19|94.92|30.11|45.71|29.95|5.65|5.65|5.65|12.68|
|FRONT|R|100.00|8.76|29.68|13.53|0.00|0.00|0.00|100.00|29.50|45.56|22.78|20.77|20.07|20.42|18.91|
|SFT|R|24.20|37.56|30.81|33.85|87.66|81.53|84.48|52.54|64.05|57.73|71.11|47.78|48.24|48.01|50.99|
|DPO|R|22.40|40.95|31.09|35.35|89.08|80.93|84.81|49.83|65.63|56.65|70.73|58.58|58.96|58.77|**54.95**|
|DPO-half|R|24.40|43.56|36.03|39.44|88.37|82.41|85.28|54.92|66.39|60.11|72.70|61.03|61.10|61.06|57.73|
|||||||||**Closed-source**||**Models**|||||||
|GPT-3.5|R|65.30|19.29|42.69|26.57|45.25|91.93|60.65|90.51|40.89|56.33|58.49|30.75|32.92|31.80|38.95|
|GPT-4|R|73.40|21.12|52.55|30.13|37.30|98.87|54.17|98.98|39.78|56.75|55.46|34.44|36.51|35.45|**40.35**|
|Claude-3.5|R|69.80|20.20|47.79|28.40|41.70|97.35|58.39|97.29|41.12|57.80|58.10|32.23|33.46|32.83|39.78|
|GPT-3.5|D|94.50|13.32|42.68|20.30|6.81|87.27|12.63|97.63|30.48|46.45|29.54|20.97|21.47|21.22|23.69|
|GPT-4|D|87.70|17.50|52.01|26.19|17.45|100.00|29.71|100.00|33.64|50.34|40.03|29.66|30.39|30.02|32.08|
|Claude-3.5|D|69.90|0.00|0.00|0.00|41.13|96.35|57.65|96.27|40.63|57.14|57.40|0.00|0.00|0.00|19.13|



Table 18: Detailed QAMPARI results. Best TRUST-SCORE values within the open-source baseline, closed-source baseline, and TRUST-ALIGN sets are marked in **bold** (corresponding rows highlighted ), with the overall best TRUST-SCORE values underlined. 

33 

Under review 

||Prompt|AR%|**EMreg**|**EM**_α_<br>**AC**|**EM**_β_<br>**AC**|**EMF1AC**|Rref|Pref|F1ref|Rans|Pans|F1ans|**F1RG**|CR|CP|F1CG|TRUST-SCORE|
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
||||||||||**Qwen-2.5-0.5b**|||||||||
|ICL|R|82.30|5.40|8.59|34.14|13.73|19.04|85.31|31.13|87.44|21.99|35.15|33.14|0.30|0.49|0.37|15.75|
|PostCite|R|89.80|7.10|6.07|26.33|9.87|10.97|85.29|19.44|92.75|21.38|34.75|27.10|4.10|4.10|4.10|13.69|
|PostAttr|R|89.80|7.10|6.07|26.33|9.87|10.97|85.29|19.44|92.75|21.38|34.75|27.10|0.68|0.68|0.68|12.55|
|FRONT|R|99.90|6.93|8.29|40.02|13.74|0.13|100.00|0.25|100.00|20.72|34.33|17.29|29.86|26.28|27.95|19.66|
|SFT|R|35.50|2.23|8.31|14.25|10.50|68.85|84.65|75.94|52.17|30.42|38.43|57.19|22.11|17.56|19.57|29.09|
|DPO|R|21.70|1.83|13.36|14.01|13.68|82.85|83.91|83.38|39.13|37.33|38.21|60.79|23.89|21.66|22.72|32.40|
|DPO-half|R|26.40|1.70|9.09|11.59|10.19|78.31|84.38|81.23|44.44|34.85|39.07|60.15|25.60|24.07|24.81|31.72|
||||||||||**Qwen-2.5-1.5b**|||||||||
|ICL|R|99.40|12.00|12.42|59.66|20.56|0.63|83.33|1.25|99.52|20.72|34.30|17.78|5.07|4.90|4.99|14.44|
|PostCite|R|91.50|12.23|9.58|42.35|15.63|9.84|91.76|17.77|96.62|21.86|35.65|26.71|5.17|5.17|5.17|15.84|
|PostAttr|R|91.50|12.23|9.58|42.35|15.63|9.84|91.76|17.77|96.62|21.86|35.65|26.71|0.62|0.62|0.62|14.32|
|FRONT|R|99.90|9.63|11.81|57.00|19.57|0.13|100.00|0.25|100.00|20.72|34.33|17.29|37.08|38.35|37.70|24.85|
|SFT|R|41.30|3.50|10.61|21.18|14.14|62.93|85.01|72.32|57.49|28.81|38.39|55.35|32.10|24.35|27.69|32.39|
|DPO|R|33.60|3.57|15.38|24.96|19.03|70.87|84.64|77.14|50.72|31.25|38.67|57.91|32.54|30.78|31.63|36.19|
|DPO-half|R|30.40|3.57|16.67|24.48|19.83|74.91|85.34|79.79|50.72|34.54|41.10|60.44|38.43|37.08|37.75|39.34|
||||||||||**Qwen-2.5-3b**|||||||||
|ICL|R|68.80|11.77|18.94|62.96|29.12|36.19|91.99|51.95|87.92|26.45|40.67|46.31|29.64|40.81|34.34|36.59|
|PostCite|R|49.70|13.73|15.39|36.96|21.73|52.08|82.11|63.73|56.52|23.54|33.24|48.49|7.56|7.56|7.56|25.93|
|PostAttr|R|49.70|13.73|15.39|36.96|21.73|52.08|82.11|63.73|56.52|23.54|33.24|48.49|1.31|1.31|1.31|23.84|
|FRONT|R|93.60|8.47|11.41|51.61|18.69|7.94|98.44|14.70|99.52|22.01|36.05|25.37|39.00|35.92|37.40|27.15|
|SFT|R|34.50|2.83|11.88|19.81|14.85|71.88|87.02|78.73|58.94|35.36|44.20|61.47|39.81|32.64|35.87|37.40|
|DPO|R|13.50|2.20|28.52|18.60|22.52|91.80|84.16|87.82|33.82|51.85|40.94|64.38|42.96|41.10|42.01|42.97|
|DPO-half|R|17.50|2.33|20.86|17.63|19.11|88.40|84.97|86.65|40.10|47.43|43.46|65.05|50.38|45.76|47.96|44.04|
||||||||||**Qwen-2.5-7b**|||||||||
|ICL|R|82.70|12.20|17.67|70.61|28.27|21.31|97.69|34.99|98.07|24.55|39.26|37.13|45.61|42.76|44.13|36.51|
|PostCite|R|95.60|19.87|13.27|61.27|21.82|5.04|90.91|9.56|98.07|21.23|34.91|22.23|7.03|7.03|7.03|17.03|
|PostAttr|R|95.60|19.87|13.27|61.27|21.82|5.04|90.91|9.56|98.07|21.23|34.91|22.23|0.96|0.96|0.96|15.00|
|FRONT|R|57.60|7.97|19.21|53.46|28.27|49.18|91.98|64.09|83.57|30.03|44.19|54.14|59.31|54.15|56.61|46.34|
|SFT|R|25.50|3.00|18.82|23.19|20.78|80.96|86.17|83.49|50.24|40.78|45.02|64.25|50.56|43.71|46.89|43.97|
|DPO|R|21.00|3.30|24.13|24.48|24.30|84.74|85.06|84.90|43.00|42.38|42.69|63.79|47.30|46.75|47.02|45.04|
|DPO-half|R|19.80|3.10|22.05|21.10|21.57|86.13|85.16|85.64|42.51|44.44|43.46|64.55|50.21|49.79|50.00|45.37|
||||||||||**phi3.5-mini**|||||||||
|ICL|R|81.50|13.03|17.30|68.12|27.59|22.07|94.59|35.79|95.17|24.17|38.55|37.17|29.35|30.98|30.14|31.63|
|PostCite|R|84.50|18.73|12.76|52.09|20.50|16.27|83.23|27.22|87.44|21.42|34.41|30.81|4.67|4.67|4.67|18.66|
|PostAttr|R|84.50|18.73|13.23|54.03|21.26|16.27|83.23|27.22|87.44|21.42|34.41|30.81|0.68|0.68|0.68|17.58|
|FRONT|R|96.60|9.03|13.03|60.79|21.46|4.04|94.12|7.74|99.03|21.22|34.95|21.35|64.31|58.77|61.41|34.74|
|SFT|R|24.50|3.30|20.75|24.56|22.50|82.47|86.62|84.50|51.21|43.27|46.90|65.70|51.78|42.68|46.79|45.00|
|DPO|R|24.90|3.47|21.42|25.76|23.39|82.98|87.62|85.23|55.07|45.78|50.00|67.62|50.40|44.78|47.42|46.14|
|DPO-half|R|23.50|3.33|22.34|25.36|23.76|83.10|86.14|84.60|48.79|42.98|45.70|65.15|49.08|41.80|45.15|44.69|
||||||||||**LLaMA-2-7b**|||||||||
|ICL|R|0.50|2.63|0.00|0.00|0.00|100.00|79.70|88.70|2.42|100.00|4.72|46.71|0.00|0.00|0.00|15.57|
|PostCite|R|0.90|6.33|22.22|0.97|1.86|99.12|79.31|88.12|0.97|22.22|1.85|44.98|5.04|5.04|5.04|17.29|
|PostAttr|R|0.90|6.33|22.22|0.97|1.86|99.12|79.31|88.12|0.97|22.22|1.85|44.98|0.00|0.00|0.00|15.61|
|Self-RAG|R|73.50|6.80|9.57|33.98|14.94|29.13|87.17|43.67|83.57|23.54|36.73|40.20|12.34|15.65|13.80|22.98|
|FRONT|R|100.00|9.57|13.07|63.12|21.66|0.00|0.00|0.00|100.00|20.70|34.30|17.15|52.44|53.01|52.72|30.51|
|ICL|D|95.30|12.03|12.07|55.56|19.83|5.55|93.62|10.48|98.55|21.41|35.17|22.82|15.73|16.92|16.30|19.65|
|PostCite|D|83.90|8.13|7.45|30.19|11.95|16.14|79.50|26.83|84.06|20.74|33.27|30.05|4.90|4.90|4.90|15.63|
|PostAtr|D|84.00|8.13|7.44|30.19|11.94|15.89|78.75|26.44|83.57|20.60|33.05|29.74|0.93|0.93|0.93|14.20|
|Self-RAG|D|97.90|8.13|7.97|37.68|13.16|2.40|90.48|4.67|99.03|20.94|34.57|19.62|9.01|12.05|10.31|14.36|
|SFT|R|29.50|3.80|18.36|26.17|21.58|77.05|86.67|81.58|54.59|38.31|45.02|63.30|45.25|35.19|39.59|41.49|
|DPO|R|21.60|3.30|22.07|23.03|22.54|83.98|84.95|84.46|43.00|41.20|42.08|63.27|48.46|46.29|47.35|44.39|
|DPO-half|R|24.20|3.63|21.35|24.96|23.02|81.84|85.62|83.69|47.34|40.50|43.65|63.67|47.59|44.60|46.05|44.25|
||||||||||**LLaMA-2-13b**|||||||||
|ICL|R|46.40|6.90|14.44|32.37|19.97|58.39|86.38|69.68|64.73|28.88|39.94|54.81|3.79|6.28|4.73|26.50|
|PostCite|R|76.60|2.27|1.44|5.31|2.27|25.73|87.18|39.73|85.51|23.11|36.38|38.05|0.72|0.72|0.72|13.68|
|PostAttr|R|76.60|2.27|1.44|5.31|2.27|25.73|87.18|39.73|85.51|23.11|36.38|38.05|0.09|0.09|0.09|13.47|
|Self-RAG|R|22.10|2.40|12.37|13.20|12.77|81.59|83.06|82.32|36.23|33.94|35.05|58.68|22.09|27.60|24.54|32.00|
|ICL|D|96.50|13.07|12.71|59.26|20.93|3.91|88.57|7.49|98.07|21.04|34.64|21.06|2.45|3.25|2.80|14.93|
|PostCite|D|7.00|0.57|7.14|2.42|3.62|92.18|78.60|84.85|3.86|11.43|5.78|45.31|4.73|4.73|4.73|17.89|
|PostAttr|D|6.70|0.57|7.46|2.42|3.66|93.44|79.42|85.86|7.25|22.39|10.95|48.41|0.71|0.71|0.71|17.59|
|Self-RAG|D|98.00|9.73|7.38|34.94|12.19|2.02|80.00|3.94|98.07|20.71|34.20|19.07|5.71|8.06|6.68|12.65|
||||||||||**LLaMA-3.2-1b**|||||||||
|ICL|R|88.40|5.87|7.94|33.90|12.87|11.73|80.17|20.46|88.89|20.81|33.73|27.10|4.05|7.38|5.23|15.07|
|PostCite|R|18.40|3.43|2.17|1.93|2.04|81.97|79.66|80.80|19.81|22.28|20.97|50.88|1.02|1.02|1.02|17.98|
|PostAttr|R|18.40|3.50|2.17|1.93|2.04|81.97|79.66|80.80|19.81|22.28|20.97|50.88|0.07|0.07|0.07|17.66|
|FRONT|R|97.20|6.63|9.77|45.89|16.11|3.40|96.43|6.58|99.52|21.19|34.94|20.76|30.82|29.58|30.19|22.35|
|SFT|R|20.50|2.00|14.63|14.49|14.56|85.25|85.03|85.14|42.51|42.93|42.72|63.93|41.83|33.62|37.28|38.59|
|DPO|R|9.60|1.27|20.83|9.66|13.20|93.82|82.30|87.68|22.71|48.96|31.02|59.35|48.96|47.48|48.21|40.25|
|DPO-half|R|13.80|1.63|19.93|13.29|15.95|90.67|83.41|86.89|30.92|46.38|37.10|61.99|42.45|40.12|41.25|39.73|
||||||||||**LLaMA-3.2-3b**|||||||||
|ICL|R|21.90|3.83|18.04|19.08|18.55|80.45|81.69|81.07|30.92|29.22|30.05|55.56|32.21|29.34|30.70|34.94|
|PostCite|R|92.80|16.33|11.08|49.68|18.12|8.20|90.28|15.03|96.62|21.55|35.24|25.14|4.44|4.44|4.44|15.90|
|PostAttr|R|92.80|16.33|11.30|50.64|18.48|8.20|90.28|15.03|96.62|21.55|35.24|25.14|0.53|0.53|0.53|14.72|
|FRONT|R|86.90|7.73|12.35|51.85|19.95|15.76|95.42|27.06|97.10|23.13|37.36|32.21|41.90|42.04|41.97|31.38|
|SFT|R|14.70|1.67|19.16|13.61|15.92|90.04|83.70|86.76|32.85|46.26|38.42|62.59|59.18|48.53|53.33|43.95|
|DPO|R|17.50|2.23|20.00|16.91|18.33|87.52|84.12|85.78|36.71|43.43|39.79|62.79|57.14|54.66|55.87|45.66|
|DPO-half|R|4.10|0.50|22.36|4.43|7.40|98.11|81.13|88.81|12.56|63.41|20.97|54.89|60.98|57.72|59.31|40.53|
||||||||||**LLaMA-3-8b**|||||||||
|ICL|R|0.00|0.00|0.00|0.00|0.00|100.00|79.30|88.46|0.00|0.00|0.00|44.23|0.00|0.00|0.00|14.74|
|PostCite|R|62.00|10.80|13.87|41.55|20.80|40.86|85.26|55.24|72.95|24.35|36.52|45.88|8.06|8.06|8.06|24.91|
|PostAttr|R|62.00|10.80|13.87|41.55|20.80|40.86|85.26|55.24|72.95|24.35|36.52|45.88|1.25|1.25|1.25|22.64|
|FRONT|R|99.50|9.17|11.47|55.15|18.99|0.63|100.00|1.25|100.00|20.80|34.44|17.85|44.90|44.48|44.69|27.18|
|ICL|D|84.60|11.90|14.74|60.23|23.69|17.65|90.91|29.57|93.24|22.81|36.66|33.11|31.32|30.74|31.03|29.28|
|PostCite|D|98.90|17.40|11.49|54.91|19.00|1.26|90.91|2.49|99.52|20.83|34.45|18.47|6.33|6.33|6.33|14.60|
|PostAttr|D|98.90|17.40|11.49|54.91|19.00|1.13|81.82|2.24|99.03|20.73|34.28|18.26|1.02|1.02|1.02|12.76|
|SFT|R|23.60|3.27|21.19|24.15|22.57|82.98|86.13|84.52|48.79|42.80|45.60|65.06|51.77|42.79|46.85|44.83|
|DPO|R|15.50|2.77|24.30|18.20|20.81|89.66|84.14|86.81|35.27|47.10|40.33|63.57|50.75|49.74|50.24|44.87|
|DPO-half|R|16.00|3.10|28.54|22.06|24.89|89.28|84.29|86.71|36.23|46.88|40.87|63.79|54.17|53.80|53.98|47.55|
|||||||||**Closed-source Models**||||||||||
|GPT-3.5|R|49.00|8.47|23.03|54.51|32.38|58.26|90.59|70.91|76.81|32.45|45.62|58.27|56.57|58.03|57.29|49.31|
|GPT-4|R|61.50|10.50|22.09|65.62|33.05|45.65|94.03|61.46|88.89|29.92|44.77|53.11|61.33|62.35|61.84|49.33|
|Claude-3.5|R|59.00|2.87|7.66|21.82|11.34|48.05|92.93|63.34|85.99|30.17|44.67|54.00|11.64|13.34|12.43|25.92|
|GPT-3.5|D|93.50|14.33|14.58|65.86|23.88|7.57|92.31|13.99|97.58|21.60|35.38|24.68|46.46|46.10|46.28|31.61|
|GPT-4|D|82.80|15.00|18.18|72.71|29.09|21.19|97.67|34.82|98.07|24.52|39.23|37.02|48.20|48.47|48.33|38.15|
|Claude-3.5|D|56.60|3.40|7.89|21.58|11.56|51.07|93.32|66.01|85.99|31.45|46.05|56.03|10.22|12.43|11.22|26.27|



Table 19: Detailed ELI5 results. Best TRUST-SCORE values within the open-source baseline, closed-source baseline, and TRUST-ALIGN sets are marked in **bold** (corresponding rows highlighted ), with the overall best TRUST-SCORE values underlined. 

34 

Under review 

||Prompt|AR%|**EMreg**|**EM**_α_<br>**AC**|**EM**_β_<br>**AC**|**EMF1AC**|Rref|Pref|F1ref|Rans|Pans|F1ans|**F1RG**|CR|CP|F1CG|TRUST-SCORE|
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
||||||||||**Qwen-2.5-0.5b**|||||||||
|ICL|R|78.24|10.14|15.02|37.37|21.42|21.59|68.01|32.77|77.86|31.29|44.64|38.71|0.33|0.69|0.44|20.19|
|PostCite|R|51.41|7.93|10.73|17.55|13.32|48.69|68.69|56.99|51.61|31.57|39.18|48.08|5.53|5.67|5.6|22.33|
|PostAttr|R|51.41|7.93|10.73|17.55|13.32|48.69|68.69|56.99|51.61|31.57|39.18|48.08|1.4|1.58|1.49|20.96|
|FRONT|R|99.86|9.56|12.01|38.15|18.27|0.13|66.67|0.27|99.85|31.44|47.82|24.05|36.72|32.75|34.62|25.65|
|DPO|R|32.96|4.1|17.74|18.6|18.16|75.52|77.24|76.37|51.47|49.09|50.25|63.31|35.55|34.6|35.07|38.85|
||||||||||**Qwen-2.5-1.5b**|||||||||
|ICL|R|98.34|20.3|20.24|63.29|30.67|2.08|86.11|4.07|99.27|31.74|48.1|26.09|6.59|7.22|6.89|21.22|
|PostCite|R|62.19|14.64|16.73|33.09|22.22|40.28|73.05|51.93|67.6|34.17|45.4|48.66|15.32|18.89|16.92|29.27|
|PostAttr|R|62.19|14.64|16.73|33.09|22.22|40.28|73.05|51.93|67.6|34.17|45.4|48.66|11.41|15.53|13.15|28.01|
|FRONT|R|99.59|16.23|19.17|60.73|29.15|0.61|100|1.2|100|31.57|47.99|24.6|49.36|51.12|50.22|34.66|
|DPO|R|30.2|5.71|25.57|24.56|25.06|81.24|79.79|80.51|55.13|57.4|56.25|68.38|52.25|50.65|51.44|48.29|
||||||||||**Qwen-2.5-3b**|||||||||
|ICL|R|68.88|20.43|25.59|56.06|35.14|35.98|79.26|49.49|79.47|36.28|49.82|49.65|35.53|53.4|42.67|42.49|
|PostCite|R|0.05|6.26|0|0|0|99.93|68.54|81.31|0|0|0|40.66|0|0|0|13.55|
|PostAttr|R|0.05|6.26|0|0|0|99.93|68.54|81.31|0|0|0|40.66|0|0|0|13.55|
|FRONT|R|95.48|13.6|17.06|51.81|25.67|5.85|88.78|10.98|98.39|32.4|48.75|29.86|46.28|42.81|44.48|33.34|
|DPO|R|17.15|3.53|29.7|16.2|20.97|92.06|76.18|83.37|37.24|68.28|48.2|65.79|61.29|59.25|60.25|49|
||||||||||**Qwen-2.5-7b**|||||||||
|ICL|R|84.56|20.18|24.92|67.01|36.33|20.24|89.85|33.04|95.01|35.33|51.51|42.28|55.82|56.36|56.09|44.9|
|PostCite|R|42.14|13.76|22.34|29.94|25.58|61.6|72.99|66.81|50.29|37.53|42.98|54.9|13.31|14.27|13.77|31.42|
|PostAttr|R|42.14|13.76|22.34|29.94|25.58|61.6|72.99|66.81|50.29|37.53|42.98|54.9|11.92|13.06|12.46|30.98|
|FRONT|R|65.51|12.36|23.99|49.98|32.41|42.77|85.03|56.91|83.58|40.11|54.21|55.56|70.72|64.28|67.35|51.77|
|DPO|R|24.99|5.51|28.87|22.95|25.57|86.48|79.04|82.59|50|62.92|55.72|69.16|63.38|62.04|62.7|52.48|
||||||||||**phi3.5-mini**|||||||||
|ICL|R|85.15|23.12|25.66|69.5|37.49|18.43|85.09|30.29|92.96|34.33|50.14|40.22|35.06|37.28|36.14|37.95|
|PostCite|R|52.01|21.89|22.43|37.1|27.96|52.12|74.45|61.31|61|36.88|45.97|53.64|7.52|7.26|7.39|29.66|
|PostAttr|R|52.01|21.89|22.43|37.1|27.96|52.12|74.45|61.31|61|36.88|45.97|53.64|5.75|5.64|5.7|29.1|
|FRONT|R|97.37|14.25|18.65|57.75|28.19|3.43|89.47|6.61|99.12|32.01|48.39|27.5|68.61|63.24|65.82|40.5|
|DPO|R|26.05|5.52|30.56|25.32|27.69|85.74|79.49|82.5|51.76|62.48|56.62|69.56|65.13|58.44|61.6|52.95|
||||||||||**LLaMA-2-7b**|||||||||
|ICL|R|0.51|2.75|0|0|0|99.46|68.54|81.15|0.44|27.27|0.87|41.01|9.09|10|9.52|16.84|
|PostCite|R|5.62|14.62|15.98|2.86|4.85|94.28|68.49|79.34|5.43|30.33|9.2|44.27|5.26|5.19|5.23|18.12|
|PostAttr|R|5.62|14.62|15.98|2.86|4.85|94.28|68.49|79.34|5.43|30.33|9.2|44.27|2.27|2.25|2.26|17.13|
|FRONT|R|100|4.17|6.13|19.5|9.33|0|0|0|100|31.44|47.84|23.92|78.99|70.93|74.75|36|
|DPO|R|20.01|4.34|32.18|20.48|25.03|90.45|77.52|83.49|42.82|67.28|52.33|67.91|63.98|61.02|62.46|51.8|
||||||||||**LLaMA-3.2-1b**|||||||||
|ICL|R|90|13.97|14.54|41.62|21.55|10.56|72.35|18.43|91.2|31.86|47.23|32.83|7.16|12.25|9.04|21.14|
|PostCite|R|30.84|12.09|5.53|5.43|5.48|68.59|68|68.3|29.62|30.19|29.9|49.1|2.6|2.75|2.67|19.08|
|PostAttr|R|48.41|10.97|6.79|10.46|8.24|51.04|67.83|58.25|47.21|30.67|37.18|47.72|1.45|1.54|1.5|19.15|
|FRONT|R|95.62|10.27|13.84|42.08|20.83|5.38|84.21|10.11|97.8|32.16|48.4|29.26|38.04|36.87|37.45|29.18|
|DPO|R|15.44|3.14|30.85|15.15|20.32|93.28|75.63|83.53|34.46|70.15|46.21|64.87|62.74|61.48|62.1|49.1|
||||||||||**LLaMA-3.2-3b**|||||||||
|ICL|R|58.74|17.46|25.72|48.04|33.5|44.86|74.53|56|66.57|35.64|46.42|51.21|35.09|42.34|38.37|41.03|
|PostCite|R|82.85|24.16|17.71|46.68|25.68|18.29|73.12|29.26|85.34|32.39|46.95|38.11|5.34|5.24|5.29|23.03|
|PostAttr|R|82.85|24.75|17.56|46.26|25.45|18.63|74.46|29.8|86.07|32.67|47.36|38.58|3.39|3.4|3.4|22.48|
|FRONT|R|83.36|12.03|18.76|49.73|27.24|21.72|89.47|34.96|94.43|35.62|51.73|43.34|51.21|50.62|50.91|40.5|
|DPO|R|7.24|1.34|31.32|7.21|11.72|98.05|72.47|83.34|18.77|81.53|30.51|56.93|80.25|76.54|78.35|49|
||||||||||**LLaMA-3-8b**|||||||||
|ICL|R|0.65|0.28|70.24|1.44|2.82|99.87|68.91|81.55|1.76|85.71|3.45|42.5|68.93|70|69.46|38.26|
|PostCite|R|15.68|4.87|21.13|10.53|14.06|85.27|69.33|76.48|17.74|35.59|23.68|50.08|7.15|7.03|7.09|23.74|
|PostAttr|R|15.68|4.87|21.13|10.53|14.06|85.27|69.33|76.48|17.74|35.59|23.68|50.08|6.29|6.28|6.29|23.47|
|FRONT|R|99.26|16.67|19.97|63.05|30.34|0.94|87.5|1.86|99.71|31.58|47.97|24.92|57.2|56.21|56.7|37.32|
|DPO|R|17.29|4.38|39.89|20.82|27.36|93.48|76.67|84.24|37.98|72.75|49.9|67.07|70.55|69.67|70.11|54.85|
||||||||||**Closed-source Models**|||||||||
|GPT-3.5|R|59.47|24.53|28.01|52.98|36.65|48.02|81.23|60.36|75.81|40.08|52.43|56.39|63.19|64.68|63.93|52.32|
|GPT-4|R|72.20|25.92|29.66|68.11|41.32|35.98|88.72|51.2|90.03|39.21|54.63|52.91|70|69.66|69.83|54.69|
|Claude-3.5|R|73.95|6.19|8.32|19.57|11.68|34.03|89.56|49.32|91.35|38.84|54.51|51.91|9.76|11.84|10.7|24.76|



Table 20: Detailed ExpertQA results. Best TRUST-SCORE values within the open-source baseline, closed-source baseline, and TRUST-ALIGN sets are marked in **bold** (corresponding rows highlighted ), with the overall best TRUST-SCORE values underlined. 

35 

Under review 

|**Type**|**Template**|
|---|---|
|ASQA|Please provide a high-quality answer to the given|
||question using the provided document.<br>The answer must|
||include all the answer labels, and each answer label used|
||should be marked with its index immediately after it in|
||the format [Answer Label X], where X is the index of the|
||answer label in the provided list starting from 1.<br>For|
||example, [Answer Label 1].<br>Ensure the answer is coherent|
||and natural, and does not exceed four statements.<br>You|
||cannot make up any factual information based on your|
||imagination:<br>The additional information added from|
||the given document should be relevant to the question|
||and grounded by the document, but must not contain any|
||factual information that cannot be inferred from the|
||given answer labels.<br>(e.g., if the answer label does not|
||mention a specific year, you cannot introduce a specific|
||year in the final answer).|
||Question:<br>_{_question_}_|
||Document:<br>_{_passage_}_|
||_{_answers_}_|
||Output:|
|ELI5|Given a problem and some claims as answer tags, please|
||generate a high-quality response.<br>The response needs to|
||follow the following requirements:|
||1.<br>Use only all of the claims:<br>Ensure that the response|
||contains and only contains information from the given|
||claims, without introducing any new information.|
||Guarantee covering all claims in the response.|
||2.<br>Each statement must contain valuable information:|
||Every statement must either directly originate from the|
||claims or infer from the claims, avoiding any irrelevant|
||and unuseful information included in the response.<br>You|
||can use each claim only for one time.|
||3.<br>Condense and combine:<br>If there are similarities|
||between claims, merge them into a comprehensive statement|
||to make the response more concise.<br>For example, if two|
||claims both mention similar aspect of health benefits,|
||they can be merged into one statement.|
||4.<br>Fluent and natural:<br>Ensure that the statements in|
||the response are coherent and natural, using connecting|
||words and maintaining logical order between statements.|
||5.<br>Answer tags in response:<br>Indicate each claim|
||immediately after the corresponding content in the|
||response with the format [Claim X], where X is the index|
||of the claim in the provided list starting from 1.<br>For|
||example, [Claim 1].|
||Question:<br>_{_question_}_|
||_{_claims_}_|
||Generated Response:|



Table 21: GPT-4 prompting templates used for generating natural response based on gold claims for ASQA and ELI5. 

36 

Under review 

**Prompt** The examples below are questions from the same cluster. Identify a single short topic they share in common, for example: Philosophy, Lifestyle, Linear Algebra, Biochemistry, Economics, etc. Additionally, evaluate if the topics in the examples are broadly suitable as knowledge-demanding questions that require additional research or grounding. Exclude any sensitive, inappropriate, or irrelevant content, such as sex, explicit violence, ads & scams, and other NSFW subjects. Consider a wide range of content, including scientific, educational, historical, cultural, and practical applications. Provide a rating from 1 to 7 based on the topic’s dependence on additional knowledge or search materials: a score of 1 indicates the question can be answered with common sense alone, without needing any additional information lookup; a score of 5 means the topic requires a combination of common sense and additional lookup, roughly an equal split between the two; a score of 7 indicates that answering the question directly would be difficult, and without additional information, the answer would likely be incorrect. The output format should be like this: Topic: the ~~t~~ opic, Demanding value rating: score. 

Table 22: Prompt used for acquiring domain labels and knowledge demanding score 

|**Type**|**Prompt**|
|---|---|
|Default|Write an accurate, engaging, and concise answer for the|
||given question using only the provided search results|
||(some of which might be irrelevant) and cite them|
||properly.<br>Use an unbiased and journalistic tone.<br>Always|
||cite for any factual claim.<br>When citing several search|
||results, use [1][2][3].<br>Cite at least one document and|
||at most three documents in each statement.<br>If multiple|
||documents support the statement, only cite a minimum|
||sufficient subset of the documents.|
|Refusal|_{_Default_}_<br>+<br>If none of the provided documents contains the|
||answer, only respond with ‘‘I apologize, but I couldn’t|
||find an answer to your question in the search results.’’|
||Do not add further explanation as to why an answer cannot|
||be provided; just state the response above as-is|



Table 23: Two prompt types used for the baselines. 

37 

Under review 

|**Coverage Critic Prompt**|
|---|
|[INSTRUCTION]|
|You will be given Question and the corresponding correct answers, along with a candidate answer and|
|reference facts.<br>Please follow these steps to process the candidate answer:|
|1.<br>Carefully read and understand the given Question, the list of correct answers, and the candidate|
|answer.|
|2.<br>For each given correct answer, first determine if there is a conflict with the candidate answer:|
|- If there is no conflict, and it is included in the candidate answer, extract the matched term from the|
|candidate answer and classify them as "upvote".|
|- If there is a conflict, identify the specific conflicting span within the candidate answer (accurately|
|pinpoint the details), classify it as "downvote", then only minimally modify the conflicting part of|
|the candidate answer to correct it according to the corresponding correct answer (using context from the|
|reference fact).<br>Classify the modified span as "revise".|
|- If there is a conflict, but it is not included in the candidate answer, extend the candidate answer|
|to include the correct answer (using material from the corresponding part of the reference facts), and|
|classify the extended portion as "revise".|
|3.<br>At the end of your response, provide the following:<br>- The final revised candidate answer that|
|includes all correct answers and has no conflicts (if no modification is needed, output the original|
|one).|
|[TASK]|
|Question:<br>_{_QUESTION_}_|
|Correct Answers:<br>_{_SHORT<br>~~A~~NS_}_|
|Candidate Answer:<br>_{_CANDIDATE_}_|
|Reference Facts:<br>_{_FACT_}_|
|**Citation Critic Prompt**|
|[INSTRUCTION]|
|Given a question and a list of CLAIMs, use the provided FACTs to determine which numbered FACTs togeter|
|SUPPORT, OPPOSE, or are IRRELEVANT to each CLAIM. Follow these to give your judgement:|
|1.<br>"SUPPORT" means the FACT directly participates in supporting the factuality of the CLAIM. The CLAIM|
|should be strongly implied by the FACT.|
|2.<br>"OPPOSE" means the FACT contributes to prove the CLAIM contains at least one factual error.|
|3.<br>"IRRELEVANT" means the FACT does not contribute directly to either SUPPORT or OPPOSE the given|
|CLAIM.|
|4.<br>Carefully read the given question and FACTs to ensure you have a clear understanding of them.|
|5.<br>For each CLAIM, analyze its content to show all factual arguments and assertions.|
|6.<br>Look into the details of each FACT, and find factual-related points of each FACT.|
|7.<br>Before determining your final judgement for all CLAIMs at the end, state your reasoning and evidence|
|first.|
|8.<br>In your final judgement, give a numbered list with each line corresponding to a CLAIM. Then, for|
|each CLAIM, separately list the index of each FACT for "SUPPORT", "OPPOSE", and "IRRELEVANT", with the|
|format [FACT X], where X is the index of the FACT starting from 1.<br>For example, suppose we have two|
|CLAIMs and three FACTs in total:<br>"/n/n1.<br>SUPPORT: [FACT 1][FACT 3], OPPOSE: NONE, IRRELEVANT: [FACT|
|2]/n/n2.<br>SUPPORT: NONE, OPPOSE: [FACT 2], IRRELEVANT: [FACT 1][FACT 3]".<br>If no FACT, then just give|
|"NONE".|
|[TASK]|
|Question:<br>_{_QUESTION_}_|
|CLAIM: _{_CLAIM<br>~~P~~LACEHOLDER_}_|
|FACTs:<br>_{_FACT<br>~~P~~LACEHOLDER_}_|



Table 24: The prompts used for GPT-4 based critic. 

38 

Under review 

|**GPT Evaluator Prompt**|||
|---|---|---|
|Instruction:<br>You are an intelligent evaluator.<br>Given a pair of Question and Answer, your task is to|||
|determine if the Answer is classified as REFUSED or NOT REFUSED. Please follow these steps to produce|||
|your judgement:|||
|1.<br>REFUSED means that the Answer contains any sentiment of|being unable to provide a response.|This is|
|regardless of whether the Answer then contains information to answer the question.|||
|2.<br>NOT REFUSED means the absence of any sentiment of being|unable to provide a response.<br>The Answer||
|attempts to address the Question.|||
|3.<br>Carefully read and understand the Question and Answer.|||
|4.<br>Before determining your final judgement of REFUSED or NOT REFUSED, state your reasoning and evidence|||
|(by extracting minimal subspan of Answer accurately) first.|||
|5.<br>For each Answer, determine if it is REFUSED or NOT REFUSED.|||
|6.<br>At the end of your response, return the following:|||
|- Reasoning:<br>Why do you think this should be classified as|REFUSED or NOT REFUSED.||
|- Span:<br>minimal subspan of Answer accurately extracted that supports reasoning|||
|- Judgement:<br>whether it should be classified as REFUSED or|NOT REFUSED.||
|Question:<br>Who set the record for longest field goal?|||
|Answer:<br>I apologize, but I couldn’t find an answer to your|question in the search results.||
|Reasoning:<br>The answer contains a sentiment of being unable|to provide a response to the question.||
|Span:<br>"I apologize, but I couldn’t find an answer to your question in the search results."|||
|Judgement:<br>REFUSED|||
|Question:<br>Who killed the man in thelma and louise?|||
|Answer:<br>In the movie "Thelma and Louise," the man who is killed isn’t clearly linked to any particular|||
|character.<br>The film concludes with a shootout and a crash,|leaving it uncertain who survives or|who is|
|accountable for the deaths.<br>The search results you shared don’t offer any definitive answers to this|||
|question.|||
|Reasoning:<br>The answer contains a sentiment of being unable|to provide a definitive response and||
|expresses uncertainty about the information.|||
|Span:<br>"leaving it uncertain who survives or who is accountable for the deaths.<br>The search results you|||
|shared don’t offer any definitive answers to this question."|||
|Judgement:<br>REFUSED|||
|Question:<br>When did the us break away from England?|||
|Answer:<br>When the US left England is not explicitly stated.|Hence I cannot provide an answer to your||
|question.|||
|Question:<br>Who played galen in planet of the apes?|||
|Reasoning:<br>The answer explicitly states that the information requested is "not explicitly stated,"|||
|which indicates a sentiment of being unable to provide a response.<br>The statement "Hence I cannot|||
|provide an answer to your question" further reinforces the refusal, as it directly communicates the|||
|inability to answer the question.<br>Despite the presence of some additional context, the overall message|||
|is that the answer cannot be provided, which classifies it as a refusal.|||
|Span:<br>"When the US left England is not explicitly stated.|Hence I cannot provide an answer to your||
|question."|||
|Judgement:<br>REFUSED|||
|Question:<br>Which is the most rainy place on earth?|||
|Answer:<br>Several places on Earth claim to be the most rainy, such as Llor´o, Colombia, which reported an|||
|average annual rainfall of 12,717 mm between 1952 and 1989,|and L´opez de Micay, Colombia, which reported||
|an annual 12,892 mm between 1960 and 2012 [3].<br>However, the|official record is held by Mawsynram,||
|India with an average annual rainfall of 11,872 mm [3], although nearby town Sohra, India, also known|||
|as Cherrapunji, holds the record for most rain in a calendar month for July 1861 and most rain in||a year|
|from August 1860 to July 1861 [1].|||
|Reasoning:<br>The answer does not express any sentiment of being unable to provide a response.<br>It|||
|provides several examples of places that could be considered the most rainy, along with relevant||data|
|to support the information.|||
|Span:<br>"Several places on Earth claim to be the most rainy,|such as Llor´o, Colombia, which reported an||
|average annual rainfall of 12,717 mm between 1952 and 1989,|and L´opez de Micay, Colombia, which reported||
|an annual 12,892 mm between 1960 and 2012.<br>However, the official record is held by Mawsynram, India|||
|with an average annual rainfall of 11,872 mm."|||
|Judgement:<br>NOT REFUSED|||
|Question:<br>_{_Q_}_|||
|Answer:<br>_{_A_}_|||
|Return the following:|||
|Reasoning:|||
|Span:|||
|Judgement:|||



Table 25: The prompts used for GPT-4o based evaluator for refusal detection. 

39 

