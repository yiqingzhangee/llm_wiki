# **RAGCHECKER: A Fine-grained Framework for Diagnosing Retrieval-Augmented Generation** 

**Dongyu Ru**[1] _[∗]_ **Lin Qiu**[1] _[∗]_ **Xiangkun Hu**[1] _[∗]_ **Tianhang Zhang**[1] _[∗]_ **Peng Shi**[1] _[∗]_ **Shuaichen Chang**[1] _[∗]_ **Cheng Jiayang**[1] _[†]_ **Cunxiang Wang**[1] _[†]_ **Shichao Sun**[2] **Huanyu Li**[2] **Zizhao Zhang**[1] _[†]_ **Binjie Wang**[1] _[†]_ **Jiarong Jiang**[1] **Tong He**[1] **Zhiguo Wang**[1] **Pengfei Liu**[2] **Yue Zhang**[3] **Zheng Zhang**[1] 

1Amazon AWS AI 2Shanghai Jiaotong University 3Westlake University 

## **Abstract** 

Despite Retrieval-Augmented Generation (RAG) showing promising capability in leveraging external knowledge, a comprehensive evaluation of RAG systems is still challenging due to the modular nature of RAG, evaluation of long-form responses and reliability of measurements. In this paper, we propose a fine-grained evaluation framework, RAGCHECKER, that incorporates a suite of diagnostic metrics for both the retrieval and generation modules. Meta evaluation verifies that RAGCHECKER has significantly better correlations with human judgments than other evaluation metrics. Using RAGCHECKER, we evaluate 8 RAG systems and conduct an indepth analysis of their performance, revealing insightful patterns and trade-offs in the design choices of RAG architectures. The metrics of RAGCHECKER can guide researchers and practitioners in developing more effective RAG systems[3] . 

## **1 Introduction** 

Retrieval-Augmented Generation (RAG) systems [18, 7] enhance Large Language Models (LLMs) by incorporating external knowledge bases, enabling more precise and contextually relevant responses [7, 53, 13]. As these systems become integral to a variety of applications [54, 2, 8], it’s imperative to develop robust and comprehensive evaluation frameworks to assess their performance and identify areas for improvement. Evaluating RAG systems, however, presents several challenges: 

(1) _modular complexity_ : The modular nature of RAG systems, comprising both a retriever and a generator, complicates the design of effective evaluation metrics. It is crucial to establish metrics that can holistically assess the entire system as well as evaluate the individual modules and their interplay [53], allowing for fully understanding the sources of the errors and misses and how they are generated. (2) _metric limitation_ : Existing metrics for evaluating RAG systems, which are often rule-based or coarse-grained, fall short in providing accurate and interpretable results. Specifically, traditional metrics like recall@k and MRR [44] for retrievers depend on annotated chunks and a rigid chunking approach, missing out on the full semantic scope of the knowledge base. For generators, typical measures such as n-gram-based (e.g., BLEU [30], ROUGE [19]), embedding-based (e.g., BERTScore [56]), and LLM-based methods [45] perform well with concise answers but fail to detect finer distinctions in longer responses. To bridge these gaps, it is essential to develop detailed, semantic-based evaluation metrics that effectively capture the intricacies and overall quality of both the retrieval and generation components in RAG systems. (3) _metric reliability_ : the reliability 

> _∗_ Shared first authorship. 

> _†_ Work done during internship at Amazon. 

> 3This work has been open sourced at `https://github.com/amazon-science/RAGChecker` 

Preprint. Under review. 

of existing metrics for RAG remains under-explored. Effective evaluation metrics must not only accurately reflect system performance but also align with human judgments to ensure their utility in real-world scenarios. 

To overcome these challenges, we introduce RAGCHECKER, an innovative evaluation framework designed for detailed analysis of both retrieval and generation processes. RAGCHECKER is based on claim-level entailment checking which involves operations of extracting claims from the response and ground truth answer and checking them against other texts. This approach enables fine-grained evaluation instead of response-level assessment. RAGCHECKER processes the user query, retrieved context, response, and ground truth answer, producing a suite of metrics: 

1. **Overall Metrics** to provide a holistic view of the system performance, assessing the overall quality of the generated responses. 

2. **Diagnostic Retriever Metrics** to evaluate the effectiveness of the retriever, identifying its strengths and weaknesses in finding relevant information from the knowledge base. 

3. **Diagnostic Generator Metrics** to assess the performance of the generator, diagnosing how well the generator utilizes the retrieved context, handles noisy information, and generates accurate and faithful responses. 

Compared to existing evaluation frameworks, RAGCHECKER provides a more comprehensive assessment of RAG systems. While some frameworks offer fine-grained evaluation only on certain metrics (e.g., RAGAS [5], TruLens [6], ARES [35]) or evaluate specific aspects of RAG (e.g., RGB [4], RECALL [22], NoMIRACL [40]), RAGCHECKER’s metrics are all based on fine-grained claim-level checking and are designed to provide actionable insights into the sources of errors. 

To ensure the reliability of RAGCHECKER, we annotate a human judgment dataset to assess the correlations between the proposed metrics and human judgments. This meta-evaluation validates the effectiveness of RAGCHECKER in capturing the quality and reliability of RAG systems from a human perspective. We demonstrate the effectiveness of RAGCHECKER through comprehensive experiments evaluating 8 state-of-the-art RAG systems on a benchmark repurposed from public datasets across 10 domains. In-depth analysis of the evaluation results reveals that RAGCHECKER provides insightful diagnostic signals (Sec. 4.3) pointing the directions for improvements of RAG systems (Sec. 4.4). 

The main contributions of this paper are as follows: 

- We propose RAGCHECKER, a novel RAG evaluation framework that offers fine-grained evaluation for both the retriever and generator components, introducing new diagnostic metrics to provide actionable insights into the sources of errors. 

- We conduct meta evaluation and verified RAGCHECKER has significantly better correlations with human judgements than other evaluation metrics. 

- We perform extensive experiments evaluating 8 RAG systems on our curated benchmark across 10 domains, and uncover valuable insights, such as the trade-off between retrieval improvement and noise introduction, and the tendency of faithful open-source models to blind trust on context. 

## **2 Related Work** 

## **2.1 Retrieval Augmented Generation** 

Large Language Models (LLMs) demonstrate strong capabilities in generating text, but there are also obstacles such as outdated information and the potential to hallucinate [42, 46, 12]. To address these issues, RAG retrieves external knowledge to generate responses with improved accuracy and factuality [7, 53, 13]. Integrating external knowledge is especially crucial in fields like legal, medical and finance, where precision and reliability are essential [24, 50, 55]. 

RAG systems have shown impressive performance across a range of tasks, including open-domain question answering [27, 10, 18], code generation [32, 57, 38] and dialogue [37, 16, 41]. Additionally, real world products like Bing Search[4] and Langchain [3] have integrated applications based on RAG. 

> 4 `https://www.bing.com/chat` 

2 

## **2.2 Evaluation of RAG** 

Existing evaluation practices for RAG systems can be categorized into two main approaches: evaluating essential capabilities of generators only and assessing end-to-end performance of RAG systems. 

Within the two components of a RAG system, the retriever has been well studied in recent years, thus a line of recent work focused on evaluating essential generator capabilities. RGB [4] evaluated 4 fundamental abilities required for generators including Noise Robustness, Negative Rejection, Information Integration and Counterfactual Robustness by manually constructed test sets. RECALL [22] introduced manually edited counterfactual contexts into QA and text generation datasets to evaluate the counterfactual robustness of LLMs. NoMIRACL [40] evaluated LLMs’ robustness against firststage retrieval errors of RAG systems with manually judged relevant and non-relevant datasets. Wu et al. [49] quantified the tug-of-war between LLMs’ faithfulness and internal prior by introducing varying levels of perturbations on the provided contexts. FaaF [15] introduced a fine-grained fact verification formulation to improve previous prompting-based approaches in evaluating factuality of generators. However, we argue that above generator-only evaluation approaches with manually constructed datasets cannot serve as a general RAG evaluation framework to reveal the entanglement of between generation results and different retrieval behaviors, as shown in the analysis of Sec. 4.3. 

Another line of work focused on assessing end-to-end quality scores of RAG systems. TruLens [6] introduced the concept of RAG Triad, which decompose the quality scores into three aspects: context relevance, groundedness and answer relevance, then predicted the score by prompting LLMs or using NLI models. RAGAS [5] and ARES [35] followed the RAG Triad concept and improved the score prediction approaches on different datasets. CRUD-RAG [25] refered to the CRUD (Create, Read, Update and Delete) actions between users and knowledge bases to develop corresponding datasets and evaluation metrics for RAG systems. We compare the above four evaluation frameworks with RAGCHECKER in the meta evaluation of Sec. 4.2. 

Besides, the following work also provided good insight or high quality datasets for end-to-end RAG evaluation. Liu et al. [21] conducted human evaluation to audit four popular generative search engines in terms of fluency, perceived utility, and verifiability. MEDRAG [50] constructed a medical RAG benchmark from medical QA datasets and evaluated medical RAG systems with QA accuracy. MultiHop-RAG [39] generated multi-hop queries from news articles and evaluated RAG systems with QA accuracy. CDQA [52] proposed a novel approach to generate dynamic QA questions which requires latest information to answer. However, the evaluation metrics used in the work mentioned above rely either on human evaluation or simple textual accuracy, making them incapable of complex RAG scenarios that require long answer evaluation. Therefore, we do not include them in the meta evaluation. 

## **3 RAGCHECKER Framework** 

**Formulation** Define a modular RAG system as RAG = _{_ R _,_ G _}_ , where R is the retriever and G is the generator. Given a query _q_ and documents _D_ , it first retrieves top- _k_ relevant context _{_ chunk _j}_ = R( _q, D, k_ ), and then generates a model response m = G( _{_ chunk _j}, q_ ). For simplicity, we can also represent the overall RAG generation process as m = RAG( _q, D_ ). 

**Design Principle** Given the compositional nature of RAG, we observe there are two major personae using a RAG evaluation framework. The first persona is a user that cares about the overall performance of RAGs and might choose a system with the best performance. Such a persona prefers a single value metric to compare and rank among RAG systems against a benchmark. The second persona is a developer that focuses on improving a RAG system with the need to identify causes of mistakes and potential rooms for improvements. Causes of errors in response can be classified into 1) retrieval errors, where the retriever fails to return complete and relevant context, and 2) generator errors, where the generator struggles to identify and leverage relevant information from context. 

Consequently, metrics that reveal error causes should be different from those for overall performance, in the sense that error causes are module-specific or even reflected only by a certain behavior of a module. To help both personae to assess RAG performance, we design RAGCHECKER , a evaluation framework of RAG systems that consists of a benchmark with rich annotations and a set of diversely-purposed fine-grained metrics. 

3 

## **3.1 Inputs to RAGCHECKER** 

We prepare each sample in our benchmark dataset in the format of a tuple _⟨q, D, gt⟩_ representing query, documents, and ground-truth answer, where query is the input question to a RAG system, documents form the database providing possible context and are processed into chunks with the same number of tokens, and ground-truth answer is a complete and correct answer for the input question. Further information is provided in Sec. 4.1. 

## **3.2 Fine-grained Evaluation with Claim Entailment** 

As illustrated in Fig. 1, a response generated by a RAG system might be a mixture of correct ( ) and incorrect claims ( ), while also missing some in-ground-truth claims ( ). In this sense, evaluating responses at a finer granularity is crucial to comprehensively assess the quality of an answer. For this purpose, we introduce two components: 1) a text-to-claim extractor that decomposes a given text _T_ into a set of claims _{ci}_ , and 2) a claim-entailment checker to determine whether a given claim _c_ is entailed ( _∈_ ) in a reference text _Ref_ or not ( _∈/_ ). 

## **3.3 RAGCHECKER Metrics** 

With the annotation and claim-level entailment functions specified, we next define the metrics. For a RAG user, we design metrics to compare the performance among RAG systems, including a single-value F1 score as an overall metric. For a RAG developer, on the other hand, we propose two sets of modular metrics for the retriever and the generator in a RAG system respectively, that aim to decompose the system and diagnose the source of errors. In the rest of this section, we will first introduce the overall metrics and then go over modular metrics for retriever and generator separately. The formulas for each metric are summarized in Appendix B. 


![](raw/papers/2_LLM_NLP/LLM_paper_list/rag/nips_2024_RAGChecker_A_Fine-grained_Framework_for_Diagnosing_Retrieval-Augmented_Generation_images/nips_2024_RAGChecker_A_Fine-grained_Framework_for_Diagnosing_Retrieval-Augmented_Generation.pdf-0004-06.png)


Figure 1: Illustration of the proposed metrics in RAGCHECKER . The upper Venn diagram depicts the comparison between a model response and the ground truth answer, showing possible correct( ), incorrect( ), and missing claims( ). The retrieved chunks are classified into two categories based on the type of claims they contain. Below, we define the overall, retriever, and generator metrics, illustrating how each component of the RAG system is evaluated for its performance. 

4 

## **3.3.1 Overall Metrics** 

To assess the overall response quality of a RAG system from a user’s perspective, we can compute the precision and recall at claim level for each model generated response against its paired ground-truth answer. Specifically, we first extract claims from a model response _m_ and a ground-truth answer _gt_ as _{c_[(] _i[m]_[)] _}_ and _{c_[(] _i[gt]_[)] _}_ respectively. Then, we define correct claims in the response as _{c_[(] _i[m]_[)] _|c_[(] _i[m]_[)] _∈ gt}_ , and correct claims in the ground-truth answer as _{c_[(] _i[gt]_[)] _|c_[(] _i[gt]_[)] _∈ m}_ . Two metrics can be computed directly: **precision** is the proportion of correct claims in all response claims, and **recall** is the proportion of correct claims in all ground-truth answer claims. Further, the harmonic average of precision and recall gives the **F1** score, as the overall performance metric. 

## **3.3.2 Retriever Metrics** 

Ideally, a perfect retriever returns precisely all claims needed to generate the ground-truth answer. Completeness-wise, we can measure how many claims made in the ground-truth answer are covered by retrieved chunks. With retrieved chunks as the reference text, we compute **claim recall** as the proportion of _{c_[(] _i[gt]_[)] _|c_[(] _i[gt]_[)] _∈{_ chunk _j}}_ . 

Differently, we define the retriever precision at chunk-level instead of claim-level. A retrieved chunk is called _relevant chunk_ (r-chunk), if any ground-truth claim is entailed in it. In other words, chunk _j_ is a relevant chunk if _∃i, s.t. c_[(] _i[gt]_[)] _∈_ chunk _j_ . The rest retrieved chunks are called _irrelevant chunk_ (irr-chunk). The retriever’s **context precision** is defined as _|{_ r-chunk _j}|/k_ , where _k_ is the number of all retrieved chunks. 

Note that a chunk-level precision provides better interpretability than a claim-level one, because in practice RAG systems usually work with documents processed to be text chunks in a fixed size. That being said, it is likely that a chunk may contain relevant claims and irrelevant or misleading information at the same time. As a result, the best possible retriever can only achieve a claim-level precision score lower than 100%, and such an upper-bound varies depending on the actual text distribution in _D_ and chunking strategy. 

## **3.3.3 Generator Metrics** 

Given _k_ retrieved chunks (possibly mixing relevant and irrelevant information), a perfect generator would identify and include all ground-truth-relevant claims and ignore any that are not. Because the generator’s results have dependency on retrieved chunks, we provide in total six metrics characterizing different aspects of its performance. 

Given a model response _m_ and its claims _{c_[(] _i[m]_[)] _}_ , we first compute the proportion of _c_[(] _i[m]_[)] that are entailed in retrieved chunks. This metric is **faithfulness** , as it describes how faithful the generator is to the provided context, thus the higher the better. 

Next, we examine three types of incorrect response claims, i.e. _{c_[(] _i[m]_[)] _|c_[(] _i[m]_[)] _∈/ gt}_ . 

1. The first type includes incorrect claim that are entailed in a relevant chunk, then it indicates the generator is sensitive to noise coupled with useful information. The proportion of this type of claims to all _{c_[(] _i[m]_[)] _}_ is **relevant noise sensitivity** . 

2. The second type includes incorrect claim that are entailed in an irrelevant chunk, then it indicates the generator is also sensitive to noise even in an irrelevant context. The proportion of these incorrect claims is **irrelevant noise sensitivity** . 

3. Finally, the third type includes incorrect claims that are not entailed in any retrieved chunk, meaning all such claims are generated by the generator itself. Its proportion is **hallucination** . 

Note that for simplicity we group the two noise sensitivities in Fig. 1, but later in Sec. 4.3 we can see that generators generally has different sensitivity to relevant and irrelevant noise. 

Finally, we characterize how a generator uses information sources to produce correct claims. A correct claim not entailed by any chunk can only be based on generator’s **self-knowledge** , thus the proportion of these claims reflects how many correct claims are generated on its own. A lower self-knowledge score is better, when the generator is expected to fully depend on retrieved context only in a RAG system. On the other hand, we also check how much retrieved relevant information is 

5 

used by the generator. Retrieved relevant information is measured by the number of ground-truth answer claims entailed in retrieved chunks, while the evidence of being used by generator is reflected by entailment in model response. Therefore, the **context utilization** is computed as the ratio between _|{ci_[(] _[gt]_[)] _|c_[(] _i[gt]_[)] _∈{_ chunk _j}_ and _c_[(] _i[gt]_[)] _∈ m}|_ and _|{c_[(] _i[gt]_[)] _|c_[(] _i[gt]_[)] _∈{_ chunk _j}|_ . Generally a higher context utilization is preferred. 

## **4 Experiments** 

## **4.1 Experimental Setup** 

**Baseline RAG Systems** We apply RAGCHECKER to 8 customized RAG systems to demonstrate how these metrics reflect the properties and differences among them, and how they guide the refinement of these systems. The 8 RAG systems are combinations with 2 retrievers and 4 generators. For retrievers, we choose BM25 [33], a representative classic sparse retrieval framework, and E5-Mistral [48], the SOTA open-source dense retriever. Our four generators are GPT-4 [29], Mixtral8x7B [14], Llama3-8B, and Llama3-70B [1], covering open-source and proprietary LLMs in various sizes. Further details are deferred to Appendix D. We employ Llama3-70B as both the claim extractor and checker models implemented by an open-sourced framework RefChecker[5] [11]. As a validation of its performance on the RefChecker’s hallucination detection benchmark, this setup outperforms the best purely open-sourced combinations reported in RefChecker’s paper (see Appendix G). 

**Benchmark Datasets** For comprehensive evaluations, we curate a benchmark containing 4,162 queries across 10 domains. This benchmark is repurposed from public datasets of open domain question answering, spanning domains of Wikipedia, AI science, novel, biomedical, finance, lifestyle, recreation, science, technology and writing. We convert the short answers to long-form answers in the datasets to align with the current LLM-based RAG systems. Please refer to Appendix A for the details of the benchmark curation process. The statistics of the benchmark are shown in Tab. 1. 

## **4.2 Meta Evaluation** 

We first conduct the meta evaluation to verify the soundness of RAGCHECKER and compare with existing baseline RAG evaluation frameworks. 

**Baseline RAG Evaluation Frameworks** We include a total of 10 metrics from Trulens [6], RAGAS [5], ARES [35] and CRUD-RAG [25] in the meta evaluation, as they are capable to evaluate endto-end performance with long answers. Metrics selected for comparison along with their descriptions are summarized in Tab. 4 of Appendix C. To ensure a fair comparison, we use Llama3-70B-Instruct as the LLM backbone when applicable. Since models in the Llama3 family don’t provide an embedding model, baseline metrics requiring embedding capability still use their corresponding default LLM backbones. In addition to the 10 metrics detailed in the table, we also incorporate BLEU [31], ROUGE-L [20], and BERTScore [56] to assess the correlation between the generated responses and the ground truth answers. 

**Meta Evaluation Dataset** All baseline metrics are designed with different aspects and functionalities to a certain degree, thus making an exact comparison over metric scores inapplicable. However, we argue that a good metric should reflect the relative human preference over different RAG systems. In this spirit, we construct the meta evaluation dataset with sampled instances from the generated responses of 8 baseline RAG systems introduced in Sec. 4.1 on our benchmark. Each meta evaluation instance is a pair of responses from two baseline RAG systems given the same query. By considering all combinations over 10 domains and 28 baseline pairs, we end up with 280 instances for pairwise human preference labeling. For each instance, annotators compare a pair of responses based on correctness, completeness, and overall assessment. For each aspect, annotators measure their preferences as one out of five relative choices, including significantly better, slightly better, tie, slightly worse and significantly worse. For quality control, each instance is annotated by two annotators, and their overall agreement and correlation are measured. To conclude, we build a meta evaluation dataset with 280 instances, each instance is labeled by two annotators with their preference in terms of correctness, completeness and overall assessment. 

> 5 `https://github.com/amazon-science/RefChecker` 

6 

Table 1: Statistics of the RAG benchmark. This benchmark is repurposed from public datasets across 10 domains, containing 4,162 questions. For the domains of Finance, Lifestyle, Recreation, Technology, Science and Novel, the short answers are extended to long-form answers with GPT-4. 

|**Dataset**|**Domain**|**# Query**|**# Doc.**|**Source**|**Example Query**|
|---|---|---|---|---|---|
|ClapNQ|Wikipedia|300|4,293|ClapNQ|Difference between russian blue and british<br>blue cat|
|NovelQA|Novel|280|19|NovelQA|When do the Ewell kids go to school?|
|RobustQA<br>Writing|Writing|500|199,994|LoTTE,<br>RobustQA|What is the difference between online and<br>internet?|
|RobustQA<br>BioASQ|Biomedical|511|197,816|BioASQ|What hand deformities do patients with<br>Apert syndrome present with?|
|RobustQA<br>Finance|Finance|500|57,638|FiQA,<br>RobustQA|Is it safer to send credit card number via<br>unsecured website form or by e-mail?<br>What safer options are there?|
|RobustQA<br>Lifestyle|Lifestyle|500|119,461|LoTTE,<br>RobustQA|Can i eat a day old peanut butter sandwich?|
|RobustQA<br>Recreation|Recreation|500|166,975|LoTTE,<br>RobustQA|Why are so many american (spy) movies<br>set in europe?|
|RobustQA<br>Science|Science|500|125,368|LoTTE,<br>RobustQA|Where is the faw in this proof that 1=2?<br>(derivative of repeated addition)|
|RobustQA<br>Technology|Technology|500|638,509|LoTTE,<br>RobustQA|Why not use larger cipher keys?|
||||||What are the prior approaches proposed to|
|KIWI|AI<br>Science|71|429|KIWI|improve faithfulness of the reasoning steps<br>generated by LLMs and what tasks are they|
||||||applied on?|



**Meta Evaluation Process and Results** Based on the meta evaluation dataset, we perform the following evaluation process. Since the human preference labels can be seen as the score difference of a response pair: _hi_ = _H_ ( _ri_[2][)] _[ −][H]_[(] _[r] i_[1][)] _[ ∈{−]_[2] _[,][ −]_[1] _[,]_[ 0] _[,]_[ 1] _[,]_[ 2] _[}]_[, with a baseline RAG evaluation model] _E_ , we compute a normalized score difference as _ei_ = _f_ ( _E_ ( _ri_[2][)] _[ −][E]_[(] _[r] i_[1][))] _[∈]_[[] _[−]_[2] _[,]_[ 2]][, where] _[ f]_[is a] linear normalization function. Our meta evaluation is the correlation between _hi_ and _ei_ overall 280 instances as reported in Tab. 2, together with the correlation between _hi_ and _h[′] i_[from two annotators] as the upper-bound. In addition, we further compute human agreement rate as the proportion of instances satisfying abs( _hi − h[′] i_[)] _[ ≤]_[1][, and the result is 90.95%.] 

Table 2: Correlation results with Human Evaluation of Correctness, Completeness, and Overall Assessment. We only show the metric with the best correlation for each baseline framework. Full results can be found in Tab. 5 of Appendix C. 

|**Baseline**<br>**Metric**|**Correctness**<br>**Pearson**<br>**Spearman**|**Completeness**<br>**Pearson**<br>**Spearman**|**Overall Assessment**<br>**Pearson**<br>**Spearman**|
|---|---|---|---|
|BLEU<br>BLEU-avg<br>ROUGE<br>ROUGE-L<br>BERTScore<br>BERTScore<br>TruLens<br>Answer Relevance<br>ARES<br>Answer Relevance<br>RAGAS<br>Answer Similarity<br>CRUD-RAG<br>Recall<br>RAGChecker<br>Same metric as human|38.89<br>35.32<br>31.75<br>31.72<br>30.34<br>27.05<br>35.01<br>27.37<br>18.63<br>16.84<br>41.07<br>43.21<br>30.93<br>27.13<br>**49.66**<br>**46.95**|32.13<br>21.85<br>47.88<br>45.67<br>37.93<br>40.05<br>37.24<br>37.91<br>20.13<br>18.13<br>53.16<br>**61.35**<br>45.11<br>43.76<br>**60.67**<br>58.11|35.14<br>29.42<br>43.10<br>43.21<br>33.51<br>35.57<br>35.15<br>33.59<br>17.81<br>16.26<br>48.31<br>57.23<br>41.25<br>39.71<br>**61.93**<br>**60.90**|
|Human<br>Annotator correlation|63.67<br>59.19|71.91<br>68.36|70.09<br>68.89|



From the table, we can observe that RAGCHECKER has the strongest correlation with human preference in terms of three aspects. Among other baseline metrics, Answer Similarity of RAGAS, which is based on the stronger backbone model text-embedding-ada-002 [28], shows the best performance. We also provide a detailed comparison between RAGCHECKER and this strongest baseline in Fig. 4 

7 

of Appendix C. As an upper bound, the human correlations at the bottom show that there is still a clear gap between model predictions and human annotators. 

## **4.3 Main Results** 

We present the averaged evaluation results for 8 RAG systems across 10 diverse domain datasets in Tab. 3. Additional results for all datasets are provided in Appendix E. The RAG system that exhibited the best performance in our experiments is E5-Mistral_GPT-4, owing to the strong retrieval capability of E5-Mistral coupled with the adept comprehension abilities of GPT-4. Next, we provide a list of insights induced from Tab. 3, along with their interpretation and possible directions for improvements. 

Table 3: The averaged evaluation results for different RAG systems across 10 datasets. The overall performance of the RAG system is quantified using precision (Prec.), recall (Rec.), and F1 scores. The retriever component is evaluated based on claim recall (CR) and context precision (CP), while the generator component is diagnosed through context utilization (CU), relevant noise sensitivity (NS(I)), irrelevant noise sensitivity (NS(II)), hallucination (Hallu.), self-knowledge (SK), and faithfulness (Faith.). Additionally, the average number of response claims for each RAG system is provided. 

|RAG systems|Overall|Retriever|Generator<br>#Claim<br>CU_↑_<br>NS(I)_↓_<br>NS(II)_↓_<br>Hallu._↓_<br>SK_↓_<br>Faith._↑_|
|---|---|---|---|
||Prec._↑_<br>Rec._↑_<br>F1_↑_|CR_↑_<br>CP_↑_||
|BM25_GPT-4<br>BM25_Llama3-8b<br>BM25_Llama3-70b<br>BM25_Mixtral-8x7b|61.0<br>49.7<br>50.3<br>52.1<br>43.9<br>42.1<br>59.1<br>44.9<br>46.3<br>52.5<br>44.3<br>42.9|74.0<br>52.3<br>74.0<br>52.3<br>74.0<br>52.3<br>74.0<br>52.3|61.4<br>26.2<br>4.1<br>8.7<br>3.4<br>87.9<br>12<br>54.9<br>31.3<br>6.1<br>9.8<br>1.8<br>88.4<br>11<br>56.2<br>30.4<br>5.3<br>5.1<br>1.7<br>93.2<br>9<br>54.9<br>34.3<br>5.8<br>6.2<br>1.8<br>92.0<br>9|
|E5-Mistral_GPT-4<br>E5-Mistral_Llama3-8b<br>E5-Mistral_Llama3-70b<br>E5-Mistral_Mixtral-8x7b|62.0<br>53.0<br>52.7<br>53.8<br>48.3<br>45.0<br>60.6<br>50.4<br>50.2<br>53.1<br>48.6<br>45.7|83.5<br>61.8<br>83.5<br>61.8<br>83.5<br>61.8<br>83.5<br>61.8|60.4<br>28.9<br>3.5<br>5.7<br>1.4<br>92.9<br>12<br>55.0<br>33.5<br>5.5<br>6.6<br>0.8<br>92.7<br>11<br>57.6<br>31.7<br>4.3<br>3.3<br>0.8<br>95.9<br>10<br>55.2<br>36.5<br>5.1<br>4.0<br>0.8<br>95.2<br>10|



**Retriever Matters Consistently.** The quality of retrieval is crucial, as evidenced by the notable differences in overall Precision, Recall and F1 scores when comparing BM25 with E5-Mistral with the generator fixed. This improvement is agnostic to the specific choice of generator, suggesting a consistent benefit from employing a better retriever. 

**Generator Model Size Brings All-Round Improvement.** Paired to the same retriever, Llama3-70B consistently achieves better overall performance than Llama3-8B. More concretely, this superiority is supported by a better performance over every generator metric, such as improved context utilization, reduced noise sensitivity, and less hallucination. 

**Stable and Performant Context Utilization is Key.** Among all generator metrics, we observe that context utilization strongly correlates to the overall F1 score, while such correlation is relatively weaker for other generator metrics. Also, generators’ context utilization are relatively stable between the two retrievers, meaning their overall recall can be improved with a better retriever. These observations indicate that the capability to fully utilize retrieved context is key, which is intuitive because the generator in a RAG system is expected to leverage context to surpass its self-knowledge. 

**Informative Context Improves Faithfulness and Reduces Hallucination.** As E5-Mistral achieves better claim recall, we observe generators paired to it achieves better faithfulness, indicating generators are all capable to identify and leverage information in context. Similarly, hallucination and selfknowledge are both reduced as well. 

**Retriever Recall Trades-off with Generator Noise Sensitivity.** Claim recall for a retriever characterizes the coverage of all information necessary to produce ground-truth answer. In practice, however, because of the fixed-size chunking strategy, retrieved relevant chunks may inevitably also carry over noise as part of the context. As retriever claim recall increases, all generators become more sensitive to such noise, which can be explained as their faithfulness to certain context is not discriminative enough. This observation shows that generators’ capability to precisely leverage relevant context is still a challenge. 

**Relevant Noise Sensitivity is More Challenging.** For every baseline RAG system, there’s an apparent gap between its relevant and irrelevant noise sensitivity. In correlation to the last paragraph, it further enhance the point that generators demonstrate a chunk-level faithfulness. It means a relevant 

8 

chunk is trusted as a whole, while an irrelevant one only has minimal impact. This subtle yet significant distinction supports and explains the importance of the quality and specification of the database for a RAG system. 

**Open-Source Models are Worse at Distinguishing Accurate Information from Noise.** GPT-4 has both higher context utilization and lower noise sensitivity than the other three open source models. Open source models are faithful but tend to trust the context blindly especially when retrieval gets better. This observation raises the need for improving open source models’ reasoning ability. 

## **4.4 Diagnosis on RAG Settings for Improvements** 

Guided by observations in Sec. 4.3, we modify settings commonly tuned in RAG systems that may lead to improvements, diagnose their working mechanisms with RAGCHECKER metrics, and provide suggestions for improvements on certain aspects. We experiment with different numbers of chunks, chunk sizes, chunk overlap ratios, and generation prompts. We highlight our main findings and suggestions as below, please refer to Appendix F for detailed analysis and results. 

**More Context Enhances Faithfulness.** Increasing the number ( _k_ ) and size of chunks improves the recall of more useful information ( _claim recall_ 61.5 _→_ 77.6 with _k_ 5 _→_ 20, 70.3 _→_ 77.6 with size 150 _→_ 300). Consequently, this provides more context for the generators to be more faithful to ( _faithfulness_ 88.1 _→_ 92.2 with _k_ 5 _→_ 20, 91.2 _→_ 92.2 with size 150 _→_ 300), though at the same time they also become more sensitive to additional noise ( _noise sensitivity_ 34.0 _→_ 35.4 with _k_ 5 _→_ 20, 34.5 _→_ 35.4 with size 150 _→_ 300). Improvements in the overall performance ( _F1_ 51.7 _→_ 53.4 with _k_ 5 _→_ 20, 52.6 _→_ 53.4 with size 150 _→_ 300) indicates benefits from more context. 

**Explicit Requirements in Prompts Affect Generation Preferences.** When prompts introduces explicit requirements for better _faithfulness_ , _context utilization_ , and lower _noise sensitivity_ , generators show improvements in _faithfulness_ (92.2 _→_ 93.6), but struggle with the subtle tension between _context utilization_ (59.2 _→_ 63.7) and _noise sensitivity_ (35.4 _→_ 38.1). 

**Chunk Overlap Does Not Matter a Lot.** The chunk overlap ratio is usually set to be non-zero to help generators better utilize surrounding information and identify chunks with coherent logic. However, it minimally affects generation performance, as retrieving more chunks sharing similar useful information (increased _context precision_ 69.3 _→_ 71.1) does not necessarily increase the total amount of retrieved useful information (comparable _claim recall_ 77.8 _→_ 78.1). 

## **Suggestions to RAG Builders** 

Improving the retriever is an effective way to enhance overall performance. While a better embedding model leads to improvements in both _precision_ and _recall_ , moderately increasing the number and size of chunks improves _recall_ and thus _F1_ with minimal efforts in practice. Note that the effect saturates as the total amount of relevant information is fixed, so they need not be too large for a balanced cost-performance. On the other hand, given a limited number of context, larger chunk sizes with fewer chunks are preferred for better _context precision_ . However, when targeting better _context utilization_ or reduced _noise sensitivity_ , opposite adjustments should be made to alleviate the influence of noise. 

When tuning the generator, the trilemma of _context utilization_ , _noise sensitivity_ , and _faithfulness_ makes it difficult to improve all aspects simultaneously. RAG builders should prioritize certain aspects in the prompt based on their targets, user preferences and the generator’s capability. 

## **5 Conclusion** 

This paper presents RAGCHECKER , a novel evaluation framework designed for RAG systems. We validate our comprehensive suite of metrics, both overall and modular, through rigorous human assessments, demonstrating a strong correlation with evaluations conducted by human annotators. We have undertaken a detailed evaluation of eight distinct RAG systems using these metrics, yielding pivotal insights into the behaviors of the retriever and generator components and the trade-offs inherent in RAG system designs. These findings not only deepen our understanding of RAG system architectures but also furnish critical guidance for future advancements in RAG applications. 

9 

## **References** 

- [1] AI@Meta. Llama 3 model card. 2024. 

- [2] A. Asai, S. Min, Z. Zhong, and D. Chen. Retrieval-based language models and applications. In _Proceedings of the 61st Annual Meeting of the Association for Computational Linguistics (Volume 6: Tutorial Abstracts)_ , pages 41–46, 2023. 

- [3] H. Chase. LangChain. `https://github.com/langchain-ai/langchain` , Oct. 2022. 

- [4] J. Chen, H. Lin, X. Han, and L. Sun. Benchmarking large language models in retrievalaugmented generation. In _Proceedings of the AAAI Conference on Artificial Intelligence_ , volume 38, pages 17754–17762, 2024. 

- [5] S. Es, J. James, L. Espinosa-Anke, and S. Schockaert. Ragas: Automated evaluation of retrieval augmented generation. _arXiv preprint arXiv:2309.15217_ , 2023. 

- [6] J. Ferrara, Ethan-Tonic, and O. M. Ozturk. _The RAG Triad_ , January 2024. `https://www. trulens.org/trulens_eval/core_concepts_rag_triad/` . 

- [7] Y. Gao, Y. Xiong, X. Gao, K. Jia, J. Pan, Y. Bi, Y. Dai, J. Sun, and H. Wang. Retrieval-augmented generation for large language models: A survey. _arXiv preprint arXiv:2312.10997_ , 2023. 

- [8] Y. Guo, Z. Li, X. Jin, Y. Liu, Y. Zeng, W. Liu, X. Li, P. Yang, L. Bai, J. Guo, et al. Retrieval-augmented code generation for universal information extraction. _arXiv preprint arXiv:2311.02962_ , 2023. 

- [9] R. Han, P. Qi, Y. Zhang, L. Liu, J. Burger, W. Wang, Z. Huang, B. Xiang, and D. Roth. Robustqa: Benchmarking the robustness of domain adaptation for open-domain question answering. In _ACL Findings 2023_ , 2023. 

- [10] X. He, Y. Tian, Y. Sun, N. V. Chawla, T. Laurent, Y. LeCun, X. Bresson, and B. Hooi. G-retriever: Retrieval-augmented generation for textual graph understanding and question answering. _arXiv preprint arXiv:2402.07630_ , 2024. 

- [11] X. Hu, D. Ru, L. Qiu, Q. Guo, T. Zhang, Y. Xu, Y. Luo, P. Liu, Y. Zhang, and Z. Zhang. Refchecker: Reference-based fine-grained hallucination checker and benchmark for large language models. 2024. 

- [12] L. Huang, W. Yu, W. Ma, W. Zhong, Z. Feng, H. Wang, Q. Chen, W. Peng, X. Feng, B. Qin, et al. A survey on hallucination in large language models: Principles, taxonomy, challenges, and open questions. _arXiv preprint arXiv:2311.05232_ , 2023. 

- [13] Y. Huang and J. Huang. A survey on retrieval-augmented text generation for large language models. _arXiv preprint arXiv:2404.10981_ , 2024. 

- [14] A. Q. Jiang, A. Sablayrolles, A. Roux, A. Mensch, B. Savary, C. Bamford, D. S. Chaplot, D. d. l. Casas, E. B. Hanna, F. Bressand, et al. Mixtral of experts. _arXiv preprint arXiv:2401.04088_ , 2024. 

- [15] V. Katranidis and G. Barany. Faaf: Facts as a function for the evaluation of rag systems. _arXiv preprint arXiv:2403.03888_ , 2024. 

- [16] M. Komeili, K. Shuster, and J. Weston. Internet-augmented dialogue generation. _arXiv preprint arXiv:2107.07566_ , 2021. 

- [17] T. Kwiatkowski, J. Palomaki, O. Redfield, M. Collins, A. Parikh, C. Alberti, D. Epstein, I. Polosukhin, J. Devlin, K. Lee, K. Toutanova, L. Jones, M. Kelcey, M.-W. Chang, A. M. Dai, J. Uszkoreit, Q. Le, and S. Petrov. Natural questions: A benchmark for question answering research. _Transactions of the Association for Computational Linguistics_ , 7:452–466, 2019. 

- [18] P. Lewis, E. Perez, A. Piktus, F. Petroni, V. Karpukhin, N. Goyal, H. Küttler, M. Lewis, W.-t. Yih, T. Rocktäschel, et al. Retrieval-augmented generation for knowledge-intensive nlp tasks. _Advances in Neural Information Processing Systems_ , 33:9459–9474, 2020. 

- [19] C.-Y. Lin. ROUGE: A package for automatic evaluation of summaries. In _Text Summarization Branches Out_ , pages 74–81, Barcelona, Spain, July 2004. Association for Computational Linguistics. 

- [20] C.-Y. Lin. Rouge: A package for automatic evaluation of summaries. In _Text summarization branches out_ , pages 74–81, 2004. 

10 

- [21] N. F. Liu, T. Zhang, and P. Liang. Evaluating verifiability in generative search engines. In _Findings of the Association for Computational Linguistics: EMNLP 2023_ , pages 7001–7025, 2023. 

- [22] Y. Liu, L. Huang, S. Li, S. Chen, H. Zhou, F. Meng, J. Zhou, and X. Sun. Recall: A benchmark for llms robustness against external counterfactual knowledge. _arXiv preprint arXiv:2311.08147_ , 2023. 

- [23] K. Lo, L. L. Wang, M. Neumann, R. Kinney, and D. Weld. S2ORC: The semantic scholar open research corpus. In D. Jurafsky, J. Chai, N. Schluter, and J. Tetreault, editors, _Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics_ , pages 4969–4983, Online, July 2020. Association for Computational Linguistics. 

- [24] A. Louis, G. van Dijck, and G. Spanakis. Interpretable long-form legal question answering with retrieval-augmented large language models. In _Proceedings of the AAAI Conference on Artificial Intelligence_ , volume 38, pages 22266–22275, 2024. 

- [25] Y. Lyu, Z. Li, S. Niu, F. Xiong, B. Tang, W. Wang, H. Wu, H. Liu, T. Xu, and E. Chen. Crud-rag: A comprehensive chinese benchmark for retrieval-augmented generation of large language models. _arXiv preprint arXiv:2401.17043_ , 2024. 

- [26] M. Maia, S. Handschuh, A. Freitas, B. Davis, R. McDermott, M. Zarrouk, and A. Balahur. Www’18 open challenge: financial opinion mining and question answering. In _Companion proceedings of the the web conference 2018_ , pages 1941–1942, 2018. 

- [27] Y. Mao, P. He, X. Liu, Y. Shen, J. Gao, J. Han, and W. Chen. Generation-augmented retrieval for open-domain question answering. _arXiv preprint arXiv:2009.08553_ , 2020. 

- [28] A. Neelakantan, T. Xu, R. Puri, A. Radford, J. M. Han, J. Tworek, Q. Yuan, N. Tezak, J. W. Kim, C. Hallacy, et al. Text and code embeddings by contrastive pre-training. _arXiv preprint arXiv:2201.10005_ , 2022. 

- [29] OpenAI, :, J. Achiam, S. Adler, S. Agarwal, L. Ahmad, I. Akkaya, F. L. Aleman, D. Almeida, J. Altenschmidt, S. Altman, S. Anadkat, R. Avila, I. Babuschkin, S. Balaji, V. Balcom, P. Baltescu, H. Bao, M. Bavarian, J. Belgum, I. Bello, J. Berdine, G. Bernadett-Shapiro, C. Berner, L. Bogdonoff, O. Boiko, M. Boyd, A.-L. Brakman, G. Brockman, T. Brooks, M. Brundage, K. Button, T. Cai, R. Campbell, A. Cann, B. Carey, C. Carlson, R. Carmichael, B. Chan, C. Chang, F. Chantzis, D. Chen, S. Chen, R. Chen, J. Chen, M. Chen, B. Chess, C. Cho, C. Chu, H. W. Chung, D. Cummings, J. Currier, Y. Dai, C. Decareaux, T. Degry, N. Deutsch, D. Deville, A. Dhar, D. Dohan, S. Dowling, S. Dunning, A. Ecoffet, A. Eleti, T. Eloundou, D. Farhi, L. Fedus, N. Felix, S. P. Fishman, J. Forte, I. Fulford, L. Gao, E. Georges, C. Gibson, V. Goel, T. Gogineni, G. Goh, R. Gontijo-Lopes, J. Gordon, M. Grafstein, S. Gray, R. Greene, J. Gross, S. S. Gu, Y. Guo, C. Hallacy, J. Han, J. Harris, Y. He, M. Heaton, J. Heidecke, C. Hesse, A. Hickey, W. Hickey, P. Hoeschele, B. Houghton, K. Hsu, S. Hu, X. Hu, J. Huizinga, S. Jain, S. Jain, J. Jang, A. Jiang, R. Jiang, H. Jin, D. Jin, S. Jomoto, B. Jonn, H. Jun, T. Kaftan, Łukasz Kaiser, A. Kamali, I. Kanitscheider, N. S. Keskar, T. Khan, L. Kilpatrick, J. W. Kim, C. Kim, Y. Kim, H. Kirchner, J. Kiros, M. Knight, D. Kokotajlo, Łukasz Kondraciuk, A. Kondrich, A. Konstantinidis, K. Kosic, G. Krueger, V. Kuo, M. Lampe, I. Lan, T. Lee, J. Leike, J. Leung, D. Levy, C. M. Li, R. Lim, M. Lin, S. Lin, M. Litwin, T. Lopez, R. Lowe, P. Lue, A. Makanju, K. Malfacini, S. Manning, T. Markov, Y. Markovski, B. Martin, K. Mayer, A. Mayne, B. McGrew, S. M. McKinney, C. McLeavey, P. McMillan, J. McNeil, D. Medina, A. Mehta, J. Menick, L. Metz, A. Mishchenko, P. Mishkin, V. Monaco, E. Morikawa, D. Mossing, T. Mu, M. Murati, O. Murk, D. Mély, A. Nair, R. Nakano, R. Nayak, A. Neelakantan, R. Ngo, H. Noh, L. Ouyang, C. O’Keefe, J. Pachocki, A. Paino, J. Palermo, A. Pantuliano, G. Parascandolo, J. Parish, E. Parparita, A. Passos, M. Pavlov, A. Peng, A. Perelman, F. de Avila Belbute Peres, M. Petrov, H. P. de Oliveira Pinto, Michael, Pokorny, M. Pokrass, V. Pong, T. Powell, A. Power, B. Power, E. Proehl, R. Puri, A. Radford, J. Rae, A. Ramesh, C. Raymond, F. Real, K. Rimbach, C. Ross, B. Rotsted, H. Roussez, N. Ryder, M. Saltarelli, T. Sanders, S. Santurkar, G. Sastry, H. Schmidt, D. Schnurr, J. Schulman, D. Selsam, K. Sheppard, T. Sherbakov, J. Shieh, S. Shoker, P. Shyam, S. Sidor, E. Sigler, M. Simens, J. Sitkin, K. Slama, I. Sohl, B. Sokolowsky, Y. Song, N. Staudacher, F. P. Such, N. Summers, I. Sutskever, J. Tang, N. Tezak, M. Thompson, P. Tillet, A. Tootoonchian, E. Tseng, P. Tuggle, N. Turley, J. Tworek, J. F. C. Uribe, A. Vallone, A. Vijayvergiya, C. Voss, C. Wainwright, J. J. Wang, A. Wang, B. Wang, J. Ward, J. Wei, C. Weinmann, A. Welihinda, P. Welinder, J. Weng, L. Weng, M. Wiethoff, D. Willner, C. Winter, 

11 

S. Wolrich, H. Wong, L. Workman, S. Wu, J. Wu, M. Wu, K. Xiao, T. Xu, S. Yoo, K. Yu, Q. Yuan, W. Zaremba, R. Zellers, C. Zhang, M. Zhang, S. Zhao, T. Zheng, J. Zhuang, W. Zhuk, and B. Zoph. Gpt-4 technical report, 2023. 

- [30] K. Papineni, S. Roukos, T. Ward, and W.-J. Zhu. Bleu: a method for automatic evaluation of machine translation. In P. Isabelle, E. Charniak, and D. Lin, editors, _Proceedings of the 40th Annual Meeting of the Association for Computational Linguistics_ , pages 311–318, Philadelphia, Pennsylvania, USA, July 2002. Association for Computational Linguistics. 

- [31] K. Papineni, S. Roukos, T. Ward, and W.-J. Zhu. Bleu: a method for automatic evaluation of machine translation. In _Proceedings of the 40th annual meeting of the Association for Computational Linguistics_ , pages 311–318, 2002. 

- [32] M. R. Parvez, W. U. Ahmad, S. Chakraborty, B. Ray, and K.-W. Chang. Retrieval augmented code generation and summarization. _arXiv preprint arXiv:2108.11601_ , 2021. 

- [33] S. Robertson, H. Zaragoza, et al. The probabilistic relevance framework: Bm25 and beyond. _Foundations and Trends® in Information Retrieval_ , 3(4):333–389, 2009. 

- [34] S. Rosenthal, A. Sil, R. Florian, and S. Roukos. Clapnq: Cohesive long-form answers from passages in natural questions for rag systems, 2024. 

- [35] J. Saad-Falcon, O. Khattab, C. Potts, and M. Zaharia. Ares: An automated evaluation framework for retrieval-augmented generation systems, 2023. 

- [36] K. Santhanam, O. Khattab, J. Saad-Falcon, C. Potts, and M. Zaharia. ColBERTv2: Effective and efficient retrieval via lightweight late interaction. In M. Carpuat, M.-C. de Marneffe, and I. V. Meza Ruiz, editors, _Proceedings of the 2022 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies_ , pages 3715–3734, Seattle, United States, July 2022. Association for Computational Linguistics. 

- [37] K. Shuster, S. Poff, M. Chen, D. Kiela, and J. Weston. Retrieval augmentation reduces hallucination in conversation. _arXiv preprint arXiv:2104.07567_ , 2021. 

- [38] H. Tan, Q. Luo, L. Jiang, Z. Zhan, J. Li, H. Zhang, and Y. Zhang. Prompt-based code completion via multi-retrieval augmented generation. _arXiv preprint arXiv:2405.07530_ , 2024. 

- [39] Y. Tang and Y. Yang. Multihop-rag: Benchmarking retrieval-augmented generation for multihop queries. _arXiv preprint arXiv:2401.15391_ , 2024. 

- [40] N. Thakur, L. Bonifacio, X. Zhang, O. Ogundepo, E. Kamalloo, D. Alfonso-Hermelo, X. Li, Q. Liu, B. Chen, M. Rezagholizadeh, et al. Nomiracl: Knowing when you don’t know for robust multilingual retrieval-augmented generation. _arXiv preprint arXiv:2312.11361_ , 2023. 

- [41] D. Thulke, N. Daheim, C. Dugast, and H. Ney. Efficient retrieval augmented generation from unstructured knowledge for task-oriented dialog. _arXiv preprint arXiv:2102.04643_ , 2021. 

- [42] S. Tonmoy, S. Zaman, V. Jain, A. Rani, V. Rawte, A. Chadha, and A. Das. A comprehensive survey of hallucination mitigation techniques in large language models. _arXiv preprint arXiv:2401.01313_ , 2024. 

- [43] G. Tsatsaronis, G. Balikas, P. Malakasiotis, I. Partalas, M. Zschunke, M. R. Alvers, D. Weissenborn, A. Krithara, S. Petridis, D. Polychronopoulos, et al. An overview of the bioasq large-scale biomedical semantic indexing and question answering competition. _BMC bioinformatics_ , 16:1–28, 2015. 

- [44] E. M. Voorhees et al. The trec-8 question answering track report. In _Trec_ , volume 99, pages 77–82, 1999. 

- [45] C. Wang, S. Cheng, Q. Guo, Y. Yue, B. Ding, Z. Xu, Y. Wang, X. Hu, Z. Zhang, and Y. Zhang. Evaluating open-qa evaluation. _Advances in Neural Information Processing Systems_ , 36, 2024. 

- [46] C. Wang, X. Liu, Y. Yue, X. Tang, T. Zhang, C. Jiayang, Y. Yao, W. Gao, X. Hu, Z. Qi, et al. Survey on factuality in large language models: Knowledge, retrieval and domain-specificity. _arXiv preprint arXiv:2310.07521_ , 2023. 

- [47] C. Wang, R. Ning, B. Pan, T. Wu, Q. Guo, C. Deng, G. Bao, Q. Wang, and Y. Zhang. Novelqa: A benchmark for long-range novel question answering, 2024. 

- [48] L. Wang, N. Yang, X. Huang, L. Yang, R. Majumder, and F. Wei. Improving text embeddings with large language models. _arXiv preprint arXiv:2401.00368_ , 2023. 

12 

- [49] K. Wu, E. Wu, and J. Zou. How faithful are rag models? quantifying the tug-of-war between rag and llms’ internal prior. _arXiv preprint arXiv:2404.10198_ , 2024. 

- [50] G. Xiong, Q. Jin, Z. Lu, and A. Zhang. Benchmarking retrieval-augmented generation for medicine. _arXiv preprint arXiv:2402.13178_ , 2024. 

- [51] F. Xu, K. Lo, L. Soldaini, B. Kuehl, E. Choi, and D. Wadden. Kiwi: A dataset of knowledgeintensive writing instructions for answering research questions, 2024. 

- [52] Z. Xu, Y. Li, R. Ding, X. Wang, B. Chen, Y. Jiang, X. Deng, J. Ma, H.-T. Zheng, W. Lu, et al. Let llms take on the latest challenges! a chinese dynamic question answering benchmark. _arXiv preprint arXiv:2402.19248_ , 2024. 

- [53] H. Yu, A. Gan, K. Zhang, S. Tong, Q. Liu, and Z. Liu. Evaluation of retrieval-augmented generation: A survey. _arXiv preprint arXiv:2405.07437_ , 2024. 

- [54] C. Zakka, R. Shad, A. Chaurasia, A. R. Dalal, J. L. Kim, M. Moor, R. Fong, C. Phillips, K. Alexander, E. Ashley, et al. Almanac—retrieval-augmented language models for clinical medicine. _NEJM AI_ , 1(2):AIoa2300068, 2024. 

- [55] B. Zhang, H. Yang, T. Zhou, M. Ali Babar, and X.-Y. Liu. Enhancing financial sentiment analysis via retrieval augmented large language models. In _Proceedings of the Fourth ACM International Conference on AI in Finance_ , pages 349–356, 2023. 

- [56] T. Zhang, V. Kishore, F. Wu, K. Q. Weinberger, and Y. Artzi. Bertscore: Evaluating text generation with bert. _arXiv preprint arXiv:1904.09675_ , 2019. 

- [57] S. Zhou, U. Alon, F. F. Xu, Z. Wang, Z. Jiang, and G. Neubig. Docprompting: Generating code by retrieving the docs. _arXiv preprint arXiv:2207.05987_ , 2022. 

13 

## **A Details for Benchmark Curation** 

In this section, we introduce the benchmark datasets and the curation process for RAG evaluation. This benchmark datasets are derived from existing open-domain question answering (ODQA) datasets, including RobustQA [9], KIWI [51], ClapNQ [34], and NovelQA [47]. However, most of the ground truth answers in existing ODQA datasets are short answers, while the answers provided by modern LLM-based RAG systems tend to be long-form answers. Therefore, we repurpose the ODQA datasets by eliminating overly simple questions and converting the short answers into long-form answers to match the capabilities of current RAG systems. The statistics of the benchmark are summarized in Tab. 1. In the rest of this section, we describe the datasets we use and the curation process for each domain. 

## **A.1 Data Sources** 

**RobustQA** We choose 7 domains from RobustQA’s collection of datasets: Biomedical, Finance, Lifestyle, Recreation, Technology, Science, and Writing. For the Biomedical domain, following RobustQA, we employ the BioASQ [43] dataset, which contains human expert-written questionanswer pairs and ground truth documents based on abstracts of articles from PubMed. We use the test sets for Task b from 2014 to 2023 and the corpus of v.2022 to construct the benchmark. We keep QA pairs whose answers are relatively long (more than 50 words), obtaining 511 QA pairs for the biomedical domain. The other 6 domains are sourced from FiQA [26] and LoTTE [36], each of their question is annotated with a list of short answers that are spans of ground truth passages. We convert the short answers to long-form answers using GPT-4 and only keep the generated answers with no hallucinations, as checked by RefChecker. Finally, we sample 500 examples for each domain. 

**ClapNQ** Derived from NaturalQuestions (NQ) [17], an ODQA dataset based on Wikipedia, ClapNQ has long-form answers annotated for a subset of NQ for evaluating RAG. We employ the dev set of ClapNQ in our benchmark and take the annotated long-form answers as the ground truth. 

**KIWI** is constructed by asking LLMs research questions about a set of NLP papers and guiding the LLMs to reach satisfactory long-form answers. The authors validated the quality of the generated answers by rating them as “good”, “neutral”, or “bad”. We take the answers labeled “good” as the ground truth answers and query the full text of the papers from S2ORC [23] as the corpus. As a result, we obtain 71 QA pairs and 429 papers as the corpus. 

**NovelQA** is a benchmark for question answering over long novels containing over 100K tokens on average. Originally designed for benchmarking long-context LLMs, we repurpose it for evaluating RAG. In contrast with the other domains, each question in NovelQA is associated with a single novel, so when we use this dataset for RAG, we constrain the retrieval to within the corresponding novel. We select 19 copyright-free novels and convert the corresponding short answers to long-form answers following the same process for RobustQA. 

## **A.2 Long-form Answer Generation** 

We employ GPT-4 ( `gpt-4-turbo-2024-04-09` ) to convert the human annotated short answers to long-form answers in the dataset of RobustQA and NovelQA. For RobustQA, the short answers are spans of the annotated ground truth passages, we take all the annotated short answers and the corresponding passages in the prompt and ask GPT-4 to convert them to one single long-form answer. For NovelQA, we take the human written evidences as the ground truth passage content and the human written short answers for the long-form answer generation. The prompt is shown in Fig. 2. 

For quality control, we ask GPT-4 to generate the passage IDs associated with the long-form answer. We use RefChecker to check whether all the claims of a long-form answer are entailed by these passages, and we only keep the long-form answers that meet this criteria. The RefChecker we used here are described in Appendix G. 

14 

```
Here is a question and human annotated short answers, the short
answers are extracted from a passage, you should help me to
convert the short answers to a long-form answer based on the
provided passage. There could be more than one annotation, so you
should identify the best answers and merge them. The long-form
answer should only depend on the provide information, you should
not hallucinate anything.
### Question
{question}
### Annotations
#### Annotation 1
[Passage ID]: {passage_id}
[Passage Content]: {passage_content}
[Short Answers]:
{answers}
#### Annotation 2
[Passage ID]: {passage_id}
[Passage Content]: {passage_content}
[Short Answers]:
{answers}
...
Your should output the converted long-form answer and the passage
IDs you used. Always follow this format for your response:
[Long-form answer]: <the content of the long-form answer>
[Passage IDs]: ID1, ID2, ...
```

Figure 2: The prompt used for converting short answers to long-form answers for the domains of Novel, Finance, Lifestyle, Recreation, Technology, Science, and Writing. 

## **A.3 Corpus Downsampling for Science and Biomedical Domains** 

In addition to long-form answer generation, we also perform downsampling for the corpora of Science and Biomedical domains as they are much larger than the others, with over 1 million documents each. Building indexes for a dense retriever is very costly for large corpora, so we downsample these domains to lower the evaluation cost for the community. For the biomedical domain, we first use BM25 retriever to obtain top 400 documents for each question. The subsampled corpus is formed by combining all documents from the retriever with annotated relevant documents from the datasets. Based on our initial study, we observe that the BM25 retriever yeild competitive performance against the dense retriever, so we decide to only use the BM25 retriever for downsampling purpose to save compuation cost. For the science domain, we leverage both the BM25 retriever and e5-mistral7b-instruct based dense retriever to obtain document candidates. Specifically, we retrieve the top 200 documents from both retrievers (400 documents in total before deduplication). Similarly, the combination of all documents from the retrievers and annotated relevant documents from datasets forms the downsampled corpus. 

## **A.4 License of The Datasets** 

The annotations from RobustQA, ClapNQ and NovelQA are under Apache-2.0 License. The corpora of Finance and annotations of KIWI are under CC-BY-SA-4.0. BioASQ is under CC BY 2.5 license. The license for the corpora of LoTTE are not specified. 

## **B The complete formula for all metrics** 

Denote the model response as _m_ , the ground truth answer as _gt_ , and the retrieved chunks as _{_ chunk _j}_ . Leveraging RefChecker, we decompose the text into a set of claims _{ci}_ and assess whether a specific claim _ci_ can entail ( _∈_ ) or not entail ( _∈/_ ) a given reference text _Ref_ , where _Ref_ may represent _m_ , _gt_ , or _{_ chunk _j}_ . We assign an entailment label to each ground-truth claim relative to a chunk, and subsequently classify these chunks into relevant chunks _{_ r-chunk _j}_ and irrelevant chunks 

15 

_{_ irr-chunk _j}_ . Specifically, chunk _j_ is considered relevant if it contains at least one claim _c_[(] _i[gt]_[)] such that _c_[(] _i[gt]_[)] _∈_ chunk _j_ . 

In accordance with the definitions provided in Section 3.3, we compute each metric using the following formulations: 

## **B.1 Overall Metrics** 


![](raw/papers/2_LLM_NLP/LLM_paper_list/rag/nips_2024_RAGChecker_A_Fine-grained_Framework_for_Diagnosing_Retrieval-Augmented_Generation_images/nips_2024_RAGChecker_A_Fine-grained_Framework_for_Diagnosing_Retrieval-Augmented_Generation.pdf-0016-03.png)


## **B.2 Retriever Metrics** 


![](raw/papers/2_LLM_NLP/LLM_paper_list/rag/nips_2024_RAGChecker_A_Fine-grained_Framework_for_Diagnosing_Retrieval-Augmented_Generation_images/nips_2024_RAGChecker_A_Fine-grained_Framework_for_Diagnosing_Retrieval-Augmented_Generation.pdf-0016-05.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/rag/nips_2024_RAGChecker_A_Fine-grained_Framework_for_Diagnosing_Retrieval-Augmented_Generation_images/nips_2024_RAGChecker_A_Fine-grained_Framework_for_Diagnosing_Retrieval-Augmented_Generation.pdf-0016-06.png)


## **B.3 Generator Metrics** 


![](raw/papers/2_LLM_NLP/LLM_paper_list/rag/nips_2024_RAGChecker_A_Fine-grained_Framework_for_Diagnosing_Retrieval-Augmented_Generation_images/nips_2024_RAGChecker_A_Fine-grained_Framework_for_Diagnosing_Retrieval-Augmented_Generation.pdf-0016-08.png)


## **C Details of Meta Evaluation** 

In the meta evaluation, we ask 10 annotators compare two responses from the RAG system for each instance in the meta evaluation dataset. Seven of the annotators are in-house annotators, and three of them are graduate students. We pay the students 15 USD per hour and totally cost 255 dollars. 

Annotators are required to choose their preference from five options: significantly better, slightly better, tie, slightly worse, or significantly worse. The annotation is based on three metrics: correctness, completeness, and overall assessment. The annotation interface with instructions are shown in Fig. 3 

To make sure the human evaluation to be agnostic to specific evaluation metrics, we provide the annotators with a detailed annotation guideline which contains detailed instruction and 5 examples. 

16 


![](raw/papers/2_LLM_NLP/LLM_paper_list/rag/nips_2024_RAGChecker_A_Fine-grained_Framework_for_Diagnosing_Retrieval-Augmented_Generation_images/nips_2024_RAGChecker_A_Fine-grained_Framework_for_Diagnosing_Retrieval-Augmented_Generation.pdf-0017-00.png)


Figure 3: The human annotation interface and instructions of the meta evaluation dataset. 

In the UI of the annotation tool, each response is shown with critiques generated by GPT-4 and we ask the annotators to refer to the content of the response and the critiques for labeling. The critiques are generated by prompting GPT-4 to compare the response with ground truth answer to ease the annotation job. In addition, each of the example the guideline are shown with a human-written explanation for the labeling. 

The 10 metrics included in the meta evaluation are selected from Trulens [6], RAGAS [5], ARES [35] and CRUD-RAG [25] as explained in Sec. 4.2. Their descriptions are summarized in Tab. 4. As a supplement of Tab. 2, the full correlation results of meta evaluation is shown in Tab. 5. For a detailed comparison between RAGCHECKER and the strongest baseline metric, RAGAS Answer Similarity, we plot the prediction score distribution of two metrics in Fig. 4. From the prediction score distribution and the mean line (dashed line) of the plot, we can observe a stronger correlation of RAGCHECKER than RAGAS Answer Similarity. 

## **D Details of the Experiment Setup** 

**Models in Baseline RAG Systems** We use the version of `e5-mistral-7b-instruct` for the E5-Mistral retriever. For the generators, we use `gpt-4-turbo-2024-04-09` version for GPT-4, `Llama3-8B-Instruct` for Llama3-8B and `Llama3-70B-Instruct` for Llama3-70B, and `Mixtral-8x7B-Instruct-v0.1` for Mixtral-8x7B. 

We adopt OpenSearch[6] as the tool to implement the inverted index for BM25 and the approximate KNN search for dense retrieval. We use a g5.48xlarge instance with 8 NVIDIA A10G GPUs on AWS for inference of open-source models. We split documents in the corpus to chunks of 300 tokens with an overlap ratio of 0.2 by default. We use the tokenizer of E5-Mistral for both retrievers to control the chunking. For each query, top-20 chunks ranked by retrievers are used as context for LLM generation. The default prompt for all generators is shown in Fig. 5. We set the generation temperature to 0.0 (deterministic) and the maximum generation length to 2,048 tokens when calling proprietary LLMs. 

## **E Detailed Experiment Results** 

The detailed evaluation results for all our benchmark datasets can be found in Tab. 6 to Tab. 15. 

> 6 `https://opensearch.org/` 

17 

Table 4: Summary of the metrics included in the meta evaluation. 

||||
|---|---|---|
|**Baseline**|**Metric**|**Description**|
|TruLens|Groundedness|Assesses the overlap between each statement in the response and<br>the provided context using an LLM.|
||Answer Relevance|Prompts an LLM to give a relevance score between the response<br>and question.|
|RAGAS|Faithfulness|Measures the proportion of claims in the response that can be<br>inferred from the context.|
||Answer Relevance|Computes the mean cosine similarity between the original<br>question and a series of LLM-generated questions derived from<br>the response and context.|
||Answer Similarity|Measures the semantic similarity between the response and the<br>ground truth answer based on text-embedding-ada-002 [28].|
||Answer Correctness|Quantifes both the semantic similarity and the factual overlap<br>between the response and the ground truth answer.|
|ARES|Answer Faithfulness|Prompts an LLM to determine whether the response is faithful to<br>the context.|
||Answer Relevance|Prompts an LLM to measure whether the response addresses all<br>aspects of the question and provides only correct information<br>from the context.|
|CRUD-RAG|Recall|Computes the ratio of all questions generated from ground truth<br>answers that can be answered by response.|
||Precision|Evaluates if the generated text is accurate and consistent with the<br>ground truth answer.|



Table 5: Full Correlation results with Human Evaluation of Correctness, Completeness, and Overall Assessment 

|Assessment||||
|---|---|---|---|
|**Baseline**<br>**Metric**|**Correctness**<br>**Pearson**<br>**Spearman**|**Completeness**<br>**Pearson**<br>**Spearman**|**Overall Assessment**<br>**Pearson**<br>**Spearman**|
|BLEU<br>BLEU-avg<br>ROUGE<br>ROUGE-L<br>BERTScore<br>BERTScore<br>TruLens<br>Groundedness<br>TruLens<br>Answer Relevance<br>ARES<br>Answer Relevance<br>ARES<br>Answer Faithfulness<br>RAGAS<br>Faithfulness<br>RAGAS<br>Answer Correctness<br>RAGAS<br>Answer Similarity<br>RAGAS<br>Answer Relevance<br>CRUD-RAG<br>Precision<br>CRUD-RAG<br>Recall<br>RAGChecker<br>Same metric as human|38.89<br>35.32<br>31.75<br>31.72<br>30.34<br>27.05<br>21.11<br>18.21<br>35.01<br>27.37<br>18.63<br>16.84<br>9.46<br>7.60<br>8.22<br>7.53<br>39.11<br>36.30<br>41.07<br>43.21<br>11.59<br>8.19<br>20.73<br>15.67<br>30.93<br>27.13<br>**49.66**<br>**46.95**|32.13<br>21.85<br>47.88<br>45.67<br>37.93<br>40.05<br>14.01<br>6.02<br>37.24<br>37.91<br>20.13<br>18.13<br>10.25<br>8.99<br>4.90<br>1.19<br>36.42<br>36.04<br>53.16<br>**61.35**<br>9.39<br>13.57<br>25.58<br>20.33<br>45.11<br>43.76<br>**60.67**<br>58.11|35.14<br>29.42<br>43.10<br>43.21<br>33.51<br>35.57<br>19.45<br>14.42<br>35.15<br>33.59<br>17.81<br>16.26<br>8.80<br>7.58<br>7.83<br>5.55<br>38.01<br>37.14<br>48.31<br>57.23<br>10.27<br>11.83<br>25.59<br>19.63<br>41.25<br>39.71<br>**61.93**<br>**60.90**|
|Human<br>Annotator sanity check|63.67<br>59.19|71.91<br>68.36|70.09<br>68.89|



18 


![](raw/papers/2_LLM_NLP/LLM_paper_list/rag/nips_2024_RAGChecker_A_Fine-grained_Framework_for_Diagnosing_Retrieval-Augmented_Generation_images/nips_2024_RAGChecker_A_Fine-grained_Framework_for_Diagnosing_Retrieval-Augmented_Generation.pdf-0019-00.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/rag/nips_2024_RAGChecker_A_Fine-grained_Framework_for_Diagnosing_Retrieval-Augmented_Generation_images/nips_2024_RAGChecker_A_Fine-grained_Framework_for_Diagnosing_Retrieval-Augmented_Generation.pdf-0019-01.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/rag/nips_2024_RAGChecker_A_Fine-grained_Framework_for_Diagnosing_Retrieval-Augmented_Generation_images/nips_2024_RAGChecker_A_Fine-grained_Framework_for_Diagnosing_Retrieval-Augmented_Generation.pdf-0019-02.png)


Figure 4: Comparison of prediction score distribution between RAGCHECKER and RAGAS Answer Similarity. Each point in the plot represents an instance in the meta evaluation dataset, where the x-axis is the human preference label under corresponding aspect and y-axis is the prediction score of RAGCHECKER and RAGAS Answer Similarity. The distribution of prediction score is represented by the colored area and the dashed line is the mean line. 

19 

```
Please answer the given question based on the context.
<context>
<content>
{chunk_1}
</content>
<content>
{chunk_2}
</content>
...
<content>
{chunk_k}
</content>
</context>
Question: {question}
Please answer the question and tag your answer with <answer></answer>.
```

Figure 5: The default prompt used for response generation in the main experiments for the 8 RAG baseline systems. 

## **F Diagnosis on RAG for Improvements** 

We modify hyper-parameters commonly tuned in RAG systems to observe performance variance under the metrics defined by RAGCHECKER . We focus on how RAGCHECKER explains this variance and provides tuning suggestions for improvements on certain aspects. In this section, we evaluate three RAG baselines (BM25_GPT-4, E5-Mistral_GPT-4, and E5-Mistral_Llama3-70B) across three domains with increasing difficulty: Writing, Finance, and KIWI. We use our default settings (Appendix D) in main experiments as controls. We experiment with different numbers of chunks selected as context _k ∈_ {5,10,20}, different chunk size {150,300,600}[7] , different chunk overlap ratio {0.0,0.2,0.4}, and different generation prompts. 

**More Context Enhances Faithfulness** Top-k selection and chunk size both balance the amount of noise and useful information presented to the generator, but in different manners. Corresponding results are demonstrated in Fig. 6 and Fig. 7. Increasing _k_ adds more context that could be less relevant, while increasing chunk size provides more surrounding context of relevant facts. Thus _context precision_ decreases with larger _k_ but increases with larger chunk sizes. Despite this, they both lead to better _claim recall_ in Retrieval. 

Generators tend to be more faithful when provided with more context, though this trend is less pronounced for Llama3, which already exhibits high faithfulness. _Context utilization_ generally worsens with more context due to increasing noise, leading to higher _relevant noise sensitivity_ . 

Overall, the end-to-end RAG performance is slightly better with more context, primarily due to improved _recall_ . We recommend moderately increasing the two parameters for more faithful generation, noting that saturation occurs at high values as the amount of useful information is limited. Given a limited context length, a larger chunk size with a smaller k is preferred, especially for easier datasets (Finance, Writing). This is evident when comparing a chunk size of 150 with _k_ =20 against a chunk size of 300 with _k_ =10. 

**Explicit Requirements in Prompts Affect Generation Preferences** To validate the effect of the generation prompt, we added more detailed requirements to guide the generation for better _faithfulness_ , _context utilization_ , and lower _noise sensitivity_ . The optimized prompt is shown in Fig. 9. 

As shown in Fig. 8, we observed a general improvement in _context utilization_ . However, as a counterpart to _context utilization_ , _noise sensitivity_ generally worsened. It demonstrates the difficulty of meeting all prompt requirements when there are subtle tension between them. 

For the two generators, GPT-4 generally showes improvements in metrics related to faithfulness ( _hallucination_ , _self-knowledge_ , _faithfulness_ ), whereas Llama3 does not exhibit the same behavior. This aligns with our previous observation (Sec. 4.3) that Llama3 already performs well on _faithfulness_ , 

> 7We omit chunk size of 600 for E5-Mistral_Llama3-70B due to the limited 8K context window of Llama3. 

20 


![](raw/papers/2_LLM_NLP/LLM_paper_list/rag/nips_2024_RAGChecker_A_Fine-grained_Framework_for_Diagnosing_Retrieval-Augmented_Generation_images/nips_2024_RAGChecker_A_Fine-grained_Framework_for_Diagnosing_Retrieval-Augmented_Generation.pdf-0021-00.png)


**----- Start of picture text -----**<br>
CU CU CU 5<br>(62.8~81.8) (45.6~80.9) (41.7~69.8) 10<br>Faith. CP Faith. CP Faith. CP 20<br>(80.9~100) (63.3~82.4) (64.7~100) (47.4~82.6) (71.9~100) (67.7~95.8)<br>100 - Hallu. CR 100 - Hallu. CR 100 - Hallu. CR<br>(80.9~100) (72.6~91.7) (64.7~100) (51.0~86.3) (71.9~100) (36.6~64.7)<br>100 - SK 100 - NS(II) 100 - SK 100 - NS(II) 100 - SK 100 - NS(II)<br>(80.9~100) (80.9~100) (64.7~100) (64.7~100) (71.9~100) (71.9~100)<br>100 - NS(I) 100 - NS(I) 100 - NS(I)<br>(70.8~89.9) (53.7~89.0) (39.4~67.5)<br>CU CU CU<br>(62.8~81.8) (45.6~80.9) (41.7~69.8)<br>Faith. CP Faith. CP Faith. CP<br>(80.9~100) (63.3~82.4) (64.7~100) (47.4~82.6) (71.9~100) (67.7~95.8)<br>100 - Hallu. CR 100 - Hallu. CR 100 - Hallu. CR<br>(80.9~100) (72.6~91.7) (64.7~100) (51.0~86.3) (71.9~100) (36.6~64.7)<br>100 - SK 100 - NS(II) 100 - SK 100 - NS(II) 100 - SK 100 - NS(II)<br>(80.9~100) (80.9~100) (64.7~100) (64.7~100) (71.9~100) (71.9~100)<br>100 - NS(I) 100 - NS(I) 100 - NS(I)<br>(70.8~89.9) (53.7~89.0) (39.4~67.5)<br>CU CU CU<br>(62.8~81.8) (45.6~80.9) (41.7~69.8)<br>Faith. CP Faith. CP Faith. CP<br>(80.9~100) (63.3~82.4) (64.7~100) (47.4~82.6) (71.9~100) (67.7~95.8)<br>100 - Hallu. CR 100 - Hallu. CR 100 - Hallu. CR<br>(80.9~100) (72.6~91.7) (64.7~100) (51.0~86.3) (71.9~100) (36.6~64.7)<br>100 - SK 100 - NS(II) 100 - SK 100 - NS(II) 100 - SK 100 - NS(II)<br>(80.9~100) (80.9~100) (64.7~100) (64.7~100) (71.9~100) (71.9~100)<br>100 - NS(I) 100 - NS(I) 100 - NS(I)<br>(70.8~89.9) (53.7~89.0) (39.4~67.5)<br>Writing Finance KIWI<br>BM25_GPT-4<br>E5-Mistral_GPT-4<br>E5-Mistral_Llama3-70B<br>**----- End of picture text -----**<br>


Figure 6: Diagnosis on Top-k Selection 

while GPT-4 tends to rely on self-knowledge without explicit requirements. Consequently, there is a steady improvement in overall _F1_ for GPT-4 when switched to the optimized prompt, while the difference for Llama3 is negligible. 

RAG builders can optimize prompts by combining performance on modular metrics provided by RAGCHECKER with user preferences and generator capabilities on different aspects. 

**Chunk Overlap Does Not Matter a Lot** Chunk overlap ratio between adjacent chunks is usually set to be non-zero to help the generator better utilize surrounding information and identify chunks with coherent logic, thus alleviating the impact of hard splits in significant semantics. According to our results in Fig. 10, higher overlap ratios generally lead to improved _context precision_ . However, this does not necessarily translate to an increase in the total amount of useful information retrieved. This phenomenon can be attributed to the retrieval of more chunks that contain the same segment of useful information. Consequently, we observed that overlap ratio adjustments do not have a significant impact on other performance metrics in a consistent and obvious manner. This suggests that the overlap ratio may not require extensive tuning in practice. 

## **G Performance Validation of RefChecker with Llama3 Extractor and Checker** 

We use Llama3-70B-Instruct for the extractor and checker in RefChecker. To validate the effectiveness of this combination, we test its performance on the RefChecker benchmark. As shown in Tab. 16, Llama 3 based RefChecker outperforms the best purely open-sourced combinations reported in the RefChecker paper in all the three context settings. 

21 


![](raw/papers/2_LLM_NLP/LLM_paper_list/rag/nips_2024_RAGChecker_A_Fine-grained_Framework_for_Diagnosing_Retrieval-Augmented_Generation_images/nips_2024_RAGChecker_A_Fine-grained_Framework_for_Diagnosing_Retrieval-Augmented_Generation.pdf-0022-00.png)


**----- Start of picture text -----**<br>
CU CU CU 150<br>(64.3~75.8) (47.8~75.0) (35.0~63.2) 300<br>Faith. CP Faith. CP Faith. CP 600<br>(88.4~99.9) (58.7~70.2) (72.8~100) (44.3~71.5) (71.7~100) (63.8~92.1)<br>100 - Hallu. CR 100 - Hallu. CR 100 - Hallu. CR<br>(88.5~100) (82.1~93.6) (72.8~100) (61.4~88.6) (71.7~100) (47.6~75.8)<br>100 - SK 100 - NS(II) 100 - SK 100 - NS(II) 100 - SK 100 - NS(II)<br>(88.5~100) (88.5~100) (72.8~100) (72.8~100) (71.7~100) (71.7~100)<br>100 - NS(I) 100 - NS(I) 100 - NS(I)<br>(73.6~85.1) (57.0~84.2) (42.8~71.1)<br>CU CU CU<br>(64.3~75.8) (47.8~75.0) (35.0~63.2)<br>Faith. CP Faith. CP Faith. CP<br>(88.4~99.9) (58.7~70.2) (72.8~100) (44.3~71.5) (71.7~100) (63.8~92.1)<br>100 - Hallu. CR 100 - Hallu. CR 100 - Hallu. CR<br>(88.5~100) (82.1~93.6) (72.8~100) (61.4~88.6) (71.7~100) (47.6~75.8)<br>100 - SK 100 - NS(II) 100 - SK 100 - NS(II) 100 - SK 100 - NS(II)<br>(88.5~100) (88.5~100) (72.8~100) (72.8~100) (71.7~100) (71.7~100)<br>100 - NS(I) 100 - NS(I) 100 - NS(I)<br>(73.6~85.1) (57.0~84.2) (42.8~71.1)<br>CU CU CU<br>(64.3~75.8) (47.8~75.0) (35.0~63.2)<br>Faith. CP Faith. CP Faith. CP<br>(88.4~99.9) (58.7~70.2) (72.8~100) (44.3~71.5) (71.7~100) (63.8~92.1)<br>100 - Hallu. CR 100 - Hallu. CR 100 - Hallu. CR<br>(88.5~100) (82.1~93.6) (72.8~100) (61.4~88.6) (71.7~100) (47.6~75.8)<br>100 - SK 100 - NS(II) 100 - SK 100 - NS(II) 100 - SK 100 - NS(II)<br>(88.5~100) (88.5~100) (72.8~100) (72.8~100) (71.7~100) (71.7~100)<br>100 - NS(I) 100 - NS(I) 100 - NS(I)<br>(73.6~85.1) (57.0~84.2) (42.8~71.1)<br>Writing Finance KIWI<br>BM25_GPT-4<br>E5-Mistral_GPT-4<br>E5-Mistral_Llama3-70B<br>**----- End of picture text -----**<br>


Figure 7: Diagnosis on Chunk Size 

## **H Limitations** 

While RAGCHECKER provides a comprehensive evaluation framework for RAG systems, it has a few limitations that should be acknowledged and addressed in future research. 

First, the diagnostic metrics for the retriever component are less insightful compared to those for the generator. The retrieval metrics primarily focus on the recall of ground truth claims and precision of retrieved context, but they may not fully capture the nuances and complexities of the retrieval process. Developing more sophisticated metrics that consider factors such as the information density, diversity and coherence of the retrieved context could provide deeper insights into the retriever’s performance. 

Second, the metrics proposed in RAGCHECKER do not differentiate between `Neutral` and `Contradiction` checking results from RefChecker when evaluating the generated responses. These two types of results may have different impacts on the final response quality, and treating them equally could lead to an incomplete assessment. Future work should explore ways to incorporate the distinction between neutral and contradiction results into the evaluation metrics, potentially assigning different weights or penalties based on their severity. 

Finally, the evaluation benchmark used in this study is curated based on existing text-only datasets and is limited to English queries and corpus. While this allows for a focused evaluation of RAG systems, it may not fully represent the diverse range of tasks and languages that RAG systems can be applied to. Expanding the benchmark to include datasets from different modalities (e.g., images, audio) and languages would provide a more comprehensive assessment of RAG systems’ capabilities and generalization. Additionally, creating benchmark datasets specifically designed for evaluating RAG systems, rather than repurposing existing ones, could help to better capture the unique challenges and requirements of this task. 

22 


![](raw/papers/2_LLM_NLP/LLM_paper_list/rag/nips_2024_RAGChecker_A_Fine-grained_Framework_for_Diagnosing_Retrieval-Augmented_Generation_images/nips_2024_RAGChecker_A_Fine-grained_Framework_for_Diagnosing_Retrieval-Augmented_Generation.pdf-0023-00.png)


**----- Start of picture text -----**<br>
basic<br>100 - Hallu. Faith. 100 - Hallu. Faith. 100 - Hallu. Faith. optim.<br>(92.1~100) (90.5~98.4) (82.6~97.1) (79.9~94.4) (88.2~100) (88.2~100)<br>100 - SK CU 100 - SK CU 100 - SK CU<br>(92.1~100) (69.0~76.9) (85.5~100) (57.5~72.1) (88.2~100) (45.3~57.0)<br>100 - NS(I) 100 - NS(II) 100 - NS(I) 100 - NS(II) 100 - NS(I) 100 - NS(II)<br>(74.6~82.5) (92.1~100) (60.6~75.1) (85.5~100) (46.9~58.7) (88.2~100)<br>100 - Hallu. Faith. 100 - Hallu. Faith. 100 - Hallu. Faith.<br>(92.1~100) (90.5~98.4) (82.6~97.1) (79.9~94.4) (88.2~100) (88.2~100)<br>100 - SK CU 100 - SK CU 100 - SK CU<br>(92.1~100) (69.0~76.9) (85.5~100) (57.5~72.1) (88.2~100) (45.3~57.0)<br>100 - NS(I) 100 - NS(II) 100 - NS(I) 100 - NS(II) 100 - NS(I) 100 - NS(II)<br>(74.6~82.5) (92.1~100) (60.6~75.1) (85.5~100) (46.9~58.7) (88.2~100)<br>100 - Hallu. Faith. 100 - Hallu. Faith. 100 - Hallu. Faith.<br>(92.1~100) (90.5~98.4) (82.6~97.1) (79.9~94.4) (88.2~100) (88.2~100)<br>100 - SK CU 100 - SK CU 100 - SK CU<br>(92.1~100) (69.0~76.9) (85.5~100) (57.5~72.1) (88.2~100) (45.3~57.0)<br>100 - NS(I) 100 - NS(II) 100 - NS(I) 100 - NS(II) 100 - NS(I) 100 - NS(II)<br>(74.6~82.5) (92.1~100) (60.6~75.1) (85.5~100) (46.9~58.7) (88.2~100)<br>Writing Finance KIWI<br>BM25_GPT-4<br>E5-Mistral_GPT-4<br>E5-Mistral_Llama3-70B<br>**----- End of picture text -----**<br>


Figure 8: Diagnosis on Generation Prompts 

By refining the diagnostic metrics, incorporating the impact of different checking results, and expanding the evaluation benchmark, researchers can gain an even more comprehensive understanding of RAG systems’ performance and identify targeted areas for improvement. 

## **I Potential Negative Societal Impacts** 

The RAGCHECKER evaluation framework, while beneficial for assessing RAG systems, could inadvertently lead to several negative societal impacts. There is a risk that developers may focus on optimizing for RAGCHECKER’s specific metrics to the detriment of broader utility and ethical considerations. The computational and financial requirements to meet RAGCHECKER standards could disadvantage smaller organizations, potentially centralizing innovation among well-resourced entities. Moreover, an overreliance on quantitative measures might neglect qualitative factors like user experience and ethical implications. 

Table 6: Evaluation results for different RAG systems on ClapNQ dataset 

|RAG systems|Overall|Retriever|Generator<br>#Claim<br>CU_↑_<br>NS(I)_↓_<br>NS(II)_↓_<br>Hallu._↓_<br>SK_↓_<br>Faith._↑_|
|---|---|---|---|
||Prec._↑_<br>Rec._↑_<br>F1_↑_|CR_↑_<br>CP_↑_||
|BM25_GPT-4<br>BM25_Llama3-8b<br>BM25_Llama3-70b<br>BM25_Mixtral-8x7b|56.9<br>50.0<br>46.7<br>49.6<br>48.6<br>42.2<br>56.9<br>48.7<br>45.3<br>47.9<br>49.6<br>42.1|81.1<br>41.3<br>81.1<br>41.3<br>81.1<br>41.3<br>81.1<br>41.3|56.4<br>29.4<br>5.9<br>7.5<br>2.2<br>90.3<br>8<br>55.2<br>31.9<br>7.5<br>10.8<br>2.0<br>87.2<br>10<br>55.7<br>30.1<br>7.1<br>5.9<br>1.6<br>92.4<br>7<br>55.8<br>36.9<br>7.3<br>6.9<br>2.3<br>90.9<br>9|
|E5-Mistral_GPT-4<br>E5-Mistral_Llama3-8b<br>E5-Mistral_Llama3-70b<br>E5-Mistral_Mixtral-8x7b|59.7<br>51.1<br>47.9<br>50.4<br>50.9<br>43.5<br>58.7<br>52.8<br>48.1<br>51.1<br>54.4<br>45.7|81.5<br>43.6<br>81.5<br>43.6<br>81.5<br>43.6<br>81.5<br>43.6|59.9<br>31.1<br>3.8<br>5.4<br>2.3<br>92.3<br>9<br>59.4<br>33.2<br>6.4<br>10.0<br>1.5<br>88.5<br>10<br>61.4<br>32.0<br>5.1<br>4.2<br>2.1<br>93.7<br>8<br>63.2<br>37.0<br>5.2<br>5.5<br>1.5<br>93.0<br>10|



23 

```
YouareanaccurateandreliableAIassistantcapableofanswering
questionsusingexternaldocuments.Alwaysbefaithfultotheprovided
documentsandleveragerelevant,accurateinformationfromthemas
muchaspossible.Beawarethatexternaldocumentsmightcontainnoisy
orfactuallyincorrectdata.Applycriticalreasoningtodiscernand
usethecorrectinformationfromthesesources.
<context>
<content>
{chunk_1}
</content>
<content>
{chunk_2}
</content>
...
<content>
{chunk_k}
</content>
</context>
Question:{question}
Pleaseanswerthequestionandtagyouranswerwith<answer></answer>.
```

Figure 9: The optimized prompt for response generation. In this prompt, we explicitly instruct the LLMs to be faithful to the context and identify relevant information as possible. 

Table 7: Evaluation results for different RAG systems on NovelQA dataset 

|RAG systems|Overall|Retriever|Generator<br>#Claim<br>CU_↑_<br>NS(I)_↓_<br>NS(II)_↓_<br>Hallu._↓_<br>SK_↓_<br>Faith._↑_|
|---|---|---|---|
||Prec._↑_<br>Rec._↑_<br>F1_↑_|CR_↑_<br>CP_↑_||
|BM25_GPT-4<br>BM25_Llama3-8b<br>BM25_Llama3-70b<br>BM25_Mixtral-8x7b|71.0<br>56.2<br>56.4<br>60.2<br>47.8<br>45.9<br>65.0<br>51.8<br>51.9<br>56.0<br>50.2<br>46.0|82.1<br>42.6<br>82.1<br>42.6<br>82.1<br>42.6<br>82.1<br>42.6|64.9<br>17.6<br>5.4<br>6.1<br>2.2<br>91.7<br>4<br>55.2<br>23.1<br>7.1<br>9.6<br>1.5<br>88.8<br>3<br>59.6<br>21.4<br>7.5<br>6.1<br>2.1<br>91.8<br>3<br>58.4<br>24.8<br>6.4<br>10.9<br>2.3<br>86.8<br>4|
|E5-Mistral_GPT-4<br>E5-Mistral_Llama3-8b<br>E5-Mistral_Llama3-70b<br>E5-Mistral_Mixtral-8x7b|69.4<br>56.2<br>55.7<br>58.7<br>48.1<br>45.7<br>64.5<br>50.2<br>49.6<br>54.2<br>48.3<br>43.6|82.7<br>45.1<br>82.7<br>45.1<br>82.7<br>45.1<br>82.7<br>45.1|63.6<br>19.4<br>6.1<br>5.1<br>1.7<br>93.2<br>4<br>55.1<br>23.5<br>8.1<br>9.2<br>1.5<br>89.3<br>3<br>56.9<br>23.7<br>5.5<br>6.0<br>1.5<br>92.4<br>3<br>54.7<br>29.6<br>6.9<br>7.5<br>1.6<br>90.9<br>4|



Table 8: Evaluation results for different RAG systems on RobustQA - Writing dataset 

|RAG systems|Overall|Retriever|Generator<br>#Claim<br>CU_↑_<br>NS(I)_↓_<br>NS(II)_↓_<br>Hallu._↓_<br>SK_↓_<br>Faith._↑_|
|---|---|---|---|
||Prec._↑_<br>Rec._↑_<br>F1_↑_|CR_↑_<br>CP_↑_||
|BM25_GPT-4<br>BM25_Llama3-8b<br>BM25_Llama3-70b<br>BM25_Mixtral-8x7b|76.3<br>63.6<br>66.0<br>65.0<br>59.7<br>57.7<br>72.2<br>62.1<br>63.6<br>67.0<br>60.1<br>59.8|86.3<br>64.3<br>86.3<br>64.3<br>86.3<br>64.3<br>86.3<br>64.3|70.0<br>17.5<br>1.0<br>5.1<br>4.0<br>90.9<br>10<br>66.1<br>26.0<br>1.8<br>6.2<br>2.3<br>91.4<br>10<br>68.4<br>23.1<br>1.5<br>3.2<br>2.2<br>94.7<br>8<br>66.1<br>25.2<br>1.5<br>4.0<br>2.2<br>93.8<br>8|
|E5-Mistral_GPT-4<br>E5-Mistral_Llama3-8b<br>E5-Mistral_Llama3-70b<br>E5-Mistral_Mixtral-8x7b|77.1<br>65.0<br>67.3<br>67.4<br>62.8<br>60.6<br>73.1<br>65.7<br>66.2<br>66.4<br>62.2<br>61.3|91.7<br>66.3<br>91.7<br>66.3<br>91.7<br>66.3<br>91.7<br>66.3|69.0<br>17.9<br>1.2<br>3.8<br>1.3<br>94.9<br>10<br>66.8<br>25.5<br>2.2<br>4.5<br>0.6<br>95.0<br>9<br>70.1<br>23.5<br>1.9<br>1.5<br>0.5<br>98.0<br>9<br>66.4<br>26.3<br>2.0<br>3.2<br>0.4<br>96.4<br>9|



24 


![](raw/papers/2_LLM_NLP/LLM_paper_list/rag/nips_2024_RAGChecker_A_Fine-grained_Framework_for_Diagnosing_Retrieval-Augmented_Generation_images/nips_2024_RAGChecker_A_Fine-grained_Framework_for_Diagnosing_Retrieval-Augmented_Generation.pdf-0025-00.png)


**----- Start of picture text -----**<br>
CU CU CU 0.0<br>(66.6~73.7) (51.6~69.1) (39.3~55.2) 0.2<br>Faith. CP Faith. CP Faith. CP 0.4<br>(90.9~98.0) (61.5~68.6) (78.9~96.3) (51.6~69.0) (84.1~100) (71.2~87.1)<br>100 - Hallu. CR 100 - Hallu. CR 100 - Hallu. CR<br>(92.9~100) (86.3~93.4) (81.6~99.1) (69.4~86.8) (84.1~100) (57.8~73.7)<br>100 - SK 100 - NS(II) 100 - SK 100 - NS(II) 100 - SK 100 - NS(II)<br>(92.9~100) (92.9~100) (82.6~100) (82.6~100) (84.1~100) (84.1~100)<br>100 - NS(I) 100 - NS(I) 100 - NS(I)<br>(75.8~83.0) (60.1~77.5) (45.8~61.8)<br>CU CU CU<br>(66.6~73.7) (51.6~69.1) (39.3~55.2)<br>Faith. CP Faith. CP Faith. CP<br>(90.9~98.0) (61.5~68.6) (78.9~96.3) (51.6~69.0) (84.1~100) (71.2~87.1)<br>100 - Hallu. CR 100 - Hallu. CR 100 - Hallu. CR<br>(92.9~100) (86.3~93.4) (81.6~99.1) (69.4~86.8) (84.1~100) (57.8~73.7)<br>100 - SK 100 - NS(II) 100 - SK 100 - NS(II) 100 - SK 100 - NS(II)<br>(92.9~100) (92.9~100) (82.6~100) (82.6~100) (84.1~100) (84.1~100)<br>100 - NS(I) 100 - NS(I) 100 - NS(I)<br>(75.8~83.0) (60.1~77.5) (45.8~61.8)<br>CU CU CU<br>(66.6~73.7) (51.6~69.1) (39.3~55.2)<br>Faith. CP Faith. CP Faith. CP<br>(90.9~98.0) (61.5~68.6) (78.9~96.3) (51.6~69.0) (84.1~100) (71.2~87.1)<br>100 - Hallu. CR 100 - Hallu. CR 100 - Hallu. CR<br>(92.9~100) (86.3~93.4) (81.6~99.1) (69.4~86.8) (84.1~100) (57.8~73.7)<br>100 - SK 100 - NS(II) 100 - SK 100 - NS(II) 100 - SK 100 - NS(II)<br>(92.9~100) (92.9~100) (82.6~100) (82.6~100) (84.1~100) (84.1~100)<br>100 - NS(I) 100 - NS(I) 100 - NS(I)<br>(75.8~83.0) (60.1~77.5) (45.8~61.8)<br>Writing Finance KIWI<br>BM25_GPT-4<br>E5-Mistral_GPT-4<br>E5-Mistral_Llama3-70B<br>**----- End of picture text -----**<br>


Figure 10: Diagnosis on Chunk Overlap Ratio 

Table 9: Evaluation results for different RAG systems on RobustQA - BioASQ dataset 

|RAG systems|Overall|Retriever|Generator<br>#Claim<br>CU_↑_<br>NS(I)_↓_<br>NS(II)_↓_<br>Hallu._↓_<br>SK_↓_<br>Faith._↑_|
|---|---|---|---|
||Prec._↑_<br>Rec._↑_<br>F1_↑_|CR_↑_<br>CP_↑_||
|BM25_GPT-4<br>BM25_Llama3-8b<br>BM25_Llama3-70b<br>BM25_Mixtral-8x7b|66.1<br>43.2<br>46.5<br>64.1<br>34.5<br>38.5<br>70.7<br>35.4<br>42.1<br>60.6<br>43.7<br>45.0|82.1<br>39.4<br>82.1<br>39.4<br>82.1<br>39.4<br>82.1<br>39.4|51.0<br>25.0<br>5.4<br>3.4<br>0.7<br>95.9<br>10<br>41.0<br>21.6<br>5.8<br>5.7<br>0.5<br>93.8<br>7<br>41.8<br>20.3<br>6.0<br>2.9<br>0.6<br>96.4<br>7<br>50.7<br>28.3<br>6.5<br>4.5<br>0.9<br>94.6<br>9|
|E5-Mistral_GPT-4<br>E5-Mistral_Llama3-8b<br>E5-Mistral_Llama3-70b<br>E5-Mistral_Mixtral-8x7b|65.7<br>44.0<br>46.7<br>65.1<br>36.0<br>40.0<br>69.9<br>37.3<br>43.5<br>58.2<br>44.5<br>44.6|84.4<br>45.2<br>84.4<br>45.2<br>84.4<br>45.2<br>84.4<br>45.2|50.5<br>26.2<br>5.9<br>2.2<br>0.3<br>97.5<br>10<br>41.4<br>22.5<br>5.8<br>3.7<br>0.3<br>96.0<br>8<br>43.0<br>22.2<br>5.7<br>2.2<br>0.4<br>97.3<br>7<br>50.3<br>31.6<br>7.0<br>2.9<br>0.7<br>96.4<br>10|



Table 10: Evaluation results for different RAG systems on RobustQA - Finance dataset 

|RAG systems|Overall|Retriever|Generator<br>#Claim<br>CU_↑_<br>NS(I)_↓_<br>NS(II)_↓_<br>Hallu._↓_<br>SK_↓_<br>Faith._↑_|
|---|---|---|---|
||Prec._↑_<br>Rec._↑_<br>F1_↑_|CR_↑_<br>CP_↑_||
|BM25_GPT-4<br>BM25_Llama3-8b<br>BM25_Llama3-70b<br>BM25_Mixtral-8x7b|54.0<br>50.1<br>48.2<br>44.0<br>42.6<br>39.3<br>53.8<br>43.6<br>44.5<br>46.4<br>41.3<br>39.3|69.4<br>52.1<br>69.4<br>52.1<br>69.4<br>52.1<br>69.4<br>52.1|62.3<br>26.8<br>3.9<br>15.4<br>4.7<br>79.9<br>15<br>55.0<br>35.5<br>6.6<br>13.5<br>2.5<br>84.0<br>14<br>56.7<br>34.3<br>5.2<br>6.5<br>2.1<br>91.4<br>10<br>53.0<br>37.4<br>6.4<br>6.7<br>2.2<br>91.1<br>11|
|E5-Mistral_GPT-4<br>E5-Mistral_Llama3-8b<br>E5-Mistral_Llama3-70b<br>E5-Mistral_Mixtral-8x7b|56.0<br>54.6<br>51.8<br>46.9<br>50.1<br>43.9<br>56.0<br>51.4<br>49.9<br>49.0<br>49.0<br>45.0|86.3<br>67.4<br>86.3<br>67.4<br>86.3<br>67.4<br>86.3<br>67.4|60.2<br>31.7<br>2.7<br>9.5<br>1.4<br>89.1<br>15<br>55.5<br>38.4<br>4.6<br>9.5<br>1.1<br>89.4<br>14<br>57.5<br>35.2<br>3.9<br>4.9<br>0.6<br>94.4<br>12<br>54.7<br>39.0<br>5.3<br>3.9<br>0.6<br>95.5<br>12|



25 

Table 11: Evaluation results for different RAG systems on RobustQA - Lifestyle dataset 

|RAG systems|Overall|Retriever|Generator<br>#Claim<br>CU_↑_<br>NS(I)_↓_<br>NS(II)_↓_<br>Hallu._↓_<br>SK_↓_<br>Faith._↑_|
|---|---|---|---|
||Prec._↑_<br>Rec._↑_<br>F1_↑_|CR_↑_<br>CP_↑_||
|BM25_GPT-4<br>BM25_Llama3-8b<br>BM25_Llama3-70b<br>BM25_Mixtral-8x7b|63.3<br>50.5<br>53.0<br>49.7<br>44.8<br>43.8<br>59.6<br>44.4<br>47.5<br>52.8<br>43.5<br>44.1|70.2<br>47.0<br>70.2<br>47.0<br>70.2<br>47.0<br>70.2<br>47.0|64.8<br>24.0<br>2.7<br>10.0<br>6.0<br>84.0<br>12<br>59.2<br>33.2<br>6.1<br>11.1<br>2.2<br>86.8<br>12<br>58.5<br>30.7<br>3.8<br>5.9<br>2.4<br>91.8<br>9<br>56.8<br>34.5<br>4.9<br>6.8<br>2.3<br>90.9<br>10|
|E5-Mistral_GPT-4<br>E5-Mistral_Llama3-8b<br>E5-Mistral_Llama3-70b<br>E5-Mistral_Mixtral-8x7b|66.4<br>57.6<br>58.9<br>54.6<br>55.0<br>51.3<br>64.8<br>56.7<br>57.3<br>56.2<br>51.8<br>50.5|89.7<br>64.0<br>89.7<br>64.0<br>89.7<br>64.0<br>89.7<br>64.0|62.5<br>26.2<br>2.2<br>5.2<br>1.4<br>93.4<br>13<br>59.7<br>34.6<br>4.4<br>6.3<br>0.6<br>93.0<br>14<br>61.7<br>29.3<br>3.3<br>2.6<br>0.7<br>96.7<br>12<br>56.5<br>34.8<br>4.6<br>3.0<br>0.8<br>96.2<br>11|



Table 12: Evaluation results for different RAG systems on RobustQA - Recreation dataset 

|RAG systems|Overall|Retriever|Generator<br>#Claim<br>CU_↑_<br>NS(I)_↓_<br>NS(II)_↓_<br>Hallu._↓_<br>SK_↓_<br>Faith._↑_|
|---|---|---|---|
||Prec._↑_<br>Rec._↑_<br>F1_↑_|CR_↑_<br>CP_↑_||
|BM25_GPT-4<br>BM25_Llama3-8b<br>BM25_Llama3-70b<br>BM25_Mixtral-8x7b|62.9<br>51.9<br>53.1<br>50.6<br>45.1<br>43.1<br>58.1<br>45.7<br>46.5<br>50.5<br>44.3<br>42.4|70.4<br>37.5<br>70.4<br>37.5<br>70.4<br>37.5<br>70.4<br>37.5|65.7<br>21.4<br>4.6<br>11.1<br>5.2<br>83.7<br>11<br>58.1<br>31.0<br>10.1<br>8.1<br>1.8<br>90.2<br>11<br>60.2<br>30.4<br>6.3<br>4.9<br>2.0<br>93.1<br>8<br>56.2<br>33.1<br>8.3<br>6.7<br>1.8<br>91.5<br>9|
|E5-Mistral_GPT-4<br>E5-Mistral_Llama3-8b<br>E5-Mistral_Llama3-70b<br>E5-Mistral_Mixtral-8x7b|62.4<br>57.0<br>56.1<br>51.1<br>52.5<br>47.3<br>60.4<br>53.7<br>52.7<br>52.1<br>51.8<br>47.9|85.1<br>51.1<br>85.1<br>51.1<br>85.1<br>51.1<br>85.1<br>51.1|64.2<br>27.8<br>4.1<br>5.7<br>1.5<br>92.8<br>12<br>59.4<br>34.1<br>9.2<br>5.3<br>0.7<br>94.0<br>13<br>60.3<br>30.8<br>6.2<br>2.6<br>0.7<br>96.7<br>10<br>58.6<br>34.1<br>8.2<br>3.9<br>0.6<br>95.5<br>11|



Table 13: Evaluation results for different RAG systems on RobustQA - Science dataset 

|RAG systems|Overall|Retriever|Generator<br>#Claim<br>CU_↑_<br>NS(I)_↓_<br>NS(II)_↓_<br>Hallu._↓_<br>SK_↓_<br>Faith._↑_|
|---|---|---|---|
||Prec._↑_<br>Rec._↑_<br>F1_↑_|CR_↑_<br>CP_↑_||
|BM25_GPT-4<br>BM25_Llama3-8b<br>BM25_Llama3-70b<br>BM25_Mixtral-8x7b|58.7<br>52.6<br>51.8<br>47.9<br>45.2<br>41.7<br>55.8<br>45.0<br>45.5<br>51.5<br>45.1<br>44.0|71.3<br>62.6<br>71.3<br>62.6<br>71.3<br>62.6<br>71.3<br>62.6|66.6<br>27.6<br>2.4<br>11.3<br>4.3<br>84.5<br>14<br>58.2<br>32.2<br>5.3<br>14.2<br>1.7<br>84.1<br>14<br>57.7<br>33.0<br>4.5<br>6.7<br>1.6<br>91.7<br>10<br>58.5<br>34.8<br>5.4<br>7.1<br>1.8<br>91.1<br>10|
|E5-Mistral_GPT-4<br>E5-Mistral_Llama3-8b<br>E5-Mistral_Llama3-70b<br>E5-Mistral_Mixtral-8x7b|57.9<br>55.0<br>53.0<br>48.8<br>48.5<br>43.5<br>56.7<br>51.3<br>49.6<br>54.5<br>49.2<br>47.4|85.0<br>71.8<br>85.0<br>71.8<br>85.0<br>71.8<br>85.0<br>71.8|61.5<br>31.5<br>2.3<br>8.4<br>1.5<br>90.2<br>15<br>54.8<br>39.1<br>4.6<br>6.6<br>0.6<br>92.8<br>13<br>57.7<br>36.1<br>3.9<br>3.2<br>0.5<br>96.3<br>11<br>55.3<br>37.1<br>3.7<br>4.4<br>0.6<br>95.0<br>11|



Table 14: Evaluation results for different RAG systems on RobustQA - Technology dataset 

|RAG systems|Overall|Retriever|Generator<br>#Claim<br>CU_↑_<br>NS(I)_↓_<br>NS(II)_↓_<br>Hallu._↓_<br>SK_↓_<br>Faith._↑_|
|---|---|---|---|
||Prec._↑_<br>Rec._↑_<br>F1_↑_|CR_↑_<br>CP_↑_||
|BM25_GPT-4<br>BM25_Llama3-8b<br>BM25_Llama3-70b<br>BM25_Mixtral-8x7b|57.5<br>48.6<br>48.9<br>47.2<br>46.3<br>42.1<br>55.9<br>44.4<br>45.5<br>51.7<br>43.4<br>42.9|69.5<br>63.8<br>69.5<br>63.8<br>69.5<br>63.8<br>69.5<br>63.8|63.4<br>28.1<br>3.1<br>11.2<br>4.3<br>84.5<br>14<br>61.2<br>36.3<br>5.8<br>10.1<br>1.7<br>88.2<br>14<br>59.2<br>35.6<br>4.4<br>4.0<br>1.6<br>94.4<br>10<br>58.2<br>36.5<br>5.4<br>5.6<br>1.7<br>92.7<br>11|
|E5-Mistral_GPT-4<br>E5-Mistral_Llama3-8b<br>E5-Mistral_Llama3-70b<br>E5-Mistral_Mixtral-8x7b|59.9<br>55.0<br>53.6<br>47.9<br>51.1<br>44.8<br>56.9<br>53.5<br>50.7<br>52.6<br>51.5<br>47.9|83.7<br>76.4<br>83.7<br>76.4<br>83.7<br>76.4<br>83.7<br>76.4|63.3<br>31.8<br>2.3<br>6.0<br>1.2<br>92.8<br>15<br>59.0<br>40.4<br>5.6<br>5.9<br>0.5<br>93.6<br>15<br>62.1<br>36.7<br>3.8<br>2.6<br>0.3<br>97.1<br>13<br>59.6<br>40.5<br>3.4<br>3.2<br>0.2<br>96.6<br>12|



Table 15: Evaluation results for different RAG systems on KIWI dataset 

|RAG systems|Overall|Retriever|Generator<br>#Claim<br>CU_↑_<br>NS(I)_↓_<br>NS(II)_↓_<br>Hallu._↓_<br>SK_↓_<br>Faith._↑_|
|---|---|---|---|
||Prec._↑_<br>Rec._↑_<br>F1_↑_|CR_↑_<br>CP_↑_||
|BM25_GPT-4<br>BM25_Llama3-8b<br>BM25_Llama3-70b<br>BM25_Mixtral-8x7b|42.8<br>30.0<br>32.4<br>43.0<br>24.7<br>26.6<br>42.7<br>27.4<br>31.0<br>40.2<br>21.6<br>23.8|57.8<br>72.5<br>57.8<br>72.5<br>57.8<br>72.5<br>57.8<br>72.5|49.1<br>45.0<br>6.2<br>6.0<br>0.7<br>93.3<br>18<br>39.6<br>41.8<br>4.8<br>9.1<br>1.7<br>89.2<br>16<br>43.8<br>45.2<br>6.8<br>5.2<br>0.7<br>94.0<br>18<br>35.5<br>51.0<br>5.8<br>3.1<br>0.3<br>96.6<br>13|
|E5-Mistral_GPT-4<br>E5-Mistral_Llama3-8b<br>E5-Mistral_Llama3-70b<br>E5-Mistral_Mixtral-8x7b|45.5<br>34.0<br>36.0<br>47.2<br>27.8<br>29.8<br>45.2<br>30.9<br>34.0<br>36.7<br>23.1<br>23.5|64.6<br>86.7<br>64.6<br>86.7<br>64.6<br>86.7<br>64.6<br>86.7|49.0<br>44.9<br>4.0<br>5.5<br>1.5<br>93.0<br>20<br>39.1<br>43.4<br>4.2<br>4.6<br>0.3<br>95.2<br>15<br>45.3<br>47.5<br>3.7<br>3.6<br>0.3<br>96.1<br>18<br>32.2<br>55.0<br>4.9<br>2.9<br>0.5<br>96.7<br>14|



26 

Table 16: Performance of RefChecker on the RefChecker benchmark using Llama 3 70B Instruct as both the extractor and checker. We compare the results with the best performed purely open-sourced combinations reported in the RefChecker paper. 

||Accuracy|Fact. F1|Non-Fact. F1|Pearson|Spearman|
|---|---|---|---|---|---|
||Zero Context|||||
|Mistral-SFT + RepC|89.38|80.43|92.72|77.14|76.74|
|Llama3 + Llama3|**91.89**|**83.06**|**94.67**|**81.77**|**80.83**|
||Noisy Context|||||
|Mistral-SFT + NLI|70.82|75.12|**64.72**|52.21|45.61|
|Llama3 + Llama3|**71.75**|**76.69**|64.15|**57.67**|**50.31**|
||Accurate Context|||||
|Mistral-SFT + AlignScore|74.12|81.6|56.38|46.34|43.22|
|Llama3 + Llama3|**78.35**|**84.87**|**61.92**|**59.48**|**52.03**|



27 

